# -*- coding: utf-8 -*-
"""
version of GPAV and Segmentation-Based GPAV.

Key additions:
- `verbose` flag on both `gpav(...)` and `segmentation_based_gpav(...)`.
- Pretty-printers for the Hasse diagram and evolving block structure.
- Step-by-step logs for:
  * creating each singleton block
  * each merge (violation fix) inside GPAV
  * post-merge block states
  * per-segment local GPAV results
  * construction of the block DAG G_B via Min–Max rule (with witnesses)
  * block-level GPAV (including merges)
  * final propagation back to original nodes


Usage (example):
----------------
from segmented_gpav import segmentation_based_gpav, gpav
u = segmentation_based_gpav(poset, Y, T, verbose=True)

You can also directly inspect GPAV on a tiny Hasse (DAG) with:
u, blocks, map_ = gpav(Y, poset, topo_order=T, verbose=True)
"""

from typing import List, Tuple, Dict, Optional
import heapq
import numpy as np
import networkx as nx

try:
    import hasse  # optional if you pass an nx.DiGraph directly
except Exception:
    hasse = None


# ---------------------------------------------------------------------
# Helpers: Hasse access + induced/reduced subgraph
# ---------------------------------------------------------------------

def _hasse_graph(poset):
    if isinstance(poset, nx.DiGraph):
        return poset
    h = getattr(poset, "hasse", None)
    if h is None:
        raise AttributeError("Expected a PoSet with a `.hasse` attribute/method.")
    G = h() if callable(h) else h
    if not isinstance(G, nx.DiGraph):
        raise TypeError("`.hasse` must yield a networkx.DiGraph.")
    return G



def _induced_hasse(G: nx.DiGraph, nodes):
    """
    Induce on `nodes` and re-reduce (induction can introduce transitives).
    """
    H = G.subgraph(list(nodes)).copy()
    return nx.transitive_reduction(H)


# ---------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------

def _fmt_tuple_list(ls):
    return "[" + ", ".join(f"{a}->{b}" for (a, b) in ls) + "]"

def _print_hasse(G: nx.DiGraph, title: str = "Hasse diagram", indent: str = ""):
    print(f"{indent}{title}:")
    print(f"{indent}  Nodes: {list(G.nodes())}")
    print(f"{indent}  Covers (edges): {_fmt_tuple_list(list(G.edges()))}")

def _print_blocks_state(blocks: Dict[int, Dict], nodes: List, indent: str = ""):
    """
    blocks: dict keyed by current head (LOCAL index), each with:
      'elements': [local ids], 'weight': float, 'value': float, 'pred_heads': set()
    """
    lines = []
    for head, b in blocks.items():
        elems_lbl = [nodes[e] for e in b['elements']]
        preds_lbl = [nodes[p] for p in b.get('pred_heads', set())]
        lines.append(
            f"  head={nodes[head]!r} elems={elems_lbl} "
            f"w={b['weight']:.3g} val={b['value']:.6g} pred_heads={preds_lbl}"
        )
    if not lines:
        print(indent + "(no blocks)")
    else:
        print(indent + "Current blocks:")
        for line in lines:
            print(indent + line)

def _print_block_list(all_blocks: List[List], av: List[float], wt: List[float], indent: str = ""):
    for i, (members, v, w) in enumerate(zip(all_blocks, av, wt)):
        print(f"{indent}B{i}: members={members}, av={v:.6g}, weight={w:.6g}")


# ---------------------------------------------------------------------
# LowerY (trend-following) topological order — heap-based
# ---------------------------------------------------------------------

def trend_following_order_lowery_fast(poset, Y: np.ndarray) -> List:
    """
    Algorithm 4 (LowerY) from "A Segmentation-Based Algorithm for Large-Scale
Partially Ordered Monotonic Regression".

    At each step:
      - among current minimal elements (in-degree 0 w.r.t. remaining nodes),
        pick the one with smallest Y.

    Returns a list of node labels (not local indices).
    """
    G = _hasse_graph(poset)
    nodes = list(G.nodes())
    n = len(nodes)

    Y = np.asarray(Y)
    if Y.shape[0] != n:
        try:
            Y = np.array([Y[v] for v in nodes], dtype=float)
        except Exception as e:
            raise ValueError("Y must align with poset node labels or with the Hasse node order.") from e
    else:
        Y = Y.astype(float, copy=False)

    node_to_idx = {v: i for i, v in enumerate(nodes)}
    indeg = np.fromiter((G.in_degree(v) for v in nodes), dtype=np.int32, count=n)
    succ = [[node_to_idx[w] for w in G.successors(v)] for v in nodes]

    heap = [(Y[i], i) for i in range(n) if indeg[i] == 0]
    heapq.heapify(heap)

    order_idx = []
    while heap:
        _, i = heapq.heappop(heap)
        order_idx.append(i)
        for j in succ[i]:
            indeg[j] -= 1
            if indeg[j] == 0:
                heapq.heappush(heap, (Y[j], j))

    if len(order_idx) != n:
        raise ValueError("Cycle detected in poset (unexpected).")

    return [nodes[i] for i in order_idx]


# ---------------------------------------------------------------------
# GPAV (returns blocks) — with verbose tracing
# ---------------------------------------------------------------------

# GPAV algorithm
def gpav(
    Y: np.ndarray,
    poset,
    topo_order: Optional[List] = None,   # list of node labels
    weights: Optional[np.ndarray] = None,
    *,
    verbose: bool = False,
    name: str = "GPAV",
    indent: str = ""
) -> Tuple[np.ndarray, List[Dict], np.ndarray]:
    """
    Generalized Pool Adjacent Violators over a partial order from "DATA PREORDERING IN GENERALIZED PAV ALGORITHJM
FOR JVIONOTONIC REGRESSION".

    Returns:
      u : fitted values aligned to internal node order
      block_list : each dict has {'elements': [local ids], 'weight': float, 'value': float}
      elem_to_block : LOCAL element index -> final block id
    """
    G = _hasse_graph(poset)
    N = list(G.nodes())
    n = len(N)
    node_to_idx = {v: i for i, v in enumerate(N)}
    idx_to_node = dict(enumerate(N))

    # Align Y
    Y = np.asarray(Y)
    if Y.shape[0] != n:
        try:
            Y = np.array([Y[v] for v in N], dtype=float)
        except Exception as e:
            raise ValueError("Y must align with poset node labels or Hasse node order.") from e
    else:
        Y = Y.astype(float, copy=False)

    # Align weights
    if weights is None:
        w = np.ones(n, dtype=float)
    else:
        weights = np.asarray(weights)
        if weights.shape[0] != n:
            try:
                w = np.array([weights[v] for v in N], dtype=float)
            except Exception as e:
                raise ValueError("weights must align with poset node labels or Hasse node order.") from e
        else:
            w = weights.astype(float, copy=False)

    # Topological order (labels) -> local indices
    if topo_order is None:
        topo_labels = list(nx.topological_sort(G))
    else:
        topo_labels = list(topo_order)
    topo = [node_to_idx[v] for v in topo_labels]

    # Dictionary of (immediate) children
    children = {i: [node_to_idx[u] for u in G.predecessors(N[i])] for i in range(n)}

    if verbose:
        print(indent + f"== {name}: starting ==")
        _print_hasse(G, title=f"{name} input Hasse", indent=indent)
        print(indent + f"Node order used: {topo_labels}")
        print(indent + f"Y aligned to nodes: {[(N[i], float(Y[i])) for i in range(n)]}")
        if weights is not None:
            print(indent + f"Weights aligned: {[(N[i], float(w[i])) for i in range(n)]}")
        print(indent + "----")

    # Initialize
    blocks: Dict[int, Dict] = {}
    current_block_of = {i: i for i in range(n)}  # element -> current head

    # Sweep
    for k in topo: # change j to k
        # Create singleton block for j
        blocks[k] = {
            'elements': [k],
            'weight': float(w[k]),
            'value': float(Y[k]),
            'children_k': set(),
        }

        # Heads of predecessor blocks that still exist
        B_k_minus = set()
        for p in children[k]:
            h = current_block_of[p]
            if h in blocks:
                B_k_minus.add(h) # Children of k that are blocks = B_k_minus

        if verbose:
            print(indent + f"Visit node {N[k]!r}: start block {{ {N[k]!r} }} "
                  f"(val={Y[k]:.6g}, w={w[k]:.6g}); B_k_minus={[N[h] for h in B_k_minus]}")

        # Merge as long as a predecessor block has larger value (violations)
        while B_k_minus:
            B_k_minus = {h for h in B_k_minus if h in blocks}
            violators = [h for h in B_k_minus if blocks[h]['value'] >= blocks[k]['value']]
            if not violators:
                break
            # pick max-value violator
            j = max(violators, key=lambda h: blocks[h]['value'])

            B_k = blocks[k]
            B_j = blocks[j]
            U_k = B_k['value']
            U_j = B_j['value']

            new_w = B_k['weight'] + B_j['weight']
            new_val = (B_k['weight'] * B_k['value'] + B_j['weight'] * B_j['value']) / new_w
            new_elems = B_j['elements'] + B_k['elements'] # B_k=B_k U B_j

            if verbose:
                print(indent + f"  Merge due to violation: head {N[j]!r} (val={U_j:.6g})"
                      f" into head {N[k]!r} (val={U_k:.6g})"
                      f" -> new_val={new_val:.6g}, new_w={new_w:.6g}, elems={[N[e] for e in new_elems]}")

            # Update current block
            B_k['elements'] = new_elems
            B_k['weight'] = new_w
            B_k['value'] = new_val

            # Remap merged elements to head j
            for e in B_j['elements']:
                current_block_of[e] = k # For the most problematic child, we change its parent to k

            # Update predecessor heads of j: (B_j_minus ∪ B_k_minus) \ {j}
            # Take children of j and make them children of k, IF they are blocks
            B_k_minus |= {h for h in B_j.get('children_k', set()) if h in blocks}

            if j in B_k_minus:
                B_k_minus.remove(j) # taking out {j}

            # Remove merged block B_j-
            del blocks[j]  # but other blocks can make reference to this deleted one

            if verbose:
                _print_blocks_state(blocks, N, indent=indent + "    ")

        # Keep only existing heads (if part is for safety)
        blocks[k]['children_k'] = set(h for h in B_k_minus if h in blocks)
        

    preds = [{node_to_idx[p] for p in G.predecessors(N[i])} for i in range(n)]
    succs = [{node_to_idx[s] for s in G.successors(N[i])} for i in range(n)]
    # Assemble outputs (LOCAL indexing 0..n-1)
    u = np.zeros(n, dtype=float)
    block_list: List[Dict] = []
    elem_to_block = np.empty(n, dtype=int)

    for head, b in blocks.items(): # b are the elements of the dictionary; head, the index
        b_id = len(block_list)

        S = set(b['elements'])
        _min_idx = [i for i in b['elements'] if not (preds[i] & S)]
        _max_idx = [i for i in b['elements'] if not (succs[i] & S)]
        _min_labels = [idx_to_node[i] for i in _min_idx]
        _max_labels = [idx_to_node[i] for i in _max_idx]

        block_list.append({
            'elements': list(b['elements']),
            'labels': [idx_to_node[x] for x in b['elements']],
            'weight': float(b['weight']),
            'value': float(b['value']),
            'min_labels': _min_labels,
            'max_labels': _max_labels,
        }) # We copy the blocks without the children
        for e in b['elements']:
            u[e] = b['value']
            elem_to_block[e] = b_id

    transl = {N[i]: float(u[i]) for i in range(n)}

    if verbose:
        print(indent + f"== {name}: finished ==")
        print(indent + "Final blocks:")
        for b_id, b in enumerate(block_list):
            print(indent + f"  B{b_id}: elems={[N[e] for e in b['elements']]}, "
                  f"w={b['weight']:.6g}, val={b['value']:.6g}")
        print(indent + f"u (aligned to nodes): {[(N[i], float(u[i])) for i in range(n)]}")
        print(indent + "----")
    return u, block_list, elem_to_block


# ---------------------------------------------------------------------
# Segmentation-based algorithm
# ---------------------------------------------------------------------

def _minimal_in_subset(G: nx.DiGraph, subset: List) -> List:
    S = set(subset)
    return [v for v in subset if all(u not in S for u in G.predecessors(v))]

def _maximal_in_subset(G: nx.DiGraph, subset: List) -> List:
    S = set(subset)
    return [v for v in subset if all(u not in S for u in G.successors(v))]


class _BlockGraphPoset:
    """
    Tiny wrapper so we can reuse `gpav` on the block-level graph.
    It only needs to expose `.hasse()` returning a networkx.DiGraph.
    """
    def __init__(self, G_B: nx.DiGraph):
        self._G_B = G_B
    def hasse(self) -> nx.DiGraph:
        return self._G_B


def segmentation_based_gpav(
    poset,
    Y: np.ndarray,
    T: List,
    weights: Optional[np.ndarray] = None,
    *,
    segment_size: int = 512,
    use_trend_following_first: bool = True,   # kept for API parity; T is already passed
    use_trend_following_blocks: bool = False,
    verbose: bool = False
) -> np.ndarray:
    """
    Implementation of the Segmentation-Based algorithm (Algorithm 3 A Segmentation-Based Algorithm for Large-Scale
Partially Ordered Monotonic Regression).

    Returns:
      u_final : fitted values aligned to `poset.hasse().nodes()` order.
    """
    G = _hasse_graph(poset)
    nodes = list(G.nodes())
    n = len(nodes)
    node_to_idx = {v: i for i, v in enumerate(nodes)}

    # Align Y, weights
    Y = np.asarray(Y)
    if Y.shape[0] != n:
        try:
            Y = np.array([Y[v] for v in nodes], dtype=float)
        except Exception as e:
            raise ValueError("Y must align with poset node labels or Hasse node order.") from e
    else:
        Y = Y.astype(float, copy=False)

    if weights is None:
        weights = np.ones(n, dtype=float)
    else:
        weights = np.asarray(weights)
        if weights.shape[0] != n:
            try:
                weights = np.array([weights[v] for v in nodes], dtype=float)
            except Exception as e:
                raise ValueError("weights must align with poset node labels or Hasse node order.") from e
        else:
            weights = weights.astype(float, copy=False)

    if verbose:
        print("== Segmentation-Based GPAV: starting ==")
        _print_hasse(G, title="Original Hasse", indent="")
        print(f"Topological order T (given): {T}")
        print(f"Y aligned: {[(nodes[i], float(Y[i])) for i in range(n)]}")
        print("----")

    # 2) Segment T
    segments: List[List] = [T[i:i + segment_size] for i in range(0, n, segment_size)]
    if verbose:
        print(f"Stage 1 — Segmentation into {len(segments)} segment(s):")
        for k, seg in enumerate(segments):
            print(f"  seg[{k}] = {seg}")
        print("----")

    # 3) Local GPAV per segment -> collect blocks
    all_blocks: List[List] = []     # list of block member labels
    block_values: List[float] = []  # av(B)
    block_weights: List[float] = [] # sum of member weights

    for k, seg in enumerate(segments):
        subG = _induced_hasse(G, seg)
        if verbose:
            _print_hasse(subG, title=f"Segment {k} induced Hasse", indent="  ")
        labels_temp = list(subG)
        Y_seg = np.array([Y[node_to_idx[v]] for v in labels_temp], dtype=float) 
        W_seg = np.array([weights[node_to_idx[v]] for v in labels_temp], dtype=float) if weights is not None else None

        # Run GPAV on the segment using the segment order (labels)
        u_seg, local_blocks, _ = gpav(Y_seg, subG, topo_order=seg, weights=W_seg,
                                      verbose=verbose, name=f"GPAV(seg={k})", indent="  ")

        # Map local indices back to labels
        if verbose:
            print("  Segment blocks (local -> labels):")
        for b in local_blocks:
            members = b['labels']
            all_blocks.append(members)
            block_values.append(float(b['value']))
            block_weights.append(float(np.sum([weights[node_to_idx[v]] for v in members])))
            if verbose:
                print(f"    members={members}, av={b['value']:.6g}, w={block_weights[-1]:.6g}")
        if verbose:
            print("----")

    B = len(all_blocks)
    if verbose:
        print(f"Collected {B} block(s) from all segments.")
        _print_block_list(all_blocks, block_values, block_weights, indent="  ")
        print("----")

    # Stage 2: Build G_B via Min–Max rule
    if verbose:
        print("Stage 2 — Build block DAG G_B via Min–Max rule:")
    G_B = nx.DiGraph()
    G_B.add_nodes_from(range(B))

    block_mins = [_minimal_in_subset(G, block) for block in all_blocks]
    block_maxs = [_maximal_in_subset(G, block) for block in all_blocks]

    for a in range(B):
        mins_a = block_mins[a]
        if not mins_a:
            continue
        for b in range(B):
            if a == b:
                continue
            maxs_b = block_maxs[b]
            if not maxs_b:
                continue
            # edge a->b if ∃ path from any min(a) to any max(b)
            found = False
            for i_min in mins_a:
                for j_max in maxs_b:
                    if nx.has_path(G, i_min, j_max):
                        G_B.add_edge(a, b)
                        if verbose:
                            print(f"  edge {a}->{b} (witness: min(a)={i_min} -> max(b)={j_max})")
                        found = True
                        break
                if found:
                    break

    if verbose:
        _print_hasse(G_B, title="Block DAG G_B", indent="  ")

    # Choose block-level order TB
    Y_B = np.asarray(block_values, dtype=float)
    if use_trend_following_blocks:
        TB_labels = trend_following_order_lowery_fast(_BlockGraphPoset(G_B), Y_B)
    else:
        TB_labels = list(nx.topological_sort(G_B))
    if verbose:
        print(f"Block-level order TB: {TB_labels}")

    # Run GPAV on blocks
    if verbose:
        print("Run GPAV on block DAG:")
    u_blocks, block_blocks, elem_to_block_B = gpav(
        Y=Y_B,
        poset=_BlockGraphPoset(G_B),
        topo_order=TB_labels,
        weights=np.asarray(block_weights, dtype=float),
        verbose=verbose,
        name="GPAV(blocks)",
        indent="  "
    )

    # Final value per original block id
    final_block_value = np.zeros(B, dtype=float)
    final_weight_value = np.zeros(B, dtype=float)
    for b_id in range(B):
        final_block_value[b_id] = block_blocks[elem_to_block_B[b_id]]['value']
        final_weight_value[b_id] = block_blocks[elem_to_block_B[b_id]]['weight']
    # Propagate to original items (aligned to nodes order)
    nodes_final: List[Dict] = []
    u_final = np.zeros(n, dtype=float)
    for b_id, members in enumerate(all_blocks):
        val = final_block_value[b_id]
        weight_ = final_weight_value[b_id]
        for v in members:
            u_final[node_to_idx[v]] = val
            
            nodes_final.append({
                "label": v,
                "value": float(val),
                "weight": float(weight_)
            })
    if verbose:
        print("Final per-block values after block-level GPAV:")
        for b_id, members in enumerate(all_blocks):
            print(f"  B{b_id} (members={members}) -> u={final_block_value[b_id]:.6g}")
        print("Propagate to original nodes (aligned to Hasse nodes):")
        print("  " + str([(nodes[i], float(u_final[i])) for i in range(n)]))
        print("== Segmentation-Based GPAV: finished ==")

    return u_final, nodes_final


# ---------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Tiny example DAG (can run without 'hasse' installed)
    # a < c, b < c, c < d
    G = nx.DiGraph()
    G.add_edges_from([('a', 'c'), ('b', 'c'), ('c', 'd')])
    Y = np.array([3.0, 1.0, 2.5, 4.0])  # aligned to node order ['a','b','c','d'] if that order is used
    # Make sure Y matches node order; here we'll remap explicitly:
    nodes = list(G.nodes())
    Y_map = {'a': 3.0, 'b': 1.0, 'c': 2.5, 'd': 4.0}
    Y_aligned = np.array([Y_map[v] for v in nodes], dtype=float)

    # A topological order T 
    T = trend_following_order_lowery_fast(G, Y_aligned)# list(nx.topological_sort(G))

    print("\n=== DEMO: gpav on full graph ===")
    u, block_list, _ = gpav(Y_aligned, G, topo_order=T, verbose=True)

    print("\n=== DEMO: segmentation_based_gpav (segment_size=2) ===")
    u_seg, _ = segmentation_based_gpav(G, Y_aligned, T, segment_size=2, verbose=True)
    print("u_seg:", u_seg)



# ====== BEGIN: Dict-input wrappers (clean, dict-friendly, return_dict support) ======
from collections.abc import Mapping as _Mapping  # local alias
import numpy as _np

def _nodes_in_hasse_for_labeldict(_poset):
    if isinstance(_poset, nx.DiGraph):
        return list(_poset.nodes())
    G = _hasse_graph(_poset)
    if not isinstance(G, nx.DiGraph):
        raise TypeError("poset.hasse() must return a networkx.DiGraph")
    return list(G.nodes())


def _align_map_by_label(_map_like, _nodes, _name, _default=None):
    if not isinstance(_map_like, _Mapping):
        raise TypeError(f"{_name} must be a dict mapping node labels to values.")
    if _default is None:
        return _np.array([float(_map_like[v]) for v in _nodes], dtype=float)
    else:
        return _np.array([float(_map_like.get(v, _default)) for v in _nodes], dtype=float)

def _align_any_by_label_or_nodeorder(_obj, _nodes, _name, _default=None):
    """Accept mapping {label: value} or array already aligned to Hasse node order."""
    if isinstance(_obj, _Mapping):
        return _align_map_by_label(_obj, _nodes, _name, _default)
    arr = _np.asarray(_obj)
    if arr.shape[0] != len(_nodes):
        raise TypeError(f"{_name} must be a dict of labels or an array aligned to Hasse node order (length {len(_nodes)}).")
    return arr.astype(float, copy=False)

def _map_array_to_labeldict(_nodes, _arr):
    return {int(v): float(_arr[i]) for i, v in enumerate(_nodes)}

# ---- trend_following_order_lowery_fast ----
try:
    _old_trend_following_order_lowery_fast = trend_following_order_lowery_fast  # type: ignore[name-defined]
except NameError:
    _old_trend_following_order_lowery_fast = None

if _old_trend_following_order_lowery_fast is not None:
    def trend_following_order_lowery_fast(poset, Y, *args, **kwargs):
        """
        Public wrapper: accepts dict {label:value} or array aligned to Hasse node order.
        Returns a LowerY topological order; delegates to the original implementation.
        """
        nodes = _nodes_in_hasse_for_labeldict(poset)
        Y_arr = _align_any_by_label_or_nodeorder(Y, nodes, "Y")
        return _old_trend_following_order_lowery_fast(poset=poset, Y=Y_arr, *args, **kwargs)

# ---- gpav ----
try:
    _old_gpav = gpav  # type: ignore[name-defined]
except NameError:
    _old_gpav = None

if _old_gpav is not None:
    def gpav(Y, poset, topo_order, weights=None, return_dict=False, *args, **kwargs):
        """
        Public wrapper: accepts dict {label:value} or array aligned to Hasse node order.
        Keeps output identical to the original unless return_dict=True, in which case
        the leading 'u' array is converted to {label: value}.
        """
        nodes = _nodes_in_hasse_for_labeldict(poset)
        Y_arr = _align_any_by_label_or_nodeorder(Y, nodes, "Y")
        w_arr = None if weights is None else _align_any_by_label_or_nodeorder(weights, nodes, "weights", _default=1.0)

        res = _old_gpav(Y_arr, poset, topo_order, weights=w_arr, *args, **kwargs)

        if return_dict:
            if isinstance(res, tuple):
                u = res[0]
                if hasattr(u, 'shape') and len(u) == len(nodes):
                    u_dict = _map_array_to_labeldict(nodes, u)
                    res = (u_dict,) + res[1:]
            else:
                u = res
                if hasattr(u, 'shape') and len(u) == len(nodes):
                    res = _map_array_to_labeldict(nodes, u)
        return res

# ---- segmentation_based_gpav ----
try:
    _old_segmentation_based_gpav = segmentation_based_gpav  # type: ignore[name-defined]
except NameError:
    _old_segmentation_based_gpav = None

if _old_segmentation_based_gpav is not None:
    def segmentation_based_gpav(poset, Y, weights=None, return_dict=False, *args, **kwargs):
        """
        Public wrapper: accepts dict {label:value} or array aligned to Hasse node order.
        Keeps output identical to the original unless return_dict=True, in which case
        the leading 'u' array is converted to {label: value}.
        """
        nodes = _nodes_in_hasse_for_labeldict(poset)
        Y_arr = _align_any_by_label_or_nodeorder(Y, nodes, "Y")
        w_arr = None if weights is None else _align_any_by_label_or_nodeorder(weights, nodes, "weights", _default=1.0)

        res = _old_segmentation_based_gpav(poset, Y_arr, weights=w_arr, *args, **kwargs)

        if return_dict:
            if isinstance(res, tuple):
                u = res[0]
                if hasattr(u, 'shape') and len(u) == len(nodes):
                    u_dict = _map_array_to_labeldict(nodes, u)
                    res = (u_dict,) + res[1:]
            else:
                u = res
                if hasattr(u, 'shape') and len(u) == len(nodes):
                    res = _map_array_to_labeldict(nodes, u)
        return res

# ====== END: Dict-input wrappers ======

