# -*- coding: utf-8 -*-
"""
Segmentation-based GPAV and GPAV that returns blocks,


Dependencies:
  - numpy
  - networkx
  - hasse (pip install hasse)
"""

from typing import List, Tuple, Dict, Optional
import heapq
import numpy as np
import networkx as nx
import hasse



def _hasse_graph(poset):
    if isinstance(poset, nx.DiGraph):
        return poset
    h = getattr(poset, "hasse", None)
    if h is None:
        raise AttributeError("Expected a PoSet with a `.hasse` attribute/method.")
    return h() if callable(h) else h


def _induced_hasse(G: nx.DiGraph, nodes):
    """
    Induce on `nodes` and re-reduce (induction can introduce transitives).
    """
    H = G.subgraph(list(nodes)).copy()
    return nx.transitive_reduction(H)


# -----------------------------------------------------------------------------
# LowerY (trend-following) topological order — heap-based 
# -----------------------------------------------------------------------------

def trend_following_order_lowery_fast(poset: hasse.PoSet, Y: np.ndarray) -> List:
    """
    Algorithm 4 (LowerY) from the paper, optimized for large posets.

    At each step:
      - among current minimal elements (in-degree 0 w.r.t. remaining nodes),
        pick the one with smallest Y.

    Works for arbitrary node labels. Returns a list of node labels.

    Complexity ~ O((V+E) log V). Uses the Hasse diagram (transitive reduction).
    """
    G = _hasse_graph(poset)
    #G = poset.hasse()  # networkx.DiGraph, already transitive-reduced
    nodes = list(G.nodes())
    n = len(nodes)

    # Align Y to 'nodes' order if needed
    Y = np.asarray(Y)
    if Y.shape[0] != n:
        # Try mapping by labels if Y is indexed by labels (0..n-1 is common but not guaranteed)
        try:
            Y = np.array([Y[v] for v in nodes], dtype=float)
        except Exception as e:
            raise ValueError("Y must align with poset node labels or with the `poset.hasse()` node order.") from e
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
        # Should not happen if input is a poset
        raise ValueError("Cycle detected in poset (unexpected).")

    return [nodes[i] for i in order_idx]


# -----------------------------------------------------------------------------
# GPAV (version needed for the segmentation based algorithm; returns  blocks)
# -----------------------------------------------------------------------------

def gpav(
    Y: np.ndarray,
    poset: hasse.PoSet,
    topo_order: Optional[List] = None,   # list of node labels
    weights: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, List[Dict], np.ndarray]:
    """
    Generalized Pool Adjacent Violators over a partial order (Hasse diagram).

    Inputs
    ------
    Y : array-like, shape (n,)
        Observed values aligned to the node labels in `poset.hasse()`.
        If not aligned, we try to pick by label: Y[label].
    poset : hasse.PoSet
        Must be a DAG. We use poset.hasse() as the graph.
    topo_order : list of node labels, optional
        A topological order to process nodes. If None, we compute one.
        (You may use `trend_following_order_lowery_fast` for LowerY.)
    weights : array-like, shape (n,), optional
        Positive weights aligned with nodes. Defaults to 1s.

    Returns
    -------
    u : np.ndarray, shape (n,)
        Fitted values, aligned to `G.nodes()` order.
    block_list : List[Dict]
        Each dict has {'elements': [local_ids], 'weight': float, 'value': float}.
        NOTE: 'elements' are LOCAL indices 0..n-1 of the nodes order used internally.
    elem_to_block : np.ndarray, shape (n,)
        Map LOCAL element index -> final block id.
    """
    G = _hasse_graph(poset)
    nodes = list(G.nodes())
    n = len(nodes)

    # Align Y to nodes order
    Y = np.asarray(Y)
    if Y.shape[0] != n:
        try:
            Y = np.array([Y[v] for v in nodes], dtype=float)
        except Exception as e:
            raise ValueError("Y must align with poset node labels or `poset.hasse()` node order.") from e
    else:
        Y = Y.astype(float, copy=False)

    # Align weights
    if weights is None:
        w = np.ones(n, dtype=float)
    else:
        weights = np.asarray(weights)
        if weights.shape[0] != n:
            try:
                w = np.array([weights[v] for v in nodes], dtype=float)
            except Exception as e:
                raise ValueError("weights must align with poset node labels or `poset.hasse()` node order.") from e
        else:
            w = weights.astype(float, copy=False)

    # Topological order (labels) -> local indices
    if topo_order is None:
        topo_labels = list(nx.topological_sort(G))
    else:
        topo_labels = list(topo_order)

    node_to_idx = {v: i for i, v in enumerate(nodes)}
    topo = [node_to_idx[v] for v in topo_labels]

    # Precompute local predecessors for speed
    preds = {i: [node_to_idx[u] for u in G.predecessors(nodes[i])] for i in range(n)}

    # Initialize blocks keyed by current head (local index)
    blocks: Dict[int, Dict] = {}
    head_of = {i: i for i in range(n)}  # element -> current head

    for j in topo:
        # Create singleton block for j
        blocks[j] = {
            'elements': [j],
            'weight': float(w[j]),
            'value': float(Y[j]),
            'pred_heads': set(),
        }

        # Heads of predecessor blocks that still exist
        pred_heads = set()
        for p in preds[j]:
            h = head_of[p]
            if h in blocks:
                pred_heads.add(h)

        # Merge as long as a predecessor block has larger value (violations)
        while pred_heads:
            violators = [h for h in pred_heads if blocks[h]['value'] > blocks[j]['value']]
            if not violators:
                break
            h_star = max(violators, key=lambda h: blocks[h]['value'])

            bj = blocks[j]
            bh = blocks[h_star]

            new_w = bj['weight'] + bh['weight']
            new_val = (bj['weight'] * bj['value'] + bh['weight'] * bh['value']) / new_w
            new_elems = bh['elements'] + bj['elements']

            # Update current block j
            bj['elements'] = new_elems
            bj['weight'] = new_w
            bj['value'] = new_val

            # Remap merged elements to head j
            for e in bh['elements']:
                head_of[e] = j

            # Update predecessor heads of j: (Bj^- ∪ Bh^-) \ {h_star}
            pred_heads |= bh.get('pred_heads', set())
            if h_star in pred_heads:
                pred_heads.remove(h_star)

            # Remove merged block
            del blocks[h_star]

        # Keep only existing heads
        blocks[j]['pred_heads'] = set(h for h in pred_heads if h in blocks)

    # Assemble outputs (LOCAL indexing 0..n-1)
    u = np.zeros(n, dtype=float)
    block_list: List[Dict] = []
    elem_to_block = np.empty(n, dtype=int)

    for head, b in blocks.items():
        b_id = len(block_list)
        block_list.append({
            'elements': list(b['elements']),
            'weight': float(b['weight']),
            'value': float(b['value']),
        })
        for e in b['elements']:
            u[e] = b['value']
            elem_to_block[e] = b_id

    return u, block_list, elem_to_block


# -----------------------------------------------------------------------------
# Segmentation-based algorithm
# -----------------------------------------------------------------------------

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
        # Paper ensures GB is a DAG; we don't check cycles to keep it light.
        self._G_B = G_B

    def hasse(self) -> nx.DiGraph:
        # We don't need to transitive-reduce GB; GPAV works on any DAG.
        # Returning as-is avoids extra O(VE) work.
        return self._G_B


def segmentation_based_gpav(
    poset: hasse.PoSet,
    Y: np.ndarray,  T:  np.array,
    weights: Optional[np.ndarray] = None,
    *,
    segment_size: int = 512,
    use_trend_following_first: bool = True,
    use_trend_following_blocks: bool = False
) -> np.ndarray:
    """
    Implementation of the Segmentation-Based algorithm (Algorithm 3).

    Stage 1 (Segmentation)
      1) Choose a topological order T (LowerY if requested).
      2) Split T into contiguous segments.
      3) For each segment, run GPAV on the induced subposet using that segment’s order,
         and collect the TRUE blocks (no grouping-by-value).

    Stage 2 (Assembling)
      1) Let R be the union of local blocks; create GB with nodes = blocks.
      2) Add edges via Min–Max: (A,B) if ∃ path from min(A) to max(B) in the original poset.
      3) Choose a topological order TB for GB (LowerY on av(B) if requested).
      4) Run GPAV on GB with weights W(B) and Y(B)=av(B).
      5) Propagate final fitted values to all original elements.

    Returns
    -------
    u_final : np.ndarray, shape (n,)
        Fitted values aligned to `poset.hasse().nodes()` order.
    """
    G = _hasse_graph(poset) 
    nodes = list(G.nodes())
    n = len(nodes)
    node_to_idx = {v: i for i, v in enumerate(nodes)}

    # Align Y, weights to nodes
    Y = np.asarray(Y)
    if Y.shape[0] != n:
        try:
            Y = np.array([Y[v] for v in nodes], dtype=float)
        except Exception as e:
            raise ValueError("Y must align with poset node labels or `poset.hasse()` node order.") from e
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
                raise ValueError("weights must align with poset node labels or `poset.hasse()` node order.") from e
        else:
            weights = weights.astype(float, copy=False)



    # 2) Segment T
    segments: List[List] = [T[i:i + segment_size] for i in range(0, n, segment_size)]

    # 3) Local GPAV per segment -> collect] blocks
    all_blocks: List[List] = []     # list of block member labels
    block_values: List[float] = []  # av(B)
    block_weights: List[float] = [] # sum of member weights

    for seg in segments:
        #subP = poset.subposet(seg)   # hasse already reduced & induced
        subG = _induced_hasse(G, seg)
        # Align segment arrays to seg labels
        Y_seg = np.array([Y[node_to_idx[v]] for v in seg], dtype=float)
        W_seg = np.array([weights[node_to_idx[v]] for v in seg], dtype=float) if weights is not None else None

        #Y_seg = np.array([Y[node_to_idx[v]] for v in seg], dtype=float)
        #w_seg = np.array([weights[node_to_idx[v]] for v in seg], dtype=float)

        # Run GPAV on the segment using the segment order (labels)
        #u_seg, local_blocks, _ = gpav(Y_seg, subP, topo_order=seg, weights=w_seg)
        u_seg, local_blocks, _ = gpav(Y_seg, subG, topo_order=seg, weights=W_seg)  
        # local_blocks[*]['elements'] are LOCAL indices 0..len(seg)-1
        # map back to labels
        for b in local_blocks:
            members = [seg[i_local] for i_local in b['elements']]
            all_blocks.append(members)
            block_values.append(float(b['value']))
            block_weights.append(float(np.sum([weights[node_to_idx[v]] for v in members])))

    B = len(all_blocks)

    # Stage 2: Build GB via Min–Max rule
    G_B = nx.DiGraph()
    G_B.add_nodes_from(range(B))

    # Precompute min/max members of each block in the ORIGINAL poset G
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
                        found = True
                        break
                if found:
                    break

    # Choose block-level order TB
    Y_B = np.asarray(block_values, dtype=float)
    if use_trend_following_blocks:
        TB_labels = trend_following_order_lowery_fast(_BlockGraphPoset(G_B), Y_B)
    else:
        TB_labels = list(nx.topological_sort(G_B))

    # Run GPAV on blocks
    u_blocks, block_blocks, elem_to_block_B = gpav(
        Y=Y_B,
        poset=_BlockGraphPoset(G_B),
        topo_order=TB_labels,
        weights=np.asarray(block_weights, dtype=float),
    )

    # Final value per original block id
    final_block_value = np.zeros(B, dtype=float)
    for b_id in range(B):
        final_block_value[b_id] = block_blocks[elem_to_block_B[b_id]]['value']

    # Propagate to original items (aligned to nodes order)
    u_final = np.zeros(n, dtype=float)
    for b_id, members in enumerate(all_blocks):
        val = final_block_value[b_id]
        for v in members:
            u_final[node_to_idx[v]] = val

    return u_final


# -----------------------------------------------------------------------------
# Minimal smoke test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    
    pass

