from typing import List, Tuple, Dict, Optional, Union
import heapq
import networkx as nx
import numpy as np

try:
    import hasse  # optional if you pass an nx.DiGraph directly
except Exception:
    hasse = None

# ---------------------------------------------------------------------
# Helpers: Hasse access + pretty print
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

# ---------------------------------------------------------------------
# GPAV algorithm for segmented GPAV
# ---------------------------------------------------------------------
def gpav_seg(
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
    Generalized Pool Adjacent Violators over a partial order from "DATA PREORDERING IN GENERALIZED PAV ALGORITHM
FOR MNOTONIC REGRESSION".

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

    # Assemble outputs (LOCAL indexing 0..n-1)
    u = np.zeros(n, dtype=float)
    block_list: List[Dict] = []
    elem_to_block = np.empty(n, dtype=int)

    for head, b in blocks.items(): # b are the elements of the dictionary; head, the index
        b_id = len(block_list)
        block_list.append({
            'elements': list(b['elements']),
            'labels': [idx_to_node[x] for x in b['elements']],
            'weight': float(b['weight']),
            'value': float(b['value']),
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
# GPAV algorithm for operadic GPAV
# ---------------------------------------------------------------------
def gpav_op(
    Y: np.ndarray,
    poset,
    topo_order: Optional[List] = None,   # list of node labels
    weights: Optional[np.ndarray] = None,
    *,
    verbose: bool = False,
    name: str = "GPAV",
    indent: str = "",
    return_block_edges: bool = False,
) -> Tuple[np.ndarray, List[Dict], np.ndarray]:
    """
    Generalized Pool Adjacent Violators over a partial order from
    "DATA PREORDERING IN GENERALIZED PAV ALGORITHM FOR MONOTONIC REGRESSION".

    This variant is designed to be label-aware.

    Conceptually:

      * The input data `Y` is a mapping from **poset node labels** to values.
        You may pass this either as:
          - a 1D NumPy array (or array-like) already aligned with the internal
            Hasse node order, or
          - a dictionary `{label -> value}` keyed by the node labels of `poset`.

      * Optional `weights` behave similarly:
          - 1D array-like aligned with the Hasse node order, or
          - dict `{label -> weight}` keyed by node labels.

    Internally, the function builds a node list `N = list(G.nodes())` from the
    Hasse diagram `G` of the given `poset`, and aligns the numeric arrays so
    that index `i` corresponds to label `N[i]`.

    Parameters
    ----------
    Y :
        1D array-like or dict keyed by poset node labels. The values to be
        isotonic-regressed.
    poset :
        Either a `hasse.PoSet` instance or a `networkx.DiGraph` representing
        the Hasse diagram of the partial order.
    topo_order :
        Optional list of node labels specifying the processing order. If not
        given, a topological sort of the Hasse graph is used.
    weights :
        Optional 1D array-like or dict keyed by node labels giving observation
        weights. If `None`, all weights are taken as 1.
    verbose :
        If True, prints debugging information about the blocks and values.
    name :
        Label used in verbose prints.
    indent :
        String prefix used to indent verbose output.
    return_block_edges :
        If True, also returns a list of block-level edges in the final block DAG.

    Returns
    -------
    u : np.ndarray
        Fitted values aligned to the internal node order `N`
        (i.e., `u[i]` corresponds to node label `N[i]`).
    block_list : List[Dict]
        Each dict has (at least) the keys:
          - 'elements': list of LOCAL indices (0..n-1),
          - 'labels'  : list of corresponding node labels,
          - 'weight'  : total block weight,
          - 'value'   : block value.
    elem_to_block : np.ndarray
        LOCAL element index -> final block id.
    """
    G = _hasse_graph(poset)
    N = list(G.nodes())
    n = len(N)
    node_to_idx = {v: i for i, v in enumerate(N)}
    idx_to_node = dict(enumerate(N))

     # Align Y
    if isinstance(Y, dict):
        # Assume keys are node labels in the poset
        try:
            Y = np.array([Y[v] for v in N], dtype=float)
        except Exception as e:
            raise ValueError(
                "Y dict must be keyed by poset node labels."
            ) from e
    else:
        Y = np.asarray(Y)
        if Y.ndim == 0:
            # np.asarray(scalar) gives shape (), which caused your IndexError
            raise ValueError(
                "Y must be a 1D array-like or a dict keyed by node labels, not a scalar."
            )
        if Y.shape[0] != n:
            # Try interpreting Y as a mapping indexed by node labels
            try:
                Y = np.array([Y[v] for v in N], dtype=float)
            except Exception as e:
                raise ValueError(
                    "Y must align with poset node labels or Hasse node order."
                ) from e
        else:
            Y = Y.astype(float, copy=False)

    # Align weights
    if weights is None:
        w = np.ones(n, dtype=float)
    else:
        if isinstance(weights, dict):
            try:
                w = np.array([weights[v] for v in N], dtype=float)
            except Exception as e:
                raise ValueError(
                    "weights dict must be keyed by poset node labels."
                ) from e
        else:
            weights = np.asarray(weights)
            if weights.ndim == 0:
                raise ValueError(
                    "weights must be a 1D array-like or a dict keyed by node labels, not a scalar."
                )
            if weights.shape[0] != n:
                try:
                    w = np.array([weights[v] for v in N], dtype=float)
                except Exception as e:
                    raise ValueError(
                        "weights must align with poset node labels or Hasse node order."
                    ) from e
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

    block_min_idx_list: List[List[int]] = []   # ADDED: local indices per block
    block_max_idx_list: List[List[int]] = []   # ADDED: local indices per block
    for head, b in blocks.items(): # b are the elements of the dictionary; head, the index

        b_id = len(block_list)

        S = set(b['elements'])
        _min_idx = [i for i in b['elements'] if not (preds[i] & S)]
        _max_idx = [i for i in b['elements'] if not (succs[i] & S)]
        _min_labels = [idx_to_node[i] for i in _min_idx]
        _max_labels = [idx_to_node[i] for i in _max_idx]
        block_min_idx_list.append(_min_idx)   # ADDED
        block_max_idx_list.append(_max_idx)   # ADDED
        
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

    # ADDED: compute local block edges via Min–Max reachability on the local graph
    block_edges: List[Tuple[int, int]] = []
    if block_min_idx_list and block_max_idx_list:
        reach_cache: Dict[int, set] = {}

        def _reach_from(s: int) -> set:
            R = reach_cache.get(s)
            if R is not None:
                return R
            R = set()
            stack = [s]
            while stack:
                u_ = stack.pop()
                for v_ in succs[u_]:
                    if v_ not in R:
                        R.add(v_); stack.append(v_)
            reach_cache[s] = R
            return R

        B_ = len(block_min_idx_list)
        for a in range(B_):
            if not block_min_idx_list[a]:
                continue
            RA = set().union(*(_reach_from(m) for m in block_min_idx_list[a])) if block_min_idx_list[a] else set()
            for b in range(B_):
                if a == b or not block_max_idx_list[b]:
                    continue
                if RA & set(block_max_idx_list[b]):
                    block_edges.append((a, b))

    if verbose:
        print(indent + f"== {name}: finished ==")
        print(indent + "Final blocks:")
        for b_id, b in enumerate(block_list):
            print(indent + f"  B{b_id}: elems={[N[e] for e in b['elements']]}, "
                  f"w={b['weight']:.6g}, val={b['value']:.6g}")
        print(indent + f"u (aligned to nodes): {[(N[i], float(u[i])) for i in range(n)]}")
        print(indent + "----")

    if return_block_edges:
        return u, block_list, elem_to_block, block_edges
    return u, block_list, elem_to_block
