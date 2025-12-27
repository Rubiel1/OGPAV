from __future__ import annotations
from typing import Dict, Hashable, Iterable, List, Tuple
import networkx as nx
from typing import Union
import numpy as np

def trend_following_order(
    G: nx.DiGraph,
    Y: Union[Dict[Hashable, float], np.ndarray],
    *,
    stable_tiebreak: bool = True,
) -> List[Hashable]:
    """
    Faithful implementation of the SB paper's trend-following topological order:
    Algorithm 5: LowerY(P), where P0 is the sequence of observations sorted by Y.

    Inputs
    ------
    G : nx.DiGraph
        DAG encoding the partial order (edge u->v means u ≺ v).
        Can be Hasse-reduced or not; must be acyclic.
    Y : dict[node -> float]
        Observed response values for nodes in G.
    stable_tiebreak : bool
        If True, ties are broken deterministically using the (Y, rank) order induced
        by sorting nodes by (Y, node_as_str). If False, ties follow Python's sort
        on node objects (must be comparable).

    Output
    ------
    A list T (topological order) produced by the published LowerY procedure.
    """

    # --- 0) Basic checks
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("G must be a DAG for trend following order.")

    nodes = list(G.nodes())
    m = len(nodes)
    if m == 0:
        return []

    # accept aligned array
    if not isinstance(Y, dict):
        Y_arr = np.asarray(Y, dtype=float)
        if Y_arr.shape[0] != m:
            raise ValueError(f"Y array must have length {m}, got {Y_arr.shape[0]}")
        Y = {nodes[i]: float(Y_arr[i]) for i in range(m)}  # convert to dict once

    # Ensure all nodes have a Y-value
    missing = [v for v in nodes if v not in Y]
    if missing:
        raise KeyError(f"Missing Y-values for {len(missing)} nodes, e.g. {missing[:5]}")

    # --- 1) Map nodes to indices 0..m-1 (for bitsets)
    # Keep deterministic mapping: preserve G.nodes() iteration order.
    idx_of = {v: i for i, v in enumerate(nodes)}
    node_of = nodes  # inverse map by index

    # --- 2) Prepare a stable Y-sorted master sequence P0
    # Paper: "Sort D by Yi" => P0 is Y-sorted; LowerY always picks first of current P.
    # We implement that as: pick remaining node with minimal (Y, tie_key).
    if stable_tiebreak:
        # tie_key uses string form to be stable even for unorderable node types
        y_order = sorted(nodes, key=lambda v: (float(Y[v]), str(v)))
    else:
        y_order = sorted(nodes, key=lambda v: float(Y[v]))

    y_order_idx = [idx_of[v] for v in y_order]

    # --- 3) Compute transitive predecessor masks pred_mask[v]
    # pred_mask[i] is a bitmask of ALL predecessors of node i (transitive), excluding i itself.
    parents: List[List[int]] = [[] for _ in range(m)]
    for u, v in G.edges():
        parents[idx_of[v]].append(idx_of[u])

    # Use any topological order (graph-based) to DP predecessor sets:
    topo_idx = [idx_of[v] for v in nx.topological_sort(G)]
    pred_mask = [0] * m
    for v in topo_idx:
        pm = 0
        for u in parents[v]:
            pm |= pred_mask[u] | (1 << u)
        pred_mask[v] = pm

    # --- 4) Faithful LowerY(P) recursion, but implemented over bitmasks for speed.
    # Pred(i,P) = {j in P : j ≺ i} = remaining ∩ pred_mask[i]
    #
    # LowerY(remaining):
    #   while remaining nonempty:
    #     i = first element of the Y-sorted sequence among remaining
    #     output LowerY(Pred(i, remaining))
    #     output i
    #     remove Pred(i,remaining) and i from remaining
    #
    # We implement i = argmin_{v in remaining} (Y, tie_key) by scanning y_order
    # (O(m) per chosen i) and use bitmasks for set ops.

    all_mask = (1 << m) - 1
    remaining0 = all_mask

    def pick_min_by_Y(rem_mask: int) -> int:
        # Return index of smallest-Y remaining node, according to y_order
        for vi in y_order_idx:
            if rem_mask & (1 << vi):
                return vi
        raise RuntimeError("pick_min_by_Y called with empty mask")

    def lowerY_mask(rem_mask: int) -> List[int]:
        out: List[int] = []
        while rem_mask:
            i = pick_min_by_Y(rem_mask)
            preds = rem_mask & pred_mask[i]  # Pred(i,P)
            if preds:
                out.extend(lowerY_mask(preds))
                rem_mask &= ~preds
            out.append(i)
            rem_mask &= ~(1 << i)
        return out

    order_idx = lowerY_mask(remaining0)

    # --- 5) Sanity: ensure topological
    # (Not strictly required, but useful for comparison/debug)
    pos = {node_of[i]: k for k, i in enumerate(order_idx)}
    for u, v in G.edges():
        if pos[u] > pos[v]:
            raise AssertionError("LowerY output is not topological (unexpected).")

    return [node_of[i] for i in order_idx]

# ---------------------------------------------------------------------
# topological order
# ---------------------------------------------------------------------

def Kahn_order(poset, Y: np.ndarray) -> List:
    """
    Candidate order
    
    At each step:
      - among current minimal elements (in-degree 0 w.r.t. remaining nodes),
        pick the one with smallest Y.
    Inputs

    poset: poset / graph

    Y: array-like aligned with nodes = list(G.nodes()), or mapping Y[v]
    
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
