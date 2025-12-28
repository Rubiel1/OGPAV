"""
sb_gpav_paper.py

A "paper-faithful" Segmentation-Based GPAV (SB-GPAV) implementation following:

  Sysoev, Burdakov, Grimvall (2011)
  "A segmentation based algorithm for large scale partially ordered monotonic regression"

Design goals:
- Data-only interface: inputs are X (n,p) and Y (n,) (+ optional weights).
- Uses strict dominance to define the underlying partial order:
      i ≺ j  iff  X[i] <= X[j] componentwise AND X[i] != X[j].
  This guarantees a DAG even when duplicates exist (duplicates become incomparable).
- Segmentation stage:
    1) Choose a topological order T via the paper's trend-following "LowerY" order
       (implemented in trend_following.py) on the dominance DAG.
    2) Split T into S segments.
    3) For each segment, build the induced poset and run GPAV on that segment, using
       the segment-specific order to mimic the paper.
- Assembly stage:
    - Treat segment blocks as nodes; each block has (min_x, max_x, weight, value).
    - Create the block precedence DAG using Min–Max comparability:
          A ≺ B  if max(A) <= min(B) componentwise and max(A) != min(B).
      This is the Min–Max test underlying the paper’s Algorithm 3.
    - Hasse-reduce with transitive reduction (equivalent to redundancy removal in Algorithm 2).
    - Run weighted GPAV on the block DAG.
    - Map fitted block values back to original observations.

Notes:
- The paper contains more elaborate data structures to avoid O(n^2) work; this module
  prioritizes correctness and clarity. For very large n, replace _dominance_dag
  with a faster edge generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import networkx as nx

from gpav import gpav_seg
from trend_following import trend_following_order


ArrayLike = Union[np.ndarray, Sequence[float]]


def _as_1d_float(a: ArrayLike, *, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D; got shape {arr.shape!r}")
    return arr


def _as_2d_float(X: Union[np.ndarray, Sequence[Sequence[float]]]) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n,p); got shape {X.shape!r}")
    return X


def _strict_dominates(xi: np.ndarray, xj: np.ndarray) -> bool:
    """Return True iff xi <= xj componentwise and xi != xj."""
    le = np.all(xi <= xj)
    if not le:
        return False
    return np.any(xi < xj)


def _dominance_dag(
    X: np.ndarray,
    nodes: Optional[Sequence[int]] = None,
) -> nx.DiGraph:
    """
    Build the strict-dominance DAG on indices in `nodes` (or all indices).
    Naive O(m^2 p) construction.
    """
    n = X.shape[0]
    if nodes is None:
        nodes = list(range(n))
    nodes = list(map(int, nodes))
    G = nx.DiGraph()
    G.add_nodes_from(nodes)

    # naive pairwise scan
    for a_i, i in enumerate(nodes):
        xi = X[i]
        for j in nodes[a_i + 1 :]:
            xj = X[j]
            if _strict_dominates(xi, xj):
                G.add_edge(i, j)
            elif _strict_dominates(xj, xi):
                G.add_edge(j, i)
    return G


def _induced_hasse_dag(
    X: np.ndarray,
    members: Sequence[int],
) -> nx.DiGraph:
    """
    Induced strict-dominance DAG on `members`, reduced to Hasse via transitive reduction.
    """
    G = _dominance_dag(X, members)
    if G.number_of_edges() > 0:
        if not nx.is_directed_acyclic_graph(G):
            raise ValueError("Dominance graph is not a DAG. Check X for NaNs or invalid values.")
        G = nx.transitive_reduction(G)
    return G


def _split_into_segments(order: Sequence[int], n_segments: int) -> List[List[int]]:
    order = list(map(int, order))
    n = len(order)
    if n_segments <= 0:
        raise ValueError("n_segments must be a positive integer.")
    n_segments = min(n_segments, n)  # cannot have more segments than points
    base = n // n_segments
    rem = n % n_segments
    segs = []
    start = 0
    for s in range(n_segments):
        size = base + (1 if s < rem else 0)
        segs.append(order[start : start + size])
        start += size
    return segs


@dataclass(frozen=True)
class SegmentBlock:
    """A block produced by GPAV inside a segment."""

    id: int
    members: np.ndarray  # global indices of points
    value: float  # block common value from segment GPAV
    weight: float  # sum of weights in block
    min_x: np.ndarray  # componentwise min of X over members
    max_x: np.ndarray  # componentwise max of X over members


def _blocks_from_segment_gpav(
    X: np.ndarray,
    Y: np.ndarray,
    weights: np.ndarray,
    segment_members: Sequence[int],
    *,
    topo_order: Sequence[int],
    verbose: bool = False,
) -> Tuple[np.ndarray, List[SegmentBlock], Dict[int, int]]:
    """
    Run Hasse-reduced GPAV on the induced segment poset. Return:
      fitted values on segment members (aligned to segment_members),
      list of SegmentBlock,
      mapping point_index -> block_id.
    """
    segment_members = list(map(int, segment_members))
    G_seg = _induced_hasse_dag(X, segment_members)

    # Provide dicts keyed by node labels so gpav_seg can align consistently with
    # the internal node order N = list(G_seg.nodes()). (This avoids relying on
    # `segment_members` matching that internal order.)
    Y_map = {i: float(Y[i]) for i in segment_members}
    w_map = {i: float(weights[i]) for i in segment_members}

    fitted_seg, block_list, _ = gpav_seg(
        Y=Y_map,
        poset=G_seg,
        topo_order=list(topo_order),
        weights=w_map,
        verbose=verbose,
        name="GPAV(segment)",
        indent="  ",
    )

    # gpav_seg aligns outputs to internal node list N = list(G.nodes()).
    N = list(G_seg.nodes())
    idx_of_node = {v: k for k, v in enumerate(N)}
    fitted_by_member = np.array([float(fitted_seg[idx_of_node[i]]) for i in segment_members], dtype=float)

    blocks: List[SegmentBlock] = []
    point_to_block: Dict[int, int] = {}

    for b_id, b in enumerate(block_list):
        local_elems = list(map(int, b["elements"]))
        global_members = np.array([N[loc] for loc in local_elems], dtype=int)

        for gi in global_members:
            point_to_block[int(gi)] = int(b_id)

        bx = X[global_members]
        blocks.append(
            SegmentBlock(
                id=int(b_id),
                members=global_members,
                value=float(b["value"]),
                weight=float(b["weight"]),
                min_x=bx.min(axis=0),
                max_x=bx.max(axis=0),
            )
        )

    return fitted_by_member, blocks, point_to_block


def _block_precedes(a: SegmentBlock, b: SegmentBlock) -> bool:
    """Min–Max precedence test: a ≺ b iff max_x(a) <= min_x(b) componentwise and strict."""
    le = np.all(a.max_x <= b.min_x)
    if not le:
        return False
    return np.any(a.max_x < b.min_x)


def _build_block_graph(blocks: Sequence[SegmentBlock]) -> nx.DiGraph:
    """
    Build the block precedence DAG using Min–Max comparability and Hasse-reduce it.
    """
    m = len(blocks)
    GB = nx.DiGraph()
    GB.add_nodes_from(range(m))

    for i in range(m):
        for j in range(i + 1, m):
            bi, bj = blocks[i], blocks[j]
            if _block_precedes(bi, bj):
                GB.add_edge(i, j)
            elif _block_precedes(bj, bi):
                GB.add_edge(j, i)

    if GB.number_of_edges() > 0:
        if not nx.is_directed_acyclic_graph(GB):
            raise ValueError("Block precedence graph is not a DAG (unexpected under strict Min–Max).")
        GB = nx.transitive_reduction(GB)
    return GB


def sb_gpav_fit(
    X: Union[np.ndarray, Sequence[Sequence[float]]],
    Y: ArrayLike,
    *,
    weights: Optional[ArrayLike] = None,
    n_segments: int = 10,
    use_trend_following: bool = True,
    stable_tiebreak: bool = True,
    debug: bool = False,
    verbose: bool = False,
) -> Tuple[np.ndarray, List[Dict], nx.DiGraph]:
    """
    Fit SB-GPAV to data (X,Y) with optional weights.

    Returns
    -------
    fitted : np.ndarray, shape (n,)
        Monotone fitted values.
    final_blocks : list of dict
        Final GPAV blocks on the block-graph, each dict has:
            {'elements': [block_ids], 'weight': float, 'value': float}
    GB : nx.DiGraph
        The Hasse-reduced block graph used in the assembly stage.
    """
    X = _as_2d_float(X)
    Y = _as_1d_float(Y, name="Y")
    n = X.shape[0]
    if Y.shape[0] != n:
        raise ValueError(f"Y length {Y.shape[0]} does not match X rows {n}")

    if weights is None:
        w = np.ones(n, dtype=float)
    else:
        w = _as_1d_float(weights, name="weights")
        if w.shape[0] != n:
            raise ValueError(f"weights length {w.shape[0]} does not match X rows {n}")
        if np.any(w < 0):
            raise ValueError("weights must be nonnegative")

    # Full dominance DAG to compute a topological order.
    G_full = _dominance_dag(X)

    if use_trend_following:
        Y_map = {i: float(Y[i]) for i in range(n)}
        T = trend_following_order(G_full, Y_map, stable_tiebreak=stable_tiebreak)
    else:
        T = list(nx.topological_sort(G_full))

    if debug:
        if set(T) != set(range(n)):
            raise ValueError("Topological order does not include all nodes.")
        pos = {v: k for k, v in enumerate(T)}
        for u, v in G_full.edges():
            if pos[u] >= pos[v]:
                raise ValueError("Computed order is not topological (unexpected).")

    segments = _split_into_segments(T, n_segments=n_segments)

    all_blocks: List[SegmentBlock] = []
    point_to_global_block: Dict[int, int] = {}
    fitted = np.empty(n, dtype=float)

    block_offset = 0
    for seg in segments:
        if len(seg) == 0:
            continue

        topo_seg = list(seg)

        fitted_seg, blocks_seg, point_to_block_seg = _blocks_from_segment_gpav(
            X, Y, w, seg, topo_order=topo_seg, verbose=verbose
        )

        for k, i in enumerate(seg):
            fitted[int(i)] = float(fitted_seg[k])

        for b in blocks_seg:
            gbid = block_offset + b.id
            all_blocks.append(
                SegmentBlock(
                    id=gbid,
                    members=b.members,
                    value=b.value,
                    weight=b.weight,
                    min_x=b.min_x,
                    max_x=b.max_x,
                )
            )
        for pt, bid in point_to_block_seg.items():
            point_to_global_block[int(pt)] = block_offset + int(bid)

        block_offset += len(blocks_seg)

    GB = _build_block_graph(all_blocks)

    # Use dicts keyed by block id so gpav_seg aligns robustly with whatever
    # internal node ordering `list(GB.nodes())` happens to take.
    block_Y = {int(b.id): float(b.value) for b in all_blocks}
    block_w = {int(b.id): float(b.weight) for b in all_blocks}

    topo_B = list(nx.topological_sort(GB))

    fitted_B, final_blocks, _ = gpav_seg(
        Y=block_Y,
        poset=GB,
        topo_order=topo_B,
        weights=block_w,
        verbose=verbose,
        name="GPAV(block)",
        indent="",
    )

    N_B = list(GB.nodes())
    idx_of_bnode = {v: k for k, v in enumerate(N_B)}
    fitted_block_value = {int(bid): float(fitted_B[idx_of_bnode[bid]]) for bid in N_B}

    fitted_final = np.empty(n, dtype=float)
    for i in range(n):
        bid = point_to_global_block[int(i)]
        fitted_final[int(i)] = fitted_block_value[int(bid)]

    if debug:
        for u, v in G_full.edges():
            if fitted_final[u] > fitted_final[v] + 1e-12:
                raise ValueError("Fitted values violate monotonicity on dominance order (debug check).")

    return fitted_final, final_blocks, GB
