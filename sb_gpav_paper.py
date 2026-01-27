"""
sb_gpav_paper.py

Segmentation-Based GPAV (SB-GPAV) following:

  Sysoev, Burdakov, Grimvall (2011)
  "A segmentation based algorithm for large scale partially ordered monotonic regression"
- Inputs are a finite poset given directly as a Hasse diagram (covers) + Y (+ optional weights).
- Segmentation stage (Algorithm 4, Segmentation stage):
    S1) Choose any topological order T (we optionally choose the paper’s trend-following order).
    S2) Split T into s contiguous segments.
    S3-S4) For each segment, build the induced subposet (on that segment’s nodes) and run GPAV.
         IMPORTANT (paper-faithful): within each segment we use the segment-restricted order
         coming from T (or an explicit per-segment override), NOT a second trend-following call.
- Assembly stage (Algorithm 4, Assembly stage; Algorithms 2–3):
    - Treat the segment blocks as "supernodes" (blocks).
    - Build a block precedence DAG via the Min–Max test for finite posets:
        Block A ≺ Block B  iff  (∀ a ∈ MAX(A)) (∀ b ∈ MIN(B))   a ≺ b
      where MIN(B) and MAX(A) are sets of minimal/maximal elements in the block (in the poset sense).
    - Remove redundant edges (we use transitive reduction).
    - Run GPAV on this block DAG using block averages and block weights.
    - Propagate fitted block values back to original nodes.


"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union, Any

import numpy as np
import networkx as nx

from gpav import gpav_seg
from trend_following import trend_following_order


ArrayLike = Union[np.ndarray, Sequence[float]]


# ---------------------------------------------------------------------
# Basic helpers: Hasse input + alignment
# ---------------------------------------------------------------------

def _hasse_graph(poset_or_graph) -> nx.DiGraph:
    """
    Accept:
      - nx.DiGraph that is already a Hasse diagram (covers),
      - hasse.PoSet-like with a `.hasse` attribute or method returning nx.DiGraph.
    """
    if isinstance(poset_or_graph, nx.DiGraph):
        G = poset_or_graph
    else:
        h = getattr(poset_or_graph, "hasse", None)
        if h is None:
            raise TypeError("Expected nx.DiGraph or a PoSet-like object with `.hasse`.")
        G = h() if callable(h) else h
        if not isinstance(G, nx.DiGraph):
            raise TypeError("`.hasse` must yield a networkx.DiGraph.")
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Input Hasse diagram must be a DAG.")
    return G


def _as_1d_float(a: ArrayLike, *, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D; got shape {arr.shape!r}")
    return arr


def _align_map_to_nodes(values: Union[Dict, ArrayLike], nodes: List, *, name: str) -> np.ndarray:
    """
    Align either:
      - dict[label->float], or
      - array-like already aligned to `nodes`
    into an ndarray of shape (n,).
    """
    n = len(nodes)
    if isinstance(values, dict):
        return np.array([float(values[v]) for v in nodes], dtype=float)
    arr = _as_1d_float(values, name=name)
    if arr.shape[0] != n:
        # try mapping-like indexing by label
        try:
            return np.array([float(values[v]) for v in nodes], dtype=float)
        except Exception as e:
            raise ValueError(f"Could not align {name} with poset nodes.") from e
    return arr.astype(float, copy=False)


def _as_reduced_hasse(G: nx.DiGraph, *, inputs_are_reduced: bool) -> nx.DiGraph:
    """
    If inputs_are_reduced=False, we transitive-reduce (covers).
    If True, we trust it's already a Hasse diagram.
    """
    if inputs_are_reduced:
        return G
    if G.number_of_edges() == 0:
        return G
    # For a DAG, networkx.transitive_reduction is deterministic for a fixed adjacency.
    return nx.transitive_reduction(G)


# ---------------------------------------------------------------------
# Paper Algorithm 4 (S1): choose topological order
# ---------------------------------------------------------------------

def _choose_processing_order(
    H: nx.DiGraph,
    Y_aligned: np.ndarray,
    *,
    use_trend_following: bool,
    stable_tiebreak: bool,
) -> List:
    """
    Returns a list of node labels forming a valid topological order of H.

    The SB paper allows any topological order. If use_trend_following=True, we use
    the paper's "LowerY" / trend-following heuristic (Algorithm 5), already implemented
    in trend_following.py, to pick a specific topological order.
    """
    if use_trend_following:
        # trend_following_order expects (G, Y) aligned with list(G.nodes())
        return list(trend_following_order(H, Y_aligned, stable_tiebreak=stable_tiebreak))
    return list(nx.topological_sort(H))


# ---------------------------------------------------------------------
# Paper Algorithm 4 (S2): split order into segments
# ---------------------------------------------------------------------

def _split_topo_into_segments(T: Sequence, n_segments: int) -> List[List]:
    T = list(T)
    n = len(T)
    if n_segments <= 0:
        raise ValueError("n_segments must be a positive integer.")
    n_segments = min(n_segments, n)
    base = n // n_segments
    rem = n % n_segments
    segs: List[List] = []
    start = 0
    for s in range(n_segments):
        size = base + (1 if s < rem else 0)
        segs.append(T[start:start + size])
        start += size
    return segs


# ---------------------------------------------------------------------
# Induced subposet on a subset (Hasse input version)
# ---------------------------------------------------------------------

def _descendants_cached(G: nx.DiGraph, u, cache: Dict) -> set:
    """
    Cache nx.descendants(G,u) for repeated reachability queries.
    """
    if u in cache:
        return cache[u]
    d = nx.descendants(G, u)
    cache[u] = d
    return d


def _induced_hasse_on_nodes(H_full: nx.DiGraph, nodes_subset: Sequence, *, reach_cache: Dict) -> nx.DiGraph:
    """
    Build the Hasse diagram of the induced subposet on nodes_subset.

    Faithful induced-poset construction:
      1) Build the induced comparability DAG (restricted transitive closure):
            edge u->v if u ≺ v in H_full AND u,v in subset
      2) Transitive reduction -> induced Hasse.
    """
    nodes_subset = list(nodes_subset)
    subset_set = set(nodes_subset)

    G_tc = nx.DiGraph()
    G_tc.add_nodes_from(nodes_subset)

    # Build restricted transitive closure edges using cached descendants in H_full.
    for u in nodes_subset:
        d = _descendants_cached(H_full, u, reach_cache)
        for v in d:
            if v in subset_set:
                G_tc.add_edge(u, v)

    if G_tc.number_of_edges() == 0:
        return G_tc

    if not nx.is_directed_acyclic_graph(G_tc):
        raise ValueError("Induced comparability graph is not a DAG (unexpected).")

    return nx.transitive_reduction(G_tc)


# ---------------------------------------------------------------------
# Blocks and extrema (finite-poset min/max sets)
# ---------------------------------------------------------------------

def _block_extrema_nodes(
    H_full: nx.DiGraph,
    block_members: Sequence,
    *,
    reach_cache: Dict,
) -> Tuple[List, List]:
    """
    For a block B (a set of nodes), compute:
      MIN(B): nodes with no predecessor within B
      MAX(B): nodes with no successor within B

    Using reachability in the FULL poset, restricted to the block.
    """
    B = list(block_members)
    Bset = set(B)

    has_pred = {x: False for x in B}
    has_succ = {x: False for x in B}

    # For each y in B, mark its descendants within B as having a predecessor.
    # Also mark y as having successor if it reaches someone in B.
    for y in B:
        dy = _descendants_cached(H_full, y, reach_cache)
        # Restrict dy to the block
        inside = [z for z in dy if z in Bset]
        if inside:
            has_succ[y] = True
            for z in inside:
                has_pred[z] = True

    mins = [x for x in B if not has_pred[x]]
    maxs = [x for x in B if not has_succ[x]]
    return mins, maxs


def _block_precedes_poset(
    H_full: nx.DiGraph,
    maxA: Sequence,
    minB: Sequence,
    *,
    reach_cache: Dict,
) -> bool:
    """
    SB Min–Max precedence for finite posets (set-based):
      A ≺ B  iff  ∀ a ∈ MAX(A), ∀ b ∈ MIN(B):  a ≺ b  (reachability)

    """
    if not maxA or not minB:
        return False

    minB = list(minB)
    for a in maxA:
        da = _descendants_cached(H_full, a, reach_cache)
        for b in minB:
            if b not in da:
                return False
    return True


# ---------------------------------------------------------------------
# Segment-local GPAV (Algorithm 4, S3–S4)
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class SegmentBlock:
    """
    A block produced by GPAV inside a segment.
    """
    global_id: int            # assigned after flattening across segments
    segment_index: int
    local_block_id: int
    members: List             # node labels
    value: float              # segment GPAV block value
    weight: float             # sum weights
    mins: List                # MIN(block) as node labels
    maxs: List                # MAX(block) as node labels


def _local_blocks_for_segment(
    H_full: nx.DiGraph,
    Y_map_full: Dict,
    W_map_full: Dict,
    segment_nodes: Sequence,
    *,
    seg_topo: Sequence,
    stable_tiebreak: bool, # to remove
    verbose: bool,
    reach_cache: Dict,
    seg_index: int,
) -> Tuple[np.ndarray, List[Dict], List[List], List[float], List[float]]:
    """
    Run GPAV on the induced segment poset.

    - GPAV needs a linear extension of the induced segment poset.
    - We use `seg_topo`, i.e., the restriction of the global topo order T to this segment
      (or an explicit per-segment override).

    Returns:
      u_seg_aligned_to_seg_nodes : np.ndarray (same order as `segment_nodes`)
      block_list_gpav            : List[Dict] (gpav.py format)
      blocks_members_labels      : List[List[label]]
      block_values               : List[float]
      block_weights              : List[float]
    """
    segment_nodes = list(segment_nodes)
    seg_topo = list(seg_topo)

    # Induced Hasse on this segment
    H_seg = _induced_hasse_on_nodes(H_full, segment_nodes, reach_cache=reach_cache)

    # Align Y/W to H_seg's internal node list
    N_seg = list(H_seg.nodes())

    # Provide mapping dicts to gpav_seg keyed by node labels, so alignment is unambiguous.
    Y_map_seg = {v: float(Y_map_full[v]) for v in N_seg}
    W_map_seg = {v: float(W_map_full[v]) for v in N_seg}

    # Use the segment-restricted order (or the override) directly:
    topo_used = seg_topo

    u_seg, block_list, _ = gpav_seg(
        Y=Y_map_seg,
        poset=H_seg,
        topo_order=topo_used,
        weights=W_map_seg,
        verbose=verbose,
        name=f"GPAV(segment {seg_index})",
        indent="  ",
    )

    # Convert u_seg (aligned to N_seg) to the segment_nodes order
    idx_of = {v: i for i, v in enumerate(N_seg)}
    u_by_segment_nodes = np.array([float(u_seg[idx_of[v]]) for v in segment_nodes], dtype=float)

    # Extract block members by labels + weights/values
    blocks_members_labels: List[List] = []
    block_values: List[float] = []
    block_weights: List[float] = []

    for b in block_list:
        local_elems = list(map(int, b["elements"]))
        members = [N_seg[loc] for loc in local_elems]
        blocks_members_labels.append(members)
        block_values.append(float(b["value"]))
        block_weights.append(float(b["weight"]))

    return u_by_segment_nodes, block_list, blocks_members_labels, block_values, block_weights


def _flatten_segment_blocks(
    H_full: nx.DiGraph,
    segments: List[List],
    segments_blocks_members: List[List[List]],
    segments_block_values: List[List[float]],
    segments_block_weights: List[List[float]],
    *,
    reach_cache: Dict,
) -> Tuple[
    List[SegmentBlock],
    Dict,
    np.ndarray,
    np.ndarray,
]:
    """
    Flatten all segment-local blocks into a single global list.

    Returns:
      blocks               : List[SegmentBlock] with global_id assigned 0..B-1
      label_to_blockid     : Dict[label -> global block id]
      block_values_global  : np.ndarray (B,)
      block_weights_global : np.ndarray (B,)
    """
    blocks: List[SegmentBlock] = []
    label_to_blockid: Dict[Any, int] = {}

    values: List[float] = []
    weights: List[float] = []

    gid = 0
    for s_idx, _seg_nodes in enumerate(segments):
        members_list = segments_blocks_members[s_idx]
        vals_list = segments_block_values[s_idx]
        wts_list = segments_block_weights[s_idx]

        for local_bid, members in enumerate(members_list):
            mins, maxs = _block_extrema_nodes(H_full, members, reach_cache=reach_cache)

            blk = SegmentBlock(
                global_id=gid,
                segment_index=s_idx,
                local_block_id=int(local_bid),
                members=list(members),
                value=float(vals_list[local_bid]),
                weight=float(wts_list[local_bid]),
                mins=list(mins),
                maxs=list(maxs),
            )
            blocks.append(blk)
            values.append(blk.value)
            weights.append(blk.weight)

            for v in blk.members:
                label_to_blockid[v] = gid

            gid += 1

    return blocks, label_to_blockid, np.array(values, dtype=float), np.array(weights, dtype=float)


# ---------------------------------------------------------------------
# Assembly stage (Algorithms 2–3): build block DAG via Min–Max test
# ---------------------------------------------------------------------

def _build_block_dag_minmax(
    H_full: nx.DiGraph,
    blocks: List[SegmentBlock],
    *,
    reach_cache: Dict,
) -> nx.DiGraph:
    """
    Nodes: global block ids 0..B-1
    Edge i->j if block i precedes block j per the SB Min–Max test.

    After edge generation, we remove redundant edges via transitive reduction.
    """
    B = len(blocks)
    G_B = nx.DiGraph()
    G_B.add_nodes_from(range(B))

    for i in range(B):
        bi = blocks[i]
        for j in range(i + 1, B):
            bj = blocks[j]
            if _block_precedes_poset(H_full, bi.maxs, bj.mins, reach_cache=reach_cache):
                G_B.add_edge(i, j)
            elif _block_precedes_poset(H_full, bj.maxs, bi.mins, reach_cache=reach_cache):
                G_B.add_edge(j, i)

    if G_B.number_of_edges() > 0:
        if not nx.is_directed_acyclic_graph(G_B):
            raise ValueError("Block precedence graph is not a DAG (unexpected under exact Min–Max).")
        G_B = nx.transitive_reduction(G_B)

    return G_B


def _choose_block_topo_order(
    G_B: nx.DiGraph,
    block_values: np.ndarray,
    *,
    use_trend_following: bool,
    stable_tiebreak: bool,
) -> List[int]:
    """
    SB paper: any topological order of the assembled block DAG.
    Optionally choose trend-following order on blocks (still a topo order).
    """
    if use_trend_following:
        # trend_following expects Y aligned to list(G.nodes()).
        # We inserted nodes as 0..B-1, so this aligns.
        return list(trend_following_order(G_B, block_values, stable_tiebreak=stable_tiebreak))
    return list(nx.topological_sort(G_B))


def _gpav_on_block_dag(
    G_B: nx.DiGraph,
    block_values: np.ndarray,
    block_weights: np.ndarray,
    *,
    use_trend_following_blocks: bool,
    stable_tiebreak: bool,
    verbose: bool,
) -> Tuple[np.ndarray, List[Dict], np.ndarray, List[int]]:
    """
    Run GPAV on the assembled block DAG (SB Algorithm 4 assembly stage).
    """
    topo_blocks = _choose_block_topo_order(
        G_B,
        block_values,
        use_trend_following=use_trend_following_blocks,
        stable_tiebreak=stable_tiebreak,
    )

    u_blocks, final_block_list, blockid_to_finalblock = gpav_seg(
        Y=block_values,
        poset=G_B,
        topo_order=topo_blocks,
        weights=block_weights,
        verbose=verbose,
        name="GPAV(blocks)",
        indent="  ",
    )

    return u_blocks, final_block_list, blockid_to_finalblock, topo_blocks


def _propagate_blocks_to_nodes(
    nodes_full: List,
    label_to_blockid: Dict,
    u_blocks_aligned: np.ndarray,
) -> np.ndarray:
    """
    Map the block-level fitted values back to original nodes.
    Output aligned to nodes_full = list(H_full.nodes()).
    """
    return np.array([float(u_blocks_aligned[label_to_blockid[v]]) for v in nodes_full], dtype=float)


# ---------------------------------------------------------------------
# Public entry point: sb_gpav
# ---------------------------------------------------------------------

def sb_gpav(
    poset,
    Y: Union[Dict, ArrayLike],
    *,
    weights: Optional[Union[Dict, ArrayLike]] = None,
    n_segments: int = 10,
    use_trend_following_first: bool = True,
    use_trend_following_blocks: bool = True,
    stable_tiebreak: bool = True,
    inputs_are_reduced: bool = True,
    verbose: bool = False,
    debug: bool = False,
    return_by_label: bool = True,
    segment_topo_orders: Optional[List[Optional[List]]] = None,
) -> Union[np.ndarray, Dict, Tuple]:
    """
    SB-GPAV using a Hasse diagram + Y (+ optional weights).

    Parameters
    ----------
    poset:
        nx.DiGraph (Hasse diagram) or PoSet-like with `.hasse`.
    Y:
        dict[label->float] or array aligned to list(H.nodes()).
    weights:
        dict[label->float] or array aligned to list(H.nodes()). Default all ones.
    n_segments:
        number of segments s (Algorithm 4, S2).
    use_trend_following_first:
        if True, choose topo order T using the paper's trend-following order (Algorithm 5).
        otherwise, use any topo sort.
    use_trend_following_blocks:
        if True, choose topo order on G_B using trend-following (Algorithm 5) on blocks.
    stable_tiebreak:
        passed through to trend_following_order, to be removed.
    inputs_are_reduced:
        if False, transitive-reduce the given DAG first.
    debug:
        if True, return (u, debug_info) where debug_info includes blocks + G_B, etc.
    return_by_label:
        by default True, returns dict[label->value], if False returns an array aligned to list(H.nodes()).
    segment_topo_orders:
        Optional list of per-segment topo orders (each is a list of node labels).
        If provided, it must have length = number of segments, and each non-None entry
        overrides that segment’s processing order (paper still allows any topo order).

    Returns
    -------
    If debug=False:
        u_final : dict[label->float] OR np.ndarray (aligned to list(H.nodes())) if return_by_label=False
    If debug=True:
        (u_final, debug_info)
    """
    H0 = _hasse_graph(poset)
    H_full = _as_reduced_hasse(H0, inputs_are_reduced=inputs_are_reduced)

    nodes_full = list(H_full.nodes())
    n = len(nodes_full)

    Y_aligned = _align_map_to_nodes(Y, nodes_full, name="Y")

    if weights is None:
        W_aligned = np.ones(n, dtype=float)
    else:
        W_aligned = _align_map_to_nodes(weights, nodes_full, name="weights")

    # Map form (convenient for segment extraction)
    Y_map_full = {v: float(Y_aligned[i]) for i, v in enumerate(nodes_full)}
    W_map_full = {v: float(W_aligned[i]) for i, v in enumerate(nodes_full)}

    # Reachability cache (used across induced subposets + min/max + block comparisons)
    reach_cache: Dict[Any, set] = {}

    # Algorithm 4 (S1): choose T
    T = _choose_processing_order(
        H_full,
        Y_aligned,
        use_trend_following=use_trend_following_first,
        stable_tiebreak=stable_tiebreak,
    )

    # Algorithm 4 (S2): segment
    segments = _split_topo_into_segments(T, n_segments=n_segments)

    if segment_topo_orders is not None:
        if len(segment_topo_orders) != len(segments):
            raise ValueError("segment_topo_orders must have the same length as the number of segments.")

    # Algorithm 4 (S3–S4): local GPAV per segment
    segments_blocks_members: List[List[List]] = []
    segments_block_values: List[List[float]] = []
    segments_block_weights: List[List[float]] = []

    for s_idx, seg_nodes in enumerate(segments):
        seg_topo = seg_nodes
        if segment_topo_orders is not None and segment_topo_orders[s_idx] is not None:
            seg_topo = list(segment_topo_orders[s_idx])

        # we pass seg_topo directly into gpav_seg.
        _, _, blk_members, blk_vals, blk_wts = _local_blocks_for_segment(
            H_full=H_full,
            Y_map_full=Y_map_full,
            W_map_full=W_map_full,
            segment_nodes=seg_nodes,
            seg_topo=seg_topo,
            stable_tiebreak=stable_tiebreak,
            verbose=verbose,
            reach_cache=reach_cache,
            seg_index=s_idx,
        )

        segments_blocks_members.append(blk_members)
        segments_block_values.append(blk_vals)
        segments_block_weights.append(blk_wts)

    # Flatten blocks
    blocks, label_to_blockid, block_values, block_weights = _flatten_segment_blocks(
        H_full=H_full,
        segments=segments,
        segments_blocks_members=segments_blocks_members,
        segments_block_values=segments_block_values,
        segments_block_weights=segments_block_weights,
        reach_cache=reach_cache,
    )

    # Assembly stage: build block DAG and run GPAV on it
    G_B = _build_block_dag_minmax(H_full, blocks, reach_cache=reach_cache)

    u_blocks, final_block_list, blockid_to_finalblock, topo_blocks = _gpav_on_block_dag(
        G_B,
        block_values=block_values,
        block_weights=block_weights,
        use_trend_following_blocks=use_trend_following_blocks,
        stable_tiebreak=stable_tiebreak,
        verbose=verbose,
    )

    # Propagate to nodes (aligned to nodes_full)
    u_final = _propagate_blocks_to_nodes(nodes_full, label_to_blockid, u_blocks)

    if return_by_label:
        u_out = {v: float(u_final[i]) for i, v in enumerate(nodes_full)}
    else:
        u_out = u_final

    if not debug:
        return u_out

    debug_info = {
        "H_full": H_full,
        "nodes_full": nodes_full,
        "T": T,
        "segments": segments,
        "blocks": blocks,  # List[SegmentBlock], includes mins/maxs sets
        "label_to_blockid": label_to_blockid,
        "block_values": block_values,
        "block_weights": block_weights,
        "G_B": G_B,
        "topo_blocks": topo_blocks,
        "u_blocks": u_blocks,
        "final_block_list": final_block_list,
        "blockid_to_finalblock": blockid_to_finalblock,
    }
    return u_out, debug_info
