# -*- coding: utf-8 -*-
"""
============================== BIG-PICTURE (7 STEPS) ==============================
This file implements the "operadic GPAV" strategy for the
lexicographic sum P = Q(R_1, ..., R_m). 

1) Inputs (Q and a list of subposets R_i).  We run a local GPAV on each R_i.
2) For each R_i, GPAV computes blocks (pooled elements with a common value)
   and the block averages/weights.
3) For each R_i, we also build the block-poset poset(GPAV(R_i)) as a Hasse
   (cover-relation) DAG between those blocks.
4) For each R_i, we identify the extreme blocks of that block-poset:
   minima and maxima (by in/out-degree in the local Hasse DAG).  We also keep
   their element-label lists for optional debugging.
5) Stage 2 assembly: Create a global block DAG G_B. Nodes are all blocks
   from all R_i. Add two kinds of edges:
   (i) Intra: insert all intra-R_i block Hasse edges.
   (ii) Inter: for each cover edge i->j in Hasse(Q), connect every
        maximal block of R_i to every minimal block of R_j.
   This yields a graph that is already a transitive reduction (a Hasse) by
   construction.
6) Run GPAV again, now on G_B using block averages/weights. This produces the
   block-level fitted values u_blocks.
7) Propagate each block's fitted value back to its member elements to obtain the
   final vector u aligned with the original indices.
===================================================================================
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Hashable
import numpy as np
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed
import hasse

# NOTE: The algorithm expects GPAV with block-returns and a Trend-Following ordering helper.
from trend_following import trend_following_order
from gpav import gpav_op as gpav
import warnings
# --- Aliases---
NodeLabel = Hashable       # label of a node in a poset (local or global)
LocalIndex = int           # index 0..m_i-1 within a fiber R_i (position in nodes list)
GlobalIndex = int          # index 0..N-1 in the lexicographic sum
BlockId = int              # index of a block in a block-level DAG

# -------------------------
# Utilities
# -------------------------




def _offsets_from_subposets(R_subposets: List[hasse.PoSet]) -> List[int]:
    """Compute contiguous offsets so that the concatenation of R_i element-sets
    maps to [0..N-1].  This is only indexing bookkeeping; it does not touch
    the order structure.
    """
    offs = []
    cur = 0
    for R in R_subposets:
        offs.append(cur)
        cur += len(R)
    return offs

def _assemble_global_arrays_from_A_list(
    H_R_list: List[nx.DiGraph],
    offs: List[int],
    A_list: List[Dict[NodeLabel, float]],
    weights_list: Optional[List[Dict[NodeLabel, float]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build global A_array and W_array (length N) from per-fiber dicts keyed by local node labels.

    Each local label is an int
    and its global index is offset + int(lbl).
    """
    N = sum(len(H) for H in H_R_list)
    A_array = np.empty(N, dtype=float)
    W_array = np.ones(N, dtype=float)

    if len(A_list) != len(H_R_list):
        raise ValueError(f"A_list must have length {len(H_R_list)} (one per fiber).")

    if weights_list is not None and len(weights_list) != len(H_R_list):
        raise ValueError(f"weights_list must have length {len(H_R_list)} (one per fiber).")

    for i, H_R in enumerate(H_R_list):
        offset = offs[i]
        nodes = list(H_R.nodes())
        A_i = A_list[i]
        W_i = weights_list[i] if weights_list is not None else None

        for lbl in nodes:
            g = offset + int(lbl)  # same convention used elsewhere in this file
            if lbl not in A_i:
                raise KeyError(f"Missing A_list[{i}][{lbl}] for fiber {i}.")
            A_array[g] = float(A_i[lbl])
            if W_i is not None:
                W_array[g] = float(W_i.get(lbl, 1.0))

    return A_array, W_array



def as_reduced_hasse(poset_or_graph) -> nx.DiGraph:
    """Return the Hasse diagram (transitive reduction) of a DAG.
    Accepts either a PoSet-like object with `.hasse` (property or method)
    or a raw nx.DiGraph. Call this once per input.
    """
    H = getattr(poset_or_graph, "hasse", poset_or_graph)
    G = H() if callable(H) else H
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Expected a DAG to build a Hasse diagram.")
    return nx.transitive_reduction(G)




# -------------------------
# Local stage: run GPAV on each R_i and compute its block-level Hasse
# -------------------------

def _local_blocks_for_R(
    H_R: nx.DiGraph,
    A_seg,
    W_seg,
    *,
    use_trend_following: bool = True,
    verbose: bool = False,
    debug: bool=False,
    group_index: Optional[int] = None,
    custom_topo: Optional[List[NodeLabel]] = None,
) -> Tuple[
    List[List[LocalIndex]],           # members
    List[float],                      # block_values
    List[float],                      # block_weights
    np.ndarray,                       # u_seg (length m)
    nx.DiGraph,                       # G_loc
    Dict[LocalIndex, BlockId],        # elem_to_block
    List[BlockId],                    # mins_local
    List[BlockId],                    # maxs_local
    List[NodeLabel],                  # min_node_labels
    List[NodeLabel],                  # max_node_labels
]:
    """
    === Implements Steps 1–4 for a single R_i ===

    Run GPAV on one input R_i and return:

    Parameters
    ----------
    H_R :
        Hasse diagram of the input poset R_i.
    A_seg, W_seg :
    Either 1D array-like objects indexed by integer local labels 0..m−1,
    or dictionaries keyed by integer local node labels of H_R.

    Non-integer labels are not supported.

    custom_topo :
        Optional user-provided topological order (list of node labels of H_R).
        If given, this is passed as `topo_order` to `gpav` and `use_trend_following`
        is ignored for this segment.
    use_trend_following : bool
        If custom_topo is None, controls how the topological order for GPAV is chosen:
            True: use trend_following_order(H_R, Y_map_seg) 
            ( trend-following linear extension).
            False: use nx.topological_sort(H_R).

    verbose : bool
       If True, print progress and local diagnostic information for this fiber
        (segment size, chosen topo order, number of blocks, block summary, and local block Hasse edges).

    group_index : Optional[int]
        Index i of the fiber R_i within the lexicographic sum. Used only for labeling debug output;
         does not affect computation.
    
    Returns:

      - members:
          list of blocks, each as a list of local indices in 0..m-1.
          Local index j means "the j-th entry in nodes = list(H_R.nodes())".

      - block_values:
          weighted averages per block (same order as `members`).

      - block_weights:
          total weights per block.

      - u_seg:
          per-element fitted values, 1D array of length m, aligned with
          `nodes` (i.e. u_seg[j] is the value for nodes[j]).

      - G_loc:
          block-level Hasse DAG (nodes are BlockId values, edges are covers).

      - elem_to_block:
          mapping from LOCAL element index j (0..m-1) to BlockId.

      - mins_local / maxs_local:
          lists of BlockIds corresponding to minimal / maximal blocks
          in G_loc (by in/out-degree).

      - min_node_labels / max_node_labels:
          These are local node labels (labels from H_R), not global indices.
    """
    # Local node labels and count
    nodes = list(H_R.nodes())
    m = len(nodes)
    if verbose:
        print(f"[R{group_index}] Start segment: m={m}, |E|={H_R.number_of_edges()}")

    # ------------------------------------------------------------------
    # Normalize A_seg into a 1D float array A_arr and define a label->value
    # accessor _val(lbl). For array input we keep the existing semantics:
    # A_seg[j] is interpreted as the value for local label j (assumed int).
    # ------------------------------------------------------------------
    if isinstance(A_seg, dict):
        def _val(lbl):
            return float(A_seg[lbl])
        # For convenience (trivial / edgeless cases), also build an array
        # aligned to the current node list.
        A_arr = np.array([_val(v) for v in nodes], dtype=float)
    else:
        A_arr = np.asarray(A_seg, dtype=float)
        if A_arr.ndim != 1:
            raise ValueError(
                f"A_seg must be 1D or a dict keyed by local labels; "
                f"got shape {A_arr.shape!r}"
            )
        if A_arr.shape[0] < m:
            raise ValueError(
                f"A_seg has length {A_arr.shape[0]} but segment has {m} node(s)."
            )

        def _val(lbl):
            # keep current assumption: label is an int index into A_arr
            return float(A_arr[int(lbl)])

    # ------------------------------------------------------------------
    # Normalize W_seg into a 1D float array W_arr and define label->weight
    # accessor _wt(lbl) with the same conventions as for A_seg.
    # ------------------------------------------------------------------
    if isinstance(W_seg, dict):
        def _wt(lbl):
            return float(W_seg[lbl])
        W_arr = np.array([_wt(v) for v in nodes], dtype=float)
    else:
        W_arr = np.asarray(W_seg, dtype=float)
        if W_arr.ndim != 1:
            raise ValueError(
                f"W_seg must be 1D or a dict keyed by local labels; "
                f"got shape {W_arr.shape!r}"
            )
        if W_arr.shape[0] < m:
            raise ValueError(
                f"W_seg has length {W_arr.shape[0]} but segment has {m} node(s)."
            )

        def _wt(lbl):
            return float(W_arr[int(lbl)])
    min_node_labels = None 
    max_node_labels = None 

    # (Step 2 trivial case) If the segment is size-0/1, pooling is trivial.
    if m <= 1:
        if verbose:
            print(f"[R{group_index}] Trivial segment (m={m}). Returning identity.")
        members = [[0]] if m == 1 else []
        block_values = [float(A_arr[0])] if m == 1 else []
        block_weights = [float(W_arr[0])] if m == 1 else []
        u_seg = A_arr.copy()
        G_loc = nx.DiGraph()
        G_loc.add_nodes_from(range(m))  # node id 0 for m==1, else empty
        elem_to_block = {i: i for i in range(m)}

        mins_local = [0] if m == 1 else []
        maxs_local = [0] if m == 1 else []
        if debug:
            min_node_labels = [0] if m == 1 else []
            max_node_labels = [0] if m == 1 else []
        return members, block_values, block_weights, u_seg, G_loc, elem_to_block, mins_local, maxs_local, min_node_labels, max_node_labels

    # (Step 2 fast path) Edgeless segment -> already isotonic, keep singletons.
    if H_R.number_of_edges() == 0:
        if verbose:
            print(f"[R{group_index}] Edgeless segment. Skipping GPAV; returning singletons.")
        members = [[i] for i in range(m)]
        block_values = [float(A_arr[i]) for i in range(m)]
        block_weights = [float(W_arr[i]) for i in range(m)]
        u_seg = A_arr.copy()
        G_loc = nx.DiGraph(); G_loc.add_nodes_from(range(m))
        elem_to_block = {i: i for i in range(m)}

        ids = list(range(m))
        mins_local = ids[:]
        maxs_local = ids[:]
        if debug:
            min_node_labels = ids[:]    # singletons: each node is min & max
            max_node_labels = ids[:]
        return members, block_values, block_weights, u_seg, G_loc, elem_to_block, mins_local, maxs_local, min_node_labels, max_node_labels

    # (Step 1) Choose a topological order for GPAV.
    # Build label-based map Y_map_seg keyed by local node labels.
    Y_map_seg = {lbl: _val(lbl) for lbl in nodes}

    if custom_topo is not None:
        # User-provided linearization (labels of H_R)
        topo = list(custom_topo)

        # (Optional sanity check)
        node_set = set(nodes)
        if len(topo) != len(nodes) or set(topo) != node_set:
            raise ValueError(
                f"custom_topo for R[{group_index}] must be a permutation of the "
                f"segment's node labels. "
            )
        if verbose:
            print(f"[R{group_index}] Using custom topo order (first 20 entries): {topo[:20]}")
    else:
        # Default behavior: Trend_following or plain topological sort
        topo = (
            trend_following_order(H_R, Y_map_seg)
            if use_trend_following
            else list(nx.topological_sort(H_R))
        )
        if verbose:
            print(f"[R{group_index}] Topological order for GPAV (20 entries): {topo[:20]}")


    # Run GPAV on the segment (Step 2) and get the local block list and
    # the block-level edges (pre-Hasse); elem_to_block aligns elements->blocks.
    W_map_seg = {lbl: _wt(lbl) for lbl in nodes}
    u_seg, local_blocks, elem_to_block, block_edges = gpav(
        Y_map_seg, H_R, topo_order=topo, weights=W_map_seg, return_block_edges=True)

    # Extract members/values/weights of local blocks (Step 2 results).
    members: List[List[int]] = []
    block_values: List[float] = []
    block_weights: List[float] = []
    for b in local_blocks:
        elems = list(b["elements"])  # local indices
        members.append(elems)
        block_values.append(float(b["value"]))
        block_weights.append(float(b["weight"]))

    if verbose:
        print(f"[R{group_index}] Local GPAV produced {len(members)} block(s).")
        for k, (mem, val, wt) in enumerate(zip(members, block_values, block_weights)):
            print(f"  - B{k} members={mem}, av={val:.6g}, w={wt:.6g}")

    # (Step 3) Build the block-poset for this segment and Hasse-reduce it.
    G_loc = nx.DiGraph(); G_loc.add_nodes_from(range(len(members)))
    for a, b in block_edges:
        if a != b:
            G_loc.add_edge(int(a), int(b))
    if G_loc.number_of_edges() > 0:
        G_loc = nx.transitive_reduction(G_loc)
    if verbose:
        print(f"[R{group_index}] Local block Hasse edges (first 20): {list(G_loc.edges)[:20]}")

    # (Step 4) Identify extrema in the local block-poset.
    mins_local = [n for n in G_loc.nodes if G_loc.in_degree(n) == 0]
    maxs_local = [n for n in G_loc.nodes if G_loc.out_degree(n) == 0]

    if debug:
        min_node_labels = [lbl for b in mins_local for lbl in local_blocks[b]['min_labels']]
        max_node_labels = [lbl for b in maxs_local for lbl in local_blocks[b]['max_labels']]

    return members, block_values, block_weights, u_seg, G_loc, elem_to_block, mins_local, maxs_local, min_node_labels, max_node_labels


# -------------------------
# Main: factorized SB-GPAV for lexicographic sum
# -------------------------

def OGPAV(
    Q: hasse.PoSet,
    R_subposets: List[hasse.PoSet],
    A: Optional[np.ndarray | Dict[int, float]]=None,
    weights: Optional[np.ndarray | Dict[int, float]] = None,
    *,
    A_list: Optional[List[np.ndarray | Dict[NodeLabel, float]]] = None,
    weights_list: Optional[List[np.ndarray| Dict[int, float]]] = None,
    use_trend_following_first: bool = True,
    use_trend_following_blocks: bool = True,
    max_workers: Optional[int] = None,
    inputs_are_reduced: bool = False,
    verbose: bool = False,
    debug: bool = False,
    return_by_local_index: bool = False,
    segment_topo_orders: Optional[List[Optional[List[NodeLabel]]]] = None,
) -> np.ndarray:
    """
    Factorized SB-GPAV for the lexicographic sum P = Q(R_1, ..., R_m).

    Stage 1 (parallel):
      (Step 1) For each i, run GPAV on R_i using its Hasse H_Ri.
      (Step 2) Keep the local blocks and their block averages/weights.
      (Step 3) Build the local block-poset (Hasse) for each R_i.
      (Step 4) Record local minima/maxima blocks for later cross-wiring.

    Stage 2 (assemble, Hasse by construction):
      (Step 5.i) Start with the disjoint union of all intra-R_i block Hasse edges.
      (Step 5.ii) For each cover edge i->j in Hasse(Q), add edges from
                 every maximal block of R_i to every minimal block of R_j.
                 Only cover relations of Q are considered (immediate successors).
      (Step 6) Run GPAV on this global block DAG G_B.
      (Step 7) Propagate the block-level fitted values back to elements.


    Parameters:

    Q : hasse.PoSet The “outer” poset Q with m = len(R_subposets) elements, in the lexicographic sum 
    P=Q(R1,…,Rm)
    The code uses its Hasse diagram H_Q (cover relations). 
    If inputs_are_reduced=False, it reduces Q once via as_reduced_hasse.

    R_subposets : List[hasse.PoSet]
    List [R_0, ..., R_{m-1}] of input posets. Each R_i is run independently in Stage 1 
    (potentially in parallel).

    The total number of “inputs” in the lexicographic sum is N = sum(len(R_i)).

    
    A : Optional[np.ndarray | Dict[int, float]]
    Global observed data on the lexicographic sum.

    If provided as a dict:
        Interpreted as {global_index -> value}, where global_index ranges
        from 0 to N−1 according to the internal concatenation order induced
        by R_subposets.

    If provided as an array-like:
        Interpreted as already aligned with the internal global order
        0..N−1. This option is discouraged unless the user is certain of
        the internal ordering.

    If A_list is provided, A is ignored.


    A_list : Optional[List[Dict[NodeLabel, float]]]
    Optional alternative to A.

    Each A_list[i] must be a dictionary keyed by the local node labels of R_i.
    These labels are expected to be integers in {0, …, |R_i|−1}.

    The algorithm uses these integer labels to compute global indices via
    fixed offsets. Non-integer labels are not supported.


    weights : Optional[np.ndarray | Dict[int, float]]
    Optional global weights, using the same conventions as A.

    If None, all weights default to 1.0.
    If a dict, missing keys default to weight 1.0.

    Ignored if weights_list is provided.

    weights_list : Optional[List[Dict[NodeLabel, float]]]
    Optional per-fiber weights, analogous to A_list.

    Keys must be the same integer node labels used in R_i.

    use_trend_following_first : bool
    Controls the local ordering used for GPAV inside each input R_i when an order
    is not explicitly provided via segment_topo_orders[i].
        True: use trend_following_order (trend-following order).
        False: use a plain topological sort of H_Ri.

    use_trend_following_blocks : bool
    Controls the ordering used for GPAV on the global block DAG G_B in Stage 2.
    True: use trend_following order on G_B based on the current block averages.
    False: use nx.topological_sort(G_B).

    max_workers : Optional[int]
    Passed to ThreadPoolExecutor(max_workers=...).
    Controls the maximum number of worker threads used to process fibers R_i in parallel.
    If None, Python chooses a default.

    inputs_are_reduced : bool
    If False (default), the code reduces Q and each R_i to their Hasse diagrams once 
    using as_reduced_hasse.
    If True, the code assumes the user already supplied Hasse diagrams 
    for Q and all R_i, and skips reductions.

    verbose : bool
    If True, prints progress messages and diagnostic summaries for both stages.
    
    segment_topo_orders : Optional[List[Optional[List[NodeLabel]]]]
    Optional per-input R_i explicit processing orders.
    If provided, must have length len(R_subposets).
    segment_topo_orders[i] is either:
        None: no override; local order is chosen by use_trend_following_first, or
        a list of node labels of H_R_list[i]: explicit topological order used for fiber R_i, 
        overriding use_trend_following_first for that fiber.

    If return_by_local_index =True, returns List[Dict[NodeLabel, float]] aligned with A_list.

    Returns

    u_final : np.ndarray
    1D NumPy array of length N = sum(len(R_i)).
    u_final[g] is the fitted isotonic regression value for global atom index g 
    (where indexing is the internal concatenation order induced by R_subposets offsets).

    This is obtained by solving GPAV on each fiber, then GPAV on the induced global block DAG, 
    and finally propagating block-level fits back to atoms.  


    Important note on node labels
    -----------------------------
    This implementation assumes that all node labels in each input poset R_i
    (and hence in Q) are integers starting from 0, i.e. labels in {0, …, |R_i|−1}
    for each fiber R_i.

    These integer labels are used as local indices and are mapped to global
    indices via contiguous offsets. Internally, global indices are computed as

        global_index = offset_i + int(local_label)

    Using non-integer or non-contiguous labels will lead to incorrect indexing
    or runtime errors.

    """
    verbose = verbose or debug
    if verbose:
        print("== Operadic / Factorized SB-GPAV: start ==")
        print(f"Flags: use_trend_following_first={use_trend_following_first}, use_trend_following_blocks={use_trend_following_blocks}, "
              f"inputs_are_reduced={inputs_are_reduced}")
        a_len = (len(A) if A is not None else sum(len(d) for d in A_list) if A_list is not None else 0)
        print(f"Q nodes={len(Q)}, #R={len(R_subposets)}, |A|={a_len}")
        if weights is not None:
            print("Weights provided.")

    # Normalize Q once to its Hasse (covers only). (Prep for Step 5.)
    H_Q_attr = getattr(Q, "hasse", Q)
    H_Q = H_Q_attr() if callable(H_Q_attr) else H_Q_attr
    if not inputs_are_reduced:
        H_Q = as_reduced_hasse(H_Q)
        if verbose:
            print("Reduced Q to Hasse once.")
    else:
        if verbose:
            print("Assuming Q is already reduced (Hasse).")

    q_nodes = list(H_Q.nodes())
    if any(not isinstance(v, (int, np.integer)) for v in q_nodes):
        bad = [v for v in q_nodes if not isinstance(v, (int, np.integer))][:5]
        raise TypeError(
            f"Q has non-integer node labels (example: {bad}). "
            "This implementation expects Q labels to be integers 0..m-1."
        )
    if set(q_nodes) != set(range(len(q_nodes))):
        raise ValueError(
            f"Q labels must be exactly {{0..{len(q_nodes)-1}}}."
        )

    # Normalize each R_i once to its Hasse (covers only). (Used in Steps 1–4.)
    H_R_list: List[nx.DiGraph] = []
    for idx, R in enumerate(R_subposets):
        H_attr = getattr(R, "hasse", R)
        H = H_attr() if callable(H_attr) else H_attr
        if not inputs_are_reduced:
            H = as_reduced_hasse(H)
            if verbose:
                print(f"Reduced R[{idx}] to Hasse once.")
        H_R_list.append(H)
        # Early-fail: this implementation assumes integer local labels
        nodes = list(H.nodes())
        if any(not isinstance(v, (int, np.integer)) for v in nodes):
            bad = [v for v in nodes if not isinstance(v, (int, np.integer))][:5]
            raise TypeError(
                f"R[{idx}] has non-integer node labels (example: {bad}). "
                "This implementation expects labels to be integers."
            )
        m = len(nodes)
        # labels should be exactly {0..m-1}
        if set(nodes) != set(range(m)):
            raise ValueError(
                f"R[{idx}] labels must be exactly {{0..{m-1}}}, got {sorted(set(nodes))[:10]}..."
            )



    if segment_topo_orders is not None:
        if len(segment_topo_orders) != len(R_subposets):
            raise ValueError(
                f"segment_topo_orders must have length {len(R_subposets)} "
                f"(one per R_i), got {len(segment_topo_orders)}."
            )

    # Basics & alignment for vectorized data.
    offs = _offsets_from_subposets(R_subposets)
    N = sum(len(R) for R in R_subposets)

    # ------------------------------------------------------------------
    # Global data model:
    #   There are N nodes in the lexicographic sum P = Q(R_0,...,R_{m-1}).
    #   We treat their labels as integers 0..N-1.
    #   A_array[g] is the observed value at global label g.
    #   W_array[g] is the weight at global label g.
    #
    # The mapping "which global label belongs to which fiber R_i" is
    # controlled solely by `offs` (these offsets are contiguous sums
    # of |R_i|).
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Global data model (user can provide either A (global) OR A_list (per fiber)).
    # ------------------------------------------------------------------
    if A_list is not None:
        if A is not None:
            warnings.warn(
                "Both A and A_list were provided. Ignoring A and using A_list.",
                category=UserWarning,
                stacklevel=2,
            )
        A_array, W_array = _assemble_global_arrays_from_A_list(
            H_R_list=H_R_list,
            offs=offs,
            A_list=A_list,
            weights_list=weights_list,
        )
    else:
        if A is None:
            raise ValueError("You must provide either A (global) or A_list (per poset R_i).")

        if isinstance(A, dict):
            A_array = np.array([float(A[i]) for i in range(N)], dtype=float)
        else:
            warnings.warn(
                "A was provided as a sequence/array. GPAV assumes A[i] corresponds to internal index "
                "i=0..N-1 (the internal concatenation order). If your A is keyed by node labels or a "
                "different ordering, pass a dict-like mapping to avoid misalignment.",
                category=UserWarning,
                stacklevel=2,
            )
            A_array = np.asarray(A, dtype=float)

        if A_array.ndim != 1:
            raise ValueError(f"A must be 1D; got shape {A_array.shape!r}")
        if A_array.shape[0] != N:
            raise ValueError(f"A has length {A_array.shape[0]} but sum|R_i| = {N}")

        # Normalize weights
        if weights is None:
            W_array = np.ones(N, dtype=float)
        else:
            if isinstance(weights, dict):
                W_array = np.array([float(weights.get(i, 1.0)) for i in range(N)], dtype=float)
            else:
                W_array = np.asarray(weights, dtype=float)
            if W_array.ndim != 1:
                raise ValueError(f"weights must be 1D; got shape {W_array.shape!r}")
            if W_array.shape[0] != N:
                raise ValueError(f"weights has length {W_array.shape[0]} but sum|R_i| = {N}")
    

    # Quick sanity checks on Q vs. number of components.
    if H_Q.number_of_nodes()!= len(H_R_list):
        raise ValueError(f"base poset has length {H_Q.number_of_nodes()} but #|R_i| = {len(H_R_list)}, should put unary posets if needed")
    if H_Q.number_of_nodes() <=1:
        raise ValueError("Trivial composition, if R_0 is big it is better to use segmented_gpav, if the R_0 is small just use gpav")

    if verbose:
        print("Stage 1 — per-R_i local GPAV (parallel).")

    # --- Stage 1: local GPAV per R_i (parallel). Collect true local blocks and local Hasse.

    # Global collectors over all blocks from all R_i.
    block_members_global: List[List[int]] = []      # per block: list of global element indices
    block_values: List[float] = []
    block_weights: List[float] = []
    block_group: List[int] = []                     # which R_i a block came from

    # Per-group structures (indexed by i)
    G_loc_list: List[Optional[nx.DiGraph]] = [None] * len(R_subposets)
    group_min_global: List[Optional[List[int]]] = [None] * len(R_subposets)
    group_max_global: List[Optional[List[int]]] = [None] * len(R_subposets)
    if debug:
        group_min_node_labels: List[Optional[List[int]]] = [None] * len(R_subposets)
        group_max_node_labels: List[Optional[List[int]]] = [None] * len(R_subposets)
    else:
        group_min_node_labels = None 
        group_max_node_labels = None
    # Worker to process each R_i independently.
    def _worker(i: int):
        """
        Process one R_i:
          Build label-based segment maps from the global arrays.
          Run local GPAV to obtain blocks and segment fits.
          Map local block membership to global element indices.
        """
        H_R = H_R_list[i]
        offset = offs[i]

        # Local node labels for this fiber (e.g. 0..m_i-1). We never rely on
        # their order beyond what H_R gives us; local indices 0..m-1 are
        # positions in this list.
        local_nodes: List[NodeLabel] = list(H_R.nodes())
        m = len(local_nodes)

        # Build label-based segment dictionaries from the GLOBAL arrays.
        #
        # Convention:
        #   Each local label l is assumed to be an integer in 0..m_i-1.
        #   The corresponding GLOBAL label is (offset + l).
        #
        # A_array[g] / W_array[g] are the canonical global value/weight vectors.
        seg_vals = {
            lbl: float(A_array[offset + int(lbl)])
            for lbl in local_nodes
        }
        seg_w = {
            lbl: float(W_array[offset + int(lbl)])
            for lbl in local_nodes
        }

        if verbose:
            print(f"[R{i}] Dispatch worker: size={m}, off={offset}")

        (
            members,
            vals,
            wts,
            u_seg,
            G_loc,
            elem_to_block,
            mins_local,
            maxs_local,
            min_node_labels,
            max_node_labels,
        ) = _local_blocks_for_R(
            H_R,
            seg_vals,
            seg_w,
            use_trend_following=use_trend_following_first,
            verbose=verbose,
            debug= debug,
            group_index=i,
            custom_topo=None if segment_topo_orders is None else segment_topo_orders[i],
        )

        # Map local block membership (indices into local_nodes) -> global
        # element indices 0..N-1.
        #
        # LOCAL index j refers to the element whose label is local_nodes[j].
        # GLOBAL label is then (offset + local_label).
        members_global: List[List[GlobalIndex]] = [
            [offset + int(local_nodes[j]) for j in block]
            for block in members
        ]

        return (
            i,
            members_global,
            vals,
            wts,
            u_seg,
            G_loc,
            mins_local,
            maxs_local,
            min_node_labels,
            max_node_labels,
        )


    # Parallel map over subposets.
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_worker, i) for i in range(len(R_subposets))]
        for fut in as_completed(futures):
            i, members_g, vals, wts, u_seg, G_loc, mins_local, maxs_local, min_node_labels, max_node_labels = fut.result()
            if verbose:
                print(f"[R{i}] Completed: produced {len(members_g)} block(s).")

            # Determine starting global block index for this group.
            start_idx = len(block_members_global)

            # Collect blocks and attributes (Step 2 results).
            for k, mg in enumerate(members_g):
                block_members_global.append(mg)
                block_values.append(float(vals[k]))
                block_weights.append(float(wts[k]))
                block_group.append(i)

            # Build mapping local_block_id -> global_block_id for this group.
            local_to_global = {k: start_idx + k for k in range(len(members_g))}

            # Store intra-group block Hasse edges in global coordinates (Step 3).
            if G_loc.number_of_nodes() > 0:
                G_loc_global = nx.DiGraph(); G_loc_global.add_nodes_from(local_to_global.values())
                for u, v in G_loc.edges:
                    G_loc_global.add_edge(local_to_global[u], local_to_global[v])
                G_loc_list[i] = G_loc_global
            else:
                G_loc_list[i] = nx.DiGraph()

            # Compute minimal/maximal blocks for cross wiring (Step 4), then map globally.
            if G_loc.number_of_nodes() == 0:
                group_min_global[i] = []
                group_max_global[i] = []
            else:
                group_min_global[i] = [local_to_global[n] for n in mins_local]
                group_max_global[i] = [local_to_global[n] for n in maxs_local]

                if debug:
                    off = offs[i]
                    group_min_node_labels[i] = [off + int(v) for v in min_node_labels]
                    group_max_node_labels[i] = [off + int(v) for v in max_node_labels]

    B = len(block_members_global)
    block_values = np.asarray(block_values, dtype=float)
    block_weights = np.asarray(block_weights, dtype=float)

    if verbose:
        print(f"Stage 1 complete. Total blocks B={B}.")
        for b_idx, (mem, val, wt, g) in enumerate(zip(block_members_global, block_values, block_weights, block_group)):
            print(f"  B{b_idx} (from R{g}): members={mem}, av={val:.6g}, w={wt:.6g}")

    # --- Stage 2: build G_B (Hasse by construction). Implements Step 5.
    if verbose:
        print("Stage 2 — Assemble block DAG G_B (Hasse by construction).")
    G_B = nx.DiGraph(); G_B.add_nodes_from(range(B))

    # (Step 5.i) Add intra-group block Hasse edges.
    for gi, G_loc_g in enumerate(G_loc_list):
        if G_loc_g is not None:
            G_B.add_edges_from(G_loc_g.edges)
            if verbose and G_loc_g.number_of_edges() > 0:
                print(f"  Added intra-group edges from R{gi} (at most 20): {list(G_loc_g.edges)[:20]}")

    # (Step 5.ii) Add inter-group edges: for each Hasse(Q) cover i->j, add
    # edges from every max block of R_i to every min block of R_j.
    for i, j in H_Q.edges:
        max_i = group_max_global[i] or []
        min_j = group_min_global[j] or []
        for a in max_i:
            for b in min_j:
                G_B.add_edge(a, b)
                if verbose:
                    print(f"  Added cross edge (Q {i}->{j}): B{a} -> B{b}")

    # Debug-only runtime verification that G_B is already transitive-reduced.
    if debug and G_B.number_of_edges() > 0:
        GB_red = nx.transitive_reduction(G_B)
        if set(G_B.edges) != set(GB_red.edges):
            if verbose:
                print("WARNING: G_B differs from its transitive reduction.")
            raise AssertionError("Stage-2 block graph is not reduced; check max->min wiring and/or intra-group Hasse edges.")
        elif verbose:
            print("Verified: G_B equals its transitive reduction.")

    # (Step 6) Choose a block-level order, then run GPAV on the block DAG.
    if use_trend_following_blocks:
        block_vals_map = {b: float(block_values[b]) for b in G_B.nodes()}
        TB = trend_following_order(G_B, block_vals_map)
        if verbose:
            print(f"Block-level order : {TB[:20]}")
    else:
        TB = list(nx.topological_sort(G_B))
        if verbose:
            print(f"Block-level order (topological): {TB[:20]}")

    if verbose:
        print("Run GPAV on block DAG...")
    block_vals_map = {b: float(block_values[b]) for b in G_B.nodes()}
    block_wts_map  = {b: float(block_weights[b]) for b in G_B.nodes()}
    u_blocks, _, _ = gpav(block_vals_map, G_B, topo_order=TB, weights=block_wts_map)

    # (Step 7) Propagate block-level fits back to elements.
    u_final = np.empty_like(A_array)
    for b_idx, members in enumerate(block_members_global):
        u_final[members] = u_blocks[b_idx]

    if verbose:
        print("Final per-block values:")
        for b_idx, val in enumerate(u_blocks):
            print(f"  B{b_idx}: u={float(val):.6g}")
        print("Propagate back to original indices (first 20 shown):")
        preview = [(i, float(u_final[i])) for i in range(min(len(u_final), 20))]
        print(f"  {preview}")
        print("== Operadic / Factorized SB-GPAV: finished ==")

    if return_by_local_index:
        u_list = []
        for i, H_R in enumerate(H_R_list):
            offset = offs[i]
            u_i = {lbl: float(u_final[offset + int(lbl)]) for lbl in H_R.nodes()}
            u_list.append(u_i)
        return u_list


    return u_final


if __name__ == "__main__":
    # Minimal smoke tests / examples (unchanged algorithmically).
    #
    # Example 1: Small Q with 4 components R_i
    poset = hasse.PoSet.from_chains([0, 3], [1, 3], [1, 2])
    R_subposets = [
        hasse.PoSet.from_chains([0]),
        hasse.PoSet.from_chains([0], [1]),
        hasse.PoSet.from_chains([0, 1], [0, 2]),
        hasse.PoSet.from_chains([0], [1, 2]),
    ]
    Y = [2.0, 8.0, 2.1, 3.0, 4.0, 1.0, 2.0, 1.0, 4.0]
    print("=== Example 1: factorized SB-GPAV (verbose) ===")
    print(f"Input data: {Y}")
    print(f"Q hasse edges: {[ (x,y) for x in range(len(poset)) for y in poset.hasse.successors(x) ]}")
    Y_map = {i: float(y) for i, y in enumerate(Y)}
    u = OGPAV(
        Q=poset,
        R_subposets=R_subposets,
        A=Y_map,
        use_trend_following_first=True,
        use_trend_following_blocks=True,
        inputs_are_reduced=False,
        verbose=True,
    )
    print("Adjusted values (operadic/factorized):", u)

    # Example 2: Direct GPAV comparison on another small poset
    print("\n=== Example 2: plain GPAV on a small poset (verbose) ===")
    posetx = hasse.PoSet.from_chains([0, 6], [0, 7, 8], [1, 6], [1, 7], [1, 3, 4], [2, 6], [2, 7], [2, 3, 5])
    Y_map2 = {i: float(y) for i, y in enumerate(Y)}
    order = trend_following_order(posetx, Y_map2)
    print(f"Trend_following order: {order}")
    u_gpav, _, _ = gpav(Y_map2, posetx, order, verbose=True, name="GPAV(direct)")
    print("Adjusted values (gpav):", u_gpav)
