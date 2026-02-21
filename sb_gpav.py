"""
sb_gpav.py

SB-GPAV implementation accepting inputs (X, Y, L, f) as described:
  X: Dataset vectors (list or array)
  Y: Target values (real numbers)
  L: Topological order (list of indices of X)
  f: Optional comparison function f(a, b) -> bool (True if a <= b)

If f is not provided, defaults to coordinate-wise dominance:
  a <= b iff a[k] <= b[k] for all k.

References:
  Sysoev, Burdakov, Grimvall (2011) "A segmentation based algorithm..."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Callable, Any, Tuple, Dict, Union, Sequence

import numpy as np
import networkx as nx
from gpav import gpav_seg
from trend_following import _build_dag_incrementally
ArrayLike = Union[np.ndarray, Sequence[float]]

# ---------------------------------------------------------------------
# Comparator and Defaults
# ---------------------------------------------------------------------

def default_comparator(a: Any, b: Any) -> bool:
    """
    Default comparator: Coordinate-wise dominance (a <= b).
    Assumes a, b are array-like or comparable.
    """
    if a is b:
        return True
    
    # Numpy-aware check
    oa = np.asanyarray(a)
    ob = np.asanyarray(b)
    
    return bool(np.all(oa <= ob))




# ---------------------------------------------------------------------
# Helper: Build Hasse Diagram from Data (Induced on a subset)
# ---------------------------------------------------------------------

def _build_induced_hasse(
    indices: List[int],
    X: Any, 
    f: Callable[[Any, Any], bool],
    assume_component_wise: bool = False
) -> nx.DiGraph:
    """
    Build local Hasse diagram for a segment using memory-efficient construction.
    """
    def check_precedence(i: int, j: int) -> bool:
        # i, j are indices in X
        # Strict precedence: f(X[i], X[j]) AND i != j
        # Assuming topo order means if i precedes j in list, check f(i, j).
        # We enforce strict inequality for graph edges if values identical but distinct indices?
        # Actually standard GPAV works on strict poset.
        # If X[i] == X[j], they are equivalent. 
        # For simplicity, we use strict f(X[i], X[j]) is True.
        # If f(a,b) means a <= b, we check a <= b.
        # But for Hasse diagram, we usually want strict < relation or non-equivalence.
        # We rely on user f(a,b) meaning a <= b.
        # We add edge if a <= b. 
        # Incremental builder handles transitive reduction, so equal elements might form loops?
        # No, because loop `range(idx_j - 1, -1, -1)` enforces index order.
        return f(X[i], X[j])

    return _build_dag_incrementally(indices, check_precedence, assume_component_wise)


# ---------------------------------------------------------------------
# Helper: Block Extrema
# ---------------------------------------------------------------------

def _get_block_extrema(
    block_indices: List[int],
    X: Any,
    f: Callable[[Any, Any], bool]
) -> Tuple[List[int], List[int]]:
    """
    Find MIN(B) and MAX(B) within the block B using array comparator.
    
    MIN(B) = { u in B | not exists v in B s.t. v < u }
    MAX(B) = { u in B | not exists v in B s.t. u < v }
    """
    mins = []
    maxs = []
    
    for u in block_indices:
        # Check if u is minimal
        is_minimal = True
        val_u = X[u]
        for v in block_indices:
            if u == v: 
                continue
            val_v = X[v]
            if f(val_v, val_u):
                # v <= u. Check strictness v < u
                if not f(val_u, val_v):
                    is_minimal = False
                    break
        if is_minimal:
            mins.append(u)

        # Check if u is maximal
        is_maximal = True
        for v in block_indices:
            if u == v:
                continue
            val_v = X[v]
            if f(val_u, val_v):
                # u <= v. Check strictness u < v
                if not f(val_v, val_u):
                    is_maximal = False
                    break
        if is_maximal:
            maxs.append(u)
            
    return mins, maxs


# ---------------------------------------------------------------------
# Helper: Block Precedence (Min-Max Condition)
# ---------------------------------------------------------------------

def _block_precedes(
    minB_indices: List[int],
    maxA_indices: List[int],
    X: Any,
    f: Callable[[Any, Any], bool]
) -> bool:
    """
    Returns True if Block B precedes Block A (B -> A).
    Condition: exists b in min(B), a in max(A) s.t. b <= a.
    """
    for b in minB_indices:
        val_b = X[b]
        for a in maxA_indices:
            val_a = X[a]
            if f(val_b, val_a):
                return True
    return False


# ---------------------------------------------------------------------
# SB_GPAV Main
# ---------------------------------------------------------------------

@dataclass
class SegmentBlock:
    global_id: int
    members: List[int] # Indices in X
    value: float
    weight: float
    # These are computed locally:
    mins: List[int] 
    maxs: List[int]

def sb_gpav(
    X: np.ndarray,
    Y: ArrayLike,
    L: List[int],
    f: Optional[Callable[[Any, Any], bool]] = None,
    *,
    weights: Optional[ArrayLike] = None,
    n_segments: int = 10,
    assume_component_wise: bool = False,
    verbose: bool = False,
    debug: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
    """
    Segmentation-Based GPAV for array inputs with memory-efficient construction.
    
    Parameters
    ----------
    X : np.ndarray
        Dataset array of shape (N, d) where N is number of elements.
    """
    
    # 0. Setup
    N = len(Y)
    if weights is None:
        W = np.ones(N, dtype=float)
    else:
        W = np.asarray(weights, dtype=float)
        
    if f is None:
        f = default_comparator

    # 1. Segmentation
    if n_segments <= 0:
        n_segments = 1
    n_segments = min(n_segments, N)
    
    # Chunk L into s segments
    base = N // n_segments
    rem = N % n_segments
    
    segments = []
    start = 0
    for s_i in range(n_segments):
        size = base + (1 if s_i < rem else 0)
        seg_indices = L[start : start + size]
        segments.append(seg_indices)
        start += size
        
    # 2. Local GPAV per segment
    all_blocks: List[SegmentBlock] = []
    global_block_counter = 0
    
    
    for s_idx, seg_indices in enumerate(segments):
        if not seg_indices:
            continue
            
        # Build from array elements
        H_seg = _build_induced_hasse(seg_indices, X, f, assume_component_wise=assume_component_wise)
        
        Y_map = {idx: float(Y[idx]) for idx in seg_indices}
        W_map = {idx: float(W[idx]) for idx in seg_indices}
        
        _, seg_block_dicts, _ = gpav_seg(
            Y=Y_map,
            poset=H_seg,
            topo_order=seg_indices,
            weights=W_map,
            verbose=verbose,
            name=f"Seg{s_idx}"
        )
        
        # Convert dicts to SegmentBlock objects
        for b_dict in seg_block_dicts:
            members = b_dict['labels'] # list of indices
            # Compute extrema using array comparator
            mins, maxs = _get_block_extrema(members, X, f)
            
            blk = SegmentBlock(
                global_id=global_block_counter,
                members=members,
                value=b_dict['value'],
                weight=b_dict['weight'],
                mins=mins,
                maxs=maxs
            )
            all_blocks.append(blk)
            global_block_counter += 1
            
    # 3. Assembly: Build Block DAG (Memory Efficient)
    B_total = len(all_blocks)
    
    def check_block_precedence(i: int, j: int) -> bool:
        # i, j are global block IDs
        return _block_precedes(all_blocks[i].mins, all_blocks[j].maxs, X, f)
    
    # Use incremental builder for blocks too.
    # Assumes sequential block IDs are topological order (which they are by segment ordering).
    block_indices = list(range(B_total))
    G_blocks = _build_dag_incrementally(block_indices, check_block_precedence, assume_component_wise)
        
    # 4. Global GPAV on blocks
    Y_blocks = np.array([b.value for b in all_blocks])
    W_blocks = np.array([b.weight for b in all_blocks])
    
    block_topo = list(nx.topological_sort(G_blocks))
    
    u_blocks_fitted, _, _ = gpav_seg(
        Y=Y_blocks,
        poset=G_blocks,
        topo_order=block_topo,
        weights=W_blocks,
        verbose=verbose,
        name="GPAV_Blocks"
    )
    
    # 5. Propagate back
    u_final = np.zeros(N, dtype=float)
    for i in range(B_total):
        val = u_blocks_fitted[i]
        for member_idx in all_blocks[i].members:
            u_final[member_idx] = val
            
    if debug:
        return u_final, {
            "all_blocks": all_blocks,
            "G_blocks": G_blocks,
            "u_blocks": u_blocks_fitted
        }
        
    return u_final
