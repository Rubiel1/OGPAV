# -*- coding: utf-8 -*-
"""
OperadicGPAV.py

Implements the "Operadic" case of SB-GPAV algorithm for the 
lexicographic sum P = Q(R_1, ..., R_m).

This module provides a high-level interface `OperadicGPAV` that accepts:
  - Q: The outer poset (NetworkX DiGraph).
  - R_datasets: A list/iterator of datasets [X_1, ..., X_m].
  - Y: Global response vector.
  - f: Optional comparator function for X elements.
  - indices_list: Optional explicit mapping of global Y indices to local R_i datasets.

Optimized for:
1. Memory efficiency: Disk-based caching of intermediate block results.
2. Performance: Parallel execution of local GPAV steps (Stage 1).
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Hashable, Union, Any, Callable, Sequence, Iterable
import numpy as np
import networkx as nx
import warnings
import gc
import pickle
import tempfile
import os
import shutil
import concurrent.futures

# Use optimized components
from utils.trend_following import trend_following_order, _build_dag_incrementally, default_comparator
from utils.gpav import gpav_seg
from utils.sb_gpav import _get_block_extrema, _block_precedes
import utils.geometric_sb_dataset as geometric_sb_dataset 

# --- Aliases ---
NodeLabel = Hashable
LocalIndex = int
GlobalIndex = int
BlockId = int

# ---------------------------------------------------------------------
# External Helper: Mapping Logic
# ---------------------------------------------------------------------
def create_lexicographic_mapping(R_datasets: Any) -> List[List[GlobalIndex]]:
    """
    Generates a default mapping where Y is assumed to be the concatenation
    of values corresponding to R_datasets in order.
    
    If R_datasets or any of its fibers is a generator, it must implement `get_fiber_lengths()`.
    Otherwise, we require all elements to be sized collections (e.g. List, Tuple, np.ndarray) so
    we can evaluate lengths without consuming them.
    """
    if hasattr(R_datasets, "get_fiber_lengths"):
        lengths = R_datasets.get_fiber_lengths()
    else:
        # Enforce that R_datasets is not a single-pass generator.
        if not hasattr(R_datasets, "__len__") and not hasattr(R_datasets, "__getitem__"):
            raise TypeError(
                "When providing an iterator/generator for R_datasets without an explicit "
                "`indices_list`, the parent iterator must implement a `get_fiber_lengths()` method "
                "to prevent memory exhaustion. Otherwise, provide a list/tuple/array."
            )
        
        lengths = []
        for x in R_datasets:
            if not hasattr(x, "__len__"):
                raise TypeError(
                    "Elements of R_datasets are generators/iterators. You must provide a "
                    "`get_fiber_lengths()` method on `R_datasets` so OperadicGPAV knows "
                    "their dimensions ahead of time without consuming them."
                )
            lengths.append(len(x))

    indices_list = []
    current = 0
    for length in lengths:
        indices = list(range(current, current + length))
        indices_list.append(indices)
        current += length
    return indices_list


# ---------------------------------------------------------------------
# Worker Function (Must be top-level for multiprocessing)
# ---------------------------------------------------------------------

def _process_fiber_task(
    i: int,
    X_i: Any,
    local_Y_indices: List[int],
    Y_snapshot: np.ndarray, # Copy or shared mem
    f: Callable,
    temp_dir: str,
    use_trend_following: bool,
    custom_topo_order: Optional[List[int]] = None,
    assume_component_wise: bool = False
) -> Tuple[int, int, List[BlockId], List[BlockId]]:
    """
    Worker task to process a single fiber and save result to disk.
    Returns: (index, block_count, mins, maxs)
    """
    try:
        # 1. Antichain Bypass
        if f is None:
            from utils.border_cases import package_local_antichain
            return package_local_antichain(i, X_i, local_Y_indices, Y_snapshot, temp_dir)

        # 2. Build Hasse from array elements
        n_i = len(X_i)
        
        # Sort heuristic for better incremental construction
        if assume_component_wise:
            sums = np.sum(X_i, axis=1)
            topo_indices = np.argsort(sums).tolist()
        else:
            topo_indices = list(range(n_i))
             
        def check_precedence(a_idx, b_idx):
            return f(X_i[a_idx], X_i[b_idx])
            
        H_R = _build_dag_incrementally(topo_indices, check_precedence, assume_component_wise)
        
        # 3. Dynamic Antichain Bypass
        if H_R.number_of_edges() == 0:
            from utils.border_cases import package_local_antichain
            return package_local_antichain(i, X_i, local_Y_indices, Y_snapshot, temp_dir)

        # 4. Prepare A_i (sequential node indices 0, 1, 2, ...)
        A_i = {j: float(Y_snapshot[glob_idx]) 
               for j, glob_idx in enumerate(local_Y_indices)}
        
        # 3. Run Local GPAV
        if custom_topo_order is not None:
            topo = custom_topo_order
        elif use_trend_following:
            topo = trend_following_order(G=H_R, Y=A_i, stable_tiebreak=True)
        else:
            topo = list(nx.topological_sort(H_R))
            
        _, blocks, _ = gpav_seg(
            Y=A_i, 
            poset=H_R, 
            topo_order=topo
        )
        
        # 4. Build Block DAG using comparator (Algorithm 3 / Theorem 4)
        block_extrema = []
        for b in blocks:
            mins_b, maxs_b = _get_block_extrema(b['labels'], X_i, f)
            block_extrema.append((mins_b, maxs_b))
        
        def check_block_prec(bi, bj):
            return _block_precedes(block_extrema[bi][0], block_extrema[bj][1], X_i, f)
        
        # We can safely use assume_component_wise=False now to do full O(N^2) 
        # cycle verification because the true blocks are guaranteed structurally sound!
        G_loc = _build_dag_incrementally(list(range(len(blocks))), check_block_prec)
        
        mins = [n for n in G_loc.nodes() if G_loc.in_degree(n) == 0]
        maxs = [n for n in G_loc.nodes() if G_loc.out_degree(n) == 0]
        
        # 5. Save to Disk
        with open(os.path.join(temp_dir, f"fiber_{i}.pkl"), "wb") as f_out:
            pickle.dump((blocks, G_loc), f_out)
            
        return (i, len(blocks), mins, maxs)
        
    except Exception as e:
        # Re-raise to make debugging easier
        print(f"Error processing fiber {i}: {e}")
        raise


# ---------------------------------------------------------------------
# Main Interface: OperadicGPAV
# ---------------------------------------------------------------------

def OperadicGPAV(
    Q: Union[nx.DiGraph, Any],  # Can be nx.DiGraph or hasse.PoSet
    R_datasets: Union[List[Any], Iterable[Any]],
    Y: Union[np.ndarray, Sequence[float], Dict[int, float]],
    f: Optional[Union[Callable[[Any, Any], bool], List[Callable[[Any, Any], bool]]]] = None,
    indices_list: Optional[List[List[GlobalIndex]]] = None,
    segment_topo_orders: Optional[List[Optional[List[int]]]] = None,
    *,
    use_trend_following_first: bool = True,
    use_trend_following_blocks: bool = True,
    assume_component_wise: bool = False,
    max_workers: Optional[int] = None,
    verbose: bool = False,
    debug: bool = False,
    temp_dir: Optional[str] = None
) -> np.ndarray:
    """
    Optimized SB-GPAV for input of the form P = Q(R_1, ..., R_m).
    
    Parameters
    ----------
    Q : nx.DiGraph or hasse.PoSet
        Outer poset with m nodes (one per fiber R_i).
        Can be a NetworkX DiGraph or a hasse.PoSet object.
    
    R_datasets : List[array-like] or Iterable
        List of m datasets [R_0, ..., R_{m-1}].
        Each R_i is an array-like of vectors.
    
    Y : array-like
        Global response vector of length N = sum(len(R_i)).
    
    f : Callable or List[Callable], optional
        Comparator function(s) for partial order on R_i elements.
        
        - If single function: f(a, b) -> bool, used for ALL R_i
        - If list of functions: [f_0, ..., f_{m-1}], one per R_i
        - If None: defaults to coordinate-wise comparison for all R_i
        
        Each f_i(a, b) should return True if a <= b in the partial order.
    
    indices_list : List[List[int]], optional
        Explicit mapping of Y indices to R_i datasets.
        If None, assumes lexicographic concatenation.
    
    segment_topo_orders : List[Optional[List[int]]], optional
        Custom topological orders for each R_i fiber.
        If provided, must be a list of length m where each element is either:
        - A list of node indices (custom order for that fiber)
        - None (use default trend_following or topological sort)
        If None, all fibers use default ordering.
        
    assume_component_wise : bool, optional
        If True, assumes R_datasets elements are given in a valid topological order 
        respecting `f`, enabling a fast graph building path. If False (default), 
        verifies all O(N^2) pairs for safety.

    Returns
    -------
    u : np.ndarray
        Fitted isotonic values aligned with Y.
    """
    
    # --- 1. Validate Q Nodes ---
    # Count fibers from R_datasets safely
    if hasattr(R_datasets, "get_fiber_lengths"):
        m = len(R_datasets.get_fiber_lengths())
    elif hasattr(R_datasets, "__len__"):
        m = len(R_datasets)
    else:
        raise TypeError(
            "R_datasets must either be a sequence (like a list/array) "
            "or an object implementing `get_fiber_lengths()`."
        )

    expected_nodes = set(range(m))
    actual_nodes = set(Q.nodes())
    
    if m != len(actual_nodes):
        raise ValueError(
            f"The number of datasets ({m}) must match the number of Q nodes ({len(actual_nodes)})."
        )
    
    if actual_nodes != expected_nodes:
        raise ValueError(
            f"Q nodes must be {{0, 1, ..., {m-1}}} for {m} fibers. "
            f"Got: {sorted(actual_nodes)}"
        )
    
    # --- 2. Setup ---
    
    # Q must be a NetworkX DiGraph
    if not isinstance(Q, nx.DiGraph):
        raise TypeError("Q must be a NetworkX DiGraph")
    
    # Y must be array
    Y = np.asarray(Y, dtype=float)
    
    # Handle f: single function or list of functions
    if f is None:
        # Default: coordinate-wise comparison for all R_i
        f_list = None
        f_global = None  # Ensure this is explicitly defined
    elif callable(f):
        # Single function for all R_i
        f_list = None
        f_global = f
    elif isinstance(f, (list, tuple)):
        # List of functions, one per R_i
        f_list = list(f)
        f_global = None
    else:
        raise TypeError("f must be a callable, a list of callables, or None")

    if assume_component_wise and f is not None:
        # We need to ensure that the user doesn't pass assume_component_wise=True with a custom comparator, 
        # as assume_component_wise assumes default geometric properties for its incremental graph construction.
        if isinstance(f, (list, tuple)) and all(comp is None for comp in f):
            pass # Exception: All elements in f are None, so it safely defaults entirely to geometric
        else:
            raise ValueError(
                "Cannot set `assume_component_wise=True` when a custom comparator `f` is provided. "
                "The `assume_component_wise` flag uses a sum-of-coordinates heuristic that is only guaranteed "
                "to be correct for the default geometric component-wise comparison."
            )
    
    if indices_list is None:# Warning is best.
        warnings.warn(
            "No indices_list provided. We assume R_i corresponds to node i in Q,"+
            " and that the first N_0 elements of Y correspond to R_0,"+
            " the next N_1 to R_1, etc.",
            UserWarning
        )

        indices_list = create_lexicographic_mapping(R_datasets)
    
    m = len(indices_list)
    Y = np.asarray(Y)

    # Prepare Temp Directory
    if temp_dir is None:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="ogpav_intermediary_")
        temp_dir_path = temp_dir_obj.name
    else:
        temp_dir_obj = None
        temp_dir_path = temp_dir
        os.makedirs(temp_dir_path, exist_ok=True)

    try:
        if Q.number_of_edges() == 0:
            from utils.border_cases import handle_q_no_edges
            return handle_q_no_edges(
                m=m,
                R_datasets=R_datasets,
                indices_list=indices_list,
                Y=Y,
                f_list=f_list,
                f_global=f_global,
                segment_topo_orders=segment_topo_orders,
                use_trend_following_first=use_trend_following_first,
                assume_component_wise=assume_component_wise,
                max_workers=max_workers,
                verbose=verbose
            )

        # --- 2. STAGE 1: LOCAL PROCESSING (Parallel) ---
        if verbose:
            print(f"Stage 1: Processing {m} fibers (max_workers={max_workers})...")
            
        group_min_blocks = [[] for _ in range(m)]
        group_max_blocks = [[] for _ in range(m)]
        block_counts = [0] * m
        
        # Prepare inputs lazily: only pull R_i when we're ready to process it.
        # For indexable datasets (list, LazyRiemannianDataset), use __getitem__.
        # For raw iterators/generators, zip with indices.
        if isinstance(R_datasets, (list, tuple)) or hasattr(R_datasets, "__getitem__"):
             inputs = ((i, R_datasets[i], indices_list[i]) for i in range(m))
        else:
             inputs = zip(range(m), R_datasets, indices_list)

        # Helper: select comparator for fiber i
        def _get_comparator(i):
            if assume_component_wise:
                return default_comparator
                
            if f_list is not None:
                return f_list[i] if i < len(f_list) else None
            else:
                return f_global

        # Pickling Check: Multiprocessing crashes if it tries to pickle a lambda or nested function
        if max_workers != 1:
            can_pickle = True
            def _is_picklable(obj):
                if obj is None: return True
                try:
                    pickle.dumps(obj)
                    return True
                except Exception:
                    return False

            if f_global is not None and not _is_picklable(f_global):
                can_pickle = False
            if f_list is not None:
                for func in f_list:
                    if not _is_picklable(func):
                        can_pickle = False
                        break

            if not can_pickle:
                warnings.warn(
                    "One or more custom comparators cannot be pickled (e.g., lambda functions). "
                    "Forcing max_workers=1 (sequential execution) to prevent multiprocessing crashes.",
                    UserWarning
                )
                max_workers = 1

        # Execution Strategy:
        # - max_workers=1: sequential (no pickling overhead)
        # - max_workers>1: sliding window parallel — only max_workers fibers
        #   are loaded into memory at a time. When a worker finishes, the next
        #   R_i is pulled from the iterator and submitted.

        if max_workers == 1:
            # Sequential: process one fiber at a time
            for i, X_i, idxs in inputs:
                f_i = _get_comparator(i)
                custom_order_i = segment_topo_orders[i] if segment_topo_orders and i < len(segment_topo_orders) else None
                
                _, count, mins, maxs = _process_fiber_task(
                    i, X_i, idxs, Y, f_i, temp_dir_path, use_trend_following_first, custom_order_i, assume_component_wise
                )
                block_counts[i] = count
                group_min_blocks[i] = mins
                group_max_blocks[i] = maxs
        else:
            # Parallel with sliding window: only max_workers fibers in memory
            effective_workers = max_workers or os.cpu_count() or 4
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                inputs_iter = iter(inputs)
                
                # Fill initial window (submit up to effective_workers tasks)
                for _ in range(min(effective_workers, m)):
                    try:
                        i, X_i, idxs = next(inputs_iter)
                    except StopIteration:
                        break
                    f_i = _get_comparator(i)
                    custom_order_i = segment_topo_orders[i] if segment_topo_orders and i < len(segment_topo_orders) else None
                    fut = executor.submit(
                        _process_fiber_task,
                        i, X_i, idxs, Y, f_i, temp_dir_path, use_trend_following_first, custom_order_i, assume_component_wise
                    )
                    futures[fut] = i
                
                # Process completed tasks and submit next fibers
                while futures:
                    done, _ = concurrent.futures.wait(
                        futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for fut in done:
                        i = futures.pop(fut)
                        try:
                            _, count, mins, maxs = fut.result()
                            block_counts[i] = count
                            group_min_blocks[i] = mins
                            group_max_blocks[i] = maxs
                            if verbose and i % 10 == 0:
                                print(f"  Fiber {i} done.")
                        except Exception as exc:
                            print(f"Fiber {i} generated an exception: {exc}")
                            raise exc
                        
                        # Submit next fiber from iterator (if any)
                        try:
                            ni, nX_i, nidxs = next(inputs_iter)
                            f_i = _get_comparator(ni)
                            custom_order_i = segment_topo_orders[ni] if segment_topo_orders and ni < len(segment_topo_orders) else None
                            new_fut = executor.submit(
                                _process_fiber_task,
                                ni, nX_i, nidxs, Y, f_i, temp_dir_path, use_trend_following_first, custom_order_i, assume_component_wise
                            )
                            futures[new_fut] = ni
                        except StopIteration:
                            pass  # No more fibers to submit

        # --- 3. STAGE 2: GLOBAL ASSEMBLY ---
        
        # Reduce Q to Hasse diagram if needed
        if Q.number_of_edges() > 0:
            H_Q = nx.transitive_reduction(Q)
        else:
            H_Q = nx.DiGraph()
            H_Q.add_nodes_from(Q.nodes())
        # Calculate offsets
        offsets = [0] * (m + 1)
        for i in range(m):
            offsets[i+1] = offsets[i] + block_counts[i]
        total_blocks = offsets[-1]
        
        if verbose:
            print(f"Stage 2: Building Global Block Graph G_B with {total_blocks} blocks...")

        G_B = nx.DiGraph()
        G_B.add_nodes_from(range(total_blocks))
        
        Y_blocks = np.zeros(total_blocks)
        W_blocks = np.zeros(total_blocks)
        
        # Pass 1: Intra edges (Load sequentially)
        for i in range(m):
            off = offsets[i]
            p_path = os.path.join(temp_dir_path, f"fiber_{i}.pkl")
            if not os.path.exists(p_path):
                 raise RuntimeError(f"Missing intermediate file for fiber {i}")
                 
            with open(p_path, "rb") as f_in:
                 blocks, G_loc = pickle.load(f_in)
            
            for u, v in G_loc.edges():
                G_B.add_edge(u + off, v + off)
                
            for b_idx, b in enumerate(blocks):
                Y_blocks[off + b_idx] = b['value']
                W_blocks[off + b_idx] = b['weight']
                
            del blocks, G_loc
                
        # Pass 2: Inter edges (from Q)
        for i, j in H_Q.edges():
            for u_loc in group_max_blocks[i]:
                for v_loc in group_min_blocks[j]:
                    u_glob = u_loc + offsets[i]
                    v_glob = v_loc + offsets[j]
                    G_B.add_edge(u_glob, v_glob)
                    
        # 4. Global GPAV
        if verbose:
            print("Running Global GPAV on G_B...")
            

        if use_trend_following_blocks:
            Y_map = {i: Y_blocks[i] for i in range(total_blocks)}
            topo_B = trend_following_order(G=G_B, Y=Y_map)
        else:
            topo_B = list(nx.topological_sort(G_B))
            
        u_blocks, _, _ = gpav_seg(
            Y=Y_blocks, 
            poset=G_B, 
            topo_order=topo_B, 
            weights=W_blocks
        )
        
        # --- 5. SCATTER BACK ---
        if verbose:
            print("Scattering results back to u_final...")
            
        u_final_global = np.zeros_like(Y, dtype=float)
        
        for i in range(m):
            off = offsets[i]
            mapping = indices_list[i]
            
            with open(os.path.join(temp_dir_path, f"fiber_{i}.pkl"), "rb") as f_in:
                 blocks, _ = pickle.load(f_in)
            
            for b_idx, b in enumerate(blocks):
                val = u_blocks[off + b_idx]
                for local_elem in b['elements']:
                     try:
                        global_idx = mapping[local_elem]
                        u_final_global[global_idx] = val
                     except IndexError as e:
                        # Index mismatch indicates a bug in mapping logic
                        print(f"ERROR: IndexError in fiber {i}, block {b_idx}, local_elem {local_elem}")
                        print(f"  Mapping length: {len(mapping)}, local_elem: {local_elem}")
                        print(f"  Block elements: {b['elements']}")
                        raise RuntimeError(
                            f"Index mismatch in fiber {i}: local element {local_elem} "
                            f"not found in mapping (length {len(mapping)}). This indicates a bug."
                        ) from e
                        
            del blocks

        return u_final_global

    finally:
        if temp_dir_obj:
            temp_dir_obj.cleanup()
