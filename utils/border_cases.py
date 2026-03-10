# -*- coding: utf-8 -*-
"""
border_cases.py

Handles extremely degenerate graph topological structures to massively accelerate OperadicGPAV.
It provides fast-paths for operations when the outer poset Q has no edges,
or when a local fiber R_i is known to be a disjoint union of points (antichain).
"""

from __future__ import annotations
import numpy as np
import networkx as nx
from typing import Any, Callable, List, Dict, Optional
import os
import pickle

from .gpav import gpav_seg
from .sb_gpav import _get_block_extrema

def handle_q_no_edges(
    m: int,
    R_datasets: Any,
    indices_list: List[List[int]],
    Y: np.ndarray,
    f_list: Optional[List[Callable]],
    f_global: Optional[Callable],
    segment_topo_orders: Optional[List[Optional[List[int]]]],
    use_trend_following_first: bool,
    assume_component_wise: bool,
    max_workers: int,
    verbose: bool
) -> np.ndarray:
    """
    Fast-path for when the outer poset Q has absolutely no edges (or is m=1).
    Because the fibers do not interact globally, Stage 2 is totally unnecessary.
    We just run Stage 1 concurrently and map the results straight to the output matrix u.
    """
    import concurrent.futures
    from OperadicGPAV import _process_fiber_task
    from .trend_following import default_comparator
    import tempfile
    
    u_final = np.zeros_like(Y, dtype=float)
    
    if verbose:
        print(f"Border Case Detected: Q has no edges. Skipping Stage 2. Processing {m} independent fibers...")

    # We still need temp_dir just because _process_fiber_task dumps blocks to disk
    # Alternatively we could rewrite _process_fiber_task to yield in-memory, but 
    # letting it dump and immediately reading it back is safe enough for now.
    temp_dir_obj = tempfile.TemporaryDirectory(prefix="ogpav_fastpath_")
    temp_dir_path = temp_dir_obj.name

    try:
        if isinstance(R_datasets, (list, tuple)) or hasattr(R_datasets, "__getitem__"):
             inputs = ((i, R_datasets[i], indices_list[i]) for i in range(m))
        else:
             inputs = zip(range(m), R_datasets, indices_list)

        def _get_comparator(i):
            if assume_component_wise:
                return default_comparator
                
            if f_list is not None:
                return f_list[i] if i < len(f_list) else None
            else:
                return f_global

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
                import warnings
                warnings.warn(
                    "One or more custom comparators cannot be pickled (e.g., lambda functions). "
                    "Forcing max_workers=1 (sequential execution) to prevent multiprocessing crashes.",
                    UserWarning
                )
                max_workers = 1

        if max_workers == 1:
            for i, X_i, idxs in inputs:
                f_i = _get_comparator(i)
                custom_order_i = segment_topo_orders[i] if segment_topo_orders and i < len(segment_topo_orders) else None
                _process_fiber_task(
                    i, X_i, idxs, Y, f_i, temp_dir_path, use_trend_following_first, custom_order_i, assume_component_wise
                )
                
                # Immediately map results back
                p_path = os.path.join(temp_dir_path, f"fiber_{i}.pkl")
                with open(p_path, "rb") as f_in:
                    blocks, _ = pickle.load(f_in)
                
                for b in blocks:
                    val = b['value']
                    for local_elem in b['labels']:
                        u_final[idxs[local_elem]] = val
        else:
            effective_workers = max_workers or os.cpu_count() or 4
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                inputs_iter = iter(inputs)
                
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
                    futures[fut] = (i, idxs)
                    
                while futures:
                    done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                    for fut in done:
                        i, idxs = futures.pop(fut)
                        fut.result() # trigger exceptions if any
                        
                        # Immediately map results back for completed fiber
                        p_path = os.path.join(temp_dir_path, f"fiber_{i}.pkl")
                        with open(p_path, "rb") as f_in:
                            blocks, _ = pickle.load(f_in)
                        
                        for b in blocks:
                            val = b['value']
                            for local_elem in b['labels']:
                                u_final[idxs[local_elem]] = val
                        
                        # Submit next
                        try:
                            ni, nX_i, nidxs = next(inputs_iter)
                            f_i = _get_comparator(ni)
                            custom_order_i = segment_topo_orders[ni] if segment_topo_orders and ni < len(segment_topo_orders) else None
                            new_fut = executor.submit(
                                _process_fiber_task,
                                ni, nX_i, nidxs, Y, f_i, temp_dir_path, use_trend_following_first, custom_order_i, assume_component_wise
                            )
                            futures[new_fut] = (ni, nidxs)
                        except StopIteration:
                            pass

        return u_final

    finally:
        temp_dir_obj.cleanup()

def package_local_antichain(
    i: int,
    X_i: Any,
    local_Y_indices: List[int],
    Y_snapshot: np.ndarray,
    temp_dir: str
) -> tuple[int, int, list[int], list[int]]:
    """
    Fast-path for a local fiber that is definitively an antichain (disjoint points).
    Bypasses GPAV evaluation, returning every node as an individual 1-size block.
    """
    n_i = len(X_i)
    
    blocks = []
    for local_idx in range(n_i):
        global_idx = local_Y_indices[local_idx]
        val = float(Y_snapshot[global_idx])
        # A 1-element block
        b = {
            'value': val,
            'weight': 1.0, # single point weight
            'labels': [local_idx],
            'elements': [local_idx]
        }
        blocks.append(b)
        
    G_loc = nx.DiGraph()
    G_loc.add_nodes_from(range(n_i))
    
    # Since G_loc has no edges, every single block is natively both a minimum and a maximum extrema.
    mins = list(range(n_i))
    maxs = list(range(n_i))
    
    p_path = os.path.join(temp_dir, f"fiber_{i}.pkl")
    with open(p_path, "wb") as f_out:
        pickle.dump((blocks, G_loc), f_out)
        
    return (i, n_i, mins, maxs)
