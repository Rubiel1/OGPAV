# -*- coding: utf-8 -*-
"""
experiment_compare_ogpav_vs_sb.py

Compare OperadicGPAV vs segmentation-based GPAV on the SAME synthetic dataset,
without constructing the full data poset in advance.

What this script does
---------------------
1. Generate lazy synthetic fibers using generate_dataset_lazy(...)
2. Use the generated outer poset Q_hasse directly for OperadicGPAV
3. Build the full data matrix X only once (needed by sb_gpav), but do NOT build
   the global Hasse diagram / adjacency matrix
4. Create noiseless truth y_true and noisy observations y_noisy
5. Run:
      - OperadicGPAV on (Q, R_datasets, y_noisy)
      - sb_gpav on (X, y_noisy, L)
6. Compare both fitted vectors against y_true

Important
---------
- This script uses assume_component_wise=True for both methods.
- That means the order relation is coordinate-wise <=.
- No full poset is materialized from X.
- For sb_gpav, we only need X and one topological order L.

Recommended usage
-----------------
python experiment_compare_ogpav_vs_sb.py
"""

from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
import os
import time
import math
import json
import traceback
import tracemalloc
from typing import Dict, List, Tuple, Optional

import numpy as np
import networkx as nx
import psutil as _psutil
_proc = _psutil.Process(os.getpid())

from OperadicGPAV import OperadicGPAV
from utils.geometric_sb_dataset import (
    generate_standard,
    make_dataset_params,
    MODEL_FUNCS,
)
from utils.sb_gpav import sb_gpav, _build_induced_hasse, default_comparator
from utils.gpav import gpav_op

import csv
import gc
import os
import matplotlib.pyplot as plt

# ============================================================
# Helpers
# ============================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return standard regression error metrics."""
    err = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    max_abs = float(np.max(np.abs(err)))
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "max_abs": max_abs,
    }


def make_noise(
    n: int,
    *,
    noise: str = "normal",
    noise_scale: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate additive noise."""
    rg = np.random.default_rng(seed)

    if noise == "none":
        return np.zeros(n, dtype=float)
    if noise == "normal":
        return rg.normal(0.0, noise_scale, size=n).astype(float)
    if noise == "laplace":
        return rg.laplace(0.0, noise_scale, size=n).astype(float)

    raise ValueError("noise must be one of {'none', 'normal', 'laplace'}")


def build_full_X_from_lazy(R_datasets) -> Tuple[np.ndarray, List[int], List[List[int]]]:
    """
    Build the full X by loading fibers one-by-one, without constructing a global poset.

    Returns
    -------
    X : np.ndarray, shape (N, d)
        Full stacked dataset.
    lengths : List[int]
        Fiber sizes.
    indices_list : List[List[int]]
        Mapping from fiber-local indices to global indices in X / Y.
    """
    lengths = R_datasets.get_fiber_lengths()
    m = len(lengths)
    total_n = int(sum(lengths))

    # Find dimensionality from first non-empty fiber
    d = None
    for i in range(m):
        Ri = np.asarray(R_datasets[i], dtype=float)
        if Ri.ndim != 2:
            raise ValueError(f"Fiber {i} must be a 2D array, got shape {Ri.shape}")
        if Ri.shape[0] > 0:
            d = Ri.shape[1]
            break
    if d is None:
        raise ValueError("All fibers are empty. Cannot build X.")

    X = np.empty((total_n, d), dtype=float)

    indices_list: List[List[int]] = []
    start = 0
    for i, ni in enumerate(lengths):
        stop = start + ni
        Ri = np.asarray(R_datasets[i], dtype=float)
        if Ri.shape[0] != ni:
            raise RuntimeError(
                f"Fiber length mismatch at fiber {i}: "
                f"metadata says {ni}, actual loaded size is {Ri.shape[0]}"
            )
        if ni > 0:
            X[start:stop, :] = Ri
        indices_list.append(list(range(start, stop)))
        start = stop

    return X, lengths, indices_list


def make_y_true_from_fibers(
    R_datasets,
    *,
    model: str = "nonlinear",
) -> np.ndarray:
    """
    Build noiseless ground truth y_true by applying the chosen model fiber-by-fiber.
    """
    if model not in MODEL_FUNCS:
        raise ValueError(f"Unknown model {model!r}. Available: {sorted(MODEL_FUNCS)}")

    f_model = MODEL_FUNCS[model]
    parts = []

    for i in range(len(R_datasets)):
        Ri = np.asarray(R_datasets[i], dtype=float)
        yi = f_model(Ri).astype(float)
        parts.append(yi)

    return np.concatenate(parts, axis=0)


def topological_order_from_coordinates(X: np.ndarray) -> List[int]:
    """
    Build a simple topological order consistent with coordinate-wise monotonicity
    using sum of coordinates.

    If x <= y component-wise, then sum(x) <= sum(y), so sorting by coordinate sum
    is a valid topological order up to ties.
    """
    sums = np.sum(X, axis=1)
    return list(np.argsort(sums, kind="mergesort"))


def ensure_q_nodes_match_num_fibers(Q: nx.DiGraph, m: int) -> nx.DiGraph:
    """
    Make sure Q uses nodes {0,1,...,m-1}. If needed, relabel.
    """
    q_nodes = list(Q.nodes())
    expected = set(range(m))
    actual = set(q_nodes)

    if actual == expected:
        return Q

    if len(actual) != m:
        raise ValueError(
            f"Q has {len(actual)} nodes but there are {m} fibers. "
            f"Q nodes: {sorted(actual)}"
        )

    # Deterministic relabeling by sorted order
    mapping = {old: new for new, old in enumerate(sorted(q_nodes))}
    return nx.relabel_nodes(Q, mapping, copy=True)

def topological_order_from_lazy_fibers(R_datasets):
    sums = []
    index = 0

    for i in range(len(R_datasets)):
        Ri = np.asarray(R_datasets[i], dtype=float)

        for row in Ri:
            sums.append((np.sum(row), index))
            index += 1

    sums.sort(key=lambda x: x[0])
    return [idx for _, idx in sums]
# ============================================================
# One trial
# ============================================================

def run_one_trial(
    *,
    q: int = 50,
    fiber_count_dist: str = "poisson",
    model: str = "nonlinear",
    noise: str = "normal",
    noise_scale: float = 1.0,
    data_seed: int = 0,
    noise_seed: int = 1,
    n_segments: Optional[int] = None,
    max_workers: Optional[int] = None,
    use_trend_following_first: bool = True,
    use_trend_following_blocks: bool = True,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Run one comparison trial.
    """
    params = make_dataset_params(q)
    nQ    = params["nQ"]
    avg_R = params["avg_R"]
    try:
        # --------------------------------------------------------
        # 1. Generate lazy dataset
        # --------------------------------------------------------
        if verbose:
            print("=" * 80)
            print(f"Generating lazy dataset: q={q}, nQ={nQ}, avg_R={avg_R}, seed={data_seed}")

        data = generate_standard(
            q,
            seed=data_seed,
            fiber_count_dist=fiber_count_dist,
            lazy=True,
        )


        R_datasets = data["R_points_list"]
        Q_raw = data["Q_hasse"]
        if not isinstance(Q_raw, nx.DiGraph):
            raise TypeError("Expected Q_hasse to be a networkx.DiGraph")

        m = len(R_datasets)
        Q = ensure_q_nodes_match_num_fibers(Q_raw, m)

        print("Q edges:", list(Q.edges()))  
        print("Number of Q edges:", Q.number_of_edges())
        print("fiber lengths:", R_datasets.get_fiber_lengths())

        # --------------------------------------------------------
        # 1.5. Build full X, Y, and Comparators
        # --------------------------------------------------------
        lengths = R_datasets.get_fiber_lengths()
        total_n = int(sum(lengths))
        if verbose:
            print(f"Number of fibers: {m}")
            print(f"Total sample size N: {total_n}")

        t0_build = time.perf_counter()
        L = topological_order_from_lazy_fibers(R_datasets)
        X, lengths_check, indices_list = build_full_X_from_lazy(R_datasets)
        t_build_X = time.perf_counter() - t0_build

        y_true = make_y_true_from_fibers(R_datasets, model=model)
        eps = make_noise(
            total_n,
            noise=noise,
            noise_scale=noise_scale,
            seed=noise_seed,
        )
        y_noisy = y_true + eps
        baseline_metrics = compute_metrics(y_true, y_noisy)

        # Build Custom Comparator for SB-GPAV and PGPAV
        # To prove exactness, we must constrain the full array-based algorithms
        # to the exact macroscopic topological shape that OperadicGPAV operates on.
        Q_tc = nx.transitive_closure(Q)
        
        # We need `idx_to_fiber` map, but we haven't built indices_list yet.
        # Let's derive it directly from fiber lengths.
        lengths_for_map = R_datasets.get_fiber_lengths()
        idx_to_fiber = {}
        curr_idx = 0
        for fib_id, n_points in enumerate(lengths_for_map):
            for _ in range(n_points):
                idx_to_fiber[curr_idx] = fib_id
                curr_idx += 1

        # We must identify the min/max elements for each fiber 
        # based on simple component-wise dominance internal to that fiber.
        def _get_fiber_mins_maxs(fiber_indices: List[int]) -> Tuple[List[int], List[int]]:
            mins, maxs = [], []
            for u in fiber_indices:
                val_u = X[u]
                is_min = True
                is_max = True
                for v in fiber_indices:
                    if u == v: continue
                    val_v = X[v]
                    if np.all(val_v <= val_u) and not np.all(val_u <= val_v):
                        is_min = False
                    if np.all(val_u <= val_v) and not np.all(val_v <= val_u):
                        is_max = False
                if is_min: mins.append(u)
                if is_max: maxs.append(u)
            return mins, maxs

        fiber_mins = {}
        fiber_maxs = {}
        for f_id, fib_indices in enumerate(indices_list):
            fiber_mins[f_id], fiber_maxs[f_id] = _get_fiber_mins_maxs(fib_indices)

        def check_ogpav_precedence(i: int, j: int, X_ref: np.ndarray) -> bool:
            f_i = idx_to_fiber[i]
            f_j = idx_to_fiber[j]
            # Same fiber
            if f_i == f_j: 
                return bool(np.all(X_ref[i] <= X_ref[j]))
            
            # Different fibers: OperadicGPAV connects MAX(f_i) -> MIN(f_j).
            # So `i` only precedes `j` if there's a path i -> max_i -> ... -> min_j -> j.
            # This requires: i <= max_i AND min_j <= j internally, plus a path in Q.
            if f_i in Q_tc and f_j in Q_tc[f_i]:
                # OperadicGPAV adds edges from ALL maxs of parent to ALL mins of child.
                # So if `i` is less than ANY max of its fiber, and `j` is greater than ANY min of its fiber,
                # there is a valid path.
                
                # Check i <= *some* max of its own fiber
                i_can_reach_max = False
                for m_i in fiber_maxs[f_i]:
                    if np.all(X_ref[i] <= X_ref[m_i]):
                        i_can_reach_max = True
                        break
                        
                # Check *some* min of j's fiber <= j
                min_can_reach_j = False
                for m_j in fiber_mins[f_j]:
                    if np.all(X_ref[m_j] <= X_ref[j]):
                        min_can_reach_j = True
                        break
                        
                return bool(i_can_reach_max and min_can_reach_j)
                
            return False


        print("len(y_true):", len(y_true))
        print("len(y_noisy):", len(y_noisy))

        # --------------------------------------------------------
        # 2. Run OperadicGPAV directly on lazy fibers
        # --------------------------------------------------------
        if verbose:
            print("\nRunning OperadicGPAV...")

        tracemalloc.start()
        rss_before_og = _proc.memory_info().rss
        t0 = time.perf_counter()
        u_ogpav = OperadicGPAV(
            Q=Q,
            R_datasets=R_datasets,
            Y=y_noisy,
            indices_list=None,  # let OperadicGPAV infer lexicographic mapping
            segment_topo_orders=None,
            use_trend_following_first=use_trend_following_first,
            use_trend_following_blocks=use_trend_following_blocks,
            assume_component_wise=True,
            max_workers=max_workers,
            verbose=verbose,
            debug=False,
            temp_dir=None,
        )
        t_ogpav = time.perf_counter() - t0
        _, peak_og = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        rss_after_og = _proc.memory_info().rss
        mem_og_mb = peak_og / 1024 / 1024
        rss_og_mb = (rss_after_og - rss_before_og) / 1024 / 1024

        obj_ogpav = float(np.sum((np.asarray(u_ogpav, dtype=float) - np.asarray(y_noisy, dtype=float)) ** 2))

        ogpav_metrics = compute_metrics(y_true, u_ogpav)

        if lengths != lengths_check:
            raise RuntimeError("Fiber lengths changed unexpectedly while building X")

        # --------------------------------------------------------
        # 3. Setup and Run sb_gpav
        # --------------------------------------------------------
        # Choose number of segments if not supplied
        # Paper suggests segment size around 2000; translate that into n_segments.
        if n_segments is None:
            target_segment_size = 2000
            n_segments = max(1, math.ceil(total_n / target_segment_size))

        if verbose:
            print(f"sb_gpav will use n_segments={n_segments}")

        # --------------------------------------------------------
        # 5. Run sb_gpav on the same X and same y_noisy
        # --------------------------------------------------------
        if verbose:
            print("\nRunning sb_gpav...")

        tracemalloc.start()
        rss_before_sb = _proc.memory_info().rss
        t0 = time.perf_counter()
        u_sb = sb_gpav(
            X=X,
            Y=y_noisy,
            L=L,
            f=None,  # ignore value comparator
            weights=None,
            n_segments=n_segments,
            assume_component_wise=True,
            verbose=False,
            debug=False,
            f_idx=lambda i, j: check_ogpav_precedence(i, j, X),  # use index-aware comparator
        )
        t_sb = time.perf_counter() - t0
        _, peak_sb = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        rss_after_sb = _proc.memory_info().rss
        mem_sb_mb = peak_sb / 1024 / 1024
        rss_sb_mb = (rss_after_sb - rss_before_sb) / 1024 / 1024
        
        obj_sb = float(np.sum((np.asarray(u_sb, dtype=float) - np.asarray(y_noisy, dtype=float)) ** 2))

        sb_metrics = compute_metrics(y_true, u_sb)

        # --------------------------------------------------------
        # 6. Run Original PGPAV IF N <= 15000 (q <= 100) to avoid OOM
        # --------------------------------------------------------
        if q <= 100:
            if verbose:
                print("\nBuilding Hasse and running PGPAV for dev comparison...")
            t0 = time.perf_counter()
            
            # Use incremental builder with our custom checker
            from utils.trend_following import _build_dag_incrementally
            
            # Lambda wraps our function since _build_dag expects (i,j)
            def _pgpav_checker(i: int, j: int) -> bool:
                 return check_ogpav_precedence(i, j, X)
                 
            H_full_induced = _build_dag_incrementally(L, _pgpav_checker, assume_component_wise=True)

            # PGPAV expects Y to be aligned with list(H_full_induced.nodes()). 
            # Since the nodes were added in L order, passing a flat array scrambles the data.
            # We MUST pass Y as a dictionary mapping node_idx -> value.
            y_noisy_dict = {i: float(y_noisy[i]) for i in range(total_n)}

            u_pgpav_raw, _, _, _ = gpav_op(
                Y=y_noisy_dict,
                poset=H_full_induced,
                topo_order=L,
                weights=None,
                verbose=False,
                name="PGPAV",
                return_block_edges=False,
            )
            t_pgpav = time.perf_counter() - t0
            
            # gpav_op returns `u` aligned to `list(H.nodes())`.
            # the nodes of H_full_induced are indices in X (0..N-1).
            # We must map them back to standard array order buffer.
            nodes_pgpav = list(H_full_induced.nodes())
            u_pgpav = np.zeros(total_n, dtype=float)
            for i, node_idx in enumerate(nodes_pgpav):
                u_pgpav[node_idx] = u_pgpav_raw[i]

            obj_pgpav = float(np.sum((np.asarray(u_pgpav, dtype=float) - np.asarray(y_noisy, dtype=float)) ** 2))
        else:
            t_pgpav = np.nan
            obj_pgpav = np.nan

        result = {
            "config": {
                "q": q,
                "nQ": nQ,
                "avg_R": avg_R,
                "radius": params["radius"],
                "fiber_grid_step": params["fiber_grid_step"],
                "center_grid_step": params["center_grid_step"],
                "square_max": params["square_max"],
                "fiber_count_dist": fiber_count_dist,
                "model": model,
                "noise": noise,
                "noise_scale": noise_scale,
                "data_seed": data_seed,
                "noise_seed": noise_seed,
                "n_segments": n_segments,
                "max_workers": max_workers,
                "use_trend_following_first": use_trend_following_first,
                "use_trend_following_blocks": use_trend_following_blocks,
            },
            "sizes": {
                "num_fibers": m,
                "fiber_lengths": lengths,
                "N": total_n,
                "Q_num_edges": int(Q.number_of_edges()),
            },
            "timings_sec": {
                "build_full_X": t_build_X,
                "operadic_gpav": t_ogpav,
                "sb_gpav": t_sb,
                "pgpav": t_pgpav,
            },
            "memory_mb": {
                "operadic_gpav": mem_og_mb,
                "sb_gpav": mem_sb_mb,
            },
            "rss_delta_mb": {
                "operadic_gpav": rss_og_mb,
                "sb_gpav": rss_sb_mb,
            },
            "objective": {
                "operadic_gpav": obj_ogpav,
                "sb_gpav": obj_sb,
                "pgpav": obj_pgpav,
            },
            "baseline_noisy_vs_truth": baseline_metrics,
            "operadic_vs_truth": ogpav_metrics,
            "segmented_vs_truth": sb_metrics,
            "raw_outputs": {
                "y_true": y_true,
                "y_noisy": y_noisy,
                "u_ogpav": u_ogpav,
                "u_sb": u_sb,
            },
        }

        return result

    finally:
        # Clean up the lazy dataset cache directory
        try:
            if "data" in locals():
                R_datasets = data.get("R_points_list", None)
                if R_datasets is not None and hasattr(R_datasets, "cleanup"):
                    R_datasets.cleanup()
        except Exception:
            print("Warning: cleanup failed.")
            traceback.print_exc()


# ============================================================
# Repeated experiment
# ============================================================

def run_repeated_experiment(
    *,
    n_trials: int = 5,
    q: int = 50,
    fiber_count_dist: str = "poisson",
    model: str = "nonlinear",
    noise: str = "normal",
    noise_scale: float = 1.0,
    base_seed: int = 123,
    n_segments: Optional[int] = None,
    max_workers: Optional[int] = None,
    use_trend_following_first: bool = True,
    use_trend_following_blocks: bool = True,
    verbose: bool = False,
) -> List[Dict[str, object]]:
    """
    Run multiple independent trials.
    """
    tasks = [
        (base_seed + trial * 2, base_seed + trial * 2 + 1)
        for trial in range(n_trials)
    ]

    results = []
    for data_seed, noise_seed in tasks:
        res = run_one_trial(
            q=q,
            fiber_count_dist=fiber_count_dist,
            model=model,
            noise=noise,
            noise_scale=noise_scale,
            data_seed=data_seed,
            noise_seed=noise_seed,
            n_segments=n_segments,
            max_workers=max_workers,
            use_trend_following_first=use_trend_following_first,
            use_trend_following_blocks=use_trend_following_blocks,
            verbose=False,
        )
        results.append(res)

    return results


def summarize_results(results: List[Dict[str, object]]) -> Dict[str, object]:
    """Aggregate repeated-trial results."""
    def collect(path1: str, path2: str) -> np.ndarray:
        return np.array([r[path1][path2] for r in results], dtype=float)

    summary = {
        "n_trials": len(results),
        "baseline_noisy_vs_truth": {},
        "operadic_vs_truth": {},
        "segmented_vs_truth": {},
        "timings_sec": {},
        "memory_mb": {},
        "rss_delta_mb": {},
        "objective": {},
    }

    for metric in ["mse", "rmse", "mae", "max_abs"]:
        arr = collect("baseline_noisy_vs_truth", metric)
        summary["baseline_noisy_vs_truth"][metric] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=0)),
        }

        arr = collect("operadic_vs_truth", metric)
        summary["operadic_vs_truth"][metric] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=0)),
        }

        arr = collect("segmented_vs_truth", metric)
        summary["segmented_vs_truth"][metric] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=0)),
        }

    for metric in ["build_full_X", "operadic_gpav", "sb_gpav"]:
        arr = collect("timings_sec", metric)
        summary["timings_sec"][metric] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=0)),
        }

    for top_key in ("memory_mb", "rss_delta_mb", "objective"):
        for metric in ["operadic_gpav", "sb_gpav"]:
            arr = collect(top_key, metric)
            summary[top_key][metric] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=0)),
            }

    # Improvement over raw noisy signal
    base_mse = np.array([r["baseline_noisy_vs_truth"]["mse"] for r in results], dtype=float)
    og_mse = np.array([r["operadic_vs_truth"]["mse"] for r in results], dtype=float)
    sb_mse = np.array([r["segmented_vs_truth"]["mse"] for r in results], dtype=float)

    summary["mse_improvement_fraction"] = {
        "operadic_over_noisy_mean": float(np.mean((base_mse - og_mse) / base_mse)),
        "segmented_over_noisy_mean": float(np.mean((base_mse - sb_mse) / base_mse)),
    }

    # objective deviations (dev = (obj_SB - obj_PGPAV) / obj_PGPAV * 100)
    # dev2 = (obj_OGPAV - obj_PGPAV) / obj_PGPAV * 100
    og_obj = np.array([r["objective"]["operadic_gpav"] for r in results], dtype=float)
    sb_obj = np.array([r["objective"]["sb_gpav"] for r in results], dtype=float)
    pg_obj = np.array([r["objective"]["pgpav"] for r in results], dtype=float)
    
    # safeguard division
    pg_obj_safe = np.maximum(pg_obj, 1e-12)
    devs = (sb_obj - pg_obj) / pg_obj_safe * 100.0
    devs2 = (og_obj - pg_obj) / pg_obj_safe * 100.0

    # Handles the nan case if q > 100
    if np.isnan(devs).all():
        dev_mean, dev_std = np.nan, np.nan
        dev2_mean, dev2_std = np.nan, np.nan
    else:
        dev_mean = float(np.nanmean(devs))
        dev_std  = float(np.nanstd(devs, ddof=0))
        dev2_mean = float(np.nanmean(devs2))
        dev2_std  = float(np.nanstd(devs2, ddof=0))

    summary["dev_percent"] = {
        "mean": dev_mean,
        "std":  dev_std,
    }
    summary["dev2_percent"] = {
        "mean": dev2_mean,
        "std":  dev2_std,
    }

    return summary


def pretty_print_summary(summary: Dict[str, object]) -> None:
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"Trials: {summary['n_trials']}")

    print("\nBaseline noisy vs truth:")
    for k, v in summary["baseline_noisy_vs_truth"].items():
        print(f"  {k:8s} mean={v['mean']:.6f}  std={v['std']:.6f}")

    print("\nOperadicGPAV vs truth:")
    for k, v in summary["operadic_vs_truth"].items():
        print(f"  {k:8s} mean={v['mean']:.6f}  std={v['std']:.6f}")

    print("\nSB-GPAV vs truth:")
    for k, v in summary["segmented_vs_truth"].items():
        print(f"  {k:8s} mean={v['mean']:.6f}  std={v['std']:.6f}")

    print("\nTimings (seconds):")
    for k, v in summary["timings_sec"].items():
        print(f"  {k:12s} mean={v['mean']:.6f}  std={v['std']:.6f}")

    imp = summary["mse_improvement_fraction"]
    print("\nMean fractional MSE improvement over noisy observations:")
    print(f"  OperadicGPAV: {imp['operadic_over_noisy_mean']:.4%}")
    print(f"  SB-GPAV     : {imp['segmented_over_noisy_mean']:.4%}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    import math
    import json
    import csv
    import gc
    import os
    import traceback
    import argparse

    import matplotlib.pyplot as plt
    import numpy as np

    parser = argparse.ArgumentParser(description="Run complete OperadicGPAV vs SB-GPAV comparison.")
    parser.add_argument("--model", type=str, default=None,
                        help="Which slope config to run (linear_low, linear_mix, linear_high). "
                             "If omitted, runs all three.")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    # Three slope combinations from Sysoev et al. (Table 1/2).
    # noise_scale = q (from make_dataset_params) keeps SNR constant as q grows.
    SLOPE_CONFIGS = [
        {"model": "linear_low",  "label": "low  (α=0.2, 0.2)"},
        {"model": "linear_mix",  "label": "mix  (α=0.2, 2.0)"},
        {"model": "linear_high", "label": "high (α=2.0, 2.0)"},
    ]

    if args.model:
        SLOPE_CONFIGS = [sc for sc in SLOPE_CONFIGS if sc["model"] == args.model]
        if not SLOPE_CONFIGS:
            raise ValueError(f"Unknown model {args.model!r}. Choose from linear_low, linear_mix, linear_high.")

    # Medium-scale: size points chosen to test SB-GPAV limits
    # SB-GPAV crashed in the paper at N=500,000. 
    # q=30   -> N~900
    # q=100  -> N~10,000
    # q=316  -> N~100,000
    # q=1000 -> N~1,000,000 (Expected to OOM SB-GPAV)
    q_values = [30, 100, 316, 1000]
    R_values = [q * q for q in q_values]

    # Auto-stop after a certain time, or None defaults to off.
    max_allowed_seconds = None

    import multiprocessing
    n_cpus      = multiprocessing.cpu_count()
    max_workers = max(1, n_cpus)
    print(f"Detected {n_cpus} CPU(s) → using max_workers={max_workers}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def save_progress(rows, failures, out_csv, out_json):
        """Save CSV and JSON after every successful or failed experiment."""
        import json, csv
        if rows:
            fieldnames = list(rows[0].keys())
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"results": rows, "failures": failures}, f, indent=2)
    def make_plots_from_rows(rows):
        """Create plots from already-saved rows."""
        if not rows:
            print("No successful rows available; skipping plots.")
            return

        # Sort by R in case rows were appended out of order
        rows = sorted(rows, key=lambda r: r["R_target"])

        x = np.array([row["R_target"] for row in rows], dtype=float)

        operadic_time_mean = np.array([row["operadic_time_mean"] for row in rows], dtype=float)
        operadic_time_std = np.array([row["operadic_time_std"] for row in rows], dtype=float)
        sb_time_mean = np.array([row["sb_time_mean"] for row in rows], dtype=float)
        sb_time_std = np.array([row["sb_time_std"] for row in rows], dtype=float)

        operadic_rmse_mean = np.array([row["operadic_rmse_mean"] for row in rows], dtype=float)
        operadic_rmse_std = np.array([row["operadic_rmse_std"] for row in rows], dtype=float)
        sb_rmse_mean = np.array([row["sb_rmse_mean"] for row in rows], dtype=float)
        sb_rmse_std = np.array([row["sb_rmse_std"] for row in rows], dtype=float)

        # --------------------------------------------------------------
        # Plot 1: time vs R (log-log)
        # --------------------------------------------------------------
        plt.figure(figsize=(8, 5))
        plt.plot(x, operadic_time_mean, marker="o", label="OperadicGPAV time")
        plt.plot(x, sb_time_mean, marker="o", label="SB-GPAV time")

        plt.xscale("log")
        plt.yscale("log")

        plt.xlabel("Target dataset size R")
        plt.ylabel("Time (seconds)")
        plt.title("Runtime vs dataset size (log-log)")
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig("time_vs_R_loglog.png", dpi=220)
        plt.close()

        # --------------------------------------------------------------
        # Plot 2: RMSE tube plot (mean ± std)
        # --------------------------------------------------------------
        plt.figure(figsize=(8, 5))

        plt.plot(x, operadic_rmse_mean, marker="o", label="OperadicGPAV RMSE")
        plt.fill_between(
            x,
            operadic_rmse_mean - operadic_rmse_std,
            operadic_rmse_mean + operadic_rmse_std,
            alpha=0.2,
        )

        plt.plot(x, sb_rmse_mean, marker="o", label="SB-GPAV RMSE")
        plt.fill_between(
            x,
            sb_rmse_mean - sb_rmse_std,
            sb_rmse_mean + sb_rmse_std,
            alpha=0.2,
        )

        plt.xlabel("Target dataset size R")
        plt.ylabel("RMSE")
        plt.title("RMSE vs dataset size (mean ± std over 3 trials)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("rmse_tube_vs_R.png", dpi=220)
        plt.close()

        print("Saved plots:")
        print("  time_vs_R_loglog.png")
        print("  rmse_tube_vs_R.png")

    all_slope_rows = {}  # model_key -> list of row dicts

    for slope_cfg in SLOPE_CONFIGS:
        model_key = slope_cfg["model"]
        label     = slope_cfg["label"]

        out_csv  = f"scaling_{model_key}.csv"
        out_json = f"scaling_{model_key}.json"

        print(f"\n{'#'*80}")
        print(f"# Slope config: {label}")
        print(f"{'#'*80}")
        # Resume support  per slope config
        rows = []
        failures = []
        completed_R = set()

        if os.path.exists(out_json):
            try:
                with open(out_json, "r", encoding="utf-8") as f:
                    old = json.load(f)
                rows = old.get("results", [])
                failures = old.get("failures", [])
                completed_R = {row["R_target"] for row in rows}
                print(f"  Loaded existing progress: {len(rows)} done, {len(failures)} failures.")
            except Exception as e:
                print(f"  Warning: could not load {out_json}: {e}")

        # ------------------------------------------------------------------
        # Inner experiment loop (fixed slope config)
        # ------------------------------------------------------------------
        stop_slope = False
        for q, R in zip(q_values, R_values):
            if stop_slope:
                break
            if R in completed_R:
                print(f"  Skipping already completed R={R}")
                continue

            n_segments  = q
            _p          = make_dataset_params(q)
            noise_scale = _p["noise_scale"]   # float(q)
            print(f"\n  {'='*70}")
            print(
                f"  R={R}  q={q}  nQ={_p['nQ']}  noise_scale={noise_scale:.0f}  "
                f"model={model_key}"
            )

            try:
                results = run_repeated_experiment(
                    n_trials=3,
                    q=q,
                    fiber_count_dist="poisson",
                    model=model_key,
                    noise="normal",
                    noise_scale=noise_scale,
                    base_seed=2026 + q,
                    n_segments=n_segments,
                    max_workers=32,
                    use_trend_following_first=False,
                    use_trend_following_blocks=False,
                    verbose=False,
                )

                summary = summarize_results(results)
                realized_N       = [int(r["sizes"]["N"]) for r in results]
                realized_Q_edges = [int(r["sizes"]["Q_num_edges"]) for r in results]

                N_actual_mean = float(np.mean(realized_N))

                row = {
                    "slope_config": model_key,
                    "R_target":     int(R),
                    "sqrt_R":       int(q),
                    "nQ_input":     int(_p["nQ"]),
                    "avg_R_input":  int(_p["avg_R"]),
                    "noise_scale":  noise_scale,
                    "n_segments":   int(n_segments),

                    "N_actual_mean": N_actual_mean,
                    "N_actual_std": float(np.std(realized_N, ddof=0)),
                    "Q_edges_mean": float(np.mean(realized_Q_edges)),
                    "Q_edges_std": float(np.std(realized_Q_edges, ddof=0)),

                    "build_full_X_mean": float(summary["timings_sec"]["build_full_X"]["mean"]),
                    "build_full_X_std": float(summary["timings_sec"]["build_full_X"]["std"]),
                    "operadic_time_mean": float(summary["timings_sec"]["operadic_gpav"]["mean"]),
                    "operadic_time_std": float(summary["timings_sec"]["operadic_gpav"]["std"]),
                    "sb_time_mean": float(summary["timings_sec"]["sb_gpav"]["mean"]),
                    "sb_time_std": float(summary["timings_sec"]["sb_gpav"]["std"]),

                    "operadic_mem_mb_mean": float(summary["memory_mb"]["operadic_gpav"]["mean"]),
                    "operadic_mem_mb_std":  float(summary["memory_mb"]["operadic_gpav"]["std"]),
                    "sb_mem_mb_mean":       float(summary["memory_mb"]["sb_gpav"]["mean"]),
                    "sb_mem_mb_std":        float(summary["memory_mb"]["sb_gpav"]["std"]),

                    "operadic_rss_mb_mean": float(summary["rss_delta_mb"]["operadic_gpav"]["mean"]),
                    "operadic_rss_mb_std":  float(summary["rss_delta_mb"]["operadic_gpav"]["std"]),
                    "sb_rss_mb_mean":       float(summary["rss_delta_mb"]["sb_gpav"]["mean"]),
                    "sb_rss_mb_std":        float(summary["rss_delta_mb"]["sb_gpav"]["std"]),

                    "operadic_obj_mean": float(summary["objective"]["operadic_gpav"]["mean"]),
                    "operadic_obj_std":  float(summary["objective"]["operadic_gpav"]["std"]),
                    "sb_obj_mean":       float(summary["objective"]["sb_gpav"]["mean"]),
                    "sb_obj_std":        float(summary["objective"]["sb_gpav"]["std"]),

                    "dev_percent_mean": float(summary["dev_percent"]["mean"]),
                    "dev_percent_std":  float(summary["dev_percent"]["std"]),

                    "dev2_percent_mean": float(summary["dev2_percent"]["mean"]),
                    "dev2_percent_std":  float(summary["dev2_percent"]["std"]),

                    "baseline_rmse_mean": float(summary["baseline_noisy_vs_truth"]["rmse"]["mean"]),
                    "operadic_rmse_mean": float(summary["operadic_vs_truth"]["rmse"]["mean"]),
                    "operadic_rmse_std": float(summary["operadic_vs_truth"]["rmse"]["std"]),
                    "sb_rmse_mean": float(summary["segmented_vs_truth"]["rmse"]["mean"]),
                    "sb_rmse_std": float(summary["segmented_vs_truth"]["rmse"]["std"]),
                    
                    # --- Normalised metrics (should be ~constant across q) ---
                    "obj_normalised_og": float(summary["objective"]["operadic_gpav"]["mean"]
                                               / max(N_actual_mean * noise_scale**2, 1e-12)),
                    "obj_normalised_sb": float(summary["objective"]["sb_gpav"]["mean"]
                                               / max(N_actual_mean * noise_scale**2, 1e-12)),
                    "rmse_normalised_og": float(summary["operadic_vs_truth"]["rmse"]["mean"] / max(noise_scale, 1e-12)),
                    "rmse_normalised_sb": float(summary["segmented_vs_truth"]["rmse"]["mean"] / max(noise_scale, 1e-12)),
                    "baseline_rmse_normalised": float(summary["baseline_noisy_vs_truth"]["rmse"]["mean"] / max(noise_scale, 1e-12)),
                }

                rows.append(row)
                save_progress(rows, failures, out_csv, out_json)

                print(
                    f"  Done R={R:,} | N={row['N_actual_mean']:,.0f} | "
                    f"Time: OG={row['operadic_time_mean']:.3f}s, SB={row['sb_time_mean']:.3f}s | "
                    f"RSS: OG={row['operadic_rss_mb_mean']:.1f}MB, SB={row['sb_rss_mb_mean']:.1f}MB | "
                    f"dev(SB/PG)={row['dev_percent_mean']:.3f}%, dev2(OG/PG)={row['dev2_percent_mean']:.3f}%"
                )

                if max_allowed_seconds is not None and (
                    row["operadic_time_mean"] > max_allowed_seconds
                    or row["sb_time_mean"] > max_allowed_seconds
                ):
                    print("  Stopping slope config: runtime limit reached.")
                    stop_slope = True

            except Exception as e:
                failures.append({
                    "slope_config":  model_key,
                    "R_target":      int(R),
                    "sqrt_R":        int(q),
                    "nQ_input":      int(_p["nQ"]),
                    "avg_R_input":   int(_p["avg_R"]),
                    "n_segments":    int(n_segments),
                    "error":         str(e),
                    "traceback":     traceback.format_exc(),
                })
                save_progress(rows, failures, out_csv, out_json)
                print(f"  FAILED at R={R}: {e}")
                stop_slope = True

            finally:
                gc.collect()

        all_slope_rows[model_key] = rows

    # ------------------------------------------------------------------
    # Plots — all three slopes on the same axes
    # ------------------------------------------------------------------
    colors = {
        "linear_low":  "tab:blue",
        "linear_mix":  "tab:orange",
        "linear_high": "tab:green",
    }
    LABELS = {sc["model"]: sc["label"] for sc in SLOPE_CONFIGS}

    def slope_plot(mean_key, std_key, ylabel, fname, logy=False):
        plt.figure(figsize=(9, 5))
        has_data = False
        for key, rs_all in all_slope_rows.items():
            rs = sorted(rs_all, key=lambda r: r["R_target"])
            if not rs:
                continue
            has_data = True
            x    = np.array([r["R_target"] for r in rs], dtype=float)
            mean = np.array([r[mean_key]   for r in rs], dtype=float)
            std  = np.array([r[std_key]    for r in rs], dtype=float)
            plt.plot(x, mean, marker="o", label=LABELS[key], color=colors[key])
            plt.fill_between(x, mean-std, mean+std, alpha=0.15, color=colors[key])
        if not has_data:
            plt.close(); return
        plt.xscale("log")
        if logy: plt.yscale("log")
        plt.xlabel("Target dataset size R")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs R  (OGPAV vs SB-GPAV, noise_scale=q)")
        plt.legend(); plt.grid(True, which="both" if logy else "major", alpha=0.3)
        plt.tight_layout()
        plt.savefig(fname, dpi=220); plt.close()
        print(f"Saved {fname}")

    try:
        slope_plot("operadic_time_mean", "operadic_time_std",
                   "OperadicGPAV time (s)", "time_ogpav_vs_R.png", logy=True)
        slope_plot("sb_time_mean",       "sb_time_std",
                   "SB-GPAV time (s)",     "time_sb_vs_R.png",    logy=True)
        slope_plot("operadic_rss_mb_mean", "operadic_rss_mb_std",
                   "OperadicGPAV Memory RSS (MB)", "mem_rss_ogpav_vs_R.png", logy=True)
        slope_plot("sb_rss_mb_mean", "sb_rss_mb_std",
                   "SB-GPAV Memory RSS (MB)", "mem_rss_sb_vs_R.png", logy=True)
        slope_plot("rmse_normalised_og", "operadic_rmse_std",
                   "OperadicGPAV norm RMSE",    "norm_rmse_ogpav_vs_R.png")
        slope_plot("rmse_normalised_sb", "sb_rmse_std",
                   "SB-GPAV norm RMSE",         "norm_rmse_sb_vs_R.png")
        slope_plot("dev_percent_mean", "dev_percent_std",
                   "Deviation SB vs PGPAV (%)", "dev_vs_R.png")
    except Exception as e:
        print(f"Could not generate plots: {e}")

    print("\nAll done.")