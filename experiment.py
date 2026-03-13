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
from typing import Dict, List, Tuple, Optional

import numpy as np
import networkx as nx

from OperadicGPAV import OperadicGPAV
from utils.geometric_sb_dataset import (
    generate_dataset_lazy,
    MODEL_FUNCS,
)
from utils.sb_gpav import sb_gpav

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
    nQ: int = 50,
    avg_R: int = 200,
    radius: float = 1/3,
    min_dist: float = 0.02,
    square_min: int = 0,
    square_max: int = 100,
    fiber_count_dist: str = "poisson",
    model: str = "nonlinear",
    noise: str = "normal",
    noise_scale: float = 1.0,
    data_seed: int = 0,
    noise_seed: int = 1,
    min_center_dist: float = 0.0,
    n_segments: Optional[int] = None,
    max_workers: Optional[int] = None,
    use_trend_following_first: bool = True,
    use_trend_following_blocks: bool = True,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Run one comparison trial.
    """
    temp_cache_dir = None
    try:
        # --------------------------------------------------------
        # 1. Generate lazy dataset
        # --------------------------------------------------------
        if verbose:
            print("=" * 80)
            print(f"Generating lazy dataset: nQ={nQ}, avg_R={avg_R}, seed={data_seed}")

        data = generate_dataset_lazy(
            nQ=nQ,
            avg_R=avg_R,
            radius=radius,
            min_dist=min_dist,
            square_min=square_min,
            square_max=square_max,
            min_center_dist=min_center_dist,
            seed=data_seed,
            fiber_count_dist=fiber_count_dist,
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
        # 2. Build truth/noisy response from fibers only
        # --------------------------------------------------------
        lengths = R_datasets.get_fiber_lengths()
        total_n = int(sum(lengths))
        if verbose:
            print(f"Number of fibers: {m}")
            print(f"Total sample size N: {total_n}")

        y_true = make_y_true_from_fibers(R_datasets, model=model)
        eps = make_noise(
            total_n,
            noise=noise,
            noise_scale=noise_scale,
            seed=noise_seed,
        )
        y_noisy = y_true + eps

        baseline_metrics = compute_metrics(y_true, y_noisy)

        print("len(y_true):", len(y_true))
        print("len(y_noisy):", len(y_noisy))

        # --------------------------------------------------------
        # 3. Run OperadicGPAV directly on lazy fibers
        # --------------------------------------------------------
        if verbose:
            print("\nRunning OperadicGPAV...")

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

        ogpav_metrics = compute_metrics(y_true, u_ogpav)

        # --------------------------------------------------------
        # 4. Build full X ONCE for sb_gpav
        # --------------------------------------------------------
        if verbose:
            print("\nBuilding full X for sb_gpav (without building global poset)...")

        t0 = time.perf_counter()
        L = topological_order_from_lazy_fibers(R_datasets)

        # only build X AFTER the order is known
        X, lengths_check, indices_list = build_full_X_from_lazy(R_datasets)
        #X, lengths_check, indices_list = build_full_X_from_lazy(R_datasets)
        t_build_X = time.perf_counter() - t0

        if lengths != lengths_check:
            raise RuntimeError("Fiber lengths changed unexpectedly while building X")

        # One valid topological order under coordinate-wise <=
        #L = topological_order_from_coordinates(X)

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

        t0 = time.perf_counter()
        u_sb = sb_gpav(
            X=X,
            Y=y_noisy,
            L=L,
            f=None,  # default comparator = coordinate-wise <=
            weights=None,
            n_segments=n_segments,
            assume_component_wise=True,
            verbose=False,
            debug=False,
        )
        t_sb = time.perf_counter() - t0

        sb_metrics = compute_metrics(y_true, u_sb)

        result = {
            "config": {
                "nQ": nQ,
                "avg_R": avg_R,
                "radius": radius,
                "min_dist": min_dist,
                "square_min": square_min,
                "square_max": square_max,
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
    nQ: int = 50,
    avg_R: int = 200,
    radius: float = 1/3,
    min_dist: float = 0.02,
    square_min: int = 0,
    square_max: int = 100,
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
    min_center_dist: float = 0.0,
) -> List[Dict[str, object]]:
    """
    Run multiple independent trials.
    """

    tasks = []
    for trial in range(n_trials):

        data_seed = base_seed + trial * 2
        noise_seed = base_seed + trial * 2 + 1

        tasks.append((data_seed, noise_seed))

    results = []
    for data_seed, noise_seed in tasks:
        res = run_one_trial(
            nQ=nQ,
            avg_R=avg_R,
            radius=radius,
            min_dist=min_dist,
            square_min=square_min,
            square_max=square_max,
            min_center_dist=min_center_dist,
            fiber_count_dist=fiber_count_dist,
            model=model,
            noise=noise,
            noise_scale=noise_scale,
            data_seed=data_seed,
            noise_seed=noise_seed,
            n_segments=n_segments,
            max_workers=max_workers,   # pass through
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

    # Improvement over raw noisy signal
    base_mse = np.array([r["baseline_noisy_vs_truth"]["mse"] for r in results], dtype=float)
    og_mse = np.array([r["operadic_vs_truth"]["mse"] for r in results], dtype=float)
    sb_mse = np.array([r["segmented_vs_truth"]["mse"] for r in results], dtype=float)

    summary["mse_improvement_fraction"] = {
        "operadic_over_noisy_mean": float(np.mean((base_mse - og_mse) / base_mse)),
        "segmented_over_noisy_mean": float(np.mean((base_mse - sb_mse) / base_mse)),
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

    import matplotlib.pyplot as plt
    import numpy as np

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    # We use only perfect squares R = q^2, so that:
    #   nQ = q
    #   avg_R = q
    #   n_segments = q
    #
    # This means the intended segment size is also about q = sqrt(R).
    #
    # Start around 6000 total samples and then grow fast toward big data.
    # Since R must be a square, 80^2 = 6400 is the closest clean start.
    q_values = [
        100, #10, 000
        1000,  # 1,000,000
        10000,  # 100,000,000
        100000,  # 10,000,000,000
    ]
    R_values = [q * q for q in q_values]

    # Stop automatically if either algorithm becomes too slow.
    # Set to None to disable.
    max_allowed_seconds = None  # example: 600

    out_csv = "scaling_results.csv"
    out_json = "scaling_results.json"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def save_progress(rows, failures):
        """Save CSV and JSON after every successful or failed experiment."""
        if rows:
            fieldnames = list(rows[0].keys())
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "results": rows,
                    "failures": failures,
                },
                f,
                indent=2,
            )

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

    # ------------------------------------------------------------------
    # Resume support: if JSON exists, load prior successful rows/failures
    # ------------------------------------------------------------------
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
            print(f"Loaded existing progress: {len(rows)} successful runs, {len(failures)} failures.")
        except Exception as e:
            print(f"Warning: could not load existing {out_json}: {e}")

    # ------------------------------------------------------------------
    # Main experiment loop
    # ------------------------------------------------------------------
    for q, R in zip(q_values, R_values):
        if R in completed_R:
            print(f"Skipping already completed R={R}")
            continue

        nQ = q
        avg_R = q
        n_segments = q  # because we want segment size ~ sqrt(R)
        scale = math.sqrt(q / 80)
        radius = (1/3) * scale
        square_max = 1.25 * q
        print("\n" + "=" * 80)
        print(f"Running target R={R}")
        print(f"sqrt(R)={q}, nQ={nQ}, avg_R={avg_R}, n_segments={n_segments}")

        try:
            # 3 repetitions so we can build mean/std tubes
            results = run_repeated_experiment(
                n_trials=3,
                nQ=nQ,
                avg_R=avg_R,
                radius=radius,
                min_dist=0.02,
                square_min=0,
                square_max=square_max,
                fiber_count_dist="poisson",
                model="nonlinear",
                noise="normal",
                noise_scale=1.0,
                base_seed=2026 + q,   # vary seed with size
                n_segments=n_segments,
                max_workers=32,
                use_trend_following_first=False,
                use_trend_following_blocks=False,
                verbose=False,
            )

            summary = summarize_results(results)

            # actual realized sample sizes can vary because avg_R is an average
            realized_N = [int(r["sizes"]["N"]) for r in results]
            realized_Q_edges = [int(r["sizes"]["Q_num_edges"]) for r in results]

            row = {
                "R_target": int(R),
                "sqrt_R": int(q),
                "nQ_input": int(nQ),
                "avg_R_input": int(avg_R),
                "n_segments": int(n_segments),

                "N_actual_mean": float(np.mean(realized_N)),
                "N_actual_std": float(np.std(realized_N, ddof=0)),
                "Q_edges_mean": float(np.mean(realized_Q_edges)),
                "Q_edges_std": float(np.std(realized_Q_edges, ddof=0)),

                "build_full_X_mean": float(summary["timings_sec"]["build_full_X"]["mean"]),
                "build_full_X_std": float(summary["timings_sec"]["build_full_X"]["std"]),
                "operadic_time_mean": float(summary["timings_sec"]["operadic_gpav"]["mean"]),
                "operadic_time_std": float(summary["timings_sec"]["operadic_gpav"]["std"]),
                "sb_time_mean": float(summary["timings_sec"]["sb_gpav"]["mean"]),
                "sb_time_std": float(summary["timings_sec"]["sb_gpav"]["std"]),

                "baseline_rmse_mean": float(summary["baseline_noisy_vs_truth"]["rmse"]["mean"]),
                "baseline_rmse_std": float(summary["baseline_noisy_vs_truth"]["rmse"]["std"]),
                "operadic_rmse_mean": float(summary["operadic_vs_truth"]["rmse"]["mean"]),
                "operadic_rmse_std": float(summary["operadic_vs_truth"]["rmse"]["std"]),
                "sb_rmse_mean": float(summary["segmented_vs_truth"]["rmse"]["mean"]),
                "sb_rmse_std": float(summary["segmented_vs_truth"]["rmse"]["std"]),

                "baseline_mse_mean": float(summary["baseline_noisy_vs_truth"]["mse"]["mean"]),
                "baseline_mse_std": float(summary["baseline_noisy_vs_truth"]["mse"]["std"]),
                "operadic_mse_mean": float(summary["operadic_vs_truth"]["mse"]["mean"]),
                "operadic_mse_std": float(summary["operadic_vs_truth"]["mse"]["std"]),
                "sb_mse_mean": float(summary["segmented_vs_truth"]["mse"]["mean"]),
                "sb_mse_std": float(summary["segmented_vs_truth"]["mse"]["std"]),

                "baseline_mae_mean": float(summary["baseline_noisy_vs_truth"]["mae"]["mean"]),
                "baseline_mae_std": float(summary["baseline_noisy_vs_truth"]["mae"]["std"]),
                "operadic_mae_mean": float(summary["operadic_vs_truth"]["mae"]["mean"]),
                "operadic_mae_std": float(summary["operadic_vs_truth"]["mae"]["std"]),
                "sb_mae_mean": float(summary["segmented_vs_truth"]["mae"]["mean"]),
                "sb_mae_std": float(summary["segmented_vs_truth"]["mae"]["std"]),

                "baseline_max_abs_mean": float(summary["baseline_noisy_vs_truth"]["max_abs"]["mean"]),
                "baseline_max_abs_std": float(summary["baseline_noisy_vs_truth"]["max_abs"]["std"]),
                "operadic_max_abs_mean": float(summary["operadic_vs_truth"]["max_abs"]["mean"]),
                "operadic_max_abs_std": float(summary["operadic_vs_truth"]["max_abs"]["std"]),
                "sb_max_abs_mean": float(summary["segmented_vs_truth"]["max_abs"]["mean"]),
                "sb_max_abs_std": float(summary["segmented_vs_truth"]["max_abs"]["std"]),
            }

            rows.append(row)
            save_progress(rows, failures)

            print(
                f"Done R={R} | "
                f"N_actual_mean={row['N_actual_mean']:.1f} | "
                f"Operadic time={row['operadic_time_mean']:.3f}s | "
                f"SB time={row['sb_time_mean']:.3f}s | "
                f"Operadic RMSE={row['operadic_rmse_mean']:.6g} | "
                f"SB RMSE={row['sb_rmse_mean']:.6g}"
            )

            # Optional runtime stop
            if max_allowed_seconds is not None:
                if (
                    row["operadic_time_mean"] > max_allowed_seconds
                    or row["sb_time_mean"] > max_allowed_seconds
                ):
                    print("Stopping because runtime limit was reached.")
                    break

        except Exception as e:
            failures.append(
                {
                    "R_target": int(R),
                    "sqrt_R": int(q),
                    "nQ_input": int(nQ),
                    "avg_R_input": int(avg_R),
                    "n_segments": int(n_segments),
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            )
            save_progress(rows, failures)
            print(f"FAILED at R={R}: {e}")
            break

        finally:
            # Force cleanup between experiment sizes
            gc.collect()

    # ------------------------------------------------------------------
    # Always try to make plots from whatever has been saved so far
    # ------------------------------------------------------------------
    try:
        # Reload from JSON to ensure we plot the on-disk data
        if os.path.exists(out_json):
            with open(out_json, "r", encoding="utf-8") as f:
                saved = json.load(f)
            saved_rows = saved.get("results", [])
            make_plots_from_rows(saved_rows)
        else:
            make_plots_from_rows(rows)
    except Exception as e:
        print(f"Could not generate plots: {e}")

    print("\nSaved:")
    print(f"  {out_csv}")
    print(f"  {out_json}")