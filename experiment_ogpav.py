# -*- coding: utf-8 -*-
"""
experiment_ogpav.py

OperadicGPAV-only scaling study.

Use this after experiment.py has established the comparison at moderate sizes
and confirmed that sb_gpav cannot scale further.  This script runs OperadicGPAV
alone — no sb_gpav, no full-X materialisation — so it can handle much larger q.

What it does
------------
1. For each q in q_values, derive geometry with make_dataset_params(q).
2. Generate a lazy dataset (fibers on disk, never build global X).
3. Build y_true fiber-by-fiber; add noise.
4. Run OperadicGPAV and record timing + RMSE.
5. Save progress to CSV/JSON after each size (resume-safe).
6. Plot timing and RMSE vs R on completion.
"""

from __future__ import annotations

import csv
import gc
import json
import math
import os
import time
import traceback
from typing import Dict, List, Optional

import numpy as np
import networkx as nx

from OperadicGPAV import OperadicGPAV
from utils.geometric_sb_dataset import (
    generate_standard,
    make_dataset_params,
    MODEL_FUNCS,
)

# ============================================================
# Helpers  (self-contained, no import from experiment.py)
# ============================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    mse  = float(np.mean(err ** 2))
    return {
        "mse":     mse,
        "rmse":    float(np.sqrt(mse)),
        "mae":     float(np.mean(np.abs(err))),
        "max_abs": float(np.max(np.abs(err))),
    }


def make_noise(n: int, *, noise: str = "normal", noise_scale: float = 1.0,
               seed: Optional[int] = None) -> np.ndarray:
    rg = np.random.default_rng(seed)
    if noise == "none":
        return np.zeros(n, dtype=float)
    if noise == "normal":
        return rg.normal(0.0, noise_scale, size=n).astype(float)
    if noise == "laplace":
        return rg.laplace(0.0, noise_scale, size=n).astype(float)
    raise ValueError(f"Unknown noise '{noise}'")


def make_y_true_from_fibers(R_datasets, *, model: str = "nonlinear") -> np.ndarray:
    f = MODEL_FUNCS[model]
    return np.concatenate([f(np.asarray(R_datasets[i], dtype=float))
                           for i in range(len(R_datasets))], axis=0)


def ensure_q_nodes_match_num_fibers(Q: nx.DiGraph, m: int) -> nx.DiGraph:
    q_nodes = list(Q.nodes())
    if set(q_nodes) == set(range(m)):
        return Q
    if len(q_nodes) != m:
        raise ValueError(f"Q has {len(q_nodes)} nodes but {m} fibers.")
    mapping = {old: new for new, old in enumerate(sorted(q_nodes))}
    return nx.relabel_nodes(Q, mapping, copy=True)


# ============================================================
# One trial  (OperadicGPAV only)
# ============================================================

def run_one_trial(
    *,
    q: int,
    fiber_count_dist: str = "poisson",
    model: str = "nonlinear",
    noise: str = "normal",
    noise_scale: float = 1.0,
    data_seed: int = 0,
    noise_seed: int = 1,
    max_workers: Optional[int] = None,
    use_trend_following_first: bool = False,
    use_trend_following_blocks: bool = False,
    verbose: bool = False,
) -> Dict[str, object]:
    params = make_dataset_params(q)
    nQ    = params["nQ"]
    avg_R = params["avg_R"]

    try:
        if verbose:
            print(f"  Generating q={q} (nQ={nQ}, avg_R={avg_R}, seed={data_seed}) ...")

        data = generate_standard(q, seed=data_seed,
                                 fiber_count_dist=fiber_count_dist, lazy=True)

        R_datasets = data["R_points_list"]
        Q = ensure_q_nodes_match_num_fibers(data["Q_hasse"], len(R_datasets))

        lengths  = R_datasets.get_fiber_lengths()
        total_n  = int(sum(lengths))

        y_true  = make_y_true_from_fibers(R_datasets, model=model)
        y_noisy = y_true + make_noise(total_n, noise=noise,
                                      noise_scale=noise_scale, seed=noise_seed)

        baseline_metrics = compute_metrics(y_true, y_noisy)

        if verbose:
            print(f"  Running OperadicGPAV (N={total_n}) ...")

        t0 = time.perf_counter()
        u_ogpav = OperadicGPAV(
            Q=Q,
            R_datasets=R_datasets,
            Y=y_noisy,
            indices_list=None,
            segment_topo_orders=None,
            use_trend_following_first=use_trend_following_first,
            use_trend_following_blocks=use_trend_following_blocks,
            assume_component_wise=True,
            max_workers=max_workers,
            verbose=False,
            debug=False,
            temp_dir=None,
        )
        t_ogpav = time.perf_counter() - t0

        return {
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
                "max_workers": max_workers,
            },
            "sizes": {
                "num_fibers": len(R_datasets),
                "fiber_lengths": lengths,
                "N": total_n,
                "Q_num_edges": int(Q.number_of_edges()),
            },
            "timings_sec": {"operadic_gpav": t_ogpav},
            "baseline_noisy_vs_truth": baseline_metrics,
            "operadic_vs_truth": compute_metrics(y_true, u_ogpav),
        }

    finally:
        try:
            rds = data.get("R_points_list") if "data" in locals() else None
            if rds is not None and hasattr(rds, "cleanup"):
                rds.cleanup()
        except Exception:
            traceback.print_exc()


# ============================================================
# Repeated experiment
# ============================================================

def run_repeated_experiment(
    *,
    n_trials: int = 3,
    q: int,
    fiber_count_dist: str = "poisson",
    model: str = "nonlinear",
    noise: str = "normal",
    noise_scale: float = 1.0,
    base_seed: int = 123,
    max_workers: Optional[int] = None,
    use_trend_following_first: bool = False,
    use_trend_following_blocks: bool = False,
    verbose: bool = False,
) -> List[Dict[str, object]]:
    return [
        run_one_trial(
            q=q,
            fiber_count_dist=fiber_count_dist,
            model=model,
            noise=noise,
            noise_scale=noise_scale,
            data_seed=base_seed + trial * 2,
            noise_seed=base_seed + trial * 2 + 1,
            max_workers=max_workers,
            use_trend_following_first=use_trend_following_first,
            use_trend_following_blocks=use_trend_following_blocks,
            verbose=verbose,
        )
        for trial in range(n_trials)
    ]


def summarize_results(results: List[Dict[str, object]]) -> Dict[str, object]:
    def arr(k1, k2):
        return np.array([r[k1][k2] for r in results], dtype=float)

    summary: Dict[str, object] = {"n_trials": len(results)}

    for section in ("baseline_noisy_vs_truth", "operadic_vs_truth"):
        summary[section] = {}
        for metric in ("mse", "rmse", "mae", "max_abs"):
            a = arr(section, metric)
            summary[section][metric] = {"mean": float(a.mean()), "std": float(a.std(ddof=0))}

    a = arr("timings_sec", "operadic_gpav")
    summary["timings_sec"] = {"operadic_gpav": {"mean": float(a.mean()), "std": float(a.std(ddof=0))}}

    return summary


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # ------------------------------------------------------------------
    # Configuration
    # Extend q_values well beyond what sb_gpav can handle.
    # R = q^2 is the approximate total dataset size.
    # ------------------------------------------------------------------
    q_values = [
        100,      # R ~  10 000
        1_000,    # R ~   1 000 000
        10_000,   # R ~ 100 000 000
        100_000,  # R ~  10^10   (run only if hardware allows)
    ]
    R_values = [q * q for q in q_values]

    max_allowed_seconds = None  # set e.g. 3600 to auto-stop

    out_csv  = "scaling_ogpav_results.csv"
    out_json = "scaling_ogpav_results.json"

    # ------------------------------------------------------------------
    # Resume support
    # ------------------------------------------------------------------
    rows: List[dict]     = []
    failures: List[dict] = []
    completed_R: set     = set()

    if os.path.exists(out_json):
        try:
            with open(out_json, "r", encoding="utf-8") as f:
                old = json.load(f)
            rows       = old.get("results", [])
            failures   = old.get("failures", [])
            completed_R = {row["R_target"] for row in rows}
            print(f"Loaded prior progress: {len(rows)} done, {len(failures)} failed.")
        except Exception as e:
            print(f"Warning: could not load {out_json}: {e}")

    def save_progress(rows, failures):
        if rows:
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"results": rows, "failures": failures}, f, indent=2)

    # ------------------------------------------------------------------
    # Experiment loop
    # ------------------------------------------------------------------
    for q, R in zip(q_values, R_values):
        if R in completed_R:
            print(f"Skipping already completed R={R:,}")
            continue

        _p = make_dataset_params(q)
        print(f"\n{'='*80}")
        print(
            f"Running R={R:,}  (q={q}, nQ={_p['nQ']}, avg_R={_p['avg_R']}, "
            f"radius={_p['radius']:.3f}, center_step={_p['center_grid_step']}, "
            f"square_max={_p['square_max']})"
        )

        try:
            results = run_repeated_experiment(
                n_trials=3,
                q=q,
                fiber_count_dist="poisson",
                model="nonlinear",
                noise="normal",
                noise_scale=1.0,
                base_seed=2026 + q,
                max_workers=32,
                use_trend_following_first=False,
                use_trend_following_blocks=False,
                verbose=True,
            )

            summary = summarize_results(results)
            realized_N      = [int(r["sizes"]["N"])          for r in results]
            realized_Q_edges = [int(r["sizes"]["Q_num_edges"]) for r in results]

            row = {
                "R_target":            int(R),
                "sqrt_R":              int(q),
                "nQ_input":            int(_p["nQ"]),
                "avg_R_input":         int(_p["avg_R"]),
                "N_actual_mean":       float(np.mean(realized_N)),
                "N_actual_std":        float(np.std(realized_N, ddof=0)),
                "Q_edges_mean":        float(np.mean(realized_Q_edges)),
                "Q_edges_std":         float(np.std(realized_Q_edges, ddof=0)),
                "operadic_time_mean":  float(summary["timings_sec"]["operadic_gpav"]["mean"]),
                "operadic_time_std":   float(summary["timings_sec"]["operadic_gpav"]["std"]),
                "baseline_rmse_mean":  float(summary["baseline_noisy_vs_truth"]["rmse"]["mean"]),
                "baseline_rmse_std":   float(summary["baseline_noisy_vs_truth"]["rmse"]["std"]),
                "operadic_rmse_mean":  float(summary["operadic_vs_truth"]["rmse"]["mean"]),
                "operadic_rmse_std":   float(summary["operadic_vs_truth"]["rmse"]["std"]),
                "operadic_mse_mean":   float(summary["operadic_vs_truth"]["mse"]["mean"]),
                "operadic_mse_std":    float(summary["operadic_vs_truth"]["mse"]["std"]),
                "operadic_mae_mean":   float(summary["operadic_vs_truth"]["mae"]["mean"]),
                "operadic_mae_std":    float(summary["operadic_vs_truth"]["mae"]["std"]),
            }

            rows.append(row)
            save_progress(rows, failures)

            print(
                f"Done R={R:,} | N_mean={row['N_actual_mean']:.0f} | "
                f"time={row['operadic_time_mean']:.3f}s | "
                f"RMSE={row['operadic_rmse_mean']:.6g}"
            )

            if max_allowed_seconds and row["operadic_time_mean"] > max_allowed_seconds:
                print("Stopping: runtime limit reached.")
                break

        except Exception as e:
            failures.append({
                "R_target":  int(R),
                "sqrt_R":    int(q),
                "nQ_input":  int(_p["nQ"]),
                "avg_R_input": int(_p["avg_R"]),
                "error":     str(e),
                "traceback": traceback.format_exc(),
            })
            save_progress(rows, failures)
            print(f"FAILED at R={R:,}: {e}")
            break

        finally:
            gc.collect()

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    if not rows:
        print("No successful rows; skipping plots.")
    else:
        rows_sorted = sorted(rows, key=lambda r: r["R_target"])
        x      = np.array([r["R_target"]           for r in rows_sorted], dtype=float)
        t_mean = np.array([r["operadic_time_mean"]  for r in rows_sorted], dtype=float)
        t_std  = np.array([r["operadic_time_std"]   for r in rows_sorted], dtype=float)
        e_mean = np.array([r["operadic_rmse_mean"]  for r in rows_sorted], dtype=float)
        e_std  = np.array([r["operadic_rmse_std"]   for r in rows_sorted], dtype=float)

        # Time (log-log)
        plt.figure(figsize=(8, 5))
        plt.plot(x, t_mean, marker="o", label="OperadicGPAV time")
        plt.fill_between(x, t_mean - t_std, t_mean + t_std, alpha=0.2)
        plt.xscale("log"); plt.yscale("log")
        plt.xlabel("Target dataset size R"); plt.ylabel("Time (s)")
        plt.title("OperadicGPAV runtime vs R (log-log)")
        plt.legend(); plt.grid(True, which="both", alpha=0.3); plt.tight_layout()
        plt.savefig("ogpav_time_vs_R.png", dpi=220); plt.close()

        # RMSE tube
        plt.figure(figsize=(8, 5))
        plt.plot(x, e_mean, marker="o", label="OperadicGPAV RMSE")
        plt.fill_between(x, e_mean - e_std, e_mean + e_std, alpha=0.2)
        plt.xscale("log")
        plt.xlabel("Target dataset size R"); plt.ylabel("RMSE")
        plt.title("OperadicGPAV RMSE vs R (mean ± std, 3 trials)")
        plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig("ogpav_rmse_vs_R.png", dpi=220); plt.close()

        print("\nSaved plots: ogpav_time_vs_R.png, ogpav_rmse_vs_R.png")

    print(f"\nSaved: {out_csv}  {out_json}")
