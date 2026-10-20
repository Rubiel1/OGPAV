# -*- coding: utf-8 -*-
"""
experiment_ogpav.py

OperadicGPAV-only scaling study with adjacency-memory instrumentation.

Three slope configurations from Sysoev et al. (Table 1/2) are tested:
  low  (α=0.2, 0.2),  mix  (α=0.2, 2.0),  high (α=2.0, 2.0)

The expected fiber size is held at 2,000 while the number of Q nodes grows.
The noise scale is matched to that fiber-size scale.

Results are written to variant-specific ``*_adjacency_*`` CSV/JSON files for
easy resume without mixing them with earlier, non-instrumented runs.
"""

from __future__ import annotations

import csv
import gc
import json
import math
import os
import time
import traceback
import tracemalloc
from typing import Dict, List, Optional

import numpy as np
import networkx as nx
import psutil as _psutil
_proc = _psutil.Process(os.getpid())

from OperadicGPAV import OperadicGPAV
from utils.geometric_sb_dataset import (
    generate_dataset_lazy,
    make_dataset_params,
)


FIBER_SIZE_TARGET = 2_000

# Slope combinations from Sysoev et al. (Table 1/2).  These live here rather
# than in geometric_sb_dataset because the newest generator only supplies the
# geometry; it no longer exports these experiment-specific response models.
SLOPE_COEFFICIENTS = {
    "linear_low": (0.2, 0.2),
    "linear_mix": (0.2, 2.0),
    "linear_high": (2.0, 2.0),
}

# ============================================================
# Helpers  (self-contained, no import from experiment.py)
# ============================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    mse = float(np.mean(err ** 2))
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


def make_y_true_from_fibers(R_datasets, *, model: str = "linear_high") -> np.ndarray:
    try:
        alpha_1, alpha_2 = SLOPE_COEFFICIENTS[model]
    except KeyError as exc:
        choices = ", ".join(sorted(SLOPE_COEFFICIENTS))
        raise ValueError(f"Unknown model {model!r}; choose from {choices}.") from exc

    values = []
    for i in range(len(R_datasets)):
        points = np.asarray(R_datasets[i], dtype=float)
        values.append(alpha_1 * points[:, 0] + alpha_2 * points[:, 1])
    return np.concatenate(values, axis=0)


def geometry_sizes(R_target: int) -> tuple[int, int]:
    """Return the actual outer-poset and expected-fiber sizes for a target N."""
    if R_target <= 0:
        raise ValueError("R_target must be positive.")
    nQ = max(1, int(R_target / FIBER_SIZE_TARGET))
    return nQ, FIBER_SIZE_TARGET


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
    R_target: int,
    fiber_count_dist: str = "poisson",
    model: str = "linear_high",
    noise: str = "normal",
    noise_scale: float = 1.0,
    data_seed: int = 0,
    noise_seed: int = 1,
    max_workers: Optional[int] = None,
    variant: str = "default",
    verbose: bool = False,
) -> Dict[str, object]:

    # Approximate target size: E[N] = nQ * avg_R.  Keep fibers at a fixed
    # expected size and scale the experiment by increasing the size of Q.
    nQ, avg_R = geometry_sizes(R_target)

    # Base radius and spacing tightly around avg_R so disks physically fit all points
    params = make_dataset_params(avg_R)
    params["nQ"] = nQ
    params["avg_R"] = avg_R

    # We must expand the bounding box to fit nQ centers with these large radii
    required_grid_span = params["center_grid_step"] * int(math.ceil(math.sqrt(nQ)))
    params["square_max"] = max(params["square_max"], required_grid_span * 2)

    try:
        if verbose:
            print(f"    Generating R={R_target} (nQ={nQ}, avg_R={avg_R}, "
                  f"noise_scale={noise_scale:.0f}, seed={data_seed}) ...")

        # Extract only the parameters needed for geometry generation
        geom = {k: v for k, v in params.items() if k not in ("noise_scale",)}

        data = generate_dataset_lazy(
            **geom,
            seed=data_seed,
            fiber_count_dist=fiber_count_dist,
            cache_dir=None
        )

        R_datasets = data["R_points_list"]
        Q = ensure_q_nodes_match_num_fibers(data["Q_hasse"], len(R_datasets))

        lengths = R_datasets.get_fiber_lengths()
        total_n = int(sum(lengths))

        y_true  = make_y_true_from_fibers(R_datasets, model=model)
        y_noisy = y_true + make_noise(total_n, noise=noise,
                                      noise_scale=noise_scale, seed=noise_seed)

        baseline_metrics = compute_metrics(y_true, y_noisy)

        if verbose:
            print(f"    Running OperadicGPAV variant={variant!r} (N={total_n}) ...")

        tracemalloc.start()
        rss_before = _proc.memory_info().rss
        t0 = time.perf_counter()
        u_ogpav, adjacency_memory = OperadicGPAV(
            Q=Q,
            R_datasets=R_datasets,
            Y=y_noisy,
            indices_list=None,
            segment_topo_orders=None,
            assume_component_wise=True,
            variant=variant,
            max_workers=max_workers,
            verbose=False,
            debug=False,
            temp_dir=None,
            return_adjacency_memory=True,
        )
        t_ogpav = time.perf_counter() - t0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rss_after   = _proc.memory_info().rss
        mem_mb      = peak_bytes / 1024 / 1024
        rss_delta_mb = (rss_after - rss_before) / 1024 / 1024

        # Objective function: what GPAV actually minimises
        obj_ogpav = float(np.sum((np.asarray(u_ogpav, dtype=float)
                                  - np.asarray(y_noisy, dtype=float)) ** 2))

        return {
            "config": {
                "R_target": R_target,
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
                "variant": variant,
            },
            "sizes": {
                "num_fibers": len(R_datasets),
                "fiber_lengths": lengths,
                "N": total_n,
                "Q_num_edges": int(Q.number_of_edges()),
            },
            "timings_sec": {"operadic_gpav": t_ogpav},
            "memory_mb":   {"operadic_gpav": mem_mb},
            "rss_delta_mb": {"operadic_gpav": rss_delta_mb},
            "adjacency_memory": adjacency_memory,
            "objective":   {"operadic_gpav": obj_ogpav},
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
    n_trials: int = 1,
    R_target: int,
    fiber_count_dist: str = "poisson",
    model: str = "linear_high",
    noise: str = "normal",
    noise_scale: float = 1.0,
    base_seed: int = 123,
    max_workers: Optional[int] = None,
    variant: str = "default",
    verbose: bool = False,
) -> List[Dict[str, object]]:
    return [
        run_one_trial(
            R_target=R_target,
            fiber_count_dist=fiber_count_dist,
            model=model,
            noise=noise,
            noise_scale=noise_scale,
            data_seed=base_seed + trial * 2,
            noise_seed=base_seed + trial * 2 + 1,
            max_workers=max_workers,
            variant=variant,
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

    for top_key in ("timings_sec", "memory_mb", "rss_delta_mb", "objective"):
        a = arr(top_key, "operadic_gpav")
        summary[top_key] = {"operadic_gpav": {"mean": float(a.mean()), "std": float(a.std(ddof=0))}}

    summary["adjacency_memory"] = {}
    for metric in ("paper_style_kib", "paper_style_peak_edges", "csr32_kib", "csr32_mib"):
        a = np.array([r["adjacency_memory"][metric] for r in results], dtype=float)
        summary["adjacency_memory"][metric] = {
            "mean": float(a.mean()),
            "std": float(a.std(ddof=0)),
        }

    summary["adjacency_groups"] = {}
    group_names = results[0]["adjacency_memory"]["groups"].keys()
    for group_name in group_names:
        summary["adjacency_groups"][group_name] = {}
        for metric in ("max_nodes", "max_edges", "max_paper_style_bytes", "max_csr32_bytes"):
            a = np.array(
                [r["adjacency_memory"]["groups"][group_name][metric] for r in results],
                dtype=float,
            )
            summary["adjacency_groups"][group_name][metric] = {
                "mean": float(a.mean()),
                "std": float(a.std(ddof=0)),
            }

    return summary


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import argparse
    # import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Run OperadicGPAV scaling experiment.")
    parser.add_argument("--model", type=str, default=None,
                        help="Which slope config to run (linear_low, linear_mix, linear_high). "
                             "If omitted, runs all three.")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[10_000, 100_000, 1_000_000],
        help="Approximate dataset sizes to test, in increasing order.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Number of local worker processes. By default, use "
            "SLURM_CPUS_PER_TASK when running under Slurm, otherwise all "
            "CPUs visible to Python."
        ),
    )
    parser.add_argument(
        "--variant",
        choices=("default", "review", "fast"),
        default="default",
        help="OperadicGPAV variant to run (default: default).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    # Three slope combinations from Sysoev et al. (Table 1/2).
    # noise_scale matches the fixed expected fiber size used by the geometry.
    SLOPE_CONFIGS = [
        {"model": "linear_low",  "label": "low  (α=0.2, 0.2)"},
        {"model": "linear_mix",  "label": "mix  (α=0.2, 2.0)"},
        {"model": "linear_high", "label": "high (α=2.0, 2.0)"},
    ]

    if args.model:
        SLOPE_CONFIGS = [sc for sc in SLOPE_CONFIGS if sc["model"] == args.model]
        if not SLOPE_CONFIGS:
            raise ValueError(f"Unknown model {args.model!r}. Choose from linear_low, linear_mix, linear_high.")

    R_values = args.sizes
    if any(R <= 0 for R in R_values):
        raise ValueError("All --sizes values must be positive.")

    # Auto-stop if any size's average trial time exceeds this (seconds).
    # NOTE: the check happens AFTER a full trial completes — it cannot
    # interrupt a running OperadicGPAV call mid-way.
    # Set to None to disable and run all sizes regardless.
    max_allowed_seconds = None   # Disabled; set a number of seconds to enable.

    # Respect the Slurm allocation instead of accidentally launching one worker
    # for every CPU installed on the compute node.
    import multiprocessing
    multiprocessing.freeze_support()
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    allocated_cpus = int(slurm_cpus) if slurm_cpus else multiprocessing.cpu_count()
    max_workers = args.workers if args.workers is not None else allocated_cpus
    if max_workers <= 0:
        raise ValueError("--workers must be positive.")
    print(
        f"Available/allocated CPUs: {allocated_cpus} "
        f"→ using max_workers={max_workers}"
    )

    # ------------------------------------------------------------------
    # Outer loop over slope configurations
    # ------------------------------------------------------------------
    all_slope_rows: Dict[str, List[dict]] = {}

    for slope_cfg in SLOPE_CONFIGS:
        model_key = slope_cfg["model"]
        label     = slope_cfg["label"]

        out_csv  = f"scaling_ogpav_{args.variant}_adjacency_{model_key}.csv"
        out_json = f"scaling_ogpav_{args.variant}_adjacency_{model_key}.json"

        print(f"\n{'#'*80}")
        print(f"# Slope config: {label}")
        print(f"{'#'*80}")

        rows: List[dict]     = []
        failures: List[dict] = []
        completed_R: set     = set()

        if os.path.exists(out_json):
            try:
                with open(out_json, "r", encoding="utf-8") as f:
                    old = json.load(f)
                rows        = old.get("results", [])
                failures    = old.get("failures", [])
                completed_R = {row["R_target"] for row in rows}
                print(f"  Loaded prior progress: {len(rows)} done, {len(failures)} failed.")
            except Exception as e:
                print(f"  Warning: could not load {out_json}: {e}")

        def _save(rows, failures, _csv=out_csv, _json=out_json):
            if rows:
                with open(_csv, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
            with open(_json, "w", encoding="utf-8") as f:
                json.dump({"results": rows, "failures": failures}, f, indent=2)

        stop_slope = False

        for R in R_values:
            if stop_slope:
                break
            if R in completed_R:
                print(f"  Skipping already completed R={R:,}")
                continue

            nQ, avg_R = geometry_sizes(R)

            # Keep noise linearly scaling with the average spread of values
            noise_scale = float(avg_R)

            print(f"\n  {'='*66}")
            print(
                f"  R={R:,}  nQ={nQ}  avg_R={avg_R}  "
                f"noise_scale={noise_scale:.0f}  model={model_key}"
            )

            try:
                results = run_repeated_experiment(
                    n_trials=1,
                    R_target=R,
                    fiber_count_dist="poisson",
                    model=model_key,
                    noise="normal",
                    noise_scale=noise_scale,
                    base_seed=2026 + nQ,
                    max_workers=max_workers,
                    variant=args.variant,
                    verbose=True,
                )

                summary          = summarize_results(results)
                realized_N       = [int(r["sizes"]["N"])           for r in results]
                realized_Q_edges = [int(r["sizes"]["Q_num_edges"])  for r in results]

                baseline_rmse = summary["baseline_noisy_vs_truth"]["rmse"]["mean"]
                operadic_rmse = summary["operadic_vs_truth"]["rmse"]["mean"]
                rel_imp       = 1.0 - operadic_rmse / max(baseline_rmse, 1e-12)

                # Compute actual mean N first so we can use it in normalisation
                N_actual_mean = float(np.mean(realized_N))

                row = {
                    "slope_config":         model_key,
                    "variant":              args.variant,
                    "R_target":             int(R),
                    "nQ_input":             nQ,
                    "avg_R_input":          avg_R,
                    "noise_scale":          noise_scale,
                    "N_actual_mean":        N_actual_mean,
                    "N_actual_std":         float(np.std(realized_N, ddof=0)),
                    "Q_edges_mean":         float(np.mean(realized_Q_edges)),
                    "Q_edges_std":          float(np.std(realized_Q_edges, ddof=0)),
                    "operadic_time_mean":   float(summary["timings_sec"]["operadic_gpav"]["mean"]),
                    "operadic_time_std":    float(summary["timings_sec"]["operadic_gpav"]["std"]),
                    "operadic_mem_mb_mean": float(summary["memory_mb"]["operadic_gpav"]["mean"]),
                    "operadic_mem_mb_std":  float(summary["memory_mb"]["operadic_gpav"]["std"]),
                    "operadic_rss_mb_mean": float(summary["rss_delta_mb"]["operadic_gpav"]["mean"]),
                    "operadic_rss_mb_std":  float(summary["rss_delta_mb"]["operadic_gpav"]["std"]),
                    # Paper-style adjacency storage is distinct from whole-process RAM.
                    # The paper does not state its sparse-matrix byte layout, so we
                    # save both the one-byte/nonzero reconstruction and CSR32 estimate.
                    "paper_style_adj_kib_mean": float(summary["adjacency_memory"]["paper_style_kib"]["mean"]),
                    "paper_style_adj_kib_std":  float(summary["adjacency_memory"]["paper_style_kib"]["std"]),
                    "paper_style_adj_peak_edges_mean": float(summary["adjacency_memory"]["paper_style_peak_edges"]["mean"]),
                    "paper_style_adj_peak_graph": results[0]["adjacency_memory"]["paper_style_peak_graph"],
                    "csr32_adj_kib_mean": float(summary["adjacency_memory"]["csr32_kib"]["mean"]),
                    "csr32_adj_kib_std":  float(summary["adjacency_memory"]["csr32_kib"]["std"]),
                    "csr32_adj_peak_graph": results[0]["adjacency_memory"]["csr32_peak_graph"],
                    "local_hasse_peak_edges_mean": float(summary["adjacency_groups"]["local_hasse"]["max_edges"]["mean"]),
                    "local_block_peak_edges_mean": float(summary["adjacency_groups"]["local_block"]["max_edges"]["mean"]),
                    "outer_hasse_edges_mean": float(summary["adjacency_groups"]["outer_hasse"]["max_edges"]["mean"]),
                    "global_block_edges_mean": float(summary["adjacency_groups"]["global_block"]["max_edges"]["mean"]),
                    "operadic_obj_mean":    float(summary["objective"]["operadic_gpav"]["mean"]),
                    "operadic_obj_std":     float(summary["objective"]["operadic_gpav"]["std"]),
                    "baseline_rmse_mean":  baseline_rmse,
                    "baseline_rmse_std":   float(summary["baseline_noisy_vs_truth"]["rmse"]["std"]),
                    "operadic_rmse_mean":  operadic_rmse,
                    "operadic_rmse_std":   float(summary["operadic_vs_truth"]["rmse"]["std"]),
                    "operadic_mse_mean":   float(summary["operadic_vs_truth"]["mse"]["mean"]),
                    "operadic_mse_std":    float(summary["operadic_vs_truth"]["mse"]["std"]),
                    "operadic_mae_mean":   float(summary["operadic_vs_truth"]["mae"]["mean"]),
                    "operadic_mae_std":    float(summary["operadic_vs_truth"]["mae"]["std"]),
                    "rel_rmse_improvement": float(rel_imp),
                    # --- Normalised metrics (should be ~constant across q) ---
                    "obj_normalised":       float(summary["objective"]["operadic_gpav"]["mean"]
                                                  / max(N_actual_mean * noise_scale**2, 1e-12)),
                    "rmse_normalised":      float(operadic_rmse / max(noise_scale, 1e-12)),
                    "baseline_rmse_normalised": float(baseline_rmse / max(noise_scale, 1e-12)),
                }

                rows.append(row)
                _save(rows, failures)

                print(
                    f"  Done | N={row['N_actual_mean']:.0f} | "
                    f"time={row['operadic_time_mean']:.3f}s | "
                    f"tracemalloc={row['operadic_mem_mb_mean']:.1f}MB | "
                    f"RSS_delta={row['operadic_rss_mb_mean']:.1f}MB | "
                    f"paper_style_adj={row['paper_style_adj_kib_mean']:.1f}KiB "
                    f"({row['paper_style_adj_peak_graph']}) | "
                    f"CSR32_adj={row['csr32_adj_kib_mean'] / 1024.0:.1f}MiB | "
                    f"obj={row['operadic_obj_mean']:.4g} (norm={row['obj_normalised']:.3f}) | "
                    f"RMSE={operadic_rmse:.4g} (norm={row['rmse_normalised']:.3f}) | "
                    f"baseline={baseline_rmse:.4g} (norm={row['baseline_rmse_normalised']:.3f}) | "
                    f"improvement={rel_imp:.1%}"
                )

                if max_allowed_seconds and row["operadic_time_mean"] > max_allowed_seconds:
                    print("  Stopping slope config: runtime limit reached.")
                    stop_slope = True

            except Exception as e:
                failures.append({
                    "slope_config":  model_key,
                    "variant":       args.variant,
                    "R_target":      int(R),
                    "nQ":            nQ,
                    "avg_R":         avg_R,
                    "error":         str(e),
                    "traceback":     traceback.format_exc(),
                })
                _save(rows, failures)
                print(f"  FAILED at R={R:,}: {e}")
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

    # def slope_plot(mean_key: str, std_key: str, ylabel: str,
    #               fname: str, logy: bool = False):
    #    plt.figure(figsize=(9, 5))
    #    has_data = False
    #    for sc in SLOPE_CONFIGS:
    #        key = sc["model"]
    #        rs  = sorted(all_slope_rows.get(key, []), key=lambda r: r["R_target"])
    #        if not rs:
    #            continue
    #        has_data = True
    #        x    = np.array([r["R_target"] for r in rs], dtype=float)
    #        mean = np.array([r[mean_key]   for r in rs], dtype=float)
    #        std  = np.array([r[std_key]    for r in rs], dtype=float)
    #        plt.plot(x, mean, marker="o", label=sc["label"], color=colors[key])
    #        plt.fill_between(x, mean - std, mean + std, alpha=0.15, color=colors[key])
    #    if not has_data:
    #        plt.close()
    #        return
    #    plt.xscale("log")
    #    if logy:
    #        plt.yscale("log")
    #    plt.xlabel("Target dataset size R")
    #    plt.ylabel(ylabel)
    #    plt.title(f"OperadicGPAV default — {ylabel} vs R")
    #    plt.legend()
    #    plt.grid(True, which="both" if logy else "major", alpha=0.3)
    #    plt.tight_layout()
    #    plt.savefig(fname, dpi=220)
    #    plt.close()
    #    print(f"Saved {fname}")

    # slope_plot("operadic_time_mean",   "operadic_time_std",
    #           "Time (s)",     "ogpav_time_vs_R.png",        logy=True)
    # slope_plot("operadic_rmse_mean",   "operadic_rmse_std",
    #            "RMSE",         "ogpav_rmse_vs_R.png")
    # slope_plot("baseline_rmse_mean",   "baseline_rmse_std",
    #           "Baseline RMSE (noisy signal)", "ogpav_baseline_vs_R.png")
    # slope_plot("rel_rmse_improvement", "operadic_rmse_std",
    #            "Relative RMSE improvement", "ogpav_improvement_vs_R.png")

    print("\nAll done.")
