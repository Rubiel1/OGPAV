# -*- coding: utf-8 -*-
"""
review.py

Benchmark helpers for comparing OperadicGPAV with SB-GPAV under the same rules
("review" setting). Use these when reporting results against
Sysoev, Burdakov & Grimvall (2011).

The rules, applied to both algorithms:

* Same building blocks: the same Hasse/DAG builder (_build_dag_incrementally),
  the same GPAV core (gpav_seg) and the same LowerY implementation.
* First order, trend-following (the paper's Algorithm 5, LowerY):
    - SB-GPAV: LowerY over ALL N elements of the lexicographic sum P = Q(R_1..R_m),
      as Algorithm 4, step 1 recommends. This needs the Hasse diagram of P; building
      it and computing LowerY are timed as part of SB-GPAV.
    - OperadicGPAV: LowerY inside each fiber R_i (variant="review").
* Second order, on the block graph: trend-following (LowerY on the block values)
  for both. The paper allows any topological order in this step; we use LowerY
  because, measured against the exact optimum, it is clearly more accurate than
  a Y-blind topological sort, so SB-GPAV is run in its most accurate form.
* OperadicGPAV builds its block graph as in variant="default": a Q-edge may be routed
  through one weight-0 gateway node when that needs fewer edges. This is part of
  OperadicGPAV's own construction (SB-GPAV has no Q); in every test it left the fit unchanged.

OperadicGPAV's advantages measured this way come from the algorithm itself:
segmenting along Q, running LowerY and the first GPAV per fiber, the Stage-2
max/min shortcut, and (optionally) parallel Stage 1.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from .sb_gpav import sb_gpav
from .trend_following import _build_dag_incrementally, default_comparator, trend_following_order


def lexicographic_sum(
    Q: nx.DiGraph,
    R_datasets: Sequence[Any],
    f: Optional[Callable[[Any, Any], bool]] = None,
) -> Tuple[List[Tuple[int, Any]], Callable[[Any, Any], bool], List[int]]:
    """
    Materialise P = Q(R_0, ..., R_{m-1}) for algorithms that need the whole poset.

    Returns
    -------
    X : list of (fiber index, element)
        Element k of X corresponds to Y[k] under the default lexicographic mapping
        (fiber 0 first, then fiber 1, ...).
    f_P : comparator on X
        (i, x) <= (j, y) iff i = j and f(x, y), or i < j in Q.
    linear_extension : list of int
        A topological order of P (Q-rank of the fiber, then coordinate sum), so the
        DAG can be built with the incremental builder.
    """
    if f is not None:
        raise ValueError(
            "lexicographic_sum currently supports the coordinate-wise order only "
            "(f=None): the linear extension uses coordinate sums."
        )
    f_inner = default_comparator
    reach = {i: nx.descendants(Q, i) for i in Q.nodes()}
    X = [(i, np.asarray(r)[k]) for i, r in enumerate(R_datasets) for k in range(len(r))]

    def f_P(a, b):
        if a[0] == b[0]:
            return bool(f_inner(a[1], b[1]))
        return b[0] in reach[a[0]]

    rank = {v: k for k, v in enumerate(nx.lexicographical_topological_sort(Q))}
    linear_extension = sorted(range(len(X)), key=lambda t: (rank[X[t][0]], float(np.sum(X[t][1])), t))
    return X, f_P, linear_extension


def sb_gpav_review(
    Q: nx.DiGraph,
    R_datasets: Sequence[Any],
    Y: Sequence[float],
    n_segments: Optional[int] = None,
    weights: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    SB-GPAV on the lexicographic sum, under the review rules (module docstring).

    Parameters
    ----------
    Q, R_datasets, Y : as for OperadicGPAV (coordinate-wise order inside fibers,
        Y concatenated fiber by fiber).
    n_segments : int, optional
        Number of SB segments. Defaults to m = number of fibers, so both algorithms
        split the data into the same number of pieces.
    weights : array-like of length N, optional
        Observation weights, aligned with Y (default: all 1).

    Returns
    -------
    u : np.ndarray
        Fitted values aligned with Y.
    timings : dict
        Seconds spent in "hasse" (DAG of P), "order" (LowerY on P), "sb" (SB-GPAV
        proper, including LowerY on its block graph) and "total".
    """
    Y = np.asarray(Y, dtype=float)
    m = Q.number_of_nodes()
    if n_segments is None:
        n_segments = m
    t0 = time.perf_counter()
    X, f_P, lin = lexicographic_sum(Q, R_datasets)
    if len(X) != len(Y):
        raise ValueError(f"Y has length {len(Y)} but the fibers have {len(X)} elements.")
    G = _build_dag_incrementally(lin, lambda a, b: f_P(X[a], X[b]), True)
    G.add_nodes_from(range(len(X)))
    t1 = time.perf_counter()
    L = trend_following_order(G=G, Y={k: float(Y[k]) for k in range(len(X))}, sparse_data=False)
    t2 = time.perf_counter()
    u = sb_gpav(X, Y, L, f_P, weights=weights, n_segments=n_segments, assume_component_wise=True,
                block_order="trend")
    t3 = time.perf_counter()
    return u, {"hasse": t1 - t0, "order": t2 - t1, "sb": t3 - t2, "total": t3 - t0}


def compare_review(
    Q: nx.DiGraph,
    R_datasets: Sequence[Any],
    Y: Sequence[float],
    n_segments: Optional[int] = None,
    max_workers: int = 1,
    weights: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Run OperadicGPAV(variant="review") and sb_gpav_review on the same input.

    Returns a dict with the two fits, their (weighted) SSE, and wall-clock times. Use
    max_workers=1 for the algorithmic comparison; report parallel runs separately.
    """
    from OperadicGPAV import OperadicGPAV

    Y = np.asarray(Y, dtype=float)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # default index mapping warning
        u_og = OperadicGPAV(Q=Q, R_datasets=R_datasets, Y=Y, assume_component_wise=True,
                            max_workers=max_workers, variant="review", weights=weights)
    t_og = time.perf_counter() - t0
    u_sb, t_sb = sb_gpav_review(Q, R_datasets, Y, n_segments=n_segments, weights=weights)
    w = np.ones_like(Y) if weights is None else np.asarray(weights, dtype=float)
    return {
        "u_ogpav": u_og, "u_sb": u_sb,
        "sse_ogpav": float(np.sum(w * (u_og - Y) ** 2)), "sse_sb": float(np.sum(w * (u_sb - Y) ** 2)),
        "time_ogpav": t_og, "time_sb": t_sb["total"], "time_sb_parts": t_sb,
    }
