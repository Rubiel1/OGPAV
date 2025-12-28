# -*- coding: utf-8 -*-
"""
aux_dataset_segmentation.py

Auxiliary dataset generator for:
  - the synthetic 2D datasets described in *A Segmentation Based Algorithm for Large Scale* (Sysoev et al.)
  - your operadic GPAV pipeline (lexicographic sum P = Q(R_1,...,R_m)) in operadic_gpav.py
  - your segmented GPAV pipeline in segmented_gpav.py

What it generates
-----------------
1) Q centers q_i:
   - integer coordinate points sampled uniformly from the square [0,100]^2 (customizable),
     with uniqueness enforced.
2) Fibers R_i around each q_i:
   - points sampled uniformly in the disk centered at q_i with radius `radius` (default 1/3)
   - points that are too close (min pairwise distance < `min_dist`) are pruned greedily
3) Posets (as NetworkX Hasse DAGs under strict dominance):
   - Q_hasse: Hasse diagram on the q_i nodes
   - R_hasse_list: Hasse diagrams within each fiber R_i
   - P_hasse: Hasse diagram on the full set X = ⨆_i R_i (optional but useful for segmented_gpav)

4) Observations y = f(x) + noise for several choices of f:
   - linear:      f(x)=a1*x1 + a2*x2
   - quadratic:   f(x)=a1*x1^2 + a2*x2^2
   - sinusoid:    f(x)=a1*sin(w1*x1) + a2*cos(w2*x2)
   - radial:      f(x)=a1*sqrt((x1-c1)^2+(x2-c2)^2)
   - saddle:      f(x)=a1*x1 - a2*x2

The output is a plain Python dict so you can use it in tests, save it, or convert to
your hasse.PoSet objects if/when available.

Notes on orders
---------------
We use **strict dominance** to keep a DAG even if there are ties:
    u ≺ v  iff  u_k <= v_k for all k, and u != v.
Cover relations are extracted via transitive reduction.

Plotting helpers
----------------
- plot_geometry: scatter q_i and all x points; draw a radius disk around each q_i; optionally show r_i label
- plot_3d: 3D scatter (x1,x2,y) for a chosen model
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import math
import numpy as np
import networkx as nx


ArrayLike = Union[np.ndarray, Sequence[float]]


# ---------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------

def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


def sample_integer_centers(
    nQ: int,
    *,
    square_min: int = 0,
    square_max: int = 100,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Sample nQ distinct integer points in [square_min, square_max]^2.

    Returns
    -------
    q : (nQ,2) array of ints
    """
    if nQ <= 0:
        raise ValueError("nQ must be positive.")
    side = square_max - square_min + 1
    if nQ > side * side:
        raise ValueError("nQ exceeds number of integer lattice points in the square.")
    rg = _rng(seed)
    # Sample without replacement from flattened grid indices
    idx = rg.choice(side * side, size=nQ, replace=False)
    xs = idx % side
    ys = idx // side
    q = np.stack([xs + square_min, ys + square_min], axis=1).astype(int)
    return q


def sample_points_in_disk(
    center: np.ndarray,
    k: int,
    *,
    radius: float = 1/3,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Sample k points uniformly at random in a disk (2D).
    """
    if k <= 0:
        return np.zeros((0, 2), dtype=float)
    rg = _rng(seed)
    # Polar: r = R*sqrt(U), theta = 2πV
    u = rg.random(k)
    v = rg.random(k)
    r = radius * np.sqrt(u)
    theta = 2.0 * math.pi * v
    pts = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    return pts + center.reshape(1, 2)


def prune_too_close(
    pts: np.ndarray,
    *,
    min_dist: float = 0.02,
) -> np.ndarray:
    """
    Greedy pruning: keep points in input order, drop any point within min_dist
    of an already kept point.
    """
    if pts.shape[0] <= 1:
        return pts
    keep: List[np.ndarray] = []
    md2 = float(min_dist) ** 2
    for p in pts:
        ok = True
        for q in keep:
            if float(np.sum((p - q) ** 2)) < md2:
                ok = False
                break
        if ok:
            keep.append(p)
    if not keep:
        return np.zeros((0, 2), dtype=float)
    return np.stack(keep, axis=0)


# ---------------------------------------------------------------------
# Poset helpers (strict dominance, then transitive reduction)
# ---------------------------------------------------------------------

def strict_dominance_edges(points: np.ndarray) -> List[Tuple[int, int]]:
    """
    Edges (i -> j) for strict dominance: points[i] <= points[j] coordinatewise
    and not equal. This is the full DAG, NOT yet reduced.
    """
    n = points.shape[0]
    edges: List[Tuple[int, int]] = []
    for i in range(n):
        pi = points[i]
        # vectorized compare against all j
        le = np.all(pi <= points, axis=1)
        neq = np.any(pi != points, axis=1)
        js = np.where(le & neq)[0]
        for j in js:
            edges.append((i, int(j)))
    return edges


def hasse_from_points(points: np.ndarray) -> nx.DiGraph:
    """
    Build Hasse diagram (transitive reduction) of strict-dominance order
    on points indexed 0..n-1.
    """
    n = points.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    edges = strict_dominance_edges(points)
    G.add_edges_from(edges)
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Constructed dominance graph is not a DAG. Check for NaNs.")
    if G.number_of_edges() > 0:
        G = nx.transitive_reduction(G)
    return G


# ---------------------------------------------------------------------
# Observation models
# ---------------------------------------------------------------------

def f_linear(x: np.ndarray, a1: float = 1.0, a2: float = 1.0) -> np.ndarray:
    return a1 * x[:, 0] + a2 * x[:, 1]


def f_quadratic(x: np.ndarray, a1: float = 1.0, a2: float = 1.0) -> np.ndarray:
    return a1 * (x[:, 0] ** 2) + a2 * (x[:, 1] ** 2)


def f_sinusoid(
    x: np.ndarray, a1: float = 1.0, a2: float = 1.0, w1: float = 0.25, w2: float = 0.25
) -> np.ndarray:
    return a1 * np.sin(w1 * x[:, 0]) + a2 * np.cos(w2 * x[:, 1])


def f_radial(
    x: np.ndarray, a1: float = 1.0, c1: float = 50.0, c2: float = 50.0
) -> np.ndarray:
    return a1 * np.sqrt((x[:, 0] - c1) ** 2 + (x[:, 1] - c2) ** 2)


def f_saddle(x: np.ndarray, a1: float = 1.0, a2: float = 1.0) -> np.ndarray:
    return a1 * x[:, 0] - a2 * x[:, 1]


MODEL_FUNCS: Dict[str, Callable[..., np.ndarray]] = {
    "linear": f_linear,
    "quadratic": f_quadratic,
    "sinusoid": f_sinusoid,
    "radial": f_radial,
    "saddle": f_saddle,
}


def make_observations(
    X: np.ndarray,
    *,
    model: str = "linear",
    noise: str = "normal",
    noise_scale: float = 1.0,
    seed: Optional[int] = None,
    **model_kwargs,
) -> np.ndarray:
    """
    Create y = f(X) + eps.

    noise:
      - "normal": N(0, noise_scale^2)
      - "laplace": Laplace(0, noise_scale)
      - "none": zeros
    """
    if model not in MODEL_FUNCS:
        raise ValueError(f"Unknown model {model!r}. Choose from {sorted(MODEL_FUNCS)}")
    f = MODEL_FUNCS[model]
    y = f(X, **model_kwargs).astype(float)
    rg = _rng(seed)
    if noise == "none":
        eps = np.zeros_like(y)
    elif noise == "normal":
        eps = rg.normal(0.0, noise_scale, size=y.shape[0])
    elif noise == "laplace":
        eps = rg.laplace(0.0, noise_scale, size=y.shape[0])
    else:
        raise ValueError("noise must be one of {'normal','laplace','none'}")
    return y + eps

def make_observations_dict(
    X: np.ndarray,
    *,
    model: str = "linear",
    noise: str = "normal",
    noise_scale: float = 1.0,
    seed: Optional[int] = None,
    **model_kwargs,
) -> Dict[int, float]:
    """
    Return observations as a dict: node_label -> y_value.
    Node labels are assumed to be 0..N-1 aligned with X rows.
    """
    y = make_observations(
        X,
        model=model,
        noise=noise,
        noise_scale=noise_scale,
        seed=seed,
        **model_kwargs,
    )
    return {int(i): float(y[i]) for i in range(len(y))}

# ---------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------

def generate_q_and_fibers(
    *,
    nQ: int,
    avg_R: int,
    radius: float = 1/3,
    min_dist: float = 0.02,
    square_min: int = 0,
    square_max: int = 100,
    seed: Optional[int] = None,
    fiber_count_dist: str = "poisson",
) -> Dict[str, object]:
    """
    Generate Q centers and fibers R_i, plus their Hasse DAGs.

    Parameters
    ----------
    nQ : number of q_i centers (size of Q)
    avg_R : average number of raw (pre-prune) points per fiber
    radius : disk radius around each q_i (default 1/3)
    min_dist : pruning threshold inside each fiber
    fiber_count_dist : "poisson" or "uniform"
        - poisson: k_i ~ Poisson(avg_R) + 1
        - uniform: k_i ~ Uniform{1,...,2*avg_R-1} (clipped to >=1)

    Returns dict with keys:
      q_points: (nQ,2) int array
      r_i:      float array length nQ (all radius, but stored per-center for convenience)
      R_points_list: list of (n_i,2) float arrays
      X: (N,2) float array of all points (concatenation)
      fiber_of: (N,) int array mapping each X row -> fiber index i
      Q_hasse: nx.DiGraph on nodes 0..nQ-1
      R_hasse_list: list of nx.DiGraph (each on nodes 0..n_i-1)
      P_hasse: nx.DiGraph on nodes 0..N-1
    """
    rg = _rng(seed)
    q = sample_integer_centers(nQ, square_min=square_min, square_max=square_max, seed=seed)

    R_list: List[np.ndarray] = []
    fiber_of: List[int] = []
    # independent seeds per fiber for reproducibility
    seeds = rg.integers(0, 2**32 - 1, size=nQ, dtype=np.uint32)

    for i in range(nQ):
        if fiber_count_dist == "poisson":
            k = int(rg.poisson(lam=max(1, avg_R))) + 1
        elif fiber_count_dist == "uniform":
            hi = max(1, 2 * avg_R - 1)
            k = int(rg.integers(1, hi + 1))
        else:
            raise ValueError("fiber_count_dist must be 'poisson' or 'uniform'")

        pts = sample_points_in_disk(q[i].astype(float), k, radius=radius, seed=int(seeds[i]))
        pts = prune_too_close(pts, min_dist=min_dist)
        R_list.append(pts)
        fiber_of.extend([i] * pts.shape[0])

    if len(fiber_of) == 0:
        X = np.zeros((0, 2), dtype=float)
        fiber_of_arr = np.zeros((0,), dtype=int)
    else:
        X = np.concatenate(R_list, axis=0)
        fiber_of_arr = np.asarray(fiber_of, dtype=int)

    Q_hasse = hasse_from_points(q.astype(float))
    R_hasse_list = [hasse_from_points(pts) for pts in R_list]
    P_hasse = hasse_from_points(X) if X.shape[0] > 0 else nx.DiGraph()

    return dict(
        q_points=q,
        r_i=np.full((nQ,), float(radius), dtype=float),
        R_points_list=R_list,
        X=X,
        fiber_of=fiber_of_arr,
        Q_hasse=Q_hasse,
        R_hasse_list=R_hasse_list,
        P_hasse=P_hasse,

        # NEW (empty by default; filled by helper)
        Y_dict=None,
        Y_array=None,
    )


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_geometry(
    data: Dict[str, object],
    *,
    show_r_labels: bool = True,
    figsize: Tuple[float, float] = (7, 7),
):
    """
    2D plot: q_i centers, all x points, and radius disks.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    q = np.asarray(data["q_points"])
    X = np.asarray(data["X"])
    r_i = np.asarray(data["r_i"])

    fig, ax = plt.subplots(figsize=figsize)
    if X.size:
        ax.scatter(X[:, 0], X[:, 1], s=10, alpha=0.7, label="x points (⋃ R_i)")
    ax.scatter(q[:, 0], q[:, 1], s=60, marker="x", label="q centers (Q)")

    for i in range(q.shape[0]):
        circ = Circle((q[i, 0], q[i, 1]), r_i[i], fill=False, linewidth=1.0, alpha=0.7)
        ax.add_patch(circ)
        if show_r_labels:
            ax.text(q[i, 0] + r_i[i], q[i, 1] + r_i[i], f"r={r_i[i]:.3g}", fontsize=8)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(min(0, q[:, 0].min() - 1), max(100, q[:, 0].max() + 1))
    ax.set_ylim(min(0, q[:, 1].min() - 1), max(100, q[:, 1].max() + 1))
    ax.set_title("Q centers, fibers, and radius disks")
    ax.legend(loc="best")
    ax.grid(True, linewidth=0.3, alpha=0.4)
    plt.show()
    return fig, ax


def plot_3d(
    X: np.ndarray,
    y: np.ndarray,
    *,
    title: str = "",
    figsize: Tuple[float, float] = (8, 6),
):
    """
    3D scatter plot for (x1, x2, y).
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X[:, 0], X[:, 1], y, s=10, alpha=0.8)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")
    ax.set_title(title or "3D scatter: (x1, x2, y)")
    plt.show()
    return fig, ax


# ---------------------------------------------------------------------
# Convenience: make several models and plot them
# ---------------------------------------------------------------------

def make_and_plot_models(
    data: Dict[str, object],
    *,
    models: Sequence[str] = ("linear", "quadratic", "sinusoid", "radial", "saddle"),
    noise: str = "normal",
    noise_scale: float = 1.0,
    seed: Optional[int] = None,
):
    X = np.asarray(data["X"])
    out: Dict[str, np.ndarray] = {}
    for m in models:
        y = make_observations(X, model=m, noise=noise, noise_scale=noise_scale, seed=seed)
        out[m] = y
        plot_3d(X, y, title=f"model={m}, noise={noise} (scale={noise_scale})")
    return out

def attach_observations(
    data: Dict[str, object],
    *,
    y: Optional[np.ndarray] = None,
    Y_dict: Optional[Dict[int, float]] = None,
    model: str = "linear",
    noise: str = "normal",
    noise_scale: float = 1.0,
    seed: Optional[int] = None,
    **model_kwargs,
) -> Dict[str, object]:
    """
    Attach Y_dict and Y_array to an existing dataset dict.

    You may either:
      - provide y (array aligned with X rows), or
      - provide Y_dict (mapping i -> y_i), or
      - omit both and generate using (model, noise, noise_scale, seed, **model_kwargs).
    """
    X = np.asarray(data["X"])
    n = X.shape[0]

    if y is not None and Y_dict is not None:
        raise ValueError("Provide only one of y or Y_dict, not both.")

    if y is not None:
        y = np.asarray(y, dtype=float)
        if y.ndim != 1:
            raise ValueError(f"y must be 1D; got shape {y.shape!r}")
        if y.shape[0] != n:
            raise ValueError(f"y length {y.shape[0]} does not match X rows {n}")
        Y_dict = {int(i): float(y[i]) for i in range(n)}
        data["Y_dict"] = Y_dict
        data["Y_array"] = y.copy()
        return data

    if Y_dict is not None:
        if len(Y_dict) != n:
            raise ValueError(f"Y_dict has length {len(Y_dict)} but X has {n} rows")
        data["Y_dict"] = {int(i): float(Y_dict[i]) for i in range(n)}
        data["Y_array"] = np.array([data["Y_dict"][i] for i in range(n)], dtype=float)
        return data

    # Default behavior: generate observations using the chosen model/noise.
    Y_dict = make_observations_dict(
        X,
        model=model,
        noise=noise,
        noise_scale=noise_scale,
        seed=seed,
        **model_kwargs,
    )
    data["Y_dict"] = Y_dict
    data["Y_array"] = np.array([Y_dict[i] for i in range(len(Y_dict))], dtype=float)
    return data


if __name__ == "__main__":
    # Quick smoke test / demo
    data = generate_q_and_fibers(nQ=20, avg_R=40, seed=0, radius=1/3, min_dist=0.02)
    plot_geometry(data)
    _ = make_and_plot_models(data, noise_scale=0.5, seed=1)
