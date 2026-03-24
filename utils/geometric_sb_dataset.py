# -*- coding: utf-8 -*-
"""
geometric_sb_dataset_grid.py

Auxiliary dataset generator for:
  - the synthetic 2D datasets described in *A Segmentation Based Algorithm for Large Scale* (Sysoev et al.)
  - your operadic GPAV pipeline (lexicographic sum P = Q(R_1,...,R_m)) in operadic_gpav.py
  - your segmented GPAV pipeline in segmented_gpav.py

Grid-based generation strategy (fast, no pruning needed)
---------------------------------------------------------
1) Q centers q_i:
   - Sampled from a regular integer grid with step `center_grid_step` (default 1).
   - By choosing center_grid_step >= ceil(2 * radius), the fiber disks are
     guaranteed to be non-overlapping without any greedy pruning.
   - `nQ` centers are selected uniformly at random from the valid grid points.

2) Fibers R_i around each q_i:
   - A mini regular grid with spacing `fiber_grid_step` (= old min_dist, default 0.02)
     is laid over a bounding square around each q_i.
   - Candidate points strictly inside the disk (distance <= radius) are retained.
   - `k` points are drawn uniformly at random from those candidates (no pruning needed,
     since all candidates are already at least fiber_grid_step apart).

3) Posets (as NetworkX Hasse DAGs under strict dominance):
   - Q_hasse: Hasse diagram on the q_i nodes
   - R_hasse_list: Hasse diagrams within each fiber R_i
   - P_hasse: Hasse diagram on the full set X = ⨆_i R_i (optional)

4) Observations y = f(x) + noise for several choices of f:
   - nonlinear:      g(x1)+g(x2) where g(t)=t^(1/3) if t<=0, t^3 if t>0
   - linear_weak:    0.1*x1 + 0.1*x2
   - linear_strong:  x1 + x2

This module supports **Lazy Generation** to disk to handle large N without RAM spikes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union, Any, Iterator
import math
import numpy as np
import networkx as nx
import os
import pickle
import tempfile
import shutil

ArrayLike = Union[np.ndarray, Sequence[float]]

# ---------------------------------------------------------------------
# RNG helper
# ---------------------------------------------------------------------

def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------
# Q center sampling (coarse grid)
# ---------------------------------------------------------------------

def sample_grid_centers(
    nQ: int,
    *,
    square_min: int = 0,
    square_max: int = 100,
    center_grid_step: int = 1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Sample nQ distinct grid points from a regular lattice in [square_min, square_max]^2.

    The lattice has step `center_grid_step` in both dimensions.  Setting
    center_grid_step >= ceil(2 * radius) guarantees that the fiber disks are
    non-overlapping without any additional pruning.

    Parameters
    ----------
    nQ : int
        Number of Q centers to sample.
    square_min, square_max : int
        Lattice bounding box (inclusive).
    center_grid_step : int
        Step size of the coarse grid (in the same integer units as square_min/max).
        Default 1 reproduces the original integer-point sampling.
    seed : int, optional
        RNG seed.

    Returns
    -------
    np.ndarray, shape (nQ, 2), dtype int
    """
    if nQ <= 0:
        raise ValueError("nQ must be positive.")
    if center_grid_step < 1:
        raise ValueError("center_grid_step must be >= 1.")

    xs = np.arange(square_min, square_max + 1, center_grid_step, dtype=int)
    ys = np.arange(square_min, square_max + 1, center_grid_step, dtype=int)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    candidates = np.stack([gx.ravel(), gy.ravel()], axis=1)  # (M, 2)

    if nQ > len(candidates):
        raise ValueError(
            f"nQ={nQ} exceeds the number of grid points ({len(candidates)}) "
            f"in [{square_min},{square_max}]^2 with step={center_grid_step}. "
            f"Reduce nQ, decrease center_grid_step, or increase square_max."
        )

    rg = _rng(seed)
    idx = rg.choice(len(candidates), size=nQ, replace=False)
    return candidates[idx]


# Keep old name as an alias so existing imports don't break.
def sample_integer_centers(
    nQ: int,
    *,
    square_min: int = 0,
    square_max: int = 100,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Backward-compatible alias for sample_grid_centers with step=1."""
    return sample_grid_centers(nQ, square_min=square_min, square_max=square_max,
                               center_grid_step=1, seed=seed)


# ---------------------------------------------------------------------
# Fiber point sampling (mini grid inside disk)
# ---------------------------------------------------------------------

def _build_disk_grid(radius: float, grid_step: float) -> np.ndarray:
    """
    Return local offsets (relative to center) of grid points inside a disk.

    The grid has spacing `grid_step` in both dimensions.  Only points with
    Euclidean distance <= radius are included.  Since all retained points are
    grid-spaced, no further pruning is needed.
    """
    n = int(math.ceil(radius / grid_step))
    offsets_1d = np.arange(-n, n + 1) * grid_step
    gx, gy = np.meshgrid(offsets_1d, offsets_1d, indexing="xy")
    local = np.stack([gx.ravel(), gy.ravel()], axis=1)           # (M, 2)
    dists_sq = (local ** 2).sum(axis=1)
    return local[dists_sq <= radius * radius]                     # filter to disk


def sample_points_in_disk_grid(
    center: np.ndarray,
    k: int,
    *,
    radius: float = 1 / 3,
    fiber_grid_step: float = 0.02,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Sample k distinct points from a regular grid inside a disk – no pruning needed.

    A mini-grid with spacing `fiber_grid_step` is built around `center`.
    All grid points within `radius` of the center form the candidate pool.
    Up to k points are drawn uniformly at random without replacement.

    Parameters
    ----------
    center : array-like, shape (2,)
    k : int
        Desired number of points.  If fewer candidates exist, all are returned.
    radius : float
    fiber_grid_step : float
        Grid spacing (serves as the minimum pairwise distance between points).
    seed : int, optional

    Returns
    -------
    np.ndarray, shape (<= k, 2), dtype float
    """
    if k <= 0:
        return np.zeros((0, 2), dtype=float)

    local_pts = _build_disk_grid(radius, fiber_grid_step)
    if len(local_pts) == 0:
        return np.zeros((0, 2), dtype=float)

    rg = _rng(seed)
    k_actual = min(k, len(local_pts))
    idx = rg.choice(len(local_pts), size=k_actual, replace=False)
    return local_pts[idx] + center.reshape(1, 2)


# Keep old disk-sampling name as a fallback (still used if someone calls directly).
def sample_points_in_disk(
    center: np.ndarray,
    k: int,
    *,
    radius: float = 1 / 3,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Original random-disk sampler (kept for compatibility; prefer sample_points_in_disk_grid)."""
    if k <= 0:
        return np.zeros((0, 2), dtype=float)
    rg = _rng(seed)
    u = rg.random(k)
    v = rg.random(k)
    r = radius * np.sqrt(u)
    theta = 2.0 * math.pi * v
    pts = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    return pts + center.reshape(1, 2)


# ---------------------------------------------------------------------
# Poset helpers
# ---------------------------------------------------------------------

def strict_dominance_edges(points: np.ndarray) -> List[Tuple[int, int]]:
    """Edges (i -> j) for strict dominance."""
    n = points.shape[0]
    edges: List[Tuple[int, int]] = []
    for i in range(n):
        pi = points[i]
        le = np.all(pi <= points, axis=1)
        neq = np.any(pi != points, axis=1)
        js = np.where(le & neq)[0]
        for j in js:
            edges.append((i, int(j)))
    return edges


def hasse_from_points(points: np.ndarray) -> nx.DiGraph:
    """Build Hasse diagram (transitive reduction) of strict-dominance order."""
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

def _piecewise_scalar(t: np.ndarray) -> np.ndarray:
    out = np.empty_like(t, dtype=float)
    pos = t > 0
    out[pos] = t[pos] ** 3
    out[~pos] = np.sign(t[~pos]) * np.abs(t[~pos]) ** (1.0 / 3.0)
    return out

def f_nonlinear(x: np.ndarray) -> np.ndarray:
    return _piecewise_scalar(x[:, 0]) + _piecewise_scalar(x[:, 1])

def f_linear_weak(x: np.ndarray) -> np.ndarray:
    return 0.1 * x[:, 0] + 0.1 * x[:, 1]

def f_linear_strong(x: np.ndarray) -> np.ndarray:
    return x[:, 0] + x[:, 1]

# Paper's three slope combinations (Sysoev et al., Table 1/2)
# α = (0.2, 0.2) = "low", (0.2, 2.0) = "mix", (2.0, 2.0) = "high"
def _make_linear(a1: float, a2: float) -> Callable[..., np.ndarray]:
    def _f(x: np.ndarray) -> np.ndarray:
        return a1 * x[:, 0] + a2 * x[:, 1]
    _f.__name__ = f"f_linear_{a1}_{a2}"
    return _f

MODEL_FUNCS: Dict[str, Callable[..., np.ndarray]] = {
    "nonlinear":     f_nonlinear,
    "linear_weak":   f_linear_weak,
    "linear_strong": f_linear_strong,
    # Paper slope combinations (coordinates fed in raw, noise_scale=q keeps SNR constant)
    "linear_low":    _make_linear(0.2, 0.2),   # low slope
    "linear_mix":    _make_linear(0.2, 2.0),   # asymmetric slope
    "linear_high":   _make_linear(2.0, 2.0),   # high slope
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
    y = make_observations(X, model=model, noise=noise, noise_scale=noise_scale, seed=seed, **model_kwargs)
    return {int(i): float(y[i]) for i in range(len(y))}


# ---------------------------------------------------------------------
# Lazy Dataset Support
# ---------------------------------------------------------------------

class LazyRiemannianDataset(Sequence):
    """
    A sequence-like object that stores cached R_i datasets on disk
    and loads them on demand.
    """
    def __init__(self, cache_dir: str, num_fibers: int):
        self.cache_dir = cache_dir
        self.num_fibers = num_fibers
        self._lengths = {}  # Cache for lengths

    def __len__(self) -> int:
        return self.num_fibers

    def __getitem__(self, i: int) -> np.ndarray:
        if i < 0 or i >= self.num_fibers:
            raise IndexError("Fiber index out of range")
        path = os.path.join(self.cache_dir, f"fiber_{i}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Cached fiber {i} not found at {path}")
        data = np.load(path)
        self._lengths[i] = len(data)
        return data

    def __iter__(self) -> Iterator[np.ndarray]:
        for i in range(self.num_fibers):
            yield self[i]

    def get_fiber_lengths(self) -> List[int]:
        """Returns lengths of all fibers (loads from disk if not cached)."""
        lens = []
        for i in range(self.num_fibers):
            if i in self._lengths:
                lens.append(self._lengths[i])
            else:
                data = self[i]
                lens.append(len(data))
        return lens

    def cleanup(self):
        """Remove cache directory."""
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)


class InMemoryLazyDataset(Sequence):
    """
    A sequence-like object that wraps in-memory R_i datasets
    and simulates a lazy iterator interface, exposing lengths.
    """
    def __init__(self, R_list: List[np.ndarray]):
        self.R_list = R_list
        self._lengths = [len(x) for x in R_list]

    def __len__(self) -> int:
        return len(self.R_list)

    def __getitem__(self, i: int) -> np.ndarray:
        return self.R_list[i]

    def __iter__(self) -> Iterator[np.ndarray]:
        for array in self.R_list:
            yield array

    def get_fiber_lengths(self) -> List[int]:
        return self._lengths

    def cleanup(self):
        pass


# ---------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------

def generate_q_and_fibers(
    *,
    nQ: int,
    avg_R: int,
    radius: float = 1 / 3,
    fiber_grid_step: float = 0.02,
    center_grid_step: int = 1,
    square_min: int = 0,
    square_max: int = 100,
    seed: Optional[int] = None,
    fiber_count_dist: str = "poisson",
    build_global_hasse: bool = True,
) -> Dict[str, object]:
    """
    In-memory grid-based dataset generator.

    Parameters
    ----------
    nQ : int
        Number of Q centers.
    avg_R : int
        Average number of fiber points per center.
    radius : float
        Disk radius for each fiber.
    fiber_grid_step : float
        Mini-grid spacing inside each disk.  Points are pre-separated by this
        distance, so no pruning is needed.  Should be > 0 and < radius.
    center_grid_step : int
        Lattice step for Q center placement.  Set to ceil(2 * radius) to ensure
        non-overlapping disks.  Default 1 (original integer lattice).
    square_min, square_max : int
        Bounding box for Q centers.
    seed : int, optional
    fiber_count_dist : {'poisson', 'uniform'}
    build_global_hasse : bool
        Whether to build the global P_hasse on all fiber points.
    """
    return _generate_core(
        nQ=nQ, avg_R=avg_R, radius=radius,
        fiber_grid_step=fiber_grid_step,
        center_grid_step=center_grid_step,
        square_min=square_min, square_max=square_max,
        seed=seed,
        fiber_count_dist=fiber_count_dist,
        build_global_hasse=build_global_hasse,
        use_cache=False,
    )


def generate_dataset_lazy(
    *,
    nQ: int,
    avg_R: int,
    radius: float = 1 / 3,
    fiber_grid_step: float = 0.02,
    center_grid_step: int = 1,
    square_min: int = 0,
    square_max: int = 100,
    seed: Optional[int] = None,
    fiber_count_dist: str = "poisson",
    cache_dir: Optional[str] = None,
) -> Dict[str, object]:
    """
    Lazy (disk-cached) grid-based dataset generator.
    Returns 'R_points_list' as a LazyRiemannianDataset object.
    Global X and P_hasse are NOT built to save memory.
    """
    if cache_dir is None:
        cache_dir = tempfile.TemporaryDirectory(prefix="ogpav_cache_")
    else:
        os.makedirs(cache_dir, exist_ok=True)

    return _generate_core(
        nQ=nQ, avg_R=avg_R, radius=radius,
        fiber_grid_step=fiber_grid_step,
        center_grid_step=center_grid_step,
        square_min=square_min, square_max=square_max,
        seed=seed,
        fiber_count_dist=fiber_count_dist,
        build_global_hasse=False,
        use_cache=True, cache_dir=cache_dir,
    )


def _generate_core(
    nQ, avg_R, radius, fiber_grid_step, center_grid_step,
    square_min, square_max, seed,
    fiber_count_dist, build_global_hasse,
    use_cache=False, cache_dir=None,
):
    rg = _rng(seed)

    # --- Sample Q centers from coarse grid ---
    q = sample_grid_centers(
        nQ,
        square_min=square_min,
        square_max=square_max,
        center_grid_step=center_grid_step,
        seed=seed,
    )

    # Pre-build the disk-local grid offsets (shared across all fibers)
    _disk_offsets = _build_disk_grid(radius, fiber_grid_step)
    n_candidates = len(_disk_offsets)

    R_list = []
    fiber_of = []

    if use_cache:
        lazy_dataset = LazyRiemannianDataset(cache_dir, nQ)

    seeds = rg.integers(0, 2**32 - 1, size=nQ, dtype=np.uint32)

    for i in range(nQ):
        if fiber_count_dist == "poisson":
            k = int(rg.poisson(lam=max(1, avg_R))) + 1
        elif fiber_count_dist == "uniform":
            hi = max(1, 2 * avg_R - 1)
            k = int(rg.integers(1, hi + 1))
        else:
            raise ValueError("fiber_count_dist must be 'poisson' or 'uniform'")

        # Draw from the pre-built disk grid (no pruning needed)
        if n_candidates == 0:
            pts = np.zeros((0, 2), dtype=float)
        else:
            rg_i = _rng(int(seeds[i]))
            k_actual = min(k, n_candidates)
            idx = rg_i.choice(n_candidates, size=k_actual, replace=False)
            pts = _disk_offsets[idx] + q[i].reshape(1, 2).astype(float)

        if use_cache:
            np.save(os.path.join(cache_dir, f"fiber_{i}.npy"), pts)
        else:
            R_list.append(pts)
            fiber_of.extend([i] * pts.shape[0])

    if use_cache:
        X = None
        P_hasse = None
        fiber_of_arr = None
        R_out = lazy_dataset
        R_hasse_list = None
    else:
        if len(fiber_of) == 0:
            X = np.zeros((0, 2), dtype=float)
            fiber_of_arr = np.zeros((0,), dtype=int)
        else:
            X = np.concatenate(R_list, axis=0)
            fiber_of_arr = np.asarray(fiber_of, dtype=int)

        if build_global_hasse and X.shape[0] > 0:
            P_hasse = hasse_from_points(X)
        else:
            P_hasse = None

        R_hasse_list = [hasse_from_points(pts) for pts in R_list]
        R_out = InMemoryLazyDataset(R_list)

    Q_hasse = hasse_from_points(q.astype(float))

    return dict(
        q_points=q,
        r_i=np.full((nQ,), float(radius), dtype=float),
        R_points_list=R_out,
        X=X,
        fiber_of=fiber_of_arr,
        Q_hasse=Q_hasse,
        R_hasse_list=R_hasse_list,
        P_hasse=P_hasse,
        Y_dict=None,
        Y_array=None,
    )


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def linearization_simple(X: np.ndarray) -> List[int]:
    """Return topological order indices for X."""
    if X.shape[0] == 0:
        return []
    sums = X.sum(axis=1)
    return list(np.argsort(sums))


def plot_geometry(
    data: Dict[str, object],
    *,
    show_r_labels: bool = True,
    figsize: Tuple[float, float] = (7, 7),
):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    q = np.asarray(data["q_points"])
    r_i = np.asarray(data["r_i"])
    R_list = data["R_points_list"]

    fig, ax = plt.subplots(figsize=figsize)

    for i in range(q.shape[0]):
        circ = Circle((q[i, 0], q[i, 1]), r_i[i], fill=False, edgecolor="black", linewidth=2.5, alpha=0.9, zorder=1)
        ax.add_patch(circ)
        if show_r_labels:
            ax.text(q[i, 0] + r_i[i], q[i, 1] + r_i[i], f"r={r_i[i]:.3g}", fontsize=8)

    for i, Ri in enumerate(R_list):
        if Ri is None or len(Ri) == 0:
            continue
        ax.scatter(Ri[:, 0], Ri[:, 1], s=40, color="tab:blue", edgecolor="white", linewidth=0.6, zorder=3)

    ax.scatter(q[:, 0], q[:, 1], s=70, marker="x", color="tab:red", linewidths=1.0, zorder=4, label="q centers (Q)")
    ax.set_aspect("equal", adjustable="box")
    pad = 1.0
    ax.set_xlim(q[:, 0].min() - pad, q[:, 0].max() + pad)
    ax.set_ylim(q[:, 1].min() - pad, q[:, 1].max() + pad)
    ax.set_title("Q centers, fibers, and radius disks")
    ax.legend(loc="best")
    ax.grid(True, linewidth=0.3, alpha=0.4)
    plt.show()
    return fig, ax


def plot_3d(X: np.ndarray, y: np.ndarray, *, title: str = "", figsize: Tuple[float, float] = (8, 6)):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
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
# Convenience
# ---------------------------------------------------------------------

def make_and_plot_models(
    data: Dict[str, object],
    *,
    models: Sequence[str] = ("nonlinear", "linear_weak", "linear_strong"),
    noise: str = "normal",
    noise_scale: float = 1.0,
    seed: Optional[int] = None,
):
    if data.get("X") is None:
        print("Skipping plots: X is None (Lazy mode).")
        return {}
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
    model: str = "nonlinear",
    noise: str = "normal",
    noise_scale: float = 1.0,
    seed: Optional[int] = None,
    **model_kwargs,
) -> Dict[str, object]:
    if data.get("X") is None:
        return data
    X = np.asarray(data["X"])
    n = X.shape[0]

    if y is not None:
        Y_dict = {int(i): float(y[i]) for i in range(n)}
        data["Y_dict"] = Y_dict
        data["Y_array"] = np.asarray(y, dtype=float)
        return data

    if Y_dict is not None:
        data["Y_dict"] = {int(i): float(Y_dict[i]) for i in range(n)}
        data["Y_array"] = np.array([data["Y_dict"][i] for i in range(n)], dtype=float)
        return data

    Y_dict = make_observations_dict(X, model=model, noise=noise, noise_scale=noise_scale, seed=seed, **model_kwargs)
    data["Y_dict"] = Y_dict
    data["Y_array"] = np.array([Y_dict[i] for i in range(len(Y_dict))], dtype=float)
    return data


# ---------------------------------------------------------------------
# Self-consistent parameter set
# ---------------------------------------------------------------------

def make_dataset_params(q: int, coverage: float = 3.0) -> dict:
    """
    Derive a self-consistent parameter set for a dataset of scale *q*.

    The single integer ``q`` controls everything:

    * ``nQ = avg_R = q``
    * ``radius = sqrt(q) / 2``  — disk area ∝ q = avg_R, so point density
      (selected points per unit area) is preserved as q grows.
    * ``fiber_grid_step ≈ sqrt(π/3) / 2 ≈ 0.51``  — constant minimum spacing
      between fiber points, follows from the density constraint.
    * ``center_grid_step = ceil(2 * radius)``  — guarantees non-overlapping
      disks by construction, no pruning required.
    * ``square_max = 4 * q``  — provides ~16× headroom for Q-center placement.
    * ``noise_scale = float(q)``  — recommended experiment parameter: scales with
      coordinate range so SNR stays constant as q grows (signal ∝ q for linear
      models, noise ∝ q). **Not** returned in the dict; compute as ``float(q)``
      in your experiment script.

    Parameters
    ----------
    q : int
        Scale parameter (number of Q centers = avg fiber size).
    coverage : float
        Target ratio of disk-grid candidates to avg_R (default 3).
        Higher values give more variety when sampling fiber points.

    Returns
    -------
    dict with keys: nQ, avg_R, radius, fiber_grid_step, center_grid_step,
                    square_min, square_max, noise_scale
    """
    if q <= 0:
        raise ValueError("q must be a positive integer.")
    radius           = math.sqrt(q) / 2
    fiber_grid_step  = math.sqrt(math.pi / coverage) / 2
    center_grid_step = math.ceil(2 * radius)
    return dict(
        nQ               = q,
        avg_R            = q,
        radius           = radius,
        fiber_grid_step  = fiber_grid_step,
        center_grid_step = center_grid_step,
        square_min       = 0,
        square_max       = 4 * q,
        # --- observation parameter (y = f(x) + ε) ---
        # Not passed to geometry functions; use in your experiment script.
        noise_scale      = float(q),
    )


def generate_standard(
    q: int,
    *,
    coverage: float = 3.0,
    seed: Optional[int] = None,
    fiber_count_dist: str = "poisson",
    build_global_hasse: bool = True,
    lazy: bool = False,
    cache_dir: Optional[str] = None,
) -> Dict[str, object]:
    """
    Generate a dataset using the self-consistent scale parameter *q*.

    A convenience wrapper around :func:`generate_q_and_fibers` /
    :func:`generate_dataset_lazy` that accepts a single scale parameter
    and derives all geometric parameters automatically via
    :func:`make_dataset_params`.

    Parameters
    ----------
    q : int
        Scale parameter — sets nQ, avg_R, radius, grid steps, and square_max.
    coverage : float
        Passed to :func:`make_dataset_params` (default 3).
    seed : int, optional
    fiber_count_dist : {'poisson', 'uniform'}
    build_global_hasse : bool
        Only used when ``lazy=False``.
    lazy : bool
        If True, use disk-cached lazy generation (large-N friendly).
    cache_dir : str, optional
        Directory for lazy caching; a temp dir is used if None.

    Returns
    -------
    dict — same structure as :func:`generate_q_and_fibers`.
    """
    params = make_dataset_params(q, coverage=coverage)
    # Strip observation-only keys — geometry functions don't accept them.
    geom = {k: v for k, v in params.items()
            if k not in ("noise_scale",)}
    if lazy:
        return generate_dataset_lazy(
            **geom,
            seed=seed,
            fiber_count_dist=fiber_count_dist,
            cache_dir=cache_dir,
        )
    return generate_q_and_fibers(
        **geom,
        seed=seed,
        fiber_count_dist=fiber_count_dist,
        build_global_hasse=build_global_hasse,
    )


if __name__ == "__main__":
    print(f"{'q':>6} {'radius':>8} {'ctr_step':>9} {'fiber_step':>11} {'disk_cands':>11} {'grid_pts':>10}")
    print("-" * 62)
    for q in [10, 40, 100, 500, 1000]:
        p = make_dataset_params(q)
        r, h, s = p["radius"], p["fiber_grid_step"], p["center_grid_step"]
        n_disk = int(math.pi * (r / h) ** 2)
        sq = p["square_max"]
        n_grid = ((sq // s) + 1) ** 2
        print(f"{q:>6} {r:>8.3f} {s:>9} {h:>11.4f} {n_disk:>11} {n_grid:>10}")

    print("\nSmoke test (q=40, seed=0) ...")
    data = generate_standard(40, seed=0)
    lens = data["R_points_list"].get_fiber_lengths()
    print(f"  fibers={len(lens)}  min={min(lens)}  max={max(lens)}  total={sum(lens)}")
    print("Done.")
