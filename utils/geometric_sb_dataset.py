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
   - nonlinear:      g(x1)+g(x2) where g(t)=t^(1/3) if t<=0, t^3 if t>0
   - linear_weak:    0.1*x1 + 0.1*x2
   - linear_strong:  x1 + x2

This module now supports **Lazy Generation** to disk to handle large N without RAM spikes.
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
# Geometry helpers
# ---------------------------------------------------------------------

def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)

def sample_integer_centers(
    nQ: int,
    *,
    square_min: int = 0,
    square_max: int = 100,
    min_center_dist: float = 0.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Sample nQ distinct integer points in [square_min, square_max]^2.

    If min_center_dist > 0, enforce pairwise Euclidean separation between centers.
    """
    if nQ <= 0:
        raise ValueError("nQ must be positive.")

    side = square_max - square_min + 1
    total_pts = side * side
    if nQ > total_pts:
        raise ValueError("nQ exceeds number of integer lattice points in the square.")

    rg = _rng(seed)

    # Fast path: no separation constraint beyond uniqueness
    if min_center_dist <= 0:
        idx = rg.choice(total_pts, size=nQ, replace=False)
        xs = idx % side
        ys = idx // side
        q = np.stack([xs + square_min, ys + square_min], axis=1).astype(int)
        return q

    # Separation-constrained path: random greedy selection from the lattice
    grid_x, grid_y = np.meshgrid(
        np.arange(square_min, square_max + 1, dtype=int),
        np.arange(square_min, square_max + 1, dtype=int),
        indexing="xy",
    )
    candidates = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    order = rg.permutation(candidates.shape[0])

    selected = []
    min_center_dist_sq = float(min_center_dist) ** 2

    for idx in order:
        c = candidates[idx]
        ok = True
        for s in selected:
            dx = float(c[0] - s[0])
            dy = float(c[1] - s[1])
            if dx * dx + dy * dy < min_center_dist_sq:
                ok = False
                break
        if ok:
            selected.append(c)
            if len(selected) == nQ:
                return np.asarray(selected, dtype=int)

    raise ValueError(
        f"Could not place {nQ} centers in [{square_min},{square_max}]^2 "
        f"with min_center_dist={min_center_dist:.4f}. "
        f"Increase square_max or reduce min_center_dist."
    )


def sample_points_in_disk(
    center: np.ndarray,
    k: int,
    *,
    radius: float = 1/3,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Sample k points uniformly at random in a disk (2D)."""
    if k <= 0:
        return np.zeros((0, 2), dtype=float)
    rg = _rng(seed)
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
    """Greedy pruning: keep points in input order, drop any point within min_dist."""
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

MODEL_FUNCS: Dict[str, Callable[..., np.ndarray]] = {
    "nonlinear": f_nonlinear,
    "linear_weak": f_linear_weak,
    "linear_strong": f_linear_strong,
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
        self._lengths = {} # Cache for lengths

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
        """Returns lengths of all fibers without loading data (if cached metadata) 
        or by loading. Here we load to check."""
        lens = []
        for i in range(self.num_fibers):
            if i in self._lengths:
                lens.append(self._lengths[i])
            else:
                 # Minimal load
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
    radius: float = 1/3,
    min_dist: float = 0.02,
    square_min: int = 0,
    square_max: int = 100,
    min_center_dist: float = 0.0,
    seed: Optional[int] = None,
    fiber_count_dist: str = "poisson",
    build_global_hasse: bool = True,
) -> Dict[str, object]:
    """
    Legacy in-memory generator. Use generate_dataset_lazy for large N.
    """
    return _generate_core(
        nQ=nQ, avg_R=avg_R, radius=radius, min_dist=min_dist,
        square_min=square_min, square_max=square_max,
        min_center_dist=min_center_dist,
        seed=seed,
        fiber_count_dist=fiber_count_dist, build_global_hasse=build_global_hasse,
        use_cache=False
    )
def generate_dataset_lazy(
    *,
    nQ: int,
    avg_R: int,
    radius: float = 1/3,
    min_dist: float = 0.02,
    square_min: int = 0,
    square_max: int = 100,
    min_center_dist: float = 0.0,
    seed: Optional[int] = None,
    fiber_count_dist: str = "poisson",
    cache_dir: Optional[str] = None,
) -> Dict[str, object]:
    """
    Generates dataset using disk caching for fibers.
    Returns 'R_points_list' as a LazyRiemannianDataset object.
    Global X and P_hasse are NOT built to save memory.
    """
    if cache_dir is None:
        cache_dir = tempfile.mkdtemp(prefix="ogpav_cache_")
    else:
        os.makedirs(cache_dir, exist_ok=True)

    return _generate_core(
        nQ=nQ, avg_R=avg_R, radius=radius, min_dist=min_dist,
        square_min=square_min, square_max=square_max,
        min_center_dist=min_center_dist,
        seed=seed,
        fiber_count_dist=fiber_count_dist, build_global_hasse=False,
        use_cache=True, cache_dir=cache_dir
    )

def _generate_core(
    nQ, avg_R, radius, min_dist, square_min, square_max, min_center_dist, seed,
    fiber_count_dist, build_global_hasse,
    use_cache=False, cache_dir=None
):
    rg = _rng(seed)
    q = sample_integer_centers(
        nQ,
        square_min=square_min,
        square_max=square_max,
        min_center_dist=min_center_dist,
        seed=seed,
    )
    R_list = []
    fiber_of = []
    
    # Save cache metadata if needed
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

        pts = sample_points_in_disk(q[i].astype(float), k, radius=radius, seed=int(seeds[i]))
        pts = prune_too_close(pts, min_dist=min_dist)
        
        if use_cache:
            # Save to disk immediately and discard
            np.save(os.path.join(cache_dir, f"fiber_{i}.npy"), pts)
            # We don't append to R_list to save RAM
        else:
            R_list.append(pts)
            fiber_of.extend([i] * pts.shape[0])

    if use_cache:
        # X and P_hasse are skipped for lazy mode
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
    # X might be None in lazy mode
    # X = np.asarray(data["X"]) 
    r_i = np.asarray(data["r_i"])
    R_list = data["R_points_list"]
    
    fig, ax = plt.subplots(figsize=figsize)

    for i in range(q.shape[0]):
        circ = Circle((q[i, 0], q[i, 1]), r_i[i], fill=False, edgecolor="black", linewidth=2.5, alpha=0.9, zorder=1)
        ax.add_patch(circ)
        if show_r_labels:
            ax.text(q[i, 0] + r_i[i], q[i, 1] + r_i[i], f"r={r_i[i]:.3g}", fontsize=8)
            
    # Iterate safely if lazy
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
    # Only works if X is present (in-memory mode)
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
    # Requires X (in-memory)
    if data.get("X") is None:
         # Need to iterate fibers to generate Y?
         # For lazy mode, user handles Y construction externally
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

if __name__ == "__main__":
    # Quick smoke test
    data = generate_q_and_fibers(nQ=20, avg_R=40, seed=0, radius=1/3, min_dist=0.02)
    # Lazy test
    # data_lazy = generate_dataset_lazy(nQ=20, avg_R=40, seed=0)
    # print(len(data_lazy['R_points_list']))
