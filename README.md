# OGPAV
OGPAV is an operadic version of GPAV for data with topological information.

![Tests](https://github.com/Rubiel1/OGPAV/actions/workflows/python-package.yml/badge.svg?branch=main)

Based on GPAV — from
Burdakov, Grimvall, Sysoev (2006)
“Data preordering in generalized PAV algorithm for monotonic regression” and Segmentation-Based GPAV (SB-GPAV) — from Sysoev, Burdakov, Grimvall (2011) “A segmentation based algorithm for large scale partially ordered monotonic regression”.

If the data was aggregated across multiple administrative levels e.g. regional and federal statistics, one can construct a poset Q representing the relationship between states at the federal level. Each node of this poset represent a region. We use the Q poset to reduce the complexity of the algorithm SB-GPAV. 

This allows the most expensive steps of the SB-GPAV to be performed locally on the regional input rather than on the global input. Moreover, the full dataset is never loaded and certain computations can be simplified using the underlying structure.
The first stage of OGPAV is also ready to run in parallel.

## Installation

```bash
pip install -r requirements.txt
```

Requirements:

- numpy >= 2.0.2
- networkx >= 2.8.8
- hasse >= 0.2.0
- python >= 3.9
- matplotlib 
## Running examples on Linux and macOS

All example scripts that use `max_workers > 1` should be run through a `main()` function protected by:


```python
from multiprocessing import freeze_support

def main():
    # build Q, R_datasets, Y, ...
    # call OperadicGPAV(...)
    pass

if __name__ == "__main__":
    freeze_support()  # optional on Linux/macOS, harmless to keep
    main()
```

This is required on macOS because multiprocessing uses `spawn`.



---

## Example Usage

### Basic Example with Geometric Dataset

Generate structured geometric data:

```python
import numpy as np
import networkx as nx
from multiprocessing import freeze_support
from utils.geometric_sb_dataset import generate_dataset_lazy
from OperadicGPAV import OperadicGPAV

def main():
    # Generate dataset with 3 fibers
    result = generate_dataset_lazy(
        nQ=3,           # Number of Q nodes (outer poset)
        avg_R=25,       # Average number of points per fiber
        radius=1/3,     # Radius for point sampling
        min_dist=0.02,  # Minimum distance between points
        seed=42         # Random seed for reproducibility
    )

    # Extract the fiber datasets (lazy sequence)
    R_datasets = result['R_points_list']

    # Create outer poset Q: 0 -> 1 -> 2 (chain structure)
    Q = nx.DiGraph()
    Q.add_edges_from([(0, 1), (1, 2)])

    # Get fiber lengths without loading the data into memory
    lengths = R_datasets.get_fiber_lengths()
    n0, n1, n2 = lengths[0], lengths[1], lengths[2]

    # Create response vector Y (concatenated across fibers)
    Y = np.concatenate([
        np.random.RandomState(42).uniform(0, 3, n0),
        np.random.RandomState(43).uniform(3, 6, n1),
        np.random.RandomState(44).uniform(6, 9, n2)
    ])

    # Run OperadicGPAV
    u = OperadicGPAV(
        Q=Q,
        R_datasets=R_datasets,
        Y=Y,
        max_workers=2,
        assume_component_wise=True,  # coordinate-wise order on vectors
        verbose=True
    )

    print(f"Fitted values shape: {u.shape}")
    print(f"Output range: [{u.min():.2f}, {u.max():.2f}]")

if __name__ == "__main__":
    freeze_support()
    main()
```

### Parameters Explained

**Required Parameters:**

- **`Q`** *(nx.DiGraph)*: The outer poset structure with **m nodes labeled 0, 1, ..., m-1** (one per fiber).  
  Example: `Q.add_edges_from([(0, 1), (1, 2)])` creates a 3-node chain.

- **`R_datasets`** *(List[np.ndarray] or LazyIterator)*: List of **m datasets**, where `R_datasets[i]` is an array of shape `(n_i, d)` with `n_i` vectors of dimension `d`.  
  Each dataset corresponds to one fiber in Q. Can also be a lazy generator if it implements a `get_fiber_lengths()` method to prevent memory exhaustion.

- **`Y`** *(np.ndarray)*: Global response vector of length **N = sum(n_i)**.  
  By default, assumes lexicographic order: Y[:n_0] for fiber 0, Y[n_0:n_0+n_1] for fiber 1, etc.

**Optional Parameters:**

- **`f`** *(Callable or List[Callable], default=None)*:  
  Comparator function(s) defining the partial order on fiber elements.
  - If **single function**: `f(a, b) -> bool` used for all fibers
  - If **list**: exactly `m` entries `[f_0, ..., f_{m-1}]`, one per fiber, each a function or `None`. A list of any other length raises `ValueError`.
  - If **None** (or `None` at position `i` of the list): **no order is asserted** on that
    fiber. `R_i` is treated as an antichain (a disjoint union of points); all local DAG
    construction and `gpav_seg` steps are skipped and the fiber feeds `n_i` singleton
    blocks into Stage 2, constrained only through `Q`. This is the least-assumption
    default — coordinate-wise dominance is *not* inferred from the coordinates.
  - If **None and `assume_component_wise=True`**: coordinate-wise comparison
    `a <= b ⟺ a[k] <= b[k] ∀k`, with the low-memory incremental DAG build.

- **`indices_list`** *(List[List[int]], default=None)*:  
  Explicit mapping of Y indices to fibers. If None, assumes lexicographic concatenation.

- **`segment_topo_orders`** *(List[Optional[List[int]]], default=None)*:  
  Custom topological orders for each fiber. Useful for controlling GPAV processing order.

- **`variant`** *(str, default="default")*:  
  `"default"`, `"review"` or `"fast"`. See [Variants](#variants) below.

- **`weights`** *(array of length N, default=None)*:  
  Positive weight of each observation, aligned with `Y` (default: all 1). The fit minimises `sum_k weights[k] * (u[k] - Y[k])**2`.

- **`use_trend_following_first`** *(bool or None, default=None)*:  
  Order inside each fiber (Stage 1). None/True: trend-following (LowerY, or its DFS approximation in `"fast"`). False: a plain topological sort that ignores Y. Not allowed with `variant="review"`.

- **`use_trend_following_blocks`** *(bool or None, default=None)*:  
  Order on the block graph (Stage 2). None/True: trend-following. False: Kahn's topological sort with ties broken by block id, cheaper on very large block graphs but less accurate; the gateway is then switched off. Not allowed with `variant="review"`.

- **`max_workers`** *(int, default=None)*:  
  Number of parallel workers for fiber processing. If None, uses CPU count. Set to 1 for sequential execution.

- **`verbose`** *(bool, default=False)*:  
  Print progress information during execution.

**Returns:**

- **`u`** *(np.ndarray)*: Fitted isotonic values of length N, aligned with input Y.  
  Satisfies the partial order constraints induced by Q and each `R_i`.

---

### Advanced Example: Custom Comparator

```python
import numpy as np
import networkx as nx
from OperadicGPAV import OperadicGPAV

# Define outer poset Q with 2 fibers
Q = nx.DiGraph()
Q.add_edge(0, 1)

# Create fiber datasets
R_datasets = [
    np.array([[1, 2], [2, 1], [3, 3]]),  # R_0: 3 points in 2D
    np.array([[4, 4], [5, 5]])           # R_1: 2 points in 2D
]

Y = np.array([1.0, 3.0, 2.0, 5.0, 4.0])

# Custom comparator
def custom_comparator(a, b):
    return a[0] < b[0]

# Run with custom comparator
u = OperadicGPAV(
    Q=Q,
    R_datasets=R_datasets,
    Y=Y,
    f=custom_comparator,  # Apply to all fibers
    max_workers=1,        # Sequential example: safe on Linux and macOS
    assume_component_wise=False,
    verbose=False
)

print(f"Fitted values: {u}")
```

### Example: Per-Fiber Comparators

```python
import numpy as np
import networkx as nx
from OperadicGPAV import OperadicGPAV

Q = nx.DiGraph()
Q.add_edge(0, 1)

# Use different comparators for different fibers
R_datasets = [
    np.array([[0], [1], [2]]),  # R_0: 1D points
    np.array([[0, 0], [1, 1]])  # R_1: 2D points
]

# Default coordinate-wise for R_0, custom for R_1
def r1_comparator(a, b):
    # Compare by L1 norm
    return np.sum(np.abs(a)) <= np.sum(np.abs(b))

Y = np.array([3.0, 1.0, 2.0, 4.0, 5.0])

u = OperadicGPAV(
    Q=Q,
    R_datasets=R_datasets,
    Y=Y,
    f=[None, r1_comparator],  #  R_0: no order asserted -> antichain (Y passes through unchanged);  custom order for R_1
    max_workers=1,
    assume_component_wise=False,  # must be False if custom comparators are provided
)

print(f"Output: {u}")
```

### Example: Loading Fibers from a Directory or `.zip`

The `dataset.py` library provides `CustomFiberDataset`, allowing you to load fiber subsets lazily from `.npy`, `.csv`, or `.txt` files directly off disk or extracted from a zip archive without consuming full memory.

```python
import numpy as np
import networkx as nx
from multiprocessing import freeze_support
from OperadicGPAV import OperadicGPAV
from utils.dataset import CustomFiberDataset

def main():
    # Load fiber data locally without storing it completely in memory
    # Automatically enables fast memory-mapped `get_fiber_lengths()` for `.npy`
    dataset = CustomFiberDataset(
        source_path="my_large_fibers.zip",
        node_to_file={
            0: "R_0.npy",
            1: "R_1.csv",
            2: "R_2.txt",
        }
        # If `node_to_file` is None, files are sorted alphabetically
        # and node 0 is mapped to the first alphabetical file, with a warning.
    )

    # Q-nodes chain: 0 -> 1 -> 2
    Q = nx.DiGraph([(0, 1), (1, 2)])
    Y = np.random.uniform(0, 5, sum(dataset.get_fiber_lengths()))

    u = OperadicGPAV(
        Q=Q,
        R_datasets=dataset,
        Y=Y,
        max_workers=2
    )
    print(u)

if __name__ == "__main__":
    freeze_support()
    main()
```

## Variants

`OperadicGPAV(..., variant=...)` selects one of three versions. All three return a fit that satisfies every constraint of `P = Q(R_1, ..., R_m)`; they differ in the greedy path GPAV takes, and therefore in accuracy, time and memory.

| | `"default"` | `"review"` | `"fast"` |
|---|---|---|---|
| Intended for | normal use | researchers comparing with GPAV / SB-GPAV | limited time or memory |
| Stage 1 order (inside each fiber) | LowerY | LowerY | DFS approximation of LowerY |
| Stage 2 order (block graph) | LowerY | LowerY | DFS approximation of LowerY |
| Q-edge `i -> j` in the block graph | every max-block of `R_i` to every min-block of `R_j`, or through one weight-0 gateway node when that needs fewer edges | same as default | same as default |
| Can the orders be overridden? | yes | no (locked) | yes |
| Fits | identical to the original release in every test (see Gateway) | identical to default | may be slightly less accurate |

Shared by all three: input validation (empty fibers, `Y` length, repeated rows), the incremental DAG builder, and per-fiber slicing of `Y` for the parallel workers.

**LowerY** is the trend-following order of Sysoev, Burdakov & Grimvall (2011), Algorithm 5: process observations in increasing `Y` while respecting the partial order, always taking the smallest-`Y` remaining ancestor first. It runs with an explicit stack, so there is no recursion limit; its cost grows quadratically with the size of the graph it is applied to.

**DFS** also starts from the `Y`-sorted sequence but follows parent links branch by branch, which makes it nearly linear. Among the ancestors of a node it can place a high-`Y` element before a low-`Y` one, so the fit is usually slightly worse.

**Gateway.** A Q-edge means every block of `R_i` lies below every block of `R_j`. With `a` maximal blocks in `R_i` and `b` minimal blocks in `R_j`, this can be written as `a * b` edges, or routed through one weight-0 node with `a + b` edges. The constraints are the same and the gateway never enters an average, but GPAV absorbs it into the first successor it processes, so the order of Stage 2 matters:

- With LowerY in Stage 2 (default, review), the gateway did not change the fit in any test: 22,456 instances over many poset families, including nested lexicographic sums, Boolean lattices, very thick and very large `Q` (`gateway_exploration_2026-10-09.md`). This is an empirical observation, not a proof. All three variants use it only on Q-edges where it saves edges, `a * b > a + b`, i.e. never when `a = 1` or `b = 1` (for example a fiber with a greatest or a least element, or any chain) and not when `a = b = 2`. Elsewhere the extra node only costs time: on a very thick `Q` with compressible fibers, always using it was 5x slower.
- With DFS in Stage 2 (fast), the gateway can change the fit, in either direction. Fast uses the same per-edge rule as default. Compared with fast using the gateway on every Q-edge: on the 500 instances below it changed 15 fits, all closer to the exact optimum; on 2,096 exploration instances it changed 217 fits, 114 better and 103 worse, by at most 3.3% of the total sum of squares; it was faster overall (66 s vs 83 s on the large cases, 2.2 s vs 14.6 s on a thick `Q`).
- With a `Y`-blind order in Stage 2 it can pool incomparable blocks, so it is switched off (in every variant) when `use_trend_following_blocks=False`.

### Measured accuracy

Against the exact isotonic regression (CVXPY/Clarabel), 500 random instances over five shapes of `Q` (chain, tree, diamond, fan-in, random), 4-7 fibers of 2-13 points in 1-3 dimensions:

| | optimal fit | mean excess SSE / TSS | worst |
|---|---|---|---|
| `"default"` / `"review"` | 463/500 | 1.3e-4 | 1.9% |
| `"fast"` | 449/500 | 2.1e-4 | 1.9% |
| for reference: Kahn order in Stage 2 | 380/500 | 1.4e-3 | 9.6% |

### Measured cost

Fibers that GPAV cannot compress (antichain fibers under a chain `Q`, 200 points each), `max_workers=1`:

| | m = 10 (N = 2,000) | m = 20 (N = 4,000) |
|---|---|---|
| `"default"` | 8.9 s, 3.5 MB peak | 31.7 s, 7.0 MB peak |
| `"fast"` | 3.2 s, 3.5 MB peak | 5.3 s, 6.7 MB peak |

Without the gateway (writing every Q-edge as `a * b` edges) default took 42 s / 84 MB and 231 s / 177 MB on the same inputs.

On compressible data the difference is small: 8 fibers of 300 points under a tree `Q` (N = 2,400) took 4.2 s with default and 2.8 s with fast.

**Rule of thumb:** use `"default"`. Switch to `"fast"` when the inputs are very large or time is tight; LowerY is quadratic in the number of blocks that reach Stage 2.

### Benchmarking against SB-GPAV (`"review"`)

`utils/review.py` runs both algorithms under the same rules:

- the same DAG builder, GPAV core and LowerY implementation;
- first order: LowerY over all N elements of the lexicographic sum for SB-GPAV (Algorithm 4, step 1), LowerY per fiber for OperadicGPAV;
- second order (block graph): LowerY for both. The SB paper allows any topological order in this step; LowerY is the more accurate choice, so SB-GPAV is run in its best form.
- OperadicGPAV may route a Q-edge through a gateway node (as in default). This is part of OperadicGPAV's own construction of the block graph, which SB-GPAV does not have; in every test it left OperadicGPAV's fit unchanged.

```python
from utils.review import compare_review
res = compare_review(Q, R_datasets, Y)          # n_segments defaults to m
res["sse_ogpav"], res["sse_sb"], res["time_ogpav"], res["time_sb_parts"]
```

Example, tree `Q` with 8 fibers of 2-D points, `max_workers=1`:

| N | OperadicGPAV | SB-GPAV (Hasse of P + LowerY on P + SB) | SSE / TSS (OGPAV, SB) |
|---|---|---|---|
| 800 | 0.17 s | 0.33 s (0.13 + 0.11 + 0.10) | 0.3528, 0.3526 |
| 1,600 | 0.49 s | 1.28 s (0.51 + 0.49 + 0.28) | 0.4032, 0.4044 |
| 3,200 | 1.60 s | 5.34 s (2.15 + 2.19 + 1.00) | 0.4053, 0.4060 |

Timings are from one machine and indicative only.

## Running tests

Run tests from the project root:

```bash
python -m pytest tests/
```

## Plot the artificial dataset
```
from utils.geometric_sb_dataset import (
    generate_q_and_fibers,
    plot_geometry,
    attach_observations,
    plot_3d,
)

data = generate_q_and_fibers(
    nQ=5,
    avg_R=10,
    radius=1/3,
    min_dist=0.02,
    seed=0,
    square_max= 2,
    square_min = -2,
)
data = attach_observations(
    data,
    model="nonlinear",
    noise="normal",
    noise_scale=0.5,
    seed=1,
)
plot_geometry(data, show_r_labels=True)

X = data["X"]
y = data["Y_array"]

plot_3d(X, y, title="nonlinear + normal noise")
```


## Notes on correctness

All algorithms assume acyclic partial orders (posets).

**A block is never its own predecessor.** The GPAV update `B_k^- = B_j^- U B_k^- \ {j}` (Burdakov, Grimvall & Sysoev 2006) does not remove `k`. In exact arithmetic `k` cannot re-enter its own predecessor list, but floating-point rounding of a weighted average can push a block value just above equal inputs and let `k` re-enter through a diamond (`j1 < j2 < k`). Before the fix `k` then "violated" itself, was merged into itself and deleted (`KeyError` in `gpav_seg`). The code now removes `k` as well, which matches the paper's definition of `B_k^-` (the blocks adjacent to `B_k`). It was found with weights in a randomized exploration and never in 390,824 unweighted runs; results that did not crash are unchanged (10,000 runs compared bit for bit).

**Repeated rows.** A fiber must not contain the same point twice: equal rows compare both ways, so the input is not a poset and OperadicGPAV raises a `ValueError`. Equal rows must receive equal fitted values, so merging each group of repeated rows into one row whose `Y` is the weighted mean of the group, with the group's total weight (its size, if all weights are 1) passed in `weights`, gives a problem with the same optimal solution as the original data; every copy then takes the fitted value of its group. OperadicGPAV does not merge rows itself, so that the user decides how repeated rows are treated.

```python
import numpy as np
uniq, inv = np.unique(R_i, axis=0, return_inverse=True)      # one fiber
inv = inv.ravel()
counts = np.bincount(inv)
Y_i_merged = np.bincount(inv, weights=Y_i) / counts           # mean Y per group
# ... call OperadicGPAV with uniq as the fiber, Y_i_merged and weights=counts,
# then u_i = u_merged[inv] gives the fit for the original rows
```

Please index the nodes of `R_i` with indices from `0` to `n_i - 1`.
## Authors

Eric Dolores Cuenca, Susana Lopez Moreno, Jonathan Toledo Toledo, Anh Nguyen, Sangil Kim, Jose Mendoza Cortes
