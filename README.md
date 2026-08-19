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
  - If **list of functions**: `[f_0, ..., f_{m-1}]`, one per fiber
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

- **`use_trend_following_first`** *(bool, default=True)*:  
  Use trend-following heuristic for local GPAV (Stage 1). Improves performance on structured data.

- **`use_trend_following_blocks`** *(bool, default=True)*:  
  Use trend-following for global block GPAV (Stage 2).

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

## Running tests

Run tests from the project root:

```bash
python -m tests.gpav_test
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

Please index the nodes of `R_i` with indices from `0` to `n_i - 1`.
`OperadicGPAV` never infers a partial order from your coordinates. A fiber with no
comparator is an antichain. If you want the geometry to matter, say so with `f` or
`assume_component_wise=True`. (`utils.sb_gpav` uses the opposite convention: there,
`f=None` means coordinate-wise dominance.)
## Authors

Eric Dolores Cuenca, Susana Lopez Moreno, Jonathan Toledo Toledo, Anh Nguyen, Sangil Kim
