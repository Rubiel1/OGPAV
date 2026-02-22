# OGPAV
OGPAV is an operadic version of GPAV for data with topological information.

![Tests](https://github.com/Rubiel1/OGPAV/actions/workflows/python-package.yml/badge.svg?branch=master)

Consider GPAV — from
Burdakov, Grimvall, Sysoev (2006)
“Data preordering in generalized PAV algorithm for monotonic regression”

Segmentation-Based GPAV (SB-GPAV) — from
Sysoev, Burdakov, Grimvall (2011)
“A segmentation based algorithm for large scale partially ordered monotonic regression”

Assuming the data has structure, we use the extra information to reduce the number of calls to the min-max algorithm.
The first stage of the algorithm is also ready to run in parallel.


Installation 

```
pip install -r requirements.txt
```

Requirements:

    numpy >= 2.0.2

    networkx >= 2.8.8

    hasse >= 0.2.0



## Example Usage

### Basic Example with Geometric Dataset

Generate structured geometric data:
```python
import numpy as np
import networkx as nx
from geometric_sb_dataset import generate_dataset_lazy
from OperadicGPAV import OperadicGPAV

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
    assume_component_wise=True, # True if X are vectors and the order is v <= w if every coordinate of v is less equal to every coordinate of w
    verbose=True
)

print(f"Fitted values shape: {u.shape}")
print(f"Output range: [{u.min():.2f}, {u.max():.2f}]")
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
  - If **None**: uses coordinate-wise comparison `a <= b ⟺ a[k] <= b[k] ∀k`

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
  Satisfies the partial order constraints induced by Q and each R_i.

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
    np.array([[4, 4], [5, 5]])            # R_1: 2 points in 2D
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
    max_workers=1,
    assume_component_wise=False,
    verbose=False
)

print(f"Fitted values: {u}")
```

### Example: Per-Fiber Comparators

```python
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
    f=[None, r1_comparator],  # None uses default for R_0
    max_workers=1,
    assume_component_wise=False, #you cannot have this as True if you have custom comparators
)

print(f"Output: {u}")
```




Notes on Correctness

All algorithms assume acyclic partial orders (posets).

Please, Index the nodes of $R_i$ with indexes from $0$ to $n-1$.


Authors: Eric Dolores Cuenca, Susana Lopez Moreno, Jonathan Toledo Toledo, Anh Nguyen, Sangil Kim
