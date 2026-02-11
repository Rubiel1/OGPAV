# OGPAV
OGPAV is an operadic version of GPAV for big data.

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



Example use:

Generate structured geometric data
```
from geometric_sb_dataset import (
    generate_q_and_fibers,
    plot_geometry,
    attach_observations,
    plot_3d,
)

data = generate_q_and_fibers(
    nQ=30,
    avg_R=50,
    radius=1/3,
    min_dist=0.02,
    seed=0,
)
data = attach_observations(
    data,
    model="nonlinear",
    noise="normal",
    noise_scale=0.5,
    seed=1,
)
```
Visualize the data
```
plot_geometry(data, show_r_labels=True)

X = data["X"]
y = data["Y_array"]

plot_3d(X, y, title="nonlinear + normal noise")
```
Running Operadic GPAV (OGPAV)

```
from operadic_gpav import OGPAV

u_hat = OGPAV(
    data["Q_hasse"],
    data["R_hasse_list"],
    data["Y_dict"],
    inputs_are_reduced=True,
)

print(u_hat[:30])

```
If input is only locally indexed:
```
import hasse
from operadic_gpav import OGPAV

# --------------------------------------------------
# Outer poset Q with 3 elements {0,1,2}
# --------------------------------------------------
Q = hasse.PoSet.from_chains([0, 1], [0, 2])

R_subposets = [
    hasse.PoSet.from_chains([0, 1]),        # R_0
    hasse.PoSet.from_chains([0], [1]),      # R_1
    hasse.PoSet.from_chains([0, 1, 2]),     # R_2
]

# --------------------------------------------------
# Per-R_i observed data
# Keys are LOCAL node labels of each R_i
# --------------------------------------------------
A_list = [
    {0: 3.0, 1: 1.0},           # data on R_0
    {0: 2.0, 1: 4.0},           # data on R_1
    {0: 1.0, 1: 2.0, 2: 3.0},   # data on R_2
]

# --------------------------------------------------
# Run OGPAV and return output aligned with R_i
# --------------------------------------------------
u_list = OGPAV(
    Q=Q,
    R_subposets=R_subposets,
    A=None,                          # ignored when A_list is provided
    A_list=A_list,
    return_by_local_index=True,     # return per-fiber dictionaries
    verbose=True,
)

for i, u_i in enumerate(u_list):
    print(f"Fitted values on R_{i}: {u_i}")
```




Notes on Correctness

All algorithms assume acyclic partial orders (posets).

Please, Index the nodes of $R_i$ with indexes from $0$ to $n-1$.


Authors: Eric Dolores Cuenca, Susana Lopez Moreno, Jonathan Toledo Toledo, Anh Nguyen, Sangil Kim
