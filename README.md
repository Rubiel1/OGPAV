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
    model="linear",
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

plot_3d(X, y, title="linear + normal noise")
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

To compare with Segmentation-Based GPAV
```
from sb_gpav_paper import sb_gpav_fit

X = data["X"]
y = data["Y_array"]

fitted, blocks_hat, GB = sb_gpav_fit(
    X,
    y,
    n_segments=10,
    use_trend_following=True,
    debug=False,
)
```
Notes on Correctness

All algorithms assume acyclic partial orders (posets).



Authors: Eric Dolores Cuenca, Susana Lopez Moreno, Jonathan Toledo Toledo, Anh Nguyen, Sangil Kim