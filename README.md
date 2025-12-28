# OGPAV
Operadic version of GPAV

![Tests](https://github.com/Rubiel1/OGPAV/actions/workflows/python-package.yml/badge.svg?branch=master)

Assuming the data has structure, we use the extra information to reduce the number of calls to the min-max algorithm.
The first stage of the algorithm is also ready to run in parallel.



Authors: Eric Dolores Cuenca, Susana Lopez Moreno, Jonathan Toledo Toledo, Anh Nguyen, Sangil Kim

Example use:

```from geometric_sb_dataset import (
    generate_q_and_fibers, plot_geometry, make_observations, plot_3d, attach_observations
)

data = generate_q_and_fibers(
    nQ=30,
    avg_R=50,
    radius=1/3,
    min_dist=0.02,
    seed=0,
)
data = attach_observations(data, model="linear", noise="normal", noise_scale=0.5, seed=1)   # <-- this populates Y_dict and Y_array

# 2D geometry plot (disks around q_i + x points)
plot_geometry(data, show_r_labels=True)

X = data["X"]

# Make y values (choose a model)
y = make_observations(X, model="linear", noise="normal", noise_scale=0.5, seed=1)

# 3D plot (x1,x2,y)
plot_3d(X, y, title="linear + normal noise")
```


```
u_hat = OGPAV(
    data["Q_hasse"],
    data["R_hasse_list"],
    data["Y_dict"],          # now it is a real dict
    inputs_are_reduced=True,
)
print(u_hat[:30])```

To compare with previous versions ```from sb_gpav_paper import sb_gpav_fit as segmentation_based_gpav


# X: (n,p), Y: (n,)
fitted, blocks_hat, GB = segmentation_based_gpav(
    X, y,
    weights=None,          # or an (n,) array
    n_segments=10,         # s in the paper
    use_trend_following=True,
    debug=False,
)```