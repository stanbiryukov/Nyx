# Nyx
# Fast and scalable RBF interpolation
##  Jax and PyKeOps RBF interpolation methods are wrapped in a simple scikit-learn API
# Examples
## Jax flavor
```python
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.pipeline import make_pipeline

from nyx.utils import to_cartesian, from_cartesian
from nyx.jax.interpolate import Nyx as jNyx

import numpy as np

dtype = np.float32
# Create `n` random points on a 500 x 500 grid
rng = np.random.default_rng(3934)

n = 1000
nx, ny = 500, 500

x = rng.uniform(low=-180, high=180, size=(n,),).astype(dtype) # longitude
y = rng.uniform(low=-90, high=90, size=(n,)).astype(dtype) # latitude
z = rng.random(size=(n)).astype(dtype)
xy = np.column_stack([x, y])

# Create corresponding grid
xi = np.linspace(x.min(), x.max(), nx).astype(dtype)
yi = np.linspace(y.min(), y.max(), ny).astype(dtype)
xi, yi = np.meshgrid(xi, yi)
xi, yi = xi.flatten(), yi.flatten()
xygrid = np.column_stack([xi.flatten(), yi.flatten()])

# create a sklearn pipeline transformer that projects to cartesian and scales to unit variance
cartt = make_pipeline(FunctionTransformer(func=to_cartesian, inverse_func=from_cartesian, check_inverse=False), StandardScaler() )
jrbf = jNyx(x_scaler=cartt)
jrbf.fit(xy, z)
hat = jrbf.predict(X = xygrid)
```
## PyKeOps flavor
```python
from nyx.keops.interpolate import Nyx as keoNyx
krbf = keoNyx(x_scaler=cartt)
krbf.fit(xy, z)
hat = krbf.predict(X = xygrid)
```
