import functools
from typing import Callable, Optional

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

import jax
import jax.numpy as jnp


@jax.jit
def sqeuclidean_distance(x: jnp.array, y: jnp.array) -> float:
    return jnp.sum((x - y) ** 2)


@jax.jit
def euclidean_distance(x: jnp.array, y: jnp.array) -> float:
    return jnp.sqrt(sqeuclidean_distance(x, y))


@functools.partial(jax.jit, static_argnums=(0))
def distmat(func: Callable, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    vmap distance matrix
    """
    return jax.vmap(lambda x1: jax.vmap(lambda y1: func(x1, y1))(y))(x)


@jax.jit
def cdist(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    euclidean distance matrix
    -------------------
    equivalent to:
    from scipy.spatial.distance import cdist
    cdist(X, X, 'euclidean')
    """
    return distmat(euclidean_distance, x, y)


@jax.jit
def RBF_kernel(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    D_ij = cdist(x, y)
    return D_ij


@jax.jit
def gaussian_kernel(
    x: jnp.ndarray, y: jnp.ndarray, epsilon: Optional[float] = 0.1
) -> jnp.ndarray:
    D_ij = cdist(x, y)
    return jnp.exp(jnp.negative((jnp.power(jnp.divide(D_ij, epsilon), 2))))


@jax.jit
def multiquadric_kernel(
    x: jnp.ndarray, y: jnp.ndarray, epsilon: Optional[float] = 0.1
) -> jnp.ndarray:
    D_ij = cdist(x, y)
    return jnp.power(jnp.add(jnp.power(jnp.divide(D_ij, epsilon), 2), 1), 0.5)


class Nyx(BaseEstimator):
    """
    Jax accelerated scikit-learn friendly RBF interpolation
    """

    def __init__(
        self,
        x_scaler=StandardScaler(),
        y_scaler=StandardScaler(),
        kernel=RBF_kernel,
    ):
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.kernel = kernel

    def _setfit(self, X, y):
        self.y = jnp.array(
            self.y_scaler.fit_transform(y.reshape(-1, 1))
            .astype(np.float32)
            .reshape(
                -1,
            )
        )
        self.X = jnp.array(self.x_scaler.fit_transform(X).astype(np.float32))

    def fit(self, X, y):
        self._setfit(X=X, y=y)
        # Kernel distances between observations
        self.internal_dist = self.kernel(self.X, self.X)
        # Solve for weights such that distance at the observations is minimized
        self.weights = jax.jit(jnp.linalg.solve)(self.internal_dist, self.y)

    def predict(self, X):
        # Kernel distances between inputs and grid
        dist = self.kernel(
            jnp.array(self.X.astype(self.X.dtype)), self.x_scaler.transform(X)
        )
        # Matrix multiply the weights for each interpolated point by the distances
        zi = jax.jit(jnp.dot)(dist.T, self.weights)
        # Cast back to original space
        zi = self.y_scaler.inverse_transform(zi.reshape(-1, 1)).reshape(
            -1,
        )
        return zi
