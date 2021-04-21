import functools
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from pykeops.numpy import LazyTensor
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler


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
def jcdist(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    euclidean distance matrix
    -------------------
    equivalent to:
    from scipy.spatial.distance import cdist
    cdist(X, X, 'euclidean')
    """
    return distmat(euclidean_distance, x, y)


@jax.jit
def jax_RBF_kernel(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    D_ij = jcdist(x, y)
    return D_ij


@jax.jit
def jax_gaussian_kernel(
    x: jnp.ndarray, y: jnp.ndarray, epsilon: Optional[float] = 0.1
) -> jnp.ndarray:
    D_ij = jcdist(x, y)
    return jnp.exp(jnp.negative((jnp.power(jnp.divide(D_ij, epsilon), 2))))


@jax.jit
def jax_multiquadric_kernel(
    x: jnp.ndarray, y: jnp.ndarray, epsilon: Optional[float] = 0.1
) -> jnp.ndarray:
    D_ij = jcdist(x, y)
    return jnp.power(jnp.add(jnp.power(jnp.divide(D_ij, epsilon), 2), 1), 0.5)


class jaxNyx(BaseEstimator):
    """
    Jax accelerated scikit-learn friendly RBF interpolation
    """

    def __init__(
        self,
        x_scaler=StandardScaler(),
        y_scaler=StandardScaler(),
        kernel=jax_RBF_kernel,
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


def keops_cdist(x, y):
    x_i = LazyTensor(x[:, None, :])  # (M, 1, 1)
    y_j = LazyTensor(y[None, :, :])  # (1, N, 1)
    D_ij = (
        ((x_i - y_j) ** 2).sum(-1).sqrt()
    )  # (M, N) symbolic matrix of euclidean distances
    return D_ij


def keops_RBF_kernel(x, y):
    D_ij = keops_cdist(x, y)
    return D_ij


def keops_gaussian_kernel(x, y, epsilon=0.1):
    D_ij = keops_cdist(x, y)
    return (-((D_ij / epsilon) ** 2)).exp()


def keops_multiquadric_kernel(x, y, epsilon=0.1):
    D_ij = keops_cdist(x, y)
    return ((D_ij / epsilon) ** 2 + 1).sqrt()


class keopsNyx(BaseEstimator):
    """
    KeOps accelerated scikit-learn friendly RBF interpolation
    the ridge regularization parameter, **alpha**, controls the trade-off
    between a perfect fit (**alpha** = 0) and a
    smooth interpolation (**alpha** = :math:`+\infty`)
    """

    def __init__(
        self,
        x_scaler=StandardScaler(),
        y_scaler=StandardScaler(),
        kernel=keops_RBF_kernel,
        alpha=1e-3,
    ):
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.kernel = kernel
        self.alpha = alpha

    def _setfit(self, X, y):
        self.data_dim = X.shape[1]
        self.y = self.y_scaler.fit_transform(y.reshape(-1, 1)).astype(np.float32)
        self.X = self.x_scaler.fit_transform(X).astype(np.float32)

    def fit(self, X, y):
        self._setfit(X=X, y=y)
        # Pairwise distances between observations
        self.internal_dist = self.kernel(self.X, self.X)
        # Solve for weights such that distance at the observations is minimized
        self.weights = self.internal_dist.solve(self.y, alpha=self.alpha)

    def predict(self, X):
        # Pairwise euclidean distance between inputs and grid
        dist = self.kernel(self.x_scaler.transform(X).astype(self.X.dtype), self.X)
        # Matrix multiply the weights for each interpolated point by the distances
        zi = dist @ self.weights
        # Cast back to original space
        zi = self.y_scaler.inverse_transform(zi.reshape(-1, 1)).reshape(
            -1,
        )
        return zi
