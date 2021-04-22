import numpy as np
from pykeops.numpy import LazyTensor
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler


def cdist(x, y):
    x_i = LazyTensor(x[:, None, :])  # (M, 1, 1)
    y_j = LazyTensor(y[None, :, :])  # (1, N, 1)
    D_ij = (
        ((x_i - y_j) ** 2).sum(-1).sqrt()
    )  # (M, N) symbolic matrix of euclidean distances
    return D_ij


def linear_kernel(x, y, epsilon=0.1):
    D_ij = cdist(x, y)
    return D_ij


def gaussian_kernel(x, y, epsilon=0.1):
    D_ij = cdist(x, y)
    return (-D_ij / (2 * epsilon ** 2)).exp()


class Nyx(BaseEstimator):
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
        kernel=gaussian_kernel,
        alpha=1e-10,
        epsilon=None,
    ):
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.kernel = kernel
        self.alpha = alpha
        self.epsilon = epsilon

    def _setfit(self, X, y):
        self.data_dim = X.shape[1]
        self.y = self.y_scaler.fit_transform(y.reshape(-1, 1)).astype(np.float64)
        self.X = self.x_scaler.fit_transform(X).astype(np.float64)
        if self.epsilon is None:
            xyt = self.X.T
            # default epsilon is the "the average distance between nodes" based on a bounding hypercube
            ximax = np.amax(xyt, axis=1)
            ximin = np.amin(xyt, axis=1)
            edges = ximax - ximin
            edges = edges[np.nonzero(edges)]
            self.epsilon = np.power(np.prod(edges) / xyt.shape[-1], 1.0 / edges.size)

    def fit(self, X, y):
        self._setfit(X=X, y=y)
        # Pairwise distances between observations
        self.internal_dist = self.kernel(self.X, self.X, epsilon=self.epsilon)
        # Solve for weights such that distance at the observations is minimized
        self.weights = self.internal_dist.solve(self.y, alpha=self.alpha)

    def predict(self, X):
        # Pairwise euclidean distance between inputs and grid
        dist = self.kernel(
            self.x_scaler.transform(X).astype(self.X.dtype),
            self.X,
            epsilon=self.epsilon,
        )
        # Matrix multiply the weights for each interpolated point by the distances
        zi = dist @ self.weights
        # Cast back to original space
        zi = self.y_scaler.inverse_transform(zi.reshape(-1, 1)).reshape(
            -1,
        )
        return zi
