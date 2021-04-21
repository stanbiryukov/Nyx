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


def RBF_kernel(x, y):
    D_ij = cdist(x, y)
    return D_ij


def gaussian_kernel(x, y, epsilon=0.1):
    D_ij = cdist(x, y)
    return (-((D_ij / epsilon) ** 2)).exp()


def multiquadric_kernel(x, y, epsilon=0.1):
    D_ij = cdist(x, y)
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
        kernel=RBF_kernel,
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
