from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

import torch


@torch.jit.script
def cdist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    euclidean distance matrix
    -------------------
    equivalent to:
    from scipy.spatial.distance import cdist
    cdist(X, X, 'euclidean')
    """
    return torch.cdist(x, y, p=2.0, compute_mode="use_mm_for_euclid_dist")


@torch.jit.script
def linear_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    D_ij = cdist(x, y)
    return D_ij


@torch.jit.script
def gaussian_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    D_ij = cdist(x, y)
    return torch.exp(-D_ij / (2 * epsilon ** 2))


class Nyx(BaseEstimator):
    """
    Torch accelerated scikit-learn friendly RBF interpolation
    """

    def __init__(
        self,
        x_scaler=StandardScaler(),
        y_scaler=StandardScaler(),
        kernel=linear_kernel,
        epsilon=None,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.kernel = kernel
        self.epsilon = epsilon
        self.device = device

    def _to_tensor(self, array, dtype=torch.float64):
        return torch.as_tensor(array, dtype=dtype).to(self.device)

    def _setfit(self, X, y):
        self.y = self._to_tensor(
            self.y_scaler.fit_transform(y.reshape(-1, 1)).reshape(
                -1,
            )
        )
        self.X = self._to_tensor(self.x_scaler.fit_transform(X))
        if self.epsilon is None:
            xyt = self.X.T
            # default epsilon is the "the average distance between nodes" based on a bounding hypercube
            ximax = torch.amax(xyt, axis=1)
            ximin = torch.amin(xyt, axis=1)
            edges = ximax - ximin
            edges = edges[torch.nonzero(edges)]
            self.epsilon = torch.pow(
                torch.prod(edges) / xyt.shape[-1], 1.0 / edges.size(0)
            )

    def fit(self, X, y):
        self._setfit(X=X, y=y)
        # Kernel distances between observations
        self.internal_dist = self.kernel(self.X, self.X, epsilon=self.epsilon)
        # Solve for weights such that distance at the observations is minimized
        self.weights = torch.linalg.solve(self.internal_dist, self.y)

    def predict(self, X):
        # Kernel distances between inputs and grid
        dist = self.kernel(
            self.X,
            self._to_tensor(self.x_scaler.transform(X), dtype=self.X.dtype),
            epsilon=self.epsilon,
        )
        # Matrix multiply the weights for each interpolated point by the distances
        zi = torch.matmul(self.weights, dist).cpu().numpy()
        # Cast back to original space
        zi = self.y_scaler.inverse_transform(zi.reshape(-1, 1)).reshape(
            -1,
        )
        return zi
