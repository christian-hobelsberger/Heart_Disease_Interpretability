import numpy as np
from .base import TrainableModel

class RidgeRegression(TrainableModel):
    """
    Ridge regression (L2-regularized linear regression) with optional sample weights.

    Parameters:
        - alpha (float): Regularization strength.
        - fit_intercept (bool): Whether to include an intercept term.
    """

    def __init__(self, alpha=1.0, fit_intercept=True):
        super().__init__(optimizer=None)
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X, y, sample_weight=None):
        """
        Fit the model to data.

        Parameters:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
            sample_weight (np.ndarray or None): Optional per-sample weights.
        """
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])

        if sample_weight is not None:
            W = np.diag(sample_weight)
            XTWX = X.T @ W @ X
            XTWy = X.T @ W @ y
        else:
            XTWX = X.T @ X
            XTWy = X.T @ y

        identity = np.eye(X.shape[1])
        if self.fit_intercept:
            identity[0, 0] = 0  # don't regularize bias

        theta = np.linalg.solve(XTWX + self.alpha * identity, XTWy)
        if self.fit_intercept:
            self.intercept_ = theta[0]
            self.coef_ = theta[1:]
        else:
            self.coef_ = theta

    def predict(self, X):
        """
        Predict outputs for input data.

        Parameters:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predictions.
        """
        if self.fit_intercept:
            return X @ self.coef_ + self.intercept_
        return X @ self.coef_

    def _get_params(self):
        return {"coef_": self.coef_}

    def loss_grad(self, X, y):
        raise NotImplementedError("Not used in closed-form regression.")
