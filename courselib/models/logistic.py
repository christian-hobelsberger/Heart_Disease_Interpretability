import numpy as np
from courselib.models.base import TrainableModel
from courselib.utils.functions import sigmoid

class LogisticRegression(TrainableModel):
    """
    Binary logistic regression model with optional regularization.

    Parameters:
        - w: Initial weights (array-like)
        - b: Initial bias (scalar)
        - optimizer: Optimizer object (e.g., GDOptimizer)
        - penalty: One of {"none", "ridge", "lasso"}
        - lam: Regularization strength (default: 0.0)
    """

    def __init__(self, w, b, optimizer, penalty="none", lam=0.0):
        super().__init__(optimizer)
        self.w = np.array(w, dtype=float)
        self.b = np.array([b], dtype=float)
        self.penalty = penalty
        self.lam = float(lam)

    def loss_grad(self, X, y):
        residual = self.decision_function(X) - y
        grad_w = X.T @ residual / len(X)
        grad_b = np.array([np.mean(residual)])


        if self.penalty == "ridge":
            grad_w += self.lam * self.w
        elif self.penalty == "lasso":
            grad_w += self.lam * np.sign(self.w)

        return {"w": grad_w, "b": grad_b}

    def decision_function(self, X):
        return sigmoid(X @ self.w + self.b)

    def _get_params(self):
        """Return model parameters as a dict for the optimizer."""
        return {"w": self.w, "b": self.b}

    def __call__(self, X):
        return (self.decision_function(X) >= 0.5).astype(int)