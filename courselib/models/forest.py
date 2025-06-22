import numpy as np
from courselib.models.base import TrainableModel
from courselib.models.tree import DecisionTree

class RandomForest(TrainableModel):
    """
    Random forest ensemble of decision trees.

    Parameters:
        - n_estimators: Number of trees
        - max_depth: Depth for each tree
        - min_samples_split: Minimum samples per split
        - criterion: Splitting criterion ("gini" or "entropy")
        - random_state: Seed for reproducibility
    """

    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=2, criterion="gini", random_state=None):
        super().__init__(optimizer=None)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.random_state = random_state
        self.trees = []
        self.n_classes = None
        self.rng = np.random.default_rng(random_state)

    def __call__(self, X):
        return self.predict(X)

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.trees = []
        for _ in range(self.n_estimators):
            indices = self.rng.choice(len(X), len(X), replace=True)
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion=self.criterion
            )
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(
            lambda x: np.bincount(x, minlength=self.n_classes).argmax(),
            axis=0,
            arr=tree_preds
        )

    def _get_params(self):
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "random_state": self.random_state
        }

    def loss_grad(self, X, y):
        raise NotImplementedError("loss_grad not applicable for random forest.")
