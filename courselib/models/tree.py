import numpy as np
from courselib.models.base import TrainableModel

class DecisionNode:
    """
    Represents a single node in the decision tree.
    """
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree(TrainableModel):
    """
    CART-style decision tree classifier.

    Parameters:
        - max_depth: Maximum depth of the tree
        - min_samples_split: Minimum number of samples to perform a split
        - criterion: Split criterion ("gini" or "entropy")
    """

    def __init__(self, max_depth=5, min_samples_split=2, criterion="gini"):
        super().__init__(optimizer=None)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _get_params(self):
        return {"max_depth": self.max_depth, "criterion": self.criterion}

    def loss_grad(self, X, y):
        raise NotImplementedError("loss_grad is not applicable to tree models.")

    def _gini(self, y):
        classes = np.unique(y)
        return 1.0 - sum((np.sum(y == c) / len(y)) ** 2 for c in classes)

    def _entropy(self, y):
        probs = np.bincount(y) / len(y)
        return -np.sum([p * np.log2(p + 1e-10) for p in probs if p > 0])

    def _information_gain(self, parent, left, right):
        impurity = self._gini if self.criterion == "gini" else self._entropy
        n = len(parent)
        return impurity(parent) - (len(left) / n) * impurity(left) - (len(right) / n) * impurity(right)

    def _best_split(self, X, y):
        best_gain, best_feat, best_thresh = -1, None, None
        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for t in thresholds:
                left_mask = X[:, feature_index] <= t
                right_mask = ~left_mask
                if np.sum(left_mask) < self.min_samples_split or np.sum(right_mask) < self.min_samples_split:
                    continue
                gain = self._information_gain(y, y[left_mask], y[right_mask])
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feature_index, t
        return best_feat, best_thresh

    def _grow_tree(self, X, y, depth=0):
        if len(set(y)) == 1 or depth >= self.max_depth:
            return DecisionNode(value=np.bincount(y).argmax())

        feat_idx, threshold = self._best_split(X, y)
        if feat_idx is None:
            return DecisionNode(value=np.bincount(y).argmax())

        left_mask = X[:, feat_idx] <= threshold
        left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._grow_tree(X[~left_mask], y[~left_mask], depth + 1)
        return DecisionNode(feat_idx, threshold, left, right)

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        branch = node.left if x[node.feature_index] <= node.threshold else node.right
        return self._traverse_tree(x, branch)