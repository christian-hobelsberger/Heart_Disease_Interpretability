import numpy as np

def train_test_split(X, y, test_size=0.2, seed=None):
    """
    Split the dataset into training and testing sets.

    Parameters:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.
        test_size (float): Proportion of test set.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Train/test splits.
    """
    np.random.seed(seed)
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    split = int(len(X) * (1 - test_size))
    return X[idx[:split]], X[idx[split:]], y[idx[:split]], y[idx[split:]]
