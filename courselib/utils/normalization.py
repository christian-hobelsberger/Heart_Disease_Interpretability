import numpy as np

def minmax_normalize(X):
    """
    Normalize features to the [0, 1] range.

    Parameters:
        X (np.ndarray): Input feature matrix.

    Returns:
        np.ndarray: Min-max normalized features.
    """
    return (X - np.min(X, axis=0)) / (np.ptp(X, axis=0) + 1e-10)

def zscore_normalize(X):
    """
    Normalize features to have 0 mean and unit variance.

    Parameters:
        X (np.ndarray): Input feature matrix.

    Returns:
        np.ndarray: Z-score normalized features.
    """
    return (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-10)
