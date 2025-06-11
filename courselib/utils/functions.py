import numpy as np

def sigmoid(x):
    """
    Compute the sigmoid activation function.

    Parameters:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: Sigmoid-transformed array.
    """
    return 1 / (1 + np.exp(-x))
