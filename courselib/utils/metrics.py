import numpy as np

def accuracy(y_true, y_pred):
    """
    Compute classification accuracy.

    Parameters:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.

    Returns:
        float: Accuracy score.
    """
    return np.mean(y_true == y_pred)

def binary_accuracy(y_pred,y_true, class_labels=[0, 1]):
    """
    Accuracy function for binary classification models. 
    This function assumes that the predicted labels are continuous values and converts them to binary labels based on a threshold.

    Parameters:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.

    Returns:
        float: Accuracy score.
    """
    threshold = min(class_labels) + (max(class_labels) - min(class_labels)) / 2.
    pred_labels = np.where(y_pred >= threshold, max(class_labels), min(class_labels))
    return np.mean(pred_labels == y_true)*100

def precision(y_true, y_pred):
    """
    Compute precision score.

    Parameters:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.

    Returns:
        float: Precision score.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp + 1e-10)

def recall(y_true, y_pred):
    """
    Compute recall score.

    Parameters:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.

    Returns:
        float: Recall score.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn + 1e-10)

def f1_score(y_true, y_pred):
    """
    Compute F1 score.

    Parameters:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.

    Returns:
        float: F1 score.
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r + 1e-10)

def mean_squared_error(y_pred,y_true):
    return 0.5*np.mean((y_pred - y_true)**2)
