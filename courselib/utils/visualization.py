import matplotlib.pyplot as plt
import numpy as np
import os

def compute_confusion_matrix(y_true, y_pred, labels=(0, 1)):
    """
    Computes a 2x2 confusion matrix for binary classification.
    Returns a numpy array: [[TN, FP], [FN, TP]]
    """
    cm = np.zeros((2, 2), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        cm[int(yt)][int(yp)] += 1
    return cm

def plot_confusion_matrix(
    y_true, y_pred, 
    model_name="Model", 
    threshold=0.5, 
    labels=("No Disease", "Disease"),
    save_path=None
):
    """
    Compute and plot a confusion matrix for binary classification.

    Parameters:
    - y_true: Ground-truth labels (0 or 1)
    - y_pred: Raw predictions (either probabilities or labels)
    - model_name: Title to display above the plot
    - threshold: Threshold to binarize probabilistic predictions
    - labels: Tuple of class label strings
    - save_path: If provided, saves the plot to this file path
    """
    if not set(np.unique(y_pred)).issubset({0, 1}):
        y_pred = (y_pred > threshold).astype(int)
    
    cm = compute_confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap='Blues')
    ax.set_title(f"Confusion Matrix — {model_name}")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=12, fontweight='bold')

    plt.grid(False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # adjust layout to avoid clipping title

    # Save if path is specified
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")  # also clip padding

    plt.show()
