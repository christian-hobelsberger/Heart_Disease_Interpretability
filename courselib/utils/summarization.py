import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def summarize_after_preprocessing(df, target_col='num', X=None, X_train=None, X_test=None):
    """
    Print summary after preprocessing and splitting.

    Parameters:
    -----------
    df : pd.DataFrame
        Full dataset including target.
    target_col : str
        Name of the target column.
    X : np.ndarray or pd.DataFrame, optional
        Full feature matrix (to get shape).
    X_train : np.ndarray or pd.DataFrame, optional
        Training set features.
    X_test : np.ndarray or pd.DataFrame, optional
        Test set features.
    """
    if X_train is not None and X_test is not None:
        print(f"\n📦 Train/test split:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples:     {len(X_test)}")

    class_counts = df[target_col].value_counts().sort_index()
    total = class_counts.sum()
    print("\n🧾 Binary target distribution:")
    for cls, count in class_counts.items():
        print(f"  Class {cls}: {count} samples ({(count / total) * 100:.1f}%)")
    
    if X is not None:
        print(f"\n🔢 Feature matrix shape: {X.shape[0]} samples × {X.shape[1]} features")
    
