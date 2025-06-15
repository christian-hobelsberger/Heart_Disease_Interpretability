import pandas as pd
import numpy as np
import sys
from courselib.utils.encoding import SimpleLabelEncoder

def preprocess_dataframe(df, target_col="target"):
    """
    Encode object columns, remove NaNs, and extract features/labels.

    Parameters:
        df (pd.DataFrame): Raw dataframe.
        target_col (str): Name of target column.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Feature matrix X and label vector y.
    """
    df = df.copy()

    # Encode categorical columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = SimpleLabelEncoder().fit_transform(df[col])

    # Count and report missing values
    nan_counts = df.isna().sum()
    nan_counts = nan_counts[nan_counts > 0]
    if not nan_counts.empty:
        sys.stdout.write("🧼 Dropping rows with missing values:\n")
        for col, count in nan_counts.items():
            sys.stdout.write(f"  - {col}: {count} missing\n")
        original_len = len(df)
        df = df.dropna()
        sys.stdout.write(f"🧹 Total rows dropped: {original_len - len(df)}\n")
    else:
        sys.stdout.write("✅ No missing values found.\n")

    # Extract features and labels
    X = df.drop(columns=target_col).values
    y = df[target_col].values
    return X, y
