import pandas as pd
import os
from ucimlrepo import fetch_ucirepo

def load_heart_data():
    """
    Load the Heart Disease dataset from the UCI Machine Learning Repository.

    If fetching the data fails (e.g., due to a timeout or network error),
    fallback data is loaded from local CSV files:
        - data/X_df.csv
        - data/y_series.csv

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Features and target.
    """
    try:
        heart = fetch_ucirepo(id=45)
        X = heart.data.features
        y = heart.data.targets
    except Exception as e:
        print(f"⚠️ Could not fetch Heart Disease dataset from UCI: {e}")
        print("📦 Falling back to local files: data/X_df.csv and data/y_series.csv")
        X = pd.read_csv(os.path.join("data", "X_df.csv"))
        y = pd.read_csv(os.path.join("data", "y_series.csv")).squeeze("columns")  # Ensure it's a Series

    return X, y