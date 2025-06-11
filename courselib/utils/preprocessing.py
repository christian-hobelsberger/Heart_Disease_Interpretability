import pandas as pd
from courselib.utils.encoding import SimpleLabelEncoder

def preprocess_dataframe(df, target_col="target"):
    """
    Encode object columns and extract features/labels.

    Parameters:
        df (pd.DataFrame): Raw dataframe.
        target_col (str): Name of target column.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Feature matrix X and label vector y.
    """
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = SimpleLabelEncoder().fit_transform(df[col])
    X = df.drop(columns=target_col).values
    y = df[target_col].values
    return X, y
