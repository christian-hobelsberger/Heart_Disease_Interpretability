from ucimlrepo import fetch_ucirepo

def load_heart_data():
    """
    Load the Heart Disease dataset from the UCI Machine Learning Repository.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Features and target.
    """
    heart = fetch_ucirepo(id=45)
    X = heart.data.features
    y = heart.data.targets
    return X, y
