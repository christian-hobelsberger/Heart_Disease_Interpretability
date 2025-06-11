import numpy as np

class SimpleLabelEncoder:
    """
    Label encoding for categorical variables.
    """

    def __init__(self):
        self.mapping = {}

    def fit(self, values):
        """
        Learn integer encoding.

        Parameters:
            values (Iterable): Input categorical values.

        Returns:
            self
        """
        unique_vals = sorted(set(values))
        self.mapping = {val: idx for idx, val in enumerate(unique_vals)}
        return self

    def transform(self, values):
        """
        Transform input using learned mapping.

        Parameters:
            values (Iterable): Input categorical values.

        Returns:
            np.ndarray: Encoded integers.
        """
        return np.array([self.mapping[val] for val in values])

    def fit_transform(self, values):
        """
        Fit encoder and return transformed result.

        Parameters:
            values (Iterable): Input categorical values.

        Returns:
            np.ndarray: Encoded integers.
        """
        return self.fit(values).transform(values)
