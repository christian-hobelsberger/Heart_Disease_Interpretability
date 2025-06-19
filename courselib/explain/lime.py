import numpy as np
from courselib.models.ridge import RidgeRegression
import pandas as pd
import matplotlib.pyplot as plt

class LimeTabularExplainer:
    """
    Custom LIME implementation for tabular data using RidgeRegression.

    Parameters:
        training_data (np.ndarray): Full training data used to calculate means and std.
        feature_names (List[str]): List of feature names.
        kernel_width (float): Controls the width of the exponential kernel for locality.
    """

    def __init__(self, training_data, feature_names=None, kernel_width=0.75):
        self.feature_names = feature_names
        self.kernel_width = kernel_width
        self.feature_means = np.mean(training_data, axis=0)
        self.feature_std = np.std(training_data, axis=0) + 1e-10

    def explain_instance(self, instance, model_predict_fn, num_samples=500):
        """
        Explain a single instance by locally approximating the model.

        Parameters:
            instance (np.ndarray): 1D input array to explain.
            model_predict_fn (Callable): Function to get model predictions.
            num_samples (int): Number of samples to generate around instance.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of (feature weights, sorted feature indices).
        """
        # Step 1: Generate perturbations around the instance
        perturbed = np.random.normal(loc=instance, scale=self.feature_std, size=(num_samples, len(instance)))
        perturbed = np.clip(perturbed, 0, 1)  # clip if normalized data
        perturbed = np.vstack([instance, perturbed])

        # Step 2: Get model predictions
        preds = model_predict_fn(perturbed)
        preds = preds if preds.ndim == 1 else preds[:, 1]  # use class 1 probability if applicable

        # Step 3: Compute distances and weights
        distances = np.linalg.norm((perturbed - instance) / self.feature_std, axis=1)
        weights = np.exp(-distances ** 2 / self.kernel_width ** 2)

        # Step 4: Train Ridge regression model with sample weights
        ridge = RidgeRegression(alpha=1.0, fit_intercept=True)
        ridge.fit(perturbed, preds, sample_weight=weights)

        feature_weights = ridge.coef_
        sorted_idx = np.argsort(np.abs(feature_weights))[::-1]
        return feature_weights, sorted_idx

    def as_list(self, weights, sorted_idx, top_k=5):
        """
        Format feature weights as list of tuples.

        Parameters:
            weights (np.ndarray): Feature weights.
            sorted_idx (np.ndarray): Indices of sorted feature importance.
            top_k (int): Number of top features to return.

        Returns:
            List[Tuple[str, float]]: Top features and weights.
        """
        return [
            (self.feature_names[i] if self.feature_names else f"Feature {i}", weights[i])
            for i in sorted_idx[:top_k]
        ]
    
def run_lime_multiple_times(
    explainer,
    instance,
    model_predict_fn,
    feature_names,
    n_runs=250,
    num_samples=300,
    top_k=7,
) -> tuple:
    """
    Repeatedly apply LIME to a single instance and return top-k average weights with std.

    Parameters:
        explainer (LimeTabularExplainer): LIME explainer instance.
        instance (np.ndarray): Instance to explain.
        model_predict_fn (Callable): Function mapping input → predictions.
        feature_names (List[str]): Feature names.
        n_runs (int): Number of repeated explanations.
        num_samples (int): Perturbations per run.
        top_k (int): Number of top features to return.

    Returns:
        Tuple[pd.Series, pd.Series, pd.Index]: means, stds, and top feature names
    """
    all_weights = []
    for _ in range(n_runs):
        weights, _ = explainer.explain_instance(instance, model_predict_fn, num_samples=num_samples)
        all_weights.append(weights)

    df_weights = pd.DataFrame(all_weights, columns=feature_names)
    mean_weights = df_weights.mean().sort_values(ascending=False)
    std_weights = df_weights.std()
    top_features = mean_weights.head(top_k).index

    return mean_weights[top_features], std_weights[top_features], top_features


def plot_lime_aggregated(
    top_means,
    top_stds,
    top_features,
    title="LIME Explanation Stability",
    color="cornflowerblue"
):
    """
    Plot average LIME weights with error bars.

    Parameters:
        top_means (pd.Series): Mean weights.
        top_stds (pd.Series): Standard deviations.
        top_features (Index): Selected top-k features.
        title (str): Plot title.
        color (str): Bar color.
    """
    plt.figure(figsize=(10, 5))
    plt.barh(
        top_features[::-1],
        top_means[::-1],
        xerr=top_stds[top_features][::-1],
        color=color
    )
    plt.axvline(0, color='k', linewidth=0.5)
    plt.title(title, fontsize=13)
    plt.xlabel("Average Feature Weight")
    plt.tight_layout()
    plt.show()
