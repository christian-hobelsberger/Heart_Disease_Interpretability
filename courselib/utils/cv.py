import itertools
import numpy as np
from typing import Dict, List, Callable, Tuple, Any
from courselib.utils.splits import train_test_split

def k_fold_indices(n_samples: int, k: int, seed: int = None) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate k (train_idx, val_idx) splits.

    Parameters
    ----------
    n_samples : int
        Number of data points.
    k : int
        Number of folds.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        List containing (train_indices, val_indices) pairs.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    folds = np.array_split(indices, k)
    splits = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.hstack([folds[j] for j in range(k) if j != i])
        splits.append((train_idx, val_idx))
    return splits


def grid_search_cv(
    ModelClass: Any,
    param_grid: Dict[str, List[Any]],
    X: np.ndarray,
    y: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    k: int = 5,
    seed: int = None,
    **model_init_kwargs,
) -> Tuple[Any, Dict[str, Any], float]:
    """
    Exhaustive grid search with k-fold CV.

    Parameters
    ----------
    ModelClass : Any
        The class of the model to instantiate (must inherit TrainableModel).
    param_grid : Dict[str, List[Any]]
        Dict of parameter name → list of candidate values.
    X, y : np.ndarray
        Data and labels.
    metric_fn : Callable
        Scoring function (higher = better).
    k : int
        Number of folds.
    seed : int, optional
        Random seed.
    **model_init_kwargs
        Extra arguments passed to the model constructor (e.g., optimizer).

    Returns
    -------
    best_model : object
        Fitted model with best parameters on full data.
    best_params : Dict[str, Any]
        Parameter set giving highest mean CV score.
    best_score : float
        Mean CV score for the best parameter set.
    """
    param_names = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))

    best_score, best_params = -np.inf, None
    splits = k_fold_indices(len(X), k, seed=seed)

    for combo in combos:
        params = dict(zip(param_names, combo))
        fold_scores = []

        for train_idx, val_idx in splits:
            model = ModelClass(**params, **model_init_kwargs)  # type: ignore
            model.fit(X[train_idx], y[train_idx])
            y_val_pred = model(X[val_idx])
            fold_scores.append(metric_fn(y[val_idx], y_val_pred))

        mean_score = np.mean(fold_scores)
        if mean_score > best_score:
            best_score, best_params = mean_score, params

    # Re-fit best model on entire dataset
    best_model = ModelClass(**best_params, **model_init_kwargs)  # type: ignore
    best_model.fit(X, y)
    return best_model, best_params, best_score
