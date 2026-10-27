"""Search spaces for single trees and their forest base estimators.

Install ``sgtlearn[tune]`` for Optuna. This module does not import Optuna during
ordinary estimator use. Users own the objective, data splits, and trial budget.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from optuna.trial import FixedTrial, Trial

__all__ = ["suggest_params", "suggested_params"]


def suggested_params(
    *, use_tao: bool = False, use_bivariate: bool = False
) -> dict[str, list[Any]]:
    """Return a finite search space for sklearn grid or randomized search.

    Pass the dictionary as ``GridSearchCV(param_grid=...)`` or
    ``RandomizedSearchCV(param_distributions=...)``. Depth and branching limits
    use evenly spaced values; leaf counts use powers of two. Impurity and
    complexity costs include zero and geometrically spaced positive values.
    TAO uses one sweep when enabled; bivariate branching searches positive pair
    budgets when enabled. Both options default to off, as in ``suggest_params``.

    The full Cartesian grid is large: prefer randomized search, or narrow the
    lists before an exhaustive grid search. Criterion and forest-specific
    settings retain their estimator defaults. Optuna is not required here.
    """
    return {
        "inner_max_depth": [None],
        "tao_lambda": [1e-4],
        "tao_n_runs": [1 if use_tao else 0],
        "tao_pair_scale": [2.0],
        "coordinate_descent_max_iters": [5],
        "coordinate_descent_patience": [5],
        "max_depth": [4, 6, 8, 10],
        "num_partitions": [2, 3],
        "inner_max_leaf_nodes": [2, 4, 8, 16, 32, 64],
        "min_samples_leaf": [1, 2, 4, 8, 16, 32],
        "inner_min_samples_leaf": [1, 2, 4, 8, 16, 32],
        "min_impurity_decrease": [0.0, 0.01, 0.1, 1.0, 10.0],
        "inner_min_impurity_decrease": [0.0, 0.001, 0.01, 0.1],
        "pairwise_candidates": [0.25, 0.5, 1.0, 2.0] if use_bivariate else [0.0],
        "pairwise_penalty": [0.0, 0.01, 0.1, 1.0] if use_bivariate else [0.0],
        "branching_penalty": [0.0, 0.01, 0.1, 1.0],
    }


def suggest_params(
    trial: Trial | FixedTrial, *, use_tao: bool = False, use_bivariate: bool = False
) -> dict[str, Any]:
    """Return suggested keyword arguments for an SGT classifier or regressor.

    Accepts an Optuna ``Trial`` or ``FixedTrial``. No estimator is fitted and no
    study is created. The returned dictionary includes fixed settings as well
    as sampled ones; reconstruct it using ``FixedTrial(best_trial.params)``
    with the same feature flags. ``use_tao`` enables one refinement sweep;
    ``use_bivariate`` searches positive pair budgets and their penalties.
    Both are disabled by default.
    Criterion and forest-specific settings use the estimator's own defaults.
    See the hyperparameter-tuning tutorial for a complete example.
    """
    num_partitions = trial.suggest_int("num_partitions", 2, 3)
    pairs = (
        trial.suggest_categorical("pairwise_candidates", [0.25, 0.5, 1.0, 2.0])
        if use_bivariate
        else 0.0
    )
    return {
        "inner_max_depth": None,
        "tao_lambda": 1e-4,
        "tao_n_runs": 1 if use_tao else 0,
        "tao_pair_scale": 2.0,
        "coordinate_descent_max_iters": 5,
        "coordinate_descent_patience": 5,
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "inner_max_leaf_nodes": trial.suggest_int(
            "inner_max_leaf_nodes", 2, 64, log=True
        ),
        "num_partitions": num_partitions,
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 32, log=True),
        "inner_min_samples_leaf": trial.suggest_int(
            "inner_min_samples_leaf", 1, 32, log=True
        ),
        "min_impurity_decrease": trial.suggest_categorical(
            "min_impurity_decrease", [0.0, 0.01, 0.1, 1.0, 10.0]
        ),
        "inner_min_impurity_decrease": trial.suggest_categorical(
            "inner_min_impurity_decrease", [0.0, 0.001, 0.01, 0.1]
        ),
        "pairwise_candidates": pairs,
        "pairwise_penalty": (
            trial.suggest_categorical("pairwise_penalty", [0.0, 0.01, 0.1, 1.0])
            if pairs > 0
            else 0.0
        ),
        "branching_penalty": (
            trial.suggest_categorical("branching_penalty", [0.0, 0.01, 0.1, 1.0])
            if num_partitions > 2
            else 0.0
        ),
    }
