"""Optuna parameter suggestions for single trees and their forest base estimators.

Install ``sgtlearn[tune]`` for Optuna. This module does not import Optuna during
ordinary estimator use. Users own the objective, data splits, and trial budget.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from optuna.trial import FixedTrial, Trial

__all__ = ["recommended_params", "suggest_params"]


def recommended_params(
    *, use_tao: bool = False, use_bivariate: bool = False
) -> dict[str, Any]:
    """Return the shared study-based settings without requiring Optuna.

    Works with both single-tree tasks and forest base trees. ``use_tao`` enables
    one refinement sweep with lambda 1e-4; ``use_bivariate`` enables a pair budget
    of 0.5 per logical feature. Both are opt-in. The reported reserved-test
    results use both enabled; forests were not benchmarked. Criterion and
    forest-specific settings retain their defaults.
    """
    return {
        "max_depth": 4,
        "max_leaf_nodes": None,
        "num_partitions": 7,
        "inner_max_depth": None,
        "inner_max_leaf_nodes": 4,
        "min_samples_leaf": 4,
        "inner_min_samples_leaf": 13,
        "min_impurity_decrease": 1.0,
        "inner_min_impurity_decrease": 0.01,
        "coordinate_descent_max_iters": 5,
        "coordinate_descent_patience": 5,
        "pairwise_candidates": 0.5 if use_bivariate else 0.0,
        "pairwise_penalty": 0.0,
        "branching_penalty": 0.0,
        "tao_n_runs": 1 if use_tao else 0,
        "tao_lambda": 1e-4,
        "tao_pair_scale": 2.0,
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
    See the tuning guide for the study and limitations behind these ranges.
    """
    num_partitions = trial.suggest_int("num_partitions", 2, 8)
    pairs = (
        trial.suggest_categorical("pairwise_candidates", [0.25, 0.5, 1.0, 2.0])
        if use_bivariate
        else 0.0
    )
    return {
        **recommended_params(use_tao=use_tao, use_bivariate=use_bivariate),
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
