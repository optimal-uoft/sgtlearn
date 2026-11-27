"""Search spaces for single trees and their forest base estimators.

Install ``sgtlearn[tune]`` for Optuna. This module does not import Optuna during
ordinary estimator use. Users own the objective, data splits, and trial budget.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from optuna.trial import FixedTrial, Trial

__all__ = ["suggest_params", "suggested_params"]

_MAX_DEPTH_GRID = [4, 6, 8, 10]
_INNER_MAX_LEAF_NODES_GRID = [2, 4, 8, 16, 32, 64]
_MIN_SAMPLES_LEAF_GRID = [1, 2, 4, 8, 16, 32]
_MIN_IMPURITY_DECREASE_GRID = [0.0, 0.01, 0.1, 1.0, 10.0]
_INNER_MIN_IMPURITY_DECREASE_GRID = [0.0, 0.001, 0.01, 0.1]
_PAIRWISE_CANDIDATES_GRID = [0.25, 0.5, 1.0, 2.0]
_PENALTY_GRID = [0.0, 0.01, 0.1, 1.0]
_TAO_N_RUNS = 10
_TAO_LAMBDA_GRID = [0.01, 0.001, 0.0001]
_TAO_PAIR_SCALE_GRID = [1.5, 1.75, 2.0]


def suggested_params(
    *, use_tao: bool = False, use_bivariate: bool = False
) -> list[dict[str, list[Any]]]:
    """Return a finite search space for sklearn grid or randomized search.

    Pass the list of dictionaries as ``GridSearchCV(param_grid=...)`` or
    ``RandomizedSearchCV(param_distributions=...)``. Only tunable parameters
    are included, except ``tao_n_runs=[10]`` when TAO is enabled. The binary
    and ternary grids are separate so only ternary trees search
    ``branching_penalty``. Optuna is not required here.
    """
    params: dict[str, list[Any]] = {
        "max_depth": list(_MAX_DEPTH_GRID),
        "inner_max_leaf_nodes": list(_INNER_MAX_LEAF_NODES_GRID),
        "min_samples_leaf": list(_MIN_SAMPLES_LEAF_GRID),
        "inner_min_samples_leaf": list(_MIN_SAMPLES_LEAF_GRID),
        "min_impurity_decrease": list(_MIN_IMPURITY_DECREASE_GRID),
        "inner_min_impurity_decrease": list(_INNER_MIN_IMPURITY_DECREASE_GRID),
    }
    if use_tao:
        params.update(tao_lambda=list(_TAO_LAMBDA_GRID), tao_n_runs=[_TAO_N_RUNS])
    if use_bivariate:
        params.update(
            pairwise_candidates=list(_PAIRWISE_CANDIDATES_GRID),
            pairwise_penalty=list(_PENALTY_GRID),
        )
    if use_tao and use_bivariate:
        params["tao_pair_scale"] = list(_TAO_PAIR_SCALE_GRID)
    return [
        {**params, "num_partitions": [2]},
        {
            **params,
            "num_partitions": [3],
            "branching_penalty": list(_PENALTY_GRID),
        },
    ]


def suggest_params(
    trial: Trial | FixedTrial, *, use_tao: bool = False, use_bivariate: bool = False
) -> dict[str, Any]:
    """Return suggested keyword arguments for an SGT classifier or regressor.

    Accepts an Optuna ``Trial`` or ``FixedTrial``. Reconstruct using
    ``FixedTrial(best_trial.params)`` with the same feature flags.
    """
    num_partitions = trial.suggest_int("num_partitions", 2, 3)
    params: dict[str, Any] = {
        "max_depth": trial.suggest_int(
            "max_depth", _MAX_DEPTH_GRID[0], _MAX_DEPTH_GRID[-1]
        ),
        "inner_max_leaf_nodes": trial.suggest_int(
            "inner_max_leaf_nodes",
            _INNER_MAX_LEAF_NODES_GRID[0],
            _INNER_MAX_LEAF_NODES_GRID[-1],
            log=True,
        ),
        "num_partitions": num_partitions,
        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf",
            _MIN_SAMPLES_LEAF_GRID[0],
            _MIN_SAMPLES_LEAF_GRID[-1],
            log=True,
        ),
        "inner_min_samples_leaf": trial.suggest_int(
            "inner_min_samples_leaf",
            _MIN_SAMPLES_LEAF_GRID[0],
            _MIN_SAMPLES_LEAF_GRID[-1],
            log=True,
        ),
        "min_impurity_decrease": trial.suggest_categorical(
            "min_impurity_decrease", _MIN_IMPURITY_DECREASE_GRID
        ),
        "inner_min_impurity_decrease": trial.suggest_categorical(
            "inner_min_impurity_decrease", _INNER_MIN_IMPURITY_DECREASE_GRID
        ),
    }
    if use_bivariate:
        params["pairwise_candidates"] = trial.suggest_categorical(
            "pairwise_candidates", _PAIRWISE_CANDIDATES_GRID
        )
        params["pairwise_penalty"] = trial.suggest_categorical(
            "pairwise_penalty", _PENALTY_GRID
        )
    if num_partitions > 2:
        params["branching_penalty"] = trial.suggest_categorical(
            "branching_penalty", _PENALTY_GRID
        )
    if use_tao:
        params["tao_n_runs"] = _TAO_N_RUNS
        params["tao_lambda"] = trial.suggest_float(
            "tao_lambda", _TAO_LAMBDA_GRID[-1], _TAO_LAMBDA_GRID[0], log=True
        )
    if use_tao and use_bivariate:
        params["tao_pair_scale"] = trial.suggest_float(
            "tao_pair_scale", _TAO_PAIR_SCALE_GRID[0], _TAO_PAIR_SCALE_GRID[-1]
        )
    return params
