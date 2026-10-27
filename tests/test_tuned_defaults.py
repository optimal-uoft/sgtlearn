"""The chosen common configuration is the public out-of-the-box contract."""

import subprocess
import sys

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sgtlearn import (
    RandomSGForestClassifier,
    RandomSGForestRegressor,
    SGTClassifier,
    SGTRegressor,
)

SELECTED = {
    "max_depth": 4,
    "num_partitions": 7,
    "max_leaf_nodes": None,
    "min_samples_leaf": 4,
    "min_impurity_decrease": 1.0,
    "inner_max_depth": None,
    "inner_max_leaf_nodes": 4,
    "inner_min_samples_leaf": 13,
    "inner_min_impurity_decrease": 0.01,
    "coordinate_descent_max_iters": 5,
    "coordinate_descent_patience": 5,
    "pairwise_candidates": 0.5,
    "pairwise_penalty": 0.0,
    "branching_penalty": 0.0,
    "tao_n_runs": 1,
    "tao_lambda": 1e-4,
    "tao_pair_scale": 2.0,
}


@pytest.mark.parametrize(
    "estimator",
    [SGTClassifier, SGTRegressor, RandomSGForestClassifier, RandomSGForestRegressor],
)
def test_public_defaults_match_selected_pareto_configuration(estimator):
    model = estimator()
    for key, value in SELECTED.items():
        assert model.get_params()[key] == value, key
    assert clone(model).get_params() == model.get_params()


@pytest.mark.parametrize(
    "estimator,reference",
    [
        (RandomSGForestClassifier, RandomForestClassifier),
        (RandomSGForestRegressor, RandomForestRegressor),
    ],
)
def test_forest_specific_defaults_match_sklearn(estimator, reference):
    params, expected = estimator().get_params(), reference().get_params()
    for key in (
        "n_estimators",
        "bootstrap",
        "max_samples",
        "n_jobs",
        "verbose",
        "max_features",
        "random_state",
    ):
        assert params[key] == expected[key], key


def test_default_trees_fit_both_tasks_and_return_original_regression_units():
    X = np.arange(300.0)[:, None]
    labels = np.where(X[:, 0] < 150, "low", "high")
    classifier = SGTClassifier().fit(X, labels)
    assert classifier.score(X, labels) > 0.9
    targets = 1000 + 3 * X[:, 0]
    regressor = SGTRegressor().fit(X, targets)
    prediction = regressor.predict(X)
    assert prediction.shape == targets.shape
    assert regressor.score(X, targets) > 0.9
    assert prediction.min() > 900


def test_regular_import_does_not_require_or_load_optuna():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import sgtlearn; assert 'optuna' not in sys.modules",
        ],
        check=True,
    )
