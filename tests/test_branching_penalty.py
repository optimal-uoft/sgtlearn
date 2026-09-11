"""Public branching-penalty API contracts."""

import numpy as np
import pytest
from sklearn.base import clone

from sgtlearn import (
    RandomSGForestClassifier,
    RandomSGForestRegressor,
    SGTClassifier,
    SGTRegressor,
)


@pytest.mark.parametrize(
    "estimator",
    [SGTClassifier, SGTRegressor, RandomSGForestClassifier, RandomSGForestRegressor],
)
def test_branching_penalty_is_cloneable(estimator):
    model = estimator(branching_penalty=1.25)
    assert model.get_params()["branching_penalty"] == 1.25
    assert clone(model).get_params()["branching_penalty"] == 1.25


@pytest.mark.parametrize(
    "estimator", [RandomSGForestClassifier, RandomSGForestRegressor]
)
def test_forest_forwards_branching_penalty(estimator):
    y = (
        np.array([0, 0, 1, 1])
        if estimator is RandomSGForestClassifier
        else np.array([0.0, 0.0, 1.0, 1.0])
    )
    forest = estimator(
        branching_penalty=1.25,
        n_estimators=2,
        bootstrap=False,
        tao_n_runs=0,
        max_depth=1,
        n_jobs=1,
    ).fit(np.array([[0.0], [0.0], [1.0], [1.0]]), y)
    assert forest.branching_penalty == 1.25
    assert all(tree.branching_penalty == 1.25 for tree in forest.estimators_)


@pytest.mark.parametrize("estimator", [SGTClassifier, SGTRegressor])
def test_branching_penalty_charges_only_extra_ternary_child(estimator):
    y = (
        np.array([0, 0, 1, 1, 2, 2])
        if estimator is SGTClassifier
        else np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
    )
    X = np.array([[0.0], [0.0], [1.0], [1.0], [2.0], [2.0]])
    common = dict(
        num_partitions=3,
        max_depth=1,
        inner_max_depth=2,
        inner_max_leaf_nodes=3,
        tao_n_runs=0,
    )
    binary = estimator(**common, branching_penalty=0.0).fit(X, y).tree_export()
    penalized = estimator(**common, branching_penalty=10.0).fit(X, y).tree_export()
    assert binary["num_nodes"] == 4
    assert len(binary["nodes"][0]["children"]) == 3
    assert penalized["num_nodes"] == 3
    assert len(penalized["nodes"][0]["children"]) == 2
