"""Public pairwise-parameter behavior for random SGT forests."""

from __future__ import annotations

import numpy as np
import pytest
import warnings

from sgtlearn import RandomSGForestClassifier, RandomSGForestRegressor


def test_classifier_forest_propagates_pairwise_options_and_fits_interaction() -> None:
    quadrants = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    counts = [40, 30, 30, 28]
    X = np.repeat(quadrants, counts, axis=0)
    y = np.repeat(np.array([0, 1, 1, 0]), counts)

    forest = RandomSGForestClassifier(
        n_estimators=2,
        bootstrap=False,
        max_depth=1,
        inner_max_depth=2,
        inner_max_leaf_nodes=4,
        pairwise_candidates=1,
        pairwise_penalty=0.25,
        max_features=None,
        tao_n_runs=0,
        random_state=0,
    ).fit(X, y)

    assert forest.score(X, y) == 1.0
    assert all(tree.pairwise_candidates == 1 for tree in forest.estimators_)
    assert all(tree.pairwise_penalty == 0.25 for tree in forest.estimators_)
    with pytest.warns(UserWarning, match="equally") as caught:
        forest.mean_feature_importances_
    assert len(caught) == 1


def test_classifier_forest_pairwise_zero_does_not_warn_for_importances() -> None:
    X = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    y = np.array([0, 1, 1, 0])
    forest = RandomSGForestClassifier(
        n_estimators=1, bootstrap=False, max_depth=1, tao_n_runs=0, random_state=0
    ).fit(X, y)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        forest.mean_feature_importances_
    assert len(caught) == 0


def test_regressor_forest_accepts_pairwise_options() -> None:
    forest = RandomSGForestRegressor(pairwise_candidates=2, pairwise_penalty=0.5)
    assert forest.get_params()["pairwise_candidates"] == 2
    assert forest.get_params()["pairwise_penalty"] == 0.5


def test_regressor_forest_emits_one_pair_importance_warning() -> None:
    states = np.array(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]
    )
    X = np.tile(states, (32, 1))
    y = np.tile([1.0, 0.0, 0.0, 2.0], 32)
    forest = RandomSGForestRegressor(
        n_estimators=2,
        bootstrap=False,
        max_features=None,
        num_partitions=3,
        max_depth=1,
        inner_max_depth=2,
        inner_max_leaf_nodes=4,
        pairwise_candidates=1,
        tao_n_runs=0,
        random_state=0,
    ).fit(X, y)

    with pytest.warns(UserWarning, match="equally") as caught:
        forest.mean_feature_importances_
    assert len(caught) == 1
