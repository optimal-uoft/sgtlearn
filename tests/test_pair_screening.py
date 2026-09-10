"""Pair eligibility depends on raw improvement, independently of growth costs."""

import numpy as np
import pytest

from sgtlearn import SGTClassifier, SGTRegressor


def test_or_pair_survives_unprofitable_univariate_growth():
    X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = np.array([0, 1, 1, 1])
    # Each feature improves total Gini by 0.5; the pair improves it by 1.5.
    model = SGTClassifier(
        max_depth=1, inner_max_depth=2, inner_max_leaf_nodes=4,
        min_impurity_decrease=0.8, pairwise_candidates=1, tao_n_runs=0,
    ).fit(X, y)
    assert model.tree_export()["nodes"][0]["features"] == [0, 1]
    np.testing.assert_array_equal(model.predict(X), y)


@pytest.mark.parametrize("estimator,criterion,alpha", [
    (SGTClassifier, "gini", 0.8),
    (SGTClassifier, "entropy", 1.5),
    (SGTRegressor, "squared_error", 0.5),
    (SGTRegressor, "absolute_error", 1.5),
])
@pytest.mark.parametrize("outputs", [1, 2])
def test_weighted_pair_screening_is_independent_of_growth_costs(
    estimator, criterion, alpha, outputs, monkeypatch,
):
    monkeypatch.setenv("SGTLEARN_MAE_CD", "1")
    X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = np.array([0, 1, 2, 0] if criterion == "absolute_error" else [0, 1, 1, 1])
    if outputs == 2:
        y = np.column_stack([y, y.max() - y])
    kwargs = dict(
        criterion=criterion, num_partitions=3, max_leaf_nodes=3, max_depth=1,
        inner_max_depth=2, inner_max_leaf_nodes=4, min_impurity_decrease=alpha,
        pairwise_candidates=1, tao_n_runs=0,
    )
    weights = np.array([1., 2., 2., 1.])
    model = estimator(**kwargs).fit(X, y, sample_weight=weights)
    assert model.tree_export()["nodes"][0]["features"] == [0, 1]
    np.testing.assert_allclose(model.predict(X), y)
    # Admission does not authorize growth: gamma still must be paid afterward.
    penalized = estimator(**kwargs, pairwise_penalty=100.).fit(X, y, sample_weight=weights)
    assert penalized.tree_export()["nodes"][0]["is_leaf"]


@pytest.mark.parametrize("minimum_leaf", [1, 3])
def test_raw_zero_gain_or_infeasible_features_do_not_create_pairs(minimum_leaf):
    X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    # XOR has zero univariate gain. In the OR case, each feature isolates only
    # two samples, although the pair could separate four negatives/four positives.
    y = np.array([0, 1, 1, 0] if minimum_leaf == 1 else [0, 1, 1, 1])
    if minimum_leaf == 3:
        X, y = np.repeat(X, [4, 2, 2, 0], axis=0), np.repeat(y, [4, 2, 2, 0])
    model = SGTClassifier(
        max_depth=1, min_samples_leaf=minimum_leaf, pairwise_candidates=1,
        inner_max_depth=2, inner_max_leaf_nodes=4, tao_n_runs=0,
    ).fit(X, y)
    assert model.tree_export()["nodes"][0]["is_leaf"]
