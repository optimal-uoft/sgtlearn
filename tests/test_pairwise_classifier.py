"""Public acceptance tests for opt-in classifier pair routing."""

from __future__ import annotations

import numpy as np
import pytest

from sgtlearn import SGTClassifier
from sgtlearn import SGTRegressor


@pytest.mark.parametrize("estimator", [SGTClassifier, SGTRegressor])
def test_pairwise_candidates_zero_preserves_legacy_predictions(estimator) -> None:
    X = np.array([[-2.0, 0.0], [-1.0, 1.0], [1.0, -1.0], [2.0, 0.0]])
    y = np.array([0, 0, 1, 1]) if estimator is SGTClassifier else np.array([0.0, 0.0, 1.0, 1.0])
    common = dict(max_depth=2, tao_n_runs=0, random_state=0)
    implicit = estimator(**common).fit(X, y)
    explicit = estimator(pairwise_candidates=0, **common).fit(X, y)
    np.testing.assert_array_equal(implicit.predict(X), explicit.predict(X))


@pytest.mark.parametrize("estimator", [SGTClassifier, SGTRegressor])
@pytest.mark.parametrize("value", [True, -1, np.nan, np.inf, "one"])
def test_pairwise_candidates_rejects_invalid_values(estimator, value) -> None:
    y = np.array([0, 1, 0, 1]) if estimator is SGTClassifier else np.array([0.0, 1.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        estimator(pairwise_candidates=value, tao_n_runs=0).fit(np.zeros((4, 2)), y)


@pytest.mark.parametrize("estimator", [SGTClassifier, SGTRegressor])
@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf, "one"])
def test_pairwise_penalty_rejects_invalid_values(estimator, value) -> None:
    y = np.array([0, 1, 0, 1]) if estimator is SGTClassifier else np.array([0.0, 1.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        estimator(pairwise_penalty=value, tao_n_runs=0).fit(np.zeros((4, 2)), y)


@pytest.mark.parametrize("estimator", [SGTClassifier, SGTRegressor])
def test_pair_candidates_are_restricted_to_max_features_subset(estimator) -> None:
    X = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    y = np.array([0, 1, 1, 0]) if estimator is SGTClassifier else np.array([0.0, 1.0, 1.0, 0.0])
    model = estimator(
        max_features=1, pairwise_candidates=1, max_depth=1,
        inner_max_depth=2, inner_max_leaf_nodes=4, tao_n_runs=0, random_state=0,
    ).fit(X, y)
    assert model.tree_export()["nodes"][0].get("routing_kind") != "pair"


def test_exact_final_tie_favors_univariate_route() -> None:
    X = np.zeros((8, 2))
    y = np.array([0, 1] * 4)
    model = SGTClassifier(
        pairwise_candidates=1, max_depth=1, inner_max_depth=2,
        inner_max_leaf_nodes=4, tao_n_runs=0, random_state=0,
    ).fit(X, y)
    assert model.tree_export()["nodes"][0].get("routing_kind") != "pair"


def test_continuous_pair_captures_xor_and_exports_native_router() -> None:
    quadrants = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    counts = [40, 30, 30, 28]
    X = np.repeat(quadrants, counts, axis=0)
    y = np.repeat(np.array([0, 1, 1, 0]), counts)

    baseline = SGTClassifier(max_depth=1, tao_n_runs=0, random_state=0).fit(X, y)
    paired = SGTClassifier(
        max_depth=1,
        inner_max_depth=2,
        inner_max_leaf_nodes=4,
        pairwise_candidates=1,
        tao_n_runs=0,
        random_state=0,
    ).fit(X, y)

    assert paired.score(X, y) == 1.0
    assert paired.score(X, y) > baseline.score(X, y)

    root = paired.tree_export()["nodes"][0]
    assert root["routing_kind"] == "pair"
    assert root["pair_features"] == [0, 1]
    assert root["features"] == [0, 1]
    assert root["pair_inner_tree"]
    assert len(root["pair_leaf_bins"]) == len(root["bin_to_partition"])

    with pytest.warns(UserWarning, match="equally"):
        np.testing.assert_allclose(paired.feature_importances_, [0.5, 0.5])


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_pair_search_keeps_zero_gain_marginals_for_balanced_xor(
    criterion: str,
) -> None:
    states = np.array(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]
    )
    X = np.repeat(states, 40, axis=0)
    y = np.repeat([0, 1, 1, 0], 40)

    model = SGTClassifier(
        criterion=criterion,
        max_depth=1,
        inner_max_depth=2,
        inner_max_leaf_nodes=4,
        pairwise_candidates=1,
        tao_n_runs=0,
        random_state=0,
    ).fit(X, y)

    assert model.tree_export()["nodes"][0]["routing_kind"] == "pair"
    assert model.score(X, y) == 1.0


@pytest.mark.parametrize("budget", [0.3, 99])
def test_float_and_excess_pair_budgets_fit_available_interaction(
    budget: int | float,
) -> None:
    states = np.array(
        [[-1.0, -1.0, 0.0], [-1.0, 1.0, 0.0],
         [1.0, -1.0, 0.0], [1.0, 1.0, 0.0]]
    )
    X = np.repeat(states, 40, axis=0)
    y = np.repeat([0, 1, 1, 0], 40)
    model = SGTClassifier(
        max_depth=1,
        inner_max_depth=2,
        inner_max_leaf_nodes=4,
        pairwise_candidates=budget,
        tao_n_runs=0,
        random_state=0,
    ).fit(X, y)

    assert model.tree_export()["nodes"][0]["pair_features"] == [0, 1]
    assert model.score(X, y) == 1.0


def test_pair_ranking_tie_uses_logical_feature_indices() -> None:
    base = np.array(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]
    )
    X = np.repeat(np.column_stack([base, base[:, 1]]), 40, axis=0)
    y = np.repeat([0, 1, 1, 0], 40)
    model = SGTClassifier(
        max_depth=1,
        inner_max_depth=2,
        inner_max_leaf_nodes=4,
        pairwise_candidates=1,
        tao_n_runs=0,
        random_state=0,
    ).fit(X, y)

    assert model.tree_export()["nodes"][0]["pair_features"] == [0, 1]


def test_pairwise_penalty_switches_selection_without_blocking_univariate() -> None:
    states = np.array(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]
    )
    X = np.repeat(states, [40, 10, 30, 5], axis=0)
    y = np.repeat([0, 1, 1, 0], [40, 10, 30, 5])
    common = dict(
        max_depth=1,
        inner_max_depth=2,
        inner_max_leaf_nodes=4,
        pairwise_candidates=1,
        tao_n_runs=0,
        random_state=0,
    )

    pair = SGTClassifier(pairwise_penalty=0.0, **common).fit(X, y)
    univariate = SGTClassifier(pairwise_penalty=0.5, **common).fit(X, y)

    assert pair.tree_export()["nodes"][0]["routing_kind"] == "pair"
    assert univariate.tree_export()["nodes"][0].get("routing_kind") != "pair"
    assert univariate.tree_export()["num_nodes"] > 1


def test_continuous_pair_supports_three_way_routing() -> None:
    X = np.tile(
        np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]),
        (32, 1),
    )
    y = np.tile(np.array([0, 1, 1, 2]), 32)

    paired = SGTClassifier(
        num_partitions=3,
        max_depth=1,
        inner_max_depth=2,
        inner_max_leaf_nodes=4,
        pairwise_candidates=1,
        tao_n_runs=0,
        random_state=0,
    ).fit(X, y)

    assert paired.score(X, y) == 1.0
    root = paired.tree_export()["nodes"][0]
    assert root["routing_kind"] == "pair"
    assert len(root["children"]) == 3
