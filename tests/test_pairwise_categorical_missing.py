"""Public acceptance tests for categorical and joint-missing pair routing."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from sgtlearn import SGTClassifier, SGTRegressor


def test_classifier_continuous_categorical_pair_routes_joint_interaction() -> None:
    states = np.array(
        [
            [-1.0, 1.0, 0.0],
            [-1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ]
    )
    counts = [40, 30, 30, 28]
    X = np.repeat(states, counts, axis=0)
    y = np.repeat([0, 1, 1, 0], counts)

    model = SGTClassifier(
        max_depth=1,
        inner_max_depth=2,
        inner_max_leaf_nodes=4,
        pairwise_candidates=1,
        tao_n_runs=0,
        random_state=0,
    ).fit(X, y, feature_dict={0: [0], 1: [1, 2]})

    assert model.score(X, y) == 1.0
    root = model.tree_export()["nodes"][0]
    assert root["routing_kind"] == "pair"
    assert root["features"] == [0, 1, 2]
    assert root["pair_axes"] == [
        {
            "logical_feature": 0,
            "kind": "continuous",
            "columns": [0],
            "categories": [],
            "catchall": None,
        },
        {
            "logical_feature": 1,
            "kind": "categorical",
            "columns": [1, 2],
            "categories": [1, 2],
            "catchall": "missing",
        },
    ]


@pytest.mark.parametrize("estimator", [SGTClassifier, SGTRegressor])
def test_continuous_pair_routes_each_joint_missing_state(estimator: type) -> None:
    levels = [-1.0, 1.0, np.nan]
    states = np.array([(first, second) for first in levels for second in levels])
    counts = [40, 31, 30, 29, 28, 27, 26, 25, 24]
    X = np.repeat(states, counts, axis=0)
    y = np.repeat(np.arange(len(states)), counts)

    model = estimator(
        num_partitions=9,
        max_depth=1,
        inner_max_depth=2,
        inner_max_leaf_nodes=5,
        pairwise_candidates=1,
        tao_n_runs=0,
        random_state=0,
    ).fit(X, y)

    np.testing.assert_array_equal(model.predict(states), np.arange(len(states)))
    inner = model.tree_export()["nodes"][0]["pair_inner_tree"]
    assert all("missing" in node for node in inner if not node["is_leaf"])
    assert any(
        not inner[node["missing"]]["is_leaf"]
        for node in inner
        if not node["is_leaf"]
    ), "a missing edge must remain splittable by the other feature"


def _mixed_states() -> tuple[np.ndarray, dict[int, list[int]], list[str]]:
    continuous = [-1.0, 1.0, np.nan]
    categorical = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
    return (
        np.array([[value, *category] for value in continuous for category in categorical]),
        {0: [0], 1: [1, 2]},
        ["continuous", "categorical"],
    )


def _categorical_states() -> tuple[np.ndarray, dict[int, list[int]], list[str]]:
    categories = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
    return (
        np.array([[*first, *second] for first in categories for second in categories]),
        {0: [0, 1], 1: [2, 3]},
        ["categorical", "categorical"],
    )


@pytest.mark.parametrize("estimator", [SGTClassifier, SGTRegressor])
@pytest.mark.parametrize("make_case", [_mixed_states, _categorical_states])
def test_categorical_pairs_route_no_active_as_axis_specific_missing(
    estimator: type,
    make_case: Callable[[], tuple[np.ndarray, dict[int, list[int]], list[str]]],
) -> None:
    states, feature_dict, expected_kinds = make_case()
    counts = [40, 31, 30, 29, 28, 27, 26, 25, 24]
    X = np.repeat(states, counts, axis=0)
    y = np.repeat(np.arange(len(states)), counts)

    model = estimator(
        num_partitions=9,
        max_depth=1,
        inner_max_depth=2,
        inner_max_leaf_nodes=5,
        pairwise_candidates=1,
        tao_n_runs=0,
        random_state=0,
    ).fit(X, y, feature_dict=feature_dict)

    np.testing.assert_array_equal(model.predict(states), np.arange(len(states)))
    root = model.tree_export()["nodes"][0]
    assert root["routing_kind"] == "pair"
    assert [axis["kind"] for axis in root["pair_axes"]] == expected_kinds
    assert all(
        axis["catchall"] == "missing"
        for axis in root["pair_axes"]
        if axis["kind"] == "categorical"
    )


@pytest.mark.parametrize("estimator", [SGTClassifier, SGTRegressor])
def test_missing_only_pair_bin_is_not_rejected_by_inner_min_leaf(
    estimator: type,
) -> None:
    finite = np.array(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]
    )
    X = np.vstack([np.repeat(finite, [40, 30, 30, 28], axis=0), [np.nan, -1.0]])
    y = np.concatenate([np.repeat([0, 1, 1, 0], [40, 30, 30, 28]), [2]])

    model = estimator(
        num_partitions=3,
        max_depth=1,
        inner_min_samples_leaf=10,
        inner_max_depth=2,
        inner_max_leaf_nodes=4,
        pairwise_candidates=1,
        tao_n_runs=0,
        random_state=0,
    ).fit(X, y)

    assert model.predict(np.array([[np.nan, -1.0]])).item() == 2


@pytest.mark.parametrize("estimator", [SGTClassifier, SGTRegressor])
def test_pair_predict_time_missing_uses_majority_inner_route(estimator: type) -> None:
    states = np.array(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]
    )
    X = np.repeat(states, [5, 20, 30, 40], axis=0)
    y = np.repeat([0, 1, 1, 0], [5, 20, 30, 40])
    model = estimator(
        max_depth=1,
        inner_max_depth=2,
        inner_max_leaf_nodes=4,
        pairwise_candidates=1,
        tao_n_runs=0,
        random_state=0,
    ).fit(X, y)

    assert model.tree_export()["nodes"][0]["routing_kind"] == "pair"
    np.testing.assert_array_equal(
        model.predict(np.array([[-1.0, np.nan], [1.0, np.nan]])),
        model.predict(np.array([[-1.0, 1.0], [1.0, 1.0]])),
    )
