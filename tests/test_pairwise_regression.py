"""Public acceptance tests for opt-in regression pair routing."""

from __future__ import annotations

import numpy as np
import pytest

from sgtlearn import SGTRegressor


def _three_way_quadrants() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.tile(
        np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]),
        (32, 1),
    )
    return (
        X,
        np.tile(np.array([1.0, 0.0, 0.0, 2.0]), 32),
        np.tile(np.array([2.0, 1.0, 1.0, 2.0]), 32),
    )


@pytest.mark.parametrize("criterion", ["squared_error", "absolute_error"])
def test_continuous_pair_regression_uses_three_way_router(criterion: str) -> None:
    X, y, sample_weight = _three_way_quadrants()

    baseline = SGTRegressor(
        criterion=criterion, num_partitions=3, max_depth=1, tao_n_runs=0, random_state=0
    ).fit(X, y, sample_weight=sample_weight)
    paired = SGTRegressor(
        criterion=criterion,
        num_partitions=3,
        max_depth=1,
        inner_max_depth=2,
        inner_max_leaf_nodes=4,
        pairwise_candidates=1,
        tao_n_runs=0,
        random_state=0,
    ).fit(X, y, sample_weight=sample_weight)

    paired_error = np.mean(np.abs(paired.predict(X) - y))
    baseline_error = np.mean(np.abs(baseline.predict(X) - y))
    root = paired.tree_export()["nodes"][0]
    assert root["routing_kind"] == "pair", root
    assert paired_error == 0.0
    assert paired_error < baseline_error

    assert root["pair_features"] == [0, 1]
    assert root["features"] == [0, 1]
    assert root["pair_inner_tree"]
    assert len(root["children"]) == 3
    assert all(0 <= part < 3 for part in root["bin_to_partition"])


@pytest.mark.parametrize("criterion", ["squared_error", "absolute_error"])
def test_continuous_pair_regression_preserves_multioutput_contract(
    criterion: str,
) -> None:
    X, y0, sample_weight = _three_way_quadrants()
    y = np.column_stack([y0, 10.0 + y0])

    paired = SGTRegressor(
        criterion=criterion,
        num_partitions=3,
        max_depth=1,
        inner_max_depth=2,
        inner_max_leaf_nodes=4,
        pairwise_candidates=1,
        tao_n_runs=0,
        random_state=0,
    ).fit(X, y, sample_weight=sample_weight)

    np.testing.assert_allclose(paired.predict(X), y, rtol=0, atol=0)
    root = paired.tree_export()["nodes"][0]
    assert root["routing_kind"] == "pair"
    assert len(root["children"]) == 3


def test_regression_pair_importances_warn_and_split_gain_equally() -> None:
    X, y, sample_weight = _three_way_quadrants()
    model = SGTRegressor(
        num_partitions=3,
        max_depth=1,
        inner_max_depth=2,
        inner_max_leaf_nodes=4,
        pairwise_candidates=1,
        tao_n_runs=0,
        random_state=0,
    ).fit(X, y, sample_weight=sample_weight)

    with pytest.warns(UserWarning, match="equally"):
        np.testing.assert_allclose(model.feature_importances_, [0.5, 0.5])
