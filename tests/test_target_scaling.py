"""End-to-end target standardization invariants for regressors and forests."""

from __future__ import annotations

import copy

import numpy as np
import pytest

import sgtlearn.tao
from sgtlearn import RandomSGForestRegressor, SGTRegressor


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(2718)
    X = rng.normal(size=(48, 3))
    y = np.column_stack(
        (
            20.0 + 4.0 * X[:, 0] - 2.0 * X[:, 1],
            -35.0 + 3.0 * X[:, 2] + X[:, 0],
        )
    )
    query = rng.normal(size=(13, 3))
    return X, y, query


@pytest.mark.parametrize("multiple_outputs", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("tao_n_runs", [0, 2])
def test_target_unit_changes_preserve_inverse_transformed_tree_predictions(
    regression_data, multiple_outputs, weighted, tao_n_runs
):
    X, y, query = regression_data
    if not multiple_outputs:
        y = y[:, 0]
    scale = np.array([2.0, 0.5]) if multiple_outputs else 2.0
    offset = np.array([64.0, -16.0]) if multiple_outputs else 64.0
    weights = np.linspace(0.5, 2.0, len(X)) if weighted else None
    parameters = {
        "max_depth": 2,
        "inner_max_leaf_nodes": 3,
        "random_state": 19,
        "tao_n_runs": tao_n_runs,
        "tao_lambda": 0.05,
    }
    original = SGTRegressor(**parameters).fit(X, y, sample_weight=weights)
    converted = SGTRegressor(**parameters).fit(
        X, y * scale + offset, sample_weight=weights
    )

    np.testing.assert_allclose(
        (converted.predict(query) - offset) / scale,
        original.predict(query),
        rtol=2e-4,
        atol=2e-4,
    )


@pytest.mark.parametrize("multiple_outputs", [False, True])
def test_later_tree_refinement_accepts_original_target_units(
    regression_data, multiple_outputs
):
    X, y, query = regression_data
    if not multiple_outputs:
        y = y[:, 0]
    scale = np.array([2.0, 0.5]) if multiple_outputs else 2.0
    offset = np.array([64.0, -16.0]) if multiple_outputs else 64.0
    parameters = {
        "max_depth": 2,
        "inner_max_leaf_nodes": 3,
        "tao_n_runs": 0,
        "random_state": 19,
    }
    original = SGTRegressor(**parameters).fit(X, y)
    converted = SGTRegressor(**parameters).fit(X, y * scale + offset)

    sgtlearn.tao.TAO_refine(original, X, y, n_runs=2, lambda_=0.05)
    sgtlearn.tao.TAO_refine(
        converted, X, y * scale + offset, n_runs=2, lambda_=0.05
    )

    np.testing.assert_allclose(
        (converted.predict(query) - offset) / scale,
        original.predict(query),
        rtol=2e-4,
        atol=2e-4,
    )


@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize("tao_n_runs", [0, 1])
def test_target_unit_changes_preserve_inverse_transformed_forest_predictions(
    regression_data, n_jobs, tao_n_runs
):
    X, y, query = regression_data
    scale = np.array([2.0, 0.5])
    offset = np.array([64.0, -16.0])
    parameters = {
        "n_estimators": 3,
        "max_depth": 2,
        "inner_max_leaf_nodes": 3,
        "random_state": 19,
        "bootstrap": True,
        "tao_n_runs": tao_n_runs,
        "tao_lambda": 0.05,
        "n_jobs": n_jobs,
    }
    original = RandomSGForestRegressor(**parameters).fit(X, y)
    converted = RandomSGForestRegressor(**parameters).fit(
        X, y * scale + offset
    )

    np.testing.assert_allclose(
        (converted.predict(query) - offset) / scale,
        original.predict(query),
        rtol=2e-4,
        atol=2e-4,
    )


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_later_forest_refinement_accepts_original_target_units(
    regression_data, n_jobs
):
    X, y, query = regression_data
    scale = np.array([2.0, 0.5])
    offset = np.array([64.0, -16.0])
    parameters = {
        "n_estimators": 3,
        "max_depth": 2,
        "inner_max_leaf_nodes": 3,
        "random_state": 19,
        "bootstrap": False,
        "tao_n_runs": 0,
        "n_jobs": n_jobs,
    }
    original = RandomSGForestRegressor(**parameters).fit(X, y)
    converted = RandomSGForestRegressor(**parameters).fit(
        X, y * scale + offset
    )

    sgtlearn.tao.TAO_refine(original, X, y, n_runs=1, lambda_=0.05)
    sgtlearn.tao.TAO_refine(
        converted, X, y * scale + offset, n_runs=1, lambda_=0.05
    )

    np.testing.assert_allclose(
        (converted.predict(query) - offset) / scale,
        original.predict(query),
        rtol=2e-4,
        atol=2e-4,
    )


@pytest.mark.parametrize("multiple_outputs", [False, True])
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_forest_constant_targets_keep_original_units_and_output_shape(
    regression_data, multiple_outputs, n_jobs
):
    X, _, query = regression_data
    constants = np.array([137.5, -23.0]) if multiple_outputs else 137.5
    shape = (len(X), 2) if multiple_outputs else (len(X),)
    target = np.broadcast_to(constants, shape).copy()
    model = RandomSGForestRegressor(
        n_estimators=3,
        max_depth=2,
        random_state=19,
        tao_n_runs=1,
        n_jobs=n_jobs,
    ).fit(X, target)

    prediction = model.predict(query)
    expected_shape = (len(query), 2) if multiple_outputs else (len(query),)
    assert prediction.shape == expected_shape
    np.testing.assert_allclose(
        prediction,
        np.broadcast_to(constants, expected_shape),
        rtol=2e-5,
        atol=2e-5,
    )


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("multiple_outputs", [False, True])
def test_one_leaf_predicts_and_exports_original_target_units(
    regression_data, weighted, multiple_outputs
):
    X, y, _ = regression_data
    if not multiple_outputs:
        y = y[:, 0]
    weights = np.linspace(0.5, 2.0, len(X)) if weighted else None

    model = SGTRegressor(
        criterion="squared_error",
        min_samples_leaf=len(X),
        tao_n_runs=0,
        random_state=42,
    ).fit(X, y, sample_weight=weights)

    expected = np.average(y, axis=0, weights=weights)
    np.testing.assert_allclose(
        model.predict(X),
        np.broadcast_to(expected, y.shape),
        rtol=2e-5,
        atol=2e-5,
    )
    nodes = model.tree_export()["nodes"]
    assert len(nodes) == 1
    np.testing.assert_allclose(
        np.asarray(nodes[0]["value"]).reshape(-1),
        np.asarray(expected).reshape(-1),
        rtol=2e-5,
        atol=2e-5,
    )
    if not multiple_outputs:
        assert nodes[0]["impurity"] == pytest.approx(1.0, rel=2e-4, abs=2e-4)


@pytest.mark.parametrize("tao_n_runs", [0, 1])
@pytest.mark.parametrize("multiple_outputs", [False, True])
def test_tree_constant_target_column_keeps_original_constant(
    regression_data, tao_n_runs, multiple_outputs
):
    X, y, _ = regression_data
    target = (
        np.column_stack((y[:, 0], np.full(len(X), 137.5)))
        if multiple_outputs
        else np.full(len(X), 137.5)
    )
    model = SGTRegressor(
        max_depth=2, tao_n_runs=tao_n_runs, random_state=42
    ).fit(X, target)

    prediction = model.predict(X)
    constant_prediction = prediction[:, 1] if multiple_outputs else prediction
    np.testing.assert_allclose(constant_prediction, 137.5, atol=2e-5)


def test_refit_replaces_target_scaling_statistics(regression_data):
    X, y, _ = regression_data
    y0 = y[:, 0]
    model = SGTRegressor(
        criterion="squared_error",
        min_samples_leaf=len(X),
        tao_n_runs=0,
        random_state=42,
    ).fit(X, y0)
    new_y = 200.0 + 7.0 * y0
    weights = np.linspace(1.0, 2.0, len(X))

    model.fit(X, new_y, sample_weight=weights)

    expected = np.average(new_y, weights=weights)
    np.testing.assert_allclose(model.predict(X), expected, rtol=2e-5)
    np.testing.assert_allclose(
        model.tree_export()["nodes"][0]["value"], expected, rtol=2e-5
    )


def test_prediction_batches_do_not_mutate_fitted_scale(regression_data):
    X, y, _ = regression_data
    model = SGTRegressor(max_depth=2, tao_n_runs=0, random_state=42).fit(
        X, y[:, 0]
    )
    expected = model.predict(X).copy()
    export_before = copy.deepcopy(model.tree_export())

    individually = np.array([model.predict(row[None, :])[0] for row in X])
    model.predict(np.full((5, X.shape[1]), 1000.0))

    np.testing.assert_allclose(individually, expected, atol=1e-6)
    np.testing.assert_array_equal(model.predict(X), expected)
    assert model.tree_export() == export_before
