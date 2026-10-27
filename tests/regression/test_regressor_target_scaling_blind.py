import numpy as np
import pytest

import sgtlearn.tao
from sgtlearn import RandomSGForestRegressor, SGTRegressor


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(2718)
    X = rng.normal(size=(48, 3))
    y = np.column_stack((
        20.0 + 4.0 * X[:, 0] - 2.0 * X[:, 1],
        -35.0 + 3.0 * X[:, 2] + X[:, 0],
    ))
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
    parameters = dict(
        max_depth=2,
        inner_max_leaf_nodes=3,
        random_state=19,
        tao_n_runs=tao_n_runs,
        tao_lambda=0.05,
    )
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
    parameters = dict(
        max_depth=2, inner_max_leaf_nodes=3, tao_n_runs=0, random_state=19
    )
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
    parameters = dict(
        n_estimators=3,
        max_depth=2,
        inner_max_leaf_nodes=3,
        random_state=19,
        bootstrap=True,
        tao_n_runs=tao_n_runs,
        tao_lambda=0.05,
        n_jobs=n_jobs,
    )
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
    parameters = dict(
        n_estimators=3,
        max_depth=2,
        inner_max_leaf_nodes=3,
        random_state=19,
        bootstrap=False,
        tao_n_runs=0,
        n_jobs=n_jobs,
    )
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
