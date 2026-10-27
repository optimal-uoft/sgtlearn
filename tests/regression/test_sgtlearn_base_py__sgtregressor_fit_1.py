import copy

import numpy as np
import pytest

from sgtlearn import SGTRegressor


@pytest.fixture
def data():
    X = np.array([
        [-3.0, 0.0],
        [-2.0, 1.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [2.0, 1.0],
        [3.0, 0.0],
        [4.0, 1.0],
    ])
    y = np.array([-8.0, -3.0, 2.0, 9.0, 11.0, 18.0, 20.0, 31.0])
    return X, y


def one_leaf(n_samples, **kwargs):
    return SGTRegressor(
        criterion="squared_error",
        min_samples_leaf=n_samples,
        tao_n_runs=0,
        random_state=42,
        **kwargs,
    )


def assert_nested_equal(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_nested_equal(left, right)
    else:
        np.testing.assert_equal(actual, expected)


def test_fit_returns_self(data):
    X, y = data
    model = SGTRegressor(max_depth=2, tao_n_runs=0)
    assert model.fit(X, y) is model


@pytest.mark.parametrize("target_form", ["vector", "column", "multi"])
def test_prediction_shape_matches_output_count(data, target_form):
    X, y = data
    if target_form == "column":
        y = y[:, None]
    elif target_form == "multi":
        y = np.column_stack((y, 5.0 - 2.0 * y))

    model = SGTRegressor(max_depth=2, tao_n_runs=0).fit(X, y)

    expected_outputs = 2 if target_form == "multi" else 1
    expected_shape = (3, 2) if expected_outputs == 2 else (3,)
    assert model.n_features_in_ == X.shape[1]
    assert model.n_outputs_ == expected_outputs
    assert model.predict(X[:3]).shape == expected_shape


def test_missing_feature_values_are_accepted(data):
    X, y = data
    X = X.copy()
    X[1, 0] = np.nan
    X[5, 1] = np.nan

    model = one_leaf(len(y)).fit(X, y)

    np.testing.assert_allclose(model.predict(X), y.mean(), atol=1e-5)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("multiple_outputs", [False, True])
def test_one_leaf_predicts_training_target_mean(data, weighted, multiple_outputs):
    X, y = data
    if multiple_outputs:
        y = np.column_stack((y, 100.0 + 3.0 * y))
    weights = np.array([0., 1., 2., 3., 0., 5., 1., 4.]) if weighted else None

    model = one_leaf(len(y)).fit(X, y, sample_weight=weights)
    expected = np.average(y, axis=0, weights=weights)

    np.testing.assert_allclose(
        model.predict(X),
        np.broadcast_to(expected, y.shape),
        rtol=2e-5,
        atol=2e-5,
    )


@pytest.mark.parametrize("multiple_outputs", [False, True])
def test_exported_leaf_values_use_original_target_units(data, multiple_outputs):
    X, y = data
    if multiple_outputs:
        y = np.column_stack((y, 100.0 + 3.0 * y))
    weights = np.arange(1.0, len(y) + 1.0)
    model = one_leaf(len(y)).fit(X, y, sample_weight=weights)

    nodes = model.tree_export()["nodes"]
    assert len(nodes) == 1
    np.testing.assert_allclose(
        np.asarray(nodes[0]["value"]).reshape(-1),
        np.asarray(np.average(y, axis=0, weights=weights)).reshape(-1),
        rtol=2e-5,
        atol=2e-5,
    )


@pytest.mark.parametrize("weighted", [False, True])
def test_one_leaf_impurity_uses_standardized_population_variance(data, weighted):
    X, y = data
    weights = np.arange(1.0, len(y) + 1.0) if weighted else None
    model = one_leaf(len(y)).fit(X, y, sample_weight=weights)

    assert model.tree_export()["nodes"][0]["impurity"] == pytest.approx(
        1.0, rel=2e-5, abs=2e-5
    )


@pytest.mark.parametrize("tao_n_runs", [0, 1])
@pytest.mark.parametrize("multiple_outputs", [False, True])
def test_constant_target_columns_predict_their_original_constant(
    data, tao_n_runs, multiple_outputs
):
    X, y = data
    target = (
        np.column_stack((y, np.full(len(y), 137.5)))
        if multiple_outputs
        else np.full(len(y), 137.5)
    )
    model = SGTRegressor(
        max_depth=2, tao_n_runs=tao_n_runs, random_state=42
    ).fit(X, target)

    prediction = model.predict(X)
    constant_prediction = prediction[:, 1] if multiple_outputs else prediction
    np.testing.assert_allclose(constant_prediction, 137.5, atol=2e-5)


@pytest.mark.parametrize("tao_n_runs", [0, 1])
def test_fit_does_not_mutate_input_arrays(data, tao_n_runs):
    X, y = data
    y = np.column_stack((y, 100.0 + 3.0 * y))
    weights = np.arange(1.0, len(y) + 1.0)
    originals = [array.copy() for array in (X, y, weights)]

    SGTRegressor(
        max_depth=2, tao_n_runs=tao_n_runs, random_state=42
    ).fit(X, y, sample_weight=weights)

    for actual, original in zip((X, y, weights), originals):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize(
    "bad_y",
    [
        np.arange(7.0),
        np.ones((7, 2)),
        np.ones((8, 1, 1)),
        np.array(["invalid"] * 8),
        np.array([0., 1., 2., np.nan, 4., 5., 6., 7.]),
        np.array([0., 1., 2., np.inf, 4., 5., 6., 7.]),
        np.array([0., 1., 2., -np.inf, 4., 5., 6., 7.]),
        np.full(8, float(np.finfo(np.float32).max) * 2.0),
        np.full(8, -float(np.finfo(np.float32).max) * 2.0),
    ],
    ids=[
        "length-mismatch", "multioutput-length-mismatch", "rank-three",
        "nonnumeric", "nan", "positive-infinity", "negative-infinity",
        "positive-float32-overflow", "negative-float32-overflow",
    ],
)
def test_invalid_targets_raise_value_error(data, bad_y):
    X, _ = data
    with pytest.raises(ValueError):
        SGTRegressor(tao_n_runs=0).fit(X, bad_y)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf, 1e40])
def test_invalid_values_in_later_target_columns_raise_value_error(data, bad_value):
    X, y = data
    target = np.column_stack((y, y + 100.0))
    target[3, 1] = bad_value

    with pytest.raises(ValueError):
        SGTRegressor(tao_n_runs=0).fit(X, target)


@pytest.mark.parametrize(
    "weights",
    [
        np.ones(7),
        np.zeros(8),
        np.array([1., 1., -1., 1., 1., 1., 1., 1.]),
        np.array([1., 1., np.nan, 1., 1., 1., 1., 1.]),
        np.array([1., 1., np.inf, 1., 1., 1., 1., 1.]),
        np.array([1., 1., -np.inf, 1., 1., 1., 1., 1.]),
    ],
    ids=["length", "all-zero", "negative", "nan", "positive-inf", "negative-inf"],
)
def test_invalid_sample_weights_raise_value_error(data, weights):
    X, y = data
    with pytest.raises(ValueError):
        SGTRegressor(tao_n_runs=0).fit(X, y, sample_weight=weights)


@pytest.mark.parametrize("shape", [(8,), (8, 2, 1)])
def test_features_must_be_a_matrix(data, shape):
    _, y = data
    with pytest.raises(ValueError):
        SGTRegressor(tao_n_runs=0).fit(np.zeros(shape), y)


@pytest.mark.parametrize("overflow", [1e40, -1e40])
def test_features_outside_float32_range_raise_value_error(data, overflow):
    X, y = data
    X = X.copy()
    X[2, 1] = overflow

    with pytest.raises(ValueError):
        SGTRegressor(tao_n_runs=0).fit(X, y)


@pytest.mark.parametrize("failure", ["target", "weights", "shape", "feature-range"])
def test_failed_refit_preserves_predictions_and_export(data, failure):
    X, y = data
    model = SGTRegressor(max_depth=2, tao_n_runs=0).fit(X, y)
    prediction_before = model.predict(X).copy()
    export_before = copy.deepcopy(model.tree_export())

    new_X = X.copy()
    new_y = 100.0 + 7.0 * y
    weights = np.ones(len(y))
    if failure == "target":
        new_y[-1] = np.nan
    elif failure == "weights":
        weights[-1] = -1.0
    elif failure == "shape":
        new_y = new_y[:-1]
    else:
        new_X[-1, 0] = 1e40

    with pytest.raises(ValueError):
        model.fit(new_X, new_y, sample_weight=weights)

    np.testing.assert_array_equal(model.predict(X), prediction_before)
    assert_nested_equal(model.tree_export(), export_before)


def test_successful_refit_replaces_training_statistics(data):
    X, y = data
    model = one_leaf(len(y)).fit(X, y)
    new_y = 200.0 + 7.0 * y
    weights = np.arange(1.0, len(y) + 1.0)

    model.fit(X, new_y, sample_weight=weights)

    expected = np.average(new_y, weights=weights)
    np.testing.assert_allclose(model.predict(X), expected, rtol=2e-5)
    np.testing.assert_allclose(
        model.tree_export()["nodes"][0]["value"], expected, rtol=2e-5
    )


def test_prediction_batches_do_not_change_fitted_scale(data):
    X, y = data
    model = SGTRegressor(max_depth=2, tao_n_runs=0).fit(X, y)
    expected = model.predict(X).copy()
    export_before = copy.deepcopy(model.tree_export())

    individually = np.array([model.predict(row[None, :])[0] for row in X])
    model.predict(np.full((5, X.shape[1]), 1000.0))

    np.testing.assert_allclose(individually, expected, atol=1e-6)
    np.testing.assert_array_equal(model.predict(X), expected)
    assert_nested_equal(model.tree_export(), export_before)
