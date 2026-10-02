import numpy as np
import pytest

from sgtlearn import RandomSGForestRegressor
from tests.constants import TEST_TAO_N_RUNS

_ATTRS = (
    "estimators_",
    "n_outputs_",
    "n_features_in_",
    "feature_names_in_",
    "processed_features_",
)


def _make_regressor() -> RandomSGForestRegressor:
    return RandomSGForestRegressor(n_estimators=3, tao_n_runs=TEST_TAO_N_RUNS, random_state=0)


def _training_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(0)
    X = rng.uniform(-1.0, 1.0, size=(30, 3))
    y = X[:, 0] + 2.0 * X[:, 1]
    return X, y


def _fitted_regressor() -> RandomSGForestRegressor:
    X, y = _training_data()
    return _make_regressor().fit(X, y)


def test_successful_call_does_not_set_fitted_attributes() -> None:
    reg = _make_regressor()
    X, y = _training_data()

    reg._check_X_y(X, y)

    for attr in _ATTRS:
        assert not hasattr(reg, attr)


def test_successful_call_does_not_change_params() -> None:
    reg = _make_regressor()
    params_before = reg.get_params()
    X, y = _training_data()

    reg._check_X_y(X, y)

    assert reg.get_params() == params_before


def test_successful_call_on_fitted_forest_keeps_attributes() -> None:
    reg = _fitted_regressor()
    before = {attr: getattr(reg, attr) for attr in _ATTRS}
    X_new = np.arange(10, dtype=np.float64).reshape(5, 2)
    y_new = np.array([[1.0, 2.0]] * 5)

    reg._check_X_y(X_new, y_new)

    for attr in _ATTRS:
        assert getattr(reg, attr) is before[attr]


def test_failed_call_on_fitted_forest_keeps_attributes() -> None:
    reg = _fitted_regressor()
    before = {attr: getattr(reg, attr) for attr in _ATTRS}
    X_bad = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_bad = np.array([1.0, 1e39])

    with pytest.raises(ValueError):
        reg._check_X_y(X_bad, y_bad)

    for attr in _ATTRS:
        assert getattr(reg, attr) is before[attr]


def test_failed_call_on_fitted_forest_keeps_predictions() -> None:
    reg = _fitted_regressor()
    X, _ = _training_data()
    preds_before = reg.predict(X)

    with pytest.raises(ValueError):
        reg._check_X_y(np.array([[np.inf, 1.0], [2.0, 3.0]]), np.array([1.0, 2.0]))

    np.testing.assert_array_equal(reg.predict(X), preds_before)


def test_rejected_refit_with_y_float32_overflow_keeps_fitted_attributes() -> None:
    reg = _fitted_regressor()
    before = {attr: getattr(reg, attr) for attr in _ATTRS}
    X_bad = np.arange(10, dtype=np.float64).reshape(5, 2)
    y_bad = np.array([1.0, 2.0, 3.0, 4.0, 1e39])

    with pytest.raises(ValueError, match="float32"):
        reg.fit(X_bad, y_bad)

    for attr in _ATTRS:
        assert getattr(reg, attr) is before[attr]


def test_rejected_refit_with_x_float32_overflow_keeps_predictions() -> None:
    reg = _fitted_regressor()
    X, _ = _training_data()
    preds_before = reg.predict(X)
    X_bad = np.arange(10, dtype=np.float64).reshape(5, 2)
    X_bad[0, 0] = 1e39

    with pytest.raises(ValueError, match="float32"):
        reg.fit(X_bad, np.arange(5, dtype=np.float64))

    np.testing.assert_array_equal(reg.predict(X), preds_before)


def test_input_arrays_are_not_modified() -> None:
    X = np.array([[1.0, np.nan], [3.0, 4.0]])
    y = np.array([1.0, 2.0])
    X_copy = X.copy()
    y_copy = y.copy()

    _make_regressor()._check_X_y(X, y)

    np.testing.assert_array_equal(X, X_copy)
    np.testing.assert_array_equal(y, y_copy)
