import numpy as np
import pytest

from sgtlearn import RandomSGForestRegressor
from tests.constants import TEST_TAO_N_RUNS


def _make_regressor() -> RandomSGForestRegressor:
    return RandomSGForestRegressor(n_estimators=3, tao_n_runs=TEST_TAO_N_RUNS, random_state=0)


def test_empty_y_raises_value_error() -> None:
    X = np.arange(4, dtype=np.float64).reshape(2, 2)
    y = np.empty((0,))

    with pytest.raises(ValueError):
        _make_regressor()._check_X_y(X, y)


def test_three_dimensional_y_raises_value_error() -> None:
    X = np.arange(4, dtype=np.float64).reshape(2, 2)
    y = np.ones((2, 2, 2))

    with pytest.raises(ValueError):
        _make_regressor()._check_X_y(X, y)


def test_non_numeric_y_raises_value_error() -> None:
    X = np.arange(4, dtype=np.float64).reshape(2, 2)
    y = np.array(["a", "b"], dtype=object)

    with pytest.raises(ValueError):
        _make_regressor()._check_X_y(X, y)


def test_nan_in_y_raises_value_error() -> None:
    X = np.arange(6, dtype=np.float64).reshape(3, 2)
    y = np.array([1.0, np.nan, 3.0])

    with pytest.raises(ValueError):
        _make_regressor()._check_X_y(X, y)


def test_positive_infinity_in_y_raises_value_error() -> None:
    X = np.arange(6, dtype=np.float64).reshape(3, 2)
    y = np.array([1.0, np.inf, 3.0])

    with pytest.raises(ValueError):
        _make_regressor()._check_X_y(X, y)


def test_negative_infinity_in_y_raises_value_error() -> None:
    X = np.arange(6, dtype=np.float64).reshape(3, 2)
    y = np.array([1.0, 2.0, -np.inf])

    with pytest.raises(ValueError):
        _make_regressor()._check_X_y(X, y)


def test_nan_in_multi_output_y_raises_value_error() -> None:
    X = np.arange(6, dtype=np.float64).reshape(3, 2)
    y = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]])

    with pytest.raises(ValueError):
        _make_regressor()._check_X_y(X, y)


def test_y_positive_float32_overflow_raises_value_error_naming_float32() -> None:
    X = np.arange(6, dtype=np.float64).reshape(3, 2)
    y = np.array([1.0, 1e39, 3.0])

    with pytest.raises(ValueError, match="float32"):
        _make_regressor()._check_X_y(X, y)


def test_y_negative_float32_overflow_raises_value_error_naming_float32() -> None:
    X = np.arange(6, dtype=np.float64).reshape(3, 2)
    y = np.array([1.0, -1e39, 3.0])

    with pytest.raises(ValueError, match="float32"):
        _make_regressor()._check_X_y(X, y)


def test_y_float32_overflow_in_last_row_is_rejected() -> None:
    n_samples = 50
    X = np.arange(n_samples * 2, dtype=np.float64).reshape(n_samples, 2)
    y = np.ones(n_samples)
    y[-1] = 1e39

    with pytest.raises(ValueError, match="float32"):
        _make_regressor()._check_X_y(X, y)


def test_y_float32_overflow_in_multi_output_y_raises_value_error() -> None:
    X = np.arange(6, dtype=np.float64).reshape(3, 2)
    y = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 1e39]])

    with pytest.raises(ValueError, match="float32"):
        _make_regressor()._check_X_y(X, y)


def test_y_float32_overflow_rejected_with_max_samples_set() -> None:
    n_samples = 40
    X = np.arange(n_samples * 2, dtype=np.float64).reshape(n_samples, 2)
    y = np.ones(n_samples)
    y[-1] = 1e39
    reg = RandomSGForestRegressor(
        n_estimators=1, max_samples=0.1, tao_n_runs=TEST_TAO_N_RUNS, random_state=0
    )

    with pytest.raises(ValueError, match="float32"):
        reg._check_X_y(X, y)
