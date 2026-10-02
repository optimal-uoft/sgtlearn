import numpy as np
import pytest

from sgtlearn import SGTRegressor
from tests.constants import TEST_TAO_N_RUNS

FLOAT32_OVERFLOW = 1e39


def _data(n_samples: int = 60, n_features: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    y = X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.1, size=n_samples)
    return X, y


def _model(**kwargs) -> SGTRegressor:
    kwargs.setdefault("tao_n_runs", TEST_TAO_N_RUNS)
    kwargs.setdefault("max_depth", 2)
    return SGTRegressor(**kwargs)


def test_fit_allows_nan_in_X() -> None:
    X, y = _data()
    X[::7, 1] = np.nan

    model = _model().fit(X, y)

    assert model.n_features_in_ == X.shape[1]


@pytest.mark.parametrize("bad", [np.inf, -np.inf])
def test_fit_rejects_infinite_X(bad: float) -> None:
    X, y = _data()
    X[3, 0] = bad

    with pytest.raises(ValueError):
        _model().fit(X, y)


@pytest.mark.parametrize("bad", [FLOAT32_OVERFLOW, -FLOAT32_OVERFLOW])
def test_fit_rejects_X_value_overflowing_float32(bad: float) -> None:
    X, y = _data()
    X[3, 0] = bad

    with pytest.raises(ValueError):
        _model().fit(X, y)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_fit_rejects_non_finite_y(bad: float) -> None:
    X, y = _data()
    y[5] = bad

    with pytest.raises(ValueError):
        _model().fit(X, y)


@pytest.mark.parametrize("bad", [FLOAT32_OVERFLOW, -FLOAT32_OVERFLOW])
def test_fit_rejects_y_value_overflowing_float32(bad: float) -> None:
    X, y = _data()
    y[5] = bad

    with pytest.raises(ValueError):
        _model().fit(X, y)


def test_fit_rejects_non_finite_value_in_2d_y() -> None:
    X, y = _data()
    Y = np.column_stack([y, y])
    Y[2, 1] = np.inf

    with pytest.raises(ValueError):
        _model().fit(X, Y)


def test_fit_with_check_input_false_still_rejects_X_float32_overflow() -> None:
    X, y = _data()
    X[3, 0] = FLOAT32_OVERFLOW

    with pytest.raises(ValueError):
        _model().fit(X, y, check_input=False)


def test_fit_with_check_input_false_still_rejects_y_float32_overflow() -> None:
    X, y = _data()
    y[5] = FLOAT32_OVERFLOW

    with pytest.raises(ValueError):
        _model().fit(X, y, check_input=False)


def test_fit_rejects_sample_weight_with_wrong_length() -> None:
    X, y = _data()

    with pytest.raises(ValueError):
        _model().fit(X, y, sample_weight=np.ones(X.shape[0] - 1))


def test_fit_rejects_negative_sample_weight() -> None:
    X, y = _data()
    weights = np.ones(X.shape[0])
    weights[0] = -1.0

    with pytest.raises(ValueError):
        _model().fit(X, y, sample_weight=weights)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_fit_rejects_non_finite_sample_weight(bad: float) -> None:
    X, y = _data()
    weights = np.ones(X.shape[0])
    weights[0] = bad

    with pytest.raises(ValueError):
        _model().fit(X, y, sample_weight=weights)
