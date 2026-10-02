import numpy as np
import pytest

from sgtlearn import SGTRegressor
from tests.constants import TEST_TAO_N_RUNS


def _data(n_samples: int = 60, n_features: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    y = X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.1, size=n_samples)
    return X, y


def _model(**kwargs) -> SGTRegressor:
    kwargs.setdefault("tao_n_runs", TEST_TAO_N_RUNS)
    kwargs.setdefault("max_depth", 2)
    return SGTRegressor(**kwargs)


@pytest.mark.parametrize(
    "param", ["min_impurity_decrease", "pairwise_penalty", "branching_penalty"]
)
@pytest.mark.parametrize("bad", [-0.1, np.inf, np.nan])
def test_fit_rejects_negative_or_non_finite_penalty_hyperparameters(
    param: str, bad: float
) -> None:
    X, y = _data()

    with pytest.raises(ValueError):
        _model(**{param: bad}).fit(X, y)


@pytest.mark.parametrize(
    "param", ["min_impurity_decrease", "pairwise_penalty", "branching_penalty"]
)
def test_fit_accepts_positive_finite_penalty_hyperparameters(param: str) -> None:
    X, y = _data()

    model = _model(**{param: 0.01}).fit(X, y)

    assert model.n_features_in_ == X.shape[1]


def test_fit_rejects_invalid_pairwise_candidates() -> None:
    X, y = _data()

    with pytest.raises(ValueError):
        _model(pairwise_candidates=-1).fit(X, y)


def test_fit_rejects_invalid_tao_pair_scale() -> None:
    X, y = _data()

    with pytest.raises(ValueError):
        _model(tao_pair_scale=np.nan).fit(X, y)


@pytest.mark.parametrize("bad", [-1, True, False, 1.0, 2.5, None])
def test_fit_rejects_invalid_tao_n_runs(bad) -> None:
    X, y = _data()

    with pytest.raises(ValueError):
        _model(tao_n_runs=bad).fit(X, y)


def test_fit_accepts_zero_tao_n_runs() -> None:
    X, y = _data()

    model = _model(tao_n_runs=0).fit(X, y)

    assert model._tao_refined_ is False
