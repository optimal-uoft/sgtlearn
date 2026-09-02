"""Fidelity tests: ``UnivariateRegressionDiscretizer`` vs ``sklearn.tree.DecisionTreeRegressor``.

Compares bin-wise predictions on synthetic univariate data across MSE/MAE-style
criteria where sklearn support exists.
"""

import sys

import numpy as np
import pytest

from Discretizers import UnivariateRegressionDiscretizer
from sklearn.tree import DecisionTreeRegressor


def regression_predict(
    ud: UnivariateRegressionDiscretizer, x: np.ndarray
) -> np.ndarray:
    """Predict by mapping transform() bin indices to bin predictions."""
    bin_locs = ud.transform(x)
    bin_preds = ud.getBinPredictions()
    return np.asarray(bin_preds[bin_locs], dtype=np.float32)


def sklearn_regression_criterion(user_criterion: str) -> str:
    """Map our aliases to sklearn 1.0+ names (squared_error, absolute_error)."""
    if user_criterion == "mse":
        return "squared_error"
    if user_criterion == "mae":
        return "absolute_error"
    return user_criterion


def _pred_discrepancy(sk: np.ndarray, ud: np.ndarray, *, use_mae: bool) -> float:
    """RMSE or mean absolute error between sklearn and native prediction vectors."""
    d = sk.astype(np.float64) - ud.astype(np.float64)
    if use_mae:
        return float(np.mean(np.abs(d)))
    return float(np.sqrt(np.mean(d**2)))


PARITY_CASES = [
    pytest.param("squared_error", 1, 0.0, 0, 0, 1, id="mse-unconstrained"),
    pytest.param("absolute_error", 1, 0.0, 0, 0, 2, id="mae-multioutput"),
    pytest.param("squared_error", 1, 0.0, 4, 0, 2, id="depth-limited"),
    pytest.param("squared_error", 1, 0.0, 0, 8, 1, id="leaf-limited"),
]


@pytest.mark.parametrize(
    "criterion,min_leaf_size,min_gain_split,max_depth,max_leaf,n_outputs",
    PARITY_CASES,
)
def test_univariate_regression_discretizer_vs_sklearn_fidelity(
    criterion: str,
    min_leaf_size: int,
    min_gain_split: float,
    max_depth: int,
    max_leaf: int,
    n_outputs: int,
) -> None:
    """Bin predictions should track ``DecisionTreeRegressor`` within tolerance (MSE or MAE criterion)."""
    rng = np.random.default_rng(12345)
    n_samples = 1000
    x = rng.random((n_samples, 1), dtype=np.float64)
    x32 = x.astype(np.float32, copy=False)
    y_shape = (n_samples,) if n_outputs == 1 else (n_samples, n_outputs)
    y = rng.standard_normal(y_shape).astype(np.float32, copy=False)

    sk_crit = sklearn_regression_criterion(criterion)
    reg = DecisionTreeRegressor(
        criterion=sk_crit,
        splitter="best",
        min_samples_leaf=min_leaf_size,
        min_impurity_decrease=min_gain_split,
        max_depth=None if max_depth == 0 else max_depth,
        random_state=0,
        max_leaf_nodes=None if max_leaf == 0 else max_leaf,
    )
    reg.fit(x32, y)
    sklearn_preds = reg.predict(x32).astype(np.float32, copy=False)

    ud = UnivariateRegressionDiscretizer(criterion=criterion)
    features = np.array([0], dtype=np.uintp)
    ud.Train(
        x32,
        features,
        y,
        min_leaf_size,
        min_gain_split,
        max_depth,
        max_leaf,
    )

    ud_preds = regression_predict(ud, x32)
    assert sklearn_preds.shape == ud_preds.shape
    assert np.all(np.isfinite(ud_preds))
    assert ud.numLeaves > 0

    # Classification grid matches exactly; continuous regression gains do not guarantee
    # per-sample equality with sklearn. Bound discrepancy vs label spread (RMSE for
    # squared-error/MSE; mean absolute pred gap for MAE / absolute_error).
    sigma = float(np.std(y)) + 1e-8
    use_mae_metric = criterion in ("absolute_error", "mae")
    disc = _pred_discrepancy(sklearn_preds, ud_preds, use_mae=use_mae_metric)
    assert disc < 0.9 * sigma


@pytest.mark.parametrize("criterion", ["squared_error", "absolute_error"])
def test_univariate_regression_leaf_limit_binds_without_reference(
    criterion: str,
) -> None:
    rng = np.random.default_rng(9)
    x = rng.random((200, 1), dtype=np.float32)
    y = rng.standard_normal(200).astype(np.float32)
    ud = UnivariateRegressionDiscretizer(criterion=criterion)
    ud.Train(x, np.array([0], dtype=np.uintp), y, 1, 0.0, 0, 4)
    bins = ud.transform(x)
    assert ud.numLeaves == 4
    assert np.all(bins < ud.numLeaves)
    assert np.all(np.isfinite(regression_predict(ud, x)))


@pytest.mark.parametrize("criterion", ["squared_error", "absolute_error"])
def test_univariate_regression_respects_minimum_leaf_size(criterion: str) -> None:
    x = np.arange(12, dtype=np.float32).reshape(-1, 1)
    y = np.repeat(np.array([0.0, 10.0], dtype=np.float32), 6)
    features = np.array([0], dtype=np.uintp)
    allowed = UnivariateRegressionDiscretizer(criterion=criterion)
    allowed.Train(x, features, y, 6, 0.0, 0, 0)
    blocked = UnivariateRegressionDiscretizer(criterion=criterion)
    blocked.Train(x, features, y, 7, 0.0, 0, 0)

    assert allowed.numLeaves == 2
    assert blocked.numLeaves == 1


def test_univariate_regression_gain_threshold_blocks_known_split() -> None:
    x = np.arange(12, dtype=np.float32).reshape(-1, 1)
    y = np.repeat(np.array([0.0, 10.0], dtype=np.float32), 6)
    features = np.array([0], dtype=np.uintp)
    split = UnivariateRegressionDiscretizer(criterion="squared_error")
    split.Train(x, features, y, 1, 0.0, 0, 0)
    blocked = UnivariateRegressionDiscretizer(criterion="squared_error")
    blocked.Train(x, features, y, 1, 1_000.0, 0, 0)

    assert split.numLeaves == 2
    assert blocked.numLeaves == 1


@pytest.mark.skipif(
    sys.version_info >= (3, 14),
    reason="sklearn absolute_error best-first builder can crash on Python 3.14+",
)
def test_univariate_mae_leaf_limit_matches_sklearn_when_reference_is_safe() -> None:
    rng = np.random.default_rng(11)
    x = rng.random((200, 1), dtype=np.float32)
    y = rng.standard_normal(200).astype(np.float32)
    sk = DecisionTreeRegressor(criterion="absolute_error", max_leaf_nodes=4).fit(x, y)
    ud = UnivariateRegressionDiscretizer(criterion="absolute_error")
    ud.Train(x, np.array([0], dtype=np.uintp), y, 1, 0.0, 0, 4)
    assert sk.get_n_leaves() == ud.numLeaves
    assert _pred_discrepancy(
        sk.predict(x), regression_predict(ud, x), use_mae=True
    ) < np.std(y)


@pytest.mark.parametrize(
    "alias,canonical", [("mse", "squared_error"), ("mae", "absolute_error")]
)
def test_regression_criterion_aliases_equivalent(alias: str, canonical: str) -> None:
    """``mse``/``mae`` are aliases: identical bin predictions to their canonical name."""
    rng = np.random.default_rng(12345)
    x32 = rng.random((2000, 1), dtype=np.float64).astype(np.float32)
    y = rng.standard_normal(2000).astype(np.float32)
    features = np.array([0], dtype=np.uintp)

    preds = {}
    for crit in (alias, canonical):
        ud = UnivariateRegressionDiscretizer(criterion=crit)
        ud.Train(x32, features, y, 1, 0.0, 0, 0)
        preds[crit] = regression_predict(ud, x32)
    np.testing.assert_array_equal(preds[alias], preds[canonical])
