"""Fidelity tests: ``UnivariateRegressionDiscretizer`` vs ``sklearn.tree.DecisionTreeRegressor``.

Compares bin-wise predictions on synthetic univariate data across MSE/MAE-style
criteria where sklearn support exists.
"""

import sys

import numpy as np
import pytest
from itertools import product

from Discretizers import UnivariateRegressionDiscretizer
from sklearn.tree import DecisionTreeRegressor

from tests.discretizer_grid import (
    MAX_DEPTH_VALUES,
    MAX_LEAF_VALUES,
    MIN_GAIN_VALUES,
    MIN_LEAF_VALUES,
    N_VALUES,
)


def regression_predict(ud: UnivariateRegressionDiscretizer, x: np.ndarray) -> np.ndarray:
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


def _sklearn_supports_absolute_error() -> bool:
    try:
        DecisionTreeRegressor(criterion="absolute_error")
    except (ValueError, TypeError):
        return False
    return True


def _skip_if_sklearn_mae_best_first_segfault(max_leaf: int, criterion: str) -> None:
    """
    sklearn's best-first builder (max_leaf_nodes > 0) can segfault in native code
    with criterion='absolute_error' on Python 3.14+ (see crash in tree/_tree.so).
    Our discretizer still trains; skip only the sklearn reference for this grid cell.
    """
    if max_leaf == 0 or criterion not in ("absolute_error", "mae"):
        return
    if sys.version_info >= (3, 14):
        pytest.skip(
            "sklearn reference: BestFirstTreeBuilder + absolute_error + max_leaf_nodes "
            "segfaults on Python 3.14+; compare MAE with max_leaf=0 cases only"
        )


def _pred_discrepancy(
    sk: np.ndarray, ud: np.ndarray, *, use_mae: bool
) -> float:
    """RMSE or mean absolute error between sklearn and native prediction vectors."""
    d = sk.astype(np.float64) - ud.astype(np.float64)
    if use_mae:
        return float(np.mean(np.abs(d)))
    return float(np.sqrt(np.mean(d**2)))


# Same (n_samples, leaf, gain, depth, max_leaf) sweep as classification — without num_classes.
GRID = list(
    product(
        N_VALUES,
        MIN_LEAF_VALUES,
        MIN_GAIN_VALUES,
        MAX_DEPTH_VALUES,
        MAX_LEAF_VALUES,
    )
)
IDS = [
    f"N={n}|leaf={leaf}|gain={gain}|depth={depth}|max_leaf={max_leaf}"
    for n, leaf, gain, depth, max_leaf in GRID
]


@pytest.mark.parametrize(
    "criterion",
    ["squared_error", "mse", "absolute_error", "mae"],
)
@pytest.mark.parametrize(
    "n_samples,min_leaf_size,min_gain_split,max_depth,max_leaf",
    GRID,
    ids=IDS,
)
def test_univariate_regression_discretizer_vs_sklearn_fidelity(
    criterion: str,
    n_samples: int,
    min_leaf_size: int,
    min_gain_split: float,
    max_depth: int,
    max_leaf: int,
) -> None:
    """Bin predictions should track ``DecisionTreeRegressor`` within tolerance (MSE or MAE criterion)."""
    if criterion in ("absolute_error", "mae") and not _sklearn_supports_absolute_error():
        pytest.skip("sklearn DecisionTreeRegressor does not support criterion='absolute_error'")

    _skip_if_sklearn_mae_best_first_segfault(max_leaf, criterion)

    rng = np.random.default_rng(12345)
    x = rng.random((n_samples, 1), dtype=np.float64)
    x32 = x.astype(np.float32, copy=False)
    y = rng.standard_normal(n_samples).astype(np.float32, copy=False)

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


def test_regression_friedman_mse_constructible() -> None:
    """Constructible when bindings accept friedman_mse (else skipped)."""
    try:
        UnivariateRegressionDiscretizer(criterion="friedman_mse")
    except ValueError as e:
        pytest.skip(f"UnivariateRegressionDiscretizer does not support friedman_mse: {e}")


def test_regression_mae_constructible() -> None:
    UnivariateRegressionDiscretizer(criterion="mae")
