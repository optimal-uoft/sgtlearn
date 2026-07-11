"""Fidelity tests: ``CategoricalRegressionDiscretizer`` vs ``sklearn.tree.DecisionTreeRegressor``.

Compares bin-wise predictions on one-hot features across MSE/MAE-style criteria
where sklearn support exists.
"""

from __future__ import annotations

import sys
from itertools import product

import numpy as np
import pytest
from Discretizers import CategoricalRegressionDiscretizer
from sklearn.tree import DecisionTreeRegressor

from tests.discretizer_grid import (
    MAX_DEPTH_VALUES,
    MAX_LEAF_VALUES,
    MIN_GAIN_VALUES,
    MIN_LEAF_VALUES,
    N_VALUES,
    NUM_CATEGORIES_VALUES,
    n_outputs_params,
)


def _make_onehot(
    n_samples: int,
    n_categories: int,
    rng: np.random.Generator,
    n_outputs: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    cats = rng.integers(0, n_categories, size=n_samples)
    x = np.zeros((n_samples, n_categories), dtype=np.float32)
    x[np.arange(n_samples), cats] = 1.0
    y_shape = (n_samples,) if n_outputs == 1 else (n_samples, n_outputs)
    y = rng.standard_normal(y_shape).astype(np.float32)
    return x, y


def regression_predict(
    disc: CategoricalRegressionDiscretizer, x: np.ndarray
) -> np.ndarray:
    bin_locs = disc.transform(x)
    bin_preds = disc.getBinPredictions()
    return np.asarray(bin_preds[bin_locs], dtype=np.float32)


def sklearn_regression_criterion(user_criterion: str) -> str:
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
    if max_leaf == 0 or criterion not in ("absolute_error", "mae"):
        return
    if sys.version_info >= (3, 14):
        pytest.skip(
            "sklearn reference: BestFirstTreeBuilder + absolute_error + max_leaf_nodes "
            "segfaults on Python 3.14+; compare MAE with max_leaf=0 cases only"
        )


def _pred_discrepancy(sk: np.ndarray, ud: np.ndarray, *, use_mae: bool) -> float:
    """RMSE or mean absolute error between sklearn and native prediction vectors."""
    d = sk.astype(np.float64) - ud.astype(np.float64)
    if use_mae:
        return float(np.mean(np.abs(d)))
    return float(np.sqrt(np.mean(d**2)))


GRID = list(
    product(
        N_VALUES,
        NUM_CATEGORIES_VALUES,
        MIN_LEAF_VALUES,
        MIN_GAIN_VALUES,
        MAX_DEPTH_VALUES,
        MAX_LEAF_VALUES,
    )
)
IDS = [
    f"N={n}|cat={c}|leaf={leaf}|gain={gain}|depth={depth}|max_leaf={max_leaf}"
    for n, c, leaf, gain, depth, max_leaf in GRID
]


@pytest.mark.parametrize("n_outputs", n_outputs_params())
@pytest.mark.parametrize("criterion", ["squared_error", "absolute_error"])
@pytest.mark.parametrize(
    "n_samples,n_categories,min_leaf_size,min_gain_split,max_depth,max_leaf",
    GRID,
    ids=IDS,
)
def test_categorical_onehot_regression_discretizer_vs_sklearn_fidelity(
    criterion: str,
    n_samples: int,
    n_categories: int,
    min_leaf_size: int,
    min_gain_split: float,
    max_depth: int,
    max_leaf: int,
    n_outputs: int,
) -> None:
    """Bin predictions should track ``DecisionTreeRegressor`` within tolerance."""
    if criterion in ("absolute_error", "mae") and not _sklearn_supports_absolute_error():
        pytest.skip("sklearn DecisionTreeRegressor does not support criterion='absolute_error'")

    _skip_if_sklearn_mae_best_first_segfault(max_leaf, criterion)

    rng = np.random.default_rng(12345)
    x, y = _make_onehot(n_samples, n_categories, rng, n_outputs=n_outputs)

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
    reg.fit(x, y)
    sklearn_preds = reg.predict(x).astype(np.float32, copy=False)

    features = np.arange(n_categories, dtype=np.uintp)
    disc = CategoricalRegressionDiscretizer(criterion=criterion)
    disc.Train(
        x,
        features,
        y,
        min_leaf_size,
        min_gain_split,
        max_depth,
        max_leaf,
    )

    disc_preds = regression_predict(disc, x)
    assert sklearn_preds.shape == disc_preds.shape
    assert np.all(np.isfinite(disc_preds))
    assert disc.numLeaves > 0

    sigma = float(np.std(y)) + 1e-8
    use_mae_metric = criterion in ("absolute_error", "mae")
    discrepancy = _pred_discrepancy(sklearn_preds, disc_preds, use_mae=use_mae_metric)
    assert discrepancy < 0.9 * sigma


@pytest.mark.parametrize("alias,canonical", [("mse", "squared_error"), ("mae", "absolute_error")])
def test_categorical_onehot_regression_criterion_aliases_equivalent(
    alias: str, canonical: str
) -> None:
    """``mse``/``mae`` aliases yield identical bin predictions to canonical names."""
    rng = np.random.default_rng(12345)
    n_cat = 4
    x, y = _make_onehot(2000, n_cat, rng)
    features = np.arange(n_cat, dtype=np.uintp)

    preds = {}
    for crit in (alias, canonical):
        disc = CategoricalRegressionDiscretizer(criterion=crit)
        disc.Train(x, features, y, 1, 0.0, 0, 0)
        preds[crit] = regression_predict(disc, x)
    np.testing.assert_array_equal(preds[alias], preds[canonical])
