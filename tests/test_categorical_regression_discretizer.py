"""Fidelity tests: ``CategoricalRegressionDiscretizer`` vs ``sklearn.tree.DecisionTreeRegressor``.

Compares bin-wise predictions on one-hot features across MSE/MAE-style criteria
where sklearn support exists.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from Discretizers import CategoricalRegressionDiscretizer
from sklearn.tree import DecisionTreeRegressor


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


def _pred_discrepancy(sk: np.ndarray, ud: np.ndarray, *, use_mae: bool) -> float:
    """RMSE or mean absolute error between sklearn and native prediction vectors."""
    d = sk.astype(np.float64) - ud.astype(np.float64)
    if use_mae:
        return float(np.mean(np.abs(d)))
    return float(np.sqrt(np.mean(d**2)))


PARITY_CASES = [
    pytest.param("squared_error", 2, 1, 0.0, 0, 0, 1, id="mse-binary"),
    pytest.param("absolute_error", 3, 1, 0.0, 0, 0, 2, id="mae-multioutput"),
    pytest.param("squared_error", 4, 1, 0.0, 2, 0, 2, id="depth-limited"),
    pytest.param("squared_error", 4, 1, 0.0, 0, 2, 1, id="leaf-limited"),
]


@pytest.mark.parametrize(
    "criterion,n_categories,min_leaf_size,min_gain_split,max_depth,max_leaf,n_outputs",
    PARITY_CASES,
)
def test_categorical_onehot_regression_discretizer_vs_sklearn_fidelity(
    criterion: str,
    n_categories: int,
    min_leaf_size: int,
    min_gain_split: float,
    max_depth: int,
    max_leaf: int,
    n_outputs: int,
) -> None:
    """Bin predictions should track ``DecisionTreeRegressor`` within tolerance."""
    rng = np.random.default_rng(12345)
    x, y = _make_onehot(1000, n_categories, rng, n_outputs=n_outputs)

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


@pytest.mark.parametrize("criterion", ["squared_error", "absolute_error"])
def test_categorical_regression_leaf_limit_binds_without_reference(
    criterion: str,
) -> None:
    rng = np.random.default_rng(8)
    x, y = _make_onehot(300, 5, rng)
    disc = CategoricalRegressionDiscretizer(criterion=criterion)
    disc.Train(x, np.arange(5, dtype=np.uintp), y, 1, 0.0, 0, 3)
    bins = disc.transform(x)
    assert disc.numLeaves == 3
    assert np.all(bins < disc.numLeaves)
    assert np.all(np.isfinite(regression_predict(disc, x)))


@pytest.mark.parametrize("criterion", ["squared_error", "absolute_error"])
def test_categorical_regression_respects_minimum_leaf_size(criterion: str) -> None:
    x = np.repeat(np.eye(3, dtype=np.float32), 4, axis=0)
    y = np.repeat(np.array([0.0, 10.0, 20.0], dtype=np.float32), 4)
    features = np.arange(3, dtype=np.uintp)
    allowed = CategoricalRegressionDiscretizer(criterion=criterion)
    allowed.Train(x, features, y, 4, 0.0, 0, 0)
    blocked = CategoricalRegressionDiscretizer(criterion=criterion)
    blocked.Train(x, features, y, 5, 0.0, 0, 0)

    assert allowed.numLeaves == 3
    assert blocked.numLeaves == 1


def test_categorical_regression_gain_threshold_blocks_known_split() -> None:
    x = np.repeat(np.eye(3, dtype=np.float32), 4, axis=0)
    y = np.repeat(np.array([0.0, 10.0, 20.0], dtype=np.float32), 4)
    features = np.arange(3, dtype=np.uintp)
    split = CategoricalRegressionDiscretizer(criterion="squared_error")
    split.Train(x, features, y, 1, 0.0, 0, 0)
    blocked = CategoricalRegressionDiscretizer(criterion="squared_error")
    blocked.Train(x, features, y, 1, 1_000.0, 0, 0)

    assert split.numLeaves == 3
    assert blocked.numLeaves == 0


@pytest.mark.skipif(
    sys.version_info >= (3, 14),
    reason="sklearn absolute_error best-first builder can crash on Python 3.14+",
)
def test_categorical_mae_leaf_limit_matches_sklearn_when_reference_is_safe() -> None:
    rng = np.random.default_rng(13)
    x, y = _make_onehot(300, 5, rng)
    sk = DecisionTreeRegressor(criterion="absolute_error", max_leaf_nodes=3).fit(x, y)
    disc = CategoricalRegressionDiscretizer(criterion="absolute_error")
    disc.Train(x, np.arange(5, dtype=np.uintp), y, 1, 0.0, 0, 3)
    assert sk.get_n_leaves() == disc.numLeaves
    assert _pred_discrepancy(
        sk.predict(x), regression_predict(disc, x), use_mae=True
    ) < np.std(y)


@pytest.mark.parametrize(
    "alias,canonical", [("mse", "squared_error"), ("mae", "absolute_error")]
)
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
