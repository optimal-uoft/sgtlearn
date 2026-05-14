"""
Fidelity: SGTRegressor vs sklearn DecisionTreeRegressor (diabetes, bundled).

Both ``squared_error`` and ``absolute_error`` are exercised via ``criterion``
parametrization (mirroring ``test_sgt_classifier_fidelity``). Training MSE is
compared to a depth-matched sklearn tree for ``squared_error`` only; MAE uses a
different objective than sklearn's MAE CART, so that branch only asserts finite
training loss. With ``inner_max_depth=1``, MSE predictions match sklearn; the
MAE branch checks finite predictions only.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor

from sgtlearn import SGTRegressor


def _load_xy():
    bunch = load_diabetes()
    X = np.asarray(bunch.data, dtype=np.float32)
    y = np.asarray(bunch.target, dtype=np.float64)
    return X, y


def _mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = y_true - y_pred
    return float(np.mean(d * d))


def _mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


@pytest.mark.parametrize("criterion", ["squared_error", "absolute_error"])
def test_sgt_train_error_at_most_sklearn_decision_tree(criterion: str) -> None:
    """Training MSE vs sklearn for ``squared_error``; finite loss smoke for MAE."""
    X, y = _load_xy()
    max_depth = 4
    min_samples_leaf = 8
    sgt = SGTRegressor(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_impurity_decrease=1e-7,
        inner_max_depth=8,
        inner_max_leaf_nodes=48,
        coordinate_descent_max_iters=25,
        coordinate_descent_patience=6,
        random_state=42,
    )
    sgt.fit(X, y)
    dt = DecisionTreeRegressor(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_impurity_decrease=1e-7,
        random_state=42,
    )
    dt.fit(X, y)
    y_s = sgt.predict(X)
    y_d = dt.predict(X)
    if criterion == "squared_error":
        sgt_e = _mean_squared_error(y, y_s)
        dt_e = _mean_squared_error(y, y_d)
        assert sgt_e <= dt_e, (
            f"Training MSE SGT ({sgt_e:.8f}) should be <= sklearn DecisionTree ({dt_e:.8f})"
        )
    else:
        sgt_e = _mean_absolute_error(y, y_s)
        dt_e = _mean_absolute_error(y, y_d)
        assert np.isfinite(sgt_e) and np.isfinite(dt_e), (
            "MAE criterion: training losses should be finite (SGT vs sklearn differ structurally)"
        )


@pytest.mark.parametrize("criterion", ["squared_error", "absolute_error"])
def test_sgt_matches_sklearn_decision_tree_inner_depth_one(criterion: str) -> None:
    """``inner_max_depth=1``: MSE predictions match sklearn; MAE checks finite fit."""
    X, y = _load_xy()
    sgt = SGTRegressor(criterion=criterion, inner_max_depth=1)
    sgt.fit(X, y)
    dt = DecisionTreeRegressor(criterion=criterion, random_state=0)
    dt.fit(X, y)
    pred = sgt.predict(X)
    assert pred.shape == (X.shape[0],)
    assert np.all(np.isfinite(pred))

    np.testing.assert_allclose(pred, dt.predict(X), rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("criterion", ["squared_error", "absolute_error"])
def test_sgt_leaf_regression_stats_for_criterion(criterion: str) -> None:
    """Squared error: ``[sum y, sum y^2]`` on leaves; MAE: empty vectors per leaf."""
    X, y = _load_xy()
    sgt = SGTRegressor(criterion=criterion, max_depth=2, random_state=0)
    sgt.fit(X, y)
    stats = sgt._est.leaf_regression_stats
    counts = sgt._est.leaf_num_samples
    assert len(stats) > 0
    assert len(counts) == len(stats)
    assert all(c > 0 for c in counts)
    if criterion == "squared_error":
        assert any(len(s) == 2 for s in stats)
        for s in stats:
            if len(s) == 2:
                assert np.isfinite(s[0]) and np.isfinite(s[1])
    else:
        assert all(len(row) == 0 for row in stats), (
            "MAE leaves should expose empty statistic vectors (no per-leaf float stats)"
        )
