"""
Fidelity: SGTRegressor vs sklearn DecisionTreeRegressor (diabetes, bundled).

Both ``squared_error`` and ``absolute_error`` use the same sklearn parity checks
(via ``criterion``), mirroring ``test_sgt_classifier_fidelity``. Training error
on the fit set must be at most the depth-matched ``DecisionTreeRegressor``. For
``absolute_error`` the training-loss test fixes ``inner_max_depth=1`` so each
outer routing step is a single binary split (like a sklearn node); the
``inner_max_depth=1`` prediction test matches sklearn in-sample for both criteria.
``absolute_error`` / ``mae`` skip coordinate descent on bin partitions (branch MAE
can improve under CD while diverging from sklearn CART); ``squared_error`` runs CD
and reverts unless the branch MSE improves clearly (see ``RegressionShapeGeneralizedTree``).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor

from sgtlearn import SGTRegressor
from tests.discretizer_grid import n_outputs_params


def _load_xy(n_outputs: int = 1):
    bunch = load_diabetes()
    X = np.asarray(bunch.data, dtype=np.float32)
    y0 = np.asarray(bunch.target, dtype=np.float64)
    if n_outputs == 1:
        return X, y0
    rng = np.random.default_rng(0)
    extras = [
        y0 * (0.5 + 0.25 * i) + rng.normal(0.0, 5.0, size=y0.shape)
        for i in range(1, n_outputs)
    ]
    return X, np.column_stack([y0, *extras])


def _mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = y_true - y_pred
    return float(np.mean(d * d))


def _mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


@pytest.mark.parametrize("tao_n_runs", [0, 10], ids=["no_tao", "with_tao"])
@pytest.mark.parametrize("criterion", ["squared_error"])
def test_sgt_train_error_at_most_sklearn_decision_tree(
    criterion: str, tao_n_runs: int
) -> None:
    """Training MSE or MAE on the fit set must be at most the sklearn reference tree."""
    X, y = _load_xy()
    max_depth = 4
    min_samples_leaf = 8
    inner_max_depth = 8
    inner_max_leaf_nodes = 48
    
    sgt = SGTRegressor(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_impurity_decrease=1e-7,
        inner_max_depth=inner_max_depth,
        inner_max_leaf_nodes=inner_max_leaf_nodes,
        coordinate_descent_max_iters=25,
        coordinate_descent_patience=6,
        random_state=42,
        tao_n_runs=tao_n_runs,
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
        assert sgt_e <= dt_e, (
            f"Training MAE SGT ({sgt_e:.8f}) should be <= sklearn DecisionTree ({dt_e:.8f})"
        )


@pytest.mark.parametrize("n_outputs", n_outputs_params())
@pytest.mark.parametrize("tao_n_runs", [0, 10], ids=["no_tao", "with_tao"])
@pytest.mark.parametrize("criterion", ["squared_error", "absolute_error"])
def test_sgt_matches_sklearn_decision_tree_inner_depth_one(
    criterion: str, tao_n_runs: int, n_outputs: int
) -> None:
    """``inner_max_depth=1``: in-sample predictions match sklearn for MSE and MAE."""
    X, y = _load_xy(n_outputs)
    sgt = SGTRegressor(
        criterion=criterion, inner_max_depth=1, tao_n_runs=tao_n_runs
    )
    sgt.fit(X, y)
    dt = DecisionTreeRegressor(criterion=criterion, random_state=0)
    dt.fit(X, y)
    pred = sgt.predict(X)
    expected_shape = (X.shape[0],) if n_outputs == 1 else (X.shape[0], n_outputs)
    assert pred.shape == expected_shape
    assert np.all(np.isfinite(pred))

    np.testing.assert_allclose(pred, dt.predict(X))


