"""Fidelity contract: shape trees with NaN and ``inner_max_depth=1``.

Shape trees fit inner shape functions on finite values only, route training NaN
through the chosen partition, and fall back to the most-populated bin when a
feature had no training NaN at predict time.

With ``inner_max_depth=1`` each shape function is a single binary threshold split
(standard CART node). References are ``sklearn.tree.DecisionTreeClassifier`` /
``DecisionTreeRegressor`` with matching criteria and aligned hyperparameters.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

pytest.importorskip("sklearn")

from sgtlearn import SGTClassifier, SGTRegressor

from tests.constants import TEST_TAO_N_RUNS


def _inject_nan(
    X: np.ndarray,
    rng: np.random.Generator,
    *,
    n_cells: int | None = None,
    frac: float = 0.02,
) -> np.ndarray:
    """Return a copy of ``X`` with scattered NaN entries."""
    out = np.array(X, dtype=np.float32, copy=True)
    count = n_cells if n_cells is not None else max(1, int(frac * out.size))
    for _ in range(count):
        out[rng.integers(0, out.shape[0]), rng.integers(0, out.shape[1])] = np.nan
    return out


def _sklearn_classification_criterion(criterion: str) -> str:
    return "entropy" if criterion == "log_loss" else criterion


def _make_inner_depth_one_pair(
    task: str,
    criterion: str,
    *,
    min_samples_leaf: int = 1,
    min_impurity_decrease: float = 0.0,
):
    """SGT + sklearn estimators with aligned ``inner_max_depth=1`` defaults."""
    common = {
        "min_samples_leaf": min_samples_leaf,
        "min_impurity_decrease": min_impurity_decrease,
        "random_state": 0,
    }
    if task == "classification":
        sk_criterion = _sklearn_classification_criterion(criterion)
        return (
            SGTClassifier(
                criterion=criterion,
                inner_max_depth=1,
                tao_n_runs=TEST_TAO_N_RUNS,
                **common,
            ),
            DecisionTreeClassifier(criterion=sk_criterion, **common),
        )
    return (
        SGTRegressor(
            criterion=criterion,
            inner_max_depth=1,
            tao_n_runs=TEST_TAO_N_RUNS,
            **common,
        ),
        DecisionTreeRegressor(criterion=criterion, **common),
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


def test_sgt_classifier_inner_depth_one_matches_sklearn_breast_cancer_with_nan(
    rng: np.random.Generator,
) -> None:
    """Breast cancer + scattered NaN: in-sample labels match sklearn CART."""
    X, y = load_breast_cancer(return_X_y=True)
    X = _inject_nan(np.asarray(X, dtype=np.float32), rng, n_cells=18)

    sgt, dt = _make_inner_depth_one_pair("classification", "gini")
    sgt.fit(X, y)
    dt.fit(X, y)

    np.testing.assert_array_equal(sgt.classes_, dt.classes_)
    np.testing.assert_array_equal(sgt.predict(X), dt.predict(X))
    np.testing.assert_allclose(sgt.predict_proba(X), dt.predict_proba(X))


def test_sgt_classifier_inner_depth_one_handcrafted_multivariate_with_nan() -> None:
    """Small 2-feature example: NaN routes to the lower-impurity partition at the root."""
    X = np.array(
        [
            [1.0, 0.0],
            [2.0, 1.0],
            [np.nan, 0.0],
            [4.0, 1.0],
            [5.0, 0.0],
            [6.0, 1.0],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 0, 1, 1, 0, 1])

    sgt, dt = _make_inner_depth_one_pair("classification", "gini")
    sgt.fit(X, y)
    dt.fit(X, y)
    tree = sgt.tree_export()
    root = tree["nodes"][tree["root_index"]]
    assert root["nan_prediction_partition"] == 1
    np.testing.assert_array_equal(sgt.predict(X), dt.predict(X))


def test_sgt_regressor_handcrafted_nan_route_matches_sklearn() -> None:
    X = np.array(
        [
            [1.0, 0.0],
            [2.0, 1.0],
            [np.nan, 0.0],
            [4.0, 1.0],
            [5.0, 0.0],
            [6.0, 1.0],
        ],
        dtype=np.float32,
    )
    y = np.array([0.0, 0.0, 10.0, 10.0, 0.0, 10.0])

    sgt, dt = _make_inner_depth_one_pair("regression", "squared_error")
    sgt.fit(X, y)
    dt.fit(X, y)
    root = sgt.tree_export()["nodes"][0]

    assert root["nan_prediction_partition"] == 1
    np.testing.assert_allclose(sgt.predict(X), dt.predict(X))


def test_sgt_regressor_inner_depth_one_matches_sklearn_diabetes_with_nan(
    rng: np.random.Generator,
) -> None:
    bunch = load_diabetes()
    X = _inject_nan(np.asarray(bunch.data, dtype=np.float32), rng, n_cells=14)
    y = np.asarray(bunch.target, dtype=np.float64)

    sgt, dt = _make_inner_depth_one_pair("regression", "squared_error")
    sgt.fit(X, y)
    dt.fit(X, y)

    np.testing.assert_allclose(sgt.predict(X), dt.predict(X))


def test_sgt_classifier_inner_depth_one_predict_with_new_nan_matches_sklearn(
    rng: np.random.Generator,
) -> None:
    """NaN introduced only at predict time on clean training data."""
    X, y = load_breast_cancer(return_X_y=True)
    X = np.asarray(X, dtype=np.float32)

    sgt, dt = _make_inner_depth_one_pair("classification", "gini")
    sgt.fit(X, y)
    dt.fit(X, y)

    X_pred = X.copy()
    for idx in (0, 17, 42):
        X_pred[idx, rng.integers(0, X_pred.shape[1])] = np.nan

    np.testing.assert_array_equal(sgt.predict(X_pred), dt.predict(X_pred))


def test_sgt_regressor_inner_depth_one_predict_with_new_nan_matches_sklearn() -> None:
    """NaN introduced only at predict time: route like sklearn majority-bin fallback.

    ``absolute_error`` is omitted here: with clean training data, MAE trees can
    match sklearn in-sample yet disagree on predict-time NaN when the two fitted
    trees route samples through different feature paths.
    """
    X, y = load_diabetes(return_X_y=True)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float64)

    sgt, dt = _make_inner_depth_one_pair("regression", "squared_error")
    sgt.fit(X, y)
    dt.fit(X, y)

    X_pred = X.copy()
    X_pred[0, 0] = np.nan
    X_pred[17, 1] = np.nan
    X_pred[42, 2] = np.nan

    np.testing.assert_allclose(sgt.predict(X_pred), dt.predict(X_pred))


def test_sgt_classifier_still_rejects_inf_in_x() -> None:
    X, y = load_breast_cancer(return_X_y=True)
    X = np.asarray(X, dtype=np.float32)
    X[0, 0] = np.inf
    clf = SGTClassifier(inner_max_depth=1, random_state=42, tao_n_runs=TEST_TAO_N_RUNS)
    with pytest.raises(ValueError, match="infinity"):
        clf.fit(X, y)
