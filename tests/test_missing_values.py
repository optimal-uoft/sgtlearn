"""Fidelity contract: shape trees with NaN and ``inner_max_depth=1``.

These tests define the sklearn parity bar once full missing-value handling is in
place (fit shape functions on finite values only, route training NaN through the
chosen partition, track ``sawMissingInTraining``, fall back to the most-populated
bin when a feature had no training NaN).

With ``inner_max_depth=1`` each shape function is a single binary threshold split
(standard CART node). References are ``sklearn.tree.DecisionTreeClassifier`` /
``DecisionTreeRegressor`` with matching criterion and default hyperparameters.

Tests may fail or be marked ``xfail`` until that routing work lands; they are
written for robustness, not for the current partial implementation.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

pytest.importorskip("sklearn")

from sgtlearn import SGTClassifier, SGTRegressor


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


# (n_samples, n_features) stress shapes: thousands of rows, up to 10 features.
_LARGE_SCALE_SHAPES: list[tuple[int, int]] = [
    (1000, 3),
    (1000, 10),
    (2000, 5),
    (3000, 10),
]

# Multivariate shapes where ``inner_max_depth=1`` NaN routing matches sklearn in-sample.
_SKLEARN_PARITY_LARGE_SHAPES: list[tuple[int, int]] = [
    (1000, 3),
    (1000, 10),
    (2000, 5),
    (3000, 10),
]


def _large_scale_rng(n_samples: int, n_features: int, salt: str) -> np.random.Generator:
    seed = hash((n_samples, n_features, salt)) % (2**32)
    return np.random.default_rng(seed)


def _make_large_classification_xy(
    n_samples: int,
    n_features: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    y = rng.integers(0, 2, size=n_samples)
    return X, y


def _make_large_regression_xy(
    n_samples: int,
    n_features: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    coef = rng.standard_normal(n_features)
    y = X @ coef + 0.25 * rng.standard_normal(n_samples)
    return X, y.astype(np.float64)


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
    }
    if task == "classification":
        sk_criterion = _sklearn_classification_criterion(criterion)
        return (
            SGTClassifier(criterion=criterion, inner_max_depth=1, **common),
            DecisionTreeClassifier(criterion=sk_criterion, **common),
        )
    return (
        SGTRegressor(criterion=criterion, inner_max_depth=1, **common),
        DecisionTreeRegressor(criterion=criterion, **common),
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_sgt_classifier_inner_depth_one_matches_sklearn_breast_cancer_with_nan(
    criterion: str,
    rng: np.random.Generator,
) -> None:
    """Breast cancer + scattered NaN: in-sample labels match sklearn CART."""
    X, y = load_breast_cancer(return_X_y=True)
    X = _inject_nan(np.asarray(X, dtype=np.float32), rng, n_cells=18)

    sgt, dt = _make_inner_depth_one_pair("classification", criterion)
    sgt.fit(X, y)
    dt.fit(X, y)

    np.testing.assert_array_equal(sgt.classes_, dt.classes_)
    np.testing.assert_array_equal(sgt.predict(X), dt.predict(X))


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_sgt_classifier_predict_proba_matches_sklearn_with_nan(
    criterion: str,
    rng: np.random.Generator,
) -> None:
    X, y = load_breast_cancer(return_X_y=True)
    X = _inject_nan(np.asarray(X, dtype=np.float32), rng, n_cells=12)

    sgt, dt = _make_inner_depth_one_pair("classification", criterion)
    sgt.fit(X, y)
    dt.fit(X, y)

    np.testing.assert_allclose(sgt.predict_proba(X), dt.predict_proba(X))


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_sgt_classifier_inner_depth_one_matches_sklearn_synthetic_with_nan(
    criterion: str,
    rng: np.random.Generator,
) -> None:
    n_samples, n_features = 80, 5
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    X = _inject_nan(X, rng, n_cells=10)
    y = rng.integers(0, 2, size=n_samples)

    sgt, dt = _make_inner_depth_one_pair("classification", criterion)
    sgt.fit(X, y)
    dt.fit(X, y)

    np.testing.assert_array_equal(sgt.predict(X), dt.predict(X))


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

    sgt = SGTClassifier(
        criterion="gini",
        inner_max_depth=1,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
    )
    sgt.fit(X, y)
    tree = sgt.tree_export()
    root = tree["nodes"][tree["root_index"]]
    assert root["nan_prediction_partition"] == 1
    assert sgt.predict(X)[2] == 0


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


def test_sgt_regressor_inner_depth_one_matches_sklearn_synthetic_with_nan(
    rng: np.random.Generator,
) -> None:
    n_samples, n_features = 80, 4
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    X = _inject_nan(X, rng, n_cells=8)
    y = rng.standard_normal(n_samples)

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
    """NaN introduced only at predict time: route like sklearn majority-bin fallback."""
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


def test_sgt_regressor_absolute_error_fit_predict_with_nan(
    rng: np.random.Generator,
) -> None:
    """MAE shape tree should accept NaN once routing is complete."""
    X = rng.standard_normal((60, 4)).astype(np.float32)
    X = _inject_nan(X, rng, n_cells=6)
    y = rng.standard_normal(60)

    reg = SGTRegressor(criterion="absolute_error", inner_max_depth=1)
    reg.fit(X, y)
    preds = reg.predict(X)
    assert preds.shape == (60,)
    assert np.all(np.isfinite(preds))

    with pytest.raises(ValueError, match="NaN"):
        DecisionTreeRegressor(criterion="absolute_error").fit(X, y)


def test_sgt_classifier_still_rejects_inf_in_x() -> None:
    X, y = load_breast_cancer(return_X_y=True)
    X = np.asarray(X, dtype=np.float32)
    X[0, 0] = np.inf
    clf = SGTClassifier(inner_max_depth=1, random_state=42)
    with pytest.raises(ValueError, match="infinity"):
        clf.fit(X, y)


@pytest.mark.parametrize("n_samples,n_features", _SKLEARN_PARITY_LARGE_SHAPES)
@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_sgt_classifier_large_scale_nan_matches_sklearn(
    n_samples: int,
    n_features: int,
    criterion: str,
) -> None:
    """Thousands of rows / up to 10 features: in-sample labels match sklearn CART."""
    rng = _large_scale_rng(n_samples, n_features, criterion)
    X, y = _make_large_classification_xy(n_samples, n_features, rng)
    X = _inject_nan(X, rng, frac=0.02)

    sgt, dt = _make_inner_depth_one_pair("classification", criterion)
    sgt.fit(X, y)
    dt.fit(X, y)

    np.testing.assert_array_equal(sgt.classes_, dt.classes_)
    np.testing.assert_array_equal(sgt.predict(X), dt.predict(X))


@pytest.mark.parametrize("n_samples,n_features", _LARGE_SCALE_SHAPES)
@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_sgt_classifier_large_scale_nan_predict_proba_finite(
    n_samples: int,
    n_features: int,
    criterion: str,
) -> None:
    """Large-scale proba outputs are well-formed (routing may differ from sklearn)."""
    rng = _large_scale_rng(n_samples, n_features, f"proba_{criterion}")
    X, y = _make_large_classification_xy(n_samples, n_features, rng)
    X = _inject_nan(X, rng, frac=0.02)

    sgt, _ = _make_inner_depth_one_pair("classification", criterion)
    sgt.fit(X, y)
    proba = sgt.predict_proba(X)

    assert proba.shape == (n_samples, 2)
    assert np.all(np.isfinite(proba))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=0.0, atol=1e-6)
    assert np.all(proba >= 0.0)


@pytest.mark.parametrize("n_samples,n_features", _LARGE_SCALE_SHAPES)
@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_sgt_classifier_large_scale_nan_smoke(
    n_samples: int,
    n_features: int,
    criterion: str,
) -> None:
    """All large shapes: fit/predict with training NaN completes with valid outputs."""
    rng = _large_scale_rng(n_samples, n_features, f"smoke_{criterion}")
    X, y = _make_large_classification_xy(n_samples, n_features, rng)
    X = _inject_nan(X, rng, frac=0.02)

    sgt, _ = _make_inner_depth_one_pair("classification", criterion)
    sgt.fit(X, y)
    pred = sgt.predict(X)
    proba = sgt.predict_proba(X)

    assert pred.shape == (n_samples,)
    assert proba.shape == (n_samples, 2)
    assert np.all(np.isfinite(pred))
    assert np.all(np.isfinite(proba))


@pytest.mark.parametrize("n_samples,n_features", _LARGE_SCALE_SHAPES)
def test_sgt_classifier_large_scale_predict_with_new_nan_smoke(
    n_samples: int,
    n_features: int,
) -> None:
    """Large clean fit; NaN at predict time returns finite labels."""
    rng = _large_scale_rng(n_samples, n_features, "predict_new_nan")
    X, y = _make_large_classification_xy(n_samples, n_features, rng)

    sgt, _ = _make_inner_depth_one_pair("classification", "gini")
    sgt.fit(X, y)

    X_pred = X.copy()
    n_pred_nan = max(20, n_samples // 50)
    for _ in range(n_pred_nan):
        X_pred[rng.integers(0, n_samples), rng.integers(0, n_features)] = np.nan

    pred = sgt.predict(X_pred)
    assert pred.shape == (n_samples,)
    assert np.all(np.isfinite(pred))


@pytest.mark.parametrize("n_samples,n_features", _SKLEARN_PARITY_LARGE_SHAPES)
def test_sgt_regressor_squared_error_large_scale_nan_matches_sklearn(
    n_samples: int,
    n_features: int,
) -> None:
    """Large synthetic regression with scattered NaN matches sklearn in-sample."""
    rng = _large_scale_rng(n_samples, n_features, "squared_error")
    X, y = _make_large_regression_xy(n_samples, n_features, rng)
    X = _inject_nan(X, rng, frac=0.02)

    sgt, dt = _make_inner_depth_one_pair("regression", "squared_error")
    sgt.fit(X, y)
    dt.fit(X, y)

    np.testing.assert_allclose(sgt.predict(X), dt.predict(X), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("n_samples,n_features", _LARGE_SCALE_SHAPES)
def test_sgt_regressor_squared_error_large_scale_nan_smoke(
    n_samples: int,
    n_features: int,
) -> None:
    """All large shapes: MSE fit/predict with training NaN completes."""
    rng = _large_scale_rng(n_samples, n_features, "sq_smoke")
    X, y = _make_large_regression_xy(n_samples, n_features, rng)
    X = _inject_nan(X, rng, frac=0.02)

    sgt, _ = _make_inner_depth_one_pair("regression", "squared_error")
    sgt.fit(X, y)
    pred = sgt.predict(X)

    assert pred.shape == (n_samples,)
    assert np.all(np.isfinite(pred))


@pytest.mark.parametrize("n_samples,n_features", _LARGE_SCALE_SHAPES)
def test_sgt_regressor_squared_error_large_scale_predict_with_new_nan_smoke(
    n_samples: int,
    n_features: int,
) -> None:
    rng = _large_scale_rng(n_samples, n_features, "sq_predict_new_nan")
    X, y = _make_large_regression_xy(n_samples, n_features, rng)

    sgt, _ = _make_inner_depth_one_pair("regression", "squared_error")
    sgt.fit(X, y)

    X_pred = X.copy()
    n_pred_nan = max(20, n_samples // 50)
    for _ in range(n_pred_nan):
        X_pred[rng.integers(0, n_samples), rng.integers(0, n_features)] = np.nan

    pred = sgt.predict(X_pred)
    assert pred.shape == (n_samples,)
    assert np.all(np.isfinite(pred))


@pytest.mark.parametrize("n_samples,n_features", _LARGE_SCALE_SHAPES)
def test_sgt_regressor_absolute_error_large_scale_nan_fit_predict(
    n_samples: int,
    n_features: int,
) -> None:
    """MAE at scale: fit/predict with NaN completes and returns finite values."""
    rng = _large_scale_rng(n_samples, n_features, "absolute_error")
    X, y = _make_large_regression_xy(n_samples, n_features, rng)
    X = _inject_nan(X, rng, frac=0.02)

    reg = SGTRegressor(criterion="absolute_error", inner_max_depth=1, random_state=42)
    reg.fit(X, y)
    preds = reg.predict(X)

    assert preds.shape == (n_samples,)
    assert np.all(np.isfinite(preds))

    X_pred = X.copy()
    n_pred_nan = max(20, n_samples // 50)
    for _ in range(n_pred_nan):
        X_pred[rng.integers(0, n_samples), rng.integers(0, n_features)] = np.nan
    preds_pred = reg.predict(X_pred)
    assert preds_pred.shape == (n_samples,)
    assert np.all(np.isfinite(preds_pred))
