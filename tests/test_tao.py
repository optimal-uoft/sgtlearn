"""Unit tests for :mod:`sgtlearn.tao`.

No-regression matrices apply to single-tree base estimators only. Random forests
are smoke-tested for successful refinement and intact prediction.
"""

from __future__ import annotations

import zlib
from typing import Any, Callable, Mapping, Optional, Tuple

import numpy as np
import pytest
from sklearn.datasets import (
    load_breast_cancer,
    load_iris,
    make_classification,
    make_regression,
)
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

from sgtlearn import SGTClassifier, SGTRegressor, tao
from sgtlearn.ensemble import RandomSGForestClassifier, RandomSGForestRegressor
from sgtlearn._weights import (
    effective_sample_weight_classification,
    normalize_sample_weight,
)

pytest.importorskip("sklearn")

RTOL = 1e-9


def _no_regression_tol(before: float, *, weighted: bool) -> float:
    """Allow tiny float32 vs float64 drift when sample/class weights are active."""
    tol = max(RTOL, 1e-6 * (1.0 + abs(before)))
    if weighted:
        tol = max(tol, 2e-4)
    return tol


def _fit_classifier(
    X: np.ndarray,
    y: np.ndarray,
    *,
    criterion: str = "gini",
    class_weight: Optional[Mapping[Any, float] | str] = None,
    sample_weight: Optional[np.ndarray] = None,
    **tree_kwargs: Any,
) -> SGTClassifier:
    params = dict(
        criterion=criterion,
        max_depth=4,
        min_samples_leaf=3,
        inner_max_depth=4,
        inner_max_leaf_nodes=16,
        random_state=42,
        class_weight=class_weight,
    )
    params.update(tree_kwargs)
    est = SGTClassifier(**params)
    est.fit(X, y, sample_weight=sample_weight)
    return est


def _fit_regressor(
    X: np.ndarray,
    y: np.ndarray,
    *,
    criterion: str = "squared_error",
    sample_weight: Optional[np.ndarray] = None,
    **tree_kwargs: Any,
) -> SGTRegressor:
    params = dict(
        criterion=criterion,
        max_depth=4,
        min_samples_leaf=3,
        inner_max_depth=4,
        inner_max_leaf_nodes=16,
        random_state=42,
    )
    params.update(tree_kwargs)
    est = SGTRegressor(**params)
    est.fit(X, y, sample_weight=sample_weight)
    return est


def _classification_data(name: str) -> Tuple[np.ndarray, np.ndarray]:
    if name == "iris":
        X, y = load_iris(return_X_y=True)
    elif name == "breast_cancer":
        X, y = load_breast_cancer(return_X_y=True)
    elif name == "multiclass":
        X, y = make_classification(
            n_samples=600,
            n_features=20,
            n_informative=10,
            n_redundant=4,
            n_classes=4,
            random_state=7,
        )
    else:
        raise ValueError(f"unknown classification dataset {name!r}")
    return np.asarray(X, dtype=np.float64), y


def _regression_data(name: str) -> Tuple[np.ndarray, np.ndarray]:
    if name == "low_noise":
        X, y = make_regression(
            n_samples=400,
            n_features=10,
            n_informative=6,
            noise=5.0,
            random_state=0,
        )
    elif name == "high_noise":
        X, y = make_regression(
            n_samples=400,
            n_features=10,
            n_informative=6,
            noise=25.0,
            random_state=3,
        )
    else:
        raise ValueError(f"unknown regression dataset {name!r}")
    return np.asarray(X, dtype=np.float64), y


def _stable_seed(*parts: str) -> int:
    return zlib.adler32("|".join(parts).encode()) & 0xFFFFFFFF


def _sample_weights(n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.5, 2.0, size=n_samples)


def _classification_training_score(
    tree: SGTClassifier,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray],
) -> float:
    pred = tree.predict(X)
    if tree.class_weight is None and sample_weight is None:
        return float(accuracy_score(y, pred))

    y_enc = tree._le.transform(np.asarray(y).ravel())  # type: ignore[union-attr]
    if tree.class_weight is not None:
        sw = effective_sample_weight_classification(
            sample_weight, y_enc, tree.class_weight, tree.classes_
        )
    else:
        sw = normalize_sample_weight(sample_weight, len(y))
    correct = (pred == y).astype(np.float64)
    return float(np.average(correct, weights=sw))


def _skewed_class_weight(y: np.ndarray) -> Mapping[Any, float]:
    """Up-weight the first observed class; valid for any label dtype."""
    classes = np.unique(y)
    return {c: (2.0 if i == 0 else 1.0) for i, c in enumerate(classes)}


def _regression_training_loss(
    reg: SGTRegressor,
    X: np.ndarray,
    y: np.ndarray,
    criterion: str,
    sample_weight: Optional[np.ndarray],
) -> float:
    pred = reg.predict(X)
    if criterion in ("squared_error", "mse"):
        err = (pred - y) ** 2
        if sample_weight is None:
            return float(mean_squared_error(y, pred))
        return float(np.average(err, weights=sample_weight))
    err = np.abs(pred - y)
    if sample_weight is None:
        return float(mean_absolute_error(y, pred))
    return float(np.average(err, weights=sample_weight))


@pytest.mark.parametrize("dataset", ["iris", "breast_cancer", "multiclass"])
@pytest.mark.parametrize("criterion", ["gini", "entropy"])
@pytest.mark.parametrize(
    "use_sample_weight", [False, True], ids=["unweighted", "weighted"]
)
@pytest.mark.parametrize(
    "use_class_weight", [False, True], ids=["no_class_weight", "skewed_class_weight"]
)
@pytest.mark.parametrize("n_runs", [1, 10])
def test_tao_classification_no_regression_matrix(
    dataset: str,
    criterion: str,
    use_sample_weight: bool,
    use_class_weight: bool,
    n_runs: int,
) -> None:
    """TAO must not decrease training accuracy across the classification grid."""
    X, y = _classification_data(dataset)
    sw = (
        _sample_weights(len(y), seed=_stable_seed(dataset, criterion, "weighted"))
        if use_sample_weight
        else None
    )
    class_weight = _skewed_class_weight(y) if use_class_weight else None

    tree_kwargs: dict[str, Any] = {}
    if dataset == "multiclass":
        tree_kwargs = dict(max_depth=6, num_partitions=3)

    tree = _fit_classifier(
        X,
        y,
        criterion=criterion,
        class_weight=class_weight,
        sample_weight=sw,
        **tree_kwargs,
    )

    before = _classification_training_score(tree, X, y, sw)
    returned = tao.TAO_refine(
        tree,
        X,
        y,
        sample_weight=sw,
        n_runs=n_runs,
        lambda_=0.0,
    )
    after = _classification_training_score(tree, X, y, sw)

    assert returned is tree
    weighted_metric = use_sample_weight or use_class_weight
    assert after >= before - _no_regression_tol(before, weighted=weighted_metric), (
        f"TAO decreased training score for dataset={dataset!r}, "
        f"criterion={criterion!r}, weighted={use_sample_weight}, "
        f"class_weight={use_class_weight}, n_runs={n_runs}: "
        f"{before:.6f} -> {after:.6f}"
    )


@pytest.mark.parametrize("dataset", ["low_noise", "high_noise"])
@pytest.mark.parametrize("criterion", ["squared_error", "absolute_error"])
@pytest.mark.parametrize(
    "use_sample_weight", [False, True], ids=["unweighted", "weighted"]
)
@pytest.mark.parametrize("n_runs", [1, 10])
def test_tao_regression_no_regression_matrix(
    dataset: str,
    criterion: str,
    use_sample_weight: bool,
    n_runs: int,
) -> None:
    """TAO must not increase training loss across the regression grid."""
    X, y = _regression_data(dataset)
    sw = (
        _sample_weights(len(y), seed=_stable_seed(dataset, criterion, "weighted"))
        if use_sample_weight
        else None
    )

    reg = _fit_regressor(X, y, criterion=criterion, sample_weight=sw)

    before = _regression_training_loss(reg, X, y, criterion, sw)
    returned = tao.TAO_refine(
        reg,
        X,
        y,
        sample_weight=sw,
        n_runs=n_runs,
        lambda_=0.0,
    )
    after = _regression_training_loss(reg, X, y, criterion, sw)

    assert returned is reg
    tol = _no_regression_tol(before, weighted=use_sample_weight)
    assert after <= before + tol, (
        f"TAO increased training loss for dataset={dataset!r}, "
        f"criterion={criterion!r}, weighted={use_sample_weight}, "
        f"n_runs={n_runs}: "
        f"{before:.6f} -> {after:.6f}"
    )


@pytest.mark.parametrize(
    ("estimator_cls", "fit_fn", "data_fn"),
    [
        pytest.param(
            SGTClassifier,
            lambda X, y: _fit_classifier(X, y),
            lambda: _classification_data("iris"),
            id="classifier",
        ),
        pytest.param(
            SGTRegressor,
            lambda X, y: _fit_regressor(X, y),
            lambda: _regression_data("low_noise"),
            id="regressor",
        ),
    ],
)
def test_tao_refine_mutates_in_place(
    estimator_cls: type,
    fit_fn: Callable[..., Any],
    data_fn: Callable[[], Tuple[np.ndarray, np.ndarray]],
) -> None:
    """TAO_refine returns the same wrapper and keeps the native handle."""
    X, y = data_fn()
    est = fit_fn(X, y)
    native_before = est._est

    result = tao.TAO_refine(est, X, y)

    assert result is est
    assert est._est is native_before


@pytest.mark.parametrize(
    ("estimator", "X", "y", "exc_type"),
    [
        pytest.param(
            SGTClassifier(),
            *load_iris(return_X_y=True),
            NotFittedError,
            id="unfitted_classifier",
        ),
        pytest.param(
            SGTRegressor(),
            *make_regression(n_samples=50, n_features=4, random_state=0),
            NotFittedError,
            id="unfitted_regressor",
        ),
    ],
)
def test_tao_rejects_unfitted(estimator, X, y, exc_type) -> None:
    with pytest.raises(exc_type):
        tao.TAO_refine(estimator, X, y)


def test_tao_rejects_wrong_type() -> None:
    X, y = load_iris(return_X_y=True)
    with pytest.raises(TypeError):
        tao.TAO_refine(object(), X, y)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("fit_fn", "data_fn"),
    [
        pytest.param(
            lambda X, y: _fit_classifier(X, y),
            lambda: _classification_data("iris"),
            id="classifier",
        ),
        pytest.param(
            lambda X, y: _fit_regressor(X, y),
            lambda: _regression_data("low_noise"),
            id="regressor",
        ),
    ],
)
def test_tao_rejects_feature_mismatch(fit_fn, data_fn) -> None:
    X, y = data_fn()
    est = fit_fn(X, y)
    with pytest.raises(ValueError):
        tao.TAO_refine(est, X[:, :-1], y)


def test_tao_accepts_check_input_false() -> None:
    """Callers that pre-validate arrays can skip redundant checks."""
    X, y = _classification_data("iris")
    tree = _fit_classifier(X, y)
    before = _classification_training_score(tree, X, y, None)
    tao.TAO_refine(tree, X, y, check_input=False)
    after = _classification_training_score(tree, X, y, None)
    assert after >= before - RTOL


def _fit_forest_classifier(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_estimators: int = 5,
    n_jobs: int = 1,
    **kwargs: Any,
) -> RandomSGForestClassifier:
    params = dict(
        n_estimators=n_estimators,
        criterion="gini",
        max_depth=4,
        min_samples_leaf=3,
        inner_max_depth=4,
        inner_max_leaf_nodes=16,
        random_state=42,
        n_jobs=n_jobs,
    )
    params.update(kwargs)
    return RandomSGForestClassifier(**params).fit(X, y)


def _fit_forest_regressor(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_estimators: int = 5,
    n_jobs: int = 1,
    criterion: str = "squared_error",
    **kwargs: Any,
) -> RandomSGForestRegressor:
    params = dict(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=4,
        min_samples_leaf=3,
        inner_max_depth=4,
        inner_max_leaf_nodes=16,
        random_state=42,
        n_jobs=n_jobs,
    )
    params.update(kwargs)
    return RandomSGForestRegressor(**params).fit(X, y)


@pytest.mark.parametrize(
    ("forest_cls", "fit_fn", "data_fn", "predict_fn"),
    [
        pytest.param(
            RandomSGForestClassifier,
            lambda X, y: _fit_forest_classifier(X, y, n_estimators=8, n_jobs=1),
            lambda: _classification_data("iris"),
            lambda forest, X: forest.predict(X),
            id="classifier",
        ),
        pytest.param(
            RandomSGForestRegressor,
            lambda X, y: _fit_forest_regressor(X, y, n_estimators=8, n_jobs=1),
            lambda: _regression_data("low_noise"),
            lambda forest, X: forest.predict(X),
            id="regressor",
        ),
    ],
)
def test_tao_forest_runs_and_predicts_all_samples(
    forest_cls, fit_fn, data_fn, predict_fn
) -> None:
    """TAO on a random forest completes and leaves predict() valid for every row."""
    X, y = data_fn()
    forest = fit_fn(X, y)

    returned = tao.TAO_refine(forest, X, y, n_runs=3, n_jobs=2)

    assert returned is forest
    pred = predict_fn(forest, X)
    assert pred.shape == (X.shape[0],)
    if forest_cls is RandomSGForestClassifier:
        proba = forest.predict_proba(X)
        assert proba.shape == (X.shape[0], forest.n_classes_)


def test_tao_forest_refine_mutates_in_place() -> None:
    X, y = _classification_data("iris")
    forest = _fit_forest_classifier(X, y, n_estimators=4)
    handles_before = [est._est for est in forest.estimators_]

    result = tao.TAO_refine(forest, X, y, n_jobs=2)

    assert result is forest
    assert [est._est for est in forest.estimators_] == handles_before
