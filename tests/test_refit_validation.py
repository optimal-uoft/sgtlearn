"""A refit that fails leaves the fitted tree or forest unchanged."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest

from sgtlearn import (
    RandomSGForestClassifier,
    RandomSGForestRegressor,
    SGTClassifier,
    SGTRegressor,
)
from tests.constants import TEST_TAO_N_RUNS

FORESTS = [RandomSGForestClassifier, RandomSGForestRegressor]
ESTIMATORS = [SGTClassifier, SGTRegressor, *FORESTS]
CLASSIFIERS = [SGTClassifier, RandomSGForestClassifier]


def _fit(estimator_cls: type, **params: Any) -> tuple[Any, np.ndarray, np.ndarray]:
    """``(model, X, y)``: ``model`` fitted on 3-feature ``X`` and labels ``y``."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    y = (X[:, 0] > 0).astype(int)
    params.setdefault("tao_n_runs", TEST_TAO_N_RUNS)
    if estimator_cls in FORESTS:
        params.setdefault("n_estimators", 2)
    return estimator_cls(random_state=0, **params).fit(X, y), X, y


def _widen(X: np.ndarray) -> np.ndarray:
    """``X`` plus a column, so a half-applied refit changes ``n_features_in_``."""
    return np.column_stack([X, X[:, 0]])


def _fitted_attrs(model: Any) -> dict[str, Any]:
    """Fitted state: trailing-underscore and private instance attributes."""
    return {
        k: v for k, v in vars(model).items() if k.endswith("_") or k.startswith("_")
    }


@contextmanager
def _rejected_refit(
    model: Any, X: np.ndarray, error: type[BaseException], match: str | None = None
) -> Iterator[None]:
    """Expect the block to raise ``error`` without touching ``model``'s fit."""
    before = model.predict(X)
    fitted = _fitted_attrs(model)
    with pytest.raises(error, match=match):
        yield
    after = _fitted_attrs(model)
    assert after.keys() == fitted.keys()
    for name, value in fitted.items():
        assert after[name] is value, name
    np.testing.assert_array_equal(model.predict(X), before)


@pytest.mark.parametrize("estimator_cls", CLASSIFIERS)
def test_rejected_single_class_refit_keeps_classifier(estimator_cls) -> None:
    model, X, _ = _fit(estimator_cls)
    with _rejected_refit(model, X, ValueError, "at least two classes"):
        model.fit(X, np.full(X.shape[0], 7))


def test_rejected_unchecked_multioutput_refit_keeps_classifier() -> None:
    model, X, y = _fit(SGTClassifier)
    with _rejected_refit(model, X, ValueError, "entry per output"):
        model.fit(X, np.column_stack([y, y]), check_input=False)


@pytest.mark.parametrize("estimator_cls", CLASSIFIERS)
def test_rejected_class_weight_refit_keeps_classifier(estimator_cls) -> None:
    model, X, y = _fit(estimator_cls)
    with _rejected_refit(model, X, ValueError, "class_weight key 99"):
        model.set_params(class_weight={99: 1.0}).fit(X, y + 5)


@pytest.mark.parametrize("estimator_cls", ESTIMATORS)
def test_rejected_sample_weight_refit_keeps_fitted_tree(estimator_cls) -> None:
    model, X, y = _fit(estimator_cls)
    # New labels, so attributes written before the check would show.
    with _rejected_refit(model, X, ValueError, "sample_weight must be non-negative"):
        model.fit(_widen(X), y + 5, sample_weight=-np.ones(X.shape[0]))


@pytest.mark.parametrize("estimator_cls", ESTIMATORS)
@pytest.mark.parametrize(
    ("param", "value"),
    [
        ("min_impurity_decrease", -1.0),
        ("pairwise_penalty", np.inf),
        ("branching_penalty", -1.0),
        ("tao_pair_scale", -1.0),
        ("pairwise_candidates", -1),
        ("tao_n_runs", -1),
        ("tao_n_runs", 2.5),
        ("tao_n_runs", True),
        ("tao_n_runs", None),
    ],
)
def test_rejected_hyperparameter_refit_keeps_fitted_tree(
    estimator_cls, param, value
) -> None:
    model, X, y = _fit(estimator_cls)
    with _rejected_refit(model, X, ValueError, param):
        model.set_params(**{param: value}).fit(_widen(X), y)


@pytest.mark.parametrize("estimator_cls", ESTIMATORS)
def test_rejected_feature_dict_refit_keeps_fitted_tree(estimator_cls) -> None:
    model, X, y = _fit(estimator_cls)
    with _rejected_refit(model, X, ValueError, "out of range"):
        model.fit(_widen(X), y, feature_dict={0: [99]})


@pytest.mark.parametrize("estimator_cls", ESTIMATORS)
def test_rejected_processed_features_refit_keeps_fitted_tree(estimator_cls) -> None:
    model, X, y = _fit(estimator_cls)
    with _rejected_refit(model, X, TypeError, "processed_features"):
        model.fit(_widen(X), y, processed_features="not resolved features")


@pytest.mark.parametrize("tao_n_runs", [0, 2])
def test_rejected_float32_overflow_y_refit_keeps_regressor(tao_n_runs) -> None:
    model, X, y = _fit(SGTRegressor, tao_n_runs=tao_n_runs)
    y_bad = y.astype(float)
    y_bad[0] = 1e39  # finite in float64, inf in float32
    with _rejected_refit(model, X, ValueError, "too large for dtype"):
        model.fit(_widen(X), y_bad)


def test_rejected_float32_overflow_y_refit_keeps_forest() -> None:
    model, X, y = _fit(RandomSGForestRegressor)
    y_bad = y.astype(float)
    y_bad[0] = 1e39
    with _rejected_refit(model, X, ValueError, "too large for dtype"):
        model.fit(_widen(X), y_bad)


@pytest.mark.parametrize("estimator_cls", FORESTS)
def test_rejected_max_samples_refit_keeps_forest(estimator_cls) -> None:
    model, X, y = _fit(estimator_cls)
    with _rejected_refit(model, X, ValueError, "max_samples"):
        model.set_params(max_samples=1000).fit(_widen(X), y)


@pytest.mark.parametrize("estimator_cls", FORESTS)
@pytest.mark.parametrize("n_estimators", [0, 2.5, True])
def test_rejected_n_estimators_refit_keeps_forest(estimator_cls, n_estimators) -> None:
    model, X, y = _fit(estimator_cls)
    with _rejected_refit(model, X, ValueError, "n_estimators"):
        model.set_params(n_estimators=n_estimators).fit(_widen(X), y)


@pytest.mark.parametrize("estimator_cls", ESTIMATORS)
def test_refit_failing_in_tao_restores_fitted_tree(estimator_cls) -> None:
    # Rejected only by the native TAO call, after the new tree is trained.
    model, X, y = _fit(estimator_cls, tao_n_runs=2)
    assert all(t._tao_refined_ for t in getattr(model, "estimators_", [model]))
    with _rejected_refit(model, X, TypeError):
        model.set_params(tao_lambda="x").fit(_widen(X), y)
