"""Weighted training: SGT and random forests (inner_max_depth=1) vs sklearn."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sgtlearn import (
    RandomSGForestClassifier,
    RandomSGForestRegressor,
    SGTClassifier,
    SGTRegressor,
)
from sgtlearn._weights import (
    effective_sample_weight_classification,
    normalize_sample_weight,
)
from tests.constants import TEST_TAO_N_RUNS, algorithm_params
from tests.discretizer_grid import n_outputs_params


@pytest.mark.parametrize("n_outputs", n_outputs_params())
@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_sgt_classifier_weighted_inner_depth_one_matches_sklearn(
    criterion: str,
    n_outputs: int,
) -> None:
    from sklearn.datasets import load_breast_cancer

    X, y0 = load_breast_cancer(return_X_y=True)
    X = X.astype(np.float32)
    if n_outputs == 1:
        y = y0
    else:
        extras = [
            (X[:, i] > np.median(X[:, i])).astype(np.int64)
            for i in range(n_outputs - 1)
        ]
        y = np.column_stack([y0, *extras])
    rng = np.random.default_rng(3)
    sample_weight = rng.uniform(0.5, 2.5, size=X.shape[0]).astype(np.float32)

    sk = DecisionTreeClassifier(
        criterion=criterion,
        min_samples_leaf=1,
        random_state=42,
    )
    sk.fit(X, y, sample_weight=sample_weight)

    sgt = SGTClassifier(
        **algorithm_params(criterion=criterion,
        inner_max_depth=1,
        random_state=42,
        tao_n_runs=TEST_TAO_N_RUNS,
    ))
    sgt.fit(X, y, sample_weight=sample_weight)

    np.testing.assert_array_equal(sgt.predict(X), sk.predict(X))
    if n_outputs == 1:
        np.testing.assert_allclose(
            sgt.predict_proba(X), sk.predict_proba(X), rtol=0, atol=1e-6
        )
    else:
        for a, b in zip(sgt.predict_proba(X), sk.predict_proba(X)):
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)


def test_sgt_classifier_class_weight_times_sample_weight_matches_sklearn() -> None:
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True)
    X = X.astype(np.float32)
    rng = np.random.default_rng(4)
    sample_weight = rng.uniform(1.0, 2.0, size=len(y)).astype(np.float32)
    class_weight = {0: 1.0, 1: 3.0}

    sk = DecisionTreeClassifier(
        criterion="gini",
        class_weight=class_weight,
        random_state=7,
    )
    sk.fit(X, y, sample_weight=sample_weight)

    sgt = SGTClassifier(
        **algorithm_params(criterion="gini",
        inner_max_depth=1,
        class_weight=class_weight,
        random_state=7,
        tao_n_runs=TEST_TAO_N_RUNS,
    ))
    sgt.fit(X, y, sample_weight=sample_weight)

    np.testing.assert_array_equal(sgt.predict(X), sk.predict(X))


@pytest.mark.parametrize("n_outputs", n_outputs_params())
@pytest.mark.parametrize("criterion", ["absolute_error", "squared_error"])
def test_sgt_regressor_weighted_inner_depth_one_matches_sklearn(
    criterion: str,
    n_outputs: int,
) -> None:
    """Weighted SGT with ``inner_max_depth=1`` matches weighted sklearn in-sample."""
    from sklearn.datasets import load_diabetes

    X, y0 = load_diabetes(return_X_y=True)
    X = np.asarray(X, dtype=np.float32)
    y0 = np.asarray(y0, dtype=np.float32)
    if n_outputs == 1:
        y = y0
    else:
        rng_y = np.random.default_rng(0)
        extras = [
            y0 * (0.5 + 0.25 * i) + rng_y.normal(0.0, 5.0, size=y0.shape)
            for i in range(1, n_outputs)
        ]
        y = np.column_stack([y0, *extras]).astype(np.float32)
    rng = np.random.default_rng(5)
    sample_weight = rng.uniform(0.3, 1.5, size=X.shape[0]).astype(np.float32)

    sk = DecisionTreeRegressor(criterion=criterion, max_depth=3)
    sk.fit(X, y, sample_weight=sample_weight)

    sgt = SGTRegressor(
        **algorithm_params(criterion=criterion,
        inner_max_depth=1,
        max_depth=3,
        tao_n_runs=TEST_TAO_N_RUNS,
    ))
    sgt.fit(X, y, sample_weight=sample_weight)

    np.testing.assert_allclose(sgt.predict(X), sk.predict(X), rtol=0, atol=1e-5)


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_random_sg_forest_classifier_weighted_inner_depth_one_matches_sklearn(
    criterion: str,
) -> None:
    """One-tree forest (no bootstrap) with ``inner_max_depth=1`` matches weighted CART."""
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True)
    X = X.astype(np.float32)
    rng = np.random.default_rng(6)
    sample_weight = rng.uniform(0.5, 2.5, size=len(y)).astype(np.float32)

    sk = DecisionTreeClassifier(
        criterion=criterion,
        min_samples_leaf=1,
        random_state=42,
    )
    sk.fit(X, y, sample_weight=sample_weight)

    forest = RandomSGForestClassifier(
        **algorithm_params(criterion=criterion,
        n_estimators=1,
        bootstrap=False,
        inner_max_depth=1,
        max_features=None,
        random_state=42,
        tao_n_runs=TEST_TAO_N_RUNS,
    ))
    forest.fit(X, y, sample_weight=sample_weight)

    np.testing.assert_array_equal(forest.predict(X), sk.predict(X))
    np.testing.assert_allclose(
        forest.predict_proba(X), sk.predict_proba(X), rtol=0, atol=1e-6
    )


def test_random_sg_forest_classifier_applies_class_weight_once() -> None:
    """Forest merges class_weight into sample_weight; base trees must not."""
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True)
    X = X.astype(np.float32)
    class_weight = {0: 1.0, 1: 2.0}

    forest = RandomSGForestClassifier(
        **algorithm_params(n_estimators=2,
        bootstrap=False,
        class_weight=class_weight,
        random_state=0,
        tao_n_runs=TEST_TAO_N_RUNS,
    ))
    forest.fit(X, y)

    from sgtlearn.base import _IdentityLabelEncoder

    for est in forest.estimators_:
        assert est.class_weight is None
        assert isinstance(est._le, _IdentityLabelEncoder)
        np.testing.assert_array_equal(est.classes_, forest.classes_)


def test_random_sg_forest_classifier_class_weight_times_sample_weight_matches_sklearn() -> (
    None
):
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True)
    X = X.astype(np.float32)
    rng = np.random.default_rng(7)
    sample_weight = rng.uniform(1.0, 2.0, size=len(y)).astype(np.float32)
    class_weight = {0: 1.0, 1: 3.0}

    sk = DecisionTreeClassifier(
        criterion="gini",
        class_weight=class_weight,
        random_state=7,
    )
    sk.fit(X, y, sample_weight=sample_weight)

    forest = RandomSGForestClassifier(
        **algorithm_params(criterion="gini",
        n_estimators=1,
        bootstrap=False,
        inner_max_depth=1,
        max_features=None,
        class_weight=class_weight,
        random_state=7,
        tao_n_runs=TEST_TAO_N_RUNS,
    ))
    forest.fit(X, y, sample_weight=sample_weight)

    np.testing.assert_array_equal(forest.predict(X), sk.predict(X))


@pytest.mark.parametrize("criterion", ["absolute_error", "squared_error"])
def test_random_sg_forest_regressor_weighted_inner_depth_one_matches_sklearn(
    criterion: str,
) -> None:
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    rng = np.random.default_rng(8)
    sample_weight = rng.uniform(0.3, 1.5, size=len(y)).astype(np.float32)

    sk = DecisionTreeRegressor(criterion=criterion, max_depth=3)
    sk.fit(X, y, sample_weight=sample_weight)

    forest = RandomSGForestRegressor(
        **algorithm_params(criterion=criterion,
        n_estimators=1,
        bootstrap=False,
        inner_max_depth=1,
        max_depth=3,
        max_features=None,
        tao_n_runs=TEST_TAO_N_RUNS,
    ))
    forest.fit(X, y, sample_weight=sample_weight)

    np.testing.assert_allclose(forest.predict(X), sk.predict(X), rtol=0, atol=1e-5)


def test_effective_sample_weight_classification_helper() -> None:
    y_enc = np.array([0, 1, 0], dtype=np.int64)
    classes_ = np.array([0, 1])
    sw = effective_sample_weight_classification(
        np.array([2.0, 2.0, 2.0], dtype=np.float64),
        y_enc,
        {0: 1.0, 1: 5.0},
        classes_,
    )
    np.testing.assert_allclose(sw, [2.0, 10.0, 2.0])
    assert sw.dtype == np.float32
    assert sw.flags.c_contiguous


def test_effective_sample_weight_multioutput_list_of_dicts() -> None:
    """Multioutput ``class_weight`` as a list of dicts multiplies into one weight vector."""
    y_enc = np.array([[0, 1], [1, 0], [0, 0]], dtype=np.int64)
    classes_ = [np.array([0, 1]), np.array([0, 1])]
    sw = effective_sample_weight_classification(
        np.array([1.0, 1.0, 1.0], dtype=np.float64),
        y_enc,
        [{0: 2.0, 1: 1.0}, {0: 1.0, 1: 3.0}],
        classes_,
    )
    # row0: 2*3=6, row1: 1*1=1, row2: 2*1=2
    np.testing.assert_allclose(sw, [6.0, 1.0, 2.0])


def test_normalize_sample_weight_allows_zero_with_positive_weight() -> None:
    sw = normalize_sample_weight(np.array([0.0, 2.0]), 2)
    assert sw is not None
    np.testing.assert_array_equal(sw, [0.0, 2.0])
    assert sw.dtype == np.float32
    assert sw.flags.c_contiguous


def test_zero_weight_row_has_no_regressor_contribution() -> None:
    X = np.array([[0.0], [1.0], [2.0], [100.0]])
    y = np.array([0.0, 0.0, 1.0, 100.0])
    params = {"max_depth": 2, "tao_n_runs": 0, "random_state": 0}

    weighted = SGTRegressor(**algorithm_params(**params)).fit(X, y, sample_weight=[1.0, 1.0, 1.0, 0.0])
    omitted = SGTRegressor(**algorithm_params(**params)).fit(X[:3], y[:3])

    np.testing.assert_allclose(weighted.predict(X[:3]), omitted.predict(X[:3]))


@pytest.mark.parametrize(
    ("sample_weight", "n_samples", "match"),
    [
        ([1.0], 2, "shape"),
        ([1.0, -1.0], 2, "non-negative"),
        ([0.0, 0.0], 2, "positive"),
    ],
)
def test_normalize_sample_weight_rejects_invalid_values(
    sample_weight: list[float], n_samples: int, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        normalize_sample_weight(np.asarray(sample_weight), n_samples)


@pytest.mark.parametrize(
    ("class_weight", "classes", "match", "exc_type"),
    [
        ({2: 1.0}, np.array([0, 1]), "training classes", ValueError),
        ({1: -1.0}, np.array([0, 1]), "non-negative", ValueError),
        (1.0, np.array([0, 1]), "mapping", TypeError),
        ([{0: 1.0}], [np.array([0, 1]), np.array([0, 1])], "one mapping", ValueError),
        ({0: 1.0}, [np.array([0, 1])], "one class array", ValueError),
    ],
)
def test_effective_sample_weight_rejects_invalid_class_configuration(
    class_weight: object,
    classes: object,
    match: str,
    exc_type: type[Exception],
) -> None:
    y = np.array([[0, 1], [1, 0]]) if isinstance(classes, list) else np.array([0, 1])
    with pytest.raises(exc_type, match=match):
        effective_sample_weight_classification(None, y, class_weight, classes)  # type: ignore[arg-type]
