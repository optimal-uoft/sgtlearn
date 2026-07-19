"""Weighted training: discretizers, SGT, and random forests (inner_max_depth=1) vs sklearn."""

from __future__ import annotations

import numpy as np
import pytest
from Discretizers import (
    UnivariateClassificationDiscretizer,
    UnivariateRegressionDiscretizer,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sgtlearn import (
    RandomSGForestClassifier,
    RandomSGForestRegressor,
    SGTClassifier,
    SGTRegressor,
)
from sgtlearn._weights import effective_sample_weight_classification

from tests.constants import TEST_TAO_N_RUNS
from tests.discretizer_grid import n_outputs_params

pytest.importorskip("sklearn")


def _classification_predict(ud: UnivariateClassificationDiscretizer, x: np.ndarray) -> np.ndarray:
    bin_locs = ud.transform(x)
    bin_preds = ud.getBinPredictions()
    return np.asarray(bin_preds[bin_locs], dtype=np.uintp)


def _regression_predict(ud: UnivariateRegressionDiscretizer, x: np.ndarray) -> np.ndarray:
    bin_locs = ud.transform(x)
    bin_preds = ud.getBinPredictions()
    return np.asarray(bin_preds[bin_locs], dtype=np.float32)


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_weighted_univariate_classification_discretizer_matches_sklearn(
    criterion: str,
) -> None:
    rng = np.random.default_rng(0)
    n = 80
    x32 = rng.random((n, 1), dtype=np.float32)
    y = rng.integers(0, 2, size=n, dtype=np.uintp)
    sample_weight = rng.uniform(0.5, 3.0, size=n).astype(np.float32)

    sk = DecisionTreeClassifier(
        criterion=criterion,
        splitter="best",
        min_samples_leaf=3,
        min_impurity_decrease=1e-7,
        max_depth=4,
        random_state=0,
    )
    sk.fit(x32, y, sample_weight=sample_weight)
    sk_preds = sk.predict(x32).astype(np.uintp, copy=False)

    ud = UnivariateClassificationDiscretizer(criterion=criterion)
    ud.Train(
        x32,
        np.array([0], dtype=np.uintp),
        y,
        2,
        3,
        1e-7,
        4,
        0,
        sample_weight=sample_weight,
    )
    ud_preds = _classification_predict(ud, x32)
    np.testing.assert_array_equal(sk_preds, ud_preds)
    assert sk.get_n_leaves() == ud.numLeaves


def test_weighted_univariate_classification_class_weight_via_dict() -> None:
    rng = np.random.default_rng(1)
    n = 60
    x32 = rng.random((n, 1), dtype=np.float32)
    y = rng.integers(0, 2, size=n, dtype=np.uintp)
    class_weight = {0: 1.0, 1: 4.0}

    sk = DecisionTreeClassifier(
        criterion="gini",
        class_weight=class_weight,
        min_samples_leaf=2,
        max_depth=3,
        random_state=0,
    )
    sk.fit(x32, y)

    ud = UnivariateClassificationDiscretizer(criterion="gini")
    sw = np.array([class_weight[int(lbl)] for lbl in y], dtype=np.float32)
    ud.Train(
        x32,
        np.array([0], dtype=np.uintp),
        y,
        2,
        2,
        1e-7,
        3,
        0,
        sample_weight=sw,
    )
    np.testing.assert_array_equal(
        sk.predict(x32).astype(np.uintp), _classification_predict(ud, x32)
    )


@pytest.mark.parametrize("criterion", ["squared_error", "absolute_error"])
def test_weighted_univariate_regression_discretizer_matches_sklearn(
    criterion: str,
) -> None:
    """1-D weighted CART matches sklearn on default tree hyperparameters."""
    rng = np.random.default_rng(2)
    n = 90
    x32 = rng.random((n, 1), dtype=np.float32)
    y = rng.standard_normal(n).astype(np.float32)
    sample_weight = rng.uniform(0.25, 2.0, size=n).astype(np.float32)

    sk = DecisionTreeRegressor(
        criterion=criterion,
        splitter="best",
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        random_state=0,
    )
    sk.fit(x32, y, sample_weight=sample_weight)
    sk_preds = sk.predict(x32).astype(np.float32, copy=False)

    ud = UnivariateRegressionDiscretizer(criterion=criterion)
    ud.Train(
        x32,
        np.array([0], dtype=np.uintp),
        y,
        1,
        0.0,
        0,
        0,
        sample_weight=sample_weight,
    )
    ud_preds = _regression_predict(ud, x32)
    np.testing.assert_allclose(sk_preds, ud_preds, rtol=0, atol=1e-5)
    assert sk.get_n_leaves() == ud.numLeaves


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
        criterion=criterion,
        inner_max_depth=1,
        random_state=42,
        coordinate_descent_smart_init=False,
        tao_n_runs=TEST_TAO_N_RUNS,
    )
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
        criterion="gini",
        inner_max_depth=1,
        class_weight=class_weight,
        random_state=7,
        coordinate_descent_smart_init=False,
        tao_n_runs=TEST_TAO_N_RUNS,
    )
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
        criterion=criterion,
        inner_max_depth=1,
        max_depth=3,
        coordinate_descent_smart_init=False,
        tao_n_runs=TEST_TAO_N_RUNS,
    )
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
        criterion=criterion,
        n_estimators=1,
        bootstrap=False,
        inner_max_depth=1,
        max_features=None,
        random_state=42,
        coordinate_descent_smart_init=False,
        tao_n_runs=TEST_TAO_N_RUNS,
    )
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
        n_estimators=2,
        bootstrap=False,
        class_weight=class_weight,
        random_state=0,
        tao_n_runs=TEST_TAO_N_RUNS,
    )
    forest.fit(X, y)

    from sgtlearn.base import _IdentityLabelEncoder

    for est in forest.estimators_:
        assert est.class_weight is None
        assert isinstance(est._le, _IdentityLabelEncoder)
        np.testing.assert_array_equal(est.classes_, forest.classes_)


def test_random_sg_forest_classifier_class_weight_times_sample_weight_matches_sklearn() -> None:
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
        criterion="gini",
        n_estimators=1,
        bootstrap=False,
        inner_max_depth=1,
        max_features=None,
        class_weight=class_weight,
        random_state=7,
        coordinate_descent_smart_init=False,
        tao_n_runs=TEST_TAO_N_RUNS,
    )
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
        criterion=criterion,
        n_estimators=1,
        bootstrap=False,
        inner_max_depth=1,
        max_depth=3,
        max_features=None,
        coordinate_descent_smart_init=False,
        tao_n_runs=TEST_TAO_N_RUNS,
    )
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
