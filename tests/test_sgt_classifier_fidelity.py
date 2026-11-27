"""
Fidelity: SGTClassifier vs sklearn DecisionTreeClassifier on full data.

Uses Wisconsin breast cancer (binary). Both models use aligned depth / leaf
constraints so the comparison reflects inductive bias rather than an unrestricted
deep CART baseline.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.metrics import accuracy_score, log_loss
from sklearn.tree import DecisionTreeClassifier

from sgtlearn import RandomSGForestClassifier, SGTClassifier
from tests.constants import algorithm_params
from tests.discretizer_grid import n_outputs_params


def _load_xy(n_outputs: int = 1):
    X, y0 = load_breast_cancer(return_X_y=True)
    X = np.asarray(X, dtype=np.float32)
    if n_outputs == 1:
        return X, y0
    extras = [
        (X[:, i] > np.median(X[:, i])).astype(np.int64)
        for i in range(n_outputs - 1)
    ]
    return X, np.column_stack([y0, *extras])


@pytest.mark.parametrize("tao_n_runs", [0, 10], ids=["no_tao", "with_tao"])
@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_sgt_train_accuracy_at_least_sklearn_decision_tree(
    criterion: str, tao_n_runs: int
) -> None:
    """``SGTClassifier`` training accuracy on breast cancer must be >= a tuned ``DecisionTreeClassifier``."""
    X, y = load_breast_cancer(return_X_y=True)
    X = np.asarray(X, dtype=np.float32)

    max_depth = 5
    min_samples_leaf = 5

    sgt = SGTClassifier(
        **algorithm_params(criterion=criterion,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_impurity_decrease=1e-7,
        inner_max_depth=8,
        inner_max_leaf_nodes=48,
        coordinate_descent_max_iters=25,
        coordinate_descent_patience=6,
        random_state=42,
        tao_n_runs=tao_n_runs,
    ))
    sgt.fit(X, y)

    dt = DecisionTreeClassifier(
        criterion="entropy" if criterion == "log_loss" else criterion,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_impurity_decrease=1e-7,
        random_state=42,
    )
    dt.fit(X, y)

    sgt_acc = accuracy_score(y, sgt.predict(X))
    dt_acc = accuracy_score(y, dt.predict(X))

    assert sgt_acc >= dt_acc, (
        f"Training accuracy SGT ({sgt_acc:.6f}) should be >= sklearn "
        f"DecisionTree ({dt_acc:.6f}) for criterion={criterion!r}"
    )


@pytest.mark.parametrize("n_outputs", n_outputs_params())
@pytest.mark.parametrize("tao_n_runs", [0, 10], ids=["no_tao", "with_tao"])
@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_sgt_matches_sklearn_decision_tree_inner_depth_one_defaults(
    criterion: str,
    tao_n_runs: int,
    n_outputs: int,
) -> None:
    """``inner_max_depth=1`` only; all other hyperparameters are estimator defaults.

    Compared against ``DecisionTreeClassifier()`` with the same ``criterion`` so
    both trainers use their respective default stopping rules and depth limits.
    """
    X, y = _load_xy(n_outputs)

    sk_criterion = "entropy" if criterion == "log_loss" else criterion

    sgt = SGTClassifier(
        **algorithm_params(criterion=criterion, inner_max_depth=1, tao_n_runs=tao_n_runs
    ))
    sgt.fit(X, y)

    dt = DecisionTreeClassifier(criterion=sk_criterion)
    dt.fit(X, y)

    expected_shape = (X.shape[0],) if n_outputs == 1 else (X.shape[0], n_outputs)
    pred = sgt.predict(X)
    assert pred.shape == expected_shape
    np.testing.assert_array_equal(pred, dt.predict(X))

    if n_outputs == 1:
        np.testing.assert_array_equal(sgt.classes_, dt.classes_)
        np.testing.assert_allclose(sgt.predict_proba(X), dt.predict_proba(X))
    else:
        assert len(sgt.classes_) == len(dt.classes_) == n_outputs
        for a, b in zip(sgt.classes_, dt.classes_):
            np.testing.assert_array_equal(a, b)
        sgt_proba = sgt.predict_proba(X)
        dt_proba = dt.predict_proba(X)
        assert len(sgt_proba) == len(dt_proba) == n_outputs
        for a, b in zip(sgt_proba, dt_proba):
            np.testing.assert_allclose(a, b)


def _proba_data(n_outputs: int):
    X, y = make_classification(
        n_samples=600,
        n_features=6,
        n_informative=4,
        n_classes=3,
        weights=[0.5, 0.3, 0.2],
        random_state=0,
    )
    weights = np.random.default_rng(0).uniform(0.1, 3.0, size=len(y))
    if n_outputs == 2:
        y = np.column_stack([y, (X[:, 0] > 0).astype(int)])
    return X, y, weights


@pytest.mark.parametrize("n_outputs", [1, 2])
@pytest.mark.parametrize(
    "make_model",
    [
        lambda: SGTClassifier(max_depth=6, tao_n_runs=0, random_state=0),
        lambda: RandomSGForestClassifier(
            n_estimators=3, max_depth=6, tao_n_runs=0, random_state=0
        ),
    ],
    ids=["tree", "forest"],
)
def test_predict_proba_rows_sum_to_one(make_model, n_outputs):
    X, y, weights = _proba_data(n_outputs)
    model = make_model().fit(X, y, sample_weight=weights)
    proba = model.predict_proba(X)
    per_output = [proba] if n_outputs == 1 else proba
    for o, p in enumerate(per_output):
        assert p.dtype == np.float64
        assert np.max(np.abs(p.sum(axis=1) - 1.0)) <= 1e-12
        y_o = y if n_outputs == 1 else y[:, o]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            log_loss(y_o, p, labels=np.arange(p.shape[1]))
