"""
Fidelity: SGTClassifier vs sklearn DecisionTreeClassifier on full data.

Uses Wisconsin breast cancer (binary). Both models use aligned depth / leaf
constraints so the comparison reflects inductive bias rather than an unrestricted
deep CART baseline.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from sgtlearn import SGTClassifier

pytest.importorskip("sklearn")


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_sgt_train_accuracy_at_least_sklearn_decision_tree(criterion: str) -> None:
    """``SGTClassifier`` training accuracy on breast cancer must be >= a tuned ``DecisionTreeClassifier``."""
    X, y = load_breast_cancer(return_X_y=True)
    X = np.asarray(X, dtype=np.float32)

    max_depth = 5
    min_samples_leaf = 5

    sgt = SGTClassifier(
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


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_sgt_matches_sklearn_decision_tree_inner_depth_one_defaults(
    criterion: str,
) -> None:
    """``inner_max_depth=1`` only; all other hyperparameters are estimator defaults.

    Compared against ``DecisionTreeClassifier()`` with the same ``criterion`` so
    both trainers use their respective default stopping rules and depth limits.
    """
    X, y = load_breast_cancer(return_X_y=True)
    X = np.asarray(X, dtype=np.float32)

    sk_criterion = "entropy" if criterion == "log_loss" else criterion

    sgt = SGTClassifier(criterion=criterion, inner_max_depth=1)
    sgt.fit(X, y)

    dt = DecisionTreeClassifier(criterion=sk_criterion)
    dt.fit(X, y)

    np.testing.assert_array_equal(sgt.classes_, dt.classes_)
    np.testing.assert_array_equal(sgt.predict(X), dt.predict(X))
    np.testing.assert_allclose(
        sgt.predict_proba(X),
        dt.predict_proba(X),
        rtol=1e-6,
        atol=1e-6,
    )
