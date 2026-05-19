"""
Fidelity: RandomSGForestClassifier vs sklearn RandomForestClassifier / SGTClassifier.

Uses Wisconsin breast cancer (binary). Training-set accuracy is compared to a
depth- and leaf-aligned ``RandomForestClassifier`` (same spirit as
``test_sgt_classifier_fidelity``). A structural check ensures one tree with
``bootstrap=False`` matches a standalone ``SGTClassifier`` that uses the same
per-tree ``random_state`` draw as the forest RNG.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state

pytest.importorskip("sklearn")

from sgtlearn import RandomSGForestClassifier, SGTClassifier


def _first_tree_random_state(forest_random_state: int) -> int:
    rng = check_random_state(forest_random_state)
    return int(rng.randint(np.iinfo(np.int32).max))


def _forest_tree_defaults() -> dict:
    """Every kwarg RandomSGForest passes to base trees via ``_tree_kwargs``."""
    return dict(
        num_partitions=2,
        max_depth=None,
        max_leaf_nodes=None,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        inner_max_depth=1,
        inner_max_leaf_nodes=32,
        inner_min_samples_leaf=1,
        inner_min_impurity_decrease=0.0,
        coordinate_descent_max_iters=20,
        coordinate_descent_patience=5,
        coordinate_descent_smart_init=True,
        max_features="sqrt",
    )


def _tree_hyperparams() -> dict:
    """Tuned tree settings for sklearn RF comparisons and structural fidelity checks."""
    return {
        **_forest_tree_defaults(),
        "criterion": "gini",
        "max_depth": 5,
        "min_samples_leaf": 5,
        "min_impurity_decrease": 1e-7,
        "inner_max_depth": 8,
        "inner_max_leaf_nodes": 48,
        "coordinate_descent_max_iters": 25,
        "coordinate_descent_patience": 6,
    }


def _standalone_sgt_classifier(
    forest_random_state: int, *, label_encoder, **tree_kw
) -> SGTClassifier:
    """SGTClassifier with the same hyperparameters and label encoding as one forest tree."""
    return SGTClassifier(
        random_state=_first_tree_random_state(forest_random_state),
        label_encoder=label_encoder,
        **tree_kw,
    )


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_random_sg_forest_train_accuracy_at_least_sklearn_random_forest(
    criterion: str,
) -> None:
    """In-sample accuracy must be >= a tuned ``RandomForestClassifier`` with matched limits."""
    X, y = load_breast_cancer(return_X_y=True)
    X = np.asarray(X, dtype=np.float32)

    sk_crit = "entropy" if criterion == "log_loss" else criterion
    kw = _tree_hyperparams()
    kw["criterion"] = criterion
    n_estimators = 17
    random_state = 42

    forest = RandomSGForestClassifier(
        n_estimators=n_estimators,
        bootstrap=True,
        random_state=random_state,
        **kw,
    )
    forest.fit(X, y)

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=sk_crit,
        max_depth=kw["max_depth"],
        min_samples_leaf=kw["min_samples_leaf"],
        min_impurity_decrease=kw["min_impurity_decrease"],
        max_features=kw["max_features"],
        bootstrap=True,
        random_state=random_state,
    )
    rf.fit(X, y)

    sgf_acc = accuracy_score(y, forest.predict(X))
    rf_acc = accuracy_score(y, rf.predict(X))

    assert sgf_acc >= rf_acc, (
        f"Training accuracy RandomSGForest ({sgf_acc:.6f}) should be >= sklearn "
        f"RandomForest ({rf_acc:.6f}) for criterion={criterion!r}"
    )


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_random_sg_forest_inner_depth_one_matches_sklearn_decision_tree(
    criterion: str,
) -> None:
    """One tree, no bootstrap, ``inner_max_depth=1``: same in-sample behavior as ``DecisionTreeClassifier()`` (cf. ``test_sgt_classifier_fidelity``)."""
    X, y = load_breast_cancer(return_X_y=True)
    X = np.asarray(X, dtype=np.float32)

    sk_crit = "entropy" if criterion == "log_loss" else criterion
    forest_rs = 7
    tree_kw = {
        **_forest_tree_defaults(),
        "criterion": criterion,
        "inner_max_depth": 1,
        "max_features": None,
    }

    forest = RandomSGForestClassifier(
        n_estimators=1,
        bootstrap=False,
        random_state=forest_rs,
        **tree_kw,
    )
    forest.fit(X, y)

    sgt = _standalone_sgt_classifier(
        forest_rs, label_encoder=forest._label_encoder_, **tree_kw
    )
    sgt.fit(X, y)
    np.testing.assert_array_equal(forest.predict(X), sgt.predict(X))
    np.testing.assert_allclose(
        forest.predict_proba(X),
        sgt.predict_proba(X),
    )

    dt = DecisionTreeClassifier(criterion=sk_crit)
    dt.fit(X, y)
    np.testing.assert_array_equal(forest.classes_, dt.classes_)
    np.testing.assert_array_equal(forest.predict(X), dt.predict(X))
    np.testing.assert_allclose(
        forest.predict_proba(X),
        dt.predict_proba(X),
    )


def test_random_sg_forest_single_tree_equals_standalone_sgt() -> None:
    """``n_estimators=1``, ``bootstrap=False`` matches ``SGTClassifier`` with the first RNG draw."""
    X, y = load_breast_cancer(return_X_y=True)
    X = np.asarray(X, dtype=np.float32)
    random_state = 99
    kw = _tree_hyperparams()

    forest = RandomSGForestClassifier(
        n_estimators=1,
        bootstrap=False,
        random_state=random_state,
        **kw,
    )
    forest.fit(X, y)

    sgt = _standalone_sgt_classifier(
        random_state, label_encoder=forest._label_encoder_, **kw
    )
    sgt.fit(X, y)

    est = forest.estimators_[0]
    np.testing.assert_array_equal(est.predict(X), sgt.predict(X))
    np.testing.assert_allclose(
        est.predict_proba(X), sgt.predict_proba(X)
    )
    np.testing.assert_array_equal(forest.predict(X), sgt.predict(X))
    np.testing.assert_allclose(
        forest.predict_proba(X),
        sgt.predict_proba(X),
    )


def test_random_sg_forest_trees_share_forest_label_encoder() -> None:
    """Each fitted tree must reuse the forest's fitted ``LabelEncoder`` instance."""
    X, y = load_breast_cancer(return_X_y=True)
    X = np.asarray(X, dtype=np.float32)
    tree_kw = {**_forest_tree_defaults(), "inner_max_depth": 2}
    forest = RandomSGForestClassifier(
        n_estimators=5,
        bootstrap=True,
        random_state=0,
        **tree_kw,
    )
    forest.fit(X, y)
    le = forest._label_encoder_
    for est in forest.estimators_:
        assert est._le is le


def test_random_sg_forest_parallel_fit_matches_sequential() -> None:
    """``n_jobs > 1`` must match sequential fitting (same seeds, same trees)."""
    X, y = load_breast_cancer(return_X_y=True)
    X = np.asarray(X, dtype=np.float32)
    kw = _tree_hyperparams()
    kw["criterion"] = "gini"
    n_estimators = 9
    random_state = 3

    seq = RandomSGForestClassifier(
        n_estimators=n_estimators,
        bootstrap=True,
        random_state=random_state,
        n_jobs=1,
        **kw,
    )
    seq.fit(X, y)

    par = RandomSGForestClassifier(
        n_estimators=n_estimators,
        bootstrap=True,
        random_state=random_state,
        n_jobs=2,
        **kw,
    )
    par.fit(X, y)

    np.testing.assert_array_equal(seq.predict(X), par.predict(X))
    np.testing.assert_allclose(
        seq.predict_proba(X), par.predict_proba(X)
    )
