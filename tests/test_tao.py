"""Unit tests for :mod:`sgtlearn.tao`.

The defining contract of TAO refinement is that it never makes the tree worse
on the training data: routing rules are only swapped when they do not decrease
node-level accuracy. These tests assert that the post-refinement training
accuracy (and weighted accuracy) is at least the pre-refinement accuracy across
a few datasets and configurations.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import (
    load_breast_cancer,
    load_iris,
    make_classification,
    make_regression,
)
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
)

from sgtlearn import SGTClassifier, SGTRegressor, tao

pytest.importorskip("sklearn")


def _fit_tree(X: np.ndarray, y: np.ndarray, **kwargs) -> SGTClassifier:
    params = dict(
        criterion="gini",
        max_depth=4,
        min_samples_leaf=3,
        inner_max_depth=4,
        inner_max_leaf_nodes=16,
        random_state=42,
    )
    params.update(kwargs)
    return SGTClassifier(**params).fit(X, y)


@pytest.mark.parametrize(
    "loader", [load_breast_cancer, load_iris], ids=["breast_cancer", "iris"]
)
@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_tao_does_not_decrease_training_accuracy(loader, criterion: str) -> None:
    """Refined tree must classify the training set at least as well as before."""
    X, y = loader(return_X_y=True)
    X = np.asarray(X, dtype=np.float64)

    tree = _fit_tree(X, y, criterion=criterion)

    before = accuracy_score(y, tree.predict(X))
    returned = tao.optimize(tree, X, y)
    after = accuracy_score(y, tree.predict(X))

    assert returned is tree, "optimize should return the same tree instance"
    assert after >= before - 1e-9, (
        f"TAO decreased training accuracy ({before:.6f} -> {after:.6f}) "
        f"for criterion={criterion!r}"
    )


def test_tao_mutates_in_place() -> None:
    """optimize returns the same object and refines its underlying estimator."""
    X, y = make_classification(
        n_samples=300,
        n_features=12,
        n_informative=6,
        n_classes=3,
        random_state=0,
    )
    tree = _fit_tree(X, y)
    est_before = tree._est

    result = tao.optimize(tree, X, y)

    assert result is tree
    assert tree._est is est_before, "TAO should refine the existing estimator in place"


def test_tao_multiclass_does_not_decrease_accuracy() -> None:
    """A harder synthetic multiclass problem still respects the no-regression contract."""
    X, y = make_classification(
        n_samples=600,
        n_features=20,
        n_informative=10,
        n_redundant=4,
        n_classes=4,
        random_state=7,
    )
    tree = _fit_tree(X, y, max_depth=6, num_partitions=3)

    before = accuracy_score(y, tree.predict(X))
    tao.optimize(tree, X, y, n_runs=15)
    after = accuracy_score(y, tree.predict(X))

    assert after >= before - 1e-9, (
        f"TAO decreased multiclass training accuracy ({before:.6f} -> {after:.6f})"
    )


def test_tao_respects_sample_weight() -> None:
    """Weighted training accuracy must not decrease when sample_weight is supplied."""
    X, y = load_breast_cancer(return_X_y=True)
    X = np.asarray(X, dtype=np.float64)
    rng = np.random.default_rng(123)
    sample_weight = rng.uniform(0.5, 2.0, size=X.shape[0])

    tree = _fit_tree(X, y)
    tree.fit(X, y, sample_weight=sample_weight)

    def weighted_acc() -> float:
        correct = (tree.predict(X) == y).astype(np.float64)
        return float(np.average(correct, weights=sample_weight))

    before = weighted_acc()
    tao.optimize(tree, X, y, sample_weight=sample_weight)
    after = weighted_acc()

    assert after >= before - 1e-9, (
        f"TAO decreased weighted training accuracy ({before:.6f} -> {after:.6f})"
    )


def test_tao_rejects_unfitted_tree() -> None:
    from sklearn.exceptions import NotFittedError

    X, y = load_iris(return_X_y=True)
    tree = SGTClassifier()
    with pytest.raises(NotFittedError):
        tao.optimize(tree, X, y)


def test_tao_rejects_wrong_type() -> None:
    X, y = load_iris(return_X_y=True)
    with pytest.raises(TypeError):
        tao.optimize(object(), X, y)  # type: ignore[arg-type]


def _fit_regressor(X: np.ndarray, y: np.ndarray, **kwargs) -> SGTRegressor:
    params = dict(
        criterion="squared_error",
        max_depth=4,
        min_samples_leaf=3,
        inner_max_depth=4,
        inner_max_leaf_nodes=16,
        random_state=42,
    )
    params.update(kwargs)
    return SGTRegressor(**params).fit(X, y)


@pytest.mark.parametrize(
    ("criterion", "loss"),
    [
        ("squared_error", mean_squared_error),
        ("absolute_error", mean_absolute_error),
    ],
)
def test_tao_regression_does_not_increase_training_error(criterion, loss) -> None:
    """Refined regression tree must fit the training set at least as well.

    The guarantee is in the tree's own loss, so each criterion is checked
    against its matching metric (squared error -> MSE, MAE -> MAE).
    """
    X, y = make_regression(
        n_samples=400, n_features=10, n_informative=6, noise=5.0, random_state=0
    )
    X = np.asarray(X, dtype=np.float64)

    reg = _fit_regressor(X, y, criterion=criterion)

    before = loss(y, reg.predict(X))
    returned = tao.optimize(reg, X, y)
    after = loss(y, reg.predict(X))

    assert returned is reg, "optimize should return the same tree instance"
    assert after <= before + 1e-6 * (1.0 + before), (
        f"TAO increased training {loss.__name__} ({before:.6f} -> {after:.6f}) "
        f"for criterion={criterion!r}"
    )


def test_tao_regression_mutates_in_place() -> None:
    """optimize returns the same object and refines its underlying estimator."""
    X, y = make_regression(
        n_samples=300, n_features=12, n_informative=6, random_state=0
    )
    reg = _fit_regressor(X, y)
    est_before = reg._est

    result = tao.optimize(reg, X, y)

    assert result is reg
    assert reg._est is est_before, "TAO should refine the existing estimator in place"


def test_tao_regression_respects_sample_weight() -> None:
    """Weighted training MSE must not increase when sample_weight is supplied."""
    X, y = make_regression(
        n_samples=400, n_features=10, n_informative=6, noise=5.0, random_state=1
    )
    X = np.asarray(X, dtype=np.float64)
    rng = np.random.default_rng(123)
    sample_weight = rng.uniform(0.5, 2.0, size=X.shape[0])

    reg = _fit_regressor(X, y)
    reg.fit(X, y, sample_weight=sample_weight)

    def weighted_mse() -> float:
        err = (reg.predict(X) - y) ** 2
        return float(np.average(err, weights=sample_weight))

    before = weighted_mse()
    tao.optimize(reg, X, y, sample_weight=sample_weight)
    after = weighted_mse()

    assert after <= before + 1e-6 * (1.0 + before), (
        f"TAO increased weighted training MSE ({before:.6f} -> {after:.6f})"
    )


def test_tao_regression_rejects_unfitted_tree() -> None:
    from sklearn.exceptions import NotFittedError

    X, y = make_regression(n_samples=50, n_features=4, random_state=0)
    reg = SGTRegressor()
    with pytest.raises(NotFittedError):
        tao.optimize(reg, X, y)


def test_tao_regression_rejects_feature_mismatch() -> None:
    X, y = make_regression(n_samples=100, n_features=6, random_state=0)
    reg = _fit_regressor(X, y)
    with pytest.raises(ValueError):
        tao.optimize(reg, X[:, :-1], y)


def test_tao_rejects_feature_mismatch() -> None:
    X, y = load_iris(return_X_y=True)
    tree = _fit_tree(X, y)
    with pytest.raises(ValueError):
        tao.optimize(tree, X[:, :-1], y)
