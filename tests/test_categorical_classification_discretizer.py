"""Focused categorical-classification discretizer contracts and sklearn parity."""

from __future__ import annotations

import numpy as np
import pytest
from Discretizers import CategoricalClassificationDiscretizer
from sklearn.tree import DecisionTreeClassifier


def _make_onehot(
    n_samples: int,
    n_categories: int,
    rng: np.random.Generator,
    n_outputs: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    cats = rng.integers(0, n_categories, size=n_samples)
    x = np.zeros((n_samples, n_categories), dtype=np.float32)
    x[np.arange(n_samples), cats] = 1.0
    y_size = n_samples if n_outputs == 1 else (n_samples, n_outputs)
    y = rng.integers(0, n_categories, size=y_size, dtype=np.uintp)
    return x, y


def classification_predict(
    disc: CategoricalClassificationDiscretizer, x: np.ndarray
) -> np.ndarray:
    bin_locs = disc.transform(x)
    bin_preds = disc.getBinPredictions()
    return np.asarray(bin_preds[bin_locs], dtype=np.uintp)


PARITY_CASES = [
    pytest.param("gini", 2, 1, 0.0, 0, 0, 1, id="gini-binary"),
    pytest.param("entropy", 3, 1, 0.0, 0, 0, 2, id="entropy-multioutput"),
    pytest.param("gini", 4, 1, 0.0, 2, 0, 1, id="depth-limited"),
    pytest.param("entropy", 4, 1, 0.0, 0, 2, 2, id="leaf-limited"),
]


@pytest.mark.parametrize(
    "criterion,num_classes,min_leaf_size,min_gain_split,max_depth,max_leaf,n_outputs",
    PARITY_CASES,
)
def test_categorical_onehot_classification_discretizer_vs_sklearn_fidelity(
    criterion: str,
    num_classes: int,
    min_leaf_size: int,
    min_gain_split: float,
    max_depth: int,
    max_leaf: int,
    n_outputs: int,
) -> None:
    """Predictions should track ``DecisionTreeClassifier`` on one-hot features."""
    rng = np.random.default_rng(12345)
    x, y = _make_onehot(1000, num_classes, rng, n_outputs=n_outputs)

    clf = DecisionTreeClassifier(
        criterion=criterion,
        splitter="best",
        min_samples_leaf=min_leaf_size,
        min_impurity_decrease=min_gain_split,
        max_depth=None if max_depth == 0 else max_depth,
        random_state=0,
        max_leaf_nodes=None if max_leaf == 0 else max_leaf,
    )
    clf.fit(x, y)
    sklearn_preds = clf.predict(x).astype(np.uintp, copy=False)

    features = np.arange(num_classes, dtype=np.uintp)
    disc = CategoricalClassificationDiscretizer(criterion=criterion)
    disc.Train(
        x,
        features,
        y,
        num_classes,
        min_leaf_size,
        min_gain_split,
        max_depth,
        max_leaf,
    )

    disc_preds = classification_predict(disc, x)
    assert sklearn_preds.shape == disc_preds.shape
    assert disc.numLeaves > 0
    assert clf.get_n_leaves() == disc.numLeaves
    np.testing.assert_array_equal(sklearn_preds, disc_preds)


def test_categorical_onehot_active_category_maps_to_single_leaf() -> None:
    """Each active one-hot column routes all its samples to one inner bin."""
    rng = np.random.default_rng(2026)
    n_cat = 4
    x, y = _make_onehot(400, n_cat, rng)
    features = np.arange(n_cat, dtype=np.uintp)
    disc = CategoricalClassificationDiscretizer(criterion="gini")
    disc.Train(x, features, y, n_cat, 1, 0.0, 0, 0)
    bins = disc.transform(x)
    for k in range(n_cat):
        rows = np.where(x[:, k] >= 0.5)[0]
        if rows.size == 0:
            continue
        assert np.unique(bins[rows]).size == 1


def test_categorical_onehot_respects_min_leaf_size() -> None:
    n_cat = 3
    x = np.repeat(np.eye(n_cat, dtype=np.float32), 4, axis=0)
    y = np.repeat(np.array([0, 1, 0], dtype=np.uintp), 4)
    features = np.arange(n_cat, dtype=np.uintp)
    allowed = CategoricalClassificationDiscretizer(criterion="entropy")
    allowed.Train(x, features, y, 2, 4, 0.0, 0, 0)
    blocked = CategoricalClassificationDiscretizer(criterion="entropy")
    blocked.Train(x, features, y, 2, 5, 0.0, 0, 0)

    assert allowed.numLeaves == 2
    assert blocked.numLeaves == 1


def test_categorical_classification_gain_threshold_blocks_known_split() -> None:
    x = np.repeat(np.eye(3, dtype=np.float32), 4, axis=0)
    y = np.repeat(np.array([0, 1, 0], dtype=np.uintp), 4)
    features = np.arange(3, dtype=np.uintp)
    split = CategoricalClassificationDiscretizer(criterion="gini")
    split.Train(x, features, y, 2, 1, 0.0, 0, 0)
    blocked = CategoricalClassificationDiscretizer(criterion="gini")
    blocked.Train(x, features, y, 2, 1, 1.0, 0, 0)

    assert split.numLeaves == 2
    assert blocked.numLeaves == 0


def test_categorical_onehot_in_sample_partition_covers_rows() -> None:
    rng = np.random.default_rng(2026)
    n_cat = 4
    n_samples = 200
    x, y = _make_onehot(n_samples, n_cat, rng)
    features = np.arange(n_cat, dtype=np.uintp)
    disc = CategoricalClassificationDiscretizer(criterion="gini")
    disc.Train(x, features, y, n_cat, 1, 0.0, 0, 0)
    seen = np.zeros(n_samples, dtype=bool)
    for part in disc.getInSampleDiscretizations()[: disc.numLeaves]:
        seen[part] = True
    assert seen.all()


def test_categorical_onehot_pure_signal_matches_sklearn() -> None:
    """When labels equal category id, both models should classify almost perfectly."""
    rng = np.random.default_rng(2026)
    n_cat = 4
    n_samples = 400
    cats = rng.integers(0, n_cat, size=n_samples)
    x = np.zeros((n_samples, n_cat), dtype=np.float32)
    x[np.arange(n_samples), cats] = 1.0
    y = cats.astype(np.uintp)

    clf = DecisionTreeClassifier(criterion="gini", random_state=0)
    clf.fit(x, y)
    sklearn_acc = float(np.mean(clf.predict(x) == y))

    features = np.arange(n_cat, dtype=np.uintp)
    disc = CategoricalClassificationDiscretizer(criterion="gini")
    disc.Train(x, features, y, n_cat, 1, 0.0, 0, 0)
    disc_preds = classification_predict(disc, x)
    disc_acc = float(np.mean(disc_preds == y))

    assert sklearn_acc >= 0.99
    assert disc_acc >= 0.99
