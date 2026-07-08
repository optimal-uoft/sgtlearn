"""Fidelity tests: ``CategoricalOneHotDiscretizer`` vs ``sklearn.tree.DecisionTreeClassifier``.

Trains both on a one-hot feature block and compares predictions across the same
inner-tree constraint grid used by the univariate discretizer tests.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
from Discretizers import CategoricalOneHotDiscretizer
from sklearn.tree import DecisionTreeClassifier

from tests.discretizer_grid import (
    MAX_DEPTH_VALUES,
    MAX_LEAF_VALUES,
    MIN_GAIN_VALUES,
    MIN_LEAF_VALUES,
    N_VALUES,
    NUM_CLASSES_VALUES,
)


def _make_onehot(
    n_samples: int,
    n_categories: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    cats = rng.integers(0, n_categories, size=n_samples)
    x = np.zeros((n_samples, n_categories), dtype=np.float32)
    x[np.arange(n_samples), cats] = 1.0
    y = rng.integers(0, n_categories, size=n_samples, dtype=np.uintp)
    return x, y


def classification_predict(
    disc: CategoricalOneHotDiscretizer, x: np.ndarray
) -> np.ndarray:
    bin_locs = disc.transform(x)
    bin_preds = disc.getBinPredictions()
    return np.asarray(bin_preds[bin_locs], dtype=np.uintp)


GRID = list(
    product(
        N_VALUES,
        NUM_CLASSES_VALUES,
        MIN_LEAF_VALUES,
        MIN_GAIN_VALUES,
        MAX_DEPTH_VALUES,
        MAX_LEAF_VALUES,
    )
)
IDS = [
    f"N={n}|C={c}|leaf={leaf}|gain={gain}|depth={depth}|max_leaf={max_leaf}"
    for n, c, leaf, gain, depth, max_leaf in GRID
]


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
@pytest.mark.parametrize(
    "n_samples,num_classes,min_leaf_size,min_gain_split,max_depth,max_leaf",
    GRID,
    ids=IDS,
)
def test_categorical_onehot_classification_discretizer_vs_sklearn_fidelity(
    criterion: str,
    n_samples: int,
    num_classes: int,
    min_leaf_size: int,
    min_gain_split: float,
    max_depth: int,
    max_leaf: int,
) -> None:
    """Predictions should track ``DecisionTreeClassifier`` on one-hot features."""
    rng = np.random.default_rng(12345)
    x, y = _make_onehot(n_samples, num_classes, rng)

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
    disc = CategoricalOneHotDiscretizer(criterion=criterion)
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
    disc = CategoricalOneHotDiscretizer(criterion="gini")
    disc.Train(x, features, y, n_cat, 1, 0.0, 0, 0)
    bins = disc.transform(x)
    for k in range(n_cat):
        rows = np.where(x[:, k] >= 0.5)[0]
        if rows.size == 0:
            continue
        assert np.unique(bins[rows]).size == 1


def test_categorical_onehot_respects_min_leaf_size() -> None:
    rng = np.random.default_rng(2026)
    n_cat = 3
    x, y = _make_onehot(120, n_cat, rng)
    features = np.arange(n_cat, dtype=np.uintp)
    min_leaf = 40
    disc = CategoricalOneHotDiscretizer(criterion="entropy")
    disc.Train(x, features, y, n_cat, min_leaf, 0.0, 0, 0)
    # ``numLeaves`` counts inner-tree bins only; the trailing NaN / catch-all
    # routing bin is not subject to ``min_samples_leaf``.
    inner_partitions = disc.getInSampleDiscretizations()[: disc.numLeaves]
    for part in inner_partitions:
        assert len(part) >= min_leaf


def test_categorical_onehot_in_sample_partition_covers_rows() -> None:
    rng = np.random.default_rng(2026)
    n_cat = 4
    n_samples = 200
    x, y = _make_onehot(n_samples, n_cat, rng)
    features = np.arange(n_cat, dtype=np.uintp)
    disc = CategoricalOneHotDiscretizer(criterion="gini")
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
    disc = CategoricalOneHotDiscretizer(criterion="gini")
    disc.Train(x, features, y, n_cat, 1, 0.0, 0, 0)
    disc_preds = classification_predict(disc, x)
    disc_acc = float(np.mean(disc_preds == y))

    assert sklearn_acc >= 0.99
    assert disc_acc >= 0.99
