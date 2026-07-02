"""Fidelity tests: ``UnivariateClassificationDiscretizer`` vs ``sklearn.tree.DecisionTreeClassifier``.

Trains both on a single feature axis and asserts identical leaf predictions and
leaf counts across a grid of hyperparameters (see ``discretizer_grid``).
"""

import numpy as np
import pytest
from itertools import product
from Discretizers import UnivariateClassificationDiscretizer
from sklearn.tree import DecisionTreeClassifier

from tests.discretizer_grid import (
    MAX_DEPTH_VALUES,
    MAX_LEAF_VALUES,
    MIN_GAIN_VALUES,
    MIN_LEAF_VALUES,
    N_VALUES,
    NUM_CLASSES_VALUES,
)


def classification_predict(ud: UnivariateClassificationDiscretizer, x: np.ndarray) -> np.ndarray:
    """Predict by mapping transform() bin indices to bin predictions."""
    bin_locs = ud.transform(x)
    bin_preds = ud.getBinPredictions()
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
def test_univariate_classification_discretizer_vs_sklearn_fidelity(
    criterion: str,
    n_samples: int,
    num_classes: int,
    min_leaf_size: int,
    min_gain_split: float,
    max_depth: int,
    max_leaf: int,
) -> None:
    """Predictions and leaf counts must match sklearn on synthetic univariate data."""
    rng = np.random.default_rng(12345)
    x = rng.random((n_samples, 1), dtype=np.float64)
    # Match sklearn tree builder input path, which internally works with float32.
    x32 = x.astype(np.float32, copy=False)
    y = rng.integers(0, num_classes, size=n_samples, dtype=np.uintp)

    clf = DecisionTreeClassifier(
        criterion=criterion,
        splitter="best",
        min_samples_leaf=min_leaf_size,
        min_impurity_decrease=min_gain_split,
        max_depth=None if max_depth == 0 else max_depth,
        random_state=0,
        max_leaf_nodes=None if max_leaf == 0 else max_leaf,
    )
    clf.fit(x32, y)
    sklearn_preds = clf.predict(x32).astype(np.uintp, copy=False)
    ud = UnivariateClassificationDiscretizer(criterion=criterion)
    features = np.array([0], dtype=np.uintp)

    ud.Train(
        x32,
        features,
        y,
        num_classes,
        min_leaf_size,
        min_gain_split,
        max_depth,
        max_leaf,
    )

    ud_preds = classification_predict(ud, x)
    assert sklearn_preds.shape == ud_preds.shape
    # Leaf count is an exact invariant: both builders must grow the same-size tree.
    assert clf.get_n_leaves() == ud.numLeaves
    # Predictions need not be bit-identical to sklearn. On random labels, impurity
    # ties are dense and sklearn vs. the discretizer break equal-gain splits at
    # different thresholds, reassigning a handful of boundary points to an equally
    # good leaf. Allow a small mismatch fraction (matches the tolerance philosophy
    # of the regression discretizer test); exact fidelity holds only on signal data.
    mismatch_frac = np.mean(sklearn_preds != ud_preds)
    assert mismatch_frac <= 0.01, f"prediction mismatch {mismatch_frac:.4%} exceeds 1% tolerance"
