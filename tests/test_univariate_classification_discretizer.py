"""Focused univariate-classification discretizer contracts and sklearn parity."""

import numpy as np
import pytest
from Discretizers import UnivariateClassificationDiscretizer
from sklearn.tree import DecisionTreeClassifier


def classification_predict(
    ud: UnivariateClassificationDiscretizer, x: np.ndarray
) -> np.ndarray:
    """Predict by mapping transform() bin indices to bin predictions."""
    bin_locs = ud.transform(x)
    bin_preds = ud.getBinPredictions()
    return np.asarray(bin_preds[bin_locs], dtype=np.uintp)


PARITY_CASES = [
    pytest.param("gini", 2, 1, 0.0, 0, 0, 1, id="gini-binary"),
    pytest.param("entropy", 3, 1, 0.0, 0, 0, 2, id="entropy-multioutput"),
    pytest.param("gini", 3, 1, 0.0, 4, 0, 1, id="depth-limited"),
    pytest.param("entropy", 3, 1, 0.0, 0, 8, 2, id="leaf-limited"),
]


@pytest.mark.parametrize(
    "criterion,num_classes,min_leaf_size,min_gain_split,max_depth,max_leaf,n_outputs",
    PARITY_CASES,
)
def test_univariate_classification_discretizer_vs_sklearn_fidelity(
    criterion: str,
    num_classes: int,
    min_leaf_size: int,
    min_gain_split: float,
    max_depth: int,
    max_leaf: int,
    n_outputs: int,
) -> None:
    """Predictions and leaf counts must match sklearn on synthetic univariate data."""
    rng = np.random.default_rng(12345)
    n_samples = 1000
    x = rng.random((n_samples, 1), dtype=np.float64)
    # Match sklearn tree builder input path, which internally works with float32.
    x32 = x.astype(np.float32, copy=False)
    y_size = n_samples if n_outputs == 1 else (n_samples, n_outputs)
    y = rng.integers(0, num_classes, size=y_size, dtype=np.uintp)

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
    assert (
        mismatch_frac <= 0.01
    ), f"prediction mismatch {mismatch_frac:.4%} exceeds 1% tolerance"


def test_univariate_classification_gain_threshold_blocks_known_split() -> None:
    x = np.arange(20, dtype=np.float32).reshape(-1, 1)
    y = np.repeat(np.array([0, 1], dtype=np.uintp), 10)
    features = np.array([0], dtype=np.uintp)

    split = UnivariateClassificationDiscretizer(criterion="gini")
    split.Train(x, features, y, 2, 1, 0.0, 0, 0)
    blocked = UnivariateClassificationDiscretizer(criterion="gini")
    blocked.Train(x, features, y, 2, 1, 1.0, 0, 0)

    assert split.numLeaves == 2
    assert blocked.numLeaves == 1


def test_univariate_classification_minimum_leaf_blocks_known_split() -> None:
    x = np.arange(12, dtype=np.float32).reshape(-1, 1)
    y = np.repeat(np.array([0, 1], dtype=np.uintp), 6)
    features = np.array([0], dtype=np.uintp)
    allowed = UnivariateClassificationDiscretizer(criterion="gini")
    allowed.Train(x, features, y, 2, 6, 0.0, 0, 0)
    blocked = UnivariateClassificationDiscretizer(criterion="gini")
    blocked.Train(x, features, y, 2, 7, 0.0, 0, 0)

    assert allowed.numLeaves == 2
    assert blocked.numLeaves == 1
