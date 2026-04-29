import numpy as np
import pytest
from itertools import product
from collections import deque
from Discretizers import UnivariateDiscretizer
from sklearn.tree import DecisionTreeClassifier



def classification_predict(ud: UnivariateDiscretizer, x: np.ndarray) -> np.ndarray:
    """Predict by mapping transform() bin indices to bin predictions."""
    bin_locs = ud.transform(x)
    bin_preds = ud.getBinPredictions()
    return np.asarray(bin_preds[bin_locs], dtype=np.uintp)


N_VALUES = [1000, 5000, 10000]
NUM_CLASSES_VALUES = [2, 3]
MIN_LEAF_VALUES = [1, 10]
MIN_GAIN_VALUES = [0.0, 1e-7]
MAX_DEPTH_VALUES = [0, 4]
MAX_LEAF_VALUES = [0, 100]
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


@pytest.mark.parametrize(
    "n_samples,num_classes,min_leaf_size,min_gain_split,max_depth,max_leaf",
    GRID,
    ids=IDS,
)
def test_univariate_discretizer_vs_sklearn_fidelity(
    n_samples: int,
    num_classes: int,
    min_leaf_size: int,
    min_gain_split: float,
    max_depth: int,
    max_leaf: int,
) -> None:
    rng = np.random.default_rng(12345)
    x = rng.random((n_samples, 1), dtype=np.float64)
    # Match sklearn tree builder input path, which internally works with float32.
    x32 = x.astype(np.float32, copy=False)
    labels = rng.integers(0, num_classes, size=n_samples, dtype=np.uintp)

    clf = DecisionTreeClassifier(
        criterion="gini",
        splitter="best",
        min_samples_leaf=min_leaf_size,
        min_impurity_decrease=min_gain_split,
        max_depth=None if max_depth == 0 else max_depth,
        random_state=0,
        max_leaf_nodes=None if max_leaf == 0 else max_leaf,
    )
    clf.fit(x32, labels)
    sklearn_preds = clf.predict(x32).astype(np.uintp, copy=False)
    ud = UnivariateDiscretizer()
    features = np.array([0], dtype=np.uintp)

    ud.Train(
        x32,
        features,
        labels,
        num_classes,
        min_leaf_size,
        min_gain_split,
        max_depth,
        max_leaf,
    )

    ud_preds = classification_predict(ud, x)
    assert sklearn_preds.shape == ud_preds.shape
    np.testing.assert_array_equal(sklearn_preds, ud_preds)
    # assert clf.get_n_leaves() == ud.numLeaves
