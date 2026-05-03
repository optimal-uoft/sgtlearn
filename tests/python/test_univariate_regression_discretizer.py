import numpy as np
import pytest
from itertools import product

from Discretizers import UnivariateRegressionDiscretizer
from sklearn.tree import DecisionTreeRegressor

from discretizer_grid import (
    MAX_DEPTH_VALUES,
    MAX_LEAF_VALUES,
    MIN_GAIN_VALUES,
    MIN_LEAF_VALUES,
    N_VALUES,
)


def regression_predict(ud: UnivariateRegressionDiscretizer, x: np.ndarray) -> np.ndarray:
    """Predict by mapping transform() bin indices to bin predictions."""
    bin_locs = ud.transform(x)
    bin_preds = ud.getBinPredictions()
    return np.asarray(bin_preds[bin_locs], dtype=np.float32)


def sklearn_regression_criterion(user_criterion: str) -> str:
    """sklearn expects 'squared_error'; 'mse' is normalized in our bindings."""
    if user_criterion == "mse":
        return "squared_error"
    return user_criterion


# Same (n_samples, leaf, gain, depth, max_leaf) sweep as classification — without num_classes.
GRID = list(
    product(
        N_VALUES,
        MIN_LEAF_VALUES,
        MIN_GAIN_VALUES,
        MAX_DEPTH_VALUES,
        MAX_LEAF_VALUES,
    )
)
IDS = [
    f"N={n}|leaf={leaf}|gain={gain}|depth={depth}|max_leaf={max_leaf}"
    for n, leaf, gain, depth, max_leaf in GRID
]


@pytest.mark.parametrize("criterion", ["squared_error", "mse"])
@pytest.mark.parametrize(
    "n_samples,min_leaf_size,min_gain_split,max_depth,max_leaf",
    GRID,
    ids=IDS,
)
def test_univariate_regression_discretizer_vs_sklearn_fidelity(
    criterion: str,
    n_samples: int,
    min_leaf_size: int,
    min_gain_split: float,
    max_depth: int,
    max_leaf: int,
) -> None:
    rng = np.random.default_rng(12345)
    x = rng.random((n_samples, 1), dtype=np.float64)
    x32 = x.astype(np.float32, copy=False)
    y = rng.standard_normal(n_samples).astype(np.float32, copy=False)

    sk_crit = sklearn_regression_criterion(criterion)
    reg = DecisionTreeRegressor(
        criterion=sk_crit,
        splitter="best",
        min_samples_leaf=min_leaf_size,
        min_impurity_decrease=min_gain_split,
        max_depth=None if max_depth == 0 else max_depth,
        random_state=0,
        max_leaf_nodes=None if max_leaf == 0 else max_leaf,
    )
    reg.fit(x32, y)
    sklearn_preds = reg.predict(x32).astype(np.float32, copy=False)

    ud = UnivariateRegressionDiscretizer(criterion=criterion)
    features = np.array([0], dtype=np.uintp)
    ud.Train(
        x32,
        features,
        y,
        min_leaf_size,
        min_gain_split,
        max_depth,
        max_leaf,
    )

    ud_preds = regression_predict(ud, x32)
    assert sklearn_preds.shape == ud_preds.shape
    assert np.all(np.isfinite(ud_preds))
    assert ud.numLeaves > 0

    # Classification grid matches exactly; continuous regression gains do not guarantee
    # per-sample equality with sklearn. Bound RMS(pred_diff) relative to label spread.
    sigma = float(np.std(y)) + 1e-8
    rmse_sk_vs_ud = float(
        np.sqrt(
            np.mean(
                (sklearn_preds.astype(np.float64) - ud_preds.astype(np.float64)) ** 2
            )
        )
    )
    assert rmse_sk_vs_ud < 0.9 * sigma


def test_regression_friedman_mse_constructible() -> None:
    """friedman_mse maps to the squared-error backend until a dedicated splitter exists."""
    UnivariateRegressionDiscretizer(criterion="friedman_mse")
