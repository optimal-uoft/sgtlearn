"""
Stress UnivariateRegressionDiscretizer with MAE under aggressive settings.

Goal: native code (wavelet tree + tree build) completes without crashing.
Segfaults cannot be caught in Python; a passing run means the process survived.
"""

from __future__ import annotations

import numpy as np
import pytest
from itertools import product

from Discretizers import UnivariateRegressionDiscretizer


def _run_mae_train_predict(
    x: np.ndarray,
    y: np.ndarray,
    *,
    min_leaf_size: int,
    min_gain_split: float,
    max_depth: int,
    max_leaf: int,
    criterion: str,
) -> None:
    ud = UnivariateRegressionDiscretizer(criterion=criterion)
    features = np.array([0], dtype=np.uintp)
    ud.Train(
        x.astype(np.float32, copy=False),
        features,
        y.astype(np.float32, copy=False),
        min_leaf_size,
        min_gain_split,
        max_depth,
        max_leaf,
    )
    assert ud.numLeaves > 0
    bins = ud.transform(x.astype(np.float32, copy=False))
    preds = ud.getBinPredictions()
    n = x.shape[0]
    assert int(bins.size) == n
    assert len(preds) == ud.numLeaves
    _ = np.asarray(preds)[bins]


# More aggressive than discretizer_grid: smallest leaf, zero gain floor, deep/wide trees.
STRESS_GRID = list(
    product(
        [512, 4096, 12000],  # n_samples
        [1],  # min_leaf — maximum splitting
        [0.0],  # allow any split gain
        [0, 12],  # 0 = unlimited depth
        [0, 256, 2000],  # 0 = unlimited leaves in our builder; large caps stress queues
    )
)
STRESS_IDS = [
    f"n={n}|leaf={leaf}|gain={gain}|depth={depth}|max_leaf={ml}"
    for n, leaf, gain, depth, ml in STRESS_GRID
]


@pytest.mark.parametrize("criterion", ["mae", "absolute_error"])
@pytest.mark.parametrize(
    "n_samples,min_leaf_size,min_gain_split,max_depth,max_leaf",
    STRESS_GRID,
    ids=STRESS_IDS,
)
def test_mae_regression_stress_random_data(
    criterion: str,
    n_samples: int,
    min_leaf_size: int,
    min_gain_split: float,
    max_depth: int,
    max_leaf: int,
) -> None:
    rng = np.random.default_rng(2026)
    x = rng.random((n_samples, 1), dtype=np.float32)
    y = rng.standard_normal(n_samples, dtype=np.float64).astype(np.float32)
    _run_mae_train_predict(
        x,
        y,
        min_leaf_size=min_leaf_size,
        min_gain_split=min_gain_split,
        max_depth=max_depth,
        max_leaf=max_leaf,
        criterion=criterion,
    )


@pytest.mark.parametrize("criterion", ["mae", "absolute_error"])
@pytest.mark.parametrize(
    "name,x,y",
    [
        (
            "constant_y",
            np.linspace(0.0, 1.0, 800, dtype=np.float32).reshape(-1, 1),
            np.full(800, 3.14159, dtype=np.float32),
        ),
        (
            "two_unique_y",
            np.linspace(0.0, 1.0, 600, dtype=np.float32).reshape(-1, 1),
            np.repeat(np.array([0.0, 1.0], dtype=np.float32), 300),
        ),
        (
            "duplicate_x_runs",
            np.concatenate(
                [
                    np.full((400, 1), 0.5, dtype=np.float32),
                    np.linspace(0.0, 1.0, 400, dtype=np.float32).reshape(-1, 1),
                ],
                axis=0,
            ),
            np.arange(800, dtype=np.float32) * 0.01,
        ),
        (
            "sorted_strictly_increasing_x",
            np.linspace(0.0, 1.0, 2000, dtype=np.float32).reshape(-1, 1),
            np.sin(np.linspace(0.0, 6.28, 2000)).astype(np.float32),
        ),
    ],
)
def test_mae_regression_stress_structured_data(
    criterion: str,
    name: str,
    x: np.ndarray,
    y: np.ndarray,
) -> None:
    _ = name
    _run_mae_train_predict(
        x,
        y,
        min_leaf_size=1,
        min_gain_split=0.0,
        max_depth=0,
        max_leaf=500,
        criterion=criterion,
    )
