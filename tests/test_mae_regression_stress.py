"""
Stress UnivariateRegressionDiscretizer with MAE under aggressive settings.

Goal: native code (wavelet tree + tree build) completes without crashing.
Segfaults cannot be caught in Python; a passing run means the process survived.
"""

from __future__ import annotations

import numpy as np
import pytest
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
) -> int:
    """Train MAE discretizer, run ``transform``, and touch bin predictions (crash smoke test)."""
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
    return ud.numLeaves


@pytest.mark.parametrize(
    ("n_samples", "max_depth", "max_leaf"),
    [
        pytest.param(12000, 0, 0, id="unlimited-growth"),
        pytest.param(4096, 0, 256, id="binding-leaf-cap"),
    ],
)
def test_mae_regression_stress_random_data(
    n_samples: int,
    max_depth: int,
    max_leaf: int,
) -> None:
    """Aggressive MAE settings on random data; pass if the native extension completes without crashing."""
    rng = np.random.default_rng(2026)
    x = rng.random((n_samples, 1), dtype=np.float32)
    y = rng.standard_normal(n_samples, dtype=np.float64).astype(np.float32)
    num_leaves = _run_mae_train_predict(
        x,
        y,
        min_leaf_size=1,
        min_gain_split=0.0,
        max_depth=max_depth,
        max_leaf=max_leaf,
        criterion="absolute_error",
    )
    if max_leaf:
        assert num_leaves == max_leaf


@pytest.mark.parametrize(
    "name,x,y",
    [
        (
            "constant_y",
            np.linspace(0.0, 1.0, 800, dtype=np.float32).reshape(-1, 1),
            np.full(800, 3.14159, dtype=np.float32),
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
    ],
)
def test_mae_regression_stress_structured_data(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
) -> None:
    """Structured edge-case datasets (constants, duplicates, smooth curves) under MAE training."""
    _ = name
    _run_mae_train_predict(
        x,
        y,
        min_leaf_size=1,
        min_gain_split=0.0,
        max_depth=0,
        max_leaf=500,
        criterion="absolute_error",
    )
