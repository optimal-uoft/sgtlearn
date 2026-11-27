"""
Stress SGTRegressor with MAE (wavelet-tree inner splitter + merge/filter MAE) under aggressive settings.

Goal: native code completes without crashing and produces finite predictions.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from sgtlearn import SGTRegressor


def _run_mae_train_predict(
    x: np.ndarray,
    y: np.ndarray,
    *,
    inner_max_depth: int | None,
    inner_max_leaf_nodes: int | None,
) -> np.ndarray:
    """Fit a single-level MAE SGTRegressor and predict on training inputs."""
    model = SGTRegressor(
        criterion="absolute_error",
        max_depth=1,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        inner_min_samples_leaf=1,
        inner_min_impurity_decrease=0.0,
        inner_max_depth=inner_max_depth,
        inner_max_leaf_nodes=inner_max_leaf_nodes,
        tao_n_runs=0,
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model.fit(x, y)
    preds = model.predict(x)
    assert preds.shape == (x.shape[0],)
    assert np.all(np.isfinite(preds))
    return preds


@pytest.mark.parametrize(
    ("n_samples", "inner_max_depth", "inner_max_leaf_nodes"),
    [
        pytest.param(4096, None, 256, id="binding-leaf-cap"),
        pytest.param(4096, 12, None, id="deep-inner-tree"),
    ],
)
def test_mae_regression_stress_random_data(
    n_samples: int,
    inner_max_depth: int | None,
    inner_max_leaf_nodes: int | None,
) -> None:
    """Aggressive MAE settings on random data; pass if native training completes cleanly."""
    rng = np.random.default_rng(2026)
    x = rng.random((n_samples, 1), dtype=np.float32)
    y = rng.standard_normal(n_samples, dtype=np.float64).astype(np.float32)
    _run_mae_train_predict(
        x,
        y,
        inner_max_depth=inner_max_depth,
        inner_max_leaf_nodes=inner_max_leaf_nodes,
    )


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
    """Structured edge-case datasets (constants, duplicates) under MAE training."""
    _ = name
    _run_mae_train_predict(
        x,
        y,
        inner_max_depth=None,
        inner_max_leaf_nodes=500,
    )
