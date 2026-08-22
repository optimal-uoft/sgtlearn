"""Synthetic datasets for SGT demos."""

from __future__ import annotations

import numpy as np

__all__ = ["make_plus"]


def make_plus(
    n_samples: int = 1500,
    *,
    grid: int = 3,
    margin: float = 0.05,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the "Plus Sign" dataset.

    A ``grid x grid`` arrangement of cells where the center column and center
    row form a plus (+) of one class, and the remaining cells are the other
    class. The boundary is non-monotone in each feature -- the pattern an SGT's
    shape function captures with a Left/Right/Left split.

    Parameters
    ----------
    n_samples : int, default=1500
        Number of points to return.
    grid : int, default=3
        Number of cells per axis. The plus is the center row/column, so
        ``grid`` should be odd for a centered cross.
    margin : float, default=0.05
        Width of the empty gap kept around each cell border, so the classes
        are visually separated. Must be in ``[0, 0.5)``.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, 2)
        Feature matrix, with both features in ``[0, grid)``.
    y : ndarray of shape (n_samples,)
        Labels: ``1`` for the plus sign, ``0`` for the remaining cells.
    """
    if not 0.0 <= margin < 0.5:
        raise ValueError(f"margin must be in [0, 0.5); got {margin}")

    rng = np.random.default_rng(random_state)
    center = grid // 2

    X = np.empty((0, 2), dtype=float)
    # rejection-sample in batches until we have enough points off the borders
    while X.shape[0] < n_samples:
        batch = rng.uniform(0.0, grid, size=(n_samples * 2, 2))
        frac = batch - np.floor(batch)
        keep = ((frac > margin) & (frac < 1 - margin)).all(axis=1)
        X = np.vstack([X, batch[keep]])
    X = X[:n_samples]

    col = np.floor(X[:, 0]).astype(int)
    row = np.floor(X[:, 1]).astype(int)
    y = ((col == center) | (row == center)).astype(int)
    return X, y
