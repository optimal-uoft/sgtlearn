"""Public contracts for bundled synthetic datasets."""

from __future__ import annotations

import numpy as np
import pytest

from sgtlearn.datasets import make_plus


def test_make_plus_matches_documented_geometry() -> None:
    n_samples = 200
    grid = 5
    margin = 0.1
    X, y = make_plus(n_samples=n_samples, grid=grid, margin=margin, random_state=7)

    assert X.shape == (n_samples, 2)
    assert y.shape == (n_samples,)
    assert set(np.unique(y)) <= {0, 1}
    assert np.all((0 <= X) & (X < grid))
    fractions = X - np.floor(X)
    assert np.all((fractions > margin) & (fractions < 1 - margin))
    cells = np.floor(X).astype(int)
    expected = ((cells[:, 0] == grid // 2) | (cells[:, 1] == grid // 2)).astype(int)
    np.testing.assert_array_equal(y, expected)


def test_make_plus_is_deterministic_for_seed() -> None:
    first = make_plus(n_samples=25, random_state=42)
    second = make_plus(n_samples=25, random_state=42)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


@pytest.mark.parametrize("margin", [-0.01, 0.5, 1.0])
def test_make_plus_rejects_invalid_margin(margin: float) -> None:
    with pytest.raises(ValueError, match="margin"):
        make_plus(margin=margin)


def test_make_plus_allows_empty_dataset() -> None:
    X, y = make_plus(n_samples=0, random_state=0)
    assert X.shape == (0, 2)
    assert y.shape == (0,)
