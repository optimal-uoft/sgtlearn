"""Shared validation contracts for random SGT forests."""

from __future__ import annotations

import numpy as np
import pytest

from sklearn.utils import check_random_state

from sgtlearn import RandomSGForestClassifier, RandomSGForestRegressor, SGTRegressor
from sgtlearn.ensemble._random_sgforest import _n_samples_bootstrap


@pytest.mark.parametrize(
    ("max_samples", "expected"),
    [(None, 10), (1, 1), (10, 10), (0.01, 1), (0.5, 5), (1.0, 10)],
)
def test_bootstrap_sample_count_boundaries(
    max_samples: float | None, expected: int
) -> None:
    assert _n_samples_bootstrap(10, max_samples) == expected


@pytest.mark.parametrize("max_samples", [0, 11, 0.0, 1.01])
def test_bootstrap_sample_count_rejects_invalid_values(
    max_samples: float,
) -> None:
    with pytest.raises(ValueError, match="max_samples"):
        _n_samples_bootstrap(10, max_samples)


def test_forest_rejects_invalid_estimator_and_bootstrap_configuration() -> None:
    X = np.array([[0.0], [1.0]])
    y = np.array([0, 1])
    with pytest.raises(ValueError, match="n_estimators"):
        RandomSGForestClassifier(n_estimators=0).fit(X, y)
    with pytest.raises(ValueError, match="bootstrap"):
        RandomSGForestClassifier(bootstrap=False, max_samples=1).fit(X, y)


def test_forest_predict_rejects_feature_count_mismatch() -> None:
    X = np.array([[0.0, 0.0], [1.0, 1.0], [0.1, 0.2], [0.9, 0.8]])
    y = np.array([0, 1, 0, 1])
    forest = RandomSGForestClassifier(
        n_estimators=1,
        bootstrap=False,
        max_features=None,
        tao_n_runs=0,
        random_state=0,
    ).fit(X, y)

    with pytest.raises(ValueError, match="expecting 2 features"):
        forest.predict(X[:, :1])


def test_bootstrap_keeps_samples_targets_and_weights_aligned() -> None:
    X = np.arange(16, dtype=np.float32).reshape(8, 2)
    y = np.array([0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0])
    sample_weight = np.arange(1.0, 9.0)
    forest_seed = 4
    forest = RandomSGForestRegressor(
        n_estimators=1,
        bootstrap=True,
        max_samples=6,
        random_state=forest_seed,
        max_depth=2,
        inner_max_depth=1,
        max_features=None,
        coordinate_descent_smart_init=False,
        tao_n_runs=0,
    ).fit(X, y, sample_weight=sample_weight)

    tree_seed = int(check_random_state(forest_seed).randint(np.iinfo(np.int32).max))
    indices = check_random_state(tree_seed).randint(0, len(X), 6, dtype=np.int32)
    manual = SGTRegressor(
        random_state=tree_seed,
        max_depth=2,
        inner_max_depth=1,
        max_features=None,
        coordinate_descent_smart_init=False,
        tao_n_runs=0,
    ).fit(X[indices], y[indices], sample_weight=sample_weight[indices])

    np.testing.assert_allclose(forest.estimators_[0].predict(X), manual.predict(X))


def test_classifier_probabilities_are_the_mean_in_shared_label_space() -> None:
    X = np.arange(40, dtype=np.float32).reshape(20, 2)
    forest_seed = 2
    seed_rng = check_random_state(forest_seed)
    tree_seeds = [
        int(seed_rng.randint(np.iinfo(np.int32).max)) for _ in range(3)
    ]
    bootstrap_rows = [
        check_random_state(seed).randint(0, len(X), 3, dtype=np.int32)
        for seed in tree_seeds
    ]
    rare_row = next(
        row
        for row in range(len(X))
        if any(row in rows for rows in bootstrap_rows)
        and any(row not in rows for rows in bootstrap_rows)
    )
    y = np.full(len(X), "common")
    y[rare_row] = "rare"
    forest = RandomSGForestClassifier(
        n_estimators=3,
        bootstrap=True,
        max_samples=3,
        max_features=None,
        tao_n_runs=0,
        random_state=forest_seed,
    ).fit(X, y)

    expected = np.mean([tree.predict_proba(X) for tree in forest.estimators_], axis=0)

    assert any(rare_row not in rows for rows in bootstrap_rows)
    np.testing.assert_array_equal(forest.classes_, ["common", "rare"])
    assert all(tree.predict_proba(X).shape == (len(X), 2) for tree in forest.estimators_)
    np.testing.assert_allclose(forest.predict_proba(X), expected)


def test_regression_forest_preserves_multioutput_shape() -> None:
    X = np.arange(24, dtype=np.float32).reshape(12, 2)
    y = np.column_stack([X[:, 0], 10.0 + X[:, 1]])
    forest = RandomSGForestRegressor(
        n_estimators=2,
        bootstrap=False,
        max_features=None,
        tao_n_runs=0,
        random_state=0,
    ).fit(X, y)

    assert forest.predict(X).shape == (12, 2)
