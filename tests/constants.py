"""Shared constants for unit tests."""

from __future__ import annotations

# Disable automatic TAO in fit() unless a test explicitly exercises it.
TEST_TAO_N_RUNS = 0


def algorithm_params(**overrides):
    """Explicit pre-tuning setup for tests of growth, routing, and CART fidelity.

    Those tests need unrestricted tiny-data fits; empirical production defaults
    intentionally regularize them. Tests of defaults use the constructors directly.
    """
    return {
        "num_partitions": 2,
        "max_depth": None,
        "min_samples_leaf": 1,
        "min_impurity_decrease": 0.0,
        "inner_max_leaf_nodes": 8,
        "inner_min_samples_leaf": 1,
        "inner_min_impurity_decrease": 0.0,
        "coordinate_descent_max_iters": 20,
        "pairwise_candidates": 0,
        "tao_n_runs": 10,
        "tao_lambda": 0.0,
        "tao_pair_scale": 1.1,
    } | overrides
