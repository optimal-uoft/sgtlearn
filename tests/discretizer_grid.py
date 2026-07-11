"""Shared hyperparameter grids for univariate discretizer sklearn fidelity tests.

Constants are imported by ``test_univariate_*`` modules to build Cartesian products
of sample size, tree depth, leaf limits, and gain thresholds.
"""

import pytest

# sklearn fidelity is a correctness property, not size-dependent: one N is enough.
N_VALUES = [5000]
NUM_CLASSES_VALUES = [2, 3]
# One-hot categorical block width (number of binary columns / categories).
NUM_CATEGORIES_VALUES = [2, 3, 4]
MIN_LEAF_VALUES = [1, 10]
MIN_GAIN_VALUES = [0.0, 1e-7]
MAX_DEPTH_VALUES = [0, 4]
MAX_LEAF_VALUES = [0, 100]
# ``1`` is current scalar-y behavior; ``2``/``3`` xfail until multioutput lands.
N_OUTPUTS_VALUES = [1, 2, 3]


def n_outputs_params():
    """``n_outputs`` params; multioutput cases xfail until the feature is implemented."""
    return [
        pytest.param(
            n,
            id=f"n_outputs={n}",
            marks=(
                ()
                if n == 1
                else pytest.mark.xfail(
                    reason="multioutput support not yet implemented",
                    strict=True,
                )
            ),
        )
        for n in N_OUTPUTS_VALUES
    ]
