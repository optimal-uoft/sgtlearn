"""Shared single- versus multi-output parameters for fidelity tests."""

import pytest

# ``1`` is scalar-y behavior; ``2`` exercises the shared multi-output branch.
N_OUTPUTS_VALUES = [1, 2]


def n_outputs_params():
    """``n_outputs`` params covering single- and multi-output discretization."""
    return [pytest.param(n, id=f"n_outputs={n}") for n in N_OUTPUTS_VALUES]
