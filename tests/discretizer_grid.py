"""Shared hyperparameter grids for univariate discretizer sklearn fidelity tests.

Constants are imported by ``test_univariate_*`` modules to build Cartesian products
of sample size, tree depth, leaf limits, and gain thresholds.
"""

N_VALUES = [1000, 5000, 10000]
NUM_CLASSES_VALUES = [2, 3]
MIN_LEAF_VALUES = [1, 10]
MIN_GAIN_VALUES = [0.0, 1e-7]
MAX_DEPTH_VALUES = [0, 4]
MAX_LEAF_VALUES = [0, 100]
