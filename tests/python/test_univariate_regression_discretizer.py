# import numpy as np
# import pytest
# from itertools import product
# from Discretizers import UnivariateRegressionDiscretizer
# from sklearn.tree import DecisionTreeRegressor
#
#
#
# def regression_predict(ud: UnivariateRegressionDiscretizer, x: np.ndarray) -> np.ndarray:
#     bin_locs = ud.transform(x)
#     bin_preds = ud.getBinPredictions()
#     return np.asarray(bin_preds[bin_locs], dtype=np.float32)
#
#
# N_VALUES = [1000, 5000]
# MIN_LEAF_VALUES = [1, 10]
# MIN_GAIN_VALUES = [0.0, 1e-7]
# MAX_DEPTH_VALUES = [0, 4]
# MAX_LEAF_VALUES = [0, 100]
# GRID = list(
#     product(
#         N_VALUES,
#         MIN_LEAF_VALUES,
#         MIN_GAIN_VALUES,
#         MAX_DEPTH_VALUES,
#         MAX_LEAF_VALUES,
#     )
# )
#
#
# @pytest.mark.parametrize(
#     "n_samples,min_leaf_size,min_gain_split,max_depth,max_leaf",
#     GRID,
# )
# def test_univariate_regression_discretizer_vs_sklearn_fidelity(
#     n_samples: int,
#     min_leaf_size: int,
#     min_gain_split: float,
#     max_depth: int,
#     max_leaf: int,
# ) -> None:
#     rng = np.random.default_rng(12345)
#     x = rng.random((n_samples, 1), dtype=np.float64)
#     x32 = x.astype(np.float32, copy=False)
#     y = rng.standard_normal(n_samples).astype(np.float32, copy=False)
#
#     reg = DecisionTreeRegressor(
#         criterion="squared_error",
#         splitter="best",
#         min_samples_leaf=min_leaf_size,
#         min_impurity_decrease=min_gain_split,
#         max_depth=None if max_depth == 0 else max_depth,
#         random_state=0,
#         max_leaf_nodes=None if max_leaf == 0 else max_leaf,
#     )
#     reg.fit(x32, y)
#     sklearn_preds = reg.predict(x32).astype(np.float32, copy=False)
#
#     ud = UnivariateRegressionDiscretizer()
#     features = np.array([0], dtype=np.uintp)
#     ud.Train(
#         x32,
#         features,
#         y,
#         min_leaf_size,
#         min_gain_split,
#         max_depth,
#         max_leaf,
#     )
#
#     ud_preds = regression_predict(ud, x32)
#     assert sklearn_preds.shape == ud_preds.shape
#     np.testing.assert_allclose(sklearn_preds, ud_preds, rtol=1e-5, atol=1e-5)
#     assert reg.get_n_leaves() == ud.numLeaves
