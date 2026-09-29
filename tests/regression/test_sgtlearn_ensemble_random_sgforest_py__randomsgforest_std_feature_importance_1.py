import warnings

import sgtlearn.ensemble._random_sgforest as m
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from sgtlearn.ensemble import RandomSGForestClassifier, RandomSGForestRegressor


def _clf_data(seed=0, n=80):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    return X, y


def _reg_data(seed=0, n=80):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, 3))
    y = X[:, 0] * 2.0 + X[:, 1]
    return X, y


def _fit_clf(n_estimators=5, X=None, y=None, **kw):
    if X is None:
        X, y = _clf_data()
    f = RandomSGForestClassifier(
        n_estimators=n_estimators, max_depth=2, tao_n_runs=0, random_state=0, **kw
    )
    return f.fit(X, y)


def test_std_matches_population_std_of_per_tree_matrix():
    f = _fit_clf()
    mat = f._tree_feature_importances_matrix()
    std = f.std_feature_importance_
    np.testing.assert_allclose(std, np.std(mat, axis=0, ddof=0))


def test_std_shape_matches_mean_feature_importances():
    f = _fit_clf()
    std = f.std_feature_importance_
    assert isinstance(std, np.ndarray)
    assert std.ndim == 1
    assert std.shape == f.mean_feature_importances_.shape
    assert std.shape == (3,)


def test_std_dtype_is_float():
    f = _fit_clf()
    assert np.issubdtype(f.std_feature_importance_.dtype, np.floating)


def test_std_values_are_finite_and_non_negative():
    f = _fit_clf()
    std = f.std_feature_importance_
    assert np.all(np.isfinite(std))
    assert np.all(std >= 0)


def test_repeated_access_returns_equal_arrays():
    f = _fit_clf()
    a = f.std_feature_importance_
    b = f.std_feature_importance_
    np.testing.assert_array_equal(a, b)


def test_access_has_no_side_effect_on_mean_importances():
    f = _fit_clf()
    before = np.array(f.mean_feature_importances_, copy=True)
    _ = f.std_feature_importance_
    np.testing.assert_array_equal(f.mean_feature_importances_, before)


def test_single_estimator_gives_all_zeros():
    f = _fit_clf(n_estimators=1)
    std = f.std_feature_importance_
    assert std.shape == (3,)
    assert np.all(std == 0.0)


def test_unused_constant_feature_has_zero_std():
    X, y = _clf_data()
    X = np.column_stack([X, np.ones(X.shape[0])])
    f = _fit_clf(X=X, y=y)
    std = f.std_feature_importance_
    assert std.shape == (4,)
    assert std[3] == 0.0


def test_access_emits_no_warnings_and_no_nan():
    f = _fit_clf()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with np.errstate(all="raise"):
            std = f.std_feature_importance_
    assert not np.any(np.isnan(std))


def test_regressor_std_matches_population_std():
    X, y = _reg_data()
    f = RandomSGForestRegressor(
        n_estimators=4, max_depth=2, tao_n_runs=0, random_state=0
    ).fit(X, y)
    std = f.std_feature_importance_
    np.testing.assert_allclose(
        std, np.std(f._tree_feature_importances_matrix(), axis=0, ddof=0)
    )
    assert std.shape == f.mean_feature_importances_.shape


def test_feature_dict_length_is_number_of_logical_features():
    X, y = _clf_data()
    f = RandomSGForestClassifier(
        n_estimators=4, max_depth=2, tao_n_runs=0, random_state=0
    ).fit(X, y, feature_dict={"g0": [0, 1], "g1": [2]})
    std = f.std_feature_importance_
    assert std.shape == f.mean_feature_importances_.shape
    assert std.shape == (2,)


def test_unfitted_forest_raises_not_fitted():
    f = RandomSGForestClassifier(n_estimators=3, tao_n_runs=0)
    with pytest.raises((NotFittedError, AttributeError)):
        _ = f.std_feature_importance_


def test_tao_fitted_forest_raises_unavailable_error():
    X, y = _clf_data()
    f = RandomSGForestClassifier(
        n_estimators=2, max_depth=2, tao_n_runs=1, random_state=0
    ).fit(X, y)
    with pytest.raises(AttributeError, match="(?i)importance"):
        _ = f.std_feature_importance_
