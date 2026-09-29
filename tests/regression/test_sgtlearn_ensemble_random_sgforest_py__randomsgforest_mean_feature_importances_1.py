import sgtlearn.ensemble._random_sgforest as m
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from sgtlearn.ensemble import RandomSGForestClassifier, RandomSGForestRegressor


def _data(seed=0, n=80, p=3):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, p))
    y_cls = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    y_reg = X[:, 0] * 2.0 - X[:, 2] + 0.1 * rng.normal(size=n)
    return X, y_cls, y_reg


def _clf(n_estimators=4, **kw):
    X, y, _ = _data()
    params = dict(n_estimators=n_estimators, max_depth=2, tao_n_runs=0, random_state=0, n_jobs=1)
    params.update(kw)
    return RandomSGForestClassifier(**params).fit(X, y), X


def _reg(n_estimators=4, **kw):
    X, _, y = _data()
    params = dict(n_estimators=n_estimators, max_depth=2, tao_n_runs=0, random_state=0, n_jobs=1)
    params.update(kw)
    return RandomSGForestRegressor(**params).fit(X, y), X


def _tree_matrix(forest):
    return np.vstack([np.asarray(t.feature_importances_, dtype=float) for t in forest.estimators_])


def test_property_is_defined_on_base_class():
    assert isinstance(m.RandomSGForest.__dict__["mean_feature_importances_"], property)


def test_returns_1d_float_array_with_one_entry_per_feature():
    forest, X = _clf()
    out = forest.mean_feature_importances_
    assert isinstance(out, np.ndarray)
    assert out.ndim == 1
    assert out.shape == (X.shape[1],)
    assert np.issubdtype(out.dtype, np.floating)


def test_shape_matches_each_tree_importances():
    forest, _ = _clf()
    out = forest.mean_feature_importances_
    for t in forest.estimators_:
        assert np.asarray(t.feature_importances_).shape == out.shape


def test_equals_column_mean_of_per_tree_importances_classifier():
    forest, _ = _clf()
    np.testing.assert_allclose(forest.mean_feature_importances_, _tree_matrix(forest).mean(axis=0))


def test_equals_column_mean_of_per_tree_importances_regressor():
    forest, _ = _reg()
    np.testing.assert_allclose(forest.mean_feature_importances_, _tree_matrix(forest).mean(axis=0))


def test_values_are_finite():
    forest, _ = _clf()
    assert np.all(np.isfinite(forest.mean_feature_importances_))


def test_normalized_tree_importances_give_normalized_nonnegative_means():
    forest, _ = _clf(n_estimators=5)
    mat = _tree_matrix(forest)
    if not (np.all(mat >= 0) and np.allclose(mat.sum(axis=1), 1.0)):
        pytest.skip("per-tree importances not normalized for this fit")
    out = forest.mean_feature_importances_
    assert np.all(out >= -1e-12)
    assert out.sum() == pytest.approx(1.0, abs=1e-9)


def test_single_tree_equals_that_tree_importances():
    forest, _ = _clf(n_estimators=1)
    assert len(forest.estimators_) == 1
    np.testing.assert_allclose(
        forest.mean_feature_importances_,
        np.asarray(forest.estimators_[0].feature_importances_, dtype=float),
    )


def test_single_tree_regressor_equals_that_tree_importances():
    forest, _ = _reg(n_estimators=1)
    np.testing.assert_allclose(
        forest.mean_feature_importances_,
        np.asarray(forest.estimators_[0].feature_importances_, dtype=float),
    )


def test_repeated_reads_are_identical():
    forest, _ = _clf()
    a = np.array(forest.mean_feature_importances_, copy=True)
    b = forest.mean_feature_importances_
    np.testing.assert_array_equal(a, b)


def test_reading_does_not_change_tree_importances():
    forest, _ = _clf()
    before = _tree_matrix(forest).copy()
    n_before = len(forest.estimators_)
    _ = forest.mean_feature_importances_
    _ = forest.mean_feature_importances_
    assert len(forest.estimators_) == n_before
    np.testing.assert_array_equal(_tree_matrix(forest), before)


def test_reading_does_not_change_predictions():
    forest, X = _clf()
    before = forest.predict(X)
    _ = forest.mean_feature_importances_
    np.testing.assert_array_equal(forest.predict(X), before)


def test_consistent_with_std_feature_importance_shape():
    forest, _ = _clf()
    assert forest.std_feature_importance_.shape == forest.mean_feature_importances_.shape


def test_is_read_only():
    forest, _ = _clf()
    with pytest.raises(AttributeError):
        forest.mean_feature_importances_ = np.zeros(3)


def test_unfitted_classifier_raises():
    forest = RandomSGForestClassifier(n_estimators=2, tao_n_runs=0)
    with pytest.raises((NotFittedError, AttributeError)):
        forest.mean_feature_importances_


def test_unfitted_regressor_raises():
    forest = RandomSGForestRegressor(n_estimators=2, tao_n_runs=0)
    with pytest.raises((NotFittedError, AttributeError)):
        forest.mean_feature_importances_


def test_tao_refined_forest_raises():
    X, y, _ = _data()
    forest = RandomSGForestClassifier(
        n_estimators=2, max_depth=2, tao_n_runs=1, random_state=0, n_jobs=1
    ).fit(X, y)
    with pytest.raises((AttributeError, ValueError)):
        forest.mean_feature_importances_
