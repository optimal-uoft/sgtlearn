import sgtlearn.tao as m
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

try:
    from sgtlearn import SGTClassifier, SGTRegressor
except ImportError:  # pragma: no cover
    SGTClassifier = m.SGTClassifier
    SGTRegressor = m.SGTRegressor
from sgtlearn.ensemble import RandomSGForestClassifier, RandomSGForestRegressor


def _clf_data():
    rng = np.random.RandomState(0)
    X = rng.rand(60, 3)
    y = (X[:, 0] > 0.5).astype(int)
    return X, y


def _reg_data():
    rng = np.random.RandomState(1)
    X = rng.rand(60, 3)
    y = 2.0 * X[:, 0] + X[:, 1]
    return X, y


def _fit_clf():
    return SGTClassifier().fit(*_clf_data())


def _fit_reg():
    return SGTRegressor().fit(*_reg_data())


def _fit_forest_clf():
    return RandomSGForestClassifier(n_estimators=3, random_state=0).fit(*_clf_data())


def _fit_forest_reg():
    return RandomSGForestRegressor(n_estimators=3, random_state=0).fit(*_reg_data())


@pytest.fixture(scope="module")
def fitted_clf():
    return _fit_clf()


@pytest.fixture(scope="module")
def fitted_reg():
    return _fit_reg()


@pytest.fixture(scope="module")
def fitted_forest_clf():
    return _fit_forest_clf()


@pytest.fixture(scope="module")
def fitted_forest_reg():
    return _fit_forest_reg()


def test_single_classifier_returns_one_element_list_with_same_object(fitted_clf):
    out = m._tao_targets(fitted_clf)
    assert isinstance(out, list)
    assert len(out) == 1
    assert out[0] is fitted_clf


def test_single_regressor_returns_one_element_list_with_same_object(fitted_reg):
    out = m._tao_targets(fitted_reg)
    assert isinstance(out, list)
    assert len(out) == 1
    assert out[0] is fitted_reg


def test_subclass_of_tree_is_accepted():
    class MyReg(SGTRegressor):
        pass

    model = MyReg().fit(*_reg_data())
    out = m._tao_targets(model)
    assert len(out) == 1
    assert out[0] is model


def test_forest_classifier_returns_members_in_order_by_identity(fitted_forest_clf):
    out = m._tao_targets(fitted_forest_clf)
    members = list(fitted_forest_clf.estimators_)
    assert isinstance(out, list)
    assert len(out) == len(members)
    for a, b in zip(out, members):
        assert a is b


def test_forest_regressor_returns_members_in_order_by_identity(fitted_forest_reg):
    out = m._tao_targets(fitted_forest_reg)
    members = list(fitted_forest_reg.estimators_)
    assert len(out) == len(members)
    for a, b in zip(out, members):
        assert a is b


def test_forest_members_are_sgt_trees(fitted_forest_reg):
    out = m._tao_targets(fitted_forest_reg)
    assert all(isinstance(t, (SGTClassifier, SGTRegressor)) for t in out)


def test_forest_returned_list_is_new_list(fitted_forest_clf):
    out = m._tao_targets(fitted_forest_clf)
    assert out is not fitted_forest_clf.estimators_


def test_mutating_returned_list_does_not_change_forest(fitted_forest_clf):
    before = list(fitted_forest_clf.estimators_)
    out = m._tao_targets(fitted_forest_clf)
    out.append(object())
    out.pop(0)
    after = list(fitted_forest_clf.estimators_)
    assert len(after) == len(before)
    assert all(a is b for a, b in zip(after, before))


def test_forest_is_not_modified_by_call(fitted_forest_reg):
    before = list(fitted_forest_reg.estimators_)
    n_feat = fitted_forest_reg.n_features_in_
    m._tao_targets(fitted_forest_reg)
    assert fitted_forest_reg.n_features_in_ == n_feat
    assert all(a is b for a, b in zip(fitted_forest_reg.estimators_, before))
    assert len(fitted_forest_reg.estimators_) == len(before)


@pytest.mark.parametrize(
    "cls", [SGTClassifier, SGTRegressor, RandomSGForestClassifier, RandomSGForestRegressor]
)
def test_unfitted_model_raises_not_fitted_error_naming_class(cls):
    model = cls()
    with pytest.raises(NotFittedError) as ei:
        m._tao_targets(model)
    msg = str(ei.value)
    assert cls.__name__ in msg
    assert "not fitted" in msg.lower()


def test_tree_with_n_features_in_none_raises_not_fitted():
    model = _fit_reg()
    model.n_features_in_ = None
    with pytest.raises(NotFittedError):
        m._tao_targets(model)


def test_tree_with_n_features_in_missing_raises_not_fitted():
    model = _fit_clf()
    del model.n_features_in_
    with pytest.raises(NotFittedError):
        m._tao_targets(model)


def test_forest_with_n_features_in_none_raises_not_fitted():
    model = _fit_forest_reg()
    model.n_features_in_ = None
    with pytest.raises(NotFittedError):
        m._tao_targets(model)


def test_forest_with_empty_estimators_raises_not_fitted():
    model = _fit_forest_clf()
    model.estimators_ = []
    with pytest.raises(NotFittedError) as ei:
        m._tao_targets(model)
    assert type(model).__name__ in str(ei.value)


@pytest.mark.parametrize("bad", [None, object(), 3, "tree", [1, 2]])
def test_non_sgt_model_raises_type_error(bad):
    with pytest.raises(TypeError) as ei:
        m._tao_targets(bad)
    msg = str(ei.value)
    assert "TAO_refine" in msg
    assert type(bad).__name__ in msg


def test_sklearn_estimator_raises_type_error():
    from sklearn.tree import DecisionTreeClassifier

    X, y = _clf_data()
    est = DecisionTreeClassifier(random_state=0).fit(X, y)
    with pytest.raises(TypeError) as ei:
        m._tao_targets(est)
    assert "DecisionTreeClassifier" in str(ei.value)
