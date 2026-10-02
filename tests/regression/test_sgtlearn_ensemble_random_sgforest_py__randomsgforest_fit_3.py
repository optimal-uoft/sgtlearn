from sgtlearn import RandomSGForestClassifier, RandomSGForestRegressor
import numpy as np
import pytest


FITTED_ATTRS = (
    "estimators_",
    "n_outputs_",
    "n_features_in_",
    "feature_names_in_",
    "processed_features_",
)
CLASSIFIER_ATTRS = FITTED_ATTRS + ("classes_", "n_classes_")


def _clf_data(n_samples: int = 40, n_features: int = 3, seed: int = 0):
    rng = np.random.RandomState(seed)
    X = rng.uniform(-1.0, 1.0, size=(n_samples, n_features))
    y = (X[:, 0] > 0).astype(int)
    return X, y


def _reg_data(n_samples: int = 40, n_features: int = 3, seed: int = 0):
    rng = np.random.RandomState(seed)
    X = rng.uniform(-1.0, 1.0, size=(n_samples, n_features))
    y = 2.0 * X[:, 0] - X[:, 1]
    return X, y


def _clf(**kwargs) -> RandomSGForestClassifier:
    params = dict(n_estimators=2, tao_n_runs=0, random_state=0)
    params.update(kwargs)
    return RandomSGForestClassifier(**params)


def _reg(**kwargs) -> RandomSGForestRegressor:
    params = dict(n_estimators=2, tao_n_runs=0, random_state=0)
    params.update(kwargs)
    return RandomSGForestRegressor(**params)


def _snapshot(forest, names):
    return {name: getattr(forest, name) for name in names}


def test_failed_refit_keeps_classifier_attributes_identical() -> None:
    X, y = _clf_data()
    forest = _clf().fit(X, y)
    before = _snapshot(forest, CLASSIFIER_ATTRS)
    bad_y = np.zeros(X.shape[0], dtype=int)

    with pytest.raises(ValueError):
        forest.fit(X, bad_y)

    for name, value in before.items():
        assert getattr(forest, name) is value


def test_failed_refit_keeps_classifier_predictions() -> None:
    X, y = _clf_data()
    forest = _clf().fit(X, y)
    pred_before = forest.predict(X)
    proba_before = np.asarray(forest.predict_proba(X))
    X_bad = X.copy()
    X_bad[0, 0] = np.inf

    with pytest.raises(ValueError):
        forest.fit(X_bad, y)

    np.testing.assert_array_equal(forest.predict(X), pred_before)
    np.testing.assert_array_equal(np.asarray(forest.predict_proba(X)), proba_before)


def test_failed_refit_keeps_regressor_attributes_and_predictions() -> None:
    X, y = _reg_data()
    forest = _reg().fit(X, y)
    before = _snapshot(forest, FITTED_ATTRS)
    pred_before = forest.predict(X)

    with pytest.raises(ValueError):
        forest.set_params(n_estimators=0).fit(X, y)

    for name, value in before.items():
        assert getattr(forest, name) is value
    np.testing.assert_array_equal(forest.predict(X), pred_before)


def test_base_tree_failure_during_refit_keeps_fitted_state() -> None:
    X, y = _clf_data()
    forest = _clf().fit(X, y)
    before = _snapshot(forest, CLASSIFIER_ATTRS)
    pred_before = forest.predict(X)
    forest.set_params(min_impurity_decrease=-1.0)

    with pytest.raises((ValueError, TypeError)):
        forest.fit(X, y)

    for name, value in before.items():
        assert getattr(forest, name) is value
    np.testing.assert_array_equal(forest.predict(X), pred_before)


def test_refit_failure_with_different_feature_count_keeps_state() -> None:
    X, y = _clf_data()
    forest = _clf().fit(X, y)
    before = _snapshot(forest, CLASSIFIER_ATTRS)
    X_wide, _ = _clf_data(n_features=5)
    X_wide[1, 4] = np.inf

    with pytest.raises(ValueError):
        forest.fit(X_wide, y)

    for name, value in before.items():
        assert getattr(forest, name) is value


def test_failed_first_fit_leaves_classifier_unfitted() -> None:
    X, _ = _clf_data()
    forest = _clf()

    with pytest.raises(ValueError):
        forest.fit(X, np.ones(X.shape[0], dtype=int))

    for name in CLASSIFIER_ATTRS:
        assert not hasattr(forest, name)


def test_failed_first_fit_from_base_tree_leaves_regressor_unfitted() -> None:
    X, y = _reg_data()
    forest = _reg(min_impurity_decrease=-1.0)

    with pytest.raises((ValueError, TypeError)):
        forest.fit(X, y)

    for name in FITTED_ATTRS:
        assert not hasattr(forest, name)


def test_failed_fit_does_not_change_hyperparameters() -> None:
    X, y = _reg_data()
    forest = _reg(bootstrap=False, max_samples=0.5)
    params_before = forest.get_params()

    with pytest.raises(ValueError):
        forest.fit(X, y)

    assert forest.get_params() == params_before


def test_successful_fit_does_not_change_hyperparameters() -> None:
    X, y = _clf_data()
    forest = _clf(max_features="sqrt", n_jobs=1)
    params_before = forest.get_params()

    forest.fit(X, y)

    assert forest.get_params() == params_before


def test_keyboard_interrupt_in_refit_keeps_fitted_state(monkeypatch) -> None:
    X, y = _clf_data()
    forest = _clf(n_jobs=1).fit(X, y)
    before = _snapshot(forest, CLASSIFIER_ATTRS)
    pred_before = forest.predict(X)

    tree_cls = type(forest.estimators_[0])

    def interrupt(self, *args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(tree_cls, "fit", interrupt)

    with pytest.raises(KeyboardInterrupt):
        forest.fit(X, y)

    monkeypatch.undo()
    for name, value in before.items():
        assert getattr(forest, name) is value
    np.testing.assert_array_equal(forest.predict(X), pred_before)


def test_base_tree_error_is_passed_through_unchanged(monkeypatch) -> None:
    X, y = _reg_data()
    probe = _reg(n_estimators=1).fit(X, y)
    tree_cls = type(probe.estimators_[0])

    class SentinelError(Exception):
        pass

    def fail(self, *args, **kwargs):
        raise SentinelError("from base tree")

    monkeypatch.setattr(tree_cls, "fit", fail)
    forest = _reg(n_jobs=1)

    with pytest.raises(SentinelError, match="from base tree"):
        forest.fit(X, y)

    for name in FITTED_ATTRS:
        assert not hasattr(forest, name)
