import numpy as np
import pytest

from sgtlearn import SGTRegressor
from tests.constants import TEST_TAO_N_RUNS


def _data(n_samples: int = 60, n_features: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    y = X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.1, size=n_samples)
    return X, y


def _model(**kwargs) -> SGTRegressor:
    kwargs.setdefault("tao_n_runs", TEST_TAO_N_RUNS)
    kwargs.setdefault("max_depth", 2)
    return SGTRegressor(**kwargs)


def _assert_state_identical(before: dict, model: SGTRegressor) -> None:
    after = vars(model)
    assert set(after) == set(before)
    for key, value in before.items():
        assert after[key] is value, key


def _patch_tao(monkeypatch, exc: BaseException) -> None:
    import sgtlearn.base as base_mod
    import sgtlearn.tao as tao_mod

    def boom(*args, **kwargs):
        raise exc

    monkeypatch.setattr(tao_mod, "TAO_refine", boom)
    if hasattr(base_mod, "TAO_refine"):
        monkeypatch.setattr(base_mod, "TAO_refine", boom)


def test_failed_fit_on_unfitted_model_leaves_it_unfitted() -> None:
    X, y = _data()
    y[0] = np.inf
    model = _model()
    before = dict(vars(model))

    with pytest.raises(ValueError):
        model.fit(X, y)

    _assert_state_identical(before, model)


def test_failed_refit_with_invalid_X_keeps_fitted_attributes() -> None:
    X, y = _data()
    model = _model().fit(X, y)
    before = dict(vars(model))
    X_bad = X.copy()
    X_bad[0, 0] = np.inf

    with pytest.raises(ValueError):
        model.fit(X_bad, y)

    _assert_state_identical(before, model)


def test_failed_refit_with_invalid_X_keeps_predictions_unchanged() -> None:
    X, y = _data()
    model = _model().fit(X, y)
    expected = model.predict(X)
    X_bad = X.copy()
    X_bad[0, 0] = 1e39

    with pytest.raises(ValueError):
        model.fit(X_bad, y)

    np.testing.assert_array_equal(model.predict(X), expected)


def test_failed_refit_with_invalid_sample_weight_keeps_fitted_attributes() -> None:
    X, y = _data()
    model = _model().fit(X, y)
    before = dict(vars(model))
    weights = -np.ones(X.shape[0])

    with pytest.raises(ValueError):
        model.fit(X, y, sample_weight=weights)

    _assert_state_identical(before, model)


def test_failed_refit_with_invalid_feature_dict_keeps_fitted_attributes() -> None:
    X, y = _data()
    model = _model().fit(X, y)
    before = dict(vars(model))

    with pytest.raises(ValueError):
        model.fit(X, y, feature_dict={"cat": [0, 99]})

    _assert_state_identical(before, model)


def test_failed_refit_with_invalid_hyperparameter_keeps_fitted_attributes() -> None:
    X, y = _data()
    model = _model().fit(X, y)
    expected = model.predict(X)
    model.set_params(tao_n_runs=-1)
    before = dict(vars(model))

    with pytest.raises(ValueError):
        model.fit(X, y)

    _assert_state_identical(before, model)
    np.testing.assert_array_equal(model.predict(X), expected)


def test_interrupted_tao_on_unfitted_model_leaves_it_unfitted(monkeypatch) -> None:
    X, y = _data(n_samples=40)
    model = _model(tao_n_runs=1)
    before = dict(vars(model))
    _patch_tao(monkeypatch, KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        model.fit(X, y)

    _assert_state_identical(before, model)


def test_interrupted_tao_on_fitted_model_keeps_previous_fit(monkeypatch) -> None:
    X, y = _data(n_samples=40)
    model = _model(tao_n_runs=0).fit(X, y)
    expected = model.predict(X)
    model.set_params(tao_n_runs=1)
    before = dict(vars(model))
    X_new, y_new = _data(n_samples=40, seed=1)
    _patch_tao(monkeypatch, KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        model.fit(X_new, y_new)

    _assert_state_identical(before, model)
    np.testing.assert_array_equal(model.predict(X), expected)


def test_refit_replaces_n_features_in() -> None:
    X, y = _data(n_features=3)
    model = _model().fit(X, y)
    X2, y2 = _data(n_features=5, seed=1)

    model.fit(X2, y2)

    assert model.n_features_in_ == 5


def test_refit_replaces_n_outputs() -> None:
    X, y = _data()
    model = _model().fit(X, np.column_stack([y, -y]))

    model.fit(X, y)

    assert model.n_outputs_ == 1
    assert model.predict(X).shape == (X.shape[0],)


def test_refit_on_ndarray_after_dataframe_clears_feature_names() -> None:
    pd = pytest.importorskip("pandas")
    X, y = _data()
    model = _model().fit(pd.DataFrame(X, columns=["a", "b", "c"]), y)

    model.fit(X, y)

    assert model.feature_names_in_ is None


def test_refit_replaces_processed_features() -> None:
    X, y = _data(n_features=3)
    model = _model().fit(X, y)
    first = model.processed_features_
    X2, y2 = _data(n_features=5, seed=1)

    model.fit(X2, y2)

    assert model.processed_features_ != first
