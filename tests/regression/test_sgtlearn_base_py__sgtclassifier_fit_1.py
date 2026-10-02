import numpy as np
import pytest

from sgtlearn import SGTClassifier
from sgtlearn._features import ProcessedFeatures, configure_feature_dict
from tests.constants import TEST_TAO_N_RUNS


def _make_data(n_samples: int = 60, n_features: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features)).astype(np.float64)
    y = (X[:, 0] > 0).astype(int)
    # make sure both classes are present
    y[0], y[1] = 0, 1
    return X, y


def _make_multi_output_data(n_samples: int = 60, seed: int = 1):
    X, y0 = _make_data(n_samples=n_samples, seed=seed)
    y1 = (X[:, 1] > 0).astype(int)
    y1[0], y1[1] = 0, 1
    return X, np.column_stack([y0, y1])


def _make_categorical_data(n_samples: int = 60, seed: int = 2):
    rng = np.random.default_rng(seed)
    cats = rng.integers(0, 3, size=n_samples)
    cats[:3] = [0, 1, 2]
    one_hot = np.eye(3)[cats]
    cont = rng.normal(size=(n_samples, 1))
    X = np.hstack([one_hot, cont])
    y = (cats == 1).astype(int)
    return X, y


def _clf(**kwargs) -> SGTClassifier:
    params = {"max_depth": 2, "tao_n_runs": TEST_TAO_N_RUNS}
    params.update(kwargs)
    return SGTClassifier(**params)


_FITTED_ATTRS = (
    "_est",
    "_le",
    "classes_",
    "n_classes_",
    "n_outputs_",
    "n_features_in_",
    "feature_names_in_",
    "_processed_features",
    "_tao_refined_",
)


# ---------------------------------------------------------------------------
# Successful fits
# ---------------------------------------------------------------------------


def test_fit_returns_self() -> None:
    X, y = _make_data()
    clf = _clf()
    assert clf.fit(X, y) is clf


def test_fit_sets_single_output_attributes_unwrapped() -> None:
    X, y = _make_data()
    clf = _clf().fit(X, y)

    assert clf.n_outputs_ == 1
    assert clf.n_features_in_ == X.shape[1]
    assert isinstance(clf.n_classes_, (int, np.integer))
    assert clf.n_classes_ == 2
    assert set(np.asarray(clf.classes_).tolist()) == {0, 1}


def test_fit_with_string_labels_records_classes_and_predicts_them() -> None:
    X, y = _make_data()
    labels = np.where(y == 1, "pos", "neg")
    clf = _clf().fit(X, labels)

    assert set(np.asarray(clf.classes_).tolist()) == {"neg", "pos"}
    assert set(np.asarray(clf.predict(X)).tolist()) <= {"neg", "pos"}


def test_fit_on_ndarray_sets_feature_names_in_to_none() -> None:
    X, y = _make_data()
    clf = _clf().fit(X, y)
    assert clf.feature_names_in_ is None


def test_fit_on_dataframe_records_feature_names_in() -> None:
    pd = pytest.importorskip("pandas")
    X, y = _make_data()
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    clf = _clf().fit(df, y)

    assert list(clf.feature_names_in_) == ["a", "b", "c"]
    assert clf.n_features_in_ == 3


def test_fit_multi_output_sets_per_output_classes() -> None:
    X, Y = _make_multi_output_data()
    clf = _clf().fit(X, Y)

    assert clf.n_outputs_ == 2
    assert len(clf.classes_) == 2
    assert len(clf.n_classes_) == 2
    for classes, n_classes in zip(clf.classes_, clf.n_classes_):
        assert set(np.asarray(classes).tolist()) == {0, 1}
        assert n_classes == 2


def test_fit_accepts_nan_in_X() -> None:
    X, y = _make_data()
    X[3, 1] = np.nan
    X[7, 2] = np.nan
    clf = _clf().fit(X, y)
    assert clf.n_features_in_ == X.shape[1]


def test_fit_accepts_valid_sample_weight() -> None:
    X, y = _make_data()
    sw = np.linspace(0.5, 2.0, num=len(y))
    clf = _clf()
    assert clf.fit(X, y, sample_weight=sw) is clf


def test_fit_accepts_valid_class_weight() -> None:
    X, y = _make_data()
    clf = _clf(class_weight={0: 1.0, 1: 2.0})
    assert clf.fit(X, y) is clf


def test_fit_with_tao_n_runs_zero_skips_tao() -> None:
    X, y = _make_data()
    clf = _clf(tao_n_runs=0).fit(X, y)
    assert clf._tao_refined_ is False


def test_fit_with_tao_runs_sets_tao_refined_true() -> None:
    X, y = _make_data()
    clf = _clf(tao_n_runs=1).fit(X, y)
    assert clf._tao_refined_ is True


def test_fit_does_not_change_hyper_parameters() -> None:
    X, y = _make_data()
    clf = _clf(class_weight={0: 1.0, 1: 3.0}, random_state=None)
    before = clf.get_params()
    clf.fit(X, y)
    after = clf.get_params()
    assert before.keys() == after.keys()
    for key in before:
        assert before[key] == after[key], key


def test_fit_with_random_state_none_is_deterministic() -> None:
    X, y = _make_data()
    p1 = np.asarray(_clf(random_state=None).fit(X, y).predict_proba(X))
    p2 = np.asarray(_clf(random_state=None).fit(X, y).predict_proba(X))
    np.testing.assert_array_equal(p1, p2)


# ---------------------------------------------------------------------------
# feature_dict / processed_features
# ---------------------------------------------------------------------------


def test_fit_with_feature_dict_resolves_processed_features() -> None:
    X, y = _make_categorical_data()
    feature_dict = {"color": [0, 1, 2]}
    clf = _clf().fit(X, y, feature_dict=feature_dict)

    expected = configure_feature_dict(X.shape[1], feature_dict)
    assert isinstance(clf.processed_features_, ProcessedFeatures)
    assert clf.processed_features_ == expected


def test_fit_with_feature_dict_column_names_on_dataframe() -> None:
    pd = pytest.importorskip("pandas")
    X, y = _make_categorical_data()
    df = pd.DataFrame(X, columns=["red", "green", "blue", "size"])
    feature_dict = {"color": ["red", "green", "blue"]}
    clf = _clf().fit(df, y, feature_dict=feature_dict)

    expected = configure_feature_dict(
        X.shape[1], feature_dict, column_names=["red", "green", "blue", "size"]
    )
    assert clf.processed_features_ == expected


def test_fit_with_processed_features_uses_them() -> None:
    X, y = _make_categorical_data()
    pf = configure_feature_dict(X.shape[1], {"color": [0, 1, 2]})
    clf = _clf().fit(X, y, processed_features=pf)
    assert clf.processed_features_ == pf


def test_fit_rejects_feature_dict_column_names_without_dataframe() -> None:
    X, y = _make_categorical_data()
    with pytest.raises(ValueError):
        _clf().fit(X, y, feature_dict={"color": ["red", "green", "blue"]})


def test_fit_rejects_feature_dict_with_out_of_range_column() -> None:
    X, y = _make_categorical_data()
    with pytest.raises(ValueError):
        _clf().fit(X, y, feature_dict={"color": [0, 1, 99]})


def test_fit_rejects_processed_features_for_wrong_number_of_columns() -> None:
    X, y = _make_categorical_data()
    pf = configure_feature_dict(X.shape[1] + 2, {"color": [0, 1, 2]})
    with pytest.raises(ValueError):
        _clf().fit(X, y, processed_features=pf)


# ---------------------------------------------------------------------------
# Input validation errors
# ---------------------------------------------------------------------------


def test_fit_rejects_infinite_X() -> None:
    X, y = _make_data()
    X[4, 0] = np.inf
    with pytest.raises(ValueError):
        _clf().fit(X, y)


def test_fit_rejects_X_overflowing_float32() -> None:
    X, y = _make_data()
    X[4, 0] = 1e40
    with pytest.raises(ValueError):
        _clf().fit(X, y)


def test_fit_rejects_one_dimensional_X() -> None:
    X, y = _make_data()
    with pytest.raises(ValueError):
        _clf().fit(X[:, 0], y)


def test_fit_rejects_mismatched_sample_counts() -> None:
    X, y = _make_data()
    with pytest.raises(ValueError):
        _clf().fit(X, y[:-5])


def test_fit_rejects_single_class_y() -> None:
    X, _ = _make_data()
    y = np.zeros(X.shape[0], dtype=int)
    with pytest.raises(ValueError):
        _clf().fit(X, y)


def test_fit_rejects_multi_output_with_single_class_output() -> None:
    X, Y = _make_multi_output_data()
    Y[:, 1] = 0
    with pytest.raises(ValueError):
        _clf().fit(X, Y)


def test_fit_rejects_sparse_X_with_type_error() -> None:
    sparse = pytest.importorskip("scipy.sparse")
    X, y = _make_data()
    with pytest.raises(TypeError):
        _clf().fit(sparse.csr_matrix(X), y)


def test_fit_rejects_sample_weight_of_wrong_length() -> None:
    X, y = _make_data()
    with pytest.raises(ValueError):
        _clf().fit(X, y, sample_weight=np.ones(len(y) - 1))


def test_fit_rejects_negative_sample_weight() -> None:
    X, y = _make_data()
    sw = np.ones(len(y))
    sw[0] = -1.0
    with pytest.raises(ValueError):
        _clf().fit(X, y, sample_weight=sw)


@pytest.mark.parametrize(
    "param", ["min_impurity_decrease", "pairwise_penalty", "branching_penalty"]
)
@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
def test_fit_rejects_non_finite_or_negative_penalty_params(param, value) -> None:
    X, y = _make_data()
    with pytest.raises(ValueError):
        _clf(**{param: value}).fit(X, y)


def test_fit_rejects_negative_pairwise_candidates() -> None:
    X, y = _make_data()
    with pytest.raises(ValueError):
        _clf(pairwise_candidates=-1).fit(X, y)


def test_fit_rejects_nan_tao_pair_scale() -> None:
    X, y = _make_data()
    with pytest.raises(ValueError):
        _clf(tao_pair_scale=float("nan")).fit(X, y)


@pytest.mark.parametrize("value", [-1, True, False, 1.0, None])
def test_fit_rejects_invalid_tao_n_runs(value) -> None:
    X, y = _make_data()
    with pytest.raises(ValueError):
        _clf(tao_n_runs=value).fit(X, y)


# ---------------------------------------------------------------------------
# check_input=False
# ---------------------------------------------------------------------------


def test_fit_check_input_false_with_preset_classes_succeeds() -> None:
    X, y = _make_data()
    X32 = np.ascontiguousarray(X, dtype=np.float32)
    clf = _clf()
    clf.classes_ = np.array([0, 1])
    clf.n_classes_ = 2
    assert clf.fit(X32, y, check_input=False) is clf
    assert set(np.asarray(clf.classes_).tolist()) == {0, 1}
    assert clf.n_classes_ == 2


def test_fit_check_input_false_with_preset_multi_output_classes_succeeds() -> None:
    X, Y = _make_multi_output_data()
    X32 = np.ascontiguousarray(X, dtype=np.float32)
    clf = _clf()
    clf.classes_ = [np.array([0, 1]), np.array([0, 1])]
    clf.n_classes_ = [2, 2]
    clf.fit(X32, Y, check_input=False)
    assert clf.n_outputs_ == 2


def test_fit_check_input_false_without_preset_classes_raises() -> None:
    X, y = _make_data()
    X32 = np.ascontiguousarray(X, dtype=np.float32)
    with pytest.raises(ValueError):
        _clf().fit(X32, y, check_input=False)


def test_fit_check_input_false_with_labels_outside_preset_classes_raises() -> None:
    X, y = _make_data()
    X32 = np.ascontiguousarray(X, dtype=np.float32)
    y = y.copy()
    y[5] = 7
    clf = _clf()
    clf.classes_ = np.array([0, 1])
    clf.n_classes_ = 2
    with pytest.raises(ValueError):
        clf.fit(X32, y, check_input=False)


def test_fit_check_input_false_with_output_count_mismatch_raises() -> None:
    X, Y = _make_multi_output_data()
    X32 = np.ascontiguousarray(X, dtype=np.float32)
    clf = _clf()
    clf.classes_ = [np.array([0, 1]), np.array([0, 1]), np.array([0, 1])]
    clf.n_classes_ = [2, 2, 2]
    with pytest.raises(ValueError):
        clf.fit(X32, Y, check_input=False)


def test_failed_check_input_false_fit_leaves_preset_classes_untouched() -> None:
    X, y = _make_data()
    X32 = np.ascontiguousarray(X, dtype=np.float32)
    y = y.copy()
    y[5] = 7
    clf = _clf()
    preset_classes = np.array([0, 1])
    clf.classes_ = preset_classes
    clf.n_classes_ = 2
    with pytest.raises(ValueError):
        clf.fit(X32, y, check_input=False)
    assert clf.classes_ is preset_classes
    assert clf.n_classes_ == 2


# ---------------------------------------------------------------------------
# Exception safety
# ---------------------------------------------------------------------------


def test_failed_refit_keeps_previous_fitted_attributes_and_predictions() -> None:
    X, y = _make_data()
    clf = _clf().fit(X, y)
    before = {name: getattr(clf, name) for name in _FITTED_ATTRS}
    proba_before = np.asarray(clf.predict_proba(X)).copy()
    pred_before = np.asarray(clf.predict(X)).copy()

    X_bad = X.copy()
    X_bad[0, 0] = np.inf
    with pytest.raises(ValueError):
        clf.fit(X_bad, y)

    for name, value in before.items():
        assert getattr(clf, name) is value, name
    np.testing.assert_array_equal(np.asarray(clf.predict_proba(X)), proba_before)
    np.testing.assert_array_equal(np.asarray(clf.predict(X)), pred_before)


def test_failed_first_fit_leaves_estimator_unfitted() -> None:
    X, _ = _make_data()
    y = np.zeros(X.shape[0], dtype=int)
    clf = _clf()
    with pytest.raises(ValueError):
        clf.fit(X, y)
    assert getattr(clf, "classes_", None) is None
    assert getattr(clf, "n_classes_", None) is None
    assert getattr(clf, "n_features_in_", None) is None
    assert getattr(clf, "_est", None) is None


def test_base_exception_during_tao_keeps_previous_fit(monkeypatch) -> None:
    import sgtlearn.base as base_mod
    import sgtlearn.tao as tao_mod

    X, y = _make_data()
    clf = _clf(tao_n_runs=0).fit(X, y)
    before = {name: getattr(clf, name) for name in _FITTED_ATTRS}
    proba_before = np.asarray(clf.predict_proba(X)).copy()

    def _boom(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(tao_mod, "TAO_refine", _boom)
    monkeypatch.setattr(base_mod, "TAO_refine", _boom, raising=False)

    clf.set_params(tao_n_runs=1)
    with pytest.raises(KeyboardInterrupt):
        clf.fit(X, y)

    for name, value in before.items():
        assert getattr(clf, name) is value, name
    np.testing.assert_array_equal(np.asarray(clf.predict_proba(X)), proba_before)
