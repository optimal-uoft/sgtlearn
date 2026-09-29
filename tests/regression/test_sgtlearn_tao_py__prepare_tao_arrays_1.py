import sgtlearn.tao as m
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

import sgtlearn
from sgtlearn.ensemble import RandomSGForestClassifier, RandomSGForestRegressor

SGTClassifier = getattr(m, "SGTClassifier", None) or sgtlearn.SGTClassifier
SGTRegressor = getattr(m, "SGTRegressor", None) or sgtlearn.SGTRegressor


def _data(n=40, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.rand(n, 3)
    y = (X[:, 0] > 0.5).astype(int)
    return X, y


def _clf(class_weight=None):
    X, y = _data()
    labels = np.where(y == 1, "b", "a")
    if class_weight is None:
        model = SGTClassifier()
    else:
        model = SGTClassifier(class_weight=class_weight)
    model.fit(X, labels)
    return model, X, labels


def _reg():
    X, _ = _data()
    y = X[:, 0] * 2.0 + X[:, 1]
    return SGTRegressor().fit(X, y), X, y


def test_X32_is_c_contiguous_float32_with_same_values():
    model, X, y = _clf()
    Xf = np.asfortranarray(X)
    X32, _, _ = m._prepare_tao_arrays(model, Xf, y, None)
    assert X32.dtype == np.float32
    assert X32.flags["C_CONTIGUOUS"]
    assert X32.shape == X.shape
    np.testing.assert_allclose(X32, X.astype(np.float32))


def test_classifier_y_native_is_uint64_encoded_indices_matching_classes():
    model, X, y = _clf()
    _, y_native, _ = m._prepare_tao_arrays(model, X, y, None)
    assert y_native.dtype == np.uint64
    assert y_native.shape == (len(y),)
    classes = np.asarray(model.classes_)
    np.testing.assert_array_equal(classes[y_native.astype(np.int64)], y)


def test_sample_weight_none_gives_unit_float32_weights():
    model, X, y = _clf()
    _, _, w = m._prepare_tao_arrays(model, X, y, None)
    assert w.dtype == np.float32
    assert w.shape == (len(y),)
    np.testing.assert_allclose(w, 1.0)


def test_sample_weight_given_is_proportional_float32():
    model, X, y = _clf()
    sw = np.linspace(1.0, 3.0, len(y))
    _, _, w = m._prepare_tao_arrays(model, X, y, sw)
    assert w.dtype == np.float32
    assert w.ndim == 1 and len(w) == len(y)
    np.testing.assert_allclose(w / w[0], sw / sw[0], rtol=1e-5)


def test_class_weight_without_sample_weight_applies_class_weights():
    model, X, y = _clf(class_weight={"a": 1.0, "b": 3.0})
    _, _, w = m._prepare_tao_arrays(model, X, y, None)
    expected = np.where(y == "b", 3.0, 1.0)
    np.testing.assert_allclose(w, expected, rtol=1e-6)


def test_class_weight_multiplies_sample_weight():
    model, X, y = _clf(class_weight={"a": 2.0, "b": 5.0})
    sw = np.ones(len(y))
    sw[::2] = 0.5
    _, _, w = m._prepare_tao_arrays(model, X, y, sw)
    cw = np.where(y == "b", 5.0, 2.0)
    base = w / cw
    np.testing.assert_allclose(base / base[1], sw / sw[1], rtol=1e-5)


def test_regressor_1d_y_gives_1d_float32():
    model, X, y = _reg()
    _, y_native, w = m._prepare_tao_arrays(model, X, y, None)
    assert y_native.dtype == np.float32
    assert y_native.shape == (len(y),)
    np.testing.assert_allclose(y_native, y.astype(np.float32))
    np.testing.assert_allclose(w, 1.0)


def test_regressor_single_column_y_gives_1d():
    model, X, y = _reg()
    _, y_native, _ = m._prepare_tao_arrays(model, X, y.reshape(-1, 1), None)
    assert y_native.dtype == np.float32
    assert y_native.shape == (len(y),)
    np.testing.assert_allclose(y_native, y.astype(np.float32))


def test_regressor_multi_output_y_gives_2d():
    X, _ = _data()
    Y = np.column_stack([X[:, 0], X[:, 1] * 3.0])
    model = SGTRegressor().fit(X, Y)
    _, y_native, _ = m._prepare_tao_arrays(model, X, Y, None)
    assert y_native.dtype == np.float32
    assert y_native.shape == Y.shape
    np.testing.assert_allclose(y_native, Y.astype(np.float32))


def test_forest_classifier_encodes_labels():
    X, y = _data()
    labels = np.where(y == 1, 10, 20)
    model = RandomSGForestClassifier(n_estimators=3, random_state=0).fit(X, labels)
    X32, y_native, w = m._prepare_tao_arrays(model, X, labels, None)
    assert y_native.dtype == np.uint64
    assert y_native.shape == (len(labels),)
    np.testing.assert_array_equal(
        np.asarray(model.classes_)[y_native.astype(np.int64)], labels
    )
    assert w.shape == (len(labels),)


def test_forest_regressor_returns_float32_targets():
    X, _ = _data()
    y = X[:, 0] + X[:, 2]
    model = RandomSGForestRegressor(n_estimators=3, random_state=0).fit(X, y)
    _, y_native, _ = m._prepare_tao_arrays(model, X, y, None)
    assert y_native.dtype == np.float32
    assert y_native.shape == (len(y),)


def test_inputs_are_not_modified():
    model, X, y = _clf()
    sw = np.linspace(1.0, 2.0, len(y))
    X0, y0, sw0 = X.copy(), y.copy(), sw.copy()
    m._prepare_tao_arrays(model, X, y, sw)
    np.testing.assert_array_equal(X, X0)
    np.testing.assert_array_equal(y, y0)
    np.testing.assert_array_equal(sw, sw0)


def test_result_is_deterministic():
    model, X, y = _clf()
    a = m._prepare_tao_arrays(model, X, y, None)
    b = m._prepare_tao_arrays(model, X, y, None)
    for u, v in zip(a, b):
        np.testing.assert_array_equal(u, v)


def test_unfitted_classifier_raises_not_fitted():
    X, y = _data()
    with pytest.raises(NotFittedError):
        m._prepare_tao_arrays(SGTClassifier(), X, y, None)


def test_unfitted_regressor_raises_not_fitted():
    X, y = _data()
    with pytest.raises(NotFittedError):
        m._prepare_tao_arrays(SGTRegressor(), X, y.astype(float), None)


def test_classes_none_raises_not_fitted():
    model, X, y = _clf()
    model.classes_ = None
    with pytest.raises(NotFittedError):
        m._prepare_tao_arrays(model, X, y, None)


def test_classes_missing_raises_not_fitted():
    model, X, y = _clf()
    del model.classes_
    with pytest.raises(NotFittedError):
        m._prepare_tao_arrays(model, X, y, None)


def test_X_not_2d_raises_value_error():
    model, X, y = _clf()
    with pytest.raises(ValueError):
        m._prepare_tao_arrays(model, X[:, 0], y, None)


def test_sample_count_mismatch_raises_value_error():
    model, X, y = _clf()
    with pytest.raises(ValueError):
        m._prepare_tao_arrays(model, X, y[:-1], None)


def test_sample_weight_wrong_length_raises_value_error():
    model, X, y = _clf()
    with pytest.raises(ValueError):
        m._prepare_tao_arrays(model, X, y, np.ones(len(y) - 1))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1.0])
def test_invalid_sample_weight_values_raise_value_error(bad):
    model, X, y = _clf()
    sw = np.ones(len(y))
    sw[3] = bad
    with pytest.raises(ValueError):
        m._prepare_tao_arrays(model, X, y, sw)


def test_unseen_label_raises_value_error():
    model, X, y = _clf()
    y2 = y.copy().astype(object)
    y2[0] = "zzz"
    with pytest.raises(ValueError):
        m._prepare_tao_arrays(model, X, y2, None)


def test_non_numeric_regression_target_raises_value_error():
    model, X, y = _reg()
    y2 = y.astype(object)
    y2[0] = "not-a-number"
    with pytest.raises(ValueError):
        m._prepare_tao_arrays(model, X, y2, None)
