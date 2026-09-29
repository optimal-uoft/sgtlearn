import sgtlearn._weights as m
import numpy as np
import pytest


def test_maps_weights_to_samples() -> None:
    classes = np.array(["a", "b", "c"])
    y = np.array([0, 1, 2, 1, 0])
    out = m._per_class_multiplier(y, {"a": 2.0, "b": 0.5, "c": 3.0}, classes)
    np.testing.assert_allclose(out, [2.0, 0.5, 3.0, 0.5, 2.0])


def test_returns_float64_same_length() -> None:
    classes = np.array([10, 20])
    y = np.array([0, 1, 1])
    out = m._per_class_multiplier(y, {10: 2, 20: 3}, classes)
    assert out.dtype == np.float64
    assert out.shape == (3,)
    np.testing.assert_allclose(out, [2.0, 3.0, 3.0])


def test_unlisted_classes_get_one() -> None:
    classes = np.array([0, 1, 2])
    y = np.array([0, 1, 2])
    out = m._per_class_multiplier(y, {1: 4.0}, classes)
    np.testing.assert_allclose(out, [1.0, 4.0, 1.0])


def test_empty_mapping_gives_ones() -> None:
    classes = np.array([0, 1])
    y = np.array([0, 1, 0])
    out = m._per_class_multiplier(y, {}, classes)
    assert out.dtype == np.float64
    np.testing.assert_array_equal(out, np.ones(3))


def test_empty_y_gives_empty_float64() -> None:
    classes = np.array([0, 1])
    out = m._per_class_multiplier(np.array([], dtype=np.int64), {0: 2.0}, classes)
    assert out.dtype == np.float64
    assert out.shape == (0,)


def test_zero_weight_allowed() -> None:
    classes = np.array([0, 1])
    y = np.array([0, 1, 0])
    out = m._per_class_multiplier(y, {0: 0.0}, classes)
    np.testing.assert_array_equal(out, [0.0, 1.0, 0.0])


def test_inputs_not_modified() -> None:
    classes = np.array(["x", "y"])
    y = np.array([1, 0, 1])
    cw = {"x": 2.0}
    classes_copy, y_copy = classes.copy(), y.copy()
    m._per_class_multiplier(y, cw, classes)
    np.testing.assert_array_equal(classes, classes_copy)
    np.testing.assert_array_equal(y, y_copy)
    assert cw == {"x": 2.0}


def test_numpy_scalar_key_matches_python_label() -> None:
    classes = np.array([1, 2])
    y = np.array([0, 1])
    out = m._per_class_multiplier(y, {np.int64(2): 5.0}, classes)
    np.testing.assert_allclose(out, [1.0, 5.0])


def test_python_key_matches_numpy_label() -> None:
    classes = np.array([1, 2], dtype=np.int64)
    y = np.array([1, 0])
    out = m._per_class_multiplier(y, {1: 7.0}, classes)
    np.testing.assert_allclose(out, [1.0, 7.0])


def test_unknown_key_raises() -> None:
    classes = np.array([0, 1])
    with pytest.raises(ValueError, match="99"):
        m._per_class_multiplier(np.array([0, 1]), {99: 1.0}, classes)


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), float("-inf")])
def test_invalid_weight_value_raises(bad) -> None:
    classes = np.array([0, 1])
    with pytest.raises(ValueError):
        m._per_class_multiplier(np.array([0, 1]), {0: bad}, classes)


def test_non_numeric_weight_raises() -> None:
    classes = np.array([0, 1])
    with pytest.raises((TypeError, ValueError)):
        m._per_class_multiplier(np.array([0, 1]), {0: "abc"}, classes)


def test_non_numeric_object_weight_raises() -> None:
    classes = np.array([0, 1])
    with pytest.raises((TypeError, ValueError)):
        m._per_class_multiplier(np.array([0, 1]), {0: object()}, classes)


def test_code_too_large_raises() -> None:
    classes = np.array([0, 1])
    with pytest.raises(ValueError):
        m._per_class_multiplier(np.array([0, 2]), {0: 2.0}, classes)


def test_negative_code_raises() -> None:
    classes = np.array([0, 1])
    with pytest.raises(ValueError):
        m._per_class_multiplier(np.array([0, -1]), {1: 2.0}, classes)


def test_out_of_range_code_raises_with_empty_mapping() -> None:
    classes = np.array([0, 1])
    with pytest.raises(ValueError):
        m._per_class_multiplier(np.array([5]), {}, classes)
