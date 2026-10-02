import warnings

import sgtlearn.base as m
import numpy as np
import pytest


F32_MAX = float(np.finfo(np.float32).max)
INF_MSG = r"infinity or a value too large for dtype\('float32'\)"


def _assert_native(out: np.ndarray) -> None:
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert out.flags["C_CONTIGUOUS"]


def test_float64_1d_is_converted_to_float32_1d() -> None:
    y = np.array([1.5, -2.25, 0.0, 3.0], dtype=np.float64)
    out = m._as_native_y(y)

    _assert_native(out)
    assert out.shape == (4,)
    np.testing.assert_array_equal(out, y.astype(np.float32))


def test_float32_1d_input_keeps_values() -> None:
    y = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    out = m._as_native_y(y)

    _assert_native(out)
    assert out.shape == (3,)
    np.testing.assert_array_equal(out, y)


def test_integer_1d_is_converted_to_float32() -> None:
    y = np.array([1, -2, 3, 40], dtype=np.int64)
    out = m._as_native_y(y)

    _assert_native(out)
    assert out.shape == (4,)
    np.testing.assert_array_equal(out, np.array([1.0, -2.0, 3.0, 40.0], dtype=np.float32))


def test_nested_list_1d_is_converted_to_float32() -> None:
    out = m._as_native_y([1.0, 2.5, -3.0])

    _assert_native(out)
    assert out.shape == (3,)
    np.testing.assert_array_equal(out, np.array([1.0, 2.5, -3.0], dtype=np.float32))


def test_nested_list_2d_multioutput_is_converted_to_2d_float32() -> None:
    out = m._as_native_y([[1, 2], [3, 4], [5, 6]])

    _assert_native(out)
    assert out.shape == (3, 2)
    np.testing.assert_array_equal(
        out, np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    )


def test_multioutput_2d_array_keeps_2d_shape() -> None:
    y = np.arange(12, dtype=np.float64).reshape(4, 3)
    out = m._as_native_y(y)

    _assert_native(out)
    assert out.shape == (4, 3)
    np.testing.assert_array_equal(out, y.astype(np.float32))


def test_single_column_2d_input_is_flattened_to_1d() -> None:
    y = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
    out = m._as_native_y(y)

    _assert_native(out)
    assert out.shape == (3,)
    np.testing.assert_array_equal(out, np.array([1.0, 2.0, 3.0], dtype=np.float32))


def test_single_column_nested_list_is_flattened_to_1d() -> None:
    out = m._as_native_y([[4], [5]])

    _assert_native(out)
    assert out.shape == (2,)
    np.testing.assert_array_equal(out, np.array([4.0, 5.0], dtype=np.float32))


def test_fortran_ordered_multioutput_input_gives_c_contiguous_output() -> None:
    y = np.asfortranarray(np.arange(10, dtype=np.float64).reshape(5, 2))
    assert not y.flags["C_CONTIGUOUS"]
    out = m._as_native_y(y)

    _assert_native(out)
    assert out.shape == (5, 2)
    np.testing.assert_array_equal(out, y.astype(np.float32))


def test_strided_1d_input_gives_c_contiguous_output() -> None:
    base = np.arange(10, dtype=np.float32)
    y = base[::2]
    assert not y.flags["C_CONTIGUOUS"]
    out = m._as_native_y(y)

    _assert_native(out)
    assert out.shape == (5,)
    np.testing.assert_array_equal(out, np.array([0, 2, 4, 6, 8], dtype=np.float32))


def test_strided_float32_multioutput_input_gives_c_contiguous_output() -> None:
    base = np.arange(24, dtype=np.float32).reshape(6, 4)
    y = base[:, ::2]
    assert not y.flags["C_CONTIGUOUS"]
    out = m._as_native_y(y)

    _assert_native(out)
    assert out.shape == (6, 2)
    np.testing.assert_array_equal(out, base[:, ::2])


def test_input_array_is_not_modified() -> None:
    y = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
    original = y.copy()
    m._as_native_y(y)

    assert y.dtype == np.float64
    np.testing.assert_array_equal(y, original)


def test_float32_input_is_not_modified_when_output_is_changed() -> None:
    y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    original = y.copy()
    out = m._as_native_y(y)
    out_copy = np.array(out, copy=True)

    assert out.dtype == np.float32
    np.testing.assert_array_equal(y, original)
    np.testing.assert_array_equal(out_copy, original)


def test_input_list_is_not_modified() -> None:
    y = [[1, 2], [3, 4]]
    m._as_native_y(y)

    assert y == [[1, 2], [3, 4]]


def test_positive_infinity_raises_value_error() -> None:
    y = np.array([1.0, np.inf, 2.0])
    with pytest.raises(ValueError, match=INF_MSG):
        m._as_native_y(y)


def test_negative_infinity_raises_value_error() -> None:
    y = np.array([1.0, -np.inf, 2.0])
    with pytest.raises(ValueError, match=INF_MSG):
        m._as_native_y(y)


def test_infinity_in_float32_input_raises_value_error() -> None:
    y = np.array([1.0, np.inf], dtype=np.float32)
    with pytest.raises(ValueError, match=INF_MSG):
        m._as_native_y(y)


def test_infinity_in_multioutput_input_raises_value_error() -> None:
    y = np.array([[1.0, 2.0], [3.0, np.inf]])
    with pytest.raises(ValueError, match=INF_MSG):
        m._as_native_y(y)


def test_infinity_in_nested_list_raises_value_error() -> None:
    with pytest.raises(ValueError, match=INF_MSG):
        m._as_native_y([1.0, float("inf")])


def test_large_positive_finite_value_overflowing_float32_raises_value_error() -> None:
    y = np.array([1.0, 1e39, 2.0], dtype=np.float64)
    with pytest.raises(ValueError, match=INF_MSG):
        m._as_native_y(y)


def test_large_negative_finite_value_overflowing_float32_raises_value_error() -> None:
    y = np.array([1.0, -1e39], dtype=np.float64)
    with pytest.raises(ValueError, match=INF_MSG):
        m._as_native_y(y)


def test_overflowing_value_in_multioutput_input_raises_value_error() -> None:
    y = np.array([[1.0, 2.0], [1e300, 4.0]], dtype=np.float64)
    with pytest.raises(ValueError, match=INF_MSG):
        m._as_native_y(y)


def test_overflow_does_not_emit_runtime_warning() -> None:
    y = np.array([1.0, 1e39, -1e300], dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError, match=INF_MSG):
            m._as_native_y(y)


def test_valid_conversion_does_not_emit_runtime_warning() -> None:
    y = np.array([1.0, F32_MAX, -F32_MAX, 1e-50], dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = m._as_native_y(y)
    assert out.dtype == np.float32


def test_largest_finite_float32_and_its_negative_are_accepted() -> None:
    y = np.array([F32_MAX, -F32_MAX], dtype=np.float64)
    out = m._as_native_y(y)

    _assert_native(out)
    assert np.all(np.isfinite(out))
    np.testing.assert_array_equal(
        out, np.array([np.finfo(np.float32).max, -np.finfo(np.float32).max], dtype=np.float32)
    )


def test_largest_finite_float32_in_float32_input_is_accepted() -> None:
    fmax = np.finfo(np.float32).max
    y = np.array([fmax, -fmax], dtype=np.float32)
    out = m._as_native_y(y)

    _assert_native(out)
    np.testing.assert_array_equal(out, y)


def test_tiny_value_underflowing_to_zero_is_not_an_error() -> None:
    y = np.array([1e-50, -1e-50, 1.0], dtype=np.float64)
    out = m._as_native_y(y)

    _assert_native(out)
    assert out.shape == (3,)
    assert np.all(np.isfinite(out))
    np.testing.assert_array_equal(out, y.astype(np.float32))


def test_subnormal_float32_value_is_not_an_error() -> None:
    y = np.array([1e-40, -1e-40], dtype=np.float64)
    out = m._as_native_y(y)

    _assert_native(out)
    assert np.all(np.isfinite(out))
    np.testing.assert_array_equal(out, y.astype(np.float32))


def test_precision_loss_is_not_an_error_and_uses_float32_rounding() -> None:
    y = np.array([0.1, 1.0 / 3.0, 123456789.123456789], dtype=np.float64)
    out = m._as_native_y(y)

    _assert_native(out)
    np.testing.assert_array_equal(out, y.astype(np.float32))


def test_large_integer_losing_precision_is_not_an_error() -> None:
    y = np.array([2**53 + 1, -(2**40) - 3], dtype=np.int64)
    out = m._as_native_y(y)

    _assert_native(out)
    np.testing.assert_array_equal(out, y.astype(np.float32))


def test_empty_1d_input_gives_empty_1d_float32_array() -> None:
    out = m._as_native_y(np.array([], dtype=np.float64))

    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert out.ndim == 1
    assert out.size == 0


def test_empty_multioutput_input_gives_empty_2d_float32_array() -> None:
    out = m._as_native_y(np.empty((0, 3), dtype=np.float64))

    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert out.ndim == 2
    assert out.size == 0
