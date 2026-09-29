import math
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

import sgtlearn.base as m


def _assert_rejected(value) -> None:
    with pytest.raises(ValueError, match="tao_pair_scale"):
        m._validate_tao_pair_scale(value)


def test_positive_float_returned_unchanged_as_float() -> None:
    out = m._validate_tao_pair_scale(0.25)
    assert type(out) is float
    assert out == 0.25


def test_positive_int_converted_to_float() -> None:
    out = m._validate_tao_pair_scale(3)
    assert type(out) is float
    assert out == 3.0


def test_zero_int_returns_zero_float() -> None:
    out = m._validate_tao_pair_scale(0)
    assert type(out) is float
    assert out == 0.0


def test_zero_float_returns_zero_float() -> None:
    out = m._validate_tao_pair_scale(0.0)
    assert type(out) is float
    assert out == 0.0


def test_negative_zero_is_accepted() -> None:
    out = m._validate_tao_pair_scale(-0.0)
    assert type(out) is float
    assert out == 0.0


def test_numpy_float64_converted_to_python_float() -> None:
    out = m._validate_tao_pair_scale(np.float64(1.5))
    assert type(out) is float
    assert out == 1.5


def test_numpy_float32_converted_without_extra_rounding() -> None:
    value = np.float32(0.1)
    out = m._validate_tao_pair_scale(value)
    assert type(out) is float
    assert out == float(value)


def test_numpy_int64_converted_to_python_float() -> None:
    out = m._validate_tao_pair_scale(np.int64(7))
    assert type(out) is float
    assert out == 7.0


def test_numpy_uint8_converted_to_python_float() -> None:
    out = m._validate_tao_pair_scale(np.uint8(4))
    assert type(out) is float
    assert out == 4.0


def test_fraction_converted_to_python_float() -> None:
    out = m._validate_tao_pair_scale(Fraction(1, 3))
    assert type(out) is float
    assert out == float(Fraction(1, 3))


def test_large_finite_value_not_clipped() -> None:
    out = m._validate_tao_pair_scale(1e300)
    assert out == 1e300


def test_max_float_is_accepted() -> None:
    out = m._validate_tao_pair_scale(np.finfo(np.float64).max)
    assert out == float(np.finfo(np.float64).max)


def test_smallest_subnormal_not_rounded_to_zero() -> None:
    out = m._validate_tao_pair_scale(5e-324)
    assert out == 5e-324
    assert out > 0.0


def test_large_int_that_fits_float_is_accepted() -> None:
    out = m._validate_tao_pair_scale(10**20)
    assert out == float(10**20)


def test_deterministic_repeated_calls() -> None:
    assert m._validate_tao_pair_scale(2.5) == m._validate_tao_pair_scale(2.5)


def test_true_rejected() -> None:
    _assert_rejected(True)


def test_false_rejected() -> None:
    _assert_rejected(False)


def test_numpy_bool_true_rejected() -> None:
    _assert_rejected(np.bool_(True))


def test_numpy_bool_false_rejected() -> None:
    _assert_rejected(np.bool_(False))


def test_string_rejected() -> None:
    _assert_rejected("1.0")


def test_none_rejected() -> None:
    _assert_rejected(None)


def test_complex_rejected() -> None:
    _assert_rejected(1 + 0j)


def test_numpy_complex_rejected() -> None:
    _assert_rejected(np.complex128(1.0))


def test_decimal_rejected() -> None:
    _assert_rejected(Decimal("1.0"))


def test_list_rejected() -> None:
    _assert_rejected([1.0])


def test_numpy_array_rejected() -> None:
    _assert_rejected(np.array([1.0]))


def test_nan_rejected() -> None:
    _assert_rejected(float("nan"))


def test_numpy_nan_rejected() -> None:
    _assert_rejected(np.float64(np.nan))


def test_positive_infinity_rejected() -> None:
    _assert_rejected(math.inf)


def test_negative_infinity_rejected() -> None:
    _assert_rejected(-math.inf)


def test_numpy_infinity_rejected() -> None:
    _assert_rejected(np.float32(np.inf))


def test_negative_float_rejected() -> None:
    _assert_rejected(-0.5)


def test_negative_int_rejected() -> None:
    _assert_rejected(-1)


def test_tiny_negative_rejected() -> None:
    _assert_rejected(-5e-324)


def test_negative_fraction_rejected() -> None:
    _assert_rejected(Fraction(-1, 2))


def test_negative_numpy_int_rejected() -> None:
    _assert_rejected(np.int32(-3))


def test_huge_int_raises_value_error_not_overflow() -> None:
    _assert_rejected(10**400)


def test_huge_negative_int_raises_value_error() -> None:
    _assert_rejected(-(10**400))


def test_huge_fraction_raises_value_error_not_overflow() -> None:
    _assert_rejected(Fraction(10**400, 1))
