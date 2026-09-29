import re

import numpy as np
import pytest

import sgtlearn._weights as m


SHAPE_MSG = re.escape("(n_samples,)")


def test_valid_float_weights_return_none() -> None:
    sw = np.array([0.5, 1.0, 2.0])
    assert m._validate_sample_weight_array(sw, 3) is None


def test_valid_integer_weights_accepted() -> None:
    sw = np.array([1, 2, 3], dtype=np.int64)
    assert m._validate_sample_weight_array(sw, 3) is None


def test_zero_weights_allowed_with_one_positive() -> None:
    sw = np.array([0.0, 0.0, 1.0, 0.0])
    assert m._validate_sample_weight_array(sw, 4) is None


def test_single_positive_weight_with_one_sample_is_valid() -> None:
    sw = np.array([3.0])
    assert m._validate_sample_weight_array(sw, 1) is None


def test_input_array_not_modified() -> None:
    sw = np.array([0.0, 1.5, 2.0])
    before = sw.copy()
    m._validate_sample_weight_array(sw, 3)
    np.testing.assert_array_equal(sw, before)
    assert sw.dtype == before.dtype
    assert sw.shape == before.shape


def test_zero_dim_scalar_raises_shape_error() -> None:
    with pytest.raises(ValueError, match=SHAPE_MSG):
        m._validate_sample_weight_array(np.array(1.0), 1)


def test_two_dimensional_array_raises_shape_error() -> None:
    with pytest.raises(ValueError, match=SHAPE_MSG):
        m._validate_sample_weight_array(np.ones((3, 1)), 3)


def test_too_short_array_raises_with_lengths_in_message() -> None:
    with pytest.raises(ValueError) as exc:
        m._validate_sample_weight_array(np.ones(3), 5)
    msg = str(exc.value)
    assert "3" in msg
    assert "5" in msg
    assert "(n_samples,)" in msg


def test_too_long_array_raises_with_lengths_in_message() -> None:
    with pytest.raises(ValueError) as exc:
        m._validate_sample_weight_array(np.ones(7), 4)
    msg = str(exc.value)
    assert "7" in msg
    assert "4" in msg
    assert "(n_samples,)" in msg


def test_nan_weight_raises() -> None:
    with pytest.raises(ValueError):
        m._validate_sample_weight_array(np.array([1.0, np.nan, 2.0]), 3)


def test_positive_inf_weight_raises() -> None:
    with pytest.raises(ValueError):
        m._validate_sample_weight_array(np.array([1.0, np.inf]), 2)


def test_negative_inf_weight_raises() -> None:
    with pytest.raises(ValueError):
        m._validate_sample_weight_array(np.array([1.0, -np.inf]), 2)


def test_negative_float_weight_raises_non_negative_error() -> None:
    with pytest.raises(ValueError, match="sample_weight must be non-negative"):
        m._validate_sample_weight_array(np.array([1.0, -0.5, 2.0]), 3)


def test_negative_integer_weight_raises_non_negative_error() -> None:
    with pytest.raises(ValueError, match="sample_weight must be non-negative"):
        m._validate_sample_weight_array(np.array([2, -1], dtype=np.int64), 2)


def test_all_zero_weights_raise_no_positive_error() -> None:
    with pytest.raises(
        ValueError, match="sample_weight must contain at least one positive value"
    ):
        m._validate_sample_weight_array(np.zeros(4), 4)


def test_all_zero_integer_weights_raise_no_positive_error() -> None:
    with pytest.raises(
        ValueError, match="sample_weight must contain at least one positive value"
    ):
        m._validate_sample_weight_array(np.zeros(2, dtype=np.int64), 2)


def test_empty_array_with_zero_samples_raises_no_positive_error() -> None:
    with pytest.raises(
        ValueError, match="sample_weight must contain at least one positive value"
    ):
        m._validate_sample_weight_array(np.array([], dtype=float), 0)


def test_length_check_precedes_finiteness_check() -> None:
    with pytest.raises(ValueError, match=SHAPE_MSG):
        m._validate_sample_weight_array(np.array([np.nan, 1.0]), 3)


def test_length_check_precedes_negativity_check() -> None:
    with pytest.raises(ValueError, match=SHAPE_MSG):
        m._validate_sample_weight_array(np.array([-1.0, 1.0]), 3)


def test_finiteness_check_precedes_negativity_check() -> None:
    with pytest.raises(ValueError) as exc:
        m._validate_sample_weight_array(np.array([np.nan, -1.0]), 2)
    assert "non-negative" not in str(exc.value)


def test_finiteness_check_precedes_positivity_check() -> None:
    with pytest.raises(ValueError) as exc:
        m._validate_sample_weight_array(np.array([np.nan, 0.0]), 2)
    assert "at least one positive" not in str(exc.value)


def test_negativity_check_precedes_positivity_check() -> None:
    with pytest.raises(ValueError, match="sample_weight must be non-negative"):
        m._validate_sample_weight_array(np.array([-1.0, 0.0]), 2)
