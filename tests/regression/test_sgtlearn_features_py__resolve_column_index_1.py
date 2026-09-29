import re

import numpy as np
import pytest

import sgtlearn._features as m


def test_int_reference_returns_same_position() -> None:
    result = m._resolve_column_index(2, 5, None)
    assert result == 2
    assert type(result) is int


def test_int_reference_zero_is_valid() -> None:
    assert m._resolve_column_index(0, 1, None) == 0


def test_int_reference_last_position_is_valid() -> None:
    assert m._resolve_column_index(4, 5, None) == 4


def test_numpy_integer_reference_returns_plain_int() -> None:
    result = m._resolve_column_index(np.int64(3), 5, None)
    assert result == 3
    assert type(result) is int


def test_numpy_int32_reference_returns_plain_int() -> None:
    result = m._resolve_column_index(np.int32(1), 2, None)
    assert result == 1
    assert type(result) is int


def test_int_reference_ignores_column_names() -> None:
    assert m._resolve_column_index(1, 3, ["a", "b", "c"]) == 1


def test_int_reference_ignores_mismatched_column_names() -> None:
    assert m._resolve_column_index(2, 3, ["a"]) == 2


def test_int_reference_equal_to_n_features_raises() -> None:
    with pytest.raises(ValueError):
        m._resolve_column_index(5, 5, None)


def test_int_reference_greater_than_n_features_raises() -> None:
    with pytest.raises(ValueError):
        m._resolve_column_index(10, 5, ["a", "b", "c", "d", "e"])


def test_negative_int_reference_raises_no_wraparound() -> None:
    with pytest.raises(ValueError):
        m._resolve_column_index(-1, 5, None)


def test_negative_numpy_int_reference_raises() -> None:
    with pytest.raises(ValueError):
        m._resolve_column_index(np.int64(-2), 5, None)


def test_numpy_int_reference_out_of_range_raises() -> None:
    with pytest.raises(ValueError):
        m._resolve_column_index(np.int64(5), 5, None)


@pytest.mark.parametrize("col", [True, False])
def test_bool_reference_raises(col: bool) -> None:
    with pytest.raises(ValueError):
        m._resolve_column_index(col, 5, None)


def test_numpy_bool_reference_raises() -> None:
    with pytest.raises(ValueError):
        m._resolve_column_index(np.bool_(True), 5, None)


@pytest.mark.parametrize("col", [1.0, None, [0], (0,), b"a"])
def test_non_int_non_str_reference_raises(col: object) -> None:
    with pytest.raises(ValueError):
        m._resolve_column_index(col, 5, ["a", "b", "c", "d", "e"])


def test_name_reference_returns_position() -> None:
    result = m._resolve_column_index("b", 3, ["a", "b", "c"])
    assert result == 1
    assert type(result) is int


def test_name_reference_first_column() -> None:
    assert m._resolve_column_index("a", 3, ["a", "b", "c"]) == 0


def test_name_reference_last_column() -> None:
    assert m._resolve_column_index("c", 3, ["a", "b", "c"]) == 2


def test_name_reference_with_tuple_column_names() -> None:
    assert m._resolve_column_index("y", 2, ("x", "y")) == 1


def test_name_reference_compares_column_names_in_string_form() -> None:
    assert m._resolve_column_index("1", 3, [0, 1, 2]) == 1


def test_name_reference_with_numpy_array_column_names() -> None:
    assert m._resolve_column_index("b", 2, np.array(["a", "b"])) == 1


def test_name_reference_without_column_names_raises_mentioning_dataframe() -> None:
    with pytest.raises(ValueError, match=re.compile("dataframe", re.IGNORECASE)):
        m._resolve_column_index("a", 3, None)


def test_name_reference_without_column_names_mentions_column_names() -> None:
    with pytest.raises(ValueError, match="column_names"):
        m._resolve_column_index("a", 3, None)


def test_unknown_name_raises_with_name_in_message() -> None:
    with pytest.raises(ValueError, match=re.escape("missing_col")):
        m._resolve_column_index("missing_col", 3, ["a", "b", "c"])


def test_name_matching_is_case_sensitive() -> None:
    with pytest.raises(ValueError, match=re.escape("A")):
        m._resolve_column_index("A", 3, ["a", "b", "c"])


def test_name_matching_is_exact_not_substring() -> None:
    with pytest.raises(ValueError):
        m._resolve_column_index("col", 2, ["col1", "col2"])


def test_name_matching_does_not_strip_whitespace() -> None:
    with pytest.raises(ValueError):
        m._resolve_column_index("a", 2, [" a", "b"])


def test_name_reference_with_empty_column_names_raises() -> None:
    with pytest.raises(ValueError):
        m._resolve_column_index("a", 0, [])


def test_duplicate_name_match_raises() -> None:
    with pytest.raises(ValueError):
        m._resolve_column_index("a", 3, ["a", "b", "a"])


def test_duplicate_name_via_string_form_raises() -> None:
    with pytest.raises(ValueError):
        m._resolve_column_index("1", 2, [1, "1"])


def test_non_duplicate_name_resolves_when_other_names_duplicated() -> None:
    assert m._resolve_column_index("b", 3, ["a", "b", "a"]) == 1


def test_name_resolving_beyond_n_features_raises() -> None:
    with pytest.raises(ValueError):
        m._resolve_column_index("d", 2, ["a", "b", "c", "d"])


def test_name_resolving_at_n_features_raises() -> None:
    with pytest.raises(ValueError):
        m._resolve_column_index("c", 2, ["a", "b", "c"])


def test_name_within_n_features_resolves_even_if_extra_names() -> None:
    assert m._resolve_column_index("a", 2, ["a", "b", "c"]) == 0


def test_column_names_not_mutated() -> None:
    names = ["a", "b", "c"]
    m._resolve_column_index("b", 3, names)
    assert names == ["a", "b", "c"]
