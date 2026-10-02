import copy

import numpy as np
import pytest

import sgtlearn.base as m
from sgtlearn import ProcessedFeatures, configure_feature_dict


def _values_equal(a, b) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return bool(np.array_equal(np.asarray(a), np.asarray(b)))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_values_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_values_equal(a[k], b[k]) for k in a)
    return a == b


def _assert_same_processed(actual, expected) -> None:
    assert isinstance(actual, ProcessedFeatures)
    assert tuple(actual.logical_names) == tuple(expected.logical_names)
    assert len(actual.features) == len(expected.features)
    for got, want in zip(actual.features, expected.features):
        assert _values_equal(got, want)


def _snapshot(pf: ProcessedFeatures):
    return copy.deepcopy(list(pf.features)), tuple(pf.logical_names)


# --- processed_features given -------------------------------------------------


def test_returns_given_processed_features_identical_object() -> None:
    pf = configure_feature_dict(3)

    result = m._configure_processed_features(
        3, feature_dict=None, processed_features=pf, column_names=None
    )

    assert result is pf


def test_returns_subclass_instance_of_processed_features_unchanged() -> None:
    class SubProcessedFeatures(ProcessedFeatures):
        pass

    base = configure_feature_dict(3)
    pf = SubProcessedFeatures(features=base.features, logical_names=base.logical_names)

    result = m._configure_processed_features(
        3, feature_dict=None, processed_features=pf, column_names=None
    )

    assert result is pf
    assert type(result) is SubProcessedFeatures


def test_given_processed_features_not_revalidated_against_n_features() -> None:
    pf = configure_feature_dict(3)

    result = m._configure_processed_features(
        50, feature_dict=None, processed_features=pf, column_names=None
    )

    assert result is pf


def test_given_processed_features_not_revalidated_against_column_names() -> None:
    pf = configure_feature_dict(3)

    result = m._configure_processed_features(
        3,
        feature_dict=None,
        processed_features=pf,
        column_names=["only_one_name"],
    )

    assert result is pf


def test_valid_feature_dict_ignored_when_processed_features_given() -> None:
    pf = configure_feature_dict(3)

    result = m._configure_processed_features(
        3,
        feature_dict={"grouped": [0, 1, 2]},
        processed_features=pf,
        column_names=None,
    )

    assert result is pf


def test_invalid_out_of_range_feature_dict_ignored_when_processed_features_given() -> None:
    pf = configure_feature_dict(3)

    result = m._configure_processed_features(
        3,
        feature_dict={"bad": [99]},
        processed_features=pf,
        column_names=None,
    )

    assert result is pf


def test_invalid_name_based_feature_dict_ignored_when_processed_features_given() -> None:
    pf = configure_feature_dict(3)

    result = m._configure_processed_features(
        3,
        feature_dict={"bad": ["does_not_exist"]},
        processed_features=pf,
        column_names=None,
    )

    assert result is pf


def test_given_processed_features_is_not_mutated() -> None:
    pf = configure_feature_dict(4, {"a": [0, 1], "b": [2, 3]})
    before_features, before_names = _snapshot(pf)

    m._configure_processed_features(
        4, feature_dict={"x": [0]}, processed_features=pf, column_names=None
    )

    assert _values_equal(list(pf.features), before_features)
    assert tuple(pf.logical_names) == before_names


# --- processed_features of wrong type -----------------------------------------


@pytest.mark.parametrize(
    "bad_value",
    [
        [{"name": "x0"}],
        {"features": []},
        "processed_features",
        42,
    ],
    ids=["list_of_dicts", "dict", "str", "int"],
)
def test_non_processed_features_value_raises_type_error(bad_value) -> None:
    with pytest.raises(TypeError):
        m._configure_processed_features(
            3, feature_dict=None, processed_features=bad_value, column_names=None
        )


@pytest.mark.parametrize(
    "bad_value",
    [[{"name": "x0"}], {"features": []}, "abc"],
    ids=["list", "dict", "str"],
)
def test_type_error_message_names_parameter_and_received_type(bad_value) -> None:
    with pytest.raises(TypeError) as excinfo:
        m._configure_processed_features(
            3, feature_dict=None, processed_features=bad_value, column_names=None
        )

    message = str(excinfo.value)
    assert "processed_features" in message
    assert type(bad_value).__name__ in message


def test_type_check_happens_before_invalid_feature_dict_is_resolved() -> None:
    with pytest.raises(TypeError):
        m._configure_processed_features(
            3,
            feature_dict={"bad": [99]},
            processed_features=[{"name": "x0"}],
            column_names=None,
        )


def test_native_feature_list_from_to_native_is_rejected() -> None:
    native = configure_feature_dict(3).to_native()

    with pytest.raises(TypeError):
        m._configure_processed_features(
            3, feature_dict=None, processed_features=native, column_names=None
        )


# --- processed_features is None: delegation -----------------------------------


def test_no_grouping_matches_configure_feature_dict_default() -> None:
    expected = configure_feature_dict(3, None, column_names=None)

    result = m._configure_processed_features(
        3, feature_dict=None, processed_features=None, column_names=None
    )

    _assert_same_processed(result, expected)


def test_no_grouping_with_column_names_matches_configure_feature_dict() -> None:
    names = ["a", "b", "c"]
    expected = configure_feature_dict(3, None, column_names=names)

    result = m._configure_processed_features(
        3, feature_dict=None, processed_features=None, column_names=names
    )

    _assert_same_processed(result, expected)


def test_integer_grouping_matches_configure_feature_dict() -> None:
    fd = {"first": [0, 1], "second": [2]}
    expected = configure_feature_dict(3, fd, column_names=None)

    result = m._configure_processed_features(
        3, feature_dict=fd, processed_features=None, column_names=None
    )

    _assert_same_processed(result, expected)


def test_name_based_grouping_matches_configure_feature_dict() -> None:
    names = ["age", "height", "weight"]
    fd = {"body": ["height", "weight"], "age": ["age"]}
    expected = configure_feature_dict(3, fd, column_names=names)

    result = m._configure_processed_features(
        3, feature_dict=fd, processed_features=None, column_names=names
    )

    _assert_same_processed(result, expected)


def test_integer_logical_name_grouping_matches_configure_feature_dict() -> None:
    fd = {0: [0, 1], 1: [2, 3]}
    expected = configure_feature_dict(4, fd, column_names=None)

    result = m._configure_processed_features(
        4, feature_dict=fd, processed_features=None, column_names=None
    )

    _assert_same_processed(result, expected)


def test_result_is_processed_features_instance_when_resolved() -> None:
    result = m._configure_processed_features(
        2, feature_dict=None, processed_features=None, column_names=None
    )

    assert isinstance(result, ProcessedFeatures)


# --- processed_features is None: error propagation ----------------------------


def test_out_of_range_index_raises_value_error() -> None:
    with pytest.raises(ValueError):
        m._configure_processed_features(
            3, feature_dict={"g": [5]}, processed_features=None, column_names=None
        )


def test_unknown_column_name_raises_value_error() -> None:
    with pytest.raises(ValueError):
        m._configure_processed_features(
            3,
            feature_dict={"g": ["missing"]},
            processed_features=None,
            column_names=["a", "b", "c"],
        )


def test_duplicate_column_across_groups_raises_value_error() -> None:
    with pytest.raises(ValueError):
        m._configure_processed_features(
            3,
            feature_dict={"g1": [0], "g2": [0]},
            processed_features=None,
            column_names=None,
        )


def test_name_based_columns_without_column_names_propagates_same_error() -> None:
    fd = {"g": ["a"]}
    with pytest.raises(Exception) as direct:
        configure_feature_dict(3, fd, column_names=None)

    with pytest.raises(type(direct.value)):
        m._configure_processed_features(
            3, feature_dict=fd, processed_features=None, column_names=None
        )


def test_out_of_range_error_type_matches_configure_feature_dict() -> None:
    fd = {"g": [10]}
    with pytest.raises(Exception) as direct:
        configure_feature_dict(3, fd, column_names=None)

    with pytest.raises(Exception) as wrapped:
        m._configure_processed_features(
            3, feature_dict=fd, processed_features=None, column_names=None
        )

    assert type(wrapped.value) is type(direct.value)


# --- no side effects ----------------------------------------------------------


def test_feature_dict_is_not_mutated() -> None:
    fd = {"first": [0, 1], "second": [2]}
    before = copy.deepcopy(fd)

    m._configure_processed_features(
        3, feature_dict=fd, processed_features=None, column_names=None
    )

    assert fd == before


def test_column_names_list_is_not_mutated() -> None:
    names = ["age", "height", "weight"]
    before = list(names)

    m._configure_processed_features(
        3,
        feature_dict={"body": ["height", "weight"]},
        processed_features=None,
        column_names=names,
    )

    assert names == before
