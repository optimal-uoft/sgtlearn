import copy

import pytest

import sgtlearn._features as m


def _simplify(features):
    return [(list(f["indices"]), f["type"]) for f in features]


def test_empty_dict_gives_continuous_singletons() -> None:
    features, names = m._feature_dict_to_features(3, {})
    assert _simplify(features) == [
        ([0], "continuous"),
        ([1], "continuous"),
        ([2], "continuous"),
    ]
    assert names == ("0", "1", "2")


def test_empty_dict_orders_by_string_key() -> None:
    features, names = m._feature_dict_to_features(12, {})
    expected = sorted(range(12), key=str)
    assert names == tuple(str(i) for i in expected)
    assert [f["indices"] for f in features] == [[i] for i in expected]


def test_return_types() -> None:
    features, names = m._feature_dict_to_features(3, {"g": [0, 1]})
    assert isinstance(features, list)
    assert isinstance(names, tuple)
    assert all(isinstance(n, str) for n in names)
    assert all(isinstance(f, dict) for f in features)
    for f in features:
        assert all(type(i) is int for i in f["indices"])
    assert len(features) == len(names)


def test_multi_column_group_is_categorical_and_unlisted_filled() -> None:
    features, names = m._feature_dict_to_features(3, {0: [0, 1]})
    assert names == ("0", "2")
    assert _simplify(features) == [([0, 1], "categorical"), ([2], "continuous")]


def test_single_column_group_is_continuous() -> None:
    features, names = m._feature_dict_to_features(2, {"g": [1]})
    assert names == ("0", "g")
    assert _simplify(features) == [([0], "continuous"), ([1], "continuous")]


def test_str_keys_come_after_int_keys() -> None:
    features, names = m._feature_dict_to_features(4, {"a": [1, 2]})
    assert names == ("0", "3", "a")
    assert _simplify(features) == [
        ([0], "continuous"),
        ([3], "continuous"),
        ([1, 2], "categorical"),
    ]


def test_str_keys_sorted_lexicographically() -> None:
    features, names = m._feature_dict_to_features(4, {"b": [0, 1], "a": [2, 3]})
    assert names == ("a", "b")
    assert _simplify(features) == [([2, 3], "categorical"), ([0, 1], "categorical")]


def test_indices_keep_user_order() -> None:
    features, names = m._feature_dict_to_features(3, {"g": [2, 0, 1]})
    assert names == ("g",)
    assert features[0]["indices"] == [2, 0, 1]
    assert features[0]["type"] == "categorical"


def test_int_keys_ordered_by_string_ten_before_nine() -> None:
    features, names = m._feature_dict_to_features(11, {10: [9, 10]})
    expected = tuple(sorted([str(i) for i in range(9)] + ["10"]))
    assert names == expected
    assert names.index("10") < names.index("8")
    idx = names.index("10")
    assert features[idx]["indices"] == [9, 10]
    assert features[idx]["type"] == "categorical"


def test_user_and_auto_int_keys_interleave_by_string() -> None:
    features, names = m._feature_dict_to_features(4, {3: [2, 3]})
    assert names == ("0", "1", "3")
    assert _simplify(features) == [
        ([0], "continuous"),
        ([1], "continuous"),
        ([2, 3], "categorical"),
    ]


def test_int_key_out_of_range_never_collides() -> None:
    features, names = m._feature_dict_to_features(3, {5: [0, 1]})
    assert names == ("2", "5")
    assert _simplify(features) == [([2], "continuous"), ([0, 1], "categorical")]


def test_int_key_matching_listed_column_is_allowed() -> None:
    features, names = m._feature_dict_to_features(3, {1: [1, 2]})
    assert names == ("0", "1")
    assert _simplify(features) == [([0], "continuous"), ([1, 2], "categorical")]


def test_int_key_matching_unlisted_column_raises_collide() -> None:
    with pytest.raises(ValueError, match="collide") as exc:
        m._feature_dict_to_features(3, {1: [0, 2]})
    assert "1" in str(exc.value)


def test_multiple_colliding_keys_listed_in_message() -> None:
    with pytest.raises(ValueError, match="collide") as exc:
        m._feature_dict_to_features(6, {3: [0, 1], 4: [2, 5]})
    msg = str(exc.value)
    assert "3" in msg
    assert "4" in msg


def test_str_key_equal_to_auto_name_does_not_collide() -> None:
    features, names = m._feature_dict_to_features(3, {"0": [1, 2]})
    assert names == ("0", "0")
    assert _simplify(features) == [([0], "continuous"), ([1, 2], "categorical")]


def test_duplicate_index_within_group_raises() -> None:
    with pytest.raises(ValueError, match="Feature indices must be unique"):
        m._feature_dict_to_features(3, {"g": [0, 0]})


def test_duplicate_index_across_groups_raises() -> None:
    with pytest.raises(ValueError, match="Feature indices must be unique"):
        m._feature_dict_to_features(3, {"a": [0, 1], "b": [1, 2]})


def test_duplicate_via_name_and_index_raises() -> None:
    with pytest.raises(ValueError, match="Feature indices must be unique"):
        m._feature_dict_to_features(2, {"g": ["a", 0]}, column_names=["a", "b"])


def test_column_names_resolve_to_indices() -> None:
    features, names = m._feature_dict_to_features(
        3, {"g": ["c", "a"]}, column_names=["a", "b", "c"]
    )
    assert names == ("1", "g")
    assert _simplify(features) == [([1], "continuous"), ([2, 0], "categorical")]


def test_out_of_range_index_error_propagates() -> None:
    with pytest.raises(Exception) as ref:
        m._resolve_column_index(5, 3, None)
    with pytest.raises(type(ref.value)):
        m._feature_dict_to_features(3, {"g": [0, 5]})


def test_unknown_name_error_propagates() -> None:
    cols = ["a", "b", "c"]
    with pytest.raises(Exception) as ref:
        m._resolve_column_index("zz", 3, cols)
    with pytest.raises(type(ref.value)):
        m._feature_dict_to_features(3, {"g": ["a", "zz"]}, column_names=cols)


def test_input_mapping_not_modified() -> None:
    fd = {"g": [2, 0], 5: [1, 3]}
    snapshot = copy.deepcopy(fd)
    m._feature_dict_to_features(5, fd)
    assert fd == snapshot


def test_every_column_appears_exactly_once() -> None:
    features, names = m._feature_dict_to_features(7, {"x": [6, 1], 0: [0, 3]})
    all_idx = [i for f in features for i in f["indices"]]
    assert sorted(all_idx) == list(range(7))
    assert names == ("0", "2", "4", "5", "x")
