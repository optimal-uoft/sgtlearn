import copy
import math

import numpy as np
import pytest

import sgtlearn.base as m

INF = float("inf")


def _num(**kw):
    node = {"is_leaf": False, "is_categorical": False}
    node.update(kw)
    return node


def _cat(**kw):
    node = {"is_leaf": False, "is_categorical": True}
    node.update(kw)
    return node


def _run(node):
    tree = {"nodes": [node]}
    out = m._normalize_tree_export(tree)
    assert out is tree
    return out["nodes"][0]


def test_missing_nodes_returns_same_tree_unchanged():
    tree = {"foo": 1}
    out = m._normalize_tree_export(tree)
    assert out is tree
    assert tree == {"foo": 1}


def test_empty_nodes_returns_tree_unchanged():
    tree = {"nodes": []}
    out = m._normalize_tree_export(tree)
    assert out is tree
    assert tree == {"nodes": []}


def test_returns_same_object_modified_in_place():
    node = _num(thresholds=[1.0, INF], bin_to_partition=[0, 1, 1])
    tree = {"nodes": [node]}
    out = m._normalize_tree_export(tree)
    assert out is tree
    assert out["nodes"][0] is node
    assert node["thresholds"] == [1.0]


def test_leaf_node_left_untouched():
    node = {"is_leaf": True, "thresholds": [1.0, INF], "bin_to_partition": [0, 1, 1]}
    snap = copy.deepcopy(node)
    assert _run(node) == snap


def test_node_missing_is_leaf_treated_as_leaf():
    node = {"is_categorical": False, "thresholds": [1.0, INF], "bin_to_partition": [0, 1, 1]}
    snap = copy.deepcopy(node)
    assert _run(node) == snap


def test_pair_routing_node_left_untouched():
    node = _num(routing_kind="pair", thresholds=[1.0, INF], bin_to_partition=[0, 1, 1])
    snap = copy.deepcopy(node)
    assert _run(node) == snap


def test_pair_routing_categorical_node_left_untouched():
    node = _cat(routing_kind="pair", bin_to_partition=[0, 1, 0])
    snap = copy.deepcopy(node)
    assert _run(node) == snap


# ---------------- categorical ----------------

def test_categorical_sets_nan_partition_from_last_bin_and_removes_it():
    node = _run(_cat(bin_to_partition=[0, 1, 0, 1]))
    assert node["nan_prediction_partition"] == 1
    assert node["bin_to_partition"] == [0, 1, 0]


def test_categorical_trims_aux_arrays_matching_original_length():
    node = _run(_cat(
        bin_to_partition=[0, 1, 1],
        bin_categories=["a", "b", "nan"],
        bin_sample_counts=[5, 6, 7],
        bin_counts=[1, 2, 3],
        bin_weights=[0.1, 0.2, 0.3],
    ))
    assert node["bin_categories"] == ["a", "b"]
    assert node["bin_sample_counts"] == [5, 6]
    assert node["bin_counts"] == [1, 2]
    assert node["bin_weights"] == [0.1, 0.2]


def test_categorical_keeps_aux_arrays_with_other_length():
    node = _run(_cat(
        bin_to_partition=[0, 1, 1],
        bin_categories=["a", "b"],
        bin_sample_counts=[5, 6],
        bin_counts=[1, 2, 3, 4],
        bin_weights=[0.1],
    ))
    assert node["bin_categories"] == ["a", "b"]
    assert node["bin_sample_counts"] == [5, 6]
    assert node["bin_counts"] == [1, 2, 3, 4]
    assert node["bin_weights"] == [0.1]


def test_categorical_thresholds_not_touched():
    node = _run(_cat(bin_to_partition=[0, 1, 0], thresholds=[1.0, INF]))
    assert node["thresholds"] == [1.0, INF]


def test_categorical_empty_bin_to_partition_left_unchanged():
    node = _cat(bin_to_partition=[], bin_categories=["a"], thresholds=[INF])
    snap = copy.deepcopy(node)
    assert _run(node) == snap


def test_categorical_missing_bin_to_partition_left_unchanged():
    node = _cat(bin_categories=["a"], bin_counts=[3])
    snap = copy.deepcopy(node)
    assert _run(node) == snap


# ---------------- numeric ----------------

def test_numeric_trailing_inf_thresholds_removed():
    node = _run(_num(thresholds=[1.0, 2.0, INF, INF]))
    assert node["thresholds"] == [1.0, 2.0]


def test_numeric_non_trailing_inf_kept():
    node = _run(_num(thresholds=[1.0, INF, 2.0]))
    assert node["thresholds"] == [1.0, INF, 2.0]


def test_numeric_negative_inf_not_a_sentinel():
    node = _run(_num(thresholds=[1.0, -INF]))
    assert node["thresholds"] == [1.0, -INF]


def test_numeric_nan_threshold_kept():
    node = _run(_num(thresholds=[float("nan"), 1.0, INF]))
    th = node["thresholds"]
    assert len(th) == 2
    assert math.isnan(th[0])
    assert th[1] == 1.0


def test_numeric_trailing_nan_kept():
    node = _run(_num(thresholds=[1.0, float("nan")]))
    th = node["thresholds"]
    assert len(th) == 2
    assert th[0] == 1.0
    assert math.isnan(th[1])


def test_numeric_nan_bin_detected_sets_partition_and_shortens():
    node = _run(_num(thresholds=[1.0, 2.0, INF], bin_to_partition=[0, 1, 0, 1]))
    assert node["thresholds"] == [1.0, 2.0]
    assert node["nan_prediction_partition"] == 1
    assert node["bin_to_partition"] == [0, 1, 0]


def test_numeric_nan_bin_trims_aux_arrays_matching_original_length():
    node = _run(_num(
        thresholds=[1.0],
        bin_to_partition=[0, 1, 0],
        bin_sample_counts=[4, 5, 6],
        bin_counts=[1, 2, 3],
        bin_weights=[0.5, 0.25, 0.25],
    ))
    assert node["nan_prediction_partition"] == 0
    assert node["bin_to_partition"] == [0, 1]
    assert node["bin_sample_counts"] == [4, 5]
    assert node["bin_counts"] == [1, 2]
    assert node["bin_weights"] == [0.5, 0.25]


def test_numeric_nan_bin_keeps_aux_arrays_with_other_length():
    node = _run(_num(
        thresholds=[1.0],
        bin_to_partition=[0, 1, 0],
        bin_sample_counts=[4, 5],
        bin_counts=[1, 2, 3, 4],
        bin_weights=[0.5],
    ))
    assert node["bin_sample_counts"] == [4, 5]
    assert node["bin_counts"] == [1, 2, 3, 4]
    assert node["bin_weights"] == [0.5]


def test_numeric_nan_bin_detection_uses_trimmed_thresholds():
    # original thresholds length 3 -> trimmed to 1; 3 bins == 1 + 2
    node = _run(_num(thresholds=[1.0, INF, INF], bin_to_partition=[1, 0, 1]))
    assert node["nan_prediction_partition"] == 1
    assert node["bin_to_partition"] == [1, 0]


def test_numeric_already_normalized_bin_count_only_trims_thresholds():
    node = _num(
        thresholds=[1.0, INF],
        bin_to_partition=[0, 1],
        bin_counts=[1, 2],
        bin_sample_counts=[3, 4],
        bin_weights=[0.5, 0.5],
    )
    out = _run(node)
    assert out["thresholds"] == [1.0]
    assert "nan_prediction_partition" not in out
    assert out["bin_to_partition"] == [0, 1]
    assert out["bin_counts"] == [1, 2]
    assert out["bin_sample_counts"] == [3, 4]
    assert out["bin_weights"] == [0.5, 0.5]


def test_numeric_other_bin_length_no_nan_partition():
    node = _run(_num(thresholds=[1.0], bin_to_partition=[0, 1, 0, 1, 0], bin_counts=[1, 2, 3, 4, 5]))
    assert "nan_prediction_partition" not in node
    assert node["bin_to_partition"] == [0, 1, 0, 1, 0]
    assert node["bin_counts"] == [1, 2, 3, 4, 5]


def test_numeric_missing_bin_to_partition_only_trims_thresholds():
    node = _run(_num(thresholds=[1.0, INF], bin_counts=[1, 2, 3]))
    assert node["thresholds"] == [1.0]
    assert "nan_prediction_partition" not in node
    assert "bin_to_partition" not in node
    assert node["bin_counts"] == [1, 2, 3]


def test_numeric_empty_bin_to_partition_only_trims_thresholds():
    node = _run(_num(thresholds=[2.0, INF], bin_to_partition=[]))
    assert node["thresholds"] == [2.0]
    assert "nan_prediction_partition" not in node
    assert node["bin_to_partition"] == []


def test_numeric_all_inf_thresholds_become_empty():
    node = _run(_num(thresholds=[INF, INF]))
    assert node["thresholds"] == []


def test_numeric_all_inf_with_two_bins_reads_finite_plus_nan():
    node = _run(_num(thresholds=[INF, INF, INF], bin_to_partition=[0, 1]))
    assert node["thresholds"] == []
    assert node["nan_prediction_partition"] == 1
    assert node["bin_to_partition"] == [0]


def test_numeric_missing_thresholds_treated_as_empty():
    node = _run(_num(bin_to_partition=[1, 0]))
    assert node["nan_prediction_partition"] == 0
    assert node["bin_to_partition"] == [1]


def test_numeric_none_thresholds_treated_as_empty():
    node = _run(_num(thresholds=None, bin_to_partition=[0, 1]))
    assert node["nan_prediction_partition"] == 1
    assert node["bin_to_partition"] == [0]


def test_numeric_missing_thresholds_no_bins_raises_nothing():
    node = _run(_num())
    assert "nan_prediction_partition" not in node


def test_numeric_normalization_is_idempotent():
    node = _num(
        thresholds=[1.0, 2.0, INF],
        bin_to_partition=[0, 1, 0, 1],
        bin_counts=[1, 2, 3, 4],
        bin_sample_counts=[5, 6, 7, 8],
        bin_weights=[0.1, 0.2, 0.3, 0.4],
    )
    tree = {"nodes": [node]}
    m._normalize_tree_export(tree)
    snap = copy.deepcopy(tree)
    m._normalize_tree_export(tree)
    assert tree == snap


def test_mixed_nodes_processed_independently():
    leaf = {"is_leaf": True, "value": 3}
    num = _num(thresholds=[1.0, INF], bin_to_partition=[0, 1, 1])
    cat = _cat(bin_to_partition=[0, 1, 0])
    leaf_snap = copy.deepcopy(leaf)
    tree = {"nodes": [leaf, num, cat]}
    m._normalize_tree_export(tree)
    assert tree["nodes"][0] == leaf_snap
    assert tree["nodes"][1]["thresholds"] == [1.0]
    assert tree["nodes"][1]["nan_prediction_partition"] == 1
    assert tree["nodes"][1]["bin_to_partition"] == [0, 1]
    assert tree["nodes"][2]["nan_prediction_partition"] == 0
    assert tree["nodes"][2]["bin_to_partition"] == [0, 1]
