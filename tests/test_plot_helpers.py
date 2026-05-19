"""Unit tests for ``sgtlearn._export`` private helpers."""
from __future__ import annotations

import pytest

from sgtlearn._export import _merge_routing_regions


def test_merge_two_bins_same_partition_merges():
    regions = _merge_routing_regions(
        thresholds=[0.5], bin_to_partition=[0, 0], x_min=-1.0, x_max=1.0
    )
    assert regions == [(-1.0, 1.0, 0)]


def test_merge_two_bins_different_partition_two_slabs():
    regions = _merge_routing_regions(
        thresholds=[0.5], bin_to_partition=[0, 1], x_min=-1.0, x_max=1.0
    )
    assert regions == [(-1.0, 0.5, 0), (0.5, 1.0, 1)]


def test_merge_non_contiguous_same_partition_keeps_separate():
    regions = _merge_routing_regions(
        thresholds=[-0.5, 0.0, 0.5],
        bin_to_partition=[0, 1, 0, 1],
        x_min=-1.0,
        x_max=1.0,
    )
    assert regions == [
        (-1.0, -0.5, 0),
        (-0.5, 0.0, 1),
        (0.0, 0.5, 0),
        (0.5, 1.0, 1),
    ]


def test_merge_consecutive_runs_merge_within_run():
    regions = _merge_routing_regions(
        thresholds=[-0.5, 0.0, 0.5, 0.75],
        bin_to_partition=[0, 0, 1, 1, 0],
        x_min=-1.0,
        x_max=1.0,
    )
    assert regions == [
        (-1.0, 0.0, 0),
        (0.0, 0.75, 1),
        (0.75, 1.0, 0),
    ]


def test_merge_empty_thresholds_one_slab():
    regions = _merge_routing_regions(
        thresholds=[], bin_to_partition=[0], x_min=-1.0, x_max=1.0
    )
    assert regions == [(-1.0, 1.0, 0)]


def test_merge_x_min_greater_than_first_threshold_clamps_left_edge():
    regions = _merge_routing_regions(
        thresholds=[-2.0, 0.0],
        bin_to_partition=[0, 1, 0],
        x_min=-1.0,
        x_max=1.0,
    )
    assert regions[0] == (-1.0, 0.0, 1)
    assert regions[1] == (0.0, 1.0, 0)
    assert len(regions) == 2


import numpy as np
from sklearn.datasets import make_classification

from sgtlearn import SGTClassifier
from sgtlearn._export import _route_samples


def _fitted_clf():
    X, y = make_classification(n_samples=200, n_features=4, random_state=0)
    return SGTClassifier(
        max_depth=2, inner_max_depth=2, inner_max_leaf_nodes=8, random_state=0
    ).fit(X, y), X


def test_route_samples_root_sees_all_rows():
    est, X = _fitted_clf()
    tree = est.tree_export()
    reach = _route_samples(tree, X)
    root = tree["root_index"]
    assert len(reach[root]) == X.shape[0]


def test_route_samples_children_partition_parents_rows():
    est, X = _fitted_clf()
    tree = est.tree_export()
    reach = _route_samples(tree, X)
    nodes_by_id = {n["id"]: n for n in tree["nodes"]}
    for nid, node in nodes_by_id.items():
        if node["is_leaf"]:
            continue
        parent_rows = set(reach[nid].tolist())
        child_rows: set[int] = set()
        for cid in node["children"]:
            child_set = set(reach[cid].tolist())
            assert child_set.isdisjoint(child_rows)
            child_rows |= child_set
        assert child_rows == parent_rows


def test_route_samples_sum_at_leaves_equals_n_samples():
    est, X = _fitted_clf()
    tree = est.tree_export()
    reach = _route_samples(tree, X)
    nodes_by_id = {n["id"]: n for n in tree["nodes"]}
    leaf_total = sum(
        len(reach[nid]) for nid, n in nodes_by_id.items() if n["is_leaf"]
    )
    assert leaf_total == X.shape[0]


def test_route_samples_dtype_indices_are_int():
    est, X = _fitted_clf()
    tree = est.tree_export()
    reach = _route_samples(tree, X)
    for arr in reach.values():
        assert arr.dtype.kind in ("i", "u")
