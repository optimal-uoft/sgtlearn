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


from sgtlearn._export import _compute_layout_leafcounter


def _toy_tree() -> dict:
    """Hand-rolled tree dict matching tree_export()'s shape for layout tests.

    Structure::

        0 (root, internal, depth=0) -> [1, 2]
        1 (internal, depth=1)       -> [3, 4]
        2 (leaf, depth=1)
        3 (leaf, depth=2)
        4 (leaf, depth=2)
    """
    return {
        "num_partitions": 2,
        "num_nodes": 5,
        "root_index": 0,
        "criterion": "gini",
        "nodes": [
            {"id": 0, "depth": 0, "is_leaf": False, "children": [1, 2]},
            {"id": 1, "depth": 1, "is_leaf": False, "children": [3, 4]},
            {"id": 2, "depth": 1, "is_leaf": True, "children": []},
            {"id": 3, "depth": 2, "is_leaf": True, "children": []},
            {"id": 4, "depth": 2, "is_leaf": True, "children": []},
        ],
    }


def test_layout_returns_one_position_per_node():
    layout = _compute_layout_leafcounter(_toy_tree(), max_depth=None)
    assert set(layout) == {0, 1, 2, 3, 4}


def test_layout_x_in_unit_interval():
    layout = _compute_layout_leafcounter(_toy_tree(), max_depth=None)
    for x, _y in layout.values():
        assert 0.0 <= x <= 1.0


def test_layout_y_top_to_bottom_by_depth():
    layout = _compute_layout_leafcounter(_toy_tree(), max_depth=None)
    assert layout[0][1] > layout[1][1]
    assert layout[0][1] > layout[2][1]
    assert layout[1][1] > layout[3][1]
    assert layout[1][1] > layout[4][1]


def test_layout_parent_x_centered_over_children():
    layout = _compute_layout_leafcounter(_toy_tree(), max_depth=None)
    expected = (layout[3][0] + layout[4][0]) / 2
    assert layout[1][0] == pytest.approx(expected, abs=1e-9)
    expected_root = (layout[1][0] + layout[2][0]) / 2
    assert layout[0][0] == pytest.approx(expected_root, abs=1e-9)


def test_layout_max_depth_truncates_subtree():
    layout = _compute_layout_leafcounter(_toy_tree(), max_depth=1)
    assert 3 not in layout
    assert 4 not in layout
    assert 1 in layout


def test_layout_single_node_tree():
    tree = {
        "num_partitions": 2,
        "num_nodes": 1,
        "root_index": 0,
        "criterion": "gini",
        "nodes": [{"id": 0, "depth": 0, "is_leaf": True, "children": []}],
    }
    layout = _compute_layout_leafcounter(tree, max_depth=None)
    assert set(layout) == {0}
    assert layout[0][0] == pytest.approx(0.5, abs=1e-9)


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from sgtlearn._export import _draw_arrow_edge


def test_draw_arrow_edge_returns_fancyarrowpatch():
    fig, _ax = plt.subplots()
    patch = _draw_arrow_edge(
        fig=fig,
        parent_xy=(0.3, 0.8),
        parent_h=0.1,
        child_xy=(0.5, 0.3),
        child_h=0.05,
        color="#E8A0BF",
    )
    assert isinstance(patch, FancyArrowPatch)
    assert patch in fig.artists
    plt.close(fig)


def test_draw_arrow_edge_uses_supplied_color():
    fig, _ax = plt.subplots()
    patch = _draw_arrow_edge(
        fig=fig,
        parent_xy=(0.0, 0.0),
        parent_h=0.0,
        child_xy=(1.0, 1.0),
        child_h=0.0,
        color="#FAC898",
    )
    assert tuple(patch.get_edgecolor())[:3] == pytest.approx(
        matplotlib.colors.to_rgb("#FAC898"), abs=1e-6
    )
    plt.close(fig)
