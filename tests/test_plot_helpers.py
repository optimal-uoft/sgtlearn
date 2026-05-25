"""Unit tests for ``sgtlearn._export`` private helpers."""
from __future__ import annotations
from matplotlib.patches import Rectangle

import pytest
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
from sklearn.datasets import make_classification
from sgtlearn import SGTClassifier
from sgtlearn._export import _merge_routing_regions, _route_samples, _compute_layout_leafcounter, _draw_leaf_text, _draw_internal_panel, _draw_arrow_edge


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


def test_draw_arrow_edge_returns_fancyarrowpatch():
    fig, ax = plt.subplots()
    patch = _draw_arrow_edge(
        fig=fig,
        parent_xy=(0.3, 0.8),
        parent_h=0.1,
        child_xy=(0.5, 0.3),
        child_h=0.05,
        color="#E8A0BF",
        host_ax=ax,
    )
    assert isinstance(patch, FancyArrowPatch)
    assert patch in ax.patches
    plt.close(fig)


def test_draw_arrow_edge_uses_supplied_color():
    fig, ax = plt.subplots()
    patch = _draw_arrow_edge(
        fig=fig,
        parent_xy=(0.0, 0.0),
        parent_h=0.0,
        child_xy=(1.0, 1.0),
        child_h=0.0,
        color="#FAC898",
        host_ax=ax,
    )
    assert tuple(patch.get_edgecolor())[:3] == pytest.approx(
        matplotlib.colors.to_rgb("#FAC898"), abs=1e-6
    )
    plt.close(fig)


def _clf_leaf_node():
    return {
        "id": 5,
        "depth": 2,
        "is_leaf": True,
        "n_samples": 42,
        "impurity": 0.123,
        "class_counts": [10, 32],
        "children": [],
    }


def _reg_leaf_node():
    return {
        "id": 5,
        "depth": 2,
        "is_leaf": True,
        "n_samples": 42,
        "impurity": 17.5,
        "value": -3.14,
        "children": [],
    }


def test_draw_leaf_text_classifier_bold_class_label():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    artists = _draw_leaf_text(
        host_ax=ax,
        x=0.5,
        y=0.5,
        node=_clf_leaf_node(),
        is_classifier=True,
        class_names=["neg", "pos"],
        criterion="gini",
        precision=2,
        fontsize=10,
        color="#E8A0BF",
        label="feature",
        impurity=False,
    )
    bold_texts = [
        a for a in artists
        if hasattr(a, "get_text") and a.get_fontweight() == "bold"
    ]
    assert any(a.get_text() == "pos" for a in bold_texts)
    plt.close(fig)


def test_draw_leaf_text_regressor_value_formatted_to_precision():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    artists = _draw_leaf_text(
        host_ax=ax,
        x=0.5,
        y=0.5,
        node=_reg_leaf_node(),
        is_classifier=False,
        class_names=None,
        criterion="squared_error",
        precision=2,
        fontsize=10,
        color="#FAC898",
        label="feature",
        impurity=False,
    )
    texts = [a.get_text() for a in artists if hasattr(a, "get_text")]
    assert any(t == "-3.14" for t in texts)
    plt.close(fig)


def test_draw_leaf_text_label_all_adds_n_subtitle():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    artists = _draw_leaf_text(
        host_ax=ax,
        x=0.5,
        y=0.5,
        node=_clf_leaf_node(),
        is_classifier=True,
        class_names=["neg", "pos"],
        criterion="gini",
        precision=2,
        fontsize=10,
        color="#E8A0BF",
        label="all",
        impurity=False,
    )
    texts = [a.get_text() for a in artists if hasattr(a, "get_text")]
    assert any("n = 42" in t for t in texts)
    plt.close(fig)


def test_draw_leaf_text_label_all_with_impurity_adds_criterion_line():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    artists = _draw_leaf_text(
        host_ax=ax,
        x=0.5,
        y=0.5,
        node=_clf_leaf_node(),
        is_classifier=True,
        class_names=["neg", "pos"],
        criterion="gini",
        precision=2,
        fontsize=10,
        color="#E8A0BF",
        label="all",
        impurity=True,
    )
    texts = [a.get_text() for a in artists if hasattr(a, "get_text")]
    assert any("gini = 0.12" in t for t in texts)
    plt.close(fig)


def test_draw_leaf_text_label_none_suppresses_subtitle():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    artists = _draw_leaf_text(
        host_ax=ax,
        x=0.5,
        y=0.5,
        node=_clf_leaf_node(),
        is_classifier=True,
        class_names=["neg", "pos"],
        criterion="gini",
        precision=2,
        fontsize=10,
        color="#E8A0BF",
        label="none",
        impurity=True,
    )
    texts = [a.get_text() for a in artists if hasattr(a, "get_text")]
    assert not any("n =" in t or "gini =" in t for t in texts)
    plt.close(fig)


def test_draw_leaf_text_color_propagates():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    artists = _draw_leaf_text(
        host_ax=ax,
        x=0.5,
        y=0.5,
        node=_clf_leaf_node(),
        is_classifier=True,
        class_names=["neg", "pos"],
        criterion="gini",
        precision=2,
        fontsize=10,
        color="#E8A0BF",
        label="feature",
        impurity=False,
    )
    bold_texts = [
        a for a in artists
        if hasattr(a, "get_text") and a.get_fontweight() == "bold"
    ]
    expected_rgb = matplotlib.colors.to_rgb("#E8A0BF")
    for t in bold_texts:
        c = matplotlib.colors.to_rgb(t.get_color())
        assert c == pytest.approx(expected_rgb, abs=1e-6)
    plt.close(fig)


def _internal_node():
    return {
        "id": 0,
        "depth": 0,
        "is_leaf": False,
        "feature": 0,
        "thresholds": [-0.5, 0.5],
        "bin_to_partition": [0, 1, 0],
        "bin_sample_counts": [30, 40, 20],
        "n_samples": 90,
        "impurity": 0.5,
        "children": [1, 2],
    }


def test_draw_internal_panel_slabs_only_when_no_X():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    palette = ["#E8A0BF", "#FAC898"]
    artists = _draw_internal_panel(
        host_ax=ax,
        center=(0.5, 0.5),
        size=(0.3, 0.12),
        node=_internal_node(),
        palette=palette,
        feature_values=None,
        n_hist_bins=20,
        precision=2,
        fontsize=10,
        label="feature",
    )
    inset_axes_objs = [a for a in artists if hasattr(a, "axvspan")]
    assert inset_axes_objs, "expected an inset Axes in returned artists"
    inset = inset_axes_objs[0]
    # axvspan adds a Polygon to inset.patches; non-contiguous [0,1,0]
    # bin_to_partition yields 3 slabs.
    assert len(inset.patches) == 3
    plt.close(fig)


def test_draw_internal_panel_histogram_overlay_when_X_provided():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    palette = ["#E8A0BF", "#FAC898"]
    feat_vals = np.linspace(-1.0, 1.0, 200)
    artists = _draw_internal_panel(
        host_ax=ax,
        center=(0.5, 0.5),
        size=(0.3, 0.12),
        node=_internal_node(),
        palette=palette,
        feature_values=feat_vals,
        n_hist_bins=20,
        precision=2,
        fontsize=10,
        label="feature",
    )
    inset_axes_objs = [a for a in artists if hasattr(a, "axvspan")]
    inset = inset_axes_objs[0]
    bars = [p for p in inset.patches if isinstance(p, Rectangle)]
    assert len(bars) >= 15
    plt.close(fig)


def test_draw_internal_panel_label_none_hides_sample_count():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    artists = _draw_internal_panel(
        host_ax=ax,
        center=(0.5, 0.5),
        size=(0.3, 0.12),
        node=_internal_node(),
        palette=["#E8A0BF", "#FAC898"],
        feature_values=None,
        n_hist_bins=20,
        precision=2,
        fontsize=10,
        label="none",
    )
    text_artists = [
        a for a in artists
        if hasattr(a, "get_text") and not hasattr(a, "axvspan")
    ]
    assert not any("n=" in t.get_text() for t in text_artists)
    plt.close(fig)


def test_draw_internal_panel_label_feature_shows_sample_count():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    artists = _draw_internal_panel(
        host_ax=ax,
        center=(0.5, 0.5),
        size=(0.3, 0.12),
        node=_internal_node(),
        palette=["#E8A0BF", "#FAC898"],
        feature_values=None,
        n_hist_bins=20,
        precision=2,
        fontsize=10,
        label="feature",
    )
    text_artists = [a for a in artists if hasattr(a, "get_text")]
    assert any("n=90" in t.get_text() for t in text_artists)
    plt.close(fig)
