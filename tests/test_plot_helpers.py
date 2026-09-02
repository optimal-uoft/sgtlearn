"""Nontrivial routing contracts used by public tree plotting."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification

from sgtlearn import SGTClassifier
from sgtlearn._export import (
    _merge_routing_regions,
    _pair_switch_boundaries,
    _route_samples,
)
from tests.constants import TEST_TAO_N_RUNS


def test_merge_adjacent_bins_only_when_partition_matches() -> None:
    assert _merge_routing_regions(
        thresholds=[0.5], bin_to_partition=[0, 0], x_min=-1.0, x_max=1.0
    ) == [(-1.0, 1.0, 0)]
    assert _merge_routing_regions(
        thresholds=[0.5], bin_to_partition=[0, 1], x_min=-1.0, x_max=1.0
    ) == [(-1.0, 0.5, 0), (0.5, 1.0, 1)]


def test_merge_keeps_noncontiguous_partition_regions_separate() -> None:
    assert _merge_routing_regions(
        thresholds=[-0.5, 0.0, 0.5],
        bin_to_partition=[0, 1, 0, 1],
        x_min=-1.0,
        x_max=1.0,
    ) == [
        (-1.0, -0.5, 0),
        (-0.5, 0.0, 1),
        (0.0, 0.5, 0),
        (0.5, 1.0, 1),
    ]


def test_merge_ignores_trailing_nan_bin() -> None:
    assert _merge_routing_regions(
        thresholds=[-0.5, 0.0, 0.5],
        bin_to_partition=[0, 1, 0, 1, 0],
        x_min=-1.0,
        x_max=1.0,
    ) == [
        (-1.0, -0.5, 0),
        (-0.5, 0.0, 1),
        (0.0, 0.5, 0),
        (0.5, 1.0, 1),
    ]


def test_merge_clamps_regions_to_observed_range() -> None:
    assert _merge_routing_regions(
        thresholds=[-2.0, 0.0],
        bin_to_partition=[0, 1, 0],
        x_min=-1.0,
        x_max=1.0,
    ) == [(-1.0, 0.0, 1), (0.0, 1.0, 0)]


def test_pair_switch_boundaries_only_include_partition_changes() -> None:
    cells: list[tuple[float, float, object]] = [
        (0.0, 1.0, 0.5),
        (1.0, 2.0, 1.5),
        (2.0, 3.0, 2.5),
    ]
    partitions = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]])
    assert _pair_switch_boundaries(cells, partitions, axis=0) == [1.0, 2.0]
    assert _pair_switch_boundaries(cells, partitions, axis=1) == [1.0, 2.0]


def test_route_samples_partitions_every_parent_row_once() -> None:
    X, y = make_classification(n_samples=200, n_features=4, random_state=0)
    estimator = SGTClassifier(
        max_depth=2,
        inner_max_depth=2,
        inner_max_leaf_nodes=8,
        random_state=0,
        tao_n_runs=TEST_TAO_N_RUNS,
    ).fit(X, y)
    tree = estimator.tree_export()
    reach = _route_samples(tree, X)
    nodes = {node["id"]: node for node in tree["nodes"]}

    assert len(reach[tree["root_index"]]) == X.shape[0]
    for node_id, node in nodes.items():
        assert reach[node_id].dtype.kind in ("i", "u")
        if node["is_leaf"]:
            continue
        child_rows = [set(reach[child].tolist()) for child in node["children"]]
        assert set().union(*child_rows) == set(reach[node_id].tolist())
        assert sum(map(len, child_rows)) == len(reach[node_id])


def test_route_samples_replays_pair_missing_edges() -> None:
    tree = {
        "root_index": 0,
        "nodes": [
            {
                "id": 0,
                "is_leaf": False,
                "routing_kind": "pair",
                "features": [0, 1],
                "children": [1, 2, 3],
                "bin_to_partition": [0, 1, 2, 2, 2],
                "pair_axes": [
                    {"kind": "continuous", "columns": [0]},
                    {"kind": "continuous", "columns": [1]},
                ],
                "pair_inner_tree": [
                    {
                        "id": 0,
                        "is_leaf": False,
                        "axis": 0,
                        "kind": "continuous",
                        "feature": 0,
                        "threshold": 0.0,
                        "left": 1,
                        "right": 2,
                        "missing": 3,
                    },
                    {"id": 1, "is_leaf": True, "bin": 0},
                    {"id": 2, "is_leaf": True, "bin": 1},
                    {
                        "id": 3,
                        "is_leaf": False,
                        "axis": 1,
                        "kind": "continuous",
                        "feature": 1,
                        "threshold": 0.0,
                        "left": 4,
                        "right": 5,
                        "missing": 6,
                    },
                    {"id": 4, "is_leaf": True, "bin": 2},
                    {"id": 5, "is_leaf": True, "bin": 3},
                    {"id": 6, "is_leaf": True, "bin": 4},
                ],
            },
            {"id": 1, "is_leaf": True, "children": []},
            {"id": 2, "is_leaf": True, "children": []},
            {"id": 3, "is_leaf": True, "children": []},
        ],
    }
    X = np.array(
        [[-1.0, 1.0], [1.0, 1.0], [np.nan, -1.0], [np.nan, 1.0], [np.nan, np.nan]]
    )

    reach = _route_samples(tree, X)

    assert reach[1].tolist() == [0]
    assert reach[2].tolist() == [1]
    assert reach[3].tolist() == [2, 3, 4]
