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
