"""Public outer-tree leaf-budget and arity-selection behavior."""

import numpy as np
import pytest

from sgtlearn import SGTClassifier, SGTRegressor
from sgtlearn._export import _route_samples


@pytest.mark.parametrize("estimator", [SGTClassifier, SGTRegressor])
def test_ternary_search_obeys_a_two_leaf_budget(estimator):
    X = np.arange(18, dtype=float).reshape(-1, 1)
    y = np.repeat([0, 1, 2], 6)
    model = estimator(
        num_partitions=3, max_leaf_nodes=2, max_depth=2, tao_n_runs=0,
    ).fit(X, y)
    nodes = model.tree_export()["nodes"]
    assert sum(node["is_leaf"] for node in nodes) == 2
    assert len(nodes) == 3


def _competing_nodes(errors):
    rows, targets = [], []
    # A: perfect ternary gain 20. B: ternary feature-1 gain 13, best
    # binary feature-1 gain 7.3, binary feature-2 gain 10.
    for label in range(3):
        rows.extend([[0, label, 0]] * 10)
        targets.extend([label] * 10)
    for bin_id, histogram in enumerate([[8, 1, 1], [1, 9, 0], [1, 0, 9]]):
        for label, count in enumerate(histogram):
            rows.extend([[1, bin_id, int(label != 0)]] * count)
            targets.extend([label + 3] * count)
    # C: binary gain 8.0667 (errors=2) or 11.2667 (errors=1).
    for side in range(2):
        rows.extend([[2, 0, side]] * 15)
        targets.extend([6 + side] * (15 - errors) + [7 - side] * errors)
    return np.asarray(rows, dtype=float), np.asarray(targets)


@pytest.mark.parametrize("errors,expanded_group", [(2, 1), (1, 2)])
def test_shrinking_budget_changes_router_and_reorders_competing_nodes(errors, expanded_group):
    X, y = _competing_nodes(errors)
    model = SGTClassifier(
        num_partitions=3, max_leaf_nodes=6, max_depth=2,
        inner_max_depth=3, inner_max_leaf_nodes=8, tao_n_runs=0,
    ).fit(X, y)
    nodes = model.tree_export()["nodes"]
    assert sum(node["is_leaf"] for node in nodes) == 6
    assert nodes[0]["features"] == [0]
    # Root and A consume four additional leaves; only a binary split fits.
    # B must switch feature 1 -> 2, and its score drops 13 -> 10, below
    # C in the errors=1 case. Identify depth-one nodes by their labels.
    branches = {
        int(np.flatnonzero(np.asarray(node["class_counts"]).ravel())[0]) // 3: node
        for node in nodes if node["depth"] == 1
    }
    assert not branches[0]["is_leaf"]
    assert not branches[expanded_group]["is_leaf"]
    assert branches[3 - expanded_group]["is_leaf"]
    assert branches[expanded_group]["features"] == [2]


@pytest.mark.parametrize("strength,expanded_group", [(1.5625, 1), (1.4375, 0)])
@pytest.mark.parametrize("cap", [6, 7])
def test_regression_budget_fallback_uses_best_router_and_rekeys(strength, expanded_group, cap):
    fg = np.array([(f, g) for f in [-1., 0., 1.] for g in [-1., 1.]])
    X = np.vstack([np.column_stack([np.full(6, group), fg]) for group in [-1., 0., 1.]])
    y = np.concatenate([
        -32 + 2 * fg[:, 0] + 1.5 * fg[:, 1],
        strength * fg[:, 1],
        32 + 3 * fg[:, 0],
    ])
    model = SGTRegressor(
        num_partitions=3, max_depth=2, max_leaf_nodes=cap,
        inner_max_depth=2, inner_max_leaf_nodes=3, tao_n_runs=0, random_state=42,
    ).fit(X, y)
    nodes = model.tree_export()["nodes"]
    assert sum(node["is_leaf"] for node in nodes) == cap
    assert nodes[0]["features"] == [0]
    reach = _route_samples(model.tree_export(), X)
    branches = sorted((node for node in nodes if node["depth"] == 1),
                      key=lambda node: X[reach[node["id"]], 0].mean())
    assert not branches[2]["is_leaf"]
    if cap == 7:
        assert not branches[0]["is_leaf"]
        assert branches[0]["features"] == [1]
        assert branches[1]["is_leaf"]
    else:
        assert not branches[expanded_group]["is_leaf"]
        assert branches[expanded_group]["features"] == [2]
        assert branches[1 - expanded_group]["is_leaf"]
