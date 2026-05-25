"""Structural tests for ``SGT{Classifier,Regressor}.tree_export()``."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.exceptions import NotFittedError

from sgtlearn import SGTClassifier, SGTRegressor


def _fitted_classifier(num_partitions: int = 2) -> SGTClassifier:
    X, y = make_classification(n_samples=200, n_features=5, n_informative=3,
                               n_redundant=0, random_state=0)
    return SGTClassifier(
        num_partitions=num_partitions, max_depth=3, inner_max_depth=2,
        inner_max_leaf_nodes=8, random_state=0,
    ).fit(X, y)


def _fitted_regressor(criterion: str = "squared_error",
                      num_partitions: int = 2) -> SGTRegressor:
    X, y = make_regression(n_samples=200, n_features=5, n_informative=3,
                           random_state=0)
    return SGTRegressor(
        criterion=criterion, num_partitions=num_partitions, max_depth=3,
        inner_max_depth=2, inner_max_leaf_nodes=8, random_state=0,
    ).fit(X, y)


def test_classifier_export_top_level_keys():
    est = _fitted_classifier()
    tr = est.tree_export()
    assert set(tr) >= {"num_partitions", "num_nodes", "root_index",
                       "num_classes", "criterion", "nodes"}
    assert tr["num_partitions"] == 2
    assert tr["num_classes"] == 2
    assert tr["criterion"] == "gini"
    assert tr["num_nodes"] == len(tr["nodes"])
    assert tr["nodes"][tr["root_index"]]["depth"] == 0


def test_classifier_internal_node_invariants():
    est = _fitted_classifier()
    tr = est.tree_export()
    internals = [n for n in tr["nodes"] if not n["is_leaf"]]
    assert internals, "expected at least one internal node"
    for n in internals:
        assert n["feature"] is not None
        assert len(n["thresholds"]) + 1 == len(n["bin_to_partition"])
        assert len(n["bin_sample_counts"]) == len(n["bin_to_partition"])
        assert len(n["bin_counts"]) == len(n["bin_to_partition"])
        for p in n["bin_to_partition"]:
            assert 0 <= p < tr["num_partitions"]
        assert sum(n["bin_sample_counts"]) == n["n_samples"]
        assert n["children"] and len(n["children"]) == tr["num_partitions"]


def test_classifier_leaf_invariants():
    est = _fitted_classifier()
    tr = est.tree_export()
    leaves = [n for n in tr["nodes"] if n["is_leaf"]]
    assert leaves
    for leaf in leaves:
        assert leaf["feature"] is None
        assert leaf["children"] == []
        assert leaf["bin_to_partition"] == []
        assert leaf["bin_sample_counts"] == []
        assert sum(leaf["class_counts"]) == leaf["n_samples"]
        assert len(leaf["class_counts"]) == tr["num_classes"]


def test_regressor_mse_export():
    est = _fitted_regressor(criterion="squared_error")
    tr = est.tree_export()
    assert "num_classes" not in tr
    assert tr["criterion"] == "squared_error"
    internals = [n for n in tr["nodes"] if not n["is_leaf"]]
    assert internals
    for n in internals:
        assert len(n["bin_counts"]) == len(n["bin_to_partition"])
        for pair in n["bin_counts"]:
            assert len(pair) == 2  # [sum_y, sum_y^2]
        assert len(n["bin_sample_counts"]) == len(n["bin_to_partition"])
    leaves = [n for n in tr["nodes"] if n["is_leaf"]]
    for leaf in leaves:
        assert isinstance(leaf["value"], float)


def test_regressor_mae_has_empty_bin_counts_but_full_sample_counts():
    est = _fitted_regressor(criterion="absolute_error")
    tr = est.tree_export()
    assert tr["criterion"] == "absolute_error"
    internals = [n for n in tr["nodes"] if not n["is_leaf"]]
    assert internals
    for n in internals:
        # bin_counts is empty for MAE (regressionSplitLeafStats not populated)
        assert n["bin_counts"] == []
        # bin_sample_counts is populated regardless of criterion
        assert len(n["bin_sample_counts"]) == len(n["bin_to_partition"])
        assert sum(n["bin_sample_counts"]) == n["n_samples"]


def test_classifier_unfitted_raises():
    est = SGTClassifier()
    with pytest.raises(NotFittedError):
        est.tree_export()


def test_regressor_unfitted_raises():
    est = SGTRegressor()
    with pytest.raises(NotFittedError):
        est.tree_export()


def test_classifier_multiway_partitions():
    est = _fitted_classifier(num_partitions=3)
    tr = est.tree_export()
    assert tr["num_partitions"] == 3
    internals = [n for n in tr["nodes"] if not n["is_leaf"]]
    for n in internals:
        assert len(n["children"]) == 3
        for p in n["bin_to_partition"]:
            assert 0 <= p < 3
