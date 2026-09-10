"""Regularized budgets retain coherent routing on weighted, missing multioutput data."""

import numpy as np
import pytest

from sgtlearn import SGTClassifier, SGTRegressor
from sgtlearn._export import _route_samples


@pytest.mark.parametrize(
    "criterion,mae_cd",
    [
        ("gini", "0"),
        ("entropy", "0"),
        ("squared_error", "0"),
        ("absolute_error", "0"),
        ("absolute_error", "1"),
    ],
)
def test_regularized_multioutput_budget_preserves_missing_routing(
    criterion, mae_cd, monkeypatch
):
    monkeypatch.setenv("SGTLEARN_MAE_CD", mae_cd)
    states = np.array(
        [(a, b) for a in [-1.0, 1.0, np.nan] for b in [-1.0, 1.0, np.nan]]
    )
    counts = np.arange(9, 0, -1)
    X = np.repeat(states, counts, axis=0)
    y = np.repeat(
        np.column_stack([np.arange(9) % 3, np.arange(9) // 3]), counts, axis=0
    )
    weights = np.resize([0.0, 0.5, 1.0, 2.0], len(X))
    classifier = criterion in {"gini", "entropy"}
    estimator = SGTClassifier if classifier else SGTRegressor
    model = estimator(
        criterion=criterion,
        num_partitions=4,
        max_leaf_nodes=5,
        max_depth=3,
        min_samples_leaf=2,
        min_impurity_decrease=0.01,
        branching_penalty=0.05,
        pairwise_candidates=1,
        pairwise_penalty=0.02,
        inner_max_depth=3,
        inner_max_leaf_nodes=9,
        tao_n_runs=0,
        random_state=17,
    )
    if criterion == "absolute_error" and mae_cd == "0":
        with pytest.warns(UserWarning, match="Coordinate descent is disabled"):
            model.fit(X, y, sample_weight=weights)
    else:
        model.fit(X, y, sample_weight=weights)
    tree = model.tree_export()
    reach = _route_samples(tree, X)
    leaves = [node for node in tree["nodes"] if node["is_leaf"]]
    assert 1 < len(leaves) <= 5
    replay = np.empty_like(y, dtype=float)
    for node in tree["nodes"]:
        rows = reach[node["id"]]
        # Classification export historically reports rounded sample mass here.
        if not classifier:
            assert len(rows) == node["n_samples"]
        if not node["is_leaf"]:
            assert 2 <= len(node["children"]) <= 4
            finite_rows = len(rows)
            if "nan_prediction_partition" in node:
                finite_rows -= np.isnan(X[rows, node["feature"]]).sum()
            assert sum(node["bin_sample_counts"]) == finite_rows
            continue
        assert len(rows) >= 2
        if classifier:
            for output, histogram in enumerate(node["class_counts"]):
                expected = np.bincount(
                    y[rows, output], weights=weights[rows], minlength=3
                )
                np.testing.assert_allclose(histogram, expected)
            replay[rows] = np.argmax(node["class_counts"], axis=1)
        else:
            replay[rows] = node["value"]
    np.testing.assert_allclose(model.predict(X), replay)
