"""Regularized budgets retain coherent routing on weighted, missing multioutput data."""

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

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
@pytest.mark.parametrize("weighted_multioutput", [False, True])
def test_numeric_fallback_counts_missing_samples_in_threshold_feasibility(
    criterion, mae_cd, weighted_multioutput, monkeypatch
):
    monkeypatch.setenv("SGTLEARN_MAE_CD", mae_cd)
    X = np.array([0.0, 1.0, 2.0, np.nan])[:, None]
    y = np.array([0, 0, 1, 1])
    estimator = SGTClassifier if criterion in {"gini", "entropy"} else SGTRegressor
    weights = np.ones(4)
    if weighted_multioutput:
        weights = np.array([1.0, 0.0, 1.0, 2.0])
        y = np.column_stack([y, 1 - y if estimator is SGTClassifier else 2 * y + 3])
    model = estimator(
        criterion=criterion,
        min_samples_leaf=2,
        inner_min_samples_leaf=1,
        inner_min_impurity_decrease=100,
        max_leaf_nodes=2,
        tao_n_runs=0,
        random_state=0,
    ).fit(X, y, sample_weight=weights)
    tree = model.tree_export()
    assert tree["nodes"][0]["thresholds"] == [1.5]
    reached = _route_samples(tree, X)
    leaves = [node for node in tree["nodes"] if node["is_leaf"]]
    assert sorted(tuple(reached[node["id"]]) for node in leaves) == [(0, 1), (2, 3)]
    assert all(node["impurity"] == 0 for node in leaves)
    np.testing.assert_array_equal(model.predict(X), y)


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
def test_numeric_fallback_optimizes_loss_including_missing_samples(
    criterion, mae_cd, monkeypatch
):
    monkeypatch.setenv("SGTLEARN_MAE_CD", mae_cd)
    X = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, np.nan])[:, None]
    y = (
        [0, 0, 0, 0, 1, 0, 1]
        if criterion == "absolute_error"
        else [0, 0, 1, 0, 1, 1, 1]
    )
    estimator = SGTClassifier if criterion in {"gini", "entropy"} else SGTRegressor
    model = estimator(
        criterion=criterion,
        min_samples_leaf=2,
        inner_min_samples_leaf=1,
        inner_min_impurity_decrease=100,
        max_leaf_nodes=2,
        tao_n_runs=0,
        random_state=0,
    ).fit(X, y)
    # Finite-only fitting prefers 1.5. Including the missing target makes 3.5
    # strictly better, so lowering only finite minimum-leaf size is insufficient.
    tree = model.tree_export()
    assert tree["nodes"][0]["thresholds"] == [3.5]
    reached = _route_samples(tree, X)
    leaves = [node for node in tree["nodes"] if node["is_leaf"]]
    assert sorted(tuple(reached[node["id"]]) for node in leaves) == [
        (0, 1, 2, 3),
        (4, 5, 6),
    ]
    expected_loss = {
        "gini": 1.5,
        "entropy": -np.log2(0.25) - 3 * np.log2(0.75),
        "squared_error": 0.75,
        "absolute_error": 1.0,
    }[criterion]
    assert sum(
        node["n_samples"] * node["impurity"] for node in leaves
    ) == pytest.approx(expected_loss)


@pytest.mark.parametrize("estimator", [SGTClassifier, SGTRegressor])
@pytest.mark.parametrize("gap", [0.0, 5e-8, 2e-7])
def test_numeric_fallback_matches_inner_adjacent_value_tolerance(estimator, gap):
    X = np.array([0.0, gap, 2.0, np.nan], dtype=np.float32)[:, None]
    model = estimator(
        inner_min_impurity_decrease=100,
        max_leaf_nodes=2,
        tao_n_runs=0,
    ).fit(X, [0, 1, 1, 1])
    split_allowed = gap > 1e-7
    threshold = float(X[1, 0]) / 2 if split_allowed else 1 + float(X[1, 0]) / 2
    assert model.tree_export()["nodes"][0]["thresholds"] == [threshold]
    expected = [0, 1, 1, 1] if split_allowed else [0, 0, 1, 1]
    if estimator is SGTRegressor and not split_allowed:
        expected = [0.5, 0.5, 1, 1]
    np.testing.assert_allclose(model.predict(X), expected)


@pytest.mark.parametrize("estimator", [SGTClassifier, SGTRegressor])
def test_numeric_fallback_preserves_missing_only_split(estimator):
    X = np.array([0.0, 0.0, np.nan, np.nan])[:, None]
    model = estimator(
        inner_min_impurity_decrease=100,
        min_samples_leaf=2,
        max_leaf_nodes=2,
        tao_n_runs=0,
    ).fit(X, [0, 0, 1, 1])
    tree = model.tree_export()
    assert tree["nodes"][0]["thresholds"] == []
    assert len(tree["nodes"][0]["children"]) == 2
    np.testing.assert_array_equal(model.predict(X), [0, 0, 1, 1])


@pytest.mark.parametrize("size,threshold,loss", [(4, 1.5, 1.0), (6, 2.5, 4 / 3)])
def test_numeric_fallback_retains_outer_feasible_cart_threshold(size, threshold, loss):
    X = np.arange(size, dtype=np.float32).reshape(-1, 1)
    model = SGTClassifier(
        num_partitions=2,
        max_leaf_nodes=2,
        min_samples_leaf=size // 2,
        inner_min_samples_leaf=1,
        inner_max_depth=3,
        inner_max_leaf_nodes=8,
        tao_n_runs=0,
        random_state=0,
    ).fit(X, [0] * (size - 1) + [1])
    tree = model.tree_export()
    leaves = [node for node in tree["nodes"] if node["is_leaf"]]
    assert [node["n_samples"] for node in leaves] == [size // 2, size // 2]
    assert sum(
        node["n_samples"] * node["impurity"] for node in leaves
    ) == pytest.approx(loss)
    assert tree["nodes"][0]["thresholds"] == [threshold]
    reach = _route_samples(tree, X)
    assert sorted(tuple(reach[node["id"]]) for node in leaves) == [
        tuple(range(size // 2)),
        tuple(range(size // 2, size)),
    ]
    np.testing.assert_array_equal(model.predict(X), np.zeros(size))
    np.testing.assert_allclose(
        model.predict_proba(X),
        [[1, 0]] * (size // 2) + [[1 - 2 / size, 2 / size]] * (size // 2),
    )


@pytest.mark.parametrize(
    "criterion,mae_cd",
    [
        ("squared_error", "0"),
        ("absolute_error", "0"),
        ("absolute_error", "1"),
    ],
)
def test_numeric_regression_fallback_can_split_inside_an_inner_bin(
    criterion, mae_cd, monkeypatch
):
    monkeypatch.setenv("SGTLEARN_MAE_CD", mae_cd)
    X = np.arange(4, dtype=np.float32).reshape(-1, 1)
    # The inner stump isolates 3 at 2.5. Outer minimum two requires 1.5.
    model = SGTRegressor(
        criterion=criterion,
        min_samples_leaf=2,
        inner_min_samples_leaf=1,
        inner_max_depth=1,
        max_leaf_nodes=2,
        tao_n_runs=0,
        random_state=0,
    ).fit(X, [0, 0, 1, 3])
    tree = model.tree_export()
    leaves = [node for node in tree["nodes"] if node["is_leaf"]]
    assert [node["n_samples"] for node in leaves] == [2, 2]
    assert sum(node["n_samples"] * node["impurity"] for node in leaves) == 2.0
    assert tree["nodes"][0]["thresholds"] == [1.5]
    np.testing.assert_allclose(model.predict(X), [0, 0, 2, 2])


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
@pytest.mark.parametrize(
    "inner_settings",
    [
        {"inner_min_samples_leaf": 2},
        {"inner_min_impurity_decrease": 100},
        {"inner_max_leaf_nodes": 1},
    ],
)
@pytest.mark.parametrize("weighted_multioutput", [False, True])
def test_numeric_fallback_recovers_suboptimal_or_suppressed_inner_root(
    criterion, mae_cd, inner_settings, weighted_multioutput, monkeypatch
):
    monkeypatch.setenv("SGTLEARN_MAE_CD", mae_cd)
    X = np.arange(4, dtype=np.float32).reshape(-1, 1)
    classifier = criterion in {"gini", "entropy"}
    y = np.array([0, 0, 0, 1] if classifier else [0, 0, 1, 3])
    weights = np.ones(4)
    if weighted_multioutput:
        weights = np.array([1.0, 2.0, 1.0, 2.0])
        y = np.column_stack([y, 1 - y if classifier else 2 * y + 1])
    estimator = SGTClassifier if classifier else SGTRegressor
    cart_type = DecisionTreeClassifier if classifier else DecisionTreeRegressor
    cart = cart_type(criterion=criterion, max_depth=1, random_state=0).fit(
        X, y, sample_weight=weights
    )
    model = estimator(
        criterion=criterion,
        min_samples_leaf=1,
        max_depth=1,
        num_partitions=4,
        max_leaf_nodes=2,
        inner_max_depth=1,
        tao_n_runs=0,
        random_state=0,
        **inner_settings,
    ).fit(X, y, sample_weight=weights)
    tree = model.tree_export()
    # Stricter inner minimum chooses feasible 1.5; stopping settings leave no
    # root at all. The outer CART optimum at 2.5 must survive either case.
    assert tree["nodes"][0]["thresholds"] == [2.5]
    reached = _route_samples(tree, X)
    leaves = [node for node in tree["nodes"] if node["is_leaf"]]
    assert sorted(len(reached[node["id"]]) for node in leaves) == [1, 3]
    loss = sum(weights[reached[node["id"]]].sum() * node["impurity"] for node in leaves)
    cart_leaves = cart.tree_.children_left < 0
    cart_loss = np.sum(
        cart.tree_.weighted_n_node_samples[cart_leaves]
        * cart.tree_.impurity[cart_leaves]
    )
    assert loss == pytest.approx(cart_loss)
    np.testing.assert_allclose(model.predict(X), cart.predict(X))


@pytest.mark.parametrize("estimator,gain", [(SGTClassifier, 0.5), (SGTRegressor, 0.25)])
@pytest.mark.parametrize("cost_fraction,leaf_count", [(0.5, 2), (1.0, 1), (2.0, 1)])
def test_numeric_fallback_pays_growth_cost_and_respects_binary_budget(
    estimator, gain, cost_fraction, leaf_count
):
    X = np.arange(4, dtype=np.float32).reshape(-1, 1)
    model = estimator(
        num_partitions=4,
        max_leaf_nodes=2,
        min_samples_leaf=2,
        inner_min_samples_leaf=1,
        min_impurity_decrease=gain * cost_fraction,
        branching_penalty=100,
        tao_n_runs=0,
        random_state=0,
    ).fit(X, [0, 0, 0, 1])
    leaves = [node for node in model.tree_export()["nodes"] if node["is_leaf"]]
    assert len(leaves) == leaf_count


@pytest.mark.parametrize("estimator", [SGTClassifier, SGTRegressor])
def test_numeric_fallback_keeps_a_better_rich_shape_candidate(estimator):
    X = np.arange(6, dtype=np.float32).reshape(-1, 1)
    y = [0, 0, 1, 1, 0, 0]
    model = estimator(
        max_leaf_nodes=2,
        min_samples_leaf=2,
        inner_min_samples_leaf=1,
        inner_max_depth=3,
        inner_max_leaf_nodes=8,
        tao_n_runs=0,
        random_state=0,
    ).fit(X, y)
    assert model.tree_export()["nodes"][0]["thresholds"] == [1.5, 3.5]
    np.testing.assert_array_equal(model.predict(X), y)


@pytest.mark.parametrize("inner_gain", [0.0, 100.0])
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
    criterion, mae_cd, inner_gain, monkeypatch
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
        inner_min_impurity_decrease=inner_gain,
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
    if inner_gain:
        # The independent fallback must retain finite thresholds as well as its
        # missing route when the original inner tree never split.
        assert any(node.get("thresholds") for node in tree["nodes"])
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
