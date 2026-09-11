"""Observable consequences of total-mass outer split scoring."""

import numpy as np
import pytest

from sgtlearn import SGTClassifier, SGTRegressor


@pytest.mark.parametrize(
    "criterion", ["gini", "entropy", "squared_error", "absolute_error"]
)
def test_sample_mass_scales_gain_but_not_growth_cost(criterion, monkeypatch):
    monkeypatch.setenv("SGTLEARN_MAE_CD", "1")
    X = np.array([[0.0], [0.0], [1.0], [1.0]])
    y = np.array([0, 0, 1, 1])
    estimator = SGTClassifier if criterion in {"gini", "entropy"} else SGTRegressor
    # Unit-mass gains: 2 (Gini/MAE), 4 bits (entropy), and 1 (MSE).
    gain = {"gini": 2.0, "entropy": 4.0, "squared_error": 1.0, "absolute_error": 2.0}[
        criterion
    ]
    for mass, expected_nodes in [(1.0, 1), (4.0, 3)]:
        model = estimator(
            criterion=criterion,
            max_depth=1,
            min_impurity_decrease=1.5 * gain,
            inner_max_depth=1,
            inner_max_leaf_nodes=2,
            tao_n_runs=0,
        ).fit(X, y, sample_weight=np.full(4, mass))
        assert model.tree_export()["num_nodes"] == expected_nodes


@pytest.mark.parametrize("estimator", [SGTClassifier, SGTRegressor])
@pytest.mark.parametrize("weighted", [False, True])
def test_leaf_budget_expands_larger_total_gain_instead_of_larger_local_gain(
    estimator, weighted
):
    # Root splits feature 0. Left child: 60 samples, p=0.2; its feature-1
    # split gains .08 Gini (total 4.8). Right: 20 samples, p=.75; its split
    # gains .125 Gini (total 2.5). MSE gains are exactly half these values.
    cells = [[0, 0], [0, 1], [0, 1], [1, 0], [1, 0], [1, 1]]
    counts = [30, 18, 12, 5, 5, 10]
    X = np.repeat(cells, counts, axis=0).astype(float)
    y = np.repeat([0, 0, 1, 0, 1, 1], counts)
    weights = np.ones(len(y))
    if weighted:
        X, y = np.repeat(X, 2, axis=0), np.repeat(y, 2)
        weights = np.tile([0.25, 1.75], len(y) // 2)
    model = estimator(
        max_depth=2,
        max_leaf_nodes=3,
        inner_max_depth=1,
        inner_max_leaf_nodes=2,
        tao_n_runs=0,
    ).fit(X, y, sample_weight=weights)
    nodes = model.tree_export()["nodes"]
    assert nodes[0]["features"] == [0]
    expanded = [node for node in nodes[1:] if not node["is_leaf"]]
    assert len(expanded) == 1
    assert expanded[0]["n_samples"] == 60 * (2 if weighted else 1)


@pytest.mark.parametrize("estimator", [SGTClassifier, SGTRegressor])
def test_unlimited_outer_growth_is_also_best_first(estimator):
    cells = [[0, 0], [0, 1], [0, 1], [1, 0], [1, 0], [1, 1]]
    counts = [30, 18, 12, 5, 5, 10]
    X = np.repeat(cells, counts, axis=0).astype(float)
    y = np.repeat([0, 0, 1, 0, 1, 1], counts)
    model = estimator(
        max_depth=2,
        inner_max_depth=1,
        inner_max_leaf_nodes=2,
        tao_n_runs=0,
    ).fit(X, y)
    nodes = model.tree_export()["nodes"]
    # Children receive IDs when their parent is committed. The larger-gain
    # 60-sample branch must be expanded first even without a finite leaf cap.
    assert len(nodes) == 7
    assert nodes[3]["n_samples"] == 30


@pytest.mark.parametrize(
    "criterion", ["gini", "entropy", "squared_error", "absolute_error"]
)
def test_outer_scores_average_outputs_and_preserve_uniform_output_replication(
    criterion, monkeypatch
):
    monkeypatch.setenv("SGTLEARN_MAE_CD", "1")
    X = np.array([[0.0], [0.0], [1.0], [1.0]])
    y = np.array([0, 0, 1, 1])
    # Feature 0 perfectly predicts y but gives no gain on the second target.
    two_targets = np.column_stack([y, [0, 1, 0, 1]])
    weights = np.array([1.0, 2.0, 1.0, 2.0])
    gain = {"gini": 3.0, "entropy": 6.0, "squared_error": 1.5, "absolute_error": 3.0}[
        criterion
    ]
    second_entropy = -(np.log2(1 / 3) / 3 + 2 * np.log2(2 / 3) / 3)
    mean_impurity = {
        "gini": (0.5 + 4 / 9) / 2,
        "entropy": (1 + second_entropy) / 2,
        "squared_error": (0.25 + 2 / 9) / 2,
        "absolute_error": (0.5 + 1 / 3) / 2,
    }[criterion]
    estimator = SGTClassifier if criterion in {"gini", "entropy"} else SGTRegressor
    for targets in [two_targets, np.tile(two_targets, (1, 2))]:
        model = estimator(
            criterion=criterion,
            max_depth=1,
            inner_max_depth=1,
            min_impurity_decrease=0.75 * gain,
            tao_n_runs=0,
        ).fit(X, targets, sample_weight=weights)
        tree = model.tree_export()
        assert tree["num_nodes"] == 1
        assert tree["nodes"][0]["impurity"] == pytest.approx(mean_impurity)
    single = estimator(
        criterion=criterion,
        max_depth=1,
        inner_max_depth=1,
        min_impurity_decrease=0.75 * gain,
        tao_n_runs=0,
    ).fit(X, y, sample_weight=weights)
    assert single.tree_export()["num_nodes"] == 3


@pytest.mark.parametrize(
    "estimator,gamma", [(SGTClassifier, 1.25), (SGTRegressor, 0.625)]
)
def test_pair_cost_is_constant_total_loss_and_charged_once(estimator, gamma):
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 1])
    # OR gives total Gini gain .5 univariately and 1.5 as a pair (MSE half).
    for weight, pair_expected in [(1.0, False), (4.0, True)]:
        model = estimator(
            max_depth=1,
            pairwise_candidates=1,
            pairwise_penalty=gamma,
            inner_max_depth=2,
            inner_max_leaf_nodes=4,
            tao_n_runs=0,
        ).fit(X, y, sample_weight=np.full(4, weight))
        assert (
            model.tree_export()["nodes"][0].get("routing_kind") == "pair"
        ) == pair_expected
