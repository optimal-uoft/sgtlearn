"""Feasible split retention, root-relative quality, and the removed init flag."""

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sgtlearn import (
    SGTClassifier,
    SGTRegressor,
    RandomSGForestClassifier,
    RandomSGForestRegressor,
)
from sgtlearn._export import _route_samples


def _child_impurity(model, X, y, weights, criterion):
    tree = model.tree_export()
    reached = _route_samples(tree, X)
    root = tree["nodes"][0]
    y = np.asarray(y).reshape(len(X), -1)
    total = 0.0
    for child in root["children"]:
        rows = reached[child]
        mass = weights[rows].sum()
        if mass == 0:
            continue
        for output in y.T:
            p = np.bincount(output[rows], weights=weights[rows]) / mass
            loss = (
                1 - np.sum(p**2)
                if criterion == "gini"
                else -np.sum(p[p > 0] * np.log2(p[p > 0]))
            )
            total += mass * loss / weights.sum()
    return total


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
@pytest.mark.parametrize("outputs", [1, 2])
def test_selected_split_is_no_worse_than_feasible_binary_root(criterion, outputs):
    rng = np.random.default_rng(12)
    X = np.repeat(np.arange(10), 10).reshape(-1, 1).astype(float)
    y = rng.integers(0, 3, size=(len(X), outputs))
    if outputs == 1:
        y = y[:, 0]
    weights = rng.uniform(0.01, 0.9, len(X))
    weights[::7] = 0
    stump = DecisionTreeClassifier(
        max_depth=1, criterion=criterion, random_state=0
    ).fit(X, y, sample_weight=weights)
    children = [stump.tree_.children_left[0], stump.tree_.children_right[0]]
    baseline = (
        sum(
            stump.tree_.weighted_n_node_samples[i] * stump.tree_.impurity[i]
            for i in children
        )
        / weights.sum()
        * outputs
    )
    for k in (2, 3, 5):
        model = SGTClassifier(
            criterion=criterion,
            num_partitions=k,
            max_depth=1,
            tao_n_runs=0,
            random_state=42,
        ).fit(X, y, sample_weight=weights)
        assert not model.tree_export()["nodes"][0]["is_leaf"]
        assert _child_impurity(model, X, y, weights, criterion) <= baseline + 1e-7


@pytest.mark.parametrize("categorical", [False, True])
@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_infeasible_root_uses_feasible_binary_fallback(categorical, criterion):
    categories = np.repeat([0, 1, 2], [1, 4, 5])
    X = np.eye(3)[categories] if categorical else categories[:, None].astype(float)
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    kwargs = {"feature_dict": {"category": [0, 1, 2]}} if categorical else {}
    model = SGTClassifier(
        criterion=criterion,
        min_samples_leaf=5,
        max_depth=1,
        inner_max_depth=1 if categorical else 3,
        num_partitions=4,
        tao_n_runs=0,
        random_state=0,
    ).fit(X, y, **kwargs)
    tree = model.tree_export()
    assert not tree["nodes"][0]["is_leaf"]
    reached = _route_samples(tree, X)
    children = tree["nodes"][0]["children"]
    assert len(children) == 2
    assert sorted(len(reached[child]) for child in children) == [5, 5]
    # The fallback isolates category/value 2; export and native routing agree.
    assert all(
        len(np.unique(categories[reached[child]])) == 1
        for child in children
        if 2 in categories[reached[child]]
    )
    replay = np.empty_like(y)
    for child in children:
        replay[reached[child]] = np.argmax(tree["nodes"][child]["class_counts"][0])
    np.testing.assert_array_equal(model.predict(X), replay)


@pytest.mark.parametrize("failed_value", [0.0, np.nan])
def test_failed_feature_does_not_make_node_leaf_and_cannot_enter_pairs(failed_value):
    X = np.column_stack([np.full(12, failed_value), np.repeat([0, 1], 6)])
    y = np.repeat([0, 1], 6)
    model = SGTClassifier(
        pairwise_candidates=10, min_samples_leaf=3, max_depth=1, tao_n_runs=0
    ).fit(X, y)
    root = model.tree_export()["nodes"][0]
    assert not root["is_leaf"]
    assert root.get("routing_kind") != "pair"
    assert root["feature"] == 1
    np.testing.assert_array_equal(model.predict(X), y)
    assert (
        SGTClassifier(min_samples_leaf=7, tao_n_runs=0)
        .fit(X, y)
        .tree_export()["nodes"][0]["is_leaf"]
    )


@pytest.mark.parametrize(
    "estimator",
    [SGTClassifier, SGTRegressor, RandomSGForestClassifier, RandomSGForestRegressor],
)
def test_smart_init_flag_is_removed(estimator):
    assert "coordinate_descent_smart_init" not in estimator().get_params()
    with pytest.raises(TypeError, match="coordinate_descent_smart_init"):
        estimator(coordinate_descent_smart_init=True)


@pytest.mark.parametrize(
    "criterion", ["gini", "entropy", "squared_error", "absolute_error"]
)
@pytest.mark.parametrize("missing", [0.0, np.nan])
@pytest.mark.parametrize("outputs", [1, 2])
def test_categorical_training_missing_rows_keep_scored_routing(
    criterion, missing, outputs
):
    X = np.array([[1, 0]] * 8 + [[0, 1]] * 2 + [[missing, missing]] * 2)
    y = np.array([1] * 8 + [0] * 4)
    if outputs == 2:
        y = np.column_stack([y, 1 - y])
    weights = np.full(12, 0.25)
    weights[-1] = 0
    estimator = SGTClassifier if criterion in ("gini", "entropy") else SGTRegressor
    model = estimator(
        criterion=criterion, max_depth=1, tao_n_runs=0, random_state=0
    ).fit(X, y, sample_weight=weights, feature_dict={"cat": [0, 1]})
    np.testing.assert_allclose(model.predict(X), y)
    tree = model.tree_export()
    reached = _route_samples(tree, X)
    for child in tree["nodes"][0]["children"]:
        rows = reached[child]
        node = tree["nodes"][child]
        if estimator is SGTClassifier:
            for output, counts in zip(
                y.reshape(len(X), outputs).T, node["class_counts"]
            ):
                np.testing.assert_allclose(
                    counts,
                    np.bincount(output[rows], weights=weights[rows], minlength=2),
                )
        else:
            assert len(rows) == node["n_samples"]


@pytest.mark.parametrize("criterion", ["squared_error", "absolute_error"])
@pytest.mark.parametrize("categorical", [False, True])
@pytest.mark.parametrize("mae_cd", ["0", "1"])
def test_regression_retains_feasible_binary_fallback(
    criterion, categorical, mae_cd, monkeypatch
):
    monkeypatch.setenv("SGTLEARN_MAE_CD", mae_cd)
    categories = np.repeat([0, 1, 2], [1, 4, 5])
    X = np.eye(3)[categories] if categorical else categories[:, None].astype(float)
    y = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    kwargs = {"feature_dict": {"category": [0, 1, 2]}} if categorical else {}
    model = SGTRegressor(
        criterion=criterion,
        min_samples_leaf=5,
        max_depth=1,
        inner_max_depth=1 if categorical else 3,
        num_partitions=4,
        tao_n_runs=0,
        random_state=0,
    ).fit(X, y, **kwargs)
    tree = model.tree_export()
    root = tree["nodes"][0]
    assert not root["is_leaf"]
    reached = _route_samples(tree, X)
    assert sorted(len(reached[child]) for child in root["children"]) == [5, 5]
    np.testing.assert_allclose(
        model.predict(X),
        np.where(categories == 2, 1, 2 if criterion == "squared_error" else 0),
    )


@pytest.mark.parametrize("criterion", ["squared_error", "absolute_error"])
@pytest.mark.parametrize("outputs", [1, 2])
@pytest.mark.parametrize("mae_cd", ["0", "1"])
def test_regression_preserves_binary_root_loss(criterion, outputs, mae_cd, monkeypatch):
    monkeypatch.setenv("SGTLEARN_MAE_CD", mae_cd)
    rng = np.random.default_rng(12)
    X = np.repeat(np.arange(10), 10).reshape(-1, 1).astype(float)
    y = rng.normal(size=(len(X), outputs))
    if outputs == 1:
        y = y[:, 0]
    weights = rng.uniform(0.01, 0.9, len(X))
    weights[::7] = 0
    stump = DecisionTreeRegressor(max_depth=1, criterion=criterion, random_state=0).fit(
        X, y, sample_weight=weights
    )

    def loss(prediction):
        residual = (y - prediction).reshape(len(X), outputs)
        error = residual**2 if criterion == "squared_error" else np.abs(residual)
        return np.average(error.sum(axis=1), weights=weights)

    for k in (2, 3, 5):
        model = SGTRegressor(
            criterion=criterion,
            num_partitions=k,
            max_depth=1,
            tao_n_runs=0,
            random_state=42,
        ).fit(X, y, sample_weight=weights)
        assert loss(model.predict(X)) <= loss(stump.predict(X)) + 1e-7


@pytest.mark.parametrize("criterion", ["squared_error", "absolute_error"])
@pytest.mark.parametrize("categorical", [False, True])
@pytest.mark.parametrize("mae_cd", ["0", "1"])
def test_regression_missing_root_respects_sample_counts(
    criterion, categorical, mae_cd, monkeypatch
):
    monkeypatch.setenv("SGTLEARN_MAE_CD", mae_cd)
    X = np.array([0.0] * 4 + [1.0] * 6 + [np.nan] * 2)[:, None]
    kwargs = {}
    if categorical:
        X = np.column_stack([X[:, 0] == 0, X[:, 0] == 1]).astype(float)
        kwargs = {"feature_dict": {"category": [0, 1]}}
    y = np.array([0.0] * 4 + [1.0] * 8)
    y = np.column_stack([y, 2 * y + 3])
    weights = np.full(12, 0.2)
    weights[0] = 0
    for scale in (1, 1e-6):
        model = SGTRegressor(
            criterion=criterion,
            min_samples_leaf=5,
            max_depth=1,
            num_partitions=3,
            tao_n_runs=0,
        ).fit(X, y, sample_weight=weights * scale, **kwargs)
        tree = model.tree_export()
        root = tree["nodes"][0]
        assert not root["is_leaf"]
        reached = _route_samples(tree, X)
        assert sorted(len(reached[child]) for child in root["children"]) == [6, 6]
        prediction = model.predict(X)
        np.testing.assert_allclose(prediction[:4], np.tile(prediction[-1], (4, 1)))
        np.testing.assert_allclose(prediction[4:10], y[4:10])
        expected = 0.4 if criterion == "squared_error" else 0
        np.testing.assert_allclose(
            prediction[-1], [expected, 2 * expected + 3], atol=1e-7
        )


def test_weighted_missing_multioutput_forest_predictions_are_normalized():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(50, 3))
    X[::4, 0] = np.nan
    y = rng.integers(0, 3, size=(50, 2))
    weights = rng.uniform(0.01, 0.8, len(X))
    weights[::9] = 0
    model = RandomSGForestClassifier(
        n_estimators=3,
        num_partitions=4,
        min_samples_leaf=3,
        max_depth=2,
        random_state=0,
        tao_n_runs=0,
        class_weight=[{0: 0.2, 1: 0.4, 2: 0.8}] * 2,
    ).fit(X, y, sample_weight=weights)
    for probabilities in model.predict_proba(X):
        assert np.isfinite(probabilities).all()
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
