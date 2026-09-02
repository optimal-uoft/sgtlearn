"""Unit tests for :mod:`sgtlearn.tao`."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Tuple

import numpy as np
import pytest
from sklearn.datasets import load_iris, make_regression
from sklearn.exceptions import NotFittedError

from sgtlearn import SGTClassifier, SGTRegressor, tao
from sgtlearn.ensemble import RandomSGForestClassifier, RandomSGForestRegressor
from tests.constants import TEST_TAO_N_RUNS

pytest.importorskip("sklearn")


def _fit_classifier(
    X: np.ndarray,
    y: np.ndarray,
    *,
    criterion: str = "gini",
    class_weight: Optional[Mapping[Any, float] | str] = None,
    sample_weight: Optional[np.ndarray] = None,
    **tree_kwargs: Any,
) -> SGTClassifier:
    params = dict(
        criterion=criterion,
        max_depth=4,
        min_samples_leaf=3,
        inner_max_depth=4,
        inner_max_leaf_nodes=16,
        random_state=42,
        class_weight=class_weight,
        tao_n_runs=TEST_TAO_N_RUNS,
    )
    params.update(tree_kwargs)
    est = SGTClassifier(**params)
    est.fit(X, y, sample_weight=sample_weight)
    return est


def _fit_regressor(
    X: np.ndarray,
    y: np.ndarray,
    *,
    criterion: str = "squared_error",
    sample_weight: Optional[np.ndarray] = None,
    **tree_kwargs: Any,
) -> SGTRegressor:
    params = dict(
        criterion=criterion,
        max_depth=4,
        min_samples_leaf=3,
        inner_max_depth=4,
        inner_max_leaf_nodes=16,
        random_state=42,
        tao_n_runs=TEST_TAO_N_RUNS,
    )
    params.update(tree_kwargs)
    est = SGTRegressor(**params)
    est.fit(X, y, sample_weight=sample_weight)
    return est


def _classification_data() -> Tuple[np.ndarray, np.ndarray]:
    X, y = load_iris(return_X_y=True)
    return np.asarray(X, dtype=np.float64), y


def _regression_data() -> Tuple[np.ndarray, np.ndarray]:
    X, y = make_regression(
        n_samples=400,
        n_features=10,
        n_informative=6,
        noise=5.0,
        random_state=0,
    )
    return np.asarray(X, dtype=np.float64), y


def test_feature_importances_are_unavailable_after_tao_and_reset_on_refit() -> None:
    X, y = load_iris(return_X_y=True)
    clf = SGTClassifier(tao_n_runs=0, random_state=0).fit(X, y)

    tao.TAO_refine(clf, X, y, n_runs=0)
    assert clf.feature_importances_.shape == (X.shape[1],)

    tao.TAO_refine(clf, X, y, n_runs=1)

    with pytest.raises(AttributeError, match="unavailable after TAO"):
        clf.feature_importances_

    clf.fit(X, y)
    assert clf.feature_importances_.shape == (X.shape[1],)


@pytest.mark.parametrize(
    ("estimator_cls", "fit_fn", "data_fn"),
    [
        pytest.param(
            SGTClassifier,
            lambda X, y: _fit_classifier(X, y),
            _classification_data,
            id="classifier",
        ),
        pytest.param(
            SGTRegressor,
            lambda X, y: _fit_regressor(X, y),
            _regression_data,
            id="regressor",
        ),
    ],
)
def test_tao_refine_mutates_in_place(
    estimator_cls: type,
    fit_fn: Callable[..., Any],
    data_fn: Callable[[], Tuple[np.ndarray, np.ndarray]],
) -> None:
    """TAO_refine returns the same wrapper and keeps the native handle."""
    X, y = data_fn()
    est = fit_fn(X, y)
    native_before = est._est

    result = tao.TAO_refine(est, X, y)

    assert result is est
    assert est._est is native_before


@pytest.mark.parametrize(
    ("estimator", "X", "y", "exc_type"),
    [
        pytest.param(
            SGTClassifier(),
            *load_iris(return_X_y=True),
            NotFittedError,
            id="unfitted_classifier",
        ),
        pytest.param(
            SGTRegressor(),
            *make_regression(n_samples=50, n_features=4, random_state=0),
            NotFittedError,
            id="unfitted_regressor",
        ),
    ],
)
def test_tao_rejects_unfitted(estimator, X, y, exc_type) -> None:
    with pytest.raises(exc_type):
        tao.TAO_refine(estimator, X, y)


def test_tao_rejects_wrong_type() -> None:
    X, y = load_iris(return_X_y=True)
    with pytest.raises(TypeError):
        tao.TAO_refine(object(), X, y)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("fit_fn", "data_fn"),
    [
        pytest.param(
            lambda X, y: _fit_classifier(X, y),
            _classification_data,
            id="classifier",
        ),
        pytest.param(
            lambda X, y: _fit_regressor(X, y),
            _regression_data,
            id="regressor",
        ),
    ],
)
def test_tao_rejects_feature_mismatch(fit_fn, data_fn) -> None:
    X, y = data_fn()
    est = fit_fn(X, y)
    with pytest.raises(ValueError):
        tao.TAO_refine(est, X[:, :-1], y)
    with pytest.raises(ValueError, match="samples"):
        tao.TAO_refine(est, X, y[:-1])


def test_tao_accepts_check_input_false() -> None:
    """Callers that pre-validate arrays can skip redundant checks."""
    X, y = _tao_pair_interaction_data()
    tree = SGTClassifier(
        max_depth=1,
        pairwise_candidates=1,
        pairwise_penalty=1.0,
        tao_n_runs=0,
        random_state=0,
    ).fit(X, y)
    tao.TAO_refine(tree, X, y, n_runs=1, check_input=False)
    assert tree.tree_export()["nodes"][0]["routing_kind"] == "pair"


@pytest.mark.parametrize(
    ("forest_cls", "target"),
    [
        (RandomSGForestClassifier, lambda y: y),
        (RandomSGForestRegressor, lambda y: np.column_stack([y, 10.0 + y])),
    ],
)
def test_tao_refines_every_forest_tree(forest_cls, target) -> None:
    X, labels = _tao_pair_interaction_data()
    y = target(labels.astype(float))
    forest = forest_cls(
        n_estimators=2,
        bootstrap=False,
        max_features=None,
        max_depth=1,
        inner_max_depth=2,
        inner_max_leaf_nodes=4,
        pairwise_candidates=1,
        pairwise_penalty=1.0,
        tao_n_runs=0,
        random_state=0,
        n_jobs=1,
    ).fit(X, y)
    handles_before = [est._est for est in forest.estimators_]

    result = tao.TAO_refine(forest, X, y, n_runs=1, lambda_=0.0, n_jobs=2)

    assert result is forest
    assert [est._est for est in forest.estimators_] == handles_before
    assert all(
        est.tree_export()["nodes"][0]["routing_kind"] == "pair"
        for est in forest.estimators_
    )


def _tao_pair_interaction_data() -> tuple[np.ndarray, np.ndarray]:
    quadrants = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    counts = [40, 10, 30, 5]
    return np.repeat(quadrants, counts, axis=0), np.repeat([0, 1, 1, 0], counts)


def test_tao_reconsiders_retained_classifier_pair() -> None:
    X, y = _tao_pair_interaction_data()
    clf = SGTClassifier(
        max_depth=1,
        inner_max_depth=2,
        inner_max_leaf_nodes=4,
        pairwise_candidates=1,
        pairwise_penalty=1.0,
        tao_n_runs=0,
        random_state=0,
    ).fit(X, y)

    assert clf.tree_export()["nodes"][0].get("routing_kind") != "pair"
    assert clf.score(X, y) == pytest.approx(70 / 85)

    tao.TAO_refine(clf, X, y, n_runs=1, lambda_=0.0, tao_pair_scale=1.1)

    root = clf.tree_export()["nodes"][0]
    assert root["routing_kind"] == "pair"
    assert root["pair_features"] == [0, 1]
    assert len(root["bin_sample_counts"]) == len(root["bin_to_partition"])
    assert len(root["bin_counts"]) == len(root["bin_to_partition"])
    assert sum(root["bin_sample_counts"]) == X.shape[0]
    with pytest.raises(AttributeError, match="unavailable after TAO"):
        clf.feature_importances_
    assert clf.score(X, y) == 1.0


def test_tao_weights_change_the_accepted_classifier_update() -> None:
    X, y = _tao_pair_interaction_data()
    sample_weight = np.ones(len(y))
    sample_weight[40:50] = 20.0

    def refine(weights: np.ndarray | None) -> SGTClassifier:
        clf = SGTClassifier(
            max_depth=1,
            inner_max_depth=2,
            inner_max_leaf_nodes=4,
            pairwise_candidates=1,
            pairwise_penalty=0.3,
            tao_n_runs=0,
            random_state=0,
        ).fit(X, y)
        return tao.TAO_refine(
            clf,
            X,
            y,
            sample_weight=weights,
            n_runs=1,
            lambda_=0.5,
        )

    unweighted = refine(None)
    weighted = refine(sample_weight)
    unweighted_score = np.average(unweighted.predict(X) == y, weights=sample_weight)
    weighted_score = np.average(weighted.predict(X) == y, weights=sample_weight)

    assert not np.array_equal(unweighted.predict(X), weighted.predict(X))
    assert weighted_score > unweighted_score


def test_tao_makes_forest_feature_importances_unavailable() -> None:
    X, y = _tao_pair_interaction_data()
    forest = RandomSGForestClassifier(
        n_estimators=1,
        bootstrap=False,
        max_features=None,
        max_depth=1,
        inner_max_depth=2,
        inner_max_leaf_nodes=4,
        pairwise_candidates=1,
        pairwise_penalty=1.0,
        tao_n_runs=0,
        random_state=0,
        n_jobs=1,
    ).fit(X, y)

    tao.TAO_refine(forest, X, y, n_runs=1, lambda_=0.0)

    for attr in ("mean_feature_importances_", "std_feature_importance_"):
        with pytest.raises(AttributeError, match="unavailable after TAO"):
            getattr(forest, attr)


def test_tao_accepts_improving_retained_regression_pair_multioutput() -> None:
    X, labels = _tao_pair_interaction_data()
    y = np.column_stack([labels.astype(float), 10.0 + labels])
    reg = SGTRegressor(
        max_depth=1,
        inner_max_depth=2,
        inner_max_leaf_nodes=4,
        pairwise_candidates=1,
        pairwise_penalty=1.0,
        tao_n_runs=0,
        random_state=0,
    ).fit(X, y)

    assert reg.tree_export()["nodes"][0].get("routing_kind") != "pair"
    assert not np.array_equal(reg.predict(X), y)

    tao.TAO_refine(reg, X, y, n_runs=1, lambda_=0.0)

    root = reg.tree_export()["nodes"][0]
    assert root["routing_kind"] == "pair"
    assert root["pair_features"] == [0, 1]
    assert len(root["bin_sample_counts"]) == len(root["bin_to_partition"])
    assert len(root["bin_counts"]) == len(root["bin_to_partition"])
    assert sum(root["bin_sample_counts"]) == X.shape[0]
    np.testing.assert_array_equal(reg.predict(X), y)


def test_tao_pair_scale_changes_pair_vs_dummy_choice() -> None:
    X, y = _tao_pair_interaction_data()

    def fit_pair() -> SGTClassifier:
        return SGTClassifier(
            max_depth=1,
            inner_max_depth=2,
            inner_max_leaf_nodes=4,
            pairwise_candidates=1,
            tao_n_runs=0,
            random_state=0,
        ).fit(X, y)

    default_scale = fit_pair()
    high_scale = fit_pair()
    tao.TAO_refine(default_scale, X, y, n_runs=1, lambda_=0.3, tao_pair_scale=1.1)
    tao.TAO_refine(high_scale, X, y, n_runs=1, lambda_=0.3, tao_pair_scale=2.0)

    assert default_scale.tree_export()["nodes"][0]["routing_kind"] == "pair"
    assert default_scale.score(X, y) == 1.0
    assert high_scale.tree_export()["nodes"][0].get("routing_kind") != "pair"
    assert high_scale.score(X, y) == pytest.approx(45 / 85)


@pytest.mark.parametrize(
    ("forest_cls", "target"),
    [
        (RandomSGForestClassifier, lambda y: y),
        (RandomSGForestRegressor, lambda y: y.astype(float)),
    ],
)
def test_tao_pair_scale_defaults_and_forwards_through_forests(
    forest_cls, target
) -> None:
    assert SGTClassifier().get_params()["tao_pair_scale"] == 1.1
    assert SGTRegressor().get_params()["tao_pair_scale"] == 1.1
    assert forest_cls().get_params()["tao_pair_scale"] == 1.1

    X, y = _tao_pair_interaction_data()
    forest = forest_cls(
        n_estimators=2,
        bootstrap=False,
        max_features=None,
        pairwise_candidates=1,
        tao_n_runs=0,
        tao_pair_scale=1.7,
        random_state=0,
        n_jobs=1,
    ).fit(X, target(y))

    assert forest.tao_pair_scale == 1.7
    assert all(tree.tao_pair_scale == 1.7 for tree in forest.estimators_)


@pytest.mark.parametrize("bad_scale", [-1.0, np.inf, np.nan])
def test_tao_pair_scale_rejects_invalid_values(bad_scale: float) -> None:
    X, y = _tao_pair_interaction_data()
    with pytest.raises(ValueError, match="tao_pair_scale"):
        SGTClassifier(tao_n_runs=0, tao_pair_scale=bad_scale).fit(X, y)

    clf = SGTClassifier(tao_n_runs=0).fit(X, y)
    with pytest.raises(ValueError, match="tao_pair_scale"):
        tao.TAO_refine(clf, X, y, tao_pair_scale=bad_scale)
