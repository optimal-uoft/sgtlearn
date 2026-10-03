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
    with pytest.raises(ValueError, match="2 outputs"):
        tao.TAO_refine(est, X, np.column_stack([y, y]))
    multi = fit_fn(X, np.column_stack([y, y]))
    with pytest.raises(ValueError, match="1 outputs"):
        tao.TAO_refine(multi, X, y)


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
        pairwise_penalty=85.0,  # Former normalized cost 1.0 × 85 samples.
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
        pairwise_penalty=42.5,  # Former cost 1.0 × 85 samples / 2 outputs.
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


def test_fit_with_tao_accepts_string_labels() -> None:
    X, y = load_iris(return_X_y=True)
    labels = np.array(["setosa", "versicolor", "virginica"])[y]
    clf = SGTClassifier(tao_n_runs=1, max_depth=2).fit(X, labels)
    assert set(clf.predict(X)) <= set(labels)


def test_fit_with_tao_applies_class_weight_once(monkeypatch) -> None:
    seen = []
    real = tao.TreeAlternatingOptimization

    def spy(est, X, y, sw, **kw):
        seen.append(np.asarray(sw, dtype=np.float64).copy())
        return real(est, X, y, sw, **kw)

    monkeypatch.setattr(tao, "TreeAlternatingOptimization", spy)
    X, y = load_iris(return_X_y=True)
    y = (y > 0).astype(int)
    SGTClassifier(class_weight={0: 1.0, 1: 3.0}, tao_n_runs=1, max_depth=2).fit(X, y)
    sw = seen[0]
    assert sw[y == 1][0] / sw[y == 0][0] == pytest.approx(3.0)


def _square_in_cross_data(
    seed: int = 0, n_samples: int = 2400, n_features: int = 2, noise: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Square-in-cross labels: several children of one 3-way split share a class."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2.5, 2.5, size=(n_samples, n_features))
    ix = np.abs(X[:, 0]) < 0.85
    iy = np.abs(X[:, 1]) < 0.85
    y = np.where(ix & iy, 2, np.where(ix | iy, 1, 0))
    flip = rng.random(n_samples) < noise
    y[flip] = rng.integers(0, 3, flip.sum())
    return X, y


def test_tao_lambda0_keeps_accuracy_when_children_share_a_class() -> None:
    X, y = _square_in_cross_data()
    clf = SGTClassifier(
        max_depth=1,
        inner_max_depth=4,
        inner_max_leaf_nodes=9,
        num_partitions=3,
        pairwise_candidates=0,
        tao_n_runs=0,
        random_state=0,
    ).fit(X, y)
    before = clf.score(X, y)

    tao.TAO_refine(clf, X, y, n_runs=10, lambda_=0.0)

    assert clf.score(X, y) >= before


@pytest.mark.parametrize("tao_n_runs", [0, None], ids=["no_fit_tao", "fit_tao"])
@pytest.mark.parametrize("weighting", ["none", "sample_weight", "class_weight"])
@pytest.mark.parametrize("num_partitions", [2, 3, 4])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_tao_lambda0_never_lowers_weighted_training_accuracy(
    seed: int, num_partitions: int, weighting: str, tao_n_runs: Optional[int]
) -> None:
    X, y = _square_in_cross_data(seed, n_samples=600, n_features=3, noise=0.1)
    sample_weight = None
    class_weight = None
    effective = np.ones(len(y))
    if weighting == "sample_weight":
        sample_weight = np.random.default_rng(seed + 100).uniform(0.2, 3.0, len(y))
        effective = sample_weight
    elif weighting == "class_weight":
        # Same weights as sklearn's "balanced" (not accepted by SGTClassifier).
        counts = np.bincount(y)
        class_weight = {c: len(y) / (len(counts) * counts[c]) for c in range(len(counts))}
        effective = np.array([class_weight[c] for c in y])
    # TAO sees float32 weights.
    effective = effective.astype(np.float32).astype(np.float64)

    params: dict[str, Any] = dict(
        max_depth=1,
        num_partitions=num_partitions,
        inner_max_depth=4,
        inner_max_leaf_nodes=9,
        class_weight=class_weight,
        random_state=seed,
    )
    if tao_n_runs is not None:
        params["tao_n_runs"] = tao_n_runs
    clf = SGTClassifier(**params).fit(X, y, sample_weight=sample_weight)
    before = np.average(clf.predict(X) == y, weights=effective)

    tao.TAO_refine(clf, X, y, sample_weight=sample_weight, lambda_=0.0)

    after = np.average(clf.predict(X) == y, weights=effective)
    assert after >= before - 1e-12


def _k2_threshold_data(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """On x0, Gini's best cut isolates a pure block (600/800 correct); cutting
    after the mixed middle block is more accurate (605/800). x1 is noise."""
    rng = np.random.default_rng(seed)
    blocks = [(0.0, 200, 0), (1.0, 105, 0), (1.0, 100, 1), (2.0, 95, 0), (2.0, 300, 1)]
    x0 = np.concatenate([lo + rng.integers(0, 10, n) / 10 for lo, n, _ in blocks])
    y = np.concatenate([np.full(n, label) for _, n, label in blocks])
    x1 = rng.integers(0, 100, len(y)) / 10
    return np.column_stack([x0, x1]), y


def _best_single_threshold_accuracy(
    X: np.ndarray, y: np.ndarray, labels: tuple[int, int]
) -> float:
    """Best accuracy of routing by one threshold to two leaves with fixed labels.

    Assumes finite values whose distinct values stay more than 1e-7 apart in
    float32, as in ``_k2_threshold_data``; it does not model NaN routing or the
    splitter's tie rule.
    """
    hits = np.column_stack([y == labels[0], y == labels[1]]).astype(float)
    total = hits.sum(axis=0)
    best = total.max()  # constant routing
    for f in range(X.shape[1]):
        order = np.argsort(X[:, f], kind="stable")
        xs = X[order, f]
        left = np.cumsum(hits[order], axis=0)
        cuts = np.flatnonzero(xs[1:] > xs[:-1])  # left side = order[: cut + 1]
        keep = left[cuts, 0] + total[1] - left[cuts, 1]
        swap = left[cuts, 1] + total[0] - left[cuts, 0]
        best = max(best, keep.max(initial=0.0), swap.max(initial=0.0))
    return best / len(y)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_tao_two_children_reach_the_best_single_threshold(seed: int) -> None:
    X, y = _k2_threshold_data(seed)
    clf = SGTClassifier(
        max_depth=1,
        num_partitions=2,
        inner_max_depth=1,
        inner_max_leaf_nodes=2,
        pairwise_candidates=0,
        tao_n_runs=0,
        random_state=seed,
    ).fit(X, y)
    export = clf.tree_export()
    root = export["nodes"][export["root_index"]]
    assert len(root["children"]) == 2
    labels = tuple(
        int(np.argmax(export["nodes"][cid]["class_counts"][0]))
        for cid in root["children"]
    )
    target = _best_single_threshold_accuracy(X, y, labels)

    tao.TAO_refine(clf, X, y, lambda_=0.0)

    assert clf.score(X, y) >= target - 1e-12


@pytest.mark.parametrize("criterion", ["squared_error", "absolute_error"])
@pytest.mark.parametrize("num_partitions", [2, 3])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_tao_lambda0_never_raises_regression_training_loss(
    seed: int, num_partitions: int, criterion: str
) -> None:
    X, y = make_regression(
        n_samples=300, n_features=4, n_informative=3, noise=15.0, random_state=seed
    )
    reg = SGTRegressor(
        criterion=criterion,
        max_depth=2,
        num_partitions=num_partitions,
        inner_max_depth=3,
        inner_max_leaf_nodes=8,
        tao_n_runs=0,
        random_state=seed,
    ).fit(X, y)
    if criterion == "squared_error":
        loss = lambda: np.mean((reg.predict(X) - y) ** 2)  # noqa: E731
    else:
        loss = lambda: np.mean(np.abs(reg.predict(X) - y))  # noqa: E731
    before = loss()

    tao.TAO_refine(reg, X, y, lambda_=0.0)

    # Native targets are float32, so allow float32-level slack.
    assert loss() <= before * (1 + 1e-6)
