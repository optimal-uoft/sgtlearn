"""Tests for logical feature configuration and categorical routing."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification

from sgtlearn import SGTClassifier, configure_feature_dict
from sgtlearn._features import ProcessedFeatures


def test_configure_feature_dict_defaults_to_continuous_columns() -> None:
    pf = configure_feature_dict(4)
    assert len(pf.features) == 4
    assert all(f["type"] == "continuous" for f in pf.features)
    assert [f["indices"] for f in pf.features] == [[0], [1], [2], [3]]


def test_configure_feature_dict_fills_unmentioned_columns() -> None:
    pf = configure_feature_dict(
        5,
        feature_dict={
            0: [0, 1, 2],
            4: [4],
        },
    )
    types = {tuple(f["indices"]): f["type"] for f in pf.features}
    assert types[(0, 1, 2)] == "categorical"
    assert types[(4,)] == "continuous"
    assert types[(3,)] == "continuous"


def test_configure_feature_dict_legacy_layout() -> None:
    pf = configure_feature_dict(4, feature_dict={0: [0, 1, 2]})
    by_indices = {tuple(f["indices"]): f["type"] for f in pf.features}
    assert by_indices[(0, 1, 2)] == "categorical"
    assert by_indices[(3,)] == "continuous"


def test_configure_feature_dict_rejects_duplicate_indices() -> None:
    with pytest.raises(ValueError, match="unique"):
        configure_feature_dict(4, feature_dict={0: [0], 1: [0]})


def test_sgt_classifier_categorical_feature_group_routes_onehot() -> None:
    """A categorical feature block should use the one-hot inner discretizer."""
    rng = np.random.default_rng(2026)
    n_samples = 400
    n_cat = 4
    cats = rng.integers(0, n_cat, size=n_samples)
    X = np.zeros((n_samples, n_cat), dtype=np.float32)
    X[np.arange(n_samples), cats] = 1.0
    y = (cats % 2).astype(np.int64)

    clf = SGTClassifier(
        max_depth=4,
        inner_max_depth=2,
        inner_max_leaf_nodes=8,
        random_state=0,
        tao_n_runs=0,
        max_features=None,
    )
    clf.fit(X, y, feature_dict={0: list(range(n_cat))})

    tree = clf.tree_export()
    internal = [n for n in tree["nodes"] if not n["is_leaf"]]
    assert internal, "expected at least one split"
    has_categorical_split = any(len(n.get("features") or []) > 1 for n in internal)
    assert has_categorical_split


def test_processed_features_reused_without_reconfiguring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def _spy(n_features: int, **kwargs: object) -> ProcessedFeatures:
        calls.append(n_features)
        return configure_feature_dict(n_features, **kwargs)

    monkeypatch.setattr("sgtlearn.base.configure_feature_dict", _spy)

    X, y = make_classification(
        n_samples=120, n_features=4, n_informative=2, random_state=0
    )
    pf = configure_feature_dict(X.shape[1])
    clf = SGTClassifier(random_state=0, tao_n_runs=0)
    clf.fit(X, y, processed_features=pf)
    assert calls == []
    assert clf.processed_features_ is pf


def test_ensemble_resolves_features_once(monkeypatch: pytest.MonkeyPatch) -> None:
    from sgtlearn.ensemble import RandomSGForestClassifier

    calls: list[int] = []

    def _spy(n_features: int, **kwargs: object) -> ProcessedFeatures:
        calls.append(n_features)
        return configure_feature_dict(n_features, **kwargs)

    monkeypatch.setattr(
        "sgtlearn.ensemble._random_sgforest.configure_feature_dict", _spy
    )

    X, y = make_classification(
        n_samples=100, n_features=5, n_informative=3, random_state=0
    )
    forest = RandomSGForestClassifier(
        n_estimators=3, random_state=0, tao_n_runs=0, n_jobs=1
    )
    forest.fit(X, y, feature_dict={i: [i] for i in range(5)})
    assert calls == [5]
    assert forest.processed_features_ is not None
