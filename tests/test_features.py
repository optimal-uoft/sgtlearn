"""Tests for logical feature configuration and categorical routing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from sgtlearn import SGTClassifier, configure_feature_dict, plot_tree
from sgtlearn._export import _is_categorical_node
from sgtlearn._features import FeatureDict, ProcessedFeatures


def test_configure_feature_dict_defaults_to_continuous_columns() -> None:
    pf = configure_feature_dict(4)
    assert len(pf.features) == 4
    assert all(f["type"] == "continuous" for f in pf.features)
    assert [f["indices"] for f in pf.features] == [[0], [1], [2], [3]]
    assert pf.logical_names == ("0", "1", "2", "3")
    assert pf.to_native() == pf.features


def test_configure_feature_dict_fills_unmentioned_columns() -> None:
    pf = configure_feature_dict(
        5,
        feature_dict={
            0: [0, 1, 2],
            4: [4],
        },
    )
    assert pf.features == [
        {"type": "categorical", "indices": [0, 1, 2]},
        {"type": "continuous", "indices": [3]},
        {"type": "continuous", "indices": [4]},
    ]
    assert pf.logical_names == ("0", "3", "4")


def test_configure_feature_dict_rejects_duplicate_indices() -> None:
    with pytest.raises(ValueError, match="unique"):
        configure_feature_dict(4, feature_dict={0: [0], 1: [0]})


@pytest.mark.parametrize(
    ("feature_dict", "column_names", "match"),
    [
        ({"bad": [-1]}, None, "out of range"),
        ({"bad": [3]}, None, "out of range"),
        ({"bad": ["a"]}, None, "column_names"),
        ({"bad": ["missing"]}, ["a", "b", "c"], "not found"),
    ],
)
def test_configure_feature_dict_rejects_invalid_columns(
    feature_dict: FeatureDict,
    column_names: list[str] | None,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        configure_feature_dict(3, feature_dict, column_names=column_names)


def test_configure_feature_dict_classifies_groups_at_two_columns() -> None:
    pf = configure_feature_dict(3, {"one": [0], "two": [1, 2]})
    by_name = dict(zip(pf.logical_names, pf.features))
    assert by_name["one"]["type"] == "continuous"
    assert by_name["two"]["type"] == "categorical"


def test_configure_feature_dict_string_keys_and_column_names() -> None:
    pf = configure_feature_dict(
        4,
        feature_dict={"color": ["a", "b", "c"]},
        column_names=["a", "b", "c", "d"],
    )
    by_indices = {tuple(f["indices"]): f for f in pf.features}
    assert by_indices[(0, 1, 2)]["type"] == "categorical"
    assert by_indices[(3,)]["type"] == "continuous"


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
    has_categorical_split = any(_is_categorical_node(n) for n in internal)
    assert has_categorical_split


def test_plot_tree_categorical_node_labels_merged_categories() -> None:
    rng = np.random.default_rng(7)
    n_samples = 500
    n_cat = 6
    cats = rng.integers(0, n_cat, size=n_samples)
    columns = [f"cat_{i}" for i in range(n_cat)]
    X = pd.DataFrame(0.0, index=np.arange(n_samples), columns=columns)
    for i, c in enumerate(cats):
        X.iloc[i, c] = 1.0
    y = (cats % 2).astype(np.int64)

    clf = SGTClassifier(
        max_depth=3,
        inner_max_depth=2,
        inner_max_leaf_nodes=3,
        random_state=0,
        tao_n_runs=0,
        max_features=None,
    )
    clf.fit(X, y, feature_dict={"species": list(columns)})

    artists = plot_tree(clf)
    labels = [
        tick.get_text()
        for artist in artists
        if hasattr(artist, "get_xticklabels")
        for tick in artist.get_xticklabels()
    ]
    merged = [t for t in labels if t.startswith("[") and "," in t]
    assert merged, "expected merged-category bucket labels when leaf budget is tight"
    assert any(label == "cat_5" for label in labels)


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

    monkeypatch.setattr("sgtlearn.base.configure_feature_dict", _spy)

    X, y = make_classification(
        n_samples=100, n_features=5, n_informative=3, random_state=0
    )
    forest = RandomSGForestClassifier(
        n_estimators=3, random_state=0, tao_n_runs=0, n_jobs=1
    )
    forest.fit(X, y, feature_dict={i: [i] for i in range(5)})
    assert calls == [5]
    assert forest.processed_features_ is not None


def test_ensemble_dataframe_string_valued_feature_dict_routes_onehot() -> None:
    """Forest + DataFrame + string column-name feature_dict must resolve names.

    Regression: the forest read column names *after* check_X_y stripped them to
    an ndarray, so a categorical block referenced by name raised
    "column name ... requires a pandas DataFrame or column_names".
    """
    from sgtlearn.ensemble import RandomSGForestClassifier

    rng = np.random.default_rng(2026)
    n_samples = 400
    n_cat = 4
    cats = rng.integers(0, n_cat, size=n_samples)
    columns = [f"cat_{i}" for i in range(n_cat)]
    X = pd.DataFrame(0.0, index=np.arange(n_samples), columns=columns)
    for i, c in enumerate(cats):
        X.iloc[i, c] = 1.0
    y = (cats % 2).astype(np.int64)

    forest = RandomSGForestClassifier(
        n_estimators=4,
        max_depth=4,
        inner_max_depth=2,
        inner_max_leaf_nodes=8,
        random_state=0,
        tao_n_runs=0,
        n_jobs=1,
        max_features=None,
    )
    forest.fit(X, y, feature_dict={"species": list(columns)})

    assert forest.predict(X).shape[0] == n_samples
    pf = forest.processed_features_
    assert pf is not None
    assert "species" in pf.logical_names
    by_indices = {tuple(f["indices"]): f["type"] for f in pf.features}
    assert by_indices[tuple(range(n_cat))] == "categorical"
    assert any(
        any(
            _is_categorical_node(n)
            for n in t.tree_export()["nodes"]
            if not n["is_leaf"]
        )
        for t in forest.estimators_
    ), "expected a categorical split in at least one tree"
