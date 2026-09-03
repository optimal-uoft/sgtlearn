"""Smoke tests for ``sgtlearn._export.plot_tree``."""

from __future__ import annotations

import matplotlib
import numpy as np
from sklearn.tree import DecisionTreeClassifier

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.exceptions import NotFittedError
from sgtlearn import SGTClassifier, SGTRegressor, plot_tree

from tests.constants import TEST_TAO_N_RUNS


@pytest.fixture
def fitted_classifier():
    X, y = make_classification(n_samples=200, n_features=4, random_state=0)
    return SGTClassifier(
        max_depth=3,
        inner_max_depth=2,
        inner_max_leaf_nodes=8,
        random_state=0,
        tao_n_runs=TEST_TAO_N_RUNS,
    ).fit(X, y)


@pytest.fixture
def fitted_regressor():
    X, y = make_regression(n_samples=200, n_features=4, random_state=0)
    return SGTRegressor(
        max_depth=3,
        inner_max_depth=2,
        inner_max_leaf_nodes=8,
        random_state=0,
        tao_n_runs=TEST_TAO_N_RUNS,
    ).fit(X, y)


def test_plot_tree_classifier_returns_artists(fitted_classifier):
    artists = plot_tree(fitted_classifier)
    assert isinstance(artists, list)
    assert len(artists) > 0
    plt.close("all")


def test_plot_tree_regressor_returns_artists(fitted_regressor):
    artists = plot_tree(fitted_regressor)
    assert isinstance(artists, list)
    assert len(artists) > 0
    plt.close("all")


def test_plot_tree_unfitted_classifier_raises():
    with pytest.raises(NotFittedError):
        plot_tree(SGTClassifier())


def test_plot_tree_unfitted_regressor_raises():
    with pytest.raises(NotFittedError):
        plot_tree(SGTRegressor())


def test_plot_tree_rejects_non_sgt_estimator():
    with pytest.raises(TypeError):
        plot_tree(DecisionTreeClassifier())


def test_plot_tree_max_depth_reduces_artist_count(fitted_classifier):
    full = plot_tree(fitted_classifier)
    plt.close("all")
    shallow = plot_tree(fitted_classifier, max_depth=1)
    plt.close("all")
    assert len(shallow) < len(full)


def test_plot_tree_leaves_render_text(fitted_classifier):
    artists = plot_tree(fitted_classifier)
    text_artists = [a for a in artists if hasattr(a, "get_text") and a.get_text()]
    assert text_artists, "expected at least one text artist (leaf label)"
    plt.close("all")


def test_plot_tree_regressor_leaves_show_value(fitted_regressor):
    artists = plot_tree(fitted_regressor)
    text_artists = [a for a in artists if hasattr(a, "get_text")]

    def _is_number(s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            return False

    assert any(_is_number(t.get_text()) for t in text_artists)
    plt.close("all")


def test_plot_tree_label_all_adds_metadata(fitted_classifier):
    artists = plot_tree(fitted_classifier, label="all", impurity=True)
    text_artists = [a for a in artists if hasattr(a, "get_text")]
    texts = [t.get_text() for t in text_artists]
    assert any("n = " in t for t in texts)
    assert any("gini = " in t for t in texts)
    plt.close("all")


def test_plot_tree_label_none_suppresses_subtitles(fitted_classifier):
    artists = plot_tree(fitted_classifier, label="none")
    text_artists = [a for a in artists if hasattr(a, "get_text")]
    for t in text_artists:
        s = t.get_text()
        assert "n = " not in s
        assert "n=" not in s
        assert "gini = " not in s
    plt.close("all")


def test_plot_tree_custom_feature_names(fitted_classifier):
    names = [f"feat_{i}" for i in range(fitted_classifier.n_features_in_)]
    artists = plot_tree(fitted_classifier, feature_names=names)
    text_artists = [a for a in artists if hasattr(a, "get_text")]
    assert any("feat_" in t.get_text() for t in text_artists)
    plt.close("all")


def test_plot_tree_explicit_feature_names_override_stored_names():
    X = pd.DataFrame({"stored_name": np.arange(20, dtype=float)})
    y = np.repeat([0, 1], 10)
    estimator = SGTClassifier(max_depth=1, tao_n_runs=0, random_state=0).fit(
        X, y, feature_dict={"stored_name": ["stored_name"]}
    )

    stored = plot_tree(estimator)
    explicit = plot_tree(estimator, feature_names=["explicit_name"])
    stored_text = " ".join(
        [a.get_text() for a in stored if hasattr(a, "get_text")]
        + [a.get_xlabel() for a in stored if hasattr(a, "get_xlabel")]
    )
    explicit_text = " ".join(
        [a.get_text() for a in explicit if hasattr(a, "get_text")]
        + [a.get_xlabel() for a in explicit if hasattr(a, "get_xlabel")]
    )

    assert "stored_name" in stored_text
    assert "explicit_name" in explicit_text
    assert "stored_name" not in explicit_text
    plt.close("all")


def test_plot_tree_class_names(fitted_classifier):
    artists = plot_tree(fitted_classifier, class_names=["neg", "pos"])
    text_artists = [a for a in artists if hasattr(a, "get_text")]
    leaf_text = " ".join(a.get_text() for a in text_artists)
    assert "neg" in leaf_text or "pos" in leaf_text
    plt.close("all")


def test_plot_tree_reuses_existing_axes(fitted_classifier):
    fig, ax = plt.subplots()
    artists = plot_tree(fitted_classifier, ax=ax)
    assert artists
    # The host ax should still be the one we passed.
    assert ax in fig.axes
    plt.close("all")


def test_plot_tree_with_X_renders_fine_histograms(fitted_classifier):
    X, _ = make_classification(n_samples=200, n_features=4, random_state=0)
    without_histograms = plot_tree(fitted_classifier)
    plt.close("all")
    with_histograms = plot_tree(fitted_classifier, X=X)
    plt.close("all")

    def patch_count(artists):
        return sum(
            len(artist.patches) for artist in artists if hasattr(artist, "patches")
        )

    assert patch_count(with_histograms) > patch_count(without_histograms)


def test_plot_tree_X_shape_mismatch_raises(fitted_classifier):
    bad_X = np.zeros((10, fitted_classifier.n_features_in_ + 1))
    with pytest.raises(ValueError):
        plot_tree(fitted_classifier, X=bad_X)

    with pytest.raises(ValueError, match="X must be 2-D"):
        plot_tree(fitted_classifier, X=np.zeros(fitted_classifier.n_features_in_))


def test_plot_tree_pair_heatmap_reuses_exported_router_and_axes():
    states = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    X = np.repeat(states, [40, 30, 30, 28], axis=0)
    y = np.repeat([0, 1, 1, 0], [40, 30, 30, 28])
    est = SGTClassifier(
        max_depth=1,
        inner_max_depth=2,
        inner_max_leaf_nodes=4,
        pairwise_candidates=1,
        tao_n_runs=0,
        random_state=0,
    ).fit(X, y)
    fig, ax = plt.subplots()
    artists = plot_tree(
        est,
        X=X,
        feature_names=["first", "second"],
        precision=3,
        ax=ax,
    )
    panels = [artist for artist in artists if hasattr(artist, "patches")]
    assert ax in fig.axes
    assert panels and len(panels[0].patches) == 4
    assert panels[0].get_xlabel() == "first"
    assert panels[0].get_ylabel() == "second"
    plt.close(fig)


@pytest.mark.parametrize("pair_kind", ["mixed", "categorical"])
def test_plot_tree_pair_heatmap_renders_categories_and_missing_cells(pair_kind):
    categories = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
    if pair_kind == "mixed":
        states = np.array(
            [
                [value, *category]
                for value in [-1.0, 1.0, np.nan]
                for category in categories
            ]
        )
        feature_dict = {0: [0], 1: [1, 2]}
    else:
        states = np.array(
            [[*first, *second] for first in categories for second in categories]
        )
        feature_dict = {0: [0, 1], 1: [2, 3]}
    X = np.repeat(states, [40, 31, 30, 29, 28, 27, 26, 25, 24], axis=0)
    y = np.repeat(np.arange(9), [40, 31, 30, 29, 28, 27, 26, 25, 24])
    est = SGTClassifier(
        num_partitions=9,
        max_depth=1,
        inner_max_depth=2,
        inner_max_leaf_nodes=5,
        pairwise_candidates=1,
        tao_n_runs=0,
        random_state=0,
    ).fit(X, y, feature_dict=feature_dict)

    fig, ax = plt.subplots()
    artists = plot_tree(est, X=X, ax=ax)
    panel = next(artist for artist in artists if hasattr(artist, "patches"))
    assert len(panel.patches) == 9  # 3 × 3, including both missing margins/corner
    x_labels = [tick.get_text() for tick in panel.get_xticklabels()]
    y_labels = [tick.get_text() for tick in panel.get_yticklabels()]
    assert "NaN" in x_labels
    assert "NaN" in y_labels
    if pair_kind == "mixed":
        assert any(label not in {"", "NaN"} for label in x_labels)
    plt.close(fig)


def test_plot_tree_pair_heatmap_renders_without_training_data():
    states = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    X = np.repeat(states, [40, 30, 30, 28], axis=0)
    y = np.repeat([0, 1, 1, 0], [40, 30, 30, 28])
    est = SGTClassifier(
        max_depth=1,
        inner_max_depth=2,
        inner_max_leaf_nodes=4,
        pairwise_candidates=1,
        tao_n_runs=0,
        random_state=0,
    ).fit(X, y)

    fig, ax = plt.subplots()
    artists = plot_tree(est, ax=ax)
    panel = next(artist for artist in artists if hasattr(artist, "patches"))
    assert panel.patches
    assert "NaN" in [tick.get_text() for tick in panel.get_xticklabels()]
    assert "NaN" in [tick.get_text() for tick in panel.get_yticklabels()]
    plt.close(fig)
