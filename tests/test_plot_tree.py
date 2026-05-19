"""Smoke tests for ``sgtlearn._export.plot_tree``."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.exceptions import NotFittedError

from sgtlearn import SGTClassifier, SGTRegressor, plot_tree


@pytest.fixture
def fitted_classifier():
    X, y = make_classification(n_samples=200, n_features=4, random_state=0)
    return SGTClassifier(max_depth=3, inner_max_depth=2,
                         inner_max_leaf_nodes=8, random_state=0).fit(X, y)


@pytest.fixture
def fitted_regressor():
    X, y = make_regression(n_samples=200, n_features=4, random_state=0)
    return SGTRegressor(max_depth=3, inner_max_depth=2,
                        inner_max_leaf_nodes=8, random_state=0).fit(X, y)


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
    from sklearn.tree import DecisionTreeClassifier
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
    text_artists = [a for a in artists if hasattr(a, "get_text")]
    assert text_artists, "expected at least one text artist (leaf box)"
    samples_texts = [t.get_text() for t in text_artists if "samples" in t.get_text()]
    assert samples_texts
    plt.close("all")


def test_plot_tree_regressor_leaves_show_value(fitted_regressor):
    artists = plot_tree(fitted_regressor)
    text_artists = [a for a in artists if hasattr(a, "get_text")]
    assert any("value =" in t.get_text() for t in text_artists)
    plt.close("all")


def test_plot_tree_draws_edges(fitted_classifier):
    from matplotlib.lines import Line2D
    artists = plot_tree(fitted_classifier)
    lines = [a for a in artists if isinstance(a, Line2D)]
    # At least one edge per internal node, at least num_partitions in total.
    assert len(lines) >= 1
    plt.close("all")
