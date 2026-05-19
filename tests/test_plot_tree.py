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
