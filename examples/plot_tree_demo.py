"""Demonstration of ``sgtlearn.plot_tree``.

Runs through the main knobs of the visualization API on two real sklearn
datasets:

- ``load_breast_cancer`` for ``SGTClassifier`` (binary classification).
- ``load_diabetes`` for ``SGTRegressor`` (continuous regression target).
- A synthetic one-hot categorical block for categorical shape-function routing.

Each section saves a PNG to ``OUT_DIR`` (default: alongside this script).
Run from the repo root:

    python examples/plot_tree_demo.py

or with a custom output directory:

    python examples/plot_tree_demo.py /tmp/my_plots
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render without a display server
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes

from sgtlearn import SGTClassifier, SGTRegressor, plot_tree

OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, name: str) -> Path:
    """Save ``fig`` as ``OUT_DIR/name`` at 140 DPI and close the figure."""
    path = OUT_DIR / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
    return path


# ---------------------------------------------------------------------------
# 1. Classifier — full styling with X passed for fine histograms.
# ---------------------------------------------------------------------------
# Passing ``X`` routes training samples through the tree at plot time, so each
# internal panel gets a ~20-bin density histogram overlaid on the routing
# slabs. Without ``X``, panels show the slabs only.

data = load_breast_cancer()
X_c, y_c = data.data, data.target

clf = SGTClassifier(
    max_depth=3,
    inner_max_depth=2,
    inner_max_leaf_nodes=8,
    min_samples_leaf=5,
    random_state=42,
).fit(X_c, y_c)

plot_tree(
    clf,
    X=X_c,
    feature_names=list(data.feature_names),
    class_names=["malignant", "benign"],
    label="all",  # show n=N under each leaf
    fontsize=10,
)
save(plt.gcf(), "01_classifier_full.png")


# ---------------------------------------------------------------------------
# 2. Classifier — slabs-only (no ``X``), minimal labels.
# ---------------------------------------------------------------------------
# Skip ``X`` to get a faster, lighter-weight view. ``label="feature"`` (default)
# suppresses the per-leaf sample count.

plot_tree(
    clf,
    feature_names=list(data.feature_names),
    class_names=["malignant", "benign"],
    fontsize=10,
)
save(plt.gcf(), "02_classifier_slabs_only.png")


# ---------------------------------------------------------------------------
# 3. Classifier — depth truncation via ``max_depth``.
# ---------------------------------------------------------------------------
# Limit how many tiers are drawn. Internal nodes at the cap render as
# "draw-leaves" (their bold label is the majority class of the subtree).

plot_tree(
    clf,
    X=X_c,
    feature_names=list(data.feature_names),
    class_names=["malignant", "benign"],
    max_depth=2,
    fontsize=10,
)
save(plt.gcf(), "03_classifier_max_depth_2.png")


# ---------------------------------------------------------------------------
# 4. Regressor — diabetes dataset, full styling.
# ---------------------------------------------------------------------------
# Leaves show the predicted target value in bold (instead of a class name).

reg_data = load_diabetes()
X_r, y_r = reg_data.data, reg_data.target

reg = SGTRegressor(
    max_depth=3,
    inner_max_depth=2,
    inner_max_leaf_nodes=8,
    min_samples_leaf=10,
    random_state=42,
).fit(X_r, y_r)

plot_tree(
    reg,
    X=X_r,
    feature_names=list(reg_data.feature_names),
    label="all",
    fontsize=10,
)
save(plt.gcf(), "04_regressor_full.png")


# ---------------------------------------------------------------------------
# 5. Custom matplotlib axes — embed plot_tree inside a larger figure.
# ---------------------------------------------------------------------------
# Passing ``ax=`` lets you compose the tree alongside other plots. Useful for
# paper figures with model diagnostics next to the tree itself.

fig, axes = plt.subplots(1, 2, figsize=(22, 7))
plot_tree(
    clf,
    X=X_c,
    feature_names=list(data.feature_names),
    class_names=["malignant", "benign"],
    ax=axes[0],
)
axes[0].set_title("SGTClassifier on breast_cancer")

plot_tree(
    reg,
    X=X_r,
    feature_names=list(reg_data.feature_names),
    ax=axes[1],
)
axes[1].set_title("SGTRegressor on diabetes")
save(fig, "05_side_by_side.png")


# ---------------------------------------------------------------------------
# 6. Custom palette — pass any matplotlib colormap or a literal color list.
# ---------------------------------------------------------------------------
# The default is a hand-picked pastel sequence (pink, peach, ...). You can
# override with a colormap name like "viridis" or a list of color specs.

plot_tree(
    clf,
    X=X_c,
    feature_names=list(data.feature_names),
    class_names=["malignant", "benign"],
    cmap=["#4C72B0", "#DD8452"],  # explicit pair of colors
    fontsize=10,
)
save(plt.gcf(), "06_classifier_custom_palette.png")


# ---------------------------------------------------------------------------
# 7. Multi-way partitioning — ``num_partitions > 2`` fans out K children.
# ---------------------------------------------------------------------------
# The default pastel sequence has 8 colors; partitions beyond that cycle. For
# a K-way tree, ``cmap`` should provide at least K distinguishable colors.

clf_3way = SGTClassifier(
    num_partitions=3,
    max_depth=2,
    inner_max_depth=2,
    inner_max_leaf_nodes=8,
    min_samples_leaf=5,
    random_state=42,
).fit(X_c, y_c)

plot_tree(
    clf_3way,
    X=X_c,
    feature_names=list(data.feature_names),
    class_names=["malignant", "benign"],
    fontsize=10,
)
save(plt.gcf(), "07_classifier_3way.png")


# ---------------------------------------------------------------------------
# 8. Categorical shape function — one-hot block via ``feature_dict``.
# ---------------------------------------------------------------------------
# Group several one-hot columns into a single logical categorical feature with
# ``feature_dict``. Internal nodes then show discrete category buckets instead
# of numeric thresholds. When ``inner_max_leaf_nodes`` is smaller than the
# number of categories, multiple levels can share a bucket; those merged bins
# are labeled as a list of category names (e.g. ``[cat_0, cat_1, cat_2]``).
#
# String keys and column names work naturally with a pandas ``DataFrame``.

rng = np.random.default_rng(7)
n_samples = 500
species_levels = ["setosa", "versicolor", "virginica", "extra_a", "extra_b", "extra_c"]
cats = rng.integers(0, len(species_levels), size=n_samples)
X_cat = pd.DataFrame(0.0, index=np.arange(n_samples), columns=species_levels)
for row, cat in enumerate(cats):
    X_cat.iloc[row, cat] = 1.0
y_cat = (cats % 2).astype(np.int64)

clf_cat = SGTClassifier(
    max_depth=3,
    inner_max_depth=2,
    inner_max_leaf_nodes=3,  # force some categories to share a bucket
    min_samples_leaf=5,
    random_state=0,
    tao_n_runs=0,
    max_features=None,
).fit(X_cat, y_cat, feature_dict={"species": list(species_levels)})

plot_tree(
    clf_cat,
    X=X_cat,
    class_names=["even", "odd"],
    label="all",
    fontsize=10,
)
save(plt.gcf(), "08_categorical_feature_dict.png")


print("\nAll example plots written to:", OUT_DIR)
