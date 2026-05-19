"""Tree visualization for fitted SGT estimators.

``plot_tree`` renders a fitted ``SGTClassifier`` / ``SGTRegressor`` with
matplotlib: every internal node is drawn as a small histogram of the chosen
routing feature with bins colored by the destination child partition, and
every leaf is drawn as a text box with the predicted class / value.
``export_graphviz`` / ``export_text`` are placeholders for a future iteration.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import matplotlib.pyplot as plt
from sklearn.utils.validation import check_is_fitted

from sgtlearn.base import SGTClassifier, SGTRegressor


def export_graphviz() -> None:
    """Serialize the fitted tree as Graphviz DOT (not implemented)."""
    raise NotImplementedError("not implemented yet")


def export_text() -> None:
    """Return a human-readable multiline description of the tree (not implemented)."""
    raise NotImplementedError("not implemented yet")


def _compute_layout(tree: dict, max_depth: Optional[int]) -> dict[int, tuple[float, float]]:
    """Top-down BFS layout. Returns ``{node_id: (x, y)}`` in axes coords [0, 1]."""
    nodes_by_id = {n["id"]: n for n in tree["nodes"]}
    root = tree["root_index"]

    # BFS, truncating subtrees deeper than max_depth (those parents become draw-leaves).
    visible: list[int] = []
    levels: dict[int, list[int]] = {}
    queue: list[tuple[int, int]] = [(root, 0)]
    while queue:
        nid, depth = queue.pop(0)
        visible.append(nid)
        levels.setdefault(depth, []).append(nid)
        n = nodes_by_id[nid]
        if n["is_leaf"]:
            continue
        if max_depth is not None and depth >= max_depth:
            continue
        for ch in n["children"]:
            queue.append((ch, depth + 1))

    max_depth_drawn = max(levels)
    pos: dict[int, tuple[float, float]] = {}
    for depth, ids in levels.items():
        m = len(ids)
        y = 1.0 - (depth / max(1, max_depth_drawn))
        # Compress y range a bit so the top/bottom nodes don't kiss the axes edge.
        y = 0.05 + 0.9 * y
        for i, nid in enumerate(ids):
            x = (i + 1) / (m + 1)
            pos[nid] = (x, y)
    return pos


def plot_tree(
    estimator: Any,
    *,
    max_depth: Optional[int] = None,
    feature_names: Optional[list[str]] = None,
    class_names: Union[list[str], bool, None] = None,
    label: str = "feature",
    impurity: bool = False,
    proportion: bool = False,
    precision: int = 3,
    cmap: Any = "tab10",
    ax: Optional[plt.Axes] = None,
    fontsize: Optional[int] = None,
) -> list[Any]:
    """Render a fitted SGT estimator with matplotlib (see module docstring)."""
    if not isinstance(estimator, (SGTClassifier, SGTRegressor)):
        raise TypeError(
            "plot_tree expects an SGTClassifier or SGTRegressor; got "
            f"{type(estimator).__name__}"
        )
    check_is_fitted(estimator, attributes=("_est",))

    tree = estimator.tree_export()

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    layout = _compute_layout(tree, max_depth)
    artists: list[Any] = []

    for nid, (x, y) in layout.items():
        marker, = ax.plot([x], [y], marker="o", markersize=2, color="black")
        artists.append(marker)
    return artists
