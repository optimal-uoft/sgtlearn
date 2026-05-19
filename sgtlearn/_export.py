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


def _build_palette(cmap: Any, num_partitions: int):
    """Sample `num_partitions` colors from a matplotlib colormap or its name."""
    import numpy as np
    if isinstance(cmap, str):
        cm = plt.get_cmap(cmap)
    else:
        cm = cmap
    if num_partitions <= 1:
        return [cm(0.0)]
    points = np.linspace(0.0, 1.0, num_partitions)
    return [cm(p) for p in points]


def _bin_edges(thresholds: list[float]) -> list[float]:
    """Closed bin edges suitable for plotting. Open ends are clipped to ±delta."""
    if not thresholds:
        return [0.0, 1.0]
    if len(thresholds) == 1:
        delta = 1.0
    else:
        delta = (thresholds[-1] - thresholds[0]) / (len(thresholds) - 1)
        if delta <= 0.0:
            delta = 1.0
    return [thresholds[0] - delta] + list(thresholds) + [thresholds[-1] + delta]


def _draw_internal(
    host_ax,
    pos: tuple[float, float],
    inset_size: tuple[float, float],
    node: dict,
    palette,
    feature_name: str,
    proportion: bool,
    fontsize: Optional[int],
):
    """Draw an internal node as a histogram inset. Returns the inset Axes."""
    x, y = pos
    w, h = inset_size
    inset = host_ax.inset_axes([x - w / 2, y - h / 2, w, h], transform=host_ax.transAxes)
    edges = _bin_edges(list(node["thresholds"]))
    counts = list(node["bin_sample_counts"])
    if proportion:
        total = sum(counts) or 1
        counts = [c / total for c in counts]
    for i, count in enumerate(counts):
        left, right = edges[i], edges[i + 1]
        color = palette[node["bin_to_partition"][i]]
        inset.bar((left + right) / 2.0, count, width=(right - left),
                  color=color, align="center", edgecolor="none")
    inset.set_yticks([])
    inset.set_xticks([])
    inset.set_xlabel(feature_name, fontsize=fontsize)
    for spine in inset.spines.values():
        spine.set_visible(False)
    return inset


def _level_widths(layout: "dict[int, tuple[float, float]]") -> "dict[float, list[int]]":
    by_y: dict[float, list[int]] = {}
    for nid, (_x, y) in layout.items():
        by_y.setdefault(round(y, 6), []).append(nid)
    return by_y


def _placeholder_leaf_marker(host_ax, x: float, y: float):
    """Temporary leaf marker until Task 12 replaces it with a text box."""
    return host_ax.plot([x], [y], marker="s", markersize=4, color="lightgray")


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
    palette = _build_palette(cmap, tree["num_partitions"])

    # Resolve feature names (fall back to "X[i]" placeholders).
    n_features = estimator.n_features_in_ or 0
    feat_names = feature_names or [f"X[{i}]" for i in range(n_features)]

    # Inset size derived from level breadth and depth count.
    levels = _level_widths(layout)
    max_breadth = max(len(ids) for ids in levels.values())
    inset_w = (1.0 / (max_breadth + 1)) * 0.85
    n_levels = len(levels)
    inset_h = ((0.9 / max(1, n_levels))) * 0.7
    inset_size = (inset_w, inset_h)

    nodes_by_id = {n["id"]: n for n in tree["nodes"]}
    artists: list[Any] = []

    for nid, pos in layout.items():
        node = nodes_by_id[nid]
        # A node renders as a "draw-leaf" when it's a real leaf OR when max_depth
        # truncates its subtree (its children weren't included in the layout).
        drawn_as_leaf = node["is_leaf"] or (
            max_depth is not None and node["depth"] >= max_depth and not node["is_leaf"]
        )
        if drawn_as_leaf:
            marker = _placeholder_leaf_marker(ax, pos[0], pos[1])
            artists.extend(marker)
        else:
            feat = feat_names[node["feature"]] if node["feature"] is not None else ""
            inset = _draw_internal(ax, pos, inset_size, node, palette, feat, proportion, fontsize)
            artists.append(inset)
    return artists
