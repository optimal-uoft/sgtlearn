"""Tree visualization for fitted SGT estimators.

``plot_tree`` renders a fitted ``SGTClassifier`` / ``SGTRegressor`` with
matplotlib: every internal node is drawn as a small histogram of the chosen
routing feature with bins colored by the destination child partition, and
every leaf is drawn as a text box with the predicted class / value.
``export_graphviz`` / ``export_text`` are placeholders for a future iteration.
"""

from __future__ import annotations

from typing import Any, Optional, Union

__all__ = ["plot_tree", "export_graphviz", "export_text"]

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


def _merge_routing_regions(
    thresholds: list[float],
    bin_to_partition: list[int],
    x_min: float,
    x_max: float,
) -> list[tuple[float, float, int]]:
    """Collapse consecutive same-partition bins into contiguous slabs.

    Given the inner-tree thresholds and the bin-to-partition map, return a
    list of ``(x0, x1, partition_id)`` slabs spanning ``[x_min, x_max]``.
    Non-contiguous bins with the same partition stay as separate slabs.

    The leftmost slab starts at ``x_min`` (clamped from -inf), the rightmost
    ends at ``x_max`` (clamped from +inf); interior boundaries are the
    thresholds where the destination partition changes between adjacent
    bins. Thresholds inside a run of same-partition bins are dropped.
    """
    if not bin_to_partition:
        return []

    n_bins = len(bin_to_partition)
    edges = [float("-inf")] + list(thresholds) + [float("inf")]
    assert len(edges) == n_bins + 1

    regions: list[tuple[float, float, int]] = []
    i = 0
    while i < n_bins:
        partition = bin_to_partition[i]
        j = i + 1
        while j < n_bins and bin_to_partition[j] == partition:
            j += 1
        x0 = max(x_min, edges[i])
        x1 = min(x_max, edges[j])
        if x1 > x0:
            regions.append((x0, x1, partition))
        i = j
    return regions


def _route_samples(tree: dict, X) -> "dict[int, Any]":
    """Route ``X`` through the tree; return ``{node_id: column-indices}``.

    The returned array for each node lists the row indices of ``X`` that
    reach that node. Leaves' sample sets partition the root's sample set.

    Routing rule (matches the C++ trainer): at each internal node, look up
    the routing feature column; ``bin = np.searchsorted(thresholds, value,
    side='right')`` gives the inner-tree bin (clamped to
    ``len(bin_to_partition)-1``); the destination child is
    ``children[bin_to_partition[bin]]``.
    """
    import numpy as np

    X_arr = np.asarray(X)
    nodes_by_id = {n["id"]: n for n in tree["nodes"]}
    root = tree["root_index"]
    n = X_arr.shape[0]

    reach: dict[int, np.ndarray] = {root: np.arange(n, dtype=np.int64)}
    queue = [root]
    while queue:
        nid = queue.pop(0)
        node = nodes_by_id[nid]
        if node["is_leaf"]:
            continue
        rows = reach[nid]
        if rows.size == 0:
            for cid in node["children"]:
                reach.setdefault(cid, np.empty(0, dtype=np.int64))
                queue.append(cid)
            continue
        feature = node["feature"]
        thresholds = np.asarray(node["thresholds"], dtype=np.float64)
        b2p = np.asarray(node["bin_to_partition"], dtype=np.int64)
        children = list(node["children"])

        values = X_arr[rows, feature]
        bin_idx = np.searchsorted(thresholds, values, side="right")
        bin_idx = np.clip(bin_idx, 0, len(b2p) - 1)
        part_idx = b2p[bin_idx]

        for k, cid in enumerate(children):
            mask = part_idx == k
            reach[cid] = rows[mask]
            queue.append(cid)
    for nid in nodes_by_id:
        reach.setdefault(nid, np.empty(0, dtype=np.int64))
    return reach


def _compute_layout_leafcounter(
    tree: dict, max_depth: Optional[int]
) -> dict[int, tuple[float, float]]:
    """Leaf-counter layout in axes coords [0, 1].

    Each visible leaf (or depth-cap-truncated internal node, which renders as
    a draw-leaf) claims the next integer x in DFS left-to-right order; each
    visible internal node's x = mean of its drawn children's x. y is set
    from depth and rescaled into the band ``[0.03, 0.88]`` (top 12% reserved
    for annotations, bottom 3% margin).
    """
    nodes_by_id = {n["id"]: n for n in tree["nodes"]}
    root = tree["root_index"]

    def is_draw_leaf(nid: int, depth: int) -> bool:
        n = nodes_by_id[nid]
        if n["is_leaf"]:
            return True
        if max_depth is not None and depth >= max_depth:
            return True
        return False

    x_int: dict[int, float] = {}
    counter = [0]

    def visit(nid: int, depth: int) -> float:
        if is_draw_leaf(nid, depth):
            xv = float(counter[0])
            counter[0] += 1
            x_int[nid] = xv
            return xv
        child_xs = [
            visit(cid, depth + 1) for cid in nodes_by_id[nid]["children"]
        ]
        xv = sum(child_xs) / len(child_xs) if child_xs else float(counter[0])
        x_int[nid] = xv
        return xv

    visit(root, 0)

    drawn_depths: dict[int, int] = {}

    def assign_depth(nid: int, depth: int) -> None:
        drawn_depths[nid] = depth
        if is_draw_leaf(nid, depth):
            return
        for cid in nodes_by_id[nid]["children"]:
            assign_depth(cid, depth + 1)

    assign_depth(root, 0)
    max_drawn_depth = max(drawn_depths.values()) if drawn_depths else 0

    n_leaves = counter[0]
    if n_leaves <= 1:
        norm_x = {nid: 0.5 for nid in drawn_depths}
    else:
        norm_x = {nid: x_int[nid] / (n_leaves - 1) for nid in drawn_depths}

    if max_drawn_depth == 0:
        norm_y = {nid: 0.5 for nid in drawn_depths}
    else:
        top, bot = 0.88, 0.03
        norm_y = {
            nid: bot
            + (top - bot) * (1.0 - drawn_depths[nid] / max_drawn_depth)
            for nid in drawn_depths
        }

    return {nid: (norm_x[nid], norm_y[nid]) for nid in drawn_depths}


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
    label: str,
    impurity: bool,
    criterion: str,
    precision: int,
):
    """Draw an internal node as a histogram inset. Returns the inset Axes."""
    x, y = pos
    w, h = inset_size
    inset = host_ax.inset_axes(
        [x - w / 2, y - h / 2, w, h], transform=host_ax.transAxes
    )
    edges = _bin_edges(list(node["thresholds"]))
    counts = list(node["bin_sample_counts"])
    if proportion:
        total = sum(counts) or 1
        counts = [c / total for c in counts]
    for i, count in enumerate(counts):
        left, right = edges[i], edges[i + 1]
        color = palette[node["bin_to_partition"][i]]
        inset.bar(
            (left + right) / 2.0,
            count,
            width=(right - left),
            color=color,
            align="center",
            edgecolor="none",
        )
    inset.set_yticks([])
    inset.set_xticks([])
    for spine in inset.spines.values():
        spine.set_visible(False)

    if label == "none":
        return inset
    parts = [feature_name]
    if label == "all":
        parts.append(f"n = {node['n_samples']}")
        if impurity:
            parts.append(f"{criterion} = {node['impurity']:.{precision}f}")
    inset.set_xlabel("\n".join(parts), fontsize=fontsize)
    return inset


def _level_widths(layout: "dict[int, tuple[float, float]]") -> "dict[float, list[int]]":
    by_y: dict[float, list[int]] = {}
    for nid, (_x, y) in layout.items():
        by_y.setdefault(round(y, 6), []).append(nid)
    return by_y


def _draw_leaf(
    host_ax,
    pos: tuple[float, float],
    box_size: tuple[float, float],
    node: dict,
    *,
    is_classifier: bool,
    class_names: Optional[list[str]],
    criterion: str,
    precision: int,
    fontsize: Optional[int],
):
    """Draw a leaf as a small filled rectangle with sklearn-style text.

    Returns ``(rect, text)`` for the caller to track as artists.
    """
    from matplotlib import patches

    x, y = pos
    w, h = box_size
    rect = patches.Rectangle(
        (x - w / 2, y - h / 2),
        w,
        h,
        fill=True,
        facecolor="#f0f0f0",
        edgecolor="black",
        linewidth=0.5,
        transform=host_ax.transAxes,
    )
    host_ax.add_patch(rect)

    lines = [f"samples = {node['n_samples']}"]
    if is_classifier:
        counts = list(node["class_counts"])
        lines.append(f"value = {counts}")
        if counts:
            arg = max(range(len(counts)), key=lambda i: counts[i])
            label = class_names[arg] if class_names is not None else str(arg)
            lines.append(f"class = {label}")
    else:
        lines.append(f"value = {node['value']:.{precision}f}")
    lines.append(f"{criterion} = {node['impurity']:.{precision}f}")

    text = host_ax.text(
        x,
        y,
        "\n".join(lines),
        ha="center",
        va="center",
        fontsize=fontsize,
        transform=host_ax.transAxes,
    )
    return rect, text


def _compute_layout(
    tree: dict, max_depth: Optional[int]
) -> dict[int, tuple[float, float]]:
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

    is_classifier = isinstance(estimator, SGTClassifier)
    resolved_class_names: Optional[list[str]]
    if not is_classifier:
        resolved_class_names = None
    elif class_names is True:
        clf_estimator: SGTClassifier = estimator  # type: ignore[assignment]
        classes: Any = (
            clf_estimator.classes_ if clf_estimator.classes_ is not None else []
        )
        resolved_class_names = [str(c) for c in classes]
    elif class_names in (None, False):
        resolved_class_names = [str(c) for c in range(tree["num_classes"])]
    else:
        resolved_class_names = list(class_names)  # type: ignore[arg-type]

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

    # Edge pass: draw edges BEFORE nodes so they appear under the inset axes.
    drawn_ids = set(layout)
    for nid, pos in layout.items():
        node = nodes_by_id[nid]
        if node["is_leaf"]:
            continue
        if max_depth is not None and node["depth"] >= max_depth:
            continue  # children not drawn
        for k, child_id in enumerate(node["children"]):
            if child_id not in drawn_ids:
                continue
            cx, cy = layout[child_id]
            x, y = pos
            line = ax.plot(
                [x, cx],
                [y - inset_size[1] / 2, cy + inset_size[1] / 2],
                color=palette[k],
                linewidth=1.5,
                solid_capstyle="round",
                transform=ax.transAxes,
            )[0]
            artists.append(line)

    for nid, pos in layout.items():
        node = nodes_by_id[nid]
        # A node renders as a "draw-leaf" when it's a real leaf OR when max_depth
        # truncates its subtree (its children weren't included in the layout).
        drawn_as_leaf = node["is_leaf"] or (
            max_depth is not None and node["depth"] >= max_depth and not node["is_leaf"]
        )
        if drawn_as_leaf:
            box_size = (inset_size[0], inset_size[1])
            rect, text = _draw_leaf(
                ax,
                pos,
                box_size,
                node,
                is_classifier=is_classifier,
                class_names=resolved_class_names,
                criterion=tree["criterion"],
                precision=precision,
                fontsize=fontsize,
            )
            artists.append(rect)
            artists.append(text)
        else:
            feat = feat_names[node["feature"]] if node["feature"] is not None else ""
            inset = _draw_internal(
                ax,
                pos,
                inset_size,
                node,
                palette,
                feat,
                proportion,
                fontsize,
                label=label,
                impurity=impurity,
                criterion=tree["criterion"],
                precision=precision,
            )
            artists.append(inset)
    return artists
