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


#: Hand-picked pastel sequence used as the default ``cmap`` for ``plot_tree``.
#: The first two colors (pink, peach) match the reference visualization
#: aesthetic in ``ex_viz/*.png``; the remaining colors come from matplotlib's
#: Pastel1 palette in a deterministic order.
_DEFAULT_PALETTE_COLORS = (
    "#E8A0BF",  # pink
    "#FAC898",  # peach
    "#B3CDE3",  # light blue
    "#CCEBC5",  # light green
    "#DECBE4",  # light purple
    "#FED9A6",  # cream
    "#FFFFCC",  # light yellow
    "#E5D8BD",  # tan
)


def _build_palette(cmap: Any, num_partitions: int):
    """Sample `num_partitions` colors from a matplotlib colormap or a list.

    When ``cmap`` is a list / tuple of color specs, the first
    ``num_partitions`` entries are returned (cycling if there aren't enough).
    When ``cmap`` is a matplotlib Colormap (or its registered name),
    ``num_partitions`` colors are sampled evenly along its domain.
    """
    import numpy as np

    if isinstance(cmap, (list, tuple)):
        if num_partitions <= len(cmap):
            return [cmap[i] for i in range(num_partitions)]
        return [cmap[i % len(cmap)] for i in range(num_partitions)]
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
        child_xs = [visit(cid, depth + 1) for cid in nodes_by_id[nid]["children"]]
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
        # Leave headroom at the bottom (10%) so leaf text isn't clipped and
        # arrows to leaves can land cleanly; small top margin (12%) makes
        # space for the right-of-panel feature labels.
        top, bot = 0.88, 0.10
        norm_y = {
            nid: bot + (top - bot) * (1.0 - drawn_depths[nid] / max_drawn_depth)
            for nid in drawn_depths
        }

    return {nid: (norm_x[nid], norm_y[nid]) for nid in drawn_depths}


def _draw_arrow_edge(
    fig,
    parent_xy: tuple[float, float],
    parent_h: float,
    child_xy: tuple[float, float],
    child_h: float,
    color,
    linewidth: float = 2.0,
    mutation_scale: float = 10.0,
    host_ax=None,
):
    """Draw an arrow from a parent panel's bottom-center to a child's top-center.

    Endpoints are in ``host_ax``'s axes-fraction coordinates (matching the
    layout system used by ``_compute_layout_leafcounter``). If ``host_ax`` is
    omitted, the figure's first axes is used. The patch is added to the host
    axes via ``add_patch`` and returned for tracking.
    """
    from matplotlib.patches import FancyArrowPatch

    if host_ax is None:
        host_ax = fig.axes[0]
    px, py = parent_xy
    cx, cy = child_xy
    patch = FancyArrowPatch(
        posA=(px, py - parent_h / 2),
        posB=(cx, cy + child_h / 2),
        arrowstyle="-|>",
        color=color,
        linewidth=linewidth,
        mutation_scale=mutation_scale,
        transform=host_ax.transAxes,
        zorder=1,
        clip_on=False,
    )
    host_ax.add_patch(patch)
    return patch


def _draw_leaf_text(
    host_ax,
    x: float,
    y: float,
    node: dict,
    *,
    is_classifier: bool,
    class_names: Optional[list[str]],
    criterion: str,
    precision: int,
    fontsize: Optional[int],
    color,
    label: str,
    impurity: bool,
) -> list:
    """Render a leaf as bold colored text (no box).

    Returns a list of Text artists (one bold class/value line, optional
    subtitle lines for ``label='all'``).
    """
    artists: list = []

    if is_classifier:
        counts = list(node.get("class_counts", []))
        if counts:
            arg = max(range(len(counts)), key=lambda i: counts[i])
            primary = class_names[arg] if class_names is not None else str(arg)
        else:
            primary = ""
    else:
        primary = f"{node['value']:.{precision}f}"

    bold = host_ax.text(
        x,
        y,
        primary,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=color,
        transform=host_ax.transAxes,
    )
    artists.append(bold)

    if label == "none":
        return artists

    if label == "all":
        subtitle_parts = [f"n = {node['n_samples']}"]
        if impurity:
            subtitle_parts.append(f"{criterion} = {node['impurity']:.{precision}f}")
        sub_fontsize = (fontsize - 1) if isinstance(fontsize, int) else None
        sub = host_ax.text(
            x,
            y - 0.025,
            "\n".join(subtitle_parts),
            ha="center",
            va="top",
            fontsize=sub_fontsize,
            color="#444444",
            transform=host_ax.transAxes,
        )
        artists.append(sub)

    return artists


def _draw_internal_panel(
    host_ax,
    center: tuple[float, float],
    size: tuple[float, float],
    node: dict,
    palette,
    feature_values,
    n_hist_bins: int,
    precision: int,
    fontsize: Optional[int],
    label: str,
) -> list:
    """Render a single internal node panel: slabs + optional fine histogram.

    The panel is created as an ``inset_axes`` on ``host_ax`` at the given
    figure-relative ``center`` and ``size``. Background slabs use
    ``axvspan``; the optional fine histogram (when ``feature_values`` is
    given) is overlaid as bar Rectangles. Threshold ticks sit under the
    panel at the slab boundaries; the sample-count annotation sits in the
    panel's top-right when ``label != 'none'``. Returns
    ``[inset_axes, *extra_text_artists]``.
    """
    import numpy as np

    cx, cy = center
    w, h = size
    inset = host_ax.inset_axes(
        [cx - w / 2, cy - h / 2, w, h], transform=host_ax.transAxes
    )

    thresholds = list(node["thresholds"])
    b2p = list(node["bin_to_partition"])

    # Compute panel x extent.
    if feature_values is not None and len(feature_values):
        x_min = float(np.min(feature_values))
        x_max = float(np.max(feature_values))
    elif thresholds:
        if len(thresholds) == 1:
            delta = 1.0
        else:
            delta = (thresholds[-1] - thresholds[0]) / (len(thresholds) - 1)
            if delta <= 0.0:
                delta = 1.0
        x_min = thresholds[0] - delta
        x_max = thresholds[-1] + delta
    else:
        x_min, x_max = 0.0, 1.0
    if x_max <= x_min:
        x_max = x_min + 1.0

    slabs = _merge_routing_regions(thresholds, b2p, x_min, x_max)
    for x0, x1, p in slabs:
        inset.axvspan(x0, x1, color=palette[p], alpha=0.5, zorder=0)

    if feature_values is not None and len(feature_values):
        widths = [(x1 - x0) for (x0, x1, _p) in slabs] or [x_max - x_min]
        total = sum(widths) or 1.0
        alloc = [max(1, round(n_hist_bins * w / total)) for w in widths]
        while sum(alloc) > n_hist_bins and any(a > 1 for a in alloc):
            j = max(range(len(alloc)), key=lambda k: (widths[k], alloc[k]))
            if alloc[j] <= 1:
                break
            alloc[j] -= 1
        while sum(alloc) < n_hist_bins:
            j = max(range(len(alloc)), key=lambda k: widths[k])
            alloc[j] += 1

        bin_edges: list[float] = []
        for j, (x0, x1, _p) in enumerate(slabs):
            edges = np.linspace(x0, x1, alloc[j] + 1)
            if j == 0:
                bin_edges.extend(edges.tolist())
            else:
                bin_edges.extend(edges[1:].tolist())
        counts, _ = np.histogram(feature_values, bins=bin_edges)
        bar_colors: list = []
        for j, (_x0, _x1, p) in enumerate(slabs):
            bar_colors.extend([palette[p]] * alloc[j])
        centers = [(bin_edges[k] + bin_edges[k + 1]) / 2 for k in range(len(counts))]
        bar_widths = [bin_edges[k + 1] - bin_edges[k] for k in range(len(counts))]
        inset.bar(
            centers,
            counts,
            width=bar_widths,
            color=bar_colors,
            edgecolor="white",
            linewidth=0.3,
            zorder=1,
            align="center",
        )

    boundaries = [x1 for (_x0, x1, _p) in slabs[:-1]]
    inset.set_xlim(x_min, x_max)
    if boundaries:
        inset.set_xticks(boundaries)
        tick_fs = (fontsize - 2) if isinstance(fontsize, int) else None
        inset.set_xticklabels(
            [f"{t:.{precision}f}" for t in boundaries],
            fontsize=tick_fs,
            rotation=30,
            ha="right",
        )
    else:
        inset.set_xticks([])
    inset.set_yticks([])
    for spine in ("top", "right", "left"):
        inset.spines[spine].set_visible(False)
    inset.spines["bottom"].set_linewidth(0.5)

    extra: list = []
    if label != "none":
        annot_fs = (fontsize - 2) if isinstance(fontsize, int) else None
        annot = inset.text(
            1.0,
            1.0,
            f"n={node['n_samples']}",
            transform=inset.transAxes,
            ha="right",
            va="top",
            fontsize=annot_fs,
            color="#444444",
        )
        extra.append(annot)

    return [inset, *extra]


def plot_tree(
    estimator: Any,
    *,
    X=None,
    max_depth: Optional[int] = None,
    feature_names: Optional[list[str]] = None,
    class_names: Union[list[str], bool, None] = None,
    label: str = "feature",
    impurity: bool = False,
    proportion: bool = False,
    precision: int = 2,
    cmap: Any = _DEFAULT_PALETTE_COLORS,
    ax: Optional[plt.Axes] = None,
    fontsize: Optional[int] = None,
    node_aspect_ratio: float = 2.5,
    n_hist_bins: int = 20,
) -> list[Any]:
    """Render a fitted SGT estimator with matplotlib (see module docstring)."""
    import numpy as np

    if not isinstance(estimator, (SGTClassifier, SGTRegressor)):
        raise TypeError(
            "plot_tree expects an SGTClassifier or SGTRegressor; got "
            f"{type(estimator).__name__}"
        )
    check_is_fitted(estimator, attributes=("_est",))

    tree = estimator.tree_export()

    reach: dict[int, Any] = {}
    if X is not None:
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(
                f"X must be 2-D (n_samples, n_features); got shape {X_arr.shape}"
            )
        n_features = estimator.n_features_in_ or 0
        if X_arr.shape[1] != n_features:
            raise ValueError(
                f"X has {X_arr.shape[1]} features, but estimator was fit on "
                f"{n_features}"
            )
        reach = _route_samples(tree, X_arr)
    else:
        X_arr = None

    if ax is None:
        n_leaves_est = sum(1 for n in tree["nodes"] if n["is_leaf"]) or 1
        max_depth_seen = max(n["depth"] for n in tree["nodes"])
        fig_w = max(n_leaves_est * 1.5, 8.0)
        fig_h = max((max_depth_seen + 1) * 1.6, 4.0)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    else:
        fig = ax.figure  # type: ignore[assignment]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    layout = _compute_layout_leafcounter(tree, max_depth)
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

    n_features = estimator.n_features_in_ or 0
    feat_names = feature_names or [f"X[{i}]" for i in range(n_features)]

    nodes_by_id = {n["id"]: n for n in tree["nodes"]}
    drawn_ids = set(layout)
    n_drawn_leaves = sum(
        1
        for nid in drawn_ids
        if nodes_by_id[nid]["is_leaf"]
        or (max_depth is not None and nodes_by_id[nid]["depth"] >= max_depth)
    )
    leaf_w = min(0.10, 0.85 / max(n_drawn_leaves, 1))
    leaf_h = leaf_w / node_aspect_ratio
    internal_w = leaf_w * 1.5
    internal_h = internal_w / node_aspect_ratio

    def _size_for(nid: int) -> tuple[float, float]:
        n = nodes_by_id[nid]
        drawn_as_leaf = n["is_leaf"] or (
            max_depth is not None and n["depth"] >= max_depth
        )
        return (leaf_w, leaf_h) if drawn_as_leaf else (internal_w, internal_h)

    artists: list[Any] = []

    # Map each drawn child back to its partition index (used to color the
    # leaf text the same as the incoming edge).
    child_partition_of: dict[int, int] = {}
    for nid in drawn_ids:
        node = nodes_by_id[nid]
        if node["is_leaf"]:
            continue
        for k, cid in enumerate(node["children"]):
            if cid in drawn_ids:
                child_partition_of[cid] = k

    # Edges first, so panels render on top.
    for nid in drawn_ids:
        node = nodes_by_id[nid]
        if node["is_leaf"]:
            continue
        if max_depth is not None and node["depth"] >= max_depth:
            continue
        parent_pos = layout[nid]
        _pw, ph = _size_for(nid)
        for k, child_id in enumerate(node["children"]):
            if child_id not in drawn_ids:
                continue
            child_pos = layout[child_id]
            _cw, ch = _size_for(child_id)
            patch = _draw_arrow_edge(
                fig=fig,
                parent_xy=parent_pos,
                parent_h=ph,
                child_xy=child_pos,
                child_h=ch,
                color=palette[k],
                host_ax=ax,
            )
            artists.append(patch)

    # Nodes.
    for nid in drawn_ids:
        node = nodes_by_id[nid]
        pos = layout[nid]
        drawn_as_leaf = node["is_leaf"] or (
            max_depth is not None and node["depth"] >= max_depth
        )

        if drawn_as_leaf:
            partition = child_partition_of.get(nid)
            leaf_color = palette[partition] if partition is not None else "#444444"
            text_artists = _draw_leaf_text(
                host_ax=ax,
                x=pos[0],
                y=pos[1],
                node=node,
                is_classifier=is_classifier,
                class_names=resolved_class_names,
                criterion=tree["criterion"],
                precision=precision,
                fontsize=fontsize,
                color=leaf_color,
                label=label,
                impurity=impurity,
            )
            artists.extend(text_artists)
            continue

        # Internal panel.
        feat_idx = node["feature"]
        feat_vals = None
        if X_arr is not None and reach.get(nid) is not None and len(reach[nid]):
            feat_vals = X_arr[reach[nid], feat_idx]

        panel_artists = _draw_internal_panel(
            host_ax=ax,
            center=pos,
            size=(internal_w, internal_h),
            node=node,
            palette=palette,
            feature_values=feat_vals,
            n_hist_bins=n_hist_bins,
            precision=precision,
            fontsize=fontsize,
            label=label,
        )
        artists.extend(panel_artists)

        # Feature name floats to the right of the panel (axes coords).
        feat_name = feat_names[feat_idx] if feat_idx is not None else ""
        feat_text = ax.text(
            pos[0] + internal_w * 0.55,
            pos[1],
            feat_name,
            ha="left",
            va="center",
            fontsize=fontsize,
            transform=ax.transAxes,
        )
        artists.append(feat_text)

    return artists
