"""Tree visualization for fitted SGT estimators.

``plot_tree`` renders a fitted ``SGTClassifier`` / ``SGTRegressor`` with
matplotlib: every internal node is drawn as a small histogram of the chosen
routing feature with bins colored by the destination child partition, and
every leaf is drawn as a text box with the predicted class / value.
``export_graphviz`` / ``export_text`` are placeholders for a future iteration.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

__all__ = ["export_graphviz", "export_text", "plot_tree"]

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


def _is_categorical_node(node: dict) -> bool:
    if node.get("is_categorical") is not None:
        return bool(node["is_categorical"])
    return len(node.get("features") or []) > 1


def _bin_categories_for_node(node: dict) -> list[list[int]]:
    bc = node.get("bin_categories")
    if bc:
        return [list(cats) for cats in bc]
    if _is_categorical_node(node):
        return [[int(c)] for c in node.get("features") or []]
    return []


def _column_label(col: int, feat_names: list[str]) -> str:
    if 0 <= col < len(feat_names):
        return feat_names[col]
    return f"X[{col}]"


def _category_label_for_columns(cols: Sequence[int], feat_names: list[str]) -> str:
    names = [_column_label(int(c), feat_names) for c in cols]
    if len(names) == 1:
        return names[0]
    return "[" + ", ".join(names) + "]"


def _bin_labels_for_categorical_node(node: dict, feat_names: list[str]) -> list[str]:
    labels: list[str] = []
    for cats in _bin_categories_for_node(node):
        if not cats:
            labels.append("NaN")
        else:
            labels.append(_category_label_for_columns(cats, feat_names))
    return labels


def _logical_feature_label(
    estimator: Any,
    node: dict,
    feat_names: list[str],
    *,
    prefer_logical_name: bool = True,
) -> str:
    routing_cols = [int(c) for c in node.get("features") or []]
    processed = getattr(estimator, "processed_features_", None)
    if prefer_logical_name and processed is not None:
        for i, feat in enumerate(processed.features):
            if set(feat["indices"]) == set(routing_cols):
                if processed.logical_names and i < len(processed.logical_names):
                    return processed.logical_names[i]
                break
    if len(routing_cols) == 1:
        return _column_label(routing_cols[0], feat_names)
    return _category_label_for_columns(routing_cols, feat_names)


def _active_onehot_column(row: np.ndarray, feature_cols: Sequence[int]) -> int | None:
    active_cols = [
        int(c) for c in feature_cols if np.isfinite(row[c]) and row[c] >= 0.5
    ]
    if not active_cols:
        return None
    if len(active_cols) == 1:
        return active_cols[0]
    return max(active_cols, key=lambda c: row[c])


def _categorical_bin_index(
    active_col: int, bin_categories: Sequence[Sequence[int]]
) -> int:
    for b, cats in enumerate(bin_categories):
        if active_col in cats:
            return b
    return max(len(bin_categories) - 1, 0)


def _finite_routing_bins(
    thresholds: list[float], bin_to_partition: list[int]
) -> list[int]:
    """Drop the trailing NaN routing bin when present."""
    n_finite = len(thresholds) + 1
    if len(bin_to_partition) > n_finite:
        return bin_to_partition[:n_finite]
    return bin_to_partition


def _is_pair_node(node: dict) -> bool:
    return node.get("routing_kind") == "pair"


def _pair_axis_missing(row: np.ndarray, axis: dict) -> bool:
    columns = [int(c) for c in axis["columns"]]
    if axis["kind"] == "continuous":
        return not np.isfinite(row[columns[0]])
    return _active_onehot_column(row, columns) is None


def _route_pair_bin(node: dict, row: np.ndarray) -> int:
    """Replay one exported pair router exactly."""
    inner = {int(n["id"]): n for n in node["pair_inner_tree"]}
    axes = node["pair_axes"]
    current = 0
    while not inner[current]["is_leaf"]:
        split = inner[current]
        if _pair_axis_missing(row, axes[int(split["axis"])]):
            current = int(split["missing"])
        elif split["kind"] == "categorical":
            current = int(
                split["right"] if row[int(split["feature"])] >= 0.5 else split["left"]
            )
        else:
            current = int(
                split["left"]
                if row[int(split["feature"])] <= float(split["threshold"])
                else split["right"]
            )
    return int(inner[current]["bin"])


def _route_pair_partition(node: dict, row: np.ndarray) -> int:
    return int(node["bin_to_partition"][_route_pair_bin(node, row)])


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

    bin_to_partition = _finite_routing_bins(thresholds, bin_to_partition)
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


def _route_samples(tree: dict, X) -> dict[int, Any]:
    """Route ``X`` through the tree; return ``{node_id: column-indices}``.

    The returned array for each node lists the row indices of ``X`` that
    reach that node. Leaves' sample sets partition the root's sample set.

    Routing rule (matches the C++ trainer): at each internal node, look up
    the routing feature column. Non-finite values use
    ``nan_prediction_partition``; finite values use
    ``bin = np.searchsorted(thresholds, value, side='right')`` (clamped) and
    ``children[bin_to_partition[bin]]``.
    """

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
        feature = node.get("feature")
        if _is_pair_node(node):
            part_idx = np.fromiter(
                (_route_pair_partition(node, X_arr[int(row_i)]) for row_i in rows),
                dtype=np.int64,
                count=rows.size,
            )
            children = list(node["children"])
        elif _is_categorical_node(node):
            feature_cols = [int(c) for c in node["features"]]
            bin_categories = _bin_categories_for_node(node)
            b2p = np.asarray(list(node["bin_to_partition"]), dtype=np.int64)
            children = list(node["children"])
            nan_part = int(node.get("nan_prediction_partition", 0))
            part_idx = np.empty(rows.size, dtype=np.int64)
            for j, row_i in enumerate(rows):
                active = _active_onehot_column(X_arr[int(row_i)], feature_cols)
                if active is None:
                    part_idx[j] = nan_part
                else:
                    bin_idx = _categorical_bin_index(active, bin_categories)
                    part_idx[j] = b2p[bin_idx]
        else:
            thresholds = np.asarray(node["thresholds"], dtype=np.float64)
            b2p_list = _finite_routing_bins(
                list(node["thresholds"]), list(node["bin_to_partition"])
            )
            b2p = np.asarray(b2p_list, dtype=np.int64)
            children = list(node["children"])

            values = X_arr[rows, feature]
            nan_part = int(node.get("nan_prediction_partition", 0))
            finite_mask = np.isfinite(values)
            part_idx = np.empty(rows.size, dtype=np.int64)
            if np.any(~finite_mask):
                part_idx[~finite_mask] = nan_part
            if np.any(finite_mask):
                bin_idx = np.searchsorted(thresholds, values[finite_mask], side="right")
                bin_idx = np.clip(bin_idx, 0, len(b2p) - 1)
                part_idx[finite_mask] = b2p[bin_idx]

        for k, cid in enumerate(children):
            mask = part_idx == k
            reach[cid] = rows[mask]
            queue.append(cid)
    for nid in nodes_by_id:
        reach.setdefault(nid, np.empty(0, dtype=np.int64))
    return reach


def _compute_layout_leafcounter(
    tree: dict, max_depth: int | None
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
        return bool(max_depth is not None and depth >= max_depth)

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
    class_names: list[str] | None,
    criterion: str,
    precision: int,
    fontsize: int | None,
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
        counts_by_output = list(node.get("class_counts", []))
        counts = list(counts_by_output[0]) if counts_by_output else []
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


def _draw_internal_panel_categorical(
    host_ax,
    center: tuple[float, float],
    size: tuple[float, float],
    node: dict,
    palette,
    feat_names: list[str],
    X_rows: np.ndarray | None,
    fontsize: int | None,
    label: str,
) -> list:
    cx, cy = center
    w, h = size
    inset = host_ax.inset_axes(
        [cx - w / 2, cy - h / 2, w, h], transform=host_ax.transAxes
    )

    b2p = list(node["bin_to_partition"])
    bin_labels = _bin_labels_for_categorical_node(node, feat_names)
    feature_cols = [int(c) for c in node.get("features") or []]
    bin_categories = _bin_categories_for_node(node)
    n_bins = len(b2p)
    if n_bins == 0:
        n_bins = len(bin_labels)
    if n_bins == 0:
        return [inset]

    counts = [0] * n_bins
    if X_rows is not None and X_rows.size and feature_cols and bin_categories:
        for row in X_rows:
            active = _active_onehot_column(row, feature_cols)
            if active is None:
                continue
            counts[_categorical_bin_index(active, bin_categories)] += 1
    max_count = max(counts) if counts and max(counts) > 0 else 1

    for b in range(n_bins):
        inset.axvspan(
            b,
            b + 1,
            ymin=0.0,
            ymax=counts[b] / max_count if counts else 1.0,
            color=palette[b2p[b]] if b < len(b2p) else palette[0],
            alpha=0.5,
            zorder=0,
        )
        if counts and counts[b] > 0:
            inset.bar(
                b + 0.5,
                counts[b] / max_count,
                width=0.85,
                color=palette[b2p[b]] if b < len(b2p) else palette[0],
                edgecolor="white",
                linewidth=0.3,
                zorder=1,
                align="center",
            )

    tick_labels = bin_labels[:n_bins]
    inset.set_xlim(0, n_bins)
    inset.set_ylim(0, 1.05)
    inset.set_xticks([b + 0.5 for b in range(n_bins)])
    tick_fs = (fontsize - 2) if isinstance(fontsize, int) else None
    inset.set_xticklabels(
        tick_labels,
        fontsize=tick_fs,
        rotation=30,
        ha="right",
    )
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


def _draw_internal_panel(
    host_ax,
    center: tuple[float, float],
    size: tuple[float, float],
    node: dict,
    palette,
    feature_values,
    feat_names: list[str],
    n_hist_bins: int,
    precision: int,
    fontsize: int | None,
    label: str,
) -> list:
    """Render a single internal node panel: slabs + optional fine histogram."""
    cx, cy = center
    w, h = size
    inset = host_ax.inset_axes(
        [cx - w / 2, cy - h / 2, w, h], transform=host_ax.transAxes
    )

    thresholds = list(node["thresholds"])
    b2p = _finite_routing_bins(thresholds, list(node["bin_to_partition"]))

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


def _pair_axis_label(
    estimator: Any,
    axis: dict,
    feat_names: list[str],
    *,
    prefer_logical_name: bool = True,
) -> str:
    processed = getattr(estimator, "processed_features_", None)
    logical = int(axis["logical_feature"])
    if (
        prefer_logical_name
        and processed is not None
        and processed.logical_names
        and logical < len(processed.logical_names)
    ):
        return str(processed.logical_names[logical])
    return _category_label_for_columns(axis["columns"], feat_names)


def _pair_axis_cells(
    node: dict,
    axis_index: int,
    axis: dict,
    values: np.ndarray | None,
    *,
    include_missing: bool,
) -> list[tuple[float, float, object]]:
    if axis["kind"] == "categorical":
        categories = [int(c) for c in axis["categories"]]
        categorical_cells: list[tuple[float, float, object]] = [
            (float(i), float(i + 1), category) for i, category in enumerate(categories)
        ]
        if include_missing:
            categorical_cells.append(
                (
                    float(len(categories)) + 0.08,
                    float(len(categories)) + 0.38,
                    None,
                )
            )
        return categorical_cells
    thresholds = sorted(
        {
            float(split["threshold"])
            for split in node["pair_inner_tree"]
            if not split["is_leaf"]
            and int(split["axis"]) == axis_index
            and split["kind"] == "continuous"
        }
    )
    finite = values[np.isfinite(values)] if values is not None else np.array([])
    if finite.size:
        lo, hi = float(finite.min()), float(finite.max())
    elif thresholds:
        delta = max((thresholds[-1] - thresholds[0]) / max(len(thresholds) - 1, 1), 1.0)
        lo, hi = thresholds[0] - delta, thresholds[-1] + delta
    else:
        lo, hi = 0.0, 1.0
    if hi <= lo:
        hi = lo + 1.0
    edges = [lo, *[t for t in thresholds if lo < t < hi], hi]
    continuous_cells: list[tuple[float, float, object]] = [
        (edges[i], edges[i + 1], (edges[i] + edges[i + 1]) / 2)
        for i in range(len(edges) - 1)
    ]
    if include_missing:
        span = hi - lo
        continuous_cells.append((hi + span * 0.04, hi + span * 0.12, None))
    return continuous_cells


def _pair_cell_row(
    node: dict, x_axis: dict, x_value, y_axis: dict, y_value
) -> np.ndarray:
    width = max([int(c) for c in node["features"]] + [0]) + 1
    row = np.zeros(width, dtype=np.float64)
    for axis, value in ((x_axis, x_value), (y_axis, y_value)):
        columns = [int(c) for c in axis["columns"]]
        if value is None:
            if axis["kind"] == "continuous":
                row[columns[0]] = np.nan
        elif axis["kind"] == "continuous":
            row[columns[0]] = float(value)
        else:
            row[int(value)] = 1.0
    return row


def _pair_switch_boundaries(
    cells: list[tuple[float, float, object]],
    partitions: np.ndarray,
    *,
    axis: int,
) -> list[float]:
    """Return finite cell boundaries where the outer partition changes."""
    boundaries = []
    for index in range(1, len(cells)):
        if cells[index - 1][2] is None or cells[index][2] is None:
            continue
        before = np.take(partitions, index - 1, axis=axis)
        after = np.take(partitions, index, axis=axis)
        if np.any(before != after):
            boundaries.append(cells[index][0])
    return boundaries


def _draw_internal_panel_pair(
    host_ax,
    center: tuple[float, float],
    size: tuple[float, float],
    node: dict,
    palette,
    estimator: Any,
    feat_names: list[str],
    X_rows: np.ndarray | None,
    fontsize: int | None,
    label: str,
    prefer_logical_name: bool,
    n_hist_bins: int,
    precision: int,
) -> list:
    """Draw exported pair-routing cells, including the two missing margins."""
    cx, cy = center
    w, h = size
    left, bottom = cx - w / 2, cy - h / 2
    with_histograms = X_rows is not None and len(X_rows) > 0
    heatmap_width = w * 0.78 if with_histograms else w
    heatmap_height = h * 0.78 if with_histograms else h
    inset = host_ax.inset_axes(
        [left, bottom, heatmap_width, heatmap_height],
        transform=host_ax.transAxes,
    )
    inset.set_label("pair-heatmap")
    x_axis, y_axis = node["pair_axes"]
    x_values = (
        X_rows[:, int(x_axis["columns"][0])]
        if X_rows is not None and x_axis["kind"] == "continuous"
        else None
    )
    y_values = (
        X_rows[:, int(y_axis["columns"][0])]
        if X_rows is not None and y_axis["kind"] == "continuous"
        else None
    )
    x_has_missing = X_rows is None or any(
        _pair_axis_missing(row, x_axis) for row in X_rows
    )
    y_has_missing = X_rows is None or any(
        _pair_axis_missing(row, y_axis) for row in X_rows
    )
    x_cells = _pair_axis_cells(node, 0, x_axis, x_values, include_missing=x_has_missing)
    y_cells = _pair_axis_cells(node, 1, y_axis, y_values, include_missing=y_has_missing)
    counts: dict[tuple[int, int], int] = {}
    if X_rows is not None:
        for row in X_rows:
            for xi, (x0, x1, xv) in enumerate(x_cells):
                if _pair_axis_missing(row, x_axis) != (xv is None):
                    continue
                if (
                    xv is not None
                    and x_axis["kind"] == "continuous"
                    and not (x0 <= row[int(x_axis["columns"][0])] <= x1)
                ):
                    continue
                if (
                    xv is not None
                    and x_axis["kind"] == "categorical"
                    and _active_onehot_column(row, x_axis["columns"]) != xv
                ):
                    continue
                for yi, (y0, y1, yv) in enumerate(y_cells):
                    if _pair_axis_missing(row, y_axis) != (yv is None):
                        continue
                    if (
                        yv is not None
                        and y_axis["kind"] == "continuous"
                        and not (y0 <= row[int(y_axis["columns"][0])] <= y1)
                    ):
                        continue
                    if (
                        yv is not None
                        and y_axis["kind"] == "categorical"
                        and _active_onehot_column(row, y_axis["columns"]) != yv
                    ):
                        continue
                    counts[(xi, yi)] = counts.get((xi, yi), 0) + 1
                    break
                break
    partitions = np.empty((len(x_cells), len(y_cells)), dtype=np.intp)
    for xi, (x0, x1, xv) in enumerate(x_cells):
        for yi, (y0, y1, yv) in enumerate(y_cells):
            part = _route_pair_partition(
                node, _pair_cell_row(node, x_axis, xv, y_axis, yv)
            )
            partitions[xi, yi] = part
            inset.add_patch(
                Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    facecolor=palette[part],
                    alpha=0.55,
                    edgecolor="white",
                    linewidth=0.5,
                )
            )
    inset.set_xlim(x_cells[0][0], x_cells[-1][1])
    inset.set_ylim(y_cells[0][0], y_cells[-1][1])
    histogram_axes = []
    if with_histograms:
        x_hist = host_ax.inset_axes(
            [left, bottom + h * 0.82, heatmap_width, h * 0.18],
            transform=host_ax.transAxes,
        )
        y_hist = host_ax.inset_axes(
            [left + w * 0.82, bottom, w * 0.18, heatmap_height],
            transform=host_ax.transAxes,
        )
        x_hist.set_label("pair-x-histogram")
        y_hist.set_label("pair-y-histogram")
        x_counts = [
            sum(counts.get((xi, yi), 0) for yi in range(len(y_cells)))
            for xi in range(len(x_cells))
        ]
        y_counts = [
            sum(counts.get((xi, yi), 0) for xi in range(len(x_cells)))
            for yi in range(len(y_cells))
        ]
        if x_axis["kind"] == "continuous":
            assert x_values is not None
            finite_cells = [cell for cell in x_cells if cell[2] is not None]
            finite_values = x_values[np.isfinite(x_values)]
            x_hist.hist(
                finite_values,
                bins=n_hist_bins,
                range=(finite_cells[0][0], finite_cells[-1][1]),
                color="#777777",
                alpha=0.65,
            )
            if x_has_missing:
                start, end, _ = x_cells[-1]
                x_hist.bar(
                    (start + end) / 2,
                    x_counts[-1],
                    width=end - start,
                    color="#777777",
                    alpha=0.65,
                )
        else:
            x_hist.bar(
                [(start + end) / 2 for start, end, _ in x_cells],
                x_counts,
                width=[end - start for start, end, _ in x_cells],
                color="#777777",
                alpha=0.65,
            )
        if y_axis["kind"] == "continuous":
            assert y_values is not None
            finite_cells = [cell for cell in y_cells if cell[2] is not None]
            finite_values = y_values[np.isfinite(y_values)]
            y_hist.hist(
                finite_values,
                bins=n_hist_bins,
                range=(finite_cells[0][0], finite_cells[-1][1]),
                orientation="horizontal",
                color="#777777",
                alpha=0.65,
            )
            if y_has_missing:
                start, end, _ = y_cells[-1]
                y_hist.barh(
                    (start + end) / 2,
                    y_counts[-1],
                    height=end - start,
                    color="#777777",
                    alpha=0.65,
                )
        else:
            y_hist.barh(
                [(start + end) / 2 for start, end, _ in y_cells],
                y_counts,
                height=[end - start for start, end, _ in y_cells],
                color="#777777",
                alpha=0.65,
            )
        x_hist.set_xlim(x_cells[0][0], x_cells[-1][1])
        y_hist.set_ylim(y_cells[0][0], y_cells[-1][1])
        x_hist.set_axis_off()
        y_hist.set_axis_off()
        histogram_axes = [x_hist, y_hist]
    inset.set_xlabel(
        _pair_axis_label(
            estimator, x_axis, feat_names, prefer_logical_name=prefer_logical_name
        ),
        fontsize=(fontsize - 1) if isinstance(fontsize, int) else None,
    )
    inset.set_ylabel(
        _pair_axis_label(
            estimator, y_axis, feat_names, prefer_logical_name=prefer_logical_name
        ),
        fontsize=(fontsize - 1) if isinstance(fontsize, int) else None,
    )
    if x_axis["kind"] == "categorical":
        inset.set_xticks([(a + b) / 2 for a, b, _ in x_cells])
        inset.set_xticklabels(
            [
                _column_label(cast(int, v), feat_names) if v is not None else "NaN"
                for _, _, v in x_cells
            ],
            rotation=30,
            ha="right",
            fontsize=(fontsize - 2) if isinstance(fontsize, int) else None,
        )
    else:
        x_ticks = _pair_switch_boundaries(x_cells, partitions, axis=0)
        x_labels = [f"{value:.{precision}f}" for value in x_ticks]
        if x_cells[-1][2] is None:
            start, end, _ = x_cells[-1]
            x_ticks.append((start + end) / 2)
            x_labels.append("NaN")
        inset.set_xticks(x_ticks)
        inset.set_xticklabels(
            x_labels,
            rotation=30,
            ha="right",
            fontsize=(fontsize - 2) if isinstance(fontsize, int) else None,
        )
    if y_axis["kind"] == "categorical":
        inset.set_yticks([(a + b) / 2 for a, b, _ in y_cells])
        inset.set_yticklabels(
            [
                _column_label(cast(int, v), feat_names) if v is not None else "NaN"
                for _, _, v in y_cells
            ],
            fontsize=(fontsize - 2) if isinstance(fontsize, int) else None,
        )
    else:
        y_ticks = _pair_switch_boundaries(y_cells, partitions, axis=1)
        y_labels = [f"{value:.{precision}f}" for value in y_ticks]
        if y_cells[-1][2] is None:
            start, end, _ = y_cells[-1]
            y_ticks.append((start + end) / 2)
            y_labels.append("NaN")
        inset.set_yticks(y_ticks)
        inset.set_yticklabels(
            y_labels,
            fontsize=(fontsize - 2) if isinstance(fontsize, int) else None,
        )
    if label != "none":
        inset.text(
            1.0,
            1.0,
            f"n={node['n_samples']}",
            transform=inset.transAxes,
            ha="right",
            va="top",
            fontsize=(fontsize - 2) if isinstance(fontsize, int) else None,
            color="#444444",
        )
    return [inset, *histogram_axes]


def plot_tree(
    estimator: Any,
    *,
    X=None,
    max_depth: int | None = None,
    feature_names: list[str] | None = None,
    class_names: list[str] | bool | None = None,
    label: str = "feature",
    impurity: bool = False,
    proportion: bool = False,
    precision: int = 2,
    cmap: Any = _DEFAULT_PALETTE_COLORS,
    ax: plt.Axes | None = None,
    fontsize: int | None = None,
    node_aspect_ratio: float = 2.5,
    n_hist_bins: int = 20,
) -> list[Any]:
    """Render a fitted SGT estimator with matplotlib.

    Univariate nodes show their one-dimensional shape function. Bivariate
    nodes show the exact two-dimensional routing heatmap; when ``X`` is
    supplied, top/right marginal histograms and observed missing-value margins
    are included. Continuous axes label only thresholds where the final outer
    partition changes.

    Parameters
    ----------
    estimator : SGTClassifier or SGTRegressor
        Fitted estimator to render.
    X : array-like of shape (n_samples, n_features), optional
        Data used for node sample counts and bivariate marginal histograms.
    max_depth : int, optional
        Maximum outer-tree depth to display.
    feature_names : list of str, optional
        Display names for input columns.
    class_names : list of str, bool, or None, default=None
        Class labels shown on classifier leaves.
    label : str, default="feature"
        Controls field-name labels in node text.
    impurity : bool, default=False
        Whether to display node impurity.
    proportion : bool, default=False
        Whether to display sample proportions instead of counts.
    precision : int, default=2
        Decimal precision for displayed values and routing thresholds.
    cmap : colormap or color sequence, optional
        Colors for classes and routing partitions.
    ax : matplotlib.axes.Axes, optional
        Axes on which to draw.
    fontsize : int, optional
        Node text size.
    node_aspect_ratio : float, default=2.5
        Width-to-height ratio of node boxes.
    n_hist_bins : int, default=20
        Number of bins in each bivariate marginal histogram.

    Returns
    -------
    list
        Matplotlib artists created by the tree exporter.
    """
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
    resolved_class_names: list[str] | None
    if not is_classifier:
        resolved_class_names = None
    elif class_names is True:
        clf_estimator: SGTClassifier = estimator  # type: ignore[assignment]
        classes: Any = (
            clf_estimator.classes_ if clf_estimator.classes_ is not None else []
        )
        resolved_class_names = [str(c) for c in classes]
    elif class_names in (None, False):
        resolved_class_names = [str(c) for c in range(tree["num_classes"][0])]
    else:
        resolved_class_names = list(class_names)  # type: ignore[arg-type]

    n_features = estimator.n_features_in_ or 0
    stored_feature_names = estimator.feature_names_in_
    if feature_names is not None:
        feat_names = list(feature_names)
    elif stored_feature_names is not None:
        feat_names = [str(n) for n in stored_feature_names]
    else:
        feat_names = [f"X[{i}]" for i in range(n_features)]

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
        node_rows = None
        if X_arr is not None and reach.get(nid) is not None and len(reach[nid]):
            node_rows = X_arr[reach[nid]]
        feat_vals = None
        if node_rows is not None and feat_idx is not None:
            feat_vals = node_rows[:, feat_idx]

        if _is_pair_node(node):
            panel_artists = _draw_internal_panel_pair(
                host_ax=ax,
                center=pos,
                size=(internal_w, internal_h),
                node=node,
                palette=palette,
                estimator=estimator,
                feat_names=feat_names,
                X_rows=node_rows,
                fontsize=fontsize,
                label=label,
                prefer_logical_name=feature_names is None,
                n_hist_bins=n_hist_bins,
                precision=precision,
            )
        elif _is_categorical_node(node):
            panel_artists = _draw_internal_panel_categorical(
                host_ax=ax,
                center=pos,
                size=(internal_w, internal_h),
                node=node,
                palette=palette,
                feat_names=feat_names,
                X_rows=node_rows,
                fontsize=fontsize,
                label=label,
            )
        else:
            panel_artists = _draw_internal_panel(
                host_ax=ax,
                center=pos,
                size=(internal_w, internal_h),
                node=node,
                palette=palette,
                feature_values=feat_vals,
                feat_names=feat_names,
                n_hist_bins=n_hist_bins,
                precision=precision,
                fontsize=fontsize,
                label=label,
            )
        artists.extend(panel_artists)

        feat_name = (
            " × ".join(
                _pair_axis_label(
                    estimator,
                    axis,
                    feat_names,
                    prefer_logical_name=feature_names is None,
                )
                for axis in node["pair_axes"]
            )
            if _is_pair_node(node)
            else _logical_feature_label(
                estimator,
                node,
                feat_names,
                prefer_logical_name=feature_names is None,
            )
        )
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
