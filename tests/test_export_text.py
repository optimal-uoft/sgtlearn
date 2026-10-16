"""Contracts for ``export_text``, ``export_graphviz`` and the shared child-region rules."""

from __future__ import annotations

import re
from xml.etree import ElementTree

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeClassifier

import sgtlearn
from sgtlearn import SGTClassifier, SGTRegressor, export_graphviz, export_text
from sgtlearn._export import (
    _DEFAULT_PALETTE_COLORS,
    _active_onehot_column,
    _child_regions,
    _format_region,
    _is_categorical_node,
    _is_pair_node,
)


def _holds(cond: tuple, row: np.ndarray) -> bool:
    op = cond[0]
    if op == "range":
        return bool(cond[2] < row[cond[1]] <= cond[3])
    if op == "nan":
        return not np.isfinite(row[cond[1]])
    if op == "none":
        return _active_onehot_column(row, cond[1]) is None
    if op == "in":
        return _active_onehot_column(row, cond[1]) in cond[2]
    raise AssertionError(f"unknown condition {cond!r}")


def _threshold_rows(tree: dict, X: np.ndarray) -> np.ndarray:
    """Rows of X moved to the float32 nearest each threshold and one step either side.

    That is an exact tie wherever the threshold is float32-representable.
    """
    cuts = set()
    for node in tree["nodes"]:
        if node["is_leaf"] or _is_categorical_node(node):
            continue
        if _is_pair_node(node):
            cuts |= {
                (int(s["feature"]), float(s["threshold"]))
                for s in node["pair_inner_tree"]
                if not s["is_leaf"] and s["kind"] == "continuous"
            }
        else:
            cuts |= {(int(node["feature"]), float(t)) for t in node["thresholds"]}
    rows = []
    for col, t in sorted(cuts):
        t32 = np.float32(t)
        for v in (t32, np.nextafter(t32, -np.inf), np.nextafter(t32, np.inf)):
            probe = X[:5].copy()
            probe[:, col] = v
            rows.append(probe)
    return np.vstack(rows)


def _leaf_by_rules(tree: dict, row: np.ndarray) -> dict:
    """Follow ``_child_regions`` from the root; every node must match one child."""
    nodes = {n["id"]: n for n in tree["nodes"]}
    node = nodes[tree["root_index"]]
    while not node["is_leaf"]:
        matches = [
            k
            for k, region in enumerate(_child_regions(node))
            if any(all(_holds(c, row) for c in box) for box in region)
        ]
        assert len(matches) == 1, (node["id"], row, matches)
        node = nodes[node["children"][matches[0]]]
    return node


def _univariate_regressor():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    y = np.sin(2 * X[:, 0]) + (X[:, 1] > 0.3) + rng.normal(0, 0.1, 400)
    X[rng.random(X.shape) < 0.08] = np.nan
    model = SGTRegressor(
        num_partitions=3, max_depth=3, min_samples_leaf=5, random_state=0
    ).fit(X, y)
    return model, X


def _categorical_classifier():
    rng = np.random.default_rng(1)
    n = 500
    cat = rng.integers(0, 4, size=n)
    onehot = np.eye(4)[cat]
    onehot[rng.random(n) < 0.1] = 0.0  # missing category
    x = rng.normal(size=n)
    x[rng.random(n) < 0.1] = np.nan
    y = (np.isin(cat, [0, 2]) ^ (np.nan_to_num(x) > 0.5)).astype(int)
    X = np.column_stack([x, onehot])
    model = SGTClassifier(
        num_partitions=3, max_depth=3, tao_n_runs=0, random_state=0
    ).fit(X, y, feature_dict={0: [0], 1: [1, 2, 3, 4]})
    return model, X


def _pair_classifier():
    categories = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
    states = np.array([[v, *c] for v in [-1.0, 1.0, np.nan] for c in categories])
    X = np.repeat(states, [40, 31, 30, 29, 28, 27, 26, 25, 24], axis=0)
    y = np.repeat(np.arange(9), [40, 31, 30, 29, 28, 27, 26, 25, 24])
    model = SGTClassifier(
        num_partitions=9,
        max_depth=1,
        inner_max_depth=2,
        inner_max_leaf_nodes=5,
        pairwise_candidates=1,
        tao_n_runs=0,
        random_state=0,
    ).fit(X, y, feature_dict={0: [0], 1: [1, 2]})
    return model, X


def _pair_regressor():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(600, 2))
    y = np.where(X[:, 0] * X[:, 1] > 0, 1.0, -1.0) + rng.normal(0, 0.05, 600)
    X[rng.random(X.shape) < 0.1] = np.nan
    model = SGTRegressor(
        num_partitions=3,
        max_depth=2,
        pairwise_candidates=1,
        min_samples_leaf=5,
        tao_n_runs=0,
        random_state=0,
    ).fit(X, y)
    return model, X


@pytest.mark.parametrize(
    "make",
    [_univariate_regressor, _categorical_classifier, _pair_classifier, _pair_regressor],
)
def test_child_regions_route_like_predict(make):
    model, X = make()
    tree = model.tree_export()
    X = np.vstack([X, _threshold_rows(tree, X)])
    X = X.astype(np.float32).astype(np.float64)  # what the native router sees
    leaves = [_leaf_by_rules(tree, row) for row in X]
    if isinstance(model, SGTRegressor):
        np.testing.assert_allclose(
            [leaf["value"] for leaf in leaves], model.predict(X), rtol=1e-6
        )
    else:
        counts = np.array([leaf["class_counts"][0] for leaf in leaves])
        np.testing.assert_allclose(
            counts / counts.sum(axis=1, keepdims=True),
            model.predict_proba(X),
            atol=1e-6,
        )


def test_fixtures_cover_every_routing_kind():
    kinds = set()
    for make in (_univariate_regressor, _categorical_classifier, _pair_regressor):
        for node in make()[0].tree_export()["nodes"]:
            if not node["is_leaf"]:
                kinds.add(
                    "pair"
                    if _is_pair_node(node)
                    else "categorical" if _is_categorical_node(node) else "numeric"
                )
    assert kinds == {"numeric", "categorical", "pair"}


def test_format_region_renders_every_condition():
    names = ["x", "red", "blue"]

    def group(cols):
        return "color"

    region = [
        [("range", 0, 1.0, 2.0)],
        [("range", 0, -np.inf, -1.0)],
        [("range", 0, 3.0, np.inf), ("in", (1, 2), (2,))],
        [("range", 0, -np.inf, np.inf), ("none", (1, 2))],
        [("nan", 0)],
    ]
    assert _format_region(region, names, group, 1) == (
        "1.0 < x <= 2.0 or x <= -1.0 or (x > 3.0 and color in {blue})"
        " or (x is not missing and color is missing) or x is missing"
    )
    assert _format_region([], names, group, 1) == "(empty)"


def test_export_text_regressor_exact():
    X = np.arange(4.0)[:, None]
    model = SGTRegressor(max_leaf_nodes=2, tao_n_runs=0, random_state=0).fit(
        X, np.array([0.0, 0.0, 1.0, 1.0])
    )
    assert export_text(model, feature_names=["x"]) == (
        "|--- x <= 1.50 or x is missing\n"
        "|   |--- value: 0.00\n"
        "|--- x > 1.50\n"
        "|   |--- value: 1.00\n"
    )


def test_export_text_classifier_class_names_and_weights():
    X = np.arange(6.0)[:, None]
    y = np.array(["a", "a", "a", "b", "b", "b"])
    model = SGTClassifier(max_leaf_nodes=2, tao_n_runs=0, random_state=0).fit(X, y)
    assert export_text(model, decimals=1) == (
        "|--- X[0] <= 2.5 or X[0] is missing\n"
        "|   |--- class: a\n"
        "|--- X[0] > 2.5\n"
        "|   |--- class: b\n"
    )
    text = export_text(model, class_names=["neg", "pos"], show_weights=True)
    assert "|   |--- weights: [3.00, 0.00] class: neg\n" in text
    assert "|   |--- weights: [0.00, 3.00] class: pos\n" in text


def test_export_text_categorical_and_pair_labels():
    model, _ = _pair_classifier()
    text = export_text(model, feature_names=["x", "red", "blue"])
    assert "[red, blue] in {red}" in text
    assert "[red, blue] is missing" in text
    assert "x is missing" in text
    assert text.count("|   |--- class:") == 9


def test_export_text_single_leaf_multioutput():
    X = np.zeros((4, 1))
    model = SGTRegressor(tao_n_runs=0).fit(X, np.ones((4, 2)))
    assert export_text(model) == "|--- value: [1.00, 1.00]\n"


def test_export_text_validates_inputs():
    with pytest.raises(TypeError):
        export_text(DecisionTreeClassifier().fit([[0], [1]], [0, 1]))
    with pytest.raises(NotFittedError):
        export_text(SGTRegressor())
    model = SGTRegressor(tao_n_runs=0).fit(np.zeros((4, 2)), np.arange(4.0))
    with pytest.raises(ValueError, match="feature_names"):
        export_text(model, feature_names=["only_one"])


_DOT_LEAF = re.compile(
    r'^(\d+) \[shape=box, style="[a-z, ]+", label="((?:[^"\\]|\\.)*)"'
    r'(?:, fillcolor="(#[0-9a-f]{6})")?\] ;$'
)
_DOT_PANEL = re.compile(r"^(\d+) \[shape=plain, label=<(.*)>\] ;$")
_DOT_EDGE = re.compile(
    r'^(\d+) -> (\d+) \[color="(#[0-9A-Fa-f]{6})", tooltip="((?:[^"\\]|\\.)*)"\] ;$'
)
_PALETTE = [c.lower() for c in _DEFAULT_PALETTE_COLORS]


def _parse_dot(dot: str) -> tuple[dict, dict, list]:
    """Parse ``export_graphviz`` output; every body line must be a node or edge.

    Returns leaves ``{id: (label, fillcolor)}``, panels ``{id: XML element}``
    and edges ``[(parent, child, color, tooltip)]``.
    """
    lines = dot.splitlines()
    assert lines[0] == "digraph Tree {" and lines[-1] == "}"
    leaves, panels, edges = {}, {}, []
    for line in lines[3:-1]:
        if m := _DOT_LEAF.match(line):
            leaves[int(m[1])] = (m[2].replace("\\n", "\n"), m[3])
        elif m := _DOT_PANEL.match(line):
            panels[int(m[1])] = ElementTree.fromstring(m[2])  # well-formed HTML label
        else:
            m = _DOT_EDGE.match(line)
            assert m, line
            edges.append((int(m[1]), int(m[2]), m[3].lower(), m[4].replace('\\"', '"')))
    return leaves, panels, edges


def _panel_colors(panel: ElementTree.Element) -> set[str]:
    return {td.get("BGCOLOR").lower() for td in panel.iter("TD") if td.get("BGCOLOR")}


def _panel_text(panel: ElementTree.Element) -> list[str]:
    return [t.strip() for t in panel.itertext() if t.strip()]


@pytest.mark.parametrize(
    "make",
    [_univariate_regressor, _categorical_classifier, _pair_classifier, _pair_regressor],
)
def test_export_graphviz_matches_export_text(make):
    model, X = make()
    tree = model.tree_export()
    by_id = {n["id"]: n for n in tree["nodes"]}
    leaves, panels, edges = _parse_dot(export_graphviz(model, X=X, filled=True))
    assert set(leaves) == {n["id"] for n in tree["nodes"] if n["is_leaf"]}
    assert set(panels) == {n["id"] for n in tree["nodes"] if not n["is_leaf"]}
    assert all(color for _, color in leaves.values())
    for parent, child, color, _ in edges:
        k = by_id[parent]["children"].index(child)
        assert color == _PALETTE[k % len(_PALETTE)]
    assert sorted((a, b) for a, b, *_ in edges) == sorted(
        (n["id"], c) for n in tree["nodes"] for c in n["children"]
    )

    dfs: list[tuple[int, int]] = []

    def walk(nid: int) -> None:  # export_text's line order
        for cid in by_id[nid]["children"]:
            dfs.append((nid, cid))
            walk(cid)

    walk(tree["root_index"])
    tips = {(a, b): tip for a, b, _, tip in edges}
    child_lines = [
        line.split("|--- ", 1)[1]
        for line in export_text(model).splitlines()
        if "|--- class:" not in line and "|--- value:" not in line
    ]
    assert [tips[e] for e in dfs] == child_lines


@pytest.mark.parametrize(
    "make",
    [_univariate_regressor, _categorical_classifier, _pair_classifier, _pair_regressor],
)
def test_export_graphviz_histograms_only_with_X(make):
    """Full child colors mark histogram bars or occupied cells; tints the routing."""
    model, X = make()
    _, with_x, _ = _parse_dot(export_graphviz(model, X=X))
    _, without_x, _ = _parse_dot(export_graphviz(model))
    reached = {n["id"] for n in model.tree_export()["nodes"] if n["n_samples"]}
    for nid, panel in with_x.items():
        if nid in reached:
            assert _panel_colors(panel) & set(_PALETTE), nid
    for panel in without_x.values():
        colors = _panel_colors(panel)
        assert colors and not colors & set(_PALETTE)
        assert "#9e9e9e" not in colors  # no pair marginals


def test_export_graphviz_panels_and_out_file(tmp_path):
    X = np.array([[0.0], [1.0], [2.0], [3.0], [np.nan]])
    model = SGTRegressor(max_leaf_nodes=2, tao_n_runs=0, random_state=0).fit(
        X, np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    )
    dot = export_graphviz(model, X=X, feature_names=['x "q"'])
    leaves, panels, edges = _parse_dot(dot)
    assert _panel_text(panels[0]) == [
        'x "q"',
        "samples = 5",
        "0.00  |  1.50  |  3.00   NaN",
    ]
    assert sorted(label for label, _ in leaves.values()) == [
        "samples = 2\nvalue = 1.00",
        "samples = 3\nvalue = 0.00",
    ]
    assert [tip for *_, tip in edges] == [
        'x "q" <= 1.50 or x "q" is missing',
        'x "q" > 1.50',
    ]
    assert (
        _panel_text(_parse_dot(export_graphviz(model))[1][0])[-1]
        == "0.50  |  1.50  |  2.50"
    )
    path = tmp_path / "tree.dot"
    out = export_graphviz(model, out_file=path, X=X, feature_names=['x "q"'])
    assert out is None and path.read_text() == dot

    clf = SGTClassifier(max_leaf_nodes=2, tao_n_runs=0, random_state=0).fit(
        X[:4], ["a", "a", "b", "b"]
    )
    root = _parse_dot(export_graphviz(clf, class_names=["neg", "pos"]))[1][0]
    assert _panel_text(root)[:4] == [
        "X[0]",
        "samples = 4",
        "value = [2.00, 2.00]",
        "class = neg",
    ]


def test_export_graphviz_categorical_and_pair_panels():
    model, X = _pair_classifier()
    panels = _parse_dot(
        export_graphviz(model, X=X, feature_names=["x", "red", "blue"])
    )[1]
    text = _panel_text(panels[model.tree_export()["root_index"]])
    assert text[0] == "x & [red, blue]"
    assert text[-2:] == [
        "x-axis: x, -1.00 to 1.00",
        "y-axis: [red, blue] in {red, blue}",
    ]

    model, X = _categorical_classifier()
    root = _parse_dot(export_graphviz(model, X=X))[1][0]
    labels = _panel_text(root)[4:]
    assert labels.count("NaN") == 1  # the empty category bin is the NaN column
    assert len(labels) == 5


def test_export_graphviz_is_public_and_validates_inputs():
    assert "export_graphviz" in sgtlearn.__all__
    with pytest.raises(TypeError):
        export_graphviz(DecisionTreeClassifier().fit([[0], [1]], [0, 1]))
    model = SGTRegressor(tao_n_runs=0).fit(np.zeros((4, 2)), np.arange(4.0))
    with pytest.raises(ValueError, match="feature_names"):
        export_graphviz(model, feature_names=["only_one"])
    with pytest.raises(ValueError, match="features"):
        export_graphviz(model, X=np.zeros((3, 3)))
