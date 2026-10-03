import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pytest
from matplotlib.text import Text

from sgtlearn._export import _draw_leaf_text


@pytest.fixture
def ax():
    fig, ax = plt.subplots()
    yield ax
    plt.close("all")


def _is_bold(artist):
    weight = artist.get_fontweight()
    if isinstance(weight, str):
        return weight == "bold"
    return weight >= 700


def _draw(ax, node, *, is_classifier=False, class_names=None, criterion="gini",
          precision=2, fontsize=10, color="black", label="none", impurity=False,
          x=0.5, y=0.5):
    return _draw_leaf_text(
        ax,
        x,
        y,
        node,
        is_classifier=is_classifier,
        class_names=class_names,
        criterion=criterion,
        precision=precision,
        fontsize=fontsize,
        color=color,
        label=label,
        impurity=impurity,
    )


# ---------------------------------------------------------------- classifier

def test_classifier_single_output_shows_majority_class_name(ax):
    node = {"class_counts": [[1.0, 5.0, 2.0]]}
    artists = _draw(ax, node, is_classifier=True, class_names=[["a", "b", "c"]])
    assert artists[0].get_text() == "b"


def test_classifier_tie_picks_lowest_class_index(ax):
    node = {"class_counts": [[1.0, 4.0, 4.0]]}
    artists = _draw(ax, node, is_classifier=True, class_names=[["a", "b", "c"]])
    assert artists[0].get_text() == "b"


def test_classifier_all_equal_counts_picks_first_class(ax):
    node = {"class_counts": [[3.0, 3.0]]}
    artists = _draw(ax, node, is_classifier=True, class_names=[["yes", "no"]])
    assert artists[0].get_text() == "yes"


def test_classifier_multi_output_shows_bracketed_comma_separated_names(ax):
    node = {"class_counts": [[1.0, 9.0], [7.0, 2.0, 0.0]]}
    class_names = [["neg", "pos"], ["x", "y", "z"]]
    artists = _draw(ax, node, is_classifier=True, class_names=class_names)
    assert artists[0].get_text() == "[pos, x]"


def test_classifier_multi_output_uses_per_output_class_names(ax):
    node = {"class_counts": [[0.0, 2.0], [0.0, 2.0]]}
    class_names = [["a0", "a1"], ["b0", "b1"]]
    artists = _draw(ax, node, is_classifier=True, class_names=class_names)
    assert artists[0].get_text() == "[a1, b1]"


def test_classifier_empty_class_counts_shows_empty_string(ax):
    node = {"class_counts": []}
    artists = _draw(ax, node, is_classifier=True, class_names=[["a", "b"]])
    assert artists[0].get_text() == ""


def test_classifier_missing_class_counts_shows_empty_string(ax):
    node = {}
    artists = _draw(ax, node, is_classifier=True, class_names=[["a", "b"]])
    assert artists[0].get_text() == ""


# ----------------------------------------------------------------- regressor

def test_regressor_none_value_shows_ellipsis(ax):
    artists = _draw(ax, {"value": None})
    assert artists[0].get_text() == "\u2026"


def test_regressor_scalar_value_formatted_to_precision(ax):
    artists = _draw(ax, {"value": 1.23456}, precision=2)
    assert artists[0].get_text() == "1.23"


def test_regressor_scalar_value_respects_different_precision(ax):
    artists = _draw(ax, {"value": 3.0}, precision=4)
    assert artists[0].get_text() == "3.0000"


def test_regressor_scalar_value_zero_precision(ax):
    artists = _draw(ax, {"value": 7.0}, precision=0)
    assert artists[0].get_text() == "7"


def test_regressor_multi_output_shows_bracketed_formatted_values(ax):
    artists = _draw(ax, {"value": [1.0, 2.5, -0.333]}, precision=2)
    assert artists[0].get_text() == "[1.00, 2.50, -0.33]"


def test_regressor_multi_output_tuple_sequence(ax):
    artists = _draw(ax, {"value": (0.1, 0.2)}, precision=1)
    assert artists[0].get_text() == "[0.1, 0.2]"


def test_regressor_ignores_class_names(ax):
    artists = _draw(ax, {"value": 2.0}, class_names=None, precision=1)
    assert artists[0].get_text() == "2.0"


# ------------------------------------------------------- primary line styling

def test_primary_line_is_bold(ax):
    artists = _draw(ax, {"value": 1.0})
    assert _is_bold(artists[0])


def test_primary_line_positioned_at_xy(ax):
    artists = _draw(ax, {"value": 1.0}, x=0.25, y=0.75)
    assert artists[0].get_position() == pytest.approx((0.25, 0.75))


def test_primary_line_uses_axes_transform(ax):
    artists = _draw(ax, {"value": 1.0})
    assert artists[0].get_transform() == ax.transAxes


def test_primary_line_centred_horizontally_and_vertically(ax):
    artists = _draw(ax, {"value": 1.0})
    assert artists[0].get_ha() == "center"
    assert artists[0].get_va() == "center"


def test_primary_line_uses_given_fontsize(ax):
    artists = _draw(ax, {"value": 1.0}, fontsize=14)
    assert artists[0].get_fontsize() == pytest.approx(14)


def test_primary_line_default_fontsize_when_none(ax):
    artists = _draw(ax, {"value": 1.0}, fontsize=None)
    expected = matplotlib.font_manager.FontProperties(
        size=matplotlib.rcParams["font.size"]
    ).get_size_in_points()
    assert artists[0].get_fontsize() == pytest.approx(expected)


def test_primary_line_uses_given_color(ax):
    artists = _draw(ax, {"value": 1.0}, color="red")
    assert mcolors.same_color(artists[0].get_color(), "red")


def test_primary_line_is_added_to_host_axes(ax):
    artists = _draw(ax, {"value": 1.0})
    assert artists[0].axes is ax


# --------------------------------------------------------------- return shape

def test_label_none_returns_single_text_artist(ax):
    artists = _draw(ax, {"value": 1.0}, label="none")
    assert isinstance(artists, list)
    assert len(artists) == 1
    assert isinstance(artists[0], Text)


def test_other_label_value_returns_single_text_artist(ax):
    artists = _draw(ax, {"value": 1.0}, label="feature")
    assert len(artists) == 1


def test_label_all_returns_two_text_artists(ax):
    node = {"value": 1.0, "n_samples": 10, "impurity": 0.5}
    artists = _draw(ax, node, label="all")
    assert len(artists) == 2
    assert all(isinstance(a, Text) for a in artists)


def test_label_all_primary_artist_first(ax):
    node = {"value": 1.0, "n_samples": 10, "impurity": 0.5}
    artists = _draw(ax, node, label="all", precision=2)
    assert artists[0].get_text() == "1.00"
    assert _is_bold(artists[0])


def test_label_none_does_not_require_n_samples(ax):
    artists = _draw(ax, {"value": 1.0}, label="none", impurity=True)
    assert len(artists) == 1


# ------------------------------------------------------------------ subtitle

def test_subtitle_shows_n_samples_without_impurity(ax):
    node = {"value": 1.0, "n_samples": 42, "impurity": 0.5}
    artists = _draw(ax, node, label="all", impurity=False)
    assert artists[1].get_text() == "n = 42"


def test_subtitle_includes_impurity_line_when_requested(ax):
    node = {"value": 1.0, "n_samples": 42, "impurity": 0.123456}
    artists = _draw(ax, node, label="all", impurity=True,
                    criterion="squared_error", precision=3)
    lines = artists[1].get_text().split("\n")
    assert lines == ["n = 42", "squared_error = 0.123"]


def test_subtitle_for_classifier_node(ax):
    node = {"class_counts": [[2.0, 8.0]], "n_samples": 10, "impurity": 0.32}
    artists = _draw(ax, node, is_classifier=True, class_names=[["a", "b"]],
                    label="all", impurity=True, criterion="gini", precision=2)
    assert artists[0].get_text() == "b"
    assert artists[1].get_text().split("\n") == ["n = 10", "gini = 0.32"]


def test_subtitle_is_not_bold(ax):
    node = {"value": 1.0, "n_samples": 5, "impurity": 0.0}
    artists = _draw(ax, node, label="all")
    assert not _is_bold(artists[1])


def test_subtitle_placed_below_primary(ax):
    node = {"value": 1.0, "n_samples": 5, "impurity": 0.0}
    artists = _draw(ax, node, label="all", x=0.4, y=0.6)
    sx, sy = artists[1].get_position()
    assert sx == pytest.approx(0.4)
    assert sy < 0.6


def test_subtitle_anchored_top_and_centred(ax):
    node = {"value": 1.0, "n_samples": 5, "impurity": 0.0}
    artists = _draw(ax, node, label="all")
    assert artists[1].get_va() == "top"
    assert artists[1].get_ha() == "center"


def test_subtitle_uses_axes_transform(ax):
    node = {"value": 1.0, "n_samples": 5, "impurity": 0.0}
    artists = _draw(ax, node, label="all")
    assert artists[1].get_transform() == ax.transAxes


def test_subtitle_fontsize_one_smaller_than_integer_fontsize(ax):
    node = {"value": 1.0, "n_samples": 5, "impurity": 0.0}
    artists = _draw(ax, node, label="all", fontsize=12)
    assert artists[1].get_fontsize() == pytest.approx(11)


def test_subtitle_default_fontsize_when_fontsize_none(ax):
    node = {"value": 1.0, "n_samples": 5, "impurity": 0.0}
    artists = _draw(ax, node, label="all", fontsize=None)
    expected = matplotlib.font_manager.FontProperties(
        size=matplotlib.rcParams["font.size"]
    ).get_size_in_points()
    assert artists[1].get_fontsize() == pytest.approx(expected)


# -------------------------------------------------------------------- errors

def test_label_all_missing_n_samples_raises_key_error(ax):
    node = {"value": 1.0, "impurity": 0.5}
    with pytest.raises(KeyError):
        _draw(ax, node, label="all", impurity=False)


def test_label_all_with_impurity_missing_impurity_raises_key_error(ax):
    node = {"value": 1.0, "n_samples": 10}
    with pytest.raises(KeyError):
        _draw(ax, node, label="all", impurity=True)


def test_label_all_without_impurity_does_not_need_impurity_key(ax):
    node = {"value": 1.0, "n_samples": 10}
    artists = _draw(ax, node, label="all", impurity=False)
    assert artists[1].get_text() == "n = 10"
