"""Unit tests for the base visualisation module."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pytest
from matplotlib.figure import Figure

from xwhy.plots.visualisation.base import (
    DimensionError,
    Explanation,
    _as_explanation,
    _check_backend,
    _display_html,
    _finish_html,
    _finish_matplotlib,
    _finish_plotly,
    _format_value,
    _global_importance,
    _group_minor_features,
    _resolve_names,
    _single_instance,
    _style_axes,
    _wrap_html_document,
    convert_name,
    initjs,
)

# 1. Explanation Class Tests


def test_explanation_initialization() -> None:
    """Test explanation_initialization."""
    exp = Explanation(values=np.array([1, 2, 3]))
    assert isinstance(exp.values, np.ndarray)
    assert exp.shape == (3,)
    assert exp.ndim == 1
    assert len(exp) == 3


def test_explanation_repr() -> None:
    """Test explanation_repr."""
    exp = Explanation(values=np.ones((2, 3)), feature_names=["a", "b", "c"])
    assert repr(exp) == "Explanation(values=(2, 3), base_values=(), n_names=3)"

    exp2 = Explanation(values=np.array([1]))
    assert "n_names=0" in repr(exp2)


def test_explanation_len_empty() -> None:
    """Test explanation_len_empty."""
    exp = Explanation(values=np.array(1.0))
    assert len(exp) == 0


def test_explanation_getitem_simple() -> None:
    """Test explanation_getitem_simple."""
    exp = Explanation(
        values=np.array([[1, 2], [3, 4]]),
        base_values=np.array([0.1, 0.2]),
        data=np.array([[10, 20], [30, 40]]),
        feature_names=["f1", "f2"],
    )
    sliced = exp[0]
    assert np.array_equal(sliced.values, [1, 2])
    assert sliced.base_values == 0.1
    assert sliced.data is not None
    assert np.array_equal(sliced.data, [10, 20])
    assert sliced.feature_names == ["f1", "f2"]


def test_explanation_getitem_tuple() -> None:
    """Test explanation_getitem_tuple."""
    exp = Explanation(
        values=np.array([[1, 2, 3], [4, 5, 6]]),
        base_values=np.array([0.1, 0.2]),
        data=np.array([[10, 20, 30], [40, 50, 60]]),
        feature_names=["f1", "f2", "f3"],
    )
    sliced = exp[:, 1:]
    assert np.array_equal(sliced.values, [[2, 3], [5, 6]])
    assert np.array_equal(sliced.base_values, [0.1, 0.2])
    assert sliced.data is not None
    assert np.array_equal(sliced.data, [[20, 30], [50, 60]])
    assert sliced.feature_names is not None
    assert list(sliced.feature_names) == ["f2", "f3"]


def test_explanation_getitem_base_values_scalar() -> None:
    """Test explanation_getitem_base_values_scalar."""
    exp = Explanation(values=np.array([[1], [2]]), base_values=0.5)
    assert exp[0].base_values == 0.5


def test_explanation_getitem_base_values_invalid_index() -> None:
    """Test explanation_getitem_base_values_invalid_index."""
    exp = Explanation(values=np.array([[1], [2]]), base_values=np.array([0.1]))
    sliced = exp[1]
    assert np.array_equal(sliced.base_values, np.array([0.1]))


def test_explanation_getitem_data_invalid_index() -> None:
    """Test explanation_getitem_data_invalid_index."""
    exp = Explanation(values=np.array([[1], [2]]), data=np.array([[10]]))
    sliced = exp[1]
    assert sliced.data is None


def test_explanation_abs() -> None:
    """Test explanation_abs."""
    exp = Explanation(values=np.array([-1, 2, -3]))
    assert np.array_equal(exp.abs.values, [1, 2, 3])


def test_explanation_reductions() -> None:
    """Test explanation_reductions."""
    exp = Explanation(values=np.array([[1, 2], [3, 4]]), feature_names=["f1", "f2"])
    mean_exp = exp.mean(axis=0)
    assert np.array_equal(mean_exp.values, [2, 3])
    assert mean_exp.feature_names == ["f1", "f2"]

    sum_exp = exp.sum(axis=0)
    assert np.array_equal(sum_exp.values, [4, 6])

    max_exp = exp.max(axis=0)
    assert np.array_equal(max_exp.values, [3, 4])

    min_exp = exp.min(axis=0)
    assert np.array_equal(min_exp.values, [1, 2])

    mean1 = exp.mean(axis=1)
    assert mean1.feature_names is None


# 2. _as_explanation


def test_as_explanation_pass_through() -> None:
    """Test as_explanation_pass_through."""
    exp = Explanation(values=np.array([1]))
    assert _as_explanation(exp) is exp


def test_as_explanation_duck_type() -> None:
    """Test as_explanation_duck_type."""
    duck_type = type(
        "Explanation",
        (object,),
        {
            "values": [1, 2],
            "base_values": 0.5,
            "data": None,
            "feature_names": ["a", "b"],
        },
    )
    duck = duck_type()
    exp = _as_explanation(duck)
    assert isinstance(exp, Explanation)
    assert np.array_equal(exp.values, [1, 2])


def test_as_explanation_to_explanation() -> None:
    """Test as_explanation_to_explanation."""

    class HasToExplanation:
        def to_explanation(self) -> Explanation:
            return Explanation(values=np.array([3]))

    exp = _as_explanation(HasToExplanation())
    assert isinstance(exp, Explanation)
    assert np.array_equal(exp.values, [3])


def test_as_explanation_to_shap() -> None:
    """Test as_explanation_to_shap."""

    class ForeignExp:
        values = (4,)
        base_values = 0.1

    class HasToShap:
        def to_shap(self) -> object:
            return ForeignExp()

    exp = _as_explanation(HasToShap())
    assert isinstance(exp, Explanation)
    assert np.array_equal(exp.values, [4])

    class HasToShapTrue:
        def to_shap(self) -> object:
            return Explanation(values=np.array([5]))

    exp2 = _as_explanation(HasToShapTrue())
    assert np.array_equal(exp2.values, [5])


def test_as_explanation_numpy() -> None:
    """Test as_explanation_numpy."""
    exp = _as_explanation(np.array([5, 6]))
    assert isinstance(exp, Explanation)
    assert np.array_equal(exp.values, [5, 6])


def test_as_explanation_invalid() -> None:
    """Test as_explanation_invalid."""
    with pytest.raises(TypeError, match="Cannot build an Explanation"):
        _as_explanation("not_an_explanation")


# 3. _resolve_names


def test_resolve_names() -> None:
    """Test resolve_names."""
    assert _resolve_names(None, 2) == ["Feature 0", "Feature 1"]
    assert _resolve_names(["a"], 2) == ["a", "Feature 1"]
    assert _resolve_names(["a", "b", "c"], 2) == ["a", "b"]


# 4. _format_value


def test_format_value() -> None:
    """Test format_value."""
    assert _format_value(None) == ""
    assert _format_value("str") == "str"
    assert _format_value(1.230) == "1.23"
    assert _format_value(1.0) == "1"
    assert _format_value(0.0) == "0"

    assert _format_value(5, fmt="%d") == "5"

    class Unformattable:
        def __str__(self) -> str:
            return "unformattable"

        def __float__(self) -> float:
            raise ValueError()

    assert _format_value(Unformattable()) == "unformattable"


# 5. _global_importance


def test_global_importance() -> None:
    """Test global_importance."""
    val_1d = np.array([-2.0, 3.0])
    assert np.array_equal(_global_importance(val_1d), [2.0, 3.0])

    val_2d = np.array([[1.0, -2.0], [-3.0, 4.0]])
    assert np.array_equal(_global_importance(val_2d), [2.0, 3.0])


# 6. _group_minor_features


def test_group_minor_features() -> None:
    """Test group_minor_features."""
    values = np.array([1, 4, 2, 8, 3])
    names = ["a", "b", "c", "d", "e"]

    v, n = _group_minor_features(values, names, None)
    assert np.array_equal(v, [1, 2, 3, 4, 8])
    assert n == ["a", "c", "e", "b", "d"]

    v2, n2 = _group_minor_features(values, names, 3)
    assert np.array_equal(v2, [6, 4, 8])
    assert n2 == ["Sum of 3 other features", "b", "d"]

    v3, _n3 = _group_minor_features(values, names, 10)
    assert len(v3) == 5


# 7. _check_backend


def test_check_backend() -> None:
    """Test check_backend."""
    allowed = frozenset({"matplotlib", "plotly", "html"})
    assert _check_backend("matplotlib", allowed) == "matplotlib"
    assert _check_backend("MPL ", allowed) == "matplotlib"
    assert _check_backend(" plotly", allowed) == "plotly"
    assert _check_backend("px", allowed) == "plotly"
    assert _check_backend("html", allowed) == "html"

    with pytest.raises(ValueError, match="Unknown backend"):
        _check_backend("magic", allowed)

    with pytest.raises(ValueError, match="is not supported"):
        _check_backend("html", frozenset({"matplotlib"}))


# 8. _finish_*


def test_finish_matplotlib_save(tmp_path: Path) -> None:
    """Test finish_matplotlib_save."""
    fig = Figure()
    p = tmp_path / "plot.png"
    assert _finish_matplotlib(fig, show=False, save_path=p) is None
    assert p.exists()


def test_finish_matplotlib_show() -> None:
    """Test finish_matplotlib_show."""
    fig = Figure()
    with patch("matplotlib.pyplot.show") as mock_show:
        assert _finish_matplotlib(fig, show=True, save_path=None) is None
        mock_show.assert_called_once()


def test_finish_matplotlib_return() -> None:
    """Test finish_matplotlib_return."""
    fig = Figure()
    assert _finish_matplotlib(fig, show=False, save_path=None) is fig


def test_finish_plotly_save(tmp_path: Path) -> None:
    """Test finish_plotly_save."""
    fig = go.Figure()
    p = tmp_path / "plot.html"
    assert _finish_plotly(fig, show=False, save_path=p) is None
    assert p.exists()

    p2 = tmp_path / "plot.png"
    with patch("plotly.graph_objects.Figure.write_image") as mock_write:
        assert _finish_plotly(fig, show=False, save_path=p2) is None
        mock_write.assert_called_once_with(str(p2))


def test_finish_plotly_show() -> None:
    """Test finish_plotly_show."""
    fig = go.Figure()
    with patch("plotly.graph_objects.Figure.show") as mock_show:
        assert _finish_plotly(fig, show=True, save_path=None) is None
        mock_show.assert_called_once()


def test_finish_plotly_return() -> None:
    """Test finish_plotly_return."""
    fig = go.Figure()
    assert _finish_plotly(fig, show=False, save_path=None) is fig


def test_finish_html_save(tmp_path: Path) -> None:
    """Test finish_html_save."""
    p = tmp_path / "plot.html"
    assert _finish_html("<b></b>", show=False, save_path=p) == "<b></b>"
    assert p.exists()
    assert "<b></b>" in p.read_text()


def test_finish_html_show() -> None:
    """Test finish_html_show."""
    with patch("xwhy.plots.visualisation.base._display_html") as mock_disp:
        assert _finish_html("<b></b>", show=True, save_path=None) == "<b></b>"
        mock_disp.assert_called_once_with("<b></b>")


def test_finish_html_return() -> None:
    """Test finish_html_return."""
    assert _finish_html("<b></b>", show=False, save_path=None) == "<b></b>"


def test_display_html() -> None:
    """Test display_html."""
    with (
        patch("sys.modules", dict(sys.modules, IPython=MagicMock())),
        patch("IPython.display.display"),
    ):
        _display_html("<b>hi</b>")
        pass


def test_wrap_html_document() -> None:
    """Test wrap_html_document."""
    html = _wrap_html_document("<div>", "My Title")
    assert "<title>My Title</title>" in html
    assert "<div>" in html
    assert html.startswith("<!doctype html>")


def test_style_axes() -> None:
    """Test style_axes."""
    fig, ax = plt.subplots()
    _style_axes(ax)
    assert not ax.spines["left"].get_visible()
    assert not ax.spines["right"].get_visible()
    assert not ax.spines["top"].get_visible()
    assert ax.get_axisbelow()
    plt.close(fig)


def test_single_instance() -> None:
    """Test single_instance."""
    exp1 = Explanation(values=np.array([1, 2]), base_values=np.array([]))
    v, b, n, d = _single_instance(exp1)
    assert np.array_equal(v, [1, 2])
    assert b == 0.0
    assert n == ["Feature 0", "Feature 1"]

    exp2 = Explanation(
        values=np.array([[1, 2]]),
        base_values=np.array([0.5]),
        data=np.array([[10, 20]]),
    )
    v, b, n, d = _single_instance(exp2)
    assert np.array_equal(v, [1, 2])
    assert b == 0.5
    assert np.array_equal(d, [10, 20])

    # 2D, but data is 1D (skips len(np.shape(data)) == 2 branch)
    exp4 = Explanation(
        values=np.array([[1, 2]]),
        base_values=0.5,
        data=np.array([10, 20]),
    )
    _v4, _b4, _n4, d4 = _single_instance(exp4)
    assert np.array_equal(d4, [10, 20])

    # 2D, but base is array of size > 1
    exp5 = Explanation(
        values=np.array([[1, 2]]),
        base_values=np.array([0.5, 0.6]),
    )
    _v5, b5, _, _ = _single_instance(exp5)
    assert b5 == 0.5  # falls through to line 826 and takes [0]

    exp3 = Explanation(values=np.array([[1], [2]]))
    with pytest.raises(ValueError, match="explains a single instance"):
        _single_instance(exp3)


def test_convert_name() -> None:
    """Test convert_name."""
    assert convert_name(None, None, None) is None
    assert convert_name(2, None, None) == 2
    assert convert_name("f2", None, ["f1", "f2"]) == 1
    assert convert_name("sum()", None, None) == "sum()"

    with pytest.raises(ValueError, match="Could not find feature named"):
        convert_name("unknown", None, ["f1"])

    with pytest.raises(ValueError, match="shap_values must be provided"):
        convert_name("rank(0)", None, None)

    shap_vals = np.array([[1, 4], [2, 3]])
    assert convert_name("rank(0)", shap_vals, None) == 1
    assert convert_name("rank(1)", shap_vals, None) == 0


def test_initjs() -> None:
    """Test initjs."""
    assert initjs() is None  # type: ignore[func-returns-value]


def test_dimension_error() -> None:
    """Test dimension_error."""
    with pytest.raises(DimensionError):
        raise DimensionError("msg")
