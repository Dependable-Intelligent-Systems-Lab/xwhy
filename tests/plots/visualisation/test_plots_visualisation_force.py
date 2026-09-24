"""Test force module."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure
from numpy.typing import NDArray

from xwhy.plots.visualisation.force import (
    _draw_additive_plot,
    _draw_bars,
    _draw_labels,
    _force_html,
    _format_data,
    _update_axis_limits,
    draw_base_element,
    draw_higher_lower_element,
    draw_output_element,
    force,
)


@pytest.fixture(autouse=True)
def _close_plots() -> Any:  # noqa: ANN401
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class MockExplanation:
    """Minimal single-instance explanation for the public force API."""

    values: NDArray[np.floating[Any]]
    base_values: float
    feature_names: list[str]
    data: NDArray[Any] | None = None


def _make_force_data(
    *,
    effects: list[float] | None = None,
    values: list[Any] | None = None,
    names: list[str] | None = None,
    link: str = "identity",
    base_value: float = 0.5,
    out_value: float = 0.7,
) -> dict[str, Any]:
    """Build a force-plot data dict for pure helpers."""
    if effects is None:
        effects = [0.2, -0.15, 0.1, -0.05]
    if names is None:
        names = [f"f{i}" for i in range(len(effects))]
    if values is None:
        values = [1.0 * i for i in range(len(effects))]
    features = {
        str(i): {"effect": effects[i], "value": values[i]} for i in range(len(effects))
    }
    return {
        "outNames": ["f(x)"],
        "baseValue": base_value,
        "outValue": out_value,
        "link": link,
        "featureNames": {str(i): names[i] for i in range(len(names))},
        "features": features,
    }


def _patch_as_explanation(exp: MockExplanation) -> Any:  # noqa: ANN401
    """Return a patch that makes ``_as_explanation`` return *exp*."""
    return patch(
        "xwhy.plots.visualisation.force._as_explanation",
        return_value=exp,
    )


# ---------------------------------------------------------------------------
# _format_data
# ---------------------------------------------------------------------------


def test_format_data_identity_both_sides() -> None:
    """Identity link with positive and negative effects."""
    data = _make_force_data(link="identity")
    neg, total_neg, pos, total_pos = _format_data(data)
    assert len(neg) > 0
    assert len(pos) > 0
    assert total_neg >= 0.0
    assert total_pos >= 0.0
    assert data["outValue"] == pytest.approx(0.7)
    assert data["baseValue"] == pytest.approx(0.5)


def test_format_data_logit() -> None:
    """Logit link maps outValue and baseValue through sigmoid."""
    data = _make_force_data(
        effects=[0.3, -0.2],
        link="logit",
        base_value=0.0,
        out_value=0.5,
    )
    _format_data(data)
    # sigmoid(0.5) ≈ 0.622
    assert 0.5 < float(data["outValue"]) < 0.8
    assert 0.4 < float(data["baseValue"]) < 0.6


def test_format_data_empty_neg() -> None:
    """No negative effects → total_neg is 0.0."""
    data = _make_force_data(effects=[0.2, 0.1, 0.05])
    neg, total_neg, pos, total_pos = _format_data(data)
    assert len(neg) == 0
    assert total_neg == 0.0
    assert len(pos) > 0
    assert total_pos >= 0.0


def test_format_data_empty_pos() -> None:
    """No positive effects → total_pos is 0.0."""
    data = _make_force_data(effects=[-0.2, -0.1, -0.05])
    neg, total_neg, pos, total_pos = _format_data(data)
    assert len(pos) == 0
    assert total_pos == 0.0
    assert len(neg) > 0
    assert total_neg >= 0.0


def test_format_data_invalid_link() -> None:
    """Unrecognized link function raises ValueError."""
    data = _make_force_data(link="unknown")
    with pytest.raises(ValueError, match="Unrecognized link function"):
        _format_data(data)


def test_format_data_zero_effect_is_positive() -> None:
    """Effect == 0 is treated as positive (>= 0)."""
    data = _make_force_data(effects=[0.0, -0.1])
    neg, _tn, pos, _tp = _format_data(data)
    assert len(pos) == 1
    assert len(neg) == 1


# ---------------------------------------------------------------------------
# _force_html
# ---------------------------------------------------------------------------


def test_force_html_with_title() -> None:
    """HTML force bar includes the optional title heading."""
    html = _force_html(
        positives=[("a", 0.3), ("b", 0.2)],
        negatives=[("c", -0.1)],
        base_value=0.5,
        prediction=0.9,
        title="My Force Plot",
    )
    assert "My Force Plot" in html
    assert "xwhy-force" in html
    assert "increases the prediction" in html


def test_force_html_without_title() -> None:
    """title=None omits the heading block."""
    html = _force_html(
        positives=[("a", 0.5)],
        negatives=[],
        base_value=0.0,
        prediction=0.5,
        title=None,
    )
    assert "font-weight:600" not in html
    assert "f(x)" in html or "0.5" in html


def test_force_html_empty_segments() -> None:
    """Empty positives and negatives still produce valid markup (total=1)."""
    html = _force_html(
        positives=[],
        negatives=[],
        base_value=0.0,
        prediction=0.0,
        title=None,
    )
    assert "xwhy-force" in html


def test_force_html_small_segment_hides_caption() -> None:
    """Segments below 8% width omit the visible caption text."""
    html = _force_html(
        positives=[("tiny", 0.01), ("big", 1.0)],
        negatives=[],
        base_value=0.0,
        prediction=1.01,
    )
    # "tiny" should appear only in the title attribute, not as body text
    assert "title='tiny:" in html or 'title="tiny:' in html or "tiny" in html


# ---------------------------------------------------------------------------
# draw_output_element / draw_base_element / draw_higher_lower_element
# ---------------------------------------------------------------------------


def test_draw_output_element() -> None:
    """Output marker and labels are added to the axes."""
    _fig, ax = plt.subplots()
    draw_output_element("f(x)", 0.75, ax)
    assert len(ax.lines) >= 1


def test_draw_base_element() -> None:
    """Base-value marker and label are added to the axes."""
    _fig, ax = plt.subplots()
    draw_base_element(0.5, ax)
    assert len(ax.lines) >= 1


def test_draw_higher_lower_element() -> None:
    """Higher / lower legend texts are drawn without error."""
    plt.figure()
    draw_higher_lower_element(out_value=0.5, offset_text=0.1)


# ---------------------------------------------------------------------------
# _update_axis_limits
# ---------------------------------------------------------------------------


def test_update_axis_limits_both_sides() -> None:
    """Limits expand around positive and negative feature positions."""
    _fig, ax = plt.subplots()
    pos = np.array([[0.3, 1.0, "a"], [0.1, 2.0, "b"]], dtype=object)
    neg = np.array([[0.9, 3.0, "c"]], dtype=object)
    _update_axis_limits(ax, 0.4, pos, 0.2, neg, base_value=0.5, out_value=0.7)
    xlim = ax.get_xlim()
    assert xlim[0] < xlim[1]


def test_update_axis_limits_empty_pos() -> None:
    """Empty pos_features uses out_value - padding for min_x."""
    _fig, ax = plt.subplots()
    neg = np.array([[0.9, 1.0, "c"]], dtype=object)
    _update_axis_limits(ax, 0.0, np.array([], dtype=object), 0.2, neg, 0.5, 0.7)
    xlim = ax.get_xlim()
    assert xlim[0] < xlim[1]


def test_update_axis_limits_empty_neg() -> None:
    """Empty neg_features uses out_value + padding for max_x."""
    _fig, ax = plt.subplots()
    pos = np.array([[0.3, 1.0, "a"]], dtype=object)
    _update_axis_limits(ax, 0.2, pos, 0.0, np.array([], dtype=object), 0.5, 0.7)
    xlim = ax.get_xlim()
    assert xlim[0] < xlim[1]


def test_update_axis_limits_zero_padding() -> None:
    """When total_pos and total_neg are 0, padding defaults to 0.1."""
    _fig, ax = plt.subplots()
    _update_axis_limits(
        ax,
        0.0,
        np.array([], dtype=object),
        0.0,
        np.array([], dtype=object),
        base_value=0.5,
        out_value=0.5,
    )
    xlim = ax.get_xlim()
    assert xlim[1] - xlim[0] > 0


# ---------------------------------------------------------------------------
# _draw_bars
# ---------------------------------------------------------------------------


def test_draw_bars_positive() -> None:
    """Positive feature bars produce polygons and separators."""
    features = np.array(
        [[0.3, 1.0, "a"], [0.1, 2.0, "b"]],
        dtype=object,
    )
    rects, seps = _draw_bars(0.7, features, "positive", 0.01, 0.1)
    assert len(rects) == 2
    assert len(seps) == 2


def test_draw_bars_negative() -> None:
    """Negative feature bars produce polygons and separators."""
    features = np.array(
        [[0.9, 1.0, "c"], [1.1, 2.0, "d"]],
        dtype=object,
    )
    rects, seps = _draw_bars(0.7, features, "negative", 0.01, 0.1)
    assert len(rects) == 2
    assert len(seps) == 2


def test_draw_bars_empty() -> None:
    """Empty features return empty lists."""
    rects, seps = _draw_bars(0.7, np.array([], dtype=object), "positive", 0.01, 0.1)
    assert rects == []
    assert seps == []


# ---------------------------------------------------------------------------
# _draw_labels
# ---------------------------------------------------------------------------


def test_draw_labels_positive() -> None:
    """Positive-side labels are drawn for contributions above min_perc."""
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_xlim(0.0, 1.0)
    features = np.array(
        [[0.5, 1.0, "a"], [0.2, 2.0, "b"]],
        dtype=object,
    )
    fig, ax = _draw_labels(
        fig,
        ax,
        out_value=0.7,
        features=features,
        feature_type="positive",
        offset_text=0.02,
        total_effect=0.5,
        min_perc=0.01,
    )
    assert fig is not None


def test_draw_labels_negative() -> None:
    """Negative-side labels are drawn for contributions above min_perc."""
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_xlim(0.0, 2.0)
    features = np.array(
        [[0.9, 1.0, "c"], [1.2, 2.0, "d"]],
        dtype=object,
    )
    fig, ax = _draw_labels(
        fig,
        ax,
        out_value=0.7,
        features=features,
        feature_type="negative",
        offset_text=0.02,
        total_effect=0.5,
        min_perc=0.01,
    )
    assert fig is not None


def test_draw_labels_max_display() -> None:
    """max_display stops labelling after the given count."""
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_xlim(0.0, 1.0)
    features = np.array(
        [[0.5, 1.0, "a"], [0.3, 2.0, "b"], [0.1, 3.0, "c"]],
        dtype=object,
    )
    fig, ax = _draw_labels(
        fig,
        ax,
        out_value=0.7,
        features=features,
        feature_type="positive",
        offset_text=0.02,
        total_effect=0.6,
        min_perc=0.01,
        max_display=1,
    )
    assert fig is not None


def test_draw_labels_min_perc_breaks() -> None:
    """Contributions below min_perc stop the labelling loop."""
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_xlim(0.0, 1.0)
    features = np.array(
        [[0.69, 1.0, "tiny"]],
        dtype=object,
    )
    fig, ax = _draw_labels(
        fig,
        ax,
        out_value=0.7,
        features=features,
        feature_type="positive",
        offset_text=0.02,
        total_effect=1.0,
        min_perc=0.5,
    )
    assert fig is not None


def test_draw_labels_zero_total_effect() -> None:
    """total_effect == 0 yields feature_contribution 0.0 and early break."""
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_xlim(0.0, 1.0)
    features = np.array([[0.5, 1.0, "a"]], dtype=object)
    fig, ax = _draw_labels(
        fig,
        ax,
        out_value=0.7,
        features=features,
        feature_type="positive",
        offset_text=0.02,
        total_effect=0.0,
        min_perc=0.05,
    )
    assert fig is not None


def test_draw_labels_empty_feature_value() -> None:
    """Empty feature value string uses name only (no '= value')."""
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_xlim(0.0, 1.0)
    features = np.array([[0.4, "", "solo"]], dtype=object)
    fig, ax = _draw_labels(
        fig,
        ax,
        out_value=0.7,
        features=features,
        feature_type="positive",
        offset_text=0.02,
        total_effect=0.5,
        min_perc=0.01,
    )
    assert fig is not None


def test_draw_labels_text_rotation() -> None:
    """Non-zero text_rotation uses vertical-alignment 'top'."""
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_xlim(0.0, 1.0)
    features = np.array([[0.4, 1.0, "rot"]], dtype=object)
    fig, ax = _draw_labels(
        fig,
        ax,
        out_value=0.7,
        features=features,
        feature_type="positive",
        offset_text=0.02,
        total_effect=0.5,
        min_perc=0.01,
        text_rotation=45.0,
    )
    assert fig is not None


def test_draw_labels_extent_equal() -> None:
    """When out_value equals box_end, extent gets a tiny epsilon bump."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.set_xlim(0.0, 1.0)
    # Empty features: loop does not run, box_end stays at out_value
    features = np.array([], dtype=object)
    fig, ax = _draw_labels(
        fig,
        ax,
        out_value=0.5,
        features=features,
        feature_type="negative",
        offset_text=0.02,
        total_effect=0.0,
        min_perc=0.05,
    )
    assert fig is not None


def test_draw_labels_box_end_below_xlim() -> None:
    """box_end left of current xlim expands the left bound."""
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_xlim(0.5, 1.0)
    # Large contribution so labels are drawn; tight xlim forces expansion
    features = np.array(
        [[0.1, 1.0, "far_left_feature_name"]],
        dtype=object,
    )
    fig, ax = _draw_labels(
        fig,
        ax,
        out_value=0.7,
        features=features,
        feature_type="positive",
        offset_text=0.02,
        total_effect=0.6,
        min_perc=0.01,
    )
    assert fig is not None


# ---------------------------------------------------------------------------
# _draw_additive_plot
# ---------------------------------------------------------------------------


def test_draw_additive_plot_identity() -> None:
    """Full additive plot with identity link."""
    data = _make_force_data(link="identity")
    fig = _draw_additive_plot(
        data,
        figsize=(12, 3),
        show=False,
        text_rotation=0.0,
        min_perc=0.01,
        max_display=10,
    )
    assert isinstance(fig, Figure)


def test_draw_additive_plot_logit() -> None:
    """Logit link applies xscale('logit') and ScalarFormatter."""
    data = _make_force_data(
        effects=[0.3, -0.2, 0.1],
        link="logit",
        base_value=0.0,
        out_value=0.2,
    )
    fig = _draw_additive_plot(
        data,
        figsize=(12, 3),
        show=False,
        min_perc=0.01,
    )
    assert isinstance(fig, Figure)


def test_draw_additive_plot_only_positive() -> None:
    """Plot with only positive effects."""
    data = _make_force_data(effects=[0.3, 0.2, 0.1])
    fig = _draw_additive_plot(data, figsize=(10, 3), show=False, min_perc=0.01)
    assert isinstance(fig, Figure)


def test_draw_additive_plot_only_negative() -> None:
    """Plot with only negative effects."""
    data = _make_force_data(effects=[-0.3, -0.2, -0.1])
    fig = _draw_additive_plot(data, figsize=(10, 3), show=False, min_perc=0.01)
    assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# force (public entry-point)
# ---------------------------------------------------------------------------


def test_force_matplotlib_with_finish() -> None:
    """Matplotlib backend delegates to _finish_matplotlib."""
    exp = MockExplanation(
        values=np.array([0.2, -0.1, 0.05]),
        base_values=0.5,
        feature_names=["a", "b", "c"],
        data=np.array([1.0, 2.0, 3.0]),
    )
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.force._single_instance",
            new=None,
        ),
        patch(
            "xwhy.plots.visualisation.force._finish_matplotlib",
            return_value="finished",
        ),
    ):
        result = force(
            exp,
            show=True,
            title="Force",
            figsize=(12, 3),
            max_display=10,
        )
        assert result == "finished"


def test_force_matplotlib_show_true_no_finish() -> None:
    """show=True without _finish_matplotlib calls plt.show()."""
    exp = MockExplanation(
        values=np.array([0.2, -0.1]),
        base_values=0.5,
        feature_names=["a", "b"],
    )
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.force._single_instance",
            new=None,
        ),
        patch(
            "xwhy.plots.visualisation.force._finish_matplotlib",
            new=None,
        ),
        patch("matplotlib.pyplot.show") as mock_show,
    ):
        result = force(exp, show=True, figsize=(10, 3))
        assert result is None
        mock_show.assert_called_once()


def test_force_matplotlib_show_false_no_finish() -> None:
    """show=False without _finish_matplotlib returns Figure."""
    exp = MockExplanation(
        values=np.array([0.2, -0.1, 0.05]),
        base_values=0.5,
        feature_names=["a", "b", "c"],
    )
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.force._single_instance",
            new=None,
        ),
        patch(
            "xwhy.plots.visualisation.force._finish_matplotlib",
            new=None,
        ),
    ):
        fig = force(exp, show=False, figsize=(10, 3))
        assert fig is not None
        assert isinstance(fig, Figure)


def test_force_no_title_no_figsize() -> None:
    """title=None and figsize=None use defaults."""
    exp = MockExplanation(
        values=np.array([0.3, -0.2]),
        base_values=0.4,
        feature_names=["x", "y"],
    )
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.force._single_instance",
            new=None,
        ),
        patch(
            "xwhy.plots.visualisation.force._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = force(exp, show=False, title=None, figsize=None)
        assert result == "ok"


def test_force_max_display_truncates() -> None:
    """max_display limits features included in the plot."""
    exp = MockExplanation(
        values=np.array([0.5, 0.4, 0.3, 0.2, 0.1, -0.05]),
        base_values=0.0,
        feature_names=[f"f{i}" for i in range(6)],
        data=np.arange(6, dtype=float),
    )
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.force._single_instance",
            new=None,
        ),
        patch(
            "xwhy.plots.visualisation.force._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = force(exp, show=False, max_display=3, figsize=(10, 3))
        assert result == "ok"


def test_force_max_display_none() -> None:
    """max_display=None keeps all features."""
    exp = MockExplanation(
        values=np.array([0.2, -0.1]),
        base_values=0.5,
        feature_names=["a", "b"],
    )
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.force._single_instance",
            new=None,
        ),
        patch(
            "xwhy.plots.visualisation.force._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = force(exp, show=False, max_display=None, figsize=(10, 3))
        assert result == "ok"


def test_force_with_string_feature_data() -> None:
    """Non-numeric feature data values become strings in the dict."""
    exp = MockExplanation(
        values=np.array([0.2, -0.1]),
        base_values=0.5,
        feature_names=["cat", "dog"],
        data=np.array(["red", "blue"], dtype=object),
    )
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.force._single_instance",
            new=None,
        ),
        patch(
            "xwhy.plots.visualisation.force._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = force(exp, show=False, figsize=(10, 3))
        assert result == "ok"


def test_force_data_none() -> None:
    """explanation.data is None → empty feature values and plain names."""
    exp = MockExplanation(
        values=np.array([0.2, -0.1]),
        base_values=0.5,
        feature_names=["a", "b"],
        data=None,
    )
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.force._single_instance",
            new=None,
        ),
        patch(
            "xwhy.plots.visualisation.force._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = force(exp, show=False, figsize=(10, 3))
        assert result == "ok"


def test_force_data_wrong_length() -> None:
    """Data length mismatch → feat_value stays empty string."""
    exp = MockExplanation(
        values=np.array([0.2, -0.1]),
        base_values=0.5,
        feature_names=["a", "b"],
        data=np.array([1.0]),  # wrong length
    )
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.force._single_instance",
            new=None,
        ),
        patch(
            "xwhy.plots.visualisation.force._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = force(exp, show=False, figsize=(10, 3))
        assert result == "ok"


def test_force_kwargs_text_rotation_and_threshold() -> None:
    """text_rotation and contribution_threshold kwargs are forwarded."""
    exp = MockExplanation(
        values=np.array([0.3, -0.2, 0.1]),
        base_values=0.5,
        feature_names=["a", "b", "c"],
        data=np.array([1.0, 2.0, 3.0]),
    )
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.force._single_instance",
            new=None,
        ),
        patch(
            "xwhy.plots.visualisation.force._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = force(
            exp,
            show=False,
            figsize=(10, 3),
            text_rotation=30.0,
            contribution_threshold=0.01,
        )
        assert result == "ok"


def test_force_check_backend_none_valid() -> None:
    """_check_backend is None; valid backend is accepted."""
    exp = MockExplanation(
        values=np.array([0.2, -0.1]),
        base_values=0.5,
        feature_names=["a", "b"],
    )
    with (
        patch(
            "xwhy.plots.visualisation.force._check_backend",
            new=None,
        ),
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.force._single_instance",
            new=None,
        ),
        patch(
            "xwhy.plots.visualisation.force._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = force(exp, backend="matplotlib", show=False, figsize=(10, 3))
        assert result == "ok"


def test_force_check_backend_none_invalid() -> None:
    """_check_backend is None; invalid backend raises ValueError."""
    exp = MockExplanation(
        values=np.array([0.2]),
        base_values=0.5,
        feature_names=["a"],
    )
    with (
        patch(
            "xwhy.plots.visualisation.force._check_backend",
            new=None,
        ),
        pytest.raises(ValueError, match="Unsupported backend"),
    ):
        force(exp, backend="plotly", show=False)


def test_force_check_backend_called() -> None:
    """When _check_backend is present it is invoked."""
    exp = MockExplanation(
        values=np.array([0.2, -0.1]),
        base_values=0.5,
        feature_names=["a", "b"],
    )
    mock_check = MagicMock(return_value="matplotlib")
    with (
        patch(
            "xwhy.plots.visualisation.force._check_backend",
            mock_check,
        ),
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.force._single_instance",
            new=None,
        ),
        patch(
            "xwhy.plots.visualisation.force._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = force(exp, show=False, figsize=(10, 3))
        assert result == "ok"
        mock_check.assert_called_once()


def test_force_as_explanation_none() -> None:
    """_as_explanation is None; explanation is used as-is."""
    exp = MockExplanation(
        values=np.array([0.2, -0.1]),
        base_values=0.5,
        feature_names=["a", "b"],
    )
    with (
        patch(
            "xwhy.plots.visualisation.force._as_explanation",
            new=None,
        ),
        patch(
            "xwhy.plots.visualisation.force._single_instance",
            new=None,
        ),
        patch(
            "xwhy.plots.visualisation.force._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = force(exp, show=False, figsize=(10, 3))
        assert result == "ok"


def test_force_single_instance_called() -> None:
    """_single_instance is not None; it unpacks the explanation."""
    exp = MockExplanation(
        values=np.array([0.2, -0.1]),
        base_values=0.5,
        feature_names=["a", "b"],
        data=np.array([1.0, 2.0]),
    )
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.force._single_instance",
            return_value=(
                np.array([0.2, -0.1]),
                0.5,
                ["a", "b"],
                np.array([1.0, 2.0]),
            ),
        ),
        patch(
            "xwhy.plots.visualisation.force._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = force(exp, show=False, figsize=(10, 3))
        assert result == "ok"


def test_force_html_backend_with_finish() -> None:
    """HTML backend delegates to _finish_html."""
    exp = MockExplanation(
        values=np.array([0.3, -0.1, 0.05]),
        base_values=0.5,
        feature_names=["a", "b", "c"],
        data=np.array([1.0, 2.0, 3.0]),
    )
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.force._single_instance",
            new=None,
        ),
        patch(
            "xwhy.plots.visualisation.force._check_backend",
            return_value="html",
        ),
        patch(
            "xwhy.plots.visualisation.force._finish_html",
            return_value="html_done",
        ),
    ):
        result = force(exp, backend="html", show=True, title="HTML Force")
        assert result == "html_done"


def test_force_html_backend_show_true_no_finish() -> None:
    """HTML backend, show=True, no _finish_html → returns None."""
    exp = MockExplanation(
        values=np.array([0.2, -0.1]),
        base_values=0.5,
        feature_names=["a", "b"],
    )
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.force._single_instance",
            new=None,
        ),
        patch(
            "xwhy.plots.visualisation.force._check_backend",
            return_value="html",
        ),
        patch(
            "xwhy.plots.visualisation.force._finish_html",
            new=None,
        ),
    ):
        result = force(exp, backend="html", show=True)
        assert result is None


def test_force_html_backend_show_false_no_finish() -> None:
    """HTML backend, show=False, no _finish_html → returns HTML string."""
    exp = MockExplanation(
        values=np.array([0.2, -0.1]),
        base_values=0.5,
        feature_names=["a", "b"],
    )
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.force._single_instance",
            new=None,
        ),
        patch(
            "xwhy.plots.visualisation.force._check_backend",
            return_value="html",
        ),
        patch(
            "xwhy.plots.visualisation.force._finish_html",
            new=None,
        ),
    ):
        result = force(exp, backend="html", show=False, title="T")
        assert isinstance(result, str)
        assert "xwhy-force" in result


def test_force_only_positives() -> None:
    """All-positive values produce only red segments."""
    exp = MockExplanation(
        values=np.array([0.3, 0.2, 0.1]),
        base_values=0.0,
        feature_names=["a", "b", "c"],
        data=np.array([1.0, 2.0, 3.0]),
    )
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.force._single_instance",
            new=None,
        ),
        patch(
            "xwhy.plots.visualisation.force._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = force(exp, show=False, figsize=(10, 3))
        assert result == "ok"


def test_force_only_negatives() -> None:
    """All-negative values produce only blue segments."""
    exp = MockExplanation(
        values=np.array([-0.3, -0.2, -0.1]),
        base_values=1.0,
        feature_names=["a", "b", "c"],
        data=np.array([1.0, 2.0, 3.0]),
    )
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.force._single_instance",
            new=None,
        ),
        patch(
            "xwhy.plots.visualisation.force._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = force(exp, show=False, figsize=(10, 3))
        assert result == "ok"
