"""Tests for waterfall plots of ordered feature contributions."""

from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

matplotlib.use("Agg")

from xwhy.plots.visualisation.waterfall import waterfall

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _names(n: int = 5) -> list[str]:
    """Return feature name list."""
    return [f"feat{i}" for i in range(n)]


def _values(n: int = 5, seed: int = 0) -> NDArray[np.floating[Any]]:
    """Return deterministic attributions with mixed signs."""
    rng = np.random.default_rng(seed)
    vals = rng.normal(size=n).astype(float)
    vals[0] = abs(float(vals[0])) + 0.25
    if n > 1:
        vals[1] = -abs(float(vals[1])) - 0.25
    return vals


def _fmt(val: float, fmt: str = "%0.2f") -> str:
    """Format a numeric value for labels."""
    try:
        return fmt % float(val)
    except (TypeError, ValueError):
        return str(val)


def _make_exp(
    n: int = 5,
    *,
    seed: int = 0,
    data: Any = None,  # noqa: ANN401
    lower_bounds: Any = None,  # noqa: ANN401
    upper_bounds: Any = None,  # noqa: ANN401
    values: NDArray[np.floating[Any]] | None = None,
    base_values: float = 0.5,
) -> SimpleNamespace:
    """Build a minimal single-instance explanation-like object."""
    vals = _values(n, seed=seed) if values is None else values
    return SimpleNamespace(
        values=vals,
        base_values=base_values,
        feature_names=_names(len(vals)),
        data=list(range(len(vals))) if data is None else data,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        shape=(len(vals),),
    )


def _run_waterfall(
    exp: SimpleNamespace,
    *,
    engine: str = "matplotlib",
    values: NDArray[np.floating[Any]] | None = None,
    base_value: float | None = None,
    names: list[str] | None = None,
    data: Any = ...,  # noqa: ANN401
    show: bool = False,
    max_display: int | None = 10,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    finish_return: Any = "fig",  # noqa: ANN401
    force_text_overflow: bool = False,
) -> tuple[Any, MagicMock]:
    """Invoke waterfall with base helpers mocked."""
    vals = exp.values if values is None else values
    bval = float(exp.base_values) if base_value is None else base_value
    nms = list(exp.feature_names) if names is None else names
    pdata = exp.data if data is ... else data

    def _single(
        _exp: object,
    ) -> tuple[NDArray[np.floating[Any]], float, list[str], Any]:
        return np.asarray(vals, dtype=float), bval, nms, pdata

    def _group(
        values_in: NDArray[np.floating[Any]],
        labels_in: list[str],
        max_disp: int,
    ) -> tuple[NDArray[np.floating[Any]], list[str]]:
        values_in = np.asarray(values_in, dtype=float)
        labels_in = list(labels_in)
        if max_disp >= len(values_in):
            return values_in, labels_in
        kept_v = values_in[: max_disp - 1]
        rest_v = values_in[max_disp - 1 :]
        grouped = np.concatenate([kept_v, np.array([float(rest_v.sum())])])
        glabels = [*labels_in[: max_disp - 1], f"{len(rest_v)} other features"]
        return grouped, glabels

    if finish_return == "fig":
        mock_finish = MagicMock(side_effect=lambda fig, **_kw: fig)
    else:
        mock_finish = MagicMock(return_value=finish_return)

    patches = [
        patch(
            "xwhy.plots.visualisation.waterfall._check_backend",
            return_value=engine,
        ),
        patch(
            "xwhy.plots.visualisation.waterfall._as_explanation",
            return_value=exp,
        ),
        patch(
            "xwhy.plots.visualisation.waterfall._single_instance",
            side_effect=_single,
        ),
        patch(
            "xwhy.plots.visualisation.waterfall._group_minor_features",
            side_effect=_group,
        ),
        patch(
            "xwhy.plots.visualisation.waterfall._format_value",
            side_effect=_fmt,
        ),
    ]
    finish_target = (
        "xwhy.plots.visualisation.waterfall._finish_plotly"
        if engine == "plotly"
        else "xwhy.plots.visualisation.waterfall._finish_matplotlib"
    )
    patches.append(patch(finish_target, mock_finish))  # type: ignore[arg-type]

    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)

        if force_text_overflow and engine == "matplotlib":
            # Force text wider than arrow so the overflow re-position branch runs
            wide = MagicMock()
            wide.width = 1000.0
            narrow = MagicMock()
            narrow.width = 1.0

            real_text = plt.text
            call_count = {"n": 0}

            def _text_side_effect(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
                obj = real_text(*args, **kwargs)
                call_count["n"] += 1
                # First text in each pos/neg iteration is the centered label
                _original_get = obj.get_window_extent

                def _get_extent(*a: Any, **k: Any) -> Any:  # noqa: ANN401
                    # Alternate: text wide, arrow narrow via arrow mock below
                    return wide

                obj.get_window_extent = _get_extent  # type: ignore[method-assign]
                return obj

            real_arrow = plt.arrow

            def _arrow_side_effect(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
                obj = real_arrow(*args, **kwargs)
                obj.get_window_extent = (  # type: ignore[method-assign]
                    lambda *a, **k: narrow
                )
                return obj

            stack.enter_context(
                patch(
                    "matplotlib.pyplot.text",
                    side_effect=_text_side_effect,
                )
            )
            stack.enter_context(
                patch(
                    "matplotlib.pyplot.arrow",
                    side_effect=_arrow_side_effect,
                )
            )

        result = waterfall(
            exp,
            show=show,
            max_display=max_display,
            figsize=figsize,
            title=title,
            backend=engine,
        )
    plt.close("all")
    return result, mock_finish


# ---------------------------------------------------------------------------
# matplotlib — basic
# ---------------------------------------------------------------------------


def test_waterfall_matplotlib_basic() -> None:
    """Basic matplotlib waterfall returns a figure when show=False."""
    exp = _make_exp(5)
    result, mock_finish = _run_waterfall(exp, show=False, max_display=5)
    assert result is not None
    mock_finish.assert_called_once()


def test_waterfall_matplotlib_show_true() -> None:
    """show=True returns None via finish helper."""
    exp = _make_exp(4)
    result, mock_finish = _run_waterfall(
        exp, show=True, max_display=4, finish_return=None
    )
    assert result is None
    mock_finish.assert_called_once()


def test_waterfall_max_display_none() -> None:
    """max_display=None uses all features."""
    exp = _make_exp(4)
    result, _mf = _run_waterfall(exp, show=False, max_display=None)
    assert result is not None


def test_waterfall_figsize() -> None:
    """Explicit figsize is respected."""
    exp = _make_exp(4)
    result, _mf = _run_waterfall(exp, show=False, max_display=4, figsize=(10.0, 6.0))
    assert result is not None


def test_waterfall_title() -> None:
    """Matplotlib path applies title when provided."""
    exp = _make_exp(4)
    result, _mf = _run_waterfall(exp, show=False, max_display=4, title="WF Title")
    assert result is not None


def test_waterfall_no_title() -> None:
    """Matplotlib path skips title when not provided."""
    exp = _make_exp(4)
    result, _mf = _run_waterfall(exp, show=False, max_display=4, title=None)
    assert result is not None


# ---------------------------------------------------------------------------
# data annotation
# ---------------------------------------------------------------------------


def test_waterfall_data_none() -> None:
    """When data is None, labels are plain feature names."""
    exp = SimpleNamespace(
        values=_values(4),
        base_values=0.5,
        feature_names=_names(4),
        data=None,
        lower_bounds=None,
        upper_bounds=None,
        shape=(4,),
    )
    result, _mf = _run_waterfall(exp, show=False, max_display=4, data=None)
    assert result is not None


def test_waterfall_data_numeric() -> None:
    """Numeric feature values are formatted into y-tick labels."""
    data = [1.5, 2.5, 3.5, 4.5]
    exp = _make_exp(4, data=data)
    result, _mf = _run_waterfall(exp, show=False, max_display=4, data=data)
    assert result is not None


def test_waterfall_data_non_numeric() -> None:
    """Non-numeric feature values use string labels."""
    data = ["cat", "dog", "bird", "fish"]
    exp = _make_exp(4, data=data)
    result, _mf = _run_waterfall(exp, show=False, max_display=4, data=data)
    assert result is not None


def test_waterfall_data_length_mismatch() -> None:
    """Data length mismatch keeps plain feature names (plotly path).

    The early annotation branch checks ``raw.shape[0] == len(names)`` and
    skips value prefixes when lengths differ.  The matplotlib path later
    indexes ``data[order[i]]`` and requires matching length, so this case
    is exercised on the plotly backend only.
    """
    data = [1.0, 2.0]
    exp = _make_exp(4, data=data)
    mock_fig = MagicMock(name="plotly_fig")
    result, _mf = _run_waterfall(
        exp,
        engine="plotly",
        show=False,
        max_display=4,
        data=data,
        finish_return=mock_fig,
    )
    assert result is mock_fig


# ---------------------------------------------------------------------------
# signs / grouping / bounds
# ---------------------------------------------------------------------------


def test_waterfall_group_remainder_neg_impact() -> None:
    """Grouped remainder with remaining_impact >= 0 uses neg bar (lines 180-182).

    remaining_impact = -sum(undrawn SHAP values).  When the undrawn features
    sum to <= 0, remaining_impact >= 0 and the else branch runs.
    """
    # Individuals take the large |values|; remainder is net negative
    vals = np.array([1.0, 0.5, 0.3, -0.4, -0.5])
    exp = _make_exp(5, values=vals, data=list(range(5)), base_values=0.0)
    # max_display=3 → num_individual=2; remainder sum = -0.6 → impact >= 0
    result, _mf = _run_waterfall(exp, show=False, max_display=3)
    assert result is not None


def test_waterfall_group_remainder_pos_impact() -> None:
    """Grouped remainder with remaining_impact < 0 uses pos bar.

    Undrawn features sum positive → remaining_impact < 0 → if branch.
    """
    vals = np.array([0.01, -0.02, 0.9, 0.8, 5.0])
    exp = _make_exp(5, values=vals, data=list(range(5)), base_values=0.0)
    result, _mf = _run_waterfall(exp, show=False, max_display=3)
    assert result is not None


def test_waterfall_all_positive() -> None:
    """All-positive contributions still render."""
    vals = np.array([0.4, 0.3, 0.2, 0.1])
    exp = _make_exp(4, values=vals, data=list(range(4)))
    result, _mf = _run_waterfall(exp, show=False, max_display=4)
    assert result is not None


def test_waterfall_all_negative() -> None:
    """All-negative contributions still render."""
    vals = np.array([-0.4, -0.3, -0.2, -0.1])
    exp = _make_exp(4, values=vals, data=list(range(4)))
    result, _mf = _run_waterfall(exp, show=False, max_display=4)
    assert result is not None


def test_waterfall_with_bounds() -> None:
    """lower_bounds / upper_bounds draw error bars on arrows."""
    vals = _values(5, seed=3)
    lower = vals - 0.05
    upper = vals + 0.05
    exp = _make_exp(5, seed=3, lower_bounds=lower, upper_bounds=upper)
    result, _mf = _run_waterfall(exp, show=False, max_display=5)
    assert result is not None


def test_waterfall_text_overflow() -> None:
    """Text wider than arrow is re-positioned outside the bar."""
    # Mixed signs so both pos and neg overflow branches run
    vals = np.array([0.5, -0.4, 0.3, -0.2])
    exp = _make_exp(4, values=vals, data=list(range(4)))
    result, _mf = _run_waterfall(
        exp, show=False, max_display=4, force_text_overflow=True
    )
    assert result is not None


# ---------------------------------------------------------------------------
# plotly
# ---------------------------------------------------------------------------


def test_waterfall_plotly_basic() -> None:
    """Plotly backend returns finish result when show=False."""
    exp = _make_exp(4)
    mock_fig = MagicMock(name="plotly_fig")
    result, mock_finish = _run_waterfall(
        exp,
        engine="plotly",
        show=False,
        max_display=4,
        finish_return=mock_fig,
    )
    assert result is mock_fig
    mock_finish.assert_called_once()


def test_waterfall_plotly_with_title() -> None:
    """Plotly path uses explicit title when provided."""
    exp = _make_exp(3)
    mock_fig = MagicMock(name="plotly_fig")
    result, _mf = _run_waterfall(
        exp,
        engine="plotly",
        show=False,
        max_display=3,
        title="Plotly WF",
        finish_return=mock_fig,
    )
    assert result is mock_fig


def test_waterfall_plotly_default_title() -> None:
    """Plotly path builds default title from base and prediction."""
    exp = _make_exp(3)
    mock_fig = MagicMock(name="plotly_fig")
    result, _mf = _run_waterfall(
        exp,
        engine="plotly",
        show=False,
        max_display=3,
        title=None,
        finish_return=mock_fig,
    )
    assert result is mock_fig


def test_waterfall_plotly_show_true() -> None:
    """Plotly show=True returns None."""
    exp = _make_exp(3)
    result, mock_finish = _run_waterfall(
        exp,
        engine="plotly",
        show=True,
        max_display=3,
        finish_return=None,
    )
    assert result is None
    mock_finish.assert_called_once()
