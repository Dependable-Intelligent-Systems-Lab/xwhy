"""Unit tests for the bar plot visualisation."""

from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pytest
from matplotlib.figure import Figure

from xwhy.plots.visualisation.bar import _bar_colors, bar
from xwhy.plots.visualisation.base import BLUE, RED, Explanation


@pytest.fixture(autouse=True)
def _cleanup_plots() -> Generator[None, None, None]:
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


def test_bar_colors() -> None:
    """Verify that _bar_colors correctly maps values to RED/BLUE."""
    values = np.array([1.0, -0.5, 0.0, 2.5])
    colors = _bar_colors(values)
    assert colors == [RED, BLUE, RED, RED]


def test_bar_matplotlib_local() -> None:
    """Verify matplotlib rendering for a local (1D) explanation."""
    exp = Explanation(
        values=np.array([0.1, -0.2, 0.3]),
        feature_names=["f1", "f2", "f3"],
    )
    fig = bar(exp, backend="matplotlib", show=False)

    assert isinstance(fig, Figure)
    ax = fig.gca()
    assert len(ax.patches) == 3
    # Check vertical line at 0 for local plots
    lines = [line for line in ax.get_lines() if line.get_xdata()[0] == 0]  # type: ignore[index]
    assert len(lines) > 0


def test_bar_matplotlib_global() -> None:
    """Verify matplotlib rendering for a global (2D) explanation with grouping."""
    exp = Explanation(
        values=np.array([[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]]),
        feature_names=["f1", "f2", "f3"],
    )

    fig = bar(
        exp,
        backend="matplotlib",
        show=False,
        max_display=2,
        title="Global Plot",
        figsize=(10, 5),
    )

    assert isinstance(fig, Figure)
    ax = fig.gca()
    # 2 display rows: 1 kept + 1 summary group
    assert len(ax.patches) == 2
    assert ax.get_title(loc="left") == "Global Plot"


def test_bar_matplotlib_save(tmp_path: Path) -> None:
    """Verify that matplotlib plots can be saved to disk."""
    exp = Explanation(values=np.array([0.1, 0.2]))
    path = tmp_path / "plot.png"
    result = bar(exp, backend="matplotlib", save_path=path, show=False)

    assert result is None
    assert path.exists()


def test_bar_matplotlib_existing_ax() -> None:
    """Verify that matplotlib uses an existing Axes if provided."""
    exp = Explanation(values=np.array([0.1, -0.2]))
    fig, ax = plt.subplots()
    result = bar(exp, backend="matplotlib", ax=ax, show=False)

    assert result is fig
    assert len(ax.patches) == 2


def test_bar_plotly_local() -> None:
    """Verify plotly rendering for a local (1D) explanation."""
    exp = Explanation(
        values=np.array([0.1, -0.2, 0.3]),
        feature_names=["A", "B", "C"],
    )
    fig = bar(exp, backend="plotly", show=False, title="Plotly Local")

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "bar"
    assert fig.layout.title.text == "Plotly Local"


def test_bar_plotly_global() -> None:
    """Verify plotly rendering for a global (2D) explanation."""
    exp = Explanation(
        values=np.array([[0.1, -0.2, 0.3], [-0.4, 0.5, 0.6]]),
        feature_names=["A", "B", "C"],
    )
    fig = bar(exp, backend="plotly", show=False, max_display=2)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


def test_bar_plotly_save(tmp_path: Path) -> None:
    """Verify that plotly plots can be saved to disk as HTML."""
    exp = Explanation(values=np.array([0.1, 0.2]))
    path = tmp_path / "plot.html"
    result = bar(exp, backend="plotly", save_path=path, show=False)

    assert result is None
    assert path.exists()


def test_bar_plotly_save_image(tmp_path: Path) -> None:
    """Verify that plotly plots can be saved to disk as PNG."""
    exp = Explanation(values=np.array([0.1, 0.2]))
    path = tmp_path / "plot.png"
    # Note: Requires kaleido installed to test in real environment
    with patch("plotly.graph_objects.Figure.write_image") as mock_write:
        result = bar(exp, backend="plotly", save_path=path, show=False)
        assert result is None
        mock_write.assert_called_once_with(str(path))


def test_bar_invalid_backend() -> None:
    """Verify that an unknown backend raises an error."""
    exp = Explanation(values=np.array([1.0]))
    with pytest.raises(ValueError, match="Unknown backend"):
        bar(exp, backend="unknown")


def test_bar_unsupported_backend() -> None:
    """Verify that a known but unsupported backend raises an error."""
    exp = Explanation(values=np.array([1.0]))
    with pytest.raises(ValueError, match="is not supported by this plot"):
        bar(exp, backend="html")


def test_bar_show_matplotlib() -> None:
    """Verify that plt.show is called when show=True."""
    exp = Explanation(values=np.array([0.1]))
    with patch("matplotlib.pyplot.show") as mock_show:
        result = bar(exp, backend="matplotlib", show=True)
        assert result is None
        mock_show.assert_called_once()


def test_bar_show_plotly() -> None:
    """Verify that fig.show is called when show=True."""
    exp = Explanation(values=np.array([0.1]))
    with patch("plotly.graph_objects.Figure.show") as mock_show:
        result = bar(exp, backend="plotly", show=True)
        assert result is None
        mock_show.assert_called_once()


def test_bar_empty() -> None:
    """Verify that an empty explanation renders safely."""
    exp = Explanation(values=np.array([]))
    fig = bar(exp, backend="matplotlib", show=False)

    assert isinstance(fig, Figure)
    ax = fig.gca()
    assert len(ax.patches) == 0


def test_bar_zero_variance() -> None:
    """Verify that x-limits handle zero variance gracefully."""
    exp = Explanation(values=np.array([0.0, 0.0]))
    fig = bar(exp, backend="matplotlib", show=False)

    assert isinstance(fig, Figure)
    ax = fig.gca()
    xmin, xmax = ax.get_xlim()
    assert xmin < xmax


def test_bar_equal_limits() -> None:
    """Verify that equal x-limits are expanded."""
    exp = Explanation(values=np.array([1.0]))
    _, ax = plt.subplots()

    with (
        patch("matplotlib.axes.Axes.get_xlim", return_value=(2.0, 2.0)),
        patch("matplotlib.axes.Axes.set_xlim") as mock_set_xlim,
    ):
        bar(exp, backend="matplotlib", ax=ax, show=False)
        mock_set_xlim.assert_called_with(1.0, 3.0)
