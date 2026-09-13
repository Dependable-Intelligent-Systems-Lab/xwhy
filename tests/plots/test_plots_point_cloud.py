"""Unit tests for point cloud plotting utilities."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from xwhy.core.result import PointCloudXWhyResult
from xwhy.plots.point_cloud import (
    create_clean_3d_layout,
    create_point_cloud_trace,
    create_rotation_frames,
    display_plotly_figure,
    plot_3d_mesh,
    plot_3d_point_cloud,
    plot_colored_3d_point_cloud,
    plot_point_cloud,
    plot_point_cloud_clusters,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_vertices() -> np.ndarray:
    """Return a small set of 3-D vertices."""
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


@pytest.fixture
def sample_faces() -> np.ndarray:
    """Return a minimal face index array."""
    return np.array([[0, 1, 2], [0, 2, 3]], dtype=int)


@pytest.fixture
def sample_result(sample_vertices: np.ndarray) -> PointCloudXWhyResult:
    """Return a PointCloudXWhyResult with deterministic data."""
    return PointCloudXWhyResult(
        coefficients=np.array([0.5, -0.2, 0.8]),
        metrics=MagicMock(),
        important_clusters=np.array([0, 2]),
        sample_points=sample_vertices,
        cluster_labels=np.array([0, 0, 1, 2]),
    )


# ---------------------------------------------------------------------------
# create_rotation_frames
# ---------------------------------------------------------------------------


def test_create_rotation_frames_default() -> None:
    """Generate the expected number of camera frames."""
    frames = create_rotation_frames()
    assert len(frames) == 100
    assert "layout" in frames[0]
    assert "scene" in frames[0]["layout"]
    assert "camera" in frames[0]["layout"]["scene"]


def test_create_rotation_frames_custom_steps() -> None:
    """Honour a custom number of animation steps."""
    frames = create_rotation_frames(num_steps=10, radius=3.0, height=1.5)
    assert len(frames) == 10
    eye = frames[0]["layout"]["scene"]["camera"]["eye"]
    assert eye["z"] == 1.5


# ---------------------------------------------------------------------------
# plot_3d_mesh / plot_3d_point_cloud / plot_colored_3d_point_cloud
# ---------------------------------------------------------------------------


@patch("xwhy.plots.point_cloud.go.Figure")
@patch("xwhy.plots.point_cloud.go.Mesh3d")
def test_plot_3d_mesh(
    mock_mesh: MagicMock,
    mock_figure: MagicMock,
    sample_vertices: np.ndarray,
    sample_faces: np.ndarray,
) -> None:
    """Build a Plotly mesh figure with rotation frames."""
    mock_figure.return_value = MagicMock()
    fig = plot_3d_mesh(sample_vertices, sample_faces, opacity=0.7)
    mock_mesh.assert_called_once()
    mock_figure.assert_called_once()
    assert fig is mock_figure.return_value


@patch("xwhy.plots.point_cloud.go.Figure")
@patch("xwhy.plots.point_cloud.go.Scatter3d")
def test_plot_3d_point_cloud(
    mock_scatter: MagicMock,
    mock_figure: MagicMock,
    sample_vertices: np.ndarray,
) -> None:
    """Build a Plotly scatter figure for a plain point cloud."""
    mock_figure.return_value = MagicMock()
    fig = plot_3d_point_cloud(sample_vertices, marker_size=3)
    mock_scatter.assert_called_once()
    assert fig is mock_figure.return_value


@patch("xwhy.plots.point_cloud.go.Figure")
@patch("xwhy.plots.point_cloud.go.Scatter3d")
def test_plot_colored_3d_point_cloud_with_colorbar(
    mock_scatter: MagicMock,
    mock_figure: MagicMock,
    sample_vertices: np.ndarray,
) -> None:
    """Include a colorbar when show_colorbar is True."""
    mock_figure.return_value = MagicMock()
    importance = np.array([0.1, 0.5, 0.9, 0.3])
    fig = plot_colored_3d_point_cloud(sample_vertices, importance, show_colorbar=True)
    call_kwargs = mock_scatter.call_args.kwargs
    assert call_kwargs["marker"]["colorbar"] is not None
    assert fig is mock_figure.return_value


@patch("xwhy.plots.point_cloud.go.Figure")
@patch("xwhy.plots.point_cloud.go.Scatter3d")
def test_plot_colored_3d_point_cloud_without_colorbar(
    mock_scatter: MagicMock,
    mock_figure: MagicMock,
    sample_vertices: np.ndarray,
) -> None:
    """Omit the colorbar when show_colorbar is False."""
    mock_figure.return_value = MagicMock()
    importance = np.array([0.1, 0.5, 0.9, 0.3])
    plot_colored_3d_point_cloud(sample_vertices, importance, show_colorbar=False)
    call_kwargs = mock_scatter.call_args.kwargs
    assert call_kwargs["marker"]["colorbar"] is None


# ---------------------------------------------------------------------------
# display_plotly_figure
# ---------------------------------------------------------------------------


@patch("xwhy.plots.point_cloud.display")
@patch("xwhy.plots.point_cloud.HTML")
def test_display_plotly_figure(
    mock_html: MagicMock,
    mock_display: MagicMock,
) -> None:
    """Convert the figure to HTML and hand it to IPython display."""
    fig = MagicMock()
    fig.to_html.return_value = "<div>plot</div>"
    mock_html.return_value = "html_obj"

    display_plotly_figure(fig)

    fig.to_html.assert_called_once_with(
        include_plotlyjs="cdn", full_html=False, auto_play=False
    )
    mock_html.assert_called_once_with("<div>plot</div>")
    mock_display.assert_called_once_with("html_obj")


# ---------------------------------------------------------------------------
# plot_point_cloud_clusters
# ---------------------------------------------------------------------------


@patch("xwhy.plots.point_cloud.go.Figure")
@patch("xwhy.plots.point_cloud.go.Scatter3d")
@patch("xwhy.plots.point_cloud.plt.get_cmap")
def test_plot_point_cloud_clusters(
    mock_cmap: MagicMock,
    mock_scatter: MagicMock,
    mock_figure: MagicMock,
) -> None:
    """Create one scatter trace per cluster segment."""
    mock_cmap.return_value = lambda _: (0.1, 0.2, 0.3, 1.0)
    mock_figure.return_value = MagicMock()
    segments = [
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.array([[0.0, 1.0, 0.0]]),
    ]
    fig = plot_point_cloud_clusters(segments)
    assert mock_scatter.call_count == 2
    assert fig is mock_figure.return_value


# ---------------------------------------------------------------------------
# create_point_cloud_trace / create_clean_3d_layout
# ---------------------------------------------------------------------------


@patch("xwhy.plots.point_cloud.go.Scatter3d")
def test_create_point_cloud_trace(mock_scatter: MagicMock) -> None:
    """Build a Scatter3d trace with the supplied coordinates and color."""
    mock_scatter.return_value = MagicMock()
    xs = np.array([0.0, 1.0])
    ys = np.array([0.0, 1.0])
    zs = np.array([0.0, 1.0])
    trace = create_point_cloud_trace(xs, ys, zs, color="red", name="test")
    mock_scatter.assert_called_once()
    assert trace is mock_scatter.return_value


@patch("xwhy.plots.point_cloud.go.Layout")
def test_create_clean_3d_layout(mock_layout: MagicMock) -> None:
    """Build a minimal layout with the given title."""
    mock_layout.return_value = MagicMock()
    layout = create_clean_3d_layout(title="My Plot")
    mock_layout.assert_called_once()
    assert layout is mock_layout.return_value
    call_kwargs = mock_layout.call_args.kwargs
    assert call_kwargs["title"] == "My Plot"


# ---------------------------------------------------------------------------
# plot_point_cloud - main entry point
# ---------------------------------------------------------------------------


@patch("xwhy.plots.point_cloud.display_plotly_figure")
@patch("xwhy.plots.point_cloud.go.Figure")
@patch("xwhy.plots.point_cloud.create_point_cloud_trace")
@patch("xwhy.plots.point_cloud.create_clean_3d_layout")
def test_plot_point_cloud_show_true_returns_none(
    mock_layout: MagicMock,
    mock_trace: MagicMock,
    mock_figure: MagicMock,
    mock_display: MagicMock,
    sample_result: PointCloudXWhyResult,
) -> None:
    """Display the figure and return None when show is True."""
    mock_figure.return_value = MagicMock()
    result = plot_point_cloud(sample_result, show=True)
    mock_display.assert_called_once()
    assert result is None


@patch("xwhy.plots.point_cloud.display_plotly_figure")
@patch("xwhy.plots.point_cloud.go.Figure")
@patch("xwhy.plots.point_cloud.create_point_cloud_trace")
@patch("xwhy.plots.point_cloud.create_clean_3d_layout")
def test_plot_point_cloud_show_false_returns_figure(
    mock_layout: MagicMock,
    mock_trace: MagicMock,
    mock_figure: MagicMock,
    mock_display: MagicMock,
    sample_result: PointCloudXWhyResult,
) -> None:
    """Return the figure object when show is False."""
    fig_instance = MagicMock()
    mock_figure.return_value = fig_instance
    result = plot_point_cloud(sample_result, show=False)
    mock_display.assert_not_called()
    assert result is fig_instance


@patch("xwhy.plots.point_cloud.display_plotly_figure")
@patch("xwhy.plots.point_cloud.go.Figure")
@patch("xwhy.plots.point_cloud.create_point_cloud_trace")
@patch("xwhy.plots.point_cloud.create_clean_3d_layout")
def test_plot_point_cloud_save_html(
    mock_layout: MagicMock,
    mock_trace: MagicMock,
    mock_figure: MagicMock,
    mock_display: MagicMock,
    sample_result: PointCloudXWhyResult,
    tmp_path: Path,
) -> None:
    """Write an HTML file when save_path ends with .html."""
    fig_instance = MagicMock()
    mock_figure.return_value = fig_instance
    html_path = tmp_path / "plot.html"

    plot_point_cloud(sample_result, save_path=html_path, show=False)

    fig_instance.write_html.assert_called_once_with(
        str(html_path), include_plotlyjs="cdn"
    )
    fig_instance.write_image.assert_not_called()


@patch("xwhy.plots.point_cloud.display_plotly_figure")
@patch("xwhy.plots.point_cloud.go.Figure")
@patch("xwhy.plots.point_cloud.create_point_cloud_trace")
@patch("xwhy.plots.point_cloud.create_clean_3d_layout")
def test_plot_point_cloud_save_image(
    mock_layout: MagicMock,
    mock_trace: MagicMock,
    mock_figure: MagicMock,
    mock_display: MagicMock,
    sample_result: PointCloudXWhyResult,
    tmp_path: Path,
) -> None:
    """Write an image file when save_path does not end with .html."""
    fig_instance = MagicMock()
    mock_figure.return_value = fig_instance
    img_path = tmp_path / "plot.png"

    plot_point_cloud(sample_result, save_path=img_path, show=False)

    fig_instance.write_image.assert_called_once_with(str(img_path))
    fig_instance.write_html.assert_not_called()


@patch("xwhy.plots.point_cloud.display_plotly_figure")
@patch("xwhy.plots.point_cloud.go.Figure")
@patch("xwhy.plots.point_cloud.create_point_cloud_trace")
@patch("xwhy.plots.point_cloud.create_clean_3d_layout")
def test_plot_point_cloud_highlights_important_clusters(
    mock_layout: MagicMock,
    mock_trace: MagicMock,
    mock_figure: MagicMock,
    mock_display: MagicMock,
    sample_result: PointCloudXWhyResult,
) -> None:
    """Colour important clusters with the highlight colour."""
    mock_figure.return_value = MagicMock()
    plot_point_cloud(
        sample_result,
        base_color="blue",
        highlight_color="red",
        show=False,
    )

    # The colour array passed to create_point_cloud_trace must contain
    # "red" for points belonging to important clusters 0 and 2.
    call_kwargs = mock_trace.call_args.kwargs
    colors = call_kwargs["color"]
    # points 0,1 => cluster 0 (important) => red
    # point 2 => cluster 1 (not important) => blue
    # point 3 => cluster 2 (important) => red
    assert colors[0] == "red"
    assert colors[1] == "red"
    assert colors[2] == "blue"
    assert colors[3] == "red"
