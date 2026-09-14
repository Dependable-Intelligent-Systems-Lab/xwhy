"""Point cloud plotting utilities for visualization and explanations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from IPython.display import HTML, display

from xwhy.core.result import PointCloudXWhyResult


def create_rotation_frames(
    radius: float = 2.0,
    height: float = 0.8,
    num_steps: int = 100,
) -> list[dict[str, Any]]:
    """Generate camera rotation frames for 3D Plotly animations.

    Args:
        radius: Distance of the camera from the origin in the XY plane.
        height: Z-axis position of the camera.
        num_steps: Number of animation frames.

    Returns:
        A list of Plotly animation frame dictionaries for a rotating view.

    """
    angles = np.linspace(0, 2 * np.pi, num_steps)
    frames: list[dict[str, Any]] = []

    for angle in angles:
        frame = {
            "layout": {
                "scene": {
                    "camera": {
                        "eye": {
                            "x": radius * float(np.cos(angle)),
                            "y": radius * float(np.sin(angle)),
                            "z": height,
                        }
                    }
                }
            }
        }
        frames.append(frame)

    return frames


def plot_3d_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    opacity: float = 0.5,
) -> go.Figure:
    """Create a 3D mesh visualization with a rotation animation.

    Args:
        vertices: Array of vertices of shape (N, 3).
        faces: Array of faces of shape (F, 3).
        opacity: Transparency level of the mesh (0.0 to 1.0).

    Returns:
        A Plotly figure containing the animated mesh.

    """
    x, y, z = vertices.T
    i, j, k = faces.T

    mesh = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=i,
        j=j,
        k=k,
        opacity=opacity,
    )

    fig = go.Figure(
        data=[mesh],
        frames=create_rotation_frames(),
        layout={
            "updatemenus": [
                {
                    "type": "buttons",
                    "showactive": False,
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [None],
                        }
                    ],
                }
            ]
        },
    )

    return fig


def plot_3d_point_cloud(
    vertices: np.ndarray,
    marker_size: int = 2,
) -> go.Figure:
    """Create a 3D point cloud visualization with a rotation animation.

    Args:
        vertices: Array of point coordinates of shape (N, 3).
        marker_size: Size of each point marker.

    Returns:
        A Plotly figure containing the animated point cloud.

    """
    x, y, z = vertices.T

    scatter = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker={"size": marker_size},
    )

    fig = go.Figure(
        data=[scatter],
        frames=create_rotation_frames(),
    )

    return fig


def plot_colored_3d_point_cloud(
    vertices: np.ndarray,
    importance: np.ndarray,
    marker_size: int = 4,
    title: str = "Point Cloud",
    show_colorbar: bool = True,
) -> go.Figure:
    """Create a colored 3D point cloud visualization with rotation.

    Args:
        vertices: Array of point coordinates of shape (N, 3).
        importance: Array of importance values per point of shape (N,).
        marker_size: Size of each point marker.
        title: Title of the figure.
        show_colorbar: Whether to display a colorbar next to the plot.

    Returns:
        A Plotly figure containing the animated and colored point cloud.

    """
    x, y, z = vertices.T

    scatter = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        name="Point Cloud",
        marker={
            "size": marker_size,
            "color": importance,
            "colorscale": "Viridis",
            "colorbar": {"title": "Importance"} if show_colorbar else None,
            "line": {"width": 0},
        },
    )

    fig = go.Figure(
        data=[scatter],
        frames=create_rotation_frames(),
        layout={
            "title": title,
            "margin": {"l": 0, "r": 0, "b": 0, "t": 40},
            "scene": {
                "xaxis": {"title": "X"},
                "yaxis": {"title": "Y"},
                "zaxis": {"title": "Z"},
            },
            "updatemenus": [
                {
                    "type": "buttons",
                    "showactive": False,
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [None],
                        }
                    ],
                }
            ],
        },
    )

    return fig


def display_plotly_figure(fig: go.Figure) -> None:
    """Display a Plotly figure to persist after reopening a Jupyter notebook.

    This works by converting the plot to raw HTML and loading the Plotly JS
    library via CDN.

    Args:
        fig: The Plotly figure object to display.

    """
    html_content: str = fig.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        auto_play=False,
    )
    display(HTML(html_content))  # type: ignore[no-untyped-call]


def plot_point_cloud_clusters(segments: list[np.ndarray]) -> go.Figure:
    """Visualize clustered point cloud segments in 3D.

    Args:
        segments: A list of NumPy arrays, where each array represents
            a cluster of points of shape (N_i, 3).

    Returns:
        A Plotly figure containing the clustered point clouds.

    """
    plot_data: list[go.Scatter3d] = []

    for segment_idx, segment in enumerate(segments):
        x_vals, y_vals, z_vals = segment[:, 0], segment[:, 1], segment[:, 2]

        color = plt.get_cmap("tab20")(segment_idx % 20)
        color_rgba = (
            f"rgba({int(color[0] * 255)}, {int(color[1] * 255)}, "
            f"{int(color[2] * 255)}, 1.0)"
        )

        scatter = go.Scatter3d(
            x=x_vals,
            y=y_vals,
            z=z_vals,
            mode="markers",
            marker={
                "size": 2,
                "color": color_rgba,
            },
            name=f"Cluster {segment_idx}",
        )
        plot_data.append(scatter)

    return go.Figure(data=plot_data)


def create_point_cloud_trace(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    color: np.ndarray | str,
    name: str,
) -> go.Scatter3d:
    """Create a Plotly 3D scatter trace for point cloud visualization.

    Args:
        xs: X coordinates array of shape (N,).
        ys: Y coordinates array of shape (N,).
        zs: Z coordinates array of shape (N,).
        color: Color array of shape (N,) or a single color string.
        name: Name of the trace to display in the legend.

    Returns:
        A configured Plotly scatter trace object.

    """
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="markers",
        marker={
            "size": 2,
            "color": color,
            "line": {"width": 2},
        },
        name=name,
    )


def create_clean_3d_layout(title: str = "") -> go.Layout:
    """Create a minimal 3D Plotly layout without axes clutter.

    Args:
        title: The title string to display on the plot.

    Returns:
        A configured Plotly layout object.

    """
    return go.Layout(
        title=title,
        scene={
            "xaxis": {
                "title": "",
                "showticklabels": False,
                "showgrid": False,
                "showbackground": False,
            },
            "yaxis": {
                "title": "",
                "showticklabels": False,
                "showgrid": False,
                "showbackground": False,
            },
            "zaxis": {
                "title": "",
                "showticklabels": False,
                "showgrid": False,
                "showbackground": False,
            },
        },
    )


def plot_point_cloud(
    result: PointCloudXWhyResult,
    **kwargs: Any,  # noqa: ANN401
) -> go.Figure | None:
    """Visualize explanation over point cloud by highlighting important clusters.

    Extracts the required point cloud data, cluster labels, and importance
    scores from the explanation result and generates an interactive 3D plot.

    Args:
        result: Point cloud explanation result container.
        **kwargs: Additional plotting arguments including:
            base_color (str): Default color for unimportant points.
            highlight_color (str): Color for important clusters.
            title (str): Title of the plot.
            save_path (str | Path | None): File path to save the plot.
            show (bool): Whether to immediately display the figure.

    Returns:
        The generated Plotly figure, or None if only saved/shown.

    """
    points: np.ndarray = result.sample_points
    cluster_labels: np.ndarray = result.cluster_labels
    important_clusters: np.ndarray = result.important_clusters

    base_color: str = str(kwargs.pop("base_color", "blue"))
    highlight_color: str = str(kwargs.pop("highlight_color", "red"))
    title: str = str(kwargs.pop("title", "Explanation Point Cloud"))
    save_path: str | Path | None = kwargs.pop("save_path", None)
    show: bool = bool(kwargs.pop("show", True))

    colors = np.full(cluster_labels.shape, base_color, dtype=object)
    for cluster_id in important_clusters:
        colors[cluster_labels == cluster_id] = highlight_color
    color_array = np.asarray(colors, dtype=str)

    trace = create_point_cloud_trace(
        xs=points[:, 0],
        ys=points[:, 1],
        zs=points[:, 2],
        color=color_array,
        name="Explanation",
    )

    fig = go.Figure(
        data=[trace],
        layout=create_clean_3d_layout(title=title),
    )

    if save_path:
        path_str = str(save_path)
        if path_str.endswith(".html"):
            fig.write_html(path_str, include_plotlyjs="cdn")
        else:
            fig.write_image(path_str)

    if show:
        display_plotly_figure(fig)
        return None

    return fig
