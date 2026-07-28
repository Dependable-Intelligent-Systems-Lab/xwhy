"""Unit tests for tabular plotting utilities."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from xwhy.plots.tabular import (
    plot_dataset,
    plot_explanation_waterfall,
    plot_feature_contributions,
    plot_method_contributions,
    set_plot_style,
)


@pytest.fixture
def mock_result() -> MagicMock:
    """Create a basic mock TabularXWhyResult for testing plotting functions."""
    mock = MagicMock()
    mock.coefficients = np.array([0.5, -0.3])
    mock.feature_list = ["Feature_A", "Feature_B"]
    mock.raw_data = {}
    return mock


@patch("xwhy.plots.tabular.plt")
def test_set_plot_style(mock_plt: MagicMock) -> None:
    """Verify plot style correctly configures axes limits and labels."""
    set_plot_style()
    mock_plt.axis.assert_called_once_with((-2, 2, -2, 2))
    mock_plt.xlabel.assert_called_once_with("x1")
    mock_plt.ylabel.assert_called_once_with("x2")


@patch("xwhy.plots.tabular.plt")
def test_plot_dataset_single_point(mock_plt: MagicMock) -> None:
    """Verify plot_dataset handles a single 2D point array correctly."""
    x = np.array([1.0, 2.0])
    plot_dataset(x, show=True)

    mock_plt.scatter.assert_called_once_with(1.0, 2.0)
    mock_plt.show.assert_called_once()


@patch("xwhy.plots.tabular.plt")
def test_plot_dataset_2d_without_y(mock_plt: MagicMock) -> None:
    """Verify plot_dataset handles 2D array without labels."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    plot_dataset(x, show=False)

    mock_plt.scatter.assert_called_once()
    mock_plt.show.assert_not_called()


@patch("xwhy.plots.tabular.plt")
def test_plot_dataset_2d_with_y(mock_plt: MagicMock) -> None:
    """Verify plot_dataset correctly maps color labels when y is provided."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0, 1])
    plot_dataset(x, y=y, scatter_kwargs={"alpha": 0.5}, show=False)

    mock_plt.scatter.assert_called_once()
    _, kwargs = mock_plt.scatter.call_args
    assert "c" in kwargs
    assert kwargs["alpha"] == 0.5


def test_plot_dataset_invalid_shape() -> None:
    """Ensure ValueError is raised for unsupported data array shapes."""
    x = np.array([1.0, 2.0, 3.0])  # Invalid shape (3,)
    with pytest.raises(ValueError, match="x must be either shape"):
        plot_dataset(x)


@patch("xwhy.plots.tabular.plt")
def test_plot_dataset_extra_point_branches(mock_plt: MagicMock) -> None:
    """Verify extra point plotting handles default and custom styles gracefully."""
    x = np.array([1.0, 2.0])
    point = np.array([0.0, 0.0])

    # Test with default point style
    plot_dataset(x, point=point, show=False)
    _, kwargs1 = mock_plt.scatter.call_args_list[1]
    assert kwargs1["c"] == "blue"  # default behavior

    # Test with custom point style
    style = {"c": "red", "marker": "x"}
    plot_dataset(x, point=point, point_style=style, show=False)
    _, kwargs2 = mock_plt.scatter.call_args_list[3]
    assert kwargs2["c"] == "red"
    assert kwargs2["marker"] == "x"


@patch("xwhy.plots.tabular.plt")
def test_plot_dataset_save_path(mock_plt: MagicMock) -> None:
    """Verify plot is saved to the correct path when save_path is provided."""
    x = np.array([1.0, 2.0])
    plot_dataset(x, save_path="test.png")

    mock_plt.savefig.assert_called_once_with("test.png", bbox_inches="tight")
    mock_plt.show.assert_not_called()


@patch("xwhy.plots.tabular.px")
def test_plot_feature_contributions_show(
    mock_px: MagicMock, mock_result: MagicMock
) -> None:
    """Verify feature contributions plot delegates to show when no path is given."""
    plot_feature_contributions(mock_result, title="Test")

    mock_px.bar.assert_called_once()
    mock_px.bar.return_value.show.assert_called_once()


@patch("xwhy.plots.tabular.px")
def test_plot_feature_contributions_save_html(
    mock_px: MagicMock, mock_result: MagicMock
) -> None:
    """Verify feature contributions plot generates names and saves as HTML."""
    mock_result.feature_list = []  # Trigger auto-naming branch
    plot_feature_contributions(mock_result, save_path="out.html")

    mock_px.bar.assert_called_once()
    mock_px.bar.return_value.write_html.assert_called_once_with("out.html")


@patch("xwhy.plots.tabular.px")
def test_plot_feature_contributions_save_image(
    mock_px: MagicMock, mock_result: MagicMock
) -> None:
    """Verify feature contributions plot saves as an image format."""
    plot_feature_contributions(mock_result, save_path=Path("out.png"))

    mock_px.bar.return_value.write_image.assert_called_once_with("out.png")


@patch("xwhy.plots.tabular.px")
def test_plot_method_contributions_explicit_method(
    mock_px: MagicMock, mock_result: MagicMock
) -> None:
    """Verify method contributions plot uses the explicitly provided method name."""
    plot_method_contributions(mock_result, method_name="SHAP")

    _, kwargs = mock_px.bar.call_args
    assert "SHAP" in kwargs["title"]
    mock_px.bar.return_value.show.assert_called_once()


@patch("xwhy.plots.tabular.px")
def test_plot_method_contributions_raw_data_method(
    mock_px: MagicMock, mock_result: MagicMock
) -> None:
    """Verify method contributions plot extracts the method name from raw_data."""

    class MockEnum:
        value = "LIME"

    mock_result.raw_data = {"surrogate_method": MockEnum()}
    plot_method_contributions(mock_result)

    _, kwargs = mock_px.bar.call_args
    assert "LIME" in kwargs["title"]


@patch("xwhy.plots.tabular.px")
def test_plot_method_contributions_default_and_auto_features(
    mock_px: MagicMock, mock_result: MagicMock
) -> None:
    """Verify method contributions plot falls back to SMILE and auto-features."""
    mock_result.feature_list = []
    plot_method_contributions(mock_result, save_path="test.html")

    _, kwargs = mock_px.bar.call_args
    assert "SMILE" in kwargs["title"]
    mock_px.bar.return_value.write_html.assert_called_once_with("test.html")


@patch("xwhy.plots.tabular.px")
def test_plot_method_contributions_save_image(
    mock_px: MagicMock, mock_result: MagicMock
) -> None:
    """Verify method contributions plot saves as a static image."""
    plot_method_contributions(mock_result, save_path="test.jpg")
    mock_px.bar.return_value.write_image.assert_called_once_with("test.jpg")


@patch("xwhy.plots.tabular.go")
def test_plot_explanation_waterfall_show(
    mock_go: MagicMock, mock_result: MagicMock
) -> None:
    """Verify waterfall plot creates a figure object and displays it."""
    plot_explanation_waterfall(mock_result)

    mock_go.Figure.assert_called_once()
    mock_go.Figure.return_value.show.assert_called_once()


@patch("xwhy.plots.tabular.go")
def test_plot_explanation_waterfall_save_html_and_auto_features(
    mock_go: MagicMock, mock_result: MagicMock
) -> None:
    """Verify waterfall plot generates fallback feature names and saves to HTML."""
    mock_result.feature_list = []
    plot_explanation_waterfall(mock_result, save_path="waterfall.html")

    mock_go.Figure.return_value.write_html.assert_called_once_with("waterfall.html")


@patch("xwhy.plots.tabular.go")
def test_plot_explanation_waterfall_save_image(
    mock_go: MagicMock, mock_result: MagicMock
) -> None:
    """Verify waterfall plot processes Path objects and saves as an image."""
    plot_explanation_waterfall(mock_result, save_path=Path("waterfall.pdf"))

    mock_go.Figure.return_value.write_image.assert_called_once_with("waterfall.pdf")
