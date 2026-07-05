"""Unit tests for the visualization module."""

from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from xwhy.visualization.factory import TextVisualizerFactory
from xwhy.visualization.types import TextVisualizerType


@patch("xwhy.visualization.text.plt.show")
@patch("xwhy.visualization.text.plt.savefig")
def test_native_heatmap_visualizer(mock_savefig: object, mock_show: object) -> None:
    """Test standard execution of NativeHeatmapVisualizer."""
    visualizer = TextVisualizerFactory.create(TextVisualizerType.NATIVE_HEATMAP)
    words = ["This", "is", "a", "test"]
    scores = np.array([0.1, -0.5, 0.8, 0.0])

    visualizer.plot(words=words, scores=scores, title="Test Plot", verbose=1)

    assert mock_show.called  # type: ignore

    plt.close("all")


@patch("xwhy.visualization.text.plt.show")
@patch("xwhy.visualization.text.plt.savefig")
def test_native_heatmap_visualizer_save_path(
    mock_savefig: object, mock_show: object
) -> None:
    """Test saving functionality of NativeHeatmapVisualizer."""
    visualizer = TextVisualizerFactory.create(TextVisualizerType.NATIVE_HEATMAP)
    words = ["test"]
    scores = np.array([1.0])

    visualizer.plot(words=words, scores=scores, save_path="dummy.png", verbose=0)

    mock_savefig.assert_called_once_with("dummy.png", bbox_inches="tight")  # type: ignore
    assert mock_show.called  # type: ignore

    plt.close("all")


def test_visualizer_factory_invalid() -> None:
    """Test factory raises error on invalid input."""
    with pytest.raises(ValueError, match="Unsupported visualizer method"):
        TextVisualizerFactory.create("invalid_method")  # type: ignore
