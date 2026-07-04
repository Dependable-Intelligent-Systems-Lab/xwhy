"""Factory for creating text visualizers."""

from typing import ClassVar

from xwhy.visualization.base import BaseTextVisualizer
from xwhy.visualization.text import NativeHeatmapVisualizer
from xwhy.visualization.types import TextVisualizerType


class TextVisualizerFactory:
    """Factory for creating text visualization instances."""

    _registry: ClassVar[dict[TextVisualizerType, type[BaseTextVisualizer]]] = {
        TextVisualizerType.NATIVE_HEATMAP: NativeHeatmapVisualizer,
    }

    @classmethod
    def create(
        cls, method: TextVisualizerType = TextVisualizerType.NATIVE_HEATMAP
    ) -> BaseTextVisualizer:
        """Create a text visualizer instance.

        Args:
            method: The backend visualizer to use.

        Returns:
            BaseTextVisualizer: Instantiated visualizer object.

        Raises:
            ValueError: If the requested visualizer is not supported.

        """
        try:
            visualizer_cls = cls._registry[method]
            return visualizer_cls()
        except KeyError as exc:
            raise ValueError(f"Unsupported visualizer method: {method}") from exc
