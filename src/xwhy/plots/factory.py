"""Factory for creating text visualization."""

from typing import ClassVar

from xwhy.plots.base import BaseTextPlotter
from xwhy.plots.text import NativeHeatmapPlotter
from xwhy.plots.types import TextPlotterType


class TextPlotterFactory:
    """Factory for creating text visualization instances."""

    _registry: ClassVar[dict[TextPlotterType, type[BaseTextPlotter]]] = {
        TextPlotterType.NATIVE_HEATMAP: NativeHeatmapPlotter,
    }

    @classmethod
    def create(
        cls, method: TextPlotterType = TextPlotterType.NATIVE_HEATMAP
    ) -> BaseTextPlotter:
        """Create a text plotter instance.

        Args:
            method: The backend plotter to use.

        Returns:
            BaseTextPlotter: Instantiated plotter object.

        Raises:
            ValueError: If the requested plotter is not supported.

        """
        try:
            plotter_cls = cls._registry[method]
            return plotter_cls()
        except KeyError as exc:
            raise ValueError(f"Unsupported plotter method: {method}") from exc
