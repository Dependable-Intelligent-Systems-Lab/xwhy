"""Plot types definition."""

from enum import StrEnum


class TextPlotterType(StrEnum):
    """Enumeration for supported text plot backends."""

    NATIVE_HEATMAP = "native_heatmap"
