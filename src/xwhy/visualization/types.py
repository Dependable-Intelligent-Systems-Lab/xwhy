"""Visualization types definition."""

from enum import StrEnum


class TextVisualizerType(StrEnum):
    """Enumeration for supported text visualization backends."""

    NATIVE_HEATMAP = "native_heatmap"
    SHAP = "shap"  # Reserved for future integration
