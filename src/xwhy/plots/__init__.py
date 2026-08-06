"""Visualization plotting functions for XWhy results."""

from xwhy.plots.base import BaseTextPlotter
from xwhy.plots.factory import TextPlotterFactory
from xwhy.plots.image import image_heatmap, plot_image
from xwhy.plots.plots import (
    bar,
    beeswarm,
    decision,
    embedding,
    force,
    group_difference,
    heatmap,
    image,
    image_to_text,
    initjs,
    monitoring,
    partial_dependence,
    scatter,
    text,
    text_heatmap,
    violin,
    waterfall,
)
from xwhy.plots.tabular import (
    plot_dataset,
    plot_explanation_waterfall,
    plot_feature_contributions,
    plot_method_contributions,
)
from xwhy.plots.text import NativeHeatmapPlotter
from xwhy.plots.types import TextPlotterType
from xwhy.plots.visualisation import Explanation

__all__ = [
    "BaseTextPlotter",
    "Explanation",
    "NativeHeatmapPlotter",
    "TextPlotterFactory",
    "TextPlotterType",
    "bar",
    "beeswarm",
    "decision",
    "embedding",
    "force",
    "group_difference",
    "heatmap",
    "image",
    "image_heatmap",
    "image_to_text",
    "initjs",
    "monitoring",
    "partial_dependence",
    "plot_dataset",
    "plot_explanation_waterfall",
    "plot_feature_contributions",
    "plot_image",
    "plot_method_contributions",
    "scatter",
    "text",
    "text_heatmap",
    "violin",
    "waterfall",
]
