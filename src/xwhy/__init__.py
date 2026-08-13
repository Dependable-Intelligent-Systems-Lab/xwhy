"""Core abstractions for xwhy."""

from xwhy.config import settings  # noqa: I001

import xwhy.logger  # noqa: F401
from xwhy import plots
from xwhy.bootstrap import register_all
from xwhy.explainers.image import (
    ImageClassificationExplainer,
    ImageGenerationAndEditingExplainer,
)
from xwhy.explainers.llm import LLMExplainer
from xwhy.explainers.pointcloud import PointCloudExplainer
from xwhy.explainers.tabular import TabularExplainer
from xwhy.explainers.text import TextExplainer

register_all()

__all__ = [
    "ImageClassificationExplainer",
    "ImageGenerationAndEditingExplainer",
    "LLMExplainer",
    "PointCloudExplainer",
    "TabularExplainer",
    "TextExplainer",
    "plots",
    "settings",
]
