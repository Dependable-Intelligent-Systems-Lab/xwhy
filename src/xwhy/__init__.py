"""Core abstractions for xwhy."""

from xwhy import plots
from xwhy.bootstrap import register_all
from xwhy.explainers.image import ImageExplainer
from xwhy.explainers.llm import LLMExplainer
from xwhy.explainers.pix2pix import Pix2PixExplainer
from xwhy.explainers.pointcloud import PointCloudExplainer
from xwhy.explainers.tabular import TabularExplainer
from xwhy.explainers.text import TextExplainer

register_all()

__all__ = [
    "ImageExplainer",
    "LLMExplainer",
    "Pix2PixExplainer",
    "PointCloudExplainer",
    "TabularExplainer",
    "TextExplainer",
    "plots",
]
