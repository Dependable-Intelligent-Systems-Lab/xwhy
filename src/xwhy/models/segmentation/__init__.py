"""Segmentation module public API.

Registers and exposes available segmentation model implementations.
"""

from xwhy.models.segmentation.base import BaseSegmentation
from xwhy.models.segmentation.factory import SegmentationFactory
from xwhy.models.segmentation.torchvision_models import TorchvisionSegmentation
from xwhy.models.segmentation.types import SegmentationType

__all__ = [
    "BaseSegmentation",
    "SegmentationFactory",
    "SegmentationType",
    "TorchvisionSegmentation",
]
