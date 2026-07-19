"""Classification module public API.

Registers and exposes available classification model implementations.
"""

from xwhy.models.classification.base import BaseClassification
from xwhy.models.classification.factory import ClassificationFactory
from xwhy.models.classification.torchvision_models import TorchvisionClassification
from xwhy.models.classification.types import ClassificationType

__all__ = [
    "BaseClassification",
    "ClassificationFactory",
    "ClassificationType",
    "TorchvisionClassification",
]
