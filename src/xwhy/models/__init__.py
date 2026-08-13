"""Model module public API.

Registers and exposes available model implementations.
"""

from xwhy.models.classification.base import BaseClassification
from xwhy.models.classification.factory import ClassificationFactory
from xwhy.models.classification.torchvision_models import TorchvisionClassification
from xwhy.models.classification.types import ClassificationType
from xwhy.models.embeddings.base import BaseEmbedding
from xwhy.models.embeddings.factory import EmbeddingFactory
from xwhy.models.embeddings.types import EmbeddingType
from xwhy.models.embeddings.word2vec import Word2VecEmbedding
from xwhy.models.segmentation.base import BaseSegmentation
from xwhy.models.segmentation.factory import SegmentationFactory
from xwhy.models.segmentation.torchvision_models import TorchvisionSegmentation
from xwhy.models.segmentation.types import SegmentationType
from xwhy.models.tabular.adapter import TabularModelAdapter

__all__ = [
    "BaseClassification",
    "BaseEmbedding",
    "BaseSegmentation",
    "ClassificationFactory",
    "ClassificationType",
    "EmbeddingFactory",
    "EmbeddingType",
    "SegmentationFactory",
    "SegmentationType",
    "TabularModelAdapter",
    "TorchvisionClassification",
    "TorchvisionSegmentation",
    "Word2VecEmbedding",
]
