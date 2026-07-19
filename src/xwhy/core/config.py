"""Explainer config abstractions."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from xwhy.models.classification.types import ClassificationType
from xwhy.models.embeddings.types import EmbeddingType
from xwhy.models.segmentation.types import SegmentationType
from xwhy.surrogate.types import SurrogateType


class ExplainerConfig(BaseModel):
    """Explainer config."""

    pass


class ImageClassificationConfig(ExplainerConfig):
    """Configuration for the Image Classification explainer."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    use_model_preprocess: bool = False
    need_normalization: bool = False

    use_embedding_model: bool = False
    embedding_type: EmbeddingType = EmbeddingType.DINOV2

    classification_type: ClassificationType = ClassificationType.INCEPTION_V3

    use_segmentation_model: bool = True
    segmentation_type: SegmentationType = SegmentationType.DEEPLABV3_RESNET101
    device: str = "cpu"  # or "cuda"

    seed: int = 222

    kernel_size: int = Field(default=4, ge=1)
    max_dist: int = Field(default=200, gt=0)
    ratio: float = Field(default=0.2, gt=0.0, le=1.0)
    num_perturb: int = Field(default=150, gt=0)

    distance_metric: str = "wasserstein"
    surrogate_type: SurrogateType = SurrogateType.LIME
    use_best_surrogate: bool = True

    num_top_features: int = Field(default=4, gt=0)
    num_top_predictions: int = Field(default=5, gt=0)
