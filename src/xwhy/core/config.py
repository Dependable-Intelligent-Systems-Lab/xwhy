"""Explainer config abstractions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from xwhy.distance.types import DistanceType
from xwhy.models.classification.types import ClassificationType
from xwhy.models.embeddings.types import EmbeddingType
from xwhy.models.segmentation.types import SegmentationType
from xwhy.providers.types import ProviderType
from xwhy.surrogate.types import SurrogateType


class ExplainerConfig(BaseModel):
    """Explainer config."""

    pass


class LLMConfig(ExplainerConfig):
    """Configuration for the LLM explainer."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    provider_type: ProviderType | str = ProviderType.OPENAI
    model_name: str = "gpt-3.5-turbo-instruct"
    max_tokens: int = Field(default=200, gt=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int = 1024
    num_perturbations: int = Field(default=64, gt=0)
    embedding_type: EmbeddingType | str = EmbeddingType.WORD2VEC
    surrogate_type: SurrogateType | str = SurrogateType.LIME
    use_best_surrogate: bool = True


class ImageClassificationConfig(ExplainerConfig):
    """Configuration for the Image Classification explainer."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    use_model_preprocess: bool = True

    use_embedding_model: bool = False
    embedding_type: EmbeddingType | str = EmbeddingType.DINOV2

    classification_type: ClassificationType | str = ClassificationType.INCEPTION_V3

    # Custom Model
    custom_model: Any = None
    custom_preprocess: Callable[..., Any] | None = None
    categories: Any = None

    use_segmentation_model: bool = True
    segmentation_type: SegmentationType | str = SegmentationType.DEEPLABV3_RESNET101
    device: str = "cpu"  # or "cuda"

    seed: int = 222

    kernel_size: int = Field(default=4, ge=1)
    max_dist: int = Field(default=200, gt=0)
    ratio: float = Field(default=0.2, gt=0.0, le=1.0)
    num_perturb: int = Field(default=150, gt=0)

    distance_type: DistanceType | str = DistanceType.WASSERSTEIN
    surrogate_type: SurrogateType | str = SurrogateType.LIME
    use_best_surrogate: bool = True

    num_top_features: int = Field(default=4, gt=0)
    num_top_predictions: int = Field(default=5, gt=0)
