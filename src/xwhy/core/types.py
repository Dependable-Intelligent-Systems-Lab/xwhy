"""Type aliases."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch import device

from xwhy.models.classification.base import BaseClassification
from xwhy.models.embeddings.base import BaseEmbedding
from xwhy.models.segmentation.base import BaseSegmentation
from xwhy.perturbation.image import ImagePerturbation
from xwhy.perturbation.text import TextPerturbation
from xwhy.providers.base import BaseProvider


class LLMState:
    """Runtime state for the LLM explainer."""

    def __init__(self) -> None:
        """Initialize the runtime state.

        This object stores runtime resources that are created during the
        explainer lifecycle. Unlike the configuration, these values are
        mutable and are populated as models and providers are initialized.
        """
        self.provider: BaseProvider | None = None
        self.perturbator: TextPerturbation | None = None
        self.embedding_model: BaseEmbedding | None = None


class ImageClassificationState:
    """Runtime state for the Image Classification explainer."""

    def __init__(self, device_: device) -> None:
        """Initialize the runtime state.

        This object stores runtime resources that are created during the
        explainer lifecycle. Unlike the configuration, these values are
        mutable and are populated as models are loaded.

        Args:
            device_: Torch device used to load and run all models.

        """
        self.device = device_
        self.perturbator: ImagePerturbation | None = None

        self.classification_model: BaseClassification | None = None
        self.transform_fn: Callable[..., Any] | None = None

        self.segmentation_model: BaseSegmentation | None = None

        self.embedding_model: BaseEmbedding | None = None


class TabularState:
    """Runtime state for the Tabular explainer."""

    def __init__(self) -> None:
        """Initialize the runtime state.

        This object stores the loaded predictive model to prevent redundant
        initializations across multiple explanation requests.
        """
        self.model: Any | None = None


class ImageGenerationState:
    """Runtime state for the Image Generation and Editing explainer."""

    def __init__(self, device_: device) -> None:
        """Initialize the runtime state.

        This object stores runtime resources that are created during the
        explainer lifecycle. Unlike the configuration, these values are
        mutable and are populated as models and providers are initialized.

        Args:
            device_: Torch device used to load and run all models.

        """
        self.device = device_

        # Generation/Editing Resources
        self.provider: BaseProvider | None = None
        self.generation_model: Any | None = None
        self.transform_fn: Callable[..., Any] | None = None

        # Explainability Resources
        self.perturbator: Any | None = (
            None  # Can hold TextPerturbation or ImagePerturbation
        )
        self.segmentation_model: BaseSegmentation | None = None
        self.embedding_model: BaseEmbedding | None = None
