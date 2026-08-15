"""Type aliases."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch

from xwhy.models.classification.base import BaseClassification
from xwhy.models.embeddings.base import BaseEmbedding
from xwhy.models.segmentation.base import BaseSegmentation
from xwhy.perturbation.image import ImagePerturbation
from xwhy.perturbation.text import TextPerturbation
from xwhy.providers.base import BaseProvider


class BaseImageGenerationAndEditing(ABC):
    """Abstract base class for all image generation and editing engines.

    Any cloud provider (e.g., OpenAI, Gemini) or custom local model
    used for image generation/editing must inherit from this class
    and implement its methods.
    """

    @abstractmethod
    def generate_image(
        self,
        prompt: str,
        output_dir: str,
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[bool, str]:
        """Generate an image from a text prompt.

        Args:
            prompt: The text prompt describing the desired image.
            output_dir: Directory to save the generated image.
            **kwargs: Additional parameters specific to the underlying model/API.

        Returns:
            A tuple containing a boolean success flag and the path to the
            generated image (or error message if failed).

        Raises:
            NotImplementedError: Implemented by subclasses.

        """
        raise NotImplementedError

    @abstractmethod
    def edit_image(
        self,
        prompt: str,
        image_path: str,
        output_dir: str,
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[bool, str]:
        """Edit an existing image based on a text prompt.

        Args:
            prompt: The text prompt describing the desired edits.
            image_path: Path to the original input image.
            output_dir: Directory to save the edited image.
            **kwargs: Additional parameters specific to the underlying model/API.

        Returns:
            A tuple containing a boolean success flag and the path to the
            edited image (or error message if failed).

        Raises:
            NotImplementedError: Implemented by subclasses.

        """
        raise NotImplementedError


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

    def __init__(self, device_: torch.device) -> None:
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


class ImageGenerationAndEditingState:
    """Runtime state for the Image Generation and Editing explainer."""

    def __init__(self, device_: torch.device) -> None:
        """Initialize the runtime state.

        This object stores runtime resources that are created during the
        explainer lifecycle. Unlike the configuration, these values are
        mutable and are populated as models and providers are initialized.

        Args:
            device_: Torch device used to load and run all models.

        """
        self.device = device_

        # Unified Generation/Editing Resource
        self.engine: BaseImageGenerationAndEditing | None = None

        # Explainability Resources
        self.text_perturbator: TextPerturbation | None = None
        self.segmentation_model: BaseSegmentation | None = None
        self.image_embedding_model: BaseEmbedding | None = None
        self.text_embedding_model: BaseEmbedding | None = None


class TextState:
    """Runtime state for the Text explainer."""

    def __init__(self) -> None:
        """Initialize the runtime state.

        This object stores runtime resources created during the explainer
        lifecycle, including models, prediction callables, perturbators, and
        embeddings.
        """
        self.model: Any = None
        self.predict_fn: Callable[[Sequence[str]], np.ndarray] | None = None
        self.perturbator: TextPerturbation | None = None
        self.embedding_model: BaseEmbedding | None = None
