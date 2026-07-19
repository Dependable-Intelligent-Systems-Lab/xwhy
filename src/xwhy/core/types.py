"""Type aliases."""

from __future__ import annotations

from torch import device
from torch.nn import Module
from torchvision.models._api import WeightsEnum

from xwhy.models.embeddings.base import BaseEmbedding


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

        self.classification_weights: WeightsEnum | None = None
        self.classification_model: Module | None = None

        self.segmentation_weights: WeightsEnum | None = None
        self.segmentation_model: Module | None = None
        self.segmentation_class_names: list[str] = []

        self.embedding_model: BaseEmbedding | None = None
