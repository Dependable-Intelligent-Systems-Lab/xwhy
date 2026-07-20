"""Type aliases."""

from __future__ import annotations

from torch import device

from xwhy.models.classification.base import BaseClassification
from xwhy.models.embeddings.base import BaseEmbedding
from xwhy.models.segmentation.base import BaseSegmentation
from xwhy.perturbation.image import ImagePerturbation


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
        self.transform_fn: BaseClassification | None = None

        self.segmentation_model: BaseSegmentation | None = None

        self.embedding_model: BaseEmbedding | None = None
