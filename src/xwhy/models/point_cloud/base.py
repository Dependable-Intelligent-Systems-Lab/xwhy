"""Base point cloud model abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BasePointCloudModel(ABC):
    """Abstract base class for point cloud model wrappers."""

    @abstractmethod
    def predict(
        self,
        sample_input: torch.Tensor,
        sample_label: int | None = None,
    ) -> tuple[int, torch.Tensor, list[int]]:
        """Perform inference on point cloud sample.

        Args:
            sample_input: Input point cloud tensor of shape (N, 3) or (1, N, 3).
            sample_label: Optional ground truth class label.

        Returns:
            Tuple containing:
                - Predicted class index.
                - Raw model output probabilities or logits.
                - Top predicted class indices list.

        """
        raise NotImplementedError

    @abstractmethod
    def get_output_probabilities(
        self,
        samples: list[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """Batch inference to get output probabilities for perturbed samples.

        Args:
            samples: List of perturbed point cloud tensors.
            device: Computation torch device.

        Returns:
            Tensor of model output probabilities.

        """
        raise NotImplementedError
