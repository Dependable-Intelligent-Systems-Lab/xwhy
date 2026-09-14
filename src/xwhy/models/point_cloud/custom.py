"""Custom model wrapper for point cloud models."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from xwhy.models.point_cloud.base import BasePointCloudModel


class CustomPointCloudModel(BasePointCloudModel):
    """Wrap user-defined custom PyTorch point cloud models or callables."""

    def __init__(
        self,
        model: torch.nn.Module | None = None,
        predict_fn: Callable[..., Any] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize custom point cloud model wrapper.

        Args:
            model: PyTorch module instance.
            predict_fn: Optional user custom inference function.
            **kwargs: Extra arguments for execution.

        Raises:
            ValueError: If neither model nor predict_fn is provided.

        """
        if model is None and predict_fn is None:
            raise ValueError("Either 'model' or 'predict_fn' must be provided.")

        self.model = model
        self.predict_fn = predict_fn
        self.kwargs = kwargs

    def predict(
        self,
        sample_input: torch.Tensor,
        sample_label: int | None = None,
    ) -> tuple[int, torch.Tensor, list[int]]:
        """Perform inference on a single point cloud tensor.

        Args:
            sample_input: Input point cloud tensor.
            sample_label: Optional label.

        Returns:
            Tuple of predicted class index, output probabilities/logits, top classes.

        Raises:
            RuntimeError: If model is missing when needed.

        """
        if self.predict_fn is not None:
            res: tuple[int, torch.Tensor, list[int]] = self.predict_fn(
                sample_input=sample_input,
                sample_label=sample_label,
                model=self.model,
                **self.kwargs,
            )
            return res

        if self.model is None:
            raise RuntimeError("Underlying PyTorch model is missing.")

        if sample_input.ndim == 2:
            input_batch = sample_input.unsqueeze(0).float()
        else:
            input_batch = sample_input.float()

        self.model.eval()
        with torch.no_grad():
            out = self.model(input_batch.transpose(1, 2))
            output = out[0] if isinstance(out, tuple) else out

            _, predicted_class = torch.max(output.data, 1)
            k_top = min(5, output.shape[1]) if output.ndim == 2 else 1
            _, top_indices = torch.topk(output.data, k_top, dim=1)
            top_classes = top_indices.cpu().numpy().flatten().tolist()

        return int(predicted_class.item()), output, top_classes

    def get_output_probabilities(
        self,
        samples: list[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """Get prediction probabilities for perturbed point cloud samples.

        Args:
            samples: List of point cloud tensors.
            device: PyTorch device.

        Returns:
            Tensor of output probabilities.

        Raises:
            RuntimeError: If model is missing.

        """
        if self.model is None and self.predict_fn is None:
            raise RuntimeError("Model or predict_fn is required for batch execution.")

        outputs: list[torch.Tensor] = []
        for sample in samples:
            _, logits, _ = self.predict(sample)

            # Apply softmax to convert logits to probabilities
            probs = torch.softmax(logits, dim=1)

            outputs.append(probs.to(device))

        return torch.cat(outputs, dim=0)
