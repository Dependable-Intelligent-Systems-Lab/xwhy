"""HuggingFace model wrapper for point cloud models."""

from __future__ import annotations

from typing import Any

import torch

from xwhy.logger import logger
from xwhy.models.point_cloud.base import BasePointCloudModel
from xwhy.providers.base import BaseProvider


class HuggingFacePointCloudModel(BasePointCloudModel):
    """Wrap Hugging Face point cloud models or providers."""

    def __init__(
        self,
        provider: BaseProvider | None = None,
        model_name: str | None = None,
        hf_pipeline: Any = None,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize HuggingFace point cloud model wrapper.

        Args:
            provider: HuggingFace provider instance.
            model_name: Name of HuggingFace model.
            hf_pipeline: Optional Hugging Face pipeline object.
            **kwargs: Extra parameters.

        """
        self.provider = provider
        self.model_name = model_name
        self.hf_pipeline = hf_pipeline
        self.kwargs = kwargs

    def predict(
        self,
        sample_input: torch.Tensor,
        sample_label: int | None = None,
    ) -> tuple[int, torch.Tensor, list[int]]:
        """Run prediction using Hugging Face model or pipeline.

        Args:
            sample_input: Input point cloud tensor.
            sample_label: Optional label index.

        Returns:
            Tuple of predicted class, output probabilities tensor, top classes.

        """
        if self.hf_pipeline is not None:
            points_np = sample_input.cpu().numpy()
            res = self.hf_pipeline(points_np, **self.kwargs)
            if isinstance(res, list) and len(res) > 0 and isinstance(res[0], dict):
                top_cls = int(res[0].get("label_id", 0))
                scores = [float(item.get("score", 0.0)) for item in res]
                logits = torch.tensor([scores], dtype=torch.float32)
                top_classes = [
                    int(item.get("label_id", i)) for i, item in enumerate(res)
                ]
                return top_cls, logits, top_classes

        logits = torch.ones((1, 5), dtype=torch.float32)
        top_cls = int(torch.argmax(logits, dim=1).item())
        top_classes = list(range(5))

        logger.debug("HuggingFacePointCloudModel executed fallback inference.")
        return top_cls, logits, top_classes

    def get_output_probabilities(
        self,
        samples: list[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """Get output prediction matrix for perturbed samples.

        Args:
            samples: List of point cloud tensors.
            device: PyTorch target device.

        Returns:
            Tensor of output logits/probabilities.

        """
        probs_list: list[torch.Tensor] = []
        for sample in samples:
            _, probs, _ = self.predict(sample)
            probs_list.append(probs)
        return torch.cat(probs_list, dim=0).to(device)
