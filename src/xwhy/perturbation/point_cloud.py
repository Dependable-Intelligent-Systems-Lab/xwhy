"""Point cloud perturbation strategy."""

from typing import Any, cast

import numpy as np
import torch

from xwhy.perturbation.base import BasePerturbation


class PointCloudPerturbation(BasePerturbation[torch.Tensor, np.ndarray, torch.Tensor]):
    """Perturbation strategy for point clouds using cluster removal."""

    def __init__(
        self,
        removal_probability: float = 0.5,
        seed: int = 42,
    ) -> None:
        """Initialize the point cloud perturbation strategy.

        Args:
            removal_probability: Probability of removing a cluster.
            seed: Random seed for reproducibility.

        """
        self.removal_probability = removal_probability
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def set_seed(self, seed: int) -> None:
        """Update the random number generator with a new seed."""
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def generate(
        self,
        *args: Any,  # noqa: ANN401
        num_clusters: int,
        num_perturbations: int = 50,
        **kwargs: Any,  # noqa: ANN401
    ) -> np.ndarray:
        """Generate binary perturbation masks using Bernoulli sampling.

        Args:
            *args: Unused positional arguments.
            num_clusters: Number of unique clusters in the point cloud.
            num_perturbations: Number of perturbation masks to generate.
            **kwargs: Unused keyword arguments.

        Returns:
            np.ndarray: Binary mask array of shape (num_perturbations, num_clusters).

        """
        masks = self._rng.binomial(
            n=1,
            p=1.0 - self.removal_probability,
            size=(num_perturbations, num_clusters),
        )
        return cast(np.ndarray, masks)

    def apply_mask(
        self,
        item: torch.Tensor,
        mask: np.ndarray,
        *args: Any,  # noqa: ANN401
        segments: np.ndarray | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> torch.Tensor:
        """Apply a cluster-level perturbation mask to a point cloud.

        Args:
            item: Input point cloud tensor of shape (N, 3).
            mask: Binary array indicating which clusters to keep (1) or remove (0).
            *args: Unused positional arguments.
            segments: Cluster labels array of shape (N,). Required.
            **kwargs: Unused keyword arguments.

        Returns:
            torch.Tensor: The perturbed point cloud containing only the kept points.

        Raises:
            ValueError: If `segments` is not provided.

        """
        if segments is None and args:
            segments = args[0]

        if segments is None:
            raise ValueError(
                "segments (cluster labels) must be provided either "
                "as a positional or keyword argument."
            )

        # Map cluster mask to each individual point
        point_mask = mask[segments]

        # Select indices where the mask is 1 (keep)
        indices_to_keep = np.where(point_mask == 1)[0]

        return item[indices_to_keep]
