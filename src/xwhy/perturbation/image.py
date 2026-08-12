"""Image perturbation strategy using Quickshift superpixels."""

from typing import Any, cast

import numpy as np
import skimage.segmentation
import torch

from xwhy.perturbation.base import BasePerturbation


class ImagePerturbation(BasePerturbation[np.ndarray, np.ndarray, np.ndarray]):
    """Perturbation strategy for images using superpixels and Bernoulli sampling."""

    def __init__(
        self,
        kernel_size: int = 4,
        max_dist: int = 200,
        ratio: float = 0.2,
        seed: int = 42,
    ) -> None:
        """Initialize the image perturbation strategy with default parameters."""
        self.kernel_size = kernel_size
        self.max_dist = max_dist
        self.ratio = ratio
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def set_seed(self, seed: int) -> None:
        """Update the random number generator with a new seed."""
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def generate_superpixels(
        self, image: torch.Tensor | np.ndarray
    ) -> tuple[np.ndarray, int]:
        """Generate superpixels using the Quickshift algorithm.

        Args:
            image: Input image in either CHW or HWC format.

        Returns:
            tuple[np.ndarray, int]: The superpixel label map (H, W) and the number
                of unique superpixel regions.

        Raises:
            ValueError: If a PyTorch tensor has an unexpected shape.
            TypeError: If the image is neither a Tensor nor an ndarray.

        """
        if isinstance(image, torch.Tensor):
            if image.ndim == 3 and image.shape[0] == 3:
                img_np = image.permute(1, 2, 0).cpu().numpy()
            elif image.ndim == 3 and image.shape[2] == 3:
                img_np = image.cpu().numpy()
            else:
                raise ValueError(f"Unexpected tensor shape: {image.shape}")
        elif isinstance(image, np.ndarray):
            img_np = image
        else:
            raise TypeError("image must be a torch.Tensor or np.ndarray")

        superpixels = skimage.segmentation.quickshift(
            img_np,
            kernel_size=self.kernel_size,
            max_dist=self.max_dist,
            ratio=self.ratio,
            rng=self.seed,
        )  # type: ignore[no-untyped-call]

        superpixels_np = cast(np.ndarray, superpixels)
        num_superpixels = int(np.unique(superpixels_np).shape[0])

        return superpixels_np, num_superpixels

    def generate(
        self,
        *args: Any,  # noqa: ANN401
        num_superpixels: int,
        num_perturbations: int = 64,
        keep_probability: float = 0.5,
        **kwargs: Any,  # noqa: ANN401
    ) -> np.ndarray:
        """Generate binary perturbation masks using Bernoulli sampling.

        Args:
            *args: Unused positional arguments.
            num_superpixels: Number of superpixel regions.
            num_perturbations: Number of perturbation samples to generate.
            keep_probability: Probability of keeping a superpixel (value = 1).
            **kwargs: Unused keyword arguments.

        Returns:
            np.ndarray: Binary matrix of shape (num_perturbations, num_superpixels).

        """
        masks = self._rng.binomial(
            n=1,
            p=keep_probability,
            size=(num_perturbations, num_superpixels),
        )
        return cast(np.ndarray, masks)

    def apply_mask(
        self,
        item: np.ndarray,
        mask: np.ndarray,
        *args: Any,  # noqa: ANN401
        segments: np.ndarray | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> np.ndarray:
        """Apply a binary perturbation mask to an image based on segments.

        Args:
            item: Input image array of shape (H, W, C).
            mask: Binary array indicating which superpixels to keep (1)
                or remove (0). Shape: (num_superpixels,).
            *args: Unused positional arguments.
            segments: Segmentation map labeling each pixel with its superpixel ID.
                Required parameter provided via kwargs.
            **kwargs: Unused keyword arguments.

        Returns:
            np.ndarray: Perturbed image where inactive superpixels are zeroed out.

        Raises:
            ValueError: If `segments` is not provided.

        """
        if segments is None and args:
            segments = args[0]

        if segments is None:
            raise ValueError(
                "segments (superpixel map) must be provided either "
                "as a positional or keyword argument."
            )

        active_pixels = np.where(mask == 1)[0]
        binary_mask = np.zeros_like(segments, dtype=float)

        for active in active_pixels:
            binary_mask[segments == active] = 1.0

        perturbed_image = item.copy()
        perturbed_image = perturbed_image * binary_mask[..., np.newaxis]

        return cast(np.ndarray, perturbed_image)
