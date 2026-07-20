"""Image utility functions for loading, preprocessing, and format conversion."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def get_default_transform() -> Callable[[Image.Image], torch.Tensor]:
    """Create the fallback preprocessing pipeline for images.

    Returns:
        Callable: A sequence of torchvision transforms.

    """
    pipeline = transforms.Compose(
        [
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    return cast(Callable[[Image.Image], torch.Tensor], pipeline)


def load_image_as_tensor(
    image_path: str | Path,
    transform_fn: Callable[[Image.Image], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, Image.Image]:
    """Load an image from disk and apply preprocessing transforms.

    Args:
        image_path: Path to the input image.
        transform_fn: Optional transform callable. Uses default if None.

    Returns:
        tuple[torch.Tensor, Image.Image]: Preprocessed tensor (1, C, H, W)
            and the original PIL image.

    """
    img = Image.open(image_path).convert("RGB")
    pipeline = transform_fn if transform_fn is not None else get_default_transform()

    processed_tensor = pipeline(img).unsqueeze(0)
    return processed_tensor, img


def numpy_image_to_tensor(
    np_array: np.ndarray,
    transform_fn: Callable[[Image.Image], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Preprocess a numpy image array using the specified transforms.

    Args:
        np_array: Numpy array of shape (H, W, C).
        transform_fn: Optional transform callable. Uses default if None.

    Returns:
        torch.Tensor: Preprocessed torch tensor of shape (1, C, H, W).

    """
    if np_array.dtype != np.uint8:
        img_uint8 = (np_array * 255).astype(np.uint8)
    else:
        img_uint8 = np_array

    img = Image.fromarray(img_uint8)
    pipeline = transform_fn if transform_fn is not None else get_default_transform()

    tensor = pipeline(img)
    return tensor.unsqueeze(0)


def denormalize_tensor(
    img_tensor: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    """Reverse the normalization applied to an image tensor.

    Args:
        img_tensor: Normalized image tensor (B, C, H, W).
        mean: Sequence of mean values used for normalization.
        std: Sequence of standard deviation values used.

    Returns:
        torch.Tensor: The denormalized image tensor.

    """
    # Clone the tensor to avoid modifying the original in-place
    denorm_img = img_tensor.clone()
    device = denorm_img.device

    # Reshape mean and std to be broadcastable (1, C, 1, 1)
    mean_tensor = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std_tensor = torch.tensor(std, device=device).view(1, -1, 1, 1)

    # Reverse normalization: (output * std) + mean
    return (denorm_img * std_tensor) + mean_tensor


def tensor_to_numpy_image(
    tensor_batch: torch.Tensor,
    denormalize: bool = False,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
) -> np.ndarray:
    """Convert a batch tensor (1, C, H, W) to a NumPy image (H, W, C).

    Args:
        tensor_batch: Input tensor batch.
        denormalize: If True, reverses normalization using mean and std.
        mean: Mean values for denormalization (required if denormalize=True).
        std: Standard deviation values (required if denormalize=True).

    Returns:
        np.ndarray: Denormalized image array (H, W, C) clipped to [0, 1].

    Raises:
        ValueError: If denormalize is True but mean or std are not provided.

    """
    img_tensor = tensor_batch.squeeze(0)

    if denormalize:
        if mean is None or std is None:
            raise ValueError("mean and std must be provided to denormalize.")

        # Add batch dim back for denormalization, then remove it
        img_batch = img_tensor.unsqueeze(0)
        img_tensor = denormalize_tensor(img_batch, mean, std).squeeze(0)

    # Convert (C, H, W) -> (H, W, C)
    img_np = img_tensor.permute(1, 2, 0).detach().cpu().numpy()

    # Clip to ensure valid range
    clipped_array = np.clip(img_np, 0, 1)

    return cast(np.ndarray, clipped_array)
