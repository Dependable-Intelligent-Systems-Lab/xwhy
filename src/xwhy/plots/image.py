"""Plotting utilities for images."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from xwhy.utils.image import denormalize_tensor


def _prepare_image_for_display(
    img: Any,  # noqa: ANN401
    denormalize: bool = False,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
) -> np.ndarray:
    """Prepare an image (Tensor, Numpy, PIL) for matplotlib display.

    Args:
        img: Input image (torch.Tensor, np.ndarray, or PIL.Image).
        denormalize: Whether to apply denormalization.
        mean: Sequence of mean values (required if denormalize=True).
        std: Sequence of std values (required if denormalize=True).

    Returns:
        np.ndarray: Prepared image array (H, W, C) in [0, 1] range.

    Raises:
        ValueError: If denormalize is True but mean or std are missing.
        TypeError: If the image type is unsupported.

    """
    # -----------------------------
    # Case 0: numpy array and need denormalization
    # -----------------------------
    if isinstance(img, np.ndarray) and denormalize:
        if mean is None or std is None:
            raise ValueError("mean and std must be provided to denormalize.")
        tensor_chw = torch.from_numpy(img).permute(2, 0, 1).float()
        tensor_bchw = tensor_chw.unsqueeze(0)

        # Denormalize the marked image tensor
        img = denormalize_tensor(tensor_bchw, mean, std)

    # -----------------------------
    # Case 1: PyTorch Tensor
    # -----------------------------
    if torch.is_tensor(img):
        img_tensor = img.detach().cpu()

        # Remove batch dimension (1,3,H,W)
        if img_tensor.ndim == 4 and img_tensor.shape[0] == 1:
            img_tensor = img_tensor[0]

        # Convert CXHXW => HxWxC
        if img_tensor.ndim == 3 and img_tensor.shape[0] == 3:
            img_tensor = img_tensor.permute(1, 2, 0)

        img_np = img_tensor.numpy()

    # -----------------------------
    # Case 2: PIL Image => convert to numpy
    # -----------------------------
    elif isinstance(img, Image.Image):
        img_np = np.array(img).astype(np.float32) / 255.0

    # -----------------------------
    # Case 3: already numpy array
    # -----------------------------
    elif isinstance(img, np.ndarray):
        img_np = img.astype(np.float32)

        # If uint8 image, convert to [0,1]
        if img_np.max() > 1.0:
            img_np = img_np / 255.0

    else:
        raise TypeError(f"Unsupported image type: {type(img)}")

    # -----------------------------
    # Fix normalized images: [-1,1] => [0,1]
    # -----------------------------
    if img_np.min() < 0:
        img_np = (img_np + 1.0) / 2.0

    # Clip to ensure valid range
    clipped_array = np.clip(img_np, 0, 1)

    return cast(np.ndarray, clipped_array)


def plot_image(
    img: Any,  # noqa: ANN401
    title: str | None = None,
    denormalize: bool = False,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    save_path: str | Path | None = None,
) -> None:
    """Display or save an image (Tensor, Numpy, PIL).

    Args:
        img: Input image. Can be a torch.Tensor, numpy.ndarray, or PIL.Image.
        title: Optional title to show above the image.
        denormalize: Whether to apply denormalization prior to display.
        mean: Sequence of mean values (required if denormalize=True).
        std: Sequence of std values (required if denormalize=True).
        save_path: Path to save the plot. If None, plt.show() is called.

    """
    prepared_img = _prepare_image_for_display(
        img,
        denormalize=denormalize,
        mean=mean,
        std=std,
    )

    plt.figure(figsize=(8, 6))
    plt.imshow(prepared_img)
    if title:
        plt.title(title)
    plt.axis("off")

    if save_path:
        plt.savefig(str(save_path), bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def create_image_heat_mask(
    superpixels: np.ndarray,
    coeffs: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Create a heatmap mask from superpixels and importance coefficients.

    Args:
        superpixels: Superpixel segmentation mask (2D array).
        coeffs: Importance coefficients corresponding to each superpixel.

    Returns:
        np.ndarray: Heatmap mask array matching the superpixels shape.

    """
    # Initialize an empty image
    heat_mask = np.zeros_like(superpixels, dtype=float)

    # Iterate over the unique labels of the superpixels
    for idx, label in enumerate(np.unique(superpixels)):
        # Set the pixels of the current superpixel to its corresponding coefficient
        heat_mask[superpixels == label] = coeffs[idx]

    return heat_mask


def plot_image_heatmap(
    superpixels: np.ndarray,
    coeffs: Sequence[float] | np.ndarray,
    title: str = "Heatmap of Coefficients",
    save_path: str | Path | None = None,
) -> np.ndarray:
    """Plot a heatmap of feature importance over image superpixels.

    Args:
        superpixels: Superpixel segmentation mask of the image.
        coeffs: Importance coefficients for each superpixel.
        title: Title for the heatmap plot.
        save_path: Path to save the plot. If None, plt.show() is called.

    Returns:
        np.ndarray: The generated heatmap array.

    """
    heat_mask = create_image_heat_mask(superpixels, coeffs)

    plt.figure(figsize=(8, 6))
    plt.imshow(heat_mask, cmap="plasma", interpolation="nearest")
    plt.colorbar()
    plt.title(title)
    plt.axis("off")

    if save_path:
        plt.savefig(str(save_path), bbox_inches="tight")
    else:
        plt.show()

    plt.close()

    return heat_mask
