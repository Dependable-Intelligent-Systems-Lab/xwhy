"""Plotting utilities for images."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from skimage.color import label2rgb
from skimage.segmentation import mark_boundaries

from xwhy.core.result import ImageClassificationXWhyResult
from xwhy.utils.image import denormalize_tensor


def _prepare_image_for_display(
    img: Any,  # noqa: ANN401
    denormalize: bool = False,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
) -> np.ndarray:
    """Prepare an image (Tensor, Numpy, PIL, or Path) for matplotlib display.

    Args:
        img: Input image (torch.Tensor, np.ndarray, PIL.Image or Path).
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
    # Case 0: String or Path (Load image)
    # -----------------------------
    if isinstance(img, (str, Path)):
        img = Image.open(str(img)).convert("RGB")

    # -----------------------------
    # Case 1: numpy array and need denormalization
    # -----------------------------
    if isinstance(img, np.ndarray) and denormalize:
        if mean is None or std is None:
            raise ValueError("mean and std must be provided to denormalize.")
        tensor_chw = torch.from_numpy(img).permute(2, 0, 1).float()
        tensor_bchw = tensor_chw.unsqueeze(0)

        # Denormalize the marked image tensor
        img = denormalize_tensor(tensor_bchw, mean, std)

    # -----------------------------
    # Case 2: PyTorch Tensor
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
    # Case 3: PIL Image => convert to numpy
    # -----------------------------
    elif isinstance(img, Image.Image):
        img_np = np.array(img).astype(np.float32) / 255.0

    # -----------------------------
    # Case 4: already numpy array
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
    """Display an image (Tensor, Numpy, PIL, or Path).

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


def image_heatmap(
    result: ImageClassificationXWhyResult,
    **kwargs: Any,  # noqa: ANN401
) -> None:
    """Plot a heatmap of feature importance over image superpixels.

    Args:
        result: Text explanation result.
        **kwargs: Additional plotting arguments (e.g., title, save_path).

    """
    superpixels: np.ndarray = result.superpixels
    coeffs: Sequence[float] | np.ndarray = result.coefficients
    title: str = str(kwargs.pop("title", "Image Heatmap"))
    save_path: str | Path | None = kwargs.pop("save_path", None)

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


def get_image_and_mask(
    result: ImageClassificationXWhyResult,
    positive_only: bool = True,
    negative_only: bool = False,
    hide_rest: bool = False,
    num_features: int = 5,
    min_weight: float = 0.0,
    *,
    boost_channels: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the segmented image and feature mask from an explanation result.

    When ``boost_channels=True``, color channels are intensified for boundary
    marking (`mark_boundaries`). When ``boost_channels=False``, the clean image
    is preserved so region overlay colors (`label2rgb`) remain accurate.

    Args:
        result: The classification explanation result containing the original
            image, superpixel segments, and feature coefficients.
        positive_only: If True, only include superpixels that positively
            contribute to the prediction.
        negative_only: If True, only include superpixels that negatively
            contribute to the prediction.
        hide_rest: If True, zero out unallotted background superpixels.
        num_features: Maximum number of top superpixels to include.
        min_weight: Minimum absolute coefficient weight required for a
            superpixel to be included.
        boost_channels: If True, maximize the red or green channel intensity
            for highlighted negative or positive superpixels.

    Returns:
        A tuple of ``(image, mask)`` where ``image`` is the processed display
        array and ``mask`` is an integer array indicating highlighted regions.

    Raises:
        ValueError: If both ``positive_only`` and ``negative_only`` are True.

    """
    if positive_only and negative_only:
        raise ValueError(
            "positive_only and negative_only cannot be true at the same time."
        )

    segments = result.superpixels
    image = _prepare_image_for_display(result.original_image)

    if image.shape[:2] != segments.shape[:2]:
        import skimage.transform

        image = skimage.transform.resize(  # type: ignore[no-untyped-call]
            image,
            segments.shape[:2],
            preserve_range=True,
            anti_aliasing=True,
        ).astype(image.dtype)

    coeffs = np.asarray(result.coefficients)
    mask = np.zeros(segments.shape, dtype=np.int32)
    temp = np.zeros_like(image) if hide_rest else image.copy()

    exp = sorted(
        [(i, float(coeffs[i])) for i in range(len(coeffs))],
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    if positive_only:
        fs = [x[0] for x in exp if x[1] > 0 and x[1] > min_weight][:num_features]
        for f in fs:
            temp[segments == f] = image[segments == f]
            mask[segments == f] = 1
        return temp, mask

    if negative_only:
        fs = [x[0] for x in exp if x[1] < 0 and abs(x[1]) > min_weight][:num_features]
        for f in fs:
            temp[segments == f] = image[segments == f]
            mask[segments == f] = 1
        return temp, mask

    # both positive and negative
    for f, w in exp[:num_features]:
        if abs(w) < min_weight:
            continue

        temp[segments == f] = image[segments == f]

        if boost_channels:  # Only for mark_boundaries
            c = 0 if w < 0 else 1
            temp[segments == f, c] = np.max(image)

        mask[segments == f] = 1 if w > 0 else 2

    return temp, mask


def image_boundaries(
    result: ImageClassificationXWhyResult,
    positive_only: bool = True,
    hide_rest: bool = True,
    num_features: int = 5,
    min_weight: float = 0.0,
    title: str | None = None,
    save_path: str | Path | None = None,
    boost_channels: bool = True,
) -> None:
    """Plot the image superpixel boundaries for the most important features.

    Args:
        result: The classification explanation result.
        positive_only: If True, only plot positive superpixels.
        hide_rest: If True, hide the rest of the image.
        num_features: Number of superpixels to include.
        min_weight: Minimum weight of the superpixels to include.
        title: Title of the plot.
        save_path: Path to save the plot. If None, the plot is displayed.
        boost_channels: If True, boost color channels in highlighted regions
            before drawing boundaries.

    """
    temp, mask = get_image_and_mask(
        result,
        positive_only=positive_only,
        hide_rest=hide_rest,
        num_features=num_features,
        min_weight=min_weight,
        boost_channels=boost_channels,
    )

    # Apply mark_boundaries directly on the normalized [0, 1] image
    img_bound = mark_boundaries(temp, mask)  # type: ignore[no-untyped-call]

    plt.figure(figsize=(8, 6))
    plt.imshow(img_bound)
    if title:
        plt.title(title)
    plt.axis("off")

    if save_path:
        plt.savefig(str(save_path), bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def image_regions(
    result: ImageClassificationXWhyResult,
    positive_only: bool = True,
    num_features: int = 5,
    min_weight: float = 0.0,
    title: str | None = None,
    save_path: str | Path | None = None,
) -> None:
    """Plot highlighted positive and negative regions of the explanation.

    Args:
        result: The classification explanation result.
        positive_only: If True, only plot positive superpixels.
        num_features: Number of superpixels to include.
        min_weight: Minimum weight of the superpixels to include.
        title: Title of the plot.
        save_path: Path to save the plot. If None, the plot is displayed.

    """
    temp, mask = get_image_and_mask(
        result,
        positive_only=positive_only,
        hide_rest=False,
        num_features=num_features,
        min_weight=min_weight,
    )

    # Convert -1/1 mask convention to 1/2 for label2rgb
    mask_for_rgb = np.where(mask == -1, 2, mask)

    img_reg = label2rgb(  # type: ignore[no-untyped-call]
        mask_for_rgb,
        temp,
        bg_label=0,
        colors=["green", "red"],
        saturation=1,  # Required for full color intensity in modern skimage
    )

    plt.figure(figsize=(8, 6))
    plt.imshow(img_reg, interpolation="nearest")
    if title:
        plt.title(title)
    plt.axis("off")

    if save_path:
        plt.savefig(str(save_path), bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def image_regions_side_by_side(
    result: ImageClassificationXWhyResult,
    num_features_pos: int = 5,
    num_features_all: int = 10,
    min_weight: float = 0.0,
    title1: str = "Positive Regions",
    title2: str = "Positive/Negative Regions",
    save_path: str | Path | None = None,
    boost_channels: bool = False,
) -> None:
    """Plot positive and combined positive/negative regions side by side.

    Args:
        result: The classification explanation result.
        num_features_pos: Number of superpixels to include in the positive plot.
        num_features_all: Number of superpixels to include in the all-regions
            plot.
        min_weight: Minimum weight of the superpixels to include.
        title1: Title for the first subplot.
        title2: Title for the second subplot.
        save_path: Path to save the plot. If None, the plot is displayed.
        boost_channels: If True, boost color channels in highlighted regions.

    """
    # Left: positive only (background=0, positive=1)
    temp1, mask1 = get_image_and_mask(
        result,
        positive_only=True,
        hide_rest=False,
        num_features=num_features_pos,
        min_weight=min_weight,
        boost_channels=boost_channels,
    )

    # Right: positive + negative
    temp2, mask2 = get_image_and_mask(
        result,
        positive_only=False,
        hide_rest=False,
        num_features=num_features_all,
        min_weight=min_weight,
        boost_channels=boost_channels,
    )

    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # Default color mapping:
    # 1 -> 'red' (positive), 2 -> 'blue' (negative), 3 -> 'yellow' (background)
    default_colors = ["red", "blue", "yellow"]

    # Left panel: positive regions only
    ax1.imshow(
        label2rgb(  # type: ignore[no-untyped-call]
            mask1,
            temp1,
            bg_label=0,
            colors=["red"],
            alpha=0.3,
        ),
        interpolation="nearest",
    )
    ax1.set_title(title1)

    # Right panel: map labels so 1 = positive ('red'), 2 = negative ('blue'),
    # and 3 = background ('yellow')
    mask2_mapped = np.full_like(mask2, fill_value=3, dtype=int)
    mask2_mapped[mask2 == 1] = 1
    mask2_mapped[(mask2 == 2) | (mask2 == -1)] = 2

    present_labels = np.unique(mask2_mapped)  # Subset of [1, 2, 3]
    right_colors = [default_colors[label - 1] for label in present_labels]

    # Since labels are {1, 2, 3}, bg_label=0 colors all 3 regions
    ax2.imshow(
        label2rgb(  # type: ignore[no-untyped-call]
            mask2_mapped,
            temp2,
            bg_label=0,
            colors=right_colors,
            alpha=0.3,
        ),
        interpolation="nearest",
    )
    ax2.set_title(title2)

    plt.tight_layout()
    if save_path:
        plt.savefig(str(save_path), bbox_inches="tight")
    else:
        plt.show()

    plt.close()
