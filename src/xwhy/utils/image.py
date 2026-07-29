"""Image utility functions for loading, preprocessing, and format conversion."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import skimage.transform
import torch
from PIL import Image
from torchvision import transforms

from xwhy.logger import logger

# Distinct high-contrast palette for visualization (0 is background gray)
VIS_PALETTE: list[tuple[int, int, int]] = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (170, 0, 0),
    (0, 170, 0),
    (0, 0, 170),
]


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


def create_sequential_segmentation_mask(
    prediction: np.ndarray,
    class_names: Sequence[str] | None = None,
) -> tuple[Image.Image, np.ndarray]:
    """Map class ID predictions to sequential integer labels and a visual image.

    Args:
        prediction: 2D numpy array of predicted class IDs (H, W).
        class_names: Optional sequence of class name strings corresponding to IDs.

    Returns:
        tuple[Image.Image, np.ndarray]: Visualization PIL image and semantic
            mask with sequential labels (0 for background, 1, 2... for objects).

    """
    height, width = prediction.shape
    # sem_mask will hold your sequential labels (0, 1, 2, ...)
    sem_mask = np.zeros((height, width), dtype=np.uint8)
    # visual_np will hold the colored visualization
    visual_np = np.full(
        (height, width, 3), 128, dtype=np.uint8
    )  # Start with gray BG (128, 128, 128)

    # Get the unique class IDs found by the segmentation model (e.g., 8, 12, 15)
    unique_class_ids = np.unique(prediction)
    # Filter out the background ID (typically 0)
    object_class_ids = [cls_id for cls_id in unique_class_ids if cls_id != 0]

    # Map detected objects to sequential labels (1, 2, 3...)

    # Start sequential label counter at 1 (0 is always background)
    sequential_label = 1

    for cls_id in object_class_ids:
        # Create a mask for all pixels belonging to the current object class ID
        mask = prediction == cls_id
        # Assign the sequential label to the semantic mask
        sem_mask[mask] = sequential_label

        # Assign a unique visualization color (cycle through palette)
        color = VIS_PALETTE[(sequential_label - 1) % len(VIS_PALETTE)]
        visual_np[mask] = color

        if class_names is not None and cls_id < len(class_names):
            class_name = class_names[cls_id]
        else:
            class_name = f"Unknown_ID_{cls_id}"

        logger.debug(
            "Mapping model class '%s' (ID: %s) to sequential label: %d",
            class_name,
            cls_id,
            sequential_label,
        )

        # Increment the sequential label counter for the next object
        sequential_label += 1

    # Final conversion
    visual_image = Image.fromarray(visual_np)
    return visual_image, sem_mask


def get_segmentation_mask(
    image_path: str | Path,
    segmentation_model: Callable[[torch.Tensor], Any],
    transform_fn: Callable[[Image.Image], torch.Tensor] | None = None,
    device: str | torch.device = "cpu",
    class_names: Sequence[str] | None = None,
) -> tuple[Image.Image, np.ndarray]:
    """Generate a semantic segmentation mask and colored visualization image.

    Args:
        image_path: Path to the input image file.
        segmentation_model: Callable model receiving tensor (1, C, H, W) and
            returning either a tensor or a dict with 'out' key.
        transform_fn: Preprocessing transform for PIL image. Uses default if None.
        device: Target device for running model inference.
        class_names: Optional class name list for logging.

    Returns:
        tuple[Image.Image, np.ndarray]: Visualization PIL image and 2D NumPy
            array of sequential labels (0 for background).

    Raises:
        TypeError: If the segmentation model returns an unsupported type.

    """
    # Load and Preprocess Image
    img_pil = Image.open(image_path).convert("RGB")
    width, height = img_pil.size

    pipeline = transform_fn if transform_fn is not None else get_default_transform()
    input_tensor = pipeline(img_pil).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = segmentation_model(input_tensor)

    if isinstance(output, dict) and "out" in output:
        output_tensor = cast(torch.Tensor, output["out"][0])
    elif torch.is_tensor(output):
        output_tensor = output[0] if output.ndim == 4 else output
    else:
        raise TypeError(f"Unsupported model output type: {type(output)}")

    # Get the predicted class index for each pixel (H, W)
    prediction = output_tensor.argmax(0).byte().cpu().numpy()

    # Resize mask back to original image size
    if prediction.shape != (height, width):
        resized = skimage.transform.resize(
            prediction,
            (height, width),
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        )  # type: ignore[no-untyped-call]

        prediction = cast(np.ndarray, resized).astype(np.uint8)

    return create_sequential_segmentation_mask(prediction, class_names=class_names)


def get_binary_mask(
    image_path: str | Path,
    segmentation_model: Callable[[torch.Tensor], Any],
    transform_fn: Callable[[Image.Image], torch.Tensor] | None = None,
    device: str | torch.device = "cpu",
    class_names: Sequence[str] | None = None,
) -> Image.Image:
    """Generate a binary 0/255 mask using the generic segmentation utility.

    Args:
        image_path: Path to the input image file.
        segmentation_model: Callable model for inference.
        transform_fn: Preprocessing transform for PIL image.
        device: Target device for running model inference.
        class_names: Optional class name list for logging.

    Returns:
        Image.Image: Grayscale binary mask (0 for background, 255 for objects).

    """
    try:
        # Generate semantic mask using the generic segmentation utility
        _, sem_mask_np = get_segmentation_mask(
            image_path=image_path,
            segmentation_model=segmentation_model,
            transform_fn=transform_fn,
            device=device,
            class_names=class_names,
        )

        # Convert sequential mask to binary mask (non-zero labels become 255)
        binary_mask_np = (sem_mask_np > 0).astype(np.uint8) * 255

        # Convert to PIL Image with Grayscale mode
        return Image.fromarray(binary_mask_np, mode="L")

    except Exception as e:
        logger.exception(f"Failed to generate binary mask via generic utility: {e}")
        # Return a blank white mask as fallback to allow full editing
        original_img = Image.open(image_path)
        return Image.new("L", original_img.size, 255)
