"""Tests for the types module."""

import torch

from xwhy.core.types import ImageClassificationState


def test_image_classification_state_init() -> None:
    """Test the initialization of ImageClassificationState."""
    expected_device = torch.device("cpu")

    state = ImageClassificationState(device_=expected_device)

    assert state.device == expected_device
    assert state.classification_model is None
    assert state.segmentation_model is None
    assert state.embedding_model is None


def test_image_classification_state_init_with_cuda() -> None:
    """Test the initialization with a different device string/type."""
    expected_device = torch.device("cuda:0")

    state = ImageClassificationState(device_=expected_device)

    assert state.device == expected_device
