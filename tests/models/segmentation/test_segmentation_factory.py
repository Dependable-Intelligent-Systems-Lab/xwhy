"""Tests for SegmentationFactory."""

from __future__ import annotations

import pytest

from xwhy.models.segmentation.factory import SegmentationFactory
from xwhy.models.segmentation.torchvision_models import TorchvisionSegmentation
from xwhy.models.segmentation.types import SegmentationType


def test_register_and_create_segmentation() -> None:
    """Register and create segmentation successfully."""
    SegmentationFactory.clear()

    def _builder(**kwargs: object) -> TorchvisionSegmentation:
        return TorchvisionSegmentation(settings=kwargs["settings"])  # type: ignore[arg-type]

    SegmentationFactory.register(
        segmentation=SegmentationType.DEEPLABV3_RESNET101,
        builder=_builder,
    )

    segmentation = SegmentationFactory.create(
        segmentation=SegmentationType.DEEPLABV3_RESNET101,
        settings=object(),
    )

    assert isinstance(segmentation, TorchvisionSegmentation)


def test_register_duplicate_segmentation() -> None:
    """Ensure duplicate registration raises error."""
    SegmentationFactory.clear()

    SegmentationFactory.register(
        segmentation=SegmentationType.DEEPLABV3_RESNET101,
        builder=lambda **kwargs: TorchvisionSegmentation(settings=kwargs["settings"]),
    )

    with pytest.raises(ValueError, match="already registered"):
        SegmentationFactory.register(
            segmentation=SegmentationType.DEEPLABV3_RESNET101,
            builder=lambda **kwargs: TorchvisionSegmentation(
                settings=kwargs["settings"]
            ),
        )


def test_unsupported_segmentation_raises() -> None:
    """Test that unsupported segmentation type raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported segmentation"):
        SegmentationFactory.create(
            segmentation="non_existent_segmentation",  # type: ignore[arg-type]
            settings=object(),
        )
