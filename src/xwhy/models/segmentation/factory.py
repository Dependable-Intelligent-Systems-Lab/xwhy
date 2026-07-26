"""Factory for segmentation implementations."""

from collections.abc import Callable
from typing import ClassVar

from xwhy.models.segmentation.base import BaseSegmentation
from xwhy.models.segmentation.types import SegmentationType


class SegmentationFactory:
    """Manage segmentation model instantiation via a registry."""

    _registry: ClassVar[dict[SegmentationType, Callable[..., BaseSegmentation]]] = {}

    @classmethod
    def register(
        cls,
        segmentation: SegmentationType,
        builder: Callable[..., BaseSegmentation],
    ) -> None:
        """Register a builder function for a segmentation type.

        Args:
            segmentation: The type of segmentation model to register.
            builder: A callable (function/lambda) that accepts keyword arguments
                and returns a BaseSegmentation instance.

        Raises:
            ValueError: If the segmentation type is already registered.

        """
        if segmentation in cls._registry:
            raise ValueError(f"Segmentation already registered: {segmentation}")
        cls._registry[segmentation] = builder

    @classmethod
    def create(
        cls, segmentation: SegmentationType, **kwargs: object
    ) -> BaseSegmentation:
        """Instantiate and configure a segmentation model.

        Args:
            segmentation: The type of segmentation model to create.
            **kwargs: Arbitrary keyword arguments passed to the builder function,
                such as 'settings' or 'model_name'.

        Returns:
            An instantiated BaseSegmentation object.

        Raises:
            ValueError: If the segmentation type is not registered.

        """
        if segmentation not in cls._registry:
            raise ValueError(f"Unsupported segmentation: {segmentation}")

        return cls._registry[segmentation](**kwargs)

    @classmethod
    def clear(cls) -> None:
        """Reset registry to defaults."""
        cls._registry.clear()
