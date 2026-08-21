"""Factory for point cloud model instantiation."""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

from xwhy.models.point_cloud.base import BasePointCloudModel
from xwhy.models.point_cloud.types import PointCloudModelType


class PointCloudModelFactory:
    """Manage point cloud model instantiation via registry."""

    _registry: ClassVar[
        dict[PointCloudModelType, Callable[..., BasePointCloudModel]]
    ] = {}

    @classmethod
    def register(
        cls,
        model_type: PointCloudModelType,
        builder: Callable[..., BasePointCloudModel],
    ) -> None:
        """Register a builder function for a point cloud model type.

        Args:
            model_type: Type of point cloud model.
            builder: Callable that returns BasePointCloudModel instance.

        Raises:
            ValueError: If model_type is already registered.

        """
        if model_type in cls._registry:
            raise ValueError(f"Model type already registered: {model_type}")
        cls._registry[model_type] = builder

    @classmethod
    def create(
        cls,
        model_type: PointCloudModelType,
        **kwargs: object,
    ) -> BasePointCloudModel:
        """Instantiate point cloud model wrapper.

        Args:
            model_type: Model enum type.
            **kwargs: Extra parameters.

        Returns:
            Instantiated BasePointCloudModel object.

        Raises:
            ValueError: If model_type is not registered.

        """
        if model_type not in cls._registry:
            raise ValueError(f"Unsupported point cloud model type: {model_type}")

        return cls._registry[model_type](**kwargs)

    @classmethod
    def clear(cls) -> None:
        """Reset registry to default state."""
        cls._registry.clear()
