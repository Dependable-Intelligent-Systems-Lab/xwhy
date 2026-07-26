"""Factory for classification implementations."""

from collections.abc import Callable
from typing import ClassVar

from xwhy.models.classification.base import BaseClassification
from xwhy.models.classification.types import ClassificationType


class ClassificationFactory:
    """Manage classification model instantiation via a registry."""

    _registry: ClassVar[
        dict[ClassificationType, Callable[..., BaseClassification]]
    ] = {}

    @classmethod
    def register(
        cls,
        classification: ClassificationType,
        builder: Callable[..., BaseClassification],
    ) -> None:
        """Register a builder function for a classification type.

        Args:
            classification: The type of classification model to register.
            builder: A callable (function/lambda) that accepts keyword arguments
                and returns a BaseClassification instance.

        Raises:
            ValueError: If the classification type is already registered.

        """
        if classification in cls._registry:
            raise ValueError(f"Classification already registered: {classification}")
        cls._registry[classification] = builder

    @classmethod
    def create(
        cls, classification: ClassificationType, **kwargs: object
    ) -> BaseClassification:
        """Instantiate and configure a classification model.

        Args:
            classification: The type of classification model to create.
            **kwargs: Arbitrary keyword arguments passed to the builder function,
                such as 'settings' or 'model_name'.

        Returns:
            An instantiated BaseClassification object.

        Raises:
            ValueError: If the classification type is not registered.

        """
        if classification not in cls._registry:
            raise ValueError(f"Unsupported classification: {classification}")

        return cls._registry[classification](**kwargs)

    @classmethod
    def clear(cls) -> None:
        """Reset registry to defaults."""
        cls._registry.clear()
