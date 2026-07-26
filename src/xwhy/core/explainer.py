"""Base class for all xwhy explainers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from xwhy.core.config import ExplainerConfig
from xwhy.core.result import BaseXWhyResult


class BaseExplainer(ABC):
    """Abstract base class for all xwhy explainers.

    This class provides the common interface shared by all explainers.
    Concrete explainers are responsible for initializing their own runtime
    state (models, providers, embeddings, etc.).
    """

    def __init__(
        self,
        config: ExplainerConfig | None = None,
    ) -> None:
        """Initialize the explainer.

        Args:
            config: Explainer configuration object.

        """
        self.config = config

    @abstractmethod
    def explain(
        self,
        instance: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> BaseXWhyResult:
        """Generate an explanation for the given input instance.

        Args:
            instance: Input instance to explain.
            **kwargs: Additional explainer-specific arguments.

        Returns:
            Structured explanation result.

        Raises:
            NotImplementedError:
                If the subclass does not implement this method.

        """
        raise NotImplementedError("Subclasses must implement explain method.")
