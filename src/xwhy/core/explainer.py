"""Base class for all xwhy explainers."""

from abc import ABC, abstractmethod
from typing import Any

from xwhy.core.config import ExplainerConfig
from xwhy.core.result import BaseXWhyResult


class BaseExplainer(ABC):
    """Abstract base class for all xwhy explainers."""

    def __init__(self, model: Any, config: ExplainerConfig | None = None) -> None:  # noqa: ANN401
        """Initialize the explainer."""
        self.model = model
        self.config = config or ExplainerConfig()

    @abstractmethod
    def explain(self, instance: Any, **kwargs: Any) -> BaseXWhyResult:  # noqa: ANN401
        """Generate explanation for the given instance."""
        raise NotImplementedError("Subclasses must implement explain method.")
