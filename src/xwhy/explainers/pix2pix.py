"""Pix2Pix explainer abstractions."""

from xwhy.core.config import ExplainerConfig
from xwhy.core.explainer import BaseExplainer
from xwhy.core.result import BaseXWhyResult


class Pix2PixExplainer(BaseExplainer):
    """Explainer for Pix2Pix / Instruct tasks."""

    def __init__(
        self,
        model: object,
        config: ExplainerConfig | None = None,
    ) -> None:
        """Initialize the explainer."""
        super().__init__(model, config)

    def explain(
        self,
        instance: object,
        **kwargs: object,
    ) -> BaseXWhyResult:
        """Generate an explanation for the given input instance.

        Args:
            instance: The input object to explain.
            **kwargs: Additional explainer-specific options.

        Returns:
            An ``XWhyResult`` containing the explanation output.

        Raises:
            NotImplementedError: Always raised in Phase 1.

        """
        raise NotImplementedError(
            "Pix2PixExplainer.explain to be implemented in later phases."
        )
