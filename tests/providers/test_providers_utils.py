"""Unit tests for provider utilities."""

from unittest.mock import MagicMock

from xwhy.providers.base import BaseProvider
from xwhy.providers.utils import score_perturbations


def test_score_perturbations() -> None:
    """Test score_perturbations with a mocked provider."""
    # Create a mock provider inheriting from BaseProvider
    mock_provider = MagicMock(spec=BaseProvider)

    # Configure the mock score method to return predictable values
    mock_provider.score.side_effect = ["0.8", "0.2"]

    perturbations = ["Text A", "Text B"]
    model_name = "test-model"

    results = score_perturbations(
        provider=mock_provider,
        model=model_name,
        perturbations=perturbations,
        max_tokens=5,
        temperature=0.1,
    )

    # Assert correct return values
    assert len(results) == 2
    assert results[0] == ("Text A", "0.8")
    assert results[1] == ("Text B", "0.2")

    # Assert provider was called with correct parameters
    assert mock_provider.score.call_count == 2
    mock_provider.score.assert_any_call(
        prompt="Text A",
        model=model_name,
        max_tokens=5,
        temperature=0.1,
    )
    mock_provider.score.assert_any_call(
        prompt="Text B",
        model=model_name,
        max_tokens=5,
        temperature=0.1,
    )
