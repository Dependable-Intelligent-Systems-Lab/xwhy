"""Unit tests for CustomPointCloudModel."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from xwhy.models.point_cloud.custom import CustomPointCloudModel

# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


def test_init_requires_model_or_predict_fn() -> None:
    """Raise ValueError when neither model nor predict_fn is supplied."""
    with pytest.raises(ValueError, match="Either 'model' or 'predict_fn'"):
        CustomPointCloudModel(model=None, predict_fn=None)


def test_init_with_model_only() -> None:
    """Store a provided model and leave predict_fn as None."""
    model = MagicMock(spec=torch.nn.Module)
    wrapper = CustomPointCloudModel(model=model)
    assert wrapper.model is model
    assert wrapper.predict_fn is None


def test_init_with_predict_fn_only() -> None:
    """Store a provided predict_fn and leave model as None."""

    def fake_predict(**_: Any) -> tuple[int, torch.Tensor, list[int]]:  # noqa: ANN401
        return 0, torch.tensor([[0.9, 0.1]]), [0]

    wrapper = CustomPointCloudModel(predict_fn=fake_predict)
    assert wrapper.model is None
    assert wrapper.predict_fn is fake_predict


def test_init_with_both_and_kwargs() -> None:
    """Accept both model and predict_fn together with extra kwargs."""
    model = MagicMock(spec=torch.nn.Module)

    def fake_predict(**_: Any) -> tuple[int, torch.Tensor, list[int]]:  # noqa: ANN401
        return 1, torch.tensor([[0.2, 0.8]]), [1]

    wrapper = CustomPointCloudModel(model=model, predict_fn=fake_predict, extra_arg=42)
    assert wrapper.model is model
    assert wrapper.predict_fn is fake_predict
    assert wrapper.kwargs == {"extra_arg": 42}


# ---------------------------------------------------------------------------
# predict - predict_fn path
# ---------------------------------------------------------------------------


def test_predict_delegates_to_predict_fn() -> None:
    """Call the user-supplied predict_fn when it is present."""
    expected = (3, torch.tensor([[0.1, 0.2, 0.7]]), [2, 1, 0])

    def fake_predict(
        sample_input: torch.Tensor,
        sample_label: int | None,
        model: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[int, torch.Tensor, list[int]]:
        assert sample_label == 5
        assert kwargs.get("scale") == 2.0
        return expected

    wrapper = CustomPointCloudModel(predict_fn=fake_predict, scale=2.0)
    result = wrapper.predict(sample_input=torch.rand(10, 3), sample_label=5)
    assert result == expected


# ---------------------------------------------------------------------------
# predict - model path error
# ---------------------------------------------------------------------------


def test_predict_raises_when_model_missing() -> None:
    """Raise RuntimeError when predict_fn is absent and model is None."""
    # Bypass the __init__ guard by setting attributes after construction
    wrapper = CustomPointCloudModel(predict_fn=lambda **_: (0, torch.zeros(1, 2), [0]))
    wrapper.predict_fn = None
    wrapper.model = None

    with pytest.raises(RuntimeError, match="Underlying PyTorch model is missing"):
        wrapper.predict(sample_input=torch.rand(5, 3))


# ---------------------------------------------------------------------------
# predict - model path, dimensionality & output handling
# ---------------------------------------------------------------------------


def _make_model(output: torch.Tensor | tuple[torch.Tensor, ...]) -> MagicMock:
    """Create a mock torch.nn.Module that returns the given output."""
    model = MagicMock(spec=torch.nn.Module)
    model.return_value = output
    model.eval = MagicMock()
    return model


def test_predict_unsqueezes_two_dimensional_input() -> None:
    """Add a batch dimension when the input tensor is two-dimensional."""
    logits = torch.tensor([[0.1, 0.8, 0.1]])
    model = _make_model(logits)
    wrapper = CustomPointCloudModel(model=model)

    pred, output, top = wrapper.predict(sample_input=torch.rand(8, 3))

    assert pred == 1
    assert output is logits
    assert isinstance(top, list)
    # model must have received a 3-D tensor
    call_arg = model.call_args.args[0]
    assert call_arg.ndim == 3


def test_predict_accepts_three_dimensional_input() -> None:
    """Pass a three-dimensional tensor through without extra unsqueeze."""
    logits = torch.tensor([[0.6, 0.3, 0.1]])
    model = _make_model(logits)
    wrapper = CustomPointCloudModel(model=model)

    pred, output, _ = wrapper.predict(sample_input=torch.rand(1, 8, 3))

    assert pred == 0
    assert output is logits
    call_arg = model.call_args.args[0]
    assert call_arg.ndim == 3


def test_predict_handles_tuple_model_output() -> None:
    """Extract the first element when the model returns a tuple."""
    logits = torch.tensor([[0.05, 0.15, 0.8]])
    model = _make_model((logits, torch.rand(1, 16)))
    wrapper = CustomPointCloudModel(model=model)

    pred, output, top = wrapper.predict(sample_input=torch.rand(6, 3))

    assert pred == 2
    assert output is logits
    assert 2 in top


def test_predict_topk_when_output_is_one_dimensional() -> None:
    """Use k_top=1 when the model output has only one dimension."""
    # Simulate a model that returns a 1-D tensor (edge case)
    logits = torch.tensor([0.2, 0.5, 0.3])
    model = _make_model(logits)
    wrapper = CustomPointCloudModel(model=model)

    # Force the code path that sees ndim != 2
    with (
        patch.object(torch, "max", return_value=(torch.tensor(0.5), torch.tensor(1))),
        patch.object(
            torch,
            "topk",
            return_value=(torch.tensor([0.5]), torch.tensor([[1]])),
        ) as mock_topk,
    ):
        pred, output, _ = wrapper.predict(sample_input=torch.rand(4, 3))

    assert isinstance(pred, int)
    assert output is logits
    # k_top must have been 1
    assert mock_topk.call_args.kwargs.get("k") == 1 or mock_topk.call_args.args[1] == 1


# ---------------------------------------------------------------------------
# get_output_probabilities
# ---------------------------------------------------------------------------


def test_get_output_probabilities_raises_without_model_or_fn() -> None:
    """Raise RuntimeError when both model and predict_fn are missing."""
    wrapper = CustomPointCloudModel(predict_fn=lambda **_: (0, torch.zeros(1, 2), [0]))
    wrapper.model = None
    wrapper.predict_fn = None

    with pytest.raises(RuntimeError, match="Model or predict_fn is required"):
        wrapper.get_output_probabilities(
            samples=[torch.rand(5, 3)], device=torch.device("cpu")
        )


def test_get_output_probabilities_returns_concatenated_probs() -> None:
    """Softmax each prediction and concatenate the probability tensors."""
    logits_batch = [
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[0.0, 2.0]]),
    ]
    call_idx = {"i": 0}

    def fake_predict(
        sample_input: torch.Tensor,
        sample_label: int | None = None,
        model: Any = None,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[int, torch.Tensor, list[int]]:
        out = logits_batch[call_idx["i"]]
        call_idx["i"] += 1
        return 0, out, [0]

    wrapper = CustomPointCloudModel(predict_fn=fake_predict)
    samples = [torch.rand(4, 3), torch.rand(4, 3)]
    probs = wrapper.get_output_probabilities(
        samples=samples, device=torch.device("cpu")
    )

    assert isinstance(probs, torch.Tensor)
    assert probs.shape[0] == 2

    expected_0 = torch.softmax(logits_batch[0], dim=1).squeeze(0)
    expected_1 = torch.softmax(logits_batch[1], dim=1).squeeze(0)
    assert torch.allclose(probs[0], expected_0, atol=1e-5)
    assert torch.allclose(probs[1], expected_1, atol=1e-5)
