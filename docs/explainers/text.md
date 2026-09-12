---
title: XWhy Text Explainer
description: Use XWhy TextExplainer for model-agnostic local explanations of conventional text-model predictions.
---

# Text explainer

!!! success "Available"
    `TextExplainer` is implemented and exported by XWhy for conventional text prediction workflows.

`TextExplainer` is intended for text classifiers and compatible prediction functions that are distinct from prompt-response LLM explanation. It perturbs the input text, queries the black-box prediction function, measures semantic distance between the original and perturbed texts, and fits a weighted local surrogate model.

## Basic use

```python
from xwhy import TextExplainer

explainer = TextExplainer(
    model=classifier,
    num_perturbations=64,
)

result = explainer.explain(
    "The service was reliable and easy to use.",
    class_index=1,
)
```

You can provide a model exposing `predict_proba`, `predict`, or `__call__`, or pass a compatible `predict_fn` directly.

## Current behaviour

The current implementation supports:

- string inputs for conventional text prediction;
- model or direct prediction-function interfaces;
- configurable perturbation counts and random seed;
- word-presence perturbation masks;
- text embeddings and Word Mover's Distance for local weighting;
- configurable surrogate models and automatic surrogate selection;
- word-level surrogate coefficients;
- surrogate fidelity metrics;
- optional fidelity plotting through `fidelity_plot=True`.

## Text explainer versus LLM explainer

Use `TextExplainer` when the black-box target is a conventional text prediction function, such as a classifier returning class scores or labels.

Use [`LLMExplainer`](llm/index.md) when the target behaviour is the relationship between an LLM prompt and its generated response.

## Interpretation

Word coefficients describe a local surrogate approximation around the selected text and perturbation strategy. They do not reveal hidden reasoning and should not be treated as causal effects.

For reproducible use, report the perturbation count, embedding and distance configuration, target class, surrogate configuration, random seed, and fidelity metrics.

[View the current API reference](../reference/xwhy/explainers/text.md)
