---
title: LLM Explainability with XWhy
description: Explain local prompt-to-response alignment with XWhy using text perturbations, semantic-distance measures, surrogate models, and token heatmaps.
---

# LLM explainer

!!! success "Available"
    `LLMExplainer` is implemented and has a complete tutorial with an executed worked example.

The LLM explainer treats the language model as a black-box provider. It obtains the original response, perturbs the input prompt, measures semantic distance between that response and the perturbed prompt variants, and fits a local surrogate model to estimate term contributions to the resulting response-alignment score.

Current documented capabilities include:

- commercial, cloud, router, and local providers;
- runtime, `.env`, and configuration-object setup;
- text embeddings used by the semantic-distance workflow;
- automatic or selected surrogate models;
- token-level and contribution plots; and
- embedding comparison with fidelity metrics and executed outputs.

[Read the complete LLM tutorial and worked example](../../llm_explainer.md)

[Open the LLM explainer API reference](../../reference/xwhy/explainers/llm/)

## Interpretation boundary

The result is a local perturbation-based response-alignment approximation. The current implementation queries the provider for the original response only; it does not reveal private chain-of-thought or reconstruct the model's exact internal computation.
