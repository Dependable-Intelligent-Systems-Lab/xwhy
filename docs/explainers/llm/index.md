---
title: LLM Explainability with XWhy
description: Explain local prompt influence on LLM responses with XWhy using text perturbations, response-distance measures, surrogate models, and token heatmaps.
---

# LLM explainer

!!! success "Available"
    `LLMExplainer` is implemented and has a complete tutorial with an executed worked example.

The LLM explainer treats the language model as a black-box provider. It perturbs the input prompt, obtains responses, measures response changes, and fits a local surrogate model to estimate input influence.

Current documented capabilities include:

- commercial, cloud, router, and local providers;
- runtime, `.env`, and configuration-object setup;
- text embeddings used by the response-distance workflow;
- automatic or selected surrogate models;
- token-level and contribution plots; and
- embedding comparison with fidelity metrics and executed outputs.

[Read the complete LLM tutorial and worked example](../../llm_explainer.md)

[Open the LLM explainer API reference](../../reference/xwhy/explainers/llm/)

## Interpretation boundary

The result is a local perturbation-based approximation. It does not reveal private chain-of-thought or reconstruct the model's exact internal computation.