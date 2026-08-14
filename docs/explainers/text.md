---
title: XWhy Text Explainer
description: Development status and planned documentation for conventional text-model explainability with XWhy.
---

# Text explainer

!!! warning "Under construction"
    `TextExplainer` is exported by XWhy, but its current `explain()` method raises `NotImplementedError`.

This explainer is intended for conventional text tasks that are distinct from prompt-response LLM explanation.

Planned documentation will cover:

- text classification and regression interfaces;
- token, word, phrase, and sentence perturbation units;
- masking and replacement strategies;
- embedding and distance selection;
- token-level visualisation;
- stability and faithfulness evaluation.

For implemented prompt-response analysis, use the [LLM explainer](llm/index.md).

[View the current API reference](../reference/xwhy/explainers/text.md)
