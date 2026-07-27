---
title: XWhy Explainability Documentation
description: Use XWhy and SMILE to explain image classifiers and LLM responses, with a scalable roadmap for image generation, text, tabular, point-cloud, time-series, and multimodal explainability.
---

# XWhy explainability documentation

**Explain black-box behaviour with a SMILE.**

XWhy is a Python library for model-agnostic local explainability. It uses **SMILE**—Statistical Model-agnostic Interpretability with Local Explanations—to perturb an input, observe changes in model behaviour, fit a local surrogate model, and report feature-level influence together with explanation-quality evidence.

!!! info "Current package maturity"
    XWhy `0.0.2` is currently classified as **pre-alpha**. Image-classification and LLM explainers are implemented. Other capabilities are clearly labelled as under construction or coming soon.

## Start here

- [Install XWhy](getting-started/installation.md)
- [Generate your first explanation](getting-started/quick-start.md)
- [Choose the correct explainer](getting-started/choosing-an-explainer.md)
- [Browse all explainers and their status](explainers/index.md)
- [Read the generated API reference](reference/)

## Capability overview

| Capability | Public component | Documentation status | Implementation status |
| --- | --- | --- | --- |
| Image classification | `ImageClassificationExplainer` | Available | Available |
| Image generation and editing | `Pix2PixExplainer` prototype | Under construction | Interface only |
| LLM prompt-response | `LLMExplainer` | Available | Available |
| Tabular | `TabularExplainer` | Under construction | Interface only |
| Text | `TextExplainer` | Under construction | Interface only |
| Point cloud | `PointCloudExplainer` | Under construction | Interface only |
| Time series | Planned | Coming soon | Not yet implemented |
| Multimodal | Planned | Coming soon | Not yet implemented |

Pix2Pix is documented as one image-editing model family within the broader [image-generation explainability](explainers/image-generation/index.md) section.

## What XWhy explanations mean

XWhy produces local, perturbation-based approximations of model behaviour around a selected input. An explanation can identify associations between input components and changes in model output, but it does not expose a model's private internal reasoning or prove causality.

Read [limitations and responsible use](concepts/limitations.md) before using explanations in safety-critical, medical, legal, financial, or high-impact decisions.
