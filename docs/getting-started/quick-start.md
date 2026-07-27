---
title: XWhy Quick Start
description: Choose an implemented XWhy workflow and generate a first local explanation for an image classifier or an LLM response.
---

# Quick start

XWhy currently provides two executable explanation workflows.

## Explain an image classifier

Use `ImageClassificationExplainer` for a PyTorch image-classification model. The explainer estimates how image regions influence a selected prediction.

[Open the image-classification tutorial](../image_classification_explainer.md)

## Explain an LLM response

Use `LLMExplainer` to perturb a text prompt, compare the resulting model responses, and estimate local token or word influence.

[Open the LLM tutorial](../llm_explainer.md)

## Do not use development interfaces as implemented explainers

`TabularExplainer`, `TextExplainer`, and `PointCloudExplainer` are currently public development interfaces whose `explain()` methods are not implemented.

Image-generation explainability is also under construction. The current `Pix2PixExplainer` class is an early interface within the broader image-generation and image-editing roadmap; it is not yet an executable workflow.

See the [explainer status matrix](../explainers/index.md) before designing a workflow.
