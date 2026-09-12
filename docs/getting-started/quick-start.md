---
title: XWhy Quick Start
description: Choose an implemented XWhy workflow for image classification, image generation and editing, LLM responses, tabular prediction, or text prediction.
---

# Quick start

XWhy currently provides five executable explainer areas. Choose the workflow that matches the behaviour you want to explain.

## Explain an image classifier

Use `ImageClassificationExplainer` for a PyTorch image-classification model. The explainer estimates how image regions influence a selected prediction.

[Open the image-classification tutorial](../image_classification_explainer.md)

## Explain image generation or editing

Use `ImageGenerationAndEditingExplainer` for supported image-generation or image-editing providers, pipelines, or compatible custom generation functions. The workflow perturbs the textual instruction, generates or edits images, measures changes in the image output, and fits a local surrogate model.

[Open the image generation & editing guide](../explainers/image-generation/index.md)

## Explain an LLM response

Use `LLMExplainer` to perturb a text prompt, compare the resulting model responses, and estimate local word influence.

[Open the LLM tutorial](../llm_explainer.md)

## Explain a tabular prediction

Use `TabularExplainer` for structured classification or regression. The implementation generates local perturbations, computes feature-distribution distances, queries the black-box model, and fits a weighted surrogate explanation.

[Open the tabular explainer guide](../explainers/tabular.md)

## Explain a conventional text prediction

Use `TextExplainer` for text classifiers or compatible prediction functions. It perturbs the input text, evaluates the black-box model, computes text distance, and estimates local word contributions.

[Open the text explainer guide](../explainers/text.md)

## Development and roadmap capabilities

`PointCloudExplainer` is currently an exported development interface whose `explain()` method raises `NotImplementedError`.

Time Series, Multimodal, Agentic AI, and Multi-Agent AI are documented roadmap capabilities and are not yet exported as supported explainers.

See the [explainer status matrix](../explainers/index.md) before designing a workflow.
