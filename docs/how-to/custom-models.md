---
title: Connect a Custom Model to XWhy
description: Connect image, generative-image, LLM, tabular, and text models to the currently implemented XWhy explainers.
---

# Connect a custom model

## Image classification

Custom PyTorch image classifiers are currently documented. Supply the model, its matching preprocessing pipeline, and optional category labels.

[Follow the custom PyTorch model guide](../image_classification_explainer.md#using-a-custom-pytorch-model)

## LLM providers

The LLM explainer accepts built-in provider identifiers and provider-specific client arguments. See [provider configuration](providers.md).

## Image generation and editing

`ImageGenerationAndEditingExplainer` supports multiple integration routes, including supported providers, a pre-loaded compatible pipeline, or a custom model/generation function.

For a custom integration, provide a model or pipeline together with a compatible `custom_generate_fn` that XWhy can call when producing the reference and perturbed outputs. The explainer then applies its perturbation, output-distance, and surrogate-modelling workflow around that interface.

See the [image generation and editing guide](../explainers/image-generation/index.md).

## Tabular models

`TabularExplainer` is available for structured classification and regression. Pass the trained black-box model to the explainer and ensure its prediction interface is compatible with the XWhy tabular adapter. Use the same preprocessing and feature scaling used by the model during normal inference.

See the [tabular explainer guide](../explainers/tabular.md).

## Text models

`TextExplainer` accepts either a model exposing `predict_proba`, `predict`, or `__call__`, or a direct `predict_fn` that accepts a sequence of texts and returns predictions or scores.

See the [text explainer guide](../explainers/text.md).

## Development and roadmap modalities

!!! warning "Not yet supported end to end"
    `PointCloudExplainer` remains a development interface. Time Series and Multimodal explainability are planned capabilities. Agentic AI and Multi-Agent AI are currently research-roadmap areas rather than exported explainers.
