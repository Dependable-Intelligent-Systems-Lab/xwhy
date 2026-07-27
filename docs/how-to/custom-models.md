---
title: Connect a Custom Model to XWhy
description: Current and planned custom-model integration guidance for XWhy explainers.
---

# Connect a custom model

## Image classification

Custom PyTorch image classifiers are currently documented. Supply the model, its matching preprocessing pipeline, and optional category labels.

[Follow the custom PyTorch model guide](../image_classification_explainer.md#using-a-custom-pytorch-model)

## LLM providers

The LLM explainer accepts built-in provider identifiers and provider-specific client arguments. See [provider configuration](providers.md).

## Other modalities

!!! warning "Under construction"
    Stable adapter contracts for tabular, text, point-cloud, Pix2Pix, time-series, and multimodal models will be documented after their implementations are available.
