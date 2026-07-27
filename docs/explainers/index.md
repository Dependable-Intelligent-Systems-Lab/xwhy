---
title: XWhy Explainers
description: Compare available, under-construction, experimental, and planned XWhy explainers across image classification, image generation, LLM, text, tabular, point-cloud, time-series, and multimodal data.
---

# XWhy explainers

This page is the authoritative capability map for the current documentation release.

| Explainer area | Input | Intended task | Status |
| --- | --- | --- | --- |
| [Image Classification](image/index.md) | Image | Classification | **Available** |
| [Image Generation](image-generation/index.md) | Prompt, source image, mask, or other conditioning input | Generation and image editing | **Under construction** |
| [LLM](llm/index.md) | Prompt and generated response | Local prompt influence | **Available** |
| [Tabular](tabular.md) | Structured features | Classification and regression | **Under construction** |
| [Text](text.md) | Tokens, words, or phrases | Text prediction | **Under construction** |
| [Point Cloud](point-cloud.md) | 3D points or point groups | 3D prediction | **Under construction** |
| [Time Series](time-series.md) | Ordered observations | Classification and forecasting | **Coming soon** |
| [Multimodal](multimodal.md) | Two or more modalities | Cross-modal prediction | **Coming soon** |

The image-generation area contains planned [image-editing documentation](image-generation/image-editing.md) and a [Pix2Pix model example](image-generation/pix2pix-models.md). The current `Pix2PixExplainer` class is an early interface and should not be treated as an implemented workflow.

## Status definitions

**Available** means an executable implementation and dedicated guide exist in the repository.

**Under construction** means a public interface or planned capability is present, but a supported end-to-end workflow is not yet available.

**Experimental interface** refers to an early code interface, such as the current `Pix2PixExplainer`, that should not yet be treated as a supported workflow.

**Coming soon** means the documentation reserves a stable location for a planned capability that is not currently implemented.
