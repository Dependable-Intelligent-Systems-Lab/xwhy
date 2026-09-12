---
title: XWhy Explainers
description: Compare available, under-construction, and planned XWhy explainers across image classification, image generation and editing, LLM, text, tabular, point-cloud, time-series, multimodal, agentic AI, and multi-agent AI systems.
---

# XWhy explainers

This page is the authoritative capability map for the current documentation release.

| Explainer area | Input or system context | Intended explanation task | Status |
| --- | --- | --- | --- |
| [Image Classification](image/index.md) | Image | Local image-region influence on classification | **Available** |
| [Image Generation & Editing](image-generation/index.md) | Prompt, optional source image, and other model conditioning | Prompt and conditioning influence on generated or edited images | **Available** |
| [LLM](llm/index.md) | Prompt and generated response | Local prompt influence on response behaviour | **Available** |
| [Tabular](tabular.md) | Structured features | Feature influence on classification and regression | **Available** |
| [Text](text.md) | Text input | Word-level influence on conventional text-model predictions | **Available** |
| [Point Cloud](point-cloud.md) | 3D points or point groups | Local influence on 3D prediction | **Under construction** |
| [Time Series](time-series.md) | Ordered observations | Influence of observations, windows, and temporal patterns | **Coming soon** |
| [Multimodal](multimodal.md) | Two or more modalities | Modality-specific and cross-modal contributions | **Coming soon** |
| [Agentic AI](agentic-ai.md) | Agent trajectory including plans, retrieval, memory, tools, states, and actions | Explain decisions, actions, state transitions, and uncertainty or failure propagation within an autonomous agent | **Coming soon** |
| [Multi-Agent AI](multi-agent-ai.md) | Interacting agents, messages, roles, dependencies, and shared state | Explain agent contribution, communication, coordination, disagreement, and uncertainty or failure propagation across agents | **Coming soon** |

The current generative-image API is `ImageGenerationAndEditingExplainer`. The [Pix2Pix page](image-generation/pix2pix-models.md) describes Pix2Pix-style models as one conditional image-to-image model family that can be considered within the broader generation and editing capability.

Agentic AI and Multi-Agent AI are currently research-roadmap areas rather than exported explainers. Their documentation defines intended explanation targets and evaluation principles without implying that executable implementations already exist.

## Status definitions

**Available** means an executable implementation and dedicated guide exist in the repository.

**Under construction** means a public interface is present, but a supported end-to-end workflow is not yet available.

**Coming soon** means the documentation reserves a stable location for a planned capability that is not currently implemented.
