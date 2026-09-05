---
title: XWhy Explainers
description: Compare available, under-construction, experimental, and planned XWhy explainers across image classification, image generation, LLM, text, tabular, point-cloud, time-series, multimodal, agentic AI, and multi-agent AI systems.
---

# XWhy explainers

This page is the authoritative capability map for the current documentation release.

| Explainer area | Input or system context | Intended explanation task | Status |
| --- | --- | --- | --- |
| [Image Classification](image/index.md) | Image | Local image-region influence on classification | **Available** |
| [Image Generation](image-generation/index.md) | Prompt, source image, mask, or other conditioning input | Influence on generation and image editing | **Under construction** |
| [LLM](llm/index.md) | Prompt and generated response | Local prompt influence on response behaviour | **Available** |
| [Tabular](tabular.md) | Structured features | Feature influence on classification and regression | **Under construction** |
| [Text](text.md) | Tokens, words, or phrases | Local influence on text prediction | **Under construction** |
| [Point Cloud](point-cloud.md) | 3D points or point groups | Local influence on 3D prediction | **Under construction** |
| [Time Series](time-series.md) | Ordered observations | Influence of observations, windows, and temporal patterns | **Coming soon** |
| [Multimodal](multimodal.md) | Two or more modalities | Modality-specific and cross-modal contributions | **Coming soon** |
| [Agentic AI](agentic-ai.md) | Agent trajectory including plans, retrieval, memory, tools, states, and actions | Explain decisions, actions, state transitions, and uncertainty or failure propagation within an autonomous agent | **Coming soon** |
| [Multi-Agent AI](multi-agent-ai.md) | Interacting agents, messages, roles, dependencies, and shared state | Explain agent contribution, communication, coordination, disagreement, and uncertainty or failure propagation across agents | **Coming soon** |

The image-generation area contains planned [image-editing documentation](image-generation/image-editing.md) and a [Pix2Pix model example](image-generation/pix2pix-models.md). The current `Pix2PixExplainer` class is an early interface and should not be treated as an implemented workflow.

Agentic AI and Multi-Agent AI are currently research-roadmap areas rather than exported explainers. Their documentation defines the intended explanation targets and evaluation principles without implying that executable implementations already exist.

## Status definitions

**Available** means an executable implementation and dedicated guide exist in the repository.

**Under construction** means a public interface or planned capability is present, but a supported end-to-end workflow is not yet available.

**Experimental interface** refers to an early code interface, such as the current `Pix2PixExplainer`, that should not yet be treated as a supported workflow.

**Coming soon** means the documentation reserves a stable location for a planned capability that is not currently implemented.
