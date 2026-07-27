---
title: XWhy Explainers
description: Compare available, under-construction, experimental, and planned XWhy explainers across image, LLM, text, tabular, point-cloud, time-series, and multimodal data.
---

# XWhy explainers

This page is the authoritative capability map for the current documentation release.

| Explainer | Input | Intended task | Status |
| --- | --- | --- | --- |
| [Image Classification](image/index.md) | Image | Classification | **Available** |
| [LLM](llm/index.md) | Prompt and generated response | Local prompt influence | **Available** |
| [Tabular](tabular.md) | Structured features | Classification and regression | **Under construction** |
| [Text](text.md) | Tokens, words, or phrases | Text prediction | **Under construction** |
| [Point Cloud](point-cloud.md) | 3D points or point groups | 3D prediction | **Under construction** |
| [Pix2Pix](pix2pix.md) | Image or instruction-conditioned transformation | Image-to-image explanation | **Experimental interface** |
| [Time Series](time-series.md) | Ordered observations | Classification and forecasting | **Coming soon** |
| [Multimodal](multimodal.md) | Two or more modalities | Cross-modal prediction | **Coming soon** |

## Status definitions

**Available** means an executable implementation and dedicated guide exist in the repository.

**Under construction** means a public class exists, but its current `explain()` method is not implemented.

**Experimental interface** means an early public interface exists, but it should not yet be treated as a supported workflow.

**Coming soon** means the documentation reserves a stable location for a planned capability that is not currently implemented.
