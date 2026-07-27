---
title: Choose an XWhy Explainer
description: Select an XWhy explainer by input modality, task, model interface, and current implementation status.
---

# Choose an explainer

| Your input and task | Recommended component | Current status | Next page |
| --- | --- | --- | --- |
| PyTorch image classification | `ImageClassificationExplainer` | Available | [Image explainer](../explainers/image/index.md) |
| LLM prompt-response behaviour | `LLMExplainer` | Available | [LLM explainer](../explainers/llm/index.md) |
| Structured tabular prediction | `TabularExplainer` | Under construction | [Tabular status](../explainers/tabular.md) |
| Conventional text classification | `TextExplainer` | Under construction | [Text status](../explainers/text.md) |
| 3D point-cloud prediction | `PointCloudExplainer` | Under construction | [Point-cloud status](../explainers/point-cloud.md) |
| Image-to-image transformation | `Pix2PixExplainer` | Experimental interface | [Pix2Pix status](../explainers/pix2pix.md) |
| Time-series prediction | Planned | Coming soon | [Time-series roadmap](../explainers/time-series.md) |
| Cross-modal model | Planned | Coming soon | [Multimodal roadmap](../explainers/multimodal.md) |

!!! tip
    Choose by **task and implemented capability**, not only by class name. A class exported by the package may still be a development interface.
