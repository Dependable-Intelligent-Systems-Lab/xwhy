---
title: Choose an XWhy Explainer
description: Select an XWhy explainer by input modality, task, model interface, system architecture, and current implementation status.
---

# Choose an explainer

| Your input, task, or system | Recommended component | Current status | Next page |
| --- | --- | --- | --- |
| PyTorch image classification | `ImageClassificationExplainer` | Available | [Image explainer](../explainers/image/index.md) |
| Image generation or image editing | `ImageGenerationAndEditingExplainer` | Available | [Image generation & editing](../explainers/image-generation/index.md) |
| LLM prompt-response behaviour | `LLMExplainer` | Available | [LLM explainer](../explainers/llm/index.md) |
| Structured tabular prediction | `TabularExplainer` | Available | [Tabular explainer](../explainers/tabular.md) |
| Conventional text classification | `TextExplainer` | Available | [Text explainer](../explainers/text.md) |
| 3D point-cloud prediction | `PointCloudExplainer` | Under construction | [Point-cloud status](../explainers/point-cloud.md) |
| Time-series prediction | Planned | Coming soon | [Time-series roadmap](../explainers/time-series.md) |
| Cross-modal model | Planned | Coming soon | [Multimodal roadmap](../explainers/multimodal.md) |
| Autonomous agent using planning, retrieval, memory, tools, or iterative actions | Planned Agentic AI explainability | Coming soon | [Agentic AI roadmap](../explainers/agentic-ai.md) |
| System of interacting AI agents | Planned Multi-Agent AI explainability | Coming soon | [Multi-Agent AI roadmap](../explainers/multi-agent-ai.md) |

!!! tip
    Choose by **task and implemented capability**, not only by class name. `PointCloudExplainer` is currently an exported development interface whose `explain()` method is not implemented. Pix2Pix is one image-to-image model family, not the public explainer name. Agentic AI and Multi-Agent AI are currently documented research directions, not exported XWhy explainers.
