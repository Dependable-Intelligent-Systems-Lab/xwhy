# XWhy explainers

This page is the authoritative capability map for the current documentation release.

| Explainer area                                                                                                               | Input or system context                                                         | Intended explanation task                                                                                                   | Status                 |
| ---------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ---------------------- |
| [Image Classification](https://dependable-intelligent-systems-lab.github.io/xwhy/explainers/image/index.md)                  | Image                                                                           | Local image-region influence on classification                                                                              | **Available**          |
| [Image Generation & Editing](https://dependable-intelligent-systems-lab.github.io/xwhy/explainers/image-generation/index.md) | Prompt, optional source image, and other model conditioning                     | Prompt and conditioning influence on generated or edited images                                                             | **Available**          |
| [LLM](https://dependable-intelligent-systems-lab.github.io/xwhy/explainers/llm/index.md)                                     | Prompt and generated response                                                   | Local prompt influence on response behaviour                                                                                | **Available**          |
| [Tabular](https://dependable-intelligent-systems-lab.github.io/xwhy/explainers/tabular/index.md)                             | Structured features                                                             | Feature influence on classification and regression                                                                          | **Available**          |
| [Text](https://dependable-intelligent-systems-lab.github.io/xwhy/explainers/text/index.md)                                   | Text input                                                                      | Word-level influence on conventional text-model predictions                                                                 | **Available**          |
| [Point Cloud](https://dependable-intelligent-systems-lab.github.io/xwhy/explainers/point-cloud/index.md)                     | 3D points or point groups                                                       | Local influence on 3D prediction                                                                                            | **Under construction** |
| [Time Series](https://dependable-intelligent-systems-lab.github.io/xwhy/explainers/time-series/index.md)                     | Ordered observations                                                            | Influence of observations, windows, and temporal patterns                                                                   | **Coming soon**        |
| [Multimodal](https://dependable-intelligent-systems-lab.github.io/xwhy/explainers/multimodal/index.md)                       | Two or more modalities                                                          | Modality-specific and cross-modal contributions                                                                             | **Coming soon**        |
| [Agentic AI](https://dependable-intelligent-systems-lab.github.io/xwhy/explainers/agentic-ai/index.md)                       | Agent trajectory including plans, retrieval, memory, tools, states, and actions | Explain decisions, actions, state transitions, and uncertainty or failure propagation within an autonomous agent            | **Coming soon**        |
| [Multi-Agent AI](https://dependable-intelligent-systems-lab.github.io/xwhy/explainers/multi-agent-ai/index.md)               | Interacting agents, messages, roles, dependencies, and shared state             | Explain agent contribution, communication, coordination, disagreement, and uncertainty or failure propagation across agents | **Coming soon**        |

The current generative-image API is `ImageGenerationAndEditingExplainer`. The [Pix2Pix page](https://dependable-intelligent-systems-lab.github.io/xwhy/explainers/image-generation/pix2pix-models/index.md) describes Pix2Pix-style models as one conditional image-to-image model family that can be considered within the broader generation and editing capability.

Agentic AI and Multi-Agent AI are currently research-roadmap areas rather than exported explainers. Their documentation defines intended explanation targets and evaluation principles without implying that executable implementations already exist.

## Status definitions

**Available** means an executable implementation and dedicated guide exist in the repository.

**Under construction** means a public interface is present, but a supported end-to-end workflow is not yet available.

**Coming soon** means the documentation reserves a stable location for a planned capability that is not currently implemented.
