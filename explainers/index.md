# XWhy explainers

This page is the authoritative capability map for the current documentation release.

| Explainer area                                                                                                     | Input                                                   | Intended task                  | Status                 |
| ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------- | ------------------------------ | ---------------------- |
| [Image Classification](https://dependable-intelligent-systems-lab.github.io/xwhy/explainers/image/index.md)        | Image                                                   | Classification                 | **Available**          |
| [Image Generation](https://dependable-intelligent-systems-lab.github.io/xwhy/explainers/image-generation/index.md) | Prompt, source image, mask, or other conditioning input | Generation and image editing   | **Under construction** |
| [LLM](https://dependable-intelligent-systems-lab.github.io/xwhy/explainers/llm/index.md)                           | Prompt and generated response                           | Local prompt influence         | **Available**          |
| [Tabular](https://dependable-intelligent-systems-lab.github.io/xwhy/explainers/tabular/index.md)                   | Structured features                                     | Classification and regression  | **Under construction** |
| [Text](https://dependable-intelligent-systems-lab.github.io/xwhy/explainers/text/index.md)                         | Tokens, words, or phrases                               | Text prediction                | **Under construction** |
| [Point Cloud](https://dependable-intelligent-systems-lab.github.io/xwhy/explainers/point-cloud/index.md)           | 3D points or point groups                               | 3D prediction                  | **Under construction** |
| [Time Series](https://dependable-intelligent-systems-lab.github.io/xwhy/explainers/time-series/index.md)           | Ordered observations                                    | Classification and forecasting | **Coming soon**        |
| [Multimodal](https://dependable-intelligent-systems-lab.github.io/xwhy/explainers/multimodal/index.md)             | Two or more modalities                                  | Cross-modal prediction         | **Coming soon**        |

The image-generation area contains planned [image-editing documentation](https://dependable-intelligent-systems-lab.github.io/xwhy/explainers/image-generation/image-editing/index.md) and a [Pix2Pix model example](https://dependable-intelligent-systems-lab.github.io/xwhy/explainers/image-generation/pix2pix-models/index.md). The current `Pix2PixExplainer` class is an early interface and should not be treated as an implemented workflow.

## Status definitions

**Available** means an executable implementation and dedicated guide exist in the repository.

**Under construction** means a public interface or planned capability is present, but a supported end-to-end workflow is not yet available.

**Experimental interface** refers to an early code interface, such as the current `Pix2PixExplainer`, that should not yet be treated as a supported workflow.

**Coming soon** means the documentation reserves a stable location for a planned capability that is not currently implemented.
