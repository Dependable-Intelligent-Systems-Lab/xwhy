# Choose an explainer

| Your input and task               | Recommended component                                          | Current status     | Next page                                                                                                                 |
| --------------------------------- | -------------------------------------------------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| PyTorch image classification      | `ImageClassificationExplainer`                                 | Available          | [Image explainer](https://dependable-intelligent-systems-lab.github.io/xwhy/explainers/image/index.md)                    |
| Image generation or image editing | Image-generation roadmap; current `Pix2PixExplainer` prototype | Under construction | [Image-generation status](https://dependable-intelligent-systems-lab.github.io/xwhy/explainers/image-generation/index.md) |
| LLM prompt-response behaviour     | `LLMExplainer`                                                 | Available          | [LLM explainer](https://dependable-intelligent-systems-lab.github.io/xwhy/explainers/llm/index.md)                        |
| Structured tabular prediction     | `TabularExplainer`                                             | Under construction | [Tabular status](https://dependable-intelligent-systems-lab.github.io/xwhy/explainers/tabular/index.md)                   |
| Conventional text classification  | `TextExplainer`                                                | Under construction | [Text status](https://dependable-intelligent-systems-lab.github.io/xwhy/explainers/text/index.md)                         |
| 3D point-cloud prediction         | `PointCloudExplainer`                                          | Under construction | [Point-cloud status](https://dependable-intelligent-systems-lab.github.io/xwhy/explainers/point-cloud/index.md)           |
| Time-series prediction            | Planned                                                        | Coming soon        | [Time-series roadmap](https://dependable-intelligent-systems-lab.github.io/xwhy/explainers/time-series/index.md)          |
| Cross-modal model                 | Planned                                                        | Coming soon        | [Multimodal roadmap](https://dependable-intelligent-systems-lab.github.io/xwhy/explainers/multimodal/index.md)            |

Tip

Choose by **task and implemented capability**, not only by class name. A class exported by the package may still be a development interface. Pix2Pix is one image-editing model family, not the name of the overall image-generation capability.
