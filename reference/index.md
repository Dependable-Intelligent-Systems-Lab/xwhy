# API Reference

This page is generated from the public objects declared in `src/xwhy/**/__init__.py`. Adding or removing an object from a package's `__all__` updates this index during the next documentation build.

Select an object for its full signature, parameters, return values, and documented members.

## Explainers

| Object                                                                                                                                                                      | Description                                                                |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| [`xwhy.ImageClassificationExplainer`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/ImageClassificationExplainer/index.md)             | Explainer for image classification models.                                 |
| [`xwhy.ImageGenerationAndEditingExplainer`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/ImageGenerationAndEditingExplainer/index.md) | Explainer for image generation and editing tasks.                          |
| [`xwhy.LLMExplainer`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/LLMExplainer/index.md)                                             | Explainer for LLM tasks integrating the full GSMILE pipeline.              |
| [`xwhy.PointCloudExplainer`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/PointCloudExplainer/index.md)                               | Explainer for Pointcloud tasks.                                            |
| [`xwhy.TabularExplainer`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/TabularExplainer/index.md)                                     | Explainer for Tabular models utilizing the SMILE algorithm.                |
| [`xwhy.TextExplainer`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/TextExplainer/index.md)                                           | Explainer for natural language processing (NLP) text classification tasks. |

## Core abstractions and results

| Object                                                                                                                                                              | Description                                             |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| [`xwhy.core.BaseExplainer`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/core/BaseExplainer/index.md)                         | Abstract base class for all xwhy explainers.            |
| [`xwhy.core.BaseXWhyResult`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/core/BaseXWhyResult/index.md)                       | Abstract base container for shared explanation results. |
| [`xwhy.core.ExplainerConfig`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/core/ExplainerConfig/index.md)                     | Explainer config.                                       |
| [`xwhy.core.ExplanationPipeline`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/core/ExplanationPipeline/index.md)             | Abstract pipeline orchestrator for explanation process. |
| [`xwhy.core.ImageClassificationConfig`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/core/ImageClassificationConfig/index.md) | Configuration for the Image Classification explainer.   |
| [`xwhy.core.LLMConfig`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/core/LLMConfig/index.md)                                 | Configuration for the LLM explainer.                    |
| [`xwhy.core.TabularConfig`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/core/TabularConfig/index.md)                         | Configuration for the Tabular explainer.                |
| [`xwhy.core.XWhyError`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/core/XWhyError/index.md)                                 | Base exception for xwhy package.                        |

## Plots

| Object                                                                                                                                                                  | Description                                                                |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| [`xwhy.plots.bar`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/plots/bar/index.md)                                               | Create a bar plot of a set of XWhy values.                                 |
| [`xwhy.plots.BaseTextPlotter`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/plots/BaseTextPlotter/index.md)                       | Abstract base class for text plots.                                        |
| [`xwhy.plots.beeswarm`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/plots/beeswarm/index.md)                                     | Create a beeswarm plot (requires multiple instances/2D data).              |
| [`xwhy.plots.decision`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/plots/decision/index.md)                                     | Visualize model decisions using cumulative XWhy values.                    |
| [`xwhy.plots.embedding`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/plots/embedding/index.md)                                   | Use the XWhy values as an embedding projected to 2D (requires 2D data).    |
| [`xwhy.plots.Explanation`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/plots/Explanation/index.md)                               | Container for attribution values, mirroring `shap.Explanation`.            |
| [`xwhy.plots.force`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/plots/force/index.md)                                           | Visualize the given XWhy values with an additive force layout.             |
| [`xwhy.plots.group_difference`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/plots/group_difference/index.md)                     | Plot the difference in mean XWhy values between two groups (2D data).      |
| [`xwhy.plots.heatmap`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/plots/heatmap/index.md)                                       | Create a heatmap plot (requires multiple instances/2D data).               |
| [`xwhy.plots.image`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/plots/image/index.md)                                           | Plot XWhy values for image inputs.                                         |
| [`xwhy.plots.image_heatmap`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/plots/image_heatmap/index.md)                           | Plot a heatmap of feature importance over image superpixels.               |
| [`xwhy.plots.image_to_text`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/plots/image_to_text/index.md)                           | Plot XWhy values for image inputs with text outputs.                       |
| [`xwhy.plots.initjs`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/plots/initjs/index.md)                                         | Do nothing; kept so SHAP-style notebooks keep running unchanged.           |
| [`xwhy.plots.monitoring`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/plots/monitoring/index.md)                                 | Create a monitoring plot over time or indices (requires 2D data).          |
| [`xwhy.plots.NativeHeatmapPlotter`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/plots/NativeHeatmapPlotter/index.md)             | Native matplotlib implementation of text heatmap plot.                     |
| [`xwhy.plots.partial_dependence`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/plots/partial_dependence/index.md)                 | Plot the partial dependence of a model on a single feature.                |
| [`xwhy.plots.plot_dataset`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/plots/plot_dataset/index.md)                             | Plot dataset or single point with flexible matplotlib-style arguments.     |
| [`xwhy.plots.plot_explanation_waterfall`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/plots/plot_explanation_waterfall/index.md) | Create a dynamic waterfall plot for explanation method coefficients.       |
| [`xwhy.plots.plot_feature_bar_chart`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/plots/plot_feature_bar_chart/index.md)         | Generate and optionally save a Plotly bar chart for feature contributions. |
| [`xwhy.plots.plot_feature_box_plot`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/plots/plot_feature_box_plot/index.md)           | Generate and optionally save a Plotly box plot for feature contributions.  |
| [`xwhy.plots.plot_feature_contributions`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/plots/plot_feature_contributions/index.md) | Visualize feature contributions using a horizontal bar chart.              |
| [`xwhy.plots.plot_image`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/plots/plot_image/index.md)                                 | Display an image (Tensor, Numpy, PIL, or Path).                            |
| [`xwhy.plots.plot_method_contributions`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/plots/plot_method_contributions/index.md)   | Visualize feature contributions for a given explanation method.            |
| [`xwhy.plots.scatter`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/plots/scatter/index.md)                                       | Create a dependence scatter plot (requires multiple instances/2D data).    |
| [`xwhy.plots.text`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/plots/text/index.md)                                             | Plot a text explanation using coloured, self-contained HTML.               |
| [`xwhy.plots.text_heatmap`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/plots/text_heatmap/index.md)                             | Plot a heatmap visualization for the given explanation result.             |
| [`xwhy.plots.TextPlotterFactory`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/plots/TextPlotterFactory/index.md)                 | Factory for creating text visualization instances.                         |
| [`xwhy.plots.TextPlotterType`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/plots/TextPlotterType/index.md)                       | Enumeration for supported text plot backends.                              |
| [`xwhy.plots.violin`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/plots/violin/index.md)                                         | Create a violin plot (requires multiple instances/2D data).                |
| [`xwhy.plots.waterfall`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/plots/waterfall/index.md)                                   | Plot an explanation of a single prediction as a waterfall plot.            |

## Distances

| Object                                                                                                                                                                  | Description                                                             |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| [`xwhy.distance.AndersonDarlingDistance`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/distance/AndersonDarlingDistance/index.md) | Anderson-Darling distance metric (Custom Implementation).               |
| [`xwhy.distance.BaseDistance`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/distance/BaseDistance/index.md)                       | Abstract base class for unified distance implementations.               |
| [`xwhy.distance.BaseNumericDistance`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/distance/BaseNumericDistance/index.md)         | Base class for handling dimensionality of numerical distances.          |
| [`xwhy.distance.CosineDistance`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/distance/CosineDistance/index.md)                   | Cosine distance metric.                                                 |
| [`xwhy.distance.CvMDistance`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/distance/CvMDistance/index.md)                         | Cramer-Von Mises distance metric (Custom Implementation).               |
| [`xwhy.distance.DistanceNormalizer`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/distance/DistanceNormalizer/index.md)           | Normalize distance values into similarity scores.                       |
| [`xwhy.distance.DistanceType`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/distance/DistanceType/index.md)                       | Enumeration for supported distance metrics.                             |
| [`xwhy.distance.DTSDistance`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/distance/DTSDistance/index.md)                         | DTS distance metric (Custom Implementation: Combination of AD and CVM). |
| [`xwhy.distance.KSDistance`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/distance/KSDistance/index.md)                           | Kolmogorov-Smirnov distance metric (Custom Implementation).             |
| [`xwhy.distance.KuiperDistance`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/distance/KuiperDistance/index.md)                   | Kuiper distance metric (Custom Implementation).                         |
| [`xwhy.distance.WassersteinDistance`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/distance/WassersteinDistance/index.md)         | Wasserstein distance metric (Custom Implementation).                    |
| [`xwhy.distance.WMDDistance`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/distance/WMDDistance/index.md)                         | Word Mover's Distance implementation for Text Data.                     |

## Models

| Object                                                                                                                                                                  | Description                                                    |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| [`xwhy.models.BaseClassification`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/models/BaseClassification/index.md)               | Base class for all classification implementations.             |
| [`xwhy.models.BaseEmbedding`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/models/BaseEmbedding/index.md)                         | Base class for all embedding implementations.                  |
| [`xwhy.models.BaseSegmentation`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/models/BaseSegmentation/index.md)                   | Base class for all segmentation implementations.               |
| [`xwhy.models.ClassificationFactory`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/models/ClassificationFactory/index.md)         | Manage classification model instantiation via a registry.      |
| [`xwhy.models.ClassificationType`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/models/ClassificationType/index.md)               | Supported classification backends.                             |
| [`xwhy.models.EmbeddingFactory`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/models/EmbeddingFactory/index.md)                   | Manage embedding model instantiation via a registry.           |
| [`xwhy.models.EmbeddingType`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/models/EmbeddingType/index.md)                         | Supported embedding backends.                                  |
| [`xwhy.models.SegmentationFactory`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/models/SegmentationFactory/index.md)             | Manage segmentation model instantiation via a registry.        |
| [`xwhy.models.SegmentationType`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/models/SegmentationType/index.md)                   | Supported segmentation backends.                               |
| [`xwhy.models.TabularModelAdapter`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/models/TabularModelAdapter/index.md)             | Wrap tabular models to provide a unified prediction interface. |
| [`xwhy.models.TorchvisionClassification`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/models/TorchvisionClassification/index.md) | Classification backend for standard torchvision models.        |
| [`xwhy.models.TorchvisionSegmentation`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/models/TorchvisionSegmentation/index.md)     | Segmentation backend for standard torchvision models.          |
| [`xwhy.models.Word2VecEmbedding`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/models/Word2VecEmbedding/index.md)                 | Word2Vec embedding backend.                                    |

## Providers

| Object                                                                                                                                                      | Description                                                     |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| [`xwhy.providers.BaseProvider`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/providers/BaseProvider/index.md)         | Abstract interface for external AI providers.                   |
| [`xwhy.providers.OpenAIProvider`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/providers/OpenAIProvider/index.md)     | OpenAI implementation of the provider interface.                |
| [`xwhy.providers.ProviderFactory`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/providers/ProviderFactory/index.md)   | Factory for provider implementations.                           |
| [`xwhy.providers.ProviderResolver`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/providers/ProviderResolver/index.md) | Resolver mapping provider types to default instantiation logic. |
| [`xwhy.providers.ProviderType`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/providers/ProviderType/index.md)         | Supported provider types.                                       |

## Perturbation

| Object                                                                                                                                                              | Description                                                                |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| [`xwhy.perturbation.BasePerturbation`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/perturbation/BasePerturbation/index.md)   | Abstract base class for perturbation strategies.                           |
| [`xwhy.perturbation.ImagePerturbation`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/perturbation/ImagePerturbation/index.md) | Perturbation strategy for images using superpixels and Bernoulli sampling. |
| [`xwhy.perturbation.TextPerturbation`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/perturbation/TextPerturbation/index.md)   | Generate binary perturbations for text.                                    |

## Surrogate models

| Object                                                                                                                                                                        | Description                                                             |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| [`xwhy.surrogate.BaseSurrogate`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/surrogate/BaseSurrogate/index.md)                         | Abstract base class for all surrogate models.                           |
| [`xwhy.surrogate.LinearRegressionSurrogate`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/surrogate/LinearRegressionSurrogate/index.md) | Surrogate wrapper for linear models like OLS and Ridge.                 |
| [`xwhy.surrogate.SurrogateFactory`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/surrogate/SurrogateFactory/index.md)                   | Factory class for instantiating surrogate models.                       |
| [`xwhy.surrogate.SurrogateTrainer`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/surrogate/SurrogateTrainer/index.md)                   | Service for training and evaluating surrogate models.                   |
| [`xwhy.surrogate.SurrogateType`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/surrogate/SurrogateType/index.md)                         | Enumeration for supported surrogate model types.                        |
| [`xwhy.surrogate.TreeBasedSurrogate`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/surrogate/TreeBasedSurrogate/index.md)               | Surrogate wrapper for tree-based models like Random Forest and XGBoost. |

## Metrics

| Object                                                                                                                                                                    | Description                                                        |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| [`xwhy.metrics.calculate_stability_score`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/metrics/calculate_stability_score/index.md) | Calculate stability metrics between two explanation results.       |
| [`xwhy.metrics.calculate_token_auc`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/metrics/calculate_token_auc/index.md)             | Calculate the Area Under the ROC Curve (AUC) for token importance. |
| [`xwhy.metrics.ImageCoverageMetrics`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/metrics/ImageCoverageMetrics/index.md)           | Metrics for evaluating image explanation coverage.                 |
| [`xwhy.metrics.RegressionMetricResult`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/metrics/RegressionMetricResult/index.md)       | Data container for regression evaluation metrics.                  |
| [`xwhy.metrics.RegressionMetrics`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/metrics/RegressionMetrics/index.md)                 | Utility for calculating comprehensive regression metrics.          |

## Configuration

| Object                                                                                                                                | Description                        |
| ------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| [`xwhy.config.Settings`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/config/Settings/index.md) | Global application settings.       |
| [`xwhy.settings`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/settings/index.md)               | Public XWhy API object `settings`. |

## Utilities

| Object                                                                                                                                                        | Description                                                     |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| [`xwhy.utils.denormalize_tensor`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/utils/denormalize_tensor/index.md)       | Reverse the normalization applied to an image tensor.           |
| [`xwhy.utils.load_image_as_tensor`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/utils/load_image_as_tensor/index.md)   | Load an image from disk and apply preprocessing transforms.     |
| [`xwhy.utils.numpy_image_to_tensor`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/utils/numpy_image_to_tensor/index.md) | Preprocess a numpy image array using the specified transforms.  |
| [`xwhy.utils.tensor_to_numpy_image`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/utils/tensor_to_numpy_image/index.md) | Convert a batch tensor (1, C, H, W) to a NumPy image (H, W, C). |

## Datasets

| Object                                                                                                                                                                      | Description                                                     |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| [`xwhy.datasets.download_i2ebench_dataset`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/datasets/download_i2ebench_dataset/index.md) | Download the I2EBench dataset from Google Drive and extract it. |
| [`xwhy.datasets.load_i2ebench_data`](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/generated/xwhy/datasets/load_i2ebench_data/index.md)               | Parse the I2EBench dataset with limits and file validation.     |

______________________________________________________________________

Module-level documentation is also generated under `reference/xwhy/` for maintainers and existing deep links, but it is intentionally excluded from the public API index.
