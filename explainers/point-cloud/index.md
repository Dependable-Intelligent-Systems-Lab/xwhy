# Point-cloud explainer

Available

`PointCloudExplainer` is implemented and exported by XWhy for 3D point-cloud model workflows.

`PointCloudExplainer` is intended for 3D point-cloud classifiers (such as PointNet). It clusters the input point cloud into spatial regions, perturbs these regions by removing them, queries the black-box prediction function, measures distance, and fits a weighted local surrogate model to attribute importance to each cluster.

## Basic use

```
from xwhy import PointCloudExplainer

explainer = PointCloudExplainer(
    model=model,
    num_clusters=32,
    num_perturbations=500,
    removal_probability=0.5,
    clustering_mode="kmeans",
    distance_type="wasserstein",
)

result = explainer.explain(
    instance=sample_input,
    sample_label=sample_label,
)
```

## Current behaviour

The current implementation supports:

- tensor inputs for 3D point clouds;
- model or direct prediction-function interfaces (including Hugging Face wrappers via `CustomPointCloudModel`);
- configurable perturbation counts and removal probabilities;
- k-means and other spatial clustering modes;
- configurable distance metrics (e.g., `wasserstein`, `cosine`, `ks`, `cramer_von_mises`), with Wasserstein distance as the default;
- configurable surrogate models and automatic surrogate selection;
- cluster-level surrogate coefficients;
- 3D attribution visualization and Jaccard stability scoring;

## Interpretation

Cluster coefficients describe a local surrogate approximation around the selected point cloud and perturbation strategy. They do not reveal hidden reasoning and should not be treated as causal effects.

For reproducible use, report the perturbation count, clustering configuration, distance configuration, target class, surrogate configuration, random seed, and stability metrics.

[View the current API reference](https://dependable-intelligent-systems-lab.github.io/xwhy/explainers/reference/xwhy/explainers/pointcloud.md)
