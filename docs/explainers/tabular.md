---
title: XWhy Tabular Explainer
description: Use XWhy TabularExplainer for model-agnostic local explanations of tabular classification and regression predictions.
---

# Tabular explainer

!!! success "Available"
    `TabularExplainer` is implemented and exported by XWhy for structured classification and regression workflows.

`TabularExplainer` applies the SMILE workflow to a selected structured instance. It creates a local distribution around the instance, generates perturbation samples, queries the black-box model, measures distributional distance feature by feature, and fits a weighted surrogate model to estimate local feature influence.

## Basic use

```python
from xwhy import TabularExplainer

explainer = TabularExplainer(
    model,
    mode="classification",
    num_perturbations=500,
)

result = explainer.explain(
    instance,
    feature_names=feature_names,
)
```

For regression, set `mode="regression"`.

## Current behaviour

The current implementation supports:

- classification and regression modes;
- array-like input instances;
- configurable perturbation counts and random seed;
- local Gaussian feature distributions;
- configurable distance metrics, with Wasserstein distance as the default;
- configurable surrogate models and automatic surrogate selection;
- surrogate coefficients as local feature contributions;
- regression-style fidelity metrics for the local surrogate;
- optional fidelity plotting through `fidelity_plot=True`.

## Input scaling

The implementation is designed for appropriately scaled structured features. With normalization validation enabled, XWhy warns when an input appears substantially out of scale. Use a preprocessing pipeline that is consistent with the model being explained and report that preprocessing when publishing results.

## Interpretation

The returned coefficients describe the fitted local surrogate around the selected instance and perturbation distribution. They should not be interpreted as global feature importance or causal effects.

For reproducible research, report the perturbation count, seed, distance metric, surrogate configuration, preprocessing, and fidelity metrics together with the explanation.

[View the current API reference](../reference/xwhy/explainers/tabular.md)
