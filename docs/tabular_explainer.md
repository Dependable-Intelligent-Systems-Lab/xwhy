# Tabular Explainer

Tabular machine learning models can be difficult to interpret because their predictions depend on complex interactions between multiple features.

The **XWhy Tabular Explainer** provides a model-agnostic way to examine a tabular model as a black-box system. It perturbs the input instance, observes how the model prediction changes, and estimates which features had the strongest local influence on the generated output.

This guide starts with the simplest setup using a trained tabular model. Advanced configuration for custom distances, surrogates, and configurations is provided later.

> **Important:** The tabular explainer does not strictly require standardized or normalized input data for optimal perturbation and distance calculations, though doing so is recommended.

---

## Quick Start: Explain without a Config Object

For a first test, you only need a short Python script. The explainer can wrap any standard machine learning model (e.g., XGBoost, scikit-learn) and generate a local explanation.

### 1. Run a Basic Explanation

The following example uses the default configuration and explains a single local instance from the Boston housing dataset:

```python
import pandas as pd
from sklearn.datasets import fetch_openml
import xgboost
from xwhy import TabularExplainer

# Standard Scaler
class StandardScalerWrapper:
    """
    Simple reusable normalization utility for SMILE/LIME/SHAP consistency.
    """

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0) + 1e-8
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

# Load Boston housing dataset
bunch = fetch_openml(name="boston", version=1, as_frame=True, parser="auto")
X = bunch.data
y = bunch.target.rename("MEDV")
scaler = StandardScalerWrapper()
X_scaled = scaler.fit_transform(X.values)

# Train an XGBoost Regressor
xg_model = xgboost.XGBRegressor(
    enable_categorical=True,
    tree_method="hist",
    random_state=42,
    n_estimators=100
).fit(X, y)

instance = X_scaled[0]

try:
    explainer = TabularExplainer(model=xg_model)
    # or use `explainer.run`
    result = explainer.explain(instance=instance, feature_names=X.columns.to_list())
    
    print(result.metrics)
    print("Explanation successful!")

except Exception as e:
    print(f"Error during pipeline execution: {e}")

```

### 2. Using a `TabularConfig` Object

You can also centralize settings in a `TabularConfig` instance and pass it to the `TabularExplainer` constructor. This allows you to explicitly define the explanation mode, distance metrics, and surrogate parameters:

```python
from xwhy.core import TabularConfig
from xwhy.distance import DistanceType
from xwhy.surrogate import SurrogateType
from xwhy import TabularExplainer

tabular_cfg = TabularConfig(
    mode="classification",  # You can choose "classification" or "regression"
    num_perturbations=500,
    kernel_width=0.2,
    num_distribution_samples=100,
    local_noise=0.05,
    perturbation_noise=0.4,
    epsilon=0.01,
    distance_type=DistanceType.WASSERSTEIN,  # You can set it as string too like "wasserstein"
    surrogate_type=SurrogateType.LIME,  # You can set it as string too like "lime"
    use_best_surrogate=True,
    seed=42,
    validate_normalization=True,
)

try:
    # Assuming `model` is a trained classifier (e.g., RandomForestClassifier)
    # and `instance` is a standardized 1D numpy array
    explainer = TabularExplainer(config=tabular_cfg, model=model)
    
    # or use `explainer.run`
    result = explainer.explain(instance=instance)
    
    print(result.metrics)
    print("Explanation successful!")

except Exception as e:
    print(f"Error during pipeline execution: {e}")

```

### 3. Read the Result

The explanation highlights the tabular features according to their estimated influence on the model prediction.

The returned `result` object also contains evaluation metrics that can be used to examine the quality and reliability of the local explanation. These metrics should be interpreted as evidence about the explanation produced by XWhy, rather than as direct access to the classifier’s internal decision process.

---

## Additional Explanation Plots

After generating a valid `result`, you can use the following visualisations to interpret the feature contributions:

```python
import xwhy.plots
from xwhy.plots import (
    plot_explanation_waterfall,
    plot_feature_contributions,
    plot_method_contributions,
)

# Standard built-in plot
result.plot()

# Custom contribution plots
plot_explanation_waterfall(result)
plot_feature_contributions(result)
plot_method_contributions(result)

# SHAP-style visualisations provided by xwhy
xwhy.plots.bar(result)
xwhy.plots.waterfall(result)
xwhy.plots.force(result)
xwhy.plots.decision(result)

```

---

## Distance Metrics and Surrogate Models

XWhy allows you to specify how distances are calculated between perturbed tabular samples and how the interpretable surrogate model is trained.

### Distance Metrics

Pass the desired identifier via the `distance_type` argument in `TabularConfig`:

* `DistanceType.WASSERSTEIN` (or `"wasserstein"`)



### Surrogate Models

Select the underlying surrogate logic via the `surrogate_type` argument:

* `SurrogateType.LIME` (or `"lime"`)


* Enable `use_best_surrogate=True` to let the framework automatically optimize for the best surrogate configuration.



---

## Data Preprocessing Note

Unlike image classification models that handle raw pixels, tabular explainers require normalized or standardized features to ensure stable distance calculations (e.g., using `StandardScaler` to remove mean and scale to unit variance). Setting `validate_normalization=True` in the configuration will check if the input data meets this requirement.

---

## Interpretation Note

XWhy produces a local, perturbation-based approximation of feature influence. It can help identify which parts of an instance are associated with changes in a particular classification or regression decision. It does not expose the model’s exact internal reasoning process or prove that a highlighted feature was the sole cause of the predicted output.
