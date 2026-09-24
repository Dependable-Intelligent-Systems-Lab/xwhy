# ATT fidelity

**ATT fidelity** asks whether the interpretable local surrogate reproduces the black-box model's response within the perturbation neighbourhood used to generate the explanation.

It evaluates the explanation mechanism, not the original model's task accuracy. A classifier can be accurate while its local explanation has poor fidelity, and a highly faithful surrogate can accurately approximate undesirable black-box behaviour.

## General formulation

For perturbations (z_i), let:

- (y_i) be the black-box response or response shift;
- (\\hat{y}\_i) be the local surrogate prediction;
- (w_i) be the locality weight assigned to the perturbation.

ATT fidelity compares (y_i) and (\\hat{y}\_i), with local samples receiving the weights specified by the explanation method.

## What changes by modality

| Modality             | Black-box target (y)                                                        | Surrogate target (\\hat{y})                                                  |
| -------------------- | --------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Image classification | Predicted-class probability after masking superpixels                       | Probability shift predicted from the retained-superpixel vector              |
| LLM / gSMILE         | Semantic or distributional output shift after prompt perturbation           | Output shift predicted from retained or altered tokens                       |
| Image editing        | Perceptual, embedding, structural, or semantic change in the edited image   | Change predicted from instruction-token or condition perturbations           |
| Point cloud          | Class, detection, or segmentation response after point/cluster perturbation | Response predicted from retained points, clusters, or parts                  |
| KG-RAG               | Answer or semantic-response shift after removing graph evidence             | Shift predicted from retained entities, relations, paths, or retrieved items |
| ConceptSMILE         | Concept-confidence or concept-response shift after evidence perturbation    | Concept-response shift predicted from the perturbation vector                |

The metrics may have the same names across modalities, but their raw values are only comparable when the response variables and scaling are comparable.

## Metrics currently implemented in XWhy

XWhy provides `RegressionMetrics.calculate()` and the immutable `RegressionMetricResult` container.

```
from xwhy.metrics import RegressionMetrics

fidelity = RegressionMetrics.calculate(
    y_true=y_target,
    y_pred=y_pred,
    weights=locality_weights,
    num_features=num_interpretable_features,
)

print(fidelity.weighted_r2)
print(fidelity.weighted_adj_r2)
print(fidelity.weighted_mse)
print(fidelity.weighted_mae)
```

The function currently returns:

| Field              | Interpretation                                     | Preferred direction |
| ------------------ | -------------------------------------------------- | ------------------- |
| `weighted_r2`      | Weighted coefficient of determination              | Higher              |
| `weighted_adj_r2`  | Weighted R-squared adjusted for feature count      | Higher              |
| `weighted_mse`     | Locality-weighted squared prediction error         | Lower               |
| `weighted_mae`     | Locality-weighted absolute prediction error        | Lower               |
| `mean_loss`        | Difference between mean target and mean prediction | Lower               |
| `mean_l1_loss`     | Mean absolute residual                             | Lower               |
| `mean_l2_loss`     | Mean squared residual                              | Lower               |
| `weighted_l1_norm` | Weighted absolute residual norm                    | Lower               |
| `weighted_l2_norm` | Weighted squared residual norm                     | Lower               |

## Reading fidelity from an explanation result

Both currently implemented explainers attach the fidelity result to `result.metrics`.

```
result = explainer.explain(instance)

print(result.metrics)
print(result.metrics.weighted_r2)
print(result.metrics.weighted_mae)
```

The result also stores the arrays used for fidelity evaluation:

```
y_target = result.raw_data["y_target"]
y_pred = result.raw_data["y_pred"]
weights = result.raw_data["weights"]
```

A fidelity plot can be generated from an XWhy result:

```
result.plot(show=True)
```

## LLM example

In gSMILE, prompt perturbations are represented by token-retention vectors. The black-box target is the semantic output shift caused by each perturbed prompt, and the local surrogate predicts that shift from the interpretable token vector. The paper evaluates alignment using R-squared variants and error-based metrics.

ATT fidelity in gSMILE compares the black-box response signal with the local surrogate signal over the same prompt perturbations. Source: Dehghani et al., [Figure 7](https://arxiv.org/html/2505.21657#S4.SS4.SSS5).

## Interpretation requirements

Report fidelity together with:

- the response variable and its scale;
- the perturbation count and generation procedure;
- the locality distance and weighting kernel;
- the surrogate family and feature count;
- whether model outputs were deterministic;
- the distribution of locality weights;
- both an accuracy-based and an error-based fidelity metric.

High fidelity is local and conditional

A high score only demonstrates that the selected surrogate approximates the sampled black-box responses under the stated neighbourhood and weighting scheme. It does not establish global equivalence, causal validity, attribution accuracy, stability, consistency, or human usefulness.

## Research basis

- [gSMILE](https://arxiv.org/abs/2505.21657) defines ATT fidelity for LLM token attribution using local surrogate agreement.
- [ConceptSMILE](https://arxiv.org/abs/2607.09649) applies surrogate fidelity to concept-response behaviour and shows that fidelity can differ by concept pathway and locality weighting.
