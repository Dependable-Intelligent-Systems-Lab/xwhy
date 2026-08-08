# ATT faithfulness

**ATT faithfulness** asks whether attributed importance reflects the model's actual response behaviour rather than only producing a plausible-looking explanation.

A faithful explanation should assign greater importance to evidence whose controlled removal, replacement, or preservation causes a correspondingly meaningful change in the black-box response.

## General intervention principle

For an interpretable feature (j):

1. obtain its attribution score (a_j);
1. intervene on feature (j) or on a feature set ranked by attribution;
1. measure the resulting black-box response change (\\Delta_j);
1. test whether attribution magnitude and response effect agree.

The intervention and response variable must be meaningful for the modality. Masking a superpixel, removing a token, deleting a graph relation, and suppressing a concept region are different experimental operations.

## Modality-specific operationalisations

| Modality             | Intervention                                                        | Response effect                                               |
| -------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------- |
| Image classification | Mask or preserve attributed regions                                 | Change in class probability or prediction                     |
| LLM / gSMILE         | Remove, replace, or retain attributed tokens/phrases                | Semantic or distributional change in the generated output     |
| Image editing        | Modify attributed instruction terms or conditioning evidence        | Change in edited-image semantics, structure, or target region |
| Point cloud          | Remove or alter attributed points/clusters                          | Change in class, detection, or segmentation response          |
| KG-RAG               | Remove attributed entities, relations, paths, or retrieved passages | Change in evidence use, answer support, or semantic output    |
| ConceptSMILE         | Perturb evidence relevant or irrelevant to a concept                | Change in concept confidence or semantic concept response     |

## Two paper-specific definitions

The SMILE-family papers use related but not identical operationalisations.

### gSMILE

The gSMILE paper defines the general goal as ensuring that important tokens genuinely drive the LLM response. Its reported quantitative evaluation treats faithfulness as an **externally validated property**, using Pearson correlation between attribution metrics and published benchmark accuracies for the evaluated models. A stronger positive correlation indicates that the attribution measurements track external model competence.

### ConceptSMILE

ConceptSMILE uses a direct perturbation-response interpretation. Concept-relevant perturbations should produce larger concept-confidence shifts than irrelevant perturbations. This evaluates whether the concept explanation is connected to the pathway's observed behaviour rather than only appearing clinically or semantically plausible.

Do not treat these definitions as interchangeable

External benchmark correlation and direct intervention-response agreement answer different questions. A study must state which operationalisation is used and why it is appropriate for the modality and claim being tested.

## Current XWhy support

XWhy currently has **no standalone public `ATTFaithfulness` evaluator**. The explainers do, however, return raw perturbation and response data that can support modality-specific analysis.

For an LLM result, relevant fields include:

```
result.raw_data["perturbed_texts"]
result.raw_data["wmd_scores"]
result.raw_data["similarities"]
result.raw_data["weights"]
result.raw_data["y_target"]
result.coefficients
```

For an image-classification result, relevant fields include:

```
result.raw_data["x_matrix"]
result.raw_data["predictions"]
result.raw_data["distances"]
result.raw_data["weights"]
result.raw_data["y_target"]
result.coefficients
```

These arrays expose the sampled perturbations and observed response signals, but they do not by themselves constitute a complete faithfulness test. The user must define the intervention ranking, response measure, feature alignment, baseline, and aggregation procedure.

## Example study design

A direct perturbation faithfulness experiment can compare cumulative response change as increasingly important features are removed:

```
1. Rank features by |attribution|.
2. Remove the top 1, top 2, ..., top k features.
3. Re-query the black-box model after each intervention.
4. Measure output change relative to the unmodified input.
5. Compare against random-feature and low-attribution baselines.
```

For generative systems, output change should be measured semantically or distributionally rather than through exact string or pixel equality.

## Reporting checklist

Report:

- the exact faithfulness definition;
- whether the test is direct intervention-based or externally validated;
- the intervention operator and realism constraints;
- the output-distance or response-change function;
- positive, negative, and absolute attribution handling;
- random and low-importance baselines;
- intervention order and cumulative-removal policy;
- model stochasticity and repeated-query uncertainty.

Perturbation artefacts can imitate faithfulness

An intervention may create an unrealistic input that changes the model response for reasons unrelated to the attributed evidence. Use modality-appropriate perturbations, compare against controls, and report whether the modified input remains valid.

## Research basis

- [gSMILE](https://arxiv.org/abs/2505.21657) defines token-level faithfulness and reports an external correlation-based evaluation.
- [ConceptSMILE](https://arxiv.org/abs/2607.09649) evaluates whether concept-relevant evidence perturbations produce the expected concept-response shifts.
