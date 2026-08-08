# ATT consistency

**ATT consistency** asks whether the same input, model, explanation configuration, and intended operating conditions produce reproducible attributions across repeated executions.

Unlike [ATT stability](https://dependable-intelligent-systems-lab.github.io/xwhy/evaluation/attribution-stability/index.md), the evaluated input is not deliberately changed. Consistency isolates execution variability arising from random perturbation sampling, surrogate fitting, stochastic model generation, provider behaviour, hardware, or other uncontrolled factors.

## General protocol

For a fixed input (x), run the explanation procedure (R) times:

```
a¹(x), a²(x), ..., aᴿ(x)
```

Compare the attribution vectors, rankings, selected feature sets, and any relevant output signals across runs.

## Sources of inconsistency

- random perturbation sampling;
- random initialisation or optimisation in the surrogate;
- non-deterministic black-box inference;
- LLM sampling temperature and provider-side variability;
- stochastic segmentation or clustering;
- changes in retrieved evidence for RAG systems;
- unstable feature construction or alignment;
- different distance, kernel, or hyperparameter choices when these are intended to be fixed.

## Modality-specific interpretation

| Modality             | Repeated object                                                | Consistency evidence                                          |
| -------------------- | -------------------------------------------------------------- | ------------------------------------------------------------- |
| Image classification | Same image and explanation configuration                       | Similar superpixel rankings and attribution magnitudes        |
| LLM / gSMILE         | Same prompt, model, parameters, and provider setup             | Similar token weights and generated response behaviour        |
| Image editing        | Same source image, instruction, seed policy, and editor        | Similar instruction attributions and edited-image effects     |
| Point cloud          | Same point cloud and preprocessing                             | Similar point/cluster importance despite execution randomness |
| KG-RAG               | Same query, graph snapshot, retrieval configuration, and model | Similar evidence ranking and attribution paths                |
| ConceptSMILE         | Same image, concept pathway, prompt, and perturbation design   | Low variation in concept scores and feature importance        |

## Measures

ConceptSMILE treats lower variance and standard deviation across repeated executions as stronger reproducibility. Depending on the modality, report:

- per-feature mean, variance, and standard deviation;
- confidence intervals for attribution values;
- Jaccard overlap of top-k feature sets;
- rank correlation between repeated attribution orderings;
- frequency with which each feature enters the top-k set;
- variance of summary metrics such as fidelity or coverage.

A simple user-side analysis can use XWhy result coefficients:

```
import numpy as np

runs = [explainer.explain(instance) for _ in range(10)]
coefficient_matrix = np.vstack([result.coefficients for result in runs])

mean_attribution = coefficient_matrix.mean(axis=0)
std_attribution = coefficient_matrix.std(axis=0)
variance = coefficient_matrix.var(axis=0)
```

This assumes that the feature space is identical across runs. For image explanations, regenerated superpixels may differ; for point-cloud explanations, clusters may change; and for LLMs, word or token segmentation must remain aligned.

## Current XWhy support

XWhy exposes the attribution vector and raw evaluation data needed for repeated-run analysis, but there is currently **no dedicated public `ATTConsistency` function**.

Relevant controls include:

- explainer random seeds;
- LLM temperature and provider options;
- perturbation count;
- surrogate type and automatic surrogate selection;
- distance metric and locality weighting;
- segmentation or feature-construction configuration.

For LLMs, setting `temperature=0.0` and a fixed seed reduces known sampling variation, but an external provider may still be non-deterministic. Report the provider, model version, request parameters, date, and repeated outputs.

## Consistency across hyperparameters

The gSMILE discussion also considers consistency across model runs or hyperparameter settings. This should be reported separately from strict repeated-run consistency:

- **repeatability:** same input and same configuration;
- **configuration robustness:** same input under deliberately varied but plausible settings.

Mixing these experiments into one score obscures whether variability came from randomness or an intentional design change.

## Reporting checklist

Report:

- number of repeated runs;
- all fixed and varying parameters;
- random seeds and deterministic settings;
- provider/model version and retrieval snapshot where relevant;
- feature-alignment procedure;
- per-feature variability and aggregate overlap;
- uncertainty intervals rather than only one average score.

Consistency is not stability or validity

Repeatable explanations may still be inaccurate or unfaithful. Conversely, a stochastic model may produce some attribution variability even when the explanation method is behaving appropriately. Interpret consistency together with the task and model's expected randomness.

## Research basis

- [gSMILE](https://arxiv.org/abs/2505.21657) defines ATT consistency through repeatable token weights and outputs for repeated use of the same prompt.
- [ConceptSMILE](https://arxiv.org/abs/2607.09649) evaluates reproducibility through repeated execution and reports variance and standard deviation as consistency indicators.
