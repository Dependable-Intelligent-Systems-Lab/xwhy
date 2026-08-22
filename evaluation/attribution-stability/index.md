# ATT stability

**ATT stability** asks whether an explanation remains similar when the input is changed slightly without materially changing the evidence or task meaning.

The input is deliberately changed. This distinguishes stability from [ATT consistency](https://dependable-intelligent-systems-lab.github.io/xwhy/evaluation/attribution-consistency/index.md), where the input and intended configuration remain the same and only repeated execution varies.

## Stability experiment

A stability test requires paired inputs:

```
original input x
nearby input x' that should preserve the relevant meaning or evidence
```

Generate explanations (a(x)) and (a(x')), align their feature spaces, and quantify the difference. The perturbation used to test stability must be separate from the perturbations used internally to construct each local explanation.

## What counts as a small change

| Modality             | Stability perturbation examples                                                                   | Features compared                                        |
| -------------------- | ------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| Image classification | Mild brightness/noise change, non-destructive crop, irrelevant border or acquisition artefact     | Attribution maps, superpixels, ranked regions            |
| LLM / gSMILE         | Semantically neutral insertion, minor paraphrase, punctuation or wording variation                | Token/phrase rankings after semantic alignment           |
| Image editing        | Equivalent rephrasing of an instruction; irrelevant instruction detail                            | Instruction-token attribution or affected visual regions |
| Point cloud          | Small coordinate jitter, order permutation, class-preserving rotation, limited irrelevant dropout | Points, clusters, object parts                           |
| KG-RAG               | Add/remove irrelevant graph evidence or equivalent retrieval context                              | Entities, relations, paths, retrieved items              |
| ConceptSMILE         | Non-clinical artefacts or changes outside concept-relevant evidence                               | Concept scores, concept regions, rankings                |

A transformation must be justified as meaning-preserving for the task. For example, an arbitrary rotation may preserve an object class in one point-cloud task but invalidate orientation-sensitive tasks.

## LLM definition in gSMILE

The gSMILE paper examines whether token attributions remain similar after small prompt changes. It uses the Jaccard index to compare sets of important elements, with values closer to 1 indicating greater overlap.

```
# User-side stability recipe; not a dedicated XWhy API function.
def jaccard(set_a: set[int], set_b: set[int]) -> float:
    union = set_a | set_b
    return 1.0 if not union else len(set_a & set_b) / len(union)
```

A top-k comparison using XWhy result coefficients can be constructed as follows:

```
import numpy as np

result_a = explainer.explain(prompt_a)
result_b = explainer.explain(prompt_b)

k = 5
top_a = set(np.argsort(np.abs(result_a.coefficients))[-k:])
top_b = set(np.argsort(np.abs(result_b.coefficients))[-k:])
score = jaccard(top_a, top_b)
```

Feature alignment is required

Direct token indices are only comparable when the two prompts have compatible token or word positions. Paraphrases require semantic phrase alignment, not naive index matching. Image superpixels and point-cloud clusters may also change between inputs and require spatial matching.

## Current XWhy support

XWhy result objects expose:

- `result.coefficients` for feature attributions;
- `result.feature_names` for tokens or generated feature labels;
- `result.raw_data` for perturbation and response arrays.

There is currently **no dedicated public `ATTStability` evaluator**. The comparison procedure must therefore be defined by the user and should be reported as an external evaluation built from XWhy results.

## Recommended measures

Use one or more of the following, depending on modality:

- Jaccard overlap of top-k important features;
- rank correlation after feature alignment;
- cosine similarity between aligned attribution vectors;
- spatial overlap or distance between attribution maps;
- change in concept attribution scores;
- explanation-distribution divergence across a set of benign transformations.

Do not use only a set-overlap score when attribution magnitude and direction are important.

## Reporting checklist

Report:

- why the input change should preserve task meaning;
- the transformation magnitude and random seed;
- how features were aligned across inputs;
- whether predictions or generated outputs also remained equivalent;
- the attribution comparison measure and top-k rule;
- results across multiple samples and transformations, not one pair only.

Stable does not mean correct

A consistently wrong or unfaithful explanation can be highly stable. Stability must be interpreted alongside ATT accuracy, fidelity, and faithfulness.

## Research basis

- [gSMILE](https://arxiv.org/abs/2505.21657) evaluates token-attribution stability under minor prompt changes and uses Jaccard similarity.
- [ConceptSMILE](https://arxiv.org/abs/2607.09649) tests whether concept explanations remain robust under controlled non-clinical artefacts and irrelevant visual changes.
