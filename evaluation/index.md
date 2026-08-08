# Evaluate explanations across modalities

Explanation quality is **multi-dimensional**. The SMILE research family evaluates explanations through five complementary questions:

1. **[ATT fidelity](https://dependable-intelligent-systems-lab.github.io/xwhy/evaluation/attribution-fidelity/index.md):** does the local surrogate reproduce the black-box model's behaviour in the sampled neighbourhood?
1. **[ATT accuracy](https://dependable-intelligent-systems-lab.github.io/xwhy/evaluation/attribution-accuracy/index.md):** do high attribution scores align with an appropriate reference attribution?
1. **[ATT stability](https://dependable-intelligent-systems-lab.github.io/xwhy/evaluation/attribution-stability/index.md):** does the explanation remain similar after a small, semantically irrelevant change to the input?
1. **[ATT consistency](https://dependable-intelligent-systems-lab.github.io/xwhy/evaluation/attribution-consistency/index.md):** is the explanation reproducible when the same analysis is repeated under equivalent conditions?
1. **[ATT faithfulness](https://dependable-intelligent-systems-lab.github.io/xwhy/evaluation/attribution-faithfulness/index.md):** does attributed importance correspond to genuine changes in the model response when the attributed evidence is changed?

`ATT` means **attribution**, following the terminology used in [gSMILE](https://arxiv.org/abs/2505.21657).

The metric question may be shared; the metric definition is modality-specific

A metric name is not a complete specification. For every evaluation, report the feature unit, perturbation, model response, reference evidence, aggregation rule, and direction of improvement. A token-level ATT accuracy score is not operationally equivalent to a superpixel-coverage score, even though both test attribution alignment.

## Why definitions change by modality

Each modality exposes a different interpretable unit and a different observable model response.

| Modality                        | Interpretable unit                                  | Typical perturbation                             | Response being explained                                    | Possible reference evidence                                  |
| ------------------------------- | --------------------------------------------------- | ------------------------------------------------ | ----------------------------------------------------------- | ------------------------------------------------------------ |
| Image classification            | Pixel, superpixel, region, or concept               | Mask, replace, blur, or preserve a region        | Class score or probability shift                            | Semantic mask, object region, expert annotation              |
| Large language model            | Token, word, phrase, or prompt component            | Remove, mask, replace, or paraphrase text        | Semantic or distributional output shift                     | Human-labelled influential tokens, benchmark evidence        |
| Instruction-based image editing | Instruction token or phrase; sometimes image region | Modify the instruction or condition              | Perceptual, embedding, structural, or semantic image change | Required edit concepts, edited regions, human judgement      |
| Point cloud                     | Point, cluster, segment, or object part             | Remove, mask, jitter, rotate, or resample points | Class or detection response shift                           | Ground-truth object part, spatial cluster, expert annotation |
| Knowledge graph / GraphRAG      | Entity, relation, path, subgraph, or retrieved item | Remove or replace graph evidence                 | Answer, retrieval, or semantic-response shift               | Gold evidence, supported path, answer annotation             |
| Concept-based explanation       | Human-understandable concept and its evidence       | Perturb concept-relevant regions or inputs       | Concept confidence or semantic concept response             | Domain annotation, concept mask, expert judgement            |

The same broad reliability question therefore requires a modality-specific operationalisation. For example, gSMILE evaluates token attributions around an LLM prompt, whereas ConceptSMILE evaluates visual or semantic concept responses under image-region perturbation. The latter paper also shows why the five dimensions should not be collapsed into one score: an explanation pathway may be spatially accurate but less faithful, stable but inconsistent, or locally well approximated while weak on another reliability dimension.

## LLM example from gSMILE

The following gSMILE figure illustrates how ATT accuracy, faithfulness, stability, and consistency are formulated around **token-level** explanations. Ground-truth token importance, repeated runs, and minimally changed prompts provide different comparison conditions.

Token-level attribution evaluation in gSMILE. Source: Dehghani et al., [Explaining Large Language Models with gSMILE](https://arxiv.org/html/2505.21657#S4.SS4), Figure 6.

ATT fidelity requires a separate comparison between the black-box response signal and the local surrogate response over the same perturbations.

ATT fidelity workflow in gSMILE. Source: Dehghani et al., [Explaining Large Language Models with gSMILE](https://arxiv.org/html/2505.21657#S4.SS4.SSS5), Figure 7.

## Required evaluation specification

Before reporting any metric, define the following tuple:

```
(feature unit, perturbation operator, response variable,
 reference evidence, neighbourhood, aggregation, randomness controls)
```

For example, an LLM stability experiment might use words as features, semantically neutral phrase insertion as the perturbation, top-k token attribution as the explanation, Jaccard overlap as the aggregation, and a fixed model configuration. An image stability experiment might instead compare attribution maps after a benign acquisition artefact or non-destructive transformation.

## Current XWhy implementation coverage

The documentation distinguishes **implemented package functions** from evaluation procedures described in the papers.

| Dimension        | Current XWhy support                 | Notes                                                                                                                                                      |
| ---------------- | ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ATT fidelity     | **Implemented**                      | `RegressionMetrics.calculate()` is used by the image-classification and LLM explainers. Result objects expose weighted error and R-squared metrics.        |
| ATT accuracy     | **Partially implemented for images** | `ImageCoverageMetrics` measures spatial agreement with a semantic mask. This is an image-specific coverage measure, not a universal ATT accuracy function. |
| ATT stability    | **Methodology documented**           | No dedicated public package evaluator is currently implemented. Explanations can be compared using returned coefficients and feature names.                |
| ATT consistency  | **Methodology documented**           | No dedicated public package evaluator is currently implemented. Repeated-run analysis must currently be performed by the user.                             |
| ATT faithfulness | **Methodology documented**           | No standalone public package evaluator is currently implemented. The required intervention and response comparison depends on the modality.                |

Do not invent universal thresholds

Metric values depend on the response scale, perturbation protocol, feature granularity, neighbourhood, model stochasticity, reference annotations, and aggregation method. Report configurations and comparative baselines rather than applying one acceptance threshold across modalities.

## Source papers

The evaluation framework is grounded primarily in:

- [Explaining Large Language Models with gSMILE](https://arxiv.org/abs/2505.21657), which defines the five ATT dimensions for token-level LLM explanations.
- [ConceptSMILE: Auditing the Trustworthiness of Concept-Based Explainable AI](https://arxiv.org/abs/2607.09649), which reformulates the dimensions for concept-level visual and semantic explanations.
- [XWhy and SMILE publications](https://dependable-intelligent-systems-lab.github.io/xwhy/research/publications/index.md), which lists the broader modality-specific research family.
