# ATT accuracy

**ATT accuracy** asks whether the features assigned high importance by an explanation align with suitable reference evidence.

This is **not** the predictive accuracy of the black-box model. It is the accuracy of the attribution relative to a reference that identifies which tokens, regions, points, graph elements, or concepts should be relevant.

## Required inputs

An ATT accuracy evaluation needs:

1. attribution scores for interpretable features;
1. a reference attribution or relevance annotation;
1. a rule for matching explanation features to reference features;
1. an aggregation metric and thresholding policy, where applicable.

The reference may come from human annotation, a semantic mask, domain-expert evidence, a controlled synthetic task, or another defensible source. The quality and uncertainty of that reference must be reported.

## Modality-specific definitions

| Modality             | Attribution unit                      | Reference example                               | Suitable evaluation examples                                            |
| -------------------- | ------------------------------------- | ----------------------------------------------- | ----------------------------------------------------------------------- |
| Image classification | Pixels, superpixels, regions          | Object or lesion mask                           | Coverage, weighted coverage, overlap, ranking against annotated regions |
| LLM / gSMILE         | Tokens, words, phrases                | Human-labelled influential text elements        | ROC-AUC, precision, recall, F1, top-k agreement                         |
| Image editing        | Instruction tokens or visual regions  | Required edit terms or target edit region       | Token-ranking agreement, edited-region overlap, expert judgement        |
| Point cloud          | Points, clusters, object parts        | Part segmentation or annotated relevant cluster | Point/cluster overlap, coverage, top-k part agreement                   |
| KG-RAG               | Entities, relations, paths            | Gold evidence graph or supported reasoning path | Evidence precision/recall, ranking, path overlap                        |
| ConceptSMILE         | Concepts and concept-relevant regions | Domain concept annotation                       | Concept-region alignment, concept detection agreement                   |

## LLM example from gSMILE

The gSMILE paper compares token attribution scores with ground-truth labels identifying relevant input words. It operationalises ATT accuracy with the ROC-AUC of attribution scores: a score of 1 indicates perfect ranking of relevant tokens above irrelevant tokens, while 0.5 corresponds to random ranking.

The gSMILE example compares token-level ground truth with generated token attributions. Source: Dehghani et al., [Figure 6 and Section 4.4.1](https://arxiv.org/html/2505.21657#S4.SS4.SSS1).

## Image functions currently implemented in XWhy

XWhy currently provides image-specific spatial coverage functions through `ImageCoverageMetrics`:

```
from xwhy.metrics import ImageCoverageMetrics

coverage = ImageCoverageMetrics.calculate_coverage(
    explanation_image=explanation_map,
    semantic_mask=ground_truth_mask,
    class_of_interest=1,
)

weighted_coverage = ImageCoverageMetrics.calculate_weighted_coverage(
    explanation_image=explanation_map,
    semantic_mask=ground_truth_mask,
    class_of_interest=1,
)

coverage, weighted_coverage = ImageCoverageMetrics.evaluate_all(
    explanation_image=explanation_map,
    semantic_mask=ground_truth_mask,
    class_of_interest=1,
)
```

`calculate_coverage()` rewards selected explanation pixels within the target class and penalises selected pixels belonging to other labelled objects. `calculate_weighted_coverage()` additionally uses the magnitude of the explanation map.

The image-classification explainer can calculate these values when a ground-truth mask is supplied or when its configured segmentation pathway produces a semantic mask:

```
result = explainer.explain(
    instance="image.jpg",
    ground_truth_mask=mask,
)

print(result.coverage)
print(result.weighted_coverage)
```

Coverage is an image-specific ATT accuracy proxy

These functions do not implement a universal ATT accuracy metric. They assume spatial arrays with matching image dimensions and a semantic mask. They cannot be directly applied to LLM tokens, point-cloud clusters, knowledge-graph evidence, or concept lists.

## Current support for other modalities

XWhy does not currently expose a public token-level AttAUC/F1 evaluator or a generic attribution-reference alignment class. For an LLM evaluation, the expected inputs would be an attribution score per token and a binary or graded reference relevance label aligned to the same tokenisation. That evaluator remains to be implemented in the package.

Similarly, point-cloud, image-editing, GraphRAG, and concept-level ATT accuracy require modality-specific feature alignment before a metric can be computed.

## Reporting checklist

Report:

- who or what produced the reference attribution;
- annotation granularity and uncertainty;
- feature-alignment and tokenisation rules;
- whether negative attributions are included or transformed;
- threshold or top-k selection rules;
- class imbalance;
- both ranking and set/coverage results where possible.

Plausibility is not accuracy

An explanation may look reasonable without aligning with a defensible reference. Conversely, disagreement with one annotation does not automatically prove an explanation is wrong when the task admits multiple valid evidence sets.

## Research basis

- [gSMILE](https://arxiv.org/abs/2505.21657) defines token-level ATT accuracy using annotated relevant text elements and AttAUC.
- [ConceptSMILE](https://arxiv.org/abs/2607.09649) evaluates whether concept attributions align with clinically meaningful retinal evidence.
