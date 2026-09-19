# Research

SMILE—**Statistical Model-Agnostic Interpretability with Local Explanations**—is a family of local explainability methods for analysing black-box artificial-intelligence systems. The foundational method explains machine-learning classifiers by fitting a local surrogate model whose perturbed samples are weighted using statistical distance measures. Later work adapts this principle to spatial, generative, language, retrieval-augmented, and concept-based systems.

## Research map

| Research area                  | Method or study                                     | Explained system                               | Publication status               |
| ------------------------------ | --------------------------------------------------- | ---------------------------------------------- | -------------------------------- |
| Core method                    | SMILE                                               | Machine-learning and deep-learning classifiers | Peer-reviewed journal article    |
| Vision and spatial AI          | Point-cloud SMILE                                   | Point-cloud neural networks                    | arXiv preprint                   |
| Vision and generative AI       | Image-editing SMILE                                 | Instruction-based image-editing models         | arXiv preprint                   |
| Language and generative AI     | gSMILE                                              | Large language models                          | arXiv preprint                   |
| Retrieval-augmented generation | KG-SMILE                                            | Knowledge-graph and GraphRAG systems           | arXiv preprint                   |
| Concept-based explainability   | ConceptSMILE                                        | Concept-based explainable-AI methods           | arXiv preprint                   |
| Financial language analysis    | Local perturbation explanations derived from gSMILE | LLM financial-sentiment reasoning              | Peer-reviewed conference chapter |
| Consolidated academic study    | Generative-AI SMILE thesis                          | LLMs and instruction-based image editing       | MSc thesis                       |

## How the research family is organised

### Core method

The foundational SMILE paper defines the statistical-distance-based local explanation framework for black-box classifiers. This is the principal methodological citation for general references to SMILE.

### Vision and spatial AI

The point-cloud extension explains influential groups of 3D points and evaluates explanation fidelity, stability, and robustness. The image-editing extension instead perturbs natural-language editing instructions and measures how words or phrases affect generated visual changes.

### Language and generative AI

gSMILE explains large-language-model behaviour by perturbing prompt components, measuring changes in model outputs with statistical distances such as Wasserstein distance, and fitting a locally weighted surrogate model. A related financial-sentiment study applies local perturbation explanations to behavioural and robustness analysis.

### Retrieval-augmented generation

KG-SMILE explains the contribution of retrieved entities, relations, graph paths, and contextual evidence within knowledge-graph retrieval-augmented generation workflows.

### Concept-based explainability

ConceptSMILE audits whether human-understandable concepts provide explanations that are faithful, locally representative, stable, and consistent.

## Research resources

- [Browse the complete publication catalogue](https://dependable-intelligent-systems-lab.github.io/xwhy/research/publications/index.md)
- [Choose the correct citation for your use case](https://dependable-intelligent-systems-lab.github.io/xwhy/research/citation/index.md)
- [Follow the reproducibility guidance](https://dependable-intelligent-systems-lab.github.io/xwhy/how-to/reproducibility/index.md)

Publication metadata

The publication types, identifiers, links, and BibTeX records in this section are maintained from the project bibliography. Preprints and online-first records may later receive updated journal, conference, volume, issue, or pagination metadata.
