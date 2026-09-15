# Agentic AI explainability

Coming soon

Agentic AI explainability is a research-roadmap capability. XWhy does not currently export or implement a supported Agentic AI explainer.

Agentic AI systems differ from conventional prediction models because their behaviour unfolds across a sequence of decisions. A single execution may include planning, retrieval, memory access, tool selection, tool outputs, intermediate evaluation, replanning, and action. The explanation target is therefore not only the final answer, but the **observable trajectory that produced it**.

## Planned explanation targets

XWhy's Agentic AI roadmap is intended to support explanations of:

- **plan formation and revision** — which observable inputs or intermediate results influenced a plan or replan;
- **tool selection and use** — why one tool or action was selected over another and how its output affected later steps;
- **retrieval influence** — which retrieved documents, passages, or evidence materially changed the agent trajectory;
- **memory influence** — how accessible short-term or persistent memory affected decisions;
- **state transitions** — which observations or intermediate states were associated with changes in agent behaviour;
- **uncertainty and failure propagation** — where uncertainty, ambiguity, or an error first appeared and how it affected downstream decisions;
- **final outcome influence** — which observable parts of the trajectory had the strongest measured relationship with the final response or action.

## Proposed XWhy approach

The planned direction is to extend the existing perturbation-based XWhy workflow from static inputs to **agent trajectories**. Depending on the agent architecture, perturbations may be applied to prompts, retrieved evidence, memory items, tool outputs, state variables, or selected actions. The resulting changes in observable behaviour can then be analysed using local surrogate models, attribution measures, and intervention-based validation.

Potential explanation units include:

- prompt terms and phrases;
- retrieved passages;
- memory entries;
- plan steps;
- tool calls and tool outputs;
- state variables;
- evaluator decisions;
- actions and action sequences.

## Evaluation priorities

Agentic explanations should be evaluated beyond visual plausibility. Planned evaluation areas include:

- **fidelity** — how well the explanation approximates observed agent behaviour locally;
- **faithfulness** — whether perturbing highly attributed elements changes the trajectory or outcome as expected;
- **stability** — whether small changes to the input or state produce reasonable changes in the explanation;
- **consistency** — whether repeated executions with comparable trajectories produce comparable explanations;
- **temporal sensitivity** — whether explanations correctly capture when an influential event occurred;
- **failure localisation** — whether the explanation can identify the stage at which an error or uncertainty first became consequential.

## Responsible interpretation

Agentic explainability must not be presented as access to hidden chain-of-thought. XWhy should explain **observable inputs, states, actions, tool interactions, and intervention effects**. A local explanation may identify strong associations or intervention sensitivity, but it does not by itself prove causality, guarantee correctness, or constitute a complete safety assurance argument.

The longer-term objective is to make Agentic AI behaviour auditable at runtime while retaining the same principle used elsewhere in XWhy: explanation claims should be measurable, testable, and explicit about their limitations.
