# Methodology and source fidelity

The paper is the primary methodological reference. This implementation preserves the requested distinction between a fixed baseline, initial automatic agent generation, and real-time generation/integration. It also exposes the paper-related prompting and selection comparisons as explicit components.

## Paper-derived behavior

- Agents can be generated from task/conversation requirements.
- IAAG creates the initial automatically selected/generated roles before collaboration.
- DRTAG revisits the requirements during collaboration and dynamically integrates a new role.
- Selection is compared using LLM/prompt-based, round-robin, and randomized methods.
- Collaboration is evaluated for task-related content and language/topical properties.

## Prompting

Persona prompting is used in the generated role and standard agent instructions. Chain prompting is represented as a concise ordered checklist in the manager/generator prompt; hidden chain-of-thought is not requested or stored. Few-shot examples are editable in `prompts/few_shot_examples.txt` and their count is controlled by `FEW_SHOT_EXAMPLES`.

## Reproduction scope

The original experiments used GPT-4o. This project uses configurable cloud models, defaulting to NVIDIA Nemotron with Groq Qwen failover, and is therefore a resource-efficient implementation/reproduction of the core methodology rather than an exact numerical reproduction.
