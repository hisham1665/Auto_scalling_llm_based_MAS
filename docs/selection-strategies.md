# Selection strategies

## LLM selection

The model receives the task, available roles/prompts, recent context, and previous turns. It returns `{ "selected_agent": "...", "reason": "..." }`. The manager validates the name against the authoritative registry. If the response cannot be parsed or names an unavailable agent, a deterministic round-robin fallback keeps the run bounded and records a warning.

## Round robin

Registered agents retain insertion order. A cursor advances through the current list. Newly registered agents are appended and therefore enter the cycle at their registry position. A generated agent receives its first forced participation turn before normal round-robin selection resumes.

## Random

The selector uses an isolated `random.Random` seeded from `RANDOM_SEED`. The seed is stored in every experiment JSON document. The generated agent is still guaranteed a first turn for DRTAG validation; subsequent choices are seeded random choices from the active registry.
