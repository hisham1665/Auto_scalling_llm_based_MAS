# Implementation decisions

## A. Paper-derived behavior

- Dynamic roles are selected from the task and evolving conversation.
- IAAG precedes the first conversational response.
- DRTAG evaluates after interactions and integrates executable new agents.
- LLM, round-robin, and random selection are separately comparable.
- Persona, ordered/controlled chain, and few-shot prompting are explicit prompt assets.

## B. Engineering implementation decisions

- A single synchronous `CloudFailoverLLM` instance is shared by all agents. NVIDIA NIM/Nemotron is primary and Groq/Qwen is the fallback for rate limits, token limits, authentication failures, endpoint failures, and other provider errors.
- Pydantic models validate generation, selection, and termination outputs.
- The manager forces a newly generated DRTAG agent to receive one contribution before normal termination. This makes the integration invariant testable and prevents a valid new role from being generated and immediately ignored.
- Agents are unique by normalized name and role. Registry insertion order defines round-robin order.
- Context snapshots are bounded by character count and recent-message count.
- Invalid structured output receives one correction request; further failure becomes a logged no-op.
- Results use date directories plus UUID-suffixed IDs so previous runs are not overwritten.
- Optional packages are used only when installed. The metric output labels lexical BERTScore fallback explicitly.

## C. Experimental extensions

- The software architecture scenario demonstrates domain generalization beyond medicine.
- Configurable repeated runs, plots, Pearson correlation, and optional neural BERTScore provide practical local experiment support.
- The configured keyword coverage vocabulary is a transparent local proxy rather than a claim of exact paper metric replication.
