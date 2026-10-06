# Architecture

## Runtime components

1. `ConversationManager` owns the run loop, termination, generation limits, and event timeline.
2. `AgentRegistry` owns the active-agent set and rejects duplicate names, roles, and IDs.
3. `AgentFactory` validates generated specifications and attaches the shared `BaseLLM` provider.
4. `GlobalConversation` stores structured user and agent messages with timestamps, IDs, types, and turn numbers.
5. `ContextManager` produces bounded snapshots so a local model is not sent unbounded history.
6. `AgentGenerator` requests a structured `AgentSpec` and returns a controlled no-op on invalid output after the provider retry.
7. `AgentSelector` implements the three selection strategies.
8. `OllamaLLM` is the only production provider currently implemented.

## Ownership rule

Agents only respond. They cannot register or remove agents. Generation and registration pass through the manager and factory, which enforces `MAX_AGENTS`, duplicate prevention, prompt validation, and generation count limits.

## Event timeline

Each run records UTC timestamp, elapsed time, event type, and relevant IDs. DRTAG emits `RESPONSE_RECEIVED`, `MANAGER_ANALYSIS`, `AGENT_CREATED`, `AGENT_REGISTERED`, `AGENT_SELECTED`, and a generated-agent `RESPONSE_RECEIVED` in order.

## Single-model design

There is one `OllamaLLM` instance and one Ollama model. Role specialization is represented by each agent's name, role-specific system prompt, and the shared conversation context. Calls are synchronous, avoiding multiple local model instances.
