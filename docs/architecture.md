# Architecture

## Runtime components

1. `ConversationManager` owns the run loop, termination, generation limits, and event timeline.
2. `AgentRegistry` owns the active-agent set and rejects duplicate names, roles, and IDs.
3. `AgentFactory` validates generated specifications and attaches the shared `BaseLLM` provider.
4. `GlobalConversation` stores structured user and agent messages with timestamps, IDs, types, and turn numbers.
5. `ContextManager` produces bounded snapshots so a cloud provider is not sent unbounded history.
6. `AgentGenerator` requests a structured `AgentSpec` and returns a controlled no-op on invalid output after the provider retry.
7. `AgentSelector` implements the three selection strategies.
8. `CloudFailoverLLM` uses NVIDIA NIM first and switches to Groq when the active NVIDIA request fails.

## Ownership rule

Agents only respond. They cannot register or remove agents. Generation and registration pass through the manager and factory, which enforces `MAX_AGENTS`, duplicate prevention, prompt validation, and generation count limits.

## Event timeline

Each run records UTC timestamp, elapsed time, event type, and relevant IDs. DRTAG emits `RESPONSE_RECEIVED`, `MANAGER_ANALYSIS`, `AGENT_CREATED`, `AGENT_REGISTERED`, `AGENT_SELECTED`, and a generated-agent `RESPONSE_RECEIVED` in order.

## Shared cloud-client design

There is one `CloudFailoverLLM` instance shared by the run. It attempts the configured NVIDIA Nemotron model first and keeps using the configured Groq Qwen model after failover. Role specialization is represented by each agent's name, role-specific system prompt, and the shared conversation context. Calls are synchronous, avoiding concurrent provider requests and unnecessary quota consumption.
