# Troubleshooting checklist

## Fast diagnostic sequence

Run these commands in order:

```bash
python --version
python main.py --help
python main.py list-agents --scenario medical
python -m pytest -q
```

If these commands work, the Python project is installed correctly. A real `demo` additionally requires at least one configured cloud API key and network access.

## Provider errors

| Error | Cause | Fix |
|---|---|---|
| `Configure NVIDIA_API_KEY and/or GROQ_API_KEY` | Neither cloud provider is configured | Add at least one key to the local `.env` file |
| `NVIDIA request failed ... switching to Groq fallback` | NVIDIA rate limit, token limit, auth, endpoint, timeout, or other provider error | Check NVIDIA account/model settings; the current run continues through Groq when its key is configured |
| Both provider requests fail | The primary and fallback providers both rejected the request | Check status details in the terminal/result `errors`, account quotas, model names, and endpoint URLs |
| Request timeout | Provider response exceeded the configured timeout | Increase `NVIDIA_TIMEOUT`/`GROQ_TIMEOUT`, or reduce response/context limits |
| Repeated retries | Temporary HTTP/provider failure | Inspect provider status/quota and adjust `NVIDIA_RETRIES`/`GROQ_RETRIES` |

## Resource errors

Begin with:

```dotenv
MAX_TURNS=4
MAX_AGENTS=5
MAX_AGENT_GENERATIONS_PER_RUN=2
MAX_RESPONSE_TOKENS=256
CONTEXT_MAX_CHARS=8000
```

The application deliberately uses one shared cloud client and synchronous calls. It does not start one provider session per agent.

## Prompt/output errors

The model must return JSON for manager decisions. The parser tries direct JSON, fenced JSON, and extracted JSON objects before issuing one correction request. If the second attempt fails, the run records an error and continues using the current registry.

Inspect the result JSON fields:

```text
errors
retries
events
termination_reason
```

Do not remove validation to make a run appear successful; malformed decisions should remain observable.

## No cloud access in the development sandbox

This repository can be compiled and tested without cloud access:

```bash
python -m pytest -q
```

The real demo fails with a clear configuration/provider error if no cloud key is available. It does not silently switch to a fake or local model.
