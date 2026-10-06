# Troubleshooting checklist

## Fast diagnostic sequence

Run these commands in order:

```bash
python --version
python main.py --help
ollama --version
ollama list
curl http://localhost:11434/api/tags
python main.py list-agents --scenario medical
python -m pytest -q
```

If all commands except the last model-dependent command work, the Python project is installed correctly and the remaining issue is Ollama/model availability.

## Provider errors

| Error | Cause | Fix |
|---|---|---|
| `Ollama is not reachable at ...` | Ollama is stopped or the URL is wrong | Start Ollama and check `OLLAMA_BASE_URL` |
| `Model ... is not installed` | The configured tag is absent | Run `ollama pull <tag>` or update `OLLAMA_MODEL` |
| Request timeout | Model generation is slower than the configured timeout | Increase `OLLAMA_TIMEOUT` or reduce response/context limits |
| Repeated retries | Temporary HTTP/provider failure | Inspect Ollama logs and reduce `OLLAMA_RETRIES` while debugging |

## Resource errors

Begin with:

```dotenv
MAX_TURNS=4
MAX_AGENTS=5
MAX_AGENT_GENERATIONS_PER_RUN=2
MAX_RESPONSE_TOKENS=256
CONTEXT_MAX_CHARS=8000
```

The application deliberately uses one model provider and synchronous calls. It does not start one model instance per agent.

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

## No Ollama in the development sandbox

This repository can be compiled and tested without Ollama:

```bash
python -m pytest -q
```

The real demo intentionally fails with a clear connectivity error if Ollama is absent. It does not silently switch to a fake model.
