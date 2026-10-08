# Installation and cloud setup guide

This guide explains how to install and run the IAAG/DRTAG research engine with cloud chat models. The production runtime uses NVIDIA NIM/Nemotron first and Groq/Qwen as automatic failover. It does not use Ollama or a local model.

## 1. System requirements

### Required

- Linux, macOS, or Windows 10/11.
- Python 3.10 or newer. Python 3.11 or 3.12 is recommended.
- Git.
- An NVIDIA API key for the primary provider and/or a Groq API key for fallback.
- Network access to the configured provider endpoints.

### Not required for unit tests

Cloud keys and network access are not required to run the automated tests. Tests inject a deterministic mock provider and use fake HTTP transports for cloud-client behavior.

## 2. Get the project

From a terminal:

```bash
git clone <repository-url> multi-agent
cd multi-agent
```

If the project was provided as a folder instead of a Git repository, open a terminal in that folder and verify that `main.py`, `requirements.txt`, `agents/`, `manager/`, and `llm/` are present.

## 3. Create a Python virtual environment

### Linux and macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation for the current user, use the standard Python executable directly or consult the Python/PowerShell execution-policy documentation. Activation is convenient but is not required after the environment has been created.

### Windows Command Prompt

```bat
py -3 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
```

## 4. Install dependencies

For runtime, evaluation, and plotting:

```bash
python -m pip install -r requirements.txt
```

For development and tests:

```bash
python -m pip install -r requirements-dev.txt
```

Optional dependencies are not needed for the core CLI:

```bash
python -m pip install -r requirements-optional.txt
```

The optional file adds:

- FastAPI and Uvicorn for the API.
- `bert-score` for optional neural BERTScore evaluation.

## 5. Verify Python installation

Run these commands from the project root:

```bash
python --version
python -m compileall -q .
python main.py --help
python main.py list-agents --scenario medical
```

Expected behavior is a CLI help screen, successful compilation, and a list containing Doctor, Nurse, Radiologist, Surgeon, and Gastroenterologist for the medical static scenario.

## 6. Configure cloud providers

Copy the template:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Fill in the provider keys without putting them in source files or command history. `.env` is ignored by Git:

```dotenv
NVIDIA_API_KEY=your_nvidia_api_key
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=nvidia/llama-3.3-nemotron-super-49b-v1.5
NVIDIA_TIMEOUT=90
NVIDIA_RETRIES=0
GROQ_API_KEY=your_groq_api_key
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_MODEL=qwen/qwen3-32b
GROQ_TIMEOUT=90
GROQ_RETRIES=0
CLOUD_REASONING_FORMAT=hidden
MAX_TURNS=10
MAX_AGENTS=8
MAX_AGENT_GENERATIONS_PER_RUN=4
CONTEXT_MAX_CHARS=12000
RANDOM_SEED=42
RUNS_PER_CONFIGURATION=1
```

NVIDIA is attempted first whenever `NVIDIA_API_KEY` is configured. If its request fails because of a quota/rate limit, token limit, authentication problem, endpoint problem, timeout, or another provider error, the client switches to Groq and continues the run with `GROQ_MODEL`. If only one key is present, that provider is used directly. At least one key is required for a real run.

The model names can be changed to models enabled for your accounts. The default NVIDIA model is Nemotron and the default Groq model is Qwen 3 32B.

Existing `.env` files may also use `Nvidia`/`Nmodel` and `Groq`/`GModel` for the corresponding API keys/model names. The standard uppercase variable names take precedence when both are present.

`CLOUD_REASONING_FORMAT=hidden` keeps internal reasoning out of returned text and reduces the chance that structured JSON responses consume the output budget. For Nemotron 3.5 Lightning this sends NVIDIA's `chat_template_kwargs={"enable_thinking": false}` option; for older Nemotron variants it adds `/no_think`; for Groq Qwen it sends Groq's `reasoning_format=hidden` option.

### Request-friendly starting configuration

For a first cloud run, begin with:

```dotenv
MAX_TURNS=4
MAX_AGENTS=5
MAX_AGENT_GENERATIONS_PER_RUN=2
MAX_RESPONSE_TOKENS=256
CONTEXT_MAX_CHARS=8000
RUNS_PER_CONFIGURATION=1
LLM_TEMPERATURE=0.1
```

These limits reduce request size and cloud quota consumption. Increase them gradually after the single demo completes successfully.

## 7. Run the first real demo

```bash
python main.py demo
```

The command runs DRTAG with LLM-based selection on the synthetic medical scenario. The terminal and saved result should show a sequence similar to:

```text
TASK_RECEIVED
INITIAL_AGENTS_REGISTERED: Doctor, Nurse
AGENT_SELECTED: Doctor or Nurse
RESPONSE_RECEIVED
MANAGER_ANALYSIS
AGENT_CREATED: dynamically generated specialist
AGENT_REGISTERED: dynamically generated specialist
AGENT_SELECTED: dynamically generated specialist
RESPONSE_RECEIVED: dynamically generated specialist
TERMINATION
EXPERIMENT_COMPLETE
```

The generated name is model-dependent. Do not expect the model to always choose the same specialist.

## 8. Run individual configurations

```bash
python main.py run --approach static --selection llm --scenario medical
python main.py run --approach iaag --selection round_robin --scenario medical
python main.py run --approach drtag --selection random --scenario medical
```

Available approaches:

```text
static
iaag
drtag
```

Available selection strategies:

```text
llm
round_robin
random
```

Available scenarios:

```text
medical
software_architecture
```

The medical scenario is synthetic and educational. It must not be used for diagnosis or treatment decisions.

## 9. Run all nine configurations

Start with one run per configuration:

```bash
RUNS_PER_CONFIGURATION=1 python main.py experiment --all --scenario medical
```

For the non-medical scenario:

```bash
RUNS_PER_CONFIGURATION=1 python main.py experiment --all --scenario software_architecture
```

For a larger study:

```bash
RUNS_PER_CONFIGURATION=10 python main.py experiment --all --scenario medical
```

Use one run first. Ten repetitions can consume substantial cloud quota.

## 10. Inspect and evaluate results

Results are saved as non-overwriting JSON documents:

```text
results/
└── YYYY-MM-DD/
    ├── experiment_HHMMSS_<id>.json
    └── failed/
        └── experiment_HHMMSS_<id>.json
```

Recompute and print metrics:

```bash
python main.py evaluate --results results/
```

Print experiment summaries, statistical comparisons, and create plots:

```bash
python main.py compare --results results/
```

Plots are written to:

```text
results/plots/
```

Each result contains the active provider/model configuration, scenario, approach, selection strategy, random seed, initial agents, generated agents, final agents, full conversation, event timeline, token information when provided by the cloud API, termination reason, execution duration, errors, retries, and metrics.

## 11. Run tests without cloud calls

```bash
python -m pytest -q
```

The tests verify:

- Agent validation and serialization.
- Registry duplicate prevention.
- Structured response parsing.
- All three selection strategies.
- Conversation and termination behavior.
- IAAG generation before the first turn.
- DRTAG generation during a conversation.
- Generated-agent registration and participation.
- Cloud request construction and NVIDIA-to-Groq failover with fake HTTP transports.
- Result saving and evaluation.

The test mock is test-only. The production `main.py demo` command always uses the NVIDIA-first cloud client.

## 12. Run the optional API

Install optional dependencies:

```bash
python -m pip install -r requirements-optional.txt
```

Start the API:

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Check health:

```bash
curl http://127.0.0.1:8000/health
```

Run an experiment:

```bash
curl -X POST http://127.0.0.1:8000/run \
  -H "Content-Type: application/json" \
  -d '{"approach":"drtag","selection":"llm","scenario":"medical","max_turns":4}'
```

On Windows PowerShell, use `Invoke-RestMethod` or a REST client if the multiline `curl` syntax is inconvenient.

## 13. Troubleshooting

### `Configure NVIDIA_API_KEY and/or GROQ_API_KEY`

Add at least one cloud provider key to `.env`. With both keys present, NVIDIA is tried first and Groq is used after an NVIDIA failure.

### NVIDIA or Groq returns a model/endpoint error

Confirm that the model is enabled for the corresponding account and that the base URL includes `/v1`:

```dotenv
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
GROQ_BASE_URL=https://api.groq.com/openai/v1
```

The application sends `POST /chat/completions` to those base URLs. It does not send requests to Ollama.

### The model response is malformed JSON

The generator, selector, and termination manager each use structured Pydantic validation. The provider receives one correction request, and the run continues with existing agents if parsing still fails. To improve model reliability:

```dotenv
LLM_TEMPERATURE=0.0
MAX_RESPONSE_TOKENS=384
```

The editable correction prompt is `prompts/correction.txt`.

### The run is too slow

Reduce request size and cloud quota usage:

```dotenv
MAX_TURNS=4
MAX_RESPONSE_TOKENS=256
CONTEXT_MAX_CHARS=8000
MAX_AGENT_GENERATIONS_PER_RUN=2
```

Run evaluation after generation instead of enabling neural BERTScore during the run. The project makes no concurrent model calls; all agents share one cloud client.

### The run stops at the turn limit

This is expected when the termination model does not return `end_conversation: true`. Increase `MAX_TURNS` only after confirming that the conversation is not needlessly repetitive.

### Import errors after installation

Confirm the virtual environment is active and install the correct dependency set:

```bash
python -m pip install -r requirements-dev.txt
python -c "import pydantic, requests; print('dependencies available')"
```

### API import fails

Install the optional requirements:

```bash
python -m pip install -r requirements-optional.txt
```

## 14. Prompt customization

Prompt files are loaded from `PROMPTS_DIR` and can be edited without modifying Python code:

```text
prompts/conversation_manager.txt
prompts/agent_generator.txt
prompts/agent_selector.txt
prompts/termination.txt
prompts/few_shot_examples.txt
prompts/correction.txt
```

Keep the JSON response shapes unchanged when modifying prompts. The parser and Pydantic models are intentionally strict so experiments do not silently accept invalid manager decisions.

## 15. Important research interpretation

This project is a cloud-backed, resource-efficient reproduction of the core IAAG and DRTAG methodology. It uses configurable NVIDIA Nemotron and Groq Qwen models rather than the GPT-4o setup used in the original paper. Results should be used to inspect architecture behavior and provider/model trends, not presented as exact numerical replication of the paper.
