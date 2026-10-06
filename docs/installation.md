# Installation and local setup guide

This guide explains how to install and run the IAAG/DRTAG research engine on a laptop using one local Ollama model. The project does not require OpenAI, Anthropic, Google, or any other cloud API.

## 1. System requirements

### Required

- Linux, macOS, or Windows 10/11.
- Python 3.10 or newer. Python 3.11 or 3.12 is recommended.
- Git.
- Ollama installed and running for real demonstrations.
- Enough disk space for the selected Ollama model and its runtime cache.
- A machine capable of running the configured Spark model. Ollama model memory requirements depend on the model tag and quantization.

### Not required for unit tests

Ollama and the model are not required to run the automated tests. Tests inject a deterministic mock provider and never pretend that the mock is the production runtime.

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

## 6. Install and start Ollama

Install Ollama from the official download page:

<https://ollama.com/download>

The exact installation process is maintained by Ollama and differs between operating systems.

### Linux

After installing Ollama, start the service in one terminal:

```bash
ollama serve
```

Keep that terminal running. On systems configured with the Ollama service, the service may already be running.

### macOS and Windows

Start the Ollama desktop application. If the `ollama` command is available in a terminal, the same commands below can be used to check the installation.

Verify the client is available:

```bash
ollama --version
```

## 7. Install the Spark model

The default project setting is:

```dotenv
OLLAMA_MODEL=spark-x2.5-4b
```

Pull that model:

```bash
ollama pull spark-x2.5-4b
ollama list
```

Perform a direct model smoke test before using the project:

```bash
ollama run spark-x2.5-4b "Reply with one short sentence confirming that the local model is available."
```

If your Ollama installation uses a different model tag, use the exact name shown by `ollama list` and update `OLLAMA_MODEL` accordingly. The code does not hard-code the model beyond the default configuration.

## 8. Configure the project

Copy the template:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

The most important settings are:

```dotenv
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=spark-x2.5-4b
OLLAMA_TIMEOUT=120
OLLAMA_RETRIES=1
MAX_TURNS=10
MAX_AGENTS=8
MAX_AGENT_GENERATIONS_PER_RUN=4
CONTEXT_MAX_CHARS=12000
RANDOM_SEED=42
RUNS_PER_CONFIGURATION=1
```

### Laptop-friendly starting configuration

For a machine with limited memory or slow inference, begin with:

```dotenv
MAX_TURNS=4
MAX_AGENTS=5
MAX_AGENT_GENERATIONS_PER_RUN=2
MAX_RESPONSE_TOKENS=256
CONTEXT_MAX_CHARS=8000
RUNS_PER_CONFIGURATION=1
LLM_TEMPERATURE=0.1
```

Increase these values gradually after the single demo completes successfully.

## 9. Run the first real demo

Make sure Ollama is running and the model is installed, then execute:

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

## 10. Run individual configurations

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

## 11. Run all nine configurations

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

Use one run first. Local LLM inference can make 10 repetitions expensive.

## 12. Inspect and evaluate results

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

Each result contains the model name, configuration, scenario, approach, selection strategy, random seed, initial agents, generated agents, final agents, full conversation, event timeline, token information when provided by Ollama, termination reason, execution duration, errors, retries, and metrics.

## 13. Run tests without Ollama

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
- Ollama request construction with a fake HTTP transport.
- Result saving and evaluation.

The test mock is test-only. The production `main.py demo` command always uses Ollama.

## 14. Run the optional API

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

## 15. Troubleshooting

### `Ollama is not reachable at http://localhost:11434`

Start Ollama:

```bash
ollama serve
```

Then verify:

```bash
curl http://localhost:11434/api/tags
```

If Ollama is running on another host or port, set `OLLAMA_BASE_URL` in `.env`.

### `Model 'spark-x2.5-4b' is not installed`

Check installed models:

```bash
ollama list
```

Then either install the configured model:

```bash
ollama pull spark-x2.5-4b
```

or update `.env` to the exact installed model tag.

### The model response is malformed JSON

The generator, selector, and termination manager each use structured Pydantic validation. The provider receives one correction request, and the run continues with existing agents if parsing still fails. To improve model reliability:

```dotenv
LLM_TEMPERATURE=0.0
MAX_RESPONSE_TOKENS=384
```

The editable correction prompt is `prompts/correction.txt`.

### The run is too slow

Reduce local work:

```dotenv
MAX_TURNS=4
MAX_RESPONSE_TOKENS=256
CONTEXT_MAX_CHARS=8000
MAX_AGENT_GENERATIONS_PER_RUN=2
```

Run evaluation after generation instead of enabling neural BERTScore during the run. The project makes no concurrent model calls; all agents share one local provider.

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

## 16. Prompt customization

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

## 17. Important research interpretation

This project is a local, resource-efficient reproduction of the core IAAG and DRTAG methodology. It uses Spark-X2.5-4B through Ollama rather than the GPT-4o setup used in the original paper. Results should be used to inspect architecture behavior and local experimental trends, not presented as exact numerical replication of the paper.
