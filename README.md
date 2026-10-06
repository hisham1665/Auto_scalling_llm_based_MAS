# Resource-efficient IAAG/DRTAG multi-agent reproduction

This repository is a standalone, local implementation of the core dynamic-agent methodology described in:

> Ravindu Perera, Anuradha Basnayake, and Manjusri Wickramasinghe, “Auto-scaling LLM-based multi-agent systems through dynamic integration of agents,” *Frontiers in Artificial Intelligence* (2025), DOI [10.3389/frai.2025.1638227](https://doi.org/10.3389/frai.2025.1638227).

It implements a resource-efficient reproduction of the core **Initial Automatic Agent Generation (IAAG)** and **Dynamic Real-Time Agent Generation (DRTAG)** ideas. It does not claim the paper’s exact GPT-4o numerical results. The default runtime provider is one local Ollama model, configured as `spark-x2.5-4b`.

> **Start here:** the complete installation guide is [docs/installation.md](docs/installation.md). It includes Linux, macOS, Windows PowerShell, Ollama setup, model verification, laptop-safe settings, troubleshooting, testing, API usage, and experiment commands.

## What is implemented

- Static/user-defined baseline, IAAG, and DRTAG approaches.
- LLM-based, round-robin, and seeded-random selection.
- One Ollama process/model shared by all role-specialized agents.
- Pydantic-validated generation, selection, and termination outputs.
- Bounded global conversation memory and context truncation.
- Hard limits for turns, agents, and generation requests.
- Event timelines proving that dynamically generated agents participate.
- Medical demonstration (lower-right abdominal pain) and software architecture review scenario.
- JSON result persistence, evaluation metrics, statistics helpers, and plots.
- Unit and integration tests that use a mock provider only inside tests.

The medical text is synthetic. **This demonstration is for research/educational purposes only and does not provide medical advice.**

## Architecture

```text
user task
   |
ConversationManager -- inspects task, registry, and global conversation
   |                  |
   |                  +--> AgentGenerator --> AgentFactory --> AgentRegistry
   |
AgentSelector --> selected Agent --> shared OllamaLLM --> GlobalConversation
   |
   +--> termination decision or bounded next turn
```

The registry is the authoritative active-agent state. Agents cannot modify it directly. A generated agent is created by the factory, registered, made eligible, and—in DRTAG—given a participation turn before normal termination is considered.

## Installation

For the full setup, read [docs/installation.md](docs/installation.md). The shortest Linux/macOS path is:

```bash
git clone <repository-url> multi-agent
cd multi-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

On Windows, use the PowerShell or Command Prompt instructions in the installation guide.

For development and tests:

```bash
python -m pip install -r requirements-dev.txt
```

Optional API and neural BERTScore dependencies:

```bash
python -m pip install -r requirements-optional.txt
```

## Ollama and Spark-X2.5-4B setup

Install Ollama using the instructions for your operating system at <https://ollama.com/download>. Start the local service, then install the configured model:

```bash
ollama serve
ollama pull spark-x2.5-4b
ollama list
```

If the model is published under a different exact Ollama tag, set it in `.env`:

```dotenv
OLLAMA_MODEL=your-installed-spark-model-tag
OLLAMA_BASE_URL=http://localhost:11434
```

The program validates reachability and model presence before a real experiment. It reports either `Ollama is not reachable at ...` or the exact missing-model command. No cloud API key is used.

Before running the project, verify Ollama directly:

```bash
ollama --version
ollama list
ollama run spark-x2.5-4b "Reply with one short sentence."
```

## Run a demo

With Ollama running and the model installed:

```bash
python main.py demo
```

The demo runs DRTAG with LLM selection on the medical scenario. Terminal logs and the saved JSON show `TASK_RECEIVED`, initial Doctor/Nurse registration, a manager analysis, a generated specialist, registration, selection, specialist response, and termination. Output is stored under `results/YYYY-MM-DD/`.

Individual configurations:

```bash
python main.py run --approach drtag --selection llm --scenario medical
python main.py run --approach iaag --selection round_robin --scenario medical
python main.py run --approach static --selection random --scenario software_architecture
```

List scenario agents without starting Ollama:

```bash
python main.py list-agents --scenario medical
```

## Nine-configuration experiment

```bash
# laptop smoke experiment: one run per configuration
RUNS_PER_CONFIGURATION=1 python main.py experiment --all --scenario medical

# configurable repeated evaluation
RUNS_PER_CONFIGURATION=10 python main.py experiment --all --scenario medical
```

The runner executes the same task, model, turn budget, vocabulary, and evaluation process across:

| Approach | LLM | Round robin | Random |
|---|---:|---:|---:|
| Static | yes | yes | yes |
| IAAG | yes | yes | yes |
| DRTAG | yes | yes | yes |

The random strategy uses `RANDOM_SEED`; each repeated run uses a deterministic increment from that base seed. New round-robin agents are appended to registry order. DRTAG gives a newly generated agent a first participation turn so dynamic integration is observable and testable.

## Evaluation and comparison

```bash
python main.py evaluate --results results/
python main.py compare --results results/
```

Metrics include configurable vocabulary coverage, TF-IDF richness, MTLD, pairwise topical consistency, and thematic relevance. Scikit-learn is used when available. BERTScore can be enabled after generation with `ENABLE_BERTSCORE=true`; otherwise the result explicitly labels its lexical F1 fallback. Mann-Whitney U, Pearson correlation, and Cliff’s Delta are available in `evaluation/statistics.py`.

Plots are written to `results/plots/` when matplotlib is installed.

## Testing

```bash
python3 -m pytest -q
```

The tests do not call Ollama. They inject a deterministic mock LLM to test parsing, registry behavior, all selection modes, IAAG ordering, DRTAG ordering, generated-agent participation, persistence, and evaluation. The production CLI always constructs `OllamaLLM`; there is no fake runtime mode.

## Optional API

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Endpoints: `GET /health`, `POST /run`, `GET /experiments`, `GET /experiments/{id}`, and `GET /agents`.

## Project structure

```text
agents/          Agent object, factory, and authoritative registry
api/              Optional FastAPI adapter
approaches/       Static, IAAG, and DRTAG entry points
cli/              Command-line interface
config/           Environment-backed laptop/runtime settings
conversation/     Message model, memory, and context truncation
docs/              Architecture, methodology, evaluation, and limitations
evaluation/       Coverage, TF-IDF, MTLD, BERTScore, and statistics
experiments/      Scenarios, nine configurations, runner, plots
llm/              Base interface, Ollama client, structured parser
manager/          Conversation manager, generator, and selector
prompts/          Editable persona/chain/few-shot prompt assets
results/          Generated JSON and plots (created at runtime)
tests/             Deterministic unit and integration tests
```

## Documentation map

- [Installation and local setup](docs/installation.md)
- [Architecture](docs/architecture.md)
- [Methodology and source fidelity](docs/methodology.md)
- [IAAG](docs/iaag.md)
- [DRTAG](docs/drtag.md)
- [Selection strategies](docs/selection-strategies.md)
- [Evaluation](docs/evaluation.md)
- [Experiments and result format](docs/experiments.md)
- [Implementation decisions](docs/implementation-decisions.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Limitations](docs/limitations.md)

## Resume-ready description

Designed and implemented a resource-efficient Python reproduction of IAAG and DRTAG dynamic agent integration for local Ollama LLMs. Built a registry-controlled conversation manager with Pydantic-validated agent generation, LLM/round-robin/random selection, bounded context and termination guards, event-level observability, reproducible nine-configuration experiments, lexical/neural evaluation hooks, statistical comparison utilities, persistence, CLI/API interfaces, and tests proving that a specialist generated during DRTAG becomes an active participant.

## Limitations

See [docs/limitations.md](docs/limitations.md). In particular, Spark-X2.5-4B is smaller than the paper’s GPT-4o setup, model decisions can be weaker or stochastic, local latency is hardware-dependent, this project does not reproduce exact numerical results, and the transparent fallback metrics are not equivalent to neural BERTScore.
