# Experiments and result format

`ExperimentRunner` keeps scenario/task, model, turn budget, vocabulary, and evaluation procedure constant while changing only approach and selection strategy. `run_all` executes the nine combinations and supports repeated runs with `RUNS_PER_CONFIGURATION`.

Each completed result includes:

- experiment ID and UTC timestamp;
- provider/model configuration and random seed;
- approach, selection strategy, scenario, task, and limits;
- initial, generated, and final agents;
- full structured conversation;
- event timeline, token counts when Ollama supplies them, errors/retries;
- termination reason and execution duration;
- evaluation metrics.

Results are saved under `results/YYYY-MM-DD/`. Failed configurations are written under the corresponding `failed/` directory rather than aborting the all-configurations loop.

`compare` prints configuration summaries, approach-pair Mann-Whitney U/Cliff’s Delta reports for the primary metrics, and the Pearson correlation between final agent count and task coverage, in addition to creating plots when matplotlib is installed.
