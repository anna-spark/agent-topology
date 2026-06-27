# Emergent Topological Capabilities of Agentic AI

A controlled prototype testing one question:

> **When the agent model, number of agents, task, prompts, and compute budget are all
> fixed, does changing only the communication *topology* change collective
> problem-solving performance?**

`N = 20` identical LLM agents each receive **one private clue** to a distributed
logic-grid puzzle (5 fictional music artists, each with a unique
genre/decade/nationality/award). No single agent can solve the puzzle alone. Agents
exchange short messages over a communication graph for `R = 3` rounds, then everyone
casts a vote and the **majority answer** is the system's answer. The *only* thing that
varies across conditions is the graph.

## Experimental controls

Held fixed across every topology: model (`gemini/gemini-2.5-flash-lite`), number of
agents, system + agent prompts (`prompts.py`), number of rounds, max message length,
final-answer procedure (majority vote), task instances, and random seed. Each run
records its token usage and LLM-call count so the "fixed budget" assumption can be
*verified* rather than assumed (`budget_summary.csv` — `n_llm_calls` is constant across
topologies by construction; only context size varies).

## Topologies (`topologies.py`)

`chain`, `tree` (balanced binary), `random` (Erdős–Rényi, connected), `small_world`
(Watts–Strogatz), `modular` (community cliques + bridges), `scale_free`
(Barabási–Albert), `fully_connected` (upper-bound baseline).

## Module layout

| File | Responsibility |
|------|----------------|
| `agents.py` | `Agent` (LLM call + budget tracking) and `MultiAgentSimulator` (rounds, messaging, voting). |
| `topologies.py` | Builds the communication graphs and computes network statistics. |
| `tasks.py` | Generates/validates distributed clue puzzles (CSP-checked unique solution); single-agent baseline. |
| `prompts.py` | System / message / vote prompts (shared across all topologies). |
| `run_experiments.py` | Runs the topology sweep; appends to `results/metrics.csv`; saves per-run traces to `results/raw_logs/`. |
| `analysis.py` | Graph metrics, figures, and summary tables. |

## Setup

```bash
python -m venv venv
./venv/bin/pip install -r requirements.txt
# Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment or .env
```

## Reproduce

```bash
# 1. Generate task instances (writes results/tasks.json); --verify-only re-checks solvability
./venv/bin/python tasks.py --n 25

# 2. Single-agent baseline (same model + prompts, no communication) -> results/baseline.csv
./venv/bin/python tasks.py --baseline

# 3. Main topology sweep -> results/metrics.csv (+ results/raw_logs/*.json)
./venv/bin/python run_experiments.py

# 4. Robustness sweep (edge deletion); appends rows tagged with edge_drop_rate
./venv/bin/python run_experiments.py --drop-rates 0.1 0.2 0.3

# 5. Figures + tables -> results/figures/, results/topology_summary.csv, budget_summary.csv, failure_analysis.csv
./venv/bin/python analysis.py
```

Useful flags: `run_experiments.py --topologies chain tree --tasks 0 1 --rounds 1`
(quick smoke test), `--n-agents`, `--seed`, `--model`.

## Outputs (`results/`)

- `metrics.csv` — one row per (task × topology × drop-rate): correctness, vote
  agreement, and budget columns. Heavy message logs are **not** here.
- `raw_logs/*.json` — full per-message communication trace for every run.
- `figures/` — accuracy bar (with bootstrap CIs + baseline line), path-length vs
  accuracy, per-task boxplot, graph visualizations colored by betweenness, vote-agreement
  heatmap, budget bars, robustness curves.
- `topology_summary.csv`, `budget_summary.csv`, `graph_stats.csv`, `failure_analysis.csv`.
- `archive/` — superseded result files from earlier (broken) runs, kept for reference.

## Notes

- Reproducible: a fixed `--seed` controls graph construction, edge-deletion, and
  vote tie-breaking; LLM calls use `temperature=0.1`.
- This is a prototype to demonstrate a controlled experiment, not a definitive
  benchmark.
