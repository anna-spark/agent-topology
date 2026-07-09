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
(Barabási–Albert), `fully_connected` (upper-bound baseline), `empty` (no edges —
the no-communication lower bookend at identical compute budget).

**Replication & rounds.** Each condition can be run across multiple `--seeds`; the seed
varies both the drawn graph instance (for random/small-world/modular/scale-free) and LLM
stochasticity. The shipped results use a single fixed seed (42), so the reported 95%
confidence intervals are bootstrap over the 25 task instances at that seed — they capture
task-to-task variance, *not* variance across graph draws. Multi-seed replication (to
separate graph-instance variance from LLM stochasticity) is left as future work. Agents
vote after *every* round (not just the last), yielding an accuracy-vs-round curve;
pass `--final-vote-only` to vote once at the end and save calls.

## Module layout

| File | Responsibility |
|------|----------------|
| `agents.py` | `Agent` (LLM call + budget tracking) and `MultiAgentSimulator` (rounds, messaging, voting). |
| `topologies.py` | Builds the communication graphs and computes network statistics. |
| `tasks.py` | Generates/validates distributed clue puzzles (CSP-checked unique solution); single-agent baseline. |
| `prompts.py` | System / message / vote prompts (shared across all topologies). |
| `run_experiments.py` | Runs the topology sweep; appends to `results/metrics.csv`; saves per-run traces to `results/raw_logs/`. |
| `analysis.py` | Graph metrics, figures, and summary tables. |

## Two task families

The same harness runs two distinct experiments, selected with `--task-type`:

- **`logic_grid`** (default) — the distributed clue puzzle described above. Outcome is
  binary exact-match `correct`. Uses `results/tasks.json` → `results/metrics.csv` →
  `results/figures/`.
- **`fragment`** — distributed secret-code reconstruction (a pure information-flow task).
  Each agent holds one `(position, letter)` fragment; the code can only be assembled by
  relaying fragments across the graph. Outcome is a *continuous* `collective_recovery`
  score (fraction of positions correctly reconstructed by the per-position majority),
  which is far more sensitive than exact match. Uses `results/tasks_fragment.json` →
  `results/metrics_fragment.csv` → `results/figures_fragment/`.

Every step (`tasks.py`, `run_experiments.py`, `analysis.py`) takes `--task-type` and
keeps the two experiments in completely separate files, so they never clobber each other.
`analysis.py` also **auto-detects** the family from the metrics columns, so a bare
`./venv/bin/python analysis.py` analyses the logic-grid run and
`--task-type fragment` (or pointing `--metrics` at the fragment CSV) switches to recovery.

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

# 3. Main topology sweep across 3 seeds -> results/metrics.csv (+ results/raw_logs/*.json)
./venv/bin/python run_experiments.py --seeds 42 123 2024

# 4. Robustness sweep (edge deletion) — kept separate/smaller to bound cost; appends rows
./venv/bin/python run_experiments.py --topologies chain tree random fully_connected \
    --seeds 42 --drop-rates 0.1 0.2 0.3

# 5. Figures + tables -> results/figures/, results/topology_summary.csv, budget_summary.csv, failure_analysis.csv
./venv/bin/python analysis.py
```

### Fragment experiment (distributed secret-code reconstruction)

```bash
# 1. Generate fragment instances (writes results/tasks_fragment.json)
./venv/bin/python tasks.py --task-type fragment --n 25

# 2. Reference baselines (same model + prompts, no network):
#    ceiling = one agent handed every fragment; floor = one agent with a single fragment
./venv/bin/python tasks.py --task-type fragment --baseline --baseline-mode all  # -> baseline_fragment_all.csv
./venv/bin/python tasks.py --task-type fragment --baseline --baseline-mode one  # -> baseline_fragment_one.csv

# 3. Topology sweep -> results/metrics_fragment.csv (+ results/raw_logs_fragment/*.json)
./venv/bin/python run_experiments.py --task-type fragment --seeds 42

# 4. Figures + tables -> results/figures_fragment/, results/topology_summary_fragment.csv, etc.
#    (recovery metric + floor/ceiling baseline lines; auto-detected from the metrics columns)
./venv/bin/python analysis.py --task-type fragment
```

Useful flags: `run_experiments.py --topologies chain empty --tasks 0 1 --seeds 42 --rounds 2`
(quick smoke test), `--n-agents`, `--final-vote-only`, `--model`. For fragment generation,
`tasks.py --task-type fragment --code-len N` sets the secret-code length (default = n_agents).
`analysis.py` flags: `--task-type {logic_grid,fragment}`, `--metrics PATH` (override input),
`--no-graphs` (skip the slow topology visualizations).

## Outputs (`results/`)

- `metrics.csv` — one row per (task × topology × seed × drop-rate): correctness, vote
  agreement, per-round results (`round_results`), budget columns, `timestamp`, and
  `duration_sec`. Heavy message logs are **not** here.
- `raw_logs/*.json` — full per-message communication trace for every run.
- `figures/` — accuracy bar (with bootstrap CIs + baseline line), path-length vs
  accuracy, per-task boxplot, graph visualizations colored by betweenness, vote-agreement
  heatmap, budget bars, accuracy-vs-round curves, robustness curves.
- `topology_summary.csv`, `budget_summary.csv`, `graph_stats.csv`, `failure_analysis.csv`.
- **Fragment experiment** (mirrors the above with a `_fragment` suffix):
  `metrics_fragment.csv` (adds `collective_recovery`, `mean_recovery`, `best_recovery`
  columns), `tasks_fragment.json`, `raw_logs_fragment/*.json`, `baseline_fragment_all.csv`
  (ceiling) and `baseline_fragment_one.csv` (floor), `figures_fragment/`, and
  `topology_summary_fragment.csv` / `budget_summary_fragment.csv` /
  `failure_analysis_fragment.csv`. The summary table reports recovery (with bootstrap CIs)
  plus `exact_match_acc` as a secondary number; the failure table lists incompletely-
  reconstructed runs (recovery < 1.0), worst first.
- `archive/` — superseded result files from earlier (broken) runs, kept for reference.

## Notes

- Reproducible: a fixed `--seed` controls graph construction, edge-deletion, and
  vote tie-breaking; LLM calls use `temperature=0.1`.
- This is a prototype to demonstrate a controlled experiment, not a definitive
  benchmark.
