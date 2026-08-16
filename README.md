# Emergent Topological Capabilities of Agentic AI

A controlled prototype testing one question:

> **When the agent model, number of agents, task, prompts, and compute budget are all
> fixed, does changing only the communication *topology* change how well a collective can
> move information to where it's needed?**

The headline experiment is **fragment**: a distributed secret-code reconstruction task
that isolates pure *information flow*. `N = 20` identical LLM agents each hold **one
private `(position, letter)` fragment** of a 20-character secret code. No agent can
reconstruct the code alone; a fragment can only reach the rest of the swarm by being
relayed across the communication graph. Agents exchange short messages for `R = 3`
rounds, then everyone emits their best guess at the full code and the system's answer is
the **per-position majority vote**. The *only* thing that varies across conditions is the
graph.

Because the outcome is a *continuous* recovery score (the fraction of positions the
majority vote gets right) rather than a binary correct/incorrect, it exposes graded
differences between topologies that an exact-match metric would flatten to all-or-nothing.

## Headline result

Recovery per topology at the shipped seed (42), sorted by `collective_recovery`, against a
floor baseline of **0.05** (one agent, one fragment = `1/N`) and a ceiling of **1.0** (one
agent handed every fragment):

| topology | collective | mean | best | avg shortest path | diameter | tokens |
|---|---|---|---|---|---|---|
| fully_connected | 1.00 | 1.00 | 1.00 | 1.00 | 1 | 287k |
| random | 1.00 | 1.00 | 1.00 | 1.71 | 3 | 94k |
| scale_free | 1.00 | 1.00 | 1.00 | 1.97 | 3 | 65k |
| small_world | 1.00 | 0.98 | 1.00 | 2.14 | 4 | 63k |
| modular | 1.00 | 0.80 | 1.00 | 2.63 | 5 | 62k |
| tree | 0.35 | 0.43 | 0.80 | 4.06 | 7 | 43k |
| chain | 0.00 | 0.32 | 0.35 | 7.00 | 19 | 42k |

Two takeaways from this table:

- Recovery tracks average shortest path length. A fragment can only travel `R` hops, so
  once the graph is wider than `R` rounds can cover (tree, chain), most positions never
  reach a majority.
- The five dense graphs *tie* at `collective = 1.0`, a ceiling effect. They separate not
  on the round-3 endpoint but on **convergence speed** (how many rounds to reach 1.0) and
  **token cost**: `modular` hits the same recovery as `fully_connected` at ~4.6× fewer
  tokens.

The chain's `collective = 0.00` while `mean = 0.32` is explained by the metric's design;
see [Why the chain scores 0.00](#why-the-chain-scores-000-a-feature-not-a-bug).

## Experimental controls

Held fixed across every topology: model (`gemini/gemini-2.5-flash-lite`), number of
agents, system + agent prompts (`prompts.py`), number of rounds, max message length,
final-answer procedure (per-position majority vote), task instances, and random seed. Each
run records its token usage and LLM-call count so the "fixed budget" assumption can be
*verified* rather than assumed (`budget_summary_fragment.csv`: `n_llm_calls` is constant
at **120** across topologies by construction; only context/token *volume* varies).

## Topologies (`topologies.py`)

`chain`, `tree` (balanced binary), `random` (Erdős–Rényi, connected), `small_world`
(Watts–Strogatz), `modular` (community cliques + bridges), `scale_free`
(Barabási–Albert), `fully_connected` (upper-bound baseline), `empty` (no edges,
the no-communication lower bookend at identical compute budget).

**Replication & rounds.** Each condition can be run across multiple `--seeds`; the seed
varies both the drawn graph instance (for random/small-world/modular/scale-free) and LLM
stochasticity. The shipped results use a single fixed seed (42), so the reported 95%
confidence intervals are bootstrap over the 25 task instances at that seed; they capture
task-to-task variance, *not* variance across graph draws. Multi-seed replication (to
separate graph-instance variance from LLM stochasticity) is left as future work. Agents
vote after *every* round (not just the last), yielding a recovery-vs-round curve that is
what separates the ceiling-tied dense graphs; pass `--final-vote-only` to vote once at the
end and save calls.

## Why the chain scores 0.00 (a feature, not a bug)

`collective_recovery` is the recovery of the **per-position majority vote** across all 20
agents. For any position `p` in a chain, only ~7 agents (those within `R = 3` hops of the
agent holding `p`) ever learn its letter; the other ~13 still emit `?`. In
`_position_majority`, `?` **counts as a vote** and only loses to a real letter on an exact
tie, so every position tallies roughly "letter: 7, `?`: 13", and `?` wins outright: the
majority code comes out all-`?`, giving `collective_recovery = 0.0` even though agents near
each fragment clearly know their letters (`mean_recovery = 0.32`).

This is by design. Excluding `?` from the majority would push the chain to ~1.0 (every
position is known by someone, with no competing wrong letter), which would erase the
topology signal we're trying to measure. Counting abstentions is what makes
`collective_recovery` a **threshold** metric: a fragment must reach a *majority* of agents
to win its position, which is what separates dense graphs (info floods to a majority) from
sparse ones (it doesn't).

## Module layout

| File | Responsibility |
|------|----------------|
| `agents.py` | `Agent` (LLM call + budget tracking) and `MultiAgentSimulator` (rounds, messaging, per-position voting, recovery scoring). |
| `topologies.py` | Builds the communication graphs and computes network statistics. |
| `tasks.py` | Generates fragment instances (one `(position, letter)` per agent) and floor/ceiling baselines. |
| `prompts.py` | System / message / vote prompts (shared across all topologies). |
| `run_experiments.py` | Runs the topology sweep; appends to `results/metrics_fragment.csv`; saves per-run traces to `results/raw_logs_fragment/`. |
| `analysis.py` | Graph metrics, figures, and summary tables (auto-detects the task family). |

## Setup

```bash
python -m venv venv
./venv/bin/pip install -r requirements.txt
# Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment or .env
```

## Reproduce

```bash
# 1. Generate instances (writes results/tasks_fragment.json)
./venv/bin/python tasks.py --n 25

# 2. Reference baselines (same model + prompts, no network):
#    ceiling = one agent handed every fragment; floor = one agent with a single fragment
./venv/bin/python tasks.py --baseline --baseline-mode all  # -> baseline_fragment_all.csv (1.0)
./venv/bin/python tasks.py --baseline --baseline-mode one  # -> baseline_fragment_one.csv (0.05)

# 3. Topology sweep -> results/metrics_fragment.csv (+ results/raw_logs_fragment/*.json)
./venv/bin/python run_experiments.py --seeds 42

# 4. Figures + tables -> results/figures_fragment/, results/topology_summary_fragment.csv, etc.
#    (recovery metric + floor/ceiling baseline lines)
./venv/bin/python analysis.py
```

`tasks.py --code-len N` sets the secret-code length (default = n_agents). Useful flags:
`run_experiments.py --topologies chain empty --tasks 0 1 --seeds 42 --rounds 2` (quick
smoke test), `--n-agents`, `--final-vote-only`, `--model`. `analysis.py` flags:
`--metrics PATH` (override input), `--no-graphs` (skip the slow topology visualizations).

The harness also carries a second task family (`--task-type logic_grid`, a distributed
clue puzzle scored by binary exact-match). It's kept around but discriminates far less than
fragment; every step takes `--task-type` and writes to separate files (no `_fragment`
suffix), so the two never collide.

## Outputs (`results/`)

- `metrics_fragment.csv`: one row per (task × topology × seed × drop-rate), with the
  recovery columns (`collective_recovery`, `mean_recovery`, `best_recovery`), vote
  agreement, per-round results (`round_results`), budget columns, `timestamp`, and
  `duration_sec`. Heavy message logs are **not** here.
- `raw_logs_fragment/*.json`: full per-message communication trace for every run (watch a
  fragment hop one neighbor per round).
- `baseline_fragment_all.csv` (ceiling, 1.0) and `baseline_fragment_one.csv` (floor, 0.05).
- `figures_fragment/`: recovery bar (with bootstrap CIs + floor/ceiling lines), path-length
  vs recovery, per-task boxplot, graph visualizations colored by betweenness, vote-agreement
  heatmap, budget bars, recovery-vs-round curves, robustness curves.
- `topology_summary_fragment.csv` (recovery + CIs + graph stats + tokens, with
  `exact_match_acc` as a secondary number), `budget_summary_fragment.csv`,
  `failure_analysis_fragment.csv` (incompletely-reconstructed runs, recovery < 1.0, worst
  first).

- `raw_logs_archive_20260620/`: superseded per-message traces from earlier runs, kept for reference.

## Notes

- Reproducible: a fixed `--seed` controls graph construction, edge-deletion, and
  vote tie-breaking; LLM calls use `temperature=0.1`.
- This is a prototype to demonstrate a controlled experiment, not a definitive
  benchmark. It shows topology *gates information flow under a tight round budget*, not
  that topology changes agents' *reasoning* ability.
