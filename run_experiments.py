"""
run_experiments.py
------------------
Runs all topology experiments and saves results incrementally to prevent data loss.

Experimental control: everything (model, #agents, #rounds, prompts, tasks, seed) is
held fixed across topologies; only the communication graph varies. Each run also
records its token/message budget so the "fixed budget" assumption can be verified.

Examples:
    # Main sweep (all topologies, all tasks, no edge dropping)
    python run_experiments.py

    # Robustness sweep (edge-deletion at several rates)
    python run_experiments.py --drop-rates 0.0 0.1 0.2 0.3
"""

import json
import time
import argparse
import random
from datetime import datetime
import networkx as nx
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from tasks import load_tasks
from agents import MultiAgentSimulator
from topologies import get_topology, TOPOLOGY_BUILDERS
from dotenv import load_dotenv

load_dotenv()

# Ordered most-expensive-first (by measured/estimated input tokens per call): the dense
# topologies dominate cost, so running them first means a mid-run interruption (e.g. API
# balance exhausted) leaves only the cheap topologies to finish on a small top-up.
DEFAULT_TOPOLOGIES = ["fully_connected", "scale_free", "random", "modular", "small_world", "tree", "chain"]
DEFAULT_N_AGENTS   = 20
DEFAULT_N_ROUNDS   = 3
DEFAULT_MODEL      = "gemini/gemini-2.5-flash-lite"
DEFAULT_SEED       = 42

# Columns persisted to metrics.csv (the heavy per-message `log` is saved separately).
METRIC_COLUMNS = [
    "task_id", "topology", "question", "correct_answer", "majority_answer", "correct",
    "votes", "vote_counts", "vote_agreement", "round_results", "n_rounds", "model", "seed",
    "edge_drop_rate", "total_input_tokens", "total_output_tokens", "total_tokens",
    "n_messages", "n_llm_calls", "timestamp", "duration_sec",
]
# Fragment runs add a continuous recovery score (fraction of code positions
# reconstructed), a more sensitive signal than exact-match accuracy.
FRAGMENT_EXTRA_COLUMNS = ["collective_recovery", "mean_recovery", "best_recovery"]


def metric_columns(task_type: str) -> list[str]:
    if task_type == "fragment":
        # Insert recovery columns right after `correct` for readability.
        i = METRIC_COLUMNS.index("correct") + 1
        return METRIC_COLUMNS[:i] + FRAGMENT_EXTRA_COLUMNS + METRIC_COLUMNS[i:]
    return METRIC_COLUMNS


def save_result_incrementally(result: dict, path: Path, columns: list[str]):
    """Appends a single result row to the CSV (header written only when the file is new)."""
    df = pd.DataFrame([{k: result.get(k) for k in columns}])
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def _run_key(task_id, topology, seed, drop_rate) -> tuple:
    """Identity of one (task, topology, seed, drop_rate) run, normalized for set lookup."""
    return (int(task_id), str(topology), int(seed), float(drop_rate))


def load_completed_keys(metrics_path: Path) -> set:
    """Read already-finished runs from metrics.csv so a restarted sweep resumes instead of
    re-running (and duplicate-appending) completed work."""
    if not metrics_path.exists():
        return set()
    df = pd.read_csv(metrics_path)
    needed = {"task_id", "topology", "seed", "edge_drop_rate"}
    if not needed.issubset(df.columns):
        return set()
    return {
        _run_key(r.task_id, r.topology, r.seed, r.edge_drop_rate)
        for r in df.itertuples(index=False)
    }


def save_log(log: list, result: dict, log_dir: Path):
    """Persists the full per-message communication trace for one run as JSON."""
    log_dir.mkdir(parents=True, exist_ok=True)
    drop = result["edge_drop_rate"]
    suffix = f"_drop{drop}" if drop and drop > 0 else ""
    fname = log_dir / f"task{result['task_id']:02d}_{result['topology']}_seed{result['seed']}{suffix}.json"
    with open(fname, "w") as f:
        json.dump({"metrics": result, "log": log}, f, indent=2, default=str)


def run_one(
    task: dict,
    topology_name: str,
    n_agents: int,
    n_rounds: int,
    model: str,
    seed: int,
    drop_rate: float = 0.0,
    final_vote_only: bool = False,
    log_dir: Path | None = None,
    task_type: str = "fragment",
) -> dict:
    """Run one task under one topology (with optional edge-deletion robustness test)."""
    G = get_topology(topology_name, n_agents, seed=seed)

    # Robustness test: drop a fraction of edges with a per-(task,seed) reproducible RNG.
    if drop_rate > 0 and G.number_of_edges() > 0:
        G = G.copy()
        rng = random.Random(seed + task["task_id"])
        n_drop = int(len(G.edges()) * drop_rate)
        if n_drop > 0:
            G.remove_edges_from(rng.sample(list(G.edges()), n_drop))

    adj_matrix = nx.to_numpy_array(G, nodelist=range(n_agents), dtype=int).tolist()

    simulator = MultiAgentSimulator(
        task=task,
        adj_matrix=adj_matrix,
        n_rounds=n_rounds,
        model=model,
        topology_name=topology_name,
        seed=seed,
        edge_drop_rate=drop_rate,
        final_vote_only=final_vote_only,
        task_type=task_type,
    )
    t0 = time.time()
    result = simulator.run()
    result["duration_sec"] = round(time.time() - t0, 2)
    result["timestamp"] = datetime.now().isoformat(timespec="seconds")
    if log_dir is not None:
        save_log(simulator.log, result, log_dir)
    return result


def run_experiments(args):
    # Per-task-type defaults keep the fragment experiment fully separate from the
    # logic-grid data (different tasks file, metrics file, and log dir).
    is_fragment = args.task_type == "fragment"
    tasks_file = args.tasks_file or ("results/tasks_fragment.json" if is_fragment else "results/tasks.json")
    tasks = load_tasks(tasks_file)
    if args.tasks is not None:
        wanted = set(args.tasks)
        tasks = [t for t in tasks if t["task_id"] in wanted]

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    metrics_path = Path(args.metrics_file) if args.metrics_file else (
        results_dir / ("metrics_fragment.csv" if is_fragment else "metrics.csv"))
    log_dir = results_dir / ("raw_logs_fragment" if is_fragment else "raw_logs")
    columns = metric_columns(args.task_type)

    completed = set() if args.no_resume else load_completed_keys(metrics_path)

    print(f"Starting experiments (task_type={args.task_type}). Results -> {metrics_path}, logs -> {log_dir}")
    print(f"Tasks file: {tasks_file} ({len(tasks)} instances)")
    print(f"Topologies: {args.topologies}")
    print(f"Seeds: {args.seeds} | Drop rates: {args.drop_rates} | "
          f"voting: {'final-only' if args.final_vote_only else 'per-round'}")
    if completed:
        print(f"Resuming: {len(completed)} runs already in {metrics_path} will be skipped.")

    skipped = 0
    failed = 0
    for seed in args.seeds:
        for drop_rate in args.drop_rates:
            for topology in args.topologies:
                pending = [t for t in tasks
                           if _run_key(t["task_id"], topology, seed, drop_rate) not in completed]
                if not pending:
                    print(f"--- Topology: {topology} | seed={seed} | drop_rate={drop_rate} "
                          f"(all {len(tasks)} tasks already done, skipping) ---")
                    skipped += len(tasks)
                    continue
                print(f"--- Topology: {topology} | seed={seed} | drop_rate={drop_rate} "
                      f"({len(pending)}/{len(tasks)} remaining) ---")
                skipped += len(tasks) - len(pending)
                for task in tqdm(pending):
                    try:
                        result = run_one(
                            task=task,
                            topology_name=topology,
                            n_agents=args.n_agents,
                            n_rounds=args.rounds,
                            model=args.model,
                            seed=seed,
                            drop_rate=drop_rate,
                            final_vote_only=args.final_vote_only,
                            log_dir=log_dir,
                            task_type=args.task_type,
                        )
                    except Exception as e:
                        # A single run that exhausts its per-call retries (e.g. a long Gemini
                        # 503 outage) must NOT kill the whole sweep. Log it and move on; we
                        # deliberately do NOT write a metrics row, so this (task, topology,
                        # seed, drop_rate) stays "incomplete" and a later resume re-runs it
                        # instead of permanently baking a transient failure into the dataset.
                        print(f"\n  ERROR: {topology} task {task['task_id']} "
                              f"(seed={seed}, drop={drop_rate}) failed, will retry on resume: {e}")
                        failed += 1
                        continue
                    save_result_incrementally(result, metrics_path, columns)
                    completed.add(_run_key(task["task_id"], topology, seed, drop_rate))

    msg = f"\nExperiments complete. ({skipped} run(s) skipped as already done"
    if failed:
        msg += f"; {failed} run(s) failed and will be retried on the next resume"
    print(msg + ".)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topologies", nargs="+", default=DEFAULT_TOPOLOGIES,
                        choices=list(TOPOLOGY_BUILDERS.keys()))
    parser.add_argument("--tasks",       nargs="+", type=int,   default=None)
    parser.add_argument("--n-agents",    dest="n_agents", type=int, default=DEFAULT_N_AGENTS)
    parser.add_argument("--rounds",      type=int,   default=DEFAULT_N_ROUNDS)
    parser.add_argument("--model",       type=str,   default=DEFAULT_MODEL)
    parser.add_argument("--seeds",       nargs="+",  type=int, default=[DEFAULT_SEED],
                        help="One or more seeds; each replicates the full sweep (varies graph instance + LLM).")
    parser.add_argument("--drop-rates",  dest="drop_rates", nargs="+", type=float, default=[0.0],
                        help="One or more edge-deletion fractions (e.g. 0.0 0.1 0.2 0.3).")
    parser.add_argument("--final-vote-only", dest="final_vote_only", action="store_true",
                        help="Vote only after the last round (saves ~N*(R-1) calls; disables per-round curve).")
    parser.add_argument("--no-resume", dest="no_resume", action="store_true",
                        help="Ignore existing metrics.csv and re-run everything (default: skip completed runs).")
    parser.add_argument("--task-type", dest="task_type", choices=["logic_grid", "fragment"],
                        default="fragment",
                        help="Task family (default). 'fragment' = distributed secret-code "
                             "reconstruction (pure information-flow); 'logic_grid' = the "
                             "companion clue puzzle. Each uses separate tasks/metrics files.")
    parser.add_argument("--tasks-file", dest="tasks_file", type=str, default=None,
                        help="Override tasks JSON path (default derived from --task-type).")
    parser.add_argument("--metrics-file", dest="metrics_file", type=str, default=None,
                        help="Override metrics CSV path (default derived from --task-type).")
    args = parser.parse_args()

    run_experiments(args)
