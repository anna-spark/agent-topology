"""
run_experiments.py
------------------
Runs all topology experiments and saves results incrementally to prevent data loss.

Experimental control: everything (model, #agents, #rounds, prompts, tasks, seed) is
held fixed across topologies — only the communication graph varies. Each run also
records its token/message budget so the "fixed budget" assumption can be verified.

Examples:
    # Main sweep (all topologies, all tasks, no edge dropping)
    python run_experiments.py

    # Robustness sweep (edge-deletion at several rates)
    python run_experiments.py --drop-rates 0.0 0.1 0.2 0.3
"""

import json
import argparse
import random
import networkx as nx
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from tasks import load_tasks
from agents import MultiAgentSimulator
from topologies import get_topology, TOPOLOGY_BUILDERS
from dotenv import load_dotenv

load_dotenv()

DEFAULT_TOPOLOGIES = ["chain", "tree", "random", "small_world", "modular", "scale_free", "fully_connected"]
DEFAULT_N_AGENTS   = 20
DEFAULT_N_ROUNDS   = 3
DEFAULT_MODEL      = "gemini/gemini-2.5-flash-lite"
DEFAULT_SEED       = 42

# Columns persisted to metrics.csv (the heavy per-message `log` is saved separately).
METRIC_COLUMNS = [
    "task_id", "topology", "question", "correct_answer", "majority_answer", "correct",
    "votes", "vote_counts", "vote_agreement", "n_rounds", "model", "seed", "edge_drop_rate",
    "total_input_tokens", "total_output_tokens", "total_tokens", "n_messages", "n_llm_calls",
]


def save_result_incrementally(result: dict, path: Path):
    """Appends a single result row to the CSV (header written only when the file is new)."""
    df = pd.DataFrame([{k: result[k] for k in METRIC_COLUMNS}])
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def save_log(log: list, result: dict, log_dir: Path):
    """Persists the full per-message communication trace for one run as JSON."""
    log_dir.mkdir(parents=True, exist_ok=True)
    drop = result["edge_drop_rate"]
    suffix = f"_drop{drop}" if drop and drop > 0 else ""
    fname = log_dir / f"task{result['task_id']:02d}_{result['topology']}{suffix}.json"
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
    log_dir: Path | None = None,
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
    )
    result = simulator.run()
    if log_dir is not None:
        save_log(simulator.log, result, log_dir)
    return result


def run_experiments(args):
    tasks = load_tasks()
    if args.tasks is not None:
        wanted = set(args.tasks)
        tasks = [t for t in tasks if t["task_id"] in wanted]

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    metrics_path = results_dir / "metrics.csv"
    log_dir = results_dir / "raw_logs"

    print(f"Starting experiments. Results -> {metrics_path}, logs -> {log_dir}")
    print(f"Topologies: {args.topologies}")
    print(f"Drop rates: {args.drop_rates}")

    for drop_rate in args.drop_rates:
        for topology in args.topologies:
            print(f"--- Topology: {topology} | drop_rate={drop_rate} ---")
            for task in tqdm(tasks):
                result = run_one(
                    task=task,
                    topology_name=topology,
                    n_agents=args.n_agents,
                    n_rounds=args.rounds,
                    model=args.model,
                    seed=args.seed,
                    drop_rate=drop_rate,
                    log_dir=log_dir,
                )
                save_result_incrementally(result, metrics_path)

    print("\nExperiments complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topologies", nargs="+", default=DEFAULT_TOPOLOGIES,
                        choices=list(TOPOLOGY_BUILDERS.keys()))
    parser.add_argument("--tasks",       nargs="+", type=int,   default=None)
    parser.add_argument("--n-agents",    dest="n_agents", type=int, default=DEFAULT_N_AGENTS)
    parser.add_argument("--rounds",      type=int,   default=DEFAULT_N_ROUNDS)
    parser.add_argument("--model",       type=str,   default=DEFAULT_MODEL)
    parser.add_argument("--seed",        type=int,   default=DEFAULT_SEED)
    parser.add_argument("--drop-rates",  dest="drop_rates", nargs="+", type=float, default=[0.0],
                        help="One or more edge-deletion fractions (e.g. 0.0 0.1 0.2 0.3).")
    args = parser.parse_args()

    run_experiments(args)
