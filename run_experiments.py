"""
run_experiments.py
------------------
Runs all topology experiments and saves results.

For each topology x task instance:
  - Builds the communication graph
  - Runs the multi-agent simulator
  - Records accuracy, vote agreement, token usage, and full log

Usage:
    python run_experiments.py                        # run all topologies, all tasks
    python run_experiments.py --topologies chain tree  # run specific topologies
    python run_experiments.py --tasks 0 1 2          # run specific task IDs
    python run_experiments.py --dry-run              # print config, don't call API
    python run_experiments.py --rounds 3             # set number of communication rounds
"""

import os
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from tasks import load_tasks
from agents import MultiAgentSimulator
from topologies import get_topology, compute_graph_stats, TOPOLOGY_BUILDERS

from dotenv import load_dotenv
load_dotenv()


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

DEFAULT_TOPOLOGIES = ["chain", "tree", "random", "small_world"]
DEFAULT_N_AGENTS   = 20
DEFAULT_N_ROUNDS   = 3
DEFAULT_MODEL      = "claude-haiku-4-5-20251001"
DEFAULT_SEED       = 42


# ---------------------------------------------------------------------------
# Single experiment run
# ---------------------------------------------------------------------------

def run_one(
    task: dict,
    topology_name: str,
    n_rounds: int,
    model: str,
    seed: int,
) -> dict:
    """Run one task under one topology. Returns result dict."""
    G = get_topology(topology_name, DEFAULT_N_AGENTS, seed=seed)

    sim = MultiAgentSimulator(
        task=task,
        graph=G,
        topology_name=topology_name,
        n_rounds=n_rounds,
        model=model,
        seed=seed,
    )

    result = sim.run()
    return result


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_log(result: dict, log_dir: Path) -> None:
    """Save full communication log for one run."""
    log_dir.mkdir(parents=True, exist_ok=True)
    fname = log_dir / f"task{result['task_id']:02d}_{result['topology']}.json"
    with open(fname, "w") as f:
        json.dump(result, f, indent=2)


def save_metrics(rows: list[dict], path: Path) -> None:
    """Save/append metrics CSV."""
    df = pd.DataFrame(rows)
    if path.exists():
        existing = pd.read_csv(path)
        df = pd.concat([existing, df], ignore_index=True).drop_duplicates(
            subset=["task_id", "topology"]
        )
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiments(
    topologies: list[str],
    task_ids: list[int] | None,
    n_rounds: int,
    model: str,
    seed: int,
    dry_run: bool,
) -> pd.DataFrame:

    tasks = load_tasks()
    if task_ids is not None:
        tasks = [t for t in tasks if t["task_id"] in task_ids]

    # Paths
    results_dir  = Path("results")
    log_dir      = results_dir / "raw_logs"
    metrics_path = results_dir / "metrics.csv"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Print config
    print("=" * 60)
    print("EXPERIMENT CONFIG")
    print(f"  Topologies : {topologies}")
    print(f"  Tasks      : {len(tasks)} instances")
    print(f"  Rounds     : {n_rounds}")
    print(f"  Model      : {model}")
    print(f"  Agents     : {DEFAULT_N_AGENTS}")
    print(f"  Seed       : {seed}")
    print(f"  Dry run    : {dry_run}")
    print("=" * 60)

    if dry_run:
        print("\nDry run complete — no API calls made.")
        return pd.DataFrame()

    # Graph stats (saved once)
    stats_rows = []
    for tname in topologies:
        G = get_topology(tname, DEFAULT_N_AGENTS, seed=seed)
        stats_rows.append(compute_graph_stats(G, tname))
    stats_df = pd.DataFrame(stats_rows).set_index("topology")
    stats_df.to_csv(results_dir / "graph_stats.csv")
    print(f"\nGraph stats saved to results/graph_stats.csv")
    print(stats_df[["n_edges", "diameter", "avg_shortest_path",
                     "avg_clustering", "max_betweenness"]].to_string())
    print()

    # Main loop
    metric_rows = []
    total = len(topologies) * len(tasks)

    with tqdm(total=total, desc="Running experiments") as pbar:
        for topology_name in topologies:
            topology_correct = 0

            for task in tasks:
                pbar.set_description(f"{topology_name} | task {task['task_id']}")

                try:
                    result = run_one(
                        task=task,
                        topology_name=topology_name,
                        n_rounds=n_rounds,
                        model=model,
                        seed=seed,
                    )
                except Exception as e:
                    print(f"\n  ERROR: {topology_name} task {task['task_id']}: {e}")
                    result = {
                        "task_id": task["task_id"],
                        "topology": topology_name,
                        "question": task["question"],
                        "correct_answer": task["answer"],
                        "majority_answer": None,
                        "correct": False,
                        "vote_agreement": 0.0,
                        "n_rounds": n_rounds,
                        "log": [],
                        "vote_counts": {},
                        "votes": {},
                        "error": str(e),
                    }

                # Save log
                save_log(result, log_dir)

                # Record metrics
                row = {
                    "task_id":        result["task_id"],
                    "topology":       result["topology"],
                    "correct":        result["correct"],
                    "majority_answer":result["majority_answer"],
                    "correct_answer": result["correct_answer"],
                    "vote_agreement": result.get("vote_agreement", 0.0),
                    "n_rounds":       n_rounds,
                    "model":          model,
                    "timestamp":      datetime.now().isoformat(),
                }
                metric_rows.append(row)
                if result["correct"]:
                    topology_correct += 1

                pbar.update(1)
                time.sleep(0.2)   # gentle rate limiting

            acc = topology_correct / len(tasks)
            print(f"\n  {topology_name:12s} accuracy: {acc:.1%}  ({topology_correct}/{len(tasks)})")

    # Save metrics
    save_metrics(metric_rows, metrics_path)
    print(f"\nMetrics saved to {metrics_path}")

    df = pd.DataFrame(metric_rows)

    # Summary table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    summary = df.groupby("topology").agg(
        accuracy=("correct", "mean"),
        n_correct=("correct", "sum"),
        n_tasks=("correct", "count"),
        avg_vote_agreement=("vote_agreement", "mean"),
    ).round(3)
    print(summary.to_string())

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topologies", nargs="+", default=DEFAULT_TOPOLOGIES,
                        choices=list(TOPOLOGY_BUILDERS.keys()),
                        help="Which topologies to run")
    parser.add_argument("--tasks", nargs="+", type=int, default=None,
                        help="Task IDs to run (default: all)")
    parser.add_argument("--rounds", type=int, default=DEFAULT_N_ROUNDS,
                        help="Number of communication rounds")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Model to use for all agents")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config only, no API calls")
    args = parser.parse_args()

    run_experiments(
        topologies=args.topologies,
        task_ids=args.tasks,
        n_rounds=args.rounds,
        model=args.model,
        seed=args.seed,
        dry_run=args.dry_run,
    )