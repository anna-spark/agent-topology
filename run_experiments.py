"""
run_experiments.py
------------------
Runs all topology experiments and saves results.

Default model: gemini/gemini-2.0-flash-lite  (~free tier / very low cost)
Set GEMINI_API_KEY in your environment before running.

To use Claude instead:
    python run_experiments.py --model claude-haiku-4-5-20251001
    (requires ANTHROPIC_API_KEY)
"""

import os
import json
import argparse
import time
import random
from pathlib import Path
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from tasks import load_tasks
from agents import MultiAgentSimulator
from topologies import get_topology, compute_graph_stats, TOPOLOGY_BUILDERS

from dotenv import load_dotenv
load_dotenv()

DEFAULT_TOPOLOGIES = ["chain", "tree", "random", "small_world", "modular", "scale_free", "fully_connected"]
DEFAULT_N_AGENTS   = 20
DEFAULT_N_ROUNDS   = 3
DEFAULT_MODEL      = "gemini/gemini-2.0-flash-lite"   # ~free / near-zero cost
DEFAULT_SEED       = 42


def run_one(
    task: dict,
    topology_name: str,
    n_rounds: int,
    model: str,
    seed: int,
    drop_rate: float = 0.0,
) -> dict:
    """Run one task under one topology with optional edge-deletion robustness test."""
    G = get_topology(topology_name, DEFAULT_N_AGENTS, seed=seed)

    if drop_rate > 0.0 and G.number_of_edges() > 0:
        rng = random.Random(seed + task["task_id"])
        edges = list(G.edges())
        num_to_drop = int(len(edges) * drop_rate)
        if num_to_drop > 0:
            to_drop = rng.sample(edges, num_to_drop)
            G = G.copy()
            G.remove_edges_from(to_drop)

    sim = MultiAgentSimulator(
        task=task,
        graph=G,
        topology_name=topology_name,
        n_rounds=n_rounds,
        model=model,
        seed=seed,
    )
    return sim.run()


def save_log(result: dict, log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    fname = log_dir / f"task{result['task_id']:02d}_{result['topology']}.json"
    with open(fname, "w") as f:
        json.dump(result, f, indent=2)


def save_metrics(rows: list[dict], path: Path) -> None:
    df = pd.DataFrame(rows)
    if path.exists():
        existing = pd.read_csv(path)
        df = pd.concat([existing, df], ignore_index=True).drop_duplicates(
            subset=["task_id", "topology"]
        )
    df.to_csv(path, index=False)


def run_experiments(
    topologies: list[str],
    task_ids: list[int] | None,
    n_rounds: int,
    model: str,
    seed: int,
    dry_run: bool,
    drop_rate: float = 0.0,
) -> pd.DataFrame:

    tasks = load_tasks()
    if task_ids is not None:
        tasks = [t for t in tasks if t["task_id"] in task_ids]

    results_dir  = Path("results")
    log_dir      = results_dir / "raw_logs"
    metrics_path = results_dir / "metrics.csv"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EXPERIMENT CONFIG")
    print(f"  Topologies : {topologies}")
    print(f"  Tasks      : {len(tasks)} instances")
    print(f"  Rounds     : {n_rounds}")
    print(f"  Model      : {model}")
    print(f"  Seed       : {seed}")
    print(f"  Drop Rate  : {drop_rate:.1%}")
    print(f"  Dry run    : {dry_run}")
    print("=" * 60)

    if dry_run:
        print("\nDry run complete — no API calls made.")
        return pd.DataFrame()

    # Compute and save graph statistics once
    stats_rows = []
    for tname in topologies:
        G = get_topology(tname, DEFAULT_N_AGENTS, seed=seed)
        stats_rows.append(compute_graph_stats(G, tname))
    stats_df = pd.DataFrame(stats_rows).set_index("topology")
    stats_df.to_csv(results_dir / "graph_stats.csv")

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
                        drop_rate=drop_rate,
                    )
                except Exception as e:
                    print(f"\n  ERROR: {topology_name} task {task['task_id']}: {e}")
                    result = {
                        "task_id":        task["task_id"],
                        "topology":       topology_name,
                        "question":       task["question"],
                        "correct_answer": task["answer"],
                        "majority_answer":None,
                        "correct":        False,
                        "vote_agreement": 0.0,
                        "n_rounds":       n_rounds,
                        "log":            [],
                        "vote_counts":    {},
                        "votes":          {},
                        "error":          str(e),
                    }

                save_log(result, log_dir)

                row = {
                    "task_id":        result["task_id"],
                    "topology":       result["topology"],
                    "correct":        result["correct"],
                    "majority_answer":result["majority_answer"],
                    "correct_answer": result["correct_answer"],
                    "vote_agreement": result.get("vote_agreement", 0.0),
                    "n_rounds":       n_rounds,
                    "model":          model,
                    "edge_drop_rate": drop_rate,
                    "timestamp":      datetime.now().isoformat(),
                }
                metric_rows.append(row)
                if result["correct"]:
                    topology_correct += 1

                pbar.update(1)
                # Small sleep to respect API rate limits; adjust per provider
                time.sleep(0.1)

            acc = topology_correct / len(tasks)
            print(f"\n  {topology_name:15s} accuracy: {acc:.1%}  ({topology_correct}/{len(tasks)})")

    save_metrics(metric_rows, metrics_path)
    df = pd.DataFrame(metric_rows)

    summary = df.groupby("topology").agg(
        accuracy=("correct", "mean"),
        n_correct=("correct", "sum"),
        n_tasks=("correct", "count"),
        avg_vote_agreement=("vote_agreement", "mean"),
    ).round(3)
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(summary.to_string())

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topologies", nargs="+", default=DEFAULT_TOPOLOGIES,
                        choices=list(TOPOLOGY_BUILDERS.keys()))
    parser.add_argument("--tasks",      nargs="+", type=int, default=None)
    parser.add_argument("--rounds",     type=int,  default=DEFAULT_N_ROUNDS)
    parser.add_argument("--model",      type=str,  default=DEFAULT_MODEL)
    parser.add_argument("--seed",       type=int,  default=DEFAULT_SEED)
    parser.add_argument("--drop-rate",  type=float, default=0.0,
                        help="Fraction of graph edges to randomly drop (robustness test)")
    parser.add_argument("--dry-run",    action="store_true")
    args = parser.parse_args()

    run_experiments(
        topologies=args.topologies,
        task_ids=args.tasks,
        n_rounds=args.rounds,
        model=args.model,
        seed=args.seed,
        dry_run=args.dry_run,
        drop_rate=args.drop_rate,
    )
