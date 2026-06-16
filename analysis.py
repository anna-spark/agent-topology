"""
analysis.py
-----------
Loads experiment results and produces figures + tables for the presentation.

Figures produced:
  1. Bar plot — accuracy by topology
  2. Scatter — avg shortest path length vs accuracy
  3. Box plot — per-task accuracy distribution by topology
  4. Graph visualizations colored by betweenness centrality
  5. Heatmap — which agents voted correctly by topology

Tables produced:
  - topology_summary.csv  : accuracy + graph stats combined
  - failure_analysis.csv  : tasks where majority was wrong

Usage:
    python analysis.py                    # run all analysis on saved results
    python analysis.py --no-graphs        # skip network visualizations (faster)
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx

from topologies import get_topology, compute_graph_stats, TOPOLOGY_BUILDERS

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

TOPOLOGY_COLORS = {
    "chain":       "#4C72B0",
    "tree":        "#DD8452",
    "random":      "#55A868",
    "small_world": "#C44E52",
}

plt.rcParams.update({
    "font.family":    "sans-serif",
    "font.size":      12,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         150,
})

FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv("results/metrics.csv")
    stats   = pd.read_csv("results/graph_stats.csv")
    return metrics, stats


# ---------------------------------------------------------------------------
# Figure 1: Accuracy by topology (bar chart)
# ---------------------------------------------------------------------------

def plot_accuracy_bar(metrics: pd.DataFrame) -> None:
    summary = metrics.groupby("topology").agg(
        accuracy=("correct", "mean"),
        se=("correct", lambda x: x.std() / np.sqrt(len(x))),
    ).reset_index()

    # Order by accuracy descending
    summary = summary.sort_values("accuracy", ascending=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(
        summary["topology"],
        summary["accuracy"],
        yerr=summary["se"],
        color=[TOPOLOGY_COLORS.get(t, "#888") for t in summary["topology"]],
        capsize=5,
        width=0.55,
        error_kw={"elinewidth": 1.5},
    )

    # Annotate bars
    for bar, acc in zip(bars, summary["accuracy"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
            f"{acc:.0%}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_xlabel("Communication Topology", fontsize=12)
    ax.set_title("Collective Accuracy by Communication Topology", fontsize=13, pad=12)
    ax.axhline(0.2, color="gray", linestyle="--", linewidth=1, label="Single-agent baseline (random)")
    ax.legend(fontsize=10)

    plt.tight_layout()
    path = FIGURES_DIR / "fig1_accuracy_bar.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 2: Avg shortest path vs accuracy (scatter)
# ---------------------------------------------------------------------------

def plot_path_vs_accuracy(metrics: pd.DataFrame, stats: pd.DataFrame) -> None:
    summary = metrics.groupby("topology")["correct"].mean().reset_index()
    summary.columns = ["topology", "accuracy"]
    merged = summary.merge(stats[["topology", "avg_shortest_path", "max_betweenness"]], on="topology")

    fig, ax = plt.subplots(figsize=(6, 4.5))

    for _, row in merged.iterrows():
        color = TOPOLOGY_COLORS.get(row["topology"], "#888")
        ax.scatter(row["avg_shortest_path"], row["accuracy"],
                   color=color, s=180, zorder=3)
        ax.annotate(
            row["topology"].replace("_", " "),
            (row["avg_shortest_path"], row["accuracy"]),
            textcoords="offset points", xytext=(8, 4),
            fontsize=10, color=color,
        )

    ax.set_xlabel("Average Shortest Path Length", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Information Reachability vs. Collective Accuracy", fontsize=13, pad=12)
    ax.set_ylim(0, 1.05)

    # Trend line
    x = merged["avg_shortest_path"].values
    y = merged["accuracy"].values
    if len(x) > 2:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        xs = np.linspace(x.min() - 0.3, x.max() + 0.3, 100)
        ax.plot(xs, p(xs), "k--", linewidth=1, alpha=0.4, label="trend")
        ax.legend(fontsize=10)

    plt.tight_layout()
    path = FIGURES_DIR / "fig2_path_vs_accuracy.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 3: Box plot — per-task accuracy spread by topology
# ---------------------------------------------------------------------------

def plot_accuracy_boxplot(metrics: pd.DataFrame) -> None:
    topologies = sorted(metrics["topology"].unique(),
                        key=lambda t: metrics[metrics["topology"]==t]["correct"].mean(),
                        reverse=True)

    data = [metrics[metrics["topology"] == t]["correct"].values.astype(float)
            for t in topologies]
    colors = [TOPOLOGY_COLORS.get(t, "#888") for t in topologies]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops={"color": "white", "linewidth": 2})

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_xticks(range(1, len(topologies) + 1))
    ax.set_xticklabels([t.replace("_", " ") for t in topologies])
    ax.set_ylabel("Correct (1) / Incorrect (0)", fontsize=12)
    ax.set_title("Per-Task Performance Distribution by Topology", fontsize=13, pad=12)
    ax.set_ylim(-0.2, 1.3)

    # Overlay jittered points
    for i, (d, color) in enumerate(zip(data, colors), start=1):
        jitter = np.random.default_rng(42).uniform(-0.1, 0.1, size=len(d))
        ax.scatter(i + jitter, d, color=color, alpha=0.4, s=25, zorder=3)

    plt.tight_layout()
    path = FIGURES_DIR / "fig3_boxplot.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 4: Graph visualizations colored by betweenness centrality
# ---------------------------------------------------------------------------

def plot_graph_visualizations(n: int = 20) -> None:
    topologies = list(TOPOLOGY_BUILDERS.keys())
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for ax, tname in zip(axes, topologies):
        G = get_topology(tname, n)
        bc = nx.betweenness_centrality(G)
        node_colors = [bc[node] for node in G.nodes()]

        if tname == "chain":
            pos = nx.spring_layout(G, seed=42, k=1.2)
        elif tname == "tree":
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot") if nx.nx_agraph else nx.spring_layout(G, seed=42)
        else:
            pos = nx.spring_layout(G, seed=42)

        nx.draw_networkx(
            G, pos=pos, ax=ax,
            node_color=node_colors,
            cmap=plt.cm.YlOrRd,
            node_size=150,
            with_labels=False,
            edge_color="#cccccc",
            width=0.8,
        )
        ax.set_title(tname.replace("_", " ").title(), fontsize=12, pad=8)
        ax.axis("off")

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)
    cbar.set_label("Betweenness Centrality", fontsize=10)

    fig.suptitle("Communication Topologies Colored by Betweenness Centrality",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    path = FIGURES_DIR / "fig4_graph_viz.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 5: Vote agreement heatmap (topology x task)
# ---------------------------------------------------------------------------

def plot_vote_agreement_heatmap(metrics: pd.DataFrame) -> None:
    pivot = metrics.pivot_table(
        index="topology", columns="task_id", values="vote_agreement"
    )

    fig, ax = plt.subplots(figsize=(14, 3.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([t.replace("_", " ") for t in pivot.index])
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"T{c}" for c in pivot.columns], fontsize=8)
    ax.set_xlabel("Task ID", fontsize=11)
    ax.set_title("Vote Agreement by Topology and Task\n(green = consensus, red = split vote)", fontsize=12)

    plt.colorbar(im, ax=ax, label="Fraction of agents voting for majority answer")
    plt.tight_layout()
    path = FIGURES_DIR / "fig5_vote_heatmap.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def make_summary_table(metrics: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    perf = metrics.groupby("topology").agg(
        accuracy=("correct", "mean"),
        n_correct=("correct", "sum"),
        n_tasks=("correct", "count"),
        avg_vote_agreement=("vote_agreement", "mean"),
    ).round(3).reset_index()

    merged = perf.merge(
        stats[["topology", "diameter", "avg_shortest_path",
               "avg_clustering", "max_betweenness", "edge_density"]],
        on="topology",
    )
    merged = merged.sort_values("accuracy", ascending=False)
    merged.to_csv("results/topology_summary.csv", index=False)
    print("\nTopology Summary Table:")
    print(merged.to_string(index=False))
    return merged


def make_failure_table(metrics: pd.DataFrame) -> pd.DataFrame:
    failures = metrics[~metrics["correct"]].copy()
    failures["error"] = failures["correct_answer"] + " → predicted " + failures["majority_answer"].fillna("None")
    failures = failures[["topology", "task_id", "correct_answer", "majority_answer",
                          "vote_agreement", "error"]]
    failures = failures.sort_values(["topology", "task_id"])
    failures.to_csv("results/failure_analysis.csv", index=False)
    print(f"\nFailure analysis: {len(failures)} failures saved to results/failure_analysis.csv")
    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_analysis(skip_graphs: bool = False) -> None:
    print("Loading results...")
    metrics, stats = load_results()

    print(f"  {len(metrics)} runs loaded across {metrics['topology'].nunique()} topologies\n")

    print("Generating figures...")
    plot_accuracy_bar(metrics)
    plot_path_vs_accuracy(metrics, stats)
    plot_accuracy_boxplot(metrics)
    if not skip_graphs:
        plot_graph_visualizations()
    plot_vote_agreement_heatmap(metrics)

    print("\nGenerating tables...")
    make_summary_table(metrics, stats)
    make_failure_table(metrics)

    print("\nDone!! All outputs saved to results/figures/ and results/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-graphs", action="store_true",
                        help="Skip network visualization (faster)")
    args = parser.parse_args()
    run_analysis(skip_graphs=args.no_graphs)