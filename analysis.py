"""
analysis.py
-----------
Loads experiment results and produces figures + tables with statistical uncertainty.
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

TOPOLOGY_COLORS = {
    "chain":           "#4C72B0",
    "tree":            "#DD8452",
    "random":          "#55A868",
    "small_world":     "#C44E52",
    "modular":         "#8172B3",
    "scale_free":      "#CCB974",
    "fully_connected": "#64B5CD",
}

plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.size":          12,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         150,
})

FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv("results/metrics.csv")
    stats   = pd.read_csv("results/graph_stats.csv")
    return metrics, stats


def compute_bootstrap_ci(data: np.ndarray, n_resamples: int = 2000, ci: float = 0.95) -> tuple[float, float]:
    """Compute empirical bootstrap confidence intervals for accuracy."""
    if len(data) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(42)
    boot_means = [np.mean(rng.choice(data, size=len(data), replace=True)) for _ in range(n_resamples)]
    lower_pct = (1.0 - ci) / 2.0 * 100
    upper_pct = (1.0 + ci) / 2.0 * 100
    return float(np.percentile(boot_means, lower_pct)), float(np.percentile(boot_means, upper_pct))


def plot_accuracy_bar(metrics: pd.DataFrame) -> None:
    summary_rows = []
    for topo, group in metrics.groupby("topology"):
        acc = group["correct"].mean()
        low_ci, high_ci = compute_bootstrap_ci(group["correct"].values)
        summary_rows.append({
            "topology":   topo,
            "accuracy":   acc,
            "ci_lower":   low_ci,
            "ci_upper":   high_ci,
            "yerr_lower": acc - low_ci,
            "yerr_upper": high_ci - acc,
        })

    summary = pd.DataFrame(summary_rows).sort_values("accuracy", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    yerr = np.array([summary["yerr_lower"].values, summary["yerr_upper"].values])

    bars = ax.bar(
        summary["topology"],
        summary["accuracy"],
        yerr=yerr,
        color=[TOPOLOGY_COLORS.get(t, "#888") for t in summary["topology"]],
        capsize=5,
        width=0.6,
        error_kw={"elinewidth": 1.5},
    )

    for bar, acc in zip(bars, summary["accuracy"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
            f"{acc:.0%}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Communication Topology")
    ax.set_title("Collective Accuracy by Topology (with 95% Bootstrap CIs)", pad=12)
    plt.xticks(rotation=15)

    plt.tight_layout()
    path = FIGURES_DIR / "fig1_accuracy_bar.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_path_vs_accuracy(metrics: pd.DataFrame, stats: pd.DataFrame) -> None:
    summary = metrics.groupby("topology")["correct"].mean().reset_index()
    summary.columns = ["topology", "accuracy"]
    merged = summary.merge(stats[["topology", "avg_shortest_path"]], on="topology")

    fig, ax = plt.subplots(figsize=(7, 5))

    for _, row in merged.iterrows():
        color = TOPOLOGY_COLORS.get(row["topology"], "#888")
        ax.scatter(row["avg_shortest_path"], row["accuracy"], color=color, s=200, zorder=3)
        ax.annotate(
            row["topology"].replace("_", " "),
            (row["avg_shortest_path"], row["accuracy"]),
            textcoords="offset points", xytext=(8, 5),
            fontsize=9, color=color, fontweight="bold",
        )

    ax.set_xlabel("Average Shortest Path Length")
    ax.set_ylabel("Accuracy")
    ax.set_title("Information Reachability vs. Collective Accuracy", pad=12)
    ax.set_ylim(-0.05, 1.1)

    x = merged["avg_shortest_path"].values
    y = merged["accuracy"].values
    if len(x) > 2:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        xs = np.linspace(x.min() - 0.2, x.max() + 0.2, 100)
        ax.plot(xs, p(xs), "k--", linewidth=1, alpha=0.4)

    plt.tight_layout()
    path = FIGURES_DIR / "fig2_path_vs_accuracy.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_accuracy_boxplot(metrics: pd.DataFrame) -> None:
    topologies = sorted(
        metrics["topology"].unique(),
        key=lambda t: metrics[metrics["topology"] == t]["correct"].mean(),
        reverse=True,
    )

    data   = [metrics[metrics["topology"] == t]["correct"].values.astype(float) for t in topologies]
    colors = [TOPOLOGY_COLORS.get(t, "#888") for t in topologies]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, patch_artist=True, medianprops={"color": "white", "linewidth": 2})

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(range(1, len(topologies) + 1))
    ax.set_xticklabels([t.replace("_", " ") for t in topologies], rotation=15)
    ax.set_ylabel("Correct (1) / Incorrect (0)")
    ax.set_title("Per-Task Performance Distribution by Topology", pad=12)
    ax.set_ylim(-0.2, 1.2)

    for i, (d, color) in enumerate(zip(data, colors), start=1):
        jitter = np.random.default_rng(42).uniform(-0.1, 0.1, size=len(d))
        ax.scatter(i + jitter, d, color=color, alpha=0.3, s=20, zorder=3)

    plt.tight_layout()
    path = FIGURES_DIR / "fig3_boxplot.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_graph_visualizations(n: int = 20) -> None:
    topologies = list(TOPOLOGY_BUILDERS.keys())
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    for idx, tname in enumerate(topologies):
        ax = axes[idx]
        G = get_topology(tname, n)
        bc = nx.betweenness_centrality(G)
        node_colors = [bc[node] for node in G.nodes()]

        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx(
            G, pos=pos, ax=ax,
            node_color=node_colors,
            cmap=plt.cm.YlOrRd,
            node_size=120,
            with_labels=False,
            edge_color="#dddddd",
            width=0.7,
        )
        ax.set_title(tname.replace("_", " ").title(), fontsize=11)
        ax.axis("off")

    if len(topologies) < len(axes):
        axes[-1].axis("off")

    sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd)
    sm.set_array([])
    fig.colorbar(sm, ax=axes.tolist(), shrink=0.5, pad=0.03, label="Betweenness Centrality")
    fig.suptitle("Communication Topologies Colored by Betweenness Centrality", fontsize=14, y=0.98)

    path = FIGURES_DIR / "fig4_graph_viz.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_vote_agreement_heatmap(metrics: pd.DataFrame) -> None:
    pivot = metrics.pivot_table(index="topology", columns="task_id", values="vote_agreement")
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([t.replace("_", " ") for t in pivot.index])
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"T{c}" for c in pivot.columns], fontsize=8)
    ax.set_xlabel("Task ID")
    ax.set_title("Vote Agreement by Topology and Task")

    plt.colorbar(im, ax=ax, label="Fraction of agents voting for majority answer")
    plt.tight_layout()
    path = FIGURES_DIR / "fig5_vote_heatmap.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def make_summary_table(metrics: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    perf_rows = []
    for topo, group in metrics.groupby("topology"):
        acc = group["correct"].mean()
        low_ci, high_ci = compute_bootstrap_ci(group["correct"].values)
        perf_rows.append({
            "topology":          topo,
            "accuracy":          round(acc, 3),
            "95% CI Lower":      round(low_ci, 3),
            "95% CI Upper":      round(high_ci, 3),
            "avg_vote_agreement":round(group["vote_agreement"].mean(), 3),
        })
    perf   = pd.DataFrame(perf_rows)
    merged = perf.merge(
        stats[["topology", "diameter", "avg_shortest_path", "max_betweenness", "edge_density"]],
        on="topology",
    )
    merged = merged.sort_values("accuracy", ascending=False)
    merged.to_csv("results/topology_summary.csv", index=False)
    print("\nTopology Summary Table:")
    print(merged.to_string(index=False))
    return merged


def make_failure_table(metrics: pd.DataFrame) -> pd.DataFrame:
    failures = metrics[~metrics["correct"]].copy()
    if not failures.empty:
        failures = failures[["topology", "task_id", "correct_answer", "majority_answer", "vote_agreement"]]
        failures = failures.sort_values(["topology", "task_id"])
    failures.to_csv("results/failure_analysis.csv", index=False)
    return failures


def run_analysis(skip_graphs: bool = False) -> None:
    metrics, stats = load_results()
    plot_accuracy_bar(metrics)
    plot_path_vs_accuracy(metrics, stats)
    plot_accuracy_boxplot(metrics)
    if not skip_graphs:
        plot_graph_visualizations()
    plot_vote_agreement_heatmap(metrics)
    make_summary_table(metrics, stats)
    make_failure_table(metrics)
    print("\nAnalysis complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-graphs", action="store_true")
    args = parser.parse_args()
    run_analysis(skip_graphs=args.no_graphs)
