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

N_AGENTS = 20  # must match run_experiments.DEFAULT_N_AGENTS

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


def regenerate_graph_stats(n: int = N_AGENTS) -> pd.DataFrame:
    """Recompute graph statistics straight from topologies.py so they always match
    the graph code, and persist to results/graph_stats.csv."""
    rows = [compute_graph_stats(get_topology(name, n), name) for name in TOPOLOGY_BUILDERS]
    stats = pd.DataFrame(rows)
    stats.to_csv("results/graph_stats.csv", index=False)
    return stats


def load_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv("results/metrics.csv")
    # Backfill edge_drop_rate for older rows that predate the column.
    if "edge_drop_rate" not in metrics.columns:
        metrics["edge_drop_rate"] = 0.0
    metrics["edge_drop_rate"] = metrics["edge_drop_rate"].fillna(0.0)
    stats = regenerate_graph_stats()
    return metrics, stats


def load_baseline_accuracy() -> float | None:
    """Single-agent baseline accuracy, if results/baseline.csv exists."""
    path = Path("results/baseline.csv")
    if not path.exists():
        return None
    return float(pd.read_csv(path)["correct"].mean())


def compute_bootstrap_ci(data: np.ndarray, n_resamples: int = 2000, ci: float = 0.95) -> tuple[float, float]:
    """Compute empirical bootstrap confidence intervals for accuracy."""
    if len(data) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(42)
    boot_means = [np.mean(rng.choice(data, size=len(data), replace=True)) for _ in range(n_resamples)]
    lower_pct = (1.0 - ci) / 2.0 * 100
    upper_pct = (1.0 + ci) / 2.0 * 100
    return float(np.percentile(boot_means, lower_pct)), float(np.percentile(boot_means, upper_pct))


def plot_accuracy_bar(metrics: pd.DataFrame, baseline: float | None = None) -> None:
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

    if baseline is not None:
        ax.axhline(baseline, color="#444", linestyle="--", linewidth=1.5,
                   label=f"Single-agent baseline ({baseline:.0%})")
        ax.legend(loc="upper right", fontsize=9)

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


def plot_budget_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Token/message budget by topology (Step 3 deliverable + budget-control evidence)."""
    budget_cols = ["total_tokens", "total_input_tokens", "total_output_tokens", "n_messages", "n_llm_calls"]
    available = [c for c in budget_cols if c in metrics.columns]
    if not available:
        print("  (no budget columns found — skipping budget summary)")
        return pd.DataFrame()

    summary = metrics.groupby("topology")[available].mean().round(1).reset_index()
    summary = summary.sort_values("total_tokens" if "total_tokens" in available else available[0],
                                  ascending=False)
    summary.to_csv("results/budget_summary.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    colors = [TOPOLOGY_COLORS.get(t, "#888") for t in summary["topology"]]

    if "total_tokens" in summary:
        ax1.bar(summary["topology"], summary["total_tokens"], color=colors, width=0.6)
        ax1.set_ylabel("Avg total tokens per run")
        ax1.set_title("Token Budget by Topology")
        ax1.tick_params(axis="x", rotation=20)

    msg_col = "n_llm_calls" if "n_llm_calls" in summary else ("n_messages" if "n_messages" in summary else None)
    if msg_col:
        ax2.bar(summary["topology"], summary[msg_col], color=colors, width=0.6)
        ax2.set_ylabel(f"Avg {msg_col} per run")
        ax2.set_title("Compute Budget by Topology (should be ~constant)")
        ax2.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    path = FIGURES_DIR / "fig6_budget.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    print("\nBudget Summary Table:")
    print(summary.to_string(index=False))
    return summary


def plot_robustness(metrics: pd.DataFrame) -> None:
    """Accuracy vs edge-deletion rate per topology (only if drop>0 runs exist)."""
    if metrics["edge_drop_rate"].max() <= 0:
        print("  (no edge-deletion runs found — skipping robustness plot)")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for topo, group in metrics.groupby("topology"):
        curve = group.groupby("edge_drop_rate")["correct"].mean().sort_index()
        ax.plot(curve.index, curve.values, marker="o",
                color=TOPOLOGY_COLORS.get(topo, "#888"), label=topo.replace("_", " "))

    ax.set_xlabel("Edge-deletion rate")
    ax.set_ylabel("Accuracy")
    ax.set_title("Robustness to Edge Deletion by Topology", pad=12)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    path = FIGURES_DIR / "fig7_robustness.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def make_summary_table(metrics: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    perf_rows = []
    for topo, group in metrics.groupby("topology"):
        acc = group["correct"].mean()
        low_ci, high_ci = compute_bootstrap_ci(group["correct"].values)
        row = {
            "topology":          topo,
            "accuracy":          round(acc, 3),
            "95% CI Lower":      round(low_ci, 3),
            "95% CI Upper":      round(high_ci, 3),
            "avg_vote_agreement":round(group["vote_agreement"].mean(), 3),
        }
        if "total_tokens" in group:
            row["avg_total_tokens"] = round(group["total_tokens"].mean(), 1)
        if "n_messages" in group:
            row["avg_n_messages"] = round(group["n_messages"].mean(), 1)
        perf_rows.append(row)
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
    baseline = load_baseline_accuracy()

    # Main accuracy/structure figures use only the intact-graph runs (drop_rate == 0).
    main = metrics[metrics["edge_drop_rate"] == 0].copy()

    plot_accuracy_bar(main, baseline=baseline)
    plot_path_vs_accuracy(main, stats)
    plot_accuracy_boxplot(main)
    if not skip_graphs:
        plot_graph_visualizations()
    plot_vote_agreement_heatmap(main)
    make_summary_table(main, stats)
    make_failure_table(main)
    plot_budget_summary(main)
    plot_robustness(metrics)  # uses all runs, including edge-deletion
    print("\nAnalysis complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-graphs", action="store_true")
    args = parser.parse_args()
    run_analysis(skip_graphs=args.no_graphs)
