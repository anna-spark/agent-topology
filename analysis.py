"""
analysis.py
-----------
Loads experiment results and produces figures + tables with statistical uncertainty.

Two task families are supported, selected with --task-type (or auto-detected from the
metrics columns):

  * fragment (default): continuous code-recovery score. Reads
    results/metrics_fragment.csv, writes results/figures_fragment/ and
    results/*_fragment.csv so the two experiments never clobber each other.
  * logic_grid: binary exact-match accuracy (companion family). Reads
    results/metrics.csv, writes results/figures/ and results/*.csv.

The headline outcome metric (binary `correct` vs continuous `collective_recovery`),
its axis labels, and the baseline reference lines are all carried in an `Outcome`
object, so every plot/table is written once and reused across task types.
"""

import ast
import argparse
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    "empty":           "#999999",
}

plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.size":          12,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         150,
})


# --------------------------------------------------------------------------- #
# Outcome / context: what metric we are analysing and where outputs go.
# --------------------------------------------------------------------------- #
@dataclass
class Outcome:
    """Describes the headline metric for a task family."""
    task_type: str
    column: str                 # metrics column holding the per-run outcome
    label: str                  # axis label, e.g. "Accuracy" / "Collective Recovery"
    metric_name: str            # short name used in figure/table titles
    round_key: str              # key inside round_results dicts for the per-round curve
    is_continuous: bool         # False -> binary 0/1 (drives box/scatter y-limits)
    # Extra continuous columns to surface in the summary table when present.
    extra_columns: list = field(default_factory=list)
    # Horizontal reference lines: list of (label, value, color).
    baselines: list = field(default_factory=list)


@dataclass
class Context:
    """Resolved paths + outcome for one analysis run."""
    outcome: Outcome
    metrics_path: Path
    figures_dir: Path
    suffix: str                 # "" for logic_grid, "_fragment" for fragment

    def fig(self, name: str) -> Path:
        return self.figures_dir / name

    def csv(self, stem: str) -> Path:
        return Path("results") / f"{stem}{self.suffix}.csv"


def load_logic_grid_baselines() -> list:
    """Single-agent baseline accuracy from results/baseline.csv, if present."""
    path = Path("results/baseline.csv")
    if not path.exists():
        return []
    val = float(pd.read_csv(path)["correct"].mean())
    return [(f"Single-agent baseline ({val:.0%})", val, "#444")]


def load_fragment_baselines() -> list:
    """Floor (one fragment) and ceiling (all fragments) recovery reference lines."""
    out = []
    ceil = Path("results/baseline_fragment_all.csv")
    floor = Path("results/baseline_fragment_one.csv")
    if ceil.exists():
        v = float(pd.read_csv(ceil)["recovery"].mean())
        out.append((f"All-fragments ceiling ({v:.0%})", v, "#2a8d4a"))
    if floor.exists():
        v = float(pd.read_csv(floor)["recovery"].mean())
        out.append((f"Single-fragment floor ({v:.0%})", v, "#b3402f"))
    return out


def make_outcome(task_type: str) -> Outcome:
    if task_type == "fragment":
        return Outcome(
            task_type="fragment",
            column="collective_recovery",
            label="Collective Recovery",
            metric_name="Collective Recovery",
            round_key="collective_recovery",
            is_continuous=True,
            extra_columns=["mean_recovery", "best_recovery"],
            baselines=load_fragment_baselines(),
        )
    return Outcome(
        task_type="logic_grid",
        column="correct",
        label="Accuracy",
        metric_name="Accuracy",
        round_key="correct",
        is_continuous=False,
        extra_columns=[],
        baselines=load_logic_grid_baselines(),
    )


DEFAULT_METRICS = {"logic_grid": "results/metrics.csv", "fragment": "results/metrics_fragment.csv"}
FIGURE_DIRS     = {"logic_grid": "results/figures",  "fragment": "results/figures_fragment"}


def color_for(topo: str) -> str:
    return TOPOLOGY_COLORS.get(topo, "#888")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def regenerate_graph_stats(n: int = N_AGENTS) -> pd.DataFrame:
    """Recompute graph statistics straight from topologies.py so they always match
    the graph code, and persist to results/graph_stats.csv."""
    rows = [compute_graph_stats(get_topology(name, n), name) for name in TOPOLOGY_BUILDERS]
    stats = pd.DataFrame(rows)
    stats.to_csv("results/graph_stats.csv", index=False)
    return stats


def detect_task_type(metrics_path: Path) -> str:
    """Sniff the header so the right metric is chosen even without --task-type."""
    try:
        cols = pd.read_csv(metrics_path, nrows=0).columns
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return "fragment"
    return "fragment" if "collective_recovery" in cols else "logic_grid"


def load_results(ctx: Context) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(ctx.metrics_path)

    # Backfill edge_drop_rate for older rows that predate the column.
    if "edge_drop_rate" not in metrics.columns:
        metrics["edge_drop_rate"] = 0.0
    metrics["edge_drop_rate"] = pd.to_numeric(metrics["edge_drop_rate"], errors="coerce").fillna(0.0)

    # Coerce every numeric column we touch so a stray/duplicated header row or a
    # half-written line can't poison a groupby. Rows with a non-numeric outcome
    # are dropped (the outcome is the one column we cannot do without).
    numeric_cols = [
        ctx.outcome.column, "vote_agreement",
        "collective_recovery", "mean_recovery", "best_recovery",
        "total_tokens", "total_input_tokens", "total_output_tokens",
        "n_messages", "n_llm_calls", "seed", "task_id", "duration_sec",
    ]
    for col in numeric_cols:
        if col in metrics.columns and col != "correct":
            metrics[col] = pd.to_numeric(metrics[col], errors="coerce")

    # `correct` is written as True/False; normalise to a real boolean regardless of
    # whether pandas inferred bool, object, or numeric.
    if "correct" in metrics.columns:
        metrics["correct"] = (
            metrics["correct"].map({True: True, False: False, "True": True, "False": False})
            .fillna(metrics["correct"].apply(lambda v: bool(v) if isinstance(v, (int, float, np.bool_)) and not pd.isna(v) else np.nan))
        )

    before = len(metrics)
    metrics = metrics[metrics[ctx.outcome.column].notna()].copy()
    dropped = before - len(metrics)
    if dropped:
        print(f"  (dropped {dropped} row(s) with missing/invalid '{ctx.outcome.column}')")

    if ctx.outcome.column == "correct":
        metrics["correct"] = metrics["correct"].astype(bool)

    stats = regenerate_graph_stats()
    return metrics, stats


def outcome_values(group: pd.DataFrame, outcome: Outcome) -> np.ndarray:
    return group[outcome.column].astype(float).values


def compute_bootstrap_ci(data: np.ndarray, n_resamples: int = 2000, ci: float = 0.95) -> tuple[float, float]:
    """Compute empirical bootstrap confidence intervals for the mean outcome."""
    data = np.asarray(data, dtype=float)
    if len(data) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(42)
    boot_means = [np.mean(rng.choice(data, size=len(data), replace=True)) for _ in range(n_resamples)]
    lower_pct = (1.0 - ci) / 2.0 * 100
    upper_pct = (1.0 + ci) / 2.0 * 100
    return float(np.percentile(boot_means, lower_pct)), float(np.percentile(boot_means, upper_pct))


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_accuracy_bar(metrics: pd.DataFrame, ctx: Context) -> None:
    outcome = ctx.outcome
    summary_rows = []
    for topo, group in metrics.groupby("topology"):
        vals = outcome_values(group, outcome)
        acc = float(np.mean(vals))
        low_ci, high_ci = compute_bootstrap_ci(vals)
        summary_rows.append({
            "topology":   topo,
            "value":      acc,
            "yerr_lower": acc - low_ci,
            "yerr_upper": high_ci - acc,
        })

    summary = pd.DataFrame(summary_rows).sort_values("value", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    yerr = np.array([summary["yerr_lower"].clip(lower=0).values,
                     summary["yerr_upper"].clip(lower=0).values])

    bars = ax.bar(
        summary["topology"],
        summary["value"],
        yerr=yerr,
        color=[color_for(t) for t in summary["topology"]],
        capsize=5,
        width=0.6,
        error_kw={"elinewidth": 1.5},
    )

    for bar, val in zip(bars, summary["value"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
            f"{val:.0%}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    for label, value, color in outcome.baselines:
        ax.axhline(value, color=color, linestyle="--", linewidth=1.5, label=label)
    if outcome.baselines:
        ax.legend(loc="upper right", fontsize=9)

    ax.set_ylim(0, 1.15)
    ax.set_ylabel(outcome.label)
    ax.set_xlabel("Communication Topology")
    ax.set_title(f"Collective {outcome.metric_name} by Topology (with 95% Bootstrap CIs)", pad=12)
    plt.xticks(rotation=15)

    plt.tight_layout()
    path = ctx.fig("fig1_accuracy_bar.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_path_vs_accuracy(metrics: pd.DataFrame, stats: pd.DataFrame, ctx: Context) -> None:
    outcome = ctx.outcome
    summary = metrics.groupby("topology")[outcome.column].apply(
        lambda s: float(s.astype(float).mean())).reset_index()
    summary.columns = ["topology", "value"]
    merged = summary.merge(stats[["topology", "avg_shortest_path"]], on="topology")
    if merged.empty:
        print("  (no overlap between metrics and graph stats: skipping path-vs-outcome)")
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    for _, row in merged.iterrows():
        color = color_for(row["topology"])
        ax.scatter(row["avg_shortest_path"], row["value"], color=color, s=200, zorder=3)
        ax.annotate(
            row["topology"].replace("_", " "),
            (row["avg_shortest_path"], row["value"]),
            textcoords="offset points", xytext=(8, 5),
            fontsize=9, color=color, fontweight="bold",
        )

    ax.set_xlabel("Average Shortest Path Length")
    ax.set_ylabel(outcome.label)
    ax.set_title(f"Information Reachability vs. Collective {outcome.metric_name}", pad=12)
    ax.set_ylim(-0.05, 1.1)

    x = merged["avg_shortest_path"].values
    y = merged["value"].values
    if len(x) > 2:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        xs = np.linspace(x.min() - 0.2, x.max() + 0.2, 100)
        ax.plot(xs, p(xs), "k--", linewidth=1, alpha=0.4)

    plt.tight_layout()
    path = ctx.fig("fig2_path_vs_accuracy.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_accuracy_boxplot(metrics: pd.DataFrame, ctx: Context) -> None:
    outcome = ctx.outcome
    topologies = sorted(
        metrics["topology"].unique(),
        key=lambda t: outcome_values(metrics[metrics["topology"] == t], outcome).mean(),
        reverse=True,
    )

    data   = [outcome_values(metrics[metrics["topology"] == t], outcome) for t in topologies]
    colors = [color_for(t) for t in topologies]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, patch_artist=True, medianprops={"color": "black", "linewidth": 2})

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(range(1, len(topologies) + 1))
    ax.set_xticklabels([t.replace("_", " ") for t in topologies], rotation=15)
    ylabel = outcome.label if outcome.is_continuous else "Correct (1) / Incorrect (0)"
    ax.set_ylabel(ylabel)
    ax.set_title(f"Per-Task {outcome.metric_name} Distribution by Topology", pad=12)
    ax.set_ylim(-0.05, 1.05) if outcome.is_continuous else ax.set_ylim(-0.2, 1.2)

    jitter_rng = np.random.default_rng(42)
    for i, (d, color) in enumerate(zip(data, colors), start=1):
        jitter = jitter_rng.uniform(-0.1, 0.1, size=len(d))
        ax.scatter(i + jitter, d, color=color, alpha=0.3, s=20, zorder=3)

    plt.tight_layout()
    path = ctx.fig("fig3_boxplot.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_graph_visualizations(ctx: Context, n: int = 20) -> None:
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

    for extra in range(len(topologies), len(axes)):
        axes[extra].axis("off")

    sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd)
    sm.set_array([])
    fig.colorbar(sm, ax=axes.tolist(), shrink=0.5, pad=0.03, label="Betweenness Centrality")
    fig.suptitle("Communication Topologies Colored by Betweenness Centrality", fontsize=14, y=0.98)

    path = ctx.fig("fig4_graph_viz.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_vote_agreement_heatmap(metrics: pd.DataFrame, ctx: Context) -> None:
    if "vote_agreement" not in metrics.columns:
        print("  (no vote_agreement column: skipping agreement heatmap)")
        return
    pivot = metrics.pivot_table(index="topology", columns="task_id", values="vote_agreement")
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([t.replace("_", " ") for t in pivot.index])
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"T{int(c)}" for c in pivot.columns], fontsize=8)
    ax.set_xlabel("Task ID")
    label = ("Mean per-position agreement" if ctx.outcome.task_type == "fragment"
             else "Fraction of agents voting for majority answer")
    ax.set_title("Vote Agreement by Topology and Task")

    plt.colorbar(im, ax=ax, label=label)
    plt.tight_layout()
    path = ctx.fig("fig5_vote_heatmap.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_budget_summary(metrics: pd.DataFrame, ctx: Context) -> pd.DataFrame:
    """Token/message budget by topology (budget-control evidence)."""
    budget_cols = ["total_tokens", "total_input_tokens", "total_output_tokens", "n_messages", "n_llm_calls"]
    available = [c for c in budget_cols if c in metrics.columns]
    if not available:
        print("  (no budget columns found: skipping budget summary)")
        return pd.DataFrame()

    summary = metrics.groupby("topology")[available].mean().round(1).reset_index()
    summary = summary.sort_values("total_tokens" if "total_tokens" in available else available[0],
                                  ascending=False)
    summary.to_csv(ctx.csv("budget_summary"), index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    colors = [color_for(t) for t in summary["topology"]]

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
    path = ctx.fig("fig6_budget.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    print("\nBudget Summary Table:")
    print(summary.to_string(index=False))
    return summary


def plot_robustness(metrics: pd.DataFrame, ctx: Context) -> None:
    """Outcome vs edge-deletion rate per topology (only if drop>0 runs exist)."""
    outcome = ctx.outcome
    if metrics["edge_drop_rate"].max() <= 0:
        print("  (no edge-deletion runs found: skipping robustness plot)")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for topo, group in metrics.groupby("topology"):
        curve = group.groupby("edge_drop_rate")[outcome.column].apply(
            lambda s: float(s.astype(float).mean())).sort_index()
        ax.plot(curve.index, curve.values, marker="o",
                color=color_for(topo), label=topo.replace("_", " "))

    ax.set_xlabel("Edge-deletion rate")
    ax.set_ylabel(outcome.label)
    ax.set_title("Robustness to Edge Deletion by Topology", pad=12)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    path = ctx.fig("fig7_robustness.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_accuracy_by_round(metrics: pd.DataFrame, ctx: Context) -> None:
    """Outcome after each communication round, per topology (propagation speed)."""
    outcome = ctx.outcome
    rows = []
    for _, row in metrics.iterrows():
        rr = row.get("round_results")
        if isinstance(rr, str):
            try:
                rr = ast.literal_eval(rr)
            except (ValueError, SyntaxError):
                continue
        if not isinstance(rr, list):
            continue
        for entry in rr:
            if not isinstance(entry, dict) or "round" not in entry:
                continue
            val = entry.get(outcome.round_key)
            if val is None:
                continue
            rows.append({"topology": row["topology"], "round": entry["round"],
                         "value": float(val)})

    if not rows:
        print(f"  (no per-round '{outcome.round_key}' found: skipping {outcome.metric_name.lower()}-by-round)")
        return

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    for topo, g in df.groupby("topology"):
        curve = g.groupby("round")["value"].mean().sort_index()
        ax.plot(curve.index, curve.values, marker="o",
                color=color_for(topo), label=topo.replace("_", " "))

    ax.set_xlabel("Communication round")
    ax.set_ylabel(outcome.label)
    ax.set_title(f"{outcome.metric_name} vs. Communication Round by Topology", pad=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(sorted(df["round"].unique()))
    ax.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    path = ctx.fig("fig8_rounds.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def make_summary_table(metrics: pd.DataFrame, stats: pd.DataFrame, ctx: Context) -> pd.DataFrame:
    outcome = ctx.outcome
    perf_rows = []
    for topo, group in metrics.groupby("topology"):
        vals = outcome_values(group, outcome)
        acc = float(np.mean(vals))
        low_ci, high_ci = compute_bootstrap_ci(vals)
        row = {
            "topology":          topo,
            outcome.column:      round(acc, 3),
            "95% CI Lower":      round(low_ci, 3),
            "95% CI Upper":      round(high_ci, 3),
            "n_runs":            int(len(group)),
        }
        if "vote_agreement" in group:
            row["avg_vote_agreement"] = round(float(group["vote_agreement"].mean()), 3)
        for extra in outcome.extra_columns:
            if extra in group:
                row[f"avg_{extra}"] = round(float(group[extra].astype(float).mean()), 3)
        # For fragment, exact-match accuracy is a useful secondary number to report.
        if outcome.column != "correct" and "correct" in group:
            row["exact_match_acc"] = round(float(group["correct"].astype(float).mean()), 3)
        if "seed" in group:
            row["n_seeds"] = int(group["seed"].nunique())
        if "total_tokens" in group:
            row["avg_total_tokens"] = round(float(group["total_tokens"].mean()), 1)
        if "n_messages" in group:
            row["avg_n_messages"] = round(float(group["n_messages"].mean()), 1)
        perf_rows.append(row)
    perf   = pd.DataFrame(perf_rows)
    merged = perf.merge(
        stats[["topology", "diameter", "avg_shortest_path", "max_betweenness", "edge_density"]],
        on="topology", how="left",
    )
    merged = merged.sort_values(outcome.column, ascending=False)
    merged.to_csv(ctx.csv("topology_summary"), index=False)
    print("\nTopology Summary Table:")
    print(merged.to_string(index=False))
    return merged


def make_failure_table(metrics: pd.DataFrame, ctx: Context) -> pd.DataFrame:
    outcome = ctx.outcome
    cols = ["topology", "task_id", "correct_answer", "majority_answer", "vote_agreement"]

    if outcome.is_continuous:
        # No exact-match notion of "failure"; surface incompletely-reconstructed runs,
        # worst first, with the recovery columns attached.
        rec_cols = [c for c in [outcome.column, "mean_recovery", "best_recovery"] if c in metrics.columns]
        failures = metrics[metrics[outcome.column] < 1.0].copy()
        keep = [c for c in cols if c in failures.columns] + rec_cols
        failures = failures[keep].sort_values([outcome.column, "topology", "task_id"]) if not failures.empty else failures
    else:
        failures = metrics[~metrics["correct"]].copy()
        keep = [c for c in cols if c in failures.columns]
        failures = failures[keep].sort_values(["topology", "task_id"]) if not failures.empty else failures

    failures.to_csv(ctx.csv("failure_analysis"), index=False)
    print(f"\nFailure cases written: {len(failures)} row(s) -> {ctx.csv('failure_analysis')}")
    return failures


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _safe(label: str, fn, *args) -> None:
    """Run one analysis step; report and continue if it raises, so a single bad
    plot never sinks the whole run."""
    try:
        fn(*args)
    except Exception as exc:  # noqa: BLE001 - intentional: isolate per-step failures
        print(f"  !! {label} failed: {type(exc).__name__}: {exc}")


def run_analysis(ctx: Context, skip_graphs: bool = False) -> None:
    print(f"Task type: {ctx.outcome.task_type}  |  metric: {ctx.outcome.column}  |  source: {ctx.metrics_path}")
    metrics, stats = load_results(ctx)
    if metrics.empty:
        print("No usable rows in metrics file, nothing to analyse.")
        return

    if ctx.outcome.baselines:
        print("Baselines: " + ", ".join(f"{lbl}" for lbl, _, _ in ctx.outcome.baselines))

    # Main outcome/structure figures use only the intact-graph runs (drop_rate == 0).
    main = metrics[metrics["edge_drop_rate"] == 0].copy()
    if main.empty:
        print("  (all runs have edge_drop_rate > 0; using full set for main figures)")
        main = metrics

    _safe("accuracy bar",      plot_accuracy_bar,          main, ctx)
    _safe("path vs outcome",   plot_path_vs_accuracy,      main, stats, ctx)
    _safe("boxplot",           plot_accuracy_boxplot,      main, ctx)
    if not skip_graphs:
        _safe("graph viz",     plot_graph_visualizations,  ctx)
    _safe("vote heatmap",      plot_vote_agreement_heatmap, main, ctx)
    _safe("summary table",     make_summary_table,         main, stats, ctx)
    _safe("failure table",     make_failure_table,         main, ctx)
    _safe("budget summary",    plot_budget_summary,        main, ctx)
    _safe("outcome by round",  plot_accuracy_by_round,     main, ctx)
    _safe("robustness",        plot_robustness,            metrics, ctx)  # uses all runs
    print("\nAnalysis complete!")


def build_context(task_type: str | None, metrics_arg: str | None) -> Context:
    metrics_path = Path(metrics_arg) if metrics_arg else None

    # Resolve task type: explicit flag wins, else sniff the file, else default.
    if task_type is None:
        sniff_path = metrics_path or Path(DEFAULT_METRICS["fragment"])
        task_type = detect_task_type(sniff_path)

    if metrics_path is None:
        metrics_path = Path(DEFAULT_METRICS[task_type])

    outcome = make_outcome(task_type)
    suffix = "_fragment" if task_type == "fragment" else ""
    figures_dir = Path(FIGURE_DIRS[task_type])
    figures_dir.mkdir(parents=True, exist_ok=True)
    return Context(outcome=outcome, metrics_path=metrics_path, figures_dir=figures_dir, suffix=suffix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate figures + tables from an experiment sweep.")
    parser.add_argument("--task-type", choices=["logic_grid", "fragment"], default=None,
                        help="Which experiment to analyse. Default: auto-detect from the metrics file.")
    parser.add_argument("--metrics", default=None,
                        help="Override the metrics CSV path (default depends on --task-type).")
    parser.add_argument("--no-graphs", action="store_true",
                        help="Skip the slow networkx topology visualizations.")
    args = parser.parse_args()

    ctx = build_context(args.task_type, args.metrics)
    run_analysis(ctx, skip_graphs=args.no_graphs)
