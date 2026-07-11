#!/usr/bin/env bash
#
# Fragment robustness sweep (edge deletion).
#
# Produces the recovery-vs-drop-rate degradation curves per topology. We scope to four
# topologies spanning the density range and ~15 tasks — not because of rate limits
# (paid Tier 1 has plenty of RPM headroom), but because the simulator issues LLM calls
# sequentially, so wall-clock is ~(#runs × ~120 calls × call latency). The four chosen
# topologies already tell the whole degradation story.
#
# Topology rationale (edges out of 190 possible at N=20):
#   fully_connected (190) — needs heavy deletion before it bends; the high-end of the curve
#   random          (~67) — bends in the mid range
#   modular         (~44) — dropping a community bridge can split it; most interesting
#   chain           ( 19) — minimally connected; collapses at ANY positive rate (sparse reference)
#
# Read the results on collective_recovery AND mean_recovery / convergence speed:
# collective_recovery is a threshold metric (sits at 1.0, then cliffs); the graceful
# degradation that separates topologies shows up earlier in mean recovery.
#
# Resumable: every completed (task,topology,seed,drop) is skipped on re-run, so if this
# dies partway just run it again — it picks up where it left off.
#
# To scale up/down, edit the four variables below. Want full coverage now that limits
# aren't the constraint? Set TOPOS to all seven and TASKS to 0..24 — just expect more hours.

set -u
cd "$(dirname "$0")"
PY=./venv/bin/python

TOPOS="fully_connected random modular chain"   # add 'tree' for a second sparse reference
SEEDS="42 123 2024"                             # >=3: which edges drop is random, so average it
TASKS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14"      # 15 tasks
RATES="0.0 0.2 0.4 0.6 0.8"                      # 0.0 = curve anchor (re-uses existing seed-42 rows)

echo "Robustness sweep: topologies=[$TOPOS] seeds=[$SEEDS] rates=[$RATES] tasks=$(echo $TASKS | wc -w | tr -d ' ')"

# Single invocation — the harness loops over rates/topologies/seeds/tasks internally and
# is resumable. On Tier 1 there's no need to pace it; the hardened retry loop in agents.py
# absorbs the occasional transient empty-response/429.
$PY run_experiments.py --task-type fragment \
    --topologies $TOPOS --seeds $SEEDS --tasks $TASKS \
    --drop-rates $RATES

# Backfill pass: re-run once to pick up any runs left for resume after a transient error.
# Cheap — completed runs are skipped.
echo ""
echo "=== backfill pass (retries any failed runs) ==="
$PY run_experiments.py --task-type fragment \
    --topologies $TOPOS --seeds $SEEDS --tasks $TASKS \
    --drop-rates $RATES

# Regenerate the fragment figures + tables, including fig7_robustness.png (recovery vs
# drop rate). analysis.py only draws the robustness curve once drop>0 rows exist.
echo ""
echo "=== regenerating fragment figures + tables ==="
$PY analysis.py --task-type fragment

echo ""
echo "Done. Check results/figures_fragment/fig7_robustness.png and results/topology_summary_fragment.csv"
