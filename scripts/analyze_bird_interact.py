#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-run analysis for overnight BIRD-Interact results.

Joins run JSONL with the source dataset (for difficulty, ambiguity types, etc.)
and prints a structured report. No LLM calls.

Usage:
    python scripts/analyze_bird_interact.py results/overnight_run_1.jsonl
    python scripts/analyze_bird_interact.py results/overnight_run_1.jsonl \
        --data /path/to/bird_interact_data_with_gt.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


class _Tee:
    """Write to both the real stdout and a file simultaneously."""
    def __init__(self, file_path: Path) -> None:
        self._real = sys.stdout
        self._f = open(file_path, "w")

    def write(self, data: str) -> int:
        self._real.write(data)
        self._f.write(data)
        return len(data)

    def flush(self) -> None:
        self._real.flush()
        self._f.flush()

    def close(self) -> None:
        sys.stdout = self._real
        self._f.close()

ONTOLOGY_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ONTOLOGY_DIR / "datasets" / "bird_interact" / "bird_interact_data_with_gt.jsonl"

# Non-critical ambiguity types to exclude from the unresolved-vs-ambiguity analysis
# (sort direction, decimal precision, null handling are noise — not real failures)
_DIFFICULTY_NORM = {
    "easy":       "Simple",
    "simple":     "Simple",
    "medium":     "Moderate",
    "moderate":   "Moderate",
    "hard":       "Challenging",
    "challenging": "Challenging",
}

def _norm_diff(raw: str | None) -> str:
    return _DIFFICULTY_NORM.get((raw or "").lower(), "Unknown")


def _diff_label(task: dict | None, dataset_label: str) -> str:
    """Difficulty/complexity grouping label for a task.

    The full dataset has no difficulty_tier field (Simple/Moderate/Challenging
    doesn't exist there) — it instead carries a boolean 'high_level' flag, so
    for full we group by High Level / Normal. Lite (and anything else that
    carries difficulty_tier) keeps the Simple/Moderate/Challenging tiers.
    """
    task = task or {}
    if dataset_label == "full":
        if "high_level" not in task:
            return "Unknown"
        return "High Level" if task.get("high_level") else "Normal"
    return _norm_diff(task.get("difficulty_tier"))


def _diff_order(dataset_label: str) -> list[str]:
    return ["High Level", "Normal", "Unknown"] if dataset_label == "full" \
        else ["Simple", "Moderate", "Challenging", "Unknown"]


_NOISE_AMBIGUITY_TYPES = frozenset({
    "sort_ambiguity",
    "decimal_ambiguity",
    "null_ambiguity",
    "order_ambiguity",
})


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def _infer_dataset_label(records: list[dict], data_path: Path) -> str:
    """Prefer the 'dataset' field written into run records; fall back to the
    --data path for older result files that predate that field."""
    labels = {r["dataset"] for r in records if r.get("dataset")}
    if len(labels) == 1:
        return labels.pop()
    if len(labels) > 1:
        print(f"Warning: mixed datasets in results ({sorted(labels)}) — treating as 'mixed'",
              file=sys.stderr)
        return "mixed"
    return "full" if "full" in str(data_path).lower() else "lite"


def _pct(num: int, denom: int) -> str:
    if denom == 0:
        return "N/A"
    return f"{num / denom * 100:.1f}%"


def _pct2(num: int, denom: int) -> str:
    if denom == 0:
        return "N/A"
    return f"{num / denom * 100:.2f}%"


def _fmt_t(secs: float | None) -> str:
    if secs is None:
        return "N/A"
    return f"{secs:.1f}s"


def _fmt_t_avg(vals: list[float]) -> str:
    if not vals:
        return "N/A"
    return f"{sum(vals) / len(vals):.1f}s"


def _abbrev_dbs(dbs: list[str]) -> dict[str, str]:
    """Map full DB names to short forms (e.g. archeology_scan -> AS) for chart labels."""
    used: set[str] = set()
    short: dict[str, str] = {}
    for db in dbs:
        parts = db.split("_")
        candidate = "".join(p[0] for p in parts).upper() if len(parts) > 1 else db[:2].upper()
        n = len(candidate) + 1
        while candidate in used:
            candidate = db[:n].upper()
            n += 1
        used.add(candidate)
        short[db] = candidate
    return short


def _make_charts(
    ok: list[dict],
    timestamped: list[dict],
    dataset: dict[str, dict],
    score_buckets: dict,
    followup_stats: dict,
    out_path: Path,
    dataset_label: str = "lite",
) -> None:
    """Generate two chart PNGs: results (3 charts) and diagnostics (4 charts)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (charts skipped — matplotlib not installed: pip install matplotlib)")
        return

    if len(ok) < 1:
        print("  (charts skipped — no valid tasks)")
        return

    # ── Shared colour palette ─────────────────────────────────────────────────
    # Pass/outcome scale  (green → yellow → red)
    C_PASS_BOTH  = "#4caf50"   # green   — passed both phases
    C_PASS_P1    = "#ffc107"   # amber   — passed phase 1 only
    C_FAIL       = "#f44336"   # red     — failed
    # Score scale  (0.0 → 1.0 : red → orange → yellow → lime → yellow-green → green)
    C_SCORE = {0.0: "#f44336", 0.5: "#ff9800", 0.7: "#ffeb3b", 0.8: "#cddc39", 0.9: "#8bc34a", 1.0: "#4caf50"}
    # Difficulty scale  (Simple pink → Moderate purple → Challenging dark-blue)
    C_DIFF = {"Simple": "#f06292", "Moderate": "#9c27b0", "Challenging": "#1565c0", "Unknown": "#9e9e9e", "All": "#9e9e9e",
              "Normal": "#f06292", "High Level": "#1565c0"}
    # Timing components  (clarification dark-grey / SQL gen mid-grey / debug steel-blue)
    C_CLARIFY = "#424242"
    C_SQLGEN  = "#9e9e9e"
    C_DEBUG   = "#1976d2"

    # Ambiguity type palette — teal/cyan family, distinct from all existing palettes
    C_AMB = ["#00897b", "#26a69a", "#4db6ac", "#80cbc4", "#b2dfdb", "#e0f2f1"]

    # ══════════════════════════════════════════════════════════════════════════
    # PNG 1 — Results  (2×2: outcome per DB, score distribution, follow-up, ambiguity types)
    # ══════════════════════════════════════════════════════════════════════════
    path1 = out_path
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
    fig1.suptitle(f"BIRD-Interact Run Analysis — Results [{dataset_label.upper()}]", fontsize=13, fontweight="bold")
    fig1.subplots_adjust(hspace=0.45, wspace=0.38, left=0.08, right=0.97, top=0.92, bottom=0.1)

    # Chart A: Outcome per DB
    ax = axes1[0, 0]
    db_stats: dict[str, dict[str, int]] = {}
    for r in ok:
        db = r["database"]
        if db not in db_stats:
            db_stats[db] = {"both": 0, "p1_only": 0, "failed": 0}
        p1 = r.get("phase1_passed", False)
        p2 = r.get("phase2_passed")
        if p1 and (p2 is None or p2):
            db_stats[db]["both"] += 1
        elif p1:
            db_stats[db]["p1_only"] += 1
        else:
            db_stats[db]["failed"] += 1
    dbs = sorted(db_stats)
    db_short = _abbrev_dbs(dbs)
    both    = [db_stats[d]["both"]    for d in dbs]
    p1_only = [db_stats[d]["p1_only"] for d in dbs]
    failed  = [db_stats[d]["failed"]  for d in dbs]
    xi = range(len(dbs))
    ax.bar(xi, both,    label="Pass (both phases)", color=C_PASS_BOTH)
    ax.bar(xi, p1_only, bottom=both, label="Pass (P1 only)", color=C_PASS_P1)
    ax.bar(xi, failed,  bottom=[a+b for a,b in zip(both, p1_only)], label="Fail", color=C_FAIL)
    ax.set_xticks(list(xi))
    ax.set_xticklabels([db_short[d] for d in dbs], rotation=45, ha="right", fontsize=7)
    ax.set_title("Outcome per DB")
    ax.set_ylabel("Tasks")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
    ax.legend(fontsize=7)

    # Chart B: Score distribution
    ax = axes1[0, 1]
    score_keys   = [0.0, 0.5, 0.7, 0.8, 0.9, 1.0]
    score_labels = ["0.0", "0.5", "0.7", "0.8", "0.9", "1.0"]
    score_counts = [score_buckets.get(k, 0) for k in score_keys]
    bars = ax.bar(score_labels, score_counts, color=[C_SCORE[k] for k in score_keys], edgecolor="white")
    for bar, cnt in zip(bars, score_counts):
        if cnt > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    str(cnt), ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Score")
    ax.set_ylabel("Tasks")
    ax.set_ylim(0, max(score_counts or [1]) * 1.2)
    ax.set_title("Score Distribution")

    # Chart C: Follow-up type P2 pass rate (overall, no difficulty stacking)
    ax = axes1[1, 0]
    if followup_stats:
        fu_types = sorted(followup_stats)
        rates = []
        counts = []
        for ft in fu_types:
            all_vals = [v for vals in followup_stats[ft].values() for v in vals]
            rates.append(sum(all_vals) / len(all_vals) * 100 if all_vals else 0)
            counts.append(len(all_vals))
        bar_colors_fu = C_AMB[:len(fu_types)] if len(fu_types) <= len(C_AMB) else C_AMB * 2
        bars = ax.bar(fu_types, rates, color=bar_colors_fu[:len(fu_types)], edgecolor="white")
        for bar, rate, n in zip(bars, rates, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{rate:.0f}%\n(n={n})", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("P2 pass rate (%)")
        ax.set_ylim(0, 110)
        ax.set_title("P2 Pass Rate by Follow-up Type")
        ax.set_xticks(list(range(len(fu_types))))
        ax.set_xticklabels(fu_types, rotation=20, ha="right", fontsize=8)
    else:
        ax.text(0.5, 0.5, "no follow-up data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("P2 Pass Rate by Follow-up Type")

    # Chart D: Critical ambiguity type × P1 pass rate
    ax = axes1[1, 1]
    if dataset:
        KNOWLEDGE_LINKING = "knowledge_linking_ambiguity"
        amb_type_stats: list[tuple[str, int, float]] = []  # (label, n, pass_rate)
        all_types_chart: set[str] = set()
        for row in dataset.values():
            for a in row.get("user_query_ambiguity", {}).get("critical_ambiguity", []):
                t = a.get("type", "")
                if t and t not in _NOISE_AMBIGUITY_TYPES:
                    all_types_chart.add(t)
        for target_type in sorted(all_types_chart):
            allowed = {target_type, KNOWLEDGE_LINKING}
            group = []
            for r in ok:
                task = dataset.get(r["instance_id"]) or {}
                crit = task.get("user_query_ambiguity", {}).get("critical_ambiguity", [])
                types_in_task = {
                    a.get("type", "") for a in crit
                    if a.get("type", "") and a.get("type", "") not in _NOISE_AMBIGUITY_TYPES
                }
                if not types_in_task:
                    continue
                if target_type == KNOWLEDGE_LINKING:
                    if types_in_task == {KNOWLEDGE_LINKING}:
                        group.append(r)
                else:
                    if types_in_task <= allowed and target_type in types_in_task:
                        group.append(r)
            if group:
                rate = sum(1 for r in group if r.get("phase1_passed")) / len(group) * 100
                # Shorten label: remove "_ambiguity" suffix
                label = target_type.replace("_ambiguity", "").replace("_", " ")
                amb_type_stats.append((label, len(group), rate))
        if amb_type_stats:
            labels_a = [s[0] for s in amb_type_stats]
            ns_a     = [s[1] for s in amb_type_stats]
            rates_a  = [s[2] for s in amb_type_stats]
            bar_colors = C_AMB[:len(labels_a)] if len(labels_a) <= len(C_AMB) else C_AMB * 2
            bars = ax.bar(labels_a, rates_a, color=bar_colors[:len(labels_a)], edgecolor="white")
            for bar, rate, n in zip(bars, rates_a, ns_a):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                        f"{rate:.0f}%\n(n={n})", ha="center", va="bottom", fontsize=8)
            ax.set_ylabel("P1 pass rate (%)")
            ax.set_ylim(0, max(rates_a or [100]) * 1.25)
            ax.set_title("P1 Pass Rate by\nCritical Ambiguity Type")
            ax.set_xticklabels(labels_a, rotation=20, ha="right", fontsize=8)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("P1 Pass Rate by Critical Ambiguity Type")
    else:
        ax.text(0.5, 0.5, "no dataset", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("P1 Pass Rate by Critical Ambiguity Type")

    fig1.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"\n  Charts (results)     → {path1}")

    # ══════════════════════════════════════════════════════════════════════════
    # PNG 2 — Diagnostics  (4 charts: turns, regression, timing, unresolved)
    # ══════════════════════════════════════════════════════════════════════════
    path2 = out_path.with_name(out_path.stem.replace("_charts", "_charts_diag") + ".png")
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle(f"BIRD-Interact Run Analysis — Diagnostics [{dataset_label.upper()}]", fontsize=13, fontweight="bold")
    fig2.subplots_adjust(hspace=0.45, wspace=0.38, left=0.08, right=0.95, top=0.92, bottom=0.1)

    # Chart D: Turns-spare histogram stacked by difficulty
    ax = axes2[0, 0]
    spare_by_diff: dict[str, list[int]] = defaultdict(list)
    for r in ok:
        if r.get("turns_used") is None or not r.get("max_turn"):
            continue
        spare = r["max_turn"] - r["turns_used"]
        diff = _diff_label(dataset.get(r["instance_id"]), dataset_label) if dataset else "Unknown"
        spare_by_diff[diff].append(spare)
    all_spares = [s for vals in spare_by_diff.values() for s in vals]
    if all_spares:
        bins = list(range(min(all_spares), max(all_spares) + 2))
        bottoms = [0] * len(bins)
        for diff in _diff_order(dataset_label):
            vals = spare_by_diff.get(diff, [])
            if not vals:
                continue
            counts = [vals.count(b) for b in bins]
            ax.bar(bins, counts, bottom=bottoms, label=diff,
                   color=C_DIFF[diff], edgecolor="white", width=0.8)
            bottoms = [b + c for b, c in zip(bottoms, counts)]
        ax.axvline(-0.5, color="#333", linewidth=1.5, linestyle="--", label="hit max →")
        ax.set_xlabel("Turns spare (max_turn − turns_used)")
        ax.set_ylabel("Tasks")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
        ax.set_title("Turn Budget Utilisation\nby Difficulty")
        ax.legend(fontsize=7)

    # Chart E: P1 runtime distribution — % of tasks per runtime bucket,
    # each bucket split into two side-by-side bars (clarify / SQL-gen share).
    # Buckets always number 4, spanning [min, max] of observed P1 runtimes.
    ax = axes2[0, 1]
    N_BUCKETS = 4
    p1_times: list[tuple[float, float, float]] = []  # (total, clarify, sqlgen)
    for r in ok:
        t = r.get("timing") or {}
        c = t.get("phase1_clarification_secs") or 0
        s = t.get("phase1_sql_gen_secs") or 0
        total = c + s
        if total > 0:
            p1_times.append((total, c, s))
    if p1_times:
        min_total = min(p[0] for p in p1_times)
        max_total = max(p[0] for p in p1_times)
        width = (max_total - min_total) / N_BUCKETS if max_total > min_total else 1.0
        bucket_clarify: list[float] = [0.0] * N_BUCKETS
        bucket_sqlgen:  list[float] = [0.0] * N_BUCKETS
        bucket_counts:  list[int]   = [0] * N_BUCKETS
        for total, c, s in p1_times:
            bi = min(int((total - min_total) // width), N_BUCKETS - 1) if width > 0 else 0
            bucket_counts[bi] += 1
            bucket_clarify[bi] += c
            bucket_sqlgen[bi]  += s
        n_all = len(p1_times)
        pct_per_bucket = [cnt / n_all * 100 for cnt in bucket_counts]
        clarify_pct, sqlgen_pct = [], []
        for i in range(N_BUCKETS):
            comp_sum = bucket_clarify[i] + bucket_sqlgen[i]
            if comp_sum > 0:
                clarify_pct.append(pct_per_bucket[i] * bucket_clarify[i] / comp_sum)
                sqlgen_pct.append(pct_per_bucket[i] * bucket_sqlgen[i] / comp_sum)
            else:
                clarify_pct.append(0.0)
                sqlgen_pct.append(0.0)
        labels_rt = [f"{min_total + i*width:.0f}-{min_total + (i+1)*width:.0f}" for i in range(N_BUCKETS)]
        xi = list(range(N_BUCKETS))
        bar_w = 0.38
        offsets = [-bar_w / 2, bar_w / 2]
        ax.bar([x + offsets[0] for x in xi], clarify_pct, width=bar_w, label="Clarification", color=C_CLARIFY)
        ax.bar([x + offsets[1] for x in xi], sqlgen_pct, width=bar_w, label="SQL gen", color=C_SQLGEN)
        for i in range(N_BUCKETS):
            if bucket_counts[i] > 0:
                top = max(clarify_pct[i], sqlgen_pct[i])
                ax.text(i, top + 1, f"n={bucket_counts[i]}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(xi)
        ax.set_xticklabels(labels_rt, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("P1 runtime (s)")
        ax.set_ylabel("Tasks (%)")
        ax.set_ylim(0, max(clarify_pct + sqlgen_pct or [10]) * 1.3)
        ax.set_title("P1 Runtime Distribution\n(side-by-side: clarify vs SQL-gen share)")
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "no P1 timing data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("P1 Runtime Distribution")

    # Chart F: Timing breakdown stacked bar per difficulty
    ax = axes2[1, 0]
    diff_clarify: dict[str, list[float]] = defaultdict(list)
    diff_sqlgen:  dict[str, list[float]] = defaultdict(list)
    diff_debug:   dict[str, list[float]] = defaultdict(list)
    for r in ok:
        t = r.get("timing") or {}
        c = t.get("phase1_clarification_secs") or 0
        s = t.get("phase1_sql_gen_secs") or 0
        d = t.get("phase1_debug_total_secs") or 0
        if c + s <= 0:
            continue
        diff = _diff_label(dataset.get(r["instance_id"]), dataset_label) if dataset else "All"
        diff_clarify[diff].append(c)
        diff_sqlgen[diff].append(s)
        diff_debug[diff].append(d)
    diff_labels = [k for k in _diff_order(dataset_label) + ["All"] if k in diff_clarify]
    if diff_labels:
        avg_c = [sum(diff_clarify[d]) / len(diff_clarify[d]) for d in diff_labels]
        avg_s = [sum(diff_sqlgen[d])  / len(diff_sqlgen[d])  for d in diff_labels]
        avg_d = [sum(diff_debug[d])   / len(diff_debug[d])   for d in diff_labels]
        xi = range(len(diff_labels))
        ax.bar(xi, avg_c, label="Clarification", color=C_CLARIFY)
        ax.bar(xi, avg_s, bottom=avg_c, label="SQL gen", color=C_SQLGEN)
        ax.bar(xi, avg_d, bottom=[c+s for c,s in zip(avg_c, avg_s)], label="Debug", color=C_DEBUG)
        ax.set_xticks(list(xi))
        ax.set_xticklabels(diff_labels, fontsize=8)
        for tick, diff in zip(ax.get_xticklabels(), diff_labels):
            tick.set_color(C_DIFF.get(diff, "#333"))
        ax.set_ylabel("Seconds")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
        ax.set_title("Avg P1 Timing by Difficulty")
        ax.legend(fontsize=7)

    # Chart G: Unresolved entities at end vs P1 pass rate
    ax = axes2[1, 1]
    ur_buckets: dict[str, list[bool]] = {"0": [], "1": [], "2+": []}
    for r in ok:
        n_ur = len(r.get("unresolved_entities_end") or [])
        key = "0" if n_ur == 0 else ("1" if n_ur == 1 else "2+")
        ur_buckets[key].append(bool(r.get("phase1_passed")))
    labels_ur, rates_ur = [], []
    for key in ["0", "1", "2+"]:
        vals = ur_buckets[key]
        if vals:
            labels_ur.append(f"{key} unresolved\n(n={len(vals)})")
            rates_ur.append(sum(vals) / len(vals) * 100)
    if labels_ur:
        ur_colors = [C_PASS_BOTH, C_PASS_P1, C_FAIL]
        bars = ax.bar(labels_ur, rates_ur, color=ur_colors[:len(labels_ur)])
        ax.set_ylabel("P1 pass rate (%)")
        ax.set_title("Unresolved Entities at End\nvs P1 Pass Rate")
        ax.set_ylim(0, max(rates_ur or [100]) * 1.2)
        for bar, rate in zip(bars, rates_ur):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    f"{rate:.0f}%", ha="center", va="bottom", fontsize=9)

    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Charts (diagnostics) → {path2}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze overnight BIRD-Interact run results")
    parser.add_argument("results", help="Path to results JSONL file")
    parser.add_argument("--data", default=str(DEFAULT_DATA),
                        help="Path to source dataset JSONL for difficulty/ambiguity join")
    args = parser.parse_args()

    results_path = Path(args.results)
    # Accept any of:
    #   "test_limit3"                              → results/test_limit3/test_limit3.jsonl
    #   "results/test_limit3"                      → results/test_limit3/test_limit3.jsonl
    #   "results/test_limit3/test_limit3.jsonl"    → used as-is
    #   "results/test_limit3.jsonl"                → try subfolder first
    if results_path.is_dir():
        stem = results_path.name
        results_path = results_path / f"{stem}.jsonl"
    elif not results_path.suffix:
        # bare name, e.g. "test_limit3"
        stem = results_path.name
        results_path = ONTOLOGY_DIR / "results" / stem / f"{stem}.jsonl"
    elif not results_path.exists() and results_path.suffix == ".jsonl":
        # e.g. "results/test_limit3.jsonl" — try subfolder location
        stem = results_path.stem
        candidate = results_path.parent / stem / f"{stem}.jsonl"
        if candidate.exists():
            results_path = candidate
    if not results_path.exists():
        print(f"Results file not found: {results_path}", file=sys.stderr)
        sys.exit(1)

    report_path = results_path.with_name(results_path.stem + "_report.txt")
    tee = _Tee(report_path)
    sys.stdout = tee

    # ── Load data ─────────────────────────────────────────────────────────────
    records = _load_jsonl(results_path)
    if not records:
        print("No records found in results file.", file=sys.stderr)
        sys.exit(1)

    # Load dataset for join (difficulty, ambiguity types, GT)
    dataset_path = Path(args.data)
    dataset: dict[str, dict] = {}
    if dataset_path.exists():
        for row in _load_jsonl(dataset_path):
            dataset[row["instance_id"]] = row
    else:
        print(f"Warning: dataset not found at {dataset_path} — difficulty/ambiguity analysis unavailable",
              file=sys.stderr)

    # Separate error records
    errors  = [r for r in records if r.get("error")]
    ok      = [r for r in records if not r.get("error")]

    dataset_label = _infer_dataset_label(records, dataset_path)

    print("=" * 70)
    print(f"BIRD-Interact Run Analysis: {results_path.name}")
    print(f"Dataset: {dataset_label.upper()}" + (f"  ({len(dataset)} total tasks)" if dataset else ""))
    print("=" * 70)

    # ── 1. Overview ───────────────────────────────────────────────────────────
    total      = len(records)
    n_errors   = len(errors)
    n_p1_pass  = sum(1 for r in ok if r.get("phase1_passed"))
    n_has_p2   = sum(1 for r in ok if r.get("phase2_passed") is not None)
    n_p2_pass  = sum(1 for r in ok if r.get("phase2_passed"))
    n_full_pass = sum(
        1 for r in ok
        if r.get("phase1_passed") and (r.get("phase2_passed") is None or r.get("phase2_passed"))
    )
    # Leaderboard-style SR metrics (denominator = len(ok) throughout; a task with
    # no follow-up auto-passes the follow-up metrics as long as P1 passed — same
    # convention as n_full_pass above).
    n_p1_pass_no_debug = sum(
        1 for r in ok if r.get("phase1_passed") and not r.get("phase1_debug_ran")
    )
    n_followup_pass_no_debug = sum(
        1 for r in ok
        if r.get("phase1_passed") and (
            r.get("phase2_passed") is None
            or (r.get("phase2_passed") and not r.get("phase2_debug_ran"))
        )
    )

    sum_rewards = sum(r.get("total_reward", 0) for r in ok)
    # Count query-only tasks in the full dataset (for denominator excluding management)
    n_query_in_dataset = sum(
        1 for row in dataset.values()
        if (row.get("category") or "").lower() == "query"
    ) if dataset else len(ok)
    print("\n── Overview ──────────────────────────────────────────────────────────")
    print(f"Total tasks:       {total}  (errors: {n_errors}  valid: {len(ok)})")
    print(f"Phase 1 pass:      {n_p1_pass}/{len(ok)}  ({_pct(n_p1_pass, len(ok))})")
    print(f"Phase 2 pass:      {n_p2_pass}/{n_has_p2}  ({_pct(n_p2_pass, n_has_p2)})  [of tasks with follow-up that reached p2]")
    print(f"Full pass (p1+p2): {n_full_pass}/{len(ok)}  ({_pct(n_full_pass, len(ok))})")
    print(f"Avg score:         {sum_rewards / max(len(ok), 1):.3f}")
    print("\n── Leaderboard-style SR ──────────────────────────────────────────────")
    print(f"  P1 SR (no debug):         {n_p1_pass_no_debug}/{len(ok)}  ({_pct2(n_p1_pass_no_debug, len(ok))})")
    print(f"  P1 SR (+debug):           {n_p1_pass}/{len(ok)}  ({_pct2(n_p1_pass, len(ok))})")
    print(f"  Follow-up SR (no debug):  {n_followup_pass_no_debug}/{len(ok)}  ({_pct2(n_followup_pass_no_debug, len(ok))})")
    print(f"  Follow-up SR (+debug):    {n_full_pass}/{len(ok)}  ({_pct2(n_full_pass, len(ok))})")
    n_completed_naturally = len(ok)   # tasks that ran without error/timeout
    n_not_natural = total - n_completed_naturally  # errors + timeouts
    # Full benchmark size (incl. management) — use the actual joined dataset
    # size when available (this is dataset-correct for both lite [300] and
    # full [600], unlike a fixed ratio). Falls back to the old ~65%-query
    # heuristic (lite-derived) only when no dataset file could be joined.
    if dataset:
        estimated_full_benchmark = len(dataset)
    else:
        estimated_full_benchmark = max(1, round(n_completed_naturally / 0.65))

    # Total and average run time from task_start_timestamp
    task_times: list[float] = []
    for r in records:
        t = r.get("timing") or {}
        total_t = (t.get("phase1_clarification_secs") or 0) + \
                  (t.get("phase1_sql_gen_secs") or 0) + \
                  (t.get("phase1_debug_total_secs") or 0) + \
                  (t.get("phase2_sql_gen_secs") or 0) + \
                  (t.get("phase2_debug_total_secs") or 0)
        if total_t > 0:
            task_times.append(total_t)
    # Fall back to wall-clock span if timing fields sparse
    timestamps = sorted(
        r["task_start_timestamp"] for r in records if r.get("task_start_timestamp")
    )
    wall_span = (timestamps[-1] - timestamps[0]) if len(timestamps) >= 2 else None

    print("\n── Run Time ──────────────────────────────────────────────────────────")
    if task_times:
        total_task_secs = sum(task_times)
        avg_task_secs = total_task_secs / len(task_times)
        print(f"  Sum of task times:   {total_task_secs / 60:.1f} min  ({total_task_secs:.0f}s)")
        print(f"  Avg time per task:   {avg_task_secs / 60:.1f} min  ({avg_task_secs:.0f}s)  [n={len(task_times)}]")
    if wall_span is not None:
        print(f"  Wall-clock span:     {wall_span / 3600:.2f}h  ({wall_span:.0f}s)  [first→last task start]")

    pct_vs_ran = sum_rewards / max(n_completed_naturally, 1) * 100
    # Scaled to reflect management tasks (fixed 410/600 query-task ratio) without
    # letting the estimate swing based on how many tasks this particular run completed.
    pct_vs_full = pct_vs_ran * 410 / 600
    print("\n── Score (BIRD-Interact SR) ───────────────────────────────────────────")
    print(f"  vs tasks ran        ({n_completed_naturally} tasks):   {pct_vs_ran:.2f}%")
    print(f"  vs full benchmark   (~{estimated_full_benchmark} tasks, incl. management):  {pct_vs_full:.2f}%")
    print("\n── Task Completion ───────────────────────────────────────────────────")
    print(f"  Completed naturally: {n_completed_naturally}/{total}  ({_pct(n_completed_naturally, total)})")
    print(f"  Did NOT complete:    {n_not_natural}/{total}  ({_pct(n_not_natural, total)})  ← errors + timeouts")

    # ── Score distribution ────────────────────────────────────────────────────
    print("\n── Score Distribution ────────────────────────────────────────────────")
    score_buckets = {1.0: 0, 0.9: 0, 0.8: 0, 0.7: 0, 0.5: 0, 0.0: 0, "other": 0}
    for r in ok:
        reward = round(r.get("total_reward", 0.0), 2)
        if reward in score_buckets:
            score_buckets[reward] += 1
        else:
            score_buckets["other"] += 1
    n_valid = len(ok)
    for score in [1.0, 0.9, 0.8, 0.7, 0.5, 0.0]:
        cnt = score_buckets[score]
        print(f"  {score:.1f}  →  {cnt:>3} tasks  ({_pct(cnt, n_valid):>6})")
    if score_buckets["other"] > 0:
        print(f"  other → {score_buckets['other']} tasks (unexpected score values)")

    # ── 2. Pass rate by difficulty ────────────────────────────────────────────
    if dataset:
        print("\n── Pass Rate by Difficulty ───────────────────────────────────────────")
        by_diff: dict[str, list[dict]] = defaultdict(list)
        for r in ok:
            diff = _diff_label(dataset.get(r["instance_id"]), dataset_label)
            by_diff[diff].append(r)
        for diff in _diff_order(dataset_label):
            bucket = by_diff.get(diff, [])
            if not bucket:
                continue
            p1 = sum(1 for r in bucket if r.get("phase1_passed"))
            full = sum(
                1 for r in bucket
                if r.get("phase1_passed") and (r.get("phase2_passed") is None or r.get("phase2_passed"))
            )
            print(f"  {diff:<12}  n={len(bucket):>3}  p1={_pct(p1, len(bucket)):>6}  full={_pct(full, len(bucket)):>6}")

    # ── 3. Turn budget usage ──────────────────────────────────────────────────
    print("\n── Turn Budget Usage ─────────────────────────────────────────────────")
    turn_pairs = [(r["turns_used"], r["max_turn"]) for r in ok
                  if r.get("turns_used") is not None and r.get("max_turn")]
    if turn_pairs:
        gaps = [mt - tu for tu, mt in turn_pairs]
        at_max = sum(1 for g in gaps if g == 0)
        print(f"  Avg turns used:   {sum(tu for tu, _ in turn_pairs) / len(turn_pairs):.1f}")
        print(f"  Avg max_turn:     {sum(mt for _, mt in turn_pairs) / len(turn_pairs):.1f}")
        print(f"  Avg turns spare:  {sum(gaps) / len(gaps):.1f}")
        print(f"  Hit max turns:    {at_max}/{len(turn_pairs)}  ({_pct(at_max, len(turn_pairs))})")
        # Distribution of turns_spare
        spare_dist: dict[int, int] = defaultdict(int)
        for g in gaps:
            spare_dist[g] += 1
        print(f"  Spare-turns distribution: { {k: spare_dist[k] for k in sorted(spare_dist)} }")

    # ── Zero-turn proceed ─────────────────────────────────────────────────────
    zero_turn = [r for r in ok if r.get("turns_used") == 0]
    if zero_turn:
        zt_pass = sum(1 for r in zero_turn if r.get("phase1_passed"))
        zt_fail = len(zero_turn) - zt_pass
        print("\n── Clarify Proceeded with 0 Turns ────────────────────────────────────")
        print(f"  Tasks: {len(zero_turn)}  →  P1 pass: {zt_pass}  ({_pct(zt_pass, len(zero_turn))})  |  P1 fail: {zt_fail}  ({_pct(zt_fail, len(zero_turn))})")
        print(f"  Instance IDs: {', '.join(r['instance_id'] for r in zero_turn)}")

    # ── 4. Ambiguity coverage comparison ─────────────────────────────────────
    # A = everything we ever identified as unresolvable = unresolved_end ∪ resolved_entities
    # B = dataset's meaningful ambiguity terms (non-noise)
    # x = |A|, y = |B|, list1 = A\B, list2 = B\A  (fuzzy substring match)
    # Note: A may include noise terms we extracted (e.g. "sort order") that B excludes —
    # this inflates list1. String matching is a heuristic; treat list1/list2 as approximate.
    def _fuzzy_match(a: str, b: str) -> bool:
        a, b = a.lower(), b.lower()
        return a in b or b in a

    per_instance_coverage: list[dict] = []
    x_minus_y_vals: list[int] = []
    list1_lens: list[int] = []
    list2_lens: list[int] = []
    for r in ok:
        # Our total initial entities = all entities extracted on turn 0 from the original
        # ambiguous query (before any VDB/KB resolution), stored in initial_extracted_entities.
        # Falls back to extracted_entities (SQL gen) if not yet populated (old runs).
        A = list(set(r.get("initial_extracted_entities") or r.get("extracted_entities") or []))

        task = dataset.get(r["instance_id"]) if dataset else None
        if task:
            critical  = task.get("user_query_ambiguity", {}).get("critical_ambiguity", [])
            knowledge = task.get("knowledge_ambiguity", [])
            B = [a["term"] for a in (critical + knowledge)
                 if a.get("type", "") not in _NOISE_AMBIGUITY_TYPES]
        else:
            B = []

        x = len(A)
        y = len(B)
        list1 = [a for a in A if not any(_fuzzy_match(a, b) for b in B)]
        list2 = [b for b in B if not any(_fuzzy_match(b, a) for a in A)]

        per_instance_coverage.append({
            "instance_id": r["instance_id"],
            "our_identified_ambiguities": x,
            "dataset_meaningful_ambiguities": y,
            "over_under_identification": x - y,
            "we_flagged_not_in_dataset": list1,
            "dataset_missed_by_us": list2,
        })
        x_minus_y_vals.append(x - y)
        list1_lens.append(len(list1))
        list2_lens.append(len(list2))

    if dataset:
        print("\n── Ambiguity Coverage ────────────────────────────────────────────────")
        print("  (fuzzy string match — approximate; we may flag noise terms the dataset excludes)")
        n = len(x_minus_y_vals)
        if n:
            print(f"  Avg terms we identified:         {sum(r['our_identified_ambiguities'] for r in per_instance_coverage) / n:.2f}")
            print(f"  Avg terms dataset expects:       {sum(r['dataset_meaningful_ambiguities'] for r in per_instance_coverage) / n:.2f}")
            print(f"  Avg over/under-identification:   {sum(x_minus_y_vals) / n:+.2f}  (+ = we flagged more, - = we missed some)")
            print(f"  Avg we flagged not in dataset:   {sum(list1_lens) / n:.2f}")
            print(f"  Avg dataset terms we missed:     {sum(list2_lens) / n:.2f}")

    # ── Unresolved at end × turn budget ──────────────────────────────────────
    print("\n── Unresolved Entities at End × Turn Budget ──────────────────────────")
    has_unresolved   = [r for r in ok if r.get("unresolved_entities_end")]
    no_unresolved    = [r for r in ok if not r.get("unresolved_entities_end")]
    ur_hit_max  = [r for r in has_unresolved
                   if r.get("turns_used") is not None and r.get("max_turn")
                   and r["turns_used"] >= r["max_turn"]]
    ur_spare    = [r for r in has_unresolved
                   if r.get("turns_used") is not None and r.get("max_turn")
                   and r["turns_used"] < r["max_turn"]]
    ur_hit_p1   = sum(1 for r in has_unresolved if r.get("phase1_passed"))
    no_ur_p1    = sum(1 for r in no_unresolved  if r.get("phase1_passed"))
    print(f"  Tasks with unresolved entities at end: {len(has_unresolved)}/{len(ok)}  ({_pct(len(has_unresolved), len(ok))})")
    print(f"    Of those — hit max_turn:  {len(ur_hit_max)}  ({_pct(len(ur_hit_max), len(has_unresolved))})")
    print(f"    Of those — turns spare:   {len(ur_spare)}  ({_pct(len(ur_spare), len(has_unresolved))})")
    print(f"    P1 pass rate (unresolved): {_pct(ur_hit_p1, len(has_unresolved))}")
    print(f"  Tasks with no unresolved:  {len(no_unresolved)}/{len(ok)}")
    print(f"    P1 pass rate (no unresolved): {_pct(no_ur_p1, len(no_unresolved))}")

    # ── 5. Debug rescue rate ──────────────────────────────────────────────────
    # Denominators: tasks that failed the phase on first try.
    # P1 failed first try = debug ran (submitted but wrong) OR passed=False and debug didn't run (didn't submit)
    # P2 failed first try = same logic, scoped to tasks that reached P2
    print("\n── Debug Rescue Rate ─────────────────────────────────────────────────")

    p1_passed_first_try = [r for r in ok if r.get("phase1_passed") and not r.get("phase1_debug_ran")]
    p1_failed_first_try = [r for r in ok if not r.get("phase1_passed") or r.get("phase1_debug_ran")]
    # Of those that failed first try, how many had debug triggered (i.e. at least submitted SQL)?
    p1_debug_ran  = [r for r in p1_failed_first_try if r.get("phase1_debug_ran")]
    p1_debug_pass = [r for r in p1_debug_ran if r.get("phase1_passed")]

    p2_tasks = [r for r in ok if r.get("phase2_passed") is not None]
    p2_passed_first_try = [r for r in p2_tasks if r.get("phase2_passed") and not r.get("phase2_debug_ran")]
    p2_failed_first_try = [r for r in p2_tasks if not r.get("phase2_passed") or r.get("phase2_debug_ran")]
    p2_debug_ran  = [r for r in p2_failed_first_try if r.get("phase2_debug_ran")]
    p2_debug_pass = [r for r in p2_debug_ran if r.get("phase2_passed")]

    print(f"  Phase 1  (n={len(ok)}):")
    print(f"    Passed first try:  {len(p1_passed_first_try)}/{len(ok)}  ({_pct(len(p1_passed_first_try), len(ok))})")
    print(f"    Failed first try:  {len(p1_failed_first_try)}/{len(ok)}  ({_pct(len(p1_failed_first_try), len(ok))})")
    print(f"      → Debug rescued: {len(p1_debug_pass)}/{len(p1_failed_first_try)}  ({_pct(len(p1_debug_pass), len(p1_failed_first_try))})  [of failed-first-try]")
    if p2_tasks:
        print(f"  Phase 2  (n={len(p2_tasks)} tasks that reached P2):")
        print(f"    Passed first try:  {len(p2_passed_first_try)}/{len(p2_tasks)}  ({_pct(len(p2_passed_first_try), len(p2_tasks))})")
        print(f"    Failed first try:  {len(p2_failed_first_try)}/{len(p2_tasks)}  ({_pct(len(p2_failed_first_try), len(p2_tasks))})")
        print(f"      → Debug rescued: {len(p2_debug_pass)}/{len(p2_failed_first_try)}  ({_pct(len(p2_debug_pass), len(p2_failed_first_try))})  [of failed-first-try]")

    # Exec errors (anomaly — should be rare)
    exec_errs = [r for r in ok if r.get("exec_error_p1") or r.get("exec_error_p2")]
    if exec_errs:
        print(f"  ⚠ Exec errors (SQL not executable): {len(exec_errs)}")
        for r in exec_errs:
            err = r.get("exec_error_p1") or r.get("exec_error_p2") or ""
            print(f"    {r['instance_id']}: {err[:120]}")

    # ── 6. Timing by difficulty ────────────────────────────────────────────────
    if dataset:
        print("\n── Timing by Difficulty ──────────────────────────────────────────────")
        diff_p1_times: dict[str, list[float]] = defaultdict(list)
        diff_total_times: dict[str, list[float]] = defaultdict(list)
        for r in ok:
            t = r.get("timing") or {}
            diff = _diff_label(dataset.get(r["instance_id"]), dataset_label)
            p1_t = (t.get("phase1_clarification_secs") or 0) + (t.get("phase1_sql_gen_secs") or 0)
            if p1_t > 0:
                diff_p1_times[diff].append(p1_t)
            total_t = p1_t + (t.get("phase1_debug_total_secs") or 0) + \
                      (t.get("phase2_sql_gen_secs") or 0) + (t.get("phase2_debug_total_secs") or 0)
            if total_t > 0:
                diff_total_times[diff].append(total_t)

        for diff in _diff_order(dataset_label):
            p1_vals = diff_p1_times.get(diff, [])
            total_vals = diff_total_times.get(diff, [])
            if not p1_vals:
                continue
            print(f"  {diff:<12}  avg_p1={_fmt_t_avg(p1_vals):>7}  avg_total={_fmt_t_avg(total_vals):>7}")

    # ── Slow P1 tasks (>120s) ──────────────────────────────────────────────────
    # Flag a task when EITHER individual component (clarify or sqlgen) exceeds
    # the threshold on its own — not the combined total.
    SLOW_THRESHOLD = 180.0
    slow_tasks: list[tuple[str, float, float, float, str]] = []  # (id, total, clarify, sqlgen, tag)
    for r in ok:
        t = r.get("timing") or {}
        c = t.get("phase1_clarification_secs") or 0
        s = t.get("phase1_sql_gen_secs") or 0
        clarify_over = c > SLOW_THRESHOLD
        sqlgen_over = s > SLOW_THRESHOLD
        if not (clarify_over or sqlgen_over):
            continue
        if clarify_over and sqlgen_over:
            tag = "both"
        elif clarify_over:
            tag = "clarify"
        else:
            tag = "sql gen"
        slow_tasks.append((r["instance_id"], c + s, c, s, tag))
    if slow_tasks:
        slow_tasks.sort(key=lambda x: x[1], reverse=True)
        print(f"\n── Slow P1 Tasks (>{SLOW_THRESHOLD:.0f}s) ────────────────────────────────────────")
        print(f"  {len(slow_tasks)} task(s) exceeded {SLOW_THRESHOLD:.0f}s in phase 1 "
              f"(clarify={_fmt_t(sum(x[2] for x in slow_tasks) / len(slow_tasks))} avg, "
              f"sqlgen={_fmt_t(sum(x[3] for x in slow_tasks) / len(slow_tasks))} avg):")
        for instance_id, total, c, s, tag in slow_tasks:
            print(f"    {instance_id:<40}  {total:>6.1f}s  "
                  f"(clarify={c:.1f}s, sqlgen={s:.1f}s)  ({tag})")

    # ── 7. Per-database stats ──────────────────────────────────────────────────
    print("\n── Per-Database Stats ────────────────────────────────────────────────")
    db_records: dict[str, list[dict]] = defaultdict(list)
    for r in ok:
        db_records[r["database"]].append(r)
    db_short = _abbrev_dbs(sorted(db_records))
    for db, recs in sorted(db_records.items()):
        p1_first  = sum(1 for r in recs if r.get("phase1_passed") and not r.get("phase1_debug_ran"))
        p1_total  = sum(1 for r in recs if r.get("phase1_passed"))
        avg_score = sum(r.get("total_reward", 0) for r in recs) / len(recs)
        times = []
        for r in recs:
            t = r.get("timing") or {}
            total_t = (t.get("phase1_clarification_secs") or 0) + (t.get("phase1_sql_gen_secs") or 0) + \
                      (t.get("phase1_debug_total_secs") or 0) + (t.get("phase2_sql_gen_secs") or 0) + \
                      (t.get("phase2_debug_total_secs") or 0)
            if total_t > 0:
                times.append(total_t)
        print(f"  {db_short[db]:<5} {db:<20}  n={len(recs):>3}  p1_1st={_pct(p1_first, len(recs)):>6}  p1_final={_pct(p1_total, len(recs)):>6}  avg_score={avg_score:.2f}  avg_time={_fmt_t_avg(times):>7}")

    # ── 8. Performance regression over time ────────────────────────────────────
    print("\n── Performance Regression Over Time ──────────────────────────────────")
    timestamped = sorted(
        [r for r in ok if r.get("task_start_timestamp")],
        key=lambda r: r["task_start_timestamp"],
    )
    if len(timestamped) >= 4:
        # Split into quarters and compare pass rate + avg time
        q = len(timestamped) // 4
        quarters = [
            timestamped[:q],
            timestamped[q:2*q],
            timestamped[2*q:3*q],
            timestamped[3*q:],
        ]
        print(f"  {'Quarter':<10}  {'n':>4}  {'p1_pass':>8}  {'avg_p1_time':>12}")
        for i, bucket in enumerate(quarters, 1):
            p1 = sum(1 for r in bucket if r.get("phase1_passed"))
            times = []
            for r in bucket:
                t = r.get("timing") or {}
                p1_t = (t.get("phase1_clarification_secs") or 0) + (t.get("phase1_sql_gen_secs") or 0)
                if p1_t > 0:
                    times.append(p1_t)
            print(f"  Q{i:<9}  {len(bucket):>4}  {_pct(p1, len(bucket)):>8}  {_fmt_t_avg(times):>12}")
        # Wall-clock span of the run
        t_first = timestamped[0]["task_start_timestamp"]
        t_last  = timestamped[-1]["task_start_timestamp"]
        print(f"  Total run span: {(t_last - t_first) / 3600:.2f}h")
    else:
        print("  (need ≥4 tasks with timestamps for regression analysis)")

    # ── Stuck phrases ─────────────────────────────────────────────────────────
    STUCK_PHRASES = ["out of scope", "not certain", "uncertain", "cannot answer",
                     "can't answer", "unable to answer", "not able to answer"]
    stuck_per_task: list[int] = []
    for r in ok:
        history = r.get("dialogue_history") or []
        user_turns = [t["content"].lower() for t in history if t.get("role") == "user"]
        stuck_count = sum(
            1 for content in user_turns
            if any(phrase in content for phrase in STUCK_PHRASES)
        )
        stuck_per_task.append(stuck_count)
    n_with_stuck     = sum(1 for c in stuck_per_task if c > 0)
    n_with_multi_stuck = sum(1 for c in stuck_per_task if c > 1)
    total_stuck      = sum(stuck_per_task)
    print("\n── Stuck Phrase Responses ────────────────────────────────────────────")
    print(f"  Tasks with ≥1 stuck answer:   {n_with_stuck}/{len(ok)}  ({_pct(n_with_stuck, len(ok))})")
    print(f"  Tasks with >1 stuck answer:   {n_with_multi_stuck}/{len(ok)}  ({_pct(n_with_multi_stuck, len(ok))})")
    print(f"  Avg stuck answers per task:   {total_stuck / max(len(ok), 1):.2f}  (ratio: {total_stuck}/{len(ok)})")

    # ── Follow-up type pass rate ───────────────────────────────────────────────
    # For each follow-up type: given p1 passed and has follow-up, what % pass p2?
    # Broken down by difficulty tier of the FOLLOW-UP question.
    followup_stats: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    if dataset:
        for r in ok:
            if r.get("phase2_passed") is None:
                continue  # no follow-up or p1 failed
            task = dataset.get(r["instance_id"]) or {}
            fu = task.get("follow_up") or {}
            fu_type = fu.get("type", "unknown")
            # lite's follow_up carries its own difficulty_tier (more precise than the
            # parent task's); full's follow_up has no difficulty/high_level flag at
            # all, so fall back to the parent task's high_level label there.
            fu_diff = _diff_label(fu, dataset_label) if dataset_label != "full" else _diff_label(task, dataset_label)
            followup_stats[fu_type][fu_diff].append(bool(r.get("phase2_passed")))
        if followup_stats:
            print("\n── Follow-up Type P2 Pass Rate (given P1 passed) ────────────────────")
            for fu_type in sorted(followup_stats):
                by_diff = followup_stats[fu_type]
                parts = []
                for diff in _diff_order(dataset_label):
                    vals = by_diff.get(diff, [])
                    if vals:
                        parts.append(f"{sum(vals)}/{len(vals)} {diff.lower()}")
                total_vals = [v for vals in by_diff.values() for v in vals]
                pct = _pct(sum(total_vals), len(total_vals))
                print(f"  {fu_type:<20}  {pct:>6}  [{', '.join(parts)}]")

    # ── Critical ambiguity type × pass rate ───────────────────────────────────
    # Grouping rule:
    #   knowledge_linking: tasks where ALL critical_ambiguity types = knowledge_linking only
    #   other type T:      tasks where ALL critical_ambiguity types ⊆ {T, knowledge_linking}
    # Multiple entries of the same type within a task are fine.
    # knowledge_ambiguity (deleted KB entries) treated as a separate count metric.
    if dataset:
        KNOWLEDGE_LINKING = "knowledge_linking_ambiguity"

        # Collect all non-noise types present in the dataset
        all_types: set[str] = set()
        for row in dataset.values():
            for a in row.get("user_query_ambiguity", {}).get("critical_ambiguity", []):
                t = a.get("type", "")
                if t and t not in _NOISE_AMBIGUITY_TYPES:
                    all_types.add(t)

        # For each type, collect matching tasks and their pass/score
        type_groups: dict[str, list[dict]] = {}
        for target_type in sorted(all_types):
            allowed = {target_type, KNOWLEDGE_LINKING}
            group = []
            for r in ok:
                task = dataset.get(r["instance_id"]) or {}
                crit = task.get("user_query_ambiguity", {}).get("critical_ambiguity", [])
                types_in_task = {
                    a.get("type", "") for a in crit
                    if a.get("type", "") and a.get("type", "") not in _NOISE_AMBIGUITY_TYPES
                }
                if not types_in_task:
                    continue
                if target_type == KNOWLEDGE_LINKING:
                    # Only tasks with exclusively knowledge_linking
                    if types_in_task == {KNOWLEDGE_LINKING}:
                        group.append(r)
                else:
                    # Tasks where types ⊆ {target_type, knowledge_linking}
                    if types_in_task <= allowed and target_type in types_in_task:
                        group.append(r)
            type_groups[target_type] = group

        if any(type_groups.values()):
            print("\n── Critical Ambiguity Type × Pass Rate ───────────────────────────────")
            print("  (knowledge_linking: pure only; others: allow knowledge_linking alongside)")
            print(f"  {'Type':<35}  {'n':>4}  {'p1_pass':>8}  {'avg_score':>10}")
            for t, group in type_groups.items():
                if not group:
                    continue
                p1 = sum(1 for r in group if r.get("phase1_passed"))
                avg_s = sum(r.get("total_reward", 0) for r in group) / len(group)
                print(f"  {t:<35}  {len(group):>4}  {_pct(p1, len(group)):>8}  {avg_s:>10.3f}")

        # knowledge_ambiguity count vs pass rate
        kb_deleted_groups: dict[str, list[dict]] = {"0": [], "1": [], "2+": []}
        for r in ok:
            task = dataset.get(r["instance_id"]) or {}
            n_del = len(task.get("knowledge_ambiguity", []))
            key = "0" if n_del == 0 else ("1" if n_del == 1 else "2+")
            kb_deleted_groups[key].append(r)
        print("\n── Deleted KB Entries (knowledge_ambiguity count) × Pass Rate ────────")
        for key in ["0", "1", "2+"]:
            group = kb_deleted_groups[key]
            if not group:
                continue
            p1 = sum(1 for r in group if r.get("phase1_passed"))
            avg_s = sum(r.get("total_reward", 0) for r in group) / len(group)
            print(f"  {key} deleted KB entries:  n={len(group):>3}  p1_pass={_pct(p1, len(group)):>6}  avg_score={avg_s:.3f}")

    # ── 9. Errors ─────────────────────────────────────────────────────────────
    if errors:
        print("\n── Errors / Timeouts ─────────────────────────────────────────────────")
        err_types: dict[str, int] = defaultdict(int)
        for r in errors:
            err = r.get("error", "unknown")
            key = err.split(":")[0] if ":" in err else err[:40]
            err_types[key] += 1
        for key, cnt in sorted(err_types.items(), key=lambda x: -x[1]):
            print(f"  {cnt:>3}x  {key}")
        print(f"  Affected instances: {', '.join(r['instance_id'] for r in errors)}")

    # ── Write analysis JSON ───────────────────────────────────────────────────
    analysis_out = results_path.with_name(results_path.stem + "_analysis.json")
    analysis_data = {
        "source": str(results_path),
        "dataset": dataset_label,
        "total_tasks": total,
        "n_errors": n_errors,
        "n_valid": len(ok),
        "phase1_pass": n_p1_pass,
        "phase2_pass": n_p2_pass,
        "full_pass": n_full_pass,
        "p1_pass_no_debug": n_p1_pass_no_debug,
        "p1_sr_no_debug_pct": round(n_p1_pass_no_debug / max(len(ok), 1) * 100, 2),
        "p1_sr_with_debug_pct": round(n_p1_pass / max(len(ok), 1) * 100, 2),
        "followup_pass_no_debug": n_followup_pass_no_debug,
        "followup_sr_no_debug_pct": round(n_followup_pass_no_debug / max(len(ok), 1) * 100, 2),
        "followup_sr_with_debug_pct": round(n_full_pass / max(len(ok), 1) * 100, 2),
        "sum_rewards": round(sum_rewards, 4),
        "avg_reward": round(sum_rewards / max(len(ok), 1), 4),
        "score_vs_completed_naturally_pct": round(sum_rewards / max(n_completed_naturally, 1) * 100, 2),
        "score_vs_query_tasks_pct": round(sum_rewards / max(n_query_in_dataset, 1) * 100, 2),
        "score_vs_full_benchmark_pct": round(pct_vs_full, 2),
        "n_query_in_dataset": n_query_in_dataset,
        "n_completed_naturally": n_completed_naturally,
        "n_not_completed_naturally": n_not_natural,
        "pct_not_completed_naturally": round(n_not_natural / max(total, 1) * 100, 2),
        "score_distribution": {str(k): v for k, v in score_buckets.items()},
        # Per-task ambiguity coverage (x, y, x-y, list1, list2)
        "ambiguity_coverage_by_instance": per_instance_coverage,
        "has_unresolved_count": len(has_unresolved),
        "unresolved_hit_max_turn": len(ur_hit_max),
        "unresolved_spare_turns": len(ur_spare),
        "errors": [
            {"instance_id": r["instance_id"], "error": r.get("error")}
            for r in errors
        ],
    }
    analysis_out.write_text(json.dumps(analysis_data, indent=2))

    # ── Instance index ────────────────────────────────────────────────────────
    instances_out = _write_instances_file(results_path, ok, errors, dataset, dataset_label)

    # ── Charts ────────────────────────────────────────────────────────────────
    charts_out = results_path.with_name(results_path.stem + "_charts.png")
    _make_charts(ok, timestamped, dataset, score_buckets, followup_stats, charts_out, dataset_label)

    print(f"\n{'=' * 70}")
    print("Done.")
    print(f"  Results:  {results_path}")
    print(f"  Analysis: {analysis_out}")
    print(f"  Instances: {instances_out}")
    charts_diag = charts_out.with_name(charts_out.stem.replace("_charts", "_charts_diag") + ".png")
    if charts_out.exists():
        print(f"  Charts (results):     {charts_out}")
    if charts_diag.exists():
        print(f"  Charts (diagnostics): {charts_diag}")
    print(f"  Report:   {report_path}")

    tee.close()


def _write_instances_file(
    results_path: Path,
    ok: list[dict],
    errors: list[dict],
    dataset: dict[str, dict],
    dataset_label: str,
) -> Path:
    """Write a human-readable instance index alongside the analysis JSON.

    Instances are split into three outcome buckets:
      P1 FAIL / P1 PASS + P2 FAIL / P1 PASS + P2 PASS

    Within each bucket every dimension lists the instance_ids that belong to it.

    NOTE on Ambiguity Type counts: an instance is assigned to at most ONE type
    bucket (same bucketing rule as the graph — knowledge_linking pure-only;
    others allow knowledge_linking alongside).  Instances whose critical
    ambiguity spans two *different* non-noise, non-knowledge-linking types
    (e.g. schema_linking + intent) fall into "mixed_types" and are excluded
    from the per-type lists.  "no_critical_ambiguity" covers instances with
    no critical ambiguity entries after filtering noise types.  Therefore
    per-type lists + no_critical + mixed_types = total bucket; per-type lists
    alone will not sum to total.
    """
    KNOWLEDGE_LINKING = "knowledge_linking_ambiguity"

    # ── helpers ───────────────────────────────────────────────────────────────
    def _ambiguity_bucket(r: dict) -> str:
        """Return the ambiguity type bucket for a result row, or a sentinel."""
        task = dataset.get(r["instance_id"]) or {}
        crit = task.get("user_query_ambiguity", {}).get("critical_ambiguity", [])
        types = {
            a.get("type", "")
            for a in crit
            if a.get("type", "") and a.get("type", "") not in _NOISE_AMBIGUITY_TYPES
        }
        if not types:
            return "no_critical_ambiguity"
        if types == {KNOWLEDGE_LINKING}:
            return KNOWLEDGE_LINKING
        non_kl = types - {KNOWLEDGE_LINKING}
        if len(non_kl) == 1:
            return next(iter(non_kl))
        return "mixed_types"  # spans multiple non-KL types — excluded from per-type lists

    def _diff(r: dict) -> str:
        return _diff_label(dataset.get(r["instance_id"]), dataset_label)

    def _fu_type(r: dict) -> str:
        task = dataset.get(r["instance_id"]) or {}
        return (task.get("follow_up") or {}).get("type", "unknown")

    def _section(lines: list[str], title: str, records: list[dict], *, extra: str = "") -> None:
        tag = f" [{extra}]" if extra else ""
        lines.append(f"  {title}{tag}:")
        if not records:
            lines.append("    —")
        else:
            lines.append("    " + ", ".join(r["instance_id"] for r in records))

    # ── outcome buckets ───────────────────────────────────────────────────────
    p1_fail     = [r for r in ok if not r.get("phase1_passed")]
    p1p_p2f     = [r for r in ok if r.get("phase1_passed") and r.get("phase2_passed") is False]
    full_pass   = [r for r in ok if r.get("phase1_passed")
                   and (r.get("phase2_passed") is None or r.get("phase2_passed"))]

    # ── collect all non-noise ambiguity types in run ──────────────────────────
    all_amb_types: set[str] = set()
    for r in ok:
        task = dataset.get(r["instance_id"]) or {}
        for a in task.get("user_query_ambiguity", {}).get("critical_ambiguity", []):
            t = a.get("type", "")
            if t and t not in _NOISE_AMBIGUITY_TYPES:
                all_amb_types.add(t)
    sorted_amb = sorted(all_amb_types - {KNOWLEDGE_LINKING}) + [KNOWLEDGE_LINKING]

    # ── difficulty ordering ───────────────────────────────────────────────────
    diff_order = _diff_order(dataset_label)

    # ── build sections per bucket ─────────────────────────────────────────────
    lines: list[str] = []
    lines.append(
        "Instance index — same ambiguity bucketing as graph\n"
        "(per-type lists exclude mixed-type instances; see 'mixed_types' entry)\n"
    )

    for bucket_label, bucket in [
        (f"P1 FAIL (n={len(p1_fail)})", p1_fail),
        (f"P1 PASS + P2 FAIL (n={len(p1p_p2f)})", p1p_p2f),
        (f"P1 PASS + P2 PASS (n={len(full_pass)})", full_pass),
    ]:
        lines.append("═" * 70)
        lines.append(f"  {bucket_label}")
        lines.append("═" * 70)
        lines.append("")

        # Ambiguity type
        lines.append("  Ambiguity Type:")
        for atype in sorted_amb + ["no_critical_ambiguity", "mixed_types"]:
            members = [r for r in bucket if _ambiguity_bucket(r) == atype]
            _section(lines, f"  {atype}", members)
        lines.append("")

        # Difficulty
        lines.append("  Difficulty:")
        for d in diff_order:
            members = [r for r in bucket if _diff(r) == d]
            if members:
                _section(lines, f"  {d}", members)
        lines.append("")

        # Turn budget (hit max)
        hit_max = [r for r in bucket
                   if r.get("turns_used") is not None and r.get("max_turn")
                   and r["turns_used"] >= r["max_turn"]]
        lines.append("  Hit Turn Budget (turns_used == max_turn):")
        if hit_max:
            lines.append("    " + ", ".join(
                f"{r['instance_id']} ({r['turns_used']}/{r['max_turn']})" for r in hit_max
            ))
        else:
            lines.append("    —")
        lines.append("")

        # Follow-up type (only meaningful for p2 — always show, empty for p1 fail)
        if bucket_label.startswith("P1 PASS"):
            lines.append("  Follow-up Type:")
            fu_types_in_bucket = sorted({_fu_type(r) for r in bucket})
            for ft in fu_types_in_bucket:
                members = [r for r in bucket if _fu_type(r) == ft]
                _section(lines, f"  {ft}", members)
            lines.append("")

        # Debug rescue (only for p2 section — p1 debug rescue is a p1 outcome)
        if bucket_label.startswith("P1 PASS + P2"):
            rescued = [r for r in bucket if r.get("phase2_debug_ran") and r.get("phase2_passed")]
            lines.append("  Debug Rescue (p2_debug rescued p2):")
            _section(lines, "  rescued", rescued)
            lines.append("")

        if bucket_label.startswith("P1 FAIL"):
            # p1 debug rescued → phase1_passed=True → not in p1_fail bucket
            lines.append("  Debug Ran but P1 Still Failed:")
            debug_ran_failed = [r for r in bucket if r.get("phase1_debug_ran")]
            _section(lines, "  debug_ran_still_failed", debug_ran_failed)
            lines.append("")

        # Exec errors
        exec_p1 = [r for r in bucket if r.get("exec_error_p1")]
        exec_p2 = [r for r in bucket if r.get("exec_error_p2")]
        if exec_p1 or exec_p2:
            lines.append("  Exec Errors:")
            if exec_p1:
                _section(lines, "  exec_error_p1", exec_p1)
            if exec_p2:
                _section(lines, "  exec_error_p2", exec_p2)
            lines.append("")

        # Harness errors (error field set — separate from ok)
        # (errors are outside buckets, but list them once after the p1-fail bucket)
        lines.append("")

    # ── harness errors (not in any outcome bucket) ────────────────────────────
    lines.append("═" * 70)
    lines.append(f"  HARNESS ERRORS (n={len(errors)})  — excluded from all buckets above")
    lines.append("═" * 70)
    lines.append("")
    if errors:
        for r in errors:
            lines.append(f"  {r['instance_id']}: {r.get('error', '?')}")
    else:
        lines.append("  —")
    lines.append("")

    out_path = results_path.with_name(results_path.stem + "_instances.txt")
    out_path.write_text("\n".join(lines))
    return out_path


if __name__ == "__main__":
    main()
