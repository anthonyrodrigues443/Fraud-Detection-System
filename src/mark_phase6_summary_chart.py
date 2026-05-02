"""Phase 6 (Mark) -- LinkedIn / X-style summary chart of the production system.

Combines (a) the production leaderboard, (b) the inference latency distribution,
and (c) the latency vs LLM frontier gap. Saved to results/mark_phase6_*.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
MODELS = REPO / "models"

metrics = json.loads((MODELS / "production_metrics.json").read_text())
latency = json.loads((RESULTS / "mark_phase6_latency.json").read_text())


# ---------------------------------------------------------------------------- #
# 1. Production leaderboard (4 models on test set)
# ---------------------------------------------------------------------------- #
def leaderboard_chart():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    names = ["catboost", "xgboost", "lightgbm", "ensemble_simple_avg"]
    labels = ["CatBoost", "XGBoost", "LightGBM", "Ensemble\n(simple avg)"]
    auprc = [metrics["models"][n]["auprc"] for n in names]
    cost = [metrics["models"][n]["cost_at_optimal_threshold"] for n in names]
    colors = ["#A23B72", "#A23B72", "#A23B72", "#2E86AB"]

    bars = axes[0].bar(labels, auprc, color=colors)
    axes[0].set_ylim(0.97, 0.99)
    axes[0].set_ylabel("Test AUPRC")
    axes[0].set_title("Production leaderboard — AUPRC", fontsize=13)
    for b, v in zip(bars, auprc):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 0.0005, f"{v:.4f}",
                      ha="center", fontsize=11, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)

    bars = axes[1].bar(labels, cost, color=colors)
    axes[1].set_ylabel("Min expected $-cost on test set (lower is better)")
    axes[1].set_title("Production leaderboard — cost @ optimal threshold", fontsize=13)
    for b, v in zip(bars, cost):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 50, f"${v:,.0f}",
                      ha="center", fontsize=11, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = RESULTS / "mark_phase6_leaderboard.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------- #
# 2. The headline -- single-call vs batch vs LLMs (log scale)
# ---------------------------------------------------------------------------- #
def headline_latency_chart():
    fig, ax = plt.subplots(1, 1, figsize=(11, 6))
    h = latency["headline"]
    median_single = h["median_predict_one_ms"]
    p99_single = h["p99_predict_one_ms"]
    median_batch_per_row_us = h["median_batch_per_row_us"]
    median_batch_per_row_ms = median_batch_per_row_us / 1000
    haiku_ms = h["claude_haiku_per_row_ms_phase5"]
    opus_ms = h["claude_opus_per_row_ms_phase5"]

    cats = [
        ("Ensemble\nbatch (1k)", median_batch_per_row_ms, "#2E86AB"),
        ("Ensemble\nsingle p50", median_single, "#2E86AB"),
        ("Ensemble\nsingle p99", p99_single, "#2E86AB"),
        ("Claude Haiku\n4.5", haiku_ms, "#A23B72"),
        ("Claude Opus\n4.6", opus_ms, "#F18F01"),
    ]
    labels = [c[0] for c in cats]
    vals = [c[1] for c in cats]
    colors = [c[2] for c in cats]

    bars = ax.bar(labels, vals, color=colors)
    ax.set_yscale("log")
    ax.set_ylim(0.005, 60_000)
    ax.set_ylabel("Latency per prediction (ms, log scale)")
    ax.set_title("Phase 6 production latency vs Phase 5 LLM frontier",
                  fontsize=14, pad=15)
    for bar, val in zip(bars, vals):
        if val < 1:
            txt = f"{val * 1000:.1f} µs"
        elif val < 100:
            txt = f"{val:,.2f} ms"
        else:
            txt = f"{val:,.0f} ms"
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.4, txt,
                  ha="center", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, which="major")

    # Speedup callouts
    s_haiku = h["speedup_vs_haiku_at_p99"]
    s_opus = h["speedup_vs_opus_at_p99"]
    ax.text(0.02, 0.96,
              f"Specialist ensemble vs Phase-5 LLM frontier:\n"
              f"  vs Haiku 4.5 (at p99 single):  {s_haiku:>7.0f}× faster\n"
              f"  vs Opus 4.6  (at p99 single):  {s_opus:>7.0f}× faster\n"
              f"  vs Opus 4.6  (at batch median): {opus_ms / median_batch_per_row_ms:>7.0f}× faster",
              transform=ax.transAxes, fontsize=10, verticalalignment="top",
              fontfamily="monospace",
              bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    plt.tight_layout()
    out = RESULTS / "mark_phase6_headline_latency.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------- #
# 3. The single-vs-batch overhead chart (Phase 6 finding)
# ---------------------------------------------------------------------------- #
def single_vs_batch_chart():
    fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))
    h = latency["headline"]
    s = latency["single_call_predict_one_ms"]
    b = latency["batch_predict_batch"]["per_row_ms"]

    cats = ["p50", "p90", "p95", "p99"]
    sing = [s["p50_ms"], s["p90_ms"], s["p95_ms"], s["p99_ms"]]
    batch = [b["p50_ms"], b["p90_ms"], b["p95_ms"], b["p99_ms"]]

    x = np.arange(len(cats))
    width = 0.35
    bars1 = ax.bar(x - width/2, sing, width, label="single-call (predict_one)",
                    color="#A23B72")
    bars2 = ax.bar(x + width/2, batch, width, label="batch (chunks of 1,000)",
                    color="#2E86AB")
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_yscale("log")
    ax.set_ylim(0.005, 200)
    ax.set_ylabel("Latency per row (ms, log scale)")
    ax.set_title("Single-call vs batched inference — same ensemble, ~840× throughput gap",
                  fontsize=13, pad=15)
    for b_, v in zip(bars1, sing):
        ax.text(b_.get_x() + b_.get_width() / 2, v * 1.5,
                  f"{v:.2f} ms", ha="center", fontsize=9, fontweight="bold")
    for b_, v in zip(bars2, batch):
        if v < 1:
            txt = f"{v * 1000:.1f} µs"
        else:
            txt = f"{v:.2f} ms"
        ax.text(b_.get_x() + b_.get_width() / 2, v * 1.5,
                  txt, ha="center", fontsize=9, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3, which="major")

    plt.tight_layout()
    out = RESULTS / "mark_phase6_single_vs_batch.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print("Wrote", leaderboard_chart())
    print("Wrote", headline_latency_chart())
    print("Wrote", single_vs_batch_chart())
