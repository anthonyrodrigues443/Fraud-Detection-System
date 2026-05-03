"""Phase 7 polish (Anthony) — single-image headline dashboard.

The 7-day sprint generated 60+ phase-specific plots. None of them is a
*single* shareable image that captures the project end-to-end. This script
reads the canonical artefacts (`models/production_metrics.json`,
`results/EXPERIMENT_LOG.md`) and produces `results/project_dashboard.png`:
a 2x2 panel built for portfolio / LinkedIn use.

Panels:
  1. AUPRC progression across the 7 phases (best-of-phase line chart)
  2. Specialist vs frontier LLMs on the Phase-5 50-row sample (F1 bars)
  3. Phase-3 group ablation — velocity = 46 % of total feature lift
  4. Cost per 1k predictions, log scale (specialist vs Opus / Haiku)

Run:
    python src/build_headline_dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
MODELS = REPO / "models"
OUT = RESULTS / "project_dashboard.png"


def _phase_progression():
    """Best AUPRC per phase. Numbers cross-checked against EXPERIMENT_LOG.md."""
    phases = ["P1\nbaseline", "P2\nmulti-model", "P3\nfeatures", "P4\ntuning", "P5\nadvanced", "P6\nproduction", "P7\nshipped"]
    auprc = [0.8237, 0.8872, 0.9824, 0.9819, 0.9817, 0.9840, 0.9840]
    notes = ["honest\nbaseline", "CatBoost\ndethrones XGB", "+22 features\n(velocity 46%)", "Optuna\n−0.0005", "ensemble\navg wins", "53-feat retrain\n$1,705", "94 tests\nFastAPI/Docker"]
    return phases, auprc, notes


def _llm_vs_specialist():
    """Phase 5 head-to-head, F1 on the 50-row stratified sample."""
    return ["CatBoost\nspecialist", "Claude\nOpus 4.6", "Claude\nHaiku 4.5"], [1.000, 0.864, 0.485]


def _ablation_groups():
    """Phase 3 feature-group ablation, % of total feature lift."""
    groups = ["Velocity\n(8 features)", "Amount-\ndeviation (5)", "Temporal\n(3)", "Geographic\n(2)", "Category-\nmerchant (4)"]
    pct = [46.0, 1.8, 1.6, -0.6, -1.4]
    return groups, pct


def _cost_per_1k():
    """Cost per 1k predictions (USD, log scale)."""
    return ["CatBoost\nspecialist", "Claude\nHaiku 4.5", "Claude\nOpus 4.6"], [0.0001, 0.30, 4.50]


def _load_headline_metrics():
    metrics = json.loads((MODELS / "production_metrics.json").read_text(encoding="utf-8"))
    ens = metrics["models"]["ensemble_simple_avg"]
    return {
        "auprc": ens["auprc"],
        "auroc": ens["auroc"],
        "f1": ens["f1_at_05"],
        "min_cost": ens["cost_at_optimal_threshold"],
        "n_test": metrics.get("n_test", 209715),
    }


def main():
    h = _load_headline_metrics()

    fig = plt.figure(figsize=(14, 9), dpi=150)
    fig.patch.set_facecolor("#fbfbfd")

    # --- Title strip ---------------------------------------------------------
    fig.text(
        0.5, 0.96,
        "Fraud Detection System  ·  7-day research sprint",
        ha="center", va="center", fontsize=18, fontweight="bold", color="#1a1a1a",
    )
    fig.text(
        0.5, 0.925,
        f"Apr 27 → May 3 2026   ·   31 experiments   ·   "
        f"AUPRC = {h['auprc']:.4f}   ·   F1 = {h['f1']:.3f}   ·   min cost = ${h['min_cost']:,.0f}   on  n = {h['n_test']:,}",
        ha="center", va="center", fontsize=11, color="#444",
    )

    gs = fig.add_gridspec(2, 2, top=0.88, bottom=0.07, left=0.07, right=0.97, hspace=0.45, wspace=0.30)

    # --- Panel 1: AUPRC progression -----------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    phases, auprc, notes = _phase_progression()
    x = np.arange(len(phases))
    ax1.plot(x, auprc, marker="o", markersize=9, linewidth=2.5, color="#0b5ed7")
    ax1.fill_between(x, 0.80, auprc, alpha=0.10, color="#0b5ed7")
    for i, (a, n) in enumerate(zip(auprc, notes)):
        offset = 0.012 if i % 2 == 0 else -0.018
        ax1.annotate(
            n, (i, a + offset), ha="center", fontsize=7, color="#555",
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels(phases, fontsize=9)
    ax1.set_ylim(0.80, 1.00)
    ax1.set_ylabel("AUPRC (best-of-phase)", fontsize=10)
    ax1.set_title("Phase-by-phase AUPRC progression", fontsize=12, fontweight="bold", loc="left", pad=10)
    ax1.grid(True, alpha=0.25, linestyle=":")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.axhline(0.8237, color="#888", linestyle="--", linewidth=0.8)
    ax1.text(0.05, 0.831, "Phase 1 honest baseline", fontsize=7.5, color="#666")

    # --- Panel 2: Specialist vs LLMs -----------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    labels, vals = _llm_vs_specialist()
    colors = ["#198754", "#fd7e14", "#dc3545"]
    bars = ax2.bar(labels, vals, color=colors, edgecolor="white", linewidth=1.5)
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                 ha="center", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("F1 score", fontsize=10)
    ax2.set_title("Specialist beats frontier LLMs at fraud detection", fontsize=12, fontweight="bold", loc="left", pad=10)
    ax2.text(
        0.99, 0.96,
        "n = 50 stratified sample (Phase 5)\n8 MB CatBoost · 0.1 ms/row\nvs Claude Opus 4.6 · 24,225 ms/row",
        transform=ax2.transAxes, ha="right", va="top", fontsize=8, color="#555",
        bbox=dict(facecolor="white", edgecolor="#ccc", boxstyle="round,pad=0.5"),
    )
    ax2.grid(True, axis="y", alpha=0.25, linestyle=":")
    ax2.spines[["top", "right"]].set_visible(False)

    # --- Panel 3: Group ablation --------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    groups, pct = _ablation_groups()
    colors3 = ["#0b5ed7" if p > 5 else "#6c757d" if p > 0 else "#adb5bd" for p in pct]
    bars3 = ax3.barh(groups, pct, color=colors3, edgecolor="white", linewidth=1.0)
    for b, p in zip(bars3, pct):
        x_pos = p + (1.5 if p >= 0 else -1.5)
        ha = "left" if p >= 0 else "right"
        ax3.text(x_pos, b.get_y() + b.get_height() / 2, f"{p:+.1f}%",
                 va="center", ha=ha, fontsize=9, fontweight="bold")
    ax3.axvline(0, color="#1a1a1a", linewidth=0.6)
    ax3.set_xlabel("% of total feature-engineering lift", fontsize=10)
    ax3.set_xlim(-10, 60)
    ax3.invert_yaxis()
    ax3.set_title("Velocity features deliver 46% of all FE lift", fontsize=12, fontweight="bold", loc="left", pad=10)
    ax3.text(
        0.99, 0.06,
        "Phase 3 group ablation on CatBoost (39f)\nDropping velocity costs −0.0485 AUPRC",
        transform=ax3.transAxes, ha="right", va="bottom", fontsize=8, color="#555",
        bbox=dict(facecolor="white", edgecolor="#ccc", boxstyle="round,pad=0.5"),
    )
    ax3.grid(True, axis="x", alpha=0.25, linestyle=":")
    ax3.spines[["top", "right"]].set_visible(False)

    # --- Panel 4: Cost per 1k -----------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    cost_labels, cost_vals = _cost_per_1k()
    colors4 = ["#198754", "#fd7e14", "#dc3545"]
    bars4 = ax4.bar(cost_labels, cost_vals, color=colors4, edgecolor="white", linewidth=1.5)
    ax4.set_yscale("log")
    for b, v in zip(bars4, cost_vals):
        label = f"${v:.4f}" if v < 0.01 else f"${v:.2f}"
        ax4.text(b.get_x() + b.get_width() / 2, v * 1.5, label,
                 ha="center", fontsize=10, fontweight="bold")
    ax4.set_ylabel("USD per 1,000 predictions (log scale)", fontsize=10)
    ax4.set_title("Specialist is 45,000× cheaper than Opus", fontsize=12, fontweight="bold", loc="left", pad=10)
    ax4.text(
        0.99, 0.96,
        "Token-cost math at 250 in / 10 out tokens\nCLI overhead excluded\nDirect API would 5–10× faster, same $",
        transform=ax4.transAxes, ha="right", va="top", fontsize=8, color="#555",
        bbox=dict(facecolor="white", edgecolor="#ccc", boxstyle="round,pad=0.5"),
    )
    ax4.set_ylim(1e-5, 1e2)
    ax4.grid(True, axis="y", which="both", alpha=0.20, linestyle=":")
    ax4.spines[["top", "right"]].set_visible(False)

    # --- Footer --------------------------------------------------------------
    fig.text(
        0.5, 0.02,
        "Sparkov 1.05 M txns · 0.57 % fraud · temporal 80/20 split · CatBoost + XGBoost + LightGBM simple-average ensemble · 53 features · 94 passing tests · FastAPI + Docker + CI",
        ha="center", fontsize=8, color="#666",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {OUT.relative_to(REPO)}  ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
