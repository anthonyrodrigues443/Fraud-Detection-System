"""Phase 5 (Mark) plots:

  results/mark_phase5_group_ablation.png    -- group ablation bar chart
  results/mark_phase5_stacking.png           -- stacking leaderboard with deltas
  results/mark_phase5_calibration.png        -- reliability + Brier/ECE/cost
  results/mark_phase5_llm_vs_catboost.png    -- frontier vs specialist
  results/mark_phase5_linkedin_chart.png     -- LinkedIn headline chart
  results/mark_phase5_tweet1_*.png  ... 3   -- short-form variants
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

CACHE = Path("results/mark_phase5_cache")
RESULTS = Path("results")

mpl.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "font.family": "DejaVu Sans",
})


def plot_group_ablation():
    df = pd.read_csv(CACHE / "group_ablation.csv")
    base = df[df["drop_group"].str.startswith("(none)")]
    drops = df[~df["drop_group"].str.startswith("(none)")].copy()
    drops = drops.sort_values("delta_auprc")  # most-hurtful first (most negative)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#d62728" if d < -0.005 else ("#ff7f0e" if d < 0 else "#2ca02c")
              for d in drops["delta_auprc"]]

    axes[0].barh(drops["drop_group"], drops["delta_auprc"], color=colors)
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("ΔAUPRC vs full 53-feature stack (negative = group is critical)")
    axes[0].set_title("Group ablation: AUPRC drop when each feature group is removed")
    for y, v in enumerate(drops["delta_auprc"]):
        axes[0].text(v - 0.0008 if v < 0 else v + 0.0005, y,
                     f"{v:+.4f}", va="center",
                     ha="right" if v < 0 else "left", fontsize=9)
    axes[0].grid(axis="x", linestyle=":", alpha=0.5)

    # Cost panel
    drops2 = drops.sort_values("delta_cost", ascending=False)
    colors2 = ["#d62728" if d > 200 else ("#ff7f0e" if d > 0 else "#2ca02c")
               for d in drops2["delta_cost"]]
    axes[1].barh(drops2["drop_group"], drops2["delta_cost"], color=colors2)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Δ min-expected-cost vs full stack ($, FN=amt, FP=$1.50)")
    axes[1].set_title("Group ablation: dollar loss penalty when each group is removed")
    for y, v in enumerate(drops2["delta_cost"]):
        axes[1].text(v + (50 if v >= 0 else -50), y,
                     f"${v:+,.0f}", va="center",
                     ha="left" if v >= 0 else "right", fontsize=9)
    axes[1].grid(axis="x", linestyle=":", alpha=0.5)

    fig.suptitle(f"Mark Phase 5 — Group ablation on 53-feat CatBoost "
                  f"(baseline AUPRC = {base['auprc'].iloc[0]:.4f}, "
                  f"min cost = ${base['min_expected_cost'].iloc[0]:,.0f})",
                  fontsize=12)
    fig.tight_layout()
    out = RESULTS / "mark_phase5_group_ablation.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def plot_stacking():
    df = pd.read_csv(CACHE / "stacking.csv")
    df = df.sort_values("auprc")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = ["#1f77b4"] * len(df)
    # highlight stack
    for i, m in enumerate(df["model"]):
        if "stack" in m.lower():
            colors[i] = "#d62728"
        elif "average" in m.lower():
            colors[i] = "#ff7f0e"

    axes[0].barh(df["model"], df["auprc"], color=colors)
    axes[0].set_xlim(df["auprc"].min() - 0.005, df["auprc"].max() + 0.005)
    axes[0].set_xlabel("Test AUPRC")
    axes[0].set_title("AUPRC: stacking vs single learners")
    for y, v in enumerate(df["auprc"]):
        axes[0].text(v + 0.0002, y, f"{v:.4f}", va="center", fontsize=9)
    axes[0].grid(axis="x", linestyle=":", alpha=0.5)

    df2 = df.sort_values("min_expected_cost", ascending=False)
    colors2 = []
    for m in df2["model"]:
        if "stack" in m.lower():
            colors2.append("#d62728")
        elif "average" in m.lower():
            colors2.append("#ff7f0e")
        else:
            colors2.append("#1f77b4")
    axes[1].barh(df2["model"], df2["min_expected_cost"], color=colors2)
    axes[1].set_xlabel("Min expected $-loss on test (cost-optimal threshold)")
    axes[1].set_title("Cost: stacking vs single learners")
    for y, v in enumerate(df2["min_expected_cost"]):
        axes[1].text(v + 30, y, f"${v:,.0f}", va="center", fontsize=9)
    axes[1].grid(axis="x", linestyle=":", alpha=0.5)

    df3 = df.sort_values("f1_at_thr05")
    colors3 = []
    for m in df3["model"]:
        if "stack" in m.lower():
            colors3.append("#d62728")
        elif "average" in m.lower():
            colors3.append("#ff7f0e")
        else:
            colors3.append("#1f77b4")
    axes[2].barh(df3["model"], df3["f1_at_thr05"], color=colors3)
    axes[2].set_xlabel("F1 @ thr=0.5 on test")
    axes[2].set_title("F1@0.5: stacking vs single learners")
    for y, v in enumerate(df3["f1_at_thr05"]):
        axes[2].text(v + 0.005, y, f"{v:.3f}", va="center", fontsize=9)
    axes[2].grid(axis="x", linestyle=":", alpha=0.5)

    fig.suptitle("Mark Phase 5 — Stacking ensemble (LogReg meta on CB+XGB+LGB calibration-slice probs)",
                  fontsize=12)
    fig.tight_layout()
    out = RESULTS / "mark_phase5_stacking.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def plot_calibration():
    df = pd.read_csv(CACHE / "calibration.csv")
    cb_test = np.load(CACHE / "stack_cb_test.npy")
    p_iso = np.load(CACHE / "p_test_iso.npy")
    p_pl = np.load(CACHE / "p_test_pl.npy")
    y_test = np.load(CACHE / "y_test_for_calib.npy") if (CACHE / "y_test_for_calib.npy").exists() else None
    if y_test is None:
        # Recompute y_test
        sys.path.insert(0, "src")
        import mark_phase5_advanced as p5
        _, _, _, _, _, y_test = p5.load_phase4_data("data/processed/mark_phase3_full.parquet")
        np.save(CACHE / "y_test_for_calib.npy", y_test)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Reliability diagram
    bins = np.linspace(0, 1, 11)
    centers = (bins[:-1] + bins[1:]) / 2
    for name, p, color in [("Uncalibrated", cb_test, "#1f77b4"),
                            ("Isotonic", p_iso, "#2ca02c"),
                            ("Platt(sigmoid)", p_pl, "#ff7f0e")]:
        bin_ids = np.clip(np.digitize(p, bins) - 1, 0, 9)
        avg_p, frac_pos = [], []
        for b in range(10):
            mask = bin_ids == b
            if mask.sum() > 0:
                avg_p.append(p[mask].mean())
                frac_pos.append(y_test[mask].mean())
            else:
                avg_p.append(np.nan)
                frac_pos.append(np.nan)
        axes[0].plot(avg_p, frac_pos, "o-", label=name, color=color, linewidth=2)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect calibration")
    axes[0].set_xlabel("Predicted probability")
    axes[0].set_ylabel("Empirical fraud rate")
    axes[0].set_title("Reliability diagram (10 bins)")
    axes[0].legend()
    axes[0].grid(linestyle=":", alpha=0.5)

    # Brier / ECE / cost panel
    x = np.arange(len(df))
    width = 0.25
    axes[1].bar(x - width, df["brier"], width, label="Brier", color="#1f77b4")
    axes[1].bar(x, df["ece"], width, label="ECE", color="#d62728")
    ax2 = axes[1].twinx()
    ax2.bar(x + width, df["min_expected_cost"], width, label="min $-cost", color="#2ca02c", alpha=0.6)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["method"], rotation=15)
    axes[1].set_ylabel("Brier / ECE (lower is better)")
    ax2.set_ylabel("Min expected cost ($)")
    axes[1].set_title("Calibration scores + cost-optimal $-loss")
    axes[1].legend(loc="upper left")
    ax2.legend(loc="upper right")
    for i, (b, e, c, t) in enumerate(zip(df["brier"], df["ece"],
                                            df["min_expected_cost"],
                                            df["cost_optimal_threshold"])):
        axes[1].text(i - width, b + 0.0001, f"{b:.4f}", ha="center", fontsize=8)
        axes[1].text(i, e + 0.0001, f"{e:.4f}", ha="center", fontsize=8)
        ax2.text(i + width, c + 60, f"${c:,.0f}\n@ thr={t:.3f}",
                  ha="center", fontsize=8)
    axes[1].grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle("Mark Phase 5 — Probability calibration of CatBoost test scores",
                  fontsize=12)
    fig.tight_layout()
    out = RESULTS / "mark_phase5_calibration.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def plot_llm_vs_catboost():
    df = pd.read_csv(CACHE / "llm_vs_catboost_final.csv")
    df = df.sort_values("f1", ascending=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # F1 panel
    colors = ["#1f77b4" if "CatBoost" in m else "#d62728"
               for m in df["model"]]
    axes[0, 0].barh(df["model"], df["f1"], color=colors)
    axes[0, 0].set_xlim(0, max(1.0, df["f1"].max() + 0.1))
    axes[0, 0].set_xlabel("F1 (50-sample stratified test)")
    axes[0, 0].set_title("F1: specialist vs frontier LLMs")
    for y, v in enumerate(df["f1"]):
        axes[0, 0].text(v + 0.01, y, f"{v:.3f}", va="center", fontsize=9)
    axes[0, 0].grid(axis="x", linestyle=":", alpha=0.5)

    # Recall panel
    df2 = df.sort_values("recall", ascending=True)
    colors2 = ["#1f77b4" if "CatBoost" in m else "#d62728"
                for m in df2["model"]]
    axes[0, 1].barh(df2["model"], df2["recall"], color=colors2)
    axes[0, 1].set_xlim(0, 1.05)
    axes[0, 1].set_xlabel("Recall on 25 fraud rows")
    axes[0, 1].set_title("Recall: who catches fraud?")
    for y, v in enumerate(df2["recall"]):
        axes[0, 1].text(v + 0.01, y, f"{v:.3f}", va="center", fontsize=9)
    axes[0, 1].grid(axis="x", linestyle=":", alpha=0.5)

    # Latency panel (log-x)
    df3 = df.sort_values("latency_ms_estimate", ascending=True)
    colors3 = ["#1f77b4" if "CatBoost" in m else "#d62728"
                for m in df3["model"]]
    axes[1, 0].barh(df3["model"], df3["latency_ms_estimate"], color=colors3)
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_xlabel("Estimated latency per prediction (ms, log)")
    axes[1, 0].set_title("Latency: 1000× gap between specialist and LLM")
    for y, v in enumerate(df3["latency_ms_estimate"]):
        axes[1, 0].text(v * 1.1, y, f"{v:,.1f}", va="center", fontsize=9)
    axes[1, 0].grid(axis="x", linestyle=":", alpha=0.5, which="both")

    # Cost panel (log-x)
    df4 = df.sort_values("cost_per_1k_usd", ascending=True)
    colors4 = ["#1f77b4" if "CatBoost" in m else "#d62728"
                for m in df4["model"]]
    axes[1, 1].barh(df4["model"], df4["cost_per_1k_usd"], color=colors4)
    axes[1, 1].set_xscale("symlog", linthresh=0.001)
    axes[1, 1].set_xlabel("$-cost per 1,000 predictions (log)")
    axes[1, 1].set_title("Cost-per-1k-predictions")
    for y, v in enumerate(df4["cost_per_1k_usd"]):
        axes[1, 1].text(v * 1.1 if v > 0 else 0.001, y,
                         f"${v:,.4f}" if v < 1 else f"${v:,.2f}",
                         va="center", fontsize=9)
    axes[1, 1].grid(axis="x", linestyle=":", alpha=0.5, which="both")

    fig.suptitle("Mark Phase 5 — Frontier LLMs vs the 53-feat CatBoost specialist (50-sample stratified test)",
                  fontsize=13)
    fig.tight_layout()
    out = RESULTS / "mark_phase5_llm_vs_catboost.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def plot_linkedin():
    """Headline LinkedIn chart: F1 vs $-per-1k bubble + 'CatBoost wins on every axis' text."""
    df = pd.read_csv(CACHE / "llm_vs_catboost_final.csv")

    fig, ax = plt.subplots(figsize=(11, 6.5))
    colors = ["#1f77b4" if "CatBoost" in m else "#d62728"
               for m in df["model"]]
    sizes = [400 + 200 * np.log10(max(v, 0.0001) * 1000)
              for v in df["latency_ms_estimate"] / 1000.0]
    sizes = [max(s, 100) for s in sizes]
    ax.scatter(df["cost_per_1k_usd"] + 0.0001, df["f1"], s=sizes,
                 c=colors, alpha=0.7, edgecolor="black", linewidth=1.2)
    for _, r in df.iterrows():
        ax.annotate(r["model"],
                     xy=(r["cost_per_1k_usd"] + 0.0001, r["f1"]),
                     xytext=(8, 8), textcoords="offset points", fontsize=9)
    ax.set_xscale("symlog", linthresh=0.001)
    ax.set_xlabel("$-cost per 1,000 predictions (log scale)")
    ax.set_ylabel("F1 on 50-sample stratified test (25 fraud + 25 legit)")
    ax.set_title("Mark Phase 5 — Frontier LLMs vs 53-feat CatBoost on credit card fraud\n"
                 "(bubble size = log latency)")
    ax.grid(linestyle=":", alpha=0.5)
    fig.tight_layout()
    out = RESULTS / "mark_phase5_linkedin_chart.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def plot_tweets():
    """Three short-form charts for X/threads."""
    # Tweet 1: which group hurts most when removed
    df = pd.read_csv(CACHE / "group_ablation.csv")
    drops = df[~df["drop_group"].str.startswith("(none)")].copy()
    drops = drops.sort_values("delta_auprc")
    colors = ["#d62728" if d < -0.005 else ("#ff7f0e" if d < 0 else "#2ca02c")
              for d in drops["delta_auprc"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(drops["drop_group"], drops["delta_auprc"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("ΔAUPRC when group is removed")
    ax.set_title("Drop a feature group, measure the AUPRC penalty")
    for y, v in enumerate(drops["delta_auprc"]):
        ax.text(v - 0.0008 if v < 0 else v + 0.0005, y, f"{v:+.4f}",
                va="center", ha="right" if v < 0 else "left", fontsize=10)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(RESULTS / "mark_phase5_tweet1_groups.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {RESULTS / 'mark_phase5_tweet1_groups.png'}")

    # Tweet 2: stack vs single learners
    df = pd.read_csv(CACHE / "stacking.csv")
    df = df.sort_values("auprc")
    colors = ["#d62728" if "stack" in m.lower() else
              ("#ff7f0e" if "average" in m.lower() else "#1f77b4")
              for m in df["model"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df["model"], df["auprc"], color=colors)
    ax.set_xlim(df["auprc"].min() - 0.005, df["auprc"].max() + 0.005)
    ax.set_xlabel("Test AUPRC")
    ax.set_title("Stacking 3 boosters vs each one alone")
    for y, v in enumerate(df["auprc"]):
        ax.text(v + 0.0003, y, f"{v:.4f}", va="center", fontsize=10)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(RESULTS / "mark_phase5_tweet2_stack.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {RESULTS / 'mark_phase5_tweet2_stack.png'}")

    # Tweet 3: LLM latency
    df = pd.read_csv(CACHE / "llm_vs_catboost_final.csv")
    df = df.sort_values("latency_ms_estimate", ascending=True)
    colors = ["#1f77b4" if "CatBoost" in m else "#d62728" for m in df["model"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df["model"], df["latency_ms_estimate"], color=colors)
    ax.set_xscale("log")
    ax.set_xlabel("Latency per prediction (ms, log)")
    ax.set_title("Specialist 0.1 ms vs frontier LLMs >2,000 ms")
    for y, v in enumerate(df["latency_ms_estimate"]):
        ax.text(v * 1.1, y, f"{v:,.1f}", va="center", fontsize=10)
    ax.grid(axis="x", linestyle=":", alpha=0.5, which="both")
    fig.tight_layout()
    fig.savefig(RESULTS / "mark_phase5_tweet3_latency.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {RESULTS / 'mark_phase5_tweet3_latency.png'}")


if __name__ == "__main__":
    print("Plotting Phase 5 results ...")
    plot_group_ablation()
    plot_stacking()
    plot_calibration()
    plot_llm_vs_catboost()
    plot_linkedin()
    plot_tweets()
    print("\nDone.")
