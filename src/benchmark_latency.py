"""
Phase 6 (Mark) -- inference latency benchmark.

Loads the production FraudDetector and measures wall-clock latency on:
    1. 10,000 random test rows  -- single-call (predict_one) per row.
    2. 10,000 random test rows  -- batch (predict_batch) in chunks of 1k.
    3. Per-base-learner microbenchmarks (CatBoost / XGBoost / LightGBM alone).

Reports p50 / p90 / p95 / p99 / max / mean per configuration. Writes
results/mark_phase6_latency.json and results/mark_phase6_latency.png.

The headline target from the Phase-5 report: "specialist ML beats frontier
LLM on every measurable axis." Quantifying THIS in production-relevant
percentiles is the Phase-6 contribution.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from data_pipeline import (
    CLEAN_STACK_53, FREQ_FEATURES, load_full_dataset,
    fit_frequency_encoders, apply_frequency_encoders, materialize_features,
)
from predict import FraudDetector


REPO = Path(__file__).resolve().parent.parent
PARQUET = REPO / "data" / "processed" / "mark_phase3_full.parquet"
RESULTS = REPO / "results"


def percentiles(arr_ms):
    a = np.asarray(arr_ms)
    return {
        "p50_ms": float(np.percentile(a, 50)),
        "p90_ms": float(np.percentile(a, 90)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "max_ms": float(a.max()),
        "mean_ms": float(a.mean()),
        "n": int(a.size),
    }


def main(n_single: int = 10_000, n_batch: int = 10_000, batch_size: int = 1_000) -> None:
    print("[1/4] Loading test data and detector ...")
    train_df, test_df = load_full_dataset(PARQUET)
    encoders = fit_frequency_encoders(train_df)
    test_df = apply_frequency_encoders(test_df, encoders)

    rng = np.random.default_rng(13)
    sample = test_df.sample(n=max(n_single, n_batch), random_state=13).reset_index(drop=True)

    detector = FraudDetector.load()
    X = materialize_features(sample)

    # Warm-up (CatBoost / XGB JIT-compile on first call)
    _ = detector.cb.predict_proba(X.head(8))
    _ = detector.xgb.predict_proba(X.head(8))
    _ = detector.lgb.predict(X.head(8).values)

    # ----- 1. Per-base-learner microbenchmarks (1k rows, 30 reps) -----
    print("[2/4] Per-base-learner microbenchmarks ...")
    micro_n = 1_000
    Xm = X.head(micro_n)
    learner_results = {}
    for name, fn in (
        ("catboost", lambda x: detector.cb.predict_proba(x)[:, 1]),
        ("xgboost", lambda x: detector.xgb.predict_proba(x)[:, 1]),
        ("lightgbm", lambda x: detector.lgb.predict(x.values)),
    ):
        times = []
        for _ in range(30):
            t0 = time.perf_counter()
            fn(Xm)
            times.append((time.perf_counter() - t0) * 1000)
        per_row = np.array(times) / micro_n
        learner_results[name] = {
            "per_call_ms_for_1000_rows": percentiles(times),
            "per_row_ms": percentiles(per_row.tolist()),
        }
        print(f"    {name}: median {np.median(per_row):.4f} ms/row "
              f"(p99 {np.percentile(per_row, 99):.4f} ms/row over 1k batch)")

    # ----- 2. Single-call predict_one (full FraudDetector path) -----
    print(f"[3/4] Single-call predict_one over {n_single:,} rows ...")
    single_times = []
    for i in range(n_single):
        row = sample.iloc[i].to_dict()
        # row has all 53 features + freq + raw merchant/state/city
        t0 = time.perf_counter()
        detector.predict_one(row, top_k=0)  # skip top-K for pure latency
        single_times.append((time.perf_counter() - t0) * 1000)
        if (i + 1) % 2000 == 0:
            print(f"    {i+1:,} / {n_single:,}")
    single_summary = percentiles(single_times)
    print(f"    median {single_summary['p50_ms']:.3f} ms  "
          f"p95 {single_summary['p95_ms']:.3f} ms  "
          f"p99 {single_summary['p99_ms']:.3f} ms")

    # ----- 3. Batch predict_batch in chunks of `batch_size` -----
    print(f"[4/4] Batch predict_batch in chunks of {batch_size} (total {n_batch:,}) ...")
    batch_times = []
    for start in range(0, n_batch, batch_size):
        chunk = sample.iloc[start:start + batch_size]
        t0 = time.perf_counter()
        detector.predict_batch(chunk)
        batch_times.append((time.perf_counter() - t0) * 1000)
    batch_per_row = (np.array(batch_times) / batch_size).tolist()
    batch_summary = {
        "per_call_ms_for_batch": percentiles(batch_times),
        "per_row_ms": percentiles(batch_per_row),
    }
    print(f"    per-call median {batch_summary['per_call_ms_for_batch']['p50_ms']:.2f} ms "
          f"({batch_summary['per_row_ms']['p50_ms']*1000:.1f} us/row)")

    # ----- Export -----
    out = {
        "config": {
            "n_single": n_single, "n_batch": n_batch, "batch_size": batch_size,
            "test_rows_total": int(len(test_df)),
        },
        "per_base_learner": learner_results,
        "single_call_predict_one_ms": single_summary,
        "batch_predict_batch": batch_summary,
        "headline": {
            "median_predict_one_ms": single_summary["p50_ms"],
            "p99_predict_one_ms": single_summary["p99_ms"],
            "median_batch_per_row_us": batch_summary["per_row_ms"]["p50_ms"] * 1000,
            "claude_opus_per_row_ms_phase5": 24225,
            "claude_haiku_per_row_ms_phase5": 12906,
            "speedup_vs_opus_at_p99": 24225 / single_summary["p99_ms"],
            "speedup_vs_haiku_at_p99": 12906 / single_summary["p99_ms"],
        },
    }
    (RESULTS / "mark_phase6_latency.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {RESULTS / 'mark_phase6_latency.json'}")

    # ----- Plot: latency distribution -----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(single_times, bins=50, color="#2E86AB", edgecolor="white")
    axes[0].axvline(single_summary["p50_ms"], color="black", linestyle="--",
                    label=f"p50 = {single_summary['p50_ms']:.2f} ms")
    axes[0].axvline(single_summary["p99_ms"], color="red", linestyle="--",
                    label=f"p99 = {single_summary['p99_ms']:.2f} ms")
    axes[0].set_xlabel("Latency per single-row prediction (ms)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"FraudDetector.predict_one — n={n_single:,} test rows")
    axes[0].legend()

    # Comparison vs LLMs (log scale)
    labels = ["Ensemble\n(p50)", "Ensemble\n(p99)", "Claude Haiku\n4.5", "Claude Opus\n4.6"]
    vals_ms = [single_summary["p50_ms"], single_summary["p99_ms"],
               12906, 24225]
    colors = ["#2E86AB", "#2E86AB", "#A23B72", "#F18F01"]
    bars = axes[1].bar(labels, vals_ms, color=colors)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Latency per prediction (ms, log scale)")
    axes[1].set_title("Specialist ensemble vs Phase-5 LLM frontier")
    for bar, val in zip(bars, vals_ms):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val * 1.15,
                      f"{val:,.2f} ms" if val < 100 else f"{val:,.0f} ms",
                      ha="center", fontsize=10)

    plt.tight_layout()
    out_png = RESULTS / "mark_phase6_latency.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
