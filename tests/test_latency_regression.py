"""Phase 7 (Mark) -- latency regression guards.

Anthony's tests in test_train_production.py encode quality-metric floors
(AUPRC, AUROC, F1, cost). They DO NOT encode latency floors. These tests
fill that gap.

The Phase 6 latency benchmark (`results/mark_phase6_latency.json`) is the
source of truth. A future commit that re-runs the benchmark and degrades
the percentiles below these floors will fail before it ships:

  - single-call p50  : 12.4 ms      -> floor 25 ms
  - single-call p95  : 37.4 ms      -> floor 60 ms
  - single-call p99  : 80.4 ms      -> floor 150 ms
  - batch  p50/row   : 14.7 us      -> floor 30 us
  - batch  p99/row   : 15.9 us      -> floor 40 us
  - per-call CB p50  : 4.2 ms/1k    -> floor 10 ms
  - per-call XGB p50 : 8.6 ms/1k    -> floor 20 ms
  - per-call LGB p50 : 28.6 ms/1k   -> floor 60 ms

All thresholds give ~2x headroom over Phase-6 measured values, so genuine
hardware/library regressions trip them but expected noise (5-10%) does not.

Plus: the headline Phase-6 finding ("840x single-vs-batch gap") is encoded
as a regression test - if that gap shrinks below 100x, the production-mode
recommendation in the README needs to be re-checked.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
LATENCY_JSON = REPO / "results" / "mark_phase6_latency.json"


@pytest.fixture(scope="module")
def latency():
    if not LATENCY_JSON.exists():
        pytest.skip(f"Latency benchmark not found: {LATENCY_JSON}")
    return json.loads(LATENCY_JSON.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Single-call percentiles (production REST/Streamlit traffic shape)
# ---------------------------------------------------------------------------

def test_single_call_p50_below_floor(latency):
    p50 = latency["single_call_predict_one_ms"]["p50_ms"]
    assert p50 < 25.0, f"single-call p50 = {p50:.2f} ms exceeds 25 ms floor"


def test_single_call_p95_below_floor(latency):
    p95 = latency["single_call_predict_one_ms"]["p95_ms"]
    assert p95 < 60.0, f"single-call p95 = {p95:.2f} ms exceeds 60 ms floor"


def test_single_call_p99_below_floor(latency):
    p99 = latency["single_call_predict_one_ms"]["p99_ms"]
    assert p99 < 150.0, f"single-call p99 = {p99:.2f} ms exceeds 150 ms floor"


def test_single_call_sample_size_at_least_5000(latency):
    n = latency["single_call_predict_one_ms"]["n"]
    assert n >= 5000, f"single-call benchmark only has n={n}, need >=5000 for stable percentiles"


# ---------------------------------------------------------------------------
# Batch throughput (production batch-job traffic shape)
# ---------------------------------------------------------------------------

def test_batch_per_row_p50_below_floor(latency):
    p50_us = latency["batch_predict_batch"]["per_row_ms"]["p50_ms"] * 1000
    assert p50_us < 30.0, f"batch p50/row = {p50_us:.2f} us exceeds 30 us floor"


def test_batch_per_row_p99_below_floor(latency):
    p99_us = latency["batch_predict_batch"]["per_row_ms"]["p99_ms"] * 1000
    assert p99_us < 40.0, f"batch p99/row = {p99_us:.2f} us exceeds 40 us floor"


def test_batch_throughput_at_least_30k_rows_per_sec(latency):
    p50_per_row_s = latency["batch_predict_batch"]["per_row_ms"]["p50_ms"] / 1000.0
    rows_per_sec = 1.0 / p50_per_row_s
    assert rows_per_sec >= 30_000, (
        f"batch throughput = {rows_per_sec:.0f} rows/s below 30k floor"
    )


# ---------------------------------------------------------------------------
# Per-base-learner microbenchmarks (regressions in any one learner are
# isolated by these tests, vs the ensemble-level tests above which would
# only flag the slowest of the three)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "learner,floor_ms",
    [
        ("catboost", 10.0),
        ("xgboost", 20.0),
        ("lightgbm", 60.0),
    ],
)
def test_base_learner_p50_per_1000_rows(latency, learner, floor_ms):
    p50 = latency["per_base_learner"][learner]["per_call_ms_for_1000_rows"]["p50_ms"]
    assert p50 < floor_ms, f"{learner} p50 = {p50:.2f} ms/1k rows exceeds {floor_ms} ms floor"


# ---------------------------------------------------------------------------
# Headline regression: the 840x single-vs-batch gap claim
# ---------------------------------------------------------------------------

def test_single_vs_batch_speedup_at_least_100x(latency):
    """
    Phase-6 finding: batch inference is ~840x faster per row than single calls.
    This isn't a soft 'nice to have' - the README and PRODUCTION recommendation
    rely on telling production teams 'use the batch path for offline jobs'.

    A regression below 100x would mean the single-call overhead has been amortised
    (e.g., someone enabled a process pool for predict_one) and the batch-vs-single
    architectural advice in the README needs to change. That's a real change
    requiring acknowledgement, not silent acceptance.
    """
    single_p50_ms = latency["single_call_predict_one_ms"]["p50_ms"]
    batch_p50_per_row_ms = latency["batch_predict_batch"]["per_row_ms"]["p50_ms"]
    speedup = single_p50_ms / batch_p50_per_row_ms
    assert speedup >= 100, (
        f"single-vs-batch speedup collapsed to {speedup:.0f}x (expected >=100x); "
        f"check whether predict_one was vectorised - README needs update"
    )


def test_speedup_vs_claude_opus_at_p99_at_least_100x(latency):
    """
    Phase-5 LLM head-to-head measured Claude Opus 4.6 at 24225 ms/row.
    Even at OUR p99 latency, we should be >=100x faster than Opus's median.
    This is the headline 'specialist beats frontier' claim.
    """
    speedup = latency["headline"]["speedup_vs_opus_at_p99"]
    assert speedup >= 100, (
        f"speedup vs Opus at p99 = {speedup:.1f}x below 100x floor; "
        f"the 'specialist beats frontier' headline no longer holds"
    )
