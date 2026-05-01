"""LLM frontier head-to-head: 50 test transactions sent to Claude (Haiku +
Sonnet) and GPT-5.4 (codex). Compare to the CatBoost champion.

Sample design:
  - 50 transactions, stratified 25 fraud + 25 legit, sampled with a fixed
    random_state for reproducibility, drawn from the test set (last 20% by
    time, per Phase 1 temporal split).
  - Each LLM is asked to reply with FRAUD/LEGIT and a probability (0.0-1.0).
  - We cache every call to results/mark_phase5_cache/llm_calls.json so this
    script is idempotent and resumable.
"""

from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
import mark_phase5_advanced as p5

CACHE = Path("results/mark_phase5_cache")
CACHE.mkdir(parents=True, exist_ok=True)
LLM_CACHE = CACHE / "llm_calls.json"
SAMPLE_PATH = CACHE / "llm_sample_idx.json"

print("=== Phase 5 (Mark) LLM frontier head-to-head ===")

# Load test data
print("\n[1] Loading data ...")
train_df, test_df, X_train, X_test, y_train, y_test = p5.load_phase4_data(
    "data/processed/mark_phase3_full.parquet"
)
test_df = test_df.reset_index(drop=True)
print(f"  test: {len(test_df)} rows, {y_test.sum()} fraud")

# Stratified sample design (reproducible)
if SAMPLE_PATH.exists():
    sample_idx = np.array(json.load(open(SAMPLE_PATH)))
    print(f"  loaded sample from cache: {len(sample_idx)} indices")
else:
    rng = np.random.default_rng(42)
    fraud_idx = np.where(y_test == 1)[0]
    legit_idx = np.where(y_test == 0)[0]
    n_per = 25
    fraud_pick = rng.choice(fraud_idx, n_per, replace=False)
    legit_pick = rng.choice(legit_idx, n_per, replace=False)
    sample_idx = np.concatenate([fraud_pick, legit_pick])
    rng.shuffle(sample_idx)
    json.dump(sample_idx.tolist(), open(SAMPLE_PATH, "w"))
    print(f"  built new stratified sample: {len(sample_idx)} indices "
          f"({len(fraud_pick)} fraud + {len(legit_pick)} legit)")

# Run each LLM
print("\n[2] Querying Claude (haiku) ...")
df_haiku = p5.run_llm_eval(test_df, sample_idx, LLM_CACHE,
                             llm="claude", model="haiku")
print(f"  haiku rows total in cache: {len(df_haiku)}")

print("\n[3] Querying Claude (opus) ...")
df_opus = p5.run_llm_eval(test_df, sample_idx, LLM_CACHE,
                            llm="claude", model="opus")
print(f"  opus rows total in cache: {len(df_opus)}")

print("\n[4] Querying Codex (gpt-5.4) -- single probe to check availability ...")
# As of 2026-05-01 the codex CLI is rate-limited (Plus quota); we run one
# probe call to confirm and document. If it succeeds we run all 50.
probe_idx = sample_idx[:1]
df_codex_probe = p5.run_llm_eval(test_df, probe_idx, LLM_CACHE,
                                   llm="codex", model="gpt-5.4")
codex_probe_ok = (df_codex_probe[(df_codex_probe["llm"] == "codex") &
                                    df_codex_probe["pred_label"].notna()]
                   .shape[0]) > 0
print(f"  codex probe ok: {codex_probe_ok}")
if codex_probe_ok:
    print("  Running full codex eval ...")
    df_codex = p5.run_llm_eval(test_df, sample_idx, LLM_CACHE,
                                 llm="codex", model="gpt-5.4")
    print(f"  codex rows total in cache: {len(df_codex)}")
else:
    print("  Codex unavailable (likely usage limit). Documenting in report.")
    print("  Skipping full codex eval.")

# Aggregate metrics
print("\n[5] Aggregating ...")
all_rows = json.load(open(LLM_CACHE))
df_all = pd.DataFrame(all_rows)
print("  raw call counts:", df_all.groupby(["llm", "model"]).size().to_dict())
print("  parse-success:", df_all.dropna(subset=["pred_label"]).groupby(["llm", "model"]).size().to_dict())

metrics_rows = []
for (llm, model), _ in df_all.groupby(["llm", "model"]):
    m = p5.llm_metrics(df_all, llm, model)
    if m.get("n", 0) > 0:
        metrics_rows.append(m)
    else:
        print(f"  SKIP: {llm}/{model} -- 0 successful predictions")
metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(CACHE / "llm_metrics.csv", index=False)
print("\nLLM head-to-head:")
print(metrics_df.to_string(index=False))

# Compare to CatBoost on the same 50 samples
print("\n[6] CatBoost on the same 50 samples ...")
cb_full_proba = np.load(CACHE / "cb_full53_test_proba.npy")
cb_sample = cb_full_proba[sample_idx]
y_sample = y_test[sample_idx]

# Use thr=0.5 and the cost-optimal threshold on the FULL test for fair comparison
from mark_phase4_tuning import cost_sweep, evaluate_at_threshold
amt_test = test_df["amt"].values
cs = cost_sweep(cb_full_proba, y_test, amt_test)
thr_opt = float(cs.loc[cs["expected_cost"].idxmin(), "threshold"])

cb_rows = []
for thr_name, thr in [("thr=0.5", 0.5), (f"thr={thr_opt:.3f} (cost-opt)", thr_opt)]:
    pred = (cb_sample >= thr).astype(int)
    tp = int(((pred == 1) & (y_sample == 1)).sum())
    fp = int(((pred == 1) & (y_sample == 0)).sum())
    tn = int(((pred == 0) & (y_sample == 0)).sum())
    fn = int(((pred == 0) & (y_sample == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    cb_rows.append(dict(model=f"CatBoost-53f ({thr_name})",
                         n=len(y_sample), tp=tp, fp=fp, tn=tn, fn=fn,
                         accuracy=acc, precision=prec, recall=rec, f1=f1,
                         latency_ms_estimate=0.10))  # ~0.1ms per row

# Re-evaluate the LLM rows on this 50-sample to align (they may already be there)
all_rows2 = []
for r in cb_rows:
    all_rows2.append(r)
for _, r in metrics_df.iterrows():
    all_rows2.append(dict(
        model=f"{r['llm']}/{r['model']}",
        n=int(r['n']),
        tp=int(r['tp']), fp=int(r['fp']), tn=int(r['tn']), fn=int(r['fn']),
        accuracy=r['accuracy'], precision=r['precision'],
        recall=r['recall'], f1=r['f1'],
        latency_ms_estimate=float(r['latency_mean_s']) * 1000.0,
    ))
final_df = pd.DataFrame(all_rows2)
final_df.to_csv(CACHE / "llm_vs_catboost_final.csv", index=False)
print("\n=== FINAL HEAD-TO-HEAD on 50-sample test ===")
print(final_df.to_string(index=False))

# Cost-per-1k-prediction estimates (per Anthropic and OpenAI public 2026 pricing).
#   Claude Haiku 4.5  ~ $1/MTok in, $5/MTok out. ~250in/10out per call  ~ $0.0003
#   Claude Opus 4.6   ~ $15/MTok in, $75/MTok out. ~250in/10out per call ~ $0.0045
#   GPT-5.4 (codex)   ~ $20/MTok in, $80/MTok out, plus codex agent overhead
#                     (~25k tokens/call observed earlier) -> ~ $0.05/call
#   CatBoost          ~ free at inference (single CPU core, ~0.1ms/row)
COST_PER_CALL_USD = {
    "claude/haiku": 0.0003,
    "claude/opus": 0.0045,
    "codex/gpt-5.4": 0.05,
    "CatBoost-53f (thr=0.5)": 1e-7,
    f"CatBoost-53f (thr={thr_opt:.3f} (cost-opt))": 1e-7,
}
final_df["cost_per_1k_usd"] = final_df["model"].map(
    lambda m: COST_PER_CALL_USD.get(m, 0) * 1000
)
final_df.to_csv(CACHE / "llm_vs_catboost_final.csv", index=False)
print("\nWith cost/1k:")
print(final_df[["model", "n", "accuracy", "precision", "recall", "f1",
                  "latency_ms_estimate", "cost_per_1k_usd"]].to_string(index=False))

print("\nDone.")
