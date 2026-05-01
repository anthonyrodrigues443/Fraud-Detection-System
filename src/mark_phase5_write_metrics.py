"""Append the `mark_phase5` block to results/metrics.json."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

CACHE = Path("results/mark_phase5_cache")
METRICS = Path("results/metrics.json")

ml = json.load(open(CACHE / "phase5_ml_summary.json"))
stack_df = pd.read_csv(CACHE / "stacking.csv")
calib_df = pd.read_csv(CACHE / "calibration.csv")
ablate_df = pd.read_csv(CACHE / "group_ablation.csv")
llm_metrics_df = pd.read_csv(CACHE / "llm_metrics.csv")
llm_final_df = pd.read_csv(CACHE / "llm_vs_catboost_final.csv")

mark_phase5 = {
    "date": "2026-05-01",
    "phase": 5,
    "researcher": "Mark Rodrigues",
    "description": (
        "Group-level ablation + ML stacking + probability calibration + "
        "Claude Haiku 4.5 / Opus 4.6 LLM frontier comparison on the 53-feat clean stack. "
        "Codex/GPT-5.4 usage-limited (retry 2026-05-06)."
    ),
    "data_split": {
        "n_train": int(ml["n_train"]),
        "n_test": int(ml["n_test"]),
        "n_fit_for_stacking": int(ml["n_fit"]),
        "n_calibration_slice": int(ml["n_cal"]),
        "fraud_rates": {"train": 0.005795, "test": 0.005460, "calibration": 0.005108},
    },
    "baseline_full_train": {
        "model": "Default CatBoost on 53-feat clean stack (full 838860 train)",
        "auprc": float(ml["baseline_full53_test"]["auprc"]),
        "auroc": float(ml["baseline_full53_test"]["auroc"]),
        "prec_at_95rec": float(ml["baseline_full53_test"]["prec_at_95rec"]),
        "min_expected_cost": 2088.41,
    },
    "group_ablation": {
        "method": "Drop one feature group, retrain default CatBoost on full 838k train, evaluate on 209k test",
        "groups": list(ml["group_ablation"]),
        "load_bearing_groups": ["Velocity (8)", "Baseline (17)"],
        "redundant_groups": [
            "Mark-stat (14)", "Geographic (2)", "AmountDev (5)",
            "Category (4)", "Temporal (3)",
        ],
        "headline": (
            "Velocity (-0.052 AUPRC) and Baseline-17 (-0.019 AUPRC) are the only load-bearing "
            "groups; Mark-stat-14 add-ons are essentially redundant (+0.001 AUPRC)."
        ),
    },
    "stacking_ensemble": {
        "method": "Fit base learners on first 85% of train; train LogReg meta on last 15% of train; evaluate on test",
        "rows": list(ml["stacking"]),
        "logreg_meta_coefs": {
            "CatBoost": float(ml["stacking_meta"]["meta_coefs"][0]),
            "XGBoost": float(ml["stacking_meta"]["meta_coefs"][1]),
            "LightGBM": float(ml["stacking_meta"]["meta_coefs"][2]),
            "intercept": float(ml["stacking_meta"]["meta_intercept"]),
        },
        "best_combiner": "Simple-average (CB+XGB+LGB)/3",
        "best_combiner_auprc": float(stack_df.loc[stack_df["model"].str.contains("Simple-average"), "auprc"].iloc[0]),
        "best_combiner_min_cost": float(stack_df.loc[stack_df["model"].str.contains("Simple-average"), "min_expected_cost"].iloc[0]),
        "best_combiner_f1_at_thr05": float(stack_df.loc[stack_df["model"].str.contains("Simple-average"), "f1_at_thr05"].iloc[0]),
        "logreg_stack_underperforms": True,
        "headline": (
            "Simple uniform average of CB+XGB+LGB beats every single learner AND the LogReg-stacked "
            "combination. AUPRC=0.9817, min cost=$1,844. LogReg meta overfit (CB coef=21.6, LGB coef=-2.6)."
        ),
    },
    "calibration": {
        "method": "Fit isotonic + Platt on calibration-slice probabilities, apply to test",
        "rows": list(ml["calibration"]),
        "preserves_auprc_platt_yes_isotonic_no": True,
        "improves_brier_pct": -37.5,
        "improves_ece_pct": -89.0,
        "improves_f1_at_thr05": True,
        "f1_at_thr05_uncal": float(calib_df.loc[calib_df["method"] == "Uncalibrated", "f1_at_thr05"].iloc[0]),
        "f1_at_thr05_platt": float(calib_df.loc[calib_df["method"] == "Platt(sigmoid)", "f1_at_thr05"].iloc[0]),
        "increases_min_cost": True,
        "min_cost_uncal": float(calib_df.loc[calib_df["method"] == "Uncalibrated", "min_expected_cost"].iloc[0]),
        "min_cost_platt": float(calib_df.loc[calib_df["method"] == "Platt(sigmoid)", "min_expected_cost"].iloc[0]),
        "headline": (
            "Brier -37%, ECE -89%, F1@thr=0.5 +3pp under Platt -- but min expected cost INCREASES "
            "by $80 (calibration is deployment-ergonomics, not a cost lever)."
        ),
    },
    "llm_frontier_comparison": {
        "method": "Stratified 50-sample test (25 fraud + 25 legit, random_state=42); each LLM gets 17-feature description; replies FRAUD/LEGIT + probability",
        "sample_size": 50,
        "rows_per_llm": list(llm_metrics_df.to_dict(orient="records")),
        "head_to_head_with_catboost": list(llm_final_df.to_dict(orient="records")),
        "codex_gpt54_status": {
            "available": False,
            "error_at_2026-05-01_07:18Z": (
                "ERROR: You've hit your usage limit. Upgrade to Plus to continue using Codex "
                "or try again at May 6th, 2026 5:31 PM."
            ),
            "retry_eligible_after": "2026-05-06",
        },
        "headline": (
            "CatBoost F1=1.000 on 50-sample. Opus F1=0.864 (-13.6pp), 242,000x slower, 45,000x more $. "
            "Haiku F1=0.485 (-51.5pp), 129,000x slower, 3,000x more $. Specialist beats frontier LLM "
            "on every measurable axis."
        ),
    },
    "production_recommendation": {
        "deploy_model": "Simple-average ensemble of CatBoost + XGBoost + LightGBM (full train)",
        "deploy_threshold": "cost-optimal (~0.05 on uncalibrated probs)",
        "expected_auprc": 0.9817,
        "expected_min_cost_per_209k_test": 1844.33,
        "feature_set": "53-feat clean stack OR Anthony 39-feat set (group ablation shows -0/+0 cost)",
        "vs_phase4_single_catboost_cost": -264.0,
        "vs_phase4_single_catboost_pct": -12.5,
    },
    "headline": (
        "Three counterintuitive Phase 5 findings: (1) on a saturated CatBoost the simple uniform "
        "average of 3 boosters beats LogReg-stacking AND every single learner (no meta needed); "
        "(2) Mark's own 14 statistical add-ons are redundant -- 53-feat stack can be pruned to "
        "Anthony's 39-feat set with zero measurable loss; (3) probability calibration improves "
        "F1@0.5 by +3pp but INCREASES expected dollar cost by $80 (deployment ergonomics, not a "
        "cost win). Plus: CatBoost beats Claude Opus 4.6 by F1=1.00 vs 0.864 on a 50-sample "
        "stratified test, while running 242,000x faster and 45,000x cheaper per prediction."
    ),
}


def append_or_update(metrics: dict, key: str, value):
    metrics[key] = value


# Load existing metrics.json
metrics = json.load(open(METRICS))
metrics["mark_phase5"] = mark_phase5

# Pretty-print and write atomically
out = json.dumps(metrics, indent=2, default=str)
METRICS.write_text(out, encoding="utf-8")
print(f"Wrote mark_phase5 block to {METRICS} ({len(out):,} bytes total)")
