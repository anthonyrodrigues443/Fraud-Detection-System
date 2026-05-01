"""
Phase 5 (Mark) -- Advanced techniques + ablation + LLM frontier comparison
on the 53-feature clean stack.

Anthony's Phase 5 (merged) covered: SHAP, Isolation Forest hybrid (zero signal),
per-category thresholds, single-feature ablation. He explicitly DEFERRED the
LLM frontier comparison.

Mark's complementary angle:
  1. Real ML stacking ensemble (CatBoost + XGBoost + LightGBM + LogReg meta)
     trained leak-free on the calibration slice. Counter-test: if Anthony's
     IsoForest hybrid found zero, can a true ML stack add anything?
  2. GROUP-level ablation (vs Anthony's single-feature ablation): drop entire
     feature families (Velocity, Amount-Dev, Temporal, Geographic, Category,
     Mark-stat) one at a time. Quantify each group's marginal AUPRC/F1/cost.
  3. Probability calibration -- isotonic vs Platt(sigmoid) on top of CatBoost
     -- evaluated on AUPRC (preserved by both), Brier score, ECE, and the
     SHIFT in cost-optimal threshold. Does well-calibrated probability change
     the production deployment recipe?
  4. LLM frontier head-to-head -- 50 test transactions sent to Claude Opus 4.6
     (via `claude --print`) and GPT-5.4 (via `codex exec`). Compare to the
     CatBoost champion on accuracy, recall, latency, $cost-per-prediction.

All experiments leak-free under the temporal split established in Phase 1.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

# Re-use Phase 4 feature definitions and helpers
from mark_phase4_tuning import (
    ANTHONY_BASELINE,
    ANTHONY_NEW,
    CLEAN_STACK_53,
    DEFAULT_CB_PARAMS,
    FREQ_FEATURES,
    MARK_NON_TE,
    cost_sweep,
    evaluate_at_threshold,
    fit_catboost,
    find_threshold_at_recall,
    load_phase4_data,
    metric_summary,
    temporal_calibration_split,
)

# ---------------------------------------------------------------------------- #
# Feature group definitions (for group-ablation)
# ---------------------------------------------------------------------------- #
VELOCITY_GROUP = [
    "vel_count_1h", "vel_count_6h", "vel_count_24h", "vel_count_7d",
    "vel_amt_1h", "vel_amt_6h", "vel_amt_24h", "vel_amt_7d",
]
AMOUNT_DEV_GROUP = [
    "amt_zscore", "amt_ratio_to_mean", "amt_card_mean", "amt_card_std",
    "amt_cat_zscore",
]
TEMPORAL_GROUP = ["log_time_since_last", "log_avg_time_between", "hour_deviation"]
GEO_GROUP = ["log_dist_centroid", "impossible_travel"]
CATEGORY_GROUP = [
    "cat_fraud_rate", "card_cat_count", "is_new_merchant", "card_txn_number",
]
MARK_STAT_GROUP = MARK_NON_TE + FREQ_FEATURES  # Mark's 14 add-ons

GROUPS = {
    "Velocity (8)": VELOCITY_GROUP,
    "AmountDev (5)": AMOUNT_DEV_GROUP,
    "Temporal (3)": TEMPORAL_GROUP,
    "Geographic (2)": GEO_GROUP,
    "Category (4)": CATEGORY_GROUP,
    "Mark-stat (14)": MARK_STAT_GROUP,
    "Baseline (17)": ANTHONY_BASELINE,
}


# ---------------------------------------------------------------------------- #
# 1. Group-level ablation
# ---------------------------------------------------------------------------- #
def run_group_ablation(X_train: pd.DataFrame, y_train: np.ndarray,
                        X_test: pd.DataFrame, y_test: np.ndarray,
                        amt_test: np.ndarray, baseline_proba: np.ndarray,
                        cache_dir: Path) -> pd.DataFrame:
    """Drop each group in turn, retrain CatBoost defaults on the remaining
    feature set, and report ΔAUPRC + Δcost vs the full-stack baseline."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    full_summary = metric_summary(baseline_proba, y_test, "full_53")
    base_cost_df = cost_sweep(baseline_proba, y_test, amt_test)
    base_cost = float(base_cost_df["expected_cost"].min())

    rows = [{
        "drop_group": "(none) full 53f",
        "n_features": len(CLEAN_STACK_53),
        "auprc": full_summary["auprc"],
        "auroc": full_summary["auroc"],
        "prec_at_95rec": full_summary["prec_at_95rec"],
        "delta_auprc": 0.0,
        "min_expected_cost": base_cost,
        "delta_cost": 0.0,
    }]

    for group_name, group_cols in GROUPS.items():
        cache_file = cache_dir / f"ablation_{group_name.split()[0].lower()}.npy"
        kept = [c for c in CLEAN_STACK_53 if c not in group_cols]
        if cache_file.exists():
            proba = np.load(cache_file)
        else:
            t0 = time.time()
            mdl = fit_catboost(X_train[kept], y_train)
            proba = mdl.predict_proba(X_test[kept])[:, 1].astype(np.float32)
            np.save(cache_file, proba)
            print(f"  {group_name}: trained {len(kept)}f in {time.time()-t0:.1f}s")
        s = metric_summary(proba, y_test, f"drop_{group_name}")
        cs = cost_sweep(proba, y_test, amt_test)
        rows.append({
            "drop_group": group_name,
            "n_features": len(kept),
            "auprc": s["auprc"],
            "auroc": s["auroc"],
            "prec_at_95rec": s["prec_at_95rec"],
            "delta_auprc": s["auprc"] - full_summary["auprc"],
            "min_expected_cost": float(cs["expected_cost"].min()),
            "delta_cost": float(cs["expected_cost"].min()) - base_cost,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- #
# 2. Real ML stacking ensemble: CatBoost + XGBoost + LightGBM + LogReg meta
# ---------------------------------------------------------------------------- #
def fit_xgb(X, y, **kwargs):
    from xgboost import XGBClassifier
    pos = float((y == 0).sum() / max((y == 1).sum(), 1))
    p = dict(n_estimators=400, max_depth=6, learning_rate=0.1,
             eval_metric="aucpr", scale_pos_weight=pos,
             tree_method="hist", n_jobs=-1, random_state=42, verbosity=0)
    p.update(kwargs)
    m = XGBClassifier(**p)
    m.fit(X, y, verbose=False)
    return m


def fit_lgb(X, y, **kwargs):
    from lightgbm import LGBMClassifier
    pos = float((y == 0).sum() / max((y == 1).sum(), 1))
    p = dict(n_estimators=400, max_depth=-1, num_leaves=63, learning_rate=0.05,
             objective="binary", scale_pos_weight=pos,
             n_jobs=-1, random_state=42, verbosity=-1)
    p.update(kwargs)
    m = LGBMClassifier(**p)
    m.fit(X, y)
    return m


def stacking_pipeline(Xfit, yfit, Xcal, ycal, Xtest, ytest, cache_dir: Path):
    """Train 3 base learners on Xfit, get OOC predictions on Xcal (held-out
    last 15% of train), train a logistic meta-learner on those Xcal predictions,
    then evaluate on Xtest."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cb_cal_p = cache_dir / "cb_cal.npy"
    cb_test_p = cache_dir / "cb_test.npy"
    xgb_cal_p = cache_dir / "xgb_cal.npy"
    xgb_test_p = cache_dir / "xgb_test.npy"
    lgb_cal_p = cache_dir / "lgb_cal.npy"
    lgb_test_p = cache_dir / "lgb_test.npy"

    if cb_cal_p.exists():
        cb_cal = np.load(cb_cal_p); cb_test = np.load(cb_test_p)
    else:
        t = time.time()
        cb = fit_catboost(Xfit, yfit)
        cb_cal = cb.predict_proba(Xcal)[:, 1].astype(np.float32)
        cb_test = cb.predict_proba(Xtest)[:, 1].astype(np.float32)
        np.save(cb_cal_p, cb_cal); np.save(cb_test_p, cb_test)
        print(f"  CatBoost: {time.time()-t:.1f}s")

    if xgb_cal_p.exists():
        xgb_cal = np.load(xgb_cal_p); xgb_test = np.load(xgb_test_p)
    else:
        t = time.time()
        xgb = fit_xgb(Xfit, yfit)
        xgb_cal = xgb.predict_proba(Xcal)[:, 1].astype(np.float32)
        xgb_test = xgb.predict_proba(Xtest)[:, 1].astype(np.float32)
        np.save(xgb_cal_p, xgb_cal); np.save(xgb_test_p, xgb_test)
        print(f"  XGBoost: {time.time()-t:.1f}s")

    if lgb_cal_p.exists():
        lgb_cal = np.load(lgb_cal_p); lgb_test = np.load(lgb_test_p)
    else:
        t = time.time()
        lgb = fit_lgb(Xfit, yfit)
        lgb_cal = lgb.predict_proba(Xcal)[:, 1].astype(np.float32)
        lgb_test = lgb.predict_proba(Xtest)[:, 1].astype(np.float32)
        np.save(lgb_cal_p, lgb_cal); np.save(lgb_test_p, lgb_test)
        print(f"  LightGBM: {time.time()-t:.1f}s")

    Z_cal = np.column_stack([cb_cal, xgb_cal, lgb_cal])
    Z_test = np.column_stack([cb_test, xgb_test, lgb_test])

    meta = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    meta.fit(Z_cal, ycal)
    stack_proba = meta.predict_proba(Z_test)[:, 1].astype(np.float32)

    # Simple-average ensemble for comparison
    avg_proba = Z_test.mean(axis=1).astype(np.float32)

    return dict(
        cb_test=cb_test, xgb_test=xgb_test, lgb_test=lgb_test,
        stack_test=stack_proba, avg_test=avg_proba,
        meta_coefs=meta.coef_[0].tolist(), meta_intercept=float(meta.intercept_[0]),
    )


# ---------------------------------------------------------------------------- #
# 3. Probability calibration (isotonic vs Platt)
# ---------------------------------------------------------------------------- #
def expected_calibration_error(y_true, y_prob, n_bins=20):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    ece = 0.0
    n = len(y_prob)
    bin_data = []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        avg_p = float(y_prob[mask].mean())
        emp = float(y_true[mask].mean())
        w = mask.sum() / n
        ece += w * abs(avg_p - emp)
        bin_data.append({"bin_low": float(bins[b]), "bin_high": float(bins[b+1]),
                          "n": int(mask.sum()), "avg_pred": avg_p,
                          "frac_pos": emp, "weight": float(w)})
    return float(ece), bin_data


def fit_isotonic(p_cal: np.ndarray, y_cal: np.ndarray):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_cal, y_cal)
    return iso


def fit_platt(p_cal: np.ndarray, y_cal: np.ndarray):
    """Platt scaling = LR on logit(p_cal). Use simple sklearn LR."""
    from scipy.special import logit
    clip = np.clip(p_cal, 1e-6, 1 - 1e-6)
    lr = LogisticRegression()
    lr.fit(logit(clip).reshape(-1, 1), y_cal)
    return lr


def apply_platt(model, p_test: np.ndarray) -> np.ndarray:
    from scipy.special import logit
    clip = np.clip(p_test, 1e-6, 1 - 1e-6)
    return model.predict_proba(logit(clip).reshape(-1, 1))[:, 1]


def calibration_report(p_cal_uncal, y_cal, p_test_uncal, y_test, amt_test):
    """Fit isotonic + Platt on calibration slice, evaluate on test."""
    iso = fit_isotonic(p_cal_uncal, y_cal)
    pl = fit_platt(p_cal_uncal, y_cal)
    p_test_iso = iso.predict(p_test_uncal).astype(np.float32)
    p_test_pl = apply_platt(pl, p_test_uncal).astype(np.float32)

    rows = []
    for name, p in [("Uncalibrated", p_test_uncal),
                     ("Isotonic", p_test_iso),
                     ("Platt(sigmoid)", p_test_pl)]:
        ece, _ = expected_calibration_error(y_test, p, n_bins=20)
        cs = cost_sweep(p, y_test, amt_test)
        cost_min = float(cs["expected_cost"].min())
        thr_min = float(cs.loc[cs["expected_cost"].idxmin(), "threshold"])
        thr_05 = 0.5
        op_05 = evaluate_at_threshold(p, y_test, thr_05, "thr=0.5")
        rows.append(dict(
            method=name,
            auprc=float(average_precision_score(y_test, p)),
            auroc=float(roc_auc_score(y_test, p)),
            brier=float(brier_score_loss(y_test, p)),
            ece=ece,
            cost_optimal_threshold=thr_min,
            min_expected_cost=cost_min,
            f1_at_thr05=op_05.realized_f1,
            recall_at_thr05=op_05.realized_recall,
            precision_at_thr05=op_05.realized_precision,
        ))
    return pd.DataFrame(rows), p_test_iso, p_test_pl


# ---------------------------------------------------------------------------- #
# 4. LLM frontier head-to-head
# ---------------------------------------------------------------------------- #
LLM_PROMPT_TEMPLATE = (
    "You are a credit card fraud analyst. Reply with EXACTLY one word: FRAUD or LEGIT. "
    "Then on a new line: a number 0.0-1.0 for fraud probability. No explanation.\n\n"
    "Transaction features:\n{features}"
)


def format_transaction_for_llm(row: dict) -> str:
    """Build a compact transaction description from the most informative features."""
    feats = []
    feats.append(f"amount=${row['amt']:.2f}")
    feats.append(f"hour={int(row['hour'])}")
    feats.append(f"is_night={int(row['is_night'])}")
    feats.append(f"category={row.get('category','?')}")
    feats.append(f"category_fraud_rate={row.get('cat_fraud_rate', 0):.4f}")
    feats.append(f"distance_km={row['distance_km']:.0f}")
    feats.append(f"impossible_travel={int(row['impossible_travel'])}")
    feats.append(f"vel_count_1h={int(row['vel_count_1h'])}")
    feats.append(f"vel_count_24h={int(row['vel_count_24h'])}")
    feats.append(f"vel_amt_24h=${row['vel_amt_24h']:.2f}")
    feats.append(f"amt_zscore={row['amt_zscore']:.2f}")
    feats.append(f"amt_cat_zscore={row['amt_cat_zscore']:.2f}")
    feats.append(f"amt_ratio_to_mean={row['amt_ratio_to_mean']:.2f}")
    feats.append(f"is_new_merchant={int(row['is_new_merchant'])}")
    feats.append(f"merch_fraud_rate={row.get('merch_fraud_rate', 0):.4f}")
    feats.append(f"age_years={row['age']:.0f}")
    feats.append(f"city_pop={int(row['city_pop'])}")
    return "\n".join("  - " + f for f in feats)


CLAUDE_CMD = r"C:\Users\antho\AppData\Roaming\npm\claude.cmd"
CODEX_CMD = r"C:\Users\antho\AppData\Roaming\npm\codex.cmd"


def call_claude(prompt: str, model: str = "haiku", timeout: float = 90.0) -> tuple[str, float]:
    """Call `claude --print --model <model>` (prompt via stdin) and return
    (text, latency_seconds). Returns ("__ERROR__:<reason>", elapsed) on failure."""
    t0 = time.time()
    try:
        proc = subprocess.run(
            [CLAUDE_CMD, "--print", "--model", model, "--no-session-persistence",
             "--disable-slash-commands"],
            input=prompt, capture_output=True, text=True, timeout=timeout,
        )
        elapsed = time.time() - t0
        if proc.returncode != 0:
            return f"__ERROR__:rc={proc.returncode}:{proc.stderr[:200]}", elapsed
        return proc.stdout.strip(), elapsed
    except subprocess.TimeoutExpired:
        return "__ERROR__:timeout", time.time() - t0
    except Exception as e:
        return f"__ERROR__:exc:{type(e).__name__}:{str(e)[:200]}", time.time() - t0


def call_codex(prompt: str, timeout: float = 180.0) -> tuple[str, float]:
    """Call `codex exec - --sandbox read-only` (prompt via stdin) and return
    (text, latency_seconds). Returns ("__ERROR__:<reason>", elapsed) on failure."""
    t0 = time.time()
    try:
        # `-` argument tells codex exec to read prompt from stdin
        proc = subprocess.run(
            [CODEX_CMD, "exec", "--skip-git-repo-check", "--sandbox", "read-only", "-"],
            input=prompt, capture_output=True, text=True, timeout=timeout,
        )
        elapsed = time.time() - t0
        if proc.returncode != 0:
            return f"__ERROR__:rc={proc.returncode}:{proc.stderr[:200]}", elapsed
        # codex output includes session metadata; pull the codex response section
        out = proc.stdout
        if "codex\n" in out:
            # Take everything after the last "codex\n" up to "tokens used"
            tail = out.rsplit("codex\n", 1)[1]
            if "tokens used" in tail:
                tail = tail.split("tokens used")[0]
            return tail.strip(), elapsed
        return out.strip(), elapsed
    except subprocess.TimeoutExpired:
        return "__ERROR__:timeout", time.time() - t0
    except Exception as e:
        return f"__ERROR__:exc:{type(e).__name__}:{str(e)[:200]}", time.time() - t0


def parse_llm_response(text: str) -> tuple[Optional[int], Optional[float]]:
    """Extract (label, probability) from LLM response.

    Returns (1, prob) for FRAUD, (0, prob) for LEGIT, (None, None) on parse error.
    """
    if not text or text.startswith("__ERROR__"):
        return None, None
    upper = text.upper()
    label = None
    if "FRAUD" in upper.split("\n")[0]:
        label = 1
    elif "LEGIT" in upper.split("\n")[0]:
        label = 0
    elif "FRAUD" in upper:
        label = 1
    elif "LEGIT" in upper:
        label = 0

    # Try to find a probability number
    prob = None
    import re
    for ln in text.split("\n"):
        m = re.search(r"\b(0?\.\d+|1\.0+|0\.0+|0|1)\b", ln.strip())
        if m:
            try:
                v = float(m.group(1))
                if 0.0 <= v <= 1.0:
                    prob = v
                    break
            except ValueError:
                pass
    if prob is None and label is not None:
        prob = 0.85 if label == 1 else 0.15
    return label, prob


def run_llm_eval(test_df: pd.DataFrame, sample_idx: np.ndarray,
                 cache_path: Path, llm: str = "claude",
                 model: str = "haiku") -> pd.DataFrame:
    """Evaluate LLM on a list of test-set indices. Cache to JSON."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        cached = json.load(open(cache_path))
    else:
        cached = []
    seen = {(c["llm"], c["model"], c["test_idx"]) for c in cached}

    out_rows = list(cached)
    for i, idx in enumerate(sample_idx):
        key = (llm, model, int(idx))
        if key in seen:
            continue
        row = test_df.iloc[idx].to_dict()
        prompt = LLM_PROMPT_TEMPLATE.format(features=format_transaction_for_llm(row))
        if llm == "claude":
            text, elapsed = call_claude(prompt, model=model)
        elif llm == "codex":
            text, elapsed = call_codex(prompt)
        else:
            raise ValueError(f"unknown llm: {llm}")
        label, prob = parse_llm_response(text)
        rec = dict(
            llm=llm, model=model, test_idx=int(idx),
            true_label=int(row["is_fraud"]),
            pred_label=label, pred_prob=prob,
            latency_s=elapsed, raw=text[:300] if text else "",
        )
        out_rows.append(rec)
        # Persist after every call so a partial run is recoverable
        json.dump(out_rows, open(cache_path, "w"), indent=2)
        if (i + 1) % 5 == 0:
            print(f"  [{llm}/{model}] {i+1}/{len(sample_idx)} done "
                  f"(last latency={elapsed:.1f}s)")
    return pd.DataFrame(out_rows)


def llm_metrics(df: pd.DataFrame, llm: str, model: str) -> dict:
    """Compute accuracy, precision, recall, F1, latency stats for an LLM run."""
    sub = df[(df["llm"] == llm) & (df["model"] == model) &
              df["pred_label"].notna()].copy()
    if len(sub) == 0:
        return dict(n=0)
    y_true = sub["true_label"].astype(int).values
    y_pred = sub["pred_label"].astype(int).values
    p_pred = sub["pred_prob"].astype(float).values
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    return dict(
        llm=llm, model=model, n=int(len(sub)),
        tp=tp, fp=fp, tn=tn, fn=fn,
        accuracy=float(acc), precision=float(prec), recall=float(rec), f1=float(f1),
        auprc=float(average_precision_score(y_true, p_pred)) if len(np.unique(y_true)) > 1 else None,
        latency_mean_s=float(sub["latency_s"].mean()),
        latency_median_s=float(sub["latency_s"].median()),
        latency_max_s=float(sub["latency_s"].max()),
    )
