"""One-shot script to build the full Phase 3 (Anthony + Mark) feature dataset
and persist to data/processed/mark_phase3_full.parquet so the notebook can
load it instantly and iterate on models / encodings / ablations."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.mark_phase3_features import build_full_phase3_dataset  # noqa: E402


def main() -> None:
    raw = ROOT / "data" / "raw" / "fraud_transactions.csv"
    out = ROOT / "data" / "processed" / "mark_phase3_full.parquet"

    if out.exists():
        print(f"[skip] {out} already exists ({out.stat().st_size / 1e6:.1f} MB)")
        return

    t0 = time.time()
    df = build_full_phase3_dataset(raw)
    elapsed = time.time() - t0

    print(f"\nBuilt {len(df):,} rows x {df.shape[1]} cols in {elapsed/60:.1f} min")
    print(f"Saving to {out}")

    keep_cols = [c for c in df.columns if c not in (
        "first", "last", "street", "trans_num", "Unnamed: 0",
    )]
    df[keep_cols].to_parquet(out, index=False, compression="snappy")
    print(f"Saved {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
