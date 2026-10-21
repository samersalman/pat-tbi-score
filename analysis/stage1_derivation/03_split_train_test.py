"""
PAT-TBI — Step 3: Stratified 2/3-1/3 train/test split
================================================================================

Reads the Step 2 candidate matrix and produces a stratified (by
received_platelets_4h) 2/3 derivation / 1/3 validation split using
sklearn.train_test_split with random_state=42. Writes two lightweight index
files (PUF_YEAR, inc_key) and a small balance table for Table 1a / methods.

Cohort years: 2020-2024.

Inputs
------
  PAT_TBI/v4/derived/candidates.parquet            (Step 2 output)

Outputs
-------
  PAT_TBI/v4/derived/train_inc_keys.csv            (columns: PUF_YEAR, inc_key)
  PAT_TBI/v4/derived/test_inc_keys.csv             (columns: PUF_YEAR, inc_key)
  PAT_TBI/v4/tables/table_split_balance.csv        (columns: split, N, N_events,
                                                     event_rate)

Output contract for downstream scripts (Step 4, Step 6, Step 10)
----------------------------------------------------------------
  * Split identity is stored as (PUF_YEAR, inc_key) pairs, NOT as row indices
    of the candidates parquet. Both columns together are the join key because
    inc_key alone is not guaranteed unique across PUF years.
  * Downstream usage pattern:
        cands = pd.read_parquet(CAND_PARQUET)
        train_keys = pd.read_csv(TRAIN_KEYS_CSV)
        train = cands.merge(train_keys, on=["PUF_YEAR", "inc_key"], how="inner")
  * train ∪ test == full candidates cohort; train ∩ test == ∅.
  * Stratified on received_platelets_4h so train/test platelet receipt rates
    match within 0.05 pp.
  * balance CSV columns: split, N, N_events, event_rate
      where event_rate is a float in [0, 1] (not a percentage).

Seed: 42
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)

import os as _os, sys as _sys  # path handling: make analysis/config.py importable
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import config  # noqa: E402  (analysis/config.py: PAT_TBI_WORKDIR, PAT_TBI_NTDB_DUCKDB)
PAT = str(config.V4_DIR)
DERIVED = os.path.join(PAT, "derived")
TABLES = os.path.join(PAT, "tables")
os.makedirs(DERIVED, exist_ok=True)
os.makedirs(TABLES, exist_ok=True)

CAND_PARQUET = os.path.join(DERIVED, "candidates.parquet")
TRAIN_KEYS_CSV = os.path.join(DERIVED, "train_inc_keys.csv")
TEST_KEYS_CSV = os.path.join(DERIVED, "test_inc_keys.csv")
BALANCE_CSV = os.path.join(TABLES, "table_split_balance.csv")

OUTCOME_COL = "received_platelets_4h"

TEST_FRAC = 1.0 / 3.0
EVENT_RATE_DIFF_TOL = 0.0005  # half of 0.1 percentage point


def banner(msg):
    line = "=" * 78
    print(f"\n{line}\n{msg}\n{line}", flush=True)


def summarize(name, df):
    n = len(df)
    n_events = int(df[OUTCOME_COL].sum())
    rate = n_events / n if n else float("nan")
    print(f"  {name:<6s}  N={n:>9,}   events={n_events:>7,}   rate={rate:7.5f}",
          flush=True)
    return n, n_events, rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0,
                        help="Optional: pass-through smoke-test flag. This "
                             "script does NOT resample; it just reads whatever "
                             "Step 2 wrote. Flag exists so the pipeline driver "
                             "can forward --sample uniformly.")
    args = parser.parse_args()

    banner("Step 3 — Stratified 2/3-1/3 train/test split (seed 42, 2020-2024)")
    if args.sample:
        print(f"  [SMOKE TEST pass-through] --sample={args.sample:,} "
              f"(no downsampling here; relies on Step 2 output)", flush=True)

    # -----------------------------------------------------------------
    # Load candidates — only the columns we need
    # -----------------------------------------------------------------
    if not os.path.exists(CAND_PARQUET):
        sys.exit(f"ERROR: candidates parquet not found: {CAND_PARQUET}\n"
                 f"Run Step 2 (02_engineer_features.py) first.")

    print(f"Reading: {CAND_PARQUET}", flush=True)
    df = pd.read_parquet(CAND_PARQUET,
                         columns=["PUF_YEAR", "inc_key", OUTCOME_COL])
    print(f"  candidates: {len(df):,} rows", flush=True)

    # Type coercion
    df["PUF_YEAR"] = pd.to_numeric(df["PUF_YEAR"], errors="coerce").astype("Int64")
    df["inc_key"] = pd.to_numeric(df["inc_key"], errors="coerce").astype("Int64")
    df[OUTCOME_COL] = pd.to_numeric(df[OUTCOME_COL], errors="coerce")

    # Drop any rows with null outcome / key — stratified split cannot stratify NaN
    n_raw = len(df)
    df = df.dropna(subset=["PUF_YEAR", "inc_key", OUTCOME_COL]).copy()
    df[OUTCOME_COL] = df[OUTCOME_COL].astype(int)
    if len(df) < n_raw:
        print(f"  dropped {n_raw - len(df):,} rows with null "
              f"PUF_YEAR/inc_key/{OUTCOME_COL}", flush=True)

    # Sanity: no duplicate (PUF_YEAR, inc_key) in full cohort
    dup_mask = df.duplicated(subset=["PUF_YEAR", "inc_key"], keep=False)
    if dup_mask.any():
        n_dup = int(dup_mask.sum())
        sys.exit(f"ERROR: {n_dup:,} duplicate (PUF_YEAR, inc_key) pairs in "
                 f"candidates parquet — cannot produce a clean split.")

    df = df.reset_index(drop=True)

    # -----------------------------------------------------------------
    # Stratified split
    # -----------------------------------------------------------------
    banner(f"Running stratified split (test_size=1/3, stratify={OUTCOME_COL})")
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=TEST_FRAC,
        stratify=df[OUTCOME_COL],
        random_state=SEED,
    )

    train = df.loc[train_idx].copy()
    test = df.loc[test_idx].copy()

    # -----------------------------------------------------------------
    # Summaries
    # -----------------------------------------------------------------
    banner("Split balance")
    n_full, d_full, r_full = summarize("full", df)
    n_tr,   d_tr,   r_tr   = summarize("train", train)
    n_te,   d_te,   r_te   = summarize("test", test)

    # -----------------------------------------------------------------
    # Hard assertions
    # -----------------------------------------------------------------
    banner("Assertions")

    # 1) Partition integrity (on (PUF_YEAR, inc_key) tuples)
    full_pairs = set(map(tuple, df[["PUF_YEAR", "inc_key"]].astype("int64").values))
    train_pairs = set(map(tuple, train[["PUF_YEAR", "inc_key"]].astype("int64").values))
    test_pairs = set(map(tuple, test[["PUF_YEAR", "inc_key"]].astype("int64").values))

    assert len(train_pairs) == len(train), \
        f"duplicate (PUF_YEAR, inc_key) inside train: {len(train) - len(train_pairs):,}"
    assert len(test_pairs) == len(test), \
        f"duplicate (PUF_YEAR, inc_key) inside test: {len(test) - len(test_pairs):,}"
    assert train_pairs.isdisjoint(test_pairs), \
        "train ∩ test is non-empty — split leakage"
    assert train_pairs | test_pairs == full_pairs, \
        "train ∪ test != full cohort"
    print("  [OK] partition: train ∪ test == full, train ∩ test == ∅", flush=True)
    print("  [OK] no duplicate (PUF_YEAR, inc_key) inside train or test",
          flush=True)

    # 2) Stratification: |train_rate - test_rate| < 0.0005
    diff = abs(r_tr - r_te)
    assert diff < EVENT_RATE_DIFF_TOL, (
        f"event-rate gap between train and test ({diff:.6f}) "
        f"exceeds tolerance {EVENT_RATE_DIFF_TOL}"
    )
    print(f"  [OK] |train_event_rate - test_event_rate| = {diff:.6f} "
          f"< {EVENT_RATE_DIFF_TOL}", flush=True)

    # -----------------------------------------------------------------
    # Write key files
    # -----------------------------------------------------------------
    banner("Writing outputs")

    train_out = train[["PUF_YEAR", "inc_key"]].sort_values(
        ["PUF_YEAR", "inc_key"]).reset_index(drop=True)
    test_out = test[["PUF_YEAR", "inc_key"]].sort_values(
        ["PUF_YEAR", "inc_key"]).reset_index(drop=True)

    train_out.to_csv(TRAIN_KEYS_CSV, index=False)
    test_out.to_csv(TEST_KEYS_CSV, index=False)
    print(f"  wrote {TRAIN_KEYS_CSV}  ({len(train_out):,} rows)", flush=True)
    print(f"  wrote {TEST_KEYS_CSV}   ({len(test_out):,} rows)", flush=True)

    balance = pd.DataFrame([
        {"split": "full",  "N": n_full, "N_events": d_full, "event_rate": r_full},
        {"split": "train", "N": n_tr,   "N_events": d_tr,   "event_rate": r_tr},
        {"split": "test",  "N": n_te,   "N_events": d_te,   "event_rate": r_te},
    ])
    balance.to_csv(BALANCE_CSV, index=False)
    print(f"  wrote {BALANCE_CSV}", flush=True)

    print("\nStep 3 complete.")


if __name__ == "__main__":
    main()
