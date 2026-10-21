"""
PAT-TBI NTDB 2020-2024 Refit — Step 6: Apply all scores to held-out TEST set
================================================================================

Applies five scores to the PAT-TBI held-out test slice:
  1. PAT-TBI          — refit PAT integer weights (Step 5)
  2. GCS              — inverted (risk = 15 - GCS)
  3. GCS-P            — Brennan 2018 (risk = 15 - (GCS - PRS))
  4. IMPACT-Core      — Steyerberg 2008 logistic probability
  5. CRASH-Basic HIC  — Perel 2008 14-day logistic probability

A helper compute_pat_tbi() accepts PAT-TBI's arbitrary-length weight table.

Inputs
------
  PAT_TBI/v4/derived/candidates.parquet     (Step 2 — all engineered vars)
  PAT_TBI/v4/derived/test_inc_keys.csv      (Step 3 — held-out test inc_keys)
  PAT_TBI/v4/tables/table_pat_weights.csv   (Step 5 — PAT integer weights)
  PAT_TBI/v4/derived/pat_refit.json         (Step 5 — refit metadata)

Output
------
  PAT_TBI/v4/derived/cohort_test_scored.parquet
    Columns:
      PUF_YEAR, inc_key, received_platelets_4h,
      AGE_NUMBER, SEX, RACE, ETHNICITY,
      iss_cont, max_ais_head, max_ais_other, major_extracranial,
      GCS2, MOTOR2, TQIP_TBI_PUPIL_RESPONSE,
      pat_tbi_score,    pat_tbi_score_mask,
      gcs_risk,         gcs_mask,
      gcs_p_risk,       gcs_p_mask,
      impact_core_prob, impact_mask,
      crash_basic_prob, crash_mask

Output contract for Steps 7/8/9/10
----------------------------------
  - All 5 score columns and their masks are ALWAYS present (no conditional cols).
  - `received_platelets_4h` is int 0/1 with no NaN.
  - Demographics and raw comparator inputs (GCS2, MOTOR2, pupils, AIS, age, etc.)
    pass through unchanged for subgroup / stratified analyses downstream.
  - `pat_tbi_score` is an integer-typed score (np.int64).
  - `gcs_risk` / `gcs_p_risk` are float risks (lower = better).
  - `impact_core_prob` / `crash_basic_prob` are float probabilities in [0,1];
    may be NaN where the row failed the score's own row-filter (mask=False).
  - `*_mask` columns are boolean; every score column is numeric where mask=True.
  - AIS sentinel 9 is replaced with NaN in `max_ais_head` / `max_ais_other`.
    `major_extracranial` stays as a pure 0/1 pass-through.

Hard assertions
---------------
  - Test-slice merge is inner-joined to test_inc_keys BEFORE scoring,
    and `len(merged) == len(test_inc_keys)`.
  - Every PAT-TBI-weight Variable must exist as a column in the candidate parquet.
  - `len(out) == len(test_inc_keys)` and `received_platelets_4h.isna().sum() == 0`.
  - PAT-TBI score column is integer-typed at write time.

Conventions
-----------
  SEED = 42
  Missing pupils -> 'both'   (PRS=0, null-biased)
  Missing GCS2   -> 15       (best, risk=0 for GCS/GCS-P)
  IMPACT: drop rows missing AGE_NUMBER or MOTOR2 not in 1..6
  CRASH:  drop rows missing AGE_NUMBER, GCS2, or major_extracranial
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Absolute paths
# ---------------------------------------------------------------------------
import os as _os, sys as _sys  # path handling: make analysis/config.py importable
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import config  # noqa: E402  (analysis/config.py: PAT_TBI_WORKDIR, PAT_TBI_NTDB_DUCKDB)
PAT = str(config.V4_DIR)
DERIVED = os.path.join(PAT, "derived")
TABLES = os.path.join(PAT, "tables")
os.makedirs(DERIVED, exist_ok=True)

CANDIDATES_PARQUET = os.path.join(DERIVED, "candidates.parquet")
TEST_KEYS_CSV = os.path.join(DERIVED, "test_inc_keys.csv")
PAT_WEIGHTS_CSV = os.path.join(TABLES, "table_pat_weights.csv")
PAT_REFIT_JSON = os.path.join(DERIVED, "pat_refit.json")
OUT_PARQUET = os.path.join(DERIVED, "cohort_test_scored.parquet")

OUTCOME_COL = "received_platelets_4h"


def banner(msg):
    line = "=" * 78
    print(f"\n{line}\n{msg}\n{line}", flush=True)


# ============================================================================
# Published coefficients
# ============================================================================
# IMPACT Core — Steyerberg 2008 PLoS Med
IMPACT_INTERCEPT = -3.109
IMPACT_AGE_BETA = 0.034
IMPACT_MOTOR_BETAS = {6: 0.000, 5: 0.000, 4: 0.390,
                     3: 0.797, 2: 1.397, 1: 1.447}
IMPACT_PUPIL_BETAS = {'both': 0.000, 'one': 0.514, 'none': 1.239}

# CRASH Basic HIC 14-day — Perel 2008 BMJ (LSHTM calculator JS coefficients)
CRASH_INTERCEPT = -4.3281
CRASH_AGE_BETA = 0.0542
CRASH_AGE_CENTER = 40.0
CRASH_GCS_BETA = -0.2099  # NEGATIVE
CRASH_PUPIL_BETAS = {'both': 0.000, 'one': 0.943, 'none': 1.712}
CRASH_EXTRACRANIAL_BETA = 0.4253

# GCS-P pupil reactivity score (Brennan 2018)
GCS_P_PRS = {'both': 0, 'one': 1, 'none': 2}


# ============================================================================
# Helpers
# ============================================================================
def _sigmoid(x):
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def _as_pupil(series):
    """Normalize TQIP_TBI_PUPIL_RESPONSE to {'both','one','none', <NA>}.

    NTDB uses 'NEITHER' instead of 'NONE' for "neither reactive" (both
    fixed pupils). We accept both encodings.
    """
    if series is None:
        return pd.Series([pd.NA] * 0, dtype=object)
    s = series.astype(str).str.strip().str.upper()
    mapping = {'BOTH': 'both', 'ONE': 'one',
               'NEITHER': 'none', 'NONE': 'none'}
    out = s.map(mapping)
    out[~s.isin(mapping.keys())] = pd.NA
    return out


# ============================================================================
# Primary score: PAT-TBI — arbitrary-length integer refit
# ============================================================================
def compute_pat_tbi(df, weights_csv):
    """Apply the PAT-TBI refit integer weights to df.

    The PAT-TBI weights CSV uses the published PLAT weight-table schema
    (Variable, Points, ...) but has an arbitrary number of rows. We:
      - load rows, drop blanks
      - coerce Points to int
      - assert every Variable is a column of df
      - sum Points * (binary var) across all predictors
      - return (score_int, mask_all_true, cov_dict, used_list)

    The PAT-TBI candidate parquet pre-builds all candidate binaries (Step 2),
    so every row is usable — mask is all True.
    """
    print(f"  Loading PAT-TBI REFIT weights from {weights_csv}")
    w = pd.read_csv(weights_csv)
    w = w.dropna(subset=['Variable'])
    w = w[w['Variable'].astype(str).str.strip() != ''].copy()
    w['Points'] = pd.to_numeric(w['Points'], errors='coerce').fillna(0).astype(int)
    # Drop zero-point rows defensively (they contribute nothing anyway)
    w = w[w['Points'] != 0].copy()
    print(f"  PAT-TBI has {len(w)} non-zero predictors")

    if len(w) == 0:
        raise RuntimeError("PAT-TBI weights table has zero usable predictors")

    score = np.zeros(len(df), dtype=np.float64)
    used = []
    for _, row in w.iterrows():
        var = str(row['Variable']).strip()
        pts = int(row['Points'])
        if var not in df.columns:
            raise RuntimeError(f"PAT-TBI predictor '{var}' missing from df")
        score += pts * df[var].astype(np.float64).values
        used.append((var, pts))

    pupil_raw = _as_pupil(df.get('TQIP_TBI_PUPIL_RESPONSE'))
    n_pupil_missing = int(pupil_raw.isna().sum())

    mask = np.ones(len(df), dtype=bool)  # pure binaries — all rows usable
    cov = {'N_used': int(mask.sum()),
           'N_dropped': int((~mask).sum()),
           'N_pupil_imputed': n_pupil_missing,
           'N_predictors': len(used)}
    return score.astype(np.int64), mask, cov, used


# ============================================================================
# Comparator 1: GCS total (inverted -> risk)
# ============================================================================
def compute_gcs_risk(df):
    gcs_raw = pd.to_numeric(df.get('GCS2'), errors='coerce')
    n_missing_gcs = int(gcs_raw.isna().sum())
    gcs_filled = gcs_raw.fillna(15).values
    risk = 15.0 - gcs_filled  # 0..12 typical, 0..15 worst-case
    mask = np.ones(len(df), dtype=bool)
    cov = {'N_used': int(mask.sum()),
           'N_dropped': int((~mask).sum()),
           'N_pupil_imputed': 0,
           'N_missing_gcs_imputed': n_missing_gcs}
    return risk, mask, cov


# ============================================================================
# Comparator 2: GCS-P (Brennan 2018, inverted -> risk)
# ============================================================================
def compute_gcs_p_risk(df):
    gcs_raw = pd.to_numeric(df.get('GCS2'), errors='coerce')
    gcs_filled = gcs_raw.fillna(15).values

    pupil_raw = _as_pupil(df.get('TQIP_TBI_PUPIL_RESPONSE'))
    n_pupil_missing = int(pupil_raw.isna().sum())
    pupil_filled = pupil_raw.fillna('both').values

    prs = np.array([GCS_P_PRS[p] for p in pupil_filled], dtype=np.float64)
    gcs_p = gcs_filled - prs  # range -2..15
    risk = 15.0 - gcs_p       # range 0..17

    mask = np.ones(len(df), dtype=bool)
    cov = {'N_used': int(mask.sum()),
           'N_dropped': int((~mask).sum()),
           'N_pupil_imputed': n_pupil_missing,
           'N_missing_gcs_imputed': int(gcs_raw.isna().sum())}
    return risk, mask, cov


# ============================================================================
# Comparator 3: IMPACT-Core (Steyerberg 2008, probability)
# ============================================================================
def compute_impact_core(df):
    age = pd.to_numeric(df.get('AGE_NUMBER'), errors='coerce')
    motor = pd.to_numeric(df.get('MOTOR2'), errors='coerce')
    pupil_raw = _as_pupil(df.get('TQIP_TBI_PUPIL_RESPONSE'))

    mask_age = ~age.isna()
    mask_motor = ~motor.isna() & motor.isin(list(IMPACT_MOTOR_BETAS.keys()))
    mask = (mask_age & mask_motor).values

    n_pupil_missing = int(pupil_raw[mask].isna().sum())
    pupil_filled = pupil_raw.fillna('both').values

    motor_beta = np.array(
        [IMPACT_MOTOR_BETAS.get(int(m), np.nan)
         if not pd.isna(m) else np.nan
         for m in motor.values],
        dtype=np.float64)
    pupil_beta = np.array([IMPACT_PUPIL_BETAS[p] for p in pupil_filled],
                          dtype=np.float64)

    logit = (IMPACT_INTERCEPT
             + IMPACT_AGE_BETA * age.values
             + motor_beta
             + pupil_beta)
    prob = _sigmoid(logit)

    prob_out = np.full(len(df), np.nan, dtype=np.float64)
    prob_out[mask] = prob[mask]

    cov = {'N_used': int(mask.sum()),
           'N_dropped': int((~mask).sum()),
           'N_pupil_imputed': n_pupil_missing}
    return prob_out, mask, cov


# ============================================================================
# Comparator 4: CRASH-Basic HIC 14-day (Perel 2008, probability)
# ============================================================================
def compute_crash_basic(df):
    age = pd.to_numeric(df.get('AGE_NUMBER'), errors='coerce')
    gcs = pd.to_numeric(df.get('GCS2'), errors='coerce')
    # Use TRUE major extracranial (max_ais_other >= 3),
    # computed upstream into major_extracranial (0/1, no NaN).
    extracranial_raw = df.get('major_extracranial')
    if extracranial_raw is None:
        raise RuntimeError("major_extracranial column missing from input — "
                           "CRASH cannot be computed")
    extracranial = pd.to_numeric(extracranial_raw, errors='coerce')
    pupil_raw = _as_pupil(df.get('TQIP_TBI_PUPIL_RESPONSE'))

    # Drop only rows with missing age or GCS (extracranial is already imputed).
    mask = (~age.isna() & ~gcs.isna() & ~extracranial.isna()).values

    n_pupil_missing = int(pupil_raw[mask].isna().sum())
    pupil_filled = pupil_raw.fillna('both').values

    pupil_beta = np.array([CRASH_PUPIL_BETAS[p] for p in pupil_filled],
                          dtype=np.float64)
    extracranial_arr = extracranial.fillna(0).astype(np.float64).values

    logit = (CRASH_INTERCEPT
             + CRASH_AGE_BETA * (age.values - CRASH_AGE_CENTER)
             + CRASH_GCS_BETA * gcs.values
             + pupil_beta
             + CRASH_EXTRACRANIAL_BETA * extracranial_arr)
    prob = _sigmoid(logit)

    prob_out = np.full(len(df), np.nan, dtype=np.float64)
    prob_out[mask] = prob[mask]

    cov = {'N_used': int(mask.sum()),
           'N_dropped': int((~mask).sum()),
           'N_pupil_imputed': n_pupil_missing}
    return prob_out, mask, cov


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="PAT-TBI — Step 6: score the held-out TEST set "
                    "(NTDB 2020-2024)")
    parser.add_argument("--sample", type=int, default=0,
                        help="Optional: randomly sample N test rows for smoke test")
    args = parser.parse_args()

    banner("Step 6 — Apply 5 scores to held-out TEST set (NTDB 2020-2024)")
    print(f"Candidates  : {CANDIDATES_PARQUET}")
    print(f"Test keys   : {TEST_KEYS_CSV}")
    print(f"PAT weights : {PAT_WEIGHTS_CSV}")
    print(f"PAT refit   : {PAT_REFIT_JSON}")
    print(f"Output      : {OUT_PARQUET}")

    # ------------------------------------------------------------------
    # 1. Load candidate parquet + test inc_keys, inner-merge
    # ------------------------------------------------------------------
    banner("Loading candidates + test inc_keys (inner merge)")
    print("Reading candidate parquet...", flush=True)
    df_all = pd.read_parquet(CANDIDATES_PARQUET)
    print(f"  candidates: {df_all.shape}")

    print("Reading test inc_keys...", flush=True)
    test_keys = pd.read_csv(TEST_KEYS_CSV)
    print(f"  test_inc_keys: {test_keys.shape}")

    # Coerce key cols to matching numeric types for the merge
    for frame in (df_all, test_keys):
        if 'inc_key' in frame.columns:
            frame['inc_key'] = pd.to_numeric(
                frame['inc_key'], errors='coerce').astype('Int64')
        if 'PUF_YEAR' in frame.columns:
            frame['PUF_YEAR'] = pd.to_numeric(
                frame['PUF_YEAR'], errors='coerce').astype('Int64')

    # Merge keys: prefer (PUF_YEAR, inc_key) if both present in test_keys,
    # else fall back to inc_key alone.
    if {'PUF_YEAR', 'inc_key'}.issubset(test_keys.columns):
        merge_on = ['PUF_YEAR', 'inc_key']
    else:
        merge_on = ['inc_key']
    print(f"  merging on: {merge_on}")

    df = test_keys.merge(df_all, on=merge_on, how='inner', validate='1:1')
    print(f"  test slice: {df.shape}")

    # Hard assertion: merged length equals test_inc_keys length
    if len(df) != len(test_keys):
        raise RuntimeError(
            f"Test-slice merge length {len(df):,} != test_inc_keys "
            f"length {len(test_keys):,}. Missing keys in candidate parquet?")
    print(f"  ASSERT pass: len(merged) == len(test_inc_keys) == {len(df):,}")

    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=SEED).reset_index(drop=True)
        print(f"  [SMOKE TEST] downsampled to {len(df):,}", flush=True)

    # Ensure outcome is int 0/1, no NaN
    if OUTCOME_COL not in df.columns:
        raise RuntimeError(f"{OUTCOME_COL} column missing from candidate parquet")
    outcome = pd.to_numeric(df[OUTCOME_COL], errors='coerce')
    if outcome.isna().any():
        raise RuntimeError(
            f"{OUTCOME_COL} has {int(outcome.isna().sum())} NaN — must be int 0/1")
    df[OUTCOME_COL] = outcome.astype(int)

    n_total = len(df)
    n_events = int(df[OUTCOME_COL].sum())
    print(f"  TEST: N={n_total:,}   events={n_events:,} "
          f"({100*n_events/n_total:.2f}%)")

    # ------------------------------------------------------------------
    # 2. Compute all 5 scores
    # ------------------------------------------------------------------
    banner("Computing PAT-TBI (refit integer score)")
    pat_tbi_score, pat_tbi_mask, pat_tbi_cov, pat_tbi_used = \
        compute_pat_tbi(df, PAT_WEIGHTS_CSV)
    print(f"  weights used: {pat_tbi_used}")
    print(f"  {pat_tbi_cov}")

    banner("Computing GCS risk (15 - GCS)")
    gcs_score, gcs_mask, gcs_cov = compute_gcs_risk(df)
    print(f"  {gcs_cov}")

    banner("Computing GCS-P risk (Brennan 2018)")
    gcs_p_score, gcs_p_mask, gcs_p_cov = compute_gcs_p_risk(df)
    print(f"  {gcs_p_cov}")

    banner("Computing IMPACT-Core probability (Steyerberg 2008)")
    impact_prob, impact_mask, impact_cov = compute_impact_core(df)
    print(f"  {impact_cov}")

    banner("Computing CRASH-Basic HIC 14-day probability (Perel 2008)")
    crash_prob, crash_mask, crash_cov = compute_crash_basic(df)
    print(f"  {crash_cov}")

    # ------------------------------------------------------------------
    # 3. Sentinel handling
    # ------------------------------------------------------------------
    banner("AIS sentinel handling")
    max_ais_head = pd.to_numeric(df['max_ais_head'], errors='coerce')
    max_ais_other = pd.to_numeric(df['max_ais_other'], errors='coerce')
    n_ais_head_sentinel = int((max_ais_head == 9).sum())
    n_ais_other_sentinel = int((max_ais_other == 9).sum())
    max_ais_head = max_ais_head.where(max_ais_head != 9, np.nan)
    max_ais_other = max_ais_other.where(max_ais_other != 9, np.nan)
    print(f"  AIS sentinel 9 -> NaN: head={n_ais_head_sentinel:,}  "
          f"other={n_ais_other_sentinel:,}")

    # ------------------------------------------------------------------
    # 4. Assemble output DataFrame
    # ------------------------------------------------------------------
    banner("Assembling output DataFrame")

    # iss_cont may be named ISS in some upstream outputs — prefer iss_cont,
    # fall back to ISS, else NaN.
    if 'iss_cont' in df.columns:
        iss_cont = pd.to_numeric(df['iss_cont'], errors='coerce').values
    elif 'ISS' in df.columns:
        iss_cont = pd.to_numeric(df['ISS'], errors='coerce').values
    else:
        iss_cont = np.full(len(df), np.nan)
        print("  WARN: neither iss_cont nor ISS found — writing NaN")

    out = pd.DataFrame({
        'PUF_YEAR': df['PUF_YEAR'].values,
        'inc_key':  df['inc_key'].values,
        OUTCOME_COL: df[OUTCOME_COL].astype(int).values,
        # Demographics
        'AGE_NUMBER': df['AGE_NUMBER'].values,
        'SEX':        df['SEX'].values,
        'RACE':       df['RACE'].values,
        'ETHNICITY':  df['ETHNICITY'].values,
        # Injury severity pass-throughs
        'iss_cont':           iss_cont,
        'max_ais_head':       max_ais_head.values,
        'max_ais_other':      max_ais_other.values,
        'major_extracranial': df['major_extracranial'].values,
        # Raw clinical pass-throughs for comparators
        'GCS2':   df['GCS2'].values,
        'MOTOR2': df['MOTOR2'].values,
        'TQIP_TBI_PUPIL_RESPONSE': df['TQIP_TBI_PUPIL_RESPONSE'].values,
        # Scores
        'pat_tbi_score':      pat_tbi_score,
        'pat_tbi_score_mask': pat_tbi_mask,
        'gcs_risk':           gcs_score,
        'gcs_mask':           gcs_mask,
        'gcs_p_risk':         gcs_p_score,
        'gcs_p_mask':         gcs_p_mask,
        'impact_core_prob':   impact_prob,
        'impact_mask':        impact_mask,
        'crash_basic_prob':   crash_prob,
        'crash_mask':         crash_mask,
    })

    # ------------------------------------------------------------------
    # 5. Hard assertions on output contract
    # ------------------------------------------------------------------
    banner("Output contract assertions")
    if len(out) != len(test_keys):
        raise RuntimeError(
            f"Output length {len(out):,} != test_inc_keys "
            f"length {len(test_keys):,}")
    print(f"  ASSERT pass: len(out) == len(test_inc_keys) == {len(out):,}")

    if out[OUTCOME_COL].isna().any():
        raise RuntimeError(f"{OUTCOME_COL} has NaN in output — must be int 0/1")
    if out[OUTCOME_COL].dtype != np.int64 and out[OUTCOME_COL].dtype != int:
        out[OUTCOME_COL] = out[OUTCOME_COL].astype(int)
    print(f"  ASSERT pass: {OUTCOME_COL} is int 0/1, no NaN")

    # PAT-TBI score must be integer-typed (ensure dtype, no NaN)
    if np.isnan(out['pat_tbi_score'].astype(float)).any():
        raise RuntimeError("pat_tbi_score has NaN — must be integer")
    out['pat_tbi_score'] = out['pat_tbi_score'].astype(np.int64)
    print(f"  ASSERT pass: pat_tbi_score is int64, no NaN")

    # Score columns must be numeric (may have NaN only where mask=False)
    score_mask_pairs = [
        ('pat_tbi_score',    'pat_tbi_score_mask'),
        ('gcs_risk',         'gcs_mask'),
        ('gcs_p_risk',       'gcs_p_mask'),
        ('impact_core_prob', 'impact_mask'),
        ('crash_basic_prob', 'crash_mask'),
    ]
    for score_col, mask_col in score_mask_pairs:
        s = pd.to_numeric(out[score_col], errors='coerce')
        m = out[mask_col].astype(bool)
        n_nan_in_mask = int(s[m].isna().sum())
        if n_nan_in_mask > 0:
            raise RuntimeError(
                f"{score_col} has {n_nan_in_mask} NaN where {mask_col}=True")
        print(f"  ASSERT pass: {score_col} numeric; NaN only where "
              f"{mask_col}=False (n_used={int(m.sum()):,})")

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    banner("Score distribution summary (TEST set)")
    for score_col, mask_col in score_mask_pairs:
        s = pd.to_numeric(out[score_col], errors='coerce')
        m = out[mask_col].astype(bool)
        s_m = s[m].dropna()
        if len(s_m) == 0:
            print(f"  {score_col:<20s} n=0")
            continue
        print(f"  {score_col:<20s} n={len(s_m):>8,}  "
              f"min={s_m.min():>8.3f}  "
              f"max={s_m.max():>8.3f}  "
              f"mean={s_m.mean():>8.3f}  "
              f"median={s_m.median():>8.3f}")

    # ------------------------------------------------------------------
    # 7. Write
    # ------------------------------------------------------------------
    banner(f"Writing {OUT_PARQUET}")
    out.to_parquet(OUT_PARQUET, index=False)
    print(f"  wrote {out.shape[0]:,} rows × {out.shape[1]} cols")
    print("\nStep 6 complete.")


if __name__ == '__main__':
    main()
