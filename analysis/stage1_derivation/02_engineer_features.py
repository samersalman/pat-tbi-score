"""
PAT-TBI v4 - Step 2: Engineer ED-rapid LASSO candidate matrix + platelet outcomes
================================================================================

Builds the 27-variable parsimonious ED-rapid feature set (4 demographics +
11 ED vitals + 4 GCS + 1 pupil + 4 mechanism + 1 transfer + 2 coag-relevant
comorbidities) that will be fed to LASSO stability selection in Step 4, and
derives the platelet-receipt outcome columns:
  - received_platelets_4h (primary):     1 if PLATELETS4HOURS > 0, else 0
  - received_platelets_or_wholeblood_4h (sensitivity): 1 if PLATELETS4HOURS > 0
                                           OR WHOLEBLOOD4HOURS > 0. Defined for
                                           the full 2020-2024 cohort.

v4 parsimony rationale
----------------------
Relative to v3, v4 drops 15 candidates that are either non-bedside-extractable,
hospital-system level, or history comorbidities not directly tied to platelet/
coag biology. Dropped (15):
  Hospital-system (3): teaching_status_binary, verification_level_1,
                       highest_activation_highest
  Insurance (1)      : ins_uninsured
  History comorbids (11): cc_chf, cc_cva, cc_htn, cc_dm, cc_copd, cc_smoker,
                          cc_ckd, cc_cirrhosis, cc_cancer, cc_dementia, cc_alcohol
The two coagulation-relevant comorbidities (cc_anticoag, cc_bleeding) are kept
as bedside-extractable proxies for platelet/coag biology.

Physiology-derived binary features follow the standard ED-rapid engineering
rules. AIS columns (max_ais_head, max_ais_other, major_extracranial, iss_cont)
are carried as NON-X columns - needed for CRASH-Basic comparator scoring and
subgroup stratification but forbidden from the LASSO X matrix.

Inputs
------
  PAT_TBI/v4/derived/cohort_raw.parquet                   (Step 1 output)

Outputs
-------
  PAT_TBI/v4/derived/candidates.parquet
  PAT_TBI/v4/tables/table_candidate_coverage.csv

Schema contract (consumed by Step 3 split and Step 4 LASSO):
  Keys + outcome : PUF_YEAR, inc_key, received_platelets_4h,
                   received_platelets_or_wholeblood_4h
  Demographics   : AGE_NUMBER, SEX, RACE, ETHNICITY  (raw, for Table 1 + subgroups)
  Raw comparator : GCS2, MOTOR2, TQIP_TBI_PUPIL_RESPONSE  (GCS / GCS-P inputs)
  Non-X columns  : iss_cont, max_ais_head, max_ais_other, major_extracranial
  X candidates   : LASSO_X_COLS list below (27 vars)

Leakage audit (assertion):
  FORBIDDEN in LASSO_X_COLS:
    received_platelets_4h, received_platelets_or_wholeblood_4h,
    PLATELETS4HOURS, WHOLEBLOOD4HOURS,
    dc_death, HOSPDISCHARGEDISPOSITION, TOTALICULOS, TOTALLOSDAYS,
    HOSPEVENT*, EMDISPOSITION, VENTILATORDAYS, TIMETOFIRSTBRAINCT,
    TBIMIDLINESHIFT, any ICD procedure cols,
    iss_cont, max_ais_head, max_ais_other, major_extracranial,
    tox_positive, bmi_*, etoh_positive
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd

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

IN_RAW = os.path.join(DERIVED, "cohort_raw.parquet")
OUT_CANDIDATES = os.path.join(DERIVED, "candidates.parquet")
OUT_COVERAGE = os.path.join(TABLES, "table_candidate_coverage.csv")

# Outcome columns
OUTCOME_PRIMARY = "received_platelets_4h"
OUTCOME_SENSITIVITY = "received_platelets_or_wholeblood_4h"

# Coagulation-relevant comorbidities kept in v4 (the other 11 cc_* dropped).
CC_COLS = ["cc_anticoag", "cc_bleeding"]

# ---- LASSO candidate X columns (27 ED-rapid predictors, v4 parsimonious) --
# All binary except shock_index (continuous). Order is fixed and meaningful
# for downstream consumers.
LASSO_X_COLS = [
    # Demographics (4)
    "age_45_64", "age_65_74", "age_75plus", "female",
    # ED vitals (11) — shock_index kept continuous
    "sbp_hypotensive", "sbp_hypertensive",
    "hr_bradycardic", "hr_tachycardic",
    "rr_low", "rr_high",
    "hypoxic",
    "shock_index",
    "si_elevated", "si_high",
    "temp_hypothermic",
    # GCS (4)
    "gcs_severe", "gcs_moderate", "gcs_motor_low", "gcs_motor_high",
    # Pupils (1)
    "tbi_pupil_none",
    # Mechanism (4)
    "mech_mvc", "mech_assault", "mech_fall", "risk_penetrating",
    # Transfer (1)
    "is_transfer",
    # Coagulation-relevant comorbidities (2)
    "cc_anticoag", "cc_bleeding",
]

# ---- FORBIDDEN columns (leakage audit) ------------------------------------
FORBIDDEN_X = {
    # The outcome columns themselves (leakage -> ~perfect AUROC)
    "received_platelets_4h", "received_platelets_or_wholeblood_4h",
    "PLATELETS4HOURS", "WHOLEBLOOD4HOURS",
    # Legacy mortality outcome (not the PAT outcome, just an audit safety)
    "dc_death",
    # Post-baseline disposition / resource use
    "HOSPDISCHARGEDISPOSITION", "TOTALICULOS", "TOTALLOSDAYS",
    "EMDISPOSITION", "VENTILATORDAYS", "TIMETOFIRSTBRAINCT", "TBIMIDLINESHIFT",
    # Explicit ED-rapid exclusions (still computed for comparator scoring)
    "iss_cont", "max_ais_head", "max_ais_other", "major_extracranial",
    "tox_positive", "etoh_positive",
}


def banner(msg):
    line = "=" * 78
    print(f"\n{line}\n{msg}\n{line}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0)
    args = parser.parse_args()

    banner("PAT-TBI v4 Step 2 - Engineer ED-rapid candidate feature matrix + outcomes")

    print(f"Reading {IN_RAW}", flush=True)
    df = pd.read_parquet(IN_RAW)
    print(f"  raw shape: {df.shape}", flush=True)

    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=SEED).reset_index(drop=True)
        print(f"  [SMOKE TEST] downsampled to {len(df):,}", flush=True)

    df["inc_key"] = pd.to_numeric(df["inc_key"], errors="coerce").astype("Int64")
    df["PUF_YEAR"] = pd.to_numeric(df["PUF_YEAR"], errors="coerce").astype("Int64")

    # --------------------------------------------------------------
    # Numeric coercions (keep NaN for comparator inputs)
    # --------------------------------------------------------------
    gcs = pd.to_numeric(df["GCS2"], errors="coerce")
    motor = pd.to_numeric(df["MOTOR2"], errors="coerce")
    sbp = pd.to_numeric(df["SBP2"], errors="coerce")
    p = pd.to_numeric(df["P2"], errors="coerce")
    rr = pd.to_numeric(df["RR2"], errors="coerce")
    ox = pd.to_numeric(df["OX2"], errors="coerce")
    age = pd.to_numeric(df["AGE_NUMBER"], errors="coerce")
    iss = pd.to_numeric(df["ISS"], errors="coerce")
    temp = pd.to_numeric(df.get("temperature_raw"), errors="coerce")

    # --------------------------------------------------------------
    # Demographics bands (null-biased: missing→0)
    # --------------------------------------------------------------
    df["age_45_64"] = ((age >= 45) & (age < 65)).fillna(False).astype(int)
    df["age_65_74"] = ((age >= 65) & (age < 75)).fillna(False).astype(int)
    df["age_75plus"] = (age >= 75).fillna(False).astype(int)

    sex = df["SEX"].astype("string").str.upper().str.strip()
    df["female"] = (sex == "F").fillna(False).astype(int)

    # --------------------------------------------------------------
    # ED vitals
    # --------------------------------------------------------------
    df["sbp_hypotensive"] = (sbp < 90).fillna(False).astype(int)
    df["sbp_hypertensive"] = (sbp >= 140).fillna(False).astype(int)
    df["hr_bradycardic"] = (p < 60).fillna(False).astype(int)
    df["hr_tachycardic"] = (p >= 120).fillna(False).astype(int)
    df["rr_low"] = (rr < 12).fillna(False).astype(int)
    df["rr_high"] = (rr >= 30).fillna(False).astype(int)
    df["hypoxic"] = (ox <= 92).fillna(False).astype(int)

    with np.errstate(divide="ignore", invalid="ignore"):
        si_raw = p / sbp.where(sbp > 0)
    # Continuous shock index: NaN → 0.7 (typical normal — neutral)
    df["shock_index"] = si_raw.fillna(0.7)
    df["si_elevated"] = (si_raw >= 0.9).fillna(False).astype(int)
    df["si_high"] = (si_raw >= 1.3).fillna(False).astype(int)

    # Temperature (hypothermia <35°C). If values look Fahrenheit (>50),
    # treat as NaN-effectively by requiring °C range.
    df["temp_hypothermic"] = ((temp >= 20) & (temp < 35)).fillna(False).astype(int)

    # --------------------------------------------------------------
    # GCS
    # --------------------------------------------------------------
    df["gcs_severe"] = (gcs <= 8).fillna(False).astype(int)
    df["gcs_moderate"] = ((gcs >= 9) & (gcs <= 12)).fillna(False).astype(int)
    df["gcs_motor_low"] = (motor <= 3).fillna(False).astype(int)
    df["gcs_motor_high"] = (motor == 6).fillna(False).astype(int)

    # --------------------------------------------------------------
    # Pupils (null-biased missing → "BOTH")
    # --------------------------------------------------------------
    pupil = df["TQIP_TBI_PUPIL_RESPONSE"].astype("string").str.upper().str.strip()
    df["tbi_pupil_none"] = pupil.isin(["NEITHER", "NONE"]).fillna(False).astype(int)

    # --------------------------------------------------------------
    # Mechanism (ICD-10 e-code prefixes)
    # --------------------------------------------------------------
    ecode = df["cause_e_code"].astype("string").str.upper().str.strip().fillna("")

    MVC_PREFIXES = tuple([f"V{i:02d}" for i in range(40, 90)])   # V40-V89 motor vehicle traffic
    FALL_PREFIXES = tuple([f"W{i:02d}" for i in range(0, 20)])    # W00-W19 falls
    ASSAULT_PREFIXES = tuple(
        [f"X{i}" for i in range(85, 100)] + [f"Y0{i}" for i in range(0, 10)]
    )
    PEN_PREFIXES = tuple([f"X{i}" for i in range(93, 100)])       # X93-X99 penetrating assault

    df["mech_mvc"] = ecode.str.startswith(MVC_PREFIXES).astype(int)
    df["mech_fall"] = ecode.str.startswith(FALL_PREFIXES).astype(int)
    df["mech_assault"] = ecode.str.startswith(ASSAULT_PREFIXES).astype(int)
    df["risk_penetrating"] = ecode.str.startswith(PEN_PREFIXES).astype(int)

    # Transfer
    if "HOSPITAL_TRANSFER" in df.columns:
        tr = df["HOSPITAL_TRANSFER"].astype("string").str.upper().str.strip()
        df["is_transfer"] = (tr == "Y").fillna(False).astype(int)
    else:
        df["is_transfer"] = 0

    # --------------------------------------------------------------
    # Hospital / system factors — DROPPED in v4 (not patient-level/bedside).
    # teaching_status_binary, verification_level_1, highest_activation_highest
    # are no longer engineered. Raw columns may still be present in
    # cohort_raw.parquet from Step 1; that is fine — they are just not promoted
    # into the candidate matrix.
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # AIS columns — already present in the input parquet (Step 1 derives them
    # from DuckDB raw_ais_diagnosis). Verify and coerce to int with NaN→0.
    # --------------------------------------------------------------
    banner("Verifying AIS columns from cohort parquet (comparator-only, NOT in X)")
    for col in ("max_ais_head", "max_ais_other", "major_extracranial"):
        assert col in df.columns, f"Required AIS column missing from input parquet: {col}"
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Continuous ISS for subgroup stratification
    df["iss_cont"] = iss.fillna(0)

    # --------------------------------------------------------------
    # Comorbidity sanity — already 0/1 from Step 1.
    # v4 keeps only the two coagulation-relevant comorbidities; the other
    # 11 history comorbidities (cc_chf, cc_cva, cc_htn, cc_dm, cc_copd,
    # cc_smoker, cc_ckd, cc_cirrhosis, cc_cancer, cc_dementia, cc_alcohol)
    # are not promoted into the candidate matrix.
    # --------------------------------------------------------------
    for cc in CC_COLS:
        if cc not in df.columns:
            print(f"  WARN: {cc} missing from Step 1 output — filling with 0", flush=True)
            df[cc] = 0
        df[cc] = pd.to_numeric(df[cc], errors="coerce").fillna(0).astype(int)

    # --------------------------------------------------------------
    # Outcome derivation
    # --------------------------------------------------------------
    banner("Deriving platelet-receipt outcomes")
    plt4 = pd.to_numeric(df["PLATELETS4HOURS"], errors="coerce")
    wb4 = pd.to_numeric(df.get("WHOLEBLOOD4HOURS"), errors="coerce") if "WHOLEBLOOD4HOURS" in df.columns else pd.Series(pd.NA, index=df.index)

    # Primary: any platelet product within 4h of arrival.
    # NTDB mandated consistent reporting from 2020 onward; empty = not recorded
    # at the facility. Per NTDB data-dictionary convention and the ≥90% fill rate
    # for PLATELETS4HOURS in 2020-2024, missing is treated as 0 (not transfused).
    df[OUTCOME_PRIMARY] = ((plt4.fillna(0) > 0)).astype(int)

    # Sensitivity: platelets OR whole blood (whole blood contains platelets).
    # In the 2020-2024 cohort, WHOLEBLOOD4HOURS is reported across all years, so
    # the sensitivity outcome is defined for the entire cohort.
    wb4_present = df["PUF_YEAR"].isin([2020, 2021, 2022, 2023, 2024])
    sens = pd.Series(pd.NA, index=df.index, dtype="Int64")
    sens.loc[wb4_present] = (
        (plt4.loc[wb4_present].fillna(0) > 0) | (wb4.loc[wb4_present].fillna(0) > 0)
    ).astype(int)
    df[OUTCOME_SENSITIVITY] = sens

    n_prim = int(df[OUTCOME_PRIMARY].sum())
    n_sens = int(df.loc[wb4_present, OUTCOME_SENSITIVITY].astype(int).sum())
    print(f"  N (cohort)                         : {len(df):,}", flush=True)
    print(f"  {OUTCOME_PRIMARY:<35s}: {n_prim:,} ({100*n_prim/len(df):.2f}%)", flush=True)
    print(f"  N (2020-2024 for sensitivity)      : {int(wb4_present.sum()):,}", flush=True)
    print(f"  {OUTCOME_SENSITIVITY:<35s}: {n_sens:,} "
          f"({100*n_sens/max(int(wb4_present.sum()),1):.2f}%)", flush=True)

    # --------------------------------------------------------------
    # Leakage audit - assert none of the forbidden columns sneak into X
    # --------------------------------------------------------------
    banner("Leakage audit")
    x_set = set(LASSO_X_COLS)
    intersect = x_set & FORBIDDEN_X
    assert not intersect, f"LEAKAGE: forbidden columns in LASSO_X_COLS: {intersect}"
    assert OUTCOME_PRIMARY not in LASSO_X_COLS, f"{OUTCOME_PRIMARY} CANNOT be a predictor"
    # v4 parsimonious keep list: 27 vars enumerated by the spec
    # (4 demo + 11 ED vitals + 4 GCS + 1 pupil + 4 mech + 1 transfer + 2 coag-cc).
    # The plan's headline ("26") undercounts by one; the bulleted list is the
    # source of truth (mirrors the same convention used for the drop list,
    # where the headline said "14" but the bullets enumerate 15).
    assert len(LASSO_X_COLS) == 27, (
        f"v4 candidate list must be exactly 27 vars (per enumerated keep list); "
        f"got {len(LASSO_X_COLS)}"
    )
    missing_x = [c for c in LASSO_X_COLS if c not in df.columns]
    assert not missing_x, f"X columns missing from df: {missing_x}"
    for c in LASSO_X_COLS:
        n_nan = int(df[c].isna().sum())
        assert n_nan == 0, f"{c} has {n_nan} NaN - forbidden in X matrix"
        assert np.issubdtype(df[c].dtype, np.number), f"{c} is non-numeric"
    print(f"  OK  {len(LASSO_X_COLS)} X columns clean, no leakage, no NaN", flush=True)

    # --------------------------------------------------------------
    # Select final columns and write
    # --------------------------------------------------------------
    keep = [
        # Keys + outcomes
        "PUF_YEAR", "inc_key", OUTCOME_PRIMARY, OUTCOME_SENSITIVITY,
        # Demographics raw (for subgroups / Table 1)
        "AGE_NUMBER", "SEX", "RACE", "ETHNICITY",
        # Comparator inputs (raw)
        "GCS2", "MOTOR2", "TQIP_TBI_PUPIL_RESPONSE",
        # Non-X columns (comparator + subgroups)
        "iss_cont", "max_ais_head", "max_ais_other", "major_extracranial",
        # X candidates
        *LASSO_X_COLS,
    ]
    out = df[keep].copy()
    print(f"\n  candidate matrix shape: {out.shape}", flush=True)

    out.to_parquet(OUT_CANDIDATES, index=False)
    print(f"  wrote {OUT_CANDIDATES}", flush=True)

    # --------------------------------------------------------------
    # Coverage table
    # --------------------------------------------------------------
    banner("Candidate coverage table")
    rows = []
    for c in LASSO_X_COLS:
        rows.append({
            "Variable": c,
            "N": len(out),
            "N_positive": int(out[c].sum()) if out[c].dtype.kind in "biu" else int((out[c] != 0).sum()),
            "Mean": round(float(out[c].mean()), 4),
            "Min": round(float(out[c].min()), 4),
            "Max": round(float(out[c].max()), 4),
        })
    cov = pd.DataFrame(rows)
    cov.to_csv(OUT_COVERAGE, index=False)
    print(f"  wrote {OUT_COVERAGE}", flush=True)

    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(cov.to_string(index=False))

    print("\nPAT-TBI v4 Step 2 complete.")


if __name__ == "__main__":
    main()
