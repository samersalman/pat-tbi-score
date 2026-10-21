"""
PAT-TBI — Step 11: Sensitivity / generalization to all adult NTDB 2020-2024
================================================================================

Applies the FROZEN PAT-TBI integer-weight score (table_pat_weights.csv) to
adult patients in NTDB 2020-2024 across the full all-trauma cohort (no TBI
restriction). Confirms that the score's discrimination generalizes beyond the
S06 derivation cohort.

No refitting. No variable selection. The integer weights are locked.

Cohort window: 2020-2024 (~6.2M expected after exclusions).

Data sources
------------
  DuckDB:  _DATA/NTDB/ntdb_database/ntdb_research.duckdb
    - v_patients_2019_2024  : per-patient flat view
    - raw_preexisting       : long-format preexisting conditions

  Raw PUF (only for TEACHINGSTATUS / VERIFICATIONLEVEL / HIGHESTACTIVATION,
  which are not exposed in DuckDB):
    _DATA/NTDB/PUF/PUF AY {yr}/CSV/PUF_TRAUMA.csv  for yr in 2020..2024

Exclusions (mirroring Step 1)
-----------------------------
  1. start_2020_2024_age_18plus_from_duckdb
  2. exclude_prehospital_arrest        (PREHOSPITAL_ARREST == 'Y')
  3. exclude_ed_disposition_expired    (ED_DISPOSITION_CODE == 'MOR')
  4. exclude_transferred_out           (ED_DISPOSITION_CODE in {'5A_S','5BMS'})
  5. exclude_missing_ed_disposition    (ED_DISPOSITION_CODE IS NULL)

NOTE: this is the all-trauma sensitivity analysis -- patients are NOT
restricted to S06 (TBI) diagnoses.

Inputs
------
  PAT_TBI/v4/tables/table_pat_weights.csv   (frozen integer weights from Step 5)

Outputs
-------
  PAT_TBI/v4/tables/table_sensitivity_all_trauma.csv
  PAT_TBI/v4/derived/all_trauma_scored.parquet     (optional cache)
  PAT_TBI/v4/figures/figure_sensitivity_all_trauma.{pdf,png}
"""
import argparse
import json
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, brier_score_loss
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

N_BOOT = 2000
BOOT_SUBSAMPLE = 100_000
EPS = 1e-6

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
import os as _os, sys as _sys  # path handling: make analysis/config.py importable
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import config  # noqa: E402  (analysis/config.py: PAT_TBI_WORKDIR, PAT_TBI_NTDB_DUCKDB)
PAT = config.V4_DIR
TABLES = PAT / "tables"
FIGURES = PAT / "figures"
DERIVED = PAT / "derived"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
DERIVED.mkdir(parents=True, exist_ok=True)

DUCKDB_PATH = config.DUCKDB_PATH
PUF_TRAUMA_TPL = config.PUF_TRAUMA_TPL

PAT_WEIGHTS_CSV = TABLES / "table_pat_weights.csv"

OUT_TABLE = TABLES / "table_sensitivity_all_trauma.csv"
OUT_PARQUET = DERIVED / "all_trauma_scored.parquet"
OUT_FIG_PDF = FIGURES / "figure_sensitivity_all_trauma.pdf"
OUT_FIG_PNG = FIGURES / "figure_sensitivity_all_trauma.png"

YEARS = [2020, 2021, 2022, 2023, 2024]

# --------------------------------------------------------------------------
# Comorbidity code map (NTDB PUF Variable Formats)
# --------------------------------------------------------------------------
CODE_TO_CC = {
    2: "cc_alcohol",
    4: "cc_bleeding",
    7: "cc_chf",
    8: "cc_smoker",
    9: "cc_ckd",
    10: "cc_cva",
    11: "cc_dm",
    12: "cc_cancer",
    19: "cc_htn",
    23: "cc_copd",
    25: "cc_cirrhosis",
    26: "cc_dementia",
    31: "cc_anticoag",
}
CC_COLS = sorted(set(CODE_TO_CC.values()))

# --------------------------------------------------------------------------
# ED_DISPOSITION_CODE classification (verified 2020-2024)
# --------------------------------------------------------------------------
ED_DISP_TRANSFERRED_OUT = ("5A_S", "5BMS")
ED_DISP_EXPIRED = ("MOR",)

# --------------------------------------------------------------------------
# Raw PUF columns (not in DuckDB) -> output schema names
# --------------------------------------------------------------------------
RAW_PUF_COLMAP = {
    "TEACHINGSTATUS": "teaching_status_raw",
    "VERIFICATIONLEVEL": "verification_level_raw",
    "HIGHESTACTIVATION": "highest_activation_raw",
}


def banner(msg):
    line = "=" * 78
    print(f"\n{line}\n{msg}\n{line}", flush=True)


def log_step(name, n, log):
    log[name] = int(n)
    print(f"  [{name}] N = {n:,}", flush=True)


# --------------------------------------------------------------------------
# A. Pull all-trauma base cohort from DuckDB (no S06 restriction)
# --------------------------------------------------------------------------
def fetch_base_cohort(con, log):
    banner("Pull all-trauma cohort from DuckDB (2020-2024, age 18+)")

    n_initial = con.execute(
        """
        SELECT COUNT(*)
        FROM v_patients_2019_2024
        WHERE puf_year BETWEEN 2020 AND 2024
          AND AGE_NUMBER IS NOT NULL
          AND AGE_NUMBER >= 18
        """
    ).fetchone()[0]
    log_step("start_2020_2024_age_18plus_from_duckdb", n_initial, log)

    print("  pulling all-trauma cohort frame...", flush=True)
    df = con.execute(
        """
        SELECT
            v.inc_key                   AS inc_key,
            v.puf_year                  AS PUF_YEAR,
            v.AGE_NUMBER                AS AGE_NUMBER,
            v.SEX                       AS SEX,
            v.PAYMENT_SOURCE            AS PAYMENT_SOURCE,
            v.GCS2                      AS GCS2,
            v.MOTOR2                    AS MOTOR2,
            v.EYE2                      AS EYE2,
            v.VERBAL2                   AS VERBAL2,
            v.TQIP_TBI_PUPIL_RESPONSE   AS TQIP_TBI_PUPIL_RESPONSE,
            v.SBP2                      AS SBP2,
            v.P2                        AS P2,
            v.RR2                       AS RR2,
            v.OX2                       AS OX2,
            v.TEMPS2                    AS temperature_raw,
            v.CAUSE_E_CODES10           AS cause_e_code,
            v.HOSPITAL_TRANSFER         AS HOSPITAL_TRANSFER,
            v.ED_DISPOSITION_CODE       AS ED_DISPOSITION_CODE,
            v.PREHOSPITAL_ARREST        AS PREHOSPITAL_ARREST,
            v.ISS                       AS ISS,
            v.TQIP_HC_PLATE_4HR         AS PLATELETS4HOURS
        FROM v_patients_2019_2024 v
        WHERE v.puf_year BETWEEN 2020 AND 2024
          AND v.AGE_NUMBER IS NOT NULL
          AND v.AGE_NUMBER >= 18
        """
    ).fetchdf()
    print(f"  base rows fetched: {len(df):,}", flush=True)
    return df


# --------------------------------------------------------------------------
# B. Comorbidity pivot (cohort-key restricted, DuckDB-side)
# --------------------------------------------------------------------------
def fetch_comorbidities(con, cohort_keys_df):
    banner("Pivot raw_preexisting -> 13 comorbidity flags (cohort keys only)")

    keys = cohort_keys_df[["inc_key", "PUF_YEAR"]].rename(
        columns={"PUF_YEAR": "puf_year"}).copy()
    keys["inc_key"] = pd.to_numeric(keys["inc_key"], errors="coerce").astype("Int64")
    keys["puf_year"] = pd.to_numeric(keys["puf_year"], errors="coerce").astype("Int32")
    con.register("_cohort_keys_df", keys)
    con.execute(
        "CREATE OR REPLACE TEMP TABLE _cohort_keys AS "
        "SELECT inc_key, puf_year FROM _cohort_keys_df"
    )

    code_list_sql = ",".join(str(k) for k in CODE_TO_CC.keys())
    print(f"  aggregating preexisting answers (codes: "
          f"{sorted(CODE_TO_CC.keys())})", flush=True)
    long_df = con.execute(
        f"""
        SELECT p.inc_key, p.puf_year,
               p.preexisting_condition AS code
        FROM raw_preexisting p
        INNER JOIN _cohort_keys c
          ON p.inc_key = c.inc_key AND p.puf_year = c.puf_year
        WHERE p.puf_year BETWEEN 2020 AND 2024
          AND p.preexisting_condition_answer = 1
          AND p.preexisting_condition IN ({code_list_sql})
        """
    ).fetchdf()
    print(f"  positive preexisting rows in cohort: {len(long_df):,}", flush=True)

    if len(long_df) == 0:
        wide = pd.DataFrame(columns=["inc_key", "PUF_YEAR"] + CC_COLS)
        wide["inc_key"] = wide["inc_key"].astype("Int64")
        wide["PUF_YEAR"] = wide["PUF_YEAR"].astype("Int64")
        return wide

    long_df["cc_name"] = long_df["code"].map(CODE_TO_CC)
    long_df["one"] = 1
    wide = (
        long_df.pivot_table(
            index=["inc_key", "puf_year"],
            columns="cc_name",
            values="one",
            aggfunc="max",
            fill_value=0,
        )
        .reset_index()
    )
    for cc in CC_COLS:
        if cc not in wide.columns:
            wide[cc] = 0
    wide = wide[["inc_key", "puf_year"] + CC_COLS]
    wide = wide.rename(columns={"puf_year": "PUF_YEAR"})
    wide["inc_key"] = pd.to_numeric(wide["inc_key"], errors="coerce").astype("Int64")
    wide["PUF_YEAR"] = pd.to_numeric(wide["PUF_YEAR"], errors="coerce").astype("Int64")
    return wide


# --------------------------------------------------------------------------
# C. Hospital factor columns from raw PUF_TRAUMA.csv (not in DuckDB)
# --------------------------------------------------------------------------
def fetch_raw_puf_extras(cohort_keys_df):
    banner("Load TEACHINGSTATUS / VERIFICATIONLEVEL / HIGHESTACTIVATION "
           "from raw PUF_TRAUMA.csv (per year)")

    raw_cols = list(RAW_PUF_COLMAP.keys())
    out_cols = ["inc_key", "PUF_YEAR"] + [RAW_PUF_COLMAP[c] for c in raw_cols]

    out = []
    for yr in YEARS:
        path = PUF_TRAUMA_TPL.format(yr=yr)
        if not os.path.exists(path):
            print(f"  [{yr}] WARN: PUF_TRAUMA.csv not found at {path}; skipping",
                  flush=True)
            continue
        keys_yr = set(
            pd.to_numeric(
                cohort_keys_df.loc[cohort_keys_df["PUF_YEAR"] == yr, "inc_key"],
                errors="coerce",
            ).dropna().astype("int64").tolist()
        )
        if not keys_yr:
            print(f"  [{yr}] no cohort keys in this year; skipping", flush=True)
            continue

        header = pd.read_csv(path, nrows=0)
        cols_present = set(header.columns)

        present_raw = [c for c in raw_cols if c in cols_present]
        missing_raw = [c for c in raw_cols if c not in cols_present]
        if missing_raw:
            print(f"  [{yr}] WARN: missing PUF columns -> {missing_raw}; "
                  f"those columns emitted as NA for this year", flush=True)
        if not present_raw:
            print(f"  [{yr}] WARN: none of the requested PUF columns present; "
                  f"skipping year", flush=True)
            continue

        usecols = ["inc_key"] + present_raw
        # Read in chunks to keep memory bounded for the all-trauma cohort
        chunk_iter = pd.read_csv(path, usecols=usecols, low_memory=False,
                                 chunksize=500_000)
        per_year_parts = []
        for chunk in chunk_iter:
            chunk["inc_key"] = pd.to_numeric(chunk["inc_key"],
                                             errors="coerce").astype("Int64")
            chunk = chunk[chunk["inc_key"].isin(keys_yr)].copy()
            if len(chunk):
                per_year_parts.append(chunk)
        if not per_year_parts:
            continue
        df_yr = pd.concat(per_year_parts, ignore_index=True)
        df_yr["PUF_YEAR"] = int(yr)

        for raw_c in raw_cols:
            if raw_c not in df_yr.columns:
                df_yr[raw_c] = pd.NA
        df_yr = df_yr.rename(columns=RAW_PUF_COLMAP)

        n_match = len(df_yr)
        ts = RAW_PUF_COLMAP["TEACHINGSTATUS"]
        vl = RAW_PUF_COLMAP["VERIFICATIONLEVEL"]
        ha = RAW_PUF_COLMAP["HIGHESTACTIVATION"]
        print(
            f"  [{yr}] cohort-matched PUF rows: {n_match:,}  | "
            f"teaching non-null: {int(df_yr[ts].notna().sum()):,}  | "
            f"verif non-null: {int(df_yr[vl].notna().sum()):,}  | "
            f"activation non-null: {int(df_yr[ha].notna().sum()):,}",
            flush=True,
        )
        out.append(df_yr[out_cols])

    if not out:
        return pd.DataFrame(columns=out_cols)
    combined = pd.concat(out, ignore_index=True)
    combined["PUF_YEAR"] = pd.to_numeric(
        combined["PUF_YEAR"], errors="coerce").astype("Int64")
    return combined


# --------------------------------------------------------------------------
# D. Feature engineering -- replicates v4/code/02_engineer_features.py logic
# --------------------------------------------------------------------------
def engineer_binaries(df):
    """Build the ED-rapid candidate features used by the LASSO refit.

    Logic mirrors v4/code/02_engineer_features.py so that every variable
    referenced by table_pat_weights.csv is present and coded identically.
    """
    gcs = pd.to_numeric(df["GCS2"], errors="coerce")
    motor = pd.to_numeric(df["MOTOR2"], errors="coerce")
    sbp = pd.to_numeric(df["SBP2"], errors="coerce")
    p = pd.to_numeric(df["P2"], errors="coerce")
    rr = pd.to_numeric(df["RR2"], errors="coerce")
    ox = pd.to_numeric(df["OX2"], errors="coerce")
    age = pd.to_numeric(df["AGE_NUMBER"], errors="coerce")
    temp = pd.to_numeric(df.get("temperature_raw"), errors="coerce")

    # Demographics
    df["age_45_64"] = ((age >= 45) & (age < 65)).fillna(False).astype(int)
    df["age_65_74"] = ((age >= 65) & (age < 75)).fillna(False).astype(int)
    df["age_75plus"] = (age >= 75).fillna(False).astype(int)

    sex = df["SEX"].astype("string").str.upper().str.strip()
    df["female"] = (sex == "F").fillna(False).astype(int)

    pay = df["PAYMENT_SOURCE"].astype("string").str.upper().str.strip()
    df["ins_uninsured"] = pay.isin(["SELF", "CHARITY", "NONE"]).fillna(False).astype(int)

    # ED vitals
    df["sbp_hypotensive"] = (sbp < 90).fillna(False).astype(int)
    df["sbp_hypertensive"] = (sbp >= 140).fillna(False).astype(int)
    df["hr_bradycardic"] = (p < 60).fillna(False).astype(int)
    df["hr_tachycardic"] = (p >= 120).fillna(False).astype(int)
    df["rr_low"] = (rr < 12).fillna(False).astype(int)
    df["rr_high"] = (rr >= 30).fillna(False).astype(int)
    df["hypoxic"] = (ox <= 92).fillna(False).astype(int)

    with np.errstate(divide="ignore", invalid="ignore"):
        si_raw = p / sbp.where(sbp > 0)
    df["shock_index"] = si_raw.fillna(0.7)
    df["si_elevated"] = (si_raw >= 0.9).fillna(False).astype(int)
    df["si_high"] = (si_raw >= 1.3).fillna(False).astype(int)

    df["temp_hypothermic"] = ((temp >= 20) & (temp < 35)).fillna(False).astype(int)

    # GCS
    df["gcs_severe"] = (gcs <= 8).fillna(False).astype(int)
    df["gcs_moderate"] = ((gcs >= 9) & (gcs <= 12)).fillna(False).astype(int)
    df["gcs_motor_low"] = (motor <= 3).fillna(False).astype(int)
    df["gcs_motor_high"] = (motor == 6).fillna(False).astype(int)

    # Pupils
    pupil = df["TQIP_TBI_PUPIL_RESPONSE"].astype("string").str.upper().str.strip()
    df["tbi_pupil_none"] = pupil.isin(["NEITHER", "NONE"]).fillna(False).astype(int)

    # Mechanism (ICD-10 e-code prefixes)
    ecode = df["cause_e_code"].astype("string").str.upper().str.strip().fillna("")
    MVC_PREFIXES = tuple([f"V{i:02d}" for i in range(40, 90)])
    FALL_PREFIXES = tuple([f"W{i:02d}" for i in range(0, 20)])
    ASSAULT_PREFIXES = tuple(
        [f"X{i}" for i in range(85, 100)] + [f"Y0{i}" for i in range(0, 10)]
    )
    PEN_PREFIXES = tuple([f"X{i}" for i in range(93, 100)])
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

    # Hospital / system factors
    ts_raw = df.get("teaching_status_raw", pd.Series([pd.NA] * len(df)))
    ts_str = ts_raw.astype("string").str.upper().str.strip()
    ts_num = pd.to_numeric(ts_raw, errors="coerce")
    teaching_from_str = ts_str.isin(["ACADEMIC", "UNIVERSITY"]).fillna(False)
    teaching_from_num = ts_num.isin([1, 4]).fillna(False)
    df["teaching_status_binary"] = (teaching_from_str | teaching_from_num).astype(int)

    vl_raw = df.get("verification_level_raw", pd.Series([pd.NA] * len(df)))
    vl_num = pd.to_numeric(vl_raw, errors="coerce")
    vl_str = vl_raw.astype("string").str.upper().str.strip()
    verif_from_num = (vl_num == 1).fillna(False)
    verif_from_str = vl_str.isin(["I", "1", "LEVEL I", "LEVEL 1"]).fillna(False)
    df["verification_level_1"] = (verif_from_num | verif_from_str).astype(int)

    ha_raw = df.get("highest_activation_raw", pd.Series([pd.NA] * len(df)))
    ha_num = pd.to_numeric(ha_raw, errors="coerce")
    df["highest_activation_highest"] = (ha_num == 1).fillna(False).astype(int)

    return df


# --------------------------------------------------------------------------
# E. Frozen integer-weight scoring
# --------------------------------------------------------------------------
def compute_pat_tbi(df, weights_csv):
    """Apply the frozen integer-weight score from table_pat_weights.csv."""
    w = pd.read_csv(weights_csv)
    w = w.dropna(subset=["Variable"])
    w = w[w["Variable"].astype(str).str.strip() != ""].copy()
    w["Points"] = pd.to_numeric(w["Points"], errors="coerce").fillna(0).astype(int)
    w = w[w["Points"] != 0].copy()
    if len(w) == 0:
        raise RuntimeError("weights table has zero usable predictors")

    score = np.zeros(len(df), dtype=np.float64)
    used = []
    for _, row in w.iterrows():
        var = str(row["Variable"]).strip()
        pts = int(row["Points"])
        if var not in df.columns:
            raise RuntimeError(f"predictor '{var}' missing from df")
        score += pts * df[var].astype(np.float64).values
        used.append((var, pts))
    return score.astype(np.int64), used


# --------------------------------------------------------------------------
# F. Validation metrics
# --------------------------------------------------------------------------
def bootstrap_auc_ci_subsample(y, s, n_boot, sub_size, rng):
    """Percentile bootstrap CI for AUROC (subsampled)."""
    n = len(y)
    eff = min(sub_size, n)
    aucs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=eff)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == eff:
            aucs[b] = np.nan
            continue
        aucs[b] = roc_auc_score(yb, s[idx])
    aucs = aucs[~np.isnan(aucs)]
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)


def safe_logit(p, eps=EPS):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_logistic_recalibration(y, x):
    X = sm.add_constant(np.asarray(x, dtype=float), has_constant="add")
    model = sm.Logit(np.asarray(y, dtype=int), X)
    res = model.fit(disp=False, method="lbfgs", maxiter=200)
    return float(res.params[0]), float(res.params[1]), res.predict(X)


def calibration_slope_via_logit(y, prob):
    lp = safe_logit(prob)
    intercept, slope, _ = fit_logistic_recalibration(y, lp)
    return intercept, slope


def citl(y, prob):
    return float(np.mean(prob) - np.mean(y))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample", type=int, default=0,
        help="Optional: randomly sample N rows from the cohort for smoke test")
    args = parser.parse_args()

    if not PAT_WEIGHTS_CSV.exists():
        sys.exit(f"FATAL: weights CSV not found at {PAT_WEIGHTS_CSV}")
    if not DUCKDB_PATH.exists():
        sys.exit(f"FATAL: DuckDB not found at {DUCKDB_PATH}")

    banner(f"PAT-TBI Step 11 — All-trauma sensitivity analysis "
           f"(NTDB {YEARS[0]}-{YEARS[-1]})")
    print(f"  DuckDB  : {DUCKDB_PATH}", flush=True)
    print(f"  Weights : {PAT_WEIGHTS_CSV}", flush=True)

    log = {}

    # ----------------------------------------------------------------------
    # A. Pull base all-trauma cohort
    # ----------------------------------------------------------------------
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    cohort = fetch_base_cohort(con, log)

    cohort["inc_key"] = pd.to_numeric(cohort["inc_key"], errors="coerce").astype("Int64")
    cohort["PUF_YEAR"] = pd.to_numeric(cohort["PUF_YEAR"], errors="coerce").astype("Int64")

    # ----------------------------------------------------------------------
    # Apply Step-1 exclusions in order
    # ----------------------------------------------------------------------
    banner("Apply exclusion filters (mirrors Step 1)")

    pre = cohort["PREHOSPITAL_ARREST"].astype("string").str.upper().str.strip()
    cohort = cohort[~(pre == "Y")].copy()
    log_step("exclude_prehospital_arrest", len(cohort), log)

    edc = cohort["ED_DISPOSITION_CODE"].astype("string").str.upper().str.strip()
    cohort = cohort[~edc.isin([c.upper() for c in ED_DISP_EXPIRED])].copy()
    log_step("exclude_ed_disposition_expired", len(cohort), log)

    edc = cohort["ED_DISPOSITION_CODE"].astype("string").str.upper().str.strip()
    cohort = cohort[~edc.isin([c.upper() for c in ED_DISP_TRANSFERRED_OUT])].copy()
    log_step("exclude_transferred_out", len(cohort), log)

    edc = cohort["ED_DISPOSITION_CODE"].astype("string").str.strip()
    cohort = cohort[~(edc.isna() | (edc == "") | (edc.str.upper() == "NONE"))].copy()
    log_step("exclude_missing_ed_disposition", len(cohort), log)

    cohort = cohort.reset_index(drop=True)

    # ----------------------------------------------------------------------
    # B. Comorbidities
    # ----------------------------------------------------------------------
    cc = fetch_comorbidities(con, cohort)
    cohort = cohort.merge(cc, on=["PUF_YEAR", "inc_key"], how="left", validate="1:1")
    for col in CC_COLS:
        cohort[col] = pd.to_numeric(cohort[col], errors="coerce").fillna(0).astype(int)

    con.close()

    # ----------------------------------------------------------------------
    # C. Hospital factor extras from raw PUF
    # ----------------------------------------------------------------------
    extras = fetch_raw_puf_extras(cohort)
    extra_cols = ["teaching_status_raw", "verification_level_raw",
                  "highest_activation_raw"]
    if len(extras):
        cohort = cohort.merge(
            extras, on=["PUF_YEAR", "inc_key"], how="left", validate="1:1"
        )
    for col in extra_cols:
        if col not in cohort.columns:
            cohort[col] = pd.NA
        cohort[col] = cohort[col].astype("string")

    # Drop scratch columns no longer needed
    cohort = cohort.drop(columns=[
        c for c in ("PREHOSPITAL_ARREST", "ED_DISPOSITION_CODE")
        if c in cohort.columns
    ])

    cohort["cause_e_code"] = cohort["cause_e_code"].astype("string")
    cohort["PLATELETS4HOURS"] = pd.to_numeric(cohort["PLATELETS4HOURS"], errors="coerce")

    # Optional smoke-test downsample
    if args.sample and args.sample < len(cohort):
        cohort = cohort.sample(n=args.sample, random_state=SEED).reset_index(drop=True)
        print(f"  [SMOKE TEST] downsampled to {len(cohort):,}", flush=True)

    banner("Cohort year breakdown (all-trauma, post-exclusion)")
    print(cohort["PUF_YEAR"].value_counts().sort_index().to_string(), flush=True)
    print(f"  total: {len(cohort):,}", flush=True)

    # ----------------------------------------------------------------------
    # D. Engineer features + derive primary outcome
    # ----------------------------------------------------------------------
    banner("Engineering ED-rapid binary predictors")
    cohort = engineer_binaries(cohort)

    plt4 = pd.to_numeric(cohort["PLATELETS4HOURS"], errors="coerce")
    cohort["received_platelets_4h"] = (plt4.fillna(0) > 0).astype(int)

    # Validate weight predictors are clean 0/1 ints
    weights_vars = pd.read_csv(PAT_WEIGHTS_CSV)
    weights_vars = weights_vars.dropna(subset=["Variable"])
    weights_vars = weights_vars[
        weights_vars["Variable"].astype(str).str.strip() != ""
    ]
    weights_vars["Points"] = pd.to_numeric(
        weights_vars["Points"], errors="coerce").fillna(0).astype(int)
    weights_vars = weights_vars[weights_vars["Points"] != 0]
    wvars = [str(v).strip() for v in weights_vars["Variable"].tolist()]

    missing = [v for v in wvars if v not in cohort.columns]
    assert not missing, f"missing weight vars: {missing}"
    for v in wvars:
        col = cohort[v]
        n_nan = int(col.isna().sum())
        assert n_nan == 0, f"{v} has {n_nan} NaN"
        # shock_index is continuous; all other LASSO vars are 0/1
        if v == "shock_index":
            cohort[v] = pd.to_numeric(col, errors="coerce")
            assert cohort[v].isna().sum() == 0, "shock_index has NaN"
        else:
            uniq = set(pd.unique(col.dropna()))
            assert uniq.issubset({0, 1}), f"{v} not 0/1: {uniq}"
            cohort[v] = col.astype(int)
    assert cohort["received_platelets_4h"].isna().sum() == 0
    assert set(pd.unique(cohort["received_platelets_4h"])).issubset({0, 1})
    cohort["received_platelets_4h"] = cohort["received_platelets_4h"].astype(int)
    print(f"  OK  {len(wvars)} weight vars present, no NaN", flush=True)

    # ----------------------------------------------------------------------
    # E. Apply frozen integer-point score
    # ----------------------------------------------------------------------
    banner("Applying frozen PAT-TBI integer weights")
    score, used = compute_pat_tbi(cohort, PAT_WEIGHTS_CSV)
    cohort["pat_tbi_score"] = score
    assert cohort["pat_tbi_score"].isna().sum() == 0
    assert cohort["pat_tbi_score"].dtype == np.int64
    print(f"  predictors used: {len(used)}", flush=True)
    print(f"  score range: [{cohort['pat_tbi_score'].min()}, "
          f"{cohort['pat_tbi_score'].max()}]  "
          f"mean={cohort['pat_tbi_score'].mean():.2f}", flush=True)

    # Optional cache of the scored frame for downstream exploration
    cache_cols = ["PUF_YEAR", "inc_key", "AGE_NUMBER", "SEX",
                  "received_platelets_4h", "pat_tbi_score"] + wvars
    cache_cols = [c for c in cache_cols if c in cohort.columns]
    cohort[cache_cols].to_parquet(OUT_PARQUET, index=False)
    print(f"  wrote cache parquet: {OUT_PARQUET}", flush=True)

    # ----------------------------------------------------------------------
    # F. Discrimination + calibration on the all-trauma cohort
    # ----------------------------------------------------------------------
    banner("All-trauma sensitivity — discrimination + calibration")
    rng = np.random.default_rng(SEED)
    y = cohort["received_platelets_4h"].to_numpy(dtype=np.int64)
    s = cohort["pat_tbi_score"].to_numpy(dtype=np.float64)

    n = len(y)
    ev = int(y.sum())
    er = ev / n if n > 0 else float("nan")

    if n == 0 or ev == 0 or ev == n:
        sys.exit(f"FATAL: degenerate cohort -- N={n:,}, events={ev:,}")

    auc = float(roc_auc_score(y, s))
    lo, hi = bootstrap_auc_ci_subsample(y, s, N_BOOT, BOOT_SUBSAMPLE, rng)

    try:
        _a, _b, prob = fit_logistic_recalibration(y, s)
        prob = np.asarray(prob, dtype=np.float64)
        citl_v = citl(y, prob)
        _i, slope = calibration_slope_via_logit(y, prob)
        brier = float(brier_score_loss(y, prob))
    except Exception as exc:
        print(f"  WARN calibration failed: {exc}", flush=True)
        citl_v, slope, brier = float("nan"), float("nan"), float("nan")

    print(f"  N           : {n:,}", flush=True)
    print(f"  N_events    : {ev:,}  ({100*er:.3f}%)", flush=True)
    print(f"  AUROC       : {auc:.4f}  [{lo:.4f}, {hi:.4f}]", flush=True)
    print(f"  Brier       : {brier:.6f}", flush=True)
    print(f"  CITL        : {citl_v:+.6f}", flush=True)
    print(f"  Calib slope : {slope:.4f}", flush=True)

    # ----------------------------------------------------------------------
    # G. Output table
    # ----------------------------------------------------------------------
    banner("Assembling output table")
    row = {
        "Stratum": "all-trauma (NTDB 2020-2024)",
        "N": n,
        "N_events": ev,
        "Event_Rate": round(er, 6),
        "AUROC": round(auc, 4),
        "CI_Low": round(lo, 4),
        "CI_High": round(hi, 4),
        "Brier": round(brier, 6) if not np.isnan(brier) else float("nan"),
        "CITL": round(citl_v, 6) if not np.isnan(citl_v) else float("nan"),
        "Calibration_Slope": round(slope, 4) if not np.isnan(slope) else float("nan"),
    }
    out = pd.DataFrame([row], columns=[
        "Stratum", "N", "N_events", "Event_Rate",
        "AUROC", "CI_Low", "CI_High",
        "Brier", "CITL", "Calibration_Slope",
    ])
    out.to_csv(OUT_TABLE, index=False)
    print(f"  wrote {OUT_TABLE}", flush=True)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(out.to_string(index=False), flush=True)

    # ----------------------------------------------------------------------
    # H. Figure — single horizontal AUROC bar with CI
    # ----------------------------------------------------------------------
    banner("Figure — AUROC (95% bootstrap CI) on the all-trauma cohort")
    fig, ax = plt.subplots(figsize=(7.5, 2.2))
    label = f"all-trauma (NTDB 2020-2024)\nN={n:,}, events={ev:,}"
    err_lo = auc - lo
    err_hi = hi - auc
    ax.barh([0], [auc], xerr=[[err_lo], [err_hi]],
            color="#4477AA", edgecolor="black", capsize=4, height=0.45)
    ax.text(auc + err_hi + 0.005, 0,
            f"{auc:.3f}  [{lo:.3f}, {hi:.3f}]",
            va="center", ha="left", fontsize=9)
    ax.set_yticks([0])
    ax.set_yticklabels([label])
    ax.invert_yaxis()
    ax.set_xlabel("AUROC (95% bootstrap CI)")
    ax.set_xlim(0.5, 1.0)
    ax.axvline(0.5, color="gray", ls=":", lw=0.8)
    ax.set_title("PAT-TBI score — all-trauma sensitivity (NTDB 2020-2024)")
    plt.tight_layout()
    plt.savefig(OUT_FIG_PDF)
    plt.savefig(OUT_FIG_PNG, dpi=300)
    plt.close(fig)
    print(f"  wrote {OUT_FIG_PDF}", flush=True)
    print(f"  wrote {OUT_FIG_PNG}", flush=True)

    # Persist the exclusion log alongside the table for traceability
    log_path = DERIVED / "all_trauma_exclusions.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  wrote exclusion log: {log_path}", flush=True)

    print("\nStep 11 complete.", flush=True)


if __name__ == "__main__":
    main()
