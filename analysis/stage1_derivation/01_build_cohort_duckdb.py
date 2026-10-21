"""
PAT-TBI v4 — Step 1: Build adult-TBI cohort directly from NTDB DuckDB
================================================================================

Builds a fresh, fully reproducible adult-TBI cohort for the PAT-TBI v4 refit on
NTDB 2020-2024 by querying the institutional DuckDB instance directly. No prior
derived artifact is consumed.

Data sources
------------
  DuckDB:  _DATA/NTDB/ntdb_database/ntdb_research.duckdb
    - v_patients_2019_2024  : per-patient flat view (filtered to 2019-2024)
    - raw_icd_diagnosis     : long-format ICD-10 diagnosis codes
    - raw_preexisting       : long-format preexisting conditions
    - raw_ais_diagnosis     : long-format AIS injury list
    - lookup_ais            : (ais_predot, ais_severity) -> iss_region (1=Head)

  Raw PUF (only for WHOLEBLOOD4HOURS, which is not in DuckDB):
    _DATA/NTDB/PUF/PUF AY {yr}/CSV/PUF_TRAUMA.csv  for yr in 2020..2024

Cohort window
-------------
  PUF_YEAR in {2020, 2021, 2022, 2023, 2024}; AGE_NUMBER >= 18.
  TBI = any ICD-10 S06* diagnosis on the encounter.

Exclusions (logged in order to cohort_exclusions.json)
------------------------------------------------------
  1. start_2020_2024_age_18plus_from_duckdb
  2. tbi_S06_diagnosis
  3. exclude_prehospital_arrest        (PREHOSPITAL_ARREST == 'Y')
  4. exclude_ed_disposition_expired    (ED_DISPOSITION_CODE in {'MOR'})
  5. exclude_transferred_out           (ED_DISPOSITION_CODE in {'5A_S','5BMS'})
  6. exclude_missing_ed_disposition    (ED_DISPOSITION_CODE IS NULL)

  ED_DISPOSITION_CODE encoding (verified empirically, 2020-2024):
    4A_S, 4B_IM, 4F_S, 4CNI, 4ESI, CDU  -> admitted to this hospital (kept)
    5A_S, 5BMS                          -> transferred out (excluded)
    HOME, AMA                           -> discharged from ED (kept)
    MOR                                 -> expired in ED (excluded)
    None                                -> missing (excluded)

Outputs
-------
  v4/derived/cohort_raw.parquet        : final cohort (schema below)
  v4/derived/cohort_exclusions.json    : {step_name: N_remaining}

Output schema (consumed unchanged by 02_engineer_features.py)
-------------------------------------------------------------
  PUF_YEAR (uppercase, int)
  inc_key (lowercase, int)
  AGE_NUMBER, SEX, RACE, ETHNICITY, PAYMENT_SOURCE
  GCS2, MOTOR2, EYE2, VERBAL2, TQIP_TBI_PUPIL_RESPONSE
  SBP2, P2, RR2, OX2
  temperature_raw            (renamed from TEMPS2)
  cause_e_code               (renamed from CAUSE_E_CODES10, string)
  HOSPITAL_TRANSFER
  teaching_status_raw        (merged from raw PUF_TRAUMA.csv per year; TEACHINGSTATUS)
  verification_level_raw     (merged from raw PUF_TRAUMA.csv per year; VERIFICATIONLEVEL)
  highest_activation_raw     (merged from raw PUF_TRAUMA.csv per year; HIGHESTACTIVATION)
  ISS
  cc_alcohol cc_anticoag cc_bleeding cc_cancer cc_chf cc_cirrhosis cc_ckd
  cc_copd cc_cva cc_dementia cc_dm cc_htn cc_smoker         (13 binary flags)
  PLATELETS4HOURS            (renamed from TQIP_HC_PLATE_4HR)
  WHOLEBLOOD4HOURS           (merged from raw PUF_TRAUMA.csv per year)
  max_ais_head, max_ais_other, major_extracranial   (derived from raw_ais_diagnosis)

Schema notes (inline, for downstream subagents)
-----------------------------------------------
  * teaching_status_raw / verification_level_raw / highest_activation_raw are
    NOT exposed in v_patients_2019_2024. They are loaded from each year's raw
    PUF_TRAUMA.csv (TEACHINGSTATUS, VERIFICATIONLEVEL, HIGHESTACTIVATION
    columns) in the same per-year semi-join pass that loads WHOLEBLOOD4HOURS
    so that the LASSO candidates teaching_status_binary, verification_level_1,
    and highest_activation_highest are properly populated downstream.
  * WHOLEBLOOD4HOURS is loaded from raw PUF_TRAUMA.csv per year because it is
    NOT present in DuckDB (verified). It is available in all of 2020-2024.
  * DuckDB TQIP_HC_BLOOD_4HR is RBC (BLOOD4HOURS in raw PUF), NOT whole blood.

Seed: 42
"""
import argparse
import json
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
import os as _os, sys as _sys  # path handling: make analysis/config.py importable
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import config  # noqa: E402  (analysis/config.py: PAT_TBI_WORKDIR, PAT_TBI_NTDB_DUCKDB)
PAT_V4 = config.V4_DIR
DERIVED = PAT_V4 / "derived"
DERIVED.mkdir(parents=True, exist_ok=True)

DUCKDB_PATH = config.DUCKDB_PATH
PUF_TRAUMA_TPL = config.PUF_TRAUMA_TPL

OUT_PARQUET = DERIVED / "cohort_raw.parquet"
OUT_EXCLUSIONS = DERIVED / "cohort_exclusions.json"

YEARS = [2020, 2021, 2022, 2023, 2024]

# --------------------------------------------------------------------------
# Comorbidity code map (NTDB PUF Variable Formats, code -> short flag)
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
# ED_DISPOSITION_CODE classification (verified against 2020-2024 distribution)
# --------------------------------------------------------------------------
ED_DISP_TRANSFERRED_OUT = ("5A_S", "5BMS")
ED_DISP_EXPIRED = ("MOR",)


def banner(msg):
    line = "=" * 78
    print(f"\n{line}\n{msg}\n{line}", flush=True)


def log_step(name, n, log):
    log[name] = int(n)
    print(f"  [{name}] N = {n:,}", flush=True)


# --------------------------------------------------------------------------
# Step A: pull base cohort from DuckDB (single SQL, no full-table fetch)
# --------------------------------------------------------------------------
def fetch_base_cohort(con, log):
    """SELECT base columns from v_patients_2019_2024, restricted to age 18+,
    years 2020-2024, with TBI (S06*) diagnosis (semi-join on raw_icd_diagnosis).

    Exclusion filters are applied subsequently in pandas so that we can log
    exact N at each step.
    """
    banner("Pull base cohort from DuckDB (2020-2024, age 18+)")

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

    # Semi-join via temp table with distinct (inc_key, puf_year) for any S06* dx.
    print("  building S06 TBI key set...", flush=True)
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE _tbi_keys AS
        SELECT DISTINCT inc_key, puf_year
        FROM raw_icd_diagnosis
        WHERE puf_year BETWEEN 2020 AND 2024
          AND icd_diagnosis_code LIKE 'S06%'
        """
    )
    n_tbi_keys = con.execute("SELECT COUNT(*) FROM _tbi_keys").fetchone()[0]
    print(f"  distinct S06 (inc_key, puf_year) rows: {n_tbi_keys:,}", flush=True)

    print("  pulling cohort frame...", flush=True)
    df = con.execute(
        """
        SELECT
            v.inc_key                   AS inc_key,
            v.puf_year                  AS PUF_YEAR,
            v.AGE_NUMBER                AS AGE_NUMBER,
            v.SEX                       AS SEX,
            v.RACE                      AS RACE,
            v.ETHNICITY                 AS ETHNICITY,
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
        INNER JOIN _tbi_keys t
          ON v.inc_key = t.inc_key AND v.puf_year = t.puf_year
        WHERE v.puf_year BETWEEN 2020 AND 2024
          AND v.AGE_NUMBER IS NOT NULL
          AND v.AGE_NUMBER >= 18
        """
    ).fetchdf()
    log_step("tbi_S06_diagnosis", len(df), log)
    return df


# --------------------------------------------------------------------------
# Step B: comorbidity pivot (DuckDB-side aggregation, only for cohort keys)
# --------------------------------------------------------------------------
def fetch_comorbidities(con, cohort_keys_df):
    """Pivot raw_preexisting wide for the cohort keys, returning one row per
    (inc_key, puf_year) with 13 binary cc_* columns. Missing rows -> all-zero
    (null-biased convention).
    """
    banner("Pivot raw_preexisting -> 13 comorbidity flags (cohort keys only)")

    # Register cohort keys as a DuckDB temp view for fast semi-join
    keys = cohort_keys_df[["inc_key", "PUF_YEAR"]].rename(columns={"PUF_YEAR": "puf_year"}).copy()
    keys["inc_key"] = pd.to_numeric(keys["inc_key"], errors="coerce").astype("Int64")
    keys["puf_year"] = pd.to_numeric(keys["puf_year"], errors="coerce").astype("Int32")
    con.register("_cohort_keys_df", keys)
    con.execute("CREATE OR REPLACE TEMP TABLE _cohort_keys AS SELECT inc_key, puf_year FROM _cohort_keys_df")

    code_list_sql = ",".join(str(k) for k in CODE_TO_CC.keys())
    print(f"  aggregating preexisting answers (codes: {sorted(CODE_TO_CC.keys())})", flush=True)
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
    # Ensure all 13 columns exist
    for cc in CC_COLS:
        if cc not in wide.columns:
            wide[cc] = 0
    wide = wide[["inc_key", "puf_year"] + CC_COLS]
    wide = wide.rename(columns={"puf_year": "PUF_YEAR"})
    wide["inc_key"] = pd.to_numeric(wide["inc_key"], errors="coerce").astype("Int64")
    wide["PUF_YEAR"] = pd.to_numeric(wide["PUF_YEAR"], errors="coerce").astype("Int64")
    return wide


# --------------------------------------------------------------------------
# Step C: AIS-derived columns
# --------------------------------------------------------------------------
def fetch_ais_features(con, cohort_keys_df):
    """Compute (max_ais_head, max_ais_other, major_extracranial) per cohort
    encounter using raw_ais_diagnosis joined to lookup_ais. iss_region == 1
    encodes Head/Neck.
    """
    banner("Derive max_ais_head / max_ais_other / major_extracranial from AIS")

    # _cohort_keys is already registered by fetch_comorbidities, but be defensive.
    if not con.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name='_cohort_keys'").fetchone()[0]:
        keys = cohort_keys_df[["inc_key", "PUF_YEAR"]].rename(columns={"PUF_YEAR": "puf_year"}).copy()
        keys["inc_key"] = pd.to_numeric(keys["inc_key"], errors="coerce").astype("Int64")
        keys["puf_year"] = pd.to_numeric(keys["puf_year"], errors="coerce").astype("Int32")
        con.register("_cohort_keys_df", keys)
        con.execute("CREATE OR REPLACE TEMP TABLE _cohort_keys AS SELECT inc_key, puf_year FROM _cohort_keys_df")

    df = con.execute(
        """
        SELECT a.inc_key,
               a.puf_year,
               MAX(CASE WHEN l.iss_region = 1 THEN a.ais_severity END)  AS max_ais_head,
               MAX(CASE WHEN l.iss_region IS NOT NULL AND l.iss_region <> 1
                        THEN a.ais_severity END)                         AS max_ais_other,
               MAX(CASE WHEN l.iss_region IS NOT NULL AND l.iss_region <> 1
                             AND a.ais_severity >= 3
                        THEN 1 ELSE 0 END)                               AS major_extracranial
        FROM raw_ais_diagnosis a
        INNER JOIN _cohort_keys c
          ON a.inc_key = c.inc_key AND a.puf_year = c.puf_year
        LEFT JOIN lookup_ais l
          ON a.ais_predot = l.ais_predot AND a.ais_severity = l.ais_severity
        WHERE a.puf_year BETWEEN 2020 AND 2024
        GROUP BY a.inc_key, a.puf_year
        """
    ).fetchdf()
    df = df.rename(columns={"puf_year": "PUF_YEAR"})
    df["inc_key"] = pd.to_numeric(df["inc_key"], errors="coerce").astype("Int64")
    df["PUF_YEAR"] = pd.to_numeric(df["PUF_YEAR"], errors="coerce").astype("Int64")
    print(f"  AIS aggregate rows: {len(df):,}", flush=True)
    return df


# --------------------------------------------------------------------------
# Step D: WHOLEBLOOD4HOURS + hospital factor columns from raw PUF (not in DuckDB)
# --------------------------------------------------------------------------
# Mapping of raw PUF_TRAUMA columns -> output column names in the parquet schema.
RAW_PUF_COLMAP = {
    "WHOLEBLOOD4HOURS": "WHOLEBLOOD4HOURS",
    "TEACHINGSTATUS": "teaching_status_raw",
    "VERIFICATIONLEVEL": "verification_level_raw",
    "HIGHESTACTIVATION": "highest_activation_raw",
}


def fetch_raw_puf_extras(cohort_keys_df):
    """Load WHOLEBLOOD4HOURS and the three hospital-factor columns
    (TEACHINGSTATUS, VERIFICATIONLEVEL, HIGHESTACTIVATION) from each year's raw
    PUF_TRAUMA.csv, filtered to the cohort keys via in-memory semi-join, and
    return a long frame keyed by (inc_key, PUF_YEAR).

    Single per-year pass through PUF_TRAUMA.csv (low-memory: usecols restricted
    to inc_key + the 4 columns we need).
    """
    banner("Load WHOLEBLOOD4HOURS + hospital factors from raw PUF_TRAUMA.csv "
           "(per year)")

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
        df = pd.read_csv(path, usecols=usecols, low_memory=False)
        df["inc_key"] = pd.to_numeric(df["inc_key"], errors="coerce").astype("Int64")
        df = df[df["inc_key"].isin(keys_yr)].copy()
        df["PUF_YEAR"] = int(yr)

        # Backfill any missing raw columns with NA, then rename to schema names
        for raw_c in raw_cols:
            if raw_c not in df.columns:
                df[raw_c] = pd.NA
        df = df.rename(columns=RAW_PUF_COLMAP)

        n_match = len(df)
        wb = RAW_PUF_COLMAP["WHOLEBLOOD4HOURS"]
        ts = RAW_PUF_COLMAP["TEACHINGSTATUS"]
        vl = RAW_PUF_COLMAP["VERIFICATIONLEVEL"]
        ha = RAW_PUF_COLMAP["HIGHESTACTIVATION"]
        print(
            f"  [{yr}] cohort-matched PUF rows: {n_match:,}  | "
            f"WB non-null: {int(df[wb].notna().sum()):,}  | "
            f"teaching non-null: {int(df[ts].notna().sum()):,}  | "
            f"verif non-null: {int(df[vl].notna().sum()):,}  | "
            f"activation non-null: {int(df[ha].notna().sum()):,}",
            flush=True,
        )
        out.append(df[out_cols])

    if not out:
        return pd.DataFrame(columns=out_cols)
    combined = pd.concat(out, ignore_index=True)
    combined["PUF_YEAR"] = pd.to_numeric(combined["PUF_YEAR"], errors="coerce").astype("Int64")
    return combined


# Back-compat alias (in case any external script imports the old name)
fetch_whole_blood = fetch_raw_puf_extras


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Optional: randomly sample N rows from the final cohort for smoke test",
    )
    args = parser.parse_args()

    log = {}

    if not DUCKDB_PATH.exists():
        sys.exit(f"FATAL: DuckDB not found at {DUCKDB_PATH}")

    banner(f"PAT-TBI v4 Step 1: build cohort from DuckDB ({YEARS[0]}-{YEARS[-1]})")
    print(f"  DuckDB: {DUCKDB_PATH}", flush=True)

    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)

    # ----------------------------------------------------------------------
    # A. Base cohort (years, age, TBI)
    # ----------------------------------------------------------------------
    cohort = fetch_base_cohort(con, log)

    # Coerce keys + numeric types
    cohort["inc_key"] = pd.to_numeric(cohort["inc_key"], errors="coerce").astype("Int64")
    cohort["PUF_YEAR"] = pd.to_numeric(cohort["PUF_YEAR"], errors="coerce").astype("Int64")

    # ----------------------------------------------------------------------
    # Apply exclusion filters in order, logging N at each step
    # ----------------------------------------------------------------------
    banner("Apply exclusion filters")

    # 3. Prehospital arrest = 'Y'
    pre = cohort["PREHOSPITAL_ARREST"].astype("string").str.upper().str.strip()
    cohort = cohort[~(pre == "Y")].copy()
    log_step("exclude_prehospital_arrest", len(cohort), log)

    # 4. ED_DISPOSITION_CODE indicates expired in ED ('MOR')
    edc = cohort["ED_DISPOSITION_CODE"].astype("string").str.upper().str.strip()
    cohort = cohort[~edc.isin([c.upper() for c in ED_DISP_EXPIRED])].copy()
    log_step("exclude_ed_disposition_expired", len(cohort), log)

    # 5. ED_DISPOSITION_CODE indicates transferred out
    edc = cohort["ED_DISPOSITION_CODE"].astype("string").str.upper().str.strip()
    cohort = cohort[~edc.isin([c.upper() for c in ED_DISP_TRANSFERRED_OUT])].copy()
    log_step("exclude_transferred_out", len(cohort), log)

    # 6. Missing ED disposition (None / empty)
    edc = cohort["ED_DISPOSITION_CODE"].astype("string").str.strip()
    cohort = cohort[~(edc.isna() | (edc == "") | (edc.str.upper() == "NONE"))].copy()
    log_step("exclude_missing_ed_disposition", len(cohort), log)

    # ----------------------------------------------------------------------
    # B. Comorbidity flags
    # ----------------------------------------------------------------------
    cc = fetch_comorbidities(con, cohort)
    cohort = cohort.merge(cc, on=["PUF_YEAR", "inc_key"], how="left", validate="1:1")
    for col in CC_COLS:
        cohort[col] = pd.to_numeric(cohort[col], errors="coerce").fillna(0).astype(int)

    # ----------------------------------------------------------------------
    # C. AIS-derived columns
    # ----------------------------------------------------------------------
    ais = fetch_ais_features(con, cohort)
    cohort = cohort.merge(ais, on=["PUF_YEAR", "inc_key"], how="left", validate="1:1")
    for col in ("max_ais_head", "max_ais_other", "major_extracranial"):
        cohort[col] = pd.to_numeric(cohort[col], errors="coerce").fillna(0).astype(int)

    # Close DuckDB connection before opening raw CSVs (memory hygiene)
    con.close()

    # ----------------------------------------------------------------------
    # D. WHOLEBLOOD4HOURS + hospital factors (TEACHINGSTATUS, VERIFICATIONLEVEL,
    #    HIGHESTACTIVATION) from raw PUF -- pulled in one pass per year
    # ----------------------------------------------------------------------
    extras = fetch_raw_puf_extras(cohort)
    extra_cols = ["WHOLEBLOOD4HOURS", "teaching_status_raw",
                  "verification_level_raw", "highest_activation_raw"]
    if len(extras):
        cohort = cohort.merge(
            extras, on=["PUF_YEAR", "inc_key"], how="left", validate="1:1"
        )
    else:
        for col in extra_cols:
            cohort[col] = pd.NA

    # Ensure all extras are present even if a year was missing
    for col in extra_cols:
        if col not in cohort.columns:
            cohort[col] = pd.NA

    cohort["WHOLEBLOOD4HOURS"] = pd.to_numeric(cohort["WHOLEBLOOD4HOURS"], errors="coerce")
    # Hospital factor columns are categorical/string codes -> keep as nullable string
    for col in ("teaching_status_raw", "verification_level_raw", "highest_activation_raw"):
        cohort[col] = cohort[col].astype("string")

    # cause_e_code -> string (already pulled from CAUSE_E_CODES10)
    cohort["cause_e_code"] = cohort["cause_e_code"].astype("string")

    # PLATELETS4HOURS already aliased from TQIP_HC_PLATE_4HR; coerce numeric
    cohort["PLATELETS4HOURS"] = pd.to_numeric(cohort["PLATELETS4HOURS"], errors="coerce")

    # Drop scratch columns not part of the schema contract
    cohort = cohort.drop(columns=[c for c in ("PREHOSPITAL_ARREST", "ED_DISPOSITION_CODE")
                                  if c in cohort.columns])

    # ----------------------------------------------------------------------
    # F. Reorder columns to match documented schema contract
    # ----------------------------------------------------------------------
    schema_order = (
        ["PUF_YEAR", "inc_key",
         "AGE_NUMBER", "SEX", "RACE", "ETHNICITY", "PAYMENT_SOURCE",
         "GCS2", "MOTOR2", "EYE2", "VERBAL2", "TQIP_TBI_PUPIL_RESPONSE",
         "SBP2", "P2", "RR2", "OX2",
         "temperature_raw", "cause_e_code",
         "HOSPITAL_TRANSFER",
         "teaching_status_raw", "verification_level_raw", "highest_activation_raw",
         "ISS"]
        + CC_COLS
        + ["PLATELETS4HOURS", "WHOLEBLOOD4HOURS",
           "max_ais_head", "max_ais_other", "major_extracranial"]
    )
    missing = [c for c in schema_order if c not in cohort.columns]
    if missing:
        sys.exit(f"FATAL: required schema columns missing from cohort: {missing}")
    cohort = cohort[schema_order].copy()

    # ----------------------------------------------------------------------
    # G. Optional sample for smoke testing
    # ----------------------------------------------------------------------
    if args.sample and args.sample < len(cohort):
        cohort = cohort.sample(n=args.sample, random_state=SEED).reset_index(drop=True)
        log_step("smoke_test_sample", len(cohort), log)
        print(f"  [SMOKE TEST] downsampled cohort to {len(cohort):,}", flush=True)

    # ----------------------------------------------------------------------
    # Summaries
    # ----------------------------------------------------------------------
    banner("Summary - cohort year breakdown")
    print(cohort["PUF_YEAR"].value_counts().sort_index().to_string(), flush=True)

    banner("Summary - comorbidity prevalence")
    n = max(len(cohort), 1)
    for cc_col in CC_COLS:
        k = int(cohort[cc_col].sum())
        print(f"  {cc_col:<16s}  n={k:>9,}  ({100 * k / n:5.2f}%)", flush=True)

    banner("Summary - transfusion field coverage")
    for col in ("PLATELETS4HOURS", "WHOLEBLOOD4HOURS"):
        numeric = pd.to_numeric(cohort[col], errors="coerce")
        miss = int(numeric.isna().sum())
        zero = int((numeric == 0).sum())
        pos = int((numeric > 0).sum())
        print(
            f"  {col:<20s}  miss={miss:>9,} ({100 * miss / n:5.2f}%)  "
            f"zero={zero:>9,} ({100 * zero / n:5.2f}%)  "
            f"pos={pos:>9,} ({100 * pos / n:5.2f}%)",
            flush=True,
        )

    banner("Summary - AIS-derived")
    for col in ("max_ais_head", "max_ais_other", "major_extracranial"):
        v = cohort[col]
        print(f"  {col:<22s}  mean={v.mean():.3f}  max={int(v.max())}  "
              f"pct>0={(100 * (v > 0).mean()):5.2f}%", flush=True)

    banner("Summary - hospital factor columns (from raw PUF_TRAUMA.csv)")
    for col in ("teaching_status_raw", "verification_level_raw",
                "highest_activation_raw"):
        s = cohort[col]
        nn = int(s.notna().sum())
        print(
            f"  {col:<26s}  non-null={nn:>9,} ({100 * nn / n:5.2f}%)  "
            f"unique={int(s.nunique(dropna=True))}",
            flush=True,
        )
        # Show top value counts (capped) so reviewers can sanity-check coding
        vc = s.value_counts(dropna=True).head(6)
        for val, k in vc.items():
            print(f"      {str(val):<20s}  n={int(k):>9,}  ({100 * int(k) / n:5.2f}%)",
                  flush=True)

    # ----------------------------------------------------------------------
    # Write outputs
    # ----------------------------------------------------------------------
    banner("Writing outputs")
    print(f"  parquet : {OUT_PARQUET}", flush=True)
    cohort.to_parquet(OUT_PARQUET, index=False)
    print(f"            wrote {cohort.shape[0]:,} rows x {cohort.shape[1]} cols",
          flush=True)

    print(f"  log     : {OUT_EXCLUSIONS}", flush=True)
    with open(OUT_EXCLUSIONS, "w") as f:
        json.dump(log, f, indent=2)
    print("            exclusion log:", json.dumps(log, indent=2), flush=True)

    print("\nPAT-TBI v4 Step 1 complete.")


if __name__ == "__main__":
    main()
