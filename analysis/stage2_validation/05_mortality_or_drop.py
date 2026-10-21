"""
PAT-TBI v6 — Phase 0, finding #8: does 20,033 in-hospital deaths trace?
================================================================================

The v5 abstract states: "The analytic cohort included 623,587 adult TBI
patients, of whom 20,033 died in the hospital and 13,473 received platelets
within 4 hours of ED arrival."

13,473 traces exactly (8,982 derivation events + 4,491 validation events).
20,033 appears in no v4 table, log or derived artefact, and no derived parquet
carries a mortality column. This script goes back to the registry and measures
it against the same encounter keys.

The plan is explicit: if the registry does not reproduce 20,033, the claim is
deleted from the abstract rather than adjusted to fit. This script therefore
reports every plausible death definition it can measure and does NOT choose
whichever one lands closest to 20,033.

Method
------
  1. Load the 623,587 analytic encounter keys from v4/derived/cohort_raw.parquet
     (PUF_YEAR, inc_key). Cross-check against the union of train_inc_keys.csv
     and test_inc_keys.csv, which is the same key set.
  2. Semi-join those keys against the registry through v_patients_2019_2024,
     year-restricted to 2020-2024 (the v4 cohort window). The raw `patients`
     table is never queried unfiltered.
  3. Print the full DC_DISPOSITION_CODE value distribution, nulls included,
     before counting anything.
  4. Count in-hospital death as DC_DISPOSITION_CODE == 'D', which is the rule
     the registry's own v_mortality view applies (verified from its DDL).
  5. Report deaths by year, and deaths among the 4,491 validation platelet
     recipients versus the validation non-recipients.
  6. Report alternative definitions for the response letter: ED disposition
     expired, deaths in the pre-exclusion adult TBI pool for both the 2020-2024
     and the 2019-2022 windows, registry-wide deaths, and whether any
     hospital-events code encodes death.

Run with the framework interpreter:
    PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
    $PY 05_mortality_or_drop.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

import os as _os, sys as _sys  # path handling: make analysis/config.py importable
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import config  # noqa: E402  (analysis/config.py: PAT_TBI_WORKDIR, PAT_TBI_NTDB_DUCKDB)
DB_PATH = config.DUCKDB_PATH

YEARS = (2020, 2021, 2022, 2023, 2024)
V3_YEARS = (2019, 2020, 2021, 2022)   # the older PAT-TBI window, for provenance
ABSTRACT_CLAIM = 20033
DEATH_CODE = "D"

TABLE_OUT = C.V6 / "tables" / "table_v6_mortality.csv"
HANDOFF_OUT = C.V6 / "logs" / "05_HANDOFF.md"


# ---------------------------------------------------------------------------
# 1. The analytic keys
# ---------------------------------------------------------------------------
def load_analytic_keys() -> pd.DataFrame:
    C.banner("Step 1 — the 623,587 analytic encounter keys")

    raw = C.load_cohort_raw(columns=["PUF_YEAR", "inc_key"])
    keys = raw.copy()
    keys["PUF_YEAR"] = pd.to_numeric(keys["PUF_YEAR"], errors="coerce").astype("int64")
    keys["inc_key"] = pd.to_numeric(keys["inc_key"], errors="coerce").astype("int64")

    n_rows = len(keys)
    n_distinct = len(keys.drop_duplicates(subset=["PUF_YEAR", "inc_key"]))
    print(f"  cohort_raw.parquet rows           : {n_rows:,}")
    print(f"  distinct (PUF_YEAR, inc_key) keys : {n_distinct:,}")
    assert n_rows == C.V4_FROZEN["n_cohort"], (
        f"cohort_raw has {n_rows:,} rows, expected {C.V4_FROZEN['n_cohort']:,}")
    assert n_distinct == C.V4_FROZEN["n_cohort"], (
        f"cohort_raw has {n_distinct:,} distinct keys, "
        f"expected {C.V4_FROZEN['n_cohort']:,}")

    # Cross-check: the union of the two split key files must be the same set.
    tr = pd.read_csv(C.TRAIN_KEYS)
    te = pd.read_csv(C.TEST_KEYS)
    print(f"  train_inc_keys.csv rows           : {len(tr):,}")
    print(f"  test_inc_keys.csv rows            : {len(te):,}")
    union = set(zip(pd.to_numeric(tr["PUF_YEAR"]).astype("int64"),
                    pd.to_numeric(tr["inc_key"]).astype("int64"))) | \
            set(zip(pd.to_numeric(te["PUF_YEAR"]).astype("int64"),
                    pd.to_numeric(te["inc_key"]).astype("int64")))
    cohort_set = set(zip(keys["PUF_YEAR"], keys["inc_key"]))
    print(f"  union of the two split key files  : {len(union):,}")
    print(f"  union equals the cohort key set   : {union == cohort_set}")
    assert len(union) == C.V4_FROZEN["n_cohort"]
    assert union == cohort_set, "split key union differs from cohort_raw keys"

    yr = keys["PUF_YEAR"].value_counts().sort_index()
    print("\n  keys by PUF year")
    for y, k in yr.items():
        print(f"    {y}   n = {k:>9,}")
    assert set(yr.index) == set(YEARS), f"unexpected years: {sorted(set(yr.index))}"
    print("\n  GATE PASSED: 623,587 distinct keys, years 2020 to 2024 only.")
    return keys


# ---------------------------------------------------------------------------
# 2. Registry join
# ---------------------------------------------------------------------------
def fetch_disposition(con, keys: pd.DataFrame) -> pd.DataFrame:
    C.banner("Step 2 — join the analytic keys to the registry (year restricted)")

    con.register("_pat_keys_df", keys[["PUF_YEAR", "inc_key"]]
                 .rename(columns={"PUF_YEAR": "puf_year"}))
    con.execute("CREATE OR REPLACE TEMP TABLE _pat_keys AS "
                "SELECT inc_key, puf_year FROM _pat_keys_df")
    n_reg = con.execute("SELECT COUNT(*) FROM _pat_keys").fetchone()[0]
    print(f"  keys registered in DuckDB : {n_reg:,}")

    t0 = time.time()
    df = con.execute(
        f"""
        SELECT v.inc_key                AS inc_key,
               v.puf_year               AS PUF_YEAR,
               v.DC_DISPOSITION_CODE    AS DC_DISPOSITION_CODE,
               v.ED_DISPOSITION_CODE    AS ED_DISPOSITION_CODE,
               v.DISCHARGE_DATE         AS DISCHARGE_DATE
        FROM v_patients_2019_2024 v
        INNER JOIN _pat_keys k
          ON v.inc_key = k.inc_key AND v.puf_year = k.puf_year
        WHERE v.puf_year BETWEEN {min(YEARS)} AND {max(YEARS)}
        """
    ).fetchdf()
    print(f"  registry rows returned    : {len(df):,}   ({time.time() - t0:.1f}s)")

    df["PUF_YEAR"] = pd.to_numeric(df["PUF_YEAR"], errors="coerce").astype("int64")
    df["inc_key"] = pd.to_numeric(df["inc_key"], errors="coerce").astype("int64")
    n_distinct = len(df.drop_duplicates(subset=["PUF_YEAR", "inc_key"]))
    print(f"  distinct keys matched     : {n_distinct:,}")

    n_unmatched = C.V4_FROZEN["n_cohort"] - n_distinct
    if n_unmatched:
        print(f"  UNMATCHED KEYS            : {n_unmatched:,}")
        matched = set(zip(df["PUF_YEAR"], df["inc_key"]))
        miss = [k for k in zip(keys["PUF_YEAR"], keys["inc_key"]) if k not in matched]
        mdf = pd.DataFrame(miss, columns=["PUF_YEAR", "inc_key"])
        print(mdf["PUF_YEAR"].value_counts().sort_index().to_string())
    else:
        print("  UNMATCHED KEYS            : 0  (every analytic key found)")

    if len(df) != n_distinct:
        print(f"  WARNING: {len(df) - n_distinct:,} duplicate registry rows; "
              f"collapsing to one row per key")
        df = df.drop_duplicates(subset=["PUF_YEAR", "inc_key"], keep="first")
    return df, n_distinct, n_unmatched


# ---------------------------------------------------------------------------
# 3. Value distribution, before counting anything
# ---------------------------------------------------------------------------
def show_distribution(df: pd.DataFrame, col: str) -> pd.Series:
    C.banner(f"Step 3 — full {col} distribution over the matched analytic rows")
    vc = df[col].value_counts(dropna=False)
    vc = vc.sort_values(ascending=False)
    n = len(df)
    print(f"  {'value':<14s} {'n':>10s}  {'pct':>7s}")
    for val, k in vc.items():
        shown = "<NULL>" if (val is None or (isinstance(val, float) and np.isnan(val))
                             or pd.isna(val)) else str(val)
        print(f"  {shown:<14s} {k:>10,}  {100.0 * k / n:>6.2f}%")
    print(f"  {'TOTAL':<14s} {n:>10,}  {100.0:>6.2f}%")
    return vc


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    C.start_log("05_mortality_or_drop")
    C.banner("PAT-TBI v6 — finding #8: 20,033 in-hospital deaths, or drop the claim")
    print(f"  abstract claim under test : {ABSTRACT_CLAIM:,} in-hospital deaths")
    print(f"  stated denominator        : {C.V4_FROZEN['n_cohort']:,}")
    print(f"  registry                  : {DB_PATH}")
    print(f"  opened read-only, year restricted to {min(YEARS)} to {max(YEARS)}")

    keys = load_analytic_keys()

    if not DB_PATH.exists():
        sys.exit(f"FATAL: DuckDB not found at {DB_PATH}")
    con = duckdb.connect(str(DB_PATH), read_only=True)

    df, n_matched, n_unmatched = fetch_disposition(con, keys)

    dc = df["DC_DISPOSITION_CODE"].astype("string").str.strip().str.upper()
    ed = df["ED_DISPOSITION_CODE"].astype("string").str.strip().str.upper()
    df["_dc"] = dc
    df["_ed"] = ed

    vc_dc = show_distribution(df, "DC_DISPOSITION_CODE")
    _ = show_distribution(df, "ED_DISPOSITION_CODE")

    # -------------------------------------------------------------------
    # 4. The primary count
    # -------------------------------------------------------------------
    C.banner("Step 4 — in-hospital deaths, discharge disposition D")
    is_death = (dc == DEATH_CODE).fillna(False)
    n_death = int(is_death.sum())
    n_dc_null = int(dc.isna().sum())
    n_dc_known = n_matched - n_dc_null
    pct = 100.0 * n_death / n_matched
    lo, hi = C.wilson_ci(n_death, n_matched)
    pct_cc = 100.0 * n_death / n_dc_known
    lo_cc, hi_cc = C.wilson_ci(n_death, n_dc_known)
    print(f"  deaths (DC_DISPOSITION_CODE == 'D') : {n_death:,}")
    print(f"  denominator (matched analytic rows) : {n_matched:,}")
    print(f"  in-hospital mortality               : {pct:.2f}% "
          f"(95% CI {100 * lo:.2f}% to {100 * hi:.2f}%)")
    print(f"  discharge disposition missing       : {n_dc_null:,} "
          f"({100.0 * n_dc_null / n_matched:.2f}%)")
    print(f"  complete-case denominator           : {n_dc_known:,}")
    print(f"  complete-case mortality             : {pct_cc:.2f}% "
          f"(95% CI {100 * lo_cc:.2f}% to {100 * hi_cc:.2f}%)")
    print(f"\n  abstract claim                      : {ABSTRACT_CLAIM:,}")
    print(f"  difference (measured minus claim)   : {n_death - ABSTRACT_CLAIM:,}")
    traces = bool(n_death == ABSTRACT_CLAIM)
    print(f"\n  REPRODUCES {ABSTRACT_CLAIM:,}?  {'YES' if traces else 'NO'}")

    # -------------------------------------------------------------------
    # 5. Deaths by year
    # -------------------------------------------------------------------
    C.banner("Step 5 — in-hospital deaths by PUF year")
    df["_death"] = is_death.astype(int)
    by_year = df.groupby("PUF_YEAR").agg(n=("inc_key", "size"),
                                         deaths=("_death", "sum"))
    by_year["pct"] = 100.0 * by_year["deaths"] / by_year["n"]
    print(f"  {'year':<6s} {'n':>10s} {'deaths':>9s} {'pct':>8s}")
    for y, r in by_year.iterrows():
        print(f"  {y:<6d} {int(r['n']):>10,} {int(r['deaths']):>9,} "
              f"{r['pct']:>7.2f}%")
    print(f"  {'TOTAL':<6s} {n_matched:>10,} {n_death:>9,} {pct:>7.2f}%")

    # -------------------------------------------------------------------
    # 6. Deaths among validation platelet recipients vs non-recipients
    # -------------------------------------------------------------------
    C.banner("Step 6 — deaths among the validation platelet recipients")
    train, test = C.load_split(with_score=False,
                               columns=["PUF_YEAR", "inc_key", C.OUTCOME])
    print(f"  derivation n = {len(train):,}, validation n = {len(test):,}")
    print(f"  derivation events = {int(train[C.OUTCOME].sum()):,}, "
          f"validation events = {int(test[C.OUTCOME].sum()):,}")

    deaths_map = df.set_index(["PUF_YEAR", "inc_key"])["_death"]
    strata = {}
    for name, frame in (("derivation", train), ("validation", test)):
        f = frame.copy()
        f["PUF_YEAR"] = pd.to_numeric(f["PUF_YEAR"]).astype("int64")
        f["inc_key"] = pd.to_numeric(f["inc_key"]).astype("int64")
        f["_death"] = f.set_index(["PUF_YEAR", "inc_key"]).index.map(deaths_map)
        f["_death"] = pd.to_numeric(f["_death"], errors="coerce").fillna(0).astype(int)
        for grp, mask in (("recipients", f[C.OUTCOME] == 1),
                          ("non-recipients", f[C.OUTCOME] == 0)):
            sub = f[mask]
            k, n = int(sub["_death"].sum()), int(len(sub))
            strata[(name, grp)] = (k, n)
            print(f"  {name:<11s} {grp:<15s} deaths {k:>7,} of {n:>9,}  "
                  f"({100.0 * k / n:>6.2f}%)")

    v_rec_k, v_rec_n = strata[("validation", "recipients")]
    v_non_k, v_non_n = strata[("validation", "non-recipients")]
    assert v_rec_n == C.V4_FROZEN["n_test_events"], (
        f"validation recipients n = {v_rec_n:,}, "
        f"expected {C.V4_FROZEN['n_test_events']:,}")

    # -------------------------------------------------------------------
    # 7. Alternative definitions, reported without cherry-picking
    # -------------------------------------------------------------------
    C.banner("Step 7 — alternative death definitions, for the response letter")
    alts = []

    def add_alt(label, k, n, scope):
        alts.append({"definition": label, "scope": scope, "deaths": int(k),
                     "denominator": int(n),
                     "pct": round(100.0 * k / n, 4) if n else float("nan"),
                     "equals_abstract_claim": bool(int(k) == ABSTRACT_CLAIM)})
        print(f"  {label:<58s} {int(k):>9,} of {int(n):>10,}  "
              f"({100.0 * k / n if n else float('nan'):>6.2f}%)"
              f"{'   <== EQUALS CLAIM' if int(k) == ABSTRACT_CLAIM else ''}")

    add_alt("discharge disposition D (primary definition)", n_death, n_matched,
            "analytic cohort 2020-2024")

    # (a) ED disposition expired: excluded by construction, so expect zero
    n_ed_mor = int((ed == "MOR").fillna(False).sum())
    add_alt("ED disposition MOR (expired in ED)", n_ed_mor, n_matched,
            "analytic cohort 2020-2024")

    # (b) any non-'D' discharge code that could plausibly encode death or hospice
    other_codes = [str(v) for v in vc_dc.index if pd.notna(v)
                   and str(v).strip().upper() != DEATH_CODE]
    print(f"\n  non-D discharge codes present: {sorted(other_codes)}")
    for code in sorted(other_codes):
        k = int((dc == code.strip().upper()).fillna(False).sum())
        if k == ABSTRACT_CLAIM:
            print(f"    NOTE: code {code!r} has n = {k:,}, which equals the claim")

    # (c) death plus missing discharge disposition, an upper bound
    add_alt("discharge disposition D or missing (upper bound)",
            n_death + n_dc_null, n_matched, "analytic cohort 2020-2024")

    # (d) pre-exclusion adult TBI pools
    C.banner("Step 7b — pre-exclusion adult TBI pools (registry side)")
    pre = {}
    for lo_y, hi_y, tag in ((2020, 2024, "2020-2024"), (2019, 2022, "2019-2022")):
        t0 = time.time()
        row = con.execute(
            f"""
            WITH tbi AS (
                SELECT DISTINCT inc_key, puf_year
                FROM raw_icd_diagnosis
                WHERE puf_year BETWEEN {lo_y} AND {hi_y}
                  AND icd_diagnosis_code LIKE 'S06%'
            )
            SELECT COUNT(*) AS n,
                   SUM(CASE WHEN upper(trim(v.DC_DISPOSITION_CODE)) = '{DEATH_CODE}'
                            THEN 1 ELSE 0 END) AS deaths
            FROM v_patients_2019_2024 v
            INNER JOIN tbi t ON v.inc_key = t.inc_key AND v.puf_year = t.puf_year
            WHERE v.puf_year BETWEEN {lo_y} AND {hi_y}
              AND v.AGE_NUMBER IS NOT NULL AND v.AGE_NUMBER >= 18
            """
        ).fetchone()
        pre[tag] = (int(row[1]), int(row[0]))
        print(f"  [{tag}] adult S06 pool, pre-exclusion "
              f"({time.time() - t0:.1f}s)")
        add_alt(f"discharge disposition D, adult TBI before exclusions",
                int(row[1]), int(row[0]), f"adult S06 pool {tag}")

    # (e) registry-wide, from the registry's own mortality view
    C.banner("Step 7c — registry-wide mortality view, for context only")
    vm = con.execute(
        f"SELECT * FROM v_mortality WHERE puf_year BETWEEN {min(YEARS)} "
        f"AND {max(YEARS)} ORDER BY puf_year").fetchdf()
    print(vm.to_string(index=False))
    print("\n  v_mortality DDL uses DC_DISPOSITION_CODE = 'D' over the whole "
          "patients table (all ages, all injuries, no TBI filter).")
    add_alt("discharge disposition D, entire registry",
            int(vm["deaths"].sum()), int(vm["total_patients"].sum()),
            "all NTDB patients 2020-2024")

    # (f) hospital-events codes: is there a death flag at all?
    C.banner("Step 7d — hospital events codes present for the analytic cohort")
    he = con.execute(
        """
        SELECT h.hospital_event AS code,
               COUNT(*) AS n_rows,
               SUM(CASE WHEN h.hospital_event_answer = 1 THEN 1 ELSE 0 END) AS n_yes
        FROM raw_hospital_events h
        INNER JOIN _pat_keys k
          ON h.inc_key = k.inc_key AND h.puf_year = k.puf_year
        GROUP BY h.hospital_event
        ORDER BY h.hospital_event
        """
    ).fetchdf()
    print(he.to_string(index=False))
    print("\n  The hospital-events table carries only a numeric complication "
          "code and a yes/no answer. No lookup table for it exists in this "
          "database, and no code in it is a death flag, so it cannot be the "
          "source of an in-hospital death count.")

    con.close()

    # -------------------------------------------------------------------
    # 8. Verdict
    # -------------------------------------------------------------------
    C.banner("Step 8 — verdict")
    hits = [a for a in alts if a["equals_abstract_claim"]]
    print(f"  measured in-hospital deaths, analytic cohort : {n_death:,}")
    print(f"  denominator                                  : {n_matched:,}")
    print(f"  abstract claim                               : {ABSTRACT_CLAIM:,}")
    print(f"  traces                                       : {traces}")
    print(f"  alternative definitions matching the claim   : {len(hits)}")
    for a in hits:
        print(f"    - {a['definition']} ({a['scope']})")
    if not hits:
        print("    none; no definition measured here lands on the claimed value")

    # -------------------------------------------------------------------
    # 9. Table
    # -------------------------------------------------------------------
    rows = []
    for a in alts:
        rows.append({"row_type": "definition", **a})
    for y, r in by_year.iterrows():
        rows.append({"row_type": "by_year", "definition": "discharge disposition D",
                     "scope": f"analytic cohort {y}", "deaths": int(r["deaths"]),
                     "denominator": int(r["n"]),
                     "pct": round(float(r["pct"]), 4),
                     "equals_abstract_claim": bool(int(r["deaths"]) == ABSTRACT_CLAIM)})
    for (split, grp), (k, n) in strata.items():
        rows.append({"row_type": "by_platelet_receipt",
                     "definition": "discharge disposition D",
                     "scope": f"{split} {grp}", "deaths": int(k),
                     "denominator": int(n), "pct": round(100.0 * k / n, 4),
                     "equals_abstract_claim": bool(int(k) == ABSTRACT_CLAIM)})
    for val, k in vc_dc.items():
        shown = "MISSING" if pd.isna(val) else str(val)
        rows.append({"row_type": "dc_disposition_value",
                     "definition": f"discharge disposition {shown}",
                     "scope": "analytic cohort 2020-2024", "deaths": int(k),
                     "denominator": int(n_matched),
                     "pct": round(100.0 * k / n_matched, 4),
                     "equals_abstract_claim": bool(int(k) == ABSTRACT_CLAIM)})
    tbl = pd.DataFrame(rows, columns=["row_type", "definition", "scope", "deaths",
                                      "denominator", "pct", "equals_abstract_claim"])
    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(TABLE_OUT, index=False)
    print(f"\n  wrote {TABLE_OUT}  ({len(tbl)} rows)")

    # -------------------------------------------------------------------
    # 10. Bundle part
    # -------------------------------------------------------------------
    if traces:
        verdict_display = ("The in hospital death count stated in the abstract "
                           "reproduces exactly against the registry on the same "
                           "encounter keys, so the claim stands with the analytic "
                           "cohort as its denominator.")
    else:
        verdict_display = ("The in hospital death count stated in the abstract "
                           "does not reproduce against the registry on the same "
                           "encounter keys, and no alternative death definition "
                           "measured here lands on it, so the claim is deleted "
                           "from the abstract rather than adjusted to fit.")

    # Part payloads are namespaced by file name when 06_build_bundle.py merges
    # them, so the block below is the mortality node itself and is not wrapped
    # in a second "mortality" key. This matches the other Phase 0 parts.
    payload = {
            "_meta": {
                "script": "05_mortality_or_drop.py",
                "finding": 8,
                "source": {"display":
                    "Counted directly in the trauma registry on the analytic "
                    "encounter keys, restricted to the study years. No analysis "
                    "output carries a mortality field, so this is the only "
                    "artefact behind the claim."},
            },
            "traces": traces,
            "abstract_claim": {"value": ABSTRACT_CLAIM},
            "deaths_in_hospital": {"n": n_death, "N": n_matched},
            "mortality_rate": {"estimate": round(pct / 100.0, 6),
                               "lo": round(lo, 6), "hi": round(hi, 6)},
            "keys_supplied": {"value": int(len(keys))},
            "keys_matched": {"value": int(n_matched)},
            "keys_unmatched": {"value": int(n_unmatched)},
            "discharge_disposition_missing": {"n": n_dc_null, "N": n_matched},
            "deaths_complete_case": {"n": n_death, "N": n_dc_known},
            "mortality_rate_complete_case": {"estimate": round(pct_cc / 100.0, 6),
                                             "lo": round(lo_cc, 6),
                                             "hi": round(hi_cc, 6)},
            "ed_disposition_expired": {"n": n_ed_mor, "N": n_matched},
            "deaths_validation_recipients": {"n": v_rec_k, "N": v_rec_n},
            "deaths_validation_non_recipients": {"n": v_non_k, "N": v_non_n},
            "deaths_by_year": {
                str(int(y)): {"n": int(r["deaths"]), "N": int(r["n"])}
                for y, r in by_year.iterrows()},
            "deaths_pre_exclusion_2020_2024": {"n": pre["2020-2024"][0],
                                               "N": pre["2020-2024"][1]},
            "deaths_pre_exclusion_2019_2022": {"n": pre["2019-2022"][0],
                                               "N": pre["2019-2022"][1]},
            "alternatives_matching_claim": {"value": len(hits)},
            "death_definition": {"display":
                "In hospital death was taken as a discharge disposition of "
                "died, which is the same rule the registry mortality view "
                "applies."},
            "denominator_definition": {"display":
                "The denominator is the analytic cohort after every cohort "
                "exclusion, which already removes patients recorded as having "
                "expired in the emergency department."},
            "missingness_caveat": {"display":
                "Roughly one in five encounters carries no discharge "
                "disposition at all, so the death count is a floor and the "
                "mortality percentage is lower than the complete case "
                "percentage. The count itself, which is what the abstract "
                "states, is unaffected."},
            "verdict": {"display": verdict_display},
            "action": {"display": (
                "Keep the sentence and report the measured count with the "
                "analytic cohort as its denominator."
                if traces else
                "Delete the in hospital death clause from the abstract, since "
                "mortality is neither the outcome of this work nor a quantity "
                "any analysis output carries.")},
    }
    p = C.write_part("mortality", payload)
    print(f"  wrote {p}")

    # -------------------------------------------------------------------
    # 11. Handoff
    # -------------------------------------------------------------------
    write_handoff(traces, n_death, n_matched, n_unmatched, pct, lo, hi,
                  n_dc_null, n_dc_known, pct_cc, lo_cc, hi_cc,
                  n_ed_mor, vc_dc, by_year, strata, alts, pre, hits)
    print(f"  wrote {HANDOFF_OUT}")

    C.banner("05_mortality_or_drop complete")
    return 0


def write_handoff(traces, n_death, n_matched, n_unmatched, pct, lo, hi,
                  n_dc_null, n_dc_known, pct_cc, lo_cc, hi_cc,
                  n_ed_mor, vc_dc, by_year, strata, alts, pre, hits):
    L = []
    A = L.append
    A("# Finding #8 handoff — the in-hospital mortality claim\n")
    A(f"Script: `v6/code/05_mortality_or_drop.py`  ")
    A(f"Log: `v6/logs/05_mortality_or_drop.log`  ")
    A(f"Table: `v6/tables/table_v6_mortality.csv`  ")
    A(f"Bundle part: `v6/bundle/parts/mortality.json` (`mortality.traces = "
      f"{str(traces).lower()}`)\n")

    A("## Verdict\n")
    if traces:
        A(f"**The mortality claim stands. Keep the sentence.** The registry "
          f"reproduces {n_death:,} in-hospital deaths on exactly the "
          f"{n_matched:,} analytic encounter keys, which is precisely the number "
          f"the abstract states. This was not a lucky near-miss and it was not "
          f"back-fitted: the first definition tested, the one the registry's own "
          f"mortality view uses, landed on the claimed value to the patient.\n")
        A(f"- **Number to report:** {n_death:,}")
        A(f"- **Denominator:** {n_matched:,}, the full analytic cohort")
        A(f"- **Rate:** {pct:.2f}% (95% CI {100 * lo:.2f}% to {100 * hi:.2f}%)")
        A(f"- **Definition:** discharge disposition `D`, in-hospital death, "
          f"NTDB 2020 to 2024\n")
        A(f"The abstract sentence \"The analytic cohort included {n_matched:,} "
          f"adult TBI patients, of whom {n_death:,} died in the hospital and "
          f"13,473 received platelets within 4 hours of ED arrival\" is correct "
          f"as written and needs no edit for this finding.\n")
        A(f"**One caveat to carry, since a reviewer will find it.** "
          f"{n_dc_null:,} of {n_matched:,} encounters ({100.0 * n_dc_null / n_matched:.1f}%) "
          f"carry no discharge disposition at all. The death *count* is "
          f"therefore a floor and is unaffected by the missingness, but any "
          f"*rate* depends on which denominator is used: {pct:.2f}% over the "
          f"whole cohort, or {pct_cc:.2f}% (95% CI {100 * lo_cc:.2f}% to "
          f"{100 * hi_cc:.2f}%) over the {n_dc_known:,} encounters with a "
          f"recorded disposition. The abstract states a count, not a rate, so it "
          f"is safe. If a rate is ever added to the manuscript, state the "
          f"denominator alongside it.\n")
    else:
        A(f"**The mortality claim must be deleted from the abstract.** The "
          f"abstract states that 20,033 of the 623,587 analytic-cohort patients "
          f"died in the hospital. Going back to the registry on exactly those "
          f"{n_matched:,} encounter keys, restricted to the same 2020 to 2024 "
          f"window, the in-hospital death count is **{n_death:,}**, not 20,033. "
          f"The difference is {n_death - 20033:+,} patients. No alternative "
          f"death definition tested here lands on 20,033 either.\n")

    A("## What the registry actually shows\n")
    A(f"- Analytic keys supplied: {C.V4_FROZEN['n_cohort']:,} distinct "
      f"(PUF_YEAR, inc_key) pairs, 2020 to 2024.")
    A(f"- Keys matched in the registry: {n_matched:,} "
      f"({'no unmatched keys' if n_unmatched == 0 else f'{n_unmatched:,} unmatched'}).")
    A(f"- In-hospital deaths, discharge disposition `D`: **{n_death:,}** "
      f"of {n_matched:,} ({pct:.2f}%, 95% CI {100 * lo:.2f}% to {100 * hi:.2f}%).")
    A(f"- Discharge disposition missing: {n_dc_null:,} "
      f"({100.0 * n_dc_null / n_matched:.2f}%).")
    A(f"- Expired in the emergency department (ED disposition `MOR`): "
      f"{n_ed_mor:,}, as expected, since that group is an explicit cohort "
      f"exclusion in `v4/code/01_build_cohort_duckdb.py`.\n")

    A("### Full discharge disposition distribution\n")
    A("| Code | n | % |")
    A("|---|---:|---:|")
    for val, k in vc_dc.items():
        shown = "MISSING" if pd.isna(val) else str(val)
        A(f"| {shown} | {int(k):,} | {100.0 * int(k) / n_matched:.2f}% |")
    A(f"| **Total** | **{n_matched:,}** | **100.00%** |\n")

    A("### Deaths by year\n")
    A("| Year | n | Deaths | % |")
    A("|---|---:|---:|---:|")
    for y, r in by_year.iterrows():
        A(f"| {int(y)} | {int(r['n']):,} | {int(r['deaths']):,} | "
          f"{float(r['pct']):.2f}% |")
    A(f"| **Total** | **{n_matched:,}** | **{n_death:,}** | **{pct:.2f}%** |\n")

    A("### Deaths by platelet receipt\n")
    A("| Split | Group | Deaths | n | % |")
    A("|---|---|---:|---:|---:|")
    for (split, grp), (k, n) in strata.items():
        A(f"| {split} | {grp} | {k:,} | {n:,} | {100.0 * k / n:.2f}% |")
    A("")
    vk, vn = strata[("validation", "recipients")]
    nk, nn = strata[("validation", "non-recipients")]
    A(f"Among the {vn:,} validation platelet recipients, {vk:,} died in hospital "
      f"({100.0 * vk / vn:.1f}%), against {nk:,} of {nn:,} "
      f"({100.0 * nk / nn:.1f}%) among validation non-recipients. Platelet "
      f"recipients are a far sicker group, which is exactly what a reader would "
      f"expect and is not evidence about the score.\n")

    A("### Alternative definitions tested\n")
    A("| Definition | Scope | Deaths | Denominator | % | Equals 20,033? |")
    A("|---|---|---:|---:|---:|---|")
    for a in alts:
        A(f"| {a['definition']} | {a['scope']} | {a['deaths']:,} | "
          f"{a['denominator']:,} | {a['pct']:.2f}% | "
          f"{'YES' if a['equals_abstract_claim'] else 'no'} |")
    A("")

    A("## The straight explanation Dumas should get\n")
    if traces:
        A(f"The in-hospital mortality figure in the abstract was flagged in the "
          f"audit as untraceable, since it appears in no table, log or derived "
          f"data file anywhere in the analysis. That flag was correct about the "
          f"provenance but wrong about the number. We went back to the National "
          f"Trauma Data Bank and counted in-hospital deaths directly on the same "
          f"{n_matched:,} encounter keys that define the analytic cohort, over "
          f"the same 2020 to 2024 window, using a discharge disposition of died. "
          f"The answer is {n_death:,}, which matches the abstract exactly. The "
          f"claim is correct; it simply had no artefact behind it until now. "
          f"That artefact is `v6/tables/table_v6_mortality.csv`, and the "
          f"reproduction is logged in `v6/logs/05_mortality_or_drop.log`.")
        A("")
        A(f"So the sentence stays. What changes is that mortality is now a "
          f"checked number rather than an unsourced one, and we can defend it "
          f"if asked. In-hospital mortality in the cohort was {pct:.2f}% "
          f"(95% CI {100 * lo:.2f}% to {100 * hi:.2f}%). Mortality remains a "
          f"descriptive feature of the cohort only: it is not the outcome of "
          f"this study, it plays no part in the derivation or the validation of "
          f"the score, and nothing in the manuscript should read as though the "
          f"score has anything to say about survival.")
        A("")
        A(f"Two details are worth having ready. First, {n_dc_null:,} encounters "
          f"({100.0 * n_dc_null / n_matched:.1f}%) have no discharge disposition "
          f"recorded, so {n_death:,} is a floor; among the {n_dc_known:,} "
          f"encounters that do carry a disposition, mortality is {pct_cc:.2f}%. "
          f"Second, patients who died in the emergency department are not in "
          f"this count, and cannot be, since they were removed by an explicit "
          f"cohort exclusion before any modelling.\n")
    else:
        A(f"The abstract sentence \"of whom 20,033 died in the hospital\" cannot "
          f"be traced to anything the analysis produced. Mortality is not an "
          f"outcome of this study, no analysis script ever computed it, and no "
          f"table, log or derived data file in the v4 pipeline carries a "
          f"mortality column. When we went back to the National Trauma Data Bank "
          f"and counted in-hospital deaths directly on the same {n_matched:,} "
          f"encounter keys that define the analytic cohort, the answer was "
          f"{n_death:,} ({pct:.2f}%), not 20,033. We also tested every other "
          f"death definition the registry supports, including deaths in the "
          f"emergency department, deaths before the cohort exclusions were "
          f"applied, and deaths in the earlier 2019 to 2022 study window; none "
          f"of them produces 20,033 either.")
        A("")
        A(f"The most likely history is that the number was written into the "
          f"abstract by hand and was never produced by the pipeline, in the same "
          f"way that the score range in that paragraph was written as minus "
          f"eleven to sixty-three when the observed range is much narrower. "
          f"Rather than substitute the correct count, we have removed the clause "
          f"entirely. Mortality plays no part in the derivation, the outcome, or "
          f"the validation of this score, so the abstract has nothing to gain "
          f"from carrying it, and a number that no output can support has no "
          f"place in a submitted abstract. The corrected sentence reads: \"The "
          f"analytic cohort included 623,587 adult TBI patients, of whom 13,473 "
          f"received platelets within 4 hours of ED arrival.\"")
        A("")
        A(f"If a reviewer specifically asks for cohort mortality, we can now give "
          f"it and defend it: {n_death:,} of {n_matched:,} patients ({pct:.2f}%, "
          f"95% CI {100 * lo:.2f}% to {100 * hi:.2f}%) died in hospital, counted "
          f"as a discharge disposition of died, which is the same definition the "
          f"registry's own mortality view uses.\n")

    A("## Provenance note for the response letter\n")
    if traces:
        A(f"The definitions that land on {ABSTRACT_CLAIM:,} are listed below. "
          f"The pre-specified definition was tested first and it is the one that "
          f"matches; nothing was searched for after the fact to make the number "
          f"work.\n")
        for a in hits:
            A(f"- {a['definition']} ({a['scope']}): {a['deaths']:,} of "
              f"{a['denominator']:,}.")
        A("")
        A(f"The definitions that do **not** match are listed in the table above "
          f"and are worth knowing so nobody re-derives the wrong one: deaths in "
          f"the adult TBI pool before the cohort exclusions are applied "
          f"({pre['2020-2024'][0]:,} for 2020 to 2024 and "
          f"{pre['2019-2022'][0]:,} for the older 2019 to 2022 window), and "
          f"deaths across the entire registry regardless of injury or age. The "
          f"registry's hospital-events table carries only complication codes and "
          f"no death flag, so it is not an alternative source.")
    elif hits:
        A("The following alternative definitions do land on 20,033 and are "
          "reported here for completeness. They are reported, not adopted: the "
          "plan directs that the claim be deleted rather than back-fitted.\n")
        for a in hits:
            A(f"- {a['definition']} ({a['scope']}): {a['deaths']:,} of "
              f"{a['denominator']:,}.")
    else:
        A("No definition tested lands on 20,033. Specifically, it is not the "
          "death count in the analytic cohort, not the death count before the "
          "cohort exclusions in either the 2020 to 2024 or the 2019 to 2022 "
          "window, not the count of emergency-department deaths, not the count "
          "of records with a missing discharge disposition, and not any single "
          "discharge-disposition category. The registry's hospital-events table "
          "carries only complication codes and no death flag, so it cannot be "
          "the source either. The number does not come out of the data.")
    A("")

    HANDOFF_OUT.parent.mkdir(parents=True, exist_ok=True)
    HANDOFF_OUT.write_text("\n".join(L))


if __name__ == "__main__":
    raise SystemExit(main())
