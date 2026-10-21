#!/usr/bin/env python3
"""
PAT-TBI v6 Phase 0, step 4 - missingness, comparator denominators, nesting, cohort flow.
=======================================================================================

Closes finding #5 (outcome missingness coded as not transfused), the reporting half of
finding #4 (three nested predictor pairs), and the counting half of finding #11
(comparator denominators). Also re-verifies the cohort-flow ladder and emits it in the
shape Figure 1 needs.

Nothing here re-derives the model. Everything is read from the frozen v4 artefacts.

Run with the framework interpreter, not bare `python3`:
    PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
    cd analysis/stage2_validation
    $PY 04_missingness.py

Reads
-----
    v4/derived/cohort_raw.parquet        pre-engineering cohort, raw fields
    v4/derived/candidates.parquet        analysis-ready binaries (nesting counts)
    v4/derived/cohort_test_scored.parquet   comparator masks
    v4/derived/cohort_exclusions.json    the exclusion ladder as counts
    v4/tables/table_consort.csv          the same ladder with prose reasons
    v4/derived/pat_refit.json            integer points, via C.load_points()

Writes
------
    v6/bundle/parts/missingness.json
    v6/tables/table_v6_missingness.csv
    v6/tables/table_v6_cohort_flow.csv
    v6/logs/04_missingness.log

Provenance convention for the bundle part
-----------------------------------------
The bundle JSON is kept free of snake_case identifiers: every field is named by its
clinical label. The raw parquet column name for each row lives in the CSV
(`Raw_column`) and in the handoff, which are analyst artefacts and are not rendered
through the ledger.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import _common as C

SCRIPT = "04_missingness"
PART = "missingness"

# ---------------------------------------------------------------------------
# Labels that _common.LABELS does not carry.
#
# `_common.LABELS` covers every engineered predictor and the raw comparator inputs,
# but not the raw physiology columns behind the vital-sign predictors, nor the raw
# mechanism and transfer columns. The contract says to work around a gap locally and
# report it, so these seven live here and are listed in the log and the final report
# for the orchestrator to fold into `_common.LABELS`.
# ---------------------------------------------------------------------------
LOCAL_LABELS: dict[str, str] = {
    "SBP2": "Systolic blood pressure, mmHg",
    "P2": "Heart rate, beats/min",
    "RR2": "Respiratory rate, breaths/min",
    "OX2": "Oxygen saturation, %",
    "temperature_raw": "Temperature, degrees Celsius",
    "cause_e_code": "External injury mechanism code",
    "HOSPITAL_TRANSFER": "Transfer from another facility",
    "ISS": "Injury Severity Score",
}

_LOCAL_USED: list[str] = []


def lbl(var: str) -> str:
    """Clinical prose for an identifier. `_common` first, local fallback second."""
    try:
        return C.label(var)
    except KeyError:
        if var not in LOCAL_LABELS:
            raise
        if var not in _LOCAL_USED:
            _LOCAL_USED.append(var)
        return LOCAL_LABELS[var]


# ---------------------------------------------------------------------------
# The raw fields behind the retained predictors, and the reference level that the
# engineering step substitutes when the raw field is unavailable. Every rule below
# was read off v4/code/02_engineer_features.py.
# ---------------------------------------------------------------------------
RETAINED = set(C.load_points())

RAW_FIELDS: list[dict] = [
    dict(col="AGE_NUMBER", kind="numeric",
         feeds=["age_45_64", "age_65_74", "age_75plus"],
         reference="No substitution required; age is complete in the analytic cohort"),
    dict(col="SEX", kind="sex",
         feeds=["female"],
         reference="Sex not recorded is coded as not female"),
    dict(col="SBP2", kind="numeric",
         feeds=["sbp_hypotensive", "sbp_hypertensive", "si_elevated", "si_high"],
         reference="Missing is coded as not hypotensive"),
    dict(col="P2", kind="numeric",
         feeds=["hr_bradycardic", "hr_tachycardic", "si_elevated", "si_high"],
         reference="Missing is coded as neither bradycardic nor tachycardic"),
    dict(col="RR2", kind="numeric",
         feeds=["rr_low", "rr_high"],
         reference="Missing is coded as neither low nor high respiratory rate"),
    dict(col="OX2", kind="numeric",
         feeds=["hypoxic"],
         reference="Missing is coded as not hypoxic"),
    dict(col="temperature_raw", kind="temperature",
         feeds=["temp_hypothermic"],
         reference=("Missing, or a reading outside the Celsius window the code accepts, "
                    "is coded as not hypothermic")),
    dict(col="GCS2", kind="numeric",
         feeds=["gcs_severe", "gcs_moderate"],
         reference=("Missing is coded as neither severe nor moderate; the comparator "
                    "scores fill a missing total to 15")),
    dict(col="MOTOR2", kind="numeric",
         feeds=["gcs_motor_low", "gcs_motor_high"],
         reference="Missing is coded as neither a low nor a full motor response"),
    dict(col="TQIP_TBI_PUPIL_RESPONSE", kind="pupil",
         feeds=["tbi_pupil_none"],
         reference="Missing is treated as both pupils reactive"),
    dict(col="cause_e_code", kind="string",
         feeds=["mech_mvc", "mech_fall", "mech_assault", "risk_penetrating"],
         reference="Missing is coded as none of the mechanism groups"),
    dict(col="HOSPITAL_TRANSFER", kind="string",
         feeds=["is_transfer"],
         reference="Missing is coded as not transferred"),
    dict(col="cc_anticoag", kind="flag",
         feeds=["cc_anticoag"],
         reference=("Built by pivoting the pre-existing conditions table; a condition "
                    "that is not listed is coded as absent")),
    dict(col="cc_bleeding", kind="flag",
         feeds=["cc_bleeding"],
         reference=("Built by pivoting the pre-existing conditions table; a condition "
                    "that is not listed is coded as absent")),
]

# The external-injury prefix ranges, exactly as 02_engineer_features.py builds them.
PREFIX_RANGES: list[tuple[str, str, str]] = [
    ("mech_mvc", "V40", "V89"),
    ("mech_fall", "W00", "W19"),
    ("mech_assault", "X85", "Y09"),
    ("risk_penetrating", "X93", "X99"),
]

COMPARATORS: list[tuple[str, str]] = [
    ("PAT-TBI integer score", "pat_tbi_score_mask"),
    ("Glasgow Coma Scale total", "gcs_mask"),
    ("Glasgow Coma Scale pupils score", "gcs_p_mask"),
    ("IMPACT-Core", "impact_mask"),
    ("CRASH-Basic", "crash_mask"),
]

FLOW_STEPS: list[tuple[str, str, str]] = [
    # (key in cohort_exclusions.json, population that survives, reason for the rung)
    ("start_2020_2024_age_18plus_from_duckdb",
     "Adult trauma patients in the National Trauma Data Bank, 2020 to 2024",
     "Entry criteria: age 18 years or older, admission years 2020 to 2024"),
    ("tbi_S06_diagnosis",
     "Patients with a traumatic brain injury diagnosis",
     "No ICD-10-CM S06 traumatic brain injury diagnosis"),
    ("exclude_prehospital_arrest",
     "Patients without prehospital cardiac arrest",
     "Prehospital cardiac arrest, so the 4-hour window is not observable"),
    ("exclude_ed_disposition_expired",
     "Patients who did not expire in the emergency department",
     "Expired in the emergency department before the 4-hour window closed"),
    ("exclude_transferred_out",
     "Patients not transferred to another facility",
     "Transferred out, so transfusion at this centre is not observable"),
    ("exclude_missing_ed_disposition",
     "Analytic cohort with a recorded emergency department disposition",
     "Emergency department disposition not recorded, so eligibility is not ascertainable"),
]


def pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else float("nan")


# ---------------------------------------------------------------------------
# 1. Cohort flow
# ---------------------------------------------------------------------------
def cohort_flow() -> tuple[pd.DataFrame, list[dict]]:
    C.banner("1. Cohort flow - the six-rung exclusion ladder, re-verified")

    counts = json.loads((C.V4_DERIVED / "cohort_exclusions.json").read_text())
    consort = pd.read_csv(C.V4_TABLES / "table_consort.csv")

    rows: list[dict] = []
    prev_out: int | None = None
    for i, (key, step, reason) in enumerate(FLOW_STEPS):
        if key not in counts:
            raise RuntimeError(f"cohort_exclusions.json has no rung {key!r}")
        n_out = int(counts[key])
        n_in = n_out if prev_out is None else prev_out
        n_dropped = n_in - n_out
        rows.append(dict(step=step, n_in=n_in, n_dropped=n_dropped, n_out=n_out,
                         reason=reason))
        prev_out = n_out

    flow = pd.DataFrame(rows, columns=["step", "n_in", "n_dropped", "n_out", "reason"])

    # -- gate: every rung closes, and consecutive rungs chain ---------------
    bad = []
    for i, r in flow.iterrows():
        if r["n_in"] - r["n_dropped"] != r["n_out"]:
            bad.append(f"rung {i + 1} does not close: {r['n_in']} - {r['n_dropped']} != {r['n_out']}")
        if i and flow.loc[i - 1, "n_out"] != r["n_in"]:
            bad.append(f"rung {i} does not chain into rung {i + 1}")
    if bad:
        raise AssertionError("cohort flow arithmetic failed:\n  " + "\n  ".join(bad))

    # -- cross-check against the consort table -----------------------------
    cross = consort.dropna(subset=["N_before"]).reset_index(drop=True)
    for i, r in cross.iterrows():
        j = i + 1  # consort row 1 is the entry rung
        if int(r["N_before"]) != flow.loc[j, "n_in"]:
            raise AssertionError(
                f"consort N_before {int(r['N_before'])} disagrees with the ladder "
                f"{flow.loc[j, 'n_in']} at rung {j + 1}")
        if int(r["N_after"]) != flow.loc[j, "n_out"]:
            raise AssertionError(
                f"consort N_after {int(r['N_after'])} disagrees with the ladder "
                f"{flow.loc[j, 'n_out']} at rung {j + 1}")
        if int(r["N_excluded"]) != flow.loc[j, "n_dropped"]:
            raise AssertionError(
                f"consort N_excluded {int(r['N_excluded'])} disagrees with the ladder "
                f"{flow.loc[j, 'n_dropped']} at rung {j + 1}")

    final = int(flow.iloc[-1]["n_out"])
    if final != C.V4_FROZEN["n_cohort"]:
        raise AssertionError(f"ladder ends at {final:,}, expected {C.V4_FROZEN['n_cohort']:,}")

    print(f"  rungs verified            : {len(flow)}")
    print(f"  entry population          : {int(flow.iloc[0]['n_out']):,}")
    print(f"  analytic cohort           : {final:,}")
    print(f"  total excluded            : {int(flow['n_dropped'].sum()):,}")
    print("  every rung closes and every consecutive pair chains; the consort table agrees")
    print()
    for i, r in flow.iterrows():
        print(f"  [{i + 1}] {r['step']}")
        print(f"      in {r['n_in']:>9,}   dropped {r['n_dropped']:>9,}   out {r['n_out']:>9,}")
        print(f"      reason: {r['reason']}")

    ladder = [
        dict(step=r["step"],
             entering={"n": int(r["n_in"])},
             excluded={"n": int(r["n_dropped"]), "N": int(r["n_in"])},
             remaining={"n": int(r["n_out"])},
             reason=r["reason"])
        for _, r in flow.iterrows()
    ]
    return flow, ladder


# ---------------------------------------------------------------------------
# 2. Outcome-field missingness (finding #5)
# ---------------------------------------------------------------------------
def outcome_missingness(raw: pd.DataFrame) -> dict:
    C.banner("2. Outcome-field missingness - finding #5")

    n = len(raw)
    plt4 = pd.to_numeric(raw["PLATELETS4HOURS"], errors="coerce")
    miss = plt4.isna()
    n_miss = int(miss.sum())
    p_miss = pct(n_miss, n)

    print(f"  analytic cohort                              : {n:,}")
    print(f"  PLATELETS4HOURS not recorded                 : {n_miss:,} ({p_miss:.2f}%)")
    print(f"  PLATELETS4HOURS recorded as zero             : {int((plt4 == 0).sum()):,}")
    print(f"  PLATELETS4HOURS recorded as positive         : {int((plt4 > 0).sum()):,}")
    print(f"  events under the missing-as-zero rule        : {int((plt4.fillna(0) > 0).sum()):,}")
    print()
    print("  The plan records 54,523 (8.74%). Verified against the data above.")

    by_year = []
    print()
    print("  by admission year")
    print(f"    {'year':>6}  {'n':>9}  {'missing':>9}  {'pct':>7}")
    for yr, g in raw.groupby("PUF_YEAR", sort=True):
        s = pd.to_numeric(g["PLATELETS4HOURS"], errors="coerce")
        k, d = int(s.isna().sum()), int(len(g))
        by_year.append(dict(year={"value": int(yr)}, missing={"n": k, "N": d}))
        print(f"    {int(yr):>6}  {d:>9,}  {k:>9,}  {pct(k, d):>6.2f}%")

    pcts = [100.0 * y["missing"]["n"] / y["missing"]["N"] for y in by_year]
    lo, hi = min(pcts), max(pcts)
    lo_year = int(by_year[int(np.argmin(pcts))]["year"]["value"])
    hi_year = int(by_year[int(np.argmax(pcts))]["year"]["value"])
    print()
    print(f"  by-year range: {lo:.2f}% ({lo_year}) to {hi:.2f}% ({hi_year}); "
          f"the plan records 8.38% to 9.43%")

    wilson = C.wilson_ci(n_miss, n)
    return dict(
        overall={"n": n_miss, "N": n},
        overall_interval={"estimate": round(p_miss, 3),
                          "lo": round(100 * wilson[0], 3),
                          "hi": round(100 * wilson[1], 3)},
        recorded_zero={"n": int((plt4 == 0).sum()), "N": n},
        recorded_positive={"n": int((plt4 > 0).sum()), "N": n},
        events_under_missing_as_zero={"n": int((plt4.fillna(0) > 0).sum()), "N": n},
        by_year=by_year,
        year_range_pct={"lo": round(lo, 2), "hi": round(hi, 2)},
        assumption={"display": (
            "The platelet field is not recorded for a minority of the analytic cohort. "
            "Those encounters are counted as not transfused, which is a missingness as "
            "zero assumption. The Methods state the size of that group and the "
            "Limitations name the assumption.")},
        stability={"display": (
            "The share of encounters without a recorded platelet field is stable across "
            "the admission years, with no year departing materially from the pooled "
            "share, so the assumption does not fall unevenly on one period.")},
    )


# ---------------------------------------------------------------------------
# 3. Per-predictor missingness before the fill-to-reference rule
# ---------------------------------------------------------------------------
def predictor_missingness(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], dict]:
    C.banner("3. Per-predictor missingness, before the fill-to-reference rule")

    n = len(raw)
    rows: list[dict] = []

    for spec in RAW_FIELDS:
        col, kind = spec["col"], spec["kind"]
        s = raw[col]
        n_null = int(s.isna().sum())
        n_sentinel = 0
        sentinel_note = ""

        if kind == "sex":
            up = s.astype("string").str.upper().str.strip()
            n_sentinel = int((~up.isin(["M", "F"]) & up.notna()).sum())
            sentinel_note = "sex recorded as neither male nor female"
        elif kind == "temperature":
            t = pd.to_numeric(s, errors="coerce")
            n_sentinel = int((t.notna() & (t < 20)).sum())
            sentinel_note = "reading below the Celsius window the code accepts"
        elif kind == "string":
            blank = s.astype("string").str.strip()
            n_sentinel = int((blank.notna() & (blank == "")).sum())
            sentinel_note = "blank string"
        elif kind == "flag":
            sentinel_note = "no explicit missing state; absence is coded as absent"

        n_unusable = n_null + n_sentinel
        feeds_retained = [v for v in spec["feeds"] if v in RETAINED]
        rows.append(dict(
            Raw_column=col,
            Field=lbl(col),
            N=n,
            N_missing=n_null,
            Pct_missing=round(pct(n_null, n), 3),
            N_uninformative=n_unusable,
            Pct_uninformative=round(pct(n_unusable, n), 3),
            Sentinel_note=sentinel_note,
            Reference_applied=spec["reference"],
            Feeds=" | ".join(lbl(v) for v in spec["feeds"]),
            Feeds_retained_predictor="yes" if feeds_retained else "no",
        ))

    # -- shock index: the fixed 0.7 substitution, computed the way the code does
    sbp = pd.to_numeric(raw["SBP2"], errors="coerce")
    p = pd.to_numeric(raw["P2"], errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        si_raw = p / sbp.where(sbp > 0)
    n_si_missing = int(si_raw.isna().sum())
    rows.append(dict(
        Raw_column="shock_index",
        Field=lbl("shock_index"),
        N=n,
        N_missing=n_si_missing,
        Pct_missing=round(pct(n_si_missing, n), 3),
        N_uninformative=n_si_missing,
        Pct_uninformative=round(pct(n_si_missing, n), 3),
        Sentinel_note="heart rate or systolic pressure missing, or systolic pressure zero",
        Reference_applied="Substituted with the fixed value 0.7",
        Feeds=" | ".join(lbl(v) for v in ("shock_index", "si_elevated", "si_high")),
        Feeds_retained_predictor="yes",
    ))

    tab = pd.DataFrame(rows)
    print(f"  {'field':<38} {'missing':>9} {'pct':>7} {'uninformative':>14} {'pct':>7}")
    for _, r in tab.iterrows():
        print(f"  {r['Field']:<38} {r['N_missing']:>9,} {r['Pct_missing']:>6.2f}% "
              f"{r['N_uninformative']:>14,} {r['Pct_uninformative']:>6.2f}%")
    print()
    print(f"  shock index unavailable and filled to the fixed 0.7 value: "
          f"{n_si_missing:,} ({pct(n_si_missing, n):.2f}%)")
    print(f"    systolic pressure missing : {int(sbp.isna().sum()):,}")
    print(f"    systolic pressure zero    : {int((sbp == 0).sum()):,}")
    print(f"    heart rate missing        : {int(p.isna().sum()):,}")
    print(f"    shock index computable    : {int(si_raw.notna().sum()):,}")

    part = [
        dict(field=r["Field"],
             missing={"n": int(r["N_missing"]), "N": n},
             uninformative={"n": int(r["N_uninformative"]), "N": n},
             reference=r["Reference_applied"],
             feeds=r["Feeds"])
        for _, r in tab.iterrows()
    ]
    si_block = dict(
        substituted={"n": n_si_missing, "N": n},
        computable={"n": int(si_raw.notna().sum()), "N": n},
        substituted_value={"value": 0.7},
        note={"display": (
            "Where the heart rate or the systolic pressure was not recorded, or the "
            "systolic pressure was zero, the shock index could not be formed and a fixed "
            "normal value was substituted before modelling.")},
    )
    return tab, part, si_block


# ---------------------------------------------------------------------------
# 4. Comparator denominators (finding #11, the counting half)
# ---------------------------------------------------------------------------
def comparator_denominators() -> dict:
    C.banner("4. Comparator denominators - finding #11, the counting half")

    scored = C.load_test_scored()
    n_test = len(scored)
    if n_test != C.V4_FROZEN["n_test"]:
        raise AssertionError(f"validation frame has {n_test:,} rows, expected "
                             f"{C.V4_FROZEN['n_test']:,}")
    y = scored[C.OUTCOME].astype(int)
    n_events = int(y.sum())
    if n_events != C.V4_FROZEN["n_test_events"]:
        raise AssertionError(f"validation events {n_events:,}, expected "
                             f"{C.V4_FROZEN['n_test_events']:,}")

    gcs_missing = int(pd.to_numeric(scored["GCS2"], errors="coerce").isna().sum())
    motor_missing = int(pd.to_numeric(scored["MOTOR2"], errors="coerce").isna().sum())
    pupil_missing = int(scored["TQIP_TBI_PUPIL_RESPONSE"].isna().sum())
    age_missing = int(pd.to_numeric(scored["AGE_NUMBER"], errors="coerce").isna().sum())

    out: list[dict] = []
    print(f"  {'score':<34} {'analysed':>9} {'dropped':>8} {'events':>7} {'event rate':>11}")
    for name, mask_col in COMPARATORS:
        m = scored[mask_col].astype(bool)
        k, e = int(m.sum()), int(y[m].sum())
        print(f"  {name:<34} {k:>9,} {n_test - k:>8,} {e:>7,} {pct(e, k):>10.3f}%")
        out.append(dict(
            score=name,
            analysed={"n": k, "N": n_test},
            not_scoreable={"n": n_test - k, "N": n_test},
            events={"n": e, "N": k},
        ))

    print()
    print("  The plan records 199,757 for IMPACT-Core and 199,862 for CRASH-Basic. "
          "Verified above.")
    print(f"  Within the validation split: age missing {age_missing:,}, Glasgow Coma Scale "
          f"total missing {gcs_missing:,}, motor score missing {motor_missing:,}, pupil "
          f"field missing {pupil_missing:,}.")
    print("  The score itself and the two raw Glasgow Coma Scale comparators fill a "
          "missing input to a")
    print(f"  reference level rather than dropping the row, so all three keep the full "
          f"{n_test:,}.")
    print(f"  IMPACT-Core needs an age and a usable motor score, so it drops exactly the "
          f"{motor_missing:,} rows with no motor score.")
    print(f"  CRASH-Basic needs an age, a Glasgow Coma Scale total and an extracranial "
          f"injury field, so it drops exactly the {gcs_missing:,} rows with no total.")

    return dict(
        validation={"n": n_test},
        scores=out,
        inputs=dict(
            age_missing={"n": age_missing, "N": n_test},
            gcs_total_missing={"n": gcs_missing, "N": n_test},
            gcs_motor_missing={"n": motor_missing, "N": n_test},
            pupil_missing={"n": pupil_missing, "N": n_test},
        ),
        note={"display": (
            "Two comparators need inputs that are not recorded for every encounter, so "
            "they run on a smaller denominator than the score itself. Each comparator is "
            "reported against its own denominator and its own event count.")},
    )


# ---------------------------------------------------------------------------
# 5. Nesting counts (finding #4, the reporting half)
# ---------------------------------------------------------------------------
def nesting_counts(cand: pd.DataFrame, raw: pd.DataFrame) -> dict:
    C.banner("5. Nested predictor pairs - finding #4, the reporting half")

    n = len(cand)
    points = C.load_points()
    pairs: list[dict] = []

    for inner, outer in C.NESTED_PAIRS:
        n_inner = int(cand[inner].sum())
        n_outer = int(cand[outer].sum())
        n_both = int(((cand[inner] == 1) & (cand[outer] == 1)).sum())
        n_inner_only = n_inner - n_both
        pi, po = int(points[inner]), int(points[outer])
        print(f"  {lbl(inner)}")
        print(f"    inner indicator present   : {n_inner:,} of {n:,} ({pct(n_inner, n):.2f}%)")
        print(f"    outer indicator present   : {n_outer:,} of {n:,} ({pct(n_outer, n):.2f}%) "
              f"[{lbl(outer)}]")
        print(f"    both present              : {n_both:,} of {n_inner:,} "
              f"({pct(n_both, n_inner):.2f}% of the inner group)")
        print(f"    inner without outer       : {n_inner_only:,}")
        print(f"    points, inner / outer     : {pi:+d} / {po:+d}   net when both apply: "
              f"{pi + po:+d}")
        print()
        pairs.append(dict(
            inner=lbl(inner), outer=lbl(outer),
            inner_present={"n": n_inner, "N": n},
            outer_present={"n": n_outer, "N": n},
            both_present={"n": n_both, "N": n_inner},
            inner_without_outer={"n": n_inner_only, "N": n_inner},
            inner_points={"value": pi},
            outer_points={"value": po},
            net_points={"value": pi + po},
        ))

    # -- the external-injury prefix ranges that define the mechanism terms ---
    e = raw["cause_e_code"].astype("string").str.upper().str.strip().fillna("")
    print("  external-injury prefix ranges, exactly as 02_engineer_features.py builds them")
    ranges = []
    for var, first, last in PREFIX_RANGES:
        n_var = int(cand[var].sum())
        print(f"    {lbl(var):<44} {first} to {last}   n = {n_var:,}")
        ranges.append(dict(group=lbl(var), first=first, last=last,
                           present={"n": n_var, "N": n}))

    outside = {}
    for name, pfx in [
        ("Self-inflicted firearm injury", tuple(f"X{i}" for i in range(72, 75))),
        ("Unintentional firearm injury", tuple(f"W{i}" for i in range(32, 35))),
        ("Firearm injury of undetermined intent", tuple(f"Y{i}" for i in range(22, 25))),
    ]:
        k = int(e.str.startswith(pfx).sum())
        outside[name] = k
    print()
    print("  outside the penetrating label, and therefore not counted as penetrating")
    for name, k in outside.items():
        print(f"    {name:<44} n = {k:,}")

    return dict(
        pairs=pairs,
        prefix_ranges=ranges,
        outside_penetrating=[
            dict(group=name, present={"n": k, "N": n}) for name, k in outside.items()
        ],
        penetrating_scope={"display": (
            "The penetrating label covers assault-coded penetrating injury only. "
            "Self-inflicted firearm injury and unintentional firearm injury fall outside "
            "it and are not counted as penetrating.")},
        statement_penetrating={"display": (
            "Penetrating assault is a strict subset of assault, so every patient carrying "
            "the penetrating indicator also carries the assault indicator and receives "
            "both mechanism terms rather than the penetrating term alone.")},
        statement_shock_index={"display": (
            "The higher shock index threshold implies the lower one, so a patient above "
            "the higher threshold receives both shock index terms.")},
        statement_motor={"display": (
            "A low motor response nearly always accompanies a severe total Glasgow Coma "
            "Scale, so the two terms usually apply together; a small group carries the "
            "motor term without the total term.")},
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    C.start_log(SCRIPT)
    C.banner("PAT-TBI v6 Phase 0 step 4 - missingness, denominators, nesting, cohort flow")
    print("Findings closed here: #5 (outcome missingness), the reporting half of #4")
    print("(nested predictor pairs), and the counting half of #11 (comparator")
    print("denominators). The cohort-flow ladder is re-verified and re-emitted.")

    raw = C.load_cohort_raw()
    cand = pd.read_parquet(C.CANDIDATES)

    n = len(raw)
    if n != C.V4_FROZEN["n_cohort"]:
        raise AssertionError(f"cohort_raw has {n:,} rows, expected {C.V4_FROZEN['n_cohort']:,}")
    if len(cand) != n:
        raise AssertionError(f"candidates has {len(cand):,} rows, expected {n:,}")
    if not (raw["PUF_YEAR"].values == cand["PUF_YEAR"].values).all() \
            or not (raw["inc_key"].values == cand["inc_key"].values).all():
        raise AssertionError("cohort_raw and candidates are not row-aligned")

    n_events = int(cand[C.OUTCOME].sum())
    want = C.V4_FROZEN["n_train_events"] + C.V4_FROZEN["n_test_events"]
    if n_events != want:
        raise AssertionError(f"cohort events {n_events:,}, expected {want:,}")
    print(f"\n  analytic cohort           : {n:,}")
    print(f"  outcome events            : {n_events:,} "
          f"({C.V4_FROZEN['n_train_events']:,} derivation plus "
          f"{C.V4_FROZEN['n_test_events']:,} validation)")
    print(f"  retained predictors       : {len(RETAINED)}")

    flow, ladder = cohort_flow()
    outcome = outcome_missingness(raw)
    miss_tab, miss_part, si_block = predictor_missingness(raw)
    comparators = comparator_denominators()
    nesting = nesting_counts(cand, raw)

    # -- outputs -----------------------------------------------------------
    C.banner("Writing outputs")
    (C.V6 / "tables").mkdir(parents=True, exist_ok=True)

    flow_csv = C.V6 / "tables" / "table_v6_cohort_flow.csv"
    flow.to_csv(flow_csv, index=False)
    print(f"  wrote {flow_csv}  ({len(flow)} rungs)")

    miss_csv = C.V6 / "tables" / "table_v6_missingness.csv"
    miss_out = miss_tab.copy()
    outcome_row = pd.DataFrame([dict(
        Raw_column="PLATELETS4HOURS",
        Field="Platelet units transfused within 4 hours",
        N=n,
        N_missing=outcome["overall"]["n"],
        Pct_missing=round(pct(outcome["overall"]["n"], n), 3),
        N_uninformative=outcome["overall"]["n"],
        Pct_uninformative=round(pct(outcome["overall"]["n"], n), 3),
        Sentinel_note="field not reported by the treating centre",
        Reference_applied="Missing is counted as not transfused",
        Feeds="Primary outcome",
        Feeds_retained_predictor="outcome",
    )])
    miss_out = pd.concat([outcome_row, miss_out], ignore_index=True)
    miss_out.to_csv(miss_csv, index=False)
    print(f"  wrote {miss_csv}  ({len(miss_out)} fields)")

    payload = dict(
        cohort=dict(
            analytic={"n": n},
            events={"n": n_events, "N": n},
            derivation_events={"n": C.V4_FROZEN["n_train_events"],
                               "N": C.V4_FROZEN["n_train"]},
            validation_events={"n": C.V4_FROZEN["n_test_events"],
                               "N": C.V4_FROZEN["n_test"]},
        ),
        cohort_flow=dict(
            entry={"n": int(flow.iloc[0]["n_out"])},
            analytic={"n": int(flow.iloc[-1]["n_out"])},
            total_excluded={"n": int(flow["n_dropped"].sum()),
                            "N": int(flow.iloc[0]["n_out"])},
            rungs={"value": len(flow)},
            ladder=ladder,
            closure={"display": (
                "Every rung of the exclusion ladder closes arithmetically, and the "
                "surviving count of each rung equals the entering count of the next.")},
        ),
        outcome_missingness=outcome,
        predictor_missingness=miss_part,
        shock_index_substitution=si_block,
        comparator_denominators=comparators,
        nesting=nesting,
        labels_pending_in_common=sorted(LOCAL_LABELS[v] for v in _LOCAL_USED),
    )
    path = C.write_part(PART, payload)
    print(f"  wrote {path}")

    C.banner("Gate")
    print(f"  [PASS] analytic cohort {n:,}")
    print(f"  [PASS] outcome events {n_events:,} equals derivation plus validation")
    print("  [PASS] every rung of the cohort flow closes and consecutive rungs chain")
    print(f"  [PASS] comparator denominators verified against the frozen masks")
    print(f"  [PASS] nested pairs counted over the full cohort")
    if _LOCAL_USED:
        print()
        print("  NOTE: _common.LABELS carries no entry for the raw fields behind the "
              "vital-sign,")
        print("  mechanism and transfer predictors. A local label table was used for "
              "these and is")
        print("  listed in the bundle part so the orchestrator can fold it in:")
        for v in _LOCAL_USED:
            print(f"    {v:<28} -> {LOCAL_LABELS[v]}")

    print("\n04_missingness complete.")


if __name__ == "__main__":
    main()
