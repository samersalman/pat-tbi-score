#!/usr/bin/env python3
"""
PAT-TBI v6 Phase 0, step 6a - the statistics behind manuscript Table 1.
======================================================================

Cohort characteristics by platelet receipt within four hours of arrival.

Closes finding #12 (fifteen empty Table 1 rows) and the Table 1 half of finding #2
(six cutpoint labels that contradicted the code). Every label is read from
`_common.LABELS`, which was transcribed line by line off
`v4/code/02_engineer_features.py`; the v4 `14_table1.py` LABEL_MAP disagreed with the
code on six variables, four of which are in the final score.

Nothing here re-derives the model. The cohort, the outcome and the labels come from
the frozen v4 artefacts and from `_common`.

Run with the framework interpreter, not bare `python3`:
    PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
    cd analysis/stage2_validation
    $PY 04_missingness.py     # must run first; supplies the missingness column
    $PY 06a_table1_stats.py

Reads
-----
    v4/derived/candidates.parquet        analysis-ready binaries and the outcome
    v4/derived/cohort_raw.parquet        recorded values behind the continuous rows
    v6/bundle/parts/missingness.json     the missingness column, from step 4

Writes
------
    v6/bundle/parts/table1.json
    v6/tables/table_v6_table1.csv
    v6/logs/06a_table1_stats.log

Why a standardized difference and not a p-value
-----------------------------------------------
With 623,587 encounters every two-group test falls below the reporting floor, so a
p-value separates nothing and says nothing about the size of a difference. The
standardized mean difference is scale free and does not grow with the sample, so a
value above one tenth marks a difference worth reading.

Continuous rows are summarised from the recorded values, not from the engineered
column, so a reference substitution made during feature engineering never enters a
median. The missingness column counts the encounters for which the underlying field
was not recorded, or could not inform the row, and for which a reference level was
therefore substituted.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import _common as C

SCRIPT = "06a_table1_stats"
PART = "table1"

SMD_THRESHOLD = 0.10

# ---------------------------------------------------------------------------
# Labels that _common.LABELS does not carry: the raw physiology columns behind the
# vital-sign predictors. Reported in the log and in the bundle part so the
# orchestrator can fold them into `_common.LABELS`. Everything else comes from
# `C.label()`.
# ---------------------------------------------------------------------------
LOCAL_LABELS: dict[str, str] = {
    "SBP2": "Systolic blood pressure, mmHg",
    "P2": "Heart rate, beats/min",
    "RR2": "Respiratory rate, breaths/min",
    "OX2": "Oxygen saturation, %",
    "temperature_raw": "Temperature, degrees Celsius",
}
_LOCAL_USED: list[str] = []


def lbl(var: str) -> str:
    try:
        return C.label(var)
    except KeyError:
        if var not in LOCAL_LABELS:
            raise
        if var not in _LOCAL_USED:
            _LOCAL_USED.append(var)
        return LOCAL_LABELS[var]


# ---------------------------------------------------------------------------
# The six v4 labels that contradicted the code, kept here only so the log can show
# what changed. The corrected strings live in `_common.LABELS`.
# ---------------------------------------------------------------------------
V4_WRONG_LABELS: dict[str, str] = {
    "sbp_hypertensive": "Systolic blood pressure > 180 mmHg",
    "rr_low": "Respiratory rate < 10 breaths/min",
    "rr_high": "Respiratory rate > 24 breaths/min",
    "hypoxic": "Oxygen saturation < 90%",
    "si_high": "Shock index >= 1.0",
    "temp_hypothermic": "Temperature < 36 degrees C",
    "hr_tachycardic": "Heart rate > 120 beats/min",
}

# ---------------------------------------------------------------------------
# Table layout.
#
#   block      the clinical grouping the row prints under
#   var        identifier, resolved to prose through lbl() and never printed
#   kind       "cont" -> median with interquartile range, "bin" -> n and per cent
#   source     "cand" (engineered 0/1 column) or a raw-value recipe
#   digits     decimal places for a continuous row
#   miss       the field label in the missingness part that supplies this row's
#              missingness, or None to compute it from the row's own validity rule
# ---------------------------------------------------------------------------
ROWS: list[dict] = [
    dict(block="Demographics", var="AGE_NUMBER", kind="cont", source="raw", digits=0,
         miss="Age, years"),
    dict(block="Demographics", var="age_45_64", kind="bin", source="cand",
         miss="Age, years"),
    dict(block="Demographics", var="age_65_74", kind="bin", source="cand",
         miss="Age, years"),
    dict(block="Demographics", var="age_75plus", kind="bin", source="cand",
         miss="Age, years"),
    dict(block="Demographics", var="female", kind="bin", source="cand", miss="Sex"),

    dict(block="Mechanism", var="mech_mvc", kind="bin", source="cand",
         miss="External injury mechanism code"),
    dict(block="Mechanism", var="mech_fall", kind="bin", source="cand",
         miss="External injury mechanism code"),
    dict(block="Mechanism", var="mech_assault", kind="bin", source="cand",
         miss="External injury mechanism code"),
    dict(block="Mechanism", var="risk_penetrating", kind="bin", source="cand",
         miss="External injury mechanism code"),
    dict(block="Mechanism", var="is_transfer", kind="bin", source="cand",
         miss="Transfer from another facility"),

    dict(block="Emergency department vital signs", var="SBP2", kind="cont",
         source="raw", digits=0, miss="Systolic blood pressure, mmHg"),
    dict(block="Emergency department vital signs", var="P2", kind="cont",
         source="raw", digits=0, miss="Heart rate, beats/min"),
    dict(block="Emergency department vital signs", var="RR2", kind="cont",
         source="raw", digits=0, miss="Respiratory rate, breaths/min"),
    dict(block="Emergency department vital signs", var="OX2", kind="cont",
         source="raw", digits=0, miss="Oxygen saturation, %"),
    dict(block="Emergency department vital signs", var="temperature_raw", kind="cont",
         source="temperature", digits=1, miss="Temperature, degrees Celsius"),
    dict(block="Emergency department vital signs", var="shock_index", kind="cont",
         source="shock_index", digits=2,
         miss="Shock index (heart rate divided by systolic blood pressure)"),
    dict(block="Emergency department vital signs", var="sbp_hypotensive", kind="bin",
         source="cand", miss="Systolic blood pressure, mmHg"),
    dict(block="Emergency department vital signs", var="sbp_hypertensive", kind="bin",
         source="cand", miss="Systolic blood pressure, mmHg"),
    dict(block="Emergency department vital signs", var="hr_bradycardic", kind="bin",
         source="cand", miss="Heart rate, beats/min"),
    dict(block="Emergency department vital signs", var="hr_tachycardic", kind="bin",
         source="cand", miss="Heart rate, beats/min"),
    dict(block="Emergency department vital signs", var="rr_low", kind="bin",
         source="cand", miss="Respiratory rate, breaths/min"),
    dict(block="Emergency department vital signs", var="rr_high", kind="bin",
         source="cand", miss="Respiratory rate, breaths/min"),
    dict(block="Emergency department vital signs", var="hypoxic", kind="bin",
         source="cand", miss="Oxygen saturation, %"),
    dict(block="Emergency department vital signs", var="si_elevated", kind="bin",
         source="cand",
         miss="Shock index (heart rate divided by systolic blood pressure)"),
    dict(block="Emergency department vital signs", var="si_high", kind="bin",
         source="cand",
         miss="Shock index (heart rate divided by systolic blood pressure)"),
    dict(block="Emergency department vital signs", var="temp_hypothermic", kind="bin",
         source="cand", miss="Temperature, degrees Celsius"),

    dict(block="Glasgow Coma Scale and pupils", var="GCS2", kind="cont", source="raw",
         digits=0, miss="Glasgow Coma Scale total"),
    dict(block="Glasgow Coma Scale and pupils", var="MOTOR2", kind="cont", source="raw",
         digits=0, miss="Glasgow Coma Scale motor"),
    dict(block="Glasgow Coma Scale and pupils", var="gcs_severe", kind="bin",
         source="cand", miss="Glasgow Coma Scale total"),
    dict(block="Glasgow Coma Scale and pupils", var="gcs_moderate", kind="bin",
         source="cand", miss="Glasgow Coma Scale total"),
    dict(block="Glasgow Coma Scale and pupils", var="gcs_motor_low", kind="bin",
         source="cand", miss="Glasgow Coma Scale motor"),
    dict(block="Glasgow Coma Scale and pupils", var="gcs_motor_high", kind="bin",
         source="cand", miss="Glasgow Coma Scale motor"),
    dict(block="Glasgow Coma Scale and pupils", var="tbi_pupil_none", kind="bin",
         source="cand", miss="Pupillary response"),

    dict(block="Comorbidities", var="cc_anticoag", kind="bin", source="cand",
         miss="Anticoagulant therapy"),
    dict(block="Comorbidities", var="cc_bleeding", kind="bin", source="cand",
         miss="Bleeding disorder"),

    dict(block="Injury severity", var="iss_cont", kind="cont", source="iss", digits=0,
         miss=None),
    dict(block="Injury severity", var="max_ais_head", kind="cont", source="ais_head",
         digits=0, miss=None),
    dict(block="Injury severity", var="max_ais_other", kind="cont", source="ais_other",
         digits=0, miss=None),
    dict(block="Injury severity", var="major_extracranial", kind="bin", source="cand",
         miss=None),
]

BLOCK_ORDER = ["Demographics", "Mechanism", "Emergency department vital signs",
               "Glasgow Coma Scale and pupils", "Comorbidities", "Injury severity"]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def smd_binary(p1: float, p0: float) -> float:
    d = np.sqrt((p1 * (1 - p1) + p0 * (1 - p0)) / 2.0)
    return 0.0 if d == 0 else float((p1 - p0) / d)


def smd_continuous(m1: float, m0: float, s1: float, s0: float) -> float:
    d = np.sqrt((s1 ** 2 + s0 ** 2) / 2.0)
    return 0.0 if d == 0 else float((m1 - m0) / d)


def fmt_n_pct(k: int, d: int) -> str:
    return f"{k:,} ({100.0 * k / d:.1f}%)" if d else "0 (0.0%)"


def fmt_med_iqr(med: float, q1: float, q3: float, digits: int) -> str:
    f = f"{{:,.{digits}f}}"
    return f"{f.format(med)} (IQR {f.format(q1)} to {f.format(q3)})"


def continuous_row(values: pd.Series, valid: pd.Series, y: pd.Series,
                   digits: int) -> dict:
    """Median with interquartile range overall and by group, plus the SMD."""
    v = values[valid].astype(float)
    yy = y[valid].astype(int)
    v1, v0 = v[yy == 1], v[yy == 0]

    def q(s):
        return dict(median=float(np.median(s)),
                    q1=float(np.quantile(s, 0.25)),
                    q3=float(np.quantile(s, 0.75)))

    qa, q1_, q0_ = q(v), q(v1), q(v0)
    smd = smd_continuous(float(v1.mean()), float(v0.mean()),
                         float(v1.std(ddof=1)), float(v0.std(ddof=1)))
    return dict(
        n_overall=int(len(v)), n_received=int(len(v1)), n_not_received=int(len(v0)),
        overall=qa, received=q1_, not_received=q0_,
        overall_txt=fmt_med_iqr(qa["median"], qa["q1"], qa["q3"], digits),
        received_txt=fmt_med_iqr(q1_["median"], q1_["q1"], q1_["q3"], digits),
        not_received_txt=fmt_med_iqr(q0_["median"], q0_["q1"], q0_["q3"], digits),
        smd=smd,
    )


def binary_row(flag: pd.Series, y: pd.Series) -> dict:
    """n and per cent overall and by group, plus the SMD."""
    f = flag.astype(int)
    yy = y.astype(int)
    d_all, d1, d0 = int(len(f)), int((yy == 1).sum()), int((yy == 0).sum())
    k_all = int(f.sum())
    k1 = int(f[yy == 1].sum())
    k0 = int(f[yy == 0].sum())
    smd = smd_binary(k1 / d1 if d1 else np.nan, k0 / d0 if d0 else np.nan)
    return dict(
        n_overall=d_all, n_received=d1, n_not_received=d0,
        overall={"n": k_all, "N": d_all},
        received={"n": k1, "N": d1},
        not_received={"n": k0, "N": d0},
        overall_txt=fmt_n_pct(k_all, d_all),
        received_txt=fmt_n_pct(k1, d1),
        not_received_txt=fmt_n_pct(k0, d0),
        smd=smd,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    C.start_log(SCRIPT)
    C.banner("PAT-TBI v6 Phase 0 step 6a - the statistics behind manuscript Table 1")
    print("Closes finding #12 (fifteen empty rows) and the Table 1 half of finding #2")
    print("(six cutpoint labels that contradicted the code).")

    # -- inputs ------------------------------------------------------------
    part_path = C.V6_PARTS / "missingness.json"
    if not part_path.is_file():
        raise SystemExit(
            f"{part_path} is absent. Run 04_missingness.py first: Table 1 takes its "
            f"missingness column from that part.")
    miss_part = json.loads(part_path.read_text())
    miss_lookup = {r["field"]: r for r in miss_part["predictor_missingness"]}

    cand = pd.read_parquet(C.CANDIDATES)
    raw = C.load_cohort_raw()
    n = len(cand)

    if n != C.V4_FROZEN["n_cohort"]:
        raise AssertionError(f"cohort n = {n:,}, expected {C.V4_FROZEN['n_cohort']:,}")
    if len(raw) != n:
        raise AssertionError("cohort_raw and candidates differ in length")
    if not (raw["PUF_YEAR"].values == cand["PUF_YEAR"].values).all() \
            or not (raw["inc_key"].values == cand["inc_key"].values).all():
        raise AssertionError("cohort_raw and candidates are not row-aligned")

    y = cand[C.OUTCOME].astype(int)
    n_events = int(y.sum())
    want = C.V4_FROZEN["n_train_events"] + C.V4_FROZEN["n_test_events"]
    if n_events != want:
        raise AssertionError(f"events {n_events:,}, expected {want:,}")
    n_no = n - n_events

    print(f"\n  analytic cohort                       : {n:,}")
    print(f"  received platelets within four hours  : {n_events:,} "
          f"({100.0 * n_events / n:.2f}%)")
    print(f"  did not receive platelets             : {n_no:,} "
          f"({100.0 * n_no / n:.2f}%)")
    print(f"  events reconcile                      : "
          f"{C.V4_FROZEN['n_train_events']:,} derivation plus "
          f"{C.V4_FROZEN['n_test_events']:,} validation equals {want:,}")

    C.banner("Why the between-group column is a standardized difference")
    print("  At this cohort size every two-group significance test falls below the")
    print("  reporting floor. A p-value would print as a floor value on almost every row,")
    print("  separate nothing, and say nothing about the size of a difference. The")
    print("  standardized mean difference is scale free and does not grow with the sample,")
    print(f"  so an absolute value above {SMD_THRESHOLD:.2f} marks a difference worth")
    print("  reading. No p-value is computed or reported in this table.")

    C.banner("Label corrections carried into this table - finding #2")
    print("  Every label below is read from _common.LABELS, which was transcribed off")
    print("  v4/code/02_engineer_features.py line by line. The v4 14_table1.py LABEL_MAP")
    print("  disagreed with the code on these rows:")
    points = C.load_points()
    for var, wrong in V4_WRONG_LABELS.items():
        flag = " [in the final score]" if var in points else ""
        print(f"    was : {wrong}")
        print(f"    now : {C.label(var)}{flag}")

    # -- raw value recipes -------------------------------------------------
    sbp = pd.to_numeric(raw["SBP2"], errors="coerce")
    hr = pd.to_numeric(raw["P2"], errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        si_raw = hr / sbp.where(sbp > 0)
    temp = pd.to_numeric(raw["temperature_raw"], errors="coerce")
    iss = pd.to_numeric(raw["ISS"], errors="coerce")
    ais_head = pd.to_numeric(raw["max_ais_head"], errors="coerce")
    ais_other = pd.to_numeric(raw["max_ais_other"], errors="coerce")

    def source_values(row: dict) -> tuple[pd.Series, pd.Series]:
        """(values, valid) for a continuous row, using the recorded value only."""
        src, var = row["source"], row["var"]
        if src == "raw":
            v = pd.to_numeric(raw[var], errors="coerce")
            return v, v.notna()
        if src == "temperature":
            # The engineering step only accepts a reading of 20 degrees or above as a
            # Celsius value; anything below it cannot set the hypothermia flag.
            return temp, temp.notna() & (temp >= 20)
        if src == "shock_index":
            return si_raw, si_raw.notna()
        if src == "iss":
            return iss, iss.notna()
        if src == "ais_head":
            # AIS severity 9 is "unknown"; 0 means no head injury was coded at all.
            return ais_head, ais_head.between(1, 6)
        if src == "ais_other":
            # 0 is a real finding here (no extracranial injury coded); 9 is unknown.
            return ais_other, ais_other.between(0, 6)
        raise RuntimeError(f"unknown continuous source {src!r}")

    # -- build the rows ----------------------------------------------------
    leftovers = set(C.V3_LEFTOVER_ROWS)
    out_rows: list[dict] = []
    part_rows: list[dict] = []

    for row in ROWS:
        var, kind = row["var"], row["kind"]
        if var in leftovers:
            raise AssertionError(
                f"{var!r} is a v3 leftover and cannot be a Table 1 row")
        label = lbl(var)

        if kind == "bin":
            if var not in cand.columns:
                raise AssertionError(
                    f"{var!r} is not a column of candidates.parquet, so no row can be "
                    f"built for it; this is exactly the defect that produced the v4 "
                    f"empty rows")
            stats = binary_row(cand[var], y)
            n_missing = 0 if row["miss"] is None \
                else int(miss_lookup[row["miss"]]["uninformative"]["n"])
        else:
            values, valid = source_values(row)
            stats = continuous_row(values, valid, y, row["digits"])
            n_missing = n - int(valid.sum())
            if row["miss"] is not None:
                expect = int(miss_lookup[row["miss"]]["uninformative"]["n"])
                if n_missing != expect:
                    raise AssertionError(
                        f"{label}: missingness {n_missing:,} computed here disagrees "
                        f"with {expect:,} in the missingness part")

        if stats["n_received"] + stats["n_not_received"] != stats["n_overall"]:
            raise AssertionError(f"{label}: group sizes do not sum to the denominator")

        out_rows.append(dict(
            Block=row["block"],
            Variable=label,
            Summary="median (IQR)" if kind == "cont" else "n (%)",
            Missing_n=n_missing,
            Missing_pct=round(100.0 * n_missing / n, 2),
            N_overall=stats["n_overall"],
            Overall=stats["overall_txt"],
            N_received=stats["n_received"],
            Received_platelets=stats["received_txt"],
            N_not_received=stats["n_not_received"],
            No_platelets=stats["not_received_txt"],
            SMD=round(stats["smd"], 3),
        ))
        part_rows.append(dict(
            block=row["block"],
            label=label,
            summary="quantile" if kind == "cont" else "count",
            overall=stats["overall"],
            received=stats["received"],
            not_received=stats["not_received"],
            analysed={"n": stats["n_overall"], "N": n},
            missing={"n": n_missing, "N": n},
            smd={"value": round(stats["smd"], 3)},
        ))

    tab = pd.DataFrame(out_rows)

    # -- gates -------------------------------------------------------------
    C.banner("Gate")

    emitted_vars = {r["var"] for r in ROWS}
    intersect = emitted_vars & leftovers
    if intersect:
        raise AssertionError(f"v3 leftovers reached the table: {sorted(intersect)}")
    leaked = [v for v in C.V3_LEFTOVER_ROWS if v in set(tab["Variable"])]
    if leaked:
        raise AssertionError(f"v3 leftover identifiers printed as labels: {leaked}")

    import re
    snake = re.compile(r"\b[a-z0-9]+_[a-z0-9_]+\b")
    bad_label = [v for v in tab["Variable"] if snake.search(v)]
    if bad_label:
        raise AssertionError(f"snake_case identifiers leaked into labels: {bad_label}")
    bad_upper = [v for v in tab["Variable"] if v in emitted_vars]
    if bad_upper:
        raise AssertionError(f"raw identifiers printed as labels: {bad_upper}")

    count_cols = ["Missing_n", "N_overall", "N_received", "N_not_received", "SMD"]
    if tab[count_cols].isna().any().any():
        nulls = tab.loc[tab[count_cols].isna().any(axis=1), "Variable"].tolist()
        raise AssertionError(f"rows carrying a null count: {nulls}")
    text_cols = ["Overall", "Received_platelets", "No_platelets"]
    empty = tab.loc[
        tab[text_cols].isna().any(axis=1)
        | (tab[text_cols].astype(str).apply(lambda s: s.str.strip() == "")).any(axis=1),
        "Variable"].tolist()
    if empty:
        raise AssertionError(f"rows carrying an empty cell: {empty}")
    if (tab["N_overall"] <= 0).any():
        raise AssertionError("a row has an empty denominator")

    print(f"  [PASS] analytic cohort {n:,}")
    print(f"  [PASS] outcome events {n_events:,} equals "
          f"{C.V4_FROZEN['n_train_events']:,} plus {C.V4_FROZEN['n_test_events']:,}")
    print(f"  [PASS] rows emitted {len(tab)}, in {tab['Block'].nunique()} blocks")
    print(f"  [PASS] none of the {len(C.V3_LEFTOVER_ROWS)} v3 leftovers appears")
    print("  [PASS] no row carries a null count or an empty cell")
    print("  [PASS] every label is clinical prose, no identifier survived")
    print("  [PASS] no p-value computed; the between-group column is a standardized "
          "difference")

    # -- print the table ---------------------------------------------------
    C.banner("Table 1 - cohort characteristics by platelet receipt within four hours")
    for block in BLOCK_ORDER:
        sub = tab[tab["Block"] == block]
        if sub.empty:
            continue
        print()
        print(f"  {block}")
        print(f"    {'variable':<58} {'overall':>26} {'received':>26} "
              f"{'not received':>26} {'missing':>16} {'SMD':>7}")
        for _, r in sub.iterrows():
            print(f"    {r['Variable']:<58} {r['Overall']:>26} "
                  f"{r['Received_platelets']:>26} {r['No_platelets']:>26} "
                  f"{f'{r.Missing_n:,} ({r.Missing_pct:.1f}%)':>16} {r['SMD']:>7.3f}")

    strong = tab.loc[tab["SMD"].abs() >= SMD_THRESHOLD].sort_values(
        "SMD", key=lambda s: s.abs(), ascending=False)
    print()
    print(f"  rows whose absolute standardized difference reaches {SMD_THRESHOLD:.2f}: "
          f"{len(strong)} of {len(tab)}")
    for _, r in strong.iterrows():
        print(f"    {r['SMD']:+.3f}  {r['Variable']}")

    # -- outputs -----------------------------------------------------------
    C.banner("Writing outputs")
    (C.V6 / "tables").mkdir(parents=True, exist_ok=True)
    csv = C.V6 / "tables" / "table_v6_table1.csv"
    tab.to_csv(csv, index=False)
    print(f"  wrote {csv}  ({len(tab)} rows)")

    payload = dict(
        cohort=dict(
            overall={"n": n},
            received={"n": n_events, "N": n},
            not_received={"n": n_no, "N": n},
            rows={"value": len(tab)},
            blocks={"value": len(BLOCK_ORDER)},
        ),
        comparison=dict(
            threshold={"value": SMD_THRESHOLD},
            rows_above_threshold={"n": int(len(strong)), "N": int(len(tab))},
            method={"display": (
                "Groups are compared with the standardized mean difference rather than a "
                "significance test. At this cohort size every two-group test falls below "
                "the reporting floor, so it separates nothing and says nothing about the "
                "size of a difference. An absolute standardized difference above one "
                "tenth marks a difference worth reading.")},
        ),
        presentation={"display": (
            "Continuous rows are summarised as a median with its interquartile range from "
            "the recorded values, so a reference substitution made during feature "
            "engineering never enters a median. Binary rows are counts with their per "
            "cent of the column. The missingness column counts the encounters for which "
            "the underlying field was not recorded, or could not inform the row, and for "
            "which a reference level was substituted; a binary row therefore keeps a full "
            "denominator even where its source field was missing.")},
        labels={"display": (
            "Every row label is read from the shared label table, which was transcribed "
            "off the feature engineering code line by line. The earlier table disagreed "
            "with the code on six cutpoints, four of them in the final score, and carried "
            "fifteen rows for variables that the candidate matrix does not contain.")},
        rows=part_rows,
        labels_pending_in_common=sorted(LOCAL_LABELS[v] for v in _LOCAL_USED),
    )
    path = C.write_part(PART, payload)
    print(f"  wrote {path}")

    if _LOCAL_USED:
        print()
        print("  NOTE: _common.LABELS carries no entry for the raw physiology columns "
              "behind the")
        print("  vital-sign predictors. A local label table was used for these:")
        for v in _LOCAL_USED:
            print(f"    {v:<20} -> {LOCAL_LABELS[v]}")

    print("\n06a_table1_stats complete.")


if __name__ == "__main__":
    main()
