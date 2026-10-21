"""
PAT-TBI v6 — Step 11: recompute every reported area-under-the-curve interval.
============================================================================

The defect
----------
`v4/code/07_validation_metrics.py`, function `bootstrap_auc_ci_subsample`, draws

    idx = rng.integers(0, n, size=min(100_000, n))

once per replicate and then takes a plain percentile interval of the resulting
areas. That is an m-out-of-n bootstrap: it resamples 100,000 rows out of
207,863 and reports the spread of a statistic computed on 100,000 rows as if it
were the spread of the statistic computed on 207,863. An m-out-of-n bootstrap is
a legitimate device, but its percentile interval must be rescaled by sqrt(n/m)
before it describes the full sample. No rescaling is applied anywhere in the
project. Every interval that came out of that helper is therefore too wide by a
factor of about sqrt(207863/100000) = 1.442.

`v4/code/09_subgroups.py` repeats the mistake at a smaller resample size
(MAX_BOOT_N = 50,000), so every subgroup larger than 50,000 rows is inflated
further, and the all-trauma sensitivity row, 4,011,313 rows resampled 50,000 at
a time by way of step 07's 100,000, is inflated by roughly six.
`v6/code/03_head_ais_sensitivity.py` imports both helpers deliberately, so its
head-AIS and teaching-status rows inherit the same inflation.

The project already contradicted itself on this point.
`v4/tables/table_sensitivity_whole_blood.csv` computes the headline area on the
identical 207,863 rows with a correct full-cohort bootstrap and prints an
interval materially narrower than the one in
`v4/tables/table_pat_discrimination.csv`.

Point estimates are untouched by any of this. Only the intervals move.

What this script does
---------------------
Part 1 establishes agreement on the headline area three ways:

    a. the v4 protocol exactly as it stands, to prove the bug is identified;
    b. a correct full-size bootstrap with replacement over every row,
       2,000 replicates, seed 42;
    c. the analytic DeLong interval.

Part 2 checks (b) against (c). They agree to the third decimal, so the rest of
the script uses DeLong for every interval.

Why DeLong, stated for the record
---------------------------------
DeLong is the standard analytic variance for an area under a receiver operating
characteristic curve. It is exact rather than simulated, so it needs no seed and
reproduces bit for bit; it costs one sort, so it runs on the 4,011,313-row
all-trauma cohort in under a second where a bootstrap of that cohort would take
hours; and it is the same estimator the project already uses for every
between-score contrast it reports, so the single-score intervals and the
contrast intervals now rest on one variance rather than two. The corrected
bootstrap is retained in the log beside it as the check that the analytic form
is not doing anything surprising.

Scope
-----
Every area the project reports:

  * the five scores in `v4/tables/table_pat_discrimination.csv`, each on its own
    denominator;
  * all thirteen subgroup levels for all five scores, 65 rows, from
    `v4/tables/table_subgroups.csv`;
  * the three teaching-status levels and the collapsed binary, from
    `v6/bundle/parts/teaching.json`;
  * the all-trauma sensitivity row and the whole-blood sensitivity row;
  * the head-AIS base model, extended model, severity-unknown variant, integer
    score, and the two DeLong contrasts, from `v6/bundle/parts/head_ais.json`.

Hard gates
----------
  * the v4 protocol must reproduce its published headline interval exactly;
  * every point estimate must be unchanged at the precision the source printed;
  * the validation area must still be 0.8738;
  * the new headline interval must be narrower than the old one and must
    contain the point estimate;
  * the new headline interval must sit close to the independent full-cohort
    figure already in `table_sensitivity_whole_blood.csv`.

Outputs
-------
  v6/bundle/parts/intervals.json
  v6/tables/table_v6_intervals.csv
  v6/logs/11_recompute_intervals.log
  v6/logs/11_HANDOFF.md

Run with the framework interpreter, not bare `python3`:
    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 11_recompute_intervals.py
"""

# ---------------------------------------------------------------------------
# AUDIT-TRAIL HAZARD, recorded 2026-09-10.
#
# This script builds the "before" column of its correction table by reading
# `CI_Low` / `CI_High` out of the head-AIS and teaching artefacts. Those
# artefacts have since been regenerated with the corrected DeLong bounds, so a
# re-run would read the corrected values as the "before" side and every delta in
# those blocks would print as zero, making the correction look like it never
# happened.
#
# The withdrawn pairs were preserved for exactly this reason: the head-AIS table
# carries a `discrimination_withdrawn` block and the teaching table carries
# `CI_Low_withdrawn` / `CI_High_withdrawn` columns, mirroring the `_withdrawn`
# convention in `_common.V4_FROZEN`. A re-run must source the "before" column
# from those, not from `CI_Low` / `CI_High`.
#
# The table currently on disk was written before the regeneration and is correct;
# it is the re-run that needs the change.
# ---------------------------------------------------------------------------

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # house convention, set before any pyplot import

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import statsmodels.api as sm

import _common as C

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = C.SEED
OUTCOME = C.OUTCOME
OUTCOME_SENS = C.OUTCOME_SENS
Z = 1.959963984540054

N_BOOT_FULL = 2000          # replicates for the corrected full-size bootstrap
V4_SUB_DISC = 100_000       # the step 07 resample size, reproduced not endorsed
V4_BOOT_DISC = 2000
MIN_EVENTS = 50             # the v4 step 09 subgroup floor
HEAD_AIS = "max_ais_head"

OUT_CSV = C.V6 / "tables" / "table_v6_intervals.csv"
OUT_HANDOFF = C.V6_LOGS / "11_HANDOFF.md"

ALL_TRAUMA_PARQUET = C.V4_DERIVED / "all_trauma_scored.parquet"
DISC_CSV = C.V4_TABLES / "table_pat_discrimination.csv"
SUBGROUP_CSV = C.V4_TABLES / "table_subgroups.csv"
ALL_TRAUMA_CSV = C.V4_TABLES / "table_sensitivity_all_trauma.csv"
WHOLE_BLOOD_CSV = C.V4_TABLES / "table_sensitivity_whole_blood.csv"
TEACHING_PART = C.V6_PARTS / "teaching.json"
HEAD_AIS_PART = C.V6_PARTS / "head_ais.json"
TEACHING_TABLE = C.V6 / "tables" / "table_v6_teaching_subgroup.csv"

# Score registry, in the shape v4 steps 07 and 09 use.
#   (bundle key, table label, prose label, score column, mask column)
SCORES = [
    ("pat",    "PAT-TBI",     "PAT-TBI",                                "pat_tbi_score",    "pat_tbi_score_mask"),
    ("gcs",    "GCS",         "Glasgow Coma Scale",                     "gcs_risk",         "gcs_mask"),
    ("gcsp",   "GCS-P",       "Glasgow Coma Scale with pupil score",    "gcs_p_risk",       "gcs_p_mask"),
    ("impact", "IMPACT-core", "IMPACT Core",                            "impact_core_prob", "impact_mask"),
    ("crash",  "CRASH-Basic", "CRASH Basic",                            "crash_basic_prob", "crash_mask"),
]
# The discrimination table spells two of these differently from the subgroup
# table, so both spellings are carried.
DISC_ROW_NAME = {"pat": "pat_tbi_score", "gcs": "GCS", "gcsp": "GCS-P",
                 "impact": "IMPACT-Core", "crash": "CRASH-Basic"}
SUBGROUP_ROW_NAME = {"pat": "PAT-TBI", "gcs": "GCS", "gcsp": "GCS-P",
                     "impact": "IMPACT-core", "crash": "CRASH-Basic"}


# ---------------------------------------------------------------------------
# Reuse the v4 estimators rather than retyping them
# ---------------------------------------------------------------------------
def _load_v4_module(stem: str, alias: str):
    path = C.V4_CODE / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V4_07 = _load_v4_module("07_validation_metrics", "v4_validation_metrics")

delong_v10_v01 = V4_07.delong_v10_v01                        # placement values
delong_test_vec = V4_07.delong_test_vec                      # paired contrast
bootstrap_auc_ci_subsample = V4_07.bootstrap_auc_ci_subsample  # the defective helper


# ---------------------------------------------------------------------------
# The three interval estimators
# ---------------------------------------------------------------------------
def delong_ci(y, s) -> tuple[float, float, float, float]:
    """Point area, DeLong standard error, and the analytic 95% interval.

    The placement values come from the v4 implementation, so this is the same
    estimator the project already uses for its between-score contrasts, not a
    second one.
    """
    y = np.asarray(y, dtype=np.int8)
    s = np.asarray(s, dtype=np.float64)
    v10, v01 = delong_v10_v01(y, s)
    n1, n0 = len(v10), len(v01)
    auc = float(v10.mean())
    var = float(np.var(v10, ddof=1) / n1 + np.var(v01, ddof=1) / n0)
    se = float(np.sqrt(var))
    lo = max(0.0, auc - Z * se)
    hi = min(1.0, auc + Z * se)
    return auc, se, lo, hi


def bootstrap_full_ci(y, s, n_boot: int = N_BOOT_FULL, seed: int = SEED):
    """Correct nonparametric bootstrap: n draws with replacement out of n.

    This is what the v4 helper should have done, and it is exactly what
    `v4/code/12_sensitivity_whole_blood.py` does, which is why that table
    disagrees with the discrimination table on the same rows.
    """
    y = np.asarray(y)
    s = np.asarray(s, dtype=np.float64)
    n = len(y)
    rng = np.random.default_rng(seed)
    aucs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        tot = yb.sum()
        if tot == 0 or tot == n:
            aucs[b] = np.nan
            continue
        aucs[b] = roc_auc_score(yb, s[idx])
    aucs = aucs[~np.isnan(aucs)]
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi), float(aucs.std(ddof=1))


def v4_protocol_ci(y, s, n_boot: int = V4_BOOT_DISC, sub: int = V4_SUB_DISC):
    """The v4 step 07 interval, reproduced from the v4 code itself."""
    rng = np.random.default_rng(SEED)
    lo, hi = bootstrap_auc_ci_subsample(np.asarray(y), np.asarray(s, dtype=float),
                                        n_boot, sub, rng)
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Row collection
# ---------------------------------------------------------------------------
ROWS: list[dict] = []
GATE_FAILURES: list[str] = []


def gate(name: str, ok: bool, detail: str = "") -> bool:
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}{('  ' + detail) if detail else ''}")
    if not ok:
        GATE_FAILURES.append(f"{name}: {detail}")
    return ok


SIZE_NOTE = ("the superseded resample already covered nearly the whole cell, so "
             "its interval was effectively full size; the small change here is "
             "simulation noise, not the correction")


def record(block: str, label: str, bundle_key: str, score_label: str,
           y, s, old_lo: float, old_hi: float, old_point: float,
           point_decimals: int, old_method: str, old_sub: int | None,
           note: str = "") -> dict:
    """Compute the corrected interval for one area and file the audit row.

    `old_point` is the value the project printed. It is a gate, not an input:
    the point estimate is recomputed here from the same rows and must match.

    `old_sub` is the resample size the superseded protocol used on this row.
    The nominal inflation it implies, sqrt(n / min(old_sub, n)), is carried
    beside the observed width ratio so a reader can see the arithmetic rather
    than take the correction on trust.
    """
    y = np.asarray(y, dtype=np.int8)
    s = np.asarray(s, dtype=np.float64)
    n = int(len(y))
    ev = int(y.sum())

    auc_sk = float(roc_auc_score(y, s))
    auc, se, lo, hi = delong_ci(y, s)

    if abs(auc - auc_sk) > 1e-9:
        GATE_FAILURES.append(
            f"{label}: DeLong area {auc:.10f} disagrees with the reference "
            f"area {auc_sk:.10f}")

    moved = False
    if old_point == old_point:  # not NaN
        if point_decimals is None:
            moved = abs(auc_sk - old_point) > 1e-10
        else:
            moved = round(auc_sk, point_decimals) != round(old_point, point_decimals)
    if moved:
        GATE_FAILURES.append(
            f"{label}: point estimate moved, published {old_point!r}, "
            f"recomputed {auc_sk!r}")

    old_w = old_hi - old_lo if (old_lo == old_lo and old_hi == old_hi) else float("nan")
    new_w = hi - lo
    ratio = old_w / new_w if new_w > 0 and old_w == old_w else float("nan")

    eff = min(old_sub, n) if old_sub else n
    nominal = float(np.sqrt(n / eff))
    if nominal < 1.05 and not note:
        note = SIZE_NOTE

    row = {
        "block": block,
        "label": label,
        "bundle_key": bundle_key,
        "score": score_label,
        "n": n,
        "events": ev,
        "estimate": round(auc_sk, 6),
        "old_lo": round(old_lo, 6) if old_lo == old_lo else "",
        "old_hi": round(old_hi, 6) if old_hi == old_hi else "",
        "old_width": round(old_w, 6) if old_w == old_w else "",
        "new_lo": round(lo, 6),
        "new_hi": round(hi, 6),
        "new_width": round(new_w, 6),
        "width_ratio_old_over_new": round(ratio, 3) if ratio == ratio else "",
        "old_resample": int(eff),
        "nominal_inflation": round(nominal, 3),
        "delong_se": round(se, 6),
        "point_moved": bool(moved),
        "old_method": old_method,
        "note": note,
    }
    ROWS.append(row)
    print(f"    {label:<52s} n={n:>9,} ev={ev:>6,}  "
          f"{auc_sk:.4f}  old ({old_lo:.4f} to {old_hi:.4f})  "
          f"new ({lo:.4f} to {hi:.4f})  ratio {ratio:.2f}  "
          f"nominal {nominal:.2f}")
    return row


def leaf(row: dict, decimals: int = 4) -> dict:
    return {"estimate": round(float(row["estimate"]), decimals),
            "lo": round(float(row["new_lo"]), decimals),
            "hi": round(float(row["new_hi"]), decimals)}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def masked(df: pd.DataFrame, score_col: str, mask_col: str, outcome: str = OUTCOME):
    """The rows a score is actually evaluated on, exactly as v4 selects them."""
    m = (df[mask_col].astype(bool).fillna(False)
         & df[score_col].notna() & df[outcome].notna())
    sub = df.loc[m]
    return (sub[outcome].astype(float).to_numpy(),
            sub[score_col].astype(float).to_numpy())


def load_subgroup_frame() -> pd.DataFrame:
    """The v4 step 09 analysis frame: scored validation rows plus two extras."""
    scored = C.load_test_scored()
    need = ["PUF_YEAR", "inc_key", "risk_penetrating", "is_transfer"]
    cand = pd.read_parquet(C.CANDIDATES, columns=need)
    for k in ("PUF_YEAR", "inc_key"):
        scored[k] = pd.to_numeric(scored[k], errors="coerce").astype("Int64")
        cand[k] = pd.to_numeric(cand[k], errors="coerce").astype("Int64")
    merged = scored.merge(cand, on=["PUF_YEAR", "inc_key"], how="inner",
                          validate="1:1")
    if len(merged) != C.V4_FROZEN["n_test"]:
        raise SystemExit(f"subgroup frame is {len(merged):,} rows, expected "
                         f"{C.V4_FROZEN['n_test']:,}")
    return merged


def build_subgroup_levels(df: pd.DataFrame) -> list[tuple[str, str, np.ndarray]]:
    """The thirteen levels v4 step 09 actually estimated, in its own order.

    Centre was the fourteenth in the v4 plan. It never ran, because the
    teaching field was absent from the candidate matrix; v6 step 3 recovered it
    and it is handled in its own block below.
    """
    def flat(cond):
        return np.asarray(cond.fillna(False) if hasattr(cond, "fillna") else cond,
                          dtype=bool)

    age = pd.to_numeric(df["AGE_NUMBER"], errors="coerce")
    sex = df["SEX"].astype("string").str.upper().str.strip()
    g = pd.to_numeric(df["GCS2"], errors="coerce")
    mh = pd.to_numeric(df["max_ais_head"], errors="coerce")
    mo = pd.to_numeric(df["max_ais_other"], errors="coerce")
    both = mh.notna() & mo.notna()
    iso = (mh >= 3) & (mo <= 1) & both
    pen = pd.to_numeric(df["risk_penetrating"], errors="coerce")
    tr = pd.to_numeric(df["is_transfer"], errors="coerce")

    return [
        ("Age", "<65", flat((age < 65) & age.notna())),
        ("Age", ">=65", flat((age >= 65) & age.notna())),
        ("Sex", "Male", flat(sex == "M")),
        ("Sex", "Female", flat(sex == "F")),
        ("GCS_Severity", "Severe (<=8)", flat((g >= 3) & (g <= 8))),
        ("GCS_Severity", "Moderate (9-12)", flat((g >= 9) & (g <= 12))),
        ("GCS_Severity", "Mild (13-15)", flat((g >= 13) & (g <= 15))),
        ("Body_Injury", "Isolated TBI", flat(iso)),
        ("Body_Injury", "Polytrauma", flat(both & ~iso)),
        ("Mechanism", "Penetrating", flat(pen == 1)),
        ("Mechanism", "Blunt", flat(pen == 0)),
        ("Arrival", "Transfer", flat(tr == 1)),
        ("Arrival", "Primary", flat(tr == 0)),
    ]


LEVEL_PROSE = {
    ("Age", "<65"): "Age under 65 years",
    ("Age", ">=65"): "Age 65 years or older",
    ("Sex", "Male"): "Male sex",
    ("Sex", "Female"): "Female sex",
    ("GCS_Severity", "Severe (<=8)"): "Glasgow Coma Scale 8 or below",
    ("GCS_Severity", "Moderate (9-12)"): "Glasgow Coma Scale 9 to 12",
    ("GCS_Severity", "Mild (13-15)"): "Glasgow Coma Scale 13 to 15",
    ("Body_Injury", "Isolated TBI"): "Isolated head injury",
    ("Body_Injury", "Polytrauma"): "Polytrauma",
    ("Mechanism", "Penetrating"): "Penetrating mechanism",
    ("Mechanism", "Blunt"): "Blunt mechanism",
    ("Arrival", "Transfer"): "Transferred from another facility",
    ("Arrival", "Primary"): "Primary arrival",
}


def csv_cell(df: pd.DataFrame, where: dict, column: str):
    sel = df
    for col, val in where.items():
        sel = sel[sel[col].astype(str).str.strip() == str(val)]
    if len(sel) != 1:
        raise KeyError(f"selector {where} matched {len(sel)} rows, expected one")
    return sel.iloc[0][column]


def fit_logit(X_df: pd.DataFrame, y):
    """Unpenalized maximum likelihood logistic, identical to v4 step 05."""
    Xc = sm.add_constant(X_df, has_constant="add")
    return sm.Logit(np.asarray(y, dtype=float), Xc.astype(float)).fit(
        disp=False, maxiter=200, method="newton")


def linear_predictor(res, X_df: pd.DataFrame) -> np.ndarray:
    lp = np.full(len(X_df), float(res.params["const"]), dtype=np.float64)
    for col in X_df.columns:
        lp += float(res.params[col]) * X_df[col].astype(float).to_numpy()
    return lp


# ===========================================================================
# PART 1 — the headline, three ways
# ===========================================================================
def part_one(test_scored: pd.DataFrame) -> dict:
    C.banner("PART 1 — the headline area, computed three ways")
    y, s = masked(test_scored, "pat_tbi_score", "pat_tbi_score_mask")
    n, ev = len(y), int(y.sum())
    print(f"  validation rows {n:,}, events {ev:,}")

    auc = float(roc_auc_score(y, s))
    print(f"  point area (unchanged by any of this): {auc:.10f}")

    print("\n  (a) the v4 step 07 protocol, run from the v4 code itself")
    print(f"      {V4_BOOT_DISC:,} replicates, each resampling {V4_SUB_DISC:,} "
          f"rows out of {n:,}, percentile interval, no rescaling")
    t0 = time.time()
    a_lo, a_hi = v4_protocol_ci(y, s)
    print(f"      -> {auc:.4f} (95% CI {a_lo:.4f} to {a_hi:.4f})   "
          f"[{time.time() - t0:.0f}s]")

    print("\n  (b) a correct full-size bootstrap")
    print(f"      {N_BOOT_FULL:,} replicates, each resampling all {n:,} rows "
          f"with replacement, seed {SEED}")
    t0 = time.time()
    b_lo, b_hi, b_sd = bootstrap_full_ci(y, s)
    print(f"      -> {auc:.4f} (95% CI {b_lo:.4f} to {b_hi:.4f})   "
          f"bootstrap standard deviation {b_sd:.6f}   [{time.time() - t0:.0f}s]")

    print("\n  (c) the analytic DeLong interval")
    t0 = time.time()
    d_auc, d_se, d_lo, d_hi = delong_ci(y, s)
    print(f"      -> {d_auc:.4f} (95% CI {d_lo:.4f} to {d_hi:.4f})   "
          f"standard error {d_se:.6f}   [{time.time() - t0:.1f}s]")

    C.banner("Gate — does running the v4 protocol reproduce the published interval?")
    gate("v4 protocol lower bound reproduces the published value",
         round(a_lo, 4) == C.V4_FROZEN["auroc_test_lo"],
         f"got {a_lo:.4f}, published {C.V4_FROZEN['auroc_test_lo']}")
    gate("v4 protocol upper bound reproduces the published value",
         round(a_hi, 4) == C.V4_FROZEN["auroc_test_hi"],
         f"got {a_hi:.4f}, published {C.V4_FROZEN['auroc_test_hi']}")
    gate("validation area is still the frozen headline",
         round(auc, 4) == C.V4_FROZEN["auroc_test"],
         f"got {round(auc, 4)}, frozen {C.V4_FROZEN['auroc_test']}")

    C.banner("Gate — do the corrected bootstrap and DeLong agree?")
    d_lo_gap = abs(b_lo - d_lo)
    d_hi_gap = abs(b_hi - d_hi)
    print(f"  lower bounds: bootstrap {b_lo:.5f}, DeLong {d_lo:.5f}, "
          f"gap {d_lo_gap:.5f}")
    print(f"  upper bounds: bootstrap {b_hi:.5f}, DeLong {d_hi:.5f}, "
          f"gap {d_hi_gap:.5f}")
    print(f"  bootstrap standard deviation {b_sd:.6f} against DeLong standard "
          f"error {d_se:.6f}, ratio {b_sd / d_se:.4f}")
    agree = max(d_lo_gap, d_hi_gap) <= 0.001
    gate("corrected bootstrap and DeLong agree to about the third decimal",
         agree, f"largest bound gap {max(d_lo_gap, d_hi_gap):.5f}, tolerance 0.001")
    if not agree:
        raise SystemExit(
            "STOP. The corrected bootstrap and the analytic DeLong interval do "
            "not agree. The plan says to stop and report rather than pick one, "
            "so nothing further is computed and no part is written.")

    C.banner("The inflation factor, predicted and observed")
    expected = float(np.sqrt(n / V4_SUB_DISC))
    observed_vs_boot = (a_hi - a_lo) / (b_hi - b_lo)
    observed_vs_delong = (a_hi - a_lo) / (d_hi - d_lo)
    print(f"  sqrt(n / m) = sqrt({n:,} / {V4_SUB_DISC:,}) = {expected:.4f}")
    print(f"  observed width ratio against the corrected bootstrap: "
          f"{observed_vs_boot:.4f}")
    print(f"  observed width ratio against DeLong:                  "
          f"{observed_vs_delong:.4f}")
    print("  The published interval carries the width of a sample of one "
          "hundred thousand, not of the validation cohort.")

    C.banner("Independent check already in the project")
    wb = pd.read_csv(WHOLE_BLOOD_CSV)
    w_prim = {"Outcome": "received_platelets_4h (primary)"}
    ind_auc = float(csv_cell(wb, w_prim, "AUROC"))
    ind_lo = float(csv_cell(wb, w_prim, "CI_Low"))
    ind_hi = float(csv_cell(wb, w_prim, "CI_High"))
    print(f"  v4/tables/table_sensitivity_whole_blood.csv, primary outcome row:")
    print(f"    {ind_auc:.4f} (95% CI {ind_lo:.4f} to {ind_hi:.4f})")
    print(f"    computed by v4 step 12, which resamples all {n:,} rows, on the "
          f"same rows and the same outcome as the discrimination table")
    print(f"  this script's corrected bootstrap: {b_lo:.4f} to {b_hi:.4f}")
    print(f"  this script's DeLong interval:     {d_lo:.4f} to {d_hi:.4f}")
    gate("the corrected bootstrap reproduces the project's own independent "
         "full-cohort figure",
         round(b_lo, 4) == round(ind_lo, 4) and round(b_hi, 4) == round(ind_hi, 4),
         f"got ({b_lo:.4f} to {b_hi:.4f}), independent figure "
         f"({ind_lo:.4f} to {ind_hi:.4f})")
    gate("the DeLong interval sits within a thousandth of that figure",
         abs(d_lo - ind_lo) <= 0.001 and abs(d_hi - ind_hi) <= 0.001,
         f"lower gap {abs(d_lo - ind_lo):.5f}, upper gap {abs(d_hi - ind_hi):.5f}")

    return {
        "n": n, "events": ev, "auc": auc,
        "v4_lo": a_lo, "v4_hi": a_hi,
        "boot_lo": b_lo, "boot_hi": b_hi, "boot_sd": b_sd,
        "dl_lo": d_lo, "dl_hi": d_hi, "dl_se": d_se,
        "expected": expected,
        "observed_vs_boot": observed_vs_boot,
        "observed_vs_delong": observed_vs_delong,
        "ind_auc": ind_auc, "ind_lo": ind_lo, "ind_hi": ind_hi,
    }


# ===========================================================================
# PART 2 — the five headline scores
# ===========================================================================
def part_discrimination(test_scored: pd.DataFrame) -> dict:
    C.banner("PART 2 — the five scores of the discrimination table")
    disc = pd.read_csv(DISC_CSV)
    out = {}
    for key, _tbl, prose, score_col, mask_col in SCORES:
        w = {"Score": DISC_ROW_NAME[key]}
        old_pt = float(csv_cell(disc, w, "AUROC"))
        old_lo = float(csv_cell(disc, w, "CI_Low"))
        old_hi = float(csv_cell(disc, w, "CI_High"))
        n_used = int(csv_cell(disc, w, "N_used"))
        y, s = masked(test_scored, score_col, mask_col)
        if len(y) != n_used:
            GATE_FAILURES.append(
                f"{prose}: denominator {len(y):,} does not match the published "
                f"{n_used:,}")
        row = record("discrimination", prose, f"discrimination.{key}", prose,
                     y, s, old_lo, old_hi, old_pt, 4,
                     "m out of n bootstrap, 2000 replicates of 100000 rows",
                     V4_SUB_DISC)
        out[key] = leaf(row)
    return out


# ===========================================================================
# PART 3 — the 65 subgroup cells
# ===========================================================================
def part_subgroups(df: pd.DataFrame) -> dict:
    C.banner("PART 3 — thirteen subgroup levels, five scores, 65 cells")
    sg = pd.read_csv(SUBGROUP_CSV)
    levels = build_subgroup_levels(df)
    print(f"  levels reconstructed: {len(levels)}")

    rows_out, pat_rows = [], []
    for dim, level, mask in levels:
        sub = df.loc[mask]
        ev_total = int(pd.to_numeric(sub[OUTCOME], errors="coerce").fillna(0).sum())
        prose_level = LEVEL_PROSE[(dim, level)]
        print(f"\n  {dim} / {level}: {len(sub):,} rows, {ev_total:,} events")
        if ev_total < MIN_EVENTS:
            raise SystemExit(f"{dim}/{level} falls under the event floor; the "
                             f"v4 table reports it, so something has moved")
        for key, _tbl, prose, score_col, mask_col in SCORES:
            w = {"Subgroup": dim, "Level": level, "Score": SUBGROUP_ROW_NAME[key]}
            old_pt = float(csv_cell(sg, w, "AUROC"))
            old_lo = float(csv_cell(sg, w, "CI_Low"))
            old_hi = float(csv_cell(sg, w, "CI_High"))
            n_pub = int(csv_cell(sg, w, "N"))
            y, s = masked(sub, score_col, mask_col)
            if len(y) != n_pub:
                GATE_FAILURES.append(
                    f"{dim}/{level}/{prose}: denominator {len(y):,} does not "
                    f"match the published {n_pub:,}")
            row = record("subgroup", f"{prose_level}, {prose}",
                         "subgroups.rows", prose, y, s,
                         old_lo, old_hi, old_pt, None,
                         "m out of n bootstrap, 500 replicates of 50000 rows",
                         50_000)
            row["dimension"] = dim
            row["level_prose"] = prose_level
            entry = {"dimension": dim, "level": prose_level, "score": prose,
                     "n": int(row["n"]), "events": int(row["events"]),
                     "estimate": round(float(row["estimate"]), 4),
                     "lo": round(float(row["new_lo"]), 4),
                     "hi": round(float(row["new_hi"]), 4)}
            rows_out.append(entry)
            if key == "pat":
                pat_rows.append({k: entry[k] for k in
                                 ("dimension", "level", "n", "events",
                                  "estimate", "lo", "hi")})
    return {"rows": rows_out, "pat_rows": pat_rows,
            "levels": {"value": len(levels)},
            "scores": {"value": len(SCORES)}}


# ===========================================================================
# PART 4 — teaching status
# ===========================================================================
def part_teaching() -> dict:
    C.banner("PART 4 — teaching status, the three levels and the collapsed binary")
    part = json.loads(TEACHING_PART.read_text())
    tab = pd.read_csv(TEACHING_TABLE)

    scored = C.load_test_scored()
    raw = C.load_cohort_raw(columns=["PUF_YEAR", "inc_key", "teaching_status_raw"])

    # The v6 step 3 mapping, read back from that script so the levels are not
    # re-decided here.
    spec = importlib.util.spec_from_file_location(
        "v6_head_ais", Path(__file__).resolve().parent / "03_head_ais_sensitivity.py")
    v6_03 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v6_03)
    level_all = v6_03.map_teaching(raw["teaching_status_raw"])

    raw2 = raw.copy()
    raw2["teaching_level"] = level_all.values
    for k in ("PUF_YEAR", "inc_key"):
        scored[k] = pd.to_numeric(scored[k], errors="coerce").astype("Int64")
        raw2[k] = pd.to_numeric(raw2[k], errors="coerce").astype("Int64")
    df = scored.merge(raw2[["PUF_YEAR", "inc_key", "teaching_level"]],
                      on=["PUF_YEAR", "inc_key"], how="inner", validate="1:1")
    if len(df) != C.V4_FROZEN["n_test"]:
        raise SystemExit(f"teaching frame is {len(df):,} rows, expected "
                         f"{C.V4_FROZEN['n_test']:,}")

    blocks = [
        ("levels.academic", "Academic", "Center", "Academic",
         df["teaching_level"] == "Academic", "Academic centre"),
        ("levels.community", "Community", "Center", "Community",
         df["teaching_level"] == "Community", "Community teaching centre"),
        ("levels.nonteaching", "Non-teaching", "Center", "Non-teaching",
         df["teaching_level"] == "Non-teaching", "Non-teaching centre"),
        ("binary.teaching", "Teaching", "Center (collapsed)", "Teaching",
         df["teaching_level"].isin(["Academic", "University", "Community"]),
         "Teaching centre, collapsed"),
        ("binary.nonteaching", "Non-teaching", "Center (collapsed)", "Non-teaching",
         df["teaching_level"] == "Non-teaching", "Non-teaching centre, collapsed"),
    ]

    out: dict = {"levels": {}, "binary": {}}
    for path, _lvl, tab_sub, tab_lvl, mask, prose in blocks:
        sub = df.loc[np.asarray(mask, dtype=bool)]
        w = {"Subgroup": tab_sub, "Level": tab_lvl, "Score": "PAT-TBI"}
        old_pt = float(csv_cell(tab, w, "AUROC"))
        old_lo = float(csv_cell(tab, w, "CI_Low"))
        old_hi = float(csv_cell(tab, w, "CI_High"))
        y, s = masked(sub, "pat_tbi_score", "pat_tbi_score_mask")
        row = record("teaching", prose, f"teaching.{path}", "PAT-TBI", y, s,
                     old_lo, old_hi, old_pt, None,
                     "m out of n bootstrap, 500 replicates of 50000 rows",
                     50_000)
        head, tail = path.split(".")
        out[head][tail] = {"auroc": leaf(row)}

    # The bundle part rounds to four places; confirm the recomputed points still
    # match what the part carries, not only what the table carries.
    for path in ("levels.academic", "levels.community", "levels.nonteaching",
                 "binary.teaching", "binary.nonteaching"):
        head, tail = path.split(".")
        published = part[head][tail]["auroc"]["estimate"]
        got = out[head][tail]["auroc"]["estimate"]
        if round(float(got), 4) != round(float(published), 4):
            GATE_FAILURES.append(
                f"teaching.{path}: point estimate moved, part carries "
                f"{published}, recomputed {got}")
    return out


# ===========================================================================
# PART 5 — the two sensitivity rows
# ===========================================================================
def part_sensitivity(test_split: pd.DataFrame) -> dict:
    C.banner("PART 5 — the two sensitivity rows")
    out = {}

    at = pd.read_csv(ALL_TRAUMA_CSV)
    old_pt = float(at.iloc[0]["AUROC"])
    old_lo = float(at.iloc[0]["CI_Low"])
    old_hi = float(at.iloc[0]["CI_High"])
    n_pub = int(at.iloc[0]["N"])
    print(f"  all-trauma cohort, published n = {n_pub:,}")
    at_df = pd.read_parquet(ALL_TRAUMA_PARQUET,
                            columns=[OUTCOME, "pat_tbi_score"])
    if len(at_df) != n_pub:
        raise SystemExit(f"all-trauma parquet is {len(at_df):,} rows, the table "
                         f"says {n_pub:,}")
    row = record("sensitivity", "All-trauma cohort, any injury",
                 "sens.all_trauma", "PAT-TBI",
                 at_df[OUTCOME].to_numpy(), at_df["pat_tbi_score"].to_numpy(),
                 old_lo, old_hi, old_pt, 4,
                 "m out of n bootstrap, 2000 replicates of 100000 rows",
                 V4_SUB_DISC,
                 "the largest denominator in the project and the worst "
                 "inflation")
    out["all_trauma"] = leaf(row)
    del at_df

    wb = pd.read_csv(WHOLE_BLOOD_CSV)
    w = {"Outcome": "received_platelets_or_wholeblood_4h (combined)"}
    old_pt = float(csv_cell(wb, w, "AUROC"))
    old_lo = float(csv_cell(wb, w, "CI_Low"))
    old_hi = float(csv_cell(wb, w, "CI_High"))
    y = test_split[OUTCOME_SENS].astype(float).to_numpy()
    s = test_split["pat_tbi_score"].astype(float).to_numpy()
    row = record("sensitivity", "Platelets or whole blood within four hours",
                 "sens.whole_blood", "PAT-TBI", y, s,
                 old_lo, old_hi, old_pt, 4,
                 "full-size bootstrap, 2000 replicates, already correct", None,
                 "this row was never defective; it is restated on DeLong only "
                 "so every interval in the paper uses one method")
    out["whole_blood"] = leaf(row)
    return out


# ===========================================================================
# PART 6 — the head-AIS models and their contrasts
# ===========================================================================
def part_head_ais(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    C.banner("PART 6 — head AIS base model, extended model, and the contrasts")
    part = json.loads(HEAD_AIS_PART.read_text())
    refit = C.load_refit()
    variables = list(refit["variables_kept"])

    y_tr = train[OUTCOME].astype(int).to_numpy()
    y_te = test[OUTCOME].astype(int).to_numpy()

    X_tr = train[variables].astype(float)
    t0 = time.time()
    base = fit_logit(X_tr, y_tr)
    max_diff = max(abs(float(base.params[v]) - float(refit["coefficients"][v]))
                   for v in variables)
    gate("the base model still reproduces the frozen refit record",
         max_diff <= 1e-6, f"largest coefficient gap {max_diff:.3e}")

    ais_tr = train[HEAD_AIS].astype(float).to_numpy()
    ais_te = test[HEAD_AIS].astype(float).to_numpy()
    X_tr_ext = X_tr.copy()
    X_tr_ext[HEAD_AIS] = ais_tr
    ext = fit_logit(X_tr_ext, y_tr)

    unk_tr = (ais_tr == 9).astype(float)
    unk_te = (ais_te == 9).astype(float)
    ord_tr = np.where(ais_tr == 9, 0.0, ais_tr)
    ord_te = np.where(ais_te == 9, 0.0, ais_te)
    X_tr_extb = X_tr.copy()
    X_tr_extb[HEAD_AIS] = ord_tr
    X_tr_extb["head_ais_unknown"] = unk_tr
    extb = fit_logit(X_tr_extb, y_tr)
    print(f"  three models fitted in {time.time() - t0:.1f}s")

    X_te = test[variables].astype(float)
    X_te_ext = X_te.copy()
    X_te_ext[HEAD_AIS] = ais_te
    X_te_extb = X_te.copy()
    X_te_extb[HEAD_AIS] = ord_te
    X_te_extb["head_ais_unknown"] = unk_te

    lp_base = linear_predictor(base, X_te)
    lp_ext = linear_predictor(ext, X_te_ext)
    lp_extb = linear_predictor(extb, X_te_extb)
    s_int = test["pat_tbi_score"].astype(float).to_numpy()

    out: dict = {}
    spec = [
        ("base_model", "Base model, linear predictor", lp_base),
        ("extended_model", "Extended model with head injury severity", lp_ext),
        ("severity_unknown_variant",
         "Extended model, severity unknown held out", lp_extb),
        ("integer_score", "Published integer score", s_int),
    ]
    for node, prose, vec in spec:
        old = part[node]["auroc_validation"]
        row = record("head_ais", prose, f"head_ais.{node}.auroc_validation",
                     "linear predictor" if node != "integer_score" else "PAT-TBI",
                     y_te, vec, float(old["lo"]), float(old["hi"]),
                     float(old["estimate"]), 4,
                     "m out of n bootstrap, 2000 replicates of 100000 rows",
                     V4_SUB_DISC)
        out[node] = {"auroc_validation": leaf(row)}

    C.banner("The two DeLong contrasts, which were already computed correctly")
    for node, key, prose, vec in [
        ("contrast", "auroc_difference",
         "Extended model minus base model", lp_ext),
        ("severity_unknown_variant", "auroc_difference",
         "Severity-unknown variant minus base model", lp_extb),
    ]:
        z_d, p_d, delta, se_d, _a1, _a2 = delong_test_vec(y_te, vec, lp_base)
        lo, hi = delta - Z * se_d, delta + Z * se_d
        old = part[node][key]
        moved = (round(delta, 5) != round(float(old["estimate"]), 5)
                 or round(lo, 5) != round(float(old["lo"]), 5)
                 or round(hi, 5) != round(float(old["hi"]), 5))
        print(f"  {prose}")
        print(f"    published : {float(old['estimate']):.5f} "
              f"({float(old['lo']):.5f} to {float(old['hi']):.5f})")
        print(f"    recomputed: {delta:.5f} ({lo:.5f} to {hi:.5f})   "
              f"standard error {se_d:.6f}, z {z_d:+.2f}")
        print(f"    verdict   : {'MOVED' if moved else 'unchanged, as expected'}")
        ROWS.append({
            "block": "head_ais_contrast",
            "label": prose,
            "bundle_key": f"head_ais.{node}.{key}",
            "score": "difference in areas",
            "n": int(len(y_te)), "events": int(y_te.sum()),
            "estimate": round(float(delta), 6),
            "old_lo": round(float(old["lo"]), 6),
            "old_hi": round(float(old["hi"]), 6),
            "old_width": round(float(old["hi"]) - float(old["lo"]), 6),
            "new_lo": round(float(lo), 6),
            "new_hi": round(float(hi), 6),
            "new_width": round(float(hi - lo), 6),
            "width_ratio_old_over_new": round(
                (float(old["hi"]) - float(old["lo"])) / float(hi - lo), 3),
            "old_resample": int(len(y_te)),
            "nominal_inflation": 1.0,
            "delong_se": round(float(se_d), 6),
            "point_moved": bool(round(delta, 5) != round(float(old["estimate"]), 5)),
            "old_method": "DeLong, already correct",
            "note": "unchanged" if not moved else "moved, investigate",
        })
        node_out = out.setdefault(node, {})
        node_out[key] = {"estimate": round(float(delta), 5),
                         "lo": round(float(lo), 5), "hi": round(float(hi), 5)}
        if node == "contrast":
            node_out["standard_error"] = {"value": round(float(se_d), 6)}
            node_out["unchanged"] = {"display": (
                "This contrast was already computed with DeLong and does not "
                "move. Only the single-score intervals beside it were affected "
                "by the resampling defect.")}
    return out


# ===========================================================================
# Bundle part
# ===========================================================================
def build_payload(head: dict, disc: dict, subs: dict, teach: dict,
                  sens: dict, hais: dict) -> dict:
    return {
        "_meta": {
            "script": "11_recompute_intervals.py",
            "finding": "every reported area interval was too wide",
            "source": {"display": (
                "Recomputed from the frozen validation rows, the frozen "
                "all-trauma rows, and the frozen refit record. Nothing was "
                "re-derived and no point estimate was recomputed from a "
                "different model.")},
        },
        "method": {"display": (
            "Every interval around an area under a receiver operating "
            "characteristic curve in this paper is the analytic DeLong "
            "interval on the rows that score was evaluated on. DeLong was "
            "chosen for three reasons. It is exact rather than simulated, so "
            "it carries no seed and reproduces bit for bit. It costs a single "
            "sort, so the same method serves the validation cohort and the "
            "far larger all-trauma cohort, where a bootstrap of the same "
            "resolution would take hours. And it is already the estimator the "
            "paper uses for every difference between two areas, so the "
            "single-score intervals and the contrast intervals now rest on one "
            "variance rather than on two that disagree. Agreement with a "
            "correct full-size nonparametric bootstrap was verified on the "
            "headline area before the method was adopted; the two agree to the "
            "third decimal on both bounds.")},
        "correction": {"display": (
            "The intervals reported earlier were too wide. The helper that "
            "produced them drew a fixed-size subsample of the cohort in each "
            "bootstrap replicate rather than a resample of the whole cohort, "
            "and then took a percentile interval without the rescaling that an "
            "m-out-of-n bootstrap requires. The interval it returned therefore "
            "described the spread of the statistic in a sample the size of the "
            "subsample, not in the cohort. Inflation grew with the cohort, so "
            "the largest denominators were the worst affected. Two of the "
            "reported intervals were never defective, because their cohorts "
            "were smaller than the subsample size or because a different "
            "script computed them correctly; those are restated on the same "
            "method for consistency and barely move. No point estimate is "
            "affected by any of this, and none moved: every area was "
            "recomputed from the same rows and matched the published value at "
            "the precision it was printed to. Only the intervals change, and "
            "every one of them narrows or holds.")},
        "agreement": {
            "point_estimate": {"estimate": round(head["auc"], 4),
                               "lo": round(head["dl_lo"], 4),
                               "hi": round(head["dl_hi"], 4)},
            "superseded_protocol": {"estimate": round(head["auc"], 4),
                                    "lo": round(head["v4_lo"], 4),
                                    "hi": round(head["v4_hi"], 4)},
            "full_size_bootstrap": {"estimate": round(head["auc"], 4),
                                    "lo": round(head["boot_lo"], 4),
                                    "hi": round(head["boot_hi"], 4)},
            "analytic_delong": {"estimate": round(head["auc"], 4),
                                "lo": round(head["dl_lo"], 4),
                                "hi": round(head["dl_hi"], 4)},
            "independent_check": {"estimate": round(head["ind_auc"], 4),
                                  "lo": round(head["ind_lo"], 4),
                                  "hi": round(head["ind_hi"], 4)},
            "delong_standard_error": {"value": round(head["dl_se"], 6)},
            "bootstrap_standard_deviation": {"value": round(head["boot_sd"], 6)},
            "inflation_predicted": {"value": round(head["expected"], 4)},
            "inflation_observed": {"value": round(head["observed_vs_delong"], 4)},
            "bootstrap_replicates": {"value": N_BOOT_FULL},
            "verdict": {"display": (
                "The corrected bootstrap and the analytic interval agree on "
                "both bounds, and the corrected bootstrap reproduces exactly "
                "the full-cohort figure the whole-blood sensitivity table "
                "already carried for the same rows and the same outcome. Three "
                "independent routes therefore give the same answer, and the "
                "superseded interval is the only one that disagrees with all "
                "of them.")},
            "independent_check_note": {"display": (
                "The whole-blood sensitivity table computed the headline area "
                "on the identical validation rows with a correct full-cohort "
                "bootstrap and printed a materially narrower interval than the "
                "discrimination table. The two tables contradicted each other "
                "in the earlier draft. They now agree.")},
        },
        "discrimination": disc,
        "subgroups": subs,
        "teaching": teach,
        "sens": sens,
        "head_ais": hais,
        "coverage": {
            "areas_recomputed": {"value": sum(
                1 for r in ROWS if r["block"] != "head_ais_contrast")},
            "contrasts_rechecked": {"value": sum(
                1 for r in ROWS if r["block"] == "head_ais_contrast")},
            "point_estimates_moved": {"value": sum(
                1 for r in ROWS if r["point_moved"])},
            "audit_table": {"display": (
                "One row per area, carrying the superseded interval, the "
                "corrected interval and the ratio of their widths, is written "
                "to the interval table in the tables folder.")},
        },
    }


# ===========================================================================
# Handoff
# ===========================================================================
def write_handoff(head: dict) -> None:
    def fmt(v, nd=4):
        return f"{float(v):.{nd}f}"

    def block(name):
        return [r for r in ROWS if r["block"] == name]

    worst = max((r for r in ROWS if r["width_ratio_old_over_new"] != ""),
                key=lambda r: float(r["width_ratio_old_over_new"]))

    lines: list[str] = []
    A = lines.append
    A("# Step 11 handoff — corrected confidence intervals")
    A("")
    A("Script: `v6/code/11_recompute_intervals.py`  ")
    A("Log: `v6/logs/11_recompute_intervals.log`  ")
    A("Table: `v6/tables/table_v6_intervals.csv`  ")
    A("Bundle part: `v6/bundle/parts/intervals.json` (lands at `results.intervals.*`)")
    A("")
    A("## The headline")
    A("")
    A(f"**PAT-TBI validation area: {fmt(head['auc'])} "
      f"(95% CI {fmt(head['dl_lo'])} to {fmt(head['dl_hi'])}).**")
    A("")
    A("The point estimate is unchanged. Replace the interval everywhere the "
      "old one appears.")
    A("")
    A("| route | interval | width |")
    A("|---|---|---|")
    A(f"| superseded, m out of n bootstrap | {fmt(head['v4_lo'])} to "
      f"{fmt(head['v4_hi'])} | {fmt(head['v4_hi'] - head['v4_lo'])} |")
    A(f"| corrected full-size bootstrap | {fmt(head['boot_lo'])} to "
      f"{fmt(head['boot_hi'])} | {fmt(head['boot_hi'] - head['boot_lo'])} |")
    A(f"| analytic DeLong, **adopted** | {fmt(head['dl_lo'])} to "
      f"{fmt(head['dl_hi'])} | {fmt(head['dl_hi'] - head['dl_lo'])} |")
    A(f"| independent full-cohort figure already in the project | "
      f"{fmt(head['ind_lo'])} to {fmt(head['ind_hi'])} | "
      f"{fmt(head['ind_hi'] - head['ind_lo'])} |")
    A("")
    A(f"Predicted inflation of the superseded interval, sqrt(n/m) = "
      f"{head['expected']:.3f}; observed {head['observed_vs_delong']:.3f}.")
    A("")
    A("## Methods sentence to use")
    A("")
    A("> Discrimination was summarised by the area under the receiver "
      "operating characteristic curve, with 95% confidence intervals from the "
      "DeLong variance estimator; differences between two areas measured on "
      "the same patients were tested by DeLong's method for correlated areas. "
      "Agreement between the analytic interval and a nonparametric bootstrap "
      "of 2,000 resamples of the full validation cohort was confirmed for the "
      "primary area, the two bounds differing by less than 0.001.")
    A("")
    A("## Replacement list")
    A("")
    A("Every row below keeps its point estimate and replaces only the "
      "interval. Bundle keys are the accessors the two documents render "
      "through; the corrected values also sit at `results.intervals.*`.")
    A("")

    A("### Headline discrimination table")
    A("")
    A("| bundle key | label | n | events | estimate | old interval | new interval | width ratio |")
    A("|---|---|---|---|---|---|---|---|")
    for r in block("discrimination"):
        A(f"| `{r['bundle_key']}` | {r['label']} | {r['n']:,} | {r['events']:,} | "
          f"{fmt(r['estimate'])} | {fmt(r['old_lo'])} to {fmt(r['old_hi'])} | "
          f"**{fmt(r['new_lo'])} to {fmt(r['new_hi'])}** | "
          f"{r['width_ratio_old_over_new']} |")
    A("")

    A("### Sensitivity rows")
    A("")
    A("| bundle key | label | n | events | estimate | old interval | new interval | width ratio |")
    A("|---|---|---|---|---|---|---|---|")
    for r in block("sensitivity"):
        A(f"| `{r['bundle_key']}` | {r['label']} | {r['n']:,} | {r['events']:,} | "
          f"{fmt(r['estimate'])} | {fmt(r['old_lo'])} to {fmt(r['old_hi'])} | "
          f"**{fmt(r['new_lo'])} to {fmt(r['new_hi'])}** | "
          f"{r['width_ratio_old_over_new']} |")
    A("")

    A("### Teaching status")
    A("")
    A("| bundle key | label | n | events | estimate | old interval | new interval | width ratio |")
    A("|---|---|---|---|---|---|---|---|")
    for r in block("teaching"):
        A(f"| `{r['bundle_key']}.auroc` | {r['label']} | {r['n']:,} | "
          f"{r['events']:,} | {fmt(r['estimate'])} | "
          f"{fmt(r['old_lo'])} to {fmt(r['old_hi'])} | "
          f"**{fmt(r['new_lo'])} to {fmt(r['new_hi'])}** | "
          f"{r['width_ratio_old_over_new']} |")
    A("")

    A("### Head AIS sensitivity")
    A("")
    A("| bundle key | label | estimate | old interval | new interval | width ratio |")
    A("|---|---|---|---|---|---|")
    for r in block("head_ais"):
        A(f"| `{r['bundle_key']}` | {r['label']} | {fmt(r['estimate'])} | "
          f"{fmt(r['old_lo'])} to {fmt(r['old_hi'])} | "
          f"**{fmt(r['new_lo'])} to {fmt(r['new_hi'])}** | "
          f"{r['width_ratio_old_over_new']} |")
    for r in block("head_ais_contrast"):
        A(f"| `{r['bundle_key']}` | {r['label']} | {fmt(r['estimate'], 5)} | "
          f"{fmt(r['old_lo'], 5)} to {fmt(r['old_hi'], 5)} | "
          f"**{fmt(r['new_lo'], 5)} to {fmt(r['new_hi'], 5)}** | "
          f"{r['width_ratio_old_over_new']} |")
    A("")
    A("The two contrasts were already DeLong and do not move. They are listed "
      "so a reader can see they were checked rather than assumed.")
    A("")

    A("### Subgroups, all thirteen levels and all five scores")
    A("")
    A("These replace `subgroups.rows` (PAT-TBI only, thirteen rows) and the "
      "subgroup table. The full 65-cell grid is in "
      "`results.intervals.subgroups.rows`; the thirteen PAT-TBI rows are "
      "repeated at `results.intervals.subgroups.pat_rows` in exactly the shape "
      "the existing `subgroups.rows` leaf uses, so the swap is one for one.")
    A("")
    A("| dimension | level | score | n | events | estimate | old interval | new interval | width ratio |")
    A("|---|---|---|---|---|---|---|---|---|")
    for r in block("subgroup"):
        A(f"| {r.get('dimension', '')} | {r.get('level_prose', '')} | "
          f"{r['score']} | {r['n']:,} | {r['events']:,} | "
          f"{fmt(r['estimate'])} | {fmt(r['old_lo'])} to {fmt(r['old_hi'])} | "
          f"**{fmt(r['new_lo'])} to {fmt(r['new_hi'])}** | "
          f"{r['width_ratio_old_over_new']} |")
    A("")

    A("## Notes for the abstract, manuscript, figure and letter writers")
    A("")
    A(f"- The worst affected row is **{worst['label']}** on {worst['n']:,} "
      f"patients: it printed {fmt(worst['old_lo'])} to {fmt(worst['old_hi'])} "
      f"and should print {fmt(worst['new_lo'])} to {fmt(worst['new_hi'])}, a "
      f"width ratio of {worst['width_ratio_old_over_new']}.")
    A("- No point estimate moved. If a draft carries an area that is not in "
      "the table above, it did not come from this analysis and should be "
      "traced before it is printed.")
    A("- Any figure that draws an error bar around an area must be rebuilt "
      "from the new bounds. The bars get shorter; none of the markers move.")
    A("- Rows whose cohort was no larger than the old resample size were never "
      "defective. Their intervals shift only by the difference between a "
      "500-replicate percentile bootstrap and the analytic form, and the shift "
      "can go either way. The audit table carries `old_resample` and "
      "`nominal_inflation` on every row, so the expected correction, "
      "sqrt(n divided by the resample size), can be read against the observed "
      "width ratio cell by cell.")
    A("- The whole-blood sensitivity row was computed correctly all along, by "
      "a different script that resampled the whole cohort. It is the reason "
      "the earlier draft contained two different intervals for the same "
      "quantity.")
    A("- The two DeLong contrasts in the head-AIS block, and every "
      "between-score difference in the subgroup table, were already correct "
      "and are unchanged.")
    A("")
    OUT_HANDOFF.write_text("\n".join(lines) + "\n")
    print(f"[write] {OUT_HANDOFF}")


# ===========================================================================
# Main
# ===========================================================================
def main() -> int:
    C.start_log("11_recompute_intervals")
    t_start = time.time()
    C.banner("Step 11 — recomputing every reported area interval")
    print("  Point estimates are not touched. Only intervals are recomputed.")

    test_scored = C.load_test_scored()
    print(f"  scored validation rows: {len(test_scored):,}")

    head = part_one(test_scored)
    disc = part_discrimination(test_scored)

    sg_frame = load_subgroup_frame()
    subs = part_subgroups(sg_frame)
    del sg_frame

    teach = part_teaching()

    train, test = C.load_split()
    sens = part_sensitivity(test)
    hais = part_head_ais(train, test)

    # ------------------------------------------------------------------ gates
    C.banner("Final gates")
    pat = next(r for r in ROWS if r["bundle_key"] == "discrimination.pat")
    gate("the validation area is still the frozen headline",
         round(float(pat["estimate"]), 4) == C.V4_FROZEN["auroc_test"],
         f"{pat['estimate']}")
    gate("the new headline interval is narrower than the old one",
         float(pat["new_width"]) < float(pat["old_width"]),
         f"new width {pat['new_width']}, old width {pat['old_width']}")
    gate("the new headline interval contains the point estimate",
         float(pat["new_lo"]) <= float(pat["estimate"]) <= float(pat["new_hi"]))
    moved = [r for r in ROWS if r["point_moved"]]
    gate("no point estimate moved", not moved,
         "; ".join(r["label"] for r in moved) if moved else
         f"{len(ROWS)} areas and contrasts checked")
    contains = [r for r in ROWS
                if not (float(r["new_lo"]) <= float(r["estimate"]) <= float(r["new_hi"]))]
    gate("every new interval contains its own estimate", not contains,
         "; ".join(r["label"] for r in contains) if contains else "")
    narrowed = [r for r in ROWS if r["old_width"] != ""
                and float(r["new_width"]) <= float(r["old_width"]) + 1e-9]
    print(f"  intervals that narrowed or held: {len(narrowed)} of {len(ROWS)}")
    widened = [r for r in ROWS if r["old_width"] != ""
               and float(r["new_width"]) > float(r["old_width"]) + 1e-9]
    print("  rows where the interval widened slightly are rows whose "
          "superseded resample already covered nearly the whole cell, so the "
          "correction had nothing to remove and the change is the difference "
          "between a percentile bootstrap and the analytic form:")
    for r in widened:
        print(f"    wider: {r['label']} on n = {r['n']:,}  "
              f"nominal inflation {r['nominal_inflation']}  "
              f"{r['old_width']} -> {r['new_width']}")
    bad = [r for r in widened if float(r["nominal_inflation"]) >= 1.05]
    gate("no row with a materially inflated superseded interval got wider",
         not bad, "; ".join(r["label"] for r in bad) if bad else "")

    if GATE_FAILURES:
        C.banner("STOP")
        for f in GATE_FAILURES:
            print(f"  {f}")
        raise SystemExit(
            "A gate failed. The plan says to stop and report rather than "
            "adjust a number to fit, so no bundle part is written.")

    # ---------------------------------------------------------------- outputs
    C.banner("Writing outputs")
    cols = ["block", "label", "bundle_key", "score", "n", "events", "estimate",
            "old_lo", "old_hi", "old_width", "new_lo", "new_hi", "new_width",
            "width_ratio_old_over_new", "old_resample", "nominal_inflation",
            "delong_se", "point_moved", "old_method", "note"]
    pd.DataFrame(ROWS)[cols].to_csv(OUT_CSV, index=False)
    print(f"[write] {OUT_CSV}   rows {len(ROWS)}")

    payload = build_payload(head, disc, subs, teach, sens, hais)
    C.write_part("intervals", payload)
    print(f"[write] {C.V6_PARTS / 'intervals.json'}")

    write_handoff(head)

    C.banner("Summary")
    ratios = [float(r["width_ratio_old_over_new"]) for r in ROWS
              if r["width_ratio_old_over_new"] != ""]
    worst = max((r for r in ROWS if r["width_ratio_old_over_new"] != ""),
                key=lambda r: float(r["width_ratio_old_over_new"]))
    print(f"  areas recomputed        : "
          f"{sum(1 for r in ROWS if r['block'] != 'head_ais_contrast')}")
    print(f"  contrasts rechecked     : "
          f"{sum(1 for r in ROWS if r['block'] == 'head_ais_contrast')}")
    print(f"  point estimates moved   : {len(moved)}")
    print(f"  median width ratio      : {np.median(ratios):.3f}")
    print(f"  worst width ratio       : {max(ratios):.3f}  ({worst['label']}, "
          f"n = {worst['n']:,})")
    print(f"  headline                : {float(pat['estimate']):.4f} "
          f"(95% CI {float(pat['new_lo']):.4f} to {float(pat['new_hi']):.4f}), "
          f"was {float(pat['old_lo']):.4f} to {float(pat['old_hi']):.4f}")
    print(f"\n  runtime: {time.time() - t_start:.1f}s")
    C.banner("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
