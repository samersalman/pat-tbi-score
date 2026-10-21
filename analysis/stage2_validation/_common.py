"""
PAT-TBI v6 — shared loader for every Phase 0 repair script.
==========================================================

Single source for: the cohort matrix, the derivation/validation split, the
integer PAT-TBI score, the comparator scores, and the variable label table.

Every v6 repair script imports from here so that the derivation and validation
cohorts, the score, and the labels are identical across scripts. Nothing here
refits the model: the retained predictors, coefficients and integer points are
read from the frozen v4 artefacts.

Run with the framework interpreter, not bare `python3`:
    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3

Self-check (reproduces the v4 headline numbers):
    $PY _common.py --self-test
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
import os as _os, sys as _sys  # path handling: make analysis/config.py importable
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import config  # noqa: E402  (analysis/config.py: PAT_TBI_WORKDIR, PAT_TBI_NTDB_DUCKDB)
V4 = config.V4_DIR
V6 = config.V6_DIR
V4_CODE = config.V4_CODE_DIR

V4_DERIVED = V4 / "derived"
V4_TABLES = V4 / "tables"
V6_PARTS = V6 / "bundle" / "parts"
V6_LOGS = V6 / "logs"

CANDIDATES = V4_DERIVED / "candidates.parquet"
COHORT_RAW = V4_DERIVED / "cohort_raw.parquet"
TEST_SCORED = V4_DERIVED / "cohort_test_scored.parquet"
REFIT_JSON = V4_DERIVED / "pat_refit.json"
TRAIN_KEYS = V4_DERIVED / "train_inc_keys.csv"
TEST_KEYS = V4_DERIVED / "test_inc_keys.csv"
WEIGHTS_CSV = V4_TABLES / "table_pat_weights.csv"

SEED = 42
OUTCOME = "received_platelets_4h"
OUTCOME_SENS = "received_platelets_or_wholeblood_4h"

# ---------------------------------------------------------------------------
# Frozen v4 headline values. Any v6 script that moves one of these has broken
# the score, and the plan says the cause is found before proceeding.
# ---------------------------------------------------------------------------
V4_FROZEN = {
    "n_cohort": 623587,
    "n_train": 415724,
    "n_test": 207863,
    "n_train_events": 8982,
    "n_test_events": 4491,
    "n_predictors": 22,
    "auroc_test": 0.8738,
    # Corrected 2026-09-08. The v4 interval of 0.8645 to 0.8827 came from an
    # m-out-of-n bootstrap that resampled 100,000 of 207,863 rows and took the
    # percentile interval without the sqrt(n/m) rescaling, so it was about 44%
    # too wide. 11_recompute_intervals.py re-derived every interval with DeLong,
    # cross-checked against a correct full-size bootstrap. The point estimate
    # did not move. The withdrawn bounds are kept below for the audit trail and
    # must never be reported.
    "auroc_test_lo": 0.8675,
    "auroc_test_hi": 0.8801,
    "auroc_test_lo_withdrawn": 0.8645,
    "auroc_test_hi_withdrawn": 0.8827,
    "score_min_train": -8,
    "score_max_train": 35,
    "score_min_test": -8,
    "score_max_test": 37,
    "youden_cutpoint": 4,
}

# ---------------------------------------------------------------------------
# Corrected variable labels.
#
# Each string below was read off `v4/code/02_engineer_features.py` line by line.
# The v4 `14_table1.py` LABEL_MAP disagreed with the code on seven of these
# (sbp_hypertensive, rr_low, rr_high, hypoxic, si_high, temp_hypothermic and
# hr_tachycardic), five of which are in the final score. The plan's finding 2
# counted six and four, missing hr_tachycardic, whose number matches but whose
# operator does not: the label said `> 120` where the code applies `>= 120`, so
# a patient at exactly 120 is counted by one and not the other. These are the
# definitions that ran.
# ---------------------------------------------------------------------------
LABELS: dict[str, str] = {
    # Demographics
    "AGE_NUMBER": "Age, years",
    "age_45_64": "Age 45 to 64 years",
    "age_65_74": "Age 65 to 74 years",
    "age_75plus": "Age 75 years or older",
    "female": "Female sex",
    # Mechanism (ICD-10-CM external cause code prefix)
    "mech_mvc": "Motor vehicle traffic crash (V40 to V89)",
    "mech_fall": "Fall (W00 to W19)",
    "mech_assault": "Assault (X85 to X99, Y00 to Y09)",
    "risk_penetrating": "Penetrating assault (X93 to X99)",
    "is_transfer": "Transferred from another facility",
    # ED vitals — cutpoints exactly as coded in 02_engineer_features.py
    "sbp_hypotensive": "Systolic blood pressure below 90 mmHg",
    "sbp_hypertensive": "Systolic blood pressure 140 mmHg or above",
    "hr_bradycardic": "Heart rate below 60 beats/min",
    "hr_tachycardic": "Heart rate 120 beats/min or above",
    "rr_low": "Respiratory rate below 12 breaths/min",
    "rr_high": "Respiratory rate 30 breaths/min or above",
    "hypoxic": "Oxygen saturation 92% or below",
    "shock_index": "Shock index (heart rate divided by systolic blood pressure)",
    "si_elevated": "Shock index 0.9 or above",
    "si_high": "Shock index 1.3 or above",
    "temp_hypothermic": "Temperature 20 to below 35 degrees Celsius",
    # GCS and pupils
    "gcs_severe": "Glasgow Coma Scale total 8 or below",
    "gcs_moderate": "Glasgow Coma Scale total 9 to 12",
    "gcs_motor_low": "Glasgow Coma Scale motor 3 or below",
    "gcs_motor_high": "Glasgow Coma Scale motor 6",
    "tbi_pupil_none": "Neither pupil reactive",
    # Comorbidities
    "cc_anticoag": "Anticoagulant therapy",
    "cc_bleeding": "Bleeding disorder",
    # Non-X columns used for stratification and comparators
    "iss_cont": "Injury Severity Score",
    "max_ais_head": "Maximum head Abbreviated Injury Scale",
    "max_ais_other": "Maximum extracranial Abbreviated Injury Scale",
    "major_extracranial": "Major extracranial injury",
    "SEX": "Sex",
    "RACE": "Race",
    "ETHNICITY": "Ethnicity",
    "GCS2": "Glasgow Coma Scale total",
    "MOTOR2": "Glasgow Coma Scale motor",
    "TQIP_TBI_PUPIL_RESPONSE": "Pupillary response",
    # Raw emergency department fields, for the missingness table
    "SBP2": "Systolic blood pressure, mmHg",
    "P2": "Heart rate, beats/min",
    "RR2": "Respiratory rate, breaths/min",
    "OX2": "Oxygen saturation, %",
    "temperature_raw": "Temperature, degrees Celsius",
    "cause_e_code": "External injury mechanism code",
    "HOSPITAL_TRANSFER": "Transfer from another facility",
    "teaching_status_raw": "Facility teaching status",
    "PLATELETS4HOURS": "Platelet units within four hours",
    "WHOLEBLOOD4HOURS": "Whole blood units within four hours",
    "ISS": "Injury Severity Score",
}

# The 15 v3 leftovers that print as empty rows in the v4 Table 1. They were
# dropped from the v4 candidate set in 02_engineer_features.py and are not
# columns of candidates.parquet, so no Table 1 row can be built for them.
V3_LEFTOVER_ROWS: tuple[str, ...] = (
    "ins_uninsured",
    "cc_alcohol", "cc_cancer", "cc_chf", "cc_cirrhosis", "cc_ckd",
    "cc_copd", "cc_cva", "cc_dementia", "cc_dm", "cc_htn", "cc_smoker",
    "teaching_status_binary", "verification_level_1", "highest_activation_highest",
)

# Predictor pairs where one indicator contains, or nearly contains, the other,
# so the points do not add the way a reader expects. Verified on the data, not
# only in the code; see 04_missingness.py and the nesting block of the bundle.
#
# Only the first two are strictly nested. The Glasgow Coma Scale pair is NOT:
# 635 of 27,540 whole-cohort patients with motor 3 or below have a total above
# 8 (419 of 18,362 in derivation). Describing that pair as nested in the
# manuscript would be wrong. It is near-complete overlap, and it is the only one
# of the three that the correlation screen actually catches.
NESTED_PAIRS: tuple[tuple[str, str], ...] = (
    ("risk_penetrating", "mech_assault"),   # X93-X99 sits inside X85-X99, strict
    ("si_high", "si_elevated"),             # SI >= 1.3 implies SI >= 0.9, strict
    ("gcs_motor_low", "gcs_severe"),        # near-complete overlap, NOT strict
)
STRICTLY_NESTED_PAIRS: tuple[tuple[str, str], ...] = NESTED_PAIRS[:2]


def label(var: str) -> str:
    """Clinical prose for an identifier. Raises rather than leaking a slug."""
    if var not in LABELS:
        raise KeyError(f"no label for {var!r}; add it to _common.LABELS")
    return LABELS[var]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_points() -> dict[str, int]:
    """The 22 retained predictors and their integer points, from the v4 refit."""
    refit = json.loads(REFIT_JSON.read_text())
    pts = {k: int(v) for k, v in refit["points"].items()}
    if len(pts) != V4_FROZEN["n_predictors"]:
        raise RuntimeError(f"expected 22 predictors, found {len(pts)}")
    return pts


def load_refit() -> dict:
    """The full v4 refit record: intercept, coefficients, points, diagnostics."""
    return json.loads(REFIT_JSON.read_text())


def load_weights_table() -> pd.DataFrame:
    """Coefficient, SE, OR, interval and points for the 22 predictors."""
    return pd.read_csv(WEIGHTS_CSV)


def _key_series(df: pd.DataFrame) -> pd.Series:
    """A single hashable key per encounter: PUF_YEAR and inc_key."""
    yr = pd.to_numeric(df["PUF_YEAR"], errors="coerce").astype("Int64")
    ik = pd.to_numeric(df["inc_key"], errors="coerce").astype("Int64")
    return yr.astype(str) + "_" + ik.astype(str)


def add_score(df: pd.DataFrame, points: dict[str, int] | None = None) -> pd.DataFrame:
    """Attach `pat_tbi_score` exactly as 06_score_test_set.py computes it.

    Every predictor is a pre-built binary in candidates.parquet, so the score is
    a plain weighted sum and every row is scoreable.
    """
    points = points or load_points()
    score = np.zeros(len(df), dtype=np.float64)
    for var, pts in points.items():
        if var not in df.columns:
            raise RuntimeError(f"predictor {var!r} missing from the frame")
        score += float(pts) * df[var].astype(np.float64).values
    out = df.copy()
    out["pat_tbi_score"] = score.astype(np.int64)
    return out


def load_split(with_score: bool = True, columns: list[str] | None = None):
    """Return (derivation, validation) frames carrying the outcome and the score.

    The split is the frozen v4 split, read from train_inc_keys.csv and
    test_inc_keys.csv, not re-drawn. Nothing here depends on SEED; the constant
    is kept only so a script that samples uses the house value.
    """
    cand = pd.read_parquet(CANDIDATES, columns=columns) if columns else pd.read_parquet(CANDIDATES)
    cand["_key"] = _key_series(cand)

    tr = pd.read_csv(TRAIN_KEYS)
    te = pd.read_csv(TEST_KEYS)
    tr_keys = set(_key_series(tr))
    te_keys = set(_key_series(te))

    train = cand[cand["_key"].isin(tr_keys)].copy()
    test = cand[cand["_key"].isin(te_keys)].copy()

    if len(train) != V4_FROZEN["n_train"]:
        raise RuntimeError(f"derivation n = {len(train):,}, expected {V4_FROZEN['n_train']:,}")
    if len(test) != V4_FROZEN["n_test"]:
        raise RuntimeError(f"validation n = {len(test):,}, expected {V4_FROZEN['n_test']:,}")

    if with_score:
        train = add_score(train)
        test = add_score(test)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def load_test_scored() -> pd.DataFrame:
    """The v4 scored validation frame, carrying every comparator and its mask."""
    return pd.read_parquet(TEST_SCORED)


def load_cohort_raw(columns: list[str] | None = None) -> pd.DataFrame:
    """The pre-engineering cohort, for outcome-field and predictor missingness."""
    return pd.read_parquet(COHORT_RAW, columns=columns) if columns else pd.read_parquet(COHORT_RAW)


# ---------------------------------------------------------------------------
# Small shared statistics
# ---------------------------------------------------------------------------
def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion, as a pair of fractions.

    The bounds are clamped to contain the observed proportion. Without that, a
    proportion of exactly 1 returns an upper bound of 0.9999999999999998 from
    the floating-point clamp, giving an interval that does not contain its own
    estimate, which a ledger check rejects.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z / d) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo = min(max(0.0, centre - half), p)
    hi = max(min(1.0, centre + half), p)
    return (lo, hi)


def write_part(name: str, payload: dict) -> Path:
    """Write one bundle part. Parts are merged by 06_build_bundle.py."""
    V6_PARTS.mkdir(parents=True, exist_ok=True)
    path = V6_PARTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, default=_json_default))
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o)} is not JSON serialisable")


class Tee:
    """Mirror stdout into a log file, so every script leaves a transcript."""

    def __init__(self, path: Path):
        V6_LOGS.mkdir(parents=True, exist_ok=True)
        self.fh = open(path, "w")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.fh.write(s)

    def flush(self):
        self.stdout.flush()
        self.fh.flush()


def start_log(script_name: str) -> Tee:
    tee = Tee(V6_LOGS / f"{script_name}.log")
    sys.stdout = tee
    return tee


def banner(msg: str) -> None:
    print("\n" + "=" * 78)
    print(msg)
    print("=" * 78, flush=True)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _self_test() -> int:
    from sklearn.metrics import roc_auc_score

    ok = True

    def check(name, got, want, tol=None):
        nonlocal ok
        if tol is None:
            good = got == want
        else:
            good = abs(float(got) - float(want)) <= tol
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] {name}: got {got}, expected {want}")

    banner("_common self-test — reproducing the frozen v4 numbers")
    points = load_points()
    check("predictor count", len(points), V4_FROZEN["n_predictors"])

    train, test = load_split()
    check("derivation n", len(train), V4_FROZEN["n_train"])
    check("validation n", len(test), V4_FROZEN["n_test"])
    check("derivation events", int(train[OUTCOME].sum()), V4_FROZEN["n_train_events"])
    check("validation events", int(test[OUTCOME].sum()), V4_FROZEN["n_test_events"])
    check("derivation score min", int(train["pat_tbi_score"].min()), V4_FROZEN["score_min_train"])
    check("derivation score max", int(train["pat_tbi_score"].max()), V4_FROZEN["score_max_train"])
    check("validation score min", int(test["pat_tbi_score"].min()), V4_FROZEN["score_min_test"])
    check("validation score max", int(test["pat_tbi_score"].max()), V4_FROZEN["score_max_test"])

    auroc = roc_auc_score(test[OUTCOME].values, test["pat_tbi_score"].values)
    check("validation AUROC", round(auroc, 4), V4_FROZEN["auroc_test"], tol=0.0001)

    scored = load_test_scored()
    merged = test[["_key", "pat_tbi_score"]].merge(
        scored.assign(_key=_key_series(scored))[["_key", "pat_tbi_score"]],
        on="_key", suffixes=("_v6", "_v4"))
    n_diff = int((merged["pat_tbi_score_v6"] != merged["pat_tbi_score_v4"]).sum())
    check("score rows disagreeing with v4", n_diff, 0)

    banner("nesting, verified on the derivation and validation rows together")
    both = pd.concat([train, test], ignore_index=True)
    for inner, outer in NESTED_PAIRS:
        n_inner = int(both[inner].sum())
        n_both = int(((both[inner] == 1) & (both[outer] == 1)).sum())
        print(f"  {inner} = 1 in {n_inner:,}; also {outer} = 1 in {n_both:,} "
              f"({100.0 * n_both / n_inner:.2f}%)")

    print("\n" + ("SELF-TEST PASSED" if ok else "SELF-TEST FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    print(__doc__)
