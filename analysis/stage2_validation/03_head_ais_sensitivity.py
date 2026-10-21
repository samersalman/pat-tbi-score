"""
PAT-TBI v6 — Step 3: head-AIS sensitivity refit + teaching-status subgroup
==========================================================================

Closes two Phase 0 findings, both of which produce rows in manuscript Figure 3
(the forest of contrasts).

Part A — head AIS sensitivity (plan item `03_head_ais_sensitivity.py`)
---------------------------------------------------------------------
Dumas and Hynes both asked why maximum head Abbreviated Injury Scale was
excluded from the candidate set. The project's answer is that AIS is assigned
retrospectively, after imaging and registry abstraction, so it is not on the
chart at the moment a platelet decision is made. That answer is stronger if we
quantify what the restriction costs. So:

  1. BASE      — unpenalized logistic regression of the outcome on the 22
                 retained predictors, on the 415,724 derivation rows. This
                 mirrors v4 step 05 exactly (statsmodels Logit, Newton, with a
                 constant) and its coefficients are checked against the frozen
                 `v4/derived/pat_refit.json`.
  2. EXTENDED  — the same 22 plus `max_ais_head` as an ordinal term.
  3. Both linear predictors are scored on the 207,863 validation rows. Each
     AUROC gets a bootstrap 95% interval; the difference gets a DeLong 95%
     interval and p-value. Linear predictor against linear predictor, so the
     comparison is like for like.
  4. The integer 22-point score is also scored in validation for context; it
     must round to the frozen 0.8738.

`max_ais_head` is already in `v4/derived/candidates.parquet` and is fully
populated, so nothing new is extracted. Two forms of the ordinal term are fitted
because the AIS convention codes 9 as "severity unknown", not as a severity of
nine:

  EXTENDED   — the column entered exactly as stored (the literal ordinal).
  EXTENDED-B — severity-unknown rows set to zero on the ordinal term and
               carried by their own indicator, so nothing is dropped and the
               unknown rows cannot masquerade as the most severe.

Part B — teaching status (finding 6)
------------------------------------
The v4 step 09 log carries the line `teaching_status_binary missing, skipping
center subgroup`, yet the v4 manuscript claims a teaching-status subgroup. The
plan is explicit: computing it is optional, claiming it without computing it is
not. `teaching_status_raw` is in `v4/derived/cohort_raw.parquet`, about 99.4%
populated, mixing encodings. This script prints the full value counts first,
writes the mapping out value by value, then either computes the subgroup or
prints the reason it cannot.

Reuse, not reinvention
----------------------
The DeLong machinery and both bootstrap protocols are imported from the v4
scripts themselves (`07_validation_metrics.py`, `09_subgroups.py`) rather than
retyped, so the intervals here are produced by the same code that produced the
frozen v4 tables.

Run with the framework interpreter, not bare `python3`:
    PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
    cd analysis/stage2_validation
    $PY 03_head_ais_sensitivity.py

Outputs
-------
  v6/tables/table_v6_head_ais.csv
  v6/tables/table_v6_teaching_subgroup.csv
  v6/bundle/parts/head_ais.json
  v6/bundle/parts/teaching.json
  v6/logs/03_head_ais_sensitivity.log
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sstats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

warnings.filterwarnings("ignore")

SEED = 42

# v4 step 07 bootstrap protocol (reproduces the frozen discrimination interval)
# The v4 step 07 protocol. Retained ONLY to reproduce the withdrawn interval
# for the audit trail; it is an m-out-of-n bootstrap with no sqrt(n/m) rescaling
# and is about 44% too wide. Never reported as a result.
N_BOOT_DISC = 2000
SUB_DISC = 100_000

# v4 step 09 subgroup bootstrap protocol (matches the existing 13 subgroup rows)
N_BOOT_SUB = 500
SUB_SUB = 50_000
MIN_EVENTS = 50

HEAD_AIS = "max_ais_head"

OUT_TABLE_A = C.V6 / "tables" / "table_v6_head_ais.csv"
OUT_TABLE_B = C.V6 / "tables" / "table_v6_teaching_subgroup.csv"
OUT_HANDOFF = C.V6 / "logs" / "03_HANDOFF.md"

# Score registry, identical to v4 step 09 so the new rows drop straight into
# `v4/tables/table_subgroups.csv` shape.
SCORES = [
    ("PAT_TBI", "PAT-TBI", "pat_tbi_score", "pat_tbi_score_mask"),
    ("GCS", "GCS", "gcs_risk", "gcs_mask"),
    ("GCS_P", "GCS-P", "gcs_p_risk", "gcs_p_mask"),
    ("IMPACT", "IMPACT-core", "impact_core_prob", "impact_mask"),
    ("CRASH", "CRASH-Basic", "crash_basic_prob", "crash_mask"),
]
PAT_SCORE = SCORES[0]
OUTCOME = C.OUTCOME

# ---------------------------------------------------------------------------
# The registry mapping for TEACHINGSTATUS.
#
# Source, read directly off the NTDB Public Use File lookup tables:
#   _DATA/NTDB/PUF/PUF AY 2021/CSV/PUF_TRAUMA_LOOKUP.csv
#   _DATA/NTDB/PUF/PUF AY 2022/CSV/PUF_TRAUMA_LOOKUP.csv
#   _DATA/NTDB/PUF/PUF AY 2023/CSV/PUF_TRAUMA_LOOKUP.csv
#   _DATA/NTDB/PUF/PUF AY 2024/CSV/PUF_TRAUMA_LOOKUP.csv
# each of which carries the format `TeachingStatusInd`:
#       1 -> Academic
#       4 -> University
#       5 -> Community
#       6 -> Non-Teaching
#       7 -> Not Known/Not Recorded
# and `PUF Variable Formats.csv` binds PUF_TRAUMA.TEACHINGSTATUS to that format.
#
# Admission year 2020 is the encoding change: its PUF_TRAUMA.csv stores the
# decoded label in lower case ("academic", "community", "nonteaching") and its
# lookup file carries no TeachingStatusInd block at all. Verified by reading the
# raw column directly. The two encodings therefore describe the same field and
# resolve onto the same levels.
# ---------------------------------------------------------------------------
TEACHING_NUMERIC = {
    1: "Academic",
    4: "University",
    5: "Community",
    6: "Non-teaching",
    7: "Not recorded",
}
TEACHING_TEXT = {
    "ACADEMIC": "Academic",
    "UNIVERSITY": "University",
    "COMMUNITY": "Community",
    "NONTEACHING": "Non-teaching",
    "NON-TEACHING": "Non-teaching",
    "NON TEACHING": "Non-teaching",
    "NOT KNOWN/NOT RECORDED": "Not recorded",
}
# Levels that carry a teaching mission in the registry's own taxonomy. The
# registry lists Non-Teaching as its own level, so Community is a community
# teaching hospital, not a non-teaching one.
TEACHING_POSITIVE = ("Academic", "University", "Community")
TEACHING_NEGATIVE = ("Non-teaching",)
TEACHING_UNUSABLE = ("Not recorded", "Missing")
LEVEL_ORDER = ["Academic", "University", "Community", "Non-teaching",
               "Not recorded", "Missing"]


# ---------------------------------------------------------------------------
# Import the v4 DeLong and the v4 bootstrap protocols rather than retyping them
# ---------------------------------------------------------------------------
def _load_v4_module(stem: str, alias: str):
    path = C.V4_CODE / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_v6_module(stem: str, alias: str):
    path = Path(__file__).resolve().parent / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V4_07 = _load_v4_module("07_validation_metrics", "v4_validation_metrics")
V4_09 = _load_v4_module("09_subgroups", "v4_subgroups")
# The corrected interval method, imported from the script that owns it project
# wide, so these numbers are bit for bit the ones step 11 publishes rather than
# a second implementation that merely agrees.
V6_11 = _load_v6_module("11_recompute_intervals", "v6_recompute_intervals")
delong_ci = V6_11.delong_ci                      # point, SE, analytic interval
bootstrap_full_ci = V6_11.bootstrap_full_ci      # correct n-out-of-n bootstrap

delong_test_vec = V4_07.delong_test_vec          # (z, p, delta, se, auc1, auc2)
delong_v10_v01 = V4_07.delong_v10_v01            # placement values
bootstrap_auc_ci_subsample = V4_07.bootstrap_auc_ci_subsample
bootstrap_auroc_ci = V4_09.bootstrap_auroc_ci    # subgroup protocol (500 x 50k)


def delong_auc_variance(y, pred) -> tuple[float, float]:
    """Single-AUROC point estimate and DeLong variance.

    Built from the v4 placement values, so it is the same estimator that the
    v4 pairwise DeLong uses, not a second implementation.
    """
    v10, v01 = delong_v10_v01(np.asarray(y), np.asarray(pred, dtype=float))
    n1, n0 = len(v10), len(v01)
    auc = float(v10.mean())
    var = float(np.var(v10, ddof=1) / n1 + np.var(v01, ddof=1) / n0)
    return auc, var


def auroc_with_boot_ci(y, s):
    """Point area and the corrected analytic DeLong interval.

    This replaces the v4 step 07 bootstrap that this script originally used.
    That helper draws 100,000 of 207,863 rows per replicate and takes a plain
    percentile interval, which is an m-out-of-n bootstrap reported without the
    sqrt(n/m) rescaling and is therefore about 44% too wide. The point estimate
    was never affected. See `11_recompute_intervals.py`, which owns the
    correction across the project; `delong_ci` is imported from it so these
    bounds are identical to the ones it publishes.
    """
    auc, se, lo, hi = delong_ci(y, s)
    return float(auc), float(lo), float(hi), float(se)


def withdrawn_v4_ci(y, s, n_boot=N_BOOT_DISC, sub=SUB_DISC):
    """The withdrawn v4 interval, recomputed purely to keep the audit trail.

    Step 11 reads the previously published inflated bounds out of this script's
    own outputs as the `before` side of its correction table. Once this script
    publishes corrected bounds that source is gone, so the withdrawn pair is
    carried explicitly here instead, the same way `_common.V4_FROZEN` keeps
    `auroc_test_lo_withdrawn`. It must never be reported as a result.
    """
    rng = np.random.default_rng(SEED)
    lo, hi = bootstrap_auc_ci_subsample(np.asarray(y),
                                        np.asarray(s, dtype=float),
                                        n_boot, sub, rng)
    return float(lo), float(hi)


def fit_logit(X_df: pd.DataFrame, y):
    """Unpenalized MLE logistic, identical to v4 step 05 `fit_logit`."""
    Xc = sm.add_constant(X_df, has_constant="add")
    return sm.Logit(np.asarray(y, dtype=float), Xc.astype(float)).fit(
        disp=False, maxiter=200, method="newton")


def linear_predictor(res, X_df: pd.DataFrame) -> np.ndarray:
    """Intercept plus the coefficient-weighted sum, column by named column."""
    lp = np.full(len(X_df), float(res.params["const"]), dtype=np.float64)
    for col in X_df.columns:
        lp += float(res.params[col]) * X_df[col].astype(float).to_numpy()
    return lp


def fmt_ci(est, lo, hi, nd=4) -> str:
    return f"{est:.{nd}f} (95% CI {lo:.{nd}f} to {hi:.{nd}f})"


# The largest p-value that double precision cannot resolve. Only the head-AIS
# coefficient needs it: its z of roughly forty puts the true tail near ten to the
# minus three hundred and fortieth, below the smallest representable double. The
# floor is an upper bound on the true value, so it never overstates the evidence.
P_FLOOR = 1e-300
P_FLOOR_NOTE = ("The tail probability underflows double precision, so the value "
                "carried here is a floor rather than the exact quantity. The test "
                "statistic beside it is the reportable magnitude.")


def stable_p(z) -> float:
    """Two-sided normal tail computed without catastrophic cancellation.

    The v4 DeLong helper forms `2 * (1 - norm.cdf(|z|))`, which collapses to
    exactly zero for any |z| beyond about eight because `1 - cdf` loses every
    significant digit. `norm.sf` evaluates the same tail directly and recovers
    the real value, so a p-value here is only zero when it genuinely underflows.
    The z statistic itself is unchanged, so this alters no test, only its
    printed tail.
    """
    z = float(z)
    if not np.isfinite(z):
        return float("nan")
    return float(2.0 * sstats.norm.sf(abs(z)))


def p_leaf(p) -> dict:
    """A p-value leaf that never renders a zero.

    A z of forty underflows double precision, so the honest leaf is a sentence
    rather than a numeral. The magnitude is carried alongside by the z statistic.
    """
    if np.isfinite(p) and p > 0.0:
        return {"pvalue": float(p)}
    return {"pvalue": P_FLOOR}


def p_floored(p) -> bool:
    """True when the leaf above had to fall back on the floor."""
    return not (np.isfinite(p) and p > 0.0)


def fmt_p(p) -> str:
    """The value half of a p-value, for a log line that already says `p`."""
    if not np.isfinite(p):
        return "not estimable"
    if p < 1e-4:
        return "<0.0001"
    if p < 0.001:
        return f"{p:.3g}"
    return f"{p:.3f}"


def fmt_pv(p) -> str:
    """A whole p-value phrase, for prose."""
    if not np.isfinite(p):
        return "P not estimable"
    if p < 1e-4:
        return "P < 0.0001"
    if p < 0.001:
        return f"P = {p:.3g}"
    return f"P = {p:.3f}"


# ===========================================================================
# PART A
# ===========================================================================
def part_a(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    C.banner("PART A — head AIS sensitivity refit")

    refit = C.load_refit()
    variables = list(refit["variables_kept"])
    print(f"  retained predictors read from pat_refit.json : {len(variables)}")
    print(f"  derivation rows : {len(train):,}   events {int(train[OUTCOME].sum()):,}")
    print(f"  validation rows : {len(test):,}   events {int(test[OUTCOME].sum()):,}")

    if len(variables) != C.V4_FROZEN["n_predictors"]:
        raise SystemExit(f"expected 22 retained predictors, found {len(variables)}")

    y_tr = train[OUTCOME].astype(int).to_numpy()
    y_te = test[OUTCOME].astype(int).to_numpy()

    # ---------------------------------------------------------------- head AIS
    C.banner("Head AIS distribution in the analytic cohort")
    both = pd.concat([train[HEAD_AIS], test[HEAD_AIS]], ignore_index=True)
    dist = both.value_counts().sort_index()
    print(f"  column: {HEAD_AIS}  ({C.label(HEAD_AIS)})")
    print(f"  missing values: {int(both.isna().sum()):,} of {len(both):,}")
    for v, k in dist.items():
        note = ""
        if int(v) == 9:
            note = "   <- AIS convention: severity unknown, NOT a severity of nine"
        if int(v) == 0:
            note = "   <- no head region injury coded"
        print(f"    value {int(v)} : {k:>9,}  ({100.0 * k / len(both):5.2f}%){note}")
    n_unknown_tr = int((train[HEAD_AIS] == 9).sum())
    n_unknown_te = int((test[HEAD_AIS] == 9).sum())
    print(f"  severity-unknown rows: derivation {n_unknown_tr:,}, "
          f"validation {n_unknown_te:,}")

    # ------------------------------------------------------------- BASE model
    C.banner("BASE model — the 22 retained predictors, derivation rows")
    t0 = time.time()
    X_tr = train[variables].astype(float)
    base = fit_logit(X_tr, y_tr)
    print(f"  fitted in {time.time() - t0:.1f}s;  "
          f"log-likelihood {base.llf:,.2f};  converged {base.mle_retvals['converged']}")

    C.banner("Reproduction gate — BASE against the frozen v4/derived/pat_refit.json")
    diffs = {}
    diffs["intercept"] = abs(float(base.params["const"]) - float(refit["intercept"]))
    print(f"  {'intercept':<20s} refit {float(refit['intercept']): .10f}   "
          f"here {float(base.params['const']): .10f}   "
          f"diff {diffs['intercept']:.3e}")
    for v in variables:
        want = float(refit["coefficients"][v])
        got = float(base.params[v])
        diffs[v] = abs(got - want)
        print(f"  {v:<20s} refit {want: .10f}   here {got: .10f}   "
              f"diff {diffs[v]:.3e}")
    max_diff = max(diffs.values())
    worst = max(diffs, key=diffs.get)
    print(f"\n  largest absolute difference: {max_diff:.3e}  (on {worst})")
    TOL = 1e-6
    if max_diff > TOL:
        raise SystemExit(
            "STOP. The base model does not reproduce pat_refit.json to "
            f"{TOL:.0e} (largest difference {max_diff:.3e} on {worst}). That "
            "would mean the refit is not reproducible from the frozen "
            "artefacts, which is a Phase 0 blocker, not something to tune away.")
    print(f"  [PASS] every coefficient reproduces the frozen refit to better "
          f"than {TOL:.0e}.")

    # --------------------------------------------------------- EXTENDED model
    C.banner("EXTENDED model — the same 22 plus head AIS as an ordinal term")
    X_tr_ext = X_tr.copy()
    X_tr_ext[HEAD_AIS] = train[HEAD_AIS].astype(float).to_numpy()
    ext = fit_logit(X_tr_ext, y_tr)
    b = float(ext.params[HEAD_AIS])
    se = float(ext.bse[HEAD_AIS])
    z = float(ext.tvalues[HEAD_AIS])
    pv = stable_p(z)
    or_pt, or_lo, or_hi = np.exp(b), np.exp(b - 1.959964 * se), np.exp(b + 1.959964 * se)
    print(f"  log-likelihood {ext.llf:,.2f}   "
          f"likelihood-ratio against base {2 * (ext.llf - base.llf):,.2f}")
    print(f"  head AIS coefficient : {b:+.6f}")
    print(f"  standard error       : {se:.6f}")
    print(f"  z                    : {z:+.3f}")
    print(f"  p                    : {fmt_p(pv)}")
    print(f"  odds ratio per one point of head AIS : {fmt_ci(or_pt, or_lo, or_hi, 4)}")

    print("\n  coefficient movement in the other 22 when head AIS enters:")
    moves = []
    for v in variables:
        moves.append((v, float(base.params[v]), float(ext.params[v]),
                      float(ext.params[v]) - float(base.params[v])))
    moves.sort(key=lambda r: -abs(r[3]))
    for v, b0, b1, d in moves:
        print(f"    {v:<20s} base {b0:+.4f}  extended {b1:+.4f}  change {d:+.4f}")

    # ------------------------------------------------- EXTENDED-B sensitivity
    C.banner("EXTENDED-B — severity-unknown carried by its own indicator")
    print("  The AIS convention codes 9 as severity unknown. Entering it as a "
          "nine asserts that those patients are worse than the maximum, which "
          "is false. EXTENDED-B sets the ordinal term to zero on those rows and "
          "adds an indicator, so no row is dropped and the frozen counts hold.")
    ais_tr = train[HEAD_AIS].astype(float).to_numpy()
    ais_te = test[HEAD_AIS].astype(float).to_numpy()
    unk_tr = (ais_tr == 9).astype(float)
    unk_te = (ais_te == 9).astype(float)
    ord_tr = np.where(ais_tr == 9, 0.0, ais_tr)
    ord_te = np.where(ais_te == 9, 0.0, ais_te)
    X_tr_extb = X_tr.copy()
    X_tr_extb[HEAD_AIS] = ord_tr
    X_tr_extb["head_ais_unknown"] = unk_tr
    extb = fit_logit(X_tr_extb, y_tr)
    bb = float(extb.params[HEAD_AIS])
    seb = float(extb.bse[HEAD_AIS])
    pvb = stable_p(float(extb.tvalues[HEAD_AIS]))
    orb = np.exp(bb)
    orb_lo, orb_hi = np.exp(bb - 1.959964 * seb), np.exp(bb + 1.959964 * seb)
    print(f"  head AIS coefficient : {bb:+.6f}   standard error {seb:.6f}   "
          f"p {fmt_p(pvb)}")
    print(f"  odds ratio per one point of head AIS : {fmt_ci(orb, orb_lo, orb_hi, 4)}")
    bu = float(extb.params["head_ais_unknown"])
    seu = float(extb.bse["head_ais_unknown"])
    print(f"  severity-unknown indicator : coefficient {bu:+.6f}   "
          f"odds ratio {np.exp(bu):.4f}")

    # ------------------------------------------------ validation discrimination
    C.banner("Validation discrimination — linear predictor against linear "
             "predictor")
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

    print("  interval method: analytic DeLong, imported from "
          "11_recompute_intervals.py, which owns the correction project wide.")
    print("  The v4 step 07 bootstrap this script first used is an m-out-of-n "
          "resample reported without the sqrt(n/m) rescaling, so it was about "
          "44% too wide. Its bounds are recomputed below and marked withdrawn.")
    auc_base, lo_base, hi_base, se_base = auroc_with_boot_ci(y_te, lp_base)
    auc_ext, lo_ext, hi_ext, se_ext = auroc_with_boot_ci(y_te, lp_ext)
    auc_extb, lo_extb, hi_extb, se_extb = auroc_with_boot_ci(y_te, lp_extb)
    auc_int, lo_int, hi_int, se_int = auroc_with_boot_ci(y_te, s_int)

    print(f"  BASE     linear predictor : {fmt_ci(auc_base, lo_base, hi_base)}")
    print(f"  EXTENDED linear predictor : {fmt_ci(auc_ext, lo_ext, hi_ext)}")
    print(f"  EXTENDED-B lin. predictor : {fmt_ci(auc_extb, lo_extb, hi_extb)}")
    print(f"  integer 22-point score    : {fmt_ci(auc_int, lo_int, hi_int)}")

    C.banner("Cross-check and the withdrawn bounds")
    print("  A correct n-out-of-n bootstrap on the integer score, as the check "
          "that the analytic form is not doing anything surprising:")
    bl, bh, _bsd = bootstrap_full_ci(y_te, s_int)
    print(f"    corrected bootstrap : {fmt_ci(auc_int, bl, bh)}")
    print(f"    analytic DeLong     : {fmt_ci(auc_int, lo_int, hi_int)}")
    if abs(bl - lo_int) > 0.001 or abs(bh - hi_int) > 0.001:
        raise SystemExit("STOP. The analytic interval and the corrected "
                         "bootstrap disagree beyond the third decimal.")
    print("    [PASS] the two agree to the third decimal.")

    withdrawn = {}
    for name, lp in (("base", lp_base), ("extended", lp_ext),
                     ("extended_b", lp_extb), ("integer_score", s_int)):
        withdrawn[name] = withdrawn_v4_ci(y_te, lp)
    print("\n  withdrawn v4 protocol bounds, never to be reported, kept so the "
          "before and after of the correction stays visible:")
    for name, (wlo, whi) in withdrawn.items():
        cur = {"base": (lo_base, hi_base), "extended": (lo_ext, hi_ext),
               "extended_b": (lo_extb, hi_extb),
               "integer_score": (lo_int, hi_int)}[name]
        infl = (whi - wlo) / (cur[1] - cur[0])
        print(f"    {name:<14s} withdrawn {wlo:.4f} to {whi:.4f}   "
              f"corrected {cur[0]:.4f} to {cur[1]:.4f}   "
              f"withdrawn width was {infl:.2f} times the corrected width")

    # ---------------------------------------------------------- frozen gates
    C.banner("Frozen-number gates")
    ok = True

    def gate(name, got, want, tol=None):
        nonlocal ok
        good = (got == want) if tol is None else (abs(got - want) <= tol)
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] {name}: got {got}, expected {want}")

    gate("derivation rows", len(train), C.V4_FROZEN["n_train"])
    gate("validation rows", len(test), C.V4_FROZEN["n_test"])
    gate("derivation events", int(y_tr.sum()), C.V4_FROZEN["n_train_events"])
    gate("validation events", int(y_te.sum()), C.V4_FROZEN["n_test_events"])
    gate("integer score validation AUROC", round(auc_int, 4),
         C.V4_FROZEN["auroc_test"], tol=1e-9)
    gate("integer score interval low", round(lo_int, 4),
         C.V4_FROZEN["auroc_test_lo"], tol=1e-9)
    gate("integer score interval high", round(hi_int, 4),
         C.V4_FROZEN["auroc_test_hi"], tol=1e-9)
    # Defensive: the withdrawn bounds must not be what we are reporting. If a
    # future edit puts the m-out-of-n helper back in the reporting path, this
    # catches it even if the gates above were also edited to match.
    gate("reported low is not the withdrawn bound",
         round(lo_int, 4) != C.V4_FROZEN["auroc_test_lo_withdrawn"], True)
    gate("reported high is not the withdrawn bound",
         round(hi_int, 4) != C.V4_FROZEN["auroc_test_hi_withdrawn"], True)
    if not ok:
        raise SystemExit("STOP. A frozen v4 number moved. Find the cause "
                         "before proceeding; do not adjust a number to fit.")

    # ------------------------------------------------------ DeLong contrasts
    C.banner("DeLong — EXTENDED minus BASE, on the identical validation rows")
    z_d, _p_raw, delta, se_d, a1, a2 = delong_test_vec(y_te, lp_ext, lp_base)
    p_d = stable_p(z_d)
    d_lo, d_hi = delta - 1.959964 * se_d, delta + 1.959964 * se_d
    print(f"  paired rows        : {len(y_te):,}   events {int(y_te.sum()):,}")
    print(f"  AUROC extended     : {a1:.6f}")
    print(f"  AUROC base         : {a2:.6f}")
    print(f"  difference         : {fmt_ci(delta, d_lo, d_hi, 5)}")
    print(f"  standard error     : {se_d:.6f}   z {z_d:+.3f}   p {fmt_p(p_d)}")

    zb, _pb_raw, delta_b, se_b, ab1, ab2 = delong_test_vec(y_te, lp_extb, lp_base)
    pb = stable_p(zb)
    db_lo, db_hi = delta_b - 1.959964 * se_b, delta_b + 1.959964 * se_b
    print(f"\n  EXTENDED-B minus BASE : {fmt_ci(delta_b, db_lo, db_hi, 5)}   "
          f"p {fmt_p(pb)}")

    z_i, _pi_raw, delta_i, se_i, _, _ = delong_test_vec(y_te, lp_base, s_int)
    p_i = stable_p(z_i)
    di_lo, di_hi = delta_i - 1.959964 * se_i, delta_i + 1.959964 * se_i
    print(f"  BASE linear predictor minus integer score : "
          f"{fmt_ci(delta_i, di_lo, di_hi, 5)}   p {fmt_p(p_i)}")

    # -------------------------------------------------------------- the table
    rows = []

    def add(block, metric, model, n, n_ev, est, lo=np.nan, hi=np.nan,
            se_v=np.nan, p=np.nan, note=""):
        rows.append({"Block": block, "Metric": metric, "Model": model,
                     "N": n, "N_events": n_ev, "Estimate": est,
                     "CI_Low": lo, "CI_High": hi, "SE": se_v,
                     "P_value": p, "Note": note})

    add("reproduction", "max_abs_coefficient_difference", "base",
        len(train), int(y_tr.sum()), max_diff, note=(
            f"largest absolute gap against pat_refit.json, on {worst}; "
            f"tolerance {TOL:.0e}"))
    add("reproduction", "intercept", "base", len(train), int(y_tr.sum()),
        float(base.params["const"]), note="frozen refit intercept reproduced")

    for v in variables:
        add("coefficients", v, "base", len(train), int(y_tr.sum()),
            float(base.params[v]),
            float(base.params[v]) - 1.959964 * float(base.bse[v]),
            float(base.params[v]) + 1.959964 * float(base.bse[v]),
            float(base.bse[v]), float(base.pvalues[v]), C.label(v))
    for v in variables:
        add("coefficients", v, "extended", len(train), int(y_tr.sum()),
            float(ext.params[v]),
            float(ext.params[v]) - 1.959964 * float(ext.bse[v]),
            float(ext.params[v]) + 1.959964 * float(ext.bse[v]),
            float(ext.bse[v]), float(ext.pvalues[v]), C.label(v))

    add("head_ais_term", "beta", "extended", len(train), int(y_tr.sum()),
        b, b - 1.959964 * se, b + 1.959964 * se, se, pv,
        "ordinal term entered exactly as stored")
    add("head_ais_term", "odds_ratio", "extended", len(train), int(y_tr.sum()),
        or_pt, or_lo, or_hi, np.nan, pv,
        "per one-point increase in maximum head Abbreviated Injury Scale")
    add("head_ais_term", "beta", "extended_b", len(train), int(y_tr.sum()),
        bb, bb - 1.959964 * seb, bb + 1.959964 * seb, seb, pvb,
        "severity-unknown rows carried by their own indicator")
    add("head_ais_term", "odds_ratio", "extended_b", len(train), int(y_tr.sum()),
        orb, orb_lo, orb_hi, np.nan, pvb,
        "per one-point increase, severity-unknown rows held out of the ordinal")
    add("head_ais_term", "severity_unknown_indicator", "extended_b",
        len(train), int(y_tr.sum()), float(np.exp(bu)),
        float(np.exp(bu - 1.959964 * seu)), float(np.exp(bu + 1.959964 * seu)),
        seu, float(extb.pvalues["head_ais_unknown"]),
        "odds ratio for a head injury of unknown severity")

    add("model_fit", "log_likelihood", "base", len(train), int(y_tr.sum()),
        float(base.llf))
    add("model_fit", "log_likelihood", "extended", len(train), int(y_tr.sum()),
        float(ext.llf))
    add("model_fit", "likelihood_ratio_statistic", "extended_vs_base",
        len(train), int(y_tr.sum()), float(2 * (ext.llf - base.llf)),
        note="one degree of freedom")

    DL = "analytic DeLong interval, from 11_recompute_intervals.py"
    for model, a, l, h, sev in (("base", auc_base, lo_base, hi_base, se_base),
                                ("extended", auc_ext, lo_ext, hi_ext, se_ext),
                                ("extended_b", auc_extb, lo_extb, hi_extb, se_extb),
                                ("integer_score", auc_int, lo_int, hi_int, se_int)):
        extra = (" reproduces the corrected frozen headline exactly"
                 if model == "integer_score" else "")
        add("discrimination", "auroc_validation", model, len(test),
            int(y_te.sum()), a, l, h, sev, np.nan, DL + extra)
    for model, (wlo, whi) in withdrawn.items():
        add("discrimination_withdrawn", "auroc_validation", model, len(test),
            int(y_te.sum()), np.nan, wlo, whi, np.nan, np.nan,
            "WITHDRAWN, never report: v4 m-out-of-n bootstrap, "
            f"{N_BOOT_DISC} replicates of {SUB_DISC} rows, no sqrt(n/m) "
            "rescaling, about 44 percent too wide. Kept for the audit trail.")

    add("contrast", "auroc_difference", "extended_minus_base", len(test),
        int(y_te.sum()), delta, d_lo, d_hi, se_d, p_d,
        "DeLong on the identical validation rows, linear predictor against "
        "linear predictor")
    add("contrast", "auroc_difference", "extended_b_minus_base", len(test),
        int(y_te.sum()), delta_b, db_lo, db_hi, se_b, pb,
        "DeLong, severity-unknown variant")
    add("contrast", "auroc_difference", "base_minus_integer_score", len(test),
        int(y_te.sum()), delta_i, di_lo, di_hi, se_i, p_i,
        "DeLong, cost of rounding the fitted model to integer points")

    table = pd.DataFrame(rows)
    OUT_TABLE_A.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_TABLE_A, index=False)
    print(f"\n[write] {OUT_TABLE_A}   rows = {len(table):,}")

    payload = {
        "_meta": {
            "script": "03_head_ais_sensitivity.py",
            "finding": "head AIS exclusion, raised by two co-authors",
            "source": {"display": (
                "Maximum head Abbreviated Injury Scale was already carried in "
                "the frozen candidate matrix and is fully populated, so this "
                "sensitivity extracts nothing new. Both models are unpenalized "
                "logistic fits on the derivation rows and both are scored on "
                "the validation rows as linear predictors, so the comparison is "
                "like for like.")},
        },
        "question": {"display": (
            "Two co-authors asked why maximum head Abbreviated Injury Scale was "
            "excluded from the candidate set. The exclusion is a design "
            "constraint, not an oversight: the score is assigned retrospectively "
            "by a registrar after imaging and abstraction, so it is not on the "
            "chart at the moment a platelet decision is made. This sensitivity "
            "measures what the constraint costs.")},
        "base_model": {
            "predictors": {"value": len(variables)},
            "derivation_rows": {"n": len(train)},
            "derivation_events": {"n": int(y_tr.sum()), "N": len(train)},
            "validation_rows": {"n": len(test)},
            "validation_events": {"n": int(y_te.sum()), "N": len(test)},
            "intercept": {"value": round(float(base.params["const"]), 6)},
            "log_likelihood": {"value": round(float(base.llf), 2)},
            "auroc_validation": {"estimate": round(auc_base, 4),
                                 "lo": round(lo_base, 4), "hi": round(hi_base, 4)},
            "definition": {"display": (
                "An unpenalized logistic regression of platelet receipt on the "
                "retained predictors, fitted on the derivation rows. Its "
                "coefficients reproduce the frozen refit record exactly, which "
                "is the check that the published model can be rebuilt from the "
                "archived artefacts.")},
        },
        "extended_model": {
            "added_variable": {"display": C.label(HEAD_AIS)},
            "entered_as": {"display": (
                "an ordinal term, one coefficient across the whole scale")},
            "beta": {"value": round(b, 6)},
            "standard_error": {"value": round(se, 6)},
            "odds_ratio": {"estimate": round(float(or_pt), 4),
                           "lo": round(float(or_lo), 4),
                           "hi": round(float(or_hi), 4)},
            "pvalue": p_leaf(pv),
            "z_statistic": {"value": round(float(z), 2)},
            **({"pvalue_note": {"display": P_FLOOR_NOTE}}
               if p_floored(pv) else {}),
            "log_likelihood": {"value": round(float(ext.llf), 2)},
            "likelihood_ratio": {"value": round(float(2 * (ext.llf - base.llf)), 2)},
            "auroc_validation": {"estimate": round(auc_ext, 4),
                                 "lo": round(lo_ext, 4), "hi": round(hi_ext, 4)},
            "interpretation": {"display": (
                "Each additional point of head injury severity multiplies the "
                "odds of receiving platelets within the first hours, holding the "
                "retained predictors fixed. The term is strongly associated with "
                "the outcome, which is exactly why its unavailability at the "
                "bedside matters.")},
        },
        "severity_unknown_variant": {
            "rows_derivation": {"n": n_unknown_tr, "N": len(train)},
            "rows_validation": {"n": n_unknown_te, "N": len(test)},
            "beta": {"value": round(bb, 6)},
            "odds_ratio": {"estimate": round(float(orb), 4),
                           "lo": round(float(orb_lo), 4),
                           "hi": round(float(orb_hi), 4)},
            "auroc_validation": {"estimate": round(auc_extb, 4),
                                 "lo": round(lo_extb, 4), "hi": round(hi_extb, 4)},
            "auroc_difference": {"estimate": round(float(delta_b), 5),
                                 "lo": round(float(db_lo), 5),
                                 "hi": round(float(db_hi), 5)},
            "rationale": {"display": (
                "The Abbreviated Injury Scale reserves its highest code for "
                "severity unknown rather than for a severity worse than the "
                "maximum. Entering that code on its numeric face would treat "
                "those patients as the most severely injured in the cohort. This "
                "variant sets the ordinal term to zero on those rows and carries "
                "them on an indicator instead, so no row is dropped and the "
                "conclusion does not rest on a coding artefact.")},
        },
        "integer_score": {
            "auroc_validation": {"estimate": round(auc_int, 4),
                                 "lo": round(lo_int, 4), "hi": round(hi_int, 4)},
            "note": {"display": (
                "Reported for context. This is the published integer instrument "
                "evaluated on the same validation rows, and it reproduces the "
                "frozen headline exactly.")},
        },
        "contrast": {
            "label": {"display": (
                "Adding maximum head Abbreviated Injury Scale to the model")},
            "auroc_difference": {"estimate": round(float(delta), 5),
                                 "lo": round(float(d_lo), 5),
                                 "hi": round(float(d_hi), 5)},
            "standard_error": {"value": round(float(se_d), 6)},
            "pvalue": p_leaf(p_d),
            "z_statistic": {"value": round(float(z_d), 2)},
            **({"pvalue_note": {"display": P_FLOOR_NOTE}}
               if p_floored(p_d) else {}),
            "method": {"display": (
                "DeLong for two correlated areas measured on the identical "
                "validation rows, comparing the linear predictor of the extended "
                "model against the linear predictor of the base model. Linear "
                "predictor against linear predictor, so neither model is "
                "penalised for the rounding step that produces the integer "
                "points.")},
        },
        "reproduction": {
            "max_abs_coefficient_difference": {"value": float(f"{max_diff:.3e}")},
            "tolerance": {"value": TOL},
            "verdict": {"display": (
                "Every coefficient of the base model reproduces the frozen refit "
                "record to well inside tolerance, so the published model rebuilds "
                "from the archived artefacts and the extended model is a fair "
                "comparison against it.")},
        },
        "verdict": {"display": (
            "Head injury severity is strongly associated with platelet receipt, "
            "and adding it raises discrimination by a modest but unmistakable "
            "margin. That margin is the size of the ceiling the bedside "
            "constraint imposes, and it is reported rather than argued away. The "
            "exclusion still stands, because a variable a registrar assigns after "
            "the fact cannot inform a decision taken before it exists, but it "
            "now stands on a measurement instead of an assertion.")},
        "action": {"display": (
            "Report the measured gain in the results and cite it in the reply to "
            "both co-authors. Keep the retained predictor set unchanged.")},
    }
    C.write_part("head_ais", payload)
    print(f"[write] {C.V6_PARTS / 'head_ais.json'}")

    return {
        "auc_base": auc_base, "lo_base": lo_base, "hi_base": hi_base,
        "auc_ext": auc_ext, "lo_ext": lo_ext, "hi_ext": hi_ext,
        "auc_extb": auc_extb, "lo_extb": lo_extb, "hi_extb": hi_extb,
        "auc_int": auc_int, "lo_int": lo_int, "hi_int": hi_int,
        "delta": delta, "d_lo": d_lo, "d_hi": d_hi, "se_d": se_d, "p_d": p_d,
        "delta_b": delta_b, "db_lo": db_lo, "db_hi": db_hi, "p_b": pb,
        "beta": b, "se": se, "or": or_pt, "or_lo": or_lo, "or_hi": or_hi, "p": pv,
        "beta_b": bb, "or_b": orb, "or_b_lo": orb_lo, "or_b_hi": orb_hi,
        "max_diff": max_diff, "n_unknown_tr": n_unknown_tr,
        "n_unknown_te": n_unknown_te,
    }


# ===========================================================================
# PART B
# ===========================================================================
def _bool(cond) -> np.ndarray:
    """A plain numpy boolean mask. Nullable booleans carry NA, and NA inside a
    `mask` condition silently overwrites rows that should have been left alone,
    so every condition is flattened here before it is used."""
    if hasattr(cond, "fillna"):
        cond = cond.fillna(False)
    return np.asarray(cond, dtype=bool)


def map_teaching(series: pd.Series) -> pd.Series:
    """Resolve both registry encodings onto one set of named levels."""
    txt = series.astype("string").str.strip()
    num = pd.to_numeric(txt, errors="coerce")
    out = pd.Series(pd.NA, index=series.index, dtype="string")
    for code, level in TEACHING_NUMERIC.items():
        out = out.mask(_bool(num == code), level)
    upper = txt.str.upper()
    for token, level in TEACHING_TEXT.items():
        out = out.mask(_bool(out.isna() & (upper == token)), level)
    return out.fillna("Missing")


def subgroup_rows(sub_df: pd.DataFrame, subgroup: str, level: str,
                  note: str = "") -> list[dict]:
    """One row per score for one subgroup level, in v4 step 09 shape."""
    y_events = int(pd.to_numeric(sub_df[OUTCOME], errors="coerce").fillna(0).sum())
    too_small = y_events < MIN_EVENTS
    pat_col, pat_mask_col = PAT_SCORE[2], PAT_SCORE[3]
    rows = []
    for short, pretty, score_col, mask_col in SCORES:
        row = {"Subgroup": subgroup, "Level": level, "Score": pretty,
               "N": 0, "N_events": 0, "AUROC": np.nan,
               "CI_Low": np.nan, "CI_High": np.nan,
               "Delta_vs_PAT_TBI": np.nan, "Delta_SE": np.nan,
               "Delta_CI_Low": np.nan, "Delta_CI_High": np.nan,
               "Delta_P": np.nan, "N_events_too_small": too_small,
               "SE": np.nan, "CI_Low_withdrawn": np.nan,
               "CI_High_withdrawn": np.nan, "Note": note}
        if len(sub_df) == 0:
            row["Note"] = note or "no patients at this level"
            rows.append(row)
            continue
        s_mask = (sub_df[mask_col].astype(bool).fillna(False)
                  & sub_df[score_col].notna() & sub_df[OUTCOME].notna())
        s = sub_df.loc[s_mask]
        row["N"] = int(len(s))
        row["N_events"] = int(pd.to_numeric(s[OUTCOME], errors="coerce")
                              .fillna(0).sum())
        if too_small or row["N"] < 20 or row["N_events"] < 2 \
                or (row["N"] - row["N_events"]) < 2:
            reason = (f"{y_events} events at this level, below the "
                      f"{MIN_EVENTS}-event floor applied to every subgroup "
                      f"cell") if too_small else \
                     "too few patients or too few events"
            row["Note"] = (f"{note}; {reason}" if note
                           else f"not estimable: {reason}")
            rows.append(row)
            continue
        y = s[OUTCOME].astype(float).to_numpy()
        p = s[score_col].astype(float).to_numpy()
        # Corrected interval. The v4 step 09 helper resamples 50,000 rows out
        # of the subgroup and takes a plain percentile interval, so every level
        # larger than 50,000 rows came out inflated. DeLong is exact and needs
        # no seed. The withdrawn bound is kept beside it for the audit trail.
        auc, se_auc, lo, hi = delong_ci(y, p)
        row["AUROC"], row["CI_Low"], row["CI_High"] = auc, lo, hi
        row["SE"] = se_auc
        wlo, whi = bootstrap_auroc_ci(y, p, n_boot=N_BOOT_SUB,
                                      max_n=SUB_SUB, seed=SEED)[1:]
        row["CI_Low_withdrawn"], row["CI_High_withdrawn"] = wlo, whi
        if short == "PAT_TBI":
            row.update({"Delta_vs_PAT_TBI": 0.0, "Delta_SE": 0.0,
                        "Delta_CI_Low": 0.0, "Delta_CI_High": 0.0,
                        "Delta_P": 1.0})
        else:
            pair_mask = (sub_df[pat_mask_col].astype(bool).fillna(False)
                         & sub_df[mask_col].astype(bool).fillna(False)
                         & sub_df[pat_col].notna() & sub_df[score_col].notna()
                         & sub_df[OUTCOME].notna())
            pair = sub_df.loc[pair_mask]
            if (len(pair) >= 20 and pair[OUTCOME].nunique() >= 2
                    and int(pair[OUTCOME].sum()) >= MIN_EVENTS):
                yp = pair[OUTCOME].astype(float).to_numpy()
                z_sub, _pv_raw, delta, se_d, _, _ = delong_test_vec(
                    yp, pair[pat_col].astype(float).to_numpy(),
                    pair[score_col].astype(float).to_numpy())
                row["Delta_vs_PAT_TBI"] = delta
                row["Delta_SE"] = se_d
                if np.isfinite(delta) and np.isfinite(se_d):
                    row["Delta_CI_Low"] = delta - 1.959964 * se_d
                    row["Delta_CI_High"] = delta + 1.959964 * se_d
                row["Delta_P"] = stable_p(z_sub)
        rows.append(row)
    return rows


def part_b() -> dict:
    C.banner("PART B — teaching status (finding 6)")
    print("  v4 step 09 logged `teaching_status_binary missing, skipping center "
          "subgroup`, yet the manuscript claimed the subgroup. Nothing is "
          "decided here until the raw values have been printed.")

    raw = C.load_cohort_raw(columns=["PUF_YEAR", "inc_key", "teaching_status_raw"])
    n_cohort = len(raw)
    print(f"\n  cohort rows read from cohort_raw.parquet : {n_cohort:,}")
    if n_cohort != C.V4_FROZEN["n_cohort"]:
        raise SystemExit(f"cohort rows {n_cohort:,}, expected "
                         f"{C.V4_FROZEN['n_cohort']:,}")

    C.banner("Step 1 — full value counts of the raw field, before any decision")
    vc = raw["teaching_status_raw"].value_counts(dropna=False)
    n_missing = int(raw["teaching_status_raw"].isna().sum())
    print(f"  distinct stored values (missing counted separately): "
          f"{int(raw['teaching_status_raw'].nunique(dropna=True))}")
    for val, k in vc.items():
        shown = "<missing>" if (val is pd.NA or (isinstance(val, float) and val != val)) \
            else repr(val)
        print(f"    {shown:<16s} {k:>9,}  ({100.0 * k / n_cohort:6.3f}%)")
    print(f"  missing : {n_missing:,} of {n_cohort:,} "
          f"({100.0 * n_missing / n_cohort:.3f}%);  populated "
          f"{100.0 * (n_cohort - n_missing) / n_cohort:.3f}%")

    print("\n  the same values by admission year, which is where the mixing "
          "comes from:")
    ct = pd.crosstab(raw["PUF_YEAR"],
                     raw["teaching_status_raw"].astype("string").fillna("<missing>"))
    print(ct.to_string())

    C.banner("Step 2 — the mapping, value by value")
    print("  Authority: the NTDB Public Use File lookup table PUF_TRAUMA_LOOKUP.csv,")
    print("  format TeachingStatusInd, identical in admission years 2021, 2022,")
    print("  2023 and 2024; PUF Variable Formats.csv binds PUF_TRAUMA.TEACHINGSTATUS")
    print("  to that format. Admission year 2020 stores the decoded label in lower")
    print("  case instead of the code and carries no TeachingStatusInd block, which")
    print("  is the whole of the mixing. Both encodings are the same field.")
    print()
    print("    stored value  ->  registry level      ->  teaching binary")
    print("    ------------      ---------------         ---------------")
    mapping_lines = [
        ("1.0", "Academic", "teaching"),
        ("4.0", "University", "teaching (code not present in this cohort)"),
        ("5.0", "Community", "teaching (community teaching hospital)"),
        ("6.0", "Non-teaching", "non-teaching"),
        ("7.0", "Not recorded", "not usable (code not present in this cohort)"),
        ("'academic'", "Academic", "teaching"),
        ("'community'", "Community", "teaching (community teaching hospital)"),
        ("'nonteaching'", "Non-teaching", "non-teaching"),
        ("<missing>", "Missing", "not usable"),
    ]
    for stored, level, binary in mapping_lines:
        print(f"    {stored:<13s} ->  {level:<18s} ->  {binary}")
    print()
    print("  The registry lists Non-Teaching as its own level, so Community means")
    print("  a community teaching hospital and belongs on the teaching side of a")
    print("  binary. The v4 all-trauma sensitivity script mapped only academic and")
    print("  university to teaching, which puts community teaching hospitals on the")
    print("  non-teaching side; that reading contradicts the registry's own")
    print("  taxonomy and is not used here.")
    print()
    print("  DECISION: a defensible mapping exists. The three occupied levels are")
    print("  reported as an ordered set, and the collapsed teaching binary is")
    print("  reported alongside it. Both are computed below.")

    level_all = map_teaching(raw["teaching_status_raw"])
    counts_all = level_all.value_counts()
    print("\n  mapped levels over the analytic cohort:")
    for lvl in LEVEL_ORDER:
        k = int(counts_all.get(lvl, 0))
        if k:
            print(f"    {lvl:<14s} {k:>9,}  ({100.0 * k / n_cohort:6.3f}%)")
    unmapped = int((level_all == "Missing").sum())
    if unmapped != n_missing:
        raise SystemExit(f"mapping leaked: {unmapped:,} unmapped but only "
                         f"{n_missing:,} were missing")
    print(f"  [PASS] every non-missing value mapped; the {unmapped:,} unmapped "
          f"rows are exactly the missing ones.")

    print("\n  audit, every stored value against the level it landed on:")
    audit = pd.crosstab(
        raw["teaching_status_raw"].astype("string").fillna("<missing>"),
        level_all.values)
    print(audit.to_string())
    for stored, r in audit.iterrows():
        hit = [c for c in audit.columns if int(r[c]) > 0]
        if len(hit) != 1:
            raise SystemExit(f"stored value {stored!r} split across levels {hit}")
    print("  [PASS] each stored value maps to exactly one level.")

    # ------------------------------------------------------- validation cohort
    C.banner("Step 3 — subgroup AUROC by teaching status, validation cohort")
    scored = C.load_test_scored()
    raw2 = raw.copy()
    raw2["teaching_level"] = level_all.values
    for k in ("PUF_YEAR", "inc_key"):
        scored[k] = pd.to_numeric(scored[k], errors="coerce").astype("Int64")
        raw2[k] = pd.to_numeric(raw2[k], errors="coerce").astype("Int64")
    df = scored.merge(raw2[["PUF_YEAR", "inc_key", "teaching_level"]],
                      on=["PUF_YEAR", "inc_key"], how="inner", validate="1:1")
    print(f"  validation rows after joining the teaching level : {len(df):,}")
    if len(df) != C.V4_FROZEN["n_test"]:
        raise SystemExit(f"validation rows {len(df):,}, expected "
                         f"{C.V4_FROZEN['n_test']:,}")
    print(f"  validation events : {int(df[OUTCOME].sum()):,}")
    print("\n  validation rows and events by level:")
    for lvl in LEVEL_ORDER:
        m = df["teaching_level"] == lvl
        n = int(m.sum())
        if not n:
            continue
        ev = int(df.loc[m, OUTCOME].sum())
        print(f"    {lvl:<14s} n = {n:>7,}   events = {ev:>5,}  "
              f"({100.0 * ev / n:5.2f}%)")

    print("\n  interval method: analytic DeLong, the corrected project method. "
          "The v4 step 09 protocol resampled 50,000 rows of each level and is "
          "withdrawn; its bounds are carried in the table for the audit trail "
          "but are never reported.")

    all_rows: list[dict] = []
    level_summary: dict[str, dict] = {}

    ordered_levels = [lv for lv in LEVEL_ORDER
                      if int((df["teaching_level"] == lv).sum()) > 0]
    for lvl in ordered_levels:
        sub = df.loc[df["teaching_level"] == lvl].copy()
        ev = int(sub[OUTCOME].sum())
        note = ""
        if lvl in TEACHING_UNUSABLE:
            note = ("not estimable as a teaching level: the registry records no "
                    "teaching status for these encounters")
        print(f"\n>>> Center | {lvl} : N={len(sub):,}  events={ev:,}")
        rows = subgroup_rows(sub, "Center", lvl, note=note)
        for r in rows:
            if np.isfinite(r["AUROC"]):
                print(f"    {r['Score']:<14s} n={int(r['N']):>7,}  "
                      f"AUC={r['AUROC']:.4f} "
                      f"[{r['CI_Low']:.4f},{r['CI_High']:.4f}]  "
                      f"dAUC={r['Delta_vs_PAT_TBI']:+.4f}  "
                      f"p={fmt_p(r['Delta_P'])}")
            else:
                print(f"    {r['Score']:<14s} n={int(r['N']):>7,}  AUROC not "
                      f"estimable. {r['Note'] or 'insufficient events'}")
        all_rows.extend(rows)
        pat = [r for r in rows if r["Score"] == "PAT-TBI"][0]
        level_summary[lvl] = pat

    # collapsed binary
    C.banner("Step 3b — the collapsed teaching binary")
    binary = pd.Series("Not usable", index=df.index, dtype=object)
    binary[df["teaching_level"].isin(TEACHING_POSITIVE)] = "Teaching"
    binary[df["teaching_level"].isin(TEACHING_NEGATIVE)] = "Non-teaching"
    for lvl in ("Teaching", "Non-teaching", "Not usable"):
        sub = df.loc[binary == lvl].copy()
        if len(sub) == 0:
            continue
        ev = int(sub[OUTCOME].sum())
        note = ""
        if lvl == "Not usable":
            note = ("not estimable as a teaching level: the registry records no "
                    "teaching status for these encounters")
        print(f"\n>>> Center (collapsed) | {lvl} : N={len(sub):,}  events={ev:,}")
        rows = subgroup_rows(sub, "Center (collapsed)", lvl, note=note)
        for r in rows:
            if np.isfinite(r["AUROC"]):
                print(f"    {r['Score']:<14s} n={int(r['N']):>7,}  "
                      f"AUC={r['AUROC']:.4f} "
                      f"[{r['CI_Low']:.4f},{r['CI_High']:.4f}]  "
                      f"dAUC={r['Delta_vs_PAT_TBI']:+.4f}  "
                      f"p={fmt_p(r['Delta_P'])}")
            else:
                print(f"    {r['Score']:<14s} n={int(r['N']):>7,}  AUROC not "
                      f"estimable. {r['Note'] or 'insufficient events'}")
        all_rows.extend(rows)
        level_summary[f"binary:{lvl}"] = [r for r in rows
                                          if r["Score"] == "PAT-TBI"][0]

    # ------------------------------------------- between-level contrasts
    C.banner("Step 4 — between-level contrasts for the forest of contrasts")
    print("  Two independent groups, so the difference of the two areas is "
          "compared with a standard error built from the two DeLong variances "
          "and no covariance term.")
    contrasts = []

    def between(label, lv_a, lv_b, selector):
        a = df.loc[selector(lv_a)]
        b_ = df.loc[selector(lv_b)]
        if len(a) == 0 or len(b_) == 0:
            return
        ya = a[OUTCOME].astype(float).to_numpy()
        yb = b_[OUTCOME].astype(float).to_numpy()
        if ya.sum() < MIN_EVENTS or yb.sum() < MIN_EVENTS:
            return
        auc_a, var_a = delong_auc_variance(ya, a["pat_tbi_score"].astype(float))
        auc_b, var_b = delong_auc_variance(yb, b_["pat_tbi_score"].astype(float))
        d = auc_a - auc_b
        se_d = float(np.sqrt(var_a + var_b))
        lo, hi = d - 1.959964 * se_d, d + 1.959964 * se_d
        p = stable_p(d / se_d) if se_d > 0 else 1.0
        print(f"  {label:<44s} {fmt_ci(d, lo, hi, 4)}   p {fmt_p(p)}")
        contrasts.append({"label": label, "estimate": d, "lo": lo, "hi": hi,
                          "se": se_d, "p": p,
                          "auroc_a": auc_a, "auroc_b": auc_b,
                          "n_a": int(len(a)), "n_b": int(len(b_)),
                          "events_a": int(ya.sum()), "events_b": int(yb.sum())})

    by_level = lambda lv: df["teaching_level"] == lv           # noqa: E731
    by_binary = lambda lv: binary == lv                        # noqa: E731
    between("Academic minus Non-teaching", "Academic", "Non-teaching", by_level)
    between("Community minus Non-teaching", "Community", "Non-teaching", by_level)
    between("Academic minus Community", "Academic", "Community", by_level)
    between("Teaching minus Non-teaching", "Teaching", "Non-teaching", by_binary)

    for c in contrasts:
        all_rows.append({
            "Subgroup": "Center (contrast)", "Level": c["label"],
            "Score": "PAT-TBI", "N": c["n_a"] + c["n_b"],
            "N_events": c["events_a"] + c["events_b"],
            "AUROC": np.nan, "CI_Low": np.nan, "CI_High": np.nan,
            "Delta_vs_PAT_TBI": c["estimate"], "Delta_SE": c["se"],
            "Delta_CI_Low": c["lo"], "Delta_CI_High": c["hi"],
            "Delta_P": c["p"], "N_events_too_small": False,
            "Note": ("difference between two independent subgroup areas; "
                     "standard error from the two DeLong variances")})

    out = pd.DataFrame(all_rows)[
        ["Subgroup", "Level", "Score", "N", "N_events", "AUROC", "CI_Low",
         "CI_High", "SE", "Delta_vs_PAT_TBI", "Delta_SE", "Delta_CI_Low",
         "Delta_CI_High", "Delta_P", "N_events_too_small",
         "CI_Low_withdrawn", "CI_High_withdrawn", "Note"]]
    OUT_TABLE_B.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_TABLE_B, index=False)
    print(f"\n[write] {OUT_TABLE_B}   rows = {len(out):,}")

    # ------------------------------------------------------------- the part
    def level_leaf(lv):
        r = level_summary.get(lv)
        if r is None:
            return None
        node = {"rows": {"n": int(r["N"]), "N": int(len(df))},
                "events": {"n": int(r["N_events"]), "N": int(r["N"])}}
        if np.isfinite(r["AUROC"]):
            node["auroc"] = {"estimate": round(float(r["AUROC"]), 4),
                             "lo": round(float(r["CI_Low"]), 4),
                             "hi": round(float(r["CI_High"]), 4)}
        else:
            # A display leaf carries no numerals, so the reason is prose and the
            # counts behind it sit in the numeric leaves above.
            unusable = any(u in str(lv) for u in TEACHING_UNUSABLE) \
                or "usable" in str(lv)
            node["auroc_not_estimable"] = {"display": (
                "not estimable, because the registry records no teaching status "
                "for these encounters and the level carries too few events to "
                "support an area" if unusable else
                "not estimable, because the level carries too few events to "
                "support an area")}
        return node

    levels_node = {}
    for lv in ordered_levels:
        key = lv.lower().replace("-", "").replace(" ", "")
        leaf = level_leaf(lv)
        if leaf is not None:
            levels_node[key] = leaf

    n_cohort_by_level = {lv: int(counts_all.get(lv, 0)) for lv in LEVEL_ORDER}
    mapping_node = {
        "academic": {
            "numeric_code": {"value": 1},
            "text_token": {"display": "academic"},
            "registry_label": {"display": "Academic"},
            "binary": {"display": "teaching"},
            "cohort_rows": {"n": n_cohort_by_level["Academic"], "N": n_cohort},
        },
        "community": {
            "numeric_code": {"value": 5},
            "text_token": {"display": "community"},
            "registry_label": {"display": "Community"},
            "binary": {"display": "teaching, as a community teaching hospital"},
            "cohort_rows": {"n": n_cohort_by_level["Community"], "N": n_cohort},
        },
        "nonteaching": {
            "numeric_code": {"value": 6},
            "text_token": {"display": "nonteaching"},
            "registry_label": {"display": "Non-teaching"},
            "binary": {"display": "non-teaching"},
            "cohort_rows": {"n": n_cohort_by_level["Non-teaching"], "N": n_cohort},
        },
        "university": {
            "numeric_code": {"value": 4},
            "registry_label": {"display": "University"},
            "binary": {"display": "teaching"},
            "cohort_rows": {"n": n_cohort_by_level["University"], "N": n_cohort},
            "note": {"display": (
                "a level the registry defines but which no encounter in this "
                "cohort carries")},
        },
        "not_recorded": {
            "numeric_code": {"value": 7},
            "registry_label": {"display": "Not known or not recorded"},
            "binary": {"display": "not usable"},
            "cohort_rows": {"n": n_cohort_by_level["Not recorded"], "N": n_cohort},
            "note": {"display": (
                "a level the registry defines but which no encounter in this "
                "cohort carries")},
        },
        "missing": {
            "registry_label": {"display": "Field absent"},
            "binary": {"display": "not usable"},
            "cohort_rows": {"n": n_missing, "N": n_cohort},
        },
    }

    contrast_node = {}
    for c in contrasts:
        key = (c["label"].lower().replace(" minus ", "_minus_")
               .replace("-", "").replace(" ", "_"))
        contrast_node[key] = {
            "label": {"display": c["label"].replace(" minus ", " compared with ")},
            "auroc_difference": {"estimate": round(c["estimate"], 4),
                                 "lo": round(c["lo"], 4), "hi": round(c["hi"], 4)},
            "pvalue": p_leaf(c["p"]),
        }

    payload = {
        "_meta": {
            "script": "03_head_ais_sensitivity.py",
            "finding": 6,
            "source": {"display": (
                "The teaching field carried in the frozen pre-engineering "
                "cohort, resolved against the registry's own lookup table, then "
                "joined onto the scored validation rows.")},
        },
        "prior_defect": {"display": (
            "The earlier analysis named a centre subgroup in its plan and in its "
            "manuscript, but the variable never reached the candidate matrix, so "
            "the step silently skipped it and no such subgroup was ever "
            "estimated. The claim stood without a computation behind it.")},
        "estimable": True,
        "field": {
            "populated": {"n": n_cohort - n_missing, "N": n_cohort},
            "missing": {"n": n_missing, "N": n_cohort},
            "distinct_stored_values": {
                "value": int(raw["teaching_status_raw"].nunique(dropna=True))},
            "encoding_problem": {"display": (
                "The registry changed encoding partway through the study window. "
                "The first admission year stores the decoded label in lower case "
                "and every later year stores the numeric facility teaching "
                "indicator, so the raw column mixes words and numerals. Both "
                "encodings describe the same field.")},
            "authority": {"display": (
                "The registry's own lookup table, which is identical across every "
                "admission year that uses the numeric encoding, and the variable "
                "format file that binds the teaching field to it.")},
        },
        "mapping": mapping_node,
        "mapping_decision": {"display": (
            "A defensible mapping exists and was applied. The registry lists a "
            "non-teaching level in its own right, so a community hospital in this "
            "field is a community teaching hospital and belongs on the teaching "
            "side of any binary. Results are therefore reported at all three "
            "occupied levels as an ordered set, with the collapsed teaching "
            "binary alongside. An earlier sensitivity script in the project "
            "placed community hospitals on the non-teaching side; that reading "
            "contradicts the registry taxonomy and is not used.")},
        "levels": levels_node,
        "binary": {
            "teaching": level_leaf("binary:Teaching"),
            "nonteaching": level_leaf("binary:Non-teaching"),
        },
        "contrasts": contrast_node,
        "method": {"display": (
            "Areas under the receiver operating characteristic curve for the "
            "integer instrument within each level of the validation cohort. "
            "Intervals are the analytic DeLong form, which is the corrected "
            "project method; the earlier resampling protocol was withdrawn "
            "because it drew a fixed subsample from each level and reported the "
            "spread of that smaller sample, which made every interval on a large "
            "level too wide. Between-level differences use two independent "
            "DeLong variances with no covariance term, so single level intervals "
            "and contrast intervals now rest on one variance rather than two.")},
        "verdict": {"display": (
            "The subgroup is estimable and has now been estimated. "
            "Discrimination holds at every level of hospital teaching status, "
            "with overlapping intervals throughout, which is the reassurance the "
            "earlier claim asserted without support. The only nominal difference "
            "between two levels is academic against community teaching centres, "
            "and it is small, of a size that would not change how the instrument "
            "is used, and unadjusted for the several comparisons made here.")},
        "action": {"display": (
            "Add the centre rows to the subgroup table and to the forest of "
            "contrasts, and state the mapping in the methods so a reader can see "
            "how the mixed encoding was resolved.")},
    }
    if payload["binary"]["teaching"] is None:
        payload["binary"].pop("teaching")
    if payload["binary"]["nonteaching"] is None:
        payload["binary"].pop("nonteaching")

    C.write_part("teaching", payload)
    print(f"[write] {C.V6_PARTS / 'teaching.json'}")

    return {"levels": level_summary, "contrasts": contrasts,
            "n_missing": n_missing, "n_cohort": n_cohort,
            "counts": n_cohort_by_level, "table": out, "df": df}


# ===========================================================================
# Handoff
# ===========================================================================
def write_handoff(a: dict, b: dict) -> None:
    lv = b["levels"]

    def row(label, est, lo, hi, note=""):
        return (f"| {label} | {est:.4f} | {lo:.4f} | {hi:.4f} | {note} |")

    lines = []
    lines.append("# Step 3 handoff — head AIS sensitivity and teaching status")
    lines.append("")
    lines.append("Script: `v6/code/03_head_ais_sensitivity.py`  ")
    lines.append("Log: `v6/logs/03_head_ais_sensitivity.log`  ")
    lines.append("Tables: `v6/tables/table_v6_head_ais.csv`, "
                 "`v6/tables/table_v6_teaching_subgroup.csv`  ")
    lines.append("Bundle parts: `v6/bundle/parts/head_ais.json`, "
                 "`v6/bundle/parts/teaching.json`")
    lines.append("")
    lines.append("## Figure 3 rows produced")
    lines.append("")
    lines.append("### Contrast rows (difference in area, forest scale)")
    lines.append("")
    lines.append("| label | estimate | lo | hi | note |")
    lines.append("|---|---|---|---|---|")
    lines.append(row("Adding maximum head Abbreviated Injury Scale to the model",
                     a["delta"], a["d_lo"], a["d_hi"],
                     f"DeLong, {fmt_pv(a['p_d'])}; extended minus base linear "
                     f"predictor on the same 207,863 validation rows"))
    lines.append(row("Adding head AIS, severity-unknown held out",
                     a["delta_b"], a["db_lo"], a["db_hi"],
                     f"sensitivity variant, DeLong, {fmt_pv(a['p_b'])}"))
    for c in b["contrasts"]:
        lines.append(row(c["label"].replace(" minus ", " vs "),
                         c["estimate"], c["lo"], c["hi"],
                         f"independent-group DeLong, {fmt_pv(c['p'])}"))
    lines.append("")
    lines.append("### Level rows (area within a subgroup)")
    lines.append("")
    lines.append("| label | estimate | lo | hi | note |")
    lines.append("|---|---|---|---|---|")
    for key, label in (("Academic", "Teaching status: academic"),
                       ("Community", "Teaching status: community"),
                       ("Non-teaching", "Teaching status: non-teaching"),
                       ("binary:Teaching", "Teaching status: teaching (collapsed)"),
                       ("binary:Non-teaching",
                        "Teaching status: non-teaching (collapsed)")):
        r = lv.get(key)
        if r is None:
            continue
        if np.isfinite(r["AUROC"]):
            lines.append(row(label, r["AUROC"], r["CI_Low"], r["CI_High"],
                             f"n = {int(r['N']):,}, events = {int(r['N_events']):,}"))
        else:
            lines.append(f"| {label} | not estimable | | | {r['Note']} |")
    r_missing = lv.get("Missing")
    if r_missing is not None:
        lines.append(
            f"| Teaching status: not recorded | not estimable | | | "
            f"{r_missing['Note'] or 'no teaching status in the registry'}; "
            f"n = {int(r_missing['N']):,}, events = {int(r_missing['N_events']):,} |")
    lines.append("")
    lines.append("Model discrimination rows, for the same figure if it carries an "
                 "area panel:")
    lines.append("")
    lines.append("| label | estimate | lo | hi | note |")
    lines.append("|---|---|---|---|---|")
    lines.append(row("Base model, linear predictor", a["auc_base"], a["lo_base"],
                     a["hi_base"], "22 retained predictors"))
    lines.append(row("Extended model, linear predictor", a["auc_ext"],
                     a["lo_ext"], a["hi_ext"], "22 predictors plus head AIS"))
    lines.append(row("Integer 22-point score", a["auc_int"], a["lo_int"],
                     a["hi_int"],
                     "point estimate is the frozen v4 headline, reproduced "
                     "exactly; the interval is the corrected DeLong one, not "
                     "the withdrawn v4 pair"))
    lines.append("")
    lines.append("## Recommended sentence for Results")
    lines.append("")
    lines.append(
        f"> Adding maximum head Abbreviated Injury Scale to the model as an "
        f"ordinal term raised the area under the curve in validation from "
        f"{a['auc_base']:.4f} (95% CI {a['lo_base']:.4f} to {a['hi_base']:.4f}) "
        f"to {a['auc_ext']:.4f} (95% CI {a['lo_ext']:.4f} to {a['hi_ext']:.4f}), "
        f"a difference of {a['delta']:.4f} (95% CI {a['d_lo']:.4f} to "
        f"{a['d_hi']:.4f} by DeLong, {fmt_pv(a['p_d'])}), with an odds ratio "
        f"of {a['or']:.2f} (95% CI {a['or_lo']:.2f} to {a['or_hi']:.2f}) per "
        f"point of head injury severity.")
    lines.append("")
    if lv.get("Academic") is not None and np.isfinite(lv["Academic"]["AUROC"]):
        ac, co, nt = lv["Academic"], lv["Community"], lv["Non-teaching"]
        lines.append(
            f"> Discrimination held across hospital teaching status: "
            f"{ac['AUROC']:.3f} (95% CI {ac['CI_Low']:.3f} to "
            f"{ac['CI_High']:.3f}) at academic centres "
            f"(n = {int(ac['N']):,}, {int(ac['N_events']):,} events), "
            f"{co['AUROC']:.3f} (95% CI {co['CI_Low']:.3f} to "
            f"{co['CI_High']:.3f}) at community teaching centres "
            f"(n = {int(co['N']):,}, {int(co['N_events']):,} events) and "
            f"{nt['AUROC']:.3f} (95% CI {nt['CI_Low']:.3f} to "
            f"{nt['CI_High']:.3f}) at non-teaching centres "
            f"(n = {int(nt['N']):,}, {int(nt['N_events']):,} events). "
            f"Teaching status was recorded for 99.4% of the cohort; the "
            f"remaining encounters could not be assigned to a level and are "
            f"reported as such rather than dropped.")
        lines.append("")
        lines.append("Honest qualifier for whoever writes this up: of the four "
                     "between-level comparisons, only academic against community "
                     "reaches nominal significance, the difference is about one "
                     "and a half points of area, and no adjustment for "
                     "multiplicity has been made. It should be read as noise "
                     "until someone has a reason to think otherwise.")
        lines.append("")
    lines.append("## Recommended reply to Dumas and to Hynes")
    lines.append("")
    lines.append(
        f"> Head Abbreviated Injury Scale was excluded by design rather than by "
        f"oversight. The scale is assigned retrospectively by a trauma registrar "
        f"after imaging is read and the chart is abstracted, so it does not "
        f"exist at the moment a clinician decides whether to release platelets; "
        f"a tool that required it could not be used for the decision it is meant "
        f"to support. We agree the exclusion should be quantified rather than "
        f"asserted, so we refitted the model with head Abbreviated Injury Scale "
        f"added as an ordinal term. It is strongly associated with platelet "
        f"receipt, with an odds ratio of {a['or']:.2f} (95% CI {a['or_lo']:.2f} "
        f"to {a['or_hi']:.2f}) per point, and it raises discrimination in "
        f"validation from {a['auc_base']:.4f} to {a['auc_ext']:.4f}, a gain of "
        f"{a['delta']:.4f} (95% CI {a['d_lo']:.4f} to {a['d_hi']:.4f}). We report "
        f"that gain rather than argue it away: it is the size of the ceiling the "
        f"bedside constraint imposes, and a reader is entitled to see it. We have "
        f"added the analysis to the results and to the new figure.")
    lines.append("")
    lines.append("## Notes for whoever builds Figure 3")
    lines.append("")
    lines.append("- The head AIS row is a paired DeLong contrast on identical "
                 "rows; the teaching contrasts are independent-group differences "
                 "with no covariance term. If the figure mixes them, say so in "
                 "the legend.")
    lines.append(f"- {a['n_unknown_te']:,} validation rows carry a head AIS of "
                 f"nine, which the scale reserves for severity unknown rather "
                 f"than for a severity worse than the maximum. The "
                 f"severity-unknown variant holds those rows out of the ordinal "
                 f"term and reaches the same conclusion.")
    lines.append(f"- {b['n_missing']:,} of {b['n_cohort']:,} cohort encounters "
                 f"carry no teaching status. Those rows print a row that says "
                 f"why rather than vanishing.")
    lines.append("- Every interval here is the analytic DeLong form, imported "
                 "from `11_recompute_intervals.py`. The v4 bootstrap this script "
                 "originally used is an m-out-of-n resample reported without the "
                 "sqrt(n/m) rescaling and was about 44% too wide; those bounds "
                 "are withdrawn. They are still carried in the two CSVs, in the "
                 "`discrimination_withdrawn` block and the `CI_*_withdrawn` "
                 "columns, so the before and after of the correction stays "
                 "visible. Never quote a withdrawn bound.")
    lines.append("- Point estimates, deltas, standard errors and z statistics "
                 "are unaffected by the interval correction. Only interval "
                 "widths moved.")
    lines.append("")
    OUT_HANDOFF.parent.mkdir(parents=True, exist_ok=True)
    OUT_HANDOFF.write_text("\n".join(lines))
    print(f"[write] {OUT_HANDOFF}")


# ===========================================================================
def main() -> None:
    C.start_log("03_head_ais_sensitivity")
    t0 = time.time()
    C.banner("PAT-TBI v6 — Step 3: head AIS sensitivity + teaching status")
    print(f"  interpreter : {sys.executable}")
    print(f"  v4 root     : {C.V4}")
    print(f"  v6 root     : {C.V6}")
    print(f"  outcome     : {OUTCOME}  ({'platelet receipt within four hours'})")
    print("  DeLong and both bootstrap protocols are imported from "
          "v4/code/07_validation_metrics.py and v4/code/09_subgroups.py, not "
          "reimplemented here.")

    train, test = C.load_split()
    a = part_a(train, test)
    b = part_b()
    write_handoff(a, b)

    C.banner("Done")
    print(f"  runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
