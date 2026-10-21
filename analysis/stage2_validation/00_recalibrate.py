"""
PAT-TBI v6 — Phase 0, step 00: honest recalibration (findings #1, #3, #10)
=========================================================================

WHAT WAS WRONG (finding #1, circular calibration)
-------------------------------------------------
v4 `07_validation_metrics.py` Part C fitted the score-to-probability map

    logit(y) = a + b * score

on the VALIDATION rows and then reported the calibration of exactly those
fitted predictions on those same rows. A logistic fit is, by construction,
mean-unbiased on its own fitting sample and has calibration slope one when the
slope is re-estimated on its own linear predictor. The published
calibration-in-the-large of 1e-06 and slope of 1.0000 are therefore arithmetic
identities of the fitting procedure, not empirical results. The Hosmer and
Lemeshow degrees of freedom in that table (6, 1 and 2 for the three raw scores)
also collapsed silently, because `pd.qcut(..., duplicates="drop")` on a tied
integer score returns far fewer than ten groups (finding #10).

THREE INCOMPATIBLE CONVENTIONS (finding #3)
-------------------------------------------
  step 07 : logistic map fitted on the VALIDATION rows for the raw scores only
            (PAT-TBI, Glasgow Coma Scale total, Glasgow Coma Scale pupil);
            IMPACT-Core and CRASH-Basic passed through on their native
            published probabilities.
  step 08 : logistic map fitted on the VALIDATION rows for EVERY score,
            including IMPACT-Core and CRASH-Basic, so the two published
            instruments were silently re-estimated before their calibration
            curve, their decision curve and their Brier score were drawn.
  step 10 : isotonic regression fitted on the VALIDATION rows for the raw
            scores (a non-parametric monotone map used only to feed the
            decision curve); IMPACT-Core and CRASH-Basic passed through.
Three different maps, all fitted in the evaluation sample, all reported as if
they were one convention.

THE SINGLE CONVENTION FOR THE WHOLE PROJECT
-------------------------------------------
From here on, for every raw-score instrument (PAT-TBI, Glasgow Coma Scale
total, Glasgow Coma Scale pupil), the map

    logit(y) = a + b * score

is fitted ONCE, on the 415,724 DERIVATION rows, and applied unchanged to the
207,863 VALIDATION rows. This is the only recalibration convention used
anywhere in v6: every calibration statistic, calibration plot, decision curve
and risk band in the project must use these frozen coefficients. Nothing is
refitted in validation.

The two published instruments that already emit probabilities (IMPACT-Core,
CRASH-Basic) are left on their native published probabilities and are NOT
remapped. This asymmetry is deliberate: refitting a map for them would measure
something other than what a reader gets by applying the published equations.

WHAT IS REPORTED, ON THE VALIDATION ROWS ONLY
---------------------------------------------
  calibration-in-the-large : intercept alpha of  logit(y) = alpha + offset(lp),
                             lp = the derivation-fitted linear predictor.
                             The plain difference mean(p) - mean(y) is reported
                             separately under its own label.
  calibration slope        : beta of  logit(y) = alpha' + beta * lp.
  Brier score              : mean squared difference of probability and event.
  grouped calibration      : deciles of estimated risk via pd.qcut with
                             duplicates="drop"; the ACHIEVED group count is
                             reported, never asserted to be ten, because the
                             integer score is heavily tied.
  Hosmer and Lemeshow      : relegated. Reported with its achieved degrees of
                             freedom and a note that at this sample size the
                             test rejects regardless.

Run with the framework interpreter, not bare `python3`:
    PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
    cd analysis/stage2_validation
    $PY 00_recalibrate.py

Outputs
-------
  v6/bundle/parts/calibration.json
  v6/tables/table_v6_calibration.csv
  v6/tables/table_v6_calibration_deciles.csv
  v6/logs/00_recalibrate.log
  v6/logs/00_HANDOFF.md
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sstats
from sklearn.metrics import brier_score_loss, roc_auc_score

import _common as C

warnings.filterwarnings("ignore")

EPS = 1e-12
N_GROUPS_REQUESTED = 10

OUT_CALIB = C.V6 / "tables" / "table_v6_calibration.csv"
OUT_DECILES = C.V6 / "tables" / "table_v6_calibration_deciles.csv"
OUT_HANDOFF = C.V6 / "logs" / "00_HANDOFF.md"

# Pupil reactivity score, copied verbatim from v4/code/06_score_test_set.py
GCS_P_PRS = {"both": 0, "one": 1, "none": 2}

CONV_DERIVED = "derivation-fitted logistic map, logit(y) = a + b * score"
CONV_NATIVE = "native published probability, not remapped"


# ---------------------------------------------------------------------------
# Comparator recomputation, verbatim from v4/code/06_score_test_set.py
# ---------------------------------------------------------------------------
def _as_pupil(series):
    """Normalise TQIP_TBI_PUPIL_RESPONSE to {'both','one','none', NA}.

    Verbatim from v4/code/06_score_test_set.py: NTDB writes 'NEITHER' where
    the published instruments write 'NONE'; both encodings are accepted.
    """
    s = series.astype(str).str.strip().str.upper()
    mapping = {"BOTH": "both", "ONE": "one", "NEITHER": "none", "NONE": "none"}
    out = s.map(mapping)
    out[~s.isin(mapping.keys())] = pd.NA
    return out


def compute_gcs_risk(df: pd.DataFrame) -> np.ndarray:
    """gcs_risk = 15 - GCS total, with missing total filled to 15."""
    gcs_raw = pd.to_numeric(df.get("GCS2"), errors="coerce")
    gcs_filled = gcs_raw.fillna(15).values
    return 15.0 - gcs_filled


def compute_gcs_p_risk(df: pd.DataFrame) -> np.ndarray:
    """gcs_p_risk = 15 - (GCS total - pupil reactivity score).

    Missing total filled to 15; missing pupil response filled to 'both'.
    """
    gcs_raw = pd.to_numeric(df.get("GCS2"), errors="coerce")
    gcs_filled = gcs_raw.fillna(15).values
    pupil_filled = _as_pupil(df.get("TQIP_TBI_PUPIL_RESPONSE")).fillna("both").values
    prs = np.array([GCS_P_PRS[p] for p in pupil_filled], dtype=np.float64)
    gcs_p = gcs_filled - prs
    return 15.0 - gcs_p


# ---------------------------------------------------------------------------
# Calibration primitives
# ---------------------------------------------------------------------------
def safe_logit(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def fit_map_on_derivation(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """Fit logit(y) = a + b * score with statsmodels Logit. Returns (a, b)."""
    X = sm.add_constant(np.asarray(score, dtype=np.float64), has_constant="add")
    res = sm.Logit(np.asarray(y, dtype=int), X).fit(disp=False, method="newton",
                                                    maxiter=200)
    return float(res.params[0]), float(res.params[1])


def citl_offset(y: np.ndarray, lp: np.ndarray) -> float:
    """Calibration-in-the-large: alpha from logit(y) = alpha + offset(lp)."""
    const = np.ones((len(y), 1), dtype=np.float64)
    glm = sm.GLM(np.asarray(y, dtype=float), const,
                 family=sm.families.Binomial(),
                 offset=np.asarray(lp, dtype=np.float64)).fit()
    return float(glm.params[0])


def calibration_slope(y: np.ndarray, lp: np.ndarray) -> float:
    """Calibration slope: beta from logit(y) = alpha' + beta * lp."""
    X = sm.add_constant(np.asarray(lp, dtype=np.float64), has_constant="add")
    res = sm.Logit(np.asarray(y, dtype=int), X).fit(disp=False, method="newton",
                                                    maxiter=200)
    return float(res.params[1])


def grouped_calibration(y: np.ndarray, prob: np.ndarray,
                        n_groups: int = N_GROUPS_REQUESTED) -> pd.DataFrame:
    """Observed versus estimated risk by group of estimated risk.

    The integer score is heavily tied, so pd.qcut with duplicates="drop" will
    generally return fewer than n_groups. The achieved count is what is
    reported; nothing here asserts ten.
    """
    frame = pd.DataFrame({"y": np.asarray(y, dtype=int),
                          "p": np.asarray(prob, dtype=np.float64)})
    frame["g"] = pd.qcut(frame["p"], n_groups, duplicates="drop")
    agg = (frame.groupby("g", observed=True)
                .agg(n=("y", "size"), obs_events=("y", "sum"),
                     mean_pred=("p", "mean"))
                .reset_index())
    agg = agg.sort_values("mean_pred").reset_index(drop=True)
    agg["group"] = np.arange(1, len(agg) + 1)
    agg["obs_rate"] = agg["obs_events"] / agg["n"]
    lo, hi = zip(*[C.wilson_ci(int(k), int(n))
                   for k, n in zip(agg["obs_events"], agg["n"])])
    agg["obs_rate_lo"] = lo
    agg["obs_rate_hi"] = hi
    agg["exp_events"] = agg["mean_pred"] * agg["n"]
    return agg[["group", "n", "mean_pred", "exp_events", "obs_events",
                "obs_rate", "obs_rate_lo", "obs_rate_hi"]]


def hosmer_lemeshow(agg: pd.DataFrame) -> tuple[float, int, float]:
    """Hosmer and Lemeshow statistic from an already-built group table.

    Relegated evidence. Groups with no expected events or no expected
    non-events are dropped, so the achieved degrees of freedom are returned
    rather than assumed to be eight.
    """
    a = agg.copy()
    a["obs_non"] = a["n"] - a["obs_events"]
    a["exp_non"] = a["n"] - a["exp_events"]
    a = a[(a["exp_events"] > 0) & (a["exp_non"] > 0)]
    stat = float(((a["obs_events"] - a["exp_events"]) ** 2 / a["exp_events"]).sum()
                 + ((a["obs_non"] - a["exp_non"]) ** 2 / a["exp_non"]).sum())
    df = int(max(len(a) - 2, 1))
    # sf, not 1 - cdf: the v4 form underflowed to exactly zero and printed P = 0.
    return stat, df, float(sstats.chi2.sf(stat, df=df))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    C.start_log("00_recalibrate")

    C.banner("PAT-TBI v6 step 00 — honest recalibration (findings 1, 3, 10)")
    print("Single convention: the score-to-probability map is fitted ONCE on the")
    print("derivation rows and applied unchanged to the validation rows. Nothing")
    print("is refitted in validation. The two published probability instruments")
    print("(IMPACT-Core, CRASH-Basic) keep their native probabilities.")

    # -- load -------------------------------------------------------------
    C.banner("Loading the frozen v4 split")
    train, test = C.load_split()
    y_tr = train[C.OUTCOME].to_numpy(dtype=int)
    y_te = test[C.OUTCOME].to_numpy(dtype=int)
    print(f"  derivation  n = {len(train):,}   events = {y_tr.sum():,} "
          f"({100 * y_tr.mean():.2f}%)")
    print(f"  validation  n = {len(test):,}   events = {y_te.sum():,} "
          f"({100 * y_te.mean():.2f}%)")

    # -- discrimination must not move -------------------------------------
    auroc = float(roc_auc_score(y_te, test["pat_tbi_score"].to_numpy()))
    print(f"\n  validation AUROC of the integer score = {auroc:.6f} "
          f"(rounds to {round(auroc, 4)})")
    assert round(auroc, 4) == C.V4_FROZEN["auroc_test"], (
        f"discrimination moved: AUROC {auroc:.6f} does not round to "
        f"{C.V4_FROZEN['auroc_test']}")
    print("  [PASS] discrimination unchanged from v4")

    # -- recompute the two raw comparators on BOTH splits -----------------
    C.banner("Recomputing the raw-score comparators on the derivation rows")
    print("  The v4 scored parquet carries gcs_risk and gcs_p_risk for the")
    print("  validation rows only, so both are recomputed here exactly as")
    print("  v4/code/06_score_test_set.py computes them, then verified against")
    print("  the v4 columns on the validation rows.")
    for frame in (train, test):
        frame["gcs_risk_v6"] = compute_gcs_risk(frame)
        frame["gcs_p_risk_v6"] = compute_gcs_p_risk(frame)

    scored = C.load_test_scored()
    scored = scored.assign(_key=C._key_series(scored))
    chk = test[["_key", "gcs_risk_v6", "gcs_p_risk_v6", "pat_tbi_score"]].merge(
        scored[["_key", "gcs_risk", "gcs_p_risk", "gcs_mask", "gcs_p_mask",
                "impact_core_prob", "impact_mask",
                "crash_basic_prob", "crash_mask", "pat_tbi_score"]],
        on="_key", suffixes=("_v6", "_v4"), validate="one_to_one")
    assert len(chk) == len(test), f"merge lost rows: {len(chk)} of {len(test)}"

    n_bad_gcs = int((chk["gcs_risk_v6"].to_numpy() != chk["gcs_risk"].to_numpy()).sum())
    n_bad_gcsp = int((chk["gcs_p_risk_v6"].to_numpy() != chk["gcs_p_risk"].to_numpy()).sum())
    print(f"\n  Glasgow Coma Scale total risk, rows disagreeing with v4 : {n_bad_gcs}")
    print(f"  Glasgow Coma Scale pupil risk, rows disagreeing with v4 : {n_bad_gcsp}")
    assert n_bad_gcs == 0, "recomputed gcs_risk does not match the v4 column"
    assert n_bad_gcsp == 0, "recomputed gcs_p_risk does not match the v4 column"
    print("  [PASS] both comparators reproduce the v4 columns exactly")

    n_bad_pat = int((chk["pat_tbi_score_v6"].to_numpy()
                     != chk["pat_tbi_score_v4"].to_numpy()).sum())
    assert n_bad_pat == 0, "the integer score does not match the v4 column"
    print(f"  [PASS] integer score, rows disagreeing with v4 : {n_bad_pat}")

    # v4 built gcs_mask and gcs_p_mask as all-True; record it rather than assume.
    print(f"\n  Glasgow Coma Scale total mask true in {int(chk['gcs_mask'].sum()):,} "
          f"of {len(chk):,}")
    print(f"  Glasgow Coma Scale pupil mask true in {int(chk['gcs_p_mask'].sum()):,} "
          f"of {len(chk):,}")

    # -----------------------------------------------------------------------
    # Raw-score instruments: fit the map in derivation, evaluate in validation
    # -----------------------------------------------------------------------
    RAW = [
        ("PAT-TBI", "pat_tbi_score", "pat_tbi_score"),
        ("GCS", "gcs_risk_v6", "gcs_risk"),
        ("GCS-P", "gcs_p_risk_v6", "gcs_p_risk"),
    ]

    rows: list[dict] = []
    decile_blocks: list[pd.DataFrame] = []
    detail: dict[str, dict] = {}

    for disp, col, _v4col in RAW:
        C.banner(f"{disp} — map fitted on derivation, evaluated on validation")

        s_tr = train[col].to_numpy(dtype=np.float64)
        s_te = test[col].to_numpy(dtype=np.float64)
        assert not np.isnan(s_tr).any() and not np.isnan(s_te).any()

        a, b = fit_map_on_derivation(y_tr, s_tr)
        print(f"  derivation-fitted map:  a = {a:.6f}   b = {b:.6f}")
        print(f"  fitted on n = {len(y_tr):,} derivation rows, "
              f"{y_tr.sum():,} events")

        # What v4 did, shown for contrast: the same fit re-done in validation.
        a_te, b_te = fit_map_on_derivation(y_te, s_te)
        print(f"  (the v4 step 07 convention would instead fit "
              f"a = {a_te:.6f}, b = {b_te:.6f} on the validation rows)")

        lp_te = a + b * s_te
        p_te = sigmoid(lp_te)

        citl = citl_offset(y_te, lp_te)
        citl_plain = float(p_te.mean() - y_te.mean())
        slope = calibration_slope(y_te, lp_te)
        brier = float(brier_score_loss(y_te, p_te))

        agg = grouped_calibration(y_te, p_te)
        k = len(agg)
        hl_chi, hl_df, hl_p = hosmer_lemeshow(agg)

        # In-derivation identity, printed so the circularity is visible.
        lp_tr = a + b * s_tr
        citl_tr = citl_offset(y_tr, lp_tr)
        slope_tr = calibration_slope(y_tr, lp_tr)

        print(f"\n  VALIDATION, derivation-fitted map applied unchanged:")
        print(f"    calibration in the large (offset intercept) = {citl:+.6f}")
        print(f"    mean estimated minus mean observed          = {citl_plain:+.6f}")
        print(f"    calibration slope                           = {slope:.4f}")
        print(f"    Brier score                                 = {brier:.6f}")
        print(f"    groups achieved by pd.qcut(duplicates=drop) = {k} "
              f"of {N_GROUPS_REQUESTED} requested")
        print(f"  DERIVATION, same map on its own fitting sample "
              f"(the v4 identity):")
        print(f"    calibration in the large = {citl_tr:+.9f}  "
              f"slope = {slope_tr:.6f}")
        print(f"  Hosmer and Lemeshow (relegated): chi-square = {hl_chi:.2f}, "
              f"df = {hl_df}, P = {hl_p:.3g}")

        if disp == "PAT-TBI":
            C.banner("HARD GATE — the circularity must be gone")
            print(f"  calibration slope            = {slope:.4f}   "
                  f"(fails if exactly 1.0000)")
            print(f"  calibration in the large     = {citl:.6f}   "
                  f"(fails if exactly 0.000000)")
            if round(slope, 4) == 1.0:
                raise SystemExit(
                    "GATE FAILED: PAT-TBI calibration slope is exactly 1.0000. "
                    "The circularity survived. Stopping.")
            if round(citl, 6) == 0.0:
                raise SystemExit(
                    "GATE FAILED: PAT-TBI calibration-in-the-large is exactly "
                    "zero. The circularity survived. Stopping.")
            print("  [PASS] neither statistic is at the fitting identity")

            # The map is monotone increasing, so discrimination is untouched.
            auroc_p = float(roc_auc_score(y_te, p_te))
            assert round(auroc_p, 4) == C.V4_FROZEN["auroc_test"], (
                "the map changed discrimination")
            print(f"  [PASS] AUROC on the mapped probability = {auroc_p:.6f}, "
                  f"unchanged")

        print("\n  observed versus estimated risk by group of estimated risk:")
        print(f"  {'grp':>3} {'n':>9} {'mean est':>10} {'events':>8} "
              f"{'observed':>10} {'Wilson low':>11} {'Wilson high':>12}")
        for _, r in agg.iterrows():
            print(f"  {int(r['group']):>3} {int(r['n']):>9,} "
                  f"{r['mean_pred']:>10.5f} {int(r['obs_events']):>8,} "
                  f"{r['obs_rate']:>10.5f} {r['obs_rate_lo']:>11.5f} "
                  f"{r['obs_rate_hi']:>12.5f}")

        blk = agg.copy()
        blk.insert(0, "Score", disp)
        decile_blocks.append(blk)

        rows.append({
            "Score": disp,
            "N": int(len(y_te)),
            "Events": int(y_te.sum()),
            "Convention": CONV_DERIVED,
            "Map_Intercept_a": round(a, 6),
            "Map_Slope_b": round(b, 6),
            "CITL_offset": round(citl, 6),
            "CITL_mean_difference": round(citl_plain, 6),
            "Calibration_Slope": round(slope, 4),
            "Brier": round(brier, 6),
            "Groups_Achieved": int(k),
            "HL_ChiSq": round(hl_chi, 3),
            "HL_df": int(hl_df),
            "HL_P": hl_p,
        })
        detail[disp] = {
            "a": a, "b": b, "citl": citl, "citl_plain": citl_plain,
            "slope": slope, "brier": brier, "k": k,
            "hl_chi": hl_chi, "hl_df": hl_df, "hl_p": hl_p,
            "n": int(len(y_te)), "events": int(y_te.sum()),
        }

    # -----------------------------------------------------------------------
    # Published probability instruments: native scale, own denominators
    # -----------------------------------------------------------------------
    NATIVE = [
        ("IMPACT-Core", "impact_core_prob", "impact_mask"),
        ("CRASH-Basic", "crash_basic_prob", "crash_mask"),
    ]

    y_scored = scored[C.OUTCOME].to_numpy(dtype=int)
    for disp, col, mask_col in NATIVE:
        C.banner(f"{disp} — native published probability, deliberately not remapped")
        m = scored[mask_col].to_numpy(dtype=bool)
        s = scored[col].to_numpy(dtype=np.float64)
        use = m & ~np.isnan(s)
        y_u = y_scored[use]
        p_u = np.clip(s[use], 1e-9, 1 - 1e-9)
        print(f"  own denominator: n = {use.sum():,} of {len(scored):,} "
              f"validation rows; events = {y_u.sum():,}")
        print("  No map is fitted for this instrument. Refitting one would")
        print("  measure something other than what a reader obtains by applying")
        print("  the published equation, so the native probability stands.")

        lp_u = safe_logit(p_u)
        citl = citl_offset(y_u, lp_u)
        citl_plain = float(p_u.mean() - y_u.mean())
        slope = calibration_slope(y_u, lp_u)
        brier = float(brier_score_loss(y_u, p_u))
        agg = grouped_calibration(y_u, p_u)
        k = len(agg)
        hl_chi, hl_df, hl_p = hosmer_lemeshow(agg)

        print(f"\n    calibration in the large (offset intercept) = {citl:+.6f}")
        print(f"    mean estimated minus mean observed          = {citl_plain:+.6f}")
        print(f"    calibration slope                           = {slope:.4f}")
        print(f"    Brier score                                 = {brier:.6f}")
        print(f"    groups achieved                             = {k} "
              f"of {N_GROUPS_REQUESTED} requested")
        print(f"  Hosmer and Lemeshow (relegated): chi-square = {hl_chi:.2f}, "
              f"df = {hl_df}, P = {hl_p:.3g}")

        blk = agg.copy()
        blk.insert(0, "Score", disp)
        decile_blocks.append(blk)

        rows.append({
            "Score": disp,
            "N": int(use.sum()),
            "Events": int(y_u.sum()),
            "Convention": CONV_NATIVE,
            "Map_Intercept_a": "",
            "Map_Slope_b": "",
            "CITL_offset": round(citl, 6),
            "CITL_mean_difference": round(citl_plain, 6),
            "Calibration_Slope": round(slope, 4),
            "Brier": round(brier, 6),
            "Groups_Achieved": int(k),
            "HL_ChiSq": round(hl_chi, 3),
            "HL_df": int(hl_df),
            "HL_P": hl_p,
        })
        detail[disp] = {
            "a": None, "b": None, "citl": citl, "citl_plain": citl_plain,
            "slope": slope, "brier": brier, "k": k,
            "hl_chi": hl_chi, "hl_df": hl_df, "hl_p": hl_p,
            "n": int(use.sum()), "events": int(y_u.sum()),
        }

    # -----------------------------------------------------------------------
    # Tables
    # -----------------------------------------------------------------------
    C.banner("Writing tables")
    calib_df = pd.DataFrame(rows, columns=[
        "Score", "N", "Events", "Convention", "Map_Intercept_a", "Map_Slope_b",
        "CITL_offset", "CITL_mean_difference", "Calibration_Slope", "Brier",
        "Groups_Achieved", "HL_ChiSq", "HL_df", "HL_P"])
    OUT_CALIB.parent.mkdir(parents=True, exist_ok=True)
    calib_df.to_csv(OUT_CALIB, index=False)
    print(f"  wrote {OUT_CALIB}")

    dec_df = pd.concat(decile_blocks, ignore_index=True)
    dec_df = dec_df.rename(columns={
        "group": "Group", "n": "N", "mean_pred": "Mean_Predicted_Risk",
        "exp_events": "Expected_Events", "obs_events": "Observed_Events",
        "obs_rate": "Observed_Rate", "obs_rate_lo": "Observed_Rate_Wilson_Low",
        "obs_rate_hi": "Observed_Rate_Wilson_High"})
    dec_df.to_csv(OUT_DECILES, index=False)
    print(f"  wrote {OUT_DECILES}")

    with pd.option_context("display.width", 200, "display.max_columns", None):
        print("\n" + calib_df.drop(columns=["Convention"]).to_string(index=False))

    # -----------------------------------------------------------------------
    # Bundle part
    # -----------------------------------------------------------------------
    C.banner("Writing the bundle part")
    pat = detail["PAT-TBI"]

    def block(d: dict, native: bool) -> dict:
        out = {
            "n": {"n": d["n"]},
            "events": {"n": d["events"], "N": d["n"]},
            "citl": {"value": round(d["citl"], 4)},
            "citl_mean_difference": {"value": round(d["citl_plain"], 6)},
            "slope": {"value": round(d["slope"], 4)},
            "brier": {"value": round(d["brier"], 5)},
            "groups_achieved": {"value": d["k"]},
            "hl_chisq": {"value": round(d["hl_chi"], 1)},
            "hl_df": {"value": d["hl_df"]},
            # A chi-square this large underflows to exactly zero in double
            # precision. A p-value leaf of zero would render as "P = 0", which
            # is not a number anyone should print, so it is floored at a value
            # that is unambiguously below every conventional threshold.
            "hl_p": {"pvalue": max(float(d["hl_p"]), 1e-300)},
        }
        if not native:
            out["map_intercept"] = {"value": round(d["a"], 4)}
            out["map_slope"] = {"value": round(d["b"], 4)}
        return out

    payload = {
        "convention": {
            "statement": {"display":
                "The map from the integer score to an estimated probability was "
                "fitted once on the derivation rows and then applied unchanged to "
                "the validation rows. This single map is the recalibration "
                "convention for the whole project; every calibration statistic, "
                "calibration plot and decision curve uses it, and nothing is "
                "refitted in the validation cohort."},
            "equation": {"display":
                "The logit of the observed event is modelled as an intercept plus "
                "a coefficient times the score."},
            "intercept": {"value": round(pat["a"], 4)},
            "slope": {"value": round(pat["b"], 4)},
            "fitted_on": {"n": C.V4_FROZEN["n_train"]},
            "fitted_events": {"n": C.V4_FROZEN["n_train_events"],
                              "N": C.V4_FROZEN["n_train"]},
            "evaluated_on": {"n": C.V4_FROZEN["n_test"]},
            "evaluated_events": {"n": C.V4_FROZEN["n_test_events"],
                                 "N": C.V4_FROZEN["n_test"]},
            "prior_defect": {"display":
                "The earlier analysis fitted the same map inside the validation "
                "cohort and then reported the calibration of those same fitted "
                "values, so the reported intercept and slope were identities of "
                "the fitting procedure rather than findings. Three separate "
                "steps also used three different maps, one logistic on the raw "
                "scores only, one logistic on every score including the two "
                "published instruments, and one isotonic. All of that is "
                "replaced by the single derivation-fitted map described above."},
            "published_instruments": {"display":
                "IMPACT-Core and CRASH-Basic are deliberately left on their own "
                "published probabilities and are not remapped. Both already emit "
                "a probability, so fitting a map for them would measure "
                "something other than what a reader obtains by applying the "
                "published equations. Their calibration is reported on the "
                "native scale and on their own denominators, which are smaller "
                "than the full validation cohort, since each instrument "
                "requires covariates that are not recorded for every patient."},
        },
        "grouping": {
            "requested_groups": {"value": N_GROUPS_REQUESTED},
            "achieved_groups": {"value": pat["k"]},
            "note": {"display":
                "Groups of estimated risk were cut at quantiles with duplicate "
                "edges dropped. The score is an integer with heavy ties, so "
                "fewer groups are achieved than were requested, and the "
                "achieved count is what is reported."},
        },
        "hosmer_lemeshow": {
            "relegation": {"display":
                "At a validation cohort of this size, and with the heavy ties an "
                "integer score produces, the Hosmer and Lemeshow test rejects "
                "almost regardless of how closely estimated and observed risks "
                "agree, and its degrees of freedom fall well below the "
                "conventional value once tied groups are merged. It is reported "
                "for completeness only. The Brier score and the grouped observed "
                "versus estimated table are the primary evidence on calibration."},
            "underflow": {"display":
                "The chi-square values are large enough that the tail "
                "probability underflows to zero in double precision "
                "arithmetic. Each of these probabilities lies far below any "
                "conventional threshold and should be reported as such rather "
                "than as a bare zero."},
        },
        "pat_tbi": block(pat, native=False),
        "gcs": block(detail["GCS"], native=False),
        "gcs_p": block(detail["GCS-P"], native=False),
        "impact_core": block(detail["IMPACT-Core"], native=True),
        "crash_basic": block(detail["CRASH-Basic"], native=True),
        "discrimination_unchanged": {
            "auroc": {"estimate": C.V4_FROZEN["auroc_test"],
                      "lo": C.V4_FROZEN["auroc_test_lo"],
                      "hi": C.V4_FROZEN["auroc_test_hi"]},
            "note": {"display":
                "The map is monotone increasing, so it moves no ranking and the "
                "area under the receiver operating characteristic curve is "
                "identical before and after recalibration."},
        },
    }
    path = C.write_part("calibration", payload)
    print(f"  wrote {path}")

    # -----------------------------------------------------------------------
    # Handoff
    # -----------------------------------------------------------------------
    write_handoff(detail, calib_df)
    print(f"  wrote {OUT_HANDOFF}")

    C.banner("Step 00 complete")
    return 0


def write_handoff(detail: dict, calib_df: pd.DataFrame) -> None:
    pat = detail["PAT-TBI"]
    gcs = detail["GCS"]
    gcsp = detail["GCS-P"]
    imp = detail["IMPACT-Core"]
    cra = detail["CRASH-Basic"]

    txt = f"""# 00_recalibrate — handoff

Owner: `v6/code/00_recalibrate.py`. Repairs findings #1 (circular calibration),
#3 (three incompatible recalibration conventions) and #10 (collapsed
Hosmer-Lemeshow degrees of freedom).

## The single recalibration convention (use this wording, do not invent another)

The map is

    logit(y) = a + b * pat_tbi_score

fitted **once** with statsmodels `Logit` on the **{C.V4_FROZEN['n_train']:,} derivation
rows** ({C.V4_FROZEN['n_train_events']:,} events) and applied **unchanged** to the
**{C.V4_FROZEN['n_test']:,} validation rows** ({C.V4_FROZEN['n_test_events']:,} events).

| Score | a (intercept) | b (slope) |
|---|---|---|
| PAT-TBI | {pat['a']:.6f} | {pat['b']:.6f} |
| GCS (total, as risk = 15 - GCS) | {gcs['a']:.6f} | {gcs['b']:.6f} |
| GCS-P (as risk = 15 - GCS-P) | {gcsp['a']:.6f} | {gcsp['b']:.6f} |

Any downstream script that needs a probability for PAT-TBI **must** use
a = {pat['a']:.6f}, b = {pat['b']:.6f}. Do not refit. Do not use isotonic
regression. Do not fit anything in validation.

IMPACT-Core and CRASH-Basic are **not** remapped: they keep their native
published probabilities, on their own (smaller) denominators.

## Headline validation calibration (all on the {C.V4_FROZEN['n_test']:,} validation rows)

| Score | CITL (offset intercept) | mean pred - mean obs | Slope | Brier | n | events |
|---|---|---|---|---|---|---|
| PAT-TBI | {pat['citl']:+.4f} | {pat['citl_plain']:+.6f} | {pat['slope']:.4f} | {pat['brier']:.5f} | {pat['n']:,} | {pat['events']:,} |
| GCS | {gcs['citl']:+.4f} | {gcs['citl_plain']:+.6f} | {gcs['slope']:.4f} | {gcs['brier']:.5f} | {gcs['n']:,} | {gcs['events']:,} |
| GCS-P | {gcsp['citl']:+.4f} | {gcsp['citl_plain']:+.6f} | {gcsp['slope']:.4f} | {gcsp['brier']:.5f} | {gcsp['n']:,} | {gcsp['events']:,} |
| IMPACT-Core (native) | {imp['citl']:+.4f} | {imp['citl_plain']:+.6f} | {imp['slope']:.4f} | {imp['brier']:.5f} | {imp['n']:,} | {imp['events']:,} |
| CRASH-Basic (native) | {cra['citl']:+.4f} | {cra['citl_plain']:+.6f} | {cra['slope']:.4f} | {cra['brier']:.5f} | {cra['n']:,} | {cra['events']:,} |

Two different quantities are both called "calibration-in-the-large" in the
literature. The **primary** one here is the intercept of
`logit(y) = alpha + offset(lp)`, on the log-odds scale. The **secondary** one,
`mean(predicted) - mean(observed)`, is on the probability scale and is reported
under its own label. Never mix them in one sentence without saying which is
which. The v4 table reported the probability-scale version and called it CITL.

They can disagree in sign, and for PAT-TBI they do: the offset intercept is
{pat['citl']:+.4f} (log-odds scale, so the map runs very slightly high) while
mean predicted minus mean observed is {pat['citl_plain']:+.6f}. The log-odds
version is dominated by the very large low-risk stratum; the probability-scale
version is dominated by the small high-risk stratum. Report the offset
intercept as the headline and give the probability-scale difference beside it
with an explicit label. Do not present one as a check on the other.

## Achieved group count

`pd.qcut` on estimated risk with `duplicates="drop"`, ten groups requested:

| Score | groups achieved |
|---|---|
| PAT-TBI | {pat['k']} |
| GCS | {gcs['k']} |
| GCS-P | {gcsp['k']} |
| IMPACT-Core | {imp['k']} |
| CRASH-Basic | {cra['k']} |

Never write "deciles" for PAT-TBI, GCS or GCS-P. Write "groups of estimated
risk" and give the achieved number. The integer score has heavy ties, so the
quantile cuts collapse.

## Hosmer-Lemeshow is relegated, not led with

| Score | chi-square | df | P |
|---|---|---|---|
| PAT-TBI | {pat['hl_chi']:.1f} | {pat['hl_df']} | {pat['hl_p']:.3g} |
| GCS | {gcs['hl_chi']:.1f} | {gcs['hl_df']} | {gcs['hl_p']:.3g} |
| GCS-P | {gcsp['hl_chi']:.1f} | {gcsp['hl_df']} | {gcsp['hl_p']:.3g} |
| IMPACT-Core | {imp['hl_chi']:.1f} | {imp['hl_df']} | {imp['hl_p']:.3g} |
| CRASH-Basic | {cra['hl_chi']:.1f} | {cra['hl_df']} | {cra['hl_p']:.3g} |

Lead with the Brier score and the grouped observed-versus-estimated table.
Mention the test once, with its achieved df, and say that at n =
{C.V4_FROZEN['n_test']:,} with a tied integer score it rejects regardless.

Note the df column. The conventional Hosmer-Lemeshow df is 8. PAT-TBI achieves
{pat['hl_df']}, GCS achieves {gcs['hl_df']} and GCS-P achieves {gcsp['hl_df']},
because the tied integer score collapses the quantile cuts. A manuscript that
writes "df = 8" for these three would be wrong. That is finding #10.

The tail probabilities for IMPACT-Core and CRASH-Basic underflow to exactly
zero in double precision. Write "P < 0.001", never "P = 0". The bundle floors
the p-value leaf at a positive value below every conventional threshold for
exactly this reason.

## How the three v4 conventions differed (for the Methods and the response letter)

All three v4 steps fitted their score-to-probability map inside the evaluation
sample, and each fitted a different map. Step 07 fitted a logistic map
`logit(y) = a + b x score` on the validation rows for the three raw scores only
and passed IMPACT-Core and CRASH-Basic through on their published
probabilities. Step 08 fitted the same logistic form on the validation rows for
**every** score, so the two published instruments were silently re-estimated
before their calibration curve, decision curve and Brier score were drawn.
Step 10 instead fitted an isotonic (non-parametric monotone) map on the
validation rows for the raw scores and again passed the two published
instruments through unchanged. Three maps, three denominators for what was
called "the calibration", and every one of them fitted in the sample it was
then evaluated on. v6 replaces all three with the single derivation-fitted
logistic map above.

## Recommended Methods sentence (single convention)

> PAT-TBI is an integer score rather than a probability, so a
> recalibration map of the form logit(y) = a + b x score was fitted once by
> maximum likelihood on the derivation cohort and applied unchanged to the
> validation cohort; the same map was used for every calibration statistic,
> for the calibration plot and for the decision-curve analysis, and no map was
> refitted in the validation cohort. Calibration-in-the-large was taken as the
> intercept of a logistic model containing the derivation-fitted linear
> predictor as an offset, and the calibration slope as its coefficient when
> that offset was instead entered as a free covariate. IMPACT-Core and
> CRASH-Basic already return probabilities and were evaluated on their
> published scale without remapping, each on the subset of patients with the
> covariates their published equations require.

## Surprises and things the next agent must not trip over

1. **The v4 calibration table is unusable.** `v4/tables/table_pat_calibration.csv`
   reports CITL 1e-06 and slope 1.0000 for PAT-TBI. Those are fitting
   identities, not results. Do not carry a single number out of that file.
   `v6/tables/table_v6_calibration.csv` replaces it entirely.
2. **The comparator risk scores are not in `candidates.parquet`.** Only
   `GCS2`, `MOTOR2` and `TQIP_TBI_PUPIL_RESPONSE` are. `gcs_risk` and
   `gcs_p_risk` were recomputed here from those columns using the v4 rules
   (missing GCS filled to 15, missing pupil filled to "both") and verified to
   match the v4 validation columns on all {C.V4_FROZEN['n_test']:,} rows with
   zero disagreements.
3. **GCS and GCS-P have full denominators, the other two do not.** `gcs_mask`
   and `gcs_p_mask` are all-true in v4 because missing values are imputed
   rather than dropped. IMPACT-Core (n = {imp['n']:,}) and CRASH-Basic
   (n = {cra['n']:,}) drop patients missing their required covariates. Any
   table putting all five side by side must show the denominators.
4. **Imputation is doing work in the comparators.** Filling a missing GCS to
   15 and a missing pupil response to "both" pushes those patients to the
   low-risk end. That is the published v4 behaviour and is preserved here for
   comparability, but it should be named in the limitations.
5. **Discrimination is untouched.** The map is monotone increasing, so the
   validation AUROC of the mapped probability equals that of the integer
   score: {C.V4_FROZEN['auroc_test']} (95% CI {C.V4_FROZEN['auroc_test_lo']} to {C.V4_FROZEN['auroc_test_hi']}).
   Both are asserted in the script and it stops if either moves.
6. **Bundle part leaves.** `calibration.json` puts a and b in numeric leaves
   (`convention.intercept`, `convention.slope`) so the templates render them.
   No number is baked into a display string.

## Files this step owns

- `v6/code/00_recalibrate.py`
- `v6/bundle/parts/calibration.json`
- `v6/tables/table_v6_calibration.csv`
- `v6/tables/table_v6_calibration_deciles.csv`
- `v6/logs/00_recalibrate.log`
- `v6/logs/00_HANDOFF.md`
"""
    OUT_HANDOFF.parent.mkdir(parents=True, exist_ok=True)
    OUT_HANDOFF.write_text(txt)


if __name__ == "__main__":
    raise SystemExit(main())
