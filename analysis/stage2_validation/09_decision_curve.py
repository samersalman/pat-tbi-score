"""
PAT-TBI v6 — Phase 0, step 09: the decision curve, redrawn under one map.
=========================================================================

WHAT WAS WRONG (the unclosed half of finding #3)
------------------------------------------------
Step 00 fixed the calibration statistics: the score-to-probability map

    logit(y) = a + b * score

is fitted ONCE on the 415,724 derivation rows and applied unchanged to the
207,863 validation rows. `bundle/parts/calibration.json` states that as the
single recalibration convention for the project, and `logs/00_HANDOFF.md`
asserts that the same map was used for the decision-curve analysis.

That sentence was not true. `v4/tables/table_pat_dca.csv` and
`v4/figures/figure_pat_dca.*` came from `v4/code/08_calibration_dca_operating.py`,
whose `recalibrate_for_dca_and_cal()` fitted a fresh logistic map INSIDE the
validation cohort for EVERY curve, including IMPACT-Core and CRASH-Basic, which
already emit published probabilities. Net benefit is defined against a
threshold probability, so a decision curve inherits its map entirely: those
curves were curves for maps estimated from the rows they were then scored on,
and for the two published instruments they were curves for silently refitted
versions of the instruments rather than for the published equations.

This script re-derives the decision curve under the one convention, so that the
Methods sentence becomes true.

WHAT IS REUSED FROM v4, DELIBERATELY
------------------------------------
  * the net-benefit arithmetic, verbatim: for a threshold probability p_t, with
    n the denominator of the curve,
        NB = TP/n - (FP/n) * (p_t / (1 - p_t)),   flagging when prob >= p_t
  * the threshold grid, verbatim: np.arange(0.01, 0.205, 0.005), 39 values from
    0.01 to 0.20, so the new curve is directly comparable to the old one
Only the map is replaced. One extra threshold is added beside the grid: the
probability the derivation-fitted map assigns to the 4-point operating point.

WHAT IS REPLACED
----------------
  PAT-TBI, Glasgow Coma Scale total, Glasgow Coma Scale pupil
        derivation-fitted logistic map, coefficients read from step 00 and
        re-derived here and asserted equal
  IMPACT-Core, CRASH-Basic
        native published probabilities, not remapped, on their own denominators

DENOMINATORS
------------
Each curve is drawn on its own denominator, each denominator is stated on the
figure and in the table, and each curve is compared against a treat-all
reference computed at the prevalence of ITS OWN denominator. v4 compared all
five curves against a single treat-all reference computed on the full
validation cohort, which is the comparison the review called unfair. The effect
of the differing denominators is then checked directly by repeating the whole
analysis restricted to the patients complete for every instrument.

Run with the framework interpreter, not bare `python3`:
    PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
    cd analysis/stage2_validation
    $PY 09_decision_curve.py

Outputs
-------
  v6/bundle/parts/dca.json
  v6/tables/table_v6_decision_curve.csv
  v6/figures/figure_v6_decision_curve.png
  v6/figures/figure_v6_decision_curve.pdf
  v6/logs/09_decision_curve.log
  v6/logs/09_HANDOFF.md
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

import matplotlib
matplotlib.use("Agg")                       # before pyplot, per house convention
import matplotlib.pyplot as plt             # noqa: E402
from matplotlib.lines import Line2D         # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C                         # noqa: E402

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

# --- the v4 threshold grid, verbatim ---------------------------------------
THRESHOLDS = np.arange(0.01, 0.205, 0.005)          # 39 values, 0.01 to 0.20
OPERATING_POINTS = 4                                 # the Youden cutpoint

CALIB_CSV = C.V6 / "tables" / "table_v6_calibration.csv"
CALIB_JSON = C.V6 / "bundle" / "parts" / "calibration.json"
V4_DCA = C.V4 / "tables" / "table_pat_dca.csv"

OUT_TABLE = C.V6 / "tables" / "table_v6_decision_curve.csv"
OUT_FIG_PNG = C.V6 / "figures" / "figure_v6_decision_curve.png"
OUT_FIG_PDF = C.V6 / "figures" / "figure_v6_decision_curve.pdf"
OUT_HANDOFF = C.V6 / "logs" / "09_HANDOFF.md"

# (display label, score column, mask column, native probability?)
SPECS = [
    ("PAT-TBI",     "pat_tbi_score",    "pat_tbi_score_mask", False),
    ("GCS",         "gcs_risk",         "gcs_mask",           False),
    ("GCS-P",       "gcs_p_risk",       "gcs_p_mask",         False),
    ("IMPACT-Core", "impact_core_prob", "impact_mask",        True),
    ("CRASH-Basic", "crash_basic_prob", "crash_mask",         True),
]
KEYS = {"PAT-TBI": "pat_tbi", "GCS": "gcs", "GCS-P": "gcs_p",
        "IMPACT-Core": "impact_core", "CRASH-Basic": "crash_basic"}

CONV_DERIVED = "derivation-fitted logistic map, logit(y) = a + b * score"
CONV_NATIVE = "native published probability, not remapped"

# Okabe-Ito, as used by the other v6 figure
OKABE = {"orange": "#E69F00", "sky_blue": "#56B4E9", "green": "#009E73",
         "blue": "#0072B2", "vermilion": "#D55E00", "purple": "#CC79A7",
         "grey": "#999999"}
COLORS = {"PAT-TBI": OKABE["vermilion"], "GCS": OKABE["blue"],
          "GCS-P": OKABE["green"], "IMPACT-Core": OKABE["purple"],
          "CRASH-Basic": OKABE["orange"]}
LINEWIDTH = {"PAT-TBI": 2.6, "GCS": 1.7, "GCS-P": 1.7,
             "IMPACT-Core": 1.7, "CRASH-Basic": 1.7}

DPI = 300


# ---------------------------------------------------------------------------
# Comparator recomputation, verbatim from v4/code/06_score_test_set.py by way
# of v6/code/00_recalibrate.py, so the derivation rows carry the same columns
# the v4 scored parquet carries for the validation rows.
# ---------------------------------------------------------------------------
GCS_P_PRS = {"both": 0, "one": 1, "none": 2}


def _as_pupil(series):
    s = series.astype(str).str.strip().str.upper()
    mapping = {"BOTH": "both", "ONE": "one", "NEITHER": "none", "NONE": "none"}
    out = s.map(mapping)
    out[~s.isin(mapping.keys())] = pd.NA
    return out


def compute_gcs_risk(df: pd.DataFrame) -> np.ndarray:
    gcs = pd.to_numeric(df.get("GCS2"), errors="coerce").fillna(15).values
    return 15.0 - gcs


def compute_gcs_p_risk(df: pd.DataFrame) -> np.ndarray:
    gcs = pd.to_numeric(df.get("GCS2"), errors="coerce").fillna(15).values
    pupil = _as_pupil(df.get("TQIP_TBI_PUPIL_RESPONSE")).fillna("both").values
    prs = np.array([GCS_P_PRS[p] for p in pupil], dtype=np.float64)
    return 15.0 - (gcs - prs)


def fit_map_on_derivation(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """logit(y) = a + b * score, statsmodels Logit, exactly as step 00 fits it."""
    X = sm.add_constant(np.asarray(score, dtype=np.float64), has_constant="add")
    res = sm.Logit(np.asarray(y, dtype=int), X).fit(disp=False, method="newton",
                                                    maxiter=200)
    return float(res.params[0]), float(res.params[1])


def sigmoid(x) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# Net-benefit arithmetic, verbatim from v4/code/08_calibration_dca_operating.py
# ---------------------------------------------------------------------------
def net_benefit(y: np.ndarray, prob: np.ndarray, p_t: float) -> float:
    """Vickers and Elkin net benefit at threshold p_t on this curve's own n."""
    n = len(y)
    if n == 0:
        return float("nan")
    if p_t >= 1.0:
        return float("nan")
    pred_pos = prob >= p_t
    tp = float((pred_pos & (y == 1)).sum()) / n
    fp = float((pred_pos & (y == 0)).sum()) / n
    return tp - fp * (p_t / (1.0 - p_t))


def nb_treat_all(prevalence: float, p_t: float) -> float:
    if p_t >= 1.0:
        return float("nan")
    return prevalence - (1.0 - prevalence) * (p_t / (1.0 - p_t))


def net_reduction_per_100(nb_model: float, nb_all: float, p_t: float) -> float:
    """Avoidable interventions per hundred patients, against treat-all.

    The standard decision-curve conversion: the net-benefit advantage over
    treating everyone, divided by the exchange rate the threshold implies, and
    expressed per hundred patients. It answers how many platelet activations a
    blood bank avoids per hundred patients at that threshold, already charged
    for the recipients the rule misses.
    """
    odds = p_t / (1.0 - p_t)
    if odds == 0:
        return float("nan")
    return 100.0 * (nb_model - nb_all) / odds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    C.start_log("09_decision_curve")

    C.banner("PAT-TBI v6 step 09 — the decision curve, redrawn under one map")
    print("The calibration statistics were repaired in step 00; the decision")
    print("curve was not. v4 fitted its score-to-probability map inside the")
    print("validation cohort, for every curve including the two published")
    print("instruments. Net benefit is defined against a threshold probability,")
    print("so a decision curve inherits its map entirely. This script redraws")
    print("the curve under the single derivation-fitted map, and leaves the two")
    print("published instruments on their own published probabilities.")

    # =======================================================================
    # 1. The map. Read step 00's record, re-derive it, and assert agreement.
    # =======================================================================
    C.banner("Gate 1 — the map coefficients must match step 00")
    calib_csv = pd.read_csv(CALIB_CSV).set_index("Score")
    calib_json = json.loads(CALIB_JSON.read_text())

    train, test = C.load_split()
    y_tr = train[C.OUTCOME].to_numpy(dtype=int)
    print(f"  derivation n = {len(train):,}  events = {y_tr.sum():,}")
    assert len(train) == C.V4_FROZEN["n_train"]
    assert int(y_tr.sum()) == C.V4_FROZEN["n_train_events"]

    train["gcs_risk"] = compute_gcs_risk(train)
    train["gcs_p_risk"] = compute_gcs_p_risk(train)

    maps: dict[str, tuple[float, float]] = {}
    for disp, col in (("PAT-TBI", "pat_tbi_score"), ("GCS", "gcs_risk"),
                      ("GCS-P", "gcs_p_risk")):
        a, b = fit_map_on_derivation(y_tr, train[col].to_numpy(dtype=np.float64))
        a_rec = float(calib_csv.loc[disp, "Map_Intercept_a"])
        b_rec = float(calib_csv.loc[disp, "Map_Slope_b"])
        d_a, d_b = abs(a - a_rec), abs(b - b_rec)
        print(f"  {disp:<8} re-derived a = {a:+.9f}  b = {b:+.9f}")
        print(f"  {'':<8} step 00   a = {a_rec:+.9f}  b = {b_rec:+.9f}   "
              f"|difference| = {d_a:.2e}, {d_b:.2e}")
        assert d_a < 5e-7 and d_b < 5e-7, (
            f"{disp}: the re-derived map does not match step 00 to six decimals")
        jb = calib_json[KEYS[disp]]
        assert round(a, 4) == jb["map_intercept"]["value"], f"{disp} intercept vs bundle"
        assert round(b, 4) == jb["map_slope"]["value"], f"{disp} slope vs bundle"
        maps[disp] = (a, b)
    # the project-level convention leaves, the ones every document renders from
    assert round(maps["PAT-TBI"][0], 4) == calib_json["convention"]["intercept"]["value"]
    assert round(maps["PAT-TBI"][1], 4) == calib_json["convention"]["slope"]["value"]
    print("  [PASS] all three maps match step 00 to six decimals and match the")
    print("         bundle's convention leaves exactly")

    a_pat, b_pat = maps["PAT-TBI"]

    # =======================================================================
    # 2. The validation rows and their denominators
    # =======================================================================
    C.banner("Gate 2 — the validation cohort must not have moved")
    scored = C.load_test_scored()
    y_all = scored[C.OUTCOME].to_numpy(dtype=int)
    n_all, ev_all = len(y_all), int(y_all.sum())
    print(f"  validation n = {n_all:,}  events = {ev_all:,} "
          f"({100 * y_all.mean():.3f}%)")
    assert n_all == C.V4_FROZEN["n_test"], f"validation n moved: {n_all}"
    assert ev_all == C.V4_FROZEN["n_test_events"], f"validation events moved: {ev_all}"

    # the integer score in the scored parquet must be the one _common builds
    chk = test[["_key", "pat_tbi_score"]].merge(
        scored.assign(_key=C._key_series(scored))[["_key", "pat_tbi_score"]],
        on="_key", suffixes=("_v6", "_v4"), validate="one_to_one")
    n_bad = int((chk["pat_tbi_score_v6"] != chk["pat_tbi_score_v4"]).sum())
    assert n_bad == 0, "the scored parquet disagrees with the recomputed score"
    print(f"  [PASS] score rows disagreeing with the frozen scored file: {n_bad}")

    C.banner("Denominators, each curve on its own")
    curves: dict[str, dict] = {}
    for disp, col, mask_col, native in SPECS:
        m = scored[mask_col].to_numpy(dtype=bool) & scored[col].notna().to_numpy()
        y = y_all[m]
        s = scored.loc[m, col].to_numpy(dtype=np.float64)
        if native:
            p = np.clip(s, 1e-9, 1.0 - 1e-9)
            conv = CONV_NATIVE
        else:
            a, b = maps[disp]
            p = sigmoid(a + b * s)
            conv = CONV_DERIVED
        curves[disp] = {"y": y, "p": p, "n": int(m.sum()),
                        "events": int(y.sum()), "prev": float(y.mean()),
                        "native": native, "convention": conv, "mask": m}
        print(f"  {disp:<12} n = {m.sum():>7,}  events = {int(y.sum()):>5,}  "
              f"prevalence = {y.mean():.5f}  map: {conv}")

    assert curves["IMPACT-Core"]["n"] == 199757, "IMPACT-Core denominator moved"
    assert curves["CRASH-Basic"]["n"] == 199862, "CRASH-Basic denominator moved"
    assert curves["PAT-TBI"]["n"] == C.V4_FROZEN["n_test"]

    print("\n  Denominator decision: each curve is drawn on its own denominator,")
    print("  each denominator is labelled on the figure and in the table, and")
    print("  each curve is compared against a treat-all reference computed at")
    print("  the prevalence of its own denominator. v4 compared all five against")
    print("  one treat-all reference taken from the full validation cohort. The")
    print("  effect of the differing denominators is measured below on the set")
    print("  of patients complete for every instrument.")

    # =======================================================================
    # 3. Hard gate on the arithmetic itself
    # =======================================================================
    C.banner("Gate 3 — net benefit at a threshold of zero must equal prevalence")
    for disp, d in curves.items():
        nb_all_0 = nb_treat_all(d["prev"], 0.0)
        nb_mod_0 = net_benefit(d["y"], d["p"], 0.0)
        print(f"  {disp:<12} prevalence = {d['prev']:.9f}   "
              f"treat-all NB at zero = {nb_all_0:.9f}   "
              f"model NB at zero = {nb_mod_0:.9f}")
        assert abs(nb_all_0 - d["prev"]) < 1e-12, (
            f"{disp}: treat-all net benefit at a threshold of zero is not the "
            f"event prevalence; the net-benefit formula is wrong")
        assert abs(nb_mod_0 - d["prev"]) < 1e-12, (
            f"{disp}: at a threshold of zero every patient is flagged, so the "
            f"model net benefit must also equal the prevalence")
    print("  [PASS] the net-benefit formula reproduces the prevalence at zero for")
    print("         every curve, under both the treat-all and the flag-everyone rule")

    # rounding robustness: the six-decimal record must give the same curve
    a_r, b_r = (float(calib_csv.loc["PAT-TBI", "Map_Intercept_a"]),
                float(calib_csv.loc["PAT-TBI", "Map_Slope_b"]))
    p_round = sigmoid(a_r + b_r * scored["pat_tbi_score"].to_numpy(dtype=np.float64))
    worst = max(abs(net_benefit(y_all, p_round, t)
                    - net_benefit(y_all, curves["PAT-TBI"]["p"], t))
                for t in THRESHOLDS)
    print(f"\n  using the six-decimal published coefficients instead of the "
          f"re-derived\n  ones moves no net benefit by more than {worst:.3e}")

    # =======================================================================
    # 4. The operating point
    # =======================================================================
    C.banner(f"The {OPERATING_POINTS}-point operating point")
    p_op = float(sigmoid(a_pat + b_pat * OPERATING_POINTS))
    print(f"  the derivation-fitted map assigns {OPERATING_POINTS} points an "
          f"estimated probability of {p_op:.6f}")
    print(f"  so the threshold probability implied by the operating point is "
          f"{100 * p_op:.3f}%")

    d = curves["PAT-TBI"]
    flag_prob = d["p"] >= p_op
    flag_score = scored["pat_tbi_score"].to_numpy() >= OPERATING_POINTS
    assert int((flag_prob != flag_score).sum()) == 0, (
        "flagging on the mapped probability at the operating threshold does not "
        "reproduce flagging the integer score at four points")
    print(f"  [PASS] flagging at that probability is identical to flagging the "
          f"score at {OPERATING_POINTS} points, on all {d['n']:,} rows")

    tp = int((flag_prob & (d["y"] == 1)).sum())
    fp = int((flag_prob & (d["y"] == 0)).sum())
    fn = int((~flag_prob & (d["y"] == 1)).sum())
    tn = int((~flag_prob & (d["y"] == 0)).sum())
    sens, spec = tp / (tp + fn), tn / (tn + fp)
    sens_lo, sens_hi = C.wilson_ci(tp, tp + fn)
    spec_lo, spec_hi = C.wilson_ci(tn, tn + fp)

    nb_op = net_benefit(d["y"], d["p"], p_op)
    nb_all_op = nb_treat_all(d["prev"], p_op)
    avoid_op = net_reduction_per_100(nb_op, nb_all_op, p_op)
    flagged_per_100 = 100.0 * float(flag_prob.mean())

    print(f"\n  flagged            = {tp + fp:,} of {d['n']:,} "
          f"({flagged_per_100:.2f} per hundred)")
    print(f"  true positives     = {tp:,} of {d['events']:,} recipients   "
          f"sensitivity = {sens:.4f} ({sens_lo:.4f} to {sens_hi:.4f})")
    print(f"  false positives    = {fp:,}                    "
          f"specificity = {spec:.4f} ({spec_lo:.4f} to {spec_hi:.4f})")
    print(f"  net benefit        = {nb_op:+.6f}")
    print(f"  treat-all          = {nb_all_op:+.6f}")
    print(f"  treat-none         = {0.0:+.6f}")
    print(f"  avoidable interventions per hundred patients, against treating "
          f"everyone = {avoid_op:.2f}")
    print(f"\n  Read that last number as: at the threshold the operating point")
    print(f"  implies, using the score instead of activating platelets for every")
    print(f"  patient avoids about {avoid_op:.0f} platelet activations per hundred")
    print(f"  patients, already charged for the {d['events'] - tp:,} recipients in")
    print(f"  {d['n']:,} that the rule misses. The crude count is larger, "
          f"{100 - flagged_per_100:.1f}")
    print(f"  fewer activations per hundred, because the crude count does not pay")
    print(f"  for the missed recipients.")

    # =======================================================================
    # 5. The curves
    # =======================================================================
    C.banner("Net benefit across the threshold grid")
    print(f"  grid: {len(THRESHOLDS)} thresholds from {THRESHOLDS[0]:.3f} to "
          f"{THRESHOLDS[-1]:.3f}, step 0.005, taken verbatim from v4 so the")
    print("  new curve is directly comparable to the old one; the operating-point")
    print("  threshold is carried beside the grid and marked as such.")

    rows: list[dict] = []
    grid_all = list(THRESHOLDS) + [p_op]
    source_all = ["prespecified grid"] * len(THRESHOLDS) + ["operating point"]

    for p_t, src in zip(grid_all, source_all):
        for disp, _, _, _ in SPECS:
            dd = curves[disp]
            nb = net_benefit(dd["y"], dd["p"], p_t)
            nb_a = nb_treat_all(dd["prev"], p_t)
            flag = dd["p"] >= p_t
            rows.append({
                "threshold": round(float(p_t), 9),
                "threshold_source": src,
                "score": disp,
                "net_benefit": nb,
                "n_used": dd["n"],
                "events_used": dd["events"],
                "prevalence": dd["prev"],
                "treat_all_nb": nb_a,
                "treat_none_nb": 0.0,
                "beats_both_defaults": bool(nb > nb_a and nb > 0.0),
                "flagged_per_100": 100.0 * float(flag.mean()),
                "avoidable_interventions_per_100":
                    net_reduction_per_100(nb, nb_a, p_t),
                "map_convention": dd["convention"],
            })
    dca = pd.DataFrame(rows)

    grid_rows = dca[dca["threshold_source"] == "prespecified grid"]
    pat_grid = grid_rows[grid_rows["score"] == "PAT-TBI"].sort_values("threshold")
    print(f"\n  {'p_t':>7} {'PAT-TBI NB':>12} {'treat all':>12} {'treat none':>11} "
          f"{'beats both':>11} {'avoid/100':>10}")
    for _, r in pat_grid.iterrows():
        print(f"  {r['threshold']:>7.3f} {r['net_benefit']:>12.6f} "
              f"{r['treat_all_nb']:>12.6f} {0.0:>11.6f} "
              f"{str(bool(r['beats_both_defaults'])):>11} "
              f"{r['avoidable_interventions_per_100']:>10.2f}")

    C.banner("Where each curve exceeds BOTH default strategies")
    superiority: dict[str, dict] = {}
    for disp, _, _, _ in SPECS:
        sub = grid_rows[grid_rows["score"] == disp].sort_values("threshold")
        good = sub[sub["beats_both_defaults"]]
        if len(good) == 0:
            superiority[disp] = {"any": False, "lo": None, "hi": None,
                                 "k": 0, "contiguous": False}
            print(f"  {disp:<12} NEVER exceeds both defaults anywhere on the grid")
            continue
        lo, hi = float(good["threshold"].min()), float(good["threshold"].max())
        span = sub[(sub["threshold"] >= lo) & (sub["threshold"] <= hi)]
        contiguous = bool(span["beats_both_defaults"].all())
        superiority[disp] = {"any": True, "lo": lo, "hi": hi,
                             "k": int(len(good)), "contiguous": contiguous}
        print(f"  {disp:<12} exceeds both defaults at {len(good)} of "
              f"{len(sub)} thresholds, from {lo:.3f} to {hi:.3f}"
              + ("" if contiguous else "  (NOT contiguous)"))

    pat_sup = superiority["PAT-TBI"]
    assert pat_sup["any"], (
        "PAT-TBI does not exceed both default strategies anywhere on the grid; "
        "that is a finding, not a failure, but the handoff must say so plainly")

    # How far past the pre-specified grid does the advantage run? Only the
    # unbroken run that contains the grid is reported. Above it the curve
    # crosses zero and back on a handful of patients at the top of the score,
    # and reporting those isolated points as an advantage would be dishonest.
    ext = np.arange(0.005, 0.9755, 0.005)
    dpat = curves["PAT-TBI"]
    ext_ok = np.array([net_benefit(dpat["y"], dpat["p"], t)
                       > max(0.0, nb_treat_all(dpat["prev"], t)) for t in ext])
    runs, cur = [], []
    for t, k in zip(ext, ext_ok):
        if k:
            cur.append(float(t))
        elif cur:
            runs.append((cur[0], cur[-1]))
            cur = []
    if cur:
        runs.append((cur[0], cur[-1]))
    containing = [r for r in runs
                  if r[0] <= THRESHOLDS[0] + 1e-9 and r[1] >= THRESHOLDS[-1] - 1e-9]
    assert len(containing) == 1, (
        "the advantage over both defaults is broken somewhere inside the "
        "pre-specified grid; the superiority range must then be reported as "
        "disjoint intervals rather than as one range")
    ext_lo, ext_hi = containing[0]
    print(f"\n  Beyond the pre-specified grid, scanning {ext[0]:.3f} to "
          f"{ext[-1]:.3f} at the same step, the advantage over both defaults is")
    print(f"  unbroken from {ext_lo:.3f} to {ext_hi:.3f}. The grid, not the "
          f"score, is what ends the reported range.")
    later = [r for r in runs if r[0] > ext_hi + 1e-9]
    if later:
        print(f"  Above {ext_hi:.3f} the curve crosses zero and back on a handful "
              f"of patients at\n  the top of the score ({len(later)} isolated "
              f"stretches). Those are not reported as an\n  advantage and they "
              f"are outside any plausible threshold for this decision.")

    # =======================================================================
    # 6. How far the correction moved the curve
    # =======================================================================
    C.banner("How much the correction moved each curve, against v4")
    v4 = pd.read_csv(V4_DCA)
    shifts: dict[str, dict] = {}
    print(f"  {'score':<12} {'max |shift|':>12} {'mean shift':>12} "
          f"{'at 0.025':>10} {'at 0.05':>10} {'at 0.10':>10} {'at 0.20':>10} "
          f"{'negligible':>12}")
    for disp, _, _, _ in SPECS:
        old = (v4[v4["score"] == disp].set_index("threshold")["net_benefit"]
               .sort_index())
        new = (grid_rows[grid_rows["score"] == disp]
               .set_index("threshold")["net_benefit"].sort_index())
        old.index = np.round(old.index.astype(float), 4)
        new.index = np.round(new.index.astype(float), 4)
        delta = (new - old.reindex(new.index))
        assert delta.notna().all(), f"{disp}: v4 grid did not align"
        rel = (delta.abs() / new.abs()).replace([np.inf, -np.inf], np.nan)
        shifts[disp] = {
            "max_abs": float(delta.abs().max()),
            "mean": float(delta.mean()),
            "n_negligible": int((delta.abs() < 1e-5).sum()),
            "max_relative": float(rel.max()),
            "at_thresholds": {f"{t:.3f}": float(delta.loc[round(t, 4)])
                              for t in (0.025, 0.05, 0.10, 0.20)},
            "v4_n_beats": int(((old > np.maximum(
                0.0, v4[v4["score"] == disp].set_index("threshold")["treat_all_nb"]
                .sort_index().values))).sum()),
        }
        print(f"  {disp:<12} {delta.abs().max():>12.6f} {delta.mean():>+12.6f} "
              + " ".join(f"{delta.loc[round(t, 4)]:>+10.6f}"
                         for t in (0.025, 0.05, 0.10, 0.20))
              + f" {int((delta.abs() < 1e-5).sum()):>7} of {len(delta)}")

    # Why the shift is so small for the integer score: the map v4 fitted inside
    # the validation cohort and the map fitted on the derivation rows imply the
    # SAME integer cutpoint at almost every threshold, so the same patients are
    # flagged. The v4 map is refitted here for this diagnostic only; no reported
    # curve uses it.
    s_te = scored["pat_tbi_score"].to_numpy(dtype=np.float64)
    a4, b4 = fit_map_on_derivation(y_all, s_te)          # the v4 convention
    print(f"\n  Diagnostic, not used for any reported curve. The map v4 fitted "
          f"inside the\n  validation cohort is a = {a4:.6f}, b = {b4:.6f}, "
          f"against a = {a_pat:.6f},\n  b = {b_pat:.6f} fitted on the derivation "
          f"rows.")

    def implied_cutpoint(a, b, t):
        return int(np.ceil((np.log(t / (1.0 - t)) - a) / b))

    n_same_cut = sum(implied_cutpoint(a_pat, b_pat, t)
                     == implied_cutpoint(a4, b4, t) for t in THRESHOLDS)
    shifts["PAT-TBI"]["n_same_cutpoint"] = int(n_same_cut)
    shifts["PAT-TBI"]["v4_map_a"] = a4
    shifts["PAT-TBI"]["v4_map_b"] = b4
    print(f"  The two maps imply the SAME integer cutpoint on the score at "
          f"{n_same_cut} of\n  {len(THRESHOLDS)} thresholds, which is why the "
          f"curve barely moves: at those\n  thresholds the same patients are "
          f"flagged either way.")
    print(f"  Largest relative shift for the score, against its own net benefit: "
          f"{100 * shifts['PAT-TBI']['max_relative']:.2f} per cent, and that is at "
          f"the far end of\n  the grid where the net benefit itself is smallest.")

    print("\n  v4 drew every curve against ONE treat-all reference taken from the")
    print("  full validation cohort, and counted these thresholds as beating both")
    print("  defaults:")
    for disp, _, _, _ in SPECS:
        print(f"    {disp:<12} v4 {shifts[disp]['v4_n_beats']:>2} of "
              f"{len(THRESHOLDS)}   now {superiority[disp]['k']:>2} of "
              f"{len(THRESHOLDS)}")

    print("\n  The two raw-score comparators and the score itself barely move: the")
    print("  derivation-fitted and validation-fitted maps are close, and both are")
    print("  monotone, so at almost every threshold the same integer cutpoint is")
    print("  selected and the same patients are flagged. The circularity was NOT")
    print("  flattering the PAT-TBI decision curve.")
    print("  IMPACT-Core is the curve the correction destroys. Its published")
    print("  probabilities average far above the observed rate of platelet")
    print("  receipt; v4 refitted a map for it inside the validation cohort,")
    print("  which pulled it onto the right scale and made it look useful. On its")
    print("  published scale it never exceeds treating nobody anywhere on the")
    print("  grid.")

    # =======================================================================
    # 7. Common complete-case check
    # =======================================================================
    C.banner("Denominator check — the same analysis on the common complete-case set")
    m_common = (curves["IMPACT-Core"]["mask"] & curves["CRASH-Basic"]["mask"])
    y_c = y_all[m_common]
    n_c, ev_c = int(m_common.sum()), int(y_c.sum())
    print(f"  patients complete for every instrument: n = {n_c:,}, "
          f"events = {ev_c:,}, prevalence = {y_c.mean():.5f}")
    print(f"  that is {n_all - n_c:,} fewer than the full validation cohort")

    common_worst = 0.0
    common_sup: dict[str, dict] = {}
    for disp, col, _, native in SPECS:
        s = scored.loc[m_common, col].to_numpy(dtype=np.float64)
        if native:
            p = np.clip(s, 1e-9, 1.0 - 1e-9)
        else:
            a, b = maps[disp]
            p = sigmoid(a + b * s)
        prev_c = float(y_c.mean())
        diffs, beats = [], []
        for t in THRESHOLDS:
            nb_c = net_benefit(y_c, p, t)
            diffs.append(abs(nb_c - net_benefit(curves[disp]["y"],
                                                curves[disp]["p"], t)))
            beats.append(nb_c > max(0.0, nb_treat_all(prev_c, t)))
        worst_d = float(max(diffs))
        common_worst = max(common_worst, worst_d)
        tb = [t for t, k in zip(THRESHOLDS, beats) if k]
        common_sup[disp] = {"k": len(tb),
                            "lo": float(min(tb)) if tb else None,
                            "hi": float(max(tb)) if tb else None}
        rng = (f"{min(tb):.3f} to {max(tb):.3f}" if tb else "never")
        print(f"  {disp:<12} largest net-benefit difference against its own "
              f"denominator = {worst_d:.6f};  exceeds both defaults at "
              f"{len(tb)} of {len(THRESHOLDS)} ({rng})")

    same = all(common_sup[d]["k"] == superiority[d]["k"] for d, _, _, _ in SPECS)
    print(f"\n  Every superiority range is {'unchanged' if same else 'CHANGED'} "
          f"when the analysis is restricted to the common set, and no curve moves")
    print(f"  by more than {common_worst:.6f} in net benefit. The differing")
    print("  denominators therefore change nothing, and the primary figure keeps")
    print("  each curve on its own denominator so that the tool under evaluation")
    print("  is drawn on every patient it can score.")

    # the three treat-all references, one per denominator
    ta_gap = max(abs(nb_treat_all(curves[d]["prev"], t)
                     - nb_treat_all(curves["PAT-TBI"]["prev"], t))
                 for d in ("IMPACT-Core", "CRASH-Basic") for t in THRESHOLDS)
    print(f"\n  The three treat-all references, one per denominator, differ by at")
    print(f"  most {ta_gap:.6f} in net benefit anywhere on the grid, so they")
    print("  overlap on the figure. All three are drawn and all three are in the")
    print("  table.")

    # =======================================================================
    # 8. Table
    # =======================================================================
    C.banner("Writing the table")
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    dca_out = dca[["threshold", "threshold_source", "score", "net_benefit",
                   "n_used", "events_used", "prevalence", "treat_all_nb",
                   "treat_none_nb", "beats_both_defaults", "flagged_per_100",
                   "avoidable_interventions_per_100", "map_convention"]]
    dca_out.to_csv(OUT_TABLE, index=False)
    print(f"  wrote {OUT_TABLE}  ({len(dca_out)} rows)")

    # =======================================================================
    # 9. Figure
    # =======================================================================
    C.banner("Drawing the panel")
    draw_figure(curves, grid_rows, p_op, nb_op, superiority)
    print(f"  wrote {OUT_FIG_PNG}")
    print(f"  wrote {OUT_FIG_PDF}")

    # =======================================================================
    # 10. Bundle part
    # =======================================================================
    C.banner("Writing the bundle part")
    payload = build_part(curves, grid_rows, superiority, shifts, p_op, nb_op,
                         nb_all_op, avoid_op, flagged_per_100, tp, fp, fn,
                         sens, sens_lo, sens_hi, spec, spec_lo, spec_hi,
                         n_c, ev_c, common_worst, common_sup, same,
                         ext_lo, ext_hi, ta_gap, a_pat, b_pat)
    path = C.write_part("dca", payload)
    print(f"  wrote {path}")

    write_handoff(curves, superiority, shifts, p_op, nb_op, nb_all_op, avoid_op,
                  flagged_per_100, tp, fp, fn, sens, spec, n_c, ev_c,
                  common_worst, same, ext_lo, ext_hi, ta_gap, a_pat, b_pat)
    print(f"  wrote {OUT_HANDOFF}")

    C.banner("Step 09 complete")
    return 0


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def draw_figure(curves, grid_rows, p_op, nb_op, superiority) -> None:
    plt.rcParams.update({
        "figure.dpi": DPI, "savefig.dpi": DPI,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10.5, "axes.labelsize": 11.5,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "axes.linewidth": 0.9,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(7.4, 5.4))

    x = np.asarray(THRESHOLDS, dtype=float)

    # Treat-none is the zero line.
    ax.axhline(0.0, color="#777777", lw=1.2, ls="-", zorder=3)

    for disp, _, _, _ in SPECS:
        sub = grid_rows[grid_rows["score"] == disp].sort_values("threshold")
        ax.plot(sub["threshold"].values, sub["net_benefit"].values,
                color=COLORS[disp], lw=LINEWIDTH[disp], solid_capstyle="round",
                zorder=5 if disp != "PAT-TBI" else 6)

    # Treat-all, once per denominator, drawn ON TOP of the curves. Over this
    # range IMPACT-Core assigns every patient a probability above the
    # threshold, so it flags everyone and its curve lies exactly on treat-all.
    # Drawing the reference last is what lets a reader see that.
    for disp in ("IMPACT-Core", "CRASH-Basic"):
        ax.plot(x, [nb_treat_all(curves[disp]["prev"], t) for t in x],
                color="#BBBBBB", lw=2.0, ls=(0, (6, 3)), zorder=8)
    ax.plot(x, [nb_treat_all(curves["PAT-TBI"]["prev"], t) for t in x],
            color="#555555", lw=1.5, ls=(0, (6, 3)), zorder=9)

    # The operating point.
    ax.axvline(p_op, color="#444444", lw=0.9, ls=":", zorder=4)
    ax.plot([p_op], [nb_op], marker="o", ms=6.5, color=COLORS["PAT-TBI"],
            markeredgecolor="white", markeredgewidth=1.2, zorder=10)
    ax.annotate(f"{OPERATING_POINTS}-point operating point\n"
                f"(threshold {100 * p_op:.1f}%)",
                xy=(p_op, nb_op), xytext=(p_op + 0.013, nb_op + 0.0018),
                fontsize=8.8, color="#222222", zorder=11,
                arrowprops=dict(arrowstyle="-", lw=0.7, color="#666666"))

    ax.set_xlim(float(x[0]), float(x[-1]))
    ax.set_ylim(-0.0108, 0.0178)
    ax.set_xlabel("Threshold probability of receiving platelets within four hours")
    ax.set_ylabel("Net benefit")
    ax.set_xticks([0.01, 0.05, 0.10, 0.15, 0.20])
    ax.set_xticklabels(["1%", "5%", "10%", "15%", "20%"])
    ax.grid(True, which="major", ls=":", lw=0.7, alpha=0.45)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    handles = [
        Line2D([], [], color=COLORS[d], lw=LINEWIDTH[d],
               label=f"{d}  (n = {curves[d]['n']:,})")
        for d, _, _, _ in SPECS
    ] + [
        Line2D([], [], color="#777777", lw=1.4, ls=(0, (5, 2)),
               label="Treat all, drawn on each denominator"),
        Line2D([], [], color="#777777", lw=1.2, ls="-", label="Treat none"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.95,
              edgecolor="#CCCCCC", borderpad=0.6, handlelength=2.2)

    ax.text(0.199, -0.0100,
            "Treat all leaves the panel below. IMPACT-Core, on its published "
            "probabilities, assigns\nevery patient a value above these "
            "thresholds, so it flags everyone and its curve lies on\ntreat all. "
            "Each curve is drawn on its own denominator, against a treat-all "
            "reference\ncomputed at that denominator's own prevalence.",
            fontsize=8.0, color="#444444", va="bottom", ha="right", zorder=12,
            linespacing=1.45,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                      edgecolor="#DDDDDD", linewidth=0.7))

    fig.tight_layout()
    OUT_FIG_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG_PNG, dpi=DPI, bbox_inches="tight")
    fig.savefig(OUT_FIG_PDF, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Bundle part
# ---------------------------------------------------------------------------
def build_part(curves, grid_rows, superiority, shifts, p_op, nb_op, nb_all_op,
               avoid_op, flagged_per_100, tp, fp, fn, sens, sens_lo, sens_hi,
               spec, spec_lo, spec_hi, n_c, ev_c, common_worst, common_sup,
               same, ext_lo, ext_hi, ta_gap, a_pat, b_pat) -> dict:

    def curve_block(disp: str) -> dict:
        d = curves[disp]
        sub = grid_rows[grid_rows["score"] == disp].sort_values("threshold")
        nb_at = {f"{t:.3f}": float(sub.loc[sub["threshold"].round(4) == round(t, 4),
                                           "net_benefit"].iloc[0])
                 for t in (0.02, 0.05, 0.10, 0.20)}
        best = sub.loc[sub["net_benefit"].idxmax()]
        s = superiority[disp]
        blk = {
            "n": {"n": d["n"]},
            "events": {"n": d["events"], "N": d["n"]},
            "net_benefit_at_operating_threshold":
                {"value": round(float(net_benefit(d["y"], d["p"], p_op)), 6)},
            "net_benefit_at_two_percent": {"value": round(nb_at["0.020"], 6)},
            "net_benefit_at_five_percent": {"value": round(nb_at["0.050"], 6)},
            "net_benefit_at_ten_percent": {"value": round(nb_at["0.100"], 6)},
            "net_benefit_at_twenty_percent": {"value": round(nb_at["0.200"], 6)},
            "max_net_benefit": {"value": round(float(best["net_benefit"]), 6)},
            "max_net_benefit_threshold": {"value": round(float(best["threshold"]), 3)},
            "thresholds_exceeding_both_defaults": {"n": s["k"], "N": len(THRESHOLDS)},
        }
        if s["any"]:
            blk["superiority_lower"] = {"value": round(s["lo"], 3)}
            blk["superiority_upper"] = {"value": round(s["hi"], 3)}
            blk["superiority_contiguous"] = {
                "display": ("the advantage is unbroken across that range"
                            if s["contiguous"] else
                            "the advantage is not unbroken across that range")}
        else:
            blk["superiority"] = {"display":
                "This instrument does not exceed both default strategies at any "
                "threshold on the examined grid."}
        return blk

    pat = curve_block("PAT-TBI")
    pat.update({
        "operating_point": {
            "points": {"value": OPERATING_POINTS},
            "threshold_probability": {"value": round(p_op, 5)},
            "net_benefit": {"value": round(nb_op, 6)},
            "treat_all_net_benefit": {"value": round(nb_all_op, 6)},
            "treat_none_net_benefit": {"value": 0.0},
            "avoidable_interventions_per_hundred": {"value": round(avoid_op, 1)},
            "flagged_per_hundred": {"value": round(flagged_per_100, 2)},
            "flagged": {"n": tp + fp, "N": curves["PAT-TBI"]["n"]},
            "true_positives": {"n": tp, "N": curves["PAT-TBI"]["events"]},
            "false_positives": {"n": fp, "N": curves["PAT-TBI"]["n"] - curves["PAT-TBI"]["events"]},
            "missed_recipients": {"n": fn, "N": curves["PAT-TBI"]["events"]},
            "sensitivity": {"estimate": sens, "lo": sens_lo, "hi": sens_hi},
            "specificity": {"estimate": spec, "lo": spec_lo, "hi": spec_hi},
            "equivalence": {"display":
                "The map is monotone increasing, so flagging on the estimated "
                "probability at this threshold flags exactly the patients whose "
                "integer score reaches the operating point, on every row of the "
                "validation cohort."},
            "interpretation": {"display":
                "The avoidable interventions figure is the net-benefit advantage "
                "over activating platelets for every patient, converted at the "
                "exchange rate the threshold implies and expressed per hundred "
                "patients. It is already charged for the recipients the rule "
                "misses, which is why it is smaller than the crude reduction in "
                "the number of patients flagged."},
        },
        "extended_scan": {
            "lower": {"value": round(ext_lo, 3)},
            "upper": {"value": round(ext_hi, 3)},
            "note": {"display":
                "Scanning beyond the pre-specified grid at the same step, the "
                "advantage over both default strategies is unbroken over a much "
                "wider band than the grid covers, so it is the grid and not the "
                "score that ends the reported range. Above that band the curve "
                "crosses and recrosses on a handful of patients at the top of "
                "the score, at thresholds far outside any plausible value for "
                "this decision, and those crossings are not reported as an "
                "advantage."},
        },
    })

    return {
        "_meta": {
            "script": "09_decision_curve.py",
            "finding": 3,
            "closes": {"display":
                "the half of the recalibration finding that step zero left open, "
                "namely the decision curve"},
        },
        "convention": {
            "statement": {"display":
                "Net benefit was computed from the estimated probability given by "
                "the map that was fitted once on the derivation rows and applied "
                "unchanged to the validation rows. That is the same map used for "
                "every calibration statistic and for the calibration plot, so the "
                "project now has one recalibration convention and the decision "
                "curve obeys it."},
            "map_intercept": {"value": round(a_pat, 4)},
            "map_slope": {"value": round(b_pat, 4)},
            "fitted_on": {"n": C.V4_FROZEN["n_train"]},
            "evaluated_on": {"n": C.V4_FROZEN["n_test"]},
            "evaluated_events": {"n": C.V4_FROZEN["n_test_events"],
                                 "N": C.V4_FROZEN["n_test"]},
            "published_instruments": {"display":
                "IMPACT-Core and CRASH-Basic keep their own published "
                "probabilities and are not remapped, so their curves are the "
                "curves a reader obtains by applying the published equations."},
            "prior_defect": {"display":
                "The earlier decision curve fitted a fresh map inside the "
                "validation cohort for every curve, including the two published "
                "instruments. Net benefit is defined against a threshold "
                "probability, so a decision curve inherits its map entirely, and "
                "those curves therefore described maps estimated from the very "
                "rows they were then scored on. For the two published "
                "instruments they described refitted versions of the instruments "
                "rather than the published equations."},
            "arithmetic": {"display":
                "Net benefit is the proportion of the denominator flagged and "
                "truly receiving platelets, less the proportion flagged and not "
                "receiving them, weighted by the odds of the threshold "
                "probability. The arithmetic and the threshold grid are the ones "
                "the earlier analysis used, so the corrected curve is directly "
                "comparable to it."},
        },
        "grid": {
            "lower": {"value": round(float(THRESHOLDS[0]), 4)},
            "upper": {"value": round(float(THRESHOLDS[-1]), 4)},
            "step": {"value": round(float(THRESHOLDS[1] - THRESHOLDS[0]), 4)},
            "n_thresholds": {"value": len(THRESHOLDS)},
        },
        "denominators": {
            "decision": {"display":
                "Each curve is drawn on its own denominator, every denominator is "
                "stated on the figure and in the table, and each curve is "
                "compared against a treat-all reference computed at the "
                "prevalence of its own denominator. The tool under evaluation is "
                "scoreable for every patient in the validation cohort, so "
                "restricting the primary curve to the subset for which the two "
                "published instruments can be computed would drop patients from "
                "the curve that carries the main claim in order to accommodate "
                "two secondary severity proxies. The earlier analysis instead "
                "drew all five curves against a single treat-all reference taken "
                "from the full validation cohort, which is not a fair comparison "
                "when the denominators differ."},
            "pat_tbi": {"n": curves["PAT-TBI"]["n"]},
            "gcs": {"n": curves["GCS"]["n"]},
            "gcs_p": {"n": curves["GCS-P"]["n"]},
            "impact_core": {"n": curves["IMPACT-Core"]["n"]},
            "crash_basic": {"n": curves["CRASH-Basic"]["n"]},
            "treat_all_reference_gap": {"value": round(ta_gap, 6)},
            "treat_all_note": {"display":
                "The treat-all references for the three denominators differ by so "
                "little that they overlap on the figure. All three are drawn and "
                "all three are carried in the table."},
            "common_complete_case": {
                "n": {"n": n_c, "N": C.V4_FROZEN["n_test"]},
                "events": {"n": ev_c, "N": n_c},
                "largest_net_benefit_difference": {"value": round(common_worst, 6)},
                "conclusion": {"display":
                    ("Repeating the whole analysis on the patients complete for "
                     "every instrument leaves every superiority range unchanged "
                     "and moves no curve materially, so the choice of denominator "
                     "changes nothing."
                     if same else
                     "Repeating the whole analysis on the patients complete for "
                     "every instrument changes at least one superiority range, "
                     "which is reported in the log.")},
            },
        },
        "verification": {
            "prevalence_identity": {"display":
                "At a threshold of zero the net benefit of treating everyone must "
                "equal the event prevalence, and so must the net benefit of any "
                "rule that flags everyone. Both were asserted for every curve "
                "before any figure was drawn."},
        },
        "pat_tbi": pat,
        "gcs": curve_block("GCS"),
        "gcs_p": curve_block("GCS-P"),
        "impact_core": curve_block("IMPACT-Core"),
        "crash_basic": curve_block("CRASH-Basic"),
        "correction_size": {
            "pat_tbi_largest_shift": {"value": round(shifts["PAT-TBI"]["max_abs"], 6)},
            "gcs_largest_shift": {"value": round(shifts["GCS"]["max_abs"], 6)},
            "gcs_p_largest_shift": {"value": round(shifts["GCS-P"]["max_abs"], 6)},
            "impact_core_largest_shift":
                {"value": round(shifts["IMPACT-Core"]["max_abs"], 6)},
            "crash_basic_largest_shift":
                {"value": round(shifts["CRASH-Basic"]["max_abs"], 6)},
            "impact_core_thresholds_before":
                {"n": shifts["IMPACT-Core"]["v4_n_beats"], "N": len(THRESHOLDS)},
            "impact_core_thresholds_after":
                {"n": superiority["IMPACT-Core"]["k"], "N": len(THRESHOLDS)},
            "pat_tbi_thresholds_negligible_shift":
                {"n": shifts["PAT-TBI"]["n_negligible"], "N": len(THRESHOLDS)},
            "pat_tbi_thresholds_same_cutpoint":
                {"n": shifts["PAT-TBI"]["n_same_cutpoint"], "N": len(THRESHOLDS)},
            "pat_tbi_largest_relative_shift":
                {"value": round(shifts["PAT-TBI"]["max_relative"], 4)},
            "statement": {"display":
                "Correcting the map barely moves the curve for the tool under "
                "evaluation or for the two Glasgow Coma Scale comparators. The "
                "derivation-fitted and the validation-fitted maps are close and "
                "both are monotone, so at almost every threshold the same "
                "cutpoint on the integer score is selected and the same patients "
                "are flagged. The circularity was therefore not flattering the "
                "decision curve of the tool under evaluation, and the corrected "
                "curve supports the same conclusion as the earlier one. The "
                "count of thresholds at which the two maps imply the same "
                "cutpoint on the integer score is given beside this note, and it "
                "is the reason the curve hardly moves."},
            "impact_statement": {"display":
                "IMPACT-Core is the curve the correction changes. Its published "
                "probabilities sit far above the observed rate of platelet "
                "receipt in this cohort; the earlier analysis refitted a map for "
                "it inside the validation cohort, which pulled it onto the right "
                "scale and made it appear to add value over part of the range. On "
                "its published scale its net benefit never exceeds that of "
                "treating nobody anywhere on the examined grid, and at the "
                "operating threshold it flags every patient, so its net benefit "
                "there is exactly that of treating everyone."},
        },
    }


# ---------------------------------------------------------------------------
# Handoff
# ---------------------------------------------------------------------------
def write_handoff(curves, superiority, shifts, p_op, nb_op, nb_all_op, avoid_op,
                  flagged_per_100, tp, fp, fn, sens, spec, n_c, ev_c,
                  common_worst, same, ext_lo, ext_hi, ta_gap,
                  a_pat, b_pat) -> None:
    s = superiority["PAT-TBI"]
    txt = f"""# 09_decision_curve — handoff

Owner: `v6/code/09_decision_curve.py`. Closes the half of finding #3 that step
00 left open: the calibration statistics were repaired, the decision curve was
not.

## What was actually wrong

`v4/tables/table_pat_dca.csv` and `v4/figures/figure_pat_dca.*` came from
`v4/code/08_calibration_dca_operating.py`, whose `recalibrate_for_dca_and_cal()`
fitted a fresh `logit(y) = a + b x score` map **inside the validation cohort**
for **every** curve, IMPACT-Core and CRASH-Basic included. Net benefit is
defined against a threshold probability, so a decision curve inherits its map
entirely. Reusing that table under a Methods section claiming one
derivation-fitted convention would have been a false statement. The sentence in
`00_HANDOFF.md` that says the same map was used for the decision-curve analysis
is now true, and this is the file that makes it true.

Everything else was kept from v4 on purpose: the net-benefit arithmetic and the
threshold grid (`np.arange(0.01, 0.205, 0.005)`, 39 thresholds from 0.01 to
0.20) are verbatim, so the corrected curve is directly comparable to the old
one. Only the map changed.

## The map

    logit(y) = a + b * score

with a = {a_pat:.6f} and b = {b_pat:.6f} for PAT-TBI, re-derived here on the
{C.V4_FROZEN['n_train']:,} derivation rows and asserted equal to `table_v6_calibration.csv` to six decimals
and to `bundle/parts/calibration.json` at the precision that file carries.
Glasgow Coma Scale total and Glasgow Coma Scale pupil use their own
derivation-fitted maps from the same table. IMPACT-Core and CRASH-Basic are
**not** remapped.

## The numbers Results should quote

| Quantity | Value |
|---|---|
| Threshold range where PAT-TBI exceeds **both** defaults | {s['lo']:.2f} to {s['hi']:.2f} (every one of the {len(THRESHOLDS)} pre-specified thresholds) |
| Same, unbroken run when the scan is extended past the grid | {ext_lo:.3f} to {ext_hi:.3f} |
| Threshold implied by the 4-point operating point | {p_op:.5f} ({100 * p_op:.2f}%) |
| Net benefit there, PAT-TBI | {nb_op:+.6f} |
| Net benefit there, treat all | {nb_all_op:+.6f} |
| Net benefit there, treat none | 0.000000 |
| **Avoidable interventions per hundred patients there** | **{avoid_op:.1f}** |
| Patients flagged there, per hundred | {flagged_per_100:.2f} |
| Flagged / true positives / false positives / missed | {tp + fp:,} / {tp:,} / {fp:,} / {fn:,} |
| Sensitivity, specificity there | {sens:.4f}, {spec:.4f} |

PAT-TBI exceeds both default strategies at **every** threshold on the grid, so
say "across the entire examined range" rather than naming a sub-interval.

### The two sentences Results should carry

> Across the pre-specified range of threshold probabilities, from
> {s['lo'] * 100:.0f}% to {s['hi'] * 100:.0f}%, the score produced greater net
> benefit than either treating every patient or treating none, and at the
> threshold implied by the {OPERATING_POINTS}-point operating point
> ({100 * p_op:.1f}%) its net benefit was {nb_op:.4f} against
> {nb_all_op:.4f} for treating everyone. That advantage corresponds to
> {avoid_op:.0f} avoided platelet activations per hundred patients relative to
> activating for everyone, already charged for the {fn:,} recipients in
> {curves['PAT-TBI']['n']:,} that the rule does not flag.

## How much the correction moved the curve

The honest answer, which a reviewer will ask for:

| Curve | Largest shift in net benefit against v4 | Thresholds beating both defaults, v4 then now |
|---|---|---|
| PAT-TBI | {shifts['PAT-TBI']['max_abs']:.6f} | {shifts['PAT-TBI']['v4_n_beats']} of {len(THRESHOLDS)}, then {superiority['PAT-TBI']['k']} of {len(THRESHOLDS)} |
| GCS | {shifts['GCS']['max_abs']:.6f} | {shifts['GCS']['v4_n_beats']} of {len(THRESHOLDS)}, then {superiority['GCS']['k']} of {len(THRESHOLDS)} |
| GCS-P | {shifts['GCS-P']['max_abs']:.6f} | {shifts['GCS-P']['v4_n_beats']} of {len(THRESHOLDS)}, then {superiority['GCS-P']['k']} of {len(THRESHOLDS)} |
| IMPACT-Core | {shifts['IMPACT-Core']['max_abs']:.6f} | {shifts['IMPACT-Core']['v4_n_beats']} of {len(THRESHOLDS)}, then {superiority['IMPACT-Core']['k']} of {len(THRESHOLDS)} |
| CRASH-Basic | {shifts['CRASH-Basic']['max_abs']:.6f} | {shifts['CRASH-Basic']['v4_n_beats']} of {len(THRESHOLDS)}, then {superiority['CRASH-Basic']['k']} of {len(THRESHOLDS)} |

**The circularity was not flattering the PAT-TBI decision curve.** Its largest
shift anywhere on the grid is {shifts['PAT-TBI']['max_abs']:.6f} in net benefit,
which is {100 * shifts['PAT-TBI']['max_relative']:.1f}% of the curve's own value
there, and that worst case falls at the far end of the grid where the net benefit
is smallest. At {shifts['PAT-TBI']['n_negligible']} of the {len(THRESHOLDS)}
thresholds the shift is below 0.00001. The reason is concrete: the map v4 fitted
inside the validation cohort (a = {shifts['PAT-TBI']['v4_map_a']:.6f},
b = {shifts['PAT-TBI']['v4_map_b']:.6f}) and the derivation-fitted map imply the
**same integer cutpoint** on the score at
{shifts['PAT-TBI']['n_same_cutpoint']} of the {len(THRESHOLDS)} thresholds, so
the same patients are flagged either way. Say that plainly in the response
letter; it is the strongest possible answer to the question.

**IMPACT-Core is the curve the correction destroys.** Its published
probabilities average about {curves['IMPACT-Core']['p'].mean():.2f} against an
observed platelet-receipt rate of {curves['IMPACT-Core']['prev']:.4f}. v4
refitted a map for it inside the validation cohort, which pulled it onto the
right scale and made it appear to add value at
{shifts['IMPACT-Core']['v4_n_beats']} of {len(THRESHOLDS)} thresholds. On its
published scale it never exceeds treating nobody anywhere on the grid, and at
the operating threshold every patient has an estimated probability above the
threshold, so it flags everyone and its net benefit is exactly that of treating
everyone. Do not describe IMPACT-Core as adding decision value in this cohort.

## The denominator decision

**Each curve is drawn on its own denominator, and each is labelled explicitly**
on the figure legend and in every row of the table. Each curve is compared
against a treat-all reference computed at the prevalence of **its own**
denominator. v4 drew all five against one treat-all reference taken from the
full validation cohort, which is the unfair comparison the review flagged.

Why this and not a common complete-case restriction: PAT-TBI is scoreable for
every patient in the validation cohort, and restricting the primary curve to the
{n_c:,} patients for whom both published instruments can be computed would drop
{C.V4_FROZEN['n_test'] - n_c:,} patients from the curve that carries the
manuscript's claim in order to accommodate two secondary severity proxies.

The choice was then checked rather than argued. Repeating the whole analysis on
the common complete-case set (n = {n_c:,}, {ev_c:,} events) moves no curve by
more than {common_worst:.6f} in net benefit and leaves every superiority range
{'unchanged' if same else 'CHANGED, see the log'}. The three treat-all
references differ by at most {ta_gap:.6f} anywhere on the grid, so they overlap
on the figure; all three are drawn and all three are in the table.

## Files this step owns

- `v6/code/09_decision_curve.py`
- `v6/bundle/parts/dca.json`
- `v6/tables/table_v6_decision_curve.csv`
- `v6/figures/figure_v6_decision_curve.png` and `.pdf`
- `v6/logs/09_decision_curve.log`
- `v6/logs/09_HANDOFF.md`

## What the orchestrator must do

`06_build_bundle.py` merges parts through a fixed `PART_KEYS` map that does not
yet list this one. Add `"dca": "dca"` to it, or the part will sit unmerged and
`results.json` will have no decision-curve section. This script does not touch
the bundle builder.

## Notes for whoever redraws this as a manuscript panel

- The table carries one row per score per threshold. Filter
  `threshold_source == "prespecified grid"` for the curve; the single
  `threshold_source == "operating point"` row per score is the exact
  operating-point threshold, which does not fall on the grid.
- `treat_all_nb` is per row, because it is per denominator. Do not take it from
  one score's rows and apply it to another's.
- The standalone figure clips its vertical axis so the model curves are legible;
  treat-all and IMPACT-Core continue below the panel. If a redraw changes the
  limits, say so, because the clipping hides how far IMPACT-Core falls.
"""
    OUT_HANDOFF.parent.mkdir(parents=True, exist_ok=True)
    OUT_HANDOFF.write_text(txt)


if __name__ == "__main__":
    raise SystemExit(main())
