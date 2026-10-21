"""
PAT-TBI (NTDB 2020-2024) — Step 7: Headline discrimination + calibration
================================================================================

Computes the headline TEST-set metrics for PAT-TBI and all four comparator
scores:

  Part A — Per-score AUROC with subsample percentile bootstrap 95% CI
  Part B — Full pairwise DeLong matrix (5 scores -> 10 pairs), classified by
           head-to-head category (primary_ed_rapid / secondary_ceiling /
           other)
  Part C — Compact calibration summary (CITL, slope, Brier, HL) per score
  Part D — ROC data pickle consumed by Step 8 figure generation

Inputs
------
  PAT_TBI/v4/derived/cohort_test_scored.parquet  (Step 6 output)

Outputs
-------
  PAT_TBI/v4/tables/table_pat_discrimination.csv
  PAT_TBI/v4/tables/table_pat_delong_matrix.csv
  PAT_TBI/v4/tables/table_pat_calibration.csv
  PAT_TBI/v4/derived/pat_roc_data.pkl

Conventions
-----------
  SEED = 42, np rng seeded from 42
  Bootstrap: 2000 iters (default), subsample 100,000 (Sun-Xu fast form)
  DeLong pairwise masks: INTERSECTION of per-comparator masks (never union
    with NaNs); scores must be non-NaN on both sides.
  Raw-score calibration (pat_tbi_score, GCS, GCS-P): fit logistic
    recalibration `logit(y) = a + b*score` on the masked TEST rows to derive
    a probability, then compute CITL / slope / Brier / HL on that derived
    prob. This is an external-cohort re-mapping for calibration evaluation;
    Step 10 re-does this fitting on TRAIN for a sensitivity analysis — not
    the same thing.
  Already-probability scores (IMPACT-Core, CRASH-Basic): use raw prob as-is.

  Absolute paths, banner prints, argparse --bootstraps flag for smoke tests.
"""
from __future__ import annotations

import argparse
import itertools
import os
import pickle
import sys
import time
import warnings
from typing import Tuple

import numpy as np
import pandas as pd

from scipy import stats as sstats
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
N_BOOT_DEFAULT = 2000
BOOT_SUBSAMPLE = 100_000
EPS = 1e-6

import os as _os, sys as _sys  # path handling: make analysis/config.py importable
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import config  # noqa: E402  (analysis/config.py: PAT_TBI_WORKDIR, PAT_TBI_NTDB_DUCKDB)
PAT = str(config.V4_DIR)
DERIVED = os.path.join(PAT, "derived")
TABLES = os.path.join(PAT, "tables")

IN_PARQUET = os.path.join(DERIVED, "cohort_test_scored.parquet")
OUT_DISC = os.path.join(TABLES, "table_pat_discrimination.csv")
OUT_DELONG = os.path.join(TABLES, "table_pat_delong_matrix.csv")
OUT_CALIB = os.path.join(TABLES, "table_pat_calibration.csv")
OUT_PICKLE = os.path.join(DERIVED, "pat_roc_data.pkl")

# Score registry
#   (display name, score column, mask column, is_already_prob, ed_rapid)
SCORES = [
    ("pat_tbi_score",   "pat_tbi_score",    "pat_tbi_score_mask", False, 1),
    ("GCS",             "gcs_risk",         "gcs_mask",           False, 1),
    ("GCS-P",           "gcs_p_risk",       "gcs_p_mask",         False, 1),
    ("IMPACT-Core",     "impact_core_prob", "impact_mask",        True,  0),
    ("CRASH-Basic",     "crash_basic_prob", "crash_mask",         True,  0),
]

# Head-to-head classification for DeLong rows
PRIMARY_ED_RAPID = {("pat_tbi_score", "GCS"), ("pat_tbi_score", "GCS-P")}
SECONDARY_CEILING = {("pat_tbi_score", "CRASH-Basic"), ("pat_tbi_score", "IMPACT-Core")}


def banner(msg: str) -> None:
    line = "=" * 78
    print(f"\n{line}\n{msg}\n{line}", flush=True)


# ---------------------------------------------------------------------------
# Vectorized DeLong (Sun-Xu fast form)
# ---------------------------------------------------------------------------
def compute_midrank(x: np.ndarray) -> np.ndarray:
    """rank with ties averaged (scipy 'average')."""
    return rankdata(x, method="average")


def delong_v10_v01(y: np.ndarray, pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute DeLong placement values V10 (over positives) and V01 (over negatives).

    V10[i] = P(pred_neg < pred_pos[i]) + 0.5 * P(pred_neg == pred_pos[i])
    V01[j] = P(pred_pos > pred_neg[j]) + 0.5 * P(pred_pos == pred_neg[j])

    Derivation: if tx is the midrank of x among its own group and tz is its
    midrank among the combined sample, then
        (tz[pos] - tx[pos]) / n0 = V10
        1 - (tz[neg] - tx[neg]) / n1 = V01
    with 'average' ranks handling ties correctly.
    """
    y = np.asarray(y, dtype=np.int8)
    pred = np.asarray(pred, dtype=np.float64)
    pos = pred[y == 1]
    neg = pred[y == 0]
    n1, n0 = len(pos), len(neg)

    tx1 = compute_midrank(pos)
    tx0 = compute_midrank(neg)
    tz = compute_midrank(pred)
    tz1 = tz[y == 1]
    tz0 = tz[y == 0]

    V10 = (tz1 - tx1) / n0
    V01 = 1.0 - (tz0 - tx0) / n1
    return V10, V01


def delong_test_vec(y: np.ndarray, p1: np.ndarray, p2: np.ndarray):
    """Vectorized DeLong test.

    Returns (z, p, delta, se_delta, auc1, auc2).
    delta = auc1 - auc2 (positive => p1 is better).
    """
    V10_1, V01_1 = delong_v10_v01(y, p1)
    V10_2, V01_2 = delong_v10_v01(y, p2)
    n1, n0 = V10_1.shape[0], V01_1.shape[0]

    auc1 = float(V10_1.mean())
    auc2 = float(V10_2.mean())

    S10 = np.cov(np.vstack([V10_1, V10_2]))
    S01 = np.cov(np.vstack([V01_1, V01_2]))
    S = S10 / n1 + S01 / n0

    delta = auc1 - auc2
    var = float(S[0, 0] + S[1, 1] - 2.0 * S[0, 1])
    if var <= 0.0:
        return 0.0, 1.0, float(delta), 0.0, auc1, auc2
    se = float(np.sqrt(var))
    z = delta / se
    p = 2.0 * (1.0 - sstats.norm.cdf(abs(z)))
    return float(z), float(p), float(delta), se, auc1, auc2


# ---------------------------------------------------------------------------
# Bootstrap CI for AUROC (subsampled for speed)
# ---------------------------------------------------------------------------
def bootstrap_auc_ci_subsample(
    y: np.ndarray,
    s: np.ndarray,
    n_boot: int,
    sub_size: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """Percentile bootstrap CI using subsampling to keep runtime manageable
    on very large cohorts.
    """
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


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------
def hosmer_lemeshow(y_true, y_pred, n_groups=10):
    """Calculate Hosmer-Lemeshow goodness-of-fit statistic."""
    try:
        groups = pd.qcut(y_pred, n_groups, duplicates="drop")
        table = pd.DataFrame({"observed": y_true, "predicted": y_pred, "group": groups})
        agg = table.groupby("group").agg(
            obs_events=("observed", "sum"),
            exp_events=("predicted", "sum"),
            n=("observed", "count"),
        ).reset_index()
        agg["obs_nonevents"] = agg["n"] - agg["obs_events"]
        agg["exp_nonevents"] = agg["n"] - agg["exp_events"]

        agg = agg[(agg["exp_events"] > 0) & (agg["exp_nonevents"] > 0)]

        hl_stat = (
            ((agg["obs_events"] - agg["exp_events"]) ** 2 / agg["exp_events"]).sum()
            + ((agg["obs_nonevents"] - agg["exp_nonevents"]) ** 2 / agg["exp_nonevents"]).sum()
        )
        df = max(len(agg) - 2, 1)
        hl_pval = 1 - sstats.chi2.cdf(hl_stat, df=df)
        return hl_stat, df, hl_pval
    except Exception as exc:
        print(f"  [HL warn] {exc}")
        return np.nan, np.nan, np.nan


def safe_logit(p, eps=EPS):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_logistic_recalibration(y, x):
    """Fit y ~ const + x via statsmodels Logit. Returns (intercept, slope, prob_pred)."""
    X = sm.add_constant(np.asarray(x, dtype=float), has_constant="add")
    model = sm.Logit(np.asarray(y, dtype=int), X)
    res = model.fit(disp=False, method="lbfgs", maxiter=200)
    intercept = float(res.params[0])
    slope = float(res.params[1])
    prob = res.predict(X)
    return intercept, slope, prob


def calibration_slope_via_logit(y, prob):
    """beta from logit(y) = a + b * logit(prob)."""
    lp = safe_logit(prob)
    intercept, slope, _ = fit_logistic_recalibration(y, lp)
    return intercept, slope


def citl(y, prob):
    return float(np.mean(prob) - np.mean(y))


# ---------------------------------------------------------------------------
# Head-to-head classifier
# ---------------------------------------------------------------------------
def classify_pair(name1: str, name2: str) -> str:
    """Return 'primary_ed_rapid' | 'secondary_ceiling' | 'other'.
    Order-insensitive.
    """
    key = (name1, name2)
    key_rev = (name2, name1)
    if key in PRIMARY_ED_RAPID or key_rev in PRIMARY_ED_RAPID:
        return "primary_ed_rapid"
    if key in SECONDARY_CEILING or key_rev in SECONDARY_CEILING:
        return "secondary_ceiling"
    return "other"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstraps", type=int, default=N_BOOT_DEFAULT,
                        help="number of bootstrap iterations for AUROC CI "
                             f"(default {N_BOOT_DEFAULT}); use a small value "
                             "for smoke tests")
    args = parser.parse_args()
    n_boot = int(args.bootstraps)

    t0 = time.time()
    os.makedirs(TABLES, exist_ok=True)
    os.makedirs(DERIVED, exist_ok=True)

    banner("Step 7 — Validation metrics (discrimination + DeLong + calibration)")
    print(f"Cohort window: NTDB 2020-2024", flush=True)
    print(f"Input : {IN_PARQUET}", flush=True)
    print(f"Output: {OUT_DISC}", flush=True)
    print(f"        {OUT_DELONG}", flush=True)
    print(f"        {OUT_CALIB}", flush=True)
    print(f"        {OUT_PICKLE}", flush=True)
    print(f"Bootstrap iterations: {n_boot}", flush=True)

    # -----------------------------------------------------------------------
    # Load
    # -----------------------------------------------------------------------
    banner("Loading test-set scored parquet")
    usecols = ["received_platelets_4h"] + [c for _, c, _, _, _ in SCORES] + \
              [m for _, _, m, _, _ in SCORES]
    df = pd.read_parquet(IN_PARQUET, columns=usecols)
    print(f"  rows: {len(df):,}  events: {int(df['received_platelets_4h'].sum()):,}  "
          f"({100*df['received_platelets_4h'].mean():.2f}%)", flush=True)

    y_all = df["received_platelets_4h"].to_numpy(dtype=np.int8)

    # -----------------------------------------------------------------------
    # Part A — AUROC + bootstrap CI, per score
    # -----------------------------------------------------------------------
    banner(f"Part A — AUROC + bootstrap 95% CI (n_boot={n_boot}, "
           f"sub={BOOT_SUBSAMPLE:,}, seed={SEED})")

    rng = np.random.default_rng(SEED)
    disc_rows = []
    roc_data = {}  # pickle payload
    score_vectors = {}  # name -> (mask_bool, score_float)

    for name, col, mask_col, _is_prob, ed_rapid in SCORES:
        t_s = time.time()
        mask = df[mask_col].to_numpy(dtype=bool)
        s = df[col].to_numpy(dtype=np.float64)
        nan_extra = np.isnan(s)
        use = mask & ~nan_extra
        y_sub = y_all[use]
        s_sub = s[use]

        if use.sum() == 0 or y_sub.sum() == 0 or y_sub.sum() == len(y_sub):
            print(f"  {name:<18}  DEGENERATE — skipping", flush=True)
            disc_rows.append({
                "Score": name,
                "N_used": int(use.sum()),
                "N_events": int(y_sub.sum()),
                "AUROC": np.nan,
                "CI_Low": np.nan,
                "CI_High": np.nan,
                "ED_rapid": ed_rapid,
            })
            score_vectors[name] = (use, s)
            continue

        auc = float(roc_auc_score(y_sub, s_sub))
        lo, hi = bootstrap_auc_ci_subsample(
            y_sub, s_sub, n_boot, BOOT_SUBSAMPLE, rng)

        disc_rows.append({
            "Score": name,
            "N_used": int(use.sum()),
            "N_events": int(y_sub.sum()),
            "AUROC": round(auc, 4),
            "CI_Low": round(lo, 4),
            "CI_High": round(hi, 4),
            "ED_rapid": ed_rapid,
        })
        score_vectors[name] = (use, s)

        # ROC curve — downsample to <=1000 points for plotting speed
        fpr, tpr, _thr = roc_curve(y_sub, s_sub)
        if len(fpr) > 1000:
            idx = np.linspace(0, len(fpr) - 1, 1000).astype(int)
            fpr = fpr[idx]
            tpr = tpr[idx]
        roc_data[name] = {
            "fpr": np.asarray(fpr, dtype=np.float64),
            "tpr": np.asarray(tpr, dtype=np.float64),
            "auc": float(auc),
            "lo": float(lo),
            "hi": float(hi),
        }

        print(f"  {name:<18}  N={use.sum():>9,}  events={int(y_sub.sum()):>7,}  "
              f"AUROC={auc:.4f}  CI=[{lo:.4f}, {hi:.4f}]  "
              f"({time.time()-t_s:5.1f}s)", flush=True)

    disc_df = pd.DataFrame(disc_rows, columns=[
        "Score", "N_used", "N_events", "AUROC", "CI_Low", "CI_High", "ED_rapid",
    ])
    disc_df.to_csv(OUT_DISC, index=False)
    print(f"\n  wrote {OUT_DISC}", flush=True)

    # -----------------------------------------------------------------------
    # Part B — full pairwise DeLong matrix (5 scores -> 10 pairs)
    # -----------------------------------------------------------------------
    banner("Part B — full pairwise DeLong matrix")

    names = [n for n, _, _, _, _ in SCORES]
    pairs = list(itertools.combinations(names, 2))
    delong_rows = []

    for name1, name2 in pairs:
        t_p = time.time()
        use1, s1 = score_vectors[name1]
        use2, s2 = score_vectors[name2]
        # INTERSECTION: both masks True AND both scores non-NaN
        pair_mask = use1 & use2 & ~np.isnan(s1) & ~np.isnan(s2)
        y_pair = y_all[pair_mask]

        if pair_mask.sum() == 0 or y_pair.sum() == 0 or y_pair.sum() == len(y_pair):
            print(f"  {name1:<18} vs {name2:<18}  DEGENERATE — skipping",
                  flush=True)
            delong_rows.append({
                "Score_1": name1,
                "Score_2": name2,
                "N_paired": int(pair_mask.sum()),
                "AUROC_1": np.nan,
                "AUROC_2": np.nan,
                "Delta_AUC": np.nan,
                "SE_Delta": np.nan,
                "CI_Low_Delta": np.nan,
                "CI_High_Delta": np.nan,
                "Z": np.nan,
                "DeLong_P": np.nan,
                "Significant_0.05": False,
                "Head_to_head_class": classify_pair(name1, name2),
            })
            continue

        p1 = s1[pair_mask]
        p2 = s2[pair_mask]
        z, pval, delta, se, auc1, auc2 = delong_test_vec(y_pair, p1, p2)
        ci_lo = delta - 1.96 * se
        ci_hi = delta + 1.96 * se
        h2h = classify_pair(name1, name2)

        delong_rows.append({
            "Score_1": name1,
            "Score_2": name2,
            "N_paired": int(pair_mask.sum()),
            "AUROC_1": round(auc1, 4),
            "AUROC_2": round(auc2, 4),
            "Delta_AUC": round(delta, 4),
            "SE_Delta": round(se, 6),
            "CI_Low_Delta": round(ci_lo, 4),
            "CI_High_Delta": round(ci_hi, 4),
            "Z": round(z, 4),
            "DeLong_P": pval,
            "Significant_0.05": bool(pval < 0.05),
            "Head_to_head_class": h2h,
        })
        print(f"  {name1:<18} vs {name2:<18}  "
              f"[{h2h:<18}]  "
              f"dAUC={delta:+.4f}  SE={se:.5f}  "
              f"Z={z:+.2f}  p={pval:.2e}  "
              f"({time.time()-t_p:4.1f}s)", flush=True)

    delong_df = pd.DataFrame(delong_rows, columns=[
        "Score_1", "Score_2", "N_paired", "AUROC_1", "AUROC_2",
        "Delta_AUC", "SE_Delta", "CI_Low_Delta", "CI_High_Delta",
        "Z", "DeLong_P", "Significant_0.05", "Head_to_head_class",
    ])
    delong_df.to_csv(OUT_DELONG, index=False)
    print(f"\n  wrote {OUT_DELONG}", flush=True)

    # -----------------------------------------------------------------------
    # Part C — compact calibration summary
    # -----------------------------------------------------------------------
    banner("Part C — calibration (CITL / slope / Brier / HL) per score")

    calib_rows = []
    for name, col, mask_col, is_prob, _ed in SCORES:
        use, s = score_vectors[name]
        y_sub = y_all[use].astype(int)
        s_sub = s[use]
        n = int(use.sum())

        if n == 0 or y_sub.sum() == 0 or y_sub.sum() == n:
            print(f"  {name:<18}  DEGENERATE — skipping", flush=True)
            calib_rows.append({
                "Score": name,
                "CITL": np.nan,
                "Calibration_Slope": np.nan,
                "Brier": np.nan,
                "HL_ChiSq": np.nan,
                "HL_df": np.nan,
                "HL_P": np.nan,
            })
            continue

        if is_prob:
            # Already a probability — use as-is. No TEST refit.
            prob = np.clip(s_sub.astype(np.float64), EPS, 1 - EPS)
        else:
            # Raw-score calibration: fit logistic recalibration
            #   logit(y) = a + b * score
            # on the masked TEST rows to derive a prob. This is a standard
            # external-cohort re-mapping for calibration evaluation. Step 10
            # does a separate sensitivity fitting on TRAIN.
            try:
                _a, _b, prob = fit_logistic_recalibration(y_sub, s_sub)
                prob = np.asarray(prob, dtype=np.float64)
            except Exception as exc:
                print(f"  {name:<18}  recalibration failed: {exc}", flush=True)
                calib_rows.append({
                    "Score": name,
                    "CITL": np.nan,
                    "Calibration_Slope": np.nan,
                    "Brier": np.nan,
                    "HL_ChiSq": np.nan,
                    "HL_df": np.nan,
                    "HL_P": np.nan,
                })
                continue

        citl_val = citl(y_sub, prob)
        try:
            _int, cs_slope = calibration_slope_via_logit(y_sub, prob)
        except Exception as exc:
            print(f"  {name:<18}  slope fit failed: {exc}", flush=True)
            cs_slope = np.nan
        brier = float(brier_score_loss(y_sub, prob))
        hl_chi, hl_df, hl_p = hosmer_lemeshow(y_sub, prob, n_groups=10)

        calib_rows.append({
            "Score": name,
            "CITL": round(float(citl_val), 6),
            "Calibration_Slope": round(float(cs_slope), 4)
                if not np.isnan(cs_slope) else np.nan,
            "Brier": round(brier, 6),
            "HL_ChiSq": round(float(hl_chi), 4)
                if not np.isnan(hl_chi) else np.nan,
            "HL_df": int(hl_df) if not np.isnan(hl_df) else np.nan,
            "HL_P": hl_p,
        })
        print(f"  {name:<18}  CITL={citl_val:+.4f}  "
              f"slope={cs_slope:.4f}  Brier={brier:.5f}  "
              f"HL X^2={hl_chi:.2f} (df={hl_df}) p={hl_p:.3g}", flush=True)

    calib_df = pd.DataFrame(calib_rows, columns=[
        "Score", "CITL", "Calibration_Slope", "Brier",
        "HL_ChiSq", "HL_df", "HL_P",
    ])
    calib_df.to_csv(OUT_CALIB, index=False)
    print(f"\n  wrote {OUT_CALIB}", flush=True)

    # -----------------------------------------------------------------------
    # Part D — pickle ROC data for Step 8
    # -----------------------------------------------------------------------
    banner("Part D — pickling ROC data for Step 8")
    with open(OUT_PICKLE, "wb") as f:
        pickle.dump(roc_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  wrote {OUT_PICKLE}  "
          f"({len(roc_data)} scores)", flush=True)

    # -----------------------------------------------------------------------
    # Stdout summary
    # -----------------------------------------------------------------------
    banner("HEADLINE AUROC (Table PAT-TBI discrimination)")
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(disc_df.to_string(index=False))

    banner("PRIMARY ED-RAPID DeLong rows (pat_tbi_score vs GCS / GCS-P)")
    primary = delong_df[delong_df["Head_to_head_class"] == "primary_ed_rapid"]
    with pd.option_context("display.width", 200, "display.max_columns", None,
                           "display.float_format", lambda v: f"{v:.4g}"):
        print(primary.to_string(index=False))

    banner("SECONDARY CEILING DeLong rows (pat_tbi_score vs IMPACT-Core / CRASH-Basic)")
    secondary = delong_df[delong_df["Head_to_head_class"] == "secondary_ceiling"]
    with pd.option_context("display.width", 200, "display.max_columns", None,
                           "display.float_format", lambda v: f"{v:.4g}"):
        print(secondary.to_string(index=False))

    banner("CALIBRATION")
    with pd.option_context("display.width", 160, "display.max_columns", None,
                           "display.float_format", lambda v: f"{v:.4g}"):
        print(calib_df.to_string(index=False))

    print(f"\nTotal runtime: {time.time()-t0:.1f}s", flush=True)
    print("Step 7 complete.", flush=True)


if __name__ == "__main__":
    main()
