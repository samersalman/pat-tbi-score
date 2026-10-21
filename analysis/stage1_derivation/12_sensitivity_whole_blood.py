"""
PAT-TBI (NTDB 2020-2024) — Step 12: Sensitivity to outcome definition
================================================================================

Applies the unchanged integer-point PAT-TBI score (Step 5 weights) to the
held-out TEST set under a broader transfusion-receipt outcome that includes
whole blood:

    received_platelets_or_wholeblood_4h = (PLATELETS4HOURS > 0) OR
                                          (WHOLEBLOOD4HOURS > 0)

No re-derivation, no refit. Reports AUROC with bootstrap 95% CI, Brier,
calibration-in-the-large (CITL), calibration slope, and Hosmer-Lemeshow on
the combined endpoint, and — for context — the same metrics on the primary
outcome (received_platelets_4h) on the same patients. The pair of rows lets
the manuscript show robustness to outcome definition.

Inputs
------
  PAT_TBI/v4/derived/candidates.parquet      (Step 2 — has both outcomes + Xs)
  PAT_TBI/v4/derived/test_inc_keys.csv       (Step 3 — held-out test inc_keys)
  PAT_TBI/v4/tables/table_pat_weights.csv    (Step 5 — final integer weights)

Outputs
-------
  PAT_TBI/v4/tables/table_sensitivity_whole_blood.csv
    One row per (Outcome, metric set):
      Outcome, N, N_events, Event_Rate, AUROC, CI_Low, CI_High,
      Brier, CITL, Calibration_Slope, HL_ChiSq, HL_df, HL_P
  PAT_TBI/v4/figures/figure_sensitivity_whole_blood_calibration.{png,pdf}
    Calibration overlay: primary outcome vs combined outcome.

Conventions
-----------
  SEED = 42, np rng seeded from 42
  Bootstrap: 2000 iters (default; --bootstraps N for smoke), full-cohort
    resampling with replacement.
  Calibration: fit logistic recalibration `logit(y) = a + b * pat_tbi_score`
    on the TEST rows to derive a probability, then compute CITL / slope /
    Brier / HL on that derived prob. Identical convention to Step 7 for
    raw integer scores (an external-cohort re-mapping for calibration
    evaluation, not a refit of the score).

Helpers (copied, not imported, to keep the step file self-contained per
project convention):
  - compute_pat_tbi : adapted from Step 6 (integer-point score from weights CSV)
  - bootstrap_auc_ci : full-cohort bootstrap (Step 7's helper, full N rather
                       than subsampled — N≈260K test rows is tractable)
  - hosmer_lemeshow, safe_logit, fit_logistic_recalibration,
    calibration_slope_via_logit, citl : copied from Step 7
"""
from __future__ import annotations

import argparse
import os
import time
import warnings
from typing import Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats as sstats
from sklearn.metrics import roc_auc_score, brier_score_loss
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config / paths
# ---------------------------------------------------------------------------
SEED = 42
N_BOOT_DEFAULT = 2000
EPS = 1e-6

import os as _os, sys as _sys  # path handling: make analysis/config.py importable
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import config  # noqa: E402  (analysis/config.py: PAT_TBI_WORKDIR, PAT_TBI_NTDB_DUCKDB)
PAT = str(config.V4_DIR)
DERIVED = os.path.join(PAT, "derived")
TABLES = os.path.join(PAT, "tables")
FIGURES = os.path.join(PAT, "figures")

CANDIDATES_PARQUET = os.path.join(DERIVED, "candidates.parquet")
TEST_KEYS_CSV = os.path.join(DERIVED, "test_inc_keys.csv")
PAT_WEIGHTS_CSV = os.path.join(TABLES, "table_pat_weights.csv")

OUT_TABLE = os.path.join(TABLES, "table_sensitivity_whole_blood.csv")
OUT_FIG_PNG = os.path.join(FIGURES, "figure_sensitivity_whole_blood_calibration.png")
OUT_FIG_PDF = os.path.join(FIGURES, "figure_sensitivity_whole_blood_calibration.pdf")

PRIMARY_OUTCOME = "received_platelets_4h"
COMBINED_OUTCOME = "received_platelets_or_wholeblood_4h"


def banner(msg: str) -> None:
    line = "=" * 78
    print(f"\n{line}\n{msg}\n{line}", flush=True)


# ============================================================================
# Helper — apply PAT-TBI integer-point score (adapted from Step 6)
# ============================================================================
def compute_pat_tbi(df: pd.DataFrame, weights_csv: str) -> Tuple[np.ndarray, list]:
    """Apply the PAT-TBI refit integer weights to df.

    The PAT-TBI weights CSV has columns (Variable, Points, ...). We:
      - drop blank / zero-point rows
      - assert every Variable is a column of df
      - return score = sum_i Points_i * df[Variable_i].astype(float)
    All candidate vars are pure binaries pre-built in Step 2, so no rows
    require dropping.
    """
    print(f"  Loading PAT-TBI weights from {weights_csv}", flush=True)
    w = pd.read_csv(weights_csv)
    w = w.dropna(subset=["Variable"])
    w = w[w["Variable"].astype(str).str.strip() != ""].copy()
    w["Points"] = pd.to_numeric(w["Points"], errors="coerce").fillna(0).astype(int)
    w = w[w["Points"] != 0].copy()
    print(f"  PAT-TBI has {len(w)} non-zero predictors", flush=True)

    if len(w) == 0:
        raise RuntimeError("PAT-TBI weights table has zero usable predictors")

    score = np.zeros(len(df), dtype=np.float64)
    used = []
    for _, row in w.iterrows():
        var = str(row["Variable"]).strip()
        pts = int(row["Points"])
        if var not in df.columns:
            raise RuntimeError(
                f"PAT-TBI predictor '{var}' missing from candidate parquet")
        score += pts * df[var].astype(np.float64).values
        used.append((var, pts))

    return score.astype(np.int64), used


# ============================================================================
# Helper — bootstrap AUROC CI (full-cohort resampling)
# ============================================================================
def bootstrap_auc_ci(
    y: np.ndarray,
    s: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for AUROC using full-cohort resampling
    with replacement. The held-out TEST set has ~260K rows, which is
    tractable for full-cohort bootstrap (Step 7 uses subsampling because
    it iterates over multiple scores; here we have only one score).
    """
    n = len(y)
    aucs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == n:
            aucs[b] = np.nan
            continue
        aucs[b] = roc_auc_score(yb, s[idx])
    aucs = aucs[~np.isnan(aucs)]
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)


# ============================================================================
# Calibration helpers (copied from Step 7)
# ============================================================================
def hosmer_lemeshow(y_true, y_pred, n_groups=10):
    """Hosmer-Lemeshow goodness-of-fit (deciles of predicted prob)."""
    try:
        groups = pd.qcut(y_pred, n_groups, duplicates="drop")
        table = pd.DataFrame({"observed": y_true, "predicted": y_pred,
                              "group": groups})
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
            + ((agg["obs_nonevents"] - agg["exp_nonevents"]) ** 2
               / agg["exp_nonevents"]).sum()
        )
        df = max(len(agg) - 2, 1)
        hl_pval = 1 - sstats.chi2.cdf(hl_stat, df=df)
        return float(hl_stat), int(df), float(hl_pval)
    except Exception as exc:
        print(f"  [HL warn] {exc}", flush=True)
        return np.nan, np.nan, np.nan


def safe_logit(p, eps=EPS):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_logistic_recalibration(y, x):
    """Fit y ~ const + x via statsmodels Logit. Returns (a, b, prob_pred)."""
    X = sm.add_constant(np.asarray(x, dtype=float), has_constant="add")
    model = sm.Logit(np.asarray(y, dtype=int), X)
    res = model.fit(disp=False, method="lbfgs", maxiter=200)
    intercept = float(res.params[0])
    slope = float(res.params[1])
    prob = res.predict(X)
    return intercept, slope, np.asarray(prob, dtype=np.float64)


def calibration_slope_via_logit(y, prob):
    """beta from logit(y) = a + b * logit(prob)."""
    lp = safe_logit(prob)
    intercept, slope, _ = fit_logistic_recalibration(y, lp)
    return float(intercept), float(slope)


def citl(y, prob):
    return float(np.mean(prob) - np.mean(y))


# ============================================================================
# Per-outcome metrics
# ============================================================================
def evaluate_outcome(
    name: str,
    y: np.ndarray,
    score: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> Tuple[dict, np.ndarray, np.ndarray]:
    """Compute discrimination + calibration metrics for one outcome.

    Returns (row_dict, prob_recal, decile_calibration_xy).
    decile_calibration_xy is a 2-tuple (mean_pred_per_decile, obs_rate_per_decile)
    for the calibration figure; computed from the recalibrated probabilities.
    """
    n = int(len(y))
    n_events = int(y.sum())
    rate = float(n_events) / n if n > 0 else float("nan")
    print(f"\n  Outcome: {name}", flush=True)
    print(f"    N={n:,}  events={n_events:,}  rate={100*rate:.2f}%", flush=True)

    if n == 0 or n_events == 0 or n_events == n:
        print(f"    DEGENERATE — returning NaN row", flush=True)
        return ({
            "Outcome": name,
            "N": n,
            "N_events": n_events,
            "Event_Rate": rate,
            "AUROC": np.nan,
            "CI_Low": np.nan,
            "CI_High": np.nan,
            "Brier": np.nan,
            "CITL": np.nan,
            "Calibration_Slope": np.nan,
            "HL_ChiSq": np.nan,
            "HL_df": np.nan,
            "HL_P": np.nan,
        }, np.full(n, np.nan), (np.array([]), np.array([])))

    t0 = time.time()
    auc = float(roc_auc_score(y, score))
    print(f"    AUROC = {auc:.4f}  (computed in {time.time()-t0:.1f}s)",
          flush=True)

    t0 = time.time()
    lo, hi = bootstrap_auc_ci(y, score.astype(np.float64), n_boot, rng)
    print(f"    95% CI = [{lo:.4f}, {hi:.4f}]  "
          f"(bootstrap n={n_boot}, {time.time()-t0:.1f}s)", flush=True)

    # Calibration via logistic recalibration on the integer score
    t0 = time.time()
    a, b, prob = fit_logistic_recalibration(y, score.astype(np.float64))
    print(f"    recalibration: intercept={a:+.4f}  slope={b:+.4f}  "
          f"({time.time()-t0:.1f}s)", flush=True)

    citl_val = citl(y, prob)
    try:
        _int, slope = calibration_slope_via_logit(y, prob)
    except Exception as exc:
        print(f"    slope fit failed: {exc}", flush=True)
        slope = np.nan
    brier = float(brier_score_loss(y, prob))
    hl_chi, hl_df, hl_p = hosmer_lemeshow(y, prob, n_groups=10)
    print(f"    Brier={brier:.5f}  CITL={citl_val:+.4f}  "
          f"slope={slope:.4f}  "
          f"HL X^2={hl_chi:.2f} (df={hl_df}) p={hl_p:.3g}", flush=True)

    # Decile calibration for the figure
    try:
        df_cal = pd.DataFrame({"y": y.astype(int), "p": prob})
        df_cal["decile"] = pd.qcut(df_cal["p"], 10, duplicates="drop",
                                   labels=False)
        agg = df_cal.groupby("decile").agg(
            mean_pred=("p", "mean"),
            obs_rate=("y", "mean"),
            n=("y", "size"),
        ).reset_index()
        xy = (agg["mean_pred"].values, agg["obs_rate"].values)
    except Exception as exc:
        print(f"    decile calibration failed: {exc}", flush=True)
        xy = (np.array([]), np.array([]))

    row = {
        "Outcome": name,
        "N": n,
        "N_events": n_events,
        "Event_Rate": round(rate, 6),
        "AUROC": round(auc, 4),
        "CI_Low": round(lo, 4),
        "CI_High": round(hi, 4),
        "Brier": round(brier, 6),
        "CITL": round(float(citl_val), 6),
        "Calibration_Slope": round(float(slope), 4)
            if not np.isnan(slope) else np.nan,
        "HL_ChiSq": round(float(hl_chi), 4)
            if not np.isnan(hl_chi) else np.nan,
        "HL_df": int(hl_df) if not np.isnan(hl_df) else np.nan,
        "HL_P": hl_p,
    }
    return row, prob, xy


# ============================================================================
# Calibration figure (primary vs combined overlay)
# ============================================================================
def plot_calibration_overlay(xy_primary, xy_combined, out_png, out_pdf):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    # Reference 45-deg line
    ax.plot([0, 1], [0, 1], color="0.5", linestyle="--", linewidth=1.0,
            label="Perfect calibration")

    if len(xy_primary[0]) > 0:
        ax.plot(xy_primary[0], xy_primary[1],
                marker="o", linestyle="-", color="#1f77b4", linewidth=1.5,
                markersize=6, label="Primary outcome (platelets only)")
    if len(xy_combined[0]) > 0:
        ax.plot(xy_combined[0], xy_combined[1],
                marker="s", linestyle="-", color="#d62728", linewidth=1.5,
                markersize=6, label="Combined outcome (platelets or whole blood)")

    # Tight axes around observed range
    all_x = np.concatenate([
        xy_primary[0] if len(xy_primary[0]) else np.array([0.0]),
        xy_combined[0] if len(xy_combined[0]) else np.array([0.0]),
    ])
    all_y = np.concatenate([
        xy_primary[1] if len(xy_primary[1]) else np.array([0.0]),
        xy_combined[1] if len(xy_combined[1]) else np.array([0.0]),
    ])
    upper = float(max(0.05, np.nanmax(np.concatenate([all_x, all_y])) * 1.1))
    upper = min(upper, 1.0)
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)

    ax.set_xlabel("Predicted probability (decile mean)")
    ax.set_ylabel("Observed event rate")
    ax.set_title("Calibration — PAT-TBI applied to platelets vs platelets+whole-blood")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}", flush=True)
    print(f"  wrote {out_pdf}", flush=True)


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="PAT-TBI Step 12 — sensitivity to outcome definition "
                    "(NTDB 2020-2024)")
    parser.add_argument("--bootstraps", type=int, default=N_BOOT_DEFAULT,
                        help="bootstrap iterations for AUROC 95% CI "
                             f"(default {N_BOOT_DEFAULT}); use a small "
                             "value for smoke tests")
    parser.add_argument("--no-figure", action="store_true",
                        help="skip the calibration overlay figure")
    args = parser.parse_args()
    n_boot = int(args.bootstraps)

    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    os.makedirs(TABLES, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)

    banner("Step 12 — Sensitivity: whole-blood combined outcome (NTDB 2020-2024)")
    print(f"Candidates : {CANDIDATES_PARQUET}", flush=True)
    print(f"Test keys  : {TEST_KEYS_CSV}", flush=True)
    print(f"PAT weights: {PAT_WEIGHTS_CSV}", flush=True)
    print(f"Output CSV : {OUT_TABLE}", flush=True)
    if not args.no_figure:
        print(f"Output fig : {OUT_FIG_PNG}", flush=True)
        print(f"             {OUT_FIG_PDF}", flush=True)
    print(f"Bootstrap iterations: {n_boot}", flush=True)

    # ------------------------------------------------------------------
    # 1. Load candidates + test inc_keys, inner-merge to TEST slice
    # ------------------------------------------------------------------
    banner("Loading candidates + test inc_keys (inner merge)")
    print("Reading candidate parquet...", flush=True)
    df_all = pd.read_parquet(CANDIDATES_PARQUET)
    print(f"  candidates: {df_all.shape}", flush=True)

    print("Reading test inc_keys...", flush=True)
    test_keys = pd.read_csv(TEST_KEYS_CSV)
    print(f"  test_inc_keys: {test_keys.shape}", flush=True)

    for frame in (df_all, test_keys):
        if "inc_key" in frame.columns:
            frame["inc_key"] = pd.to_numeric(
                frame["inc_key"], errors="coerce").astype("Int64")
        if "PUF_YEAR" in frame.columns:
            frame["PUF_YEAR"] = pd.to_numeric(
                frame["PUF_YEAR"], errors="coerce").astype("Int64")

    if {"PUF_YEAR", "inc_key"}.issubset(test_keys.columns):
        merge_on = ["PUF_YEAR", "inc_key"]
    else:
        merge_on = ["inc_key"]
    print(f"  merging on: {merge_on}", flush=True)

    df = test_keys.merge(df_all, on=merge_on, how="inner", validate="1:1")
    print(f"  test slice: {df.shape}", flush=True)

    if len(df) != len(test_keys):
        raise RuntimeError(
            f"Test-slice merge length {len(df):,} != test_inc_keys "
            f"length {len(test_keys):,}. Missing keys in candidate parquet?")
    print(f"  ASSERT pass: len(merged) == len(test_inc_keys) == {len(df):,}",
          flush=True)

    # ------------------------------------------------------------------
    # 2. Coerce outcomes to int 0/1
    # ------------------------------------------------------------------
    banner("Coercing outcomes")
    for col in (PRIMARY_OUTCOME, COMBINED_OUTCOME):
        if col not in df.columns:
            raise RuntimeError(f"{col} column missing from candidate parquet")
        s = pd.to_numeric(df[col], errors="coerce")
        n_nan = int(s.isna().sum())
        if n_nan > 0:
            # 2020-2024 cohort has WHOLEBLOOD4HOURS available all years per
            # plan, so this should not happen — fail loud rather than silently
            # impute.
            raise RuntimeError(
                f"{col} has {n_nan:,} NaN in test slice — outcome must be int "
                f"0/1 across the full TEST set")
        df[col] = s.astype(int)
    n_total = len(df)
    n_prim = int(df[PRIMARY_OUTCOME].sum())
    n_comb = int(df[COMBINED_OUTCOME].sum())
    print(f"  N={n_total:,}", flush=True)
    print(f"  primary  events: {n_prim:,}  ({100*n_prim/n_total:.2f}%)",
          flush=True)
    print(f"  combined events: {n_comb:,}  ({100*n_comb/n_total:.2f}%)",
          flush=True)
    n_extra = n_comb - n_prim
    print(f"  whole-blood-only additional events: {n_extra:,}", flush=True)
    if n_comb < n_prim:
        raise RuntimeError(
            f"Combined outcome ({n_comb:,}) is LESS than primary ({n_prim:,}); "
            "outcome definition error upstream — combined is a superset.")

    # ------------------------------------------------------------------
    # 3. Compute the unchanged PAT-TBI integer-point score
    # ------------------------------------------------------------------
    banner("Computing PAT-TBI integer score (unchanged Step 5 weights)")
    score, used = compute_pat_tbi(df, PAT_WEIGHTS_CSV)
    print(f"  weights used: {used}", flush=True)
    print(f"  score: min={int(score.min())}  max={int(score.max())}  "
          f"mean={float(score.mean()):.3f}  "
          f"median={float(np.median(score)):.3f}", flush=True)

    # ------------------------------------------------------------------
    # 4. Per-outcome metrics
    # ------------------------------------------------------------------
    banner("Evaluating PAT-TBI on primary and combined outcomes")
    rows = []
    xy_dict = {}
    for outcome_label, outcome_col in (
        ("received_platelets_4h (primary)", PRIMARY_OUTCOME),
        ("received_platelets_or_wholeblood_4h (combined)", COMBINED_OUTCOME),
    ):
        y = df[outcome_col].to_numpy(dtype=np.int8)
        row, _prob, xy = evaluate_outcome(
            outcome_label, y, score, n_boot, rng)
        rows.append(row)
        xy_dict[outcome_col] = xy

    out_df = pd.DataFrame(rows, columns=[
        "Outcome", "N", "N_events", "Event_Rate",
        "AUROC", "CI_Low", "CI_High",
        "Brier", "CITL", "Calibration_Slope",
        "HL_ChiSq", "HL_df", "HL_P",
    ])

    # ------------------------------------------------------------------
    # 5. Write outputs
    # ------------------------------------------------------------------
    banner("Writing sensitivity table")
    out_df.to_csv(OUT_TABLE, index=False)
    print(f"  wrote {OUT_TABLE}", flush=True)

    if not args.no_figure:
        banner("Writing calibration overlay figure")
        plot_calibration_overlay(
            xy_dict[PRIMARY_OUTCOME],
            xy_dict[COMBINED_OUTCOME],
            OUT_FIG_PNG,
            OUT_FIG_PDF,
        )

    # ------------------------------------------------------------------
    # 6. Stdout summary
    # ------------------------------------------------------------------
    banner("SENSITIVITY — outcome definition robustness")
    with pd.option_context("display.width", 200, "display.max_columns", None,
                           "display.float_format", lambda v: f"{v:.4g}"):
        print(out_df.to_string(index=False), flush=True)

    print("\nStep 12 complete.", flush=True)


if __name__ == "__main__":
    main()
