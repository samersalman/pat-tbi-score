"""
PAT-TBI NTDB Refit — Step 8: Calibration, DCA, Operating Characteristics, ROC
==========================================================================================

Consumes the PAT-TBI test-split scored file (Step 6) and produces the confirmatory
evidence artifacts for the PAT-TBI manuscript:

  1. Calibration  — 2x3 grid of reliability curves (5 scores + summary panel)
  2. DCA          — Vickers-Elkin decision curve across p_t = 0.01..0.20 step 0.005
                    (pat_tbi_score bold + treat-all / treat-none references)
  3. OpChar       — sens/spec/PPV/NPV/LR+/LR-/F1/Youden J with Wilson 95% CIs
                    reported at THREE cutoff rules per score:
                      - youden   (argmax tpr - fpr)
                      - sens_90  (lowest threshold giving sensitivity >= 0.90)
                      - spec_90  (highest threshold giving specificity >= 0.90)
  4. ROC overlay  — loads pat_roc_data.pkl from Step 7 if present, else recomputes
  5. Score distribution histogram (PAT-TBI integer score by event status)

All four comparators use per-comparator masks from Step 6.

Cohort years: 2020-2024.

Inputs
------
  PAT_TBI/v4/derived/cohort_test_scored.parquet   (Step 6)
  PAT_TBI/v4/derived/pat_roc_data.pkl             (Step 7, optional)

Outputs
-------
  PAT_TBI/v4/tables/table_pat_dca.csv
  PAT_TBI/v4/tables/table_pat_operating_characteristics.csv
  PAT_TBI/v4/figures/figure_pat_dca.{pdf,png}
  PAT_TBI/v4/figures/figure_pat_calibration.{pdf,png}
  PAT_TBI/v4/figures/figure_pat_roc.{pdf,png}
  PAT_TBI/v4/figures/figure_pat_score_distribution.{pdf,png}

Score columns (from Step 6)
---------------------------
  pat_tbi_score        (integer refit score)     + pat_tbi_score_mask
  gcs_risk             (continuous)              + gcs_mask
  gcs_p_risk           (continuous)              + gcs_p_mask
  impact_core_prob     (published probability)   + impact_mask
  crash_basic_prob     (published probability)   + crash_mask

Conventions
-----------
  SEED = 42
  matplotlib Agg backend set BEFORE pyplot import
  Absolute paths, banner prints
  --sample is a no-op (kept for CLI parity with earlier steps)
"""
import argparse
import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # non-interactive backend BEFORE pyplot
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint
from statsmodels.nonparametric.smoothers_lowess import lowess

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
EPS = 1e-6

import os as _os, sys as _sys  # path handling: make analysis/config.py importable
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import config  # noqa: E402  (analysis/config.py: PAT_TBI_WORKDIR, PAT_TBI_NTDB_DUCKDB)
PAT = str(config.V4_DIR)
DERIVED = os.path.join(PAT, "derived")
TABLES = os.path.join(PAT, "tables")
FIGURES = os.path.join(PAT, "figures")
os.makedirs(TABLES, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

IN_PARQUET = os.path.join(DERIVED, "cohort_test_scored.parquet")
ROC_PKL = os.path.join(DERIVED, "pat_roc_data.pkl")

OUT_TABLE_DCA = os.path.join(TABLES, "table_pat_dca.csv")
OUT_TABLE_OPCHAR = os.path.join(TABLES, "table_pat_operating_characteristics.csv")

OUT_FIG_DCA_PNG = os.path.join(FIGURES, "figure_pat_dca.png")
OUT_FIG_DCA_PDF = os.path.join(FIGURES, "figure_pat_dca.pdf")
OUT_FIG_CAL_PNG = os.path.join(FIGURES, "figure_pat_calibration.png")
OUT_FIG_CAL_PDF = os.path.join(FIGURES, "figure_pat_calibration.pdf")
OUT_FIG_ROC_PNG = os.path.join(FIGURES, "figure_pat_roc.png")
OUT_FIG_ROC_PDF = os.path.join(FIGURES, "figure_pat_roc.pdf")
OUT_FIG_DIST_PNG = os.path.join(FIGURES, "figure_pat_score_distribution.png")
OUT_FIG_DIST_PDF = os.path.join(FIGURES, "figure_pat_score_distribution.pdf")

# DCA threshold range 1-20%, step 0.005 (39 thresholds)
THRESHOLDS = np.arange(0.01, 0.205, 0.005)

# Primary score specs
# (label, score_col, mask_col, is_already_prob, is_integer_score)
SCORE_SPECS = [
    ('PAT-TBI',      'pat_tbi_score',    'pat_tbi_score_mask',    False, True),
    ('GCS',          'gcs_risk',         'gcs_mask',         False, False),
    ('GCS-P',        'gcs_p_risk',       'gcs_p_mask',       False, False),
    ('IMPACT-Core',  'impact_core_prob', 'impact_mask', True,  False),
    ('CRASH-Basic',  'crash_basic_prob', 'crash_mask', True,  False),
]

COLORS = {
    'PAT-TBI':     '#D62728',
    'GCS':         '#1F77B4',
    'GCS-P':       '#2CA02C',
    'IMPACT-Core': '#9467BD',
    'CRASH-Basic': '#FF7F0E',
}
LINEWIDTH = {
    'PAT-TBI':     2.6,
    'GCS':         1.8,
    'GCS-P':       1.8,
    'IMPACT-Core': 1.8,
    'CRASH-Basic': 1.8,
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def logistic_recalibrate(y, score):
    """Fit sm.Logit(y ~ const + score) on the cohort; return (intercept, slope, probs)."""
    X = sm.add_constant(np.asarray(score, dtype=float))
    model = sm.Logit(np.asarray(y, dtype=int), X).fit(disp=False, maxiter=200)
    probs = model.predict(X)
    intercept = float(model.params[0])
    slope = float(model.params[1])
    return intercept, slope, np.clip(probs, EPS, 1.0 - EPS)


def net_benefit(y, prob, p_t):
    """
    Vickers-Elkin net benefit at threshold p_t.
    n = len(y) (the full mask-filtered denominator for this score).
    """
    n = len(y)
    if n == 0:
        return np.nan
    pred_pos = prob >= p_t
    tp = float(((pred_pos) & (y == 1)).sum()) / n
    fp = float(((pred_pos) & (y == 0)).sum()) / n
    if p_t >= 1.0:
        return np.nan
    return tp - fp * (p_t / (1.0 - p_t))


def wilson_ci(count, nobs):
    """Wilson score interval via statsmodels."""
    if nobs == 0:
        return (np.nan, np.nan)
    lo, hi = proportion_confint(count=count, nobs=nobs, alpha=0.05, method="wilson")
    return (lo, hi)


def youden_threshold(y_true, score):
    """Return threshold maximizing Youden's J (tpr - fpr)."""
    fpr, tpr, thresholds = roc_curve(y_true, score)
    J = tpr - fpr
    best_idx = int(np.argmax(J))
    return float(thresholds[best_idx]), float(J[best_idx])


def compute_metrics(y_true, score, cutoff, score_name, cutoff_type):
    """Compute operating characteristics for a score thresholded at cutoff (>=)."""
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    flag = (score >= cutoff).astype(int)

    TP = int(((flag == 1) & (y_true == 1)).sum())
    FP = int(((flag == 1) & (y_true == 0)).sum())
    TN = int(((flag == 0) & (y_true == 0)).sum())
    FN = int(((flag == 0) & (y_true == 1)).sum())
    N = TP + FP + TN + FN
    N_flagged = TP + FP

    sens = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    spec = TN / (TN + FP) if (TN + FP) > 0 else np.nan
    ppv = TP / (TP + FP) if (TP + FP) > 0 else np.nan
    npv = TN / (TN + FN) if (TN + FN) > 0 else np.nan
    acc = (TP + TN) / N if N > 0 else np.nan
    lr_pos = sens / (1 - spec) if (spec is not None and (1 - spec) > 0) else np.inf
    lr_neg = (1 - sens) / spec if (spec is not None and spec > 0) else np.nan
    f1 = (
        2 * ppv * sens / (ppv + sens)
        if (ppv is not None and sens is not None and (ppv + sens) > 0)
        else np.nan
    )

    sens_lo, sens_hi = wilson_ci(TP, TP + FN)
    spec_lo, spec_hi = wilson_ci(TN, TN + FP)
    ppv_lo, ppv_hi = wilson_ci(TP, TP + FP)
    npv_lo, npv_hi = wilson_ci(TN, TN + FN)

    return {
        "Score": score_name,
        "Cutoff_Type": cutoff_type,
        "Cutoff_Value": cutoff,
        "N": N,
        "N_flagged": N_flagged,
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
        "Sensitivity": sens,
        "Sens_CI_Low": sens_lo,
        "Sens_CI_High": sens_hi,
        "Specificity": spec,
        "Spec_CI_Low": spec_lo,
        "Spec_CI_High": spec_hi,
        "PPV": ppv,
        "PPV_CI_Low": ppv_lo,
        "PPV_CI_High": ppv_hi,
        "NPV": npv,
        "NPV_CI_Low": npv_lo,
        "NPV_CI_High": npv_hi,
        "Accuracy": acc,
        "LR_Positive": lr_pos,
        "LR_Negative": lr_neg,
        "F1": f1,
    }


# -----------------------------------------------------------------------------
# Local helpers
# -----------------------------------------------------------------------------
def banner(msg):
    line = "=" * 78
    print(f"\n{line}\n{msg}\n{line}", flush=True)


def nb_treat_all(prevalence, p_t):
    if p_t >= 1.0:
        return np.nan
    return prevalence - (1.0 - prevalence) * (p_t / (1.0 - p_t))


def citl_and_slope(y, prob):
    """Calibration-in-the-large + calibration slope."""
    p = np.clip(prob, EPS, 1.0 - EPS)
    citl = float(p.mean() - y.mean())
    logit_p = np.log(p / (1.0 - p))
    X = sm.add_constant(logit_p)
    try:
        m = sm.Logit(y.astype(int), X).fit(disp=False, maxiter=200)
        slope = float(m.params[1])
    except Exception:
        slope = np.nan
    return citl, slope


def sens_90_threshold(y_true, score):
    """Lowest threshold giving sensitivity >= 0.90 (on roc_curve grid).

    Returns (threshold, achieved_sens, achieved_spec). NaN if unreachable.
    """
    fpr, tpr, thr = roc_curve(y_true, score)
    ok = np.where(tpr >= 0.90)[0]
    if len(ok) == 0:
        return np.nan, np.nan, np.nan
    # roc_curve returns thresholds in descending order; lowest threshold with
    # sens>=0.90 is the largest index in `ok`.
    idx = int(ok.max())
    return float(thr[idx]), float(tpr[idx]), float(1.0 - fpr[idx])


def spec_90_threshold(y_true, score):
    """Highest threshold giving specificity >= 0.90.

    Returns (threshold, achieved_sens, achieved_spec). NaN if unreachable.
    """
    fpr, tpr, thr = roc_curve(y_true, score)
    spec = 1.0 - fpr
    ok = np.where(spec >= 0.90)[0]
    if len(ok) == 0:
        return np.nan, np.nan, np.nan
    # Highest threshold with spec>=0.90 is the smallest index in `ok`
    # (roc_curve thresholds are descending).
    idx = int(ok.min())
    return float(thr[idx]), float(tpr[idx]), float(spec[idx])


def load_score_data(df, spec):
    """Filter by mask, drop NaNs, return (y, s, n, prev, label, is_prob, is_int)."""
    label, score_col, mask_col, is_prob, is_int = spec
    m = df[mask_col].astype(bool).values
    sub = df.loc[m, [score_col, 'received_platelets_4h']].copy()
    sub = sub.dropna(subset=[score_col, 'received_platelets_4h'])
    y = sub['received_platelets_4h'].astype(int).values
    s = sub[score_col].astype(float).values
    return {
        'label': label,
        'y': y,
        'score': s,
        'n': len(y),
        'prev': float(y.mean()) if len(y) else np.nan,
        'is_prob': is_prob,
        'is_int': is_int,
    }


def recalibrate_for_dca_and_cal(data):
    """For DCA + calibration plots we need probs for EVERY score on a common scale.

    For raw scores (pat_tbi_score, gcs_risk, gcs_p_risk): fit logistic recalibration.
    For published probs (impact_core_prob, crash_basic_prob): also fit logistic
    recalibration for a fair DCA comparison.
    """
    y = data['y']
    s = data['score']
    if len(y) == 0:
        data['prob'] = np.array([])
        data['intercept'] = np.nan
        data['slope'] = np.nan
        return data
    intercept, slope, probs = logistic_recalibrate(y, s)
    data['prob'] = probs
    data['intercept'] = intercept
    data['slope'] = slope
    return data


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0,
                        help="No-op kept for CLI parity with earlier PAT-TBI steps")
    args = parser.parse_args()
    _ = args.sample  # intentional no-op

    t0 = time.time()
    banner("Step 8 — Calibration, DCA, Operating Characteristics, ROC overlay (NTDB 2020-2024)")

    # -------------------------------------------------------------------------
    # Load Step 6 test-scored parquet
    # -------------------------------------------------------------------------
    print(f"Loading {IN_PARQUET}", flush=True)
    if not os.path.exists(IN_PARQUET):
        raise SystemExit(f"ERROR: Step 6 output not found: {IN_PARQUET}")
    df = pd.read_parquet(IN_PARQUET)
    print(f"  rows = {len(df):,}   cols = {len(df.columns)}", flush=True)

    required = ['received_platelets_4h'] + [c for (_, c, _, _, _) in SCORE_SPECS] \
        + [m for (_, _, m, _, _) in SCORE_SPECS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"ERROR: missing required columns: {missing}")

    y_all = df['received_platelets_4h'].astype(int).values
    overall_prev = float(y_all.mean())
    print(f"  Overall received_platelets_4h prevalence = {overall_prev:.4f}  "
          f"(n_events = {int(y_all.sum()):,} / {len(y_all):,})", flush=True)

    # -------------------------------------------------------------------------
    # Part A — Load + recalibrate each score
    # -------------------------------------------------------------------------
    banner("Part A — Loading + recalibrating each score")
    score_data = {}
    for spec in SCORE_SPECS:
        d = load_score_data(df, spec)
        d = recalibrate_for_dca_and_cal(d)
        score_data[d['label']] = d
        print(f"  {d['label']:<12}  N={d['n']:>9,}  prev={d['prev']:.4f}  "
              f"raw range=[{d['score'].min():.4f}, {d['score'].max():.4f}]  "
              f"recal intercept={d['intercept']:+.4f}  slope={d['slope']:+.4f}",
              flush=True)

    # -------------------------------------------------------------------------
    # Part B — Decision Curve Analysis (table + figure)
    # -------------------------------------------------------------------------
    banner("Part B — Decision Curve Analysis")
    print(f"Computing NB across {len(THRESHOLDS)} thresholds "
          f"({THRESHOLDS[0]:.3f} -> {THRESHOLDS[-1]:.3f})", flush=True)

    dca_rows = []
    for p_t in THRESHOLDS:
        ta_nb = nb_treat_all(overall_prev, p_t)
        for label in [s[0] for s in SCORE_SPECS]:
            d = score_data[label]
            nb = net_benefit(d['y'], d['prob'], p_t) if d['n'] > 0 else np.nan
            dca_rows.append({
                'threshold': round(float(p_t), 4),
                'score': label,
                'net_benefit': nb,
                'n_used': d['n'],
                'treat_all_nb': ta_nb,
                'treat_none_nb': 0.0,
            })

    dca_df = pd.DataFrame(dca_rows)[
        ['threshold', 'score', 'net_benefit', 'n_used', 'treat_all_nb', 'treat_none_nb']
    ]
    dca_df.to_csv(OUT_TABLE_DCA, index=False)
    print(f"  Wrote {OUT_TABLE_DCA}  ({len(dca_df)} rows)", flush=True)

    # --- DCA figure ---
    print("  Rendering DCA figure", flush=True)
    fig, ax = plt.subplots(figsize=(9.0, 6.2), dpi=150)

    # Pivot for plotting
    ref_rows = dca_df.drop_duplicates('threshold').sort_values('threshold')
    x_ref = ref_rows['threshold'].values
    ax.plot(x_ref, ref_rows['treat_all_nb'].values,
            color='#888888', linewidth=1.1, linestyle='--', label='Treat all')
    ax.plot(x_ref, ref_rows['treat_none_nb'].values,
            color='#888888', linewidth=1.0, linestyle='-', label='Treat none')

    for label in [s[0] for s in SCORE_SPECS]:
        sub = dca_df[dca_df['score'] == label].sort_values('threshold')
        ax.plot(sub['threshold'].values, sub['net_benefit'].values,
                color=COLORS[label], linewidth=LINEWIDTH[label], label=label)

    ax.set_xlabel('Threshold probability', fontsize=12)
    ax.set_ylabel('Net benefit', fontsize=12)
    ax.set_title('Decision Curve Analysis — PAT-TBI (NTDB 2020-2024 test split)', fontsize=13)
    ax.set_xlim(float(THRESHOLDS[0]), float(THRESHOLDS[-1]))

    model_nb = dca_df[dca_df['score'].isin([s[0] for s in SCORE_SPECS])]['net_benefit']
    y_model_max = float(np.nanmax(model_nb.values)) if len(model_nb) else 0.12
    y_upper = max(0.12, y_model_max + 0.01)
    ax.set_ylim(-0.01, y_upper)

    ax.axhline(0, color='#000000', linewidth=0.5, alpha=0.4)
    ax.grid(True, which='major', linestyle=':', alpha=0.45)
    ax.minorticks_off()
    ax.legend(loc='upper right', frameon=True, fontsize=10)

    fig.tight_layout()
    fig.savefig(OUT_FIG_DCA_PNG, dpi=300, bbox_inches='tight')
    fig.savefig(OUT_FIG_DCA_PDF, bbox_inches='tight')
    plt.close(fig)
    print(f"    -> {OUT_FIG_DCA_PNG}", flush=True)
    print(f"    -> {OUT_FIG_DCA_PDF}", flush=True)

    # Max-NB summary print
    print("\n  Max net benefit per score over 0.01-0.20 grid:")
    for label in [s[0] for s in SCORE_SPECS]:
        sub = dca_df[dca_df['score'] == label]
        if sub['net_benefit'].notna().any():
            idx = sub['net_benefit'].idxmax()
            print(f"    {label:<12}  max NB = {sub.loc[idx, 'net_benefit']:+.5f}  "
                  f"at p_t = {sub.loc[idx, 'threshold']*100:.1f}%")

    # -------------------------------------------------------------------------
    # Part C — Calibration plots (2x3 grid)
    # -------------------------------------------------------------------------
    banner("Part C — Calibration plots")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes_flat = axes.flatten()

    rng = np.random.default_rng(SEED)
    cal_summary_rows = []

    for idx, spec in enumerate(SCORE_SPECS):
        label = spec[0]
        d = score_data[label]
        ax = axes_flat[idx]

        y = d['y']
        prob = d['prob']
        if len(y) == 0:
            ax.text(0.5, 0.5, f"{label}\n(no data)",
                    ha='center', va='center', transform=ax.transAxes)
            continue

        # Decile reliability
        try:
            frac_pos, mean_pred = calibration_curve(
                y, prob, n_bins=10, strategy='quantile')
        except Exception as exc:
            print(f"  [calibration_curve warn for {label}] {exc}")
            frac_pos, mean_pred = np.array([]), np.array([])

        # Lowess smoother — subsample to 200k at large N
        if len(y) > 200_000:
            sel = rng.choice(len(y), size=200_000, replace=False)
            y_lo, p_lo = y[sel], prob[sel]
        else:
            y_lo, p_lo = y, prob
        try:
            lo = lowess(y_lo, p_lo, frac=0.3, it=0, return_sorted=True)
            ax.plot(lo[:, 0], lo[:, 1], color=COLORS[label], lw=1.5, alpha=0.85,
                    label='LOWESS')
        except Exception as exc:
            print(f"  [lowess warn for {label}] {exc}")

        # Diagonal
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)

        # Decile points
        if len(mean_pred) > 0:
            ax.plot(mean_pred, frac_pos, 'o', color=COLORS[label], ms=6,
                    markeredgecolor='black', mew=0.5, label='Deciles')

        # Axis bounds clipped to observed range
        if len(mean_pred) > 0:
            upper = max(mean_pred.max(), frac_pos.max()) * 1.15
            upper = min(max(upper, 0.05), 1.0)
        else:
            upper = 1.0
        ax.set_xlim(0, upper)
        ax.set_ylim(0, upper)

        citl, slope = citl_and_slope(y, prob)
        brier = brier_score_loss(y, prob)
        cal_summary_rows.append({
            'score': label, 'n': len(y), 'prev': float(y.mean()),
            'citl': citl, 'slope': slope, 'brier': brier,
        })

        ax.set_title(f"{label}\nCITL={citl:+.3f}  slope={slope:.2f}  "
                     f"N={len(y):,}", fontsize=10)
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Observed fraction')
        ax.grid(alpha=0.3)
        ax.legend(loc='upper left', fontsize=8)

    # Summary panel — all 5 overlaid deciles
    ax = axes_flat[5]
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Perfect')
    max_up = 0.05
    for spec in SCORE_SPECS:
        label = spec[0]
        d = score_data[label]
        if len(d['y']) == 0:
            continue
        try:
            frac_pos, mean_pred = calibration_curve(
                d['y'], d['prob'], n_bins=10, strategy='quantile')
            ax.plot(mean_pred, frac_pos, '-o', color=COLORS[label], ms=4, lw=1.4,
                    label=label, alpha=0.85)
            if len(mean_pred):
                max_up = max(max_up, float(max(mean_pred.max(), frac_pos.max())))
        except Exception as exc:
            print(f"  [summary calib warn for {label}] {exc}")
    up = min(1.0, max(0.05, max_up * 1.15))
    ax.set_xlim(0, up)
    ax.set_ylim(0, up)
    ax.set_title('Reliability summary — all 5 scores', fontsize=10)
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Observed fraction')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=8)

    fig.suptitle('PAT-TBI — Calibration (NTDB 2020-2024 test split)', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_FIG_CAL_PNG, dpi=300, bbox_inches='tight')
    fig.savefig(OUT_FIG_CAL_PDF, bbox_inches='tight')
    plt.close(fig)
    print(f"    -> {OUT_FIG_CAL_PNG}", flush=True)
    print(f"    -> {OUT_FIG_CAL_PDF}", flush=True)

    if cal_summary_rows:
        print("\n  Calibration summary:")
        for r in cal_summary_rows:
            print(f"    {r['score']:<12}  N={r['n']:>9,}  "
                  f"CITL={r['citl']:+.4f}  slope={r['slope']:+.3f}  "
                  f"Brier={r['brier']:.4f}")

    # -------------------------------------------------------------------------
    # Part D — Operating Characteristics (3 cutoff rules per score)
    # -------------------------------------------------------------------------
    banner("Part D — Operating Characteristics")
    op_rows = []

    for spec in SCORE_SPECS:
        label = spec[0]
        d = score_data[label]
        y = d['y']
        s_raw = d['score']  # cutoffs reported on native scale (integer for pat_tbi_score,
                             # continuous for others)
        if len(y) == 0:
            print(f"  {label}: no data — skipping", flush=True)
            continue
        print(f"\n  {label}  N={len(y):,}  prev={y.mean():.4f}")

        # --- Youden ---
        thr_y, J_y = youden_threshold(y, s_raw)
        row_y = compute_metrics(y, s_raw, thr_y, label, 'youden')
        row_y['Cutoff_Rule'] = 'youden'
        row_y['Youden_J'] = float(row_y['Sensitivity'] + row_y['Specificity'] - 1.0) \
            if (not np.isnan(row_y['Sensitivity']) and not np.isnan(row_y['Specificity'])) \
            else np.nan
        print(f"    youden   cutoff={thr_y:.4f}  J={J_y:.4f}  "
              f"sens={row_y['Sensitivity']:.3f}  spec={row_y['Specificity']:.3f}")
        op_rows.append(row_y)

        # --- Sens >= 0.90 ---
        thr_s, ach_sens, ach_spec = sens_90_threshold(y, s_raw)
        if not np.isnan(thr_s):
            row_s = compute_metrics(y, s_raw, thr_s, label, 'sens_90')
            row_s['Cutoff_Rule'] = 'sens_90'
            row_s['Youden_J'] = float(row_s['Sensitivity'] + row_s['Specificity'] - 1.0) \
                if (not np.isnan(row_s['Sensitivity']) and not np.isnan(row_s['Specificity'])) \
                else np.nan
            print(f"    sens_90  cutoff={thr_s:.4f}  "
                  f"sens={row_s['Sensitivity']:.3f}  spec={row_s['Specificity']:.3f}")
            op_rows.append(row_s)
        else:
            print(f"    sens_90  unreachable (max sens < 0.90)")

        # --- Spec >= 0.90 ---
        thr_sp, ach_sens_sp, ach_spec_sp = spec_90_threshold(y, s_raw)
        if not np.isnan(thr_sp):
            row_sp = compute_metrics(y, s_raw, thr_sp, label, 'spec_90')
            row_sp['Cutoff_Rule'] = 'spec_90'
            row_sp['Youden_J'] = float(row_sp['Sensitivity'] + row_sp['Specificity'] - 1.0) \
                if (not np.isnan(row_sp['Sensitivity']) and not np.isnan(row_sp['Specificity'])) \
                else np.nan
            print(f"    spec_90  cutoff={thr_sp:.4f}  "
                  f"sens={row_sp['Sensitivity']:.3f}  spec={row_sp['Specificity']:.3f}")
            op_rows.append(row_sp)
        else:
            print(f"    spec_90  unreachable (max spec < 0.90)")

    # Build final opchar table with requested column order
    op_df = pd.DataFrame(op_rows)
    op_df = op_df.rename(columns={
        'Cutoff_Value': 'Cutoff',
        'LR_Positive': 'LR_pos',
        'LR_Negative': 'LR_neg',
    })
    out_cols = [
        'Score', 'Cutoff', 'Cutoff_Rule', 'N',
        'Sensitivity', 'Sens_CI_Low', 'Sens_CI_High',
        'Specificity', 'Spec_CI_Low', 'Spec_CI_High',
        'PPV', 'NPV', 'LR_pos', 'LR_neg', 'F1', 'Youden_J',
    ]
    for c in out_cols:
        if c not in op_df.columns:
            op_df[c] = np.nan
    op_df = op_df[out_cols]
    op_df.to_csv(OUT_TABLE_OPCHAR, index=False)
    print(f"\n  Wrote {OUT_TABLE_OPCHAR}  ({len(op_df)} rows)", flush=True)

    # -------------------------------------------------------------------------
    # Part E — ROC overlay figure
    # -------------------------------------------------------------------------
    banner("Part E — ROC overlay figure")

    roc_curves = {}  # label -> (fpr, tpr, auc, n)

    # Try to consume Step 7 pickle first
    if os.path.exists(ROC_PKL):
        try:
            with open(ROC_PKL, 'rb') as f:
                pkl = pickle.load(f)
            print(f"  Loaded ROC data from {ROC_PKL}", flush=True)

            # Expected pickle shape: dict keyed by either display label or col name,
            # each entry containing fpr/tpr/auc keys (liberal about schema).
            def _extract(entry):
                fpr = entry.get('fpr') if isinstance(entry, dict) else None
                tpr = entry.get('tpr') if isinstance(entry, dict) else None
                auc = entry.get('auc') if isinstance(entry, dict) else None
                n = entry.get('n') if isinstance(entry, dict) else None
                if fpr is None or tpr is None:
                    return None
                return (np.asarray(fpr), np.asarray(tpr), auc, n)

            # Accept either display names or Step-6 column names as keys
            colname_to_label = {c: lab for (lab, c, _, _, _) in SCORE_SPECS}
            if isinstance(pkl, dict):
                for key, val in pkl.items():
                    label = None
                    if key in COLORS:
                        label = key
                    elif key in colname_to_label:
                        label = colname_to_label[key]
                    if label is None:
                        continue
                    ext = _extract(val)
                    if ext is None:
                        continue
                    roc_curves[label] = ext
        except Exception as exc:
            print(f"  WARN loading pickle ({exc}); will recompute from parquet", flush=True)

    # For any score missing from the pickle, recompute from the parquet on the raw score
    for spec in SCORE_SPECS:
        label = spec[0]
        if label in roc_curves:
            continue
        d = score_data[label]
        if d['n'] == 0:
            continue
        fpr, tpr, _ = roc_curve(d['y'], d['score'])
        try:
            auc = float(roc_auc_score(d['y'], d['score']))
        except Exception:
            auc = np.nan
        roc_curves[label] = (fpr, tpr, auc, d['n'])
        print(f"  Recomputed ROC for {label}  AUC={auc:.4f}  N={d['n']:,}", flush=True)

    fig, ax = plt.subplots(figsize=(8.0, 7.2), dpi=150)
    ax.plot([0, 1], [0, 1], color='#888888', linewidth=1.0,
            linestyle='--', label='Chance')

    plot_order = [s[0] for s in SCORE_SPECS]
    for label in plot_order:
        if label not in roc_curves:
            continue
        fpr, tpr, auc, n = roc_curves[label]
        lw = LINEWIDTH.get(label, 1.6)
        auc_str = f"{auc:.3f}" if (auc is not None and not np.isnan(auc)) else "NA"
        n_str = f"{n:,}" if n is not None else ""
        ax.plot(fpr, tpr, color=COLORS.get(label, '#444444'),
                linewidth=lw, linestyle='-',
                label=f"{label}  AUROC={auc_str}" + (f"  (N={n_str})" if n_str else ""))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('1 - Specificity (False Positive Rate)', fontsize=12)
    ax.set_ylabel('Sensitivity (True Positive Rate)', fontsize=12)
    ax.set_title('ROC — PAT-TBI vs comparators (NTDB 2020-2024 test split)', fontsize=13)
    ax.grid(True, linestyle=':', alpha=0.45)
    ax.legend(loc='lower right', frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_FIG_ROC_PNG, dpi=300, bbox_inches='tight')
    fig.savefig(OUT_FIG_ROC_PDF, bbox_inches='tight')
    plt.close(fig)
    print(f"    -> {OUT_FIG_ROC_PNG}", flush=True)
    print(f"    -> {OUT_FIG_ROC_PDF}", flush=True)

    # -------------------------------------------------------------------------
    # Part F — Score distribution histogram (PAT-TBI integer score by event)
    # -------------------------------------------------------------------------
    banner("Part F — PAT-TBI score distribution by event status")

    d_pat = score_data['PAT-TBI']
    if d_pat['n'] > 0:
        y_pat = d_pat['y']
        s_pat = d_pat['score']
        s_min = int(np.floor(np.nanmin(s_pat)))
        s_max = int(np.ceil(np.nanmax(s_pat)))
        bins = np.arange(s_min, s_max + 2) - 0.5  # integer-centered bins

        fig, ax = plt.subplots(figsize=(9.0, 5.6), dpi=150)
        ax.hist(s_pat[y_pat == 0], bins=bins,
                color='#7FB3D5', edgecolor='#1F4E79', linewidth=0.4,
                alpha=0.75, label=f'No platelets in 4h (N={int((y_pat == 0).sum()):,})')
        ax.hist(s_pat[y_pat == 1], bins=bins,
                color='#D62728', edgecolor='#7B1A1B', linewidth=0.4,
                alpha=0.75, label=f'Received platelets in 4h (N={int((y_pat == 1).sum()):,})')
        ax.set_xlabel('PAT-TBI integer score', fontsize=12)
        ax.set_ylabel('Number of patients', fontsize=12)
        ax.set_title('PAT-TBI score distribution by event status (NTDB 2020-2024 test split)',
                     fontsize=13)
        ax.grid(True, axis='y', linestyle=':', alpha=0.45)
        ax.legend(loc='upper right', frameon=True, fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT_FIG_DIST_PNG, dpi=300, bbox_inches='tight')
        fig.savefig(OUT_FIG_DIST_PDF, bbox_inches='tight')
        plt.close(fig)
        print(f"    -> {OUT_FIG_DIST_PNG}", flush=True)
        print(f"    -> {OUT_FIG_DIST_PDF}", flush=True)
    else:
        print("  PAT-TBI: no data — skipping score-distribution figure", flush=True)

    banner(f"Step 8 complete — runtime = {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
