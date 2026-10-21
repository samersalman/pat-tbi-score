"""
PAT-TBI Step 10 - Severity-score benchmarks (NOT endpoint-matched)
================================================================================

Methodologically honest severity benchmark for PAT-TBI on the 2020-2024 NTDB
TBI cohort:

    * AUROC for PAT-TBI and for GCS, GCS-P, IMPACT-Core and CRASH-Basic,
      each on the rows where the comparator's own inputs were available
      (its `_mask` column from Step 6).
    * Percentile bootstrap 95% CI (2000 iterations, 100k subsample).
    * Pairwise DeLong test vs PAT-TBI, using the intersection of masks.
    * Decision-curve net benefit across 1%-20% thresholds in 1% steps.

IMPORTANT: GCS, GCS-P, IMPACT-Core and CRASH-Basic were all derived to predict
TBI MORTALITY, not platelet-transfusion receipt. They appear here as severity
proxies to answer the question "does existing severity triage already identify
likely platelet recipients well enough?" The answer from AUROC will typically
be "no, PAT-TBI adds meaningful discrimination"; the comparison is not
methodologically equivalent to a head-to-head between two scores predicting
the same endpoint.

Inputs
------
  PAT_TBI/v4/derived/cohort_test_scored.parquet    (Step 6)

Outputs
-------
  PAT_TBI/v4/tables/table_pat_severity_benchmarks.csv
  PAT_TBI/v4/tables/table_pat_severity_dca.csv
  PAT_TBI/v4/figures/figure_pat_severity_benchmark.{pdf,png}

Conventions
-----------
  SEED = 42
  Bootstrap: 2000 iters (100 under --smoke-test), subsample 100,000.
  DCA: integer/raw-score comparators are mapped to a probability via isotonic
       regression (monotonic, not a fitted model). Already-probability
       comparators (IMPACT-Core, CRASH-Basic) pass through as-is.
  No recalibration of any comparator's coefficients is performed.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from typing import Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats as sstats
from scipy.stats import rankdata
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Config / paths
# ---------------------------------------------------------------------------
SEED = 42
N_BOOT_FULL = 2000
N_BOOT_SMOKE = 100
BOOT_SUBSAMPLE = 100_000
EPS = 1e-6

import os as _os, sys as _sys  # path handling: make analysis/config.py importable
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import config  # noqa: E402  (analysis/config.py: PAT_TBI_WORKDIR, PAT_TBI_NTDB_DUCKDB)
PAT = str(config.V4_DIR)
DERIVED = os.path.join(PAT, "derived")
TABLES = os.path.join(PAT, "tables")
FIGURES = os.path.join(PAT, "figures")

IN_PARQUET = os.path.join(DERIVED, "cohort_test_scored.parquet")
OUT_TABLE = os.path.join(TABLES, "table_pat_severity_benchmarks.csv")
OUT_DCA = os.path.join(TABLES, "table_pat_severity_dca.csv")
OUT_PDF = os.path.join(FIGURES, "figure_pat_severity_benchmark.pdf")
OUT_PNG = os.path.join(FIGURES, "figure_pat_severity_benchmark.png")

OUTCOME_COL = "received_platelets_4h"

# Score registry: (score column in parquet, display name, mask column, note)
# Uses Step 6's actual column names (gcs_risk, gcs_p_risk, impact_core_prob,
# crash_basic_prob; masks = pat_tbi_score_mask, gcs_mask, gcs_p_mask,
# impact_mask, crash_mask).
SCORE_SPECS = [
    ("pat_tbi_score",    "PAT-TBI",      "pat_tbi_score_mask",
     "derived endpoint"),
    ("gcs_risk",         "GCS",          "gcs_mask",
     "severity proxy; endpoint = mortality"),
    ("gcs_p_risk",       "GCS-P",        "gcs_p_mask",
     "severity proxy; endpoint = mortality"),
    ("impact_core_prob", "IMPACT-Core",  "impact_mask",
     "severity proxy; endpoint = mortality"),
    ("crash_basic_prob", "CRASH-Basic",  "crash_mask",
     "severity proxy; endpoint = mortality"),
]

DCA_THRESHOLDS = np.arange(0.01, 0.205, 0.01)  # 1% .. 20% in 1% steps


def banner(msg: str) -> None:
    line = "=" * 78
    print(f"\n{line}\n{msg}\n{line}", flush=True)


# ---------------------------------------------------------------------------
# DeLong (Sun-Xu fast form) - identical formulation to Step 7
# ---------------------------------------------------------------------------
def _midrank(x: np.ndarray) -> np.ndarray:
    return rankdata(x, method="average")


def delong_v10_v01(y: np.ndarray, pred: np.ndarray
                   ) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.int8)
    pred = np.asarray(pred, dtype=np.float64)
    pos = pred[y == 1]
    neg = pred[y == 0]
    n1, n0 = len(pos), len(neg)

    tx1 = _midrank(pos)
    tx0 = _midrank(neg)
    tz = _midrank(pred)
    tz1 = tz[y == 1]
    tz0 = tz[y == 0]

    V10 = (tz1 - tx1) / n0
    V01 = 1.0 - (tz0 - tx0) / n1
    return V10, V01


def delong_test(y: np.ndarray, p1: np.ndarray, p2: np.ndarray):
    """Vectorized DeLong. Returns (z, p, delta, se_delta, auc1, auc2).

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
# Bootstrap helpers
# ---------------------------------------------------------------------------
def bootstrap_auroc_ci(y: np.ndarray,
                       scores: np.ndarray,
                       B: int,
                       sub_size: int,
                       rng: np.random.Generator) -> Tuple[float, float]:
    """Percentile bootstrap 95% CI for AUROC, optionally on a subsample of
    size sub_size each iter (ties CI accuracy to sub_size, not cohort size)."""
    n = len(y)
    if n == 0:
        return (np.nan, np.nan)
    eff = min(sub_size, n)
    aucs = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, n, size=eff)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == eff:
            aucs[b] = np.nan
            continue
        aucs[b] = roc_auc_score(yb, scores[idx])
    aucs = aucs[~np.isnan(aucs)]
    if aucs.size == 0:
        return (np.nan, np.nan)
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_auroc_delta_ci(y: np.ndarray,
                             s_ref: np.ndarray,
                             s_comp: np.ndarray,
                             B: int,
                             sub_size: int,
                             rng: np.random.Generator
                             ) -> Tuple[float, float]:
    """Percentile bootstrap CI for delta = AUROC(ref) - AUROC(comp), using
    paired resampling (same row indices for both scores)."""
    n = len(y)
    if n == 0:
        return (np.nan, np.nan)
    eff = min(sub_size, n)
    deltas = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, n, size=eff)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == eff:
            deltas[b] = np.nan
            continue
        a_ref = roc_auc_score(yb, s_ref[idx])
        a_cmp = roc_auc_score(yb, s_comp[idx])
        deltas[b] = a_ref - a_cmp
    deltas = deltas[~np.isnan(deltas)]
    if deltas.size == 0:
        return (np.nan, np.nan)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# DCA helpers
# ---------------------------------------------------------------------------
def net_benefit(y: np.ndarray, prob: np.ndarray, p_t: float) -> float:
    """Vickers-Elkin net benefit at threshold p_t, with n = len(y).

    Matches Step 8's implementation exactly.
    """
    n = len(y)
    if n == 0:
        return np.nan
    if p_t >= 1.0:
        return np.nan
    pred_pos = prob >= p_t
    tp = float(((pred_pos) & (y == 1)).sum()) / n
    fp = float(((pred_pos) & (y == 0)).sum()) / n
    return tp - fp * (p_t / (1.0 - p_t))


def nb_treat_all(prevalence: float, p_t: float) -> float:
    if p_t >= 1.0:
        return np.nan
    return prevalence - (1.0 - prevalence) * (p_t / (1.0 - p_t))


def score_to_prob_isotonic(y: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Monotonic mapping from raw score to probability via isotonic regression.

    This is NOT a fitted model recalibration: IsotonicRegression with
    increasing=True just gives the isotonic least-squares fit of prob ~ score.
    We need it only so that integer-valued comparators (GCS, GCS-P, PAT-TBI
    integer) can be passed into a threshold-based net-benefit calculation.
    """
    iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
    iso.fit(score, y)
    prob = iso.predict(score)
    return np.clip(prob, EPS, 1.0 - EPS)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description=("PAT-TBI Step 10 - severity-proxy benchmarks "
                     "(NOT endpoint-matched)"))
    parser.add_argument("--smoke-test", action="store_true",
                        help="Reduce bootstrap iterations to 100.")
    args = parser.parse_args()

    n_boot = N_BOOT_SMOKE if args.smoke_test else N_BOOT_FULL

    t0 = time.time()
    os.makedirs(TABLES, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)

    banner("Step 10 - Severity-score benchmarks (NOT endpoint-matched)")
    print(f"Cohort years: 2020-2024")
    print(f"Input : {IN_PARQUET}")
    print(f"Output: {OUT_TABLE}")
    print(f"        {OUT_DCA}")
    print(f"        {OUT_PDF}")
    print(f"        {OUT_PNG}")
    print(f"Bootstrap iters: {n_boot}  (subsample {BOOT_SUBSAMPLE:,})")
    if args.smoke_test:
        print("  [SMOKE TEST] reduced bootstrap to 100 iters")

    rng = np.random.default_rng(SEED)

    # -----------------------------------------------------------------------
    # Load
    # -----------------------------------------------------------------------
    banner("Loading Step 6 test-set scored parquet")
    usecols = [OUTCOME_COL]
    for sc, _, m, _ in SCORE_SPECS:
        usecols += [sc, m]
    df = pd.read_parquet(IN_PARQUET, columns=usecols)
    print(f"  loaded {df.shape[0]:,} rows x {df.shape[1]} cols")

    y_all = df[OUTCOME_COL].astype(int).values
    n_events_all = int(y_all.sum())
    prevalence_all = float(y_all.mean())
    print(f"  TEST: N={len(y_all):,}  events={n_events_all:,}  "
          f"prevalence={prevalence_all:.4f}")

    # Reference (PAT-TBI): must be defined for every row per Step 6 contract
    pat_sc_col, _, pat_mask_col, _ = SCORE_SPECS[0]
    pat_mask_all = df[pat_mask_col].astype(bool).values
    if not pat_mask_all.all():
        raise RuntimeError(
            "PAT-TBI mask has False rows - Step 6 contract says it should be "
            "all-True. Aborting.")
    pat_scores_all = df[pat_sc_col].astype(float).values

    # -----------------------------------------------------------------------
    # Part A - per-score AUROC + bootstrap CI + DeLong vs PAT-TBI
    # -----------------------------------------------------------------------
    banner("Part A - AUROC with bootstrap 95% CI and DeLong vs PAT-TBI")
    rows = []
    for score_col, disp, mask_col, note in SCORE_SPECS:
        is_ref = (score_col == pat_sc_col)

        m = df[mask_col].astype(bool).values
        s = df[score_col].astype(float).values

        # Restrict to rows where inputs for THIS score were available AND
        # the raw score is non-NaN (already-prob comparators may have NaN
        # where mask=False; guard defensively).
        keep = m & ~np.isnan(s)
        y_k = y_all[keep]
        s_k = s[keep]
        n_k = int(keep.sum())
        n_ev = int(y_k.sum())
        print(f"  {disp:<12s}  n={n_k:,}  events={n_ev:,}", flush=True)

        if n_k < 10 or n_ev == 0 or n_ev == n_k:
            auc = np.nan
            lo, hi = np.nan, np.nan
            delta = np.nan
            dlo, dhi = np.nan, np.nan
            pval = np.nan
        else:
            auc = float(roc_auc_score(y_k, s_k))
            lo, hi = bootstrap_auroc_ci(
                y_k, s_k, B=n_boot, sub_size=BOOT_SUBSAMPLE, rng=rng)

            if is_ref:
                delta = 0.0
                dlo, dhi = 0.0, 0.0
                pval = np.nan
            else:
                # DeLong on the paired INTERSECTION of PAT-TBI and comparator.
                pair_keep = pat_mask_all & keep
                y_pair = y_all[pair_keep]
                p_ref = pat_scores_all[pair_keep]
                p_cmp = s[pair_keep]
                if (y_pair.sum() == 0 or y_pair.sum() == len(y_pair)
                        or len(y_pair) < 10):
                    delta = np.nan
                    dlo, dhi = np.nan, np.nan
                    pval = np.nan
                else:
                    _z, pval, delta, _se, _a1, _a2 = delong_test(
                        y_pair, p_ref, p_cmp)
                    dlo, dhi = bootstrap_auroc_delta_ci(
                        y_pair, p_ref, p_cmp,
                        B=n_boot, sub_size=BOOT_SUBSAMPLE, rng=rng)

        rows.append({
            "Score": disp,
            "N": n_k,
            "N_events": n_ev,
            "AUROC": auc,
            "CI_Low": lo,
            "CI_High": hi,
            "Delta_vs_PAT_TBI": delta,
            "Delta_CI_Low": dlo,
            "Delta_CI_High": dhi,
            "DeLong_p_vs_PAT_TBI": pval,
            "Note": note,
        })

        if not np.isnan(auc):
            ci_str = f"[{lo:.3f}, {hi:.3f}]"
            if is_ref:
                print(f"    AUROC={auc:.4f} {ci_str}  (reference)")
            else:
                print(f"    AUROC={auc:.4f} {ci_str}   "
                      f"delta={delta:+.4f} [{dlo:+.4f}, {dhi:+.4f}]  "
                      f"DeLong p={pval:.3g}")

    bench_df = pd.DataFrame(rows, columns=[
        "Score", "N", "N_events",
        "AUROC", "CI_Low", "CI_High",
        "Delta_vs_PAT_TBI", "Delta_CI_Low", "Delta_CI_High",
        "DeLong_p_vs_PAT_TBI", "Note",
    ])
    bench_df.to_csv(OUT_TABLE, index=False)
    print(f"\n  wrote {OUT_TABLE}  ({bench_df.shape[0]} rows)")

    # -----------------------------------------------------------------------
    # Part B - Decision-curve analysis across 1%-20% thresholds
    # -----------------------------------------------------------------------
    banner("Part B - DCA net benefit across 1%-20% thresholds")
    dca_rows = []

    # Treat-none is net benefit 0 at every threshold by definition.
    for p_t in DCA_THRESHOLDS:
        dca_rows.append({
            "Threshold": float(round(p_t, 4)),
            "Score": "Treat None",
            "NetBenefit": 0.0,
        })
        dca_rows.append({
            "Threshold": float(round(p_t, 4)),
            "Score": "Treat All",
            "NetBenefit": nb_treat_all(prevalence_all, float(p_t)),
        })

    for score_col, disp, mask_col, _ in SCORE_SPECS:
        m = df[mask_col].astype(bool).values
        s = df[score_col].astype(float).values
        keep = m & ~np.isnan(s)
        y_k = y_all[keep]
        s_k = s[keep]
        if len(y_k) < 10 or y_k.sum() == 0:
            print(f"  {disp:<12s}  insufficient data for DCA (n={len(y_k)})")
            for p_t in DCA_THRESHOLDS:
                dca_rows.append({
                    "Threshold": float(round(p_t, 4)),
                    "Score": disp,
                    "NetBenefit": np.nan,
                })
            continue

        # Map raw score to probability via isotonic regression (monotonic
        # mapping only; no parametric recalibration fit).
        prob_k = score_to_prob_isotonic(y_k, s_k)

        print(f"  {disp:<12s}  n={len(y_k):,}  prob range "
              f"[{prob_k.min():.3f}, {prob_k.max():.3f}]")

        for p_t in DCA_THRESHOLDS:
            nb = net_benefit(y_k, prob_k, float(p_t))
            dca_rows.append({
                "Threshold": float(round(p_t, 4)),
                "Score": disp,
                "NetBenefit": float(nb) if not np.isnan(nb) else np.nan,
            })

    dca_df = pd.DataFrame(dca_rows, columns=["Threshold", "Score", "NetBenefit"])
    dca_df = dca_df.sort_values(["Score", "Threshold"]).reset_index(drop=True)
    dca_df.to_csv(OUT_DCA, index=False)
    print(f"  wrote {OUT_DCA}  ({dca_df.shape[0]} rows)")

    # -----------------------------------------------------------------------
    # Part C - Forest-style AUROC bar chart with 95% CI error bars
    # -----------------------------------------------------------------------
    banner("Part C - Rendering severity-benchmark AUROC figure")

    plot_df = bench_df.dropna(subset=["AUROC"]).copy()
    # Keep registry order (PAT-TBI first, then comparators)
    order_map = {disp: i for i, (_, disp, _, _) in enumerate(SCORE_SPECS)}
    plot_df["_order"] = plot_df["Score"].map(order_map)
    plot_df = plot_df.sort_values("_order", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    y_pos = np.arange(len(plot_df))
    aucs = plot_df["AUROC"].values
    lo = plot_df["CI_Low"].values
    hi = plot_df["CI_High"].values
    xerr = np.vstack([aucs - lo, hi - aucs])
    colors = ["#1f77b4" if s == "PAT-TBI" else "#7f7f7f"
              for s in plot_df["Score"].values]

    ax.barh(y_pos, aucs, color=colors, edgecolor="black",
            alpha=0.85, height=0.55)
    ax.errorbar(aucs, y_pos, xerr=xerr, fmt="none",
                ecolor="black", elinewidth=1.0, capsize=3)
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1.0,
               label="No discrimination (AUROC=0.5)")

    for i, (a, l, h) in enumerate(zip(aucs, lo, hi)):
        ax.text(min(a + 0.02, 0.98), i,
                f"{a:.3f} [{l:.3f}, {h:.3f}]",
                va="center", ha="left", fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["Score"].values)
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("AUROC (95% CI)")
    ax.set_title("PAT-TBI vs severity proxies for platelet-receipt prediction")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()

    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT_PDF}")
    print(f"  wrote {OUT_PNG}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    banner("Summary")
    print("  Score         AUROC  [95% CI]         delta vs PAT-TBI  DeLong p")
    for _, r in bench_df.iterrows():
        if np.isnan(r["AUROC"]):
            print(f"  {r['Score']:<12s}  (no data)")
            continue
        if r["Score"] == "PAT-TBI":
            print(f"  {r['Score']:<12s}  {r['AUROC']:.4f} "
                  f"[{r['CI_Low']:.3f}, {r['CI_High']:.3f}]  "
                  f"(reference)")
        else:
            print(f"  {r['Score']:<12s}  {r['AUROC']:.4f} "
                  f"[{r['CI_Low']:.3f}, {r['CI_High']:.3f}]   "
                  f"delta={r['Delta_vs_PAT_TBI']:+.4f}   "
                  f"p={r['DeLong_p_vs_PAT_TBI']:.3g}")

    # Max net benefit per score on the 1%-20% grid
    print("\n  Max net benefit on 1%-20% threshold grid:")
    for sc in bench_df["Score"].tolist() + ["Treat All", "Treat None"]:
        sub = dca_df[dca_df["Score"] == sc]
        if sub["NetBenefit"].notna().any():
            j = sub["NetBenefit"].idxmax()
            row = sub.loc[j]
            print(f"    {sc:<12s}  max NB = {row['NetBenefit']:+.5f}  "
                  f"at threshold {row['Threshold']:.2f}")

    banner(f"Step 10 complete in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
