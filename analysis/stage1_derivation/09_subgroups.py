"""
PAT-TBI NTDB Refit — Step 9: Pre-specified subgroup AUROCs + forest plot
================================================================================

Computes subgroup-stratified AUROCs for 5 scores (pat_tbi_score, GCS, GCS-P,
IMPACT-Core, CRASH-Basic) on the NTDB held-out test set, with vectorized DeLong
deltas of pat_tbi_score vs each comparator. Renders one forest-plot panel per
comparator.

Cohort years: 2020-2024.

Inputs
------
  PAT_TBI/v4/derived/cohort_test_scored.parquet   (Step 6 output; has scores + masks)
  PAT_TBI/v4/derived/candidates.parquet           (Step 2 output; loaded for
                                                   risk_penetrating, is_transfer,
                                                   teaching_status_binary only —
                                                   Step 6 parquet does not carry them)

Outputs
-------
  PAT_TBI/v4/tables/table_subgroups.csv
    columns:
      Subgroup, Level, Score, N, N_events,
      AUROC, CI_Low, CI_High,
      Delta_vs_PAT_TBI, Delta_SE, Delta_CI_Low, Delta_CI_High, Delta_P,
      N_events_too_small
  PAT_TBI/v4/figures/figure_subgroups_forest.pdf
  PAT_TBI/v4/figures/figure_subgroups_forest.png

Subgroups (pre-specified, PAT-TBI plan):
  1. Age  <65  vs  >=65                       (AGE_NUMBER)
  2. Sex  Male vs Female                      (SEX == 'M' / 'F')
  3. GCS  Severe (<=8) / Moderate (9-12) /
         Mild (13-15)                         (GCS2, drop NaN)
  4. Body injury  Isolated TBI vs Polytrauma  (max_ais_head>=3 & max_ais_other<=1
                                               vs the complement)
  5. Mechanism    Penetrating vs Blunt        (risk_penetrating from candidates parquet)
  6. Arrival      Transfer vs Primary         (is_transfer from candidates parquet)
  7. Center       Teaching vs Non-teaching    (teaching_status_binary from candidates parquet)

Subgroup cell rule: if N_events < 50 -> AUROC/CI/delta reported as NaN and
N_events_too_small flagged True.

Seed: 42
"""
import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
import os as _os, sys as _sys  # path handling: make analysis/config.py importable
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import config  # noqa: E402  (analysis/config.py: PAT_TBI_WORKDIR, PAT_TBI_NTDB_DUCKDB)
PAT = str(config.V4_DIR)
DERIVED = os.path.join(PAT, "derived")
TABLES_DIR = os.path.join(PAT, "tables")
FIGURES_DIR = os.path.join(PAT, "figures")
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

SCORED_PARQUET = os.path.join(DERIVED, "cohort_test_scored.parquet")
CANDIDATES_PARQUET = os.path.join(DERIVED, "candidates.parquet")

OUT_TABLE = os.path.join(TABLES_DIR, "table_subgroups.csv")
OUT_FIG_PDF = os.path.join(FIGURES_DIR, "figure_subgroups_forest.pdf")
OUT_FIG_PNG = os.path.join(FIGURES_DIR, "figure_subgroups_forest.png")

MIN_EVENTS = 50
OUTCOME_COL = "received_platelets_4h"

# Score spec: (short_label, pretty_label, score_col, mask_col)
SCORES = [
    ("PAT_TBI",  "PAT-TBI",      "pat_tbi_score",    "pat_tbi_score_mask"),
    ("GCS",      "GCS",          "gcs_risk",         "gcs_mask"),
    ("GCS_P",    "GCS-P",        "gcs_p_risk",       "gcs_p_mask"),
    ("IMPACT",   "IMPACT-core",  "impact_core_prob", "impact_mask"),
    ("CRASH",    "CRASH-Basic",  "crash_basic_prob", "crash_mask"),
]
COMPARATORS = [s for s in SCORES if s[0] != "PAT_TBI"]
PAT_SCORE = SCORES[0]  # ("PAT_TBI", ..., "pat_tbi_score", "pat_tbi_score_mask")


def banner(msg):
    line = "=" * 78
    print(f"\n{line}\n{msg}\n{line}", flush=True)


# ============================================================================
# Vectorized DeLong (Sun & Xu 2014 midrank form)
# ============================================================================
def _compute_midrank(x):
    """
    Fast midrank computation via argsort (O(n log n)).
    Ties receive the mean of the positions they occupy.
    Returns a float ndarray of midranks aligned to the input order.
    """
    J = np.argsort(x, kind='mergesort')
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1  # 1-indexed midrank
        i = j
    T2 = np.empty(N, dtype=np.float64)
    T2[J] = T
    return T2


def delong_roc_variance(y_true, predictions):
    """
    Vectorized DeLong via midranks.

    Parameters
    ----------
    y_true : 1-D array of 0/1 labels
    predictions : 2-D array (k, n) — each row is one score's predictions

    Returns
    -------
    aucs : shape (k,) AUCs
    delongcov : shape (k, k) covariance matrix of AUCs
    """
    y_true = np.asarray(y_true, dtype=np.int8)
    order = np.argsort(-y_true, kind='mergesort')  # positives first
    label_1_count = int(np.sum(y_true == 1))
    n = len(y_true)
    m = label_1_count
    nn = n - m

    preds_sorted = predictions[:, order]  # (k, n)
    k = preds_sorted.shape[0]

    tx = np.empty((k, m), dtype=np.float64)
    ty = np.empty((k, nn), dtype=np.float64)
    tz = np.empty((k, n), dtype=np.float64)

    for r in range(k):
        tx[r] = _compute_midrank(preds_sorted[r, :m])
        ty[r] = _compute_midrank(preds_sorted[r, m:])
        tz[r] = _compute_midrank(preds_sorted[r, :])

    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2.0) / nn
    v01 = (tz[:, :m] - tx) / nn
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    if k == 1:
        sx = np.array([[sx]])
        sy = np.array([[sy]])
    delongcov = sx / m + sy / nn
    return aucs, delongcov


def delong_test_vec(y_true, pred1, pred2):
    """
    Compare two correlated AUROCs via vectorized DeLong.
    Returns (auc1, auc2, delta, se_delta, z, p)
    """
    y_true = np.asarray(y_true)
    if np.sum(y_true == 1) == 0 or np.sum(y_true == 0) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    preds = np.vstack([pred1, pred2])
    aucs, cov = delong_roc_variance(y_true, preds)
    delta = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), float(delta), 0.0, 0.0, 1.0
    se = float(np.sqrt(var))
    z = delta / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(aucs[0]), float(aucs[1]), float(delta), se, float(z), float(p)


# ============================================================================
# AUROC + bootstrap CI
# ============================================================================
N_BOOT = 500
MAX_BOOT_N = 50_000


def bootstrap_auroc_ci(y, p, n_boot=N_BOOT, max_n=MAX_BOOT_N, seed=SEED):
    """Return point AUROC and (lo, hi) 95% CI via percentile bootstrap."""
    y = np.asarray(y)
    p = np.asarray(p)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan
    try:
        auc = roc_auc_score(y, p)
    except Exception:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    n = len(y)
    take_n = min(n, max_n)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=take_n)
        yi = y[idx]
        if len(np.unique(yi)) < 2:
            boots[i] = np.nan
            continue
        try:
            boots[i] = roc_auc_score(yi, p[idx])
        except Exception:
            boots[i] = np.nan
    boots = boots[~np.isnan(boots)]
    if len(boots) < 10:
        return float(auc), np.nan, np.nan
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(auc), float(lo), float(hi)


# ============================================================================
# Load scored test set + merge extras from candidates parquet
# ============================================================================
def load_data():
    banner("Loading scored test parquet + candidate extras")
    print(f"[load] {SCORED_PARQUET}", flush=True)
    df = pd.read_parquet(SCORED_PARQUET)
    print(f"       scored shape = {df.shape}", flush=True)

    need_cols = ["PUF_YEAR", "inc_key", "risk_penetrating",
                 "is_transfer", "teaching_status_binary"]
    print(f"[load] {CANDIDATES_PARQUET}", flush=True)
    print(f"       (only columns: {need_cols})", flush=True)
    import pyarrow.parquet as pq
    cand_schema = pq.ParquetFile(CANDIDATES_PARQUET).schema_arrow
    cand_cols = set(cand_schema.names)
    avail = [c for c in need_cols if c in cand_cols]
    missing = [c for c in need_cols if c not in cand_cols]
    if missing:
        print(f"       [warn] missing from candidates parquet: {missing}", flush=True)
    cand = pd.read_parquet(CANDIDATES_PARQUET, columns=avail)
    print(f"       candidates subset shape = {cand.shape}", flush=True)

    for key in ("PUF_YEAR", "inc_key"):
        if key in df.columns:
            df[key] = pd.to_numeric(df[key], errors="coerce").astype("Int64")
        if key in cand.columns:
            cand[key] = pd.to_numeric(cand[key], errors="coerce").astype("Int64")

    merged = df.merge(cand, on=["PUF_YEAR", "inc_key"], how="inner",
                      validate="1:1")
    print(f"       after inner-join to candidates: {merged.shape}", flush=True)
    lost = len(df) - len(merged)
    if lost:
        print(f"       [warn] inner-join dropped {lost:,} rows with no match", flush=True)
    return merged


# ============================================================================
# Subgroup definitions
# ============================================================================
def build_subgroups(df):
    """Yield (subgroup_name, level_label, boolean_mask)."""
    subs = []

    # 1. Age
    if "AGE_NUMBER" in df.columns:
        age = pd.to_numeric(df["AGE_NUMBER"], errors="coerce")
        subs.append(("Age",          "<65",          (age < 65) & age.notna()))
        subs.append(("Age",          ">=65",         (age >= 65) & age.notna()))
    else:
        print("  [warn] AGE_NUMBER missing — skipping age subgroup", flush=True)

    # 2. Sex
    if "SEX" in df.columns:
        sex = df["SEX"].astype("string").str.upper().str.strip()
        subs.append(("Sex",          "Male",         sex == "M"))
        subs.append(("Sex",          "Female",       sex == "F"))
    else:
        print("  [warn] SEX missing — skipping sex subgroup", flush=True)

    # 3. GCS severity (drop NaN)
    if "GCS2" in df.columns:
        g = pd.to_numeric(df["GCS2"], errors="coerce")
        sev = (g >= 3) & (g <= 8)
        mod = (g >= 9) & (g <= 12)
        mil = (g >= 13) & (g <= 15)
        subs.append(("GCS_Severity", "Severe (<=8)",  sev))
        subs.append(("GCS_Severity", "Moderate (9-12)", mod))
        subs.append(("GCS_Severity", "Mild (13-15)",  mil))
    else:
        print("  [warn] GCS2 missing — skipping severity subgroup", flush=True)

    # 4. Body injury: isolated TBI vs polytrauma
    if "max_ais_head" in df.columns and "max_ais_other" in df.columns:
        mh = pd.to_numeric(df["max_ais_head"], errors="coerce")
        mo = pd.to_numeric(df["max_ais_other"], errors="coerce")
        both = mh.notna() & mo.notna()
        iso = (mh >= 3) & (mo <= 1) & both
        poly = both & ~iso
        subs.append(("Body_Injury", "Isolated TBI",  iso))
        subs.append(("Body_Injury", "Polytrauma",    poly))
    else:
        print("  [warn] max_ais_head / max_ais_other missing — skipping body-injury subgroup",
              flush=True)

    # 5. Mechanism: penetrating vs blunt (from candidates parquet)
    if "risk_penetrating" in df.columns:
        pen = pd.to_numeric(df["risk_penetrating"], errors="coerce")
        subs.append(("Mechanism",    "Penetrating",   pen == 1))
        subs.append(("Mechanism",    "Blunt",         pen == 0))
    else:
        print("  [warn] risk_penetrating missing — skipping mechanism subgroup", flush=True)

    # 6. Transfer vs primary
    if "is_transfer" in df.columns:
        tr = pd.to_numeric(df["is_transfer"], errors="coerce")
        subs.append(("Arrival",      "Transfer",      tr == 1))
        subs.append(("Arrival",      "Primary",       tr == 0))
    else:
        print("  [warn] is_transfer missing — skipping arrival subgroup", flush=True)

    # 7. Teaching status
    if "teaching_status_binary" in df.columns:
        ts = pd.to_numeric(df["teaching_status_binary"], errors="coerce")
        subs.append(("Center",       "Teaching",      ts == 1))
        subs.append(("Center",       "Non-teaching",  ts == 0))
    else:
        print("  [warn] teaching_status_binary missing — skipping center subgroup", flush=True)

    # Fill-NA -> False for boolean masks
    clean = []
    for name, level, mask in subs:
        m = mask.fillna(False).astype(bool) if hasattr(mask, "fillna") else np.asarray(mask, bool)
        clean.append((name, level, m))
    return clean


# ============================================================================
# Per-subgroup computation
# ============================================================================
def score_block_for_subgroup(sub_df, subgroup_name, level, y_all_events):
    """Return list of row dicts, one per score, for this subgroup level."""
    rows = []
    too_small = y_all_events < MIN_EVENTS

    # Pre-extract plat arrays for delong pair computation
    pat_col = PAT_SCORE[2]
    pat_mask_col = PAT_SCORE[3]

    for short, pretty, score_col, mask_col in SCORES:
        row = {
            "Subgroup": subgroup_name,
            "Level": level,
            "Score": pretty,
            "N": np.nan,
            "N_events": np.nan,
            "AUROC": np.nan,
            "CI_Low": np.nan,
            "CI_High": np.nan,
            "Delta_vs_PAT_TBI": np.nan,
            "Delta_SE": np.nan,
            "Delta_CI_Low": np.nan,
            "Delta_CI_High": np.nan,
            "Delta_P": np.nan,
            "N_events_too_small": too_small,
        }

        if score_col not in sub_df.columns or mask_col not in sub_df.columns:
            row["N"] = 0
            row["N_events"] = 0
            rows.append(row)
            continue

        s_mask = (
            sub_df[mask_col].astype(bool).fillna(False)
            & sub_df[score_col].notna()
            & sub_df[OUTCOME_COL].notna()
        )
        s = sub_df.loc[s_mask]
        n = int(len(s))
        n_ev = int(pd.to_numeric(s[OUTCOME_COL], errors="coerce").fillna(0).sum())
        row["N"] = n
        row["N_events"] = n_ev

        if too_small or n < 20 or n_ev < 2 or (n - n_ev) < 2:
            rows.append(row)
            continue

        y = s[OUTCOME_COL].astype(float).to_numpy()
        p = s[score_col].astype(float).to_numpy()
        auc, lo, hi = bootstrap_auroc_ci(y, p)
        row["AUROC"] = auc
        row["CI_Low"] = lo
        row["CI_High"] = hi

        # Delta vs pat_tbi_score (DeLong on the intersection)
        if short == "PAT_TBI":
            row["Delta_vs_PAT_TBI"] = 0.0
            row["Delta_SE"] = 0.0
            row["Delta_CI_Low"] = 0.0
            row["Delta_CI_High"] = 0.0
            row["Delta_P"] = 1.0
        else:
            pair_mask = (
                sub_df[pat_mask_col].astype(bool).fillna(False)
                & sub_df[mask_col].astype(bool).fillna(False)
                & sub_df[pat_col].notna()
                & sub_df[score_col].notna()
                & sub_df[OUTCOME_COL].notna()
            )
            pair = sub_df.loc[pair_mask]
            if (
                len(pair) >= 20
                and pair[OUTCOME_COL].nunique() >= 2
                and int(pair[OUTCOME_COL].sum()) >= MIN_EVENTS
            ):
                yp = pair[OUTCOME_COL].astype(float).to_numpy()
                p1 = pair[pat_col].astype(float).to_numpy()
                p2 = pair[score_col].astype(float).to_numpy()
                try:
                    _, _, delta, se, _, pval = delong_test_vec(yp, p1, p2)
                    row["Delta_vs_PAT_TBI"] = delta
                    row["Delta_SE"] = se
                    if np.isfinite(delta) and np.isfinite(se):
                        row["Delta_CI_Low"] = delta - 1.96 * se
                        row["Delta_CI_High"] = delta + 1.96 * se
                    row["Delta_P"] = pval
                except Exception as e:
                    print(f"      [warn] DeLong failed ({pretty}): {e}", flush=True)

        rows.append(row)
    return rows


# ============================================================================
# Forest plot — one panel per comparator
# ============================================================================
def make_forest(out_df):
    banner("Rendering forest plot — delta AUROC (pat_tbi_score − comparator)")

    # Only keep rows for actual comparators (not pat_tbi_score itself)
    comp_pretty = [c[1] for c in COMPARATORS]
    df = out_df[out_df["Score"].isin(comp_pretty)].copy()
    if df.empty:
        print("  [warn] no comparator rows — skipping forest plot", flush=True)
        return

    # Build stable row order from the (Subgroup, Level) order we see
    row_order = []
    seen = set()
    for _, r in out_df.iterrows():
        key = (r["Subgroup"], r["Level"])
        if key not in seen:
            seen.add(key)
            row_order.append(key)
    n_rows = len(row_order)

    n_panels = len(comp_pretty)
    fig_h = max(6.0, 0.38 * n_rows + 1.8)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(4.2 * n_panels + 2.0, fig_h),
        sharey=True,
    )
    if n_panels == 1:
        axes = [axes]

    # Compute global x-range for readability
    all_deltas = df["Delta_vs_PAT_TBI"].dropna().tolist()
    all_los = df["Delta_CI_Low"].dropna().tolist()
    all_his = df["Delta_CI_High"].dropna().tolist()
    xmin = min(all_los + all_deltas + [-0.02]) if all_los else -0.05
    xmax = max(all_his + all_deltas + [0.02]) if all_his else 0.05
    pad = 0.01 + 0.05 * (xmax - xmin)
    xlim = (xmin - pad, xmax + pad)

    # Y positions: one bar per (subgroup, level), grouped by subgroup
    y_labels = []
    y_positions = []
    header_rows = []  # (y_pos, subgroup_name)
    pos = 0
    prev_sub = None
    row_to_y = {}
    for sub, lvl in row_order:
        if sub != prev_sub:
            if prev_sub is not None:
                pos += 1  # gap between subgroup blocks
            header_rows.append((pos, sub))
            pos += 1
            prev_sub = sub
        row_to_y[(sub, lvl)] = pos
        y_labels.append(f"    {lvl}")
        y_positions.append(pos)
        pos += 1

    for ax, comp in zip(axes, COMPARATORS):
        short, pretty, _, _ = comp
        sub_df = df[df["Score"] == pretty]
        ys = []
        deltas = []
        los = []
        his = []
        colors = []
        for _, r in sub_df.iterrows():
            key = (r["Subgroup"], r["Level"])
            if key not in row_to_y:
                continue
            y = row_to_y[key]
            d = r["Delta_vs_PAT_TBI"]
            lo = r["Delta_CI_Low"]
            hi = r["Delta_CI_High"]
            pval = r["Delta_P"]
            if not np.isfinite(d):
                continue
            ys.append(y)
            deltas.append(d)
            los.append(lo if np.isfinite(lo) else d)
            his.append(hi if np.isfinite(hi) else d)
            sig = np.isfinite(pval) and pval < 0.05
            colors.append("#c0392b" if sig else "#555555")

        if ys:
            ys = np.array(ys)
            deltas = np.array(deltas)
            los = np.array(los)
            his = np.array(his)
            err_lo = deltas - los
            err_hi = his - deltas
            for y, d, elo, ehi, c in zip(ys, deltas, err_lo, err_hi, colors):
                ax.errorbar(d, y, xerr=[[elo], [ehi]],
                            fmt='o', color=c, ecolor=c,
                            elinewidth=1.3, markersize=5, capsize=2.5)

        ax.axvline(0.0, color="black", lw=1.0, ls="-")
        ax.set_xlim(xlim)
        ax.set_title(f"PAT-TBI − {pretty}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Delta AUROC", fontsize=10)
        ax.grid(axis="x", ls=":", alpha=0.4)

    # Shared y-axis labels on leftmost panel
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(y_labels, fontsize=9)
    for hpos, sub_name in header_rows:
        axes[0].text(
            -0.02, hpos, sub_name,
            ha="right", va="center",
            transform=axes[0].get_yaxis_transform(),
            fontsize=10, fontweight="bold",
        )
    axes[0].invert_yaxis()

    fig.suptitle(
        "Figure. Subgroup Delta-AUROC: PAT-TBI vs comparators "
        "(red = p<0.05 via DeLong)",
        fontsize=12, fontweight="bold", y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    fig.savefig(OUT_FIG_PDF, bbox_inches="tight")
    fig.savefig(OUT_FIG_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {OUT_FIG_PDF}", flush=True)
    print(f"[write] {OUT_FIG_PNG}", flush=True)


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0,
                        help="Optional (no-op for this step): preserved for CLI parity")
    args = parser.parse_args()

    t0 = time.time()
    banner("Step 9 — PAT-TBI pre-specified subgroup AUROCs + forest (NTDB 2020-2024)")

    df = load_data()

    if OUTCOME_COL not in df.columns:
        print(f"[fatal] outcome column '{OUTCOME_COL}' not in scored parquet", flush=True)
        sys.exit(1)

    # Quick sanity snapshot
    n_total = len(df)
    n_ev = int(pd.to_numeric(df[OUTCOME_COL], errors="coerce").fillna(0).sum())
    print(f"[info] test-set rows: {n_total:,}   total events: {n_ev:,}", flush=True)

    subs = build_subgroups(df)
    print(f"[info] {len(subs)} subgroup levels defined", flush=True)

    # Compute per-subgroup x per-score rows
    all_rows = []
    for sub_name, level, mask in subs:
        sub_df = df.loc[mask].copy()
        n_sub = len(sub_df)
        y_ev = int(pd.to_numeric(sub_df[OUTCOME_COL], errors="coerce").fillna(0).sum())
        print(f"\n>>> {sub_name} | {level} : N={n_sub:,}  events={y_ev:,}", flush=True)
        if n_sub == 0:
            # emit placeholder rows
            for _, pretty, _, _ in SCORES:
                all_rows.append({
                    "Subgroup": sub_name, "Level": level, "Score": pretty,
                    "N": 0, "N_events": 0,
                    "AUROC": np.nan, "CI_Low": np.nan, "CI_High": np.nan,
                    "Delta_vs_PAT_TBI": np.nan, "Delta_SE": np.nan,
                    "Delta_CI_Low": np.nan, "Delta_CI_High": np.nan,
                    "Delta_P": np.nan,
                    "N_events_too_small": True,
                })
            continue

        rows = score_block_for_subgroup(sub_df, sub_name, level, y_ev)
        for r in rows:
            auc = r["AUROC"]
            if np.isfinite(auc):
                print(
                    f"    {r['Score']:<14s} n={int(r['N']):>7,}  "
                    f"AUC={auc:.4f} [{r['CI_Low']:.4f},{r['CI_High']:.4f}]  "
                    f"dAUC={r['Delta_vs_PAT_TBI']:+.4f}  p={r['Delta_P']:.3g}",
                    flush=True,
                )
            else:
                flag = " (events<50)" if r["N_events_too_small"] else ""
                print(f"    {r['Score']:<14s} n={int(r['N']):>7,}  AUC=NaN{flag}",
                      flush=True)
        all_rows.extend(rows)

    out_df = pd.DataFrame(all_rows)
    col_order = [
        "Subgroup", "Level", "Score", "N", "N_events",
        "AUROC", "CI_Low", "CI_High",
        "Delta_vs_PAT_TBI", "Delta_SE", "Delta_CI_Low", "Delta_CI_High", "Delta_P",
        "N_events_too_small",
    ]
    out_df = out_df[col_order]

    banner(f"Writing {OUT_TABLE}")
    out_df.to_csv(OUT_TABLE, index=False)
    print(f"  rows = {len(out_df):,}", flush=True)

    make_forest(out_df)

    # Verification summary
    banner("Verification")
    n_valid = int(out_df["AUROC"].notna().sum())
    n_small = int(out_df["N_events_too_small"].sum())
    print(f"  total rows:           {len(out_df):,}", flush=True)
    print(f"  rows with AUROC:      {n_valid:,}", flush=True)
    print(f"  rows flagged small:   {n_small:,}", flush=True)
    pat_rows = out_df[out_df["Score"] == PAT_SCORE[1]]
    if pat_rows["AUROC"].notna().any():
        print(
            f"  PAT-TBI AUROC range:  "
            f"{pat_rows['AUROC'].min():.4f} .. {pat_rows['AUROC'].max():.4f}",
            flush=True,
        )
    sig_wins = out_df[
        (out_df["Score"] != PAT_SCORE[1])
        & (out_df["Delta_vs_PAT_TBI"] > 0)
        & (out_df["Delta_P"] < 0.05)
    ]
    print(f"  significant PAT-TBI wins: {len(sig_wins)}", flush=True)
    sig_loss = out_df[
        (out_df["Score"] != PAT_SCORE[1])
        & (out_df["Delta_vs_PAT_TBI"] < 0)
        & (out_df["Delta_P"] < 0.05)
    ]
    print(f"  significant PAT-TBI losses: {len(sig_loss)}", flush=True)

    dt = time.time() - t0
    print(f"\n[done] runtime: {dt:.1f}s ({dt/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
