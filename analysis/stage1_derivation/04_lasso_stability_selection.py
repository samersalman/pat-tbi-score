"""
PAT-TBI v4 — Step 4: LASSO stability selection (parsimonious refit)
================================================================================

Runs a per-bootstrap LASSO with **lambda.1se equivalent** C selection on
the training split. For each of 100 bootstrap iterations, a 250,000-row
subsample is drawn with replacement; on that subsample we (a) run
LogisticRegressionCV (L1, saga) to obtain per-fold validation AUROCs at
each candidate C, (b) compute the lambda.1se C as the smallest C (i.e.
strongest L1 penalty / sparsest model) whose mean CV AUROC is within 1
SE of the best mean CV AUROC, and (c) refit a single LogisticRegression
at that C and accumulate nonzero-coefficient indicators. Selection
frequency per feature is accumulated; features with frequency >= 0.85
become the final LASSO-selected set. **No fallback** — if fewer than 5
features hit 0.85, a warning is printed and the script proceeds (Step 5
handles a small set explicitly; per the v4 plan, fallback is human-driven,
not automatic).

Why lambda.1se (not lambda.min)?
--------------------------------
sklearn's LogisticRegressionCV picks C maximizing CV AUROC by default —
this is the lambda.min equivalent. v4's parsimony objective prefers
lambda.1se (Hastie/Tibshirani/Friedman, ESL §7.10): the sparsest model
whose CV performance is statistically indistinguishable from the best
(within 1 standard error). In sklearn's parametrization, smaller C means
stronger L1 regularization, so the lambda.1se C is the *smallest* C with
mean CV AUROC >= (best mean - 1 SE).

Cohort years: 2020-2024.

Inputs
------
  PAT_TBI/v4/derived/candidates.parquet         (from Step 2 — numeric, NaN-free
                                                  for the LASSO_X_COLS listed below)
  PAT_TBI/v4/derived/train_inc_keys.csv         (from Step 3 — (PUF_YEAR, inc_key)
                                                  join keys for the derivation split)

Outputs
-------
  PAT_TBI/v4/derived/lasso_cv_fit.pkl          pickle dict:
      {
        "best_C":        float,    (median of per-bootstrap lambda.1se C values)
        "scaler_mean":   np.ndarray (n_features,),
        "scaler_scale":  np.ndarray (n_features,),
        "feature_names": list[str]   (order matches scaler and cv_beta),
        "cv_beta":       np.ndarray (n_features,)  (on scaled X, refit on full
                                                    train at the global lambda.1se C)
      }
  PAT_TBI/v4/tables/table_lasso_stability.csv
      columns: Variable, Selection_Frequency, CV_Beta, CV_Abs_Beta_Rank
  PAT_TBI/v4/tables/table_lasso_selected.csv
      columns: Variable, Selection_Frequency, CV_Beta
      (only rows where Selection_Frequency >= 0.85 — no fallback)

Output contract for Step 5
--------------------------
  * `table_lasso_selected.csv` has columns
    Variable, Selection_Frequency, CV_Beta
  * `lasso_cv_fit.pkl` contains the StandardScaler parameters
    (`scaler_mean`, `scaler_scale`) so Step 5 can either refit unpenalized on
    the RAW (not scaled) values or re-scale with the identical transform.
  * `best_C` is stored as a plain Python float in the pickle. (For v4 this is
    the global lambda.1se C computed on the full training data; per-bootstrap
    C values are used internally during stability selection but not exported.)
  * `feature_names` order matches the columns of `scaler_mean`, `scaler_scale`,
    and `cv_beta` 1:1.

Algorithm
---------
  1. Load candidates parquet; inner-join to train_inc_keys on (PUF_YEAR, inc_key).
  2. X_train = df[LASSO_X_COLS].values; y_train = df['received_platelets_4h'].values.
  3. Fit StandardScaler on X_train; cache mean_/scale_ for the pickle.
  4. X_scaled = scaler.transform(X_train).
  5. GLOBAL lambda.1se: LogisticRegressionCV(penalty='l1', solver='saga',
       Cs=50, cv=5, scoring='roc_auc', max_iter=10000, n_jobs=-1,
       random_state=42) on the full X_scaled, then derive lambda.1se C from
       clf.scores_; refit a single LogisticRegression at that C to obtain
       cv_beta for the pickle. (This C is also a useful summary; it is
       reported in the banner.)
  6. PER-BOOTSTRAP lambda.1se: 100 bootstrap iterations, each drawing 250k
       rows with replacement; each bootstrap must contain >= 500 events
       (redraw with a distinct seed if not). At each iteration:
         (a) LogisticRegressionCV(...) on the bootstrap → derive lambda.1se C
             from clf.scores_;
         (b) refit LogisticRegression(penalty='l1', solver='saga', C=C_1se,
             max_iter=5000) on the bootstrap and accumulate nonzero-coefficient
             indicators into a 27-dim counter.
  7. selection_freq = selection_counts / n_bootstraps.
  8. Threshold at 0.85 (no fallback). Print a warning if n_selected < 5.

Guards
------
  * Each bootstrap enforced >= 500 events; redrawn with a distinct seed
    (42 + i + 10000) if the draw fails.
  * No automatic threshold relaxation (per v4 plan; relaxation is human-driven).

Run
---
    python 04_lasso_stability_selection.py
    python 04_lasso_stability_selection.py --bootstraps 10 --subsample 25000
        # smoke test with reduced bootstraps and subsample size

Seed: 42
Runtime: ~40-90 min on a workstation (full run; per-bootstrap CV is more
expensive than v3's single global CV, but acceptable for a one-off refit).
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)

import os as _os, sys as _sys  # path handling: make analysis/config.py importable
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import config  # noqa: E402  (analysis/config.py: PAT_TBI_WORKDIR, PAT_TBI_NTDB_DUCKDB)
PAT = str(config.V4_DIR)
DERIVED = os.path.join(PAT, "derived")
TABLES = os.path.join(PAT, "tables")
os.makedirs(DERIVED, exist_ok=True)
os.makedirs(TABLES, exist_ok=True)

CAND_PARQUET = os.path.join(DERIVED, "candidates.parquet")
TRAIN_KEYS_CSV = os.path.join(DERIVED, "train_inc_keys.csv")

PKL_OUT = os.path.join(DERIVED, "lasso_cv_fit.pkl")
TABLE_STABILITY = os.path.join(TABLES, "table_lasso_stability.csv")
TABLE_SELECTED = os.path.join(TABLES, "table_lasso_selected.csv")

# LASSO candidate X columns (27 ED-rapid predictors, v4 parsimonious).
# Order MUST match 02_engineer_features.py exactly for downstream
# reproducibility (StandardScaler + cv_beta column alignment).
LASSO_X_COLS = [
    # Demographics (4)
    "age_45_64", "age_65_74", "age_75plus", "female",
    # ED vitals (11) — shock_index kept continuous
    "sbp_hypotensive", "sbp_hypertensive",
    "hr_bradycardic", "hr_tachycardic",
    "rr_low", "rr_high",
    "hypoxic",
    "shock_index",
    "si_elevated", "si_high",
    "temp_hypothermic",
    # GCS (4)
    "gcs_severe", "gcs_moderate", "gcs_motor_low", "gcs_motor_high",
    # Pupils (1)
    "tbi_pupil_none",
    # Mechanism (4)
    "mech_mvc", "mech_assault", "mech_fall", "risk_penetrating",
    # Transfer (1)
    "is_transfer",
    # Coagulation-relevant comorbidities (2)
    "cc_anticoag", "cc_bleeding",
]

OUTCOME_COL = "received_platelets_4h"
JOIN_KEYS = ["PUF_YEAR", "inc_key"]

# Stability threshold (v4: parsimonious refit; no fallback)
STABILITY_THRESHOLD = 0.85
MIN_SELECTED_WARN = 5  # warn (do not auto-relax) if fewer than this many vars selected

# Defaults (full run); overridable with --bootstraps / --subsample for smoke test
DEFAULT_N_BOOTSTRAPS = 100
DEFAULT_SUBSAMPLE = 250_000
DEFAULT_MIN_EVENTS = 500

SAMPLE_MIN_EVENTS = 20

# CV grid for inner LogisticRegressionCV (used both globally and per-bootstrap)
CV_CS = 50
CV_FOLDS = 5


def banner(msg):
    line = "=" * 78
    print(f"\n{line}\n{msg}\n{line}", flush=True)


def log(msg):
    print(msg, flush=True)


def load_data():
    banner("STEP 1 — Load candidates and restrict to train split")
    if not os.path.exists(CAND_PARQUET):
        raise FileNotFoundError(CAND_PARQUET)
    if not os.path.exists(TRAIN_KEYS_CSV):
        raise FileNotFoundError(TRAIN_KEYS_CSV)

    log(f"Reading candidates : {CAND_PARQUET}")
    df = pd.read_parquet(CAND_PARQUET)
    log(f"  candidates shape : {df.shape}")

    log(f"Reading train keys : {TRAIN_KEYS_CSV}")
    keys = pd.read_csv(TRAIN_KEYS_CSV)
    log(f"  train keys shape : {keys.shape}")

    # Sanity-check join keys
    for k in JOIN_KEYS:
        if k not in df.columns:
            raise KeyError(f"Candidates parquet missing join key: {k}")
        if k not in keys.columns:
            raise KeyError(f"Train keys CSV missing join key: {k}")

    keys = keys[JOIN_KEYS].drop_duplicates()
    df_train = df.merge(keys, on=JOIN_KEYS, how="inner")
    log(f"  after inner-join : {df_train.shape}")

    # Check required columns
    missing = [c for c in LASSO_X_COLS if c not in df_train.columns]
    if missing:
        raise KeyError(
            f"Candidates parquet missing {len(missing)} LASSO_X_COLS: {missing}"
        )
    if OUTCOME_COL not in df_train.columns:
        raise KeyError(f"Candidates parquet missing outcome column: {OUTCOME_COL}")

    # NaN guard (Step 2 claims NaN-free, but verify cheaply)
    na_counts = df_train[LASSO_X_COLS + [OUTCOME_COL]].isna().sum()
    bad = na_counts[na_counts > 0]
    if len(bad) > 0:
        raise ValueError(
            f"Unexpected NaNs in LASSO inputs (Step 2 should have cleaned):\n{bad}"
        )

    X = df_train[LASSO_X_COLS].to_numpy(dtype=np.float64)
    y = df_train[OUTCOME_COL].to_numpy(dtype=np.int8)

    log(f"  X shape          : {X.shape}")
    log(f"  y shape          : {y.shape}")
    log(f"  y positives      : {int(y.sum())} ({y.mean() * 100:.2f}%)")

    return X, y


def pick_lambda_1se_C(clf):
    """
    Compute lambda.1se equivalent C from a fitted LogisticRegressionCV.

    glmnet's lambda.1se rule: among all lambdas (= 1/C in sklearn) whose mean
    CV score is within 1 SE of the best mean CV score, pick the largest
    (sparsest model). In sklearn parametrization that means picking the
    *smallest* C among those satisfying mean_score >= best_mean - 1 * SE.

    Parameters
    ----------
    clf : fitted sklearn.linear_model.LogisticRegressionCV
        Must have been fit with cv > 1 so that clf.scores_ is populated.

    Returns
    -------
    C_1se : float
        The lambda.1se equivalent C (smallest C within 1 SE of best mean CV).
    diagnostics : dict
        keys: best_idx, best_mean, best_se, threshold, idx_1se, candidate_count
    """
    # clf.scores_ is a dict keyed by class label; for binary classification
    # there is a single key. Take whichever key is present.
    keys = list(clf.scores_.keys())
    if len(keys) != 1:
        # Multi-class fallback: average across classes (shouldn't happen here,
        # but be defensive).
        scores = np.mean(np.stack([clf.scores_[k] for k in keys], axis=0), axis=0)
    else:
        scores = clf.scores_[keys[0]]
    # scores shape: (n_folds, n_Cs)
    n_folds = scores.shape[0]
    mean_score = scores.mean(axis=0)
    se_score = scores.std(axis=0, ddof=1) / np.sqrt(n_folds)

    best_idx = int(np.argmax(mean_score))
    best_mean = float(mean_score[best_idx])
    best_se = float(se_score[best_idx])
    threshold = best_mean - best_se

    # Cs vector (sklearn stores the actual C values used, in the order they
    # appear along axis=1 of scores).
    Cs = np.asarray(clf.Cs_, dtype=np.float64)
    # Candidate indices: those whose mean CV score is within 1 SE of the best.
    candidate_mask = mean_score >= threshold
    if not candidate_mask.any():
        # Defensive: should never happen since best_idx itself is a candidate.
        C_1se = float(Cs[best_idx])
        idx_1se = best_idx
    else:
        # Among candidates, pick smallest C (= largest lambda = sparsest).
        candidate_Cs = Cs.copy()
        candidate_Cs[~candidate_mask] = np.inf  # mask out non-candidates
        idx_1se = int(np.argmin(candidate_Cs))
        C_1se = float(Cs[idx_1se])

    diagnostics = {
        "best_idx": best_idx,
        "best_C": float(Cs[best_idx]),
        "best_mean": best_mean,
        "best_se": best_se,
        "threshold": float(threshold),
        "idx_1se": idx_1se,
        "C_1se": C_1se,
        "candidate_count": int(candidate_mask.sum()),
        "n_Cs": int(len(Cs)),
    }
    return C_1se, diagnostics


def fit_lasso_cv_global(X_scaled, y):
    """
    Fit LogisticRegressionCV on the full training data and return the
    lambda.1se C plus the refit coefficients at that C. This is a global
    summary used to populate `cv_beta` in the pickle and to log the global
    1se C in the banner; per-bootstrap selection still recomputes its own
    1se C inside the bootstrap loop.
    """
    banner("STEP 2 — Global LogisticRegressionCV (lambda.1se) on full train")
    log("Fitting LogisticRegressionCV ...")
    log(
        f"  penalty='l1', solver='saga', Cs={CV_CS}, cv={CV_FOLDS}, "
        "scoring='roc_auc',"
    )
    log("  max_iter=10000, n_jobs=-1, random_state=42")
    t0 = time.time()
    clf = LogisticRegressionCV(
        penalty="l1",
        solver="saga",
        Cs=CV_CS,
        cv=CV_FOLDS,
        scoring="roc_auc",
        max_iter=10000,
        n_jobs=-1,
        random_state=SEED,
        refit=True,
    )
    clf.fit(X_scaled, y)
    dt = time.time() - t0
    log(f"  LogisticRegressionCV fit time: {dt:.1f}s ({dt / 60:.1f} min)")

    C_1se, diag = pick_lambda_1se_C(clf)
    log(f"  best_C (lambda.min equiv) = {diag['best_C']:.6g}")
    log(f"  best mean CV AUROC        = {diag['best_mean']:.4f}")
    log(f"  best SE                   = {diag['best_se']:.4f}")
    log(f"  1se threshold (mean - SE) = {diag['threshold']:.4f}")
    log(f"  C_1se (lambda.1se equiv)  = {C_1se:.6g}")
    log(
        f"  candidates within 1 SE    = "
        f"{diag['candidate_count']} / {diag['n_Cs']}"
    )

    # Refit at C_1se to obtain "cv_beta" (used as the pickle's coefficient
    # snapshot at the 1se penalty).
    log("Refitting LogisticRegression at C_1se on full train ...")
    lr_1se = LogisticRegression(
        penalty="l1",
        solver="saga",
        C=C_1se,
        max_iter=10000,
        random_state=SEED,
        n_jobs=-1,
    )
    lr_1se.fit(X_scaled, y)
    cv_beta = lr_1se.coef_[0].astype(np.float64)
    log(f"  cv_beta nonzero count = {int(np.sum(cv_beta != 0))} / {len(cv_beta)}")

    return C_1se, cv_beta


def fit_lasso_cv(X, y):
    """
    Fit StandardScaler and the global lambda.1se LogisticRegressionCV.
    Returns X_scaled, scaler, global C_1se, and cv_beta (refit at C_1se).
    """
    log("Fitting StandardScaler on X_train ...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    log(f"  scaler.mean_[:5]  = {np.round(scaler.mean_[:5], 4).tolist()}")
    log(f"  scaler.scale_[:5] = {np.round(scaler.scale_[:5], 4).tolist()}")

    C_1se, cv_beta = fit_lasso_cv_global(X_scaled, y)
    return X_scaled, scaler, C_1se, cv_beta


def run_bootstraps(X_scaled, y, n_bootstraps, subsample, min_events):
    """
    Per-bootstrap lambda.1se stability selection.

    For each bootstrap:
      1. Draw a 250k-row subsample (with replacement); enforce >=500 events.
      2. Fit LogisticRegressionCV on the bootstrap to obtain per-fold AUROCs
         at each candidate C, then derive lambda.1se C via pick_lambda_1se_C.
      3. Refit a single LASSO at that C and record nonzero-coefficient
         indicators.

    Per-bootstrap (rather than once-globally) lambda.1se selection is the
    standard glmnet-style stability selection approach: it is honest about
    CV variance across resamples and avoids fixing a single global C that
    may not be optimally sparse on every bootstrap.
    """
    banner(
        f"STEP 3 — {n_bootstraps} per-bootstrap lambda.1se LASSO iterations "
        f"(n_sub={subsample:,}, min_events={min_events})"
    )
    n_features = X_scaled.shape[1]
    n_rows = X_scaled.shape[0]
    selection_counts = np.zeros(n_features, dtype=np.int64)
    boot_C_1se_values = np.zeros(n_bootstraps, dtype=np.float64)
    t_start = time.time()

    for i in range(n_bootstraps):
        t_iter = time.time()
        rng = np.random.RandomState(SEED + i)
        idx = rng.choice(n_rows, size=subsample, replace=True)
        yb = y[idx]

        redraws = 0
        while yb.sum() < min_events:
            redraws += 1
            rng2 = np.random.RandomState(SEED + i + 10000 * redraws)
            idx = rng2.choice(n_rows, size=subsample, replace=True)
            yb = y[idx]
            if redraws > 25:
                raise RuntimeError(
                    f"Bootstrap {i}: failed to draw >= {min_events} events "
                    f"after {redraws} redraws"
                )

        Xb = X_scaled[idx]

        # (a) per-bootstrap CV to get lambda.1se C
        inner_cv = LogisticRegressionCV(
            penalty="l1",
            solver="saga",
            Cs=CV_CS,
            cv=CV_FOLDS,
            scoring="roc_auc",
            max_iter=5000,
            n_jobs=-1,
            random_state=SEED + i,
            refit=False,  # we only need scores_; we'll refit ourselves
        )
        inner_cv.fit(Xb, yb)
        C_1se, _diag = pick_lambda_1se_C(inner_cv)
        boot_C_1se_values[i] = C_1se

        # (b) refit single LASSO at C_1se
        lr = LogisticRegression(
            penalty="l1",
            solver="saga",
            C=C_1se,
            max_iter=5000,
            random_state=SEED + i,
            n_jobs=-1,
        )
        lr.fit(Xb, yb)
        nonzero_mask = (lr.coef_[0] != 0)
        selection_counts += nonzero_mask.astype(np.int64)

        dt_iter = time.time() - t_iter
        dt_total = time.time() - t_start
        eta = (dt_total / (i + 1)) * (n_bootstraps - (i + 1))
        log(
            f"  boot {i + 1:3d}/{n_bootstraps} | "
            f"events={int(yb.sum()):5d} | redraws={redraws} | "
            f"C_1se={C_1se:.4g} | "
            f"nonzero={int(nonzero_mask.sum()):2d}/{n_features} | "
            f"iter={dt_iter:5.1f}s | elapsed={dt_total / 60:5.1f}min | "
            f"eta={eta / 60:5.1f}min"
        )

    selection_freq = selection_counts / float(n_bootstraps)
    log(f"  selection_counts.sum() = {int(selection_counts.sum())}")
    log(
        f"  per-bootstrap C_1se: median={np.median(boot_C_1se_values):.4g}, "
        f"min={boot_C_1se_values.min():.4g}, max={boot_C_1se_values.max():.4g}"
    )
    return selection_freq


def build_tables(feature_names, cv_beta, selection_freq):
    banner("STEP 4 — Build stability and selected tables (threshold=0.85, no fallback)")
    abs_beta = np.abs(cv_beta)
    # Rank by |cv_beta| descending (1 = largest). Zeros get the worst ranks.
    order = np.argsort(-abs_beta, kind="stable")
    rank = np.empty(len(feature_names), dtype=np.int64)
    rank[order] = np.arange(1, len(feature_names) + 1)

    stability = pd.DataFrame(
        {
            "Variable": feature_names,
            "Selection_Frequency": np.round(selection_freq, 4),
            "CV_Beta": cv_beta,
            "CV_Abs_Beta_Rank": rank,
        }
    ).sort_values(
        ["Selection_Frequency", "CV_Abs_Beta_Rank"],
        ascending=[False, True],
    ).reset_index(drop=True)

    log(f"Writing stability table: {TABLE_STABILITY}")
    stability.to_csv(TABLE_STABILITY, index=False)
    log(f"  rows = {len(stability)}")

    # v4: hard threshold at 0.85, no fallback. Step 05 handles small sets.
    threshold = STABILITY_THRESHOLD
    selected = stability[stability["Selection_Frequency"] >= threshold].copy()

    if len(selected) < MIN_SELECTED_WARN:
        log(
            f"  WARNING: only {len(selected)} feature(s) hit "
            f"Selection_Frequency >= {threshold:.2f} "
            f"(< MIN_SELECTED_WARN={MIN_SELECTED_WARN}). "
            "v4 does NOT auto-relax — this small set is being passed to Step 5 "
            "as data. Per the v4 plan, any threshold relaxation must be a "
            "human-driven decision documented in the risk-fallback section."
        )

    selected = selected[["Variable", "Selection_Frequency", "CV_Beta"]].reset_index(
        drop=True
    )
    log(f"Writing selected table: {TABLE_SELECTED}")
    log(f"  threshold applied = {threshold:.2f}")
    log(f"  selected rows     = {len(selected)}")
    selected.to_csv(TABLE_SELECTED, index=False)

    return stability, selected, threshold


def write_pickle(scaler, best_C, cv_beta, feature_names):
    banner("STEP 5 — Write lasso_cv_fit.pkl")
    payload = {
        "best_C": float(best_C),
        "scaler_mean": np.asarray(scaler.mean_, dtype=np.float64),
        "scaler_scale": np.asarray(scaler.scale_, dtype=np.float64),
        "feature_names": list(feature_names),
        "cv_beta": np.asarray(cv_beta, dtype=np.float64),
    }
    with open(PKL_OUT, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    log(f"Wrote pickle: {PKL_OUT}")
    log(f"  best_C (lambda.1se) = {payload['best_C']:.6g}")
    log(f"  scaler_mean shape   = {payload['scaler_mean'].shape}")
    log(f"  scaler_scale shape  = {payload['scaler_scale'].shape}")
    log(f"  n feature_names     = {len(payload['feature_names'])}")
    log(f"  cv_beta shape       = {payload['cv_beta'].shape}")


def main():
    parser = argparse.ArgumentParser(
        description="PAT-TBI v4 Step 4: LASSO stability selection (lambda.1se, threshold=0.85)"
    )
    parser.add_argument(
        "--bootstraps",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Number of bootstrap iterations. Default: 100 (full run). "
            "Use a smaller value for smoke tests."
        ),
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Bootstrap subsample size. Default: 250000 (full run). "
            "Use a smaller value for smoke tests."
        ),
    )
    args = parser.parse_args()

    banner("PAT-TBI v4 — Step 4: LASSO stability selection (lambda.1se, threshold=0.85)")
    log(f"SEED                = {SEED}")
    log(f"Candidates parquet  = {CAND_PARQUET}")
    log(f"Train keys CSV      = {TRAIN_KEYS_CSV}")
    log(f"Pickle out          = {PKL_OUT}")
    log(f"Stability table out = {TABLE_STABILITY}")
    log(f"Selected table out  = {TABLE_SELECTED}")
    log(f"# LASSO_X_COLS      = {len(LASSO_X_COLS)}")
    log(f"Stability threshold = {STABILITY_THRESHOLD:.2f} (no fallback)")
    log("Lambda rule         = 1se equivalent (smallest C within 1 SE of best CV AUROC)")

    smoke_mode = (args.bootstraps is not None) or (args.subsample is not None)

    n_bootstraps = args.bootstraps if args.bootstraps is not None else DEFAULT_N_BOOTSTRAPS
    subsample = args.subsample if args.subsample is not None else DEFAULT_SUBSAMPLE
    min_events = SAMPLE_MIN_EVENTS if smoke_mode else DEFAULT_MIN_EVENTS

    if smoke_mode:
        log(
            f"SMOKE TEST MODE     : {n_bootstraps} bootstraps, "
            f"subsample={subsample:,}, min_events={min_events}"
        )
    else:
        log(
            f"FULL RUN            : {n_bootstraps} bootstraps, "
            f"subsample={subsample:,}, min_events={min_events}"
        )

    X, y = load_data()

    # If subsample > n_rows, np.random.choice with replace=True still works;
    # warn so a smoke test on a tiny dataset doesn't silently mislead.
    if subsample > len(X):
        log(
            f"  NOTE: subsample ({subsample:,}) > rows ({len(X):,}); "
            f"bootstraps will oversample with replacement."
        )

    X_scaled, scaler, best_C, cv_beta = fit_lasso_cv(X, y)

    selection_freq = run_bootstraps(
        X_scaled, y,
        n_bootstraps=n_bootstraps,
        subsample=subsample,
        min_events=min_events,
    )

    stability, selected, threshold = build_tables(
        LASSO_X_COLS, cv_beta, selection_freq
    )

    write_pickle(scaler, best_C, cv_beta, LASSO_X_COLS)

    banner("DONE")
    log(f"global lambda.1se C              = {best_C:.6g}")
    log(f"n bootstraps                     = {n_bootstraps}")
    log(f"threshold applied                = {threshold:.2f} (no fallback)")
    log(f"n features selected              = {len(selected)}")
    log(
        "selected variables               = "
        + ", ".join(selected["Variable"].tolist())
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
