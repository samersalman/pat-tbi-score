"""
PAT-TBI Native NTDB Refit — Step 5: Unpenalized refit + integer conversion
================================================================================

Cohort years: 2020-2024.

Takes the LASSO-stability-selected feature set from Step 4, refits an UNPENALIZED
logistic regression on the training rows to recover unbiased β with analytical
SEs, drops features with |β| < 0.15 (OR < 1.16) or VIF>5, and converts to the
integer point system via round(β/|β_min_positive|) with a ±1 floor.

Inputs
------
  PAT_TBI/v4/derived/candidates.parquet            (Step 2)
  PAT_TBI/v4/derived/train_inc_keys.csv            (Step 3)
  PAT_TBI/v4/tables/table_lasso_selected.csv       (Step 4)

Outputs
-------
  PAT_TBI/v4/tables/table_pat_weights.csv
    Schema: Variable, β, SE, OR, 95% CI lower, 95% CI upper, Points
    (Row count is arbitrary — PAT-TBI may retain 5–15 predictors.)
  PAT_TBI/v4/derived/pat_refit.json
    Keys: intercept, coefficients, points, variables_kept, variables_dropped_beta,
          variables_dropped_vif, training_score_distribution, training_auroc

Algorithm
---------
  1. Inner-join candidates → train_inc_keys (raw-scale design matrix, not scaled).
  2. statsmodels Logit fit with constant.
  3. Drop features where |β| < 0.15 (OR < 1.16) → refit. (Parsimonious v4 cutoff.)
  4. VIF loop: drop highest-VIF feature if VIF > 5; repeat until all VIF ≤ 5.
     Tiebreak: drop the lowest-stability-frequency member.
  5. Integer conversion:
       anchor = min(positive β among retained)
       points[v] = round(β_v / anchor)
       ±1 floor if |β_v| >= 0.15 and round is 0
  6. Report training score distribution + training AUROC.

Output contract for Step 6
--------------------------
  Step 6 uses its own compute_pat_tbi() loader (no hard-coded length) to
  drop in the new weights with only a WEIGHTS_CSV path swap.
"""
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import roc_auc_score

SEED = 42
np.random.seed(SEED)

import os as _os, sys as _sys  # path handling: make analysis/config.py importable
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import config  # noqa: E402  (analysis/config.py: PAT_TBI_WORKDIR, PAT_TBI_NTDB_DUCKDB)
PAT = str(config.V4_DIR)
DERIVED = os.path.join(PAT, "derived")
TABLES = os.path.join(PAT, "tables")

CANDIDATES = os.path.join(DERIVED, "candidates.parquet")
TRAIN_KEYS = os.path.join(DERIVED, "train_inc_keys.csv")
LASSO_SEL = os.path.join(TABLES, "table_lasso_selected.csv")
OUT_WEIGHTS = os.path.join(TABLES, "table_pat_weights.csv")
OUT_JSON = os.path.join(DERIVED, "pat_refit.json")

BETA_DROP = 0.15
VIF_MAX = 5.0
MIN_KEPT = 5


def banner(msg):
    line = "=" * 78
    print(f"\n{line}\n{msg}\n{line}", flush=True)


def fit_logit(X_df, y):
    """Unpenalized MLE logistic. Returns the results object."""
    Xc = sm.add_constant(X_df, has_constant="add")
    model = sm.Logit(y, Xc.astype(float))
    return model.fit(disp=False, maxiter=200, method="newton")


def compute_vif(X_df):
    """Return dict var→VIF for each non-constant column in X_df."""
    Xc = sm.add_constant(X_df, has_constant="add").astype(float).values
    cols = ["const"] + list(X_df.columns)
    vifs = {}
    for i, name in enumerate(cols):
        if name == "const":
            continue
        try:
            vifs[name] = float(variance_inflation_factor(Xc, i))
        except Exception:
            vifs[name] = float("nan")
    return vifs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0,
                        help="Noop passthrough — honored only by upstream steps")
    args = parser.parse_args()

    banner("PAT-TBI v4 — Step 5: Unpenalized refit + integer conversion "
           "(parsimonious cutoff |β| ≥ 0.15, OR ≥ 1.16; cohort 2020-2024)")

    # ----------------------------------------------------------------
    # Load inputs
    # ----------------------------------------------------------------
    print(f"Loading candidates: {CANDIDATES}", flush=True)
    df = pd.read_parquet(CANDIDATES)
    df["inc_key"] = pd.to_numeric(df["inc_key"], errors="coerce").astype("Int64")
    df["PUF_YEAR"] = pd.to_numeric(df["PUF_YEAR"], errors="coerce").astype("Int64")
    print(f"  candidates shape: {df.shape}", flush=True)

    print(f"Loading train inc_keys: {TRAIN_KEYS}", flush=True)
    train_keys = pd.read_csv(TRAIN_KEYS)
    train_keys["inc_key"] = pd.to_numeric(train_keys["inc_key"], errors="coerce").astype("Int64")
    train_keys["PUF_YEAR"] = pd.to_numeric(train_keys["PUF_YEAR"], errors="coerce").astype("Int64")
    print(f"  train rows: {len(train_keys):,}", flush=True)

    df = df.merge(train_keys, on=["PUF_YEAR", "inc_key"], how="inner")
    print(f"  after inner-join to train: {df.shape}", flush=True)
    assert len(df) == len(train_keys), "train join row count drift"

    print(f"Loading LASSO selected: {LASSO_SEL}", flush=True)
    sel = pd.read_csv(LASSO_SEL)
    print(f"  {len(sel)} selected variables from stability selection", flush=True)
    freq_lookup = dict(zip(sel["Variable"].astype(str), sel["Selection_Frequency"].astype(float)))

    variables = list(sel["Variable"].astype(str))
    assert all(v in df.columns for v in variables), \
        f"Missing selected vars in candidates: {[v for v in variables if v not in df.columns]}"

    # ----------------------------------------------------------------
    # Initial unpenalized fit
    # ----------------------------------------------------------------
    banner("Initial unpenalized Logit fit")
    X = df[variables].astype(float)
    y = df["received_platelets_4h"].astype(int).values
    res = fit_logit(X, y)
    print(f"  converged: {res.mle_retvals.get('converged', True)}", flush=True)
    print(f"  n_obs = {int(res.nobs):,}   n_features = {len(variables)}", flush=True)
    print(f"  intercept β = {float(res.params.iloc[0]):.4f}", flush=True)
    n_before_cutoff = len(variables)
    print(f"  variables before |β| cutoff: {n_before_cutoff}", flush=True)

    # ----------------------------------------------------------------
    # Drop |β| < 0.15 (OR < 1.16), refit  — v4 parsimonious cutoff
    # ----------------------------------------------------------------
    banner(f"Dropping features with |β| < {BETA_DROP} (OR < {np.exp(BETA_DROP):.2f})")
    beta = res.params.drop("const")
    weak = beta[beta.abs() < BETA_DROP].index.tolist()
    print(f"  weak features (|β|<{BETA_DROP}): {weak}", flush=True)
    dropped_beta = list(weak)
    variables = [v for v in variables if v not in weak]
    n_after_cutoff = len(variables)
    print(f"  variables before |β|≥{BETA_DROP} cutoff: {n_before_cutoff}", flush=True)
    print(f"  variables after  |β|≥{BETA_DROP} cutoff: {n_after_cutoff}", flush=True)
    assert len(variables) >= MIN_KEPT, f"Only {len(variables)} vars left after β drop"

    X = df[variables].astype(float)
    res = fit_logit(X, y)
    print(f"  refit with {len(variables)} features", flush=True)

    # ----------------------------------------------------------------
    # VIF loop (drop highest until max VIF ≤ 5)
    # ----------------------------------------------------------------
    banner(f"VIF loop — dropping features with VIF > {VIF_MAX}")
    dropped_vif = []
    for iteration in range(20):
        vifs = compute_vif(X)
        max_var = max(vifs, key=lambda k: vifs[k])
        max_vif = vifs[max_var]
        print(f"  iter {iteration:2d}: max VIF = {max_vif:.2f} ({max_var})", flush=True)
        if max_vif <= VIF_MAX:
            break
        # Tiebreak: if multiple vars share max VIF within 0.5, drop the lowest-frequency one
        near_max = [v for v, q in vifs.items() if q >= max_vif - 0.5]
        if len(near_max) > 1:
            drop_var = min(near_max, key=lambda v: freq_lookup.get(v, 0.0))
        else:
            drop_var = max_var
        print(f"    → dropping {drop_var} (freq={freq_lookup.get(drop_var, 0):.2f})", flush=True)
        dropped_vif.append(drop_var)
        variables = [v for v in variables if v != drop_var]
        if len(variables) < MIN_KEPT:
            raise RuntimeError(f"VIF loop reduced features below MIN_KEPT={MIN_KEPT}")
        X = df[variables].astype(float)
        res = fit_logit(X, y)
    else:
        raise RuntimeError("VIF loop did not converge in 20 iterations")

    # ----------------------------------------------------------------
    # Extract final β / SE / CI / OR
    # ----------------------------------------------------------------
    banner("Final coefficients")
    params = res.params
    bse = res.bse
    conf = res.conf_int()
    conf.columns = ["CI_low", "CI_high"]

    intercept = float(params.loc["const"])
    print(f"  intercept = {intercept:.4f}", flush=True)

    rows = []
    for v in variables:
        b = float(params.loc[v])
        se = float(bse.loc[v])
        ci_lo = float(conf.loc[v, "CI_low"])
        ci_hi = float(conf.loc[v, "CI_high"])
        odds = float(np.exp(b))
        rows.append({
            "Variable": v,
            "β": round(b, 4),
            "SE": round(se, 4),
            "OR": round(odds, 3),
            "95% CI lower": round(float(np.exp(ci_lo)), 3),
            "95% CI upper": round(float(np.exp(ci_hi)), 3),
            "_beta_raw": b,
        })

    # ----------------------------------------------------------------
    # Integer conversion
    # ----------------------------------------------------------------
    banner("Integer conversion")
    positive = {r["Variable"]: r["_beta_raw"] for r in rows if r["_beta_raw"] > 0}
    if not positive:
        raise RuntimeError("No positive-β risk factors retained — refuse to build a score")
    anchor = min(positive.values())
    print(f"  anchor |β_min_positive| = {anchor:.4f}", flush=True)

    for r in rows:
        b = r["_beta_raw"]
        raw_pts = b / anchor
        pts = int(round(raw_pts))
        # ±1 floor for features that made it through but would round to 0
        if pts == 0 and abs(b) >= BETA_DROP:
            pts = 1 if b > 0 else -1
        # Preserve sign direction
        if b > 0:
            assert pts >= 0
        else:
            assert pts <= 0
        r["Points"] = pts

    # Build final DataFrame in schema order; drop _beta_raw
    weights_df = pd.DataFrame([{
        "Variable":       r["Variable"],
        "β":              r["β"],
        "SE":             r["SE"],
        "OR":             r["OR"],
        "95% CI lower":   r["95% CI lower"],
        "95% CI upper":   r["95% CI upper"],
        "Points":         r["Points"],
    } for r in rows])

    # Order columns explicitly
    weights_df = weights_df[[
        "Variable", "β", "SE", "OR", "95% CI lower", "95% CI upper", "Points"
    ]]

    print(weights_df.to_string(index=False), flush=True)

    weights_df.to_csv(OUT_WEIGHTS, index=False)
    print(f"\n  wrote {OUT_WEIGHTS}", flush=True)

    # ----------------------------------------------------------------
    # Training score distribution + AUROC
    # ----------------------------------------------------------------
    banner("Training score distribution + AUROC")
    score = np.zeros(len(df), dtype=np.int64)
    for _, r in weights_df.iterrows():
        score += int(r["Points"]) * df[r["Variable"]].astype(int).values

    train_auroc = float(roc_auc_score(y, score))
    dist = {
        "min":    int(np.min(score)),
        "max":    int(np.max(score)),
        "mean":   float(np.mean(score)),
        "median": float(np.median(score)),
        "p25":    float(np.percentile(score, 25)),
        "p75":    float(np.percentile(score, 75)),
        "p90":    float(np.percentile(score, 90)),
        "p95":    float(np.percentile(score, 95)),
    }
    print(f"  training integer score range: {dist['min']} .. {dist['max']}", flush=True)
    print(f"  training score mean={dist['mean']:.3f}  median={dist['median']:.0f}", flush=True)
    print(f"  training AUROC (integer score) = {train_auroc:.4f}", flush=True)

    # Also compute logit-predicted AUROC for comparison
    logit_pred = res.predict(sm.add_constant(df[variables].astype(float), has_constant="add"))
    logit_auroc = float(roc_auc_score(y, logit_pred))
    print(f"  training AUROC (continuous logit) = {logit_auroc:.4f}", flush=True)

    # ----------------------------------------------------------------
    # Save refit JSON
    # ----------------------------------------------------------------
    out = {
        "intercept": intercept,
        "coefficients": {v: float(params.loc[v]) for v in variables},
        "points": {v: int(weights_df.loc[weights_df["Variable"] == v, "Points"].values[0])
                   for v in variables},
        "variables_kept": variables,
        "variables_dropped_beta": dropped_beta,
        "variables_dropped_vif": dropped_vif,
        "training_score_distribution": dist,
        "training_auroc_integer": train_auroc,
        "training_auroc_logit": logit_auroc,
        "anchor_min_positive_beta": float(anchor),
        "n_train": int(len(df)),
        "n_train_events": int(y.sum()),
        "beta_cutoff": float(BETA_DROP),
        "n_selected_before_cutoff": int(n_before_cutoff),
        "n_selected_after_cutoff": int(n_after_cutoff),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote {OUT_JSON}", flush=True)

    # ----------------------------------------------------------------
    # Sanity assertions
    # ----------------------------------------------------------------
    assert len(variables) >= MIN_KEPT, f"Only {len(variables)} vars survived"
    assert list(weights_df.columns) == [
        "Variable", "β", "SE", "OR", "95% CI lower", "95% CI upper", "Points"
    ], "weights CSV schema mismatch"
    print("\nStep 5 complete.")


if __name__ == "__main__":
    main()
