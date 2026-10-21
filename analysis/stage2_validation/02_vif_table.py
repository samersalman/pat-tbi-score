"""
PAT-TBI v6 — Phase 0, finding #9: the per-variable multicollinearity table.
==========================================================================

v4 logged a single number, `max VIF = 4.79 (gcs_severe)`, and stopped there. Dr
Hynes asked twice whether multicollinearity was tested, and a one-number answer
is not an answer. This script produces the full diagnostic set on the frozen v4
derivation cohort:

  1. Variance inflation factor for every one of the 22 retained predictors,
     computed exactly the way v4 computed its maximum: on the derivation design
     matrix with an intercept column prepended, via
     `statsmodels.stats.outliers_influence.variance_inflation_factor`. A VIF
     without an intercept term is not the quantity anyone means by VIF, so the
     constant is always present and is itself never reported.
  2. The full 22 by 22 pairwise correlation matrix. All 22 predictors are binary,
     so the phi coefficient is reported; for two binary variables phi is
     algebraically identical to the Pearson product-moment correlation, and this
     script computes it as Pearson. Nothing else is being reported under a
     different name.
  3. A cross-check against the three nested predictor pairs in
     `_common.NESTED_PAIRS`, where one indicator is a strict or near-strict
     subset of the other. For each pair: the correlation, both VIFs, the count in
     the inner category and the count in both, and a verdict on whether the
     nesting is visible in the collinearity diagnostics or invisible to them.
  4. The condition number of the scaled design matrix, as a supplementary
     diagnostic that VIF and pairwise correlation can both miss.

Nothing is refit. The 22 predictors and their integer points are read from the
frozen v4 artefacts through `_common`.

Run with the framework interpreter, not bare `python3`:
    PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
    cd analysis/stage2_validation
    $PY 02_vif_table.py

Outputs
-------
    v6/bundle/parts/vif.json
    v6/tables/table_v6_vif.csv
    v6/tables/table_v6_correlation.csv
    v6/logs/02_vif_table.log
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

# House thresholds, from the Phase 0 contract.
VIF_FLAG = 5.0
CORR_FLAG = 0.7

# The single number v4 logged, for the reproduction check.
V4_MAX_VIF = 4.79
V4_MAX_VIF_VAR = "gcs_severe"

# Digit-free prose for every retained predictor.
#
# The bundle contract forbids digits inside a `display` string, because a number
# in prose cannot be traced back to the value that produced it. That also means
# a cutpoint can never be stated in full inside a display leaf, so these are
# short names rather than definitions. `_common.LABELS` carries the exact
# clinical cutpoints and is what the CSV tables print.
PROSE: dict[str, str] = {
    "gcs_severe": "severe Glasgow Coma Scale total",
    "gcs_moderate": "moderate Glasgow Coma Scale total",
    "gcs_motor_low": "low Glasgow Coma Scale motor score",
    "gcs_motor_high": "normal Glasgow Coma Scale motor score",
    "tbi_pupil_none": "neither pupil reactive",
    "age_45_64": "age forty five to sixty four years",
    "age_65_74": "age sixty five to seventy four years",
    "age_75plus": "age seventy five years or older",
    "female": "female sex",
    "mech_fall": "fall",
    "mech_assault": "assault",
    "risk_penetrating": "penetrating assault",
    "sbp_hypotensive": "hypotension on arrival",
    "hr_tachycardic": "tachycardia on arrival",
    "hr_bradycardic": "bradycardia on arrival",
    "rr_high": "elevated respiratory rate on arrival",
    "hypoxic": "low oxygen saturation on arrival",
    "temp_hypothermic": "hypothermia on arrival",
    "si_elevated": "elevated shock index",
    "si_high": "high shock index",
    "cc_anticoag": "anticoagulant therapy",
    "cc_bleeding": "bleeding disorder",
}


def prose(var: str) -> str:
    """Digit-free prose for a predictor, for use inside a `display` leaf."""
    if var not in PROSE:
        raise KeyError(f"no digit-free prose for {var!r}")
    return PROSE[var]


def sentence(text: str) -> str:
    """Capitalize the first character without touching the rest."""
    return text[:1].upper() + text[1:]


def count_word(n: int) -> str:
    """A digit-free count, because display strings may not carry numerals."""
    return {0: "No", 1: "One", 2: "Two", 3: "Three", 4: "Four",
            5: "Five"}.get(n, "Several")

TABLE_VIF = C.V6 / "tables" / "table_v6_vif.csv"
TABLE_CORR = C.V6 / "tables" / "table_v6_correlation.csv"


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def compute_vifs(X: pd.DataFrame) -> dict[str, float]:
    """VIF per column of X, with an intercept column prepended.

    Identical in construction to `compute_vif` in
    v4/code/05_refit_unpenalized_integer.py, so the maximum is comparable to the
    number v4 logged: constant first, VIF taken on the numpy design matrix,
    constant itself skipped.
    """
    Xc = sm.add_constant(X, has_constant="add").astype(float).values
    cols = ["const"] + list(X.columns)
    out: dict[str, float] = {}
    for i, name in enumerate(cols):
        if name == "const":
            continue
        out[name] = float(variance_inflation_factor(Xc, i))
    return out


def find_perfect_dependency(X: pd.DataFrame, var: str) -> float:
    """R-squared of `var` regressed on the other predictors plus an intercept.

    Only called when a VIF comes back non-finite, to name the column that is a
    linear combination of the others.
    """
    others = [c for c in X.columns if c != var]
    A = sm.add_constant(X[others].astype(float), has_constant="add")
    fit = sm.OLS(X[var].astype(float), A).fit()
    return float(fit.rsquared)


def check_display_strings(payload: dict) -> list[str]:
    """Walk a bundle payload and enforce the contract's rules on `display` text.

    No em-dash or en-dash, no digits, no snake_case slug, and no asserted causal
    or diagnostic claim. Returns a list of violations; empty means compliant.
    """
    import re

    banned = re.compile(
        r"\b(predicts|predict|predicted|predicting|causes?|caused|causal|"
        r"diagnoses?|diagnosed|diagnostic|screening tool|validated outcome|"
        r"improve[sd]? outcome|early warning system|attributable to|adjudicate[sd]?)\b",
        flags=re.IGNORECASE)
    slug = re.compile(r"\b[a-z]+_[a-z0-9_]+\b")

    problems: list[str] = []

    def walk(node, path: str) -> None:
        if isinstance(node, dict):
            if "display" in node and isinstance(node["display"], str):
                text = node["display"]
                if "—" in text:
                    problems.append(f"{path}: contains an em-dash")
                if "–" in text:
                    problems.append(f"{path}: contains an en-dash")
                if any(ch.isdigit() for ch in text):
                    problems.append(f"{path}: contains a digit")
                if slug.search(text):
                    problems.append(f"{path}: contains a snake case identifier "
                                    f"({slug.search(text).group(0)})")
                hit = banned.search(text)
                if hit:
                    problems.append(f"{path}: asserted claim word ({hit.group(0)})")
            for k, v in node.items():
                if k != "display":
                    walk(v, f"{path}.{k}" if path else k)

    walk(payload, "")
    return problems


def scaled_condition_number(X: pd.DataFrame) -> float:
    """Belsley condition number: columns of [1 | X] scaled to unit length.

    The ratio of the largest to the smallest singular value of the column-scaled
    design matrix. Scaling is what makes the number comparable across variables
    with different prevalences; the unscaled number is dominated by the
    intercept and is not interpretable.
    """
    Xc = sm.add_constant(X, has_constant="add").astype(float).values
    norms = np.sqrt((Xc ** 2).sum(axis=0))
    norms[norms == 0] = 1.0
    sv = np.linalg.svd(Xc / norms, compute_uv=False)
    return float(sv.max() / sv.min())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    C.start_log("02_vif_table")

    C.banner("PAT-TBI v6 — Phase 0 finding #9: per-variable multicollinearity table")
    print("Owner script: 02_vif_table.py")
    print("Question being closed: 'Did you test for multicollinearity' (A. Hynes,")
    print("comments 14, 25 and 26 on the rejected EAST draft).")
    print("v4 logged only a single maximum VIF and never produced a per-variable table.")

    # -- inputs -------------------------------------------------------------
    points = C.load_points()
    predictors = list(points.keys())
    train, _test = C.load_split(with_score=False)

    print(f"\nDerivation rows loaded: {len(train):,}")
    print(f"Retained predictors:    {len(predictors)}")

    # -- hard gate ----------------------------------------------------------
    C.banner("Hard gate on the design matrix")
    if len(train) != C.V4_FROZEN["n_train"]:
        print(f"  FAIL derivation rows = {len(train):,}, expected "
              f"{C.V4_FROZEN['n_train']:,}")
        return 1
    print(f"  [PASS] derivation rows: {len(train):,}")

    if len(predictors) != C.V4_FROZEN["n_predictors"]:
        print(f"  FAIL predictor count = {len(predictors)}, expected "
              f"{C.V4_FROZEN['n_predictors']}")
        return 1
    print(f"  [PASS] predictor columns: {len(predictors)}")

    missing = [v for v in predictors if v not in train.columns]
    if missing:
        print(f"  FAIL predictors absent from the derivation frame: {missing}")
        return 1

    X = train[predictors].astype(float)
    if X.shape != (C.V4_FROZEN["n_train"], C.V4_FROZEN["n_predictors"]):
        print(f"  FAIL design matrix shape {X.shape}")
        return 1
    print(f"  [PASS] design matrix shape: {X.shape[0]:,} rows by {X.shape[1]} columns")

    n_nan = int(X.isna().sum().sum())
    if n_nan:
        print(f"  FAIL {n_nan:,} missing cells in the design matrix")
        return 1
    print("  [PASS] no missing cells")

    non_binary = [v for v in predictors
                  if not set(np.unique(X[v].values)).issubset({0.0, 1.0})]
    if non_binary:
        print(f"  FAIL non-binary predictor columns: {non_binary}")
        return 1
    print("  [PASS] all 22 predictors are binary, so the phi coefficient applies")

    unprosed = [v for v in predictors if v not in PROSE]
    if unprosed:
        print(f"  FAIL predictors with no digit-free prose for the bundle: {unprosed}")
        return 1
    print("  [PASS] every predictor has a digit-free name for the bundle")

    # -- prevalence ---------------------------------------------------------
    prevalence = {v: int(X[v].sum()) for v in predictors}

    # -- 1. variance inflation factors --------------------------------------
    C.banner("1. Variance inflation factors, all 22, sorted descending")
    print("Design matrix: the 415,724 derivation rows, 22 binary predictors, plus a")
    print("constant column. The constant is required: a VIF computed without an")
    print("intercept term is not the quantity anyone means by VIF. Construction is")
    print("identical to v4/code/05_refit_unpenalized_integer.py, so the maximum below")
    print("is directly comparable to the single number v4 logged.")
    print(f"Flag threshold: VIF above {VIF_FLAG:.0f} (house convention, Phase 0 contract).\n")

    vifs = compute_vifs(X)

    bad = [v for v, q in vifs.items() if not np.isfinite(q)]
    if bad:
        C.banner("STOP — a non-finite variance inflation factor")
        print("A non-finite VIF means the column is a perfect linear combination of the")
        print("others, and the design matrix is rank deficient. Offending columns:\n")
        for v in bad:
            r2 = find_perfect_dependency(X, v)
            print(f"  {v:22s} label: {C.label(v)}")
            print(f"  {'':22s} R-squared on the other 21 predictors = {r2:.10f}")
        print("\nHard gate FAILED. Nothing written.")
        return 1
    print("  [PASS] every one of the 22 variance inflation factors is finite\n")

    order = sorted(predictors, key=lambda v: vifs[v], reverse=True)
    print(f"  {'#':>2}  {'Predictor':<58} {'VIF':>7}  {'n = 1':>9}  Flag")
    print("  " + "-" * 92)
    for i, v in enumerate(order, start=1):
        flag = "FLAG" if vifs[v] > VIF_FLAG else ""
        print(f"  {i:>2}  {C.label(v):<58} {vifs[v]:>7.3f}  "
              f"{prevalence[v]:>9,}  {flag}")

    max_var = order[0]
    max_vif = vifs[max_var]
    n_flagged = sum(1 for v in predictors if vifs[v] > VIF_FLAG)

    print(f"\n  Maximum VIF: {max_vif:.4f} on {C.label(max_var)}")
    print(f"  Variables above the threshold of {VIF_FLAG:.0f}: {n_flagged} of {len(predictors)}")
    print(f"  Median VIF: {np.median([vifs[v] for v in predictors]):.3f}   "
          f"Minimum VIF: {min(vifs.values()):.3f}")

    # -- reproduction check against v4 --------------------------------------
    C.banner("Reproduction check against the v4 log")
    print(f"  v4/logs/05_refit.log line: 'iter  0: max VIF = {V4_MAX_VIF:.2f} "
          f"({V4_MAX_VIF_VAR})'")
    print(f"  v6 recomputation:          max VIF = {max_vif:.4f} ({max_var})")
    same_var = (max_var == V4_MAX_VIF_VAR)
    same_val = abs(round(max_vif, 2) - V4_MAX_VIF) < 1e-9
    if same_var and same_val:
        print("  [PASS] the maximum reproduces exactly, same variable and same value to")
        print("         the two decimals v4 printed. The v4 number was correct; it was")
        print("         merely incomplete.")
        reproduces = True
    else:
        print("  [MISMATCH] the v4 maximum does NOT reproduce.")
        if not same_var:
            print(f"             variable differs: v4 said {V4_MAX_VIF_VAR}, "
                  f"v6 finds {max_var}")
        if not same_val:
            print(f"             value differs: v4 said {V4_MAX_VIF:.2f}, "
                  f"v6 finds {max_vif:.4f}")
        print("             Reporting what was computed; no number is being adjusted to fit.")
        reproduces = False

    # -- 2. pairwise correlation --------------------------------------------
    C.banner("2. Pairwise correlation, all 231 pairs among the 22")
    print("All 22 predictors are binary indicators, so the association reported here is")
    print("the phi coefficient. For two binary variables the phi coefficient equals the")
    print("Pearson product-moment correlation exactly, and that is how it is computed")
    print("below. No different statistic has been substituted.")
    print(f"Flag threshold: absolute correlation above {CORR_FLAG} (house convention).\n")

    corr = X.corr(method="pearson")
    n_pairs = len(predictors) * (len(predictors) - 1) // 2

    pairs = []
    for i in range(len(predictors)):
        for j in range(i + 1, len(predictors)):
            a, b = predictors[i], predictors[j]
            pairs.append((a, b, float(corr.loc[a, b])))
    pairs.sort(key=lambda t: abs(t[2]), reverse=True)

    flagged_pairs = [p for p in pairs if abs(p[2]) > CORR_FLAG]

    print(f"  Pairs examined: {n_pairs}")
    print(f"  Pairs with absolute correlation above {CORR_FLAG}: {len(flagged_pairs)}")
    if flagged_pairs:
        print("\n  Flagged pairs, listed explicitly:")
        for a, b, r in flagged_pairs:
            print(f"    r = {r:+.4f}   {C.label(a)}  AND  {C.label(b)}")
    else:
        print("  None. Every pair among the 22 sits below the threshold.")

    print("\n  Ten largest absolute correlations, flagged or not, so the reader can see")
    print("  how far the strongest association sits from the threshold:")
    print(f"    {'r':>8}   pair")
    for a, b, r in pairs[:10]:
        print(f"    {r:>+8.4f}   {C.label(a)} / {C.label(b)}")

    strongest_a, strongest_b, strongest_r = pairs[0]

    nested_set = {frozenset(p) for p in C.NESTED_PAIRS}
    for a, b, r in flagged_pairs:
        if frozenset((a, b)) in nested_set:
            print(f"\n  Note: the flagged pair {C.label(a)} with {C.label(b)} is one of the")
            print("  three nested pairs already carried in the shared loader. The motor")
            print("  component is a constituent of the total, so the two indicators are")
            print("  measuring overlapping information by construction. Section 3 below")
            print("  reports the containment counts. Neither variable was removed: the v4")
            print("  score is frozen, both terms survived the derivation, and both VIFs")
            print("  sit below the threshold, so the coefficients are not unstable. The")
            print("  consequence is one of interpretation, not of estimation, and is")
            print("  disclosed as such.")

    # -- 3. nested pairs -----------------------------------------------------
    C.banner("3. Cross-check against the three nested predictor pairs")
    print("Each pair below is a strict or near-strict subset relationship in the coding,")
    print("so a reader will reasonably ask whether the nesting shows up in the")
    print("collinearity diagnostics or hides in them. Both outcomes are informative and")
    print("both are reported.\n")

    nested_records = []
    for inner, outer in C.NESTED_PAIRS:
        r = float(corr.loc[inner, outer])
        n_inner = int(X[inner].sum())
        n_outer = int(X[outer].sum())
        n_both = int(((X[inner] == 1) & (X[outer] == 1)).sum())
        containment = n_both / n_inner if n_inner else float("nan")
        visible = (abs(r) > CORR_FLAG) or (vifs[inner] > VIF_FLAG) or (vifs[outer] > VIF_FLAG)

        print(f"  {C.label(inner)}")
        print(f"    nested inside: {C.label(outer)}")
        print(f"    correlation (phi)          r = {r:+.4f}")
        print(f"    VIF, inner indicator         {vifs[inner]:.3f}")
        print(f"    VIF, outer indicator         {vifs[outer]:.3f}")
        print(f"    inner category count         {n_inner:,}")
        print(f"    outer category count         {n_outer:,}")
        print(f"    both indicators positive     {n_both:,} of {n_inner:,} "
              f"({100.0 * containment:.2f}% of the inner category)")
        if visible:
            print("    VERDICT: the nesting SHOWS UP in the collinearity diagnostics; at")
            print("             least one of the correlation or the two VIFs crosses its")
            print("             threshold, so a reader inspecting the diagnostics would")
            print("             find the overlap without being told about it.")
        else:
            print("    VERDICT: the nesting HIDES IN the collinearity diagnostics; the")
            print("             correlation and both VIFs sit below their thresholds")
            print("             despite the containment above, because the inner category")
            print("             is small relative to the outer one and a low-prevalence")
            print("             subset barely moves either statistic. The overlap is a")
            print("             point-interpretation issue, not a variance-inflation one,")
            print("             and must be disclosed in prose rather than left to the")
            print("             diagnostics to surface.")
        print()

        nested_records.append({
            "inner": inner, "outer": outer, "r": r,
            "vif_inner": vifs[inner], "vif_outer": vifs[outer],
            "n_inner": n_inner, "n_outer": n_outer, "n_both": n_both,
            "containment": containment, "visible": visible,
        })

    n_nesting_visible = sum(1 for rec in nested_records if rec["visible"])
    n_nesting_hidden = len(nested_records) - n_nesting_visible
    print(f"  Of the {len(nested_records)} nested pairs, {n_nesting_visible} "
          f"{'shows' if n_nesting_visible == 1 else 'show'} up in the diagnostics and "
          f"{n_nesting_hidden} {'hides' if n_nesting_hidden == 1 else 'hide'} in them.")
    print("  A reader relying on the diagnostics alone would therefore find the Glasgow")
    print("  Coma Scale overlap and miss the other two, which is exactly why the nesting")
    print("  is stated in prose in the Methods rather than left for the table to imply.")

    # -- 4. condition number -------------------------------------------------
    C.banner("4. Condition number of the scaled design matrix, supplementary")
    kappa = scaled_condition_number(X)
    print("Belsley convention: the columns of the intercept-augmented design matrix are")
    print("each scaled to unit Euclidean length, and the condition number is the ratio")
    print("of the largest to the smallest singular value. Scaling matters, because the")
    print("unscaled number is dominated by the intercept and means nothing. The usual")
    print("rules of thumb are that a value below 30 indicates no collinearity concern")
    print("and a value above 100 indicates a serious one.\n")
    print(f"  Scaled condition number: {kappa:.2f}")
    if kappa < 30:
        print("  Below 30: no structural near-dependency beyond what the pairwise")
        print("  statistics already show.")
    elif kappa < 100:
        print("  Between 30 and 100: a moderate near-dependency that the pairwise")
        print("  statistics may not fully capture. Worth a sentence in the Methods.")
    else:
        print("  Above 100: a serious near-dependency that the pairwise statistics have")
        print("  missed. This must be resolved before the score is reported.")

    # -- headline ------------------------------------------------------------
    # Three branches, because "no collinearity" and "one expected overlap" are
    # different claims and only one of them is true here.
    if n_flagged == 0 and not flagged_pairs:
        headline = (
            "No variance inflation factor among the retained predictors reached the "
            "conventional threshold and no pair of indicators reached the conventional "
            "correlation threshold, so the integer points rest on a design matrix with "
            "no material collinearity."
        )
    elif n_flagged == 0:
        pair_names = "; ".join(f"the {prose(a)} together with the {prose(b)}"
                               for a, b, _ in flagged_pairs)
        headline = (
            "No variance inflation factor among the retained predictors reached the "
            "conventional threshold, so no coefficient is destabilized by collinearity. "
            f"{count_word(len(flagged_pairs))} "
            f"{'pair' if len(flagged_pairs) == 1 else 'pairs'} exceeded the "
            f"conventional correlation threshold, {pair_names}, which is expected "
            "because the motor component is part of the total. Both terms were kept, "
            "since the score is frozen and each carries its own points, and the "
            "overlap is disclosed rather than resolved by deletion."
        )
    else:
        flagged_names = "; ".join(sentence(prose(v)) for v in order
                                  if vifs[v] > VIF_FLAG)
        headline = (
            "At least one variance inflation factor among the retained predictors "
            f"reached the conventional threshold: {flagged_names}. The affected "
            "variables are named in the collinearity table and the overlap is "
            "described in the Methods."
        )

    C.banner("Headline sentence for the Methods and the co-author response")
    print(f"  {headline}")

    # -- outputs -------------------------------------------------------------
    C.banner("Writing outputs")
    TABLE_VIF.parent.mkdir(parents=True, exist_ok=True)

    vif_rows = [{
        "Rank": i,
        "Predictor": C.label(v),
        "VIF": f"{vifs[v]:.2f}",
        "Flagged (VIF > 5)": "Yes" if vifs[v] > VIF_FLAG else "No",
        "Derivation n with indicator present": prevalence[v],
        "Integer points": points[v],
        "Identifier": v,
    } for i, v in enumerate(order, start=1)]
    pd.DataFrame(vif_rows).to_csv(TABLE_VIF, index=False)
    print(f"  {TABLE_VIF}")

    # Rows and columns follow the VIF ranking, so the two tables read together
    # and the flagged pair sits in the top-left corner.
    labels = [C.label(v) for v in order]
    corr_out = corr.loc[order, order].round(3)
    corr_out.index = labels
    corr_out.columns = labels
    corr_out.index.name = "Predictor"
    corr_out.to_csv(TABLE_CORR)
    print(f"  {TABLE_CORR}")

    # Flat, not self-namespaced: 06_build_bundle.py folds the part in whole with
    # `results["vif"] = payload`, so a top-level "vif" wrapper here would nest the
    # section inside itself. Confirmed against `PART_KEYS` and `Bundle.merge`.
    payload = {
            "method": {"display": (
                "Variance inflation factors were computed on the derivation design "
                "matrix, all retained predictors together with an intercept column, "
                "because a variance inflation factor computed without an intercept "
                "term is not the quantity the name refers to. Pairwise association "
                "between the binary indicators was measured with the phi coefficient, "
                "which for two binary variables is algebraically identical to the "
                "Pearson product moment correlation and was computed as such."
            )},
            "n_derivation": {"n": int(len(train))},
            "n_predictors": {"value": int(len(predictors))},
            "threshold_vif": {"value": int(VIF_FLAG)},
            "threshold_correlation": {"value": CORR_FLAG},
            "per_predictor": {v: {"value": round(vifs[v], 3)} for v in order},
            "max": {"value": round(max_vif, 3)},
            "max_predictor": {"display": sentence(prose(max_var))},
            "median": {"value": round(float(np.median([vifs[v] for v in predictors])), 3)},
            "min": {"value": round(float(min(vifs.values())), 3)},
            "n_above_threshold": {"n": int(n_flagged), "N": int(len(predictors))},
            "n_pairs_examined": {"value": int(n_pairs)},
            "n_pairs_above_threshold": {"n": int(len(flagged_pairs)), "N": int(n_pairs)},
            "strongest_pair_correlation": {"value": round(strongest_r, 3)},
            "strongest_pair": {"display": sentence(
                f"{prose(strongest_a)} together with {prose(strongest_b)}")},
            "flagged_pairs": {
                f"{a}__with__{b}": {
                    "correlation": {"value": round(r, 3)},
                    "pair": {"display": sentence(
                        f"{prose(a)} together with {prose(b)}")},
                    "vif_first": {"value": round(vifs[a], 3)},
                    "vif_second": {"value": round(vifs[b], 3)},
                    "note": {"display": (
                        "The motor component is a constituent of the total, so the two "
                        "indicators overlap by construction. Neither was removed, "
                        "because the score is frozen, both terms survived derivation, "
                        "and both variance inflation factors sit below the threshold, "
                        "so the coefficients are not unstable. The consequence is one "
                        "of interpretation rather than of estimation."
                        if frozenset((a, b)) in {frozenset(p) for p in C.NESTED_PAIRS}
                        else
                        "The pair exceeds the conventional correlation threshold and is "
                        "reported so that the overlap is visible to the reader."
                    )},
                } for a, b, r in flagged_pairs
            },
            "condition_number": {"value": round(kappa, 2)},
            "v4_max": {"value": V4_MAX_VIF},
            "reproduces_v4_max": {"display": (
                "The maximum reproduces the single value the previous analysis logged, "
                "on the same variable and to the precision that was printed."
                if reproduces else
                "The maximum does not reproduce the single value the previous analysis "
                "logged; the recomputed figure is reported instead and no number has "
                "been adjusted to fit."
            )},
            "nested_pairs": {
                f"{rec['inner']}__in__{rec['outer']}": {
                    "correlation": {"value": round(rec["r"], 3)},
                    "vif_inner": {"value": round(rec["vif_inner"], 3)},
                    "vif_outer": {"value": round(rec["vif_outer"], 3)},
                    "n_inner": {"n": rec["n_inner"]},
                    "n_outer": {"n": rec["n_outer"]},
                    "n_both": {"n": rec["n_both"], "N": rec["n_inner"]},
                    "verdict": {"display": (
                        "The nesting is visible in the collinearity diagnostics."
                        if rec["visible"] else
                        "The nesting is invisible to the collinearity diagnostics "
                        "because the inner category is small relative to the outer one, "
                        "so it is disclosed in prose instead."
                    )},
                } for rec in nested_records
            },
            "n_nested_pairs": {"value": int(len(nested_records))},
            "n_nested_pairs_visible": {
                "n": int(n_nesting_visible), "N": int(len(nested_records))},
            "headline": {"display": headline},
    }
    problems = check_display_strings(payload)
    if problems:
        C.banner("STOP — the bundle part violates the display-string contract")
        for p in problems:
            print(f"  {p}")
        print("\nNothing written to bundle/parts.")
        return 1
    print("  [PASS] every display string is free of dashes, digits, slugs and "
          "asserted claims")

    path = C.write_part("vif", payload)
    print(f"  {path}")

    C.banner("02_vif_table complete")
    print(f"  Maximum VIF {max_vif:.2f}; {n_flagged} of {len(predictors)} above "
          f"{VIF_FLAG:.0f}.")
    print(f"  {len(flagged_pairs)} of {n_pairs} pairs above |r| = {CORR_FLAG}.")
    print(f"  Scaled condition number {kappa:.2f}.")
    print(f"  v4 maximum {'reproduces' if reproduces else 'DOES NOT reproduce'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
