"""
PAT-TBI v4 - Step 0: Extractability audit of v3 candidate variables
================================================================================

Documents which v3 LASSO candidates are kept vs dropped for the v4 parsimonious
refit, with explicit rationale for every drop. The v4 candidate set is restricted
to bedside-extractable predictors only (i.e., information that is reliably known
to the trauma team in the platelet-decision moment): demographics, ED vital signs,
GCS, pupillary response, mechanism / transfer status, and the two coagulation-
relevant comorbidities (anticoagulant use, known bleeding diathesis) that are
typically conveyed by EMS handoff or MedicAlert.

Excluded categories (with rationale, see exclusion_rationale.md):
  - Hospital-system variables (teaching status, verification level, highest
    activation)  -- not patient-level; not bedside-extractable; activation level
    is itself a downstream triage decision and would be circular for a triage
    tool.
  - Insurance status (uninsured)  -- not bedside-extractable; v3 |beta| ~ 0.005.
  - History-only comorbidities (CHF, CVA, HTN, DM, COPD, smoker, CKD, cirrhosis,
    cancer, dementia, alcohol)  -- require chart or family interview, not
    directly tied to platelet/coagulation biology in the acute moment.

Source for v3 candidate list:
  PAT_TBI/v3/code/02_engineer_features.py  (LASSO_X_COLS)

Inputs
------
  PAT_TBI/v3/code/02_engineer_features.py    (read-only, source of truth)

Outputs
-------
  PAT_TBI/v4/extractability_audit/candidate_audit_table.csv
  PAT_TBI/v4/extractability_audit/exclusion_rationale.md

Seed: 42
"""
import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
import os as _os, sys as _sys  # path handling: make analysis/config.py importable
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import config  # noqa: E402  (analysis/config.py: PAT_TBI_WORKDIR, PAT_TBI_NTDB_DUCKDB)
PAT_V3 = config.V3_DIR
PAT_V4 = config.V4_DIR
AUDIT_DIR = PAT_V4 / "extractability_audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

V3_FEATURES_SCRIPT = PAT_V3 / "code" / "02_engineer_features.py"
OUT_CSV = AUDIT_DIR / "candidate_audit_table.csv"
OUT_MD = AUDIT_DIR / "exclusion_rationale.md"


def banner(msg):
    line = "=" * 78
    print(f"\n{line}\n{msg}\n{line}", flush=True)


# --------------------------------------------------------------------------
# Single source of truth: kept and dropped variables with rationale
# --------------------------------------------------------------------------
# Each entry: (variable, category, kept_in_v4, rationale)
# Rationale strings are reused verbatim in the markdown artifact.

_HISTORY_ONLY_RATIONALE = (
    "History-only comorbidity; not directly tied to platelet/coagulation "
    "biology in the acute moment; requires chart or family interview "
    "and is not reliably known at the bedside."
)

DROPPED = [
    ("teaching_status_binary",     "Hospital-system",
     "Not a patient-level predictor; unavailable at bedside."),
    ("verification_level_1",       "Hospital-system",
     "Not a patient-level predictor; unavailable at bedside."),
    ("highest_activation_highest", "Hospital-system",
     "Downstream triage decision -- circular for a triage tool."),
    ("ins_uninsured",              "Insurance",
     "Not bedside-extractable; v3 beta was ~0.005 (already negligible)."),
    ("cc_chf",                     "History comorbidity", _HISTORY_ONLY_RATIONALE),
    ("cc_cva",                     "History comorbidity", _HISTORY_ONLY_RATIONALE),
    ("cc_htn",                     "History comorbidity", _HISTORY_ONLY_RATIONALE),
    ("cc_dm",                      "History comorbidity", _HISTORY_ONLY_RATIONALE),
    ("cc_copd",                    "History comorbidity", _HISTORY_ONLY_RATIONALE),
    ("cc_smoker",                  "History comorbidity", _HISTORY_ONLY_RATIONALE),
    ("cc_ckd",                     "History comorbidity", _HISTORY_ONLY_RATIONALE),
    ("cc_cirrhosis",               "History comorbidity", _HISTORY_ONLY_RATIONALE),
    ("cc_cancer",                  "History comorbidity", _HISTORY_ONLY_RATIONALE),
    ("cc_dementia",                "History comorbidity", _HISTORY_ONLY_RATIONALE),
    ("cc_alcohol",                 "History comorbidity", _HISTORY_ONLY_RATIONALE),
]

_BEDSIDE_RATIONALE = (
    "Bedside-extractable at the platelet-decision moment "
    "(demographic / measured ED vital / GCS / pupil / mechanism / transfer "
    "status / EMS-conveyed coagulation-relevant comorbidity)."
)

KEPT = [
    # Demographics (4)
    ("age_45_64",        "Demographics", _BEDSIDE_RATIONALE),
    ("age_65_74",        "Demographics", _BEDSIDE_RATIONALE),
    ("age_75plus",       "Demographics", _BEDSIDE_RATIONALE),
    ("female",           "Demographics", _BEDSIDE_RATIONALE),
    # ED vitals (11)
    ("sbp_hypotensive",  "ED vitals",    _BEDSIDE_RATIONALE),
    ("sbp_hypertensive", "ED vitals",    _BEDSIDE_RATIONALE),
    ("hr_bradycardic",   "ED vitals",    _BEDSIDE_RATIONALE),
    ("hr_tachycardic",   "ED vitals",    _BEDSIDE_RATIONALE),
    ("rr_low",           "ED vitals",    _BEDSIDE_RATIONALE),
    ("rr_high",          "ED vitals",    _BEDSIDE_RATIONALE),
    ("hypoxic",          "ED vitals",    _BEDSIDE_RATIONALE),
    ("shock_index",      "ED vitals",    _BEDSIDE_RATIONALE),
    ("si_elevated",      "ED vitals",    _BEDSIDE_RATIONALE),
    ("si_high",          "ED vitals",    _BEDSIDE_RATIONALE),
    ("temp_hypothermic", "ED vitals",    _BEDSIDE_RATIONALE),
    # GCS (4)
    ("gcs_severe",       "GCS",          _BEDSIDE_RATIONALE),
    ("gcs_moderate",     "GCS",          _BEDSIDE_RATIONALE),
    ("gcs_motor_low",    "GCS",          _BEDSIDE_RATIONALE),
    ("gcs_motor_high",   "GCS",          _BEDSIDE_RATIONALE),
    # Pupils (1)
    ("tbi_pupil_none",   "Pupils",       _BEDSIDE_RATIONALE),
    # Mechanism (4)
    ("mech_mvc",         "Mechanism",    _BEDSIDE_RATIONALE),
    ("mech_assault",     "Mechanism",    _BEDSIDE_RATIONALE),
    ("mech_fall",        "Mechanism",    _BEDSIDE_RATIONALE),
    ("risk_penetrating", "Mechanism",    _BEDSIDE_RATIONALE),
    # Transfer (1)
    ("is_transfer",      "Transfer",     _BEDSIDE_RATIONALE),
    # Coagulation-relevant comorbidities (2)
    ("cc_anticoag",      "Coagulation comorbidity",
     "Anticoagulant use directly affects the platelet-transfusion decision; "
     "typically known from EMS handoff, patient report, or MedicAlert."),
    ("cc_bleeding",      "Coagulation comorbidity",
     "Known bleeding diathesis directly affects the platelet-transfusion "
     "decision; typically known from EMS handoff, patient report, or MedicAlert."),
]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def parse_v3_candidate_list(script_path: Path) -> list:
    """Statically parse PAT_TBI/v3/code/02_engineer_features.py and extract
    LASSO_X_COLS as a flat list of strings.

    Handles the *CC_COLS splat so the list is fully resolved (no symbolic
    references left). Returns variable names in declaration order.
    """
    src = script_path.read_text()
    tree = ast.parse(src)

    # Pass 1: gather top-level list assignments to a name table.
    name_to_list = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if isinstance(node.value, ast.List):
                vals = []
                for elt in node.value.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        vals.append(elt.value)
                    elif isinstance(elt, ast.Starred):
                        # cannot resolve here; defer
                        continue
                name_to_list[name] = (node.value, vals)

    if "LASSO_X_COLS" not in name_to_list:
        raise RuntimeError("LASSO_X_COLS not found in v3 02_engineer_features.py")

    list_node, _ = name_to_list["LASSO_X_COLS"]
    resolved = []
    for elt in list_node.elts:
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
            resolved.append(elt.value)
        elif isinstance(elt, ast.Starred):
            inner = elt.value
            if isinstance(inner, ast.Name) and inner.id in name_to_list:
                resolved.extend(name_to_list[inner.id][1])
            else:
                raise RuntimeError(
                    f"Unresolvable splat in LASSO_X_COLS: {ast.dump(elt)}"
                )
        else:
            raise RuntimeError(
                f"Unsupported element in LASSO_X_COLS: {ast.dump(elt)}"
            )
    return resolved


def main():
    parser = argparse.ArgumentParser(
        description="Audit which v3 LASSO candidates are kept vs dropped in v4."
    )
    # --sample is accepted for pipeline-style uniformity; this script does
    # not consume any data and so the flag is inert.
    parser.add_argument("--sample", type=int, default=0,
                        help="Inert; accepted for pipeline uniformity.")
    args = parser.parse_args()

    banner("PAT-TBI v4 Step 0 - Extractability audit")

    # ----------------------------------------------------------------------
    # Load v3 authoritative candidate list
    # ----------------------------------------------------------------------
    print(f"Reading v3 candidate list from {V3_FEATURES_SCRIPT}", flush=True)
    v3_candidates = parse_v3_candidate_list(V3_FEATURES_SCRIPT)
    v3_set = set(v3_candidates)
    print(f"  v3 LASSO_X_COLS contains {len(v3_candidates)} variables.", flush=True)

    # ----------------------------------------------------------------------
    # Build audit rows
    # ----------------------------------------------------------------------
    kept_set = {row[0] for row in KEPT}
    dropped_set = {row[0] for row in DROPPED}

    overlap = kept_set & dropped_set
    assert not overlap, f"Variable in BOTH kept and dropped: {overlap}"

    audit_rows = []
    for var, cat, rat in KEPT:
        audit_rows.append({
            "variable": var,
            "category": cat,
            "kept_in_v4": "Yes",
            "rationale": rat,
        })
    for var, cat, rat in DROPPED:
        audit_rows.append({
            "variable": var,
            "category": cat,
            "kept_in_v4": "No",
            "rationale": rat,
        })

    audit_df = pd.DataFrame(audit_rows, columns=["variable", "category", "kept_in_v4", "rationale"])

    # ----------------------------------------------------------------------
    # Exhaustiveness check against v3 source
    # ----------------------------------------------------------------------
    audit_set = kept_set | dropped_set

    in_v3_not_in_audit = sorted(v3_set - audit_set)
    in_audit_not_in_v3 = sorted(audit_set - v3_set)

    if in_v3_not_in_audit:
        print("\nERROR: v3 candidates not accounted for in either kept or dropped list:",
              flush=True)
        for v in in_v3_not_in_audit:
            print(f"  - {v}", flush=True)
    if in_audit_not_in_v3:
        print("\nERROR: variables in kept/dropped list that do NOT exist in v3 LASSO_X_COLS:",
              flush=True)
        for v in in_audit_not_in_v3:
            print(f"  - {v}", flush=True)

    assert not in_v3_not_in_audit, (
        f"Audit not exhaustive vs v3: missing {in_v3_not_in_audit}"
    )
    assert not in_audit_not_in_v3, (
        f"Audit lists contain non-v3 variables: {in_audit_not_in_v3}"
    )

    # ----------------------------------------------------------------------
    # Write CSV
    # ----------------------------------------------------------------------
    audit_df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}", flush=True)
    print(f"  rows: {len(audit_df)}", flush=True)

    # ----------------------------------------------------------------------
    # Write markdown
    # ----------------------------------------------------------------------
    write_markdown(OUT_MD, kept=KEPT, dropped=DROPPED, n_v3=len(v3_candidates))
    print(f"Wrote {OUT_MD}", flush=True)

    # ----------------------------------------------------------------------
    # Stdout summary
    # ----------------------------------------------------------------------
    n_kept = len(KEPT)
    n_dropped = len(DROPPED)
    n_total = n_kept + n_dropped

    banner("Summary")
    print(f"  v3 LASSO_X_COLS total      : {len(v3_candidates)}", flush=True)
    print(f"  Kept in v4                 : {n_kept}", flush=True)
    print(f"  Dropped from v4            : {n_dropped}", flush=True)
    print(f"  Kept + Dropped             : {n_total}", flush=True)
    print(f"  Expected (v3 total)        : {len(v3_candidates)}", flush=True)

    if n_total != len(v3_candidates):
        print(f"\n  WARNING: kept + dropped ({n_total}) != v3 total "
              f"({len(v3_candidates)})", flush=True)
        sys.exit(1)

    print("\nPAT-TBI v4 Step 0 complete.", flush=True)


def write_markdown(path: Path, kept, dropped, n_v3: int):
    # Group dropped by category, preserving declaration order
    drop_by_cat = {}
    drop_cat_order = []
    for var, cat, rat in dropped:
        if cat not in drop_by_cat:
            drop_by_cat[cat] = []
            drop_cat_order.append(cat)
        drop_by_cat[cat].append((var, rat))

    keep_by_cat = {}
    keep_cat_order = []
    for var, cat, rat in kept:
        if cat not in keep_by_cat:
            keep_by_cat[cat] = []
            keep_cat_order.append(cat)
        keep_by_cat[cat].append((var, rat))

    lines = []
    lines.append("# PAT-TBI v4 -- Extractability Audit\n")
    lines.append("## Overview\n")
    lines.append(
        "The PAT-TBI v3 LASSO candidate set contained "
        f"{n_v3} variables, including hospital-system attributes "
        "(teaching status, verification level, highest activation), insurance "
        "status, and history-only comorbidities (CHF, CVA, HTN, DM, COPD, smoker, "
        "CKD, cirrhosis, cancer, dementia, alcohol). None of these are reliably "
        "available *at the bedside* in the platelet-transfusion-decision moment: "
        "hospital-system variables are not patient-level predictors and the "
        "highest-activation flag is itself a downstream triage decision (using "
        "it would be circular for a triage tool); insurance status is not "
        "bedside-extractable and contributed a negligible v3 coefficient; "
        "history-only comorbidities require chart review or family interview "
        "and are not directly tied to acute platelet/coagulation biology. "
        "v4 therefore enforces a strictly bedside-extractable candidate set "
        "before LASSO so that every variable a clinician sees on the final "
        "score is one they can plausibly know in the first minutes of resuscitation."
        "\n"
    )

    lines.append(f"## Drop list ({len(dropped)} variables)\n")
    for cat in drop_cat_order:
        lines.append(f"### {cat}\n")
        for var, rat in drop_by_cat[cat]:
            lines.append(f"- `{var}` -- {rat}")
        lines.append("")

    lines.append(f"## Keep list ({len(kept)} variables)\n")
    for cat in keep_cat_order:
        lines.append(f"### {cat}")
        var_list = ", ".join(f"`{v}`" for v, _ in keep_by_cat[cat])
        lines.append(f"- {var_list}\n")

    lines.append("## Downstream implications\n")
    lines.append(
        "This bedside-extractable candidate matrix is the input to v4 LASSO "
        "stability selection, which will further prune to roughly 10-15 final "
        "score variables via lambda.1se (the sparsest model within 1 SE of the "
        "best cross-validated deviance), stability selection at frequency >= 0.85 "
        "across bootstrap resamples, and a |beta| >= 0.15 floor on the unpenalized "
        "refit. The combined effect is a parsimonious integer score whose every "
        "term is both statistically robust and clinically extractable in the "
        "platelet-decision moment."
        "\n"
    )

    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
