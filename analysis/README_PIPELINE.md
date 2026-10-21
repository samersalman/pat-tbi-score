# Analysis pipeline: run order and file dependencies

This folder holds the 23 scripts that produced the analysis, plus `config.py`. The scripts keep their original file names and numbering. They were written in two stages:

- `stage1_derivation/` builds the cohort, fits the score and runs the first round of validation and sensitivity analyses.
- `stage2_validation/` reads the frozen outputs of stage 1 and recomputes validation results under one calibration convention, with corrected intervals. It does not change the predictors, coefficients or integer points of the frozen score. The models it fits are auxiliary and none of them replaces the score: the score-to-probability map (steps 00 and 09), the head-injury-severity sensitivity models (steps 03 and 11) and the regressions behind the variance inflation factors (step 02).

Stage 2 reads patient-level frozen outputs of stage 1 (the cohort, candidate-feature and scored-validation Parquet files and the lists of patient keys). Those files are produced by running stage 1 on your own copy of the NTDB data. They are never shipped, and the stage 2 scripts cannot be run without them.

## Environment

- Python 3.12 with the packages in `requirements.txt` (repository root).
- `PAT_TBI_NTDB_DUCKDB` (path to the NTDB DuckDB database) and `PAT_TBI_WORKDIR` (a writable folder outside the repository) are read by `config.py`. `PAT_TBI_NTDB_PUF_DIR` is optional. See [`../docs/REPRODUCE.md`](../docs/REPRODUCE.md).
- Every script reaches `config.py`, so each can be started from any working directory: `python analysis/stage1_derivation/03_split_train_test.py`. The 13 stage 1 scripts, `stage2_validation/_common.py` and `stage2_validation/05_mortality_or_drop.py` import it directly; the other eight stage 2 scripts (00, 01, 02, 03, 04, 06a, 09 and 11) reach it only through `_common.py`.
- The base random seed is 42. Step 04 derives the seed of each bootstrap resample from it (42 plus the resample index, with a further offset if a resample has to be redrawn); the other scripts that draw random numbers use 42 directly.

## Where the files live

Every path in the table is relative to `PAT_TBI_WORKDIR`. Running the steps in the order given below creates the folders they write to. The folder names `v4` and `v6` are the names the two stages carried when the code was run; stage 1 uses `v4/` and stage 2 uses `v6/`. Abbreviations used in the table:

| Abbreviation | Folder |
|---|---|
| `D4` | `v4/derived/` |
| `T4` | `v4/tables/` |
| `F4` | `v4/figures/` |
| `T6` | `v6/tables/` |
| `P6` | `v6/bundle/parts/` (JSON parts) |
| `L6` | `v6/logs/` (transcript and hand-off note per script) |
| `F6` | `v6/figures/` |

Files marked **(PL)** are patient-level. They are created in the work folder and are never committed or shipped. The `NTDB DB` is the DuckDB database named by `PAT_TBI_NTDB_DUCKDB`; `PUF_TRAUMA.csv` means the yearly raw files described in [`../docs/DATA_AVAILABILITY.md`](../docs/DATA_AVAILABILITY.md).

## Run order

Stage 1 (`analysis/stage1_derivation/`):

```
01_build_cohort_duckdb.py
02_engineer_features.py
03_split_train_test.py
04_lasso_stability_selection.py      (hours)
05_refit_unpenalized_integer.py
06_score_test_set.py
07_validation_metrics.py
08_calibration_dca_operating.py
09_subgroups.py
10_severity_score_benchmarks.py
11_sensitivity_all_trauma.py         (needs the NTDB DB; can run any time after 05)
12_sensitivity_whole_blood.py
00_audit_extractability.py           (optional; see the notes below)
```

Stage 2 (`analysis/stage2_validation/`), after all of stage 1:

```
_common.py --self-test               (optional check of the frozen stage 1 outputs)
00_recalibrate.py
01_risk_bands.py
02_vif_table.py
03_head_ais_sensitivity.py
04_missingness.py                    (must precede 06a; cannot be run from this repository alone: requires table_consort.csv from a script that is not shipped)
05_mortality_or_drop.py              (needs the NTDB DB)
06a_table1_stats.py                  (cannot be run from this repository alone: needs the step 04 output, which requires table_consort.csv from a script that is not shipped)
09_decision_curve.py                 (needs 00)
11_recompute_intervals.py            (needs 03; run last)
```

## Stage 1 scripts

| Script | Purpose | Inputs | Outputs |
|---|---|---|---|
| `00_audit_extractability.py` | Optional audit that lists which earlier candidate predictors are kept or dropped, with a written reason for each. Nothing downstream reads its output. | `v3/code/02_engineer_features.py` (an earlier-stage script that is not part of this repository) | `v4/extractability_audit/candidate_audit_table.csv`, `v4/extractability_audit/exclusion_rationale.md` |
| `01_build_cohort_duckdb.py` | Builds the adult (18 or older) traumatic brain injury cohort (any S06 ICD-10-CM diagnosis) for admission years 2020 to 2024 and logs a six-step exclusion ledger. The ledger drops prehospital cardiac arrest, then ED_DISPOSITION_CODE MOR, then ED_DISPOSITION_CODE 5A_S and 5BMS, then missing ED_DISPOSITION_CODE. Adds injury-severity maxima, pre-existing-condition flags and four fields from the yearly files. | NTDB DB (`v_patients_2019_2024`, `raw_icd_diagnosis`, `raw_preexisting`, `raw_ais_diagnosis`, `lookup_ais`); `PUF_TRAUMA.csv` for 2020 to 2024 | `D4/cohort_raw.parquet` **(PL)**, `D4/cohort_exclusions.json` |
| `02_engineer_features.py` | Engineers the bedside candidate predictors (age bands, sex, vital-sign cut points, Glasgow Coma Scale bands, pupils, mechanism, arrival route, anticoagulant use, bleeding disorder) and the two outcome columns (platelets within 4 hours; platelets or whole blood within 4 hours). Runs a leakage check. | `D4/cohort_raw.parquet` | `D4/candidates.parquet` **(PL)**, `T4/table_candidate_coverage.csv` |
| `03_split_train_test.py` | Stratified split into a two-thirds derivation sample and a one-third validation sample (seed 42, stratified on the primary outcome). | `D4/candidates.parquet` | `D4/train_inc_keys.csv` **(PL)**, `D4/test_inc_keys.csv` **(PL)**, `T4/table_split_balance.csv` |
| `04_lasso_stability_selection.py` | LASSO stability selection on the derivation sample: 100 bootstrap resamples, L1-penalized logistic regression at the one-standard-error penalty, keeps predictors selected in at least 85% of resamples. **Takes hours.** | `D4/candidates.parquet`, `D4/train_inc_keys.csv` | `D4/lasso_cv_fit.pkl` **(PL)**, `T4/table_lasso_stability.csv`, `T4/table_lasso_selected.csv` |
| `05_refit_unpenalized_integer.py` | Unpenalized logistic refit of the stable predictors; drops predictors below an absolute coefficient floor or above a variance-inflation limit; converts coefficients to integer points (coefficient divided by the smallest positive coefficient, rounded, with a floor of one point in magnitude). | `D4/candidates.parquet`, `D4/train_inc_keys.csv`, `T4/table_lasso_selected.csv` | `T4/table_pat_weights.csv`, `D4/pat_refit.json` |
| `06_score_test_set.py` | Scores the validation sample with the PAT-TBI points and four comparator scores (Glasgow Coma Scale, GCS-Pupils, IMPACT-Core, CRASH-Basic), each with its own availability mask. | `D4/candidates.parquet`, `D4/test_inc_keys.csv`, `T4/table_pat_weights.csv`, `D4/pat_refit.json` | `D4/cohort_test_scored.parquet` **(PL)** |
| `07_validation_metrics.py` | Discrimination (AUROC with a subsample-bootstrap interval), pairwise DeLong tests, a calibration summary and ROC data for all five scores. **SUPERSEDED** for the reported intervals by stage 2 step 11 (the subsample-bootstrap interval is replaced by DeLong intervals) and for the calibration statistics by stage 2 step 00. Its helper functions are still imported by stage 2 steps 03 and 11. | `D4/cohort_test_scored.parquet` | `T4/table_pat_discrimination.csv`, `T4/table_pat_delong_matrix.csv`, `T4/table_pat_calibration.csv`, `D4/pat_roc_data.pkl` **(PL)** |
| `08_calibration_dca_operating.py` | Calibration plot, decision curve, operating characteristics at three cutoff rules, ROC overlay and score-distribution plot. **SUPERSEDED** for the reported decision curves by stage 2 step 09; calibration is recomputed in stage 2 step 00. | `D4/cohort_test_scored.parquet`, `D4/pat_roc_data.pkl` (optional) | `T4/table_pat_dca.csv`, `T4/table_pat_operating_characteristics.csv`, `F4/figure_pat_dca`, `F4/figure_pat_calibration`, `F4/figure_pat_roc`, `F4/figure_pat_score_distribution` (each `.png` and `.pdf`) |
| `09_subgroups.py` | Subgroup AUROCs (age, sex, Glasgow Coma Scale band, isolated head injury versus polytrauma, mechanism, arrival route, teaching status) with DeLong contrasts against the comparators, and a forest plot. Its helper functions are imported by stage 2 step 03. | `D4/cohort_test_scored.parquet`, `D4/candidates.parquet` | `T4/table_subgroups.csv`, `F4/figure_subgroups_forest` (`.pdf`, `.png`) |
| `10_severity_score_benchmarks.py` | Benchmarks PAT-TBI against the four comparators as severity proxies (they were derived for mortality, not platelet receipt): AUROC, DeLong tests and decision-curve net benefit. | `D4/cohort_test_scored.parquet` | `T4/table_pat_severity_benchmarks.csv`, `T4/table_pat_severity_dca.csv`, `F4/figure_pat_severity_benchmark` (`.pdf`, `.png`) |
| `11_sensitivity_all_trauma.py` | Applies the frozen integer weights to all adult NTDB patients for 2020 to 2024 without the traumatic brain injury restriction. Uses the same ED_DISPOSITION_CODE exclusions as step 01. | `T4/table_pat_weights.csv`; NTDB DB (`v_patients_2019_2024`, `raw_preexisting`); `PUF_TRAUMA.csv` for 2020 to 2024 | `T4/table_sensitivity_all_trauma.csv`, `D4/all_trauma_scored.parquet` **(PL)**, `D4/all_trauma_exclusions.json`, `F4/figure_sensitivity_all_trauma` (`.pdf`, `.png`) |
| `12_sensitivity_whole_blood.py` | Applies the unchanged integer score to the validation sample under a broader outcome (platelets or whole blood within 4 hours) and reports discrimination and calibration for both outcomes. | `D4/candidates.parquet`, `D4/test_inc_keys.csv`, `T4/table_pat_weights.csv` | `T4/table_sensitivity_whole_blood.csv`, `F4/figure_sensitivity_whole_blood_calibration` (`.png`, `.pdf`) |

## Stage 2 scripts

All stage 2 scripts load the stage 1 frozen outputs through `_common.py`, so every row below also reads some of: `D4/candidates.parquet`, `D4/cohort_raw.parquet`, `D4/cohort_test_scored.parquet`, `D4/train_inc_keys.csv`, `D4/test_inc_keys.csv`, `D4/pat_refit.json`, `T4/table_pat_weights.csv`. The table lists the loaders each script uses and anything it reads beyond them.

| Script | Purpose | Inputs | Outputs |
|---|---|---|---|
| `_common.py` | Shared module imported by the other stage 2 scripts: paths, the frozen split and score loaders, variable labels, the reference checkpoints from stage 1, JSON-part and log helpers. Run with `--self-test` to re-check the stage 1 outputs against those checkpoints. | `D4/candidates.parquet`, `D4/train_inc_keys.csv`, `D4/test_inc_keys.csv`, `D4/cohort_test_scored.parquet`, `D4/cohort_raw.parquet`, `D4/pat_refit.json`, `T4/table_pat_weights.csv` | none by itself (its helpers write the JSON parts and logs of the other scripts) |
| `00_recalibrate.py` | Recalibration under one convention: the score-to-probability map is fitted once on the derivation sample and applied unchanged to the validation sample. Reports calibration-in-the-large, slope, Brier score and grouped calibration. | split and scored loaders | `T6/table_v6_calibration.csv`, `T6/table_v6_calibration_deciles.csv`, `P6/calibration.json`, `L6/00_recalibrate.log`, `L6/00_HANDOFF.md` |
| `01_risk_bands.py` | Bedside risk bands defined on the derivation sample and applied unchanged to the validation sample; the attainable score range by exhaustive enumeration of the allowed predictor patterns. | split loader, points loader | `T6/table_v6_risk_bands.csv`, `T6/table_v6_score_distribution.csv`, `P6/risk_bands.json`, `L6/01_risk_bands.log`, `L6/01_HANDOFF.md` |
| `02_vif_table.py` | Per-predictor variance inflation factors, the full pairwise correlation matrix and the condition number on the derivation sample. | split loader, points loader | `T6/table_v6_vif.csv`, `T6/table_v6_correlation.csv`, `P6/vif.json`, `L6/02_vif_table.log` |
| `03_head_ais_sensitivity.py` | Sensitivity refit that adds maximum head Abbreviated Injury Scale, and the teaching-status subgroup analysis. Imports functions from stage 1 steps 07 and 09 and from stage 2 step 11. | split, scored and cohort-raw loaders; refit JSON | `T6/table_v6_head_ais.csv`, `T6/table_v6_teaching_subgroup.csv`, `P6/head_ais.json`, `P6/teaching.json`, `L6/03_head_ais_sensitivity.log`, `L6/03_HANDOFF.md` |
| `04_missingness.py` | Missingness of the outcome and predictor fields, comparator denominators, nested-predictor counts, and a re-verification of the six-step cohort flow against the exclusion ledger. | `D4/cohort_exclusions.json`, `T4/table_consort.csv` (see the notes below), cohort-raw, candidates, scored and points loaders | `T6/table_v6_missingness.csv`, `T6/table_v6_cohort_flow.csv`, `P6/missingness.json`, `L6/04_missingness.log` |
| `05_mortality_or_drop.py` | Re-measures in-hospital mortality among the analytic cohort directly from the registry, under several death definitions, and reports each. | `D4/cohort_raw.parquet`, `D4/train_inc_keys.csv`, `D4/test_inc_keys.csv`; NTDB DB (`v_patients_2019_2024`, `raw_icd_diagnosis`, `v_mortality`, `raw_hospital_events`) | `T6/table_v6_mortality.csv`, `P6/mortality.json`, `L6/05_mortality_or_drop.log`, `L6/05_HANDOFF.md` |
| `06a_table1_stats.py` | The statistics behind Table 1: cohort characteristics by platelet receipt, standardized mean differences, and the missingness column. Computes the statistics, not the table layout. | `P6/missingness.json` (from step 04), `D4/candidates.parquet`, cohort-raw and points loaders | `T6/table_v6_table1.csv`, `P6/table1.json`, `L6/06a_table1_stats.log` |
| `09_decision_curve.py` | Decision curves redrawn under the single recalibration convention of step 00, each curve on its own denominator, plus a complete-case check. Replaces the decision curves of stage 1 step 08. | `T6/table_v6_calibration.csv`, `P6/calibration.json` (from step 00), `T4/table_pat_dca.csv` (for comparison with the old curves), split and scored loaders | `T6/table_v6_decision_curve.csv`, `P6/dca.json`, `F6/figure_v6_decision_curve` (`.png`, `.pdf`), `L6/09_decision_curve.log`, `L6/09_HANDOFF.md` |
| `11_recompute_intervals.py` | Recomputes every reported AUROC interval with DeLong intervals (cross-checked against a full-size bootstrap), for the five scores, all subgroup levels, the teaching-status levels, the all-trauma and whole-blood sensitivity rows and the head-injury-severity models. Replaces the subsample-bootstrap intervals of stage 1 steps 07 and 09. | `T4/table_pat_discrimination.csv`, `T4/table_subgroups.csv`, `T4/table_sensitivity_all_trauma.csv`, `T4/table_sensitivity_whole_blood.csv`, `D4/all_trauma_scored.parquet`, `P6/teaching.json`, `P6/head_ais.json`, `T6/table_v6_teaching_subgroup.csv`, candidates, scored, cohort-raw, split and refit loaders; imports stage 1 step 07 | `T6/table_v6_intervals.csv`, `P6/intervals.json`, `L6/11_recompute_intervals.log`, `L6/11_HANDOFF.md` |

## Notes

- **ED disposition exclusion.** The cohort script excludes ED_DISPOSITION_CODE values 5A_S and 5BMS. These are documented here literally by code; they are not described as interhospital transfer. Some identifiers, comments and labels inside the scripts use the words "transferred out" for this step (for example the ledger key `exclude_transferred_out`). They are names from the code as run. This documentation defines the step by the code values only.
- **Supersession.** Stage 1 step 07 (subsample-bootstrap intervals) is superseded by stage 2 step 11 for the reported intervals, and stage 1 step 08 (decision curves) is superseded by stage 2 step 09 for the reported decision curves. Calibration statistics are recomputed under one convention in stage 2 step 00. Steps 07 and 08 are kept because the stage 2 scripts import from them and read their outputs.
- **Row order.** The frozen cohort pull has no ORDER BY, so cohort counts and the exclusion ledger reproduce exactly, but derivation/validation membership and LASSO subsamples may not; the stage 2 scripts also read patient-level frozen outputs that are not shipped.
- **Stage 2 checks.** Stage 2 scripts compare what they load against reference values stored in `_common.py` (cohort size, split sizes, event counts, score range, validation AUROC) and stop with an error if a value differs. If your stage 1 outputs differ from the reference run, for instance because the derivation/validation membership differs, expect that stop; it is the check working, not a defect in your environment.
- **Step 00 of stage 1** parses `v3/code/02_engineer_features.py` under the work folder. That earlier-stage script is not part of this repository, so step 00 runs only if you supply it. It is an audit that documents dropped candidates; steps 01 to 12 do not depend on it.
- **`T4/table_consort.csv`** is read by stage 2 step 04 (columns `N_before`, `N_after` and `N_excluded`; six rows, the first an entry row with a blank `N_before`, one row per later ledger step). It is written by a cohort-flow figure script that is not part of this release. Without it, step 04 stops, and step 06a (which needs the step 04 JSON part) cannot run. Step 04 only uses the file to cross-check its own re-derivation of the ladder from `D4/cohort_exclusions.json`.
- **Module imports.** Stage 2 steps 03 and 11 load stage 1 steps 07 and 09, and each other, as modules by file path. Keep the two stage folders in the layout shown here. Step 11 also loads step 03 and step 03 loads step 11; only their functions are used, not their outputs, for that import.
- **Step 11 re-run hazard.** A comment block near the top of `11_recompute_intervals.py` records that the "before" column of its correction table reads interval bounds from the outputs of step 03; if step 03 has already been re-run with the corrected intervals, a re-run of step 11 reads the corrected bounds as the "before" side and those changes print as zero. Read that comment before re-running step 11.
- **Smoke runs.** Stage 1 steps 00, 01, 02, 03, 05, 06, 08, 09 and 11 accept `--sample N`; steps 04, 07 and 12 accept `--bootstraps N` (step 04 also `--subsample N`); step 10 accepts `--smoke-test`. A smoke run overwrites the output files of that step, so re-run the step in full before moving on.
- Comments and docstrings inside the scripts that refer to project plans, finding numbers, hand-off notes or earlier versions point to planning documents that are not distributed with this repository.
