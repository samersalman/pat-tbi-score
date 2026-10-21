# Reproducing the analysis

This guide covers re-running the cohort build, the PAT-TBI score and its validation from the NTDB research data. The repository ships code only, so there are no result tables or figures to inspect: every result is produced by running the scripts on your own copy of the NTDB.

The script-by-script list of what each step reads and writes is in [`../analysis/README_PIPELINE.md`](../analysis/README_PIPELINE.md). It is not repeated here.

## What can and cannot be re-run

- With NTDB access and a DuckDB database built from the NTDB research data files, the whole of stage 1 (steps 01 to 12) can be run, and stage 2 after it, with two exceptions: stage 2 steps 04 and 06a cannot be run from this repository alone (step 04 requires `table_consort.csv`, which is written by a script that is not shipped, and step 06a needs the output of step 04).
- Without NTDB access, nothing can be run end to end. `python analysis/config.py` still runs and shows how the paths resolve.
- The cohort script excludes ED_DISPOSITION_CODE values 5A_S and 5BMS. These are documented here literally by code; they are not described as interhospital transfer.
- The frozen cohort pull has no ORDER BY, so cohort counts and the exclusion ledger reproduce exactly, but derivation/validation membership and LASSO subsamples may not; the stage 2 scripts also read patient-level frozen outputs that are not shipped.
- Stage 2 reads patient-level outputs of stage 1 (Parquet files and lists of patient keys). They exist only after you have run stage 1 on your own data.
- Stage 1 step 00 is an optional audit that needs an earlier-stage script that is not part of this repository, and stage 2 step 04 needs a cohort-flow table that is written by a script that is not part of this release. Both points are explained in the notes of [`README_PIPELINE.md`](../analysis/README_PIPELINE.md).

## 1. Prerequisites

- Python 3.12 and the packages in `requirements.txt`. A few script docstrings show a Python interpreter path from the machine the code was written on; ignore it and use your own environment.
- The NTDB research data for admission years 2020 to 2024, obtained from the ACS ([`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md)), loaded into a DuckDB database that has the tables and views listed there.
- The yearly `PUF_TRAUMA.csv` files (admission years 2020 to 2024), for four fields the database view does not expose.
- Memory and disk for several million encounters. Stage 1 step 11 works on the all-trauma cohort, which is much larger than the traumatic brain injury cohort, so allow for that.

## 2. Setup

```bash
git clone https://github.com/samersalman/pat-tbi-score.git
cd pat-tbi-score
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Point the scripts at your data

| Variable | Required | What it is |
|---|---|---|
| `PAT_TBI_NTDB_DUCKDB` | stage 1 steps 01 and 11, stage 2 step 05 | Full path of the NTDB DuckDB database file. |
| `PAT_TBI_WORKDIR` | every script | A writable folder **outside this repository**. The scripts create `v4/` and `v6/` beneath it and read and write everything there. It is created at run time. |
| `PAT_TBI_NTDB_PUF_DIR` | optional | Folder that holds the yearly `PUF AY <year>/CSV/PUF_TRAUMA.csv` files. If unset, the scripts look for a folder called `PUF` next to the folder that holds the DuckDB file. |

```bash
export PAT_TBI_NTDB_DUCKDB="/path/to/ntdb_database/ntdb_research.duckdb"
export PAT_TBI_WORKDIR="$HOME/pat_tbi_work"
python analysis/config.py        # prints how each value resolves
```

If a variable a script needs is unset, the script stops with a message that names the variable. If the `PUF` folder cannot be found, steps 01 and 11 print a `WARN` line per year and continue with the four fields empty, which affects the results that use them (the whole-blood outcome and the teaching-status subgroup). Check the log for those lines.

The work folder is laid out as follows. The folder names `v4` and `v6` are the names the two stages carried when the code was run.

```
$PAT_TBI_WORKDIR/
├── v4/                     stage 1
│   ├── derived/            patient-level intermediates and JSON
│   ├── tables/             result tables (CSV)
│   ├── figures/            figures
│   └── extractability_audit/   step 00 output
└── v6/                     stage 2
    ├── tables/             result tables (CSV)
    ├── figures/            figures
    ├── logs/               transcripts and hand-off notes
    └── bundle/parts/       JSON parts
```

The patient-level files in `v4/derived/` must stay in the work folder. Do not put the work folder inside the repository, do not share it, and check any table you plan to publish against your own data use agreement.

## 4. Run order

Every script reaches `analysis/config.py` (directly, or through `analysis/stage2_validation/_common.py`), so each can be started from any directory. The base random seed is 42. Step 04 derives the seed of each bootstrap resample from it (42 plus the resample index, with a further offset if a resample has to be redrawn); the other scripts that draw random numbers use 42 directly.

Stage 1:

```bash
python analysis/stage1_derivation/01_build_cohort_duckdb.py
python analysis/stage1_derivation/02_engineer_features.py
python analysis/stage1_derivation/03_split_train_test.py
python analysis/stage1_derivation/04_lasso_stability_selection.py
python analysis/stage1_derivation/05_refit_unpenalized_integer.py
python analysis/stage1_derivation/06_score_test_set.py
python analysis/stage1_derivation/07_validation_metrics.py
python analysis/stage1_derivation/08_calibration_dca_operating.py
python analysis/stage1_derivation/09_subgroups.py
python analysis/stage1_derivation/10_severity_score_benchmarks.py
python analysis/stage1_derivation/11_sensitivity_all_trauma.py
python analysis/stage1_derivation/12_sensitivity_whole_blood.py
```

Stage 2, after all of stage 1:

```bash
python analysis/stage2_validation/_common.py --self-test
python analysis/stage2_validation/00_recalibrate.py
python analysis/stage2_validation/01_risk_bands.py
python analysis/stage2_validation/02_vif_table.py
python analysis/stage2_validation/03_head_ais_sensitivity.py
python analysis/stage2_validation/04_missingness.py        # cannot be run from this repository alone: requires table_consort.csv from a script that is not shipped
python analysis/stage2_validation/05_mortality_or_drop.py
python analysis/stage2_validation/06a_table1_stats.py      # cannot be run from this repository alone: needs the step 04 output, which requires table_consort.csv from a script that is not shipped
python analysis/stage2_validation/09_decision_curve.py
python analysis/stage2_validation/11_recompute_intervals.py
```

Step 00 of stage 1 (`00_audit_extractability.py`) is not in the list because it needs a file that is not shipped; nothing downstream reads its output.

Stage 1 steps 07 and 08 are superseded for the reported intervals and decision curves by stage 2 steps 11 and 09. They still run, and the stage 2 scripts import from them and read their outputs.

Smoke runs: stage 1 steps 00, 01, 02, 03, 05, 06, 08, 09 and 11 accept `--sample N`; steps 04, 07 and 12 accept `--bootstraps N` (step 04 also `--subsample N`); step 10 accepts `--smoke-test`. A smoke run overwrites the output files of that step, so re-run the step in full before moving on.

## 5. Runtime

- Stage 1 step 04 (LASSO stability selection, 100 bootstrap resamples) takes **one to a few hours**, depending on the hardware (the original full run took about 140 minutes). Plan for it to run unattended.
- Stage 1 steps 01 to 03 take well under a minute each on a laptop with the DuckDB file on a local SSD; the time of the database queries in steps 01 and 11 depends on how that file is stored. The later steps, which include the bootstrap, validation and sensitivity analyses, take longer.
- Stage 2 steps take seconds to minutes each.

## 6. What to check

After stage 1 steps 01 and 03, these values should match. They are the reproducibility checkpoints; steps 04 onward are not expected to match to the last decimal (see below).

**Cohort exclusion ledger.** `v4/derived/cohort_exclusions.json` (written by step 01) holds the number of encounters left after each step. The step names are labels carried over from the code as run; each step is defined by the filter shown here, not by its label.

| Step | Ledger key | Filter applied | Encounters remaining |
|---|---|---|---|
| 1 | `start_2020_2024_age_18plus_from_duckdb` | admission years 2020 to 2024, age 18 or older | 5,159,006 |
| 2 | `tbi_S06_diagnosis` | any S06 ICD-10-CM diagnosis | 1,161,027 |
| 3 | `exclude_prehospital_arrest` | PREHOSPITAL_ARREST = Y removed | 1,129,900 |
| 4 | `exclude_ed_disposition_expired` | ED_DISPOSITION_CODE = MOR removed | 1,122,417 |
| 5 | `exclude_transferred_out` | ED_DISPOSITION_CODE = 5A_S or 5BMS removed | 648,667 |
| 6 | `exclude_missing_ed_disposition` | ED_DISPOSITION_CODE missing removed | 623,587 |

The analytic cohort is the last row, 623,587 encounters.

**Split.** `v4/tables/table_split_balance.csv` (written by step 03; columns `split`, `N`, `N_events`, `event_rate`).

| Split | N | N_events |
|---|---|---|
| `train` (derivation) | 415,724 | 8,982 |
| `test` (validation) | 207,863 | 4,491 |

The split sizes and the event counts are fixed by the cohort size and the total event count (the split is stratified on the primary outcome), so they are expected to reproduce even when the membership of the two samples does not.

**What may differ.** The cohort query that produced the frozen results has no ORDER BY. DuckDB returns rows in no guaranteed order, so a re-run can hand the seeded split and the seeded LASSO resamples the rows in a different order. The ledger counts, the split sizes and the event counts do not depend on row order and reproduce exactly. Which encounters fall in the derivation and validation samples, which bootstrap subsamples the LASSO sees, and therefore the selection frequencies, coefficients, integer points and validation metrics downstream may differ from the original run.

**Stage 2 checks.** Stage 2 scripts compare what they load with reference values stored in `analysis/stage2_validation/_common.py` (cohort size, split sizes, event counts, score range, validation AUROC) and stop with an error if a value differs. `python analysis/stage2_validation/_common.py --self-test` runs those comparisons on their own. If your stage 1 outputs differ from the original run, expect a stop there; it is the check working. In that case, stage 2 can be inspected but not run unchanged.

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ConfigError: PAT_TBI_WORKDIR is not set` | The variable is not set in this shell | `export PAT_TBI_WORKDIR=...` (section 3) |
| `ConfigError: PAT_TBI_NTDB_DUCKDB is not set` | The variable is not set in this shell | `export PAT_TBI_NTDB_DUCKDB=...` (section 3) |
| `FATAL: DuckDB not found at ...` | Wrong path in `PAT_TBI_NTDB_DUCKDB` | Check the path |
| `WARN` lines naming `PUF_TRAUMA.csv` in step 01 or 11 | The `PUF` folder is not next to the DuckDB folder | Set `PAT_TBI_NTDB_PUF_DIR` |
| A step cannot find `cohort_raw.parquet`, `candidates.parquet`, a keys file or a weights table | An earlier step has not been run in this work folder | Run the earlier steps in order, with the same `PAT_TBI_WORKDIR` |
| Stage 2 stops with `RuntimeError` or `AssertionError` about a count, a score range or a validation AUROC | Your stage 1 outputs differ from the reference run | See "What may differ" in section 6 |
| Stage 2 step 04 stops because `table_consort.csv` is missing | That table is written by a script that is not part of this release | See the notes in `README_PIPELINE.md` |
| Stage 1 step 00 stops because `v3/code/02_engineer_features.py` is missing | That earlier-stage script is not shipped | Skip step 00; nothing downstream reads its output |
