# PAT-TBI Score: analysis code

Code for the PAT-TBI score (Platelet Allocation Tool for Traumatic Brain Injury), an integer score that estimates the chance that an adult patient with traumatic brain injury receives a platelet transfusion within 4 hours of emergency department arrival. The score was derived and internally validated on the American College of Surgeons National Trauma Data Bank (NTDB) research data, admission years 2020 to 2024. This repository holds the analysis code and its documentation. It contains no patient-level data, no result tables and no figures.

<!-- add_doi:badges:begin -->
<!-- placeholder: scripts/add_doi.py replaces this block after the Zenodo release -->
[![DOI](https://zenodo.org/badge/DOI/@@DOI_CONCEPT@@.svg)](https://doi.org/@@DOI_CONCEPT@@)
<!-- add_doi:badges:end -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Research use only.** The score was derived and validated on a held-out part of the same national registry data. It has not been validated in an independent external cohort or prospectively. It is not a diagnostic device, it does not tell a clinician whether to give platelets, and it does not replace clinical judgment. The outcome it predicts is receipt of a platelet transfusion as recorded in the registry; that is not a measure of clinical need.

## What the code does

The scripts run in two stages.

- **Stage 1, derivation** (`analysis/stage1_derivation/`). Builds the adult traumatic brain injury cohort from the NTDB database, engineers the candidate predictors, splits the cohort into a derivation sample and a validation sample, selects predictors by LASSO stability selection, fits the unpenalized logistic model and converts it to integer points, scores the validation sample with the PAT-TBI points and with four comparator scores, and runs the validation, calibration, decision-curve, subgroup, severity-benchmark and sensitivity analyses.
- **Stage 2, validation and repair** (`analysis/stage2_validation/`). Reads the frozen outputs of stage 1 and recomputes the validation results under a single calibration convention (the score-to-probability map is fitted on the derivation sample and applied unchanged to the validation sample). It adds the risk bands, the collinearity table, the head-injury-severity sensitivity analysis, the missingness and cohort-flow checks, the statistics behind Table 1, the decision curves, and the interval estimates. It does not change the predictors, coefficients or integer points of the frozen score. The models it fits are auxiliary and none of them replaces the score: the score-to-probability map, the head-injury-severity sensitivity models and the regressions behind the collinearity table.

Which script reads and writes which file is in [`analysis/README_PIPELINE.md`](analysis/README_PIPELINE.md). Two stage 1 scripts (07 and 08) are superseded for the reported intervals and decision curves by two stage 2 scripts (11 and 09); the pipeline table says so.

## Reproduction

This needs your own access to the NTDB research data (see [`docs/DATA_AVAILABILITY.md`](docs/DATA_AVAILABILITY.md)), loaded into a DuckDB database. The full guide is [`docs/REPRODUCE.md`](docs/REPRODUCE.md).

```bash
pip install -r requirements.txt
export PAT_TBI_NTDB_DUCKDB="/path/to/ntdb_research.duckdb"
export PAT_TBI_WORKDIR="/path/to/a/writable/work/folder"

python analysis/stage1_derivation/01_build_cohort_duckdb.py
python analysis/stage1_derivation/02_engineer_features.py
python analysis/stage1_derivation/03_split_train_test.py
# ... steps 04 to 12 of stage 1, then stage 2: see docs/REPRODUCE.md
```

`PAT_TBI_NTDB_DUCKDB` is the DuckDB database built from the NTDB research data files. `PAT_TBI_WORKDIR` is a folder outside the repository where the scripts write every intermediate file, table and figure; the scripts create it. `python analysis/config.py` prints how both resolve.

### Known limits of reproduction

- The cohort script excludes ED_DISPOSITION_CODE values 5A_S and 5BMS. These are documented here literally by code; they are not described as interhospital transfer.
- The frozen cohort pull has no ORDER BY, so cohort counts and the exclusion ledger reproduce exactly, but derivation/validation membership and LASSO subsamples may not; the stage 2 scripts also read patient-level frozen outputs that are not shipped.
- Stage 1 step 04 (LASSO stability selection) takes hours.
- The scripts that built the DuckDB database from the NTDB research data files are not part of this repository.
- Stage 2 steps 04 and 06a cannot be run from this repository alone: step 04 requires `table_consort.csv`, which is written by a script that is not shipped, and step 06a needs the output of step 04.

## Repository structure

```
pat-tbi-score/
├── README.md
├── LICENSE
├── CITATION.cff
├── .zenodo.json
├── MANIFEST.sha256           # SHA-256 checksum of every other shipped file
├── requirements.txt          # pinned packages for analysis/
├── .gitignore
├── analysis/
│   ├── README_PIPELINE.md    # script-by-script inputs and outputs
│   ├── config.py             # reads PAT_TBI_NTDB_DUCKDB and PAT_TBI_WORKDIR
│   ├── stage1_derivation/    # derivation scripts
│   └── stage2_validation/    # validation and repair scripts
├── scripts/
│   └── add_doi.py            # release helper: fills the DOI placeholders
└── docs/
    ├── REPRODUCE.md
    ├── DATA_AVAILABILITY.md
    └── CHANGES_FROM_RUN_CODE.md
```

The scripts keep their original file names and numbering. They were copied from the frozen working folders in which the two stages were run, `v4` (stage 1) and `v6` (stage 2), with only path handling changed: every hard-coded local path now comes from `analysis/config.py`. [`docs/CHANGES_FROM_RUN_CODE.md`](docs/CHANGES_FROM_RUN_CODE.md) lists every changed line.

`MANIFEST.sha256` lists every other shipped file and does not list itself. To check that the files are intact, run `shasum -a 256 -c MANIFEST.sha256` from the repository root; every line should report `OK`.

## Data availability

Patient-level NTDB data are not redistributed here and must be obtained from the American College of Surgeons at <https://www.facs.org/quality-programs/trauma/quality/national-trauma-data-bank/>. This repository ships code only. Details are in [`docs/DATA_AVAILABILITY.md`](docs/DATA_AVAILABILITY.md).

## Citation

A manuscript describing the score is under review; its reference will be added here once it is published. Until then, please cite the software record. A machine-readable citation is in [`CITATION.cff`](CITATION.cff), and the "Cite this repository" button on GitHub gives BibTeX and APA versions of it.

<!-- add_doi:citation:begin -->
<!-- placeholder: scripts/add_doi.py replaces this block after the Zenodo release -->
Salman SG. PAT-TBI Score: analysis code (version 1.0.0). Zenodo; 2026. https://doi.org/@@DOI_VERSION@@

Concept DOI (all versions): [@@DOI_CONCEPT@@](https://doi.org/@@DOI_CONCEPT@@)
<!-- add_doi:citation:end -->

## License

MIT. See [`LICENSE`](LICENSE).
