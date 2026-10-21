# Data availability

## What is in this repository

Code and documentation only. No patient-level data, no patient identifiers, no free-text fields, no result tables and no figures are stored here.

- `analysis/stage1_derivation/` and `analysis/stage2_validation/`: the analysis scripts.
- `analysis/config.py`: the two environment variables that point the scripts at your data and at a work folder.
- `analysis/README_PIPELINE.md`: which script reads and writes which file.
- `docs/`: this file, the reproduction guide and the list of changes made to the scripts since they were run.
- `scripts/add_doi.py`: a release helper that fills in the DOI placeholders. It does not touch data.
- `requirements.txt`, `CITATION.cff`, `.zenodo.json`, `LICENSE`, `MANIFEST.sha256`.

`.gitignore` keeps data files, intermediate files, tables, figures and manuscript documents out of the repository.

## What is not in this repository

Patient-level NTDB data are not redistributed and must be obtained directly from the American College of Surgeons (ACS). This covers the raw research data files and every file derived from them at the level of the individual patient, including the cohort file, the engineered-feature table, the scored validation set, the lists of patient keys that define the derivation and validation samples, and serialized model objects. Running the pipeline creates these locally under the folder named by `PAT_TBI_WORKDIR`, outside the repository.

The result tables and figures of the study are also not here. They are produced by running the scripts on your own copy of the NTDB.

Stage 2 reads patient-level outputs of stage 1. Those outputs exist only after you have run stage 1 on your own data, so the stage 2 scripts cannot be run from a clean clone alone. The frozen cohort pull has no ORDER BY, so cohort counts and the exclusion ledger reproduce exactly, but derivation/validation membership and LASSO subsamples may not; the stage 2 scripts also read patient-level frozen outputs that are not shipped. See [`REPRODUCE.md`](REPRODUCE.md) for what can be checked and how.

## How to obtain the NTDB research data

The NTDB research data (Public Use Files) are distributed by the ACS under a data use agreement.

1. Go to the National Trauma Data Bank page of the ACS: <https://www.facs.org/quality-programs/trauma/quality/national-trauma-data-bank/>.
2. Submit a research data application and sign the data use agreement.
3. Request admission years 2020 to 2024 to match the cohort used here. The database the scripts query also holds 2019, but only 2020 to 2024 enter the cohort.

The pipeline was run on a DuckDB database built from the NTDB research data files, together with the yearly `PUF_TRAUMA.csv` files for four fields that the database view does not expose (teaching status, verification level, highest activation level and the 4-hour whole-blood count). The scripts that built that database are not part of this repository. The objects the scripts query are:

- `v_patients_2019_2024`, a per-patient flat view restricted to admission years 2019 to 2024 (the scripts then restrict to 2020 to 2024);
- `raw_icd_diagnosis`, `raw_preexisting`, `raw_ais_diagnosis` and `raw_hospital_events`, long-format tables of diagnosis codes, pre-existing conditions, injury severity codes and hospital events;
- `lookup_ais`, a lookup from injury code and severity to body region;
- `v_mortality`, a per-patient mortality view (read by one stage 2 script).

The columns each script reads are named in the script itself; `01_build_cohort_duckdb.py` is the specification for the cohort query. A reader who builds an equivalent database from the raw files can run the pipeline from step 01, but the database build itself cannot be re-run from this repository. The ACS files are de-identified at source before release.

## Using the files you produce

Running the pipeline writes unsuppressed, patient-level intermediate files into the work folder. Keep that folder outside the repository, do not share its contents, and check any table you plan to publish against your own data use agreement.
