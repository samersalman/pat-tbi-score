# Changes from the as-run code

The scripts under `analysis/stage1_derivation/` and `analysis/stage2_validation/` are copied from the frozen v4 (stage 1) and v6 (stage 2) working folders with only path handling changed, and keep their original file names and numbering. This file lists every difference between the copies in this repository and the copies in those working folders (labelled `as-run/` in the diffs below). There is one kind of difference: how a script finds its input and output folders.

**Rule: every changed line is a path line.** No statistical, modelling, filtering, seed or output logic was edited, and no comment text was changed except lines that contained a local folder name. The rule was checked by script: each added and each removed line was matched against the explicit whitelist below, and the diff was accepted only if no line was left over.

The working-folder scripts pointed at a fixed local folder tree. In the removed lines below that folder is shown as `<LOCAL_ROOT>`; the actual text was a user-specific absolute path, which this repository does not ship. Added lines are shown exactly as they are in the repository. `analysis/config.py` is new (it holds the path logic and is not part of the diff); it reads `PAT_TBI_NTDB_DUCKDB` and `PAT_TBI_WORKDIR`, and optionally `PAT_TBI_NTDB_PUF_DIR`.

## Whitelist

Added lines may be only:

- import of os/sys under private names, to make analysis/config.py importable: `^import os as _os, sys as _sys  # path handling: make analysis/config\.py importable$`
- sys.path insertion of the analysis/ folder (where config.py lives): `^_sys\.path\.insert\(0, _os\.path\.dirname\(_os\.path\.dirname\(_os\.path\.abspath\(__file__\)\)\)\)$`
- import of config: `^import config  # noqa: E402  \(analysis/config\.py: PAT_TBI_WORKDIR, PAT_TBI_NTDB_DUCKDB\)$`
- assignment of a path constant from config: `^(PAT|PAT_V3|PAT_V4|V4|V6|V4_CODE|DB_PATH|DUCKDB_PATH|PUF_TRAUMA_TPL) = (str\()?config\.(V3_DIR|V4_DIR|V6_DIR|V4_CODE_DIR|DUCKDB_PATH|PUF_TRAUMA_TPL)\)?$`
- location of the stage 1 scripts that stage 2 loads as modules: `^    path = C\.V4_CODE / f"\{stem\}\.py"$`
- neutral cd line in a usage docstring: `^    cd analysis/stage2_validation$`

Removed lines may be only:

- hard-coded v4 folder as a string: `^PAT = "<LOCAL_ROOT>/PAT_TBI/v4"$`
- hard-coded base folder: `^BASE = Path\("<LOCAL_ROOT>"\)$`
- folder built from the base folder: `^(PAT_V3|PAT_V4|PAT) = BASE / "PAT_TBI" / "v[34]"$`
- DuckDB path built from the base folder: `^DUCKDB_PATH = BASE / "_DATA" / "NTDB" / "ntdb_database" / "ntdb_research\.duckdb"$`
- raw PUF template built from the base folder: `^PUF_TRAUMA_TPL = str\(BASE / "_DATA" / "NTDB" / "PUF" / "PUF AY \{yr\}" / "CSV" / "PUF_TRAUMA\.csv"\)$`
- hard-coded v4 / v6 folder as a Path: `^V[46] = Path\("<LOCAL_ROOT>/PAT_TBI/v[46]"\)$`
- hard-coded DuckDB path (first line): `^DB_PATH = Path\("<LOCAL_ROOT>/_DATA/NTDB/"$`
- hard-coded DuckDB path (continuation line): `^               "ntdb_database/ntdb_research\.duckdb"\)$`
- old location of the stage 1 scripts: `^    path = C\.V4 / "code" / f"\{stem\}\.py"$`
- cd line in a usage docstring that embedded the local folder: `^    cd "<LOCAL_ROOT>/PAT_TBI/v6/code"$`

(Patterns are Python regular expressions, matched against the whole line; `<LOCAL_ROOT>` stands for the local folder.)

Unchanged files (1): `stage2_validation/01_risk_bands.py` (already reaches everything through `_common`; nothing to change).

## Diff

### `stage1_derivation/00_audit_extractability.py`  (-3 +5)

```diff
--- as-run/v4/code/00_audit_extractability.py
+++ analysis/stage1_derivation/00_audit_extractability.py
@@ -49,7 +49,9 @@
 # Paths
 # --------------------------------------------------------------------------
-BASE = Path("<LOCAL_ROOT>")
-PAT_V3 = BASE / "PAT_TBI" / "v3"
-PAT_V4 = BASE / "PAT_TBI" / "v4"
+import os as _os, sys as _sys  # path handling: make analysis/config.py importable
+_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
+import config  # noqa: E402  (analysis/config.py: PAT_TBI_WORKDIR, PAT_TBI_NTDB_DUCKDB)
+PAT_V3 = config.V3_DIR
+PAT_V4 = config.V4_DIR
 AUDIT_DIR = PAT_V4 / "extractability_audit"
 AUDIT_DIR.mkdir(parents=True, exist_ok=True)
```

### `stage1_derivation/01_build_cohort_duckdb.py`  (-4 +6)

```diff
--- as-run/v4/code/01_build_cohort_duckdb.py
+++ analysis/stage1_derivation/01_build_cohort_duckdb.py
@@ -95,11 +95,13 @@
 # Paths
 # --------------------------------------------------------------------------
-BASE = Path("<LOCAL_ROOT>")
-PAT_V4 = BASE / "PAT_TBI" / "v4"
+import os as _os, sys as _sys  # path handling: make analysis/config.py importable
+_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
+import config  # noqa: E402  (analysis/config.py: PAT_TBI_WORKDIR, PAT_TBI_NTDB_DUCKDB)
+PAT_V4 = config.V4_DIR
 DERIVED = PAT_V4 / "derived"
 DERIVED.mkdir(parents=True, exist_ok=True)
 
-DUCKDB_PATH = BASE / "_DATA" / "NTDB" / "ntdb_database" / "ntdb_research.duckdb"
-PUF_TRAUMA_TPL = str(BASE / "_DATA" / "NTDB" / "PUF" / "PUF AY {yr}" / "CSV" / "PUF_TRAUMA.csv")
+DUCKDB_PATH = config.DUCKDB_PATH
+PUF_TRAUMA_TPL = config.PUF_TRAUMA_TPL
 
 OUT_PARQUET = DERIVED / "cohort_raw.parquet"
```

### `stage1_derivation/02_engineer_features.py`  (-1 +4)

```diff
--- as-run/v4/code/02_engineer_features.py
+++ analysis/stage1_derivation/02_engineer_features.py
@@ -66,5 +66,8 @@
 np.random.seed(SEED)
 
-PAT = "<LOCAL_ROOT>/PAT_TBI/v4"
+import os as _os, sys as _sys  # path handling: make analysis/config.py importable
+_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
+import config  # noqa: E402  (analysis/config.py: PAT_TBI_WORKDIR, PAT_TBI_NTDB_DUCKDB)
+PAT = str(config.V4_DIR)
 DERIVED = os.path.join(PAT, "derived")
 TABLES = os.path.join(PAT, "tables")
```

### `stage1_derivation/03_split_train_test.py`  (-1 +4)

```diff
--- as-run/v4/code/03_split_train_test.py
+++ analysis/stage1_derivation/03_split_train_test.py
@@ -48,5 +48,8 @@
 np.random.seed(SEED)
 
-PAT = "<LOCAL_ROOT>/PAT_TBI/v4"
+import os as _os, sys as _sys  # path handling: make analysis/config.py importable
+_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
+import config  # noqa: E402  (analysis/config.py: PAT_TBI_WORKDIR, PAT_TBI_NTDB_DUCKDB)
+PAT = str(config.V4_DIR)
 DERIVED = os.path.join(PAT, "derived")
 TABLES = os.path.join(PAT, "tables")
```

### `stage1_derivation/04_lasso_stability_selection.py`  (-1 +4)

```diff
--- as-run/v4/code/04_lasso_stability_selection.py
+++ analysis/stage1_derivation/04_lasso_stability_selection.py
@@ -119,5 +119,8 @@
 np.random.seed(SEED)
 
-PAT = "<LOCAL_ROOT>/PAT_TBI/v4"
+import os as _os, sys as _sys  # path handling: make analysis/config.py importable
+_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
+import config  # noqa: E402  (analysis/config.py: PAT_TBI_WORKDIR, PAT_TBI_NTDB_DUCKDB)
+PAT = str(config.V4_DIR)
 DERIVED = os.path.join(PAT, "derived")
 TABLES = os.path.join(PAT, "tables")
```

### `stage1_derivation/05_refit_unpenalized_integer.py`  (-1 +4)

```diff
--- as-run/v4/code/05_refit_unpenalized_integer.py
+++ analysis/stage1_derivation/05_refit_unpenalized_integer.py
@@ -57,5 +57,8 @@
 np.random.seed(SEED)
 
-PAT = "<LOCAL_ROOT>/PAT_TBI/v4"
+import os as _os, sys as _sys  # path handling: make analysis/config.py importable
+_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
+import config  # noqa: E402  (analysis/config.py: PAT_TBI_WORKDIR, PAT_TBI_NTDB_DUCKDB)
+PAT = str(config.V4_DIR)
 DERIVED = os.path.join(PAT, "derived")
 TABLES = os.path.join(PAT, "tables")
```

### `stage1_derivation/06_score_test_set.py`  (-1 +4)

```diff
--- as-run/v4/code/06_score_test_set.py
+++ analysis/stage1_derivation/06_score_test_set.py
@@ -75,5 +75,8 @@
 # Absolute paths
 # ---------------------------------------------------------------------------
-PAT = "<LOCAL_ROOT>/PAT_TBI/v4"
+import os as _os, sys as _sys  # path handling: make analysis/config.py importable
+_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
+import config  # noqa: E402  (analysis/config.py: PAT_TBI_WORKDIR, PAT_TBI_NTDB_DUCKDB)
+PAT = str(config.V4_DIR)
 DERIVED = os.path.join(PAT, "derived")
 TABLES = os.path.join(PAT, "tables")
```

### `stage1_derivation/07_validation_metrics.py`  (-1 +4)

```diff
--- as-run/v4/code/07_validation_metrics.py
+++ analysis/stage1_derivation/07_validation_metrics.py
@@ -69,5 +69,8 @@
 EPS = 1e-6
 
-PAT = "<LOCAL_ROOT>/PAT_TBI/v4"
+import os as _os, sys as _sys  # path handling: make analysis/config.py importable
+_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
+import config  # noqa: E402  (analysis/config.py: PAT_TBI_WORKDIR, PAT_TBI_NTDB_DUCKDB)
+PAT = str(config.V4_DIR)
 DERIVED = os.path.join(PAT, "derived")
 TABLES = os.path.join(PAT, "tables")
```

### `stage1_derivation/08_calibration_dca_operating.py`  (-1 +4)

```diff
--- as-run/v4/code/08_calibration_dca_operating.py
+++ analysis/stage1_derivation/08_calibration_dca_operating.py
@@ -80,5 +80,8 @@
 EPS = 1e-6
 
-PAT = "<LOCAL_ROOT>/PAT_TBI/v4"
+import os as _os, sys as _sys  # path handling: make analysis/config.py importable
+_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
+import config  # noqa: E402  (analysis/config.py: PAT_TBI_WORKDIR, PAT_TBI_NTDB_DUCKDB)
+PAT = str(config.V4_DIR)
 DERIVED = os.path.join(PAT, "derived")
 TABLES = os.path.join(PAT, "tables")
```

### `stage1_derivation/09_subgroups.py`  (-1 +4)

```diff
--- as-run/v4/code/09_subgroups.py
+++ analysis/stage1_derivation/09_subgroups.py
@@ -64,5 +64,8 @@
 # Paths
 # ------------------------------------------------------------
-PAT = "<LOCAL_ROOT>/PAT_TBI/v4"
+import os as _os, sys as _sys  # path handling: make analysis/config.py importable
+_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
+import config  # noqa: E402  (analysis/config.py: PAT_TBI_WORKDIR, PAT_TBI_NTDB_DUCKDB)
+PAT = str(config.V4_DIR)
 DERIVED = os.path.join(PAT, "derived")
 TABLES_DIR = os.path.join(PAT, "tables")
```

### `stage1_derivation/10_severity_score_benchmarks.py`  (-1 +4)

```diff
--- as-run/v4/code/10_severity_score_benchmarks.py
+++ analysis/stage1_derivation/10_severity_score_benchmarks.py
@@ -73,5 +73,8 @@
 EPS = 1e-6
 
-PAT = "<LOCAL_ROOT>/PAT_TBI/v4"
+import os as _os, sys as _sys  # path handling: make analysis/config.py importable
+_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
+import config  # noqa: E402  (analysis/config.py: PAT_TBI_WORKDIR, PAT_TBI_NTDB_DUCKDB)
+PAT = str(config.V4_DIR)
 DERIVED = os.path.join(PAT, "derived")
 TABLES = os.path.join(PAT, "tables")
```

### `stage1_derivation/11_sensitivity_all_trauma.py`  (-4 +6)

```diff
--- as-run/v4/code/11_sensitivity_all_trauma.py
+++ analysis/stage1_derivation/11_sensitivity_all_trauma.py
@@ -72,6 +72,8 @@
 # Paths
 # --------------------------------------------------------------------------
-BASE = Path("<LOCAL_ROOT>")
-PAT = BASE / "PAT_TBI" / "v4"
+import os as _os, sys as _sys  # path handling: make analysis/config.py importable
+_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
+import config  # noqa: E402  (analysis/config.py: PAT_TBI_WORKDIR, PAT_TBI_NTDB_DUCKDB)
+PAT = config.V4_DIR
 TABLES = PAT / "tables"
 FIGURES = PAT / "figures"
@@ -81,6 +83,6 @@
 DERIVED.mkdir(parents=True, exist_ok=True)
 
-DUCKDB_PATH = BASE / "_DATA" / "NTDB" / "ntdb_database" / "ntdb_research.duckdb"
-PUF_TRAUMA_TPL = str(BASE / "_DATA" / "NTDB" / "PUF" / "PUF AY {yr}" / "CSV" / "PUF_TRAUMA.csv")
+DUCKDB_PATH = config.DUCKDB_PATH
+PUF_TRAUMA_TPL = config.PUF_TRAUMA_TPL
 
 PAT_WEIGHTS_CSV = TABLES / "table_pat_weights.csv"
```

### `stage1_derivation/12_sensitivity_whole_blood.py`  (-1 +4)

```diff
--- as-run/v4/code/12_sensitivity_whole_blood.py
+++ analysis/stage1_derivation/12_sensitivity_whole_blood.py
@@ -78,5 +78,8 @@
 EPS = 1e-6
 
-PAT = "<LOCAL_ROOT>/PAT_TBI/v4"
+import os as _os, sys as _sys  # path handling: make analysis/config.py importable
+_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
+import config  # noqa: E402  (analysis/config.py: PAT_TBI_WORKDIR, PAT_TBI_NTDB_DUCKDB)
+PAT = str(config.V4_DIR)
 DERIVED = os.path.join(PAT, "derived")
 TABLES = os.path.join(PAT, "tables")
```

### `stage2_validation/00_recalibrate.py`  (-1 +1)

```diff
--- as-run/v6/code/00_recalibrate.py
+++ analysis/stage2_validation/00_recalibrate.py
@@ -71,5 +71,5 @@
 Run with the framework interpreter, not bare `python3`:
     PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
-    cd "<LOCAL_ROOT>/PAT_TBI/v6/code"
+    cd analysis/stage2_validation
     $PY 00_recalibrate.py
 
```

### `stage2_validation/02_vif_table.py`  (-1 +1)

```diff
--- as-run/v6/code/02_vif_table.py
+++ analysis/stage2_validation/02_vif_table.py
@@ -32,5 +32,5 @@
 Run with the framework interpreter, not bare `python3`:
     PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
-    cd "<LOCAL_ROOT>/PAT_TBI/v6/code"
+    cd analysis/stage2_validation
     $PY 02_vif_table.py
 
```

### `stage2_validation/03_head_ais_sensitivity.py`  (-2 +2)

```diff
--- as-run/v6/code/03_head_ais_sensitivity.py
+++ analysis/stage2_validation/03_head_ais_sensitivity.py
@@ -56,5 +56,5 @@
 Run with the framework interpreter, not bare `python3`:
     PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
-    cd "<LOCAL_ROOT>/PAT_TBI/v6/code"
+    cd analysis/stage2_validation
     $PY 03_head_ais_sensitivity.py
 
@@ -171,5 +171,5 @@
 # ---------------------------------------------------------------------------
 def _load_v4_module(stem: str, alias: str):
-    path = C.V4 / "code" / f"{stem}.py"
+    path = C.V4_CODE / f"{stem}.py"
     spec = importlib.util.spec_from_file_location(alias, path)
     mod = importlib.util.module_from_spec(spec)
```

### `stage2_validation/04_missingness.py`  (-1 +1)

```diff
--- as-run/v6/code/04_missingness.py
+++ analysis/stage2_validation/04_missingness.py
@@ -13,5 +13,5 @@
 Run with the framework interpreter, not bare `python3`:
     PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
-    cd "<LOCAL_ROOT>/PAT_TBI/v6/code"
+    cd analysis/stage2_validation
     $PY 04_missingness.py
 
```

### `stage2_validation/05_mortality_or_drop.py`  (-2 +4)

```diff
--- as-run/v6/code/05_mortality_or_drop.py
+++ analysis/stage2_validation/05_mortality_or_drop.py
@@ -54,6 +54,8 @@
 import _common as C  # noqa: E402
 
-DB_PATH = Path("<LOCAL_ROOT>/_DATA/NTDB/"
-               "ntdb_database/ntdb_research.duckdb")
+import os as _os, sys as _sys  # path handling: make analysis/config.py importable
+_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
+import config  # noqa: E402  (analysis/config.py: PAT_TBI_WORKDIR, PAT_TBI_NTDB_DUCKDB)
+DB_PATH = config.DUCKDB_PATH
 
 YEARS = (2020, 2021, 2022, 2023, 2024)
```

### `stage2_validation/06a_table1_stats.py`  (-1 +1)

```diff
--- as-run/v6/code/06a_table1_stats.py
+++ analysis/stage2_validation/06a_table1_stats.py
@@ -17,5 +17,5 @@
 Run with the framework interpreter, not bare `python3`:
     PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
-    cd "<LOCAL_ROOT>/PAT_TBI/v6/code"
+    cd analysis/stage2_validation
     $PY 04_missingness.py     # must run first; supplies the missingness column
     $PY 06a_table1_stats.py
```

### `stage2_validation/09_decision_curve.py`  (-1 +1)

```diff
--- as-run/v6/code/09_decision_curve.py
+++ analysis/stage2_validation/09_decision_curve.py
@@ -57,5 +57,5 @@
 Run with the framework interpreter, not bare `python3`:
     PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
-    cd "<LOCAL_ROOT>/PAT_TBI/v6/code"
+    cd analysis/stage2_validation
     $PY 09_decision_curve.py
 
```

### `stage2_validation/11_recompute_intervals.py`  (-1 +1)

```diff
--- as-run/v6/code/11_recompute_intervals.py
+++ analysis/stage2_validation/11_recompute_intervals.py
@@ -177,5 +177,5 @@
 # ---------------------------------------------------------------------------
 def _load_v4_module(stem: str, alias: str):
-    path = C.V4 / "code" / f"{stem}.py"
+    path = C.V4_CODE / f"{stem}.py"
     spec = importlib.util.spec_from_file_location(alias, path)
     mod = importlib.util.module_from_spec(spec)
```

### `stage2_validation/_common.py`  (-2 +6)

```diff
--- as-run/v6/code/_common.py
+++ analysis/stage2_validation/_common.py
@@ -30,6 +30,10 @@
 # Paths
 # ---------------------------------------------------------------------------
-V4 = Path("<LOCAL_ROOT>/PAT_TBI/v4")
-V6 = Path("<LOCAL_ROOT>/PAT_TBI/v6")
+import os as _os, sys as _sys  # path handling: make analysis/config.py importable
+_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
+import config  # noqa: E402  (analysis/config.py: PAT_TBI_WORKDIR, PAT_TBI_NTDB_DUCKDB)
+V4 = config.V4_DIR
+V6 = config.V6_DIR
+V4_CODE = config.V4_CODE_DIR
 
 V4_DERIVED = V4 / "derived"
```

## Summary

22 files changed of 23 copied; 108 lines changed (33 removed, 75 added); all 108 match the path-handling whitelist.
