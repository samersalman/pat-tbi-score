"""
Path configuration for the PAT-TBI analysis scripts.
====================================================

The scripts in ``stage1_derivation/`` and ``stage2_validation/`` are the code
that produced the reported results.  They originally pointed at a fixed local
folder tree.  Every such path now comes from this module, and this module reads
two environment variables.  Nothing else about the scripts changed (see
``docs/CHANGES_FROM_RUN_CODE.md``).

Environment variables
---------------------
PAT_TBI_NTDB_DUCKDB   Path to the NTDB research DuckDB file (the file that was
                      called ``ntdb_research.duckdb`` when the results were
                      produced).  Needed by stage 1 steps 01 and 11 and by
                      stage 2 step 05.
PAT_TBI_WORKDIR       A writable directory outside this repository.  Every
                      script reads and writes beneath it.  Needed by all
                      scripts.
PAT_TBI_NTDB_PUF_DIR  Optional.  Directory that holds the raw NTDB PUF year
                      folders (``PUF AY 2020/CSV/PUF_TRAUMA.csv`` and so on).
                      Used by stage 1 steps 01 and 11 for four hospital-level
                      columns and the whole-blood field.  When unset it
                      defaults to the folder called ``PUF`` that sits next to
                      the folder holding the DuckDB file, i.e. it is derived
                      from PAT_TBI_NTDB_DUCKDB.  If that folder is absent the
                      two steps print a WARN line per year and continue with
                      those columns empty.

Layout created under PAT_TBI_WORKDIR
------------------------------------
The as-run folder structure is reproduced one for one, so that stage 2 finds
what stage 1 wrote::

    <WORKDIR>/v4/derived/               stage 1 patient-level intermediates
    <WORKDIR>/v4/tables/                stage 1 result tables
    <WORKDIR>/v4/figures/               stage 1 figures
    <WORKDIR>/v4/extractability_audit/  stage 1 step 00 output
    <WORKDIR>/v6/tables/                stage 2 result tables
    <WORKDIR>/v6/figures/               stage 2 figures
    <WORKDIR>/v6/logs/                  stage 2 transcripts and hand-off notes
    <WORKDIR>/v6/bundle/parts/          stage 2 JSON parts
    <WORKDIR>/v3/code/02_engineer_features.py
                                        only stage 1 step 00 reads this file;
                                        it is a superseded script that is not
                                        part of this repository

The folder names ``v4`` and ``v6`` are the names the stages carried when the
code was run; the scripts create the folders they write to.

Behaviour
---------
Importing this module never fails.  A value is resolved when it is first asked
for (``config.WORKDIR``, ``config.V4_DIR``, ``config.DUCKDB_PATH`` and so on, or
the equivalent functions).  If the environment variable it depends on is unset,
a ``ConfigError`` that names the variable and says what it should point to is
raised at that point.  ``python analysis/config.py`` prints how every value
resolves.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ENV_DUCKDB = "PAT_TBI_NTDB_DUCKDB"
ENV_WORKDIR = "PAT_TBI_WORKDIR"
ENV_PUF = "PAT_TBI_NTDB_PUF_DIR"

# Locations inside the repository (do not depend on any environment variable).
ANALYSIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ANALYSIS_DIR.parent
STAGE1_DIR = ANALYSIS_DIR / "stage1_derivation"
STAGE2_DIR = ANALYSIS_DIR / "stage2_validation"
# Stage 2 steps 03 and 11 import the stage 1 scripts 07 and 09 as modules.
V4_CODE_DIR = STAGE1_DIR


class ConfigError(RuntimeError):
    """A required PAT-TBI path setting is missing."""


def _read_env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return value.strip()


def duckdb_path() -> Path:
    """Path to the NTDB research DuckDB file (PAT_TBI_NTDB_DUCKDB)."""
    value = _read_env(ENV_DUCKDB)
    if value is None:
        raise ConfigError(
            f"{ENV_DUCKDB} is not set. Set it to the full path of the NTDB "
            "research DuckDB file built from the ACS NTDB Participant User "
            "Files (the file called ntdb_research.duckdb in the original "
            "analysis), for example:\n"
            f"    export {ENV_DUCKDB}=/path/to/ntdb_database/ntdb_research.duckdb\n"
            "See docs/REPRODUCE.md and docs/DATA_AVAILABILITY.md."
        )
    return Path(os.path.expanduser(value))


def workdir() -> Path:
    """Directory under which every script reads and writes (PAT_TBI_WORKDIR)."""
    value = _read_env(ENV_WORKDIR)
    if value is None:
        raise ConfigError(
            f"{ENV_WORKDIR} is not set. Set it to a writable directory outside "
            "this repository; the scripts create v4/ and v6/ (derived, tables, "
            "figures, logs, bundle) beneath it, for example:\n"
            f"    export {ENV_WORKDIR}=$HOME/pat_tbi_work\n"
            "See docs/REPRODUCE.md."
        )
    return Path(os.path.expanduser(value)).resolve()


def puf_dir() -> Path:
    """Directory holding the raw ``PUF AY <year>`` folders.

    PAT_TBI_NTDB_PUF_DIR if set, otherwise the folder called ``PUF`` next to
    the folder that holds the DuckDB file (the layout used when the results
    were produced).
    """
    value = _read_env(ENV_PUF)
    if value is not None:
        return Path(os.path.expanduser(value))
    return duckdb_path().parent.parent / "PUF"


def puf_trauma_template() -> str:
    """``str.format`` template (field ``{yr}``) for each year's PUF_TRAUMA.csv."""
    return str(puf_dir() / "PUF AY {yr}" / "CSV" / "PUF_TRAUMA.csv")


def v3_dir() -> Path:
    return workdir() / "v3"


def v4_dir() -> Path:
    return workdir() / "v4"


def v6_dir() -> Path:
    return workdir() / "v6"


# Constants resolved on first access (PEP 562), so that ``import config`` works
# with no environment set and a script only fails if it needs a missing value.
_LAZY = {
    "DUCKDB_PATH": duckdb_path,
    "WORKDIR": workdir,
    "PUF_DIR": puf_dir,
    "PUF_TRAUMA_TPL": puf_trauma_template,
    "V3_DIR": v3_dir,
    "V4_DIR": v4_dir,
    "V6_DIR": v6_dir,
}


def __getattr__(name: str):
    try:
        resolver = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return resolver()


def __dir__():
    return sorted(list(globals()) + list(_LAZY))


def _show() -> int:
    status = 0
    print("PAT-TBI path configuration")
    print(f"  repository root       : {REPO_ROOT}")
    print(f"  stage 1 code (V4_CODE): {V4_CODE_DIR}")
    for label, attr in (
        ("WORKDIR", "WORKDIR"),
        ("V3_DIR", "V3_DIR"),
        ("V4_DIR", "V4_DIR"),
        ("V6_DIR", "V6_DIR"),
        ("DUCKDB_PATH", "DUCKDB_PATH"),
        ("PUF_TRAUMA_TPL", "PUF_TRAUMA_TPL"),
    ):
        try:
            print(f"  {label:<22}: {getattr(sys.modules[__name__], attr)}")
        except ConfigError as exc:
            status = 1
            print(f"  {label:<22}: UNSET")
            print("    " + str(exc).replace("\n", "\n    "))
    return status


if __name__ == "__main__":
    sys.exit(_show())
