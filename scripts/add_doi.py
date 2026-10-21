#!/usr/bin/env python3
"""Fill the release placeholders after the Zenodo release.

Run once the DOIs and the release date are known:

    python scripts/add_doi.py \\
        --doi-concept 10.5281/zenodo.AAAAAAA \\
        --doi-version 10.5281/zenodo.BBBBBBB \\
        --date 2026-10-10

What it patches
  README.md      the two regions between <!-- add_doi:NAME:begin --> and
                 <!-- add_doi:NAME:end --> (badges, citation)
  CITATION.cff   the release block (doi = the concept DOI, date-released, url, and an
                 identifiers list holding the concept DOI and the version DOI)
  .zenodo.json   only if it still contains a placeholder (it normally does not,
                 because Zenodo reads that file at release time, before any DOI exists)

It is idempotent. Each region is rewritten as a whole from the values given, so running
the script twice with the same values changes nothing, and running it again with
corrected values replaces the earlier values. When it finishes it checks that no
placeholder (two at-signs, a name, two at-signs) is left in the three files and exits
non-zero if one is.

Options
  --root PATH   repository root (default: the folder above this script)
  --dry-run     report what would change and write nothing
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DOI_RE = re.compile(r"^10\.5281/zenodo\.\d+$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
PLACEHOLDER_RE = re.compile(r"@@[A-Z_]+@@")
REPO_URL = "https://github.com/samersalman/pat-tbi-score"
VERSION = "1.0.0"
SOFTWARE_TITLE = "PAT-TBI Score: analysis code"


def region_badges(a) -> str:
    return (
        f"[![DOI](https://zenodo.org/badge/DOI/{a.doi_concept}.svg)](https://doi.org/{a.doi_concept})"
    )


def region_citation(a) -> str:
    year = a.date[:4]
    return (
        f"Salman SG. {SOFTWARE_TITLE} (version {VERSION}). Zenodo; {year}. "
        f"https://doi.org/{a.doi_version}\n"
        "\n"
        f"Concept DOI (all versions): [{a.doi_concept}](https://doi.org/{a.doi_concept})"
    )


def cff_release(a) -> str:
    return (
        f"doi: {a.doi_concept}\n"
        f"date-released: '{a.date}'\n"
        f"url: 'https://doi.org/{a.doi_concept}'\n"
        "identifiers:\n"
        "  - type: doi\n"
        f"    value: {a.doi_concept}\n"
        "    description: Concept DOI (all versions)\n"
        "  - type: doi\n"
        f"    value: {a.doi_version}\n"
        f"    description: Version DOI (v{VERSION})"
    )


README_REGIONS = {"badges": region_badges, "citation": region_citation}


def replace_region(src: str, name: str, body: str, open_fmt: str, close_fmt: str) -> tuple[str, bool]:
    begin, end = open_fmt.format(name), close_fmt.format(name)
    pat = re.compile(re.escape(begin) + r"\n.*?\n" + re.escape(end), re.S)
    if not pat.search(src):
        return src, False
    new = begin + "\n" + body + "\n" + end
    return pat.sub(lambda m: new, src, count=1), True


def patch_readme(src: str, a) -> str:
    for name, fn in README_REGIONS.items():
        src, found = replace_region(src, name, fn(a), "<!-- add_doi:{}:begin -->", "<!-- add_doi:{}:end -->")
        if not found:
            raise SystemExit(f"README.md: region add_doi:{name} not found")
    return src


def patch_cff(src: str, a) -> str:
    src, found = replace_region(src, "release", cff_release(a), "# add_doi:{}:begin", "# add_doi:{}:end")
    if not found:
        raise SystemExit("CITATION.cff: region add_doi:release not found")
    return src


def patch_zenodo(src: str, a) -> str:
    if not PLACEHOLDER_RE.search(src):
        return src
    src = src.replace("@@DOI_CONCEPT@@", a.doi_concept).replace("@@DOI_VERSION@@", a.doi_version)
    src = src.replace("@@RELEASE_DATE@@", a.date)
    return src


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--doi-concept", required=True, help="Zenodo concept DOI, 10.5281/zenodo.NNNNNNN (all versions)")
    ap.add_argument("--doi-version", required=True, help=f"Zenodo version DOI of release v{VERSION}")
    ap.add_argument("--date", required=True, help="release date, YYYY-MM-DD")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    problems = []
    if not DOI_RE.match(a.doi_concept):
        problems.append("--doi-concept must look like 10.5281/zenodo.NNNNNNN")
    if not DOI_RE.match(a.doi_version):
        problems.append("--doi-version must look like 10.5281/zenodo.NNNNNNN")
    if a.doi_concept == a.doi_version:
        problems.append("--doi-concept and --doi-version are the same; the concept DOI and the version DOI differ")
    if not DATE_RE.match(a.date):
        problems.append("--date must be YYYY-MM-DD")
    if problems:
        for p in problems:
            print("ERROR:", p, file=sys.stderr)
        return 2

    root = Path(a.root)
    plan = [("README.md", patch_readme), ("CITATION.cff", patch_cff), (".zenodo.json", patch_zenodo)]
    changed, left = [], []
    for rel, fn in plan:
        path = root / rel
        if not path.exists():
            print(f"ERROR: {path} not found", file=sys.stderr)
            return 2
        old = path.read_text(encoding="utf-8")
        new = fn(old, a)
        if new != old:
            changed.append(rel)
            if not a.dry_run:
                path.write_text(new, encoding="utf-8")
        if PLACEHOLDER_RE.search(new):
            left.append(rel)

    print(("Would change: " if a.dry_run else "Changed: ") + (", ".join(changed) if changed else "nothing (already up to date)"))
    if left:
        print(f"ERROR: placeholders remain in {left}", file=sys.stderr)
        return 1

    print()
    print("Next: review the change with `git diff`, then commit it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
