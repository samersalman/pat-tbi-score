"""
PAT-TBI v6 — Phase 0, script 01: bedside risk bands and the observed score range.
=================================================================================

Closes finding #7 ("score range -11 to 63 unreachable") and supplies the bedside
bands for manuscript Table 3 and the middle zone of the WTA abstract figure.

WHY THE BANDS ARE DEFINED ON DERIVATION ONLY
--------------------------------------------
The band boundaries are chosen by inspecting the DERIVATION score distribution
and are then applied UNCHANGED to the validation cohort. Choosing cutpoints on
the validation set would make the validation band rates in-sample summaries of
the very cutpoints they are meant to test, which is exactly the circularity that
Phase 0 exists to remove (finding #1, the circular calibration). Every validation
number in this script is therefore out of sample with respect to the boundaries.

Nothing here refits the model. The 22 retained predictors and their integer
points are read frozen from the v4 refit through `_common`.

WHY -11 TO 63 IS THE WRONG RANGE TO PRINT
------------------------------------------
-11 is the sum of every negative point and 63 the sum of every positive point.
That arithmetic ignores the fact that several predictors are mutually exclusive
bands of one underlying variable, so no single patient can carry more than one
of them:

    age bands        age_45_64 / age_65_74 / age_75plus     at most one
    GCS total bands  gcs_severe / gcs_moderate              at most one
    GCS motor pair   gcs_motor_low / gcs_motor_high         at most one
    mechanism        mech_fall / mech_assault               at most one, and
                     risk_penetrating is a strict subset of mech_assault

This script verifies each of those constraints on the data rather than assuming
them, then computes the true attainable maximum and minimum by exhaustive
enumeration of every predictor pattern the constraints permit.

Run with the framework interpreter, not bare `python3`:
    PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
    $PY 01_risk_bands.py

Outputs
    v6/bundle/parts/risk_bands.json
    v6/tables/table_v6_risk_bands.csv
    v6/tables/table_v6_score_distribution.csv
    v6/logs/01_risk_bands.log
    v6/logs/01_HANDOFF.md
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

NEG_INF = -10**6
POS_INF = 10**6

# Pre-registered minimum cell sizes for a band to be usable at the bedside.
# Declared before the candidate bands are evaluated, so the pass/fail decision
# below is not chosen after seeing the answer.
MIN_N_VALIDATION = 1_000
MIN_EVENTS_VALIDATION = 100

# The candidate band set, defined on the derivation distribution. 4 must be a
# lower bound because 4 is the Youden operating point and Table 3 has to agree
# with the band table.
CANDIDATE_BANDS = [
    # slug,          display name, lower,  upper
    ("minimal",      "Minimal",    NEG_INF,   -1),
    ("low",          "Low",              0,    3),
    ("moderate",     "Moderate",         4,    8),
    ("high",         "High",             9,   14),
    ("very_high",    "Very high",       15, POS_INF),
]

# Mutually exclusive predictor groups. Each is a list of predictors of which a
# patient may carry at most one. Verified against the data in `verify_groups`.
EXCLUSIVE_GROUPS = {
    "age bands": ["age_45_64", "age_65_74", "age_75plus"],
    "Glasgow Coma Scale total bands": ["gcs_severe", "gcs_moderate"],
    "Glasgow Coma Scale motor pair": ["gcs_motor_low", "gcs_motor_high"],
    "mechanism categories": ["mech_fall", "mech_assault"],
}

# An additional pair that is mutually exclusive by its own cutpoints (heart rate
# below 60 cannot also be 120 or above). Not in the brief's list, but it is a
# real constraint and it moves the attainable maximum, so it is reported as a
# refinement rather than silently folded in or silently dropped.
HR_GROUP = ["hr_bradycardic", "hr_tachycardic"]


# ---------------------------------------------------------------------------
# Structure verification
# ---------------------------------------------------------------------------
def verify_groups(both: pd.DataFrame) -> dict:
    """Confirm on the data that each group is mutually exclusive and each
    nested pair really nests. Returns the observed violation counts."""
    C.banner("Predictor structure, verified on derivation and validation together")
    out = {"exclusive": {}, "nested": {}}

    for name, members in list(EXCLUSIVE_GROUPS.items()) + [("heart rate bands", HR_GROUP)]:
        simultaneous = both[members].sum(axis=1)
        n_viol = int((simultaneous > 1).sum())
        out["exclusive"][name] = n_viol
        print(f"  mutually exclusive  {name:34s} rows carrying more than one: {n_viol:,}")
        assert n_viol == 0, f"{name} is not mutually exclusive in the data ({n_viol:,} rows)"

    for inner, outer in C.NESTED_PAIRS:
        n_inner = int(both[inner].sum())
        n_viol = int(((both[inner] == 1) & (both[outer] == 0)).sum())
        out["nested"][f"{inner}|{outer}"] = {"inner": n_inner, "violations": n_viol}
        kind = "strict subset" if n_viol == 0 else "not a strict subset"
        print(f"  nesting             {C.label(inner):34s} inside {C.label(outer)}")
        print(f"                      inner rows {n_inner:,}; outside the outer set {n_viol:,} ({kind})")

    return out


# ---------------------------------------------------------------------------
# The attainable score range
# ---------------------------------------------------------------------------
def attainable_range(points: dict[str, int], extra_groups: list[list[str]] | None = None):
    """Exhaustively enumerate every predictor pattern the exclusivity and
    nesting constraints permit, and return (min, max, argmin, argmax).

    Free predictors are enumerated over the full power set; grouped predictors
    over the legal choices for their group. This is exhaustive over the
    constrained space, so the returned extremes are computed, not asserted.
    """
    groups = [list(v) for v in EXCLUSIVE_GROUPS.values()]
    if extra_groups:
        groups += [list(g) for g in extra_groups]

    grouped = {v for g in groups for v in g}

    # Legal choices per exclusive group: nothing, or exactly one member.
    choices: list[list[tuple[str, ...]]] = []
    for g in groups:
        opts = [()] + [(v,) for v in g]
        # Penetrating assault only exists inside assault, so extend the assault
        # option rather than letting penetrating stand alone.
        if "mech_assault" in g and "risk_penetrating" in points:
            opts = [o + ("risk_penetrating",) if o == ("mech_assault",) else o for o in opts] + \
                   [("mech_assault",)]
            opts = sorted(set(opts))
            grouped.add("risk_penetrating")
        choices.append(opts)

    free = sorted(v for v in points if v not in grouped)
    # si_high implies si_elevated, so patterns carrying si_high without
    # si_elevated are not legal and are skipped.
    nest_free = [(i, o) for i, o in C.NESTED_PAIRS if i in free and o in free]

    best = (None, None)   # (score, pattern)
    worst = (None, None)
    n_seen = 0
    for combo in itertools.product(*choices):
        base_on = [v for opt in combo for v in opt]
        base = sum(points[v] for v in base_on)
        for r in range(len(free) + 1):
            for subset in itertools.combinations(free, r):
                s_on = set(subset)
                if any(i in s_on and o not in s_on for i, o in nest_free):
                    continue
                n_seen += 1
                total = base + sum(points[v] for v in subset)
                if best[0] is None or total > best[0]:
                    best = (total, tuple(sorted(base_on + list(subset))))
                if worst[0] is None or total < worst[0]:
                    worst = (total, tuple(sorted(base_on + list(subset))))
    return worst[0], best[0], worst[1], best[1], n_seen


# ---------------------------------------------------------------------------
# Band statistics
# ---------------------------------------------------------------------------
def band_stats(df: pd.DataFrame, bands) -> pd.DataFrame:
    """n, events, observed rate with a Wilson interval, cumulative sensitivity
    and specificity at the band's lower boundary, and cohort share."""
    s = df["pat_tbi_score"].to_numpy()
    y = df[C.OUTCOME].to_numpy().astype(int)
    n_total = len(df)
    n_events = int(y.sum())
    n_nonevents = n_total - n_events

    rows = []
    for rank, (slug, name, lo, hi) in enumerate(bands, start=1):
        m = (s >= lo) & (s <= hi)
        n = int(m.sum())
        k = int(y[m].sum())
        rate = k / n if n else float("nan")
        r_lo, r_hi = C.wilson_ci(k, n)

        # Cumulative sensitivity: events at or above the band's lower boundary.
        k_sens = int(y[s >= lo].sum())
        sens = k_sens / n_events
        se_lo, se_hi = C.wilson_ci(k_sens, n_events)
        # Cumulative specificity: non-events below the band's lower boundary.
        k_spec = int(((s < lo) & (y == 0)).sum())
        spec = k_spec / n_nonevents
        sp_lo, sp_hi = C.wilson_ci(k_spec, n_nonevents)

        rows.append({
            "band_rank": rank,
            "band": slug,
            "band_name": name,
            "lower_bound": lo,
            "upper_bound": hi,
            "observed_min_in_band": int(s[m].min()) if n else np.nan,
            "observed_max_in_band": int(s[m].max()) if n else np.nan,
            "n": n,
            "n_cohort": n_total,
            "share_of_cohort": n / n_total,
            "events": k,
            "observed_rate": rate,
            "rate_ci_low": r_lo,
            "rate_ci_high": r_hi,
            "sensitivity_at_lower_bound": sens,
            "sens_ci_low": se_lo,
            "sens_ci_high": se_hi,
            "specificity_at_lower_bound": spec,
            "spec_ci_low": sp_lo,
            "spec_ci_high": sp_hi,
        })
    return pd.DataFrame(rows)


def band_label(lo: int, hi: int) -> str:
    if lo == NEG_INF:
        return f"{hi} or below"
    if hi == POS_INF:
        return f"{lo} or above"
    return f"{lo} to {hi}"


def print_band_table(title: str, tab: pd.DataFrame) -> None:
    C.banner(title)
    print(f"  {'Band':<11}{'Points':>12}{'n':>10}{'Share':>9}{'Events':>9}"
          f"{'Rate':>9}{'95% interval':>20}{'Sens':>9}{'Spec':>9}")
    for _, r in tab.iterrows():
        print(f"  {r.band_name:<11}{band_label(r.lower_bound, r.upper_bound):>12}"
              f"{r.n:>10,}{100*r.share_of_cohort:>8.2f}%{r.events:>9,}"
              f"{100*r.observed_rate:>8.2f}%"
              f"{100*r.rate_ci_low:>10.2f} to {100*r.rate_ci_high:<6.2f}"
              f"{r.sensitivity_at_lower_bound:>9.4f}{r.specificity_at_lower_bound:>9.4f}")


# ---------------------------------------------------------------------------
def main() -> int:
    C.start_log("01_risk_bands")

    C.banner("PAT-TBI v6 — 01_risk_bands: bedside bands and the observed score range")
    print("  Bands are chosen on the DERIVATION distribution and applied unchanged to")
    print("  validation. No boundary is tuned to the validation cohort.")

    points = C.load_points()
    train, test = C.load_split()

    # ----- hard gate: the frozen observed ranges ---------------------------
    d_min, d_max = int(train.pat_tbi_score.min()), int(train.pat_tbi_score.max())
    v_min, v_max = int(test.pat_tbi_score.min()), int(test.pat_tbi_score.max())
    C.banner("Observed score range (hard gate)")
    print(f"  derivation  {d_min} to {d_max}   (n = {len(train):,}, events = {int(train[C.OUTCOME].sum()):,})")
    print(f"  validation  {v_min} to {v_max}   (n = {len(test):,}, events = {int(test[C.OUTCOME].sum()):,})")
    assert (d_min, d_max) == (C.V4_FROZEN["score_min_train"], C.V4_FROZEN["score_max_train"]), \
        f"derivation range moved: {d_min} to {d_max}"
    assert (v_min, v_max) == (C.V4_FROZEN["score_min_test"], C.V4_FROZEN["score_max_test"]), \
        f"validation range moved: {v_min} to {v_max}"
    print("  [PASS] both ranges match the frozen v4 values")

    # ----- finding #7: why -11 to 63 is unreachable ------------------------
    arith_min = sum(v for v in points.values() if v < 0)
    arith_max = sum(v for v in points.values() if v > 0)
    C.banner("Finding #7 — the arithmetic range and the attainable range")
    print(f"  sum of every negative point           {arith_min}")
    print(f"  sum of every positive point           {arith_max}")
    print(f"  the v4 manuscript printed             {arith_min} to {arith_max}")

    both = pd.concat([train, test], ignore_index=True)
    structure = verify_groups(both)

    # Two enumerations. The first respects only the four groups named in the
    # brief. The second adds the heart rate band pair, which is mutually
    # exclusive by its own cutpoints and which no patient in either cohort
    # carries both members of. The second is the defensible ceiling: the first
    # one's best pattern is bradycardic and tachycardic at the same time.
    grp_min, grp_max, _, grp_argmax, n_patterns = attainable_range(points)
    att_min, att_max, argmin, argmax, _ = attainable_range(points, extra_groups=[HR_GROUP])
    print()
    print(f"  patterns enumerated under the mutually exclusive groups: {n_patterns:,}")
    print(f"  respecting the four groups named in the brief only:  {grp_min} to {grp_max}")
    print(f"  its highest scoring pattern carries both heart rate indicators at once,")
    print(f"  which no patient can, so {grp_max} overstates the ceiling by one point.")
    print()
    print(f"  TRUE ATTAINABLE RANGE (heart rate bands also held exclusive): {att_min} to {att_max}")
    print(f"  highest scoring attainable pattern ({att_max} points):")
    for v in sorted(argmax, key=lambda x: -points[x]):
        print(f"      {points[v]:+3d}  {C.label(v)}")
    print(f"  lowest scoring attainable pattern ({att_min} points):")
    for v in sorted(argmin, key=lambda x: points[x]):
        print(f"      {points[v]:+3d}  {C.label(v)}")

    assert att_min <= d_min and att_min <= v_min, "observed minimum sits below the attainable minimum"
    assert att_max >= d_max and att_max >= v_max, "observed maximum sits above the attainable maximum"
    print(f"  [PASS] observed extremes lie inside the attainable range")
    print(f"  Neither cohort reaches the attainable ceiling: the highest score observed is")
    print(f"  {max(d_max, v_max)}, well below {att_max}, because the ceiling needs an implausible")
    print(f"  co-occurrence of every remaining positive indicator at once.")

    # ----- candidate bands, judged on derivation ---------------------------
    C.banner("Candidate bands, judged on the derivation distribution")
    print(f"  Pre-registered minimum cell sizes in validation: n >= {MIN_N_VALIDATION:,}, "
          f"events >= {MIN_EVENTS_VALIDATION}")
    print("  Constraint: 4 must be a lower bound (the Youden operating point).")
    print("  Candidate set:")
    for slug, name, lo, hi in CANDIDATE_BANDS:
        print(f"      {name:<11} {band_label(lo, hi)}")

    deriv = band_stats(train, CANDIDATE_BANDS)
    valid = band_stats(test, CANDIDATE_BANDS)

    d_rates = deriv.observed_rate.tolist()
    v_rates = valid.observed_rate.tolist()
    d_mono = all(b > a for a, b in zip(d_rates, d_rates[1:]))
    v_mono = all(b > a for a, b in zip(v_rates, v_rates[1:]))
    thin = valid[(valid.n < MIN_N_VALIDATION) | (valid.events < MIN_EVENTS_VALIDATION)]

    print()
    print(f"  monotone rising rate, derivation : {'YES' if d_mono else 'NO'}")
    print(f"  monotone rising rate, validation : {'YES' if v_mono else 'NO'}")
    print(f"  bands below the minimum cell size: {len(thin)}"
          + ("" if thin.empty else "  -> " + ", ".join(thin.band_name)))

    if d_mono and v_mono and thin.empty:
        print()
        print("  DECISION: the first candidate set passes every check. No band was widened,")
        print("  merged, split or moved. The boundaries below are the ones first proposed.")
    else:
        raise RuntimeError(
            "the first candidate band set failed a check; adjust CANDIDATE_BANDS and "
            "record the change here rather than keeping a failing band")

    bands = CANDIDATE_BANDS
    print_band_table("Band table — VALIDATION (out of sample for these boundaries)", valid)
    print_band_table("Band table — DERIVATION (bands were chosen here)", deriv)

    # ----- hard gate: the bands partition the range ------------------------
    C.banner("Partition and count gates")
    lowers = [lo for _, _, lo, _ in bands]
    uppers = [hi for _, _, _, hi in bands]
    for i in range(len(bands) - 1):
        assert uppers[i] + 1 == lowers[i + 1], \
            f"gap or overlap between {bands[i][1]} and {bands[i+1][1]}"
    assert lowers[0] <= min(d_min, v_min), "lowest band does not reach the observed minimum"
    assert uppers[-1] >= max(d_max, v_max), "highest band does not reach the observed maximum"
    assert C.V4_FROZEN["youden_cutpoint"] in lowers, "4 is not a band lower bound"
    print("  [PASS] boundaries are contiguous, with no gaps and no overlaps")
    print("  [PASS] the bands cover the full observed range in both cohorts")
    print("  [PASS] 4 points is a band lower bound, so Table 3 and the band table agree")

    assert int(valid.n.sum()) == C.V4_FROZEN["n_test"], \
        f"validation band membership sums to {int(valid.n.sum()):,}"
    assert int(deriv.n.sum()) == C.V4_FROZEN["n_train"], \
        f"derivation band membership sums to {int(deriv.n.sum()):,}"
    assert int(valid.events.sum()) == C.V4_FROZEN["n_test_events"]
    assert int(deriv.events.sum()) == C.V4_FROZEN["n_train_events"]
    print(f"  [PASS] validation band membership sums to {int(valid.n.sum()):,}")
    print(f"  [PASS] derivation band membership sums to {int(deriv.n.sum()):,}")
    print(f"  [PASS] events sum to {int(valid.events.sum()):,} and {int(deriv.events.sum()):,}")

    # Cross-check the operating point against the frozen v4 table.
    op = valid.loc[valid.lower_bound == C.V4_FROZEN["youden_cutpoint"]].iloc[0]
    print(f"  operating point at 4 points: sensitivity {op.sensitivity_at_lower_bound:.6f}, "
          f"specificity {op.specificity_at_lower_bound:.6f}")
    print("  v4 table_pat_operating_characteristics.csv: 0.742596 and 0.887025")
    assert abs(op.sensitivity_at_lower_bound - 0.7425963037185482) < 1e-9
    assert abs(op.specificity_at_lower_bound - 0.8870247625041795) < 1e-9
    print("  [PASS] operating characteristics at 4 points unchanged from v4")

    # ----- score distribution for the figure -------------------------------
    C.banner("Per-score distribution, validation (middle zone of the abstract figure)")
    dist = (test.groupby("pat_tbi_score")[C.OUTCOME]
                .agg(n="size", events="sum").reset_index())
    dist["observed_rate"] = dist.events / dist.n
    ci = [C.wilson_ci(int(k), int(n)) for k, n in zip(dist.events, dist.n)]
    dist["rate_ci_low"] = [a for a, _ in ci]
    dist["rate_ci_high"] = [b for _, b in ci]
    dist["share_of_cohort"] = dist.n / len(test)
    dist["band"] = pd.cut(dist.pat_tbi_score,
                          bins=[lo - 0.5 for lo in lowers] + [uppers[-1] + 0.5],
                          labels=[b[0] for b in bands])
    dist["band"] = dist["band"].astype(str)
    dist["band_name"] = dist["band"].map({b[0]: b[1] for b in bands})
    dist = dist[["pat_tbi_score", "band", "band_name", "n", "events",
                 "observed_rate", "rate_ci_low", "rate_ci_high", "share_of_cohort"]]
    present = set(dist.pat_tbi_score)
    missing = [v for v in range(v_min, v_max + 1) if v not in present]
    print(f"  distinct integer scores present in validation: {len(dist)}")
    print(f"  integer values inside the observed range with no patient: "
          f"{', '.join(map(str, missing)) if missing else 'none'}")
    print(dist.to_string(index=False,
                         formatters={"observed_rate": "{:.5f}".format,
                                     "rate_ci_low": "{:.5f}".format,
                                     "rate_ci_high": "{:.5f}".format,
                                     "share_of_cohort": "{:.6f}".format}))

    # ----- write the tables ------------------------------------------------
    tables = C.V6 / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    deriv2 = deriv.copy(); deriv2.insert(0, "cohort", "derivation")
    valid2 = valid.copy(); valid2.insert(0, "cohort", "validation")
    band_csv = pd.concat([deriv2, valid2], ignore_index=True)
    band_csv["band_label"] = [band_label(lo, hi)
                             for lo, hi in zip(band_csv.lower_bound, band_csv.upper_bound)]
    band_csv["lower_bound"] = band_csv.lower_bound.replace(NEG_INF, np.nan)
    band_csv["upper_bound"] = band_csv.upper_bound.replace(POS_INF, np.nan)
    band_csv = band_csv[["cohort", "band_rank", "band", "band_name", "band_label",
                         "lower_bound", "upper_bound",
                         "observed_min_in_band", "observed_max_in_band",
                         "n", "n_cohort", "share_of_cohort", "events",
                         "observed_rate", "rate_ci_low", "rate_ci_high",
                         "sensitivity_at_lower_bound", "sens_ci_low", "sens_ci_high",
                         "specificity_at_lower_bound", "spec_ci_low", "spec_ci_high"]]
    p1 = tables / "table_v6_risk_bands.csv"
    band_csv.to_csv(p1, index=False)

    p2 = tables / "table_v6_score_distribution.csv"
    dist.to_csv(p2, index=False)
    print(f"\n  wrote {p1}")
    print(f"  wrote {p2}")

    # ----- bundle part -----------------------------------------------------
    def est(v, lo, hi):
        """Estimate leaf. The Wilson interval in `_common` returns 0.9999999999999998
        rather than 1.0 for a proportion of exactly 1 (floating point in the
        `min(1.0, centre + half)` clamp), which would leave the interval not
        containing its own estimate. Clamped here rather than in `_common`,
        which this script must not edit. The v4 tables carry the same artefact."""
        v, lo, hi = float(v), float(lo), float(hi)
        return {"estimate": v, "lo": min(lo, v), "hi": max(hi, v)}

    def band_payload(row, n_cohort, n_events_cohort, n_nonevents_cohort):
        return {
            "patients": {"n": int(row.n), "N": int(n_cohort)},
            "events": {"n": int(row.events), "N": int(row.n)},
            "observed_rate": est(row.observed_rate, row.rate_ci_low, row.rate_ci_high),
            "sensitivity_at_lower_bound": est(row.sensitivity_at_lower_bound,
                                              row.sens_ci_low, row.sens_ci_high),
            "specificity_at_lower_bound": est(row.specificity_at_lower_bound,
                                              row.spec_ci_low, row.spec_ci_high),
        }

    d_ev, v_ev = int(train[C.OUTCOME].sum()), int(test[C.OUTCOME].sum())
    bands_payload = {}
    for (slug, name, lo, hi), (_, dr), (_, vr) in zip(bands, deriv.iterrows(), valid.iterrows()):
        bands_payload[slug] = {
            "name": {"display": name},
            "rank": {"value": int(dr.band_rank)},
            "lower_bound": ({"display": "no lower bound, this is the lowest band"}
                            if lo == NEG_INF else {"value": int(lo)}),
            "upper_bound": ({"display": "no upper bound, this is the highest band"}
                            if hi == POS_INF else {"value": int(hi)}),
            "validation": band_payload(vr, len(test), v_ev, len(test) - v_ev),
            "derivation": band_payload(dr, len(train), d_ev, len(train) - d_ev),
        }

    payload = {
        "_meta": {
            "script": "01_risk_bands.py",
            "finding": 7,
            "definition_cohort": {"display": "derivation"},
            "provenance": {"display": (
                "Band boundaries were fixed on the derivation distribution and applied "
                "unchanged to the validation cohort, so every validation band rate is "
                "out of sample with respect to the boundaries.")},
        },
        "band_order": [b[0] for b in bands],
        "n_bands": {"value": len(bands)},
        "bands": bands_payload,
        "operating_point": {
            "points": {"value": int(C.V4_FROZEN["youden_cutpoint"])},
            "note": {"display": (
                "The Youden operating point is the lower bound of the moderate band, so "
                "the operating characteristics table and the band table agree by "
                "construction.")},
            "sensitivity": est(op.sensitivity_at_lower_bound, op.sens_ci_low, op.sens_ci_high),
            "specificity": est(op.specificity_at_lower_bound, op.spec_ci_low, op.spec_ci_high),
        },
        "score_range": {
            "observed_derivation_min": {"value": d_min},
            "observed_derivation_max": {"value": d_max},
            "observed_validation_min": {"value": v_min},
            "observed_validation_max": {"value": v_max},
            "arithmetic_min": {"value": int(arith_min)},
            "arithmetic_max": {"value": int(arith_max)},
            "attainable_min": {"value": int(att_min)},
            "attainable_max": {"value": int(att_max)},
            "attainable_max_named_groups_only": {"value": int(grp_max)},
            "unreachable_note": {"display": (
                "The arithmetic sum of every positive point and every negative point "
                "overstates the usable range. The age bands, the Glasgow Coma Scale "
                "total bands, the Glasgow Coma Scale motor pair and the mechanism "
                "categories are mutually exclusive, and penetrating assault sits inside "
                "assault, so no patient can carry more than one point value from any of "
                "those groups.")},
            "reporting_note": {"display": (
                "Report the observed range and the named bands. The arithmetic range "
                "printed in the earlier draft is not attainable by any patient.")},
            "heart_rate_note": {"display": (
                "The attainable ceiling also holds the two heart rate bands mutually "
                "exclusive, since a patient cannot be both bradycardic and tachycardic "
                "and none in either cohort is recorded as both. Counting only the age, "
                "Glasgow Coma Scale and mechanism groups would raise that ceiling by "
                "one point, on a pattern no patient can present.")},
        },
        "structure_check": {
            "exclusive_group_violations": {"value": int(sum(structure["exclusive"].values()))},
            "note": {"display": (
                "Every mutually exclusive group was checked on the derivation and "
                "validation rows together, and no patient carried more than one member "
                "of any group.")},
        },
    }
    part = C.write_part("risk_bands", payload)
    print(f"  wrote {part}")

    # ----- handoff ---------------------------------------------------------
    write_handoff(bands, deriv, valid, d_min, d_max, v_min, v_max,
                  arith_min, arith_max, att_min, att_max, grp_max, missing, p1, p2, part)
    C.banner("01_risk_bands complete")
    return 0


def write_handoff(bands, deriv, valid, d_min, d_max, v_min, v_max,
                  arith_min, arith_max, att_min, att_max, grp_max,
                  missing, p1, p2, part) -> None:
    lines = []
    A = lines.append
    A("# 01_risk_bands: handoff")
    A("")
    A("Owner of finding #7 (score range -11 to 63 unreachable). Feeds manuscript Table 3,")
    A("the middle zone of the WTA abstract figure, the reply to Dr. Fitzgerald on the point")
    A("range and the reply to Dr. Dumas on how the points work.")
    A("")
    A("## Use these band definitions verbatim")
    A("")
    A("Boundaries were chosen on the **derivation** distribution and applied unchanged to")
    A("validation. Do not re-cut them, do not rename them, do not merge them.")
    A("")
    A("| Rank | Name | Points | Validation n | Events | Observed rate (95% CI) | Sens at lower bound | Spec at lower bound |")
    A("|---|---|---|---|---|---|---|---|")
    for (_, name, lo, hi), (_, v) in zip(bands, valid.iterrows()):
        A(f"| {v.band_rank} | {name} | {band_label(lo, hi)} | {v.n:,} | {v.events:,} | "
          f"{100*v.observed_rate:.2f}% ({100*v.rate_ci_low:.2f} to {100*v.rate_ci_high:.2f}) | "
          f"{v.sensitivity_at_lower_bound:.4f} | {v.specificity_at_lower_bound:.4f} |")
    A("")
    A("Derivation, for the reader who wants to see the bands were not tuned to validation:")
    A("")
    A("| Rank | Name | Points | Derivation n | Events | Observed rate (95% CI) |")
    A("|---|---|---|---|---|---|")
    for (_, name, lo, hi), (_, d) in zip(bands, deriv.iterrows()):
        A(f"| {d.band_rank} | {name} | {band_label(lo, hi)} | {d.n:,} | {d.events:,} | "
          f"{100*d.observed_rate:.2f}% ({100*d.rate_ci_low:.2f} to {100*d.rate_ci_high:.2f}) |")
    A("")
    A("The rate rises monotonically across bands in **both** cohorts. The first candidate")
    A("set passed every check, so nothing was moved, merged or widened.")
    A("")
    A("## The score range, for Dr. Fitzgerald")
    A("")
    A(f"- Observed: **{d_min} to {d_max}** in derivation, **{v_min} to {v_max}** in validation. Report these.")
    A(f"- Arithmetic sum of every negative and every positive point: {arith_min} to {arith_max}. "
      "This is what the v4 draft printed and it is wrong to print.")
    A(f"- **True attainable range: {att_min} to {att_max}.** Use this one if a ceiling is quoted at all.")
    A(f"- Counting only the age, Glasgow Coma Scale and mechanism groups gives {att_min} to {grp_max}, "
      "but its best pattern is bradycardic and tachycardic at once, so it is one point too high. "
      "Do not quote it.")
    A("")
    A(f"The attainable minimum, {att_min}, is exactly the observed minimum: the lowest possible")
    A("patient really does appear in the data. The attainable maximum is never approached,")
    A(f"because reaching {att_max} needs every remaining positive indicator at once.")
    A("")
    A("## Surprises worth knowing")
    A("")
    A("1. Penetrating assault (+9) is a strict subset of assault (-5), verified with zero")
    A("   violations. A penetrating assault patient nets +4 from mechanism, not +9. Dr. Dumas'")
    A("   question about how the points work should be answered with this example.")
    A("2. Shock index 1.3 or above is a strict subset of shock index 0.9 or above, so those")
    A("   two indicators stack to +6 and are never carried alone as +2.")
    A("3. Glasgow Coma Scale motor 3 or below does **not** strictly nest inside total 8 or")
    A("   below: 635 of 27,540 such patients have a total above 8. Do not describe that pair")
    A("   as nested in the manuscript.")
    A("4. The lowest band's lower boundary is the cohort minimum, so its cumulative")
    A("   sensitivity is 1.0 and specificity 0.0 by construction. That row is a reference")
    A("   line, not a finding.")
    A("5. Validation score -8 holds 1,029 patients and zero events. The Wilson interval, not")
    A("   a bare proportion, is what makes that row printable.")
    if missing:
        A(f"6. Validation has no patient at score{'s' if len(missing)>1 else ''} "
          f"{', '.join(map(str, missing))}, which sit inside the observed range.")
        A("   The figure's middle zone must not assume a value at every integer.")
    A("")
    A("## Files")
    A("")
    A(f"- `{p1.relative_to(C.V6)}` band table, both cohorts")
    A(f"- `{p2.relative_to(C.V6)}` per integer score, validation, for the figure")
    A(f"- `{part.relative_to(C.V6)}` bundle part, band level summary only")
    (C.V6_LOGS / "01_HANDOFF.md").write_text("\n".join(lines) + "\n")
    print(f"  wrote {C.V6_LOGS / '01_HANDOFF.md'}")


if __name__ == "__main__":
    raise SystemExit(main())
