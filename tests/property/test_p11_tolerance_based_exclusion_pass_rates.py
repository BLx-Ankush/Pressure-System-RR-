# Feature: pressure-rating-system, Property 11: For any table of check results (any mix of applicable checks, pass flags and Runs_Check_Miss values) and any Score_Tolerance t ≥ 0: a match is excluded exactly when it fails the over check, fails the wickets-margin check, or has a runs-margin, chase or tie Runs_Check_Miss above t, so a match whose only failure is a miss of exactly t is kept and one with a miss of t + 1 is excluded; `excluded_matches` holds exactly one row per condition met, with the Runs_Check_Miss blank for the over and wickets-margin checks; each Within_Tolerance_Pass_Rate equals 100 × within-tolerance passes ÷ applicable matches, is at least the matching Exact_Pass_Rate, and is filled only on the selected encoding's row; the excluded count at each grid tolerance equals the number of distinct excluded matches at that tolerance and never rises as the tolerance rises; and at t = 0 the excluded set equals the set of matches in `failing_matches`.
"""Property 11: Tolerance-based exclusion and pass rates (Requirement 5, criteria 7, 21, 23, 24, 27 and 28).

**Validates: Requirements 5.21, 5.24, 5.27, 5.28**

Each generated example starts from ``tests.strategies.check_tables()``: the ``score_checks`` rows
of 1-6 matches under a drawn non-empty subset of the Config's candidate Extras_Encodings, with the
non-applicable rows that run_checks also emits, Runs_Check_Miss values clustered at 0, t and t + 1,
a selected encoding and a Score_Tolerance t drawn from 0-3 and from D20 - 1 to D20 + 1. The
example adds the candidates in a drawn order (the pass-rate table order, so the F30_Encoding is
not always listed first), the Config reporting grid plus t - 1, t and t + 1 in a drawn order, and
the ``match_id`` of 1-3 Deliveries per match in a drawn order.

Each result is compared with a direct transcription of its definition, computed from the rows in
plain Python:

* ``exact_pass_rates``: per candidate and check, 100 x passes / applicable matches, missing when
  the check applies to no match (glossary entry Exact_Pass_Rate);
* ``select_encoding``: the highest mean of the defined rates, with the means compared exactly as
  fractions; the F30_Encoding when it ties for the highest, otherwise the first listed tied
  candidate; StageError with fewer than 2 candidates, no F30_Encoding or no defined rate
  (criterion 7);
* ``within_tolerance_pass_rates`` of the selected encoding: an applicable runs-margin, chase or tie
  check passes within tolerance when its miss is at most t, a wickets-margin or over check only
  when it passes exactly, over the same denominator; no rate is below the check's Exact_Pass_Rate
  (glossary entry Within_Tolerance_Pass_Rate);
* ``pass_rate_table``: one row per candidate in table order with the exact rates, the
  within-tolerance rates on the selected row only and the ``selected`` flag (criterion 21);
* ``failing_matches``: exactly the failed applicable checks, sorted by match and check
  (criterion 23);
* ``exclusion_conditions`` at t: one row per condition of criterion 24 met (a failed over check, a
  failed wickets-margin check, a runs check with a miss above t), with the miss blank on over and
  wickets-margin rows. A match whose only failures are runs checks with misses of at most t is
  kept, and a match with a miss of t + 1 is excluded (criteria 24, 27 and 28);
* at a tolerance of 0, the excluded matches are exactly the matches in ``failing_matches``;
* ``exclusion_counts``: in grid order, the number of distinct excluded matches at each grid
  tolerance, never rising as the tolerance rises, with ``is_config_value`` true on the row of t
  only, whose count is the number of matches excluded at t (criterion 27);
* ``excluded_flags``: with ``exclude_failing_matches`` on, a Delivery is flagged exactly when its
  match is excluded; with it off, no Delivery is (criterion 24).

The explicit examples are built with ``score_checks_frame`` at t = D20, read from the Config. Each
puts rows under the F30_Encoding and one other candidate whose mean Exact_Pass_Rates tie, so the
F30_Encoding wins the tie and is the selected encoding, and gives the excluded matches by hand. The
first four put the same rows under both candidates:

* matches whose only failure is a runs-margin, chase or tie miss of exactly D20 are kept;
* matches whose only failure is a miss of D20 + 1 on one of those checks are excluded;
* a match that fails only the over check is excluded, next to a match that passes every check;
* a match that fails only the wickets-margin check is excluded, next to a match that passes every
  check.

The fifth gives the two candidates different rows whose means are equal as fractions but not as
floats. It has three wins by wickets, which fail the wickets-margin check, and three ties, one of
which fails the over check. The F30_Encoding passes 0/3 chase and 3/3 tie checks, and the other
candidate 1/3 and 2/3. Both means are 275/6, but they come out as 45.83333333333333 and
45.833333333333336 as floats. The F30_Encoding must still be selected (criterion 7).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from fractions import Fraction
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import pytest
from hypothesis import event, example, given, note
from hypothesis import strategies as st

from superr.errors import StageError
from superr.reconstruct import (
    F30_ENCODING,
    exact_pass_rates,
    excluded_flags,
    exclusion_conditions,
    exclusion_counts,
    failing_matches,
    pass_rate_table,
    select_encoding,
    within_tolerance_pass_rates,
)
from superr.settings import load_settings
from tests.strategies import (
    CANDIDATE_ENCODINGS,
    CHECKS,
    ROOT,
    RUNS_CHECKS,
    SCORE_CHECK_COLUMNS,
    SCORE_TOLERANCE,
    CheckTables,
    check_tables,
    score_checks_frame,
)

#: The Score_Tolerance D20 in runs, read from config.py (design_choices.D20_score_tolerance).
D20 = SCORE_TOLERANCE
#: The Config reporting grid data_rules.score_tolerance_grid (criterion 27).
CONFIG_GRID: tuple[int, ...] = tuple(load_settings(ROOT).get("data_rules.score_tolerance_grid"))

# Columns of the pass_rates, excluded_matches and exclusion_counts tables (design, Data Models)
# and of failing_matches.csv (criterion 23).
EXACT = {check: f"exact_{check}" for check in CHECKS}
WITHIN = {check: f"within_tol_{check}" for check in CHECKS}
RATE_COLUMNS = ["encoding", *EXACT.values()]
PASS_RATE_COLUMNS = [*RATE_COLUMNS, *WITHIN.values(), "selected"]
FAILING_COLUMNS = ["match_id", "check"]
CONDITION_COLUMNS = ["match_id", "check", "runs_check_miss"]
COUNT_COLUMNS = ["score_tolerance", "n_excluded_matches", "is_config_value"]

if F30_ENCODING not in CANDIDATE_ENCODINGS or len(CANDIDATE_ENCODINGS) < 2:
    raise RuntimeError(
        f"data_rules.candidate_encodings must list {F30_ENCODING} and another candidate (criterion 7); "
        f"got {CANDIDATE_ENCODINGS}"
    )
_OTHER = next(name for name in CANDIDATE_ENCODINGS if name != F30_ENCODING)
#: The candidates of the explicit examples: the F30_Encoding and one other, in Config order.
EXAMPLE_ORDER: tuple[str, ...] = tuple(name for name in CANDIDATE_ENCODINGS if name in (F30_ENCODING, _OTHER))


# --------------------------------------------------------------------------------- examples

class Row(NamedTuple):
    """One score_checks row as plain Python values, None for a missing value."""

    match_id: int
    check: str
    applies: bool
    passed: bool | None
    miss: int | None


class Case:
    """One example: the check rows, the selected encoding and t, the pass-rate table order, the grid
    passed to exclusion_counts, the match_id of each Delivery and, for an explicit example, the
    excluded matches worked out by hand."""

    __slots__ = ("checks", "selected", "tolerance", "order", "grid", "deliveries", "hand_excluded")

    def __init__(
        self,
        tables: CheckTables,
        order: Sequence[str],
        grid: Sequence[int],
        deliveries: pd.Series,
        hand_excluded: Iterable[int] | None = None,
    ) -> None:
        self.checks, self.selected, self.tolerance = tables
        self.order, self.grid, self.deliveries = tuple(order), tuple(grid), deliveries
        self.hand_excluded = None if hand_excluded is None else frozenset(hand_excluded)

    def __repr__(self) -> str:
        hand = "not given" if self.hand_excluded is None else sorted(self.hand_excluded)
        return (
            f"selected {self.selected}, t = {self.tolerance}, table order {list(self.order)}, "
            f"grid {list(self.grid)}, excluded by hand {hand}\n"
            f"Delivery match_ids {self.deliveries.tolist()}\n{self.checks.to_string()}"
        )


def boundary_grid(t: int) -> list[int]:
    """The Config reporting grid plus t - 1 (for t > 0), t and t + 1, in ascending order."""
    return sorted({*CONFIG_GRID, t, t + 1, *([t - 1] if t > 0 else [])})


def delivery_ids(match_ids: Sequence[int]) -> pd.Series:
    """A Delivery table's match_id column (Int64, as the Data_Loader types it) on an index from 1000."""
    return pd.Series(
        pd.array(list(match_ids), dtype="Int64"), index=pd.RangeIndex(1000, 1000 + len(match_ids)), name="match_id"
    )


@st.composite
def cases(draw: Any) -> Case:
    """A check_tables() draw plus a drawn candidate order, grid order and list of Deliveries."""
    tables = draw(check_tables())
    names = tables.checks["encoding"].drop_duplicates().tolist()
    ids = sorted(set(tables.checks["match_id"].tolist()))
    per_match = [draw(st.integers(1, 3)) for _ in ids]
    deliveries = [match_id for match_id, n in zip(ids, per_match) for _ in range(n)]
    return Case(
        tables,
        draw(st.permutations(names)),
        draw(st.permutations(boundary_grid(tables.tolerance))),
        delivery_ids(draw(st.permutations(deliveries))),
    )


def _match(
    match_id: int, *, misses: Mapping[str, int] | None = None, passes: Mapping[str, bool] | None = None
) -> list[tuple[Any, ...]]:
    """The five score_checks rows of one match, without the encoding. A runs check in ``misses``
    applies with that Runs_Check_Miss and passes exactly when it is 0; a check in ``passes`` applies
    with that result; no other check applies."""
    misses, passes = misses or {}, passes or {}
    rows: list[tuple[Any, ...]] = []
    for check in CHECKS:
        if check in misses:
            rows.append((match_id, check, True, misses[check] == 0, misses[check]))
        elif check in passes:
            rows.append((match_id, check, True, passes[check], None))
        else:
            rows.append((match_id, check, False, None, None))
    return rows


def _explicit(
    matches: Sequence[list[tuple[Any, ...]]],
    excluded: Iterable[int],
    other: Sequence[list[tuple[Any, ...]]] | None = None,
) -> Case:
    """An explicit example at t = D20: the rows of ``matches`` under the F30_Encoding and those of
    ``other`` (by default the same rows) under the other EXAMPLE_ORDER candidate, the F30_Encoding
    selected, the boundary grid in ascending order, 2 Deliveries per match and the excluded
    matches given by hand."""
    by_name = {F30_ENCODING: matches, _OTHER: matches if other is None else other}
    rows = [(name, *row) for name in EXAMPLE_ORDER for match in by_name[name] for row in match]
    ids = sorted({row[0] for match in matches for row in match})
    return Case(
        CheckTables(score_checks_frame(rows), F30_ENCODING, D20),
        EXAMPLE_ORDER,
        boundary_grid(D20),
        delivery_ids([match_id for match_id in ids for _ in range(2)]),
        excluded,
    )


KEPT_AT_D20 = _explicit(
    [
        _match(101, misses={"runs_margin": D20}, passes={"over": True}),
        _match(102, misses={"chase": D20}, passes={"wickets_margin": True, "over": True}),
        _match(103, misses={"tie": D20}, passes={"over": True}),
    ],
    excluded=[],
)
EXCLUDED_AT_D20_PLUS_1 = _explicit(
    [
        _match(201, misses={"runs_margin": D20 + 1}, passes={"over": True}),
        _match(202, misses={"chase": D20 + 1}, passes={"wickets_margin": True, "over": True}),
        _match(203, misses={"tie": D20 + 1}, passes={"over": True}),
    ],
    excluded=[201, 202, 203],
)
OVER_ONLY = _explicit(
    [
        _match(301, misses={"runs_margin": 0}, passes={"over": False}),
        _match(302, misses={"runs_margin": 0}, passes={"over": True}),
    ],
    excluded=[301],
)
WICKETS_MARGIN_ONLY = _explicit(
    [
        _match(401, misses={"chase": 0}, passes={"wickets_margin": False, "over": True}),
        _match(402, misses={"chase": 0}, passes={"wickets_margin": True, "over": True}),
    ],
    excluded=[401],
)
# Means equal as fractions but not as floats (criterion 7). No runs-margin check applies, so each
# mean covers the chase, wickets-margin, tie and over rates. The F30_Encoding passes 0/3 chases and
# 3/3 ties; the other candidate passes 1/3 and 2/3. Both means are 275/6, but math.fsum / 4 gives
# 45.83333333333333 for the F30_Encoding and 45.833333333333336 for the other candidate. The
# F30_Encoding ties for the highest mean, so it is selected.
FRACTION_TIE = _explicit(
    [
        _match(501, misses={"chase": 1}, passes={"wickets_margin": False, "over": True}),
        _match(502, misses={"chase": 1}, passes={"wickets_margin": False, "over": True}),
        _match(503, misses={"chase": 1}, passes={"wickets_margin": False, "over": True}),
        _match(504, misses={"tie": 0}, passes={"over": True}),
        _match(505, misses={"tie": 0}, passes={"over": True}),
        _match(506, misses={"tie": 0}, passes={"over": False}),
    ],
    excluded=[501, 502, 503, 506],
    other=[
        _match(501, misses={"chase": 0}, passes={"wickets_margin": False, "over": True}),
        _match(502, misses={"chase": 1}, passes={"wickets_margin": False, "over": True}),
        _match(503, misses={"chase": 1}, passes={"wickets_margin": False, "over": True}),
        _match(504, misses={"tie": 0}, passes={"over": True}),
        _match(505, misses={"tie": 1}, passes={"over": True}),
        _match(506, misses={"tie": 0}, passes={"over": False}),
    ],
)


# ------------------------------------------------------------------------ direct definitions

def _rows(checks: pd.DataFrame) -> dict[str, list[Row]]:
    """The rows of each encoding, in table order, as plain Python values."""
    rows: dict[str, list[Row]] = {}
    for encoding, match_id, check, applies, passed, miss in zip(*(checks[name].tolist() for name in SCORE_CHECK_COLUMNS)):
        rows.setdefault(str(encoding), []).append(
            Row(
                int(match_id),
                str(check),
                bool(applies),
                None if pd.isna(passed) else bool(passed),
                None if pd.isna(miss) else int(miss),
            )
        )
    return rows


def _rate(rows: Sequence[Row], check: str, passes: Callable[[Row], bool]) -> Fraction | None:
    """100 x (applicable rows of ``check`` that pass) / (applicable rows of ``check``); None when none applies."""
    applicable = [row for row in rows if row.check == check and row.applies]
    return Fraction(100 * sum(map(passes, applicable)), len(applicable)) if applicable else None


def _exact(row: Row) -> bool:
    """Passes exactly (glossary entry Exact_Pass_Rate)."""
    return bool(row.passed)


def _within(t: int) -> Callable[[Row], bool]:
    """Passes within the Score_Tolerance t (glossary entry Within_Tolerance_Pass_Rate)."""
    return lambda row: row.miss <= t if row.check in RUNS_CHECKS else bool(row.passed)


def _is_rate(value: float, expected: Fraction | None) -> bool:
    """Whether a rate of a table equals the reference: missing for None, else equal to within 1e-12."""
    if expected is None:
        return math.isnan(value)
    return math.isclose(value, float(expected), rel_tol=1e-12, abs_tol=1e-12)


def _selection(order: Sequence[str], rows: Mapping[str, Sequence[Row]]) -> tuple[str | None, str]:
    """Criterion 7 with exact means: the candidate to select (None where select_encoding must raise
    StageError) and the rule that decides it."""
    if len(order) < 2 or F30_ENCODING not in order:
        return None, "StageError (fewer than 2 candidates or no F30_Encoding)"
    means: dict[str, Fraction] = {}
    for name in order:
        defined = [rate for rate in (_rate(rows[name], check, _exact) for check in CHECKS) if rate is not None]
        if defined:
            means[name] = sum(defined, Fraction(0)) / len(defined)
    if not means:
        return None, "StageError (no Exact_Pass_Rate is defined)"
    best = max(means.values())
    tied = [name for name in order if means.get(name) == best]
    if len(tied) == 1:
        return tied[0], "a unique highest mean"
    if F30_ENCODING in tied:
        return F30_ENCODING, "the F30_Encoding on a tie"
    return tied[0], "the first listed candidate on a tie"


def _table_order(item: tuple[Any, ...]) -> tuple[int, int]:
    """Sort key of a (match_id, check, ...) tuple: match_id, then check in table order."""
    return item[0], CHECKS.index(item[1])


def _conditions(rows: Sequence[Row], t: int) -> list[tuple[int, str, int | None]]:
    """Criterion 24 at t: (match_id, check, Runs_Check_Miss or None) of each condition met, in table order."""
    met = [
        (row.match_id, row.check, row.miss if row.check in RUNS_CHECKS else None)
        for row in rows
        if row.applies and (row.miss > t if row.check in RUNS_CHECKS else not row.passed)
    ]
    return sorted(met, key=_table_order)


def _records(frame: pd.DataFrame, columns: Sequence[str]) -> list[tuple[Any, ...]]:
    """The rows of ``columns`` as tuples of Python values, None for a missing value."""
    return [
        tuple(None if pd.isna(value) else value for value in row)
        for row in zip(*(frame[name].tolist() for name in columns))
    ]


class _Table:
    """A DataFrame for hypothesis.note, formatted only when Hypothesis prints the note."""

    def __init__(self, title: str, frame: pd.DataFrame) -> None:
        self.title, self.frame = title, frame

    def __repr__(self) -> str:
        return f"{self.title}:\n{self.frame.to_string()}"


# ------------------------------------------------------------------------------------- test

@given(case=cases())
@example(case=KEPT_AT_D20)
@example(case=EXCLUDED_AT_D20_PLUS_1)
@example(case=OVER_ONLY)
@example(case=WICKETS_MARGIN_ONLY)
@example(case=FRACTION_TIE)
def test_tolerance_based_exclusion_and_pass_rates(case: Case) -> None:
    """Pass rates, the selected encoding, failed checks and match exclusion follow Requirement 5,
    criteria 7, 21, 23, 24 and 27, on both sides of the Score_Tolerance boundary.

    **Validates: Requirements 5.21, 5.24, 5.27, 5.28**
    """
    t, selected = case.tolerance, case.selected
    rows = _rows(case.checks)
    chosen_rows = rows[selected]
    if t == D20:
        event("t = D20")

    # Exact_Pass_Rates of every candidate, in table order (criteria 7 and 21).
    by_encoding = {name: case.checks[case.checks["encoding"] == name].reset_index(drop=True) for name in case.order}
    rates = exact_pass_rates(by_encoding)
    note(_Table("exact_pass_rates", rates))
    assert rates.columns.tolist() == RATE_COLUMNS
    assert rates["encoding"].tolist() == list(case.order)
    for i, name in enumerate(case.order):
        for check in CHECKS:
            value, expected = rates.at[i, EXACT[check]], _rate(rows[name], check, _exact)
            assert _is_rate(value, expected), f"{name} {check}: Exact_Pass_Rate {value}, expected {expected}"

    # The selection rule (criterion 7).
    choice, rule = _selection(case.order, rows)
    event(f"select_encoding: {rule}")
    if choice is None:
        with pytest.raises(StageError):
            select_encoding(rates)
    else:
        assert select_encoding(rates) == choice, f"criterion 7 selects {choice} by {rule}"

    # Within_Tolerance_Pass_Rates of the selected encoding, each at least its Exact_Pass_Rate.
    chosen = by_encoding[selected]
    within = within_tolerance_pass_rates(chosen, t)
    assert within.index.tolist() == list(CHECKS)
    exact_of_selected = rates.loc[rates["encoding"] == selected, list(EXACT.values())].iloc[0]
    for check in CHECKS:
        value, expected = within[check], _rate(chosen_rows, check, _within(t))
        assert _is_rate(value, expected), f"{check}: Within_Tolerance_Pass_Rate {value}, expected {expected}"
        exact = exact_of_selected[EXACT[check]]
        assert math.isnan(value) == math.isnan(exact)
        assert math.isnan(exact) or value >= exact, f"{check}: Within_Tolerance_Pass_Rate {value} < {exact}"

    # The pass-rate table (criterion 21): within-tolerance rates on the selected row only.
    table = pass_rate_table(rates, selected, within)
    assert table.columns.tolist() == PASS_RATE_COLUMNS
    assert table["encoding"].tolist() == list(case.order)
    on_row = np.array([name == selected for name in case.order])
    assert table["selected"].tolist() == on_row.tolist()
    for check in CHECKS:
        exact_column = table[EXACT[check]].to_numpy(dtype=float)
        assert np.array_equal(exact_column, rates[EXACT[check]].to_numpy(dtype=float), equal_nan=True)
        within_column = table[WITHIN[check]].to_numpy(dtype=float)
        assert np.isnan(within_column[~on_row]).all(), f"{WITHIN[check]} is filled on a row that is not selected"
        assert np.array_equal(within_column[on_row], [within[check]], equal_nan=True)

    # failing_matches (criterion 23): every failed applicable check, misses within t included.
    failing = failing_matches(chosen)
    assert failing.columns.tolist() == FAILING_COLUMNS
    failed = sorted(((row.match_id, row.check) for row in chosen_rows if row.applies and not row.passed), key=_table_order)
    assert _records(failing, FAILING_COLUMNS) == failed

    # exclusion_conditions at t (criteria 24 and 27): one row per condition met.
    conditions = exclusion_conditions(chosen, t)
    note(_Table(f"exclusion_conditions at t = {t}", conditions))
    assert conditions.columns.tolist() == CONDITION_COLUMNS
    expected_conditions = _conditions(chosen_rows, t)
    assert _records(conditions, CONDITION_COLUMNS) == expected_conditions
    excluded = set(conditions["match_id"].tolist())
    assert excluded == {match_id for match_id, _, _ in expected_conditions}
    if case.hand_excluded is not None:
        assert excluded == case.hand_excluded, f"excluded {sorted(excluded)}; by hand {sorted(case.hand_excluded)}"

    # The keep/exclude boundary (criteria 24 and 28).
    for match_id in sorted({row.match_id for row in chosen_rows}):
        applicable = [row for row in chosen_rows if row.match_id == match_id and row.applies]
        failures = [row for row in applicable if not row.passed]
        runs_only = bool(failures) and all(row.check in RUNS_CHECKS for row in failures)
        if runs_only and all(row.miss <= t for row in failures):
            assert match_id not in excluded, f"match {match_id} fails only runs checks within t = {t} but is excluded"
            if any(row.miss == t for row in failures):
                event("a match whose only failure is a miss of exactly t is kept")
        if any(row.check in RUNS_CHECKS and row.miss == t + 1 for row in applicable):
            assert match_id in excluded, f"match {match_id} has a Runs_Check_Miss of t + 1 = {t + 1} but is kept"
            if runs_only and len(failures) == 1:
                event("a match whose only failure is a miss of t + 1 is excluded")

    # At a tolerance of 0 the excluded matches are the matches in failing_matches.
    at_zero = set(exclusion_conditions(chosen, 0)["match_id"].tolist())
    assert at_zero == set(failing["match_id"].tolist())

    # exclusion_counts over a grid that includes t (criterion 27).
    counts = exclusion_counts(chosen, list(case.grid), tolerance=t)
    note(_Table("exclusion_counts", counts))
    assert counts.columns.tolist() == COUNT_COLUMNS
    assert counts["score_tolerance"].tolist() == list(case.grid)
    n_excluded = counts["n_excluded_matches"].tolist()
    assert n_excluded == [len({match_id for match_id, _, _ in _conditions(chosen_rows, g)}) for g in case.grid]
    assert counts["is_config_value"].tolist() == [g == t for g in case.grid]
    by_tolerance = dict(zip(case.grid, n_excluded))
    rising = sorted(by_tolerance)
    assert all(by_tolerance[low] >= by_tolerance[high] for low, high in zip(rising, rising[1:])), by_tolerance
    assert by_tolerance[t] == len(excluded)
    if 0 in by_tolerance:
        assert by_tolerance[0] == len(at_zero)

    # excluded_flags (criterion 24): every Delivery of an excluded match, only where the option is on.
    flags = excluded_flags(case.deliveries, conditions, exclude_failing_matches=True)
    assert flags.name == "excluded" and flags.index.equals(case.deliveries.index)
    assert flags.tolist() == [match_id in excluded for match_id in case.deliveries.tolist()]
    off = excluded_flags(case.deliveries, conditions, exclude_failing_matches=False)
    assert off.index.equals(case.deliveries.index) and not off.any()
