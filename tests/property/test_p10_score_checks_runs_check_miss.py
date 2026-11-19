# Feature: pressure-rating-system, Property 10: For any synthetic match generated from a scorecard (Scheduled_Balls, innings totals, wickets and result) by emitting Deliveries under a given encoding, every applicable check of Requirement 5 criteria 16–20 passes under that encoding, with a Runs_Check_Miss of 0 for the runs-margin, chase and tie checks. For any such match with Delivery runs changed so that a total moves by any number of runs (including 0, the Score_Tolerance and the Score_Tolerance + 1), the runs-margin, chase and tie checks return the Runs_Check_Miss of a direct transcription of criteria 16, 17 and 19, with chase cases driven by the shortfall and by the excess; each of those checks passes exactly when its Runs_Check_Miss is 0; every check that depends on a changed total fails when the change is non-zero; and the match appears in `failing_matches` once per failed check.
"""Property 10: Score checks and Runs_Check_Miss (Requirement 5, criteria 16-20, 23 and 28).

**Validates: Requirements 5.16, 5.17, 5.18, 5.19, 5.20, 5.23, 5.28**

The test is parametrised over the miss that the matches of an example must show: 0, the
Score_Tolerance t (D20, read from ``config.py``) and t + 1. Every run therefore gives the
runs-margin, chase and tie checks Runs_Check_Miss values of exactly 0, D20 and D20 + 1, the two
sides of the keep/exclude boundary of criterion 24 (criterion 28). Each example draws one
candidate Extras_Encoding of the Config and builds all of its matches under it with
``tests.strategies.scorecard_match`` (up to 3 Overs per innings), with distinct ``match_id``
values:

* miss 0: a win by runs, a win by wickets and a tie or super-over win, each as generated (no
  shift, or a shift of 0) and with no bad Over, so every applicable check must pass;
* miss t and miss t + 1: a win by runs and a tie (at t) or super-over win (at t + 1) whose
  second-innings total moves up or down by the miss, a chase whose total moves down by the miss
  (driven by the shortfall) and a chase whose total moves up by it (driven by the excess). One of
  these four matches always gets a bad Over, and each of the others a third of the time;
* in every example, one more match of any of the four result types, with no shift or a shift of
  -3(t + 1) to 3(t + 1) runs on either innings, and a bad Over a third of the time.

The Deliveries go through the Score_Reconstructor functions that build the inputs of run_checks
(classify, encoding_inputs, candidate_team_runs, wicket_flags with the Config's D14 codes and the
Counted_Wicket codes 1-6 and 9, innings_aggregates, parse_results and over_check), then
run_checks and failing_matches. The test checks that:

* each match has exactly its applicable checks: the runs-margin check for a win by runs (whose
  winner batted first), the chase and wickets-margin checks for a win by wickets, the tie check
  for a tie or super-over win, and the over check for every match. Only the applicable runs check
  has a Runs_Check_Miss;
* that Runs_Check_Miss equals a direct transcription of criterion 16, 17 or 19 from the scorecard
  (Target, second-innings total, final-ball runs and recorded margin). For the chase it is the
  larger of the shortfall and the excess, each floored at 0;
* each runs check passes exactly when its Runs_Check_Miss is 0;
* the Runs_Check_Miss is |shift| when the shift moved a total that the check depends on (the
  second-innings total, or the innings-1 total when there is no ``revised_target``, because the
  Target is then that total + 1), and 0 otherwise. Each match of the miss cases misses by exactly
  that miss, and the two chases are driven by the shortfall and by the excess as intended;
* the wickets-margin check passes exactly when the margin equals 10 minus the second-innings
  wickets, and the over check fails exactly for the matches with a bad Over;
* so the failed checks of each match are its moved runs check and, with a bad Over, the over
  check, and a match with no moved total and no bad Over passes every applicable check;
* the run_checks table equals the strategy's ``expected`` tables (its own transcription of
  criteria 16-20) in rows, values and dtypes: five rows per match, sorted by ``match_id`` and check;
* ``failing_matches`` lists exactly those failed checks, one row per failed check, sorted by
  ``match_id`` and then check (criterion 23).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import pandas as pd
import pytest
from hypothesis import event, given, note
from hypothesis import strategies as st

from superr.reconstruct import (
    SUPER_OVER_WIN,
    TIE,
    WIN_BY_RUNS,
    WIN_BY_WICKETS,
    candidate_team_runs,
    classify,
    encoding_inputs,
    failing_matches,
    innings_aggregates,
    over_check,
    parse_results,
    run_checks,
    wicket_flags,
)
from tests.strategies import (
    CANDIDATE_ENCODINGS,
    CHECKS,
    COUNTED_WICKET_CODES,
    EXTRAS_TYPE_CLASSES,
    RUNS_CHECKS,
    SCORE_TOLERANCE,
    MatchCase,
    scorecard_match,
)

if TYPE_CHECKING:
    from superr.settings import Settings

#: (miss, result type of the tie-check match) of each parametrised case. The miss-0 case draws a tie
#: or a super-over win in each example; the other two cases fix one each, so both always occur.
BOUNDARIES = [
    pytest.param(0, None, id="miss_0"),
    pytest.param(SCORE_TOLERANCE, TIE, id="miss_D20"),
    pytest.param(SCORE_TOLERANCE + 1, SUPER_OVER_WIN, id="miss_D20_plus_1"),
]
#: The runs check (criterion 16, 17 or 19) that applies to each result type scorecard_match builds.
RUNS_CHECK_OF: dict[str, str] = {
    WIN_BY_RUNS: "runs_margin",
    WIN_BY_WICKETS: "chase",
    TIE: "tie",
    SUPER_OVER_WIN: "tie",
}
if set(RUNS_CHECK_OF.values()) != set(RUNS_CHECKS):
    raise RuntimeError(f"Property 10 maps result types to {sorted(set(RUNS_CHECK_OF.values()))}, not {RUNS_CHECKS}")
RESULT_TYPES = tuple(RUNS_CHECK_OF)
TIE_TYPES = (TIE, SUPER_OVER_WIN)
#: Largest |shift| of the random match: three times t + 1, the smallest miss that criterion 24 excludes.
MAX_RANDOM_SHIFT = 3 * (SCORE_TOLERANCE + 1)
MATCH_IDS = st.integers(10_000, 99_999)
#: bad_over draws: a bad Over a third of the time.
SOMETIMES_BAD = (False, False, True)
#: Laws of Cricket: a chase won by w wickets ends with 10 - w wickets fallen (criterion 18).
WICKETS_PER_INNINGS = 10
FAILING_COLUMNS = ["match_id", "check"]


# --------------------------------------------------------------------------------- strategy

class Slot(NamedTuple):
    """What one match of an example is asked to be, and the Runs_Check_Miss it must show."""

    label: str
    result_type: str
    shift: int | None  # passed to scorecard_match; None for no shift
    shift_innings: int | None  # 2 for the matches of the miss cases; None lets scorecard_match draw it
    bad_over: bool
    miss: int | None  # the Runs_Check_Miss the match must show; None for the random match
    driven_by: str | None = None  # "shortfall" or "excess" for the two chases of a miss case


class Generated(NamedTuple):
    """A match that scorecard_match built for a Slot."""

    slot: Slot
    case: MatchCase


def _match_id(case: MatchCase) -> int:
    return int(case.matches["match_id"].iloc[0])


def _describe(generated: Generated) -> str:
    """One line on a generated match, for assertion messages and falsifying examples."""
    slot, case = generated
    card, row = case.scorecard, case.matches.iloc[0]
    revised = "blank" if pd.isna(row["revised_target"]) else row["revised_target"]
    return (
        f"{slot.label}: match {_match_id(case)}, {card.result_type}, margin {card.margin}, totals {card.totals}, "
        f"wickets {card.wickets}, final-ball runs {card.final_ball_runs}, Target {card.target} "
        f"(revised_target {revised}), shift {card.shift} on innings {card.shift_innings}, "
        f"bad Over {slot.bad_over}, result {row['result']!r}"
    )


class Example:
    """The matches of one example, all built under one Extras_Encoding, shown one line per match."""

    __slots__ = ("encoding", "matches")

    def __init__(self, encoding: str, matches: Sequence[Generated]) -> None:
        self.encoding, self.matches = encoding, tuple(matches)

    def __repr__(self) -> str:
        return "\n".join([f"encoding {self.encoding}", *map(_describe, self.matches)])


def _miss_slots(draw: Any, miss: int, tie_type: str | None) -> list[Slot]:
    """The matches that must show ``miss``: one per runs check, and both chase drivers when ``miss`` > 0."""
    tie = tie_type if tie_type is not None else draw(st.sampled_from(TIE_TYPES))
    if miss == 0:
        return [
            Slot(f"{check} as generated", result_type, draw(st.sampled_from((None, 0))), None, False, 0)
            for check, result_type in (("runs_margin", WIN_BY_RUNS), ("chase", WIN_BY_WICKETS), ("tie", tie))
        ]
    plans = [  # (label, result type, shift of the second-innings total, chase driver)
        ("runs_margin", WIN_BY_RUNS, draw(st.sampled_from((miss, -miss))), None),
        ("chase by shortfall", WIN_BY_WICKETS, -miss, "shortfall"),
        ("chase by excess", WIN_BY_WICKETS, miss, "excess"),
        ("tie", tie, draw(st.sampled_from((miss, -miss))), None),
    ]
    forced = draw(st.integers(0, len(plans) - 1))  # this match always gets a bad Over
    return [
        Slot(f"{label} miss {miss}", result_type, shift, 2, i == forced or draw(st.sampled_from(SOMETIMES_BAD)),
             miss, driven_by)
        for i, (label, result_type, shift, driven_by) in enumerate(plans)
    ]


def _random_slot(draw: Any) -> Slot:
    shift = draw(st.one_of(st.none(), st.integers(-MAX_RANDOM_SHIFT, MAX_RANDOM_SHIFT)))
    return Slot("random", draw(st.sampled_from(RESULT_TYPES)), shift, None, draw(st.sampled_from(SOMETIMES_BAD)), None)


@st.composite
def examples(draw: Any, miss: int, tie_type: str | None) -> Example:
    """One example: the matches that must show ``miss`` plus the random match, under one drawn encoding."""
    encoding = draw(st.sampled_from(CANDIDATE_ENCODINGS))
    slots = [*_miss_slots(draw, miss, tie_type), _random_slot(draw)]
    ids = draw(st.lists(MATCH_IDS, min_size=len(slots), max_size=len(slots), unique=True))
    return Example(
        encoding,
        [
            Generated(
                slot,
                draw(scorecard_match(encoding=encoding, shift=slot.shift, shift_innings=slot.shift_innings,
                                     result_type=slot.result_type, bad_over=slot.bad_over, match_id=match_id)),
            )
            for slot, match_id in zip(slots, ids)
        ],
    )


# ---------------------------------------------------------------------------------- helpers

class _Table:
    """A DataFrame for hypothesis.note, formatted only when Hypothesis prints the note."""

    def __init__(self, title: str, frame: pd.DataFrame) -> None:
        self.title, self.frame = title, frame

    def __repr__(self) -> str:
        return f"{self.title}:\n{self.frame.to_string()}"


def _score_checks(balls: pd.DataFrame, matches: pd.DataFrame, encoding: str, d14: tuple[Any, Any, Any]) -> pd.DataFrame:
    """run_checks of ``matches`` under ``encoding``, from inputs built from the Deliveries as in a Run."""
    cls = classify(balls["extras_type"], EXTRAS_TYPE_CLASSES)
    team_runs = candidate_team_runs(encoding_inputs(balls, cls), (encoding,))[encoding]
    counted = wicket_flags(balls, COUNTED_WICKET_CODES, *d14)["counted_wicket"]
    agg = innings_aggregates(balls, team_runs, counted)
    return run_checks(matches, agg, parse_results(matches), over_check(balls, cls), encoding=encoding)


def _rows(checks: pd.DataFrame) -> dict[tuple[int, str], tuple[bool, bool | None, int | None]]:
    """(applies, passed, runs_check_miss) of each (match_id, check) row, with None for a missing value."""
    names = ("match_id", "check", "applies", "passed", "runs_check_miss")
    found: dict[tuple[int, str], tuple[bool, bool | None, int | None]] = {}
    for match_id, check, applies, passed, miss in zip(*(checks[name].tolist() for name in names)):
        found[int(match_id), str(check)] = (
            bool(applies),
            None if pd.isna(passed) else bool(passed),
            None if pd.isna(miss) else int(miss),
        )
    assert len(found) == len(checks), "run_checks gives a match more than one row for a check"
    return found


def _failed_checks(generated: Generated, rows: dict[tuple[int, str], tuple[bool, bool | None, int | None]]) -> list[str]:
    """Check one match's rows against criteria 16-20 directly; return the checks it fails, in table order."""
    slot, case = generated
    card, where = case.scorecard, _describe(generated)
    got = {check: rows.get((_match_id(case), check)) for check in CHECKS}
    assert None not in got.values(), f"{where}: run_checks lacks a row for a check"

    # Applicable checks (criteria 16-20); only the applicable runs check has a Runs_Check_Miss.
    runs_check = RUNS_CHECK_OF[card.result_type]
    applicable = {runs_check, "over"} | ({"wickets_margin"} if card.result_type == WIN_BY_WICKETS else set())
    assert {check for check, (applies, _, _) in got.items() if applies} == applicable, f"{where}: applicable checks"
    for check, (applies, passed, miss) in got.items():
        assert (passed is not None) == applies, f"{where}: {check} passed is {passed} where applies is {applies}"
        assert (miss is not None) == (check == runs_check), f"{where}: {check} Runs_Check_Miss is {miss}"

    # Runs_Check_Miss: a direct transcription of criterion 16, 17 or 19 (T = Target, S2 = second-innings
    # total, f = team runs of its final Delivery).
    target, total, final = card.target, card.totals[1], card.final_ball_runs
    shortfall, excess = target - total, (total - final) - (target - 1)
    if runs_check == "runs_margin":
        reference = abs(((target - 1) - total) - card.margin)
    elif runs_check == "chase":
        reference = max(max(shortfall, 0), max(excess, 0))
    else:
        reference = abs(total - (target - 1))
    _, passed, miss = got[runs_check]
    assert miss == reference, f"{where}: {runs_check} Runs_Check_Miss {miss}; criteria 16-19 give {reference}"
    assert passed == (miss == 0), f"{where}: {runs_check} passed is {passed} with Runs_Check_Miss {miss}"

    # The shift: the miss is |shift| exactly when it moved a total that the check depends on. With a
    # revised_target the Target does not depend on the innings-1 total (criterion 9).
    revised = pd.notna(case.matches["revised_target"].iloc[0])
    moved = card.shift != 0 and (card.shift_innings == 2 or not revised)
    assert miss == (abs(card.shift) if moved else 0), f"{where}: {runs_check} Runs_Check_Miss {miss}"
    if slot.miss is not None:
        assert miss == slot.miss, f"{where}: {runs_check} Runs_Check_Miss {miss}, not the case's miss {slot.miss}"
    if slot.driven_by is not None and slot.miss:
        drivers = {"shortfall": (slot.miss, 0), "excess": (0, slot.miss)}
        assert (max(shortfall, 0), max(excess, 0)) == drivers[slot.driven_by], (
            f"{where}: the chase miss is not driven by the {slot.driven_by} (shortfall {shortfall}, excess {excess})"
        )

    # Wickets-margin (criterion 18) and over (criterion 20) checks, which no shift changes.
    failed = [runs_check] if moved else []
    if card.result_type == WIN_BY_WICKETS:
        margin_ok = card.margin == WICKETS_PER_INNINGS - card.wickets[1]
        assert got["wickets_margin"][1] == margin_ok, f"{where}: wickets-margin check"
        failed += [] if margin_ok else ["wickets_margin"]
    assert got["over"][1] == (not slot.bad_over), f"{where}: the over check passed is {got['over'][1]}"
    failed += ["over"] if slot.bad_over else []
    failed.sort(key=CHECKS.index)
    assert [check for check in CHECKS if got[check][0] and not got[check][1]] == failed, f"{where}: failed checks"
    return failed


# ------------------------------------------------------------------------------------- test

@pytest.mark.parametrize(("miss", "tie_type"), BOUNDARIES)
@given(data=st.data())
def test_score_checks_and_runs_check_miss(miss: int, tie_type: str | None, data: st.DataObject, settings: Settings) -> None:
    """Score_Checks and Runs_Check_Miss follow Requirement 5, criteria 16-20, and failing_matches criterion 23.

    **Validates: Requirements 5.16, 5.17, 5.18, 5.19, 5.20, 5.23, 5.28**
    """
    example = data.draw(examples(miss, tie_type), label="example")
    event(f"encoding {example.encoding}")
    d14 = (
        settings.get("design_choices.D14_bowler_wicket_codes"),
        settings.get("design_choices.D14_run_out_code"),
        settings.get("design_choices.D14_retire_out_code"),
    )
    ordered = sorted(example.matches, key=lambda generated: _match_id(generated.case))
    balls = pd.concat([generated.case.balls for generated in example.matches], ignore_index=True)
    matches = pd.concat([generated.case.matches for generated in example.matches], ignore_index=True)
    checks = _score_checks(balls, matches, example.encoding, d14)
    note(_Table("run_checks", checks))

    # Each match against criteria 16-20 directly.
    rows = _rows(checks)
    assert len(rows) == len(CHECKS) * len(ordered), f"run_checks gives {len(rows)} rows for {len(ordered)} matches"
    failures = [(_match_id(generated.case), check) for generated in ordered for check in _failed_checks(generated, rows)]

    # The whole table equals the strategy's transcription, schema included.
    expected = pd.concat([generated.case.expected for generated in ordered], ignore_index=True)
    pd.testing.assert_frame_equal(checks, expected)

    # failing_matches (criterion 23): one row per failed applicable check, sorted by match_id and check.
    failing = failing_matches(checks)
    assert failing.columns.tolist() == FAILING_COLUMNS
    assert list(zip(failing["match_id"].tolist(), failing["check"].tolist())) == failures
