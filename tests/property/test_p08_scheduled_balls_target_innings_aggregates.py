# Feature: pressure-rating-system, Property 8: For any match with any combination of present and missing `first_inn_revised_overs`, `second_inn_revised_overs` and `revised_target`, Scheduled_Balls and Target follow Requirement 5 criteria 4–6 and 9–10; and for any innings under any candidate encoding, the innings total equals the sum of per-Delivery team runs under that encoding and the wickets fallen equals the number of rows with `is_wicket` true and a Counted_Wicket code.
"""Property 8: Scheduled_Balls, Target and innings aggregates (Requirement 5, criteria 4-6 and 8-10).

**Validates: Requirements 5.4, 5.5, 5.6, 5.8, 5.9, 5.10**

Each example holds 1-5 matches with distinct ``match_id`` values. The first 1-3 are played:
``tests.strategies.scorecard_match`` builds each from a scorecard under one Extras_Encoding, with
up to 3 Overs per innings. At least a third of them are super-over wins, and most of those carry
Super_Over_Innings 3 and 4. The other matches have no Deliveries. The example also draws the
Counted_Wicket codes passed to ``wicket_flags``: the glossary's codes 1-6 and Retire Out (9), or
those plus code 10, as the Auditor classified the real data. The Retired_Hurt code 7 is never
among them.

The test runs in two steps.

1. The played matches as generated. Under each match's own encoding, ``scheduled_balls``,
   ``innings_aggregates`` and ``targets`` reproduce the scorecard that the strategy built the
   match from: the Scheduled_Balls, totals and wickets fallen of both innings, the team runs of
   the final innings-2 Delivery and the Target.
2. The matches after changes that bring up every case of the criteria:

   * each of ``first_inn_revised_overs``, ``second_inn_revised_overs`` and ``revised_target`` of a
     played match is redrawn half of the time, as blank or as a value (1-120 balls, or a Target
     of 1-300), and the matches without Deliveries draw all three this way, so every
     combination of present and missing values occurs;
   * some played matches lose the Deliveries of innings 1 or of innings 2;
   * a few Deliveries get a new ``is_wicket`` (true, false or missing) and ``dismissal_type_id``
     (a Counted_Wicket code, code 10, the Retired_Hurt code 7 or blank). Every example holds a
     Retired_Hurt dismissal and, with 2 or more Deliveries, a Retire Out. Only these two columns
     change, because the functions under test read nothing else about a dismissal;
   * the Deliveries and the matches rows are shuffled, so no function can rely on row order.

   The results then equal a direct transcription of the criteria, applied to the changed tables:

   * Scheduled_Balls (criteria 4-6). ``scheduled_balls`` gives exactly innings 1 and 2 of every
     match, sorted. Innings 1 takes ``first_inn_revised_overs`` and innings 2
     ``second_inn_revised_overs``, each read as a count of balls; innings 2 falls back to
     ``first_inn_revised_overs``; an innings with neither gets 120. ``delivery_scheduled_balls``
     gives each Delivery of innings 1 and 2 its innings' value and each Delivery of a
     Super_Over_Innings 120, whatever the revised columns hold.
   * Team runs. Under every candidate encoding of the Config, each Delivery's team runs equal its
     cell of the design's DD-4 table, transcribed below.
   * Innings aggregates (criterion 8). ``innings_aggregates`` gives one row per innings with
     Deliveries, Super_Over_Innings included, sorted. Its total is the sum of the DD-4 team runs
     of its Deliveries, its wickets fallen count the rows with ``is_wicket`` true and a
     Counted_Wicket code (so Retire Out counts and the Retired_Hurt code does not), and its
     final-ball runs are the team runs of the Delivery with the highest ``innings_ball_num``.
   * Target (criteria 9-10). Under every candidate encoding, ``targets`` gives each match its
     ``revised_target`` when that holds a value, else that encoding's innings-1 total plus 1. A
     match with neither (no ``revised_target`` and no innings-1 Delivery, as for a match with no
     Deliveries) has no Target.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import pandas as pd
from hypothesis import given, note
from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy

from superr.reconstruct import (
    SUPER_OVER_WIN,
    candidate_team_runs,
    classify,
    delivery_scheduled_balls,
    encoding_inputs,
    innings_aggregates,
    scheduled_balls,
    targets,
    wicket_flags,
)
from tests.strategies import (
    CANDIDATE_ENCODINGS,
    COUNTED_WICKET_CODES,
    EXTRAS_TYPE_CLASSES,
    MATCH_COLUMNS,
    RETIRE_OUT,
    RETIRED_HURT,
    RETIRED_HURT_CODES,
    MatchCase,
    scorecard_match,
)

if TYPE_CHECKING:
    from superr.settings import Settings

#: Scheduled_Balls of an innings that no revised-overs value applies to (criterion 6): 20 Overs of 6 balls.
FULL_INNINGS_BALLS = 120
#: The innings that the revised-overs columns cover; innings 3 and 4 are Super_Over_Innings.
REGULAR_INNINGS = (1, 2)
#: The code that the Auditor confirmed as a Counted_Wicket on the real data, beyond the glossary's 1-6 and 9.
AUDITOR_CODE = 10
#: Counted_Wicket codes passed to wicket_flags: the glossary's (which the generator uses) or the real data's.
COUNTED_SETS: tuple[tuple[int, ...], ...] = (COUNTED_WICKET_CODES, (*COUNTED_WICKET_CODES, AUDITOR_CODE))
if any(RETIRE_OUT not in codes or set(RETIRED_HURT_CODES) & set(codes) for codes in COUNTED_SETS):
    raise RuntimeError("Property 8 needs Retire Out among the Counted_Wicket codes and no Retired_Hurt code among them")

# The design's DD-4 table, with b = runs and e = extras_2_runs. Each cell is written as
# (constant, coefficient of b, coefficient of e), so "1 + b + e" is (1, 1, 1). Transcribed from the
# design rather than imported from superr.reconstruct, so the reference stays independent.
_B_E, _1_E, _1_B_E, _1_B, _5_B_E = (0, 1, 1), (1, 0, 1), (1, 1, 1), (1, 1, 0), (5, 1, 1)
DD4_COLUMNS = ("legal", "bye", "penalty", "wide", "no_ball")
DD4: dict[str, tuple[tuple[int, int, int], ...]] = {
    #                    Legal  Bye/leg-bye  Penalty  Wide  No_Ball
    "F30":               (_B_E, _B_E,        _B_E,    _1_E, _1_B_E),
    "byes_one":          (_B_E, _1_E,        _B_E,    _1_E, _1_B_E),
    "byes_one_nb_plain": (_B_E, _1_E,        _B_E,    _1_E, _1_B),
    "byes_one_penalty5": (_B_E, _1_E,        _5_B_E,  _1_E, _1_B_E),
}
_no_rule = [name for name in CANDIDATE_ENCODINGS if name not in DD4]
if _no_rule:
    raise RuntimeError(f"Property 8 has no DD-4 row for the candidate encoding(s) {_no_rule}; add them to DD4")
#: extras_type values of the Legal_Balls that take the bye / leg-bye column and the Penalty column of DD-4.
BYE_TYPES = ("byes", "legbyes")
PENALTY_TYPE = "Penalty"

#: Columns of the innings_aggregates table that Property 8 covers.
AGG_COLUMNS = ["match_id", "innings", "total", "wickets", "final_ball_runs"]
#: Columns of the Deliveries shown when Hypothesis reports a falsifying example.
NOTE_COLUMNS = [
    "match_id", "innings", "innings_ball_num", "extras_type", "runs", "extras_2_runs", "is_wicket", "dismissal_type_id",
]

# --------------------------------------------------------------------------------- strategy

#: 1-5 distinct match ids. The first 1-3 belong to played matches; the rest have no Deliveries.
MATCH_IDS = st.lists(st.integers(1, 99_999), min_size=1, max_size=5, unique=True)
MAX_PLAYED = 3
#: Result types asked of scorecard_match. None lets it draw one; a super-over win mostly brings innings 3 and 4.
RESULT_TYPES = (None, None, SUPER_OVER_WIN)
#: A redrawn value of each matches column; None leaves the field blank. Revised overs are counts of balls.
REVISED_VALUES: dict[str, SearchStrategy[int | None]] = {
    "first_inn_revised_overs": st.one_of(st.none(), st.integers(1, FULL_INNINGS_BALLS)),
    "second_inn_revised_overs": st.one_of(st.none(), st.integers(1, FULL_INNINGS_BALLS)),
    "revised_target": st.one_of(st.none(), st.integers(1, 300)),
}
#: The innings whose Deliveries a played match loses; None keeps them all.
DROPPED_INNINGS = (None, None, None, None, 1, 2)
#: A new (is_wicket, dismissal_type_id) for a Delivery; None is a missing value.
WICKET_EDITS = st.tuples(
    st.sampled_from((True, True, False, None)),
    st.sampled_from((RETIRED_HURT, RETIRE_OUT, AUDITOR_CODE, *COUNTED_WICKET_CODES, None)),
)
#: Dismissals that every example holds: a Retired_Hurt one, and a Retire Out when there are 2+ Deliveries.
ALWAYS_PRESENT = ((True, RETIRED_HURT), (True, RETIRE_OUT))
UNPLAYED_RESULT = "Match abandoned without a ball bowled"


class Example(NamedTuple):
    """One example of Property 8."""

    cases: list[MatchCase]  # the played matches as scorecard_match built them (step 1)
    balls: pd.DataFrame  # their Deliveries after the changes of step 2, shuffled
    matches: pd.DataFrame  # one row per match, played or not, after the changes of step 2, shuffled
    counted: tuple[int, ...]  # the Counted_Wicket codes passed to wicket_flags


@st.composite
def _examples(draw: Any) -> Example:
    ids = draw(MATCH_IDS)
    n_played = draw(st.integers(1, min(MAX_PLAYED, len(ids))))
    cases = [
        draw(scorecard_match(result_type=draw(st.sampled_from(RESULT_TYPES)), match_id=match_id))
        for match_id in ids[:n_played]
    ]

    rows: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    for case in cases:
        row = {name: case.matches[name].iloc[0] for name in MATCH_COLUMNS}
        for name, values in REVISED_VALUES.items():
            if draw(st.booleans()):  # redraw the field; otherwise keep the generated value
                row[name] = draw(values)
        rows.append(row)
        dropped = draw(st.sampled_from(DROPPED_INNINGS))
        frames.append(case.balls if dropped is None else case.balls[case.balls["innings"] != dropped])
    for match_id in ids[n_played:]:
        revised = {name: draw(values) for name, values in REVISED_VALUES.items()}
        rows.append({"match_id": match_id, **revised, "win_team_id": None, "result": UNPLAYED_RESULT})
    matches = pd.DataFrame(
        {name: pd.array([row[name] for row in rows], dtype=dtype) for name, dtype in MATCH_COLUMNS.items()}
    )

    balls = pd.concat(frames, ignore_index=True)
    n = len(balls)
    edits = draw(st.dictionaries(st.integers(0, n - 1), WICKET_EDITS, max_size=4))
    fixed = draw(st.lists(st.integers(0, n - 1), min_size=min(n, 2), max_size=2, unique=True))
    edits.update(zip(fixed, ALWAYS_PRESENT))
    is_wicket, code = balls["is_wicket"].tolist(), balls["dismissal_type_id"].tolist()
    for position, (new_flag, new_code) in edits.items():
        is_wicket[position], code[position] = new_flag, new_code
    balls["is_wicket"] = pd.array(is_wicket, dtype="boolean")
    balls["dismissal_type_id"] = pd.array(code, dtype="Int64")

    return Example(
        cases=cases,
        balls=balls.iloc[draw(st.permutations(range(n)))],
        matches=matches.iloc[draw(st.permutations(range(len(matches))))],
        counted=draw(st.sampled_from(COUNTED_SETS)),
    )


# -------------------------------------------------------------------------------- reference

def _dd4_runs(name: str, extras_type: str, b: int, e: int) -> int:
    """Team runs of one Delivery under encoding ``name``: its cell of the DD-4 table.

    The column is Wide or No_Ball by the Config class mapping; among Legal_Balls, ``byes`` and
    ``legbyes`` take the bye / leg-bye column, ``Penalty`` the Penalty column and the rest (as
    delivered, ``None``) the Legal column.
    """
    if extras_type in EXTRAS_TYPE_CLASSES["wide"]:
        column = "wide"
    elif extras_type in EXTRAS_TYPE_CLASSES["no_ball"]:
        column = "no_ball"
    elif extras_type in BYE_TYPES:
        column = "bye"
    else:
        column = "penalty" if extras_type == PENALTY_TYPE else "legal"
    constant, per_bat_run, per_extra_run = DD4[name][DD4_COLUMNS.index(column)]
    return constant + per_bat_run * b + per_extra_run * e


def _reference_scheduled_balls(matches: pd.DataFrame) -> dict[int, tuple[int, int]]:
    """Scheduled_Balls of innings 1 and 2 of every match (criteria 4-6); revised values are counts of balls."""
    found: dict[int, tuple[int, int]] = {}
    columns = (matches[name].tolist() for name in ("match_id", "first_inn_revised_overs", "second_inn_revised_overs"))
    for match_id, first, second in zip(*columns):
        innings_1 = first if pd.notna(first) else FULL_INNINGS_BALLS  # criterion 4, else criterion 6
        if pd.notna(second):
            innings_2 = second  # criterion 4
        elif pd.notna(first):
            innings_2 = first  # criterion 5: the fallback to first_inn_revised_overs
        else:
            innings_2 = FULL_INNINGS_BALLS  # criterion 6
        found[match_id] = (innings_1, innings_2)
    return found


def _reference_aggregates(
    balls: pd.DataFrame, team_runs: Sequence[int], counted: Collection[int]
) -> list[tuple[int, int, int, int, int]]:
    """(match_id, innings, total, wickets fallen, final-ball runs) of every innings with Deliveries, sorted.

    The total adds the team runs of the innings' Deliveries. The wickets fallen count its rows with
    ``is_wicket`` true and a ``dismissal_type_id`` among ``counted`` (criterion 8). The final-ball
    runs are the team runs of its Delivery with the highest ``innings_ball_num``.
    """
    found: dict[tuple[int, int], list[int]] = {}  # (match_id, innings) -> [total, wickets, highest ball, its runs]
    names = ("match_id", "innings", "innings_ball_num", "is_wicket", "dismissal_type_id")
    for match_id, innings, ball, is_wicket, code, runs in zip(*(balls[name].tolist() for name in names), team_runs):
        entry = found.setdefault((match_id, innings), [0, 0, 0, 0])
        entry[0] += runs
        entry[1] += bool(pd.notna(is_wicket) and is_wicket and pd.notna(code) and code in counted)
        if ball > entry[2]:
            entry[2], entry[3] = ball, runs
    return [(match_id, innings, total, wickets, last) for (match_id, innings), (total, wickets, _, last) in sorted(found.items())]


def _reference_targets(matches: pd.DataFrame, first_totals: Mapping[int, int]) -> dict[int, int | None]:
    """Target of every match (criteria 9-10): ``revised_target``, else the innings-1 total + 1, else none."""
    found: dict[int, int | None] = {}
    for match_id, revised in zip(matches["match_id"].tolist(), matches["revised_target"].tolist()):
        if pd.notna(revised):
            found[match_id] = revised  # criterion 9
        elif match_id in first_totals:
            found[match_id] = first_totals[match_id] + 1  # criterion 10
        else:
            found[match_id] = None  # no revised_target and no innings-1 Delivery
    return found


# ---------------------------------------------------------------------------------- helpers

class _Shown:
    """Columns of a table for hypothesis.note, formatted only when Hypothesis prints the note."""

    def __init__(self, title: str, frame: pd.DataFrame, columns: Sequence[str]) -> None:
        self.title, self.frame, self.columns = title, frame, list(columns)

    def __repr__(self) -> str:
        return f"{self.title}:\n{self.frame[self.columns].to_string()}"


def _team_runs(balls: pd.DataFrame) -> dict[str, Any]:
    """Team runs of every Delivery under each candidate encoding, in the row order of ``balls``."""
    return candidate_team_runs(encoding_inputs(balls, classify(balls["extras_type"], EXTRAS_TYPE_CLASSES)), CANDIDATE_ENCODINGS)


def _check_scorecards(cases: Sequence[MatchCase], counted: Collection[int], d14: tuple[Any, Any, Any]) -> None:
    """Step 1: on the matches as generated, each match's results under its encoding reproduce its scorecard."""
    balls = pd.concat([case.balls for case in cases], ignore_index=True)
    matches = pd.concat([case.matches for case in cases], ignore_index=True)
    sched = {(match_id, innings): value for match_id, innings, value in scheduled_balls(matches).itertuples(index=False, name=None)}
    runs = _team_runs(balls)
    flags = wicket_flags(balls, counted, *d14)["counted_wicket"]
    for name in dict.fromkeys(case.encoding for case in cases):
        agg = innings_aggregates(balls, runs[name], flags)
        found = {(m, i): (total, wickets, last) for m, i, total, wickets, last in agg[AGG_COLUMNS].itertuples(index=False, name=None)}
        target = targets(matches, agg)
        target_of = dict(zip(target.index.tolist(), target.tolist()))
        for case in cases:
            if case.encoding != name:
                continue
            match_id, card = int(case.matches["match_id"].iloc[0]), case.scorecard
            (total_1, wickets_1, _), (total_2, wickets_2, final_2) = found[match_id, 1], found[match_id, 2]
            got = (sched[match_id, 1], sched[match_id, 2]), (total_1, total_2), (wickets_1, wickets_2), final_2, target_of[match_id]
            want = card.sched_balls, card.totals, card.wickets, card.final_ball_runs, card.target
            assert got == want, (
                f"match {match_id} under {name}: (Scheduled_Balls, totals, wickets fallen, final-ball runs, Target) "
                "differ from the scorecard it was generated from"
            )


# ------------------------------------------------------------------------------------- test

@given(example=_examples())
def test_scheduled_balls_target_and_innings_aggregates(example: Example, settings: Settings) -> None:
    """Scheduled_Balls, team runs, innings aggregates and Targets follow Requirement 5, criteria 4-6 and 8-10.

    **Validates: Requirements 5.4, 5.5, 5.6, 5.8, 5.9, 5.10**
    """
    d14 = (
        settings.get("design_choices.D14_bowler_wicket_codes"),
        settings.get("design_choices.D14_run_out_code"),
        settings.get("design_choices.D14_retire_out_code"),
    )
    counted = example.counted
    note(f"Counted_Wicket codes: {counted}")
    _check_scorecards(example.cases, counted, d14)

    balls, matches = example.balls, example.matches
    note(_Shown("Matches (step 2)", matches, ["match_id", *REVISED_VALUES]))
    note(_Shown("Deliveries (step 2)", balls, NOTE_COLUMNS))

    # Scheduled_Balls (criteria 4-6): innings 1 and 2 of every match; 120 for each Super_Over_Innings Delivery.
    expected_sched = _reference_scheduled_balls(matches)
    sched = scheduled_balls(matches)
    assert list(sched.itertuples(index=False, name=None)) == [
        (match_id, innings, expected_sched[match_id][innings - 1])
        for match_id in sorted(expected_sched)
        for innings in REGULAR_INNINGS
    ]
    per_delivery = delivery_scheduled_balls(balls, sched)
    assert per_delivery.index.equals(balls.index)
    assert per_delivery.tolist() == [
        expected_sched[match_id][innings - 1] if innings in REGULAR_INNINGS else FULL_INNINGS_BALLS
        for match_id, innings in zip(balls["match_id"].tolist(), balls["innings"].tolist())
    ]

    # Team runs, innings aggregates (criterion 8) and Targets (criteria 9-10) under every candidate encoding.
    runs = _team_runs(balls)
    flags = wicket_flags(balls, counted, *d14)["counted_wicket"]
    inputs = list(zip(balls["extras_type"].tolist(), balls["runs"].tolist(), balls["extras_2_runs"].tolist()))
    for name in CANDIDATE_ENCODINGS:
        expected_runs = [_dd4_runs(name, *values) for values in inputs]
        assert runs[name].tolist() == expected_runs, f"team runs under {name} differ from the DD-4 table"

        expected_agg = _reference_aggregates(balls, expected_runs, counted)
        agg = innings_aggregates(balls, runs[name], flags)
        assert list(agg[AGG_COLUMNS].itertuples(index=False, name=None)) == expected_agg, (
            f"innings (match_id, innings, total, wickets, final_ball_runs) under {name}"
        )

        first_totals = {match_id: total for match_id, innings, total, _, _ in expected_agg if innings == 1}
        expected_target = _reference_targets(matches, first_totals)
        target = targets(matches, agg)
        assert target.index.tolist() == sorted(expected_target)
        assert [None if pd.isna(value) else value for value in target.tolist()] == [
            expected_target[match_id] for match_id in sorted(expected_target)
        ], f"Targets under {name}"
