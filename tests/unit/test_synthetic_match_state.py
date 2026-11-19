"""Synthetic two-innings match test (Requirement 6, criterion 16).

``tests/fixtures/synthetic_match.csv`` holds the 34 Deliveries of one synthetic match
(``match_id`` 900001) in the Requirement 5 order. Each row gives the Delivery's raw
``ball_by_ball.csv`` columns, its team runs under the Extras_Encoding the test holds fixed
(``exp_team_runs``), the hand-computed value of every state feature of Requirement 6,
criteria 2-14 (``exp_<feature>``), the three recent-window features for a window of 4
Legal_Balls (``exp_<feature>_w4``) and a ``note`` with the working of the tricky values.
``tests/fixtures/synthetic_match_xi.csv`` is the match's ``playing_eleven.csv``, read with the
Data_Loader. Every expected value was worked out by hand from the requirement rules, never
copied from the code's output. Ball numbers below are ``innings_ball_num`` values.

The match
    Royals (team 91) bat first for 3 Overs (Scheduled_Balls 18, from
    ``first_inn_revised_overs``) and make 29. Capitals (team 92) chase a ``revised_target`` of
    20 in 2 Overs (Scheduled_Balls 12, from ``second_inn_revised_overs``) and reach it off the
    last ball, winning by 9 wickets. Each side lists 12 ``is_played`` players with
    ``batting_order`` 1-12, among them one Impact Player (``is_impact`` = 1) and the player he
    replaced (``is_impact_off`` = 1), plus one squad player with ``is_played`` false. The XI
    rows are not in ``batting_order``.

    * Innings 1, Over 1 (bowler 207): a Wide opens the match; 101 has two dots and a leg-bye,
      102 a single, 101 a dot and a four.
    * Over 2 (205, then 212): 102 has a dot, a No_Ball (``other``) hit for six and a dot, then
      retires hurt on ball 10 (code 7, "Retired Hurt"). Royals' Impact Player 103 (replacing
      108) comes in. 205 leaves injured after ball 11 and Capitals' Impact Player 212
      (replacing 205) bowls the rest of the Over: his first appearance in the match. 101 is
      caught off the last ball and 102 resumes his innings.
    * Over 3 (207 again): 103 hits a six, has a dot, a Wide, then is bowled; 104 takes a
      single; 102 a dot and a four.
    * Innings 2, against 110 (Over 1) and 109 (Over 2): 201 a single, 202 a four, a dot and
      caught; 203 comes in; a six, a Wide, and ones and twos to 20.

How ``balls`` is built
    The fixture's raw columns get the Data_Loader dtypes and go through the reconstruct
    functions in the order ``reconstruct.run_stage`` chains them: order_deliveries, classify,
    wicket_flags, the team runs, innings_aggregates, parse_results, over_check, run_checks,
    exclusion_conditions and excluded_flags, then free_hit_flags, super_over_flags,
    dismissed_batter_flags, scheduled_balls with delivery_scheduled_balls, targets,
    match_columns and balls_table. The class mapping, the D14 codes, the Score_Tolerance and
    ``exclude_failing_matches`` come from the Config. Two inputs are fixed by the test:

    * The Extras_Encoding is ENCODING (``byes_one``), not the one select_encoding picks. One
      match says nothing about the encoding: its Score_Checks pass under every candidate, so
      the rule would pick F30. The State_Builder only reads ``team_runs`` (criterion 15 also
      holds the encoding fixed). ``byes_one`` is the encoding the delivered data selects, and
      under it the leg-bye scores 1 team run, so leaving it out of ``bowler_over_runs`` shows.
    * The Counted_Wicket codes are 1-6, 9 and 10, the Auditor's classification of the
      delivered data. The Retired_Hurt code 7 is not one of them.

    The Modal_Bowling_Type table, which a Run reads from the ``audit`` checkpoint, is
    MODAL_BOWLING_TYPES: each bowler's most frequent ``bowling_type`` in the fixture.

How the expected values were derived
    L is the number of Legal_Balls before the Delivery in its innings and SB its
    Scheduled_Balls. Rates are written as unreduced fractions, 6 x runs / balls (``96/9``), and
    compared within 1e-9; an empty field is a missing value, which must be missing in the
    output, never 0 or infinite. Phase codes are 0 powerplay, 1 middle and 2 death.

    * ``lb_bowled`` = L, ``lb_left`` = SB - L, ``score`` = team runs of the earlier Deliveries
      of the innings, ``wickets`` = earlier Counted_Wickets: the retirement on ball 10 never
      counts. ``crr`` = 6 x score / L is missing while L = 0, which holds on the opening Wide
      and on the ball after it, whose score is already 1.
    * ``runs_needed`` = 20 - score and ``rrr`` = 6 x runs_needed / lb_left in innings 2; both
      are missing in innings 1.
    * ``ball_share`` = (L + 1) / SB. With SB 18 and splits 0.30 / 0.75 the Phase is powerplay
      for L <= 4, middle for L 5-12 and death for L >= 13; with SB 12 it is powerplay for
      L <= 2, middle for L 3-8 and death for L >= 9. L = 8 gives 9/12 = 0.75 exactly, which is
      still middle. Both innings change Phase inside an Over. ``cell`` = 3 x (innings - 1) +
      Phase.
    * Recent window. With D6 = 18 no Delivery has 18 earlier Legal_Balls in its innings, so the
      window is the whole innings so far: ``win_lb`` = L, ``win_wkts`` = ``wickets`` and
      ``win_rr`` = ``crr``. With a window of 4 the window runs from the 4th most recent earlier
      Legal_Ball through the previous Delivery. Wides and No_Balls before that Legal_Ball drop
      out (the opening Wide from ball 6 on, the No_Ball from ball 14 on, the innings-2 Wide on
      ball 13); those after it stay in.
    * Striker. ``striker_lb_faced`` counts the striker's earlier Legal_Balls in the innings and
      ``striker_dots`` the Dot_Balls (zero bat runs) at the end of that sequence. The leg-bye
      is a Dot_Ball for 101. The No_Ball six and the Wides neither extend nor end a streak.
      102's count and streak carry over his retirement (balls 20 and 21).
    * ``bowler_over_runs``: team runs of the earlier Deliveries with the same innings,
      ``overid`` and ``bowler_id``, leaving out the leg-bye. 212 starts from 0 in the middle of
      Over 2, and 207 starts from 0 again in Over 3.
    * Previous-ball flags. ``prev_flag``: the previous Delivery of the innings is a four, a six
      or has an ``extras_type`` other than ``None`` (the leg-bye, Wides and the No_Ball
      included); false on the first Delivery of each innings, although innings 1 ends with a
      four. ``prev_in_over_flag``: the same test within the Over, so false on ball 8, which
      follows the four that ended Over 1. ``prev_in_over_boundary``: the previous Delivery of
      the Over is a four or a six (the No_Ball six included).
    * ``lb_since_boundary`` and ``lb_since_wicket``: Legal_Balls after the most recent earlier
      Boundary or Counted_Wicket, or L when there is none. The No_Ball six is a Boundary; the
      retirement is not a Counted_Wicket.
    * Batters_To_Come: the side's ``is_played`` players in ``batting_order`` who have not been
      striker or non-striker yet (counting the current Delivery), with an Impact Player only
      after his first appearance in any role in the match, capped at 11 - ``batted_count`` by
      removing ``is_impact_off`` players first. ``exp_btc`` lists the IDs in order; the test
      pads them with NO_PLAYER (-1) to ``btc_1``-``btc_11``. For Royals the cap is not binding
      before 103 appears (9 players for a cap of 9, 108 included); from his first ball (11) it
      drops 108. Capitals' 212 bowled in innings 1, so he qualifies from the first ball of
      innings 2, and the cap drops 205 there. ``next_batter`` is the first listed player. The
      returning 102 has already batted, so ``batted_count`` stays 3.
    * ``modal_bowling_type`` is the bowler's entry in MODAL_BOWLING_TYPES (ball 16 is recorded
      as Right-arm fast-medium, but 207's type is Right-arm Fast) and ``hand_x_style`` the
      striker's ``batting_type``, ``|`` and that type.
"""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest

from superr.io import DATE_FORMAT, PANDAS_DTYPES, SCHEMA_REGISTRY, load_csv
from superr.reconstruct import (
    balls_table,
    candidate_team_runs,
    check_mapping,
    classify,
    delivery_scheduled_balls,
    dismissed_batter_flags,
    encoding_inputs,
    excluded_flags,
    exclusion_conditions,
    free_hit_flags,
    innings_aggregates,
    match_columns,
    order_deliveries,
    over_check,
    parse_results,
    run_checks,
    scheduled_balls,
    super_over_flags,
    targets,
    wicket_flags,
)
from superr.state import BTC_SLOT_COLUMNS, NO_PLAYER, STATE_COLUMNS, WINDOW_COLUMNS, build_state

if TYPE_CHECKING:
    from superr.settings import Settings

FIXTURE_NAME = "synthetic_match.csv"
XI_FIXTURE_NAME = "synthetic_match_xi.csv"
EXPECTED_PREFIX = "exp_"
NOTE_COLUMN = "note"
SMALL_WINDOW_SUFFIX = "_w4"

#: The Extras_Encoding held fixed for the match (see the module docstring).
ENCODING = "byes_one"
#: Counted_Wicket codes: the Auditor's classification of the delivered data (1-6, 9 and 10).
COUNTED_WICKET_CODES = (1, 2, 3, 4, 5, 6, 9, 10)
#: The Retired_Hurt code of the retirement on ball 10 of innings 1, labelled "Retired Hurt".
RETIRED_HURT_CODE = 7
#: The second window length, which makes the window start inside the innings.
SMALL_WINDOW = 4
#: The D8 splits for which the Phases were worked out.
HAND_COMPUTED_SPLITS = (0.30, 0.75)
#: The most Legal_Balls in an innings of the match: with D6 at least this, the window is the whole innings so far.
LONGEST_INNINGS = 18
RATE_TOLERANCE = 1e-9

# Hand-computed inputs of the state: the scorecard behind the exp_* columns.
FIRST_INNINGS_TOTAL = 29
TARGET = 20  # revised_target; the chase reaches it exactly off the last ball
SCHEDULED_BALLS = {1: 18, 2: 12}
COUNTED_WICKET_BALLS = {(1, 14), (1, 18), (2, 4)}  # (innings, innings_ball_num): caught, bowled, caught

#: The match's matches.csv row and its competition's competitions.csv row, as text.
MATCH_ROW: dict[str, str] = {
    "match_id": "900001",
    "comp_id": "9001",
    "match_date": "2026-04-01",
    "venue_id": "501",
    "venue_country": "India",
    "first_inn_revised_overs": "18",
    "second_inn_revised_overs": "12",
    "revised_target": "20",
    "knock_out": "0",
    "win_team_id": "92",
    "result": "Capitals won by 9 wickets (DLS method)",
}
COMPETITION_ROW: dict[str, str] = {"comp_id": "9001", "comp_name": "IPL 2026", "match_format": "T20", "season": "2026"}

#: Modal_Bowling_Type of each bowler: his most frequent bowling_type in the fixture.
MODAL_BOWLING_TYPES: dict[int, str] = {
    207: "Right-arm Fast",  # 13 of his 14 Deliveries; ball 16 of innings 1 is recorded as Right-arm fast-medium
    205: "Right-arm offbreak",
    212: "Slow left-arm orthodox",
    110: "Left-arm fast-medium",
    109: "Right-arm Leg-Break",
}

# How each state column is compared: counts exactly, flags exactly, rates within RATE_TOLERANCE, texts exactly.
COUNT, FLAG, RATE, TEXT = "count", "flag", "rate", "text"
FEATURE_KINDS: dict[str, str] = {
    **dict.fromkeys(
        (
            "lb_bowled", "lb_left", "score", "wickets", "runs_needed", "phase", "cell", "win_lb", "win_wkts",
            "striker_lb_faced", "striker_dots", "bowler_over_runs", "lb_since_boundary", "lb_since_wicket",
            "btc_count", "next_batter", "batted_count", *BTC_SLOT_COLUMNS,
        ),
        COUNT,
    ),
    **dict.fromkeys(("prev_flag", "prev_in_over_flag", "prev_in_over_boundary"), FLAG),
    **dict.fromkeys(("crr", "rrr", "ball_share", "win_rr"), RATE),
    **dict.fromkeys(("modal_bowling_type", "hand_x_style"), TEXT),
}  # fmt: skip

_FLAG_OF_TEXT = {"true": True, "false": False}


# ------------------------------------------------------------------------------- building balls

def as_loaded(text: pd.DataFrame, file_key: str) -> pd.DataFrame:
    """``text`` (columns of Source_CSV ``file_key``, read as text) with the dtypes the Data_Loader gives them."""
    schema = SCHEMA_REGISTRY[file_key]
    unknown = [column for column in text.columns if column not in schema]
    assert not unknown, f"columns that {file_key} does not have: {unknown}"
    columns: dict[str, Any] = {}
    for column in text.columns:
        kind, values = schema[column], text[column]
        if kind == "bool":
            flags = values.map(_FLAG_OF_TEXT)
            assert flags.notna().equals(values.notna()), f"{column} must hold true, false or nothing"
            columns[column] = flags.astype(PANDAS_DTYPES["bool"])
        elif kind == "date":
            columns[column] = pd.to_datetime(values, format=DATE_FORMAT).astype(PANDAS_DTYPES["date"])
        elif kind == "str":
            columns[column] = values.astype(PANDAS_DTYPES["str"])
        else:
            columns[column] = pd.to_numeric(values).astype(PANDAS_DTYPES[kind])
    return pd.DataFrame(columns)


def build_balls(
    deliveries: pd.DataFrame, matches: pd.DataFrame, competitions: pd.DataFrame, settings: Settings
) -> pd.DataFrame:
    """The ``balls`` table of the Deliveries, chained as in reconstruct.run_stage with the encoding held at ENCODING."""
    classes = settings.get("data_rules.extras_type_classes")
    check_mapping(classes)
    ordered = order_deliveries(deliveries)
    cls = classify(ordered["extras_type"], classes)
    wickets = wicket_flags(
        ordered,
        COUNTED_WICKET_CODES,
        settings.get("design_choices.D14_bowler_wicket_codes"),
        settings.get("design_choices.D14_run_out_code"),
        settings.get("design_choices.D14_retire_out_code"),
    )
    team_runs = candidate_team_runs(encoding_inputs(ordered, cls), [ENCODING])[ENCODING]
    aggregates = innings_aggregates(ordered, team_runs, wickets["counted_wicket"])
    parsed = parse_results(matches)
    checks = run_checks(matches, aggregates, parsed, over_check(ordered, cls), encoding=ENCODING)
    conditions = exclusion_conditions(checks, settings.get("design_choices.D20_score_tolerance"))
    exclude_on = settings.get("data_rules.exclude_failing_matches")
    derived = pd.concat(
        [
            cls,
            super_over_flags(ordered["innings"]),
            free_hit_flags(ordered, cls),
            pd.Series(team_runs, index=ordered.index, name="team_runs"),
            wickets,
            dismissed_batter_flags(ordered, wickets),
            delivery_scheduled_balls(ordered, scheduled_balls(matches)),
            excluded_flags(ordered["match_id"], conditions, exclude_failing_matches=exclude_on),
        ],
        axis=1,
    )
    info = match_columns(matches, competitions, parsed, targets(matches, aggregates))
    return balls_table(ordered, derived, info)


def modal_table() -> pd.DataFrame:
    """MODAL_BOWLING_TYPES as the ``modal_bowling_type`` table of the audit checkpoint."""
    return pd.DataFrame(
        {
            "bowler_id": pd.array(list(MODAL_BOWLING_TYPES), dtype="Int64"),
            "modal_bowling_type": pd.array(list(MODAL_BOWLING_TYPES.values()), dtype="string"),
        }
    )


# ---------------------------------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def fixture_table(fixtures_dir: Path) -> pd.DataFrame:
    """The synthetic match, read as text so that the extras_type text None stays a value."""
    return pd.read_csv(fixtures_dir / FIXTURE_NAME, dtype=str, keep_default_na=False, na_values=[""])


@pytest.fixture(scope="module")
def balls(fixture_table: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    raw = [column for column in fixture_table.columns if not column.startswith(EXPECTED_PREFIX)]
    deliveries = as_loaded(fixture_table[[column for column in raw if column != NOTE_COLUMN]], "ball_by_ball")
    matches = as_loaded(pd.DataFrame([MATCH_ROW], dtype=str), "matches")
    competitions = as_loaded(pd.DataFrame([COMPETITION_ROW], dtype=str), "competitions")
    return build_balls(deliveries, matches, competitions, settings)


@pytest.fixture(scope="module")
def xi(fixtures_dir: Path) -> pd.DataFrame:
    """The match's playing_eleven rows, read by the Data_Loader."""
    return load_csv(fixtures_dir / XI_FIXTURE_NAME, "playing_eleven")


@pytest.fixture(scope="module")
def phase_splits(settings: Settings) -> list[float]:
    splits = settings.get("design_choices.D8_phase_splits")
    assert tuple(splits) == HAND_COMPUTED_SPLITS, f"the Phases were worked out for D8 = {HAND_COMPUTED_SPLITS}; got {splits}"
    return splits


@pytest.fixture(scope="module")
def state(balls: pd.DataFrame, xi: pd.DataFrame, phase_splits: list[float], settings: Settings) -> pd.DataFrame:
    """The state table with the Config window D6 and splits D8, as the state stage builds it."""
    window = settings.get("design_choices.D6_window")
    assert window >= LONGEST_INNINGS, (
        f"the window columns were worked out for a D6 window of at least {LONGEST_INNINGS} Legal_Balls; got {window}"
    )
    return build_state(balls, xi, modal_table(), window=window, phase_splits=phase_splits)


@pytest.fixture(scope="module")
def state_small_window(balls: pd.DataFrame, xi: pd.DataFrame, phase_splits: list[float]) -> pd.DataFrame:
    """The state table with a window of SMALL_WINDOW Legal_Balls, so the window starts inside the innings."""
    return build_state(balls, xi, modal_table(), window=SMALL_WINDOW, phase_splits=phase_splits)


# ------------------------------------------------------------------------------------- comparison

def _keys(frame: pd.DataFrame) -> list[tuple[int, int]]:
    return list(zip(frame["innings"].astype(int), frame["innings_ball_num"].astype(int)))


def _expected_rows(fixture_table: pd.DataFrame, balls: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
    """The fixture row of each state row, in the row order of ``state``, matched through ``row`` in ``balls``."""
    position = {key: index for index, key in enumerate(_keys(fixture_table))}
    assert len(position) == len(fixture_table), "the fixture must hold each (innings, innings_ball_num) once"
    wanted = _keys(balls.set_index("row").loc[state["row"].to_numpy()])
    assert sorted(wanted) == sorted(position), "state must hold one row per Delivery of the fixture"
    return fixture_table.iloc[[position[key] for key in wanted]].reset_index(drop=True)


def _btc_slots(text: Any) -> list[int]:
    """The hand-computed Batters_To_Come IDs of ``exp_btc``, padded with NO_PLAYER to the slot columns."""
    ids = [] if pd.isna(text) else [int(value) for value in str(text).split()]
    assert len(ids) <= len(BTC_SLOT_COLUMNS)
    return ids + [NO_PLAYER] * (len(BTC_SLOT_COLUMNS) - len(ids))


def _expected(kind: str, text: Any) -> Any:
    if pd.isna(text):
        return None
    if kind == COUNT:
        return int(text)
    if kind == FLAG:
        return _FLAG_OF_TEXT[text]
    if kind == RATE:
        return float(Fraction(text))
    return str(text)


def _same(kind: str, expected: Any, actual: Any) -> bool:
    if expected is None or actual is None:
        return expected is None and actual is None
    if kind == COUNT:
        return isinstance(actual, int) and not isinstance(actual, bool) and actual == expected
    if kind == FLAG:
        return isinstance(actual, bool) and actual == expected
    if kind == RATE:
        return isinstance(actual, float) and math.isfinite(actual) and abs(actual - expected) <= RATE_TOLERANCE
    return isinstance(actual, str) and actual == expected


def mismatches(state: pd.DataFrame, rows: pd.DataFrame, *, window_suffix: str = "") -> list[str]:
    """Every state value that differs from its hand-computed value, one line per value.

    ``rows`` holds the fixture row of each state row. The window columns are compared with
    ``exp_<column><window_suffix>``, every other column with ``exp_<column>``.
    """
    labels = [f"innings {innings} ball {ball} (row {row})" for (innings, ball), row in zip(_keys(rows), state["row"])]
    btc = [_btc_slots(text) for text in rows[f"{EXPECTED_PREFIX}btc"]]
    found = []
    for column in STATE_COLUMNS:
        kind = FEATURE_KINDS[column]
        if column in BTC_SLOT_COLUMNS:
            slot = BTC_SLOT_COLUMNS.index(column)
            expected = [slots[slot] for slots in btc]
        else:
            source = f"{EXPECTED_PREFIX}{column}{window_suffix if column in WINDOW_COLUMNS else ''}"
            expected = [_expected(kind, text) for text in rows[source]]
        actual = [None if pd.isna(value) else value for value in state[column].tolist()]
        found += [
            f"{label}: {column} expected {want!r}, got {got!r}"
            for label, want, got in zip(labels, expected, actual)
            if not _same(kind, want, got)
        ]
    return found


# ------------------------------------------------------------------------------------------ tests

def test_the_match_holds_every_event_that_criterion_16_names(
    fixture_table: pd.DataFrame, xi: pd.DataFrame, settings: Settings
) -> None:
    table = fixture_table
    keys = _keys(table)
    assert table["match_id"].nunique() == 1 and keys == sorted(keys), "one match, in the Requirement 5 order"
    assert sorted({innings for innings, _ in keys}) == [1, 2]

    classes = settings.get("data_rules.extras_type_classes")
    extras, code = table["extras_type"], pd.to_numeric(table["dismissal_type_id"])
    assert extras.isin(classes["wide"]).any()
    assert extras.isin(classes["no_ball"]).any()
    assert (extras == "legbyes").any()
    assert (table["is_four"] == "true").any() and (table["is_six"] == "true").any()
    assert code.isin(COUNTED_WICKET_CODES).any()
    assert RETIRED_HURT_CODE not in COUNTED_WICKET_CODES

    # A Retired_Hurt dismissal whose batter bats again later in the same innings.
    hurt = table.index[code == RETIRED_HURT_CODE]
    assert len(hurt) and (table.loc[hurt, "dismissal_type"] == "Retired Hurt").all()
    for at in hurt:
        innings, batter = table.at[at, "innings"], table.at[at, "dismissed_player_id"]
        later = table.loc[at + 1 :]
        later = later[later["innings"] == innings]
        assert ((later["striker_id"] == batter) | (later["non_striker_id"] == batter)).any()

    # A bowler change within an Over.
    assert (table.groupby(["innings", "overid"])["bowler_id"].nunique() > 1).any()

    # Impact Players who first appear mid-match; Batters_To_Come changes on one's first appearance.
    impact = xi.loc[(xi["is_played"] & xi["is_impact"].eq(1)).fillna(False).to_numpy(dtype=bool), "player_id"]
    assert len(impact)
    changed = []
    for player in impact.astype(str):
        seen = table[["striker_id", "non_striker_id", "bowler_id"]].eq(player).any(axis=1).to_numpy()
        first = int(seen.argmax())
        assert seen.any() and first > 0, f"Impact Player {player} must first appear after the first Delivery"
        changed.append(table.at[first, "exp_btc"] != table.at[first - 1, "exp_btc"])
    assert any(changed)

    # Phase changes inside each innings.
    assert (table.groupby("innings")["exp_phase"].nunique() > 1).all()


def test_balls_holds_the_hand_computed_inputs_of_the_state(fixture_table: pd.DataFrame, balls: pd.DataFrame) -> None:
    assert _keys(balls) == _keys(fixture_table), "balls keeps the Requirement 5 order of the fixture"
    assert balls["team_runs"].tolist() == fixture_table["exp_team_runs"].astype(int).tolist()
    assert balls.groupby("innings")["team_runs"].sum().to_dict() == {1: FIRST_INNINGS_TOTAL, 2: TARGET}
    assert set(_keys(balls[balls["counted_wicket"]])) == COUNTED_WICKET_BALLS  # the retirement is not one
    assert {innings: sorted(set(values)) for innings, values in balls.groupby("innings")["sched_balls"]} == {
        innings: [balls_] for innings, balls_ in SCHEDULED_BALLS.items()
    }
    assert balls["target"].tolist() == [TARGET] * len(balls)


def test_state_reproduces_every_hand_computed_feature(
    fixture_table: pd.DataFrame, balls: pd.DataFrame, state: pd.DataFrame
) -> None:
    assert set(FEATURE_KINDS) == set(STATE_COLUMNS), "work out the expected values of every state column"
    assert state.columns.tolist() == ["row", *STATE_COLUMNS]
    found = mismatches(state, _expected_rows(fixture_table, balls, state))
    assert not found, f"{len(found)} state values differ from the hand-computed ones:\n" + "\n".join(found)


def test_a_window_of_4_legal_balls_reproduces_the_hand_computed_window_features(
    fixture_table: pd.DataFrame, balls: pd.DataFrame, state_small_window: pd.DataFrame
) -> None:
    # The window columns follow the shorter window; every other column is as with D6.
    rows = _expected_rows(fixture_table, balls, state_small_window)
    found = mismatches(state_small_window, rows, window_suffix=SMALL_WINDOW_SUFFIX)
    assert not found, f"{len(found)} state values differ from the hand-computed ones:\n" + "\n".join(found)
