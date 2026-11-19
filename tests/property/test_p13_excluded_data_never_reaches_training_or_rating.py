# Feature: pressure-rating-system, Property 13: Excluded data never reaches training or rating
"""Property 13: excluded data never reaches training or rating (Requirement 5, criteria 11 and 24;
Requirement 11, criterion 3; Requirement 14, criterion 1).

**Validates: Requirements 5.11, 5.24, 11.3, 14.1**

For any synthetic dataset with Super_Over_Innings and any set of matches excluded under the
Score_Tolerance rule, no Delivery from innings 3-4 and no Delivery of an excluded match appears in
any model's Training_Rows, in the Validation_Deliveries or in the rating contributions; M4
Training_Rows contain only Legal_Balls of innings 1-2.

The examples
    ``tests.strategies.check_tables()`` draws the Score_Check results of 1-6 matches under a drawn
    set of Extras_Encodings, one of them selected, and a Score_Tolerance t (0-3, or D20 - 1 to
    D20 + 1, with misses clustered at 0, t and t + 1). ``reconstruct.exclusion_conditions`` at t names the excluded
    matches, so the excluded set comes from the real rule: none, some or all of the matches. Each
    match then gets innings 1 (and in most matches innings 2) of 1-6 Deliveries and, in about half
    of the two-innings matches, one or two Super_Over_Innings (3, then 4) of 1-4 Deliveries, in
    kept and in excluded matches alike. Each Delivery is one of TEMPLATES: Legal_Balls (``None``,
    ``legbyes``, ``byes``, ``Penalty``), Wides and No_Balls, some with a Bowler_Wicket, a stumping
    off a Wide or a run out of either batter.

    The tables are built with the Pipeline's own functions: ``reconstruct.order_deliveries``
    numbers the rows; ``super_over_flags``, ``free_hit_flags`` and ``excluded_flags`` (option
    ``exclude_failing_matches`` on) set the flags; ``balls_table`` assembles the ``balls`` table with
    its dtypes; ``state.pre_ball_features`` builds the ``state`` rows (innings 1-2 only, with D6 and
    D8 from the Config); ``folds.assign_folds`` builds the Fold_Table (D15 folds, the Config fold
    seed). The inputs that no row selection reads are made up: the Live_Ball and Pressure_Ball
    flags, the NBVs and Wicket_Cost from each row id, and the Delivery_Weights from each match's
    position.

The checks, each against the drawn innings, classes and excluded matches
    0. ``is_super_over`` is set exactly on innings 3-4 and ``excluded`` exactly on every Delivery of
       an excluded match; the ``state`` table holds every Delivery of innings 1-2 and nothing else.
    1. Training_Rows. Each selector returns exactly the Deliveries of innings 1-2 of the kept
       matches (Legal_Balls only where noted), in row order:

       * ``common.training_mask`` and ``common.legal_training_mask`` of the whole ``balls`` table;
       * M1: ``skill.legal_rows(..., training_only=True)``, Legal_Balls;
       * M2: ``common.training_mask`` of ``remaining_runs._delivery_rows``, the ``trainable`` flags
         of ``remaining_runs.run_stage`` (see "Not covered");
       * M3: ``win_prob.m3_labels`` ``trainable``; its ``eligible`` rows are a subset of them;
       * M4: ``outcome.outcome_rows`` ``trainable``, Legal_Balls (Requirement 11, criterion 3);
       * Extras_Model: ``extras.extras_rows`` ``trainable``, every class.

       The state-based selectors (M2, M3, M4, Extras_Model and the value filter) never see a
       Super_Over_Innings Delivery, because the state has no row for one. So the test also gives
       them a ``state`` table that wrongly holds those rows: M2, M3, M4 and the Extras_Model raise
       ValueError, and ``legal_rows`` and the value filter still return only the kept rows.
    2. Validation_Deliveries (Requirement 14, criterion 1): ``validation.validation_mask`` of the
       whole ``balls`` table, which ``validation.guard_inputs`` applies to the state rows.
    3. Rating contributions. The value stage's filter as ``value.load_value_inputs`` applies it:
       ``common.training_mask`` of the state rows of ``value._delivery_frame``, aligned with
       ``value._aligned``. Its kept rows ``d`` give the Dismissal_Charges
       (``value.dismissal_charges``) and the ``value`` rows, and ``rating.build_contributions``
       turns them into contributions. No contribution row (charge-only rows included) belongs to
       an excluded match, and per match and Role the count ``n`` equals the rated Batter_Deliveries
       and Bowler_Deliveries, so no Super_Over_Innings Delivery is counted either.

       ``build_contributions`` does not filter by itself: it reads no innings, ``is_super_over`` or
       ``excluded`` column. Given a ``value`` table that wrongly holds every Delivery, it neither
       raises nor drops any: it counts every one, Super_Over_Innings and excluded matches included.
       The value stage's filter is therefore the only guard of the rating contributions.

Not covered here
    * ``value.load_value_inputs`` and ``value.run_stage`` themselves: they read a complete
      CheckpointStore (the Primary_Models' predictions, the M2 and Extras_Model predictions, the
      Class_Run_Values and the ``pressure`` table). The test calls their filter's functions and
      repeats their glue: the kept rows ``rows.iloc[flatnonzero(keep)]``, the ``live`` and
      ``pressure_bat`` flags taken by ``row``, and one ``value`` row per row of ``d``, as
      ``value._value_table`` makes them, with made-up NBVs.
    * The M2 ``eligible`` flags: ``remaining_runs.run_stage`` computes them inline as ``trainable``
      and a finite target (and innings 1 for the first-innings variant), so no function returns
      them. The ``&`` can only drop rows, so the check of ``trainable`` bounds them.
    * The fits: ``common.fit_registered`` keeps the ``eligible`` rows of a fit's train folds, and
      ``outcome.fit_outcome`` and ``extras.fit_extras`` stop when a fit's Training_Rows differ from
      the ``trainable`` rows of its train folds. The leakage guard is Property 19.
    * Per-fold statistics built with the same masks (the value stage's cell tables, F20(a), the M2
      lookup baseline). Property 43 checks that they use the Training_Folds only.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import pandas as pd
import pytest
from hypothesis import event, given, note
from hypothesis import strategies as st

from superr import rating, validation
from superr import value as value_stage
from superr.folds import assign_folds
from superr.models import common, extras, outcome, remaining_runs, win_prob
from superr.rating import CompClass, Contributions, Period
from superr.reconstruct import (
    CLASS_KEYS,
    COMP_A_TEAM,
    COMP_IPL,
    COMP_T20I,
    COMPETITION_TYPES,
    DEFAULT_SCHEDULED_BALLS,
    RESULT_TYPES,
    WIN_BY_RUNS,
    WIN_BY_WICKETS,
    DeliveryClass,
    balls_table,
    excluded_flags,
    exclusion_conditions,
    free_hit_flags,
    order_deliveries,
    super_over_flags,
)
from superr.skill import legal_rows
from superr.state import STATE_INNINGS, pre_ball_features
from superr.value import BATTER_ROLE, BOWLER_ROLE
from tests.strategies import EXTRAS_TYPE_CLASSES, FOLD_SEED, N_FOLDS, CheckTables, check_tables

if TYPE_CHECKING:
    from superr.settings import Settings

_LEGAL, _WIDE, _NO_BALL = int(DeliveryClass.LEGAL), int(DeliveryClass.WIDE), int(DeliveryClass.NO_BALL)
_CLASS_NAMES = {_LEGAL: "Legal_Ball", _WIDE: "Wide", _NO_BALL: "No_Ball"}

#: Innings 3 and 4 are the Super_Over_Innings (glossary).
_FIRST_SUPER_OVER = 3

#: The ValueError text of a state-based selector given a state row outside innings 1 and 2.
_OUTSIDE = "outside innings 1 and 2"

_SHOWN = 5  # rows named in one failure message

# ------------------------------------------------------------------------------- Delivery templates

# Who a Delivery's Counted_Wicket dismisses, and how.
_NO_WICKET, _BOWLED, _STUMPED, _STRIKER_RUN_OUT, _NON_STRIKER_RUN_OUT = (
    "none", "bowled", "stumped", "striker run out", "non-striker run out",
)  # fmt: skip
#: dismissal_type_id and dismissal_type of each wicket, as in ball_by_ball.csv.
_DISMISSALS: dict[str, tuple[int, str]] = {
    _BOWLED: (1, "Bowled"),
    _STUMPED: (5, "Stump"),
    _STRIKER_RUN_OUT: (4, "RunOut"),
    _NON_STRIKER_RUN_OUT: (4, "RunOut"),
}


class Template(NamedTuple):
    """One kind of Delivery: its DeliveryClass code, extras_type text, bat runs, extras_2_runs and wicket."""

    cls: int
    extras_type: str
    runs: int
    extras_2_runs: int
    wicket: str


#: The Deliveries a generated innings is made of; the first, a dot ball, is where shrinking ends.
TEMPLATES: tuple[Template, ...] = (
    Template(_LEGAL, "None", 0, 0, _NO_WICKET),
    Template(_LEGAL, "None", 1, 0, _NO_WICKET),
    Template(_LEGAL, "None", 4, 0, _NO_WICKET),
    Template(_LEGAL, "None", 6, 0, _NO_WICKET),
    Template(_LEGAL, "None", 2, 0, _NO_WICKET),
    Template(_LEGAL, "None", 0, 0, _BOWLED),
    Template(_LEGAL, "None", 1, 0, _NON_STRIKER_RUN_OUT),
    Template(_LEGAL, "None", 0, 0, _STRIKER_RUN_OUT),
    Template(_LEGAL, "legbyes", 0, 1, _NO_WICKET),
    Template(_LEGAL, "byes", 0, 4, _NO_WICKET),
    Template(_LEGAL, "Penalty", 0, 5, _NO_WICKET),
    Template(_WIDE, "wide", 0, 0, _NO_WICKET),
    Template(_WIDE, "wide", 0, 4, _NO_WICKET),
    Template(_WIDE, "wide", 0, 0, _STUMPED),
    Template(_WIDE, "wide", 0, 0, _NON_STRIKER_RUN_OUT),
    Template(_NO_BALL, "other", 0, 0, _NO_WICKET),
    Template(_NO_BALL, "Overstep", 4, 0, _NO_WICKET),
    Template(_NO_BALL, "other", 1, 0, _STRIKER_RUN_OUT),
)


def _check_templates() -> None:
    """Stop at import unless every template's extras_type is listed under its class in the Config mapping."""
    for template in TEMPLATES:
        key = CLASS_KEYS[DeliveryClass(template.cls)]
        if template.extras_type not in EXTRAS_TYPE_CLASSES[key]:
            raise RuntimeError(
                f"TEMPLATES uses extras_type {template.extras_type!r} as {key}, but data_rules.extras_type_classes "
                f"lists {EXTRAS_TYPE_CLASSES[key]!r} there; update TEMPLATES"
            )


_check_templates()

# ------------------------------------------------------------------------------------- the draws

_TEAMS = (7, 18)  # the two sides of every match
_BOWLING = (("Right-arm offbreak", "Spin"), ("Right-arm Fast", "Pace"))  # bowling_type, pace_or_spin by Over parity
_COUNTRIES = {1: "India", 2: "United Arab Emirates"}  # venue_country of each venue_id
_SEASON, _YEAR = "2024", 2024
_FIRST_DATE = pd.Timestamp("2024-03-01")
_COMP_CLASS = {COMP_IPL: CompClass.IPL, COMP_T20I: CompClass.FULL_MEMBER, COMP_A_TEAM: CompClass.ASSOCIATE}

_TEMPLATE = st.integers(0, len(TEMPLATES) - 1)
_REGULAR_INNINGS = st.lists(_TEMPLATE, min_size=1, max_size=6)
_SUPER_OVER_INNINGS = st.lists(_TEMPLATE, min_size=1, max_size=4)
_RESULT_DRAWS = (WIN_BY_RUNS, WIN_BY_WICKETS, *RESULT_TYPES)  # the two result types that train M3 come up most
_WINNERS = ("first", "second", "none")  # win_team_id: the side batting first, the other side, or blank


class MatchDraw(NamedTuple):
    """One generated match: its attributes and, per innings, the TEMPLATES positions of its Deliveries."""

    match_id: int
    comp_type: str
    venue_id: int | None
    knock_out: int
    batting_first: int
    result_type: str
    winner: str
    innings: tuple[tuple[int, tuple[int, ...]], ...]


class Draw(NamedTuple):
    """The Score_Check results that decide the excluded matches, and the Deliveries of every match."""

    tables: CheckTables
    matches: tuple[MatchDraw, ...]


@st.composite
def _match(draw: Any, match_id: int) -> MatchDraw:
    n_regular = draw(st.sampled_from((2, 2, 2, 1)))  # innings 1 and 2, or innings 1 only
    n_super = draw(st.sampled_from((0, 2, 0, 2, 1))) if n_regular == 2 else 0
    numbers = (*range(1, n_regular + 1), *range(_FIRST_SUPER_OVER, _FIRST_SUPER_OVER + n_super))
    innings = tuple(
        (number, tuple(draw(_REGULAR_INNINGS if number in STATE_INNINGS else _SUPER_OVER_INNINGS)))
        for number in numbers
    )
    return MatchDraw(
        match_id=match_id,
        comp_type=draw(st.sampled_from(COMPETITION_TYPES)),
        venue_id=draw(st.sampled_from((1, 2, None))),
        knock_out=draw(st.sampled_from((0, 1))),
        batting_first=draw(st.sampled_from(_TEAMS)),
        result_type=draw(st.sampled_from(_RESULT_DRAWS)),
        winner=draw(st.sampled_from(_WINNERS)),
        innings=innings,
    )


@st.composite
def datasets(draw: Any) -> Draw:
    """A check_tables() draw and, for each of its matches, the Deliveries of innings 1-2 and any Super_Over_Innings."""
    tables = draw(check_tables())
    ids = sorted({int(match_id) for match_id in tables.checks["match_id"].tolist()})
    return Draw(tables, tuple(draw(_match(match_id)) for match_id in ids))


# ------------------------------------------------------------------------------- building the tables

#: Columns of the generated Deliveries with their dtypes: the ball_by_ball.csv columns that
#: balls_table reads (Data_Loader dtypes), then the derived values that balls_table takes from
#: ``derived``. order_deliveries carries them all through the ordering.
_RAW_DTYPES: dict[str, Any] = {
    "match_id": "Int64", "innings": "Int64", "overid": "Int64", "over_ball_num": "Int64",
    "innings_ball_num": "Int64", "batting_team_id": "Int64", "bowling_team_id": "Int64", "striker_id": "Int64",
    "non_striker_id": "Int64", "bowler_id": "Int64", "dismissed_player_id": "Int64", "batting_type": "string",
    "bowling_type": "string", "pace_or_spin": "string", "bowl_speed": "Int64", "runs": "Int64",
    "is_four": "boolean", "is_six": "boolean", "is_wicket": "boolean", "dismissal_type_id": "Int64",
    "dismissal_type": "string", "extras_type": "string", "extras_2_runs": "Int64",
    # derived values
    "cls": np.int8, "team_runs": np.int64, "counted_wicket": bool, "bowler_wicket": bool, "run_out": bool,
    "retire_out": bool, "dismissed_is_striker": "boolean", "dismissed_fallback": bool,
}  # fmt: skip
_GIVEN_DERIVED: tuple[str, ...] = (
    "team_runs", "counted_wicket", "bowler_wicket", "run_out", "retire_out", "dismissed_is_striker",
    "dismissed_fallback",
)  # fmt: skip


class Truth(NamedTuple):
    """What the draw says about every Delivery, in ``balls`` order, and the matches that the rule excludes."""

    row: np.ndarray
    match: np.ndarray
    innings: np.ndarray
    cls: np.ndarray
    excluded_ids: frozenset[int]

    @property
    def excluded_match(self) -> np.ndarray:
        return np.isin(self.match, sorted(self.excluded_ids))

    @property
    def kept(self) -> np.ndarray:
        """The Deliveries that may be trained on, validated and rated: innings 1-2 of a kept match."""
        return np.isin(self.innings, STATE_INNINGS) & ~self.excluded_match

    def describe(self, rows: Any) -> str:
        """Each of the first rows of ``rows`` with its match, innings, class and exclusion."""
        rows = np.asarray(rows, dtype=np.int64)
        where = pd.Index(self.row).get_indexer(rows)
        parts = []
        for row, at in zip(rows[:_SHOWN].tolist(), where[:_SHOWN].tolist()):
            if at < 0:
                parts.append(f"row {row} (no such Delivery)")
                continue
            match = int(self.match[at])
            tags = [f"match {match}", f"innings {int(self.innings[at])}", _CLASS_NAMES[int(self.cls[at])]]
            if match in self.excluded_ids:
                tags.append("excluded match")
            parts.append(f"row {row} ({', '.join(tags)})")
        if len(rows) > _SHOWN:
            parts.append(f"and {len(rows) - _SHOWN} more")
        return "; ".join(parts)


class Dataset(NamedTuple):
    """The tables of one example."""

    balls: pd.DataFrame  # the reconstruct ``balls`` table (no ``fold``)
    state: pd.DataFrame  # pre_ball_features of ``balls``: one row per Delivery of innings 1-2
    fold_table: pd.DataFrame
    matches: pd.DataFrame  # ``match_id`` and ``win_team_id``, for m3_labels
    pressure: pd.DataFrame  # ``row``, ``live``, ``pressure_bat`` and ``pressure_bowl`` of every Delivery
    weights: pd.DataFrame  # the match universe of build_contributions: every match
    truth: Truth


def _other(team: int) -> int:
    return _TEAMS[1] if team == _TEAMS[0] else _TEAMS[0]


def _records(matches: Sequence[MatchDraw]) -> list[dict[str, Any]]:
    """One record of _RAW_DTYPES per Delivery, in the Requirement 5 order."""
    records: list[dict[str, Any]] = []
    for match in matches:
        for number, positions in match.innings:
            # The side batting first bats in innings 1 and 4; the other side in innings 2 and 3.
            batting = match.batting_first if number in (1, 4) else _other(match.batting_first)
            bowling = _other(batting)
            legal_before, overid, in_over = 0, 0, 0
            for k, position in enumerate(positions):
                template = TEMPLATES[position]
                over = 1 + legal_before // 6
                in_over = in_over + 1 if over == overid else 1
                overid = over
                striker, non_striker = batting * 100 + 1 + k % 2, batting * 100 + 2 - k % 2
                out = {_NO_WICKET: None, _NON_STRIKER_RUN_OUT: non_striker}.get(template.wicket, striker)
                code, label = _DISMISSALS.get(template.wicket, (None, None))
                bowling_type, pace_or_spin = _BOWLING[over % 2]
                records.append(
                    {
                        "match_id": match.match_id, "innings": number, "overid": over, "over_ball_num": in_over,
                        "innings_ball_num": k + 1, "batting_team_id": batting, "bowling_team_id": bowling,
                        "striker_id": striker, "non_striker_id": non_striker, "bowler_id": bowling * 100 + 6 + over % 2,
                        "dismissed_player_id": out, "batting_type": "RHB" if k % 2 == 0 else "LHB",
                        "bowling_type": bowling_type, "pace_or_spin": pace_or_spin, "bowl_speed": None,
                        "runs": template.runs, "is_four": template.runs == 4, "is_six": template.runs == 6,
                        "is_wicket": out is not None, "dismissal_type_id": code, "dismissal_type": label,
                        "extras_type": template.extras_type, "extras_2_runs": template.extras_2_runs,
                        "cls": template.cls,
                        "team_runs": template.runs + template.extras_2_runs + int(template.cls != _LEGAL),
                        "counted_wicket": out is not None,
                        "bowler_wicket": template.wicket in (_BOWLED, _STUMPED),
                        "run_out": template.wicket in (_STRIKER_RUN_OUT, _NON_STRIKER_RUN_OUT),
                        "retire_out": False,
                        "dismissed_is_striker": None if out is None else out == striker,
                        "dismissed_fallback": False,
                    }
                )  # fmt: skip
                legal_before += int(template.cls == _LEGAL)
    return records


def _column(values: list[Any], dtype: Any) -> Any:
    return pd.array(values, dtype=dtype) if isinstance(dtype, str) else np.array(values, dtype=dtype)


def _dates(n: int) -> pd.api.extensions.ExtensionArray:
    return pd.array([_FIRST_DATE + pd.Timedelta(days=day) for day in range(n)], dtype="datetime64[ns]")


def _match_info(matches: Sequence[MatchDraw], records: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """The match_columns table of the matches: match_id and MATCH_INFO_COLUMNS, Target = innings-1 total + 1."""
    totals: dict[tuple[int, int], int] = {}
    for record in records:
        key = (record["match_id"], record["innings"])
        totals[key] = totals.get(key, 0) + record["team_runs"]
    n = len(matches)
    targets = [totals[m.match_id, 1] + 1 if (m.match_id, 2) in totals else None for m in matches]
    return pd.DataFrame(
        {
            "match_id": np.array([m.match_id for m in matches], dtype=np.int32),
            "comp_type": pd.array([m.comp_type for m in matches], dtype="str"),
            "season": pd.array([_SEASON] * n, dtype="string"),
            "calendar_year": pd.array([_YEAR] * n, dtype="Int32"),
            "match_date": _dates(n),
            "knock_out": pd.array([m.knock_out for m in matches], dtype="Int32"),
            "venue_id": pd.array([m.venue_id for m in matches], dtype="Int32"),
            "venue_country": pd.array([_COUNTRIES.get(m.venue_id) for m in matches], dtype="string"),
            "target": pd.array(targets, dtype="Int32"),
            "result_type": pd.array([m.result_type for m in matches], dtype="str"),
        }
    )


def build(case: Draw, *, window: int, phase_splits: Sequence[float]) -> Dataset:
    """The tables of ``case``, built with the Pipeline's functions (module docstring, "The examples")."""
    checks = case.tables.checks
    conditions = exclusion_conditions(checks[checks["encoding"] == case.tables.selected], case.tables.tolerance)
    excluded_ids = frozenset(int(match_id) for match_id in conditions["match_id"].tolist())

    records = _records(case.matches)
    ordered = order_deliveries(pd.DataFrame({name: _column([r[name] for r in records], dtype)
                                             for name, dtype in _RAW_DTYPES.items()}))  # fmt: skip
    derived = pd.DataFrame(
        {
            "cls": ordered["cls"],
            "is_super_over": super_over_flags(ordered["innings"]),
            "free_hit": free_hit_flags(ordered, ordered["cls"]),
            **{name: ordered[name] for name in _GIVEN_DERIVED},
            "sched_balls": pd.Series(DEFAULT_SCHEDULED_BALLS, index=ordered.index, dtype=np.int32),
            "excluded": excluded_flags(ordered["match_id"], conditions, exclude_failing_matches=True),
        }
    )
    balls = balls_table(ordered, derived, _match_info(case.matches, records))
    state = pre_ball_features(balls, window=window, phase_splits=phase_splits)

    ids = [m.match_id for m in case.matches]
    n = len(ids)
    fold_table = assign_folds(
        pd.DataFrame({"match_id": pd.array(ids, dtype="Int64")}), [m.comp_type for m in case.matches],
        n_folds=N_FOLDS, seed=FOLD_SEED,
    )  # fmt: skip
    winners = [
        None if m.winner == "none" else m.batting_first if m.winner == "first" else _other(m.batting_first)
        for m in case.matches
    ]
    matches = pd.DataFrame({"match_id": pd.array(ids, dtype="Int64"), "win_team_id": pd.array(winners, dtype="Int64")})

    row = balls["row"].to_numpy(dtype=np.int64)
    wide = balls["cls"].to_numpy(dtype=np.int64) == _WIDE
    pressure = pd.DataFrame(
        {
            "row": balls["row"].to_numpy(),
            "live": row % 5 != 4,
            "pressure_bat": pd.arrays.BooleanArray(row % 3 == 0, wide),  # missing on Wides, as in the pressure table
            "pressure_bowl": row % 4 == 1,
        }
    )
    weights = pd.DataFrame(
        {
            "match_id": np.array(ids, dtype=np.int64),
            "match_date": _dates(n),
            "calendar_year": np.full(n, _YEAR, dtype=np.int64),
            "comp_class": np.array([int(_COMP_CLASS[m.comp_type]) for m in case.matches], dtype=np.int8),
            "period": np.full(n, int(Period.EARLY), dtype=np.int8),
            "delivery_weight": 1.0 + 0.5 * (np.arange(n) % 3),
        }
    )
    truth = Truth(
        row=row,
        match=balls["match_id"].to_numpy(dtype=np.int64),
        innings=balls["innings"].to_numpy(dtype=np.int64),
        cls=balls["cls"].to_numpy(dtype=np.int64),
        excluded_ids=excluded_ids,
    )
    return Dataset(balls, state, fold_table, matches, pressure, weights, truth)


# ------------------------------------------------------------------- the inputs of the stage steps

def _m2_state(state: pd.DataFrame) -> pd.DataFrame:
    """``state`` with ``btc_count``, which ``remaining_runs._delivery_rows`` requires but does not read."""
    return state.assign(btc_count=pd.array(np.zeros(len(state), dtype=np.int32), dtype="Int32"))


def _with_rows(state: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
    """``state`` plus a copy of its first row for each of ``rows``, numbered as that row."""
    extra = state.iloc[np.zeros(len(rows), dtype=np.intp)].reset_index(drop=True)
    extra["row"] = rows.astype(state["row"].dtype)
    return pd.concat([state, extra], ignore_index=True)


def _value_filter(
    balls: pd.DataFrame, state: pd.DataFrame, fold_table: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray]:
    """The rows and the keep mask of ``value.load_value_inputs`` (its steps 1 and 3) for the ``state`` rows."""
    frame = value_stage._delivery_frame(balls, state, fold_table)
    rows = value_stage._aligned(frame, state["row"].to_numpy(dtype=np.int64), value_stage._FRAME_COLUMNS, "balls")
    return rows, common.training_mask(rows)


def _flagged(frame: pd.DataFrame, pressure: pd.DataFrame) -> pd.DataFrame:
    """``frame`` with the ``live`` and ``pressure_bat`` flags of each of its rows, as load_value_inputs adds them."""
    at = pd.Index(pressure["row"].to_numpy(dtype=np.int64)).get_indexer(frame["row"].to_numpy(dtype=np.int64))
    out = frame.reset_index(drop=True)
    out["live"] = pressure["live"].to_numpy(dtype=bool)[at]
    out["pressure_bat"] = pressure["pressure_bat"].iloc[at].reset_index(drop=True).astype("boolean")
    return out


def _omega(rows: Any) -> np.ndarray:
    """A made-up Wicket_Cost of each row: 2-4 runs."""
    return 2.0 + np.asarray(rows, dtype=np.int64) % 3


def _value_table(frame: pd.DataFrame, cell: np.ndarray) -> pd.DataFrame:
    """The ``value`` columns that build_contributions reads, one row per row of ``frame``, with made-up NBVs."""
    row = frame["row"].to_numpy(dtype=np.int64)
    cls = frame["cls"].to_numpy(dtype=np.int64)
    return pd.DataFrame(
        {
            "row": row,
            "match_id": frame["match_id"].to_numpy(dtype=np.int64),
            "cls": cls,
            "cell": cell,
            "nbv_bat": np.where(cls == _WIDE, np.nan, (row % 7 - 3) / 4.0),  # NaN on a Wide, as in the value table
            "nbv_bowl": (2 - row % 5) / 4.0,
        }
    )


def _cells(balls: pd.DataFrame, state: pd.DataFrame) -> np.ndarray:
    """The ``state`` cell of every Delivery; 3 x (innings - 1) for a Delivery without a state row."""
    at = pd.Index(state["row"].to_numpy(dtype=np.int64)).get_indexer(balls["row"].to_numpy(dtype=np.int64))
    innings = balls["innings"].to_numpy(dtype=np.int64)
    return np.where(at >= 0, state["cell"].to_numpy(dtype=np.int64)[np.maximum(at, 0)], 3 * (innings - 1))


# --------------------------------------------------------------------------------------- checks

def _check_rows(got: Any, expected: np.ndarray, what: str, truth: Truth) -> None:
    """``got`` (row ids) holds exactly ``expected``, in ascending row order."""
    got = np.asarray(got, dtype=np.int64)
    leaked, lost = np.setdiff1d(got, expected), np.setdiff1d(expected, got)
    assert not len(leaked), f"{what} holds {len(leaked)} row(s) it must leave out: {truth.describe(leaked)}"
    assert not len(lost), f"{what} lacks {len(lost)} row(s) it must hold: {truth.describe(lost)}"
    assert np.array_equal(got, expected), f"{what} repeats a row or breaks the row order: {got.tolist()}"


def _check_flags(balls: pd.DataFrame, truth: Truth) -> None:
    """``is_super_over`` marks exactly innings 3-4 and ``excluded`` exactly the Deliveries of the excluded matches."""
    for column, expected in (("is_super_over", truth.innings >= _FIRST_SUPER_OVER), ("excluded", truth.excluded_match)):
        wrong = truth.row[balls[column].to_numpy(dtype=bool) != expected]
        assert not len(wrong), f"balls column {column!r} is wrong on {truth.describe(wrong)}"


def _check_rejects(what: str, select: Callable[[], Any]) -> None:
    """``select`` raises the ValueError of a state row outside innings 1 and 2."""
    try:
        select()
    except ValueError as error:
        assert _OUTSIDE in str(error), f"{what} raised another ValueError: {error}"
    else:
        pytest.fail(f"{what} accepted state rows of a Super_Over_Innings instead of raising ValueError")


def _check_counts(contributions: Contributions, rated: np.ndarray, truth: Truth, what: str) -> None:
    """Per match, the Role's summed ``n`` equals its Deliveries among ``rated``: Batter_Deliveries leave out Wides."""
    universe = contributions.match_ids
    for role, mask in ((BATTER_ROLE, rated & (truth.cls != _WIDE)), (BOWLER_ROLE, rated)):
        part = contributions.for_role(role)
        got = np.rint(np.bincount(part.match_idx, weights=part.n, minlength=len(universe))).astype(np.int64)
        expected = np.array([np.count_nonzero(mask & (truth.match == m)) for m in universe.tolist()], dtype=np.int64)
        assert np.array_equal(got, expected), (
            f"{what}: {role} Deliveries per match {dict(zip(universe.tolist(), got.tolist()))}; expected "
            f"{dict(zip(universe.tolist(), expected.tolist()))}"
        )


def _summary(truth: Truth, tolerance: int) -> str:
    """The example in words: the excluded matches and, per innings, its rows and Legal_Balls."""
    lines = [f"Score_Tolerance {tolerance}; excluded matches: {sorted(truth.excluded_ids) or 'none'}"]
    for match in np.unique(truth.match).tolist():
        mine = truth.match == match
        parts = []
        for number in np.unique(truth.innings[mine]).tolist():
            here = mine & (truth.innings == number)
            rows = truth.row[here]
            n_legal = np.count_nonzero(here & (truth.cls == _LEGAL))
            parts.append(f"innings {number}: rows {rows.min()}-{rows.max()}, {n_legal} of {len(rows)} Legal_Balls")
        excluded = " (excluded)" if match in truth.excluded_ids else ""
        lines.append(f"match {match}{excluded}: " + "; ".join(parts))
    return "\n".join(lines)


def _record_events(truth: Truth) -> None:
    n_matches, n_excluded = len(np.unique(truth.match)), len(truth.excluded_ids)
    event("excluded matches: " + ("none" if not n_excluded else "all" if n_excluded == n_matches else "some"))
    super_matches = set(np.unique(truth.match[truth.innings >= _FIRST_SUPER_OVER]).tolist())
    event(f"Super_Over_Innings in a kept match: {bool(super_matches - truth.excluded_ids)}")
    event(f"Super_Over_Innings in an excluded match: {bool(super_matches & truth.excluded_ids)}")


# ------------------------------------------------------------------------------------------ test

@given(case=datasets())
def test_excluded_data_never_reaches_training_or_rating(case: Draw, settings: Settings) -> None:
    """No Delivery of innings 3-4 or of an excluded match is a Training_Row, a Validation_Delivery or part of a
    rating contribution, and M4 trains on Legal_Balls of innings 1-2 only (Requirement 5, criteria 11 and 24;
    Requirement 11, criterion 3; Requirement 14, criterion 1).
    """
    data = build(
        case,
        window=settings.get("design_choices.D6_window"),
        phase_splits=settings.get("design_choices.D8_phase_splits"),
    )
    penalty = settings.get("design_choices.D14_retire_out_batter_penalty")
    balls, state, truth = data.balls, data.state, data.truth
    note(_summary(truth, case.tables.tolerance))
    _record_events(truth)
    kept = truth.kept
    kept_rows, kept_legal_rows = truth.row[kept], truth.row[kept & (truth.cls == _LEGAL)]

    # 0. The flags that every filter reads, and the state rows: every Delivery of innings 1-2, nothing else.
    _check_flags(balls, truth)
    _check_rows(state["row"], truth.row[np.isin(truth.innings, STATE_INNINGS)], "the state rows", truth)

    # 1. The Training_Rows of every model.
    _check_rows(truth.row[common.training_mask(balls)], kept_rows, "common.training_mask", truth)
    _check_rows(truth.row[common.legal_training_mask(balls)], kept_legal_rows, "common.legal_training_mask", truth)
    m1 = legal_rows(balls, state, training_only=True)
    _check_rows(m1["row"], kept_legal_rows, "M1: skill.legal_rows(training_only=True)", truth)
    m2 = remaining_runs._delivery_rows(balls, _m2_state(state))
    _check_rows(m2["row"][common.training_mask(m2)], kept_rows, "M2: training_mask of _delivery_rows", truth)
    labels = win_prob.m3_labels(balls, state, data.matches)
    _check_rows(labels["row"][labels["trainable"]], kept_rows, "M3: m3_labels trainable", truth)
    eligible = labels["row"].to_numpy(dtype=np.int64)[labels["eligible"].to_numpy(dtype=bool)]
    leaked = np.setdiff1d(eligible, kept_rows)
    assert not len(leaked), f"M3: m3_labels eligible holds row(s) it must leave out: {truth.describe(leaked)}"
    event(f"M3 eligible rows: {'some' if len(eligible) else 'none'}")
    m4 = outcome.outcome_rows(balls, state, data.fold_table)
    _check_rows(m4.row[m4.trainable], kept_legal_rows, "M4: outcome_rows trainable", truth)
    extras_model = extras.extras_rows(balls, state, data.fold_table)
    _check_rows(extras_model.row[extras_model.trainable], kept_rows, "Extras_Model: extras_rows trainable", truth)

    # 1b. A state table that wrongly holds the Super_Over_Innings Deliveries: rejected, or its extra rows dropped.
    super_rows = truth.row[truth.innings >= _FIRST_SUPER_OVER]
    if len(super_rows):
        wrong = _with_rows(state, super_rows)
        _check_rejects(
            "M2: remaining_runs._delivery_rows", lambda: remaining_runs._delivery_rows(balls, _m2_state(wrong))
        )
        _check_rejects("M3: win_prob.m3_labels", lambda: win_prob.m3_labels(balls, wrong, data.matches))
        _check_rejects("M4: outcome.outcome_rows", lambda: outcome.outcome_rows(balls, wrong, data.fold_table))
        _check_rejects("Extras_Model: extras.extras_rows", lambda: extras.extras_rows(balls, wrong, data.fold_table))
        m1_wrong = legal_rows(balls, wrong, training_only=True)
        _check_rows(m1_wrong["row"], kept_legal_rows, "M1: legal_rows with Super_Over_Innings state rows", truth)
        rows, keep = _value_filter(balls, wrong, data.fold_table)
        _check_rows(rows["row"][keep], kept_rows, "the value filter with Super_Over_Innings state rows", truth)

    # 2. The Validation_Deliveries.
    _check_rows(truth.row[validation.validation_mask(balls)], kept_rows, "validation.validation_mask", truth)

    # 3. The rating contributions, from the rows that the value stage's filter keeps.
    rows, keep = _value_filter(balls, state, data.fold_table)
    _check_rows(rows["row"][keep], kept_rows, "the value filter of load_value_inputs", truth)
    d = _flagged(rows.iloc[np.flatnonzero(keep)].reset_index(drop=True), data.pressure)
    charges = value_stage.dismissal_charges(d, _omega(d["row"]), retire_out_penalty=penalty)
    leaked = np.setdiff1d(charges["row"].to_numpy(dtype=np.int64), kept_rows)
    assert not len(leaked), f"dismissal_charges charges row(s) that are not rated: {truth.describe(leaked)}"
    event(f"Dismissal_Charges of rated Deliveries: {'some' if len(charges) else 'none'}")
    value_rows = _value_table(d, d["cell"].to_numpy(dtype=np.int64))
    contributions = rating.build_contributions(value_rows, data.pressure, charges, balls, data.weights)
    shown = sorted(set(contributions.match_id.tolist()) & truth.excluded_ids)
    assert not shown, f"the rating contributions hold rows of the excluded match(es) {shown}"
    _check_counts(contributions, kept, truth, "the rating contributions")

    # 3b. build_contributions itself keeps every row of the value table it is given, wrongly included ones too.
    every = _flagged(balls, data.pressure)
    charges_all = value_stage.dismissal_charges(every, _omega(every["row"]), retire_out_penalty=penalty)
    contributions_all = rating.build_contributions(
        _value_table(balls, _cells(balls, state)), data.pressure, charges_all, balls, data.weights
    )
    every_row = np.ones(len(truth.row), dtype=bool)
    _check_counts(contributions_all, every_row, truth, "build_contributions of every Delivery")
