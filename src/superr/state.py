"""State_Builder: the pre-ball match state of every Delivery (Requirement 6; design section
"State_Builder (``state.py``)").

Every function receives the Config values it needs as arguments: no function reads ``config.py``
or has a default value for a Config setting. The recent-window length D6
(``design_choices.D6_window``) and the Phase splits D8 (``design_choices.D8_phase_splits``) are
keyword arguments, so the Robustness_Suite can recompute the window columns for its alternative
windows (``sensitivity.D6``) with ``recent_window``, and the innings x Phase cells for its
alternative splits (``sensitivity.D8``) with ``ball_position_share``, ``phase_codes`` and
``cell_codes``.

Pre-ball rule (criteria 1 and 15)
    A Delivery's features come only from the Deliveries before it in the same innings, in the
    Requirement 5 order (``match_id``, ``innings``, ``innings_ball_num``), plus the Delivery's own
    ``match_id``, ``innings``, ``overid``, ``striker_id`` and ``bowler_id`` and its match's
    Scheduled_Balls (``sched_balls``) and Target (``target``). No Outcome_Column of the Delivery
    itself enters its own features, and neither do the columns derived from them (``cls``,
    ``team_runs``, ``counted_wicket``). Replacing the outcomes of a Delivery and of every later
    Delivery therefore leaves that Delivery's features unchanged.

Input
    ``balls`` is the ``balls`` table of the ``reconstruct`` checkpoint, or any DataFrame with the
    INPUT_COLUMNS. ``cls`` is the DeliveryClass code (Legal_Ball by the Config class mapping),
    ``team_runs`` the team runs under the selected Extras_Encoding, ``counted_wicket`` the
    Counted_Wicket flag and ``runs`` the bat runs. Only innings 1 and 2 get features: a
    Super_Over_Innings has no ``state`` row. The rows may come in any order. They are put in the
    Requirement 5 order with a stable sort, and every result is in that order, keyed by ``row``.

Features (criteria 2-11): the PRE_BALL_COLUMNS
    L is the number of Legal_Balls before the Delivery in its innings, SB its Scheduled_Balls.

    ===================== ======================================================================
    Column                Value
    ===================== ======================================================================
    lb_bowled             L, the Legal_Balls bowled
    lb_left               max(SB - L, 0), the Legal_Balls left
    score                 the team score: ``team_runs`` summed over the earlier Deliveries
    wickets               wickets fallen: Counted_Wickets on the earlier Deliveries
    crr                   current run rate 6 x score / L; missing when L = 0
    runs_needed           innings 2: Target - score; missing in innings 1
    rrr                   innings 2: Required_Rate 6 x runs_needed / lb_left; missing in innings 1
                          and when lb_left = 0
    ball_share            Ball_Position_Share (L + 1) / SB
    phase                 Phase (``Phase``): powerplay (0) when ball_share <= D8[0], middle (1)
                          when <= D8[1], death (2) otherwise
    cell                  innings x Phase cell: 3 x (innings - 1) + phase, 0 to 5
    win_lb                Legal_Balls in the recent window: min(L, D6)
    win_wkts              Counted_Wickets on the Deliveries of the recent window
    win_rr                6 x team runs in the recent window / win_lb; missing when win_lb = 0
    striker_lb_faced      earlier Legal_Balls of the innings with the Delivery's ``striker_id``
    striker_dots          Dot_Balls (zero bat runs) at the end of that sequence of Legal_Balls;
                          Wides and No_Balls neither extend nor end the streak
    bowler_over_runs      ``team_runs`` of the earlier Deliveries with the Delivery's ``overid``
                          and ``bowler_id``, leaving out ``extras_type`` ``byes`` and ``legbyes``;
                          0 when there is none
    prev_flag             the previous Delivery of the innings is a Boundary or has an
                          ``extras_type`` other than ``None``; false on the first Delivery
    prev_in_over_flag     the same test on the previous Delivery of the same Over; false on the
                          first Delivery of the Over (F25(b), Requirement 12, criterion 9)
    prev_in_over_boundary the previous Delivery of the same Over is a Boundary; false on the
                          first Delivery of the Over (the flag of the Extras_Free_PI)
    lb_since_boundary     Legal_Balls after the most recent earlier Boundary (``is_four`` or
                          ``is_six``); L when there is none
    lb_since_wicket       Legal_Balls after the most recent earlier Counted_Wicket; L when there
                          is none
    ===================== ======================================================================

    The recent window (criterion 7) runs from the D6-th most recent earlier Legal_Ball of the
    innings through the previous Delivery, and includes the Wides and No_Balls in that span.
    When fewer than D6 Legal_Balls precede the Delivery, the window is every earlier Delivery of
    the innings.

    An undefined feature is missing, never 0 or infinite (criterion 4): NaN in the float64
    columns ``crr``, ``rrr`` and ``win_rr``, and <NA> in the nullable Int32 ``runs_needed``.
    ``runs_needed`` and ``rrr`` are also missing on an innings-2 Delivery whose Target is
    missing. The counts, ``score``, ``bowler_over_runs`` and ``lb_*`` columns are int32, ``phase``
    and ``cell`` int8, the flags bool and ``ball_share`` float64.

Vectorised rules
    No loop runs over Deliveries. Each feature is a sum, a count since the last event, or a
    previous value within a group of Deliveries: the innings (``match_id``, ``innings``), the
    striker's innings (plus ``striker_id``), the Over (plus ``overid``) or the bowler's part of
    an Over (plus ``overid`` and ``bowler_id``). A stable sort by the group keys keeps each group
    in the Requirement 5 order. Then:

    * the sum over a group's earlier Deliveries is the exclusive running sum (the inclusive
      cumulative sum minus the Delivery's own value) minus the running sum before the group's
      first Delivery;
    * the count since the last event is the exclusive running count minus the inclusive running
      count at the group's most recent earlier event. That count comes from a running maximum of
      the counts at event rows, shifted down one row; when the group has no earlier event, the
      running count before the group's first Delivery is used instead;
    * the previous value is the value one row up, with the fill value on a group's first
      Delivery.

    The recent window starts at the position of Legal_Ball number L - D6 + 1 of the innings. That
    position is read from the positions of all Legal_Balls, indexed by the running Legal_Ball
    count, and the window sums are differences of prefix sums.

Data problems that make the features undefined stop the Run with a ``DataError`` (exit code 3)
that names each problem with its row count: in innings 1 or 2, a missing or non-whole value of an
integer input column (``target`` may be missing), a missing ``extras_type``, or a Scheduled_Balls
value that is not positive.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from enum import IntEnum
from numbers import Real
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from superr.errors import DataError
from superr.reconstruct import BYE_EXTRAS_TYPES, DeliveryClass

__all__ = [
    "BALLS_PER_OVER",
    "INPUT_COLUMNS",
    "PRE_BALL_COLUMNS",
    "Phase",
    "STATE_INNINGS",
    "WINDOW_COLUMNS",
    "ball_position_share",
    "cell_codes",
    "phase_codes",
    "pre_ball_features",
    "recent_window",
]


# ------------------------------------------------------------------------------------- constants

#: Legal_Balls in an Over and the unit of every run rate (runs per over), fixed by the Laws of
#: Cricket rather than a Method_Constant.
BALLS_PER_OVER = 6

#: The innings that get a ``state`` row. A Super_Over_Innings (innings 3 or 4) gets none.
STATE_INNINGS: tuple[int, int] = (1, 2)

#: The ``extras_type`` text of a Delivery without an extra (criterion 10). It is a value, not a
#: missing field, as the Data_Loader reads it.
_NO_EXTRA = "None"


class Phase(IntEnum):
    """Phase of a Delivery (Requirement 6, criterion 5), stored as the int8 ``phase`` column of ``state``."""

    POWERPLAY = 0
    MIDDLE = 1
    DEATH = 2


_N_PHASES = len(Phase)

#: Columns of ``balls`` that the State_Builder reads.
INPUT_COLUMNS: tuple[str, ...] = (
    # order keys and the Delivery's identity
    "row", "match_id", "innings", "innings_ball_num", "overid", "striker_id", "bowler_id",
    # outcomes and the columns derived from them, read from earlier Deliveries only
    "runs", "is_four", "is_six", "extras_type", "cls", "team_runs", "counted_wicket",
    # match values
    "sched_balls", "target",
)  # fmt: skip

#: The recent-window columns (criterion 7), which recent_window recomputes for any window length.
WINDOW_COLUMNS: tuple[str, ...] = ("win_lb", "win_wkts", "win_rr")

#: The pre-ball feature columns of the ``state`` table, after ``row``, in the design's order.
PRE_BALL_COLUMNS: tuple[str, ...] = (
    "lb_bowled", "lb_left", "score", "wickets", "crr", "runs_needed", "rrr", "ball_share", "phase", "cell",
    *WINDOW_COLUMNS,
    "striker_lb_faced", "striker_dots", "bowler_over_runs",
    "prev_flag", "prev_in_over_flag", "prev_in_over_boundary",
    "lb_since_boundary", "lb_since_wicket",
)  # fmt: skip

# Output columns stored as int32; phase and cell are int8, runs_needed is nullable Int32.
_INT32_COLUMNS: frozenset[str] = frozenset(
    {
        "row", "lb_bowled", "lb_left", "score", "wickets", "win_lb", "win_wkts", "striker_lb_faced",
        "striker_dots", "bowler_over_runs", "lb_since_boundary", "lb_since_wicket",
    }
)  # fmt: skip

# Integer input columns that may not be missing on an innings 1-2 Delivery.
_WHOLE_COLUMNS: tuple[str, ...] = (
    "row", "match_id", "innings_ball_num", "overid", "striker_id", "bowler_id", "runs", "team_runs", "sched_balls",
)  # fmt: skip
_FLAG_COLUMNS: tuple[str, ...] = ("is_four", "is_six", "counted_wicket")


# ----------------------------------------------------------------------------------- input checks

def _rows(count: int) -> str:
    return f"{count:,} row" if count == 1 else f"{count:,} rows"


def _whole(
    frame: pd.DataFrame, column: str, problems: list[str], *, may_be_missing: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """``column`` as int64 values (0 where missing) and a missing mask; each problem is appended to ``problems``."""
    series = frame[column]
    missing = series.isna().to_numpy(dtype=bool)
    if pd.api.types.is_integer_dtype(series.dtype):
        values = series.to_numpy(dtype=np.int64, na_value=0)
        bad = np.zeros(len(series), dtype=bool)
    else:
        floats = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
        with np.errstate(invalid="ignore"):
            whole = np.isfinite(floats) & (np.floor(floats) == floats) & (np.abs(floats) <= 2.0**53)
        bad = ~missing & ~whole
        values = np.where(whole, floats, 0.0).astype(np.int64)
    if bad.any():
        problems.append(f"{column} is not a whole number in {_rows(int(bad.sum()))}")
    if missing.any() and not may_be_missing:
        problems.append(f"{column} is missing in {_rows(int(missing.sum()))}")
    return values, missing


def _flag(frame: pd.DataFrame, column: str) -> np.ndarray:
    """A boolean column as a bool array; a missing entry of a nullable flag is False."""
    series = frame[column]
    if not pd.api.types.is_bool_dtype(series.dtype):
        raise TypeError(f"balls {column} must be boolean; got dtype {series.dtype}")
    return series.to_numpy(dtype=bool, na_value=False)


def _check_window(window: Any) -> int:
    """The recent-window length as an int; TypeError or ValueError unless it is a whole number of at least 1."""
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)):
        raise TypeError(f"window must be a whole number of Legal_Balls (D6); got {window!r}")
    if window < 1:
        raise ValueError(f"window must be at least 1 Legal_Ball (D6); got {window}")
    return int(window)


def _check_splits(phase_splits: Any) -> tuple[float, float]:
    """The two D8 Phase splits as floats; TypeError or ValueError unless they are finite with 0 < first < second."""
    values = list(phase_splits) if isinstance(phase_splits, (Sequence, np.ndarray)) else None
    if (
        isinstance(phase_splits, (str, bytes))
        or values is None
        or len(values) != 2
        or any(isinstance(value, bool) or not isinstance(value, Real) for value in values)
    ):
        raise TypeError(
            f"phase_splits must be two shares of Scheduled_Balls (D8), such as [0.30, 0.75]; got {phase_splits!r}"
        )
    first, second = float(values[0]), float(values[1])
    if not (math.isfinite(first) and math.isfinite(second) and 0.0 < first < second):
        raise ValueError(f"phase_splits must be finite shares with 0 < first < second (D8); got {phase_splits!r}")
    return first, second


def _int_values(values: Any, name: str) -> np.ndarray:
    """An array-like of integers with no missing value as int64; TypeError otherwise."""
    series = values if isinstance(values, pd.Series) else pd.Series(np.asarray(values))
    if not pd.api.types.is_integer_dtype(series.dtype) or series.isna().any():
        raise TypeError(f"{name} must hold integers with no missing value; got dtype {series.dtype}")
    return series.to_numpy(dtype=np.int64)


# ------------------------------------------------------------------------------- Delivery arrays

class _Deliveries(NamedTuple):
    """The innings 1-2 Deliveries in the Requirement 5 order, as numpy arrays of equal length."""

    row: np.ndarray  # int64: the ``row`` join key
    match_id: np.ndarray  # int64
    innings: np.ndarray  # int64: 1 or 2
    overid: np.ndarray  # int64
    striker: np.ndarray  # int64: striker_id
    bowler: np.ndarray  # int64: bowler_id
    start: np.ndarray  # int64: position of the first Delivery of the Delivery's innings
    legal: np.ndarray  # bool: Legal_Ball
    team_runs: np.ndarray  # int64: team runs under the selected Extras_Encoding
    counted: np.ndarray  # bool: Counted_Wicket
    bat_runs: np.ndarray  # int64: runs
    boundary: np.ndarray  # bool: is_four or is_six (a Boundary)
    extra: np.ndarray  # bool: extras_type other than "None"
    bye: np.ndarray  # bool: extras_type byes or legbyes
    sched: np.ndarray  # int64: Scheduled_Balls
    target: np.ndarray  # int64: Target, 0 where missing
    has_target: np.ndarray  # bool: the Target is present


def _deliveries(balls: pd.DataFrame, context: str) -> _Deliveries:
    """The innings 1-2 Deliveries of ``balls`` in the Requirement 5 order; DataError (``context``) on bad data."""
    if not isinstance(balls, pd.DataFrame):
        raise TypeError(f"balls must be a DataFrame; got {type(balls).__name__}")
    absent = [column for column in INPUT_COLUMNS if column not in balls.columns]
    if absent:
        raise ValueError(f"balls lacks the column(s) {', '.join(absent)}")
    problems: list[str] = []
    innings, _ = _whole(balls, "innings", problems)
    if problems:
        raise DataError("\n".join([f"{context}:"] + [f"  - {problem}" for problem in problems]))
    keep = np.isin(innings, STATE_INNINGS)
    frame, innings = balls.iloc[np.flatnonzero(keep)], innings[keep]

    whole = {column: _whole(frame, column, problems)[0] for column in _WHOLE_COLUMNS}
    cls, _ = _whole(frame, "cls", problems)
    target, target_missing = _whole(frame, "target", problems, may_be_missing=True)
    extras_type = frame["extras_type"]
    no_extras_type = extras_type.isna().to_numpy(dtype=bool)
    if no_extras_type.any():
        problems.append(f"extras_type is missing in {_rows(int(no_extras_type.sum()))}")
    not_positive = whole["sched_balls"] <= 0
    if not_positive.any():
        problems.append(f"sched_balls is not a positive number in {_rows(int(not_positive.sum()))}")
    if problems:
        raise DataError("\n".join([f"{context}:"] + [f"  - {problem}" for problem in problems]))
    if not np.isin(cls, [int(member) for member in DeliveryClass]).all():
        raise ValueError(
            "cls must hold DeliveryClass codes 0 (Legal_Ball), 1 (Wide) or 2 (No_Ball), as classify returns"
        )
    flags = {column: _flag(frame, column) for column in _FLAG_COLUMNS}

    match_id, ball = whole["match_id"], whole["innings_ball_num"]
    order = np.lexsort((ball, innings, match_id))  # stable: equal keys keep their input order
    match_id, innings = match_id[order], innings[order]
    n = len(order)
    first = np.ones(n, dtype=bool)
    first[1:] = (match_id[1:] != match_id[:-1]) | (innings[1:] != innings[:-1])
    plain = extras_type.isin([_NO_EXTRA]).to_numpy(dtype=bool)
    return _Deliveries(
        row=whole["row"][order],
        match_id=match_id,
        innings=innings,
        overid=whole["overid"][order],
        striker=whole["striker_id"][order],
        bowler=whole["bowler_id"][order],
        start=np.maximum.accumulate(np.where(first, np.arange(n), 0)),
        legal=(cls == DeliveryClass.LEGAL)[order],
        team_runs=whole["team_runs"][order],
        counted=flags["counted_wicket"][order],
        bat_runs=whole["runs"][order],
        boundary=(flags["is_four"] | flags["is_six"])[order],
        extra=~plain[order],
        bye=extras_type.isin(BYE_EXTRAS_TYPES).to_numpy(dtype=bool)[order],
        sched=whole["sched_balls"][order],
        target=target[order],
        has_target=~target_missing[order],
    )


# ----------------------------------------------------------------------------- grouped running sums

def _shift_down(values: np.ndarray, fill: Any) -> np.ndarray:
    """``values`` moved one row down, with ``fill`` in the first row: each row sees the row before it."""
    shifted = np.empty_like(values)
    shifted[:1] = fill
    shifted[1:] = values[:-1]
    return shifted


def _prefix_sums(values: np.ndarray) -> np.ndarray:
    """int64 prefix sums with a leading 0: entry p is the sum over the positions before p."""
    return np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(values, dtype=np.int64)))


class _Groups:
    """Deliveries grouped by equal keys, each group in the Requirement 5 order.

    The Deliveries arrive in the Requirement 5 order and np.lexsort is stable, so sorting by the
    keys alone keeps that order inside every group. Each method takes one value per Delivery in
    the Requirement 5 order and returns one result per Delivery in the same order.
    """

    __slots__ = ("_first", "_order", "_start")

    def __init__(self, *keys: np.ndarray) -> None:
        order = np.lexsort(keys[::-1])  # the first key is the most significant
        n = len(order)
        first = np.zeros(n, dtype=bool)
        first[:1] = True
        for key in keys:
            ordered = key[order]
            first[1:] |= ordered[1:] != ordered[:-1]
        self._order = order
        self._first = first  # in sorted order: the first Delivery of each group
        self._start = np.maximum.accumulate(np.where(first, np.arange(n), 0))  # sorted position of the group's first

    def _restore(self, values: np.ndarray) -> np.ndarray:
        restored = np.empty_like(values)
        restored[self._order] = values
        return restored

    def earlier_sum(self, values: np.ndarray) -> np.ndarray:
        """Sum of ``values`` over the earlier Deliveries of each Delivery's group; 0 for a group's first."""
        ordered = np.asarray(values, dtype=np.int64)[self._order]
        before = np.cumsum(ordered) - ordered
        return self._restore(before - before[self._start])

    def count_since(self, counted: np.ndarray, event: np.ndarray) -> np.ndarray:
        """Sum of ``counted`` over the earlier Deliveries of the group after its most recent earlier ``event``.

        The sum runs over every earlier Delivery of the group when none of them is an ``event``.
        """
        ordered = np.asarray(counted, dtype=np.int64)[self._order]
        through = np.cumsum(ordered)
        before = through - ordered
        # Running count through the most recent earlier event anywhere; one from an earlier group is
        # at most the count before this group's first Delivery, so the maximum below discards it.
        at_event = _shift_down(np.maximum.accumulate(np.where(event[self._order], through, -1)), -1)
        return self._restore(before - np.maximum(at_event, before[self._start]))

    def previous(self, values: np.ndarray, fill: Any) -> np.ndarray:
        """Value of the previous Delivery of each Delivery's group; ``fill`` for a group's first Delivery."""
        shifted = _shift_down(np.asarray(values)[self._order], fill)
        shifted[self._first] = fill
        return self._restore(shifted)


def _per_over(runs: np.ndarray, balls: np.ndarray, defined: np.ndarray | None = None) -> np.ndarray:
    """6 x runs / balls in runs per over (float64); NaN where ``balls`` is 0 or ``defined`` is False."""
    rate = np.full(len(runs), np.nan)
    ok = balls > 0 if defined is None else defined & (balls > 0)
    rate[ok] = BALLS_PER_OVER * runs[ok] / balls[ok]
    return rate


def _window(d: _Deliveries, window: int) -> dict[str, np.ndarray]:
    """The WINDOW_COLUMNS of every Delivery for a recent window of ``window`` Legal_Balls (criterion 7)."""
    n = len(d.row)
    position = np.arange(n)
    legal = d.legal.astype(np.int64)
    legal_before = np.cumsum(legal) - legal  # Legal_Balls before each position, over all innings
    lb = legal_before - legal_before[d.start]  # L: Legal_Balls before the Delivery in its innings
    start = d.start.copy()  # the window is the whole innings so far while L < window
    full = lb >= window
    if full.any():
        # Legal_Ball number L - window + 1 of the innings is Legal_Ball number legal_before - window + 1
        # of the whole table, whose position is entry legal_before - window of the Legal_Ball positions.
        start[full] = np.flatnonzero(d.legal)[legal_before[full] - window]
    legal_sum, run_sum, wicket_sum = (_prefix_sums(values) for values in (legal, d.team_runs, d.counted))
    win_lb = legal_sum[position] - legal_sum[start]
    return {
        "win_lb": win_lb,
        "win_wkts": wicket_sum[position] - wicket_sum[start],
        "win_rr": _per_over(run_sum[position] - run_sum[start], win_lb),
    }


def _frame(columns: Mapping[str, Any], names: Sequence[str]) -> pd.DataFrame:
    """The output DataFrame with ``names`` in order, the integer counts as int32."""
    return pd.DataFrame(
        {name: columns[name].astype(np.int32) if name in _INT32_COLUMNS else columns[name] for name in names}
    )


# ------------------------------------------------------------------------------- Phase and cells

def ball_position_share(lb_bowled: Any, sched_balls: Any) -> np.ndarray:
    """Ball_Position_Share of each Delivery: (Legal_Balls bowled before it + 1) / Scheduled_Balls.

    ``lb_bowled`` and ``sched_balls`` are equal-length arrays or Series of integers. Returns a
    float64 array. Raises TypeError for non-integer or missing values, and ValueError when the
    lengths differ or a Scheduled_Balls value is not positive.
    """
    lb = _int_values(lb_bowled, "lb_bowled")
    sb = _int_values(sched_balls, "sched_balls")
    if lb.shape != sb.shape:
        raise ValueError(f"lb_bowled and sched_balls must have the same length; got {len(lb):,} and {len(sb):,}")
    if (sb <= 0).any():
        raise ValueError("sched_balls must be positive")
    return (lb + 1) / sb


def phase_codes(share: Any, phase_splits: Sequence[float]) -> np.ndarray:
    """Phase code (``Phase``) of each Ball_Position_Share under the D8 splits (criterion 5), as int8.

    ``phase_splits`` is the Config value ``design_choices.D8_phase_splits`` (two shares s1 < s2) or
    an alternative from ``sensitivity.D8``. A Delivery is powerplay (0) when its share is at most
    s1, middle (1) when the share is above s1 and at most s2, and death (2) when it is above s2.
    A share that equals a split exactly, such as 36 / 120 = 0.30, compares as equal, because both
    are the float64 value nearest to the same fraction. With Scheduled_Balls 120, splits 0.30 and
    0.75 and complete 6-Legal_Ball Overs, ``overid`` 1-6 is powerplay, 7-15 middle and 16-20
    death (criterion 6).

    Raises TypeError or ValueError for splits that are not two finite shares with 0 < s1 < s2,
    and ValueError for a share that is not finite.
    """
    first, second = _check_splits(phase_splits)
    values = np.asarray(share, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("share must hold finite Ball_Position_Shares")
    codes = np.where(values <= first, Phase.POWERPLAY, np.where(values <= second, Phase.MIDDLE, Phase.DEATH))
    return codes.astype(np.int8)


def cell_codes(innings: Any, phase: Any) -> np.ndarray:
    """Innings x Phase cell of each Delivery (D13): 3 x (innings - 1) + phase, as int8.

    Cells 0, 1 and 2 are the powerplay, middle and death of innings 1; cells 3, 4 and 5 those of
    innings 2. Raises ValueError for an innings other than 1 or 2, a Phase code other than 0-2,
    or arrays of different lengths, and TypeError for non-integer or missing values.
    """
    innings_values = _int_values(innings, "innings")
    phase_values = _int_values(phase, "phase")
    if innings_values.shape != phase_values.shape:
        raise ValueError(
            f"innings and phase must have the same length; got {len(innings_values):,} and {len(phase_values):,}"
        )
    if not np.isin(innings_values, STATE_INNINGS).all():
        raise ValueError("innings must be 1 or 2: a Super_Over_Innings has no innings x Phase cell")
    if not np.isin(phase_values, [int(member) for member in Phase]).all():
        raise ValueError("phase must hold Phase codes 0 (powerplay), 1 (middle) or 2 (death)")
    return ((innings_values - 1) * _N_PHASES + phase_values).astype(np.int8)


# --------------------------------------------------------------------------------- the features

def recent_window(balls: pd.DataFrame, *, window: int) -> pd.DataFrame:
    """The recent-window columns of every innings 1-2 Delivery, for a window of ``window`` Legal_Balls.

    ``window`` is D6 (``design_choices.D6_window``) for the ``state`` table, or one of the
    Robustness_Suite's alternative windows (``sensitivity.D6``). The window of a Delivery runs
    from its ``window``-th most recent earlier Legal_Ball of the innings through the previous
    Delivery, Wides and No_Balls in that span included, or over every earlier Delivery of the
    innings when fewer than ``window`` Legal_Balls precede it (criterion 7).

    Returns ``row`` (int32) and the WINDOW_COLUMNS on a RangeIndex, in the row order and with the
    dtypes of pre_ball_features: ``win_lb`` and ``win_wkts`` int32 and ``win_rr`` float64 (NaN
    when ``win_lb`` is 0). With ``window`` equal to D6 the columns equal those of
    pre_ball_features.

    Raises TypeError or ValueError when ``window`` is not a whole number of at least 1, and the
    errors of pre_ball_features for bad input.
    """
    length = _check_window(window)
    d = _deliveries(balls, "The recent-window features cannot be computed")
    return _frame({"row": d.row, **_window(d, length)}, ("row", *WINDOW_COLUMNS))


def pre_ball_features(balls: pd.DataFrame, *, window: int, phase_splits: Sequence[float]) -> pd.DataFrame:
    """The pre-ball features of every innings 1-2 Delivery (Requirement 6, criteria 1-11).

    ``balls`` needs the INPUT_COLUMNS (the ``balls`` table of the ``reconstruct`` checkpoint has
    them). ``window`` is D6 (``design_choices.D6_window``, 18 Legal_Balls as delivered) and
    ``phase_splits`` is D8 (``design_choices.D8_phase_splits``, [0.30, 0.75] as delivered).

    Returns one row per Delivery of innings 1 and 2, in the Requirement 5 order, on a RangeIndex,
    with ``row`` (int32, the join key) and the PRE_BALL_COLUMNS defined in the module docstring.
    Each feature uses only the earlier Deliveries of the same innings, the Delivery's own
    identity columns and its match's Scheduled_Balls and Target. Undefined features are missing
    (NaN in ``crr``, ``rrr`` and ``win_rr``, <NA> in ``runs_needed``), never 0 or infinite.

    Raises DataError (exit code 3) for the data problems listed in the module docstring, naming
    each with its row count. Raises TypeError or ValueError when ``balls`` is not a DataFrame or
    lacks an input column, when a flag column is not boolean, when ``cls`` holds a code that is
    not a DeliveryClass, or when ``window`` or ``phase_splits`` is invalid.
    """
    length = _check_window(window)
    splits = _check_splits(phase_splits)
    d = _deliveries(balls, "The pre-ball state cannot be built")
    innings = _Groups(d.match_id, d.innings)
    striker = _Groups(d.match_id, d.innings, d.striker)
    over = _Groups(d.match_id, d.innings, d.overid)
    bowler_over = _Groups(d.match_id, d.innings, d.overid, d.bowler)

    lb = innings.earlier_sum(d.legal)
    score = innings.earlier_sum(d.team_runs)
    lb_left = np.maximum(d.sched - lb, 0)
    chase = (d.innings == 2) & d.has_target  # runs needed is defined only here
    runs_needed = np.where(chase, d.target - score, 0)
    share = ball_position_share(lb, d.sched)
    phase = phase_codes(share, splits)
    boundary_or_extra = d.boundary | d.extra
    columns = {
        "row": d.row,
        "lb_bowled": lb,
        "lb_left": lb_left,
        "score": score,
        "wickets": innings.earlier_sum(d.counted),
        "crr": _per_over(score, lb),
        "runs_needed": pd.arrays.IntegerArray(runs_needed.astype(np.int32), ~chase),
        "rrr": _per_over(runs_needed, lb_left, defined=chase),
        "ball_share": share,
        "phase": phase,
        "cell": cell_codes(d.innings, phase),
        **_window(d, length),
        "striker_lb_faced": striker.earlier_sum(d.legal),
        "striker_dots": striker.count_since(d.legal, d.legal & (d.bat_runs != 0)),
        "bowler_over_runs": bowler_over.earlier_sum(np.where(d.bye, 0, d.team_runs)),
        "prev_flag": innings.previous(boundary_or_extra, False),
        "prev_in_over_flag": over.previous(boundary_or_extra, False),
        "prev_in_over_boundary": over.previous(d.boundary, False),
        "lb_since_boundary": innings.count_since(d.legal, d.boundary),
        "lb_since_wicket": innings.count_since(d.legal, d.counted),
    }
    return _frame(columns, ("row", *PRE_BALL_COLUMNS))


# =================================================================== Batters_To_Come and the stage
#
# Task 6.2 appends the rest of the State_Builder: Batters_To_Come (criteria 4 and 12-14), the
# bowler's Modal_Bowling_Type with the striker's hand x bowling-style key, the full ``state`` table,
# the match table of criterion 17 and the ``state`` Pipeline_Stage. The imports below serve only
# this part of the module.

from pathlib import Path  # noqa: E402
from typing import TYPE_CHECKING  # noqa: E402

if TYPE_CHECKING:
    from superr.pipeline import StageContext

__all__ += [
    "BTC_COLUMNS",
    "BTC_INPUT_COLUMNS",
    "BTC_SLOT_COLUMNS",
    "HAND_STYLE_SEPARATOR",
    "MATCHUP_COLUMNS",
    "MATCHUP_INPUT_COLUMNS",
    "MATCH_KEY_COLUMNS",
    "MATCH_TABLE_FILE",
    "MAX_BATTERS",
    "MODAL_COLUMNS",
    "NO_PLAYER",
    "REPORT_MATCH_ID",
    "STAGE_NAME",
    "STATE_COLUMNS",
    "STATE_FOLDER",
    "STATE_TABLE",
    "XI_COLUMNS",
    "batters_to_come",
    "bowling_matchup",
    "build_state",
    "run_stage",
    "state_table_for_match",
    "write_state_table",
]

#: The most players a side can bat in an innings, fixed by the Laws of Cricket rather than a
#: Method_Constant. The Batters_To_Come cap is MAX_BATTERS minus the players who have batted, so
#: Batters_To_Come never holds more than MAX_BATTERS players and fits the MAX_BATTERS slot columns.
MAX_BATTERS = 11

#: The player ID that pads the Batters_To_Come slot columns, and the ``next_batter`` of a Delivery
#: whose Batters_To_Come is empty. Every real ``player_id`` is 0 or more.
NO_PLAYER = -1

#: The Batters_To_Come columns of the ``state`` table (criteria 12-14), in the design's order.
BTC_SLOT_COLUMNS: tuple[str, ...] = tuple(f"btc_{slot}" for slot in range(1, MAX_BATTERS + 1))
BTC_COLUMNS: tuple[str, ...] = (*BTC_SLOT_COLUMNS, "btc_count", "next_batter", "batted_count")

#: The bowler's Modal_Bowling_Type and the striker's ``batting_type`` x that type (the Outcome_Model's
#: ``hand_x_style``, Requirement 11, criterion 2).
MATCHUP_COLUMNS: tuple[str, ...] = ("modal_bowling_type", "hand_x_style")

#: Text between the two parts of ``hand_x_style``, as in ``RHB|Right-arm Fast``.
HAND_STYLE_SEPARATOR = "|"

#: Every column of the ``state`` table after ``row``, in the design's order (Data Models, ``state``).
STATE_COLUMNS: tuple[str, ...] = (*PRE_BALL_COLUMNS, *BTC_COLUMNS, *MATCHUP_COLUMNS)

#: Columns of ``balls`` that batters_to_come reads: the order keys and the people of each Delivery.
BTC_INPUT_COLUMNS: tuple[str, ...] = (
    "row", "match_id", "innings", "innings_ball_num", "batting_team_id", "striker_id", "non_striker_id", "bowler_id",
)  # fmt: skip

#: Columns of ``balls`` that bowling_matchup reads.
MATCHUP_INPUT_COLUMNS: tuple[str, ...] = ("row", "match_id", "innings", "innings_ball_num", "bowler_id", "batting_type")

#: Columns of the ``playing_eleven`` table (``load`` checkpoint) that batters_to_come reads.
XI_COLUMNS: tuple[str, ...] = (
    "match_id", "team_id", "player_id", "batting_order", "is_played", "is_impact", "is_impact_off",
)  # fmt: skip

#: Columns of the ``modal_bowling_type`` table of the ``audit`` checkpoint.
MODAL_COLUMNS: tuple[str, ...] = ("bowler_id", "modal_bowling_type")

#: The columns that state_table_for_match takes from ``balls``, before the state features.
MATCH_KEY_COLUMNS: tuple[str, ...] = ("match_id", "innings", "innings_ball_num")

#: Name of the Pipeline_Stage that runs the State_Builder, of its checkpoint and of its table.
STAGE_NAME = "state"
STATE_TABLE = "state"

#: Folder under the outputs folder that holds the State_Builder's file, and that file's name.
STATE_FOLDER = "state"
MATCH_TABLE_FILE = "state_{match_id}.csv"

#: The match whose state table every Run writes (Requirement 6, criterion 17): the IPL 2025 final.
REPORT_MATCH_ID = 40767

_LOAD_STAGE = "load"
_XI_TABLE = "playing_eleven"  # the load checkpoint's table of playing_eleven.csv
_IMPACT_FLAGS: tuple[str, ...] = ("is_impact", "is_impact_off")
_LAST_ORDER = np.iinfo(np.int64).max  # sort key of a missing batting_order: after every present one
_INT32_MAX = int(np.iinfo(np.int32).max)
_SHOWN_SIDES = 10  # sides named in one Run_Log warning about playing_eleven rows


def _raise_problems(context: str, problems: list[str]) -> None:
    """Raise one DataError (exit code 3) under ``context`` that lists ``problems``, if there are any."""
    if problems:
        raise DataError("\n".join([f"{context}:"] + [f"  - {problem}" for problem in problems]))


def _ordered_frame(balls: pd.DataFrame, columns: Sequence[str], context: str) -> pd.DataFrame:
    """The innings 1-2 rows of ``balls`` in the Requirement 5 order, exactly as pre_ball_features orders them.

    The sort is stable, so Deliveries with equal keys keep their input order. Raises TypeError when
    ``balls`` is not a DataFrame, ValueError when it lacks one of ``columns``, and DataError
    (``context``) when an order key is missing or not a whole number.
    """
    if not isinstance(balls, pd.DataFrame):
        raise TypeError(f"balls must be a DataFrame; got {type(balls).__name__}")
    absent = [column for column in columns if column not in balls.columns]
    if absent:
        raise ValueError(f"balls lacks the column(s) {', '.join(absent)}")
    problems: list[str] = []
    innings, _ = _whole(balls, "innings", problems)
    _raise_problems(context, problems)
    keep = np.flatnonzero(np.isin(innings, STATE_INNINGS))
    frame = balls.iloc[keep]
    match_id, _ = _whole(frame, "match_id", problems)
    ball, _ = _whole(frame, "innings_ball_num", problems)
    _raise_problems(context, problems)
    return frame.iloc[np.lexsort((ball, innings[keep], match_id))]


class _Lineup(NamedTuple):
    """The people of the innings 1-2 Deliveries in the Requirement 5 order, as int64 arrays of equal length."""

    row: np.ndarray
    match_id: np.ndarray
    innings: np.ndarray
    team: np.ndarray  # batting_team_id, 0 where missing
    has_team: np.ndarray  # bool: batting_team_id is present
    striker: np.ndarray
    non_striker: np.ndarray  # 0 where missing
    has_non_striker: np.ndarray  # bool: non_striker_id is present
    bowler: np.ndarray


def _lineup(balls: pd.DataFrame, context: str) -> _Lineup:
    """The BTC_INPUT_COLUMNS of the innings 1-2 Deliveries; DataError (``context``) on bad data."""
    frame = _ordered_frame(balls, BTC_INPUT_COLUMNS, context)
    problems: list[str] = []
    row, match_id, innings, striker, bowler = (
        _whole(frame, column, problems)[0] for column in ("row", "match_id", "innings", "striker_id", "bowler_id")
    )
    team, no_team = _whole(frame, "batting_team_id", problems, may_be_missing=True)
    non_striker, no_non_striker = _whole(frame, "non_striker_id", problems, may_be_missing=True)
    _raise_problems(context, problems)
    return _Lineup(row, match_id, innings, team, ~no_team, striker, non_striker, ~no_non_striker, bowler)


class _Slots(NamedTuple):
    """Each side's ``is_played`` players in the Batters_To_Come order, as dense (sides + 1, width) arrays.

    Row ``len(keys)`` is an empty side for the Deliveries whose batting side has no ``is_played``
    row. Slot k of a side is its player k in the Batters_To_Come order; unused slots are invalid.
    """

    keys: pd.MultiIndex  # (match_id, team_id) of each side
    player: np.ndarray  # int64 player_id; NO_PLAYER in unused slots
    valid: np.ndarray  # bool: the slot holds a player
    impact: np.ndarray  # bool: is_impact = 1
    keep_order: np.ndarray  # int64: the slots by (is_impact_off, slot), unused slots last


def _xi_slots(xi: pd.DataFrame, context: str) -> _Slots:
    """The ``is_played`` players of every side of ``xi``, one slot per player, in the Batters_To_Come order."""
    if not isinstance(xi, pd.DataFrame):
        raise TypeError(f"xi must be a DataFrame; got {type(xi).__name__}")
    absent = [column for column in XI_COLUMNS if column not in xi.columns]
    if absent:
        raise ValueError(f"xi lacks the column(s) {', '.join(absent)}")
    if not pd.api.types.is_bool_dtype(xi["is_played"].dtype):
        raise TypeError(f"xi is_played must be boolean; got dtype {xi['is_played'].dtype}")
    played = xi.iloc[np.flatnonzero(xi["is_played"].to_numpy(dtype=bool, na_value=False))]
    problems: list[str] = []
    match_id, team, player = (_whole(played, column, problems)[0] for column in ("match_id", "team_id", "player_id"))
    order, no_order = _whole(played, "batting_order", problems, may_be_missing=True)
    flags = {}
    for column in _IMPACT_FLAGS:
        values, missing = _whole(played, column, problems, may_be_missing=True)
        bad = ~missing & (values != 0) & (values != 1)
        if bad.any():
            problems.append(f"{column} is neither 0 nor 1 in {_rows(int(bad.sum()))}")
        flags[column] = ~missing & (values == 1)
    outside = (player < 0) | (player > _INT32_MAX)
    if outside.any():
        problems.append(f"player_id is negative or above {_INT32_MAX:,} in {_rows(int(outside.sum()))}")
    _raise_problems(context, problems)

    # One entry per player of a side: his lowest batting_order, either flag when any row has it.
    per_player = (
        pd.DataFrame(
            {
                "match_id": match_id,
                "team_id": team,
                "player_id": player,
                "order": np.where(no_order, _LAST_ORDER, order),
                "impact": flags["is_impact"],
                "off": flags["is_impact_off"],
                "position": np.arange(len(player)),
            }
        )
        .groupby(["match_id", "team_id", "player_id"], sort=False)
        .agg(order=("order", "min"), impact=("impact", "max"), off=("off", "max"), position=("position", "min"))
        .reset_index()
    )
    match_id, team, player, order, position = (
        per_player[column].to_numpy(dtype=np.int64) for column in ("match_id", "team_id", "player_id", "order", "position")
    )
    by_side = np.lexsort((position, order, team, match_id))
    match_id, team, player = match_id[by_side], team[by_side], player[by_side]
    impact = per_player["impact"].to_numpy(dtype=bool)[by_side]
    off = per_player["off"].to_numpy(dtype=bool)[by_side]

    new_side = np.ones(len(player), dtype=bool)
    new_side[1:] = (match_id[1:] != match_id[:-1]) | (team[1:] != team[:-1])
    side_start = np.flatnonzero(new_side)
    side = np.cumsum(new_side) - 1
    slot = np.arange(len(player)) - side_start[side]
    width = int(slot.max()) + 1 if len(slot) else 1
    shape = (len(side_start) + 1, width)  # the last row is the empty side
    player_slots = np.full(shape, NO_PLAYER, dtype=np.int64)
    valid, impact_slots, off_slots = (np.zeros(shape, dtype=bool) for _ in range(3))
    player_slots[side, slot] = player
    valid[side, slot] = True
    impact_slots[side, slot] = impact
    off_slots[side, slot] = off
    return _Slots(
        keys=pd.MultiIndex.from_arrays([match_id[side_start], team[side_start]], names=["match_id", "team_id"]),
        player=player_slots,
        valid=valid,
        impact=impact_slots,
        keep_order=np.argsort(off_slots.astype(np.int64) + 2 * ~valid, axis=1, kind="stable"),
    )


def _first_positions(groups: np.ndarray, players: np.ndarray, positions: np.ndarray) -> pd.Series:
    """The earliest of ``positions`` for each (group, player) pair, on a (group, player) MultiIndex."""
    return pd.Series(positions, dtype=np.int64).groupby([groups, players], sort=False).min()


def _position_of(first: pd.Series, groups: np.ndarray, players: np.ndarray, never: int) -> np.ndarray:
    """The ``first`` position of each (group, player) pair as int64; ``never`` for a pair that ``first`` lacks."""
    if first.empty:
        return np.full(len(players), never, dtype=np.int64)
    at = first.index.get_indexer(pd.MultiIndex.from_arrays([groups, players]))
    return np.where(at >= 0, first.to_numpy(dtype=np.int64)[np.maximum(at, 0)], never)


def _nullable_int32(values: np.ndarray, missing: np.ndarray) -> pd.arrays.IntegerArray:
    return pd.arrays.IntegerArray(values.astype(np.int32), missing.copy())


def batters_to_come(balls: pd.DataFrame, xi: pd.DataFrame) -> pd.DataFrame:
    """Batters_To_Come of every innings 1-2 Delivery (Requirement 6, criteria 4 and 12-14).

    ``balls`` needs the BTC_INPUT_COLUMNS (the ``balls`` table of the ``reconstruct`` checkpoint
    has them) and ``xi`` the XI_COLUMNS (the ``playing_eleven`` table of the ``load`` checkpoint).

    Rules
        * The candidates of a Delivery are the batting side's ``xi`` rows: the Delivery's
          ``match_id``, ``team_id`` equal to its ``batting_team_id`` and ``is_played`` true (a
          missing ``is_played`` counts as false) (criterion 12).
        * A player has batted once he is the ``striker_id`` or ``non_striker_id`` of the Delivery
          or of an earlier Delivery of the innings (criterion 12). A retired batter has batted.
        * A candidate with ``is_impact`` = 1 qualifies only on the Deliveries after his first
          appearance as striker, non-striker or bowler in any Delivery of the match, in the
          Requirement 5 order, so an appearance in innings 1 comes before every Delivery of
          innings 2 (criterion 13). An Impact Player who never appears never qualifies.
        * Batters_To_Come holds the qualifying candidates who have not batted, capped at
          MAX_BATTERS minus ``batted_count``. When more qualify, those with ``is_impact_off`` = 1
          are removed first, then those with the highest ``batting_order``, until the count
          equals the cap (criterion 14).

    ``xi`` data rules
        A player with several ``is_played`` rows for the same match and team counts once, with his
        lowest ``batting_order`` and with ``is_impact`` or ``is_impact_off`` = 1 when any of his
        rows has it. Each side's players are ordered by ``batting_order``; a missing
        ``batting_order`` comes after every present one, and ties keep the order of the ``xi``
        rows. A missing ``is_impact`` or ``is_impact_off`` is 0. Ties in the cap's removal order
        follow the same player order, so the later player is removed first.

    Returns one row per Delivery of innings 1 and 2, in the Requirement 5 order and the row order
    of pre_ball_features, on a RangeIndex, with ``row`` (int32) and the BTC_COLUMNS (nullable
    Int32):

    ==================== ========================================================================
    Column               Value
    ==================== ========================================================================
    btc_1 ... btc_11     the player IDs of Batters_To_Come in ``batting_order``, then NO_PLAYER (-1)
    btc_count            the number of players in Batters_To_Come
    next_batter          ``btc_1``: the Batters_To_Come player with the lowest ``batting_order``
                         (the next batter of the Post_Wicket_State); NO_PLAYER when it is empty
    batted_count         the distinct players seen as striker or non-striker in the innings
                         through the Delivery, including any player missing from ``xi``
    ==================== ========================================================================

    Every BTC column is missing (<NA>) on a Delivery whose batting side has no ``is_played`` row
    in ``xi`` or whose ``batting_team_id`` is missing (criterion 4); the Delivery's other state
    features are unaffected.

    Vectorised rule
        No loop runs over Deliveries. Each side's players fill the slots of a dense
        (Deliveries x slots) array, as wide as the largest side. For every run of Deliveries with
        the same match, innings and batting side, each slot gets the position of its player's
        first appearance as a batter in the innings (an unused slot never qualifies) and, for an
        Impact Player, of his first appearance in any role in the match. A slot qualifies on the
        Delivery at position i when its batter position is after i and its Impact Player
        position, if any, is before i. The cap keeps the qualifying slots whose running count, in
        the side's (``is_impact_off``, player order) order, is at most the cap.

    Raises DataError (exit code 3), naming each problem with its row count, when an order key,
    ``row``, ``striker_id`` or ``bowler_id`` of an innings 1-2 Delivery is missing or not a whole
    number, or when an ``is_played`` row of ``xi`` has a missing or non-whole ``match_id``,
    ``team_id`` or ``player_id``, a ``player_id`` below 0 or above the int32 range, or an
    ``is_impact`` or ``is_impact_off`` other than 0 or 1. Raises TypeError or ValueError when
    ``balls`` or ``xi`` is not a DataFrame, lacks an input column, or ``is_played`` is not boolean.
    """
    d = _lineup(balls, "Batters_To_Come cannot be computed")
    slots = _xi_slots(xi, "playing_eleven cannot supply Batters_To_Come")
    n = len(d.row)
    if not n:
        empty = np.zeros(0, dtype=np.int64)
        return _btc_frame(empty, np.zeros((0, MAX_BATTERS), dtype=np.int64), empty, empty, np.zeros(0, dtype=bool))
    n_sides, width = len(slots.keys), slots.player.shape[1]
    position = np.arange(n)

    # The batting side of each Delivery; the empty side n_sides when the XI has no row for it.
    side = slots.keys.get_indexer(pd.MultiIndex.from_arrays([d.match_id, d.team]))
    side = np.where(d.has_team & (side >= 0), side, n_sides)

    # Innings, and runs of Deliveries with the same match, innings and batting side ("segments").
    innings_start = np.ones(n, dtype=bool)
    innings_start[1:] = (d.match_id[1:] != d.match_id[:-1]) | (d.innings[1:] != d.innings[:-1])
    innings_id = np.cumsum(innings_start) - 1
    segment_start = innings_start.copy()
    segment_start[1:] |= side[1:] != side[:-1]
    segment = np.cumsum(segment_start) - 1
    first_row = np.flatnonzero(segment_start)
    segment_side = side[first_row]

    # First appearances: as a batter in the innings, and in any role in the match.
    as_batter = d.has_non_striker
    batter_innings = np.concatenate((innings_id, innings_id[as_batter]))
    batters = np.concatenate((d.striker, d.non_striker[as_batter]))
    batter_positions = np.concatenate((position, position[as_batter]))
    first_bat = _first_positions(batter_innings, batters, batter_positions)
    first_any = _first_positions(
        np.concatenate((d.match_id, d.match_id[as_batter], d.match_id)),
        np.concatenate((d.striker, d.non_striker[as_batter], d.bowler)),
        np.concatenate((position, position[as_batter], position)),
    )

    # Players who have batted, counting the current Delivery, and the cap.
    new_batters = np.bincount(first_bat.to_numpy(dtype=np.int64), minlength=n)
    through = np.cumsum(new_batters)
    batted = through - (through - new_batters)[np.flatnonzero(innings_start)][innings_id]
    cap = np.maximum(MAX_BATTERS - batted, 0)

    # Per segment and slot: the position from which the slot's player has batted (-1 for a slot
    # that never qualifies) and the position after which he may enter (-1 unless an Impact Player).
    segment_player = slots.player[segment_side]
    query = segment_player.ravel()
    bats_from = _position_of(first_bat, np.repeat(innings_id[first_row], width), query, never=n).reshape(-1, width)
    bats_from = np.where(slots.valid[segment_side], bats_from, -1)
    enters_after = _position_of(first_any, np.repeat(d.match_id[first_row], width), query, never=n).reshape(-1, width)
    enters_after = np.where(slots.impact[segment_side], enters_after, -1)

    # Qualifying slots of each Delivery, then the cap in the side's removal order.
    here = position[:, None]
    qualifies = (bats_from[segment] > here) & (enters_after[segment] < here)
    keep_order = slots.keep_order[segment_side][segment]
    ranked = np.take_along_axis(qualifies, keep_order, axis=1)
    kept = np.zeros_like(qualifies)
    np.put_along_axis(kept, keep_order, ranked & (np.cumsum(ranked, axis=1) <= cap[:, None]), axis=1)

    # Kept players in batting_order, left-aligned in the slot columns.
    filled = np.full((n, MAX_BATTERS), NO_PLAYER, dtype=np.int64)
    rows, columns = np.nonzero(kept)
    filled[rows, (np.cumsum(kept, axis=1) - 1)[rows, columns]] = segment_player[segment[rows], columns]
    return _btc_frame(d.row, filled, kept.sum(axis=1), batted, side == n_sides)


def _btc_frame(
    row: np.ndarray, filled: np.ndarray, count: np.ndarray, batted: np.ndarray, missing: np.ndarray
) -> pd.DataFrame:
    """The batters_to_come table: ``row`` and the BTC_COLUMNS, every BTC column missing where ``missing``."""
    result: dict[str, Any] = {"row": row.astype(np.int32)}
    for index, name in enumerate(BTC_SLOT_COLUMNS):
        result[name] = _nullable_int32(filled[:, index], missing)
    result["btc_count"] = _nullable_int32(count, missing)
    result["next_batter"] = _nullable_int32(filled[:, 0], missing)
    result["batted_count"] = _nullable_int32(batted, missing)
    return pd.DataFrame(result)


def bowling_matchup(balls: pd.DataFrame, modal_bowling_type: pd.DataFrame) -> pd.DataFrame:
    """The bowler's Modal_Bowling_Type and the striker's hand x bowling style of every innings 1-2 Delivery.

    ``balls`` needs the MATCHUP_INPUT_COLUMNS and ``modal_bowling_type`` the MODAL_COLUMNS, one row
    per ``bowler_id`` (the ``modal_bowling_type`` table of the ``audit`` checkpoint, which applies
    the glossary's latest-Delivery tie rule).

    Returns ``row`` (int32) and the MATCHUP_COLUMNS (``string``), in the row order of
    pre_ball_features, on a RangeIndex:

    * ``modal_bowling_type``: the Modal_Bowling_Type of the Delivery's ``bowler_id``; missing when
      the bowler has none or is not in ``modal_bowling_type``;
    * ``hand_x_style``: the Delivery's ``batting_type`` (the striker's batting hand), then
      HAND_STYLE_SEPARATOR, then ``modal_bowling_type``, such as ``RHB|Right-arm Fast``; missing
      when either part is missing.

    Raises DataError when an order key, ``row`` or ``bowler_id`` of an innings 1-2 Delivery, or a
    ``bowler_id`` of ``modal_bowling_type``, is missing or not a whole number; ValueError when a
    table lacks an input column or ``modal_bowling_type`` repeats a ``bowler_id``; TypeError when a
    table is not a DataFrame.
    """
    context = "The bowling matchup features cannot be computed"
    frame = _ordered_frame(balls, MATCHUP_INPUT_COLUMNS, context)
    if not isinstance(modal_bowling_type, pd.DataFrame):
        raise TypeError(f"modal_bowling_type must be a DataFrame; got {type(modal_bowling_type).__name__}")
    absent = [column for column in MODAL_COLUMNS if column not in modal_bowling_type.columns]
    if absent:
        raise ValueError(f"modal_bowling_type lacks the column(s) {', '.join(absent)}")
    problems: list[str] = []
    row, bowler = (_whole(frame, column, problems)[0] for column in ("row", "bowler_id"))
    known, _ = _whole(modal_bowling_type, "bowler_id", problems)
    _raise_problems(context, problems)
    index = pd.Index(known)
    if not index.is_unique:
        raise ValueError("modal_bowling_type must hold one row per bowler_id")
    style = modal_bowling_type["modal_bowling_type"].astype("string").array.take(
        index.get_indexer(bowler), allow_fill=True
    )
    hand = pd.Series(frame["batting_type"].astype("string").array)
    return pd.DataFrame(
        {
            "row": row.astype(np.int32),
            "modal_bowling_type": style,
            "hand_x_style": (hand + HAND_STYLE_SEPARATOR + pd.Series(style)).array,
        }
    )


def build_state(
    balls: pd.DataFrame,
    xi: pd.DataFrame,
    modal_bowling_type: pd.DataFrame,
    *,
    window: int,
    phase_splits: Sequence[float],
) -> pd.DataFrame:
    """The ``state`` table: the pre-ball state of every innings 1-2 Delivery (Requirement 6, criteria 1-14).

    ``balls`` is the ``balls`` table of the ``reconstruct`` checkpoint, ``xi`` the ``playing_eleven``
    table of the ``load`` checkpoint and ``modal_bowling_type`` the table of that name in the
    ``audit`` checkpoint. ``window`` is D6 (``design_choices.D6_window``) and ``phase_splits`` D8
    (``design_choices.D8_phase_splits``).

    Returns one row per Delivery of innings 1 and 2, in the Requirement 5 order, on a RangeIndex,
    keyed by ``row`` (int32), with the STATE_COLUMNS: the pre_ball_features columns, the
    batters_to_come columns and the bowling_matchup columns, with their dtypes. Raises the errors
    of those three functions.
    """
    pre_ball = pre_ball_features(balls, window=window, phase_splits=phase_splits)
    parts = [pre_ball]
    for part in (batters_to_come(balls, xi), bowling_matchup(balls, modal_bowling_type)):
        if not np.array_equal(part["row"].to_numpy(), pre_ball["row"].to_numpy()):
            raise RuntimeError("the state features are not in the same Delivery order")  # cannot happen
        parts.append(part.drop(columns="row"))
    return pd.concat(parts, axis=1)


def _state_keys(state: pd.DataFrame, balls: pd.DataFrame) -> pd.DataFrame:
    """The MATCH_KEY_COLUMNS of the ``balls`` row with each state row's ``row``, on a RangeIndex."""
    if not isinstance(state, pd.DataFrame) or not isinstance(balls, pd.DataFrame):
        raise TypeError("state and balls must be DataFrames")
    if "row" not in state.columns:
        raise ValueError("state lacks the column row")
    absent = [column for column in ("row", *MATCH_KEY_COLUMNS) if column not in balls.columns]
    if absent:
        raise ValueError(f"balls lacks the column(s) {', '.join(absent)}")
    index = pd.Index(_int_values(balls["row"], "balls row"))
    if not index.is_unique:
        raise ValueError("balls must hold each row once")
    at = index.get_indexer(_int_values(state["row"], "state row"))
    if (at < 0).any():
        raise ValueError(f"{_rows(int((at < 0).sum()))} of state have no Delivery in balls")
    return balls.iloc[at][list(MATCH_KEY_COLUMNS)].reset_index(drop=True)


def state_table_for_match(state: pd.DataFrame, balls: pd.DataFrame, match_id: int) -> pd.DataFrame:
    """The state table of one match (Requirement 6, criterion 17).

    ``state`` is the build_state table and ``balls`` the table it was built from, which supplies
    the MATCH_KEY_COLUMNS through ``row``. Returns one row per state row of the match (its
    Deliveries of innings 1 and 2) in the Requirement 5 order, on a RangeIndex, with the
    MATCH_KEY_COLUMNS followed by every state column except ``row``. The table is empty when the
    match has no state row. Raises ValueError when a state row has no ``balls`` row, ``balls``
    repeats a ``row`` or a table lacks a needed column, and TypeError for a non-DataFrame.
    """
    keys = _state_keys(state, balls)
    chosen = np.flatnonzero((keys["match_id"] == match_id).to_numpy(dtype=bool, na_value=False))
    features = [column for column in state.columns if column != "row" and column not in MATCH_KEY_COLUMNS]
    table = pd.concat(
        [keys.iloc[chosen].reset_index(drop=True), state.iloc[chosen][features].reset_index(drop=True)], axis=1
    )
    return table.sort_values(["innings", "innings_ball_num"], kind="stable").reset_index(drop=True)


def write_state_table(outdir: str | Path, table: pd.DataFrame, match_id: int) -> Path:
    """Write ``table`` (state_table_for_match) to ``state_<match_id>.csv`` in ``outdir``; return the path.

    ``outdir`` is the state folder (``outputs/state/`` in a Run) and is created when absent. The
    file has a header row and one row per row of ``table``, in its column order: UTF-8, ``\\n``
    line ends, floats at full precision, flags as ``True`` or ``False`` and a missing value as an
    empty field. It is written through a temporary file renamed in place, so it replaces any
    earlier copy whole.
    """
    if not isinstance(table, pd.DataFrame):
        raise TypeError(f"table must be a DataFrame; got {type(table).__name__}")
    folder = Path(outdir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / MATCH_TABLE_FILE.format(match_id=match_id)
    temporary = path.with_name(path.name + ".tmp")
    try:
        table.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _sides_text(sides: pd.DataFrame) -> str:
    shown = ", ".join(f"match {m} team {t}" for m, t in sides.head(_SHOWN_SIDES).itertuples(index=False))
    return shown + (f" and {len(sides) - _SHOWN_SIDES:,} more" if len(sides) > _SHOWN_SIDES else "")


def _xi_notes(xi: pd.DataFrame) -> list[str]:
    """Run_Log warnings about ``is_played`` rows of ``xi`` that the Batters_To_Come order has to interpret."""
    played = xi.iloc[np.flatnonzero(xi["is_played"].to_numpy(dtype=bool, na_value=False))]
    notes = []
    repeated = played.duplicated(["match_id", "team_id", "player_id"], keep=False).to_numpy(dtype=bool)
    if repeated.any():
        sides = played.loc[repeated, ["match_id", "team_id"]].drop_duplicates()
        notes.append(
            f"{len(sides):,} side(s) list a player in more than one is_played row of playing_eleven "
            f"({_sides_text(sides)}); each such player counts once in Batters_To_Come, at his lowest batting_order"
        )
    no_order = played["batting_order"].isna().to_numpy(dtype=bool)
    if no_order.any():
        sides = played.loc[no_order, ["match_id", "team_id"]].drop_duplicates()
        notes.append(
            f"{len(sides):,} side(s) have is_played rows without a batting_order in playing_eleven "
            f"({_rows(int(no_order.sum()))}: {_sides_text(sides)}); Batters_To_Come puts those players after "
            "the others, in the order of playing_eleven.csv, so next_batter for those sides is uncertain"
        )
    return notes


def _shown(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def run_stage(ctx: StageContext) -> None:
    """The ``state`` Pipeline_Stage: build the pre-ball state of every Delivery and save it.

    Reads ``balls`` from the ``reconstruct`` checkpoint, ``playing_eleven`` from the ``load``
    checkpoint and ``modal_bowling_type`` from the ``audit`` checkpoint, and the Config values D6
    (``design_choices.D6_window``) and D8 (``design_choices.D8_phase_splits``). Builds the
    build_state table, writes the state table of match REPORT_MATCH_ID (the IPL 2025 final) to
    ``outputs/state/state_40767.csv`` (Requirement 6, criterion 17) and saves the ``state`` table
    (STATE_TABLE) to the checkpoint. The Run_Log gets the row and innings counts, a warning for
    innings whose Batters_To_Come is missing (criterion 4), and a warning for ``playing_eleven``
    sides with repeated players or missing batting orders.

    Raises DataError (exit code 3) for the data problems of build_state and when match
    REPORT_MATCH_ID has no Delivery in innings 1 or 2.
    """
    from superr import audit, reconstruct  # for the checkpoint names of the stages this stage reads

    settings, store, log = ctx.settings, ctx.store, ctx.log
    window = settings.get("design_choices.D6_window")
    phase_splits = settings.get("design_choices.D8_phase_splits")
    balls = store.load_table(reconstruct.STAGE_NAME, reconstruct.BALLS_TABLE)
    xi = store.load_table(_LOAD_STAGE, _XI_TABLE)
    modal = store.load_table(audit.STAGE_NAME, audit.MODAL_TABLE)

    state = build_state(balls, xi, modal, window=window, phase_splits=phase_splits)
    match_table = state_table_for_match(state, balls, REPORT_MATCH_ID)
    if match_table.empty:
        raise DataError(
            f"match {REPORT_MATCH_ID} (the IPL 2025 final) has no Delivery in innings 1 or 2, so its state "
            "table (Requirement 6, criterion 17) cannot be written"
        )
    path = write_state_table(settings.path("outputs") / STATE_FOLDER, match_table, REPORT_MATCH_ID)
    store.save_table(STAGE_NAME, STATE_TABLE, state)

    innings = _state_keys(state, balls)[["match_id", "innings"]]
    no_xi = state["btc_count"].isna().to_numpy(dtype=bool)
    n_innings, n_no_xi = len(innings.drop_duplicates()), len(innings.loc[no_xi].drop_duplicates())
    log.info(
        f"built the state table: {len(state):,} Deliveries of innings 1 and 2 in {n_innings:,} innings, "
        f"{len(STATE_COLUMNS)} state features (D6 window {window} Legal_Balls, D8 Phase splits "
        f"{list(phase_splits)}); Batters_To_Come is missing in {n_no_xi:,} innings (Requirement 6, criterion 4)",
        stage=STAGE_NAME,
    )
    if n_no_xi:
        log.warning(
            f"Batters_To_Come is missing for {int(no_xi.sum()):,} Deliveries in {n_no_xi:,} innings whose batting "
            "side has no is_played row in playing_eleven (Requirement 6, criterion 4)",
            stage=STAGE_NAME,
        )
    for note in _xi_notes(xi):
        log.warning(note, stage=STAGE_NAME)
    log.info(
        f"wrote {_shown(path, ctx.root)}: {len(match_table):,} Deliveries of match {REPORT_MATCH_ID}, the IPL "
        "2025 final (Requirement 6, criterion 17)",
        stage=STAGE_NAME,
    )
