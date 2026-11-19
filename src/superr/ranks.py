"""Percentile ranks (glossary, Percentile_Rank; Requirement 12, criteria 12, 15 and 16; Requirement
13, criteria 2 and 4; design section "Pressure_Engine (``pressure.py``, ``ranks.py``)" and
Property 34), and the comparison of two rankings (Requirement 21, criterion 5; Requirement 22,
criteria 2 and 6; Requirement 23, criterion 11; Requirement 25, criterion 14; Property 54).

Every percentile rank of the Pipeline uses the glossary definition: the Percentile_Rank of a value
x within a reference set of n values is (the number of reference values below x + 0.5 x the
number of reference values equal to x) / n. No function reads ``config.py``: the definition has no
Config setting.

The rank (``percentile_rank``)
    ``percentile_rank(values, reference)`` sorts the reference once and counts, for each value x,
    the reference values below x with ``numpy.searchsorted(side="left")`` and the reference values
    at or below x with ``numpy.searchsorted(side="right")``; the difference is the count equal to
    x. The same formula serves values inside and outside the reference set:

    * a value of the set counts itself among the equal values (the glossary's "including the
      Delivery's own"), so its rank lies between 0.5 / n and 1 - 0.5 / n, inside (0, 1);
    * a value outside the set is ranked against it: 0 below every reference value, 1 above every
      one, and (below + 0.5 x equal) / n in between. This gives a Dead_Ball its rank against the
      Live_Ball reference set (Requirement 12, criterion 12).

    Equal values get equal ranks, and the rank never decreases as x grows. A rank depends only on x
    and the reference values, not on their order or on the other values ranked in the same call.
    The counts are whole numbers and below + 0.5 x equal is exact in float64, so each rank is the
    correctly rounded quotient: bit for bit the value of a brute-force count (Property 34). The
    values are compared as float64, so -0.0 equals 0.0 and a flag ranks as 0 or 1.

Ranks within groups (``percentile_rank_within``)
    The Pressure_Engine ranks each Role's Deliveries against reference sets that differ by
    Innings_Group: the Pillar_Reference_Set (D4: the Role's Live_Balls in the Innings_Group;
    Requirement 12, criterion 12), the ranks of LI and Stress inside PI (Requirement 13, criterion
    2) and the rank of PI behind the Pressure_Ball flag (criterion 4).
    ``percentile_rank_within(values, groups, reference)`` takes one value, one group label and one
    reference flag per row, and ranks every row against the reference rows of its own group with
    the rule of ``percentile_rank``. For one Role the caller passes the Role's Deliveries
    (Batter_Deliveries or Bowler_Deliveries), their innings as ``groups`` (D4 ``innings_group``;
    one label for every row in the pooled case of Requirement 21, criterion 4) and the Live_Ball
    flag as ``reference``. A row outside the reference set, such as a Dead_Ball, gets its rank
    against the set like any other value. The rows are sorted by group once; each group then
    costs one sort of its reference values and two ``searchsorted`` calls.

Missing values (Requirement 12, criterion 16)
    A rank needs a number. ``percentile_rank`` raises ValueError for a missing value (NaN or NA) in
    either argument. ``percentile_rank_within`` does the same by default, which suits LI, Stress
    and PI: Requirement 13, criterion 10 stops the Run before any of them is missing on a Live_Ball.
    With ``allow_missing=True`` it applies Requirement 12, criterion 16 to a Pillar_Component
    instead: a row whose value is missing leaves the reference set of its group and gets the rank
    MISSING_RANK (0.5), and the other rows of the group are ranked against the reference rows that
    have a value. The caller counts the missing values for its report (``numpy.isnan`` of the
    values, per Pillar_Component, Role and Innings_Group). An infinite value always raises
    ValueError: a Pillar_Component whose Legal_Ball window or Legal_Balls left is 0 is missing and
    must be passed as NaN, not as a division by 0.

Comparing two rankings (``compare_rankings``)
    ``compare_rankings(main, other)`` compares a ranking with a reference ranking of the same
    Role: the Main_Ranking against a Robustness_Case (Requirement 21, criterion 5), a
    Cross_Check_Ranking (Requirement 22, criteria 2 and 6), a model-choice rerun (Requirement 23,
    criterion 11) or the Activity_Comparison_Ranking (Requirement 25, criterion 14). Each ranking
    is a table with ``player_id`` and ``rank``. A player is in a ranking exactly when its rank is
    a number; a missing rank (NaN or NA) marks a player who is not an Eligible_Player there or is
    unranked. A full-data ranking ranks every Eligible_Player, so its ranked players are its
    Eligible_Players. Ranks are whole numbers from 1 and no two players share one: the tie-break
    rules make every ranking of the Pipeline a total order (Requirement 18, criterion 13;
    Requirement 22). They need not be consecutive. The result is a RankingComparison:

    * ``n_main`` and ``n_other``: the players each ranking ranks; ``n_common``: those ranked in both.
    * ``main_top`` and ``other_top``: each ranking's ``top_n`` (10) best-ranked players, best first;
      a ranking with fewer players gives all of them. ``overlap`` is the number of ``player_id``s
      in both tops, from 0 to ``top_n``. ``entering`` holds the players in ``other``'s top but not
      ``main``'s, in ``other``'s rank order, and ``leaving`` those in ``main``'s top but not
      ``other``'s, in ``main``'s rank order. Neither shares a player with the overlap, and when
      both rankings rank ``top_n`` or more players each holds ``top_n`` - ``overlap`` players
      (``len(leaving)`` is the count of Requirement 23, criterion 11).
    * ``spearman``: Spearman's rank correlation over the common players only (players ranked in
      both), of their values in the column ``on`` of each table. ``on`` is ``rank`` by default
      (the positions, Requirement 22, criterion 2), or for example ``final_score`` (Requirement 21,
      criterion 5). Each table's values are ranked among the common players, tied values sharing
      the mean of their positions (average ranks), and the result is the Pearson correlation of
      the two sets of ranks, as ``scipy.stats.spearmanr`` computes it, clipped to [-1, 1] against
      rounding. It is NaN (undefined) with fewer than MIN_COMMON_PLAYERS (3) common players, the
      not-comparable limit of Requirement 21, criterion 8, or when either set of ranks is constant
      (zero variance). The correlation depends only on order, and in both rankings rank 1 has the
      highest Final_Score, so ``on="rank"`` and ``on="final_score"`` agree unless two common
      players' Final_Scores are equal at the 6 decimals of the Rating_Engine's tie-break.
    * ``changes``: one row per entering player, then one per leaving player, in the orders above
      (CHANGE_COLUMNS): ``status`` (ENTERING or LEAVING), ``player_id``, ``player_name`` (from
      ``names``, a mapping or a function of the ``player_id``; missing without ``names``) and the
      player's rank in ``main`` and in ``other`` (``main_rank``, ``other_rank``; Int64, <NA> where
      that ranking does not rank the player).
    * ``comparable``: both rankings rank ``top_n`` or more players and MIN_COMMON_PLAYERS or more
      are in both (Requirement 21, criterion 8). ``compare_rankings`` reports every value either
      way; a caller such as the robustness table blanks what its rule asks.

Errors: TypeError for an argument of the wrong kind (not 1-D, not numbers, a reference that is not
a boolean mask); ValueError for an empty reference set (n = 0), arguments of different lengths, a
missing value where none is allowed, an infinite value, a missing group label, or a group that has
a value to rank but no reference value. ``compare_rankings`` raises TypeError for an argument of
the wrong kind and ValueError for a missing column, a missing or repeated ``player_id``, a rank
that is not a whole number from 1 or is shared, a common player without a finite ``on`` value in
both tables, a ``top_n`` below 1 or a missing name.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

__all__ = [
    "CHANGE_COLUMNS",
    "ENTERING",
    "LEAVING",
    "MIN_COMMON_PLAYERS",
    "MISSING_RANK",
    "RankingComparison",
    "compare_rankings",
    "percentile_rank",
    "percentile_rank_within",
]


#: The Percentile_Rank of a Pillar_Component value that cannot be computed (Requirement 12, criterion 16).
MISSING_RANK = 0.5

_SHOWN = 5  # group labels named in one error message


# ---------------------------------------------------------------------------------- small helpers

def _one_dimensional(values: Any, what: str) -> pd.Series:
    if values is None or isinstance(values, (str, bytes, Mapping, pd.DataFrame)) or np.ndim(values) != 1:
        raise TypeError(f"{what} must be a 1-D array or Series; got {type(values).__name__}")
    return values if isinstance(values, pd.Series) else pd.Series(values)


def _numbers(values: Any, what: str) -> np.ndarray:
    """``values`` (1-D numbers or flags) as a float64 array, NaN for a missing value; no infinite value."""
    series = _one_dimensional(values, what)
    if not len(series):
        return np.empty(0, dtype=np.float64)  # an empty list has no dtype to check
    dtype = series.dtype
    if not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_complex_dtype(dtype):
        raise TypeError(f"{what} must hold numbers; got dtype {dtype}")
    array = series.to_numpy(dtype=np.float64, na_value=np.nan)
    infinite = int(np.isinf(array).sum())
    if infinite:
        raise ValueError(
            f"{what} holds {infinite:,} infinite value(s); a value that cannot be computed is missing (NaN), "
            "not infinite (Requirement 12, criterion 16)"
        )
    return array


def _complete(array: np.ndarray, what: str) -> None:
    missing = int(np.isnan(array).sum())
    if missing:
        raise ValueError(f"{what} holds {missing:,} missing value(s); a Percentile_Rank needs a number")


def _mask(values: Any, what: str, n_rows: int) -> np.ndarray:
    """``values`` (1-D booleans, none missing) as a bool array of ``n_rows`` values."""
    series = _one_dimensional(values, what)
    if not len(series) and n_rows == 0:
        return np.zeros(0, dtype=bool)  # an empty list has no dtype to check
    if not pd.api.types.is_bool_dtype(series.dtype):
        raise TypeError(f"{what} must be a boolean mask, one flag per row; got dtype {series.dtype}")
    if series.isna().any():
        raise ValueError(f"{what} has {int(series.isna().sum()):,} missing flag(s)")
    if len(series) != n_rows:
        raise ValueError(f"{what} has {len(series):,} flags, but {n_rows:,} are needed, one per row")
    return series.to_numpy(dtype=bool)


def _group_codes(groups: Any, n_rows: int) -> tuple[np.ndarray, list[Any]]:
    """Each row's group as a code 0 .. G - 1 (int64), and the G labels in sorted order."""
    series = _one_dimensional(groups, "groups")
    if len(series) != n_rows:
        raise ValueError(f"groups has {len(series):,} labels, but {n_rows:,} are needed, one per row")
    codes, labels = pd.factorize(series, sort=True)
    missing = int((codes < 0).sum())
    if missing:
        raise ValueError(f"groups has {missing:,} missing label(s); every row needs a group")
    return codes.astype(np.int64, copy=False), pd.Index(labels).tolist()


def _ranks(sorted_reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """(below + 0.5 x equal) / n of each value against the ascending, non-empty ``sorted_reference``."""
    below = np.searchsorted(sorted_reference, values, side="left")
    at_or_below = np.searchsorted(sorted_reference, values, side="right")
    return (below + 0.5 * (at_or_below - below)) / len(sorted_reference)


# ========================================================================================= ranks

def percentile_rank(values: Any, reference: Any) -> np.ndarray:
    """The Percentile_Rank of each of ``values`` within ``reference`` (module docstring, "The rank").

    ``values``: the 1-D numbers (or flags) to rank, any number of them, each inside or outside the
    reference set. ``reference``: the 1-D numbers of the reference set, n >= 1 of them. Neither may
    hold a missing or infinite value; apply the missing-value rule first (``percentile_rank_within``
    with ``allow_missing=True`` does). Returns a new float64 array with one rank per value, in the
    order given: (the count of reference values below x + 0.5 x the count equal to x) / n, within
    [0, 1]. Raises TypeError for an argument that is not 1-D numbers, and ValueError for an empty
    reference, a missing value or an infinite value.
    """
    queries = _numbers(values, "values")
    pool = _numbers(reference, "reference")
    _complete(queries, "values")
    _complete(pool, "reference")
    if not len(pool):
        raise ValueError("reference holds no value; a Percentile_Rank needs a reference set of n >= 1 values")
    return _ranks(np.sort(pool), queries)


def percentile_rank_within(values: Any, groups: Any, reference: Any, *, allow_missing: bool = False) -> np.ndarray:
    """The Percentile_Rank of each row's value within the reference rows of its group (module docstring).

    ``values``: one number (or flag) per row. ``groups``: one label per row (whole numbers such as
    the innings, or text), none missing. ``reference``: a boolean mask, True for the rows of the
    reference sets (the Live_Balls). Every row is ranked against the reference rows of its group,
    whether or not it is one of them, with the rule of percentile_rank. With ``allow_missing``
    False (the default), a missing value raises ValueError. With ``allow_missing`` True, a missing
    value is the Pillar_Component rule of Requirement 12, criterion 16: the row leaves the
    reference set and gets MISSING_RANK (0.5), and the other rows are ranked against the reference
    rows with a value. Returns a new float64 array, one rank per row in the order given. Raises
    TypeError for an argument of the wrong kind, and ValueError for arguments of different lengths,
    an infinite value, a missing value or group label that is not allowed, or a group that has a
    value to rank but no reference value.
    """
    if not isinstance(allow_missing, (bool, np.bool_)):
        raise TypeError(f"allow_missing must be True or False; got {allow_missing!r}")
    numbers = _numbers(values, "values")
    n_rows = len(numbers)
    codes, labels = _group_codes(groups, n_rows)
    in_reference = _mask(reference, "reference", n_rows)
    if not allow_missing:
        _complete(numbers, "values")
    has_value = ~np.isnan(numbers)
    usable = in_reference & has_value  # the reference rows that enter the reference sets

    out = np.full(n_rows, MISSING_RANK)
    order = np.argsort(codes, kind="stable")  # the rows of each group, ascending, one block per group
    bounds = np.searchsorted(codes[order], np.arange(len(labels) + 1), side="left")
    empty: list[str] = []
    for code, label in enumerate(labels):
        rows = order[bounds[code] : bounds[code + 1]]
        ranked = rows[has_value[rows]]
        if not len(ranked):
            continue  # every value of the group is missing: each keeps MISSING_RANK
        pool = numbers[rows[usable[rows]]]
        if not len(pool):
            empty.append(f"{label!r} ({len(ranked):,} value(s) to rank)")
            continue
        out[ranked] = _ranks(np.sort(pool), numbers[ranked])
    if empty:
        shown = ", ".join(empty[:_SHOWN]) + (f" and {len(empty) - _SHOWN:,} more" if len(empty) > _SHOWN else "")
        raise ValueError(
            f"{len(empty):,} group(s) have values to rank but no reference value, so their Percentile_Rank is "
            f"undefined: {shown}"
        )
    return out


# ========================================================================= comparing two rankings

#: Spearman's correlation needs at least this many players ranked in both rankings; with fewer, a
#: comparison is not comparable (Requirement 21, criterion 8).
MIN_COMMON_PLAYERS = 3

#: The ``status`` of a row of RankingComparison.changes.
ENTERING = "entering"
LEAVING = "leaving"

#: Columns of RankingComparison.changes, in order (module docstring, "Comparing two rankings").
CHANGE_COLUMNS: tuple[str, ...] = ("status", "player_id", "player_name", "main_rank", "other_rank")


@dataclass(frozen=True, eq=False)
class RankingComparison:
    """How a ranking ``other`` differs from a reference ranking ``main`` (``compare_rankings``).

    ``top_n``: the size of the top lists compared. ``n_main`` and ``n_other``: the players each
    ranking ranks (its Eligible_Players); ``n_common``: the players ranked in both. ``spearman``:
    Spearman's rank correlation over the common players, NaN when undefined. ``main_top`` and
    ``other_top``: the ``player_id`` of each ranking's top ``top_n``, best first (all of its
    players when it ranks fewer). ``overlap``: the players in both tops. ``entering``: the players
    of ``other_top`` not in ``main_top``, in ``other``'s rank order; ``leaving``: the players of
    ``main_top`` not in ``other_top``, in ``main``'s rank order. ``changes``: one row per entering
    player, then one per leaving player, in those orders (CHANGE_COLUMNS).
    """

    top_n: int
    n_main: int
    n_other: int
    n_common: int
    spearman: float
    main_top: tuple[int, ...]
    other_top: tuple[int, ...]
    overlap: int
    entering: tuple[int, ...]
    leaving: tuple[int, ...]
    changes: pd.DataFrame

    @property
    def comparable(self) -> bool:
        """True when both rankings rank ``top_n`` or more players and MIN_COMMON_PLAYERS or more are in both.

        The not-comparable rule of Requirement 21, criterion 8; the caller decides what to blank.
        """
        return min(self.n_main, self.n_other) >= self.top_n and self.n_common >= MIN_COMMON_PLAYERS

    def __repr__(self) -> str:
        rho = "undefined" if math.isnan(self.spearman) else f"{self.spearman:.6f}"
        return (
            f"RankingComparison(top {self.top_n}: overlap {self.overlap}, entering {list(self.entering)}, "
            f"leaving {list(self.leaving)}; Spearman {rho} over {self.n_common:,} common players; ranked "
            f"{self.n_main:,} (main) and {self.n_other:,} (other))"
        )


class _Ranking(NamedTuple):
    ids: np.ndarray  # int64 player_id of every row
    rank: np.ndarray  # float64 rank of every row, NaN where the ranking does not rank the player
    values: np.ndarray  # float64 values of the column that Spearman's correlation compares
    top: np.ndarray  # int64 player_id of the ranked rows, best rank first


def _ranking_numbers(values: pd.Series, what: str) -> np.ndarray:
    """``values`` (numbers) as float64, NaN where missing."""
    if not len(values):
        return np.empty(0, dtype=np.float64)  # an empty column has no dtype to check
    dtype = values.dtype
    if pd.api.types.is_bool_dtype(dtype) or not pd.api.types.is_numeric_dtype(dtype) or (
        pd.api.types.is_complex_dtype(dtype)
    ):
        raise TypeError(f"{what} must hold numbers; got dtype {dtype}")
    return values.to_numpy(dtype=np.float64, na_value=np.nan)


def _ranking_ids(values: pd.Series, what: str) -> np.ndarray:
    """``values`` as int64 ``player_id``s: whole numbers, none missing, each once."""
    numbers = _ranking_numbers(values, what)
    missing = int(np.isnan(numbers).sum())
    if missing:
        raise ValueError(f"{what} has {missing:,} missing value(s)")
    if pd.api.types.is_integer_dtype(values.dtype):
        ids = values.to_numpy(dtype=np.int64)
    elif not (np.isfinite(numbers).all() and (np.floor(numbers) == numbers).all()):
        raise ValueError(f"{what} must hold whole numbers")
    else:
        ids = numbers.astype(np.int64)
    unique, counts = np.unique(ids, return_counts=True)
    if (counts > 1).any():
        raise ValueError(f"{what} holds player_id {int(unique[counts > 1][0])} more than once")
    return ids


def _ranking_ranks(values: pd.Series, what: str) -> np.ndarray:
    """``values`` as float64 ranks: whole numbers from 1, none shared, NaN for a player not ranked."""
    rank = _ranking_numbers(values, what)
    present = rank[~np.isnan(rank)]
    if np.isinf(present).any():
        raise ValueError(
            f"{what} holds an infinite rank; a player the ranking does not rank has a missing rank (NaN or NA)"
        )
    if ((present < 1) | (np.floor(present) != present)).any():
        raise ValueError(
            f"{what} must hold whole numbers from 1, or a missing value for a player the ranking does not rank"
        )
    if len(np.unique(present)) != len(present):
        raise ValueError(f"{what} gives one rank to more than one player; a ranking must be a total order")
    return rank


def _ranking_table(frame: Any, what: str, on: str) -> _Ranking:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{what} must be a DataFrame with the columns player_id and rank; got {type(frame).__name__}")
    if not frame.columns.is_unique:
        raise ValueError(f"{what} has repeated column names")
    absent = [column for column in dict.fromkeys(("player_id", "rank", on)) if column not in frame.columns]
    if absent:
        raise ValueError(f"{what} lacks the column(s) {', '.join(absent)}")
    ids = _ranking_ids(frame["player_id"], f"{what} column 'player_id'")
    rank = _ranking_ranks(frame["rank"], f"{what} column 'rank'")
    values = rank if on == "rank" else _ranking_numbers(frame[on], f"{what} column {on!r}")
    ranked = np.flatnonzero(~np.isnan(rank))
    return _Ranking(ids, rank, values, ids[ranked[np.argsort(rank[ranked], kind="stable")]])


def _name_lookup(names: Any) -> Callable[[int], str | None]:
    """A function from ``player_id`` to its checked ``player_name`` (None for every player without ``names``)."""
    if names is None:
        return lambda player_id: None
    if isinstance(names, Mapping):

        def find(player_id: int) -> Any:
            if player_id not in names:
                raise ValueError(f"names has no player_name for player_id {player_id}")
            return names[player_id]

    elif callable(names):
        find = names
    else:
        raise TypeError(
            f"names must be a mapping or a function from player_id to player_name, or None; got {type(names).__name__}"
        )

    def name_of(player_id: int) -> str:
        name = find(player_id)
        if not isinstance(name, str):
            raise TypeError(f"names gives player_id {player_id} the player_name {name!r}, which is not a string")
        if not name:
            raise ValueError(f"names gives player_id {player_id} an empty player_name")
        return name

    return name_of


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman's rank correlation of paired values: the Pearson correlation of their average ranks.

    NaN with fewer than MIN_COMMON_PLAYERS pairs or when either set of ranks is constant.
    """
    if len(x) < MIN_COMMON_PLAYERS:
        return math.nan
    rx = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    dx, dy = rx - rx.mean(), ry - ry.mean()
    sxx, syy = float(dx @ dx), float(dy @ dy)
    if sxx == 0.0 or syy == 0.0:
        return math.nan
    return min(1.0, max(-1.0, float(dx @ dy) / math.sqrt(sxx * syy)))


def _rank_cell(value: float | None) -> Any:
    return pd.NA if value is None or math.isnan(value) else int(value)


def compare_rankings(
    main: pd.DataFrame,
    other: pd.DataFrame,
    *,
    top_n: int = 10,
    on: str = "rank",
    names: Mapping[int, str] | Callable[[int], str] | None = None,
) -> RankingComparison:
    """Compare the ranking ``other`` with the reference ranking ``main`` (module docstring, "Comparing two rankings").

    ``main``, ``other``: one Role's ranking each, with ``player_id`` (whole numbers, each once) and
    ``rank`` (whole numbers from 1, none shared; NaN or NA for a player the ranking does not rank,
    such as one who is not an Eligible_Player there). ``top_n``: the size of the top lists (10).
    ``on``: the column of both tables whose values Spearman's correlation compares: ``rank``
    (default) or, for example, ``final_score``. ``names``: a mapping or a function from
    ``player_id`` to the ``player_name`` written in ``changes``; None leaves the names missing.
    Returns a RankingComparison. Raises TypeError for an argument of the wrong kind, and ValueError
    for a missing column, a missing or repeated ``player_id``, a rank that is not a whole number
    from 1 or is shared, a common player without a finite ``on`` value in both tables, a ``top_n``
    below 1, or a missing name.
    """
    if isinstance(top_n, (bool, np.bool_)) or not isinstance(top_n, (int, np.integer)):
        raise TypeError(f"top_n must be a whole number; got {top_n!r}")
    if top_n < 1:
        raise ValueError(f"top_n must be 1 or more; got {top_n}")
    if not isinstance(on, str):
        raise TypeError(f"on must be the name of a column of both rankings; got {on!r}")
    first = _ranking_table(main, "main", on)
    second = _ranking_table(other, "other", on)
    name_of = _name_lookup(names)

    # Spearman's correlation over the players ranked in both rankings, and only them.
    in_first, in_second = ~np.isnan(first.rank), ~np.isnan(second.rank)
    common, at_first, at_second = np.intersect1d(
        first.ids[in_first], second.ids[in_second], assume_unique=True, return_indices=True
    )
    x, y = first.values[in_first][at_first], second.values[in_second][at_second]
    bad = ~(np.isfinite(x) & np.isfinite(y))
    if bad.any():
        raise ValueError(
            f"column {on!r} has no finite value in both rankings for player_id {int(common[bad][0])}, whom both rank; "
            "Spearman's correlation needs one in each"
        )

    # The top lists, the overlap and the players entering and leaving.
    main_top, other_top = first.top[: int(top_n)].tolist(), second.top[: int(top_n)].tolist()
    in_main_top, in_other_top = set(main_top), set(other_top)
    entering = tuple(pid for pid in other_top if pid not in in_main_top)
    leaving = tuple(pid for pid in main_top if pid not in in_other_top)
    rows = [(ENTERING, pid) for pid in entering] + [(LEAVING, pid) for pid in leaving]
    main_rank = dict(zip(first.ids.tolist(), first.rank.tolist()))
    other_rank = dict(zip(second.ids.tolist(), second.rank.tolist()))
    changes = pd.DataFrame(
        {
            "status": pd.array([status for status, _ in rows], dtype="str"),
            "player_id": np.asarray([pid for _, pid in rows], dtype=np.int64),
            "player_name": pd.array([name_of(pid) for _, pid in rows], dtype="str"),
            "main_rank": pd.array([_rank_cell(main_rank.get(pid)) for _, pid in rows], dtype="Int64"),
            "other_rank": pd.array([_rank_cell(other_rank.get(pid)) for _, pid in rows], dtype="Int64"),
        },
        columns=list(CHANGE_COLUMNS),
    )
    return RankingComparison(
        top_n=int(top_n),
        n_main=int(in_first.sum()),
        n_other=int(in_second.sum()),
        n_common=int(len(common)),
        spearman=_spearman(x, y),
        main_top=tuple(main_top),
        other_top=tuple(other_top),
        overlap=len(in_main_top & in_other_top),
        entering=entering,
        leaving=leaving,
        changes=changes,
    )
