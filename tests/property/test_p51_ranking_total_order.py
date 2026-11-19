# Feature: pressure-rating-system, Property 51: Ranking is a deterministic total order
"""Property 51: ranking is a deterministic total order (Requirement 18, criterion 13).

**Validates: Requirements 18.13**

For any table of Final_Scores (with ties to 6 decimal places), shrunk Clutch_Levels and
``player_id`` values, presented in any row order, the ranks are 1...n with no repeats, ordered by
Final_Score rounded to 6 decimals, then higher shrunk Clutch_Level, then lower ``player_id``.

The rule under test is the one ``superr.rating`` documents (module docstring, step 8, and
``rank_order``). The tie key is rint(Final_Score x 10^6), which equals
numpy.round(Final_Score, 6) x 10^6. The shrunk Clutch_Level is compared at full precision. A
Final_Score that is not finite is unranked: NaN from ``rank_order``, and <NA> in the Int64
``rank`` column that ``rank`` adds.

The 6-decimal boundary
    numpy.rint rounds half to even, so a Final_Score exactly half-way between two 6-decimal values
    ties with the even one: 0.0078125 counts as 0.007812, and 0.0234375 as 0.023438. The test
    asserts this rule on half-way values that float64 holds exactly: m / 128 with m odd, whose
    product with 10^6 (m x 7812.5) is exact too. Every other finite score is an 8-decimal number
    (100 K + j) / 10^8 with |j| <= 49, so it lies at least 10^-8 (0.01 key units) from a half-way
    point. For scores within +-60 runs per over, the float64 error of x x 10^6 stays below 10^-8
    key units, so the 6-decimal value K x 10^-6 of such a score is unambiguous. The test works out
    the 6-decimal value of each drawn score in integer arithmetic (``_six_decimal_key``), and check
    0 confirms that numpy.round(x, 6) and Python's correctly rounded round(x, 6) both give it.

``test_rank_order_is_a_deterministic_total_order`` (``rank_order`` and ``rank``)
    Each table (``rank_tables``) holds 0-24 players with distinct ``player_id`` values from 1 to
    10^7, in drawn row order. The keys (6-decimal value x 10^6) come from a run of 1-4 neighbouring
    keys K, K + 1, ... and 0-2 keys anywhere within +-60 runs per over. In half of the tables the
    run is the four keys around a half-way value m / 128, and that value is one of the scores a
    player can get. Each player gets:

    * a Final_Score: mostly an 8-decimal value of one of the keys, with j = 0 (so equal scores are
      frequent), j = +-49 (next to a half-way point) or any j from -49 to 49; sometimes the
      half-way value; now and then NaN, +inf or -inf;
    * a shrunk Clutch_Level: mostly one of 1-3 values that the table's players share, sometimes
      joined by a value one ulp above one of them (equal to 6 decimals, but higher); otherwise any
      value from -40 to 40. An unranked player's value may be NaN, as for a player outside the
      pool of ``rank_role``.

    The tables therefore hold exact ties, ties within 6 decimals (1.0000001 and 1.00000049),
    scores just outside a tie (1.00000049 and 1.00000051, 2 x 10^-8 apart; 1.0000004 and
    1.0000014, 10^-6 apart), half-way values, 6-decimal ties broken by the shrunk Clutch_Level,
    full ties broken by the ``player_id``, and unranked players. WORKED_EXAMPLE holds each of
    these cases, with its ranks worked out by hand. Checks:

    0. The construction: numpy.round(x, 6) and round(x, 6) equal the intended 6-decimal value of
       every finite drawn score x.
    1. ``rank_order`` returns one float64 rank per input row, NaN exactly where the Final_Score is
       not finite, and the ranked players hold 1, 2, ..., n once each.
    2. The ranks equal a brute-force sort of the ranked players by (-6-decimal Final_Score,
       -shrunk Clutch_Level, ``player_id``) on the intended 6-decimal values (and, for
       WORKED_EXAMPLE, the ranks worked out by hand).
    3. A second call gives identical ranks, and neither call changes its inputs.
    4. Row order: ``rank_order`` on the rows in a drawn order gives every player the same rank.
       ``rank`` on a DataFrame of those rows, indexed by their original positions, returns a copy
       with the same columns, index and values plus an Int64 ``rank`` column holding each player's
       rank (<NA> when unranked), and leaves its input unchanged.

``test_rank_role_orders_full_ties_by_player_id`` (``rank_role``)
    Each example is ``tests.strategies.contributions(force=(TIES,))``: 1-4 random players per Role
    plus the TIES player and his twin, whose entries are identical, so their Final_Scores and
    shrunk Clutch_Levels are equal bit for bit and only the ``player_id`` separates them. The twin
    has the lower id about half the time. Every player of the Role is passed as an Eligible_Player,
    so a random player with no Pressure_Ball has no Clutch_Level and is unranked. With so few
    players tau^2 is often 0, and then every ranked player of the Role has the same Final_Score.
    ``rank_role`` runs with the Config's D12 through its current keywords only (``role``,
    ``eligible`` and ``replacement_pct``, or a RoleData from ``prepare_role``). Checks, per Role:

    1. The twins are ranked, with equal Final_Scores and shrunk Clutch_Levels (the strategy's
       promise), and the one with the lower ``player_id`` ranks first.
    2. The ranks are NaN exactly where the Final_Score is not finite, the ranked players hold
       1, 2, ..., n once each, and the ranks equal the brute-force sort by
       (-numpy.round(Final_Score, 6), -shrunk Clutch_Level, ``player_id``).
    3. A second run, through ``prepare_role`` with the Eligible_Players in a drawn order, gives
       identical Final_Scores and ranks.
    4. ``rank`` on ``RoleRanking.to_frame()`` gives back the table's own ``rank`` column.

Not covered here
    * A finite Final_Score with a NaN shrunk Clutch_Level. ``rank_role`` cannot produce one,
      because Final_Score = w_L x C* + w_U x U* is NaN when C* is, and criterion 13 gives it no
      order.
    * Final_Scores far outside +-60 runs per over. From about 10^7, float64 cannot hold 8-decimal
      values apart, so the drawn values would stop being unambiguous.
    * ``player_id`` values above 2^53: ``rank_order`` reads the ids through float64.
    * The ranking under drawn match multiplicities (``rank_role``'s ``mult`` and
      ``rankable_only``, as the bootstrap uses them): Property 52.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import pandas as pd
from hypothesis import event, example, given
from hypothesis import strategies as st

from superr.rating import prepare_role, rank, rank_order, rank_role
from tests.strategies import TIES, ContributionCase, contributions

if TYPE_CHECKING:
    from superr.settings import Settings

#: Requirement 18, criterion 13: Final_Scores equal to 6 decimal places are tied.
_DECIMALS = 6
_KEY_SCALE = 1e6  # a key is a 6-decimal value x 10^6

#: A drawn finite score is ``units`` / 10^8: an 8-decimal number. One key spans 100 units.
_UNIT_SCALE = 1e8
_UNITS_PER_KEY = 100
_HALF_KEY = _UNITS_PER_KEY // 2
_MAX_JITTER = 49  # j of an 8-decimal score (100 K + j) / 10^8, at least 1 unit from a half-way point
_KEY_LIMIT = 60_000_000  # keys within +-60 runs per over

#: m / 128 with m odd is exactly half-way between two keys (m x 7812.5); it is m x 781250 units.
_UNITS_PER_128TH = 781_250
_HALF_M_LIMIT = 60 * 128 // 2  # m = 2i + 1 with -3840 <= i < 3840 keeps m / 128 within +-60

_MAX_PLAYERS = 24
_PLAYER_IDS = st.integers(1, 10**7)
_CLUTCH = st.floats(-40.0, 40.0, allow_nan=False, allow_infinity=False)
_JITTER = st.one_of(
    st.just(0), st.sampled_from((-_MAX_JITTER, _MAX_JITTER)), st.integers(-_MAX_JITTER, _MAX_JITTER)
)
_NON_FINITE = st.sampled_from((math.nan, math.nan, math.inf, -math.inf))

#: How a player's Final_Score is drawn. HALF_WAY falls back to EIGHT_DECIMALS in a table without one.
EIGHT_DECIMALS, HALF_WAY, NOT_FINITE = "8-decimal", "half-way", "not finite"
_KINDS = (EIGHT_DECIMALS,) * 6 + (HALF_WAY, NOT_FINITE)


def _six_decimal_key(units: int) -> int:
    """``units`` x 10^-8 rounded to 6 decimals, half to even as numpy.rint rounds, times 10^6 (exact integers)."""
    whole, rest = divmod(units, _UNITS_PER_KEY)  # rest in 0..99, also for negative units
    up = rest > _HALF_KEY or (rest == _HALF_KEY and whole % 2 == 1)
    return whole + 1 if up else whole


class Player(NamedTuple):
    """One input row of ``rank_order`` and how its Final_Score was drawn."""

    player_id: int
    score: float  # Final_Score
    clutch: float  # shrunk Clutch_Level
    units: int | None  # score = units / 10^8 exactly; None when the score is not finite
    kind: str

    @property
    def key(self) -> int | None:
        """The intended 6-decimal value x 10^6; None for an unranked player."""
        return None if self.units is None else _six_decimal_key(self.units)


def _ranked(player_id: int, units: int, clutch: float, kind: str = EIGHT_DECIMALS) -> Player:
    # units / 10^8 is correctly rounded, so a half-way m x 781250 units gives m / 128 exactly.
    return Player(player_id, units / _UNIT_SCALE, clutch, units, kind)


def _unranked(player_id: int, score: float, clutch: float) -> Player:
    return Player(player_id, score, clutch, None, NOT_FINITE)


class RankTable(NamedTuple):
    """Rows for ``rank_order``, a drawn row order of them and, for a worked example, the ranks by hand."""

    players: tuple[Player, ...]
    order: tuple[int, ...]  # a permutation of the row positions
    by_hand: tuple[float, ...] | None = None  # each row's rank worked out by hand, NaN when unranked

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Final_Score and shrunk Clutch_Level (float64) and ``player_id`` (int64), in row order."""
        scores = np.array([p.score for p in self.players], dtype=np.float64)
        clutch = np.array([p.clutch for p in self.players], dtype=np.float64)
        ids = np.array([p.player_id for p in self.players], dtype=np.int64)
        return scores, clutch, ids

    def __repr__(self) -> str:
        lines = [f"{'row':>3} {'player_id':>9} {'final_score':>24} {'shrunk_clutch':>24} {'units':>11} {'key':>9}  kind"]
        lines += [
            f"{row:>3} {p.player_id:>9} {p.score!r:>24} {p.clutch!r:>24} {p.units!s:>11} {p.key!s:>9}  {p.kind}"
            for row, p in enumerate(self.players)
        ]
        lines.append(f"row order {list(self.order)}")
        return "\n".join(lines)


@st.composite
def rank_tables(draw: Any) -> RankTable:
    """Final_Scores, shrunk Clutch_Levels and ``player_id`` values with every kind of tie (module docstring)."""
    half_way: int | None = None  # the units of the table's half-way score m / 128
    if draw(st.booleans()):
        m = 2 * draw(st.integers(-_HALF_M_LIMIT, _HALF_M_LIMIT - 1)) + 1
        half_way = m * _UNITS_PER_128TH
        below = half_way // _UNITS_PER_KEY  # the half-way score lies between keys below and below + 1
        run = [below - 1, below, below + 1, below + 2]
    else:
        start = draw(st.integers(-_KEY_LIMIT, _KEY_LIMIT))
        run = [start + step for step in range(draw(st.integers(1, 4)))]
    far = draw(st.lists(st.integers(-_KEY_LIMIT, _KEY_LIMIT), max_size=2))
    keys = tuple(dict.fromkeys(run + far))
    shared = draw(st.lists(_CLUTCH, min_size=1, max_size=3))
    if draw(st.booleans()):  # one ulp above a shared value: equal to it to 6 decimals, but higher
        shared.append(float(np.nextafter(shared[0], math.inf)))
    n = draw(st.integers(0, _MAX_PLAYERS))
    players: list[Player] = []
    for player_id in draw(st.lists(_PLAYER_IDS, min_size=n, max_size=n, unique=True)):
        kind = draw(st.sampled_from(_KINDS))
        if kind == NOT_FINITE:
            players.append(_unranked(player_id, draw(_NON_FINITE), draw(st.one_of(st.just(math.nan), _CLUTCH))))
            continue
        clutch = draw(st.sampled_from(shared)) if draw(st.integers(0, 3)) else draw(_CLUTCH)
        if kind == HALF_WAY and half_way is not None:
            players.append(_ranked(player_id, half_way, clutch, HALF_WAY))
        else:
            players.append(_ranked(player_id, _UNITS_PER_KEY * draw(st.sampled_from(keys)) + draw(_JITTER), clutch))
    return RankTable(tuple(players), tuple(draw(st.permutations(range(n)))))


#: Every case of the module docstring, with its ranks worked out by hand:
#:
#: * key 1000001: 504 (1.00000051, clutch 0) and 505 (1.0000014, clutch -1), ranks 1 and 2. 504 is
#:   2 x 10^-8 above 501 and 505 is 10^-6 above 503, but neither ties with them.
#: * key 1000000: 503 (1.0000004, clutch 9), 502 (1.0000001, clutch 3) and 501 (1.00000049,
#:   clutch 2), ranks 3-5 by clutch, so 501 ranks last although its score is the highest of the three.
#: * key 500000 (0.5 exactly): 10 (clutch one ulp above 1.5), then 9 and 12 (clutch 1.5) by
#:   ``player_id``, compared as numbers: ranks 6-8.
#: * key 7813: 6 (0.0078129, clutch -5), rank 9. Key 7812: 8 (0.0078121, clutch 0.5) and the
#:   half-way 7 (0.0078125 = 1/128, clutch 0.25), which rounds half to even to 0.007812: ranks 10
#:   and 11. Rounding half up would tie 7 with 6 instead and rank 7 above 6.
#: * key 0: 3 (0.0, clutch 0.0) and 4 (-0.00000049, clutch -0.0), equal in both, so 3 then 4:
#:   ranks 12 and 13.
#: * unranked: 1 (NaN), 2 (+inf) and 11 (-inf).
_ONE_ULP_ABOVE_1_5 = float(np.nextafter(1.5, math.inf))
WORKED_EXAMPLE = RankTable(
    players=(
        _ranked(501, 100_000_049, 2.0),
        _unranked(1, math.nan, math.nan),
        _ranked(12, 50_000_000, 1.5),
        _ranked(7, 781_250, 0.25, HALF_WAY),
        _ranked(505, 100_000_140, -1.0),
        _ranked(3, 0, 0.0),
        _ranked(502, 100_000_010, 3.0),
        _unranked(2, math.inf, 100.0),
        _ranked(9, 50_000_000, 1.5),
        _ranked(6, 781_290, -5.0),
        _ranked(504, 100_000_051, 0.0),
        _ranked(4, -49, -0.0),
        _ranked(10, 50_000_000, _ONE_ULP_ABOVE_1_5),
        _ranked(8, 781_210, 0.5),
        _unranked(11, -math.inf, 0.0),
        _ranked(503, 100_000_040, 9.0),
    ),
    order=tuple(range(15, -1, -1)),
    by_hand=(5, math.nan, 8, 11, 2, 12, 4, math.nan, 7, 9, 1, 13, 6, 10, math.nan, 3),
)


def expected_ranks(rounded: Sequence[float | None], clutch: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Ranks by brute force (Requirement 18, criterion 13).

    ``rounded``: each row's 6-decimal Final_Score (or that value x 10^6), None for an unranked row.
    The other rows, sorted by (-rounded, -shrunk Clutch_Level, ``player_id``), get 1, 2, 3...
    """
    rows = [row for row, value in enumerate(rounded) if value is not None]
    rows.sort(key=lambda row: (-rounded[row], -float(clutch[row]), int(ids[row])))
    ranks = np.full(len(rounded), np.nan)
    for position, row in enumerate(rows, start=1):
        ranks[row] = position
    return ranks


def _listing(ids: np.ndarray, scores: np.ndarray, clutch: np.ndarray, ranks: np.ndarray, expected: np.ndarray) -> str:
    """The rows in expected rank order, unranked last, with both ranks: for a failure message."""
    rows = sorted(range(len(ids)), key=lambda row: (np.isnan(expected[row]), np.nan_to_num(expected[row]), row))
    head = f"{'player_id':>9} {'final_score':>24} {'6 decimals':>14} {'shrunk_clutch':>24} {'rank':>5} {'expected':>8}"
    return "\n".join(
        [head]
        + [
            f"{int(ids[row]):>9} {float(scores[row])!r:>24} {float(np.round(scores[row], _DECIMALS))!r:>14} "
            f"{float(clutch[row])!r:>24} {ranks[row]:>5} {expected[row]:>8}"
            for row in rows
        ]
    )


def _one_ulp_apart(values: set[float]) -> bool:
    ordered = sorted(values)
    return any(np.nextafter(low, math.inf) == high for low, high in zip(ordered, ordered[1:]))


def _classify(table: RankTable) -> None:
    """Hypothesis events naming the kinds of tie and near-tie that the table holds."""
    ranked = [p for p in table.players if p.units is not None]
    if len(ranked) < len(table.players):
        event("a Final_Score that is not finite")
    by_key: dict[int, list[Player]] = {}
    for p in ranked:
        by_key.setdefault(p.key, []).append(p)
    ties = [group for group in by_key.values() if len(group) > 1]
    if any(len({p.score for p in group}) < len(group) for group in ties):
        event("exact tie")
    if any(len({p.score for p in group}) > 1 for group in ties):
        event("tie within 6 decimals, scores not equal")
    if any(len({p.clutch for p in group}) < len(group) for group in ties):
        event("equal shrunk Clutch_Levels inside a 6-decimal tie")
    if any(_one_ulp_apart({p.clutch for p in group}) for group in ties):
        event("shrunk Clutch_Levels one ulp apart inside a 6-decimal tie")
    gaps = {b.units - a.units for a in ranked for b in ranked if b.key == a.key + 1}
    if any(gap < _UNITS_PER_KEY for gap in gaps):
        event("neighbouring keys less than 10^-6 apart")
    if _UNITS_PER_KEY in gaps:
        event("neighbouring keys exactly 10^-6 apart (8-decimal values)")
    half = next((p for p in ranked if p.kind == HALF_WAY), None)
    if half is not None:
        below = half.units // _UNITS_PER_KEY
        odd_neighbour = below + 1 if half.key == below else below
        if any(p.kind != HALF_WAY and p.key == half.key for p in ranked):
            event("half-way score tied with a score of its even key")
        if any(p.key == odd_neighbour for p in ranked):
            event("half-way score next to a score of its odd key")


@given(table=rank_tables())
@example(table=WORKED_EXAMPLE)
def test_rank_order_is_a_deterministic_total_order(table: RankTable) -> None:
    """``rank_order`` and ``rank`` give ranks 1..n with no repeats, by 6-decimal Final_Score, then higher shrunk
    Clutch_Level, then lower ``player_id``, whatever the row order; NaN or <NA> when the Final_Score is not finite.

    **Validates: Requirements 18.13**
    """
    scores, clutch, ids = table.arrays()
    keys = [p.key for p in table.players]
    _classify(table)

    # 0. The drawn values are unambiguous: both roundings give the intended 6-decimal value.
    for p, key in zip(table.players, keys):
        if key is not None:
            value = key / _KEY_SCALE
            assert np.round(p.score, _DECIMALS) == value and round(p.score, _DECIMALS) == value, (
                f"strategy bug: {p.score!r} does not round to {value!r} under both numpy.round and round"
            )

    # 1. One float64 rank per row, NaN exactly where the Final_Score is not finite; 1..n once each.
    inputs = (scores.copy(), clutch.copy(), ids.copy())
    ranks = rank_order(scores, clutch, ids)
    expected = expected_ranks(keys, clutch, ids)
    assert ranks.dtype == np.float64 and ranks.shape == scores.shape, (ranks.dtype, ranks.shape)
    unranked = ~np.isfinite(scores)
    assert np.array_equal(np.isnan(ranks), unranked), _listing(ids, scores, clutch, ranks, expected)
    n_ranked = int((~unranked).sum())
    assert sorted(ranks[~unranked].tolist()) == list(range(1, n_ranked + 1)), _listing(
        ids, scores, clutch, ranks, expected
    )

    # 2. The order of the brute-force sort (and of the hand-worked ranks).
    assert np.array_equal(ranks, expected, equal_nan=True), (
        "rank_order differs from the sort by (-6-decimal Final_Score, -shrunk Clutch_Level, player_id):\n"
        + _listing(ids, scores, clutch, ranks, expected)
    )
    if table.by_hand is not None:
        assert np.array_equal(expected, np.array(table.by_hand), equal_nan=True), "the brute-force sort is wrong"

    # 3. Deterministic, and the inputs are left as they were.
    assert np.array_equal(rank_order(scores, clutch, ids), ranks, equal_nan=True), "a second call differs"
    for name, after, before in zip(("final_score", "shrunk_clutch", "player_id"), (scores, clutch, ids), inputs):
        assert np.array_equal(after, before, equal_nan=True), f"rank_order changed its {name} input"

    # 4. Row order: the same rank per player, from rank_order on arrays and from rank on a table.
    order = np.array(table.order, dtype=np.int64)
    shuffled = rank_order(scores[order], clutch[order], ids[order])
    assert np.array_equal(shuffled, ranks[order], equal_nan=True), (
        f"rows in the order {order.tolist()} change the ranks:\n"
        + _listing(ids[order], scores[order], clutch[order], shuffled, ranks[order])
    )
    frame = pd.DataFrame(
        {"player_id": ids[order], "final_score": scores[order], "shrunk_clutch": clutch[order]},
        index=pd.Index(order, dtype=np.int64),
    )
    untouched = frame.copy()
    out = rank(frame)
    pd.testing.assert_frame_equal(frame, untouched, check_exact=True, obj="the input of rank")
    assert out.columns.tolist() == [*frame.columns, "rank"]
    pd.testing.assert_frame_equal(out[frame.columns], frame, check_exact=True, obj="rank's copy of its input")
    want = pd.Series(expected[order], index=frame.index, name="rank").astype("Int64")
    pd.testing.assert_series_equal(out["rank"], want, check_exact=True, obj="the rank column")


# --------------------------------------------------------------------------------- rank_role

_TIE_CONTRIBUTIONS = contributions(force=(TIES,))


class TwinCase(NamedTuple):
    """A contributions() draw with the TIES twins, and each Role's ``player_id`` values in a drawn order."""

    case: ContributionCase
    eligible: dict[str, tuple[int, ...]]  # Role -> every player of the Role, in drawn order

    def __repr__(self) -> str:
        return (
            f"twins (player, twin) {self.case.twins}\nEligible_Players in drawn order {self.eligible}\n"
            f"matches:\n{self.case.matches.to_string()}\nentries:\n{self.case.entries.to_string()}"
        )


@st.composite
def twin_cases(draw: Any) -> TwinCase:
    case = draw(_TIE_CONTRIBUTIONS)
    eligible = {
        role: tuple(draw(st.permutations(np.unique(case.contrib.for_role(role).player).tolist())))
        for role in case.roles
    }
    return TwinCase(case, eligible)


@given(twin_case=twin_cases())
def test_rank_role_orders_full_ties_by_player_id(twin_case: TwinCase, settings: Settings) -> None:
    """In ``rank_role`` the TIES twin with the lower ``player_id`` ranks first, and every Role's ranks are 1..n
    with no repeats in the order of Requirement 18, criterion 13, whatever the order of the Eligible_Players.

    **Validates: Requirements 18.13**
    """
    pct = settings.get("design_choices.D12_replacement_pct")
    case = twin_case.case
    for role in case.roles:
        ids = np.sort(np.array(twin_case.eligible[role], dtype=np.int64))
        ranking = rank_role(case.contrib, role=role, eligible=ids, replacement_pct=pct)
        assert ranking.player_id.tolist() == ids.tolist()
        score, clutch, ranks = ranking.final_score, ranking.shrunk_clutch, ranking.rank
        rounded = [float(np.round(value, _DECIMALS)) if math.isfinite(value) else None for value in score.tolist()]
        expected = expected_ranks(rounded, clutch, ids)

        # 1. The twins: equal in everything but the player_id, so the lower id ranks first.
        player, twin = case.twins[role]
        low, high = (int(np.searchsorted(ids, value)) for value in sorted((player, twin)))
        assert np.isfinite(score[[low, high]]).all(), f"strategy bug: the {role} twins {player}, {twin} are unranked"
        assert score[low] == score[high] and clutch[low] == clutch[high], (
            f"strategy bug: the {role} twins {player}, {twin} differ: Final_Scores {score[low]!r}, {score[high]!r}, "
            f"shrunk Clutch_Levels {clutch[low]!r}, {clutch[high]!r}"
        )
        assert ranks[low] < ranks[high], (
            f"{role} twins: player_id {ids[low]} has rank {ranks[low]}, the higher {ids[high]} rank {ranks[high]}:\n"
            + _listing(ids, score, clutch, ranks, expected)
        )

        # 2. NaN exactly where the Final_Score is not finite; 1..n once each; the brute-force order.
        unranked = ~np.isfinite(score)
        assert np.array_equal(np.isnan(ranks), unranked), _listing(ids, score, clutch, ranks, expected)
        n_ranked = int((~unranked).sum())
        assert sorted(ranks[~unranked].tolist()) == list(range(1, n_ranked + 1)), _listing(
            ids, score, clutch, ranks, expected
        )
        assert np.array_equal(ranks, expected, equal_nan=True), (
            f"{role} ranks differ from the sort by (-6-decimal Final_Score, -shrunk Clutch_Level, player_id):\n"
            + _listing(ids, score, clutch, ranks, expected)
        )

        # 3. Again, through prepare_role with the Eligible_Players in the drawn order.
        drawn = np.array(twin_case.eligible[role], dtype=np.int64)
        again = rank_role(prepare_role(case.contrib, role, drawn), replacement_pct=pct)
        assert np.array_equal(again.player_id, ids)
        assert np.array_equal(again.final_score, score, equal_nan=True), f"{role}: Final_Scores differ on a rerun"
        assert np.array_equal(again.rank, ranks, equal_nan=True), f"{role}: ranks differ on a rerun"

        # 4. rank on the RoleRanking table gives back its own rank column.
        table = ranking.to_frame()
        pd.testing.assert_series_equal(rank(table)["rank"], table["rank"], check_exact=True, obj=f"{role} rank")

        event("the twin has the lower player_id" if twin < player else "the twin has the higher player_id")
        finite = [value for value in rounded if value is not None]
        if finite.count(rounded[low]) > 2:
            event("another player ties the twins on the 6-decimal Final_Score")
        if len(finite) > 2 and len(set(finite)) == 1:
            event("every ranked Final_Score of the Role is equal")
        if unranked.any():
            event("an Eligible_Player without a Clutch_Level is unranked")
