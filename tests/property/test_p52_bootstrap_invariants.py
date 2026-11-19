# Feature: pressure-rating-system, Property 52: Bootstrap invariants
"""Property 52: Bootstrap invariants (Requirement 19, criteria 1, 2 and 4-8; design section
"Robustness_Suite and activity comparison").

**Validates: Requirements 19.1, 19.2, 19.4, 19.5, 19.6, 19.7, 19.8**

For any synthetic contribution table and seed: both Roles use the same resample draws; two runs
with the same seed give identical draws, P_Top10, median ranks and unranked counts, with interval
bounds equal to 6 decimal places; a resample with every match drawn once reproduces the main
Final_Scores and ranks; every resample holds the Role-level values of the ranking it resamples
(Replacement_Level, the tau^2 values, both Split_Half_Reliability values and the Blend_Weights),
so removing another Eligible_Player leaves a player's resample Final_Scores unchanged; players
with fewer than 2 Pressure_Matches or no Ordinary_Ball in a resample are unranked there, with no
Final_Score, and counted outside the top 10; every P_Top10 lies in [0, 1]; when every resample
ranks at least 10 players, a Role's P_Top10 values sum to 10 within 1e-9; and the interval
bounds equal the 2.5th and 97.5th linear-interpolation percentiles over ranked resamples.

1. ``test_resample_draws_follow_the_seed`` (``draw_resamples``, ``multiplicities``). For N from 1
   to 60 resampled matches, B from 1 to 30 resamples and a seed from 0 to 2^32 - 1, within a match
   universe of N to N + 3 matches (``positions``: the universe index of each resampled match, in a
   drawn order, or None for every match in order):

   * the draws are a (B, N) int16 array of positions 0..N-1, the same on a second call, and equal
     to ``numpy.random.Generator(PCG64(seed)).integers(0, N, (B, N))`` (design);
   * ``multiplicities`` gives, for every row and for each row on its own, the copies of each
     universe match counted here one draw at a time: each row sums to N, and a match outside
     ``positions`` always has multiplicity 0.

2. ``test_bootstrap_invariants`` (``bootstrap``, ``bootstrap_table``, ``same_top10``). Each example
   is a ``tests.strategies.contributions`` draw with ``dense=True``, 3-6 matches and, in each Role,
   10-12 random players and the forced players ONE_PRESSURE_MATCH and TIES:

   * a dense player plays every match, with a Pressure_Ball and an Ordinary_Ball in each, so he has
     N >= 3 Pressure_Matches in every resample and is always ranked. Every resample therefore
     ranks at least 10 players, and criterion 8 applies to every example;
   * the ONE_PRESSURE_MATCH player has Pressure_Balls in one match only. He is ranked only in the
     resamples that draw that match at least twice (each drawn copy is a Pressure_Match) and draw
     a match holding one of his Ordinary_Balls. His pressure charge in another match (batter Role)
     is a pressure row but not a Pressure_Ball;
   * the TIES player and his twin, with identical entries in 2-4 matches, are unranked in the
     resamples that leave them fewer than 2 Pressure_Matches or no Ordinary_Ball. When ranked, they
     are tied on everything but the ``player_id``.

   The match universe is the case's matches plus 0-2 matches with no contribution row (like the
   matches whose only Deliveries are in Super_Over_Innings, which no resample draws), and
   ``positions`` holds the universe index of each of the N resampled matches, ascending or in a
   drawn order (None for the case's matches alone, in order). Each Role's Eligible_Players are all
   its players, or all but the ONE_PRESSURE_MATCH player. Without him every Eligible_Player is
   rankable when every match is drawn once, which the ``bootstrap`` stage requires of the
   Main_Ranking. With him, the Main_Ranking (``rankable_only`` off) ranks him on his one
   Pressure_Match, but no resample with every match drawn once can. B runs from 1 to 40 and the
   seed from 0 to 2^32 - 1.

   A run rebuilds its inputs from the contributions table, as the ``bootstrap`` stage does:
   ``Contributions.from_frame``, ``prepare_role`` per Role, the Main_Ranking ``rank_role(data,
   replacement_pct=D12)``, ``RoleHyper.from_ranking`` of it, ``draw_resamples(N, B, seed)``,
   ``bootstrap`` and ``bootstrap_table``. The references count, from the strategy's entries, each
   Eligible_Player's Pressure_Balls and Ordinary_Balls in each universe match, and the
   multiplicities of each resample one draw at a time. Checks:

   1. Determinism (criterion 6): a second run gives identical draws, RoleHyper, Final_Score and
      rank arrays, P_Top10 values, median ranks and ranked and unranked counts, and interval
      bounds equal to 6 decimal places.
   2. Same draws and fixed values (criteria 1 and 2): ``multiplicities`` of the draws equals the
      counted multiplicities. Every RoleHyper holds its Main_Ranking's Replacement_Level, four
      tau^2 (after the floor), rho_L, rho_U, w_L and w_U, and ``RoleHyper.from_summary`` of a
      role_summary table built from the Main_Rankings gives it back. For every resample b and both
      Roles, ``rank_role(data, mult_b, fixed=RoleHyper, rankable_only=True)`` with the one counted
      multiplicity vector of row b holds those values; it ranks exactly the players that the
      entries make rankable, in the order of a sort by (-6-decimal Final_Score, -shrunk
      Clutch_Level, ``player_id``) (Requirement 18, criterion 13); and its Final_Scores and ranks
      are row b of both Roles' arrays.
   3. Every match drawn once (draws 0, 1, ..., N-1): when every Eligible_Player of the Role is
      rankable at these multiplicities, the resample gives the Main_Ranking's Final_Scores and
      ranks bit for bit, and ``same_top10`` is True. Otherwise (the ONE_PRESSURE_MATCH player
      eligible, with 1 Pressure_Match) he is unranked, and every other player gets his
      Main_Ranking Final_Score bit for bit and the Main_Ranking's ranks renumbered without him.
   4. Fixed values make players independent (criterion 2): with the same RoleHyper and draws, the
      bootstrap of each Role without one drawn Eligible_Player gives every other player the same
      Final_Score in every resample, bit for bit, and the same rank less 1 where the removed
      player ranked above him.
   5. Unranked players (criterion 4): a player is ranked in a resample exactly when his
      Pressure_Matches (each drawn copy of a match with a Pressure_Ball counted) and Ordinary_Balls
      reach 2 and 1. Elsewhere his Final_Score and rank are NaN.
   6. The table (criteria 5, 7 and 8), against the counted ranked flags: ``player_id``,
      ``main_rank`` and ``main_final_score`` follow the Main_Ranking; ``n_ranked`` and
      ``n_unranked`` sum to B; P_Top10 is the share of all B resamples with a rank of at most 10
      (an unranked player never counts), lies in [0, 1], and, when every resample ranks at least
      10 players, sums to 10 within 1e-9 over the Role's players (``math.fsum``); ``same_top10``
      flags exactly the resamples whose top-10 set is the Main_Ranking's; ``ci_low`` and
      ``ci_high`` equal ``numpy.percentile(..., [2.5, 97.5], method="linear")`` of the
      Final_Scores of the ranked resamples, and ``median_rank`` is ``statistics.median`` of their
      ranks, all three NaN for a player who is never ranked.

Hypothesis events record the layout of ``positions``, both branches of check 3, unranked players
of each kind, the ONE_PRESSURE_MATCH player ranked in a resample, players in the top 10 of some
resamples and unranked in others, players never ranked, twins ranked, and removed players whose
absence moves other ranks, so ``--hypothesis-show-statistics`` shows that the cases were reached.
No event is asserted.

Not covered here: the per-player estimates and SEs under multiplicities (Properties 45-47), the
Shrinkage, Split_Half_Reliability and Blend_Weights themselves (Properties 49 and 50), the
``bootstrap_warnings`` table, and the ``bootstrap`` stage's checkpoint and files.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import pandas as pd
from hypothesis import event, given, note
from hypothesis import strategies as st

from superr.rating import Contributions, RoleData, RoleHyper, RoleRanking, prepare_role, rank_role
from superr.robustness import (
    RoleBootstrap,
    bootstrap,
    bootstrap_table,
    draw_resamples,
    multiplicities,
    same_top10,
)
from tests.strategies import ONE_PRESSURE_MATCH, TIES, ContributionCase, contributions

if TYPE_CHECKING:
    from superr.settings import Settings

#: Requirement 19, criterion 4: a player is ranked in a resample with at least 2 Pressure_Matches
#: and 1 Ordinary_Ball, each drawn copy of a match counted.
MIN_PRESSURE_MATCHES = 2
MIN_ORDINARY_BALLS = 1
#: Criterion 5: P_Top10 counts ranks 1-10; the 95% interval is the 2.5th to 97.5th percentile.
TOP = 10
PERCENTILES = (2.5, 97.5)
#: Criterion 8's absolute tolerance on the sum of a Role's P_Top10 values.
SUM_TOL = 1e-9
#: Criterion 6: interval bounds agree to 6 decimal places; Requirement 18, criterion 13: Final_Scores
#: equal to 6 decimal places are tied.
DECIMALS = 6
#: The Role-level values that every resample holds (criterion 2), as RoleHyper names its fields.
ROLE_VALUES = ("replacement_level", "tau2_usual", "tau2_gap", "tau2_uplift", "tau2_clutch", "rho_L", "rho_U", "w_L",
               "w_U")  # fmt: skip

_SEEDS = st.integers(0, 2**32 - 1)


def _counted(draws: np.ndarray, positions: np.ndarray | None, n_universe: int) -> np.ndarray:
    """(B, n_universe) multiplicities, one draw at a time: draw d is a copy of universe match positions[d]."""
    universe_index = list(range(n_universe)) if positions is None else positions.tolist()
    counts = np.zeros((len(draws), n_universe), dtype=np.int64)
    for b, row in enumerate(draws.tolist()):
        for match, copies in Counter(universe_index[d] for d in row).items():
            counts[b, match] = copies
    return counts


# ------------------------------------------------------------------------------- the draws

class DrawCase(NamedTuple):
    """The arguments of ``draw_resamples`` and ``multiplicities``."""

    n_resampled: int  # N
    n_boot: int  # B
    seed: int
    n_universe: int  # the size of the match universe, N or more
    positions: np.ndarray | None  # the universe index of each resampled match; None: every match, in order


@st.composite
def _draw_cases(draw: Any) -> DrawCase:
    n = draw(st.integers(1, 60))
    n_boot = draw(st.integers(1, 30))
    seed = draw(_SEEDS)
    extra = draw(st.integers(0, 3))
    if extra == 0 and draw(st.booleans()):
        return DrawCase(n, n_boot, seed, n, None)
    order = draw(st.permutations(range(n + extra)))
    return DrawCase(n, n_boot, seed, n + extra, np.array(order[:n], dtype=np.int64))


@given(case=_draw_cases())
def test_resample_draws_follow_the_seed(case: DrawCase) -> None:
    """The same seed always gives the same (B, N) int16 draws, from ``Generator(PCG64(seed))``, and
    ``multiplicities`` counts each drawn copy of a universe match (Requirement 19, criteria 1 and 6).

    **Validates: Requirements 19.1, 19.6**
    """
    n, n_boot = case.n_resampled, case.n_boot
    draws = draw_resamples(n, n_boot, case.seed)
    assert draws.dtype == np.int16 and draws.shape == (n_boot, n), (draws.dtype, draws.shape)
    assert draws.min() >= 0 and draws.max() < n, f"a drawn position outside 0..{n - 1}"
    np.testing.assert_array_equal(draw_resamples(n, n_boot, case.seed), draws, err_msg="a second call differs")
    reference = np.random.Generator(np.random.PCG64(case.seed)).integers(0, n, size=(n_boot, n))
    np.testing.assert_array_equal(draws.astype(np.int64), reference, err_msg="not the PCG64 stream of the seed")

    counted = _counted(draws, case.positions, case.n_universe)
    mult = multiplicities(draws, case.n_universe, case.positions)
    assert mult.dtype.kind in "iu", mult.dtype
    np.testing.assert_array_equal(mult, counted, err_msg="multiplicities of every row")
    assert (mult.sum(axis=1) == n).all(), "a resample's multiplicities do not sum to N"
    outside = np.setdiff1d(np.arange(case.n_universe), np.arange(n) if case.positions is None else case.positions)
    assert (mult[:, outside] == 0).all(), "a match outside positions was drawn"
    for b in range(n_boot):
        np.testing.assert_array_equal(
            multiplicities(draws[b], case.n_universe, case.positions), counted[b], err_msg=f"multiplicities of row {b}"
        )
    event("positions: every match, in order" if case.positions is None else
          "positions: a drawn order" if case.n_universe == n else "positions: a subset of the universe")  # fmt: skip


# ------------------------------------------------------------------------------- the examples

_CONTRIBUTIONS = contributions(
    dense=True, n_players=st.integers(10, 12), n_matches=st.integers(3, 6), force=(ONE_PRESSURE_MATCH, TIES)
)
_N_BOOT = st.integers(1, 40)
_EXTRA_MATCH_IDS = st.lists(st.integers(1, 99_999), max_size=2, unique=True)


class Example(NamedTuple):
    """A contributions() draw, the bootstrap's match layout, Eligible_Players, B and seed (module docstring)."""

    case: ContributionCase
    universe: np.ndarray  # match_id of each match of the contributions' match universe, ascending
    positions: np.ndarray | None  # universe index of each of the N resampled matches; None: every match
    eligible: dict[str, np.ndarray]  # Role -> the Eligible_Players' ids, ascending
    removed: dict[str, int]  # Role -> the Eligible_Player left out in check 4
    n_boot: int
    seed: int

    @property
    def n_resampled(self) -> int:
        return len(self.universe) if self.positions is None else len(self.positions)

    def __repr__(self) -> str:
        case = self.case
        positions = None if self.positions is None else self.positions.tolist()
        return "\n".join([
            f"N {self.n_resampled}, B {self.n_boot}, seed {self.seed}",
            f"match universe {self.universe.tolist()}; positions {positions}",
            f"Eligible_Players { {role: ids.tolist() for role, ids in self.eligible.items()} }",
            f"forced {case.forced}; twins {case.twins}; removed {self.removed}",
            "matches:", case.matches[["match_id", "match_date", "delivery_weight"]].to_string(),
            "entries:", case.entries.to_string(),
        ])  # fmt: skip


@st.composite
def _examples(draw: Any) -> Example:
    case = draw(_CONTRIBUTIONS)
    own = set(case.match_ids.tolist())
    extra = [match_id for match_id in draw(_EXTRA_MATCH_IDS) if match_id not in own]
    universe = np.sort(np.array([*own, *extra], dtype=np.int64))
    at = np.searchsorted(universe, case.match_ids)
    if draw(st.booleans()):
        at = np.array(draw(st.permutations(at.tolist())), dtype=np.int64)
    positions = None if np.array_equal(at, np.arange(len(universe))) else at
    eligible: dict[str, np.ndarray] = {}
    removed: dict[str, int] = {}
    for role in case.roles:
        ids = np.unique(case.contrib.for_role(role).player)
        if draw(st.booleans()):  # leave out the player who is unrankable when every match is drawn once
            ids = ids[ids != case.forced[ONE_PRESSURE_MATCH, role]]
        eligible[role] = ids
        removed[role] = draw(st.sampled_from(ids.tolist()))
    return Example(case, universe, positions, eligible, removed, draw(_N_BOOT), draw(_SEEDS))


# ------------------------------------------------------------------------------- runs and references

class Run(NamedTuple):
    """One run of the bootstrap, from the contributions table to the bootstrap tables."""

    data: dict[str, RoleData]
    main: dict[str, RoleRanking]
    fixed: dict[str, RoleHyper]
    draws: np.ndarray
    result: dict[str, RoleBootstrap]
    tables: dict[str, pd.DataFrame]


def _role_data(example: Example, eligible: dict[str, np.ndarray]) -> dict[str, RoleData]:
    contrib = Contributions.from_frame(example.case.frame, example.universe)
    return {role: prepare_role(contrib, role, ids) for role, ids in eligible.items()}


def _run(example: Example, pct: float) -> Run:
    """The inputs rebuilt from the contributions table, as the ``bootstrap`` stage rebuilds them, then the bootstrap."""
    data = _role_data(example, example.eligible)
    main = {role: rank_role(data[role], replacement_pct=pct) for role in data}
    fixed = {role: RoleHyper.from_ranking(main[role]) for role in data}
    draws = draw_resamples(example.n_resampled, example.n_boot, example.seed)
    result = bootstrap(data, fixed, draws, example.positions, replacement_pct=pct)
    tables = {role: bootstrap_table(result[role], main[role]) for role in data}
    return Run(data, main, fixed, draws, result, tables)


class Balls(NamedTuple):
    """Each Eligible_Player's Deliveries in each universe match, from the strategy's entries: (n, U) counts."""

    pressure: np.ndarray  # Pressure_Balls
    ordinary: np.ndarray  # Ordinary_Balls


def _balls(example: Example, role: str) -> Balls:
    ids = example.eligible[role]
    row = {player: i for i, player in enumerate(ids.tolist())}
    column = {match_id: u for u, match_id in enumerate(example.universe.tolist())}
    pressure = np.zeros((len(ids), len(example.universe)), dtype=np.int64)
    ordinary = np.zeros_like(pressure)
    entries = example.case.entries
    # A Dismissal_Charge is not a ball: a pressure charge makes no Pressure_Match.
    mine = (entries["role"] == role).to_numpy(dtype=bool) & ~entries["charge"].to_numpy(dtype=bool)
    names = ("player_id", "match_id", "pressure")
    for player, match_id, flag in zip(*(entries[name].to_numpy()[mine].tolist() for name in names)):
        if player in row:
            (pressure if flag else ordinary)[row[player], column[match_id]] += 1
    return Balls(pressure, ordinary)


class Rankable(NamedTuple):
    """(B, n) per resample and Eligible_Player, under the counted multiplicities."""

    ranked: np.ndarray  # bool: Requirement 19, criterion 4
    pressure_matches: np.ndarray  # each drawn copy of a match with a Pressure_Ball counted
    ordinary_balls: np.ndarray


def _rankable(mult: np.ndarray, balls: Balls) -> Rankable:
    pressure_matches = mult @ (balls.pressure > 0).astype(np.int64).T
    ordinary_balls = mult @ balls.ordinary.T
    ranked = (pressure_matches >= MIN_PRESSURE_MATCHES) & (ordinary_balls >= MIN_ORDINARY_BALLS)
    return Rankable(ranked, pressure_matches, ordinary_balls)


def _role_values(source: RoleRanking | RoleHyper) -> dict[str, float]:
    """The Role-level values that a resample holds, from a RoleRanking (tau^2 after the floor) or a RoleHyper."""
    if isinstance(source, RoleHyper):
        return {name: getattr(source, name) for name in ROLE_VALUES}
    taus = (source.tau2_usual, source.tau2_gap, source.tau2_uplift, source.tau2_clutch)
    values = (source.replacement_level, *(t.value for t in taus), source.rho_L, source.rho_U, source.w_L, source.w_U)
    return dict(zip(ROLE_VALUES, values))


def _sorted_ranks(ranking: RoleRanking, ranked: np.ndarray) -> np.ndarray:
    """Ranks 1, 2, ... of the ``ranked`` players, sorted by (-6-decimal Final_Score, -shrunk Clutch_Level,
    player_id) as Requirement 18, criterion 13 orders them; NaN for the other players."""
    score, clutch, ids = ranking.final_score, ranking.shrunk_clutch, ranking.player_id
    order = sorted(np.flatnonzero(ranked).tolist(),
                   key=lambda j: (-float(np.round(score[j], DECIMALS)), -float(clutch[j]), int(ids[j])))  # fmt: skip
    ranks = np.full(len(ids), np.nan)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


# ------------------------------------------------------------------------------- the checks

def _check_determinism(first: Run, second: Run) -> None:
    """Check 1 (criterion 6)."""
    assert second.draws.dtype == first.draws.dtype == np.int16, (first.draws.dtype, second.draws.dtype)
    np.testing.assert_array_equal(second.draws, first.draws, err_msg="two runs with the same seed draw differently")
    for role, table in first.tables.items():
        assert second.fixed[role] == first.fixed[role], f"{role}: the RoleHyper differs between runs"
        for name in ("final_score", "rank"):
            np.testing.assert_array_equal(
                getattr(second.result[role], name), getattr(first.result[role], name), err_msg=f"{role}: {name} arrays"
            )
        again = second.tables[role]
        for name in ("player_id", "p_top10", "median_rank", "n_ranked", "n_unranked"):
            assert np.array_equal(again[name].to_numpy(), table[name].to_numpy(), equal_nan=True), f"{role}: {name}"
        for name in ("ci_low", "ci_high"):
            assert np.array_equal(
                np.round(again[name].to_numpy(), DECIMALS), np.round(table[name].to_numpy(), DECIMALS), equal_nan=True
            ), f"{role}: {name} differs between runs at 6 decimal places"


def _check_resamples(run: Run, example: Example, mult: np.ndarray, rankable: dict[str, Rankable], pct: float) -> None:
    """Check 2 (criteria 1, 2 and 4; Requirement 18, criterion 13)."""
    np.testing.assert_array_equal(
        multiplicities(run.draws, len(example.universe), example.positions), mult, err_msg="multiplicities of the draws"
    )
    summary = pd.DataFrame([run.main[role].summary() for role in run.main])
    for role, main in run.main.items():
        assert _role_values(run.fixed[role]) == _role_values(main), (
            f"{role}: RoleHyper {run.fixed[role]} does not hold the Main_Ranking's values {_role_values(main)}"
        )
        assert RoleHyper.from_summary(summary, role) == run.fixed[role], f"{role}: from_summary != from_ranking"
    for b in range(example.n_boot):
        for role, data in run.data.items():  # one multiplicity vector for both Roles
            ref = rank_role(data, mult[b], fixed=run.fixed[role], replacement_pct=pct, rankable_only=True)
            what = f"{role}, resample {b} (multiplicities {mult[b].tolist()})"
            assert _role_values(ref) == _role_values(run.main[role]), (
                f"{what}: Role-level values {_role_values(ref)}; the Main_Ranking's are {_role_values(run.main[role])}"
            )
            ranked = rankable[role].ranked[b]
            np.testing.assert_array_equal(ref.ranked, ranked, err_msg=f"{what}: rank_role ranks other players")
            expected = _sorted_ranks(ref, ranked)
            np.testing.assert_array_equal(ref.rank, expected, err_msg=f"{what}: rank_role's ranking order")
            result = run.result[role]
            np.testing.assert_array_equal(
                result.final_score[b], np.where(ranked, ref.final_score, np.nan),
                err_msg=f"{what}: bootstrap Final_Scores",
            )  # fmt: skip
            np.testing.assert_array_equal(result.rank[b], expected, err_msg=f"{what}: bootstrap ranks")


def _check_every_match_once(run: Run, example: Example, balls: dict[str, Balls], pct: float) -> None:
    """Check 3: one copy of every resampled match."""
    once = np.arange(example.n_resampled, dtype=np.int16)[None, :]
    mult = _counted(once, example.positions, len(example.universe))
    result = bootstrap(run.data, run.fixed, once, example.positions, replacement_pct=pct)
    for role, main in run.main.items():
        rankable = _rankable(mult, balls[role]).ranked[0]
        event("every match drawn once: every Eligible_Player of the Role rankable" if rankable.all() else
              "every match drawn once: the Role's ONE_PRESSURE_MATCH player unranked")  # fmt: skip
        k = int(rankable.sum())
        renumbered = np.full(len(rankable), np.nan)
        renumbered[np.argsort(np.where(rankable, main.rank, np.inf), kind="stable")[:k]] = np.arange(1, k + 1)
        got = result[role]
        np.testing.assert_array_equal(
            got.final_score[0], np.where(rankable, main.final_score, np.nan),
            err_msg=f"{role}: every match drawn once does not give the Main_Ranking's Final_Scores",
        )  # fmt: skip
        np.testing.assert_array_equal(
            got.rank[0], renumbered, err_msg=f"{role}: every match drawn once does not give the Main_Ranking's ranks"
        )
        if rankable.all():
            np.testing.assert_array_equal(got.final_score[0], main.final_score, err_msg=f"{role}: Main Final_Scores")
            np.testing.assert_array_equal(got.rank[0], main.rank, err_msg=f"{role}: Main ranks")
            assert same_top10(got, main).all(), f"{role}: every match drawn once changes the top-10 set"


def _check_removal(run: Run, example: Example, pct: float) -> None:
    """Check 4 (criterion 2)."""
    kept = {role: ids[ids != example.removed[role]] for role, ids in example.eligible.items()}
    fewer = bootstrap(_role_data(example, kept), run.fixed, run.draws, example.positions, replacement_pct=pct)
    for role, ids in example.eligible.items():
        keep = ids != example.removed[role]
        full, out = run.result[role], fewer[role]
        np.testing.assert_array_equal(out.player_id, ids[keep])
        np.testing.assert_array_equal(
            out.final_score, full.final_score[:, keep],
            err_msg=f"{role}: removing Eligible_Player {example.removed[role]} changes other players' Final_Scores",
        )  # fmt: skip
        gone = full.rank[:, ~keep]  # (B, 1), NaN where the removed player is unranked
        above = full.rank[:, keep] > gone  # False wherever either rank is NaN
        np.testing.assert_array_equal(
            out.rank, full.rank[:, keep] - above,
            err_msg=f"{role}: removing Eligible_Player {example.removed[role]} moves ranks other than those below him",
        )  # fmt: skip
        if above.any():
            event("a removed player ranked above others: their ranks move up by 1")


def _check_unranked(run: Run, rankable: dict[str, Rankable]) -> None:
    """Check 5 (criterion 4)."""
    for role, result in run.result.items():
        ranked = rankable[role].ranked
        np.testing.assert_array_equal(result.ranked, ranked, err_msg=f"{role}: ranked players")
        assert np.isnan(result.final_score[~ranked]).all() and np.isnan(result.rank[~ranked]).all(), (
            f"{role}: an unranked player has a Final_Score or rank"
        )
        assert np.isfinite(result.final_score[ranked]).all(), f"{role}: a ranked player has no Final_Score"


def _check_table(run: Run, rankable: dict[str, Rankable]) -> None:
    """Check 6 (criteria 5, 7 and 8)."""
    for role, table in run.tables.items():
        result, main, ranked = run.result[role], run.main[role], rankable[role].ranked
        n_boot, ids = result.n_boot, result.player_id
        np.testing.assert_array_equal(table["player_id"].to_numpy(), ids, err_msg=f"{role}: rows")
        np.testing.assert_array_equal(
            table["main_rank"].to_numpy(dtype=np.float64, na_value=np.nan), main.rank, err_msg=f"{role}: main_rank"
        )
        np.testing.assert_array_equal(table["main_final_score"].to_numpy(), main.final_score, err_msg=f"{role}")
        n_ranked = ranked.sum(axis=0)
        np.testing.assert_array_equal(table["n_ranked"].to_numpy(), n_ranked, err_msg=f"{role}: n_ranked")
        np.testing.assert_array_equal(table["n_unranked"].to_numpy(), n_boot - n_ranked, err_msg=f"{role}: n_unranked")

        in_top = ranked & (result.rank <= TOP)  # an unranked player is outside the top 10
        if (in_top.any(axis=0) & ~ranked.all(axis=0)).any():
            event("a player in the top 10 of some resamples and unranked in others")
        p_top10 = table["p_top10"].to_numpy()
        np.testing.assert_array_equal(p_top10, in_top.sum(axis=0) / n_boot, err_msg=f"{role}: P_Top10")
        assert ((p_top10 >= 0.0) & (p_top10 <= 1.0)).all(), f"{role}: P_Top10 outside [0, 1]: {p_top10.tolist()}"
        main_top = set(ids[main.rank <= TOP].tolist())
        expected_same = [set(ids[row].tolist()) == main_top for row in in_top]
        np.testing.assert_array_equal(same_top10(result, main), expected_same, err_msg=f"{role}: same_top10")
        fewest = int(ranked.sum(axis=1).min())
        if fewest >= TOP:  # criterion 8
            total = math.fsum(p_top10.tolist())
            assert abs(total - TOP) <= SUM_TOL, f"{role}: P_Top10 values sum to {total!r}, not 10"
            event("every resample ranks at least 10 players: the P_Top10 sum is checked")

        for j in range(len(ids)):
            keep = ranked[:, j]
            got = (table["ci_low"].iat[j], table["ci_high"].iat[j], table["median_rank"].iat[j])
            if not keep.any():
                assert all(math.isnan(value) for value in got), f"{role} {ids[j]}: never ranked, but {got}"
                event("an Eligible_Player never ranked: no interval or median rank")
                continue
            low, high = np.percentile(result.final_score[keep, j], PERCENTILES, method="linear")
            median = statistics.median(result.rank[keep, j].tolist())
            assert got == (low, high, median), (
                f"{role} {ids[j]}: interval and median rank {got}; expected ({low!r}, {high!r}, {median!r}) over "
                f"{int(keep.sum())} ranked resample(s)"
            )


def _events(example: Example, rankable: dict[str, Rankable]) -> None:
    case = example.case
    if example.positions is None:
        event("positions: None (every match of the universe, in order)")
    else:
        event("positions: " + ("ascending" if (np.diff(example.positions) > 0).all() else "a drawn order")
              + (", with matches never drawn" if len(example.universe) > example.n_resampled else ""))  # fmt: skip
    for role, counts in rankable.items():
        ids = example.eligible[role]
        unranked = ~counts.ranked
        if (unranked & (counts.pressure_matches < MIN_PRESSURE_MATCHES)).any():
            event("unranked in a resample: fewer than 2 Pressure_Matches")
        if (unranked & (counts.pressure_matches >= MIN_PRESSURE_MATCHES) & (counts.ordinary_balls < 1)).any():
            event("unranked in a resample: 2+ Pressure_Matches but no Ordinary_Ball")
        forced = np.flatnonzero(ids == case.forced[ONE_PRESSURE_MATCH, role])
        if len(forced) and counts.ranked[:, forced[0]].any():
            event("the ONE_PRESSURE_MATCH player ranked in a resample (his match drawn 2+ times)")
        twins = np.flatnonzero(np.isin(ids, case.twins[role]))
        if counts.ranked[:, twins].all(axis=1).any():
            event("both TIES twins ranked in a resample")


@given(example=_examples())
def test_bootstrap_invariants(example: Example, settings: Settings) -> None:
    """The bootstrap is deterministic, uses one resample for both Roles, holds the Main_Ranking's Role-level values,
    reproduces the Main_Ranking with every match drawn once, leaves players with fewer than 2 Pressure_Matches or no
    Ordinary_Ball unranked, and summarises the resamples with P_Top10 values in [0, 1] that sum to 10, percentile
    intervals, median ranks and unranked counts (Requirement 19, criteria 1, 2 and 4-8).

    **Validates: Requirements 19.1, 19.2, 19.4, 19.5, 19.6, 19.7, 19.8**
    """
    pct = settings.get("design_choices.D12_replacement_pct")
    first = _run(example, pct)
    second = _run(example, pct)
    _check_determinism(first, second)

    mult = _counted(first.draws, example.positions, len(example.universe))
    balls = {role: _balls(example, role) for role in example.eligible}
    rankable = {role: _rankable(mult, balls[role]) for role in balls}
    note(f"ranked players per resample: { {role: r.ranked.sum(axis=1).tolist() for role, r in rankable.items()} }")
    _check_resamples(first, example, mult, rankable, pct)
    _check_every_match_once(first, example, balls, pct)
    _check_removal(first, example, pct)
    _check_unranked(first, rankable)
    _check_table(first, rankable)
    _events(example, rankable)
