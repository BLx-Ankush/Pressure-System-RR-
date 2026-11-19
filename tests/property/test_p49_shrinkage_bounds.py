# Feature: pressure-rating-system, Property 49: Shrinkage stays between the estimate and its prior
"""Property 49: Shrinkage stays between the estimate and its prior (Requirement 18, criteria 1-6 and 14;
design DD-16).

**Validates: Requirements 18.1, 18.2, 18.3, 18.4, 18.5, 18.6, 18.14**

Over a Role's pool, ``rank_role`` shrinks four estimates. Each has its own Role tau^2 and
match-clustered SE, and a Shrinkage_Factor B = tau^2 / (tau^2 + SE^2):

* the Usual_Baseline O towards Replacement_Level RL: O* = RL + B_O (O - RL);
* the Pressure_Gap g = C - O towards 0: g* = B_g g. The shrunk Clutch_Level is C* = O* + g*;
* Uplift U towards 0: U* = B_U U;
* for a pool player without a Usual_Baseline (no Matched_Cell), Clutch_Level C towards RL instead
  (the fallback): C* = RL + B_C (C - RL).

tau^2 is the sample variance (n - 1) of the pool's estimates minus their mean squared SE, over the
pool players whose estimate and SE are both defined, floored at 0. B is 0 when the SE is undefined,
or when tau^2 and the SE are both 0. RL is the D12 percentile (25 as delivered) of the pool's
unshrunk Clutch_Levels, with linear interpolation.

1. ``test_rank_role_shrinkage_stays_between_each_estimate_and_its_prior``. Each example is a
   ``tests.strategies.contributions`` draw with 3-8 random players per Role and three forced
   players: ONE_PRESSURE_MATCH (every SE undefined, so every B of that player is 0),
   NO_MATCHED_CELL (no Usual_Baseline, so the fallback) and TIES (a player and a twin with
   identical estimates, which push tau^2 to the zero floor in small pools). Each Role's
   Eligible_Players are all its players or a drawn subset. The multiplicities are all ones (the
   Main_Ranking) or a Bootstrap_Resample's counts, with or without ``rankable_only``. For each
   Role, ``rank_role`` must give:

   * the pool: the Eligible_Players with a Pressure_Ball under the multiplicities, plus, with
     ``rankable_only``, 2 or more Pressure_Matches and an Ordinary_Ball (Requirement 19,
     criterion 4), all counted here from the strategy's entries. With an empty pool,
     ``rank_role`` raises ValueError;
   * the fallback players: exactly the pool players without a Matched_Cell in the entries, which
     are also the pool players without a Usual_Baseline;
   * RL: ``numpy.percentile(..., method="linear")`` of the pool's Clutch_Levels within 1e-9, each
     Clutch_Level being 6 x the weighted pressure NBV (Dismissal_Charges included) / the pressure
     weight, summed here from the entries;
   * each tau^2 (Usual_Baseline, Pressure_Gap, Uplift and the fallback Clutch_Level): the
     method-of-moments value recomputed here with ``math.fsum`` from the RoleRanking's estimates
     and SEs, before and after the floor, with its player count;
   * each B: NaN outside the players whose shrunk value uses it, in [0, 1], 0 where the SE is
     undefined or tau^2 = SE = 0 (and for every player when tau^2 is 0), else
     tau^2 / (tau^2 + SE^2) within 1e-12;
   * the bounds of criterion 14 within 1e-9: O* between O and RL, g* between 0 and g, U* between U
     and 0, and a fallback C* between C and RL. Each shrunk value is prior + B (estimate - prior),
     and is the prior exactly when B is 0. For the other pool players C* = O* + g*, which is not
     bounded by C and RL, so that is not asserted;
   * the warnings: one ``tau2_floor`` warning for each tau^2 at 0 that some pool player's shrunk
     value uses (the Clutch_Level tau^2 only when there is a fallback player). For each estimate,
     one ``shrinkage_zero`` warning naming the players with an undefined SE and one naming those
     with tau^2 = SE = 0, each only when there are such players. One ``clutch_fallback`` warning
     names the fallback players, when there are any. The player_ids are read from the message.

2. ``test_tau2_is_the_floored_method_of_moments_value``: ``tau2`` on drawn arrays of 0-12
   estimates (some NaN, many repeated) and SEs (some NaN, some 0) equals the same reference: the
   count of pairs with both values defined, a raw value that is NaN with fewer than 2 pairs, and
   max(0, raw) after the floor.
3. ``test_shrink_stays_between_the_estimate_and_its_prior``: ``shrink`` on such arrays with a drawn
   tau^2 (often 0) and prior gives B in [0, 1], the two zero flags exactly where the SE is NaN or
   tau^2 = SE = 0, B = tau^2 / (tau^2 + SE^2) elsewhere, a NaN shrunk value for a NaN estimate, and
   otherwise a shrunk value between the estimate and the prior within 1e-9 that equals the prior
   exactly when B is 0.

Hypothesis events record the multiplicities, fallback players, twins in a pool, and for each
estimate a floored tau^2, partial shrinkage (0 < B < 1) and both B = 0 rules, so
``--hypothesis-show-statistics`` shows that the cases were reached. No event is asserted.

Not covered here: the estimates and SEs themselves (Requirement 16; Properties 45-47), the
Split_Half_Reliability, Blend_Weights and Final_Score (Property 50) and the ranking (Property 51).
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import pytest
from hypothesis import event, given, note
from hypothesis import strategies as st

from superr.rating import (
    CLUTCH_LEVEL,
    PRESSURE_GAP,
    UPLIFT,
    USUAL_BASELINE,
    RatingWarning,
    RoleRanking,
    Tau2,
    prepare_role,
    rank_role,
    shrink,
    tau2,
)
from tests.strategies import (
    NO_MATCHED_CELL,
    ONE_PRESSURE_MATCH,
    TIES,
    ContributionCase,
    contributions,
    match_multiplicities,
)

if TYPE_CHECKING:
    from superr.settings import Settings

#: Absolute tolerance of the criterion 14 bounds and of every shrunk value, in runs per over.
TOL = 1e-9
#: Absolute tolerance of a Shrinkage_Factor against tau^2 / (tau^2 + SE^2).
B_TOL = 1e-12
#: The forced players added to every Role (module docstring).
FORCED = (ONE_PRESSURE_MATCH, NO_MATCHED_CELL, TIES)
#: Requirement 19, criterion 4: under ``rankable_only`` a pool player has at least 2 Pressure_Matches
#: and 1 Ordinary_Ball, each drawn copy of a match counted.
RANKABLE_PRESSURE_MATCHES = 2
RANKABLE_ORDINARY_BALLS = 1
BALLS_PER_OVER = 6.0
#: The player_id list of a warning message, such as "Eligible_Player(s) 3, 7 have ..." or
#: "player_id(s) 4 is set to 0 ...".
_ID_LIST = re.compile(r"\(s\) (\d+(?:, \d+)*) ")


# ------------------------------------------------------------------------------- the examples

class Example(NamedTuple):
    """One draw: the contributions, the match multiplicities and each Role's Eligible_Players."""

    case: ContributionCase
    mult: np.ndarray | None  # None: all ones (the Main_Ranking); else a Bootstrap_Resample's counts
    rankable_only: bool
    eligible: dict[str, np.ndarray]  # Role -> the Eligible_Players' ids, ascending


@st.composite
def _examples(draw: Any) -> Example:
    case = draw(contributions(force=FORCED, n_players=st.integers(3, 8)))
    mult = draw(st.none() | match_multiplicities(case.contrib.n_matches, resample=True))
    rankable_only = draw(st.booleans())
    eligible: dict[str, np.ndarray] = {}
    for role in case.roles:
        ids = sorted(set(case.entries.loc[case.entries["role"] == role, "player_id"].tolist()))
        if draw(st.booleans()):  # a drawn subset; shrinks towards every player of the Role
            ids = sorted(draw(st.lists(st.sampled_from(ids), min_size=1, max_size=len(ids), unique=True)))
        eligible[role] = np.array(ids, dtype=np.int64)
    return Example(case, mult, rankable_only, eligible)


# ------------------------------------------------------------------------------- the references

class Reference(NamedTuple):
    """Per Eligible_Player (ascending ids), counted from the strategy's entries under the multiplicities."""

    clutch: np.ndarray  # 6 x weighted pressure NBV (charges included) / pressure weight; NaN without a Pressure_Ball
    pool: np.ndarray  # bool: the players behind Replacement_Level and tau^2
    matched: np.ndarray  # bool: a cell with a Pressure_Ball and an Ordinary_Ball, so a Usual_Baseline


def _reference(
    case: ContributionCase, role: str, ids: np.ndarray, mult: np.ndarray | None, rankable_only: bool
) -> Reference:
    """A loop over the Role's entries, each counted once per drawn copy of its match.

    Every weight and entry term is exact in float64 (strategy module docstring), so these sums do
    not depend on their order and the Clutch_Levels equal ``rating``'s.
    """
    entries = case.entries
    mine = (entries["role"] == role).to_numpy(dtype=bool)
    names = ("player_id", "match_id", "cell", "pressure", "charge", "s", "w")
    columns = [entries[name].to_numpy()[mine].tolist() for name in names]
    position = {match_id: i for i, match_id in enumerate(case.match_ids.tolist())}
    copies = [1.0] * len(position) if mult is None else [float(count) for count in mult.tolist()]
    s_pressure: dict[int, float] = defaultdict(float)  # weighted pressure NBV, Dismissal_Charges included
    w_pressure: dict[int, float] = defaultdict(float)
    pressure_balls: dict[int, float] = defaultdict(float)
    ordinary_balls: dict[int, float] = defaultdict(float)
    pressure_matches: dict[int, set[int]] = defaultdict(set)
    cell_balls: dict[tuple[int, int, bool], float] = defaultdict(float)  # (player, cell, pressure) -> Deliveries
    player_cells: dict[int, set[int]] = defaultdict(set)
    for player, match_id, cell, pressure, charge, s, w in zip(*columns):
        match = position[match_id]
        k = copies[match]
        if pressure:
            s_pressure[player] += k * s
            w_pressure[player] += k * w
        if charge:  # a Dismissal_Charge adds to the NBV sum only
            continue
        cell_balls[player, cell, pressure] += k
        player_cells[player].add(cell)
        if pressure:
            pressure_balls[player] += k
            pressure_matches[player].add(match)
        else:
            ordinary_balls[player] += k

    clutch = np.full(len(ids), np.nan)
    pool = np.zeros(len(ids), dtype=bool)
    matched = np.zeros(len(ids), dtype=bool)
    for i, player in enumerate(ids.tolist()):
        if pressure_balls[player] > 0.0:
            clutch[i] = BALLS_PER_OVER * (s_pressure[player] / w_pressure[player])
            pool[i] = not rankable_only or (
                sum(copies[match] for match in pressure_matches[player]) >= RANKABLE_PRESSURE_MATCHES
                and ordinary_balls[player] >= RANKABLE_ORDINARY_BALLS
            )
        matched[i] = any(
            cell_balls[player, cell, True] > 0.0 and cell_balls[player, cell, False] > 0.0
            for cell in player_cells[player]
        )
    return Reference(clutch=clutch, pool=pool, matched=matched)


def _check_tau2(result: Tau2, estimates: np.ndarray, ses: np.ndarray, what: str) -> None:
    """``result`` is max(0, var(n - 1) - mean SE^2) over the pairs with both values defined (criteria 4-5)."""
    keep = np.isfinite(estimates) & np.isfinite(ses)
    n = int(keep.sum())
    assert result.n == n, f"{what}: tau^2 counts {result.n} player(s); {n} have a defined estimate and SE"
    assert result.value >= 0.0 and result.value == (result.raw if result.raw > 0.0 else 0.0), (
        f"{what}: tau^2 {result.value!r} is not the raw value {result.raw!r} floored at 0"
    )
    assert result.floored == (result.value == 0.0), f"{what}: floored is {result.floored} for tau^2 {result.value!r}"
    if n < 2:
        assert math.isnan(result.raw) and result.value == 0.0, (
            f"{what}: with {n} defined pair(s) tau^2 cannot be estimated; got raw {result.raw!r}, "
            f"value {result.value!r}"
        )
        return
    values, errors = estimates[keep].tolist(), ses[keep].tolist()
    mean = math.fsum(values) / n
    variance = math.fsum((value - mean) ** 2 for value in values) / (n - 1)
    noise = math.fsum(error * error for error in errors) / n
    raw = variance - noise
    tol = TOL * max(1.0, variance, noise)
    assert abs(result.raw - raw) <= tol, (
        f"{what}: raw tau^2 {result.raw!r}; the sample variance {variance!r} minus the mean squared SE {noise!r} "
        f"is {raw!r}"
    )
    assert abs(result.value - max(raw, 0.0)) <= tol, f"{what}: tau^2 {result.value!r}; expected {max(raw, 0.0)!r}"


# ------------------------------------------------------------------------------- the rank_role test

class Part(NamedTuple):
    """One shrunk estimate of a RoleRanking and what its Shrinkage_Factor uses (one value per Eligible_Player)."""

    label: str  # the estimate, as RatingWarning.estimate names it
    estimate: np.ndarray  # the unshrunk estimate
    se: np.ndarray  # its match-clustered SE, NaN where undefined
    tau: Tau2
    prior: float
    applies: np.ndarray  # bool: the players whose shrunk value uses this B
    factor: np.ndarray  # B, NaN for the other players
    shrunk: np.ndarray  # the shrunk estimate, read where ``applies``

    def zero_rules(self) -> tuple[np.ndarray, np.ndarray]:
        """The players where B must be 0 by criterion 3: an undefined SE, and tau^2 = SE = 0."""
        defined = np.isfinite(self.se)
        return self.applies & ~defined, self.applies & defined & (self.tau.value == 0.0) & (self.se == 0.0)


def _parts(ranking: RoleRanking, with_usual: np.ndarray, fallback: np.ndarray) -> tuple[Part, ...]:
    est, level, pool = ranking.estimates, ranking.replacement_level, ranking.pool
    gap = est.clutch - est.usual
    return (
        Part(USUAL_BASELINE, est.usual, est.se_usual, ranking.tau2_usual, level, with_usual, ranking.b_usual,
             ranking.shrunk_usual),
        Part(PRESSURE_GAP, gap, est.se_clutch_minus_usual, ranking.tau2_gap, 0.0, with_usual, ranking.b_gap,
             ranking.shrunk_gap),
        Part(UPLIFT, est.uplift, est.se_uplift, ranking.tau2_uplift, 0.0, pool, ranking.b_uplift,
             ranking.shrunk_uplift),
        Part(CLUTCH_LEVEL, est.clutch, est.se_clutch, ranking.tau2_clutch, level, fallback, ranking.b_clutch,
             ranking.shrunk_clutch),
    )  # fmt: skip


def _check_part(part: Part, what: str) -> None:
    """B and the shrunk values of one estimate (criteria 1-3, 5 and 14)."""
    np.testing.assert_array_equal(
        np.isfinite(part.factor), part.applies, err_msg=f"{what}: B is defined exactly for the players it applies to"
    )
    a = part.applies
    x, se, b, shrunk = part.estimate[a], part.se[a], part.factor[a], part.shrunk[a]
    t, prior = part.tau.value, part.prior
    assert np.isfinite(x).all() and np.isfinite(shrunk).all(), f"{what}: a missing estimate or shrunk value"
    assert ((b >= 0.0) & (b <= 1.0)).all(), f"{what}: B outside [0, 1]: {b.tolist()}"
    undefined, zero = (mask[a] for mask in part.zero_rules())
    assert (b[undefined | zero] == 0.0).all(), f"{what}: B is not 0 for an undefined SE or tau^2 = SE = 0"
    regular = ~(undefined | zero)
    np.testing.assert_allclose(
        b[regular], t / (t + se[regular] ** 2), rtol=0.0, atol=B_TOL, err_msg=f"{what}: B != tau^2 / (tau^2 + SE^2)"
    )
    if t == 0.0:
        assert (b == 0.0).all(), f"{what}: tau^2 is 0 but some B is not 0 (criterion 5: full shrinkage)"
    else:
        assert (b[regular] > 0.0).all(), f"{what}: tau^2 {t!r} > 0 with a defined SE but B is 0"
    np.testing.assert_allclose(
        shrunk, prior + b * (x - prior), rtol=0.0, atol=TOL, err_msg=f"{what}: shrunk != prior + B (estimate - prior)"
    )
    assert (shrunk[b == 0.0] == prior).all(), f"{what}: B is 0 but the shrunk value is not the prior {prior!r}"
    outside = (shrunk < np.minimum(x, prior) - TOL) | (shrunk > np.maximum(x, prior) + TOL)
    assert not outside.any(), (
        f"{what}: shrunk {shrunk[outside].tolist()} not between the estimate {x[outside].tolist()} and the prior "
        f"{prior!r} (criterion 14)"
    )


def _named_ids(warning: RatingWarning) -> tuple[int, ...]:
    found = _ID_LIST.search(warning.message)
    assert found is not None, f"no player_id list in the {warning.kind} warning: {warning.message!r}"
    return tuple(sorted(int(text) for text in found.group(1).split(", ")))


def _check_warnings(ranking: RoleRanking, parts: tuple[Part, ...], fallback: np.ndarray) -> None:
    """tau2_floor, shrinkage_zero and clutch_fallback warnings (criteria 1, 3 and 5)."""
    role, ids, notes = ranking.role, ranking.player_id, ranking.warnings
    assert all(w.role == role for w in notes), f"a {role} warning names another Role: {notes}"

    floors = Counter(w.estimate for w in notes if w.kind == "tau2_floor")
    expected_floors = Counter(part.label for part in parts if part.tau.value == 0.0 and part.applies.any())
    assert floors == expected_floors, (
        f"{role}: tau2_floor warnings for {dict(floors)}; expected {dict(expected_floors)}"
    )

    zero_notes = [w for w in notes if w.kind == "shrinkage_zero"]
    labels = {part.label for part in parts}
    assert {w.estimate for w in zero_notes} <= labels, f"{role}: a shrinkage_zero warning for another estimate"
    for part in parts:
        expected = sorted(tuple(ids[mask].tolist()) for mask in part.zero_rules() if mask.any())
        named = sorted(_named_ids(w) for w in zero_notes if w.estimate == part.label)
        assert named == expected, f"{role} {part.label}: shrinkage_zero warnings name {named}; expected {expected}"

    named_fallback = [(w.estimate, _named_ids(w)) for w in notes if w.kind == "clutch_fallback"]
    expected_fallback = [(CLUTCH_LEVEL, tuple(ids[fallback].tolist()))] if fallback.any() else []
    assert named_fallback == expected_fallback, (
        f"{role}: clutch_fallback warnings {named_fallback}; expected {expected_fallback}"
    )


class _Shown:
    """A RoleRanking's shrinkage values for hypothesis.note, formatted only when Hypothesis prints the note."""

    COLUMNS = [
        "player_id", "in_pool", "clutch_fallback", "clutch", "se_clutch", "usual", "se_usual", "gap",
        "se_clutch_minus_usual", "uplift", "se_uplift", "B_usual", "shrunk_usual", "B_gap", "shrunk_gap", "B_clutch",
        "shrunk_clutch", "B_uplift", "shrunk_uplift",
    ]  # fmt: skip

    def __init__(self, ranking: RoleRanking) -> None:
        self.ranking = ranking

    def __repr__(self) -> str:
        r = self.ranking
        taus = "; ".join(
            f"{label} {t.value!r} (raw {t.raw!r}, n {t.n})"
            for label, t in ((USUAL_BASELINE, r.tau2_usual), (PRESSURE_GAP, r.tau2_gap), (UPLIFT, r.tau2_uplift),
                             (CLUTCH_LEVEL, r.tau2_clutch))
        )  # fmt: skip
        warnings = "\n".join(f"  {w.kind} [{w.estimate}]: {w.message}" for w in r.warnings)
        frame = r.to_frame()[self.COLUMNS].to_string()
        return f"{r.role}: Replacement_Level {r.replacement_level!r}; tau^2 {taus}\n{frame}\nwarnings:\n{warnings}"


def _check_role(example: Example, role: str, pct: float) -> None:
    case, mult, only = example.case, example.mult, example.rankable_only
    ids = example.eligible[role]
    ref = _reference(case, role, ids, mult, only)
    note(f"{role}: Eligible_Players {ids.tolist()}; pool from the entries {ids[ref.pool].tolist()}")
    data = prepare_role(case.contrib, role, ids)
    if not ref.pool.any():
        event("a Role without a pool player: rank_role raises ValueError")
        with pytest.raises(ValueError, match="can be ranked"):
            rank_role(data, mult, replacement_pct=pct, rankable_only=only)
        return
    ranking = rank_role(data, mult, replacement_pct=pct, rankable_only=only)
    note(_Shown(ranking))
    est, pool = ranking.estimates, ref.pool

    # The pool, the Usual_Baselines and the fallback players (criterion 1).
    np.testing.assert_array_equal(ranking.player_id, ids, err_msg=f"{role}: player_id")
    np.testing.assert_array_equal(ranking.pool, pool, err_msg=f"{role}: the pool")
    with_usual, fallback = pool & ref.matched, pool & ~ref.matched
    np.testing.assert_array_equal(
        pool & np.isfinite(est.usual), with_usual, err_msg=f"{role}: a pool Usual_Baseline exactly with a Matched_Cell"
    )
    np.testing.assert_array_equal(ranking.fallback, fallback, err_msg=f"{role}: the fallback players")

    # Replacement_Level (criterion 6), before any Shrinkage; Uplift and its SE are against it.
    np.testing.assert_allclose(
        est.clutch[pool], ref.clutch[pool], rtol=1e-12, atol=0.0, err_msg=f"{role}: unshrunk Clutch_Level of the pool"
    )
    level = ranking.replacement_level
    expected_level = float(np.percentile(ref.clutch[pool], pct, method="linear"))
    assert abs(level - expected_level) <= TOL, f"{role}: Replacement_Level {level!r}; expected {expected_level!r}"
    assert est.replacement_level == level, f"{role}: the estimates carry Replacement_Level {est.replacement_level!r}"
    np.testing.assert_array_equal(ranking.gap, est.clutch - est.usual, err_msg=f"{role}: Pressure_Gap g = C - O")

    # tau^2 (criteria 4-5), B (criteria 1-3) and the bounds (criterion 14) of each estimate.
    parts = _parts(ranking, with_usual, fallback)
    for part in parts:
        _check_tau2(part.tau, part.estimate[pool], part.se[pool], f"{role} {part.label}")
        _check_part(part, f"{role} {part.label}")
    for name, values, where in (
        ("shrunk Usual_Baseline", ranking.shrunk_usual, with_usual),
        ("shrunk Pressure_Gap", ranking.shrunk_gap, with_usual),
        ("shrunk Uplift", ranking.shrunk_uplift, pool),
        ("shrunk Clutch_Level", ranking.shrunk_clutch, pool),
    ):
        np.testing.assert_array_equal(np.isfinite(values), where, err_msg=f"{role}: where the {name} is defined")
    np.testing.assert_allclose(
        ranking.shrunk_clutch[with_usual],
        (ranking.shrunk_usual + ranking.shrunk_gap)[with_usual],
        rtol=0.0,
        atol=TOL,
        err_msg=f"{role}: shrunk Clutch_Level != shrunk Usual_Baseline + shrunk Pressure_Gap",
    )
    _check_warnings(ranking, parts, fallback)

    if fallback.any():
        event("a fallback player (no Usual_Baseline) in a pool")
    twins = case.twins.get(role)
    if twins is not None and np.isin(twins, ids[pool]).all():
        event("both TIES twins in a pool")
    for part in parts:
        if not part.applies.any():
            continue
        undefined, zero = part.zero_rules()
        b = part.factor[part.applies]
        if part.tau.value == 0.0:
            event(f"{part.label}: tau^2 floored at 0")
        if ((b > 0.0) & (b < 1.0)).any():
            event(f"{part.label}: partial shrinkage (0 < B < 1)")
        if undefined.any():
            event(f"{part.label}: B = 0 for an undefined SE")
        if zero.any():
            event(f"{part.label}: B = 0 for tau^2 = SE = 0")


@given(example=_examples())
def test_rank_role_shrinkage_stays_between_each_estimate_and_its_prior(example: Example, settings: Settings) -> None:
    """rank_role's Replacement_Level, tau^2, Shrinkage_Factors, shrunk values and their warnings follow Requirement
    18, criteria 1-6, and every shrunk part lies between its estimate and its prior (criterion 14).

    **Validates: Requirements 18.1, 18.2, 18.3, 18.4, 18.5, 18.6, 18.14**
    """
    pct = settings.get("design_choices.D12_replacement_pct")
    mult = example.mult
    event(f"multiplicities: {'all ones' if mult is None else 'a Bootstrap_Resample'}, "
          f"rankable_only {example.rankable_only}")  # fmt: skip
    note(f"multiplicities: {'all ones' if mult is None else mult.tolist()}; rankable_only {example.rankable_only}")
    for role in example.case.roles:
        _check_role(example, role, pct)


# ------------------------------------------------------------------------------- tau2 and shrink

#: Estimates in runs per over: any value, a few repeated values (twins), or missing.
_ESTIMATES = st.one_of(st.floats(-200.0, 200.0), st.sampled_from((0.0, 6.0, -6.0)), st.just(math.nan))
#: Match-clustered SEs: undefined, 0, a few repeated values, or any value from 1e-6 to 50.
_SES = st.one_of(st.just(math.nan), st.just(0.0), st.sampled_from((0.5, 3.0)), st.floats(1e-6, 50.0))
_TAU2 = st.one_of(st.just(0.0), st.floats(1e-6, 1e4))
_PRIORS = st.one_of(st.just(0.0), st.floats(-200.0, 200.0))


@st.composite
def _arrays(draw: Any) -> tuple[np.ndarray, np.ndarray]:
    """0-12 estimates and as many SEs."""
    n = draw(st.integers(0, 12))
    estimates = draw(st.lists(_ESTIMATES, min_size=n, max_size=n))
    ses = draw(st.lists(_SES, min_size=n, max_size=n))
    return np.array(estimates, dtype=np.float64), np.array(ses, dtype=np.float64)


@given(values=_arrays())
def test_tau2_is_the_floored_method_of_moments_value(values: tuple[np.ndarray, np.ndarray]) -> None:
    """tau^2 is max(0, sample variance (n - 1) - mean squared SE) over the pairs with a defined estimate and SE,
    and 0 when fewer than 2 such pairs leave it unestimated (Requirement 18, criteria 4 and 5).

    **Validates: Requirements 18.4, 18.5**
    """
    estimates, ses = values
    result = tau2(estimates, ses)
    note(f"estimates {estimates.tolist()}; SEs {ses.tolist()}; tau2 {result}")
    _check_tau2(result, estimates, ses, "tau2 of the drawn arrays")
    event("tau^2 > 0" if result.value > 0.0 else "tau^2 floored at 0")


@given(values=_arrays(), tau=_TAU2, prior=_PRIORS)
def test_shrink_stays_between_the_estimate_and_its_prior(
    values: tuple[np.ndarray, np.ndarray], tau: float, prior: float
) -> None:
    """B = tau^2 / (tau^2 + SE^2) in [0, 1], set to 0 for an undefined SE or tau^2 = SE = 0, and the shrunk value
    prior + B (estimate - prior) lies between the estimate and the prior (Requirement 18, criteria 1-3 and 14).

    **Validates: Requirements 18.1, 18.2, 18.3, 18.14**
    """
    estimates, ses = values
    result = shrink(estimates, ses, tau, prior)
    note(f"estimates {estimates.tolist()}; SEs {ses.tolist()}; tau^2 {tau!r}; prior {prior!r}; {result}")
    defined = np.isfinite(ses)
    zero = defined & (tau == 0.0) & (ses == 0.0)
    np.testing.assert_array_equal(result.undefined_se, ~defined, err_msg="undefined_se is set exactly for a NaN SE")
    np.testing.assert_array_equal(result.zero_variance, zero, err_msg="zero_variance is set exactly for tau^2 = SE = 0")

    b = result.factor
    assert ((b >= 0.0) & (b <= 1.0)).all(), f"B outside [0, 1]: {b.tolist()}"
    assert (b[~defined | zero] == 0.0).all(), "B is not 0 for an undefined SE or tau^2 = SE = 0"
    regular = defined & ~zero
    np.testing.assert_allclose(b[regular], tau / (tau + ses[regular] ** 2), rtol=0.0, atol=B_TOL)
    if tau == 0.0:
        assert (b == 0.0).all(), "tau^2 is 0 but some B is not 0"
    else:
        assert (b[regular] > 0.0).all(), "tau^2 > 0 with a defined SE but B is 0"

    finite = np.isfinite(estimates)
    assert np.isnan(result.value[~finite]).all(), "a missing estimate must give a missing shrunk value"
    x, shrunk, bx = estimates[finite], result.value[finite], b[finite]
    np.testing.assert_allclose(shrunk, prior + bx * (x - prior), rtol=0.0, atol=TOL)
    assert (shrunk[bx == 0.0] == prior).all(), "B is 0 but the shrunk value is not the prior"
    outside = (shrunk < np.minimum(x, prior) - TOL) | (shrunk > np.maximum(x, prior) + TOL)
    assert not outside.any(), f"shrunk {shrunk[outside].tolist()} not between {x[outside].tolist()} and {prior!r}"
    if finite.any():
        event("B = 0 for an undefined SE" if (~defined & finite).any() else "every SE defined")
        if zero.any():
            event("B = 0 for tau^2 = SE = 0")
        if ((bx > 0.0) & (bx < 1.0)).any():
            event("partial shrinkage (0 < B < 1)")
