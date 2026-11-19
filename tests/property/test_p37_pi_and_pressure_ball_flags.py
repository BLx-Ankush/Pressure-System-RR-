# Feature: pressure-rating-system, Property 37: PI and Pressure_Ball flags
"""Property 37: PI and Pressure_Ball flags (Requirement 13, criteria 1-5, 7, 10 and 11).

**Validates: Requirements 13.1, 13.2, 13.3, 13.4, 13.5, 13.7, 13.10, 13.11**

Property 37 (design): for any set of Deliveries with WP, LI, Stress, Role masks and Innings_Groups
(with continuous LI and Stress and at least 400 Live_Balls per Role and Innings_Group): Live_Balls
are exactly those with WP in the D1 band; every Dead_Ball has PI 0 and no Pressure_Ball flag;
every Live_Ball has 0 < PI <= 100 and PI equals the formula of Requirement 13, criterion 2;
Pressure_Balls make up between 24.5% and 25.5% of each Role x Innings_Group's Live_Balls;
Deliveries with equal PI get the same flag; every Role Delivery is exactly one of Pressure_Ball or
Ordinary_Ball; Wides carry no batter-Role values; and any non-finite WP, LI or Stress stops the
engine with per-Innings_Group counts.

The tests call the functions of ``superr.pressure`` as the ``pressure`` stage calls them
(``live_balls``, ``role_masks``, ``finite_guard``, ``pressure_index``, ``pressure_flags`` and
``ordinary_balls``), with the Config's D1 band, D2 alpha and D3 share, and the innings as the rank
groups (the D4 scope ``innings_group``). Every expected value is worked out in this file from the
glossary definitions; none is read back from the engine.

The oracle (``_oracle``)
    For the n Live_Balls of one Role in one Innings_Group, the Percentile_Rank of each LI and each
    Stress is (below + 0.5 x equal) / n, counted by comparing every pair of values. The expected PI
    is 100 x [alpha x rank(LI) + (1 - alpha) x rank(Stress)] in float64, and the engine's PI must
    match it within PI_TOL (1e-9). The flags need exact ties, which float64 can split in the last
    bit, so the oracle also orders the PI exactly: with m = 2 x below + equal behind each rank and
    alpha = a / d as a binary fraction, PI = 100 x K / (2 n d) for the whole number
    K = a x m(LI) + (d - a) x m(Stress). A Live_Ball is a Pressure_Ball when the Percentile_Rank of
    its K among the n values of K exceeds 1 - D3, compared in whole numbers.

The checks of one Role (``_role_problems``, ``_group_problems``)
    * outside the Role (a Wide, for the batter Role): PI NaN and both flags <NA> (criterion 11);
    * every Delivery of the Role is exactly one of a Pressure_Ball or an Ordinary_Ball, neither flag
      <NA> (criterion 5);
    * Dead_Balls of the Role: PI exactly 0 and no Pressure_Ball (criterion 3);
    * Live_Balls of the Role, in each Innings_Group: 0 < PI <= 100 (criterion 11) and PI within
      PI_TOL of the formula (criterion 2); PI equal exactly where K is equal and rising with K, so
      Live_Balls whose PI are equal in exact arithmetic get bit-identical PI; the flags equal the
      oracle's (criterion 4); every flagged PI lies above every unflagged one, so equal PI get the
      same flag and the flags are monotone in PI; and the share of Pressure_Balls (criterion 7).

The share of Pressure_Balls (criterion 7)
    With e_at Live_Balls sharing the lowest flagged PI and e_below sharing the highest unflagged
    PI, criterion 4 puts the number of Pressure_Balls in [D3 x n - e_below / 2, D3 x n + e_at / 2).
    The test checks that interval for every Role and Innings_Group, in exact arithmetic. Criterion
    7's band, D3 +- 0.005 (24.5% to 25.5%), follows from it only while those two ties are small:
    with at most 4 Live_Balls in each and n >= 400 every whole count in the interval lies inside
    the band. Continuous LI and Stress do not rule out larger ties, because PI averages two ranks:
    Live_Balls whose rank pairs have the same sum share a PI (about 30,000 shared values in the
    delivered data). In a simulation of independent, continuous LI and Stress with 400-600
    Live_Balls, about 1 draw in 3,000 put a tie of 5-7 Live_Balls next to the cut and the share
    outside the band, so a band asserted on every draw would fail now and then on a correct
    engine. The test therefore asserts the band for each Role and Innings_Group where every whole
    count of the interval lies inside it, and records an event for the others.

Deliveries (``test_pi_and_pressure_ball_flags``)
    Each example draws, in each Innings_Group, 400-600 Live Batter_Deliveries (Legal_Balls, up to
    10% of them No_Balls), 0-40 Live Wides and 0-300 Dead_Balls (up to 30% of them Wides), and
    shuffles the rows of both innings together, so each Role has at least 400 Live_Balls in each
    Innings_Group and the bowler Role's reference sets hold Live Wides that the batter Role's lack.
    WP is drawn inside the D1 band for Live_Balls and outside it for Dead_Balls, and 0-3 Deliveries
    per value and innings take a WP exactly at a bound or one float step inside it (Live), or one
    step outside a bound, 0 or 1 (Dead). LI is lognormal and the two Stress columns logistic, each
    Stress correlated with LI through a Gaussian copula (rho from -0.3 to 0.6), so all three are
    continuous; 0-3% of the Live_Balls of an innings then copy the LI, a Stress or all three values
    of another, which adds exact ties. ``stress_bat`` is NaN on every Wide, as in the pressure
    table, and in half the examples LI and the Stress columns are NaN or infinite on a fifth of the
    Dead_Balls. finite_guard checks none of those values and pressure_index ranks none, so the
    guard must pass and both Roles must pass every check above, the share band included.
    ``live_balls`` must flag exactly the Deliveries with lower <= WP <= upper, and ``role_masks``
    the Batter_Deliveries (no Wide) and the Bowler_Deliveries (every Delivery).

Ties at the cut (``test_tied_pi_get_one_flag_even_at_the_cut``)
    Each Innings_Group holds B + t + A Live Batter_Deliveries whose PI order is fixed by
    construction: B below a tie of t (2-8) Live_Balls that share one PI, and A (0-8) above it. The
    tie either repeats one LI and Stress (equal inputs) or, with the Config's alpha of 0.5, pairs t
    successive LI ranks with the same Stress ranks in reverse order, so the members' ranks differ
    and only their sums agree. B is drawn (0-30) or set to the count that puts the tie's
    Percentile_Rank of PI exactly at 1 - D3 (B = t + 3A for D3 = 0.25). Dead_Balls, some of them
    Wides, are mixed in, and no Wide is Live, so both Roles rank the same Live_Balls. For both
    Roles each member of the tie must get the same PI, bit for bit, and the same flag; a tie at
    exactly 1 - D3 is not flagged (the rank must be greater), while every Live_Ball above it is
    flagged and none below it is; and every check of ``_role_problems`` holds except the share band
    (n is below 400). Explicit examples fix ties at the cut of both kinds, one that holds the whole
    top of its innings so the innings has no Pressure_Ball, and ties across which the cut falls.

The guard (``test_finite_guard_stops_on_non_finite_inputs_with_per_innings_group_counts``)
    Each example draws 10-60 Deliveries per Innings_Group with finite WP, LI and Stress
    (``stress_bat`` NaN on every Wide) and puts NaN, inf or -inf into WP, LI and each Stress
    column: into values that finite_guard checks (WP on every Delivery, LI on the Live_Balls,
    ``stress_bat`` on the batter Role's Live_Balls and the bowler columns on every Live_Ball), into
    values it skips (Dead_Balls, and Wides for ``stress_bat``), or anywhere; in half the examples
    only into skipped values. The values go in as float64 arrays or as pandas Float64 arrays, in
    which a NaN is <NA>. A WP that is not finite makes its Delivery a Dead_Ball, and
    ``live_balls`` must agree. When no checked value is bad the guard returns None; otherwise it
    raises PressureInputError, a StageError with exit code 5, whose ``counts`` give, for exactly
    the affected quantities, the number of affected Deliveries in innings 1 and in innings 2, and
    whose message names each affected quantity with the same two counts.
"""

from __future__ import annotations

import math
import re
from bisect import bisect_left, bisect_right
from fractions import Fraction
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, event, example, given, note
from hypothesis import strategies as st

from superr.errors import StageError
from superr.pressure import (
    BATTER,
    BOWLER,
    STRESS_COLUMNS,
    PressureInputError,
    finite_guard,
    live_balls,
    ordinary_balls,
    pressure_flags,
    pressure_index,
    role_masks,
)
from superr.reconstruct import DeliveryClass

if TYPE_CHECKING:
    from superr.settings import Settings

# ------------------------------------------------------------------------------------ constants

INNINGS_GROUPS: tuple[int, int] = (1, 2)
MIN_LIVE = 400  # Property 37: at least 400 Live_Balls per Role and Innings_Group
PI_TOL = 1e-9  # the largest |engine PI - formula PI|
BAND_HALF_WIDTH = Fraction(5, 1000)  # criterion 7: the share lies within D3 +- 0.005 (24.5% to 25.5%)
EXIT_STAGE = 5  # the exit code of a StageError

LEGAL, WIDE, NO_BALL = int(DeliveryClass.LEGAL), int(DeliveryClass.WIDE), int(DeliveryClass.NO_BALL)
#: Each Role with the Stress column of its PI.
ROLE_STRESS: tuple[tuple[str, str], ...] = ((BATTER, "stress_bat"), (BOWLER, "stress_bowl"))
#: The quantities that finite_guard reports, in its order.
QUANTITIES: tuple[str, ...] = ("WP", "LI", *STRESS_COLUMNS)


class Rules(NamedTuple):
    """The Config values that PI and the flags read."""

    band: tuple[float, float]  # D1: a Live_Ball has lower <= WP <= upper
    alpha: float  # D2: the weight of rank(LI) in PI
    share: float  # D3: the pressure share


@pytest.fixture(scope="module")
def rules(settings: Settings) -> Rules:
    """D1, D2 and D3 from the Config (``design_choices``)."""
    lower, upper = settings.get("design_choices.D1_live_band")
    return Rules(
        (float(lower), float(upper)),
        float(settings.get("design_choices.D2_alpha")),
        float(settings.get("design_choices.D3_pressure_share")),
    )


# ---------------------------------------------------------------------------------- the oracle

def _below_equal(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each value, the number of values below it and equal to it (itself included), comparing every pair."""
    below = (values[None, :] < values[:, None]).sum(axis=1)
    equal = (values[None, :] == values[:, None]).sum(axis=1)
    return below, equal


def _whole_numerators(keys: list[int]) -> list[int]:
    """2 x below + equal for each whole number of ``keys``, exactly: bisect_left counts the keys below, bisect_right
    the keys below or equal."""
    ordered = sorted(keys)
    return [bisect_left(ordered, key) + bisect_right(ordered, key) for key in keys]


class Oracle(NamedTuple):
    """The expected PI and Pressure_Ball flags of one Role's Live_Balls in one Innings_Group."""

    pi: np.ndarray  # 100 x [alpha x rank(LI) + (1 - alpha) x rank(Stress)], in float64
    keys: list[int]  # K: the exact PI is 100 x K / (2 n d)
    flags: np.ndarray  # Pressure_Ball: the Percentile_Rank of K is greater than 1 - D3
    at_cut: np.ndarray  # the Percentile_Rank of K is exactly 1 - D3


def _oracle(li: np.ndarray, stress: np.ndarray, alpha: float, share: float) -> Oracle:
    """The Oracle for the LI and Stress of the n >= 1 Live_Balls of one Role and Innings_Group (module docstring)."""
    n = len(li)
    below_li, equal_li = _below_equal(li)
    below_stress, equal_stress = _below_equal(stress)
    rank_li = (below_li + 0.5 * equal_li) / n
    rank_stress = (below_stress + 0.5 * equal_stress) / n
    pi = 100.0 * (alpha * rank_li + (1.0 - alpha) * rank_stress)
    a, d = alpha.as_integer_ratio()
    keys = [
        a * (2 * int(b_li) + int(e_li)) + (d - a) * (2 * int(b_stress) + int(e_stress))
        for b_li, e_li, b_stress, e_stress in zip(below_li, equal_li, below_stress, equal_stress)
    ]
    p, q = share.as_integer_ratio()  # D3 = p / q: rank m / 2n > 1 - D3 exactly when m q > 2n (q - p)
    numerators = _whole_numerators(keys)
    flags = np.array([m * q > 2 * n * (q - p) for m in numerators], dtype=bool)
    at_cut = np.array([m * q == 2 * n * (q - p) for m in numerators], dtype=bool)
    return Oracle(pi, keys, flags, at_cut)


def _cut_ties(oracle: Oracle) -> tuple[int, int]:
    """(e_at, e_below): the Live_Balls that share the lowest flagged PI and the highest unflagged PI (0 when none)."""
    flagged = [key for key, flag in zip(oracle.keys, oracle.flags) if flag]
    unflagged = [key for key, flag in zip(oracle.keys, oracle.flags) if not flag]
    return (
        flagged.count(min(flagged)) if flagged else 0,
        unflagged.count(max(unflagged)) if unflagged else 0,
    )


# ---------------------------------------------------------------------------------- the checks

def _group_problems(
    label: str,
    rows: np.ndarray,
    li: np.ndarray,
    stress: np.ndarray,
    pi: np.ndarray,
    pressure: np.ndarray,
    rules: Rules,
    *,
    band: bool,
) -> list[str]:
    """One line per check that the Live_Balls ``rows`` (positions) of one Role and Innings_Group break.

    ``pressure`` holds every Delivery's Pressure_Ball flag (False for <NA>); ``band``: also check
    criterion 7's band, for examples with at least MIN_LIVE Live_Balls per Role and Innings_Group.
    """
    n = len(rows)
    if band and n < MIN_LIVE:
        return [f"{label}: strategy bug: {n} Live_Balls, but Property 37 needs at least {MIN_LIVE}"]
    if not n:
        return []
    problems: list[str] = []
    got = pi[rows]
    oracle = _oracle(li[rows], stress[rows], rules.alpha, rules.share)

    # 0 < PI <= 100 (criterion 11) and PI = 100 x [alpha x rank(LI) + (1 - alpha) x rank(Stress)] (criterion 2).
    bad = ~((got > 0.0) & (got <= 100.0))
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        problems.append(
            f"{label}: {int(bad.sum())} Live_Balls have a PI outside (0, 100], first row {rows[i]}: {got[i]!r}"
        )
    error = np.abs(got - oracle.pi)
    bad = ~(error <= PI_TOL)  # a NaN PI is bad too
    if bad.any():
        i = int(np.argmax(np.nan_to_num(error, nan=np.inf)))
        problems.append(
            f"{label}: {int(bad.sum())} Live_Balls have a PI more than {PI_TOL:g} from 100 x [alpha x rank(LI) + "
            f"(1 - alpha) x rank(Stress)]; worst row {rows[i]}: {got[i]!r}, expected {oracle.pi[i]!r}"
        )

    # Equal in exact arithmetic <=> bit-identical, and PI rising with the exact PI.
    order = sorted(range(n), key=oracle.keys.__getitem__)
    keys = [oracle.keys[i] for i in order]
    ordered = got[order]
    same = np.array([low == high for low, high in zip(keys, keys[1:])], dtype=bool)
    for what, bad in (
        ("the same exact PI but different PI values", same & (ordered[1:] != ordered[:-1])),
        ("different exact PI but the same PI value", ~same & (ordered[1:] == ordered[:-1])),
        ("PI values in the opposite order to the exact PI", ordered[1:] < ordered[:-1]),
    ):
        if bad.any():
            i = int(np.flatnonzero(bad)[0])
            problems.append(
                f"{label}: {int(bad.sum())} pairs of Live_Balls next to each other in exact PI order have {what}, "
                f"first rows {rows[order[i]]} and {rows[order[i + 1]]}: {ordered[i]!r} and {ordered[i + 1]!r}"
            )

    # Pressure_Ball when the Percentile_Rank of PI is greater than 1 - D3 (criterion 4).
    flags = pressure[rows]
    bad = flags != oracle.flags
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        problems.append(
            f"{label}: {int(bad.sum())} Live_Balls have the wrong Pressure_Ball flag, first row {rows[i]} with PI "
            f"{got[i]!r}: {bool(flags[i])}, expected {bool(oracle.flags[i])}"
        )
    if flags.any() and (~flags).any() and not got[flags].min() > got[~flags].max():
        problems.append(
            f"{label}: a Pressure_Ball has PI {got[flags].min()!r}, not above the PI {got[~flags].max()!r} of an "
            "Ordinary Live_Ball, so equal PI got different flags or the flags are not monotone in PI"
        )

    # The share of Pressure_Balls (criterion 7; module docstring).
    count = int(flags.sum())
    e_at, e_below = _cut_ties(oracle)
    d3 = Fraction(rules.share)
    low, high = d3 * n - Fraction(e_below, 2), d3 * n + Fraction(e_at, 2)
    note(f"{label}: {count} Pressure_Balls of {n} Live_Balls; ties next to the cut of {e_at} (flagged) and {e_below}")
    if not low <= count < high:
        problems.append(
            f"{label}: {count} Pressure_Balls of {n} Live_Balls, outside [D3 x n - {e_below} / 2, D3 x n + {e_at} / 2) "
            f"= [{float(low):g}, {float(high):g})"
        )
    if band:
        tie = max(e_at, e_below)
        event("a PI tie next to the cut" if tie > 1 else "no PI tie next to the cut")
        floor, ceiling = (d3 - BAND_HALF_WIDTH) * n, (d3 + BAND_HALF_WIDTH) * n
        fewest, most = math.ceil(low), math.ceil(high) - 1  # the whole counts in [low, high)
        if not (floor <= fewest and most <= ceiling):
            event(f"share band not implied: a PI tie of {tie} Live_Balls next to the cut")
        elif not floor <= count <= ceiling:
            problems.append(
                f"{label}: the {count} Pressure_Balls are {100 * count / n:.3f}% of the {n} Live_Balls, outside "
                f"{float(100 * floor / n):g}% to {float(100 * ceiling / n):g}%"
            )
    return problems


def _role_problems(
    role: str,
    *,
    innings: np.ndarray,
    li: np.ndarray,
    stress: np.ndarray,
    live: np.ndarray,
    mask: np.ndarray,
    pi: object,
    flags: object,
    ordinary: object,
    rules: Rules,
    band: bool,
) -> list[str]:
    """One line per clause of Property 37 that one Role's PI, Pressure_Balls and Ordinary_Balls break.

    ``live`` and ``mask`` are the expected Live_Ball flags and Role mask; ``li`` and ``stress`` the
    inputs of the PI; ``pi``, ``flags`` and ``ordinary`` what pressure_index, pressure_flags and
    ordinary_balls returned.
    """
    n_rows = len(live)
    if not (isinstance(pi, np.ndarray) and pi.dtype == np.float64 and pi.shape == (n_rows,)):
        return [f"{role} Role: pressure_index returned {type(pi).__name__}; expected {n_rows:,} float64 values"]
    for name, values in (("pressure_flags", flags), ("ordinary_balls", ordinary)):
        if not (isinstance(values, pd.arrays.BooleanArray) and len(values) == n_rows):
            what = type(values).__name__
            return [f"{role} Role: {name} returned {what}; expected a BooleanArray of {n_rows:,} flags"]
    assert isinstance(flags, pd.arrays.BooleanArray) and isinstance(ordinary, pd.arrays.BooleanArray)
    pressure_na = np.asarray(flags.isna(), dtype=bool)
    ordinary_na = np.asarray(ordinary.isna(), dtype=bool)
    pressure = flags.to_numpy(dtype=bool, na_value=False)
    ordinary_flags = ordinary.to_numpy(dtype=bool, na_value=False)
    dead = mask & ~live
    problems: list[str] = []
    for what, bad in (
        # Criterion 11: no PI and no flag outside the Role (a Wide, for the batter Role).
        ("Deliveries outside the Role have a PI", ~mask & ~np.isnan(pi)),
        ("Deliveries outside the Role have a Pressure_Ball flag", ~mask & ~pressure_na),
        ("Deliveries outside the Role have an Ordinary_Ball flag", ~mask & ~ordinary_na),
        # Criterion 5: each Delivery of the Role is exactly one of the two.
        (
            "Deliveries of the Role are not exactly one of a Pressure_Ball or an Ordinary_Ball",
            mask & (pressure_na | ordinary_na | (pressure == ordinary_flags)),
        ),
        # Criterion 3: PI 0 and no Pressure_Ball on a Dead_Ball.
        ("Dead_Balls of the Role have a PI other than 0", dead & (pi != 0.0)),
        ("Dead_Balls of the Role are Pressure_Balls", dead & pressure),
    ):
        if bad.any():
            i = int(np.flatnonzero(bad)[0])
            problems.append(
                f"{role} Role: {int(bad.sum())} {what}, first row {i} (PI {pi[i]!r}, Pressure_Ball {flags[i]}, "
                f"Ordinary_Ball {ordinary[i]})"
            )
    for group in INNINGS_GROUPS:
        rows = np.flatnonzero(mask & live & (innings == group))
        problems += _group_problems(f"{role} Role, innings {group}", rows, li, stress, pi, pressure, rules, band=band)
    return problems


def _expected_roles(cls: np.ndarray) -> dict[str, np.ndarray]:
    """The Role masks of the glossary: Batter_Deliveries are not Wides; every Delivery is a Bowler_Delivery."""
    return {BATTER: cls != WIDE, BOWLER: np.ones(len(cls), dtype=bool)}


def _check_masks(wp: np.ndarray, cls: np.ndarray, rules: Rules) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Assert that live_balls and role_masks give the expected flags (criterion 1); return the expected ones."""
    lower, upper = rules.band
    expected = (wp >= lower) & (wp <= upper)  # a NaN WP compares False: not a Live_Ball
    live = live_balls(wp, rules.band)
    assert isinstance(live, np.ndarray) and live.dtype == bool and live.shape == wp.shape, f"live_balls gave {live!r}"
    wrong = np.flatnonzero(live != expected)
    assert not len(wrong), (
        f"live_balls flags {len(wrong)} Deliveries wrongly for the band [{lower!r}, {upper!r}], first row "
        f"{wrong[0]} with WP {wp[wrong[0]]!r}: {bool(live[wrong[0]])}"
    )
    roles, expected_roles = role_masks(cls), _expected_roles(cls)
    assert set(roles) == set(expected_roles), f"role_masks gives the Roles {sorted(roles)}"
    for role, mask in expected_roles.items():
        assert np.array_equal(roles[role], mask), f"role_masks gives the wrong {role} Role mask"
    return expected, expected_roles


# ------------------------------------------------------------------------- drawing Deliveries

def _logistic(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _wp_edges(band: tuple[float, float]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """WP values at the edges of the D1 band: (Live: each bound and one float step inside it;
    Dead: one float step outside each bound, 0 and 1)."""
    lower, upper = band
    live = (lower, upper, float(np.nextafter(lower, upper)), float(np.nextafter(upper, lower)))
    outside = (float(np.nextafter(lower, -np.inf)), float(np.nextafter(upper, np.inf)), 0.0, 1.0)
    return live, tuple(value for value in outside if not lower <= value <= upper)


def _place(rng: np.random.Generator, wp: np.ndarray, values: tuple[float, ...], count: int) -> None:
    """Give ``count`` distinct Deliveries of ``wp``, drawn at random, each of ``values``, as far as ``wp`` goes."""
    spots = rng.permutation(len(wp))[: count * len(values)]
    for k, value in enumerate(values):
        wp[spots[k * count : (k + 1) * count]] = value


class Case(NamedTuple):
    """One drawn set of Deliveries (module docstring, "Deliveries")."""

    seed: int  # seeds the numpy generator that draws the values and the row order
    live_batter: tuple[int, int]  # Live Batter_Deliveries in innings 1 and 2
    live_wides: tuple[int, int]  # Live Wides in innings 1 and 2
    dead: tuple[int, int]  # Dead_Balls in innings 1 and 2
    no_ball_share: float  # the chance that a Batter_Delivery is a No_Ball
    dead_wide_share: float  # the chance that a Dead_Ball is a Wide
    at_bounds: int  # Deliveries per edge value of WP (_wp_edges) in each innings
    li_spread: float  # the standard deviation of log LI
    rho: float  # the copula correlation of each Stress with LI
    tie_share: float  # the share of Live_Balls that copy another Live_Ball's LI, a Stress, or all three
    junk_outside: bool  # NaN and infinite LI and Stress on a fifth of the Dead_Balls


CASES = st.builds(
    Case,
    seed=st.integers(0, 2**32 - 1),
    live_batter=st.tuples(st.integers(MIN_LIVE, 600), st.integers(MIN_LIVE, 600)),
    live_wides=st.tuples(st.integers(0, 40), st.integers(0, 40)),
    dead=st.tuples(st.integers(0, 300), st.integers(0, 300)),
    no_ball_share=st.sampled_from((0.0, 0.02, 0.1)),
    dead_wide_share=st.sampled_from((0.0, 0.05, 0.3)),
    at_bounds=st.integers(0, 3),
    li_spread=st.sampled_from((0.3, 1.0, 2.5)),
    rho=st.sampled_from((-0.3, 0.0, 0.3, 0.6)),
    tie_share=st.sampled_from((0.0, 0.01, 0.03)),
    junk_outside=st.booleans(),
)

#: The fewest Live_Balls Property 37 allows: MIN_LIVE per Role and Innings_Group, with no Live Wide.
AT_THE_MINIMUM = Case(
    seed=37, live_batter=(MIN_LIVE, MIN_LIVE), live_wides=(0, 0), dead=(120, 0), no_ball_share=0.02,
    dead_wide_share=0.3, at_bounds=3, li_spread=1.0, rho=0.0, tie_share=0.03, junk_outside=True,
)  # fmt: skip
#: The other ends of the ranges: many Live Wides and Dead_Balls, a wide LI spread and negative correlation.
AT_THE_MAXIMUM = Case(
    seed=2026, live_batter=(600, MIN_LIVE), live_wides=(40, 40), dead=(300, 300), no_ball_share=0.1,
    dead_wide_share=0.3, at_bounds=3, li_spread=2.5, rho=-0.3, tie_share=0.03, junk_outside=True,
)  # fmt: skip


class Deliveries(NamedTuple):
    """The Deliveries of one Case, in one shuffled row order."""

    innings: np.ndarray  # int64: the Innings_Group, 1 or 2
    cls: np.ndarray  # int8 DeliveryClass codes, as in the balls table
    wp: np.ndarray  # float64
    li: np.ndarray  # float64
    stresses: pd.DataFrame  # STRESS_COLUMNS, float64


def _deliveries(case: Case, rules: Rules) -> Deliveries:
    """The Deliveries of ``case`` (module docstring, "Deliveries")."""
    rng = np.random.default_rng(case.seed)
    lower, upper = rules.band
    live_edges, dead_edges = _wp_edges(rules.band)
    parts: list[dict[str, np.ndarray]] = []
    for group, n_batter, n_wide, n_dead in zip(INNINGS_GROUPS, case.live_batter, case.live_wides, case.dead):
        n_live = n_batter + n_wide
        n = n_live + n_dead
        position = np.arange(n)
        batter_cls = np.where(rng.random(n) < case.no_ball_share, NO_BALL, LEGAL)
        dead_cls = np.where(rng.random(n) < case.dead_wide_share, WIDE, batter_cls)
        cls = np.select([position < n_batter, position < n_live], [batter_cls, WIDE], default=dead_cls)
        wp = np.empty(n)
        wp[:n_live] = rng.uniform(lower, upper, n_live)
        high = rng.random(n_dead) < 0.5
        wp[n_live:] = np.where(high, upper + (1.0 - upper) * (1.0 - rng.random(n_dead)), lower * rng.random(n_dead))
        _place(rng, wp[:n_live], live_edges, case.at_bounds)
        _place(rng, wp[n_live:], dead_edges, case.at_bounds)

        z = rng.standard_normal((3, n))
        spread = np.sqrt(1.0 - case.rho**2)
        li = np.exp(case.li_spread * z[0])
        stress_bat = _logistic(case.rho * z[0] + spread * z[1])
        stress_bowl = _logistic(case.rho * z[0] + spread * z[2])
        copies = round(case.tie_share * n_live)  # exact ties among the Live_Balls of the innings
        if copies:
            for columns in ((li,), (stress_bat,), (stress_bowl,), (li, stress_bat, stress_bowl)):
                to, source = rng.choice(n_live, size=copies, replace=False), rng.choice(n_live, size=copies)
                for values in columns:
                    values[to] = values[source]
        extras_free = stress_bowl.copy()
        swap = rng.random(n) < 0.05
        extras_free[swap] = _logistic(rng.standard_normal(int(swap.sum())))
        if case.junk_outside:  # values that neither finite_guard nor the ranks may look at
            dead_rows = np.flatnonzero(~((wp >= lower) & (wp <= upper)))
            for values in (li, stress_bat, stress_bowl, extras_free):
                hit = dead_rows[rng.random(len(dead_rows)) < 0.2]
                values[hit] = rng.choice(np.array([np.nan, np.inf, -np.inf]), size=len(hit))
        stress_bat[cls == WIDE] = np.nan  # as in the pressure table: no batter-Role Stress on a Wide
        parts.append(
            {
                "innings": np.full(n, group, dtype=np.int64),
                "cls": cls.astype(np.int8),
                "wp": wp,
                "li": li,
                "stress_bat": stress_bat,
                "stress_bowl": stress_bowl,
                "stress_bowl_extras_free": extras_free,
            }
        )
    joined = {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}
    order = rng.permutation(len(joined["wp"]))
    shuffled = {name: values[order] for name, values in joined.items()}
    return Deliveries(
        shuffled["innings"],
        shuffled["cls"],
        shuffled["wp"],
        shuffled["li"],
        pd.DataFrame({column: shuffled[column] for column in STRESS_COLUMNS}),
    )


# ------------------------------------------------------------------------------------ the test

@given(case=CASES)
@example(case=AT_THE_MINIMUM)
@example(case=AT_THE_MAXIMUM)
def test_pi_and_pressure_ball_flags(case: Case, rules: Rules) -> None:
    """Live_Balls, PI, Pressure_Balls and Ordinary_Balls of both Roles follow Requirement 13, criteria 1-5, 7 and
    11, on at least 400 Live_Balls per Role and Innings_Group with continuous LI and Stress, and finite_guard
    passes when every value it checks is finite (criterion 10).

    **Validates: Requirements 13.1, 13.2, 13.3, 13.4, 13.5, 13.7, 13.10, 13.11**
    """
    data = _deliveries(case, rules)
    live, masks = _check_masks(data.wp, data.cls, rules)
    lower, upper = rules.band
    event("WP exactly at a D1 bound" if ((data.wp == lower) | (data.wp == upper)).any() else "no WP at a D1 bound")
    event("Live Wides" if (live & (data.cls == WIDE)).any() else "no Live Wide")

    # Every value finite_guard checks is finite; the NaN and infinite values lie where it does not look.
    roles = role_masks(data.cls)
    guard = finite_guard(wp=data.wp, li=data.li, stresses=data.stresses, live=live, roles=roles, innings=data.innings)
    assert guard is None

    problems: list[str] = []
    for role, column in ROLE_STRESS:
        stress = data.stresses[column]
        pi = pressure_index(data.li, stress, live, roles[role], data.innings, rules.alpha)
        flags = pressure_flags(pi, live, roles[role], data.innings, rules.share)
        problems += _role_problems(
            role, innings=data.innings, li=data.li, stress=stress.to_numpy(), live=live, mask=masks[role], pi=pi,
            flags=flags, ordinary=ordinary_balls(flags), rules=rules, band=True,
        )  # fmt: skip
    assert not problems, "\n".join(problems)


# ---------------------------------------------------------------------------- ties at the cut

class TieBlock(NamedTuple):
    """The Live_Balls of one Innings_Group in a tie case (module docstring, "Ties at the cut")."""

    tie: int  # t: Live_Balls that share one PI
    above: int  # A: Live_Balls with a higher PI
    below: int | None  # B: Live_Balls with a lower PI; None: the B that puts the tie's rank exactly at 1 - D3
    swapped: bool  # with alpha 0.5, the tie pairs LI ranks with Stress ranks in reverse order
    dead: int  # Dead_Balls mixed in, some of them Wides


class TieCase(NamedTuple):
    seed: int  # seeds the WP, the Dead_Balls' values and the row order
    blocks: tuple[TieBlock, TieBlock]  # innings 1 and 2


TIE_BLOCKS = st.builds(
    TieBlock,
    tie=st.integers(2, 8),
    above=st.integers(0, 8),
    below=st.one_of(st.none(), st.integers(0, 30)),
    swapped=st.booleans(),
    dead=st.integers(0, 10),
)
TIE_CASES = st.builds(TieCase, seed=st.integers(0, 2**32 - 1), blocks=st.tuples(TIE_BLOCKS, TIE_BLOCKS))

#: Ties at exactly 1 - D3: two Live_Balls with swapped ranks, and four with equal inputs that hold the
#: whole top of innings 2, so that innings has no Pressure_Ball at all.
TIES_AT_THE_CUT = TieCase(
    seed=11,
    blocks=(
        TieBlock(tie=2, above=1, below=None, swapped=True, dead=4),
        TieBlock(tie=4, above=0, below=None, swapped=False, dead=3),
    ),
)
#: Ties of three across which the cut falls: flagged together in innings 1 (rank 0.8125 for D3 = 0.25),
#: left unflagged together in innings 2 (rank 0.6875).
TIES_ACROSS_THE_CUT = TieCase(
    seed=12,
    blocks=(
        TieBlock(tie=3, above=0, below=5, swapped=True, dead=2),
        TieBlock(tie=3, above=1, below=4, swapped=True, dead=2),
    ),
)


def _below_at_cut(tie: int, above: int, share: float) -> int | None:
    """The B that puts a tie of ``tie`` Live_Balls, with ``above`` above it, exactly at rank 1 - D3:
    B + t / 2 = (1 - D3) x (B + t + A). None when no whole B >= 0 does."""
    d3 = Fraction(share)
    below = ((1 - d3) * (tie + above) - Fraction(tie, 2)) / d3
    return int(below) if below.denominator == 1 and below >= 0 else None


class TieDeliveries(NamedTuple):
    """The Deliveries of one TieCase, in one shuffled row order."""

    innings: np.ndarray
    cls: np.ndarray
    wp: np.ndarray
    li: np.ndarray
    stresses: pd.DataFrame
    parts: tuple[dict[str, np.ndarray], ...]  # per innings: the positions of the below, tie and above Live_Balls
    at_cut: tuple[bool, ...]  # per innings: the tie was placed at rank exactly 1 - D3
    swapped: tuple[bool, ...]  # per innings: the tie comes from swapped ranks


def _tie_deliveries(case: TieCase, rules: Rules) -> TieDeliveries:
    """The Deliveries of ``case``: in each innings, B + t + A Live Batter_Deliveries with LI and Stress that rise
    with their rank positions, and the Dead_Balls (module docstring, "Ties at the cut")."""
    rng = np.random.default_rng(case.seed)
    lower, upper = rules.band
    dead_wp = np.array(_wp_edges(rules.band)[1])
    names = ("innings", "cls", "wp", "li", *STRESS_COLUMNS)
    columns: dict[str, list[np.ndarray]] = {name: [] for name in names}
    spans: list[dict[str, np.ndarray]] = []
    at_cut: list[bool] = []
    swapped_ties: list[bool] = []
    start = 0
    for group, block in zip(INNINGS_GROUPS, case.blocks):
        below = block.below if block.below is not None else _below_at_cut(block.tie, block.above, rules.share)
        assume(below is not None)  # only a D3 with a large denominator leaves no whole B
        assert below is not None
        swapped = block.swapped and rules.alpha == 0.5  # swapped ranks give equal sums, so equal PI only at 0.5
        tie, above = block.tie, block.above
        x_below = np.arange(1, below + 1)
        if swapped:
            x_tie = np.arange(below + 1, below + tie + 1)
            y_tie = x_tie[::-1]
            x_above = np.arange(below + tie + 1, below + tie + above + 1)
        else:
            x_tie = y_tie = np.full(tie, below + 1)
            x_above = np.arange(below + 2, below + above + 2)
        x = np.concatenate([x_below, x_tie, x_above]).astype(np.float64)  # LI rank positions
        y = np.concatenate([x_below, y_tie, x_above]).astype(np.float64)  # Stress rank positions
        n_live, n_dead = len(x), block.dead
        stress = y / (y.max() + 1.0)
        dead_values = np.where(rng.random((3, n_dead)) < 0.2, np.nan, rng.random((3, n_dead)))
        columns["innings"].append(np.full(n_live + n_dead, group, dtype=np.int64))
        live_cls = np.where(rng.random(n_live) < 0.1, NO_BALL, LEGAL)  # no Live Wide
        dead_cls = np.where(rng.random(n_dead) < 0.3, WIDE, LEGAL)
        columns["cls"].append(np.concatenate([live_cls, dead_cls]))
        columns["wp"].append(np.concatenate([rng.uniform(lower, upper, n_live), rng.choice(dead_wp, size=n_dead)]))
        columns["li"].append(np.concatenate([0.1 + 0.3 * x, 5.0 * dead_values[0]]))
        columns["stress_bat"].append(np.concatenate([stress, dead_values[1]]))
        columns["stress_bowl"].append(np.concatenate([stress**2, dead_values[2]]))
        columns["stress_bowl_extras_free"].append(np.concatenate([stress**2, dead_values[2]]))
        spans.append(
            {
                "below": start + np.arange(below),
                "tie": start + below + np.arange(tie),
                "above": start + below + tie + np.arange(above),
            }
        )
        at_cut.append(block.below is None)
        swapped_ties.append(swapped)
        start += n_live + n_dead
    joined = {name: np.concatenate(columns[name]) for name in names}
    joined["cls"] = joined["cls"].astype(np.int8)
    joined["stress_bat"][joined["cls"] == WIDE] = np.nan
    order = rng.permutation(start)
    moved_to = np.argsort(order)  # moved_to[i]: the position of row i after the shuffle
    shuffled = {name: values[order] for name, values in joined.items()}
    return TieDeliveries(
        shuffled["innings"],
        shuffled["cls"],
        shuffled["wp"],
        shuffled["li"],
        pd.DataFrame({column: shuffled[column] for column in STRESS_COLUMNS}),
        tuple({part: moved_to[rows] for part, rows in span.items()} for span in spans),
        tuple(at_cut),
        tuple(swapped_ties),
    )


@given(case=TIE_CASES)
@example(case=TIES_AT_THE_CUT)
@example(case=TIES_ACROSS_THE_CUT)
def test_tied_pi_get_one_flag_even_at_the_cut(case: TieCase, rules: Rules) -> None:
    """Live_Balls with equal PI get bit-identical PI and one Pressure_Ball flag, also when their Percentile_Rank of
    PI is exactly 1 - D3 or the cut falls inside the tie (Requirement 13, criteria 2-5 and 11).

    **Validates: Requirements 13.2, 13.3, 13.4, 13.5, 13.11**
    """
    data = _tie_deliveries(case, rules)
    live, masks = _check_masks(data.wp, data.cls, rules)
    assert not (live & (data.cls == WIDE)).any(), "strategy bug: a Live Wide would give the Roles different Live_Balls"
    roles = role_masks(data.cls)
    d3 = Fraction(rules.share)
    problems: list[str] = []
    for role, column in ROLE_STRESS:
        stress = data.stresses[column]
        pi = pressure_index(data.li, stress, live, roles[role], data.innings, rules.alpha)
        flags = pressure_flags(pi, live, roles[role], data.innings, rules.share)
        problems += _role_problems(
            role, innings=data.innings, li=data.li, stress=stress.to_numpy(), live=live, mask=masks[role], pi=pi,
            flags=flags, ordinary=ordinary_balls(flags), rules=rules, band=False,
        )  # fmt: skip
        pressure = flags.to_numpy(dtype=bool, na_value=False)
        for group, part, at_cut, swapped in zip(INNINGS_GROUPS, data.parts, data.at_cut, data.swapped):
            label = f"{role} Role, innings {group}"
            rows = np.flatnonzero(masks[role] & live & (data.innings == group))
            tie, below, above = part["tie"], part["below"], part["above"]
            n = len(rows)
            assert n == len(tie) + len(below) + len(above), f"strategy bug: {label} has {n} Live_Balls"
            oracle = _oracle(data.li[rows], stress.to_numpy()[rows], rules.alpha, rules.share)
            in_tie = np.isin(rows, tie)
            assert len({key for key, member in zip(oracle.keys, in_tie) if member}) == 1, (
                f"strategy bug: the tie of {label} is not one exact PI"
            )
            assert not at_cut or oracle.at_cut[in_tie].all(), f"strategy bug: the tie of {label} is not at the cut"

            values = np.unique(pi[tie])
            if len(values) != 1:
                problems.append(f"{label}: the {len(tie)} tied Live_Balls got the PI values {values.tolist()}")
            if len(np.unique(pressure[tie])) != 1:
                problems.append(f"{label}: the tied Live_Balls got different flags: {pressure[tie].tolist()}")
            rank_position = len(below) + Fraction(len(tie), 2)  # (below + 0.5 x equal) of the tie
            cut = (1 - d3) * n
            if rank_position == cut:
                kind = "tie at rank exactly 1 - D3"
                if pressure[tie].any():
                    problems.append(f"{label}: a tie whose Percentile_Rank of PI is exactly 1 - D3 was flagged")
                if not pressure[above].all() or pressure[below].any():
                    problems.append(
                        f"{label}: with the tie at exactly 1 - D3, {int(pressure[above].sum())} of the {len(above)} "
                        f"Live_Balls above it and {int(pressure[below].sum())} of the {len(below)} below it are flagged"
                    )
            elif len(below) < cut < len(below) + len(tie):
                kind = "cut inside the tie, " + ("tie flagged" if pressure[tie].all() else "tie not flagged")
            else:
                kind = "tie clear of the cut"
            event(kind)
            event("tie from swapped ranks" if swapped else "tie from equal inputs")
    assert not problems, "\n".join(problems)


# ----------------------------------------------------------------------------------- the guard

CHECKED, SKIPPED, ANYWHERE = "checked", "skipped", "anywhere"  # where a GuardCase puts its bad values
_BAD_VALUES: dict[str, float] = {"nan": np.nan, "inf": np.inf, "-inf": -np.inf}


class GuardCase(NamedTuple):
    """One drawn input of finite_guard (module docstring, "The guard")."""

    seed: int  # seeds the numpy generator that draws the values and the rows that get bad values
    rows: tuple[int, int]  # Deliveries in innings 1 and 2
    wide_share: float  # the chance that a Delivery is a Wide
    dead_share: float  # the chance that a Delivery is a Dead_Ball
    bad: tuple[tuple[str, tuple[str, ...]], ...]  # per QUANTITIES: where the bad values go, and their kinds
    only_skipped: bool  # put every bad value where finite_guard does not look
    nullable: bool  # pass pandas Float64 arrays, in which a NaN is <NA>


GUARD_CASES = st.builds(
    GuardCase,
    seed=st.integers(0, 2**32 - 1),
    rows=st.tuples(st.integers(10, 60), st.integers(10, 60)),
    wide_share=st.sampled_from((0.0, 0.1, 0.3)),
    dead_share=st.sampled_from((0.0, 0.2, 0.5)),
    bad=st.tuples(
        *(
            st.tuples(
                st.sampled_from((CHECKED, SKIPPED, ANYWHERE)),
                st.lists(st.sampled_from(tuple(_BAD_VALUES)), max_size=4).map(tuple),
            )
            for _ in QUANTITIES
        )
    ),
    only_skipped=st.booleans(),
    nullable=st.booleans(),
)

#: Bad values only where finite_guard does not look, the Wides' NaN stress_bat among them: the guard passes.
BAD_ONLY_WHERE_SKIPPED = GuardCase(
    seed=5, rows=(12, 9), wide_share=0.3, dead_share=0.2, bad=tuple((ANYWHERE, ("nan", "inf")) for _ in QUANTITIES),
    only_skipped=True, nullable=False,
)  # fmt: skip
#: Every quantity bad on values the guard checks, as <NA>, inf and -inf: the guard reports all five.
BAD_EVERYWHERE_CHECKED = GuardCase(
    seed=6, rows=(12, 9), wide_share=0.3, dead_share=0.2,
    bad=tuple((CHECKED, ("nan", "inf", "-inf")) for _ in QUANTITIES), only_skipped=False, nullable=True,
)  # fmt: skip


class GuardInputs(NamedTuple):
    """The inputs of one GuardCase and the counts finite_guard must report."""

    innings: np.ndarray
    cls: np.ndarray
    wp: np.ndarray
    li: np.ndarray
    stresses: dict[str, np.ndarray]
    expected: dict[str, dict[int, int]]  # per affected quantity, the affected Deliveries in innings 1 and 2
    skipped: int  # bad values that the guard must ignore


def _inject(rng: np.random.Generator, values: np.ndarray, pool: np.ndarray, kinds: tuple[str, ...]) -> None:
    """Put the bad values ``kinds`` into distinct random Deliveries of ``pool`` (positions), as far as it goes."""
    take = min(len(kinds), len(pool))
    if take:
        values[rng.choice(pool, size=take, replace=False)] = [_BAD_VALUES[kind] for kind in kinds[:take]]


def _guard_inputs(case: GuardCase, rules: Rules) -> GuardInputs:
    """The Deliveries of ``case`` with their bad values, and the counts the guard must report (Requirement 13,
    criterion 10): WP on every Delivery, LI on every Live_Ball, ``stress_bat`` on the batter Role's Live_Balls
    and the bowler columns on every Live_Ball (every Delivery is a Bowler_Delivery)."""
    rng = np.random.default_rng(case.seed)
    lower, upper = rules.band
    innings = rng.permutation(np.repeat(np.array(INNINGS_GROUPS, dtype=np.int64), case.rows))
    n = len(innings)
    batter_cls = np.where(rng.random(n) < 0.1, NO_BALL, LEGAL)
    cls = np.where(rng.random(n) < case.wide_share, WIDE, batter_cls).astype(np.int8)
    dead_wp = np.array(_wp_edges(rules.band)[1])
    wp = np.where(rng.random(n) < case.dead_share, rng.choice(dead_wp, size=n), rng.uniform(lower, upper, n))
    li = np.exp(rng.standard_normal(n))
    stresses = {column: _logistic(rng.standard_normal(n)) for column in STRESS_COLUMNS}
    stresses["stress_bat"][cls == WIDE] = np.nan  # as in the pressure table: no batter-Role Stress on a Wide

    where, kinds = case.bad[0]  # WP first: a WP that is not finite makes its Delivery a Dead_Ball
    if not case.only_skipped and where != SKIPPED:  # WP is checked on every Delivery, so none is skipped
        _inject(rng, wp, np.arange(n), kinds)
    live = (wp >= lower) & (wp <= upper)
    scopes = {
        "WP": np.ones(n, dtype=bool),
        "LI": live,
        "stress_bat": live & (cls != WIDE),
        "stress_bowl": live,
        "stress_bowl_extras_free": live,
    }
    values = {"WP": wp, "LI": li, **stresses}
    for quantity, (where, kinds) in zip(QUANTITIES[1:], case.bad[1:]):
        scope = scopes[quantity]
        pools = {CHECKED: scope, SKIPPED: ~scope, ANYWHERE: np.ones(n, dtype=bool)}
        _inject(rng, values[quantity], np.flatnonzero(pools[SKIPPED if case.only_skipped else where]), kinds)
    expected: dict[str, dict[int, int]] = {}
    skipped = 0
    for quantity in QUANTITIES:
        bad = ~np.isfinite(values[quantity])
        hit = bad & scopes[quantity]
        skipped += int((bad & ~scopes[quantity]).sum())
        if hit.any():
            expected[quantity] = {group: int((hit & (innings == group)).sum()) for group in INNINGS_GROUPS}
    return GuardInputs(innings, cls, wp, li, stresses, expected, skipped)


@given(case=GUARD_CASES)
@example(case=BAD_ONLY_WHERE_SKIPPED)
@example(case=BAD_EVERYWHERE_CHECKED)
def test_finite_guard_stops_on_non_finite_inputs_with_per_innings_group_counts(
    case: GuardCase, rules: Rules
) -> None:
    """finite_guard stops the Run exactly when WP, LI or a Role's Stress is missing or not finite where PI needs it,
    with the affected Deliveries of each quantity per Innings_Group, and ignores the values PI does not use
    (Requirement 13, criteria 1 and 10).

    **Validates: Requirements 13.1, 13.10**
    """
    inputs = _guard_inputs(case, rules)
    if case.nullable:
        wp, li = pd.array(inputs.wp, dtype="Float64"), pd.array(inputs.li, dtype="Float64")
        stresses = pd.DataFrame({name: pd.array(inputs.stresses[name], dtype="Float64") for name in STRESS_COLUMNS})
    else:
        wp, li = inputs.wp, inputs.li
        stresses = pd.DataFrame(inputs.stresses)
    lower, upper = rules.band
    expected_live = (inputs.wp >= lower) & (inputs.wp <= upper)
    live = live_balls(wp, rules.band)
    wrong = np.flatnonzero(live != expected_live)
    assert not len(wrong), f"live_balls flags {len(wrong)} Deliveries wrongly, first with WP {wp[wrong[0]]!r}"
    roles = role_masks(inputs.cls)
    note(f"expected counts {inputs.expected}; {inputs.skipped} bad values where the guard does not look")
    event("bad values where the guard does not look" if inputs.skipped else "no bad value outside the guard's scope")

    if not inputs.expected:
        event("guard passes")
        assert finite_guard(wp=wp, li=li, stresses=stresses, live=live, roles=roles, innings=inputs.innings) is None
        return
    event(f"guard raises for {len(inputs.expected)} of the {len(QUANTITIES)} quantities")
    with pytest.raises(PressureInputError) as raised:
        finite_guard(wp=wp, li=li, stresses=stresses, live=live, roles=roles, innings=inputs.innings)
    error = raised.value
    assert isinstance(error, StageError) and error.exit_code == EXIT_STAGE, f"{type(error).__name__}: {error}"
    counts = {quantity: dict(per_group) for quantity, per_group in error.counts.items()}
    assert counts == inputs.expected, f"counts {counts}, expected {inputs.expected}"
    message = str(error)
    for quantity, per_group in inputs.expected.items():
        named = re.search(rf"(?<!\w){re.escape(quantity)}(?!\w)[^;]*", message)
        assert named, f"the message does not name {quantity}: {message}"
        for group, count in per_group.items():
            assert re.search(rf"innings {group}: {count:,}(?!\d)", named.group(0)), (
                f"the message does not give {count} affected Deliveries of {quantity} in innings {group}: {message}"
            )
