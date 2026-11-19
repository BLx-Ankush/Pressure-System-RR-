"""Usual_Baseline and Uplift unit tests (Requirement 16, criteria 9 and 13, with criteria 4, 5, 6, 10 and 12).

Three hand-built players go through the Rating_Engine's aggregation. Their value entries, one per
Delivery with its NBV, become a contributions table through ``tests.strategies.contribution_frame``
(Requirement 16, criterion 2, transcribed), then ``Contributions.from_frame`` and ``Aggregator``.
The tests read the Matched_Cell shares from ``Aggregator.cell_table`` and the estimates from
``Aggregator.estimates``. Every expected value is worked out by hand below; none comes from the
code under test. One more check ties the ``cell_table`` shares to the Usual_Baseline that
``estimates`` returns, so the shares tested are the ones the baseline uses. Each test runs in
both Roles on the same entries, since the formulas do not depend on the Role.

Matches. The Aggregator reads each Delivery_Weight from the contributions, so the weights are set
directly: match 101 weighs 1 (the last match date), 102 weighs 0.5 (three years earlier, about one
D9 half-life) and 103 weighs 1.25 (a knockout on the last date, D10). Every weight and NBV is a
multiple of 1/4, so the weighted sums are exact and only the divisions round.

Cells are the D13 innings x Phase codes 3 x (innings - 1) + Phase: 0, 1 and 2 are the powerplay,
middle and death of innings 1, and 3, 4 and 5 those of innings 2. NBVs are in runs. W and S are a
cell's sums of Delivery_Weight and of Delivery_Weight x NBV, and mu = S_ord / W_ord is its weighted
mean NBV on Ordinary_Balls in runs per ball. Estimates are 6 x runs per ball, in runs per over.

FULL (player 11): every cell is a Matched_Cell.

====  ==============  =====  =====  =====================  =====  ======  =====
cell  Pressure_Balls  W_pr   share  Ordinary_Balls         W_ord  S_ord   mu
====  ==============  =====  =====  =====================  =====  ======  =====
0     101: +2         1      2/11   101: +1.5, 102: -1.5   1.5    0.75    0.5
1     101: -1         1      2/11   101: -0.25             1      -0.25   -0.25
2     103: +2         1.25   5/22   103: +0.25             1.25   0.3125  0.25
3     102: +1         0.5    1/11   102: +1                0.5    0.5     1
4     102: +3         0.5    1/11   101: -0.5              1      -0.5    -0.5
5     103: -2         1.25   5/22   103: +0.75             1.25   0.9375  0.75
====  ==============  =====  =====  =====================  =====  ======  =====

W_pr = 5.5, so each share is W_pr,c / 5.5 and the shares sum to 22/22 = 1. They follow the
Delivery_Weight, not the ball count (1/6 each). Cell 0's mean is weighted: (1.5 - 0.75) / 1.5 =
0.5, where the unweighted mean is 0. The cell 4 Pressure_Ball and Ordinary_Ball are in different
matches, which still makes a Matched_Cell.

* Usual_Baseline O = 6 x sum of share x mu = 6 x (2 - 1 + 1.25 + 2 - 1 + 3.75) / 22 = 6 x 7/22 =
  21/11 (about 1.909).
* Clutch_Level C = 6 x S_pr / W_pr = 6 x (2 - 1 + 2.5 + 0.5 + 1.5 - 2.5) / 5.5 = 6 x 3 / 5.5 =
  36/11 (about 3.273).
* Unmatched_Pressure_Share 0.

PARTIAL (player 12): Matched_Cells 0 and 4, cells 2 and 5 with Pressure_Balls only, and cell 1
with Ordinary_Balls only.

====  ================  =====  =================  =====  =====  ==  =======  ============
cell  Pressure_Balls    W_pr   Ordinary_Balls     W_ord  S_ord  mu  matched  share
====  ================  =====  =================  =====  =====  ==  =======  ============
0     101: +1           1      102: -1            0.5    -0.5   -1  yes      1/2.75 = 4/11
1     none              0      101: +4            1      4      4   no       0
2     101: -2           1      none                                 no       0
4     103: +2, 102: -1  1.75   101: +2, 102: -1   1.5    1.5    1   yes      1.75/2.75 = 7/11
5     102: +3           0.5    none                                 no       0
====  ================  =====  =================  =====  =====  ==  =======  ============

* W_pr = 4.25, of which 2.75 is in the Matched_Cells and 1.5 in cells 2 and 5, so the
  Unmatched_Pressure_Share is 1.5 / 4.25 = 6/17.
* The shares, taken over the Matched_Cells only, sum to 1. O = 6 x (4/11 x -1 + 7/11 x 1) = 18/11
  (about 1.636). Wrong rules give other values: shares over all the pressure weight give 6 x 3/17
  = 18/17; a weighted mean of every Ordinary_Ball gives 6 x 5/3 = 10, and one of the Matched_Cells'
  Ordinary_Balls gives 6 x 1/2 = 3.

NO_MATCH (player 13): Pressure_Balls in cells 0 (101: +1) and 2 (103: -2), Ordinary_Balls in
cells 1 (101: +0.5) and 3 (102: -0.5). No cell holds both, so there is no Matched_Cell: O is
missing (NaN, not 0), the Unmatched_Pressure_Share is 1 and every share is 0. C = 6 x (1 - 2.5) /
2.25 = -4.

Uplift U = C - max(Replacement_Level, O), and C - Replacement_Level when O is missing (criteria 6
and 10). The Replacement_Level (RL) is passed in, as ``rank_role`` does after computing it from the
Eligible_Players' Clutch_Levels:

* FULL with RL 3, above O = 21/11: U = 36/11 - 3 = 3/11 (criterion 13). Taking d runs off every
  Ordinary_Ball NBV of FULL takes d off each mu and so 6d off O, to 21/11 - 6d, further below RL.
  The edit leaves every Clutch_Level, and so the Replacement_Level, unchanged, and U stays 3/11.
* FULL with RL 1, below O: U = C - O = 15/11.
* NO_MATCH with RL -1.5: U = -4 + 1.5 = -2.5. A baseline recorded as 0 rather than missing would
  give -4 - max(-1.5, 0) = -4. With RL 3: U = -7.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from superr.rating import Aggregator, CompClass, Contributions, Period
from superr.value import CELLS, ROLES
from tests.strategies import contribution_frame

if TYPE_CHECKING:
    from superr.rating import PlayerEstimates

#: Criteria 9 and 13: the shares sum to 1, and Uplift matches, within 1e-9 (runs per over).
TOL = 1e-9

FULL, PARTIAL, NO_MATCH = 11, 12, 13
PRESSURE, ORDINARY = True, False

#: The match universe (module docstring): row i is match index i.
MATCHES = pd.DataFrame(
    {
        "match_id": np.array([101, 102, 103], dtype=np.int64),
        "match_date": pd.to_datetime(["2026-11-01", "2023-11-01", "2026-11-01"]),
        "calendar_year": np.array([2026, 2023, 2026], dtype=np.int64),
        "comp_class": np.full(3, CompClass.IPL, dtype=np.int8),
        "period": np.array([Period.LATE, Period.EARLY, Period.LATE], dtype=np.int8),
        "delivery_weight": np.array([1.0, 0.5, 1.25]),
    }
)

#: Each player's Deliveries: (match_id, cell, pressure class, NBV in runs), as in the module docstring.
ENTRIES: Mapping[int, tuple[tuple[int, int, bool, float], ...]] = {
    FULL: (
        (101, 0, PRESSURE, 2.0), (101, 0, ORDINARY, 1.5), (102, 0, ORDINARY, -1.5),
        (101, 1, PRESSURE, -1.0), (101, 1, ORDINARY, -0.25),
        (103, 2, PRESSURE, 2.0), (103, 2, ORDINARY, 0.25),
        (102, 3, PRESSURE, 1.0), (102, 3, ORDINARY, 1.0),
        (102, 4, PRESSURE, 3.0), (101, 4, ORDINARY, -0.5),
        (103, 5, PRESSURE, -2.0), (103, 5, ORDINARY, 0.75),
    ),
    PARTIAL: (
        (101, 0, PRESSURE, 1.0), (102, 0, ORDINARY, -1.0),
        (101, 1, ORDINARY, 4.0),
        (101, 2, PRESSURE, -2.0),
        (103, 4, PRESSURE, 2.0), (102, 4, PRESSURE, -1.0), (101, 4, ORDINARY, 2.0), (102, 4, ORDINARY, -1.0),
        (102, 5, PRESSURE, 3.0),
    ),
    NO_MATCH: (
        (101, 0, PRESSURE, 1.0), (103, 2, PRESSURE, -2.0),
        (101, 1, ORDINARY, 0.5), (102, 3, ORDINARY, -0.5),
    ),
}  # fmt: skip

# Hand-computed values (module docstring), in runs per over unless stated.
FULL_SHARES = (2 / 11, 2 / 11, 5 / 22, 1 / 11, 1 / 11, 5 / 22)  # cells 0-5
FULL_ORDINARY_MEANS = (3.0, -1.5, 1.5, 6.0, -3.0, 4.5)  # 6 x mu, cells 0-5
FULL_USUAL = 21 / 11
FULL_CLUTCH = 36 / 11
PARTIAL_CELLS = (0, 1, 2, 4, 5)
PARTIAL_MATCHED = (True, False, False, True, False)
PARTIAL_PRESSURE_WEIGHTS = (1.0, 0.0, 1.0, 1.75, 0.5)
PARTIAL_SHARES = (4 / 11, 0.0, 0.0, 7 / 11, 0.0)
PARTIAL_ORDINARY_MEANS = (-6.0, 24.0, math.nan, 6.0, math.nan)  # NaN: no Ordinary_Ball in the cell
PARTIAL_UNMATCHED_SHARE = 6 / 17  # a share from 0 to 1, not runs per over
PARTIAL_USUAL = 18 / 11
NO_MATCH_CLUTCH = -4.0

#: Replacement_Levels for FULL: one between O = 21/11 and C = 36/11, and one below O.
RL_ABOVE_USUAL = 3.0
RL_BELOW_USUAL = 1.0


def _approx(expected: float) -> Any:
    """``expected`` within the absolute tolerance TOL that criteria 9 and 13 set."""
    return pytest.approx(expected, rel=0, abs=TOL)


def _aggregator(role: str, lower: Mapping[int, float] | None = None) -> Aggregator:
    """The Aggregator of the three players' Deliveries in ``role``.

    ``lower[player]`` runs are taken off every Ordinary_Ball NBV of that player; the
    Pressure_Balls keep their NBVs.
    """
    drop = dict(lower or {})
    rows = [
        (role, player, match_id, cell, pressure, False, nbv if pressure else nbv - drop.get(player, 0.0))
        for player, deliveries in ENTRIES.items()
        for match_id, cell, pressure, nbv in deliveries
    ]
    entries = pd.DataFrame(rows, columns=["role", "player_id", "match_id", "cell", "pressure", "charge", "value"])
    contrib = Contributions.from_frame(contribution_frame(entries, MATCHES), MATCHES["match_id"])
    return Aggregator(contrib, role=role)


def _cells(agg: Aggregator, player: int) -> pd.DataFrame:
    """The player's ``cell_table`` rows, indexed by cell."""
    table = agg.cell_table()
    return table[table["player_id"] == player].set_index("cell").sort_index()


def _one(agg: Aggregator, player: int, replacement_level: float | None = None) -> PlayerEstimates:
    """The PlayerEstimates of one player under all-ones multiplicities (the Main_Ranking)."""
    estimates = agg.estimates(subset=[player], replacement_level=replacement_level)
    assert estimates.player_id.tolist() == [player]
    return estimates


@pytest.mark.parametrize("role", ROLES)
def test_matched_cell_shares_sum_to_one_when_every_cell_is_matched(role: str) -> None:
    """Criterion 9 for FULL: six Delivery_Weight shares that sum to 1, and O = 21/11 (criterion 4)."""
    agg = _aggregator(role)
    cells = _cells(agg, FULL)
    assert cells.index.tolist() == list(CELLS)
    assert cells["matched"].all()
    assert math.fsum(cells["share"]) == _approx(1.0)
    np.testing.assert_allclose(cells["share"], FULL_SHARES, rtol=0, atol=TOL)
    np.testing.assert_allclose(cells["ordinary_mean"], FULL_ORDINARY_MEANS, rtol=0, atol=TOL)

    estimates = _one(agg, FULL)
    assert estimates.n_matched_cells[0] == len(CELLS)
    assert estimates.unmatched_share[0] == 0.0  # criterion 5: every cell with a Pressure_Ball is matched
    assert estimates.usual[0] == _approx(FULL_USUAL)
    # The cell_table shares are the ones behind the Usual_Baseline.
    assert estimates.usual[0] == _approx(math.fsum(cells["share"] * cells["ordinary_mean"]))


@pytest.mark.parametrize("role", ROLES)
def test_matched_cell_shares_sum_to_one_with_an_unmatched_cell(role: str) -> None:
    """Criterion 9 for PARTIAL: the Matched_Cell shares sum to 1, and O = 18/11 uses those cells only."""
    agg = _aggregator(role)
    cells = _cells(agg, PARTIAL)
    assert cells.index.tolist() == list(PARTIAL_CELLS)
    assert cells["matched"].tolist() == list(PARTIAL_MATCHED)
    np.testing.assert_allclose(cells["w_pressure"], PARTIAL_PRESSURE_WEIGHTS, rtol=0, atol=TOL)
    matched = cells[cells["matched"]]
    assert math.fsum(matched["share"]) == _approx(1.0)
    np.testing.assert_allclose(cells["share"], PARTIAL_SHARES, rtol=0, atol=TOL)  # 0 off the Matched_Cells
    np.testing.assert_allclose(cells["ordinary_mean"], PARTIAL_ORDINARY_MEANS, rtol=0, atol=TOL)

    estimates = _one(agg, PARTIAL)
    assert estimates.n_matched_cells[0] == 2
    assert estimates.unmatched_share[0] == _approx(PARTIAL_UNMATCHED_SHARE)  # criterion 5
    assert estimates.usual[0] == _approx(PARTIAL_USUAL)
    assert estimates.usual[0] == _approx(math.fsum(matched["share"] * matched["ordinary_mean"]))


@pytest.mark.parametrize("drop", [0.25, 5.0])
@pytest.mark.parametrize("role", ROLES)
def test_uplift_is_clutch_minus_replacement_level_when_the_baseline_is_below_it(role: str, drop: float) -> None:
    """Criterion 13: with O below RL, U = C - RL = 3/11, and lowering O by 6 x ``drop`` leaves U unchanged."""
    before = _one(_aggregator(role), FULL, RL_ABOVE_USUAL)
    assert before.usual[0] == _approx(FULL_USUAL)
    assert before.usual[0] < RL_ABOVE_USUAL
    assert before.clutch[0] == _approx(FULL_CLUTCH)
    assert before.uplift[0] == _approx(FULL_CLUTCH - RL_ABOVE_USUAL)  # 3/11

    # Every Ordinary_Ball NBV of FULL lowered by `drop` runs; Replacement_Level set afterwards.
    after = _one(_aggregator(role, lower={FULL: drop}), FULL).with_replacement_level(RL_ABOVE_USUAL)
    assert after.usual[0] == _approx(FULL_USUAL - 6 * drop)
    assert after.clutch[0] == _approx(FULL_CLUTCH)
    assert after.uplift[0] == _approx(FULL_CLUTCH - RL_ABOVE_USUAL)
    assert after.uplift[0] == _approx(before.uplift[0])


@pytest.mark.parametrize("role", ROLES)
def test_uplift_is_clutch_minus_baseline_when_the_baseline_is_above_replacement_level(role: str) -> None:
    """Criterion 6: with O = 21/11 above RL 1, U = C - O = 15/11."""
    estimates = _one(_aggregator(role), FULL, RL_BELOW_USUAL)
    assert estimates.usual[0] > RL_BELOW_USUAL
    assert estimates.clutch[0] == _approx(FULL_CLUTCH)
    assert estimates.uplift[0] == _approx(FULL_CLUTCH - FULL_USUAL)  # 15/11


@pytest.mark.parametrize(("replacement_level", "expected_uplift"), [(-1.5, -2.5), (3.0, -7.0)])
@pytest.mark.parametrize("role", ROLES)
def test_missing_baseline_gives_clutch_minus_replacement_level(
    role: str, replacement_level: float, expected_uplift: float
) -> None:
    """Criteria 10 and 12 for NO_MATCH: O is missing, not 0, and U = C - RL."""
    agg = _aggregator(role)
    cells = _cells(agg, NO_MATCH)
    assert not cells["matched"].any()
    assert (cells["share"] == 0.0).all()

    estimates = _one(agg, NO_MATCH, replacement_level)
    assert estimates.n_matched_cells[0] == 0
    assert np.isnan(estimates.usual[0])
    assert estimates.unmatched_share[0] == 1.0
    assert estimates.clutch[0] == _approx(NO_MATCH_CLUTCH)
    assert estimates.uplift[0] == _approx(expected_uplift)
    assert estimates.uplift[0] == _approx(NO_MATCH_CLUTCH - replacement_level)
