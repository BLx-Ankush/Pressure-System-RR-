"""Hand-computed Net Ball Value cases (Requirement 15, criterion 18).

Each test builds a few synthetic Deliveries with fixed Wicket_Cost Omega, fixed M4 probability
rows, fixed Extras_Model probabilities (p_W 0.04 and p_N 0.01 on every Delivery) and fixed
Class_Run_Values. It checks the Value_Engine against NBVs worked out by hand, within 1e-6 runs.
The working is in the comments. No expected value comes from the code under test.

Cases, by criterion of Requirement 15:

* 1 and 2: ``test_expected_bat_runs_weights_each_class_run_value_by_its_probability`` and
  ``test_batter_nbv_on_legal_balls_charges_omega_only_when_the_striker_is_out`` (dot ball,
  boundary, striker caught);
* 3: ``test_batter_nbv_on_a_no_ball_uses_the_free_hit_batter_view``;
* 4 to 7: ``test_bowler_nbv_on_legal_balls_charges_bat_runs_but_never_byes_or_leg_byes``;
* 8: ``test_bowler_credit_only_for_codes_1_2_3_5_and_6``;
* 9 and 10: ``test_dismissed_batter_comes_from_dismissed_player_id_with_a_counted_striker_fallback``;
* 11: ``test_dismissal_charges_for_a_non_striker_run_out_and_a_striker_out_off_a_wide``;
* 12: ``test_retire_out_gives_no_omega_penalty_and_no_dismissal_charge``;
* 13: ``test_wide_extras_cost_is_team_runs_plus_a_replacement_ball_with_its_own_free_hit_flag``;
* 14: ``test_no_ball_extras_cost_adds_penalty_and_bat_runs_a_cleared_replacement_ball_and_the_premium``;
* 20 and 21: ``test_cell_tables_pool_one_sum_over_one_count_for_empty_training_fold_cells`` and
  ``test_bowler_nbv_uses_the_pooled_values_of_empty_training_fold_cells``.

The dismissal flags that the NBV functions read come from ``reconstruct.wicket_flags``, with the
Config's D14 codes, and ``reconstruct.dismissed_batter_flags``, which reads
``dismissed_player_id``, as in the ``reconstruct`` stage. Criteria 8 to 12 are therefore checked
from the raw dismissal code and id.

Every Delivery gets a ``bowler_view_nofh`` row that differs from its ``bowler_view`` row. In a
Run the two agree off Free_Hits; here they differ so that each test can tell which view a term
reads. The ``free_hit`` column only documents a case: ``bowler_nbv`` sees the Free_Hit flag
through the M4 rows it is given.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from superr import reconstruct
from superr.reconstruct import DeliveryClass
from superr.value import (
    BatterValue,
    BowlerValue,
    batter_nbv,
    bowler_nbv,
    dismissal_charges,
    dismissed_batters,
    expected_bat_runs,
    extras_cell_means,
    free_hit_premium,
)

if TYPE_CHECKING:
    from superr.settings import Settings

#: Criterion 18: the Value_Engine reproduces the hand-computed NBV within 1e-6 runs.
TOL = 1e-6

LEGAL, WIDE, NO_BALL = int(DeliveryClass.LEGAL), int(DeliveryClass.WIDE), int(DeliveryClass.NO_BALL)

#: Dismissal codes (``dismissal_type_id``) of Requirement 15, criterion 8, and code 10.
BOWLED, CAUGHT, LBW, RUN_OUT, STUMPED, HIT_WICKET, RETIRE_OUT, OBSTRUCTING = 1, 2, 3, 4, 5, 6, 9, 10

#: Counted_Wicket codes: 1-6 and 9 (glossary), and 10, which the Auditor classifies as a
#: Counted_Wicket as delivered (design F-7). Code 10 is not a Bowler_Wicket.
COUNTED_CODES = (BOWLED, CAUGHT, LBW, RUN_OUT, STUMPED, HIT_WICKET, RETIRE_OUT, OBSTRUCTING)

STRIKER, NON_STRIKER, NEXT_BATTER = 101, 102, 103

#: The Config's D14 codes (the d14 fixture): Bowler_Wicket codes, run-out code, Retire Out code.
D14Codes = tuple[list[int], int, int]

# Class_Run_Values in outcome.CLASSES order: dot, r1, r2, r3, r4, r6, other_runs, bowler_wicket (W),
# run_out (R). The last three are means of a fold's Legal_Balls, so any non-negative value will do.
CRV = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 5.0, 0.25, 0.5])

# Fixed M4 probability rows, in the same class order; each sums to 1. E(p) = sum of p x CRV.
P_BV = [0.40, 0.30, 0.08, 0.02, 0.08, 0.04, 0.02, 0.04, 0.02]
#   E = 0.30x1 + 0.08x2 + 0.02x3 + 0.08x4 + 0.04x6 + 0.02x5 + 0.04x0.25 + 0.02x0.5
#     = 0.30 + 0.16 + 0.06 + 0.32 + 0.24 + 0.10 + 0.01 + 0.01 = 1.20;  p[W] + p[R] = 0.04 + 0.02 = 0.06
P_BV_FH = [0.30, 0.30, 0.10, 0.02, 0.12, 0.10, 0.02, 0.02, 0.02]
#   E = 0.30 + 0.20 + 0.06 + 0.48 + 0.60 + 0.10 + 0.005 + 0.01 = 1.755;  p[W] + p[R] = 0.02 + 0.02 = 0.04
P_WV = [0.35, 0.35, 0.08, 0.02, 0.08, 0.04, 0.02, 0.05, 0.01]
#   E = 0.35 + 0.16 + 0.06 + 0.32 + 0.24 + 0.10 + 0.0125 + 0.005 = 1.2475;  p[W] = 0.05
P_WV_FH = [0.30, 0.32, 0.10, 0.02, 0.10, 0.08, 0.02, 0.04, 0.02]
#   bowler view of a Free_Hit Delivery (its own flag set):
#   E = 0.32 + 0.20 + 0.06 + 0.40 + 0.48 + 0.10 + 0.01 + 0.01 = 1.58;  p[W] = 0.04
P_WV_NOFH = [0.40, 0.35, 0.07, 0.01, 0.07, 0.03, 0.01, 0.05, 0.01]
#   E = 0.35 + 0.14 + 0.03 + 0.28 + 0.18 + 0.05 + 0.0125 + 0.005 = 1.0475

#: Extras_Model (p_wide, p_noball) of every Delivery: 1 - p_W - p_N = 0.95.
P_EXTRAS = (0.04, 0.01)

#: Fixed Training_Folds means of fold 1, cell 0 for criteria 4-14: the cell_means and fh_premium
#: columns that bowler_nbv reads. With them, on every Delivery of fold 1 and cell 0:
#:   E[Wide cost] = wide_runs + E(p_wv)                                           = 1.2 + E(p_wv)
#:   E[NB cost]   = nb_penalty_runs + nb_bat_runs + E(p_wv_nofh) + free_hit_premium = 1.1 + 1.5 + 1.0475 + 0.8 = 4.4475
CELL_TABLE = pd.DataFrame(
    {
        "fold": [1],
        "cell": [0],
        "wide_runs": [1.2],
        "nb_penalty_runs": [1.1],
        "nb_bat_runs": [1.5],
        "free_hit_premium": [0.8],
    }
)

# EV_bowl = 0.95 x (Omega x p_wv[W] - E(p_wv)) - 0.04 x E[Wide cost] - 0.01 x E[NB cost] (criteria 5 and 6).
# With CELL_TABLE and P_WV: E[Wide cost] = 1.2 + 1.2475 = 2.4475 and E[NB cost] = 4.4475, so the extras terms
# are -0.04 x 2.4475 - 0.01 x 4.4475 = -0.0979 - 0.044475 = -0.142375, and
#   Omega 15: 0.95 x (15 x 0.05 - 1.2475) - 0.142375 = 0.95 x (-0.4975) - 0.142375 = -0.472625 - 0.142375 = -0.615
#   Omega 10: 0.95 x (10 x 0.05 - 1.2475) - 0.142375 = 0.95 x (-0.7475) - 0.142375 = -0.710125 - 0.142375 = -0.8525
# With CELL_TABLE and P_WV_FH: E[Wide cost] = 1.2 + 1.58 = 2.78, so
#   Omega 15: 0.95 x (15 x 0.04 - 1.58) - 0.04 x 2.78 - 0.01 x 4.4475 = -0.931 - 0.1112 - 0.044475 = -1.086675
EV_BOWL_15 = -0.615
EV_BOWL_10 = -0.8525
EV_BOWL_FH_15 = -1.086675

# Batter EV = E(p) - Omega x (p[W] + p[R]) (criterion 2): 1.20 - 0.06 x Omega with P_BV on a Legal_Ball, and
# 1.755 - 0.04 x Omega with P_BV_FH on a No_Ball (criterion 3).
#   P_BV:    Omega 15 -> 1.20 - 0.90 = 0.30;  14 -> 1.20 - 0.84 = 0.36;  12 -> 1.20 - 0.72 = 0.48;  10 -> 0.60
#   P_BV_FH: Omega 15 -> 1.755 - 0.60 = 1.155


# ------------------------------------------------------------------------------------------- helpers

def assert_runs(actual: Any, expected: Sequence[float]) -> None:
    """``actual`` equals ``expected`` within TOL runs, value by value (NaN matches NaN)."""
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64), rtol=0.0, atol=TOL
    )


def _each(row: Sequence[float], n: int) -> np.ndarray:
    """``row`` repeated for each of ``n`` Deliveries."""
    return np.tile(np.asarray(row, dtype=np.float64), (n, 1))


def _out(code: int, player: int | None) -> dict[str, Any]:
    """A wicket with ``dismissal_type_id`` ``code`` and ``dismissed_player_id`` ``player`` (None: blank)."""
    return {"dismissal_type_id": code, "dismissed_player_id": player}


_DEFAULTS: Mapping[str, Any] = {
    "cls": LEGAL,
    "extras_type": "None",
    "runs": 0,
    "team_runs": None,  # None: the bat runs on a Legal_Ball, 1 on a Wide, 1 + bat runs on a No_Ball
    "striker_id": STRIKER,
    "non_striker_id": NON_STRIKER,
    "dismissal_type_id": None,
    "dismissed_player_id": None,
    "free_hit": False,
    "live": True,
    "pressure_bat": False,  # missing on Wides, which have no batter-Role flag
}


def _team_runs(record: Mapping[str, Any]) -> int:
    if record["team_runs"] is not None:
        return int(record["team_runs"])
    return {LEGAL: record["runs"], WIDE: 1, NO_BALL: 1 + record["runs"]}[record["cls"]]


@pytest.fixture(scope="module")
def d14(settings: Settings) -> D14Codes:
    """The Config's D14 codes: the Bowler_Wicket codes, the run-out code and the Retire Out code."""
    return (
        settings.get("design_choices.D14_bowler_wicket_codes"),
        settings.get("design_choices.D14_run_out_code"),
        settings.get("design_choices.D14_retire_out_code"),
    )


def _deliveries(d14: D14Codes, rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Deliveries of one innings of match 1 (fold 1, cell 0, balls 1, 2, ...), each ``row`` over _DEFAULTS.

    The Counted_Wicket, Bowler_Wicket, Run_Out and Retire Out flags come from
    ``reconstruct.wicket_flags`` with the Config's D14 codes, and the dismissed batter from
    ``reconstruct.dismissed_batter_flags``, as the ``reconstruct`` stage sets them.
    """
    records = [{**_DEFAULTS, **row} for row in rows]
    n = len(records)

    def column(name: str) -> list[Any]:
        return [record[name] for record in records]

    frame = pd.DataFrame(
        {
            "row": np.arange(n, dtype=np.int64),
            "match_id": np.ones(n, dtype=np.int64),
            "innings": np.ones(n, dtype=np.int64),
            "innings_ball_num": np.arange(1, n + 1, dtype=np.int64),
            "fold": np.ones(n, dtype=np.int64),
            "cell": np.zeros(n, dtype=np.int64),
            "cls": np.array(column("cls"), dtype=np.int64),
            "extras_type": column("extras_type"),
            "runs": np.array(column("runs"), dtype=np.int64),
            "team_runs": np.array([_team_runs(record) for record in records], dtype=np.int64),
            "striker_id": np.array(column("striker_id"), dtype=np.int64),
            "non_striker_id": np.array(column("non_striker_id"), dtype=np.int64),
            "is_wicket": np.array([code is not None for code in column("dismissal_type_id")], dtype=bool),
            "dismissal_type_id": pd.array(column("dismissal_type_id"), dtype="Int64"),
            "dismissed_player_id": pd.array(column("dismissed_player_id"), dtype="Int64"),
            "free_hit": np.array(column("free_hit"), dtype=bool),
            "live": np.array(column("live"), dtype=bool),
            "pressure_bat": pd.array(
                [None if record["cls"] == WIDE else record["pressure_bat"] for record in records], dtype="boolean"
            ),
        }
    )
    wickets = reconstruct.wicket_flags(frame, COUNTED_CODES, *d14)
    return frame.join(wickets).join(reconstruct.dismissed_batter_flags(frame, wickets))


def _batter(d: pd.DataFrame, omega: Sequence[float], *, retire_out_penalty: bool = False) -> BatterValue:
    """batter_nbv with P_BV on every Legal_Ball and P_BV_FH on every No_Ball."""
    n = len(d)
    return batter_nbv(
        d, _each(P_BV, n), _each(P_BV_FH, n), CRV, np.asarray(omega, dtype=np.float64),
        retire_out_penalty=retire_out_penalty,
    )  # fmt: skip


def _bowler(
    d: pd.DataFrame, omega: Sequence[float], p_wv: Sequence[Sequence[float]] | None = None
) -> BowlerValue:
    """bowler_nbv with ``p_wv`` (default P_WV on every Delivery), P_WV_NOFH, P_EXTRAS and CELL_TABLE."""
    n = len(d)
    views = _each(P_WV, n) if p_wv is None else np.asarray(p_wv, dtype=np.float64)
    return bowler_nbv(
        d, views, _each(P_WV_NOFH, n), _each(P_EXTRAS, n), CRV, np.asarray(omega, dtype=np.float64), CELL_TABLE,
        CELL_TABLE,
    )  # fmt: skip


# --------------------------------------------------------------------------- batter NBV, criteria 1-3

def test_expected_bat_runs_weights_each_class_run_value_by_its_probability() -> None:
    # Criterion 2: the expected bat runs are the sum over M4 classes of p x Class_Run_Value.
    p = np.array([P_BV, P_WV])
    assert_runs(expected_bat_runs(p, CRV), [1.20, 1.2475])  # worked out under P_BV and P_WV
    # One row of Class_Run_Values per Delivery, from its own Training_Folds: the second Delivery's folds give
    # other_runs 5.5 and run_out 1.0, so its E rises by 0.02 x 0.5 + 0.01 x 0.5 = 0.015, to 1.2625.
    own_folds = np.array([CRV, [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 5.5, 0.25, 1.0]])
    assert_runs(expected_bat_runs(p, own_folds), [1.20, 1.2625])


def test_batter_nbv_on_legal_balls_charges_omega_only_when_the_striker_is_out(d14: D14Codes) -> None:
    # Criteria 1 and 2: NBV = bat runs - Omega x D - EV, with EV = 1.20 - 0.06 x Omega from P_BV.
    d = _deliveries(d14, [{}, {"runs": 4}, _out(CAUGHT, STRIKER)])  # dot ball, boundary four, striker caught
    value = _batter(d, [15.0, 10.0, 12.0])
    assert_runs(value.ev, [0.30, 0.60, 0.48])
    # 0 - 0 - 0.30 = -0.30;  4 - 0 - 0.60 = 3.40;  0 - 12 x 1 - 0.48 = -12.48
    assert_runs(value.nbv, [-0.30, 3.40, -12.48])
    assert value.striker_out.tolist() == [False, False, True]


def test_batter_nbv_on_a_no_ball_uses_the_free_hit_batter_view(d14: D14Codes) -> None:
    # Criterion 3: a No_Ball's EV comes from P_BV_FH, 1.755 - 0.04 x Omega; the Legal_Ball keeps P_BV.
    no_ball = {"cls": NO_BALL, "extras_type": "Overstep"}
    d = _deliveries(
        d14,
        [
            {},  # Legal dot ball
            {**no_ball, "runs": 1},  # 1 bat run off a No_Ball
            {**no_ball, "free_hit": True, **_out(RUN_OUT, STRIKER)},  # the striker run out off the next No_Ball
        ],
    )
    value = _batter(d, [15.0, 15.0, 15.0])
    # 1.20 - 0.90 = 0.30;  1.755 - 0.60 = 1.155 (P_BV would give 0.30);  1.155
    assert_runs(value.ev, [0.30, 1.155, 1.155])
    # -0.30;  1 - 0 - 1.155 = -0.155;  0 - 15 x 1 - 1.155 = -16.155
    assert_runs(value.nbv, [-0.30, -0.155, -16.155])
    assert value.striker_out.tolist() == [False, False, True]


# --------------------------------------------------------------------------- bowler NBV, criteria 4-8

def test_bowler_nbv_on_legal_balls_charges_bat_runs_but_never_byes_or_leg_byes(
    d14: D14Codes,
) -> None:
    # Criteria 4 to 7: NBV = -conceded + Omega x B - Extras_Cost - EV_bowl, with Extras_Cost 0 on a Legal_Ball and
    # conceded = its bat runs. EV_bowl is EV_BOWL_15 (Omega 15) or EV_BOWL_10 (Omega 10), worked out above.
    d = _deliveries(
        d14,
        [
            {},  # dot ball
            {"runs": 4},  # 4 bat runs
            {"extras_type": "legbyes", "team_runs": 1},  # 1 leg-bye: 1 team run, 0 bat runs
            {"extras_type": "byes", "team_runs": 4},  # 4 byes
        ],
    )
    value = _bowler(d, [15.0, 10.0, 15.0, 15.0])
    assert_runs(value.exp_wide_cost, [2.4475] * 4)  # 1.2 + 1.2475 (criterion 6)
    assert_runs(value.exp_nb_cost, [4.4475] * 4)  # 1.1 + 1.5 + 1.0475 + 0.8 (criterion 6)
    assert_runs(value.ev, [EV_BOWL_15, EV_BOWL_10, EV_BOWL_15, EV_BOWL_15])
    assert_runs(value.conceded, [0, 4, 0, 0])
    assert_runs(value.extras_cost, [0, 0, 0, 0])
    assert not value.credit.any()
    # 0 + 0.615 = 0.615;  -4 + 0.8525 = -3.1475;  the leg-bye and the byes 0 + 0.615 = 0.615 (charging their team
    # runs would give -1 + 0.615 = -0.385 and -4 + 0.615 = -3.385)
    assert_runs(value.nbv, [0.615, -3.1475, 0.615, 0.615])


def test_bowler_credit_only_for_codes_1_2_3_5_and_6(d14: D14Codes) -> None:
    # Criterion 8: B = 1 for bowled, caught, LBW, stumped and hit wicket only, also off a Wide. Omega 15 everywhere.
    codes = [BOWLED, CAUGHT, LBW, RUN_OUT, STUMPED, HIT_WICKET, RETIRE_OUT, OBSTRUCTING]
    wide_stumping = {"cls": WIDE, "extras_type": "wide", **_out(STUMPED, STRIKER)}
    d = _deliveries(d14, [_out(code, STRIKER) for code in codes] + [wide_stumping])
    assert d["counted_wicket"].all()
    value = _bowler(d, [15.0] * 9)
    assert value.credit.tolist() == [True, True, True, False, True, True, False, False, True]
    # Legal_Ball with credit: -0 + 15 x 1 - 0 + 0.615 = 15.615; without: 0.615.
    # Stumped off a Wide: Extras_Cost 1 + 1.2475 = 2.2475, so -0 + 15 - 2.2475 + 0.615 = 13.3675.
    assert_runs(value.nbv, [15.615, 15.615, 15.615, 0.615, 15.615, 15.615, 0.615, 0.615, 13.3675])


# --------------------------------------------------- dismissed batters and Dismissal_Charges, criteria 9-12

def test_dismissed_batter_comes_from_dismissed_player_id_with_a_counted_striker_fallback(
    d14: D14Codes,
) -> None:
    # Criteria 9 and 10, with criterion 1's D.
    d = _deliveries(
        d14,
        [
            {"runs": 1, **_out(RUN_OUT, NON_STRIKER)},  # the non-striker run out: the striker is not out
            _out(RUN_OUT, STRIKER),  # the striker run out
            _out(CAUGHT, None),  # dismissed_player_id blank: the striker, as a fallback
            _out(RUN_OUT, 999),  # dismissed_player_id names neither batter: the striker, as a fallback
        ],
    )
    out = dismissed_batters(d, retire_out_penalty=False)
    assert out.striker.tolist() == [False, True, True, True]
    assert out.fallback.tolist() == [False, False, True, True]
    assert int(out.fallback.sum()) == 2  # the count that criterion 10 reports
    omega = [14.0, 15.0, 12.0, 10.0]
    # EV = 1.20 - 0.06 x Omega: 0.36, 0.30, 0.48, 0.60
    # NBV: 1 - 0 - 0.36 = 0.64;  0 - 15 - 0.30 = -15.30;  0 - 12 - 0.48 = -12.48;  0 - 10 - 0.60 = -10.60
    assert_runs(_batter(d, omega).nbv, [0.64, -15.30, -12.48, -10.60])
    # The run-out penalty falls only on the batter who is out: one Dismissal_Charge, -14, for the non-striker.
    charges = dismissal_charges(d, omega, retire_out_penalty=False)
    assert charges["player_id"].tolist() == [NON_STRIKER]
    assert_runs(charges["charge"], [-14.0])


def test_dismissal_charges_for_a_non_striker_run_out_and_a_striker_out_off_a_wide(
    d14: D14Codes,
) -> None:
    # Criterion 11: a Dismissal_Charge of -Omega, with no Expected_Value term.
    d = _deliveries(
        d14,
        [
            {"pressure_bat": False},  # row 0: an Ordinary_Ball
            {"runs": 1, "pressure_bat": True, **_out(RUN_OUT, NON_STRIKER)},  # row 1: a Pressure_Ball
            {"cls": WIDE, "extras_type": "wide", "non_striker_id": NEXT_BATTER, **_out(STUMPED, STRIKER)},  # row 2
        ],
    )
    omega = [15.0, 14.0, 11.0]
    charges = dismissal_charges(d, omega, retire_out_penalty=False)
    assert charges["row"].tolist() == [1, 2]
    assert charges["player_id"].tolist() == [NON_STRIKER, STRIKER]
    assert_runs(charges["charge"], [-14.0, -11.0])
    assert charges["non_striker"].tolist() == [True, False]
    # Pressure class: row 1 has its own batter-Role flag (true). The Wide has none (Requirement 13, criterion 11),
    # so it takes, by the module's Wide rule, the flag of the latest earlier Batter_Delivery of the innings: row 1.
    assert charges["pressure"].tolist() == [True, True]
    assert charges["flag_row"].tolist() == [1, 1]
    # No charge enters a Batter_Delivery's NBV: the striker of row 1 is not out (1 - 0.36 = 0.64), and the Wide has
    # no batter NBV and no D.
    value = _batter(d, omega)
    assert_runs(value.nbv, [-0.30, 0.64, np.nan])
    assert value.striker_out.tolist() == [False, False, False]


def test_retire_out_gives_no_omega_penalty_and_no_dismissal_charge(d14: D14Codes) -> None:
    # Criterion 12, under the Config default D14_retire_out_batter_penalty = False.
    d = _deliveries(
        d14,
        [
            _out(RETIRE_OUT, STRIKER),  # the striker retires out
            {"runs": 1, **_out(RETIRE_OUT, NON_STRIKER)},  # the non-striker retires out
            _out(RETIRE_OUT, None),  # dismissed_player_id blank
        ],
    )
    assert d["counted_wicket"].all() and d["retire_out"].all()  # Counted_Wickets for the match state
    omega = [15.0, 15.0, 15.0]
    value = _batter(d, omega)
    # D = 0 on every row: 0 - 0.30 = -0.30;  1 - 0.30 = 0.70;  0 - 0.30 = -0.30
    assert_runs(value.nbv, [-0.30, 0.70, -0.30])
    assert not value.striker_out.any()
    assert dismissal_charges(d, omega, retire_out_penalty=False).empty
    assert not dismissed_batters(d, retire_out_penalty=False).fallback.any()  # the blank id is not a fallback
    # The D14 Robustness_Case (Requirement 21, criterion 4) charges each one: 0 - 15 - 0.30 = -15.30 for the
    # striker and the blank id (the striker), and a charge of -15 for the non-striker.
    assert_runs(_batter(d, omega, retire_out_penalty=True).nbv, [-15.30, 0.70, -15.30])
    assert dismissal_charges(d, omega, retire_out_penalty=True)["player_id"].tolist() == [NON_STRIKER]


# ---------------------------------------------------------------------- Extras_Cost, criteria 13 and 14

def test_wide_extras_cost_is_team_runs_plus_a_replacement_ball_with_its_own_free_hit_flag(
    d14: D14Codes,
) -> None:
    # Criterion 13: Extras_Cost = the Wide's team runs + E(p_wv), p_wv keeping the Delivery's own Free_Hit flag.
    wide = {"cls": WIDE, "extras_type": "wide"}
    d = _deliveries(d14, [wide, {**wide, "team_runs": 5}, {**wide, "free_hit": True}])  # 1, 5 and 1 team runs
    value = _bowler(d, [15.0] * 3, p_wv=[P_WV, P_WV, P_WV_FH])
    # 1 + 1.2475 = 2.2475;  5 + 1.2475 = 6.2475;  1 + 1.58 = 2.58 (E(p_wv_nofh) would give 1 + 1.0475 = 2.0475)
    assert_runs(value.extras_cost, [2.2475, 6.2475, 2.58])
    assert_runs(value.conceded, [0, 0, 0])
    assert_runs(value.ev, [EV_BOWL_15, EV_BOWL_15, EV_BOWL_FH_15])
    # NBV = -Extras_Cost - EV_bowl: -2.2475 + 0.615 = -1.6325;  -6.2475 + 0.615 = -5.6325;
    # -2.58 + 1.086675 = -1.493325
    assert_runs(value.nbv, [-1.6325, -5.6325, -1.493325])


def test_no_ball_extras_cost_adds_penalty_and_bat_runs_a_cleared_replacement_ball_and_the_premium(
    d14: D14Codes,
) -> None:
    # Criterion 14: Extras_Cost = penalty runs + bat runs + E(p_wv_nofh) + Free_Hit_Premium (0.8 in CELL_TABLE).
    no_ball = {"cls": NO_BALL, "extras_type": "Overstep"}
    d = _deliveries(
        d14,
        [
            {**no_ball, "runs": 4},  # team runs 5: the 1-run penalty and 4 bat runs
            {**no_ball, "team_runs": 2},  # team runs 2: penalty runs 2 (1 + 1 extras_2_run), 0 bat runs
            {**no_ball, "runs": 1, "free_hit": True},  # a No_Ball on a Free_Hit: p_wv has the flag set
        ],
    )
    value = _bowler(d, [15.0] * 3, p_wv=[P_WV, P_WV, P_WV_FH])
    # 1 + 4 + 1.0475 + 0.8 = 6.8475;  2 + 0 + 1.0475 + 0.8 = 3.8475;  1 + 1 + 1.0475 + 0.8 = 3.8475 (the replacement
    # ball has the flag cleared even on a Free_Hit; E(p_wv) would give 1 + 1 + 1.58 + 0.8 = 4.38)
    assert_runs(value.extras_cost, [6.8475, 3.8475, 3.8475])
    assert_runs(value.conceded, [0, 0, 0])  # the bat runs reach the bowler only through Extras_Cost (criterion 7)
    assert_runs(value.ev, [EV_BOWL_15, EV_BOWL_15, EV_BOWL_FH_15])
    # NBV: -6.8475 + 0.615 = -6.2325;  -3.8475 + 0.615 = -3.2325;  -3.8475 + 1.086675 = -2.760825
    assert_runs(value.nbv, [-6.2325, -3.2325, -2.760825])


# ------------------------------------------------------------- pooled fallback, criteria 20 and 21

# A tiny Delivery frame of two folds: fold, cell, class, bat runs, team runs, Free_Hit. Innings = 1 + cell // 3.
# Fold 2 is the Training_Folds of the fold-1 Deliveries, whose cell tables therefore come from fold 2 alone. Fold 1
# holds every kind, so the pooled sets of target fold 2 are not empty either.
_CELL_FRAME_ROWS: tuple[tuple[int, int, int, int, int, bool], ...] = (
    # fold 2, cell 0 (innings 1, powerplay): every kind
    (2, 0, WIDE, 0, 1, False), (2, 0, WIDE, 0, 1, False),
    (2, 0, NO_BALL, 0, 1, False), (2, 0, NO_BALL, 1, 2, False),
    (2, 0, LEGAL, 4, 4, True), (2, 0, LEGAL, 0, 0, True),
    (2, 0, LEGAL, 0, 0, False), (2, 0, LEGAL, 1, 1, False),
    # fold 2, cell 1 (innings 1, middle): no Wide
    (2, 1, NO_BALL, 2, 3, False), (2, 1, LEGAL, 6, 6, True), (2, 1, LEGAL, 1, 1, False),
    # fold 2, cell 2 (innings 1, death): no No_Ball
    (2, 2, WIDE, 0, 5, False), (2, 2, LEGAL, 2, 2, True), (2, 2, LEGAL, 2, 2, False),
    # fold 2, cell 3 (innings 2, powerplay): no Free_Hit Legal_Ball; the No_Ball has penalty runs 2 and 4 bat runs
    (2, 3, WIDE, 0, 1, False), (2, 3, NO_BALL, 4, 6, False), (2, 3, LEGAL, 3, 3, False),
    # fold 2, cells 4 and 5: no Delivery
    # fold 1: the Bowler_Deliveries valued in test_bowler_nbv_uses_the_pooled_values_of_empty_training_fold_cells
    (1, 0, LEGAL, 1, 1, False),
    (1, 1, WIDE, 0, 1, False),
    (1, 2, NO_BALL, 0, 1, False),
    (1, 3, NO_BALL, 1, 2, False),
    (1, 3, LEGAL, 4, 4, True),  # the Free_Hit after that No_Ball
)  # fmt: skip


def _cell_frame() -> pd.DataFrame:
    """_CELL_FRAME_ROWS as a Delivery frame: one match per fold, every row passing the training filters."""
    fold, cell, cls, runs, team_runs, free_hit = (np.array(values) for values in zip(*_CELL_FRAME_ROWS))
    n = len(_CELL_FRAME_ROWS)
    return pd.DataFrame(
        {
            "row": np.arange(n, dtype=np.int64),
            "match_id": fold.astype(np.int64),
            "innings": (1 + cell // 3).astype(np.int64),
            "fold": fold.astype(np.int64),
            "cell": cell.astype(np.int64),
            "cls": cls.astype(np.int64),
            "runs": runs.astype(np.int64),
            "team_runs": team_runs.astype(np.int64),
            "free_hit": free_hit.astype(bool),
            "is_super_over": np.zeros(n, dtype=bool),
            "excluded": np.zeros(n, dtype=bool),
            "bowler_wicket": np.zeros(n, dtype=bool),
        }
    )


def test_cell_tables_pool_one_sum_over_one_count_for_empty_training_fold_cells() -> None:
    # Criteria 6, 20 and 21, for target fold 1 (from fold 2's Deliveries), cells 0-5.
    frame = _cell_frame()
    means = extras_cell_means(frame, "cell")
    premium = free_hit_premium(frame, "cell")
    means = means[means["fold"] == 1].reset_index(drop=True)
    premium = premium[premium["fold"] == 1].reset_index(drop=True)
    assert means["cell"].tolist() == premium["cell"].tolist() == [0, 1, 2, 3, 4, 5]

    # Wide team runs: cell 0 (1 + 1) / 2 = 1.0, cell 2 5 / 1 = 5.0, cell 3 1 / 1 = 1.0. Cells 1, 4 and 5 hold no
    # Wide and take the pooled (1 + 1 + 5 + 1) / 4 = 2.0; the average of the cell means would be 7 / 3.
    assert means["n_wide"].tolist() == [2, 0, 1, 1, 0, 0]
    assert_runs(means["wide_runs"], [1.0, 2.0, 5.0, 1.0, 2.0, 2.0])
    assert means["pooled_wide_runs"].tolist() == [False, True, False, False, True, True]

    # No_Ball penalty runs (team - bat) and bat runs: cell 0 (1 + 1) / 2 = 1.0 and (0 + 1) / 2 = 0.5; cell 1 1.0
    # and 2.0; cell 3 2.0 and 4.0. Cells 2, 4 and 5 hold no No_Ball and take the pooled (1 + 1 + 1 + 2) / 4 = 1.25
    # and (0 + 1 + 2 + 4) / 4 = 1.75; the averages of the cell means would be 4 / 3 and 6.5 / 3.
    assert means["n_nb"].tolist() == [2, 1, 0, 1, 0, 0]
    assert_runs(means["nb_penalty_runs"], [1.0, 1.0, 1.25, 2.0, 1.25, 1.25])
    assert_runs(means["nb_bat_runs"], [0.5, 2.0, 1.75, 4.0, 1.75, 1.75])
    assert means["pooled_nb_penalty_runs"].tolist() == [False, False, True, False, True, True]
    assert means["pooled_nb_bat_runs"].tolist() == [False, False, True, False, True, True]

    # Free_Hit_Premium = mean bat runs per Free_Hit Legal_Ball - mean per non-Free_Hit Legal_Ball:
    # cell 0 (4 + 0) / 2 - (0 + 1) / 2 = 1.5; cell 1 6 - 1 = 5.0; cell 2 2 - 2 = 0.0. Cells 3, 4 and 5 hold no
    # Free_Hit Legal_Ball and take the pooled (4 + 0 + 6 + 2) / 4 - (0 + 1 + 1 + 2 + 3) / 5 = 3.0 - 1.4 = 1.6; the
    # average of the cell premiums would be 6.5 / 3.
    assert premium["n_fh_legal"].tolist() == [2, 1, 1, 0, 0, 0]
    assert premium["n_non_fh_legal"].tolist() == [2, 1, 1, 1, 0, 0]
    assert_runs(premium["free_hit_premium"], [1.5, 5.0, 0.0, 1.6, 1.6, 1.6])
    assert premium["pooled_free_hit_premium"].tolist() == [False, False, False, True, True, True]


def test_bowler_nbv_uses_the_pooled_values_of_empty_training_fold_cells() -> None:
    # Criteria 5, 6, 13, 14, 20 and 21 on the fold-1 Deliveries of _cell_frame, valued with the cell tables built
    # from the frame (values worked out in the previous test). Omega 15, p_W 0.04 and p_N 0.01 on each.
    frame = _cell_frame()
    means = extras_cell_means(frame, "cell")
    premium = free_hit_premium(frame, "cell")
    d = frame[frame["fold"] == 1].reset_index(drop=True)
    # Cell 0 Legal_Ball (1 bat run), cell 1 Wide, cell 2 No_Ball (0 bat runs), cell 3 No_Ball (1 bat run), and the
    # cell 3 Free_Hit Legal_Ball (4 bat runs), whose bowler view has the Free_Hit flag set.
    value = bowler_nbv(
        d, np.array([P_WV, P_WV, P_WV, P_WV, P_WV_FH]), _each(P_WV_NOFH, 5), _each(P_EXTRAS, 5), CRV, np.full(5, 15.0),
        means, premium,
    )  # fmt: skip
    # E[Wide cost] = wide_runs + E(p_wv);  E[NB cost] = nb_penalty_runs + nb_bat_runs + E(p_wv_nofh) + premium:
    #   cell 0:          1.0 + 1.2475 = 2.2475;           1.0 + 0.5 + 1.0475 + 1.5 = 4.0475
    #   cell 1:          2.0 (pooled) + 1.2475 = 3.2475;  1.0 + 2.0 + 1.0475 + 5.0 = 9.0475
    #   cell 2:          5.0 + 1.2475 = 6.2475;           1.25 + 1.75 (pooled) + 1.0475 + 0.0 = 4.0475
    #   cell 3:          1.0 + 1.2475 = 2.2475;           2.0 + 4.0 + 1.0475 + 1.6 (pooled) = 8.6475
    #   cell 3, P_WV_FH: 1.0 + 1.58 = 2.58;               8.6475
    assert_runs(value.exp_wide_cost, [2.2475, 3.2475, 6.2475, 2.2475, 2.58])
    assert_runs(value.exp_nb_cost, [4.0475, 9.0475, 4.0475, 8.6475, 8.6475])
    # EV_bowl = 0.95 x (15 x p_wv[W] - E(p_wv)) - 0.04 x E[Wide cost] - 0.01 x E[NB cost], where
    # 0.95 x (15 x 0.05 - 1.2475) = -0.472625 with P_WV and 0.95 x (15 x 0.04 - 1.58) = -0.931 with P_WV_FH:
    #   -0.472625 - 0.0899 - 0.040475 = -0.603;  -0.472625 - 0.1299 - 0.090475 = -0.693;
    #   -0.472625 - 0.2499 - 0.040475 = -0.763;  -0.472625 - 0.0899 - 0.086475 = -0.649;
    #   -0.931 - 0.1032 - 0.086475 = -1.120675
    assert_runs(value.ev, [-0.603, -0.693, -0.763, -0.649, -1.120675])
    # Extras_Cost: 0;  Wide 1 + 1.2475 = 2.2475;  No_Ball 1 + 0 + 1.0475 + 0.0 = 2.0475;
    # No_Ball 1 + 1 + 1.0475 + 1.6 (the pooled premium, criterion 21) = 4.6475;  0
    assert_runs(value.extras_cost, [0.0, 2.2475, 2.0475, 4.6475, 0.0])
    # NBV = -conceded - Extras_Cost - EV_bowl:
    #   -1 + 0.603 = -0.397;  -2.2475 + 0.693 = -1.5545;  -2.0475 + 0.763 = -1.2845;  -4.6475 + 0.649 = -3.9985;
    #   -4 + 1.120675 = -2.879325
    assert_runs(value.nbv, [-0.397, -1.5545, -1.2845, -3.9985, -2.879325])
