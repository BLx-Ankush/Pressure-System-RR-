"""Value_Engine: Wicket_Cost Omega, the Training_Folds cell tables, the Net Ball Value (NBV) of both
Roles, the Dismissal_Charges and the ``value`` Pipeline_Stage (Requirement 9, criteria 8-10 and 13;
Requirement 15; Requirement 2, criteria 4 and 12; design sections "Remaining_Runs_Model M2
(``models/remaining_runs.py``)", paragraph "Omega", and "Value_Engine (``value.py``)").

The module holds Wicket_Cost, which the ``remaining_runs`` stage computes from the predictions of
the outer capped M2 fits and saves in its ``m2_preds`` table; the per-fold tables of extras means
and Free_Hit_Premium that the bowler NBV reads; the batter and bowler NBV formulas and the
Dismissal_Charges; and ``run_stage``, the ``value`` Pipeline_Stage. No function reads
``config.py``. The one Config setting the formulas use, D14's
``design_choices.D14_retire_out_batter_penalty``, is the required keyword ``retire_out_penalty``
(False by default in the Config, Requirement 15, criterion 12; True is the Robustness_Case of
Requirement 21, criterion 4); the D14 bowler codes reach the formulas through the ``balls`` flag
``bowler_wicket``, which the ``reconstruct`` stage sets from the Config.

Wicket_Cost (``wicket_cost``, ``terminal_post_wicket``)
    For a Delivery in innings 1 or 2, r_pre is M2's expected remaining runs for the pre-ball state
    and r_post its expected remaining runs for the Post_Wicket_State (glossary), both from the M2
    fit that gives the Delivery's Out_Of_Fold_Prediction (criterion 8): the outer capped fit of the
    Delivery's fold. Then:

    1. r_post is taken as 0 when the Post_Wicket_State has no innings left
       (``terminal_post_wicket``): TERMINAL_WICKETS (9) or more wickets fallen before the
       Delivery, or an empty Batters_To_Come (criterion 13). Either way a dismissal ends the
       innings, because no batter is left to come in. A Batters_To_Come that is undefined (the
       batting side has no ``is_played`` row in ``playing_eleven.csv``, Requirement 6, criterion
       4) is not empty, so it keeps its M2 prediction; the delivered data has no such Delivery.
    2. Omega = r_pre - r_post.
    3. A negative Omega is set to 0 (criterion 10): ``clipped_low``.
    4. In innings 2, an Omega above the runs needed before the Delivery is set to the runs needed
       (criterion 9): ``clipped_high``. Runs needed of 0 or less (the chase was already won before
       the ball, 249 Deliveries as delivered, kept in the data) cap Omega at 0, so a positive
       Omega there is clipped high. An innings-2 Delivery whose runs needed is missing (no Target)
       has no upper clip.

    Omega therefore lies in [0, max(runs needed, 0)] in innings 2 and is 0 or more in innings 1.
    A Delivery is never clipped both ways: a negative Omega is below every cap.

Cell means, Free_Hit_Premium and the pooled fallback (``extras_cell_means``, ``free_hit_premium``)
    The expected Wide and No_Ball Extras_Costs of a Bowler_Delivery (Requirement 15, criterion 6)
    and the actual Extras_Cost of a No_Ball (criterion 14) use four quantities of the Delivery's
    Training_Folds in its innings x Phase cell. Each table has one row per target fold k and cell
    c of CELLS (the six codes 3 x (innings - 1) + Phase of ``state.cell_codes``). The target folds
    are the folds among the rows given, and row (k, c) holds the values for the fold-k Deliveries
    of cell c. The values come only from the rows of the other folds that pass the model-training
    filters (``common.training_mask``: innings 1 or 2, no Super_Over_Innings, no excluded match),
    so nothing a fold-k Delivery uses depends on a fold-k row (criterion 15):

    ==================== ================================================ ==================
    Column               Mean, over the cell's Deliveries of the other    Requirement 15
                         folds, of
    ==================== ================================================ ==================
    ``wide_runs``        team runs per Wide                               criteria 6 and 13
    ``nb_penalty_runs``  penalty runs (team runs - bat runs) per No_Ball  criteria 6 and 14
    ``nb_bat_runs``      bat runs (``runs``) per No_Ball                  criteria 6 and 14
    ``free_hit_premium`` bat runs per Free_Hit Legal_Ball, minus bat runs criteria 14 and 15
                         per non-Free_Hit Legal_Ball
    ==================== ================================================ ==================

    Team runs are the ``team_runs`` column of ``balls``: the team runs under the selected
    Extras_Encoding, which the ``reconstruct`` stage records as the data-derived Method_Constant
    ``reconstruct.selected_encoding``.

    No_Ball penalty runs. Criterion 14 charges a No_Ball's penalty runs under the selected
    Extras_Encoding and its bat runs as two terms. With b the bat runs and e the ``extras_2_runs``
    of a No_Ball, every encoding of ``reconstruct.ENCODINGS`` scores it as the one-run penalty plus
    b, plus e except under byes_one_nb_plain: 1 + b + e under F30, byes_one and byes_one_penalty5,
    and 1 + b under byes_one_nb_plain, which takes e as already inside ``runs``. The penalty runs
    are therefore the No_Ball's team runs minus its bat runs: 1 + e under byes_one, the encoding
    selected as delivered (where e, the runs a No_Ball adds beyond the penalty and the bat runs,
    is 0, 1 or 4), and 1 under byes_one_nb_plain. Penalty runs plus bat runs equal the No_Ball's
    team runs, so the bowler is charged every run the No_Ball adds to the score, as criterion 13
    charges a Wide's team runs (1 + e under every encoding; e is 0 on every Wide as delivered, so
    ``wide_runs`` is 1.0 there).

    Pooled fallback (criteria 20 and 21). Where the cell lacks the Deliveries a quantity needs, the
    quantity is computed instead over the Deliveries of all six cells of the same folds, taken as
    one pooled set: one sum divided by one count, never an average of cell means. The triggers are
    no Wide in the cell (``wide_runs``), no No_Ball (``nb_penalty_runs`` and ``nb_bat_runs``), and
    no Free_Hit Legal_Ball or no non-Free_Hit Legal_Ball (``free_hit_premium``, whose two means are
    then both pooled). The boolean flag ``pooled_<column>`` of each quantity marks the rows that
    use its pooled value. The counts ``n_wide``, ``n_nb``, ``n_fh_legal`` and ``n_non_fh_legal``
    are always the cell's own, so a 0 shows which Deliveries were missing, and the pooled set of a
    fold is the sum of its six counts. A pooled set that is itself empty (no Wide, no No_Ball, no
    Free_Hit Legal_Ball or no non-Free_Hit Legal_Ball in all six cells of the other folds) would
    leave the NBV undefined, so the Run stops with a StageError (exit code 5) that names each
    quantity and target fold.

    Cells. The ``cells`` argument names the column that holds the innings x Phase cell code under
    the D8 Phase splits in use (``cell`` of ``state``), so the D8 robustness case can pass the
    cells of its alternative splits. The tables call the column ``cell`` whatever ``cells`` is.

Fallback counts (``fallback_counts``, ``cell_lookup``, ``write_fallback_counts``; criterion 22)
    Every Bowler_Delivery's expected Wide and No_Ball Extras_Costs use all four quantities of its
    (fold, cell) row, which ``cell_lookup`` finds. ``fallback_counts`` gives, for each quantity of
    POOLED_QUANTITIES, the number of Bowler_Deliveries whose row has the quantity's pooled flag (0
    when none), and ``write_fallback_counts`` writes the four counts to FALLBACK_COUNTS_FILE in
    ``outputs/value/``. The ``value`` stage logs a warning when a count is above 0.

Net Ball Value (``expected_bat_runs``, ``batter_nbv``, ``bowler_nbv``; criteria 1-8, 13 and 14)
    Notation: p is a row of M4 class probabilities in ``outcome.CLASSES`` order, v the
    Class_Run_Values of the Delivery's own fold (the Legal_Balls of its Training_Folds, ``crv``),
    E(p) = sum over classes of p x v (``expected_bat_runs``), W the Bowler_Wicket class
    (W_CLASS) and R the Run_Out class (R_CLASS). Omega is the Delivery's Wicket_Cost (the final
    ``omega`` of ``m2_preds``, used as given). The four M4 variants are the Primary_Model's
    Out_Of_Fold predictions with the Role player's theta at the fold's Average_Skill and the
    opponent's actual theta kept (Requirement 2, criterion 4): ``batter_view`` p_bv and
    ``batter_view_fh`` p_bv_fh (the striker average; p_bv_fh with the Free_Hit flag set), and
    ``bowler_view`` p_wv and ``bowler_view_nofh`` p_wv_nofh (the bowler average; p_wv_nofh with the
    flag cleared). The M4 feature ``free_hit`` is the Delivery's own flag, so p_bv and p_wv keep it.
    No other opposition adjustment enters (Requirement 2, criterion 12).

    Batter NBV, on each Batter_Delivery (a Legal_Ball or No_Ball; a Wide has none, NaN):
        NBV = bat runs - Omega x D - EV_bat, with EV_bat = E(p) - Omega x (p[W] + p[R]),
        where p is p_bv on a Legal_Ball and p_bv_fh on a No_Ball (criteria 1-3), and D is 1 when
        the Delivery holds a charged Counted_Wicket (``dismissed_batters``) whose dismissed batter
        is the striker.

    Bowler NBV, on every Bowler_Delivery:
        NBV = -conceded + Omega x B - Extras_Cost - EV_bowl (criterion 4), where conceded is the
        bat runs of a Legal_Ball and 0 on a Wide or No_Ball, so byes, leg-byes and penalty runs
        of a Legal_Ball are never charged (criterion 7); B is the ``balls`` flag
        ``bowler_wicket`` (a Counted_Wicket with D14 code 1, 2, 3, 5 or 6, whatever the Delivery
        class: a stumping off a Wide credits the bowler; criterion 8); and
        Extras_Cost = 0 on a Legal_Ball,
                    = Wide team runs + E(p_wv) on a Wide (criterion 13),
                    = No_Ball team runs + E(p_wv_nofh) + Free_Hit_Premium on a No_Ball (criterion
                      14; the team runs are its penalty runs plus its bat runs, as above).
        EV_bowl = (1 - pW - pN) x (Omega x p_wv[W] - E(p_wv)) - pW x E[Wide cost] - pN x E[NB cost]
        (criterion 5), with pW and pN the Extras_Model Wide and No_Ball probabilities and the
        expected costs of criterion 6: E[Wide cost] = ``wide_runs`` + E(p_wv) and
        E[NB cost] = ``nb_penalty_runs`` + ``nb_bat_runs`` + E(p_wv_nofh) + ``free_hit_premium``,
        from the Delivery's (fold, cell) row of the cell tables, pooled where flagged.

Dismissed batters and Dismissal_Charges (``dismissed_batters``, ``dismissal_charges``; 9-12)
    The dismissed batter of a Counted_Wicket is the ``balls`` flag ``dismissed_is_striker``
    (``reconstruct.dismissed_batter_flags``): the striker or non-striker whose id equals
    ``dismissed_player_id`` (criterion 9), else the striker, flagged ``dismissed_fallback``
    (criterion 10; the ``value`` stage reports the count). A Retire Out (code 9) charges no
    batter: no Omega x D and no Dismissal_Charge (criterion 12; D14). With
    ``retire_out_penalty=True`` (the D14 Robustness_Case) a Retire Out is charged as any other
    batter dismissal, and an unresolved one falls back to the striker and is counted.

    A charged dismissal of a batter on a Delivery that is not that batter's Batter_Delivery (the
    non-striker out on any Delivery, or the striker out off a Wide) gives that batter a
    Dismissal_Charge of -Omega, with no Expected_Value term (criterion 11). It joins the batter's
    summed batter-Role NBV only: no Batter_Delivery, Pressure_Ball or weight is added (the
    ``rating`` stage adds ``charge`` to the NBV sums and nothing to the counts or weights). The
    charge is pressure or ordinary by the Delivery's batter-Role Pressure_Ball flag
    (``pressure_bat``).

    Wide rule. A Wide has no batter-Role flag (``pressure_bat`` is <NA>: Wides are not
    Batter_Deliveries and get no batter PI). A charge on a Wide takes the batter-Role flag of the
    latest earlier Batter_Delivery of the same innings (``flag_row`` names it), and is ordinary
    when the Wide is a Dead_Ball (``live`` false: a Dead_Ball is never a Pressure_Ball,
    Requirement 13, criterion 3) or the innings has no earlier Batter_Delivery. Reasons: the
    batter flag describes the match situation, and the previous Batter_Delivery is the nearest
    situation that has one. On the delivered data, consecutive Batter_Deliveries of an innings
    carry the same flag 87% of the time, and 70.5% of batter Pressure_Balls follow a batter
    Pressure_Ball. The bowler-Role flag of the same Wide is no guide: on Batter_Deliveries it is
    set on 23.4% of batter Pressure_Balls, about its base rate, because the two Roles' Stress
    pillars differ. An always-ordinary rule would ignore the situation that criterion 11 uses.

The own-theta check (``load_value_inputs``, ``own_theta_nbv``, ``own_theta_cell_means``; criterion 17)
    The NBV measures each player against an Average_Skill player, so its mean over an innings x
    Phase cell is not 0 where the cell's players are not average. As delivered, the innings-2
    death cell is batted mostly by lower-order players, and its batter NBV averages -0.065 runs
    per Batter_Delivery. Criterion 17 therefore checks calibration: ``own_theta_nbv`` recomputes
    both NBVs with each player's own theta, reading the Primary_Model's ``actual`` variant in
    place of ``batter_view`` and ``bowler_view`` (OWN_THETA_VIEWS). Omega, D and B, the
    Dismissal_Charges, the Extras_Cost terms, the Extras_Model probabilities and the cell tables
    stay as the stage has them. With outcomes drawn from the probabilities that the Expected_Value
    uses, NBV has zero expectation (design, Property 42), so a calibrated M4 gives own-theta cell
    means near 0.

    Approximation: no ``actual`` variant sets or clears the Free_Hit flag, so the terms that read
    ``batter_view_fh`` (the batter Expected_Value of a No_Ball, criterion 3) or
    ``bowler_view_nofh`` (the replacement ball of a No_Ball's Extras_Cost and of the expected
    No_Ball cost, criteria 6 and 14) keep those views, with the Role player's theta at
    Average_Skill. The flag matters far more than theta there: on the innings-1 death No_Balls as
    delivered, setting it raises the expected bat runs by 0.47 runs, while the own theta moves
    them by about 0.01. Reading ``actual`` with each Delivery's own flag instead would raise the
    batter cell means by 0.001 to 0.004 runs per Batter_Delivery. No_Balls are about 0.4% of
    Deliveries, and the expected No_Ball cost enters EV_bowl with weight pN.

    ``load_value_inputs`` builds the inputs exactly as the stage does (the stage calls it), so
    ``own_theta_nbv`` with NBV_VIEWS, which gives each view its own variant, returns the stage's
    NBV and Dismissal_Charges. ``own_theta_cell_means`` returns the ``nbv_cell_means`` table of
    the own-theta NBV.

Errors: ValueError or TypeError, naming the problem, for arrays of different lengths, a value of
the wrong type, an innings other than 1 or 2, or a missing or non-finite prediction where one is
needed (r_pre on every row, r_post on every row whose Post_Wicket_State is not terminal). The cell
tables raise ValueError or TypeError for a missing column, a missing or malformed value on a row
they use, or a cell code that is not a cell of the row's innings, and StageError for an empty
pooled set. The NBV functions raise ValueError or TypeError for a missing column, an array of the
wrong shape, and a missing, negative or non-finite Omega, probability or Class_Run_Value on a
Delivery that uses it.

The ``value`` Pipeline_Stage (``run_stage``)
    Reads ``balls`` (``reconstruct``), ``state``, the Fold_Table (``folds``), ``m2_preds``
    (``remaining_runs``: Omega), ``m4_class_run_values`` and ``extras_preds`` (``outcome``), the
    Primary M4 variants (``model_choice.load_primary``: the four views and ``actual``), the
    ``pressure`` table (``live`` and ``pressure_bat``) and, for the names in the lists,
    ``ball_by_ball`` and ``matches`` (``load``). Then:

    1. builds the cell tables from every Delivery (``balls`` with its fold and ``state`` cell);
    2. takes the Deliveries of the matches used for player rating: innings 1 and 2 of every match
       not excluded under Requirement 5, criterion 24 (``common.training_mask``), in ``state``
       order: every one is a Bowler_Delivery, and every Legal_Ball and No_Ball a Batter_Delivery.
       Steps 1 and 2 are ``load_value_inputs``;
    3. computes both NBVs and the Dismissal_Charges, and stops the Run with a StageError (exit
       code 5) unless every Batter_Delivery and Bowler_Delivery has a finite NBV and every charge
       is finite (criterion 16); then the own-theta cell means (``own_theta_cell_means``); then,
       when the Config names an M4 preferred family that displaced the lowest-log-loss candidate,
       re-verifies the calibration-aware tie-break on its own inputs (``m4_tiebreak_cells``;
       Requirement 23, criterion 14): the own-theta NBV cell means of the displaced family are
       recomputed on the same Omega, Extras and Free_Hit_Premium (only the M4 variants differ),
       and the Run stops with a StageError unless the selected family is within the log-loss band
       and passes every innings x Phase cell while the displaced family breaches at least one;
    4. saves VALUE_TABLE (VALUE_COLUMNS), DISMISSAL_CHARGES_TABLE (DISMISSAL_CHARGE_COLUMNS),
       CELL_MEANS_TABLE (the two cell tables merged on ``fold`` and ``cell``, CELL_MEANS_COLUMNS),
       FALLBACK_COUNTS_TABLE, NBV_CELLS_TABLE (``nbv_cell_means``) and NBV_CELLS_OWN_THETA_TABLE
       (``own_theta_cell_means``, NBV_CELL_COLUMNS);
    5. writes to ``outputs/value/``: FALLBACK_COUNTS_FILE (criterion 22; a Run_Log warning when a
       count is above 0), CELL_MEANS_FILE, NBV_CELLS_FILE (the cell means against an
       Average_Skill player, for information and the report), NBV_CELLS_OWN_THETA_FILE (the
       own-theta cell means of criterion 17), M4_TIEBREAK_CELLS_FILE when the M4 tie-break applies
       (the two-candidate own-theta diagnostic of Requirement 23, criterion 14) and, per Role,
       EXTREMES_FILES: the EXTREMES_SIZE
       Deliveries with the highest NBV and the EXTREMES_SIZE with the lowest (ties: the earlier
       Delivery first), with the columns of criterion 19 (EXTREME_COLUMNS);
    6. records the data-derived constants STRIKER_FALLBACK_CONSTANT (criterion 10),
       POOLED_FALLBACK_CONSTANT (criterion 22), DISMISSAL_CHARGES_CONSTANT, RETIRE_OUT_CONSTANT
       and NBV_CELL_MEANS_CONSTANT, and logs the NBV summaries, the Average_Skill cell means, the
       own-theta cell means against the Test_Suite band ``tests.nbv_cell_tol`` (a warning outside
       it; criterion 17) and the counts.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import pandas as pd

from superr import folds, reconstruct
from superr.errors import StageError
from superr.folds import FOLD_COLUMN
from superr.models import common, extras, model_choice, outcome
from superr.reconstruct import DeliveryClass
from superr.state import STATE_INNINGS, STATE_TABLE, Phase
from superr.state import STAGE_NAME as STATE_STAGE

if TYPE_CHECKING:
    from superr.pipeline import StageContext
    from superr.store import CheckpointStore

__all__ = [
    "BATTER_NBV_COLUMNS",
    "BATTER_ROLE",
    "BOWLER_NBV_COLUMNS",
    "BOWLER_ROLE",
    "BatterValue",
    "BowlerValue",
    "CELLS",
    "CELL_COLUMN",
    "CELL_MEANS_COLUMNS",
    "CELL_MEANS_FILE",
    "CELL_MEANS_TABLE",
    "DISMISSAL_CHARGES_CONSTANT",
    "DISMISSAL_CHARGES_TABLE",
    "DISMISSAL_CHARGE_COLUMNS",
    "DISMISSAL_CHARGE_INPUT_COLUMNS",
    "DISMISSAL_COLUMNS",
    "Dismissals",
    "EXTRAS_CELL_MEANS_COLUMNS",
    "EXTREMES_FILES",
    "EXTREMES_SIZE",
    "EXTREME_COLUMNS",
    "FALLBACK_COUNTS_COLUMNS",
    "FALLBACK_COUNTS_FILE",
    "FALLBACK_COUNTS_TABLE",
    "FREE_HIT_PREMIUM_COLUMNS",
    "M4_TIEBREAK_CELLS_FILE",
    "M4_TIEBREAK_CELL_COLUMNS",
    "M4TieBreakCells",
    "NBV_CELLS_FILE",
    "NBV_CELLS_OWN_THETA_FILE",
    "NBV_CELLS_OWN_THETA_TABLE",
    "NBV_CELLS_TABLE",
    "NBV_CELL_COLUMNS",
    "NBV_CELL_MEANS_CONSTANT",
    "NBV_VIEWS",
    "N_CELLS",
    "OWN_THETA_VIEWS",
    "POOLED_FALLBACK_CONSTANT",
    "POOLED_QUANTITIES",
    "PooledQuantity",
    "RETIRE_OUT_CONSTANT",
    "ROLES",
    "R_CLASS",
    "STAGE_NAME",
    "STRIKER_FALLBACK_CONSTANT",
    "TERMINAL_WICKETS",
    "VALUE_COLUMNS",
    "VALUE_FOLDER",
    "VALUE_TABLE",
    "VALUE_VARIANTS",
    "ValueInputs",
    "WICKET_COST_INNINGS",
    "W_CLASS",
    "WicketCost",
    "batter_nbv",
    "bowler_nbv",
    "cell_lookup",
    "dismissal_charges",
    "dismissed_batters",
    "expected_bat_runs",
    "extras_cell_means",
    "fallback_counts",
    "free_hit_premium",
    "load_value_inputs",
    "m4_tiebreak_cells",
    "nbv_cell_means",
    "own_theta_cell_means",
    "own_theta_nbv",
    "run_stage",
    "terminal_post_wicket",
    "wicket_cost",
    "write_fallback_counts",
]

#: Wickets fallen at which the next dismissal ends the innings: 10 wickets end a T20 innings,
#: fixed by the Laws of Cricket rather than a Method_Constant (criterion 13).
TERMINAL_WICKETS = 9

#: The innings whose Deliveries get an Omega; innings 2 is the chase, whose Omega is capped.
WICKET_COST_INNINGS: tuple[int, int] = (1, 2)
_CHASE = 2


class WicketCost(NamedTuple):
    """The result of wicket_cost: one value per Delivery, in input order.

    * ``omega``: Wicket_Cost in runs (float64), 0 or more;
    * ``clipped_low``: True where r_pre - r_post was negative and Omega was set to 0 (criterion 10);
    * ``clipped_high``: True where an innings-2 Omega exceeded the runs needed and was set to them,
      or to 0 when the runs needed were 0 or less (criterion 9);
    * ``r_post_wicket``: the post-wicket expected remaining runs that Omega used (float64): the
      given r_post, or 0 where the Post_Wicket_State is terminal (criterion 13).
    """

    omega: np.ndarray
    clipped_low: np.ndarray
    clipped_high: np.ndarray
    r_post_wicket: np.ndarray


# ---------------------------------------------------------------------------------- argument checks

def _vector(values: Any, what: str) -> None:
    if values is None or isinstance(values, (str, bytes, Mapping, pd.DataFrame)) or np.ndim(values) != 1:
        raise TypeError(f"{what} must be a 1-D array or Series; got {type(values).__name__}")


def _length(array: np.ndarray, what: str, n_rows: int | None) -> np.ndarray:
    if n_rows is not None and len(array) != n_rows:
        raise ValueError(f"{what} has {len(array):,} values, but {n_rows:,} are needed, one per Delivery")
    return array


def _pandas(values: Any) -> bool:
    return isinstance(values, (pd.Series, pd.Index, pd.api.extensions.ExtensionArray))


def _floats(values: Any, what: str, n_rows: int | None = None) -> np.ndarray:
    """``values`` (1-D, numeric) as a new float64 array, NaN for a missing value."""
    _vector(values, what)
    if _pandas(values):
        dtype = values.dtype
        if pd.api.types.is_bool_dtype(dtype) or not pd.api.types.is_numeric_dtype(dtype):
            raise TypeError(f"{what} must hold numbers; got dtype {dtype}")
        array = pd.Series(values).to_numpy(dtype=np.float64, na_value=np.nan)
    else:
        array = np.asarray(values)
        if array.dtype.kind not in "iuf":
            raise TypeError(f"{what} must hold numbers; got dtype {array.dtype}")
        array = array.astype(np.float64)
    return _length(array, what, n_rows)


def _whole(values: Any, what: str, n_rows: int | None = None, *, missing: bool = False) -> np.ndarray:
    """``values`` (1-D whole numbers) as float64; NaN for a missing value, which only ``missing=True`` allows."""
    array = _floats(values, what, n_rows)
    absent = np.isnan(array)
    if absent.any() and not missing:
        raise ValueError(f"{what} has {int(absent.sum()):,} missing value(s)")
    present = array[~absent]
    if not (np.isfinite(present) & (np.floor(present) == present)).all():
        raise ValueError(f"{what} must hold whole numbers")
    return array


def _flags(values: Any, what: str, n_rows: int) -> np.ndarray:
    """``values`` (1-D booleans, none missing) as a bool array."""
    _vector(values, what)
    series = pd.Series(values) if _pandas(values) else pd.Series(np.asarray(values))
    if not pd.api.types.is_bool_dtype(series.dtype):
        raise TypeError(f"{what} must be boolean; got dtype {series.dtype}")
    if series.isna().any():
        raise ValueError(f"{what} has {int(series.isna().sum()):,} missing value(s)")
    return _length(series.to_numpy(dtype=bool), what, n_rows)


# ----------------------------------------------------------------------------------- Wicket_Cost

def terminal_post_wicket(wickets: Any, btc_count: Any) -> np.ndarray:
    """True for each Delivery whose Post_Wicket_State has no innings left (criterion 13; module docstring).

    ``wickets`` is the pre-ball wickets fallen of each Delivery (whole numbers, none missing) and
    ``btc_count`` the size of its Batters_To_Come (the ``state`` columns of the same names;
    missing where Batters_To_Come is undefined). Returns a bool array: True where wickets is
    TERMINAL_WICKETS or more, or ``btc_count`` is 0. A missing ``btc_count`` is not an empty
    Batters_To_Come. Raises ValueError or TypeError for malformed arguments.
    """
    down = _whole(wickets, "wickets")
    count = _whole(btc_count, "btc_count", len(down), missing=True)
    return (down >= TERMINAL_WICKETS) | (count == 0)  # NaN == 0 is False


def wicket_cost(r_pre: Any, r_post: Any, runs_needed: Any, innings: Any, nine_down_or_empty: Any) -> WicketCost:
    """Wicket_Cost Omega of each Delivery (criteria 8-10 and 13; module docstring, "Wicket_Cost").

    ``r_pre`` and ``r_post`` are the expected remaining runs of each Delivery's pre-ball state and
    Post_Wicket_State, from the capped M2 fit that gives its Out_Of_Fold_Prediction;
    ``runs_needed`` its runs needed before the ball (Target minus team score; read only in innings
    2, missing elsewhere); ``innings`` its innings (1 or 2); ``nine_down_or_empty`` the flag of
    terminal_post_wicket. All are 1-D with one value per Delivery, in any common order.

    Returns WicketCost(omega, clipped_low, clipped_high, r_post_wicket): r_post is taken as 0
    where the flag is set, Omega = r_pre - r_post is set to 0 where negative, and an innings-2
    Omega is capped at max(runs needed, 0) unless runs needed is missing.

    Raises ValueError when the arrays differ in length, an innings is not 1 or 2, ``r_pre`` is
    missing or not finite, or ``r_post`` is missing or not finite where the flag is not set;
    TypeError for a value of the wrong type.
    """
    pre = _floats(r_pre, "r_pre")
    n_rows = len(pre)
    post = _floats(r_post, "r_post", n_rows)
    needed = _floats(runs_needed, "runs_needed", n_rows)
    inns = _whole(innings, "innings", n_rows)
    terminal = _flags(nine_down_or_empty, "nine_down_or_empty", n_rows)
    outside = ~np.isin(inns, WICKET_COST_INNINGS)
    if outside.any():
        raise ValueError(f"innings must be 1 or 2 on every Delivery; {int(outside.sum()):,} value(s) are not")
    bad = ~np.isfinite(pre)
    if bad.any():
        raise ValueError(f"r_pre is missing or not finite on {int(bad.sum()):,} Delivery(ies)")
    bad = ~terminal & ~np.isfinite(post)
    if bad.any():
        raise ValueError(
            f"r_post is missing or not finite on {int(bad.sum()):,} Delivery(ies) whose Post_Wicket_State is not "
            "terminal (fewer than 9 wickets down and a non-empty Batters_To_Come)"
        )
    if np.isinf(needed).any():
        raise ValueError("runs_needed holds an infinite value; runs needed is a number of runs, or missing")

    post_used = np.where(terminal, 0.0, post)
    raw = pre - post_used
    capped = (inns == _CHASE) & ~np.isnan(needed)
    cap = np.where(capped, np.maximum(np.nan_to_num(needed, nan=0.0), 0.0), np.inf)
    clipped_low = raw < 0.0
    clipped_high = raw > cap
    omega = np.minimum(np.maximum(raw, 0.0), cap)
    return WicketCost(omega=omega, clipped_low=clipped_low, clipped_high=clipped_high, r_post_wicket=post_used)


# ------------------------------------------------- cell means, Free_Hit_Premium and the pooled fallback

#: Column of the cell tables that holds the innings x Phase cell code, whatever column the input names.
CELL_COLUMN = "cell"

#: The innings x Phase cells (D13) of innings 1 and 2, coded 3 x (innings - 1) + Phase as
#: ``state.cell_codes`` gives them: 0, 1 and 2 are the powerplay, middle and death of innings 1,
#: and 3, 4 and 5 those of innings 2.
N_CELLS = len(STATE_INNINGS) * len(Phase)
CELLS: tuple[int, ...] = tuple(range(N_CELLS))


class PooledQuantity(NamedTuple):
    """One of the four quantities with a pooled fallback (Requirement 15, criteria 20-22)."""

    label: str  # its ``quantity`` in fallback_counts.csv (design, Data Models)
    column: str  # its value column in the extras_cell_means or free_hit_premium table
    flag: str  # its boolean pooled flag column: "pooled_" + column
    name: str  # its name in messages


_WIDE = PooledQuantity("wide_team_runs", "wide_runs", "pooled_wide_runs", "mean Wide team runs")
_NB_PENALTY = PooledQuantity(
    "nb_penalty_runs", "nb_penalty_runs", "pooled_nb_penalty_runs", "mean No_Ball penalty runs"
)
_NB_BAT = PooledQuantity("nb_bat_runs", "nb_bat_runs", "pooled_nb_bat_runs", "mean No_Ball bat runs")
_PREMIUM = PooledQuantity("free_hit_premium", "free_hit_premium", "pooled_free_hit_premium", "Free_Hit_Premium")

#: The four quantities, in the order of fallback_counts and fallback_counts.csv (criterion 22).
POOLED_QUANTITIES: tuple[PooledQuantity, ...] = (_WIDE, _NB_PENALTY, _NB_BAT, _PREMIUM)

#: Columns of the extras_cell_means and free_hit_premium tables, in order: the key, the values,
#: the cell's own counts and the pooled flags (design, Data Models, ``cell_means``).
EXTRAS_CELL_MEANS_COLUMNS: tuple[str, ...] = (
    FOLD_COLUMN, CELL_COLUMN, _WIDE.column, _NB_PENALTY.column, _NB_BAT.column, "n_wide", "n_nb",
    _WIDE.flag, _NB_PENALTY.flag, _NB_BAT.flag,
)  # fmt: skip
FREE_HIT_PREMIUM_COLUMNS: tuple[str, ...] = (
    FOLD_COLUMN, CELL_COLUMN, _PREMIUM.column, "n_fh_legal", "n_non_fh_legal", _PREMIUM.flag,
)  # fmt: skip

#: Columns of the fallback_counts table and of its file (design, Data Models, ``fallback_counts``).
FALLBACK_COUNTS_COLUMNS: tuple[str, ...] = ("quantity", "n_bowler_deliveries")

#: Folder under the outputs folder that holds the Value_Engine's files, and the fallback-count file.
VALUE_FOLDER = "value"
FALLBACK_COUNTS_FILE = "fallback_counts.csv"

_TRAINING_FILTERS = "innings 1 or 2, no Super_Over_Innings, no excluded match"


class _CellRows(NamedTuple):
    """The rows of a cell-table input that pass the training filters, with their target fold and cell."""

    targets: np.ndarray  # int64: the target folds, ascending: every fold among the rows given
    at: np.ndarray  # int64: positions of the rows that pass the training filters
    target: np.ndarray  # int64: the index in ``targets`` of each such row's fold
    cell: np.ndarray  # int64: each such row's cell code, one of CELLS


def _require(frame: Any, columns: Sequence[str], what: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{what} must be a DataFrame; got {type(frame).__name__}")
    absent = [name for name in dict.fromkeys(columns) if name not in frame.columns]
    if absent:
        raise ValueError(f"{what} lacks the column(s) {', '.join(map(repr, absent))}")


def _cell_column(cells: Any, what: str) -> str:
    if not isinstance(cells, str) or not cells:
        raise TypeError(f"cells must name the column of {what} that holds the innings x Phase cell; got {cells!r}")
    return cells


def _cell_rows(
    frame: Any, what: str, cells: Any, columns: Sequence[str], mask: Callable[[pd.DataFrame], np.ndarray]
) -> _CellRows:
    """Check a cell-table input, and find its target folds and the rows that ``mask`` keeps."""
    _require(frame, (FOLD_COLUMN, _cell_column(cells, what), *columns), what)
    folds = _whole(frame[FOLD_COLUMN], f"{what} column {FOLD_COLUMN!r}")
    if (folds < 1).any():
        raise ValueError(f"{what} column {FOLD_COLUMN!r} holds a fold below 1; folds are numbered from 1")
    at = np.flatnonzero(mask(frame))  # raises for an absent or malformed innings, is_super_over, excluded or cls
    used = f"on the Deliveries that pass the training filters ({_TRAINING_FILTERS})"
    cell = _whole(frame[cells].iloc[at], f"{what} column {cells!r} {used}")
    outside = ~np.isin(cell, CELLS)
    if outside.any():
        raise ValueError(
            f"{what} column {cells!r} must hold innings x Phase cell codes 0 to {N_CELLS - 1} {used}; "
            f"{int(outside.sum()):,} value(s) do not"
        )
    innings = _whole(frame["innings"].iloc[at], f"{what} column 'innings'")
    elsewhere = np.asarray(STATE_INNINGS)[(cell // len(Phase)).astype(np.int64)] != innings
    if elsewhere.any():
        raise ValueError(
            f"{what} column {cells!r} holds {int(elsewhere.sum()):,} cell code(s) of the other innings {used}; "
            "a Delivery's cell is 3 x (innings - 1) + Phase"
        )
    targets = np.unique(folds).astype(np.int64)
    return _CellRows(
        targets=targets,
        at=at.astype(np.int64),
        target=np.searchsorted(targets, folds[at]).astype(np.int64),
        cell=cell.astype(np.int64),
    )


def _runs(frame: pd.DataFrame, column: str, at: np.ndarray, what: str) -> np.ndarray:
    """``column`` of ``frame`` at the positions ``at``: whole numbers of runs, 0 or more, none missing (float64)."""
    values = _whole(frame[column].iloc[at], what)
    if (values < 0).any():
        raise ValueError(f"{what} holds {int((values < 0).sum()):,} negative number(s) of runs")
    return values


def _other_folds(rows: _CellRows, select: np.ndarray, values: np.ndarray | None = None) -> np.ndarray:
    """Per (target, cell): the count of the ``select`` rows of the other folds, or the sum of their ``values``.

    ``select`` (bool) and ``values`` (float64 whole numbers) are aligned with ``rows.at``. Returns
    an array of shape (len(rows.targets), N_CELLS): int64 counts, or float64 sums, which are exact
    because they add whole numbers, so a target's entries never depend on its own fold's rows.
    """
    n_targets = len(rows.targets)
    key = rows.target[select] * N_CELLS + rows.cell[select]
    weights = None if values is None else values[select]
    per_fold = np.bincount(key, weights=weights, minlength=n_targets * N_CELLS).reshape(n_targets, N_CELLS)
    others = np.zeros_like(per_fold)
    for t in range(n_targets):
        others[t] = per_fold[np.arange(n_targets) != t].sum(axis=0)
    return others


def _pooled_means(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Each target's pooled mean: one sum over one count of its six cells (the caller has checked the count)."""
    return sums.sum(axis=1) / np.maximum(counts.sum(axis=1), 1)


def _cell_or_pooled(sums: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each cell's mean, sum / count, or its target's pooled mean where the cell has no row.

    Returns (means, pooled), both of the shape of ``sums``. The caller has checked that the pooled
    count of every target is above 0.
    """
    pooled = counts == 0
    return np.where(pooled, _pooled_means(sums, counts)[:, None], sums / np.maximum(counts, 1)), pooled


def _listing(values: Sequence[int]) -> str:
    shown = [str(value) for value in values]
    return shown[0] if len(shown) == 1 else f"{', '.join(shown[:-1])} and {shown[-1]}"


def _empty_pooled_set(targets: np.ndarray, t: int, quantities: Sequence[PooledQuantity], missing: str) -> str:
    """One line of the StageError for an empty pooled set: the target fold, its quantities and what is missing."""
    fold = int(targets[t])
    others = [int(other) for other in targets.tolist() if other != fold]
    names = " and ".join(f"the {quantity.name} ({quantity.label})" for quantity in quantities)
    if not others:
        return f"target fold {fold}, {names}: the rows given hold no fold other than {fold}, so the pooled set is empty"
    return (
        f"target fold {fold}, {names}: the Deliveries of {'fold' if len(others) == 1 else 'folds'} "
        f"{_listing(others)} that pass the training filters ({_TRAINING_FILTERS}) hold no {missing} in any of "
        "the six innings x Phase cells"
    )


def _stop_if_empty(problems: Sequence[str]) -> None:
    if problems:
        raise StageError(
            "The pooled fallback of Requirement 15, criteria 20 and 21, needs a Delivery that the Training_Folds do "
            "not hold, so the bowler NBV of the target fold's Deliveries would be undefined:\n"
            + "\n".join(f"  - {line}" for line in problems)
        )


def _cell_table(targets: np.ndarray, values: Mapping[str, np.ndarray], columns: Sequence[str]) -> pd.DataFrame:
    """One row per target fold and cell of CELLS, sorted by both, from (target, cell) arrays of ``values``."""
    data = {
        FOLD_COLUMN: np.repeat(targets, N_CELLS).astype(np.int32),
        CELL_COLUMN: np.tile(np.asarray(CELLS), len(targets)).astype(np.int8),
        **{name: array.ravel() for name, array in values.items()},
    }
    return pd.DataFrame({name: data[name] for name in columns})


def extras_cell_means(deliveries: pd.DataFrame, cells: str) -> pd.DataFrame:
    """The Wide and No_Ball means of every target fold and innings x Phase cell, with the pooled fallback.

    Requirement 15, criteria 6 and 20; module docstring, "Cell means, Free_Hit_Premium and the
    pooled fallback". ``deliveries`` is a per-Delivery frame, such as the ``balls`` table of the
    ``reconstruct`` checkpoint with the ``fold`` of ``folds.attach_folds`` and the ``cell`` of
    ``state``. It needs, on every row, ``fold`` and the columns of ``common.training_mask``
    (``innings``, ``is_super_over`` and ``excluded``); on the rows that pass the training filters,
    ``cls`` (the DeliveryClass code) and the cell code in the column named by ``cells``; on those
    rows' Wides and No_Balls, ``team_runs`` (team runs under the selected Extras_Encoding); and on
    their No_Balls, ``runs`` (bat runs). Other rows may hold missing values in these columns.

    Returns EXTRAS_CELL_MEANS_COLUMNS, one row per target fold (each fold among the rows) and cell
    of CELLS, sorted by fold and cell, on a RangeIndex: ``fold`` (int32) and ``cell`` (int8); the
    means ``wide_runs``, ``nb_penalty_runs`` (team runs minus bat runs) and ``nb_bat_runs``
    (float64, runs per Wide or No_Ball) over the cell's Deliveries of the other folds; the cell's
    own counts ``n_wide`` and ``n_nb`` (int64) of those Deliveries; and the flags
    ``pooled_wide_runs``, ``pooled_nb_penalty_runs`` and ``pooled_nb_bat_runs`` (bool), true where
    the cell holds no Wide (or no No_Ball) and the mean is one sum over one count of the other
    folds' Wides (or No_Balls) in all six cells.

    Raises StageError (exit code 5) when the other folds of a target fold hold no Wide, or no
    No_Ball, in any cell, so the pooled set is empty, naming each quantity and target fold.
    Raises ValueError or TypeError when ``deliveries`` is not a DataFrame or lacks a column, a
    needed value is missing or not a whole number, a fold is below 1, a cell code is not a cell
    of the row's innings, a class code is not a DeliveryClass, or a Wide scores below 1 run or a
    No_Ball below its bat runs + 1 (no Extras_Encoding scores them so).
    """
    rows = _cell_rows(deliveries, "deliveries", cells, ("cls", "team_runs", "runs"), common.training_mask)
    used = f"on the Deliveries that pass the training filters ({_TRAINING_FILTERS})"
    cls = _whole(deliveries["cls"].iloc[rows.at], f"deliveries column 'cls' {used}")
    unknown = ~np.isin(cls, [int(code) for code in DeliveryClass])
    if unknown.any():
        raise ValueError(
            f"deliveries column 'cls' holds {int(unknown.sum()):,} value(s) that are not a DeliveryClass {used}"
        )
    wide = cls == DeliveryClass.WIDE
    no_ball = cls == DeliveryClass.NO_BALL
    extra = wide | no_ball
    team = np.zeros(len(rows.at))
    team[extra] = _runs(
        deliveries, "team_runs", rows.at[extra], f"deliveries column 'team_runs' on the Wides and No_Balls {used}"
    )
    bat = np.zeros(len(rows.at))
    bat[no_ball] = _runs(deliveries, "runs", rows.at[no_ball], f"deliveries column 'runs' on the No_Balls {used}")
    penalty = team - bat
    if (team[wide] < 1).any():
        raise ValueError(
            f"{int((team[wide] < 1).sum()):,} Wide(s) {used} score below 1 team run; every Extras_Encoding scores a "
            "Wide as the one-run penalty plus extras_2_runs"
        )
    if (penalty[no_ball] < 1).any():
        raise ValueError(
            f"{int((penalty[no_ball] < 1).sum()):,} No_Ball(s) {used} score fewer team runs than their bat runs + 1; "
            "every Extras_Encoding scores a No_Ball as the one-run penalty plus its bat runs, plus extras_2_runs "
            "except under byes_one_nb_plain"
        )

    n_wide = _other_folds(rows, wide)
    n_nb = _other_folds(rows, no_ball)
    problems: list[str] = []
    for t in range(len(rows.targets)):
        if not n_wide[t].any():
            problems.append(_empty_pooled_set(rows.targets, t, (_WIDE,), "Wide"))
        if not n_nb[t].any():
            problems.append(_empty_pooled_set(rows.targets, t, (_NB_PENALTY, _NB_BAT), "No_Ball"))
    _stop_if_empty(problems)

    wide_runs, pooled_wide = _cell_or_pooled(_other_folds(rows, wide, team), n_wide)
    nb_penalty_runs, pooled_nb = _cell_or_pooled(_other_folds(rows, no_ball, penalty), n_nb)
    nb_bat_runs, _ = _cell_or_pooled(_other_folds(rows, no_ball, bat), n_nb)
    values = {
        _WIDE.column: wide_runs,
        _NB_PENALTY.column: nb_penalty_runs,
        _NB_BAT.column: nb_bat_runs,
        "n_wide": n_wide,
        "n_nb": n_nb,
        _WIDE.flag: pooled_wide,
        _NB_PENALTY.flag: pooled_nb,
        _NB_BAT.flag: pooled_nb.copy(),
    }
    return _cell_table(rows.targets, values, EXTRAS_CELL_MEANS_COLUMNS)


def free_hit_premium(legal: pd.DataFrame, cells: str) -> pd.DataFrame:
    """The Free_Hit_Premium of every target fold and innings x Phase cell, with the pooled fallback.

    Requirement 15, criteria 14, 15 and 21; module docstring, "Cell means, Free_Hit_Premium and
    the pooled fallback". ``legal`` is a per-Delivery frame like the ``deliveries`` of
    extras_cell_means; only its Legal_Balls that pass the training filters
    (``common.legal_training_mask``) enter the premium, so the whole Delivery frame may be passed.
    It needs, on every row, ``fold``, ``cls`` and the columns of ``common.training_mask``; and on
    those Legal_Balls, the cell code in the column named by ``cells``, ``free_hit`` (the Free_Hit
    flag) and ``runs`` (bat runs).

    Returns FREE_HIT_PREMIUM_COLUMNS, one row per target fold (each fold among the rows) and cell
    of CELLS, sorted by fold and cell, on a RangeIndex: ``fold`` (int32) and ``cell`` (int8);
    ``free_hit_premium`` (float64, runs), the mean bat runs per Free_Hit Legal_Ball minus the mean
    bat runs per non-Free_Hit Legal_Ball over the cell's Legal_Balls of the other folds; the
    cell's own counts ``n_fh_legal`` and ``n_non_fh_legal`` (int64) of those Legal_Balls; and
    ``pooled_free_hit_premium`` (bool), true where the cell holds no Free_Hit Legal_Ball or no
    non-Free_Hit Legal_Ball and both means are each one sum over one count of the other folds'
    Legal_Balls in all six cells.

    Raises StageError (exit code 5) when the other folds of a target fold hold no Free_Hit
    Legal_Ball, or no non-Free_Hit Legal_Ball, in any cell, so the pooled premium is undefined,
    naming the target fold. Raises ValueError or TypeError as extras_cell_means does for a
    malformed frame, and when ``free_hit`` is not boolean or is missing on a Legal_Ball it uses.
    """
    rows = _cell_rows(legal, "legal", cells, ("cls", "free_hit", "runs"), common.legal_training_mask)
    used = f"on the Legal_Balls that pass the training filters ({_TRAINING_FILTERS})"
    free_hit = _flags(legal["free_hit"].iloc[rows.at], f"legal column 'free_hit' {used}", len(rows.at))
    bat = _runs(legal, "runs", rows.at, f"legal column 'runs' {used}")

    n_fh = _other_folds(rows, free_hit)
    n_non_fh = _other_folds(rows, ~free_hit)
    kinds = (("Free_Hit Legal_Ball", n_fh), ("non-Free_Hit Legal_Ball", n_non_fh))
    problems: list[str] = []
    for t in range(len(rows.targets)):
        missing = [kind for kind, counts in kinds if not counts[t].any()]
        if missing:
            problems.append(_empty_pooled_set(rows.targets, t, (_PREMIUM,), " and no ".join(missing)))
    _stop_if_empty(problems)

    fh_sums = _other_folds(rows, free_hit, bat)
    non_fh_sums = _other_folds(rows, ~free_hit, bat)
    pooled = (n_fh == 0) | (n_non_fh == 0)
    pooled_premium = _pooled_means(fh_sums, n_fh) - _pooled_means(non_fh_sums, n_non_fh)
    cell_premium = fh_sums / np.maximum(n_fh, 1) - non_fh_sums / np.maximum(n_non_fh, 1)
    values = {
        _PREMIUM.column: np.where(pooled, pooled_premium[:, None], cell_premium),
        "n_fh_legal": n_fh,
        "n_non_fh_legal": n_non_fh,
        _PREMIUM.flag: pooled,
    }
    return _cell_table(rows.targets, values, FREE_HIT_PREMIUM_COLUMNS)


def cell_lookup(table: pd.DataFrame, fold: Any, cell: Any) -> pd.DataFrame:
    """The row of a cell table for each Delivery: the row whose ``fold`` and ``cell`` are the Delivery's.

    ``table`` is a table of extras_cell_means or free_hit_premium (any frame with ``fold`` and
    ``cell`` columns that holds each pair once). ``fold`` and ``cell`` hold each Delivery's fold
    and innings x Phase cell code, 1-D and of the same length. Returns the matching rows of
    ``table``, with all its columns, one per Delivery in the given order, on a RangeIndex.

    Raises ValueError when ``table`` lacks ``fold`` or ``cell`` or repeats a pair, a value is
    missing or not a whole number, or a Delivery's pair has no row in ``table``; TypeError for a
    value of the wrong type.
    """
    _require(table, (FOLD_COLUMN, CELL_COLUMN), "table")
    keys = pd.MultiIndex.from_arrays(
        [
            _whole(table[FOLD_COLUMN], f"table column {FOLD_COLUMN!r}").astype(np.int64),
            _whole(table[CELL_COLUMN], f"table column {CELL_COLUMN!r}").astype(np.int64),
        ]
    )
    if not keys.is_unique:
        raise ValueError(
            "table must hold each (fold, cell) pair once; pass a table of extras_cell_means or free_hit_premium"
        )
    folds = _whole(fold, "fold").astype(np.int64)
    codes = _whole(cell, "cell", len(folds)).astype(np.int64)
    at = keys.get_indexer(pd.MultiIndex.from_arrays([folds, codes]))
    absent = at < 0
    if absent.any():
        first = int(np.flatnonzero(absent)[0])
        raise ValueError(
            f"{int(absent.sum()):,} Delivery(ies) have a (fold, cell) pair with no row in table, for example fold "
            f"{folds[first]} and cell {codes[first]}; build the table from Deliveries of the same folds and cells"
        )
    return table.iloc[at].reset_index(drop=True)


def fallback_counts(
    bowler_rows: pd.DataFrame, cell_means: pd.DataFrame, fh_premium: pd.DataFrame, *, cells: str = CELL_COLUMN
) -> pd.DataFrame:
    """The number of Bowler_Deliveries that used the pooled value of each quantity (Requirement 15, criterion 22).

    ``bowler_rows`` holds one row per Bowler_Delivery whose bowler NBV is computed (every innings
    1-2 Delivery of the matches used for player rating), with its ``fold`` and its innings x Phase
    cell code in the column ``cells`` (``cell`` by default; the D8 robustness case names its own).
    ``cell_means`` and ``fh_premium`` are the tables of extras_cell_means and free_hit_premium,
    built with the same cells. Every Bowler_Delivery's expected Wide and No_Ball Extras_Costs use
    all four quantities of its (fold, cell) row, so a quantity's count is the number of rows of
    ``bowler_rows`` whose (fold, cell) row has the quantity's pooled flag.

    Returns FALLBACK_COUNTS_COLUMNS, one row per quantity of POOLED_QUANTITIES, in that order:
    ``quantity`` (``wide_team_runs``, ``nb_penalty_runs``, ``nb_bat_runs`` or
    ``free_hit_premium``) and ``n_bowler_deliveries`` (int64, 0 when no Bowler_Delivery used the
    pooled value). The caller logs a warning when a count is above 0.

    Raises the errors of cell_lookup, and ValueError or TypeError when a frame lacks a column or a
    pooled flag is not boolean or is missing.
    """
    _require(bowler_rows, (FOLD_COLUMN, _cell_column(cells, "bowler_rows")), "bowler_rows")
    _require(cell_means, EXTRAS_CELL_MEANS_COLUMNS, "cell_means")
    _require(fh_premium, FREE_HIT_PREMIUM_COLUMNS, "fh_premium")
    fold, cell = bowler_rows[FOLD_COLUMN], bowler_rows[cells]
    rows = {"cell_means": cell_lookup(cell_means, fold, cell), "fh_premium": cell_lookup(fh_premium, fold, cell)}
    counts = []
    for quantity in POOLED_QUANTITIES:
        source = "fh_premium" if quantity is _PREMIUM else "cell_means"
        flags = _flags(rows[source][quantity.flag], f"{source} column {quantity.flag!r}", len(bowler_rows))
        counts.append(int(flags.sum()))
    return pd.DataFrame(
        {
            FALLBACK_COUNTS_COLUMNS[0]: [quantity.label for quantity in POOLED_QUANTITIES],
            FALLBACK_COUNTS_COLUMNS[1]: np.asarray(counts, dtype=np.int64),
        }
    )


def write_fallback_counts(outdir: str | Path, counts: pd.DataFrame) -> Path:
    """Write the fallback_counts table to FALLBACK_COUNTS_FILE in ``outdir`` (criterion 22); return the path.

    ``outdir`` is the Value_Engine's folder (VALUE_FOLDER under the outputs folder in a Run) and is
    created when absent. The file has a header row and one row per quantity with the
    FALLBACK_COUNTS_COLUMNS, in UTF-8 with ``\\n`` line ends. It is written through a temporary
    file renamed in place, so it replaces any earlier copy whole. Raises ValueError when
    ``counts`` lacks a column or does not list the quantities of POOLED_QUANTITIES once each, in
    order, and TypeError when it is not a DataFrame.
    """
    _require(counts, FALLBACK_COUNTS_COLUMNS, "counts")
    labels = [quantity.label for quantity in POOLED_QUANTITIES]
    if counts[FALLBACK_COUNTS_COLUMNS[0]].tolist() != labels:
        raise ValueError(
            f"counts must list the quantities {', '.join(labels)} once each, in that order; pass the table of "
            "fallback_counts"
        )
    folder = Path(outdir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / FALLBACK_COUNTS_FILE
    temporary = path.with_name(path.name + ".tmp")
    try:
        counts[list(FALLBACK_COUNTS_COLUMNS)].to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


# ================================================================================== Net Ball Value

#: The DeliveryClass codes (``balls.cls``) of the three Delivery classes.
_LEGAL_CODE = int(DeliveryClass.LEGAL)
_WIDE_CODE = int(DeliveryClass.WIDE)
_NO_BALL_CODE = int(DeliveryClass.NO_BALL)
_CLASS_CODES = np.asarray([int(code) for code in DeliveryClass], dtype=np.int64)

#: The dismissal classes of M4, as columns of every M4 probability array (``outcome.CLASSES``):
#: W, the Bowler_Wicket class, and R, the Run_Out class.
W_CLASS = outcome.BOWLER_WICKET
R_CLASS = outcome.RUN_OUT
_M4_CLASSES = outcome.N_CLASSES

#: The two Roles (glossary), named as in the Pressure_Engine.
BATTER_ROLE = "batter"
BOWLER_ROLE = "bowler"
ROLES: tuple[str, str] = (BATTER_ROLE, BOWLER_ROLE)

#: Columns of ``d`` that each function reads (module docstring). ``bowler_nbv`` also reads the
#: cell code column it is given (``cell`` by default).
DISMISSAL_COLUMNS: tuple[str, ...] = ("counted_wicket", "retire_out", "dismissed_is_striker", "dismissed_fallback")
BATTER_NBV_COLUMNS: tuple[str, ...] = ("cls", "runs", *DISMISSAL_COLUMNS)
BOWLER_NBV_COLUMNS: tuple[str, ...] = ("cls", "runs", "team_runs", "bowler_wicket", FOLD_COLUMN)
DISMISSAL_CHARGE_INPUT_COLUMNS: tuple[str, ...] = (
    "row", "match_id", "innings", "innings_ball_num", "cls", "striker_id", "non_striker_id", *DISMISSAL_COLUMNS,
    "live", "pressure_bat",
)  # fmt: skip

#: Columns of the dismissal_charges table (design, Data Models, ``dismissal_charges``): the
#: Delivery's ``row``, the charged batter, the charge (-Omega), its pressure class, whether the
#: batter was the non-striker, the Delivery's class and the ``row`` whose batter-Role flag gave the
#: class (the Delivery itself, or for a Wide the latest earlier Batter_Delivery; <NA> when none).
DISMISSAL_CHARGE_COLUMNS: tuple[str, ...] = ("row", "player_id", "charge", "pressure", "non_striker", "cls", "flag_row")


class Dismissals(NamedTuple):
    """The result of dismissed_batters: one bool value per Delivery, in input order.

    * ``charged``: a Counted_Wicket that charges the dismissed batter: every one but a Retire Out,
      unless ``retire_out_penalty`` (criteria 1, 11 and 12);
    * ``striker``: ``charged`` and the dismissed batter is the striker (criteria 9 and 10);
    * ``fallback``: ``charged`` and the striker was taken because ``dismissed_player_id`` is blank
      or names neither batter (criterion 10).
    """

    charged: np.ndarray
    striker: np.ndarray
    fallback: np.ndarray


class BatterValue(NamedTuple):
    """The result of batter_nbv, one value per Delivery in input order; NaN on a Wide.

    * ``nbv``: the batter NBV, in runs (float64);
    * ``ev``: the batter Expected_Value E(p) - Omega x (p[W] + p[R]) (float64);
    * ``striker_out``: D, the bool flag of a charged dismissal of the striker on the Batter_Delivery
      (False on a Wide: a striker out off a Wide gets a Dismissal_Charge instead).
    """

    nbv: np.ndarray
    ev: np.ndarray
    striker_out: np.ndarray


class BowlerValue(NamedTuple):
    """The result of bowler_nbv, one value per Delivery in input order (float64 unless stated).

    * ``nbv``: the bowler NBV, in runs; ``ev``: the bowler Expected_Value EV_bowl;
    * ``extras_cost``: the Extras_Cost (0 on a Legal_Ball); ``conceded``: the runs conceded (the
      bat runs of a Legal_Ball, 0 on a Wide or No_Ball); ``credit``: B (bool);
    * ``exp_wide_cost`` and ``exp_nb_cost``: the expected Wide and No_Ball Extras_Costs (criterion 6).
    """

    nbv: np.ndarray
    ev: np.ndarray
    extras_cost: np.ndarray
    conceded: np.ndarray
    credit: np.ndarray
    exp_wide_cost: np.ndarray
    exp_nb_cost: np.ndarray


# ------------------------------------------------------------------------------- argument checks

def _switch(value: Any, what: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{what} must be True or False (design_choices.D14_retire_out_batter_penalty); got {value!r}")
    return bool(value)


def _classes(values: Any, what: str, n_rows: int) -> np.ndarray:
    """DeliveryClass codes (whole numbers, none missing) as int64."""
    codes = _whole(values, what, n_rows)
    unknown = ~np.isin(codes, _CLASS_CODES)
    if unknown.any():
        raise ValueError(
            f"{what} holds {int(unknown.sum()):,} value(s) that are not a DeliveryClass code (0 Legal_Ball, 1 Wide, "
            "2 No_Ball)"
        )
    return codes.astype(np.int64)


def _maybe_flags(values: Any, what: str, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """(flags, missing) of a nullable boolean column: the flags are False where a value is missing."""
    _vector(values, what)
    series = pd.Series(values) if _pandas(values) else pd.Series(np.asarray(values))
    if not pd.api.types.is_bool_dtype(series.dtype):
        raise TypeError(f"{what} must be boolean (missing values allowed); got dtype {series.dtype}")
    missing = _length(series.isna().to_numpy(dtype=bool), what, n_rows)
    return series.fillna(False).to_numpy(dtype=bool), missing


def _probabilities(p: Any, what: str, n_rows: int, used: np.ndarray) -> np.ndarray:
    """An (n_rows, 9) float64 copy of M4 probabilities, finite and within [0, 1] on the ``used`` rows."""
    array = np.asarray(p)
    if array.ndim != 2 or array.shape[1] != _M4_CLASSES:
        raise ValueError(
            f"{what} must have shape (Deliveries, {_M4_CLASSES}): one column per M4 class, in outcome.CLASSES order; "
            f"got shape {array.shape}"
        )
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{what} must hold numbers; got dtype {array.dtype}")
    array = _length(array.astype(np.float64), what, n_rows)
    rows = array[used]
    bad = ~np.isfinite(rows).all(axis=1) | ((rows < 0.0) | (rows > 1.0)).any(axis=1)
    if bad.any():
        raise ValueError(
            f"{what} is missing, not finite or outside [0, 1] on {int(bad.sum()):,} Delivery(ies) that use it"
        )
    return array


def _run_values(crv: Any, n_rows: int) -> np.ndarray:
    """Class_Run_Values: (9,) for every Delivery, or (n_rows, 9), one row per Delivery (float64)."""
    values = np.asarray(crv)
    if values.dtype.kind not in "iuf":
        raise TypeError(f"crv must hold numbers; got dtype {values.dtype}")
    values = values.astype(np.float64)
    if values.shape not in ((_M4_CLASSES,), (n_rows, _M4_CLASSES)):
        raise ValueError(
            f"crv must hold the {_M4_CLASSES} Class_Run_Values in outcome.CLASSES order, once (shape ({_M4_CLASSES},)) "
            f"or for each Delivery from its own fold (shape ({n_rows:,}, {_M4_CLASSES})); got shape {values.shape}"
        )
    if not (np.isfinite(values).all() and (values >= 0.0).all()):
        raise ValueError("crv holds a missing, negative or non-finite Class_Run_Value")
    return values


def _expected(p: np.ndarray, values: np.ndarray) -> np.ndarray:
    """E(p): the sum over M4 classes of p x the Class_Run_Value, per row (no checks)."""
    return (p * values).sum(axis=1) if values.ndim == 2 else p @ values


def _omega(omega: Any, n_rows: int, used: np.ndarray) -> np.ndarray:
    values = _floats(omega, "omega", n_rows)
    bad = used & ~(np.isfinite(values) & (values >= 0.0))
    if bad.any():
        raise ValueError(
            f"omega is missing, negative or not finite on {int(bad.sum()):,} Delivery(ies) that use it; Wicket_Cost is a "
            "finite number of runs, 0 or more"
        )
    return values


def _extras_probabilities(p_extras: Any, n_rows: int) -> np.ndarray:
    """An (n_rows, 2) float64 copy of the Extras_Model (Wide, No_Ball) probabilities, checked."""
    array = np.asarray(p_extras)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(
            "p_extras must have shape (Deliveries, 2): the Extras_Model Wide and No_Ball probabilities (the p_wide and "
            f"p_noball columns of extras_preds); got shape {array.shape}"
        )
    if array.dtype.kind not in "iuf":
        raise TypeError(f"p_extras must hold numbers; got dtype {array.dtype}")
    array = _length(array.astype(np.float64), "p_extras", n_rows)
    bad = ~np.isfinite(array).all(axis=1) | (array < 0.0).any(axis=1) | (array.sum(axis=1) > 1.0 + 1e-9)
    if bad.any():
        raise ValueError(
            f"p_extras is missing, negative, not finite or sums above 1 on {int(bad.sum()):,} Delivery(ies)"
        )
    return array


# ------------------------------------------------------------------------------------- formulas

def expected_bat_runs(p: Any, crv: Any) -> np.ndarray:
    """E(p): the expected bat runs of each row of M4 probabilities (Requirement 15, criterion 2).

    ``p`` has shape (n, 9), classes in ``outcome.CLASSES`` order, every value finite and within
    [0, 1]. ``crv`` holds the Class_Run_Values in the same order: shape (9,) for every row, or
    (n, 9) for each row's own fold. Returns float64 (n,): the sum over classes of p x crv. Raises
    ValueError or TypeError for a wrong shape or a missing, negative or non-finite value.
    """
    array = np.asarray(p)
    n_rows = len(array) if array.ndim == 2 else -1
    probs = _probabilities(p, "p", n_rows, np.ones(max(n_rows, 0), dtype=bool))
    return _expected(probs, _run_values(crv, len(probs)))


def dismissed_batters(d: pd.DataFrame, *, retire_out_penalty: bool) -> Dismissals:
    """Which Counted_Wickets charge a batter, and whether that batter is the striker (criteria 9, 10 and 12).

    ``d`` needs DISMISSAL_COLUMNS, as the ``balls`` table holds them: ``counted_wicket`` and
    ``retire_out`` (bool), ``dismissed_is_striker`` (nullable boolean, missing without a
    Counted_Wicket and on an unresolved Retire Out) and ``dismissed_fallback`` (bool).
    ``retire_out_penalty`` is the Config value ``design_choices.D14_retire_out_batter_penalty``:
    False charges a Retire Out to nobody; True charges it as any other dismissal, the striker when
    ``dismissed_player_id`` resolves to neither batter (then counted in ``fallback``).

    Returns Dismissals(charged, striker, fallback). Raises ValueError when a Retire Out is not a
    Counted_Wicket, ``dismissed_fallback`` marks a Delivery that is not a striker dismissal by a
    Counted_Wicket other than Retire Out, or a Counted_Wicket other than Retire Out has no
    dismissed batter; TypeError for a value of the wrong type.
    """
    _require(d, DISMISSAL_COLUMNS, "d")
    penalty = _switch(retire_out_penalty, "retire_out_penalty")
    n_rows = len(d)
    counted = _flags(d["counted_wicket"], "d column 'counted_wicket'", n_rows)
    retired = _flags(d["retire_out"], "d column 'retire_out'", n_rows)
    fallback = _flags(d["dismissed_fallback"], "d column 'dismissed_fallback'", n_rows)
    striker, unresolved = _maybe_flags(d["dismissed_is_striker"], "d column 'dismissed_is_striker'", n_rows)
    if (retired & ~counted).any():
        raise ValueError(
            f"d marks {int((retired & ~counted).sum()):,} Retire Out(s) that are not Counted_Wickets; a Retire Out "
            "(code 9) is a Counted_Wicket"
        )
    stray = fallback & ~(counted & ~retired & striker & ~unresolved)
    if stray.any():
        raise ValueError(
            f"d column 'dismissed_fallback' marks {int(stray.sum()):,} Delivery(ies) that are not a striker dismissal "
            "by a Counted_Wicket other than Retire Out (Requirement 15, criterion 10)"
        )
    charged = counted & (~retired | penalty)
    open_ = charged & unresolved
    if (open_ & ~retired).any():
        raise ValueError(
            f"{int((open_ & ~retired).sum()):,} Counted_Wicket(s) other than Retire Out have no dismissed batter "
            "(dismissed_is_striker missing); pass the flags of reconstruct.dismissed_batter_flags, which fall back to "
            "the striker (Requirement 15, criterion 10)"
        )
    return Dismissals(charged=charged, striker=charged & (striker | open_), fallback=(charged & fallback) | open_)


def batter_nbv(
    d: pd.DataFrame, p_bv: Any, p_bv_fh: Any, crv: Any, omega: Any, *, retire_out_penalty: bool
) -> BatterValue:
    """The batter NBV of every Batter_Delivery (Requirement 15, criteria 1-3; module docstring).

    ``d`` holds one row per Delivery with BATTER_NBV_COLUMNS: ``cls`` (DeliveryClass code),
    ``runs`` (bat runs; read on Legal_Balls and No_Balls) and the DISMISSAL_COLUMNS of
    dismissed_batters. ``p_bv`` and ``p_bv_fh`` are the (n, 9) ``batter_view`` and
    ``batter_view_fh`` M4 probabilities (read on Legal_Balls and on No_Balls respectively),
    ``crv`` the Class_Run_Values ((9,), or (n, 9) from each Delivery's fold), ``omega`` the
    Wicket_Cost of each Delivery and ``retire_out_penalty`` the D14 Config value
    (dismissed_batters). Rows a formula does not read may hold missing values.

    Returns BatterValue(nbv, ev, striker_out): NBV = bat runs - Omega x D - EV and
    EV = E(p) - Omega x (p[W] + p[R]), with p = p_bv on a Legal_Ball and p_bv_fh on a No_Ball;
    NaN on a Wide, which is not a Batter_Delivery. Raises the errors of dismissed_batters, and
    ValueError or TypeError for a missing column, a wrong shape, or a missing, negative or
    non-finite value that a Batter_Delivery uses.
    """
    _require(d, BATTER_NBV_COLUMNS, "d")
    n_rows = len(d)
    cls = _classes(d["cls"], "d column 'cls'", n_rows)
    legal, no_ball = cls == _LEGAL_CODE, cls == _NO_BALL_CODE
    batter = legal | no_ball
    at = np.flatnonzero(batter)
    bat = np.zeros(n_rows)
    bat[at] = _runs(d, "runs", at, "d column 'runs' on the Batter_Deliveries")
    om = _omega(omega, n_rows, batter)
    p = np.where(
        no_ball[:, None],
        _probabilities(p_bv_fh, "p_bv_fh", n_rows, no_ball),
        _probabilities(p_bv, "p_bv", n_rows, legal),
    )
    values = _run_values(crv, n_rows)
    out = dismissed_batters(d, retire_out_penalty=retire_out_penalty)
    striker_out = out.striker & batter
    ev = np.full(n_rows, np.nan)
    nbv = np.full(n_rows, np.nan)
    rows = p[at]
    ev[at] = _expected(rows, values if values.ndim == 1 else values[at]) - om[at] * (rows[:, W_CLASS] + rows[:, R_CLASS])
    nbv[at] = bat[at] - om[at] * striker_out[at] - ev[at]
    return BatterValue(nbv=nbv, ev=ev, striker_out=striker_out)


def bowler_nbv(
    d: pd.DataFrame,
    p_wv: Any,
    p_wv_nofh: Any,
    p_extras: Any,
    crv: Any,
    omega: Any,
    cell_means: pd.DataFrame,
    fh_premium: pd.DataFrame,
    *,
    cells: str = CELL_COLUMN,
) -> BowlerValue:
    """The bowler NBV of every Bowler_Delivery (Requirement 15, criteria 4-8, 13 and 14; module docstring).

    ``d`` holds one row per Delivery with BOWLER_NBV_COLUMNS and the cell column ``cells``:
    ``cls``, ``runs`` (bat runs; read on Legal_Balls), ``team_runs`` (team runs under the
    selected Extras_Encoding; read on Wides and No_Balls), ``bowler_wicket`` (B), ``fold`` and the
    innings x Phase cell code. ``p_wv`` and ``p_wv_nofh`` are the (n, 9) ``bowler_view`` and
    ``bowler_view_nofh`` M4 probabilities, ``p_extras`` the (n, 2) Extras_Model Wide and No_Ball
    probabilities, ``crv`` the Class_Run_Values ((9,) or (n, 9)) and ``omega`` the Wicket_Cost.
    ``cell_means`` and ``fh_premium`` are the tables of extras_cell_means and free_hit_premium (or
    the merged CELL_MEANS_TABLE for both); each Delivery reads its (fold, cell) row.

    Returns BowlerValue(nbv, ev, extras_cost, conceded, credit, exp_wide_cost, exp_nb_cost).
    Raises the errors of cell_lookup, and ValueError or TypeError for a missing column, a wrong
    shape, or a missing, negative or non-finite value that a Delivery uses.
    """
    _require(d, (*BOWLER_NBV_COLUMNS, _cell_column(cells, "d")), "d")
    _require(cell_means, (FOLD_COLUMN, CELL_COLUMN, _WIDE.column, _NB_PENALTY.column, _NB_BAT.column), "cell_means")
    _require(fh_premium, (FOLD_COLUMN, CELL_COLUMN, _PREMIUM.column), "fh_premium")
    n_rows = len(d)
    every = np.ones(n_rows, dtype=bool)
    cls = _classes(d["cls"], "d column 'cls'", n_rows)
    legal, wide, no_ball = cls == _LEGAL_CODE, cls == _WIDE_CODE, cls == _NO_BALL_CODE
    bat = np.zeros(n_rows)
    at = np.flatnonzero(legal)
    bat[at] = _runs(d, "runs", at, "d column 'runs' on the Legal_Balls")
    team = np.zeros(n_rows)
    at = np.flatnonzero(wide | no_ball)
    team[at] = _runs(d, "team_runs", at, "d column 'team_runs' on the Wides and No_Balls")
    credit = _flags(d["bowler_wicket"], "d column 'bowler_wicket'", n_rows)
    om = _omega(omega, n_rows, every)
    p_view = _probabilities(p_wv, "p_wv", n_rows, every)
    p_nofh = _probabilities(p_wv_nofh, "p_wv_nofh", n_rows, every)
    p_wide, p_nb = _extras_probabilities(p_extras, n_rows).T
    values = _run_values(crv, n_rows)
    means = cell_lookup(cell_means, d[FOLD_COLUMN], d[cells])
    premium = cell_lookup(fh_premium, d[FOLD_COLUMN], d[cells])
    quantities: dict[str, np.ndarray] = {}
    for source, frame, quantity in (
        ("cell_means", means, _WIDE), ("cell_means", means, _NB_PENALTY), ("cell_means", means, _NB_BAT),
        ("fh_premium", premium, _PREMIUM),
    ):  # fmt: skip
        quantities[quantity.column] = _floats(frame[quantity.column], f"{source} column {quantity.column!r}", n_rows)
        if not np.isfinite(quantities[quantity.column]).all():
            raise ValueError(f"{source} column {quantity.column!r} is missing or not finite on a (fold, cell) row in use")

    e_view, e_nofh = _expected(p_view, values), _expected(p_nofh, values)
    premium_runs = quantities[_PREMIUM.column]
    exp_wide_cost = quantities[_WIDE.column] + e_view
    exp_nb_cost = quantities[_NB_PENALTY.column] + quantities[_NB_BAT.column] + e_nofh + premium_runs
    ev = (1.0 - p_wide - p_nb) * (om * p_view[:, W_CLASS] - e_view) - p_wide * exp_wide_cost - p_nb * exp_nb_cost
    conceded = np.where(legal, bat, 0.0)
    extras_cost = np.where(wide, team + e_view, np.where(no_ball, team + e_nofh + premium_runs, 0.0))
    nbv = -conceded + om * credit - extras_cost - ev
    return BowlerValue(
        nbv=nbv, ev=ev, extras_cost=extras_cost, conceded=conceded, credit=credit, exp_wide_cost=exp_wide_cost,
        exp_nb_cost=exp_nb_cost,
    )  # fmt: skip


def _previous_batter_delivery(d: pd.DataFrame, batter: np.ndarray) -> np.ndarray:
    """Position in ``d`` of the latest earlier Batter_Delivery of each row's innings (-1 when none).

    Earlier means before the row in the Requirement 5 order (``match_id``, ``innings``,
    ``innings_ball_num``), ties kept in ``d`` order.
    """
    n_rows = len(d)
    previous = np.full(n_rows, -1, dtype=np.int64)
    if not n_rows:
        return previous
    match = _whole(d["match_id"], "d column 'match_id'", n_rows)
    innings = _whole(d["innings"], "d column 'innings'", n_rows)
    ball = _whole(d["innings_ball_num"], "d column 'innings_ball_num'", n_rows)
    order = np.lexsort((ball, innings, match))
    position = np.arange(n_rows)
    latest = np.maximum.accumulate(np.where(batter[order], position, -1))
    earlier = np.concatenate(([-1], latest[:-1]))
    m, i = match[order], innings[order]
    starts = np.ones(n_rows, dtype=bool)
    starts[1:] = (m[1:] != m[:-1]) | (i[1:] != i[:-1])
    start = np.maximum.accumulate(np.where(starts, position, 0))
    found = earlier >= start
    previous[order[found]] = order[earlier[found]]
    return previous


def dismissal_charges(d: pd.DataFrame, omega: Any, *, retire_out_penalty: bool) -> pd.DataFrame:
    """The Dismissal_Charges of the Deliveries in ``d`` (Requirement 15, criteria 9-12; module docstring).

    ``d`` holds one row per Delivery with DISMISSAL_CHARGE_INPUT_COLUMNS: ``row``, the order keys
    ``match_id``, ``innings`` and ``innings_ball_num``, ``cls``, ``striker_id`` and
    ``non_striker_id``, the DISMISSAL_COLUMNS of dismissed_batters, ``live`` (bool) and
    ``pressure_bat`` (nullable boolean, missing on Wides only), as the ``balls`` and ``pressure``
    tables hold them. ``omega`` is the Wicket_Cost of each Delivery (read on the charged ones) and
    ``retire_out_penalty`` the D14 Config value.

    A charge falls on the non-striker dismissed on any Delivery and on the striker dismissed off a
    Wide. Returns DISMISSAL_CHARGE_COLUMNS, one row per charge in ``d`` order: ``row`` and
    ``player_id`` (int32), ``charge`` = -Omega (float64), ``pressure`` (bool: the Delivery's
    ``pressure_bat``; on a Wide, the flag of the latest earlier Batter_Delivery of the innings,
    false on a Dead_Ball or when there is none), ``non_striker`` (bool), ``cls`` (int8) and
    ``flag_row`` (Int32, the row whose flag was used; <NA> when none). Raises the errors of
    dismissed_batters, and ValueError or TypeError for a missing column or value it needs.
    """
    _require(d, DISMISSAL_CHARGE_INPUT_COLUMNS, "d")
    n_rows = len(d)
    cls = _classes(d["cls"], "d column 'cls'", n_rows)
    wide = cls == _WIDE_CODE
    out = dismissed_batters(d, retire_out_penalty=retire_out_penalty)
    charged = out.charged & (~out.striker | wide)
    om = _omega(omega, n_rows, charged)
    row_ids = _whole(d["row"], "d column 'row'", n_rows)
    live = _flags(d["live"], "d column 'live'", n_rows)
    flag, no_flag = _maybe_flags(d["pressure_bat"], "d column 'pressure_bat'", n_rows)
    if (no_flag & ~wide).any():
        raise ValueError(
            f"d column 'pressure_bat' is missing on {int((no_flag & ~wide).sum()):,} Batter_Delivery(ies); the "
            "batter-Role Pressure_Ball flag is missing on Wides only"
        )
    at = np.flatnonzero(charged)
    on_wide = wide[at]
    source = np.where(on_wide, _previous_batter_delivery(d, ~wide)[at], at)
    has = source >= 0
    pressure = np.zeros(len(at), dtype=bool)
    pressure[has] = flag[source[has]]
    pressure &= live[at] | ~on_wide
    striker_out = out.striker[at]
    striker_ids = _whole(d["striker_id"].iloc[at], "d column 'striker_id' on the charged Deliveries")
    non_striker_ids = _whole(
        d["non_striker_id"].iloc[at], "d column 'non_striker_id' on the charged Deliveries", missing=True
    )
    player = np.where(striker_out, striker_ids, non_striker_ids)
    if np.isnan(player).any():
        raise ValueError(
            f"d column 'non_striker_id' is missing on {int(np.isnan(player).sum()):,} Delivery(ies) whose non-striker "
            "is dismissed"
        )
    flag_row = pd.arrays.IntegerArray(
        np.where(has, row_ids[np.maximum(source, 0)], 0).astype(np.int32), ~has
    ) if len(at) else pd.array([], dtype="Int32")  # fmt: skip
    return pd.DataFrame(
        {
            "row": row_ids[at].astype(np.int32),
            "player_id": player.astype(np.int32),
            "charge": 0.0 - om[at],
            "pressure": pressure,
            "non_striker": ~striker_out,
            "cls": cls[at].astype(np.int8),
            "flag_row": flag_row,
        },
        columns=list(DISMISSAL_CHARGE_COLUMNS),
    )


# ------------------------------------------------------------------------------- cell summaries

#: Columns of the nbv_cells table (nbv_cell_means): one row per Role and innings x Phase cell.
NBV_CELL_COLUMNS: tuple[str, ...] = (
    "role", "innings", "phase", "cell", "n_deliveries", "n_dismissal_charges", "sum_nbv", "sum_dismissal_charges",
    "mean_actual", "mean_expected", "mean_per_delivery",
)  # fmt: skip

_VALUE_SUMMARY_COLUMNS: tuple[str, ...] = (
    "row", "cell", "cls", "omega", "bat_runs", "striker_out", "ev_bat", "nbv_bat", "conceded", "bowler_credit",
    "extras_cost", "ev_bowl", "nbv_bowl",
)  # fmt: skip


def nbv_cell_means(value: pd.DataFrame, charges: pd.DataFrame) -> pd.DataFrame:
    """The NBV per Delivery of each Role and innings x Phase cell (Requirement 15, criterion 17; NBV_CELL_COLUMNS).

    ``value`` is a VALUE_TABLE (its ``row``, ``cell``, ``cls``, NBV, Expected_Value and actual
    columns) and ``charges`` the DISMISSAL_CHARGES_TABLE of the same Deliveries (``row`` and
    ``charge``). For each Role and cell of CELLS: ``n_deliveries`` (Batter_Deliveries or
    Bowler_Deliveries), ``n_dismissal_charges`` and ``sum_dismissal_charges`` (the batter Role
    only; 0 for the bowler), ``sum_nbv``, and per Delivery the ``mean_actual`` (bat runs - Omega x
    D; or -conceded + Omega x B - Extras_Cost), ``mean_expected`` (the Expected_Value) and
    ``mean_per_delivery`` = (``sum_nbv`` + ``sum_dismissal_charges``) / ``n_deliveries`` (NaN for
    an empty cell). Raises ValueError when a column is missing or a charge's row is not in ``value``.
    """
    _require(value, _VALUE_SUMMARY_COLUMNS, "value")
    _require(charges, ("row", "charge"), "charges")
    n_rows = len(value)
    cell = _whole(value["cell"], "value column 'cell'", n_rows).astype(np.int64)
    if not np.isin(cell, CELLS).all():
        raise ValueError(f"value column 'cell' must hold innings x Phase cell codes 0 to {N_CELLS - 1}")
    cls = _classes(value["cls"], "value column 'cls'", n_rows)
    index = pd.Index(_whole(value["row"], "value column 'row'", n_rows).astype(np.int64))
    where = index.get_indexer(_whole(charges["row"], "charges column 'row'").astype(np.int64))
    if (where < 0).any():
        raise ValueError(f"{int((where < 0).sum()):,} Dismissal_Charge(s) have a row that is not in value")
    charge = _floats(charges["charge"], "charges column 'charge'")

    def column(name: str) -> np.ndarray:
        return _floats(value[name], f"value column {name!r}", n_rows)

    omega = column("omega")
    batter = cls != _WIDE_CODE
    roles = {
        BATTER_ROLE: (
            batter, column("nbv_bat"), column("ev_bat"),
            column("bat_runs") - omega * _flags(value["striker_out"], "value column 'striker_out'", n_rows),
        ),
        BOWLER_ROLE: (
            np.ones(n_rows, dtype=bool), column("nbv_bowl"), column("ev_bowl"),
            -column("conceded") + omega * _flags(value["bowler_credit"], "value column 'bowler_credit'", n_rows)
            - column("extras_cost"),
        ),
    }  # fmt: skip
    records: list[dict[str, Any]] = []
    for role, (mask, nbv, ev, actual) in roles.items():
        counts = np.bincount(cell[mask], minlength=N_CELLS)
        sums = {
            name: np.bincount(cell[mask], weights=values[mask], minlength=N_CELLS)
            for name, values in (("nbv", nbv), ("ev", ev), ("actual", actual))
        }
        if role == BATTER_ROLE:
            n_charges = np.bincount(cell[where], minlength=N_CELLS)
            charge_sums = np.bincount(cell[where], weights=charge, minlength=N_CELLS)
        else:
            n_charges, charge_sums = np.zeros(N_CELLS, dtype=np.int64), np.zeros(N_CELLS)
        for code in CELLS:
            n = int(counts[code])
            records.append(
                {
                    "role": role,
                    "innings": STATE_INNINGS[code // len(Phase)],
                    "phase": Phase(code % len(Phase)).name.lower(),
                    "cell": code,
                    "n_deliveries": n,
                    "n_dismissal_charges": int(n_charges[code]),
                    "sum_nbv": float(sums["nbv"][code]),
                    "sum_dismissal_charges": float(charge_sums[code]),
                    "mean_actual": float(sums["actual"][code]) / n if n else math.nan,
                    "mean_expected": float(sums["ev"][code]) / n if n else math.nan,
                    "mean_per_delivery": (float(sums["nbv"][code]) + float(charge_sums[code])) / n if n else math.nan,
                }
            )
    table = pd.DataFrame.from_records(records, columns=list(NBV_CELL_COLUMNS))
    return table.astype({"innings": np.int8, "cell": np.int8, "n_deliveries": np.int64, "n_dismissal_charges": np.int64})


# ============================================================================== the ``value`` stage

#: Name of the Pipeline_Stage that runs the Value_Engine, and of its checkpoint.
STAGE_NAME = "value"

#: Checkpoint tables of the stage (design, Data Models).
VALUE_TABLE = "value"
DISMISSAL_CHARGES_TABLE = "dismissal_charges"
CELL_MEANS_TABLE = "cell_means"
FALLBACK_COUNTS_TABLE = "fallback_counts"
NBV_CELLS_TABLE = "nbv_cells"
#: The cell means of the NBV with each player's own theta (own_theta_cell_means; criterion 17).
NBV_CELLS_OWN_THETA_TABLE = "nbv_cells_own_theta"

#: Columns of the value table: one row per Delivery of innings 1 and 2 of the matches used for
#: player rating, in ``state`` order. Keys; Omega; the batter Role (bat runs, D, EV and NBV, NaN on
#: a Wide); the bowler Role (runs conceded, B, Extras_Cost, the expected Wide and No_Ball costs,
#: EV and NBV).
VALUE_COLUMNS: tuple[str, ...] = (
    "row", "match_id", "innings", FOLD_COLUMN, CELL_COLUMN, "cls", "omega",
    "bat_runs", "striker_out", "ev_bat", "nbv_bat",
    "conceded", "bowler_credit", "extras_cost", "exp_wide_cost", "exp_nb_cost", "ev_bowl", "nbv_bowl",
)  # fmt: skip

#: Columns of the cell_means table (design, Data Models): the two cell tables merged on the keys.
CELL_MEANS_COLUMNS: tuple[str, ...] = (
    FOLD_COLUMN, CELL_COLUMN, _WIDE.column, _NB_PENALTY.column, _NB_BAT.column, _PREMIUM.column,
    "n_wide", "n_nb", "n_fh_legal", "n_non_fh_legal", _WIDE.flag, _NB_PENALTY.flag, _NB_BAT.flag, _PREMIUM.flag,
)  # fmt: skip

#: Files in VALUE_FOLDER besides FALLBACK_COUNTS_FILE.
CELL_MEANS_FILE = "cell_means.csv"
NBV_CELLS_FILE = "nbv_cells.csv"
NBV_CELLS_OWN_THETA_FILE = "nbv_cells_own_theta.csv"
EXTREMES_FILES: Mapping[str, str] = {BATTER_ROLE: "extremes_batter.csv", BOWLER_ROLE: "extremes_bowler.csv"}

#: The two-candidate own-theta NBV diagnostic of the M4 calibration-aware tie-break (Requirement
#: 23, criterion 14): one row per Role and innings x Phase cell, with the own-theta NBV per
#: Delivery of the selected preferred family and of the displaced lowest-log-loss family, computed
#: on the same inputs (only the M4 variant arrays differ).
M4_TIEBREAK_CELLS_FILE = "m4_tiebreak_cells.csv"
M4_TIEBREAK_CELL_COLUMNS: tuple[str, ...] = (
    "role", "innings", "phase", "cell", "n_deliveries", "selected_family", "selected_mean_per_delivery",
    "selected_within_tol", "displaced_family", "displaced_mean_per_delivery", "displaced_within_tol", "tolerance",
)  # fmt: skip

#: Deliveries listed at each end of each Role's NBV order (Requirement 15, criterion 19).
EXTREMES_SIZE = 20

#: Columns of each extremes file (criterion 19): which end and the rank there (1 = most extreme),
#: the Delivery, the player of the Role (striker or bowler), the pre-ball state, the outcome, Omega,
#: Expected_Value and NBV. The bowler file adds ``extras_cost`` before ``expected_value``.
EXTREME_COLUMNS: tuple[str, ...] = (
    "list", "rank", "row", "match_id", "match_date", "batting_team", "bowling_team", "innings", "over", "ball",
    "player_id", "player_name", "score", "wickets", "runs_needed", "bat_runs", "extras_type", "dismissal_type",
    "omega", "expected_value", "nbv",
)  # fmt: skip
_HIGHEST, _LOWEST = "highest", "lowest"

#: Data-derived constants of the stage (for the Report_Builder).
STRIKER_FALLBACK_CONSTANT = "value.striker_fallback_count"
POOLED_FALLBACK_CONSTANT = "value.pooled_fallback_counts"
DISMISSAL_CHARGES_CONSTANT = "value.dismissal_charge_counts"
RETIRE_OUT_CONSTANT = "value.retire_out_uncharged"
NBV_CELL_MEANS_CONSTANT = "value.nbv_cell_means"

_COMPONENT = "Value_Engine"
_RETIRE_OUT_PENALTY = "design_choices.D14_retire_out_batter_penalty"
_CELL_TOLERANCE = "tests.nbv_cell_tol"
_LOAD_STAGE = "load"
_BALL_BY_BALL_TABLE = "ball_by_ball"
_MATCHES_TABLE = "matches"
_FLOAT_FORMAT = "%.6f"

#: The M4 variants of the NBV and what each must change (``outcome.VARIANTS``): the Role player's
#: theta to the fold's Average_Skill, the opponent's actual theta kept (Requirement 2, criterion
#: 4), and the Delivery's own Free_Hit flag kept (batter_view, bowler_view), set (batter_view_fh,
#: criterion 3) or cleared (bowler_view_nofh, criterion 14).
_M4_VIEWS: Mapping[str, Mapping[str, Any]] = {
    outcome.BATTER_VIEW: {"theta_striker": outcome.AVERAGE},
    outcome.BATTER_VIEW_FH: {"theta_striker": outcome.AVERAGE, "free_hit": True},
    outcome.BOWLER_VIEW: {"theta_bowler": outcome.AVERAGE},
    outcome.BOWLER_VIEW_NOFH: {"theta_bowler": outcome.AVERAGE, "free_hit": False},
}

#: The M4 variant that own_theta_nbv reads for each of the four views: NBV_VIEWS gives each view
#: its own variant, as the stage's NBV reads them; OWN_THETA_VIEWS gives ``batter_view`` and
#: ``bowler_view`` the ``actual`` variant, each player's own theta, and keeps the two Free_Hit
#: views of the No_Ball terms (criterion 17; module docstring, "The own-theta check").
NBV_VIEWS: Mapping[str, str] = {name: name for name in _M4_VIEWS}
OWN_THETA_VIEWS: Mapping[str, str] = {
    outcome.BATTER_VIEW: outcome.ACTUAL,
    outcome.BATTER_VIEW_FH: outcome.BATTER_VIEW_FH,
    outcome.BOWLER_VIEW: outcome.ACTUAL,
    outcome.BOWLER_VIEW_NOFH: outcome.BOWLER_VIEW_NOFH,
}

#: The M4 variants that load_value_inputs loads unless told otherwise, and the stage with it.
VALUE_VARIANTS: tuple[str, ...] = (*_M4_VIEWS, outcome.ACTUAL)

#: Columns of the per-Delivery frame the stage builds: ``balls`` columns, ``fold`` and ``cell``.
_FRAME_COLUMNS: tuple[str, ...] = (
    "row", "match_id", "innings", "overid", "over_ball_num", "innings_ball_num", "batting_team_id", "bowling_team_id",
    "striker_id", "non_striker_id", "bowler_id", "runs", "extras_type", "dismissal_type_id", "dismissal_type", "cls",
    "is_super_over", "free_hit", "team_runs", "counted_wicket", "bowler_wicket", "run_out", "retire_out",
    "dismissed_is_striker", "dismissed_fallback", "match_date", "excluded", FOLD_COLUMN, CELL_COLUMN,
)  # fmt: skip


def _check_m4_views() -> None:
    wrong = [name for name, change in _M4_VIEWS.items() if dict(outcome.VARIANTS.get(name, {})) != dict(change)]
    if wrong:
        raise StageError(
            f"the M4 variant(s) {', '.join(wrong)} no longer change only the Role player's theta (and the Free_Hit flag "
            "where named), so the Expected_Value would not keep the opponent's actual theta (Requirement 2, criterion 4)"
        )


def _aligned(table: pd.DataFrame, row_ids: np.ndarray, columns: Sequence[str], what: str) -> pd.DataFrame:
    """``columns`` of ``table`` for each of ``row_ids``, in that order, on a RangeIndex."""
    _require(table, ("row", *columns), what)
    index = pd.Index(_whole(table["row"], f"{what} column 'row'").astype(np.int64))
    if not index.is_unique:
        raise ValueError(f"{what} must hold each row once")
    at = index.get_indexer(row_ids)
    if (at < 0).any():
        raise ValueError(f"{int((at < 0).sum()):,} state row(s) have no row in {what}; pass the checkpoints of one Run")
    return table.iloc[at][list(dict.fromkeys(columns))].reset_index(drop=True)


def _delivery_frame(balls: pd.DataFrame, state: pd.DataFrame, fold_table: pd.DataFrame) -> pd.DataFrame:
    """Every Delivery of ``balls`` with its ``fold`` and its ``state`` cell (<NA> without a ``state`` row)."""
    frame = folds.attach_folds(balls, fold_table)
    state_rows = pd.Index(_whole(state["row"], "state column 'row'").astype(np.int64))
    if not state_rows.is_unique:
        raise ValueError("state must hold each row once")
    at = state_rows.get_indexer(_whole(frame["row"], "balls column 'row'").astype(np.int64))
    codes = _whole(state[CELL_COLUMN], f"state column {CELL_COLUMN!r}")
    frame[CELL_COLUMN] = pd.arrays.IntegerArray(np.where(at >= 0, codes[np.maximum(at, 0)], 0).astype(np.int8), at < 0)
    return frame


def _fold_run_values(by_fold: Mapping[int, np.ndarray], fold: np.ndarray) -> np.ndarray:
    """The Class_Run_Values of each Delivery's own fold, shape (n, 9)."""
    keys = np.asarray(sorted(by_fold), dtype=np.int64)
    table = np.stack([np.asarray(by_fold[int(k)], dtype=np.float64) for k in keys])
    at = np.searchsorted(keys, fold)
    known = (at < len(keys)) & (keys[np.minimum(at, len(keys) - 1)] == fold)
    if not known.all():
        raise ValueError(
            f"the Class_Run_Values hold no row for fold(s) {sorted(set(fold[~known].tolist()))}; pass the outcome "
            "checkpoint of the same Run"
        )
    return table[at]


def _value_table(d: pd.DataFrame, omega: np.ndarray, batter: BatterValue, bowler: BowlerValue) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row": d["row"].to_numpy(dtype=np.int32),
            "match_id": d["match_id"].to_numpy(dtype=np.int32),
            "innings": d["innings"].to_numpy(dtype=np.int8),
            FOLD_COLUMN: d[FOLD_COLUMN].to_numpy(dtype=np.int8),
            CELL_COLUMN: d[CELL_COLUMN].to_numpy(dtype=np.int8),
            "cls": d["cls"].to_numpy(dtype=np.int8),
            "omega": omega,
            "bat_runs": d["runs"].to_numpy(dtype=np.int32),
            "striker_out": batter.striker_out,
            "ev_bat": batter.ev,
            "nbv_bat": batter.nbv,
            "conceded": bowler.conceded.astype(np.int32),
            "bowler_credit": bowler.credit,
            "extras_cost": bowler.extras_cost,
            "exp_wide_cost": bowler.exp_wide_cost,
            "exp_nb_cost": bowler.exp_nb_cost,
            "ev_bowl": bowler.ev,
            "nbv_bowl": bowler.nbv,
        },
        columns=list(VALUE_COLUMNS),
    )


def _check_finite(cls: np.ndarray, batter: BatterValue, bowler: BowlerValue, charges: pd.DataFrame) -> None:
    """Stop the Run unless every Role Delivery has a finite NBV and every charge is finite (criterion 16)."""
    counts = {
        "Batter_Deliveries without a finite batter NBV": int(((cls != _WIDE_CODE) & ~np.isfinite(batter.nbv)).sum()),
        "Bowler_Deliveries without a finite bowler NBV": int((~np.isfinite(bowler.nbv)).sum()),
        "Dismissal_Charges that are not finite": int((~np.isfinite(charges["charge"].to_numpy(dtype=np.float64))).sum()),
    }
    problems = [f"{count:,} {what}" for what, count in counts.items() if count]
    if problems:
        raise StageError(
            "Requirement 15, criterion 16, needs a finite NBV for every Batter_Delivery and Bowler_Delivery of the "
            f"matches used for player rating: {'; '.join(problems)}"
        )


def _merged_cell_table(means: pd.DataFrame, premium: pd.DataFrame) -> pd.DataFrame:
    merged = means.merge(premium, on=[FOLD_COLUMN, CELL_COLUMN], how="outer", validate="one_to_one", sort=True)
    if len(merged) != len(means) or len(merged) != len(premium):
        raise ValueError("the extras_cell_means and free_hit_premium tables must hold the same (fold, cell) rows")
    return merged[list(CELL_MEANS_COLUMNS)].reset_index(drop=True)


def _extremes(nbv: np.ndarray, size: int) -> list[tuple[str, np.ndarray]]:
    """Positions of the ``size`` highest and the ``size`` lowest finite NBVs; ties: the earlier position first."""
    finite = np.flatnonzero(np.isfinite(nbv))
    return [
        (_HIGHEST, finite[np.lexsort((finite, -nbv[finite]))][:size]),
        (_LOWEST, finite[np.lexsort((finite, nbv[finite]))][:size]),
    ]


def _delivery_names(deliveries: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """``striker_name`` and ``bowler_name`` of each Delivery of ``rows`` (its ``row``), from ``ball_by_ball``.

    ``ball_by_ball`` is the ``load`` table; ``reconstruct.order_deliveries`` numbers it by ``row``
    as the ``reconstruct`` stage does. Raises ValueError when the ids of a row do not match.
    """
    keys = ["match_id", "innings", "innings_ball_num", "striker_id", "bowler_id", "striker_name", "bowler_name"]
    _require(deliveries, keys, _BALL_BY_BALL_TABLE)
    ordered = reconstruct.order_deliveries(deliveries[keys])
    named = _aligned(ordered, _whole(rows["row"], "rows column 'row'").astype(np.int64), keys, _BALL_BY_BALL_TABLE)
    for column in ("striker_id", "bowler_id"):
        if not np.array_equal(
            _whole(named[column], f"{_BALL_BY_BALL_TABLE} column {column!r}"), _whole(rows[column], f"rows {column!r}")
        ):
            raise ValueError(
                f"the {_BALL_BY_BALL_TABLE} table does not give the {column} of balls; pass the checkpoints of one Run"
            )
    return named[["striker_name", "bowler_name"]]


def _team_names(matches: pd.DataFrame, match_id: np.ndarray, team_id: Any) -> list[str]:
    """The name of each (match, team) pair, from the home and away columns of ``matches`` (blank when unknown)."""
    _require(matches, ("match_id", "home_team_id", "home_team", "away_team_id", "away_team"), _MATCHES_TABLE)
    names: dict[tuple[int, int], str] = {}
    for side in ("home", "away"):
        ids = _whole(matches[f"{side}_team_id"], f"{_MATCHES_TABLE} column '{side}_team_id'", missing=True)
        for match, team, name in zip(
            _whole(matches["match_id"], f"{_MATCHES_TABLE} column 'match_id'").tolist(), ids.tolist(),
            matches[f"{side}_team"].tolist(),
        ):  # fmt: skip
            if not math.isnan(team) and isinstance(name, str):
                names.setdefault((int(match), int(team)), name)
    teams = _whole(team_id, "team ids", len(match_id), missing=True)
    return [
        "" if math.isnan(team) else names.get((int(match), int(team)), "")
        for match, team in zip(match_id.tolist(), teams.tolist())
    ]


def _extreme_table(
    role: str, d: pd.DataFrame, value: pd.DataFrame, names: pd.DataFrame, matches: pd.DataFrame
) -> pd.DataFrame:
    """The EXTREMES_SIZE highest and lowest NBVs of ``role`` with the columns of criterion 19 (EXTREME_COLUMNS)."""
    batter = role == BATTER_ROLE
    suffix = "bat" if batter else "bowl"
    nbv = value[f"nbv_{suffix}"].to_numpy(dtype=np.float64)
    parts = []
    for label, at in _extremes(nbv, EXTREMES_SIZE):
        rows = d.iloc[at].reset_index(drop=True)
        match_id = rows["match_id"].to_numpy(dtype=np.int64)
        part = pd.DataFrame(
            {
                "list": label,
                "rank": np.arange(1, len(at) + 1, dtype=np.int64),
                "row": rows["row"].to_numpy(dtype=np.int64),
                "match_id": match_id,
                "match_date": pd.to_datetime(rows["match_date"]).dt.strftime("%Y-%m-%d"),
                "batting_team": _team_names(matches, match_id, rows["batting_team_id"]),
                "bowling_team": _team_names(matches, match_id, rows["bowling_team_id"]),
                "innings": rows["innings"].to_numpy(dtype=np.int64),
                "over": rows["overid"].astype("Int64"),
                "ball": rows["over_ball_num"].astype("Int64"),
                "player_id": rows["striker_id" if batter else "bowler_id"].to_numpy(dtype=np.int64),
                "player_name": names["striker_name" if batter else "bowler_name"].iloc[at].reset_index(drop=True),
                "score": rows["score"].to_numpy(dtype=np.int64),
                "wickets": rows["wickets"].to_numpy(dtype=np.int64),
                "runs_needed": rows["runs_needed"].astype("Int64"),
                "bat_runs": rows["runs"].to_numpy(dtype=np.int64),
                "extras_type": rows["extras_type"],
                "dismissal_type": rows["dismissal_type"],
                "omega": value["omega"].to_numpy(dtype=np.float64)[at],
                "extras_cost": value["extras_cost"].to_numpy(dtype=np.float64)[at],
                "expected_value": value[f"ev_{suffix}"].to_numpy(dtype=np.float64)[at],
                "nbv": nbv[at],
            }
        )
        parts.append(part)
    columns = list(EXTREME_COLUMNS)
    if not batter:
        columns.insert(columns.index("expected_value"), "extras_cost")
    return pd.concat(parts, ignore_index=True)[columns]


def _write_csv(path: Path, table: pd.DataFrame) -> Path:
    """Write ``table`` to ``path`` (UTF-8, ``\\n`` line ends, floats ``%.6f``, NaN blank) through ``<file>.tmp``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        table.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n", float_format=_FLOAT_FORMAT)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _shown_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return str(path)


# ------------------------------------------------------------- the stage's inputs and the own-theta check

class ValueInputs(NamedTuple):
    """The inputs of the NBV formulas, as the ``value`` stage builds them (load_value_inputs).

    One row per Delivery of innings 1 and 2 of the matches used for player rating (no match
    excluded under Requirement 5, criterion 24), in ``state`` order; every array follows ``d``:

    * ``d``: the _FRAME_COLUMNS of ``balls`` with ``fold`` and ``cell`` (int8), the ``pressure``
      flags ``live`` (bool) and ``pressure_bat`` (boolean), and the ``state`` columns ``score``,
      ``wickets`` and ``runs_needed``;
    * ``omega``: Wicket_Cost (float64, from ``m2_preds``); ``crv``: the Class_Run_Values of each
      Delivery's fold (float64, shape (n, 9));
    * ``m4``: M4 variant -> the Primary_Model's probabilities (float64, shape (n, 9)), for each
      variant loaded;
    * ``p_extras``: the Extras_Model Wide and No_Ball probabilities (float64, shape (n, 2));
    * ``means`` and ``premium``: the extras_cell_means and free_hit_premium tables, built from
      every Delivery of ``balls``;
    * ``candidate``: the M4 Primary_Model's candidate family; ``n_excluded``: the number of
      innings 1-2 Deliveries of excluded matches, which ``d`` leaves out.
    """

    d: pd.DataFrame
    omega: np.ndarray
    crv: np.ndarray
    m4: Mapping[str, np.ndarray]
    p_extras: np.ndarray
    means: pd.DataFrame
    premium: pd.DataFrame
    candidate: str
    n_excluded: int


def load_value_inputs(
    store: CheckpointStore, *, variants: Sequence[str] = VALUE_VARIANTS, m4_family: str | None = None
) -> ValueInputs:
    """The NBV inputs of the ``value`` stage, from a Run's checkpoints (module docstring, the stage's steps 1 and 2).

    ``store`` holds the complete checkpoints that the stage reads: ``balls`` (``reconstruct``),
    ``state``, the Fold_Table (``folds``), ``m2_preds`` (``remaining_runs``),
    ``m4_class_run_values`` and ``extras_preds`` (``outcome``), the Primary_Models' predictions
    (``model_choice.load_primary``, which also reads ``win_prob``) and the ``pressure`` table.
    ``variants`` names the M4 variants to load: VALUE_VARIANTS, the four views and ``actual``,
    unless told otherwise. ``run_stage`` builds its NBV from this result, so own_theta_nbv works
    on the stage's own inputs.

    ``m4_family`` loads the M4 variants of that non-failed Candidate_Model (through
    ``model_choice.load_predictions``) in place of the selected M4 Primary_Model, keeping the
    selected M3 Primary_Models; every other input is unchanged, so the only difference from the
    default is the M4 arrays. The ``value`` stage uses it to re-verify the M4 calibration-aware
    tie-break (Requirement 23, criterion 14): the own-theta NBV cell means of the displaced
    lowest-log-loss family on the stage's own Omega, Extras and Free_Hit_Premium.

    Returns a ValueInputs. Raises StageError for an M4 view that no longer keeps the opponent's
    theta, an empty pooled set (extras_cell_means, free_hit_premium) or a checkpoint file that is
    missing or changed; ValueError or TypeError for tables that do not fit together, an unknown
    variant or an ``m4_family`` the comparison table does not list or marks as failed.
    """
    # Imported here, not with the module: both modules import this one when they load.
    from superr import pressure
    from superr.models import remaining_runs

    if isinstance(variants, (str, bytes)):
        raise TypeError("variants must be a sequence of M4 variant names, not one string")
    names = tuple(dict.fromkeys(variants))
    _check_m4_views()

    # 1. Every Delivery with its fold and cell, and the Training_Folds cell tables (criteria 6, 15, 20, 21).
    state = store.load_table(STATE_STAGE, STATE_TABLE)
    frame = _delivery_frame(
        store.load_table(reconstruct.STAGE_NAME, reconstruct.BALLS_TABLE), state,
        store.load_table(folds.STAGE_NAME, folds.FOLD_TABLE),
    )  # fmt: skip
    means = extras_cell_means(frame, CELL_COLUMN)
    premium = free_hit_premium(frame, CELL_COLUMN)

    # 2. The state rows in state order, with the model outputs of each.
    row_ids = _whole(state["row"], "state column 'row'").astype(np.int64)
    rows = _aligned(frame, row_ids, _FRAME_COLUMNS, "balls")
    del frame
    fold = _whole(rows[FOLD_COLUMN], "balls column 'fold'").astype(np.int64)
    if m4_family is None:
        primary = model_choice.load_primary(store, variants=names)
    else:
        chosen = model_choice.primary_models(
            store.load_table(model_choice.STAGE_NAME, model_choice.COMPARISON_TABLE)
        )
        m3 = {number: chosen[model_choice.M3_MODELS[number]] for number in model_choice.M3_INNINGS}
        primary = model_choice.load_predictions(store, m3=m3, m4=str(m4_family), variants=names)
    if not (np.array_equal(primary.row, row_ids) and np.array_equal(primary.fold, fold)):
        raise ValueError(
            "the Primary_Models' predictions do not hold the state rows in order with the folds of the Fold_Table; pass "
            "the checkpoints of one Run"
        )
    extras_rows = _aligned(
        store.load_table(extras.STAGE_NAME, extras.PREDS_TABLE), row_ids, ("p_wide", "p_noball", FOLD_COLUMN),
        extras.PREDS_TABLE,
    )  # fmt: skip
    if not np.array_equal(_whole(extras_rows[FOLD_COLUMN], "extras_preds column 'fold'"), fold):
        raise ValueError("extras_preds gives a fold other than the Fold_Table's; pass the checkpoints of one Run")
    omega_all = _floats(
        _aligned(store.load_table(remaining_runs.STAGE_NAME, remaining_runs.PREDS_TABLE), row_ids, ("omega",),
                 remaining_runs.PREDS_TABLE)["omega"],
        "m2_preds column 'omega'",
    )  # fmt: skip
    flags = _aligned(
        store.load_table(pressure.STAGE_NAME, pressure.PRESSURE_TABLE), row_ids, ("live", "pressure_bat"),
        pressure.PRESSURE_TABLE,
    )  # fmt: skip
    crv_by_fold = outcome.load_class_run_values(store, outcome.STAGE_NAME)

    # 3. The Deliveries of the matches used for player rating: innings 1-2, no excluded match.
    keep = common.training_mask(rows)
    at = np.flatnonzero(keep)
    d = rows.iloc[at].reset_index(drop=True)
    d[CELL_COLUMN] = _whole(d[CELL_COLUMN], "the cells of the rated Deliveries").astype(np.int8)
    d["live"] = _flags(flags["live"], "pressure column 'live'", len(rows))[at]
    d["pressure_bat"] = flags["pressure_bat"].iloc[at].reset_index(drop=True).astype("boolean")
    for name in ("score", "wickets", "runs_needed"):
        d[name] = state[name].iloc[at].reset_index(drop=True)
    return ValueInputs(
        d=d,
        omega=omega_all[at],
        crv=_fold_run_values(crv_by_fold, fold[at]),
        m4={name: np.asarray(primary.m4[name])[at] for name in names},
        p_extras=np.column_stack(
            [_floats(extras_rows[name], f"extras_preds column {name!r}")[at] for name in ("p_wide", "p_noball")]
        ),
        means=means,
        premium=premium,
        candidate=str(dict(primary.candidates).get(model_choice.M4_MODEL, "")),
        n_excluded=int((~keep).sum()),
    )


def _view_probabilities(views: Any, m4: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The probabilities that each of the four M4 views reads: those of the loaded variant ``views`` names."""
    if not isinstance(views, Mapping) or set(views) != set(_M4_VIEWS):
        raise ValueError(
            f"views must map each M4 view of the NBV ({', '.join(_M4_VIEWS)}) to the variant read in its place, as "
            f"NBV_VIEWS and OWN_THETA_VIEWS do; got {views!r}"
        )
    absent = sorted({str(variant) for variant in views.values()} - set(m4))
    if absent:
        raise ValueError(
            f"views names the M4 variant(s) {', '.join(absent)}, which the inputs do not hold; load them with "
            "load_value_inputs(store, variants=...)"
        )
    return {view: m4[str(views[view])] for view in _M4_VIEWS}


def own_theta_nbv(
    inputs: ValueInputs, *, retire_out_penalty: bool, views: Mapping[str, str] = OWN_THETA_VIEWS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Both NBVs and the Dismissal_Charges with each player's own theta (criterion 17; module docstring).

    ``inputs`` is a ValueInputs (load_value_inputs) and ``retire_out_penalty`` the D14 Config
    value. ``views`` maps each of the four M4 views of the NBV formulas (``batter_view``,
    ``batter_view_fh``, ``bowler_view`` and ``bowler_view_nofh``) to the variant of ``inputs.m4``
    read in its place. OWN_THETA_VIEWS, the default, reads ``actual`` for ``batter_view`` and
    ``bowler_view`` and keeps the two Free_Hit views of the No_Ball terms (the approximation in the
    module docstring). NBV_VIEWS reads each view itself and so reproduces the stage's NBV.
    batter_nbv, bowler_nbv and dismissal_charges run on the other inputs as the stage passes them.

    Returns (value, charges): a table with the VALUE_COLUMNS of VALUE_TABLE and one with the
    DISMISSAL_CHARGE_COLUMNS of DISMISSAL_CHARGES_TABLE, for the Deliveries of ``inputs`` in their
    order. The charges do not depend on ``views``. Raises ValueError when ``views`` does not map
    the four views to loaded variants, and the errors of batter_nbv, bowler_nbv and
    dismissal_charges.
    """
    p = _view_probabilities(views, inputs.m4)
    d, omega, crv = inputs.d, inputs.omega, inputs.crv
    batter = batter_nbv(
        d, p[outcome.BATTER_VIEW], p[outcome.BATTER_VIEW_FH], crv, omega, retire_out_penalty=retire_out_penalty
    )
    bowler = bowler_nbv(
        d, p[outcome.BOWLER_VIEW], p[outcome.BOWLER_VIEW_NOFH], inputs.p_extras, crv, omega, inputs.means,
        inputs.premium,
    )  # fmt: skip
    charges = dismissal_charges(d, omega, retire_out_penalty=retire_out_penalty)
    return _value_table(d, omega, batter, bowler), charges


def own_theta_cell_means(
    inputs: ValueInputs, *, retire_out_penalty: bool, views: Mapping[str, str] = OWN_THETA_VIEWS
) -> pd.DataFrame:
    """The own-theta NBV per Delivery of each Role and innings x Phase cell (criterion 17; NBV_CELL_COLUMNS).

    ``nbv_cell_means`` of own_theta_nbv(inputs, ...), in the layout of NBV_CELLS_TABLE: one row
    per Role and cell, whose ``mean_per_delivery`` is the unweighted sum of the Role's NBV in the
    cell, plus for the batter Role the cell's Dismissal_Charges, divided by the cell's
    Batter_Deliveries or Bowler_Deliveries. The ``value`` stage saves it as
    NBV_CELLS_OWN_THETA_TABLE and writes it to NBV_CELLS_OWN_THETA_FILE. Raises the errors of
    own_theta_nbv.
    """
    return nbv_cell_means(*own_theta_nbv(inputs, retire_out_penalty=retire_out_penalty, views=views))


class M4TieBreakCells(NamedTuple):
    """The two-candidate own-theta NBV diagnostic of the M4 tie-break and its verification (criterion 14).

    ``frame`` has the M4_TIEBREAK_CELL_COLUMNS. ``selected`` is the preferred M4 Primary_Model
    family and ``displaced`` the lowest-log-loss family it displaced. ``within_band`` is whether
    the selected family's pooled log loss is within the Config band of the lowest; ``selected_pass``
    whether the selected family's |mean NBV| is within ``tolerance`` in every innings x Phase cell;
    ``displaced_breach`` whether the displaced family's |mean NBV| exceeds ``tolerance`` in at
    least one cell. The tie-break is verified when all three hold.
    """

    frame: pd.DataFrame
    selected: str
    displaced: str
    within_band: bool
    selected_pass: bool
    displaced_breach: bool
    tolerance: float


def _tiebreak_failure_reason(reason: Any) -> bool:
    """True when ``reason`` is a failure reason (mirrors ``model_choice._is_failure``)."""
    if reason is None or (not isinstance(reason, str) and pd.isna(reason)):
        return False
    return bool(str(reason).strip())


def _tiebreak_frame(
    selected_cells: pd.DataFrame, displaced_cells: pd.DataFrame, selected: str, displaced: str, tolerance: float
) -> tuple[pd.DataFrame, bool, bool]:
    """The diagnostic frame and the (selected passes all, displaced breaches one) flags."""
    sel = selected_cells.set_index(["role", "cell"])
    dis = displaced_cells.set_index(["role", "cell"])
    records: list[dict[str, Any]] = []
    selected_pass, displaced_breach = True, False
    for role in ROLES:
        for cell in CELLS:
            srow, drow = sel.loc[(role, cell)], dis.loc[(role, cell)]
            n = int(srow["n_deliveries"])
            if int(drow["n_deliveries"]) != n:
                raise ValueError(
                    f"the {selected} and {displaced} own-theta cell means count different Deliveries in the {role} "
                    f"cell {cell} ({n} vs {int(drow['n_deliveries'])}); they must share the stage's Deliveries"
                )
            smean, dmean = float(srow["mean_per_delivery"]), float(drow["mean_per_delivery"])
            s_within = bool(math.isfinite(smean) and abs(smean) <= tolerance)
            d_within = bool(math.isfinite(dmean) and abs(dmean) <= tolerance)
            if n > 0:
                selected_pass = selected_pass and s_within
                displaced_breach = displaced_breach or not d_within
            records.append(
                {
                    "role": role,
                    "innings": int(srow["innings"]),
                    "phase": str(srow["phase"]),
                    "cell": int(cell),
                    "n_deliveries": n,
                    "selected_family": selected,
                    "selected_mean_per_delivery": smean,
                    "selected_within_tol": s_within,
                    "displaced_family": displaced,
                    "displaced_mean_per_delivery": dmean,
                    "displaced_within_tol": d_within,
                    "tolerance": float(tolerance),
                }
            )
    frame = pd.DataFrame.from_records(records, columns=list(M4_TIEBREAK_CELL_COLUMNS))
    return frame.astype({"innings": np.int8, "cell": np.int8, "n_deliveries": np.int64}), selected_pass, displaced_breach


def _tiebreak_summary(tiebreak: M4TieBreakCells) -> str:
    """A short text of the own-theta NBV cell means of both families, for the Run_Log and errors."""
    rows = [
        f"{record['role']} innings {int(record['innings'])} {record['phase']} "
        f"{tiebreak.selected} {float(record['selected_mean_per_delivery']):+.4f} / "
        f"{tiebreak.displaced} {float(record['displaced_mean_per_delivery']):+.4f}"
        for record in tiebreak.frame.to_dict("records")
        if int(record["n_deliveries"]) > 0
    ]
    return f"own-theta NBV per Delivery by cell ({tiebreak.selected} / {tiebreak.displaced}): " + "; ".join(rows)


def m4_tiebreak_cells(
    store: CheckpointStore, settings: Any, *, selected_cells: pd.DataFrame, retire_out_penalty: bool, tolerance: float
) -> M4TieBreakCells | None:
    """The two-candidate own-theta NBV diagnostic of the M4 calibration-aware tie-break (Requirement 23, criterion 14).

    When the Config names an M4 preferred family (``model_choice.calibration_preferred``) that was
    selected as the M4 Primary_Model and displaced a different lowest-log-loss candidate, this
    recomputes the displaced family's own-theta NBV cell means on the stage's own inputs (only the
    M4 variant arrays differ: Omega, the Extras terms and Free_Hit_Premium are shared, through
    ``load_value_inputs(store, m4_family=...)``) and returns an M4TieBreakCells with both families'
    per-cell means and the verification flags. ``selected_cells`` is the selected family's
    ``own_theta_cell_means`` (the stage's ``own_cells``). Returns None when no M4 family is
    preferred or the tie-break did not displace the lowest-log-loss candidate.
    """
    pref = model_choice.calibration_preferences(settings).get(model_choice.M4_MODEL)
    if pref is None:
        return None
    family, band = pref
    comparison = store.load_table(model_choice.STAGE_NAME, model_choice.COMPARISON_TABLE)
    selected = str(model_choice.primary_models(comparison)[model_choice.M4_MODEL])
    part = comparison.iloc[
        np.flatnonzero(comparison["model"].astype(str).to_numpy(dtype=object) == model_choice.M4_MODEL)
    ]
    kept = {
        str(candidate): float(loss)
        for candidate, loss, reason in zip(part["candidate"], part["log_loss"], part["failure_reason"])
        if not _tiebreak_failure_reason(reason)
    }
    if selected not in kept:
        return None
    best = min(kept, key=lambda name: (kept[name], model_choice.M4_TIE_ORDER.index(name)))
    if selected != family or selected == best:
        return None  # the preferred family was not selected, or is itself the lowest: no displacement
    displaced = best
    within_band = (kept[selected] - kept[displaced]) <= band * abs(kept[displaced])
    displaced_cells = own_theta_cell_means(
        load_value_inputs(store, m4_family=displaced), retire_out_penalty=retire_out_penalty
    )
    frame, selected_pass, displaced_breach = _tiebreak_frame(
        selected_cells, displaced_cells, selected, displaced, tolerance
    )
    return M4TieBreakCells(frame, selected, displaced, within_band, selected_pass, displaced_breach, tolerance)


def run_stage(ctx: StageContext) -> None:
    """The ``value`` Pipeline_Stage (module docstring, "The ``value`` Pipeline_Stage").

    Raises StageError (exit code 5) for an empty pooled set (extras_cell_means,
    free_hit_premium), an NBV or charge that is not finite (criterion 16), an M4 variant that no
    longer keeps the opponent's theta, and a checkpoint file that is missing or changed; ValueError
    or TypeError for input tables that do not fit together or a Config value of the wrong type.
    """
    started = time.perf_counter()
    settings, store = ctx.settings, ctx.store
    penalty = _switch(settings.get(_RETIRE_OUT_PENALTY), _RETIRE_OUT_PENALTY)
    tolerance = float(settings.get(_CELL_TOLERANCE))

    # 1-3. The rated Deliveries with their model outputs, and the Training_Folds cell tables (load_value_inputs).
    inputs = load_value_inputs(store)
    d, omega, crv, probs = inputs.d, inputs.omega, inputs.crv, inputs.m4
    means, premium = inputs.means, inputs.premium

    # 4. Both NBVs, the Dismissal_Charges and the finiteness check (criteria 1-14 and 16).
    batter = batter_nbv(
        d, probs[outcome.BATTER_VIEW], probs[outcome.BATTER_VIEW_FH], crv, omega, retire_out_penalty=penalty
    )
    bowler = bowler_nbv(
        d, probs[outcome.BOWLER_VIEW], probs[outcome.BOWLER_VIEW_NOFH], inputs.p_extras, crv, omega, means, premium
    )
    charges = dismissal_charges(d, omega, retire_out_penalty=penalty)
    dismissals = dismissed_batters(d, retire_out_penalty=penalty)
    cls = _classes(d["cls"], "balls column 'cls'", len(d))
    _check_finite(cls, batter, bowler, charges)
    # The cell means of the NBV with each player's own theta, which criterion 17 checks.
    own_cells = own_theta_cell_means(inputs, retire_out_penalty=penalty)
    candidate, n_excluded = inputs.candidate, inputs.n_excluded
    del inputs, probs, crv

    # The M4 calibration-aware tie-break, re-verified on the stage's own inputs (Requirement 23,
    # criterion 14): the selected preferred family must be within the log-loss band and pass every
    # own-theta NBV cell, and the displaced lowest-log-loss family must breach at least one.
    tiebreak = m4_tiebreak_cells(
        store, settings, selected_cells=own_cells, retire_out_penalty=penalty, tolerance=tolerance
    )
    if tiebreak is not None and not (tiebreak.within_band and tiebreak.selected_pass and tiebreak.displaced_breach):
        raise StageError(
            "the M4 calibration-aware tie-break (Requirement 23, criteria 8 and 14) is not justified on the "
            f"Value_Engine's own inputs: the selected family {tiebreak.selected} is "
            f"{'within' if tiebreak.within_band else 'NOT within'} the log-loss band, "
            f"{'passes' if tiebreak.selected_pass else 'does NOT pass'} every own-theta NBV innings x Phase cell "
            f"(|mean| <= {tolerance:g}), and the displaced lowest-log-loss family {tiebreak.displaced} "
            f"{'breaches' if tiebreak.displaced_breach else 'does NOT breach'} any cell. All three must hold. "
            f"{_tiebreak_summary(tiebreak)}"
        )

    value = _value_table(d, omega, batter, bowler)
    counts = fallback_counts(d, means, premium)
    cells = nbv_cell_means(value, charges)
    cell_table = _merged_cell_table(means, premium)
    names = _delivery_names(store.load_table(_LOAD_STAGE, _BALL_BY_BALL_TABLE), d)
    matches = store.load_table(_LOAD_STAGE, _MATCHES_TABLE)
    extremes = {role: _extreme_table(role, d, value, names, matches) for role in ROLES}

    # 5. The checkpoint.
    store.save_table(STAGE_NAME, VALUE_TABLE, value)
    store.save_table(STAGE_NAME, DISMISSAL_CHARGES_TABLE, charges)
    store.save_table(STAGE_NAME, CELL_MEANS_TABLE, cell_table)
    store.save_table(STAGE_NAME, FALLBACK_COUNTS_TABLE, counts)
    store.save_table(STAGE_NAME, NBV_CELLS_TABLE, cells)
    store.save_table(STAGE_NAME, NBV_CELLS_OWN_THETA_TABLE, own_cells)

    # 6. The files in outputs/value/.
    folder = settings.path("outputs") / VALUE_FOLDER
    files = [
        write_fallback_counts(folder, counts),
        _write_csv(folder / CELL_MEANS_FILE, cell_table),
        _write_csv(folder / NBV_CELLS_FILE, cells),
        _write_csv(folder / NBV_CELLS_OWN_THETA_FILE, own_cells),
        *(_write_csv(folder / EXTREMES_FILES[role], extremes[role]) for role in ROLES),
    ]
    if tiebreak is not None:  # the two-candidate M4 tie-break diagnostic (Requirement 23, criterion 14)
        files.append(_write_csv(folder / M4_TIEBREAK_CELLS_FILE, tiebreak.frame))

    # 7. The data-derived constants for the report.
    wide = cls == _WIDE_CODE
    non_striker = charges["non_striker"].to_numpy(dtype=bool)
    retired = _flags(d["retire_out"], "balls column 'retire_out'", len(d))
    charge_counts = {
        "total": int(len(charges)),
        "non_striker": int(non_striker.sum()),
        "striker_off_wide": int((~non_striker & (charges["cls"].to_numpy() == _WIDE_CODE)).sum()),
        "on_wides": int((charges["cls"].to_numpy() == _WIDE_CODE).sum()),
        "pressure": int(charges["pressure"].to_numpy(dtype=bool).sum()),
    }
    cell_means_by_role = {
        role: {
            f"innings{int(record['innings'])}_{record['phase']}": float(record["mean_per_delivery"])
            for record in cells.to_dict("records")
            if record["role"] == role and record["n_deliveries"] > 0
        }
        for role in ROLES
    }
    record = ctx.constants.record
    record(STRIKER_FALLBACK_CONSTANT, int(dismissals.fallback.sum()), _COMPONENT)
    record(POOLED_FALLBACK_CONSTANT, dict(zip(counts["quantity"].tolist(), counts["n_bowler_deliveries"].tolist())),
           _COMPONENT)  # fmt: skip
    record(DISMISSAL_CHARGES_CONSTANT, charge_counts, _COMPONENT)
    record(RETIRE_OUT_CONSTANT, 0 if penalty else int(retired.sum()), _COMPONENT)
    record(NBV_CELL_MEANS_CONSTANT, cell_means_by_role, _COMPONENT)

    _log_stage(
        ctx, d=d, value=value, charges=charges, dismissals=dismissals, counts=counts, cells=cells,
        own_cells=own_cells, wide=wide, retired=retired, penalty=penalty, tolerance=tolerance, candidate=candidate,
        n_excluded=n_excluded, tiebreak=tiebreak, files=files, seconds=time.perf_counter() - started,
    )  # fmt: skip


def _summary(values: np.ndarray) -> str:
    return (
        f"n {len(values):,}, mean {values.mean():+.4f}, sd {values.std(ddof=1):.4f}, min {values.min():+.3f}, "
        f"max {values.max():+.3f}"
        if len(values) > 1
        else f"n {len(values):,}"
    )


def _code_counts(codes: np.ndarray, labels: np.ndarray) -> str:
    if not len(codes):
        return "none"
    table = pd.DataFrame({"code": codes, "label": labels}).value_counts().sort_index()
    return ", ".join(f"code {int(code)} {label} {int(n):,}" for (code, label), n in table.items())


def _cell_texts(cells: pd.DataFrame, tolerance: float) -> tuple[list[str], list[str]]:
    """Each row of an nbv_cells table as text, and the texts of the rows whose mean lies outside +/-tolerance."""
    parts, outside = [], []
    for record in cells.to_dict("records"):
        mean = float(record["mean_per_delivery"])
        text = (
            f"{record['role']} innings {int(record['innings'])} {record['phase']} {mean:+.4f} "
            f"({int(record['n_deliveries']):,})"
        )
        parts.append(text)
        if not (math.isfinite(mean) and abs(mean) <= tolerance):
            outside.append(text)
    return parts, outside


def _log_stage(
    ctx: StageContext,
    *,
    d: pd.DataFrame,
    value: pd.DataFrame,
    charges: pd.DataFrame,
    dismissals: Dismissals,
    counts: pd.DataFrame,
    cells: pd.DataFrame,
    own_cells: pd.DataFrame,
    wide: np.ndarray,
    retired: np.ndarray,
    penalty: bool,
    tolerance: float,
    candidate: str,
    n_excluded: int,
    tiebreak: M4TieBreakCells | None,
    files: Sequence[Path],
    seconds: float,
) -> None:
    log = ctx.log
    n_matches = len(np.unique(d["match_id"].to_numpy()))
    rule = "charged as a batter dismissal (D14 alternative)" if penalty else "with no Omega and no Dismissal_Charge (D14)"
    log.info(
        f"Value_Engine inputs: the M4 Primary_Model {candidate} (batter_view, batter_view_fh, bowler_view and "
        "bowler_view_nofh: the Role player's theta at the fold's Average_Skill, the opponent's actual theta, Requirement "
        "2, criterion 4; its actual variant for the own-theta check of criterion 17), the Extras_Model p_wide and "
        "p_noball, Omega of m2_preds and each fold's Class_Run_Values; "
        f"Retire Out {rule}; "
        f"{len(d):,} Deliveries of the {n_matches:,} matches used for player rating ({int((~wide).sum()):,} "
        f"Batter_Deliveries, {len(d):,} Bowler_Deliveries; {n_excluded:,} Deliveries of excluded matches left out)",
    )
    nbv_bat = value["nbv_bat"].to_numpy(dtype=np.float64)[~wide]
    nbv_bowl = value["nbv_bowl"].to_numpy(dtype=np.float64)
    charge = charges["charge"].to_numpy(dtype=np.float64)
    log.info(
        f"NBV in runs per Delivery, all finite (Requirement 15, criterion 16): batter {_summary(nbv_bat)}; bowler "
        f"{_summary(nbv_bowl)}; Dismissal_Charges {len(charge):,} totalling {charge.sum():+.2f} runs",
    )
    parts, _ = _cell_texts(cells, tolerance)
    log.info(
        "NBV per Delivery by innings x Phase cell against an Average_Skill player, the batter Role with its "
        f"Dismissal_Charges (information; {VALUE_FOLDER}/{NBV_CELLS_FILE}): " + "; ".join(parts),
    )
    parts, outside = _cell_texts(own_cells, tolerance)
    log.info(
        "NBV per Delivery by innings x Phase cell with each player's own theta (the M4 actual variant in place of "
        "batter_view and bowler_view; the No_Ball terms keep the Free_Hit views), the batter Role with its "
        f"Dismissal_Charges (Requirement 15, criterion 17; Test_Suite band +/-{tolerance:g}; "
        f"{VALUE_FOLDER}/{NBV_CELLS_OWN_THETA_FILE}): " + "; ".join(parts),
    )
    if outside:
        log.warning(
            f"the own-theta NBV per Delivery lies outside the Test_Suite band +/-{tolerance:g} runs in: "
            f"{'; '.join(outside)}"
        )
    if tiebreak is not None:
        log.info(
            f"M4 calibration-aware tie-break verified on the Value_Engine's own inputs (Requirement 23, criterion 14; "
            f"{VALUE_FOLDER}/{M4_TIEBREAK_CELLS_FILE}): the selected M4 Primary_Model {tiebreak.selected} is "
            f"{'within' if tiebreak.within_band else 'NOT within'} the log-loss band and "
            f"{'passes every' if tiebreak.selected_pass else 'does NOT pass every'} own-theta NBV innings x Phase cell "
            f"(Test_Suite band +/-{tolerance:g}), while the displaced lowest-log-loss family {tiebreak.displaced} "
            f"{'breaches at least one cell' if tiebreak.displaced_breach else 'breaches no cell'}; "
            f"{_tiebreak_summary(tiebreak)}"
        )
    listed = ", ".join(
        f"{quantity} {int(n):,}" for quantity, n in zip(counts["quantity"].tolist(), counts["n_bowler_deliveries"].tolist())
    )
    if (counts["n_bowler_deliveries"].to_numpy() > 0).any():
        log.warning(
            "Bowler_Deliveries whose bowler NBV used a pooled Training_Folds value (Requirement 15, criteria 20-22): "
            f"{listed}; see {VALUE_FOLDER}/{FALLBACK_COUNTS_FILE}",
        )
    else:
        log.info(f"No Bowler_Delivery used a pooled Training_Folds value (Requirement 15, criterion 22): {listed}")
    counted = _flags(d["counted_wicket"], "balls column 'counted_wicket'", len(d))
    n_fallback = int(dismissals.fallback.sum())
    message = (
        f"Dismissed batters (Requirement 15, criteria 9 and 10): {int(dismissals.charged.sum()):,} charged Counted_Wickets "
        f"of {int(counted.sum()):,}; {n_fallback:,} with a blank or unmatched dismissed_player_id fell back to the striker"
    )
    (log.warning if n_fallback else log.info)(message)
    codes = _whole(d["dismissal_type_id"], "balls column 'dismissal_type_id'", len(d), missing=True)
    labels = d["dismissal_type"].astype(object).where(d["dismissal_type"].notna(), "").to_numpy(dtype=object)
    where = pd.Index(d["row"].to_numpy(dtype=np.int64)).get_indexer(charges["row"].to_numpy(dtype=np.int64))
    non_striker = charges["non_striker"].to_numpy(dtype=bool)
    on_wide = charges["cls"].to_numpy() == _WIDE_CODE
    pressure_flag = charges["pressure"].to_numpy(dtype=bool)
    no_source = charges["flag_row"].isna().to_numpy(dtype=bool)
    dead_wides = on_wide & ~d["live"].to_numpy(dtype=bool)[where]
    log.info(
        f"Dismissal_Charges (-Omega, Requirement 15, criterion 11): {len(charges):,}, of which {int(non_striker.sum()):,} "
        f"non-striker dismissals ({_code_counts(codes[where][non_striker], labels[where][non_striker])}; "
        f"{int((non_striker & on_wide).sum()):,} off a Wide) and {int((~non_striker & on_wide).sum()):,} striker "
        f"dismissals off a Wide ({_code_counts(codes[where][~non_striker], labels[where][~non_striker])}); "
        f"{int(pressure_flag.sum()):,} pressure, {int((~pressure_flag).sum()):,} ordinary. The {int(on_wide.sum()):,} "
        "charges on Wides take the batter-Role flag of the latest earlier Batter_Delivery of the innings "
        f"({int((pressure_flag & on_wide).sum()):,} pressure; {int(dead_wides.sum()):,} on Dead_Balls and "
        f"{int((no_source & on_wide).sum()):,} without an earlier Batter_Delivery set ordinary)",
    )
    log.info(
        f"Retire Out (code 9, Requirement 15, criterion 12): {int(retired.sum()):,} Deliveries, "
        + ("charged as batter dismissals (D14 alternative)" if penalty else "charged neither Omega nor a Dismissal_Charge"),
    )
    shown = ", ".join(_shown_path(path, ctx.root) for path in files)
    log.info(f"value stage wrote {shown} and its checkpoint in {seconds:.1f} s")
