"""Skill_Model M1: two sparse ridge regressions on the Legal_Balls kept for model training, solved
from per-fold sufficient statistics, the M1-derived quantities of every Delivery, and the
``skill`` Pipeline_Stage (Requirement 8; Requirement 7, criteria 5-7, 11 and 12; Requirement 2,
criterion 3; design DD-6 and sections "Skill_Model M1 (``skill.py``)" and "Fold sandboxes and
leakage control").

The functions up to ``clipped_log_loss`` are the ridge itself (task 9.1). The second part of the
module ("M1-derived quantities", "The skill Pipeline_Stage" below) computes Typical_Top7_Sum,
RBQ, Team_Strength_Gap and Match_Day_Conditions and runs the stage. Every function receives the
Config values it needs as arguments: the penalty grid ``models.m1.ridge_grid``, the probability
bounds ``models.m1.prob_clip``, the Wickets_Bucket boundaries D17
(``design_choices.D17_wicket_buckets``) and, in the stage, the number of folds D15
(``design_choices.D15_n_folds``). No function reads ``config.py`` or has a default value for a
Config setting. M1 has no random step, so it uses no seed.

Rows and targets (``legal_rows``; criterion 1)
    The Training_Rows of M1 are the Legal_Balls that ``models.common.legal_training_mask`` keeps:
    innings 1 or 2, no Super_Over_Innings, no match excluded under the Score_Tolerance rule, and
    ``cls`` Legal_Ball (181,591 Legal_Balls on the delivered data). ``legal_rows`` takes them from
    the ``balls`` table of the ``reconstruct`` checkpoint, adds the ``cell`` and ``wickets``
    columns of the ``state`` table by ``row`` and names the two targets: ``bat_runs`` (the
    Legal_Ball's ``runs``) and ``counted_wicket`` (true when the Legal_Ball carries a
    Counted_Wicket, whoever is out, Retire Out included). With ``training_only=False`` it returns
    every Legal_Ball of innings 1 and 2, excluded matches included, for predictions such as
    Match_Day_Conditions.

The design matrix (``build_encoders``, ``design_matrix``; criteria 2 and 6)
    One indicator column per category of seven groups, in GROUPS order:

    ================ ===================================== ========================================
    Group            Categories                            Read from
    ================ ===================================== ========================================
    striker          each ``striker_id``                   ``striker_id``
    bowler           each ``bowler_id``                    ``bowler_id``
    environment      each Competition_Type x Season        ``comp_type`` and ``season``
    venue            each ``venue_id``                     ``venue_id``
    country          each ``venue_country``                ``venue_country``
    cell             the 6 innings x Phase cells           ``cell`` = 3 x (innings - 1) + phase
    wickets_bucket   each D17 range of wickets fallen      ``wickets`` (before the Delivery)
    ================ ===================================== ========================================

    A Season is the competition's ``season`` text, so each distinct string (``2022``,
    ``2022/23`` ...) is its own Season within each Competition_Type (criterion 6). The
    ``OneHotEncoders`` of ``build_encoders`` fix the categories once, from the full entity
    universe: normally the whole ``balls`` table, which gives 888 strikers, 647 bowlers, 15
    environments, 97 venues and 17 countries on the delivered data, 1,673 columns with the 6
    cells and 3 buckets. Every fold and every fit therefore uses the same column positions, and a
    category without training rows in a fit is a column of zeros there. ``design_matrix`` returns
    a ``scipy.sparse.csr_matrix`` of 0/1 floats with one row per row of its frame.

Missing values
    ``striker_id``, ``bowler_id``, ``comp_type``, ``cell`` and ``wickets`` are never missing in the
    checkpoint tables, and a missing value there raises ValueError. ``season``, ``venue_id`` and
    ``venue_country`` come from nullable columns. A missing value there gets no indicator in its
    group (the row is all zeros in that group), so its effect is 0, as for a category without
    training Legal_Balls (criterion 5). ``build_encoders`` adds no category for a missing value,
    and it raises DataError when a venue has more than one ``venue_country`` (a missing value
    counts as one), because that venue's Venue_Effect would be ambiguous. On the checkpoints of
    14 November 2026 none of these columns has a missing value on any Delivery and every venue
    has one country, so the rule does not apply to the delivered data.

Sufficient statistics (``fold_stats``)
    For each fold f with rows, a ``FoldStats`` holds G = X_f'X_f (dense, p x p), X_f'y for both
    targets, the column sums s (the training Legal_Balls of each category), the row count n, the
    sums of squares y'y and the sums of y. A fit only adds up the statistics of its folds.

Ridge from the statistics (``ridge_path``, ``fit_ridge``; criteria 3, 5 and 7)
    For a training set T, G_T, s, n, Xty and ybar are summed over the folds of T. A column is
    active when the fit may use it (every column for M1; every column except the striker and
    bowler ones for the baseline) and its sum over T is above 0. On the active columns, the
    centred Gram matrix G_c = G_T - s s'/n and b_c = Xty - s ybar give, through one
    eigendecomposition G_c = Q diag(L) Q', the coefficients beta(lambda) = Q diag(1/(L + lambda))
    Q'b_c for every penalty of the grid and both targets, and the unpenalised intercept
    ybar - xbar'beta(lambda). This minimises ||y - b0 - X beta||^2 + lambda ||beta||^2, as
    scikit-learn ``Ridge(alpha=lambda)`` with an intercept does. Every inactive column gets a
    coefficient of exactly 0, so a batter, bowler, venue, venue country or Competition_Type x
    Season without training Legal_Balls has runs and dismissal effects of exactly 0 (criterion
    5). Within a group whose rows all carry one indicator, the penalty centres the active effects
    on 0 (their unweighted sum is 0), so an effect of 0 is that of an average member.

Penalty choice (criterion 3)
    For each penalty of the grid, each fold f of T is predicted by the ridge fitted on T minus f.
    The cross-validated MSE is row-pooled: the squared errors summed over every held-out row of
    the folds of T, divided by the rows of T, so each Legal_Ball counts once (it is not a mean of
    per-fold means). The squared error of fold f comes from its statistics, without predicting a
    row: SSE_f = yy_f - 2 b0 sum(y_f) - 2 beta'Xty_f + n_f b0^2 + 2 b0 beta's_f + beta'G_f beta.
    Only the folds of T take part. Each target picks its own penalty (``penalty_runs``,
    ``penalty_wkt``); the dismissal target uses the squared error of its unclipped linear
    prediction. The lowest MSE wins. When several penalties share exactly the lowest value, the
    largest of them wins (the most shrinkage), so the choice is deterministic.

Reusing eigendecompositions (``RidgeCache``)
    ``fit_ridge`` on T needs the ridge of T and of every T minus f. The 26 training sets of M1
    (the all-fold set behind the top-theta lists, 5 outer sets and 20 inner fits, which share 10
    distinct 3-fold sets) and their cross-validation subsets are only 26 distinct fold sets: one
    5-fold, five 4-fold, ten 3-fold and ten 2-fold sets. A RidgeCache passed to every call
    decomposes each (fold set, column subset, grid) once. It keeps the coefficient paths of both
    targets (2 x p x grid floats, about 0.25 MB per entry), not the eigenvectors (about 22 MB).
    One cache serves one ``fold_stats`` result.

Skill and context (``thetas``, ``venue_effects``, ``scoring_environments``; criteria 4-7)
    With m the training mean bat runs per Legal_Ball and R the training runs per Counted_Wicket
    (bat runs divided by Counted_Wickets, both over the Legal_Balls of T):

    * theta_bat = m + striker runs effect - striker dismissal effect x R, in runs per Legal_Ball;
    * theta_bowl = -bowler runs effect + bowler dismissal effect x R, in runs saved per
      Legal_Ball (higher is a stronger bowler).

    A player with no training Legal_Ball in a Role has zero effects in it, so theta_bat = m and
    theta_bowl = 0 exactly. That is not Average_Skill (glossary), the theta of a typical player:
    the mean theta_bat of the strikers of the fit's training Legal_Balls, weighted by Legal_Balls
    faced, and the mean theta_bowl of their bowlers, weighted by Legal_Balls bowled
    (``models.outcome.average_skill``, from ``bat_balls`` and ``bowl_balls``). ``thetas``
    covers every player it is given; ``player_universe`` lists the players of criterion 4.
    Scoring_Environment is the Competition_Type x Season effect, and Venue_Effect is the venue's
    effect plus its country's effect. Both come from the runs regression, in bat runs per
    Legal_Ball (the unit of theta_bat). A venue without training Legal_Balls gets exactly its
    country effect (criterion 7). The dismissal-regression counterparts, in Counted_Wickets per
    Legal_Ball, are given beside them as ``scoring_env_wkt`` and ``venue_effect_wkt``.

Baseline and metrics (``fit_baseline``, ``predict``, ``out_of_fold_predictions``,
``pooled_rmse``, ``clipped_log_loss``; criterion 8)
    The baseline is the same ridge without the striker and bowler columns: the same folds, grid
    and penalty search, from the same statistics, with the player columns kept inactive. Its
    coefficients are full length, with zeros in the player columns, so ``predict`` takes the same
    design matrix. ``out_of_fold_predictions`` predicts each row from the fit given for its fold
    and refuses a fit that trained on that fold. ``pooled_rmse`` and ``clipped_log_loss`` pool
    every row given; ``clipped_log_loss`` clips the dismissal predictions to
    ``models.m1.prob_clip`` before ``models.common.pooled_log_loss``.

Usage of the ridge::

    encoders = skill.build_encoders(balls, wicket_buckets=settings.get("design_choices.D17_wicket_buckets"))
    train = skill.legal_rows(balls, state, training_only=True)
    X = skill.design_matrix(train, encoders)
    row_fold = folds.fold_of(train["match_id"], fold_table)
    stats = skill.fold_stats(X, train[skill.RUNS_TARGET], train[skill.WICKET_TARGET], row_fold)
    grid, cache = settings.get("models.m1.ridge_grid"), skill.RidgeCache()
    fit = skill.fit_ridge(stats, spec.train_folds, grid, cache=cache)       # one FitSpec of the plan
    base = skill.fit_baseline(stats, spec.train_folds, grid, encoders, cache=cache)
    theta = skill.thetas(fit, skill.player_universe(balls, xi), encoders)
    runs_hat, wicket_hat = skill.predict(fit, X_rows)                        # rows the fit serves

M1-derived quantities (``top7_sums``, ``typical_top7_sum``, ``rbq``, ``team_strength_gap``,
``match_day_conditions``, ``view_quantities``; criteria 12-16 and Requirement 2, criterion 3)
    Each function takes the theta values of one fit as a theta table (``player_id``,
    ``theta_bat``, ``theta_bowl``, as ``thetas`` returns it) and the ``playing_eleven`` table of
    the ``load`` checkpoint (``xi``). A side's players are its ``xi`` rows with ``is_played`` true
    (a missing ``is_played`` counts as false). A player listed in several such rows of one side
    counts once, with his lowest ``batting_order``, as in Batters_To_Come (``state.batters_to_come``).

    * Typical_Top7_Sum (criterion 12): for each team-innings, the summed theta_bat of the batting
      side's players whose ``batting_order`` is 1 to TOP_POSITIONS (7); the median of these sums
      over the fit's training team-innings (innings 1 and 2 with at least one training Legal_Ball
      in the fit's training folds). A team-innings whose batting side has no such player has no
      sum and is left out of the median; on the delivered data every one has one.
    * RBQ (criterion 13): (sum of theta_bat over Batters_To_Come + the non-striker's theta_bat) /
      Typical_Top7_Sum of the same fit. For the Post_Wicket_State (glossary), the next batter is
      the Batters_To_Come player with the lowest ``batting_order`` (slot ``btc_1``):
      ``theta_next_batter`` is his theta_bat and ``rbq_post_wicket`` the RBQ with him removed
      from Batters_To_Come (the non-striker stays). When Batters_To_Come is empty,
      ``theta_next_batter`` is missing and ``rbq_post_wicket`` equals ``rbq`` (no one to remove).
      When the batting side has no ``is_played`` row (the state's Batters_To_Come columns are
      missing, Requirement 6, criterion 4), RBQ, ``rbq_post_wicket`` and ``theta_next_batter``
      are missing (NaN), never 0; so is RBQ when ``non_striker_id`` is missing.
    * Team_Strength_Gap (criterion 14; Requirement 2, criterion 3): Team_Strength is
      PLAYERS_PER_SIDE (11, the players a side fields) x the mean of theta_bat + theta_bowl over a
      side's ``is_played`` players in the match, from the fit's theta values only; no team-level
      rating enters. The mean puts a side that lists an Impact Player (12 ``is_played`` players)
      on the same 11-player scale as a side that lists 11, so a different number of listed
      players does not shift the gap. The gap of a Delivery is the batting side's Team_Strength
      minus the fielding side's, both looked up from one per-side table, so the innings-2 value
      is exactly minus the innings-1 value (floating-point subtraction is antisymmetric). It is
      missing when either side has no ``is_played`` row.
    * Match_Day_Conditions (criteria 15 and 16): 6 x (sum of bat runs - sum of M1-predicted bat
      runs) / Legal_Balls, over the Legal_Balls of the same match (innings 1 and 2) strictly
      before the Delivery in the Requirement 5 order, with the predictions of the fit that
      supplies the Delivery's theta values; 0 when no Legal_Ball comes before. The sums are
      running sums within each match, so no roundoff carries over from other matches.
    * Scoring_Environment and Venue_Effect come from the runs regression, in bat runs per
      Legal_Ball, as in ``scoring_environments`` and ``venue_effects``: the row's design-matrix
      indicators times the fit's coefficients of those groups, so a missing season, venue or
      country contributes 0. ``m1_pred_runs`` is the fit's predicted bat runs for the Delivery
      (striker, bowler and context), also on Wides and No_Balls.

    ``view_quantities(fit, encoders, top7, rows, xi)`` gives all VIEW_COLUMNS of a set of
    Deliveries from one fit; ``rows`` is a ``view_rows`` frame that holds every innings 1-2
    Delivery of each of its matches, so that Match_Day_Conditions sees the earlier Legal_Balls.

The ``skill`` Pipeline_Stage (``run_stage``)
    Reads ``balls`` (``reconstruct``), ``state`` (``state``), ``playing_eleven`` and
    ``ball_by_ball`` (``load``, for the players' names) and the Fold_Table (``folds``). Then:

    1. builds the encoders from all of ``balls``, the design matrix of the 181,591 training
       Legal_Balls and that of every ``state`` row (all innings 1-2 Deliveries), and the fold
       statistics;
    2. registers every fit with the FitRegistry (``ctx.registry``) before any fit runs: the 25
       fits of ``folds.sandbox_plan(D15)`` as model MODEL with the ids of
       ``views.view_plan(MODEL, plan)`` (``m1_outer_k<k>``, ``m1_inner_k<k>_j<j>``), the all-fold
       fit ALL_FOLDS_FIT_ID (no excluded fold, no predicted match; Requirement 8, criterion 17),
       and the 5 outer baseline fits as model BASELINE_MODEL. The training matches of a fit are
       the matches of its training Legal_Balls; the predicted matches are the matches of the rows
       it serves: every Delivery of its ``predicted_fold`` for an M1 fit, the fold-k training
       Legal_Balls (the held-out metric rows) for a baseline fit;
    3. fits the 16 distinct training sets once each (inner fits (k, j) and (j, k) train on the
       same folds and share one ridge, registered under both ids) and the 5 baselines, sharing
       one RidgeCache;
    4. evaluates view_quantities for each of the 25 fits on the rows it serves, builds the M1
       columns of views V_1 to V_5 (``views.build_view``; a row in fold i of V_k takes its values
       from M1(-{i, k}), a row in fold k from M1(-k)), checks each with ``views.check_view`` and
       saves it as ``view_<k>``;
    5. writes ``outputs/models/m1/``: METRICS_FILE (pooled Out_Of_Fold RMSE of bat runs and
       clipped log loss of dismissals for M1 and the baseline, from the outer fits, with a
       warning when M1 is not lower; the batters and bowlers with training Legal_Balls in both
       Pools, IPL and international, with a warning when either count is 0, and each Pool's own
       counts) and the top-20 lists TOP_BATTERS_FILE and TOP_BOWLERS_FILE from the all-fold fit;
    6. saves THETA_TABLE and FIT_SUMMARY_TABLE, the fit registry (``registry.save``), and records
       the all-fold penalties and Typical_Top7_Sum, and the same values of every sandbox fit, as
       data-derived Method_Constants of the Skill_Model.

Reading the stage's outputs (the ``remaining_runs`` stage and later)
    ``store.load_table("skill", views.view_table(k))`` gives V_k: one row per ``state`` row, in
    the ``state`` order, with ``row``, the VIEW_COLUMNS (float64) and ``m1_fit`` (str). The fold
    of each row comes from the Fold_Table, for example ``folds.fold_of(match_ids, fold_table)``
    with each row's ``match_id`` from ``balls``. The fits named in ``m1_fit`` are in
    ``ctx.registry``: registered in the same Run, or reloaded by the runner from this stage's
    ``fit_registry`` under ``--from-stage``. THETA_TABLE holds every fit's theta of every
    player of ``player_universe`` (THETA_TABLE_COLUMNS), keyed by the registered fit id, and
    FIT_SUMMARY_TABLE one row per M1 fit (FIT_SUMMARY_COLUMNS), including ``mean_runs``, the
    fit's training mean bat runs per Legal_Ball. The Average_Skill of a fit is the ball-weighted
    mean theta_bat of the strikers and theta_bowl of the bowlers over the fit's training
    Legal_Balls: the means of THETA_TABLE's ``theta_bat`` weighted by ``bat_balls`` and
    ``theta_bowl`` weighted by ``bowl_balls`` (``models.outcome.average_skill``).
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import scipy.linalg
from scipy import sparse

from superr import folds, leakage, reconstruct, views
from superr.errors import DataError
from superr.models.common import legal_training_mask, pooled_log_loss
from superr.reconstruct import DeliveryClass
from superr.state import BALLS_PER_OVER, BTC_SLOT_COLUMNS, NO_PLAYER, STATE_INNINGS, STATE_TABLE, Phase
from superr.state import STAGE_NAME as STATE_STAGE

if TYPE_CHECKING:
    from superr.pipeline import StageContext

__all__ = [
    "ALL_FOLDS_FIT_ID",
    "BASELINE_MODEL",
    "BOWLER",
    "CELL",
    "COUNTRY",
    "DESIGN_INPUT_COLUMNS",
    "ENVIRONMENT",
    "ENVIRONMENT_COLUMNS",
    "FIT_SUMMARY_COLUMNS",
    "FIT_SUMMARY_TABLE",
    "GROUPS",
    "LEGAL_ROW_COLUMNS",
    "M1_FOLDER",
    "METRICS_FILE",
    "MODEL",
    "N_CELLS",
    "PLAYERS_PER_SIDE",
    "PLAYER_GROUPS",
    "RBQ_COLUMNS",
    "RUNS_TARGET",
    "STAGE_NAME",
    "STRIKER",
    "THETA_COLUMNS",
    "THETA_TABLE",
    "THETA_TABLE_COLUMNS",
    "TOP_BATTERS_FILE",
    "TOP_BOWLERS_FILE",
    "TOP_LIST_SIZE",
    "TOP_POSITIONS",
    "UNIVERSE_COLUMNS",
    "VENUE",
    "VENUE_COLUMNS",
    "VIEW_COLUMNS",
    "VIEW_INPUT_COLUMNS",
    "WICKETS_BUCKET",
    "WICKET_TARGET",
    "FoldStats",
    "OneHotEncoders",
    "RidgeCache",
    "RidgeFit",
    "RidgePath",
    "build_encoders",
    "clipped_log_loss",
    "design_matrix",
    "fit_baseline",
    "fit_ridge",
    "fold_stats",
    "legal_rows",
    "match_day_conditions",
    "out_of_fold_predictions",
    "player_universe",
    "pooled_rmse",
    "predict",
    "rbq",
    "ridge_path",
    "run_stage",
    "scoring_environments",
    "team_strength_gap",
    "thetas",
    "top7_sums",
    "typical_top7_sum",
    "venue_effects",
    "view_quantities",
    "view_rows",
]


# ------------------------------------------------------------------------------------- constants

#: The indicator groups of the design matrix, in column order (Requirement 8, criterion 2).
STRIKER = "striker"
BOWLER = "bowler"
ENVIRONMENT = "environment"  # Competition_Type x Season
VENUE = "venue"
COUNTRY = "country"  # venue country
CELL = "cell"  # innings x Phase
WICKETS_BUCKET = "wickets_bucket"
GROUPS: tuple[str, ...] = (STRIKER, BOWLER, ENVIRONMENT, VENUE, COUNTRY, CELL, WICKETS_BUCKET)

#: The groups that the baseline leaves out (criterion 8).
PLAYER_GROUPS: tuple[str, ...] = (STRIKER, BOWLER)

#: Innings x Phase cells: ``cell`` = 3 x (innings - 1) + phase (``state.cell_codes``), 0 to 5.
N_CELLS = len(STATE_INNINGS) * len(Phase)

#: Columns of the universe frame that build_encoders reads.
UNIVERSE_COLUMNS: tuple[str, ...] = ("striker_id", "bowler_id", "comp_type", "season", "venue_id", "venue_country")

#: Columns of a Legal_Ball frame that design_matrix reads.
DESIGN_INPUT_COLUMNS: tuple[str, ...] = (*UNIVERSE_COLUMNS, "cell", "wickets")

#: The target columns of legal_rows: bat runs, and the Counted_Wicket flag (criterion 1).
RUNS_TARGET = "bat_runs"
WICKET_TARGET = "counted_wicket"

#: Columns of the legal_rows frame, in order.
LEGAL_ROW_COLUMNS: tuple[str, ...] = ("row", "match_id", "innings", *DESIGN_INPUT_COLUMNS, RUNS_TARGET, WICKET_TARGET)

#: Columns of the thetas, venue_effects and scoring_environments tables.
THETA_COLUMNS: tuple[str, ...] = (
    "player_id", "theta_bat", "theta_bowl", "bat_balls", "bowl_balls",
    "bat_runs_effect", "bat_wkt_effect", "bowl_runs_effect", "bowl_wkt_effect",
)  # fmt: skip
VENUE_COLUMNS: tuple[str, ...] = ("venue_id", "venue_country", "venue_effect", "venue_effect_wkt", "venue_balls")
ENVIRONMENT_COLUMNS: tuple[str, ...] = ("comp_type", "season", "scoring_env", "scoring_env_wkt", "env_balls")

_SHOWN = 5  # example values named in one error message
_ENV_NAMES = ["comp_type", "season"]


# ---------------------------------------------------------------------------------- shared helpers

def _rows(count: int) -> str:
    return f"{count:,} row" if count == 1 else f"{count:,} rows"


def _rows_have(count: int) -> str:
    return "1 row has" if count == 1 else f"{count:,} rows have"


def _examples(values: Iterable[Any]) -> str:
    shown = list(values)[:_SHOWN]
    return ", ".join(str(value) for value in shown)


def _require(frame: Any, columns: Sequence[str], what: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{what} must be a DataFrame; got {type(frame).__name__}")
    absent = [column for column in columns if column not in frame.columns]
    if absent:
        raise ValueError(f"{what} lacks the column(s) {', '.join(absent)}")


def _readonly(array: np.ndarray) -> np.ndarray:
    """A read-only view of ``array``; the array itself stays as it was."""
    view = array.view()
    view.flags.writeable = False
    return view


def _whole(frame: pd.DataFrame, column: str, *, nullable: bool, what: str) -> tuple[np.ndarray, np.ndarray]:
    """``column`` as int64 values (0 where missing) and its missing-value mask.

    Raises TypeError for a non-numeric or boolean column, and ValueError for a value that is not a
    whole number, or for a missing value when ``nullable`` is false.
    """
    series = frame[column]
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype) or not pd.api.types.is_numeric_dtype(dtype):
        raise TypeError(f"{what} column {column!r} must hold whole numbers; got dtype {dtype}")
    missing = series.isna().to_numpy(dtype=bool)
    if missing.any() and not nullable:
        raise ValueError(
            f"{what} column {column!r} is missing in {_rows(int(missing.sum()))}; every row needs a value"
        )
    if pd.api.types.is_integer_dtype(dtype):
        return series.to_numpy(dtype=np.int64, na_value=0), missing
    floats = series.to_numpy(dtype=np.float64, na_value=np.nan)
    present = floats[~missing]
    bad = ~(np.isfinite(present) & (np.floor(present) == present))
    if bad.any():
        raise ValueError(
            f"{what} column {column!r} holds {int(bad.sum()):,} value(s) that are not whole numbers "
            f"(for example {_examples(present[bad])})"
        )
    return np.where(missing, 0.0, floats).astype(np.int64), missing


def _text(frame: pd.DataFrame, column: str, *, nullable: bool, what: str) -> tuple[np.ndarray, np.ndarray]:
    """``column`` as an object array of str (None where missing) and its missing-value mask.

    Raises TypeError when the column does not hold text, and ValueError for a missing value when
    ``nullable`` is false.
    """
    series = frame[column]
    dtype = series.dtype
    if not (pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype)):
        raise TypeError(f"{what} column {column!r} must hold text; got dtype {dtype}")
    missing = series.isna().to_numpy(dtype=bool)
    if missing.any() and not nullable:
        raise ValueError(
            f"{what} column {column!r} is missing in {_rows(int(missing.sum()))}; every row needs a value"
        )
    values = series.to_numpy(dtype=object, na_value=None)
    present = values[~missing]
    if present.size and pd.api.types.infer_dtype(present, skipna=True) != "string":
        raise TypeError(f"{what} column {column!r} must hold text; it holds other values as well")
    return values, missing


def _float_array(values: Any, what: str, ndim: int) -> np.ndarray:
    """``values`` as a float64 array of ``ndim`` dimensions with finite values only."""
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{what} must be an array of numbers; got {type(values).__name__}")
    try:
        if isinstance(values, (pd.Series, pd.Index)):
            array = values.to_numpy(dtype=np.float64, na_value=np.nan)
        else:
            array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{what} must be an array of numbers: {exc}") from None
    if array.ndim != ndim:
        raise ValueError(f"{what} must be {ndim}-D; got {array.ndim} dimension(s)")
    bad = ~np.isfinite(array)
    if bad.any():
        raise ValueError(f"{what} holds {int(bad.sum()):,} missing or non-finite value(s)")
    return array


def _vector(values: Any, what: str, n_rows: int | None) -> np.ndarray:
    """A 1-D float64 array of finite values, of length ``n_rows`` when that is given."""
    if isinstance(values, pd.DataFrame):
        raise TypeError(f"{what} must be 1-D; got a DataFrame")
    array = _float_array(values, what, 1)
    if n_rows is not None and len(array) != n_rows:
        raise ValueError(f"{what} has {len(array):,} values but there are {n_rows:,} rows")
    return array


def _binary(values: Any, what: str, n_rows: int) -> np.ndarray:
    array = _vector(values, what, n_rows)
    if not np.isin(array, (0.0, 1.0)).all():
        raise ValueError(f"{what} must hold 0 or 1 (or booleans) only")
    return array


def _fold_labels(values: Any, n_rows: int) -> np.ndarray:
    array = _vector(values, "fold", n_rows)
    if not (np.floor(array) == array).all():
        raise ValueError("fold must hold whole fold numbers")
    return array.astype(np.int64)


def _as_csr(X: Any, *, n_columns: int | None = None) -> sparse.csr_matrix:
    """``X`` (sparse or dense, 2-D) as a float64 CSR matrix with finite values."""
    if sparse.issparse(X):
        matrix = sparse.csr_matrix(X, dtype=np.float64)
    else:
        matrix = sparse.csr_matrix(_float_array(X, "X", 2))
    if matrix.nnz and not np.isfinite(matrix.data).all():
        raise ValueError("X holds non-finite values")
    if n_columns is not None and matrix.shape[1] != n_columns:
        raise ValueError(
            f"X has {matrix.shape[1]:,} columns but the fit has {n_columns:,}; pass a design matrix built with "
            "the same encoders"
        )
    return matrix


def _check_buckets(buckets: Any) -> tuple[tuple[int, int], ...]:
    """The D17 Wickets_Buckets as ascending, disjoint, inclusive (lowest, highest) pairs."""
    if isinstance(buckets, (str, bytes)) or not isinstance(buckets, Iterable):
        raise TypeError("wicket_buckets (D17) must be a list of [lowest, highest] pairs of wickets fallen")
    pairs: list[tuple[int, int]] = []
    for item in buckets:
        if isinstance(item, (str, bytes)) or not isinstance(item, Sequence) or len(item) != 2:
            raise ValueError(f"each Wickets_Bucket (D17) must be a [lowest, highest] pair; got {item!r}")
        if any(isinstance(v, bool) or not isinstance(v, Integral) for v in item):
            raise TypeError(f"Wickets_Bucket (D17) bounds must be whole numbers; got {item!r}")
        low, high = int(item[0]), int(item[1])
        if low < 0 or high < low:
            raise ValueError(f"Wickets_Bucket (D17) {item!r} must satisfy 0 <= lowest <= highest")
        pairs.append((low, high))
    if not pairs:
        raise ValueError("wicket_buckets (D17) holds no Wickets_Bucket")
    for (_, high), (low, _) in zip(pairs, pairs[1:]):
        if low <= high:
            raise ValueError(
                f"the Wickets_Buckets (D17) {[list(p) for p in pairs]} must be in ascending order without overlaps"
            )
    return tuple(pairs)


def _env_index(comp: Sequence[Any], season: Sequence[Any]) -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays(
        [pd.Index(list(comp), dtype=object), pd.Index(list(season), dtype=object)], names=_ENV_NAMES
    )


# ------------------------------------------------------------------------------- the encoders

@dataclass(frozen=True, eq=False, repr=False)
class OneHotEncoders:
    """The categories of every indicator group, which fix the design-matrix columns (criterion 2).

    ``build_encoders`` makes them once from the full entity universe, so every fold and every fit
    uses the same column positions. The columns run group by group in GROUPS order and, within a
    group, in the order of its categories:

    * ``strikers`` and ``bowlers``: int64 Indexes of ``striker_id`` and ``bowler_id``, ascending;
    * ``environments``: a MultiIndex of (``comp_type``, ``season``) pairs, sorted; each distinct
      ``season`` string is its own Season (criterion 6);
    * ``venues``: an int64 Index of ``venue_id``, ascending;
    * ``countries``: an Index of ``venue_country`` strings, sorted;
    * the N_CELLS innings x Phase cells 0 to 5;
    * ``wicket_buckets``: the D17 ranges of wickets fallen, inclusive, ascending.

    ``venue_countries`` holds the ``venue_country`` of each venue in ``venues`` order, or None when
    the universe has no country for it; ``venue_effects`` adds that country's effect (criterion 7).
    """

    strikers: pd.Index
    bowlers: pd.Index
    environments: pd.MultiIndex
    venues: pd.Index
    countries: pd.Index
    venue_countries: tuple[str | None, ...]
    wicket_buckets: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        for name in ("strikers", "bowlers", "venues", "countries"):
            index = getattr(self, name)
            if not isinstance(index, pd.Index) or isinstance(index, pd.MultiIndex):
                raise TypeError(f"OneHotEncoders.{name} must be a pandas Index")
            if not (index.is_unique and index.is_monotonic_increasing):
                raise ValueError(f"OneHotEncoders.{name} must hold distinct values in ascending order")
        if not isinstance(self.environments, pd.MultiIndex) or self.environments.nlevels != 2:
            raise TypeError("OneHotEncoders.environments must be a MultiIndex of (comp_type, season) pairs")
        if not (self.environments.is_unique and self.environments.is_monotonic_increasing):
            raise ValueError("OneHotEncoders.environments must hold distinct pairs in ascending order")
        countries = tuple(self.venue_countries)
        if len(countries) != len(self.venues):
            raise ValueError(
                f"OneHotEncoders.venue_countries has {len(countries):,} entries but there are "
                f"{len(self.venues):,} venues"
            )
        unknown = sorted({c for c in countries if c is not None} - set(self.countries))
        if unknown:
            raise ValueError(f"OneHotEncoders.venue_countries names countries that are not categories: {unknown}")
        object.__setattr__(self, "venue_countries", countries)
        object.__setattr__(self, "wicket_buckets", _check_buckets(self.wicket_buckets))

    def __repr__(self) -> str:
        sizes = ", ".join(f"{group} {size:,}" for group, size in self.sizes.items())
        return f"OneHotEncoders({self.n_columns:,} columns: {sizes})"

    @property
    def sizes(self) -> dict[str, int]:
        """Number of columns of each group, in GROUPS order."""
        return {
            STRIKER: len(self.strikers),
            BOWLER: len(self.bowlers),
            ENVIRONMENT: len(self.environments),
            VENUE: len(self.venues),
            COUNTRY: len(self.countries),
            CELL: N_CELLS,
            WICKETS_BUCKET: len(self.wicket_buckets),
        }

    @property
    def offsets(self) -> dict[str, int]:
        """Position of each group's first column, in GROUPS order."""
        offsets, start = {}, 0
        for group, size in self.sizes.items():
            offsets[group] = start
            start += size
        return offsets

    @property
    def n_columns(self) -> int:
        """Number of design-matrix columns, p."""
        return sum(self.sizes.values())

    def group_slice(self, group: str) -> slice:
        """The columns of one group of GROUPS, as a slice of positions."""
        if group not in GROUPS:
            raise ValueError(f"{group!r} is not a design group; the groups are {', '.join(GROUPS)}")
        start = self.offsets[group]
        return slice(start, start + self.sizes[group])

    @property
    def player_columns(self) -> np.ndarray:
        """Boolean mask of the striker and bowler columns, the columns the baseline leaves out (criterion 8)."""
        mask = np.zeros(self.n_columns, dtype=bool)
        for group in PLAYER_GROUPS:
            mask[self.group_slice(group)] = True
        return mask

    def column_labels(self) -> pd.DataFrame:
        """One row per design column: ``column`` (position), ``group`` and a readable ``label``."""
        labels: list[str] = [str(v) for v in self.strikers] + [str(v) for v in self.bowlers]
        labels += [f"{comp} {season}" for comp, season in self.environments]
        labels += [str(v) for v in self.venues] + [str(v) for v in self.countries]
        n_phases = len(Phase)
        labels += [f"innings {cell // n_phases + 1} {Phase(cell % n_phases).name.lower()}" for cell in range(N_CELLS)]
        labels += [f"{low}-{high} wickets" for low, high in self.wicket_buckets]
        groups = [group for group, size in self.sizes.items() for _ in range(size)]
        return pd.DataFrame({"column": np.arange(self.n_columns), "group": groups, "label": labels})


def build_encoders(universe: pd.DataFrame, *, wicket_buckets: Sequence[Sequence[int]]) -> OneHotEncoders:
    """The categories of every group, from the entity universe ``universe`` (criteria 2 and 6).

    ``universe`` needs the UNIVERSE_COLUMNS; pass the whole ``balls`` table of the ``reconstruct``
    checkpoint, so that every Delivery's striker, bowler, environment, venue and country has a
    column. ``wicket_buckets`` is D17 (``design_choices.D17_wicket_buckets``), a list of inclusive
    [lowest, highest] ranges of wickets fallen in ascending order.

    A missing ``season``, ``venue_id`` or ``venue_country`` adds no category (module docstring,
    Missing values). Raises DataError (exit code 3) when a venue has more than one
    ``venue_country`` value, a missing value counting as one; ValueError for a missing
    ``striker_id``, ``bowler_id`` or ``comp_type``, a non-whole id or an invalid D17; TypeError
    for a column of the wrong kind.
    """
    _require(universe, UNIVERSE_COLUMNS, "universe")
    buckets = _check_buckets(wicket_buckets)
    what = "universe"
    strikers, _ = _whole(universe, "striker_id", nullable=False, what=what)
    bowlers, _ = _whole(universe, "bowler_id", nullable=False, what=what)
    comp, _ = _text(universe, "comp_type", nullable=False, what=what)
    season, no_season = _text(universe, "season", nullable=True, what=what)
    venue, no_venue = _whole(universe, "venue_id", nullable=True, what=what)
    country, no_country = _text(universe, "venue_country", nullable=True, what=what)

    pairs = sorted(set(zip(comp[~no_season].tolist(), season[~no_season].tolist())))
    by_venue: dict[int, set[str | None]] = {}
    for venue_id, venue_country in set(zip(venue[~no_venue].tolist(), country[~no_venue].tolist())):
        by_venue.setdefault(venue_id, set()).add(venue_country)
    conflicts = {v: c for v, c in sorted(by_venue.items()) if len(c) > 1}
    if conflicts:
        shown = "; ".join(
            f"venue {v}: {', '.join(sorted('<missing>' if c is None else repr(c) for c in cs))}"
            for v, cs in list(conflicts.items())[:_SHOWN]
        )
        raise DataError(
            f"Venue_Effect (Requirement 8, criterion 7) needs one venue_country per venue:\n"
            f"  - {len(conflicts):,} venue(s) have more than one venue_country value ({shown})"
        )
    venues = sorted(by_venue)
    return OneHotEncoders(
        strikers=pd.Index(np.unique(strikers), dtype=np.int64),
        bowlers=pd.Index(np.unique(bowlers), dtype=np.int64),
        environments=_env_index([p[0] for p in pairs], [p[1] for p in pairs]),
        venues=pd.Index(np.asarray(venues, dtype=np.int64), dtype=np.int64),
        countries=pd.Index(sorted(set(country[~no_country].tolist())), dtype=object),
        venue_countries=tuple(next(iter(by_venue[v])) for v in venues),
        wicket_buckets=buckets,
    )


# ------------------------------------------------------------------------------ rows and design

def legal_rows(balls: pd.DataFrame, state: pd.DataFrame, *, training_only: bool) -> pd.DataFrame:
    """The Legal_Balls of M1 with the design inputs and both targets (criterion 1).

    ``balls`` is the ``balls`` table of the ``reconstruct`` checkpoint and ``state`` the ``state``
    table of the ``state`` checkpoint (``cell`` and ``wickets`` are read from it by ``row``). With
    ``training_only`` true the rows are the Legal_Balls kept for model training
    (``models.common.legal_training_mask``); with it false they are every Legal_Ball of innings 1
    and 2, excluded matches included, for predictions.

    Returns the LEGAL_ROW_COLUMNS on a RangeIndex, in the row order of ``balls``: ``row``,
    ``match_id``, ``innings``, the DESIGN_INPUT_COLUMNS with the dtypes of their tables,
    ``bat_runs`` (the Legal_Ball's ``runs``) and ``counted_wicket`` (bool). Raises ValueError when a
    column is absent, ``state`` repeats a ``row`` or lacks a row of a selected Legal_Ball, and
    TypeError for a non-boolean ``training_only``.
    """
    if not isinstance(training_only, bool):
        raise TypeError(f"training_only must be True or False; got {training_only!r}")
    needed = ("row", "match_id", "innings", "cls", "is_super_over", *UNIVERSE_COLUMNS, "runs", "counted_wicket")
    _require(balls, needed, "balls")
    _require(state, ("row", "cell", "wickets"), "state")
    if training_only:
        mask = legal_training_mask(balls)
    else:
        innings, _ = _whole(balls, "innings", nullable=False, what="balls")
        cls, _ = _whole(balls, "cls", nullable=False, what="balls")
        super_over = balls["is_super_over"].to_numpy(dtype=bool)
        mask = np.isin(innings, STATE_INNINGS) & ~super_over & (cls == int(DeliveryClass.LEGAL))
    picked = balls.iloc[np.flatnonzero(mask)]
    state_rows = pd.Index(state["row"].to_numpy())
    if not state_rows.is_unique:
        raise ValueError("state must hold each row once; pass the state table of the state checkpoint")
    at = state_rows.get_indexer(picked["row"].to_numpy())
    absent = at < 0
    if absent.any():
        raise ValueError(
            f"state has no row for {int(absent.sum()):,} of the {len(picked):,} Legal_Balls (for example row "
            f"{_examples(picked['row'].to_numpy()[absent])}); pass the state table built from the same balls"
        )
    out = picked[["row", "match_id", "innings", *UNIVERSE_COLUMNS]].reset_index(drop=True)
    out["cell"] = state["cell"].to_numpy()[at]
    out["wickets"] = state["wickets"].to_numpy()[at]
    out[RUNS_TARGET] = picked["runs"].to_numpy()
    out[WICKET_TARGET] = picked["counted_wicket"].to_numpy(dtype=bool)
    return out[list(LEGAL_ROW_COLUMNS)]


def _lookup(categories: pd.Index, keys: Any, missing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Category positions of the present rows (-1 where missing) and the mask of unknown values."""
    codes = np.full(len(missing), -1, dtype=np.int64)
    present = np.flatnonzero(~missing)
    if present.size:
        codes[present] = categories.get_indexer(keys)
    return codes, ~missing & (codes < 0)


def _bucket_codes(wickets: np.ndarray, buckets: tuple[tuple[int, int], ...]) -> np.ndarray:
    """Wickets_Bucket position of each wickets-fallen value; DataError when one is in no bucket."""
    top = max(high for _, high in buckets)
    table = np.full(top + 1, -1, dtype=np.int64)
    for position, (low, high) in enumerate(buckets):
        table[low : high + 1] = position
    codes = np.full(len(wickets), -1, dtype=np.int64)
    inside = (wickets >= 0) & (wickets <= top)
    codes[inside] = table[wickets[inside]]
    outside = codes < 0
    if outside.any():
        values = pd.Series(wickets[outside]).value_counts().sort_index()
        raise DataError(
            f"the Wickets_Bucket (Requirement 8, criterion 2) cannot be set:\n  - {_rows_have(int(outside.sum()))} "
            f"wickets fallen outside every D17 bucket {[list(p) for p in buckets]} "
            f"(value: rows {', '.join(f'{v}: {c:,}' for v, c in values.items())})"
        )
    return codes


def design_matrix(legal: pd.DataFrame, encoders: OneHotEncoders) -> sparse.csr_matrix:
    """The M1 design matrix of ``legal``: one row per row, one 0/1 column per category (criterion 2).

    ``legal`` needs the DESIGN_INPUT_COLUMNS (``legal_rows`` gives them). Each row has a 1 in the
    column of its striker, bowler, Competition_Type x Season, venue, venue country, innings x Phase
    cell and Wickets_Bucket, at the positions of ``encoders``; a missing ``season``, ``venue_id`` or
    ``venue_country`` leaves its group all zeros (module docstring). Returns a canonical float64
    ``scipy.sparse.csr_matrix`` of shape (len(legal), encoders.n_columns).

    Raises ValueError when a value is not one of the encoders' categories (build the encoders from
    a frame that holds every Delivery), a ``cell`` lies outside 0 to N_CELLS - 1, or a column that
    may not be missing is; DataError (exit code 3) when wickets fallen lies outside every D17
    Wickets_Bucket; TypeError for a column of the wrong kind.
    """
    _require(legal, DESIGN_INPUT_COLUMNS, "legal")
    if not isinstance(encoders, OneHotEncoders):
        raise TypeError(f"encoders must be OneHotEncoders; got {type(encoders).__name__}")
    what = "legal"
    n_rows = len(legal)
    strikers, no_striker = _whole(legal, "striker_id", nullable=False, what=what)
    bowlers, no_bowler = _whole(legal, "bowler_id", nullable=False, what=what)
    comp, no_comp = _text(legal, "comp_type", nullable=False, what=what)
    season, no_season = _text(legal, "season", nullable=True, what=what)
    venue, no_venue = _whole(legal, "venue_id", nullable=True, what=what)
    country, no_country = _text(legal, "venue_country", nullable=True, what=what)
    cell, _ = _whole(legal, "cell", nullable=False, what=what)
    wickets, _ = _whole(legal, "wickets", nullable=False, what=what)

    codes: dict[str, np.ndarray] = {}
    problems: list[str] = []
    for group, label, categories, keys, missing in (
        (STRIKER, "striker_id", encoders.strikers, strikers, no_striker),
        (BOWLER, "bowler_id", encoders.bowlers, bowlers, no_bowler),
        (VENUE, "venue_id", encoders.venues, venue, no_venue),
        (COUNTRY, "venue_country", encoders.countries, country, no_country),
    ):
        codes[group], unknown = _lookup(categories, pd.Index(keys[~missing], dtype=categories.dtype), missing)
        if unknown.any():
            problems.append(
                f"{_rows_have(int(unknown.sum()))} a {label} that the encoders do not hold (for example "
                f"{_examples(pd.unique(keys[unknown]))})"
            )
    no_env = no_comp | no_season
    codes[ENVIRONMENT], unknown = _lookup(encoders.environments, _env_index(comp[~no_env], season[~no_env]), no_env)
    if unknown.any():
        pairs = pd.unique(np.array([f"{c} {s}" for c, s in zip(comp[unknown], season[unknown])], dtype=object))
        problems.append(
            f"{_rows_have(int(unknown.sum()))} a Competition_Type x Season that the encoders do not hold (for "
            f"example {_examples(pairs)})"
        )
    bad_cell = (cell < 0) | (cell >= N_CELLS)
    if bad_cell.any():
        problems.append(f"{_rows_have(int(bad_cell.sum()))} a cell outside 0 to {N_CELLS - 1}")
    if problems:
        raise ValueError(
            "The M1 design matrix cannot be built:\n"
            + "\n".join(f"  - {problem}" for problem in problems)
            + "\nBuild the encoders with build_encoders from a frame that holds every Delivery (the balls table)."
        )
    codes[CELL] = cell
    codes[WICKETS_BUCKET] = _bucket_codes(wickets, encoders.wicket_buckets)

    offsets = encoders.offsets
    row_parts, column_parts = [], []
    for group in GROUPS:
        code = codes[group]
        keep = np.flatnonzero(code >= 0)
        row_parts.append(keep)
        column_parts.append(offsets[group] + code[keep])
    rows = np.concatenate(row_parts)
    columns = np.concatenate(column_parts)
    matrix = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, columns)), shape=(n_rows, encoders.n_columns)
    )
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix


# ------------------------------------------------------------------------- sufficient statistics

@dataclass(frozen=True, eq=False, repr=False)
class FoldStats:
    """Sufficient statistics of one fold's rows for both targets (design, Skill_Model M1).

    ``G`` = X_f'X_f (p x p), ``Xty_runs`` and ``Xty_wkt`` = X_f'y, ``s`` = the column sums of X_f,
    ``n`` = the number of rows, ``yy_*`` = y'y and ``y_*_sum`` = the sum of y, for the bat-runs
    target (``runs``) and the Counted_Wicket target (``wkt``). Arrays are stored as read-only
    float64 views.
    """

    G: np.ndarray
    Xty_runs: np.ndarray
    Xty_wkt: np.ndarray
    s: np.ndarray
    n: int
    yy_runs: float
    yy_wkt: float
    y_runs_sum: float
    y_wkt_sum: float

    def __post_init__(self) -> None:
        gram = _float_array(self.G, "FoldStats.G", 2)
        p = gram.shape[1]
        if gram.shape[0] != p:
            raise ValueError(f"FoldStats.G must be square; got shape {gram.shape}")
        object.__setattr__(self, "G", _readonly(gram))
        for name in ("Xty_runs", "Xty_wkt", "s"):
            vector = _float_array(getattr(self, name), f"FoldStats.{name}", 1)
            if vector.shape != (p,):
                raise ValueError(f"FoldStats.{name} must hold {p:,} values, one per column of G; got {len(vector):,}")
            object.__setattr__(self, name, _readonly(vector))
        if isinstance(self.n, bool) or not isinstance(self.n, Integral) or self.n < 1:
            raise ValueError(f"FoldStats.n must be a whole number of rows of at least 1; got {self.n!r}")
        object.__setattr__(self, "n", int(self.n))
        for name in ("yy_runs", "yy_wkt", "y_runs_sum", "y_wkt_sum"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
                raise ValueError(f"FoldStats.{name} must be a finite number; got {value!r}")
            object.__setattr__(self, name, float(value))

    def __repr__(self) -> str:
        return f"FoldStats(n={self.n:,}, p={self.p:,}, y_runs_sum={self.y_runs_sum:g}, y_wkt_sum={self.y_wkt_sum:g})"

    @property
    def p(self) -> int:
        """Number of design columns."""
        return self.G.shape[0]


def fold_stats(X: Any, y_runs: Any, y_wkt: Any, fold: Any) -> dict[int, FoldStats]:
    """The FoldStats of every fold that has rows, keyed by fold number in ascending order.

    ``X`` is the design matrix (sparse or dense, n x p); ``y_runs`` the bat runs of each row;
    ``y_wkt`` 1 (or True) when the row carries a Counted_Wicket and 0 otherwise; ``fold`` the
    fold of each row's match from the Fold_Table (``folds.fold_of``). All three hold one value per
    row of ``X``, in its row order.

    Raises ValueError for a length mismatch, a missing or non-finite value, a ``y_wkt`` other than
    0 or 1, or a fold that is not a whole number, and TypeError for non-numeric input.
    """
    matrix = _as_csr(X)
    n_rows = matrix.shape[0]
    runs = _vector(y_runs, "y_runs", n_rows)
    wkt = _binary(y_wkt, "y_wkt", n_rows)
    labels = _fold_labels(fold, n_rows)
    out: dict[int, FoldStats] = {}
    for f in np.unique(labels).tolist():
        rows = np.flatnonzero(labels == f)
        part = matrix[rows]
        part_t = part.T.tocsr()
        r, w = runs[rows], wkt[rows]
        out[int(f)] = FoldStats(
            G=(part_t @ part).toarray(),
            Xty_runs=np.asarray(part_t @ r, dtype=np.float64),
            Xty_wkt=np.asarray(part_t @ w, dtype=np.float64),
            s=np.asarray(part.sum(axis=0), dtype=np.float64).ravel(),
            n=len(rows),
            yy_runs=float(r @ r),
            yy_wkt=float(w @ w),
            y_runs_sum=float(r.sum()),
            y_wkt_sum=float(w.sum()),
        )
    return out


# ---------------------------------------------------------------------------------- ridge paths

@dataclass(frozen=True, eq=False, repr=False)
class RidgePath:
    """The ridge of one training set for every penalty of a grid and both targets.

    * ``train_folds``: the folds whose rows were fitted; ``grid``: the penalties, ascending;
    * ``columns``: the columns the fit may use (bool, p); ``active``: those with training rows;
    * ``counts``: the column sums over the training rows (training Legal_Balls per category);
    * ``n_rows``, ``runs_sum``, ``wickets_sum``: training rows and target sums;
    * ``coef_runs``, ``coef_wkt``: p x len(grid) coefficients, exactly 0 outside ``active``;
    * ``intercept_runs``, ``intercept_wkt``: the unpenalised intercept for each penalty.
    """

    train_folds: frozenset[int]
    grid: tuple[float, ...]
    columns: np.ndarray
    active: np.ndarray
    counts: np.ndarray
    n_rows: int
    runs_sum: float
    wickets_sum: float
    coef_runs: np.ndarray
    coef_wkt: np.ndarray
    intercept_runs: np.ndarray
    intercept_wkt: np.ndarray

    def __repr__(self) -> str:
        return (
            f"RidgePath(train_folds={sorted(self.train_folds)}, active={int(self.active.sum()):,}/{len(self.active):,}, "
            f"rows={self.n_rows:,}, grid={list(self.grid)})"
        )


def _compute_path(
    stats: Mapping[int, FoldStats], folds: frozenset[int], allowed: np.ndarray, grid: tuple[float, ...]
) -> RidgePath:
    """One eigendecomposition of the centred Gram matrix of ``folds``, reused for the grid and both targets."""
    members = [stats[f] for f in sorted(folds)]
    gram = np.array(members[0].G, dtype=np.float64, copy=True)
    s = np.array(members[0].s, dtype=np.float64, copy=True)
    xty = np.column_stack((members[0].Xty_runs, members[0].Xty_wkt))
    for member in members[1:]:
        gram += member.G
        s += member.s
        xty += np.column_stack((member.Xty_runs, member.Xty_wkt))
    n_rows = sum(member.n for member in members)
    sums = np.array([sum(m.y_runs_sum for m in members), sum(m.y_wkt_sum for m in members)])
    ybar = sums / n_rows

    p, n_grid = len(s), len(grid)
    active = allowed & (s > 0)
    idx = np.flatnonzero(active)
    coef = np.zeros((2, p, n_grid), dtype=np.float64)
    if idx.size:
        s_a = s[idx]
        centred = gram[np.ix_(idx, idx)]
        del gram
        centred -= np.outer(s_a, s_a) / n_rows
        try:
            eigenvalues, Q = scipy.linalg.eigh(centred, driver="evd", overwrite_a=True, check_finite=False)
        except np.linalg.LinAlgError as exc:
            raise np.linalg.LinAlgError(
                f"the eigendecomposition of the centred Gram matrix of training folds {sorted(folds)} failed: {exc}"
            ) from None
        eigenvalues = np.maximum(eigenvalues, 0.0)  # G_c is positive semi-definite; drop roundoff below 0
        rotated = Q.T @ (xty[idx] - np.outer(s_a, ybar))  # Q'b_c for both targets, k x 2
        shrink = 1.0 / (eigenvalues[:, None] + np.asarray(grid, dtype=np.float64)[None, :])  # k x grid
        for target in range(2):
            coef[target][idx] = Q @ (rotated[:, target : target + 1] * shrink)
    intercepts = ybar[:, None] - np.einsum("p,tpg->tg", s / n_rows, coef)
    return RidgePath(
        train_folds=folds,
        grid=grid,
        columns=_readonly(allowed.copy()),
        active=_readonly(active),
        counts=_readonly(s),
        n_rows=int(n_rows),
        runs_sum=float(sums[0]),
        wickets_sum=float(sums[1]),
        coef_runs=_readonly(coef[0]),
        coef_wkt=_readonly(coef[1]),
        intercept_runs=_readonly(intercepts[0].copy()),
        intercept_wkt=_readonly(intercepts[1].copy()),
    )


class RidgeCache:
    """Ridge paths by (training folds, column subset, grid), so each set is decomposed once.

    Pass one cache to every ``fit_ridge``, ``fit_baseline`` and ``ridge_path`` call on the same
    ``fold_stats`` result (module docstring, Reusing eigendecompositions). The cache binds to the
    first statistics it serves and raises ValueError when given others.
    """

    def __init__(self) -> None:
        self._stats: Mapping[int, FoldStats] | None = None
        self._paths: dict[tuple[frozenset[int], bytes, tuple[float, ...]], RidgePath] = {}
        self._decompositions = 0

    def __repr__(self) -> str:
        return f"RidgeCache(paths={len(self._paths)}, decompositions={self._decompositions})"

    def __len__(self) -> int:
        return len(self._paths)

    @property
    def decompositions(self) -> int:
        """Number of eigendecompositions computed so far (one per distinct cached path)."""
        return self._decompositions

    def _path(
        self, stats: Mapping[int, FoldStats], folds: frozenset[int], allowed: np.ndarray, grid: tuple[float, ...]
    ) -> RidgePath:
        if self._stats is None:
            self._stats = stats
        elif stats is not self._stats:
            raise ValueError(
                "this RidgeCache holds ridge paths of other fold statistics; use one RidgeCache per fold_stats result"
            )
        key = (folds, np.packbits(allowed).tobytes(), grid)
        path = self._paths.get(key)
        if path is None:
            path = _compute_path(stats, folds, allowed, grid)
            self._paths[key] = path
            self._decompositions += 1
        return path


def _check_stats(stats: Any) -> int:
    """Number of design columns of ``stats``, after checking it is a fold -> FoldStats mapping."""
    if not isinstance(stats, Mapping):
        raise TypeError(f"stats must map fold numbers to FoldStats, as fold_stats returns; got {type(stats).__name__}")
    if not stats:
        raise ValueError("stats holds no fold; fold_stats gives one entry per fold with rows")
    widths = set()
    for fold, item in stats.items():
        if isinstance(fold, bool) or not isinstance(fold, Integral):
            raise TypeError(f"stats keys must be whole fold numbers; got {fold!r}")
        if not isinstance(item, FoldStats):
            raise TypeError(f"stats[{fold!r}] must be FoldStats; got {type(item).__name__}")
        widths.add(item.p)
    if len(widths) != 1:
        raise ValueError(f"the FoldStats of stats have different column counts {sorted(widths)}")
    return widths.pop()


def _train_folds(train_folds: Any, stats: Mapping[int, FoldStats], minimum: int) -> frozenset[int]:
    if isinstance(train_folds, (str, bytes)) or not isinstance(train_folds, Iterable):
        raise TypeError("train_folds must be a collection of fold numbers, such as FitSpec.train_folds")
    values = list(train_folds)
    if any(isinstance(f, bool) or not isinstance(f, Integral) for f in values):
        raise TypeError(f"train_folds must hold whole fold numbers; got {values!r}")
    folds = frozenset(int(f) for f in values)
    absent = sorted(folds - {int(f) for f in stats})
    if absent:
        raise ValueError(
            f"train_folds holds fold(s) {absent} that have no fold statistics (stats holds folds {sorted(stats)}; "
            "fold_stats gives one entry per fold with rows)"
        )
    if len(folds) < minimum:
        reason = "leave-one-fold-out cross-validation needs" if minimum > 1 else "a ridge needs"
        raise ValueError(f"{reason} at least {minimum} training fold(s); got {sorted(folds)}")
    return folds


def _grid(grid: Any) -> tuple[float, ...]:
    if isinstance(grid, (str, bytes)) or not isinstance(grid, Iterable):
        raise TypeError("grid must be a list of ridge penalties, such as models.m1.ridge_grid")
    values = list(grid)
    if not values:
        raise ValueError("the ridge penalty grid is empty")
    if any(isinstance(v, bool) or not isinstance(v, Real) for v in values):
        raise TypeError(f"ridge penalties must be numbers; got {values!r}")
    floats = [float(v) for v in values]
    bad = [v for v in floats if not (math.isfinite(v) and v > 0.0)]
    if bad:
        raise ValueError(f"every ridge penalty must be a finite number above 0; got {bad}")
    ordered = tuple(sorted(floats))
    if len(set(ordered)) != len(ordered):
        raise ValueError(f"the ridge penalty grid repeats a value: {values!r}")
    return ordered


def _allowed(columns: Any, p: int) -> np.ndarray:
    if columns is None:
        return np.ones(p, dtype=bool)
    mask = np.asarray(columns)
    if mask.dtype != bool or mask.shape != (p,):
        raise ValueError(f"columns must be a boolean mask with one value per design column ({p:,}); got {mask.shape}")
    return mask.copy()


def _cache(cache: Any) -> RidgeCache:
    if cache is None:
        return RidgeCache()
    if not isinstance(cache, RidgeCache):
        raise TypeError(f"cache must be a RidgeCache; got {type(cache).__name__}")
    return cache


def ridge_path(
    stats: Mapping[int, FoldStats],
    train_folds: Iterable[int],
    grid: Sequence[float],
    *,
    columns: Any = None,
    cache: RidgeCache | None = None,
) -> RidgePath:
    """The ridge on ``train_folds`` for every penalty of ``grid`` and both targets (no penalty choice).

    ``columns`` is None (every column) or a boolean mask of the columns the fit may use. Pass a
    RidgeCache to reuse the eigendecomposition across calls. Raises the errors of fit_ridge, except
    that one training fold is enough.
    """
    p = _check_stats(stats)
    folds = _train_folds(train_folds, stats, minimum=1)
    return _cache(cache)._path(stats, folds, _allowed(columns, p), _grid(grid))


# ----------------------------------------------------------------------------------- the fit

@dataclass(frozen=True, eq=False, repr=False)
class RidgeFit:
    """One M1 fit (or baseline fit): both targets at their selected penalties (criteria 1, 3 and 5).

    * ``train_folds``, ``grid``, ``columns``, ``active``, ``counts``, ``n_rows``, ``runs_sum`` and
      ``wickets_sum`` as in RidgePath;
    * ``coef_runs``, ``coef_wkt``: length-p coefficients at the selected penalties, exactly 0
      outside ``active``; ``intercept_runs``, ``intercept_wkt``: the unpenalised intercepts;
    * ``penalty_runs``, ``penalty_wkt``: the penalties chosen by leave-one-fold-out CV;
    * ``cv_mse_runs``, ``cv_mse_wkt``: the row-pooled CV MSE of every grid penalty, in grid order.
    """

    train_folds: frozenset[int]
    grid: tuple[float, ...]
    columns: np.ndarray
    active: np.ndarray
    counts: np.ndarray
    n_rows: int
    runs_sum: float
    wickets_sum: float
    coef_runs: np.ndarray
    coef_wkt: np.ndarray
    intercept_runs: float
    intercept_wkt: float
    penalty_runs: float
    penalty_wkt: float
    cv_mse_runs: np.ndarray
    cv_mse_wkt: np.ndarray

    def __repr__(self) -> str:
        return (
            f"RidgeFit(train_folds={sorted(self.train_folds)}, columns={int(self.columns.sum()):,}/{self.n_columns:,}, "
            f"active={int(self.active.sum()):,}, rows={self.n_rows:,}, penalty_runs={self.penalty_runs:g}, "
            f"penalty_wkt={self.penalty_wkt:g})"
        )

    @property
    def n_columns(self) -> int:
        """Number of design columns, p."""
        return len(self.coef_runs)

    @property
    def mean_runs(self) -> float:
        """Training mean bat runs per Legal_Ball: the theta_bat of a player with zero effects.

        Not the theta_bat of Average_Skill, which is the mean theta_bat of the strikers of the
        training Legal_Balls, weighted by Legal_Balls faced (``models.outcome.average_skill``).
        """
        return self.runs_sum / self.n_rows

    @property
    def mean_wkt(self) -> float:
        """Training share of Legal_Balls that carry a Counted_Wicket."""
        return self.wickets_sum / self.n_rows

    @property
    def runs_per_wicket(self) -> float:
        """Training bat runs per Counted_Wicket on the Legal_Balls; NaN when there is no Counted_Wicket."""
        return self.runs_sum / self.wickets_sum if self.wickets_sum > 0 else math.nan


def _held_out_sse(held_out: FoldStats, path: RidgePath) -> tuple[np.ndarray, np.ndarray]:
    """Squared error of ``path`` (every penalty) on the rows of ``held_out``, for both targets."""
    n_grid = len(path.grid)
    coef = np.hstack((path.coef_runs, path.coef_wkt))  # p x 2 grid
    quadratic = np.einsum("ij,ij->j", coef, held_out.G @ coef)
    out = []
    for target, (xty, yy, y_sum, intercept) in enumerate(
        (
            (held_out.Xty_runs, held_out.yy_runs, held_out.y_runs_sum, path.intercept_runs),
            (held_out.Xty_wkt, held_out.yy_wkt, held_out.y_wkt_sum, path.intercept_wkt),
        )
    ):
        part = slice(target * n_grid, (target + 1) * n_grid)
        beta = coef[:, part]
        out.append(
            yy
            - 2.0 * intercept * y_sum
            - 2.0 * (xty @ beta)
            + held_out.n * intercept**2
            + 2.0 * intercept * (held_out.s @ beta)
            + quadratic[part]
        )
    return out[0], out[1]


def _select(mse: np.ndarray) -> int:
    """Grid position of the lowest MSE; among exact ties the largest penalty (the grid is ascending)."""
    return int(np.flatnonzero(mse == mse.min())[-1])


def fit_ridge(
    stats: Mapping[int, FoldStats],
    train_folds: Iterable[int],
    grid: Sequence[float],
    *,
    columns: Any = None,
    cache: RidgeCache | None = None,
) -> RidgeFit:
    """Both M1 ridges on the folds ``train_folds``, each at its cross-validated penalty (criteria 1, 3 and 5).

    ``stats`` is the fold_stats result; ``train_folds`` the folds the fit trains on (at least 2,
    for example ``FitSpec.train_folds``); ``grid`` the penalty candidates (``models.m1.ridge_grid``).
    ``columns`` is None (every column) or a boolean mask of the columns the fit may use
    (fit_baseline passes the non-player columns). Pass one RidgeCache to every call on the same
    statistics, so each fold set is decomposed once.

    For each target, every grid penalty is scored by leave-one-fold-out CV over the folds of
    ``train_folds`` only, with the row-pooled MSE computed from the fold statistics, and the lowest
    MSE wins (the largest penalty among exact ties). The coefficients are those of the ridge on all
    of ``train_folds`` at that penalty; a column without training rows has a coefficient of exactly
    0. Raises ValueError for fewer than 2 training folds, a fold without statistics, an empty grid
    or a penalty that is not a finite number above 0, and TypeError for input of the wrong kind.
    """
    p = _check_stats(stats)
    folds = _train_folds(train_folds, stats, minimum=2)
    penalties = _grid(grid)
    allowed = _allowed(columns, p)
    cache = _cache(cache)
    path = cache._path(stats, folds, allowed, penalties)
    sse_runs = np.zeros(len(penalties))
    sse_wkt = np.zeros(len(penalties))
    held_out_rows = 0
    for fold in sorted(folds):
        runs, wkt = _held_out_sse(stats[fold], cache._path(stats, folds - {fold}, allowed, penalties))
        sse_runs += runs
        sse_wkt += wkt
        held_out_rows += stats[fold].n
    mse_runs, mse_wkt = sse_runs / held_out_rows, sse_wkt / held_out_rows
    pick_runs, pick_wkt = _select(mse_runs), _select(mse_wkt)
    return RidgeFit(
        train_folds=folds,
        grid=penalties,
        columns=path.columns,
        active=path.active,
        counts=path.counts,
        n_rows=path.n_rows,
        runs_sum=path.runs_sum,
        wickets_sum=path.wickets_sum,
        coef_runs=_readonly(np.ascontiguousarray(path.coef_runs[:, pick_runs])),
        coef_wkt=_readonly(np.ascontiguousarray(path.coef_wkt[:, pick_wkt])),
        intercept_runs=float(path.intercept_runs[pick_runs]),
        intercept_wkt=float(path.intercept_wkt[pick_wkt]),
        penalty_runs=penalties[pick_runs],
        penalty_wkt=penalties[pick_wkt],
        cv_mse_runs=_readonly(mse_runs),
        cv_mse_wkt=_readonly(mse_wkt),
    )


def fit_baseline(
    stats: Mapping[int, FoldStats],
    train_folds: Iterable[int],
    grid: Sequence[float],
    encoders: OneHotEncoders,
    *,
    cache: RidgeCache | None = None,
) -> RidgeFit:
    """The M1 baseline (criterion 8): fit_ridge without the striker and bowler columns.

    Same statistics, folds, grid and penalty search as M1. Raises ValueError when ``encoders``
    does not describe the columns of ``stats``, and the errors of fit_ridge.
    """
    p = _check_stats(stats)
    if not isinstance(encoders, OneHotEncoders):
        raise TypeError(f"encoders must be OneHotEncoders; got {type(encoders).__name__}")
    if encoders.n_columns != p:
        raise ValueError(f"the encoders describe {encoders.n_columns:,} columns but the statistics have {p:,}")
    return fit_ridge(stats, train_folds, grid, columns=~encoders.player_columns, cache=cache)


# ------------------------------------------------------------------------------ skill and context

def _check_fit(fit: Any, encoders: Any) -> None:
    if not isinstance(fit, RidgeFit):
        raise TypeError(f"fit must be a RidgeFit; got {type(fit).__name__}")
    if not isinstance(encoders, OneHotEncoders):
        raise TypeError(f"encoders must be OneHotEncoders; got {type(encoders).__name__}")
    if fit.n_columns != encoders.n_columns:
        raise ValueError(
            f"the fit has {fit.n_columns:,} design columns but the encoders describe {encoders.n_columns:,}; use "
            "the encoders that built the fit's design matrix"
        )


def _group_values(fit: RidgeFit, encoders: OneHotEncoders, group: str, keys: pd.Index) -> dict[str, np.ndarray]:
    """Runs effect, dismissal effect and training count of each key in one group (0 when absent)."""
    categories = {STRIKER: encoders.strikers, BOWLER: encoders.bowlers}[group]
    position = categories.get_indexer(keys)
    column = encoders.offsets[group] + np.maximum(position, 0)
    known = position >= 0
    return {
        "runs": np.where(known, fit.coef_runs[column], 0.0),
        "wkt": np.where(known, fit.coef_wkt[column], 0.0),
        "balls": np.where(known, fit.counts[column], 0.0).astype(np.int64),
    }


def _player_ids(players: Any) -> np.ndarray:
    if isinstance(players, (str, bytes, pd.DataFrame)) or np.ndim(players) != 1:
        raise TypeError("players must be a 1-D array or Series of player ids")
    series = players if isinstance(players, pd.Series) else pd.Series(np.asarray(players))
    ids, _ = _whole(series.to_frame("player_id"), "player_id", nullable=False, what="players")
    repeated = pd.Index(ids).duplicated()
    if repeated.any():
        raise ValueError(
            f"players repeats {int(repeated.sum()):,} id(s) (for example {_examples(np.unique(ids[repeated]))}); "
            "give each player once"
        )
    return ids


def thetas(fit: RidgeFit, players: Any, encoders: OneHotEncoders) -> pd.DataFrame:
    """theta_bat and theta_bowl of every player in ``players`` from one fit (criteria 4 and 5).

    ``players`` holds distinct player ids (``player_universe`` gives those of criterion 4);
    ``encoders`` are the encoders of the fit's design matrix. With m the fit's training mean bat
    runs per Legal_Ball and R its training runs per Counted_Wicket:
    theta_bat = m + striker runs effect - striker dismissal effect x R (runs per Legal_Ball) and
    theta_bowl = -bowler runs effect + bowler dismissal effect x R (runs saved per Legal_Ball). A
    player without training Legal_Balls in a Role has effects of exactly 0 there, so theta_bat = m
    and theta_bowl = 0.

    Returns the THETA_COLUMNS, one row per player in the order given: ``player_id`` (int64),
    ``theta_bat``, ``theta_bowl``, ``bat_balls`` and ``bowl_balls`` (the fit's training
    Legal_Balls as striker and as bowler, int64) and the four effects (``bat_runs_effect``,
    ``bat_wkt_effect``, ``bowl_runs_effect``, ``bowl_wkt_effect``). Raises ValueError when the
    training rows hold no Counted_Wicket (R is undefined), a player id is missing, not whole or
    repeated, or the encoders do not match the fit.
    """
    _check_fit(fit, encoders)
    ids = _player_ids(players)
    per_wicket = fit.runs_per_wicket
    if not math.isfinite(per_wicket):
        raise ValueError(
            f"the fit's {fit.n_rows:,} training Legal_Balls hold no Counted_Wicket, so the runs per Counted_Wicket "
            "that theta_bat and theta_bowl need is undefined"
        )
    keys = pd.Index(ids, dtype=np.int64)
    bat = _group_values(fit, encoders, STRIKER, keys)
    bowl = _group_values(fit, encoders, BOWLER, keys)
    return pd.DataFrame(
        {
            "player_id": ids,
            "theta_bat": fit.mean_runs + bat["runs"] - bat["wkt"] * per_wicket,
            "theta_bowl": -bowl["runs"] + bowl["wkt"] * per_wicket,
            "bat_balls": bat["balls"],
            "bowl_balls": bowl["balls"],
            "bat_runs_effect": bat["runs"],
            "bat_wkt_effect": bat["wkt"],
            "bowl_runs_effect": bowl["runs"],
            "bowl_wkt_effect": bowl["wkt"],
        },
        columns=list(THETA_COLUMNS),
    )


def venue_effects(fit: RidgeFit, encoders: OneHotEncoders) -> pd.DataFrame:
    """Venue_Effect of every venue of the encoders: its own effect plus its country's (criterion 7).

    Returns the VENUE_COLUMNS, one row per venue in ``encoders.venues`` order: ``venue_id``
    (int64), ``venue_country`` (string, missing when the universe has none), ``venue_effect``
    (runs regression, bat runs per Legal_Ball), ``venue_effect_wkt`` (dismissal regression,
    Counted_Wickets per Legal_Ball) and ``venue_balls`` (training Legal_Balls at the venue). A
    venue without training Legal_Balls has an own effect of exactly 0, so its Venue_Effect is
    exactly its country's effect. Raises ValueError when the encoders do not match the fit.
    """
    _check_fit(fit, encoders)
    venue_columns = encoders.offsets[VENUE] + np.arange(len(encoders.venues))
    country_columns = np.full(len(encoders.venues), -1, dtype=np.int64)
    known = [i for i, country in enumerate(encoders.venue_countries) if country is not None]
    if known:
        names = pd.Index([encoders.venue_countries[i] for i in known], dtype=object)
        country_columns[known] = encoders.offsets[COUNTRY] + encoders.countries.get_indexer(names)
    has_country = country_columns >= 0
    safe = np.maximum(country_columns, 0)

    def combined(coef: np.ndarray) -> np.ndarray:
        return coef[venue_columns] + np.where(has_country, coef[safe], 0.0)

    return pd.DataFrame(
        {
            "venue_id": encoders.venues.to_numpy(dtype=np.int64),
            "venue_country": pd.array(list(encoders.venue_countries), dtype="string"),
            "venue_effect": combined(fit.coef_runs),
            "venue_effect_wkt": combined(fit.coef_wkt),
            "venue_balls": fit.counts[venue_columns].astype(np.int64),
        },
        columns=list(VENUE_COLUMNS),
    )


def scoring_environments(fit: RidgeFit, encoders: OneHotEncoders) -> pd.DataFrame:
    """Scoring_Environment of every Competition_Type x Season of the encoders (criteria 5 and 6).

    Returns the ENVIRONMENT_COLUMNS, one row per pair in ``encoders.environments`` order:
    ``comp_type`` and ``season`` (str), ``scoring_env`` (runs regression, bat runs per Legal_Ball),
    ``scoring_env_wkt`` (dismissal regression, Counted_Wickets per Legal_Ball) and ``env_balls``
    (training Legal_Balls). A pair without training Legal_Balls has effects of exactly 0. Raises
    ValueError when the encoders do not match the fit.
    """
    _check_fit(fit, encoders)
    columns = encoders.offsets[ENVIRONMENT] + np.arange(len(encoders.environments))
    return pd.DataFrame(
        {
            "comp_type": pd.array(list(encoders.environments.get_level_values(0)), dtype="str"),
            "season": pd.array(list(encoders.environments.get_level_values(1)), dtype="str"),
            "scoring_env": fit.coef_runs[columns],
            "scoring_env_wkt": fit.coef_wkt[columns],
            "env_balls": fit.counts[columns].astype(np.int64),
        },
        columns=list(ENVIRONMENT_COLUMNS),
    )


def player_universe(balls: pd.DataFrame, xi: pd.DataFrame) -> np.ndarray:
    """Every player that each fit must rate (criterion 4), as ascending distinct int64 ids.

    The players who appear as ``striker_id``, ``non_striker_id`` or ``bowler_id`` in any Delivery
    of ``balls`` (every innings), or who have ``is_played`` true in ``xi`` (the ``playing_eleven``
    table of the ``load`` checkpoint; a missing ``is_played`` counts as false). Raises ValueError
    when a column is absent and TypeError when ``is_played`` is not boolean.
    """
    _require(balls, ("striker_id", "non_striker_id", "bowler_id"), "balls")
    _require(xi, ("player_id", "is_played"), "xi")
    if not pd.api.types.is_bool_dtype(xi["is_played"].dtype):
        raise TypeError(f"xi column 'is_played' must be boolean; got dtype {xi['is_played'].dtype}")
    parts = []
    for column in ("striker_id", "non_striker_id", "bowler_id"):
        values, missing = _whole(balls, column, nullable=True, what="balls")
        parts.append(values[~missing])
    played = xi["is_played"].fillna(False).to_numpy(dtype=bool)
    ids, missing = _whole(xi, "player_id", nullable=True, what="xi")
    parts.append(ids[played & ~missing])
    return np.unique(np.concatenate(parts)).astype(np.int64)


# ------------------------------------------------------------------------- predictions and metrics

def predict(fit: RidgeFit, X: Any) -> tuple[np.ndarray, np.ndarray]:
    """Predicted bat runs and unclipped dismissal probability of each row of the design matrix ``X``.

    ``X`` must have the fit's columns (design_matrix with the same encoders). Returns two float64
    arrays in the row order of ``X``. Raises ValueError for a column mismatch.
    """
    if not isinstance(fit, RidgeFit):
        raise TypeError(f"fit must be a RidgeFit; got {type(fit).__name__}")
    matrix = _as_csr(X, n_columns=fit.n_columns)
    runs = np.asarray(matrix @ fit.coef_runs, dtype=np.float64) + fit.intercept_runs
    wkt = np.asarray(matrix @ fit.coef_wkt, dtype=np.float64) + fit.intercept_wkt
    return runs, wkt


def out_of_fold_predictions(fits: Mapping[int, RidgeFit], X: Any, fold: Any) -> tuple[np.ndarray, np.ndarray]:
    """Out_Of_Fold_Predictions of both targets: each row predicted by ``fits[its fold]``.

    ``fits`` maps each fold k to a fit that excludes fold k (the outer fits); ``fold`` gives the
    fold of each row of ``X``. Returns two float64 arrays in the row order of ``X``. Raises
    ValueError when a fold of ``fold`` has no fit or its fit trained on that fold (Requirement 7,
    criterion 6).
    """
    if not isinstance(fits, Mapping):
        raise TypeError(f"fits must map fold numbers to RidgeFits; got {type(fits).__name__}")
    matrix = _as_csr(X)
    labels = _fold_labels(fold, matrix.shape[0])
    runs = np.empty(matrix.shape[0], dtype=np.float64)
    wkt = np.empty(matrix.shape[0], dtype=np.float64)
    for k in np.unique(labels).tolist():
        rows = np.flatnonzero(labels == k)
        fit = fits.get(int(k))
        if fit is None:
            raise ValueError(f"fits has no fit for fold {k}, which holds {_rows(len(rows))}")
        if not isinstance(fit, RidgeFit):
            raise TypeError(f"fits[{k}] must be a RidgeFit; got {type(fit).__name__}")
        if int(k) in fit.train_folds:
            raise ValueError(
                f"the fit given for fold {k} trained on fold {k}; an Out_Of_Fold_Prediction needs a fit that "
                "excludes the row's fold (Requirement 7, criterion 6)"
            )
        runs[rows], wkt[rows] = predict(fit, matrix[rows])
    return runs, wkt


def pooled_rmse(y: Any, pred: Any) -> float:
    """Root mean squared error over every row given, pooled across folds (criterion 8)."""
    actual = _vector(y, "y", None)
    if not len(actual):
        raise ValueError("RMSE needs at least one row")
    predicted = _vector(pred, "pred", len(actual))
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def clipped_log_loss(y_wkt: Any, p_wkt: Any, prob_clip: Sequence[float]) -> float:
    """Pooled log loss of dismissal predictions clipped to ``prob_clip`` first (criterion 8).

    ``prob_clip`` is ``models.m1.prob_clip``, [lower, upper] with 0 < lower < upper < 1. Each
    prediction is clipped to it, and then ``models.common.pooled_log_loss`` gives the mean over
    every row. Raises ValueError for invalid bounds and the errors of pooled_log_loss.
    """
    if isinstance(prob_clip, (str, bytes)) or not isinstance(prob_clip, Sequence) or len(prob_clip) != 2:
        raise TypeError("prob_clip must be a [lower, upper] pair, such as models.m1.prob_clip")
    if any(isinstance(v, bool) or not isinstance(v, Real) for v in prob_clip):
        raise TypeError(f"prob_clip bounds must be numbers; got {prob_clip!r}")
    lower, upper = float(prob_clip[0]), float(prob_clip[1])
    if not 0.0 < lower < upper < 1.0:
        raise ValueError(f"prob_clip must satisfy 0 < lower < upper < 1; got {list(prob_clip)}")
    probabilities = _vector(p_wkt, "p_wkt", None)
    return pooled_log_loss(y_wkt, np.clip(probabilities, lower, upper))


# ================================================================================ M1-derived quantities
# Requirement 8, criteria 12-16; Requirement 2, criterion 3 (module docstring, "M1-derived quantities").

#: Batting positions 1 to TOP_POSITIONS are the top of the order in Typical_Top7_Sum (glossary;
#: criterion 12). The number is fixed by that definition, not a Method_Constant.
TOP_POSITIONS = 7

#: The number of players a side fields, fixed by the Laws of Cricket, so neither a Method_Constant
#: nor a Config value. Team_Strength is PLAYERS_PER_SIDE x the mean of theta_bat + theta_bowl over
#: a side's is_played players (glossary; criterion 14), which puts a side that lists an Impact
#: Player (12 is_played players) on the same 11-player scale as a side that lists 11.
PLAYERS_PER_SIDE = 11

#: Columns of the rbq result, in order.
RBQ_COLUMNS: tuple[str, ...] = ("theta_next_batter", "rbq", "rbq_post_wicket")

#: The M1 quantity columns of each sandbox view V_k, in the design's order (Data Models, view_k).
VIEW_COLUMNS: tuple[str, ...] = (
    "theta_striker", "theta_non_striker", "theta_bowler", "theta_next_batter", "scoring_env", "venue_effect",
    "rbq", "rbq_post_wicket", "tsg", "match_day", "m1_pred_runs",
)  # fmt: skip

#: Columns of ``balls`` and of ``state`` that view_rows copies, and the columns view_quantities reads.
_VIEW_BALL_COLUMNS: tuple[str, ...] = (
    "match_id", "innings", "innings_ball_num", "cls", "runs", "striker_id", "non_striker_id", "bowler_id",
    "batting_team_id", "bowling_team_id", "comp_type", "season", "venue_id", "venue_country",
)  # fmt: skip
_VIEW_STATE_COLUMNS: tuple[str, ...] = ("cell", "wickets", *BTC_SLOT_COLUMNS)
VIEW_INPUT_COLUMNS: tuple[str, ...] = (*_VIEW_BALL_COLUMNS, *_VIEW_STATE_COLUMNS)

_XI_COLUMNS: tuple[str, ...] = ("match_id", "team_id", "player_id", "batting_order", "is_played")
_MDC_COLUMNS: tuple[str, ...] = ("match_id", "innings", "innings_ball_num", "cls", "runs")
_SIDE_COLUMNS: tuple[str, ...] = ("match_id", "batting_team_id", "bowling_team_id")


@dataclass(frozen=True, eq=False)
class _Theta:
    """theta_bat (and theta_bowl) of each player of a theta table, for lookups by player id."""

    index: pd.Index
    bat: np.ndarray
    bowl: np.ndarray | None

    @classmethod
    def of(cls, theta: Any, *, with_bowl: bool) -> _Theta:
        columns = ("player_id", "theta_bat", "theta_bowl") if with_bowl else ("player_id", "theta_bat")
        _require(theta, columns, "theta")
        ids, _ = _whole(theta, "player_id", nullable=False, what="theta")
        index = pd.Index(ids)
        if not index.is_unique:
            raise ValueError("theta repeats a player_id; pass one row per player, as thetas gives")
        bat = _vector(theta["theta_bat"], "theta column 'theta_bat'", len(ids))
        bowl = _vector(theta["theta_bowl"], "theta column 'theta_bowl'", len(ids)) if with_bowl else None
        return cls(index, bat, bowl)

    def lookup(self, values: np.ndarray, ids: np.ndarray, what: str) -> np.ndarray:
        """``values`` of each player id in ``ids``; ValueError naming ``what`` for an id theta lacks."""
        at = self.index.get_indexer(ids)
        absent = at < 0
        if absent.any():
            raise ValueError(
                f"theta has no row for {int(absent.sum()):,} {what} (for example player "
                f"{_examples(np.unique(ids[absent]))}); pass the thetas of every player that player_universe lists"
            )
        return values[at]


def _played_players(xi: Any) -> pd.DataFrame:
    """The distinct ``is_played`` players of each side of ``xi``, each with his lowest ``batting_order``.

    Returns ``match_id``, ``team_id``, ``player_id`` (int64) and ``batting_order`` (float64, NaN
    when none of the player's rows has one), sorted by side and player.
    """
    _require(xi, _XI_COLUMNS, "xi")
    if not pd.api.types.is_bool_dtype(xi["is_played"].dtype):
        raise TypeError(f"xi column 'is_played' must be boolean; got dtype {xi['is_played'].dtype}")
    played = xi.iloc[np.flatnonzero(xi["is_played"].to_numpy(dtype=bool, na_value=False))]
    match, _ = _whole(played, "match_id", nullable=False, what="xi")
    team, _ = _whole(played, "team_id", nullable=False, what="xi")
    player, _ = _whole(played, "player_id", nullable=False, what="xi")
    order, no_order = _whole(played, "batting_order", nullable=True, what="xi")
    frame = pd.DataFrame(
        {
            "match_id": match,
            "team_id": team,
            "player_id": player,
            "batting_order": np.where(no_order, np.nan, order.astype(np.float64)),
        }
    )
    return frame.groupby(["match_id", "team_id", "player_id"], sort=True, as_index=False)["batting_order"].min()


def _in_matches(players: pd.DataFrame, match_ids: np.ndarray) -> pd.DataFrame:
    return players.iloc[np.flatnonzero(np.isin(players["match_id"].to_numpy(), np.unique(match_ids)))]


def _per_side(players: pd.DataFrame, values: np.ndarray, *, mean: bool = False) -> pd.Series:
    """The sum of ``values`` (one per row of ``players``) over each side, or with ``mean`` their mean.

    The result is on a (match_id, team_id) MultiIndex with one entry per side that has a row.
    """
    keys = [players["match_id"].to_numpy(), players["team_id"].to_numpy()]
    groups = pd.Series(values, dtype=np.float64).groupby(keys, sort=True)
    return groups.mean() if mean else groups.sum()


def _side_values(per_side: pd.Series | None, match: np.ndarray, team: np.ndarray, no_team: np.ndarray) -> np.ndarray:
    """The ``per_side`` value of each (match, team) pair; NaN where the side is absent or the team missing."""
    out = np.full(len(match), np.nan)
    if per_side is None or not len(per_side) or not len(match):
        return out
    at = per_side.index.get_indexer(pd.MultiIndex.from_arrays([match, team]))
    found = (at >= 0) & ~no_team
    out[found] = per_side.to_numpy(dtype=np.float64)[at[found]]
    return out


def top7_sums(theta: pd.DataFrame, xi: pd.DataFrame, team_innings: pd.DataFrame) -> np.ndarray:
    """The summed theta_bat of batting positions 1-7 of each team-innings (criterion 12).

    ``theta`` is a theta table of one fit (``player_id``, ``theta_bat``; ``thetas`` gives one);
    ``xi`` the ``playing_eleven`` table; ``team_innings`` one row per team-innings with
    ``match_id`` and ``batting_team_id``. The batting positions 1-7 of a side are its distinct
    ``is_played`` players whose lowest ``batting_order`` is 1 to TOP_POSITIONS.

    Returns a float64 array, one value per row of ``team_innings``: NaN when the batting side has
    no such player (no ``is_played`` row, or none in positions 1-7) or ``batting_team_id`` is
    missing. Raises ValueError when theta lacks one of those players, a column is absent or a
    value malformed, and TypeError when ``is_played`` is not boolean.
    """
    _require(team_innings, ("match_id", "batting_team_id"), "team_innings")
    table = _Theta.of(theta, with_bowl=False)
    match, _ = _whole(team_innings, "match_id", nullable=False, what="team_innings")
    team, no_team = _whole(team_innings, "batting_team_id", nullable=True, what="team_innings")
    players = _in_matches(_played_players(xi), match)
    order = players["batting_order"].to_numpy()
    top = players.iloc[np.flatnonzero((order >= 1) & (order <= TOP_POSITIONS))]  # NaN compares False
    if top.empty:
        return np.full(len(match), np.nan)
    bat = table.lookup(table.bat, top["player_id"].to_numpy(), "players in batting positions 1-7")
    return _side_values(_per_side(top, bat), match, team, no_team)


def typical_top7_sum(theta: pd.DataFrame, xi: pd.DataFrame, team_innings: pd.DataFrame) -> float:
    """Typical_Top7_Sum of one fit: the median of ``top7_sums`` over its training team-innings (criterion 12).

    ``team_innings`` holds each innings-1 and innings-2 team-innings of the fit's training data
    once (``match_id``, ``batting_team_id``). Team-innings without a top-7 sum are left out of the
    median. Raises ValueError when none has one, and the errors of top7_sums.
    """
    sums = top7_sums(theta, xi, team_innings)
    defined = sums[~np.isnan(sums)]
    if not len(defined):
        raise ValueError(
            f"Typical_Top7_Sum (Requirement 8, criterion 12) is undefined: none of the {len(sums):,} team-innings "
            "has an is_played player in batting positions 1-7 of its batting side in xi"
        )
    return float(np.median(defined))


def _positive_number(value: Any, what: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{what} must be a number; got {value!r}")
    number = float(value)
    if not (math.isfinite(number) and number > 0.0):
        raise ValueError(f"{what} must be a finite number above 0; got {value!r}")
    return number


def rbq(lineup: pd.DataFrame, theta: pd.DataFrame, top7: float) -> pd.DataFrame:
    """RBQ, the post-wicket RBQ and the next batter's theta_bat of each Delivery (criterion 13).

    ``lineup`` has one row per Delivery with ``non_striker_id`` and the Batters_To_Come slots
    ``btc_1`` ... ``btc_11`` of the ``state`` table (player ids in ``batting_order``, NO_PLAYER
    padding, all missing when the batting side has no ``is_played`` row); a ``view_rows`` frame
    has them. ``theta`` is the theta table of the fit that supplies the Delivery's values and
    ``top7`` that fit's Typical_Top7_Sum (above 0).

    Returns the RBQ_COLUMNS (float64) on a RangeIndex, in the row order of ``lineup``:

    * ``theta_next_batter``: theta_bat of the first filled slot (``btc_1``, the lowest
      ``batting_order``); NaN when Batters_To_Come is empty;
    * ``rbq``: (sum of theta_bat over Batters_To_Come + the non-striker's theta_bat) / top7;
    * ``rbq_post_wicket``: the same without the next batter (equal to ``rbq`` when
      Batters_To_Come is empty).

    All three are NaN on a row whose slots are missing, and ``rbq`` and ``rbq_post_wicket`` on a
    row whose ``non_striker_id`` is missing. Raises ValueError for a row with only some slots
    missing, a slot id below 0 other than NO_PLAYER, a player that theta lacks or a ``top7``
    that is not a finite number above 0.
    """
    _require(lineup, ("non_striker_id", *BTC_SLOT_COLUMNS), "lineup")
    divisor = _positive_number(top7, "top7 (Typical_Top7_Sum)")
    table = _Theta.of(theta, with_bowl=False)
    n_rows, n_slots = len(lineup), len(BTC_SLOT_COLUMNS)
    slots = np.empty((n_rows, n_slots), dtype=np.int64)
    missing = np.empty((n_rows, n_slots), dtype=bool)
    for position, column in enumerate(BTC_SLOT_COLUMNS):
        slots[:, position], missing[:, position] = _whole(lineup, column, nullable=True, what="lineup")
    no_xi = missing.all(axis=1)
    partly = missing.any(axis=1) & ~no_xi
    if partly.any():
        raise ValueError(
            f"lineup has {_rows(int(partly.sum()))} with some Batters_To_Come slots missing and others present; the "
            "slots of a row are all missing (no playing XI) or all present, as the state table gives them"
        )
    filled = ~missing & (slots != NO_PLAYER)
    bad = ~missing & (slots < 0) & (slots != NO_PLAYER)
    if bad.any():
        raise ValueError(f"lineup holds {int(bad.sum()):,} Batters_To_Come slot id(s) below 0 other than {NO_PLAYER}")
    values = np.zeros((n_rows, n_slots), dtype=np.float64)
    values[filled] = table.lookup(table.bat, slots[filled], "Batters_To_Come players")
    has_next = filled.any(axis=1)
    first = np.argmax(filled, axis=1)
    at = np.arange(n_rows)
    theta_next = np.where(has_next, values[at, first], np.nan)
    others = filled.copy()
    others[at[has_next], first[has_next]] = False
    rest = np.where(others, values, 0.0).sum(axis=1)
    non_striker, no_non_striker = _whole(lineup, "non_striker_id", nullable=True, what="lineup")
    theta_non_striker = np.full(n_rows, np.nan)
    theta_non_striker[~no_non_striker] = table.lookup(table.bat, non_striker[~no_non_striker], "non-strikers")
    after_wicket = rest + theta_non_striker  # Batters_To_Come without the next batter, plus the non-striker
    whole_side = after_wicket + np.where(has_next, theta_next, 0.0)
    return pd.DataFrame(
        {
            "theta_next_batter": np.where(no_xi, np.nan, theta_next),
            "rbq": np.where(no_xi, np.nan, whole_side / divisor),
            "rbq_post_wicket": np.where(no_xi, np.nan, after_wicket / divisor),
        },
        columns=list(RBQ_COLUMNS),
    )


def team_strength_gap(sides: pd.DataFrame, theta: pd.DataFrame, xi: pd.DataFrame) -> np.ndarray:
    """Team_Strength_Gap of each Delivery: batting side's Team_Strength minus the fielding side's (criterion 14).

    ``sides`` has ``match_id``, ``batting_team_id`` and ``bowling_team_id`` per Delivery;
    ``theta`` is the theta table of the fit that supplies the Delivery's values (``player_id``,
    ``theta_bat``, ``theta_bowl``); ``xi`` the ``playing_eleven`` table. Team_Strength is
    PLAYERS_PER_SIDE (11) x the mean of theta_bat + theta_bowl over a side's distinct
    ``is_played`` players in the match, a player listed in several rows counting once (Requirement
    2, criterion 3: player theta values only, no team-level rating). The mean puts a side that
    lists an Impact Player (12 players) on the same 11-player scale as a side that lists 11, so a
    different number of listed players does not shift the gap. Both terms come from one per-side
    table, so swapping the sides negates the gap exactly: the innings-2 value is minus the
    innings-1 value.

    Returns a float64 array in the row order of ``sides``: NaN when either side has no
    ``is_played`` row in ``xi`` or a team id is missing. Raises ValueError when theta lacks an
    ``is_played`` player of a match in ``sides``, and the errors of a malformed column.
    """
    _require(sides, _SIDE_COLUMNS, "sides")
    table = _Theta.of(theta, with_bowl=True)
    match, _ = _whole(sides, "match_id", nullable=False, what="sides")
    batting, no_batting = _whole(sides, "batting_team_id", nullable=True, what="sides")
    bowling, no_bowling = _whole(sides, "bowling_team_id", nullable=True, what="sides")
    players = _in_matches(_played_players(xi), match)
    strength = None
    if not players.empty:
        ids = players["player_id"].to_numpy()
        assert table.bowl is not None
        both = table.lookup(table.bat, ids, "is_played players") + table.lookup(table.bowl, ids, "is_played players")
        strength = PLAYERS_PER_SIDE * _per_side(players, both, mean=True)
    return _side_values(strength, match, batting, no_batting) - _side_values(strength, match, bowling, no_bowling)


def match_day_conditions(deliveries: pd.DataFrame, pred_runs: Any) -> np.ndarray:
    """Match_Day_Conditions of each Delivery, in runs per over (criteria 15 and 16).

    ``deliveries`` has ``match_id``, ``innings``, ``innings_ball_num``, ``cls`` and ``runs`` (bat
    runs) and holds every innings 1-2 Delivery of each of its matches, in any order;
    ``pred_runs`` the M1-predicted bat runs of each row from the fit that supplies the row's
    theta values (finite on Legal_Balls; other rows are not read and may be NaN).

    For each Delivery, over the Legal_Balls of the same match that come strictly before it in
    the Requirement 5 order (``match_id``, ``innings``, ``innings_ball_num``): 6 x (sum of bat
    runs - sum of predicted bat runs) / the number of those Legal_Balls; 0 when there is none.
    Returns a float64 array in the row order of ``deliveries``. Raises ValueError for a row
    outside innings 1 and 2, a ``pred_runs`` of the wrong length, a non-finite prediction on a
    Legal_Ball, and the errors of a malformed column.
    """
    _require(deliveries, _MDC_COLUMNS, "deliveries")
    what = "deliveries"
    match, _ = _whole(deliveries, "match_id", nullable=False, what=what)
    innings, _ = _whole(deliveries, "innings", nullable=False, what=what)
    ball, _ = _whole(deliveries, "innings_ball_num", nullable=False, what=what)
    cls, _ = _whole(deliveries, "cls", nullable=False, what=what)
    runs, _ = _whole(deliveries, "runs", nullable=False, what=what)
    n_rows = len(match)
    outside = ~np.isin(innings, STATE_INNINGS)
    if outside.any():
        raise ValueError(
            f"deliveries holds {_rows(int(outside.sum()))} outside innings 1 and 2; Match_Day_Conditions covers "
            "innings 1 and 2 only"
        )
    if isinstance(pred_runs, (str, bytes, pd.DataFrame)):
        raise TypeError("pred_runs must be a 1-D array of predicted bat runs, one per row of deliveries")
    if isinstance(pred_runs, pd.Series):
        pred = pred_runs.to_numpy(dtype=np.float64, na_value=np.nan)
    else:
        pred = np.asarray(pred_runs, dtype=np.float64)
    if pred.ndim != 1 or len(pred) != n_rows:
        raise ValueError(f"pred_runs must hold one value per row of deliveries ({n_rows:,}); got shape {pred.shape}")
    legal = cls == int(DeliveryClass.LEGAL)
    unusable = legal & ~np.isfinite(pred)
    if unusable.any():
        raise ValueError(f"pred_runs is missing or not finite on {int(unusable.sum()):,} Legal_Ball(s)")
    order = np.lexsort((ball, innings, match))
    groups = match[order]
    sums = pd.DataFrame(
        {
            "residual": np.where(legal, runs - np.where(legal, pred, 0.0), 0.0)[order],
            "legal_balls": legal[order].astype(np.float64),
        }
    )
    # Running sums within each match, shifted one row down: the sums over the earlier Deliveries.
    earlier = sums.groupby(groups, sort=False).cumsum().groupby(groups, sort=False).shift(1, fill_value=0.0)
    count = earlier["legal_balls"].to_numpy()
    total = earlier["residual"].to_numpy()
    value = np.zeros(n_rows)
    seen = count > 0
    value[seen] = BALLS_PER_OVER * total[seen] / count[seen]
    out = np.empty(n_rows)
    out[order] = value
    return out


# ------------------------------------------------------------------------------- the view rows

def view_rows(balls: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
    """One row per ``state`` row (every Delivery of innings 1 and 2), in the ``state`` order, for the views.

    ``balls`` is the ``balls`` table of the ``reconstruct`` checkpoint and ``state`` the ``state``
    table. Returns ``row`` and the VIEW_INPUT_COLUMNS on a RangeIndex: the ball columns with the
    dtypes of ``balls``, and ``cell``, ``wickets`` and the Batters_To_Come slots with those of
    ``state``. Raises ValueError when a column is absent, ``balls`` repeats a ``row`` or lacks a
    ``state`` row.
    """
    _require(balls, ("row", *_VIEW_BALL_COLUMNS), "balls")
    _require(state, ("row", *_VIEW_STATE_COLUMNS), "state")
    index = pd.Index(balls["row"].to_numpy())
    if not index.is_unique:
        raise ValueError("balls must hold each row once; pass the balls table of the reconstruct checkpoint")
    at = index.get_indexer(state["row"].to_numpy())
    absent = at < 0
    if absent.any():
        raise ValueError(f"balls has no Delivery for {_rows(int(absent.sum()))} of state")
    return pd.concat(
        [
            balls.iloc[at][["row", *_VIEW_BALL_COLUMNS]].reset_index(drop=True),
            state[list(_VIEW_STATE_COLUMNS)].reset_index(drop=True),
        ],
        axis=1,
    )


def _group_coefficients(coef: np.ndarray, encoders: OneHotEncoders, groups: Sequence[str]) -> np.ndarray:
    """``coef`` with every column outside ``groups`` set to 0."""
    keep = np.zeros(len(coef), dtype=bool)
    for group in groups:
        keep[encoders.group_slice(group)] = True
    return np.where(keep, coef, 0.0)


def view_quantities(
    fit: RidgeFit,
    encoders: OneHotEncoders,
    top7: float,
    rows: pd.DataFrame,
    xi: pd.DataFrame,
    *,
    X: Any = None,
) -> pd.DataFrame:
    """The M1 quantities (VIEW_COLUMNS) of ``rows`` from one fit (module docstring, M1-derived quantities).

    ``fit`` is the M1 fit that supplies the rows' values, ``encoders`` the encoders of its design
    matrix and ``top7`` its Typical_Top7_Sum; ``rows`` a ``view_rows`` frame that holds every innings
    1-2 Delivery of each of its matches; ``xi`` the ``playing_eleven`` table. ``X``, when given, is
    ``design_matrix(rows, encoders)``, so a caller that has it saves the rebuild.

    Returns the VIEW_COLUMNS (float64) on a RangeIndex, one row per row of ``rows``:
    ``theta_striker`` and ``theta_non_striker`` (theta_bat), ``theta_bowler`` (theta_bowl),
    ``theta_next_batter``, ``rbq`` and ``rbq_post_wicket`` (rbq), ``scoring_env`` and
    ``venue_effect`` (runs regression), ``tsg`` (team_strength_gap), ``match_day``
    (match_day_conditions with this fit's predictions) and ``m1_pred_runs``. Raises the errors of
    the functions it calls, and ValueError when ``X`` does not have one row per row of ``rows``.
    """
    _check_fit(fit, encoders)
    _require(rows, VIEW_INPUT_COLUMNS, "rows")
    matrix = design_matrix(rows, encoders) if X is None else _as_csr(X, n_columns=fit.n_columns)
    if matrix.shape[0] != len(rows):
        raise ValueError(f"X has {matrix.shape[0]:,} rows but rows has {len(rows):,}; pass design_matrix(rows, encoders)")
    striker, _ = _whole(rows, "striker_id", nullable=False, what="rows")
    non_striker, no_non_striker = _whole(rows, "non_striker_id", nullable=True, what="rows")
    bowler, _ = _whole(rows, "bowler_id", nullable=False, what="rows")
    slot_ids = [_whole(rows, column, nullable=True, what="rows") for column in BTC_SLOT_COLUMNS]
    named = [striker, non_striker[~no_non_striker], bowler, _played_players(xi)["player_id"].to_numpy()]
    named += [ids[~missing & (ids >= 0)] for ids, missing in slot_ids]
    theta = thetas(fit, np.unique(np.concatenate(named)), encoders)
    table = _Theta.of(theta, with_bowl=True)
    assert table.bowl is not None

    pred_runs = np.asarray(matrix @ fit.coef_runs, dtype=np.float64) + fit.intercept_runs
    environment = np.asarray(matrix @ _group_coefficients(fit.coef_runs, encoders, (ENVIRONMENT,)), dtype=np.float64)
    venue = np.asarray(matrix @ _group_coefficients(fit.coef_runs, encoders, (VENUE, COUNTRY)), dtype=np.float64)
    theta_non_striker = np.full(len(rows), np.nan)
    theta_non_striker[~no_non_striker] = table.lookup(table.bat, non_striker[~no_non_striker], "non-strikers")
    lineup = rbq(rows, theta, top7)
    return pd.DataFrame(
        {
            "theta_striker": table.lookup(table.bat, striker, "strikers"),
            "theta_non_striker": theta_non_striker,
            "theta_bowler": table.lookup(table.bowl, bowler, "bowlers"),
            "theta_next_batter": lineup["theta_next_batter"].to_numpy(),
            "scoring_env": environment,
            "venue_effect": venue,
            "rbq": lineup["rbq"].to_numpy(),
            "rbq_post_wicket": lineup["rbq_post_wicket"].to_numpy(),
            "tsg": team_strength_gap(rows, theta, xi),
            "match_day": match_day_conditions(rows, pred_runs),
            "m1_pred_runs": pred_runs,
        },
        columns=list(VIEW_COLUMNS),
    )


# ============================================================================ the skill Pipeline_Stage

#: Name of the Pipeline_Stage that fits M1, and of its checkpoint.
STAGE_NAME = "skill"

#: Model names of the M1 fits and the baseline fits in the FitRegistry (fit ids from leakage.fit_id_for).
MODEL = "m1"
BASELINE_MODEL = "m1_baseline"

#: Fit id of the all-fold M1 fit behind the top-theta lists (Requirement 8, criterion 17). It
#: trains on all D15 folds, excludes no fold and gives no output to any Delivery.
ALL_FOLDS_FIT_ID = f"{MODEL}_all_folds"

#: Checkpoint tables: every M1 fit's theta of every player, and one summary row per M1 fit.
THETA_TABLE = "m1_theta"
FIT_SUMMARY_TABLE = "m1_fits"
THETA_TABLE_COLUMNS: tuple[str, ...] = ("fit_id", "player_id", "theta_bat", "theta_bowl", "bat_balls", "bowl_balls")
FIT_SUMMARY_COLUMNS: tuple[str, ...] = (
    "fit_id", "sandbox_k", "inner_j", "excluded_folds", "train_folds", "n_train_rows", "n_train_matches",
    "penalty_runs", "penalty_wkt", "mean_runs", "mean_wkt", "runs_per_wicket", "typical_top7_sum",
    "n_top7_innings", "n_top7_missing",
)  # fmt: skip

#: Folder under the outputs folder for the stage's files (Requirement 8, criterion 17), and their names.
M1_FOLDER = Path("models") / "m1"
METRICS_FILE = "metrics.csv"
TOP_BATTERS_FILE = "top20_batters.csv"
TOP_BOWLERS_FILE = "top20_bowlers.csv"

#: Length of each top-theta list, fixed by Requirement 8, criterion 17 (not a Method_Constant).
TOP_LIST_SIZE = 20

#: Data-derived Method_Constants that the stage records (Requirement 2, criterion 8).
PENALTY_RUNS_CONSTANT = "skill.all_folds.penalty_runs"
PENALTY_WKT_CONSTANT = "skill.all_folds.penalty_wkt"
TOP7_CONSTANT = "skill.all_folds.typical_top7_sum"
PENALTY_RUNS_BY_FIT_CONSTANT = "skill.penalty_runs_by_fit"
PENALTY_WKT_BY_FIT_CONSTANT = "skill.penalty_wkt_by_fit"
TOP7_BY_FIT_CONSTANT = "skill.typical_top7_sum_by_fit"

_COMPONENT = "Skill_Model"  # the glossary component that produces the constants
_LOAD_STAGE = "load"
_XI_TABLE = "playing_eleven"  # the load checkpoint's tables of playing_eleven.csv and ball_by_ball.csv
_BALL_BY_BALL_TABLE = "ball_by_ball"
_FLOAT_FORMAT = "%.6f"
_M1_LABEL, _BASELINE_LABEL = "M1", "baseline"


def _shown(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _write_csv(path: Path, table: pd.DataFrame) -> None:
    """Write ``table`` to ``path`` through ``<file>.tmp`` renamed in place (UTF-8, ``\\n`` line ends)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        table.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n", float_format=_FLOAT_FORMAT)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _folds_text(values: Iterable[int]) -> str:
    return "/".join(str(fold) for fold in sorted(values))


def _training_team_innings(train: pd.DataFrame, balls: pd.DataFrame, train_fold: np.ndarray) -> pd.DataFrame:
    """Each team-innings of the training Legal_Balls once: ``match_id``, ``innings``, ``batting_team_id``, ``fold``."""
    at = pd.Index(balls["row"].to_numpy()).get_indexer(train["row"].to_numpy())
    frame = pd.DataFrame(
        {
            "match_id": train["match_id"].to_numpy(),
            "innings": train["innings"].to_numpy(),
            "batting_team_id": balls["batting_team_id"].iloc[at].reset_index(drop=True),
            "fold": train_fold,
        }
    )
    return frame.drop_duplicates(["match_id", "innings", "batting_team_id"]).reset_index(drop=True)


def _most_frequent_names(frame: pd.DataFrame, id_column: str, name_column: str) -> pd.Series:
    """Each player's most frequent ``name_column`` value (ties: the first name in sort order), by player id."""
    _require(frame, (id_column, name_column), "ball_by_ball")
    ids, no_id = _whole(frame, id_column, nullable=True, what="ball_by_ball")
    names, no_name = _text(frame, name_column, nullable=True, what="ball_by_ball")
    keep = ~no_id & ~no_name
    counts = pd.DataFrame({"player_id": ids[keep], "name": names[keep]}).value_counts().rename("n").reset_index()
    first = counts.sort_values(["player_id", "n", "name"], ascending=[True, False, True], kind="stable")
    first = first.drop_duplicates("player_id")
    return pd.Series(first["name"].to_numpy(dtype=object), index=pd.Index(first["player_id"].to_numpy(dtype=np.int64)))


def _top_list(theta: pd.DataFrame, names: pd.Series, theta_column: str, balls_column: str) -> pd.DataFrame:
    """The TOP_LIST_SIZE players with the highest ``theta_column`` among those with training Legal_Balls in the Role."""
    rated = theta.iloc[np.flatnonzero(theta[balls_column].to_numpy() > 0)]
    order = np.lexsort((rated["player_id"].to_numpy(), -rated[theta_column].to_numpy()))
    top = rated.iloc[order[:TOP_LIST_SIZE]]
    ids = top["player_id"].to_numpy(dtype=np.int64)
    return pd.DataFrame(
        {
            "rank": np.arange(1, len(top) + 1, dtype=np.int64),
            "player_id": ids,
            "player_name": pd.array(names.reindex(ids).to_numpy(dtype=object), dtype="string"),
            theta_column: top[theta_column].to_numpy(dtype=np.float64),
            "legal_balls": top[balls_column].to_numpy(dtype=np.int64),
        }
    )


def _pool_counts(train: pd.DataFrame) -> dict[str, int]:
    """Distinct batters and bowlers with training Legal_Balls in both Pools and in each Pool (criterion 10)."""
    comp, _ = _text(train, "comp_type", nullable=False, what="train")
    ipl = comp == reconstruct.COMP_IPL  # the international Pool is T20I and A_Team
    counts: dict[str, int] = {}
    for role, column in (("batters", "striker_id"), ("bowlers", "bowler_id")):
        ids, _ = _whole(train, column, nullable=False, what="train")
        in_ipl, in_international = np.unique(ids[ipl]), np.unique(ids[~ipl])
        counts[f"{role}_in_both_pools"] = len(np.intersect1d(in_ipl, in_international, assume_unique=True))
        counts[f"{role}_in_ipl_pool"] = len(in_ipl)
        counts[f"{role}_in_international_pool"] = len(in_international)
    return counts


def _metrics_table(
    accuracy: Mapping[str, tuple[float, float]], pools: Mapping[str, int], n_rows: int, prob_clip: Sequence[float]
) -> tuple[pd.DataFrame, list[str]]:
    """The metrics table (criteria 8-11) and its warnings, in the order they appear in the table."""
    describe = {
        "rmse_bat_runs": "pooled held-out RMSE of bat runs per training Legal_Ball",
        "log_loss_dismissal": (
            f"pooled held-out log loss of the Counted_Wicket target, predictions clipped to "
            f"[{prob_clip[0]:g}, {prob_clip[1]:g}] (models.m1.prob_clip)"
        ),
    }
    source = {
        _M1_LABEL: "Out_Of_Fold_Predictions of the 5 outer M1 fits",
        _BASELINE_LABEL: "Out_Of_Fold_Predictions of the 5 outer baseline fits, without batter and bowler indicators",
    }
    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    for metric, (m1_value, baseline_value) in accuracy.items():
        for label, value in ((_M1_LABEL, m1_value), (_BASELINE_LABEL, baseline_value)):
            warning = None
            if label == _M1_LABEL and not m1_value < baseline_value:
                warning = (
                    f"M1 is not better than the baseline on {metric}: M1 {m1_value:.6f}, baseline "
                    f"{baseline_value:.6f} (Requirement 8, criterion 9)"
                )
                warnings.append(warning)
            records.append(
                {
                    "metric": metric,
                    "model": label,
                    "value": f"{value:.6f}",
                    "n_legal_balls": str(n_rows),
                    "description": f"{describe[metric]}; {source[label]} (Requirement 8, criterion 8)",
                    "warning": warning,
                }
            )
    role_text = {"batters": ("batters", "striker_id", "batter"), "bowlers": ("bowlers", "bowler_id", "bowler")}
    for name, count in pools.items():
        role, scope = name.split("_in_", 1)
        plural, column, singular = role_text[role]
        where = {
            "both_pools": "both the IPL Pool and the international Pool (T20I and A_Team)",
            "ipl_pool": "the IPL Pool",
            "international_pool": "the international Pool (T20I and A_Team)",
        }[scope]
        warning = None
        if scope == "both_pools" and count == 0:
            warning = (
                f"theta values are not comparable across the Pools: no {singular} has a training Legal_Ball in both "
                "Pools (Requirement 8, criterion 11)"
            )
            warnings.append(warning)
        records.append(
            {
                "metric": name,
                "model": None,
                "value": str(count),
                "n_legal_balls": None,
                "description": (
                    f"distinct {plural} ({column}) with at least one training Legal_Ball in {where} "
                    "(Requirement 8, criterion 10)"
                ),
                "warning": warning,
            }
        )
    columns = ["metric", "model", "value", "n_legal_balls", "description", "warning"]
    return pd.DataFrame.from_records(records, columns=columns).astype(object), warnings


def _tsg_mismatches(tsg: np.ndarray, rows: pd.DataFrame) -> tuple[int, int]:
    """(matches whose gaps are not exactly antisymmetric between the innings, matches with both innings)."""
    match = rows["match_id"].to_numpy()
    innings = rows["innings"].to_numpy()
    frame = pd.DataFrame({"match_id": match, "innings": innings, "tsg": tsg})
    reference = frame.drop_duplicates("match_id").set_index("match_id")  # each match's first row, NaN kept
    at = reference.index.get_indexer(match)
    first_innings = reference["innings"].to_numpy()[at]
    first_value = reference["tsg"].to_numpy()[at]
    expected = np.where(innings == first_innings, first_value, -first_value)
    agree = (tsg == expected) | (np.isnan(tsg) & np.isnan(expected))
    bad = int(pd.Series(~agree).groupby(match).any().sum())
    both = int((frame.groupby("match_id")["innings"].nunique() > 1).sum())
    return bad, both


def _no_earlier_legal_ball(rows: pd.DataFrame) -> np.ndarray:
    """True for each row with no Legal_Ball of the same match before it (Match_Day_Conditions 0, criterion 16)."""
    match = rows["match_id"].to_numpy()
    order = np.lexsort((rows["innings_ball_num"].to_numpy(), rows["innings"].to_numpy(), match))
    legal = (rows["cls"].to_numpy() == int(DeliveryClass.LEGAL)).astype(np.int64)[order]
    inclusive = pd.Series(legal).groupby(match[order], sort=False).cumsum().to_numpy()
    flags = np.empty(len(rows), dtype=bool)
    flags[order] = (inclusive - legal) == 0
    return flags


def run_stage(ctx: StageContext) -> None:
    """The ``skill`` Pipeline_Stage (module docstring, "The skill Pipeline_Stage").

    Raises FitLeakageError or ViewLeakageError (exit code 4) when a fit or a view breaks the
    leakage guard (Requirement 7, criterion 12), DataError (exit code 3) for the data problems of
    the functions it calls, and ValueError or TypeError for a malformed Config value.
    """
    settings, store, log, registry = ctx.settings, ctx.store, ctx.log, ctx.registry
    n_folds = settings.get("design_choices.D15_n_folds")
    grid = settings.get("models.m1.ridge_grid")
    prob_clip = settings.get("models.m1.prob_clip")
    buckets = settings.get("design_choices.D17_wicket_buckets")
    plan = folds.sandbox_plan(n_folds)  # checks D15 before any table is read
    m1_plan = views.view_plan(MODEL, plan)
    outer = [spec for spec in plan if spec.kind == folds.OUTER]
    all_folds = tuple(range(1, n_folds + 1))

    balls = store.load_table(reconstruct.STAGE_NAME, reconstruct.BALLS_TABLE)
    state = store.load_table(STATE_STAGE, STATE_TABLE)
    xi = store.load_table(_LOAD_STAGE, _XI_TABLE)
    deliveries = store.load_table(_LOAD_STAGE, _BALL_BY_BALL_TABLE)
    fold_table = store.load_table(folds.STAGE_NAME, folds.FOLD_TABLE)

    # 1. Design matrices of the training Legal_Balls and of every view row, and the fold statistics.
    encoders = build_encoders(balls, wicket_buckets=buckets)
    train = legal_rows(balls, state, training_only=True)
    X_train = design_matrix(train, encoders)
    train_fold = folds.fold_of(train["match_id"], fold_table)
    rows = view_rows(balls, state)
    X_rows = design_matrix(rows, encoders)
    row_fold = folds.fold_of(rows["match_id"], fold_table)
    universe = player_universe(balls, xi)
    stats = fold_stats(X_train, train[RUNS_TARGET], train[WICKET_TARGET], train_fold)

    # 2. Register every fit before any fit runs (Requirement 7, criteria 6, 7, 9 and 12).
    train_match = train["match_id"].to_numpy()
    row_match = rows["match_id"].to_numpy()

    def trained_on(train_folds: Iterable[int]) -> np.ndarray:
        return np.isin(train_fold, np.fromiter(train_folds, dtype=np.int64))

    fit_sets: dict[str, tuple[int, ...]] = {}  # registered M1 fit id -> its training folds
    for spec in plan:
        mask = trained_on(spec.train_folds)
        record = registry.register_spec(
            MODEL,
            spec,
            np.unique(train_match[mask]),
            np.unique(row_match[row_fold == spec.predicted_fold]),
            n_train_rows=int(mask.sum()),
        )
        fit_sets[record.fit_id] = tuple(spec.train_folds)
    registry.register(
        MODEL, ALL_FOLDS_FIT_ID, (), np.unique(train_match), np.zeros(0, dtype=np.int64), n_train_rows=len(train)
    )
    fit_sets[ALL_FOLDS_FIT_ID] = all_folds
    for spec in outer:
        mask = trained_on(spec.train_folds)
        registry.register_spec(
            BASELINE_MODEL,
            spec,
            np.unique(train_match[mask]),
            np.unique(train_match[train_fold == spec.sandbox_k]),
            n_train_rows=int(mask.sum()),
        )

    # 3. One ridge per distinct training set, and the outer baselines, from one RidgeCache.
    cache = RidgeCache()
    training_sets = sorted(set(fit_sets.values()), key=lambda folds_: (-len(folds_), folds_))
    fits = {folds_: fit_ridge(stats, folds_, grid, cache=cache) for folds_ in training_sets}
    baselines = {spec.sandbox_k: fit_baseline(stats, spec.train_folds, grid, encoders, cache=cache) for spec in outer}

    team_innings = _training_team_innings(train, balls, train_fold)
    theta_of: dict[tuple[int, ...], pd.DataFrame] = {}
    top7_of: dict[tuple[int, ...], float] = {}
    top7_count: dict[tuple[int, ...], tuple[int, int]] = {}
    for folds_ in training_sets:
        theta = thetas(fits[folds_], universe, encoders)
        chosen = team_innings.iloc[np.flatnonzero(np.isin(team_innings["fold"].to_numpy(), folds_))]
        sums = top7_sums(theta, xi, chosen)
        theta_of[folds_] = theta
        top7_of[folds_] = typical_top7_sum(theta, xi, chosen)
        top7_count[folds_] = (int((~np.isnan(sums)).sum()), int(np.isnan(sums).sum()))

    # 4. The M1 quantities of every view row from the fit that serves it, then the views V_1 ... V_D15.
    parts = {}
    for fold in all_folds:
        at = np.flatnonzero(row_fold == fold)
        parts[fold] = (rows.iloc[at].reset_index(drop=True), X_rows[at])
    values: dict[str, dict[str, np.ndarray]] = {name: {} for name in VIEW_COLUMNS}
    for spec in plan:
        fit_id = leakage.fit_id_for(MODEL, spec)
        folds_ = fit_sets[fit_id]
        part_rows, part_X = parts[spec.predicted_fold]
        quantities = view_quantities(fits[folds_], encoders, top7_of[folds_], part_rows, xi, X=part_X)
        for name in VIEW_COLUMNS:
            values[name][fit_id] = quantities[name].to_numpy()
    sources = {name: views.FoldValues(views.M1_FIT, m1_plan, values[name]) for name in VIEW_COLUMNS}
    row_ids = rows["row"].to_numpy()
    missing_counts: dict[str, list[int]] = {name: [] for name in ("rbq", "rbq_post_wicket", "theta_next_batter", "tsg")}
    asymmetric: list[int] = []
    n_both_innings = 0
    for k in all_folds:
        view = views.build_view(k, row_fold, sources, rows=row_ids)
        views.check_view(k, view, row_fold, registry)  # before any later fit reads V_k
        store.save_table(STAGE_NAME, views.view_table(k), view)
        for name, counts in missing_counts.items():
            counts.append(int(view[name].isna().sum()))
        bad, n_both_innings = _tsg_mismatches(view["tsg"].to_numpy(), rows)
        asymmetric.append(bad)
        del view

    # 5. Metrics (criteria 8-11) and the top-theta lists of the all-fold fit (criterion 17).
    outer_fits = {spec.sandbox_k: fits[tuple(spec.train_folds)] for spec in outer}
    y_runs = train[RUNS_TARGET].to_numpy(dtype=np.float64)
    y_wkt = train[WICKET_TARGET].to_numpy(dtype=bool)
    m1_runs, m1_wkt = out_of_fold_predictions(outer_fits, X_train, train_fold)
    base_runs, base_wkt = out_of_fold_predictions(baselines, X_train, train_fold)
    accuracy = {
        "rmse_bat_runs": (pooled_rmse(y_runs, m1_runs), pooled_rmse(y_runs, base_runs)),
        "log_loss_dismissal": (
            clipped_log_loss(y_wkt, m1_wkt, prob_clip),
            clipped_log_loss(y_wkt, base_wkt, prob_clip),
        ),
    }
    pools = _pool_counts(train)
    metrics, metric_warnings = _metrics_table(accuracy, pools, len(train), prob_clip)
    outdir = settings.path("outputs") / M1_FOLDER
    _write_csv(outdir / METRICS_FILE, metrics)
    all_theta = theta_of[all_folds]
    batters = _top_list(all_theta, _most_frequent_names(deliveries, "striker_id", "striker_name"), "theta_bat", "bat_balls")
    bowlers = _top_list(all_theta, _most_frequent_names(deliveries, "bowler_id", "bowler_name"), "theta_bowl", "bowl_balls")
    _write_csv(outdir / TOP_BATTERS_FILE, batters)
    _write_csv(outdir / TOP_BOWLERS_FILE, bowlers)

    # 6. Per-fit tables, the fit registry and the data-derived Method_Constants.
    theta_parts, summary = [], []
    for fit_id, folds_ in fit_sets.items():
        theta = theta_of[folds_]
        theta_parts.append(
            pd.DataFrame(
                {
                    "fit_id": np.full(len(theta), fit_id, dtype=object),
                    **{column: theta[column].to_numpy() for column in THETA_TABLE_COLUMNS[1:]},
                }
            )
        )
        record, fit = registry.get(fit_id), fits[folds_]
        summary.append(
            {
                "fit_id": fit_id,
                "sandbox_k": record.sandbox_k,
                "inner_j": record.inner_j,
                "excluded_folds": sorted(record.excluded_folds),
                "train_folds": list(folds_),
                "n_train_rows": fit.n_rows,
                "n_train_matches": len(record.train_match_ids),
                "penalty_runs": fit.penalty_runs,
                "penalty_wkt": fit.penalty_wkt,
                "mean_runs": fit.mean_runs,
                "mean_wkt": fit.mean_wkt,
                "runs_per_wicket": fit.runs_per_wicket,
                "typical_top7_sum": top7_of[folds_],
                "n_top7_innings": top7_count[folds_][0],
                "n_top7_missing": top7_count[folds_][1],
            }
        )
    theta_table = pd.concat(theta_parts, ignore_index=True)
    theta_table["fit_id"] = theta_table["fit_id"].astype("str")
    fit_table = pd.DataFrame.from_records(summary, columns=list(FIT_SUMMARY_COLUMNS))
    fit_table = fit_table.astype({"fit_id": "str", "sandbox_k": "Int32", "inner_j": "Int32"})
    fit_table["excluded_folds"] = pd.Series([item["excluded_folds"] for item in summary], dtype=object)
    fit_table["train_folds"] = pd.Series([item["train_folds"] for item in summary], dtype=object)
    store.save_table(STAGE_NAME, THETA_TABLE, theta_table)
    store.save_table(STAGE_NAME, FIT_SUMMARY_TABLE, fit_table)
    n_saved = registry.save(store, STAGE_NAME)

    full = fits[all_folds]
    ctx.constants.record(PENALTY_RUNS_CONSTANT, full.penalty_runs, _COMPONENT)
    ctx.constants.record(PENALTY_WKT_CONSTANT, full.penalty_wkt, _COMPONENT)
    ctx.constants.record(TOP7_CONSTANT, top7_of[all_folds], _COMPONENT)
    sandbox_ids = [leakage.fit_id_for(MODEL, spec) for spec in plan]
    ctx.constants.record(PENALTY_RUNS_BY_FIT_CONSTANT, {i: fits[fit_sets[i]].penalty_runs for i in sandbox_ids}, _COMPONENT)
    ctx.constants.record(PENALTY_WKT_BY_FIT_CONSTANT, {i: fits[fit_sets[i]].penalty_wkt for i in sandbox_ids}, _COMPONENT)
    ctx.constants.record(TOP7_BY_FIT_CONSTANT, {i: top7_of[fit_sets[i]] for i in sandbox_ids}, _COMPONENT)

    _log_stage(
        ctx,
        _StageSummary(
            plan_size=len(plan),
            n_outer=len(outer),
            training_sets=training_sets,
            fits=fits,
            all_folds=all_folds,
            top7_of=top7_of,
            top7_count=top7_count,
            n_decompositions=cache.decompositions,
            n_train=len(train),
            n_columns=encoders.n_columns,
            n_players=len(universe),
            rows=rows,
            missing_counts=missing_counts,
            asymmetric=asymmetric,
            n_both_innings=n_both_innings,
            accuracy=accuracy,
            pools=pools,
            metric_warnings=metric_warnings,
            outdir=outdir,
            n_saved=n_saved,
            n_theta_rows=len(theta_table),
            batters=batters,
            bowlers=bowlers,
        ),
    )


@dataclass(frozen=True, eq=False)
class _StageSummary:
    """What the Run_Log reports about one run of the stage."""

    plan_size: int
    n_outer: int
    training_sets: list[tuple[int, ...]]
    fits: Mapping[tuple[int, ...], RidgeFit]
    all_folds: tuple[int, ...]
    top7_of: Mapping[tuple[int, ...], float]
    top7_count: Mapping[tuple[int, ...], tuple[int, int]]
    n_decompositions: int
    n_train: int
    n_columns: int
    n_players: int
    rows: pd.DataFrame
    missing_counts: Mapping[str, list[int]]
    asymmetric: list[int]
    n_both_innings: int
    accuracy: Mapping[str, tuple[float, float]]
    pools: Mapping[str, int]
    metric_warnings: list[str]
    outdir: Path
    n_saved: int
    n_theta_rows: int
    batters: pd.DataFrame
    bowlers: pd.DataFrame


def _per_view(counts: Sequence[int]) -> str:
    """``"N in every view"`` when the views agree, else the count of each view."""
    if len(set(counts)) == 1:
        return f"{counts[0]:,} in every view"
    return f"V_1-V_{len(counts)} " + "/".join(f"{count:,}" for count in counts)


def _log_stage(ctx: StageContext, s: _StageSummary) -> None:
    log = ctx.log
    full = s.fits[s.all_folds]
    sandbox = [folds_ for folds_ in s.training_sets if folds_ != s.all_folds]
    penalties = sorted({(s.fits[f].penalty_runs, s.fits[f].penalty_wkt) for f in sandbox})
    log.info(
        f"fitted M1 (Requirement 8, criteria 1-3) on {s.n_train:,} training Legal_Balls with {s.n_columns:,} design "
        f"columns: {s.plan_size} sandbox fits of the fold-sandbox plan ({s.n_outer} outer, "
        f"{s.plan_size - s.n_outer} inner, which share {len(sandbox) - s.n_outer} distinct training sets) and the "
        f"all-fold fit, {len(s.training_sets)} distinct ridges in all, plus {s.n_outer} outer baseline fits; "
        f"{s.n_decompositions} eigendecompositions. Penalties (runs, dismissal): all-fold "
        f"({full.penalty_runs:g}, {full.penalty_wkt:g}); sandbox fits "
        + ", ".join(f"({runs:g}, {wkt:g})" for runs, wkt in penalties),
        stage=STAGE_NAME,
    )
    values = [s.top7_of[f] for f in sandbox]
    used, left_out = s.top7_count[s.all_folds]  # the all-fold fit sees every training team-innings once
    log.info(
        f"Typical_Top7_Sum (Requirement 8, criterion 12): all-fold fit {s.top7_of[s.all_folds]:.4f}, the median over "
        f"{used:,} training team-innings; sandbox fits {min(values):.4f} to {max(values):.4f}. {left_out:,} of the "
        f"{used + left_out:,} training team-innings have no is_played player with batting_order 1-7 in "
        "playing_eleven and are left out of the medians",
        stage=STAGE_NAME,
    )
    rows = s.rows
    slot_one = rows[BTC_SLOT_COLUMNS[0]]
    no_xi = int(slot_one.isna().sum())
    empty = int((slot_one == NO_PLAYER).fillna(False).sum())
    no_non_striker = int(rows["non_striker_id"].isna().sum())
    first_ball = int(_no_earlier_legal_ball(rows).sum())
    log.info(
        f"built the M1 columns of views V_1-V_{len(s.all_folds)} ({len(rows):,} Deliveries of innings 1 and 2 each; "
        f"fold-i rows of V_k from M1(-{{i,k}}), fold-k rows from M1(-k)) and checked each with check_view "
        f"(Requirement 7, criteria 7, 11 and 12). Missing values, never 0: rbq {_per_view(s.missing_counts['rbq'])} "
        f"({no_xi:,} Deliveries whose batting side has no is_played row, {no_non_striker:,} without a non-striker); "
        f"rbq_post_wicket {_per_view(s.missing_counts['rbq_post_wicket'])}; theta_next_batter "
        f"{_per_view(s.missing_counts['theta_next_batter'])} ({empty:,} Deliveries with an empty Batters_To_Come); "
        f"tsg {_per_view(s.missing_counts['tsg'])} (missing when a side has no is_played row). match_day is 0 on the "
        f"{first_ball:,} Deliveries with no earlier Legal_Ball in the match (Requirement 8, criterion 16)",
        stage=STAGE_NAME,
    )
    if any(s.asymmetric):
        log.warning(
            f"Team_Strength_Gap of innings 2 is not exactly minus that of innings 1 in some matches (Requirement 8, "
            f"criterion 14), because their innings do not swap the batting and fielding sides: "
            f"{_per_view(s.asymmetric)}",
            stage=STAGE_NAME,
        )
    else:
        log.info(
            f"Team_Strength_Gap of innings 2 is exactly minus that of innings 1 in all {s.n_both_innings:,} matches "
            "with both innings, in every view (Requirement 8, criterion 14)",
            stage=STAGE_NAME,
        )
    (rmse_m1, rmse_base), (ll_m1, ll_base) = s.accuracy["rmse_bat_runs"], s.accuracy["log_loss_dismissal"]
    log.info(
        f"M1 against the baseline without player indicators, pooled over the Out_Of_Fold_Predictions of all "
        f"{len(s.all_folds)} folds (Requirement 8, criterion 8): RMSE of bat runs {rmse_m1:.5f} vs {rmse_base:.5f}; "
        f"log loss of dismissals {ll_m1:.5f} vs {ll_base:.5f}",
        stage=STAGE_NAME,
    )
    p = s.pools
    log.info(
        f"players with training Legal_Balls in both Pools (Requirement 8, criterion 10): {p['batters_in_both_pools']:,} "
        f"batters and {p['bowlers_in_both_pools']:,} bowlers (IPL Pool {p['batters_in_ipl_pool']:,} batters, "
        f"{p['bowlers_in_ipl_pool']:,} bowlers; international Pool {p['batters_in_international_pool']:,} batters, "
        f"{p['bowlers_in_international_pool']:,} bowlers)",
        stage=STAGE_NAME,
    )
    shown = _shown(s.outdir, ctx.root)
    for warning in s.metric_warnings:
        log.warning(f"{warning}; see {shown}/{METRICS_FILE}", stage=STAGE_NAME)
    for label, table, column in (("batters", s.batters, "theta_bat"), ("bowlers", s.bowlers, "theta_bowl")):
        leaders = ", ".join(
            f"{name} {value:.3f}" for name, value in zip(table["player_name"].head(3).tolist(), table[column].head(3).tolist())
        )
        log.info(f"top {label} by {column} in the all-fold fit (Requirement 8, criterion 17): {leaders}, ...", stage=STAGE_NAME)
    log.info(
        f"wrote {shown}/{METRICS_FILE}, {shown}/{TOP_BATTERS_FILE} and {shown}/{TOP_BOWLERS_FILE}; saved the views "
        f"view_1-view_{len(s.all_folds)}, {THETA_TABLE} ({s.n_theta_rows:,} rows: {s.n_players:,} players x every M1 "
        f"fit id), {FIT_SUMMARY_TABLE} and the fit registry ({s.n_saved} fits)",
        stage=STAGE_NAME,
    )
