# Feature: pressure-rating-system, Property 14: For any synthetic match, any Delivery index i and any replacement of the Outcome_Columns of Delivery i and every later Delivery with values copied from other Deliveries (encoding fixed), the state features of Delivery i, its pillar components and its batter- and bowler-Role PI inputs are identical to those computed from the unmodified match.
"""Property 14: pre-ball state and PI use only the past (Requirement 6, criteria 1 and 15;
Requirement 13, criterion 6).

**Validates: Requirements 6.1, 6.15, 13.6**

Task 6.5 checked the state features. Task 15.6 adds the Pillar_Components of both Roles and the
batter- and bowler-Role PI inputs as further builders of ``FEATURE_BUILDERS``; the example, the
replacement and the comparison rule stay as they were.

Each example is one synthetic match:

* ``tests.strategies.scorecard_match`` builds a two-innings match of up to 5 Overs per innings
  under one Extras_Encoding, with its ``matches`` row; most super-over wins also carry
  Super_Over_Innings 3 and 4. The row gets the match columns that ``match_columns`` reads beyond
  those (competition, date, knock-out flag and venue), with fixed values.
* The playing XI of each side lists every player who appears for that side in the Deliveries: as
  striker or non-striker when it bats, and as bowler when it bowls. The batting order follows the
  player ids, which the generator numbers by batting slot. One player may be flagged as the
  Impact Player (``is_impact`` = 1) and a teammate as the player he replaced (``is_impact_off`` = 1).
* The cut Delivery i is any Delivery of innings 1 or 2, taken by its position in the
  Requirement 5 order. The Outcome_Columns of i and of every later Delivery of the match (super
  overs included) are replaced by the Outcome_Columns of a donor Delivery of the same match. The
  ten columns are copied from the donor as one block, so each replaced row is still a Delivery
  the generator could have produced. The donor of i has Outcome_Columns that differ from those of
  i whenever the match has such a Delivery, so the cut Delivery's own outcome always changes; the
  donors of later Deliveries are any Deliveries of the match.

The original and the modified Deliveries then go through the same pipeline steps, with every
Config value, the Extras_Encoding and every model input that does not come from the match held
fixed (criterion 15):

1. the ``reconstruct`` stage's ``balls`` table, assembled as ``reconstruct.run_stage`` does it:
   ``order_deliveries``, ``classify``, ``wicket_flags`` with the Counted_Wicket codes 1-6 and 9
   (the Retired_Hurt code 7 does not count), the team runs of the match's encoding from
   ``ENCODINGS``, ``free_hit_flags``, ``dismissed_batter_flags``, ``delivery_scheduled_balls``, the
   Target from ``innings_aggregates`` and ``targets``, ``match_columns`` and ``balls_table``. No
   match is excluded: ``excluded`` is a match-level flag that the State_Builder does not read;
2. the ``audit`` stage's ``modal_bowling_type`` table (``bowling_type_stability``). It reads
   ``bowling_type``, which is not an Outcome_Column, so it is computed once and shared;
3. the ``state`` stage's ``state`` table from ``build_state``, with D6 and D8 read from the Config;
4. the Fold_Table, which puts the match in fold FOLD (1), and view V_1, the model-derived values
   that the ``skill`` and ``remaining_runs`` stages give each Delivery. The real functions build
   it from fixed stand-ins for the fits, which in a Run are trained on other matches:

   * M1: a theta table set from each player's id (``_theta_table``), Typical_Top7_Sum from the
     XI of both sides (``skill.typical_top7_sum``), fixed Scoring_Environment and Venue_Effect,
     and predicted bat runs from the striker, the bowler, the innings x Phase cell and the wickets
     fallen. ``skill.rbq`` gives RBQ, the post-wicket RBQ and the next batter's theta_bat,
     ``skill.team_strength_gap`` the Team_Strength_Gap and ``skill.match_day_conditions`` the
     Match_Day_Conditions;
   * M2: ``_FirstInningsM2``, a deterministic first-innings fit in which every M2 feature moves
     the prediction, with fixed par means and par bias (PAR_MEANS, PAR_BIAS).
     ``remaining_runs.m2_features``, ``first_innings_par`` and ``achievable_rate`` give Par_Score,
     Par_Score at L + 1, Par_Rate and Achievable_Rate, and F19 is Par_Score - score in innings 1
     and Required_Rate - Achievable_Rate in innings 2, as the ``remaining_runs`` stage computes it;

5. every builder in ``FEATURE_BUILDERS``:

   ===================== ======================================================================
   Builder               Values
   ===================== ======================================================================
   state features        the ``state`` table (Requirement 6, criteria 1-14)
   view values           view V_1 (step 4)
   Pillar_Components     ``pressure.pillar_components``: the 13 components of both Roles (c19 to
                         c22b for the batter Role, c23 to c26 for the bowler Role) and
                         c25b_extras_free, wired as the ``pressure`` stage wires them: the view
                         values from ``own_fold_view``, and the F20(a) means from
                         ``train_wicket_means`` over the match's rows (fold 1) and TRAINING_ROWS
                         of other matches (folds 2-5), read with ``wicket_means_for``. The
                         fold-1 means therefore come from the other matches only
   pre-ball M3 features  ``win_prob.m3_features`` of each Delivery's innings
   Post_Outcome_States   ``pressure.post_outcome_states`` with fixed Class_Run_Values
                         (CLASS_RUN_VALUES), the stand-in M2 fit and its par tables: every column
                         of the 9 states of a Delivery, side by side (``r4.score`` is the score
                         after a four)
   pre-ball M4 features  ``outcome.m4_features`` with ``outcome_rows``; the categorical
                         ``phase`` and ``hand_x_style`` are compared by their labels
   ===================== ======================================================================

Why these are the PI inputs of both Roles (Requirement 13, criterion 6)
    A Live_Ball's PI for a Role is 100 x [alpha x rank(LI) + (1 - alpha) x rank(Stress)]. The
    unscaled LI is the sum over the 9 classes of P(class) x |WP(Post_Outcome_State) - WP(pre-ball
    state)|. P is the fixed M4 Primary_Model on the pre-ball M4 features with theta_bat and
    theta_bowl both set to the fold's Average_Skill, a constant. WP(pre-ball state) is the fixed
    M3 Primary_Model, or a terminal rule, on the pre-ball M3 features, which hold every column
    that the terminal rules read; it also sets the Live_Ball flag. WP(Post_Outcome_State) is the
    same on the state's columns. So identical rows give an identical unscaled LI, which the batter
    Role and the bowler Role share (Requirement 11, criterion 11), and an identical Live_Ball
    flag. The batter Role's Stress is built from c19 to c22b and the bowler Role's from c23 to
    c26 (c25b_extras_free for the Extras_Free_PI), so identical components give both Roles
    identical Stress inputs. The replaced block holds Delivery i's own ``runs``,
    ``extras_2_runs``, ``is_wicket`` and ``dismissal_type_id`` (PI_UNREAD_COLUMNS), so the
    identity also shows that no PI input reads those values of the current Delivery. No builder
    calls the Extras_Model, and none of the Pressure_Engine functions used takes an Extras_Model
    output.

What the identity does not cover
    * The steps that compare a Delivery with all others: the pooled mean that scales the LI and
      the Percentile_Ranks of the components, pillars, LI, Stress and PI, with the D3 cut. By
      Requirement 11, criterion 10, Requirement 12, criterion 12 and Requirement 13, criteria 2
      and 4, they rank each Delivery's past-only values against those of every Delivery in the
      reference set, later Deliveries of the same match included.
    * Role membership. A Wide is not a Batter_Delivery and has no batter-Role PI (Requirement 13,
      criteria 2 and 11), and whether Delivery i is a Wide reads its own ``extras_type``, which
      criterion 6 does not list. The replacement can make Delivery i a Wide or stop it being one,
      so the Role masks are not compared.
    * The fits themselves. The theta table, the M1 predictions, the M2 fit with its par tables
      and the Class_Run_Values are fixed stand-ins, and the M3 and M4 Primary_Models are not
      called: the test compares their inputs. That each fit trains on other folds only is checked
      by Properties 19 and 20 (leakage guard, sandbox views).

Each builder returns a table with one row per Delivery it covers, keyed by ``row``. For Delivery
i and for every Delivery before it, each column must be identical in the two runs: the same
dtype, floats equal bit for bit (missing values included) and other values equal, with missing
values in the same rows. A failure lists each changed (Delivery, column) pair with both values.

Hypothesis events record whether the replacement changed the cut Delivery's own state inputs and
each of its PI_UNREAD_COLUMNS, and whether any later Delivery's values changed, so
``--hypothesis-show-statistics`` shows how often the replacement had an effect. No event is
asserted, since a replacement may legitimately leave every later value unchanged (for example
when i is the last Delivery of innings 2).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import pandas as pd
from hypothesis import event, given, note
from hypothesis import strategies as st

from superr.audit import bowling_type_stability
from superr.folds import fold_of
from superr.models.common import training_mask
from superr.models.outcome import CATEGORICAL_FEATURES, CLASSES, N_CLASSES, m4_features, outcome_rows
from superr.models.remaining_runs import (
    ACH_RATE_COLUMN,
    F19_COLUMN,
    M2_FEATURES,
    PAR_BIAS_COLUMNS,
    PAR_BUCKETS,
    PAR_COLUMNS,
    PAR_MEAN_FEATURES,
    VIEW_M2_COLUMNS,
    achievable_rate,
    first_innings_par,
    m2_features,
)
from superr.models.win_prob import M3_INNINGS, m3_features
from superr.pressure import (
    POST_COLUMNS,
    POST_INNINGS,
    own_fold_view,
    pillar_components,
    post_outcome_states,
    train_wicket_means,
    wicket_means_for,
)
from superr.reconstruct import (
    ENCODINGS,
    ORDER_KEYS,
    balls_table,
    classify,
    delivery_scheduled_balls,
    dismissed_batter_flags,
    encoding_inputs,
    free_hit_flags,
    innings_aggregates,
    match_columns,
    order_deliveries,
    parse_results,
    scheduled_balls,
    super_over_flags,
    targets,
    wicket_flags,
)
from superr.skill import VIEW_COLUMNS, match_day_conditions, rbq, team_strength_gap, typical_top7_sum, view_rows
from superr.state import BALLS_PER_OVER, MODAL_COLUMNS, STATE_INNINGS, build_state
from superr.views import M2_FIT
from tests.strategies import COUNTED_WICKET_CODES, scorecard_match

if TYPE_CHECKING:
    from superr.settings import Settings

#: The Outcome_Columns (glossary): the columns of a Delivery that record what happened on that ball.
OUTCOME_COLUMNS: tuple[str, ...] = (
    "runs", "is_four", "is_six", "is_wicket", "dismissal_type_id", "dismissal_type", "dismissed_player_id",
    "extras_type", "extras1_runs", "extras_2_runs",
)  # fmt: skip
#: The current Delivery's values that no PI input may read (Requirement 13, criterion 6). Each is an
#: Outcome_Column, so the replacement changes it on Delivery i whenever the donor's value differs.
PI_UNREAD_COLUMNS: tuple[str, ...] = ("runs", "extras_2_runs", "is_wicket", "dismissal_type_id")
#: The columns of ``balls`` that the State_Builder reads (``superr.state.INPUT_COLUMNS``) and that
#: come from the Outcome_Columns. Used only for the event on the cut Delivery's own outcome.
OUTCOME_INPUTS: tuple[str, ...] = ("runs", "is_four", "is_six", "extras_type", "cls", "team_runs", "counted_wicket")
#: Most Overs per innings. With 4 or 5 Overs, more than D6 (18 as delivered) Legal_Balls precede
#: the last Deliveries of an innings, so their recent window starts inside the innings.
MAX_OVERS = 5
#: Changed (Delivery, column) pairs listed in a failure message.
MAX_SHOWN = 20
#: Columns of the Deliveries shown when Hypothesis reports a falsifying example.
NOTE_COLUMNS: tuple[str, ...] = ("innings", "innings_ball_num", "striker_id", "bowler_id", *OUTCOME_COLUMNS)

#: The competition of every generated match, typed as the Data_Loader types competitions.csv.
COMPETITIONS = pd.DataFrame(
    {
        "comp_id": pd.array([1], dtype="Int64"),
        "comp_name": pd.array(["IPL 2025"], dtype="string"),
        "match_format": pd.array(["T20"], dtype="string"),
        "season": pd.array(["2025"], dtype="string"),
    }
)
#: matches.csv columns that match_columns reads beyond those of scorecard_match: (dtype, fixed value).
MATCH_EXTRAS: dict[str, tuple[str, Any]] = {
    "comp_id": ("Int64", 1),
    "match_date": ("datetime64[us]", "2025-05-03"),
    "knock_out": ("Int64", 0),
    "venue_id": ("Int64", 1),
    "venue_country": ("string", "India"),
}


# ------------------------------------------------------------------------------ model stand-ins

#: The match's fold in the Fold_Table. Its rows read every model-derived value from view V_FOLD, as
#: a fold-k row reads V_k in a Run.
FOLD = 1
#: The folds of the fixed Training_Rows of other matches (D15 = 5 as delivered): the Training_Folds of FOLD.
OTHER_FOLDS: tuple[int, ...] = (2, 3, 4, 5)
#: The first-innings M2 fit that every view row names in ``m2_fit``: the outer fit M2(-FOLD)
#: (``leakage.fit_id_for``: ``<model>_<family>_outer_k<k>``).
M2_FIT_ID = "m2_first_lgbm_outer_k1"
#: The stand-in M1 fit's Scoring_Environment and Venue_Effect, the same on every Delivery (one
#: competition season, one venue), and the effects of the innings x Phase cell and of each wicket
#: fallen in its predicted bat runs, all in bat runs per Legal_Ball.
SCORING_ENV, VENUE_EFFECT = 0.06, -0.02
M1_CELL_EFFECT, M1_WICKET_EFFECT = 0.04, -0.03
#: The stand-in M2 fit's expected runs with no Legal_Ball left, its runs per Legal_Ball left when
#: every other feature is 0, and the weight of each other M2 feature in those runs per Legal_Ball.
M2_LAST_BALL_RUNS = 2.0
M2_BASE_RATE = 1.2
M2_RATE_WEIGHTS: dict[str, float] = {
    "wickets": -0.06, "rbq": 0.25, "theta_striker": 0.08, "striker_lb_faced": 0.002, "theta_non_striker": 0.05,
    "scoring_env": 0.6, "venue_effect": 0.5, "match_day": 0.01,
}  # fmt: skip
#: The Class_Run_Values of fold FOLD in CLASSES order: exactly 0, 1, 2, 3, 4 and 6 for the dot and
#: run classes, and means of the delivered size for other_runs, bowler_wicket and run_out.
CLASS_RUN_VALUES: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 5.0, 0.0, 0.43)
#: The last Legal_Balls bowled L of the innings-2 TRAINING_ROWS. An innings-2 Delivery with a later
#: L has no Training_Folds mean, so its F20(a) is missing (Requirement 12, criterion 16).
LAST_TRAINING_LB = 27


class _FirstInningsM2:
    """A deterministic stand-in for the first-innings M2 fit M2_FIT_ID (``remaining_runs.Regressor``).

    ``remaining_runs.predict`` accepts any object with ``predict``. The expected remaining runs are
    M2_LAST_BALL_RUNS + Legal_Balls left x (M2_BASE_RATE + the weighted sum of the other M2
    features), so a changed feature changes every Achievable_Rate and Par_Score it enters. A
    missing value counts as 0, as LightGBM reads a missing ``theta_next_batter``
    (``remaining_runs``, "Features"). Each row's value depends on that row alone, so the Par_Score
    at Scheduled_Balls of an end-of-innings state equals the view's ``par_score_next`` bit for bit.
    """

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        x = {name: np.nan_to_num(X[name].to_numpy(dtype=np.float64), nan=0.0) for name in M2_FEATURES}
        per_ball = M2_BASE_RATE + sum(weight * x[name] for name, weight in M2_RATE_WEIGHTS.items())
        return M2_LAST_BALL_RUNS + x["lb_left"] * per_ball


M2_FIRST_FIT = _FirstInningsM2()


def _par_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fixed par means and par bias of M2_FIT_ID per bucket 0-120, as ``par_means`` and ``par_bias`` return them.

    The par-state means move from the start to the end of an innings as a fit's first-innings
    Training_Rows do: wickets and striker Legal_Balls faced rise, RBQ and theta_bat fall. The bias
    rises with the bucket, as the delivered fits' bias does.
    """
    bucket = np.arange(PAR_BUCKETS + 1)
    share = bucket / PAR_BUCKETS
    index = pd.RangeIndex(PAR_BUCKETS + 1, name="bucket")
    n_rows = np.full(PAR_BUCKETS + 1, 40, dtype=np.int64)
    means = pd.DataFrame(
        {
            "wickets": 6.5 * share**2,
            "rbq": 1.0 - 0.7 * share,
            "theta_striker": 1.25 - 0.2 * share,
            "striker_lb_faced": 12.0 * share,
            "theta_non_striker": 1.2 - 0.15 * share,
            "n_rows": n_rows,
        },
        index=index,
    )
    bias = pd.DataFrame({"beta": 0.08 * bucket, "n_rows": n_rows}, index=index)
    return means, bias


#: The par means and par bias of M2_FIT_ID on the index ``bucket`` (``first_innings_par`` reads them).
PAR_MEANS, PAR_BIAS = _par_tables()
#: The same values as the remaining_runs checkpoint tables ``m2_par_means`` and ``m2_par_bias``,
#: one block per fit id (``post_outcome_states`` reads them through ``win_prob.innings_end_par``).
PAR_MEANS_TABLE = PAR_MEANS.reset_index().assign(fit_id=M2_FIT_ID)[["fit_id", "bucket", *PAR_MEAN_FEATURES, "n_rows"]]
PAR_BIAS_TABLE = PAR_BIAS.reset_index().assign(fit_id=M2_FIT_ID)[["fit_id", "bucket", *PAR_BIAS_COLUMNS]]


def _training_rows() -> pd.DataFrame:
    """Fixed Training_Rows of matches in OTHER_FOLDS for the F20(a) means (Requirement 12, criterion 3).

    Two rows per fold, innings and Legal_Balls bowled L: L = 0 to 6 x MAX_OVERS in innings 1 and 0
    to LAST_TRAINING_LB in innings 2. Wickets fallen rise with L, at a pace that differs by fold,
    innings and row, and stop at 9.
    """
    grids = np.meshgrid(OTHER_FOLDS, STATE_INNINGS, np.arange(BALLS_PER_OVER * MAX_OVERS + 1), (0, 1), indexing="ij")
    fold, innings, lb, copy = (grid.ravel() for grid in grids)
    keep = (innings == STATE_INNINGS[0]) | (lb <= LAST_TRAINING_LB)
    wickets = np.minimum((lb * (fold + innings) + 7 * copy) // 19, 9)
    frame = pd.DataFrame({"fold": fold, "innings": innings, "lb_bowled": lb, "wickets": wickets})
    return frame[keep].reset_index(drop=True)


#: The Training_Rows of other matches behind the F20(a) means of fold FOLD.
TRAINING_ROWS = _training_rows()


def _theta_table(xi: pd.DataFrame) -> pd.DataFrame:
    """The stand-in M1 fit's theta table: theta_bat and theta_bowl of every player of ``xi``, set from his id.

    The generator numbers a player team_id x 100 + batting slot (1-11): the top of the order bats
    best, and slots 6-11 bowl.
    """
    ids = np.unique(xi["player_id"].to_numpy(dtype=np.int64))
    slot, team = ids % 100, ids // 100
    return pd.DataFrame(
        {
            "player_id": ids,
            "theta_bat": 1.5 - 0.06 * slot + 0.01 * (team % 7),
            "theta_bowl": np.where(slot >= 6, 0.03 * (slot - 5), -0.02) + 0.004 * (team % 5),
        }
    )


def _theta_of(theta: pd.DataFrame, column: str, ids: pd.Series) -> np.ndarray:
    """``column`` of ``theta`` for each player id in ``ids`` (float64), NaN where the id is missing."""
    present = ids.notna().to_numpy(dtype=bool)
    at = pd.Index(theta["player_id"]).get_indexer(ids.to_numpy(dtype=np.float64, na_value=-1.0).astype(np.int64))
    if (at[present] < 0).any():
        raise AssertionError("strategy bug: a player of the Deliveries has no row in the playing XI")
    return np.where(present, theta[column].to_numpy(dtype=np.float64)[at], np.nan)


# ------------------------------------------------------------------------------ feature builders

class Inputs(NamedTuple):
    """What the stages before the Pressure_Engine give a feature builder, for one version of the match."""

    deliveries: pd.DataFrame  # the ball_by_ball rows (load stage), in the Requirement 5 order
    matches: pd.DataFrame  # the matches row (load stage)
    balls: pd.DataFrame  # the balls table (reconstruct stage)
    xi: pd.DataFrame  # the playing_eleven rows (load stage)
    modal_bowling_type: pd.DataFrame  # the modal_bowling_type table (audit stage)
    state: pd.DataFrame  # the state table (state stage)
    fold_table: pd.DataFrame  # the Fold_Table (folds stage): the match in FOLD
    view: pd.DataFrame  # view V_FOLD (skill and remaining_runs stages, stand-in fits)
    settings: Settings  # the Config


class FeatureBuilder(NamedTuple):
    """Per-Delivery values that Property 14 requires to come from the past only."""

    name: str
    build: Callable[[Inputs], pd.DataFrame]  # a table keyed by ``row``, one row per Delivery it covers


def _state_rows(balls: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
    """The ``balls`` rows of the ``state`` rows, in ``state`` order, on a RangeIndex."""
    at = pd.Index(balls["row"]).get_indexer(state["row"])
    return balls.iloc[at].reset_index(drop=True)


def _state_features(inputs: Inputs) -> pd.DataFrame:
    """The ``state`` table (Requirement 6, criteria 1-14), as the ``state`` stage builds it (``_inputs``)."""
    return inputs.state


def _view_values(inputs: Inputs) -> pd.DataFrame:
    """View V_FOLD: every M1 and M2 value that the Pressure_Engine and the M3 and M4 features read (``_view``)."""
    return inputs.view


def _pillar_components(inputs: Inputs) -> pd.DataFrame:
    """The Pillar_Components of both Roles (Requirement 12, criteria 1-11), wired as the ``pressure`` stage does.

    The F20(a) means come from ``train_wicket_means`` over the match's rows and TRAINING_ROWS, so
    the means of fold FOLD are taken over the other matches only (Requirement 12, criterion 3).
    """
    state, rows = inputs.state, _state_rows(inputs.balls, inputs.state)
    fold = fold_of(rows["match_id"], inputs.fold_table)
    innings = rows["innings"].to_numpy(dtype=np.int64)
    lb = state["lb_bowled"].to_numpy(dtype=np.int64)
    means = train_wicket_means(
        fold=np.concatenate([fold, TRAINING_ROWS["fold"].to_numpy()]),
        innings=np.concatenate([innings, TRAINING_ROWS["innings"].to_numpy()]),
        lb_bowled=np.concatenate([lb, TRAINING_ROWS["lb_bowled"].to_numpy()]),
        wickets=np.concatenate([state["wickets"].to_numpy(dtype=np.int64), TRAINING_ROWS["wickets"].to_numpy()]),
        training=np.concatenate([training_mask(rows), np.ones(len(TRAINING_ROWS), dtype=bool)]),
    )
    return pillar_components(
        state,
        own_fold_view({FOLD: inputs.view}, fold, state["row"]),
        innings=innings,
        mean_wickets=wicket_means_for(means, fold=fold, innings=innings, lb_bowled=lb),
        new_batter_cap=inputs.settings.get("design_choices.D7_new_batter_cap"),
    )


def _m3_pre_ball(inputs: Inputs) -> pd.DataFrame:
    """The pre-ball M3 features of each Delivery's innings (``win_prob.m3_features``): the input of its pre-ball WP.

    One column per feature of either innings, NaN on the rows of the innings that lacks it.
    """
    rows = _state_rows(inputs.balls, inputs.state)
    labels = rows[["row", "innings", "sched_balls"]]  # the m3_labels columns that m3_features reads
    innings = rows["innings"].to_numpy(dtype=np.int64)
    parts = [
        m3_features(number, inputs.view, inputs.state, labels).set_axis(np.flatnonzero(innings == number))
        for number in M3_INNINGS
    ]
    table = pd.concat(parts).sort_index().reset_index(drop=True)
    table.insert(0, "row", inputs.state["row"].to_numpy())
    return table


def _post_states(inputs: Inputs) -> pd.DataFrame:
    """Every column of each Delivery's 9 Post_Outcome_States (Requirement 11, criterion 15), side by side.

    Column ``<class>.<name>`` holds the POST_COLUMNS value ``name`` of the class's state, for the
    classes of CLASSES. A column of one innings only is NaN on the rows of the other.
    """
    rows = _state_rows(inputs.balls, inputs.state)
    post = post_outcome_states(
        inputs.state, inputs.balls, {FOLD: inputs.view}, fold_of(rows["match_id"], inputs.fold_table),
        {FOLD: CLASS_RUN_VALUES}, {M2_FIT_ID: M2_FIRST_FIT}, par_means=PAR_MEANS_TABLE, par_bias=PAR_BIAS_TABLE,
    )  # fmt: skip
    row_ids = inputs.state["row"].to_numpy()
    n_rows = len(row_ids)
    frames = {number: post.frame(number) for number in POST_INNINGS}
    for number, frame in frames.items():  # state s is class s % 9 of the Delivery at positions[s // 9]
        layout = frame[["row", "cls"]].to_numpy(dtype=np.int64).reshape(-1, N_CLASSES, 2)
        rows_ok = (layout[..., 0] == row_ids[post.positions[number], None]).all()
        if not (rows_ok and (layout[..., 1] == np.arange(N_CLASSES)).all()):
            raise AssertionError(f"test bug: the innings-{number} PostStates are not 9 states per Delivery in order")
    names = [
        name
        for name in dict.fromkeys(name for number in POST_INNINGS for name in POST_COLUMNS[number])
        if name not in ("row", "cls")
    ]
    by_class: dict[str, np.ndarray] = {}  # name -> (Deliveries, N_CLASSES) values, Deliveries in state order
    for name in names:
        held = [number for number in POST_INNINGS if name in frames[number].columns]
        dtypes = {frames[number][name].dtype for number in held}
        dtype = dtypes.pop()
        if dtypes or (len(held) < len(POST_INNINGS) and dtype.kind != "f"):
            raise AssertionError(f"test bug: the Post_Outcome_State column {name!r} cannot be joined across innings")
        values = np.full((n_rows, N_CLASSES), np.nan) if dtype.kind == "f" else np.empty((n_rows, N_CLASSES), dtype)
        for number in held:
            values[post.positions[number]] = frames[number][name].to_numpy().reshape(-1, N_CLASSES)
        by_class[name] = values
    columns = {f"{label}.{name}": by_class[name][:, cls] for cls, label in enumerate(CLASSES) for name in names}
    return pd.DataFrame({"row": row_ids, **columns})


def _m4_pre_ball(inputs: Inputs) -> pd.DataFrame:
    """The pre-ball M4 features (``outcome.m4_features``), the input of the class probabilities P of the LI.

    The categorical ``phase`` and ``hand_x_style`` are compared by each row's label: their category
    sets span every Delivery, and LightGBM maps a label to its training category.
    """
    table = m4_features(inputs.view, inputs.state, outcome_rows(inputs.balls, inputs.state, inputs.fold_table))
    for name in CATEGORICAL_FEATURES:
        table[name] = table[name].astype("string")
    table.insert(0, "row", inputs.state["row"].to_numpy())
    return table


#: The values that Property 14 checks: the state features (task 6.5), then the view values, the
#: Pillar_Components and the PI inputs of both Roles (task 15.6).
FEATURE_BUILDERS: tuple[FeatureBuilder, ...] = (
    FeatureBuilder("state features", _state_features),
    FeatureBuilder("view values", _view_values),
    FeatureBuilder("Pillar_Components", _pillar_components),
    FeatureBuilder("pre-ball M3 features", _m3_pre_ball),
    FeatureBuilder("Post_Outcome_States", _post_states),
    FeatureBuilder("pre-ball M4 features", _m4_pre_ball),
)


# ------------------------------------------------------------------------------------- strategy

class Example(NamedTuple):
    """One example of Property 14."""

    encoding: str  # the Extras_Encoding of the generated match, held fixed
    deliveries: pd.DataFrame  # the match's Deliveries in the Requirement 5 order, on a RangeIndex
    matches: pd.DataFrame  # its matches row, with the columns that match_columns reads
    xi: pd.DataFrame  # the playing_eleven rows of both sides
    cut: int  # i: the position (and ``row``) of the cut Delivery, in innings 1 or 2
    donors: tuple[int, ...]  # the donor position of Deliveries i, i + 1, ... in order
    modified: pd.DataFrame  # ``deliveries`` with the Outcome_Columns of i and every later Delivery replaced


def _matches_row(matches: pd.DataFrame) -> pd.DataFrame:
    """The scorecard_match ``matches`` row plus the MATCH_EXTRAS columns."""
    row = matches.reset_index(drop=True)
    for column, (dtype, value) in MATCH_EXTRAS.items():
        row[column] = pd.array([value] * len(row), dtype=dtype)
    return row


def _outcome_keys(deliveries: pd.DataFrame) -> list[tuple[Any, ...]]:
    """The Outcome_Columns of each Delivery as a tuple, with None for a missing value."""
    columns = [deliveries[column].tolist() for column in OUTCOME_COLUMNS]
    return [tuple(None if pd.isna(value) else value for value in values) for values in zip(*columns, strict=True)]


def _with_outcomes_from(deliveries: pd.DataFrame, source: np.ndarray) -> pd.DataFrame:
    """``deliveries`` with the Outcome_Columns of the Delivery at position p copied from position ``source[p]``."""
    modified = deliveries.copy()
    for column in OUTCOME_COLUMNS:
        modified[column] = deliveries[column].array.take(source)
    return modified


def _playing_xi(draw: Any, deliveries: pd.DataFrame) -> pd.DataFrame:
    """The ``is_played`` rows of both sides: every player seen for the side, one optional Impact Player."""
    squads: dict[int, set[int]] = {}
    for team_column, player_columns in (
        ("batting_team_id", ("striker_id", "non_striker_id")),
        ("bowling_team_id", ("bowler_id",)),
    ):
        for player_column in player_columns:
            pairs = zip(deliveries[team_column].tolist(), deliveries[player_column].tolist(), strict=True)
            for team, player in pairs:
                squads.setdefault(int(team), set()).add(int(player))
    match_id = int(deliveries["match_id"].iloc[0])
    rows = [
        (team, player, order)
        for team in sorted(squads)
        for order, player in enumerate(sorted(squads[team]), start=1)
    ]
    impact = draw(st.one_of(st.none(), st.sampled_from(range(len(rows)))))
    off = None
    if impact is not None:
        teammates = [k for k, row in enumerate(rows) if row[0] == rows[impact][0] and k != impact]
        off = draw(st.one_of(st.none(), st.sampled_from(teammates))) if teammates else None
    n = len(rows)
    return pd.DataFrame(
        {
            "match_id": pd.array([match_id] * n, dtype="Int64"),
            "team_id": pd.array([team for team, _, _ in rows], dtype="Int64"),
            "player_id": pd.array([player for _, player, _ in rows], dtype="Int64"),
            "batting_order": pd.array([order for _, _, order in rows], dtype="Int64"),
            "is_played": pd.array([True] * n, dtype="boolean"),
            "is_impact": pd.array([int(k == impact) for k in range(n)], dtype="Int64"),
            "is_impact_off": pd.array([int(k == off) for k in range(n)], dtype="Int64"),
        }
    )


@st.composite
def _examples(draw: Any) -> Example:
    case = draw(scorecard_match(max_overs=MAX_OVERS))
    deliveries = case.balls.sort_values(list(ORDER_KEYS), kind="stable").reset_index(drop=True)
    n = len(deliveries)
    # One match, so innings 1 and 2 come first in the Requirement 5 order and the cut is a position among them.
    regular = np.flatnonzero(deliveries["innings"].isin(STATE_INNINGS).to_numpy(dtype=bool))
    if len(regular) == 0 or not np.array_equal(regular, np.arange(len(regular))):
        raise AssertionError("strategy bug: the match has no Delivery in innings 1 or 2, or they do not come first")
    cut = draw(st.integers(0, len(regular) - 1))

    keys = _outcome_keys(deliveries)
    different = [position for position, key in enumerate(keys) if key != keys[cut]]
    first = draw(st.sampled_from(different)) if different else cut
    later = draw(st.lists(st.integers(0, n - 1), min_size=n - cut - 1, max_size=n - cut - 1))
    donors = (first, *later)
    source = np.array([*range(cut), *donors], dtype=np.intp)
    return Example(
        encoding=case.encoding,
        deliveries=deliveries,
        matches=_matches_row(case.matches),
        xi=_playing_xi(draw, deliveries),
        cut=cut,
        donors=donors,
        modified=_with_outcomes_from(deliveries, source),
    )


# ------------------------------------------------------------------------------- pipeline steps

def _balls(deliveries: pd.DataFrame, matches: pd.DataFrame, encoding: str, settings: Settings) -> pd.DataFrame:
    """The ``balls`` table of ``deliveries``, assembled as ``reconstruct.run_stage`` does, under ``encoding``."""
    ordered = order_deliveries(deliveries)
    cls = classify(ordered["extras_type"], settings.get("data_rules.extras_type_classes"))
    wickets = wicket_flags(
        ordered,
        COUNTED_WICKET_CODES,
        settings.get("design_choices.D14_bowler_wicket_codes"),
        settings.get("design_choices.D14_run_out_code"),
        settings.get("design_choices.D14_retire_out_code"),
    )
    team_runs = ENCODINGS[encoding](encoding_inputs(ordered, cls))
    derived = pd.concat(
        [
            cls,
            super_over_flags(ordered["innings"]),
            free_hit_flags(ordered, cls),
            pd.Series(team_runs, index=ordered.index, name="team_runs"),
            wickets,
            dismissed_batter_flags(ordered, wickets),
            delivery_scheduled_balls(ordered, scheduled_balls(matches)),
            pd.Series(False, index=ordered.index, name="excluded"),
        ],
        axis=1,
    )
    target = targets(matches, innings_aggregates(ordered, team_runs, wickets["counted_wicket"]))
    return balls_table(ordered, derived, match_columns(matches, COMPETITIONS, parse_results(matches), target))


def _modal_bowling_type(example: Example) -> pd.DataFrame:
    """The ``audit`` stage's ``modal_bowling_type`` table of the match (``AuditResult.modal_bowling_type``).

    It reads ``bowling_type``, which is not an Outcome_Column, so both versions of the match share it.
    """
    table = bowling_type_stability(example.deliveries, example.matches)
    return table[list(MODAL_COLUMNS)].reset_index(drop=True)


def _fold_table(matches: pd.DataFrame) -> pd.DataFrame:
    """The Fold_Table (``folds.FOLD_TABLE_COLUMNS``) of the match, which lies in fold FOLD."""
    n = len(matches)
    return pd.DataFrame(
        {
            "match_id": matches["match_id"].to_numpy(dtype=np.int64),
            "comp_type": pd.array(["IPL"] * n, dtype="string"),
            "fold": np.full(n, FOLD, dtype=np.int32),
        }
    )


def _view(balls: pd.DataFrame, state: pd.DataFrame, xi: pd.DataFrame) -> pd.DataFrame:
    """View V_FOLD of the match: ``row``, ``skill.VIEW_COLUMNS``, ``remaining_runs.VIEW_M2_COLUMNS`` and ``m2_fit``.

    The ``skill`` and ``remaining_runs`` functions build it from the stand-in fits (module
    docstring, step 4), with the values that the stages compute for the rows of fold FOLD.
    """
    rows = view_rows(balls, state)  # each state row's balls columns, with cell, wickets and Batters_To_Come
    theta = _theta_table(xi)
    sides = xi[["match_id", "team_id"]].drop_duplicates().rename(columns={"team_id": "batting_team_id"})
    lineup = rbq(rows, theta, typical_top7_sum(theta, xi, sides))
    theta_striker = _theta_of(theta, "theta_bat", rows["striker_id"])
    theta_bowler = _theta_of(theta, "theta_bowl", rows["bowler_id"])
    pred_runs = (
        theta_striker - theta_bowler + SCORING_ENV + VENUE_EFFECT
        + M1_CELL_EFFECT * rows["cell"].to_numpy(dtype=np.float64)
        + M1_WICKET_EFFECT * rows["wickets"].to_numpy(dtype=np.float64)
    )  # fmt: skip
    n_rows = len(rows)
    m1 = {
        "theta_striker": theta_striker,
        "theta_non_striker": _theta_of(theta, "theta_bat", rows["non_striker_id"]),
        "theta_bowler": theta_bowler,
        "theta_next_batter": lineup["theta_next_batter"].to_numpy(),
        "scoring_env": np.full(n_rows, SCORING_ENV),
        "venue_effect": np.full(n_rows, VENUE_EFFECT),
        "rbq": lineup["rbq"].to_numpy(),
        "rbq_post_wicket": lineup["rbq_post_wicket"].to_numpy(),
        "tsg": team_strength_gap(rows, theta, xi),
        "match_day": match_day_conditions(rows, pred_runs),
        "m1_pred_runs": pred_runs,
    }
    view = pd.DataFrame({"row": state["row"].to_numpy(), **{name: m1[name] for name in VIEW_COLUMNS}})

    # The M2 values, as the remaining_runs stage serves the rows of one first-innings fit.
    X = m2_features(view, state)
    sched = _state_rows(balls, state)["sched_balls"].to_numpy(dtype=np.int64)
    innings = rows["innings"].to_numpy(dtype=np.int64)
    m2 = {name: np.full(n_rows, np.nan) for name in VIEW_M2_COLUMNS}
    one, two = np.flatnonzero(innings == STATE_INNINGS[0]), np.flatnonzero(innings == STATE_INNINGS[1])
    if len(one):
        par = first_innings_par(
            M2_FIRST_FIT, PAR_MEANS, PAR_BIAS, X.iloc[one],
            state["lb_bowled"].to_numpy()[one], sched[one], state["win_lb"].to_numpy()[one],
        )  # fmt: skip
        for name in PAR_COLUMNS:
            m2[name][one] = par[name].to_numpy(dtype=np.float64)
        m2[F19_COLUMN][one] = m2[PAR_COLUMNS[0]][one] - state["score"].to_numpy(dtype=np.float64)[one]
    if len(two):
        m2[ACH_RATE_COLUMN][two] = achievable_rate(M2_FIRST_FIT, X.iloc[two])
        m2[F19_COLUMN][two] = state["rrr"].to_numpy(dtype=np.float64)[two] - m2[ACH_RATE_COLUMN][two]
    for name in VIEW_M2_COLUMNS:
        view[name] = m2[name]
    view[M2_FIT] = pd.array([M2_FIT_ID] * n_rows, dtype="string")
    return view


def _inputs(deliveries: pd.DataFrame, example: Example, modal: pd.DataFrame, settings: Settings) -> Inputs:
    """The feature builders' inputs for one version of the match's Deliveries (module docstring, steps 1-4)."""
    balls = _balls(deliveries, example.matches, example.encoding, settings)
    state = build_state(
        balls,
        example.xi,
        modal,
        window=settings.get("design_choices.D6_window"),
        phase_splits=settings.get("design_choices.D8_phase_splits"),
    )
    return Inputs(
        deliveries=deliveries,
        matches=example.matches,
        balls=balls,
        xi=example.xi,
        modal_bowling_type=modal,
        state=state,
        fold_table=_fold_table(example.matches),
        view=_view(balls, state, example.xi),
        settings=settings,
    )


# -------------------------------------------------------------------------------------- helpers

def _changed(before: pd.DataFrame, after: pd.DataFrame) -> np.ndarray:
    """A (rows x columns) bool array, True where a value differs between two tables of the same shape.

    Floats are compared bit for bit, so -0.0 differs from 0.0 and a NaN equals only a NaN with the
    same bits, and they must be missing in the same rows. Every other value must be missing in
    both tables or present and equal in both. A column whose dtype differs is changed in every row.
    """
    cells = np.zeros(before.shape, dtype=bool)
    blocks: dict[np.dtype, list[int]] = {}  # NumPy bool, integer and float columns, compared per dtype below
    for k, (dtype, other) in enumerate(zip(before.dtypes, after.dtypes, strict=True)):
        if dtype != other:
            cells[:, k] = True
            continue
        if isinstance(dtype, np.dtype) and dtype.kind in "biuf":
            blocks.setdefault(dtype, []).append(k)
            continue
        a, b = before.iloc[:, k].reset_index(drop=True), after.iloc[:, k].reset_index(drop=True)
        missing_a, missing_b = a.isna().to_numpy(dtype=bool), b.isna().to_numpy(dtype=bool)
        if pd.api.types.is_float_dtype(a.dtype):
            bits_a = a.to_numpy(dtype=np.float64, na_value=np.nan).view(np.uint64)
            bits_b = b.to_numpy(dtype=np.float64, na_value=np.nan).view(np.uint64)
            same = (missing_a == missing_b) & (bits_a == bits_b)
        else:
            equal = a.eq(b).fillna(False).to_numpy(dtype=bool)
            same = (missing_a & missing_b) | (~missing_a & ~missing_b & equal)
        cells[:, k] = ~same
    # A NumPy bool or integer value is never missing, and a NumPy float is missing only as NaN, so
    # equal bits mean equal values that are missing in the same rows.
    for dtype, ks in blocks.items():
        a, b = before.iloc[:, ks].to_numpy(dtype=dtype), after.iloc[:, ks].to_numpy(dtype=dtype)
        if dtype.kind == "f":
            a, b = a.view(f"u{dtype.itemsize}"), b.view(f"u{dtype.itemsize}")
        cells[:, ks] = a != b
    return cells


def _listing(before: pd.DataFrame, after: pd.DataFrame, changed: np.ndarray, rows: Sequence[int], cut: int) -> str:
    """The changed (Delivery, column) pairs among ``rows``, with the original and the modified value."""
    lines = []
    for position in np.flatnonzero(changed.any(axis=1)):
        row = rows[position]
        where = "Delivery i" if row == cut else "a Delivery before i"
        for k in np.flatnonzero(changed[position]):
            column = before.columns[k]
            lines.append(
                f"  row {row} ({where}), {column}: {before[column].iloc[position]!r} -> {after[column].iloc[position]!r}"
                f" (dtype {before[column].dtype} -> {after[column].dtype})"
            )
    shown = lines[:MAX_SHOWN]
    if len(lines) > MAX_SHOWN:
        shown.append(f"  ... and {len(lines) - MAX_SHOWN} more")
    return "\n".join(shown)


class _Shown:
    """Rows and columns of a table for hypothesis.note, formatted only when Hypothesis prints the note."""

    def __init__(self, title: str, frame: pd.DataFrame, columns: Sequence[str]) -> None:
        self.title, self.frame, self.columns = title, frame, list(columns)

    def __repr__(self) -> str:
        return f"{self.title}:\n{self.frame[self.columns].to_string()}"

    __str__ = __repr__


# ----------------------------------------------------------------------------------------- test

@given(example=_examples())
def test_pre_ball_state_and_pi_use_only_the_past(example: Example, settings: Settings) -> None:
    """Replacing the outcomes of Delivery i and every later Delivery leaves i's values and earlier ones unchanged.

    **Validates: Requirements 6.1, 6.15, 13.6**
    """
    unread = [column for column in PI_UNREAD_COLUMNS if column not in OUTCOME_COLUMNS]
    assert not unread, f"the replacement must cover the values of Requirement 13, criterion 6; it misses {unread}"
    cut = example.cut
    note(f"Extras_Encoding {example.encoding}; cut Delivery i at row {cut}")
    note(f"donor positions of rows {cut}, {cut + 1}, ...: {list(example.donors)}")
    note(_Shown("Original Deliveries from i on", example.deliveries.iloc[cut:], NOTE_COLUMNS))
    note(_Shown("Modified Deliveries from i on", example.modified.iloc[cut:], NOTE_COLUMNS))
    note(_Shown("Playing XI", example.xi, example.xi.columns))

    modal = _modal_bowling_type(example)
    original = _inputs(example.deliveries, example, modal, settings)
    modified = _inputs(example.modified, example, modal, settings)

    # Sanity events (not asserted): did the replacement change the cut Delivery's own state inputs,
    # and the values that no PI input may read (Requirement 13, criterion 6)?
    event(f"Delivery i in innings {int(example.deliveries['innings'].iloc[cut])}")
    own_before = original.balls.loc[original.balls["row"] == cut, list(OUTCOME_INPUTS)]
    own_after = modified.balls.loc[modified.balls["row"] == cut, list(OUTCOME_INPUTS)]
    own = bool(_changed(own_before, own_after).any())
    event("Delivery i's own state inputs changed" if own else "Delivery i's own state inputs unchanged")
    pi_changed = _changed(
        example.deliveries.iloc[[cut]][list(PI_UNREAD_COLUMNS)], example.modified.iloc[[cut]][list(PI_UNREAD_COLUMNS)]
    )[0]
    for column, differs in zip(PI_UNREAD_COLUMNS, pi_changed, strict=True):
        if differs:
            event(f"Delivery i's own {column} changed")

    for builder in FEATURE_BUILDERS:
        before, after = builder.build(original), builder.build(modified)
        assert list(after.columns) == list(before.columns), f"{builder.name}: the columns differ"
        rows = before["row"].to_numpy(dtype=np.int64)
        assert np.array_equal(after["row"].to_numpy(dtype=np.int64), rows), (
            f"{builder.name}: the two runs cover different Deliveries or order them differently"
        )
        if not (rows == cut).any():
            event(f"{builder.name}: Delivery i has no row")  # only earlier Deliveries are compared
        changed = _changed(before, after)
        upto = rows <= cut  # Delivery i and every Delivery before it
        assert not changed[upto].any(), (
            f"{builder.name} of Delivery i (row {cut}) or of an earlier Delivery changed when the "
            f"Outcome_Columns of Delivery i and every later Delivery were replaced:\n"
            + _listing(before[upto], after[upto], changed[upto], rows[upto].tolist(), cut)
        )
        later = bool(changed[~upto].any())
        event(f"{builder.name}: a later Delivery changed" if later else f"{builder.name}: no later Delivery changed")
