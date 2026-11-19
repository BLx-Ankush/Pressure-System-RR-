"""Pressure_Validator: the checks V1-V4 of the pressure definition on held-out matches, the
Out_Of_Fold guard and the ``validation`` Pipeline_Stage (Requirement 14; Requirement 2, criteria 7
and 13; design section "Pressure_Validator (``validation.py``)"; Properties 39 and 40).

The validation asks whether the Pressure_Balls that the ``pressure`` stage flagged are the balls on
which matches swing and players do measurably differently. It reads the Pressure_Engine's table as
it is (no Pressure_Ball flag is set again) and judges it only on quantities that depend on the
outcome of the Delivery or of later Deliveries (Requirement 2, criterion 7): the Actual_WP_Swing,
Big_Moment membership, the Wides and No_Balls bowled, the bat runs, the runs conceded and the
striker's Counted_Wickets. PI reads none of them (Requirement 13, criterion 6). No function reads
``config.py``: ``run_stage`` reads D3, D8 and D15 from ``ctx.settings``, and every other function
takes the values it needs as arguments. No validation result feeds a Config value or selects a
Design_Choice (Requirement 2, criteria 7 and 13): the stage writes only its own checkpoint and
``outputs/validation/``, records no Config value or Design_Choice, and no stage reads a validation
result to choose a setting.

Validation_Deliveries (``validation_mask``; glossary; criterion 1)
    The Deliveries of innings 1 and 2 of the matches used for player rating: ``common.training_mask``
    of the ``balls`` rows, which leaves out the Super_Over_Innings and every match excluded under
    Requirement 5, criterion 24 (the ``excluded`` flag, set while ``exclude_failing_matches`` is
    on). It is the filter that model training applies too, so every Validation_Delivery lies in a
    match that the fits of the other folds could train on. The
    Pressure_Engine sets its ranks, flags and shares over every Delivery of an Innings_Group, those
    of excluded matches included, so this module selects the Validation_Deliveries after the flags
    are set and never ranks or flags them again. Excluded matches are whole matches and
    Super_Over_Innings whole innings, so the next Delivery of a Validation_Delivery's innings is
    always a Validation_Delivery too.

The Out_Of_Fold guard (``guard_inputs``, ``load_guard_inputs``, ``oof_guard``, GuardInputs,
GuardResult, ValidationLeakageError; criterion 11)
    Runs first, before any result is computed or any file is written. ``guard_inputs`` lines up,
    for every Delivery of innings 1 and 2 in ``state`` order, its match, innings, fold (from the
    Fold_Table), Validation_Delivery flag and Batter_Delivery flag (``deliveries``); its WP, LI
    and PI values (``values``); and the fit that stands behind each model input of those values
    (``sources``, FIT_SOURCES):

    ======================= ===================================================================
    Source                  The fit
    ======================= ===================================================================
    wp_fit, wp_calibrator   the ``fit_id`` and ``calibrator_fit_id`` that the M3 Primary_Model's
                            WP table of the ``win_prob`` checkpoint records for the row (the
                            outer classifier and its centred isotonic calibrator behind the
                            pre-ball WP)
    wp_outer_fit,           the outer fit and calibrator of the M3 Primary_Model for the row's
    wp_outer_calibrator     innings and fold (``m3_inn<i>_<family>_outer_k<k>``,
                            ``m3_inn<i>_cal_<family>_outer_k<k>``): the models of
                            ``CalibratedFolds`` that scored the row's Post_Outcome_States (the
                            LI) and that score its end-of-innings state below
    m4_fit, m4_m1_fit       the M4 Primary_Model's fold model of the row's fold (its fold-model
                            table) and the M1 fit behind that fold's Average_Skill (the LI's P)
    view_m1_fit,            the ``m1_fit`` and ``m2_fit`` of the row's own-fold view row (V_k for
    view_m2_fit             a row of fold k): the fits behind every model feature and Stress_Pillar
                            input of the row
    ======================= ===================================================================

    ``oof_guard(inputs, registry)`` then checks each Validation_Delivery (the CHECKS, with
    ``registry`` the Run's FitRegistry, ``ctx.registry``, which holds every fit of stages 06-10):

    1. ``value``: the row has a pre-ball WP within [0, 1], a finite LI of at least 0 and, for each
       of its Roles, a PI within [0, 100] (``pi_bat`` on a Batter_Delivery, ``pi_bowl`` and
       ``pi_bowl_extras_free`` on every Delivery);
    2. ``agreement``: the pressure table's WP equals the Primary_Models' Out_Of_Fold WP
       (``model_choice.load_primary``) and the WP of the M3 table that names its fits, and the
       pressure table's LI equals the LI recomputed from the Primary_Models' Out_Of_Fold
       predictions (``pressure.primary_leverage``), bit for bit, so the values that the
       validation reads are those whose fits the guard checks;
    3. ``fold``: the fold stored with the predictions equals the Fold_Table's fold of the match;
    4. ``fit_registered``, ``fit_excludes_match``, ``fit_serves_match``: every source names a fit
       that the registry holds, whose excluded folds contain the row's fold and whose
       Training_Rows (training and early-stopping matches) hold no Delivery of the row's match, and
       whose predicted matches include the row's match (the fit is the one that gave the row its
       output);
    5. ``sandbox``: no fit of the row's fold sandbox (``sandbox_k`` = the row's fold) of the models
       behind the row's WP, LI and PI (``sandbox_models``: M1, the first-innings M2, the M3
       Primary_Model of the row's innings and its calibrator, the M4 Primary_Model) trained on the
       row's match. These are the inner fits whose outputs became training features or calibration
       inputs of the outer fits (Requirement 2, criterion 6).

    PI reads only the LI, the WP (the Live_Ball flag) and Stress, whose model inputs are the
    own-fold view values, so checks 2 and 4 cover every model output behind the PI; the
    Extras_Model feeds no PI (Requirement 13, criterion 6). When any check fails,
    ValidationLeakageError (a ``LeakageError``, exit code 4) stops the Run: its message gives the
    number of affected Validation_Deliveries, the count of each failed check and source with
    example rows, and says that no validation table or chart was written. ``affected``,
    ``n_validation``, ``rows`` and ``table`` hold the same facts. Nothing has been saved or written
    at that point, so the Run writes no validation output (criterion 11). Otherwise the guard
    returns a GuardResult whose ``table`` (GUARD_COLUMNS: one row per check and source with the
    fits and rows checked and the affected count, all 0) the stage writes as GUARD_FILE.

WP after a Delivery and Actual_WP_Swing (``actual_wp_swing``; glossary)
    Actual_WP_Swing = |WP after the Delivery - WP before it|, with WP the batting side's win
    probability. WP before is the Delivery's pre-ball Out_Of_Fold WP of the M3 Primary_Model of
    its innings (``wp_pre``, after the terminal rules). Taking the difference between consecutive
    pre-ball WPs would put the jump at the innings break (the innings-1 WP of the batting-first
    side against 1 - the innings-2 WP of the chasing side, mean absolute 0.085 in
    ``outputs/models/m3/innings_break_gap.csv``) on the first innings-2 ball, so WP after is
    defined innings by innings (WP_AFTER_RULES):

    * ``next_delivery``: WP after = the pre-ball WP of the next Delivery of the same innings
      (next ``row``). Every Delivery but the last of its innings.
    * ``innings1_end_state``: the last Delivery of innings 1, when its actual outcome ends the
      innings (its Legal_Balls bowled after the Delivery reach Scheduled_Balls, or 10
      Counted_Wickets have fallen after it). WP after = the innings-1 Primary_Model's WP at the
      end-of-innings state of the actual outcome (Requirement 10, criterion 17), built as
      ``pressure.post_outcome_states`` builds it for an outcome that ends the innings and as
      ``win_prob.innings_break_gap`` builds it for the actual outcome: Legal_Balls bowled =
      Scheduled_Balls, the actual final score (pre-ball score + the Delivery's team runs under the
      selected Extras_Encoding), the wickets fallen before the Delivery, Par_Score at
      Scheduled_Balls (``win_prob.innings_end_par`` with the first-innings M2 fit that the row's
      own-fold view names in ``m2_fit``, ``m2_par_means`` and ``m2_par_bias``), and the row's
      Team_Strength_Gap, Scoring_Environment, Venue_Effect and Scheduled_Balls. It is scored by
      the calibrated outer model of the row's fold (``win_prob.load_fold_models``,
      ``win_prob.win_probability``), the model behind the row's pre-ball WP. The state maps
      cleanly: for the matches with both innings it is the state whose WP ``innings_break_gap``
      reports as ``wp1_end``.
    * ``innings1_stopped_state``: the last Delivery of innings 1 when its actual outcome does not
      end the innings: the innings stopped there (rain, an abandoned match or records that end
      early, while Scheduled_Balls still holds the full innings). The end-of-innings state would
      then score the stopped total as a complete innings of Scheduled_Balls, a counterfactual
      that has nothing to do with the ball. WP after is instead the innings-1 Primary_Model's WP
      (same fold model) at the actual state after the Delivery, exactly the pre-ball state that a
      next Delivery would have had: Legal_Balls bowled + 1 for a Legal_Ball (+ 0 for a Wide or
      No_Ball), the actual score after the Delivery, the wickets fallen after it, Par_Score at
      that Legal_Ball count (the view's ``par_score_next`` after a Legal_Ball, ``par_score``
      otherwise) and the other features unchanged. For a Legal_Ball whose team runs equal its
      class's Class_Run_Value this is the actual class's Post_Outcome_State; unlike that state
      it also takes the actual runs of byes and leg-byes and serves a Wide or No_Ball.
    * ``innings2_result``: the last Delivery of innings 2 of a match won by runs or by wickets:
      the match result for the batting (chasing) side, 1 for a win and 0 for a loss (the
      ``m3_labels`` label of the row: the winner is the batting side). The result is used even
      where the reconstructed final score disagrees with it (unrecorded bye and Wide runs,
      design finding F-4).
    * ``innings2_tie``: the last Delivery of innings 2 of a tied match or of a match decided by a
      super over: 0.5 (TIED_WP, Requirement 10, criterion 16), the regulation result. The super
      over is a Super_Over_Innings and no Validation_Delivery, so its outcome is credited to no
      innings-2 ball.
    * ``innings2_terminal_rule``: the last Delivery of innings 2 of a match without a decisive
      result (no result, abandoned, unparsed): the terminal-rule WP (``win_prob.terminal_rules``,
      ``apply_terminal_rules``) of the final state: runs needed - the Delivery's team runs,
      Legal_Balls left - 1 for a Legal_Ball (at least 0), wickets + its Counted_Wicket, when a
      rule decides that state (chase won, lost, tied or fractional).
    * ``innings2_post_outcome``: the same Deliveries when no terminal rule decides the final state
      (the chase was interrupted): the post-outcome WP of the class that matches the actual
      outcome (``outcome.label_outcomes`` of the bat runs and wicket flags; the column of
      ``wp_post`` of the M3 Primary_Model for innings 2, the Out_Of_Fold WP of that class's
      Post_Outcome_State after the terminal rules). The class rule is that of a Legal_Ball, so a
      Wide or No_Ball here would be read as a Legal_Ball with the same bat runs and wickets (no
      such Delivery as delivered).

    Other cases: matches without innings-2 Deliveries (abandoned or no result after innings 1)
    give their last innings-1 Delivery the innings-1 rules above and have no innings-2 row; a
    pre-ball innings-2 state whose chase is already won has WP 1 from the terminal rules and takes
    the rules above like any other Delivery; Super_Over_Innings are no Validation_Deliveries. Every
    WP comes from the Primary_Models' Out_Of_Fold predictions or the recorded result, never from a
    fit that saw the match. WP_AFTER_COLUMNS counts the Deliveries of each rule (WP_AFTER_FILE).

V1: PI deciles (``v1_deciles``; criterion 2)
    For each Role (batter: the Batter_Deliveries; bowler: every Delivery) and Innings_Group, the
    Role's Live_Balls among the Validation_Deliveries are sorted by PI, ties in PI by Delivery
    order (``row``), and position p (0-based) of n goes to decile floor(10 x p / n) + 1, so decile
    1 holds the lowest PI and the 10 counts differ by at most 1 (Property 39). Live_Balls with the
    same PI can therefore fall in two neighbouring deciles. A separate ``dead`` row holds the
    Role's Dead_Balls (PI 0). Each row (V1_COLUMNS) gives the Delivery count, the PI range and the
    mean Actual_WP_Swing.

V2: Big_Moments (``big_moments``, ``v2_precision_recall``; criteria 3 and 13)
    A match's Big_Moments are the BIG_MOMENTS_PER_MATCH (10) of its Validation_Deliveries with
    the largest Actual_WP_Swing: the match's Deliveries sorted by swing, largest first, equal
    swings by Delivery order, and the first 10 taken, so a tie at the 10th-largest swing goes to
    the earlier Delivery (criterion 13). A match with fewer than 10 Validation_Deliveries has all
    of them as Big_Moments; V2 leaves such matches out. For each of the four METHODS (the batter
    and bowler Pressure_Balls and the two SIMPLE_RULES), pooled over the matches with at least
    MIN_MATCH_DELIVERIES (10) Validation_Deliveries: the share of Validation_Deliveries flagged,
    precision = flagged Big_Moments / flagged Deliveries and recall = flagged Big_Moments / all
    Big_Moments (V2_COLUMNS). A Wide is not a Batter_Delivery, so the batter method never flags it.
    ``big_moment_share`` (all Big_Moments / all Deliveries) is the precision of a method that flags
    Deliveries at random.

V3: extras and within-player effects (``v3_extras``, ``v3_extras_free``, ``within_player_effect``,
``v3_within_player``; criteria 4-6 and 12)
    Extras: Wides plus No_Balls per 100 Bowler_Deliveries on the bowler-Role Pressure_Balls and on
    the bowler-Role Ordinary_Balls (every other Bowler_Delivery, Dead_Balls included), their
    difference (Pressure - Ordinary) and the Bowler_Delivery count behind each rate, per
    Innings_Group and for both together (V3_EXTRAS_COLUMNS). ``v3_extras`` uses the Pressure_Ball
    flags of the Rating_Engine; ``v3_extras_free`` the Extras_Free_PI selection (criterion 5):
    the top D3 (25%) of the Live_Balls of each Innings_Group by Extras_Free_PI, the bowler PI with
    the boundary-only F25(b), which the pressure table holds as ``pressure_bowl_extras_free``.
    ``extras_free_flags`` re-selects them from ``pi_bowl_extras_free`` within each Innings_Group
    over every Live_Ball (excluded matches included, as the Pressure_Engine ranks) with the
    Run's D3; the stage checks that they equal the saved flags (they do whenever D4 is the
    Innings_Group) and warns and uses the re-selection when they differ. The Rating_Engine's flags
    never change.

    Within-player effects: for one outcome y per Delivery, a pressure indicator d (1 on a
    Pressure_Ball of the Role, 0 on an Ordinary_Ball), the player and the match (the cluster),
    ``within_player_effect`` fits y = beta x d + one fixed effect per player by demeaning y and d
    within player: beta = sum(d~ y~) / sum(d~^2), the coefficient of OLS with one dummy per player
    (Frisch-Waugh-Lovell). Its match-clustered variance is (sum d~^2)^-2 x sum over matches g of
    (sum over i in g of d~_i e_i)^2, with e the within residuals, times the small-sample factor
    G / (G - 1) x (N - 1) / (N - K), with G matches, N Deliveries and K = players + 1. The 95%
    confidence interval is beta +/- t x SE with t the 97.5% quantile of the t distribution with
    G - 1 degrees of freedom (CI_LEVEL). Players with only Pressure_Balls or only Ordinary_Balls
    have d~ = 0: they add nothing to beta but count in N and K, as in the dummy regression
    (``n_players_both`` counts the players with both). ``v3_within_player`` applies it to three
    outcomes over the Validation_Deliveries (V3_EFFECT_COLUMNS):

    * batter strike rate: y = 100 x bat runs (``runs``) on each Batter_Delivery, d the batter
      Pressure_Ball flag, the striker as player (criterion 6);
    * batter dismissal rate: y = 100 x (a Counted_Wicket other than Retire Out whose dismissed
      batter is the striker, ``dismissed_is_striker`` with the fallback of Requirement 15,
      criterion 10) on each Batter_Delivery (criterion 6);
    * bowler economy: y = 6 x Bowler_Runs_Conceded on each Bowler_Delivery, d the bowler
      Pressure_Ball flag, the bowler as player (criterion 12). Bowler_Runs_Conceded is the team
      runs under the selected Extras_Encoding minus byes and leg-byes (glossary): 0 on a Delivery
      whose ``extras_type`` is ``byes`` or ``legbyes`` (all its team runs are byes or leg-byes),
      its team runs on every other Delivery (Wide and No_Ball runs included).

    beta is in the outcome's unit (runs per 100 Batter_Deliveries, dismissals per 100
    Batter_Deliveries, runs per 6 Bowler_Deliveries); ``mean_pressure`` and ``mean_ordinary`` give
    the raw pooled means for scale.

V4: Simple_Rules (``simple_rule_flags``, ``high_leverage_balls``, ``v4_tables``; criteria 7-8)
    The Simple_Rules (glossary): (a) overs 16-20 (``overid`` 16-20 of either innings,
    DEATH_OVERS); (b) a second-innings pre-ball Required_Rate above 10 runs per over
    (CHASE_RATE_ABOVE; no rule when the Required_Rate is missing). The dead-share table
    (V4_DEAD_COLUMNS) gives, for each rule and Innings_Group (and both together), the
    Validation_Deliveries the rule flags, the Dead_Balls among them and their share. A
    High_Leverage_Ball is a Live_Ball whose LI is in the top 25% (HIGH_LEVERAGE_SHARE) of the
    Live_Balls of its Innings_Group: Percentile_Rank of LI > 0.75, the rule of the Pressure_Ball
    flag (``pressure.pressure_flags``), so equal LI get the same flag, over every Live_Ball of the
    Innings_Group as the Pressure_Engine ranks. The unflagged table (V4_UNFLAGGED_COLUMNS) gives,
    per Innings_Group (and both), the High_Leverage_Balls among the Validation_Deliveries with a
    Ball_Position_Share of at most the upper D8 split (0.75), those flagged by neither Simple_Rule
    and their share.

Signal-model matrix (``signal_model_matrix``; criterion 9)
    A declared dependency map, not a computation on data. MODEL_INPUTS lists the models whose
    outputs each model reads as features (Requirement 7, criteria 7 and 8), PI_INPUTS the models
    whose outputs the PI reads directly and SIGNALS the models that each validation signal reads
    directly. A model feeds a quantity when it is among its direct inputs or feeds one of them
    (closure through MODEL_INPUTS). MATRIX_COLUMNS gives one row per signal and, for each of M1,
    M2, M3, M4 and the Extras_Model, True when the model's outputs feed both the signal and the
    PI. The extras, strike-rate, dismissal-rate and economy signals read only recorded outcomes,
    so no model feeds them; the Actual_WP_Swing, the Big_Moments and the Dead_Ball share share
    M1-M3 with the PI, and the High_Leverage_Ball share M1-M4: those checks are not independent of
    the models behind the PI, which the report says.

The ``validation`` Pipeline_Stage (``run_stage``)
    Reads ``balls`` (``reconstruct``), ``state`` (``state``), the Fold_Table (``folds``), the views
    V_1 ... V_D15, the outer first-innings M2 fits and ``m2_par_means`` and ``m2_par_bias``
    (``remaining_runs``), the M3 Primary_Models' WP tables (fit ids), outer fold models and
    ``m3_labels`` (``win_prob``), the M4 Primary_Model's fold-model table (``outcome``), the
    Primary_Models' predictions (``model_choice.load_primary``, which reads ``model_choice`` and
    ``outcome``) and the pressure table (``pressure``). Then:

    1. runs the Out_Of_Fold guard (exit code 4 on failure, before anything is computed, saved or
       written);
    2. builds the per-Delivery frame of the Validation_Deliveries (FRAME_COLUMNS): the pressure
       table's values and flags joined on ``row``, the High_Leverage_Ball and Simple_Rule flags,
       the re-selected Extras_Free flags, WP before and after, Actual_WP_Swing, Big_Moments and
       the outcomes;
    3. computes V1-V4 and the signal-model matrix;
    4. saves the frame and every table to its checkpoint;
    5. writes OUTPUT_FILES and CHART_FILES to ``outputs/validation/`` (VALIDATION_FOLDER): every
       table as CSV (UTF-8, ``\\n`` line ends, floats ``%.6f``, NaN blank) and one or more PNG
       charts for each of V1-V4 (matplotlib's Agg canvas, CHART_DPI, no time stamp), each written
       through ``<file>.tmp`` renamed in place;
    6. records the data-derived Method_Constant HIGH_LEVERAGE_CUT_CONSTANT (Pressure_Validator):
       the lowest LI of a High_Leverage_Ball in each Innings_Group, the realised cut of the 25%
       rule. Nothing is recorded as a Config value or a Design_Choice;
    7. logs the guard, the Validation_Delivery counts, the WP-after rules and the main results.

As delivered (``--from-stage validation`` on the checkpoints of 14 November 2026)
    The stage takes 15-25 s. The guard passes: 189,648 Validation_Deliveries (99,751 in innings
    1, 89,897 in innings 2) in 819 matches (797 with innings-2 Deliveries), 115 distinct fits
    checked, 85 sandbox fits per innings. WP after: 188,032 ``next_delivery``; 794
    ``innings1_end_state``, whose WP equals ``wp1_end`` of ``innings_break_gap.csv`` to the file's 6
    decimals for the 788 of them that it lists (the other 6 matches have no innings 2); 25
    ``innings1_stopped_state`` (3 abandoned and 13 no-result matches, and 9 decided matches whose
    first innings ends early in the records, one of them on a Wide); 786 ``innings2_result``; 9
    ``innings2_tie`` (the super-over matches; their last balls swing 0.48 on average, because the
    WP before the last ball lies near 0 or 1, from reconstructed final scores that are not level
    in 4 of them or from the end-of-chase WPs described below); 2 ``innings2_post_outcome`` (the
    two no-result matches with an interrupted chase); no ``innings2_terminal_rule``.

    * V1: the mean Actual_WP_Swing rises from decile 1 to decile 10 in every Role and
      Innings_Group: batter 0.0030 to 0.0166 (innings 1) and 0.0090 to 0.0487 (innings 2); bowler
      0.0030 to 0.0187 and 0.0078 to 0.0400; the Dead_Balls average 0.0030-0.0042.
    * V2 (818 matches, 189,646 Deliveries, 8,180 Big_Moments, Big_Moment share 0.043): batter
      Pressure_Balls flag 20.9% of the Deliveries with precision 0.081 and recall 0.394, bowler
      Pressure_Balls 21.9%, 0.085 and 0.431, overs 16-20 22.2%, 0.050 and 0.259, a chase
      Required_Rate above 10 21.0%, 0.060 and 0.291.
    * V3 extras: 4.66 Wides plus No_Balls per 100 Bowler_Deliveries on bowler Pressure_Balls
      against 4.13 on Ordinary_Balls (+0.53; innings 1 +0.45, innings 2 +0.63); the Extras_Free_PI
      selection, identical to the saved flags, gives +0.48 (+0.36, +0.63).
    * V3 within-player effects (819 matches, t with 818 degrees of freedom): strike rate -8.38
      runs per 100 Batter_Deliveries (95% CI -10.55 to -6.21; 882 batters, 751 with both kinds of
      ball, 182,377 Batter_Deliveries), dismissal rate -0.03 per 100 (-0.27 to +0.22), economy
      +1.35 runs per 6 Bowler_Deliveries (+1.23 to +1.48; 644 bowlers, 189,648 Bowler_Deliveries).
    * V4: Dead_Ball share of overs 16-20 0.039 in innings 1, 0.558 in innings 2 and 0.256 overall;
      of a chase Required_Rate above 10, 0.331. High_Leverage_Balls up to a Ball_Position_Share of
      0.75: all 17,284 of innings 1 are unflagged (overs 16-20 start at 91/120 > 0.75 and rule (b)
      is for chases only), 6,456 of the 11,196 of innings 2 (0.577), 0.834 overall. The lowest LI of
      a High_Leverage_Ball is 1.055 in innings 1 and 1.865 in innings 2.

    65 Validation_Deliveries swing more than 0.5, 36 of them with at most 2 Legal_Balls left in a
    chase. The largest next-Delivery swing (0.963, match 14109) is a six off the 119th ball: the
    next Delivery, 7 runs needed off the last ball, has a pre-ball WP of 0.997 because the
    first-innings M2's expected remaining runs with one Legal_Ball left (median 4.9, up to 8.9
    runs) give an Achievable_Rate above the Required_Rate of 42. These are the Primary_Models'
    Out_Of_Fold WPs; the validation reports them as they are.

Errors: ValidationLeakageError (exit code 4) from the guard; StageError (exit code 5) for a
checkpoint file that is missing or changed, an M3 Primary_Model whose WP table is not in the
``win_prob`` checkpoint (an Alternative_Model of task 26) and an empty set of
Validation_Deliveries; ValueError or TypeError for malformed arguments, such as a missing column,
tables that do not hold the same rows, an Actual_WP_Swing that is missing, or a within-player
design without two clusters, without more Deliveries than players + 1 or without within-player
variation in d.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from superr import folds, leakage, pressure, reconstruct, skill, views
from superr.errors import LeakageError, StageError
from superr.models import model_choice, outcome, remaining_runs, win_prob
from superr.models.common import training_mask
from superr.pressure import BATTER, BOWLER, LI_VARIANT, ROLES
from superr.reconstruct import DeliveryClass
from superr.state import STATE_INNINGS, STATE_TABLE
from superr.state import STAGE_NAME as STATE_STAGE

if TYPE_CHECKING:
    from superr.leakage import FitRegistry
    from superr.models.model_choice import PrimaryPredictions
    from superr.pipeline import StageContext
    from superr.store import CheckpointStore

__all__ = [
    "BIG_MOMENTS_PER_MATCH",
    "BOTH_INNINGS",
    "CHART_DPI",
    "CHART_FILES",
    "CHASE_RATE_ABOVE",
    "CHECKS",
    "CI_LEVEL",
    "DEATH_OVERS",
    "FIT_SOURCES",
    "FRAME_COLUMNS",
    "FRAME_TABLE",
    "GUARD_COLUMNS",
    "GUARD_DELIVERY_COLUMNS",
    "GUARD_FILE",
    "GUARD_VALUE_COLUMNS",
    "HIGH_LEVERAGE_CUT_CONSTANT",
    "HIGH_LEVERAGE_SHARE",
    "MATRIX_COLUMNS",
    "METHODS",
    "METHOD_BATTER",
    "METHOD_BOWLER",
    "MIN_MATCH_DELIVERIES",
    "MODEL_INPUTS",
    "MODEL_NAMES",
    "N_DECILES",
    "OUTPUT_FILES",
    "PI_INPUTS",
    "RULE_CHASE_RATE",
    "RULE_DEATH_OVERS",
    "SIGNALS",
    "SIMPLE_RULES",
    "STAGE_NAME",
    "SWING_INPUT_COLUMNS",
    "TABLES",
    "TIED_WP",
    "V1_COLUMNS",
    "V2_COLUMNS",
    "V3_EFFECT_COLUMNS",
    "V3_EXTRAS_COLUMNS",
    "V4_DEAD_COLUMNS",
    "V4_UNFLAGGED_COLUMNS",
    "VALIDATION_FOLDER",
    "WP_AFTER_COLUMNS",
    "WP_AFTER_RULES",
    "EffectEstimate",
    "GuardInputs",
    "GuardResult",
    "Signal",
    "ValidationLeakageError",
    "actual_wp_swing",
    "big_moments",
    "bowler_runs_conceded",
    "extras_free_flags",
    "guard_inputs",
    "high_leverage_balls",
    "load_guard_inputs",
    "oof_guard",
    "run_stage",
    "sandbox_models",
    "signal_model_matrix",
    "simple_rule_flags",
    "striker_dismissals",
    "v1_deciles",
    "v2_precision_recall",
    "v3_extras",
    "v3_extras_free",
    "v3_within_player",
    "v4_tables",
    "validation_mask",
    "within_player_effect",
    "wp_after_summary",
]


# ------------------------------------------------------------------------------------- constants

#: Name of the Pipeline_Stage, of its checkpoint, and of its folder under the outputs folder.
STAGE_NAME = "validation"
VALIDATION_FOLDER = "validation"

#: Resolution of every chart in dots per inch, fixed so that the same data give the same image.
CHART_DPI = 150

#: Method_Constants that the requirements fix for the validation (hand-set by the plan and the
#: glossary; none is a Design_Choice and none is chosen from data or from a validation result).
N_DECILES = 10  # V1: 10 equal-count PI deciles per Role and Innings_Group (criterion 2)
BIG_MOMENTS_PER_MATCH = 10  # Big_Moment: the 10 largest Actual_WP_Swing of a match (glossary)
MIN_MATCH_DELIVERIES = 10  # V2 pools the matches with at least 10 Validation_Deliveries (criterion 3)
DEATH_OVERS: tuple[int, int] = (16, 20)  # Simple_Rule (a): overs 16-20, both included (glossary)
CHASE_RATE_ABOVE = 10.0  # Simple_Rule (b): second-innings Required_Rate above 10 runs per over (glossary)
HIGH_LEVERAGE_SHARE = 0.25  # High_Leverage_Ball: LI in the top 25% of its Innings_Group's Live_Balls (glossary)
CI_LEVEL = 0.95  # the within-player effects' confidence intervals (criteria 6 and 12)

#: The WP after the last innings-2 Delivery of a tied match (scores level; Requirement 10, criterion 16).
TIED_WP = 0.5

#: The four V2 methods (criterion 3): the Pressure_Balls of each Role and the two Simple_Rules.
METHOD_BATTER = "batter_pressure_balls"
METHOD_BOWLER = "bowler_pressure_balls"
RULE_DEATH_OVERS = "overs_16_20"
RULE_CHASE_RATE = "chase_rrr_above_10"
SIMPLE_RULES: tuple[str, ...] = (RULE_DEATH_OVERS, RULE_CHASE_RATE)
METHODS: tuple[str, ...] = (METHOD_BATTER, METHOD_BOWLER, *SIMPLE_RULES)

#: The ``innings`` label of the V3 and V4 rows that pool both Innings_Groups.
BOTH_INNINGS = "both"
_INNINGS_LABELS: tuple[str, ...] = (*(str(number) for number in STATE_INNINGS), BOTH_INNINGS)

#: The ``group`` of a V1 row: one of the 10 PI deciles of the Live_Balls, or the Dead_Balls.
DECILE_GROUP = "decile"
DEAD_GROUP = "dead"

#: The ``selection`` of a V3 extras row: the Rating_Engine's bowler flags or the Extras_Free_PI ones.
MAIN_SELECTION = "main"
EXTRAS_FREE_SELECTION = "extras_free"

#: How the WP after a Delivery is defined (module docstring, "WP after a Delivery"), in table order.
AFTER_NEXT = "next_delivery"
AFTER_INNINGS1_END = "innings1_end_state"
AFTER_INNINGS1_STOPPED = "innings1_stopped_state"
AFTER_RESULT = "innings2_result"
AFTER_TIE = "innings2_tie"
AFTER_TERMINAL = "innings2_terminal_rule"
AFTER_POST_OUTCOME = "innings2_post_outcome"
WP_AFTER_RULES: tuple[str, ...] = (
    AFTER_NEXT, AFTER_INNINGS1_END, AFTER_INNINGS1_STOPPED, AFTER_RESULT, AFTER_TIE, AFTER_TERMINAL,
    AFTER_POST_OUTCOME,
)  # fmt: skip
_DECISIVE_RESULTS: tuple[str, ...] = (reconstruct.WIN_BY_RUNS, reconstruct.WIN_BY_WICKETS)
_TIED_RESULTS: tuple[str, ...] = (reconstruct.TIE, reconstruct.SUPER_OVER_WIN)

#: The columns that actual_wp_swing reads, one row per Validation_Delivery: the Delivery (balls),
#: its pre-ball state (state), its result label (m3_labels), its Out_Of_Fold pre-ball WP and the
#: own-fold view values of its end-of-innings and post-Delivery states.
SWING_INPUT_COLUMNS: tuple[str, ...] = (
    "row", "match_id", "innings", "fold", "cls", "runs", "team_runs", "counted_wicket", "bowler_wicket", "run_out",
    "sched_balls", "result_type", "label", "labelled", "lb_bowled", "lb_left", "score", "wickets", "runs_needed",
    "wp_pre", "tsg", "scoring_env", "venue_effect", "par_score", "par_score_next", views.M2_FIT,
)  # fmt: skip
SWING_COLUMNS: tuple[str, ...] = ("row", "wp_before", "wp_after", "swing", "wp_after_rule")
_SWING_VIEW_COLUMNS: tuple[str, ...] = ("tsg", "scoring_env", "venue_effect", "par_score", "par_score_next")

#: The per-Delivery frame of the Validation_Deliveries (checkpoint table FRAME_TABLE), in order.
FRAME_COLUMNS: tuple[str, ...] = (
    "row", "match_id", "innings", "fold", "cls", "batter", "overid", "ball_share", "rrr", "striker_id", "bowler_id",
    "bat_runs", "striker_out", "bowler_runs_conceded", "extra", "live", "li", "pi_bat", "pi_bowl",
    "pi_bowl_extras_free", "pressure_bat", "pressure_bowl", "pressure_bowl_extras_free", "high_leverage",
    RULE_DEATH_OVERS, RULE_CHASE_RATE, "wp_before", "wp_after", "wp_after_rule", "swing", "big_moment",
)  # fmt: skip

#: The validation tables (module docstring): their columns, checkpoint names and files.
V1_COLUMNS: tuple[str, ...] = ("role", "innings", "group", "decile", "n_deliveries", "pi_min", "pi_max", "mean_swing")
V2_COLUMNS: tuple[str, ...] = (
    "method", "n_matches", "n_deliveries", "n_big_moments", "n_flagged", "n_flagged_big_moments", "share_flagged",
    "precision", "recall", "big_moment_share",
)  # fmt: skip
V3_EXTRAS_COLUMNS: tuple[str, ...] = (
    "selection", "innings", "n_pressure", "n_extras_pressure", "rate_pressure", "n_ordinary", "n_extras_ordinary",
    "rate_ordinary", "difference",
)  # fmt: skip
V3_EFFECT_COLUMNS: tuple[str, ...] = (
    "measure", "role", "unit", "beta", "se", "ci_low", "ci_high", "t_crit", "df", "n_players", "n_players_both",
    "n_deliveries", "n_pressure", "n_clusters", "mean_pressure", "mean_ordinary",
)  # fmt: skip
V4_DEAD_COLUMNS: tuple[str, ...] = ("rule", "innings", "n_flagged", "n_dead", "dead_share")
V4_UNFLAGGED_COLUMNS: tuple[str, ...] = (
    "innings", "max_ball_share", "n_high_leverage", "n_unflagged", "unflagged_share",
)  # fmt: skip
MATRIX_COLUMNS: tuple[str, ...] = ("signal", "validation", "target", "models_feeding_signal", "M1", "M2", "M3", "M4",
                                   "Extras_Model")  # fmt: skip
WP_AFTER_COLUMNS: tuple[str, ...] = ("rule", "innings", "n_deliveries", "mean_wp_after", "mean_swing", "max_swing")
GUARD_COLUMNS: tuple[str, ...] = ("check", "source", "description", "n_fits", "n_rows", "n_affected")

FRAME_TABLE = "validation_deliveries"
V1_TABLE, V1_FILE = "v1_deciles", "v1_pi_deciles.csv"
V2_TABLE, V2_FILE = "v2_big_moments", "v2_big_moments.csv"
V3_EXTRAS_TABLE, V3_EXTRAS_FILE = "v3_extras", "v3_extras.csv"
V3_EXTRAS_FREE_TABLE, V3_EXTRAS_FREE_FILE = "v3_extras_free", "v3_extras_free.csv"
V3_EFFECTS_TABLE, V3_EFFECTS_FILE = "v3_within_player", "v3_within_player.csv"
V4_DEAD_TABLE, V4_DEAD_FILE = "v4_dead_share", "v4_simple_rule_dead_share.csv"
V4_UNFLAGGED_TABLE, V4_UNFLAGGED_FILE = "v4_high_leverage_unflagged", "v4_high_leverage_unflagged.csv"
MATRIX_TABLE, MATRIX_FILE = "signal_model_matrix", "signal_model_matrix.csv"
WP_AFTER_TABLE, WP_AFTER_FILE = "wp_after_rules", "wp_after_rules.csv"
GUARD_TABLE, GUARD_FILE = "oof_guard", "oof_guard.csv"
TABLES: Mapping[str, str] = MappingProxyType(
    {
        V1_TABLE: V1_FILE, V2_TABLE: V2_FILE, V3_EXTRAS_TABLE: V3_EXTRAS_FILE,
        V3_EXTRAS_FREE_TABLE: V3_EXTRAS_FREE_FILE, V3_EFFECTS_TABLE: V3_EFFECTS_FILE, V4_DEAD_TABLE: V4_DEAD_FILE,
        V4_UNFLAGGED_TABLE: V4_UNFLAGGED_FILE, MATRIX_TABLE: MATRIX_FILE, WP_AFTER_TABLE: WP_AFTER_FILE,
        GUARD_TABLE: GUARD_FILE,
    }
)  # fmt: skip
OUTPUT_FILES: tuple[str, ...] = tuple(TABLES.values())
V1_CHART = "v1_pi_deciles.png"
V2_CHART = "v2_big_moments.png"
V3_EXTRAS_CHART = "v3_extras.png"
V3_EFFECTS_CHART = "v3_within_player.png"
V4_CHART = "v4_simple_rules.png"
CHART_FILES: tuple[str, ...] = (V1_CHART, V2_CHART, V3_EXTRAS_CHART, V3_EFFECTS_CHART, V4_CHART)

#: Data-derived Method_Constant (Pressure_Validator): {"1": ..., "2": ...}, the lowest LI of a
#: High_Leverage_Ball in each Innings_Group (the realised cut of HIGH_LEVERAGE_SHARE).
HIGH_LEVERAGE_CUT_CONSTANT = "validation.lowest_high_leverage_li"

_COMPONENT = "Pressure_Validator"  # the glossary component that produces the constant
_SHARE = "design_choices.D3_pressure_share"
_PHASE_SPLITS = "design_choices.D8_phase_splits"
_N_FOLDS = "design_choices.D15_n_folds"
_FLOAT_FORMAT = "%.6f"
_SHOWN = 5  # example rows named in one error message
_PI_MAX = 100.0  # PI lies within [0, 100] (Requirement 13, criteria 2 and 3)
_LEGAL = int(DeliveryClass.LEGAL)
_EXTRA_CLASSES: tuple[int, ...] = (int(DeliveryClass.WIDE), int(DeliveryClass.NO_BALL))

#: The columns that the stage reads from ``balls`` for each ``state`` row.
_BALL_COLUMNS: tuple[str, ...] = (
    "row", "match_id", "innings", "overid", "striker_id", "bowler_id", "runs", "team_runs", "extras_type", "cls",
    "is_super_over", "excluded", "counted_wicket", "bowler_wicket", "run_out", "retire_out", "dismissed_is_striker",
    "sched_balls", "result_type",
)  # fmt: skip
_STATE_COLUMNS: tuple[str, ...] = (
    "row", "lb_bowled", "lb_left", "score", "wickets", "runs_needed", "rrr", "ball_share",
)  # fmt: skip
_PRESSURE_COLUMNS: tuple[str, ...] = (
    "row", "wp_pre", "live", "li", "pi_bat", "pi_bowl", "pi_bowl_extras_free", "pressure_bat", "pressure_bowl",
    "pressure_bowl_extras_free",
)  # fmt: skip
_GUARD_BALL_COLUMNS: tuple[str, ...] = ("row", "match_id", "innings", "cls", "is_super_over", "excluded")
_M3_TABLE_COLUMNS: tuple[str, ...] = ("row", "fold", "wp_pre", "fit_id", "calibrator_fit_id")
_M4_FOLD_COLUMNS: tuple[str, ...] = ("fold", "fit_id", "m1_fit")


# ---------------------------------------------------------------------------------- small helpers

def _rows(count: int) -> str:
    return f"{count:,} row" if count == 1 else f"{count:,} rows"


def _plural(count: int, one: str, many: str | None = None) -> str:
    return f"{count:,} {one if count == 1 else (many if many is not None else one + 's')}"


def _require(frame: Any, columns: Sequence[str], what: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{what} must be a DataFrame; got {type(frame).__name__}")
    absent = [column for column in columns if column not in frame.columns]
    if absent:
        raise ValueError(f"{what} lacks the column(s) {', '.join(absent)}")


def _series(values: Any, what: str) -> pd.Series:
    if values is None or isinstance(values, (str, bytes, Mapping, pd.DataFrame)) or np.ndim(values) != 1:
        raise TypeError(f"{what} must be a 1-D array or Series; got {type(values).__name__}")
    if isinstance(values, pd.Series):
        return values.reset_index(drop=True)
    return pd.Series(values)


def _length(array: np.ndarray, what: str, n_rows: int | None) -> np.ndarray:
    if n_rows is not None and len(array) != n_rows:
        raise ValueError(f"{what} has {len(array):,} values, but {n_rows:,} are needed, one per row")
    return array


def _floats(values: Any, what: str, n_rows: int | None = None) -> np.ndarray:
    """``values`` (1-D numbers) as a new float64 array, NaN for a missing value."""
    series = _series(values, what)
    if len(series) and (not pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype)):
        raise TypeError(f"{what} must hold numbers; got dtype {series.dtype}")
    return _length(series.to_numpy(dtype=np.float64, na_value=np.nan).copy(), what, n_rows)


def _ints(values: Any, what: str, n_rows: int | None = None) -> np.ndarray:
    """``values`` (1-D whole numbers, none missing) as a new int64 array."""
    numbers = _floats(values, what, n_rows)
    if np.isnan(numbers).any() or not np.array_equal(numbers, np.round(numbers)):
        raise ValueError(f"{what} must hold whole numbers without a missing value")
    return numbers.astype(np.int64)


def _flags(values: Any, what: str, n_rows: int | None = None) -> np.ndarray:
    """``values`` (1-D booleans, none missing) as a new bool array."""
    series = _series(values, what)
    if len(series) and not pd.api.types.is_bool_dtype(series.dtype):
        raise TypeError(f"{what} must be a boolean mask; got dtype {series.dtype}")
    if series.isna().any():
        raise ValueError(f"{what} has {int(series.isna().sum()):,} missing flag(s)")
    return _length(series.to_numpy(dtype=bool).copy(), what, n_rows)


def _texts(values: Any, what: str, n_rows: int | None = None) -> np.ndarray:
    """``values`` (1-D text) as an object array of str, None for a missing value."""
    series = _series(values, what)
    out = np.array([None if pd.isna(value) else str(value) for value in series.tolist()], dtype=object)
    return _length(out, what, n_rows)


def _positions(row_ids: Any, wanted: np.ndarray, what: str) -> np.ndarray:
    """The position in ``row_ids`` of each of ``wanted`` (-1 where absent); ValueError for a repeated id."""
    index = pd.Index(_ints(row_ids, f"{what} column 'row'"))
    if not index.is_unique:
        raise ValueError(f"{what} must hold each row once")
    return np.asarray(index.get_indexer(wanted), dtype=np.int64)


def _aligned(frame: pd.DataFrame, row_ids: np.ndarray, columns: Sequence[str], what: str) -> pd.DataFrame:
    """The ``columns`` of ``frame`` for each of ``row_ids``, in that order, on a RangeIndex."""
    _require(frame, columns, what)
    at = _positions(frame["row"], row_ids, what)
    if (at < 0).any():
        raise ValueError(f"{_rows(int((at < 0).sum()))} of the state table have no row in {what}; pass the tables of one Run")
    return frame.iloc[at][list(columns)].reset_index(drop=True)


def _share(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else math.nan


def _shown_rows(rows: np.ndarray) -> str:
    shown = ", ".join(map(str, rows[:_SHOWN].tolist()))
    return shown + (f" and {len(rows) - _SHOWN:,} more" if len(rows) > _SHOWN else "")


# =================================================================== Validation_Deliveries and outcomes

def validation_mask(balls: pd.DataFrame) -> np.ndarray:
    """True for each Validation_Delivery among the rows of ``balls`` (module docstring; glossary).

    ``balls`` needs ``innings``, ``is_super_over`` and ``excluded``, as the ``balls`` table of the
    ``reconstruct`` checkpoint holds them: innings 1 or 2, no Super_Over_Innings, no excluded
    match (``common.training_mask``). Returns a bool array in row order.
    """
    return training_mask(balls)


def bowler_runs_conceded(team_runs: Any, extras_type: Any) -> np.ndarray:
    """Bowler_Runs_Conceded of each Bowler_Delivery (glossary): team runs minus byes and leg-byes.

    ``team_runs``: the team runs under the selected Extras_Encoding (``team_runs`` of ``balls``);
    ``extras_type``: the Delivery's ``extras_type`` text. A Delivery typed ``byes`` or ``legbyes``
    (``reconstruct.BYE_EXTRAS_TYPES``) concedes 0, because every team run of it is a bye or
    leg-bye; every other Delivery concedes its team runs, Wide and No_Ball runs included. Returns
    float64. Raises ValueError for a missing or negative team-runs value.
    """
    runs = _floats(team_runs, "team_runs")
    kinds = _texts(extras_type, "extras_type", len(runs))
    if np.isnan(runs).any() or (runs < 0).any():
        raise ValueError("team_runs must hold 0 or more runs on every Delivery")
    out = runs.copy()
    out[np.isin(kinds.astype(str), reconstruct.BYE_EXTRAS_TYPES)] = 0.0
    return out


def striker_dismissals(counted_wicket: Any, retire_out: Any, dismissed_is_striker: Any) -> np.ndarray:
    """True for each Delivery on which the striker falls to a Counted_Wicket other than Retire Out (criterion 6).

    ``counted_wicket`` and ``retire_out``: the ``balls`` flags; ``dismissed_is_striker``: the
    ``balls`` flag (nullable, <NA> without a Counted_Wicket) that the dismissed batter, resolved as
    in Requirement 15, criteria 9 and 10 (the striker when the id is blank or unmatched), is the
    striker. Returns a bool array.
    """
    counted = _flags(counted_wicket, "counted_wicket")
    retired = _flags(retire_out, "retire_out", len(counted))
    striker = _series(dismissed_is_striker, "dismissed_is_striker")
    if len(striker) and not pd.api.types.is_bool_dtype(striker.dtype):
        raise TypeError(f"dismissed_is_striker must be a boolean mask; got dtype {striker.dtype}")
    flag = _length(striker.fillna(False).to_numpy(dtype=bool), "dismissed_is_striker", len(counted))
    return counted & ~retired & flag


def simple_rule_flags(overid: Any, innings: Any, rrr: Any) -> dict[str, np.ndarray]:
    """The two Simple_Rule flags of each Delivery (glossary; module docstring, "V4").

    ``overid``: the over number (1 for the first Over); ``innings``: 1 or 2; ``rrr``: the
    pre-ball Required_Rate (NaN in innings 1 and where it is undefined). Returns RULE_DEATH_OVERS
    -> overs DEATH_OVERS (16-20) of either innings, and RULE_CHASE_RATE -> innings 2 with a
    Required_Rate above CHASE_RATE_ABOVE (10); a missing Required_Rate meets no rule.
    """
    over = _floats(overid, "overid")
    number = _ints(innings, "innings", len(over))
    rate = _floats(rrr, "rrr", len(over))
    with np.errstate(invalid="ignore"):
        death = (over >= DEATH_OVERS[0]) & (over <= DEATH_OVERS[1])
        chase = (number == STATE_INNINGS[1]) & (rate > CHASE_RATE_ABOVE)
    return {RULE_DEATH_OVERS: death, RULE_CHASE_RATE: chase}


def high_leverage_balls(li: Any, live: Any, innings: Any) -> np.ndarray:
    """The High_Leverage_Ball flag of each Delivery of innings 1 and 2 (glossary; module docstring, "V4").

    One value per Delivery, over every Delivery of the Innings_Groups (excluded matches included):
    ``li`` the LI, ``live`` the Live_Ball flags, ``innings`` 1 or 2. A Live_Ball is a
    High_Leverage_Ball when the Percentile_Rank of its LI among the Live_Balls of its Innings_Group
    is greater than 1 - HIGH_LEVERAGE_SHARE (0.75), the rule of ``pressure.pressure_flags``, so equal
    LI get the same flag. Returns a bool array; a Dead_Ball is never a High_Leverage_Ball.
    """
    values = _floats(li, "li")
    groups = _ints(innings, "innings", len(values))
    flags = pressure.pressure_flags(values, _flags(live, "live", len(values)), np.ones(len(values), dtype=bool),
                                    groups, HIGH_LEVERAGE_SHARE)  # fmt: skip
    return flags.to_numpy(dtype=bool)


def extras_free_flags(pi_extras_free: Any, live: Any, innings: Any, share: float) -> np.ndarray:
    """The Extras_Free_PI Pressure_Balls of criterion 5: the top ``share`` (D3) of each Innings_Group's Live_Balls.

    One value per Delivery of innings 1 and 2: ``pi_extras_free`` the Extras_Free_PI
    (``pi_bowl_extras_free`` of the pressure table), ``live`` the Live_Ball flags, ``innings`` 1 or
    2. Every Delivery is a Bowler_Delivery. A Live_Ball is selected when the Percentile_Rank of its
    Extras_Free_PI among the Live_Balls of its Innings_Group is greater than 1 - ``share``
    (``pressure.pressure_flags``). Returns a bool array.
    """
    values = _floats(pi_extras_free, "pi_extras_free")
    groups = _ints(innings, "innings", len(values))
    flags = pressure.pressure_flags(values, _flags(live, "live", len(values)), np.ones(len(values), dtype=bool),
                                    groups, share)  # fmt: skip
    return flags.to_numpy(dtype=bool)


# ===================================================================== the Out_Of_Fold guard (14.11)

#: The columns of GuardInputs.deliveries and GuardInputs.values (module docstring, "The Out_Of_Fold guard").
GUARD_DELIVERY_COLUMNS: tuple[str, ...] = ("row", "match_id", "innings", "fold", "validation", "batter")
GUARD_VALUE_COLUMNS: tuple[str, ...] = (
    "wp_pressure", "wp_primary", "wp_m3_table", "li_pressure", "li_primary", "pi_bat", "pi_bowl",
    "pi_bowl_extras_free", "fold_primary", "fold_m3_table",
)  # fmt: skip

#: The fits behind each Delivery's WP, LI and PI (GuardInputs.sources), with what each one supplies.
FIT_SOURCES: Mapping[str, str] = MappingProxyType(
    {
        "wp_fit": "the M3 outer fit that the win_prob WP table records for the pre-ball WP",
        "wp_calibrator": "the calibrator that the win_prob WP table records for the pre-ball WP",
        "wp_outer_fit": "the M3 Primary_Model's outer fit of the fold (Post_Outcome_State WP, LI, WP after)",
        "wp_outer_calibrator": "the M3 Primary_Model's calibrator of the fold (Post_Outcome_State WP, LI, WP after)",
        "m4_fit": "the M4 Primary_Model's fold model (the LI's outcome probabilities)",
        "m4_m1_fit": "the M1 fit behind that M4 fold model's Average_Skill",
        "view_m1_fit": "the M1 fit of the own-fold view row (model features, Stress_Pillars)",
        "view_m2_fit": "the M2 fit of the own-fold view row (Par_Score, Achievable_Rate, F19 and F23)",
    }
)

#: The checks of oof_guard, in table order (module docstring).
CHECKS: tuple[str, ...] = (
    "value", "agreement", "fold", "fit_registered", "fit_excludes_match", "fit_serves_match", "sandbox",
)  # fmt: skip


def sandbox_models(innings: int, candidates: Mapping[str, str]) -> tuple[tuple[str, str | None], ...]:
    """The (model, family) pairs of the FitRegistry whose sandbox fits feed the WP, LI and PI of an innings.

    M1 (``m1``, no family), the first-innings M2 (``m2_first``, its family), the M3 Primary_Model
    of the innings and its calibrator (``m3_inn<i>`` and ``m3_inn<i>_cal`` with the Primary
    family) and the M4 Primary_Model (``m4`` with its family); ``candidates`` maps M3-1, M3-2 and
    M4 to the Primary_Models' families (``PrimaryPredictions.candidates``).
    """
    number = int(innings)
    m3_family = str(candidates[model_choice.M3_MODELS[number]])
    return (
        (skill.MODEL, None),
        (remaining_runs.FIRST_INNINGS_MODEL, remaining_runs.FAMILY),
        (win_prob.M3_MODELS[number], m3_family),
        (win_prob.CALIBRATOR_MODELS[number], m3_family),
        (outcome.MODEL, str(candidates[model_choice.M4_MODEL])),
    )


@dataclass(frozen=True, eq=False)
class GuardInputs:
    """Everything oof_guard checks, one row per Delivery of innings 1 and 2 in ``state`` order.

    ``candidates``: the Primary_Models (M3-1, M3-2, M4 -> family); ``deliveries``:
    GUARD_DELIVERY_COLUMNS (``row``, ``match_id``, ``innings``, ``fold`` from the Fold_Table,
    ``validation``, ``batter``); ``values``: GUARD_VALUE_COLUMNS (the pressure table's WP, LI and
    PIs, the Primary_Models' Out_Of_Fold WP, the WP of the M3 table that names the fits, the LI
    recomputed from the Primary_Models' predictions, the fold stored with the Primary_Models'
    predictions and in the M3 table; NaN or -1 where a source has no row); ``sources``: one column
    per FIT_SOURCES name, the fit id (str) or None where none is recorded. The three frames hold
    the same rows in the same order.
    """

    candidates: Mapping[str, str]
    deliveries: pd.DataFrame
    values: pd.DataFrame
    sources: pd.DataFrame

    def __post_init__(self) -> None:
        _require(self.deliveries, GUARD_DELIVERY_COLUMNS, "deliveries")
        _require(self.values, GUARD_VALUE_COLUMNS, "values")
        _require(self.sources, tuple(FIT_SOURCES), "sources")
        n_rows = len(self.deliveries)
        if len(self.values) != n_rows or len(self.sources) != n_rows:
            raise ValueError("deliveries, values and sources must hold the same rows, one per Delivery")
        wanted = (*model_choice.M3_MODELS.values(), model_choice.M4_MODEL)
        if not isinstance(self.candidates, Mapping) or any(name not in self.candidates for name in wanted):
            raise ValueError(f"candidates must name the Primary_Model of each of {', '.join(wanted)}")
        object.__setattr__(self, "candidates", MappingProxyType(dict(self.candidates)))


@dataclass(frozen=True, eq=False)
class GuardResult:
    """A passed Out_Of_Fold guard: the counts it checked and its table (GUARD_COLUMNS, every ``n_affected`` 0)."""

    n_validation: int
    n_matches: int
    n_fits: int
    table: pd.DataFrame


class ValidationLeakageError(LeakageError):
    """The Out_Of_Fold guard failed (exit code 4; Requirement 14, criterion 11).

    ``affected``: the number of Validation_Deliveries that fail at least one check;
    ``n_validation``: the Validation_Deliveries checked; ``rows``: the ``row`` ids of the affected
    ones (ascending, read-only); ``table``: GUARD_COLUMNS with the count of every check and source.
    """

    def __init__(self, message: str, *, affected: int, n_validation: int, rows: np.ndarray, table: pd.DataFrame) -> None:
        super().__init__(message)
        self.affected = int(affected)
        self.n_validation = int(n_validation)
        rows = np.array(rows, dtype=np.int64)
        rows.flags.writeable = False
        self.rows = rows
        self.table = table


def _view_fit_ids(view_by_k: Mapping[int, pd.DataFrame], fold: np.ndarray, row_ids: np.ndarray, column: str) -> np.ndarray:
    """The ``column`` (``m1_fit`` or ``m2_fit``) of each row's own-fold view row; None where the view or row is absent."""
    out = np.full(len(row_ids), None, dtype=object)
    for k in np.unique(fold).tolist():
        view = view_by_k.get(int(k)) if isinstance(view_by_k, Mapping) else None
        if view is None or column not in view.columns:
            continue
        at = _positions(view["row"], row_ids, f"view V_{k}")
        mine = (fold == k) & (at >= 0)
        out[mine] = _texts(view[column], f"view V_{k} column {column!r}")[at[mine]]
    return out


def guard_inputs(
    *,
    balls: pd.DataFrame,
    fold_table: pd.DataFrame,
    primary: PrimaryPredictions,
    pressure_table: pd.DataFrame,
    view_by_k: Mapping[int, pd.DataFrame],
    m3_tables: Mapping[int, pd.DataFrame],
    m4_fold_models: pd.DataFrame,
    n_folds: int,
) -> GuardInputs:
    """The GuardInputs of every Delivery of innings 1 and 2, from the tables of one Run (module docstring).

    ``balls``: the ``reconstruct`` table; ``fold_table``: the Fold_Table; ``primary``: the
    Primary_Models' predictions (``model_choice.load_primary`` with LI_VARIANT), whose rows (the
    ``state`` rows) fix the order; ``pressure_table``: the ``pressure`` table; ``view_by_k``: view
    V_k of the ``remaining_runs`` checkpoint for each fold k; ``m3_tables``: innings -> the
    ``win_prob`` WP table of that innings' M3 Primary_Model (``row``, ``fold``, ``wp_pre``,
    ``fit_id``, ``calibrator_fit_id``); ``m4_fold_models``: the M4 Primary_Model's fold-model
    table (``fold``, ``fit_id``, ``m1_fit``); ``n_folds``: D15. A row that a table lacks gets NaN,
    -1 or None, which the guard reports. Raises ValueError for tables that do not cover the
    ``state`` rows (``balls``) or hold a row twice, and the errors of ``folds.fold_of`` and
    ``pressure.primary_leverage``.
    """
    row_ids = np.asarray(primary.row, dtype=np.int64)
    n_rows = len(row_ids)
    rows = _aligned(balls, row_ids, _GUARD_BALL_COLUMNS, "balls")
    match = _ints(rows["match_id"], "balls column 'match_id'")
    innings = _ints(rows["innings"], "balls column 'innings'")
    fold = np.asarray(folds.fold_of(rows["match_id"], fold_table), dtype=np.int64)
    deliveries = pd.DataFrame(
        {
            "row": row_ids,
            "match_id": match,
            "innings": innings,
            "fold": fold,
            "validation": validation_mask(rows),
            "batter": pressure.role_masks(rows["cls"])[BATTER],
        },
        columns=list(GUARD_DELIVERY_COLUMNS),
    )

    _require(pressure_table, ("row", "wp_pre", "li", "pi_bat", "pi_bowl", "pi_bowl_extras_free"), "pressure_table")
    at = _positions(pressure_table["row"], row_ids, "pressure_table")
    found = at >= 0

    def taken(column: str) -> np.ndarray:
        out = np.full(n_rows, np.nan)
        out[found] = _floats(pressure_table[column], f"pressure_table column {column!r}")[at[found]]
        return out

    values: dict[str, np.ndarray] = {
        "wp_pressure": taken("wp_pre"),
        "wp_primary": np.array(primary.wp_pre, dtype=np.float64),
        "wp_m3_table": np.full(n_rows, np.nan),
        "li_pressure": taken("li"),
        "li_primary": np.array(pressure.primary_leverage(primary).li, dtype=np.float64),
        "pi_bat": taken("pi_bat"),
        "pi_bowl": taken("pi_bowl"),
        "pi_bowl_extras_free": taken("pi_bowl_extras_free"),
        "fold_primary": np.asarray(primary.fold, dtype=np.int64),
        "fold_m3_table": np.full(n_rows, -1, dtype=np.int64),
    }
    sources = {name: np.full(n_rows, None, dtype=object) for name in FIT_SOURCES}
    for number, table in m3_tables.items():
        what = f"the innings-{number} M3 WP table"
        _require(table, _M3_TABLE_COLUMNS, what)
        place = _positions(table["row"], row_ids, what)
        mine = (innings == int(number)) & (place >= 0)
        values["wp_m3_table"][mine] = _floats(table["wp_pre"], f"{what} column 'wp_pre'")[place[mine]]
        values["fold_m3_table"][mine] = _ints(table["fold"], f"{what} column 'fold'")[place[mine]]
        sources["wp_fit"][mine] = _texts(table["fit_id"], f"{what} column 'fit_id'")[place[mine]]
        sources["wp_calibrator"][mine] = _texts(table["calibrator_fit_id"], f"{what} column 'calibrator_fit_id'")[
            place[mine]
        ]

    outer = {spec.sandbox_k: spec for spec in folds.sandbox_plan(n_folds) if spec.kind == folds.OUTER}
    candidates = dict(primary.candidates)
    for number in STATE_INNINGS:
        family = str(candidates[model_choice.M3_MODELS[number]])
        for k, spec in outer.items():
            mine = (innings == number) & (fold == k)
            sources["wp_outer_fit"][mine] = leakage.fit_id_for(win_prob.M3_MODELS[number], spec, family=family)
            sources["wp_outer_calibrator"][mine] = leakage.fit_id_for(
                win_prob.CALIBRATOR_MODELS[number], spec, family=family
            )
    _require(m4_fold_models, _M4_FOLD_COLUMNS, "m4_fold_models")
    m4_fold = _ints(m4_fold_models["fold"], "m4_fold_models column 'fold'")
    m4_fit = _texts(m4_fold_models["fit_id"], "m4_fold_models column 'fit_id'")
    m4_m1 = _texts(m4_fold_models["m1_fit"], "m4_fold_models column 'm1_fit'")
    for k in np.unique(m4_fold).tolist():
        at_k = np.flatnonzero(m4_fold == k)
        if len(at_k) != 1:
            continue  # a fold with two M4 fold models has no defined source: left None
        mine = fold == k
        sources["m4_fit"][mine] = m4_fit[at_k[0]]
        sources["m4_m1_fit"][mine] = m4_m1[at_k[0]]
    sources["view_m1_fit"] = _view_fit_ids(view_by_k, fold, row_ids, views.M1_FIT)
    sources["view_m2_fit"] = _view_fit_ids(view_by_k, fold, row_ids, views.M2_FIT)
    return GuardInputs(
        candidates=candidates,
        deliveries=deliveries,
        values=pd.DataFrame(values, columns=list(GUARD_VALUE_COLUMNS)),
        sources=pd.DataFrame(sources, columns=list(FIT_SOURCES)),
    )


def _m3_stage(family: str) -> str:
    """The checkpoint that holds an M3 Primary_Model's WP table with fit ids, and its fold models.

    The ``win_prob`` families live in the ``win_prob`` checkpoint. The M3 Alternative_Models
    (XGBoost, random forest; task 26.1) live in the ``model_choice`` checkpoint, whose WP table of
    each candidate (``win_prob.wp_table``) also holds ``fit_id`` and ``calibrator_fit_id`` and
    whose fold models are saved under the same fit ids.
    """
    if family in win_prob.FAMILIES:
        return win_prob.STAGE_NAME
    if family in win_prob.ALTERNATIVE_FAMILIES:
        return model_choice.STAGE_NAME
    raise StageError(
        f"the M3 Primary_Model {family!r} has no WP table with fit ids in the {win_prob.STAGE_NAME} or "
        f"{model_choice.STAGE_NAME} checkpoint; the Pressure_Validator traces the win_prob families "
        f"{', '.join(win_prob.FAMILIES)} and the Alternative_Models {', '.join(win_prob.ALTERNATIVE_FAMILIES)}"
    )


def _m4_stage(store: CheckpointStore, family: str) -> str:
    """The checkpoint that holds the M4 Primary_Model's tables (SOURCES_TABLE of the model_choice checkpoint)."""
    sources = store.load_table(model_choice.STAGE_NAME, model_choice.SOURCES_TABLE)
    _require(sources, model_choice.SOURCES_COLUMNS, f"checkpoint table {model_choice.SOURCES_TABLE!r}")
    model_of = sources["model"].astype(str).to_numpy(dtype=object)
    candidate_of = sources["candidate"].astype(str).to_numpy(dtype=object)
    at = np.flatnonzero((model_of == model_choice.M4_MODEL) & (candidate_of == family))
    if len(at) != 1:
        raise StageError(f"the {model_choice.SOURCES_TABLE} table names no saved predictions of M4 candidate {family!r}")
    return str(sources["stage"].iloc[at[0]])


def _view_tables(store: CheckpointStore, n_folds: int) -> dict[int, pd.DataFrame]:
    ks = sorted(spec.sandbox_k for spec in folds.sandbox_plan(n_folds) if spec.kind == folds.OUTER)
    return {k: store.load_table(remaining_runs.STAGE_NAME, views.view_table(k)) for k in ks}


def _model_tables(
    store: CheckpointStore, candidates: Mapping[str, str]
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    """The M3 Primary_Models' WP tables (innings -> table) and the M4 Primary_Model's fold-model table."""
    m3_tables = {}
    for number in STATE_INNINGS:
        family = str(candidates[model_choice.M3_MODELS[number]])
        m3_tables[number] = store.load_table(_m3_stage(family), win_prob.wp_table(number, family))
    m4_family = str(candidates[model_choice.M4_MODEL])
    m4_fold_models = store.load_table(_m4_stage(store, m4_family), outcome.fold_models_table(m4_family))
    return m3_tables, m4_fold_models


def load_guard_inputs(store: CheckpointStore, *, n_folds: int) -> GuardInputs:
    """The GuardInputs of a Run, read from the checkpoints of ``store`` (``n_folds`` is D15).

    Reads ``balls`` (``reconstruct``), the Fold_Table (``folds``), the Primary_Models' predictions
    (``model_choice.load_primary``), the pressure table (``pressure``), the views
    (``remaining_runs``), the M3 Primary_Models' WP tables (``win_prob``) and the M4 Primary_Model's
    fold-model table (``outcome``), and returns ``guard_inputs`` of them. ``run_stage`` reads the
    same tables once and calls ``guard_inputs`` itself.
    """
    primary = model_choice.load_primary(store, variants=(LI_VARIANT,))
    m3_tables, m4_fold_models = _model_tables(store, primary.candidates)
    return guard_inputs(
        balls=store.load_table(reconstruct.STAGE_NAME, reconstruct.BALLS_TABLE),
        fold_table=store.load_table(folds.STAGE_NAME, folds.FOLD_TABLE),
        primary=primary,
        pressure_table=store.load_table(pressure.STAGE_NAME, pressure.PRESSURE_TABLE),
        view_by_k=_view_tables(store, n_folds),
        m3_tables=m3_tables,
        m4_fold_models=m4_fold_models,
        n_folds=n_folds,
    )


def _lookup(registry: Any, fit_id: Any, cache: dict[str, Any]) -> Any:
    """The FitRecord of ``fit_id`` with its Training_Rows matches, or None when the registry does not hold it."""
    if fit_id in cache:
        return cache[fit_id]
    try:
        record = registry.get(fit_id)
    except KeyError:  # leakage.UnknownFitError
        cache[fit_id] = None
        return None
    cache[fit_id] = (record, record.training_match_ids)
    return cache[fit_id]


def oof_guard(inputs: GuardInputs, registry: FitRegistry) -> GuardResult:
    """Check that every Validation_Delivery's WP, LI and PI are Out_Of_Fold (criterion 11; module docstring).

    ``inputs``: guard_inputs; ``registry``: the Run's FitRegistry (``ctx.registry``). Returns a
    GuardResult when every check passes. Raises ValidationLeakageError (exit code 4), giving the
    number of affected Validation_Deliveries and the count of each failed check, when one fails;
    StageError (exit code 5) when there is no Validation_Delivery; TypeError for arguments of the
    wrong kind.
    """
    if not isinstance(inputs, GuardInputs):
        raise TypeError(f"inputs must be GuardInputs; got {type(inputs).__name__}")
    if not callable(getattr(registry, "get", None)) or not callable(getattr(registry, "records", None)):
        raise TypeError(f"registry must be the Run's FitRegistry (superr.leakage); got {type(registry).__name__}")
    table = inputs.deliveries
    at = np.flatnonzero(_flags(table["validation"], "deliveries column 'validation'"))
    n_validation = len(at)
    if not n_validation:
        raise StageError("there is no Validation_Delivery, so the Pressure_Validator has nothing to validate")
    row = _ints(table["row"], "deliveries column 'row'")[at]
    match = _ints(table["match_id"], "deliveries column 'match_id'")[at]
    innings = _ints(table["innings"], "deliveries column 'innings'")[at]
    fold = _ints(table["fold"], "deliveries column 'fold'")[at]
    batter = _flags(table["batter"], "deliveries column 'batter'")[at]
    values = inputs.values.iloc[at].reset_index(drop=True)
    affected = np.zeros(n_validation, dtype=bool)
    records: list[dict[str, Any]] = []
    failures: list[str] = []

    def note(check: str, source: str, description: str, bad: np.ndarray, *, n_fits: Any = pd.NA,
             checked: np.ndarray | None = None) -> None:  # fmt: skip
        nonlocal affected
        rows_checked = int(n_validation if checked is None else checked.sum())
        count = int(bad.sum())
        records.append(
            {"check": check, "source": source, "description": description, "n_fits": n_fits, "n_rows": rows_checked,
             "n_affected": count}
        )  # fmt: skip
        if count:
            affected |= bad
            failures.append(
                f"{check} / {source}: {_plural(count, 'Validation_Delivery', 'Validation_Deliveries')} affected "
                f"({description}; for example row {_shown_rows(row[bad])})"
            )

    def column(name: str) -> np.ndarray:
        return _floats(values[name], f"values column {name!r}")

    with np.errstate(invalid="ignore"):
        wp = column("wp_primary")
        note("value", "wp", "no Out_Of_Fold pre-ball WP within [0, 1]", ~(np.isfinite(wp) & (wp >= 0.0) & (wp <= 1.0)))
        li = column("li_pressure")
        note("value", "li", "no finite LI of at least 0", ~(np.isfinite(li) & (li >= 0.0)))
        for name, mask in (("pi_bat", batter), ("pi_bowl", None), ("pi_bowl_extras_free", None)):
            pi = column(name)
            bad = ~(np.isfinite(pi) & (pi >= 0.0) & (pi <= _PI_MAX))
            if mask is not None:
                bad &= mask
            note("value", name, f"no {name} within [0, 100] for the Delivery's Role", bad, checked=mask)
        note(
            "agreement", "wp_pressure", "the pressure table's WP differs from the Primary_Models' Out_Of_Fold WP",
            ~(column("wp_pressure") == wp),
        )  # fmt: skip
        note(
            "agreement", "wp_m3_table", "the Primary_Models' WP differs from the WP of the M3 table that names its fits",
            ~(column("wp_m3_table") == wp),
        )  # fmt: skip
        note(
            "agreement", "li_primary",
            "the pressure table's LI differs from the LI of the Primary_Models' Out_Of_Fold predictions",
            ~(li == column("li_primary")),
        )  # fmt: skip
    for name, label in (("fold_primary", "the Primary_Models' predictions"), ("fold_m3_table", "the M3 WP table")):
        stored = _ints(values[name], f"values column {name!r}")
        note("fold", name, f"the fold stored with {label} differs from the match's fold in the Fold_Table", stored != fold)

    cache: dict[str, Any] = {}
    fit_ids: set[str] = set()
    for source, description in FIT_SOURCES.items():
        ids = _texts(inputs.sources[source].iloc[at], f"sources column {source!r}", n_validation)
        codes, uniques = pd.factorize(pd.Series(ids, dtype=object), use_na_sentinel=True)
        codes = np.asarray(codes, dtype=np.int64)
        unknown = codes < 0
        saw = np.zeros(n_validation, dtype=bool)
        idle = np.zeros(n_validation, dtype=bool)
        named = [str(value) for value in uniques.tolist()]
        for code, fit_id in enumerate(named):
            mine = codes == code
            found = _lookup(registry, fit_id, cache)
            if found is None:
                unknown |= mine
                continue
            record, training = found
            fit_ids.add(fit_id)
            excluded = np.array(sorted(record.excluded_folds), dtype=np.int64)
            saw[mine] = ~np.isin(fold[mine], excluded) | np.isin(match[mine], training)
            idle[mine] = ~np.isin(match[mine], record.predicted_match_ids)
        note(
            "fit_registered", source, f"{description}: no fit recorded, or a fit that the FitRegistry does not hold",
            unknown, n_fits=len(named),
        )  # fmt: skip
        note(
            "fit_excludes_match", source,
            f"{description}: its Training_Rows hold the Delivery's match, or it does not exclude the Delivery's fold",
            saw, n_fits=len(named),
        )  # fmt: skip
        note(
            "fit_serves_match", source, f"{description}: the Delivery's match is not among its predicted matches",
            idle, n_fits=len(named),
        )  # fmt: skip

    every = registry.records()
    for number in STATE_INNINGS:
        wanted = set(sandbox_models(number, inputs.candidates))
        mine = innings == number
        bad = np.zeros(n_validation, dtype=bool)
        n_fits = 0
        for k in np.unique(fold[mine]).tolist():
            fits = [r for r in every if r.sandbox_k == k and (r.model, r.family) in wanted]
            n_fits += len(fits)
            fit_ids.update(r.fit_id for r in fits)
            if not fits:
                continue
            union = np.unique(np.concatenate([r.training_match_ids for r in fits]))
            rows_k = mine & (fold == k)
            bad[rows_k] = np.isin(match[rows_k], union)
        note(
            "sandbox", f"innings_{number}",
            "a fit of the Delivery's fold sandbox (M1, first-innings M2, the innings' M3 Primary_Model and calibrator, "
            "M4 Primary_Model) trained on the Delivery's match",
            bad, n_fits=n_fits, checked=mine,
        )  # fmt: skip

    guard_table = pd.DataFrame(records, columns=list(GUARD_COLUMNS)).astype({"n_fits": "Int64"})
    count = int(affected.sum())
    if count:
        bad_rows = np.sort(row[affected])
        message = (
            f"Out_Of_Fold guard (Requirement 14, criterion 11): {count:,} of the {n_validation:,} Validation_Deliveries "
            "fail, each lacking an Out_Of_Fold_Prediction of WP, LI or PI or taking one from a model whose training "
            "data include its match. The validation run is marked failed and no validation table or chart is "
            "written:\n" + "\n".join(f"  - {line}" for line in failures)
        )
        raise ValidationLeakageError(
            message, affected=count, n_validation=n_validation, rows=bad_rows, table=guard_table
        )
    return GuardResult(
        n_validation=n_validation, n_matches=len(np.unique(match)), n_fits=len(fit_ids), table=guard_table
    )


# ============================================================== WP after a Delivery, Actual_WP_Swing

def actual_wp_swing(
    frame: pd.DataFrame,
    wp_post: Any,
    *,
    innings1_models: Any,
    m2_models: Mapping[str, Any],
    par_means: pd.DataFrame,
    par_bias: pd.DataFrame,
) -> pd.DataFrame:
    """WP before and after each Validation_Delivery and its Actual_WP_Swing (module docstring, "WP after a Delivery").

    ``frame``: one row per Validation_Delivery, in any order, with SWING_INPUT_COLUMNS: the
    ``balls`` columns (``runs`` the bat runs, ``team_runs`` under the selected Extras_Encoding,
    ``result_type`` of the Result_Parser), the pre-ball ``state`` columns, the ``m3_labels``
    ``label`` and ``labelled`` (the batting side won), the pre-ball Out_Of_Fold WP ``wp_pre`` of the
    M3 Primary_Model of the innings and the row's own-fold view values (``tsg``, ``scoring_env``,
    ``venue_effect``, ``par_score``, ``par_score_next``, ``m2_fit``). Every Delivery of an innings
    must be given, so that the next Delivery is found. ``wp_post``: the (rows, 9) Out_Of_Fold WP of
    each class's Post_Outcome_State (``PrimaryPredictions.wp_post``) in ``frame`` order.
    ``innings1_models``: the innings-1 M3 Primary_Model's calibrated outer models
    (``win_prob.load_fold_models``); ``m2_models``, ``par_means`` and ``par_bias``: as for
    ``win_prob.innings_end_par`` (the outer first-innings M2 fits and the ``remaining_runs`` par
    tables).

    Returns SWING_COLUMNS in ``frame`` order: ``row``, ``wp_before``, ``wp_after``, ``swing`` =
    |wp_after - wp_before| and ``wp_after_rule`` (a name of WP_AFTER_RULES). Raises ValueError for
    a missing column, a row given twice, a WP that is missing or outside [0, 1], and the errors of
    the ``win_prob`` functions.
    """
    _require(frame, SWING_INPUT_COLUMNS, "frame")
    n_rows = len(frame)
    row = _ints(frame["row"], "frame column 'row'")
    if len(np.unique(row)) != n_rows:
        raise ValueError("frame must hold each row once")
    match = _ints(frame["match_id"], "frame column 'match_id'")
    innings = _ints(frame["innings"], "frame column 'innings'")
    if not np.isin(innings, STATE_INNINGS).all():
        raise ValueError("frame must hold Deliveries of innings 1 and 2 only")
    fold = _ints(frame["fold"], "frame column 'fold'")
    legal = _ints(frame["cls"], "frame column 'cls'") == _LEGAL
    team_runs = _floats(frame["team_runs"], "frame column 'team_runs'")
    counted = _flags(frame["counted_wicket"], "frame column 'counted_wicket'").astype(np.float64)
    lb_bowled = _floats(frame["lb_bowled"], "frame column 'lb_bowled'")
    wickets = _floats(frame["wickets"], "frame column 'wickets'")
    score = _floats(frame["score"], "frame column 'score'")
    sched = _floats(frame["sched_balls"], "frame column 'sched_balls'")
    wp_before = _floats(frame["wp_pre"], "frame column 'wp_pre'")
    if not (np.isfinite(wp_before) & (wp_before >= 0.0) & (wp_before <= 1.0)).all():
        raise ValueError("frame column 'wp_pre' must hold a WP within [0, 1] on every row")
    post = np.asarray(wp_post, dtype=np.float64)
    if post.shape != (n_rows, outcome.N_CLASSES):
        raise ValueError(f"wp_post must have shape ({n_rows:,}, {outcome.N_CLASSES})")

    wp_after = np.full(n_rows, np.nan)
    rule = np.full(n_rows, None, dtype=object)
    order = np.lexsort((row, innings, match))
    same = (match[order][1:] == match[order][:-1]) & (innings[order][1:] == innings[order][:-1])
    following = np.full(n_rows, -1, dtype=np.int64)
    following[order[:-1][same]] = order[1:][same]
    has_next = following >= 0
    wp_after[has_next] = wp_before[following[has_next]]
    rule[has_next] = AFTER_NEXT

    # The last Delivery of innings 1: the end-of-innings state, or the state after an innings that stopped.
    one = np.flatnonzero(~has_next & (innings == STATE_INNINGS[0]))
    if len(one):
        lb_after = lb_bowled[one] + legal[one]
        wickets_after = wickets[one] + counted[one]
        total = score[one] + team_runs[one]
        if not (np.isfinite(total) & (total >= 0.0)).all():
            raise ValueError("the score after the last innings-1 Delivery is missing or negative for some match")
        ends = (lb_after >= sched[one]) | (wickets_after >= win_prob.ALL_OUT_WICKETS)
        view_values = {name: _floats(frame[name], f"frame column {name!r}")[one] for name in _SWING_VIEW_COLUMNS}
        par = np.where(legal[one], view_values["par_score_next"], view_values["par_score"])
        lb_state = lb_after.copy()
        wickets_state = wickets_after.copy()
        if ends.any():
            fits = _texts(frame[views.M2_FIT], f"frame column {views.M2_FIT!r}")[one][ends]
            par[ends] = win_prob.innings_end_par(
                fits, view_values["scoring_env"][ends], view_values["venue_effect"][ends], sched[one][ends].astype(np.int64),
                m2_models=m2_models, par_means=par_means, par_bias=par_bias,
            )  # fmt: skip
            lb_state[ends] = sched[one][ends]
            wickets_state[ends] = wickets[one][ends]  # the wickets before the Delivery (Requirement 10, criterion 17)
        states = pd.DataFrame(
            {
                "lb_bowled": lb_state,
                "score": total,
                "wickets": wickets_state,
                "par_score": par,
                "tsg": view_values["tsg"],
                "scoring_env": view_values["scoring_env"],
                "venue_effect": view_values["venue_effect"],
                "sched_balls": sched[one],
            },
            columns=list(win_prob.M3_FEATURES[STATE_INNINGS[0]]),
        )
        result = win_prob.win_probability(STATE_INNINGS[0], states, fold=fold[one], models={STATE_INNINGS[0]: innings1_models})
        wp_after[one] = np.asarray(result.wp, dtype=np.float64)
        rule[one] = np.where(ends, AFTER_INNINGS1_END, AFTER_INNINGS1_STOPPED)

    # The last Delivery of innings 2: the match result, else the terminal rules or the post-outcome WP.
    two = np.flatnonzero(~has_next & (innings == STATE_INNINGS[1]))
    if len(two):
        result_type = _texts(frame["result_type"], "frame column 'result_type'")[two].astype(str)
        label = _floats(frame["label"], "frame column 'label'")[two]
        labelled = _flags(frame["labelled"], "frame column 'labelled'")[two]
        decisive = np.isin(result_type, _DECISIVE_RESULTS) & labelled
        tied = np.isin(result_type, _TIED_RESULTS)
        other = ~decisive & ~tied
        if not np.isin(label[decisive], (0.0, 1.0)).all():
            raise ValueError("frame column 'label' must be 0 or 1 on the rows of decided matches")
        wp_after[two[decisive]] = label[decisive]
        rule[two[decisive]] = AFTER_RESULT
        wp_after[two[tied]] = TIED_WP
        rule[two[tied]] = AFTER_TIE
        if other.any():
            rest = two[other]
            needed = _floats(frame["runs_needed"], "frame column 'runs_needed'")[rest] - team_runs[rest]
            final = pd.DataFrame(
                {
                    "runs_needed": needed,
                    "lb_left": np.maximum(_floats(frame["lb_left"], "frame column 'lb_left'")[rest] - legal[rest], 0.0),
                    "wickets": wickets[rest] + counted[rest],
                }
            )
            codes = np.full(len(rest), win_prob.NOT_TERMINAL, dtype=np.int8)
            known = np.isfinite(needed)
            if known.any():
                codes[known] = win_prob.terminal_rules(STATE_INNINGS[1], final.iloc[np.flatnonzero(known)])
            decided = np.isin(codes, win_prob.DECIDING_RULES)
            if decided.any():
                chosen = np.flatnonzero(decided)
                wp_after[rest[chosen]] = win_prob.apply_terminal_rules(
                    STATE_INNINGS[1], final.iloc[chosen], np.full(len(chosen), np.nan)
                ).wp
                rule[rest[chosen]] = AFTER_TERMINAL
            if (~decided).any():
                open_rows = rest[~decided]
                classes = outcome.label_outcomes(
                    _ints(frame["runs"], "frame column 'runs'")[open_rows],
                    _flags(frame["bowler_wicket"], "frame column 'bowler_wicket'")[open_rows],
                    _flags(frame["run_out"], "frame column 'run_out'")[open_rows],
                )
                wp_after[open_rows] = post[open_rows, classes]
                rule[open_rows] = AFTER_POST_OUTCOME

    if not (np.isfinite(wp_after) & (wp_after >= 0.0) & (wp_after <= 1.0)).all():
        raise ValueError("a WP after a Delivery is missing or outside [0, 1]")
    return pd.DataFrame(
        {
            "row": row,
            "wp_before": wp_before,
            "wp_after": wp_after,
            "swing": np.abs(wp_after - wp_before),
            "wp_after_rule": pd.array(rule.astype(str), dtype="str"),
        },
        columns=list(SWING_COLUMNS),
    )


def wp_after_summary(innings: Any, rule: Any, wp_after: Any, swing: Any) -> pd.DataFrame:
    """WP_AFTER_COLUMNS: the Validation_Deliveries of each WP-after rule and innings, with their mean WP after and swing."""
    number = _ints(innings, "innings")
    names = _texts(rule, "rule", len(number)).astype(str)
    after = _floats(wp_after, "wp_after", len(number))
    moved = _floats(swing, "swing", len(number))
    records = []
    for name in WP_AFTER_RULES:
        for group in STATE_INNINGS:
            mine = (names == name) & (number == group)
            if not mine.any():
                continue
            records.append(
                {
                    "rule": name, "innings": int(group), "n_deliveries": int(mine.sum()),
                    "mean_wp_after": float(after[mine].mean()), "mean_swing": float(moved[mine].mean()),
                    "max_swing": float(moved[mine].max()),
                }
            )  # fmt: skip
    return pd.DataFrame(records, columns=list(WP_AFTER_COLUMNS))


# ======================================================================================== V1 (14.2)

def _role_pi(df: pd.DataFrame, role: str) -> tuple[np.ndarray, np.ndarray]:
    """The Role's Deliveries (mask) and PI of each row of ``df``."""
    if role == BATTER:
        return _flags(df["batter"], "df column 'batter'"), _floats(df["pi_bat"], "df column 'pi_bat'")
    return np.ones(len(df), dtype=bool), _floats(df["pi_bowl"], "df column 'pi_bowl'")


def v1_deciles(df: pd.DataFrame) -> pd.DataFrame:
    """V1: per Role and Innings_Group, 10 equal-count PI deciles of the Live_Balls and a Dead_Ball group (criterion 2).

    ``df``: one row per Validation_Delivery with ``row``, ``innings``, ``batter`` (Batter_Delivery),
    ``live``, ``pi_bat``, ``pi_bowl`` and ``swing`` (Actual_WP_Swing). The batter Role covers the
    Batter_Deliveries and the bowler Role every Delivery. A Role's Live_Balls of an Innings_Group
    are sorted by PI, equal PI by ``row`` (Delivery order), and position p of n (0-based) goes to
    decile floor(10 p / n) + 1, so decile 1 holds the lowest PI and the counts differ by at most 1;
    equal PI can straddle two deciles. Returns V1_COLUMNS: for each Role (batter, bowler) and
    Innings_Group (1, 2), deciles 1-10 then the ``dead`` row (``decile`` <NA>), with the Delivery
    count, the lowest and highest PI and the mean Actual_WP_Swing (NaN for an empty group).
    """
    _require(df, ("row", "innings", "batter", "live", "pi_bat", "pi_bowl", "swing"), "df")
    row = _ints(df["row"], "df column 'row'")
    innings = _ints(df["innings"], "df column 'innings'")
    live = _flags(df["live"], "df column 'live'")
    swing = _floats(df["swing"], "df column 'swing'")
    if np.isnan(swing).any():
        raise ValueError("df column 'swing' has a missing Actual_WP_Swing")
    records: list[dict[str, Any]] = []

    def add(role: str, group: int, kind: str, decile: Any, rows: np.ndarray, pi: np.ndarray) -> None:
        records.append(
            {
                "role": role, "innings": group, "group": kind, "decile": decile, "n_deliveries": len(rows),
                "pi_min": float(pi[rows].min()) if len(rows) else math.nan,
                "pi_max": float(pi[rows].max()) if len(rows) else math.nan,
                "mean_swing": float(swing[rows].mean()) if len(rows) else math.nan,
            }
        )  # fmt: skip

    for role in ROLES:
        mask, pi = _role_pi(df, role)
        for group in STATE_INNINGS:
            mine = mask & (innings == group)
            on = np.flatnonzero(mine & live)
            if np.isnan(pi[on]).any():
                raise ValueError(f"a {role}-Role Live_Ball of innings {group} has no PI")
            ordered = on[np.lexsort((row[on], pi[on]))]
            decile = (N_DECILES * np.arange(len(ordered), dtype=np.int64)) // max(len(ordered), 1) + 1
            for number in range(1, N_DECILES + 1):
                add(role, group, DECILE_GROUP, number, ordered[decile == number], pi)
            add(role, group, DEAD_GROUP, pd.NA, np.flatnonzero(mine & ~live), pi)
    return pd.DataFrame(records, columns=list(V1_COLUMNS)).astype({"decile": "Int64", "n_deliveries": "int64"})


# ===================================================================================== V2 (14.3, 14.13)

def big_moments(df: pd.DataFrame) -> np.ndarray:
    """The Big_Moment flag of each Validation_Delivery (glossary; criterion 13).

    ``df``: the Validation_Deliveries with ``match_id``, ``row`` (Delivery order) and ``swing``
    (Actual_WP_Swing). Within each match the Deliveries are sorted by swing, largest first, and
    equal swings by ``row``, earliest first; the first BIG_MOMENTS_PER_MATCH (10) are the match's
    Big_Moments, so a tie at the 10th-largest swing goes to the earlier Delivery. A match with
    fewer than 10 Validation_Deliveries has all of them. Returns a bool array in ``df`` order.
    Raises ValueError for a missing swing or a row given twice.
    """
    _require(df, ("match_id", "row", "swing"), "df")
    match = _ints(df["match_id"], "df column 'match_id'")
    row = _ints(df["row"], "df column 'row'", len(match))
    swing = _floats(df["swing"], "df column 'swing'", len(match))
    if np.isnan(swing).any():
        raise ValueError("df column 'swing' has a missing Actual_WP_Swing")
    if len(np.unique(row)) != len(row):
        raise ValueError("df must hold each row once")
    order = np.lexsort((row, -swing, match))
    sorted_match = match[order]
    starts = np.r_[True, sorted_match[1:] != sorted_match[:-1]] if len(order) else np.zeros(0, dtype=bool)
    first = np.maximum.accumulate(np.where(starts, np.arange(len(order)), 0)) if len(order) else np.zeros(0, dtype=np.int64)
    position = np.empty(len(order), dtype=np.int64)
    position[order] = np.arange(len(order)) - first
    return position < BIG_MOMENTS_PER_MATCH


def v2_precision_recall(df: pd.DataFrame, methods: Mapping[str, Any]) -> pd.DataFrame:
    """V2: the share flagged, precision and recall of each method against the Big_Moments (criterion 3).

    ``df``: the Validation_Deliveries with ``match_id`` and ``big_moment`` (big_moments);
    ``methods``: method name -> its flag on each row of ``df`` (bool, or nullable boolean whose
    <NA> counts as not flagged). Only the matches with at least MIN_MATCH_DELIVERIES (10) rows in
    ``df`` count. Returns V2_COLUMNS, one row per method in the order given: the matches and
    Deliveries pooled, the Big_Moments among them, the flagged Deliveries and flagged Big_Moments,
    ``share_flagged`` = flagged / Deliveries, ``precision`` = flagged Big_Moments / flagged and
    ``recall`` = flagged Big_Moments / Big_Moments (NaN for a zero denominator), and
    ``big_moment_share`` = Big_Moments / Deliveries, the precision of a random selection.
    """
    _require(df, ("match_id", "big_moment"), "df")
    match = _ints(df["match_id"], "df column 'match_id'")
    big = _flags(df["big_moment"], "df column 'big_moment'", len(match))
    _, codes, counts = np.unique(match, return_inverse=True, return_counts=True)
    kept = counts[codes] >= MIN_MATCH_DELIVERIES
    n_kept = int(kept.sum())
    n_big = int((big & kept).sum())
    records = []
    for name, values in methods.items():
        series = _series(values, f"methods[{name!r}]")
        if len(series) and not pd.api.types.is_bool_dtype(series.dtype):
            raise TypeError(f"methods[{name!r}] must be a boolean flag per row; got dtype {series.dtype}")
        flag = _length(series.fillna(False).to_numpy(dtype=bool), f"methods[{name!r}]", len(match)) & kept
        n_flagged = int(flag.sum())
        hits = int((flag & big).sum())
        records.append(
            {
                "method": str(name), "n_matches": int((counts >= MIN_MATCH_DELIVERIES).sum()), "n_deliveries": n_kept,
                "n_big_moments": n_big, "n_flagged": n_flagged, "n_flagged_big_moments": hits,
                "share_flagged": _share(n_flagged, n_kept), "precision": _share(hits, n_flagged),
                "recall": _share(hits, n_big), "big_moment_share": _share(n_big, n_kept),
            }
        )  # fmt: skip
    return pd.DataFrame(records, columns=list(V2_COLUMNS))


# ================================================================================= V3 (14.4-14.6, 14.12)

def _extras_table(df: pd.DataFrame, column: str, selection: str) -> pd.DataFrame:
    _require(df, ("innings", "extra", column), "df")
    innings = _ints(df["innings"], "df column 'innings'")
    extra = _flags(df["extra"], "df column 'extra'", len(innings))
    flag = _flags(df[column], f"df column {column!r}", len(innings))
    records = []
    for label in _INNINGS_LABELS:
        mine = np.ones(len(innings), dtype=bool) if label == BOTH_INNINGS else innings == int(label)
        on, off = mine & flag, mine & ~flag
        rate_on = 100.0 * _share(int((extra & on).sum()), int(on.sum()))
        rate_off = 100.0 * _share(int((extra & off).sum()), int(off.sum()))
        records.append(
            {
                "selection": selection, "innings": label, "n_pressure": int(on.sum()),
                "n_extras_pressure": int((extra & on).sum()), "rate_pressure": rate_on, "n_ordinary": int(off.sum()),
                "n_extras_ordinary": int((extra & off).sum()), "rate_ordinary": rate_off,
                "difference": rate_on - rate_off,
            }
        )  # fmt: skip
    return pd.DataFrame(records, columns=list(V3_EXTRAS_COLUMNS))


def v3_extras(df: pd.DataFrame) -> pd.DataFrame:
    """V3 extras: Wides plus No_Balls per 100 Bowler_Deliveries on the bowler Pressure_Balls and Ordinary_Balls (criterion 4).

    ``df``: the Validation_Deliveries (every one a Bowler_Delivery) with ``innings``, ``extra``
    (a Wide or No_Ball) and ``pressure_bowl`` (the Rating_Engine's bowler flags). Returns
    V3_EXTRAS_COLUMNS for innings 1, 2 and both (``selection`` MAIN_SELECTION): the Bowler_Delivery
    counts, extras and rates of the Pressure_Balls and the Ordinary_Balls (every other
    Bowler_Delivery, Dead_Balls included) and the difference Pressure - Ordinary.
    """
    return _extras_table(df, "pressure_bowl", MAIN_SELECTION)


def v3_extras_free(df: pd.DataFrame) -> pd.DataFrame:
    """V3 extras with the Extras_Free_PI selection (criterion 5): as v3_extras on ``pressure_bowl_extras_free``.

    ``df`` needs ``pressure_bowl_extras_free``: the top D3 of each Innings_Group's Live_Balls by
    Extras_Free_PI (extras_free_flags). The Rating_Engine's flags are not changed.
    """
    return _extras_table(df, "pressure_bowl_extras_free", EXTRAS_FREE_SELECTION)


@dataclass(frozen=True)
class EffectEstimate:
    """One within-player effect (within_player_effect): beta, its match-clustered SE and 95% CI, and the counts."""

    beta: float
    se: float
    ci_low: float
    ci_high: float
    t_crit: float
    df: int  # G - 1, the degrees of freedom of the t quantile
    n_obs: int  # N, the Deliveries
    n_players: int  # K - 1, the players (fixed effects)
    n_players_both: int  # the players with both d = 1 and d = 0 rows, who identify beta
    n_clusters: int  # G, the matches
    n_pressure: int  # the rows with d = 1
    mean_pressure: float  # the raw mean of y where d = 1
    mean_ordinary: float  # the raw mean of y where d = 0
    small_sample_factor: float  # G / (G - 1) x (N - 1) / (N - K)


def _labels(values: Any, what: str, n_rows: int) -> tuple[np.ndarray, int]:
    series = _series(values, what)
    _length(series.to_numpy(), what, n_rows)
    if series.isna().any():
        raise ValueError(f"{what} has {int(series.isna().sum()):,} missing label(s)")
    codes, uniques = pd.factorize(series, sort=True)
    return np.asarray(codes, dtype=np.int64), len(uniques)


def within_player_effect(y: Any, d: Any, player: Any, cluster: Any) -> EffectEstimate:
    """The within-player difference in ``y`` between d = 1 and d = 0 rows (criteria 6 and 12; module docstring).

    ``y``: one finite outcome per row; ``d``: the pressure indicator, 1 (or True) on a
    Pressure_Ball and 0 (False) on an Ordinary_Ball; ``player``: the player of each row (one fixed
    effect each); ``cluster``: the match of each row. With y~ and d~ demeaned within player,
    beta = sum(d~ y~) / sum(d~^2) and the residuals e = y~ - beta d~ (those of OLS with one dummy
    per player); Var(beta) = c x (sum d~^2)^-2 x sum_g (sum_{i in g} d~_i e_i)^2 with
    c = G / (G - 1) x (N - 1) / (N - K), G the clusters, N the rows and K the players + 1; the CI
    is beta +/- t SE, t the (1 + CI_LEVEL) / 2 quantile of Student's t with G - 1 degrees of
    freedom. Returns an EffectEstimate. Raises ValueError for inputs of different lengths, a
    missing or infinite y, a d other than 0 or 1, a missing label, fewer than 2 clusters, no more
    rows than K, or no within-player variation of d; TypeError for non-numeric y or d.
    """
    from scipy import stats

    values = _floats(y, "y")
    n_obs = len(values)
    if not np.isfinite(values).all():
        raise ValueError("y must be finite on every row")
    series = _series(d, "d")
    if len(series) and not (pd.api.types.is_bool_dtype(series.dtype) or pd.api.types.is_numeric_dtype(series.dtype)):
        raise TypeError(f"d must hold 0/1 or booleans; got dtype {series.dtype}")
    indicator = _length(series.to_numpy(dtype=np.float64, na_value=np.nan), "d", n_obs)
    if not np.isin(indicator, (0.0, 1.0)).all():
        raise ValueError("d must be 0 or 1 (False or True) on every row")
    players, n_players = _labels(player, "player", n_obs)
    clusters, n_clusters = _labels(cluster, "cluster", n_obs)
    n_params = n_players + 1
    if n_clusters < 2:
        raise ValueError(f"a match-clustered standard error needs at least 2 clusters; got {n_clusters}")
    if n_obs <= n_params:
        raise ValueError(f"the design has {n_obs:,} rows for {n_params:,} parameters (players + 1); it needs more rows")
    size = np.bincount(players, minlength=n_players).astype(np.float64)
    y_t = values - (np.bincount(players, weights=values, minlength=n_players) / size)[players]
    d_t = indicator - (np.bincount(players, weights=indicator, minlength=n_players) / size)[players]
    sxx = float(np.dot(d_t, d_t))
    if not sxx > 0.0:
        raise ValueError("d does not vary within any player, so the within-player effect is undefined")
    beta = float(np.dot(d_t, y_t)) / sxx
    residual = y_t - beta * d_t
    scores = np.bincount(clusters, weights=d_t * residual, minlength=n_clusters)
    factor = (n_clusters / (n_clusters - 1.0)) * ((n_obs - 1.0) / (n_obs - n_params))
    se = math.sqrt(factor * float(np.dot(scores, scores))) / sxx
    t_crit = float(stats.t.ppf(0.5 + CI_LEVEL / 2.0, n_clusters - 1))
    on = indicator == 1.0
    with_pressure = np.bincount(players, weights=indicator, minlength=n_players)
    both = int(((with_pressure > 0) & (with_pressure < size)).sum())
    return EffectEstimate(
        beta=beta, se=se, ci_low=beta - t_crit * se, ci_high=beta + t_crit * se, t_crit=t_crit, df=n_clusters - 1,
        n_obs=n_obs, n_players=n_players, n_players_both=both, n_clusters=n_clusters, n_pressure=int(on.sum()),
        mean_pressure=float(values[on].mean()) if on.any() else math.nan,
        mean_ordinary=float(values[~on].mean()) if (~on).any() else math.nan, small_sample_factor=factor,
    )  # fmt: skip


#: The three within-player effects of V3: measure -> (Role, unit).
_EFFECTS: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        "strike_rate": (BATTER, "bat runs per 100 Batter_Deliveries"),
        "dismissal_rate": (BATTER, "striker Counted_Wickets (not Retire Out) per 100 Batter_Deliveries"),
        "economy": (BOWLER, "Bowler_Runs_Conceded per 6 Bowler_Deliveries"),
    }
)

#: The chart title and short axis unit of each within-player effect.
_EFFECT_LABELS: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        "strike_rate": ("Batter strike rate", "bat runs per 100 Batter_Deliveries"),
        "dismissal_rate": ("Batter dismissal rate", "striker dismissals per 100 Batter_Deliveries"),
        "economy": ("Bowler economy", "runs conceded per 6 Bowler_Deliveries"),
    }
)


def v3_within_player(df: pd.DataFrame) -> pd.DataFrame:
    """V3 within-player effects of the Pressure_Balls on strike rate, dismissal rate and economy (criteria 6 and 12).

    ``df``: the Validation_Deliveries with ``match_id``, ``batter``, ``striker_id``,
    ``bowler_id``, ``bat_runs``, ``striker_out``, ``bowler_runs_conceded``, ``pressure_bat`` and
    ``pressure_bowl``. Strike rate (y = 100 x bat runs) and dismissal rate (y = 100 x striker out)
    use the Batter_Deliveries, the batter flags and one fixed effect per striker; economy (y = 6 x
    Bowler_Runs_Conceded) every Delivery, the bowler flags and one fixed effect per bowler; every
    standard error is clustered by match (within_player_effect). Returns V3_EFFECT_COLUMNS, one row
    per measure.
    """
    _require(
        df,
        ("match_id", "batter", "striker_id", "bowler_id", "bat_runs", "striker_out", "bowler_runs_conceded",
         "pressure_bat", "pressure_bowl"),
        "df",
    )  # fmt: skip
    batter = _flags(df["batter"], "df column 'batter'")
    bat = np.flatnonzero(batter)
    flags_bat = df["pressure_bat"].iloc[bat]
    if flags_bat.isna().any():
        raise ValueError("a Batter_Delivery has no batter Pressure_Ball flag")
    match = _ints(df["match_id"], "df column 'match_id'")
    inputs = {
        "strike_rate": (100.0 * _floats(df["bat_runs"], "df column 'bat_runs'")[bat], flags_bat.to_numpy(dtype=bool),
                        _ints(df["striker_id"], "df column 'striker_id'")[bat], match[bat]),
        "dismissal_rate": (100.0 * _flags(df["striker_out"], "df column 'striker_out'")[bat].astype(np.float64),
                           flags_bat.to_numpy(dtype=bool), _ints(df["striker_id"], "df column 'striker_id'")[bat],
                           match[bat]),
        "economy": (6.0 * _floats(df["bowler_runs_conceded"], "df column 'bowler_runs_conceded'"),
                    _flags(df["pressure_bowl"], "df column 'pressure_bowl'"),
                    _ints(df["bowler_id"], "df column 'bowler_id'"), match),
    }  # fmt: skip
    records = []
    for measure, (role, unit) in _EFFECTS.items():
        y, d, player, cluster = inputs[measure]
        effect = within_player_effect(y, d, player, cluster)
        records.append(
            {
                "measure": measure, "role": role, "unit": unit, "beta": effect.beta, "se": effect.se,
                "ci_low": effect.ci_low, "ci_high": effect.ci_high, "t_crit": effect.t_crit, "df": effect.df,
                "n_players": effect.n_players, "n_players_both": effect.n_players_both, "n_deliveries": effect.n_obs,
                "n_pressure": effect.n_pressure, "n_clusters": effect.n_clusters,
                "mean_pressure": effect.mean_pressure, "mean_ordinary": effect.mean_ordinary,
            }
        )  # fmt: skip
    return pd.DataFrame(records, columns=list(V3_EFFECT_COLUMNS))


# ======================================================================================= V4 (14.7, 14.8)

def v4_tables(df: pd.DataFrame, *, max_ball_share: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """V4: the Simple_Rules' Dead_Ball shares and the High_Leverage_Balls that neither rule flags (criteria 7 and 8).

    ``df``: the Validation_Deliveries with ``innings``, ``live``, ``ball_share``
    (Ball_Position_Share), ``high_leverage`` (high_leverage_balls) and the two SIMPLE_RULES flags
    (simple_rule_flags); ``max_ball_share``: the upper D8 Phase split (0.75). Returns the
    dead-share table (V4_DEAD_COLUMNS: for each Simple_Rule and innings 1, 2 and both, the
    Deliveries flagged, the Dead_Balls among them and their share, NaN when none is flagged) and
    the unflagged table (V4_UNFLAGGED_COLUMNS: for innings 1, 2 and both, the High_Leverage_Balls
    with a Ball_Position_Share of at most ``max_ball_share``, those flagged by neither rule and
    their share).
    """
    _require(df, ("innings", "live", "ball_share", "high_leverage", *SIMPLE_RULES), "df")
    cut = float(max_ball_share)
    if not 0.0 < cut <= 1.0:
        raise ValueError(f"max_ball_share must lie in (0, 1]; got {max_ball_share!r}")
    innings = _ints(df["innings"], "df column 'innings'")
    n_rows = len(innings)
    live = _flags(df["live"], "df column 'live'", n_rows)
    share = _floats(df["ball_share"], "df column 'ball_share'", n_rows)
    high = _flags(df["high_leverage"], "df column 'high_leverage'", n_rows)
    rules = {name: _flags(df[name], f"df column {name!r}", n_rows) for name in SIMPLE_RULES}
    flagged_any = np.zeros(n_rows, dtype=bool)
    for flag in rules.values():
        flagged_any |= flag
    dead_records = []
    unflagged_records = []
    for label in _INNINGS_LABELS:
        mine = np.ones(n_rows, dtype=bool) if label == BOTH_INNINGS else innings == int(label)
        for name, flag in rules.items():
            on = mine & flag
            n_dead = int((on & ~live).sum())
            dead_records.append(
                {"rule": name, "innings": label, "n_flagged": int(on.sum()), "n_dead": n_dead,
                 "dead_share": _share(n_dead, int(on.sum()))}
            )  # fmt: skip
        with np.errstate(invalid="ignore"):
            early = mine & high & (share <= cut)
        n_unflagged = int((early & ~flagged_any).sum())
        unflagged_records.append(
            {"innings": label, "max_ball_share": cut, "n_high_leverage": int(early.sum()), "n_unflagged": n_unflagged,
             "unflagged_share": _share(n_unflagged, int(early.sum()))}
        )  # fmt: skip
    dead = pd.DataFrame(dead_records, columns=list(V4_DEAD_COLUMNS))
    order = {name: index for index, name in enumerate(SIMPLE_RULES)}
    dead = dead.sort_values("rule", key=lambda column: column.map(order), kind="stable").reset_index(drop=True)
    return dead, pd.DataFrame(unflagged_records, columns=list(V4_UNFLAGGED_COLUMNS))


# =========================================================================== signal-model matrix (14.9)

#: The models of the matrix, in column order.
MODEL_NAMES: tuple[str, ...] = ("M1", "M2", "M3", "M4", "Extras_Model")

#: The models whose outputs each model reads as features (Requirement 7, criteria 7 and 8): M2 reads
#: theta_bat, RBQ, Scoring_Environment, Venue_Effect and Match_Day_Conditions from M1; M3 reads
#: Team_Strength_Gap, RBQ, Scoring_Environment and Venue_Effect from M1 and Par_Score (innings 1) or
#: Required_Rate minus Achievable_Rate (innings 2) from M2; M4 reads theta_bat, theta_bowl,
#: Scoring_Environment, Venue_Effect and Match_Day_Conditions from M1 and the F19 Pillar_Component
#: from M2; the Extras_Model reads theta_bat, Scoring_Environment, Venue_Effect and
#: Match_Day_Conditions from M1.
MODEL_INPUTS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {"M1": (), "M2": ("M1",), "M3": ("M1", "M2"), "M4": ("M1", "M2"), "Extras_Model": ("M1",)}
)

#: The models whose outputs the PI reads directly: M3 (the WP behind the Live_Ball flag and the
#: Leverage_Index), M4 (the LI's outcome probabilities), M2 (F19 and F23; Par_Score and
#: Achievable_Rate of the Post_Outcome_States) and M1 (F20(c) and F24; theta and RBQ of the
#: Post_Wicket_States). The Extras_Model feeds no PI (Requirement 13, criterion 6).
PI_INPUTS: tuple[str, ...] = ("M1", "M2", "M3", "M4")


@dataclass(frozen=True)
class Signal:
    """One validation signal of the matrix: its name, its check, what it measures and the models it reads directly."""

    name: str
    validation: str
    target: str
    inputs: tuple[str, ...]


#: The validation signals of criterion 9, in table order, each with the models it reads directly.
SIGNALS: tuple[Signal, ...] = (
    Signal("actual_wp_swing", "V1", "|WP after - WP before| of each Validation_Delivery", ("M2", "M3")),
    Signal("big_moments", "V2", "the 10 largest Actual_WP_Swing of each match", ("M2", "M3")),
    Signal("extras_rate", "V3", "Wides plus No_Balls per 100 Bowler_Deliveries (recorded extras types)", ()),
    Signal("within_player_strike_rate", "V3", "100 x bat runs per Batter_Delivery (recorded runs)", ()),
    Signal(
        "within_player_dismissal_rate", "V3",
        "striker Counted_Wickets other than Retire Out per 100 Batter_Deliveries (recorded dismissals)", (),
    ),  # fmt: skip
    Signal("within_player_economy", "V3", "6 x Bowler_Runs_Conceded per Bowler_Delivery (recorded runs)", ()),
    Signal("dead_ball_share", "V4", "share of a Simple_Rule's Deliveries whose WP lies outside the D1 band", ("M3",)),
    Signal(
        "high_leverage_unflagged_share", "V4",
        "share of the High_Leverage_Balls (top 25% LI) up to the D8 split that neither Simple_Rule flags",
        ("M3", "M4"),
    ),  # fmt: skip
)


def _closure(direct: Sequence[str]) -> set[str]:
    """``direct`` and every model whose outputs feed one of them through MODEL_INPUTS."""
    found: set[str] = set()
    pending = list(direct)
    while pending:
        name = pending.pop()
        if name not in MODEL_INPUTS:
            raise ValueError(f"unknown model {name!r}; the models are {', '.join(MODEL_NAMES)}")
        if name not in found:
            found.add(name)
            pending.extend(MODEL_INPUTS[name])
    return found


def signal_model_matrix() -> pd.DataFrame:
    """The signal-model matrix of criterion 9, from the declared dependency maps (module docstring).

    Returns MATRIX_COLUMNS, one row per signal of SIGNALS: its name, its check (V1-V4), what it
    measures, the models whose outputs feed it (``models_feeding_signal``, directly or through
    another model's features; empty for a signal of recorded outcomes only) and, for each of
    MODEL_NAMES, True when that model's outputs feed both the signal and the Pressure_Index.
    """
    pi_models = _closure(PI_INPUTS)
    records = []
    for signal in SIGNALS:
        feeding = _closure(signal.inputs)
        record: dict[str, Any] = {
            "signal": signal.name,
            "validation": signal.validation,
            "target": signal.target,
            "models_feeding_signal": ", ".join(name for name in MODEL_NAMES if name in feeding),
        }
        for name in MODEL_NAMES:
            record[name] = bool(name in feeding and name in pi_models)
        records.append(record)
    return pd.DataFrame(records, columns=list(MATRIX_COLUMNS))


# ===================================================================================== files and charts

#: The label of each V2 method in the charts.
_METHOD_LABELS: Mapping[str, str] = MappingProxyType(
    {
        METHOD_BATTER: "Batter Pressure_Balls",
        METHOD_BOWLER: "Bowler Pressure_Balls",
        RULE_DEATH_OVERS: "Simple_Rule: overs 16-20",
        RULE_CHASE_RATE: "Simple_Rule: chase RRR > 10",
    }
)
_INNINGS_COLOURS: Mapping[int, str] = MappingProxyType({1: "#1f5fa8", 2: "#d95f02"})
_GROUP_COLOURS: Mapping[str, str] = MappingProxyType({"1": "#1f5fa8", "2": "#d95f02", BOTH_INNINGS: "#5a5a5a"})
_PRESSURE_COLOUR = "#c0392b"
_ORDINARY_COLOUR = "#7f8c8d"
_GRID_COLOUR = "#e6e6e6"
_NOTE = "Held-out Validation_Deliveries; Out_Of_Fold predictions of the Primary_Models"


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


def _figure(width: float, height: float) -> Any:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(width, height), dpi=CHART_DPI, layout="constrained")
    FigureCanvasAgg(figure)
    return figure


def _save_figure(figure: Any, path: Path) -> Path:
    """Save ``figure`` as PNG at CHART_DPI without a time stamp, through ``<file>.tmp`` renamed in place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        figure.savefig(temporary, format="png", dpi=CHART_DPI, metadata={"Software": None})
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _bar_labels(ax: Any, bars: Any, values: Sequence[float], text: str) -> None:
    for bar, value in zip(bars, values):
        if math.isfinite(value):
            ax.annotate(
                text.format(value), (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
                xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7,
            )  # fmt: skip


def _write_v1_chart(path: Path, table: pd.DataFrame) -> Path:
    """V1: the mean Actual_WP_Swing of each PI decile and of the Dead_Balls, one panel per Role."""
    figure = _figure(12.0, 4.8)
    axes = figure.subplots(1, len(ROLES), sharey=True)
    for ax, role in zip(axes, ROLES):
        part = table.iloc[np.flatnonzero(table["role"].astype(str).to_numpy() == role)]
        for group in STATE_INNINGS:
            mine = part.iloc[np.flatnonzero(part["innings"].to_numpy() == group)]
            deciles = mine.iloc[np.flatnonzero(mine["group"].astype(str).to_numpy() == DECILE_GROUP)]
            dead = mine.iloc[np.flatnonzero(mine["group"].astype(str).to_numpy() == DEAD_GROUP)]
            colour = _INNINGS_COLOURS[group]
            ax.plot(
                deciles["decile"].to_numpy(dtype=np.int64), deciles["mean_swing"].to_numpy(dtype=np.float64),
                marker="o", markersize=4, linewidth=1.6, color=colour,
                label=f"Innings {group}: Live_Balls by PI decile ({int(deciles['n_deliveries'].sum()):,})",
            )  # fmt: skip
            ax.plot(
                [0], dead["mean_swing"].to_numpy(dtype=np.float64), marker="s", markersize=6, linestyle="none",
                markerfacecolor="white", color=colour, label=f"Innings {group}: Dead_Balls ({int(dead['n_deliveries'].sum()):,})",
            )  # fmt: skip
        ax.set_xticks(range(0, N_DECILES + 1))
        ax.set_xticklabels(["Dead", *(str(number) for number in range(1, N_DECILES + 1))])
        ax.set_xlim(-0.6, N_DECILES + 0.6)
        ax.set_ylim(bottom=0.0)
        ax.grid(True, color=_GRID_COLOUR, linewidth=0.6)
        ax.set_xlabel("PI decile among the Role's Live_Balls (1 = lowest PI)")
        ax.set_title(f"{role.capitalize()} Role", fontsize=10)
        ax.legend(loc="upper left", fontsize=7, frameon=False)
    axes[0].set_ylabel("Mean Actual_WP_Swing (|WP after - WP before|)")
    figure.suptitle(f"V1: mean Actual_WP_Swing by Pressure_Index decile\n({_NOTE})", fontsize=10)
    return _save_figure(figure, path)


def _write_v2_chart(path: Path, table: pd.DataFrame) -> Path:
    """V2: the share flagged, precision and recall of each method, with the Big_Moment share."""
    figure = _figure(10.0, 4.8)
    ax = figure.subplots()
    methods = table["method"].astype(str).tolist()
    x = np.arange(len(methods), dtype=np.float64)
    width = 0.26
    for offset, (column, label, colour) in zip(
        (-width, 0.0, width),
        (("share_flagged", "Share of Deliveries flagged", "#9a9a9a"), ("precision", "Precision", "#1f5fa8"),
         ("recall", "Recall", "#d95f02")),
    ):  # fmt: skip
        values = 100.0 * table[column].to_numpy(dtype=np.float64)
        bars = ax.bar(x + offset, values, width, label=label, color=colour)
        _bar_labels(ax, bars, values.tolist(), "{:.1f}")
    base = float(table["big_moment_share"].iloc[0]) if len(table) else math.nan
    if math.isfinite(base):
        ax.axhline(100.0 * base, color="#333333", linestyle=":", linewidth=1.0,
                   label=f"Big_Moment share {100.0 * base:.1f}% (precision of a random selection)")  # fmt: skip
    ax.set_xticks(x)
    ax.set_xticklabels([_METHOD_LABELS.get(name, name) for name in methods], fontsize=8)
    ax.set_ylabel("Percent")
    shown = 100.0 * table[["share_flagged", "precision", "recall"]].to_numpy(dtype=np.float64)
    top = float(np.nanmax(shown)) if np.isfinite(shown).any() else 100.0
    ax.set_ylim(0.0, min(100.0, max(10.0, 1.3 * top)))
    ax.grid(True, axis="y", color=_GRID_COLOUR, linewidth=0.6)
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    n_matches = int(table["n_matches"].iloc[0]) if len(table) else 0
    ax.set_title(
        f"V2: Big_Moments (the 10 largest Actual_WP_Swing of each match) found by each method\n"
        f"({n_matches:,} matches with at least {MIN_MATCH_DELIVERIES} Validation_Deliveries; {_NOTE})",
        fontsize=10,
    )
    return _save_figure(figure, path)


def _write_v3_extras_chart(path: Path, main: pd.DataFrame, free: pd.DataFrame) -> Path:
    """V3: Wides plus No_Balls per 100 Bowler_Deliveries on Pressure_Balls and Ordinary_Balls, both selections."""
    figure = _figure(12.0, 4.8)
    axes = figure.subplots(1, 2, sharey=True)
    titles = ("Bowler Pressure_Balls of the Rating_Engine (PI)", "Re-selected by Extras_Free_PI (top D3 per Innings_Group)")
    rates = np.concatenate([part[["rate_pressure", "rate_ordinary"]].to_numpy(dtype=np.float64).ravel() for part in (main, free)])
    top = float(np.nanmax(rates)) if np.isfinite(rates).any() else 1.0
    for ax, table, title in zip(axes, (main, free), titles):
        labels = table["innings"].astype(str).tolist()
        x = np.arange(len(labels), dtype=np.float64)
        width = 0.36
        for offset, column, label, colour in (
            (-width / 2.0, "rate_pressure", "Pressure_Balls", _PRESSURE_COLOUR),
            (width / 2.0, "rate_ordinary", "Ordinary_Balls", _ORDINARY_COLOUR),
        ):
            values = table[column].to_numpy(dtype=np.float64)
            bars = ax.bar(x + offset, values, width, label=label, color=colour)
            _bar_labels(ax, bars, values.tolist(), "{:.2f}")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [
                f"{'Both innings' if label == BOTH_INNINGS else 'Innings ' + label}\n(difference {difference:+.2f})"
                for label, difference in zip(labels, table["difference"].to_numpy(dtype=np.float64).tolist())
            ],
            fontsize=8,
        )  # fmt: skip
        ax.set_ylim(0.0, 1.3 * top)
        ax.grid(True, axis="y", color=_GRID_COLOUR, linewidth=0.6)
        ax.set_title(title, fontsize=9)
        ax.legend(loc="upper center", ncol=2, fontsize=7, frameon=False)
    axes[0].set_ylabel("Wides + No_Balls per 100 Bowler_Deliveries")
    figure.suptitle(f"V3: extras on bowler Pressure_Balls and Ordinary_Balls\n({_NOTE})", fontsize=10)
    return _save_figure(figure, path)


def _write_v3_effects_chart(path: Path, table: pd.DataFrame) -> Path:
    """V3: each within-player effect with its 95% confidence interval, one panel per measure."""
    figure = _figure(12.0, 4.4)
    axes = np.atleast_1d(figure.subplots(1, max(len(table), 1)))
    for ax, record in zip(axes, table.to_dict("records")):
        measure = str(record["measure"])
        title, unit = _EFFECT_LABELS.get(measure, (measure, str(record["unit"])))
        beta, low, high = float(record["beta"]), float(record["ci_low"]), float(record["ci_high"])
        colour = _PRESSURE_COLOUR if (low > 0.0 or high < 0.0) else "#5a5a5a"
        ax.errorbar([0.0], [beta], yerr=[[beta - low], [high - beta]], fmt="o", color=colour, capsize=6, markersize=7)
        ax.axhline(0.0, color="#333333", linewidth=0.8, linestyle=":")
        ax.set_xlim(-1.0, 1.0)
        ax.set_xticks([])
        ax.grid(True, axis="y", color=_GRID_COLOUR, linewidth=0.6)
        ax.set_ylabel(f"Pressure_Balls minus Ordinary_Balls\n({unit})", fontsize=8)
        ax.set_title(
            f"{title}: {beta:+.3f} [{low:+.3f}, {high:+.3f}]\n{int(record['n_players']):,} players, "
            f"{int(record['n_deliveries']):,} Deliveries, {int(record['n_clusters']):,} matches",
            fontsize=8,
        )
    figure.suptitle(
        "V3: within-player difference on Pressure_Balls (player fixed effects, match-clustered 95% CI)\n" f"({_NOTE})",
        fontsize=10,
    )
    return _save_figure(figure, path)


def _write_v4_chart(path: Path, dead: pd.DataFrame, unflagged: pd.DataFrame) -> Path:
    """V4: the Dead_Ball share of each Simple_Rule and the High_Leverage_Balls that neither rule flags."""
    figure = _figure(12.0, 4.8)
    left, right = figure.subplots(1, 2)
    width = 0.26
    rules = list(SIMPLE_RULES)
    x = np.arange(len(rules), dtype=np.float64)
    for offset, label in zip((-width, 0.0, width), _INNINGS_LABELS):
        part = dead.iloc[np.flatnonzero(dead["innings"].astype(str).to_numpy() == label)]
        shares = [
            100.0 * float(part["dead_share"].iloc[np.flatnonzero(part["rule"].astype(str).to_numpy() == rule)[0]])
            if (part["rule"].astype(str).to_numpy() == rule).any() else math.nan
            for rule in rules
        ]  # fmt: skip
        values = np.array(shares, dtype=np.float64)
        bars = left.bar(x + offset, np.nan_to_num(values), width, color=_GROUP_COLOURS[label],
                        label="Both innings" if label == BOTH_INNINGS else f"Innings {label}")  # fmt: skip
        _bar_labels(left, bars, values.tolist(), "{:.1f}")
    left.set_xticks(x)
    left.set_xticklabels(
        [_METHOD_LABELS[rule] + ("\n(second innings only)" if rule == RULE_CHASE_RATE else "") for rule in rules],
        fontsize=8,
    )  # fmt: skip
    left.set_ylabel("Dead_Balls among the flagged Deliveries (%)")
    left.set_ylim(0.0, 110.0)
    left.set_yticks(np.arange(0.0, 101.0, 20.0))
    left.grid(True, axis="y", color=_GRID_COLOUR, linewidth=0.6)
    left.legend(loc="upper left", fontsize=7, frameon=False)
    left.set_title("Simple_Rule Deliveries whose match is already decided (Dead_Balls)", fontsize=9)
    labels = unflagged["innings"].astype(str).tolist()
    values = 100.0 * unflagged["unflagged_share"].to_numpy(dtype=np.float64)
    bars = right.bar(np.arange(len(labels)), np.nan_to_num(values), 0.5,
                     color=[_GROUP_COLOURS.get(label, "#5a5a5a") for label in labels])  # fmt: skip
    _bar_labels(right, bars, values.tolist(), "{:.1f}")
    right.set_xticks(np.arange(len(labels)))
    right.set_xticklabels(
        [
            f"{'Both innings' if label == BOTH_INNINGS else 'Innings ' + label}\n"
            f"({int(n):,} High_Leverage_Balls)"
            for label, n in zip(labels, unflagged["n_high_leverage"].tolist())
        ],
        fontsize=8,
    )  # fmt: skip
    right.set_ylabel("Flagged by neither Simple_Rule (%)")
    right.set_ylim(0.0, 110.0)
    right.set_yticks(np.arange(0.0, 101.0, 20.0))
    right.grid(True, axis="y", color=_GRID_COLOUR, linewidth=0.6)
    cut = float(unflagged["max_ball_share"].iloc[0]) if len(unflagged) else math.nan
    right.set_title(f"High_Leverage_Balls (top 25% LI) with Ball_Position_Share <= {cut:.2f}", fontsize=9)
    figure.suptitle(f"V4: what the Simple_Rules miss and what they flag\n({_NOTE})", fontsize=10)
    return _save_figure(figure, path)


# ============================================================================ the ``validation`` stage

def _shown_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return str(path)


def _number(value: Any) -> str:
    return "n/a" if value is None or not math.isfinite(float(value)) else f"{float(value):.4f}"


def run_stage(ctx: StageContext) -> None:
    """The ``validation`` Pipeline_Stage (module docstring, "The ``validation`` Pipeline_Stage").

    Raises ValidationLeakageError (exit code 4) when the Out_Of_Fold guard fails, before anything
    is computed, saved or written (Requirement 14, criterion 11); StageError (exit code 5) for a
    checkpoint file that is missing or changed and for a Primary_Model the guard cannot trace; and
    ValueError or TypeError for input tables that do not fit together or a Config value outside its
    range.
    """
    started = time.perf_counter()
    settings, store, log = ctx.settings, ctx.store, ctx.log
    n_folds = settings.get(_N_FOLDS)
    share = settings.get(_SHARE)
    splits = settings.get(_PHASE_SPLITS)
    if not isinstance(splits, Sequence) or isinstance(splits, (str, bytes)) or len(splits) != 2:
        raise ValueError(f"{_PHASE_SPLITS} must be the two Phase splits (powerplay, middle); got {splits!r}")
    max_ball_share = float(splits[1])

    # 1. The guard's inputs and the Out_Of_Fold guard, before anything is computed, saved or written.
    balls = store.load_table(reconstruct.STAGE_NAME, reconstruct.BALLS_TABLE)
    fold_table = store.load_table(folds.STAGE_NAME, folds.FOLD_TABLE)
    primary = model_choice.load_primary(store, variants=(LI_VARIANT,))
    pressure_table = store.load_table(pressure.STAGE_NAME, pressure.PRESSURE_TABLE)
    view_by_k = _view_tables(store, n_folds)
    m3_tables, m4_fold_models = _model_tables(store, primary.candidates)
    inputs = guard_inputs(
        balls=balls, fold_table=fold_table, primary=primary, pressure_table=pressure_table, view_by_k=view_by_k,
        m3_tables=m3_tables, m4_fold_models=m4_fold_models, n_folds=n_folds,
    )  # fmt: skip
    del m3_tables, m4_fold_models, fold_table
    guard = oof_guard(inputs, ctx.registry)

    # 2. The other inputs: state, result labels, the innings-1 Primary_Model and the first-innings M2 fits.
    candidates = dict(primary.candidates)
    m3_family_1 = str(candidates[model_choice.M3_MODELS[STATE_INNINGS[0]]])
    innings1_models = win_prob.load_fold_models(store, _m3_stage(m3_family_1), STATE_INNINGS[0], m3_family_1, n_folds)
    m2_models = pressure.first_innings_models(store, n_folds)
    par_means = store.load_table(remaining_runs.STAGE_NAME, remaining_runs.PAR_MEANS_TABLE)
    par_bias = store.load_table(remaining_runs.STAGE_NAME, remaining_runs.PAR_BIAS_TABLE)
    row_ids = np.asarray(primary.row, dtype=np.int64)
    rows = _aligned(balls, row_ids, _BALL_COLUMNS, "balls")
    del balls
    state = _aligned(store.load_table(STATE_STAGE, STATE_TABLE), row_ids, _STATE_COLUMNS, "state")
    labels = _aligned(store.load_table(win_prob.STAGE_NAME, win_prob.LABELS_TABLE), row_ids, ("row", "label", "labelled"),
                      "m3_labels")  # fmt: skip
    table = _aligned(pressure_table, row_ids, _PRESSURE_COLUMNS, "pressure_table")
    del pressure_table
    deliveries = inputs.deliveries
    match = _ints(deliveries["match_id"], "match_id")
    innings = _ints(deliveries["innings"], "innings")
    fold = _ints(deliveries["fold"], "fold")
    batter = _flags(deliveries["batter"], "batter")
    validation = _flags(deliveries["validation"], "validation")
    cls = _ints(rows["cls"], "balls column 'cls'")
    live = _flags(table["live"], "pressure column 'live'")
    li = _floats(table["li"], "pressure column 'li'")

    # 3. Flags over every Delivery of innings 1 and 2 (as the Pressure_Engine ranks), then the frame.
    high = high_leverage_balls(li, live, innings)
    free = extras_free_flags(table["pi_bowl_extras_free"], live, innings, share)
    saved_free = _flags(table["pressure_bowl_extras_free"], "pressure column 'pressure_bowl_extras_free'")
    free_differs = int((free != saved_free).sum())
    rules = simple_rule_flags(rows["overid"], innings, state["rrr"])
    view_values = pressure.own_fold_view(view_by_k, fold, row_ids, columns=_SWING_VIEW_COLUMNS)
    del view_by_k
    at = np.flatnonzero(validation)
    swing_inputs = pd.DataFrame(
        {
            "row": row_ids[at], "match_id": match[at], "innings": innings[at], "fold": fold[at], "cls": cls[at],
            "runs": _ints(rows["runs"], "balls column 'runs'")[at],
            "team_runs": _floats(rows["team_runs"], "balls column 'team_runs'")[at],
            "counted_wicket": _flags(rows["counted_wicket"], "balls column 'counted_wicket'")[at],
            "bowler_wicket": _flags(rows["bowler_wicket"], "balls column 'bowler_wicket'")[at],
            "run_out": _flags(rows["run_out"], "balls column 'run_out'")[at],
            "sched_balls": _floats(rows["sched_balls"], "balls column 'sched_balls'")[at],
            "result_type": rows["result_type"].iloc[at].astype(str).to_numpy(dtype=object),
            "label": _floats(labels["label"], "m3_labels column 'label'")[at],
            "labelled": _flags(labels["labelled"], "m3_labels column 'labelled'")[at],
            **{name: _floats(state[name], f"state column {name!r}")[at]
               for name in ("lb_bowled", "lb_left", "score", "wickets", "runs_needed")},
            "wp_pre": np.asarray(primary.wp_pre, dtype=np.float64)[at],
            **{name: view_values[name].to_numpy(dtype=np.float64)[at] for name in _SWING_VIEW_COLUMNS},
            views.M2_FIT: inputs.sources["view_m2_fit"].to_numpy(dtype=object)[at],
        },
        columns=list(SWING_INPUT_COLUMNS),
    )  # fmt: skip
    swings = actual_wp_swing(
        swing_inputs, np.asarray(primary.wp_post, dtype=np.float64)[at], innings1_models=innings1_models,
        m2_models=m2_models, par_means=par_means, par_bias=par_bias,
    )  # fmt: skip
    del primary, swing_inputs
    frame = pd.DataFrame(
        {
            "row": row_ids[at].astype(np.int32), "match_id": match[at], "innings": innings[at], "fold": fold[at],
            "cls": cls[at].astype(np.int8), "batter": batter[at],
            "overid": _ints(rows["overid"], "balls column 'overid'")[at],
            "ball_share": _floats(state["ball_share"], "state column 'ball_share'")[at],
            "rrr": _floats(state["rrr"], "state column 'rrr'")[at],
            "striker_id": _ints(rows["striker_id"], "balls column 'striker_id'")[at],
            "bowler_id": _ints(rows["bowler_id"], "balls column 'bowler_id'")[at],
            "bat_runs": _ints(rows["runs"], "balls column 'runs'")[at],
            "striker_out": striker_dismissals(rows["counted_wicket"], rows["retire_out"], rows["dismissed_is_striker"])[at],
            "bowler_runs_conceded": bowler_runs_conceded(rows["team_runs"], rows["extras_type"])[at],
            "extra": np.isin(cls, _EXTRA_CLASSES)[at], "live": live[at], "li": li[at],
            "pi_bat": _floats(table["pi_bat"], "pressure column 'pi_bat'")[at],
            "pi_bowl": _floats(table["pi_bowl"], "pressure column 'pi_bowl'")[at],
            "pi_bowl_extras_free": _floats(table["pi_bowl_extras_free"], "pressure column 'pi_bowl_extras_free'")[at],
            "pressure_bat": table["pressure_bat"].iloc[at].reset_index(drop=True).astype("boolean"),
            "pressure_bowl": _flags(table["pressure_bowl"], "pressure column 'pressure_bowl'")[at],
            "pressure_bowl_extras_free": free[at], "high_leverage": high[at],
            RULE_DEATH_OVERS: rules[RULE_DEATH_OVERS][at], RULE_CHASE_RATE: rules[RULE_CHASE_RATE][at],
            "wp_before": swings["wp_before"].to_numpy(dtype=np.float64),
            "wp_after": swings["wp_after"].to_numpy(dtype=np.float64),
            "wp_after_rule": swings["wp_after_rule"].reset_index(drop=True),
            "swing": swings["swing"].to_numpy(dtype=np.float64),
            "big_moment": np.zeros(len(at), dtype=bool),
        },
        columns=list(FRAME_COLUMNS),
    )  # fmt: skip
    frame["big_moment"] = big_moments(frame)

    # 4. V1-V4, the signal-model matrix and the WP-after rule counts.
    methods = {
        METHOD_BATTER: frame["pressure_bat"], METHOD_BOWLER: frame["pressure_bowl"],
        RULE_DEATH_OVERS: frame[RULE_DEATH_OVERS], RULE_CHASE_RATE: frame[RULE_CHASE_RATE],
    }  # fmt: skip
    v4_dead, v4_unflagged = v4_tables(frame, max_ball_share=max_ball_share)
    tables: dict[str, pd.DataFrame] = {
        V1_TABLE: v1_deciles(frame),
        V2_TABLE: v2_precision_recall(frame, methods),
        V3_EXTRAS_TABLE: v3_extras(frame),
        V3_EXTRAS_FREE_TABLE: v3_extras_free(frame),
        V3_EFFECTS_TABLE: v3_within_player(frame),
        V4_DEAD_TABLE: v4_dead,
        V4_UNFLAGGED_TABLE: v4_unflagged,
        MATRIX_TABLE: signal_model_matrix(),
        WP_AFTER_TABLE: wp_after_summary(frame["innings"], frame["wp_after_rule"], frame["wp_after"], frame["swing"]),
        GUARD_TABLE: guard.table,
    }

    # 5. The checkpoint.
    store.save_table(STAGE_NAME, FRAME_TABLE, frame)
    for name, result in tables.items():
        store.save_table(STAGE_NAME, name, result)

    # 6. The files: every table as CSV and the charts of V1-V4 in outputs/validation/.
    folder = settings.path("outputs") / VALIDATION_FOLDER
    files = [_write_csv(folder / TABLES[name], result) for name, result in tables.items()]
    files.append(_write_v1_chart(folder / V1_CHART, tables[V1_TABLE]))
    files.append(_write_v2_chart(folder / V2_CHART, tables[V2_TABLE]))
    files.append(_write_v3_extras_chart(folder / V3_EXTRAS_CHART, tables[V3_EXTRAS_TABLE], tables[V3_EXTRAS_FREE_TABLE]))
    files.append(_write_v3_effects_chart(folder / V3_EFFECTS_CHART, tables[V3_EFFECTS_TABLE]))
    files.append(_write_v4_chart(folder / V4_CHART, v4_dead, v4_unflagged))

    # 7. The data-derived Method_Constant: the realised LI cut of the High_Leverage_Balls.
    cuts = {str(group): float(li[high & (innings == group)].min()) for group in STATE_INNINGS if (high & (innings == group)).any()}
    ctx.constants.record(HIGH_LEVERAGE_CUT_CONSTANT, cuts, _COMPONENT)

    _log_stage(
        ctx, guard=guard, frame=frame, tables=tables, candidates=candidates, free_differs=free_differs,
        n_bowler_live=int(live.sum()), cuts=cuts, files=files, seconds=time.perf_counter() - started,
    )  # fmt: skip


def _log_stage(
    ctx: StageContext,
    *,
    guard: GuardResult,
    frame: pd.DataFrame,
    tables: Mapping[str, pd.DataFrame],
    candidates: Mapping[str, str],
    free_differs: int,
    n_bowler_live: int,
    cuts: Mapping[str, float],
    files: Sequence[Path],
    seconds: float,
) -> None:
    log = ctx.log
    models = ", ".join(f"{name} {candidates[name]}" for name in model_choice.MODELS if name in candidates)
    checks = guard.table
    log.info(
        f"Out_Of_Fold guard passed (Requirement 14, criterion 11): {guard.n_validation:,} Validation_Deliveries in "
        f"{guard.n_matches:,} matches, each with a WP, an LI and the PI of its Roles from the Primary_Models ({models}); "
        f"{guard.n_fits:,} fits checked across {len(checks):,} checks, none trained on a Delivery's match or served it "
        "without excluding its fold",
    )
    innings = frame["innings"].to_numpy()
    counts = "; ".join(
        f"innings {group}: {int((innings == group).sum()):,} Deliveries in "
        f"{len(np.unique(frame['match_id'].to_numpy()[innings == group])):,} matches"
        for group in STATE_INNINGS
    )
    log.info(f"Validation_Deliveries (no Super_Over_Innings, no excluded match): {counts}")
    after = tables[WP_AFTER_TABLE]
    log.info(
        "WP after a Delivery (module docstring): "
        + "; ".join(
            f"{record['rule']} innings {int(record['innings'])}: {int(record['n_deliveries']):,} (mean swing "
            f"{float(record['mean_swing']):.4f}, max {float(record['max_swing']):.4f})"
            for record in after.to_dict("records")
        ),
    )
    v1 = tables[V1_TABLE]
    parts = []
    for role in ROLES:
        for group in STATE_INNINGS:
            mine = v1.iloc[np.flatnonzero((v1["role"].astype(str).to_numpy() == role) & (v1["innings"].to_numpy() == group))]
            by = {("dead" if str(r["group"]) == DEAD_GROUP else int(r["decile"])): float(r["mean_swing"])
                  for r in mine.to_dict("records")}  # fmt: skip
            parts.append(
                f"{role} innings {group}: decile 1 {_number(by.get(1))}, decile 5 {_number(by.get(5))}, decile 10 "
                f"{_number(by.get(10))}, Dead_Balls {_number(by.get('dead'))}"
            )
    log.info("V1 mean Actual_WP_Swing by PI decile (Requirement 14, criterion 2): " + "; ".join(parts))
    v2 = tables[V2_TABLE]
    log.info(
        f"V2 Big_Moments (Requirement 14, criterion 3; {int(v2['n_matches'].iloc[0]):,} matches with at least "
        f"{MIN_MATCH_DELIVERIES} Validation_Deliveries, Big_Moment share {float(v2['big_moment_share'].iloc[0]):.4f}): "
        + "; ".join(
            f"{record['method']}: share {float(record['share_flagged']):.4f}, precision {float(record['precision']):.4f}, "
            f"recall {float(record['recall']):.4f}"
            for record in v2.to_dict("records")
        ),
    )
    for name, label in ((V3_EXTRAS_TABLE, "Pressure_Ball flags"), (V3_EXTRAS_FREE_TABLE, "Extras_Free_PI selection")):
        part = tables[name]
        log.info(
            f"V3 Wides + No_Balls per 100 Bowler_Deliveries, {label} (Requirement 14, criteria 4 and 5): "
            + "; ".join(
                f"innings {record['innings']}: Pressure {float(record['rate_pressure']):.3f} "
                f"({int(record['n_pressure']):,}), Ordinary {float(record['rate_ordinary']):.3f} "
                f"({int(record['n_ordinary']):,}), difference {float(record['difference']):+.3f}"
                for record in part.to_dict("records")
            ),
        )
    message = (
        f"Extras_Free_PI re-selection (top D3 of each Innings_Group's Live_Balls by Extras_Free_PI): "
        f"{free_differs:,} of the {n_bowler_live:,} Live_Balls differ from the pressure table's flags"
    )
    if free_differs:
        ctx.log.warning(message + "; the validation uses the re-selection of Requirement 14, criterion 5")
    else:
        log.info(message)
    v3 = tables[V3_EFFECTS_TABLE]
    log.info(
        "V3 within-player effects of Pressure_Balls (player fixed effects, match-clustered SE, t-based 95% CI; "
        "Requirement 14, criteria 6 and 12): "
        + "; ".join(
            f"{record['measure']} {float(record['beta']):+.4f} (SE {float(record['se']):.4f}, 95% CI "
            f"[{float(record['ci_low']):+.4f}, {float(record['ci_high']):+.4f}]; {int(record['n_players']):,} players, "
            f"{int(record['n_deliveries']):,} Deliveries, {int(record['n_clusters']):,} matches)"
            for record in v3.to_dict("records")
        ),
    )
    dead = tables[V4_DEAD_TABLE]
    unflagged = tables[V4_UNFLAGGED_TABLE]
    log.info(
        "V4 Simple_Rules (Requirement 14, criteria 7 and 8): "
        + "; ".join(
            f"{record['rule']} innings {record['innings']}: {int(record['n_dead']):,} of {int(record['n_flagged']):,} "
            f"Dead_Balls ({_number(record['dead_share'])})"
            for record in dead.to_dict("records")
        )
        + "; High_Leverage_Balls up to the D8 split flagged by neither rule: "
        + "; ".join(
            f"innings {record['innings']}: {int(record['n_unflagged']):,} of {int(record['n_high_leverage']):,} "
            f"({_number(record['unflagged_share'])})"
            for record in unflagged.to_dict("records")
        )
        + f"; lowest LI of a High_Leverage_Ball {', '.join(f'innings {k} {v:.4f}' for k, v in cuts.items())}",
    )
    log.info(
        f"wrote {', '.join(_shown_path(path, ctx.root) for path in files)}; saved {FRAME_TABLE} and "
        f"{', '.join(tables)} in {seconds:.1f} s",
    )
