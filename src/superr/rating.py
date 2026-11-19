"""Rating_Engine: Delivery_Weights, per-player contributions and their aggregation (Requirement 16,
criteria 1-8 and 10-12; Requirement 2, criterion 12), eligibility (Requirement 17), Replacement_Level,
Shrinkage, Split_Half_Reliability, the Blend_Weights, the Final_Score and the ranking (Requirement
18, criteria 1-13 and 15; Requirement 2, criterion 5), and the ``rating`` Pipeline_Stage; design
section "Rating_Engine (``rating.py``)" and Data Models, ``contributions``, ``player_estimates`` and
``role_summary``.

No function reads ``config.py``: every Config value is an argument (``run_stage`` reads them).

Match weights (``match_weights``)
    One row per match of the ``matches`` table, sorted by ``match_id`` (MATCH_WEIGHT_COLUMNS). The
    Delivery_Weight of every Delivery of a match is the match's ``delivery_weight`` =
    ``recency_weight`` x ``stakes_weight`` (glossary). It is a property of the match alone, so no
    weight depends on PI, on the Pressure_Ball flag or on the opposition (Requirement 16,
    criterion 1; Requirement 2, criterion 12).

    * Recency_Weight = 0.5 ** (days / (D9 x 365.25)), where days runs from the match's
      ``match_date`` to the reference date, the last ``match_date`` of ``matches``. Passing every
      match of ``matches.csv`` makes the reference 2026-11-01 as delivered, which is also the
      date of the last rated match. The latest matches weigh 1, and a match D9 years older weighs
      0.5. D9 = None (the D9 Robustness_Case) means no decay: every Recency_Weight is 1.
    * Stakes_Weight = D10 for a match with ``knock_out`` = 1 and 1.0 for every other match.

    The table also holds the per-match attributes that later steps use:

    * ``calendar_year``: the year of ``match_date`` (Calendar_Year);
    * ``comp_type``: the Competition_Type (``folds.match_comp_types``);
    * ``full_member_match``: a T20I whose two teams are both Full_Members. The teams are
      ``home_team_id`` and ``away_team_id``, named through ``teams.csv`` by (``comp_id``,
      ``team_id``); a Full_Member's ``team_name`` is exactly one of ``data_rules.full_members``;
    * ``comp_class`` (CompClass): IPL; FULL_MEMBER for a Full_Member_Match; ASSOCIATE for every
      other match, including every A_Team match (Associate_Match);
    * ``period`` (Period): EARLY for a ``match_date`` on or before 2024-12-31, LATE from
      LATE_PERIOD_START (2025-01-01), whatever the Season (Early_Period, Late_Period).

Contributions (``build_contributions``, ``Contributions``)
    One row per player, Role, match, cell and pressure class. The rows of one Role hold:

    * ``s``: the sum of Delivery_Weight x NBV, plus the Delivery_Weight x charge of each
      Dismissal_Charge of the class (batter Role only);
    * ``w``: the sum of Delivery_Weight;
    * ``n``: the unweighted count of the player's Batter_Deliveries (Legal_Balls and No_Balls
      faced as striker, with ``nbv_bat`` and ``pressure_bat``) or Bowler_Deliveries (every
      Delivery bowled, with ``nbv_bowl`` and ``pressure_bowl``).

    A Wide is never a batter row: its striker gets a value only through a Dismissal_Charge. A
    Dismissal_Charge (Requirement 15, criterion 11) adds Delivery_Weight x charge to ``s`` in the
    match and cell of its Delivery, under its own ``pressure`` flag, and adds nothing to ``w`` or
    ``n`` (Requirement 16, criterion 2). A charge with no Batter_Delivery of the same player in
    the same match, cell and class makes a charge-only row (``n`` = 0, ``w`` = 0, ``s`` not 0).
    ``build_contributions`` reads the cell from the ``value`` column named by ``cells``, so the D8
    and D13 Robustness_Cases can pass other cell codes (any whole numbers from 0).

    Each row also carries the attributes of its match: ``period``, ``comp_class`` and
    ``calendar_year``. It also carries ``half``, for Split_Half_Reliability (Requirement 18,
    criterion 7). For each player and Role, the matches with at least one of the player's
    Deliveries in the Role (``n`` above 0) are numbered 1, 2, 3, ... by ``match_date`` and then
    ``match_id``. The rows of odd-numbered matches get Half.ODD and the rows of even-numbered
    matches Half.EVEN. A match in which the player has only Dismissal_Charges in the Role has no
    Delivery in the Role, so it gets no number: its rows get Half.NONE. They count in every
    full-sample estimate and in neither half; as delivered, 23 batter-matches are charge-only.

Aggregation (``Aggregator``, ``PlayerEstimates``, ``uplift``)
    ``Aggregator(contrib, n_matches)`` indexes one Role's rows once. ``estimates(mult)`` then
    computes every per-player estimate with ``numpy.bincount``. ``mult`` holds one whole-number
    multiplicity per match of the contributions' match universe: all ones for the Main_Ranking,
    or the draw counts of a Bootstrap_Resample, in which a match drawn k times counts as k
    matches and k clusters (Requirement 19, criterion 2). Every sum below is taken with these
    multiplicities. For player p, with S, W and N the sums of ``s``, ``w`` and ``n``:

    * Clutch_Level C = 6 x S_pr / W_pr over the pressure rows, for a player with at least one
      Pressure_Ball (criterion 3).
    * Matched_Cell c: a cell with N_pr,c > 0 and N_ord,c > 0. A charge is not a ball, so it never
      makes a cell matched. Usual_Baseline O = 6 x sum over Matched_Cells of pi_c x mu_c, with
      pi_c = W_pr,c / W_pr,M (W_pr,M = the pressure weight of the Matched_Cells, so the shares
      sum to 1) and mu_c = S_ord,c / W_ord,c (criterion 4). O is NaN, not 0, when the player has
      no Matched_Cell (criterion 12).
    * Unmatched_Pressure_Share = the pressure weight in cells that are not Matched_Cells / W_pr,
      from 0 to 1, and exactly 0 when every cell with a Pressure_Ball is matched (criterion 5).
    * Uplift U = C - max(RL, O), and U = C - RL when O is missing or below RL (criteria 6 and 10;
      ``uplift``). Replacement_Level RL is computed from the Eligible_Players' Clutch_Levels, so
      it is passed in: ``estimates(..., replacement_level=RL)`` or
      ``PlayerEstimates.with_replacement_level(RL)``.
    * Cluster-robust standard errors (criterion 7; design formulas). The influence of one copy of
      match m is psiC_m = 6 x (S_m - (C/6) x W_m) / W_pr over the match's pressure rows, and
      psiO_m = 6 x [sum_c W_pr,m,c x (mu_c - O/6) / W_pr,M
      + sum_c pi_c x (S_ord,m,c - mu_c x W_ord,m,c) / W_ord,c] over Matched_Cells. Then
      SE^2 = G / (G - 1) x sum_m mult_m x psi_m^2, with G = sum_m mult_m x [m contributes]. For
      C, a match contributes when it has a pressure row (a Pressure_Ball or a pressure charge).
      For O, it contributes when it has a Pressure_Ball in a Matched_Cell (which moves the cell
      shares pi_c) or an ordinary row in a Matched_Cell. For C - O, it contributes when it has a
      pressure row or an ordinary row in a Matched_Cell. The Matched_Cells are held fixed.
      ``se_clutch`` is the SE of C, ``se_usual`` the SE of O (psiO) and ``se_clutch_minus_usual``
      the SE of the Pressure_Gap C - O (psiC - psiO); ``g_clutch``, ``g_usual`` and
      ``g_clutch_minus_usual`` are their G. The SE of U is ``se_clutch_minus_usual`` when O > RL
      and ``se_clutch`` otherwise, because RL is a fixed value. Every SE is NaN for a player with
      fewer than MIN_SE_PRESSURE_MATCHES (2) Pressure_Matches, and the SEs of O and C - O are NaN
      when O is missing.
    * Hit_Rate: the share of the player's Pressure_Matches whose summed weighted pressure NBV,
      charges included, is strictly above 0 (criterion 8).
    * Worst_Quartile_Mean: each Pressure_Match has the per-match pressure NBV
      6 x S_pr,m / W_pr,m. The value is the unweighted mean of these values over the lowest
      ceil(n / 4) of the player's n Pressure_Matches, each drawn copy counted (criterion 11).
    * Counts, unweighted except for the multiplicities: Pressure_Balls, Ordinary_Balls,
      Pressure_Matches (matches with a Pressure_Ball), matches with a Delivery in the Role, and
      Matched_Cells.

    A charge-only row adds its ``s`` to the sums of its class: the Clutch_Level numerator, the
    per-match pressure sums of Hit_Rate and Worst_Quartile_Mean when its match is a
    Pressure_Match, and the ordinary cell mean of a Matched_Cell. Its match counts as a cluster.
    It adds nothing to weights, ball counts, Pressure_Matches, Matched_Cells or cell shares.

    Every quantity of a match that does not depend on ``mult`` is computed once, when the
    Aggregator is built: the per-match psi inputs, Pressure_Match and hit flags, and the order of
    each player's per-match pressure NBV. A call is then about 25 array operations on the rows of
    the Role. ``subset`` (player ids) only selects and orders the output.

Eligibility (``eligibility_counts``, ``eligibility``, ``eligibility_grid``, ``select_min_balls``,
``eligibility_report``, ``require_eligible``)
    Every count is unweighted, with no Recency_Weight or Stakes_Weight (criterion 1), and comes
    from the contributions, so it covers the Deliveries that the rating uses: innings 1-2 of the
    matches used for player rating. ``eligibility_counts`` gives, per player and Role:

    * ``n_deliveries``: Batter_Deliveries or Bowler_Deliveries (the sum of ``n``);
    * ``n_pressure_balls``: Pressure_Balls; ``n_pressure_matches``: Pressure_Matches, the distinct
      matches with at least one of them (criterion 1). Both equal the PlayerEstimates counts;
    * ``n_major_pressure_balls``: the Pressure_Balls of IPL matches and Full_Member_Matches
      (CompClass IPL and FULL_MEMBER), and ``major_share`` = that count / ``n_pressure_balls``,
      NaN without a Pressure_Ball (criterion 5);
    * ``n_calendar_years``: the distinct Calendar_Years of the matches with a Delivery of the
      player in the Role (criterion 3);
    * ``recent_activity``: ``is_played`` true in a ``playing_eleven`` row of a match whose
      Calendar_Year is a recent year (criterion 4).

    A charge-only row (``n`` = 0) is not a Delivery, so it adds to no count and gives no
    Pressure_Match or Calendar_Year.

    ``eligibility`` applies the five rules (ELIGIBILITY_RULES) at given minimums and returns one
    failure flag per rule and ``eligible``. ``eligibility_grid`` counts the Eligible_Players at
    every pair of the D11 and D18 grids with criteria 3-5 on (criterion 6). ``select_min_balls``
    picks each Role's minimum Pressure_Ball count (criteria 7 and 8) and returns its warnings for
    the caller to log. ``eligibility_report`` gives the criterion-11 summary, and
    ``require_eligible`` raises StageError (exit code 5) for a Role with fewer than LIST_SIZE
    Eligible_Players (criterion 9). The Recent_Activity_Rule and every threshold are arguments, so
    the Activity_Comparison_Ranking (rule off, same minimums) and the D11 and D18
    Robustness_Cases use the same functions.

    Decisions where the criteria leave room:

    * Excluded matches. Requirement 5, criterion 24 removes every Delivery of a match excluded
      under Score_Tolerance from player rating, so these Deliveries count for no rule: not for the
      2-Calendar_Year rule (criterion 3), and not for the players with a Delivery whom the report
      counts (criterion 11). Super_Over_Innings are never in the contributions. As delivered (48
      excluded matches), counting excluded matches would add 6 batters and 3 bowlers, whose only
      Deliveries are in excluded matches, to the report's players. It would also give 6 batters and
      3 bowlers a second Calendar_Year. None of them passes the other rules even at the loosest
      grid point (45 Pressure_Balls, 5 Pressure_Matches, Recent_Activity_Rule off), so no
      Eligible_Player changes at any grid point.
    * The Recent_Activity_Rule reads the XI, not the Deliveries: every match of ``matches``
      counts, excluded or not, whether or not the player batted or bowled in it. A missing
      ``is_played`` reads false.
    * The share rule asks for at least ``min_share`` of the Pressure_Ball count. A player without a
      Pressure_Ball meets it (0 of 0), so the report does not count them under it; the minimum
      Pressure_Ball rule excludes them. As delivered, 106 batters and 65 bowlers have Deliveries
      but no Pressure_Ball.
    * With the Recent_Activity_Rule off, no player fails it, and its report count is 0.

Shrinkage, reliability, blend and ranking (``rank_role`` and its parts)
    ``prepare_role(contrib, role, eligible)`` builds a RoleData once: three Aggregators over the
    Eligible_Players' rows of the Role (all rows, and the rows of Half.ODD and Half.EVEN
    matches). Each player's estimates depend on that player's rows alone, so they equal those
    from all the Role's rows, bit for bit. ``rank_role(data, mult, replacement_pct=D12)`` then
    runs Requirement 18 for that fixed Eligible_Player set under the match multiplicities
    ``mult`` (all ones for the Main_Ranking, draw counts for a Bootstrap_Resample):

    1. The full-sample estimates. The pool is the set of players behind Replacement_Level, tau^2
       and Split_Half_Reliability, and the only ones ranked. It holds every Eligible_Player with
       a Clutch_Level: all of them at mult all ones, since each has the minimum Pressure_Ball
       count (Requirement 17, criterion 10). With ``rankable_only`` (the bootstrap), it holds the
       Eligible_Players with at least RANKABLE_MIN_PRESSURE_MATCHES Pressure_Matches and
       RANKABLE_MIN_ORDINARY_BALLS Ordinary_Ball under ``mult``; the others are unranked
       (Requirement 19, criterion 4).
    2. Replacement_Level (``replacement_level``): the D12 percentile of the pool's unshrunk
       Clutch_Levels, with numpy's default "linear" interpolation (criterion 6). Uplift and its
       SE follow from it (``PlayerEstimates.with_replacement_level``).
    3. tau^2 (``tau2``) of four estimates: the Usual_Baseline O (SE ``se_usual``), the
       Pressure_Gap g = C - O (SE ``se_clutch_minus_usual``), Uplift U (SE ``se_uplift``) and,
       for the fallback below, Clutch_Level C (SE ``se_clutch``). Each is the sample variance
       (n - 1) of the pool's unshrunk estimates minus the mean squared match-clustered SE, over
       the pool players whose estimate and SE are defined (criterion 4). A value of at most 0
       becomes 0, with a full-shrinkage warning (criterion 5). So does a tau^2 that cannot be
       estimated because fewer than 2 players have a defined SE. The Clutch_Level tau^2 is warned
       about only when a pool player uses the fallback.
    4. Shrinkage (``shrink``; criteria 1-3): B = tau^2 / (tau^2 + SE^2) (glossary
       Shrinkage_Factor), from 0 to 1, for each estimate with its own tau^2 and SE.

       * Shrunk Usual_Baseline O* = Replacement_Level + B_O x (O - Replacement_Level).
       * Shrunk Pressure_Gap g* = B_g x g.
       * Shrunk Clutch_Level C* = O* + g*: the player's own usual level, shrunk, plus the shrunk
         change under pressure. Between-player differences in Clutch_Level are mostly
         differences in usual level (design DD-16), so the shrinkage keeps them even when the
         Pressure_Gap has no measurable between-player variance (B_g = 0, C* = O*).
       * Fallback, for a pool player without a Usual_Baseline (no Matched_Cell): C* =
         Replacement_Level + B_C x (C - Replacement_Level) with the Clutch_Level tau^2. These
         players get a ``clutch_fallback`` warning; as delivered there are none.
       * Shrunk Uplift U* = B_U x U.

       B is 0 when the SE is undefined or tau^2 and the SE are both 0, with a warning that lists
       the player_ids (criterion 3). With B = 0 the shrunk value is the prior exactly. So O*
       lies between O and Replacement_Level, g* between g and 0, U* between U and 0, and a
       fallback C* between C and Replacement_Level (criterion 14). C* itself need not lie
       between C and Replacement_Level.
    5. Split_Half_Reliability (``split_half``, ``reliability``; criteria 7-9): the unshrunk C and
       U of each half come from the rows of the player's odd and even matches
       (``Contributions.half``, fixed by the full data, so a drawn copy of a match stays in its
       half). They use the same Delivery_Weights, the same multiplicities and the full-sample
       Replacement_Level. r is the Pearson correlation over the pool players with at least one
       Pressure_Ball and one Ordinary_Ball in each half, each drawn copy counted. Every such
       player has finite half estimates, so one player count (``n_rho``) serves both rho.
       rho = 2r / (1 + r). rho is set to 0, with a warning, when it is negative or r cannot be
       computed: fewer than MIN_RELIABILITY_PLAYERS players, zero variance in a half, or r = -1.
       Zero variance means every value of the half is equal, tested exactly (max = min) rather
       than on a computed variance, which rounding can leave just above 0. r is clipped to
       [-1, 1].
    6. Blend_Weights (``blend_weights``): w_L = rho_L / (rho_L + rho_U) and w_U = 1 - w_L, or w_L
       = 1 and w_U = 0 with a warning when both are 0 (criterion 10). The Main_Ranking always uses
       these weights (Requirement 2, criterion 5). A fixed-blend or Z-score Robustness_Case
       re-blends the RoleRanking's shrunk values with ``final_scores`` and ``rank_order``.
    7. Final_Score (``final_scores``): w_L x shrunk C + w_U x shrunk U, in runs per over, with no
       Z-score (criteria 11 and 12).
    8. Ranks (``rank_order``; ``rank`` on a table; criterion 13). Two Final_Scores are tied when
       they agree to TIE_DECIMALS (6) decimal places, which is tested on the whole-number key
       rint(Final_Score x 10^6). numpy rounds half to even, and the key equals
       numpy.round(Final_Score, 6) x 10^6. numpy.lexsort orders by that key (descending), then
       by shrunk Clutch_Level at full precision (descending), then by ``player_id`` (ascending).
       The ranks 1, 2, 3... are therefore a total order, whatever the row order. A player
       without a finite Final_Score is unranked: rank NaN in a RoleRanking, <NA> in a table.

    ``rank_role`` returns a RoleRanking with the per-player estimates, shrunk values, B values,
    Final_Scores and ranks, plus the Role values and the warnings (RatingWarning). It logs
    nothing. On the delivered data a call takes about 9-11 ms, with all-ones or bootstrap
    multiplicities (three ``estimates`` calls on about 8,000-20,000 rows each, then a few array
    operations on the Eligible_Players). It also accepts Contributions with ``role=`` and
    ``eligible=``, building the RoleData for that one call.

    Fixed Role-level values (``rank_role(..., fixed=RoleHyper)``; Requirement 19, criterion 2).
    A Bootstrap_Resample holds Replacement_Level, the four tau^2, both rho and the Blend_Weights
    at the values of the ranking it resamples, because re-estimating them in each resample
    inflates tau^2. With ``fixed``, steps 2, 3 (tau^2 only), 5 and 6 take its values: the
    estimates and SEs under ``mult``, the Shrinkage_Factors (fixed tau^2, each player's SE), the
    shrunk values, Final_Score and ranks are computed as above, and the pool only decides who is
    ranked. The halves are skipped, so a call costs one ``estimates`` call (about 3 ms).
    ``RoleHyper.from_ranking`` and ``RoleHyper.from_summary`` (a ``role_summary`` row) give the
    values of a ranking; with mult all ones, the call then reproduces that ranking bit for bit.

The ``rating`` Pipeline_Stage (``run_stage``)
    Reads the ``load`` tables (matches, competitions, teams, playing_eleven, ball_by_ball), the
    ``reconstruct`` balls, the ``pressure`` table and the ``value`` and ``dismissal_charges``
    tables. Steps:

    1. ``match_weights`` (D9, D10, the Full_Members) and ``build_contributions`` on the value
       table's ``cell`` column. D13 must be ``"innings_phase"``, the cells of that column; the
       D13 alternative is a Robustness_Case, so another value stops the stage.
    2. Eligibility: ``eligibility_counts``, ``eligibility_grid`` (D11 x D18 grids),
       ``select_min_balls`` (D11 target at the D18 default), ``eligibility`` and
       ``eligibility_report`` at the selected minimums. The selection warnings are logged, and
       then ``require_eligible`` stops a short Role.
    3. Per Role, ``prepare_role`` and ``rank_role`` with mult all ones and D12, and every
       player's estimates against the Role's Replacement_Level.
    4. Checkpoint tables: ``contributions`` (Contributions.to_frame, halves included);
       ``match_universe`` (MATCH_UNIVERSE_COLUMNS: the match_weights rows, which are the
       contributions' match universe and the positions of every ``mult``, with ``resampled`` and
       ``rated``); ``eligibility_grid``; ``eligibility_report``; ``player_estimates``
       (PLAYER_TABLE_COLUMNS, one row per player and Role, whose ``eligible`` flags fix the
       Eligible_Player sets); ``role_summary`` (ROLE_SUMMARY_COLUMNS); and ``warnings``
       (WARNING_COLUMNS). The bootstrap rebuilds its inputs from these tables:
       ``Contributions.from_frame(contributions, match_universe.match_id)``, then
       ``prepare_role`` on the eligible ids.
    5. Files in ``outputs/rating/``: ``player_estimates_<role>.csv`` (with ``player_name``, ranked
       players first), ``provisional_top10_<role>.csv`` (TOP10_COLUMNS, criterion 15),
       ``role_summary.csv``, ``eligibility_grid.csv``, ``eligibility_report.csv`` and
       ``warnings.csv``. The names are provisional: each player's most frequent ``striker_name``
       (batter) or ``bowler_name`` (bowler) in ``ball_by_ball``, ties going to the first name in
       sort order. The List_Builder sets the names of the final lists.
    6. Data-derived constants ``rating.<quantity>.<role>`` for CONSTANT_QUANTITIES: the selected
       minimum Pressure_Ball count, the Eligible_Players, Replacement_Level, the four tau^2
       (Usual_Baseline, Pressure_Gap, Uplift and the fallback Clutch_Level), both rho and the
       Blend_Weights. Every warning goes to the Run_Log.

Errors: ValueError or TypeError, naming the problem, for a missing column, a value of the wrong
type, a missing or non-finite value where one is needed, a row or match that the other inputs do
not hold, a Delivery_Weight that is not above 0, and contributions or multiplicities that do not
fit together. ``require_eligible`` and ``run_stage`` raise StageError.
"""

from __future__ import annotations

import dataclasses
import math
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import pandas as pd

from superr import folds
from superr.errors import StageError
from superr.pressure import PRESSURE_TABLE
from superr.pressure import STAGE_NAME as PRESSURE_STAGE
from superr.reconstruct import BALLS_TABLE, COMP_IPL, COMP_T20I, DeliveryClass
from superr.reconstruct import STAGE_NAME as RECONSTRUCT_STAGE
from superr.value import BATTER_ROLE, BOWLER_ROLE, CELL_COLUMN, DISMISSAL_CHARGES_TABLE, ROLES, VALUE_TABLE
from superr.value import STAGE_NAME as VALUE_STAGE

if TYPE_CHECKING:
    from superr.pipeline import StageContext

__all__ = [
    "Aggregator",
    "BlendWeights",
    "CLUTCH_LEVEL",
    "COMP_CLASS_NAMES",
    "CONSTANT_QUANTITIES",
    "CONTRIBUTIONS_TABLE",
    "CONTRIBUTION_COLUMNS",
    "CompClass",
    "Contributions",
    "ELIGIBILITY_COLUMNS",
    "ELIGIBILITY_COUNT_COLUMNS",
    "ELIGIBILITY_GRID_COLUMNS",
    "ELIGIBILITY_GRID_FILE",
    "ELIGIBILITY_GRID_TABLE",
    "ELIGIBILITY_REPORT_COLUMNS",
    "ELIGIBILITY_REPORT_FILE",
    "ELIGIBILITY_REPORT_TABLE",
    "ELIGIBILITY_RULES",
    "Half",
    "LATE_PERIOD_START",
    "LIST_SIZE",
    "MATCH_UNIVERSE_COLUMNS",
    "MATCH_UNIVERSE_TABLE",
    "MATCH_WEIGHT_COLUMNS",
    "MIN_RELIABILITY_PLAYERS",
    "MIN_SE_PRESSURE_MATCHES",
    "MinBallsSelection",
    "PERIOD_NAMES",
    "PLAYER_ESTIMATES_FILES",
    "PLAYER_ESTIMATES_TABLE",
    "PLAYER_ESTIMATE_COLUMNS",
    "PLAYER_TABLE_COLUMNS",
    "PRESSURE_GAP",
    "Period",
    "PlayerEstimates",
    "RANKABLE_MIN_ORDINARY_BALLS",
    "RANKABLE_MIN_PRESSURE_MATCHES",
    "RANKING_COLUMNS",
    "RANKING_SUMMARY_FIELDS",
    "RATING_FOLDER",
    "ROLE_CODES",
    "ROLE_HYPER_FIELDS",
    "ROLE_SUMMARY_COLUMNS",
    "ROLE_SUMMARY_FILE",
    "ROLE_SUMMARY_TABLE",
    "RatingWarning",
    "Reliability",
    "RoleData",
    "RoleHyper",
    "RoleRanking",
    "STAGE_NAME",
    "Shrunk",
    "SplitHalf",
    "TIE_DECIMALS",
    "TOP10_COLUMNS",
    "TOP10_FILES",
    "Tau2",
    "UPLIFT",
    "USUAL_BASELINE",
    "WARNINGS_FILE",
    "WARNINGS_TABLE",
    "WARNING_COLUMNS",
    "WARNING_KINDS",
    "blend_weights",
    "build_contributions",
    "eligibility",
    "eligibility_counts",
    "eligibility_grid",
    "eligibility_report",
    "final_scores",
    "match_weights",
    "prepare_role",
    "rank",
    "rank_order",
    "rank_role",
    "reliability",
    "replacement_level",
    "require_eligible",
    "run_stage",
    "select_min_balls",
    "shrink",
    "split_half",
    "tau2",
    "uplift",
]


# ===================================================================================== codes

#: The code of each Role in ``Contributions.role``: its position in ``value.ROLES``.
ROLE_CODES: Mapping[str, int] = {role: code for code, role in enumerate(ROLES)}


class CompClass(IntEnum):
    """The match class behind the eligibility share and the list shares (Requirement 17, criterion 5)."""

    IPL = 0  # Competition_Type IPL
    FULL_MEMBER = 1  # Full_Member_Match: a T20I between two Full_Members
    ASSOCIATE = 2  # Associate_Match: every other match, every A_Team match included


#: Glossary names of the CompClass codes.
COMP_CLASS_NAMES: Mapping[int, str] = {
    CompClass.IPL: "IPL",
    CompClass.FULL_MEMBER: "Full_Member_Match",
    CompClass.ASSOCIATE: "Associate_Match",
}


class Period(IntEnum):
    """The time-holdout period of a match (glossary Early_Period and Late_Period; Requirement 20, criterion 2)."""

    EARLY = 0  # match_date on or before 2024-12-31
    LATE = 1  # match_date on or after LATE_PERIOD_START


#: Glossary names of the Period codes.
PERIOD_NAMES: Mapping[int, str] = {Period.EARLY: "Early_Period", Period.LATE: "Late_Period"}

#: First date of the Late_Period. The glossary fixes it, whatever the competition's Season.
LATE_PERIOD_START = pd.Timestamp("2025-01-01")


class Half(IntEnum):
    """The split-half of a contribution row (Requirement 18, criterion 7; module docstring)."""

    NONE = 0  # a match with only Dismissal_Charges for the player: not numbered
    ODD = 1  # the player's 1st, 3rd, 5th, ... match in the Role
    EVEN = 2  # the player's 2nd, 4th, 6th, ... match in the Role


#: Balls per over: an estimate in runs per over is 6 x runs per ball (Laws of Cricket).
_BALLS_PER_OVER = 6.0

#: Days per year in the Recency_Weight exponent (glossary): a unit conversion.
_DAYS_PER_YEAR = 365.25

#: The Pressure_Matches a player needs for a cluster-robust SE (Requirement 16, criterion 7).
MIN_SE_PRESSURE_MATCHES = 2

#: Worst_Quartile_Mean takes the lowest ceil(n / _QUARTILE) Pressure_Matches (criterion 11).
_QUARTILE = 4

_WIDE_CODE = int(DeliveryClass.WIDE)
_CLASS_CODES = np.asarray([int(code) for code in DeliveryClass], dtype=np.int64)
_KEY_LIMIT = 2**62  # the combined grouping key stays below this, inside int64


# ========================================================================= argument checks

def _require(frame: Any, columns: Sequence[str], what: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{what} must be a DataFrame; got {type(frame).__name__}")
    absent = [column for column in dict.fromkeys(columns) if column not in frame.columns]
    if absent:
        raise ValueError(f"{what} lacks the column(s) {', '.join(absent)}")


def _numbers(values: Any, what: str) -> np.ndarray:
    """A 1-D Series or array of numbers as a new float64 array, NaN for a missing value."""
    if isinstance(values, (pd.Series, pd.Index, pd.api.extensions.ExtensionArray)):
        dtype = values.dtype
        if pd.api.types.is_bool_dtype(dtype) or not pd.api.types.is_numeric_dtype(dtype):
            raise TypeError(f"{what} must hold numbers; got dtype {dtype}")
        array = pd.Series(values).to_numpy(dtype=np.float64, na_value=np.nan)
    else:
        array = np.asarray(values)
        if array.dtype.kind not in "iuf":
            raise TypeError(f"{what} must hold numbers; got dtype {array.dtype}")
        array = array.astype(np.float64)
    if array.ndim != 1:
        raise ValueError(f"{what} must be 1-D; got shape {array.shape}")
    return array


def _finite(values: Any, what: str) -> np.ndarray:
    array = _numbers(values, what)
    bad = int((~np.isfinite(array)).sum())
    if bad:
        raise ValueError(f"{what} holds {bad:,} missing or infinite value(s)")
    return array


def _whole(values: Any, what: str) -> np.ndarray:
    """Whole numbers, none missing, as int64."""
    array = _finite(values, what)
    if not (np.floor(array) == array).all():
        raise ValueError(f"{what} must hold whole numbers")
    return array.astype(np.int64)


def _flags(values: Any, what: str) -> tuple[np.ndarray, np.ndarray]:
    """A 1-D boolean Series or array as (flag, missing) bool arrays; a missing flag reads False."""
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    if not pd.api.types.is_bool_dtype(series.dtype):
        raise TypeError(f"{what} must be boolean; got dtype {series.dtype}")
    missing = series.isna().to_numpy(dtype=bool)
    return series.astype("boolean").to_numpy(dtype=bool, na_value=False), missing


def _dates(values: Any, what: str) -> np.ndarray:
    """A datetime Series as datetime64[ns] dates (time of day dropped), none missing."""
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    if not pd.api.types.is_datetime64_any_dtype(series.dtype):
        raise TypeError(f"{what} must be a datetime column, as the Data_Loader reads it; got {series.dtype}")
    if series.isna().any():
        raise ValueError(f"{what} has {int(series.isna().sum()):,} missing value(s)")
    return series.dt.normalize().to_numpy(dtype="datetime64[ns]")


def _unique_ids(values: Any, what: str) -> np.ndarray:
    ids = _whole(values, what)
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{what} must hold each id once")
    return ids


def _positions(ids: Any, wanted: np.ndarray, what: str) -> np.ndarray:
    """The position in ``ids`` (unique) of each value of ``wanted``; every one must be there."""
    index = pd.Index(_unique_ids(ids, f"{what} column 'row'"))
    at = index.get_indexer(wanted)
    if (at < 0).any():
        raise ValueError(f"{int((at < 0).sum()):,} Delivery row(s) of value have no row in {what}")
    return at


def _half_life(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"half_life_years (D9) must be a number of years or None; got {value!r}")
    years = float(value)
    if not (math.isfinite(years) and years > 0.0):
        raise ValueError(f"half_life_years (D9) must be above 0, or None for no decay; got {value!r}")
    return years


def _positive(value: Any, what: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{what} must be a number; got {value!r}")
    number = float(value)
    if not (math.isfinite(number) and number > 0.0):
        raise ValueError(f"{what} must be above 0; got {value!r}")
    return number


def _codes(values: Any, kind: type[IntEnum], what: str) -> np.ndarray:
    codes = _whole(values, what)
    allowed = np.asarray([int(code) for code in kind], dtype=np.int64)
    if not np.isin(codes, allowed).all():
        raise ValueError(f"{what} must hold {kind.__name__} codes {allowed.tolist()}")
    return codes


def _frozen(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


# =========================================================================== match weights

#: Columns of the match_weights table, one row per match sorted by ``match_id`` (module docstring).
MATCH_WEIGHT_COLUMNS: tuple[str, ...] = (
    "match_id", "match_date", "calendar_year", "knock_out", "comp_type", "full_member_match", "comp_class",
    "period", "days_to_last", "recency_weight", "stakes_weight", "delivery_weight",
)  # fmt: skip

_MATCH_COLUMNS: tuple[str, ...] = ("match_id", "comp_id", "match_date", "knock_out", "home_team_id", "away_team_id")


def _team_names(teams: pd.DataFrame) -> dict[tuple[int, int], str]:
    _require(teams, ("comp_id", "team_id", "team_name"), "teams")
    comp = _whole(teams["comp_id"], "teams column 'comp_id'")
    team = _whole(teams["team_id"], "teams column 'team_id'")
    names = teams["team_name"].astype("string")
    if names.isna().any():
        raise ValueError(f"teams column 'team_name' has {int(names.isna().sum()):,} missing value(s)")
    lookup: dict[tuple[int, int], str] = {}
    for key, name in zip(zip(comp.tolist(), team.tolist()), names.tolist()):
        if lookup.setdefault(key, name) != name:
            raise ValueError(f"teams gives comp_id {key[0]}, team_id {key[1]} two names")
    return lookup


def match_weights(
    matches: pd.DataFrame,
    competitions: pd.DataFrame,
    teams: pd.DataFrame,
    *,
    half_life_years: float | None,
    knockout_weight: float,
    full_members: Sequence[str],
) -> pd.DataFrame:
    """The Delivery_Weight and the attributes of every match (module docstring, "Match weights").

    ``matches``, ``competitions`` and ``teams`` are the tables of the ``load`` checkpoint (or
    frames with the same columns): ``matches`` needs ``match_id``, ``comp_id``, ``match_date``
    (datetime), ``knock_out``, ``home_team_id`` and ``away_team_id``; ``competitions`` what
    ``folds.match_comp_types`` reads; ``teams`` ``comp_id``, ``team_id`` and ``team_name``. The
    rows of ``matches`` are the match universe, and the last ``match_date`` among them is the
    Recency_Weight reference date. ``half_life_years`` is D9
    (``design_choices.D9_half_life_years``; None means no decay), ``knockout_weight`` D10
    (``design_choices.D10_knockout_weight``) and ``full_members`` the Full_Member team names
    (``data_rules.full_members``), compared exactly with ``team_name``.

    Returns MATCH_WEIGHT_COLUMNS, one row per match sorted by ``match_id``, on a RangeIndex:
    ``match_id`` (int64), ``match_date`` (datetime64[ns], the date), ``calendar_year`` (int64),
    ``knock_out`` (float64, NaN when missing), ``comp_type`` (str), ``full_member_match`` (bool),
    ``comp_class`` and ``period`` (int8 codes), ``days_to_last`` (float64, days from the match to
    the reference date), ``recency_weight``, ``stakes_weight`` and ``delivery_weight`` (float64,
    above 0). Raises ValueError or TypeError for a missing column, a missing or repeated
    ``match_id``, a missing date, a match team that ``teams`` does not name, or a Config value
    of the wrong type; and the errors of ``folds.match_comp_types``.
    """
    _require(matches, _MATCH_COLUMNS, "matches")
    half_life = _half_life(half_life_years)
    knockout = _positive(knockout_weight, "knockout_weight (D10)")
    if isinstance(full_members, (str, bytes)) or not all(isinstance(name, str) for name in full_members):
        raise TypeError("full_members must be a list of team names (data_rules.full_members)")
    members = set(full_members)

    ids = _unique_ids(matches["match_id"], "matches column 'match_id'")
    comp = _whole(matches["comp_id"], "matches column 'comp_id'")
    dates = _dates(matches["match_date"], "matches column 'match_date'")
    knock = _numbers(matches["knock_out"], "matches column 'knock_out'")
    comp_type = folds.match_comp_types(matches, competitions).to_numpy(dtype=object)
    names = _team_names(teams)
    sides = {}
    for side in ("home_team_id", "away_team_id"):
        team = _whole(matches[side], f"matches column {side!r}")
        keys = list(zip(comp.tolist(), team.tolist()))
        unnamed = [key for key in keys if key not in names]
        if unnamed:
            raise ValueError(
                f"teams names no team for {len(unnamed):,} {side} value(s) of matches (for example comp_id "
                f"{unnamed[0][0]}, team_id {unnamed[0][1]}), so the Full_Member_Matches cannot be found"
            )
        sides[side] = np.asarray([names[key] in members for key in keys], dtype=bool)
    full_member = (comp_type == COMP_T20I) & sides["home_team_id"] & sides["away_team_id"]
    comp_class = np.where(
        comp_type == COMP_IPL, CompClass.IPL, np.where(full_member, CompClass.FULL_MEMBER, CompClass.ASSOCIATE)
    ).astype(np.int8)
    period = np.where(dates < LATE_PERIOD_START.to_datetime64(), Period.EARLY, Period.LATE).astype(np.int8)
    days = (dates.max() - dates) / np.timedelta64(1, "D") if len(dates) else np.zeros(0)
    recency = np.ones(len(ids)) if half_life is None else 0.5 ** (days / (half_life * _DAYS_PER_YEAR))
    stakes = np.where(knock == 1.0, knockout, 1.0)
    order = np.argsort(ids, kind="stable")
    table = pd.DataFrame(
        {
            "match_id": ids,
            "match_date": dates,
            "calendar_year": pd.DatetimeIndex(dates).year.to_numpy(dtype=np.int64),
            "knock_out": knock,
            "comp_type": pd.array(comp_type.astype(str), dtype="str"),
            "full_member_match": full_member,
            "comp_class": comp_class,
            "period": period,
            "days_to_last": np.asarray(days, dtype=np.float64),
            "recency_weight": np.asarray(recency, dtype=np.float64),
            "stakes_weight": stakes.astype(np.float64),
            "delivery_weight": np.asarray(recency * stakes, dtype=np.float64),
        },
        columns=list(MATCH_WEIGHT_COLUMNS),
    )
    return table.iloc[order].reset_index(drop=True)


# =========================================================================== contributions

#: Columns of Contributions.to_frame, the ``contributions`` table (design, Data Models).
CONTRIBUTION_COLUMNS: tuple[str, ...] = (
    "role", "player_id", "match_id", "cell", "pressure", "s", "w", "n", "half", "period", "comp_class",
    "calendar_year",
)  # fmt: skip

_ROW_FIELDS: tuple[tuple[str, Any], ...] = (
    ("player", np.int64), ("role", np.int8), ("match_idx", np.int64), ("cell", np.int64), ("pressure", np.bool_),
    ("s", np.float64), ("w", np.float64), ("n", np.int64), ("half", np.int8), ("period", np.int8),
    ("comp_class", np.int8), ("calendar_year", np.int64),
)  # fmt: skip


def _row_array(values: Any, dtype: Any, what: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{what} must be 1-D; got shape {array.shape}")
    if dtype is np.bool_:
        if array.dtype != np.bool_:
            raise TypeError(f"{what} must be a bool array; got dtype {array.dtype}")
        return array.copy()
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{what} must hold numbers; got dtype {array.dtype}")
    if np.dtype(dtype).kind == "f":
        array = array.astype(np.float64)
        if not np.isfinite(array).all():
            raise ValueError(f"{what} holds {int((~np.isfinite(array)).sum()):,} missing or infinite value(s)")
        return array
    if array.dtype.kind == "f" and not (np.isfinite(array).all() and (np.floor(array) == array).all()):
        raise ValueError(f"{what} must hold whole numbers")
    return array.astype(dtype)


@dataclass(frozen=True, eq=False)
class Contributions:
    """Per-player contributions: one row per (player, Role, match, cell, pressure class) (module docstring).

    Row fields, all 1-D read-only copies of one length:

    * ``player`` (int64): the ``player_id``; ``role`` (int8): ROLE_CODES (0 batter, 1 bowler);
    * ``match_idx`` (int64): the match's position in ``match_ids``, the match universe that a
      multiplicity vector follows; ``cell`` (int64): the cell code, 0 or more; ``pressure``
      (bool): the pressure class (True for Pressure_Balls, False for Ordinary_Balls);
    * ``s`` (float64): sum of Delivery_Weight x NBV, plus Delivery_Weight x charge for the batter
      Role's Dismissal_Charges; ``w`` (float64): sum of Delivery_Weight, 0 or more; ``n``
      (int64): unweighted Batter_Delivery or Bowler_Delivery count. ``n`` is 0 exactly when ``w``
      is 0 (a charge-only row), so every row with a Delivery has a positive weight;
    * ``half`` (int8): Half code; ``period`` (int8): Period code; ``comp_class`` (int8):
      CompClass code; ``calendar_year`` (int64): the Calendar_Year of the match.

    ``match_ids`` (int64): the distinct ``match_id`` of each match index. Raises ValueError or
    TypeError for fields of different lengths, a wrong dtype, a non-finite value, or a code or
    index out of range.
    """

    player: np.ndarray
    role: np.ndarray
    match_idx: np.ndarray
    cell: np.ndarray
    pressure: np.ndarray
    s: np.ndarray
    w: np.ndarray
    n: np.ndarray
    half: np.ndarray
    period: np.ndarray
    comp_class: np.ndarray
    calendar_year: np.ndarray
    match_ids: np.ndarray

    def __post_init__(self) -> None:
        arrays = {name: _row_array(getattr(self, name), dtype, f"Contributions.{name}") for name, dtype in _ROW_FIELDS}
        lengths = {len(array) for array in arrays.values()}
        if len(lengths) > 1:
            raise ValueError("every Contributions row field must have one value per row")
        match_ids = _row_array(self.match_ids, np.int64, "Contributions.match_ids")
        if len(np.unique(match_ids)) != len(match_ids):
            raise ValueError("Contributions.match_ids must hold each match_id once")
        checks = (
            ("role", np.isin(arrays["role"], list(ROLE_CODES.values())), "a Role code 0 or 1"),
            ("match_idx", (arrays["match_idx"] >= 0) & (arrays["match_idx"] < len(match_ids)),
             "a position in match_ids"),
            ("cell", arrays["cell"] >= 0, "a cell code of 0 or more"),
            ("w", arrays["w"] >= 0.0, "a weight of 0 or more"),
            ("n", (arrays["n"] >= 0) & ((arrays["n"] > 0) == (arrays["w"] > 0.0)),
             "a count of 0 or more that is 0 exactly when the weight is 0"),
            ("half", np.isin(arrays["half"], [int(code) for code in Half]), "a Half code"),
            ("period", np.isin(arrays["period"], [int(code) for code in Period]), "a Period code"),
            ("comp_class", np.isin(arrays["comp_class"], [int(code) for code in CompClass]), "a CompClass code"),
        )  # fmt: skip
        for name, ok, rule in checks:
            if not ok.all():
                raise ValueError(f"Contributions.{name} breaks the rule '{rule}' on {int((~ok).sum()):,} row(s)")
        for name, array in arrays.items():
            object.__setattr__(self, name, _frozen(array))
        object.__setattr__(self, "match_ids", _frozen(match_ids))

    def __len__(self) -> int:
        return len(self.player)

    def __repr__(self) -> str:
        roles = ", ".join(f"{role} {int((self.role == code).sum()):,}" for role, code in ROLE_CODES.items())
        return f"Contributions({len(self):,} rows: {roles}; {self.n_matches:,} matches)"

    @property
    def n_matches(self) -> int:
        """The size of the match universe: the length of every multiplicity vector."""
        return len(self.match_ids)

    @property
    def match_id(self) -> np.ndarray:
        """The ``match_id`` of each row."""
        return self.match_ids[self.match_idx]

    def take(self, rows: Any) -> Contributions:
        """The rows selected by ``rows`` (a bool mask or row positions), with the same match universe."""
        selector = np.asarray(rows)
        if selector.dtype == np.bool_ and selector.shape != (len(self),):
            raise ValueError(f"a row mask must have one value per row ({len(self):,}); got shape {selector.shape}")
        values = {name: getattr(self, name)[selector] for name, _ in _ROW_FIELDS}
        return Contributions(**values, match_ids=self.match_ids)

    def for_role(self, role: str) -> Contributions:
        """The rows of one Role (``"batter"`` or ``"bowler"``)."""
        if role not in ROLE_CODES:
            raise ValueError(f"role must be one of {', '.join(ROLE_CODES)}; got {role!r}")
        return self.take(self.role == ROLE_CODES[role])

    def to_frame(self) -> pd.DataFrame:
        """The rows as the ``contributions`` table (CONTRIBUTION_COLUMNS), Role names as text."""
        return pd.DataFrame(
            {
                "role": pd.array(np.asarray(ROLES, dtype=object)[self.role].astype(str), dtype="str"),
                "player_id": self.player,
                "match_id": self.match_id,
                "cell": self.cell,
                "pressure": self.pressure,
                "s": self.s,
                "w": self.w,
                "n": self.n,
                "half": self.half,
                "period": self.period,
                "comp_class": self.comp_class,
                "calendar_year": self.calendar_year,
            },
            columns=list(CONTRIBUTION_COLUMNS),
        )

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, match_ids: Any) -> Contributions:
        """Contributions from a ``contributions`` table (to_frame) and the match universe ``match_ids``."""
        _require(frame, CONTRIBUTION_COLUMNS, "frame")
        universe = _unique_ids(match_ids, "match_ids")
        roles = frame["role"].astype("string")
        unknown = ~roles.isin(list(ROLE_CODES)).to_numpy(dtype=bool)
        if unknown.any():
            raise ValueError(f"frame column 'role' holds {int(unknown.sum()):,} value(s) other than a Role name")
        at = pd.Index(universe).get_indexer(_whole(frame["match_id"], "frame column 'match_id'"))
        if (at < 0).any():
            raise ValueError(f"{int((at < 0).sum()):,} row(s) of frame have a match_id that match_ids does not hold")
        pressure, missing = _flags(frame["pressure"], "frame column 'pressure'")
        if missing.any():
            raise ValueError("frame column 'pressure' has missing value(s)")
        return cls(
            player=_whole(frame["player_id"], "frame column 'player_id'"),
            role=roles.map(ROLE_CODES).to_numpy(dtype=np.int8),
            match_idx=at.astype(np.int64),
            cell=_whole(frame["cell"], "frame column 'cell'"),
            pressure=pressure,
            s=_finite(frame["s"], "frame column 's'"),
            w=_finite(frame["w"], "frame column 'w'"),
            n=_whole(frame["n"], "frame column 'n'"),
            half=_whole(frame["half"], "frame column 'half'"),
            period=_whole(frame["period"], "frame column 'period'"),
            comp_class=_whole(frame["comp_class"], "frame column 'comp_class'"),
            calendar_year=_whole(frame["calendar_year"], "frame column 'calendar_year'"),
            match_ids=universe,
        )


class _Items(NamedTuple):
    """One Role's value entries before grouping: one per Delivery or Dismissal_Charge."""

    player: np.ndarray  # int64 player_id
    match: np.ndarray  # int64 match index
    cell: np.ndarray  # int64 cell code
    pressure: np.ndarray  # bool
    s: np.ndarray  # Delivery_Weight x NBV (or x charge)
    w: np.ndarray  # Delivery_Weight (0 for a charge)
    n: np.ndarray  # 1 for a Delivery, 0 for a charge


def _halves(player: np.ndarray, match: np.ndarray, n: np.ndarray, n_matches: int, chrono: np.ndarray) -> np.ndarray:
    """The Half code of each grouped row of one Role (module docstring, "Contributions")."""
    pm_keys, pm_inverse = np.unique(player * n_matches + match, return_inverse=True)
    pm_player, pm_match = pm_keys // n_matches, pm_keys % n_matches
    numbered = np.flatnonzero(np.bincount(pm_inverse, n.astype(np.float64), len(pm_keys)) > 0)
    order = numbered[np.lexsort((chrono[pm_match[numbered]], pm_player[numbered]))]
    position = np.arange(len(order))
    sorted_player = pm_player[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = sorted_player[1:] != sorted_player[:-1]
    number = position - np.maximum.accumulate(np.where(first, position, 0)) + 1
    pm_half = np.full(len(pm_keys), Half.NONE, dtype=np.int8)
    pm_half[order] = np.where(number % 2 == 1, Half.ODD, Half.EVEN)
    return pm_half[pm_inverse]


def _grouped(items: _Items, n_matches: int, n_cells: int, chrono: np.ndarray) -> dict[str, np.ndarray]:
    """One Role's entries summed per (player, match, cell, class), sorted by those keys, with halves."""
    players, player_code = np.unique(items.player, return_inverse=True)
    if len(players) * n_matches * n_cells * 2 >= _KEY_LIMIT:
        raise ValueError("too many players, matches and cells to group the contributions with one int64 key")
    key = ((player_code.astype(np.int64) * n_matches + items.match) * n_cells + items.cell) * 2 + items.pressure
    keys, inverse = np.unique(key, return_inverse=True)
    size = len(keys)
    rest = keys // 2
    cell = rest % n_cells
    rest //= n_cells
    match, code = rest % n_matches, rest // n_matches
    n = np.rint(np.bincount(inverse, items.n, size)).astype(np.int64)
    return {
        "player": players[code],
        "match_idx": match,
        "cell": cell,
        "pressure": (keys % 2).astype(bool),
        "s": np.bincount(inverse, items.s, size),
        "w": np.bincount(inverse, items.w, size),
        "n": n,
        "half": _halves(code, match, n, n_matches, chrono),
    }


def build_contributions(
    value: pd.DataFrame,
    pressure: pd.DataFrame,
    charges: pd.DataFrame,
    balls: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    cells: str = CELL_COLUMN,
) -> Contributions:
    """The per-player contributions of both Roles (Requirement 16, criteria 1 and 2; module docstring).

    * ``value``: the ``value`` table of the ``value`` checkpoint, one row per rated Delivery
      (innings 1-2 of the matches used for player rating), with ``row``, ``match_id``, ``cls``,
      ``nbv_bat`` (finite on every Legal_Ball and No_Ball), ``nbv_bowl`` (finite) and the cell
      code column named by ``cells`` (``cell``, the D13 innings x Phase cell, by default).
    * ``pressure``: a table with ``row``, ``pressure_bat`` (nullable boolean, set on every
      Batter_Delivery) and ``pressure_bowl`` (bool), such as the ``pressure`` table.
    * ``charges``: the ``dismissal_charges`` table (``row``, ``player_id``, ``charge``,
      ``pressure``), whose rows are rows of ``value``.
    * ``balls``: a table with ``row``, ``striker_id`` and ``bowler_id``, such as the ``balls``
      table of the ``reconstruct`` checkpoint.
    * ``weights``: the match_weights table, the match universe: ``match_id``, ``match_date``,
      ``calendar_year``, ``comp_class``, ``period`` and ``delivery_weight`` (above 0) of every
      match of ``value`` and possibly more.

    The ``pressure`` and ``balls`` rows are matched to ``value`` by ``row``. Returns the
    Contributions of the batter Role, then the bowler Role, each sorted by player, match index,
    cell and class (ordinary first); ``match_idx`` indexes the rows of ``weights``. Raises
    ValueError or TypeError for a missing column, a repeated row, a row or match that another
    input lacks, a missing pressure flag on a Batter_Delivery, a missing or non-finite NBV,
    charge or weight, a negative cell code, or a Delivery_Weight that is not above 0.
    """
    if not isinstance(cells, str) or not cells:
        raise TypeError(f"cells must name the cell code column of value; got {cells!r}")
    _require(value, ("row", "match_id", "cls", "nbv_bat", "nbv_bowl", cells), "value")
    _require(pressure, ("row", "pressure_bat", "pressure_bowl"), "pressure")
    _require(charges, ("row", "player_id", "charge", "pressure"), "charges")
    _require(balls, ("row", "striker_id", "bowler_id"), "balls")
    _require(weights, ("match_id", "match_date", "calendar_year", "comp_class", "period", "delivery_weight"), "weights")

    # The match universe, its weights and its chronological order (match_date, then match_id).
    match_ids = _unique_ids(weights["match_id"], "weights column 'match_id'")
    n_matches = len(match_ids)
    delivery_weight = _finite(weights["delivery_weight"], "weights column 'delivery_weight'")
    if not (delivery_weight > 0.0).all():
        raise ValueError(
            f"weights column 'delivery_weight' holds {int((delivery_weight <= 0).sum()):,} value(s) of 0 or less"
        )
    dates = _dates(weights["match_date"], "weights column 'match_date'")
    chrono = np.empty(n_matches, dtype=np.int64)
    chrono[np.lexsort((match_ids, dates.astype(np.int64)))] = np.arange(n_matches)
    period = _codes(weights["period"], Period, "weights column 'period'")
    comp_class = _codes(weights["comp_class"], CompClass, "weights column 'comp_class'")
    calendar_year = _whole(weights["calendar_year"], "weights column 'calendar_year'")

    # The rated Deliveries, their match, cell, class, players and pressure flags.
    rows = _unique_ids(value["row"], "value column 'row'")
    match_idx = pd.Index(match_ids).get_indexer(_whole(value["match_id"], "value column 'match_id'")).astype(np.int64)
    if (match_idx < 0).any():
        raise ValueError(f"{int((match_idx < 0).sum()):,} row(s) of value have a match_id that weights does not hold")
    cell = _whole(value[cells], f"value column {cells!r}")
    if (cell < 0).any():
        raise ValueError(f"value column {cells!r} must hold cell codes of 0 or more")
    cls = _whole(value["cls"], "value column 'cls'")
    if not np.isin(cls, _CLASS_CODES).all():
        raise ValueError(f"value column 'cls' must hold DeliveryClass codes {_CLASS_CODES.tolist()}")
    batter = cls != _WIDE_CODE
    at_pressure = _positions(pressure["row"], rows, "pressure")
    at_balls = _positions(balls["row"], rows, "balls")
    flag_bat, no_flag = _flags(pressure["pressure_bat"].iloc[at_pressure], "pressure column 'pressure_bat'")
    if (no_flag & batter).any():
        raise ValueError(
            f"pressure column 'pressure_bat' is missing on {int((no_flag & batter).sum()):,} Batter_Delivery(ies)"
        )
    flag_bowl, no_flag = _flags(pressure["pressure_bowl"].iloc[at_pressure], "pressure column 'pressure_bowl'")
    if no_flag.any():
        raise ValueError(f"pressure column 'pressure_bowl' has {int(no_flag.sum()):,} missing value(s)")
    striker = _whole(balls["striker_id"].iloc[at_balls[batter]], "balls column 'striker_id' on the Batter_Deliveries")
    bowler = _whole(balls["bowler_id"].iloc[at_balls], "balls column 'bowler_id'")
    nbv_bat = _finite(value["nbv_bat"].iloc[np.flatnonzero(batter)], "value column 'nbv_bat' on the Batter_Deliveries")
    nbv_bowl = _finite(value["nbv_bowl"], "value column 'nbv_bowl'")
    dw = delivery_weight[match_idx]

    # The Dismissal_Charges: the match, cell and weight of their Delivery, their own class.
    at_charge = pd.Index(rows).get_indexer(_whole(charges["row"], "charges column 'row'"))
    if (at_charge < 0).any():
        raise ValueError(f"{int((at_charge < 0).sum()):,} Dismissal_Charge(s) have a row that value does not hold")
    charge_player = _whole(charges["player_id"], "charges column 'player_id'")
    charge = _finite(charges["charge"], "charges column 'charge'")
    charge_flag, no_flag = _flags(charges["pressure"], "charges column 'pressure'")
    if no_flag.any():
        raise ValueError(f"charges column 'pressure' has {int(no_flag.sum()):,} missing value(s)")

    n_bat, n_charges = int(batter.sum()), len(at_charge)
    items = {
        BATTER_ROLE: _Items(
            player=np.concatenate([striker, charge_player]),
            match=np.concatenate([match_idx[batter], match_idx[at_charge]]),
            cell=np.concatenate([cell[batter], cell[at_charge]]),
            pressure=np.concatenate([flag_bat[batter], charge_flag]),
            s=np.concatenate([dw[batter] * nbv_bat, dw[at_charge] * charge]),
            w=np.concatenate([dw[batter], np.zeros(n_charges)]),
            n=np.concatenate([np.ones(n_bat), np.zeros(n_charges)]),
        ),
        BOWLER_ROLE: _Items(
            player=bowler, match=match_idx, cell=cell, pressure=flag_bowl, s=dw * nbv_bowl, w=dw, n=np.ones(len(rows))
        ),
    }
    n_cells = int(cell.max()) + 1 if len(cell) else 1
    parts = {role: _grouped(items[role], n_matches, n_cells, chrono) for role in ROLES}
    joined = {name: np.concatenate([parts[role][name] for role in ROLES]) for name in parts[BATTER_ROLE]}
    role = np.concatenate([np.full(len(parts[r]["s"]), ROLE_CODES[r], dtype=np.int8) for r in ROLES])
    at = joined["match_idx"]
    return Contributions(
        **joined,
        role=role,
        period=period[at].astype(np.int8),
        comp_class=comp_class[at].astype(np.int8),
        calendar_year=calendar_year[at],
        match_ids=match_ids,
    )


# ============================================================================ aggregation

def uplift(clutch: Any, usual: Any, replacement_level: float) -> np.ndarray:
    """Uplift = Clutch_Level - max(Replacement_Level, Usual_Baseline) (Requirement 16, criteria 6 and 10).

    A missing (NaN) Usual_Baseline, or one below ``replacement_level``, gives Clutch_Level -
    Replacement_Level, so lowering a baseline that is already below Replacement_Level leaves the
    Uplift unchanged. NaN where Clutch_Level is NaN. Raises ValueError for a Replacement_Level
    that is not a finite number.
    """
    rl = float(replacement_level)
    if not math.isfinite(rl):
        raise ValueError(f"replacement_level must be a finite number in runs per over; got {replacement_level!r}")
    return np.asarray(clutch, dtype=np.float64) - np.fmax(np.asarray(usual, dtype=np.float64), rl)


#: Columns of PlayerEstimates.to_frame: the Requirement 16 columns of ``player_estimates``.
PLAYER_ESTIMATE_COLUMNS: tuple[str, ...] = (
    "role", "player_id", "n_pressure_balls", "n_ordinary_balls", "n_pressure_matches", "n_matches",
    "n_matched_cells", "w_pressure", "s_pressure", "clutch", "usual", "uplift", "replacement_level", "se_clutch",
    "se_uplift", "se_clutch_minus_usual", "se_usual", "g_clutch", "g_uplift", "g_clutch_minus_usual", "g_usual",
    "unmatched_share", "hit_rate", "worst_quartile",
)  # fmt: skip

_ESTIMATE_ARRAYS: tuple[str, ...] = tuple(
    name for name in PLAYER_ESTIMATE_COLUMNS[1:] if name != "replacement_level"
)

#: The estimates that are NaN, rather than 0, for a player without contributions.
_NAN_WHEN_ABSENT = frozenset(
    {"clutch", "usual", "se_clutch", "se_clutch_minus_usual", "se_usual", "unmatched_share", "hit_rate",
     "worst_quartile"}
)  # fmt: skip


@dataclass(frozen=True, eq=False)
class PlayerEstimates:
    """The Requirement 16 estimates of one Role's players (module docstring, "Aggregation").

    Every array is read-only, with one value per player, aligned with ``player_id`` (int64). The
    order is ascending ``player_id``, or the order of the ``subset`` given to
    ``Aggregator.estimates``. Counts include the match multiplicities, so a match drawn k times
    counts k times. A player that the Role's contributions do not hold has zero counts and NaN
    estimates.

    * ``n_pressure_balls``, ``n_ordinary_balls`` (int64): unweighted Pressure_Balls and
      Ordinary_Balls (Batter_Deliveries or Bowler_Deliveries);
    * ``n_pressure_matches`` (int64): Pressure_Matches, the matches with a Pressure_Ball;
      ``n_matches`` (int64): matches with at least one Delivery in the Role;
      ``n_matched_cells`` (int64): Matched_Cells;
    * ``w_pressure`` (float64): the summed Delivery_Weight of the Pressure_Balls;
      ``s_pressure`` (float64): the summed Delivery_Weight x NBV of the Pressure_Balls plus the
      weighted pressure Dismissal_Charges;
    * ``clutch``: Clutch_Level, runs per over, NaN without a Pressure_Ball (criterion 3);
      ``usual``: Usual_Baseline, runs per over, NaN without a Matched_Cell (criteria 4 and 12);
      ``unmatched_share``: Unmatched_Pressure_Share in [0, 1], NaN without a Pressure_Ball
      (criterion 5);
    * ``se_clutch``: the cluster-robust SE of Clutch_Level; ``se_usual``: the SE of the
      Usual_Baseline; ``se_clutch_minus_usual``: the SE of the Pressure_Gap, Clutch_Level -
      Usual_Baseline. The last two are NaN when the Usual_Baseline is missing, and all three are
      NaN for a player with fewer than MIN_SE_PRESSURE_MATCHES Pressure_Matches (criterion 7).
      ``g_clutch``, ``g_usual`` and ``g_clutch_minus_usual`` (float64, whole numbers) are their
      cluster counts G: the matches that contribute to the estimate, with multiplicity;
    * ``hit_rate``: Hit_Rate in [0, 1] (criterion 8); ``worst_quartile``: Worst_Quartile_Mean in
      runs per over (criterion 11); both NaN without a Pressure_Match;
    * ``replacement_level`` (float): the Replacement_Level behind ``uplift`` (NaN until one is
      set); ``uplift``: Uplift (criteria 6 and 10); ``se_uplift``: its SE,
      ``se_clutch_minus_usual`` when the Usual_Baseline is above Replacement_Level, else
      ``se_clutch``; ``g_uplift``: the matching G. All three are NaN until a Replacement_Level is
      set, by ``Aggregator.estimates(..., replacement_level=)`` or ``with_replacement_level``.
    """

    role: str | None
    player_id: np.ndarray
    n_pressure_balls: np.ndarray
    n_ordinary_balls: np.ndarray
    n_pressure_matches: np.ndarray
    n_matches: np.ndarray
    n_matched_cells: np.ndarray
    w_pressure: np.ndarray
    s_pressure: np.ndarray
    clutch: np.ndarray
    usual: np.ndarray
    uplift: np.ndarray
    se_clutch: np.ndarray
    se_uplift: np.ndarray
    se_clutch_minus_usual: np.ndarray
    se_usual: np.ndarray
    g_clutch: np.ndarray
    g_uplift: np.ndarray
    g_clutch_minus_usual: np.ndarray
    g_usual: np.ndarray
    unmatched_share: np.ndarray
    hit_rate: np.ndarray
    worst_quartile: np.ndarray
    replacement_level: float = math.nan

    def __post_init__(self) -> None:
        for name in _ESTIMATE_ARRAYS:
            array = getattr(self, name)
            if array.flags.writeable:
                _frozen(array)

    def __len__(self) -> int:
        return len(self.player_id)

    def __repr__(self) -> str:
        defined = int(np.isfinite(self.clutch).sum())
        return (
            f"PlayerEstimates({self.role}: {len(self):,} players, {defined:,} with a Clutch_Level; Replacement_Level "
            f"{self.replacement_level:g})"
        )

    def with_replacement_level(self, replacement_level: float) -> PlayerEstimates:
        """A copy with Uplift, its SE and its G for ``replacement_level`` (runs per over, held fixed)."""
        rl = float(replacement_level)
        values = uplift(self.clutch, self.usual, rl)  # checks rl
        above = self.usual > rl  # False where the Usual_Baseline is missing
        return dataclasses.replace(
            self,
            replacement_level=rl,
            uplift=values,
            se_uplift=np.where(above, self.se_clutch_minus_usual, self.se_clutch),
            g_uplift=np.where(above, self.g_clutch_minus_usual, self.g_clutch),
        )

    def to_frame(self) -> pd.DataFrame:
        """The estimates as a table with PLAYER_ESTIMATE_COLUMNS, one row per player."""
        data: dict[str, Any] = {"role": pd.array([self.role] * len(self), dtype="str")}
        for name in PLAYER_ESTIMATE_COLUMNS[1:]:
            data[name] = getattr(self, name)
        data["replacement_level"] = np.full(len(self), self.replacement_level)
        return pd.DataFrame(data, columns=list(PLAYER_ESTIMATE_COLUMNS))


class _CellSums(NamedTuple):
    """The multiplicity-weighted sums of one estimates call, per (player, cell) and per player."""

    s_pr: np.ndarray  # per (player, cell)
    w_pr: np.ndarray
    n_pr: np.ndarray
    s_ord: np.ndarray
    w_ord: np.ndarray
    n_ord: np.ndarray
    matched: np.ndarray  # bool: a Matched_Cell
    mu: np.ndarray  # S_ord / W_ord in a Matched_Cell, runs per ball; 0 elsewhere
    pi: np.ndarray  # the cell's share of the Matched_Cells' pressure weight; 0 elsewhere
    p_s_pr: np.ndarray  # per player
    p_w_pr: np.ndarray
    p_n_pr: np.ndarray
    p_n_ord: np.ndarray
    p_w_matched: np.ndarray  # pressure weight in Matched_Cells
    p_w_unmatched: np.ndarray  # pressure weight in other cells
    p_o: np.ndarray  # Usual_Baseline / 6; 0 without a Matched_Cell
    p_n_matched: np.ndarray  # Matched_Cells


def _divide(numerator: Any, denominator: np.ndarray, where: np.ndarray) -> np.ndarray:
    """numerator / denominator where ``where``, 0 elsewhere."""
    out = np.zeros(np.shape(denominator), dtype=np.float64)
    return np.divide(numerator, denominator, out=out, where=where)


class Aggregator:
    """The Requirement 16 estimates of one Role's players for any match multiplicities (module docstring).

    ``contrib`` holds one Role's rows, or both Roles with ``role`` naming the one to use;
    ``n_matches`` (optional) must equal ``contrib.n_matches``, the length of every ``mult``.
    The constructor indexes the rows once; ``estimates`` is then cheap enough for the bootstrap.
    Attributes: ``role`` (the Role name, or None for empty contributions without ``role``),
    ``players`` (the ascending player ids of the Role's rows), ``n_matches`` and ``n_cells``.
    """

    def __init__(self, contrib: Contributions, n_matches: int | None = None, *, role: str | None = None) -> None:
        if not isinstance(contrib, Contributions):
            raise TypeError(f"contrib must be Contributions; got {type(contrib).__name__}")
        if role is not None:
            contrib = contrib.for_role(role)
        codes = np.unique(contrib.role)
        if len(codes) > 1:
            raise ValueError("contrib holds both Roles; pass role= or the rows of one Role (Contributions.for_role)")
        self.role: str | None = role if role is not None else (ROLES[int(codes[0])] if len(codes) else None)
        if n_matches is not None and (isinstance(n_matches, bool) or int(n_matches) != contrib.n_matches):
            raise ValueError(
                f"n_matches ({n_matches!r}) must equal the size of the contributions' match universe "
                f"({contrib.n_matches:,})"
            )
        self.n_matches = contrib.n_matches
        n_rows, total = len(contrib), self.n_matches
        self.n_cells = int(contrib.cell.max()) + 1 if n_rows else 1

        self.players, player = np.unique(contrib.player, return_inverse=True)
        self._n_players = len(self.players)
        pc_keys, pc = np.unique(player.astype(np.int64) * self.n_cells + contrib.cell, return_inverse=True)
        pm_keys, pm = np.unique(player.astype(np.int64) * total + contrib.match_idx, return_inverse=True)
        self._n_pc, self._n_pm = len(pc_keys), len(pm_keys)
        self._pc_player = (pc_keys // self.n_cells).astype(np.intp)
        self._pc_cell = (pc_keys % self.n_cells).astype(np.intp)
        self._pm_player = (pm_keys // total).astype(np.intp)
        self._pm_match = (pm_keys % total).astype(np.intp)

        # Every row: its match, its (player, cell, class) slot and its unmultiplied sums.
        pressure = contrib.pressure
        s, w, n = contrib.s, contrib.w, contrib.n.astype(np.float64)
        self._row_match = contrib.match_idx.astype(np.intp)
        self._row_slot = (pc * 2 + pressure).astype(np.intp)  # even: ordinary, odd: pressure
        self._s, self._w, self._n = s, w, n
        # The pressure rows and the ordinary rows, for the influence functions.
        pr, od = np.flatnonzero(pressure), np.flatnonzero(~pressure)
        self._pr_player, self._pr_pc, self._pr_pm = player[pr].astype(np.intp), pc[pr], pm[pr]
        self._pr_s, self._pr_w = s[pr], w[pr]
        self._pr_ball = w[pr] > 0.0  # a pressure row with a Pressure_Ball; a charge-only row has no weight
        self._od_pc, self._od_pm = pc[od], pm[od]
        self._od_s, self._od_w = s[od], w[od]

        # Per (player, match), fixed whatever the multiplicities: the pressure sums of one copy.
        n_pm = self._n_pm
        pm_s_pr = np.bincount(pm[pr], s[pr], n_pm)
        pm_w_pr = np.bincount(pm[pr], w[pr], n_pm)
        pm_n_pr = np.bincount(pm[pr], n[pr], n_pm)
        self._pm_pressure_row = (np.bincount(pm[pr], minlength=n_pm) > 0).astype(np.float64)
        is_pm = pm_n_pr > 0
        self._pm_is_pm = is_pm.astype(np.float64)
        self._pm_hit = (is_pm & (pm_s_pr > 0.0)).astype(np.float64)
        self._pm_delivery = (np.bincount(pm, n, n_pm) > 0).astype(np.float64)

        # Worst_Quartile_Mean: each player's Pressure_Matches sorted by per-match pressure NBV.
        q = np.flatnonzero(is_pm)
        per_match = _BALLS_PER_OVER * pm_s_pr[q] / pm_w_pr[q]
        order = np.lexsort((per_match, self._pm_player[q]))
        self._wq_player = self._pm_player[q][order]
        self._wq_match = self._pm_match[q][order]
        self._wq_value = per_match[order]
        first = np.ones(len(order), dtype=bool)
        first[1:] = self._wq_player[1:] != self._wq_player[:-1]
        position = np.arange(len(order))
        self._wq_first = np.maximum.accumulate(np.where(first, position, 0))

    def __repr__(self) -> str:
        return (
            f"Aggregator({self.role}: {self._n_players:,} players, {len(self._s):,} contribution rows, "
            f"{self.n_matches:,} matches, {self.n_cells} cells)"
        )

    # ------------------------------------------------------------------------------- helpers

    def _multiplicities(self, mult: Any) -> np.ndarray:
        if mult is None:
            return np.ones(self.n_matches)
        m = np.asarray(mult)
        if m.shape != (self.n_matches,):
            raise ValueError(f"mult must hold one multiplicity per match ({self.n_matches:,}); got shape {m.shape}")
        if m.dtype.kind not in "iub":
            if m.dtype.kind != "f":
                raise TypeError(f"mult must hold whole numbers; got dtype {m.dtype}")
            if not (np.isfinite(m).all() and (np.floor(m) == m).all()):
                raise ValueError("mult must hold whole numbers")
        m = m.astype(np.float64)
        if (m < 0.0).any():
            raise ValueError("mult must not hold a negative multiplicity")
        return m

    def _cell_sums(self, m: np.ndarray) -> _CellSums:
        row_m = m[self._row_match]
        size, players = 2 * self._n_pc, self._n_players
        s = np.bincount(self._row_slot, row_m * self._s, size)
        w = np.bincount(self._row_slot, row_m * self._w, size)
        n = np.bincount(self._row_slot, row_m * self._n, size)
        s_ord, s_pr, w_ord, w_pr, n_ord, n_pr = s[0::2], s[1::2], w[0::2], w[1::2], n[0::2], n[1::2]
        matched = (n_pr > 0.0) & (n_ord > 0.0)
        cp = self._pc_player
        p_w_matched = np.bincount(cp, np.where(matched, w_pr, 0.0), players)
        mu = _divide(s_ord, w_ord, matched)
        pi = _divide(w_pr, p_w_matched[cp], matched)
        return _CellSums(
            s_pr=s_pr, w_pr=w_pr, n_pr=n_pr, s_ord=s_ord, w_ord=w_ord, n_ord=n_ord, matched=matched, mu=mu, pi=pi,
            p_s_pr=np.bincount(cp, s_pr, players),
            p_w_pr=np.bincount(cp, w_pr, players),
            p_n_pr=np.bincount(cp, n_pr, players),
            p_n_ord=np.bincount(cp, n_ord, players),
            p_w_matched=p_w_matched,
            p_w_unmatched=np.bincount(cp, np.where(matched, 0.0, w_pr), players),
            p_o=np.bincount(cp, pi * mu, players),
            p_n_matched=np.bincount(cp, matched.astype(np.float64), players),
        )  # fmt: skip

    def _positions(self, subset: Any) -> tuple[np.ndarray, np.ndarray] | None:
        if subset is None:
            return None
        ids = _whole(subset, "subset")
        at = np.searchsorted(self.players, ids)
        found = at < self._n_players
        found[found] = self.players[at[found]] == ids[found]
        return ids, np.where(found, at, -1)

    # --------------------------------------------------------------------------- estimates

    def estimates(
        self, mult: Any = None, subset: Any = None, *, replacement_level: float | None = None
    ) -> PlayerEstimates:
        """The PlayerEstimates of the Role's players under the match multiplicities ``mult``.

        ``mult``: one whole number of 0 or more per match of the match universe (None: all ones).
        ``subset``: player ids whose estimates to return, in that order (None: every player of
        the Role, ascending). ``replacement_level``: when given, Uplift and its SE are filled in
        (``PlayerEstimates.with_replacement_level``). Raises ValueError or TypeError for
        multiplicities of the wrong shape or type, or negative ones.
        """
        m = self._multiplicities(mult)
        c = self._cell_sums(m)
        players, n_pm = self._n_players, self._n_pm
        has_pressure = c.p_n_pr > 0.0
        has_usual = c.p_w_matched > 0.0
        ratio = _divide(c.p_s_pr, c.p_w_pr, has_pressure)  # Clutch_Level / 6
        scale = _divide(_BALLS_PER_OVER, c.p_w_pr, has_pressure)
        scale_matched = _divide(_BALLS_PER_OVER, c.p_w_matched, has_usual)
        cp = self._pc_player

        # The influence of one copy of each (player, match), summed over its rows.
        p = self._pr_player
        psi_c = np.bincount(self._pr_pm, scale[p] * (self._pr_s - ratio[p] * self._pr_w), n_pm)
        d = np.where(c.matched, (c.mu - c.p_o[cp]) * scale_matched[cp], 0.0)
        alpha = _divide(_BALLS_PER_OVER * c.pi, c.w_ord, c.matched)
        a = alpha[self._od_pc]
        psi_o = np.bincount(self._pr_pm, d[self._pr_pc] * self._pr_w, n_pm) + np.bincount(
            self._od_pm, a * self._od_s - (alpha * c.mu)[self._od_pc] * self._od_w, n_pm
        )
        in_matched = (np.bincount(self._od_pm[a > 0.0], minlength=n_pm) > 0).astype(np.float64)
        ball_matched = c.matched[self._pr_pc] & self._pr_ball
        pr_matched = (np.bincount(self._pr_pm[ball_matched], minlength=n_pm) > 0).astype(np.float64)
        psi_d = psi_c - psi_o

        # Per player: clusters, squared influence, Pressure_Matches, hits and matches.
        mm = m[self._pm_match]
        pp = self._pm_player
        contributes_c = mm * self._pm_pressure_row
        contributes_o = mm * in_matched
        g_c = np.bincount(pp, contributes_c, players)
        g_o = np.bincount(pp, np.maximum(mm * pr_matched, contributes_o), players)
        g_d = np.bincount(pp, np.maximum(contributes_c, contributes_o), players)
        v_c = np.bincount(pp, mm * psi_c * psi_c, players)
        v_o = np.bincount(pp, mm * psi_o * psi_o, players)
        v_d = np.bincount(pp, mm * psi_d * psi_d, players)
        n_pressure_matches = np.bincount(pp, mm * self._pm_is_pm, players)
        hits = np.bincount(pp, mm * self._pm_hit, players)
        n_matches = np.bincount(pp, mm * self._pm_delivery, players)

        enough = n_pressure_matches >= MIN_SE_PRESSURE_MATCHES
        ok_c = enough & (g_c > 1.0)
        ok_o = enough & has_usual & (g_o > 1.0)
        ok_d = enough & has_usual & (g_d > 1.0)
        se_c = np.full(players, np.nan)
        se_c[ok_c] = np.sqrt(g_c[ok_c] / (g_c[ok_c] - 1.0) * v_c[ok_c])
        se_o = np.full(players, np.nan)
        se_o[ok_o] = np.sqrt(g_o[ok_o] / (g_o[ok_o] - 1.0) * v_o[ok_o])
        se_d = np.full(players, np.nan)
        se_d[ok_d] = np.sqrt(g_d[ok_d] / (g_d[ok_d] - 1.0) * v_d[ok_d])

        # Worst_Quartile_Mean: the lowest ceil(n / 4) Pressure_Matches, each copy counted.
        k = (np.rint(n_pressure_matches).astype(np.int64) + _QUARTILE - 1) // _QUARTILE
        copies = m[self._wq_match]
        before = np.cumsum(copies) - copies
        before -= before[self._wq_first]
        taken = np.clip(k[self._wq_player] - before, 0.0, copies)
        worst = np.full(players, np.nan)
        has_match = k > 0
        worst[has_match] = np.bincount(self._wq_player, taken * self._wq_value, players)[has_match] / k[has_match]

        values = {
            "n_pressure_balls": np.rint(c.p_n_pr).astype(np.int64),
            "n_ordinary_balls": np.rint(c.p_n_ord).astype(np.int64),
            "n_pressure_matches": np.rint(n_pressure_matches).astype(np.int64),
            "n_matches": np.rint(n_matches).astype(np.int64),
            "n_matched_cells": np.rint(c.p_n_matched).astype(np.int64),
            "w_pressure": c.p_w_pr,
            "s_pressure": c.p_s_pr,
            "clutch": np.where(has_pressure, _BALLS_PER_OVER * ratio, np.nan),
            "usual": np.where(has_usual, _BALLS_PER_OVER * c.p_o, np.nan),
            "se_clutch": se_c,
            "se_clutch_minus_usual": se_d,
            "se_usual": se_o,
            "g_clutch": g_c,
            "g_clutch_minus_usual": g_d,
            "g_usual": g_o,
            "unmatched_share": np.where(has_pressure, _divide(c.p_w_unmatched, c.p_w_pr, has_pressure), np.nan),
            "hit_rate": np.where(n_pressure_matches > 0.0, _divide(hits, n_pressure_matches, n_pressure_matches > 0.0),
                                 np.nan),
            "worst_quartile": worst,
        }  # fmt: skip
        ids = self.players
        chosen = self._positions(subset)
        if chosen is not None:  # a player the Role does not hold: zero counts and sums, NaN estimates
            ids, at = chosen
            found = at >= 0
            for name, array in values.items():
                picked = np.full(len(ids), np.nan if name in _NAN_WHEN_ABSENT else 0, dtype=array.dtype)
                picked[found] = array[at[found]]
                values[name] = picked
        blank = np.full(len(ids), np.nan)
        estimates = PlayerEstimates(
            role=self.role, player_id=np.array(ids, dtype=np.int64), uplift=blank, se_uplift=blank.copy(),
            g_uplift=blank.copy(), **values,
        )  # fmt: skip
        return estimates if replacement_level is None else estimates.with_replacement_level(replacement_level)

    def cell_table(self, mult: Any = None) -> pd.DataFrame:
        """The per-(player, cell) inputs of the Usual_Baseline under ``mult`` (for checks and reports).

        One row per player and cell that holds a contribution row: ``player_id``, ``cell``,
        ``n_pressure``, ``w_pressure``, ``n_ordinary``, ``w_ordinary``, ``s_ordinary``,
        ``matched`` (a Matched_Cell), ``share`` (the cell's share of the Matched_Cells' pressure
        weight; 0 when not matched, so a player's shares sum to 1 exactly when the Usual_Baseline
        is defined) and ``ordinary_mean`` (6 x S_ord / W_ord, runs per over; NaN without an
        Ordinary_Ball). Counts include the multiplicities.
        """
        c = self._cell_sums(self._multiplicities(mult))
        has_ordinary = c.w_ord > 0.0
        ordinary_mean = np.where(has_ordinary, _BALLS_PER_OVER * _divide(c.s_ord, c.w_ord, has_ordinary), np.nan)
        return pd.DataFrame(
            {
                "player_id": self.players[self._pc_player],
                "cell": self._pc_cell.astype(np.int64),
                "n_pressure": np.rint(c.n_pr).astype(np.int64),
                "w_pressure": c.w_pr,
                "n_ordinary": np.rint(c.n_ord).astype(np.int64),
                "w_ordinary": c.w_ord,
                "s_ordinary": c.s_ord,
                "matched": c.matched,
                "share": c.pi,
                "ordinary_mean": ordinary_mean,
            }
        )


# ============================================================================ eligibility

#: Players per top-10 list, fixed by the problem statement (not a Method_Constant). A Role with
#: fewer Eligible_Players stops the Run before ranking (Requirement 17, criterion 9).
LIST_SIZE = 10

#: The five eligibility rules (Requirement 17, criteria 2-5) in report order (criterion 11). The
#: eligibility table holds the failure flag ``fail_<rule>`` of each rule, and the report its count.
ELIGIBILITY_RULES: tuple[str, ...] = (
    "min_pressure_balls",  # criterion 2: Pressure_Ball count >= the Role's minimum (D11)
    "min_pressure_matches",  # criterion 2: Pressure_Match count >= the minimum (D18)
    "calendar_years",  # criterion 3: Deliveries in >= min_years distinct Calendar_Years
    "recent_activity",  # criterion 4: is_played in a match of a recent Calendar_Year (rule on only)
    "major_share",  # criterion 5: >= min_share of the Pressure_Balls from IPL or Full_Member_Matches
)

_FAIL_COLUMNS: tuple[str, ...] = tuple(f"fail_{rule}" for rule in ELIGIBILITY_RULES)

#: Columns of eligibility_counts: one row per player and Role (module docstring, "Eligibility").
ELIGIBILITY_COUNT_COLUMNS: tuple[str, ...] = (
    "role", "player_id", "n_deliveries", "n_pressure_balls", "n_pressure_matches", "n_major_pressure_balls",
    "major_share", "n_calendar_years", "recent_activity",
)  # fmt: skip

#: Columns of eligibility: the counts, one failure flag per rule, and ``eligible``.
ELIGIBILITY_COLUMNS: tuple[str, ...] = ELIGIBILITY_COUNT_COLUMNS + _FAIL_COLUMNS + ("eligible",)

#: Columns of eligibility_grid: one row per Role and (minimum Pressure_Ball, Pressure_Match) count.
ELIGIBILITY_GRID_COLUMNS: tuple[str, ...] = ("role", "min_pressure_balls", "min_pressure_matches", "n_eligible")

#: Columns of eligibility_report: one row per Role (Requirement 17, criterion 11).
ELIGIBILITY_REPORT_COLUMNS: tuple[str, ...] = (
    "role", "min_pressure_balls", "min_pressure_matches", "recent_rule_on", "n_players", "n_eligible",
) + _FAIL_COLUMNS  # fmt: skip

#: The CompClass codes whose Pressure_Balls count toward the share rule (criterion 5).
_MAJOR_CLASSES = np.asarray([int(CompClass.IPL), int(CompClass.FULL_MEMBER)], dtype=np.int64)


class MinBallsSelection(NamedTuple):
    """The minimum Pressure_Ball count selected for each Role (Requirement 17, criteria 7 and 8).

    ``min_balls`` maps each Role name to its selected count. ``warnings`` holds one message per
    Role for which no grid value left the target number of Eligible_Players; the caller logs them.
    """

    min_balls: dict[str, int]
    warnings: tuple[str, ...]


def _whole_number(value: Any, what: str, *, least: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{what} must be a whole number; got {value!r}")
    number = int(value)
    if number < least:
        raise ValueError(f"{what} must be {least} or more; got {number}")
    return number


def _per_role(value: Any, what: str) -> dict[str, int]:
    """A minimum count of 1 or more for each Role: one number for both Roles, or a mapping by Role name."""
    if isinstance(value, Mapping):
        unknown = [key for key in value if key not in ROLE_CODES]
        if unknown:
            raise ValueError(f"{what} names {unknown[0]!r}, which is not a Role; the Roles are {', '.join(ROLES)}")
        missing = [role for role in ROLES if role not in value]
        if missing:
            raise ValueError(f"{what} gives no value for the {missing[0]} Role")
        return {role: _whole_number(value[role], f"{what}[{role!r}]", least=1) for role in ROLES}
    number = _whole_number(value, what, least=1)
    return {role: number for role in ROLES}


def _number_list(values: Any, what: str) -> list[int]:
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Iterable):
        raise TypeError(f"{what} must be a list of whole numbers; got {values!r}")
    return [_whole_number(value, f"{what} value", least=1) for value in values]


def _grid_values(values: Any, what: str) -> list[int]:
    numbers = _number_list(values, what)
    if not numbers:
        raise ValueError(f"{what} must hold at least one value")
    if len(set(numbers)) != len(numbers):
        raise ValueError(f"{what} must hold each value once; got {numbers}")
    return sorted(numbers)


def _switch(value: Any, what: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{what} must be True or False; got {value!r}")
    return bool(value)


def _share(value: Any, what: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{what} must be a number; got {value!r}")
    share = float(value)
    if not 0.0 <= share <= 1.0:
        raise ValueError(f"{what} must be a share from 0 to 1; got {value!r}")
    return share


def _role_codes(frame: pd.DataFrame, what: str) -> np.ndarray:
    roles = frame["role"].astype("string")
    unknown = ~roles.isin(list(ROLE_CODES)).to_numpy(dtype=bool)
    if unknown.any():
        raise ValueError(f"{what} column 'role' holds {int(unknown.sum()):,} value(s) other than a Role name")
    return roles.map(ROLE_CODES).to_numpy(dtype=np.int64)


def _distinct_counts(group: np.ndarray, values: np.ndarray, size: int) -> np.ndarray:
    """The number of distinct ``values`` in each group 0, 1, ..., size - 1."""
    if len(group) == 0:
        return np.zeros(size, dtype=np.int64)
    pairs = np.unique(np.column_stack([group, values]).astype(np.int64), axis=0)
    return np.bincount(pairs[:, 0], minlength=size).astype(np.int64)


def _recent_players(xi: pd.DataFrame, matches: pd.DataFrame, years: Any) -> np.ndarray:
    """The sorted ``player_id`` values with ``is_played`` true in a match of a Calendar_Year in ``years``."""
    _require(xi, ("match_id", "player_id", "is_played"), "xi")
    _require(matches, ("match_id", "match_date"), "matches")
    wanted = np.asarray(_number_list(years, "recent_years"), dtype=np.int64)
    ids = _unique_ids(matches["match_id"], "matches column 'match_id'")
    year = pd.DatetimeIndex(_dates(matches["match_date"], "matches column 'match_date'")).year.to_numpy(np.int64)
    at = pd.Index(ids).get_indexer(_whole(xi["match_id"], "xi column 'match_id'"))
    if (at < 0).any():
        raise ValueError(
            f"{int((at < 0).sum()):,} xi row(s) have a match_id that matches does not hold, so their Calendar_Year "
            "is unknown"
        )
    played, _ = _flags(xi["is_played"], "xi column 'is_played'")  # a missing is_played reads False
    keep = np.flatnonzero(played & np.isin(year[at], wanted))
    return np.unique(_whole(xi["player_id"].iloc[keep], "xi column 'player_id' on the rows that count"))


def eligibility_counts(
    contrib: Contributions, xi: pd.DataFrame, matches: pd.DataFrame, *, recent_years: Sequence[int]
) -> pd.DataFrame:
    """The unweighted eligibility counts of every player of each Role (Requirement 17, criteria 1, 3-5).

    ``contrib``: the Contributions (either or both Roles); every count comes from its rows, the
    rated Deliveries (module docstring, "Eligibility"). ``xi``: the ``playing_eleven`` table of the
    ``load`` checkpoint (``match_id``, ``player_id``, ``is_played``). ``matches``: ``match_id`` and
    ``match_date`` (datetime) of every match of ``xi``, such as the ``matches`` table of the
    ``load`` checkpoint or the match_weights table. ``recent_years``: the Calendar_Years of the
    Recent_Activity_Rule (``data_rules.recent_years``).

    Returns ELIGIBILITY_COUNT_COLUMNS, one row per player that the Role's contributions hold
    (charge-only players included, with ``n_deliveries`` 0), sorted by Role (batter first) and
    ``player_id``, on a RangeIndex: ``role`` (str), ``player_id``, ``n_deliveries``,
    ``n_pressure_balls``, ``n_pressure_matches``, ``n_major_pressure_balls`` and
    ``n_calendar_years`` (int64), ``major_share`` (float64, NaN without a Pressure_Ball) and
    ``recent_activity`` (bool). The rows of one Role are in the order of ``Aggregator.players``.
    Raises ValueError or TypeError for a missing column, a missing or repeated ``match_id`` in
    ``matches``, an ``xi`` match that ``matches`` does not hold, or a malformed ``recent_years``.
    """
    if not isinstance(contrib, Contributions):
        raise TypeError(f"contrib must be Contributions; got {type(contrib).__name__}")
    recent = _recent_players(xi, matches, recent_years)
    frames = []
    for role in ROLES:
        rows = np.flatnonzero(contrib.role == ROLE_CODES[role])
        players, code = np.unique(contrib.player[rows], return_inverse=True)
        size = len(players)
        n = contrib.n[rows].astype(np.float64)
        pressure = contrib.pressure[rows]
        delivered = contrib.n[rows] > 0
        major = np.isin(contrib.comp_class[rows], _MAJOR_CLASSES)
        n_pressure = np.rint(np.bincount(code, np.where(pressure, n, 0.0), size)).astype(np.int64)
        n_major = np.rint(np.bincount(code, np.where(pressure & major, n, 0.0), size)).astype(np.int64)
        has_pressure = n_pressure > 0
        share = np.full(size, np.nan)
        share[has_pressure] = n_major[has_pressure] / n_pressure[has_pressure]
        in_pressure = pressure & delivered
        frames.append(
            pd.DataFrame(
                {
                    "role": pd.array([role] * size, dtype="str"),
                    "player_id": players.astype(np.int64),
                    "n_deliveries": np.rint(np.bincount(code, n, size)).astype(np.int64),
                    "n_pressure_balls": n_pressure,
                    "n_pressure_matches": _distinct_counts(code[in_pressure], contrib.match_idx[rows][in_pressure], size),
                    "n_major_pressure_balls": n_major,
                    "major_share": share,
                    "n_calendar_years": _distinct_counts(code[delivered], contrib.calendar_year[rows][delivered], size),
                    "recent_activity": np.isin(players, recent),
                },
                columns=list(ELIGIBILITY_COUNT_COLUMNS),
            )
        )
    return pd.concat(frames, ignore_index=True)


def eligibility(
    counts: pd.DataFrame,
    *,
    min_balls: int | Mapping[str, int],
    min_matches: int | Mapping[str, int],
    recent_rule: bool,
    min_years: int,
    min_share: float,
) -> pd.DataFrame:
    """The five eligibility rules applied to each player (Requirement 17, criteria 2-5).

    ``counts``: the eligibility_counts table. ``min_balls``: the minimum Pressure_Ball count, one
    whole number of 1 or more for both Roles or a mapping by Role name (``MinBallsSelection.
    min_balls``). ``min_matches``: the minimum Pressure_Match count in the same form
    (``design_choices.D18_min_matches_default``). ``recent_rule``: whether the
    Recent_Activity_Rule applies (``data_rules.recent_activity_rule``; False for the
    Activity_Comparison_Ranking). ``min_years``: the distinct Calendar_Years needed
    (``data_rules.min_calendar_years``). ``min_share``: the share of the Pressure_Balls needed from
    IPL matches and Full_Member_Matches, inclusive (``data_rules.min_major_share``).

    A player fails a rule when: ``n_pressure_balls`` < the Role's ``min_balls``;
    ``n_pressure_matches`` < the Role's ``min_matches``; ``n_calendar_years`` < ``min_years``;
    the rule is on and ``recent_activity`` is False; or the player has a Pressure_Ball and
    ``major_share`` < ``min_share``. A player without a Pressure_Ball does not fail the share rule
    (module docstring) and fails the minimum Pressure_Ball rule. ``eligible`` is True when the
    player fails no rule.

    Returns ELIGIBILITY_COLUMNS, one row per row of ``counts`` in the same order, on a
    RangeIndex: the count columns, the bool failure flags ``fail_<rule>`` (ELIGIBILITY_RULES) and
    ``eligible`` (bool). Raises ValueError or TypeError for a missing column, a value of the wrong
    type, a missing count or flag, or a threshold out of range.
    """
    _require(counts, ELIGIBILITY_COUNT_COLUMNS, "counts")
    role = _role_codes(counts, "counts")
    balls_min = _per_role(min_balls, "min_balls")
    matches_min = _per_role(min_matches, "min_matches")
    rule_on = _switch(recent_rule, "recent_rule")
    years_min = _whole_number(min_years, "min_years", least=0)
    share_min = _share(min_share, "min_share")
    values: dict[str, np.ndarray] = {}
    for name in ("n_deliveries", "n_pressure_balls", "n_pressure_matches", "n_major_pressure_balls",
                 "n_calendar_years"):  # fmt: skip
        values[name] = _whole(counts[name], f"counts column {name!r}")
        if (values[name] < 0).any():
            raise ValueError(f"counts column {name!r} must hold counts of 0 or more")
    n_pressure, n_matches = values["n_pressure_balls"], values["n_pressure_matches"]
    n_years = values["n_calendar_years"]
    share = _numbers(counts["major_share"], "counts column 'major_share'")
    active, missing = _flags(counts["recent_activity"], "counts column 'recent_activity'")
    if missing.any():
        raise ValueError(f"counts column 'recent_activity' has {int(missing.sum()):,} missing value(s)")
    has_pressure = n_pressure > 0
    if not np.isfinite(share[has_pressure]).all():
        raise ValueError("counts column 'major_share' is missing for a player with Pressure_Balls")

    need_balls = np.asarray([balls_min[name] for name in ROLES], dtype=np.int64)[role]
    need_matches = np.asarray([matches_min[name] for name in ROLES], dtype=np.int64)[role]
    fails = {
        "fail_min_pressure_balls": n_pressure < need_balls,
        "fail_min_pressure_matches": n_matches < need_matches,
        "fail_calendar_years": n_years < years_min,
        "fail_recent_activity": ~active if rule_on else np.zeros(len(counts), dtype=bool),
        "fail_major_share": has_pressure & (share < share_min),
    }
    table = counts.loc[:, list(ELIGIBILITY_COUNT_COLUMNS)].reset_index(drop=True)
    for name in _FAIL_COLUMNS:
        table[name] = fails[name]
    table["eligible"] = ~np.logical_or.reduce([fails[name] for name in _FAIL_COLUMNS])
    return table


def _eligible_per_role(table: pd.DataFrame) -> np.ndarray:
    eligible = table["eligible"].to_numpy(dtype=bool)
    return np.bincount(_role_codes(table, "table")[eligible], minlength=len(ROLES)).astype(np.int64)


def eligibility_grid(
    counts: pd.DataFrame,
    *,
    min_balls_grid: Sequence[int],
    min_matches_grid: Sequence[int],
    recent_rule: bool,
    min_years: int,
    min_share: float,
) -> pd.DataFrame:
    """The Eligible_Players of each Role at every pair of minimum counts (Requirement 17, criterion 6).

    ``min_balls_grid``: the candidate minimum Pressure_Ball counts
    (``design_choices.D11_min_balls_grid``); ``min_matches_grid``: the minimum Pressure_Match
    counts (``design_choices.D18_min_matches_grid``); each a list of distinct whole numbers of 1
    or more. ``recent_rule``, ``min_years`` and ``min_share`` are those of ``eligibility``, so the
    rules of criteria 3-5 apply at every pair. Each count is the ``eligible`` count of
    ``eligibility`` at that pair.

    Returns ELIGIBILITY_GRID_COLUMNS, one row per Role (both Roles, a Role without players
    included) and pair, sorted by Role (batter first), ``min_pressure_balls`` and
    ``min_pressure_matches``: ``role`` (str) and the int64 counts. Raises the errors of
    ``eligibility``, and ValueError or TypeError for an empty grid or a repeated or malformed value.
    """
    balls_values = _grid_values(min_balls_grid, "min_balls_grid")
    matches_values = _grid_values(min_matches_grid, "min_matches_grid")
    found = {}
    for balls in balls_values:
        for least in matches_values:
            table = eligibility(
                counts, min_balls=balls, min_matches=least, recent_rule=recent_rule, min_years=min_years,
                min_share=min_share,
            )  # fmt: skip
            found[balls, least] = _eligible_per_role(table)
    records = [
        (role, balls, least, int(found[balls, least][code]))
        for role, code in ROLE_CODES.items()
        for balls in balls_values
        for least in matches_values
    ]
    grid = pd.DataFrame.from_records(records, columns=list(ELIGIBILITY_GRID_COLUMNS))
    grid["role"] = grid["role"].astype("str")
    return grid.astype({name: np.int64 for name in ELIGIBILITY_GRID_COLUMNS[1:]})


def select_min_balls(grid: pd.DataFrame, *, min_matches: int, target: int) -> MinBallsSelection:
    """The minimum Pressure_Ball count of each Role (Requirement 17, criteria 7 and 8; D11).

    ``grid``: the eligibility_grid table. ``min_matches``: the minimum Pressure_Match count applied
    (``design_choices.D18_min_matches_default``), which must be a ``min_pressure_matches`` value
    of the grid. ``target``: the Eligible_Players the selected count must leave
    (``design_choices.D11_target_eligible``).

    For each Role, the selected count is the largest ``min_pressure_balls`` of the grid whose row at
    ``min_matches`` has at least ``target`` Eligible_Players. When none has, it is the smallest
    grid value (45 as delivered), and a warning states the Role and its Eligible_Players there.
    Returns a MinBallsSelection. Raises ValueError or TypeError for a missing column, a missing
    Role, no row at ``min_matches``, or a repeated grid row.
    """
    _require(grid, ELIGIBILITY_GRID_COLUMNS, "grid")
    least = _whole_number(min_matches, "min_matches", least=1)
    needed = _whole_number(target, "target", least=0)
    role = _role_codes(grid, "grid")
    balls = _whole(grid["min_pressure_balls"], "grid column 'min_pressure_balls'")
    matches_min = _whole(grid["min_pressure_matches"], "grid column 'min_pressure_matches'")
    n_eligible = _whole(grid["n_eligible"], "grid column 'n_eligible'")
    chosen: dict[str, int] = {}
    warnings: list[str] = []
    for name, code in ROLE_CODES.items():
        at = np.flatnonzero((role == code) & (matches_min == least))
        if len(at) == 0:
            raise ValueError(f"grid has no row for the {name} Role at the minimum Pressure_Match count {least}")
        values, counts = balls[at], n_eligible[at]
        if len(np.unique(values)) != len(values):
            raise ValueError(f"grid repeats a minimum Pressure_Ball count for the {name} Role at {least}")
        enough = counts >= needed
        if enough.any():
            chosen[name] = int(values[enough].max())
            continue
        lowest = int(np.argmin(values))
        chosen[name] = int(values[lowest])
        warnings.append(
            f"no minimum Pressure_Ball count in {sorted(values.tolist())} leaves {needed:,} Eligible_Players for "
            f"the {name} Role at the minimum Pressure_Match count {least}, so {chosen[name]} is selected; it leaves "
            f"{int(counts[lowest]):,} Eligible_Players (Requirement 17, criterion 8)"
        )
    return MinBallsSelection(min_balls=chosen, warnings=tuple(warnings))


def eligibility_report(
    counts: pd.DataFrame,
    *,
    min_balls: int | Mapping[str, int],
    min_matches: int | Mapping[str, int],
    recent_rule: bool,
    min_years: int,
    min_share: float,
) -> pd.DataFrame:
    """The eligibility summary of each Role (Requirement 17, criterion 11).

    Takes the arguments of ``eligibility`` and applies it. Returns ELIGIBILITY_REPORT_COLUMNS, one
    row per Role (batter first): ``role`` (str); ``min_pressure_balls`` and
    ``min_pressure_matches``, the Role's minimums (int64); ``recent_rule_on`` (bool);
    ``n_players``, the players with at least one Batter_Delivery or Bowler_Delivery in the Role
    (``n_deliveries`` above 0; Super_Over_Innings and excluded matches are never in the
    contributions); ``n_eligible``, the Eligible_Players; and per rule ``fail_<rule>``, the
    players of ``n_players`` who fail it (int64). A player who fails several rules counts under
    each; with the Recent_Activity_Rule off, nobody fails it. Raises the errors of ``eligibility``.
    """
    table = eligibility(
        counts, min_balls=min_balls, min_matches=min_matches, recent_rule=recent_rule, min_years=min_years,
        min_share=min_share,
    )  # fmt: skip
    balls_min = _per_role(min_balls, "min_balls")
    matches_min = _per_role(min_matches, "min_matches")
    role = _role_codes(table, "counts")
    population = table["n_deliveries"].to_numpy(dtype=np.int64) > 0
    eligible = _eligible_per_role(table)
    records = []
    for name, code in ROLE_CODES.items():
        mine = population & (role == code)
        record = {
            "role": name,
            "min_pressure_balls": balls_min[name],
            "min_pressure_matches": matches_min[name],
            "recent_rule_on": bool(recent_rule),
            "n_players": int(mine.sum()),
            "n_eligible": int(eligible[code]),
        }
        record.update({column: int((table[column].to_numpy(dtype=bool) & mine).sum()) for column in _FAIL_COLUMNS})
        records.append(record)
    report = pd.DataFrame.from_records(records, columns=list(ELIGIBILITY_REPORT_COLUMNS))
    report["role"] = report["role"].astype("str")
    report["recent_rule_on"] = report["recent_rule_on"].astype(bool)
    integers = [name for name in ELIGIBILITY_REPORT_COLUMNS if name not in ("role", "recent_rule_on")]
    return report.astype({name: np.int64 for name in integers})


def require_eligible(report: pd.DataFrame) -> None:
    """Stop the Run when a Role has fewer than LIST_SIZE Eligible_Players (Requirement 17, criterion 9).

    ``report``: the eligibility_report table at the selected minimum counts, one row per Role.
    Raises StageError (exit code 5), stating each short Role, its Eligible_Players and its minimum
    counts; ValueError or TypeError for a missing column or a Role without exactly one row.
    """
    _require(report, ("role", "min_pressure_balls", "min_pressure_matches", "n_eligible"), "report")
    role = _role_codes(report, "report")
    balls = _whole(report["min_pressure_balls"], "report column 'min_pressure_balls'")
    least = _whole(report["min_pressure_matches"], "report column 'min_pressure_matches'")
    n_eligible = _whole(report["n_eligible"], "report column 'n_eligible'")
    short = []
    for name, code in ROLE_CODES.items():
        at = np.flatnonzero(role == code)
        if len(at) != 1:
            raise ValueError(f"report must hold one row for the {name} Role; it holds {len(at)}")
        i = int(at[0])
        if n_eligible[i] < LIST_SIZE:
            short.append(
                f"the {name} Role has {int(n_eligible[i]):,} Eligible_Player(s) at the minimum Pressure_Ball count "
                f"{int(balls[i])} and the minimum Pressure_Match count {int(least[i])}"
            )
    if short:
        raise StageError(
            f"too few Eligible_Players to rank: {'; '.join(short)}. Each Role needs at least {LIST_SIZE}, so the Run "
            "stops before ranking (Requirement 17, criterion 9)"
        )


# ============================================================ shrinkage, reliability and ranking

#: A player is rankable under ``rank_role(rankable_only=True)`` with at least this many
#: Pressure_Matches and Ordinary_Balls, each drawn copy counted (Requirement 19, criterion 4).
RANKABLE_MIN_PRESSURE_MATCHES = 2
RANKABLE_MIN_ORDINARY_BALLS = 1

#: Split_Half_Reliability needs at least this many qualifying players (Requirement 18, criterion 9).
MIN_RELIABILITY_PLAYERS = 3

#: Final_Scores equal to this many decimal places count as tied (Requirement 18, criterion 13).
TIE_DECIMALS = 6
_TIE_SCALE = 10.0**TIE_DECIMALS

#: The estimates of a Role, as warnings name them: the two blended estimates, and the two parts
#: of the shrunk Clutch_Level (Requirement 18, criterion 1).
CLUTCH_LEVEL = "Clutch_Level"
UPLIFT = "Uplift"
USUAL_BASELINE = "Usual_Baseline"
PRESSURE_GAP = "Pressure_Gap"

#: The kinds of RatingWarning and the rule behind each.
WARNING_KINDS: Mapping[str, str] = {
    "min_balls_fallback": "Requirement 17, criterion 8",
    "clutch_fallback": "Requirement 18, criterion 1",
    "shrinkage_zero": "Requirement 18, criterion 3",
    "tau2_floor": "Requirement 18, criterion 5",
    "reliability_zero": "Requirement 18, criterion 9",
    "blend_default": "Requirement 18, criterion 10",
    "no_clutch": "an Eligible_Player without a Clutch_Level cannot be ranked",
}


class RatingWarning(NamedTuple):
    """One Rating_Engine warning: its kind (WARNING_KINDS), Role, estimate (None for the Role) and message."""

    kind: str
    role: str
    estimate: str | None
    message: str


def _percent(value: Any, what: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{what} must be a number; got {value!r}")
    number = float(value)
    if not (math.isfinite(number) and 0.0 <= number <= 100.0):
        raise ValueError(f"{what} must be a percentile from 0 to 100; got {value!r}")
    return number


def _scalar(value: Any, what: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{what} must be a number; got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{what} must be finite; got {value!r}")
    return number


def _pair(first: Any, second: Any, names: tuple[str, str]) -> tuple[np.ndarray, np.ndarray]:
    a, b = _numbers(first, names[0]), _numbers(second, names[1])
    if len(a) != len(b):
        raise ValueError(f"{names[0]} and {names[1]} must have one value per player; got {len(a):,} and {len(b):,}")
    return a, b


def replacement_level(clutch: Any, pct: float) -> float:
    """Replacement_Level: the ``pct``-th percentile of unshrunk Clutch_Levels (Requirement 18, criterion 6; D12).

    ``clutch``: the unshrunk Clutch_Levels (runs per over) of the players behind it, the Role's
    Eligible_Players (Requirement 17, criterion 10), all finite. ``pct``: D12
    (``design_choices.D12_replacement_pct``), from 0 to 100. Linear interpolation is numpy's
    default ``"linear"`` method: with the n values sorted, the value at position
    (n - 1) x pct / 100, interpolated between its two neighbours. Raises ValueError or TypeError
    for no value, a value that is not finite, or a percentile out of range.
    """
    values = _finite(clutch, "clutch")
    p = _percent(pct, "pct (D12)")
    if not len(values):
        raise ValueError("Replacement_Level needs the Clutch_Level of at least one player")
    return float(np.percentile(values, p, method="linear"))


class Tau2(NamedTuple):
    """tau^2, the between-player variance of one Role x estimate combination (Requirement 18, criteria 4-5).

    The estimates are the Usual_Baseline, the Pressure_Gap and Uplift, plus Clutch_Level for the
    fallback of a player without a Usual_Baseline (``rank_role``).

    ``value``: tau^2 after the zero floor, 0 or more; ``raw``: the sample variance (n - 1) of the
    estimates minus the mean squared standard error, NaN with fewer than 2 players; ``n``: the
    players with a defined standard error behind it. ``floored`` is True when ``raw`` is at
    most 0 or NaN, so ``value`` is 0 and every shrunk estimate equals its prior.
    """

    value: float
    raw: float
    n: int

    @property
    def floored(self) -> bool:
        return not self.raw > 0.0


def tau2(estimates: Any, ses: Any) -> Tau2:
    """tau^2 by the method of moments, floored at 0 (Requirement 18, criteria 4 and 5).

    ``estimates``: the unshrunk estimates of the Role's Eligible_Players; ``ses``: their
    match-clustered standard errors, NaN where undefined. Only the players whose estimate and
    standard error are both defined count. tau^2 = var(estimates, ddof=1) - mean(se^2) over
    them; a value of at most 0, or no value (fewer than 2 such players), becomes 0. Raises
    ValueError or TypeError for arrays of different lengths or the wrong type.
    """
    x, se = _pair(estimates, ses, ("estimates", "ses"))
    keep = np.isfinite(se) & np.isfinite(x)
    n = int(keep.sum())
    raw = float(np.var(x[keep], ddof=1) - np.mean(se[keep] ** 2)) if n >= 2 else math.nan
    return Tau2(value=raw if raw > 0.0 else 0.0, raw=raw, n=n)


class Shrunk(NamedTuple):
    """The result of ``shrink``, one value per player.

    ``value``: the shrunk estimates (NaN where the estimate is NaN); ``factor``: the
    Shrinkage_Factor B, from 0 to 1; ``undefined_se`` and ``zero_variance``: B was set to 0
    because the standard error is undefined, or because tau^2 and the standard error are both 0
    (Requirement 18, criterion 3).
    """

    value: np.ndarray
    factor: np.ndarray
    undefined_se: np.ndarray
    zero_variance: np.ndarray


def shrink(estimates: Any, ses: Any, tau2_value: float, prior: float = 0.0) -> Shrunk:
    """Normal-normal Shrinkage of one estimate towards its prior (Requirement 18, criteria 1-3).

    B = tau^2 / (tau^2 + SE^2) (glossary Shrinkage_Factor), and the shrunk estimate is
    prior + B x (estimate - prior). ``rank_role`` uses Replacement_Level as the prior of the
    Usual_Baseline (and of Clutch_Level in the fallback for a player without a Usual_Baseline),
    and 0 for the Pressure_Gap and Uplift (B x g and B x U). B is 0 when the SE is undefined
    (NaN) or when tau^2 and the SE are both 0, so the shrunk value is then the prior exactly. A
    NaN estimate gives a NaN shrunk value. Raises ValueError or TypeError for arrays of different
    lengths, a negative SE, or a tau^2 or prior that is not a finite number (tau^2 also 0 or more).
    """
    x, se = _pair(estimates, ses, ("estimates", "ses"))
    t = _scalar(tau2_value, "tau2_value")
    if t < 0.0:
        raise ValueError(f"tau2_value must be 0 or more (the floored tau^2); got {tau2_value!r}")
    p = _scalar(prior, "prior")
    if (se < 0.0).any():
        raise ValueError("ses must not hold a negative standard error")
    defined = np.isfinite(se)
    total = np.where(defined, t + np.where(defined, se, 0.0) ** 2, 0.0)
    ok = defined & (total > 0.0)
    factor = _divide(t, total, ok)
    return Shrunk(value=p + factor * (x - p), factor=factor, undefined_se=~defined, zero_variance=defined & ~ok)


class Reliability(NamedTuple):
    """The Split_Half_Reliability of one estimate of one Role (Requirement 18, criteria 8 and 9).

    ``rho``: the value behind the Blend_Weights, 0 or more; ``raw``: 2r / (1 + r) before the zero
    rule, NaN when r cannot be computed or is -1; ``r``: the Pearson correlation of the odd-half
    and even-half estimates, NaN when it cannot be computed; ``n``: the qualifying players;
    ``reason``: why ``rho`` was set to 0, or None.
    """

    rho: float
    raw: float
    r: float
    n: int
    reason: str | None


def reliability(odd: Any, even: Any) -> Reliability:
    """rho = 2r / (1 + r) from paired odd-half and even-half estimates (Requirement 18, criteria 8 and 9).

    ``odd`` and ``even``: the two half estimates of the qualifying players, one pair per player,
    all finite. r is the Pearson correlation, clipped to [-1, 1]. rho is set to 0, with the
    reason, when it is negative or r cannot be computed: fewer than MIN_RELIABILITY_PLAYERS
    players, zero variance in a half (every value equal, tested exactly) or r = -1. Raises
    ValueError or TypeError for arrays of different lengths, the wrong type or a value that is
    not finite.
    """
    x, y = _pair(odd, even, ("odd", "even"))
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("odd and even must hold finite estimates")
    n = len(x)
    r = raw = math.nan
    reason = None
    if n < MIN_RELIABILITY_PLAYERS:
        reason = f"r cannot be computed from {n} qualifying player(s); it needs at least {MIN_RELIABILITY_PLAYERS}"
    elif x.max() == x.min() or y.max() == y.min():
        half = "odd" if x.max() == x.min() else "even"
        reason = f"r cannot be computed: the {half}-half estimates have zero variance"
    else:
        dx, dy = x - x.mean(), y - y.mean()
        scale = math.sqrt(float(dx @ dx) * float(dy @ dy))
        if not scale > 0.0:
            reason = "r cannot be computed: a half has zero variance"
        else:
            r = min(1.0, max(-1.0, float(dx @ dy) / scale))
            if r <= -1.0:
                reason = "r = -1, so 2r / (1 + r) is undefined"
            else:
                raw = 2.0 * r / (1.0 + r)
                if raw < 0.0:
                    reason = f"it is negative ({raw:.6f}, r = {r:.6f})"
    return Reliability(rho=raw if reason is None else 0.0, raw=raw, r=r, n=n, reason=reason)


class SplitHalf(NamedTuple):
    """The Split_Half_Reliability of Clutch_Level (rho_L) and Uplift (rho_U), and the players behind them."""

    clutch: Reliability
    uplift: Reliability
    qualifying: np.ndarray  # bool per player: at least one Pressure_Ball and one Ordinary_Ball in each half


def split_half(odd: PlayerEstimates, even: PlayerEstimates, keep: Any = None) -> SplitHalf:
    """Split_Half_Reliability of Clutch_Level and Uplift (Requirement 18, criteria 7-9).

    ``odd`` and ``even``: PlayerEstimates of the same players, in the same order, from the rows
    of each player's odd-numbered and even-numbered matches (``Contributions.half``) with the
    Role's full-sample Replacement_Level set, as ``Aggregator.estimates(mult, subset,
    replacement_level=RL)`` gives them. ``keep``: a bool mask of the players that may count
    (default all), such as the pool of ``rank_role``. The qualifying players are those of
    ``keep`` with at least one Pressure_Ball and one Ordinary_Ball in each half (counts with
    the multiplicities); their unshrunk half estimates are all finite. Raises ValueError or
    TypeError for estimates that do not pair up or lack the Replacement_Level, or a bad mask.
    """
    if not (isinstance(odd, PlayerEstimates) and isinstance(even, PlayerEstimates)):
        raise TypeError("odd and even must be PlayerEstimates")
    if not np.array_equal(odd.player_id, even.player_id):
        raise ValueError("odd and even must hold the same players in the same order")
    level = odd.replacement_level
    if not (math.isfinite(level) and level == even.replacement_level):
        raise ValueError("odd and even must carry the same full-sample Replacement_Level (with_replacement_level)")
    mask = np.ones(len(odd), dtype=bool) if keep is None else np.asarray(keep)
    if mask.dtype != np.bool_ or mask.shape != (len(odd),):
        raise ValueError(f"keep must be a bool mask with one value per player ({len(odd):,})")
    q = (
        mask
        & (odd.n_pressure_balls >= 1) & (odd.n_ordinary_balls >= 1)
        & (even.n_pressure_balls >= 1) & (even.n_ordinary_balls >= 1)
    )  # fmt: skip
    return SplitHalf(
        clutch=reliability(odd.clutch[q], even.clutch[q]),
        uplift=reliability(odd.uplift[q], even.uplift[q]),
        qualifying=q,
    )


class BlendWeights(NamedTuple):
    """w_L and w_U (glossary Blend_Weights); ``default`` when both reliabilities are 0 (criterion 10)."""

    w_L: float
    w_U: float
    default: bool


def blend_weights(rho_L: float, rho_U: float) -> BlendWeights:
    """w_L = rho_L / (rho_L + rho_U), w_U = 1 - w_L; (1, 0) when both are 0 (Requirement 18, criterion 10).

    ``rho_L``, ``rho_U``: the Split_Half_Reliability of Clutch_Level and Uplift after the zero
    rule of criterion 9, each from 0 to 1. Raises ValueError or TypeError for a value that is
    not a finite number of 0 or more.
    """
    a, b = _scalar(rho_L, "rho_L"), _scalar(rho_U, "rho_U")
    if a < 0.0 or b < 0.0:
        raise ValueError(f"rho_L and rho_U must be 0 or more after the zero rule; got {rho_L!r} and {rho_U!r}")
    if a + b > 0.0:
        w = a / (a + b)
        return BlendWeights(w_L=w, w_U=1.0 - w, default=False)
    return BlendWeights(w_L=1.0, w_U=0.0, default=True)


def final_scores(shrunk_clutch: Any, shrunk_uplift: Any, w_L: float, w_U: float) -> np.ndarray:
    """Final_Score = w_L x shrunk Clutch_Level + w_U x shrunk Uplift, runs per over (Requirement 18, criteria 11-12).

    No role-standardised Z-score enters. ``w_L`` and ``w_U``: 0 or more, summing to 1 within
    1e-9 (the Blend_Weights, or a fixed blend of a Robustness_Case). NaN where an input is NaN.
    Raises ValueError or TypeError for arrays of different lengths or bad weights.
    """
    c, u = _pair(shrunk_clutch, shrunk_uplift, ("shrunk_clutch", "shrunk_uplift"))
    a, b = _scalar(w_L, "w_L"), _scalar(w_U, "w_U")
    if a < 0.0 or b < 0.0 or abs(a + b - 1.0) > 1e-9:
        raise ValueError(f"w_L and w_U must be 0 or more and sum to 1; got {w_L!r} and {w_U!r}")
    return a * c + b * u


def rank_order(final_score: Any, shrunk_clutch: Any, player_id: Any) -> np.ndarray:
    """Ranks 1, 2, 3... by Final_Score, with no shared rank (Requirement 18, criterion 13; module docstring).

    Higher Final_Score ranks first. Final_Scores equal to TIE_DECIMALS decimal places are tied:
    the sort key is rint(final_score x 10^6). A tie goes to the higher shrunk Clutch_Level (full
    precision), then to the lower ``player_id``. Returns float64 ranks in the input order, NaN
    for a player whose Final_Score is not finite (unranked). Raises ValueError or TypeError for
    arrays of different lengths, a repeated or missing ``player_id``, or the wrong type.
    """
    s, c = _pair(final_score, shrunk_clutch, ("final_score", "shrunk_clutch"))
    ids = _whole(player_id, "player_id")
    if len(ids) != len(s):
        raise ValueError(f"player_id must have one value per player ({len(s):,}); got {len(ids):,}")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("player_id must hold each player once")
    out = np.full(len(s), np.nan)
    at = np.flatnonzero(np.isfinite(s))
    order = np.lexsort((ids[at], -c[at], -np.rint(s[at] * _TIE_SCALE)))
    out[at[order]] = np.arange(1, len(at) + 1, dtype=np.float64)
    return out


def rank(df: pd.DataFrame) -> pd.DataFrame:
    """A copy of ``df`` with a ``rank`` column (Int64) by ``rank_order`` (Requirement 18, criterion 13).

    ``df`` needs ``player_id``, ``final_score`` and ``shrunk_clutch``; rows keep their order and
    index. A row without a finite ``final_score`` gets <NA>. Raises the errors of rank_order.
    """
    _require(df, ("player_id", "final_score", "shrunk_clutch"), "df")
    ranks = rank_order(df["final_score"], df["shrunk_clutch"], df["player_id"])
    out = df.copy()
    out["rank"] = pd.Series(ranks, index=df.index).astype("Int64")
    return out


class RoleData(NamedTuple):
    """The prebuilt inputs of ``rank_role`` for one Role and one fixed Eligible_Player set (``prepare_role``).

    ``eligible``: the Eligible_Players' ``player_id`` values, ascending (read-only int64).
    ``full``, ``odd`` and ``even``: Aggregators over those players' rows of the Role: all of them,
    and those of Half.ODD and Half.EVEN matches. All three keep the contributions' match
    universe, so one ``mult`` serves them, and a drawn copy of a match stays in the half of the
    original match (Requirement 19, criterion 3).
    """

    role: str
    eligible: np.ndarray
    full: Aggregator
    odd: Aggregator
    even: Aggregator


def prepare_role(contrib: Contributions, role: str, eligible: Any) -> RoleData:
    """The RoleData of ``role`` (``"batter"`` or ``"bowler"``) for the Eligible_Players ``eligible`` (player ids).

    Only the Eligible_Players' rows are indexed. Each player's estimates depend on that player's
    rows alone, so they equal those from all the Role's rows. Build it once and pass it to every
    ``rank_role`` call. Raises ValueError or TypeError for a bad Role, an empty or repeated id
    list, or contributions of the wrong type.
    """
    if not isinstance(contrib, Contributions):
        raise TypeError(f"contrib must be Contributions; got {type(contrib).__name__}")
    if role not in ROLE_CODES:
        raise ValueError(f"role must be one of {', '.join(ROLE_CODES)}; got {role!r}")
    ids = np.sort(_unique_ids(eligible, "eligible"))
    if not len(ids):
        raise ValueError("eligible must hold at least one player_id")
    mine = contrib.for_role(role)
    rows = mine.take(np.isin(mine.player, ids))
    return RoleData(
        role=role,
        eligible=_frozen(ids),
        full=Aggregator(rows, role=role),
        odd=Aggregator(rows.take(rows.half == Half.ODD), role=role),
        even=Aggregator(rows.take(rows.half == Half.EVEN), role=role),
    )


#: Scalar fields of RoleRanking.summary, in role_summary order after ``role``. ``tau2_clutch``
#: is the Clutch_Level tau^2 of the fallback for a player without a Usual_Baseline, used by the
#: ``n_fallback`` pool players only.
RANKING_SUMMARY_FIELDS: tuple[str, ...] = (
    "replacement_pct", "replacement_level", "tau2_usual", "tau2_gap", "tau2_uplift", "tau2_clutch", "tau2_usual_raw",
    "tau2_gap_raw", "tau2_uplift_raw", "tau2_clutch_raw", "n_tau2_usual", "n_tau2_gap", "n_tau2_uplift",
    "n_tau2_clutch", "n_fallback", "r_L", "r_U", "rho_L_raw", "rho_U_raw", "rho_L", "rho_U", "n_rho", "w_L", "w_U",
    "n_pool", "n_ranked",
)  # fmt: skip

#: Columns of RoleRanking.to_frame: the PLAYER_ESTIMATE_COLUMNS, then the Requirement 18 values.
RANKING_COLUMNS: tuple[str, ...] = PLAYER_ESTIMATE_COLUMNS + (
    "in_pool", "in_split_half", "clutch_odd", "clutch_even", "uplift_odd", "uplift_even", "shrunk_usual", "B_usual",
    "gap", "shrunk_gap", "B_gap", "clutch_fallback", "B_clutch", "shrunk_clutch", "shrunk_uplift", "B_uplift",
    "final_score", "rank",
)  # fmt: skip


@dataclass(frozen=True, eq=False)
class RoleRanking:
    """One Role's Shrinkage, Split_Half_Reliability, Blend_Weights, Final_Scores and ranks (``rank_role``).

    Per-player arrays follow ``player_id``, the Eligible_Players in ascending order:

    * ``estimates``: the full-sample PlayerEstimates with ``replacement_level`` set; ``odd`` and
      ``even``: the half estimates with the same Replacement_Level;
    * ``pool`` (bool): the players behind Replacement_Level, tau^2 and Split_Half_Reliability,
      and the only ones ranked; ``qualifying`` (bool): the pool players behind the
      Split_Half_Reliability (Requirement 18, criterion 8);
    * ``gap``: the unshrunk Pressure_Gap g = Clutch_Level - Usual_Baseline, NaN without a
      Usual_Baseline (an estimate, like ``estimates``, so not limited to the pool);
    * ``fallback`` (bool): the pool players without a Usual_Baseline, whose shrunk Clutch_Level
      uses the Clutch_Level fallback;
    * ``shrunk_usual`` (O*), ``shrunk_gap`` (g*), ``shrunk_clutch`` (C* = O* + g*, or the
      fallback), ``shrunk_uplift`` and ``final_score``: runs per over, NaN outside the pool;
      ``b_usual``, ``b_gap`` and ``b_uplift``: their Shrinkage_Factors, NaN outside the pool and,
      for the first two, without a Usual_Baseline; ``b_clutch``: the fallback Shrinkage_Factor of
      Clutch_Level, NaN except for ``fallback`` players;
    * ``rank`` (float64): 1, 2, 3... with no shared rank, NaN for an unranked player.

    Role values: ``replacement_pct`` (D12) and ``replacement_level``; ``tau2_usual``,
    ``tau2_gap``, ``tau2_uplift`` and ``tau2_clutch`` (the fallback) (Tau2);
    ``reliability_clutch`` (rho_L) and ``reliability_uplift`` (rho_U) (Reliability); ``w_L``,
    ``w_U`` and ``blend_default``; ``rankable_only``; ``warnings``.

    From ``rank_role(fixed=RoleHyper)``, the Role values are the RoleHyper's and nothing is
    estimated: each Tau2 has ``raw`` equal to ``value`` and ``n`` 0, each Reliability has ``raw``
    equal to ``rho``, ``r`` NaN and ``n`` 0, and ``blend_default`` is True when both rho are 0.
    ``pool`` is then only the ranked players, ``odd`` and ``even`` are blank (zero counts, NaN
    estimates) and ``qualifying`` is all False.
    """

    role: str
    player_id: np.ndarray
    estimates: PlayerEstimates
    odd: PlayerEstimates
    even: PlayerEstimates
    pool: np.ndarray
    qualifying: np.ndarray
    gap: np.ndarray
    fallback: np.ndarray
    shrunk_usual: np.ndarray
    shrunk_gap: np.ndarray
    shrunk_clutch: np.ndarray
    shrunk_uplift: np.ndarray
    b_usual: np.ndarray
    b_gap: np.ndarray
    b_clutch: np.ndarray
    b_uplift: np.ndarray
    final_score: np.ndarray
    rank: np.ndarray
    replacement_pct: float
    replacement_level: float
    tau2_usual: Tau2
    tau2_gap: Tau2
    tau2_uplift: Tau2
    tau2_clutch: Tau2
    reliability_clutch: Reliability
    reliability_uplift: Reliability
    w_L: float
    w_U: float
    blend_default: bool
    rankable_only: bool
    warnings: tuple[RatingWarning, ...]

    def __len__(self) -> int:
        return len(self.player_id)

    def __repr__(self) -> str:
        return (
            f"RoleRanking({self.role}: {int(self.ranked.sum()):,} of {len(self):,} Eligible_Players ranked; "
            f"Replacement_Level {self.replacement_level:.4f}, w_L {self.w_L:.4f}, {len(self.warnings)} warning(s))"
        )

    @property
    def ranked(self) -> np.ndarray:
        """True for each ranked player."""
        return np.isfinite(self.rank)

    @property
    def rho_L(self) -> float:
        return self.reliability_clutch.rho

    @property
    def rho_U(self) -> float:
        return self.reliability_uplift.rho

    def summary(self) -> dict[str, Any]:
        """The Role values as a dict: ``role`` and RANKING_SUMMARY_FIELDS (``n_rho`` serves both rho)."""
        c, u = self.reliability_clutch, self.reliability_uplift
        taus = {"usual": self.tau2_usual, "gap": self.tau2_gap, "uplift": self.tau2_uplift, "clutch": self.tau2_clutch}
        return {
            "role": self.role,
            "replacement_pct": self.replacement_pct,
            "replacement_level": self.replacement_level,
            **{f"tau2_{key}": t.value for key, t in taus.items()},
            **{f"tau2_{key}_raw": t.raw for key, t in taus.items()},
            **{f"n_tau2_{key}": t.n for key, t in taus.items()},
            "n_fallback": int(self.fallback.sum()),
            "r_L": c.r,
            "r_U": u.r,
            "rho_L_raw": c.raw,
            "rho_U_raw": u.raw,
            "rho_L": c.rho,
            "rho_U": u.rho,
            "n_rho": c.n,
            "w_L": self.w_L,
            "w_U": self.w_U,
            "n_pool": int(self.pool.sum()),
            "n_ranked": int(self.ranked.sum()),
        }

    def to_frame(self) -> pd.DataFrame:
        """One row per Eligible_Player with RANKING_COLUMNS; ``rank`` is Int64 (<NA> when unranked)."""
        frame = self.estimates.to_frame()
        extra = {
            "in_pool": self.pool,
            "in_split_half": self.qualifying,
            "clutch_odd": self.odd.clutch,
            "clutch_even": self.even.clutch,
            "uplift_odd": self.odd.uplift,
            "uplift_even": self.even.uplift,
            "shrunk_usual": self.shrunk_usual,
            "B_usual": self.b_usual,
            "gap": self.gap,
            "shrunk_gap": self.shrunk_gap,
            "B_gap": self.b_gap,
            "clutch_fallback": self.fallback,
            "B_clutch": self.b_clutch,
            "shrunk_clutch": self.shrunk_clutch,
            "shrunk_uplift": self.shrunk_uplift,
            "B_uplift": self.b_uplift,
            "final_score": self.final_score,
        }
        for name, values in extra.items():
            frame[name] = values
        frame["rank"] = pd.Series(self.rank).astype("Int64")
        return frame.loc[:, list(RANKING_COLUMNS)]


#: Fields of RoleHyper, in order. Each is also a RANKING_SUMMARY_FIELDS name, so a
#: ``role_summary`` row or RoleRanking.summary() holds them all.
ROLE_HYPER_FIELDS: tuple[str, ...] = (
    "replacement_level", "tau2_usual", "tau2_gap", "tau2_uplift", "tau2_clutch", "rho_L", "rho_U", "w_L", "w_U",
)  # fmt: skip


@dataclass(frozen=True)
class RoleHyper:
    """The Role-level values that ``rank_role(fixed=...)`` holds instead of estimating them.

    A Bootstrap_Resample holds them at the values of the full-data ranking it resamples, such as
    the Main_Ranking (Requirement 19, criterion 2): ``replacement_level`` (runs per over); the
    tau^2 values after the zero floor, ``tau2_usual``, ``tau2_gap``, ``tau2_uplift`` and
    ``tau2_clutch`` (the Clutch_Level fallback); ``rho_L`` and ``rho_U``, the
    Split_Half_Reliability values after the zero rule; and the Blend_Weights ``w_L`` and ``w_U``.
    Build it with ``from_ranking`` (that ranking's RoleRanking) or ``from_summary`` (its
    ``role_summary`` row); both give the same values bit for bit, and ``rank_role`` with them
    and mult all ones reproduces the ranking. Values are stored as Python floats. Raises
    ValueError or TypeError for a value that is not a finite number, a negative tau^2, a rho
    outside [0, 1], or Blend_Weights that are negative or do not sum to 1 within 1e-9.
    """

    replacement_level: float
    tau2_usual: float
    tau2_gap: float
    tau2_uplift: float
    tau2_clutch: float
    rho_L: float
    rho_U: float
    w_L: float
    w_U: float

    def __post_init__(self) -> None:
        for name in ROLE_HYPER_FIELDS:
            object.__setattr__(self, name, _scalar(getattr(self, name), f"RoleHyper.{name}"))
        for name in ROLE_HYPER_FIELDS[1:5]:
            if getattr(self, name) < 0.0:
                raise ValueError(f"RoleHyper.{name} must be 0 or more (the floored tau^2); got {getattr(self, name)!r}")
        for name in ("rho_L", "rho_U"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"RoleHyper.{name} must lie in [0, 1] (after the zero rule); got {getattr(self, name)!r}")
        if self.w_L < 0.0 or self.w_U < 0.0 or abs(self.w_L + self.w_U - 1.0) > 1e-9:
            raise ValueError(f"RoleHyper w_L and w_U must be 0 or more and sum to 1; got {self.w_L!r} and {self.w_U!r}")

    @classmethod
    def from_ranking(cls, ranking: RoleRanking) -> RoleHyper:
        """The Role-level values of a RoleRanking, such as the Main_Ranking's from ``rank_role``."""
        if not isinstance(ranking, RoleRanking):
            raise TypeError(f"ranking must be a RoleRanking; got {type(ranking).__name__}")
        return cls.from_summary(ranking.summary())

    @classmethod
    def from_summary(cls, summary: pd.DataFrame | Mapping[str, Any], role: str | None = None) -> RoleHyper:
        """The Role-level values in a ``role_summary`` table (one row per Role) or in one record of it.

        ``summary``: the ``role_summary`` table of the ``rating`` checkpoint, with ``role`` naming
        the row; or one record holding ROLE_HYPER_FIELDS, such as ``RoleRanking.summary()`` or a
        row as a dict (``role`` then optional, and checked against the record's ``role`` when
        both are given). Raises ValueError for a missing field or a Role without exactly one row.
        """
        if isinstance(summary, pd.DataFrame):
            if role is None:
                raise TypeError("from_summary on a role_summary table needs role=")
            _require(summary, ("role",) + ROLE_HYPER_FIELDS, "summary")
            rows = summary.loc[summary["role"].astype("string") == role]
            if len(rows) != 1:
                raise ValueError(f"summary must hold exactly one row of the {role!r} Role; it holds {len(rows)}")
            record: Mapping[str, Any] = rows.iloc[0].to_dict()
        elif isinstance(summary, Mapping):
            record = summary
            missing = [name for name in ROLE_HYPER_FIELDS if name not in record]
            if missing:
                raise ValueError(f"summary lacks the field(s) {', '.join(missing)}")
            if role is not None and "role" in record and record["role"] != role:
                raise ValueError(f"summary is the {record['role']!r} Role's, not the {role!r} Role's")
        else:
            raise TypeError(f"summary must be a role_summary table or a mapping; got {type(summary).__name__}")
        return cls(**{name: record[name] for name in ROLE_HYPER_FIELDS})


def _blank_estimates(role: str, ids: np.ndarray, replacement_level: float) -> PlayerEstimates:
    """PlayerEstimates of ``ids`` with zero counts and sums and NaN estimates (the halves ``fixed`` skips).

    The values are those that ``Aggregator.estimates`` gives a player whom the Role does not hold.
    """
    n = len(ids)
    values: dict[str, np.ndarray] = {}
    for name in _ESTIMATE_ARRAYS:
        if name in ("player_id", "uplift", "se_uplift", "g_uplift"):
            continue
        if name in _NAN_WHEN_ABSENT:
            values[name] = np.full(n, np.nan)
        else:
            values[name] = np.zeros(n, dtype=np.int64 if name.startswith("n_") else np.float64)
    blank = np.full(n, np.nan)
    return PlayerEstimates(
        role=role, player_id=np.array(ids, dtype=np.int64), uplift=blank, se_uplift=blank.copy(),
        g_uplift=blank.copy(), **values,
    ).with_replacement_level(replacement_level)  # fmt: skip


def _id_list(ids: np.ndarray, limit: int = 20) -> str:
    shown = ", ".join(str(int(value)) for value in ids[:limit])
    return shown + (f" and {len(ids) - limit:,} more" if len(ids) > limit else "")


def _role_data(source: Any, eligible: Any, role: str | None) -> RoleData:
    if isinstance(source, RoleData):
        if eligible is not None:
            raise ValueError("a RoleData fixes its Eligible_Player set; pass eligible to prepare_role instead")
        if role is not None and role != source.role:
            raise ValueError(f"role {role!r} does not match the RoleData's Role {source.role!r}")
        return source
    if isinstance(source, Contributions):
        if role is None or eligible is None:
            raise TypeError("rank_role on Contributions needs role= and eligible=")
        return prepare_role(source, role, eligible)
    raise TypeError(f"source must be a RoleData (prepare_role) or Contributions; got {type(source).__name__}")


def rank_role(
    source: RoleData | Contributions,
    mult: Any = None,
    *,
    eligible: Any = None,
    role: str | None = None,
    replacement_pct: float,
    rankable_only: bool = False,
    fixed: RoleHyper | None = None,
) -> RoleRanking:
    """The Requirement 18 chain for one Role under the match multiplicities ``mult`` (module docstring).

    ``source``: a RoleData from ``prepare_role`` (the fast path, for repeated calls), or
    Contributions together with ``role`` and ``eligible`` (the Eligible_Players' ids; the RoleData
    is then built for this call). ``mult``: one whole number of 0 or more per match of the
    contributions' match universe (None: all ones, the Main_Ranking). ``replacement_pct``: D12.
    ``rankable_only``: rank only the Eligible_Players with at least RANKABLE_MIN_PRESSURE_MATCHES
    Pressure_Matches and RANKABLE_MIN_ORDINARY_BALLS Ordinary_Ball under ``mult`` (Requirement 19,
    criterion 4), and, without ``fixed``, compute Replacement_Level, tau^2 and
    Split_Half_Reliability from them only; the others are unranked, silently. Without it, every
    Eligible_Player with a Clutch_Level counts.

    ``fixed``: a RoleHyper (Requirement 19, criterion 2). Replacement_Level, the four tau^2, both
    Split_Half_Reliability values and the Blend_Weights are then its values, not estimates: the
    players' estimates and SEs under ``mult``, their Shrinkage_Factors (from the fixed tau^2 and
    each player's SE), the shrunk values, Final_Score and ranks are computed as usual, and the
    pool only decides who is ranked. The odd and even halves are not computed, and no
    ``tau2_floor``, ``reliability_zero`` or ``blend_default`` warning is raised (RoleRanking
    describes the fields). A resample in which no Eligible_Player can be ranked then leaves
    every player unranked. None (the default) estimates them all.

    Returns a RoleRanking whose ``warnings`` the caller logs. Raises ValueError or TypeError for
    bad inputs, and, without ``fixed``, ValueError when no Eligible_Player can be ranked.
    """
    data = _role_data(source, eligible, role)
    pct = _percent(replacement_pct, "replacement_pct (D12)")
    only = _switch(rankable_only, "rankable_only")
    if fixed is not None and not isinstance(fixed, RoleHyper):
        raise TypeError(f"fixed must be a RoleHyper or None; got {type(fixed).__name__}")
    name, ids = data.role, data.eligible
    notes: list[RatingWarning] = []

    # 1. The full-sample estimates and the pool (Requirement 17, criterion 10; Requirement 19, criterion 4).
    est = data.full.estimates(mult, ids)
    pool = np.isfinite(est.clutch)
    if only:
        pool &= (est.n_pressure_matches >= RANKABLE_MIN_PRESSURE_MATCHES) & (
            est.n_ordinary_balls >= RANKABLE_MIN_ORDINARY_BALLS
        )
    elif not pool.all():
        notes.append(RatingWarning(
            "no_clutch", name, CLUTCH_LEVEL,
            f"the {name} Role's Eligible_Player(s) {_id_list(ids[~pool])} have no Pressure_Ball under these "
            "multiplicities, so no Clutch_Level: they are left out of Replacement_Level, tau^2 and "
            "Split_Half_Reliability, and unranked",
        ))  # fmt: skip
    if not pool.any() and fixed is None:  # with fixed values nothing is estimated, so everyone is just unranked
        raise ValueError(f"no Eligible_Player of the {name} Role can be ranked under these multiplicities")

    # 2. Replacement_Level (criterion 6), or the fixed one, then Uplift and its SE against it.
    level = replacement_level(est.clutch[pool], pct) if fixed is None else fixed.replacement_level
    est = est.with_replacement_level(level)

    # 3. tau^2 (criteria 4 and 5; or the fixed values) and Shrinkage (criteria 1-3), over the pool.
    # The shrunk Clutch_Level is the shrunk Usual_Baseline plus the shrunk Pressure_Gap; a pool
    # player without a Usual_Baseline falls back to shrinking Clutch_Level towards Replacement_Level.
    with_usual = pool & np.isfinite(est.usual)
    fallback = pool & ~with_usual
    gap = est.clutch - est.usual  # the Pressure_Gap, NaN without a Usual_Baseline
    if fixed is None:
        t_o = tau2(est.usual[pool], est.se_usual[pool])
        t_g = tau2(gap[pool], est.se_clutch_minus_usual[pool])
        t_u = tau2(est.uplift[pool], est.se_uplift[pool])
        t_c = tau2(est.clutch[pool], est.se_clutch[pool])
    else:
        t_o, t_g, t_u, t_c = (
            Tau2(value=value, raw=value, n=0)
            for value in (fixed.tau2_usual, fixed.tau2_gap, fixed.tau2_uplift, fixed.tau2_clutch)
        )
    s_o = shrink(est.usual, est.se_usual, t_o.value, level)
    s_g = shrink(gap, est.se_clutch_minus_usual, t_g.value, 0.0)
    s_u = shrink(est.uplift, est.se_uplift, t_u.value, 0.0)
    s_c = shrink(est.clutch, est.se_clutch, t_c.value, level)
    if fallback.any():
        notes.append(RatingWarning(
            "clutch_fallback", name, CLUTCH_LEVEL,
            f"the {name} Role's Eligible_Player(s) {_id_list(ids[fallback])} have no Usual_Baseline (no "
            "Matched_Cell) under these multiplicities, so their shrunk Clutch_Level is Replacement_Level + B x "
            f"(Clutch_Level - Replacement_Level) with the Role's Clutch_Level tau^2 ({WARNING_KINDS['clutch_fallback']})",
        ))  # fmt: skip
    parts = (
        (USUAL_BASELINE, t_o, s_o, "Replacement_Level", with_usual),
        (PRESSURE_GAP, t_g, s_g, "0", with_usual),
        (UPLIFT, t_u, s_u, "0", pool),
        (CLUTCH_LEVEL, t_c, s_c, "Replacement_Level", fallback),
    )  # fmt: skip
    for label, t, s, prior, applies in parts:
        if not applies.any():  # the fallback Clutch_Level tau^2 when every pool player has a Usual_Baseline
            continue
        if fixed is None and t.floored:  # a fixed tau^2 is not estimated here, so it has no floor to report
            why = (
                f"is {t.raw:.6g}, at most 0" if math.isfinite(t.raw)
                else f"cannot be estimated from {t.n} player(s) with a defined standard error"
            )  # fmt: skip
            notes.append(RatingWarning(
                "tau2_floor", name, label,
                f"tau^2 of the {name} Role's {label} {why}, so it is set to 0 and every shrunk {label} equals its "
                f"prior ({prior}): full shrinkage ({WARNING_KINDS['tau2_floor']})",
            ))  # fmt: skip
        for flags, why in ((s.undefined_se, "its match-clustered standard error is undefined"),
                           (s.zero_variance, "tau^2 and its match-clustered standard error are both 0")):  # fmt: skip
            hit = flags & applies
            if hit.any():
                notes.append(RatingWarning(
                    "shrinkage_zero", name, label,
                    f"the Shrinkage_Factor of the {label} of {name} player_id(s) {_id_list(ids[hit])} is set to 0 "
                    f"because {why} ({WARNING_KINDS['shrinkage_zero']})",
                ))  # fmt: skip
    nan = np.full(len(ids), np.nan)
    shrunk_o, shrunk_g = np.where(with_usual, s_o.value, nan), np.where(with_usual, s_g.value, nan)
    shrunk_c = np.where(with_usual, shrunk_o + shrunk_g, np.where(fallback, s_c.value, nan))
    shrunk_u = np.where(pool, s_u.value, nan)

    # 4. Split_Half_Reliability (criteria 7-9) and the Blend_Weights (criterion 10; Requirement 2, criterion 5),
    # or the fixed values, for which the halves are not needed.
    if fixed is None:
        odd = data.odd.estimates(mult, ids, replacement_level=level)
        even = data.even.estimates(mult, ids, replacement_level=level)
        halves = split_half(odd, even, pool)
        for label, rel, symbol in ((CLUTCH_LEVEL, halves.clutch, "rho_L"), (UPLIFT, halves.uplift, "rho_U")):
            if rel.reason is not None:
                notes.append(RatingWarning(
                    "reliability_zero", name, label,
                    f"the Split_Half_Reliability {symbol} of the {name} Role's {label} is set to 0: {rel.reason} "
                    f"({rel.n:,} qualifying players; {WARNING_KINDS['reliability_zero']})",
                ))  # fmt: skip
        blend = blend_weights(halves.clutch.rho, halves.uplift.rho)
        if blend.default:
            notes.append(RatingWarning(
                "blend_default", name, None,
                f"both Split_Half_Reliability values of the {name} Role are 0, so w_L = 1 and w_U = 0 "
                f"({WARNING_KINDS['blend_default']})",
            ))  # fmt: skip
        qualifying, rel_clutch, rel_uplift = halves.qualifying, halves.clutch, halves.uplift
    else:
        odd = even = _blank_estimates(name, ids, level)
        qualifying = np.zeros(len(ids), dtype=bool)
        rel_clutch = Reliability(rho=fixed.rho_L, raw=fixed.rho_L, r=math.nan, n=0, reason=None)
        rel_uplift = Reliability(rho=fixed.rho_U, raw=fixed.rho_U, r=math.nan, n=0, reason=None)
        blend = BlendWeights(w_L=fixed.w_L, w_U=fixed.w_U, default=not fixed.rho_L + fixed.rho_U > 0.0)

    # 5. Final_Score (criteria 11 and 12) and the ranks (criterion 13).
    score = final_scores(shrunk_c, shrunk_u, blend.w_L, blend.w_U)
    return RoleRanking(
        role=name, player_id=ids, estimates=est, odd=odd, even=even, pool=pool, qualifying=qualifying,
        gap=gap, fallback=fallback, shrunk_usual=shrunk_o, shrunk_gap=shrunk_g, shrunk_clutch=shrunk_c,
        shrunk_uplift=shrunk_u, b_usual=np.where(with_usual, s_o.factor, nan),
        b_gap=np.where(with_usual, s_g.factor, nan), b_clutch=np.where(fallback, s_c.factor, nan),
        b_uplift=np.where(pool, s_u.factor, nan), final_score=score, rank=rank_order(score, shrunk_c, ids),
        replacement_pct=pct, replacement_level=level, tau2_usual=t_o, tau2_gap=t_g, tau2_uplift=t_u, tau2_clutch=t_c,
        reliability_clutch=rel_clutch, reliability_uplift=rel_uplift, w_L=blend.w_L, w_U=blend.w_U,
        blend_default=blend.default, rankable_only=only, warnings=tuple(notes),
    )  # fmt: skip


# ============================================================================ the ``rating`` stage

#: Name of the Pipeline_Stage that runs the Rating_Engine, and of its checkpoint.
STAGE_NAME = "rating"

#: Checkpoint tables of the stage (design, Data Models; module docstring).
CONTRIBUTIONS_TABLE = "contributions"
MATCH_UNIVERSE_TABLE = "match_universe"
ELIGIBILITY_GRID_TABLE = "eligibility_grid"
ELIGIBILITY_REPORT_TABLE = "eligibility_report"
PLAYER_ESTIMATES_TABLE = "player_estimates"
ROLE_SUMMARY_TABLE = "role_summary"
WARNINGS_TABLE = "warnings"

#: Folder under the outputs folder that holds the Rating_Engine's files, and the files.
RATING_FOLDER = "rating"
PLAYER_ESTIMATES_FILES: Mapping[str, str] = {
    BATTER_ROLE: "player_estimates_batter.csv", BOWLER_ROLE: "player_estimates_bowler.csv",
}  # fmt: skip
TOP10_FILES: Mapping[str, str] = {
    BATTER_ROLE: "provisional_top10_batter.csv", BOWLER_ROLE: "provisional_top10_bowler.csv",
}  # fmt: skip
ROLE_SUMMARY_FILE = "role_summary.csv"
ELIGIBILITY_GRID_FILE = "eligibility_grid.csv"
ELIGIBILITY_REPORT_FILE = "eligibility_report.csv"
WARNINGS_FILE = "warnings.csv"

#: Columns of the match_universe table: MATCH_WEIGHT_COLUMNS (the contributions' match universe,
#: sorted by ``match_id``), ``resampled`` (the match has a Delivery outside Super_Over_Innings, so
#: Bootstrap_Resamples draw from it; glossary) and ``rated`` (it holds a contribution row).
MATCH_UNIVERSE_COLUMNS: tuple[str, ...] = MATCH_WEIGHT_COLUMNS + ("resampled", "rated")

_ELIGIBILITY_EXTRA: tuple[str, ...] = (
    "n_deliveries", "n_major_pressure_balls", "major_share", "n_calendar_years", "recent_activity",
) + _FAIL_COLUMNS  # fmt: skip

#: Columns of the player_estimates table: one row per player and Role with a contribution row
#: (design, Data Models, plus the eligibility counts and the half estimates). ``gap`` is the
#: unshrunk Pressure_Gap, Clutch_Level - Usual_Baseline, and its SE is ``se_clutch_minus_usual``.
#: The Requirement 18 columns are NaN (``rank`` <NA>, the flags False) for a player who is not an
#: Eligible_Player. ``B_clutch`` is filled only for a ``clutch_fallback`` player, and ``B_usual``,
#: ``shrunk_usual``, ``B_gap`` and ``shrunk_gap`` only for a pool player with a Usual_Baseline.
PLAYER_TABLE_COLUMNS: tuple[str, ...] = (
    "role", "player_id", "eligible", "rank", "final_score", "clutch", "shrunk_clutch", "se_clutch", "usual",
    "shrunk_usual", "B_usual", "se_usual", "gap", "shrunk_gap", "B_gap", "se_clutch_minus_usual", "clutch_fallback",
    "B_clutch", "uplift", "shrunk_uplift", "B_uplift", "se_uplift", "replacement_level", "hit_rate", "worst_quartile",
    "unmatched_share", "n_pressure_balls", "n_ordinary_balls", "n_pressure_matches", "n_matches", "n_matched_cells",
    "w_pressure", "s_pressure", "g_clutch", "g_usual", "g_clutch_minus_usual", "g_uplift", "in_pool",
    "in_split_half", "clutch_odd", "clutch_even", "uplift_odd", "uplift_even",
) + _ELIGIBILITY_EXTRA  # fmt: skip

#: Columns of the role_summary table: the eligibility report, the Requirement 18 values and the
#: number of warnings of the Role.
ROLE_SUMMARY_COLUMNS: tuple[str, ...] = ELIGIBILITY_REPORT_COLUMNS + RANKING_SUMMARY_FIELDS + ("n_warnings",)

#: Columns of each provisional top-10 file (Requirement 18, criterion 15): the rows, then the
#: Role values repeated on each row. ``tau2_clutch`` (the fallback Clutch_Level tau^2) is filled
#: only when a listed player is a ``clutch_fallback`` player, so the table gives the tau^2 values
#: that apply to its rows.
TOP10_COLUMNS: tuple[str, ...] = (
    "rank", "player_id", "player_name", "final_score", "clutch", "shrunk_clutch", "usual", "shrunk_usual", "gap",
    "shrunk_gap", "uplift", "shrunk_uplift", "se_clutch", "se_usual", "se_clutch_minus_usual", "se_uplift",
    "hit_rate", "n_pressure_balls", "n_pressure_matches", "clutch_fallback", "rho_L", "rho_U", "w_L", "w_U",
    "tau2_usual", "tau2_gap", "tau2_uplift", "tau2_clutch", "replacement_level",
)  # fmt: skip

#: The Role values that every top-10 row repeats (``tau2_clutch`` is added only when it applies).
_TOP10_ROLE_VALUES: tuple[str, ...] = (
    "rho_L", "rho_U", "w_L", "w_U", "tau2_usual", "tau2_gap", "tau2_uplift", "replacement_level",
)  # fmt: skip

#: Columns of the warnings table and file.
WARNING_COLUMNS: tuple[str, ...] = ("role", "kind", "estimate", "message")

#: Data-derived constants of the stage, one per Role: ``rating.<quantity>.<role>``. ``tau2_clutch``
#: is the fallback Clutch_Level tau^2.
CONSTANT_QUANTITIES: tuple[str, ...] = (
    "min_pressure_balls", "n_eligible", "replacement_level", "tau2_usual", "tau2_gap", "tau2_uplift", "tau2_clutch",
    "rho_L", "rho_U", "w_L", "w_U",
)  # fmt: skip

_COMPONENT = "Rating_Engine"
_D9 = "design_choices.D9_half_life_years"
_D10 = "design_choices.D10_knockout_weight"
_D11_GRID = "design_choices.D11_min_balls_grid"
_D11_TARGET = "design_choices.D11_target_eligible"
_D12 = "design_choices.D12_replacement_pct"
_D13 = "design_choices.D13_cells"
_D18_GRID = "design_choices.D18_min_matches_grid"
_D18_DEFAULT = "design_choices.D18_min_matches_default"
_FULL_MEMBERS = "data_rules.full_members"
_RECENT_RULE = "data_rules.recent_activity_rule"
_RECENT_YEARS = "data_rules.recent_years"
_MIN_SHARE = "data_rules.min_major_share"
_MIN_YEARS = "data_rules.min_calendar_years"
#: The D13 cells of the ``value`` table's ``cell`` column; the alternative is a Robustness_Case (21.4).
_D13_CELLS = "innings_phase"
_LOAD_STAGE = "load"
_NAME_COLUMNS: Mapping[str, tuple[str, str]] = {
    BATTER_ROLE: ("striker_id", "striker_name"), BOWLER_ROLE: ("bowler_id", "bowler_name"),
}  # fmt: skip
_FLOAT_FORMAT = "%.6f"


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


def _match_universe(weights: pd.DataFrame, balls: pd.DataFrame, contrib: Contributions) -> pd.DataFrame:
    """The match_weights table with ``resampled`` and ``rated`` (MATCH_UNIVERSE_COLUMNS)."""
    _require(balls, ("match_id", "is_super_over"), "balls")
    ids = weights["match_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(ids, contrib.match_ids):
        raise ValueError("the contributions' match universe is not the match_weights table")
    super_over, missing = _flags(balls["is_super_over"], "balls column 'is_super_over'")
    if missing.any():
        raise ValueError(f"balls column 'is_super_over' has {int(missing.sum()):,} missing value(s)")
    regular = np.unique(_whole(balls["match_id"], "balls column 'match_id'")[~super_over])
    rated = np.zeros(len(ids), dtype=bool)
    rated[np.unique(contrib.match_idx)] = True
    universe = weights.loc[:, list(MATCH_WEIGHT_COLUMNS)].copy()
    universe["resampled"] = np.isin(ids, regular)
    universe["rated"] = rated
    return universe


def _player_names(deliveries: pd.DataFrame) -> dict[str, pd.Series]:
    """Per Role, the most frequent ``striker_name`` or ``bowler_name`` of each player id (ties: first by name)."""
    names = {}
    for role, (id_column, name_column) in _NAME_COLUMNS.items():
        _require(deliveries, (id_column, name_column), "ball_by_ball")
        pairs = deliveries[[id_column, name_column]].dropna()
        counted = pairs.groupby([id_column, name_column]).size().reset_index(name="n")
        counted = counted.sort_values([id_column, "n", name_column], ascending=[True, False, True], kind="stable")
        first = counted.drop_duplicates(id_column)
        names[role] = pd.Series(
            first[name_column].astype(str).to_numpy(dtype=object), index=first[id_column].to_numpy(dtype=np.int64)
        )
    return names


def _player_table(everyone: PlayerEstimates, rules: pd.DataFrame, ranking: RoleRanking) -> pd.DataFrame:
    """One Role's rows of player_estimates (PLAYER_TABLE_COLUMNS)."""
    frame = everyone.to_frame()
    ids = frame["player_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(ids, rules["player_id"].to_numpy(dtype=np.int64)):
        raise ValueError(f"the eligibility rows of the {ranking.role} Role do not follow its estimates' players")
    at = np.searchsorted(ids, ranking.player_id)
    if not np.array_equal(ids[np.minimum(at, len(ids) - 1)], ranking.player_id):
        raise ValueError(f"an Eligible_Player of the {ranking.role} Role has no contribution row")

    def spread(values: np.ndarray, fill: Any = np.nan) -> np.ndarray:
        out = np.full(len(ids), fill, dtype=np.asarray(values).dtype)
        out[at] = values
        return out

    columns = {
        "eligible": rules["eligible"].to_numpy(dtype=bool),
        "final_score": spread(ranking.final_score),
        "shrunk_clutch": spread(ranking.shrunk_clutch),
        "shrunk_usual": spread(ranking.shrunk_usual),
        "B_usual": spread(ranking.b_usual),
        "gap": frame["clutch"].to_numpy(dtype=np.float64) - frame["usual"].to_numpy(dtype=np.float64),
        "shrunk_gap": spread(ranking.shrunk_gap),
        "B_gap": spread(ranking.b_gap),
        "clutch_fallback": spread(ranking.fallback, False),
        "B_clutch": spread(ranking.b_clutch),
        "shrunk_uplift": spread(ranking.shrunk_uplift),
        "B_uplift": spread(ranking.b_uplift),
        "in_pool": spread(ranking.pool, False),
        "in_split_half": spread(ranking.qualifying, False),
        "clutch_odd": spread(ranking.odd.clutch),
        "clutch_even": spread(ranking.even.clutch),
        "uplift_odd": spread(ranking.odd.uplift),
        "uplift_even": spread(ranking.even.uplift),
    }
    for name, values in columns.items():
        frame[name] = values
    frame["rank"] = pd.Series(spread(ranking.rank)).astype("Int64")
    for name in _ELIGIBILITY_EXTRA:
        frame[name] = rules[name].to_numpy()
    return frame.loc[:, list(PLAYER_TABLE_COLUMNS)]


def _named(table: pd.DataFrame, names: pd.Series) -> pd.DataFrame:
    """``table`` with ``player_name`` after ``player_id`` (provisional names; empty when unknown)."""
    out = table.copy()
    looked_up = names.reindex(out["player_id"].to_numpy(dtype=np.int64)).to_numpy(dtype=object)
    out.insert(out.columns.get_loc("player_id") + 1, "player_name", pd.array(looked_up, dtype="string"))
    return out


def _sorted_by_rank(table: pd.DataFrame) -> pd.DataFrame:
    return table.sort_values(["rank", "player_id"], na_position="last", kind="stable").reset_index(drop=True)


def _top10(players: pd.DataFrame, summary: Mapping[str, Any]) -> pd.DataFrame:
    """One Role's provisional top-10 table (TOP10_COLUMNS; Requirement 18, criterion 15)."""
    ranked = _sorted_by_rank(players.loc[players["rank"].notna()]).head(LIST_SIZE).copy()
    for name in _TOP10_ROLE_VALUES:
        ranked[name] = float(summary[name])
    uses_fallback = bool(ranked["clutch_fallback"].to_numpy(dtype=bool).any())
    ranked["tau2_clutch"] = float(summary["tau2_clutch"]) if uses_fallback else math.nan
    return ranked.loc[:, list(TOP10_COLUMNS)].reset_index(drop=True)


def _selection_warnings(
    grid: pd.DataFrame, selection: MinBallsSelection, min_matches: int, target: int
) -> list[RatingWarning]:
    """The warnings of select_min_balls, each with its Role (it warns for the Roles in ROLES order)."""
    warned = [
        role
        for role in ROLES
        if not (grid.loc[(grid["role"] == role) & (grid["min_pressure_matches"] == min_matches), "n_eligible"]
                >= target).any()
    ]  # fmt: skip
    if len(warned) != len(selection.warnings):
        raise ValueError("select_min_balls warned for other Roles than the grid shows")
    return [RatingWarning("min_balls_fallback", role, None, text) for role, text in zip(warned, selection.warnings)]


def _warning_table(notes: Sequence[RatingWarning]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "role": pd.array([note.role for note in notes], dtype="string"),
            "kind": pd.array([note.kind for note in notes], dtype="string"),
            "estimate": pd.array([note.estimate for note in notes], dtype="string"),
            "message": pd.array([note.message for note in notes], dtype="string"),
        },
        columns=list(WARNING_COLUMNS),
    )


def _role_summary(
    report: pd.DataFrame, rankings: Mapping[str, RoleRanking], notes: Sequence[RatingWarning]
) -> pd.DataFrame:
    """The role_summary table (ROLE_SUMMARY_COLUMNS), one row per Role."""
    records = []
    for role in ROLES:
        record = report.loc[report["role"] == role].iloc[0].to_dict()
        values = rankings[role].summary()
        record.update({name: values[name] for name in RANKING_SUMMARY_FIELDS})
        record["n_warnings"] = sum(note.role == role for note in notes)
        records.append(record)
    table = pd.DataFrame.from_records(records, columns=list(ROLE_SUMMARY_COLUMNS))
    table["role"] = table["role"].astype("str")
    table["recent_rule_on"] = table["recent_rule_on"].astype(bool)
    integers = [name for name in ROLE_SUMMARY_COLUMNS if name.startswith(("n_", "min_", "fail_"))]
    return table.astype({name: np.int64 for name in integers})


def run_stage(ctx: StageContext) -> None:
    """The ``rating`` Pipeline_Stage (module docstring, "The ``rating`` Pipeline_Stage").

    Raises StageError (exit code 5) for a Role with fewer than LIST_SIZE Eligible_Players
    (Requirement 17, criterion 9), a D13 value other than the ``value`` table's cells, and a
    checkpoint file that is missing or changed; ValueError or TypeError for input tables that do
    not fit together or a Config value of the wrong type.
    """
    started = time.perf_counter()
    settings, store, log = ctx.settings, ctx.store, ctx.log
    cells = settings.get(_D13)
    if cells != _D13_CELLS:
        raise StageError(
            f"{_D13} is {cells!r}, but the rating stage rates the value table's {_D13_CELLS!r} cells; the D13 "
            "alternative is a Robustness_Case (Requirement 21, criterion 4), not a Main_Ranking setting"
        )
    pct = _percent(settings.get(_D12), _D12)
    recent_rule = _switch(settings.get(_RECENT_RULE), _RECENT_RULE)
    min_matches = settings.get(_D18_DEFAULT)
    target = settings.get(_D11_TARGET)
    rules = {"min_years": settings.get(_MIN_YEARS), "min_share": settings.get(_MIN_SHARE)}

    # 1. Delivery_Weights and the contributions (Requirement 16, criteria 1-2).
    matches = store.load_table(_LOAD_STAGE, "matches")
    weights = match_weights(
        matches, store.load_table(_LOAD_STAGE, "competitions"), store.load_table(_LOAD_STAGE, "teams"),
        half_life_years=settings.get(_D9), knockout_weight=settings.get(_D10), full_members=settings.get(_FULL_MEMBERS),
    )  # fmt: skip
    balls = store.load_table(RECONSTRUCT_STAGE, BALLS_TABLE)
    contrib = build_contributions(
        store.load_table(VALUE_STAGE, VALUE_TABLE), store.load_table(PRESSURE_STAGE, PRESSURE_TABLE),
        store.load_table(VALUE_STAGE, DISMISSAL_CHARGES_TABLE), balls, weights,
    )  # fmt: skip
    universe = _match_universe(weights, balls, contrib)
    del balls

    # 2. Eligibility at the selected minimums (Requirement 17); a short Role stops the Run here.
    counts = eligibility_counts(
        contrib, store.load_table(_LOAD_STAGE, "playing_eleven"), matches, recent_years=settings.get(_RECENT_YEARS)
    )
    grid = eligibility_grid(
        counts, min_balls_grid=settings.get(_D11_GRID), min_matches_grid=settings.get(_D18_GRID),
        recent_rule=recent_rule, **rules,
    )  # fmt: skip
    selection = select_min_balls(grid, min_matches=min_matches, target=target)
    chosen = {"min_balls": selection.min_balls, "min_matches": min_matches, "recent_rule": recent_rule, **rules}
    table = eligibility(counts, **chosen)
    report = eligibility_report(counts, **chosen)
    notes = _selection_warnings(grid, selection, min_matches, target)
    for note in notes:
        log.warning(note.message)
    require_eligible(report)

    # 3. Shrinkage, reliability, blend, Final_Score and ranks per Role (Requirement 18), mult all ones.
    rankings: dict[str, RoleRanking] = {}
    frames, timings = [], {}
    for role in ROLES:
        rules_of_role = table.loc[table["role"] == role].reset_index(drop=True)
        data = prepare_role(contrib, role, rules_of_role.loc[rules_of_role["eligible"], "player_id"])
        tick = time.perf_counter()
        rankings[role] = rank_role(data, replacement_pct=pct)
        timings[role] = time.perf_counter() - tick
        everyone = Aggregator(contrib, role=role).estimates(replacement_level=rankings[role].replacement_level)
        frames.append(_player_table(everyone, rules_of_role, rankings[role]))
        notes.extend(rankings[role].warnings)
    players = pd.concat(frames, ignore_index=True)
    summary = _role_summary(report, rankings, notes)
    warnings_table = _warning_table(notes)
    names = _player_names(store.load_table(_LOAD_STAGE, "ball_by_ball"))
    by_role = {role: _named(players.loc[players["role"] == role], names[role]) for role in ROLES}
    tops = {role: _top10(by_role[role], rankings[role].summary()) for role in ROLES}

    # 4. The checkpoint, with what the bootstrap reuses: contributions (halves included), match universe, players.
    store.save_table(STAGE_NAME, CONTRIBUTIONS_TABLE, contrib.to_frame())
    store.save_table(STAGE_NAME, MATCH_UNIVERSE_TABLE, universe)
    store.save_table(STAGE_NAME, ELIGIBILITY_GRID_TABLE, grid)
    store.save_table(STAGE_NAME, ELIGIBILITY_REPORT_TABLE, report)
    store.save_table(STAGE_NAME, PLAYER_ESTIMATES_TABLE, players)
    store.save_table(STAGE_NAME, ROLE_SUMMARY_TABLE, summary)
    store.save_table(STAGE_NAME, WARNINGS_TABLE, warnings_table)

    # 5. The files in outputs/rating/.
    folder = settings.path("outputs") / RATING_FOLDER
    files = [
        *(_write_csv(folder / PLAYER_ESTIMATES_FILES[role], _sorted_by_rank(by_role[role])) for role in ROLES),
        *(_write_csv(folder / TOP10_FILES[role], tops[role]) for role in ROLES),
        _write_csv(folder / ROLE_SUMMARY_FILE, summary),
        _write_csv(folder / ELIGIBILITY_GRID_FILE, grid),
        _write_csv(folder / ELIGIBILITY_REPORT_FILE, report),
        _write_csv(folder / WARNINGS_FILE, warnings_table),
    ]

    # 6. The data-derived constants behind the Main_Ranking.
    for role in ROLES:
        values = rankings[role].summary()
        values["min_pressure_balls"] = int(selection.min_balls[role])
        values["n_eligible"] = len(rankings[role])
        for quantity in CONSTANT_QUANTITIES:
            ctx.constants.record(f"{STAGE_NAME}.{quantity}.{role}", values[quantity], _COMPONENT)

    _log_stage(
        ctx, contrib=contrib, universe=universe, report=report, rankings=rankings, tops=tops, notes=notes,
        files=files, timings=timings, seconds=time.perf_counter() - started,
    )  # fmt: skip


def _log_stage(
    ctx: StageContext,
    *,
    contrib: Contributions,
    universe: pd.DataFrame,
    report: pd.DataFrame,
    rankings: Mapping[str, RoleRanking],
    tops: Mapping[str, pd.DataFrame],
    notes: Sequence[RatingWarning],
    files: Sequence[Path],
    timings: Mapping[str, float],
    seconds: float,
) -> None:
    log = ctx.log
    rows = {role: int((contrib.role == ROLE_CODES[role]).sum()) for role in ROLES}
    log.info(
        f"Rating_Engine inputs: {len(contrib):,} contribution rows (batter {rows[BATTER_ROLE]:,}, bowler "
        f"{rows[BOWLER_ROLE]:,}) over a match universe of {len(universe):,} matches, {int(universe['rated'].sum()):,} "
        f"of them rated and {int(universe['resampled'].sum()):,} with a Delivery outside Super_Over_Innings"
    )
    for record in report.to_dict("records"):
        fails = ", ".join(f"{rule} {int(record[f'fail_{rule}']):,}" for rule in ELIGIBILITY_RULES)
        log.info(
            f"Eligibility, {record['role']} (Requirement 17, criterion 11): minimum "
            f"{int(record['min_pressure_balls'])} Pressure_Balls and {int(record['min_pressure_matches'])} "
            "Pressure_Matches, Recent_Activity_Rule "
            f"{'on' if record['recent_rule_on'] else 'off'}; {int(record['n_eligible']):,} Eligible_Players of "
            f"{int(record['n_players']):,} players with a Delivery; failing: {fails}"
        )
    for role in ROLES:
        ranking = rankings[role]
        s = ranking.summary()
        taus = "; ".join(
            f"{label} {s[f'tau2_{key}']:.6f} (raw {s[f'tau2_{key}_raw']:.6f}, {s[f'n_tau2_{key}']:,} players)"
            for label, key in ((USUAL_BASELINE, "usual"), (PRESSURE_GAP, "gap"), (UPLIFT, "uplift"),
                               (f"fallback {CLUTCH_LEVEL}", "clutch"))
        )  # fmt: skip
        medians = ", ".join(
            f"{label} {np.nanmedian(values):.3f}" if np.isfinite(values).any() else f"{label} none"
            for label, values in ((USUAL_BASELINE, ranking.b_usual), (PRESSURE_GAP, ranking.b_gap),
                                  (UPLIFT, ranking.b_uplift))
        )  # fmt: skip
        log.info(
            f"Rating, {role} (Requirement 18): Replacement_Level {s['replacement_level']:+.4f} runs per over (the "
            f"{s['replacement_pct']:g}th percentile of {s['n_pool']:,} Clutch_Levels); tau^2 {taus}; "
            f"{s['n_fallback']:,} player(s) on the Clutch_Level fallback (no Usual_Baseline); "
            f"rho_L {s['rho_L']:.4f} (r {s['r_L']:.4f}), rho_U {s['rho_U']:.4f} (r {s['r_U']:.4f}), from "
            f"{s['n_rho']:,} players with Pressure_Balls and Ordinary_Balls in both halves; w_L {s['w_L']:.4f}, w_U "
            f"{s['w_U']:.4f}; {s['n_ranked']:,} ranked; median Shrinkage_Factor B {medians}; rank_role "
            f"{timings[role] * 1000:.1f} ms"
        )
        top = tops[role]
        log.info(
            f"Provisional top {len(top)}, {role} (Requirement 18, criterion 15; names provisional): "
            + "; ".join(
                f"{int(row['rank'])} {row['player_name']} ({int(row['player_id'])}) S {row['final_score']:+.3f}"
                for row in top.to_dict("records")
            )
        )
    for note in notes:
        if note.kind != "min_balls_fallback":  # logged before the eligibility check
            log.warning(note.message)
    log.info(
        f"wrote {', '.join(_shown_path(path, ctx.root) for path in files)}; {len(notes)} warning(s); saved "
        f"{CONTRIBUTIONS_TABLE}, {MATCH_UNIVERSE_TABLE}, {ELIGIBILITY_GRID_TABLE}, {ELIGIBILITY_REPORT_TABLE}, "
        f"{PLAYER_ESTIMATES_TABLE}, {ROLE_SUMMARY_TABLE} and {WARNINGS_TABLE} in {seconds:.1f} s"
    )
