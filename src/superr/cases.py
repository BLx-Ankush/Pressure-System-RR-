"""Robustness_Suite, part 2: layered recomputation of the Robustness_Cases (Requirement 21, criterion 7;
Requirement 23, criterion 10; design decision DD-10 and section "Robustness_Suite and activity
comparison").

Placement
    The bootstrap and its Pipeline_Stage stay in ``superr.robustness``. This module holds the
    machinery that every Robustness_Case and every model-choice rerun shares: ``BaseArtifacts``
    (the cached arrays of the Main_Ranking's run), ``CaseSpec`` (one case, as an override
    vocabulary), ``run_case`` (recompute one case from its layer down), ``repredict`` (the
    Venue_Effect and Match_Day_Conditions re-prediction with the saved outer Fold_Models) and
    ``run_with_predictions`` (any stored M3 and M4 prediction arrays in place of the
    Primary_Models', for task 26.3). It also lists the Robustness_Cases of Requirement 21
    (``all_cases``), runs them (``run_all_cases``) and builds and writes the robustness table
    (``robustness_table``, ``write_robustness_table``); task 25.4 registers the ``robustness``
    stage, which calls them and adds its Cross_Check_Ranking rows to the same table. No function
    reads ``config.py``: ``BaseArtifacts.build`` and ``all_cases`` read every Config value they
    need from the ``Settings`` they are given, and no model is fitted anywhere in this module.

Layers and caches (``BaseArtifacts``)
    ``BaseArtifacts.build(settings, store)`` reads the checkpoints of a completed Run (``state``,
    ``reconstruct``, ``folds``, the views V_1 ... V_D15, ``model_choice``, ``pressure``,
    ``value``, ``rating`` and the ``load`` tables) and caches the arrays of each layer:

    ======== ===================================================================================
    Layer    Cached arrays (all in ``state`` order unless stated)
    ======== ===================================================================================
    state    the ``state`` table (its D6 window columns ``win_wkts`` and ``win_rr``), the
             own-fold view columns of the Pillar_Components with ``par_score`` (the saved
             Par_Score curve values), the Training_Folds mean wickets of F20(a), and the D6
             window frames that cases compute (memoised by window length)
    pressure the Primary_Models' WP (``wp_pre``, ``wp_post``) and ``avg_both`` M4 probabilities,
             the Live_Ball flags, LI, the Pillar_Components, Stress_Pillars, Stress, both Roles'
             PI and Pressure_Ball flags (the saved ``pressure`` table)
    value    the ``value`` stage inputs (``value.load_value_inputs``: the rated Deliveries, Omega,
             Class_Run_Values, the four M4 views and ``avg_both``, the Extras_Model
             probabilities, the Training_Folds cell tables), the saved NBVs and
             Dismissal_Charges, and the frame behind a Free_Hit_Premium with other cells
    rating   the match weights (D9, D10), the Contributions, the eligibility counts and table at
             the Main_Ranking's minimum counts, each Role's RoleData and RoleRanking (the
             Main_Ranking, recomputed at build time and checked bit for bit against the
             ``rating`` checkpoint's ``player_estimates``), and the checkpoint's Main_Ranking
             tables (``main``)
    ======== ===================================================================================

    It also memoises the re-predictions it computes (``repredict``). The cached numpy arrays are
    read-only, so a case cannot change them by mistake.

Override vocabulary (``CaseSpec.overrides``, OVERRIDE_LAYERS)
    A CaseSpec names one layer and a mapping of overrides; every key must belong to that layer.
    A key that a case leaves out keeps the Main_Ranking's value (``BaseArtifacts.defaults``):
    the Config value of its Design_Choice, or for ``min_balls`` the Main_Ranking's selected
    minimum Pressure_Ball count of each Role (the Robustness_Case glossary entry). Every case
    keeps the Main_Ranking's minimum Pressure_Match count, and eligibility is re-applied to the
    case's own Pressure_Balls.

    ==================== ======== ============================================================
    Key                  Layer    Value and what it changes (Requirement 21)
    ==================== ======== ============================================================
    window               state    D6 window in Legal_Balls (12, 24): ``win_wkts``, ``win_rr``
                                  (``state.recent_window``) and ``par_rate_win`` over the same
                                  window for F20(b) and the innings-1 F23. Model predictions do
                                  not change
    live_band            pressure D1 band (lower, upper), both inclusive: (b, 1 - b) for the D1
                                  grid, (0.0, 1.0) for "live band removed" (every Delivery a
                                  Live_Ball)
    alpha                pressure D2 blend: 0 removes leverage, 1 removes Stress
    share                pressure D3 pressure share (0.20, 0.25, 0.30)
    scope                pressure D4: ``"pooled"`` ranks pillars, LI and Stress over the Live_Balls of
                                  both Innings_Groups; the share still applies within each Role
                                  and Innings_Group
    new_batter_cap       pressure D7 cap (10, 30) of the new-at-the-crease pillar F21
    without              pressure Stress_Pillars left out of Stress, such as ``("F21",)``: the eight
                                  pillar ablations (D5 alternatives); Stress is the
                                  equal-weighted mean of the Role's other pillars
    opposition           value    False: the Expected_Values from the ``avg_both`` M4 variant
                                  (average batter and average bowler), Extras_Model term
                                  unchanged; forced-Free_Hit terms as decided below
    phase_splits         value    D8 splits (s1, s2), such as (0.25, 0.80), for the Usual_Baseline
                                  matching cells and the Free_Hit_Premium cells; model
                                  predictions unchanged (decision below)
    retire_out_penalty   value    D14: True charges Retire Out (code 9) as a batter dismissal,
                                  Omega off the batter NBV, no bowler credit
    zero_features        value    View features set to 0 on every Delivery before the saved outer
                                  Fold_Models re-predict (``repredict``): ``("venue_effect",
                                  "match_day")`` is the Venue_Effect and Match_Day_Conditions
                                  ablation; ``()`` re-predicts with the saved features, a check
                                  that the re-prediction reproduces the stored predictions
    half_life_years      rating   D9 half-life in years; None means no decay (Recency_Weight 1,
                                  also the recency ablation)
    knockout_weight      rating   D10 knockout Stakes_Weight; 1.0 is also the Stakes_Weight
                                  ablation
    min_balls            rating   D11 minimum Pressure_Ball count: one number for both Roles or
                                  {"batter": ..., "bowler": ...}; with the Main_Ranking's
                                  minimum Pressure_Match count (the D18 default)
    replacement_pct      rating   D12 Replacement_Level percentile (20, 30)
    cells                rating   D13 matching cells: ``"innings_phase"`` (Config) or
                                  ``"innings_phase_comp_type"`` (innings x Phase x
                                  Competition_Type, 18 cells)
    blend                rating   fixed Blend_Weights w_L (1.0, 0.7, 0.5), w_U = 1 - w_L; None
                                  keeps the reliability-based weights
    zscore               rating   True: Final_Score = w_L x Role_Z(shrunk Clutch_Level) + w_U x
                                  Role_Z(shrunk Uplift) with the Main_Ranking's Blend_Weights (or
                                  ``blend``)
    ==================== ======== ============================================================

    ``default_overrides(base, layer)`` gives every key of a layer at its Main_Ranking value
    (``zero_features`` aside): a case built from it recomputes the layer and everything below it
    and must reproduce the Main_Ranking (Requirement 21, criterion 9).

Recomputation (``run_case``)
    Each step reruns only when one of its inputs is overridden or recomputed; otherwise it reads
    the cache. A key that a case names counts as changed even when its value equals the default,
    so the per-layer identity cases rerun every step of their layer. The steps, in order, with the
    keys and upstream changes that trigger them:

    1. ``window`` (state): ``state.recent_window`` for ``win_wkts`` and ``win_rr``, and
       ``par_rate_win`` = 6 x (Par_Score(L) - Par_Score(L - w)) / w on innings-1 rows, with both
       Par_Scores read from the own-fold view's saved ``par_score`` of the same innings (every
       Delivery of an innings shares one context and one first-innings M2 fit, and the innings
       holds a Delivery at every count 0 ... L, so the saved values are points of the saved par
       curve). No model is called.
    2. Pillar_Components (``pressure.pillar_components``): window, ``new_batter_cap``, new view
       values (re-prediction).
    3. Live_Balls (``pressure.live_balls``): ``live_band``, new WP.
    4. Stress_Pillars (``pressure.pillars``): new components, Live_Balls or ``scope``.
    5. Stress (``pressure.stress``): new pillars or ``without``.
    6. LI (``pressure.leverage``, pooled over every Delivery): new WP or M4 predictions.
    7. PI (``pressure.finite_guard``, then ``pressure.pressure_index`` per Role): new Stress, LI,
       Live_Balls, ``scope`` or ``alpha``.
    8. Pressure_Ball flags (``pressure.pressure_flags``, always within each Role and
       Innings_Group): new PI or ``share``.
    9. Batter NBV (``value.batter_nbv``): ``opposition``, ``retire_out_penalty``, new M4 or
       Omega. Bowler NBV (``value.bowler_nbv``): ``opposition``, ``phase_splits``, new M4,
       Omega or Extras_Model probabilities.
    10. Dismissal_Charges (``value.dismissal_charges``): new flags, Omega or
        ``retire_out_penalty``.
    11. Match weights (``rating.match_weights``): ``half_life_years``, ``knockout_weight``.
    12. Contributions (``rating.build_contributions``) and eligibility counts
        (``rating.eligibility_counts``): any change above, or ``cells``.
    13. Eligibility (``rating.eligibility``): new counts or ``min_balls``.
    14. RoleData and ranking (``rating.prepare_role``, ``rating.rank_role``): new Contributions or
        Eligible_Players, or ``replacement_pct``: Replacement_Level, Shrinkage,
        Split_Half_Reliability, Blend_Weights, Final_Score and ranks.
    15. Re-blend (``rating.final_scores``, ``rating.rank_order``): ``blend`` or ``zscore``, on the
        case's RoleRanking (the cached Main_Ranking for a pure re-blend case).

    No step draws a Bootstrap_Resample. A Role with fewer than LIST_SIZE Eligible_Players is
    ranked all the same (or left empty without any), so that task 25.2 can apply the
    not-comparable rule of criterion 8.

Decisions
    * Opposition adjustment removed (``nbv_views``). ``batter_view`` and ``bowler_view`` are
      replaced by ``avg_both``, which keeps each Delivery's own Free_Hit flag, so every
      Legal_Ball term (the batter EV, the bowler EV, the Wide's replacement ball in its
      Extras_Cost) uses an average batter and an average bowler. No stored variant has both
      players at Average_Skill with the Free_Hit flag forced. Where a Delivery's own flag already
      is the forced one, the forced prediction is the own-flag prediction of the same model and
      row, so ``avg_both`` is exact there too: ``batter_view_fh`` reads ``avg_both`` on Free_Hit
      Deliveries, ``bowler_view_nofh`` on every other Delivery. The rest keep their stored views
      and so the opponent's actual theta: ``batter_view_fh`` on the No_Balls that are not
      Free_Hits (the batter EV of a No_Ball, Requirement 15, criterion 3; 766 of the 786 rated
      No_Balls) and ``bowler_view_nofh`` on the Free_Hit Deliveries (860 rated Deliveries: the
      replacement ball of a Free_Hit No_Ball's Extras_Cost, criterion 14, and the expected
      No_Ball cost in their bowler EV, criterion 6, weighted by the No_Ball probability, about
      0.004). On the No_Balls the forced flag moves the expected bat runs by 0.45 runs on
      average and the opponent's theta by 0.02 (module ``value``, "The own-theta check", makes
      the same trade). Re-predicting an ``avg_both`` variant with the flag forced would break
      criterion 7's reuse of the saved predictions outside the Venue_Effect case.
    * D8 alternative splits. The case recomputes the Phase of every Delivery from its
      ``ball_share`` (``state.phase_codes``, ``state.cell_codes``) and uses those innings x Phase
      cells for the Usual_Baseline matching cells (the Contributions' cells) and the
      Free_Hit_Premium (``value.free_hit_premium`` on every Delivery with the new cells), as
      criterion 4 names them. The expected Wide and No_Ball cost means of criterion 6 keep the
      Config cells, and so do the model inputs (M4's ``phase``, the Extras_Model's ``cell``).
      ``value.bowler_nbv`` looks both tables up through one cell column, so the case passes a
      joint code (Config cell x 6 + new cell) and joint copies of the two tables; with the
      Config splits the lookups return the same values bit for bit.
    * D4 pooled. Pillars, LI, Stress and PI are ranked over one pooled group, and the
      Pressure_Ball flags still rank PI within each Role and Innings_Group, as criterion 4 asks.
    * Role_Z (Z-score case). Mean and standard deviation (population, ddof 0) over the Role's
      ranked Eligible_Players of the case; a value that is equal for every such player has no
      spread and gets Role_Z 0. The Final_Score then rests on the other estimate, which keeps the
      ranking defined when every shrunk Uplift is 0 (tau^2 of Uplift floored at 0).
    * Re-prediction (``repredict``). Only the outer Fold_Models predict, each Delivery with the
      model of its own fold, as in every Out_Of_Fold_Prediction, in the order of criterion 1:
      M2 (calibrated capped and first-innings ``M2Model``s; Omega, Par_Score, Par_Rate,
      Achievable_Rate and F19 from the saved par means and par bias tables), the M3
      Primary_Models of both innings (pre-ball WP, then the Post_Outcome_States rebuilt by
      ``pressure.post_outcome_states`` from the new view values, with their WP), the M4
      Primary_Model (all six variants, rounded through the float32 ``m4_<family>`` table as the
      ``outcome`` and ``model_choice`` stages store them) and the Extras_Model (``p_noball``
      capped at 1 - ``p_wide`` as ``extras.preds_frame`` stores it). A Primary_Model may be any
      Candidate_Model: the M3 fold models of the logistic and LightGBM families come from the
      ``win_prob`` checkpoint and those of XGBoost and the random forest from the
      ``model_choice`` checkpoint, and ``model_choice.load_m4_models`` reads the M4 fold models of
      any family from the checkpoint that the ``model_choice`` SOURCES_TABLE names. Calibrators,
      Legal_Balls-left calibration, par means, par bias, Average_Skill and Class_Run_Values are
      part of the fitted models and are reused unchanged. Team_Strength_Gap, theta and RBQ do not
      depend on the two features.

Prediction overrides (``run_with_predictions``; task 26.3)
    ``run_with_predictions(base, predictions, wp_pre=..., wp_post=..., m4=...)`` puts stored
    prediction arrays in place of the Primary_Models' and recomputes WP -> Live_Ball flags -> LI
    -> Stress_Pillars -> Stress -> PI -> Pressure_Ball flags -> NBV and Dismissal_Charges ->
    Contributions -> eligibility at the Main_Ranking's minimum counts -> Final_Score -> ranks,
    with every Design_Choice at its Config value. ``predictions`` is a
    ``model_choice.PrimaryPredictions`` (``model_choice.load_predictions``); ``wp_pre`` (n,),
    ``wp_post`` (n, 9) and ``m4`` (variant -> (n, 9)) are arrays in ``state`` order and take
    precedence. Any array passed counts as changed; pass only the swapped model's arrays to skip
    work (an M4 swap leaves the Live_Balls and Stress_Pillars cached). Omega, the Extras_Model
    probabilities and every view value stay as stored, because M2, M1 and the Extras_Model are
    not Candidate_Models.

Results (``CaseResult``, ``compare_with_main``)
    A CaseResult holds, per Role, the case ranking (RANKING_COLUMNS: ``player_id``,
    ``eligible``, ``rank`` (Int64, <NA> unless ranked) and ``final_score``), one row per player of
    the Role's Contributions in ascending ``player_id``, as ``base.main[role]`` holds the
    Main_Ranking; the Eligible_Player counts; the minimum Pressure_Ball counts and Blend_Weights
    used; the steps recomputed with their times; and the Rating_Engine warnings. Both tables go
    straight into ``ranks.compare_rankings``; ``compare_with_main(base, result, role)`` does it
    with ``on="final_score"`` (Requirement 21, criterion 5).

Robustness_Cases (``all_cases``; Requirement 21, criteria 1-4)
    ``all_cases(settings, base=None)`` lists the cases in the order below, each CaseSpec with its
    ``kind``. Every grid value and alternative is read from the Config's ``sensitivity`` section
    (the grid named in brackets); the ablation settings define the ablations and are not Config
    values. The delivered Config gives 50 cases: 15 ablations, 20 grid cases, the Z-score case
    and 14 alternatives.

    =========== ================================================ ========== =======================
    kind        Cases (label; Config grid)                       target_id  Override
    =========== ================================================ ========== =======================
    ablation    leverage removed (alpha = 0)                     F17        alpha 0
                Stress removed (alpha = 1)                       F19-F26    alpha 1
                F19 removed ... F26 removed                      D5         without (pillar,)
                opposition adjustment removed                    F28        opposition False
                Venue_Effect and Match_Day_Conditions removed    F11+F12    zero_features (both)
                Recency_Weight = 1                               F33        half_life_years None
                Stakes_Weight = 1                                F34        knockout_weight 1.0
                live band removed                                F18        live_band (0, 1)
    sensitivity D2 alpha = a (alpha)                             D2         alpha
                D3 share = s (share)                             D3         share
                D1 b = b (live_b)                                D1         live_band (b, 1 - b)
                D9 half-life = h (half_life; None: no decay)     D9         half_life_years
                D10 knockout = k (knockout)                      D10        knockout_weight
                blend w_L = w (blend_wL)                         F40        blend
    zscore      Z-score blend                                    F42        zscore True
    alternative D4 scope = ... (D4)                              D4         scope
                D6 window = ... (D6)                             D6         window
                D7 cap = ... (D7)                                D7         new_batter_cap
                D8 splits = s1 / s2 (D8)                         D8         phase_splits
                D11 minimum = ... (D11), both Roles              D11        min_balls
                D12 percentile = ... (D12)                       D12        replacement_pct
                D13 cells = ... (D13)                            D13        cells
                D14 Retire Out charged to the batter (D14)       D14        retire_out_penalty
    =========== ================================================ ========== =======================

    ``target_id`` is the Design_Choice that a grid case or alternative changes. An ablation names
    the factor ID of ``plan.md`` for the component it removes (F17 LI, F19-F26 the
    Stress_Pillars together, F28 the Expected_Values, F11 and F12 Venue_Effect and
    Match_Day_Conditions, F33 Recency_Weight, F34 Stakes_Weight, F18 the Live_Ball flag); the
    fixed Blend_Weights (not a Design_Choice) name F40 and the Z-score case F42 (Final_Score).
    The eight Stress_Pillar ablations count as the D5 alternatives (criterion 4), so their
    ``target_id`` is D5 and their setting names the pillar. With a component ID, the alpha 0,
    alpha 1, Recency_Weight and Stakes_Weight ablations stay out of the D2, D9 and D10 rows of
    the report's Design_Choice table, where the grid cases with the same overrides already stand;
    ``run_all_cases`` computes each such pair once. The grid cases at the Main_Ranking's values
    (alpha 0.5, share 25%, b 0.05, 3 years, 1.25, and the D11 count that the eligibility grid
    selected) stay in the list: they recompute their layer and must reproduce the Main_Ranking
    (``is_default``). ``base`` only adds the Main_Ranking's Blend_Weights to the Z-score case's
    setting text.

Running the cases (``run_all_cases``)
    ``run_all_cases(cases, base, names=..., log=...)`` runs the cases in order with ``run_case``
    and compares each Role's ranking with the Main_Ranking (``compare_with_main``), giving one
    CaseRun per case. A case with the same layer and overrides as an earlier case reuses that
    CaseResult under its own label (``CaseRun.reused``). A Role left with too few Eligible_Players
    is compared like any other, and the table blanks what criterion 8 asks. ``names`` gives the
    entering and leaving players' ``player_name``: ``PlayerNames.from_store(store)`` applies the
    List_Builder's name rule (``lists.choose_name``: the player's Candidate_Name with the most
    mapping rows, else the most frequent delivery name) and memoises it. ``log`` (the Run_Log)
    gets one line per case and one warning per not-comparable Role.

The robustness table (``robustness_table``, ``write_robustness_table``; criteria 5, 6 and 8)
    One row per case and Role (ROLES order), in case order, with ROBUSTNESS_COLUMNS:

    * ``case`` (the label), ``kind``, ``target_id``, ``setting``, ``layer`` (blank for a
      cross-check row) and ``role``;
    * ``n_eligible_case`` and ``n_eligible_main``: the Role's Eligible_Players in the case and in
      the Main_Ranking; ``n_common``: the players who are Eligible_Players of both;
    * ``spearman``: Spearman's correlation of the case's and the Main_Ranking's Final_Scores over
      the common players (criterion 5; ``ranks.compare_rankings`` with ``on="final_score"``);
    * ``overlap``: the players in both top 10s, 0 to 10;
    * ``comparable``: False when the case leaves fewer than LIST_SIZE (10) Eligible_Players in the
      Role or fewer than MIN_COMMON_PLAYERS (3) common players (criterion 8). ``spearman`` (NaN)
      and ``overlap`` (<NA>) are then blank; the counts and the players are kept;
    * ``is_default``: True when every override of the case equals the Main_Ranking's value
      (``BaseArtifacts.defaults``, read from the Settings, and for D11 the Role's selected minimum
      Pressure_Ball count). Such a row must show Spearman 1 and overlap 10, and the report's
      Design_Choice table leaves it out. A fixed-blend case is never default: the Main_Ranking's
      Blend_Weights come from the Split_Half_Reliability rule, even where they equal a grid value;
    * ``entering_ids`` and ``entering_names``: the case's top-10 players outside the
      Main_Ranking's top 10, in the case's rank order; ``leaving_ids`` and ``leaving_names``: the
      Main_Ranking's top-10 players outside the case's top 10, in the Main_Ranking's rank order.
      LIST_SEPARATOR ("; ") joins them; "" when there is none, and missing names without a name
      rule.

    ``comparison_row`` builds a row from any RankingComparison with the Main_Ranking and
    ``robustness_frame`` types a list of rows, so task 25.4 can add the Cross_Check_Ranking rows
    (kind CROSS_CHECK) with the same columns. ``write_robustness_table(table, out_dir)`` writes
    ``<out_dir>/robustness_table.csv`` (ROBUSTNESS_FILE; ``outputs/robustness/`` in a Run) in
    UTF-8 with ``\\n`` line ends, no index, ROBUSTNESS_FLOAT_FORMAT (10 decimals, as the
    Main_Lists) and a blank cell for a missing value.

As delivered (checks of 16 November 2026, on the checkpoints of 15 November 2026)
    ``BaseArtifacts.build`` takes about 6.5 s (3.4 s of it the value stage's inputs) and its
    recomputed rankings equal both Roles' Main_Ranking bit for bit (130 batter and 139 bowler
    Eligible_Players, minimum 90 Pressure_Balls and 8 Pressure_Matches). The case without
    overrides and the four per-layer cases of ``default_overrides`` reproduce the Main_Ranking
    bit for bit (same Eligible_Players, ranks and Final_Scores; largest |change| 0). Re-predicting
    with the saved features (``zero_features=()``) reproduces every stored prediction bit for bit
    (pre-ball and Post_Outcome_State WP, the six M4 variants, Omega, both Extras_Model
    probabilities, the five view values) and so the Main_Ranking, and so does
    ``run_with_predictions`` with the stored Primary_Models' arrays. The 48 cases of Requirement
    21 take about 140 s together: state cases about 3.3 s each, pressure cases 1.1-3.4 s (mean
    2.1), value cases 1.2-1.3 s, rating cases 0.0-1.6 s (mean 0.6) and the Venue_Effect and
    Match_Day_Conditions case about 70 s, 50-60 s of it the six M4 variants. With both features at
    0, the pre-ball WP moves by 0.025 on average (at most 0.31), 2,305 Live_Ball flags change,
    Omega moves by 1.6 runs and F19 by 3.0 on average. The D8 splits 0.25 / 0.80 move 20,396
    Deliveries to another cell. Every shrunk Uplift is 0 in both Roles (the Uplift tau^2 is
    floored at 0), so the fixed-blend and Z-score cases keep the Main_Ranking's order.

    The 50 cases of ``all_cases`` (task 25.2, same checkpoints): ``run_all_cases`` with the name
    rule takes about 119 s (46 cases computed, 4 reused), 60 s of it the Venue_Effect and
    Match_Day_Conditions case (53 s re-prediction, 41 s of that the M4 variants). Every row is
    comparable. Spearman: batters median 0.990, lowest 0.578 (D11 minimum = 60, top-10 overlap
    4); bowlers median 0.983, lowest 0.593 (D3 share = 30%, overlap 3). The 12 ``is_default``
    rows reproduce the Main_Ranking bit for bit. The D11 Eligible_Player counts equal the rating
    checkpoint's eligibility grid at 8 Pressure_Matches (batters 187, 162, 140, 130; bowlers 178,
    162, 147, 139); at 45 and 60 the batter Uplift reliability is above 0 (w_L 0.80 and 0.70), so
    the shrunk Uplift enters those Final_Scores. Retire Out charged (D14) moves batter
    Final_Scores by at most 1.4e-4 and no rank. With every Primary_Model an XGBoost (the
    ``model_choice`` checkpoint of a task 26.2 rerun), ``repredict`` with the saved features
    reproduces the stored WP and M4 variants bit for bit.

Errors: TypeError or ValueError, naming the problem, for a CaseSpec with an unknown key, a key of
another layer, a value out of range or an unknown kind, for prediction arrays of the wrong shape
or outside [0, 1], for an empty Config grid, two cases with one label, and robustness rows with a
missing column, an unknown kind or Role, or a not-comparable row with values; StageError when the
checkpoints no longer reproduce the Main_Ranking or the Main_Ranking was computed with other
settings than the Config's, and from the name rule for a player without any name; the errors of
the functions called (``pressure.finite_guard`` raises PressureInputError for a non-finite PI
input).
"""

from __future__ import annotations

import math
import os
import time
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import pandas as pd

from superr import folds, leakage, pressure, rating, reconstruct, views
from superr import state as state_module
from superr import value as value_module
from superr.errors import StageError
from superr.models import common, extras, model_choice, outcome, remaining_runs, win_prob
from superr.models.model_choice import PrimaryPredictions
from superr.models.outcome import AVG_BOTH, BATTER_VIEW, BATTER_VIEW_FH, BOWLER_VIEW, BOWLER_VIEW_NOFH, N_CLASSES
from superr.rating import LIST_SIZE, RatingWarning, RoleRanking
from superr.ranks import ENTERING, LEAVING, MIN_COMMON_PLAYERS, RankingComparison, compare_rankings
from superr.value import BATTER_ROLE, BOWLER_ROLE, CELL_COLUMN, ROLES, ValueInputs

if TYPE_CHECKING:
    from superr.settings import Settings
    from superr.store import CheckpointStore

__all__ = [
    "ABLATION",
    "ALTERNATIVE",
    "CASE_KINDS",
    "CROSS_CHECK",
    "D13_CELLS",
    "D13_COMP_TYPE",
    "D13_DEFAULT",
    "LAYERS",
    "LIST_SEPARATOR",
    "NBV_VARIANTS",
    "OVERRIDE_LAYERS",
    "PREDICTIONS_LAYER",
    "PRESSURE_LAYER",
    "RANKING_COLUMNS",
    "RANKING_VARIANTS",
    "RATING_LAYER",
    "ROBUSTNESS_COLUMNS",
    "ROBUSTNESS_FILE",
    "ROBUSTNESS_FLOAT_FORMAT",
    "SENSITIVITY",
    "STATE_LAYER",
    "TABLE_KINDS",
    "VALUE_LAYER",
    "ZERO_FEATURES",
    "ZSCORE",
    "BaseArtifacts",
    "CaseResult",
    "CaseRun",
    "CaseSpec",
    "PlayerNames",
    "Predictions",
    "all_cases",
    "compare_with_main",
    "comparison_row",
    "default_overrides",
    "nbv_views",
    "repredict",
    "robustness_frame",
    "robustness_table",
    "run_all_cases",
    "run_case",
    "run_with_predictions",
    "write_robustness_table",
]


# ================================================================================ constants

#: The layers of a Robustness_Case, from the earliest (DD-10): each case names one.
STATE_LAYER = "state"
PRESSURE_LAYER = "pressure"
VALUE_LAYER = "value"
RATING_LAYER = "rating"
LAYERS: tuple[str, ...] = (STATE_LAYER, PRESSURE_LAYER, VALUE_LAYER, RATING_LAYER)

#: The ``layer`` of a CaseResult from run_with_predictions: stored predictions swapped in.
PREDICTIONS_LAYER = "predictions"

#: Each override key and the layer it belongs to (module docstring, "Override vocabulary").
OVERRIDE_LAYERS: Mapping[str, str] = MappingProxyType(
    {
        "window": STATE_LAYER,
        "live_band": PRESSURE_LAYER,
        "alpha": PRESSURE_LAYER,
        "share": PRESSURE_LAYER,
        "scope": PRESSURE_LAYER,
        "new_batter_cap": PRESSURE_LAYER,
        "without": PRESSURE_LAYER,
        "opposition": VALUE_LAYER,
        "phase_splits": VALUE_LAYER,
        "retire_out_penalty": VALUE_LAYER,
        "zero_features": VALUE_LAYER,
        "half_life_years": RATING_LAYER,
        "knockout_weight": RATING_LAYER,
        "min_balls": RATING_LAYER,
        "replacement_pct": RATING_LAYER,
        "cells": RATING_LAYER,
        "blend": RATING_LAYER,
        "zscore": RATING_LAYER,
    }
)

#: The D13 matching cells: the Config value and the alternative of Requirement 21, criterion 4.
D13_DEFAULT = "innings_phase"
D13_COMP_TYPE = "innings_phase_comp_type"
D13_CELLS: tuple[str, ...] = (D13_DEFAULT, D13_COMP_TYPE)

#: The view features that ``zero_features`` may set to 0 (Requirement 21, criterion 1).
ZERO_FEATURES: tuple[str, ...] = ("venue_effect", "match_day")

#: Columns of every ranking table of a CaseResult and of ``BaseArtifacts.main``.
RANKING_COLUMNS: tuple[str, ...] = ("player_id", "eligible", "rank", "final_score")

#: The M4 variants that the NBV formulas read, and those that the ranking reads (LI adds ``avg_both``).
NBV_VARIANTS: tuple[str, ...] = (BATTER_VIEW, BATTER_VIEW_FH, BOWLER_VIEW, BOWLER_VIEW_NOFH)
RANKING_VARIANTS: tuple[str, ...] = (*NBV_VARIANTS, AVG_BOTH)

#: The ``kind`` of a Robustness_Case (``CaseSpec.kind``) and of its robustness-table rows: the
#: ablations of Requirement 21, criterion 1, the grids of criterion 2, the Z-score case of
#: criterion 3 and the Config default alternatives of criterion 4 (module docstring,
#: "Robustness_Cases"). CROSS_CHECK is the kind of the Cross_Check_Ranking rows that task 25.4 adds
#: to the same table (Requirement 22, criterion 6); no CaseSpec has it.
ABLATION = "ablation"
SENSITIVITY = "sensitivity"
ZSCORE = "zscore"
ALTERNATIVE = "alternative"
CROSS_CHECK = "cross_check"
CASE_KINDS: tuple[str, ...] = (ABLATION, SENSITIVITY, ZSCORE, ALTERNATIVE)
TABLE_KINDS: tuple[str, ...] = (*CASE_KINDS, CROSS_CHECK)

#: Columns of the robustness table, in order (module docstring, "The robustness table").
ROBUSTNESS_COLUMNS: tuple[str, ...] = (
    "case", "kind", "target_id", "setting", "layer", "role", "n_eligible_case", "n_eligible_main", "n_common",
    "spearman", "overlap", "comparable", "is_default", "entering_ids", "entering_names", "leaving_ids",
    "leaving_names",
)  # fmt: skip

#: The robustness table's file in the Robustness_Suite folder (``outputs/robustness/``), its float
#: format (10 decimals, as the Main_List files) and the separator of its id and name lists.
ROBUSTNESS_FILE = "robustness_table.csv"
ROBUSTNESS_FLOAT_FORMAT = "%.10f"
LIST_SEPARATOR = "; "

#: The Config settings that BaseArtifacts.build reads, by override key.
_CONFIG_KEYS: Mapping[str, str] = MappingProxyType(
    {
        "window": "design_choices.D6_window",
        "live_band": "design_choices.D1_live_band",
        "alpha": "design_choices.D2_alpha",
        "share": "design_choices.D3_pressure_share",
        "scope": "design_choices.D4_scope",
        "new_batter_cap": "design_choices.D7_new_batter_cap",
        "phase_splits": "design_choices.D8_phase_splits",
        "retire_out_penalty": "design_choices.D14_retire_out_batter_penalty",
        "half_life_years": "design_choices.D9_half_life_years",
        "knockout_weight": "design_choices.D10_knockout_weight",
        "replacement_pct": "design_choices.D12_replacement_pct",
        "cells": "design_choices.D13_cells",
    }
)
_D5 = "design_choices.D5_weights"
_D15 = "design_choices.D15_n_folds"
_D18_DEFAULT = "design_choices.D18_min_matches_default"
_FULL_MEMBERS = "data_rules.full_members"
_RECENT_RULE = "data_rules.recent_activity_rule"
_RECENT_YEARS = "data_rules.recent_years"
_MIN_YEARS = "data_rules.min_calendar_years"
_MIN_SHARE = "data_rules.min_major_share"
_LOAD_STAGE = "load"

#: The Stress column and PI of each Role (the Extras_Free_PI serves validation only, so no case needs it).
_STRESS_OF: Mapping[str, str] = MappingProxyType({BATTER_ROLE: "stress_bat", BOWLER_ROLE: "stress_bowl"})
_PILLAR_NAMES: tuple[str, ...] = tuple(pillar.name for pillar in pressure.PILLARS)

#: The own-fold view columns kept for the Pillar_Components and the D6 window: COMPONENT_VIEW_COLUMNS
#: and the saved Par_Score of each row.
_PAR_SCORE = "par_score"
_VIEW_COLUMNS: tuple[str, ...] = (*pressure.COMPONENT_VIEW_COLUMNS, _PAR_SCORE)

#: Columns of ``balls`` that the cases read: the State_Builder's inputs (D6 windows), the players,
#: the Competition_Type (D13) and the columns of the Free_Hit_Premium frame (D8).
_BALL_COLUMNS: tuple[str, ...] = tuple(
    dict.fromkeys((*state_module.INPUT_COLUMNS, "comp_type", "is_super_over", "excluded", "free_hit"))
)
_PREMIUM_BALL_COLUMNS: tuple[str, ...] = (
    "row", "match_id", "innings", "cls", "is_super_over", "excluded", "free_hit", "runs",
)  # fmt: skip
_VALUE_FRAME_COLUMNS: tuple[str, ...] = ("row", "match_id", "cls", "nbv_bat", "nbv_bowl", CELL_COLUMN)
_JOINT_CELL = "joint_cell"
_N_CELLS = value_module.N_CELLS
_N_COMP_TYPES = len(reconstruct.COMPETITION_TYPES)
_FIRST_INNINGS = 1


# ================================================================================ argument checks

def _number(value: Any, what: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise TypeError(f"{what} must be a finite number; got {value!r}")
    return float(value)


def _whole(value: Any, what: str, least: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{what} must be a whole number; got {value!r}")
    if int(value) < least:
        raise ValueError(f"{what} must be {least} or more; got {value!r}")
    return int(value)


def _switch(value: Any, what: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{what} must be True or False; got {value!r}")
    return bool(value)


def _pair(value: Any, what: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, (Sequence, np.ndarray)) or len(value) != 2:
        raise TypeError(f"{what} must be two numbers; got {value!r}")
    return _number(value[0], what), _number(value[1], what)


def _names(value: Any, what: str, allowed: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Collection):
        raise TypeError(f"{what} must be a collection of names such as {tuple(allowed[:1])!r}; got {value!r}")
    names = list(value)
    unknown = [name for name in names if name not in allowed]
    if unknown:
        raise ValueError(f"{what} names {unknown!r}, which are not among {', '.join(allowed)}")
    if len(set(names)) != len(names):
        raise ValueError(f"{what} names a value more than once: {names!r}")
    return tuple(name for name in allowed if name in names)


def _min_balls(value: Any) -> dict[str, int]:
    if isinstance(value, Mapping):
        if set(value) != set(ROLES):
            raise ValueError(f"min_balls must give a count for each Role ({', '.join(ROLES)}); got {sorted(value)}")
        return {role: _whole(value[role], f"min_balls[{role!r}]", 1) for role in ROLES}
    number = _whole(value, "min_balls", 1)
    return {role: number for role in ROLES}


def _checked_value(key: str, value: Any) -> Any:
    """``value`` of override ``key``, checked and normalised (module docstring, "Override vocabulary")."""
    if key in ("window", "new_batter_cap"):
        return _whole(value, key, 1)
    if key == "live_band":
        lower, upper = _pair(value, "live_band (D1)")
        if not 0.0 <= lower <= upper <= 1.0:
            raise ValueError(f"live_band (D1) must satisfy 0 <= lower <= upper <= 1; got {value!r}")
        return (lower, upper)
    if key == "alpha":
        alpha = _number(value, "alpha (D2)")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha (D2) must lie within [0, 1]; got {value!r}")
        return alpha
    if key == "share":
        share = _number(value, "share (D3)")
        if not 0.0 < share < 1.0:
            raise ValueError(f"share (D3) must lie strictly between 0 and 1; got {value!r}")
        return share
    if key == "scope":
        if value not in pressure.D4_SCOPES:
            raise ValueError(f"scope (D4) must be one of {', '.join(map(repr, pressure.D4_SCOPES))}; got {value!r}")
        return str(value)
    if key == "without":
        return _names(value, "without", _PILLAR_NAMES)
    if key in ("opposition", "retire_out_penalty", "zscore"):
        return _switch(value, key)
    if key == "phase_splits":
        first, second = _pair(value, "phase_splits (D8)")
        if not 0.0 < first < second:
            raise ValueError(f"phase_splits (D8) must satisfy 0 < s1 < s2; got {value!r}")
        return (first, second)
    if key == "zero_features":
        return _names(value, "zero_features", ZERO_FEATURES)
    if key == "half_life_years":
        if value is None:
            return None
        years = _number(value, "half_life_years (D9)")
        if years <= 0.0:
            raise ValueError(f"half_life_years (D9) must be above 0, or None for no decay; got {value!r}")
        return years
    if key == "knockout_weight":
        weight = _number(value, "knockout_weight (D10)")
        if weight <= 0.0:
            raise ValueError(f"knockout_weight (D10) must be above 0; got {value!r}")
        return weight
    if key == "min_balls":
        return _min_balls(value)
    if key == "replacement_pct":
        pct = _number(value, "replacement_pct (D12)")
        if not 0.0 <= pct <= 100.0:
            raise ValueError(f"replacement_pct (D12) must lie within [0, 100]; got {value!r}")
        return pct
    if key == "cells":
        if value not in D13_CELLS:
            raise ValueError(f"cells (D13) must be one of {', '.join(map(repr, D13_CELLS))}; got {value!r}")
        return str(value)
    if key == "blend":
        if value is None:
            return None
        w_l = _number(value, "blend (w_L)")
        if not 0.0 <= w_l <= 1.0:
            raise ValueError(f"blend (w_L) must lie within [0, 1]; got {value!r}")
        return w_l
    raise ValueError(f"unknown override {key!r}; the keys are {', '.join(OVERRIDE_LAYERS)}")


def _hashable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _hashable(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(item) for item in value)
    return value


def _read_only(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


# ================================================================================ CaseSpec

@dataclass(frozen=True)
class CaseSpec:
    """One Robustness_Case (module docstring, "Override vocabulary"; design, ``CaseSpec``).

    ``label``: the case label of the robustness table (non-empty); ``target_id``: the component
    or Design_Choice ID it changes (``"D2"``, ``"F17"``, ...); ``setting``: the setting used, as
    text; ``layer``: one of LAYERS, the layer of every key of ``overrides``; ``overrides``: key ->
    value of the override vocabulary (OVERRIDE_LAYERS); ``kind``: one of CASE_KINDS, the
    robustness-table ``kind`` (``all_cases`` sets it), or None for a case outside Requirement 21,
    such as an identity check. The overrides are checked and normalised (sequences become tuples,
    ``min_balls`` a {Role: count} mapping) and kept read-only. Raises TypeError or ValueError for a
    field of the wrong kind, an unknown key, a key of another layer or a value out of range.
    """

    label: str
    target_id: str
    setting: str
    layer: str
    overrides: Mapping[str, Any] = field(default_factory=dict)
    kind: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise TypeError(f"CaseSpec label must be a non-empty string; got {self.label!r}")
        for name in ("target_id", "setting"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"CaseSpec {name} must be a string; got {getattr(self, name)!r}")
        if self.layer not in LAYERS:
            raise ValueError(f"CaseSpec layer must be one of {', '.join(LAYERS)}; got {self.layer!r}")
        if self.kind is not None and self.kind not in CASE_KINDS:
            raise ValueError(f"CaseSpec kind must be one of {', '.join(CASE_KINDS)} or None; got {self.kind!r}")
        if not isinstance(self.overrides, Mapping):
            raise TypeError(f"CaseSpec overrides must map override keys to values; got {type(self.overrides).__name__}")
        checked: dict[str, Any] = {}
        for key, item in self.overrides.items():
            if key not in OVERRIDE_LAYERS:
                raise ValueError(
                    f"CaseSpec {self.label!r}: unknown override {key!r}; the keys are {', '.join(OVERRIDE_LAYERS)}"
                )
            if OVERRIDE_LAYERS[key] != self.layer:
                raise ValueError(
                    f"CaseSpec {self.label!r}: override {key!r} belongs to the {OVERRIDE_LAYERS[key]} layer, not the "
                    f"case's {self.layer} layer; a case changes one layer"
                )
            checked[key] = _checked_value(key, item)
        object.__setattr__(self, "overrides", MappingProxyType(checked))

    def __hash__(self) -> int:
        return hash(
            (self.label, self.target_id, self.setting, self.layer, _hashable(dict(self.overrides)), self.kind)
        )


# ================================================================================ predictions

@dataclass(frozen=True, eq=False)
class Predictions:
    """Model outputs of every innings 1-2 Delivery, in ``state`` order, that a recomputation reads.

    ``wp_pre`` (n,) and ``wp_post`` (n, 9; CLASSES order): WP after the terminal rules; ``m4``:
    variant -> (n, 9) M4 probabilities (``repredict`` gives all six); ``omega`` (n,): Wicket_Cost;
    ``p_extras`` (n, 2): the Extras_Model Wide and No_Ball probabilities; ``view``: ``row`` and
    the own-fold view columns of the Pillar_Components with ``par_score``; ``seconds``: the time
    of each re-prediction step; ``notes``: the renormalisation warnings of the predictions. The
    arrays are read-only.
    """

    wp_pre: np.ndarray
    wp_post: np.ndarray
    m4: Mapping[str, np.ndarray]
    omega: np.ndarray
    p_extras: np.ndarray
    view: pd.DataFrame
    seconds: Mapping[str, float] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        n_rows = len(self.wp_pre)
        shapes = {"wp_pre": (n_rows,), "wp_post": (n_rows, N_CLASSES), "omega": (n_rows,), "p_extras": (n_rows, 2)}
        for name, shape in shapes.items():
            array = np.array(getattr(self, name), dtype=np.float64)
            if array.shape != shape:
                raise ValueError(f"Predictions.{name} must have shape {shape}; got {array.shape}")
            object.__setattr__(self, name, _read_only(array))
        variants = {}
        for variant, values in self.m4.items():
            array = np.array(values, dtype=np.float64)
            if array.shape != (n_rows, N_CLASSES):
                raise ValueError(
                    f"Predictions.m4[{variant!r}] must have shape {(n_rows, N_CLASSES)}; got {array.shape}"
                )
            variants[variant] = _read_only(array)
        object.__setattr__(self, "m4", MappingProxyType(variants))
        object.__setattr__(self, "seconds", MappingProxyType(dict(self.seconds)))
        object.__setattr__(self, "notes", tuple(self.notes))


class _Notes:
    """Collects the warnings that ``common.renormalise`` writes (a ``common.WarningLog``)."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def warning(self, message: object) -> None:
        self.messages.append(str(message))


# ================================================================================ BaseArtifacts

@dataclass(frozen=True, eq=False)
class BaseArtifacts:
    """The cached arrays of the Main_Ranking's run (module docstring, "Layers and caches"); build with ``build``.

    Delivery arrays follow the ``state`` rows in ``state`` order; ``rated_at`` gives the positions
    of the rated Deliveries (innings 1-2 of the matches used for player rating, the rows of
    ``inputs.d`` and of the ``value`` table). ``main`` maps each Role to the Main_Ranking table
    of the ``rating`` checkpoint (RANKING_COLUMNS), and ``rankings`` to the same ranking
    recomputed from the cache (a RoleRanking, bit for bit equal). ``defaults`` holds the
    Main_Ranking's value of every override key. ``seconds`` times each build step.
    """

    settings: Settings = field(repr=False)
    store: CheckpointStore = field(repr=False)
    n_folds: int
    ks: tuple[int, ...]
    row: np.ndarray = field(repr=False)
    match_id: np.ndarray = field(repr=False)
    innings: np.ndarray = field(repr=False)
    fold: np.ndarray = field(repr=False)
    roles: Mapping[str, np.ndarray] = field(repr=False)
    innings_groups: np.ndarray = field(repr=False)
    rated_at: np.ndarray = field(repr=False)
    state: pd.DataFrame = field(repr=False)
    balls: pd.DataFrame = field(repr=False)
    own_view: pd.DataFrame = field(repr=False)
    mean_wickets: np.ndarray = field(repr=False)
    candidates: Mapping[str, str]
    wp_pre: np.ndarray = field(repr=False)
    wp_post: np.ndarray = field(repr=False)
    p_avg: np.ndarray = field(repr=False)
    live: np.ndarray = field(repr=False)
    li: np.ndarray = field(repr=False)
    components: pd.DataFrame = field(repr=False)
    pillar_ranks: pd.DataFrame = field(repr=False)
    stresses: pd.DataFrame = field(repr=False)
    pis: Mapping[str, np.ndarray] = field(repr=False)
    flags: Mapping[str, Any] = field(repr=False)
    pressure_flags: pd.DataFrame = field(repr=False)
    inputs: ValueInputs = field(repr=False)
    value: pd.DataFrame = field(repr=False)
    charges: pd.DataFrame = field(repr=False)
    charge_frame: pd.DataFrame = field(repr=False)
    comp_code: np.ndarray = field(repr=False)
    premium_frame: pd.DataFrame = field(repr=False)
    premium_state_at: np.ndarray = field(repr=False)
    matches: pd.DataFrame = field(repr=False)
    competitions: pd.DataFrame = field(repr=False)
    teams: pd.DataFrame = field(repr=False)
    xi: pd.DataFrame = field(repr=False)
    full_members: tuple[str, ...] = field(repr=False)
    recent_years: tuple[int, ...] = field(repr=False)
    rules: Mapping[str, Any]
    weights: pd.DataFrame = field(repr=False)
    contrib: rating.Contributions = field(repr=False)
    counts: pd.DataFrame = field(repr=False)
    eligibility_table: pd.DataFrame = field(repr=False)
    min_balls: Mapping[str, int]
    min_matches: int
    role_data: Mapping[str, rating.RoleData] = field(repr=False)
    rankings: Mapping[str, RoleRanking] = field(repr=False)
    main: Mapping[str, pd.DataFrame] = field(repr=False)
    defaults: Mapping[str, Any]
    d5_weights: str
    seconds: Mapping[str, float]
    _cache: dict[Any, Any] = field(default_factory=dict, repr=False)

    @property
    def n_rows(self) -> int:
        """The Deliveries of innings 1 and 2: the ``state`` rows."""
        return len(self.row)

    def n_eligible(self, role: str) -> int:
        """The Main_Ranking's Eligible_Players of ``role``."""
        return int(self.main[role]["eligible"].sum())

    @classmethod
    def build(cls, settings: Settings, store: CheckpointStore) -> BaseArtifacts:
        """Read a completed Run's checkpoints and cache every layer (module docstring, "Layers and caches").

        ``settings``: the Run's Settings, whose Design_Choices are the case defaults; ``store``:
        the CheckpointStore of the Run, read only. Raises StageError when the ``rating``
        checkpoint's Main_Ranking was computed with other eligibility settings or D12 than the
        Config's, or when the cached arrays no longer reproduce it bit for bit; ValueError for
        checkpoints that do not hold the same rows; and the errors of the store.
        """
        return _build(settings, store)


def _timed(seconds: dict[str, float], name: str, started: float) -> float:
    now = time.perf_counter()
    seconds[name] = now - started
    return now


def _positions(ids: np.ndarray, wanted: np.ndarray, what: str) -> np.ndarray:
    index = pd.Index(ids)
    if not index.is_unique:
        raise ValueError(f"{what} must hold each row once")
    at = index.get_indexer(wanted)
    if (at < 0).any():
        raise ValueError(f"{int((at < 0).sum()):,} row(s) are missing from {what}; pass the checkpoints of one Run")
    return at


def _main_table(players: pd.DataFrame, role: str) -> pd.DataFrame:
    mine = players.iloc[np.flatnonzero((players["role"].astype("string") == role).to_numpy(dtype=bool))]
    mine = mine.sort_values("player_id", kind="stable")
    return pd.DataFrame(
        {
            "player_id": mine["player_id"].to_numpy(dtype=np.int64),
            "eligible": mine["eligible"].to_numpy(dtype=bool),
            "rank": pd.array(mine["rank"].to_numpy(dtype=np.float64, na_value=np.nan), dtype="Float64").astype("Int64"),
            "final_score": mine["final_score"].to_numpy(dtype=np.float64),
        },
        columns=list(RANKING_COLUMNS),
    )


def _same_values(a: Any, b: Any) -> bool:
    return bool(np.array_equal(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), equal_nan=True))


def _role_records(table: pd.DataFrame, what: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for role in ROLES:
        found = table.iloc[np.flatnonzero((table["role"].astype("string") == role).to_numpy(dtype=bool))]
        if len(found) != 1:
            raise StageError(
                f"the rating checkpoint's {what} must hold one row of the {role} Role; it holds {len(found)}"
            )
        records[role] = found.iloc[0].to_dict()
    return records


def _build(settings: Settings, store: CheckpointStore) -> BaseArtifacts:
    seconds: dict[str, float] = {}
    started = tick = time.perf_counter()
    n_folds = int(settings.get(_D15))
    ks = tuple(sorted(spec.sandbox_k for spec in folds.sandbox_plan(n_folds) if spec.kind == folds.OUTER))
    defaults: dict[str, Any] = {key: _checked_value(key, settings.get(path)) for key, path in _CONFIG_KEYS.items()}
    defaults.update({"without": (), "opposition": True, "zero_features": (), "blend": None, "zscore": False})
    d5_weights = settings.get(_D5)

    # Delivery frame: the state rows in state order, with their balls columns and folds.
    balls_all = store.load_table(reconstruct.STAGE_NAME, reconstruct.BALLS_TABLE)
    state = store.load_table(state_module.STAGE_NAME, state_module.STATE_TABLE)
    fold_table = store.load_table(folds.STAGE_NAME, folds.FOLD_TABLE)
    row = state["row"].to_numpy(dtype=np.int64)
    aligned = balls_all.iloc[_positions(balls_all["row"].to_numpy(dtype=np.int64), row, "balls")].reset_index(drop=True)
    match_id = aligned["match_id"].to_numpy(dtype=np.int64)
    innings = aligned["innings"].to_numpy(dtype=np.int64)
    fold = folds.fold_of(match_id, fold_table).astype(np.int64)
    roles = pressure.role_masks(aligned["cls"])
    innings_groups = pressure.rank_groups(innings, pressure.D4_INNINGS_GROUP)
    balls = balls_all[list(_BALL_COLUMNS)].copy()
    premium_frame = folds.attach_folds(balls_all[list(_PREMIUM_BALL_COLUMNS)].copy(), fold_table)
    premium_state_at = pd.Index(row).get_indexer(premium_frame["row"].to_numpy(dtype=np.int64))
    del balls_all
    tick = _timed(seconds, "deliveries", tick)

    # State layer: the own-fold view columns of the components (with the saved Par_Score) and the F20(a) means.
    view_by_k = {k: store.load_table(remaining_runs.STAGE_NAME, views.view_table(k)) for k in ks}
    own_view = pressure.own_fold_view(view_by_k, fold, row, columns=_VIEW_COLUMNS)
    del view_by_k
    lb = state["lb_bowled"]
    table = pressure.train_wicket_means(
        fold=fold, innings=innings, lb_bowled=lb, wickets=state["wickets"], training=common.training_mask(aligned)
    )
    mean_wickets = pressure.wicket_means_for(table, fold=fold, innings=innings, lb_bowled=lb)
    tick = _timed(seconds, "state_layer", tick)

    # Pressure layer: the Primary_Models' predictions and the saved pressure table.
    primary = model_choice.load_primary(store, variants=(AVG_BOTH,))
    in_order = all(
        np.array_equal(got, wanted) for got, wanted in ((primary.row, row), (primary.innings, innings), (primary.fold, fold))
    )
    if not in_order:
        raise ValueError(
            "the Primary_Models' predictions do not hold the state rows in order; pass the checkpoints of one Run"
        )
    table = store.load_table(pressure.STAGE_NAME, pressure.PRESSURE_TABLE)
    if not np.array_equal(table["row"].to_numpy(dtype=np.int64), row):
        raise ValueError("the pressure table does not hold the state rows in order; pass the checkpoints of one Run")
    if not _same_values(table["wp_pre"], primary.wp_pre):
        raise StageError(
            "the pressure table's WP is not the Primary_Models' saved WP; rerun the Pipeline from the pressure stage"
        )
    ids = table[["row"]].astype({"row": np.int32})
    components = pd.concat([ids, table[list(pressure.COMPONENT_COLUMNS)]], axis=1)
    pillar_ranks = pd.concat([ids, table[list(pressure.PILLAR_COLUMNS)]], axis=1)
    stresses = pd.concat([ids, table[list(pressure.STRESS_COLUMNS)]], axis=1)
    pis = {BATTER_ROLE: _read_only(table["pi_bat"].to_numpy(dtype=np.float64)),
           BOWLER_ROLE: _read_only(table["pi_bowl"].to_numpy(dtype=np.float64))}  # fmt: skip
    flags = {BATTER_ROLE: table["pressure_bat"].array.copy(), BOWLER_ROLE: table["pressure_bowl"].to_numpy(dtype=bool)}
    pressure_frame = pd.DataFrame(
        {"row": row, "pressure_bat": flags[BATTER_ROLE].copy(), "pressure_bowl": flags[BOWLER_ROLE].copy()}
    )
    live = _read_only(table["live"].to_numpy(dtype=bool))
    li = _read_only(table["li"].to_numpy(dtype=np.float64))
    del table
    tick = _timed(seconds, "pressure_layer", tick)

    # Value layer: the value stage's inputs, its saved NBVs and charges.
    inputs = value_module.load_value_inputs(store, variants=RANKING_VARIANTS)
    d_rows = inputs.d["row"].to_numpy(dtype=np.int64)
    rated_at = _positions(row, d_rows, "the state rows")
    if (np.diff(rated_at) <= 0).any():
        raise ValueError("the value inputs do not hold the rated Deliveries in state order")
    saved_value = store.load_table(value_module.STAGE_NAME, value_module.VALUE_TABLE)
    if not np.array_equal(saved_value["row"].to_numpy(dtype=np.int64), d_rows):
        raise ValueError("the value table does not hold the rated Deliveries in order; pass the checkpoints of one Run")
    value_frame = saved_value[list(_VALUE_FRAME_COLUMNS)].copy()
    charges = store.load_table(value_module.STAGE_NAME, value_module.DISMISSAL_CHARGES_TABLE)
    charge_frame = inputs.d[list(value_module.DISMISSAL_CHARGE_INPUT_COLUMNS)]
    comp_text = aligned["comp_type"].astype("string").to_numpy(dtype=object)[rated_at]
    comp_code = np.full(len(rated_at), -1, dtype=np.int64)
    for code, name in enumerate(reconstruct.COMPETITION_TYPES):
        comp_code[comp_text == name] = code
    if (comp_code < 0).any():
        raise ValueError(f"{int((comp_code < 0).sum()):,} rated Deliveries have no Competition_Type")
    del aligned, saved_value
    tick = _timed(seconds, "value_layer", tick)

    # Rating layer: weights, Contributions, counts, eligibility at the Main_Ranking's minimums, rankings.
    matches = store.load_table(_LOAD_STAGE, "matches")
    competitions = store.load_table(_LOAD_STAGE, "competitions")
    teams = store.load_table(_LOAD_STAGE, "teams")
    xi = store.load_table(_LOAD_STAGE, "playing_eleven")
    full_members = tuple(settings.get(_FULL_MEMBERS))
    recent_years = tuple(settings.get(_RECENT_YEARS))
    rules = {
        "recent_rule": _switch(settings.get(_RECENT_RULE), _RECENT_RULE),
        "min_years": settings.get(_MIN_YEARS),
        "min_share": settings.get(_MIN_SHARE),
    }
    report = _role_records(store.load_table(rating.STAGE_NAME, rating.ELIGIBILITY_REPORT_TABLE), "eligibility_report")
    summary = _role_records(store.load_table(rating.STAGE_NAME, rating.ROLE_SUMMARY_TABLE), "role_summary")
    rerun = "rerun the Pipeline from the rating stage"
    min_matches = int(report[BATTER_ROLE]["min_pressure_matches"])
    for role in ROLES:
        if int(report[role]["min_pressure_matches"]) != int(settings.get(_D18_DEFAULT)):
            raise StageError(
                f"the rating checkpoint's {role} Main_Ranking used the minimum Pressure_Match count "
                f"{int(report[role]['min_pressure_matches'])}, not the Config's {_D18_DEFAULT}; {rerun}"
            )
        if bool(report[role]["recent_rule_on"]) != rules["recent_rule"]:
            raise StageError(
                f"the rating checkpoint's {role} Main_Ranking used another Recent_Activity_Rule setting; {rerun}"
            )
        if float(summary[role]["replacement_pct"]) != defaults["replacement_pct"]:
            raise StageError(f"the rating checkpoint's {role} Main_Ranking used another D12 percentile; {rerun}")
    min_balls = {role: int(report[role]["min_pressure_balls"]) for role in ROLES}
    defaults["min_balls"] = dict(min_balls)
    weights = rating.match_weights(
        matches, competitions, teams, half_life_years=defaults["half_life_years"],
        knockout_weight=defaults["knockout_weight"], full_members=full_members,
    )  # fmt: skip
    contrib = rating.build_contributions(value_frame, pressure_frame, charges, balls, weights)
    counts = rating.eligibility_counts(contrib, xi, matches, recent_years=recent_years)
    eligibility_table = rating.eligibility(counts, min_balls=min_balls, min_matches=min_matches, **rules)
    players = store.load_table(rating.STAGE_NAME, rating.PLAYER_ESTIMATES_TABLE)
    main = {role: _main_table(players, role) for role in ROLES}
    role_data: dict[str, rating.RoleData] = {}
    rankings: dict[str, RoleRanking] = {}
    for role in ROLES:
        ids = _eligible_ids(eligibility_table, role)
        role_data[role] = rating.prepare_role(contrib, role, ids)
        rankings[role] = rating.rank_role(role_data[role], replacement_pct=defaults["replacement_pct"])
        saved = main[role].iloc[np.flatnonzero(main[role]["eligible"].to_numpy(dtype=bool))]
        if not (
            np.array_equal(saved["player_id"].to_numpy(dtype=np.int64), rankings[role].player_id)
            and _same_values(saved["final_score"], rankings[role].final_score)
            and _same_values(saved["rank"].to_numpy(dtype=np.float64, na_value=np.nan), rankings[role].rank)
        ):
            raise StageError(
                f"the cached checkpoints no longer reproduce the {role} Main_Ranking of the rating checkpoint bit for "
                f"bit; {rerun}"
            )
    tick = _timed(seconds, "rating_layer", tick)
    seconds["total"] = tick - started

    for array in (row, match_id, innings, fold, innings_groups, rated_at, mean_wickets, comp_code, premium_state_at):
        _read_only(array)
    for mask in roles.values():
        _read_only(mask)
    return BaseArtifacts(
        settings=settings,
        store=store,
        n_folds=n_folds,
        ks=ks,
        row=row,
        match_id=match_id,
        innings=innings,
        fold=fold,
        roles=MappingProxyType(roles),
        innings_groups=innings_groups,
        rated_at=rated_at,
        state=state,
        balls=balls,
        own_view=own_view,
        mean_wickets=mean_wickets,
        candidates=MappingProxyType(dict(primary.candidates)),
        wp_pre=primary.wp_pre,
        wp_post=primary.wp_post,
        p_avg=primary.m4[AVG_BOTH],
        live=live,
        li=li,
        components=components,
        pillar_ranks=pillar_ranks,
        stresses=stresses,
        pis=MappingProxyType(pis),
        flags=MappingProxyType(flags),
        pressure_flags=pressure_frame,
        inputs=inputs,
        value=value_frame,
        charges=charges,
        charge_frame=charge_frame,
        comp_code=comp_code,
        premium_frame=premium_frame,
        premium_state_at=premium_state_at,
        matches=matches,
        competitions=competitions,
        teams=teams,
        xi=xi,
        full_members=full_members,
        recent_years=recent_years,
        rules=MappingProxyType(rules),
        weights=weights,
        contrib=contrib,
        counts=counts,
        eligibility_table=eligibility_table,
        min_balls=MappingProxyType(min_balls),
        min_matches=min_matches,
        role_data=MappingProxyType(role_data),
        rankings=MappingProxyType(rankings),
        main=MappingProxyType(main),
        defaults=MappingProxyType(defaults),
        d5_weights=d5_weights,
        seconds=MappingProxyType(seconds),
    )


def _eligible_ids(table: pd.DataFrame, role: str) -> np.ndarray:
    mine = (table["role"].astype("string") == role).to_numpy(dtype=bool) & table["eligible"].to_numpy(dtype=bool)
    return np.sort(table["player_id"].to_numpy(dtype=np.int64)[mine])


def default_overrides(base: BaseArtifacts, layer: str) -> dict[str, Any]:
    """Every override key of ``layer`` at its Main_Ranking value (``zero_features`` aside; module docstring).

    A CaseSpec of ``layer`` with these overrides reruns every step of the layer and below, and
    must reproduce the Main_Ranking (Requirement 21, criterion 9). Raises ValueError for an
    unknown layer.
    """
    if layer not in LAYERS:
        raise ValueError(f"layer must be one of {', '.join(LAYERS)}; got {layer!r}")
    return {
        key: base.defaults[key]
        for key, home in OVERRIDE_LAYERS.items()
        if home == layer and key != "zero_features"
    }


# ================================================================================ CaseResult

@dataclass(frozen=True, eq=False)
class CaseResult:
    """The result of one Robustness_Case or prediction rerun (module docstring, "Results").

    ``label`` and ``layer`` name it (``layer`` is PREDICTIONS_LAYER for run_with_predictions);
    ``case`` is its CaseSpec (None for run_with_predictions). Per Role: ``rankings`` (the ranking
    table, RANKING_COLUMNS), ``n_eligible``, ``min_balls`` (the minimum Pressure_Ball count
    applied) and ``blend`` ((w_L, w_U) behind the Final_Score). ``steps`` lists the steps
    recomputed, in order, ``timings`` their seconds and ``seconds`` the whole case;
    ``warnings`` are the Rating_Engine warnings of the case's rankings and ``notes`` the
    re-prediction warnings.
    """

    label: str
    layer: str
    case: CaseSpec | None
    rankings: Mapping[str, pd.DataFrame]
    n_eligible: Mapping[str, int]
    min_balls: Mapping[str, int]
    blend: Mapping[str, tuple[float, float]]
    steps: tuple[str, ...]
    timings: Mapping[str, float]
    seconds: float
    warnings: tuple[RatingWarning, ...] = ()
    notes: tuple[str, ...] = ()

    def __repr__(self) -> str:
        counts = ", ".join(f"{role} {self.n_eligible[role]:,}" for role in ROLES)
        return f"CaseResult({self.label!r}, {self.layer}: Eligible_Players {counts}; {self.seconds:.2f} s)"


def compare_with_main(
    base: BaseArtifacts, result: CaseResult, role: str, *, names: Any = None, top_n: int = LIST_SIZE
) -> RankingComparison:
    """``ranks.compare_rankings`` of ``result``'s ``role`` ranking against the Main_Ranking, on ``final_score``.

    The Spearman correlation compares Final_Scores over the players ranked in both (Requirement
    21, criterion 5); ``names`` is passed through for the entering and leaving players.
    """
    return compare_rankings(base.main[role], result.rankings[role], top_n=top_n, on="final_score", names=names)


# ================================================================================ the engine

class _Inputs(NamedTuple):
    """The model outputs a case reads, and which of them differ from the cached ones."""

    wp_pre: np.ndarray  # all state rows
    wp_post: np.ndarray
    p_avg: np.ndarray
    m4_rated: Mapping[str, np.ndarray]  # the RANKING_VARIANTS on the rated rows
    omega_rated: np.ndarray
    p_extras_rated: np.ndarray
    view: pd.DataFrame
    changed: frozenset[str]  # of "wp", "m4", "omega", "extras", "view"


def _stored_inputs(base: BaseArtifacts) -> _Inputs:
    return _Inputs(
        wp_pre=base.wp_pre,
        wp_post=base.wp_post,
        p_avg=base.p_avg,
        m4_rated=base.inputs.m4,
        omega_rated=base.inputs.omega,
        p_extras_rated=base.inputs.p_extras,
        view=base.own_view,
        changed=frozenset(),
    )


class _Clock:
    def __init__(self) -> None:
        self.steps: list[str] = []
        self.timings: dict[str, float] = {}

    @contextmanager
    def step(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        yield
        self.timings[name] = self.timings.get(name, 0.0) + time.perf_counter() - started
        if name not in self.steps:
            self.steps.append(name)


def _window_frame(base: BaseArtifacts, window: int, view: pd.DataFrame) -> pd.DataFrame:
    """``row`` and WINDOW_INPUT_COLUMNS for a D6 window of ``window`` Legal_Balls (module docstring, step 1)."""
    memo = ("window", window) if view is base.own_view else None
    if memo is not None and memo in base._cache:
        return base._cache[memo]
    frame = state_module.recent_window(base.balls, window=window)
    frame = frame.iloc[_positions(frame["row"].to_numpy(dtype=np.int64), base.row, "the recent-window frame")]
    frame = frame.reset_index(drop=True)
    lb = base.state["lb_bowled"].to_numpy(dtype=np.int64)
    win_lb = frame["win_lb"].to_numpy(dtype=np.int64)
    par = view[_PAR_SCORE].to_numpy(dtype=np.float64)
    rate = np.full(base.n_rows, np.nan)
    one = np.flatnonzero(base.innings == _FIRST_INNINGS)
    if len(one):
        # Par_Score(L - w) is the saved par_score of the innings' first Delivery with L - w Legal_Balls bowled.
        code = pd.factorize(base.match_id[one], sort=False)[0].astype(np.int64)
        width = int(lb[one].max()) + 1
        key = code * width + lb[one]
        if (np.diff(key) < 0).any():
            raise ValueError("the innings-1 rows of the state table are not in innings and Legal_Ball order")
        target = code * width + (lb[one] - win_lb[one])
        at = np.searchsorted(key, target, side="left")
        if (at >= len(key)).any() or not np.array_equal(key[np.minimum(at, len(key) - 1)], target):
            raise ValueError("an innings-1 Delivery has no earlier Delivery of its innings at the window's start")
        end, start = par[one], par[one][at]
        if not (np.isfinite(end).all() and np.isfinite(start).all()):
            raise ValueError("the own-fold view holds an innings-1 row without a Par_Score")
        has = win_lb[one] > 0
        values = np.full(len(one), np.nan)
        values[has] = state_module.BALLS_PER_OVER * (end[has] - start[has]) / win_lb[one][has]
        rate[one] = values
    out = pd.DataFrame(
        {
            "row": base.row.astype(np.int32),
            "win_wkts": frame["win_wkts"].to_numpy(),
            "win_rr": frame["win_rr"].to_numpy(dtype=np.float64),
            "par_rate_win": rate,
        },
        columns=["row", *pressure.WINDOW_INPUT_COLUMNS],
    )
    if memo is not None:
        base._cache[memo] = out
    return out


def _premium_cells(base: BaseArtifacts, splits: tuple[float, float]) -> np.ndarray:
    """The innings x Phase cell of every ``state`` row under the D8 ``splits`` (int8)."""
    phase = state_module.phase_codes(base.state["ball_share"].to_numpy(dtype=np.float64), splits)
    return state_module.cell_codes(base.innings, phase)


def _joint_tables(means: pd.DataFrame, premium: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Copies of the cell tables keyed by the joint code Config cell x N_CELLS + new cell (module docstring)."""
    fold_column = folds.FOLD_COLUMN
    ks = np.unique(means[fold_column].to_numpy(dtype=np.int64))
    grids = np.meshgrid(ks, np.arange(_N_CELLS), np.arange(_N_CELLS), indexing="ij")
    fold, base_cell, new_cell = (grid.ravel() for grid in grids)
    joint = base_cell * _N_CELLS + new_cell
    by_base = value_module.cell_lookup(means, fold, base_cell)
    by_new = value_module.cell_lookup(premium, fold, new_cell)
    means_joint = pd.DataFrame({fold_column: fold, CELL_COLUMN: joint})
    for name in ("wide_runs", "nb_penalty_runs", "nb_bat_runs"):
        means_joint[name] = by_base[name].to_numpy(dtype=np.float64)
    premium_joint = pd.DataFrame({fold_column: fold, CELL_COLUMN: joint})
    premium_joint["free_hit_premium"] = by_new["free_hit_premium"].to_numpy(dtype=np.float64)
    return means_joint, premium_joint


def nbv_views(m4: Mapping[str, np.ndarray], free_hit: Any, *, opposition: bool) -> dict[str, np.ndarray]:
    """The M4 probabilities that each NBV view reads (module docstring, "Decisions"; ``opposition`` False).

    ``m4`` maps the RANKING_VARIANTS to (n, 9) arrays and ``free_hit`` gives each row's own
    Free_Hit flag. With ``opposition`` True each view reads its own variant, as the ``value``
    stage does. With False, ``batter_view`` and ``bowler_view`` read ``avg_both``, and
    ``batter_view_fh`` and ``bowler_view_nofh`` read ``avg_both`` on the rows whose own flag
    already is the forced one (a Free_Hit for ``batter_view_fh``, not a Free_Hit for
    ``bowler_view_nofh``, where the forced and the own-flag predictions are the same rows of one
    model) and keep their own variant on the other rows. Returns view -> array, new arrays where
    rows are mixed. Raises ValueError when ``m4`` lacks a variant or ``free_hit`` has another length.
    """
    absent = [name for name in RANKING_VARIANTS if name not in m4]
    if absent:
        raise ValueError(f"m4 lacks the variant(s) {', '.join(absent)}")
    if opposition:
        return {view: m4[view] for view in NBV_VARIANTS}
    flag = np.asarray(free_hit, dtype=bool)
    if flag.shape != (len(m4[AVG_BOTH]),):
        raise ValueError(f"free_hit must hold one flag per row ({len(m4[AVG_BOTH]):,}); got shape {flag.shape}")
    both = m4[AVG_BOTH]
    return {
        BATTER_VIEW: both,
        BATTER_VIEW_FH: np.where(flag[:, None], both, m4[BATTER_VIEW_FH]),
        BOWLER_VIEW: both,
        BOWLER_VIEW_NOFH: np.where(flag[:, None], m4[BOWLER_VIEW_NOFH], both),
    }


def _role_z(values: np.ndarray) -> np.ndarray:
    """Role_Z of each finite value among the finite values (module docstring, "Decisions"); NaN elsewhere."""
    out = np.full(len(values), np.nan)
    finite = np.isfinite(values)
    if finite.any():
        x = values[finite]
        out[finite] = 0.0 if x.max() == x.min() else (x - x.mean()) / x.std(ddof=0)
    return out


def _ranking_frame(
    players: np.ndarray, eligible: np.ndarray, ids: np.ndarray, rank: np.ndarray, final: np.ndarray
) -> pd.DataFrame:
    """One Role's RANKING_COLUMNS table: every player of ``players`` (ascending), ranks and scores of ``ids``."""
    rank_all = np.full(len(players), np.nan)
    final_all = np.full(len(players), np.nan)
    if len(ids):
        at = np.searchsorted(players, ids)
        if (at >= len(players)).any() or not np.array_equal(players[np.minimum(at, len(players) - 1)], ids):
            raise ValueError("an Eligible_Player of the case has no row in its eligibility table")
        rank_all[at] = rank
        final_all[at] = final
    return pd.DataFrame(
        {
            "player_id": players.astype(np.int64),
            "eligible": eligible.astype(bool),
            "rank": pd.array(rank_all, dtype="Float64").astype("Int64"),
            "final_score": final_all,
        },
        columns=list(RANKING_COLUMNS),
    )


class _Found(NamedTuple):
    """What _evaluate gives each Role: its ranking table, Eligible_Players, minimum count and blend."""

    rankings: dict[str, pd.DataFrame]
    n_eligible: dict[str, int]
    min_balls: dict[str, int]
    blends: dict[str, tuple[float, float]]
    warnings: list[RatingWarning]


def _evaluate(base: BaseArtifacts, overrides: Mapping[str, Any], inputs: _Inputs, clock: _Clock) -> _Found:
    """Recompute from the earliest changed input down to the ranks (module docstring, "Recomputation")."""
    given = set(overrides)
    plan = {**base.defaults, **overrides}
    changed = inputs.changed

    # Steps 1-2: the D6 window inputs and the Pillar_Components.
    window = None
    if "window" in given:
        with clock.step("window"):
            window = _window_frame(base, plan["window"], inputs.view)
    components = base.components
    if window is not None or "new_batter_cap" in given or "view" in changed:
        with clock.step("components"):
            components = pressure.pillar_components(
                base.state, inputs.view, innings=base.innings, mean_wickets=base.mean_wickets,
                new_batter_cap=plan["new_batter_cap"], window=window,
            )  # fmt: skip

    # Steps 3-8: Live_Balls, Stress_Pillars, Stress, LI, PI and the Pressure_Ball flags.
    live = base.live
    if "live_band" in given or "wp" in changed:
        with clock.step("live"):
            live = pressure.live_balls(inputs.wp_pre, plan["live_band"])
    groups = base.innings_groups
    if "scope" in given:
        groups = pressure.rank_groups(base.innings, plan["scope"])
    pillar_ranks = base.pillar_ranks
    if components is not base.components or live is not base.live or "scope" in given:
        with clock.step("pillars"):
            pillar_ranks = pressure.pillars(components, live=live, roles=base.roles, groups=groups)
    stresses = base.stresses
    if pillar_ranks is not base.pillar_ranks or "without" in given:
        with clock.step("stress"):
            stresses = pressure.stress(pillar_ranks, weights=base.d5_weights, without=plan["without"])
    li = base.li
    if "wp" in changed or "m4" in changed:
        with clock.step("leverage"):
            li = pressure.leverage(inputs.p_avg, inputs.wp_pre, inputs.wp_post).li
    pis = base.pis
    if stresses is not base.stresses or li is not base.li or live is not base.live or given & {"scope", "alpha"}:
        with clock.step("pressure_index"):
            pressure.finite_guard(
                wp=inputs.wp_pre, li=li, stresses=stresses, live=live, roles=base.roles, innings=base.innings
            )
            pis = {
                role: pressure.pressure_index(
                    li, stresses[_STRESS_OF[role]], live, base.roles[role], groups, plan["alpha"]
                )
                for role in ROLES
            }
    flags = base.flags
    if pis is not base.pis or "share" in given:
        with clock.step("pressure_flags"):
            # The share applies within each Role and Innings_Group, whatever the D4 scope (criterion 4).
            flags = {
                role: pressure.pressure_flags(pis[role], live, base.roles[role], base.innings_groups, plan["share"])
                for role in ROLES
            }
    flags_changed = flags is not base.flags

    # Steps 9-10: NBV of both Roles and the Dismissal_Charges.
    d, crv = base.inputs.d, base.inputs.crv
    omega, p_extras = inputs.omega_rated, inputs.p_extras_rated
    penalty = plan["retire_out_penalty"]
    value_frame = base.value
    nbv_bat = nbv_bowl = match_cells = None
    recompute_bat = bool(given & {"opposition", "retire_out_penalty"} or changed & {"m4", "omega"})
    recompute_bowl = bool(given & {"opposition", "phase_splits"} or changed & {"m4", "omega", "extras"})
    p: dict[str, np.ndarray] = {}
    if recompute_bat or recompute_bowl:
        p = nbv_views(inputs.m4_rated, d["free_hit"].to_numpy(dtype=bool), opposition=plan["opposition"])
    if recompute_bat:
        with clock.step("nbv_batter"):
            nbv_bat = value_module.batter_nbv(
                d, p[BATTER_VIEW], p[BATTER_VIEW_FH], crv, omega, retire_out_penalty=penalty
            ).nbv
    if recompute_bowl:
        with clock.step("nbv_bowler"):
            p_view, p_nofh = p[BOWLER_VIEW], p[BOWLER_VIEW_NOFH]
            if "phase_splits" in given:
                cells = _premium_cells(base, plan["phase_splits"])
                frame = base.premium_frame.copy()
                at = base.premium_state_at  # -1 for a Super_Over_Innings Delivery, which has no cell
                codes = np.where(at >= 0, cells[np.maximum(at, 0)], 0).astype(np.int8)
                frame[CELL_COLUMN] = pd.arrays.IntegerArray(codes, at < 0)
                premium = value_module.free_hit_premium(frame, CELL_COLUMN)
                means_joint, premium_joint = _joint_tables(base.inputs.means, premium)
                match_cells = cells[base.rated_at].astype(np.int64)
                bowl = d[list(value_module.BOWLER_NBV_COLUMNS)].copy()
                bowl[_JOINT_CELL] = d[CELL_COLUMN].to_numpy(dtype=np.int64) * _N_CELLS + match_cells
                nbv_bowl = value_module.bowler_nbv(
                    bowl, p_view, p_nofh, p_extras, crv, omega, means_joint, premium_joint, cells=_JOINT_CELL
                ).nbv
            else:
                nbv_bowl = value_module.bowler_nbv(
                    d, p_view, p_nofh, p_extras, crv, omega, base.inputs.means, base.inputs.premium
                ).nbv
    if nbv_bat is not None or nbv_bowl is not None or match_cells is not None:
        value_frame = base.value.copy()
        if nbv_bat is not None:
            value_frame["nbv_bat"] = nbv_bat
        if nbv_bowl is not None:
            value_frame["nbv_bowl"] = nbv_bowl
        if match_cells is not None:
            value_frame[CELL_COLUMN] = match_cells
    charges = base.charges
    if flags_changed or "retire_out_penalty" in given or "omega" in changed:
        with clock.step("charges"):
            frame = base.charge_frame.copy()
            frame["live"] = live[base.rated_at]
            frame["pressure_bat"] = pd.array(flags[BATTER_ROLE], dtype="boolean")[base.rated_at]
            charges = value_module.dismissal_charges(frame, omega, retire_out_penalty=penalty)

    # Steps 11-12: weights, Contributions and eligibility counts.
    weights = base.weights
    if given & {"half_life_years", "knockout_weight"}:
        with clock.step("weights"):
            weights = rating.match_weights(
                base.matches, base.competitions, base.teams, half_life_years=plan["half_life_years"],
                knockout_weight=plan["knockout_weight"], full_members=base.full_members,
            )  # fmt: skip
    contrib, counts = base.contrib, base.counts
    if weights is not base.weights or value_frame is not base.value or charges is not base.charges or flags_changed or (
        "cells" in given
    ):
        with clock.step("contributions"):
            frame = value_frame
            if plan["cells"] == D13_COMP_TYPE:
                frame = value_frame.copy()
                frame[CELL_COLUMN] = value_frame[CELL_COLUMN].to_numpy(dtype=np.int64) * _N_COMP_TYPES + base.comp_code
            flag_frame = base.pressure_flags
            if flags_changed:
                flag_frame = pd.DataFrame(
                    {
                        "row": base.row,
                        "pressure_bat": pd.array(flags[BATTER_ROLE], dtype="boolean"),
                        "pressure_bowl": np.asarray(flags[BOWLER_ROLE], dtype=bool),
                    }
                )
            contrib = rating.build_contributions(frame, flag_frame, charges, base.balls, weights)
        with clock.step("counts"):
            counts = rating.eligibility_counts(contrib, base.xi, base.matches, recent_years=base.recent_years)

    # Step 13: eligibility at the Main_Ranking's minimum counts, or the D11 case's.
    table = base.eligibility_table
    min_balls = dict(plan["min_balls"])
    if counts is not base.counts or "min_balls" in given:
        with clock.step("eligibility"):
            table = rating.eligibility(counts, min_balls=min_balls, min_matches=base.min_matches, **base.rules)

    # Steps 14-15: ranking per Role, then any re-blend.
    rankings: dict[str, pd.DataFrame] = {}
    n_eligible: dict[str, int] = {}
    blends: dict[str, tuple[float, float]] = {}
    notes: list[RatingWarning] = []
    for role in ROLES:
        mine = np.flatnonzero((table["role"].astype("string") == role).to_numpy(dtype=bool))
        players = table["player_id"].to_numpy(dtype=np.int64)[mine]
        eligible = table["eligible"].to_numpy(dtype=bool)[mine]
        order = np.argsort(players, kind="stable")
        players, eligible = players[order], eligible[order]
        ids = players[eligible]
        n_eligible[role] = len(ids)
        if not len(ids):
            rankings[role] = _ranking_frame(players, eligible, ids, np.zeros(0), np.zeros(0))
            blends[role] = (math.nan, math.nan)
            continue
        same_players = contrib is base.contrib and np.array_equal(ids, base.rankings[role].player_id)
        if same_players and "replacement_pct" not in given and table is base.eligibility_table:
            ranking = base.rankings[role]
        else:
            with clock.step("ranking"):
                data = base.role_data[role] if same_players else rating.prepare_role(contrib, role, ids)
                ranking = rating.rank_role(data, replacement_pct=plan["replacement_pct"])
        notes.extend(ranking.warnings)
        final, rank = ranking.final_score, ranking.rank
        w_l, w_u = ranking.w_L, ranking.w_U
        if plan["blend"] is not None or plan["zscore"]:
            with clock.step("reblend"):
                if plan["blend"] is not None:
                    w_l, w_u = plan["blend"], 1.0 - plan["blend"]
                elif plan["zscore"]:
                    w_l, w_u = base.rankings[role].w_L, base.rankings[role].w_U  # the Main_Ranking's Blend_Weights
                clutch, uplift = ranking.shrunk_clutch, ranking.shrunk_uplift
                if plan["zscore"]:
                    clutch, uplift = _role_z(clutch), _role_z(uplift)
                final = rating.final_scores(clutch, uplift, w_l, w_u)
                rank = rating.rank_order(final, ranking.shrunk_clutch, ranking.player_id)
        blends[role] = (float(w_l), float(w_u))
        rankings[role] = _ranking_frame(players, eligible, ranking.player_id, rank, final)
    return _Found(rankings=rankings, n_eligible=n_eligible, min_balls=min_balls, blends=blends, warnings=notes)


def _result(
    label: str, layer: str, case: CaseSpec | None, found: _Found, clock: _Clock, started: float,
    notes: Sequence[str] = (),
) -> CaseResult:
    return CaseResult(
        label=label,
        layer=layer,
        case=case,
        rankings=MappingProxyType(found.rankings),
        n_eligible=MappingProxyType(found.n_eligible),
        min_balls=MappingProxyType(found.min_balls),
        blend=MappingProxyType(found.blends),
        steps=tuple(clock.steps),
        timings=MappingProxyType(dict(clock.timings)),
        seconds=time.perf_counter() - started,
        warnings=tuple(found.warnings),
        notes=tuple(notes),
    )


def _rated_inputs(base: BaseArtifacts, predictions: Predictions, changed: frozenset[str]) -> _Inputs:
    at = base.rated_at
    m4 = {name: predictions.m4[name][at] for name in RANKING_VARIANTS}
    return _Inputs(
        wp_pre=predictions.wp_pre,
        wp_post=predictions.wp_post,
        p_avg=predictions.m4[AVG_BOTH],
        m4_rated=m4,
        omega_rated=predictions.omega[at],
        p_extras_rated=predictions.p_extras[at],
        view=predictions.view,
        changed=changed,
    )


def run_case(case: CaseSpec, base: BaseArtifacts) -> CaseResult:
    """Recompute ``case`` from its layer down to the ranks of both Roles (module docstring, "Recomputation").

    ``case`` is a CaseSpec and ``base`` the BaseArtifacts of the Main_Ranking's run. Settings the
    case does not name keep their Main_Ranking values; no model is refitted and no
    Bootstrap_Resample is drawn. A case with ``zero_features`` re-predicts with the saved outer
    Fold_Models first (``repredict``, memoised in ``base``). Returns a CaseResult. Raises
    TypeError for arguments of the wrong kind, and the errors of the functions it calls.
    """
    if not isinstance(case, CaseSpec):
        raise TypeError(f"case must be a CaseSpec; got {type(case).__name__}")
    if not isinstance(base, BaseArtifacts):
        raise TypeError(f"base must be the BaseArtifacts of BaseArtifacts.build; got {type(base).__name__}")
    started = time.perf_counter()
    clock = _Clock()
    inputs = _stored_inputs(base)
    notes: tuple[str, ...] = ()
    overrides = dict(case.overrides)
    if "zero_features" in overrides:
        with clock.step("repredict"):
            predictions = repredict(base, overrides.pop("zero_features"))
        inputs = _rated_inputs(base, predictions, frozenset({"wp", "m4", "omega", "extras", "view"}))
        notes = predictions.notes
    found = _evaluate(base, overrides, inputs, clock)
    return _result(case.label, case.layer, case, found, clock, started, notes)


def _probabilities(values: Any, shape: tuple[int, ...], what: str) -> np.ndarray:
    array = np.array(values, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{what} must have shape {shape}, one row per state row in state order; got {array.shape}")
    if not (np.isfinite(array).all() and (array >= 0.0).all() and (array <= 1.0).all()):
        raise ValueError(f"{what} holds a value that is missing, not finite or outside [0, 1]")
    return array


def run_with_predictions(
    base: BaseArtifacts,
    predictions: PrimaryPredictions | None = None,
    *,
    wp_pre: Any = None,
    wp_post: Any = None,
    m4: Mapping[str, Any] | None = None,
    label: str = "predictions",
) -> CaseResult:
    """The Main_Ranking recomputed with other stored predictions (module docstring, "Prediction overrides").

    ``predictions``: a PrimaryPredictions (``model_choice.load_predictions``) whose WP and M4
    arrays replace the Primary_Models'; it must hold the ``state`` rows in order and the
    RANKING_VARIANTS. ``wp_pre`` (n,), ``wp_post`` (n, 9) and ``m4`` (variant -> (n, 9), at
    least the variants replaced) override single arrays, in ``state`` order, and take precedence.
    Anything not given keeps the stored Primary_Models' arrays; anything given counts as changed.
    Every Design_Choice keeps its Config value and eligibility uses the Main_Ranking's minimum
    counts (Requirement 23, criterion 10). Returns a CaseResult of layer PREDICTIONS_LAYER.
    Raises TypeError or ValueError for predictions of another Run, the wrong shape or outside
    [0, 1], and the errors of the recomputation.
    """
    if not isinstance(base, BaseArtifacts):
        raise TypeError(f"base must be the BaseArtifacts of BaseArtifacts.build; got {type(base).__name__}")
    if not isinstance(label, str) or not label:
        raise TypeError(f"label must be a non-empty string; got {label!r}")
    started = time.perf_counter()
    n_rows = base.n_rows
    new_pre, new_post = base.wp_pre, base.wp_post
    variants: dict[str, np.ndarray] = {}
    changed: set[str] = set()
    if predictions is not None:
        if not isinstance(predictions, PrimaryPredictions):
            raise TypeError(f"predictions must be a model_choice.PrimaryPredictions; got {type(predictions).__name__}")
        if not (
            np.array_equal(predictions.row, base.row)
            and np.array_equal(predictions.innings, base.innings)
            and np.array_equal(predictions.fold, base.fold)
        ):
            raise ValueError("predictions do not hold the state rows of the BaseArtifacts in order; pass one Run's")
        missing = [name for name in RANKING_VARIANTS if name not in predictions.m4]
        if missing:
            raise ValueError(f"predictions lack the M4 variant(s) {', '.join(missing)} that the ranking reads")
        new_pre, new_post = predictions.wp_pre, predictions.wp_post
        variants.update({name: predictions.m4[name] for name in RANKING_VARIANTS})
        changed |= {"wp", "m4"}
    if wp_pre is not None:
        new_pre = _probabilities(wp_pre, (n_rows,), "wp_pre")
        changed.add("wp")
    if wp_post is not None:
        new_post = _probabilities(wp_post, (n_rows, N_CLASSES), "wp_post")
        changed.add("wp")
    if m4 is not None:
        if not isinstance(m4, Mapping):
            raise TypeError(f"m4 must map M4 variant names to arrays; got {type(m4).__name__}")
        unknown = sorted(set(m4) - set(outcome.VARIANT_NAMES))
        if unknown:
            raise ValueError(f"m4 names unknown variant(s) {', '.join(unknown)}")
        for name, values in m4.items():
            variants[name] = _probabilities(values, (n_rows, N_CLASSES), f"m4[{name!r}]")
        changed.add("m4")
    at = base.rated_at
    m4_rated = dict(base.inputs.m4)
    m4_rated.update({name: values[at] for name, values in variants.items() if name in RANKING_VARIANTS})
    inputs = _Inputs(
        wp_pre=new_pre,
        wp_post=new_post,
        p_avg=variants.get(AVG_BOTH, base.p_avg),
        m4_rated=m4_rated,
        omega_rated=base.inputs.omega,
        p_extras_rated=base.inputs.p_extras,
        view=base.own_view,
        changed=frozenset(changed),
    )
    clock = _Clock()
    found = _evaluate(base, {}, inputs, clock)
    return _result(label, PREDICTIONS_LAYER, None, found, clock, started)


# ================================================================================ re-prediction

def _fit_table(table: pd.DataFrame, fit_id: str, columns: Sequence[str]) -> pd.DataFrame:
    """One first-innings fit's rows of PAR_MEANS_TABLE or PAR_BIAS_TABLE, indexed by bucket 0 ... PAR_BUCKETS."""
    part = table.iloc[np.flatnonzero(table["fit_id"].astype(str).to_numpy(dtype=object) == fit_id)]
    if part.empty:
        raise ValueError(f"the remaining_runs par tables hold no row of the first-innings M2 fit {fit_id!r}")
    part = part.sort_values("bucket", kind="stable")
    return pd.DataFrame(
        {name: part[name].to_numpy() for name in columns},
        index=pd.Index(part["bucket"].to_numpy(dtype=np.int64), name="bucket"),
        columns=list(columns),
    )


def _m3_models(store: CheckpointStore, innings: int, family: str, n_folds: int) -> win_prob.CalibratedFolds:
    """The calibrated outer M3 models of ``family`` for ``innings``.

    The ``win_prob`` families (logistic, LightGBM) come from the ``win_prob`` checkpoint and the M3
    Alternative_Models (XGBoost, random forest; task 26.1) from the ``model_choice`` checkpoint,
    which saved their outer fits and calibrators under the same fit ids (``model_choice`` module
    docstring).
    """
    if family in win_prob.FAMILIES:
        stage = win_prob.STAGE_NAME
    elif family in win_prob.ALTERNATIVE_FAMILIES:
        stage = model_choice.STAGE_NAME
    else:
        raise ValueError(
            f"the M3-{innings} Primary_Model {family!r} is none of the families {', '.join(win_prob.FAMILIES)} "
            f"({win_prob.STAGE_NAME} checkpoint) and {', '.join(win_prob.ALTERNATIVE_FAMILIES)} "
            f"({model_choice.STAGE_NAME} checkpoint); the re-prediction cannot load it"
        )
    return win_prob.load_fold_models(store, stage, innings, family, n_folds)


def repredict(base: BaseArtifacts, zero_features: Sequence[str] = ZERO_FEATURES) -> Predictions:
    """Re-predict every model output with ``zero_features`` set to 0 (Requirement 21, criterion 1; module docstring).

    The saved outer Fold_Models of each Delivery's fold predict, in order: M2 (Omega, Par_Score,
    Par_Rate, Achievable_Rate, F19), the M3 Primary_Models (pre-ball WP and the Post_Outcome_State
    WPs), the M4 Primary_Model (six variants) and the Extras_Model. Each Primary_Model may be any
    Candidate_Model family: the ``win_prob`` or ``model_choice`` checkpoint holds the M3 fold
    models, and ``model_choice.load_m4_models`` reads the M4 fold models from the checkpoint that
    the ``model_choice`` SOURCES_TABLE names. No model is fitted; the saved calibrations and par
    tables are reused. ``zero_features`` is a subset of ZERO_FEATURES; ``()`` re-predicts the
    stored predictions. The result is memoised in ``base``. Returns a Predictions. Raises
    ValueError for a Primary_Model family without saved fold models, and the errors of the model
    functions.
    """
    names = _names(zero_features, "zero_features", ZERO_FEATURES)
    memo = ("repredict", names)
    if memo in base._cache:
        return base._cache[memo]
    store, settings, state = base.store, base.settings, base.state
    notes = _Notes()
    seconds: dict[str, float] = {}
    tick = time.perf_counter()
    n_rows, fold, innings = base.n_rows, base.fold, base.innings
    balls = store.load_table(reconstruct.STAGE_NAME, reconstruct.BALLS_TABLE)
    fold_table = store.load_table(folds.STAGE_NAME, folds.FOLD_TABLE)

    # The own-fold view of every row (each row from the view of its fold), with the features set to 0.
    view_by_k = {k: store.load_table(remaining_runs.STAGE_NAME, views.view_table(k)) for k in base.ks}
    numeric = [name for name in view_by_k[base.ks[0]].columns if name != "row" and name not in views.FIT_COLUMNS]
    own = pressure.own_fold_view(view_by_k, fold, base.row, columns=numeric)
    m2_fit = np.empty(n_rows, dtype=object)
    for k, view in view_by_k.items():
        at = np.flatnonzero(fold == k)
        m2_fit[at] = view[views.M2_FIT].iloc[at].astype(str).to_numpy(dtype=object)
    own[views.M2_FIT] = m2_fit
    del view_by_k
    for name in names:
        own[name] = 0.0
    tick = _timed(seconds, "views", tick)

    # 1. M2: the outer capped fits (Omega) and the outer first-innings fits (Par_Score, Achievable_Rate, F19).
    outer = [spec for spec in folds.sandbox_plan(base.n_folds) if spec.kind == folds.OUTER]

    def fit_ids(model: str) -> dict[int, str]:
        return {spec.sandbox_k: leakage.fit_id_for(model, spec, family=remaining_runs.FAMILY) for spec in outer}

    capped_ids, first_ids = fit_ids(remaining_runs.CAPPED_MODEL), fit_ids(remaining_runs.FIRST_INNINGS_MODEL)
    for k, fit_id in first_ids.items():
        if not (m2_fit[fold == k] == fit_id).all():
            raise ValueError(f"the fold-{k} rows of view V_{k} do not name the outer first-innings M2 fit {fit_id!r}")
    m2 = remaining_runs.load_models(store, [*capped_ids.values(), *first_ids.values()])
    means_table = store.load_table(remaining_runs.STAGE_NAME, remaining_runs.PAR_MEANS_TABLE)
    bias_table = store.load_table(remaining_runs.STAGE_NAME, remaining_runs.PAR_BIAS_TABLE)
    X = remaining_runs.m2_features(own, state)
    X_post = remaining_runs.post_wicket_features(own, state)
    lb = state["lb_bowled"].to_numpy(dtype=np.int64)
    win_lb = state["win_lb"].to_numpy(dtype=np.int64)
    score = state["score"].to_numpy(dtype=np.int64)
    rrr = state["rrr"].to_numpy(dtype=np.float64)
    sched = base.balls["sched_balls"].to_numpy(dtype=np.int64)[
        _positions(base.balls["row"].to_numpy(dtype=np.int64), base.row, "balls")
    ]
    r_pre, r_post = np.full(n_rows, np.nan), np.full(n_rows, np.nan)
    derived = {name: np.full(n_rows, np.nan) for name in (*remaining_runs.VIEW_M2_COLUMNS,)}
    for k in base.ks:
        served = np.flatnonzero(fold == k)
        capped, first = m2[capped_ids[k]], m2[first_ids[k]]
        r_pre[served] = remaining_runs.predict(capped, X.iloc[served])
        r_post[served] = remaining_runs.predict(capped, X_post.iloc[served])
        one = served[innings[served] == _FIRST_INNINGS]
        two = served[innings[served] != _FIRST_INNINGS]
        if len(one):
            par = remaining_runs.first_innings_par(
                first, _fit_table(means_table, first_ids[k], remaining_runs.PAR_MEAN_FEATURES),
                _fit_table(bias_table, first_ids[k], remaining_runs.PAR_BIAS_COLUMNS), X.iloc[one], lb[one],
                sched[one], win_lb[one],
            )  # fmt: skip
            for name in remaining_runs.PAR_COLUMNS:
                derived[name][one] = par[name].to_numpy(dtype=np.float64)
            derived[remaining_runs.F19_COLUMN][one] = derived[_PAR_SCORE][one] - score[one]
        if len(two):
            rate = remaining_runs.achievable_rate(first, X.iloc[two])
            derived[remaining_runs.ACH_RATE_COLUMN][two] = rate
            derived[remaining_runs.F19_COLUMN][two] = rrr[two] - rate
    for name, values in derived.items():
        own[name] = values
    terminal = value_module.terminal_post_wicket(state["wickets"], state["btc_count"])
    omega = value_module.wicket_cost(r_pre, r_post, state["runs_needed"], innings, terminal).omega
    del X, X_post
    tick = _timed(seconds, "m2", tick)

    # 2. M3 Primary_Models: the pre-ball WP, then the Post_Outcome_States and their WP.
    primary = model_choice.primary_models(store.load_table(model_choice.STAGE_NAME, model_choice.COMPARISON_TABLE))
    labels = store.load_table(win_prob.STAGE_NAME, win_prob.LABELS_TABLE)
    m3 = {number: _m3_models(store, number, primary[model_choice.M3_MODELS[number]], base.n_folds)
          for number in win_prob.M3_INNINGS}  # fmt: skip
    wp_pre = np.full(n_rows, np.nan)
    for number in win_prob.M3_INNINGS:
        mine = np.flatnonzero(innings == number)
        result = win_prob.win_probability(
            number, win_prob.m3_features(number, own, state, labels), fold=fold[mine], models={number: m3[number]}
        )
        wp_pre[mine] = result.wp
    crv = outcome.load_class_run_values(store, outcome.STAGE_NAME)
    post = pressure.post_outcome_states(
        state, balls, {k: own for k in base.ks}, fold, crv, {fit_id: m2[fit_id] for fit_id in first_ids.values()},
        par_means=means_table, par_bias=bias_table,
    )  # fmt: skip
    wp_post = post.matrix(
        {
            number: win_prob.win_probability(
                number, post.features(number), fold=post.fold(number), models={number: m3[number]}
            ).wp
            for number in win_prob.M3_INNINGS
        }
    )
    del post
    tick = _timed(seconds, "m3", tick)

    # 3. The M4 Primary_Model of any family (from the checkpoint that SOURCES_TABLE names): all six
    #    variants, rounded through the float32 m4_<family> table as the outcome and model_choice stages store them.
    m4_models = model_choice.load_m4_models(store, settings, primary[model_choice.M4_MODEL])
    m4_rows = outcome.outcome_rows(balls, state, fold_table)
    if not (np.array_equal(m4_rows.row, base.row) and np.array_equal(m4_rows.fold, fold)):
        raise ValueError("the M4 rows do not follow the state rows and folds; pass the checkpoints of one Run")
    m4_X = outcome.m4_features(own, state, m4_rows)
    probs = outcome.predict_variants(m4_models, {k: m4_X for k in base.ks}, m4_rows.fold, notes)
    m4 = outcome.variant_arrays(outcome.variant_table(probs, m4_rows.row))
    del m4_X, probs
    tick = _timed(seconds, "m4", tick)

    # 4. The Extras_Model, with p_noball capped at 1 - p_wide as extras_preds stores it.
    extras_rows = extras.extras_rows(balls, state, fold_table)
    extras_X = extras.extras_features(own, state, extras_rows)
    extras_models = extras.load_fold_models(store, extras.STAGE_NAME)
    p = extras.predict_extras(extras_models, {k: extras_X for k in base.ks}, extras_rows.fold, notes)
    preds = extras.preds_frame(p, extras_rows, extras_models)
    p_extras = np.column_stack([preds[name].to_numpy(dtype=np.float64) for name in ("p_wide", "p_noball")])
    tick = _timed(seconds, "extras", tick)

    view = own[["row", *_VIEW_COLUMNS]].copy()
    found = Predictions(
        wp_pre=wp_pre, wp_post=wp_post, m4=m4, omega=omega, p_extras=p_extras, view=view, seconds=seconds,
        notes=tuple(notes.messages),
    )  # fmt: skip
    base._cache[memo] = found
    return found


# ================================================================================ Robustness_Cases

#: The Config section of the Requirement 21 grids: ``sensitivity.<name>``.
_SENSITIVITY_SECTION = "sensitivity"

#: The settings of four ablations of Requirement 21, criterion 1. They define the ablations, so
#: they are not Config values: alpha 0 leaves PI to the Stress rank and alpha 1 to the LI rank,
#: the band 0 to 1 makes every Delivery a Live_Ball, and 1.0 is the Stakes_Weight of every match.
_NO_LEVERAGE_ALPHA = 0.0
_NO_STRESS_ALPHA = 1.0
_EVERY_DELIVERY_LIVE = (0.0, 1.0)
_EQUAL_STAKES = 1.0

#: How case settings name the D4 scopes and the D13 cells.
_SCOPE_TEXT: Mapping[str, str] = MappingProxyType(
    {pressure.D4_INNINGS_GROUP: "within each Innings_Group", pressure.D4_POOLED: "pooled over both Innings_Groups"}
)
_CELL_TEXT: Mapping[str, str] = MappingProxyType(
    {D13_DEFAULT: "innings x Phase", D13_COMP_TYPE: "innings x Phase x Competition_Type"}
)

#: The ``load`` checkpoint tables behind the name rule (``PlayerNames.from_store``).
_MAPPING_TABLE = "mapping"
_BALLS_TABLE = "ball_by_ball"


def _shown(value: Any) -> str:
    """A number as case labels and settings show it: a whole number as is, any other to 10 significant digits."""
    if isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)):
        return str(int(value))
    return f"{float(value):.10g}"


def _ordinal(value: Any) -> str:
    """A percentile as text: 20 -> "20th", 21 -> "21st", 22.5 -> "22.5th"."""
    number = float(value)
    if not number.is_integer():
        return f"{_shown(number)}th"
    whole = int(number)
    suffix = "th" if whole % 100 in (11, 12, 13) else {1: "st", 2: "nd", 3: "rd"}.get(whole % 10, "th")
    return f"{whole}{suffix}"


def _grid(settings: Settings, name: str) -> list[Any]:
    """The values of the Config grid ``sensitivity.<name>``. ValueError when it lists none."""
    path = f"{_SENSITIVITY_SECTION}.{name}"
    values = settings.get(path)
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Sequence) or not len(values):
        raise ValueError(f"the Config grid {path} must list one or more values (Requirement 21); got {values!r}")
    return list(values)


def _unique_labels(cases: Sequence[CaseSpec]) -> None:
    counts = Counter(case.label for case in cases)
    repeated = [label for label, count in counts.items() if count > 1]
    if repeated:
        raise ValueError(
            f"every Robustness_Case needs its own label, but {', '.join(map(repr, repeated))} appear more than once; "
            "a Config grid may list each value once"
        )


def all_cases(settings: Settings, base: BaseArtifacts | None = None) -> list[CaseSpec]:
    """Every Robustness_Case of Requirement 21, criteria 1-4, in the order of the module docstring ("Robustness_Cases").

    ``settings``: the Run's Settings, from whose ``sensitivity`` section every grid and alternative
    is read; ``base``: the BaseArtifacts of the Main_Ranking, or None. ``base`` only adds the
    Main_Ranking's Blend_Weights to the Z-score case's setting text: the Z-score case reads them
    from ``base`` when it runs, and ``robustness_table`` flags the cases at the Main_Ranking's
    values (``is_default``). With the delivered Config the list holds 50 cases: the 15 ablations,
    20 grid cases, the Z-score case and 14 alternatives. Raises ValueError for an empty grid, two
    cases with the same label (a grid that lists a value twice) and the CaseSpec errors of a grid
    value out of range.
    """
    cases: list[CaseSpec] = []

    def add(kind: str, label: str, target_id: str, setting: str, layer: str, **overrides: Any) -> None:
        cases.append(CaseSpec(label, target_id, setting, layer, overrides, kind=kind))

    # 1. The 15 ablations (criterion 1), in its order; the pillar ablations are the D5 alternatives (criterion 4).
    add(ABLATION, "leverage removed (alpha = 0)", "F17", "PI from the Stress rank only (alpha 0)", PRESSURE_LAYER,
        alpha=_NO_LEVERAGE_ALPHA)  # fmt: skip
    add(ABLATION, "Stress removed (alpha = 1)", "F19-F26", "PI from the LI rank only (alpha 1)", PRESSURE_LAYER,
        alpha=_NO_STRESS_ALPHA)  # fmt: skip
    for pillar in pressure.PILLARS:
        add(ABLATION, f"{pillar.name} removed", "D5", f"{pillar.name} ({pillar.title}) removed", PRESSURE_LAYER,
            without=(pillar.name,))  # fmt: skip
    add(ABLATION, "opposition adjustment removed", "F28",
        "Expected_Values for an average batter and an average bowler (avg_both), Extras_Model term unchanged",
        VALUE_LAYER, opposition=False)  # fmt: skip
    add(ABLATION, "Venue_Effect and Match_Day_Conditions removed", "F11+F12",
        "both 0 on every Delivery; M2, M3, M4 and Extras_Model re-predicted", VALUE_LAYER,
        zero_features=ZERO_FEATURES)  # fmt: skip
    add(ABLATION, "Recency_Weight = 1", "F33", "Recency_Weight 1 for every Delivery", RATING_LAYER,
        half_life_years=None)  # fmt: skip
    add(ABLATION, f"Stakes_Weight = {_shown(_EQUAL_STAKES)}", "F34",
        f"Stakes_Weight {_shown(_EQUAL_STAKES)} for every match", RATING_LAYER,
        knockout_weight=_EQUAL_STAKES)  # fmt: skip
    add(ABLATION, "live band removed", "F18", "every Delivery a Live_Ball", PRESSURE_LAYER,
        live_band=_EVERY_DELIVERY_LIVE)  # fmt: skip

    # 2. The sensitivity grids (criterion 2), one Design_Choice at a time, in its order.
    for alpha in _grid(settings, "alpha"):
        add(SENSITIVITY, f"D2 alpha = {_shown(alpha)}", "D2", f"alpha {_shown(alpha)}", PRESSURE_LAYER, alpha=alpha)
    for share in _grid(settings, "share"):
        percent = f"{_shown(100.0 * _number(share, 'sensitivity.share value'))}%"
        add(SENSITIVITY, f"D3 share = {percent}", "D3", f"{percent} of Live_Balls", PRESSURE_LAYER, share=share)
    for b in _grid(settings, "live_b"):
        lower = _number(b, "sensitivity.live_b value")
        band = (lower, 1.0 - lower)
        add(SENSITIVITY, f"D1 b = {_shown(lower)}", "D1", f"WP {_shown(band[0])} to {_shown(band[1])}", PRESSURE_LAYER,
            live_band=band)  # fmt: skip
    for years in _grid(settings, "half_life"):
        text = "no decay" if years is None else f"{_shown(years)} years"
        add(SENSITIVITY, f"D9 half-life = {text}", "D9", text, RATING_LAYER, half_life_years=years)
    for weight in _grid(settings, "knockout"):
        add(SENSITIVITY, f"D10 knockout = {_shown(weight)}", "D10", f"knockout Stakes_Weight {_shown(weight)}",
            RATING_LAYER, knockout_weight=weight)  # fmt: skip
    for w_l in _grid(settings, "blend_wL"):
        w_u = 1.0 - _number(w_l, "sensitivity.blend_wL value")
        add(SENSITIVITY, f"blend w_L = {_shown(w_l)}", "F40", f"fixed w_L {_shown(w_l)}, w_U {_shown(w_u)}",
            RATING_LAYER, blend=w_l)  # fmt: skip

    # 3. The Z-score case (criterion 3).
    setting = "Role_Z of shrunk Clutch_Level and of shrunk Uplift, with the Main_Ranking's Blend_Weights"
    if base is not None:
        setting += " (" + ", ".join(f"{role} w_L {base.rankings[role].w_L:.3f}" for role in ROLES) + ")"
    add(ZSCORE, "Z-score blend", "F42", setting, RATING_LAYER, zscore=True)

    # 4. The Config default alternatives (criterion 4), in its order.
    for scope in _grid(settings, "D4"):
        add(ALTERNATIVE, f"D4 scope = {scope}", "D4", _SCOPE_TEXT.get(scope, str(scope)), PRESSURE_LAYER, scope=scope)
    for window in _grid(settings, "D6"):
        add(ALTERNATIVE, f"D6 window = {_shown(window)}", "D6", f"{_shown(window)} Legal_Balls", STATE_LAYER,
            window=window)  # fmt: skip
    for cap in _grid(settings, "D7"):
        add(ALTERNATIVE, f"D7 cap = {_shown(cap)}", "D7", f"{_shown(cap)} Legal_Balls", PRESSURE_LAYER,
            new_batter_cap=cap)  # fmt: skip
    for splits in _grid(settings, "D8"):
        first, second = _pair(splits, "sensitivity.D8 value")
        text = f"{_shown(first)} / {_shown(second)}"
        add(ALTERNATIVE, f"D8 splits = {text}", "D8", f"{text} of Scheduled_Balls", VALUE_LAYER,
            phase_splits=(first, second))  # fmt: skip
    for count in _grid(settings, "D11"):
        add(ALTERNATIVE, f"D11 minimum = {_shown(count)}", "D11", f"{_shown(count)} Pressure_Balls in both Roles",
            RATING_LAYER, min_balls=count)  # fmt: skip
    for pct in _grid(settings, "D12"):
        add(ALTERNATIVE, f"D12 percentile = {_shown(pct)}", "D12", f"{_ordinal(pct)} percentile", RATING_LAYER,
            replacement_pct=pct)  # fmt: skip
    for cells in _grid(settings, "D13"):
        text = _CELL_TEXT.get(cells, str(cells))
        add(ALTERNATIVE, f"D13 cells = {text}", "D13", text, RATING_LAYER, cells=cells)
    for penalty in _grid(settings, "D14"):
        charged = _switch(penalty, "sensitivity.D14 value")
        text = "Retire Out charged to the batter" if charged else "Retire Out not charged"
        add(ALTERNATIVE, f"D14 {text}", "D14", text, VALUE_LAYER, retire_out_penalty=charged)

    _unique_labels(cases)
    return cases


# ================================================================================ running the cases

@dataclass(frozen=True, eq=False)
class CaseRun:
    """One Robustness_Case as ``run_all_cases`` ran it (module docstring, "Running the cases").

    ``case``: the CaseSpec; ``result``: its CaseResult; ``comparisons``: Role -> the
    RankingComparison of the case's ranking with the Main_Ranking (``compare_with_main``: Spearman
    on ``final_score``, tops of LIST_SIZE, the entering and leaving players' names when
    ``run_all_cases`` had a name rule); ``seconds``: the wall-clock time that ``run_all_cases``
    spent on the case; ``reused``: the label of the earlier case with the same layer and overrides
    whose computation this case reuses, or None when it was computed.
    """

    case: CaseSpec
    result: CaseResult
    comparisons: Mapping[str, RankingComparison]
    seconds: float
    reused: str | None = None

    def __repr__(self) -> str:
        how = f"reuses {self.reused!r}" if self.reused else f"{self.result.seconds:.2f} s"
        overlaps = ", ".join(f"{role} {self.comparisons[role].overlap}" for role in ROLES)
        return f"CaseRun({self.case.label!r}, {self.case.kind}: top-10 overlap {overlaps}; {how})"


def _outcome(comparison: RankingComparison, n_case: int, n_main: int) -> tuple[bool, float, Any]:
    """(comparable, Spearman, top-10 overlap) under the not-comparable rule (Requirement 21, criterion 8).

    Comparable: both rankings hold LIST_SIZE or more Eligible_Players, all ranked, and
    MIN_COMMON_PLAYERS or more players are Eligible_Players of both. Otherwise the Spearman
    correlation is NaN and the overlap <NA>.
    """
    if min(n_case, n_main) >= LIST_SIZE and comparison.comparable:
        return True, float(comparison.spearman), int(comparison.overlap)
    return False, math.nan, pd.NA


def _for_role(names: Callable[[str, int], str], role: str) -> Callable[[int], str]:
    return lambda player_id: names(role, int(player_id))


def _log_run(log: Any, run: CaseRun, base: BaseArtifacts, position: int, total: int) -> None:
    case, result = run.case, run.result
    parts, blank = [], []
    for role in ROLES:
        comparable, spearman, overlap = _outcome(run.comparisons[role], result.n_eligible[role], base.n_eligible(role))
        head = f"{role} {result.n_eligible[role]:,} Eligible_Players"
        if comparable:
            rho = "undefined" if math.isnan(spearman) else f"{spearman:.6f}"
            parts.append(f"{head}, Spearman {rho}, top-10 overlap {overlap}")
        else:
            parts.append(f"{head}, not comparable")
            blank.append(role)
    how = f"reuses the computation of {run.reused!r}" if run.reused else f"{result.seconds:.1f} s"
    log.info(
        f"Robustness_Case {position}/{total} {case.label!r} ({case.kind or 'no kind'}, {case.layer} layer; {how}): "
        + "; ".join(parts)
    )
    for role in blank:
        comparison = run.comparisons[role]
        log.warning(
            f"Robustness_Case {case.label!r}, {role} Role: not comparable, with {result.n_eligible[role]:,} "
            f"Eligible_Players ({LIST_SIZE} needed) and {comparison.n_common:,} in common with the Main_Ranking "
            f"({MIN_COMMON_PLAYERS} needed); its Spearman correlation and top-10 overlap are left blank "
            "(Requirement 21, criterion 8)"
        )


def run_all_cases(
    cases: Sequence[CaseSpec],
    base: BaseArtifacts,
    *,
    names: Callable[[str, int], str] | None = None,
    log: Any = None,
) -> list[CaseRun]:
    """Run ``cases`` in order and compare each Role with the Main_Ranking (module docstring, "Running the cases").

    ``cases``: CaseSpecs with distinct labels, normally ``all_cases``; ``base``: the BaseArtifacts
    of the Main_Ranking. Each case runs through ``run_case`` (no refit, no Bootstrap_Resample), so
    a case that leaves a Role not comparable is run, compared and kept like any other. A case with
    the same layer and overrides as an earlier one reuses its CaseResult under its own label.
    ``names``: a function (Role, ``player_id``) -> ``player_name``, such as a PlayerNames, for the
    entering and leaving players; None leaves their names missing. ``log``: None or an object with
    ``info`` and ``warning`` (the Run_Log), which gets one line per case and a warning per
    not-comparable Role. Returns one CaseRun per case, in order. Raises TypeError for arguments of
    the wrong kind, ValueError for a repeated label, and the errors of ``run_case`` and of the name
    rule.
    """
    if isinstance(cases, (str, bytes)) or not isinstance(cases, Sequence):
        raise TypeError(f"cases must be a sequence of CaseSpecs; got {type(cases).__name__}")
    wrong = [type(case).__name__ for case in cases if not isinstance(case, CaseSpec)]
    if wrong:
        raise TypeError(f"cases must hold CaseSpecs only; got {', '.join(sorted(set(wrong)))}")
    if not isinstance(base, BaseArtifacts):
        raise TypeError(f"base must be the BaseArtifacts of BaseArtifacts.build; got {type(base).__name__}")
    if names is not None and not callable(names):
        raise TypeError(f"names must be a function of (role, player_id), such as a PlayerNames, or None; got {names!r}")
    _unique_labels(cases)
    computed: dict[Any, CaseResult] = {}
    first: dict[Any, str] = {}
    runs: list[CaseRun] = []
    for position, case in enumerate(cases, start=1):
        started = time.perf_counter()
        key = (case.layer, _hashable(dict(case.overrides)))
        reused = first.get(key)
        if reused is None:
            result = run_case(case, base)
            computed[key], first[key] = result, case.label
        else:
            result = replace(computed[key], label=case.label, case=case)
        comparisons = {
            role: compare_with_main(base, result, role, names=None if names is None else _for_role(names, role))
            for role in ROLES
        }
        run = CaseRun(case, result, MappingProxyType(comparisons), time.perf_counter() - started, reused)
        runs.append(run)
        if log is not None:
            _log_run(log, run, base, position, len(cases))
    return runs


class PlayerNames:
    """``player_name`` by the lists name rule (Requirement 25, criteria 2, 7 and 8), memoised per Role and player.

    ``names``: ``lists.candidate_names`` of the ``mapping`` table; ``balls``: at least the
    ``ball_by_ball`` columns of ``lists.NAME_COLUMNS``, read only for a player without a
    Candidate_Name. ``names(role, player_id)`` is the ``player_name`` that a Main_List would write
    for the player (``lists.choose_name``); ``PlayerNames.from_store(store)`` reads both tables
    from the ``load`` checkpoint. Calls raise the StageError of ``lists.choose_name`` for a player
    without any name.
    """

    def __init__(self, names: pd.DataFrame, balls: pd.DataFrame) -> None:
        if not isinstance(names, pd.DataFrame) or not isinstance(balls, pd.DataFrame):
            raise TypeError("names and balls must be DataFrames (lists.candidate_names and ball_by_ball)")
        self._names = names
        self._balls = balls
        self._chosen: dict[tuple[str, int], str] = {}

    @classmethod
    def from_store(cls, store: CheckpointStore) -> PlayerNames:
        """The name rule of a Run: ``mapping`` and the name columns of ``ball_by_ball`` from the ``load`` checkpoint."""
        from superr import lists  # on use: the List_Builder imports the Robustness_Suite's bootstrap module

        mapping = store.load_table(_LOAD_STAGE, _MAPPING_TABLE)
        balls = store.load_table(_LOAD_STAGE, _BALLS_TABLE)
        columns = list(dict.fromkeys(column for pair in lists.NAME_COLUMNS.values() for column in pair))
        return cls(lists.candidate_names(mapping), balls.loc[:, columns].copy())

    def __call__(self, role: str, player_id: int) -> str:
        key = (str(role), int(player_id))
        if key not in self._chosen:
            from superr import lists

            self._chosen[key] = lists.choose_name(key[1], self._names, self._balls, key[0]).player_name
        return self._chosen[key]


# ================================================================================ the robustness table

def _joined_ids(ids: Sequence[int]) -> str:
    return LIST_SEPARATOR.join(str(int(pid)) for pid in ids)


def _joined_names(names: pd.Series) -> str | None:
    """The names joined in order; "" for no player and None when a name is missing (no name rule)."""
    if names.isna().any():
        return None
    return LIST_SEPARATOR.join(str(name) for name in names.tolist())


def comparison_row(
    comparison: RankingComparison,
    *,
    case: str,
    kind: str,
    target_id: str,
    setting: str,
    layer: str | None,
    role: str,
    n_eligible_case: int,
    n_eligible_main: int,
    is_default: bool = False,
) -> dict[str, Any]:
    """One robustness-table row (ROBUSTNESS_COLUMNS) for a comparison with the Main_Ranking (module docstring).

    ``comparison``: a RankingComparison of a ranking with the Main_Ranking (``main``) and top lists
    of LIST_SIZE, for one Role; the keywords give the other columns: ``layer`` None for a row that
    has none (a cross-check), ``n_eligible_case`` and ``n_eligible_main`` the Eligible_Player
    counts of the compared ranking and of the Main_Ranking. The not-comparable rule (Requirement
    21, criterion 8) blanks ``spearman`` (NaN) and ``overlap`` (<NA>) and keeps every count and
    the entering and leaving players. Returns a dict for ``robustness_frame``; task 25.4 builds its
    Cross_Check_Ranking rows (kind CROSS_CHECK) the same way. Raises TypeError or ValueError for
    arguments of the wrong kind.
    """
    if not isinstance(comparison, RankingComparison):
        raise TypeError(f"comparison must be a RankingComparison; got {type(comparison).__name__}")
    if comparison.top_n != LIST_SIZE:
        raise ValueError(
            f"the robustness table compares tops of {LIST_SIZE}; the comparison's are of {comparison.top_n}"
        )
    n_case = _whole(n_eligible_case, "n_eligible_case", 0)
    n_main = _whole(n_eligible_main, "n_eligible_main", 0)
    comparable, spearman, overlap = _outcome(comparison, n_case, n_main)
    changes = comparison.changes
    status = changes["status"].astype(str).to_numpy(dtype=object)
    return {
        "case": case,
        "kind": kind,
        "target_id": target_id,
        "setting": setting,
        "layer": layer,
        "role": role,
        "n_eligible_case": n_case,
        "n_eligible_main": n_main,
        "n_common": int(comparison.n_common),
        "spearman": spearman,
        "overlap": overlap,
        "comparable": comparable,
        "is_default": _switch(is_default, "is_default"),
        "entering_ids": _joined_ids(comparison.entering),
        "entering_names": _joined_names(changes["player_name"][status == ENTERING]),
        "leaving_ids": _joined_ids(comparison.leaving),
        "leaving_names": _joined_names(changes["player_name"][status == LEAVING]),
    }


_TEXT_COLUMNS: tuple[str, ...] = (
    "case", "kind", "target_id", "setting", "layer", "role", "entering_ids", "entering_names", "leaving_ids",
    "leaving_names",
)  # fmt: skip
_COUNT_COLUMNS: tuple[str, ...] = ("n_eligible_case", "n_eligible_main", "n_common")
_FLAG_COLUMNS: tuple[str, ...] = ("comparable", "is_default")


def robustness_frame(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """The robustness table of ``rows`` (``comparison_row`` dicts), in row order (module docstring).

    Columns ROBUSTNESS_COLUMNS: the text columns as ``str`` (missing: NaN), the counts int64,
    ``spearman`` float64 (NaN when blank), ``overlap`` Int64 (<NA> when blank), ``comparable`` and
    ``is_default`` bool. Raises ValueError for a row with missing or unknown columns, a ``kind``
    outside TABLE_KINDS, a ``role`` outside ROLES, or a not-comparable row with a Spearman
    correlation or an overlap.
    """
    records = [dict(row) for row in rows]
    for position, record in enumerate(records):
        missing = [name for name in ROBUSTNESS_COLUMNS if name not in record]
        unknown = [name for name in record if name not in ROBUSTNESS_COLUMNS]
        if missing or unknown:
            raise ValueError(f"robustness row {position} lacks the column(s) {missing} and has unknown ones {unknown}")
        if record["kind"] not in TABLE_KINDS:
            raise ValueError(
                f"robustness row {position} has kind {record['kind']!r}; the kinds are {', '.join(TABLE_KINDS)}"
            )
        if record["role"] not in ROLES:
            raise ValueError(f"robustness row {position} has role {record['role']!r}; the Roles are {', '.join(ROLES)}")
        if not record["comparable"] and not (
            pd.isna(record["overlap"]) and math.isnan(float(record["spearman"]))
        ):
            raise ValueError(
                f"robustness row {position} ({record['case']!r}, {record['role']}) is not comparable, so its Spearman "
                "correlation and top-10 overlap must be blank (Requirement 21, criterion 8)"
            )

    def values(name: str) -> list[Any]:
        return [record[name] for record in records]

    data: dict[str, Any] = {name: pd.array(values(name), dtype="str") for name in _TEXT_COLUMNS}
    data.update({name: np.asarray(values(name), dtype=np.int64) for name in _COUNT_COLUMNS})
    data.update({name: np.asarray(values(name), dtype=bool) for name in _FLAG_COLUMNS})
    data["spearman"] = np.asarray(values("spearman"), dtype=np.float64)
    data["overlap"] = pd.array(values("overlap"), dtype="Int64")
    return pd.DataFrame(data, columns=list(ROBUSTNESS_COLUMNS))


def _is_default(case: CaseSpec, base: BaseArtifacts, role: str) -> bool:
    """True when every override of ``case`` equals the Main_Ranking's value; ``min_balls`` compares the Role's count."""
    for key, value in case.overrides.items():
        default = base.defaults[key]
        if key == "min_balls":
            if value[role] != default[role]:
                return False
        elif value != default:
            return False
    return True


def robustness_table(runs: Sequence[CaseRun], base: BaseArtifacts) -> pd.DataFrame:
    """The robustness table of ``runs``: one row per case and Role, in case and ROLES order (Requirement 21, criteria
    5, 6 and 8; module docstring, "The robustness table").

    ``runs``: the CaseRuns of ``run_all_cases``, each with a ``kind``; ``base``: the BaseArtifacts
    of the Main_Ranking behind them, which gives ``n_eligible_main`` and ``is_default`` (every
    override at the Main_Ranking's value, the D11 count compared per Role). Returns a
    ``robustness_frame``. Raises TypeError for arguments of the wrong kind and ValueError for a
    case without a ``kind``.
    """
    if not isinstance(base, BaseArtifacts):
        raise TypeError(f"base must be the BaseArtifacts of BaseArtifacts.build; got {type(base).__name__}")
    rows: list[dict[str, Any]] = []
    for run in runs:
        if not isinstance(run, CaseRun):
            raise TypeError(f"runs must hold the CaseRuns of run_all_cases; got {type(run).__name__}")
        case = run.case
        if case.kind is None:
            raise ValueError(
                f"Robustness_Case {case.label!r} has no kind; the robustness table needs one of {', '.join(CASE_KINDS)}"
            )
        for role in ROLES:
            rows.append(
                comparison_row(
                    run.comparisons[role], case=case.label, kind=case.kind, target_id=case.target_id,
                    setting=case.setting, layer=case.layer, role=role, n_eligible_case=run.result.n_eligible[role],
                    n_eligible_main=base.n_eligible(role), is_default=_is_default(case, base, role),
                )  # fmt: skip
            )
    return robustness_frame(rows)


def write_robustness_table(table: pd.DataFrame, out_dir: Path | str) -> Path:
    """Write ``table`` to ``<out_dir>/robustness_table.csv`` (ROBUSTNESS_FILE) through ``<file>.tmp``; return the path.

    ``out_dir`` is normally ``outputs/robustness/`` (Requirement 21, criterion 6). UTF-8, ``\\n``
    line ends, a header row and no index; ROBUSTNESS_COLUMNS come first, then any other column of
    ``table``; floats have ROBUSTNESS_FLOAT_FORMAT (10 decimals, as the Main_Lists) and a blank
    cell is a missing value. Raises TypeError for a ``table`` that is not a DataFrame and
    ValueError for one that lacks a column of ROBUSTNESS_COLUMNS.
    """
    if not isinstance(table, pd.DataFrame):
        raise TypeError(f"table must be a DataFrame; got {type(table).__name__}")
    missing = [name for name in ROBUSTNESS_COLUMNS if name not in table.columns]
    if missing:
        raise ValueError(f"the robustness table lacks the column(s) {', '.join(missing)}")
    ordered = [*ROBUSTNESS_COLUMNS, *(name for name in table.columns if name not in ROBUSTNESS_COLUMNS)]
    path = Path(out_dir) / ROBUSTNESS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        table.loc[:, ordered].to_csv(
            temporary, index=False, encoding="utf-8", lineterminator="\n", float_format=ROBUSTNESS_FLOAT_FORMAT
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
