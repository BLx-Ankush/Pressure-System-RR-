"""Outcome_Model M4: the 9 ball-outcome classes, their labels and Class_Run_Values, the M4
features, the outer multiclass fits of a Candidate_Model family, Average_Skill, the six
prediction variants, the M4 metrics and the checkpoint artifacts (Requirement 11, criteria 1-3,
5-8 and 14; Requirement 2, criterion 4; design sections "Outcome_Model M4 and Extras_Model
(``models/outcome.py``, ``models/extras.py``)" and "Fold sandboxes and leakage control").

This module is the M4 library, and it holds the ``outcome`` Pipeline_Stage (``run_stage``), which
runs M4 for the LightGBM family, fits the Extras_Model of ``models/extras.py`` beside it and saves
the outputs of both (section "The outcome Pipeline_Stage"). ``model_choice`` runs the XGBoost and
random-forest Alternative_Models through the same functions (task 26.2, section "Families");
``pressure`` and ``value`` read the saved variant probabilities, Class_Run_Values and Extras_Model
probabilities (section "Reading the stage's outputs"). Every function receives the Config values
it needs as arguments: the ``models.m4`` section, ``runtime`` and ``seeds.m4_lgbm`` through the
OutcomeFamily that ``lgbm_family`` builds (``models.xgb.m4`` and ``seeds.m4_xgb`` through
``xgb_family``, ``models.rf.m4`` and ``seeds.m4_rf`` through ``rf_family``), and the
Wickets_Bucket boundaries D17 through ``expected_runs_table``. No function reads ``config.py`` or has a default value for a Config
setting; ``run_stage`` reads them from ``ctx.settings``.

Rows (``outcome_rows``, OutcomeRows)
    M4 works on the rows of the ``state`` table: every Delivery of innings 1 and 2, Wides and
    No_Balls included, in the Requirement 5 order. ``outcome_rows(balls, state, fold_table)``
    aligns the ``balls`` table of the ``reconstruct`` checkpoint to those rows and returns, per
    row: ``row``, ``match_id``, ``fold`` (from the Fold_Table), ``innings``, the pre-ball
    ``phase`` and ``wickets`` (from ``state``), ``legal`` (``cls`` 0), ``trainable``
    (``common.legal_training_mask``: a Legal_Ball of innings 1-2, outside a Super_Over_Innings,
    in a match that the Score_Tolerance rule keeps), ``label`` (the class of a Legal_Ball;
    NO_LABEL on a Wide or No_Ball), ``bat_runs``, ``free_hit`` and the encoded ``pace_or_spin``.
    The ``trainable`` rows of a fit's train folds are its Training_Rows (criterion 3); the
    ``trainable`` rows of each fold k are the held-out Legal_Balls that the metrics evaluate with
    the outer fit of fold k.

Classes and labels (CLASSES, ``label_outcomes``; criterion 14)
    CLASSES gives the 9 classes in the column order of every probability array: ``dot``, ``r1``,
    ``r2``, ``r3``, ``r4`` and ``r6`` (0, 1, 2, 3, 4 and 6 bat runs), ``other_runs`` (any other
    bat runs; as delivered only 5 occurs, on 25 innings 1-2 Legal_Balls, 24 of them ``trainable``),
    ``bowler_wicket`` and ``run_out``.
    A Legal_Ball is labelled ``bowler_wicket`` when it holds a Bowler_Wicket (the ``balls`` flag
    ``bowler_wicket``: a Counted_Wicket with a D14 bowler code, 1, 2, 3, 5 or 6), otherwise
    ``run_out`` when it holds a Run_Out (flag ``run_out``: D14 code 4, whichever batter is out),
    otherwise the class of its bat runs. A Retire Out (code 9), a Retired_Hurt and an
    Obstructing-the-field dismissal (code 10, a Counted_Wicket that is neither) therefore fall
    under the bat-runs rule, and the runs completed before a Run_Out do not change its class.

Class_Run_Values (``class_run_values``; glossary)
    The Class_Run_Value of a class is the mean bat runs of a fit's Training_Rows labelled with
    it: exactly 0, 1, 2, 3, 4 and 6 for the dot and run classes (RUN_CLASS_RUNS), and a mean for
    ``other_runs``, ``bowler_wicket`` and ``run_out`` (MEAN_VALUE_CLASSES). Each outer fit k has
    its own values, from the Legal_Balls of its Training_Folds, and they serve the rows of fold k:
    expected bat runs (sum of class probability x Class_Run_Value), Expected_Value and the
    Post_Outcome_State (Requirement 11, criterion 15). As delivered, ``other_runs`` is 5.0 and
    ``bowler_wicket`` 0.0 in every fold, and ``run_out`` 0.42-0.44 (of the 549 ``trainable``
    Run_Out Legal_Balls of all folds, 329 score no run, 203 one and 17 two).

Features (M4_FEATURES, ``m4_features``; criterion 2)
    ``m4_features(view, state, rows)`` gives the M4_FEATURES of every row, in this order:

    ================ ==================================================== ==========
    Feature          Value                                                Source
    ================ ==================================================== ==========
    innings          1 or 2                                               ``balls``
    phase            Phase code 0-2 (categorical)                         ``state``
    lb_bowled        Legal_Balls bowled before the Delivery               ``state``
    lb_left          Legal_Balls left                                     ``state``
    wickets          wickets fallen                                       ``state``
    f19_component    the batter scoreboard-deficit Pillar_Component (F19) view V_k
    striker_lb_faced the striker's Legal_Balls faced in the innings       ``state``
    theta_striker    the striker's theta_bat                              view V_k
    theta_bowler     the bowler's theta_bowl                              view V_k
    hand_x_style     ``batting_type`` x Modal_Bowling_Type (categorical)  ``state``
    pace_or_spin     0 for ``Pace``, 1 for ``Spin``                       ``balls``
    scoring_env      Scoring_Environment                                  view V_k
    venue_effect     Venue_Effect                                         view V_k
    match_day        Match_Day_Conditions                                 view V_k
    free_hit         the Free_Hit flag, 0 or 1                            ``balls``
    ================ ==================================================== ==========

    The numeric features are float64, with NaN for a missing value, which every family reads as
    missing. ``phase`` and ``hand_x_style`` are pandas categorical columns, which LightGBM and
    XGBoost take as categorical features (``common.boosting_fitter``). ``phase`` always has the
    categories 0, 1 and 2 (PHASE_CATEGORIES); ``hand_x_style`` has the sorted distinct values of
    the ``state`` table passed in (24 as delivered, none missing), so the five feature frames of
    a Run share one category list. LightGBM stores each fit's category lists in the model and
    maps a frame's categories onto them by value when it predicts, so a category that a fit never
    saw is read as missing rather than as another category.

    ``pace_or_spin`` is a number rather than a third categorical feature (PACE_OR_SPIN_CODES:
    ``Pace`` 0, ``Spin`` 1, NaN for a missing or any other value; ``pace_or_spin_codes``). With
    two values, a tree splits it the same way either way, and the random-forest alternative can
    read a number without an encoder. Every innings 1-2 Delivery has ``Pace`` or ``Spin`` as
    delivered.

    The features of fold k's fits come from view V_k (Requirement 7, criteria 7 and 8): a row in
    fold i takes its theta values, Scoring_Environment, Venue_Effect, Match_Day_Conditions and
    F19 component from fits that exclude folds i and k (fold k rows: fits that exclude k). Run
    ``views.check_view`` on V_k before building its frame. ``bowl_speed`` (Stage-2 factor F47,
    task 30.2) is not a feature: ``fit_outcome`` takes the feature list as ``feature_names``, so
    F47 adds a column to the frames and its name to the list.

Fits (OutcomeFamily, ``lgbm_family``, ``fit_outcome``; criteria 1 and 3)
    M4 has no inner fits: its output is never a model feature, so each family has one outer fit
    per fold (design table "Fold sandboxes"). ``fit_outcome(registry, plan, family, rows,
    features, average_skills)`` runs, for each outer FitSpec k of ``folds.sandbox_plan(D15)``,
    one fit through ``common.fit_registered`` as model MODEL with the family's name, so the fit
    ids are those of ``leakage.fit_id_for`` (``m4_lgbm_outer_k1`` ... ``m4_lgbm_outer_k5``). The
    fit trains on the ``trainable`` rows of folds other than k, reading the feature frame built
    from V_k, and is registered with every match of fold k as predicted matches, before it runs.

    LightGBM (``lgbm_family``): the multiclass objective with ``num_class`` 9 and the
    ``models.m4`` learning rate, leaves, minimum rows per leaf, L2 penalty and round cap
    (``max_rounds``); no monotone constraints (none are required for M4); ``runtime.n_threads``
    and ``seeds.m4_lgbm`` with ``deterministic`` and ``force_row_wise`` in deterministic mode, all
    from ``common.lgbm_setup``. Early stopping holds out ``models.m4.es_share`` of the fit's
    training matches as whole matches (``leakage.early_stopping_split`` with ``seeds.m4_lgbm`` and
    the fit index k - 1) and keeps the round with the lowest multiclass log loss on them, after
    ``es_patience`` rounds without improvement or at ``max_rounds``. The held-out matches are
    Training_Rows too, so they count towards the Class_Run_Values.

    Each fit's Class_Run_Values, class counts and Average_Skill are kept with the fitted model in
    an OutcomeFold; the OutcomeFoldModels of a family holds the D15 folds with the family's
    feature list and predict function.

Average_Skill (``average_skill``, AverageSkill; glossary)
    The Average_Skill of fold k is the theta of a typical player in the M1 fit that supplied the
    theta values of the fold-k rows of V_k: M1(-k) (``m1_outer_k<k>``, read from the view's
    ``m1_fit`` column). Its theta_bat is the mean theta_bat of the strikers of that fit's
    training Legal_Balls, weighted by Legal_Balls faced, and its theta_bowl the mean theta_bowl
    of the bowlers of those Legal_Balls, weighted by Legal_Balls bowled. ``average_skill`` reads
    both from the fit's rows of the ``m1_theta`` table of the ``skill`` checkpoint: the weights
    are its ``bat_balls`` and ``bowl_balls``, which count the fit's training Legal_Balls by
    striker and by bowler. Each value is therefore also the mean, over the fit's training
    Legal_Balls, of the striker's theta_bat or the bowler's theta_bowl. As delivered, theta_bat
    is 1.509-1.516 and theta_bowl 0.037-0.045 in the five folds.

    Average_Skill was first the theta of a player whose M1 effects are all zero (theta_bat the
    fit's ``mean_runs``, 1.394-1.399; theta_bowl 0). Ridge gives that theta to rarely seen
    players, who are mostly weak: the 1,806 ``trainable`` rows with theta_bat exactly
    ``mean_runs`` average 0.99 bat runs at 4.7 wickets down. M4 learned this, so the views barely
    moved weak players (bottom theta decile: -0.004 bat runs per Legal_Ball for strikers, -0.005
    for bowlers). On the held-out ``trainable`` Legal_Balls, the actual bat runs exceeded the
    batter view's expected bat runs by 0.044 per Legal_Ball on average, and the bowler view's
    expected bat runs exceeded the actual ones by 0.012, so the mean NBV would have drifted
    positive. With the typical player, these gaps are -0.0002 and 0.0006. The batter view now
    moves the bottom theta_bat decile by +0.043 and the top one by -0.056, and the bowler view
    the bottom theta_bowl decile by -0.014 and the top one by +0.044. M4's expected bat runs
    barely change with theta_striker below about 1.40, so the bottom three theta_bat deciles all
    move by about +0.043.

Prediction variants (VARIANTS, ``variant_features``, ``variant_probabilities``,
``predict_variants``; criteria 1, 5, 7 and 8; Requirement 2, criterion 4)
    Every innings 1-2 Delivery of fold k is predicted by the outer fit of fold k from its V_k
    features, Wides and No_Balls included: each prediction is the probability of each class for
    a Legal_Ball bowled in the Delivery's pre-ball state, so a Wide or No_Ball row reads its own
    pre-ball features like any other row (a Wide bowled during a Free_Hit has ``free_hit`` 1).
    The six variants change only the features they name, on a copy of the actual row:

    ==================== ======================================================================
    Variant              Changes
    ==================== ======================================================================
    actual               none
    avg_both             theta_striker = Average_Skill theta_bat, theta_bowler = Average_Skill
                         theta_bowl (LI, 11.9)
    batter_view          theta_striker = Average_Skill theta_bat (criterion 7)
    batter_view_fh       batter_view and free_hit = 1 (No_Ball Batter_Deliveries, 15.3)
    bowler_view          theta_bowler = Average_Skill theta_bowl (criterion 8)
    bowler_view_nofh     bowler_view and free_hit = 0 (No_Ball replacement ball, 15.6, 15.14)
    ==================== ======================================================================

    So a batter view keeps the bowler's actual theta_bowl and a bowler view the striker's actual
    theta_bat (Requirement 2, criterion 4). ``variant_probabilities`` returns the probabilities as
    the family's model gives them; ``predict_variants`` then rescales every row to sum to 1 in
    float64 (``common.renormalise``), which writes a Run_Log warning when a row was more than
    1e-6 away from 1 (criterion 5). Both return a dict from variant name to a float64 array of
    shape (rows, 9): one row per row of ``row_fold`` (the ``state`` order in a Run), one column
    per class in CLASSES order. LightGBM's softmax rows sum to 1 within about 1e-15.

Metrics (``phase_baseline``, ``log_loss_table``, ``expected_bat_runs``,
``expected_runs_table``; criterion 6)
    All on the held-out Legal_Balls: the ``trainable`` rows of every fold, each predicted by the
    outer fit of its fold (``actual`` variant).

    * ``log_loss_table``: the Pooled_Held_Out_Log_Loss (multiclass, probabilities clipped to
      [1e-15, 1 - 1e-15], one mean over all evaluated rows) of the candidate and of the Phase-only
      baseline. ``phase_baseline`` predicts every row of fold k with the class frequencies of the
      Training_Rows of fold k's fit (``trainable`` rows of the other folds) in the row's Phase.
    * ``expected_runs_table``: for each Phase x Wickets_Bucket (D17) cell, the Legal_Ball count,
      the mean expected bat runs (sum of class probability x the Class_Run_Value of the row's
      fold; ``expected_bat_runs``) and the mean actual bat runs.

Checkpoint artifacts (``save_fold_models``, ``load_fold_models``, ``variant_table``,
``variant_arrays``)
    ``save_fold_models(store, stage, models)`` saves each fold's model under its fit id
    (``store.save_model``: LightGBM model text with its best round, which keeps the category
    lists; XGBoost JSON for an XGBoost Booster, with XGB_CATEGORIES_ATTRIBUTE; joblib for a
    random-forest pipeline) and the table ``fold_models_table(family)`` (``m4_lgbm_fold_models``),
    one row per fold with the FOLD_MODEL_COLUMNS: fit id, family, feature list, the M1 fit of
    Average_Skill (``m1_fit``) and its typical-player theta_bat and theta_bowl (``avg_theta_bat``,
    ``avg_theta_bowl``), ``best_iteration`` (the boosting rounds kept, or the trees of a random
    forest), row counts, and the Class_Run_Value (``crv_<class>``) and Training_Row count
    (``n_<class>``) of every class. ``load_fold_models(store, stage, family)``
    rebuilds the OutcomeFoldModels, for example for the robustness re-predictions with the
    saved outer Fold_Models (task 25.1).

    ``variant_table(probs, row_ids)`` lays the six variant arrays out as the design's
    ``m4_<candidate>`` table (``predictions_table(family)``: ``m4_lgbm``): ``row`` (int32), then
    ``<variant>_<class>`` for the 6 VARIANT_NAMES x 9 CLASSES in that order (``variant_columns``),
    as float32. A float32 value differs from its float64 source by at most 2^-24 of itself, so a
    row read back by ``variant_arrays`` (float64, shape (rows, 9) per variant) still sums to 1
    within 6e-8.

    ``class_run_values_frame(models)`` gives the Class_Run_Values of every fold as the table
    CLASS_RUN_VALUES_TABLE (``m4_class_run_values``; CLASS_RUN_VALUE_COLUMNS: ``fold``, then the
    ``crv_<class>`` and ``n_<class>`` of every class). The values depend only on a fit's
    Training_Rows, which every family shares, so the table names no family.
    ``load_class_run_values(store, stage)`` reads it back as a mapping from fold to its N_CLASSES
    values in CLASSES order, the form that ``pressure.post_outcome_states`` takes as ``crv``.

Families (FAMILIES, ``candidate_family``, ``xgb_family``, ``rf_family``; task 26.2)
    An OutcomeFamily names a Candidate_Model family and supplies its early stopping, its fit
    function, its predict function and, optionally, how to read a fitted model's size
    (``iterations``). ``candidate_family(name, settings)`` builds the family of any name of
    FAMILIES: ``lgbm`` (``lgbm_family``), and the two M4 Alternative_Models of Requirement 23,
    which the ``model_choice`` stage fits with ``fit_outcome`` on the same Training_Rows, labels,
    M4_FEATURES, views and folds as LightGBM, one setting each from the Config, with no monotone
    constraints (none are required for M4):

    * ``xgb`` (``xgb_family``): an XGBoost ``multi:softprob`` Booster with ``num_class`` 9 from
      ``common.xgb_setup(settings, "m4", "multiclass", num_class=N_CLASSES)``: the
      ``models.xgb.m4`` ``eta``, ``max_depth``, ``min_child_weight``, ``reg_lambda``,
      ``tree_method`` ("hist") and round cap, ``runtime.n_threads`` and ``seeds.m4_xgb``. It is
      fitted by ``common.boosting_fitter``: early stopping on the multiclass log loss of
      ``models.xgb.m4.es_share`` of the training matches, held out whole by
      ``leakage.early_stopping_split`` with ``seeds.m4_xgb`` and the fit index (so drawn from the
      training folds only), patience ``es_patience``, and the Booster cut to its best round
      (``best_iteration`` keeps the rounds it holds). ``phase`` and ``hand_x_style`` are native
      XGBoost categorical features (a DMatrix with ``enable_categorical=True``; ``hist`` splits
      ``phase``, with 3 categories, one category against the rest and ``hand_x_style`` by
      partition). XGBoost reads a pandas category by its code, and the cut to the best round drops
      the category lists that XGBoost 3 keeps in a full Booster, so the fit stores them itself in
      the Booster attribute XGB_CATEGORIES_ATTRIBUTE, which XGBoost JSON saves; ``xgb_predict``
      maps each frame's categories onto them by value before predicting, so a category the fit
      never saw is read as missing, as in LightGBM. The predictions are XGBoost's float32 softmax
      values, whose rows sum to 1 within about 1e-7.
    * ``rf`` (``rf_family``): a scikit-learn pipeline (``_rf_pipeline``) of a ColumnTransformer
      and a ``RandomForestClassifier`` from ``common.rf_setup(settings, "m4")``: the
      ``models.rf.m4`` trees, ``max_depth``, ``min_samples_leaf`` and ``max_features``,
      ``random_state=seeds.m4_rf`` and ``runtime.n_threads`` for fitting (1 thread for predicting
      in deterministic mode). The ColumnTransformer one-hot encodes the categorical features
      (``phase``, 3 columns; ``hand_x_style``, one column per category of the Training_Rows, 24 as
      delivered; an unseen category gives an all-zero row) and imputes each numeric feature with
      its median, both fitted on the fit's Training_Rows only. The M4 features have no missing
      value as delivered; the imputer guards the forest, which scikit-learn would otherwise fit
      with its own missing-value rule. No early stopping (``early_stopping=None``), and
      ``best_iteration`` keeps the trees. ``rf_predict`` places the ``predict_proba`` columns by
      ``classes_`` onto the 9 CLASSES, so a class absent from a fold's Training_Rows gets
      probability 0; the rows are means of leaf class shares and sum to 1 within about 1e-15.

    ``fit_outcome``, ``variant_probabilities``, ``predict_variants``, the metrics and the
    artifacts work unchanged for every family. ``load_fold_models(store, stage,
    candidate_family(name, settings))`` rebuilds any family's fold models: XGBoost Boosters come
    back from XGBoost JSON with their category lists, random-forest pipelines from joblib.

The ``outcome`` Pipeline_Stage (``run_stage``)
    Reads ``balls`` (``reconstruct``), ``state`` (``state``), the Fold_Table (``folds``), the
    ``m1_theta`` table (``skill``) and the views V_1 ... V_D15 (``remaining_runs``). Then:

    1. builds the OutcomeRows (``outcome_rows``) and the Extras_Model rows
       (``extras.extras_rows``) of every innings 1-2 Delivery, in ``state`` order;
    2. for each fold k, loads V_k, checks that it holds the ``state`` rows in order and runs
       ``views.check_view`` on it before any fit reads it, then builds from it the M4 feature
       frame (``m4_features``), the Extras_Model feature frame (``extras.extras_features``) and
       the fold's Average_Skill (``average_skill`` with ``m1_theta``);
    3. runs the D15 outer M4 fits of the LightGBM family (``fit_outcome``) and predicts the six
       variants of every row (``predict_variants``);
    4. runs the D15 outer Extras_Model fits (``extras.fit_extras``) and predicts every row
       (``extras.predict_extras``);
    5. saves the M4 fold models (``save_fold_models``: each Booster under its fit id, and
       ``m4_lgbm_fold_models``), the variant table ``m4_lgbm`` (``variant_table``),
       CLASS_RUN_VALUES_TABLE, the Extras_Model fold models (``extras.save_fold_models``: each
       Pipeline under its fit id, and ``extras_fold_models``) and ``extras_preds``
       (``extras.preds_frame``);
    6. writes ``outputs/models/m4/`` (M4_FOLDER): LOG_LOSS_FILE (``log_loss_table``, the
       candidate against the Phase-only baseline), EXPECTED_RUNS_FILE (``expected_runs_table``
       with D17) and FOLD_MODELS_FILE (``fold_models_frame`` without the feature list: per fold,
       the M1 fit and the theta_bat and theta_bowl of Average_Skill, the boosting rounds kept, the
       row counts, and the Class_Run_Value and Training_Rows of every class); and
       ``outputs/models/extras/`` (``extras.EXTRAS_FOLDER``): ``extras.LOG_LOSS_FILE`` (the model
       against the cell baseline), ``extras.RATES_FILE`` (predicted against actual Wide and
       No_Ball rates per innings x Phase cell) and ``extras.FOLD_MODELS_FILE`` (the
       ``extras_fold_models`` table without the feature list);
    7. saves the fit registry (the D15 M4 and D15 Extras_Model fits) and records two data-derived
       Method_Constants of the Outcome_Model: AVERAGE_SKILL_CONSTANT (fold -> the M1 fit and the
       theta_bat and theta_bowl of its Average_Skill) and BEST_ITERATION_CONSTANT (fit id -> the
       boosting rounds kept by each outer LightGBM fit).

Reading the stage's outputs (``model_choice``, ``pressure``, ``value``)
    * ``variant_arrays(store.load_table(STAGE_NAME, predictions_table(LGBM_FAMILY)))`` gives one
      float64 array of shape (rows, N_CLASSES) per variant: rows in ``state`` order (the table's
      ``row`` column), columns in CLASSES order, each row summing to 1 within 6e-8 (float32
      storage). A row of fold k (``folds.fold_of`` of its match) was predicted by
      ``m4_lgbm_outer_k<k>``, the fit of ``m4_lgbm_fold_models`` for fold k. LI uses ``avg_both``
      (Requirement 11, criterion 9). NBV uses ``batter_view`` for Legal_Ball Batter_Deliveries
      (15.2), ``batter_view_fh`` for No_Ball Batter_Deliveries (15.3), ``bowler_view`` for the
      bowler Expected_Value (15.5) and the Wide replacement ball (15.13), and ``bowler_view_nofh``
      for the No_Ball replacement ball (15.14) and the expected No_Ball Extras_Cost (15.6). Each
      comes from the M4 Primary_Model that ``model_choice`` selects; this table is the LightGBM
      candidate's.
    * ``load_class_run_values(store, STAGE_NAME)``: fold -> its N_CLASSES Class_Run_Values. A row
      takes those of its fold, for expected bat runs (``expected_bat_runs``) and the
      Post_Outcome_State (``pressure.post_outcome_states``, argument ``crv``).
    * ``store.load_table(extras.STAGE_NAME, extras.PREDS_TABLE)``: ``row``, ``p_wide``,
      ``p_noball``, ``fold`` and ``fit_id`` (``extras_outer_k<fold>``), rows in ``state`` order;
      the Legal_Ball probability is 1 - ``p_wide`` - ``p_noball``.
    * ``load_fold_models(store, STAGE_NAME, lgbm_family(settings))`` and
      ``extras.load_fold_models(store, STAGE_NAME)`` rebuild the fold models for prediction from
      other feature values (Requirement 21, criterion 1); ``outcome_rows`` rebuilds the labels and
      ``trainable`` rows behind the M4 metrics.

Errors: ValueError or TypeError, naming the problem, for malformed arguments, such as a missing
column, arrays of different lengths, tables whose rows differ from the ``state`` rows, a missing
value where none may be, a class without a Training_Row for its Class_Run_Value, an unknown
variant, or predictions of the wrong shape or not finite. ``fit_registered`` raises FitLeakageError
(exit code 4) when a fit breaks the leakage guard, ``views.check_view`` ViewLeakageError (exit
code 4) for a view row whose values come from a fit that saw its fold, and ``folds.fold_of``
DataError (exit code 3) for a match that is not in the Fold_Table.
"""

from __future__ import annotations

import functools
import json
import os
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import pandas as pd

from superr import folds, reconstruct, views
from superr.folds import FitSpec
from superr.models import common, extras
from superr.reconstruct import DeliveryClass
from superr.skill import STAGE_NAME as SKILL_STAGE
from superr.skill import THETA_TABLE as M1_THETA_TABLE
from superr.state import STAGE_NAME as STATE_STAGE
from superr.state import STATE_INNINGS, STATE_TABLE, Phase

if TYPE_CHECKING:
    from superr.leakage import FitRegistry
    from superr.pipeline import StageContext
    from superr.settings import Settings
    from superr.store import CheckpointStore

__all__ = [
    "ACTUAL",
    "ALTERNATIVE_FAMILIES",
    "AVERAGE",
    "AVERAGE_SKILL_CONSTANT",
    "AVG_BOTH",
    "BATTER_VIEW",
    "BATTER_VIEW_FH",
    "BOWLER_VIEW",
    "BOWLER_VIEW_NOFH",
    "BEST_ITERATION_CONSTANT",
    "BOWLER_WICKET",
    "CATEGORICAL_FEATURES",
    "CLASSES",
    "CLASS_RUN_VALUES_TABLE",
    "CLASS_RUN_VALUE_COLUMNS",
    "DOT",
    "EXPECTED_RUNS_COLUMNS",
    "EXPECTED_RUNS_FILE",
    "FAMILIES",
    "FOLD_MODELS_FILE",
    "FOLD_MODEL_COLUMNS",
    "LGBM_FAMILY",
    "LOG_LOSS_COLUMNS",
    "LOG_LOSS_FILE",
    "M4_FEATURES",
    "M4_FOLDER",
    "MEAN_VALUE_CLASSES",
    "MODEL",
    "NO_LABEL",
    "N_CLASSES",
    "OTHER_RUNS",
    "PACE_OR_SPIN_CODES",
    "PHASE_BASELINE",
    "PHASE_CATEGORIES",
    "R1",
    "R2",
    "R3",
    "R4",
    "R6",
    "RF_FAMILY",
    "ROW_FEATURES",
    "RUN_CLASS_RUNS",
    "RUN_OUT",
    "STAGE_NAME",
    "STATE_FEATURES",
    "VARIANTS",
    "VARIANT_NAMES",
    "VIEW_FEATURES",
    "XGB_CATEGORIES_ATTRIBUTE",
    "XGB_FAMILY",
    "AverageSkill",
    "OutcomeFamily",
    "OutcomeFold",
    "OutcomeFoldModels",
    "OutcomeRows",
    "average_skill",
    "candidate_family",
    "class_run_values",
    "class_run_values_frame",
    "expected_bat_runs",
    "expected_runs_table",
    "fit_outcome",
    "fold_models_frame",
    "fold_models_table",
    "label_outcomes",
    "lgbm_family",
    "lgbm_predict",
    "load_class_run_values",
    "load_fold_models",
    "log_loss_table",
    "m4_features",
    "outcome_rows",
    "pace_or_spin_codes",
    "phase_baseline",
    "predict_variants",
    "predictions_table",
    "rf_family",
    "rf_predict",
    "run_stage",
    "save_fold_models",
    "variant_arrays",
    "variant_columns",
    "variant_features",
    "variant_probabilities",
    "variant_table",
    "wickets_bucket_codes",
    "xgb_family",
    "xgb_predict",
]


# ------------------------------------------------------------------------------------- constants

#: Name of the Pipeline_Stage whose checkpoint holds the M4 artifacts (registered by task 12.2).
STAGE_NAME = "outcome"

#: Model name of every M4 fit in the FitRegistry; with the family it gives the fit ids
#: (``m4_lgbm_outer_k1``) and the checkpoint names (``m4_lgbm``, ``m4_lgbm_fold_models``).
MODEL = "m4"

#: The family name of the LightGBM Candidate_Model (FitRegistry family, artifact names).
LGBM_FAMILY = "lgbm"

#: The family names of the M4 Alternative_Models (Requirement 23, criterion 1), in the tie order
#: of criterion 8 after LGBM_FAMILY: XGBoost (Config ``models.xgb.m4``, seed ``seeds.m4_xgb``)
#: and the random forest (``models.rf.m4``, ``seeds.m4_rf``). FAMILIES lists every M4 family.
XGB_FAMILY = "xgb"
RF_FAMILY = "rf"
ALTERNATIVE_FAMILIES: tuple[str, ...] = (XGB_FAMILY, RF_FAMILY)
FAMILIES: tuple[str, ...] = (LGBM_FAMILY, *ALTERNATIVE_FAMILIES)

#: The attribute of an XGBoost M4 Booster that holds the category lists of its fit's categorical
#: features, as JSON (feature -> categories in code order); xgb_predict maps a frame's categories
#: onto them by value (module docstring, "Families").
XGB_CATEGORIES_ATTRIBUTE = "superr_categories"

#: The Config model key of M4 in ``common.LGBM_MODELS``, ``XGB_MODELS`` and ``RF_MODELS``: sections
#: ``models.m4``, ``models.xgb.m4`` and ``models.rf.m4``; seeds ``seeds.m4_lgbm``, ``m4_xgb``, ``m4_rf``.
_CONFIG_MODEL = "m4"
_OBJECTIVE = "multiclass"

#: The 9 Outcome_Model classes, in the column order of every probability array (criterion 1).
CLASSES: tuple[str, ...] = ("dot", "r1", "r2", "r3", "r4", "r6", "other_runs", "bowler_wicket", "run_out")
N_CLASSES = len(CLASSES)
DOT, R1, R2, R3, R4, R6, OTHER_RUNS, BOWLER_WICKET, RUN_OUT = range(N_CLASSES)

#: Class -> its bat runs, for the dot and run classes; their Class_Run_Value is exactly this.
RUN_CLASS_RUNS: Mapping[int, int] = MappingProxyType({DOT: 0, R1: 1, R2: 2, R3: 3, R4: 4, R6: 6})

#: The classes whose Class_Run_Value is the mean bat runs of their Training_Rows.
MEAN_VALUE_CLASSES: tuple[int, ...] = (OTHER_RUNS, BOWLER_WICKET, RUN_OUT)

#: The ``label`` of a Wide or No_Ball in OutcomeRows: no class (only Legal_Balls are labelled).
NO_LABEL = -1

#: The M4 features, in the column order of every M4 fit (criterion 2). Index a DataFrame with
#: ``list(M4_FEATURES)``: pandas reads a tuple as one column key.
M4_FEATURES: tuple[str, ...] = (
    "innings", "phase", "lb_bowled", "lb_left", "wickets", "f19_component", "striker_lb_faced",
    "theta_striker", "theta_bowler", "hand_x_style", "pace_or_spin", "scoring_env", "venue_effect",
    "match_day", "free_hit",
)  # fmt: skip

#: The M4 features that are pandas categorical columns (categorical features of the boosted families).
CATEGORICAL_FEATURES: tuple[str, ...] = ("phase", "hand_x_style")

#: Where m4_features takes each feature: the ``state`` table, the sandbox view, or OutcomeRows (``balls``).
STATE_FEATURES: tuple[str, ...] = ("phase", "lb_bowled", "lb_left", "wickets", "striker_lb_faced", "hand_x_style")
VIEW_FEATURES: tuple[str, ...] = (
    "f19_component", "theta_striker", "theta_bowler", "scoring_env", "venue_effect", "match_day",
)  # fmt: skip
ROW_FEATURES: tuple[str, ...] = ("innings", "pace_or_spin", "free_hit")

#: The categories of the ``phase`` feature: the Phase codes (powerplay 0, middle 1, death 2).
PHASE_CATEGORIES: tuple[int, ...] = tuple(int(member) for member in Phase)

#: The ``pace_or_spin`` text -> its feature value; a missing or any other text is NaN (missing).
PACE_OR_SPIN_CODES: Mapping[str, float] = MappingProxyType({"Pace": 0.0, "Spin": 1.0})

#: A variant setting that replaces a theta feature with the fold's Average_Skill: the typical
#: player's theta_bat for theta_striker and the typical player's theta_bowl for theta_bowler.
AVERAGE = "avg"

#: The six prediction variants (design; Requirement 11, criteria 5, 7 and 8): variant -> the
#: features it changes, each to AVERAGE (theta_striker: Average_Skill theta_bat; theta_bowler:
#: Average_Skill theta_bowl) or to a Free_Hit flag (True 1, False 0). Every other feature keeps
#: its value.
ACTUAL, AVG_BOTH, BATTER_VIEW, BATTER_VIEW_FH, BOWLER_VIEW, BOWLER_VIEW_NOFH = (
    "actual", "avg_both", "batter_view", "batter_view_fh", "bowler_view", "bowler_view_nofh",
)  # fmt: skip
VARIANTS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        ACTUAL: MappingProxyType({}),
        AVG_BOTH: MappingProxyType({"theta_striker": AVERAGE, "theta_bowler": AVERAGE}),
        BATTER_VIEW: MappingProxyType({"theta_striker": AVERAGE}),
        BATTER_VIEW_FH: MappingProxyType({"theta_striker": AVERAGE, "free_hit": True}),
        BOWLER_VIEW: MappingProxyType({"theta_bowler": AVERAGE}),
        BOWLER_VIEW_NOFH: MappingProxyType({"theta_bowler": AVERAGE, "free_hit": False}),
    }
)
VARIANT_NAMES: tuple[str, ...] = tuple(VARIANTS)
_AVERAGE_FIELD: Mapping[str, str] = MappingProxyType({"theta_striker": "theta_bat", "theta_bowler": "theta_bowl"})

#: The columns of the ``m1_theta`` table of the ``skill`` checkpoint that average_skill reads, and
#: the weight of each AverageSkill theta: the fit's training Legal_Balls faced or bowled.
_THETA_TABLE_COLUMNS: tuple[str, ...] = ("fit_id", "player_id", "theta_bat", "theta_bowl", "bat_balls", "bowl_balls")
_AVERAGE_WEIGHT: Mapping[str, str] = MappingProxyType({"theta_bat": "bat_balls", "theta_bowl": "bowl_balls"})

#: The ``model`` of the baseline row of log_loss_table, and the columns of the two metric tables.
PHASE_BASELINE = "phase_baseline"
LOG_LOSS_COLUMNS: tuple[str, ...] = ("model", "n_legal_balls", "log_loss")
EXPECTED_RUNS_COLUMNS: tuple[str, ...] = (
    "phase", "wickets_bucket", "n_legal_balls", "mean_expected_bat_runs", "mean_actual_bat_runs",
)  # fmt: skip

#: Columns of the fold-model table (fold_models_frame, save_fold_models).
FOLD_MODEL_COLUMNS: tuple[str, ...] = (
    "fold", "fit_id", "family", "features", "m1_fit", "avg_theta_bat", "avg_theta_bowl", "best_iteration",
    "n_train_rows", "n_fit_rows", "n_early_stopping_rows",
    *(f"crv_{name}" for name in CLASSES),
    *(f"n_{name}" for name in CLASSES),
)  # fmt: skip

#: Checkpoint table of the Class_Run_Values of every fold, the same for every family
#: (class_run_values_frame, load_class_run_values), and its columns.
CLASS_RUN_VALUES_TABLE = "m4_class_run_values"
CLASS_RUN_VALUE_COLUMNS: tuple[str, ...] = (
    "fold", *(f"crv_{name}" for name in CLASSES), *(f"n_{name}" for name in CLASSES),
)  # fmt: skip

#: Folder under the outputs folder for the M4 files of the ``outcome`` stage (criterion 6), and their names.
M4_FOLDER = Path("models") / "m4"
LOG_LOSS_FILE = "log_loss.csv"
EXPECTED_RUNS_FILE = "expected_runs_by_phase_wickets.csv"
FOLD_MODELS_FILE = "fold_models.csv"

#: Data-derived Method_Constants that the ``outcome`` stage records: fold -> the M1 fit and the
#: theta_bat and theta_bowl of the fold's Average_Skill; and fit id -> the boosting rounds that
#: early stopping kept, for each outer LightGBM fit (the fits behind the Out_Of_Fold_Predictions).
AVERAGE_SKILL_CONSTANT = "outcome.average_skill_by_fold"
BEST_ITERATION_CONSTANT = "outcome.outer_lgbm_best_iteration_by_fit"

_COMPONENT = "Outcome_Model"  # the glossary component that produces the constants
_N_FOLDS = "design_choices.D15_n_folds"
_WICKET_BUCKETS = "design_choices.D17_wicket_buckets"
_FLOAT_FORMAT = "%.6f"
_SUM_TOLERANCE = common.PROB_SUM_TOLERANCE  # a saved M4 row may differ from a sum of 1 by at most this (criterion 5)

_ROW_COLUMN = "row"
_FEATURE_SEPARATOR = ","
_BALL_COLUMNS: tuple[str, ...] = (
    "row", "match_id", "innings", "cls", "runs", "bowler_wicket", "run_out", "free_hit", "pace_or_spin",
    "is_super_over", "excluded",
)  # fmt: skip
_STATE_COLUMNS: tuple[str, ...] = ("row", "phase", "wickets")
_FAMILY_NAME = re.compile(r"[a-z][a-z0-9]{0,15}")
_INT32_MAX = np.iinfo(np.int32).max
_SHOWN = 5  # examples named in one error message


# ---------------------------------------------------------------------------------- small helpers

def _rows(count: int) -> str:
    return f"{count:,} row" if count == 1 else f"{count:,} rows"


def _require(frame: Any, columns: Iterable[str], what: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{what} must be a DataFrame; got {type(frame).__name__}")
    absent = [column for column in columns if column not in frame.columns]
    if absent:
        raise ValueError(f"{what} lacks the column(s) {', '.join(absent)}")


def _series(values: Any, what: str) -> pd.Series:
    """``values`` (1-D) as a Series on a RangeIndex."""
    if values is None or isinstance(values, (str, bytes, Mapping, pd.DataFrame)) or np.ndim(values) != 1:
        raise TypeError(f"{what} must be a 1-D array or Series; got {type(values).__name__}")
    if isinstance(values, (pd.Series, pd.Index, pd.api.extensions.ExtensionArray)):
        return pd.Series(pd.Series(values).array)
    return pd.Series(np.asarray(values))


def _check_length(array: np.ndarray, what: str, n_rows: int | None) -> np.ndarray:
    if n_rows is not None and len(array) != n_rows:
        raise ValueError(f"{what} has {len(array):,} values, but {n_rows:,} are needed, one per row")
    return array


def _ints(values: Any, what: str, n_rows: int | None = None, *, minimum: int | None = None) -> np.ndarray:
    """``values`` (1-D whole numbers, none missing) as a new int64 array."""
    series = _series(values, what)
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype) or not pd.api.types.is_numeric_dtype(dtype):
        raise TypeError(f"{what} must hold whole numbers; got dtype {dtype}")
    missing = int(series.isna().sum())
    if missing:
        raise ValueError(f"{what} has {missing:,} missing value(s)")
    if pd.api.types.is_integer_dtype(dtype):
        array = series.to_numpy(dtype=np.int64)
    else:
        raw = series.to_numpy(dtype=np.float64)
        whole = np.isfinite(raw) & (np.floor(raw) == raw) & (np.abs(raw) <= 2.0**53)
        if not whole.all():
            raise ValueError(f"{what} must hold whole numbers; {int((~whole).sum()):,} value(s) are not")
        array = raw.astype(np.int64)
    _check_length(array, what, n_rows)
    if minimum is not None and len(array) and int(array.min()) < minimum:
        raise ValueError(f"{what} must be {minimum} or more; {int((array < minimum).sum()):,} value(s) are below")
    return array


def _flags(values: Any, what: str, n_rows: int | None = None) -> np.ndarray:
    """``values`` (1-D booleans, none missing) as a new bool array."""
    series = _series(values, what)
    if not pd.api.types.is_bool_dtype(series.dtype):
        raise TypeError(f"{what} must be boolean; got dtype {series.dtype}")
    missing = int(series.isna().sum())
    if missing:
        raise ValueError(f"{what} has {missing:,} missing value(s)")
    return _check_length(series.to_numpy(dtype=bool), what, n_rows)


def _floats(values: Any, what: str, n_rows: int | None = None) -> np.ndarray:
    """``values`` (1-D numbers) as a new float64 array, NaN for a missing value."""
    series = _series(values, what)
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype) or not pd.api.types.is_numeric_dtype(dtype):
        raise TypeError(f"{what} must hold numbers; got dtype {dtype}")
    return _check_length(series.to_numpy(dtype=np.float64, na_value=np.nan), what, n_rows)


def _prob_matrix(values: Any, n_rows: int, what: str, *, finite: bool = True) -> np.ndarray:
    """``values`` as a float64 (n_rows, N_CLASSES) array of finite numbers; ValueError otherwise.

    With ``finite`` False, missing and non-finite values are let through (only the shape is checked).
    """
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{what} must be an array of numbers: {exc}") from None
    if array.shape != (n_rows, N_CLASSES):
        raise ValueError(
            f"{what} has shape {array.shape}; ({n_rows:,}, {N_CLASSES}) is needed, one row per Delivery and one "
            f"column per class of CLASSES"
        )
    if finite:
        bad = ~np.isfinite(array)
        if bad.any():
            raise ValueError(f"{what} holds {int(bad.sum()):,} missing or non-finite value(s)")
    return array


def _frozen(values: Any, dtype: Any) -> np.ndarray:
    array = np.array(values, dtype=dtype)
    array.flags.writeable = False
    return array


def _fold_number(value: Any, what: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{what} must be a fold number, a whole number; got {value!r}")
    if value < 1:
        raise ValueError(f"{what} must be 1 or more, folds are numbered from 1; got {value}")
    return int(value)


def _check_variants(variants: Any) -> tuple[str, ...]:
    if isinstance(variants, (str, bytes)) or not isinstance(variants, Iterable):
        raise TypeError(f"variants must be a sequence of names of VARIANTS; got {variants!r}")
    names = tuple(variants)
    unknown = [name for name in names if name not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown M4 variant(s) {unknown}; the variants are {', '.join(VARIANT_NAMES)}")
    if not names or len(set(names)) != len(names):
        raise ValueError(f"variants must name each variant at most once and at least one; got {list(names)}")
    return names


def _check_feature_names(names: Any) -> tuple[str, ...]:
    if isinstance(names, (str, bytes)) or not isinstance(names, Iterable):
        raise TypeError(f"feature_names must be a sequence of column names; got {names!r}")
    out = tuple(names)
    if not out or not all(isinstance(name, str) and name for name in out) or len(set(out)) != len(out):
        raise ValueError(f"feature_names must hold distinct, non-empty column names; got {list(out)}")
    if _FEATURE_SEPARATOR in "".join(out):
        raise ValueError(f"a feature name may not contain {_FEATURE_SEPARATOR!r}; got {list(out)}")
    return out


# ============================================================================== rows (OutcomeRows)

@dataclass(frozen=True, eq=False)
class OutcomeRows:
    """One value per innings 1-2 Delivery, in ``state`` order, of everything M4 needs besides the views.

    ``row``, ``match_id``, ``fold``, ``innings``, ``phase``, ``wickets``, ``label`` and
    ``bat_runs`` are int64; ``legal``, ``trainable`` and ``free_hit`` bool; ``pace_or_spin``
    float64 (PACE_OR_SPIN_CODES, NaN when missing). ``label`` is the class index of a Legal_Ball
    (``label_outcomes``) and NO_LABEL on a Wide or No_Ball. ``trainable`` marks the Legal_Balls
    that may be Training_Rows (``common.legal_training_mask``); they are also the held-out rows of
    the metrics. Every array is read-only. Build it with ``outcome_rows``.
    """

    row: np.ndarray
    match_id: np.ndarray
    fold: np.ndarray
    innings: np.ndarray
    phase: np.ndarray
    wickets: np.ndarray
    legal: np.ndarray
    trainable: np.ndarray
    label: np.ndarray
    bat_runs: np.ndarray
    free_hit: np.ndarray
    pace_or_spin: np.ndarray

    def __post_init__(self) -> None:
        kinds = {"legal": bool, "trainable": bool, "free_hit": bool, "pace_or_spin": np.float64}
        n_rows = None
        for item in fields(self):
            array = _frozen(getattr(self, item.name), kinds.get(item.name, np.int64))
            if array.ndim != 1:
                raise ValueError(f"OutcomeRows.{item.name} must be 1-D")
            if n_rows is None:
                n_rows = len(array)
            elif len(array) != n_rows:
                raise ValueError(f"OutcomeRows.{item.name} has {len(array):,} values; the other arrays have {n_rows:,}")
            object.__setattr__(self, item.name, array)
        if (self.trainable & ~self.legal).any():
            raise ValueError("OutcomeRows.trainable marks a Delivery that is not a Legal_Ball")
        labelled = self.label[self.legal]
        if len(labelled) and (labelled.min() < 0 or labelled.max() >= N_CLASSES):
            raise ValueError(f"OutcomeRows.label of a Legal_Ball must be a class index 0-{N_CLASSES - 1}")
        if (self.label[~self.legal] != NO_LABEL).any():
            raise ValueError(f"OutcomeRows.label of a Wide or No_Ball must be NO_LABEL ({NO_LABEL})")

    @property
    def n_rows(self) -> int:
        """The number of rows."""
        return len(self.row)


def pace_or_spin_codes(values: Any) -> np.ndarray:
    """The ``pace_or_spin`` feature of each value: PACE_OR_SPIN_CODES, NaN for a missing or other text.

    ``values`` is a 1-D array or Series of text, such as the ``pace_or_spin`` column of ``balls``.
    Returns a float64 array, one value per input value (module docstring, "Features").
    """
    texts = _series(values, "pace_or_spin").astype("string")
    codes = np.full(len(texts), np.nan)
    for text, code in PACE_OR_SPIN_CODES.items():
        codes[texts.isin([text]).to_numpy(dtype=bool)] = code
    return codes


def outcome_rows(balls: pd.DataFrame, state: pd.DataFrame, fold_table: pd.DataFrame) -> OutcomeRows:
    """The OutcomeRows of every ``state`` row (module docstring, "Rows").

    ``balls`` is the ``balls`` table of the ``reconstruct`` checkpoint (``row``, ``match_id``,
    ``innings``, ``cls``, ``runs``, ``bowler_wicket``, ``run_out``, ``free_hit``,
    ``pace_or_spin``, ``is_super_over`` and ``excluded``), with every Delivery or a superset of
    the ``state`` rows; ``state`` the ``state`` table (``row``, ``phase``, ``wickets``); and
    ``fold_table`` the Fold_Table of the ``folds`` checkpoint. The result follows the ``state``
    rows and their order.

    Raises ValueError when a column is absent, ``balls`` repeats a ``row`` or lacks a ``state``
    row, a ``state`` row lies outside innings 1 and 2, or a value is missing or malformed;
    TypeError for a column of the wrong kind; and DataError (exit code 3) for a match that is not
    in the Fold_Table.
    """
    _require(balls, _BALL_COLUMNS, "balls")
    _require(state, _STATE_COLUMNS, "state")
    index = pd.Index(_ints(balls["row"], "balls column 'row'"))
    if not index.is_unique:
        raise ValueError("balls must hold each row once")
    row = _ints(state["row"], "state column 'row'")
    at = index.get_indexer(row)
    if (at < 0).any():
        raise ValueError(f"{_rows(int((at < 0).sum()))} of state have no Delivery in balls; pass the tables of one Run")
    part = balls.iloc[at].reset_index(drop=True)
    innings = _ints(part["innings"], "balls column 'innings'")
    outside = ~np.isin(innings, STATE_INNINGS)
    if outside.any():
        raise ValueError(f"{_rows(int(outside.sum()))} of state lie outside innings 1 and 2")
    match_id = _ints(part["match_id"], "balls column 'match_id'")
    legal = _ints(part["cls"], "balls column 'cls'") == int(DeliveryClass.LEGAL)
    bat_runs = _ints(part["runs"], "balls column 'runs'", minimum=0)
    label = label_outcomes(bat_runs, part["bowler_wicket"], part["run_out"])
    n_rows = len(row)
    return OutcomeRows(
        row=row,
        match_id=match_id,
        fold=folds.fold_of(match_id, fold_table),
        innings=innings,
        phase=_ints(state["phase"], "state column 'phase'", n_rows),
        wickets=_ints(state["wickets"], "state column 'wickets'", n_rows, minimum=0),
        legal=legal,
        trainable=common.legal_training_mask(part),
        label=np.where(legal, label, NO_LABEL),
        bat_runs=bat_runs,
        free_hit=_flags(part["free_hit"], "balls column 'free_hit'"),
        pace_or_spin=pace_or_spin_codes(part["pace_or_spin"]),
    )


# ======================================================================= labels (11.14) and CRVs

def label_outcomes(bat_runs: Any, bowler_wicket: Any, run_out: Any) -> np.ndarray:
    """The Outcome_Model class of each Legal_Ball (criterion 14; module docstring, "Classes and labels").

    ``bat_runs`` (whole numbers, 0 or more), ``bowler_wicket`` and ``run_out`` (booleans: the
    ``balls`` flags of a Bowler_Wicket and a Run_Out) describe the same Legal_Balls. A ball is
    BOWLER_WICKET when it holds a Bowler_Wicket, otherwise RUN_OUT when it holds a Run_Out,
    otherwise the class of its bat runs: DOT, R1, R2, R3, R4 or R6 for 0, 1, 2, 3, 4 or 6 bat
    runs and OTHER_RUNS for any other value. Returns an int64 array of class indices into
    CLASSES, one per ball.

    Raises ValueError when the inputs differ in length or hold a missing or negative value, and
    TypeError for a non-numeric ``bat_runs`` or non-boolean flags.
    """
    runs = _ints(bat_runs, "bat_runs", minimum=0)
    bowler = _flags(bowler_wicket, "bowler_wicket", len(runs))
    run_out_flags = _flags(run_out, "run_out", len(runs))
    labels = np.full(len(runs), OTHER_RUNS, dtype=np.int64)
    for cls, value in RUN_CLASS_RUNS.items():
        labels[runs == value] = cls
    labels[run_out_flags] = RUN_OUT
    labels[bowler] = BOWLER_WICKET  # last, so a Bowler_Wicket wins over a Run_Out
    return labels


def class_run_values(bat_runs: Any, labels: Any) -> np.ndarray:
    """The Class_Run_Value of every class over one fit's Training_Rows (glossary; module docstring).

    ``bat_runs`` and ``labels`` (class indices into CLASSES, ``label_outcomes``) describe the
    Training_Rows of one fit: the Legal_Balls of its Training_Folds. Returns a float64 array of
    N_CLASSES values in CLASSES order: exactly 0, 1, 2, 3, 4 and 6 for the dot and run classes,
    and the mean bat runs of the rows of each of MEAN_VALUE_CLASSES.

    Raises ValueError when the inputs differ in length or hold a missing value, a label is not a
    class index, a row of a dot or run class has other bat runs (labels that ``label_outcomes``
    did not give), or a class of MEAN_VALUE_CLASSES has no row, so that its Class_Run_Value is
    undefined.
    """
    runs = _ints(bat_runs, "bat_runs", minimum=0)
    classes = _ints(labels, "labels", len(runs))
    if len(classes) and (classes.min() < 0 or classes.max() >= N_CLASSES):
        raise ValueError(f"labels must be class indices 0-{N_CLASSES - 1} into CLASSES")
    counts = np.bincount(classes, minlength=N_CLASSES)
    sums = np.bincount(classes, weights=runs.astype(np.float64), minlength=N_CLASSES)
    values = np.empty(N_CLASSES, dtype=np.float64)
    for cls, value in RUN_CLASS_RUNS.items():
        if sums[cls] != value * counts[cls]:
            raise ValueError(
                f"rows labelled {CLASSES[cls]!r} must all have {value} bat run(s); label the rows with label_outcomes"
            )
        values[cls] = float(value)
    empty = [CLASSES[cls] for cls in MEAN_VALUE_CLASSES if counts[cls] == 0]
    if empty:
        raise ValueError(
            f"the Class_Run_Value of {', '.join(map(repr, empty))} is undefined: the training Legal_Balls hold no row "
            "of the class"
        )
    for cls in MEAN_VALUE_CLASSES:
        values[cls] = sums[cls] / counts[cls]
    return values


# ================================================================================= features (11.2)

def _hand_x_style(values: Any) -> pd.Categorical:
    """``hand_x_style`` as a Categorical whose categories are its sorted distinct non-missing values."""
    texts = _series(values, "state column 'hand_x_style'").astype("string")
    categories = sorted(texts.dropna().unique().tolist())
    return pd.Categorical(texts, categories=categories)


def m4_features(view: pd.DataFrame, state: pd.DataFrame, rows: OutcomeRows) -> pd.DataFrame:
    """The M4_FEATURES of every row: the pre-ball state as M4 reads it (criterion 2; module docstring, "Features").

    ``view`` is a sandbox view V_k of the ``remaining_runs`` checkpoint (VIEW_FEATURES; checked
    with ``views.check_view``), ``state`` the ``state`` table (``row`` and STATE_FEATURES) and
    ``rows`` the OutcomeRows of ``outcome_rows`` (ROW_FEATURES), all with the same rows in the
    same order; the ``row`` columns of ``state`` and, when it has one, of ``view`` must equal
    ``rows.row``.

    Returns the M4_FEATURES on a RangeIndex, one row per row: ``phase`` and ``hand_x_style`` as
    pandas categorical columns (categories PHASE_CATEGORIES, and the sorted distinct values of
    ``state["hand_x_style"]``), every other feature as float64 with NaN for a missing value.
    Raises ValueError when a column is absent, the rows differ or a ``phase`` is not a Phase
    code, and TypeError for a column of the wrong kind.
    """
    if not isinstance(rows, OutcomeRows):
        raise TypeError(f"rows must be the OutcomeRows of outcome_rows; got {type(rows).__name__}")
    _require(view, VIEW_FEATURES, "view")
    _require(state, (_ROW_COLUMN, *STATE_FEATURES), "state")
    n_rows = rows.n_rows
    for frame, what in ((view, "view"), (state, "state")):
        if len(frame) != n_rows:
            raise ValueError(
                f"{what} has {_rows(len(frame))} but rows has {_rows(n_rows)}; pass a sandbox view, the state table "
                "and their OutcomeRows, which hold one row per innings 1-2 Delivery in the same order"
            )
        if _ROW_COLUMN in frame.columns and not np.array_equal(
            _ints(frame[_ROW_COLUMN], f"{what} column 'row'"), rows.row
        ):
            raise ValueError(f"{what} and rows hold different rows, or the same rows in another order")
    phase = _ints(state["phase"], "state column 'phase'")
    if not np.isin(phase, PHASE_CATEGORIES).all():
        raise ValueError(f"state column 'phase' must hold the Phase codes {list(PHASE_CATEGORIES)}")
    data: dict[str, Any] = {}
    for name in M4_FEATURES:
        if name == "phase":
            data[name] = pd.Categorical(phase, categories=list(PHASE_CATEGORIES))
        elif name == "hand_x_style":
            data[name] = _hand_x_style(state[name])
        elif name == "innings":
            data[name] = rows.innings.astype(np.float64)
        elif name == "pace_or_spin":
            data[name] = rows.pace_or_spin.astype(np.float64)
        elif name == "free_hit":
            data[name] = rows.free_hit.astype(np.float64)
        elif name in STATE_FEATURES:
            data[name] = _floats(state[name], f"state column {name!r}")
        else:
            data[name] = _floats(view[name], f"view column {name!r}")
    return pd.DataFrame(data, columns=list(M4_FEATURES))


# ================================================================================ Average_Skill

class AverageSkill(NamedTuple):
    """The Average_Skill of one fold (glossary): the theta of a typical player in one M1 fit.

    ``theta_bat`` is the mean theta_bat of the strikers of the fit's training Legal_Balls,
    weighted by Legal_Balls faced; ``theta_bowl`` the mean theta_bowl of the bowlers of those
    Legal_Balls, weighted by Legal_Balls bowled; ``m1_fit`` the id of the fit (M1(-k) for fold k).
    Build it with ``average_skill``.
    """

    theta_bat: float
    theta_bowl: float
    m1_fit: str


def average_skill(view: pd.DataFrame, k: int, row_fold: Any, m1_theta: pd.DataFrame) -> AverageSkill:
    """The Average_Skill of fold k (module docstring, "Average_Skill").

    ``view`` is the sandbox view V_k (its ``m1_fit`` column names the M1 fit of each row),
    ``row_fold`` the fold of each view row (``OutcomeRows.fold``) and ``m1_theta`` the
    ``m1_theta`` table of the ``skill`` checkpoint (``fit_id``, ``player_id``, ``theta_bat``,
    ``theta_bowl``, ``bat_balls``, ``bowl_balls``). The M1 fit is the one that supplied the theta
    values of the fold-k rows of V_k (M1(-k)). Over that fit's rows of ``m1_theta``, theta_bat is
    the mean ``theta_bat`` weighted by ``bat_balls`` (the fit's training Legal_Balls faced) and
    theta_bowl the mean ``theta_bowl`` weighted by ``bowl_balls`` (its training Legal_Balls
    bowled). A player without a Legal_Ball in a Role has weight 0 there.

    Raises ValueError when no view row lies in fold k, the fold-k rows name no M1 fit or more
    than one, ``m1_theta`` holds no row of that fit or repeats a player in it, a ball count is
    missing, negative or not whole, a theta with a ball count above 0 is missing or not finite,
    or the fit's rows count no Legal_Ball faced or bowled, or a different number of Legal_Balls
    faced and bowled (each training Legal_Ball has one striker and one bowler, so the table then
    lacks some of the fit's players); TypeError for a malformed argument.
    """
    k = _fold_number(k, "k")
    _require(view, (views.M1_FIT,), "view")
    folds_of_rows = _ints(row_fold, "row_fold", len(view))
    served = folds_of_rows == k
    if not served.any():
        raise ValueError(f"no row of view V_{k} lies in fold {k}, so fold {k} has no Average_Skill")
    named = pd.Series(view[views.M1_FIT].array)[served]
    ids = named.dropna().unique().tolist()
    if len(ids) != 1 or named.isna().any() or not isinstance(ids[0], str):
        raise ValueError(
            f"the fold-{k} rows of view V_{k} must all name one M1 fit (M1(-{k})) in {views.M1_FIT}; they name "
            f"{ids[:_SHOWN]} and {int(named.isna().sum()):,} row(s) name none"
        )
    fit_id = ids[0]
    _require(m1_theta, _THETA_TABLE_COLUMNS, "m1_theta")
    chosen = np.flatnonzero(m1_theta["fit_id"].astype("string").eq(fit_id).fillna(False).to_numpy(dtype=bool))
    if not len(chosen):
        raise ValueError(f"m1_theta holds no row of the M1 fit {fit_id!r}; pass the m1_theta table of the skill checkpoint")
    part = m1_theta.iloc[chosen]
    players = _ints(part["player_id"], "m1_theta column 'player_id'")
    repeated = pd.Index(players).duplicated()
    if repeated.any():
        shown = ", ".join(map(str, np.unique(players[repeated])[:_SHOWN].tolist()))
        raise ValueError(f"m1_theta must hold each player once for the M1 fit {fit_id!r}; it repeats player(s) {shown}")
    means: dict[str, float] = {}
    totals: dict[str, int] = {}
    for name, weight in _AVERAGE_WEIGHT.items():
        balls = _ints(part[weight], f"m1_theta column {weight!r}", minimum=0)
        theta = _floats(part[name], f"m1_theta column {name!r}")
        weighted = balls > 0
        if not np.isfinite(theta[weighted]).all():
            raise ValueError(
                f"m1_theta holds a missing or non-finite {name} for {_rows(int((~np.isfinite(theta[weighted])).sum()))} "
                f"of the M1 fit {fit_id!r} with {weight} above 0"
            )
        totals[weight] = int(balls.sum())
        if not totals[weight]:
            raise ValueError(
                f"the m1_theta rows of the M1 fit {fit_id!r} count no training Legal_Ball in {weight}, so the {name} "
                "of Average_Skill is undefined"
            )
        means[name] = float(np.dot(balls[weighted].astype(np.float64), theta[weighted]) / totals[weight])
    if totals["bat_balls"] != totals["bowl_balls"]:
        raise ValueError(
            f"the m1_theta rows of the M1 fit {fit_id!r} count {totals['bat_balls']:,} training Legal_Balls faced but "
            f"{totals['bowl_balls']:,} bowled; each Legal_Ball has one striker and one bowler, so pass the full table "
            "of the skill checkpoint, with every player of the fit"
        )
    return AverageSkill(theta_bat=means["theta_bat"], theta_bowl=means["theta_bowl"], m1_fit=fit_id)


def _check_skill(value: Any, what: str) -> AverageSkill:
    if not isinstance(value, AverageSkill):
        raise TypeError(f"{what} must be an AverageSkill (average_skill); got {type(value).__name__}")
    for name in ("theta_bat", "theta_bowl"):
        number = getattr(value, name)
        if isinstance(number, (bool, np.bool_)) or not isinstance(number, Real) or not np.isfinite(number):
            raise ValueError(f"{what}.{name} must be a finite number; got {number!r}")
    return value


# ======================================================================== variants (11.7, 11.8)

def variant_features(X: pd.DataFrame, variant: str, average_skill: AverageSkill) -> pd.DataFrame:
    """The feature rows of one prediction variant (module docstring, "Prediction variants").

    ``X`` holds actual M4 feature rows (m4_features, or any subset of its rows and columns that
    includes the features the variant changes); ``average_skill`` is the Average_Skill of the
    fold of those rows (``average_skill``). Returns a copy of ``X`` in which the features that
    VARIANTS[variant] names are replaced: ``theta_striker`` by ``average_skill.theta_bat`` and
    ``theta_bowler`` by ``average_skill.theta_bowl`` (the typical player's theta in the fold's M1
    fit), and ``free_hit`` by 1.0 (True) or 0.0 (False). Every other value, index and dtype is
    unchanged.

    Raises ValueError for an unknown variant or a changed feature that ``X`` lacks, and TypeError
    for a malformed argument.
    """
    if not isinstance(variant, str) or variant not in VARIANTS:
        raise ValueError(f"unknown M4 variant {variant!r}; the variants are {', '.join(VARIANT_NAMES)}")
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"X must be a DataFrame of M4 features; got {type(X).__name__}")
    skill = _check_skill(average_skill, "average_skill")
    changes = VARIANTS[variant]
    _require(X, changes, "X")
    out = X.copy()
    for column, setting in changes.items():
        value = float(getattr(skill, _AVERAGE_FIELD[column])) if setting == AVERAGE else (1.0 if setting else 0.0)
        out[column] = np.full(len(out), value, dtype=np.float64)
    return out


# ================================================================================ families, fits

@dataclass(frozen=True)
class OutcomeFamily:
    """One Candidate_Model family of M4 (module docstring, "Families").

    * ``name``: the family name in the FitRegistry and the artifact names (``lgbm``; 1-16
      characters from ``a-z`` and ``0-9``, starting with a letter).
    * ``early_stopping``: the family's EarlyStopping, or None for a family without one.
    * ``fitter``: ``fitter(X, y)`` returns the fit function for ``common.fit_registered``, where
      ``X`` is a feature frame of every row (feature columns in the fit's order) and ``y`` the
      class index of every row (NO_LABEL on rows that cannot train).
    * ``predict``: ``predict(model, X)`` returns the class probabilities of the rows of ``X``
      (same columns as in the fit), an array of shape (rows, N_CLASSES) in CLASSES order.
    * ``iterations``: ``iterations(model)`` gives the size of a fitted model that OutcomeFold keeps
      as ``best_iteration``: the boosting rounds kept (XGBoost) or the trees (random forest). None
      reads the model's own ``best_iteration``, the rounds a LightGBM Booster kept.
    """

    name: str
    early_stopping: common.EarlyStopping | None
    fitter: Callable[[pd.DataFrame, np.ndarray], common.FitFunction]
    predict: Callable[[Any, pd.DataFrame], np.ndarray]
    iterations: Callable[[Any], int] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _FAMILY_NAME.fullmatch(self.name):
            raise ValueError(
                f"OutcomeFamily name must be 1-16 characters from a-z and 0-9, starting with a letter; got {self.name!r}"
            )
        if self.early_stopping is not None and not isinstance(self.early_stopping, common.EarlyStopping):
            raise TypeError(f"OutcomeFamily early_stopping must be an EarlyStopping or None; got {self.early_stopping!r}")
        if not callable(self.fitter) or not callable(self.predict):
            raise TypeError("OutcomeFamily fitter and predict must be functions")
        if self.iterations is not None and not callable(self.iterations):
            raise TypeError("OutcomeFamily iterations must be a function or None")


def lgbm_predict(model: Any, X: pd.DataFrame) -> np.ndarray:
    """The class probabilities of a LightGBM multiclass Booster for the rows of ``X``, with its best round.

    Returns a float64 array of shape (rows, N_CLASSES). The Booster maps the categories of
    ``X``'s categorical columns onto those of its fit by value.
    """
    if not len(X):
        return np.zeros((0, N_CLASSES), dtype=np.float64)
    return np.asarray(model.predict(X), dtype=np.float64)


def lgbm_family(settings: Settings) -> OutcomeFamily:
    """The LightGBM OutcomeFamily (module docstring, "Fits").

    ``common.lgbm_setup(settings, "m4", "multiclass", num_class=N_CLASSES)`` builds the
    parameters from ``models.m4``, ``runtime.n_threads``, ``runtime.deterministic`` and
    ``seeds.m4_lgbm``; the fitter is ``common.boosting_fitter`` with that setup (early stopping
    on the multiclass log loss of the held-out matches) and the predict function
    ``lgbm_predict``. Raises the errors of lgbm_setup, such as KeyError for a setting absent from
    the Config.
    """
    setup = common.lgbm_setup(settings, _CONFIG_MODEL, _OBJECTIVE, num_class=N_CLASSES)
    return OutcomeFamily(
        name=LGBM_FAMILY,
        early_stopping=setup.early_stopping,
        fitter=functools.partial(common.boosting_fitter, setup),
        predict=lgbm_predict,
    )


def _category_lists(X: pd.DataFrame) -> dict[str, list[Any]]:
    """Feature -> the categories of each pandas categorical column of ``X``, in code order."""
    return {
        str(name): X[name].cat.categories.tolist()
        for name in X.columns
        if isinstance(X[name].dtype, pd.CategoricalDtype)
    }


def _xgb_fitter(setup: common.BoosterSetup, X: pd.DataFrame, y: Any) -> common.FitFunction:
    """``common.boosting_fitter`` of an XGBoost setup, whose Booster also keeps the category lists of ``X``.

    The returned function fits as boosting_fitter does (early stopping on the held-out matches, the
    Booster cut to its best round) and stores ``_category_lists(X)`` as JSON in the Booster
    attribute XGB_CATEGORIES_ATTRIBUTE, which XGBoost JSON saves with the model.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"X must be a DataFrame of M4 features; got {type(X).__name__}")
    fit = common.boosting_fitter(setup, X, y)
    categories = json.dumps(_category_lists(X), ensure_ascii=True)

    def run(rows: common.FitRows) -> Any:
        booster = fit(rows)
        booster.set_attr(**{XGB_CATEGORIES_ATTRIBUTE: categories})
        return booster

    return run


def _recoded(X: pd.DataFrame, categories: Mapping[str, Sequence[Any]]) -> pd.DataFrame:
    """``X`` with each categorical column named in ``categories`` mapped by value onto those categories.

    A value outside them becomes missing. ``X`` itself is returned when every list already matches.
    """
    changed: dict[str, pd.Series] = {}
    for name, values in categories.items():
        if name not in X.columns or not isinstance(X[name].dtype, pd.CategoricalDtype):
            continue
        if X[name].cat.categories.tolist() != list(values):
            changed[name] = X[name].cat.set_categories(list(values))
    if not changed:
        return X
    out = X.copy()
    for name, column in changed.items():
        out[name] = column
    return out


def xgb_predict(model: Any, X: pd.DataFrame, *, n_threads: int) -> np.ndarray:
    """The class probabilities of an XGBoost multiclass Booster of ``xgb_family`` for the rows of ``X``.

    The Booster is cut to its best round. Before XGBoost reads ``X`` (a DMatrix with
    ``enable_categorical=True`` and ``n_threads`` threads), each categorical column is mapped by
    value onto the category lists that the fit stored in XGB_CATEGORIES_ATTRIBUTE, so the category
    codes mean what they meant in the fit and a category that the fit never saw is read as missing.
    Returns XGBoost's float32 softmax values as a float64 array of shape (rows, N_CLASSES),
    unrescaled.
    """
    if not len(X):
        return np.zeros((0, N_CLASSES), dtype=np.float64)
    import xgboost as xgb

    stored = model.attr(XGB_CATEGORIES_ATTRIBUTE)
    frame = X if stored is None else _recoded(X, json.loads(stored))
    matrix = xgb.DMatrix(frame, nthread=n_threads, enable_categorical=True)
    return np.asarray(model.predict(matrix), dtype=np.float64)


def _xgb_rounds(model: Any) -> int:
    """The boosting rounds that an XGBoost Booster cut to its best round holds."""
    return int(model.num_boosted_rounds())


def xgb_family(settings: Settings) -> OutcomeFamily:
    """The XGBoost OutcomeFamily, an M4 Alternative_Model (module docstring, "Families").

    ``common.xgb_setup(settings, "m4", "multiclass", num_class=N_CLASSES)`` builds the parameters
    from ``models.xgb.m4``, ``runtime.n_threads`` and ``seeds.m4_xgb``; the fitter is
    ``common.boosting_fitter`` with that setup (early stopping on the multiclass log loss of whole
    matches held out from the training folds) and keeps the category lists on the Booster; the
    predict function is ``xgb_predict``. Raises the errors of xgb_setup.
    """
    setup = common.xgb_setup(settings, _CONFIG_MODEL, _OBJECTIVE, num_class=N_CLASSES)
    return OutcomeFamily(
        name=XGB_FAMILY,
        early_stopping=setup.early_stopping,
        fitter=functools.partial(_xgb_fitter, setup),
        predict=functools.partial(xgb_predict, n_threads=int(setup.params["nthread"])),
        iterations=_xgb_rounds,
    )


def _rf_pipeline(setup: common.ForestSetup, categorical: Sequence[str], numeric: Sequence[str]) -> Any:
    """A new random-forest pipeline: one-hot categories, median-imputed numbers, then the forest.

    Both encoders are fitted on the rows the pipeline is fitted on, the fit's Training_Rows. The
    one-hot encoder gives an all-zero row to a category that those rows lack.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    encode = ColumnTransformer(
        [
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64), list(categorical)),
            ("median", SimpleImputer(strategy="median", keep_empty_features=True), list(numeric)),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return Pipeline([("encode", encode), ("forest", RandomForestClassifier(**setup.params))])


def _rf_fitter(setup: common.ForestSetup, X: pd.DataFrame, y: Any) -> common.FitFunction:
    """The fit function of the random-forest pipeline (``_rf_pipeline``) for ``common.fit_registered``.

    The pandas categorical columns of ``X`` are one-hot encoded and the other columns median-imputed.
    The pipeline is fitted on ``rows.train`` (the forest uses no early stopping) and returned with
    its forest set to predict with ``setup.predict_n_jobs`` threads.
    """
    if not isinstance(setup, common.ForestSetup):
        raise TypeError(f"setup must be a ForestSetup (common.rf_setup); got {type(setup).__name__}")
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"X must be a DataFrame of M4 features; got {type(X).__name__}")
    categorical = [str(name) for name in X.columns if isinstance(X[name].dtype, pd.CategoricalDtype)]
    numeric = [str(name) for name in X.columns if str(name) not in categorical]
    fit = common.estimator_fitter(functools.partial(_rf_pipeline, setup, categorical, numeric), X, y)

    def run(rows: common.FitRows) -> Any:
        pipeline = fit(rows)
        pipeline[-1].set_params(n_jobs=setup.predict_n_jobs)  # 1 thread in deterministic mode
        return pipeline

    return run


def rf_predict(model: Any, X: pd.DataFrame) -> np.ndarray:
    """The class probabilities of a random-forest pipeline of ``rf_family`` for the rows of ``X``.

    ``predict_proba`` gives one column per class of the fit's Training_Rows (``classes_``); each is
    placed in its CLASSES column, so a class that those rows lack gets probability 0. Returns a
    float64 array of shape (rows, N_CLASSES), unrescaled.
    """
    if not len(X):
        return np.zeros((0, N_CLASSES), dtype=np.float64)
    classes = np.asarray(model.classes_)
    if (
        classes.ndim != 1
        or not np.issubdtype(classes.dtype, np.integer)
        or len(np.unique(classes)) != len(classes)
        or (len(classes) and (classes.min() < 0 or classes.max() >= N_CLASSES))
    ):
        raise ValueError(f"the random forest's classes_ must be distinct class indices 0-{N_CLASSES - 1}; got {classes}")
    proba = np.asarray(model.predict_proba(X), dtype=np.float64)
    out = np.zeros((len(X), N_CLASSES), dtype=np.float64)
    out[:, classes.astype(np.int64)] = proba
    return out


def _rf_trees(model: Any) -> int:
    """The trees of a fitted random-forest pipeline."""
    return len(model[-1].estimators_)


def rf_family(settings: Settings) -> OutcomeFamily:
    """The random-forest OutcomeFamily, an M4 Alternative_Model (module docstring, "Families").

    ``common.rf_setup(settings, "m4")`` builds the RandomForestClassifier arguments from
    ``models.rf.m4`` (trees, maximum depth, minimum rows per leaf, features per split),
    ``runtime.n_threads`` and ``seeds.m4_rf``; the fitter is ``_rf_fitter`` (one-hot categories
    and median imputation fitted on the Training_Rows), without early stopping; the predict
    function is ``rf_predict``. Raises the errors of rf_setup.
    """
    setup = common.rf_setup(settings, _CONFIG_MODEL)
    return OutcomeFamily(
        name=RF_FAMILY,
        early_stopping=None,
        fitter=functools.partial(_rf_fitter, setup),
        predict=rf_predict,
        iterations=_rf_trees,
    )


_FAMILY_BUILDERS: Mapping[str, Callable[[Settings], OutcomeFamily]] = MappingProxyType(
    {LGBM_FAMILY: lgbm_family, XGB_FAMILY: xgb_family, RF_FAMILY: rf_family}
)


def candidate_family(name: str, settings: Settings) -> OutcomeFamily:
    """The OutcomeFamily of M4 Candidate_Model ``name`` (a name of FAMILIES) with its Config settings.

    ``lgbm`` is ``lgbm_family``, ``xgb`` ``xgb_family`` and ``rf`` ``rf_family``. Use it with
    ``load_fold_models`` to rebuild any candidate's fold models. Raises ValueError for another
    name and the errors of the family's builder.
    """
    if name not in _FAMILY_BUILDERS:
        raise ValueError(f"the M4 families are {', '.join(FAMILIES)}; got {name!r}")
    return _FAMILY_BUILDERS[name](settings)


@dataclass(frozen=True, eq=False)
class OutcomeFold:
    """The outer M4 fit of one fold, with the values that serve the fold's rows.

    ``fold`` is k; ``fit_id`` the FitRegistry id; ``model`` the fitted model; ``average_skill``
    the fold's Average_Skill; ``class_run_values`` the Class_Run_Value of each class and
    ``class_counts`` the Training_Rows of each class, both of length N_CLASSES in CLASSES order
    (read-only); ``best_iteration`` the size of the model (``OutcomeFamily.iterations``): the
    boosting rounds kept by LightGBM or XGBoost, the trees of a random forest, -1 for a model
    without either; ``n_fit_rows`` and ``n_early_stopping_rows`` the two parts of the
    Training_Rows.
    """

    fold: int
    fit_id: str
    model: Any
    average_skill: AverageSkill
    class_run_values: np.ndarray
    class_counts: np.ndarray
    best_iteration: int
    n_fit_rows: int
    n_early_stopping_rows: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "fold", _fold_number(self.fold, "OutcomeFold fold"))
        _check_skill(self.average_skill, "OutcomeFold average_skill")
        values = _frozen(self.class_run_values, np.float64)
        counts = _frozen(self.class_counts, np.int64)
        if values.shape != (N_CLASSES,) or not np.isfinite(values).all():
            raise ValueError(f"OutcomeFold class_run_values must be {N_CLASSES} finite numbers in CLASSES order")
        if counts.shape != (N_CLASSES,) or (counts < 0).any():
            raise ValueError(f"OutcomeFold class_counts must be {N_CLASSES} counts in CLASSES order")
        object.__setattr__(self, "class_run_values", values)
        object.__setattr__(self, "class_counts", counts)
        for name in ("best_iteration", "n_fit_rows", "n_early_stopping_rows"):
            object.__setattr__(self, name, int(getattr(self, name)))

    @property
    def n_train_rows(self) -> int:
        """The fit's Training_Rows: its fit rows and early-stopping rows together."""
        return int(self.class_counts.sum())


@dataclass(frozen=True, eq=False)
class OutcomeFoldModels:
    """The outer fits of one M4 family (``fit_outcome``, ``load_fold_models``).

    ``family`` is the family name, ``features`` the feature columns of every fit in fit order,
    ``folds`` the OutcomeFold of each fold k (read-only, ascending by k) and ``predict`` the
    family's predict function.
    """

    family: str
    features: tuple[str, ...]
    folds: Mapping[int, OutcomeFold]
    predict: Callable[[Any, pd.DataFrame], np.ndarray]

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", _check_feature_names(self.features))
        if not isinstance(self.folds, Mapping) or not self.folds:
            raise ValueError("OutcomeFoldModels folds must map at least one fold to its OutcomeFold")
        ordered: dict[int, OutcomeFold] = {}
        for k in sorted(self.folds):
            fold = self.folds[k]
            if not isinstance(fold, OutcomeFold) or fold.fold != k:
                raise ValueError(f"OutcomeFoldModels folds[{k!r}] must be the OutcomeFold of fold {k!r}")
            ordered[int(k)] = fold
        object.__setattr__(self, "folds", MappingProxyType(ordered))
        if not callable(self.predict):
            raise TypeError("OutcomeFoldModels predict must be a function")

    def fold(self, k: int) -> OutcomeFold:
        """The OutcomeFold of fold ``k``; ValueError when the family has no fit for it."""
        if k not in self.folds:
            raise ValueError(f"the M4 {self.family} models have no fit for fold {k}; they cover folds {list(self.folds)}")
        return self.folds[k]

    @property
    def fit_ids(self) -> tuple[str, ...]:
        """The fit ids, by fold."""
        return tuple(fold.fit_id for fold in self.folds.values())

    def class_run_values_for(self, row_fold: Any) -> np.ndarray:
        """The Class_Run_Values of each row's fold: a float64 array of shape (rows, N_CLASSES)."""
        ks = _ints(row_fold, "row_fold")
        out = np.empty((len(ks), N_CLASSES), dtype=np.float64)
        for k in np.unique(ks).tolist():
            out[ks == k] = self.fold(k).class_run_values
        return out


def _outer_specs(plan: Any) -> list[FitSpec]:
    if isinstance(plan, (str, bytes, Mapping)) or not isinstance(plan, Iterable):
        raise TypeError("plan must be the FitSpecs of folds.sandbox_plan(D15)")
    specs = list(plan)
    if not all(isinstance(spec, FitSpec) for spec in specs):
        raise TypeError("plan must hold folds.FitSpec objects, as folds.sandbox_plan(D15) returns them")
    outer = [spec for spec in specs if spec.kind == folds.OUTER]
    if not outer:
        raise ValueError("plan holds no outer FitSpec; pass folds.sandbox_plan(D15)")
    ks = [spec.sandbox_k for spec in outer]
    if len(set(ks)) != len(ks):
        raise ValueError(f"plan holds more than one outer FitSpec for a fold: {ks}")
    return outer


def _feature_frame(features: Any, k: int, n_rows: int, names: tuple[str, ...]) -> pd.DataFrame:
    """``features[k]`` with the columns ``names`` in that order; ValueError unless it has ``n_rows`` rows."""
    if not isinstance(features, Mapping) or k not in features:
        raise ValueError(f"features must map fold {k} to the M4 feature frame built from view V_{k}")
    frame = features[k]
    _require(frame, names, f"features[{k}]")
    if len(frame) != n_rows:
        raise ValueError(f"features[{k}] has {_rows(len(frame))}, but {_rows(n_rows)} are needed, one per Delivery")
    return frame[list(names)]


def _best_iteration(model: Any) -> int:
    best = getattr(model, "best_iteration", None)
    if isinstance(best, (bool, np.bool_)) or not isinstance(best, Integral):
        return -1
    return int(best)


def fit_outcome(
    registry: FitRegistry,
    plan: Sequence[FitSpec],
    family: OutcomeFamily,
    rows: OutcomeRows,
    features: Mapping[int, pd.DataFrame],
    average_skills: Mapping[int, AverageSkill],
    *,
    feature_names: Sequence[str] = M4_FEATURES,
) -> OutcomeFoldModels:
    """The outer M4 fits of one family, one per outer FitSpec of ``plan`` (criteria 1 and 3; module docstring, "Fits").

    ``registry`` is the Run's FitRegistry (``ctx.registry``), ``plan`` the FitSpecs of
    ``folds.sandbox_plan(D15)`` (the inner ones are ignored: M4 has none), ``family`` the
    OutcomeFamily (``lgbm_family``, ``xgb_family`` or ``rf_family``), ``rows`` the OutcomeRows,
    ``features`` the M4 feature frame
    of every row built from view V_k for each fold k (``m4_features``), and ``average_skills`` the
    Average_Skill of each fold k (``average_skill``). ``feature_names`` are the frame columns the
    fits read, in fit order: M4_FEATURES, which Stage-2 factor F47 extends.

    For outer spec k: the Class_Run_Values and class counts of its Training_Rows (the
    ``trainable`` rows of the folds other than k) are computed first, so a class without a
    Training_Row stops the fit before it runs; then ``common.fit_registered`` registers the fit
    (model MODEL, family ``family.name``, every match of fold k predicted) and runs
    ``family.fitter(features[k], rows.label)``.

    Returns the OutcomeFoldModels. Raises FitLeakageError (exit code 4) when the leakage guard
    rejects a fit, ValueError for a missing frame, Average_Skill or column, a class without a
    Training_Row, or Training_Rows that differ from the ``trainable`` rows of the train folds (a
    Fold_Table other than the one ``rows`` used), TypeError for a malformed argument, and
    whatever the family's fit raises.
    """
    if not isinstance(family, OutcomeFamily):
        raise TypeError(f"family must be an OutcomeFamily (lgbm_family); got {type(family).__name__}")
    if not isinstance(rows, OutcomeRows):
        raise TypeError(f"rows must be the OutcomeRows of outcome_rows; got {type(rows).__name__}")
    if not isinstance(average_skills, Mapping):
        raise TypeError("average_skills must map each fold to its AverageSkill")
    names = _check_feature_names(feature_names)
    fitted_folds: dict[int, OutcomeFold] = {}
    for spec in _outer_specs(plan):
        k = spec.sandbox_k
        X = _feature_frame(features, k, rows.n_rows, names)
        if k not in average_skills:
            raise ValueError(f"average_skills has no Average_Skill for fold {k}")
        skill = _check_skill(average_skills[k], f"average_skills[{k}]")
        expected = np.flatnonzero(rows.trainable & np.isin(rows.fold, spec.train_folds))
        labels = rows.label[expected]
        values = class_run_values(rows.bat_runs[expected], labels)
        fitted = common.fit_registered(
            registry,
            spec,
            model=MODEL,
            family=family.name,
            match_ids=rows.match_id,
            eligible=rows.trainable,
            predicted_match_ids=rows.match_id[rows.fold == k],
            early_stopping=family.early_stopping,
            fit=family.fitter(X, rows.label),
        )
        if not np.array_equal(fitted.rows.training, expected):
            raise ValueError(
                f"the Training_Rows of fit {fitted.fit_id!r} differ from the trainable rows of folds "
                f"{list(spec.train_folds)}; the FitRegistry and rows must use the same Fold_Table"
            )
        fitted_folds[k] = OutcomeFold(
            fold=k,
            fit_id=fitted.fit_id,
            model=fitted.model,
            average_skill=skill,
            class_run_values=values,
            class_counts=np.bincount(labels, minlength=N_CLASSES),
            best_iteration=(
                _best_iteration(fitted.model) if family.iterations is None else family.iterations(fitted.model)
            ),
            n_fit_rows=len(fitted.rows.train),
            n_early_stopping_rows=len(fitted.rows.early_stopping),
        )
    return OutcomeFoldModels(family=family.name, features=names, folds=fitted_folds, predict=family.predict)


# ======================================================================== predictions (11.1, 11.5)

def variant_probabilities(
    models: OutcomeFoldModels,
    features: Mapping[int, pd.DataFrame],
    row_fold: Any,
    *,
    variants: Sequence[str] = VARIANT_NAMES,
    finite: bool = True,
) -> dict[str, np.ndarray]:
    """The class probabilities of every row in each variant, as the family's models give them (no rescaling).

    ``models`` is an OutcomeFoldModels, ``row_fold`` the fold of every row (``OutcomeRows.fold``)
    and ``features`` the M4 feature frame of every row built from view V_k for each fold k that
    ``row_fold`` holds (``m4_features``; with its columns changed, for example, by a
    robustness case). The rows of fold k are predicted by the fit of fold k from ``features[k]``,
    with ``variant_features`` and the fold's Average_Skill applied for each variant. ``finite``
    False lets missing and non-finite predictions through, for a caller that counts them, such as
    the failure checks of the Model_Choice_Checker (Requirement 23, criterion 12).

    Returns a dict from each name of ``variants`` (in that order) to a float64 array of shape
    (rows, N_CLASSES), rows in ``row_fold`` order and columns in CLASSES order. Raises ValueError
    for an unknown variant, a fold without a fit or a frame, a frame of the wrong length or
    without a feature, or predictions of the wrong shape or (``finite`` True) not finite.
    """
    if not isinstance(models, OutcomeFoldModels):
        raise TypeError(f"models must be an OutcomeFoldModels (fit_outcome); got {type(models).__name__}")
    names = _check_variants(variants)
    ks = _ints(row_fold, "row_fold")
    n_rows = len(ks)
    out = {name: np.full((n_rows, N_CLASSES), np.nan) for name in names}
    for k in np.unique(ks).tolist():
        fold = models.fold(k)
        served = np.flatnonzero(ks == k)
        X = _feature_frame(features, k, n_rows, models.features).iloc[served]
        for name in names:
            values = models.predict(fold.model, variant_features(X, name, fold.average_skill))
            out[name][served] = _prob_matrix(
                values, len(served), f"the {name} predictions of fit {fold.fit_id!r}", finite=finite
            )
    return out


def predict_variants(
    models: OutcomeFoldModels,
    features: Mapping[int, pd.DataFrame],
    row_fold: Any,
    log: common.WarningLog,
    *,
    variants: Sequence[str] = VARIANT_NAMES,
) -> dict[str, np.ndarray]:
    """The six M4 prediction variants of every row, each row rescaled to sum to 1 (criteria 1, 5, 7 and 8).

    The arguments are those of ``variant_probabilities``, plus ``log``, the Run_Log
    (``ctx.log``) or any object with ``warning``. Every variant array is rescaled row by row with
    ``common.renormalise``, which warns once per variant when a row summed to a value more than
    1e-6 away from 1. Returns a dict from variant name to a float64 array of shape (rows,
    N_CLASSES) whose rows sum to 1 (module docstring, "Prediction variants"). Raises the errors of
    variant_probabilities, and ValueError for a row with a negative probability or a sum of 0.
    """
    raw = variant_probabilities(models, features, row_fold, variants=variants)
    return {
        name: common.renormalise(values, log, what=f"M4 {models.family} {name} class probabilities")
        for name, values in raw.items()
    }


# ============================================================================== metrics (11.6)

def expected_bat_runs(probs: Any, class_run_values: Any) -> np.ndarray:
    """Sum over the classes of probability x Class_Run_Value, for each row of ``probs``.

    ``probs`` has shape (rows, N_CLASSES); ``class_run_values`` is one fold's N_CLASSES values or
    one row of values per row of ``probs`` (``OutcomeFoldModels.class_run_values_for``). Returns
    a float64 array, one value per row. Raises ValueError for arrays of the wrong shape.
    """
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != N_CLASSES:
        raise ValueError(f"probs must have shape (rows, {N_CLASSES}); got {p.shape}")
    values = np.asarray(class_run_values, dtype=np.float64)
    if values.shape == (N_CLASSES,):
        return p @ values
    if values.shape == p.shape:
        return np.einsum("ij,ij->i", p, values)
    raise ValueError(f"class_run_values must have shape ({N_CLASSES},) or {p.shape}; got {values.shape}")


def phase_baseline(rows: OutcomeRows) -> np.ndarray:
    """The Phase-only baseline of every row (criterion 6(b); module docstring, "Metrics").

    A row of fold k gets the class frequencies of the ``trainable`` rows of the other folds (the
    Training_Rows of fold k's M4 fit) in its Phase. Returns a float64 array of shape (rows,
    N_CLASSES) in CLASSES order; a row whose Phase has no such Training_Row gets NaN.
    """
    if not isinstance(rows, OutcomeRows):
        raise TypeError(f"rows must be the OutcomeRows of outcome_rows; got {type(rows).__name__}")
    out = np.full((rows.n_rows, N_CLASSES), np.nan)
    for k in np.unique(rows.fold).tolist():
        train = rows.trainable & (rows.fold != k)
        target = rows.fold == k
        for phase in np.unique(rows.phase).tolist():
            in_phase = rows.phase == phase
            counts = np.bincount(rows.label[train & in_phase], minlength=N_CLASSES).astype(np.float64)
            if counts.sum() > 0:
                out[target & in_phase] = counts / counts.sum()
    return out


def log_loss_table(probs: Any, rows: OutcomeRows, *, candidate: str) -> pd.DataFrame:
    """The pooled held-out multiclass log loss of a candidate and of the Phase-only baseline (criterion 6(a), (b)).

    ``probs`` holds the ``actual`` probabilities of every row (shape (rows, N_CLASSES)), each row
    from the outer fit of its fold; ``candidate`` names the candidate, such as
    ``predictions_table(family)``. Both log losses are ``common.pooled_log_loss`` over the
    ``trainable`` rows of all folds. Returns the LOG_LOSS_COLUMNS, one row for the candidate and
    one for PHASE_BASELINE. Raises ValueError for probabilities of the wrong shape or a baseline
    that is undefined on an evaluated row.
    """
    if not isinstance(rows, OutcomeRows):
        raise TypeError(f"rows must be the OutcomeRows of outcome_rows; got {type(rows).__name__}")
    if not isinstance(candidate, str) or not candidate:
        raise TypeError(f"candidate must be a non-empty name; got {candidate!r}")
    p = _prob_matrix(probs, rows.n_rows, "probs")
    evaluated = rows.trainable
    labels = rows.label[evaluated]
    baseline = phase_baseline(rows)[evaluated]
    n_rows = int(evaluated.sum())
    return pd.DataFrame(
        {
            "model": [candidate, PHASE_BASELINE],
            "n_legal_balls": np.array([n_rows, n_rows], dtype=np.int64),
            "log_loss": [common.pooled_log_loss(labels, p[evaluated]), common.pooled_log_loss(labels, baseline)],
        },
        columns=list(LOG_LOSS_COLUMNS),
    )


def _check_buckets(buckets: Any) -> tuple[tuple[int, int], ...]:
    """The D17 Wickets_Buckets as ascending, disjoint, inclusive (lowest, highest) pairs."""
    if isinstance(buckets, (str, bytes)) or not isinstance(buckets, Iterable):
        raise TypeError("wicket_buckets (D17) must be a list of [lowest, highest] pairs of wickets fallen")
    pairs: list[tuple[int, int]] = []
    for item in buckets:
        if isinstance(item, (str, bytes)) or not isinstance(item, Sequence) or len(item) != 2:
            raise ValueError(f"each Wickets_Bucket (D17) must be a [lowest, highest] pair; got {item!r}")
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in item):
            raise TypeError(f"Wickets_Bucket (D17) bounds must be whole numbers; got {item!r}")
        low, high = int(item[0]), int(item[1])
        if low < 0 or high < low:
            raise ValueError(f"Wickets_Bucket (D17) {item!r} must satisfy 0 <= lowest <= highest")
        pairs.append((low, high))
    if not pairs:
        raise ValueError("wicket_buckets (D17) holds no Wickets_Bucket")
    for (_, high), (low, _) in zip(pairs, pairs[1:]):
        if low <= high:
            raise ValueError(f"the Wickets_Buckets (D17) {[list(p) for p in pairs]} must ascend without overlaps")
    return tuple(pairs)


def wickets_bucket_codes(wickets: Any, wicket_buckets: Any) -> np.ndarray:
    """The Wickets_Bucket (D17) position of each wickets-fallen value (glossary).

    ``wicket_buckets`` is D17 (``design_choices.D17_wicket_buckets``): inclusive [lowest, highest]
    ranges of wickets fallen in ascending order. Returns an int64 array of positions into it.
    Raises ValueError when a value lies in no bucket or D17 is malformed.
    """
    pairs = _check_buckets(wicket_buckets)
    values = _ints(wickets, "wickets")
    codes = np.full(len(values), -1, dtype=np.int64)
    for position, (low, high) in enumerate(pairs):
        codes[(values >= low) & (values <= high)] = position
    outside = codes < 0
    if outside.any():
        shown = ", ".join(map(str, np.unique(values[outside])[:_SHOWN].tolist()))
        raise ValueError(
            f"{_rows(int(outside.sum()))} have wickets fallen outside every D17 Wickets_Bucket "
            f"{[list(p) for p in pairs]} (values {shown})"
        )
    return codes


def expected_runs_table(
    probs: Any, rows: OutcomeRows, models: OutcomeFoldModels, wicket_buckets: Any
) -> pd.DataFrame:
    """Mean expected against mean actual bat runs per Phase x Wickets_Bucket (criterion 6(c)).

    ``probs`` holds the ``actual`` probabilities of every row (shape (rows, N_CLASSES)), each row
    from the outer fit of its fold; ``models`` supplies the Class_Run_Values of each fold, and
    ``wicket_buckets`` is D17. Over the ``trainable`` rows of all folds, returns the
    EXPECTED_RUNS_COLUMNS, one row per Phase (by name: powerplay, middle, death) and
    Wickets_Bucket (``"<lowest>-<highest>"``) in that order: ``n_legal_balls``,
    ``mean_expected_bat_runs`` (the mean of ``expected_bat_runs`` with the Class_Run_Values of
    each row's fold) and ``mean_actual_bat_runs``; the means are NaN in a cell without a row.
    Raises the errors of expected_bat_runs and wickets_bucket_codes.
    """
    if not isinstance(rows, OutcomeRows):
        raise TypeError(f"rows must be the OutcomeRows of outcome_rows; got {type(rows).__name__}")
    if not isinstance(models, OutcomeFoldModels):
        raise TypeError(f"models must be an OutcomeFoldModels (fit_outcome); got {type(models).__name__}")
    pairs = _check_buckets(wicket_buckets)
    p = _prob_matrix(probs, rows.n_rows, "probs")
    evaluated = np.flatnonzero(rows.trainable)
    expected = expected_bat_runs(p[evaluated], models.class_run_values_for(rows.fold[evaluated]))
    actual = rows.bat_runs[evaluated].astype(np.float64)
    bucket = wickets_bucket_codes(rows.wickets[evaluated], pairs)
    phase = rows.phase[evaluated]
    records = []
    for member in Phase:
        for position, (low, high) in enumerate(pairs):
            cell = (phase == int(member)) & (bucket == position)
            n_rows = int(cell.sum())
            records.append(
                {
                    "phase": member.name.lower(),
                    "wickets_bucket": f"{low}-{high}",
                    "n_legal_balls": n_rows,
                    "mean_expected_bat_runs": float(expected[cell].mean()) if n_rows else np.nan,
                    "mean_actual_bat_runs": float(actual[cell].mean()) if n_rows else np.nan,
                }
            )
    table = pd.DataFrame.from_records(records, columns=list(EXPECTED_RUNS_COLUMNS))
    return table.astype({"phase": "str", "wickets_bucket": "str", "n_legal_balls": np.int64})


# ======================================================================== checkpoint artifacts

def fold_models_table(family: str) -> str:
    """The checkpoint table name of a family's fold models: ``m4_<family>_fold_models``."""
    return f"{MODEL}_{family}_fold_models"


def predictions_table(family: str) -> str:
    """The checkpoint table name of a family's variant probabilities, the design's ``m4_<candidate>``."""
    return f"{MODEL}_{family}"


def fold_models_frame(models: OutcomeFoldModels) -> pd.DataFrame:
    """The FOLD_MODEL_COLUMNS of ``models``, one row per fold (module docstring, "Checkpoint artifacts")."""
    if not isinstance(models, OutcomeFoldModels):
        raise TypeError(f"models must be an OutcomeFoldModels (fit_outcome); got {type(models).__name__}")
    records = []
    for k, fold in models.folds.items():
        record: dict[str, Any] = {
            "fold": k,
            "fit_id": fold.fit_id,
            "family": models.family,
            "features": _FEATURE_SEPARATOR.join(models.features),
            "m1_fit": fold.average_skill.m1_fit,
            "avg_theta_bat": float(fold.average_skill.theta_bat),
            "avg_theta_bowl": float(fold.average_skill.theta_bowl),
            "best_iteration": fold.best_iteration,
            "n_train_rows": fold.n_train_rows,
            "n_fit_rows": fold.n_fit_rows,
            "n_early_stopping_rows": fold.n_early_stopping_rows,
        }
        for cls, name in enumerate(CLASSES):
            record[f"crv_{name}"] = float(fold.class_run_values[cls])
        for cls, name in enumerate(CLASSES):
            record[f"n_{name}"] = int(fold.class_counts[cls])
        records.append(record)
    table = pd.DataFrame.from_records(records, columns=list(FOLD_MODEL_COLUMNS))
    integers = ("fold", "best_iteration", "n_train_rows", "n_fit_rows", "n_early_stopping_rows")
    types: dict[str, Any] = {name: np.int64 for name in (*integers, *(f"n_{name}" for name in CLASSES))}
    types.update({name: "str" for name in ("fit_id", "family", "features", "m1_fit")})
    return table.astype(types)


def save_fold_models(store: CheckpointStore, stage: str, models: OutcomeFoldModels) -> None:
    """Save every fold's model under its fit id and the ``fold_models_table(family)`` table into ``stage``.

    ``store.save_model`` keeps a LightGBM Booster as model text with its best round (and its
    category lists), an XGBoost Booster (cut to its best round, its category lists in
    XGB_CATEGORIES_ATTRIBUTE) as XGBoost JSON and a random-forest pipeline with joblib. Raises the
    errors of the store.
    """
    table = fold_models_frame(models)
    for fold in models.folds.values():
        store.save_model(stage, fold.fit_id, fold.model)
    store.save_table(stage, fold_models_table(models.family), table)


def load_fold_models(store: CheckpointStore, stage: str, family: OutcomeFamily) -> OutcomeFoldModels:
    """The OutcomeFoldModels of ``family`` that ``save_fold_models`` saved into ``stage``.

    Each model comes back through ``store.load_model`` and predicts with ``family.predict``.
    Raises ValueError when the table lacks a column, names another family or holds no fold, or
    its folds disagree on the features; StageError (exit code 5) for a checkpoint file that is
    missing, changed or unreadable.
    """
    if not isinstance(family, OutcomeFamily):
        raise TypeError(f"family must be an OutcomeFamily (lgbm_family); got {type(family).__name__}")
    name = fold_models_table(family.name)
    table = store.load_table(stage, name)
    _require(table, FOLD_MODEL_COLUMNS, f"checkpoint table {name!r}")
    if not len(table):
        raise ValueError(f"checkpoint table {name!r} holds no fold")
    families = set(table["family"].astype(str))
    if families != {family.name}:
        raise ValueError(f"checkpoint table {name!r} holds the families {sorted(families)}, not {family.name!r}")
    feature_lists = set(table["features"].astype(str))
    if len(feature_lists) != 1:
        raise ValueError(f"the folds of checkpoint table {name!r} disagree on the feature list")
    features = tuple(next(iter(feature_lists)).split(_FEATURE_SEPARATOR))
    loaded: dict[int, OutcomeFold] = {}
    for record in table.to_dict("records"):
        k = int(record["fold"])
        fit_id = str(record["fit_id"])
        loaded[k] = OutcomeFold(
            fold=k,
            fit_id=fit_id,
            model=store.load_model(stage, fit_id),
            average_skill=AverageSkill(
                theta_bat=float(record["avg_theta_bat"]),
                theta_bowl=float(record["avg_theta_bowl"]),
                m1_fit=str(record["m1_fit"]),
            ),
            class_run_values=[record[f"crv_{cls}"] for cls in CLASSES],
            class_counts=[record[f"n_{cls}"] for cls in CLASSES],
            best_iteration=int(record["best_iteration"]),
            n_fit_rows=int(record["n_fit_rows"]),
            n_early_stopping_rows=int(record["n_early_stopping_rows"]),
        )
    return OutcomeFoldModels(family=family.name, features=features, folds=loaded, predict=family.predict)


def class_run_values_frame(models: OutcomeFoldModels) -> pd.DataFrame:
    """The CLASS_RUN_VALUES_TABLE of ``models``: one row per fold (module docstring, "Checkpoint artifacts").

    Columns CLASS_RUN_VALUE_COLUMNS: ``fold`` (int64), the Class_Run_Value ``crv_<class>``
    (float64) and the Training_Rows ``n_<class>`` (int64) of every class, from the fit of the fold.
    """
    if not isinstance(models, OutcomeFoldModels):
        raise TypeError(f"models must be an OutcomeFoldModels (fit_outcome); got {type(models).__name__}")
    data: dict[str, Any] = {"fold": np.array(list(models.folds), dtype=np.int64)}
    folds_in_order = list(models.folds.values())
    for cls, name in enumerate(CLASSES):
        data[f"crv_{name}"] = np.array([fold.class_run_values[cls] for fold in folds_in_order], dtype=np.float64)
    for cls, name in enumerate(CLASSES):
        data[f"n_{name}"] = np.array([fold.class_counts[cls] for fold in folds_in_order], dtype=np.int64)
    return pd.DataFrame(data, columns=list(CLASS_RUN_VALUE_COLUMNS))


def load_class_run_values(store: CheckpointStore, stage: str) -> dict[int, np.ndarray]:
    """The Class_Run_Values saved as CLASS_RUN_VALUES_TABLE in ``stage``: fold -> N_CLASSES values in CLASSES order.

    Each value array is float64 and read-only. Raises ValueError when the table lacks a column,
    holds no fold or a fold twice, or holds a value that is missing, negative, not finite or, for
    a dot or run class, other than its bat runs; StageError (exit code 5) for a checkpoint file
    that is missing, changed or unreadable.
    """
    table = store.load_table(stage, CLASS_RUN_VALUES_TABLE)
    _require(table, CLASS_RUN_VALUE_COLUMNS, f"checkpoint table {CLASS_RUN_VALUES_TABLE!r}")
    ks = _ints(table["fold"], f"checkpoint table {CLASS_RUN_VALUES_TABLE!r} column 'fold'", minimum=1)
    if not len(ks) or len(set(ks.tolist())) != len(ks):
        raise ValueError(f"checkpoint table {CLASS_RUN_VALUES_TABLE!r} must hold each fold once and at least one")
    values = np.column_stack(
        [_floats(table[f"crv_{name}"], f"checkpoint table column crv_{name}") for name in CLASSES]
    )
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError(f"checkpoint table {CLASS_RUN_VALUES_TABLE!r} holds a missing, negative or non-finite value")
    for cls, runs in RUN_CLASS_RUNS.items():
        if (values[:, cls] != runs).any():
            raise ValueError(f"the Class_Run_Value of {CLASSES[cls]!r} must be exactly {runs} in every fold")
    return {int(k): _frozen(values[position], np.float64) for position, k in enumerate(ks.tolist())}


def variant_columns(variants: Sequence[str] = VARIANT_NAMES) -> tuple[str, ...]:
    """The probability columns ``<variant>_<class>`` of ``variants``, variant by variant, classes in CLASSES order."""
    return tuple(f"{variant}_{name}" for variant in _check_variants(variants) for name in CLASSES)


def variant_table(probs: Mapping[str, Any], row_ids: Any) -> pd.DataFrame:
    """The design's ``m4_<candidate>`` table of the six variants (module docstring, "Checkpoint artifacts").

    ``probs`` maps every name of VARIANT_NAMES to its array of shape (rows, N_CLASSES), as
    ``predict_variants`` returns them, and ``row_ids`` gives the ``row`` of each array row (the
    ``state`` rows, ``OutcomeRows.row``). Returns ``row`` (int32), then ``variant_columns()`` as
    float32. Raises ValueError when a variant is missing, an array has the wrong shape or is not
    finite, or a ``row`` does not fit in int32.
    """
    if not isinstance(probs, Mapping):
        raise TypeError("probs must map each variant name to its probability array")
    missing = [name for name in VARIANT_NAMES if name not in probs]
    if missing:
        raise ValueError(f"probs lacks the variant(s) {', '.join(missing)}")
    ids = _ints(row_ids, "row_ids", minimum=0)
    if len(ids) and int(ids.max()) > _INT32_MAX:
        raise ValueError("row_ids must fit in int32")
    data: dict[str, np.ndarray] = {_ROW_COLUMN: ids.astype(np.int32)}
    for variant in VARIANT_NAMES:
        values = _prob_matrix(probs[variant], len(ids), f"the {variant} probabilities")
        for cls, name in enumerate(CLASSES):
            data[f"{variant}_{name}"] = values[:, cls].astype(np.float32)
    return pd.DataFrame(data, columns=[_ROW_COLUMN, *variant_columns()])


def variant_arrays(table: pd.DataFrame, variants: Sequence[str] = VARIANT_NAMES) -> dict[str, np.ndarray]:
    """The variant arrays of a ``variant_table``: variant -> float64 array of shape (rows, N_CLASSES), in table order."""
    names = _check_variants(variants)
    _require(table, variant_columns(names), "table")
    return {
        variant: np.column_stack(
            [_floats(table[f"{variant}_{name}"], f"table column {variant}_{name}") for name in CLASSES]
        )
        for variant in names
    }


# ===================================================================== the ``outcome`` Pipeline_Stage

def _shown_path(path: Path, root: Path) -> str:
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


def _check_view_rows(k: int, view: pd.DataFrame, row_ids: np.ndarray) -> None:
    """ValueError unless view V_k holds the ``state`` rows in ``state`` order, so that the row folds line up."""
    _require(view, (_ROW_COLUMN,), f"view V_{k}")
    if len(view) != len(row_ids) or not np.array_equal(_ints(view[_ROW_COLUMN], f"view V_{k} column 'row'"), row_ids):
        raise ValueError(
            f"view V_{k} does not hold the Deliveries of the state table in its order; pass the tables of one Run"
        )


def _check_saved_variants(table: pd.DataFrame) -> dict[str, float]:
    """The largest |row sum - 1| of each variant of a variant_table, summed in float64 from its float32 columns.

    Raises ValueError when a saved probability lies outside [0, 1] or a row's sum differs from 1
    by more than 1e-6 (criterion 5), which float32 storage of rescaled rows never causes.
    """
    deviations: dict[str, float] = {}
    for variant in VARIANT_NAMES:
        columns = np.column_stack([table[f"{variant}_{name}"].to_numpy(dtype=np.float64) for name in CLASSES])
        if len(columns) and (columns.min() < 0.0 or columns.max() > 1.0):
            raise ValueError(f"the saved {variant} probabilities hold a value outside [0, 1]")
        deviations[variant] = float(np.abs(columns.sum(axis=1) - 1.0).max()) if len(columns) else 0.0
    off = {name: value for name, value in deviations.items() if value > _SUM_TOLERANCE}
    if off:
        raise ValueError(
            f"saved M4 rows differ from a sum of 1 by more than 1e-6 (Requirement 11, criterion 5): largest "
            f"differences {off}"
        )
    return deviations


def _span(values: Sequence[int]) -> str:
    low, high = min(values), max(values)
    return f"{low:,}" if low == high else f"{low:,}-{high:,}"


def run_stage(ctx: StageContext) -> None:
    """The ``outcome`` Pipeline_Stage (module docstring, "The outcome Pipeline_Stage").

    Raises FitLeakageError or ViewLeakageError (exit code 4) when a fit or a view breaks the
    leakage guard (Requirement 7, criterion 12), DataError (exit code 3) for a match that is not in
    the Fold_Table, and ValueError or TypeError for input tables that do not fit together, a
    malformed Config value, or probabilities that break Requirement 11, criteria 4 and 5.
    """
    # Imported here, not with the module: remaining_runs imports superr.value, and the value stage
    # (task 17.2) may make superr.value import this module, which would close an import cycle.
    from superr.models.remaining_runs import STAGE_NAME as M2_STAGE

    started = time.perf_counter()
    settings, store, registry, log = ctx.settings, ctx.store, ctx.registry, ctx.log
    plan = folds.sandbox_plan(settings.get(_N_FOLDS))  # checks D15 before any table is read
    family = lgbm_family(settings)
    setup = extras.extras_setup(settings)
    wicket_buckets = settings.get(_WICKET_BUCKETS)
    _check_buckets(wicket_buckets)
    ks = [spec.sandbox_k for spec in _outer_specs(plan)]

    balls = store.load_table(reconstruct.STAGE_NAME, reconstruct.BALLS_TABLE)
    state = store.load_table(STATE_STAGE, STATE_TABLE)
    fold_table = store.load_table(folds.STAGE_NAME, folds.FOLD_TABLE)
    m1_theta = store.load_table(SKILL_STAGE, M1_THETA_TABLE)

    # 1. The rows of both models: every innings 1-2 Delivery, in state order.
    rows = outcome_rows(balls, state, fold_table)
    extras_rows = extras.extras_rows(balls, state, fold_table)
    del balls

    # 2. Per fold k: V_k checked before any fit reads it, then its feature frames and Average_Skill.
    m4_X: dict[int, pd.DataFrame] = {}
    extras_X: dict[int, pd.DataFrame] = {}
    skills: dict[int, AverageSkill] = {}
    for k in ks:
        view = store.load_table(M2_STAGE, views.view_table(k))
        _check_view_rows(k, view, rows.row)
        views.check_view(k, view, rows.fold, registry)
        m4_X[k] = m4_features(view, state, rows)
        extras_X[k] = extras.extras_features(view, state, extras_rows)
        skills[k] = average_skill(view, k, rows.fold, m1_theta)
        del view
    del state, m1_theta

    # 3. M4: the outer fits of the LightGBM family, then the six variants of every row.
    began = time.perf_counter()
    m4_models = fit_outcome(registry, plan, family, rows, m4_X, skills)
    m4_fit_seconds = time.perf_counter() - began
    began = time.perf_counter()
    probs = predict_variants(m4_models, m4_X, rows.fold, log)
    m4_predict_seconds = time.perf_counter() - began
    del m4_X

    # 4. The Extras_Model: its outer fits, then the class probabilities of every row.
    began = time.perf_counter()
    extras_models = extras.fit_extras(registry, plan, setup, extras_rows, extras_X)
    p_extras = extras.predict_extras(extras_models, extras_X, extras_rows.fold, log)
    extras_seconds = time.perf_counter() - began
    del extras_X

    # 5. Checkpoint models and tables, each table checked before it is saved.
    m4_table = variant_table(probs, rows.row)
    deviations = _check_saved_variants(m4_table)
    preds = extras.preds_frame(p_extras, extras_rows, extras_models)
    save_fold_models(store, STAGE_NAME, m4_models)
    store.save_table(STAGE_NAME, predictions_table(family.name), m4_table)
    store.save_table(STAGE_NAME, CLASS_RUN_VALUES_TABLE, class_run_values_frame(m4_models))
    extras.save_fold_models(store, STAGE_NAME, extras_models)
    store.save_table(STAGE_NAME, extras.PREDS_TABLE, preds)
    del m4_table

    # 6. The held-out metrics of both models and the output files.
    m4_loss = log_loss_table(probs[ACTUAL], rows, candidate=predictions_table(family.name))
    m4_runs = expected_runs_table(probs[ACTUAL], rows, m4_models, wicket_buckets)
    extras_loss = extras.log_loss_table(p_extras, extras_rows)
    extras_rates = extras.rates_table(p_extras, extras_rows)
    outputs = settings.path("outputs")
    m4_dir, extras_dir = outputs / M4_FOLDER, outputs / extras.EXTRAS_FOLDER
    files: dict[Path, pd.DataFrame] = {
        m4_dir / LOG_LOSS_FILE: m4_loss,
        m4_dir / EXPECTED_RUNS_FILE: m4_runs,
        m4_dir / FOLD_MODELS_FILE: fold_models_frame(m4_models).drop(columns=["features"]),
        extras_dir / extras.LOG_LOSS_FILE: extras_loss,
        extras_dir / extras.RATES_FILE: extras_rates,
        extras_dir / extras.FOLD_MODELS_FILE: extras.fold_models_frame(extras_models).drop(columns=["features"]),
    }
    for path, table in files.items():
        _write_csv(path, table)

    # 7. The fit registry and the data-derived Method_Constants.
    n_saved = registry.save(store, STAGE_NAME)
    ctx.constants.record(
        AVERAGE_SKILL_CONSTANT,
        {
            str(k): {
                "m1_fit": fold.average_skill.m1_fit,
                "theta_bat": float(fold.average_skill.theta_bat),
                "theta_bowl": float(fold.average_skill.theta_bowl),
            }
            for k, fold in m4_models.folds.items()
        },
        _COMPONENT,
    )
    ctx.constants.record(
        BEST_ITERATION_CONSTANT, {fold.fit_id: fold.best_iteration for fold in m4_models.folds.values()}, _COMPONENT
    )

    _log_stage(
        ctx,
        rows=rows,
        m4_models=m4_models,
        extras_models=extras_models,
        m4_loss=m4_loss,
        m4_runs=m4_runs,
        extras_loss=extras_loss,
        extras_rates=extras_rates,
        deviations=deviations,
        preds=preds,
        files=list(files),
        n_saved=n_saved,
        seconds={
            "m4_fit": m4_fit_seconds,
            "m4_predict": m4_predict_seconds,
            "extras": extras_seconds,
            "total": time.perf_counter() - started,
        },
    )


def _log_stage(
    ctx: StageContext,
    *,
    rows: OutcomeRows,
    m4_models: OutcomeFoldModels,
    extras_models: extras.ExtrasFoldModels,
    m4_loss: pd.DataFrame,
    m4_runs: pd.DataFrame,
    extras_loss: pd.DataFrame,
    extras_rates: pd.DataFrame,
    deviations: Mapping[str, float],
    preds: pd.DataFrame,
    files: Sequence[Path],
    n_saved: int,
    seconds: Mapping[str, float],
) -> None:
    log = ctx.log
    m4_folds = list(m4_models.folds.values())
    means = "; ".join(
        f"{CLASSES[cls]} " + "/".join(f"{fold.class_run_values[cls]:.3f}" for fold in m4_folds)
        for cls in MEAN_VALUE_CLASSES
    )
    log.info(
        f"M4 {m4_models.family}: {len(m4_folds)} outer multiclass fits ({N_CLASSES} classes; Requirement 11, criteria "
        f"1-3) in {seconds['m4_fit']:.1f} s on {_span([fold.n_train_rows for fold in m4_folds])} Training_Rows each "
        f"(trainable Legal_Balls of the other folds, {_span([fold.n_early_stopping_rows for fold in m4_folds])} of "
        f"them in the early-stopping matches); boosting rounds kept "
        f"{'/'.join(str(fold.best_iteration) for fold in m4_folds)} (folds {'/'.join(map(str, m4_models.folds))}); "
        f"Average_Skill theta_bat {'/'.join(f'{fold.average_skill.theta_bat:.4f}' for fold in m4_folds)} and "
        f"theta_bowl {'/'.join(f'{fold.average_skill.theta_bowl:.4f}' for fold in m4_folds)} (typical player of "
        f"M1(-k)); Class_Run_Values per fold: {means}",
    )
    loss = dict(zip(m4_loss["model"].tolist(), m4_loss["log_loss"].tolist()))
    candidate = predictions_table(m4_models.family)
    gap = (m4_runs["mean_expected_bat_runs"] - m4_runs["mean_actual_bat_runs"]).to_numpy(dtype=np.float64)
    filled = np.flatnonzero(np.isfinite(gap))
    worst = filled[np.argmax(np.abs(gap[filled]))] if len(filled) else None
    worst_text = (
        "n/a"
        if worst is None
        else f"{m4_runs['phase'].iloc[worst]} x {m4_runs['wickets_bucket'].iloc[worst]} wickets, {gap[worst]:+.4f} "
        f"on {int(m4_runs['n_legal_balls'].iloc[worst]):,} Legal_Balls"
    )
    log.info(
        f"M4 held-out metrics on {int(m4_loss['n_legal_balls'].iloc[0]):,} trainable Legal_Balls (criterion 6): pooled "
        f"multiclass log loss {loss[candidate]:.5f} against {loss[PHASE_BASELINE]:.5f} for the Phase-only baseline "
        f"({loss[candidate] - loss[PHASE_BASELINE]:+.5f}); mean expected minus actual bat runs per Phase x "
        f"Wickets_Bucket cell from {np.nanmin(gap):+.4f} to {np.nanmax(gap):+.4f} (largest: {worst_text})",
    )
    if not loss[candidate] < loss[PHASE_BASELINE]:
        log.warning(
            f"the M4 {m4_models.family} pooled held-out log loss ({loss[candidate]:.5f}) is not below the Phase-only "
            f"baseline's ({loss[PHASE_BASELINE]:.5f})",
        )
    log.info(
        f"M4 variants {', '.join(VARIANT_NAMES)} predicted for {rows.n_rows:,} innings 1-2 Deliveries "
        f"({int((~rows.legal).sum()):,} of them Wides and No_Balls) in {seconds['m4_predict']:.1f} s; every saved "
        f"float32 row sums to 1 within {max(deviations.values()):.1e} (Requirement 11, criterion 5)",
    )
    extras_folds = list(extras_models.folds.values())
    extras_loss_of = dict(zip(extras_loss["model"].tolist(), extras_loss["log_loss"].tolist()))
    overall = extras_rates.iloc[np.flatnonzero(extras_rates["innings"].to_numpy(dtype=object) == extras.ALL)[0]]
    p_wide, p_noball = preds["p_wide"].to_numpy(), preds["p_noball"].to_numpy()
    log.info(
        f"Extras_Model: {len(extras_folds)} outer multinomial logistic fits ({'/'.join(extras.CLASSES)}; no bowler "
        f"feature; Requirement 11, criteria 4 and 16) in {seconds['extras']:.1f} s on "
        f"{_span([fold.n_train_rows for fold in extras_folds])} Training_Rows each (every trainable Delivery of the "
        f"other folds); lbfgs iterations {'/'.join(str(fold.n_iterations) for fold in extras_folds)}; pooled "
        f"held-out log loss {extras_loss_of[extras.MODEL]:.5f} against {extras_loss_of[extras.CELL_BASELINE]:.5f} "
        f"for the innings x Phase cell baseline on {int(extras_loss['n_deliveries'].iloc[0]):,} trainable Deliveries; "
        f"Wide rate {overall['wide_rate']:.4f} actual and {overall['mean_p_wide']:.4f} predicted, No_Ball rate "
        f"{overall['no_ball_rate']:.4f} actual and {overall['mean_p_noball']:.4f} predicted; over all "
        f"{len(preds):,} Deliveries p_wide {p_wide.min():.4f}-{p_wide.max():.4f}, p_noball "
        f"{p_noball.min():.4f}-{p_noball.max():.4f}, largest p_wide + p_noball {float((p_wide + p_noball).max()):.4f}",
    )
    stalled = [fold.fit_id for fold in extras_folds if not fold.converged]
    if stalled:
        log.warning(
            f"Extras_Model fit(s) {', '.join(stalled)} stopped at models.extras.max_iter before lbfgs converged; their "
            "predictions are kept",
        )
    if not extras_loss_of[extras.MODEL] < extras_loss_of[extras.CELL_BASELINE]:
        log.warning(
            f"the Extras_Model pooled held-out log loss ({extras_loss_of[extras.MODEL]:.5f}) is not below the cell "
            f"baseline's ({extras_loss_of[extras.CELL_BASELINE]:.5f})",
        )
    log.info(
        f"wrote {', '.join(_shown_path(path, ctx.root) for path in files)}; saved {candidate}, "
        f"{fold_models_table(m4_models.family)}, {CLASS_RUN_VALUES_TABLE}, {extras.PREDS_TABLE}, "
        f"{extras.FOLD_MODELS_TABLE}, {len(m4_folds) + len(extras_folds)} models under their fit ids and the fit "
        f"registry ({n_saved} fits) in {seconds['total']:.1f} s",
    )
