"""Model_Choice_Checker: the comparison of the M3 and M4 Candidate_Models, the selection of the
Primary_Models, the precomputed predictions of every Candidate_Model and the ``model_choice``
Pipeline_Stage (Requirement 23, criteria 3, 5-9, 12 and 13; Requirement 10, criteria 11, 18 and
19; design decision DD-8 and section "Model_Choice_Checker (``models/model_choice.py``)").

Task 13.2 compares the Candidate_Models that earlier stages fitted: the calibrated logistic
baseline and the calibrated LightGBM classifier of each M3 innings (stage ``win_prob``) and the
LightGBM M4 (stage ``outcome``). Task 26.1 adds the M3 Alternative_Models: this stage fits an
XGBoost and a random-forest classifier for each innings itself (``fit_alternative``), through the
same fold sandbox and centred isotonic calibrators as the ``win_prob`` families, and they enter as
two more M3Candidates with the family names XGB and RF. Task 26.2 adds the M4 Alternative_Models:
this stage fits an XGBoost and a random-forest multiclass M4 itself (``fit_m4_alternative``), with
the ``outcome`` functions that fitted the LightGBM M4, and they enter as two more M4Candidates.
Task 26.3 adds the reruns of the ranking with each Candidate_Model that was not selected, which
read the arrays saved here through ``load_predictions``. No function reads ``config.py``:
``run_stage`` reads D15, D17 and the model settings from ``ctx.settings``.

Imports
    ``win_prob``, ``remaining_runs`` and ``pressure`` are imported inside the functions that use
    them. ``remaining_runs`` imports ``superr.value``, so a module-level import here would stop
    ``superr.value`` (task 17) from importing ``load_primary`` at module level.

Candidates (M3Candidate, M4Candidate)
    An M3Candidate is one M3 family (a name of M3_TIE_ORDER) with, for each innings whose fit
    succeeded, the calibrated outer fold models (``win_prob.CalibratedFolds``) and the family's
    pre-ball WP table (``win_prob.WP_TABLE_COLUMNS``: ``row``, ``fold``, the calibrated
    Out_Of_Fold WP before the terminal rules ``wp_raw`` and after them ``wp_pre``, and
    ``terminal_rule``, one row per Delivery of the innings in ``state`` order). ``fit_failure``
    gives the reason for an innings whose fit failed. An M4Candidate is one M4 family (a name of
    M4_TIE_ORDER) with its six variant arrays (``outcome.VARIANT_NAMES`` -> float64 (rows, 9) in
    ``state`` order and CLASSES order), its ``outcome.OutcomeFoldModels`` (their Class_Run_Values
    serve the expected-runs calibration), the checkpoint that holds its ``m4_<family>`` table
    (``stage``) and ``fit_failure``. The stage builds the logistic and LightGBM M3Candidates from
    the ``win_prob`` checkpoint, the XGBoost and random-forest M3Candidates from its own fits
    (``alternative_candidate``), the LightGBM M4Candidate from the ``outcome`` checkpoint, whose
    ``m4_lgbm`` arrays it reuses rather than predicting them again, and the XGBoost and
    random-forest M4Candidates from its own fits (``fit_m4_alternative``, ``stage`` STAGE_NAME).

M4 Alternative_Models (``fit_m4_alternative``, M4AlternativeFit; criteria 1, 2, 4, 6, 8 and 12)
    ``fit_m4_alternative(registry, plan, family, rows, features, average_skills, log)`` fits one
    M4 Alternative_Model with ``outcome.fit_outcome``: ``family`` is the OutcomeFamily XGB or RF
    of ``outcome.candidate_family``, one setting each from ``models.xgb.m4`` and ``models.rf.m4``
    (criterion 1), with the overfitting limits of criterion 4 (XGBoost: ``min_child_weight``,
    ``max_depth`` and early stopping on whole matches drawn from the training folds only,
    ``es_share`` and ``es_patience``; random forest: ``min_samples_leaf`` and ``max_depth``) and no
    monotone constraints (none are required for M4). The D15 outer fits use what the LightGBM M4
    used (criterion 2): the ``trainable`` Legal_Balls of the other folds as Training_Rows, the 9
    classes, the M4_FEATURES of each Delivery built from view V_k by ``outcome.m4_features`` (the
    views checked by ``views.check_view`` first) and the Match_Grouped_Folds, so each candidate's
    Class_Run_Values are LightGBM's. Each fold's Average_Skill is the one the ``outcome`` stage
    computed for fold k from M1(-k) and saved with the LightGBM fold models (this stage does not
    read the ``skill`` checkpoint), checked against the ``m1_fit`` of the fold-k rows of V_k. Each
    fit is registered with the FitRegistry before it runs, under model ``m4`` and family ``xgb``
    or ``rf`` (fit ids ``m4_xgb_outer_k1`` ... ``m4_rf_outer_k5``), with the matches of fold k as
    its predicted matches; there are no inner fits, because M4's output is never a model feature.

    The fit of fold k then predicts the six variants (``outcome.VARIANTS``) of every innings 1-2
    Delivery of fold k, Wides and No_Balls included (``outcome.variant_probabilities``). The raw
    arrays are checked with ``m4_failure`` before ``common.renormalise`` rescales each row, then
    rounded to the float32 of their ``m4_<family>`` table, so every M4 candidate is scored on the
    values that later stages read. The M4AlternativeFit holds the M4Candidate, the largest
    |row sum - 1| of each raw variant and the fit and prediction times. A fit that raises is
    recorded as FIT_FAILED, a prediction that raises as PREDICTION_FAILED and raw arrays that
    break criterion 12 with the ``m4_failure`` reason (NON_FINITE, M4_OUTSIDE or M4_SUM), in the
    candidate's ``fit_failure``; ``_m4_results`` keeps such a reason as it is. A LeakageError,
    DataError or other PipelineError, and a MemoryError, still stop the Run. The fits of a failed
    candidate stay in the FitRegistry: each passed the guard, and none of their output is used.

    ``load_m4_models(store, settings, family)`` rebuilds the fold models of any M4 candidate that
    passed, from the checkpoint that SOURCES_TABLE names, for re-predicting other feature values
    (Requirement 21, criterion 1) whichever family is the Primary_Model.

M3 Alternative_Models (``fit_alternative``, ``alternative_candidate``; criteria 1-4 and 12)
    ``fit_alternative(innings, family, view_by_k, labels, settings, registry, state=state)`` fits
    one Alternative_Model of one innings with ``win_prob.fit_innings``. ``family`` is the
    CandidateFamily XGB or RF of ``win_prob.candidate_family``: one setting each, read from
    ``models.xgb.m3`` and ``models.rf.m3`` and shared by both innings (criterion 1), with the
    overfitting limits of criterion 4 (XGBoost: ``min_child_weight``, ``max_depth`` and early
    stopping on whole matches drawn from the training folds only, ``es_share`` and
    ``es_patience``; random forest: ``min_samples_leaf`` and ``max_depth``). fit_innings runs the
    D15 x D15 fits of ``folds.sandbox_plan`` and the D15 centred isotonic calibrators, bounded to
    ``models.m3.calibration_bounds``, exactly as for the ``win_prob`` families: the same
    Training_Rows (the ``eligible`` rows of ``m3_labels``), labels, M3_FEATURES read from the
    views V_k, monotone constraints (M3_MONOTONE; criterion 2) and Match_Grouped_Folds, so every
    metric and selection below rests on calibrated Out_Of_Fold_Predictions (criterion 3). Every fit
    is registered with the FitRegistry before it runs, under model ``m3_inn<i>`` (classifiers) or
    ``m3_inn<i>_cal`` (calibrators) and family ``xgb`` or ``rf`` (fit ids such as
    ``m3_inn1_xgb_outer_k1`` and ``m3_inn2_cal_rf_outer_k3``), so the leakage guard checks it.

    The result, an AlternativeFit, holds the FoldModels and the pre-ball WP table
    (``win_prob.pre_ball_table``: ``row``, ``fold``, ``wp_pre``, ``wp_raw``, ``terminal_rule``,
    ``fit_id``, ``calibrator_fit_id``), or a failure reason (criterion 12): NON_FINITE or
    M3_OUTSIDE when a fitted model's raw output for some Deliveries is missing, infinite or outside
    [0, 1] (``win_prob.InvalidPrediction``), and FIT_FAILED with the error otherwise. A
    LeakageError, DataError or other PipelineError, and a MemoryError, still stop the Run. The
    fits of a failed innings that were registered before the failure stay in the FitRegistry: each
    passed the guard, and none of their output is used. ``alternative_candidate(family, fits)``
    turns the AlternativeFits of both innings into the family's M3Candidate, with ``stage``
    STAGE_NAME and a ``fit_failure`` for an innings that failed. Each innings is a Candidate_Model
    of its own (M3-1 or M3-2), so a failure in one innings leaves the other in the comparison.

Precomputed predictions (``precompute_predictions``; DD-8)
    ``pressure.post_outcome_states`` builds the 9 Post_Outcome_States of every innings 1-2
    Delivery once. ``precompute_predictions(innings, candidates, post)`` evaluates every M3
    candidate of the innings on them with ``win_prob.win_probability``: a state that is not
    terminal goes through the calibrated outer model of its fold, a terminal state (innings 2
    only) through the terminal rules of Requirement 10, criteria 14-16. Each candidate gets, for
    every Delivery of the innings, ``wp_pre`` and ``wp_post_0`` ... ``wp_post_8`` (one per class,
    CLASSES order), all after the terminal rules, so the Leverage_Index sum of P(class) x
    |WP(Post_Outcome_State) - WP(pre-ball state)| (Requirement 11, criterion 9) is a row-wise sum
    of stored arrays. ``wp_pre`` is the ``win_prob`` table's ``wp_pre``, the stage's Out_Of_Fold
    WP (criterion 19).

    An outcome that ends the first innings is valued at the end-of-innings state of Requirement
    10, criterion 17 (Scheduled_Balls bowled, the pre-ball wickets, Par_Score at Scheduled_Balls),
    which the innings-1 candidate scores like any other innings-1 state. So every WP of an
    innings comes from that innings' candidate alone: the M3-1 and M3-2 candidates are evaluated
    and selected independently, and a rerun that swaps either one (task 26.3) reads only that
    candidate's stored arrays.

Failure checks (``m3_failure``, ``m4_failure``; criterion 12)
    A candidate fails when its fit failed (``fit_failure``: a reason that starts with FIT_FAILED,
    NON_FINITE or M3_OUTSIDE, as fit_alternative records it, is kept as it is, and any other text
    is recorded after FIT_FAILED), when predicting its Post_Outcome_States raises
    ``win_prob.InvalidPrediction`` (NON_FINITE or M3_OUTSIDE, with the counts; the XGB and RF
    candidates raise it) or another ValueError (PREDICTION_FAILED: the ``win_prob`` prediction
    path raises it for a model output that is not a finite probability in [0, 1]), or when its arrays
    break a rule: for M3, a value of ``wp_raw``, ``wp_pre`` or ``wp_post`` that is not finite
    (NON_FINITE) or lies outside [0, 1] (M3_OUTSIDE); for M4, a value of any variant that is not
    finite (NON_FINITE), lies outside [0,
    1] (M4_OUTSIDE, which Requirement 11, criterion 5 also forbids) or a row whose 9 probabilities
    differ from a sum of 1 by more than 1e-6 (M4_SUM, ``common.PROB_SUM_TOLERANCE``). The M4
    checks cover every row of every variant: every Legal_Ball, and the Wides and No_Balls whose
    predictions NBV also reads. An M4 Alternative_Model's arrays are checked as the model gives
    them, before ``common.renormalise`` rescales each row (``fit_m4_alternative``), and its
    rescaled float32 arrays are checked again in the comparison; the LightGBM arrays are the
    saved ones, already rescaled, whose raw softmax rows sum to 1 within about 1e-15 (the
    ``outcome`` stage warns of any raw row more than 1e-6 away). A failed candidate keeps its row
    in the comparison table with its ``failure_reason`` and is never selected; its predictions and
    fold models are not saved.

Metrics and the comparison table (``candidate_table``, CandidateResult; criteria 5, 6 and 13)
    * M3, per innings and candidate: the Pooled_Held_Out_Log_Loss, the Brier score, the AUC and
      the Calibration_Error with the Calibration_Curve (``win_prob.wp_metrics`` and
      ``win_prob.calibration_curve``; 10 equal-width bins) of the calibrated Out_Of_Fold WP before
      the terminal rules (``wp_raw``), over the ``eligible`` Deliveries of the innings (the
      ``win_prob`` stage's ``m3_labels``: kept result types, labelled, Training_Rows). Every
      candidate of an innings is scored on these same Deliveries (criterion 5), with the rule of
      the ``win_prob`` metrics, so the numbers equal those of ``outputs/models/m3/metrics.csv``.
      The curves are drawn overall and per Phase.
    * M4, per candidate: the pooled multiclass log loss of the ``actual`` variant over the
      ``trainable`` Legal_Balls of all folds (``outcome.log_loss_table``) and, per Phase x
      Wickets_Bucket (D17) cell, the mean expected bat runs against the mean actual bat runs
      (``outcome.expected_runs_table``), on the same Legal_Balls for every candidate (criterion
      6). Its ``calibration_error`` is the Legal_Ball-weighted mean over the cells of |mean
      expected - mean actual bat runs|, in runs per Legal_Ball; ``brier`` and ``auc`` are empty.

    ``candidate_table(results)`` lays the CandidateResults out as COMPARISON_COLUMNS, one row per
    Candidate_Model in the order M3-1, M3-2, M4 and the tie order within each, with ``primary``
    True on the row that ``select_primary`` picks for each model. ``n_rows`` counts the
    Deliveries (M3) or Legal_Balls (M4) behind the metrics; every candidate of a model that did
    not fail must have the same count. ``failure_reason`` is empty for a candidate that passed.
    A candidate's metrics are filled whenever its scored predictions are valid, even if it failed
    another check. ``top10_changes_batter`` and ``top10_changes_bowler`` stay empty (nullable
    Int64) until the reruns of task 26.3 fill them.

Selection (``select_primary``, ``calibration_preferences``; criteria 7 and 8; DD-17)
    ``select_primary(table, model, preferred=...)`` returns the candidate of ``model`` with the
    lowest ``log_loss`` among the candidates without a failure reason, ``best``. A calibration-aware
    tie-break then applies (criterion 8): when ``preferred`` maps ``model`` to a ``(family, band)``
    that ``calibration_preferences`` resolved from the Config (``model_choice.calibration_preferred``
    and ``model_choice.calibration_band``), the named ``family`` has no failure reason and its log
    loss lies within ``band`` x ``best`` of the lowest (a relative log-loss band), that family is
    the Primary_Model. Otherwise ``best`` is, with exact ties broken by ``math.isclose(a, b,
    rel_tol=1e-9, abs_tol=1e-9)`` (TIE_REL_TOL, TIE_ABS_TOL), the convention of
    ``reconstruct.select_encoding``, and the first tied candidate in the model's tie order winning:
    logistic, LightGBM, XGBoost, random forest for M3 (M3_TIE_ORDER) and LightGBM, XGBoost, random
    forest for M4 (M4_TIE_ORDER). A model with no ``preferred`` entry keeps the plain
    lowest-log-loss rule, so M3 (no preferred family) is unchanged. Pooled log losses lie near
    0.4-1.5 here, so the exact-tie tolerance is about 1e-9 absolute: far above the float error of a
    mean over 10^5 rows (about 1e-13) and far below any difference that a model change produces.
    Comparing every candidate with the one best value keeps the result independent of the row order.

    The delivered ``preferred`` names LightGBM for M4 with ``band`` 0.005 (0.5% of the lowest). M4
    LightGBM (log loss 1.483181) lies 0.000826 (0.0826%) above the lowest, XGBoost (1.481957), so
    LightGBM is the M4 Primary_Model: it is the family whose own-theta NBV per innings x Phase cell
    passes the Requirement 15, criterion 17 check that feeds the ratings, while XGBoost breaches it
    in the innings-1 death cell. ``match_clustered_paired_se`` gives the supporting evidence (the
    match-clustered paired SE of the per-Legal_Ball log-loss difference XGBoost - LightGBM), the
    Run_Log records it, and the ``value`` stage re-verifies the tie-break on its own inputs
    (Requirement 23, criterion 14; ``value.own_theta_cell_means``). M3 names no preferred family,
    so both M3 innings stay on the logistic baseline.

Reading the Primary_Models' predictions (``load_primary``, ``load_predictions``,
PrimaryPredictions; criterion 9)
    ``load_primary(store)`` is the one entry point for later stages (``pressure``, ``value``,
    ``validation``, the reruns): it reads the ``primary`` flags of COMPARISON_TABLE and returns a
    PrimaryPredictions with the arrays of the Primary_Models only, every row of the ``state``
    table in ``state`` order: ``row``, ``match_id``, ``innings``, ``fold``; ``wp_pre`` (n,) and
    ``wp_post`` (n, 9) of the M3-1 Primary_Model on innings-1 rows and of the M3-2 Primary_Model on
    innings-2 rows; ``m4``, the M4 Primary_Model's variant arrays (``outcome.VARIANT_NAMES`` or the
    ``variants`` asked for, each float64 (n, 9)); and ``candidates``, the selected name of each
    model. ``load_predictions(store, m3={1: ..., 2: ...}, m4=...)`` does the same for any
    combination of non-failed candidates, which is what a rerun needs: each innings reads only
    its own candidate's WP table. Both read only the
    ``model_choice`` checkpoint and the checkpoint that SOURCES_TABLE names for the M4 arrays
    (``outcome`` for LightGBM, ``model_choice`` for XGBoost and the random forest), so a stage
    that calls them lists both in its ``reads``. The fold models of an M3 Alternative_Model, for
    predicting other states, come from ``win_prob.load_wp_models(store, family, D15,
    stage=STAGE_NAME)`` (outer fits and calibrators; the ``win_prob`` families' from the
    ``win_prob`` checkpoint), and those of any M4 candidate from ``load_m4_models``.

The ``model_choice`` Pipeline_Stage (``run_stage``)
    Reads ``matches`` (``load``), ``balls`` (``reconstruct``), ``state`` (``state``), the
    Fold_Table (``folds``), the views V_1 ... V_D15, the outer first-innings M2 fits and the
    ``m2_par_means`` and ``m2_par_bias`` tables (``remaining_runs``), the M3 fold models, WP tables
    and ``m3_labels`` (``win_prob``), and the M4 LightGBM fold models, ``m4_lgbm`` table and
    Class_Run_Values (``outcome``). ``ctx.registry`` must hold the M1 and M2 fits that the views
    name (the runner registers or reloads them). Then:

    1. builds the OutcomeRows of every innings 1-2 Delivery and checks that the ``m3_labels`` and
       ``m4_lgbm`` tables hold the same rows; checks every view with ``views.check_view``;
    2. builds the Post_Outcome_States once (``pressure.post_outcome_states``);
    3. fits the M3 Alternative_Models: fit_alternative for innings 1 and 2 and for XGB and RF
       (M3_ALTERNATIVES), 2 x 2 x D15 x D15 classifier fits and 2 x 2 x D15 calibrators, all
       registered with the FitRegistry; then the M4 Alternative_Models: fit_m4_alternative for XGB
       and RF (M4_ALTERNATIVES), 2 x D15 outer fits, each with the six variants of its fold's
       Deliveries; a failure is recorded and the stage goes on;
    4. evaluates every M3 candidate (logistic, LightGBM, XGBoost, random forest) on innings 1 and
       on innings 2, each innings with its own candidates only, scores them on the same eligible
       Deliveries and selects the M3-1 and M3-2 Primary_Models; scores and checks every M4
       candidate and selects the M4 Primary_Model; checks that every candidate gives each
       Post_Outcome_State the same terminal rule, that the deciding rules agree with the states'
       ``terminal`` flags and that INNINGS_END flags exactly the innings-1 states with
       ``innings_over``;
    5. saves COMPARISON_TABLE, ROWS_TABLE, the WP table ``wp_inn<i>_<family>`` (WP_TABLE_COLUMNS:
       WP_COLUMNS in float64, then the pre-ball WP_SOURCE_COLUMNS of the candidate's pre-ball WP
       table, so each row names the outer fit and the calibrator behind it) of every candidate and
       innings that passed, SOURCES_TABLE, M3_CURVES_TABLE, M4_CALIBRATION_TABLE, POST_RULES_TABLE
       (the innings_end row counts the innings-1 states that end the innings), the variant table
       ``m4_<family>``, the fold-model table ``m4_<family>_fold_models`` and the outer fold models
       (``outcome.save_fold_models``, under their fit ids) of each M4 Alternative_Model that
       passed, the outer models and
       calibrators of every M3 Alternative_Model and innings that passed
       (``win_prob.save_fold_models`` under their fit ids; the inner fits are registered but not
       saved, because only the calibrators used their predictions), ALTERNATIVE_FITS_TABLE
       (``win_prob.FIT_TABLE_COLUMNS`` of every fit of an innings whose fits completed),
       ALTERNATIVE_FOLD_METRICS_TABLE (``win_prob.FOLD_METRICS_COLUMNS`` per outer fit: the
       in-fold and held-out log loss of Requirement 10, criterion 8, applied to the
       Alternative_Models) and the fit registry; records the selection as the data-derived
       Method_Constant PRIMARY_CONSTANT;
    6. writes ``outputs/models/model_choice/`` (MODEL_CHOICE_FOLDER): COMPARISON_FILE, per innings
       the Calibration_Curves of every M3 candidate overall and per Phase as CSV and PNG
       (``calibration_file``, ``calibration_chart_file``), M4_CALIBRATION_FILE and
       ALTERNATIVE_FOLD_METRICS_FILE; and the WP curve of match REPORT_MATCH_ID (40767) from the M3
       Primary_Models to ``outputs/models/m3/`` (WP_CURVE_FILE, WP_CURVE_CHART_FILE; Requirement
       10, criterion 18): the WP of the side that batted first before every Delivery of innings 1
       and 2 in ball order, WP in innings 1 and 1 - WP in innings 2.

    The stage stores WP as float64 rather than the design's float32: the Live_Ball flag (D1) reads
    the unrounded Out_Of_Fold WP, and float32 rounding could move a WP across 0.05 or 0.95.

As delivered (``--from-stage win_prob`` on the checkpoints of 14 November 2026; this stage 10 s)
    * Post_Outcome_States: 1,808,649 for 200,961 Deliveries, built in 3 s. Terminal rules, the
      same for every candidate and in line with ``PostStates.terminal`` and
      ``PostStates.innings_over``: innings 1, 9,219 innings_end (end-of-innings states, scored by
      the innings-1 candidate); innings 2, 6,455 chase_won, 5,229 chase_lost, 69 chase_tied and 13
      chase_fraction. No pre-ball innings-1 state ends the innings, so every saved ``wp_pre``
      equals the ``win_prob`` table's, and the reloaded fold models reproduce its ``wp_raw``
      exactly.
    * M3-1 on 96,886 eligible Deliveries: logistic log loss 0.62786 (Brier 0.21907,
      Calibration_Error 0.0101), LightGBM 0.67676. M3-2 on 88,739: logistic 0.40639 (0.13220,
      0.0104), LightGBM 0.41907. The logistic baseline is the Primary_Model of both innings. M4
      LightGBM: 1.48334 against 1.51138 for the Phase-only baseline on 181,591 Legal_Balls,
      expected-runs calibration error 0.0264 runs per Legal_Ball.
    * Saved WP lies in [0.001, 0.999] in innings 1 (the calibration bounds) and [0, 1] in innings
      2, with no missing value. For the 98,359 innings-1 Legal_Balls without extras, Run_Out or
      other wicket, the WP of the actual class's Post_Outcome_State equals the next Delivery's
      pre-ball WP exactly (none of these states ends the innings); for the 88,660 such innings-2
      balls the median gap is 0 and the mean 0.0035, because RBQ and Achievable_Rate then follow
      the actual next striker.
    * Centred isotonic calibration: within 0.05-0.95 the primary pre-ball WP takes 104,074
      distinct values on 104,090 innings-1 Deliveries and 70,666 on 70,756 innings-2 Deliveries,
      and no Delivery there keeps its WP after a dot ball. 2,499 Deliveries have an unscaled
      Leverage_Index of exactly 0, none of them inside 0.05-0.95: in innings 1, 172 whose WP is
      held at a calibration bound (111 at 0.001, 61 at 0.999); in innings 2, the 249 chases
      already won and 2,078 held at a bound.
    * The end of innings 1: on the 813 last Legal_Balls, the WP after a dot minus the WP before
      averages -0.0039 (one more ball bowled and Par_Score at Scheduled_Balls), against +0.0199
      (mean absolute 0.0848) when these states took 1 - WP2 of the chase start; their mean
      unscaled Leverage_Index is 0.0079, against 0.0096 for the other innings-1 balls (0.0873
      before). The 1,033 innings-1 Deliveries at 9 wickets down average 0.0084 (0.0189 before).
      The jump at the innings break now falls between Deliveries: for the logistic models, WP1
      at the end of innings 1 minus 1 - WP2 at the chase start averages -0.0325 (mean absolute
      0.0854) over 845 matches (``win_prob`` file ``innings_break_gap.csv``).

With the M3 Alternative_Models (task 26.1; the stage run on a temporary store against the
checkpoints of 15 November 2026: 849 s, 744 s of them in the 100 classifier fits and 20
calibrators, while another test run shared the 4 CPUs)
    * Fits: XGBoost 129 s (innings 1) and 163 s (innings 2), rounds kept 24-106 and 49-140 on the
      outer fits (an innings-1 inner fit stopped at round 1); random forest 232 s and 219 s, 300
      trees each. 120 fits registered, every one excluding the folds of its Training_Rows'
      matches; each calibrator trains on the eligible rows of S_k only.
    * Pooled calibrated log loss, Brier, Calibration_Error on the same eligible Deliveries: M3-1
      (96,886) logistic 0.62768 / 0.21902 / 0.0120, LightGBM 0.67601 / 0.24017 / 0.0246,
      XGBoost 0.69440 / 0.23925 / 0.0285, random forest 0.65674 / 0.23195 / 0.0231; M3-2
      (88,739) logistic 0.40479 / 0.13164 / 0.0097, LightGBM 0.41596 / 0.13446 / 0.0166,
      XGBoost 0.42295 / 0.13722 / 0.0163, random forest 0.42876 / 0.13826 / 0.0122. The logistic
      baseline stays the Primary_Model of both innings, and the logistic and LightGBM rows and
      arrays are unchanged.
    * Overfitting check (held-out minus in-fold raw log loss per fold): XGBoost +0.035 to +0.300
      (innings 1) and +0.037 to +0.126 (innings 2); random forest +0.038 to +0.099 and -0.015 to
      +0.074. On 100 sampled eligible Deliveries per fold, no calibrated Alternative_Model WP
      breaks a monotone direction of Requirement 10, criterion 5, and the reloaded outer models
      reproduce every saved ``wp_raw`` exactly.

With the M4 Alternative_Models (task 26.2; the stage run on a temporary store against the same
checkpoints: 887 s wall and 1,483 s CPU on 4 CPUs, with about 0.6 GB of RAM free and the process
paging; peak commit 2.1 GB)
    * M3 Alternative_Models 375 s (XGBoost 18 s and 20 s, random forest 177 s and 161 s for
      innings 1 and 2; the 26.1 times above were taken while another run shared the CPUs). Every
      M3 number above is unchanged.
    * M4 outer fits on 144,241-145,985 Training_Rows each: XGBoost 27-39 s per fit (177 s in all),
      rounds kept 228/237/177/215/236 of 400, early stopping on 65-66 held-out matches
      (14,405-14,770 Legal_Balls); random forest 28-30 s per fit (145 s), 300 trees. The six
      variants of the 200,961 Deliveries took 27 s (XGBoost) and 66 s (random forest, 1 thread).
    * Raw class probabilities before rescaling: the XGBoost rows (float32 softmax) sum to 1 within
      9.2e-8 and the random-forest rows within 1.3e-15 in every variant, so neither candidate
      meets a condition of criterion 12; every saved float32 row sums to 1 within 5.6e-8.
    * On the same 181,591 Legal_Balls (Phase-only baseline 1.51138): pooled multiclass log loss
      LightGBM 1.48318, XGBoost 1.48196, random forest 1.48677; expected-runs calibration error
      (Legal_Ball-weighted mean |expected - actual bat runs| over the 9 Phase x Wickets_Bucket
      cells) 0.0248, 0.0216 and 0.0397 runs per Legal_Ball. XGBoost is the M4 Primary_Model. Each
      candidate's largest cell gap is in powerplay x 6-9 wickets, a cell of 13 Legal_Balls
      (+0.777, +0.587 and +0.871 runs).
    * With XGBoost as the M4 Primary_Model, the ``pressure`` stage on the same temporary store
      keeps every Live_Ball flag; its LI correlates 0.9986 with the LightGBM-based LI (mean
      absolute difference 0.032), and 97.5% of the batter and 97.6% of the bowler Pressure_Balls
      stay Pressure_Balls. The Pressure_Validator's Out_Of_Fold guard passes on 189,648
      Validation_Deliveries, reading the XGBoost fold-model table and fit ids from this checkpoint.
    * The reloaded fold models (``load_m4_models``) reproduce every saved variant value within
      3e-8 (float32 rounding). Checkpoint: 252 MB in all; XGBoost adds 58 MB of variant table and
      17 MB of XGBoost JSON, the random forest 57 MB and 50 MB of joblib (about 10 MB per forest).

Errors: ValueError or TypeError, naming the problem, for malformed arguments, such as a missing
column, tables that do not hold the same rows in the same order, an unknown model or candidate
name, a candidate listed twice, a non-failed candidate without a finite log loss, or candidates
of one model scored on different rows. StageError (exit code 5) when every candidate of a model
failed, so no Primary_Model can be selected. ``views.check_view`` raises ViewLeakageError (exit
code 4) for a view row whose values come from a fit that saw its fold, and the FitRegistry raises
FitLeakageError (exit code 4) for an Alternative_Model fit whose Training_Rows hold a match of an
excluded fold; neither is recorded as a fit failure.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from superr import folds, reconstruct, views
from superr.errors import PipelineError, StageError
from superr.models import common, outcome
from superr.models.outcome import N_CLASSES, VARIANT_NAMES
from superr.state import REPORT_MATCH_ID, STATE_INNINGS, STATE_TABLE, Phase
from superr.state import STAGE_NAME as STATE_STAGE

if TYPE_CHECKING:
    from superr.cases import BaseArtifacts, CaseResult
    from superr.pipeline import StageContext
    from superr.pressure import PostStates
    from superr.ranks import RankingComparison
    from superr.store import CheckpointStore

__all__ = [
    "ALTERNATIVE_FITS_TABLE",
    "ALTERNATIVE_FOLD_METRICS_FILE",
    "ALTERNATIVE_FOLD_METRICS_TABLE",
    "CHART_DPI",
    "COMPARISON_COLUMNS",
    "COMPARISON_FILE",
    "COMPARISON_TABLE",
    "FIT_FAILED",
    "LGBM",
    "LOGISTIC",
    "M3_ALTERNATIVES",
    "M3_CURVES_TABLE",
    "M3_CURVE_COLUMNS",
    "M3_FOLDER",
    "M3_INNINGS",
    "M3_MODELS",
    "M3_OUTSIDE",
    "M3_TIE_ORDER",
    "M4_ALTERNATIVES",
    "M4_CALIBRATION_COLUMNS",
    "M4_CALIBRATION_FILE",
    "M4_CALIBRATION_TABLE",
    "M4_MODEL",
    "M4_OUTSIDE",
    "M4_SUM",
    "M4_TIE_ORDER",
    "MODELS",
    "MODEL_CHOICE_FOLDER",
    "NON_FINITE",
    "POST_RULES_COLUMNS",
    "POST_RULES_TABLE",
    "PREDICTION_FAILED",
    "PRIMARY_CONSTANT",
    "RERUN_STAGE_NAME",
    "RERUN_TOP10_COLUMNS",
    "RERUN_TOP10_TABLE",
    "RF",
    "ROWS_COLUMNS",
    "ROWS_TABLE",
    "SOURCES_COLUMNS",
    "SOURCES_TABLE",
    "STAGE_NAME",
    "TIE_ABS_TOL",
    "TIE_ORDER",
    "TIE_REL_TOL",
    "WP_COLUMNS",
    "WP_CURVE_CHART_FILE",
    "WP_CURVE_COLUMNS",
    "WP_CURVE_FILE",
    "WP_POST_COLUMNS",
    "WP_SOURCE_COLUMNS",
    "WP_TABLE_COLUMNS",
    "XGB",
    "AlternativeFit",
    "CandidateResult",
    "M3Candidate",
    "M3Evaluation",
    "M4AlternativeFit",
    "M4Candidate",
    "PrimaryPredictions",
    "M4TieBreak",
    "RerunResult",
    "alternative_candidate",
    "calibration_chart_file",
    "calibration_file",
    "calibration_preferences",
    "candidate_table",
    "fit_alternative",
    "fit_m4_alternative",
    "load_m4_models",
    "load_predictions",
    "load_primary",
    "m3_calibration_curves",
    "m3_failure",
    "m3_metrics",
    "m4_failure",
    "match_clustered_paired_se",
    "precompute_predictions",
    "primary_models",
    "rerun",
    "rerun_candidates",
    "run_rerun_stage",
    "run_stage",
    "select_primary",
    "wp_table",
]


# ------------------------------------------------------------------------------------- constants

#: Name of the Pipeline_Stage and of its checkpoint.
STAGE_NAME = "model_choice"

#: The innings with an M3 classifier, and the ``model`` label of each model in the comparison table.
M3_INNINGS: tuple[int, ...] = STATE_INNINGS
M3_MODELS: Mapping[int, str] = MappingProxyType({1: "M3-1", 2: "M3-2"})
M4_MODEL = "M4"
MODELS: tuple[str, ...] = (M3_MODELS[1], M3_MODELS[2], M4_MODEL)

#: Candidate_Model family names: the FitRegistry families and the suffix of every table name.
#: LOGISTIC and LGBM are ``win_prob.LOGISTIC`` and ``win_prob.LGBM`` (= ``outcome.LGBM_FAMILY``);
#: XGB and RF are the Alternative_Models of task 26 (Config sections ``models.xgb`` and
#: ``models.rf``).
LOGISTIC = "logistic"
LGBM = outcome.LGBM_FAMILY
XGB = "xgb"
RF = "rf"

#: The M3 Alternative_Models that this stage fits itself (task 26.1), in tie order; they are
#: ``win_prob.ALTERNATIVE_FAMILIES``.
M3_ALTERNATIVES: tuple[str, ...] = (XGB, RF)

#: The M4 Alternative_Models that this stage fits itself (task 26.2), in tie order; they are
#: ``outcome.ALTERNATIVE_FAMILIES`` (Config sections ``models.xgb.m4`` and ``models.rf.m4``).
M4_ALTERNATIVES: tuple[str, ...] = (XGB, RF)

#: The tie orders of Requirement 23, criteria 7 and 8: the first tied candidate wins.
M3_TIE_ORDER: tuple[str, ...] = (LOGISTIC, LGBM, XGB, RF)
M4_TIE_ORDER: tuple[str, ...] = (LGBM, XGB, RF)
TIE_ORDER: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {M3_MODELS[1]: M3_TIE_ORDER, M3_MODELS[2]: M3_TIE_ORDER, M4_MODEL: M4_TIE_ORDER}
)

#: Two pooled log losses tie when math.isclose(a, b, rel_tol=TIE_REL_TOL, abs_tol=TIE_ABS_TOL).
TIE_REL_TOL = 1e-9
TIE_ABS_TOL = 1e-9

#: The start of each failure reason (module docstring, "Failure checks"; criterion 12).
FIT_FAILED = "fit failed"
PREDICTION_FAILED = "prediction failed"
NON_FINITE = "non-finite prediction"
M3_OUTSIDE = "M3 probability outside [0, 1]"
M4_OUTSIDE = "M4 probability outside [0, 1]"
M4_SUM = "M4 class probabilities differ from a sum of 1 by more than 1e-6"

#: The comparison table (checkpoint table and COMPARISON_FILE) and its columns.
COMPARISON_TABLE = "model_comparison"
COMPARISON_COLUMNS: tuple[str, ...] = (
    "model", "candidate", "n_rows", "log_loss", "brier", "auc", "calibration_error", "failure_reason", "primary",
    "top10_changes_batter", "top10_changes_bowler",
)  # fmt: skip

#: Every innings 1-2 Delivery in ``state`` order, the frame that load_predictions fills.
ROWS_TABLE = "rows"
ROWS_COLUMNS: tuple[str, ...] = ("row", "match_id", "innings", "fold")

#: The precomputed WP table of one M3 candidate and innings (``wp_table``): the Delivery, its
#: fold, the pre-ball WP and the WP of the Post_Outcome_State of each class, all after the
#: terminal rules (float64).
WP_POST_COLUMNS: tuple[str, ...] = tuple(f"wp_post_{cls}" for cls in range(N_CLASSES))
WP_COLUMNS: tuple[str, ...] = ("row", "fold", "wp_pre", *WP_POST_COLUMNS)

#: The pre-ball columns that every saved WP table also holds, from the candidate's pre-ball WP
#: table (``win_prob.WP_TABLE_COLUMNS``): the calibrated WP before the terminal rules, the
#: pre-ball terminal rule, and the outer fit and calibrator behind the row. WP_TABLE_COLUMNS is
#: the full layout of a saved WP table; load_predictions needs only WP_COLUMNS.
WP_SOURCE_COLUMNS: tuple[str, ...] = ("wp_raw", "terminal_rule", "fit_id", "calibrator_fit_id")
WP_TABLE_COLUMNS: tuple[str, ...] = (*WP_COLUMNS, *WP_SOURCE_COLUMNS)

#: The sandbox fits of the M3 Alternative_Models: ``win_prob.FIT_TABLE_COLUMNS`` of every fit and
#: ``win_prob.FOLD_METRICS_COLUMNS`` of every outer fit (checkpoint tables), the latter also as a file.
ALTERNATIVE_FITS_TABLE = "m3_alternative_fits"
ALTERNATIVE_FOLD_METRICS_TABLE = "m3_alternative_fold_metrics"
ALTERNATIVE_FOLD_METRICS_FILE = "m3_alternative_fold_metrics.csv"

#: Where the predictions of each non-failed candidate are saved: (model, candidate) -> (stage, table).
SOURCES_TABLE = "prediction_tables"
SOURCES_COLUMNS: tuple[str, ...] = ("model", "candidate", "stage", "table")

#: Metric tables: the M3 Calibration_Curves (overall and per Phase) and the M4 expected-runs cells.
M3_CURVES_TABLE = "m3_calibration_curves"
M3_CURVE_COLUMNS: tuple[str, ...] = (
    "innings", "candidate", "phase", "bin", "lower", "upper", "n_deliveries", "mean_wp", "observed_win_share",
)  # fmt: skip
M4_CALIBRATION_TABLE = "m4_expected_runs"
M4_CALIBRATION_COLUMNS: tuple[str, ...] = (
    "candidate", "phase", "wickets_bucket", "n_legal_balls", "mean_expected_bat_runs", "mean_actual_bat_runs",
    "expected_minus_actual",
)  # fmt: skip

#: The terminal-rule counts of the Post_Outcome_States, the same for every candidate.
POST_RULES_TABLE = "post_terminal_rules"
POST_RULES_COLUMNS: tuple[str, ...] = ("innings", "rule", "n_states")

#: Output folders and files (Requirement 23, criterion 13; Requirement 10, criterion 18).
MODEL_CHOICE_FOLDER = Path("models") / "model_choice"
COMPARISON_FILE = "model_comparison.csv"
M4_CALIBRATION_FILE = "m4_expected_runs_by_phase_wickets.csv"
M3_FOLDER = Path("models") / "m3"
WP_CURVE_FILE = f"wp_curve_{REPORT_MATCH_ID}.csv"
WP_CURVE_CHART_FILE = f"wp_curve_{REPORT_MATCH_ID}.png"
WP_CURVE_COLUMNS: tuple[str, ...] = (
    "delivery", "row", "match_id", "innings", "innings_ball_num", "overid", "over_ball_num", "batting_team_id",
    "lb_bowled", "score", "wickets", "runs_needed", "wicket", "terminal_rule", "m3_primary_model",
    "wp_batting_side", "wp_batting_first",
)  # fmt: skip

#: Resolution of every chart in dots per inch (the ``win_prob`` charts use the same).
CHART_DPI = 150

#: Data-derived Method_Constant: model -> the selected Primary_Model ({"M3-1": ..., "M3-2": ..., "M4": ...}).
PRIMARY_CONSTANT = "model_choice.primary_models"

_COMPONENT = "Model_Choice_Checker"
_N_FOLDS = "design_choices.D15_n_folds"
_WICKET_BUCKETS = "design_choices.D17_wicket_buckets"
#: The calibration-aware tie-break Config keys (Requirement 23, criterion 8; design DD-17): a
#: comparison-table model label -> preferred family map, and the relative log-loss band delta.
_CALIBRATION_PREFERRED = "model_choice.calibration_preferred"
_CALIBRATION_BAND = "model_choice.calibration_band"
_LOAD_STAGE = "load"
_MATCHES_TABLE = "matches"
_FLOAT_FORMAT = "%.6f"
_ALL_PHASES = "all"
_PHASE_NAMES: Mapping[int, str] = MappingProxyType({int(member): member.name.lower() for member in Phase})
_SUBSETS: tuple[str, ...] = (_ALL_PHASES, *(_PHASE_NAMES[code] for code in sorted(_PHASE_NAMES)))
_FAMILY_LABELS: Mapping[str, str] = MappingProxyType(
    {LOGISTIC: "Logistic baseline", LGBM: "LightGBM", XGB: "XGBoost", RF: "Random forest"}
)
_FAMILY_COLOURS: Mapping[str, str] = MappingProxyType(
    {LOGISTIC: "#d95f02", LGBM: "#1f5fa8", XGB: "#1b9e77", RF: "#7570b3"}
)
_CURVE_CHART_SIZE = (15.0, 4.6)  # inches
_WP_CHART_SIZE = (12.0, 5.0)
_WP_TICK_OVERS = (1, 6, 11, 16)  # overs marked on the WP curve's axis
_SHOWN = 5
_REASON_LENGTH = 300  # characters of an error message kept in a failure reason (one line in the CSV)
#: The criterion-12 conditions that a recorded failure reason starts with.
_CONDITIONS: tuple[str, ...] = (FIT_FAILED, PREDICTION_FAILED, NON_FINITE, M3_OUTSIDE, M4_OUTSIDE, M4_SUM)


# ---------------------------------------------------------------------------------- small helpers

def _rows(count: int) -> str:
    return f"{count:,} row" if count == 1 else f"{count:,} rows"


def _require(frame: Any, columns: Iterable[str], what: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{what} must be a DataFrame; got {type(frame).__name__}")
    absent = [column for column in columns if column not in frame.columns]
    if absent:
        raise ValueError(f"{what} lacks the column(s) {', '.join(absent)}")


def _ints(values: Any, what: str) -> np.ndarray:
    """``values`` (1-D whole numbers, none missing) as a new int64 array."""
    series = values if isinstance(values, pd.Series) else pd.Series(np.asarray(values))
    if series.ndim != 1 or pd.api.types.is_bool_dtype(series.dtype) or not pd.api.types.is_numeric_dtype(series.dtype):
        raise TypeError(f"{what} must be a 1-D array of whole numbers; got dtype {series.dtype}")
    if series.isna().any():
        raise ValueError(f"{what} has {int(series.isna().sum()):,} missing value(s)")
    floats = series.to_numpy(dtype=np.float64)
    if not (np.isfinite(floats) & (np.floor(floats) == floats)).all():
        raise ValueError(f"{what} must hold whole numbers")
    return series.to_numpy(dtype=np.int64, copy=True)


def _floats(values: Any, what: str) -> np.ndarray:
    """``values`` (1-D numbers) as a new float64 array, NaN for a missing value."""
    series = values if isinstance(values, pd.Series) else pd.Series(np.asarray(values))
    if series.ndim != 1 or pd.api.types.is_bool_dtype(series.dtype) or not pd.api.types.is_numeric_dtype(series.dtype):
        raise TypeError(f"{what} must be a 1-D array of numbers; got dtype {series.dtype}")
    return series.to_numpy(dtype=np.float64, na_value=np.nan, copy=True)


def _flags(values: Any, what: str) -> np.ndarray:
    series = values if isinstance(values, pd.Series) else pd.Series(np.asarray(values))
    if not pd.api.types.is_bool_dtype(series.dtype) or series.isna().any():
        raise TypeError(f"{what} must hold True or False on every row; got dtype {series.dtype}")
    return series.to_numpy(dtype=bool, copy=True)


def _check_innings(innings: Any) -> int:
    if isinstance(innings, (bool, np.bool_)) or not isinstance(innings, Integral) or int(innings) not in M3_INNINGS:
        raise ValueError(f"innings must be one of {', '.join(map(str, M3_INNINGS))}; got {innings!r}")
    return int(innings)


def _check_model(model: Any) -> str:
    if model not in TIE_ORDER:
        raise ValueError(f"model must be one of {', '.join(MODELS)}; got {model!r}")
    return str(model)


def _read_only(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


def _valid_probabilities(values: np.ndarray) -> bool:
    return bool(np.isfinite(values).all() and (values >= 0.0).all() and (values <= 1.0).all())


def _is_failure(reason: Any) -> bool:
    """True when ``reason`` is a failure reason: a non-empty text (None, NA and "" mean no failure)."""
    if reason is None or (not isinstance(reason, str) and pd.isna(reason)):
        return False
    return bool(str(reason).strip())


def _same_rows(table: pd.DataFrame, row_ids: np.ndarray, what: str) -> None:
    _require(table, ("row",), what)
    if len(table) != len(row_ids) or not np.array_equal(_ints(table["row"], f"{what} column 'row'"), row_ids):
        raise ValueError(
            f"{what} does not hold the Deliveries of the state table in its order; pass the tables of one Run"
        )


def _ordered(names: Iterable[str], order: Sequence[str]) -> list[str]:
    """``names`` in ``order`` (ValueError for a name that ``order`` lacks)."""
    names = list(names)
    unknown = [name for name in names if name not in order]
    if unknown:
        raise ValueError(f"unknown Candidate_Model name(s) {unknown}; the names are {', '.join(order)}")
    return sorted(names, key=order.index)


def wp_table(innings: int, family: str) -> str:
    """The checkpoint table name of the precomputed WP of one M3 candidate and innings: ``wp_inn<innings>_<family>``.

    The ``win_prob`` stage names its pre-ball WP tables the same way (``win_prob.wp_table``).
    """
    number = _check_innings(innings)
    if family not in M3_TIE_ORDER:
        raise ValueError(f"family must be one of {', '.join(M3_TIE_ORDER)}; got {family!r}")
    return f"wp_inn{number}_{family}"


def calibration_file(innings: int) -> str:
    """The file name of the M3 Calibration_Curves of ``innings`` under MODEL_CHOICE_FOLDER."""
    return f"m3_calibration_curve_inn{_check_innings(innings)}.csv"


def calibration_chart_file(innings: int) -> str:
    """The file name of the M3 Calibration_Curve chart of ``innings`` under MODEL_CHOICE_FOLDER."""
    return f"m3_calibration_curve_inn{_check_innings(innings)}.png"


# ===================================================================================== candidates

@dataclass(frozen=True, eq=False)
class M3Candidate:
    """One M3 Candidate_Model family with its fits of both innings (module docstring, "Candidates").

    ``family``: a name of M3_TIE_ORDER. ``models``: innings -> the calibrated outer fold models of
    the innings (``win_prob.CalibratedFolds``). ``pre``: innings -> the family's pre-ball WP table
    of the innings (``win_prob.WP_TABLE_COLUMNS``: ``row``, ``fold``, ``wp_pre``, ``wp_raw``,
    ``terminal_rule``, ``fit_id``, ``calibrator_fit_id``), the innings' Deliveries in ``state``
    order. ``fit_failure``: innings -> why the innings' fit failed (module docstring, "Failure
    checks"); such an innings needs no models or table. ``stage``: the checkpoint that holds the
    fits (``win_prob``, or STAGE_NAME for an Alternative_Model).
    """

    family: str
    models: Mapping[int, Any] = field(default_factory=dict)
    pre: Mapping[int, pd.DataFrame] = field(default_factory=dict)
    fit_failure: Mapping[int, str] = field(default_factory=dict)
    stage: str = "win_prob"

    def __post_init__(self) -> None:
        if self.family not in M3_TIE_ORDER:
            raise ValueError(f"an M3 candidate family must be one of {', '.join(M3_TIE_ORDER)}; got {self.family!r}")
        for name in ("models", "pre", "fit_failure"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"M3Candidate.{name} must map each innings to its value; got {type(value).__name__}")
            object.__setattr__(self, name, MappingProxyType(dict(value)))
        for innings in M3_INNINGS:
            if _is_failure(self.fit_failure.get(innings)):
                continue
            if innings not in self.models or innings not in self.pre:
                raise ValueError(
                    f"M3 candidate {self.family!r} needs the fold models and the pre-ball WP table of innings "
                    f"{innings}, or a fit_failure reason for it"
                )
            what = f"the {self.family} pre-ball WP table of innings {innings}"
            _require(self.pre[innings], ("row", "fold", "wp_pre", *WP_SOURCE_COLUMNS), what)


@dataclass(frozen=True, eq=False)
class M4Candidate:
    """One M4 Candidate_Model family (module docstring, "Candidates").

    ``family``: a name of M4_TIE_ORDER. ``variants``: ``outcome.VARIANT_NAMES`` -> float64 arrays
    of shape (rows, 9), every innings 1-2 Delivery in ``state`` order. ``models``: its
    ``outcome.OutcomeFoldModels``. ``stage``: the checkpoint that holds (``outcome``) or receives
    (STAGE_NAME, with the fold models) its ``outcome.predictions_table(family)`` table.
    ``fit_failure``: why the candidate failed before the comparison ("" when it did not): its fit
    failed, or, for an Alternative_Model, predicting its variants failed or their raw arrays broke
    a rule of criterion 12 (``fit_m4_alternative``). A failed candidate needs no variants or
    models; when it has an ``actual`` array of valid probabilities and its models, its metrics
    are still computed.
    """

    family: str
    variants: Mapping[str, Any] = field(default_factory=dict)
    models: Any = None
    stage: str = outcome.STAGE_NAME
    fit_failure: str = ""

    def __post_init__(self) -> None:
        if self.family not in M4_TIE_ORDER:
            raise ValueError(f"an M4 candidate family must be one of {', '.join(M4_TIE_ORDER)}; got {self.family!r}")
        if not isinstance(self.variants, Mapping):
            raise TypeError(
                f"M4Candidate.variants must map each variant to its array; got {type(self.variants).__name__}"
            )
        object.__setattr__(self, "variants", MappingProxyType(dict(self.variants)))
        if _is_failure(self.fit_failure):
            return
        missing = [name for name in VARIANT_NAMES if name not in self.variants]
        if missing:
            raise ValueError(f"M4 candidate {self.family!r} lacks the variant(s) {', '.join(missing)}")
        if not isinstance(self.models, outcome.OutcomeFoldModels):
            raise TypeError(
                f"M4 candidate {self.family!r} needs its outcome.OutcomeFoldModels; got {type(self.models).__name__}"
            )


# ====================================================================== M3 Alternative_Models (23.1-23.4)

def _error_text(exc: BaseException, *, typed: bool = True) -> str:
    """The message of ``exc`` on one line, cut to _REASON_LENGTH characters, after ``<error type>: `` when ``typed``."""
    text = " ".join(str(exc).split())
    if len(text) > _REASON_LENGTH:
        text = text[: _REASON_LENGTH - 3] + "..."
    if not typed:
        return text
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


@dataclass(frozen=True, eq=False)
class AlternativeFit:
    """The sandbox fits of one M3 Alternative_Model and innings (fit_alternative; module docstring).

    ``innings`` and ``family`` (a name of M3_ALTERNATIVES); ``fold_models``: the
    ``win_prob.FoldModels`` of its D15 x D15 classifier fits and D15 calibrators, and ``pre``: its
    pre-ball WP table (``win_prob.WP_TABLE_COLUMNS``), both None when the fits failed;
    ``failure``: the failure reason, "" when every fit succeeded; ``seconds``: the wall time.
    """

    innings: int
    family: str
    fold_models: Any
    pre: pd.DataFrame | None
    failure: str
    seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "innings", _check_innings(self.innings))
        if self.family not in M3_ALTERNATIVES:
            raise ValueError(f"an M3 Alternative_Model is one of {', '.join(M3_ALTERNATIVES)}; got {self.family!r}")
        failed = _is_failure(self.failure)
        object.__setattr__(self, "failure", str(self.failure).strip() if failed else "")
        if failed != (self.fold_models is None or self.pre is None):
            raise ValueError("an AlternativeFit holds its fold models and pre-ball WP table, or a failure reason")


def fit_alternative(
    innings: int,
    family: Any,
    view_by_k: Mapping[int, pd.DataFrame],
    labels: pd.DataFrame,
    settings: Any,
    registry: Any,
    *,
    state: pd.DataFrame,
) -> AlternativeFit:
    """The sandbox fits of one M3 Alternative_Model of ``innings`` (criteria 1-4 and 12; module docstring).

    ``family``: the ``win_prob.CandidateFamily`` XGB or RF (``win_prob.candidate_family``);
    ``view_by_k``, ``labels`` (``m3_labels``), ``settings``, ``registry`` (``ctx.registry``) and
    ``state``: as for ``win_prob.fit_innings``, which runs the fits. Returns an AlternativeFit with
    the FoldModels and the pre-ball WP table, or with the failure reason when a fit raised:
    NON_FINITE or M3_OUTSIDE for ``win_prob.InvalidPrediction``, FIT_FAILED with the error for any
    other Exception. Raises PipelineError (FitLeakageError or ViewLeakageError with exit code 4,
    DataError, StageError) and MemoryError instead of recording them, and ValueError for a family
    that is not an Alternative_Model.
    """
    from superr.models import win_prob

    number = _check_innings(innings)
    if not isinstance(family, win_prob.CandidateFamily) or family.name not in M3_ALTERNATIVES:
        raise ValueError(
            f"family must be the win_prob.CandidateFamily of an M3 Alternative_Model ({', '.join(M3_ALTERNATIVES)}); "
            f"got {family!r}"
        )
    started = time.perf_counter()
    try:
        fold_models = win_prob.fit_innings(number, family, view_by_k, labels, settings, registry, state=state)
    except (PipelineError, MemoryError):
        raise  # a leakage guard, data or stage error, or no memory left, stops the Run
    except win_prob.InvalidPrediction as exc:  # a fitted model's raw output is not a finite probability
        reason = f"{NON_FINITE if exc.n_non_finite else M3_OUTSIDE}: {_error_text(exc, typed=False)}"
        return AlternativeFit(number, family.name, None, None, reason, time.perf_counter() - started)
    except Exception as exc:  # criterion 12: a fit that fails is recorded and the Run goes on
        reason = f"{FIT_FAILED}: {_error_text(exc)}"
        return AlternativeFit(number, family.name, None, None, reason, time.perf_counter() - started)
    pre = win_prob.pre_ball_table(number, fold_models, state, labels)
    return AlternativeFit(number, family.name, fold_models, pre, "", time.perf_counter() - started)


def alternative_candidate(family: str, fits: Iterable[AlternativeFit]) -> M3Candidate:
    """The M3Candidate of Alternative_Model ``family`` from its AlternativeFit of each innings (``stage`` STAGE_NAME).

    An innings whose fits failed gets its failure reason as ``fit_failure``. Raises ValueError
    unless ``fits`` holds exactly one AlternativeFit of ``family`` for each innings.
    """
    mine = [fit for fit in fits if isinstance(fit, AlternativeFit) and fit.family == family]
    by_innings = {fit.innings: fit for fit in mine}
    if len(mine) != len(by_innings) or sorted(by_innings) != list(M3_INNINGS):
        raise ValueError(f"fits must hold one AlternativeFit of {family!r} for each innings {list(M3_INNINGS)}")
    return M3Candidate(
        family=family,
        models={number: fit.fold_models.calibrated for number, fit in by_innings.items() if not fit.failure},
        pre={number: fit.pre for number, fit in by_innings.items() if not fit.failure},
        fit_failure={number: fit.failure for number, fit in by_innings.items() if fit.failure},
        stage=STAGE_NAME,
    )


# ====================================================================== M4 Alternative_Models (23.1-23.4)

@dataclass(frozen=True, eq=False)
class M4AlternativeFit:
    """The outer fits of one M4 Alternative_Model and its variant predictions (fit_m4_alternative).

    ``family``: a name of M4_ALTERNATIVES. ``candidate``: its M4Candidate (``stage`` STAGE_NAME),
    with the rescaled variant arrays, or with ``fit_failure`` set when the fits, the predictions or
    the raw-array checks failed. ``deviation``: variant -> the largest |row sum - 1| of the raw
    arrays over their finite rows, before any rescaling (empty when nothing was predicted).
    ``fit_seconds`` and ``predict_seconds``: the wall times of the D15 fits and of the
    predictions.
    """

    family: str
    candidate: M4Candidate
    deviation: Mapping[str, float]
    fit_seconds: float
    predict_seconds: float

    def __post_init__(self) -> None:
        if self.family not in M4_ALTERNATIVES:
            raise ValueError(f"an M4 Alternative_Model is one of {', '.join(M4_ALTERNATIVES)}; got {self.family!r}")
        if not isinstance(self.candidate, M4Candidate) or self.candidate.family != self.family:
            raise ValueError(f"candidate must be the M4Candidate of {self.family!r}")
        object.__setattr__(self, "deviation", MappingProxyType(dict(self.deviation)))

    @property
    def failure(self) -> str:
        """The failure reason ("" when the candidate passed)."""
        return self.candidate.fit_failure


def _largest_deviation(values: np.ndarray) -> float:
    """The largest |row sum - 1| over the rows of ``values`` whose sum is finite (NaN when there is none)."""
    deviation = np.abs(values.sum(axis=1) - 1.0)
    finite = deviation[np.isfinite(deviation)]
    return float(finite.max()) if len(finite) else math.nan


def fit_m4_alternative(
    registry: Any,
    plan: Sequence[folds.FitSpec],
    family: outcome.OutcomeFamily,
    rows: outcome.OutcomeRows,
    features: Mapping[int, pd.DataFrame],
    average_skills: Mapping[int, outcome.AverageSkill],
    log: common.WarningLog,
) -> M4AlternativeFit:
    """The outer fits of one M4 Alternative_Model and its six variants (criteria 1, 2, 4 and 12; module docstring).

    ``family``: the OutcomeFamily XGB or RF (``outcome.candidate_family``). ``registry``, ``plan``,
    ``rows``, ``features`` and ``average_skills``: as for ``outcome.fit_outcome``, which runs the
    D15 outer fits on the Training_Rows, labels, features and folds of the LightGBM M4; ``log``:
    the Run_Log. The fitted models then predict the six variants of every innings 1-2 Delivery of
    their fold (``outcome.variant_probabilities``), and the raw arrays are checked with
    ``m4_failure`` before ``common.renormalise`` rescales each row. The rescaled arrays are rounded
    to the float32 of the saved ``m4_<family>`` table (``outcome.variant_table``), so the
    candidate is scored on the values that later stages read, as the LightGBM candidate is.

    Returns an M4AlternativeFit. Its candidate has ``fit_failure`` set (criterion 12) to
    FIT_FAILED with the error when a fit raised, to PREDICTION_FAILED with the error when a
    prediction raised, and to the ``m4_failure`` reason (NON_FINITE, M4_OUTSIDE or M4_SUM) of the
    raw arrays; such a candidate keeps its models and its raw ``actual`` array, when it has them,
    for the metrics. Raises PipelineError (FitLeakageError with exit code 4, DataError, StageError)
    and MemoryError instead of recording them, and ValueError for a family that is not an
    Alternative_Model.
    """
    if not isinstance(family, outcome.OutcomeFamily) or family.name not in M4_ALTERNATIVES:
        raise ValueError(
            f"family must be the outcome.OutcomeFamily of an M4 Alternative_Model ({', '.join(M4_ALTERNATIVES)}); "
            f"got {family!r}"
        )
    name = family.name
    started = time.perf_counter()
    try:
        models = outcome.fit_outcome(registry, plan, family, rows, features, average_skills)
    except (PipelineError, MemoryError):
        raise  # a leakage guard, data or stage error, or no memory left, stops the Run
    except Exception as exc:  # criterion 12: a fit that fails is recorded and the Run goes on
        failed = M4Candidate(family=name, stage=STAGE_NAME, fit_failure=f"{FIT_FAILED}: {_error_text(exc)}")
        return M4AlternativeFit(name, failed, {}, time.perf_counter() - started, 0.0)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    try:
        raw = outcome.variant_probabilities(models, features, rows.fold, finite=False)
    except (PipelineError, MemoryError):
        raise
    except Exception as exc:
        reason = f"{PREDICTION_FAILED}: {_error_text(exc)}"
        failed = M4Candidate(family=name, models=models, stage=STAGE_NAME, fit_failure=reason)
        return M4AlternativeFit(name, failed, {}, fit_seconds, time.perf_counter() - started)
    deviation = {variant: _largest_deviation(values) for variant, values in raw.items()}
    reason = m4_failure(raw)  # the arrays as the model gives them, before any rescaling
    if reason:
        failed = M4Candidate(
            family=name, variants={outcome.ACTUAL: raw[outcome.ACTUAL]}, models=models, stage=STAGE_NAME,
            fit_failure=reason,
        )  # fmt: skip
        return M4AlternativeFit(name, failed, deviation, fit_seconds, time.perf_counter() - started)
    rescaled = {
        variant: common.renormalise(values, log, what=f"M4 {name} {variant} class probabilities")
        for variant, values in raw.items()
    }
    del raw
    variants = outcome.variant_arrays(outcome.variant_table(rescaled, rows.row))  # the saved float32 values
    candidate = M4Candidate(family=name, variants=variants, models=models, stage=STAGE_NAME)
    return M4AlternativeFit(name, candidate, deviation, fit_seconds, time.perf_counter() - started)


def _m4_average_skills(
    models: outcome.OutcomeFoldModels, view_by_k: Mapping[int, pd.DataFrame], row_fold: np.ndarray
) -> dict[int, outcome.AverageSkill]:
    """The Average_Skill of each fold k, from the LightGBM M4 fold models, checked against view V_k.

    The ``outcome`` stage computed it from the M1 fit that the fold-k rows of V_k name in
    ``m1_fit`` (``outcome.average_skill``); this checks that they still name that fit, so every M4
    candidate of fold k uses the same typical player. Raises ValueError otherwise.
    """
    skills: dict[int, outcome.AverageSkill] = {}
    for k, view in view_by_k.items():
        skill = models.fold(k).average_skill
        named = pd.Series(view[views.M1_FIT].array)[row_fold == k]
        ids = named.dropna().unique().tolist()
        if ids != [skill.m1_fit] or named.isna().any():
            raise ValueError(
                f"the fold-{k} rows of view V_{k} name the M1 fit(s) {ids[:_SHOWN]}, but the Average_Skill of the "
                f"LightGBM M4 fit of fold {k} comes from {skill.m1_fit!r}; pass the checkpoints of one Run"
            )
        skills[int(k)] = skill
    return skills


#: The array fields of an M3Evaluation (read-only once set).
_EVALUATION_ARRAYS: tuple[str, ...] = ("row", "fold", "wp_raw", "wp_pre", "wp_post", "post_rule")


@dataclass(frozen=True, eq=False)
class M3Evaluation:
    """The precomputed WP of one M3 candidate and innings (precompute_predictions; module docstring).

    For every Delivery of the innings, in ``state`` order: ``row``, ``fold``, ``wp_raw`` (the
    calibrated Out_Of_Fold WP before the terminal rules, the input of the metrics), ``wp_pre``
    (after them) and ``wp_post`` (n, 9), the WP of the Post_Outcome_State of each class after
    them; ``post_rule`` (n, 9) the ``win_prob.TERMINAL_RULES`` code of each Post_Outcome_State.
    ``failure``: the failure reason ("" when the candidate passed); the arrays are None when no
    prediction was made. ``seconds``: the evaluation time.
    """

    innings: int
    family: str
    row: np.ndarray | None
    fold: np.ndarray | None
    wp_raw: np.ndarray | None
    wp_pre: np.ndarray | None
    wp_post: np.ndarray | None
    post_rule: np.ndarray | None
    failure: str
    seconds: float

    def __post_init__(self) -> None:
        for name in _EVALUATION_ARRAYS:
            value = getattr(self, name)
            if value is not None:
                _read_only(value)


# ======================================================================== failure checks (23.12)

def _counted(counts: Mapping[str, int]) -> str:
    return "; ".join(f"{count:,} value(s) of {name}" for name, count in counts.items() if count)


def m3_failure(predictions: Mapping[str, Any]) -> str:
    """The criterion-12 failure reason of an M3 candidate's predictions, or "" when they pass.

    ``predictions`` maps a name (``wp_raw``, ``wp_pre``, ``wp_post``) to an array of WP values of
    any shape. The reason names the first condition met, NON_FINITE (a missing or
    infinite value) before M3_OUTSIDE (a value outside [0, 1]), with the count per array.
    """
    if not isinstance(predictions, Mapping):
        raise TypeError(f"predictions must map each name to an array of WP values; got {type(predictions).__name__}")
    arrays = {str(name): np.asarray(values, dtype=np.float64) for name, values in predictions.items()}
    missing = {name: int((~np.isfinite(values)).sum()) for name, values in arrays.items()}
    if any(missing.values()):
        return f"{NON_FINITE}: {_counted(missing)}"
    outside = {name: int(((values < 0.0) | (values > 1.0)).sum()) for name, values in arrays.items()}
    if any(outside.values()):
        return f"{M3_OUTSIDE}: {_counted(outside)}"
    return ""


def m4_failure(variants: Mapping[str, Any]) -> str:
    """The criterion-12 failure reason of an M4 candidate's variant arrays, or "" when they pass.

    ``variants`` maps each variant name to an array of shape (rows, 9). The reason names the first
    condition met: NON_FINITE, M4_OUTSIDE (a probability outside [0, 1]) or M4_SUM (a row whose
    probabilities differ from a sum of 1 by more than ``common.PROB_SUM_TOLERANCE``, 1e-6), with
    the count per variant and, for M4_SUM, the largest difference. Raises ValueError for an array
    that is not of shape (rows, 9).
    """
    if not isinstance(variants, Mapping) or not variants:
        raise TypeError("variants must map at least one variant name to its (rows, 9) probability array")
    arrays: dict[str, np.ndarray] = {}
    for name, values in variants.items():
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != N_CLASSES:
            raise ValueError(f"the {name} probabilities must have shape (rows, {N_CLASSES}); got {array.shape}")
        arrays[str(name)] = array
    missing = {name: int((~np.isfinite(array)).sum()) for name, array in arrays.items()}
    if any(missing.values()):
        return f"{NON_FINITE}: {_counted(missing)}"
    outside = {name: int(((array < 0.0) | (array > 1.0)).sum()) for name, array in arrays.items()}
    if any(outside.values()):
        return f"{M4_OUTSIDE}: {_counted(outside)}"
    deviation = {name: np.abs(array.sum(axis=1) - 1.0) for name, array in arrays.items()}
    off = {name: int((values > common.PROB_SUM_TOLERANCE).sum()) for name, values in deviation.items()}
    if any(off.values()):
        largest = max(float(values.max()) for values in deviation.values() if len(values))
        rows = "; ".join(f"{count:,} row(s) of {name}" for name, count in off.items() if count)
        return f"{M4_SUM}: {rows} (largest difference {largest:.3e})"
    return ""


# ========================================================================= precomputed predictions

def _failed_evaluation(innings: int, family: str, reason: str, seconds: float, **arrays: Any) -> M3Evaluation:
    values = {name: arrays.get(name) for name in _EVALUATION_ARRAYS}
    return M3Evaluation(innings=innings, family=family, failure=reason, seconds=seconds, **values)


def _fit_failure_reason(reason: Any) -> str:
    """The recorded reason of a ``fit_failure``: kept when it names a criterion-12 condition, else after FIT_FAILED."""
    text = str(reason).strip()
    return text if text.startswith(_CONDITIONS) else f"{FIT_FAILED}: {text}"


def _evaluate(win_prob: Any, number: int, candidate: M3Candidate, inputs: Mapping[str, Any]) -> M3Evaluation:
    """precompute_predictions for one candidate (module docstring, "Precomputed predictions")."""
    started = time.perf_counter()
    family = candidate.family
    reason = candidate.fit_failure.get(number)
    if _is_failure(reason):
        return _failed_evaluation(number, family, _fit_failure_reason(reason), 0.0)
    pre = candidate.pre[number]
    what = f"the {family} pre-ball WP table of innings {number}"
    row = _ints(pre["row"], f"{what} column 'row'")
    if not np.array_equal(row, inputs["state_rows"]):
        raise ValueError(f"{what} does not hold the innings-{number} Deliveries of the Post_Outcome_States in order")
    fold = _ints(pre["fold"], f"{what} column 'fold'")
    wp_raw = _floats(pre["wp_raw"], f"{what} column 'wp_raw'")
    wp_pre = _floats(pre["wp_pre"], f"{what} column 'wp_pre'")
    try:
        models = {number: candidate.models[number]}  # this innings' candidate alone (criterion 17)
        result = win_prob.win_probability(number, inputs["X"], fold=inputs["fold"], models=models)
    except win_prob.InvalidPrediction as exc:  # an XGB or RF output that is not a finite probability
        condition = NON_FINITE if exc.n_non_finite else M3_OUTSIDE
        reason, seconds = f"{condition}: {_error_text(exc, typed=False)}", time.perf_counter() - started
        return _failed_evaluation(number, family, reason, seconds, row=row, fold=fold, wp_raw=wp_raw)
    except ValueError as exc:
        reason, seconds = f"{PREDICTION_FAILED}: {exc}", time.perf_counter() - started
        return _failed_evaluation(number, family, reason, seconds, row=row, fold=fold, wp_raw=wp_raw)
    wp_post = np.array(result.wp, dtype=np.float64).reshape(-1, N_CLASSES)
    post_rule = np.array(result.rule, dtype=np.int8).reshape(-1, N_CLASSES)
    return M3Evaluation(
        innings=number,
        family=family,
        row=row,
        fold=fold,
        wp_raw=wp_raw,
        wp_pre=wp_pre,
        wp_post=wp_post,
        post_rule=post_rule,
        failure=m3_failure({"wp_raw": wp_raw, "wp_pre": wp_pre, "wp_post": wp_post}),
        seconds=time.perf_counter() - started,
    )


def precompute_predictions(
    innings: int, candidates: Sequence[M3Candidate], post: PostStates
) -> dict[str, M3Evaluation]:
    """The precomputed WP of every M3 candidate of ``innings`` (DD-8; module docstring, "Precomputed predictions").

    ``candidates``: the M3Candidates, each family once. ``post``: the PostStates of the Run. Each
    candidate's states of ``innings`` are scored by its own calibrated fold models of that innings
    only, the end-of-innings states of innings 1 included. Returns family -> M3Evaluation; the
    evaluation of a candidate whose prediction raised ValueError, or whose arrays fail
    ``m3_failure``, carries the failure reason. Raises ValueError for inputs that do not fit
    together.
    """
    from superr.models import win_prob

    number = _check_innings(innings)
    families = [candidate.family for candidate in candidates]
    if not all(isinstance(candidate, M3Candidate) for candidate in candidates) or len(set(families)) != len(families):
        raise ValueError("candidates must be M3Candidates with distinct families")
    inputs: dict[str, Any] = {
        "X": post.features(number),
        "fold": post.fold(number),
        "state_rows": post.frame(number)["row"].to_numpy(dtype=np.int64)[::N_CLASSES],
    }
    return {
        candidate.family: _evaluate(win_prob, number, candidate, inputs)
        for candidate in sorted(candidates, key=lambda item: M3_TIE_ORDER.index(item.family))
    }


def _wp_frame(evaluation: M3Evaluation, pre: pd.DataFrame) -> pd.DataFrame:
    """The WP_TABLE_COLUMNS table of one evaluation (``row`` int32 and ``fold`` int8 as in ``win_prob``; WP float64).

    ``pre`` is the candidate's pre-ball WP table of the innings, whose rows the evaluation holds
    in the same order; it gives WP_SOURCE_COLUMNS (``wp_raw`` is the evaluation's, read from it).
    """
    what = f"the {evaluation.family} pre-ball WP table of innings {evaluation.innings}"
    if len(pre) != len(evaluation.row) or not np.array_equal(_ints(pre["row"], f"{what} column 'row'"), evaluation.row):
        raise ValueError(f"{what} does not hold the rows of its precomputed WP in order")
    data: dict[str, Any] = {
        "row": evaluation.row.astype(np.int32),
        "fold": evaluation.fold.astype(np.int8),
        "wp_pre": np.array(evaluation.wp_pre, dtype=np.float64),
    }
    for cls, name in enumerate(WP_POST_COLUMNS):
        data[name] = np.array(evaluation.wp_post[:, cls], dtype=np.float64)
    data["wp_raw"] = np.array(evaluation.wp_raw, dtype=np.float64)
    for name in ("terminal_rule", "fit_id", "calibrator_fit_id"):
        data[name] = pd.array(pre[name].astype(str).to_numpy(dtype=object).astype(str), dtype="str")
    return pd.DataFrame(data, columns=list(WP_TABLE_COLUMNS))


# ================================================================== metrics and the comparison table

@dataclass(frozen=True)
class CandidateResult:
    """One row of the comparison table (module docstring, "Metrics and the comparison table").

    ``model``: M3-1, M3-2 or M4; ``candidate``: a family of the model's tie order; ``n_rows``: the
    Deliveries (M3) or Legal_Balls (M4) behind the metrics; the metrics (NaN where undefined);
    ``failure_reason``: "" for a candidate that passed every check.
    """

    model: str
    candidate: str
    n_rows: int
    log_loss: float = math.nan
    brier: float = math.nan
    auc: float = math.nan
    calibration_error: float = math.nan
    failure_reason: str = ""

    def __post_init__(self) -> None:
        order = TIE_ORDER[_check_model(self.model)]
        if self.candidate not in order:
            raise ValueError(f"{self.model} candidates must be one of {', '.join(order)}; got {self.candidate!r}")
        if isinstance(self.n_rows, (bool, np.bool_)) or not isinstance(self.n_rows, Integral) or self.n_rows < 0:
            raise ValueError(f"n_rows must be a count of rows; got {self.n_rows!r}")
        for name in ("log_loss", "brier", "auc", "calibration_error"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a number (NaN where undefined); got {value!r}")
            object.__setattr__(self, name, float(value))
        reason = "" if not _is_failure(self.failure_reason) else str(self.failure_reason)
        object.__setattr__(self, "n_rows", int(self.n_rows))
        object.__setattr__(self, "failure_reason", reason)


_NO_METRICS: Mapping[str, float] = MappingProxyType(
    {"log_loss": math.nan, "brier": math.nan, "auc": math.nan, "calibration_error": math.nan}
)


def m3_metrics(y: Any, p: Any) -> dict[str, float]:
    """The criterion-5 metrics of WP ``p`` against labels ``y`` (0 or 1), pooled over every row given.

    Returns ``log_loss`` (Pooled_Held_Out_Log_Loss), ``brier``, ``auc`` and ``calibration_error``
    from ``win_prob.wp_metrics``. Raises its errors.
    """
    from superr.models import win_prob

    values = win_prob.wp_metrics(y, p)
    return {name: float(values[name]) for name in ("log_loss", "brier", "auc", "calibration_error")}


def m3_calibration_curves(y: Any, p: Any, phase: Any, *, innings: int, candidate: str) -> pd.DataFrame:
    """The Calibration_Curves of WP ``p`` against labels ``y``, over all rows and per Phase (M3_CURVE_COLUMNS).

    ``phase`` gives the Phase code of each row. A Phase without a row gets no curve.
    """
    from superr.models import win_prob

    labels = np.asarray(y)
    values = np.asarray(p, dtype=np.float64)
    codes = _ints(phase, "phase")
    parts = []
    for subset in _SUBSETS:
        mask = np.ones(len(values), dtype=bool) if subset == _ALL_PHASES else codes == _phase_code(subset)
        if not mask.any():
            continue
        curve = win_prob.calibration_curve(values[mask], labels[mask])
        curve.insert(0, "phase", subset)
        curve.insert(0, "candidate", candidate)
        curve.insert(0, "innings", int(innings))
        parts.append(curve)
    return pd.concat(parts, ignore_index=True)[list(M3_CURVE_COLUMNS)]


def _phase_code(name: str) -> int:
    return next(code for code, text in _PHASE_NAMES.items() if text == name)


def _m3_results(
    evaluations: Mapping[str, M3Evaluation], innings: int, label: np.ndarray, eligible: np.ndarray, phase: np.ndarray
) -> tuple[list[CandidateResult], list[pd.DataFrame]]:
    """The CandidateResults and Calibration_Curves of one innings, every candidate on the same eligible Deliveries."""
    model = M3_MODELS[innings]
    y, codes = label[eligible], phase[eligible]
    results, curves = [], []
    for family in _ordered(evaluations, M3_TIE_ORDER):
        evaluation = evaluations[family]
        metrics = dict(_NO_METRICS)
        if evaluation.wp_raw is not None and _valid_probabilities(evaluation.wp_raw[eligible]):
            p = evaluation.wp_raw[eligible]
            metrics = m3_metrics(y, p)
            curves.append(m3_calibration_curves(y, p, codes, innings=innings, candidate=family))
        results.append(
            CandidateResult(model, family, int(eligible.sum()), failure_reason=evaluation.failure, **metrics)
        )
    return results, curves


def _runs_calibration_error(table: pd.DataFrame) -> float:
    """The Legal_Ball-weighted mean |mean expected - mean actual bat runs| over the cells with a Legal_Ball."""
    count = table["n_legal_balls"].to_numpy(dtype=np.float64)
    filled = count > 0
    if not filled.any():
        return math.nan
    expected = table["mean_expected_bat_runs"].to_numpy(dtype=np.float64)
    gap = np.abs(expected - table["mean_actual_bat_runs"].to_numpy(dtype=np.float64))
    return float(np.sum(count[filled] * gap[filled]) / np.sum(count[filled]))


def _m4_results(
    candidates: Sequence[M4Candidate], rows: outcome.OutcomeRows, wicket_buckets: Any
) -> tuple[list[CandidateResult], list[pd.DataFrame], dict[str, float]]:
    """The CandidateResults, expected-runs tables and Phase-only baseline log loss of the M4 candidates.

    A candidate with a ``fit_failure`` keeps that reason (``_fit_failure_reason``); any other is
    checked with ``m4_failure``. The metrics come from the ``actual`` array whenever the candidate
    has one of valid probabilities and its fold models, failed or not.
    """
    n_rows = int(rows.trainable.sum())
    results, tables, baseline = [], [], {}
    for candidate in sorted(candidates, key=lambda item: M4_TIE_ORDER.index(item.family)):
        if _is_failure(candidate.fit_failure):
            failure = _fit_failure_reason(candidate.fit_failure)
        else:
            failure = m4_failure(candidate.variants)
        log_loss = calibration = math.nan
        actual = candidate.variants.get(outcome.ACTUAL)
        if actual is None or not isinstance(candidate.models, outcome.OutcomeFoldModels):
            results.append(CandidateResult(M4_MODEL, candidate.family, n_rows, failure_reason=failure))
            continue
        actual = np.asarray(actual, dtype=np.float64)
        if actual.shape == (rows.n_rows, N_CLASSES) and _valid_probabilities(actual):
            losses = outcome.log_loss_table(actual, rows, candidate=candidate.family)
            by_model = dict(zip(losses["model"].astype(str), losses["log_loss"].astype(float)))
            log_loss = by_model[candidate.family]
            baseline[candidate.family] = by_model[outcome.PHASE_BASELINE]
            runs = outcome.expected_runs_table(actual, rows, candidate.models, wicket_buckets)
            calibration = _runs_calibration_error(runs)
            runs.insert(0, "candidate", candidate.family)
            runs["expected_minus_actual"] = runs["mean_expected_bat_runs"] - runs["mean_actual_bat_runs"]
            tables.append(runs[list(M4_CALIBRATION_COLUMNS)])
        results.append(
            CandidateResult(
                M4_MODEL,
                candidate.family,
                n_rows,
                log_loss=log_loss,
                calibration_error=calibration,
                failure_reason=failure,
            )
        )
    return results, tables, baseline


def _comparison_dtypes() -> dict[str, Any]:
    return {
        "model": "str",
        "candidate": "str",
        "n_rows": np.int64,
        "log_loss": np.float64,
        "brier": np.float64,
        "auc": np.float64,
        "calibration_error": np.float64,
        "failure_reason": "str",
        "primary": bool,
        "top10_changes_batter": "Int64",
        "top10_changes_bowler": "Int64",
    }


def candidate_table(
    results: Sequence[CandidateResult], *, preferred: Mapping[str, tuple[str, float]] | None = None
) -> pd.DataFrame:
    """The comparison table of ``results`` with the Primary_Model of each model flagged (criterion 13).

    Returns COMPARISON_COLUMNS, one row per CandidateResult, in the order of MODELS and then the
    model's tie order, with ``primary`` True on the candidate that ``select_primary`` picks for
    each model present and ``top10_changes_batter`` / ``top10_changes_bowler`` empty. ``preferred``
    is passed to ``select_primary`` for the calibration-aware tie-break (criterion 8); None (the
    default) selects purely on the lowest log loss. Raises ValueError for no result, a candidate
    listed twice, or candidates of one model that passed their checks but were scored on different
    numbers of rows (criteria 5 and 6), and the errors of select_primary.
    """
    if isinstance(results, CandidateResult) or not isinstance(results, Sequence) or not results:
        raise ValueError("results must be a non-empty sequence of CandidateResults")
    if not all(isinstance(result, CandidateResult) for result in results):
        raise TypeError("results must hold CandidateResult values")
    keys = [(result.model, result.candidate) for result in results]
    repeated = sorted({key for key in keys if keys.count(key) > 1})
    if repeated:
        raise ValueError(f"each Candidate_Model may appear once; repeated: {repeated}")
    ordered = sorted(results, key=lambda item: (MODELS.index(item.model), TIE_ORDER[item.model].index(item.candidate)))
    for model in MODELS:
        counts = {result.n_rows for result in ordered if result.model == model and not result.failure_reason}
        if len(counts) > 1:
            raise ValueError(
                f"the {model} candidates were scored on different numbers of rows ({sorted(counts)}); Requirement 23 "
                "compares every candidate of a model on the same rows"
            )
    frame = pd.DataFrame.from_records(
        [
            {
                "model": result.model,
                "candidate": result.candidate,
                "n_rows": result.n_rows,
                "log_loss": result.log_loss,
                "brier": result.brier,
                "auc": result.auc,
                "calibration_error": result.calibration_error,
                "failure_reason": result.failure_reason,
                "primary": False,
                "top10_changes_batter": pd.NA,
                "top10_changes_bowler": pd.NA,
            }
            for result in ordered
        ],
        columns=list(COMPARISON_COLUMNS),
    ).astype(_comparison_dtypes())
    model_of = frame["model"].to_numpy(dtype=object)
    candidate_of = frame["candidate"].to_numpy(dtype=object)
    primary = np.zeros(len(frame), dtype=bool)
    for model in MODELS:
        if (model_of == model).any():
            primary |= (model_of == model) & (candidate_of == select_primary(frame, model, preferred=preferred))
    frame["primary"] = primary
    return frame


def calibration_preferences(settings: Any) -> dict[str, tuple[str, float]]:
    """model -> (preferred family, relative log-loss band) from the Config, for ``select_primary``.

    Reads ``model_choice.calibration_preferred`` (a comparison-table model label -> family map)
    and ``model_choice.calibration_band`` (the relative band delta, shared by every preferred
    family) and resolves them into the mapping that drives the calibration-aware tie-break of
    Requirement 23, criterion 8: a named family within ``delta`` x (lowest log loss) of the lowest
    is preferred. An empty ``calibration_preferred`` leaves every model on the plain lowest
    log loss (M3 as delivered). Raises ValueError for a model label that is not one of MODELS or a
    family outside the model's tie order.
    """
    preferred = settings.get(_CALIBRATION_PREFERRED)
    band = float(settings.get(_CALIBRATION_BAND))
    out: dict[str, tuple[str, float]] = {}
    for model, family in dict(preferred).items():
        model, family = str(model), str(family)
        if model not in MODELS:
            raise ValueError(f"{_CALIBRATION_PREFERRED} names {model!r}, not one of {', '.join(MODELS)}")
        if family not in TIE_ORDER[model]:
            raise ValueError(
                f"{_CALIBRATION_PREFERRED} maps {model} to {family!r}, not a family of {', '.join(TIE_ORDER[model])}"
            )
        out[model] = (family, band)
    return out


def select_primary(
    table: pd.DataFrame, model: str, *, preferred: Mapping[str, tuple[str, float]] | None = None
) -> str:
    """The Primary_Model of ``model`` in a comparison table (criteria 7, 8 and 12; module docstring, "Selection").

    ``table`` holds ``model``, ``candidate``, ``log_loss`` and ``failure_reason`` (text; empty,
    None or NA for a candidate that passed). ``best`` is the candidate of ``model`` without a
    failure reason that has the lowest ``log_loss``. A calibration-aware tie-break then applies
    (Requirement 23, criterion 8): when ``preferred`` maps ``model`` to ``(family, band)``, the
    named ``family`` has no failure reason and its ``log_loss`` lies within ``band`` x ``best`` of
    the lowest (a relative log-loss band), that family is returned. Otherwise ``best`` is returned,
    with an exact tie (``math.isclose(..., rel_tol=TIE_REL_TOL, abs_tol=TIE_ABS_TOL)``) broken by
    the first tied candidate in TIE_ORDER[model] (M3: logistic, lgbm, xgb, rf; M4: lgbm, xgb, rf).
    A model with no ``preferred`` entry keeps the plain lowest-log-loss rule, so M3 is unchanged.
    Raises ValueError for an unknown model, a candidate outside the model's tie order or listed
    twice, a candidate without a failure reason whose log loss is not finite, or a malformed
    ``preferred`` band; StageError (exit code 5) when no candidate of the model is left (criterion 9).
    """
    _require(table, ("model", "candidate", "log_loss", "failure_reason"), "table")
    order = TIE_ORDER[_check_model(model)]
    part = table.iloc[np.flatnonzero(table["model"].astype(str).to_numpy(dtype=object) == model)]
    names = [str(name) for name in part["candidate"].tolist()]
    unknown = [name for name in names if name not in order]
    if unknown:
        raise ValueError(f"{model} candidates must be one of {', '.join(order)}; got {unknown}")
    if len(set(names)) != len(names):
        raise ValueError(f"the table lists a {model} candidate more than once: {names}")
    losses = pd.to_numeric(part["log_loss"], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
    failed = [_is_failure(reason) for reason in part["failure_reason"].tolist()]
    kept = [(name, float(loss)) for name, loss, bad in zip(names, losses, failed) if not bad]
    if not kept:
        reasons = "; ".join(f"{name}: {reason}" for name, reason in zip(names, part["failure_reason"].tolist()))
        raise StageError(
            f"no {model} Candidate_Model is left to select as the Primary_Model (Requirement 23, criteria 7, 8 and "
            f"12): {reasons or 'the table lists none'}"
        )
    undefined = [name for name, loss in kept if not math.isfinite(loss)]
    if undefined:
        raise ValueError(f"the {model} candidate(s) {undefined} have no failure reason but no finite log loss")
    best = min(loss for _, loss in kept)
    # Calibration-aware tie-break (criterion 8): prefer the Config-named family when it passed and
    # its log loss is within the relative band of the lowest.
    pref = (preferred or {}).get(model)
    if pref is not None:
        family, band = str(pref[0]), float(pref[1])
        if not math.isfinite(band) or band < 0:
            raise ValueError(
                f"the calibration band for {model} must be a finite, non-negative log-loss fraction; got {pref[1]!r}"
            )
        if family not in order:
            raise ValueError(f"the preferred {model} family {family!r} must be one of {', '.join(order)}")
        preferred_loss = next((loss for name, loss in kept if name == family), None)
        if preferred_loss is not None and preferred_loss - best <= band * abs(best):
            return family
    tied = [name for name, loss in kept if math.isclose(loss, best, rel_tol=TIE_REL_TOL, abs_tol=TIE_ABS_TOL)]
    return min(tied, key=order.index)


def match_clustered_paired_se(
    ll_a: Any, ll_b: Any, match_id: Any
) -> tuple[float, float, int, int]:
    """Mean and match-clustered SE of the per-row log-loss difference ``ll_a - ll_b``.

    ``ll_a`` and ``ll_b`` are per-row log losses of two candidates on the same rows (in the same
    order) and ``match_id`` the match of each row. Returns ``(mean, se, n_rows, n_clusters)``,
    where ``se`` is the CR1 cluster-robust standard error of the sample mean of the difference,
    clusters being matches: ``sqrt([G/(G-1)] * sum_g (sum_{i in g} (d_i - mean))^2) / n``. With no
    row, or a single row, the SE is NaN.
    """
    d = np.asarray(ll_a, dtype=np.float64) - np.asarray(ll_b, dtype=np.float64)
    clusters = np.asarray(match_id)
    if d.shape != clusters.shape or d.ndim != 1:
        raise ValueError("ll_a, ll_b and match_id must be 1-D arrays of the same length")
    n = int(d.size)
    if n == 0:
        return (math.nan, math.nan, 0, 0)
    mean = float(d.mean())
    _, inverse = np.unique(clusters, return_inverse=True)
    g = int(inverse.max()) + 1 if n else 0
    cluster_sums = np.bincount(inverse, weights=d - mean, minlength=g)
    meat = float(np.sum(cluster_sums**2))
    if g > 1 and n > 1:
        var = (g / (g - 1)) * meat / (n * n)
        return (mean, math.sqrt(var), n, g)
    return (mean, math.nan, n, g)


@dataclass(frozen=True)
class M4TieBreak:
    """The M4 calibration-aware tie-break and its supporting paired-SE evidence (criterion 8).

    ``preferred`` is the Config-named family and ``band`` its relative log-loss band; ``best`` is
    the lowest-log-loss non-failed candidate and ``selected`` the Primary_Model. ``applied`` is
    True when the preferred family passed and displaced a different ``best``. ``mean_diff`` and
    ``se`` are the mean and match-clustered SE of the per-Legal_Ball log-loss difference
    ``best - selected`` (XGBoost - LightGBM as delivered), with ``n_rows`` Legal_Balls in
    ``n_clusters`` matches; they are NaN/0 when the tie-break did not apply.
    """

    preferred: str
    band: float
    best: str
    best_loss: float
    selected: str
    selected_loss: float
    applied: bool
    within_band: bool
    mean_diff: float
    se: float
    n_rows: int
    n_clusters: int


def _per_ball_log_loss(actual: np.ndarray, rows: outcome.OutcomeRows) -> np.ndarray:
    """The per-Legal_Ball clipped log loss of an M4 ``actual`` variant over the trainable rows."""
    evaluated = np.flatnonzero(rows.trainable)
    p = np.asarray(actual, dtype=np.float64)[evaluated]
    observed = p[np.arange(len(evaluated)), rows.label[evaluated]]
    clipped = np.clip(observed, common.LOG_LOSS_CLIP, 1.0 - common.LOG_LOSS_CLIP)
    return -np.log(clipped)


def _m4_tiebreak_evidence(
    settings: Any, table: pd.DataFrame, m4_by_family: Mapping[str, "M4Candidate"], rows: outcome.OutcomeRows,
    selected: str,
) -> M4TieBreak | None:
    """The M4 tie-break outcome and the paired-SE evidence, or None when no M4 family is preferred (criterion 8)."""
    pref = calibration_preferences(settings).get(M4_MODEL)
    if pref is None:
        return None
    family, band = pref
    part = table.iloc[np.flatnonzero(table["model"].astype(str).to_numpy(dtype=object) == M4_MODEL)]
    kept = {
        str(c): float(loss)
        for c, loss, reason in zip(part["candidate"], part["log_loss"], part["failure_reason"])
        if not _is_failure(reason)
    }
    if not kept:
        return None
    best = min(kept, key=lambda name: (kept[name], M4_TIE_ORDER.index(name)))
    best_loss = kept[best]
    selected = str(selected)
    selected_loss = kept.get(selected, math.nan)
    applied = selected == family and family in kept and family != best
    within_band = selected in kept and (selected_loss - best_loss) <= band * abs(best_loss)
    mean_diff = se = math.nan
    n_rows = n_clusters = 0
    if applied:
        ll_best = _per_ball_log_loss(np.asarray(m4_by_family[best].variants[outcome.ACTUAL]), rows)
        ll_selected = _per_ball_log_loss(np.asarray(m4_by_family[selected].variants[outcome.ACTUAL]), rows)
        match = rows.match_id[np.flatnonzero(rows.trainable)]
        mean_diff, se, n_rows, n_clusters = match_clustered_paired_se(ll_best, ll_selected, match)
    return M4TieBreak(
        family, band, best, best_loss, selected, selected_loss, applied, within_band, mean_diff, se, n_rows, n_clusters
    )


def primary_models(table: pd.DataFrame) -> dict[str, str]:
    """model -> the candidate whose ``primary`` flag is True, for each model of MODELS, from a comparison table.

    Raises ValueError unless every model has exactly one flagged candidate.
    """
    _require(table, ("model", "candidate", "primary"), "table")
    model_of = table["model"].astype(str).to_numpy(dtype=object)
    flagged = _flags(table["primary"], "table column 'primary'")
    chosen: dict[str, str] = {}
    for model in MODELS:
        names = table["candidate"].astype(str).to_numpy(dtype=object)[flagged & (model_of == model)].tolist()
        if len(names) != 1:
            raise ValueError(f"the comparison table must flag exactly one {model} Primary_Model; it flags {names}")
        chosen[model] = str(names[0])
    return chosen


# ===================================================================== reading the saved predictions

@dataclass(frozen=True, eq=False)
class PrimaryPredictions:
    """The predictions of one set of M3 and M4 models acting as Primary_Models (module docstring, "Reading").

    Every row of the ``state`` table in ``state`` order: ``row``, ``match_id``, ``innings``,
    ``fold`` (int64); ``wp_pre`` (n,) and ``wp_post`` (n, 9; classes in CLASSES order), float64
    within [0, 1]; ``m4``: variant -> float64 (n, 9) probabilities of the M4 model; ``candidates``:
    model (M3-1, M3-2, M4) -> the candidate whose predictions these are. Arrays are read-only.
    """

    candidates: Mapping[str, str]
    row: np.ndarray
    match_id: np.ndarray
    innings: np.ndarray
    fold: np.ndarray
    wp_pre: np.ndarray
    wp_post: np.ndarray
    m4: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        n_rows = len(self.row)
        for name in ("row", "match_id", "innings", "fold", "wp_pre"):
            if getattr(self, name).shape != (n_rows,):
                raise ValueError(f"PrimaryPredictions.{name} must hold one value per row")
        if self.wp_post.shape != (n_rows, N_CLASSES):
            raise ValueError(f"PrimaryPredictions.wp_post must have shape ({n_rows:,}, {N_CLASSES})")
        if not _valid_probabilities(self.wp_pre) or not _valid_probabilities(self.wp_post):
            raise ValueError("PrimaryPredictions WP values must be finite and within [0, 1]")
        for variant, values in self.m4.items():
            if values.shape != (n_rows, N_CLASSES):
                raise ValueError(f"PrimaryPredictions.m4[{variant!r}] must have shape ({n_rows:,}, {N_CLASSES})")
            _read_only(values)
        for name in ("row", "match_id", "innings", "fold", "wp_pre", "wp_post"):
            _read_only(getattr(self, name))
        object.__setattr__(self, "candidates", MappingProxyType(dict(self.candidates)))
        object.__setattr__(self, "m4", MappingProxyType(dict(self.m4)))


def _source(sources: pd.DataFrame, model: str, candidate: str) -> tuple[str, str]:
    model_of = sources["model"].astype(str).to_numpy(dtype=object)
    candidate_of = sources["candidate"].astype(str).to_numpy(dtype=object)
    at = np.flatnonzero((model_of == model) & (candidate_of == candidate))
    if len(at) != 1:
        raise ValueError(f"the {SOURCES_TABLE} table names no saved predictions of {model} candidate {candidate!r}")
    return str(sources["stage"].iloc[at[0]]), str(sources["table"].iloc[at[0]])


def load_predictions(
    store: CheckpointStore, *, m3: Mapping[int, str], m4: str, variants: Sequence[str] = VARIANT_NAMES
) -> PrimaryPredictions:
    """The saved predictions of M3 candidates ``m3`` ({1: family, 2: family}) and M4 candidate ``m4`` (criterion 9).

    Reads ROWS_TABLE, COMPARISON_TABLE and SOURCES_TABLE of the ``model_choice`` checkpoint, the
    WP table of each M3 candidate and the M4 candidate's variant table from the checkpoint that
    SOURCES_TABLE names. Innings-1 rows take the M3-1 candidate's WP and innings-2 rows the M3-2
    candidate's, each table independent of the other innings' candidate, so a rerun that swaps
    either M3 model reads only stored arrays. ``variants`` names the M4 variants to load. Returns a
    PrimaryPredictions. Raises ValueError for a candidate that the comparison table does not list
    or marks as failed, and tables that do not hold the same rows; StageError for a checkpoint
    file that is missing or changed.
    """
    if not isinstance(m3, Mapping) or sorted(m3) != list(M3_INNINGS):
        raise ValueError(f"m3 must map each innings {list(M3_INNINGS)} to an M3 candidate family")
    wanted = {M3_MODELS[innings]: str(m3[innings]) for innings in M3_INNINGS}
    wanted[M4_MODEL] = str(m4)
    table = store.load_table(STAGE_NAME, COMPARISON_TABLE)
    model_of = table["model"].astype(str).to_numpy(dtype=object)
    candidate_of = table["candidate"].astype(str).to_numpy(dtype=object)
    for model, candidate in wanted.items():
        at = np.flatnonzero((model_of == model) & (candidate_of == candidate))
        if len(at) != 1:
            raise ValueError(f"the comparison table lists no {model} candidate {candidate!r}")
        if _is_failure(table["failure_reason"].iloc[at[0]]):
            raise ValueError(
                f"{model} candidate {candidate!r} failed ({table['failure_reason'].iloc[at[0]]}); it is excluded from "
                "selection and reruns (Requirement 23, criterion 12)"
            )
    rows = store.load_table(STAGE_NAME, ROWS_TABLE)
    sources = store.load_table(STAGE_NAME, SOURCES_TABLE)
    row = _ints(rows["row"], f"{ROWS_TABLE} column 'row'")
    innings = _ints(rows["innings"], f"{ROWS_TABLE} column 'innings'")
    n_rows = len(row)
    wp_pre = np.full(n_rows, np.nan)
    wp_post = np.full((n_rows, N_CLASSES), np.nan)
    for number in M3_INNINGS:
        model = M3_MODELS[number]
        stage, name = _source(sources, model, wanted[model])
        frame = store.load_table(stage, name)
        at = np.flatnonzero(innings == number)
        _require(frame, WP_COLUMNS, f"checkpoint table {name!r}")
        if len(frame) != len(at) or not np.array_equal(_ints(frame["row"], f"{name} column 'row'"), row[at]):
            raise ValueError(
                f"checkpoint table {name!r} does not hold the innings-{number} rows of {ROWS_TABLE} in order"
            )
        wp_pre[at] = _floats(frame["wp_pre"], f"{name} column 'wp_pre'")
        wp_post[at] = np.column_stack(
            [_floats(frame[column], f"{name} column {column!r}") for column in WP_POST_COLUMNS]
        )
    stage, name = _source(sources, M4_MODEL, wanted[M4_MODEL])
    m4_table = store.load_table(stage, name)
    _same_rows(m4_table, row, f"checkpoint table {name!r}")
    return PrimaryPredictions(
        candidates=wanted,
        row=row,
        match_id=_ints(rows["match_id"], f"{ROWS_TABLE} column 'match_id'"),
        innings=innings,
        fold=_ints(rows["fold"], f"{ROWS_TABLE} column 'fold'"),
        wp_pre=wp_pre,
        wp_post=wp_post,
        m4=outcome.variant_arrays(m4_table, variants),
    )


def load_primary(store: CheckpointStore, *, variants: Sequence[str] = VARIANT_NAMES) -> PrimaryPredictions:
    """The predictions of the selected Primary_Models only (Requirement 23, criterion 9; module docstring, "Reading").

    The entry point for every later stage: reads the ``primary`` flags of COMPARISON_TABLE
    (``primary_models``) and returns ``load_predictions`` of those candidates, with the M4
    ``variants`` asked for (all six by default; LI needs ``outcome.AVG_BOTH``).
    """
    chosen = primary_models(store.load_table(STAGE_NAME, COMPARISON_TABLE))
    m3 = {innings: chosen[M3_MODELS[innings]] for innings in M3_INNINGS}
    return load_predictions(store, m3=m3, m4=chosen[M4_MODEL], variants=variants)


def load_m4_models(store: CheckpointStore, settings: Any, family: str) -> outcome.OutcomeFoldModels:
    """The outer fold models of the non-failed M4 candidate ``family``, for predicting other feature values.

    Reads SOURCES_TABLE of the ``model_choice`` checkpoint, then ``outcome.load_fold_models`` from
    the checkpoint it names: ``outcome`` for LightGBM, STAGE_NAME for an Alternative_Model, whose
    fold models and ``m4_<family>_fold_models`` table this stage saved. The models predict with
    ``outcome.candidate_family(family, settings)``. Raises ValueError for a candidate without saved
    predictions (a failed or unknown one) and StageError for a checkpoint file that is missing or
    changed.
    """
    sources = store.load_table(STAGE_NAME, SOURCES_TABLE)
    _require(sources, SOURCES_COLUMNS, f"checkpoint table {SOURCES_TABLE!r}")
    stage, _ = _source(sources, M4_MODEL, str(family))
    return outcome.load_fold_models(store, stage, outcome.candidate_family(str(family), settings))


# =================================================================================== output files

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


def _save_figure(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        figure.savefig(temporary, format="png", dpi=CHART_DPI, metadata={"Software": None})
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_calibration_chart(path: Path, innings: int, curves: pd.DataFrame, table: pd.DataFrame) -> None:
    """The Calibration_Curves of every M3 candidate of ``innings``: one panel per subset, every candidate in each."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from superr.models import win_prob

    model = M3_MODELS[innings]
    part = table.iloc[np.flatnonzero(table["model"].astype(str).to_numpy(dtype=object) == model)]
    primary = {str(name): bool(flag) for name, flag in zip(part["candidate"], part["primary"])}
    losses = {str(name): float(value) for name, value in zip(part["candidate"], part["log_loss"])}
    n_rows = int(part["n_rows"].iloc[0]) if len(part) else 0
    figure = Figure(figsize=_CURVE_CHART_SIZE, dpi=CHART_DPI, layout="constrained")
    FigureCanvasAgg(figure)
    axes = list(np.atleast_1d(figure.subplots(1, len(_SUBSETS), sharex=True, sharey=True)))
    phase = curves["phase"].astype(str).to_numpy(dtype=object)
    family = curves["candidate"].astype(str).to_numpy(dtype=object)
    ticks = np.linspace(0.0, 1.0, 6)
    for ax, subset in zip(axes, _SUBSETS):
        ax.plot([0.0, 1.0], [0.0, 1.0], color="#9a9a9a", linestyle="--", linewidth=1.0, label="Perfect calibration")
        shown = 0
        for name in _ordered(set(family.tolist()), M3_TIE_ORDER):
            rows = curves.iloc[np.flatnonzero((phase == subset) & (family == name))]
            if rows.empty:
                continue
            filled = rows["n_deliveries"].to_numpy() > 0
            shown = int(rows["n_deliveries"].sum())
            error = win_prob.calibration_error(rows)
            mark = ", Primary_Model" if primary.get(name) else ""
            label = f"{_FAMILY_LABELS.get(name, name)}{mark} (Calibration_Error {error:.3f})"
            ax.plot(
                rows["mean_wp"].to_numpy()[filled],
                rows["observed_win_share"].to_numpy()[filled],
                marker="o",
                markersize=4,
                linewidth=2.0 if primary.get(name) else 1.2,
                color=_FAMILY_COLOURS.get(name, "#333333"),
                label=label,
            )
        title = "All Phases" if subset == _ALL_PHASES else subset.capitalize()
        ax.set_title(f"{title} ({shown:,} Deliveries)", fontsize=10)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_aspect("equal")
        ax.grid(True, color="#e6e6e6", linewidth=0.6)
        ax.set_xlabel("Mean calibrated WP in the bin")
        ax.legend(loc="upper left", fontsize=6.5, frameon=False)
    axes[0].set_ylabel("Observed share won by the batting side")
    pooled = ", ".join(
        f"{_FAMILY_LABELS.get(name, name)} {losses[name]:.4f}{' (Primary_Model)' if primary.get(name) else ''}"
        for name in _ordered(losses, M3_TIE_ORDER)
        if math.isfinite(losses[name])
    )
    figure.suptitle(
        f"{model}: Calibration_Curves of the Candidate_Models on the same {n_rows:,} eligible Deliveries of innings "
        f"{innings} (calibrated Out_Of_Fold WP before the terminal rules; 10 equal-width bins)\n"
        f"Pooled_Held_Out_Log_Loss: {pooled}",
        fontsize=11,
    )
    _save_figure(figure, path)


def _wp_curve(
    match_id: int,
    rows: outcome.OutcomeRows,
    balls: pd.DataFrame,
    state: pd.DataFrame,
    evaluations: Mapping[int, M3Evaluation],
    rules: Mapping[int, np.ndarray],
) -> pd.DataFrame:
    """The WP_CURVE_COLUMNS of every innings 1-2 Delivery of ``match_id``, in ball order (``state`` order)."""
    mine = np.flatnonzero(rows.match_id == match_id)
    index = pd.Index(_ints(balls["row"], "balls column 'row'"))
    at = index.get_indexer(rows.row[mine])
    part = balls.iloc[at].reset_index(drop=True)
    innings = rows.innings[mine]
    wp = np.full(len(mine), np.nan)
    rule = np.full(len(mine), "", dtype=object)
    primary = np.full(len(mine), "", dtype=object)
    for number, evaluation in evaluations.items():
        positions = np.flatnonzero(rows.innings == number)  # the innings' Deliveries, in the evaluation's order
        local = np.searchsorted(positions, mine[innings == number])
        wp[innings == number] = evaluation.wp_pre[local]
        rule[innings == number] = rules[number][local]
        primary[innings == number] = evaluation.family
    first = np.where(innings == 1, wp, 1.0 - wp)
    return pd.DataFrame(
        {
            "delivery": np.arange(1, len(mine) + 1, dtype=np.int64),
            "row": rows.row[mine],
            "match_id": rows.match_id[mine],
            "innings": innings,
            "innings_ball_num": _ints(part["innings_ball_num"], "balls column 'innings_ball_num'"),
            "overid": _ints(part["overid"], "balls column 'overid'"),
            "over_ball_num": _ints(part["over_ball_num"], "balls column 'over_ball_num'"),
            "batting_team_id": _ints(part["batting_team_id"], "balls column 'batting_team_id'"),
            "lb_bowled": _ints(state["lb_bowled"].iloc[mine], "state column 'lb_bowled'"),
            "score": _ints(state["score"].iloc[mine], "state column 'score'"),
            "wickets": _ints(state["wickets"].iloc[mine], "state column 'wickets'"),
            "runs_needed": _floats(state["runs_needed"].iloc[mine], "state column 'runs_needed'"),
            "wicket": _flags(part["counted_wicket"], "balls column 'counted_wicket'"),
            "terminal_rule": pd.array(rule.astype(str), dtype="str"),
            "m3_primary_model": pd.array(primary.astype(str), dtype="str"),
            "wp_batting_side": wp,
            "wp_batting_first": first,
        },
        columns=list(WP_CURVE_COLUMNS),
    )


def _team_names(matches: pd.DataFrame, match_id: int) -> dict[int, str]:
    _require(matches, ("match_id", "home_team_id", "home_team", "away_team_id", "away_team"), "matches")
    row = matches.iloc[np.flatnonzero(_ints(matches["match_id"], "matches column 'match_id'") == match_id)]
    if row.empty:
        return {}
    first = row.iloc[0]
    return {int(first["home_team_id"]): str(first["home_team"]), int(first["away_team_id"]): str(first["away_team"])}


def _write_wp_curve_chart(path: Path, curve: pd.DataFrame, matches: pd.DataFrame, match_id: int) -> None:
    """The WP curve of one match as a PNG: the WP of the side that batted first before each Delivery (10.18)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    names = _team_names(matches, match_id)
    innings = curve["innings"].to_numpy(dtype=np.int64)
    team = curve["batting_team_id"].to_numpy(dtype=np.int64)

    def side(number: int) -> str:
        if not (innings == number).any():
            return "?"
        team_id = int(team[innings == number][0])
        return names.get(team_id, f"team {team_id}")

    first_team, second_team = side(1), side(2)
    models = {int(number): str(name) for number, name in zip(innings, curve["m3_primary_model"].astype(str))}
    x = curve["delivery"].to_numpy(dtype=np.int64)
    y = curve["wp_batting_first"].to_numpy(dtype=np.float64)
    figure = Figure(figsize=_WP_CHART_SIZE, dpi=CHART_DPI, layout="constrained")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    colours = {1: "#1f5fa8", 2: "#d95f02"}
    for number in M3_INNINGS:
        part = innings == number
        if not part.any():
            continue
        batting = first_team if number == 1 else second_team
        quantity = "WP" if number == 1 else "1 - WP"
        ax.plot(
            x[part], y[part], color=colours[number], linewidth=1.6,
            label=f"Innings {number}: {batting} batting ({quantity}; M3-{number} Primary_Model "
            f"{_FAMILY_LABELS.get(models[number], models[number])})",
        )  # fmt: skip
    wicket = curve["wicket"].to_numpy(dtype=bool)
    if wicket.any():
        ax.scatter(
            x[wicket], y[wicket], marker="v", color="#b2182b", s=24, zorder=3, label="Delivery with a Counted_Wicket"
        )
    if (innings == 2).any() and (innings == 1).any():
        ax.axvline(x[innings == 2][0] - 0.5, color="#555555", linestyle="--", linewidth=1.0)
    ax.axhline(0.5, color="#9a9a9a", linestyle=":", linewidth=1.0)
    overs = curve["overid"].to_numpy(dtype=np.int64)
    starts = np.flatnonzero(
        np.r_[True, (overs[1:] != overs[:-1]) | (innings[1:] != innings[:-1])] & np.isin(overs, _WP_TICK_OVERS)
    )
    ax.set_xticks(x[starts])
    ax.set_xticklabels([f"Inn {innings[i]}\nover {overs[i]}" for i in starts], fontsize=7)
    ax.set_xlim(float(x.min()) - 1.0, float(x.max()) + 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, color="#e6e6e6", linewidth=0.6)
    ax.set_xlabel("Delivery, in ball order (innings 1, then innings 2)")
    ax.set_ylabel(f"Win probability of {first_team}")
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.set_title(
        f"Match {match_id}: win probability of {first_team} (batting first) before each Delivery, against "
        f"{second_team}",
        fontsize=11,
    )
    _save_figure(figure, path)


# ===================================================================== the model_choice Pipeline_Stage

def _rule_counts(
    win_prob: Any, evaluations: Mapping[str, M3Evaluation], post: PostStates, innings: int
) -> list[dict[str, Any]]:
    """The POST_RULES_COLUMNS records of one innings, the rules checked across candidates and against ``post``.

    The rules that set a WP (``win_prob.DECIDING_RULES``) must flag exactly the states that
    ``post.terminal`` marks, and in innings 1 INNINGS_END exactly the states with ``innings_over``,
    which the innings-1 classifier scores (Requirement 10, criterion 17).
    """
    passed = [evaluation for evaluation in evaluations.values() if evaluation.post_rule is not None]
    if not passed:
        raise StageError(f"no M3 candidate of innings {innings} could predict its Post_Outcome_States")
    rule = passed[0].post_rule
    for evaluation in passed[1:]:
        if not np.array_equal(evaluation.post_rule, rule):
            raise ValueError(
                f"the M3 candidates of innings {innings} gave Post_Outcome_States different terminal rules"
            )
    flat = rule.reshape(-1)
    decided = np.isin(flat, win_prob.DECIDING_RULES)
    if not np.array_equal(decided, post.terminal(innings)):
        raise ValueError(
            f"{int((decided != post.terminal(innings)).sum()):,} innings-{innings} Post_Outcome_States are decided "
            "by a WP rule but not terminal under PostStates.terminal, or the reverse"
        )
    if innings == 1 and not np.array_equal(flat == win_prob.INNINGS_END, post.innings_over(1)):
        raise ValueError(
            f"{int(((flat == win_prob.INNINGS_END) != post.innings_over(1)).sum()):,} innings-1 Post_Outcome_States "
            "end the innings under the WP rules but not under PostStates.innings_over, or the reverse"
        )
    return [
        {"innings": innings, "rule": win_prob.TERMINAL_RULES[code], "n_states": int((flat == code).sum())}
        for code in sorted(win_prob.TERMINAL_RULES)
        if code == win_prob.NOT_TERMINAL or (flat == code).any() or code in _innings_rules(win_prob, innings)
    ]


def _innings_rules(win_prob: Any, innings: int) -> tuple[int, ...]:
    """The rules that can flag a state of ``innings``: INNINGS_END in innings 1, the deciding rules in innings 2."""
    return (win_prob.INNINGS_END,) if innings == 1 else tuple(win_prob.DECIDING_RULES)


def run_stage(ctx: StageContext) -> None:
    """The ``model_choice`` Pipeline_Stage (module docstring, "The model_choice Pipeline_Stage").

    Raises FitLeakageError or ViewLeakageError (exit code 4) for a fit or a view that breaks the
    leakage guard, DataError (exit code 3) for a match that is not in the Fold_Table, StageError
    (exit code 5) when every candidate of a model failed or a checkpoint file is missing, and
    ValueError, TypeError or KeyError for input tables that do not fit together or a malformed or
    missing Config value. A failed Alternative_Model fit is recorded, not raised (criterion 12).
    """
    from superr import pressure
    from superr.models import remaining_runs, win_prob

    started = time.perf_counter()
    settings, store, registry, log = ctx.settings, ctx.store, ctx.registry, ctx.log
    n_folds = settings.get(_N_FOLDS)
    plan = folds.sandbox_plan(n_folds)  # checks D15 before any table is read
    ks = sorted(spec.sandbox_k for spec in plan if spec.kind == folds.OUTER)
    wicket_buckets = settings.get(_WICKET_BUCKETS)
    outcome.wickets_bucket_codes(np.zeros(0, dtype=np.int64), wicket_buckets)  # checks D17 now
    m4_family = outcome.lgbm_family(settings)
    if (win_prob.LOGISTIC, win_prob.LGBM) != (LOGISTIC, LGBM) or not set(win_prob.FAMILIES) <= set(M3_TIE_ORDER):
        raise ValueError(f"the win_prob families {win_prob.FAMILIES} must be names of M3_TIE_ORDER {M3_TIE_ORDER}")
    if tuple(win_prob.ALTERNATIVE_FAMILIES) != M3_ALTERNATIVES:
        raise ValueError(
            f"the win_prob Alternative_Model families {win_prob.ALTERNATIVE_FAMILIES} must be {M3_ALTERNATIVES}"
        )
    win_prob.calibration_bounds(settings)  # checks the bounds before any table is read; fit_innings reads them again
    alternatives = {name: win_prob.candidate_family(name, settings) for name in M3_ALTERNATIVES}  # checks their settings
    if tuple(outcome.ALTERNATIVE_FAMILIES) != M4_ALTERNATIVES or m4_family.name != M4_TIE_ORDER[0]:
        raise ValueError(
            f"the outcome families {outcome.FAMILIES} must be {M4_TIE_ORDER}, the M4 tie order of Requirement 23, "
            "criterion 8"
        )
    m4_alternatives = {name: outcome.candidate_family(name, settings) for name in M4_ALTERNATIVES}  # checks them

    balls = store.load_table(reconstruct.STAGE_NAME, reconstruct.BALLS_TABLE)
    state = store.load_table(STATE_STAGE, STATE_TABLE)
    fold_table = store.load_table(folds.STAGE_NAME, folds.FOLD_TABLE)
    matches = store.load_table(_LOAD_STAGE, _MATCHES_TABLE)
    labels = store.load_table(win_prob.STAGE_NAME, win_prob.LABELS_TABLE)

    # 1. Every innings 1-2 Delivery in state order, its labels and the views checked before use.
    rows = outcome.outcome_rows(balls, state, fold_table)
    _same_rows(labels, rows.row, f"the {win_prob.STAGE_NAME} {win_prob.LABELS_TABLE} table")
    label = _ints(labels["label"], "m3_labels column 'label'")
    eligible = _flags(labels["eligible"], "m3_labels column 'eligible'")
    began = time.perf_counter()
    view_by_k: dict[int, pd.DataFrame] = {}
    for k in ks:
        view = store.load_table(remaining_runs.STAGE_NAME, views.view_table(k))
        views.check_view(k, view, rows.fold, registry)
        view_by_k[k] = view

    # 2. The Post_Outcome_States, built once and shared by every candidate (DD-8). The end-of-innings
    #    states of innings 1 take Par_Score at Scheduled_Balls from the first-innings M2 fits' par tables.
    crv = outcome.load_class_run_values(store, outcome.STAGE_NAME)
    m2_first = pressure.first_innings_models(store, n_folds)
    post = pressure.post_outcome_states(
        state, balls, view_by_k, rows.fold, crv, m2_first,
        par_means=store.load_table(remaining_runs.STAGE_NAME, remaining_runs.PAR_MEANS_TABLE),
        par_bias=store.load_table(remaining_runs.STAGE_NAME, remaining_runs.PAR_BIAS_TABLE),
    )  # fmt: skip
    del m2_first
    post_seconds = time.perf_counter() - began

    # 3. The M3 Alternative_Models (criteria 1-4 and 12): XGBoost and the random forest of each
    #    innings, each through the fold sandbox and centred isotonic calibrators of win_prob.fit_innings,
    #    every fit registered with the FitRegistry before it runs. A failure is recorded and the stage goes on.
    fits: list[AlternativeFit] = []
    for number in M3_INNINGS:
        for name in M3_ALTERNATIVES:
            fit = fit_alternative(number, alternatives[name], view_by_k, labels, settings, registry, state=state)
            _log_alternative(ctx, fit)
            fits.append(fit)

    # 3b. The M4 Alternative_Models (criteria 1, 2, 4 and 12): XGBoost and the random forest, D15 outer
    #     fits each on the LightGBM M4's Training_Rows, labels, features (built from the same checked
    #     views) and folds, with its Average_Skill per fold; each fit predicts the six variants of every
    #     innings 1-2 Delivery of its fold. A failure is recorded and the stage goes on.
    m4_lgbm_models = outcome.load_fold_models(store, outcome.STAGE_NAME, m4_family)
    m4_X = {k: outcome.m4_features(view_by_k[k], state, rows) for k in ks}
    m4_skills = _m4_average_skills(m4_lgbm_models, view_by_k, rows.fold)
    del view_by_k
    m4_fits: list[M4AlternativeFit] = []
    for name in M4_ALTERNATIVES:
        m4_fit = fit_m4_alternative(registry, plan, m4_alternatives[name], rows, m4_X, m4_skills, log)
        _log_m4_alternative(ctx, m4_fit)
        m4_fits.append(m4_fit)
    del m4_X

    # 4. The candidates: M3 from the win_prob checkpoint and from step 3, M4 LightGBM from the outcome
    #    checkpoint and the M4 Alternative_Models from step 3b.
    m3 = [
        M3Candidate(
            family=name,
            models=win_prob.load_wp_models(store, name, n_folds),
            pre={
                number: store.load_table(win_prob.STAGE_NAME, win_prob.wp_table(number, name)) for number in M3_INNINGS
            },
            stage=win_prob.STAGE_NAME,
        )
        for name in win_prob.FAMILIES
    ]
    m3 += [alternative_candidate(name, fits) for name in M3_ALTERNATIVES]
    m4_table = store.load_table(outcome.STAGE_NAME, outcome.predictions_table(m4_family.name))
    _same_rows(m4_table, rows.row, f"the {outcome.STAGE_NAME} {outcome.predictions_table(m4_family.name)} table")
    m4 = [
        M4Candidate(
            family=m4_family.name,
            variants=outcome.variant_arrays(m4_table),
            models=m4_lgbm_models,
            stage=outcome.STAGE_NAME,
        )
    ]
    m4 += [m4_fit.candidate for m4_fit in m4_fits]
    del m4_table

    # 5. Each innings with its own candidates only (Requirement 10, criterion 17), then M4. Every M3
    #    candidate of an innings is scored on the same eligible Deliveries (criterion 5).
    by_family = {candidate.family: candidate for candidate in m3}
    evaluations: dict[int, dict[str, M3Evaluation]] = {}
    results: list[CandidateResult] = []
    curves: list[pd.DataFrame] = []
    rule_records: list[dict[str, Any]] = []
    for number in M3_INNINGS:
        evaluations[number] = precompute_predictions(number, m3, post)
        mine = rows.innings == number
        found, drawn = _m3_results(evaluations[number], number, label[mine], eligible[mine], rows.phase[mine])
        results += found
        curves += drawn
        rule_records += _rule_counts(win_prob, evaluations[number], post, number)
    found, m4_tables, m4_baseline = _m4_results(m4, rows, wicket_buckets)
    results += found

    # 6. The comparison table and the Primary_Models (criteria 7, 8 and 13). The calibration-aware
    #    tie-break prefers the Config family when it passed within the log-loss band (criterion 8);
    #    the paired-SE evidence supports it and the Value_Engine re-verifies it (criterion 14).
    table = candidate_table(results, preferred=calibration_preferences(settings))
    primary = primary_models(table)
    m4_tiebreak = _m4_tiebreak_evidence(settings, table, {candidate.family: candidate for candidate in m4}, rows,
                                        primary[M4_MODEL])  # fmt: skip

    # 7. Checkpoint tables and the fold models of the Alternative_Models that passed.
    saved: list[str] = []

    def save(name: str, frame: pd.DataFrame) -> None:
        store.save_table(STAGE_NAME, name, frame)
        saved.append(name)

    rows_frame = pd.DataFrame(
        {"row": rows.row, "match_id": rows.match_id, "innings": rows.innings, "fold": rows.fold},
        columns=list(ROWS_COLUMNS),
    )
    save(ROWS_TABLE, rows_frame)
    sources: list[dict[str, str]] = []
    ranges: dict[tuple[int, str], tuple[float, float]] = {}
    for number in M3_INNINGS:
        for family, evaluation in evaluations[number].items():
            if evaluation.failure:
                continue
            frame = _wp_frame(evaluation, by_family[family].pre[number])
            values = frame[["wp_pre", *WP_POST_COLUMNS]].to_numpy(dtype=np.float64)
            if not _valid_probabilities(values):
                raise ValueError(
                    f"the precomputed WP of innings {number} {family} holds a value outside [0, 1] or missing"
                )
            ranges[(number, family)] = (float(values.min()), float(values.max()))
            name = wp_table(number, family)
            save(name, frame)
            sources.append({"model": M3_MODELS[number], "candidate": family, "stage": STAGE_NAME, "table": name})
    m4_failed = {
        str(name)
        for name, model, reason in zip(table["candidate"], table["model"], table["failure_reason"])
        if str(model) == M4_MODEL and _is_failure(reason)
    }
    n_m4_models = 0
    for candidate in m4:
        if candidate.family in m4_failed:
            continue
        name = outcome.predictions_table(candidate.family)
        if candidate.stage == STAGE_NAME:  # an Alternative_Model fitted here (task 26.2): variants and fold models
            save(name, outcome.variant_table(candidate.variants, rows.row))
            outcome.save_fold_models(store, STAGE_NAME, candidate.models)  # each model under its fit id
            saved.append(outcome.fold_models_table(candidate.family))
            n_m4_models += len(candidate.models.folds)
        sources.append({"model": M4_MODEL, "candidate": candidate.family, "stage": candidate.stage, "table": name})
    save(SOURCES_TABLE, pd.DataFrame.from_records(sources, columns=list(SOURCES_COLUMNS)).astype("str"))
    save(COMPARISON_TABLE, table)
    curve_table = pd.concat(curves, ignore_index=True).astype({"candidate": "str", "phase": "str"})
    save(M3_CURVES_TABLE, curve_table)
    m4_calibration = (
        pd.concat(m4_tables, ignore_index=True).astype({"candidate": "str"})
        if m4_tables
        else pd.DataFrame(columns=list(M4_CALIBRATION_COLUMNS))
    )
    save(M4_CALIBRATION_TABLE, m4_calibration)
    rules_table = pd.DataFrame.from_records(rule_records, columns=list(POST_RULES_COLUMNS)).astype({"rule": "str"})
    save(POST_RULES_TABLE, rules_table)
    n_models = 0
    for fit in fits:  # outer models and calibrators under their fit ids (win_prob.load_wp_models reads them)
        if not fit.failure and not evaluations[fit.innings][fit.family].failure:
            n_models += len(win_prob.save_fold_models(store, STAGE_NAME, fit.fold_models, inner=False))
    completed = [fit.fold_models for fit in fits if fit.fold_models is not None]
    fit_table = (
        pd.concat([fm.fit_table for fm in completed], ignore_index=True)
        if completed
        else pd.DataFrame(columns=list(win_prob.FIT_TABLE_COLUMNS))
    )
    alternative_metrics = (
        pd.concat([fm.fold_metrics for fm in completed], ignore_index=True)
        if completed
        else pd.DataFrame(columns=list(win_prob.FOLD_METRICS_COLUMNS))
    )
    save(ALTERNATIVE_FITS_TABLE, fit_table)
    save(ALTERNATIVE_FOLD_METRICS_TABLE, alternative_metrics)
    n_saved = registry.save(store, STAGE_NAME)
    ctx.constants.record(PRIMARY_CONSTANT, dict(primary), _COMPONENT)

    # 8. Output files: the comparison table, the calibration curves, the Alternative_Models' fold
    #    metrics and the WP curve of match 40767.
    outputs = settings.path("outputs")
    folder = outputs / MODEL_CHOICE_FOLDER
    files: list[Path] = [folder / COMPARISON_FILE]
    _write_csv(folder / COMPARISON_FILE, table)
    innings_of_curve = curve_table["innings"].to_numpy(dtype=np.int64)
    for number in M3_INNINGS:
        part = curve_table.iloc[np.flatnonzero(innings_of_curve == number)].reset_index(drop=True)
        _write_csv(folder / calibration_file(number), part)
        _write_calibration_chart(folder / calibration_chart_file(number), number, part, table)
        files += [folder / calibration_file(number), folder / calibration_chart_file(number)]
    _write_csv(folder / M4_CALIBRATION_FILE, m4_calibration)
    files.append(folder / M4_CALIBRATION_FILE)
    _write_csv(folder / ALTERNATIVE_FOLD_METRICS_FILE, alternative_metrics)
    files.append(folder / ALTERNATIVE_FOLD_METRICS_FILE)
    m3_primary = {number: primary[M3_MODELS[number]] for number in M3_INNINGS}
    pre_rules = {
        number: by_family[m3_primary[number]].pre[number]["terminal_rule"].astype(str).to_numpy(dtype=object)
        for number in M3_INNINGS
    }
    curve = None
    if (rows.match_id == REPORT_MATCH_ID).any():
        curve = _wp_curve(
            REPORT_MATCH_ID,
            rows,
            balls,
            state,
            {number: evaluations[number][m3_primary[number]] for number in M3_INNINGS},
            pre_rules,
        )
        _write_csv(outputs / M3_FOLDER / WP_CURVE_FILE, curve)
        _write_wp_curve_chart(outputs / M3_FOLDER / WP_CURVE_CHART_FILE, curve, matches, REPORT_MATCH_ID)
        files += [outputs / M3_FOLDER / WP_CURVE_FILE, outputs / M3_FOLDER / WP_CURVE_CHART_FILE]
    else:
        log.warning(
            f"match {REPORT_MATCH_ID} has no innings 1-2 Delivery, so its WP curve (Requirement 10, criterion 18) is "
            "not written"
        )

    _log_stage(
        ctx,
        post=post,
        post_seconds=post_seconds,
        rule_records=rule_records,
        evaluations=evaluations,
        table=table,
        primary=primary,
        m4_baseline=m4_baseline,
        ranges=ranges,
        curve=curve,
        files=files,
        saved=saved,
        n_saved=n_saved,
        fits=fits,
        n_models=n_models,
        m4_fits=m4_fits,
        n_m4_models=n_m4_models,
        m4_tiebreak=m4_tiebreak,
        seconds=time.perf_counter() - started,
    )


def _log_stage(
    ctx: StageContext,
    *,
    post: PostStates,
    post_seconds: float,
    rule_records: Sequence[Mapping[str, Any]],
    evaluations: Mapping[int, Mapping[str, M3Evaluation]],
    table: pd.DataFrame,
    primary: Mapping[str, str],
    m4_baseline: Mapping[str, float],
    ranges: Mapping[tuple[int, str], tuple[float, float]],
    curve: pd.DataFrame | None,
    files: Sequence[Path],
    saved: Sequence[str],
    n_saved: int,
    fits: Sequence[AlternativeFit],
    n_models: int,
    m4_fits: Sequence[M4AlternativeFit],
    n_m4_models: int,
    m4_tiebreak: M4TieBreak | None,
    seconds: float,
) -> None:
    log = ctx.log
    def innings_counts(number: int) -> str:
        found = [record for record in rule_records if record["innings"] == number]
        return ", ".join(f"{record['rule']} {record['n_states']:,}" for record in found)

    counts = "; ".join(f"innings {number}: {innings_counts(number)}" for number in M3_INNINGS)
    log.info(
        f"{post!r}: built once in {post_seconds:.1f} s with the views checked (Requirement 11, criterion 15) and "
        "shared by every M3 candidate; terminal rules of the Post_Outcome_States (Requirement 10, criteria 14-17), "
        f"the same for every candidate: {counts}; the {int(post.innings_over(1).sum()):,} innings-1 states that end "
        "the innings are end-of-innings states (Scheduled_Balls bowled, pre-ball wickets, Par_Score at "
        "Scheduled_Balls), scored by each innings-1 candidate itself (criterion 17)",
    )
    for number in M3_INNINGS:
        for family, evaluation in evaluations[number].items():
            outcome_text = "passed" if not evaluation.failure else f"FAILED ({evaluation.failure})"
            log.info(
                f"M3 innings {number} {family}: WP of {post.n_states(number):,} Post_Outcome_States in "
                f"{evaluation.seconds:.1f} s; checks {outcome_text}",
            )
    lines = []
    for record in table.to_dict("records"):
        metrics = f"log loss {record['log_loss']:.5f}"
        if not math.isnan(record["brier"]):
            metrics += f", Brier {record['brier']:.5f}, AUC {record['auc']:.4f}"
        unit = "runs per Legal_Ball" if record["model"] == M4_MODEL else ""
        metrics += f", calibration error {record['calibration_error']:.4f}{(' ' + unit) if unit else ''}"
        flag = " [Primary_Model]" if record["primary"] else ""
        reason = f" FAILED: {record['failure_reason']}" if record["failure_reason"] else ""
        lines.append(f"{record['model']} {record['candidate']}: {metrics} on {record['n_rows']:,} rows{flag}{reason}")
    baseline = ", ".join(f"{name} {value:.5f}" for name, value in m4_baseline.items())
    log.info(
        "model comparison (Requirement 23, criteria 5-8; M3 on the eligible Deliveries of each innings, M4 on the "
        f"trainable Legal_Balls; Phase-only baseline log loss of M4: {baseline or 'n/a'}): " + "; ".join(lines),
    )
    log.info(
        f"Primary_Models (Requirement 23, criteria 7-9): M3-1 {primary[M3_MODELS[1]]}, M3-2 {primary[M3_MODELS[2]]}, "
        f"M4 {primary[M4_MODEL]}; ties within math.isclose(rel_tol={TIE_REL_TOL:g}, abs_tol={TIE_ABS_TOL:g}) go to "
        f"the first of {', '.join(M3_TIE_ORDER)} (M3) and {', '.join(M4_TIE_ORDER)} (M4)",
    )
    if m4_tiebreak is not None and m4_tiebreak.applied:
        tb = m4_tiebreak
        z = tb.mean_diff / tb.se if tb.se and math.isfinite(tb.se) and tb.se > 0 else math.nan
        log.info(
            f"M4 calibration-aware tie-break (Requirement 23, criterion 8; design DD-17): the preferred family "
            f"{tb.preferred} (log loss {tb.selected_loss:.5f}) is within the relative band delta {tb.band:g} of the "
            f"lowest log loss ({tb.best} {tb.best_loss:.5f}; gap {tb.selected_loss - tb.best_loss:+.6f}, "
            f"{(tb.selected_loss - tb.best_loss) / tb.best_loss:+.4%} of the lowest), so {tb.preferred} is the M4 "
            f"Primary_Model; it passes the own-theta NBV cell check (Requirement 15, criterion 17) that the Value_Engine "
            f"re-verifies (criterion 14). Supporting evidence, the match-clustered paired SE of the per-Legal_Ball "
            f"log-loss difference {tb.best} - {tb.preferred} over {tb.n_rows:,} Legal_Balls in {tb.n_clusters:,} "
            f"matches: mean {tb.mean_diff:+.6f}, SE {tb.se:.6f} (z {z:+.2f}); the advantage of {tb.best} is reliable "
            f"but negligible in magnitude (under a thousandth of a nat per Legal_Ball), far inside the band, so the "
            f"own-theta NBV calibration that feeds the ratings breaks the tie in favour of {tb.preferred}",
        )
    elif m4_tiebreak is not None:
        log.info(
            f"M4 calibration-aware tie-break (Requirement 23, criterion 8): the preferred family "
            f"{m4_tiebreak.preferred} did not displace the lowest-log-loss candidate ({m4_tiebreak.best}); the M4 "
            f"Primary_Model is the lowest log loss {primary[M4_MODEL]}",
        )
    for record in table.to_dict("records"):
        if record["failure_reason"]:
            log.warning(
                f"{record['model']} candidate {record['candidate']} failed and is excluded from selection and reruns "
                f"(Requirement 23, criterion 12): {record['failure_reason']}",
            )
    for number in M3_INNINGS:
        if primary[M3_MODELS[number]] == LOGISTIC:
            log.info(
                f"M3 innings {number}: the logistic baseline has the lowest pooled log loss, so the simpler model is "
                "the Primary_Model (plan addendum: if LightGBM does not beat the logistic baseline, the simpler model "
                "is used)",
            )
    spans = ", ".join(
        f"innings {number} {family} [{low:.4f}, {high:.4f}]" for (number, family), (low, high) in ranges.items()
    )
    log.info(
        f"every saved wp_pre and wp_post value is finite and within [0, 1] (Requirement 10, criterion 13): {spans}"
    )
    if curve is not None:
        wp = curve["wp_batting_first"].to_numpy(dtype=np.float64)
        log.info(
            f"WP curve of match {REPORT_MATCH_ID} (Requirement 10, criterion 18): {len(curve):,} Deliveries, WP of the "
            f"side batting first from {wp.min():.3f} to {wp.max():.3f}, {wp[0]:.3f} before the first ball and "
            f"{wp[-1]:.3f} before the last",
        )
    fitted = [fit for fit in fits if not fit.failure]
    log.info(
        f"M3 Alternative_Models (Requirement 23, criteria 1-4): {len(fitted)} of {len(fits)} innings fits completed "
        f"in {sum(fit.seconds for fit in fits):.1f} s ("
        + ", ".join(
            f"innings {fit.innings} {fit.family} {fit.seconds:.1f} s{'' if not fit.failure else ' FAILED'}"
            for fit in fits
        )
        + f"); saved {n_models} outer models and calibrators of the candidates that passed; their inner fits are "
        "registered but not saved",
    )
    passed = [fit for fit in m4_fits if not fit.failure]
    log.info(
        f"M4 Alternative_Models (Requirement 23, criteria 1, 2, 4 and 12): {len(passed)} of {len(m4_fits)} passed ("
        + ", ".join(
            f"{fit.family} fits {fit.fit_seconds:.1f} s and variants {fit.predict_seconds:.1f} s"
            f"{'' if not fit.failure else ' FAILED'}"
            for fit in m4_fits
        )
        + f"); saved the variant table, the fold-model table and the outer fold models of each one that passed "
        f"({n_m4_models} models)",
    )
    log.info(
        f"wrote {', '.join(_shown_path(path, ctx.root) for path in files)}; saved {', '.join(saved)} and the fit "
        f"registry ({n_saved} fits) in {seconds:.1f} s",
    )


def _span(values: Iterable[Any]) -> str:
    """The range of the whole numbers in ``values`` (missing ones left out) as ``low-high``, or n/a."""
    numbers = [int(value) for value in values if value is not None and not pd.isna(value)]
    if not numbers:
        return "n/a"
    low, high = min(numbers), max(numbers)
    return f"{low:,}" if low == high else f"{low:,}-{high:,}"


def _log_alternative(ctx: StageContext, fit: AlternativeFit) -> None:
    """One Run_Log line for the sandbox fits of one Alternative_Model and innings (a warning when they failed)."""
    label = f"M3 innings {fit.innings} {fit.family} (Alternative_Model)"
    if fit.failure:
        ctx.log.warning(
            f"{label}: the sandbox fits failed after {fit.seconds:.1f} s, so the candidate is excluded from selection "
            f"and reruns (Requirement 23, criterion 12): {fit.failure}",
        )
        return
    fm = fit.fold_models
    table, metrics = fm.fit_table, fm.fold_metrics
    kind = table["kind"].to_numpy(dtype=object)
    inner = table["iterations"].iloc[np.flatnonzero(kind == folds.INNER)].tolist()
    knots = [len(model.calibrator.x_knots) for model in fm.calibrated.models.values()]
    unit = "rounds" if fit.family == XGB else "trees"
    gap = metrics["held_out_minus_in_fold"].to_numpy(dtype=np.float64)
    held = metrics["held_out_log_loss_calibrated"].to_numpy(dtype=np.float64)
    ctx.log.info(
        f"{label}: {int((kind == folds.OUTER).sum())} outer and {int((kind == folds.INNER).sum())} inner fits and "
        f"{len(knots)} centred isotonic calibrators ({_span(knots)} knots) in {fm.seconds:.1f} s; {unit} kept "
        f"{_span(metrics['iterations'].tolist())} (outer), {_span(inner)} (inner); held-out minus in-fold raw log "
        f"loss per fold {' / '.join(f'{value:+.4f}' for value in gap)} (Requirement 10, criterion 8); calibrated "
        f"held-out log loss per fold {' / '.join(f'{value:.4f}' for value in held)}",
    )


def _log_m4_alternative(ctx: StageContext, fit: M4AlternativeFit) -> None:
    """One Run_Log line for the outer fits and variants of one M4 Alternative_Model (a warning when it failed)."""
    label = f"M4 {fit.family} (Alternative_Model)"
    models = fit.candidate.models
    sizes = ""
    if isinstance(models, outcome.OutcomeFoldModels):
        unit = "rounds kept" if fit.family == XGB else "trees"
        folds_in_order = list(models.folds.values())
        sizes = (
            f"; {len(folds_in_order)} outer fits on {_span(fold.n_train_rows for fold in folds_in_order)} "
            f"Training_Rows each ({_span(fold.n_early_stopping_rows for fold in folds_in_order)} of them in the "
            f"early-stopping matches); {unit} {'/'.join(str(fold.best_iteration) for fold in folds_in_order)} "
            f"(folds {'/'.join(map(str, models.folds))})"
        )
    deviation = ", ".join(f"{name} {value:.1e}" for name, value in fit.deviation.items())
    text = (
        f"{label}: fits {fit.fit_seconds:.1f} s, variants of every innings 1-2 Delivery {fit.predict_seconds:.1f} s"
        f"{sizes}; largest |row sum - 1| of the raw class probabilities before rescaling: {deviation or 'n/a'}"
    )
    if fit.failure:
        ctx.log.warning(
            f"{text}; the candidate failed and is excluded from selection and reruns (Requirement 23, criterion 12): "
            f"{fit.failure}",
        )
        return
    ctx.log.info(text)


# ============================================== the model_choice_rerun Pipeline_Stage (23.10, 23.11, 23.13)

#: Name of the rerun Pipeline_Stage and of its checkpoint (design stage 17, between ``robustness``
#: and ``activity_comparison``; folder ``17_model_choice_rerun``).
RERUN_STAGE_NAME = "model_choice_rerun"

#: The per-candidate rerun top-10 lists this stage checkpoints for the Report_Builder (task 27.1):
#: one row per reran Candidate_Model, Role and rank, the rerun's top-10 player and Final_Score.
RERUN_TOP10_TABLE = "rerun_top10"
RERUN_TOP10_COLUMNS: tuple[str, ...] = ("model", "candidate", "role", "rank", "player_id", "final_score")

#: The model-comparison column that holds each Role's criterion-11 count.
_TOP10_CHANGE_COLUMN: Mapping[str, str] = MappingProxyType(
    {"batter": "top10_changes_batter", "bowler": "top10_changes_bowler"}
)


@dataclass(frozen=True, eq=False)
class RerunResult:
    """One model-choice rerun of the ranking with a non-selected Candidate_Model (Requirement 23, criteria 10, 11).

    ``model`` (M3-1, M3-2 or M4) and ``candidate`` (the family) name the one Candidate_Model that
    took the place of the Primary_Model it competed with; every other model, every Config value and
    the Main_Ranking's D11 minimums stayed unchanged. ``result`` is the recomputed ranking
    (``cases.run_with_predictions``: WP, Live_Ball flags, LI, PI, Pressure_Ball flags, NBV,
    eligibility, Final_Score and both top-10 lists from stored predictions). ``comparisons`` maps
    each Role to the RankingComparison of the rerun's ranking against the Main_Ranking; and
    ``top10_changes`` maps each Role to the number of Main_Ranking top-10 players absent from the
    rerun's top 10 (criterion 11, an integer 0 .. LIST_SIZE). ``seconds`` times the rerun.
    """

    model: str
    candidate: str
    result: CaseResult
    comparisons: Mapping[str, RankingComparison]
    top10_changes: Mapping[str, int]
    seconds: float


def rerun_candidates(table: pd.DataFrame) -> list[tuple[str, str]]:
    """The ``(model, family)`` of every Candidate_Model to rerun: non-failed and not a Primary_Model (criterion 10).

    ``table`` is a comparison table (COMPARISON_COLUMNS). Returns the candidates in the table's
    order (MODELS, then each model's tie order), excluding the flagged Primary_Models (criterion 9)
    and the candidates with a ``failure_reason`` (criterion 12). With the delivered data that is
    M3-1 {lgbm, xgb, rf}, M3-2 {lgbm, xgb, rf} and M4 {xgb, rf}: eight reruns. Raises ValueError
    for a malformed table (``primary_models`` requires exactly one Primary_Model per model).
    """
    _require(table, ("model", "candidate", "failure_reason", "primary"), f"the {COMPARISON_TABLE} table")
    primary_models(table)  # exactly one Primary_Model per model, or ValueError
    flagged = _flags(table["primary"], f"the {COMPARISON_TABLE} column 'primary'")
    model_of = table["model"].astype(str).to_numpy(dtype=object)
    candidate_of = table["candidate"].astype(str).to_numpy(dtype=object)
    failed = np.array([_is_failure(reason) for reason in table["failure_reason"].tolist()], dtype=bool)
    return [
        (str(model), str(candidate))
        for model, candidate, bad, is_primary in zip(model_of, candidate_of, failed, flagged)
        if not bad and not is_primary
    ]


def rerun(base: BaseArtifacts, model: str, family: str, *, names: Any = None) -> RerunResult:
    """Rerun the ranking with the Candidate_Model ``model``/``family`` in the place of its Primary_Model (23.10, 23.11).

    ``base`` is the BaseArtifacts of the Main_Ranking (``cases.BaseArtifacts.build``). ``model`` is
    one of MODELS (M3-1, M3-2, M4) and ``family`` the Candidate_Model to put in that model's slot;
    every other model keeps its Primary_Model, and every Config value and the Main_Ranking's D11
    minimums stay unchanged. The stored predictions of the three models (``load_predictions``) feed
    ``cases.run_with_predictions``, which recomputes WP, Live_Ball flags, LI, PI, Pressure_Ball
    flags, NBV, eligibility, Final_Score and both top-10 lists without any refit. ``names`` is an
    optional (Role, player_id) -> ``player_name`` (a ``cases.PlayerNames``) for the entering and
    leaving players. Returns a RerunResult whose ``top10_changes`` give, per Role, the number of
    Main_Ranking top-10 players absent from the rerun's top 10 (criterion 11). Substituting a
    Primary_Model with itself reproduces the Main_Ranking (Property 53). Raises ValueError for an
    unknown model or family, or a candidate the comparison table marks as failed (criterion 12).
    """
    from superr import cases

    _check_model(model)
    if family not in TIE_ORDER[model]:
        raise ValueError(f"the {model} family must be one of {', '.join(TIE_ORDER[model])}; got {family!r}")
    started = time.perf_counter()
    chosen = dict(base.candidates)  # model -> the selected Primary_Model family
    m3 = {number: chosen[M3_MODELS[number]] for number in M3_INNINGS}
    m4 = chosen[M4_MODEL]
    if model == M4_MODEL:
        m4 = str(family)
    else:
        m3[next(number for number in M3_INNINGS if M3_MODELS[number] == model)] = str(family)
    predictions = load_predictions(base.store, m3=m3, m4=m4)
    result = cases.run_with_predictions(base, predictions, label=f"{model} {family}")
    comparisons: dict[str, RankingComparison] = {}
    changes: dict[str, int] = {}
    for role in cases.ROLES:
        role_names = None if names is None else (lambda player_id, this_role=role: names(this_role, int(player_id)))
        comparison = cases.compare_with_main(base, result, role, names=role_names)
        comparisons[role] = comparison
        changes[role] = len(comparison.leaving)  # Main top-10 players absent from the rerun's top 10 (criterion 11)
    return RerunResult(
        model=model,
        candidate=str(family),
        result=result,
        comparisons=MappingProxyType(comparisons),
        top10_changes=MappingProxyType(changes),
        seconds=time.perf_counter() - started,
    )


def _rerun_top10_frame(results: Sequence[RerunResult]) -> pd.DataFrame:
    """The rerun top-10 lists of every candidate and Role (RERUN_TOP10_COLUMNS), best rank first."""
    records: list[dict[str, Any]] = []
    for result in results:
        for role, comparison in result.comparisons.items():
            ranking = result.result.rankings[role]
            final_of = dict(
                zip(
                    ranking["player_id"].to_numpy(dtype=np.int64),
                    ranking["final_score"].to_numpy(dtype=np.float64),
                )
            )
            for rank, player_id in enumerate(comparison.other_top, start=1):
                records.append(
                    {
                        "model": result.model,
                        "candidate": result.candidate,
                        "role": role,
                        "rank": rank,
                        "player_id": int(player_id),
                        "final_score": float(final_of[int(player_id)]),
                    }
                )
    return pd.DataFrame.from_records(records, columns=list(RERUN_TOP10_COLUMNS)).astype(
        {
            "model": "str", "candidate": "str", "role": "str", "rank": np.int64, "player_id": np.int64,
            "final_score": np.float64,
        }  # fmt: skip
    )


def _with_top10_changes(table: pd.DataFrame, results: Sequence[RerunResult]) -> pd.DataFrame:
    """A copy of the comparison table with the criterion-11 counts of ``results`` in the ``top10_changes_*`` columns.

    The reran candidates' rows get their per-Role counts; the Primary_Model rows and the failed
    rows keep an empty count (criterion 13). Raises ValueError for a result whose Candidate_Model
    the table does not list exactly once, or a Role without a ``top10_changes`` column.
    """
    _require(table, COMPARISON_COLUMNS, f"the {COMPARISON_TABLE} table")
    updated = table.copy()
    model_of = updated["model"].astype(str).to_numpy(dtype=object)
    candidate_of = updated["candidate"].astype(str).to_numpy(dtype=object)
    values: dict[str, list[Any]] = {column: [pd.NA] * len(updated) for column in _TOP10_CHANGE_COLUMN.values()}
    for result in results:
        at = np.flatnonzero((model_of == result.model) & (candidate_of == result.candidate))
        if len(at) != 1:
            raise ValueError(
                f"the {COMPARISON_TABLE} table has {len(at)} row(s) for {result.model} candidate {result.candidate!r}"
            )
        index = int(at[0])
        for role, count in result.top10_changes.items():
            column = _TOP10_CHANGE_COLUMN.get(role)
            if column is None:
                raise ValueError(f"the comparison table has no top-10 change column for Role {role!r}")
            values[column][index] = int(count)
    for column, column_values in values.items():
        updated[column] = pd.array(column_values, dtype="Int64")
    return updated.astype(_comparison_dtypes())


def run_rerun_stage(ctx: StageContext) -> None:
    """The ``model_choice_rerun`` Pipeline_Stage (Requirement 23, criteria 10, 11 and 13; design stage 17).

    Rebuilds the Main_Ranking's artifacts (``cases.BaseArtifacts.build``, checked bit for bit
    against the ``rating`` checkpoint), reruns the ranking once for every non-failed Candidate_Model
    that is not a Primary_Model (``rerun_candidates``; criterion 10), and records, per rerun and
    Role, the number of Main_Ranking top-10 players absent from the rerun's top 10 (criterion 11).
    It writes those counts into the ``top10_changes_*`` columns of the model-comparison table
    (criterion 13): the checkpoint copy (COMPARISON_TABLE) and the deliverable
    ``outputs/models/model_choice/model_comparison.csv``, leaving the Primary_Model and failed rows
    blank; it also checkpoints the rerun top-10 lists (RERUN_TOP10_TABLE) for the Report_Builder.
    Every Config value and the main run's D11 minimums stay unchanged (``run_with_predictions``), so
    no model is refitted. Raises StageError (exit code 5) when the checkpoints no longer reproduce
    the Main_Ranking or a checkpoint file is missing or changed.
    """
    from superr import cases

    started = time.perf_counter()
    settings, store = ctx.settings, ctx.store

    # 1. The Main_Ranking's cached artifacts (checked against the rating checkpoint bit for bit).
    tick = time.perf_counter()
    base = cases.BaseArtifacts.build(settings, store)
    build_seconds = time.perf_counter() - tick

    # 2. The reruns of every non-selected, non-failed Candidate_Model (criterion 10), each from the
    #    stored predictions of its family in the slot of the Primary_Model it competed with.
    table = store.load_table(STAGE_NAME, COMPARISON_TABLE)
    names = cases.PlayerNames.from_store(store)
    results = [rerun(base, model, family, names=names) for model, family in rerun_candidates(table)]

    # 3. The comparison table with the criterion-11 counts and the per-candidate top-10 lists.
    updated = _with_top10_changes(table, results)
    top10 = _rerun_top10_frame(results)

    # 4. Checkpoint tables (task 27.1 reads these) and the one updated deliverable.
    store.save_table(RERUN_STAGE_NAME, COMPARISON_TABLE, updated)
    store.save_table(RERUN_STAGE_NAME, RERUN_TOP10_TABLE, top10)
    comparison_file = settings.path("outputs") / MODEL_CHOICE_FOLDER / COMPARISON_FILE
    _write_csv(comparison_file, updated)

    _log_rerun_stage(
        ctx, base=base, results=results, file=comparison_file, build_seconds=build_seconds,
        seconds=time.perf_counter() - started,
    )  # fmt: skip


def _log_rerun_stage(
    ctx: StageContext,
    *,
    base: BaseArtifacts,
    results: Sequence[RerunResult],
    file: Path,
    build_seconds: float,
    seconds: float,
) -> None:
    """The Run_Log summary of the rerun stage: one line overall and one per reran Candidate_Model."""
    from superr import cases
    from superr.ranks import ENTERING, LEAVING

    log = ctx.log
    eligible = ", ".join(f"{role} {base.n_eligible(role):,}" for role in cases.ROLES)
    log.info(
        f"{RERUN_STAGE_NAME}: rebuilt the Main_Ranking's artifacts in {build_seconds:.1f} s ({eligible} "
        f"Eligible_Players) and reran {len(results)} non-selected Candidate_Model(s) from stored predictions "
        "(Requirement 23, criterion 10)"
    )

    def named(comparison: RankingComparison, status: str) -> str:
        changes = comparison.changes
        rows = changes.loc[changes["status"].astype(str) == status]
        labels = [
            name if isinstance(name, str) and name else str(int(pid))
            for name, pid in zip(rows["player_name"].tolist(), rows["player_id"].tolist())
        ]
        return ", ".join(labels)

    for result in results:
        bits: list[str] = []
        for role, comparison in result.comparisons.items():
            bit = f"{role} {result.top10_changes[role]}"
            if comparison.leaving:
                bit += f" (out {named(comparison, LEAVING)}; in {named(comparison, ENTERING)})"
            bits.append(bit)
        log.info(
            f"  rerun {result.model} {_FAMILY_LABELS.get(result.candidate, result.candidate)}: "
            f"Main top-10 players lost {', '.join(bits)} (Requirement 23, criterion 11); {result.result.seconds:.1f} s"
        )
    log.info(
        f"wrote {_shown_path(file, ctx.root)} and saved {COMPARISON_TABLE} and {RERUN_TOP10_TABLE} to the "
        f"{RERUN_STAGE_NAME} checkpoint in {seconds:.1f} s"
    )
