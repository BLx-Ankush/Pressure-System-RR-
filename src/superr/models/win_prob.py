"""Win_Probability_Model M3: the labels and features of the two innings classifiers, the
Candidate_Model families (LightGBM and the logistic baseline), centred isotonic calibration and
the fold-sandbox fits (Requirement 10, criteria 1-10; Requirement 7, criteria 5, 6 and 9; design
sections "Win_Probability_Model M3 (``models/win_prob.py``)" and "Fold sandboxes and leakage
control"; design decision DD-14).

The first part of the module (up to ``load_fold_models``) is the M3 library of task 11.1. The
second part (task 11.2) adds the terminal rules (criteria 13-17), the Par_Score of the
end-of-innings state (criterion 17), the final WP of any states, the metrics and
Calibration_Curve plots of criteria 8, 12 and 18, the saved WP tables (criterion 19), two
diagnostics (the chase-start states and the innings-break gap) and the ``win_prob``
Pipeline_Stage. Task 26.1 adds the XGBoost and random-forest Alternative_Models (Requirement 23,
criteria 1-4) as two more CandidateFamily values, XGB and RF, which the ``model_choice`` stage fits
through fit_innings and saves with ``pre_ball_table``; the WP curve of match 40767 (criterion 18)
is drawn there too (task 13.2), once the Primary_Models are known. Every function receives the
Config values it needs through a ``Settings`` object or as arguments: no function reads
``config.py`` or has a default value for a Config setting.

Rows and labels (``m3_labels``; criteria 1 and 2)
    M3 works on the rows of the ``state`` table: every Delivery of innings 1 and 2, Wides and
    No_Balls included, in the Requirement 5 order. Each sandbox view V_k holds the same rows in
    the same order. ``m3_labels(balls, state, matches)`` gives every row (LABEL_COLUMNS):

    * ``label`` (int8): 1 when the match's ``win_team_id`` is the batting side's team
      (``batting_team_id``), 0 otherwise (criterion 1);
    * ``labelled``: the batting side and the winner are known and the winner is one of the two
      sides, so that the label means what criterion 1 says;
    * ``kept``: the Result_Parser's result type (``result_type`` of ``balls``) is a win by runs
      or by wickets (KEPT_RESULT_TYPES). Ties, super-over wins, abandoned matches, no results
      and unparsed results (DROPPED_RESULT_TYPES) are left out of training and of every metric
      (criterion 2), but their Deliveries are still predicted (criterion 19);
    * ``trainable``: ``common.training_mask``, which leaves out Super_Over_Innings and the
      matches excluded under the Score_Tolerance rule (Requirement 5, criteria 11 and 24);
    * ``eligible`` = trainable & kept & labelled: the rows an M3 fit may train on (those of its
      train folds) and the rows of every M3 metric.

    It also carries ``sched_balls``, the innings' Scheduled_Balls from ``balls``, a first-innings
    feature. On the current checkpoints, 833 of the 867 matches with Deliveries are kept (414
    won by runs, 419 by wickets) and 34 are dropped (21 no results, 9 super-over wins, 3
    abandoned, 1 tie, no unparsed result); 47 kept matches are excluded under the
    Score_Tolerance rule, which leaves 786 matches and 185,625 eligible Deliveries (96,886 in
    innings 1, 88,739 in innings 2). Every kept Delivery is labelled.

Features (``m3_features``; criteria 3 and 4)
    M3_FEATURES[innings] lists the pre-ball features in the column order of every fit of that
    innings, and M3_MONOTONE[innings] the monotone constraint of each (criterion 5):

    * innings 1: Legal_Balls bowled ``lb_bowled``, team score ``score`` (+1), wickets fallen
      ``wickets`` (-1), Par_Score ``par_score``, Team_Strength_Gap ``tsg``, Scoring_Environment
      ``scoring_env``, Venue_Effect ``venue_effect`` and Scheduled_Balls ``sched_balls``;
    * innings 2: runs needed ``runs_needed`` (Target minus team score; -1), Legal_Balls left
      ``lb_left`` (+1), ``wickets`` (-1), RBQ ``rbq`` (+1), Required_Rate minus Achievable_Rate
      ``rr_minus_ar`` and ``tsg``.

    ``lb_bowled``, ``score``, ``wickets``, ``runs_needed`` and ``lb_left`` come from the
    ``state`` table (STATE_FEATURES); ``par_score``, ``tsg``, ``scoring_env``, ``venue_effect``,
    ``rbq`` and ``rr_minus_ar`` from a sandbox view (VIEW_FEATURES gives the view column; the
    second-innings ``rr_minus_ar`` is the view's ``f19_component``, which the ``remaining_runs``
    stage sets to Required_Rate minus Achievable_Rate on innings-2 rows); ``sched_balls`` from the
    label table. So every fit of sandbox k reads its model-derived features from V_k (Requirement
    7, criteria 7 and 8). ``m3_features(innings, view, state, labels)`` returns them as float64
    columns, NaN for a missing value, for the rows of that innings in ``state`` order. As
    delivered, no innings-1 or innings-2 row of any view misses a feature.

Candidate_Model families (``Candidate``, ``CandidateFamily``, ``candidate_family``)
    A Candidate is one unfitted or fitted classifier of one family for one innings. ``fit(X, y,
    groups, seed, early_stopping=...)`` trains it on a fit's Training_Rows: X the M3_FEATURES of
    the innings, y the labels, groups the match id of each row, seed the family's named seed, and
    ``early_stopping`` a bool per row that marks the whole matches held out to decide early
    stopping (None for a family without early stopping). ``predict_raw(X)`` returns the
    uncalibrated probability that the batting side wins. ``model`` is the fitted library object
    that the CheckpointStore saves, and ``iterations`` and ``converged`` describe how the fit
    ended. A CandidateFamily names the family (the FitRegistry family and part of every fit id),
    its seed, its EarlyStopping (or None) and how to make a new Candidate for an innings or wrap a
    saved model. ``candidate_family(name, settings)`` builds each family: the two that the
    ``win_prob`` stage fits (FAMILIES) and the two Alternative_Models of task 26.1
    (ALTERNATIVE_FAMILIES), which the ``model_choice`` stage fits:

    * LGBM (``LgbmCandidate``): a LightGBM binary classifier from ``common.lgbm_setup(settings,
      "m3", "binary", monotone_constraints=M3_MONOTONE[innings])``: the ``models.m3`` learning
      rate, ``num_leaves``, ``min_data_in_leaf``, ``lambda_l2`` and round cap ``max_rounds``
      (criterion 6), the section's ``monotone_constraints_method``, ``runtime.n_threads``,
      ``runtime.deterministic`` and ``seeds.m3_lgbm``. Training stops once the binary log loss of
      the held-out matches (``models.m3.es_share`` of the fit's training matches, drawn by
      ``leakage.early_stopping_split`` with ``seeds.m3_lgbm``) has not improved for
      ``models.m3.es_patience`` rounds, and the model of the best round is kept (criterion 7).
      LightGBM enforces the monotone constraints while it grows each tree, so the raw output is
      non-decreasing in score and non-increasing in wickets in innings 1, and non-increasing in
      runs needed and wickets and non-decreasing in Legal_Balls left and RBQ in innings 2, for
      every value of the other features. ``converged`` is True when early stopping ended the fit
      before the round cap.
    * LOGISTIC (``LogisticCandidate``): the logistic-regression baseline of criterion 10, with
      the same features, labels and folds: a scikit-learn pipeline of median imputation with a
      missing-value indicator for each feature that has a missing value
      (``SimpleImputer(strategy="median", add_indicator=True)``), standardisation
      (``StandardScaler``) and ``LogisticRegression`` (L2, lbfgs) with ``models.m3.logistic.C``,
      ``models.m3.logistic.max_iter`` and ``random_state=seeds.m3_logistic``. The medians, the
      indicators and the scaling are all fitted on the fit's Training_Rows only; standardising
      makes lbfgs converge in a few iterations whatever the feature scales. A missing value at
      prediction time takes the training median, also in a feature that had none in training.
      ``converged`` is True when lbfgs stopped before ``max_iter``. No monotone constraint
      applies (criterion 5 names the LightGBM classifiers).
    * XGB (``XgbCandidate``; Requirement 23, criteria 1, 2 and 4): an XGBoost binary classifier
      (``binary:logistic``) from ``common.xgb_setup(settings, "m3", "binary",
      monotone_constraints=M3_MONOTONE[innings])``: the ``models.xgb.m3`` ``eta``, ``max_depth``,
      ``min_child_weight``, ``reg_lambda``, ``tree_method`` ("hist" in deterministic mode) and round
      cap ``max_rounds``, the constraints of criterion 5 in tuple form, ``runtime.n_threads`` and
      ``seeds.m3_xgb``. Early stopping works as for LGBM, with ``models.xgb.m3.es_share``,
      ``models.xgb.m3.es_patience`` and ``seeds.m3_xgb``: whole matches drawn from the fit's
      training folds only, and the Booster cut to its best round. XGBoost enforces the
      constraints while it grows each tree. ``iterations`` is the number of rounds kept and
      ``converged`` is True when early stopping ended the fit before the round cap.
    * RF (``RfCandidate``; Requirement 23, criteria 1, 2 and 4): a scikit-learn pipeline of
      median imputation fitted on the fit's Training_Rows (``SimpleImputer(strategy="median",
      keep_empty_features=True)``, no indicator columns, so that the forest sees the features of
      M3_FEATURES alone) and a ``RandomForestClassifier`` from ``common.rf_setup(settings, "m3",
      monotonic_cst=M3_MONOTONE[innings])``: the ``models.rf.m3`` ``n_estimators``,
      ``max_depth``, ``min_samples_leaf`` and ``max_features``, the constraints of criterion 5 as
      ``monotonic_cst`` (scikit-learn applies them to the probability of label 1 and only to data
      without missing values, hence the imputation), ``random_state=seeds.m3_rf`` and
      ``runtime.n_threads`` for fitting. In deterministic mode the fitted forest predicts with 1
      thread (``common.rf_setup``). No early stopping; ``iterations`` is the number of trees and
      ``converged`` is None.

    The XGB and RF candidates check their raw output like the other families, but a value that is
    missing, infinite or outside [0, 1] raises InvalidPrediction, a ValueError that counts each
    kind, so that the Model_Choice_Checker can record which condition of Requirement 23,
    criterion 12 the candidate met.

    ``fit_candidate`` fits one Candidate on rows that are all Training_Rows without a
    FitRegistry, with the same early-stopping split as a registered fit, for tests and checks.

Calibration (``calibration_bounds``, ``fit_calibrator``, ``CentredIsotonic``, ``calibrate``,
``CalibratedModel``; criterion 9)
    Each fold's classifier is calibrated with centred isotonic regression (CIR; Oron and
    Flournoy, 2017; design decision DD-14). ``fit_calibrator(raw, y, bounds)`` builds the
    calibrator from the raw predictions and labels of its Training_Rows in three steps:

    1. it fits ``IsotonicRegression(y_min=lower, y_max=upper, increasing=True,
       out_of_bounds="clip")``, the step function of a plain isotonic calibrator. Ties in the raw
       values are scikit-learn's: sorted raw values less than 1e-15 above the first value of a
       group of ties count as one value, with the group's mean label and its row count as weight;
    2. each block of that fit, a maximal run of consecutive sorted training predictions with one
       fitted value (the bounds can merge neighbouring runs at either end), gives one knot: x is
       the block's centroid, the mean raw prediction of its training rows (kept within the
       block's own raw range against rounding), and y is the block's fitted value;
    3. the calibrated WP is the straight line between consecutive knots (``np.interp``). Below
       the first knot it is the first knot's value, above the last knot the last knot's value,
       and every value is clipped to [lower, upper].

    Consecutive knots differ in both x and y, so the calibrated WP is non-decreasing everywhere,
    strictly increasing between the first and the last knot and flat beyond them; a fit with a
    single block gives a constant WP. ``CentredIsotonic`` holds the result: ``x_knots`` and
    ``y_knots`` (read-only float64, both strictly increasing) and ``bounds`` (lower, upper).
    ``predict(raw)`` evaluates it and also keeps each value within the WP range of its own
    segment, so that float rounding never lets a line overshoot the next knot.
    ``calibrate(calibrator, raw)`` checks the calibrator and returns ``predict(raw)``.

    Why centring: the plain isotonic calibrator is a step function. With it, the logistic
    Primary_Models took only 401 (innings 1) and 472 (innings 2) distinct pre-ball WP values
    inside the Live_Ball band (0.05-0.95), the dot-ball Post_Outcome_State had exactly the
    pre-ball WP on 61-62% of Deliveries, and 9,563 Deliveries had an unscaled Leverage_Index of
    exactly 0. CIR keeps the isotonic fit at the block centres and replaces the flat steps between
    them with straight lines, so a small change of state moves the WP. On the same classifier fits
    (rerun of 14 November 2026) the logistic calibrators have 62-97 knots; inside the band the
    logistic WP takes 104,074 distinct values on 104,090 innings-1 Deliveries and 70,666 on
    70,756 innings-2 Deliveries, and no Delivery there keeps its WP after a dot ball; 2,422
    Deliveries have an unscaled Leverage_Index of exactly 0: the 249 chases already won and
    2,173 states held at a calibration bound beyond an end knot. The pooled calibrated log loss
    went from 0.62888 to 0.62786 in innings 1 and from 0.40670 to 0.40639 in innings 2 (raw
    logistic 0.62571 and 0.40447).

    [lower, upper] is ``models.m3.calibration_bounds`` ([0.001, 0.999] as delivered), which
    ``calibration_bounds(settings)`` reads and checks (0 <= lower < upper <= 1). ``fit_innings``
    reads the bounds itself and fits every family's calibrators with fit_calibrator, so the
    XGBoost and random-forest families of task 26.1 get the same calibrator. The calibrator is
    non-decreasing, so a calibrated LightGBM, XGBoost or random-forest output keeps every monotone
    direction of the raw one. Why the bounds: where every training prediction at the low (high) end has label 0 (1),
    an unbounded isotonic fit gives exactly 0 (1) there, which ``common.pooled_log_loss`` clips to
    1e-15 (1 - 1e-15), so each held-out Delivery of a comeback match there adds about 34.5 to the
    sum of the log loss; at the delivered bounds it adds -log(0.001), about 6.9. The bounds lie
    outside the Live_Ball band (D1, 0.05-0.95), and the terminal rules of task 11.2 still give
    exactly 0, 0.5 or 1 where a state is decided. ``CalibratedModel(candidate, calibrator)`` joins
    a fitted Candidate and its CentredIsotonic: ``predict(X)`` returns the calibrated WP of any
    states with the innings' M3_FEATURES columns (before the terminal rules of task 11.2), and
    ``predict_raw(X)`` the candidate's raw output.

The fold-sandbox fits (``fit_innings``; criteria 7-9; Requirement 7, criteria 5, 6 and 9)
    ``fit_innings(innings, family, view_by_k, labels, settings, registry, state=state)`` runs the
    D15 x D15 fits of ``folds.sandbox_plan(D15)`` (25 with D15 = 5) for one innings and family,
    and D15 calibrators. For each target fold k, with S_k the other folds:

    1. it checks V_k with ``views.check_view`` before any fit reads it, and builds the M3
       features of the innings' rows from V_k (``m3_features``);
    2. the outer fit M3(-k) trains on the eligible rows of V_k in S_k and predicts the raw
       output of every fold-k row of V_k: the excluded matches and the dropped result types
       too (criterion 19);
    3. each inner fit M3^(k)(-j), j in S_k, trains on the eligible rows of V_k outside folds j
       and k and predicts the raw output of every fold-j row of V_k;
    4. the fold-k centred isotonic calibrator (fit_calibrator), bounded to
       ``models.m3.calibration_bounds``, is fitted on the inner predictions of the eligible rows
       of S_k, each from the inner fit M3^(k)(-j) of its own fold j, so it uses only predictions
       for Deliveries outside fold k, each from a fit without that Delivery's fold (criterion 9);
    5. the calibrated Out_Of_Fold_Prediction of every fold-k row is the fold-k calibrator
       applied to the outer fit's raw output.

    Every classifier fit goes through ``common.fit_registered`` with model ``M3_MODELS[innings]``
    (``m3_inn1``, ``m3_inn2``) and the family's name, so its fit id is that of
    ``leakage.fit_id_for`` and ``views.view_plan`` (``m3_inn1_lgbm_outer_k1``,
    ``m3_inn2_logistic_inner_k3_j1``, ...). The leakage guard checks its Training_Rows and
    early-stopping matches against its excluded folds ({k} or {j, k}) and its predicted matches
    (the matches of the fold-k or fold-j rows it predicts) before it runs (Requirement 7,
    criteria 6, 9 and 12). Each calibrator is registered the same way, under model
    ``CALIBRATOR_MODELS[innings]`` (``m3_inn1_cal``, ``m3_inn2_cal``) with the outer spec k: its
    Training_Rows are the S_k rows whose inner predictions it fits, its predicted matches those of
    fold k (fit ids such as ``m3_inn1_cal_lgbm_outer_k1``).

    It returns a FoldModels with, for each row of the innings in ``state`` order: ``row``,
    ``match_id``, ``fold``, ``label``, ``eligible``, the outer fit's raw output ``raw`` and the
    calibrated WP ``wp``; per k, the inner raw predictions ``inner_raw[k]`` (NaN on fold-k rows)
    and the positions of the calibrator's Training_Rows ``calibration_rows[k]``; the fitted inner
    candidates and the fit ids; the calibrated outer models as a CalibratedFolds
    (``calibrated``); FOLD_METRICS_COLUMNS per fold (``fold_metrics``) and FIT_TABLE_COLUMNS per
    fit (``fit_table``). The overfitting check of criterion 8, for each outer fit, is the raw
    in-fold log loss on the rows the classifier was fitted on (its Training_Rows other than the
    early-stopping matches), the raw held-out log loss on the eligible fold-k rows and their
    difference; the table also holds the raw log loss of the early-stopping rows and the
    calibrated held-out log loss. Every log loss is ``common.pooled_log_loss``. Both families get
    the same table.

Calibrated fold models for any state (``CalibratedFolds``)
    ``fold_models.calibrated.predict(X, fold)`` gives the calibrated WP of any rows, each from
    the calibrated outer model of its fold: X with the innings' M3_FEATURES columns, ``fold`` the
    fold of each row's match from the Fold_Table. Pre-ball states, Post_Outcome_States (the
    end-of-innings states among them) and the diagnostic chase-start states all go through it,
    with the terminal rules on top (``win_probability``). ``calibrated.models[k]`` is the
    CalibratedModel of fold k.

Saved models (``save_fold_models``, ``load_fold_models``)
    ``save_fold_models(store, stage, fold_models)`` saves, under its fit id, every outer
    candidate's model and every calibrator, and (``inner=True``) every inner candidate's model,
    with ``CheckpointStore.save_model``: a LightGBM Booster as LightGBM model text, with its best
    round; an XGBoost Booster, cut to its best round, as XGBoost JSON; a logistic or random-forest
    pipeline and a CentredIsotonic calibrator with joblib (a calibrator's pickle rebuilds it
    through its constructor, which checks the knots again). ``load_fold_models(store,
    stage, innings, family, n_folds)`` reads the outer models and calibrators of folds 1 to D15
    back as a CalibratedFolds. A checkpoint saved before the centred isotonic calibrator holds
    scikit-learn IsotonicRegression calibrators, which load_fold_models refuses (TypeError): the
    stage that saved it must run again.

Terminal rules (``terminal_rules``, ``apply_terminal_rules``, ``TerminalWP``; criteria 13-17)
    The rules flag a state before any classifier sees it, for pre-ball states and for the
    Post_Outcome_States of the Pressure_Engine alike. ``terminal_rules(innings, states)`` gives
    each row one of TERMINAL_RULES. In innings 2 it reads ``runs_needed`` (r, Target minus team
    score; fractional in a Post_Outcome_State whose score adds a Class_Run_Value such as the
    Run_Out class's, about 0.43), ``lb_left`` and ``wickets``. A chase has ended when no
    Legal_Ball is left (``lb_left`` <= 0) or ALL_OUT_WICKETS (10) Counted_Wickets have fallen:

    ============== ======================================== ========= =========
    Rule           Condition (innings 2)                    WP        Criterion
    ============== ======================================== ========= =========
    CHASE_WON      r <= 0 (score >= Target), ended or not   1         14
    CHASE_LOST     ended and r > 1 (score < Target - 1)     0         15
    CHASE_TIED     ended and r = 1 (score = Target - 1)     0.5       16
    CHASE_FRACTION ended and 0 < r < 1                      1 - r / 2 see below
    ============== ======================================== ========= =========

    CHASE_FRACTION covers the fractional scores strictly between Target - 1 and Target, which
    criteria 14-16 leave open: the WP runs on the straight line from 0.5 at r = 1 (criterion 16)
    to 1 at r = 0 (criterion 14), so the WP of an ended chase never falls as the score rises
    within that run. These four rules (DECIDING_RULES) set the WP instead of the classifier. In
    innings 1 the rule reads ``lb_bowled``, ``sched_balls`` and ``wickets``: INNINGS_END flags a
    state whose Legal_Balls bowled reach Scheduled_Balls or that has 10 Counted_Wickets fallen,
    but sets no WP. The calibrated innings-1 classifier scores every innings-1 state, so the
    outcomes of one Delivery are never scored by two different classifiers (criterion 17), and
    the flag only counts the states that end the innings. Every other state is NOT_TERMINAL.
    Terminality rests on these counts alone: a dismissal that leaves an empty Batters_To_Come with
    fewer than 10 wickets fallen does not end the innings here, because criteria 15 and 17 name 10
    Counted_Wickets (Omega's post-wicket rule, Requirement 9, criterion 13, is a separate rule). A
    missing ``lb_left``, ``wickets``, ``lb_bowled`` or ``sched_balls``, or a missing
    ``runs_needed`` on an ended chase, raises ValueError, because no rule can be chosen; a missing
    ``runs_needed`` on a chase that has not ended leaves the state NOT_TERMINAL.

    ``apply_terminal_rules(innings, states, wp)`` returns a TerminalWP: ``wp``, the final WP of
    each row, within [0, 1], and ``rule``. The ``wp`` argument holds the calibrated classifier WP
    of each row; NaN is allowed only on the rows that a rule of DECIDING_RULES decides, which never
    read it. On the checkpoints of 14 November 2026 no pre-ball state of innings 1 ends the
    innings, and 249 pre-ball states of innings 2 have r <= 0 (CHASE_WON): 209 of them are
    eligible, and 1 of those lies in a match the chasing side lost.

The end-of-innings state (``innings_end_par``; criterion 17)
    The state after an outcome that ends the first innings (Legal_Balls bowled reach
    Scheduled_Balls, or 10 Counted_Wickets) is scored at the end-of-innings state: Legal_Balls
    bowled = Scheduled_Balls; team score = the score after the outcome (pre-ball score +
    Class_Run_Value); wickets fallen = the wickets fallen before the Delivery, because once the
    innings is over the wickets in hand no longer matter, so the classifier's wickets effect stays
    out of the WP differences between the classes; Par_Score = Par_Score at Scheduled_Balls; and
    the Delivery's Team_Strength_Gap, Scoring_Environment, Venue_Effect and Scheduled_Balls.
    ``pressure.post_outcome_states`` builds these states. ``innings_end_par(m2_fit, scoring_env,
    venue_effect, sched_balls, m2_models=..., par_means=..., par_bias=...)`` gives their
    Par_Score: for each row, the corrected par curve (DD-7) of the first-innings M2 fit that the
    row's view row names in ``m2_fit`` (the outer fit M2(-k) for a fold-k row of V_k), at the
    row's context, read at Scheduled_Balls (``remaining_runs.par_contexts``, ``par_curve``,
    ``corrected_par_curve`` and ``par_lookup`` with the fit's rows of the ``remaining_runs``
    tables PAR_MEANS_TABLE and PAR_BIAS_TABLE). It is the curve behind the row's ``par_score`` and
    ``par_score_next``, so at the last Legal_Ball of an innings it equals ``par_score_next``
    exactly. A pre-ball state never ends the innings as delivered, so pre-ball WP is unchanged.

Diagnostics: chase-start states and the innings-break gap (``chase_start_states``,
``chase_start_features``, ``innings_break_gap``)
    No WP of a Delivery depends on them. ``chase_start_states(balls, state, view_by_k,
    fold_of_row, ...)`` gives one row (CHASE_COLUMNS) per match with a Delivery in innings 1 or
    2: the chasing side's state at the start of innings 2 as the fold models of the match's fold
    f see it. It reads view V_f, whose fold-f rows take their values from M1(-f) and
    M2_first(-f), the fits behind every Out_Of_Fold_Prediction of the match.

    * CHASE_FROM_DELIVERY, a match with innings-2 Deliveries (845 of the 867 matches with
      Deliveries as delivered): the RBQ, Achievable_Rate (``ach_rate``) and Team_Strength_Gap
      (``tsg``) of V_f at the chasing side's first innings-2 Delivery (the lowest
      ``innings_ball_num``), with that innings' Scheduled_Balls.
    * CHASE_FROM_XI, a match without innings-2 Deliveries (22, all abandoned or no result): the
      state is built from the chasing side's playing XI, the innings-1 bowling side. Its
      ``is_played`` players in the Batters_To_Come order (lowest ``batting_order`` first, ties in
      ``playing_eleven`` row order) put the first two at the crease as striker and non-striker,
      and ``state.batters_to_come`` gives the Batters_To_Come of a Delivery that starts innings 2
      with them, so the Impact Player and cap rules of Requirement 6, criteria 13 and 14, hold.
      The M1 fit that serves the match's rows of V_f (``m1_fit``) supplies theta_bat, and with
      its Typical_Top7_Sum (the ``m1_fits`` table) the RBQ (``skill.rbq``). TSG is minus the
      innings-1 value of V_f. Achievable_Rate comes from the first-innings M2 fit that serves
      those rows (``m2_fit``; ``remaining_runs.achievable_rate``) at Legal_Balls left = the
      innings-2 Scheduled_Balls (Requirement 5, criteria 4-6, from ``matches``), 0 wickets, 0
      Legal_Balls faced by the striker, the openers' theta_bat, the match's Scoring_Environment
      and Venue_Effect and the Match_Day_Conditions at the end of innings 1: 6 x (bat runs minus
      M1's predicted bat runs, ``m1_pred_runs`` of V_f) summed over every innings-1 Legal_Ball of
      the match / their count, or 0 when there is none.

    ``chase_start_features(chase, match_id, first_innings_total)`` gives the innings-2
    M3_FEATURES of the chase-start state of each row for any first-innings totals, fractional
    ones included: runs needed = total + 1, Legal_Balls left = the innings-2 Scheduled_Balls, 0
    wickets, the chase RBQ and TSG, and Required_Rate minus Achievable_Rate = 6 x runs needed /
    Legal_Balls left - ``ach_rate``.

    ``innings_break_gap(balls, state, view_by_k, fold_of_row, chase, models, m2_models=...,
    par_means=..., par_bias=...)`` measures the jump at the innings break. For every match with
    Deliveries in both innings and every candidate family of ``models``, it compares WP1 at the
    actual end of innings 1 with 1 - WP2 at the chase start, both from the family's calibrated
    models of the match's fold f. WP1 is the innings-1 WP of the end-of-innings state of the last
    innings-1 Delivery's actual outcome: Scheduled_Balls bowled, that Delivery's pre-ball score
    plus its team runs (the first-innings total), its pre-ball wickets, Par_Score at
    Scheduled_Balls (innings_end_par) and its V_f values of TSG, Scoring_Environment and
    Venue_Effect. WP2 is the innings-2 WP of the chase-start state for that total
    (chase_start_features). The jump falls between two Deliveries, so no Leverage_Index includes
    it. The result (INNINGS_BREAK_COLUMNS) holds one ``match`` row per family and match, with
    ``gap`` = WP1 - (1 - WP2), then one ``summary`` row per family with the mean of each WP, the
    mean gap, the mean absolute gap and ``n_matches``.

The final WP of any states (``win_probability``, ``load_wp_models``)
    ``win_probability(innings, states, fold=..., models=...)`` is the WP of criterion 13 for any
    states of one innings, pre-ball or Post_Outcome_States: ``states`` with the innings'
    M3_FEATURES columns; ``fold`` the fold of each row's match; ``models`` the calibrated fold
    models of one Candidate_Model, ``{innings: CalibratedFolds}`` or both innings
    (``load_wp_models`` reads them from the checkpoint). It applies terminal_rules first and calls
    ``models[innings]`` on every row that no rule of DECIDING_RULES decides, each from the model
    of its fold, so a decided chase never reaches a classifier, whatever its features hold. In
    innings 1 that is every row: an INNINGS_END row is scored by the innings-1 classifier, on the
    end-of-innings features its caller gives it. Returns a TerminalWP.

Metrics (``calibration_curve``, ``calibration_error``, ``brier_score``, ``roc_auc``,
``wp_metrics``; criteria 8, 12 and 20)
    ``calibration_curve(p, y)`` is the Calibration_Curve of the glossary: for each of the 10
    equal-width WP bins [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0] (CALIBRATION_EDGES; a WP equal to
    k/10 opens bin k, and 1.0 closes the last bin), the Delivery count, the mean WP and the
    observed share of label 1 (the batting side won). ``calibration_error(curve)`` is the
    Delivery-count-weighted mean of |mean WP - observed share| over the non-empty bins.
    ``wp_metrics(y, p)`` gives, pooled over every row given, the Pooled_Held_Out_Log_Loss
    (``common.pooled_log_loss``, every probability clipped to [1e-15, 1 - 1e-15]), the Brier
    score (mean squared difference), the AUC (NaN when one label is absent) and the
    Calibration_Error.

The ``win_prob`` Pipeline_Stage (``run_stage``)
    Reads ``balls`` (``reconstruct``), ``state`` (``state``), ``matches`` and ``playing_eleven``
    (``load``), the M1 theta and fit tables (``skill``), the views V_1 ... V_D15, the outer
    first-innings M2 fits with their Legal_Balls-left calibration (``remaining_runs.load_models``,
    so the end-of-innings Par_Score and the chase-start Achievable_Rate use the calibrated
    predictions behind the views) and their par means and par bias (``remaining_runs``), and the
    Fold_Table through ``ctx.registry``. Then:

    1. builds the m3_labels table and checks every view with ``views.check_view``;
    2. runs fit_innings for each innings and family (FAMILIES: logistic, then LightGBM): 2 x 2 x
       D15 x D15 classifier fits and 2 x 2 x D15 calibrators, all registered with the
       FitRegistry. Every candidate model and calibrator, inner fits included, is saved under its
       fit id (save_fold_models), as the remaining_runs stage saves all its fits;
    3. builds the diagnostics: the chase-start table (chase_start_states), saved as CHASE_TABLE,
       and the innings-break gap of both families (innings_break_gap);
    4. gives every innings 1-2 Delivery of each Candidate_Model's innings its final pre-ball WP,
       excluded matches and dropped result types included (criterion 19): the calibrated
       Out_Of_Fold WP of fit_innings with apply_terminal_rules on top. Each candidate's table is
       saved as ``wp_table(innings, family)`` (WP_TABLE_COLUMNS);
    5. writes ``outputs/models/m3/`` (M3_FOLDER): METRICS_FILE (METRICS_COLUMNS) with, per innings
       and Candidate_Model, the Pooled_Held_Out_Log_Loss, Brier score, AUC and Calibration_Error
       overall and per Phase (criterion 12), the in-fold and held-out log loss and their
       difference per fold (criterion 8, both families) and the terminal-rule counts;
       FOLD_METRICS_FILE (FOLD_METRICS_COLUMNS of every candidate); per innings
       ``calibration_file(innings)``, the Calibration_Curves of both candidates overall and per
       Phase, and ``calibration_chart_file(innings)``, a PNG with one panel per curve and both
       candidates in each, drawn with matplotlib's Agg canvas at CHART_DPI (criterion 18); and
       INNINGS_BREAK_FILE, the innings-break gap;
    6. saves LABELS_TABLE, FIT_TABLE (FIT_TABLE_COLUMNS of every fit), FOLD_METRICS_TABLE and the
       fit registry, and records the boosting rounds of the outer LightGBM fits as the
       data-derived Method_Constant BEST_ITERATION_CONSTANT.

    Every metric is computed on the eligible rows from the calibrated classifier WP before the
    terminal rules (``wp_raw``). The rules give every candidate the same WP on a decided state,
    and an eligible decided state whose label disagrees with the rule, through reconstruction
    noise, would add about 34.5 to the log-loss sum of every candidate alike. METRICS_FILE
    reports instead, per rule, the pre-ball rows that the rule flags, the eligible ones and the
    eligible ones whose label differs from the final WP by 0.5 or more.

Reading the stage's outputs (``model_choice``, ``pressure``)
    * ``store.load_table(STAGE_NAME, wp_table(innings, family))``: one row per Delivery of the
      innings, in ``state`` order, with ``row``, ``fold``, ``wp_pre`` (final, after the terminal
      rules), ``wp_raw`` (calibrated, before them), ``terminal_rule`` (a TERMINAL_RULES name,
      ``none`` on a NOT_TERMINAL row), ``fit_id`` and ``calibrator_fit_id`` (the outer fit
      M3(-fold) and the fold calibrator behind the row). LABELS_TABLE gives the labels and
      ``eligible`` flags of the same rows. ``pre_ball_table(innings, fold_models, state, labels)``
      builds the same table from any fit_innings result; the ``model_choice`` stage saves those of
      the Alternative_Models in its own checkpoint (task 26.1).
    * ``load_wp_models(store, family, n_folds)`` gives what ``win_probability`` needs for any
      other states, such as the Post_Outcome_States of ``pressure.post_outcome_states``.
      CHASE_TABLE holds the diagnostic chase-start states, which no WP reads.

Errors: ValueError or TypeError, naming the problem, for malformed arguments, such as a missing
column, tables that do not hold the same rows in the same order, a label other than 0 or 1, a
missing value where none may be, calibration bounds that break 0 <= lower < upper <= 1, a
calibrator that is not the CentredIsotonic of criterion 9 (fit_calibrator), knots that are not
strictly increasing or a model whose predictions are not finite probabilities. ``fit_innings``
raises FitLeakageError or ViewLeakageError (exit code 4) when a fit or a view breaks the leakage
guard (Requirement 7, criterion 12), and DataError (exit code 3) for a match that is not in the
Fold_Table.
"""

from __future__ import annotations

import os
import re
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import pandas as pd

from superr import folds, leakage, reconstruct, views
from superr.models import common
from superr.models.remaining_runs import ACH_RATE_COLUMN, F19_COLUMN, M2_FEATURES, achievable_rate
from superr.models.remaining_runs import FAMILY as M2_FAMILY
from superr.models.remaining_runs import FIRST_INNINGS_MODEL as M2_FIRST_INNINGS_MODEL
from superr.models.remaining_runs import PAR_BIAS_COLUMNS, PAR_BIAS_TABLE, PAR_MEAN_FEATURES, PAR_MEANS_TABLE
from superr.models.remaining_runs import STAGE_NAME as M2_STAGE
from superr.models.remaining_runs import corrected_par_curve, par_contexts, par_curve, par_lookup
from superr.models.remaining_runs import load_models as load_m2_models
from superr.reconstruct import RESULT_TYPES, UNPARSED, WIN_BY_RUNS, WIN_BY_WICKETS, DeliveryClass
from superr.skill import FIT_SUMMARY_TABLE as M1_FITS_TABLE
from superr.skill import STAGE_NAME as SKILL_STAGE
from superr.skill import THETA_TABLE as M1_THETA_TABLE
from superr.skill import rbq as skill_rbq
from superr.state import BALLS_PER_OVER, BTC_INPUT_COLUMNS, BTC_SLOT_COLUMNS, STATE_INNINGS, STATE_TABLE, XI_COLUMNS
from superr.state import STAGE_NAME as STATE_STAGE
from superr.state import Phase, batters_to_come

if TYPE_CHECKING:
    from superr.leakage import FitRegistry
    from superr.pipeline import StageContext
    from superr.settings import Settings
    from superr.store import CheckpointStore

__all__ = [
    "ALL_OUT_WICKETS",
    "ALTERNATIVE_FAMILIES",
    "BEST_ITERATION_CONSTANT",
    "CALIBRATION_BINS",
    "CALIBRATION_COLUMNS",
    "CALIBRATION_EDGES",
    "CALIBRATION_TABLE_COLUMNS",
    "CALIBRATOR",
    "CALIBRATOR_MODELS",
    "CHART_DPI",
    "CHASE_COLUMNS",
    "CHASE_FRACTION",
    "CHASE_FROM_DELIVERY",
    "CHASE_FROM_XI",
    "CHASE_LOST",
    "CHASE_TABLE",
    "CHASE_TIED",
    "CHASE_WON",
    "DECIDING_RULES",
    "DROPPED_RESULT_TYPES",
    "FAMILIES",
    "FIT_TABLE",
    "FIT_TABLE_COLUMNS",
    "FOLD_METRICS_COLUMNS",
    "FOLD_METRICS_FILE",
    "FOLD_METRICS_TABLE",
    "GAP_MATCH",
    "GAP_SUMMARY",
    "INNINGS_BREAK_COLUMNS",
    "INNINGS_BREAK_FILE",
    "INNINGS_END",
    "KEPT_RESULT_TYPES",
    "LABELS_TABLE",
    "LABEL_COLUMNS",
    "LABEL_FEATURES",
    "LGBM",
    "LOGISTIC",
    "M3_FEATURES",
    "M3_FOLDER",
    "M3_INNINGS",
    "M3_MODELS",
    "M3_MONOTONE",
    "METRICS_COLUMNS",
    "METRICS_FILE",
    "NOT_TERMINAL",
    "PHASE_NAMES",
    "RF",
    "STAGE_NAME",
    "STATE_FEATURES",
    "TERMINAL_COLUMNS",
    "TERMINAL_RULES",
    "VIEW_FEATURES",
    "WP_TABLE_COLUMNS",
    "XGB",
    "CalibratedFolds",
    "CalibratedModel",
    "Candidate",
    "CandidateFamily",
    "CentredIsotonic",
    "FoldModels",
    "InvalidPrediction",
    "LgbmCandidate",
    "LogisticCandidate",
    "RfCandidate",
    "TerminalWP",
    "XgbCandidate",
    "apply_terminal_rules",
    "brier_score",
    "calibrate",
    "calibration_bounds",
    "calibration_chart_file",
    "calibration_curve",
    "calibration_error",
    "calibration_file",
    "candidate_family",
    "chase_start_features",
    "chase_start_states",
    "fit_calibrator",
    "fit_candidate",
    "fit_innings",
    "innings_break_gap",
    "innings_end_par",
    "load_fold_models",
    "load_wp_models",
    "m3_features",
    "m3_labels",
    "pre_ball_table",
    "restore_candidate",
    "roc_auc",
    "run_stage",
    "save_fold_models",
    "terminal_rules",
    "win_probability",
    "wp_metrics",
    "wp_table",
]


# ------------------------------------------------------------------------------------- constants

#: The innings with an M3 classifier.
M3_INNINGS: tuple[int, ...] = STATE_INNINGS

#: Model names of the two classifiers in the FitRegistry, by innings (fit ids from
#: ``leakage.fit_id_for``, such as ``m3_inn1_lgbm_outer_k1``), and of their fold-k centred
#: isotonic calibrators (``m3_inn1_cal_lgbm_outer_k1``).
M3_MODELS: Mapping[int, str] = MappingProxyType({1: "m3_inn1", 2: "m3_inn2"})
CALIBRATOR_MODELS: Mapping[int, str] = MappingProxyType({1: "m3_inn1_cal", 2: "m3_inn2_cal"})

#: The pre-ball features of each innings, in the column order of every fit (criteria 3 and 4).
#: Index a DataFrame with ``list(M3_FEATURES[innings])``: pandas reads a tuple as one column key.
M3_FEATURES: Mapping[int, tuple[str, ...]] = MappingProxyType(
    {
        1: ("lb_bowled", "score", "wickets", "par_score", "tsg", "scoring_env", "venue_effect", "sched_balls"),
        2: ("runs_needed", "lb_left", "wickets", "rbq", "rr_minus_ar", "tsg"),
    }
)

#: The monotone constraint of each feature of M3_FEATURES[innings], in order (criterion 5):
#: innings 1 non-decreasing in score and non-increasing in wickets; innings 2 non-increasing in
#: runs needed and wickets and non-decreasing in Legal_Balls left and RBQ; free otherwise.
M3_MONOTONE: Mapping[int, tuple[int, ...]] = MappingProxyType(
    {1: (0, 1, -1, 0, 0, 0, 0, 0), 2: (-1, 1, -1, 1, 0, 0)}
)

#: Where each feature comes from: the ``state`` table (same column name), a sandbox view (feature
#: -> view column) or the m3_labels table (same column name).
STATE_FEATURES: tuple[str, ...] = ("lb_bowled", "score", "wickets", "runs_needed", "lb_left")
VIEW_FEATURES: Mapping[str, str] = MappingProxyType(
    {
        "par_score": "par_score",
        "tsg": "tsg",
        "scoring_env": "scoring_env",
        "venue_effect": "venue_effect",
        "rbq": "rbq",
        "rr_minus_ar": F19_COLUMN,  # Required_Rate minus Achievable_Rate on innings-2 rows
    }
)
LABEL_FEATURES: tuple[str, ...] = ("sched_balls",)

#: Result types whose matches train M3 and enter its metrics (criterion 2), and the others.
KEPT_RESULT_TYPES: tuple[str, ...] = (WIN_BY_RUNS, WIN_BY_WICKETS)
DROPPED_RESULT_TYPES: tuple[str, ...] = tuple(kind for kind in RESULT_TYPES if kind not in KEPT_RESULT_TYPES)

#: Columns of the m3_labels table, in order.
LABEL_COLUMNS: tuple[str, ...] = (
    "row", "match_id", "innings", "sched_balls", "result_type", "label", "labelled", "kept", "trainable", "eligible",
)  # fmt: skip

#: The Candidate_Model families of this module (FitRegistry family names), in the tie order of
#: Requirement 23, criterion 7: the logistic baseline first.
LGBM = "lgbm"
LOGISTIC = "logistic"
FAMILIES: tuple[str, ...] = (LOGISTIC, LGBM)

#: The M3 Alternative_Models of Requirement 23 (task 26.1), in the same tie order: XGBoost, then
#: the random forest. The ``model_choice`` stage fits them with fit_innings; FAMILIES above are
#: the families that the ``win_prob`` stage fits. The names match the Config sections
#: ``models.xgb.m3`` and ``models.rf.m3`` and the seeds ``seeds.m3_xgb`` and ``seeds.m3_rf``.
XGB = "xgb"
RF = "rf"
ALTERNATIVE_FAMILIES: tuple[str, ...] = (XGB, RF)

#: The ``kind`` of a calibrator row in FIT_TABLE_COLUMNS (the classifiers are folds.OUTER and
#: folds.INNER).
CALIBRATOR = "calibrator"

#: Columns of FoldModels.fold_metrics: one row per fold k, for the outer fit M3(-k) and the fold-k
#: calibrator (module docstring, "The fold-sandbox fits").
FOLD_METRICS_COLUMNS: tuple[str, ...] = (
    "innings", "family", "fold", "fit_id", "calibrator_fit_id", "iterations", "converged", "n_fit_rows",
    "n_early_stopping_rows", "n_held_out_rows", "n_calibration_rows", "in_fold_log_loss", "early_stopping_log_loss",
    "held_out_log_loss", "held_out_minus_in_fold", "held_out_log_loss_calibrated",
)  # fmt: skip

#: Columns of FoldModels.fit_table: one row per fit, classifiers and calibrators, in fit order.
FIT_TABLE_COLUMNS: tuple[str, ...] = (
    "fit_id", "model", "family", "kind", "sandbox_k", "inner_j", "n_train_rows", "n_fit_rows",
    "n_early_stopping_rows", "n_train_matches", "n_early_stopping_matches", "n_predicted_rows",
    "n_predicted_matches", "iterations", "converged", "seconds",
)  # fmt: skip

_CONFIG_MODEL = "m3"  # common.LGBM_MODELS / XGB_MODELS / RF_MODELS key: models.m3, models.xgb.m3, models.rf.m3
_LGBM_OBJECTIVE = "binary"
_XGB_OBJECTIVE = "binary"  # common.xgb_setup objective; the library objective is _XGB_LIBRARY_OBJECTIVE
_XGB_LIBRARY_OBJECTIVE = "binary:logistic"
_MAX_RANDOM_STATE = 2**32 - 1  # scikit-learn random_state
_LOGISTIC_SECTION = "models.m3.logistic"
_LOGISTIC_SEED = "seeds.m3_logistic"
_N_FOLDS = "design_choices.D15_n_folds"
_CALIBRATION_BOUNDS = "models.m3.calibration_bounds"  # [lower, upper] of every calibrator, all families
_MAX_SEED = 2**31 - 1
_FAMILY_NAME = re.compile(r"[a-z][a-z0-9_]{0,15}")  # fit ids such as m3_inn1_cal_<family>_inner_k1_j2 stay short
_SHOWN = 5  # examples named in one error message
_LABEL_BALL_COLUMNS: tuple[str, ...] = (
    "row", "match_id", "innings", "batting_team_id", "bowling_team_id", "sched_balls", "result_type",
    "is_super_over", "excluded",
)  # fmt: skip


# ---------------------------------------------------------------------------------- shared helpers

def _rows(count: int) -> str:
    return f"{count:,} row" if count == 1 else f"{count:,} rows"


def _require(frame: Any, columns: Sequence[str], what: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{what} must be a DataFrame; got {type(frame).__name__}")
    absent = [column for column in columns if column not in frame.columns]
    if absent:
        raise ValueError(f"{what} lacks the column(s) {', '.join(absent)}")


def _vector(values: Any, what: str) -> pd.Series:
    if values is None or isinstance(values, (str, bytes, Mapping, pd.DataFrame)) or np.ndim(values) != 1:
        raise TypeError(f"{what} must be a 1-D array or Series; got {type(values).__name__}")
    return values if isinstance(values, pd.Series) else pd.Series(np.asarray(values))


def _length(array: np.ndarray, what: str, n_rows: int | None) -> np.ndarray:
    if n_rows is not None and len(array) != n_rows:
        raise ValueError(f"{what} has {len(array):,} values, but {n_rows:,} are needed, one per row")
    return array


def _floats(values: Any, what: str, n_rows: int | None = None) -> np.ndarray:
    """``values`` (1-D, numeric) as a float64 array, NaN for a missing value; no infinite value."""
    series = _vector(values, what)
    if pd.api.types.is_bool_dtype(series.dtype) or not pd.api.types.is_numeric_dtype(series.dtype):
        raise TypeError(f"{what} must hold numbers; got dtype {series.dtype}")
    array = series.to_numpy(dtype=np.float64, na_value=np.nan)
    infinite = int(np.isinf(array).sum())
    if infinite:
        raise ValueError(f"{what} holds {infinite:,} infinite value(s); a value is a number or missing")
    return _length(array, what, n_rows)


def _ints(values: Any, what: str, n_rows: int | None = None) -> np.ndarray:
    """``values`` (1-D whole numbers, none missing) as an int64 array."""
    series = _vector(values, what)
    if pd.api.types.is_bool_dtype(series.dtype) or not pd.api.types.is_numeric_dtype(series.dtype):
        raise TypeError(f"{what} must hold whole numbers; got dtype {series.dtype}")
    missing = int(series.isna().sum())
    if missing:
        raise ValueError(f"{what} has {missing:,} missing value(s)")
    if pd.api.types.is_integer_dtype(series.dtype):
        array = series.to_numpy(dtype=np.int64)
    else:
        floats = series.to_numpy(dtype=np.float64)
        whole = np.isfinite(floats) & (np.floor(floats) == floats) & (np.abs(floats) <= 2.0**53)
        if not whole.all():
            raise ValueError(f"{what} must hold whole numbers; {int((~whole).sum()):,} value(s) are not")
        array = floats.astype(np.int64)
    return _length(array, what, n_rows)


def _flags(values: Any, what: str, n_rows: int | None = None) -> np.ndarray:
    """``values`` (1-D booleans, none missing) as a bool array."""
    series = _vector(values, what)
    if not pd.api.types.is_bool_dtype(series.dtype):
        raise TypeError(f"{what} must be boolean; got dtype {series.dtype}")
    missing = int(series.isna().sum())
    if missing:
        raise ValueError(f"{what} has {missing:,} missing value(s)")
    return _length(series.to_numpy(dtype=bool), what, n_rows)


def _binary(values: Any, what: str, n_rows: int | None = None) -> np.ndarray:
    """Labels of 0 and 1 (or booleans) as an int8 array."""
    series = _vector(values, what)
    if pd.api.types.is_bool_dtype(series.dtype):
        return _length(_flags(series, what).astype(np.int8), what, n_rows)
    labels = _ints(series, what, n_rows)
    if not np.isin(labels, (0, 1)).all():
        raise ValueError(f"{what} must hold only 0 and 1; {int((~np.isin(labels, (0, 1))).sum()):,} value(s) do not")
    return labels.astype(np.int8)


def _whole(value: Any, what: str, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{what} must be a whole number; got {value!r}")
    if value < minimum or (maximum is not None and value > maximum):
        bounds = f"{minimum} or more" if maximum is None else f"between {minimum} and {maximum}"
        raise ValueError(f"{what} must be {bounds}; got {value}")
    return int(value)


def _check_innings(innings: Any) -> int:
    if isinstance(innings, (bool, np.bool_)) or not isinstance(innings, Integral) or int(innings) not in M3_FEATURES:
        raise ValueError(f"innings must be one of {', '.join(map(str, M3_INNINGS))}; got {innings!r}")
    return int(innings)


def _same_rows(frame: pd.DataFrame, state: pd.DataFrame, what: str) -> None:
    """ValueError unless ``frame`` holds the rows of ``state`` in the same order (by ``row`` when both have it)."""
    if len(frame) != len(state):
        raise ValueError(
            f"{what} has {_rows(len(frame))} but state has {_rows(len(state))}; M3 tables hold one row per innings "
            "1-2 Delivery of the state table, in its order"
        )
    if "row" in frame.columns and "row" in state.columns:
        mine, theirs = _ints(frame["row"], f"{what} column 'row'"), _ints(state["row"], "state column 'row'")
        if not np.array_equal(mine, theirs):
            raise ValueError(
                f"{what} and state hold different rows, or the same rows in another order, at "
                f"{_rows(int((mine != theirs).sum()))}; M3 tables follow the row order of the state table"
            )


def _feature_frame(X: Any, features: Sequence[str]) -> pd.DataFrame:
    """``X``'s ``features`` as a new float64 DataFrame, in that order, on a RangeIndex."""
    _require(X, features, "X")
    data = {name: _floats(X[name], f"X column {name!r}") for name in features}
    return pd.DataFrame(data, columns=list(features))


def _probabilities(values: Any, n_rows: int, what: str) -> np.ndarray:
    """Model output as a float64 array of finite probabilities, one per row."""
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (n_rows,):
        raise ValueError(f"{what} returned shape {array.shape} for {_rows(n_rows)}; one probability per row is needed")
    bad = ~np.isfinite(array) | (array < 0.0) | (array > 1.0)
    if bad.any():
        raise ValueError(f"{what} returned {int(bad.sum()):,} value(s) that are not finite probabilities in [0, 1]")
    return array


class InvalidPrediction(ValueError):
    """Raw output of a fitted XGB or RF candidate that is not a finite probability (Requirement 23, criterion 12).

    ``what`` names the model; ``n_rows``: the rows it predicted; ``n_non_finite``: the values that
    are missing or infinite; ``n_outside``: the finite values outside [0, 1]. A ValueError, so
    every caller that handles the ValueError of the other families handles it too; the
    Model_Choice_Checker reads the counts to record the condition the candidate met.
    """

    def __init__(self, what: str, n_rows: int, n_non_finite: int, n_outside: int) -> None:
        self.what = str(what)
        self.n_rows, self.n_non_finite, self.n_outside = int(n_rows), int(n_non_finite), int(n_outside)
        super().__init__(
            f"{self.what} returned {self.n_non_finite:,} missing or infinite value(s) and {self.n_outside:,} "
            f"value(s) outside [0, 1] for {_rows(self.n_rows)}"
        )

    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), (self.what, self.n_rows, self.n_non_finite, self.n_outside))


def _checked_probabilities(values: Any, n_rows: int, what: str) -> np.ndarray:
    """An XGB or RF candidate's raw output as float64, one per row; InvalidPrediction unless each is in [0, 1]."""
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (n_rows,):
        raise ValueError(f"{what} returned shape {array.shape} for {_rows(n_rows)}; one probability per row is needed")
    finite = np.isfinite(array)
    outside = finite & ((array < 0.0) | (array > 1.0))
    if not finite.all() or outside.any():
        raise InvalidPrediction(what, n_rows, int((~finite).sum()), int(outside.sum()))
    return array


def _raw_input(raw: Any) -> np.ndarray:
    """Raw predictions given to a calibrator, as float64; ValueError unless each is a probability."""
    values = _floats(raw, "raw")
    bad = np.isnan(values) | (values < 0.0) | (values > 1.0)
    if bad.any():
        raise ValueError(f"raw holds {int(bad.sum()):,} value(s) that are missing or outside [0, 1]")
    return values


def _frozen(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


# ============================================================================ labels (10.1, 10.2)

def m3_labels(balls: pd.DataFrame, state: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """The M3 label and Training_Row flags of every innings 1-2 Delivery (criteria 1 and 2; module docstring).

    ``balls`` is the ``balls`` table of the ``reconstruct`` checkpoint (``row``, ``match_id``,
    ``innings``, ``batting_team_id``, ``bowling_team_id``, ``sched_balls``, ``result_type``,
    ``is_super_over``, ``excluded``); ``state`` the ``state`` table (``row``); ``matches`` the
    ``matches`` table of the ``load`` checkpoint (``match_id``, ``win_team_id``).

    Returns LABEL_COLUMNS on a RangeIndex, one row per ``state`` row in ``state`` order: ``row``,
    ``match_id``, ``innings`` and ``sched_balls`` (int64); ``result_type`` (str, UNPARSED where
    missing); ``label`` (int8: 1 when ``win_team_id`` is the batting side's team, else 0); and the
    bool flags ``labelled``, ``kept``, ``trainable`` and ``eligible``.

    Raises ValueError when a column is absent or malformed, ``balls`` repeats a row, a ``state``
    row has no Delivery in ``balls`` or lies outside innings 1 and 2, a Delivery's match is not in
    ``matches``, ``matches`` repeats a match or a result type is unknown; TypeError for a wrongly
    typed column.
    """
    _require(balls, _LABEL_BALL_COLUMNS, "balls")
    _require(state, ("row",), "state")
    _require(matches, ("match_id", "win_team_id"), "matches")
    index = pd.Index(_ints(balls["row"], "balls column 'row'"))
    if not index.is_unique:
        raise ValueError("balls must hold each row once")
    at = index.get_indexer(_ints(state["row"], "state column 'row'"))
    if (at < 0).any():
        raise ValueError(f"{_rows(int((at < 0).sum()))} of state have no Delivery in balls; pass the tables of one Run")
    part = balls.iloc[at]
    match = _ints(part["match_id"], "balls column 'match_id'")
    innings = _ints(part["innings"], "balls column 'innings'")
    outside = ~np.isin(innings, M3_INNINGS)
    if outside.any():
        raise ValueError(f"{_rows(int(outside.sum()))} of state lie outside innings 1 and 2")
    sched = _ints(part["sched_balls"], "balls column 'sched_balls'")
    batting = _floats(part["batting_team_id"], "balls column 'batting_team_id'")
    bowling = _floats(part["bowling_team_id"], "balls column 'bowling_team_id'")
    result = part["result_type"].astype(object).where(part["result_type"].notna(), UNPARSED).to_numpy(dtype=object)
    unknown = ~np.isin(result, RESULT_TYPES)
    if unknown.any():
        shown = ", ".join(sorted({str(value) for value in result[unknown]})[:_SHOWN])
        raise ValueError(f"balls column 'result_type' holds values that are not Result_Parser types: {shown}")

    match_index = pd.Index(_ints(matches["match_id"], "matches column 'match_id'"))
    if not match_index.is_unique:
        raise ValueError("matches must hold each match once")
    where = match_index.get_indexer(match)
    if (where < 0).any():
        shown = ", ".join(map(str, np.unique(match[where < 0])[:_SHOWN].tolist()))
        raise ValueError(
            f"{_rows(int((where < 0).sum()))} belong to matches that are not in matches (for example {shown})"
        )
    winner = _floats(matches["win_team_id"], "matches column 'win_team_id'")[where]

    label = (winner == batting).astype(np.int8)  # NaN compares unequal, so an unknown side or winner gives 0
    labelled = ~np.isnan(batting) & ~np.isnan(winner) & ((winner == batting) | (winner == bowling))
    kept = np.isin(result, KEPT_RESULT_TYPES)
    trainable = common.training_mask(part)
    return pd.DataFrame(
        {
            "row": _ints(part["row"], "balls column 'row'"),
            "match_id": match,
            "innings": innings,
            "sched_balls": sched,
            "result_type": pd.array(result.astype(str), dtype="str"),
            "label": label,
            "labelled": labelled,
            "kept": kept,
            "trainable": trainable,
            "eligible": trainable & kept & labelled,
        },
        columns=list(LABEL_COLUMNS),
    )


# ======================================================================= features (10.3, 10.4)

def m3_features(innings: int, view: pd.DataFrame, state: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """The M3_FEATURES of every Delivery of ``innings`` (criteria 3 and 4; module docstring, "Features").

    ``view`` is a sandbox view V_k (the VIEW_FEATURES columns), ``state`` the ``state`` table (the
    STATE_FEATURES of the innings) and ``labels`` the m3_labels table (``innings`` and
    ``sched_balls``), all with the same rows in the same order; when they have ``row``, the
    columns must be equal. Returns M3_FEATURES[innings] as float64 columns, in that order, on a
    RangeIndex, one row per row of the innings in ``state`` order; a missing value is NaN.

    Raises ValueError when a column is absent or the rows differ, and TypeError for a wrongly
    typed column.
    """
    innings = _check_innings(innings)
    names = M3_FEATURES[innings]
    _require(view, [VIEW_FEATURES[name] for name in names if name in VIEW_FEATURES], "view")
    _require(state, [name for name in names if name in STATE_FEATURES], "state")
    _require(labels, ("innings", *(name for name in names if name in LABEL_FEATURES)), "labels")
    _same_rows(view, state, "view")
    _same_rows(labels, state, "labels")
    mine = np.flatnonzero(_ints(labels["innings"], "labels column 'innings'") == innings)
    data: dict[str, np.ndarray] = {}
    for name in names:
        if name in STATE_FEATURES:
            column = state[name].iloc[mine]
        elif name in VIEW_FEATURES:
            column = view[VIEW_FEATURES[name]].iloc[mine]
        else:
            column = labels[name].iloc[mine]
        data[name] = _floats(column, f"feature {name!r}")
    return pd.DataFrame(data, columns=list(names))


# =========================================================================== Candidate_Models

class Candidate(Protocol):
    """One M3 classifier of one family for one innings (module docstring, "Candidate_Model families").

    ``family`` is the family name; ``features`` the innings' M3_FEATURES, the columns
    ``predict_raw`` reads. ``fit`` trains it once on a fit's Training_Rows (``early_stopping`` marks
    the rows of the whole matches held out for early stopping, or is None for a family without
    it); ``predict_raw`` returns the uncalibrated probability that the batting side wins, one per
    row. ``model`` is the fitted library object that ``CheckpointStore.save_model`` saves;
    ``iterations`` and ``converged`` describe how the fit ended (None when not known).
    """

    family: str
    features: tuple[str, ...]

    @property
    def model(self) -> Any: ...

    @property
    def iterations(self) -> int | None: ...

    @property
    def converged(self) -> bool | None: ...

    def fit(self, X: pd.DataFrame, y: Any, groups: Any, seed: int, *, early_stopping: Any = None) -> None: ...

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray: ...


def _check_features(features: Any) -> tuple[str, ...]:
    names = tuple(features) if isinstance(features, (tuple, list)) else None
    if not names or not all(isinstance(name, str) and name for name in names) or len(set(names)) != len(names):
        raise ValueError(f"features must be a non-empty sequence of distinct column names; got {features!r}")
    return names


def _training_data(
    X: Any, y: Any, groups: Any, features: tuple[str, ...]
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frame = _feature_frame(X, features)
    if not len(frame):
        raise ValueError("a fit needs at least one Training_Row")
    labels = _binary(y, "y", len(frame))
    if len(np.unique(labels)) < 2:
        raise ValueError(f"the Training_Rows hold only label {int(labels[0])}; a classifier needs both labels 0 and 1")
    return frame, labels, _ints(groups, "groups", len(frame))


def _held_out(early_stopping: Any, match_ids: np.ndarray) -> np.ndarray:
    """The early-stopping flags as a bool array, checked to mark whole matches on both sides."""
    held = _flags(early_stopping, "early_stopping", len(match_ids))
    inside, outside = np.unique(match_ids[held]), np.unique(match_ids[~held])
    if not len(inside) or not len(outside):
        raise ValueError("early_stopping must hold out some matches and leave others to fit on")
    split = np.intersect1d(inside, outside)
    if len(split):
        raise ValueError(
            f"early_stopping must mark whole matches (criterion 7), but {len(split):,} match(es) lie on both sides "
            f"(for example {', '.join(map(str, split[:_SHOWN].tolist()))})"
        )
    return held


class LgbmCandidate:
    """The LightGBM M3 classifier of one innings (criteria 5-7; module docstring, "Candidate_Model families").

    ``LgbmCandidate(features, setup)`` is a new candidate: ``setup`` the BoosterSetup of
    ``common.lgbm_setup(settings, "m3", "binary", monotone_constraints=M3_MONOTONE[innings])`` and
    ``features`` M3_FEATURES[innings], in the order of the constraints. ``LgbmCandidate(features,
    model=booster)`` wraps a fitted LightGBM Booster, such as one that ``CheckpointStore.load_model``
    returns; it predicts but cannot be fitted again.
    """

    family = LGBM

    def __init__(self, features: Sequence[str], setup: common.BoosterSetup | None = None, *, model: Any = None) -> None:
        self.features = _check_features(features)
        if (setup is None) == (model is None):
            raise ValueError(
                "give either setup (a new candidate) or model (a fitted LightGBM Booster), not both or neither"
            )
        if setup is not None:
            if not isinstance(setup, common.BoosterSetup) or setup.library != common.LIGHTGBM:
                raise TypeError(f"setup must be a LightGBM BoosterSetup from common.lgbm_setup; got {setup!r}")
            params = setup.params
            constraints = tuple(params.get("monotone_constraints", ()))
            if params.get("objective") != _LGBM_OBJECTIVE or len(constraints) != len(self.features):
                raise ValueError(
                    "setup must be a binary LightGBM setup with one monotone constraint per feature "
                    f"({len(self.features)}); got objective {params.get('objective')!r} and {len(constraints)} "
                    "constraint(s)"
                )
        if model is not None:
            import lightgbm as lgb

            if not isinstance(model, lgb.Booster):
                raise TypeError(f"model must be a fitted LightGBM Booster; got {type(model).__name__}")
            if list(model.feature_name()) != list(self.features):
                raise ValueError(f"the Booster was fitted on {model.feature_name()}, not on {list(self.features)}")
        self._setup = setup
        self._model = model

    def __repr__(self) -> str:
        state = "unfitted" if self._model is None else f"best round {self.iterations}"
        return f"LgbmCandidate(features={list(self.features)}, {state})"

    @property
    def model(self) -> Any:
        """The fitted LightGBM Booster (None before fit), which predicts with its best round."""
        return self._model

    @property
    def iterations(self) -> int | None:
        """The boosting rounds kept: the best round of early stopping."""
        return None if self._model is None else int(self._model.best_iteration)

    @property
    def converged(self) -> bool | None:
        """True when early stopping ended the fit before the round cap (None for a wrapped model)."""
        if self._model is None or self._setup is None:
            return None
        return self.iterations + self._setup.early_stopping.patience <= self._setup.max_rounds

    def fit(self, X: pd.DataFrame, y: Any, groups: Any, seed: int, *, early_stopping: Any = None) -> None:
        """Fit on the Training_Rows ``X``, ``y`` (0 or 1), ``groups`` (match ids), with LightGBM seed ``seed``.

        ``early_stopping`` (required) marks the rows of the whole matches held out for early
        stopping; the model trains on the other rows for at most the setup's round cap and keeps
        its best round on the held-out log loss (criterion 7). Raises ValueError when the
        candidate was fitted or wrapped before, the rows hold only one label, the flags are
        missing or split a match, and TypeError for malformed arguments.
        """
        if self._setup is None or self._model is not None:
            raise ValueError("a LightGBM candidate is fitted once; make a new one with its family's make")
        frame, labels, match_ids = _training_data(X, y, groups, self.features)
        if early_stopping is None:
            raise ValueError(
                "a LightGBM candidate needs early-stopping rows: whole matches of its Training_Rows (criterion 7)"
            )
        held = _held_out(early_stopping, match_ids)
        params = self._setup.params
        params["seed"] = _whole(seed, "seed", 0, _MAX_SEED)
        setup = common.BoosterSetup(
            library=common.LIGHTGBM,
            param_items=tuple(params.items()),
            max_rounds=self._setup.max_rounds,
            early_stopping=self._setup.early_stopping,
        )
        rows = common.FitRows(train=np.flatnonzero(~held), early_stopping=np.flatnonzero(held))
        self._model = common.boosting_fitter(setup, frame, labels)(rows)

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        """The raw probability that the batting side wins, for each row of ``X`` (its best round)."""
        if self._model is None:
            raise ValueError("the LightGBM candidate is not fitted")
        frame = _feature_frame(X, self.features)
        if not len(frame):
            return np.zeros(0, dtype=np.float64)
        return _probabilities(self._model.predict(frame), len(frame), "the LightGBM Booster")


class LogisticCandidate:
    """The logistic-regression baseline of one innings (criterion 10; module docstring, "Candidate_Model families").

    ``LogisticCandidate(features, C=..., max_iter=...)`` is a new candidate with the Config's
    ``models.m3.logistic.C`` and ``max_iter``; ``LogisticCandidate(features, model=pipeline)``
    wraps a fitted pipeline, such as one that ``CheckpointStore.load_model`` returns.
    """

    family = LOGISTIC

    def __init__(
        self, features: Sequence[str], *, C: float | None = None, max_iter: int | None = None, model: Any = None
    ) -> None:
        self.features = _check_features(features)
        if model is None:
            if isinstance(C, (bool, np.bool_)) or not isinstance(C, Real) or not np.isfinite(C) or C <= 0:
                raise ValueError(f"C ({_LOGISTIC_SECTION}.C) must be a positive number; got {C!r}")
            self._C, self._max_iter = float(C), _whole(max_iter, f"max_iter ({_LOGISTIC_SECTION}.max_iter)", 1)
        else:
            if C is not None or max_iter is not None:
                raise ValueError("a wrapped logistic pipeline brings its own C and max_iter")
            if not callable(getattr(model, "predict_proba", None)) or not hasattr(model, "classes_"):
                raise TypeError(f"model must be a fitted scikit-learn classifier pipeline; got {type(model).__name__}")
            names = list(getattr(model, "feature_names_in_", []))
            if names != list(self.features):
                raise ValueError(f"the pipeline was fitted on {names}, not on {list(self.features)}")
            self._C, self._max_iter = None, None
        self._model = model
        self._fitted_here = False

    def __repr__(self) -> str:
        state = "unfitted" if self._model is None else f"{self.iterations} lbfgs iterations"
        return f"LogisticCandidate(features={list(self.features)}, {state})"

    @property
    def model(self) -> Any:
        """The fitted scikit-learn pipeline (None before fit): imputer with indicators, scaler, logistic regression."""
        return self._model

    @property
    def iterations(self) -> int | None:
        """The lbfgs iterations of the fit."""
        return None if self._model is None else int(np.max(self._model[-1].n_iter_))

    @property
    def converged(self) -> bool | None:
        """True when lbfgs stopped before ``max_iter``."""
        return None if self._model is None else self.iterations < int(self._model[-1].max_iter)

    def fit(self, X: pd.DataFrame, y: Any, groups: Any, seed: int, *, early_stopping: Any = None) -> None:
        """Fit on the Training_Rows ``X``, ``y`` (0 or 1), ``groups`` (match ids), with ``random_state=seed``.

        The medians, missing-value indicators and scaling are fitted on these rows only. The
        baseline uses no early stopping: ``early_stopping`` must be None or mark no row. Raises
        ValueError when the candidate was fitted or wrapped before or the rows hold only one
        label, and TypeError for malformed arguments.
        """
        if self._model is not None:
            raise ValueError("a logistic candidate is fitted once; make a new one with its family's make")
        frame, labels, _ = _training_data(X, y, groups, self.features)
        if early_stopping is not None and _flags(early_stopping, "early_stopping", len(frame)).any():
            raise ValueError("the logistic baseline uses no early stopping; fit it on every Training_Row")
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
                (
                    "logit",
                    LogisticRegression(
                        C=self._C, max_iter=self._max_iter, random_state=_whole(seed, "seed", 0, 2**32 - 1)
                    ),
                ),
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)  # reported through ``converged``
            pipeline.fit(frame, labels)
        self._model = pipeline

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        """The raw probability that the batting side wins, for each row of ``X``."""
        if self._model is None:
            raise ValueError("the logistic candidate is not fitted")
        frame = _feature_frame(X, self.features)
        if not len(frame):
            return np.zeros(0, dtype=np.float64)
        column = list(self._model.classes_).index(1)
        return _probabilities(self._model.predict_proba(frame)[:, column], len(frame), "the logistic pipeline")


class XgbCandidate:
    """The XGBoost M3 Alternative_Model of one innings (Requirement 23, criteria 1, 2 and 4; module docstring).

    ``XgbCandidate(features, setup)`` is a new candidate: ``setup`` the BoosterSetup of
    ``common.xgb_setup(settings, "m3", "binary", monotone_constraints=M3_MONOTONE[innings])`` and
    ``features`` M3_FEATURES[innings], in the order of the constraints. ``XgbCandidate(features,
    model=booster)`` wraps a fitted XGBoost Booster, such as one that ``CheckpointStore.load_model``
    returns; it predicts but cannot be fitted again.
    """

    family = XGB

    def __init__(self, features: Sequence[str], setup: common.BoosterSetup | None = None, *, model: Any = None) -> None:
        self.features = _check_features(features)
        if (setup is None) == (model is None):
            raise ValueError(
                "give either setup (a new candidate) or model (a fitted XGBoost Booster), not both or neither"
            )
        if setup is not None:
            if not isinstance(setup, common.BoosterSetup) or setup.library != common.XGBOOST:
                raise TypeError(f"setup must be an XGBoost BoosterSetup from common.xgb_setup; got {setup!r}")
            params = setup.params
            constraints = tuple(params.get("monotone_constraints", ()))
            if params.get("objective") != _XGB_LIBRARY_OBJECTIVE or len(constraints) != len(self.features):
                raise ValueError(
                    "setup must be a binary XGBoost setup with one monotone constraint per feature "
                    f"({len(self.features)}); got objective {params.get('objective')!r} and {len(constraints)} "
                    "constraint(s)"
                )
        if model is not None:
            import xgboost as xgb

            if not isinstance(model, xgb.Booster):
                raise TypeError(f"model must be a fitted XGBoost Booster; got {type(model).__name__}")
            if list(model.feature_names or ()) != list(self.features):
                raise ValueError(f"the Booster was fitted on {model.feature_names}, not on {list(self.features)}")
        self._setup = setup
        self._model = model

    def __repr__(self) -> str:
        state = "unfitted" if self._model is None else f"{self.iterations} rounds"
        return f"XgbCandidate(features={list(self.features)}, {state})"

    @property
    def model(self) -> Any:
        """The fitted XGBoost Booster (None before fit), cut to its best round."""
        return self._model

    @property
    def iterations(self) -> int | None:
        """The boosting rounds kept: those up to the best round of early stopping, all the Booster holds."""
        return None if self._model is None else int(self._model.num_boosted_rounds())

    @property
    def converged(self) -> bool | None:
        """True when early stopping ended the fit before the round cap (None for a wrapped model)."""
        if self._model is None or self._setup is None:
            return None
        return self.iterations + self._setup.early_stopping.patience <= self._setup.max_rounds

    def fit(self, X: pd.DataFrame, y: Any, groups: Any, seed: int, *, early_stopping: Any = None) -> None:
        """Fit on the Training_Rows ``X``, ``y`` (0 or 1), ``groups`` (match ids), with XGBoost seed ``seed``.

        ``early_stopping`` (required) marks the rows of the whole matches held out for early
        stopping; the model trains on the other rows for at most the setup's round cap and keeps
        its best round on the held-out log loss (Requirement 23, criterion 4). Raises ValueError
        when the candidate was fitted or wrapped before, the rows hold only one label, the flags
        are missing or split a match, and TypeError for malformed arguments.
        """
        if self._setup is None or self._model is not None:
            raise ValueError("an XGBoost candidate is fitted once; make a new one with its family's make")
        frame, labels, match_ids = _training_data(X, y, groups, self.features)
        if early_stopping is None:
            raise ValueError(
                "an XGBoost candidate needs early-stopping rows: whole matches of its Training_Rows (Requirement 23, "
                "criterion 4)"
            )
        held = _held_out(early_stopping, match_ids)
        params = self._setup.params
        params["seed"] = _whole(seed, "seed", 0, _MAX_SEED)
        setup = common.BoosterSetup(
            library=common.XGBOOST,
            param_items=tuple(params.items()),
            max_rounds=self._setup.max_rounds,
            early_stopping=self._setup.early_stopping,
        )
        rows = common.FitRows(train=np.flatnonzero(~held), early_stopping=np.flatnonzero(held))
        self._model = common.boosting_fitter(setup, frame, labels)(rows)

    def _predict(self, frame: pd.DataFrame) -> np.ndarray:
        """The Booster's output for ``frame`` (the features, in order), before any check."""
        return self._model.inplace_predict(frame)

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        """The raw probability that the batting side wins, for each row of ``X``; InvalidPrediction unless in [0, 1]."""
        if self._model is None:
            raise ValueError("the XGBoost candidate is not fitted")
        frame = _feature_frame(X, self.features)
        if not len(frame):
            return np.zeros(0, dtype=np.float64)
        return _checked_probabilities(self._predict(frame), len(frame), "the XGBoost Booster")


class RfCandidate:
    """The random-forest M3 Alternative_Model of one innings (Requirement 23, criteria 1, 2 and 4; module docstring).

    ``RfCandidate(features, setup)`` is a new candidate: ``setup`` the ForestSetup of
    ``common.rf_setup(settings, "m3", monotonic_cst=M3_MONOTONE[innings])`` and ``features``
    M3_FEATURES[innings], in the order of the constraints. ``RfCandidate(features, model=pipeline)``
    wraps a fitted pipeline (median imputer, random forest), such as one that
    ``CheckpointStore.load_model`` returns; it predicts but cannot be fitted again.
    """

    family = RF

    def __init__(self, features: Sequence[str], setup: common.ForestSetup | None = None, *, model: Any = None) -> None:
        self.features = _check_features(features)
        if (setup is None) == (model is None):
            raise ValueError(
                "give either setup (a new candidate) or model (a fitted random-forest pipeline), not both or neither"
            )
        if setup is not None:
            if not isinstance(setup, common.ForestSetup):
                raise TypeError(f"setup must be a ForestSetup from common.rf_setup; got {setup!r}")
            constraints = tuple(setup.params.get("monotonic_cst", ()))
            if len(constraints) != len(self.features):
                raise ValueError(
                    f"setup must hold one monotonic_cst value per feature ({len(self.features)}); got "
                    f"{len(constraints)}"
                )
        if model is not None:
            if not callable(getattr(model, "predict_proba", None)) or not hasattr(model, "classes_"):
                raise TypeError(f"model must be a fitted scikit-learn classifier pipeline; got {type(model).__name__}")
            names = list(getattr(model, "feature_names_in_", []))
            if names != list(self.features):
                raise ValueError(f"the pipeline was fitted on {names}, not on {list(self.features)}")
        self._setup = setup
        self._model = model

    def __repr__(self) -> str:
        state = "unfitted" if self._model is None else f"{self.iterations} trees"
        return f"RfCandidate(features={list(self.features)}, {state})"

    @property
    def model(self) -> Any:
        """The fitted scikit-learn pipeline (None before fit): median imputer, then the random forest."""
        return self._model

    @property
    def iterations(self) -> int | None:
        """The trees of the forest."""
        return None if self._model is None else len(self._model[-1].estimators_)

    @property
    def converged(self) -> bool | None:
        """None: a random forest has no stopping rule."""
        return None

    def fit(self, X: pd.DataFrame, y: Any, groups: Any, seed: int, *, early_stopping: Any = None) -> None:
        """Fit on the Training_Rows ``X``, ``y`` (0 or 1), ``groups`` (match ids), with ``random_state=seed``.

        The medians are fitted on these rows only. The forest uses no early stopping:
        ``early_stopping`` must be None or mark no row. Raises ValueError when the candidate was
        fitted or wrapped before or the rows hold only one label, TypeError for malformed
        arguments, and whatever scikit-learn raises for a setting it refuses.
        """
        if self._setup is None or self._model is not None:
            raise ValueError("a random-forest candidate is fitted once; make a new one with its family's make")
        frame, labels, _ = _training_data(X, y, groups, self.features)
        if early_stopping is not None and _flags(early_stopping, "early_stopping", len(frame)).any():
            raise ValueError("the random forest uses no early stopping; fit it on every Training_Row")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline

        params = self._setup.params
        params["random_state"] = _whole(seed, "seed", 0, _MAX_RANDOM_STATE)
        pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("forest", RandomForestClassifier(**params)),
            ]
        )
        pipeline.fit(frame, labels)
        pipeline[-1].set_params(n_jobs=self._setup.predict_n_jobs)  # 1 thread in deterministic mode
        self._model = pipeline

    def _predict(self, frame: pd.DataFrame) -> np.ndarray:
        """The forest's probability of label 1 for ``frame`` (the features, in order), before any check."""
        column = list(self._model.classes_).index(1)
        return self._model.predict_proba(frame)[:, column]

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        """The raw probability that the batting side wins, for each row of ``X``; InvalidPrediction unless in [0, 1]."""
        if self._model is None:
            raise ValueError("the random-forest candidate is not fitted")
        frame = _feature_frame(X, self.features)
        if not len(frame):
            return np.zeros(0, dtype=np.float64)
        return _checked_probabilities(self._predict(frame), len(frame), "the random-forest pipeline")


def _restore_lgbm(innings: int, model: Any) -> Candidate:
    return LgbmCandidate(M3_FEATURES[_check_innings(innings)], model=model)


def _restore_logistic(innings: int, model: Any) -> Candidate:
    return LogisticCandidate(M3_FEATURES[_check_innings(innings)], model=model)


def _restore_xgb(innings: int, model: Any) -> Candidate:
    return XgbCandidate(M3_FEATURES[_check_innings(innings)], model=model)


def _restore_rf(innings: int, model: Any) -> Candidate:
    return RfCandidate(M3_FEATURES[_check_innings(innings)], model=model)


_RESTORE: Mapping[str, Callable[[int, Any], Candidate]] = MappingProxyType(
    {LGBM: _restore_lgbm, LOGISTIC: _restore_logistic, XGB: _restore_xgb, RF: _restore_rf}
)
_ALL_FAMILIES: tuple[str, ...] = (*FAMILIES, *ALTERNATIVE_FAMILIES)


def restore_candidate(family: str, innings: int, model: Any) -> Candidate:
    """A fitted candidate of a family of this module that wraps a saved ``model`` (load_fold_models)."""
    if family not in _RESTORE:
        raise ValueError(f"family must be one of {', '.join(map(repr, _ALL_FAMILIES))}; got {family!r}")
    return _RESTORE[family](innings, model)


@dataclass(frozen=True, eq=False)
class CandidateFamily:
    """A Candidate_Model family (module docstring, "Candidate_Model families").

    ``name``: the FitRegistry family and the family part of every fit id (``lgbm``,
    ``logistic``; task 26.1 adds its own). ``seed``: the family's named seed, given to every
    ``Candidate.fit``. ``early_stopping``: the EarlyStopping whose share, patience and seed
    ``common.fit_registered`` uses to hold out whole matches, or None for a family without early
    stopping. ``make(innings)``: a new unfitted Candidate for innings 1 or 2. ``restore(innings,
    model)``: a fitted Candidate that wraps a saved model.
    """

    name: str
    seed: int
    early_stopping: common.EarlyStopping | None
    make: Callable[[int], Candidate]
    restore: Callable[[int, Any], Candidate]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _FAMILY_NAME.fullmatch(self.name):
            raise ValueError(
                "a family name must be up to 16 lower-case letters, digits and '_', starting with a letter, so that "
                f"its fit ids can name checkpoint files; got {self.name!r}"
            )
        object.__setattr__(self, "seed", _whole(self.seed, f"the seed of family {self.name!r}", 0, _MAX_SEED))
        if self.early_stopping is not None and not isinstance(self.early_stopping, common.EarlyStopping):
            raise TypeError(f"early_stopping must be a common.EarlyStopping or None; got {self.early_stopping!r}")
        if not callable(self.make) or not callable(self.restore):
            raise TypeError("make and restore must be functions")


def candidate_family(name: str, settings: Settings) -> CandidateFamily:
    """The CandidateFamily LGBM, LOGISTIC, XGB or RF with its Config settings (module docstring, "Candidate_Model families").

    LGBM reads ``models.m3`` (through ``common.lgbm_setup``), ``runtime`` and ``seeds.m3_lgbm``;
    LOGISTIC reads ``models.m3.logistic.C``, ``models.m3.logistic.max_iter`` and
    ``seeds.m3_logistic``; XGB reads ``models.xgb.m3`` (``common.xgb_setup``), ``runtime`` and
    ``seeds.m3_xgb``; RF reads ``models.rf.m3`` (``common.rf_setup``), ``runtime`` and
    ``seeds.m3_rf``. Every family gets the monotone constraints M3_MONOTONE except the logistic
    baseline. Raises ValueError for another name or an out-of-range setting, KeyError for a
    setting absent from the Config and TypeError for a wrongly typed one; a random-forest setting
    that scikit-learn refuses raises only when a forest is fitted.
    """
    if name == LGBM:
        setups = {
            innings: common.lgbm_setup(
                settings, _CONFIG_MODEL, _LGBM_OBJECTIVE, monotone_constraints=M3_MONOTONE[innings]
            )
            for innings in M3_INNINGS
        }
        first = setups[M3_INNINGS[0]]
        return CandidateFamily(
            name=LGBM,
            seed=int(first.params["seed"]),
            early_stopping=first.early_stopping,
            make=lambda innings: LgbmCandidate(M3_FEATURES[_check_innings(innings)], setups[_check_innings(innings)]),
            restore=_restore_lgbm,
        )
    if name == LOGISTIC:
        C = settings.get(f"{_LOGISTIC_SECTION}.C")
        max_iter = settings.get(f"{_LOGISTIC_SECTION}.max_iter")
        LogisticCandidate(M3_FEATURES[M3_INNINGS[0]], C=C, max_iter=max_iter)  # checks both settings now
        return CandidateFamily(
            name=LOGISTIC,
            seed=_whole(settings.get(_LOGISTIC_SEED), _LOGISTIC_SEED, 0, _MAX_SEED),
            early_stopping=None,
            make=lambda innings: LogisticCandidate(M3_FEATURES[_check_innings(innings)], C=C, max_iter=max_iter),
            restore=_restore_logistic,
        )
    if name == XGB:
        boosters = {
            innings: common.xgb_setup(
                settings, _CONFIG_MODEL, _XGB_OBJECTIVE, monotone_constraints=M3_MONOTONE[innings]
            )
            for innings in M3_INNINGS
        }
        first_booster = boosters[M3_INNINGS[0]]
        return CandidateFamily(
            name=XGB,
            seed=int(first_booster.params["seed"]),
            early_stopping=first_booster.early_stopping,
            make=lambda innings: XgbCandidate(M3_FEATURES[_check_innings(innings)], boosters[_check_innings(innings)]),
            restore=_restore_xgb,
        )
    if name == RF:
        forests = {
            innings: common.rf_setup(settings, _CONFIG_MODEL, monotonic_cst=M3_MONOTONE[innings])
            for innings in M3_INNINGS
        }
        return CandidateFamily(
            name=RF,
            seed=int(forests[M3_INNINGS[0]].params["random_state"]),
            early_stopping=None,
            make=lambda innings: RfCandidate(M3_FEATURES[_check_innings(innings)], forests[_check_innings(innings)]),
            restore=_restore_rf,
        )
    raise ValueError(f"name must be one of {', '.join(map(repr, _ALL_FAMILIES))}; got {name!r}")


def fit_candidate(
    family: CandidateFamily, innings: int, X: pd.DataFrame, y: Any, groups: Any, *, fit_index: int
) -> Candidate:
    """One fit of ``family`` on rows that are all Training_Rows, without a FitRegistry (tests and checks).

    ``X`` (the M3_FEATURES of ``innings``), ``y`` and ``groups`` (the match id of each row)
    describe the same rows. A family with early stopping holds out the whole matches of
    ``leakage.early_stopping_split(groups, share, seed, fit_index)``, exactly as
    ``common.fit_registered`` holds them out for the same rows of the FitSpec at position
    ``fit_index`` of ``folds.sandbox_plan(D15)``. A fit whose output reaches a Delivery must go
    through fit_innings instead, so that the leakage guard checks it. Returns the fitted
    Candidate; raises the errors of its ``fit`` and of ``early_stopping_split``.
    """
    if not isinstance(family, CandidateFamily):
        raise TypeError(f"family must be a CandidateFamily (candidate_family); got {type(family).__name__}")
    match_ids = _ints(groups, "groups")
    held = None
    if family.early_stopping is not None:
        early = family.early_stopping
        index = _whole(fit_index, "fit_index", 0)
        _, held_matches = leakage.early_stopping_split(match_ids, early.share, early.seed, index)
        held = np.isin(match_ids, held_matches)
    candidate = family.make(_check_innings(innings))
    candidate.fit(X, y, match_ids, family.seed, early_stopping=held)
    return candidate


# ============================================================================ calibration (10.9)

def _check_bounds(bounds: Any, what: str) -> tuple[float, float]:
    """``bounds`` as floats (lower, upper); TypeError unless two numbers, ValueError unless 0 <= lower < upper <= 1."""
    if isinstance(bounds, (str, bytes)) or not isinstance(bounds, Sequence) or len(bounds) != 2:
        raise TypeError(f"{what} must be a [lower, upper] pair; got {bounds!r}")
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) for value in bounds):
        raise TypeError(f"{what} must hold two numbers; got {bounds!r}")
    lower, upper = float(bounds[0]), float(bounds[1])
    if not 0.0 <= lower < upper <= 1.0:  # also False for NaN
        raise ValueError(f"{what} must satisfy 0 <= lower < upper <= 1; got {list(bounds)}")
    return lower, upper


def calibration_bounds(settings: Settings) -> tuple[float, float]:
    """``models.m3.calibration_bounds`` as floats (lower, upper): the output range of every M3 calibrator.

    The same bounds serve every Candidate_Model family (module docstring, "Calibration"). Raises
    ValueError unless 0 <= lower < upper <= 1, TypeError for a value that is not a pair of
    numbers and KeyError when the setting is absent from the Config.
    """
    return _check_bounds(settings.get(_CALIBRATION_BOUNDS), _CALIBRATION_BOUNDS)


def _knot_values(values: Any, what: str) -> np.ndarray:
    """``values`` as a new 1-D float64 array of at least one finite number; TypeError or ValueError otherwise."""
    if values is None or isinstance(values, (str, bytes, Mapping, pd.DataFrame)) or np.ndim(values) != 1:
        raise TypeError(f"{what} must be a 1-D array of numbers; got {type(values).__name__}")
    try:
        array = np.array(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{what} must hold numbers: {exc}") from None
    if not len(array):
        raise ValueError(f"{what} needs at least one knot")
    if not np.isfinite(array).all():
        raise ValueError(f"{what} holds {int((~np.isfinite(array)).sum()):,} value(s) that are not finite")
    return array


@dataclass(frozen=True, eq=False)
class CentredIsotonic:
    """The centred isotonic calibrator of criterion 9 (module docstring, "Calibration"; design DD-14).

    ``x_knots``: the knots on the raw-prediction scale, within [0, 1] and strictly increasing;
    ``y_knots``: the calibrated WP at each knot, strictly increasing and within ``bounds``;
    ``bounds``: (lower, upper), the ``models.m3.calibration_bounds`` it was fitted with. Both
    arrays are read-only float64 copies of the values given, with at least one knot.
    fit_calibrator builds it: one knot per block of the bounded isotonic fit, at the block's
    centroid and fitted value. ``predict(raw)`` interpolates linearly between consecutive knots,
    keeps the end knots' values beyond them and clips to ``bounds``. The object is frozen, and
    its pickle (joblib, CheckpointStore.save_model) rebuilds it through the constructor, so a
    loaded calibrator is checked again. Raises TypeError or ValueError for knots or bounds that
    break these rules.
    """

    x_knots: np.ndarray
    y_knots: np.ndarray
    bounds: tuple[float, float]

    def __post_init__(self) -> None:
        lower, upper = _check_bounds(self.bounds, "bounds")
        x, y = _knot_values(self.x_knots, "x_knots"), _knot_values(self.y_knots, "y_knots")
        if len(x) != len(y):
            raise ValueError(f"x_knots has {len(x):,} value(s) and y_knots {len(y):,}; each knot needs both")
        if (np.diff(x) <= 0.0).any() or (np.diff(y) <= 0.0).any():
            raise ValueError("x_knots and y_knots must both be strictly increasing, one knot per isotonic block")
        if x[0] < 0.0 or x[-1] > 1.0:
            raise ValueError(
                "x_knots must lie within [0, 1], the range of a raw prediction; "
                f"got [{float(x[0])!r}, {float(x[-1])!r}]"
            )
        if y[0] < lower or y[-1] > upper:
            raise ValueError(
                f"y_knots must lie within the bounds [{lower!r}, {upper!r}] ({_CALIBRATION_BOUNDS}); "
                f"got [{float(y[0])!r}, {float(y[-1])!r}]"
            )
        object.__setattr__(self, "x_knots", _frozen(x))
        object.__setattr__(self, "y_knots", _frozen(y))
        object.__setattr__(self, "bounds", (lower, upper))

    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), (np.array(self.x_knots), np.array(self.y_knots), self.bounds))

    def __repr__(self) -> str:
        x, y = self.x_knots, self.y_knots
        return (
            f"CentredIsotonic({len(x):,} knot(s) on raw [{x[0]:.4g}, {x[-1]:.4g}], WP [{y[0]:.4g}, {y[-1]:.4g}] "
            f"within [{self.bounds[0]:g}, {self.bounds[1]:g}])"
        )

    def predict(self, raw: Any) -> np.ndarray:
        """The calibrated WP of each raw prediction in ``raw``, as float64 within ``bounds``.

        Linear between consecutive knots (``np.interp``), each value also kept within the WP
        range of its segment so that float rounding cannot make the output decrease at a knot;
        the first (last) knot's value below (above) the knots; constant with a single knot.
        Raises ValueError when a raw value is missing or not a probability.
        """
        values = _raw_input(raw)
        x, y = self.x_knots, self.y_knots
        if len(x) == 1:
            calibrated = np.full(len(values), y[0])
        else:
            segment = np.clip(np.searchsorted(x, values, side="right") - 1, 0, len(x) - 2)
            calibrated = np.clip(np.interp(values, x, y), y[segment], y[segment + 1])
        return np.clip(calibrated, self.bounds[0], self.bounds[1])


def _check_calibrator(calibrator: Any) -> tuple[float, float]:
    """The bounds (lower, upper) of a fit_calibrator result; TypeError for any other object."""
    if not isinstance(calibrator, CentredIsotonic):
        raise TypeError(
            f"calibrator must be the CentredIsotonic of criterion 9 (fit_calibrator); got {type(calibrator).__name__}"
        )
    return calibrator.bounds


def fit_calibrator(raw: Any, y: Any, bounds: Sequence[float]) -> CentredIsotonic:
    """The centred isotonic calibrator of criterion 9, fitted on raw predictions ``raw`` and labels ``y`` (0 or 1).

    ``bounds`` is the pair (lower, upper) of ``calibration_bounds(settings)``, with 0 <= lower <
    upper <= 1. Fits ``IsotonicRegression(y_min=lower, y_max=upper, increasing=True,
    out_of_bounds="clip")`` on them and returns a CentredIsotonic with one knot per block of that
    fit (a maximal run of sorted training predictions with one fitted value, ties in the raw
    values grouped as scikit-learn groups them): x the mean raw prediction of the block's rows,
    kept within the block's raw range, and y the block's fitted value (module docstring,
    "Calibration"). Every calibrated value lies in [lower, upper]. Raises ValueError when the
    bounds break that rule, a prediction is missing or not a probability, a label is not 0 or 1,
    or the lengths differ, and TypeError for bounds that are not a pair of numbers.
    """
    from sklearn.isotonic import IsotonicRegression

    lower, upper = _check_bounds(bounds, "bounds")
    values = _raw_input(raw)
    if not len(values):
        raise ValueError("a calibrator needs at least one prediction")
    labels = _binary(y, "y", len(values))
    isotonic = IsotonicRegression(y_min=lower, y_max=upper, increasing=True, out_of_bounds="clip")
    isotonic.fit(values, labels.astype(np.float64))
    # X_thresholds_ holds the distinct raw values (scikit-learn's tie groups, strictly increasing)
    # that the fit keeps: the first and the last of each run of equal fitted values. The runs of
    # y_thresholds_ are therefore the blocks; a training value belongs to the last block that
    # starts at or below it, because the next block starts above every value of its tie group.
    thresholds = np.asarray(isotonic.X_thresholds_, dtype=np.float64)
    fitted = np.asarray(isotonic.y_thresholds_, dtype=np.float64)
    starts = np.flatnonzero(np.r_[True, fitted[1:] != fitted[:-1]])
    first, last = thresholds[starts], thresholds[np.r_[starts[1:], len(fitted)] - 1]
    block = np.searchsorted(first, values, side="right") - 1
    rows = np.bincount(block, minlength=len(starts))
    centroid = np.bincount(block, weights=values, minlength=len(starts)) / rows
    return CentredIsotonic(np.clip(centroid, first, last), fitted[starts], (lower, upper))


def calibrate(calibrator: Any, raw: Any) -> np.ndarray:
    """``calibrator``'s calibrated WP for each raw prediction in ``raw``, as float64 (CentredIsotonic.predict).

    ``calibrator`` is a fit_calibrator result; every value lies within the calibration bounds it
    was fitted with. Raises ValueError when a raw value is missing or not a probability, and
    TypeError for another calibrator.
    """
    _check_calibrator(calibrator)
    return calibrator.predict(raw)


class CalibratedModel:
    """A fitted Candidate and its centred isotonic calibrator (module docstring, "Calibration").

    ``predict(X)`` returns the calibrated WP of each row of ``X`` (the innings' M3_FEATURES
    columns, any order), within the calibrator's bounds and before the terminal rules;
    ``predict_raw(X)`` the candidate's raw output. ``calibrator`` is a CentredIsotonic.
    """

    __slots__ = ("calibrator", "candidate")

    def __init__(self, candidate: Candidate, calibrator: CentredIsotonic) -> None:
        if not callable(getattr(candidate, "predict_raw", None)) or candidate.model is None:
            raise TypeError(f"candidate must be a fitted Candidate; got {candidate!r}")
        _check_calibrator(calibrator)
        self.candidate = candidate
        self.calibrator = calibrator

    def __repr__(self) -> str:
        return f"CalibratedModel({self.candidate!r}, {self.calibrator!r})"

    @property
    def family(self) -> str:
        return self.candidate.family

    @property
    def features(self) -> tuple[str, ...]:
        return self.candidate.features

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        return self.candidate.predict_raw(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return calibrate(self.calibrator, self.candidate.predict_raw(X))


# =============================================================================== fold models

@dataclass(frozen=True, eq=False)
class CalibratedFolds:
    """The calibrated outer model of every fold of one innings and family (module docstring).

    ``models[k]`` is M3(-k) with the fold-k calibrator: a CalibratedModel whose ``calibrator`` is
    the CentredIsotonic fitted on the inner predictions of S_k. ``predict(X, fold)`` gives the
    calibrated WP of any rows (``X`` with the innings' M3_FEATURES columns), each from the model
    of its ``fold``, the fold of the row's match from the Fold_Table; ``predict_raw`` the raw
    output.
    """

    innings: int
    family: str
    models: Mapping[int, CalibratedModel]

    def __post_init__(self) -> None:
        object.__setattr__(self, "innings", _check_innings(self.innings))
        if not isinstance(self.models, Mapping) or not self.models:
            raise ValueError("models must map each fold to its CalibratedModel")
        for fold, model in self.models.items():
            if not isinstance(model, CalibratedModel) or model.features != M3_FEATURES[self.innings]:
                raise TypeError(f"models[{fold!r}] must be a CalibratedModel of innings {self.innings}")
        object.__setattr__(self, "models", MappingProxyType(dict(sorted(self.models.items()))))

    def _by_fold(self, X: pd.DataFrame, fold: Any, method: str) -> np.ndarray:
        _require(X, M3_FEATURES[self.innings], "X")
        folds_of_rows = _ints(fold, "fold", len(X))
        unknown = ~np.isin(folds_of_rows, list(self.models))
        if unknown.any():
            raise ValueError(
                f"{_rows(int(unknown.sum()))} lie in folds without a model; the folds are {list(self.models)}"
            )
        out = np.full(len(X), np.nan)
        for k, model in self.models.items():
            rows = np.flatnonzero(folds_of_rows == k)
            if len(rows):
                out[rows] = getattr(model, method)(X.iloc[rows])
        return out

    def predict(self, X: pd.DataFrame, fold: Any) -> np.ndarray:
        return self._by_fold(X, fold, "predict")

    def predict_raw(self, X: pd.DataFrame, fold: Any) -> np.ndarray:
        return self._by_fold(X, fold, "predict_raw")


@dataclass(frozen=True, eq=False)
class FoldModels:
    """Everything fit_innings produced for one innings and family (module docstring, "The fold-sandbox fits").

    Per row of the innings, in ``state`` order: ``row`` (the Delivery's ``row`` id), ``match_id``,
    ``fold``, ``label``, ``eligible``, the outer fit's raw output ``raw`` and the calibrated
    Out_Of_Fold WP ``wp``. Per fold k: ``inner_raw[k]`` (the inner fits' raw predictions of the
    rows outside fold k, NaN on fold-k rows), ``calibration_rows[k]`` (positions of the fold-k
    calibrator's Training_Rows), ``outer_fit_ids[k]`` and ``calibrator_fit_ids[k]``. Per (k, j):
    ``inner[(k, j)]`` (the fitted inner Candidate) and ``inner_fit_ids[(k, j)]``. ``calibrated``:
    the calibrated outer models. ``fold_metrics`` (FOLD_METRICS_COLUMNS), ``fit_table``
    (FIT_TABLE_COLUMNS) and ``seconds``, the wall time of fit_innings. Arrays are read-only.
    """

    innings: int
    family: str
    features: tuple[str, ...]
    row: np.ndarray
    match_id: np.ndarray
    fold: np.ndarray
    label: np.ndarray
    eligible: np.ndarray
    raw: np.ndarray
    wp: np.ndarray
    inner_raw: Mapping[int, np.ndarray]
    calibration_rows: Mapping[int, np.ndarray]
    calibrated: CalibratedFolds
    outer_fit_ids: Mapping[int, str]
    calibrator_fit_ids: Mapping[int, str]
    inner: Mapping[tuple[int, int], Candidate]
    inner_fit_ids: Mapping[tuple[int, int], str]
    fold_metrics: pd.DataFrame
    fit_table: pd.DataFrame
    seconds: float

    def __post_init__(self) -> None:
        for name in ("row", "match_id", "fold", "label", "eligible", "raw", "wp"):
            _frozen(getattr(self, name))
        for name in ("inner_raw", "calibration_rows"):
            values = {key: _frozen(array) for key, array in getattr(self, name).items()}
            object.__setattr__(self, name, MappingProxyType(values))
        for name in ("outer_fit_ids", "calibrator_fit_ids", "inner", "inner_fit_ids"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    def predict(self, X: pd.DataFrame, fold: Any) -> np.ndarray:
        """The calibrated WP of any rows, each from the calibrated outer model of its fold (CalibratedFolds.predict)."""
        return self.calibrated.predict(X, fold)


def _check_views(view_by_k: Any, all_folds: tuple[int, ...]) -> None:
    if not isinstance(view_by_k, Mapping):
        raise TypeError(f"view_by_k must map each fold k to view V_k; got {type(view_by_k).__name__}")
    if sorted(view_by_k) != list(all_folds):
        raise ValueError(f"view_by_k must hold the views of folds {list(all_folds)}; got folds {sorted(view_by_k)}")


def _candidate_fitter(
    family: CandidateFamily, innings: int, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray
) -> common.FitFunction:
    """The fit function of one classifier fit, for ``common.fit_registered``: a new Candidate fitted on the rows."""

    def fit(rows: common.FitRows) -> Candidate:
        training = rows.training
        held = np.isin(training, rows.early_stopping) if family.early_stopping is not None else None
        candidate = family.make(innings)
        candidate.fit(X.iloc[training], y[training], groups[training], family.seed, early_stopping=held)
        return candidate

    return fit


def _calibrator_fitter(inner_raw: np.ndarray, y: np.ndarray, bounds: tuple[float, float]) -> common.FitFunction:
    """The fit function of one fold-k calibrator, for ``common.fit_registered``: the centred isotonic
    calibrator (fit_calibrator) of the inner predictions of its Training_Rows, bounded to ``bounds``
    (calibration_bounds). The same function serves every Candidate_Model family."""

    def fit(rows: common.FitRows) -> CentredIsotonic:
        if len(rows.early_stopping):
            raise ValueError("a calibrator uses no early stopping; call fit_registered with early_stopping=None")
        values = inner_raw[rows.train]
        missing = int((~np.isfinite(values)).sum())
        if missing:
            raise ValueError(f"{_rows(missing)} of the calibrator's Training_Rows have no inner-fit prediction")
        return fit_calibrator(values, y[rows.train], bounds)

    return fit


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    return common.pooled_log_loss(y, p) if len(y) else float("nan")


def _fit_row(
    fitted: common.RegisteredFit,
    registry: FitRegistry,
    kind: str,
    n_predicted: int,
    iterations: int | None,
    converged: bool | None,
    seconds: float,
) -> dict[str, Any]:
    record = registry.get(fitted.fit_id)
    return {
        "fit_id": fitted.fit_id,
        "model": record.model,
        "family": record.family,
        "kind": kind,
        "sandbox_k": record.sandbox_k,
        "inner_j": record.inner_j,
        "n_train_rows": len(fitted.rows.training),
        "n_fit_rows": len(fitted.rows.train),
        "n_early_stopping_rows": len(fitted.rows.early_stopping),
        "n_train_matches": len(record.training_match_ids),
        "n_early_stopping_matches": len(record.early_stopping_match_ids),
        "n_predicted_rows": n_predicted,
        "n_predicted_matches": len(record.predicted_match_ids),
        "iterations": iterations,
        "converged": converged,
        "seconds": seconds,
    }


def fit_innings(
    innings: int,
    family: str | CandidateFamily,
    view_by_k: Mapping[int, pd.DataFrame],
    labels: pd.DataFrame,
    settings: Settings,
    registry: FitRegistry,
    *,
    state: pd.DataFrame,
) -> FoldModels:
    """The D15 x D15 sandbox fits and D15 calibrators of one innings and family (module docstring).

    innings: 1 or 2. family: LGBM, LOGISTIC, XGB, RF or a CandidateFamily. view_by_k: view
    V_k of every fold k, as the ``remaining_runs`` checkpoint saves it (one row per ``state`` row,
    in ``state`` order, with ``m1_fit`` and ``m2_fit``). labels: the m3_labels table. settings:
    the Config (``design_choices.D15_n_folds``, ``models.m3.calibration_bounds`` and the
    family's settings). registry: the Run's FitRegistry, holding the M1 and M2 fits that the
    views record. state: the ``state`` table. Every family gets the same calibrators: centred
    isotonic (fit_calibrator), bounded to ``calibration_bounds(settings)``.

    Returns a FoldModels. Raises ViewLeakageError or FitLeakageError (exit code 4) when a view or
    a fit breaks the leakage guard, in which case the fit is not run; DataError (exit code 3)
    for a match not in the Fold_Table; ValueError or TypeError for tables that do not fit
    together, a malformed setting (calibration bounds included, checked before any fit) or a fit
    that fails, such as Training_Rows with one label.
    """
    started = time.perf_counter()
    innings = _check_innings(innings)
    bounds = calibration_bounds(settings)  # the default for every family, before any fit runs
    fam = family if isinstance(family, CandidateFamily) else candidate_family(family, settings)
    plan = folds.sandbox_plan(settings.get(_N_FOLDS))
    outer_specs = {spec.sandbox_k: spec for spec in plan if spec.kind == folds.OUTER}
    all_folds = tuple(sorted(outer_specs))
    _check_views(view_by_k, all_folds)
    _require(labels, LABEL_COLUMNS, "labels")
    _require(state, ("row",), "state")
    _same_rows(labels, state, "labels")

    every_match = _ints(labels["match_id"], "labels column 'match_id'")
    every_fold = registry.fold_of(every_match).astype(np.int64)  # the Fold_Table (Requirement 7, criterion 5)
    mine = np.flatnonzero(_ints(labels["innings"], "labels column 'innings'") == innings)
    if not len(mine):
        raise ValueError(f"labels hold no Delivery of innings {innings}")
    match_ids, fold = every_match[mine], every_fold[mine]
    y = _binary(labels["label"], "labels column 'label'")[mine]
    eligible = _flags(labels["eligible"], "labels column 'eligible'")[mine]
    model, calibrator_model = M3_MODELS[innings], CALIBRATOR_MODELS[innings]

    n_rows = len(mine)
    raw, wp = np.full(n_rows, np.nan), np.full(n_rows, np.nan)
    inner_raw: dict[int, np.ndarray] = {}
    calibration_rows: dict[int, np.ndarray] = {}
    calibrated: dict[int, CalibratedModel] = {}
    outer_ids: dict[int, str] = {}
    calibrator_ids: dict[int, str] = {}
    inner: dict[tuple[int, int], Candidate] = {}
    inner_ids: dict[tuple[int, int], str] = {}
    fit_rows: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for k in all_folds:
        view = view_by_k[k]
        _same_rows(view, state, f"view V_{k}")
        views.check_view(k, view, every_fold, registry)  # before any fit of sandbox k reads V_k
        X = m3_features(innings, view, state, labels)
        inner_k = np.full(n_rows, np.nan)
        outer: common.RegisteredFit | None = None
        for spec in (spec for spec in plan if spec.sandbox_k == k):  # M3(-k), then every M3^(k)(-j)
            served = np.flatnonzero(fold == spec.predicted_fold)
            began = time.perf_counter()
            fitted = common.fit_registered(
                registry,
                spec,
                model=model,
                family=fam.name,
                match_ids=match_ids,
                eligible=eligible,
                predicted_match_ids=match_ids[served],
                early_stopping=fam.early_stopping,
                fit=_candidate_fitter(fam, innings, X, y, match_ids),
            )
            candidate = fitted.model
            prediction = candidate.predict_raw(X.iloc[served])
            if spec.kind == folds.OUTER:
                raw[served], outer = prediction, fitted
                outer_ids[k] = fitted.fit_id
            else:
                inner_k[served] = prediction
                inner[(k, spec.inner_j)], inner_ids[(k, spec.inner_j)] = candidate, fitted.fit_id
            fit_rows.append(
                _fit_row(
                    fitted, registry, spec.kind, len(served), candidate.iterations, candidate.converged,
                    time.perf_counter() - began,
                )
            )
        if outer is None:
            raise ValueError(f"folds.sandbox_plan holds no outer fit of fold {k}")

        target = np.flatnonzero(fold == k)
        began = time.perf_counter()
        calibration = common.fit_registered(
            registry,
            outer_specs[k],
            model=calibrator_model,
            family=fam.name,
            match_ids=match_ids,
            eligible=eligible,  # its train folds are those of M3(-k): the S_k rows
            predicted_match_ids=match_ids[target],
            early_stopping=None,
            fit=_calibrator_fitter(inner_k, y, bounds),
        )
        wp[target] = calibrate(calibration.model, raw[target])
        seconds = time.perf_counter() - began
        fit_rows.append(_fit_row(calibration, registry, CALIBRATOR, len(target), None, None, seconds))
        calibrated[k] = CalibratedModel(outer.model, calibration.model)
        calibrator_ids[k] = calibration.fit_id
        inner_raw[k], calibration_rows[k] = inner_k, np.array(calibration.rows.train)

        candidate = outer.model
        fitted_on, held_for_stopping = outer.rows.train, outer.rows.early_stopping
        held = target[eligible[target]]
        in_fold = _log_loss(y[fitted_on], candidate.predict_raw(X.iloc[fitted_on]))
        held_out = _log_loss(y[held], raw[held])
        metrics.append(
            {
                "innings": innings,
                "family": fam.name,
                "fold": k,
                "fit_id": outer.fit_id,
                "calibrator_fit_id": calibration.fit_id,
                "iterations": candidate.iterations,
                "converged": candidate.converged,
                "n_fit_rows": len(fitted_on),
                "n_early_stopping_rows": len(held_for_stopping),
                "n_held_out_rows": len(held),
                "n_calibration_rows": len(calibration.rows.train),
                "in_fold_log_loss": in_fold,
                "early_stopping_log_loss": _log_loss(
                    y[held_for_stopping], candidate.predict_raw(X.iloc[held_for_stopping])
                ),
                "held_out_log_loss": held_out,
                "held_out_minus_in_fold": held_out - in_fold,
                "held_out_log_loss_calibrated": _log_loss(y[held], wp[held]),
            }
        )
        del view, X

    unfilled = int((~np.isfinite(raw) | ~np.isfinite(wp)).sum())
    if unfilled:
        raise ValueError(
            f"{_rows(unfilled)} of innings {innings} got no Out_Of_Fold_Prediction; every fold needs its fits"
        )
    fold_metrics = pd.DataFrame.from_records(metrics, columns=list(FOLD_METRICS_COLUMNS)).astype(
        {"family": "str", "fit_id": "str", "calibrator_fit_id": "str", "iterations": "Int64", "converged": "boolean"}
    )
    fit_table = pd.DataFrame.from_records(fit_rows, columns=list(FIT_TABLE_COLUMNS)).astype(
        {
            "fit_id": "str",
            "model": "str",
            "family": "str",
            "kind": "str",
            "sandbox_k": "Int32",
            "inner_j": "Int32",
            "iterations": "Int64",
            "converged": "boolean",
        }
    )
    return FoldModels(
        innings=innings,
        family=fam.name,
        features=M3_FEATURES[innings],
        row=_ints(labels["row"], "labels column 'row'")[mine],
        match_id=match_ids,
        fold=fold,
        label=y,
        eligible=eligible,
        raw=raw,
        wp=wp,
        inner_raw=inner_raw,
        calibration_rows=calibration_rows,
        calibrated=CalibratedFolds(innings=innings, family=fam.name, models=calibrated),
        outer_fit_ids=outer_ids,
        calibrator_fit_ids=calibrator_ids,
        inner=inner,
        inner_fit_ids=inner_ids,
        fold_metrics=fold_metrics,
        fit_table=fit_table,
        seconds=time.perf_counter() - started,
    )


# ================================================================================ saved models

def save_fold_models(
    store: CheckpointStore, stage: str, fold_models: FoldModels, *, inner: bool = True
) -> tuple[str, ...]:
    """Save the fold models of ``fold_models`` in ``stage``'s checkpoint, each under its fit id; return the names.

    Saves every outer candidate's model and every calibrator, then (``inner=True``) every inner
    candidate's model, with ``store.save_model``: a LightGBM Booster as model text with its best
    round, a logistic pipeline and a CentredIsotonic calibrator with joblib. load_fold_models
    reads the outer models and calibrators back.
    """
    if not isinstance(fold_models, FoldModels):
        raise TypeError(f"fold_models must be the FoldModels of fit_innings; got {type(fold_models).__name__}")
    if not isinstance(inner, (bool, np.bool_)):
        raise TypeError(f"inner must be True or False; got {inner!r}")
    saved: list[str] = []
    for k, model in fold_models.calibrated.models.items():
        parts = (
            (fold_models.outer_fit_ids[k], model.candidate.model),
            (fold_models.calibrator_fit_ids[k], model.calibrator),
        )
        for name, value in parts:
            store.save_model(stage, name, value)
            saved.append(name)
    if inner:
        for key in sorted(fold_models.inner_fit_ids):
            name = fold_models.inner_fit_ids[key]
            store.save_model(stage, name, fold_models.inner[key].model)
            saved.append(name)
    return tuple(saved)


def load_fold_models(
    store: CheckpointStore, stage: str, innings: int, family: str | CandidateFamily, n_folds: int
) -> CalibratedFolds:
    """The calibrated outer models of folds 1 to ``n_folds`` (D15) that save_fold_models saved in ``stage``.

    Each fold k reads the candidate model saved as ``leakage.fit_id_for(M3_MODELS[innings], outer
    spec k, family=name)`` and the calibrator saved as ``leakage.fit_id_for(CALIBRATOR_MODELS[innings],
    outer spec k, family=name)``. ``family`` is LGBM, LOGISTIC, XGB, RF or a CandidateFamily, whose
    ``restore`` wraps the saved model (the ``model_choice`` checkpoint holds those of XGB and RF). Each calibrator is a CentredIsotonic and keeps the knots and
    the bounds it was fitted with. Raises StageError for a missing or changed file, TypeError for
    a saved calibrator that is not a CentredIsotonic (such as the IsotonicRegression of a
    checkpoint saved before the centred isotonic calibrator; ``stage`` must run again), and
    TypeError or ValueError for another saved object that is not the expected model.
    """
    innings = _check_innings(innings)
    if isinstance(family, CandidateFamily):
        name, restore = family.name, family.restore
    elif family in _RESTORE:
        name, restore = family, _RESTORE[family]
    else:
        raise ValueError(
            f"family must be one of {', '.join(map(repr, _ALL_FAMILIES))} or a CandidateFamily; got {family!r}"
        )
    models: dict[int, CalibratedModel] = {}
    for spec in folds.sandbox_plan(n_folds):
        if spec.kind != folds.OUTER:
            continue
        candidate = restore(innings, store.load_model(stage, leakage.fit_id_for(M3_MODELS[innings], spec, family=name)))
        calibrator_id = leakage.fit_id_for(CALIBRATOR_MODELS[innings], spec, family=name)
        calibrator = store.load_model(stage, calibrator_id)
        if not isinstance(calibrator, CentredIsotonic):
            raise TypeError(
                f"the calibrator {calibrator_id!r} saved in stage {stage!r} is a {type(calibrator).__name__}, not the "
                f"CentredIsotonic of criterion 9; a checkpoint saved before the centred isotonic calibrator cannot be "
                f"used, so run the {stage} stage again"
            )
        models[spec.sandbox_k] = CalibratedModel(candidate, calibrator)
    return CalibratedFolds(innings=innings, family=name, models=models)


# ===================================================================== terminal rules (10.13-10.17)

#: Counted_Wickets that end an innings: 10, fixed by the Laws of Cricket rather than a
#: Method_Constant (criteria 15-17).
ALL_OUT_WICKETS = 10

#: The rule that flags a state (module docstring, "Terminal rules"): the int8 codes of
#: TerminalWP.rule, and their names, which the ``terminal_rule`` column of the WP tables holds.
NOT_TERMINAL = 0
CHASE_WON = 1  # innings 2, runs needed <= 0: WP 1 (criterion 14)
CHASE_LOST = 2  # innings 2, ended with runs needed > 1: WP 0 (criterion 15)
CHASE_TIED = 3  # innings 2, ended with runs needed = 1: WP 0.5 (criterion 16)
CHASE_FRACTION = 4  # innings 2, ended with 0 < runs needed < 1: WP 1 - runs needed / 2
INNINGS_END = 5  # innings 1, Scheduled_Balls bowled or 10 wickets: the innings-1 classifier's WP (criterion 17)
TERMINAL_RULES: Mapping[int, str] = MappingProxyType(
    {
        NOT_TERMINAL: "none",
        CHASE_WON: "chase_won",
        CHASE_LOST: "chase_lost",
        CHASE_TIED: "chase_tied",
        CHASE_FRACTION: "chase_fraction",
        INNINGS_END: "innings_end",
    }
)

#: The rules that set a state's WP instead of the classifier (criteria 14-16 and CHASE_FRACTION).
#: An INNINGS_END state is only flagged: the innings-1 classifier scores it (criterion 17).
DECIDING_RULES: tuple[int, ...] = (CHASE_WON, CHASE_LOST, CHASE_TIED, CHASE_FRACTION)

#: The state columns that terminal_rules reads in each innings; an M3_FEATURES frame of the
#: innings holds them.
TERMINAL_COLUMNS: Mapping[int, tuple[str, ...]] = MappingProxyType(
    {1: ("lb_bowled", "sched_balls", "wickets"), 2: ("runs_needed", "lb_left", "wickets")}
)

#: The rules that can flag a state of each innings.
_INNINGS_RULES: Mapping[int, tuple[int, ...]] = MappingProxyType({1: (INNINGS_END,), 2: DECIDING_RULES})
_RULE_NAMES = np.array([TERMINAL_RULES[code] for code in range(len(TERMINAL_RULES))], dtype=object)
_TIED_WP = 0.5  # criterion 16; CHASE_FRACTION runs from it (runs needed 1) to 1 (runs needed 0)


@dataclass(frozen=True, eq=False)
class TerminalWP:
    """The final WP of a set of states and the terminal rule of each (module docstring, "Terminal rules").

    ``wp``: float64, one value per state, within [0, 1]; ``rule``: the int8 TERMINAL_RULES code of
    each state: NOT_TERMINAL or INNINGS_END where the classifier's calibrated WP was kept, a code
    of DECIDING_RULES where the rule set the WP. Both are read-only.
    """

    wp: np.ndarray
    rule: np.ndarray

    def __post_init__(self) -> None:
        wp = np.array(self.wp, dtype=np.float64)
        rule = np.array(self.rule, dtype=np.int8)
        if wp.ndim != 1 or wp.shape != rule.shape:
            raise ValueError(f"wp and rule must be 1-D with one value per state; got shapes {wp.shape} and {rule.shape}")
        bad = ~np.isfinite(wp) | (wp < 0.0) | (wp > 1.0)
        if bad.any():
            raise ValueError(f"{int(bad.sum()):,} WP value(s) are missing or outside [0, 1] (criterion 13)")
        unknown = ~np.isin(rule, list(TERMINAL_RULES))
        if unknown.any():
            raise ValueError(f"{int(unknown.sum()):,} rule code(s) are not TERMINAL_RULES codes")
        object.__setattr__(self, "wp", _frozen(wp))
        object.__setattr__(self, "rule", _frozen(rule))

    @property
    def terminal(self) -> np.ndarray:
        """True for each state whose WP a rule of DECIDING_RULES set (never an INNINGS_END state)."""
        return np.isin(self.rule, DECIDING_RULES)

    @property
    def rule_names(self) -> np.ndarray:
        """The TERMINAL_RULES name of each state's rule, as an object array of str."""
        return _RULE_NAMES[self.rule]


def _state_values(states: pd.DataFrame, name: str, *, missing_ok: bool = False) -> np.ndarray:
    values = _floats(states[name], f"states column {name!r}")
    if not missing_ok:
        missing = int(np.isnan(values).sum())
        if missing:
            raise ValueError(
                f"states column {name!r} has {missing:,} missing value(s); the terminal rules need it on every state"
            )
    return values


def terminal_rules(innings: int, states: pd.DataFrame) -> np.ndarray:
    """The TERMINAL_RULES code (int8) of each state of ``innings`` (criteria 14-17; module docstring).

    ``states`` holds the TERMINAL_COLUMNS of the innings, one row per state, in any order: an
    M3_FEATURES frame of pre-ball states or of Post_Outcome_States, whose scores and runs needed
    may be fractional. An innings-1 state gets INNINGS_END when it ends the innings (a flag; the
    classifier still scores it) and an innings-2 state a code of DECIDING_RULES when the chase is
    decided. Raises ValueError for an absent or infinite column, or a missing value where a rule
    needs one (module docstring), and TypeError for a non-numeric column.
    """
    innings = _check_innings(innings)
    _require(states, TERMINAL_COLUMNS[innings], "states")
    rule = np.zeros(len(states), dtype=np.int8)
    wickets = _state_values(states, "wickets")
    if innings == 1:
        bowled = _state_values(states, "lb_bowled")
        scheduled = _state_values(states, "sched_balls")
        rule[(bowled >= scheduled) | (wickets >= ALL_OUT_WICKETS)] = INNINGS_END
        return rule
    needed = _state_values(states, "runs_needed", missing_ok=True)
    ended = (_state_values(states, "lb_left") <= 0.0) | (wickets >= ALL_OUT_WICKETS)
    unknown = ended & np.isnan(needed)
    if unknown.any():
        raise ValueError(
            f"{_rows(int(unknown.sum()))} of states end the chase (no Legal_Ball left or {ALL_OUT_WICKETS} wickets "
            "fallen) with runs_needed missing, so no terminal rule can decide them"
        )
    with np.errstate(invalid="ignore"):  # a missing runs_needed compares False: no rule
        rule[needed <= 0.0] = CHASE_WON
        rule[ended & (needed > 1.0)] = CHASE_LOST
        rule[ended & (needed == 1.0)] = CHASE_TIED
        rule[ended & (needed > 0.0) & (needed < 1.0)] = CHASE_FRACTION
    return rule


def _check_wp(values: np.ndarray, rows: np.ndarray, what: str) -> None:
    bad = rows & (np.isnan(values) | (values < 0.0) | (values > 1.0))
    if bad.any():
        raise ValueError(f"{what} is missing or outside [0, 1] on {_rows(int(bad.sum()))} where it is read")


def apply_terminal_rules(innings: int, states: pd.DataFrame, wp: Any) -> TerminalWP:
    """The final WP of each state of ``innings``: the WP of its deciding rule, else ``wp`` (criteria 13-17).

    ``states``: as for terminal_rules. ``wp``: the calibrated classifier WP of each state, within
    [0, 1] on every state that no rule of DECIDING_RULES decides; a decided state's value is not
    read and may be NaN. In innings 1 no rule decides: every state, INNINGS_END included, keeps
    ``wp`` (criterion 17). Returns a TerminalWP (module docstring, "Terminal rules"). Raises
    ValueError for a WP that is missing or outside [0, 1] where it is read, and the errors of
    terminal_rules.
    """
    innings = _check_innings(innings)
    rule = terminal_rules(innings, states)
    n_rows = len(rule)
    out = np.array(_floats(wp, "wp", n_rows), dtype=np.float64)
    _check_wp(out, ~np.isin(rule, DECIDING_RULES), "wp")
    if innings == 2:
        needed = _floats(states["runs_needed"], "states column 'runs_needed'")
        out[rule == CHASE_WON] = 1.0
        out[rule == CHASE_LOST] = 0.0
        out[rule == CHASE_TIED] = _TIED_WP
        fraction = rule == CHASE_FRACTION
        out[fraction] = 1.0 - (1.0 - _TIED_WP) * needed[fraction]  # 0.5 at 1 run needed, 1 at 0
    return TerminalWP(wp=out, rule=rule)


# ============================================================ the end-of-innings state (10.17)

def _fit_buckets(table: pd.DataFrame, fit_ids: np.ndarray, fit_id: str, columns: Sequence[str], what: str) -> pd.DataFrame:
    """The ``columns`` of one fit's rows of a per-bucket table (PAR_MEANS_TABLE or PAR_BIAS_TABLE), on the index ``bucket``."""
    part = table.iloc[np.flatnonzero(fit_ids == fit_id)]
    if part.empty:
        raise ValueError(
            f"{what} has no rows for the first-innings M2 fit {fit_id!r}; pass the {M2_STAGE} table of the same Run"
        )
    part = part.sort_values("bucket", kind="stable")
    index = pd.Index(_ints(part["bucket"], f"{what} column 'bucket'"), name="bucket")
    return pd.DataFrame({name: part[name].to_numpy() for name in columns}, index=index, columns=list(columns))


def innings_end_par(
    m2_fit: Any,
    scoring_env: Any,
    venue_effect: Any,
    sched_balls: Any,
    *,
    m2_models: Mapping[str, Any],
    par_means: pd.DataFrame,
    par_bias: pd.DataFrame,
) -> np.ndarray:
    """Par_Score at Scheduled_Balls of each first-innings row, the Par_Score of its end-of-innings state (criterion 17).

    ``m2_fit``: the first-innings M2 fit id of each row, from the ``m2_fit`` column of the view
    row that serves it; ``scoring_env``, ``venue_effect`` and ``sched_balls``: the row's par
    context (Scoring_Environment and Venue_Effect from that view row, Scheduled_Balls of the
    innings); ``m2_models``: fit id -> first-innings M2 model, for at least the fits named (the
    calibrated M2Models of ``remaining_runs.load_models``, which gave the views' Par_Score);
    ``par_means`` and ``par_bias``: the ``remaining_runs`` checkpoint tables PAR_MEANS_TABLE and
    PAR_BIAS_TABLE (``fit_id``, ``bucket`` and the PAR_MEAN_FEATURES or ``beta`` of each fit and
    bucket). For each fit, the corrected par curve of every distinct context of its rows
    (``remaining_runs.par_contexts``, ``par_curve``, ``corrected_par_curve``) is read at
    Scheduled_Balls (``par_lookup``): the curve that gave the rows' ``par_score`` and
    ``par_score_next`` in the view (module docstring, "The end-of-innings state").

    Returns float64, one value per row. Raises ValueError for a fit without a model or without
    par rows, TypeError for an argument of the wrong kind, and the errors of the remaining_runs
    functions.
    """
    fits = _text_values(m2_fit, "m2_fit")
    n_rows = len(fits)
    env = _floats(scoring_env, "scoring_env", n_rows)
    venue = _floats(venue_effect, "venue_effect", n_rows)
    balls = _ints(sched_balls, "sched_balls", n_rows)
    if not isinstance(m2_models, Mapping):
        raise TypeError(f"m2_models must map first-innings M2 fit ids to the fits; got {type(m2_models).__name__}")
    _require(par_means, ("fit_id", "bucket", *PAR_MEAN_FEATURES), "par_means")
    _require(par_bias, ("fit_id", "bucket", *PAR_BIAS_COLUMNS), "par_bias")
    means_fit = _text_values(par_means["fit_id"], "par_means column 'fit_id'")
    bias_fit = _text_values(par_bias["fit_id"], "par_bias column 'fit_id'")
    out = np.full(n_rows, np.nan)
    for fit_id in sorted(set(fits.tolist())):
        if fit_id not in m2_models:
            raise ValueError(f"m2_models lacks the first-innings M2 fit {fit_id!r} that some rows name")
        means = _fit_buckets(par_means, means_fit, fit_id, PAR_MEAN_FEATURES, "par_means")
        bias = _fit_buckets(par_bias, bias_fit, fit_id, PAR_BIAS_COLUMNS, "par_bias")
        mine = np.flatnonzero(fits == fit_id)
        contexts, context = par_contexts(env[mine], venue[mine], balls[mine])
        curves = corrected_par_curve(par_curve(m2_models[fit_id], contexts, means), contexts, bias)
        out[mine] = par_lookup(curves, contexts, context, balls[mine])
    return out


# ================================================== diagnostics: chase-start states and the innings-break gap

#: How a chase-start state was built (the ``kind`` column of the chase-start table).
CHASE_FROM_DELIVERY = "first_delivery"  # the chasing side's first innings-2 Delivery
CHASE_FROM_XI = "playing_xi"  # the chasing XI, for a match without innings-2 Deliveries

#: Columns of the chase-start table, in order (module docstring, "Diagnostics"): the
#: match, its fold and how the state was built; the innings-2 Scheduled_Balls and the RBQ,
#: Achievable_Rate and Team_Strength_Gap of the state; then the Delivery (CHASE_FROM_DELIVERY) or
#: the openers (both kinds), the M2 inputs behind Achievable_Rate and the fits that gave them.
CHASE_COLUMNS: tuple[str, ...] = (
    "match_id", "fold", "kind", "sched_balls", "rbq", ACH_RATE_COLUMN, "tsg", "row", "batting_team_id",
    "striker_id", "non_striker_id", "btc_count", "theta_striker", "theta_non_striker", "scoring_env",
    "venue_effect", "match_day", views.M1_FIT, views.M2_FIT,
)  # fmt: skip

_CHASE_BALL_COLUMNS: tuple[str, ...] = (
    "row", "match_id", "innings", "innings_ball_num", "batting_team_id", "bowling_team_id", "striker_id",
    "non_striker_id", "bowler_id", "cls", "runs", "sched_balls",
)  # fmt: skip
_CHASE_VALUES: tuple[str, ...] = (
    "rbq", ACH_RATE_COLUMN, "tsg", "theta_striker", "theta_non_striker", "scoring_env", "venue_effect", "match_day",
)  # fmt: skip
_CHASE_VIEW_COLUMNS: tuple[str, ...] = (*_CHASE_VALUES, "m1_pred_runs", views.M1_FIT, views.M2_FIT)
_CHASE_IDS: tuple[str, ...] = ("row", "batting_team_id", "striker_id", "non_striker_id", "btc_count")
_FIT_COLUMNS: tuple[str, ...] = (views.M1_FIT, views.M2_FIT)
_CHASE_FEATURE_COLUMNS: tuple[str, ...] = ("match_id", "sched_balls", "rbq", ACH_RATE_COLUMN, "tsg")
_OPENERS = 2  # batting orders 1 and 2 start the chase at the crease

#: The innings-break gap (innings_break_gap; INNINGS_BREAK_FILE under M3_FOLDER): the ``record``
#: of a row is GAP_MATCH (one family and match) or GAP_SUMMARY (one family: the means, the mean
#: absolute gap and ``n_matches``). ``last_row`` and ``last_lb_bowled`` are the last innings-1
#: Delivery and its pre-ball Legal_Balls bowled; ``sched_balls``, ``wickets``, ``total`` and
#: ``par_score_end`` describe the end-of-innings state (Legal_Balls bowled, wickets, team score and
#: Par_Score); ``runs_needed`` and ``sched_balls_chase`` the chase-start state.
GAP_MATCH = "match"
GAP_SUMMARY = "summary"
INNINGS_BREAK_COLUMNS: tuple[str, ...] = (
    "record", "family", "match_id", "fold", "last_row", "last_lb_bowled", "sched_balls", "wickets", "total",
    "par_score_end", "wp1_end", "runs_needed", "sched_balls_chase", "wp2_chase_start", "one_minus_wp2", "gap",
    "abs_gap", "n_matches",
)  # fmt: skip
_GAP_BALL_COLUMNS: tuple[str, ...] = ("row", "match_id", "innings", "innings_ball_num", "team_runs", "sched_balls")
_GAP_VIEW_COLUMNS: tuple[str, ...] = ("tsg", "scoring_env", "venue_effect", views.M2_FIT)
_GAP_INTS: tuple[str, ...] = (
    "match_id", "fold", "last_row", "last_lb_bowled", "sched_balls", "wickets", "sched_balls_chase", "n_matches",
)  # fmt: skip


def _aligned_balls(balls: pd.DataFrame, state: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """``columns`` of the ``balls`` row of each ``state`` row, in ``state`` order, on a RangeIndex."""
    _require(balls, columns, "balls")
    _require(state, ("row",), "state")
    index = pd.Index(_ints(balls["row"], "balls column 'row'"))
    if not index.is_unique:
        raise ValueError("balls must hold each row once")
    at = index.get_indexer(_ints(state["row"], "state column 'row'"))
    if (at < 0).any():
        raise ValueError(f"{_rows(int((at < 0).sum()))} of state have no Delivery in balls; pass the tables of one Run")
    return balls.iloc[at][list(columns)].reset_index(drop=True)


def _text_values(values: Any, what: str) -> np.ndarray:
    series = _vector(values, what)
    if series.isna().any():
        raise ValueError(f"{what} has {int(series.isna().sum()):,} missing value(s)")
    return series.astype(str).to_numpy(dtype=object)


def _openers(xi: pd.DataFrame, match_ids: np.ndarray, team: np.ndarray) -> list[np.ndarray]:
    """The first _OPENERS ``is_played`` players of each side in the Batters_To_Come order (float64; NaN where absent).

    Side i is (``match_ids[i]``, ``team[i]``); a NaN team has no player. Each player counts once,
    with his lowest ``batting_order`` (missing after every present one); ties keep the order of
    the ``xi`` rows, as in ``state.batters_to_come``.
    """
    _require(xi, XI_COLUMNS, "xi")
    if not pd.api.types.is_bool_dtype(xi["is_played"].dtype):
        raise TypeError(f"xi column 'is_played' must be boolean; got dtype {xi['is_played'].dtype}")
    played = xi.iloc[np.flatnonzero(xi["is_played"].to_numpy(dtype=bool, na_value=False))]
    frame = pd.DataFrame(
        {
            name: _floats(played[column], f"xi column {column!r}")
            for name, column in (("match_id", "match_id"), ("team_id", "team_id"), ("player_id", "player_id"))
        }
    )
    frame["order"] = np.nan_to_num(_floats(played["batting_order"], "xi column 'batting_order'"), nan=np.inf)
    frame["position"] = np.arange(len(frame), dtype=np.int64)
    frame = frame.dropna(subset=["match_id", "team_id", "player_id"])
    known = ~np.isnan(team)
    wanted = pd.MultiIndex.from_arrays([match_ids[known].astype(np.float64), team[known]])
    sides = pd.MultiIndex.from_arrays([frame["match_id"].to_numpy(), frame["team_id"].to_numpy()])
    frame = frame.iloc[np.flatnonzero(sides.isin(wanted))]
    players = frame.groupby(["match_id", "team_id", "player_id"], sort=False, as_index=False).agg(
        order=("order", "min"), position=("position", "min")
    )
    players = players.sort_values(["match_id", "team_id", "order", "position"], kind="stable")
    slot = players.groupby(["match_id", "team_id"], sort=False).cumcount().to_numpy()
    out = []
    for number in range(_OPENERS):
        values = np.full(len(match_ids), np.nan)
        part = players.iloc[np.flatnonzero(slot == number)]
        if len(part):
            keys = pd.MultiIndex.from_arrays([part["match_id"].to_numpy(), part["team_id"].to_numpy()])
            at = keys.get_indexer(wanted)
            ids = part["player_id"].to_numpy(dtype=np.float64)
            values[known] = np.where(at >= 0, ids[np.maximum(at, 0)], np.nan)
        out.append(values)
    return out


def _xi_chase_starts(
    rows: pd.DataFrame,
    view_by_k: Mapping[int, pd.DataFrame],
    match_ids: np.ndarray,
    match_fold: np.ndarray,
    first_innings_row: np.ndarray,
    *,
    matches: pd.DataFrame,
    xi: pd.DataFrame,
    theta: pd.DataFrame,
    m1_fits: pd.DataFrame,
    m2_models: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """The chase-start values of ``match_ids``, built from each chasing XI (module docstring, CHASE_FROM_XI).

    ``rows``: the _CHASE_BALL_COLUMNS of every view row, in view order; ``match_fold`` and
    ``first_innings_row``: each match's fold and the position of its first innings-1 Delivery.
    Only the innings-1 rows of the matches are read, so the construction also runs for a match
    that has innings-2 Deliveries (for checks). Returns the CHASE_COLUMNS values other than
    ``match_id``, ``fold``, ``kind`` and ``row`` (floats with NaN for a missing id, str fit ids).
    """
    n = len(match_ids)
    match = _ints(rows["match_id"], "balls column 'match_id'")
    innings = _ints(rows["innings"], "balls column 'innings'")
    out: dict[str, Any] = {name: np.full(n, np.nan) for name in (*_CHASE_VALUES, *_CHASE_IDS[1:])}
    out.update({name: np.full(n, None, dtype=object) for name in _FIT_COLUMNS})

    # Innings-2 Scheduled_Balls (Requirement 5, criteria 4-6) and the chasing side.
    sched = reconstruct.scheduled_balls(matches)
    second = sched.iloc[np.flatnonzero(sched["innings"].to_numpy() == 2)]
    at = pd.Index(second["match_id"].to_numpy(dtype=np.int64)).get_indexer(match_ids)
    if (at < 0).any():
        raise ValueError(f"{int((at < 0).sum()):,} match(es) with Deliveries are not in matches")
    out["sched_balls"] = second["sched_balls"].to_numpy(dtype=np.int64)[at]
    chasing = _floats(rows["bowling_team_id"], "balls column 'bowling_team_id'")[first_innings_row]
    out["batting_team_id"] = chasing

    # V_f values at the innings-1 rows, and Match_Day_Conditions at the end of innings 1.
    inn1 = np.flatnonzero(np.isin(match, match_ids) & (innings == 1))
    owner = np.searchsorted(match_ids, match[inn1])  # match_ids is sorted and holds every match of inn1
    legal = _ints(rows["cls"], "balls column 'cls'")[inn1] == int(DeliveryClass.LEGAL)
    residual = np.zeros(len(inn1))
    for k in np.unique(match_fold).tolist():
        view = view_by_k[k]
        mine = np.flatnonzero(match_fold == k)
        first = first_innings_row[mine]
        out["tsg"][mine] = -_floats(view["tsg"], f"view V_{k} column 'tsg'")[first]
        for name in ("scoring_env", "venue_effect"):
            out[name][mine] = _floats(view[name], f"view V_{k} column {name!r}")[first]
        for name in _FIT_COLUMNS:
            out[name][mine] = _text_values(view[name], f"view V_{k} column {name!r}")[first]
        part = np.flatnonzero(match_fold[owner] == k)
        predicted = _floats(view["m1_pred_runs"], f"view V_{k} column 'm1_pred_runs'")[inn1[part]]
        runs = _floats(rows["runs"], "balls column 'runs'")[inn1[part]]
        residual[part] = np.where(legal[part], runs - np.where(legal[part], predicted, 0.0), 0.0)
    count = np.bincount(owner, weights=legal.astype(np.float64), minlength=n)
    total = np.bincount(owner, weights=residual, minlength=n)
    out["match_day"] = np.where(count > 0, BALLS_PER_OVER * total / np.maximum(count, 1.0), 0.0)

    # The openers at the crease and the Batters_To_Come of a Delivery that starts innings 2 with them.
    striker, non_striker = _openers(xi, match_ids, chasing)
    out["striker_id"], out["non_striker_id"] = striker, non_striker
    lineup = pd.DataFrame(
        {"non_striker_id": pd.Series(non_striker).astype("Int64"), **{column: pd.NA for column in BTC_SLOT_COLUMNS}}
    ).astype({column: "Int64" for column in BTC_SLOT_COLUMNS})
    ready = np.flatnonzero(~np.isnan(striker))
    if len(ready):
        earlier = rows.iloc[inn1[np.isin(owner, ready)]][list(BTC_INPUT_COLUMNS)]
        starts = pd.DataFrame(
            {
                "row": -1 - np.arange(len(ready), dtype=np.int64),  # ids no Delivery has
                "match_id": match_ids[ready],
                "innings": np.full(len(ready), 2, dtype=np.int64),
                "innings_ball_num": np.ones(len(ready), dtype=np.int64),
                "batting_team_id": chasing[ready].astype(np.int64),
                "striker_id": striker[ready].astype(np.int64),
                "non_striker_id": pd.Series(non_striker[ready]).astype("Int64").array,
                # any player of the fielding side; Batters_To_Come reads the bowler only for appearances
                "bowler_id": _ints(rows["striker_id"], "balls column 'striker_id'")[first_innings_row[ready]],
            }
        )
        side_xi = xi.iloc[np.flatnonzero(np.isin(_floats(xi["match_id"], "xi column 'match_id'"), match_ids[ready]))]
        come = batters_to_come(pd.concat([earlier, starts], ignore_index=True), side_xi)
        come = come.iloc[np.flatnonzero(come["row"].to_numpy(dtype=np.int64) < 0)]
        place = ready[-1 - come["row"].to_numpy(dtype=np.int64)]
        for column in BTC_SLOT_COLUMNS:
            values = lineup[column].to_numpy(dtype=object)
            values[place] = come[column].astype("Int64").to_numpy(dtype=object)
            lineup[column] = pd.array(values, dtype="Int64")
        out["btc_count"][place] = _floats(come["btc_count"], "btc_count")

    # theta_bat and RBQ from the M1 fit of each match (with its Typical_Top7_Sum), then Achievable_Rate.
    _require(theta, ("fit_id", "player_id", "theta_bat", "theta_bowl"), "theta")
    _require(m1_fits, ("fit_id", "typical_top7_sum"), "m1_fits")
    theta_fit = _text_values(theta["fit_id"], "theta column 'fit_id'")
    top7 = pd.Series(
        _floats(m1_fits["typical_top7_sum"], "m1_fits column 'typical_top7_sum'"),
        index=_text_values(m1_fits["fit_id"], "m1_fits column 'fit_id'"),
    )
    for fit_id in sorted(set(out[views.M1_FIT].tolist())):
        if fit_id not in top7.index:
            raise ValueError(f"m1_fits has no Typical_Top7_Sum for the M1 fit {fit_id!r}")
        mine = np.flatnonzero(out[views.M1_FIT] == fit_id)
        table = theta.iloc[np.flatnonzero(theta_fit == fit_id)][["player_id", "theta_bat", "theta_bowl"]]
        table = table.reset_index(drop=True)
        players = pd.Index(_ints(table["player_id"], "theta column 'player_id'"))
        bat = _floats(table["theta_bat"], "theta column 'theta_bat'")
        for name, ids in (("theta_striker", striker), ("theta_non_striker", non_striker)):
            known = mine[~np.isnan(ids[mine])]
            where = players.get_indexer(ids[known].astype(np.int64))
            if (where < 0).any():
                raise ValueError(f"the theta table of {fit_id!r} lacks {int((where < 0).sum()):,} chasing opener(s)")
            out[name][known] = bat[where]
        values = skill_rbq(lineup.iloc[mine].reset_index(drop=True), table, float(top7[fit_id]))
        out["rbq"][mine] = values["rbq"].to_numpy(dtype=np.float64)
    features = pd.DataFrame(
        {
            "lb_left": out["sched_balls"].astype(np.float64),
            "wickets": np.zeros(n),
            "rbq": out["rbq"],
            "theta_striker": out["theta_striker"],
            "striker_lb_faced": np.zeros(n),
            "theta_non_striker": out["theta_non_striker"],
            "scoring_env": out["scoring_env"],
            "venue_effect": out["venue_effect"],
            "match_day": out["match_day"],
        },
        columns=list(M2_FEATURES),
    )
    for fit_id in sorted(set(out[views.M2_FIT].tolist())):
        if fit_id not in m2_models:
            raise ValueError(f"m2_models lacks the first-innings M2 fit {fit_id!r} that serves some chase starts")
        mine = np.flatnonzero(out[views.M2_FIT] == fit_id)
        out[ACH_RATE_COLUMN][mine] = achievable_rate(m2_models[fit_id], features.iloc[mine])
    return out


def chase_start_states(
    balls: pd.DataFrame,
    state: pd.DataFrame,
    view_by_k: Mapping[int, pd.DataFrame],
    fold_of_row: Any,
    *,
    matches: pd.DataFrame,
    xi: pd.DataFrame,
    theta: pd.DataFrame,
    m1_fits: pd.DataFrame,
    m2_models: Mapping[str, Any],
) -> pd.DataFrame:
    """The chase-start state of every match with a Delivery in innings 1 or 2 (a diagnostic; module docstring).

    ``balls``: the ``balls`` table of the ``reconstruct`` checkpoint; ``state``: the ``state``
    table (``row``, ``btc_count``); ``view_by_k``: the view V_k of each fold k from the
    ``remaining_runs`` checkpoint, rows in ``state`` order; ``fold_of_row``: the fold of each
    ``state`` row's match (``registry.fold_of``); ``matches`` and ``xi``: the ``matches`` and
    ``playing_eleven`` tables of the ``load`` checkpoint; ``theta`` and ``m1_fits``: the
    ``m1_theta`` and ``m1_fits`` tables of the ``skill`` checkpoint; ``m2_models``: fit id ->
    first-innings M2 model, for at least the fits that serve the fold-f rows of V_f (the outer
    fits ``m2_first_lgbm_outer_k<f>``, as the calibrated M2Models of
    ``remaining_runs.load_models``).

    Returns CHASE_COLUMNS on a RangeIndex, one row per match, sorted by ``match_id``: ``fold``
    (int8), ``kind`` (CHASE_FROM_DELIVERY or CHASE_FROM_XI), ``sched_balls`` (int64), ``rbq``,
    ``ach_rate``, ``tsg``, ``theta_striker``, ``theta_non_striker``, ``scoring_env``,
    ``venue_effect`` and ``match_day`` (float64, NaN where undefined), the ids ``row`` (the first
    innings-2 Delivery; <NA> for CHASE_FROM_XI), ``batting_team_id``, ``striker_id``,
    ``non_striker_id`` and ``btc_count`` (Int64), and ``m1_fit`` and ``m2_fit`` (str). Raises
    ValueError when the tables do not fit together (rows, folds, views, fits or players missing)
    and TypeError for a malformed column.
    """
    rows = _aligned_balls(balls, state, _CHASE_BALL_COLUMNS)
    _require(state, ("btc_count",), "state")
    n_rows = len(rows)
    fold = _ints(fold_of_row, "fold_of_row", n_rows)
    if not isinstance(view_by_k, Mapping) or not view_by_k:
        raise TypeError("view_by_k must map each fold k to view V_k")
    for k, view in view_by_k.items():
        _require(view, _CHASE_VIEW_COLUMNS, f"view V_{k}")
        _same_rows(view, state, f"view V_{k}")
    match = _ints(rows["match_id"], "balls column 'match_id'")
    innings = _ints(rows["innings"], "balls column 'innings'")
    ball = _ints(rows["innings_ball_num"], "balls column 'innings_ball_num'")
    if not np.isin(innings, M3_INNINGS).all():
        raise ValueError(f"{_rows(int((~np.isin(innings, M3_INNINGS)).sum()))} of state lie outside innings 1 and 2")
    if not n_rows:
        return pd.DataFrame({name: [] for name in CHASE_COLUMNS})
    per_match = pd.Series(fold).groupby(match, sort=True).agg(["min", "max"])
    if (per_match["min"] != per_match["max"]).any():
        raise ValueError("fold_of_row gives some match more than one fold; pass the fold of each row's match")
    match_ids = per_match.index.to_numpy(dtype=np.int64)
    match_fold = per_match["min"].to_numpy(dtype=np.int64)
    if not np.isin(match_fold, list(view_by_k)).all():
        raise ValueError(f"some matches lie in folds without a view; view_by_k holds folds {sorted(view_by_k)}")

    # The first Delivery of each innings of each match, in the Requirement 5 order.
    order = np.lexsort((ball, innings, match))
    starts = pd.DataFrame({"match_id": match[order], "innings": innings[order], "position": order})
    starts = starts.drop_duplicates(["match_id", "innings"])
    first = {
        int(number): pd.Series(
            part["position"].to_numpy(dtype=np.int64), index=part["match_id"].to_numpy(dtype=np.int64)
        )
        for number, part in starts.groupby("innings", sort=True)
    }
    empty = pd.Series(np.zeros(0, dtype=np.int64), index=pd.Index(np.zeros(0, dtype=np.int64)))
    second = first.get(2, empty)
    has_second = np.isin(match_ids, second.index.to_numpy())
    n = len(match_ids)
    values: dict[str, Any] = {name: np.full(n, np.nan) for name in (*_CHASE_VALUES, *_CHASE_IDS)}
    values.update({name: np.full(n, None, dtype=object) for name in _FIT_COLUMNS})
    values["sched_balls"] = np.zeros(n, dtype=np.int64)

    # CHASE_FROM_DELIVERY: V_f at the chasing side's first innings-2 Delivery.
    chosen = np.flatnonzero(has_second)
    position = second.reindex(match_ids[chosen]).to_numpy(dtype=np.int64)
    for k in np.unique(match_fold[chosen]).tolist():
        view = view_by_k[k]
        mine = match_fold[chosen] == k
        where, at = chosen[mine], position[mine]
        for name in _CHASE_VALUES:
            values[name][where] = _floats(view[name], f"view V_{k} column {name!r}")[at]
        for name in _FIT_COLUMNS:
            values[name][where] = _text_values(view[name], f"view V_{k} column {name!r}")[at]
    values["sched_balls"][chosen] = _ints(rows["sched_balls"], "balls column 'sched_balls'")[position]
    for name in _CHASE_IDS:
        source = state["btc_count"] if name == "btc_count" else rows[name]
        values[name][chosen] = _floats(source, f"column {name!r}")[position]

    # CHASE_FROM_XI: the chasing XI, for the matches without innings-2 Deliveries.
    built = np.flatnonzero(~has_second)
    if len(built):
        built_values = _xi_chase_starts(
            rows,
            view_by_k,
            match_ids[built],
            match_fold[built],
            first[1].reindex(match_ids[built]).to_numpy(dtype=np.int64),
            matches=matches,
            xi=xi,
            theta=theta,
            m1_fits=m1_fits,
            m2_models=m2_models,
        )
        for name, array in built_values.items():
            values[name][built] = array

    data: dict[str, Any] = {
        "match_id": match_ids,
        "fold": match_fold.astype(np.int8),
        "kind": pd.array(np.where(has_second, CHASE_FROM_DELIVERY, CHASE_FROM_XI), dtype="str"),
        "sched_balls": values["sched_balls"],
    }
    for name in CHASE_COLUMNS[4:]:
        if name in _CHASE_IDS:
            data[name] = pd.Series(values[name]).astype("Int64").array
        elif name in _FIT_COLUMNS:
            data[name] = pd.array(values[name].astype(str), dtype="str")
        else:
            data[name] = values[name]
    return pd.DataFrame(data, columns=list(CHASE_COLUMNS))


def chase_start_features(chase: pd.DataFrame, match_id: Any, first_innings_total: Any) -> pd.DataFrame:
    """The innings-2 M3_FEATURES of the chase-start state of each row (a diagnostic; module docstring).

    ``chase``: the chase-start table (``match_id``, ``sched_balls``, ``rbq``, ``ach_rate``,
    ``tsg``); ``match_id``: the match of each row; ``first_innings_total``: the first-innings team
    score of each row's state (fractional allowed, 0 or more). Returns M3_FEATURES[2] as float64
    columns on a RangeIndex: runs needed = total + 1, Legal_Balls left = the innings-2
    Scheduled_Balls, 0 wickets, RBQ, Required_Rate minus Achievable_Rate (6 x runs needed /
    Legal_Balls left - ``ach_rate``) and TSG. Raises ValueError for a match that ``chase`` lacks or
    repeats, or a total that is missing, infinite or below 0.
    """
    _require(chase, _CHASE_FEATURE_COLUMNS, "chase")
    index = pd.Index(_ints(chase["match_id"], "chase column 'match_id'"))
    if not index.is_unique:
        raise ValueError("chase must hold each match once")
    ids = _ints(match_id, "match_id")
    totals = _floats(first_innings_total, "first_innings_total", len(ids))
    bad = np.isnan(totals) | (totals < 0.0)
    if bad.any():
        raise ValueError(f"first_innings_total is missing or below 0 on {_rows(int(bad.sum()))}")
    at = index.get_indexer(ids)
    if (at < 0).any():
        shown = ", ".join(map(str, np.unique(ids[at < 0])[:_SHOWN].tolist()))
        raise ValueError(f"chase has no chase-start state for {_rows(int((at < 0).sum()))} (for example match {shown})")
    balls_left = _floats(chase["sched_balls"], "chase column 'sched_balls'")[at]
    if not (balls_left > 0).all():
        raise ValueError("every chase-start state needs a Scheduled_Balls above 0")
    needed = totals + 1.0
    rate = BALLS_PER_OVER * needed / balls_left
    return pd.DataFrame(
        {
            "runs_needed": needed,
            "lb_left": balls_left,
            "wickets": np.zeros(len(ids)),
            "rbq": _floats(chase["rbq"], "chase column 'rbq'")[at],
            "rr_minus_ar": rate - _floats(chase[ACH_RATE_COLUMN], f"chase column {ACH_RATE_COLUMN!r}")[at],
            "tsg": _floats(chase["tsg"], "chase column 'tsg'")[at],
        },
        columns=list(M3_FEATURES[2]),
    )


# ======================================================================== the final WP of any states

def _wp_model(models: Any, innings: int) -> Any:
    """``models[innings]``, checked to predict calibrated WP by fold for that innings."""
    if not isinstance(models, Mapping):
        raise TypeError(f"models must map each innings to its calibrated fold models; got {type(models).__name__}")
    if innings not in models:
        raise ValueError(f"models holds no calibrated fold models of innings {innings}")
    model = models[innings]
    if not callable(getattr(model, "predict", None)):
        raise TypeError(f"models[{innings}] must be a CalibratedFolds (or FoldModels); got {type(model).__name__}")
    if getattr(model, "innings", innings) != innings:
        raise ValueError(f"models[{innings}] holds the models of innings {model.innings}")
    return model


def win_probability(innings: int, states: pd.DataFrame, *, fold: Any, models: Mapping[int, Any]) -> TerminalWP:
    """The final WP of any states of ``innings`` (criteria 13-17; module docstring, "The final WP of any states").

    ``states``: the innings' M3_FEATURES, one row per state (pre-ball or Post_Outcome_States, the
    end-of-innings states included); ``fold``: the fold of each row's match; ``models``: the
    calibrated fold models of one Candidate_Model, at least ``{innings: CalibratedFolds}``
    (load_wp_models gives both innings). The classifier scores every row that no rule of
    DECIDING_RULES decides, each with the model of its fold; in innings 1 that is every row.
    Returns a TerminalWP. Raises ValueError for a fold without a model, and the errors of the
    rules and the models.
    """
    innings = _check_innings(innings)
    _require(states, M3_FEATURES[innings], "states")
    rule = terminal_rules(innings, states)
    folds_of_rows = _ints(fold, "fold", len(rule))
    model = _wp_model(models, innings)
    wp = np.full(len(rule), np.nan)
    kept = np.flatnonzero(~np.isin(rule, DECIDING_RULES))
    if len(kept):
        wp[kept] = model.predict(states.iloc[kept], folds_of_rows[kept])
    return apply_terminal_rules(innings, states, wp)


def _family_order(names: Any) -> list[str]:
    """Candidate family names in FAMILIES order, any other name after them in name order."""
    return sorted(names, key=lambda name: (FAMILIES.index(name) if name in FAMILIES else len(FAMILIES), name))


def innings_break_gap(
    balls: pd.DataFrame,
    state: pd.DataFrame,
    view_by_k: Mapping[int, pd.DataFrame],
    fold_of_row: Any,
    chase: pd.DataFrame,
    models: Mapping[str, Mapping[int, Any]],
    *,
    m2_models: Mapping[str, Any],
    par_means: pd.DataFrame,
    par_bias: pd.DataFrame,
) -> pd.DataFrame:
    """WP1 at the actual end of innings 1 against 1 - WP2 at the chase start, per match and family (a diagnostic).

    ``balls``: the ``balls`` table of the ``reconstruct`` checkpoint (``row``, ``match_id``,
    ``innings``, ``innings_ball_num``, ``team_runs``, ``sched_balls``); ``state``: the ``state``
    table (``row``, ``lb_bowled``, ``score``, ``wickets``); ``view_by_k``: view V_k of each fold k
    of the ``remaining_runs`` checkpoint, rows in ``state`` order; ``fold_of_row``: the fold of
    each ``state`` row's match; ``chase``: the chase-start table (chase_start_states); ``models``:
    family -> the family's calibrated fold models of both innings ``{1: CalibratedFolds, 2:
    CalibratedFolds}``; ``m2_models``, ``par_means`` and ``par_bias``: as for innings_end_par.

    For every match with Deliveries in both innings, the last innings-1 Delivery (highest
    ``innings_ball_num``) gives the end-of-innings state of its actual outcome (module docstring,
    "Diagnostics") and its total, and the chase-start state takes runs needed = total + 1. Both
    WPs come from the family's models of the match's fold. Returns INNINGS_BREAK_COLUMNS: the
    GAP_MATCH rows of each family in ``match_id`` order, then its GAP_SUMMARY row. Raises
    ValueError for tables that do not fit together (rows, folds, views, chase-start states or
    fits missing) and TypeError for a malformed argument.
    """
    rows = _aligned_balls(balls, state, _GAP_BALL_COLUMNS)
    _require(state, ("lb_bowled", "score", "wickets"), "state")
    n_rows = len(rows)
    fold = _ints(fold_of_row, "fold_of_row", n_rows)
    if not isinstance(view_by_k, Mapping):
        raise TypeError(f"view_by_k must map each fold k to view V_k; got {type(view_by_k).__name__}")
    if not isinstance(models, Mapping) or not models:
        raise TypeError("models must map each candidate family to its calibrated fold models of both innings")
    match = _ints(rows["match_id"], "balls column 'match_id'")
    innings = _ints(rows["innings"], "balls column 'innings'")
    ball = _ints(rows["innings_ball_num"], "balls column 'innings_ball_num'")

    # The last innings-1 Delivery of every match with Deliveries in both innings, by match.
    first = np.flatnonzero((innings == 1) & np.isin(match, match[innings == 2]))
    ordered = first[np.lexsort((ball[first], match[first]))]
    last = ordered[np.r_[match[ordered][1:] != match[ordered][:-1], True]] if len(ordered) else ordered
    match_ids, match_fold = match[last], fold[last]
    view_values = {name: np.full(len(last), np.nan) for name in _GAP_VIEW_COLUMNS[:-1]}
    fit_ids = np.full(len(last), None, dtype=object)
    for k in np.unique(match_fold).tolist():
        if k not in view_by_k:
            raise ValueError(f"view_by_k has no view for fold {k}, which matches belong to")
        view = view_by_k[k]
        _require(view, _GAP_VIEW_COLUMNS, f"view V_{k}")
        _same_rows(view, state, f"view V_{k}")
        mine = np.flatnonzero(match_fold == k)
        for name in view_values:
            view_values[name][mine] = _floats(view[name], f"view V_{k} column {name!r}")[last[mine]]
        fit_ids[mine] = _text_values(view[views.M2_FIT], f"view V_{k} column {views.M2_FIT!r}")[last[mine]]

    # The end-of-innings state of the last Delivery's actual outcome (criterion 17).
    sched = _ints(rows["sched_balls"], "balls column 'sched_balls'")[last]
    wickets = _ints(state["wickets"], "state column 'wickets'")[last]
    team_runs = _floats(rows["team_runs"], "balls column 'team_runs'")[last]
    total = _floats(state["score"], "state column 'score'")[last] + team_runs
    if not (np.isfinite(total) & (total >= 0.0)).all():
        raise ValueError(
            "the first-innings total (pre-ball score + team runs of the last innings-1 Delivery) is missing or "
            f"negative for {int((~(np.isfinite(total) & (total >= 0.0))).sum()):,} match(es)"
        )
    par_end = innings_end_par(
        fit_ids, view_values["scoring_env"], view_values["venue_effect"], sched,
        m2_models=m2_models, par_means=par_means, par_bias=par_bias,
    )  # fmt: skip
    end_states = pd.DataFrame(
        {
            "lb_bowled": sched.astype(np.float64),
            "score": total,
            "wickets": wickets.astype(np.float64),
            "par_score": par_end,
            "tsg": view_values["tsg"],
            "scoring_env": view_values["scoring_env"],
            "venue_effect": view_values["venue_effect"],
            "sched_balls": sched.astype(np.float64),
        },
        columns=list(M3_FEATURES[1]),
    )

    # The chase-start state for that total, from the match's own fold.
    _require(chase, ("match_id", "fold"), "chase")
    index = pd.Index(_ints(chase["match_id"], "chase column 'match_id'"))
    at = index.get_indexer(match_ids)
    if (at < 0).any():
        shown = ", ".join(map(str, np.unique(match_ids[at < 0])[:_SHOWN].tolist()))
        raise ValueError(f"chase has no chase-start state for {int((at < 0).sum()):,} match(es) (for example {shown})")
    if (_ints(chase["fold"], "chase column 'fold'")[at] != match_fold).any():
        raise ValueError("chase records another fold than fold_of_row for some matches; pass the tables of one Run")
    starts = chase_start_features(chase, match_ids, total)

    parts = []
    for family in _family_order(models):
        wp1 = np.array(win_probability(1, end_states, fold=match_fold, models=models[family]).wp)
        wp2 = np.array(win_probability(2, starts, fold=match_fold, models=models[family]).wp)
        gap = wp1 - (1.0 - wp2)
        matches = {
            "record": GAP_MATCH, "family": family, "match_id": match_ids, "fold": match_fold,
            "last_row": _ints(rows["row"], "balls column 'row'")[last],
            "last_lb_bowled": _ints(state["lb_bowled"], "state column 'lb_bowled'")[last], "sched_balls": sched,
            "wickets": wickets, "total": total, "par_score_end": par_end, "wp1_end": wp1,
            "runs_needed": starts["runs_needed"].to_numpy(), "sched_balls_chase": starts["lb_left"].to_numpy(),
            "wp2_chase_start": wp2, "one_minus_wp2": 1.0 - wp2, "gap": gap, "abs_gap": np.abs(gap),
            "n_matches": pd.NA,
        }  # fmt: skip
        parts.append(pd.DataFrame(matches, columns=list(INNINGS_BREAK_COLUMNS)))
        summary: dict[str, Any] = {name: pd.NA if name in _GAP_INTS else np.nan for name in INNINGS_BREAK_COLUMNS}
        summary.update({"record": GAP_SUMMARY, "family": family, "n_matches": len(last)})
        if len(last):
            summary.update(
                {
                    "wp1_end": float(wp1.mean()), "wp2_chase_start": float(wp2.mean()),
                    "one_minus_wp2": float((1.0 - wp2).mean()), "gap": float(gap.mean()),
                    "abs_gap": float(np.abs(gap).mean()),
                }
            )  # fmt: skip
        parts.append(pd.DataFrame([summary], columns=list(INNINGS_BREAK_COLUMNS)))
    out = pd.concat(parts, ignore_index=True)
    dtypes: dict[str, Any] = {name: "Int64" for name in _GAP_INTS}
    dtypes.update({"record": "str", "family": "str"})
    dtypes.update({name: np.float64 for name in INNINGS_BREAK_COLUMNS if name not in dtypes})
    return out.astype(dtypes)


def load_wp_models(
    store: CheckpointStore, family: str | CandidateFamily, n_folds: int, *, stage: str | None = None
) -> dict[int, CalibratedFolds]:
    """The calibrated fold models of one family for both innings, as ``win_probability`` takes them.

    Reads ``load_fold_models(store, stage, innings, family, n_folds)`` for innings 1 and 2 from
    ``stage`` (STAGE_NAME by default). Raises the errors of load_fold_models.
    """
    source = STAGE_NAME if stage is None else stage
    return {innings: load_fold_models(store, source, innings, family, n_folds) for innings in M3_INNINGS}


# ============================================================================ metrics (10.8, 10.12, 10.20)

#: The Calibration_Curve bins (glossary): 10 equal-width WP bins, [0, 0.1), ..., [0.9, 1.0].
CALIBRATION_BINS = 10
CALIBRATION_EDGES: np.ndarray = _frozen(np.arange(CALIBRATION_BINS + 1, dtype=np.float64) / CALIBRATION_BINS)

#: Columns of a calibration_curve result, in order.
CALIBRATION_COLUMNS: tuple[str, ...] = ("bin", "lower", "upper", "n_deliveries", "mean_wp", "observed_win_share")


def _probability_values(p: Any, what: str) -> np.ndarray:
    values = _floats(p, what)
    bad = np.isnan(values) | (values < 0.0) | (values > 1.0)
    if bad.any():
        raise ValueError(f"{what} holds {int(bad.sum()):,} value(s) that are missing or outside [0, 1]")
    return values


def _evaluated(y: Any, p: Any) -> tuple[np.ndarray, np.ndarray]:
    values = _probability_values(p, "p")
    labels = _binary(y, "y", len(values))
    if not len(values):
        raise ValueError("a metric needs at least one row")
    return labels, values


def calibration_curve(p: Any, y: Any) -> pd.DataFrame:
    """The Calibration_Curve of predictions ``p`` (WP in [0, 1]) against labels ``y`` (0 or 1; glossary).

    Returns CALIBRATION_COLUMNS, one row per bin of CALIBRATION_EDGES: ``bin`` (0-9), ``lower``,
    ``upper``, ``n_deliveries``, ``mean_wp`` and ``observed_win_share`` (NaN in an empty bin). A
    value equal to an inner edge k/10 falls in bin k, and 1.0 in the last bin. Raises ValueError
    for a value that is missing or outside [0, 1], a label other than 0 or 1, or lengths that
    differ.
    """
    values = _probability_values(p, "p")
    labels = _binary(y, "y", len(values)).astype(np.float64)
    bins = np.minimum(np.searchsorted(CALIBRATION_EDGES, values, side="right") - 1, CALIBRATION_BINS - 1)
    count = np.bincount(bins, minlength=CALIBRATION_BINS)
    filled = count > 0
    mean_wp, observed = np.full(CALIBRATION_BINS, np.nan), np.full(CALIBRATION_BINS, np.nan)
    mean_wp[filled] = np.bincount(bins, weights=values, minlength=CALIBRATION_BINS)[filled] / count[filled]
    observed[filled] = np.bincount(bins, weights=labels, minlength=CALIBRATION_BINS)[filled] / count[filled]
    return pd.DataFrame(
        {
            "bin": np.arange(CALIBRATION_BINS, dtype=np.int64),
            "lower": CALIBRATION_EDGES[:-1].copy(),
            "upper": CALIBRATION_EDGES[1:].copy(),
            "n_deliveries": count.astype(np.int64),
            "mean_wp": mean_wp,
            "observed_win_share": observed,
        },
        columns=list(CALIBRATION_COLUMNS),
    )


def calibration_error(curve: pd.DataFrame) -> float:
    """The Calibration_Error of a calibration_curve result (glossary; criterion 20).

    The Delivery-count-weighted mean of |``mean_wp`` - ``observed_win_share``| over the bins with
    at least one Delivery. Raises ValueError when no bin has a Delivery or a non-empty bin lacks
    a value.
    """
    _require(curve, ("n_deliveries", "mean_wp", "observed_win_share"), "curve")
    count = _floats(curve["n_deliveries"], "curve column 'n_deliveries'")
    filled = count > 0
    if not filled.any():
        raise ValueError("the Calibration_Curve holds no Delivery, so its Calibration_Error is undefined")
    mean = _floats(curve["mean_wp"], "curve column 'mean_wp'")[filled]
    observed = _floats(curve["observed_win_share"], "curve column 'observed_win_share'")[filled]
    if np.isnan(mean).any() or np.isnan(observed).any():
        raise ValueError("a non-empty bin of the Calibration_Curve lacks its mean WP or observed win share")
    weights = count[filled]
    return float(np.sum(weights * np.abs(mean - observed)) / np.sum(weights))


def brier_score(y: Any, p: Any) -> float:
    """The Brier score of predictions ``p`` against labels ``y``: the mean of (p - y) squared over every row."""
    labels, values = _evaluated(y, p)
    return float(np.mean((values - labels) ** 2))


def roc_auc(y: Any, p: Any) -> float:
    """The AUC of predictions ``p`` against labels ``y`` (scikit-learn ``roc_auc_score``); NaN when one label is absent."""
    from sklearn.metrics import roc_auc_score

    labels, values = _evaluated(y, p)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, values))


def wp_metrics(y: Any, p: Any) -> dict[str, float]:
    """The criterion-12 metrics of predictions ``p`` against labels ``y``, pooled over every row given.

    Returns ``n_rows``, ``n_wins`` (rows with label 1), ``log_loss`` (Pooled_Held_Out_Log_Loss,
    ``common.pooled_log_loss``), ``brier`` (brier_score), ``auc`` (roc_auc) and
    ``calibration_error`` (of calibration_curve). Raises ValueError for no row, and the errors of
    those functions.
    """
    labels, values = _evaluated(y, p)
    return {
        "n_rows": len(labels),
        "n_wins": int(labels.sum()),
        "log_loss": common.pooled_log_loss(labels, values),
        "brier": brier_score(labels, values),
        "auc": roc_auc(labels, values),
        "calibration_error": calibration_error(calibration_curve(values, labels)),
    }


# =========================================================================== the win_prob Pipeline_Stage

#: Name of the Pipeline_Stage that fits M3, and of its checkpoint.
STAGE_NAME = "win_prob"

#: Checkpoint tables (module docstring, "Reading the stage's outputs"): the m3_labels table, the
#: chase-start table, FIT_TABLE_COLUMNS of every fit and FOLD_METRICS_COLUMNS of every outer fit.
LABELS_TABLE = "m3_labels"
CHASE_TABLE = "chase_start"
FIT_TABLE = "m3_fits"
FOLD_METRICS_TABLE = "m3_fold_metrics"

#: Columns of each Candidate_Model's WP table (wp_table), in order.
WP_TABLE_COLUMNS: tuple[str, ...] = ("row", "fold", "wp_pre", "wp_raw", "terminal_rule", "fit_id", "calibrator_fit_id")

#: Folder under the outputs folder for the stage's files (criterion 18), their names and columns;
#: INNINGS_BREAK_FILE holds the innings-break gap (INNINGS_BREAK_COLUMNS), a diagnostic.
M3_FOLDER = Path("models") / "m3"
METRICS_FILE = "metrics.csv"
FOLD_METRICS_FILE = "fold_metrics.csv"
INNINGS_BREAK_FILE = "innings_break_gap.csv"
METRICS_COLUMNS: tuple[str, ...] = ("metric", "innings", "family", "phase", "fold", "rule", "value", "n_rows", "description")
CALIBRATION_TABLE_COLUMNS: tuple[str, ...] = ("innings", "family", "phase", *CALIBRATION_COLUMNS)

#: Resolution of the Calibration_Curve charts in dots per inch, fixed so that the same data give the same image.
CHART_DPI = 150

#: The name of each Phase in the metrics (``phase`` column), and of the whole innings.
PHASE_NAMES: Mapping[int, str] = MappingProxyType({int(member): member.name.lower() for member in Phase})
_ALL_PHASES = "all"

#: Data-derived Method_Constant: fit id -> the boosting rounds that early stopping kept, for each
#: outer LightGBM fit (the fits behind the LightGBM Out_Of_Fold_Predictions).
BEST_ITERATION_CONSTANT = "win_prob.outer_lgbm_best_iteration_by_fit"

_COMPONENT = "Win_Probability_Model"  # the glossary component that produces the constant
_LOAD_STAGE = "load"
_MATCHES_TABLE = "matches"
_XI_TABLE = "playing_eleven"
_CALIBRATION_ERROR_MAX = "tests.calibration_error_max"  # the Test_Suite threshold of criterion 20, for the Run_Log
_FLOAT_FORMAT = "%.6f"
_DISAGREE = 0.5  # a terminal row's label disagrees with the rule's WP when they differ by this much or more
_FAMILY_LABELS: Mapping[str, str] = MappingProxyType({LOGISTIC: "Logistic baseline", LGBM: "LightGBM"})
_FAMILY_COLOURS: Mapping[str, str] = MappingProxyType({LOGISTIC: "#d95f02", LGBM: "#1f5fa8"})
_CHART_SIZE = (15.0, 4.6)  # inches
_SUBSET_TITLES: Mapping[str, str] = MappingProxyType(
    {_ALL_PHASES: "All Phases", **{name: name.capitalize() for name in PHASE_NAMES.values()}}
)


def wp_table(innings: int, family: str) -> str:
    """The checkpoint table name of the WP table of one Candidate_Model: ``wp_inn<innings>_<family>``."""
    innings = _check_innings(innings)
    if not isinstance(family, str) or not _FAMILY_NAME.fullmatch(family):
        raise ValueError(f"family must be a Candidate_Model family name; got {family!r}")
    return f"wp_inn{innings}_{family}"


def calibration_file(innings: int) -> str:
    """The file name of the Calibration_Curve table of ``innings`` under M3_FOLDER."""
    return f"calibration_curve_inn{_check_innings(innings)}.csv"


def calibration_chart_file(innings: int) -> str:
    """The file name of the Calibration_Curve chart of ``innings`` under M3_FOLDER."""
    return f"calibration_curve_inn{_check_innings(innings)}.png"


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


def _terminal_states(innings: int, state: pd.DataFrame, labels: pd.DataFrame, mine: np.ndarray) -> pd.DataFrame:
    """The TERMINAL_COLUMNS of the pre-ball states of the rows ``mine`` (one innings), in order."""
    names = TERMINAL_COLUMNS[innings]
    data = {}
    for name in names:
        source = labels if name in LABEL_FEATURES else state
        data[name] = _floats(source[name], f"column {name!r}")[mine]
    return pd.DataFrame(data, columns=list(names))


def _fit_ids(lookup: Mapping[int, str], fold: np.ndarray) -> Any:
    """The fit id of ``lookup`` (fold -> fit id) for each row's fold, as a ``str`` array."""
    keys = np.array(sorted(lookup), dtype=np.int64)
    names = np.array([lookup[int(k)] for k in keys], dtype=object)
    at = np.searchsorted(keys, fold)
    if len(fold) and ((at >= len(keys)) | (keys[np.minimum(at, len(keys) - 1)] != fold)).any():
        raise ValueError("some rows lie in folds without a fit id")
    return pd.array(names[at].astype(str), dtype="str")


def _disagrees(label: np.ndarray, wp: np.ndarray) -> np.ndarray:
    """True where the label differs from the WP by _DISAGREE (0.5) or more.

    A tie WP of 0.5 always disagrees: an eligible match has a winner, since ties and super-over
    wins are dropped (criterion 2).
    """
    return np.abs(np.asarray(label, dtype=np.float64) - wp) >= _DISAGREE


def _wp_frame(rows: np.ndarray, fold_models: FoldModels, result: TerminalWP) -> pd.DataFrame:
    """The WP_TABLE_COLUMNS table of one Candidate_Model (module docstring, step 4)."""
    return pd.DataFrame(
        {
            "row": rows,
            "fold": fold_models.fold.astype(np.int8),
            "wp_pre": np.array(result.wp),
            "wp_raw": np.array(fold_models.wp),
            "terminal_rule": pd.array(result.rule_names.astype(str), dtype="str"),
            "fit_id": _fit_ids(fold_models.outer_fit_ids, fold_models.fold),
            "calibrator_fit_id": _fit_ids(fold_models.calibrator_fit_ids, fold_models.fold),
        },
        columns=list(WP_TABLE_COLUMNS),
    )


def pre_ball_table(innings: int, fold_models: FoldModels, state: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """The WP table (WP_TABLE_COLUMNS) of one Candidate_Model and innings, as the ``win_prob`` stage saves it.

    ``fold_models``: the fit_innings result of ``innings``; ``state`` and ``labels``: the
    ``state`` table and the m3_labels table that fit_innings read (the TERMINAL_COLUMNS of the
    pre-ball states). One row per Delivery of the innings in ``state`` order: ``row``, ``fold``,
    ``wp_pre`` (the calibrated Out_Of_Fold WP with apply_terminal_rules on top; criteria 13-17
    and 19), ``wp_raw`` (before them), ``terminal_rule``, ``fit_id`` and ``calibrator_fit_id`` (the
    outer fit and the calibrator of the row's fold). The ``model_choice`` stage builds the tables
    of the Alternative_Models with it (task 26.1). Raises TypeError for an argument of the wrong
    kind and ValueError when the tables do not hold the rows of ``fold_models`` in its order.
    """
    innings = _check_innings(innings)
    if not isinstance(fold_models, FoldModels):
        raise TypeError(f"fold_models must be the FoldModels of fit_innings; got {type(fold_models).__name__}")
    if fold_models.innings != innings:
        raise ValueError(f"fold_models holds the fits of innings {fold_models.innings}, not of innings {innings}")
    names = TERMINAL_COLUMNS[innings]
    _require(labels, ("row", "innings", *(name for name in names if name in LABEL_FEATURES)), "labels")
    _require(state, ("row", *(name for name in names if name not in LABEL_FEATURES)), "state")
    _same_rows(labels, state, "labels")
    mine = np.flatnonzero(_ints(labels["innings"], "labels column 'innings'") == innings)
    if not np.array_equal(_ints(state["row"], "state column 'row'")[mine], fold_models.row):
        raise ValueError(
            f"fold_models does not hold the innings-{innings} Deliveries of the state table in its order; pass the "
            "tables that fit_innings read"
        )
    result = apply_terminal_rules(innings, _terminal_states(innings, state, labels, mine), fold_models.wp)
    return _wp_frame(state["row"].to_numpy()[mine], fold_models, result)


_METRIC_TEXT: Mapping[str, str] = MappingProxyType(
    {
        "log_loss": (
            "Pooled_Held_Out_Log_Loss of the calibrated Out_Of_Fold WP before the terminal rules, over the eligible "
            "Deliveries (kept result types, labelled, Training_Rows) of all folds, each probability clipped to "
            "[1e-15, 1 - 1e-15] (Requirement 10, criterion 12)"
        ),
        "brier": "held-out Brier score: mean squared difference between the same WP and the label (criterion 12)",
        "auc": "held-out AUC of the same WP against the label (criterion 12)",
        "calibration_error": (
            "Calibration_Error of the same rows' Calibration_Curve (10 equal-width WP bins): the Delivery-weighted "
            "mean |mean WP - observed win share| over the non-empty bins (criteria 12 and 20)"
        ),
        "n_wins": "eligible Deliveries whose batting side won the match (label 1)",
        "in_fold_log_loss": (
            "log loss of the outer fit's raw output on the Deliveries it was fitted on: its Training_Rows other than "
            "the early-stopping matches (criterion 8)"
        ),
        "held_out_log_loss": "log loss of the outer fit's raw output on the fold's eligible Deliveries (criterion 8)",
        "held_out_minus_in_fold": "held-out minus in-fold log loss of the outer fit (criterion 8)",
        "held_out_log_loss_calibrated": "log loss of the calibrated WP on the fold's eligible Deliveries",
        "terminal_rows": (
            "pre-ball Deliveries of the innings that the terminal rule flags (criteria 14-17): a chase rule sets the "
            "WP, while an innings_end state keeps the innings-1 classifier's WP (criterion 17)"
        ),
        "terminal_rows_eligible": "eligible pre-ball Deliveries that the terminal rule flags",
        "terminal_label_disagrees": (
            "eligible pre-ball Deliveries flagged by the rule whose label differs from the final WP by 0.5 or more; "
            "the metrics above use the classifier WP on these rows"
        ),
    }
)


def _value_text(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, Integral):
        return str(int(value))
    number = float(value)
    return "" if np.isnan(number) else f"{number:.6f}"


class _Metrics:
    """The rows of METRICS_FILE, in the order they are added."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def add(
        self, metric: str, innings: int, family: str, value: Any, n_rows: int, *, phase: str = "",
        fold: int | None = None, rule: str = "",
    ) -> None:  # fmt: skip
        self.records.append(
            {
                "metric": metric,
                "innings": innings,
                "family": family,
                "phase": phase,
                "fold": fold,
                "rule": rule,
                "value": _value_text(value),
                "n_rows": int(n_rows),
                "description": _METRIC_TEXT[metric],
            }
        )

    def table(self) -> pd.DataFrame:
        frame = pd.DataFrame.from_records(self.records, columns=list(METRICS_COLUMNS))
        return frame.astype(
            {
                "metric": "str",
                "innings": np.int64,
                "family": "str",
                "phase": "str",
                "fold": "Int64",
                "rule": "str",
                "value": "str",
                "n_rows": np.int64,
                "description": "str",
            }
        )


def _subsets(eligible: np.ndarray, phase: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """(name, positions) of the eligible rows of the whole innings and of each Phase."""
    parts = [(_ALL_PHASES, np.flatnonzero(eligible))]
    parts += [(PHASE_NAMES[code], np.flatnonzero(eligible & (phase == code))) for code in sorted(PHASE_NAMES)]
    return parts


def _write_calibration_chart(path: Path, innings: int, curves: pd.DataFrame, errors: Mapping[tuple[str, str], float]) -> None:
    """The Calibration_Curve chart of one innings as a PNG: one panel per subset, both candidates in each (criterion 18)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    subsets = list(_SUBSET_TITLES)
    figure = Figure(figsize=_CHART_SIZE, dpi=CHART_DPI, layout="constrained")
    FigureCanvasAgg(figure)
    axes = list(np.atleast_1d(figure.subplots(1, len(subsets), sharex=True, sharey=True)))
    phase = curves["phase"].to_numpy(dtype=object)
    family = curves["family"].to_numpy(dtype=object)
    ticks = np.linspace(0.0, 1.0, 6)
    for ax, subset in zip(axes, subsets):
        ax.plot([0.0, 1.0], [0.0, 1.0], color="#9a9a9a", linestyle="--", linewidth=1.0, label="Perfect calibration")
        n_rows = 0
        for name in FAMILIES:
            part = curves.iloc[np.flatnonzero((phase == subset) & (family == name))]
            if part.empty:
                continue
            filled = part["n_deliveries"].to_numpy() > 0
            n_rows = int(part["n_deliveries"].sum())
            ax.plot(
                part["mean_wp"].to_numpy()[filled],
                part["observed_win_share"].to_numpy()[filled],
                marker="o",
                markersize=4,
                linewidth=1.4,
                color=_FAMILY_COLOURS.get(name, "#333333"),
                label=f"{_FAMILY_LABELS.get(name, name)} (Calibration_Error {errors[(name, subset)]:.3f})",
            )
        ax.set_title(f"{_SUBSET_TITLES[subset]} ({n_rows:,} Deliveries)", fontsize=10)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_aspect("equal")
        ax.grid(True, color="#e6e6e6", linewidth=0.6)
        ax.set_xlabel("Mean calibrated WP in the bin")
        ax.legend(loc="upper left", fontsize=7, frameon=False)
    axes[0].set_ylabel("Observed share won by the batting side")
    figure.suptitle(
        f"Innings {innings}: Calibration_Curve of the calibrated Out_Of_Fold WP (10 equal-width bins; eligible "
        "Deliveries, before the terminal rules)",
        fontsize=11,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        figure.savefig(temporary, format="png", dpi=CHART_DPI, metadata={"Software": None})
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_stage(ctx: StageContext) -> None:
    """The ``win_prob`` Pipeline_Stage (module docstring, "The win_prob Pipeline_Stage").

    Raises FitLeakageError or ViewLeakageError (exit code 4) when a fit or a view breaks the
    leakage guard (Requirement 7, criterion 12), DataError (exit code 3) for a match that is not in
    the Fold_Table, and ValueError or TypeError for input tables that do not fit together or a
    malformed Config value.
    """
    started = time.perf_counter()
    settings, store, registry, log = ctx.settings, ctx.store, ctx.registry, ctx.log
    n_folds = settings.get(_N_FOLDS)
    plan = folds.sandbox_plan(n_folds)  # checks D15 before any table is read
    calibration_bounds(settings)  # checks the bounds before any table is read; fit_innings reads them again
    families = {name: candidate_family(name, settings) for name in FAMILIES}
    error_max = settings.get(_CALIBRATION_ERROR_MAX)
    outer = [spec for spec in plan if spec.kind == folds.OUTER]
    all_folds = tuple(sorted(spec.sandbox_k for spec in outer))

    balls = store.load_table(reconstruct.STAGE_NAME, reconstruct.BALLS_TABLE)
    state = store.load_table(STATE_STAGE, STATE_TABLE)
    matches = store.load_table(_LOAD_STAGE, _MATCHES_TABLE)
    xi = store.load_table(_LOAD_STAGE, _XI_TABLE)
    theta = store.load_table(SKILL_STAGE, M1_THETA_TABLE)
    m1_fits = store.load_table(SKILL_STAGE, M1_FITS_TABLE)
    view_by_k = {k: store.load_table(M2_STAGE, views.view_table(k)) for k in all_folds}
    first_ids = [leakage.fit_id_for(M2_FIRST_INNINGS_MODEL, spec, family=M2_FAMILY) for spec in outer]
    m2_models = load_m2_models(store, first_ids)  # calibrated M2Models (Requirement 9, criterion 15)
    par_means = store.load_table(M2_STAGE, PAR_MEANS_TABLE)
    par_bias = store.load_table(M2_STAGE, PAR_BIAS_TABLE)

    # 1. Labels and Training_Row flags, the fold of every row, and the views checked before use.
    labels = m3_labels(balls, state, matches)
    row_fold = registry.fold_of(_ints(labels["match_id"], "labels column 'match_id'")).astype(np.int64)
    for k, view in view_by_k.items():
        views.check_view(k, view, row_fold, registry)

    # 2. The 2 x 2 fold sandboxes: every fit registered, every model and calibrator saved.
    fitted: dict[tuple[int, str], FoldModels] = {}
    n_models = 0
    for innings in M3_INNINGS:
        for name in FAMILIES:
            fold_models = fit_innings(innings, families[name], view_by_k, labels, settings, registry, state=state)
            n_models += len(save_fold_models(store, STAGE_NAME, fold_models, inner=True))
            fitted[(innings, name)] = fold_models
            _log_fits(ctx, fold_models)

    # 3. Diagnostics: the chase-start state of every match and the innings-break gap of both families.
    chase = chase_start_states(
        balls, state, view_by_k, row_fold, matches=matches, xi=xi, theta=theta, m1_fits=m1_fits, m2_models=m2_models
    )
    store.save_table(STAGE_NAME, CHASE_TABLE, chase)
    gap = innings_break_gap(
        balls, state, view_by_k, row_fold, chase,
        {name: {number: fitted[(number, name)].calibrated for number in M3_INNINGS} for name in FAMILIES},
        m2_models=m2_models, par_means=par_means, par_bias=par_bias,
    )  # fmt: skip
    del view_by_k, balls

    # 4. The final pre-ball WP of every innings 1-2 Delivery of each candidate (criteria 13-17 and 19).
    innings_of_row = _ints(labels["innings"], "labels column 'innings'")
    phase_of_row = _ints(state["phase"], "state column 'phase'")
    state_rows = state["row"].to_numpy()
    final: dict[tuple[int, str], TerminalWP] = {}
    for (innings, name), fold_models in fitted.items():
        mine = np.flatnonzero(innings_of_row == innings)
        result = apply_terminal_rules(innings, _terminal_states(innings, state, labels, mine), fold_models.wp)
        final[(innings, name)] = result
        store.save_table(STAGE_NAME, wp_table(innings, name), _wp_frame(state_rows[mine], fold_models, result))

    # 5. Metrics on the eligible rows, before the terminal rules (criteria 8 and 12), and the charts.
    metrics = _Metrics()
    curves: dict[int, list[pd.DataFrame]] = {innings: [] for innings in M3_INNINGS}
    errors: dict[int, dict[tuple[str, str], float]] = {innings: {} for innings in M3_INNINGS}
    summary: dict[tuple[int, str], dict[str, dict[str, float]]] = {}
    for (innings, name), fold_models in fitted.items():
        phase = phase_of_row[np.flatnonzero(innings_of_row == innings)]
        y, p = np.asarray(fold_models.label), np.asarray(fold_models.wp)
        summary[(innings, name)] = {}
        for subset, rows in _subsets(np.asarray(fold_models.eligible), phase):
            if not len(rows):
                continue
            values = wp_metrics(y[rows], p[rows])
            summary[(innings, name)][subset] = values
            for metric in ("log_loss", "brier", "auc", "calibration_error", "n_wins"):
                metrics.add(metric, innings, name, values[metric], len(rows), phase=subset)
            curve = calibration_curve(p[rows], y[rows])
            curve.insert(0, "phase", subset)
            curve.insert(0, "family", name)
            curve.insert(0, "innings", innings)
            curves[innings].append(curve)
            errors[innings][(name, subset)] = values["calibration_error"]
        for record in fold_models.fold_metrics.to_dict("records"):
            k = int(record["fold"])
            metrics.add("in_fold_log_loss", innings, name, record["in_fold_log_loss"], record["n_fit_rows"], fold=k)
            for metric in ("held_out_log_loss", "held_out_minus_in_fold", "held_out_log_loss_calibrated"):
                metrics.add(metric, innings, name, record[metric], record["n_held_out_rows"], fold=k)
        result, eligible = final[(innings, name)], np.asarray(fold_models.eligible)
        disagrees = _disagrees(y, result.wp)
        for code in _INNINGS_RULES[innings]:
            decided = result.rule == code
            rule_name = TERMINAL_RULES[code]
            metrics.add("terminal_rows", innings, name, int(decided.sum()), len(decided), rule=rule_name)
            metrics.add(
                "terminal_rows_eligible", innings, name, int((decided & eligible).sum()), int(eligible.sum()),
                rule=rule_name,
            )  # fmt: skip
            metrics.add(
                "terminal_label_disagrees", innings, name, int((decided & eligible & disagrees).sum()),
                int((decided & eligible).sum()), rule=rule_name,
            )  # fmt: skip
    outdir = settings.path("outputs") / M3_FOLDER
    metrics_table = metrics.table()
    fold_table = pd.concat([fm.fold_metrics for fm in fitted.values()], ignore_index=True)
    _write_csv(outdir / METRICS_FILE, metrics_table)
    _write_csv(outdir / FOLD_METRICS_FILE, fold_table)
    _write_csv(outdir / INNINGS_BREAK_FILE, gap)
    for innings in M3_INNINGS:
        table = pd.concat(curves[innings], ignore_index=True)[list(CALIBRATION_TABLE_COLUMNS)]
        table = table.astype({"family": "str", "phase": "str"})
        _write_csv(outdir / calibration_file(innings), table)
        _write_calibration_chart(outdir / calibration_chart_file(innings), innings, table, errors[innings])

    # 6. Checkpoint tables, the fit registry and the data-derived Method_Constant.
    fit_table = pd.concat([fm.fit_table for fm in fitted.values()], ignore_index=True)
    store.save_table(STAGE_NAME, LABELS_TABLE, labels)
    store.save_table(STAGE_NAME, FIT_TABLE, fit_table)
    store.save_table(STAGE_NAME, FOLD_METRICS_TABLE, fold_table)
    n_saved = registry.save(store, STAGE_NAME)
    boosted = fold_table.iloc[np.flatnonzero(fold_table["family"].to_numpy(dtype=object) == LGBM)]
    ctx.constants.record(
        BEST_ITERATION_CONSTANT,
        dict(zip(boosted["fit_id"].astype(str).tolist(), boosted["iterations"].astype(int).tolist())),
        _COMPONENT,
    )
    _log_stage(
        ctx,
        fitted=fitted,
        final=final,
        summary=summary,
        chase=chase,
        gap=gap,
        error_max=float(error_max),
        outdir=outdir,
        n_models=n_models,
        n_saved=n_saved,
        seconds=time.perf_counter() - started,
    )


def _span(values: Sequence[Any]) -> str:
    numbers = [int(value) for value in values if value is not None and not pd.isna(value)]
    if not numbers:
        return "n/a"
    low, high = min(numbers), max(numbers)
    return f"{low:,}" if low == high else f"{low:,}-{high:,}"


def _log_fits(ctx: StageContext, fm: FoldModels) -> None:
    table = fm.fit_table
    kind = table["kind"].to_numpy(dtype=object)
    metrics = fm.fold_metrics
    gap = metrics["held_out_minus_in_fold"].to_numpy(dtype=np.float64)
    knots = [len(model.calibrator.x_knots) for model in fm.calibrated.models.values()]
    ctx.log.info(
        f"M3 innings {fm.innings} {fm.family}: {int((kind == folds.OUTER).sum())} outer and "
        f"{int((kind == folds.INNER).sum())} inner fits and {int((kind == CALIBRATOR).sum())} centred isotonic "
        f"calibrators ({_span(knots)} knots) in {fm.seconds:.1f} s; iterations kept "
        f"{_span(metrics['iterations'].tolist())} (outer), "
        f"{_span(table['iterations'][kind == folds.INNER].tolist())} (inner); held-out minus in-fold raw log loss "
        f"per fold {' / '.join(f'{value:+.4f}' for value in gap)} (criterion 8); calibrated held-out log loss per "
        f"fold {' / '.join(f'{value:.4f}' for value in metrics['held_out_log_loss_calibrated'])}",
        stage=STAGE_NAME,
    )


def _log_stage(
    ctx: StageContext,
    *,
    fitted: Mapping[tuple[int, str], FoldModels],
    final: Mapping[tuple[int, str], TerminalWP],
    summary: Mapping[tuple[int, str], Mapping[str, Mapping[str, float]]],
    chase: pd.DataFrame,
    gap: pd.DataFrame,
    error_max: float,
    outdir: Path,
    n_models: int,
    n_saved: int,
    seconds: float,
) -> None:
    log = ctx.log
    kinds = chase["kind"].to_numpy(dtype=object)
    built = chase.iloc[np.flatnonzero(kinds == CHASE_FROM_XI)]
    log.info(
        f"chase-start states of {len(chase):,} matches (a diagnostic; no WP reads them): "
        f"{int((kinds == CHASE_FROM_DELIVERY).sum()):,} from the chasing side's first innings-2 Delivery and "
        f"{len(built):,} built from the chasing XI (no innings-2 Delivery: matches "
        f"{', '.join(map(str, built['match_id'].tolist())) or 'none'}); RBQ missing in "
        f"{int(chase['rbq'].isna().sum()):,}, Achievable_Rate missing in {int(chase[ACH_RATE_COLUMN].isna().sum()):,}, "
        f"TSG missing in {int(chase['tsg'].isna().sum()):,}",
        stage=STAGE_NAME,
    )
    rows = gap.iloc[np.flatnonzero(gap["record"].astype(str).to_numpy(dtype=object) == GAP_SUMMARY)]
    log.info(
        "innings-break gap (a diagnostic; the jump falls between two Deliveries, so no Leverage_Index includes it): "
        "WP1 of the end-of-innings state of the last innings-1 Delivery's actual outcome (Requirement 10, criterion "
        "17) minus 1 - WP2 of the chase-start state for that total: "
        + "; ".join(
            f"{record['family']}: mean {record['gap']:+.4f}, mean absolute {record['abs_gap']:.4f} (mean WP1 "
            f"{record['wp1_end']:.4f} against mean 1 - WP2 {record['one_minus_wp2']:.4f}) over "
            f"{int(record['n_matches']):,} matches with both innings"
            for record in rows.to_dict("records")
        ),
        stage=STAGE_NAME,
    )
    lines = []
    above = []
    for (innings, name), subsets in summary.items():
        whole = subsets[_ALL_PHASES]
        phases = "/".join(
            f"{subsets[PHASE_NAMES[code]]['calibration_error']:.4f}" if PHASE_NAMES[code] in subsets else "n/a"
            for code in sorted(PHASE_NAMES)
        )
        lines.append(
            f"innings {innings} {name}: log loss {whole['log_loss']:.4f}, Brier {whole['brier']:.4f}, AUC "
            f"{whole['auc']:.4f}, Calibration_Error {whole['calibration_error']:.4f} (powerplay/middle/death {phases}) "
            f"on {whole['n_rows']:,} eligible Deliveries"
        )
        if not whole["calibration_error"] <= error_max:
            above.append(f"innings {innings} {name} ({whole['calibration_error']:.4f})")
    log.info(
        "M3 Candidate_Models on their pooled Out_Of_Fold WP before the terminal rules (criterion 12; "
        f"tests.calibration_error_max {error_max:g} applies to the Primary_Model, criterion 20): " + "; ".join(lines),
        stage=STAGE_NAME,
    )
    if above:
        log.warning(
            f"Calibration_Error above tests.calibration_error_max ({error_max:g}) for {', '.join(above)}; criterion 20 "
            "tests the Primary_Model of each innings",
            stage=STAGE_NAME,
        )
    counts = []
    for (innings, name), result in final.items():
        fold_models = fitted[(innings, name)]
        eligible = np.asarray(fold_models.eligible)
        disagrees = _disagrees(fold_models.label, result.wp)
        parts = []
        for code in _INNINGS_RULES[innings]:
            decided = result.rule == code
            parts.append(
                f"{TERMINAL_RULES[code]} {int(decided.sum()):,} ({int((decided & eligible).sum()):,} eligible, "
                f"{int((decided & eligible & disagrees).sum()):,} with a disagreeing label)"
            )
        counts.append(f"innings {innings} {name}: " + ", ".join(parts))
    log.info(
        "terminal pre-ball states (criteria 14-17; an innings_end state keeps the innings-1 classifier's WP): "
        + "; ".join(counts),
        stage=STAGE_NAME,
    )
    shown = _shown_path(outdir, ctx.root)
    files = [METRICS_FILE, FOLD_METRICS_FILE]
    for innings in M3_INNINGS:
        files += [calibration_file(innings), calibration_chart_file(innings)]
    files.append(INNINGS_BREAK_FILE)
    log.info(
        f"wrote {', '.join(f'{shown}/{name}' for name in files)}; saved {', '.join(wp_table(i, f) for i, f in fitted)}, "
        f"{CHASE_TABLE}, {LABELS_TABLE}, {FIT_TABLE}, {FOLD_METRICS_TABLE}, {n_models} models and calibrators under "
        f"their fit ids and the fit registry ({n_saved} fits) in {seconds:.1f} s",
        stage=STAGE_NAME,
    )
