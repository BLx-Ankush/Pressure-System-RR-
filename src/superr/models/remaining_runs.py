"""Remaining_Runs_Model M2: the training targets, the capped and first-innings LightGBM fits and
their Legal_Balls-left calibration, the Post_Wicket_State features, the lookup-table baseline,
Par_Score, Par_Rate and Achievable_Rate (Requirement 9, criteria 1-7, 14 and 15; Requirement 7,
criteria 5-8; design DD-7, DD-15 and sections
"Remaining_Runs_Model M2 (``models/remaining_runs.py``)" and "Fold sandboxes and leakage
control"), and the ``remaining_runs`` Pipeline_Stage (Requirement 9, criteria 5, 8-10 and 12;
Requirement 7, criterion 8).

The functions up to ``achievable_rate`` are the M2 library (task 10.2). The stage
(``run_stage``, section "The remaining_runs Pipeline_Stage" below) runs the 25 capped and 25
first-innings fits of ``folds.sandbox_plan(D15)`` through ``models.common.fit_registered``, adds
the M2-derived quantities to the sandbox views and computes Wicket_Cost with
``superr.value.wicket_cost``. Every function receives the Config values it needs as arguments:
the ``models.m2`` section, ``runtime`` and ``seeds.m2`` through the BoosterSetup that
``m2_setup`` builds, and the recent window D6 through the ``win_lb`` values given to
``par_rate``; the stage reads D15 (``design_choices.D15_n_folds``) as well. No function reads
``config.py`` or has a default value for a Config setting.

Rows
    M2 works on the rows of the ``state`` table: every Delivery of innings 1 and 2, in the
    Requirement 5 order (``match_id``, ``innings``, ``innings_ball_num``). Each sandbox view V_k
    holds the same rows in the same order, and ``m2_targets``, ``m2_features`` and
    ``post_wicket_features`` return one row per such Delivery in that order. The other functions
    take equal-length 1-D arrays or Series, one value per row, in any common order.

Targets (``m2_targets``; criteria 1 and 2)
    The remaining runs of a Delivery are the team runs under the selected Extras_Encoding (the
    ``team_runs`` column of ``balls``) from that Delivery, inclusive, to the end of its innings:
    the innings total minus the team score before the Delivery. The ``innings_info`` rows of the
    selected encoding give each innings' total, which the summed ``team_runs`` must equal (so a
    ``balls`` table and an encoding that do not belong together raise ValueError), and the Target
    behind the runs needed: Target minus the score before the Delivery, as in the ``state`` table.

    * ``capped=False`` (the first-innings variant): every target is the remaining runs.
    * ``capped=True`` (the capped variant): an innings-1 target is the remaining runs; an
      innings-2 target is min(remaining runs, max(runs needed, 0)). The floor at 0 keeps every
      target at 0 or more. An innings-2 Delivery whose Target is missing has no capped target
      (NaN), so it cannot be a Training_Row.

    ``needed_nonpositive`` flags the innings-2 Deliveries whose runs needed before the ball is 0
    or less, so the stage can report them; their capped target is 0. On the checkpoints of 14
    November 2026 there are 249 such Deliveries in 96 matches (211 of them in training-eligible
    matches), 165 of them, in 72 matches, with runs needed below 0. They stay in the data as
    they are.

Features (``m2_features``, ``post_wicket_features``; criterion 3, glossary Post_Wicket_State)
    ``m2_features(view, state)`` gives the M2_FEATURES as float64 columns: ``lb_left``,
    ``wickets`` and ``striker_lb_faced`` from the ``state`` table, and ``rbq``,
    ``theta_striker``, ``theta_non_striker``, ``scoring_env``, ``venue_effect`` and
    ``match_day`` from a sandbox view, so every fit of sandbox k reads its M1 values from V_k
    (Requirement 7, criterion 7). A missing value stays NaN, which LightGBM treats as missing.

    ``post_wicket_features(view, state)`` gives the same features after one hypothetical
    dismissal of the striker: ``wickets`` + 1; ``theta_striker`` = the view's
    ``theta_next_batter``, the theta_bat of the Batters_To_Come player with the lowest
    ``batting_order``; ``striker_lb_faced`` = 0; ``rbq`` = the view's ``rbq_post_wicket``, RBQ
    with that player removed from Batters_To_Come; ``lb_left``, ``theta_non_striker``,
    ``scoring_env``, ``venue_effect`` and ``match_day`` unchanged. ``theta_next_batter`` is NaN
    when Batters_To_Come is empty (2,608 Deliveries as delivered), and LightGBM then reads it as
    0, so the post-wicket prediction of such a Delivery means nothing. Requirement 9, criterion
    13 takes the post-wicket runs of those Deliveries, and of Deliveries with 9 wickets fallen,
    as 0 instead.

Fits (``m2_setup``, ``m2_fitter``, ``fit_m2``, ``predict``; criterion 4)
    Both variants use the LightGBM objective ``models.m2.objective``, one of M2_OBJECTIVES:
    ``poisson`` as delivered, or ``regression`` (L2). Both losses are minimised by the
    conditional mean, so a prediction is the expected remaining runs either way. A Poisson fit
    predicts exp(sum of the trees), an expected count of runs that is always above 0, and its
    trees work on the log scale, where the remaining runs of the last ball (about 2) and of a
    whole innings (about 170) differ by a factor rather than by 170 runs. The Poisson deviance
    is minimised by the conditional mean whatever the distribution of runs, so the
    over-dispersion that boundaries cause does not bias it.

    Poisson rather than L2: with the Config's learning rate and early stopping, the L2 fits stop
    after 58-82 rounds, before their predictions reach the ends of the innings. On their training
    rows, the first-innings L2 fits of folds 1-3 predicted 5.6-9.4 runs too many with 1
    Legal_Ball left (the mean there is about 2), which drove Achievable_Rate up to 64-91 runs per
    over on the last ball of a chase. The Poisson fits ran 113-186 rounds, cut that error to
    1.8-6.8 runs and the largest Achievable_Rate to 24-53, and their held-out RMSE was lower or
    at most 0.09 runs higher, per innings, for both variants (checks of 14 November 2026). Both
    objectives still predict about 5-8 runs too few for the first ball of an innings, which the
    Par_Score bias correction offsets (below, "Par_Score"), and the Poisson fits still predict
    too many runs on the last balls, which the Legal_Balls-left calibration (below) removes.

    ``m2_setup`` builds the parameters with ``common.lgbm_setup(settings, "m2", "regression",
    monotone_constraints=M2_MONOTONE)`` and then sets the objective to ``models.m2.objective``:
    the ``models.m2`` learning rate, leaves, minimum rows per leaf, L2 penalty and round cap; the
    section's ``monotone_constraints_method`` (``intermediate`` as delivered);
    ``runtime.n_threads``; and ``seeds.m2``, with ``deterministic`` and ``force_row_wise`` in
    deterministic mode (Requirement 28, criterion 3). The objective's other settings keep
    LightGBM's defaults.

    M2_MONOTONE makes every fit non-decreasing in Legal_Balls left and RBQ and non-increasing in
    wickets fallen. LightGBM enforces the constraints on the sum of the trees while it grows each
    tree, and exp keeps their direction, so they hold for every prediction, whatever the other
    features. The constraints are positional, so the feature frame is always rebuilt in
    M2_FEATURES order before LightGBM sees it.

    Early stopping holds out whole matches (Requirement 10, criterion 7, applied to every
    boosted fit): ``es_share`` of the fit's training matches, drawn by
    ``leakage.early_stopping_split(training matches, es_share, seeds.m2, fit index)``, where the
    fit index is the fit's position in ``folds.sandbox_plan(D15)``. The other matches are fitted
    on. Training stops once the L2 loss (mean squared error of the predicted runs, the loss that
    the RMSE metrics of criterion 5 report) on the held-out matches has not improved for
    ``es_patience`` rounds, or at ``max_rounds``, and the model of the best round is kept. The
    held-out matches are Training_Rows too (glossary).

    ``m2_fitter(setup, X, y)`` is the fit function for ``common.fit_registered``, which draws
    that split, registers the fit with the leakage guard and only then fits: it fits the
    Booster, calibrates it on the fit's Training_Rows (below) and returns the M2Model.
    ``fit_m2(X, y, groups, setup, fit_index=...)`` fits without a registry on rows that are all
    Training_Rows, with the same split and the same fitter, for tests and checks.
    ``predict(model, X)`` returns the expected remaining runs of each row: an M2Model's
    calibrated prediction, whose Booster predicts with its best round.

Legal_Balls-left calibration (``lb_calibration``, ``M2Model``, ``load_models``; criterion 15, DD-15)
    The trees of a fit lump the last few Legal_Balls-left values together, so the Booster's
    prediction near the end of an innings is too high: on the checkpoints of 14 November 2026
    the outer first-innings fits predicted a median of 4.9 runs (at most 8.9) with 1 Legal_Ball
    left, where an innings scores about 2, and Achievable_Rate reached 53 runs per over on the
    last ball of a chase. Every M2 fit f (capped and first-innings, outer and inner) is therefore
    an M2Model: its Booster and a factor c_f estimated from f's own Training_Rows only, the rows
    it was fitted on and its early-stopping rows, never a row of a fold that f predicts:

    1. the raw prediction R_raw(x) is the Booster's prediction, taken as 0 where it is below 0
       (only an L2 fit, ``models.m2.objective`` ``regression``, can predict below 0);
    2. for each Legal_Balls-left value l of the Training_Rows, ratio(l) is the sum of their
       targets divided by the sum of their R_raw over the Training_Rows with l;
    3. c_f is the isotonic (non-decreasing) regression of ratio(l) on l, weighted by the sum of
       R_raw at l. With these weights a block of pooled values takes the ratio of its summed
       targets to its summed raw predictions, so the calibrated predictions of the Training_Rows
       add up to their targets in every block. A value l whose R_raw sum to 0 has no ratio, stays
       out of the regression and takes the factor of the nearest lower l that has one (of the
       lowest such l when none is lower). A factor that would be 0 (every Training_Row up to some
       l without a run) is raised to _MIN_LB_FACTOR (1e-6), a numerical safeguard that keeps every
       calibrated prediction above 0 and that no delivered fit reaches;
    4. the calibrated prediction is R(x) = c_f(lb_left(x)) x R_raw(x). Off the values l of the
       table, c_f is a step function: the factor of the largest l at or below lb_left, or of the
       smallest l when lb_left lies below them all, so c_f is flat beyond the range of the
       Training_Rows. No Training_Row has 0 Legal_Balls left, so c_f(0) = c_f(1) as delivered.

    The constraints of criterion 4 still hold for R. c_f depends on Legal_Balls left alone and
    is positive, so at a fixed Legal_Balls left R is R_raw times one positive number and keeps
    R_raw's directions in wickets fallen (non-increasing) and RBQ (non-decreasing). Along
    Legal_Balls left, for l1 < l2 and every other feature held, R_raw(l2) >= R_raw(l1) >= 0 (the
    Booster's constraint, and the floor at 0 keeps the order) and c_f(l2) >= c_f(l1) > 0, so
    R(l2) = c_f(l2) R_raw(l2) >= c_f(l1) R_raw(l2) >= c_f(l1) R_raw(l1) = R(l1). Rounding a
    product of non-negative floats keeps this order, so the Property 24 test needs no tolerance.

    The calibrated prediction is M2's prediction everywhere: the Out_Of_Fold_Predictions and
    their RMSE, Omega (r_pre and r_post), the par curve R(t) and so Par_Score, whose bias
    correction is estimated on top of the calibrated raw par, Achievable_Rate, the Achievable_Rate
    of the Post_Outcome_States (``pressure.post_outcome_states``), the Par_Score at Scheduled_Balls
    of the end-of-innings states (``win_prob.innings_end_par``) and the chase-start states.
    ``predict`` refuses a bare LightGBM Booster, so that no caller uses raw predictions by
    mistake; ``M2Model.predict_raw`` gives them, for the before-and-after metrics.

    On the checkpoints of 15 November 2026 the factors of the outer fits are 0.25-0.44 (capped)
    and 0.23-0.56 (first-innings) at 1 Legal_Ball left, 0.69-0.93 at 6, 0.89-0.98 at 18,
    0.99-1.00 at 60 and 1.03-1.05 at 120, and no factor of the 50 fits is at the floor. Over the
    Training_Rows of all folds, the held-out mean of the first-innings fits with 1 Legal_Ball
    left went from 5.33 runs (raw) to 2.15 (calibrated), against 2.15 scored, and with 2-6 left
    from 10.13 to 7.98 against 8.02; the pooled RMSE fell from 24.007 to 23.922 (first-innings),
    25.166 to 25.083 (capped, innings 1) and 25.004 to 24.941 (capped, innings 2). Achievable_Rate
    with 1 Legal_Ball left now lies between 11.9 and 13.4 runs per over (median 12.8; it reached
    53.2 before), and the chase that needed 7 off the last ball (match 14109) has a WP of 0.001
    instead of 0.997.

Lookup-table baseline (``lookup_baseline``, ``rmse_by_innings``, ``lookup_baseline_rmse``;
criterion 5)
    For each held-out fold f, a table of the mean training target by Legal_Balls left x wickets
    fallen is built from the training rows of the other folds (4 with D15 = 5) and predicts the
    rows of fold f. One table serves both innings, as the capped M2 does, which has no innings
    feature; the RMSE is reported for each innings. Each prediction is, in order of preference:

    1. LOOKUP_CELL: the mean target of the training rows in its Legal_Balls left x wickets
       fallen cell;
    2. LOOKUP_SAME_LB_LEFT, for a cell without training rows: the mean target of the training
       rows with the same Legal_Balls left, whatever the wickets (criterion 5);
    3. LOOKUP_NEAREST_LB_LEFT, when that Legal_Balls left has no training row either: the mean
       target of the training rows of the nearest Legal_Balls left value that has rows, pooling
       the rows of both neighbours when two are equally near. Remaining runs depend most on the
       balls left, so a neighbouring value predicts far better than an overall mean would.

    ``rmse_by_innings(y, pred, innings, rows)`` gives the RMSE of any predictions per innings,
    pooled over every given row of all folds (not a mean of per-fold values): the metric of
    M2's Out_Of_Fold_Predictions against the training target and of the baseline.
    ``lookup_baseline_rmse`` evaluates the baseline on the training rows and also counts the
    rows that used fill rules 2 and 3.

Par_Score (``par_bucket``, ``par_means``, ``par_contexts``, ``par_curve``, ``par_bias``,
``corrected_par_curve``, ``par_lookup``, ``first_innings_par``; criteria 6 and 14, DD-7)
    Par_Score comes from a first-innings M2 fit, because the capped M2 mixes capped chase
    targets into its predictions (Assumption 8). For fit f and a first-innings context c =
    (Scoring_Environment, Venue_Effect, Scheduled_Balls SB):

    * The bucket of a Legal_Ball count t is round(120 x t / SB), rounded half up and capped at
      120 (``par_bucket``). PAR_BUCKETS (120) is the Scheduled_Balls of a full T20 innings, fixed
      by the format rather than a Method_Constant, so a 120-ball innings has one bucket per count
      and a shorter one is mapped onto the same 0-120 scale of Ball_Position_Share.
    * ``par_means`` averages wickets fallen, RBQ, striker theta_bat, striker Legal_Balls faced
      and non-striker theta_bat (PAR_MEAN_FEATURES) over the fit's first-innings Training_Rows in
      each bucket, each Training_Row in the bucket of its own Legal_Balls bowled and
      Scheduled_Balls. Missing values are left out of a mean. A bucket without a value of a
      feature takes the mean over the rows of the nearest bucket that has one, pooling both
      neighbours on a tie; as delivered this happens only for bucket 120, because no Delivery
      has 0 Legal_Balls left.
    * The par state at count t, for t = 0 ... SB, has Legal_Balls left SB - t, the five bucket
      means of bucket(t), Match_Day_Conditions 0, and the context's Scoring_Environment and
      Venue_Effect. R(t) is f's calibrated prediction for it (``M2Model.predict``), at t = SB
      too, as DD-7 specifies. No Training_Row has 0 Legal_Balls left, so f's Booster predicts as
      much there as with 1 left and its calibration factor is that of 1 left, and as
      delivered bucket 120 takes the means and the bias of bucket 119: with SB = 120 the last
      Legal_Ball adds nothing to par, Par_Score(120) = Par_Score(119). Only the
      ``par_score_next`` of the innings' last Legal_Ball reads Par_Score(SB).
    * The raw Par_Score is P_raw(t) = max over u <= t of (R(0) - R(u)) = R(0) - min over u <= t
      of R(u) (``par_curve``).
    * The raw Par_Score runs low: f is evaluated at the mean state of each count rather than at
      the states the innings reach, and every objective's Booster predicts the first ball of an
      innings 5-8 runs too low, which the calibration factor at 120 Legal_Balls left (1.03-1.05
      for the outer first-innings fits) only partly lifts. So each fit gets a bias per bucket,
      estimated on top of its calibrated raw par, from its own first-innings Training_Rows
      (``par_bias``): the residual of Training_Row i is its pre-ball team score minus P_raw at
      its own context and Legal_Balls bowled L_i, and beta(b) is the mean residual over the
      Training_Rows in bucket b, each in the bucket of its own L_i and SB_i. A bucket without a
      Training_Row takes the mean of the nearest bucket that has one, pooling both neighbours on
      a tie, as for the par means.
    * Par_Score is the corrected curve (``corrected_par_curve``): P(0) = 0 and P(t) = max(P(t -
      1), P_raw(t) + beta(bucket(t))) for t = 1 ... SB. So Par_Score(0) = 0, and Par_Score is
      never below 0 and never decreases as t rises (criterion 6), whatever the model predicts
      and whatever the bias. Wherever the running maximum does not bind, the mean of Par_Score
      minus score over the Training_Rows of a bucket is 0.
    * For the outer first-innings fit of fold 1, beta rises from +0.4 runs at bucket 1 to +8.0
      at bucket 119 (+1.1 and +23.1 before the Legal_Balls-left calibration). On the held-out
      fold-1 rows, the median of Par_Score minus score goes from -0.4 runs with fewer than 36
      Legal_Balls bowled, -3.2 with 36-89 and -7.8 with 90 or more (raw; -5.3, -11.7 and -20.4
      before the calibration) to +0.0, +0.8 and -1.8 (corrected), and the mean from -1.7, -5.2
      and -9.6 (raw) to -0.2, -1.1 and -3.6 (checks of 15 November 2026).

    ``par_contexts`` lists the distinct contexts of a set of first-innings rows and gives each row
    its context; ``par_curve(model, ctx, par_means)`` evaluates the SB + 1 par states of every
    context in one prediction and returns one raw curve per context, an array of shape
    (contexts, max(120, largest SB) + 1) with NaN after each context's SB; ``par_bias`` gives a
    fit's bias table and ``corrected_par_curve(curves, ctx, par_bias)`` the corrected curves, in
    the same layout. ``par_lookup`` reads a curve at min(t, SB): a count beyond SB (0
    Legal_Balls left) takes the value at SB. ``first_innings_par`` does all of it for the
    first-innings rows that one fit serves and returns, per row, ``par_score`` = Par_Score(L),
    ``par_score_next`` = Par_Score(L + 1) and ``par_rate_win``, all from the corrected curve,
    where L is the row's Legal_Balls bowled, and ``par_score_raw`` = P_raw(L), so that the report
    can show both curves. The Post_Outcome_State of a hypothetical Legal_Ball reads
    ``par_score_next`` (Requirement 11, criterion 15).

    A row's Par_Score depends only on its Legal_Balls bowled, its context and the fit (with the
    fit's par means and bias), so two first-innings Deliveries of one view row fold (one fit,
    one M1 fit) with equal Legal_Balls bowled, Scoring_Environment, Venue_Effect and
    Scheduled_Balls get exactly equal values, whatever their runs, wickets and players
    (criterion 14). Contexts are matched on exact values; the M1 values of one fit for one
    competition season and venue are bitwise equal on every row. The par means and the bias come
    from the fit's own Training_Rows, so every Par_Score of a row served by M2^(k)(-j) uses no
    Delivery of folds j and k (Requirement 7, criterion 8).

Par_Rate (``par_rate``; glossary; Requirement 12, criterion 6)
    Par_Rate over the recent window of a first-innings Delivery is 6 x (Par_Score(L) -
    Par_Score(L - w)) / w in runs per over, where L is its Legal_Balls bowled and w its
    ``win_lb``, the Legal_Balls in the window of its ``win_rr`` (state, criterion 7 of
    Requirement 6): min(L, D6). So the window is the last D6 Legal_Balls, or every earlier
    Legal_Ball of the innings while fewer than D6 have been bowled, which gives 6 x
    Par_Score(L) / L because Par_Score(0) = 0. At the start of the innings, before its first
    Legal_Ball (L = 0, w = 0), the window is empty and Par_Rate is missing (NaN), as ``win_rr``
    is; Requirement 12, criterion 16 then gives the F23 component its missing-value rank. The
    Robustness_Suite's other windows pass their own ``win_lb`` (``state.recent_window``).

Achievable_Rate (``achievable_rate``; criterion 7)
    6 x the first-innings fit's calibrated expected remaining runs for the Delivery's M2
    features / its Legal_Balls left, in runs per over: the rate the chasing side could expect to
    score without the chase cap. It is NaN, never infinite, when no Legal_Ball is left. It is
    never below 0: above 0 with the Poisson objective, and 0 where an L2 fit's
    (``models.m2.objective`` ``regression``) Booster predicts 0 or less.

The ``remaining_runs`` Pipeline_Stage (``run_stage``)
    Reads ``balls`` and ``innings_info`` (``reconstruct``), ``state`` (``state``), the Fold_Table
    (``folds``) and the views V_1 ... V_D15 (``skill``), and the selected Extras_Encoding from the
    data-derived constant ``reconstruct.selected_encoding``. The rows are the ``state`` rows, and
    every frame is aligned to them; the Training_Rows of a fit are the ``common.training_mask``
    rows of its train folds with a finite target (capped variant) or, for the first-innings
    variant, those of innings 1. For each sandbox k, the stage:

    1. checks V_k with ``views.check_view`` before any fit reads it and builds the M2 features of
       every row from it (``m2_features``, ``post_wicket_features``);
    2. runs the D15 FitSpecs of sandbox k (outer k, then inner (k, j) by j) through
       ``common.fit_registered``, each once as CAPPED_MODEL and once as FIRST_INNINGS_MODEL, both
       of family FAMILY, so the fit ids are those of ``leakage.fit_id_for`` and
       ``views.view_plan`` (``m2_capped_lgbm_outer_k1``, ``m2_first_lgbm_inner_k1_j2``, ...).
       Every fit trains on rows of V_k only and is registered with the matches of its predicted
       fold (k for an outer fit, j for an inner one) as predicted matches. Its fit function
       (m2_fitter) calibrates the Booster on the fit's Training_Rows and returns the M2Model,
       whose Booster is saved in the checkpoint under its fit id (``store.save_model``, LightGBM
       model text) and whose calibration table joins LB_CALIBRATION_TABLE. Every prediction below
       is the M2Model's calibrated one;
    3. for each first-innings fit: its par_means and par_bias from its own Training_Rows, then the
       VIEW_M2_COLUMNS of the rows it serves (fold j of V_k for M2^(k)(-j), fold k for M2(-k);
       Requirement 7, criterion 8): ``par_score``, ``par_score_next`` and ``par_rate_win``
       (first_innings_par) on innings-1 rows; ``ach_rate`` (achievable_rate) on innings-2 rows;
       and ``f19_component``, the F19 Pillar_Component (Requirement 12, criteria 1 and 2):
       Par_Score minus the pre-ball team score in innings 1, in runs, and Required_Rate (the
       ``state`` column ``rrr``) minus Achievable_Rate in innings 2, in runs per over. A value is
       NaN where it does not apply or is undefined (``par_rate_win`` at 0 Legal_Balls bowled;
       ``ach_rate`` and ``f19_component`` with 0 Legal_Balls left or, for ``f19_component``, no
       Target), never infinite;
    4. extends V_k with these columns and ``m2_fit``, the id of the first-innings fit that served
       each row (``views.extend_view`` with ``views.view_plan(FIRST_INNINGS_MODEL, plan,
       family=FAMILY)``), checks it again with ``views.check_view`` and saves it as ``view_<k>``;
    5. predicts the fold-k rows with the outer fits: the capped M2(-k) on the pre-ball features
       (r_pre, the capped Out_Of_Fold_Predictions) and on the Post_Wicket_State features
       (r_post), and the first-innings M2(-k) on the innings-1 rows (its Out_Of_Fold_Predictions).

    The 20 inner capped fits supply no view column: every M2-derived quantity of the views comes
    from the first-innings fits, and each view row records one ``m2_fit``. They are fitted,
    registered and saved as the fold-sandbox plan specifies (design, "Fold sandboxes and leakage
    control"), so a later stage can predict the fold-j rows of V_k with them without a refit.

    Then the stage computes Wicket_Cost for every Delivery with ``value.wicket_cost`` (r_pre and
    r_post of the outer capped fit of its fold, ``value.terminal_post_wicket`` and the ``state``
    runs needed; Requirement 9, criteria 8-10 and 13) and saves PREDS_TABLE. It writes, in
    ``outputs/models/m2/``:

    * METRICS_FILE (METRICS_COLUMNS, criteria 5, 9, 10 and 15): per innings, the RMSE of M2's
      Out_Of_Fold_Predictions against the training target (``rmse_m2``), of the same fits' raw
      predictions before the Legal_Balls-left calibration (``rmse_m2_raw``) and of the lookup
      baseline (``rmse_lookup_baseline``), pooled over the Training_Rows of all folds, with the
      baseline's fill counts, and, for each Legal_Balls-left band of LB_BANDS (named in the
      metric, such as ``mean_pred_lb_left_2_6``), the mean target (``mean_target_lb_left_*``)
      and the mean Out_Of_Fold_Prediction before (``mean_pred_raw_lb_left_*``) and after
      (``mean_pred_lb_left_*``) the calibration over the same rows, for the capped variant
      (both innings) and the first-innings variant (innings 1); then, per innings, the
      Deliveries whose Omega was clipped low (``omega_clipped_low``) or, in innings 2, high
      (``omega_clipped_high``), those whose post-wicket runs were taken as 0
      (``post_wicket_runs_zero``), the innings-2 Deliveries with runs needed of 0 or less before
      the ball (``needed_nonpositive``; kept in the data) and the mean Omega (``omega_mean``).
      ``value`` is text; ``n_rows`` gives the rows evaluated or the Deliveries of the innings;
    * LB_CALIBRATION_FILE: the LB_CALIBRATION_TABLE of all 2 x D15 x D15 fits (criterion 15);
    * PAR_BIAS_FILE: the PAR_BIAS_TABLE of all first-innings fits;
    * OMEGA_TABLE_FILE (OMEGA_TABLE_COLUMNS) and OMEGA_CHART_FILE (criterion 12): the Deliveries
      and mean Omega of each innings x Legal_Balls left x wickets fallen cell, and a PNG heat map of
      the mean Omega by Legal_Balls left x wickets fallen, one panel per innings, drawn with
      matplotlib's Agg canvas at OMEGA_CHART_DPI.

    It saves LB_CALIBRATION_TABLE, PAR_MEANS_TABLE, PAR_BIAS_TABLE, FIT_SUMMARY_TABLE and the
    fit registry (``registry.save``), and records the boosting rounds of the 2 x D15 outer fits
    as the data-derived Method_Constant BEST_ITERATION_CONSTANT.

Reading the stage's outputs (``win_prob``, ``outcome``, ``model_choice``, ``pressure``, ``value``)
    * ``store.load_table(STAGE_NAME, views.view_table(k))`` gives V_k: one row per ``state`` row,
      in ``state`` order, with the ``skill`` columns, VIEW_M2_COLUMNS (float64), ``m1_fit`` and
      ``m2_fit`` (str). Read the views of this stage, not those of ``skill``. On innings-1 rows,
      ``par_score``, ``par_score_next`` and ``par_rate_win`` are set and ``ach_rate`` is NaN; on
      innings-2 rows the reverse. ``f19_component`` is Par_Score minus score in innings 1 and
      Required_Rate minus Achievable_Rate in innings 2: the M3 second-innings feature "Required_Rate
      minus Achievable_Rate" (Requirement 10, criterion 4) and, negated, the innings-2 F23
      component (Requirement 12, criterion 7). The fits named in ``m2_fit`` are in this stage's
      ``fit_registry``, which the runner reloads into ``ctx.registry`` under ``--from-stage``.
    * PREDS_TABLE (PREDS_COLUMNS): one row per ``state`` row, in ``state`` order: ``r_pre`` and
      ``r_post_wicket`` (0 where the Post_Wicket_State is terminal), ``omega`` and its two clip
      flags, all from the outer capped fit of the row's fold (``capped_fit``), and
      ``par_score_raw``, the raw Par_Score of innings-1 rows from the outer first-innings fit
      (``first_fit``; NaN in innings 2), which view V_k gives corrected as ``par_score``.
    * ``load_models(store, fit_ids)`` returns the calibrated M2Model of any of the 50 fits of
      this stage's ``fit_registry`` (``model`` CAPPED_MODEL or FIRST_INNINGS_MODEL,
      ``sandbox_k``, ``inner_j``): the Booster saved under its fit id, with its best round, and
      its rows of LB_CALIBRATION_TABLE. ``predict``, ``achievable_rate``, ``par_curve``,
      ``par_bias`` and ``first_innings_par`` take it. ``store.load_model(STAGE_NAME, fit_id)``
      alone gives the bare Booster, whose raw predictions ``predict`` refuses (module docstring,
      "Legal_Balls-left calibration"). PAR_MEANS_TABLE and PAR_BIAS_TABLE hold, per first-innings
      fit id and bucket, the par_means and par_bias that first_innings_par needs to rebuild
      Par_Score for other states with that model, for example
      ``table[table["fit_id"] == fit_id].set_index("bucket")``.

Errors: ValueError or TypeError, naming the problem, for malformed arguments, such as a missing
column, arrays of different lengths, a missing value where none may be, a target missing on a
Training_Row, a setup other than the M2 setup, an ``innings_info`` without the encoding's rows,
``team_runs`` that do not add up to the encoding's innings totals, a model whose predictions
are not finite, a bare LightGBM Booster given where an M2 fit is needed, or a calibration table
whose factors are not positive and non-decreasing in Legal_Balls left. The stage raises
FitLeakageError or ViewLeakageError (exit code 4) when a fit or
a view breaks the leakage guard (Requirement 7, criterion 12), and ValueError when its input
tables do not hold the same Deliveries in the same order.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import pandas as pd

from superr import folds, leakage, reconstruct, views
from superr.models import common
from superr.reconstruct import DEFAULT_SCHEDULED_BALLS
from superr.skill import STAGE_NAME as SKILL_STAGE
from superr.state import BALLS_PER_OVER, STATE_INNINGS, STATE_TABLE, Phase
from superr.state import STAGE_NAME as STATE_STAGE
from superr.value import TERMINAL_WICKETS, terminal_post_wicket, wicket_cost

if TYPE_CHECKING:
    from superr.pipeline import StageContext
    from superr.settings import Settings
    from superr.store import CheckpointStore

__all__ = [
    "ACH_RATE_COLUMN",
    "BASELINE_RMSE_COLUMNS",
    "BEST_ITERATION_CONSTANT",
    "CAPPED_MODEL",
    "F19_COLUMN",
    "FAMILY",
    "FIRST_INNINGS_MODEL",
    "FIT_SUMMARY_COLUMNS",
    "FIT_SUMMARY_TABLE",
    "LB_BANDS",
    "LB_CALIBRATION_COLUMNS",
    "LB_CALIBRATION_FILE",
    "LB_CALIBRATION_TABLE",
    "LB_CALIBRATION_TABLE_COLUMNS",
    "LOOKUP_CELL",
    "LOOKUP_COLUMNS",
    "LOOKUP_NEAREST_LB_LEFT",
    "LOOKUP_SAME_LB_LEFT",
    "M2_FEATURES",
    "M2_FOLDER",
    "M2_MONOTONE",
    "M2_OBJECTIVES",
    "METRICS_COLUMNS",
    "METRICS_FILE",
    "OMEGA_CHART_DPI",
    "OMEGA_CHART_FILE",
    "OMEGA_TABLE_COLUMNS",
    "OMEGA_TABLE_FILE",
    "PAR_BIAS_COLUMNS",
    "PAR_BIAS_FILE",
    "PAR_BIAS_TABLE",
    "PAR_BIAS_TABLE_COLUMNS",
    "PAR_BUCKETS",
    "PAR_COLUMNS",
    "PAR_CONTEXT_COLUMNS",
    "PAR_MEANS_TABLE",
    "PAR_MEANS_TABLE_COLUMNS",
    "PAR_MEAN_FEATURES",
    "PAR_RAW_COLUMN",
    "POST_WICKET_VIEW_COLUMNS",
    "PREDS_COLUMNS",
    "PREDS_TABLE",
    "RMSE_COLUMNS",
    "STAGE_NAME",
    "STATE_FEATURES",
    "TARGET_COLUMNS",
    "VIEW_FEATURES",
    "VIEW_M2_COLUMNS",
    "M2Model",
    "Regressor",
    "achievable_rate",
    "corrected_par_curve",
    "first_innings_par",
    "fit_m2",
    "lb_calibration",
    "load_models",
    "lookup_baseline",
    "lookup_baseline_rmse",
    "m2_features",
    "m2_fitter",
    "m2_setup",
    "m2_targets",
    "par_bias",
    "par_bucket",
    "par_contexts",
    "par_curve",
    "par_lookup",
    "par_means",
    "par_rate",
    "post_wicket_features",
    "predict",
    "rmse_by_innings",
    "run_stage",
]


# ------------------------------------------------------------------------------------- constants

#: The M2 features, in the column order of every M2 fit (criterion 3): pre-ball Legal_Balls left,
#: wickets fallen, RBQ, striker theta_bat, striker Legal_Balls faced in the innings, non-striker
#: theta_bat, Scoring_Environment, Venue_Effect and Match_Day_Conditions. Index a DataFrame with
#: ``list(M2_FEATURES)``: pandas reads a tuple as one column key.
M2_FEATURES: tuple[str, ...] = (
    "lb_left", "wickets", "rbq", "theta_striker", "striker_lb_faced", "theta_non_striker",
    "scoring_env", "venue_effect", "match_day",
)  # fmt: skip

#: The monotone constraint of each feature of M2_FEATURES, in order (criterion 4): non-decreasing
#: in Legal_Balls left and RBQ (+1), non-increasing in wickets fallen (-1), free otherwise (0).
M2_MONOTONE: tuple[int, ...] = (1, -1, 1, 0, 0, 0, 0, 0, 0)

#: The M2_FEATURES that m2_features takes from the ``state`` table; the others come from the view.
STATE_FEATURES: tuple[str, ...] = ("lb_left", "wickets", "striker_lb_faced")
VIEW_FEATURES: tuple[str, ...] = tuple(name for name in M2_FEATURES if name not in STATE_FEATURES)

#: The view columns of the Post_Wicket_State: the next batter's theta_bat and the RBQ without him.
POST_WICKET_VIEW_COLUMNS: tuple[str, ...] = ("theta_next_batter", "rbq_post_wicket")

#: Model names of the two M2 variants in the FitRegistry, and the family of both (fit ids from
#: ``leakage.fit_id_for``, such as ``m2_first_lgbm_inner_k1_j2``).
CAPPED_MODEL = "m2_capped"
FIRST_INNINGS_MODEL = "m2_first"
FAMILY = "lgbm"

#: The LightGBM objectives that ``models.m2.objective`` may name (module docstring, "Fits"):
#: Poisson regression with a log link (as delivered) and L2 regression. Both losses are minimised
#: by the conditional mean, so a prediction stays the expected remaining runs. ``common.lgbm_setup``
#: builds the parameters for its ``"regression"`` objective, whose L2 metric stays the
#: early-stopping metric, and m2_setup then sets the Config's objective.
M2_OBJECTIVES: tuple[str, ...] = ("poisson", "regression")
_SETUP_OBJECTIVE = "regression"
_OBJECTIVE_KEY = "objective"  # key of the M2 objective in the models.m2 section

#: The Config model key of M2 in ``common.LGBM_MODELS``: section ``models.m2``, seed ``seeds.m2``.
_CONFIG_MODEL = "m2"

#: Columns of a Legal_Balls-left calibration table (lb_calibration, M2Model.calibration), one row
#: per Legal_Balls-left value of a fit's Training_Rows, ascending: the value, the ratio of the
#: rows' summed targets to their summed raw predictions (NaN where those sum to 0), the factor
#: and the number of rows (module docstring, "Legal_Balls-left calibration").
LB_CALIBRATION_COLUMNS: tuple[str, ...] = ("lb_left", "ratio", "factor", "n_rows")

#: The smallest Legal_Balls-left calibration factor: a numerical safeguard, not a Method_Constant.
#: The isotonic fit gives a factor of 0 only where every Training_Row up to some Legal_Balls left
#: scored no run at all; the floor then keeps each calibrated prediction above 0, as a Poisson
#: prediction is, so a factor never removes the order of the Booster's predictions. No factor of
#: the delivered fits comes near it, so it changes no delivered value.
_MIN_LB_FACTOR = 1e-6

#: Columns of the m2_targets result, in order.
TARGET_COLUMNS: tuple[str, ...] = ("row", "match_id", "innings", "target", "needed_nonpositive")

#: How a lookup-baseline prediction was made (the ``fill`` column of lookup_baseline).
LOOKUP_CELL = 0  # mean of the Legal_Balls left x wickets fallen cell
LOOKUP_SAME_LB_LEFT = 1  # empty cell: mean of the same Legal_Balls left (criterion 5)
LOOKUP_NEAREST_LB_LEFT = 2  # that Legal_Balls left is empty too: mean of the nearest one with rows

#: Columns of the lookup_baseline result, of rmse_by_innings and of lookup_baseline_rmse.
LOOKUP_COLUMNS: tuple[str, ...] = ("pred", "fill")
RMSE_COLUMNS: tuple[str, ...] = ("innings", "n_rows", "rmse")
BASELINE_RMSE_COLUMNS: tuple[str, ...] = (*RMSE_COLUMNS, "n_same_lb_left_fill", "n_nearest_lb_left_fill")

#: The par buckets are 0 to PAR_BUCKETS: the Scheduled_Balls of a full T20 innings (20 Overs of 6
#: balls), fixed by the format rather than a Method_Constant (module docstring, "Par_Score").
PAR_BUCKETS = DEFAULT_SCHEDULED_BALLS

#: The par-state features that are bucket means of the fit's first-innings Training_Rows (DD-7).
PAR_MEAN_FEATURES: tuple[str, ...] = ("wickets", "rbq", "theta_striker", "striker_lb_faced", "theta_non_striker")

#: Columns of a par-bias table (par_bias), on the index ``bucket``: each bucket's bias and the
#: fit's first-innings Training_Rows in it.
PAR_BIAS_COLUMNS: tuple[str, ...] = ("beta", "n_rows")

#: Columns of a par-context table (par_contexts), and of the first_innings_par result: the view
#: columns PAR_COLUMNS, from the corrected curve, then PAR_RAW_COLUMN, the raw Par_Score at L.
PAR_CONTEXT_COLUMNS: tuple[str, ...] = ("scoring_env", "venue_effect", "sched_balls")
PAR_COLUMNS: tuple[str, ...] = ("par_score", "par_score_next", "par_rate_win")
PAR_RAW_COLUMN = "par_score_raw"

#: Match_Day_Conditions of every par state: scoring exactly as M1 expects (DD-7).
_PAR_MATCH_DAY = 0.0

_SHOWN = 5  # examples named in one error message


class Regressor(Protocol):
    """A fitted regression model: an M2Model (m2_fitter, fit_m2, load_models), or any object with ``predict``.

    ``predict`` receives a DataFrame with the M2_FEATURES columns in that order and returns one
    expected-remaining-runs value per row. A bare LightGBM Booster is not one: its raw predictions
    lack the Legal_Balls-left calibration, so the module's ``predict`` refuses it.
    """

    def predict(self, X: pd.DataFrame) -> Any: ...


# ---------------------------------------------------------------------------------- shared helpers

def _rows(count: int) -> str:
    return f"{count:,} row" if count == 1 else f"{count:,} rows"


def _require(frame: Any, columns: Sequence[str], what: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{what} must be a DataFrame; got {type(frame).__name__}")
    absent = [column for column in columns if column not in frame.columns]
    if absent:
        raise ValueError(f"{what} lacks the column(s) {', '.join(absent)}")


def _check_vector(values: Any, what: str) -> None:
    if values is None or isinstance(values, (str, bytes, Mapping, pd.DataFrame)) or np.ndim(values) != 1:
        raise TypeError(f"{what} must be a 1-D array or Series; got {type(values).__name__}")


def _check_length(array: np.ndarray, what: str, n_rows: int | None) -> np.ndarray:
    if n_rows is not None and len(array) != n_rows:
        raise ValueError(f"{what} has {len(array):,} values, but {n_rows:,} are needed, one per row")
    return array


def _is_pandas(values: Any) -> bool:
    return isinstance(values, (pd.Series, pd.Index, pd.api.extensions.ExtensionArray))


def _floats(values: Any, what: str, n_rows: int | None = None) -> np.ndarray:
    """``values`` (1-D, numeric) as a new float64 array, NaN for a missing value."""
    _check_vector(values, what)
    if _is_pandas(values):
        dtype = values.dtype
        if pd.api.types.is_bool_dtype(dtype) or not pd.api.types.is_numeric_dtype(dtype):
            raise TypeError(f"{what} must hold numbers; got dtype {dtype}")
        array = pd.Series(values).to_numpy(dtype=np.float64, na_value=np.nan)
    else:
        array = np.asarray(values)
        if array.dtype.kind not in "iuf":
            raise TypeError(f"{what} must hold numbers; got dtype {array.dtype}")
        array = array.astype(np.float64)
    return _check_length(array, what, n_rows)


def _ints(values: Any, what: str, n_rows: int | None = None, *, minimum: int | None = None) -> np.ndarray:
    """``values`` (1-D whole numbers, none missing) as an int64 array."""
    _check_vector(values, what)
    if _is_pandas(values):
        dtype = values.dtype
        if pd.api.types.is_bool_dtype(dtype) or not pd.api.types.is_numeric_dtype(dtype):
            raise TypeError(f"{what} must hold whole numbers; got dtype {dtype}")
        series = pd.Series(values)
        missing = int(series.isna().sum())
        if missing:
            raise ValueError(f"{what} has {missing:,} missing value(s)")
        if pd.api.types.is_integer_dtype(dtype):
            array = series.to_numpy(dtype=np.int64)
        else:
            array = _whole_floats(series.to_numpy(dtype=np.float64), what)
    else:
        raw = np.asarray(values)
        if raw.dtype.kind in "iu":
            array = raw.astype(np.int64)
        elif raw.dtype.kind == "f":
            missing = int(np.isnan(raw).sum())
            if missing:
                raise ValueError(f"{what} has {missing:,} missing value(s)")
            array = _whole_floats(raw.astype(np.float64), what)
        else:
            raise TypeError(f"{what} must hold whole numbers; got dtype {raw.dtype}")
    _check_length(array, what, n_rows)
    if minimum is not None and len(array) and int(array.min()) < minimum:
        raise ValueError(f"{what} must be {minimum} or more; {int((array < minimum).sum()):,} value(s) are below")
    return array


def _whole_floats(values: np.ndarray, what: str) -> np.ndarray:
    whole = np.isfinite(values) & (np.floor(values) == values) & (np.abs(values) <= 2.0**53)
    if not whole.all():
        raise ValueError(f"{what} must hold whole numbers; {int((~whole).sum()):,} value(s) are not")
    return values.astype(np.int64)


def _flags(values: Any, what: str, n_rows: int | None = None) -> np.ndarray:
    """``values`` (1-D booleans, none missing) as a bool array."""
    _check_vector(values, what)
    series = pd.Series(values) if _is_pandas(values) else pd.Series(np.asarray(values))
    if not pd.api.types.is_bool_dtype(series.dtype):
        raise TypeError(f"{what} must be boolean; got dtype {series.dtype}")
    missing = int(series.isna().sum())
    if missing:
        raise ValueError(f"{what} has {missing:,} missing value(s)")
    return _check_length(series.to_numpy(dtype=bool), what, n_rows)


def _same_rows(view: pd.DataFrame, state: pd.DataFrame) -> None:
    """ValueError unless ``view`` and ``state`` hold the same rows in the same order."""
    if len(view) != len(state):
        raise ValueError(
            f"view has {_rows(len(view))} but state has {_rows(len(state))}; pass a sandbox view and the state table, "
            "which hold one row per innings 1-2 Delivery in the same order"
        )
    if "row" in view.columns and "row" in state.columns:
        view_rows = _ints(view["row"], "view column 'row'")
        state_rows = _ints(state["row"], "state column 'row'")
        if not np.array_equal(view_rows, state_rows):
            differ = int((view_rows != state_rows).sum())
            raise ValueError(
                f"view and state hold different rows, or the same rows in another order, at {_rows(differ)}; "
                "a sandbox view follows the row order of the state table"
            )


def _shown(values: np.ndarray) -> str:
    return ", ".join(map(str, values[:_SHOWN].tolist()))


# ============================================================================== targets (9.1, 9.2)

_TARGET_BALL_COLUMNS: tuple[str, ...] = ("row", "match_id", "innings", "innings_ball_num", "team_runs")
_TARGET_INFO_COLUMNS: tuple[str, ...] = ("encoding", "match_id", "innings", "target", "total")


def m2_targets(balls: pd.DataFrame, innings_info: pd.DataFrame, *, capped: bool, encoding: str) -> pd.DataFrame:
    """The M2 training target of every innings 1-2 Delivery (criteria 1 and 2; module docstring, "Targets").

    ``balls`` is the ``balls`` table of the ``reconstruct`` checkpoint (``row``, ``match_id``,
    ``innings``, ``innings_ball_num`` and ``team_runs``, the team runs under the selected
    Extras_Encoding); ``innings_info`` its ``innings_info`` table (every candidate encoding);
    ``encoding`` the selected Extras_Encoding, the data-derived Method_Constant
    ``reconstruct.selected_encoding``. ``capped`` picks the variant: False for the remaining runs
    (the first-innings variant), True for the capped targets (innings 2: min(remaining runs,
    max(runs needed, 0))).

    Returns the TARGET_COLUMNS on a RangeIndex, one row per Delivery of innings 1 and 2 in the
    Requirement 5 order (the order of the ``state`` table): ``row``, ``match_id`` and ``innings``
    (int64); ``target`` (float64, 0 or more; NaN for an innings-2 Delivery without a Target in
    the capped variant); and ``needed_nonpositive`` (bool), true on the innings-2 Deliveries
    whose runs needed before the ball is 0 or less.

    Raises ValueError when a column is absent or holds a missing or malformed value, a team-runs
    value is negative, ``innings_info`` has no row of ``encoding`` for an innings of ``balls`` or
    repeats one, or the summed ``team_runs`` of an innings differ from that encoding's total; and
    TypeError for a ``capped`` other than True or False or an ``encoding`` that is not a
    non-empty str.
    """
    _require(balls, _TARGET_BALL_COLUMNS, "balls")
    _require(innings_info, _TARGET_INFO_COLUMNS, "innings_info")
    if not isinstance(capped, (bool, np.bool_)):
        raise TypeError(f"capped must be True or False; got {capped!r}")
    if not isinstance(encoding, str) or not encoding:
        raise TypeError(f"encoding must be the name of the selected Extras_Encoding, a non-empty str; got {encoding!r}")

    all_innings = _ints(balls["innings"], "balls column 'innings'")
    keep = np.flatnonzero(np.isin(all_innings, STATE_INNINGS))
    part = balls.iloc[keep]
    row = _ints(part["row"], "balls column 'row'")
    match = _ints(part["match_id"], "balls column 'match_id'")
    innings = all_innings[keep]
    ball = _ints(part["innings_ball_num"], "balls column 'innings_ball_num'")
    runs = _ints(part["team_runs"], "balls column 'team_runs'", minimum=0)
    order = np.lexsort((ball, innings, match))  # stable: the Requirement 5 order
    row, match, innings, runs = row[order], match[order], innings[order], runs[order]
    n_rows = len(row)
    if not n_rows:
        return pd.DataFrame(
            {
                "row": np.zeros(0, dtype=np.int64),
                "match_id": np.zeros(0, dtype=np.int64),
                "innings": np.zeros(0, dtype=np.int64),
                "target": np.zeros(0, dtype=np.float64),
                "needed_nonpositive": np.zeros(0, dtype=bool),
            },
            columns=list(TARGET_COLUMNS),
        )

    first = np.ones(n_rows, dtype=bool)
    first[1:] = (match[1:] != match[:-1]) | (innings[1:] != innings[:-1])
    starts = np.flatnonzero(first)
    group = np.cumsum(first) - 1
    totals = np.add.reduceat(runs, starts)
    before = np.cumsum(runs) - runs
    score = before - before[starts][group]  # team score before each Delivery, in its innings
    remaining = totals[group] - score  # team runs from the Delivery, inclusive, to the end of the innings

    chase = innings == 2
    target = _encoding_targets(innings_info, encoding, match[starts], innings[starts], totals)[group]
    has_target = chase & ~np.isnan(target)
    needed = np.where(has_target, target - score, np.nan)
    nonpositive = has_target & (np.nan_to_num(needed, nan=1.0) <= 0.0)

    values = remaining.astype(np.float64)
    if capped:
        chase_target = np.minimum(values, np.maximum(np.nan_to_num(needed, nan=0.0), 0.0))
        values = np.where(chase, np.where(has_target, chase_target, np.nan), values)
    return pd.DataFrame(
        {"row": row, "match_id": match, "innings": innings, "target": values, "needed_nonpositive": nonpositive},
        columns=list(TARGET_COLUMNS),
    )


def _encoding_targets(
    innings_info: pd.DataFrame, encoding: str, match: np.ndarray, innings: np.ndarray, totals: np.ndarray
) -> np.ndarray:
    """The Target (NaN when missing) of each innings (``match``, ``innings``) under ``encoding``.

    ``totals`` holds each innings' summed ``team_runs``; ValueError unless every innings has one
    ``innings_info`` row of the encoding whose ``total`` equals it.
    """
    names = innings_info["encoding"]
    rows = np.flatnonzero(names.eq(encoding).fillna(False).to_numpy(dtype=bool))
    if not len(rows):
        held = ", ".join(map(repr, sorted({str(name) for name in names.dropna().unique()}))) or "none"
        raise ValueError(f"innings_info has no row of the Extras_Encoding {encoding!r}; it holds {held}")
    info = innings_info.iloc[rows]
    what = f"innings_info rows of {encoding!r}"
    keys = pd.MultiIndex.from_arrays(
        [_ints(info["match_id"], f"{what}: match_id"), _ints(info["innings"], f"{what}: innings")]
    )
    if not keys.is_unique:
        raise ValueError(f"{what} repeat a (match_id, innings); innings_info holds one row per encoding and innings")
    at = keys.get_indexer(pd.MultiIndex.from_arrays([match, innings]))
    absent = at < 0
    if absent.any():
        raise ValueError(
            f"{what} lack {int(absent.sum()):,} innings that balls holds (for example match {_shown(match[absent])})"
        )
    total = _ints(info["total"], f"{what}: total")[at]
    differ = np.flatnonzero(total != totals)
    if len(differ):
        shown = ", ".join(
            f"match {match[i]} innings {innings[i]} (team_runs {totals[i]:,}, total {total[i]:,})"
            for i in differ[:_SHOWN].tolist()
        )
        raise ValueError(
            f"the team_runs of balls do not add up to the innings totals of the Extras_Encoding {encoding!r} in "
            f"{len(differ):,} innings ({shown}); pass the balls table and the encoding that the reconstruct stage "
            "selected"
        )
    return _floats(info["target"], f"{what}: target")[at]


# ===================================================================== features (9.3, Post_Wicket_State)

def m2_features(view: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
    """The M2_FEATURES of every row: the pre-ball state of each Delivery as M2 reads it (criterion 3).

    ``view`` is a sandbox view V_k (``rbq``, ``theta_striker``, ``theta_non_striker``,
    ``scoring_env``, ``venue_effect``, ``match_day``) and ``state`` the ``state`` table
    (``lb_left``, ``wickets``, ``striker_lb_faced``), with the same rows in the same order; when
    both have ``row``, the two columns must be equal.

    Returns the M2_FEATURES as float64 columns, in that order, on a RangeIndex, one row per row
    of ``state``. A missing value is NaN. Raises ValueError when a column is absent or the rows
    differ, and TypeError for a non-numeric column.
    """
    _require(view, VIEW_FEATURES, "view")
    _require(state, STATE_FEATURES, "state")
    _same_rows(view, state)
    data = {
        name: _floats((state if name in STATE_FEATURES else view)[name], f"column {name!r}") for name in M2_FEATURES
    }
    return pd.DataFrame(data, columns=list(M2_FEATURES))


def post_wicket_features(view: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
    """The M2_FEATURES of every row's Post_Wicket_State (glossary; Requirement 9, criterion 8).

    The arguments are those of m2_features; ``view`` also needs ``theta_next_batter`` and
    ``rbq_post_wicket``. Returns the m2_features frame after one hypothetical dismissal of the
    striker: ``wickets`` + 1, ``theta_striker`` = ``theta_next_batter``, ``striker_lb_faced`` =
    0 and ``rbq`` = ``rbq_post_wicket``; the other features unchanged. ``theta_striker`` is NaN
    where Batters_To_Come is empty; criterion 13 takes the post-wicket runs of such rows, and of
    rows with 9 wickets fallen, as 0 (module docstring, "Features"). Raises the errors of
    m2_features.
    """
    _require(view, POST_WICKET_VIEW_COLUMNS, "view")
    frame = m2_features(view, state)
    frame["wickets"] = frame["wickets"].to_numpy() + 1.0
    frame["theta_striker"] = _floats(view["theta_next_batter"], "column 'theta_next_batter'")
    frame["striker_lb_faced"] = 0.0
    frame["rbq"] = _floats(view["rbq_post_wicket"], "column 'rbq_post_wicket'")
    return frame


def _feature_frame(X: Any) -> pd.DataFrame:
    """``X``'s M2_FEATURES as a new float64 DataFrame, in M2_FEATURES order, on a RangeIndex."""
    _require(X, M2_FEATURES, "X")
    data = {name: _floats(X[name], f"X column {name!r}") for name in M2_FEATURES}
    return pd.DataFrame(data, columns=list(M2_FEATURES))


# ================================================================================== fits (9.4)

def m2_setup(settings: Settings) -> common.BoosterSetup:
    """The LightGBM parameters of every M2 fit, capped and first-innings (criterion 4; module docstring, "Fits").

    ``common.lgbm_setup(settings, "m2", "regression", monotone_constraints=M2_MONOTONE)`` builds
    them from the ``models.m2`` section, ``runtime.n_threads``, ``runtime.deterministic`` and
    ``seeds.m2``, with the L2 early-stopping metric; the objective is then set to
    ``models.m2.objective`` (``poisson`` as delivered). Everything else, early stopping included,
    is as lgbm_setup gives it. Raises the errors of lgbm_setup, such as KeyError for a setting
    absent from the Config; TypeError when ``models.m2.objective`` is not a str; and ValueError
    when it is not one of M2_OBJECTIVES.
    """
    section, _ = common.LGBM_MODELS[_CONFIG_MODEL]
    path = f"{section}.{_OBJECTIVE_KEY}"
    objective = settings.get(path)
    if not isinstance(objective, str):
        raise TypeError(f"{path} must be the name of a LightGBM objective, a str; got {objective!r}")
    if objective not in M2_OBJECTIVES:
        raise ValueError(
            f"{path} must be one of {', '.join(map(repr, M2_OBJECTIVES))}, the LightGBM regression objectives M2 "
            f"supports (M2_OBJECTIVES); got {objective!r}"
        )
    setup = common.lgbm_setup(settings, _CONFIG_MODEL, _SETUP_OBJECTIVE, monotone_constraints=M2_MONOTONE)
    params = setup.params
    params["objective"] = objective
    return common.BoosterSetup(
        library=setup.library,
        param_items=tuple(params.items()),
        max_rounds=setup.max_rounds,
        early_stopping=setup.early_stopping,
    )


def _check_setup(setup: Any) -> None:
    if not isinstance(setup, common.BoosterSetup):
        raise TypeError(f"setup must be the BoosterSetup of m2_setup; got {type(setup).__name__}")
    params = setup.params
    constraints = tuple(params.get("monotone_constraints", ()))
    if (
        setup.library != common.LIGHTGBM
        or params.get("objective") not in M2_OBJECTIVES
        or constraints != M2_MONOTONE
        or "monotone_constraints_method" not in params
    ):
        raise ValueError(
            f"setup must be the M2 setup of m2_setup: a LightGBM regression with an objective of M2_OBJECTIVES "
            f"({', '.join(map(repr, M2_OBJECTIVES))}), monotone_constraints {list(M2_MONOTONE)} in M2_FEATURES order "
            f"and a monotone_constraints_method (criterion 4); got library {setup.library!r}, objective "
            f"{params.get('objective')!r} and constraints {list(constraints)}"
        )


def m2_fitter(setup: common.BoosterSetup, X: pd.DataFrame, y: Any) -> common.FitFunction:
    """The fit function of an M2 fit, for ``common.fit_registered`` (module docstring, "Fits").

    ``setup`` is the BoosterSetup of m2_setup. ``X`` (with the M2_FEATURES columns, in any order;
    m2_features gives them) and ``y`` (the ``target`` column of m2_targets, NaN where there is
    none) hold every candidate row, in the order of the ``match_ids`` given to fit_registered,
    whose ``eligible`` must leave out the rows without a target. The returned function fits a
    LightGBM regression with the setup's objective (``models.m2.objective``) on ``rows.train``
    with the features in M2_FEATURES order, the monotone constraints M2_MONOTONE and early
    stopping on the L2 loss of ``rows.early_stopping``, keeps the Booster of the best round,
    calibrates it by Legal_Balls left on ``rows.training`` (the fitted and the early-stopping
    rows; lb_calibration) and returns the M2Model, whose ``predict`` gives the calibrated
    expected remaining runs (criterion 15).

    Raises TypeError or ValueError for a setup other than the M2 setup, a missing column or a
    ``y`` of the wrong length. The returned function raises ValueError when a target is missing,
    not finite or negative on a Training_Row, when there is no early-stopping row, and the errors
    of lb_calibration.
    """
    _check_setup(setup)
    features = _feature_frame(X)
    target = _floats(y, "y", len(features))
    balls_left = features["lb_left"].to_numpy()
    fit = common.boosting_fitter(setup, features, target)

    def fit_m2_rows(rows: common.FitRows) -> M2Model:
        if not isinstance(rows, common.FitRows):
            raise TypeError(f"rows must be the FitRows of common.fit_registered; got {type(rows).__name__}")
        training = rows.training
        bad = ~np.isfinite(target[training])
        if bad.any():
            raise ValueError(
                f"y is missing or not finite on {_rows(int(bad.sum()))} of the fit's Training_Rows; leave the rows "
                "without an M2 target out of eligible"
            )
        booster = fit(rows)
        raw = _raw_predictions(booster, features.iloc[training])
        return M2Model(booster, lb_calibration(raw, target[training], balls_left[training]))

    return fit_m2_rows


def fit_m2(X: pd.DataFrame, y: Any, groups: Any, setup: common.BoosterSetup, *, fit_index: int) -> M2Model:
    """One M2 fit on rows that are all its Training_Rows, without a FitRegistry (module docstring, "Fits").

    ``X``, ``y`` and ``groups`` (the match id of each row) describe the same rows. The whole
    matches of ``leakage.early_stopping_split(groups, es_share, seed, fit_index)`` (share and
    seed from ``setup.early_stopping``; ``fit_index`` the fit's position in
    ``folds.sandbox_plan(D15)``) are held out for early stopping, exactly as
    ``common.fit_registered`` holds them out for the same rows, and the model is fitted on the
    other rows by m2_fitter. Returns the M2Model: the Booster of the best round, calibrated by
    Legal_Balls left on all the given rows, as the stage calibrates a fit on its Training_Rows.

    A fit whose output reaches a Delivery must go through ``common.fit_registered`` instead, so
    that the leakage guard checks it (Requirement 7, criterion 6). Raises the errors of
    m2_fitter and of early_stopping_split.
    """
    _check_setup(setup)
    features = _feature_frame(X)
    match_ids = _ints(groups, "groups", len(features))
    if isinstance(fit_index, (bool, np.bool_)) or not isinstance(fit_index, Integral):
        raise TypeError(f"fit_index must be a whole number; got {fit_index!r}")
    early = setup.early_stopping
    _, held_matches = leakage.early_stopping_split(match_ids, early.share, early.seed, int(fit_index))
    held = np.isin(match_ids, held_matches)
    rows = common.FitRows(train=np.flatnonzero(~held), early_stopping=np.flatnonzero(held))
    return m2_fitter(setup, features, y)(rows)


def predict(model: Regressor, X: pd.DataFrame) -> np.ndarray:
    """M2's expected remaining runs for each row of ``X`` (columns M2_FEATURES; m2_features gives them).

    ``model`` is an M2Model (m2_fitter, fit_m2, load_models), whose prediction is the calibrated
    one of criterion 15, with its Booster at the best round; or any other Regressor, such as a
    test stub. The features are passed to ``model.predict`` as a float64 DataFrame in
    M2_FEATURES order. Returns a float64 array, one value per row. Raises TypeError when
    ``model`` has no ``predict`` or is a bare LightGBM Booster, whose raw predictions are not
    M2's (wrap it with its calibration: M2Model, load_models), and ValueError for a missing
    column or when the predictions have the wrong shape or are not finite.
    """
    if not callable(getattr(model, "predict", None)):
        raise TypeError(f"model must be a fitted regression model with predict; got {type(model).__name__}")
    if _is_bare_booster(model):
        raise TypeError(
            "model is a bare LightGBM Booster, whose raw predictions lack the Legal_Balls-left calibration of "
            "criterion 15; pass the M2Model of m2_fitter or fit_m2, or load the fit with load_models"
        )
    features = _feature_frame(X)
    if not len(features):
        return np.zeros(0, dtype=np.float64)
    values = np.asarray(model.predict(features), dtype=np.float64)
    if values.shape != (len(features),):
        raise ValueError(
            f"model.predict returned shape {values.shape} for {_rows(len(features))}; one value per row is needed"
        )
    bad = ~np.isfinite(values)
    if bad.any():
        raise ValueError(f"model.predict returned {int(bad.sum()):,} missing or non-finite value(s)")
    return values


# ============================================== the Legal_Balls-left calibration (9.15; DD-15)

def _is_bare_booster(model: Any) -> bool:
    """True for a LightGBM Booster, told by its class so that lightgbm need not be imported here."""
    kind = type(model)
    return kind.__name__ == "Booster" and kind.__module__.split(".")[0] == "lightgbm"


def _raw_predictions(booster: Any, features: pd.DataFrame) -> np.ndarray:
    """The Booster's own predictions of ``features`` (an M2_FEATURES frame) as float64, one per row."""
    if not len(features):
        return np.zeros(0, dtype=np.float64)
    values = np.asarray(booster.predict(features), dtype=np.float64)
    if values.shape != (len(features),):
        raise ValueError(
            f"the Booster returned shape {values.shape} for {_rows(len(features))}; one value per row is needed"
        )
    return values


def lb_calibration(raw: Any, y: Any, lb_left: Any) -> pd.DataFrame:
    """The Legal_Balls-left calibration table of one M2 fit, from its Training_Rows (criterion 15; DD-15).

    ``raw`` holds the Booster's predictions of the fit's Training_Rows (the rows it was fitted on
    and its early-stopping rows), ``y`` their M2 targets and ``lb_left`` their pre-ball
    Legal_Balls left, one value per row. Each raw prediction below 0 counts as 0 (only an L2 fit
    predicts below 0). For each Legal_Balls-left value l of the rows, ``ratio`` is the sum of y
    over the rows with l divided by the sum of their raw predictions. ``factor`` is the isotonic
    (non-decreasing) regression of the ratios on l, weighted by those sums of raw predictions, so
    a block of pooled values takes its summed targets over its summed raw predictions; a factor
    below _MIN_LB_FACTOR is raised to it. A value l whose raw predictions sum to 0 has no ratio
    (NaN), stays out of the regression and takes the factor of the nearest lower l that has one,
    or of the lowest such l when none is lower (module docstring, "Legal_Balls-left calibration").

    Returns LB_CALIBRATION_COLUMNS on a RangeIndex, one row per distinct l in ascending order:
    ``lb_left`` and ``n_rows`` (int64), ``ratio`` and ``factor`` (float64). The factors are
    finite, at least _MIN_LB_FACTOR and non-decreasing in l. Raises ValueError for arrays of
    different lengths, no row, a raw prediction that is missing or not finite, a target that is
    missing, not finite or negative, raw predictions that are all 0 or less, or a malformed
    ``lb_left`` (missing, fractional or negative); TypeError for values that are not numbers.
    """
    predicted = _floats(raw, "raw")
    n_rows = len(predicted)
    target = _floats(y, "y", n_rows)
    balls = _ints(lb_left, "lb_left", n_rows, minimum=0)
    if not n_rows:
        raise ValueError("the Legal_Balls-left calibration needs the fit's Training_Rows; no row was given")
    bad = ~np.isfinite(predicted)
    if bad.any():
        raise ValueError(f"raw is missing or not finite on {_rows(int(bad.sum()))}")
    bad = ~np.isfinite(target) | (np.nan_to_num(target, nan=0.0) < 0.0)
    if bad.any():
        raise ValueError(
            f"y is missing, not finite or negative on {_rows(int(bad.sum()))}; an M2 target is a number of runs, 0 "
            "or more"
        )
    levels, inverse, counts = np.unique(balls, return_inverse=True, return_counts=True)
    inverse = np.asarray(inverse).reshape(-1)
    raw_sum = np.bincount(inverse, weights=np.maximum(predicted, 0.0), minlength=len(levels))
    target_sum = np.bincount(inverse, weights=target, minlength=len(levels))
    known = np.flatnonzero(raw_sum > 0.0)
    if not len(known):
        raise ValueError(
            f"every raw prediction of the {_rows(n_rows)} is 0 or less, so no Legal_Balls-left value has a ratio of "
            "targets to raw predictions"
        )
    ratio = np.full(len(levels), np.nan)
    ratio[known] = target_sum[known] / raw_sum[known]
    from sklearn.isotonic import isotonic_regression

    fitted = np.asarray(isotonic_regression(ratio[known], sample_weight=raw_sum[known], increasing=True), dtype=float)
    if not np.isfinite(fitted).all():
        raise ValueError("the isotonic regression of the Legal_Balls-left ratios gave a value that is not finite")
    # The floor, then a running maximum that only guards the order against rounding in the regression.
    fitted = np.maximum.accumulate(np.maximum(fitted, _MIN_LB_FACTOR))
    source = np.clip(np.searchsorted(known, np.arange(len(levels)), side="right") - 1, 0, None)
    return pd.DataFrame(
        {
            "lb_left": levels.astype(np.int64),
            "ratio": ratio,
            "factor": fitted[source],
            "n_rows": counts.astype(np.int64),
        },
        columns=list(LB_CALIBRATION_COLUMNS),
    )


def _check_calibration(calibration: Any) -> pd.DataFrame:
    """``calibration`` (LB_CALIBRATION_COLUMNS) checked, as a new table sorted by ``lb_left`` on a RangeIndex."""
    _require(calibration, LB_CALIBRATION_COLUMNS, "calibration")
    if not len(calibration):
        raise ValueError(
            "calibration has no row; a fit's table holds one row per Legal_Balls-left value of its Training_Rows"
        )
    lb = _ints(calibration["lb_left"], "calibration column 'lb_left'", minimum=0)
    n_values = len(lb)
    ratio = _floats(calibration["ratio"], "calibration column 'ratio'", n_values)
    factor = _floats(calibration["factor"], "calibration column 'factor'", n_values)
    counts = _ints(calibration["n_rows"], "calibration column 'n_rows'", n_values, minimum=0)
    order = np.argsort(lb, kind="stable")
    lb, ratio, factor, counts = lb[order], ratio[order], factor[order], counts[order]
    repeated = np.flatnonzero(np.diff(lb) == 0)
    if len(repeated):
        raise ValueError(
            f"calibration repeats the Legal_Balls-left value(s) {_shown(lb[repeated])}; a fit's table holds each value "
            "once"
        )
    if not (np.isfinite(factor) & (factor > 0.0)).all():
        raise ValueError("calibration holds a factor that is missing, not finite or not above 0")
    falls = np.flatnonzero(np.diff(factor) < 0.0)
    if len(falls):
        raise ValueError(
            f"the calibration factor falls as Legal_Balls left rise from {_shown(lb[falls])}; it must not decrease "
            "(Requirement 9, criteria 4 and 15)"
        )
    return pd.DataFrame(
        {"lb_left": lb, "ratio": ratio, "factor": factor, "n_rows": counts}, columns=list(LB_CALIBRATION_COLUMNS)
    )


class M2Model:
    """One fitted M2: its Booster and the Legal_Balls-left calibration of its Training_Rows (criterion 15; DD-15).

    ``M2Model(booster, calibration)``: ``booster`` is the fit's LightGBM Booster (or any object,
    other than an M2Model, whose ``predict`` takes an M2_FEATURES frame and returns one value per
    row); ``calibration`` its lb_calibration table, LB_CALIBRATION_COLUMNS (a fit's rows of
    LB_CALIBRATION_TABLE give it), checked: the Legal_Balls-left values whole, 0 or more and
    distinct, the factors finite, above 0 and non-decreasing in Legal_Balls left.

    * ``predict(X)``: the calibrated expected remaining runs of each row of ``X`` (the
      M2_FEATURES columns, in any order), factor(Legal_Balls left) x max(raw prediction, 0).
      ``remaining_runs.predict``, and so every function of this module that takes a Regressor,
      calls it.
    * ``predict_raw(X)``: the Booster's own prediction, before the calibration, for diagnostics.
    * ``factor(lb_left)``: the factor of each Legal_Balls-left value, a step function of the
      table: the factor of the largest value of the table at or below it, or of the smallest
      value when it lies below them all, so the factor is flat beyond the values of the
      Training_Rows. ValueError for a missing value.
    * ``booster``, ``calibration`` (a copy of the checked table, sorted by ``lb_left``) and
      ``best_iteration`` (the Booster's, None when it has none).

    The factor depends on Legal_Balls left alone, is above 0 and never decreases, so the
    calibrated predictions keep every monotone direction of the Booster's (module docstring,
    "Legal_Balls-left calibration"). Raises TypeError for a ``booster`` without ``predict`` or that
    is an M2Model, and ValueError (or TypeError) for a malformed ``calibration``.
    """

    __slots__ = ("_booster", "_factor", "_lb_left", "_table")

    def __init__(self, booster: Any, calibration: pd.DataFrame) -> None:
        if isinstance(booster, M2Model):
            raise TypeError("booster is already an M2Model; pass its booster to calibrate it again")
        if not callable(getattr(booster, "predict", None)):
            raise TypeError(f"booster must be a fitted model with predict; got {type(booster).__name__}")
        table = _check_calibration(calibration)
        self._booster = booster
        self._table = table
        self._lb_left = table["lb_left"].to_numpy(dtype=np.float64)
        self._factor = table["factor"].to_numpy(dtype=np.float64)
        self._lb_left.flags.writeable = False
        self._factor.flags.writeable = False

    @property
    def booster(self) -> Any:
        """The fit's Booster, whose ``predict`` gives the raw predictions (with its best round)."""
        return self._booster

    @property
    def calibration(self) -> pd.DataFrame:
        """A copy of the calibration table (LB_CALIBRATION_COLUMNS), sorted by ``lb_left``."""
        return self._table.copy()

    @property
    def best_iteration(self) -> Any:
        """The boosting round that the Booster predicts with (``best_iteration``), None when it has none."""
        return getattr(self._booster, "best_iteration", None)

    def factor(self, lb_left: Any) -> np.ndarray:
        """The calibration factor of each Legal_Balls-left value of ``lb_left`` (1-D), as float64."""
        balls = _floats(lb_left, "lb_left")
        missing = np.isnan(balls)
        if missing.any():
            raise ValueError(
                f"lb_left is missing on {_rows(int(missing.sum()))}; the Legal_Balls-left calibration needs it"
            )
        at = np.searchsorted(self._lb_left, balls, side="right") - 1
        return self._factor[np.clip(at, 0, len(self._factor) - 1)]

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        """The Booster's own prediction for each row of ``X`` (M2_FEATURES), before the calibration."""
        return _raw_predictions(self._booster, _feature_frame(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """The calibrated expected remaining runs of each row of ``X`` (M2_FEATURES), as float64."""
        features = _feature_frame(X)
        raw = _raw_predictions(self._booster, features)
        return self.factor(features["lb_left"].to_numpy()) * np.maximum(raw, 0.0)

    def __repr__(self) -> str:
        return (
            f"M2Model({type(self._booster).__name__}, factor {self._factor[0]:.4f} at {int(self._lb_left[0])} to "
            f"{self._factor[-1]:.4f} at {int(self._lb_left[-1])} Legal_Balls left)"
        )


def load_models(store: CheckpointStore, fit_ids: Iterable[str]) -> dict[str, M2Model]:
    """The calibrated M2 fits ``fit_ids`` of the ``remaining_runs`` checkpoint, keyed by fit id (criterion 15).

    Each value is the M2Model of the Booster saved under the fit id (``store.load_model``, with
    its best round) and the fit's rows of LB_CALIBRATION_TABLE. Wherever an M2 prediction is
    needed, load the fit with this function rather than with ``store.load_model`` alone, whose
    Booster ``predict`` refuses. Raises TypeError for a fit id that is not a non-empty str or a
    single str for ``fit_ids``, ValueError when LB_CALIBRATION_TABLE has no row for a fit or
    malformed rows, and the errors of ``store.load_table`` and ``store.load_model`` (StageError
    when a file is missing, changed or unreadable).
    """
    if isinstance(fit_ids, (str, bytes)):
        raise TypeError("fit_ids must be a collection of fit ids, not one string")
    ids = list(fit_ids)
    for fit_id in ids:
        if not isinstance(fit_id, str) or not fit_id:
            raise TypeError(f"every fit id must be a non-empty str; got {fit_id!r}")
    table = store.load_table(STAGE_NAME, LB_CALIBRATION_TABLE)
    what = f"the {STAGE_NAME} table {LB_CALIBRATION_TABLE}"
    _require(table, LB_CALIBRATION_TABLE_COLUMNS, what)
    names = table["fit_id"].astype(str).to_numpy(dtype=object)
    models: dict[str, M2Model] = {}
    for fit_id in ids:
        mine = np.flatnonzero(names == fit_id)
        if not len(mine):
            raise ValueError(
                f"{what} has no row for the M2 fit {fit_id!r}; pass the fit ids of this checkpoint's fit_registry"
            )
        calibration = table.iloc[mine][list(LB_CALIBRATION_COLUMNS)].reset_index(drop=True)
        models[fit_id] = M2Model(store.load_model(STAGE_NAME, fit_id), calibration)
    return models


# ==================================================================== the lookup baseline (9.5)

def _nearest_means(sums: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean of each index; an index without rows takes the mean over the nearest index with rows.

    ``sums`` and ``counts`` hold, per index, the sum and the number of values. An index with no
    value takes the mean over the values of the nearest index that has some, pooling the values
    of both neighbours when two are equally near. Returns (means, borrowed), ``borrowed`` true
    where the mean came from a neighbour. Raises ValueError when no index has a value.
    """
    has = counts > 0
    if not has.any():
        raise ValueError("no value to average")
    index = np.arange(len(counts))
    filled = np.flatnonzero(has)
    at = np.searchsorted(filled, index)  # the first index with values at or after each index
    lower = filled[np.clip(at - 1, 0, len(filled) - 1)]
    upper = filled[np.clip(at, 0, len(filled) - 1)]
    below = np.where(at > 0, index - lower, np.iinfo(np.int64).max)
    above = np.where(at < len(filled), upper - index, np.iinfo(np.int64).max)
    take_lower, take_upper = below <= above, above <= below
    pooled_sum = np.where(take_lower, sums[lower], 0.0) + np.where(take_upper, sums[upper], 0.0)
    pooled_count = np.where(take_lower, counts[lower], 0) + np.where(take_upper, counts[upper], 0)
    means = np.where(has, sums / np.maximum(counts, 1), pooled_sum / np.maximum(pooled_count, 1))
    return means, ~has


def lookup_baseline(lb_left: Any, wickets: Any, target: Any, fold: Any, train: Any) -> pd.DataFrame:
    """Out-of-fold predictions of the lookup-table baseline (criterion 5; module docstring, "Lookup-table baseline").

    ``lb_left`` and ``wickets`` are each row's pre-ball Legal_Balls left and wickets fallen (whole
    numbers of 0 or more), ``target`` its M2 training target (NaN where none), ``fold`` its fold
    from the Fold_Table and ``train`` true on the rows that may enter a table: the Training_Rows
    of the M2 variant compared, such as ``common.training_mask`` with a finite capped target. A
    row of fold f is predicted from the table built from the ``train`` rows of every other fold,
    whether or not the row is a ``train`` row itself.

    Returns LOOKUP_COLUMNS on a RangeIndex, one row per input row: ``pred`` (float64) and
    ``fill`` (int8: LOOKUP_CELL, LOOKUP_SAME_LB_LEFT or LOOKUP_NEAREST_LB_LEFT). Raises
    ValueError when the arrays differ in length, a value is missing or malformed, a ``train`` row
    has no finite target, or a fold of ``fold`` has no ``train`` row outside it, and TypeError for
    a value of the wrong type.
    """
    lb = _ints(lb_left, "lb_left", minimum=0)
    n_rows = len(lb)
    wk = _ints(wickets, "wickets", n_rows, minimum=0)
    y = _floats(target, "target", n_rows)
    folds = _ints(fold, "fold", n_rows, minimum=1)
    used = _flags(train, "train", n_rows)
    bad = used & ~np.isfinite(y)
    if bad.any():
        raise ValueError(f"target is missing or not finite on {_rows(int(bad.sum()))} where train is true")
    pred = np.full(n_rows, np.nan)
    fill = np.zeros(n_rows, dtype=np.int8)
    if not n_rows:
        return pd.DataFrame({"pred": pred, "fill": fill}, columns=list(LOOKUP_COLUMNS))
    n_lb, n_wk = int(lb.max()) + 1, int(wk.max()) + 1
    cell = lb * n_wk + wk
    for held_out in np.unique(folds).tolist():
        source = used & (folds != held_out)
        if not source.any():
            raise ValueError(f"no train row lies outside fold {held_out}, so its lookup table would be empty")
        values = y[source]
        cell_sum = np.bincount(cell[source], weights=values, minlength=n_lb * n_wk)
        cell_count = np.bincount(cell[source], minlength=n_lb * n_wk)
        lb_mean, borrowed = _nearest_means(
            np.bincount(lb[source], weights=values, minlength=n_lb), np.bincount(lb[source], minlength=n_lb)
        )
        rows = np.flatnonzero(folds == held_out)
        cells, balls_left = cell[rows], lb[rows]
        in_cell = cell_count[cells] > 0
        pred[rows] = np.where(in_cell, cell_sum[cells] / np.maximum(cell_count[cells], 1), lb_mean[balls_left])
        fill[rows] = np.where(
            in_cell, LOOKUP_CELL, np.where(borrowed[balls_left], LOOKUP_NEAREST_LB_LEFT, LOOKUP_SAME_LB_LEFT)
        )
    return pd.DataFrame({"pred": pred, "fill": fill}, columns=list(LOOKUP_COLUMNS))


def rmse_by_innings(y: Any, pred: Any, innings: Any, rows: Any = None) -> pd.DataFrame:
    """RMSE of ``pred`` against ``y`` for each innings, pooled over every given row (criterion 5).

    ``y`` holds the training targets, ``pred`` the predictions (M2's Out_Of_Fold_Predictions or
    the baseline's) and ``innings`` the innings (1 or 2) of each row; ``rows``, when given, is a
    bool per row that picks the rows evaluated (None: every row). The squared errors of all
    folds are pooled, so the result does not depend on how the rows fall into folds.

    Returns RMSE_COLUMNS with one row per innings 1 and 2: ``innings``, ``n_rows`` (the rows
    evaluated) and ``rmse`` (NaN when the innings has none). Raises ValueError when the arrays
    differ in length, an evaluated row lies outside innings 1 and 2 or has a missing or
    non-finite ``y`` or ``pred``, and TypeError for a value of the wrong type.
    """
    target = _floats(y, "y")
    n_rows = len(target)
    predicted = _floats(pred, "pred", n_rows)
    inns = _ints(innings, "innings", n_rows)
    use = np.ones(n_rows, dtype=bool) if rows is None else _flags(rows, "rows", n_rows)
    outside = use & ~np.isin(inns, STATE_INNINGS)
    if outside.any():
        raise ValueError(f"{_rows(int(outside.sum()))} to evaluate lie outside innings 1 and 2")
    bad = use & ~(np.isfinite(target) & np.isfinite(predicted))
    if bad.any():
        raise ValueError(f"y or pred is missing or not finite on {_rows(int(bad.sum()))} to evaluate")
    counts, rmse = [], []
    for number in STATE_INNINGS:
        picked = use & (inns == number)
        count = int(picked.sum())
        counts.append(count)
        rmse.append(float(np.sqrt(np.mean((target[picked] - predicted[picked]) ** 2))) if count else np.nan)
    return pd.DataFrame(
        {
            "innings": np.array(STATE_INNINGS, dtype=np.int64),
            "n_rows": np.array(counts, dtype=np.int64),
            "rmse": np.array(rmse, dtype=np.float64),
        },
        columns=list(RMSE_COLUMNS),
    )


def lookup_baseline_rmse(lb_left: Any, wickets: Any, target: Any, fold: Any, innings: Any, train: Any) -> pd.DataFrame:
    """The baseline's RMSE per innings, on the ``train`` rows of every fold (criterion 5).

    The arguments are those of lookup_baseline plus ``innings``. Each ``train`` row is predicted
    from the table of the other folds and the squared errors are pooled per innings, as
    rmse_by_innings does. Returns BASELINE_RMSE_COLUMNS: the RMSE_COLUMNS, then
    ``n_same_lb_left_fill`` and ``n_nearest_lb_left_fill``, the evaluated rows that used fill
    rules 2 and 3. Raises the errors of lookup_baseline and rmse_by_innings.
    """
    table = lookup_baseline(lb_left, wickets, target, fold, train)
    used = _flags(train, "train", len(table))
    inns = _ints(innings, "innings", len(table))
    out = rmse_by_innings(target, table["pred"].to_numpy(), inns, used)
    fill = table["fill"].to_numpy()
    for column, level in zip(BASELINE_RMSE_COLUMNS[len(RMSE_COLUMNS):], (LOOKUP_SAME_LB_LEFT, LOOKUP_NEAREST_LB_LEFT)):
        out[column] = np.array([int((used & (inns == number) & (fill == level)).sum()) for number in STATE_INNINGS])
    return out[list(BASELINE_RMSE_COLUMNS)]


# ===================================================================== Par_Score (9.6, 9.14; DD-7)

def par_bucket(lb_bowled: Any, sched_balls: Any) -> np.ndarray:
    """The par bucket of each Legal_Ball count: round(PAR_BUCKETS x t / SB), rounded half up, at most PAR_BUCKETS.

    ``lb_bowled`` holds counts t (whole numbers of 0 or more) and ``sched_balls`` the
    Scheduled_Balls SB of each (1 or more). The rounding is exact integer arithmetic,
    floor((2 x 120 x t + SB) / (2 x SB)), so 120 x t / SB = 1.5 gives bucket 2 on every machine.
    Returns an int64 array. Raises ValueError or TypeError for malformed values.
    """
    t = _ints(lb_bowled, "lb_bowled", minimum=0)
    sb = _ints(sched_balls, "sched_balls", len(t), minimum=1)
    return np.minimum((2 * PAR_BUCKETS * t + sb) // (2 * sb), PAR_BUCKETS)


def par_means(X: pd.DataFrame, lb_bowled: Any, sched_balls: Any, rows: Any) -> pd.DataFrame:
    """The par-state means of one first-innings fit, per bucket 0 to PAR_BUCKETS (DD-7; module docstring, "Par_Score").

    ``X`` holds the PAR_MEAN_FEATURES of the fit's candidate rows (m2_features of view V_k gives
    them), ``lb_bowled`` and ``sched_balls`` each row's Legal_Balls bowled and Scheduled_Balls,
    and ``rows`` the positions of the fit's Training_Rows in them (``RegisteredFit.rows.training``
    of a first-innings fit: the rows it was fitted on and the early-stopping rows, all of
    innings 1).

    Returns a DataFrame on the index ``bucket`` (0 to PAR_BUCKETS) with the PAR_MEAN_FEATURES
    (float64) and ``n_rows`` (the Training_Rows in the bucket). Each feature's value is its mean
    over the bucket's Training_Rows without a missing value; a bucket without such a row takes
    the mean of the nearest bucket that has one, pooling both neighbours on a tie. Raises
    ValueError when ``rows`` is empty or out of range, a feature has no value on any Training_Row,
    or an argument is malformed.
    """
    _require(X, PAR_MEAN_FEATURES, "X")
    n_rows = len(X)
    positions = _training_positions(rows, n_rows, "the par means need")
    lb = _ints(lb_bowled, "lb_bowled", n_rows, minimum=0)
    sb = _ints(sched_balls, "sched_balls", n_rows, minimum=1)
    bucket = par_bucket(lb[positions], sb[positions])
    n_buckets = PAR_BUCKETS + 1
    data: dict[str, np.ndarray] = {}
    for name in PAR_MEAN_FEATURES:
        values = _floats(X[name], f"X column {name!r}")[positions]
        defined = np.isfinite(values)
        if not defined.any():
            raise ValueError(f"X column {name!r} has no value on any of the {_rows(len(positions))} given")
        data[name], _ = _nearest_means(
            np.bincount(bucket[defined], weights=values[defined], minlength=n_buckets),
            np.bincount(bucket[defined], minlength=n_buckets),
        )
    data["n_rows"] = np.bincount(bucket, minlength=n_buckets).astype(np.int64)
    return pd.DataFrame(data, index=pd.RangeIndex(n_buckets, name="bucket"), columns=[*PAR_MEAN_FEATURES, "n_rows"])


def _training_positions(rows: Any, n_rows: int, need: str) -> np.ndarray:
    """``rows``, the positions of a fit's first-innings Training_Rows among ``n_rows`` rows, as int64."""
    positions = _ints(rows, "rows", minimum=0)
    if not len(positions):
        raise ValueError(f"rows is empty; {need} the fit's first-innings Training_Rows")
    if int(positions.max()) >= n_rows:
        raise ValueError(f"rows holds a position beyond the {_rows(n_rows)} of X")
    return positions


def _bucket_columns(table: Any, columns: Sequence[str], what: str) -> dict[str, np.ndarray]:
    """``columns`` of the per-bucket table ``what`` (par_means or par_bias) as finite float64 arrays, bucket 0 first."""
    _require(table, columns, what)
    if len(table) != PAR_BUCKETS + 1 or not np.array_equal(np.asarray(table.index), np.arange(PAR_BUCKETS + 1)):
        raise ValueError(f"{what} must hold buckets 0 to {PAR_BUCKETS} in order, as {what} returns them")
    out = {name: _floats(table[name], f"{what} column {name!r}") for name in columns}
    for name, values in out.items():
        if not np.isfinite(values).all():
            raise ValueError(f"{what} column {name!r} holds a missing or non-finite value")
    return out


def _check_par_means(par_means: Any) -> dict[str, np.ndarray]:
    return _bucket_columns(par_means, PAR_MEAN_FEATURES, "par_means")


def _check_par_bias(par_bias: Any) -> np.ndarray:
    return _bucket_columns(par_bias, ("beta",), "par_bias")["beta"]


def par_contexts(scoring_env: Any, venue_effect: Any, sched_balls: Any) -> tuple[pd.DataFrame, np.ndarray]:
    """The distinct first-innings contexts of a set of rows, and each row's context (criterion 14).

    ``scoring_env``, ``venue_effect`` and ``sched_balls`` are each row's Scoring_Environment and
    Venue_Effect (from the view, finite) and Scheduled_Balls (1 or more). Rows share a context
    when the three values are exactly equal.

    Returns (contexts, context): ``contexts`` holds PAR_CONTEXT_COLUMNS (``scoring_env`` and
    ``venue_effect`` float64, ``sched_balls`` int64), one row per distinct context, sorted by
    ``sched_balls``, ``scoring_env`` and ``venue_effect``; ``context`` the int64 position of each
    row's context in it. Raises ValueError for a missing or non-finite value and the errors of a
    malformed argument.
    """
    env = _floats(scoring_env, "scoring_env")
    n_rows = len(env)
    venue = _floats(venue_effect, "venue_effect", n_rows)
    sb = _ints(sched_balls, "sched_balls", n_rows, minimum=1)
    bad = ~(np.isfinite(env) & np.isfinite(venue))
    if bad.any():
        raise ValueError(f"scoring_env or venue_effect is missing or not finite on {_rows(int(bad.sum()))}")
    keys = np.column_stack([sb.astype(np.float64), env, venue])
    if n_rows:
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    else:
        unique, inverse = keys, np.zeros(0, dtype=np.int64)
    contexts = pd.DataFrame(
        {"scoring_env": unique[:, 1], "venue_effect": unique[:, 2], "sched_balls": unique[:, 0].astype(np.int64)},
        columns=list(PAR_CONTEXT_COLUMNS),
    )
    return contexts, np.asarray(inverse, dtype=np.int64).reshape(-1)


def _context_balls(ctx: Any) -> np.ndarray:
    _require(ctx, PAR_CONTEXT_COLUMNS, "ctx")
    return _ints(ctx["sched_balls"], "ctx column 'sched_balls'", minimum=1)


def par_curve(model: Regressor, ctx: pd.DataFrame, par_means: pd.DataFrame) -> np.ndarray:
    """The raw Par_Score by Legal_Balls bowled for each first-innings context, from one first-innings fit (DD-7).

    ``model`` is the first-innings M2 fit, an M2Model (or any Regressor); ``ctx`` holds one row
    per context with PAR_CONTEXT_COLUMNS (par_contexts gives it); ``par_means`` is that fit's
    par_means.

    For each context with Scheduled_Balls SB, the par states t = 0 ... SB have Legal_Balls left
    SB - t, the par_means of bucket ``par_bucket(t, SB)``, Match_Day_Conditions 0 and the
    context's Scoring_Environment and Venue_Effect; R(t) is the model's prediction, for an
    M2Model the calibrated one (criterion 15). All par
    states of all contexts are predicted in one call. The raw Par_Score is P_raw(t) = R(0) - min
    over u <= t of R(u); corrected_par_curve turns it into Par_Score with the fit's par_bias.

    Returns a float64 array of shape (len(ctx), max(PAR_BUCKETS, largest SB) + 1): row c holds
    P_raw(0) ... P_raw(SB) of context c, then NaN. P_raw(0) is 0 and every curve is
    non-decreasing up to its SB, whatever the model predicts. Raises ValueError for a malformed
    ``ctx`` or ``par_means`` and the errors of predict.
    """
    sb = _context_balls(ctx)
    n_ctx = len(sb)
    env = _floats(ctx["scoring_env"], "ctx column 'scoring_env'", n_ctx)
    venue = _floats(ctx["venue_effect"], "ctx column 'venue_effect'", n_ctx)
    means = _check_par_means(par_means)
    width = max(PAR_BUCKETS, int(sb.max()) if n_ctx else 0) + 1
    if not n_ctx:
        return np.full((0, width), np.nan)
    counts = sb + 1  # one par state per count t = 0 ... SB
    owner = np.repeat(np.arange(n_ctx), counts)
    t = np.arange(len(owner)) - np.repeat(np.cumsum(counts) - counts, counts)
    balls = sb[owner]
    bucket = par_bucket(t, balls)
    columns: dict[str, Any] = {
        "lb_left": (balls - t).astype(np.float64),
        "scoring_env": env[owner],
        "venue_effect": venue[owner],
        "match_day": np.full(len(owner), _PAR_MATCH_DAY),
    }
    columns.update({name: values[bucket] for name, values in means.items()})
    remaining = np.full((n_ctx, width), np.nan)
    remaining[owner, t] = predict(model, pd.DataFrame(columns, columns=list(M2_FEATURES)))
    curves = remaining[:, :1] - np.minimum.accumulate(remaining, axis=1)  # NaN after SB stays NaN
    curves[:, 0] = 0.0
    return curves


def _check_curves(curves: Any, sb: np.ndarray) -> np.ndarray:
    array = np.asarray(curves, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != len(sb) or (len(sb) and array.shape[1] <= int(sb.max())):
        raise ValueError(
            f"curves must have one row per context and a column for every count up to the largest Scheduled_Balls, "
            f"as par_curve returns them; got shape {array.shape} for {len(sb):,} context(s)"
        )
    return array


def par_lookup(curves: Any, ctx: pd.DataFrame, context: Any, t: Any) -> np.ndarray:
    """The curve value at min(t, SB) of each row, from the ``curves`` of the contexts ``ctx`` (criterion 6).

    ``curves`` are par curves of ``ctx``: corrected_par_curve's for Par_Score, par_curve's for the
    raw Par_Score. ``context`` gives each row's position in ``ctx`` (par_contexts) and ``t`` its
    Legal_Ball count (0 or more); a count beyond the context's Scheduled_Balls SB takes the value
    at SB. Returns a float64 array. Raises ValueError for a context position out of range, a
    negative count, curves of the wrong shape or malformed arguments.
    """
    sb = _context_balls(ctx)
    array = _check_curves(curves, sb)
    ids = _ints(context, "context", minimum=0)
    if len(ids) and int(ids.max()) >= len(sb):
        raise ValueError(f"context holds a position beyond the {len(sb):,} context(s) of ctx")
    counts = _ints(t, "t", len(ids), minimum=0)
    return array[ids, np.minimum(counts, sb[ids])]


def par_bias(
    model: Regressor,
    par_means: pd.DataFrame,
    X: pd.DataFrame,
    lb_bowled: Any,
    sched_balls: Any,
    score: Any,
    rows: Any,
) -> pd.DataFrame:
    """The Par_Score bias of one first-innings fit, per bucket 0 to PAR_BUCKETS (module docstring, "Par_Score").

    ``model`` is the first-innings M2 fit and ``par_means`` its par_means. ``X`` holds the
    ``scoring_env`` and ``venue_effect`` of the fit's candidate rows (m2_features of view V_k),
    ``lb_bowled``, ``sched_balls`` and ``score`` each row's Legal_Balls bowled, Scheduled_Balls
    and pre-ball team score (the ``score`` column of ``state``), and ``rows`` the positions of
    the fit's first-innings Training_Rows in them, as for par_means
    (``RegisteredFit.rows.training``).

    The residual of Training_Row i is its score minus the raw Par_Score (par_curve) of its own
    context at its own Legal_Balls bowled L_i (par_lookup). The bias beta of a bucket is the mean
    residual over the Training_Rows in it, each row in bucket ``par_bucket(L_i, SB_i)``; a bucket
    without a Training_Row takes the mean of the nearest bucket that has one, pooling both
    neighbours on a tie. No row outside ``rows`` is read, so the bias uses no Delivery of the
    folds the fit excludes.

    Returns PAR_BIAS_COLUMNS on the index ``bucket`` (0 to PAR_BUCKETS): ``beta`` (float64, in
    runs) and ``n_rows`` (int64, the Training_Rows in the bucket; 0 where ``beta`` comes from a
    neighbour). Raises ValueError when ``rows`` is empty or out of range or an argument is
    malformed, and the errors of par_contexts and par_curve.
    """
    _require(X, ("scoring_env", "venue_effect"), "X")
    n_rows = len(X)
    positions = _training_positions(rows, n_rows, "the par bias needs")
    lb = _ints(lb_bowled, "lb_bowled", n_rows, minimum=0)[positions]
    sb = _ints(sched_balls, "sched_balls", n_rows, minimum=1)[positions]
    runs = _ints(score, "score", n_rows, minimum=0)[positions]
    env = _floats(X["scoring_env"], "X column 'scoring_env'", n_rows)[positions]
    venue = _floats(X["venue_effect"], "X column 'venue_effect'", n_rows)[positions]
    contexts, context = par_contexts(env, venue, sb)
    raw = par_curve(model, contexts, par_means)
    residual = runs - par_lookup(raw, contexts, context, lb)
    bucket = par_bucket(lb, sb)
    n_buckets = PAR_BUCKETS + 1
    counts = np.bincount(bucket, minlength=n_buckets)
    beta, _ = _nearest_means(np.bincount(bucket, weights=residual, minlength=n_buckets), counts)
    return pd.DataFrame(
        {"beta": beta, "n_rows": counts.astype(np.int64)},
        index=pd.RangeIndex(n_buckets, name="bucket"),
        columns=list(PAR_BIAS_COLUMNS),
    )


def corrected_par_curve(curves: Any, ctx: pd.DataFrame, par_bias: pd.DataFrame) -> np.ndarray:
    """Par_Score by Legal_Balls bowled: the raw par_curve ``curves`` of ``ctx`` corrected by a fit's bias (criterion 6).

    ``par_bias`` is the par_bias table of the fit whose par_curve gave ``curves``. For each
    context with Scheduled_Balls SB, Par_Score is P(0) = 0 and P(t) = max(P(t - 1), P_raw(t) +
    beta(``par_bucket(t, SB)``)) for t = 1 ... SB.

    Returns a new float64 array in the layout of ``curves``: row c holds P(0) ... P(SB) of
    context c, then NaN. P(0) is 0 and every curve is non-decreasing up to its SB, so never below
    0, whatever the raw curve and the bias. Raises ValueError for curves of the wrong shape or
    with a missing or non-finite value up to a context's SB, and for a malformed ``ctx`` or
    ``par_bias``.
    """
    sb = _context_balls(ctx)
    raw = _check_curves(curves, sb)
    beta = _check_par_bias(par_bias)
    owner, t = np.nonzero(np.arange(raw.shape[1]) <= sb[:, None])  # every count t = 0 ... SB of every context
    values = raw[owner, t]
    if not np.isfinite(values).all():
        raise ValueError(
            "curves hold a missing or non-finite value at a count up to the context's Scheduled_Balls; pass the "
            "curves of par_curve"
        )
    shifted = np.full(raw.shape, np.nan)
    shifted[owner, t] = values + beta[par_bucket(t, sb[owner])]
    shifted[:, :1] = 0.0
    return np.maximum.accumulate(shifted, axis=1)  # NaN after SB stays NaN


def par_rate(curves: Any, ctx: pd.DataFrame, context: Any, lb_bowled: Any, win_lb: Any) -> np.ndarray:
    """Par_Rate over each first-innings row's recent window, in runs per over (glossary; Requirement 12, criterion 6).

    ``curves``, ``ctx`` and ``context`` are as for par_lookup; ``lb_bowled`` is each row's
    Legal_Balls bowled L and ``win_lb`` the Legal_Balls w in the window of its ``win_rr``
    (``state``, or ``state.recent_window`` for another window length), 0 to L. Returns
    6 x (Par_Score(L) - Par_Score(L - w)) / w as float64, NaN where w = 0 (module docstring,
    "Par_Rate"). Raises ValueError when ``win_lb`` exceeds ``lb_bowled`` and the errors of
    par_lookup.
    """
    lb = _ints(lb_bowled, "lb_bowled", minimum=0)
    window = _ints(win_lb, "win_lb", len(lb), minimum=0)
    over = window > lb
    if over.any():
        raise ValueError(
            f"win_lb exceeds lb_bowled on {_rows(int(over.sum()))}; the window holds only Legal_Balls bowled before "
            "the Delivery"
        )
    end = par_lookup(curves, ctx, context, lb)
    start = par_lookup(curves, ctx, context, lb - window)
    rate = np.full(len(lb), np.nan)
    has = window > 0
    rate[has] = BALLS_PER_OVER * (end[has] - start[has]) / window[has]
    return rate


def first_innings_par(
    model: Regressor,
    par_means: pd.DataFrame,
    par_bias: pd.DataFrame,
    X: pd.DataFrame,
    lb_bowled: Any,
    sched_balls: Any,
    win_lb: Any,
) -> pd.DataFrame:
    """Par_Score, Par_Score at L + 1, Par_Rate and raw Par_Score of a first-innings fit's rows (criteria 6 and 14).

    ``model`` is the first-innings M2 fit, ``par_means`` its par_means and ``par_bias`` its
    par_bias. ``X`` holds the rows' ``scoring_env`` and ``venue_effect`` (m2_features of the
    view, restricted to the rows), ``lb_bowled`` their Legal_Balls bowled L, ``sched_balls``
    their Scheduled_Balls and ``win_lb`` the Legal_Balls in their recent window. The rows must be
    first-innings Deliveries; Par_Score is defined for those only.

    Evaluates one raw par curve per distinct context of the rows (par_contexts, par_curve),
    corrects it with the bias (corrected_par_curve) and returns, on a RangeIndex, one row per row
    of ``X``: the PAR_COLUMNS from the corrected curve, ``par_score`` = Par_Score(L),
    ``par_score_next`` = Par_Score(L + 1) and ``par_rate_win`` (par_rate, NaN at L = 0); then
    PAR_RAW_COLUMN, ``par_score_raw`` = the raw Par_Score at L, for the report. All are float64.
    Raises the errors of the functions it calls.
    """
    _require(X, ("scoring_env", "venue_effect"), "X")
    _check_par_bias(par_bias)
    contexts, context = par_contexts(X["scoring_env"], X["venue_effect"], sched_balls)
    raw = par_curve(model, contexts, par_means)
    curves = corrected_par_curve(raw, contexts, par_bias)
    lb = _ints(lb_bowled, "lb_bowled", len(X), minimum=0)
    return pd.DataFrame(
        {
            "par_score": par_lookup(curves, contexts, context, lb),
            "par_score_next": par_lookup(curves, contexts, context, lb + 1),
            "par_rate_win": par_rate(curves, contexts, context, lb, win_lb),
            PAR_RAW_COLUMN: par_lookup(raw, contexts, context, lb),
        },
        columns=[*PAR_COLUMNS, PAR_RAW_COLUMN],
    )


# ===================================================================== Achievable_Rate (9.7)

def achievable_rate(model_first: Regressor, X2: pd.DataFrame) -> np.ndarray:
    """Achievable_Rate of each row, in runs per over (criterion 7; glossary).

    ``model_first`` is the first-innings M2 fit (uncapped), an M2Model whose calibrated
    prediction is used (criterion 15), and ``X2`` the M2_FEATURES of innings-2 rows
    (m2_features, or the features of a Post_Outcome_State). Returns 6 x predict(model_first, X2)
    / ``lb_left`` as float64, NaN where ``lb_left`` is 0 or missing, so never infinite. Raises
    the errors of predict.
    """
    expected = predict(model_first, X2)
    balls_left = _floats(X2["lb_left"], "X2 column 'lb_left'")
    rate = np.full(len(expected), np.nan)
    has = np.isfinite(balls_left) & (balls_left > 0)
    rate[has] = BALLS_PER_OVER * expected[has] / balls_left[has]
    return rate


# ===================================================================== the remaining_runs Pipeline_Stage

#: Name of the Pipeline_Stage that fits M2 and computes Wicket_Cost, and of its checkpoint.
STAGE_NAME = "remaining_runs"

#: The M2-derived quantities that the stage adds to each sandbox view V_k (Data Models, view_k):
#: PAR_COLUMNS on innings-1 rows, Achievable_Rate on innings-2 rows and the F19 Pillar_Component,
#: all from the first-innings fits, whose ids the provenance column ``views.M2_FIT`` records.
ACH_RATE_COLUMN = "ach_rate"
F19_COLUMN = "f19_component"
VIEW_M2_COLUMNS: tuple[str, ...] = (*PAR_COLUMNS, ACH_RATE_COLUMN, F19_COLUMN)

#: Checkpoint table with the Wicket_Cost of every innings 1-2 Delivery, in ``state`` order (Data
#: Models, m2_preds), with the raw Par_Score and the ids of the two outer fits that gave the values.
PREDS_TABLE = "m2_preds"
PREDS_COLUMNS: tuple[str, ...] = (
    "row", "r_pre", "r_post_wicket", "omega", "omega_clipped_low", "omega_clipped_high", PAR_RAW_COLUMN,
    "capped_fit", "first_fit",
)  # fmt: skip

#: Checkpoint tables: the par-state means and the Par_Score bias of every first-innings fit, by
#: bucket, and one summary row per M2 fit.
PAR_MEANS_TABLE = "m2_par_means"
PAR_MEANS_TABLE_COLUMNS: tuple[str, ...] = ("fit_id", "bucket", *PAR_MEAN_FEATURES, "n_rows")
PAR_BIAS_TABLE = "m2_par_bias"
PAR_BIAS_TABLE_COLUMNS: tuple[str, ...] = ("fit_id", "bucket", *PAR_BIAS_COLUMNS)
FIT_SUMMARY_TABLE = "m2_fits"
FIT_SUMMARY_COLUMNS: tuple[str, ...] = (
    "fit_id", "model", "sandbox_k", "inner_j", "n_train_rows", "n_fit_rows", "n_early_stopping_rows",
    "n_train_matches", "n_early_stopping_matches", "n_predicted_matches", "best_iteration",
)  # fmt: skip

#: Checkpoint table with the Legal_Balls-left calibration of every M2 fit (criterion 15): the
#: LB_CALIBRATION_COLUMNS of each fit, after its fit id. load_models rebuilds the M2Models from it.
LB_CALIBRATION_TABLE = "m2_lb_calibration"
LB_CALIBRATION_TABLE_COLUMNS: tuple[str, ...] = ("fit_id", *LB_CALIBRATION_COLUMNS)

#: Folder under the outputs folder for the stage's files (Requirement 9, criteria 5, 9, 10 and 12),
#: their names, and the columns of its two tables.
M2_FOLDER = Path("models") / "m2"
METRICS_FILE = "metrics.csv"
LB_CALIBRATION_FILE = "lb_calibration.csv"
PAR_BIAS_FILE = "par_bias.csv"
OMEGA_TABLE_FILE = "omega_by_lb_left_wickets.csv"
OMEGA_CHART_FILE = "omega_by_lb_left_wickets.png"
METRICS_COLUMNS: tuple[str, ...] = ("metric", "variant", "innings", "value", "n_rows", "description")
OMEGA_TABLE_COLUMNS: tuple[str, ...] = ("innings", "lb_left", "wickets", "n_deliveries", "mean_omega")

#: Resolution of the Omega chart in dots per inch, fixed so that the same data give the same image.
OMEGA_CHART_DPI = 150

#: The Legal_Balls-left bands of the calibration metrics in METRICS_FILE, as inclusive (low, high)
#: pairs: the last Legal_Ball, the rest of the last over, the rest of the last 3 overs, overs 11-17
#: and the first 10 overs. They only group a report table; no value of the Run depends on them.
LB_BANDS: tuple[tuple[int, int], ...] = ((1, 1), (2, 6), (7, 18), (19, 60), (61, 120))

#: Data-derived Method_Constant: fit id -> the boosting rounds that early stopping kept, for each
#: outer fit (the fits behind the Out_Of_Fold_Predictions, Omega and the fold-k rows of the views).
BEST_ITERATION_CONSTANT = "remaining_runs.outer_best_iteration_by_fit"

_COMPONENT = "Remaining_Runs_Model"  # the glossary component that produces the constant
_VARIANTS: tuple[tuple[str, str], ...] = ((CAPPED_MODEL, "capped"), (FIRST_INNINGS_MODEL, "first_innings"))
_STAGE_BALL_COLUMNS: tuple[str, ...] = ("row", "match_id", "innings", "sched_balls", "is_super_over", "excluded")
_STAGE_STATE_COLUMNS: tuple[str, ...] = (
    "row", "lb_bowled", "lb_left", "score", "wickets", "runs_needed", "rrr", "win_lb", "btc_count", "phase",
)  # fmt: skip
_FIRST_INNINGS = 1
_CHASE = 2
_FLOAT_FORMAT = "%.6f"
_CHART_SIZE = (11.0, 4.4)  # inches
_CHART_COLOURS = "viridis"
_CHART_EMPTY_CELL = "#eeeeee"  # a Legal_Balls left x wickets fallen cell without a Delivery


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


def _delivery_rows(balls: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
    """The _STAGE_BALL_COLUMNS of ``balls`` for each ``state`` row, in ``state`` order, on a RangeIndex."""
    _require(balls, _STAGE_BALL_COLUMNS, "balls")
    _require(state, _STAGE_STATE_COLUMNS, "state")
    index = pd.Index(_ints(balls["row"], "balls column 'row'"))
    if not index.is_unique:
        raise ValueError("balls must hold each row once")
    at = index.get_indexer(_ints(state["row"], "state column 'row'"))
    if (at < 0).any():
        raise ValueError(
            f"{_rows(int((at < 0).sum()))} of state have no Delivery in balls; pass the tables of one Run"
        )
    rows = balls.iloc[at][list(_STAGE_BALL_COLUMNS)].reset_index(drop=True)
    outside = ~np.isin(_ints(rows["innings"], "balls column 'innings'"), STATE_INNINGS)
    if outside.any():
        raise ValueError(f"{_rows(int(outside.sum()))} of state lie outside innings 1 and 2")
    return rows


def _check_order(rows: Any, expected: np.ndarray, what: str) -> None:
    got = _ints(rows, f"{what}: column 'row'")
    if len(got) != len(expected) or not np.array_equal(got, expected):
        raise ValueError(f"{what} do not hold the Deliveries of the state table in its order; pass the tables of one Run")


def _served_values(
    model: Regressor,
    means: pd.DataFrame,
    bias: pd.DataFrame,
    X: pd.DataFrame,
    served: np.ndarray,
    innings: np.ndarray,
    lb_bowled: np.ndarray,
    sched: np.ndarray,
    win_lb: np.ndarray,
    score: np.ndarray,
    rrr: np.ndarray,
) -> dict[str, np.ndarray]:
    """VIEW_M2_COLUMNS and PAR_RAW_COLUMN of the rows ``served`` (ascending positions) from one first-innings fit.

    Innings-1 rows get the PAR_COLUMNS, the raw Par_Score and F19 = Par_Score - score; innings-2
    rows get Achievable_Rate and F19 = Required_Rate - Achievable_Rate. Every other value is NaN.
    Raises ValueError for an infinite value, which a view quantity never holds.
    """
    out = {name: np.full(len(served), np.nan) for name in (*VIEW_M2_COLUMNS, PAR_RAW_COLUMN)}
    inns = innings[served]
    one, two = np.flatnonzero(inns == _FIRST_INNINGS), np.flatnonzero(inns == _CHASE)
    if len(one):
        rows = served[one]
        par = first_innings_par(model, means, bias, X.iloc[rows], lb_bowled[rows], sched[rows], win_lb[rows])
        for name in (*PAR_COLUMNS, PAR_RAW_COLUMN):
            out[name][one] = par[name].to_numpy(dtype=np.float64)
        out[F19_COLUMN][one] = out[PAR_COLUMNS[0]][one] - score[rows]
    if len(two):
        rows = served[two]
        rate = achievable_rate(model, X.iloc[rows])
        out[ACH_RATE_COLUMN][two] = rate
        out[F19_COLUMN][two] = rrr[rows] - rate  # NaN where either rate is undefined
    for name, values in out.items():
        infinite = int(np.isinf(values).sum())
        if infinite:
            raise ValueError(f"{name} holds {infinite:,} infinite value(s); a view quantity is a number or missing")
    return out


def _best_iteration(model: Any) -> int:
    best = getattr(model, "best_iteration", None)
    return int(best) if best is not None else -1


def _fit_summary(fitted: common.RegisteredFit, model: str, registry: leakage.FitRegistry) -> dict[str, Any]:
    record = registry.get(fitted.fit_id)
    return {
        "fit_id": fitted.fit_id,
        "model": model,
        "sandbox_k": record.sandbox_k,
        "inner_j": record.inner_j,
        "n_train_rows": len(fitted.rows.training),
        "n_fit_rows": len(fitted.rows.train),
        "n_early_stopping_rows": len(fitted.rows.early_stopping),
        "n_train_matches": len(record.train_match_ids),
        "n_early_stopping_matches": len(record.early_stopping_match_ids),
        "n_predicted_matches": len(record.predicted_match_ids),
        "best_iteration": _best_iteration(fitted.model),
    }


def _bucket_table(tables: Mapping[str, pd.DataFrame], columns: Sequence[str]) -> pd.DataFrame:
    """The per-bucket tables of several fits as one table: ``fit_id`` (str), ``bucket``, then ``columns``."""
    frame = pd.concat(dict(tables), names=["fit_id"]).reset_index()
    frame["fit_id"] = frame["fit_id"].astype("str")
    return frame[["fit_id", "bucket", *columns]]


def _calibration_table(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """The calibration tables of several fits as LB_CALIBRATION_TABLE (LB_CALIBRATION_TABLE_COLUMNS), in fit order."""
    frame = pd.concat(
        [table[list(LB_CALIBRATION_COLUMNS)].assign(fit_id=fit_id) for fit_id, table in tables.items()],
        ignore_index=True,
    )
    frame = frame.astype({"fit_id": "str", "lb_left": np.int64, "ratio": np.float64, "factor": np.float64})
    return frame.astype({"n_rows": np.int64})[list(LB_CALIBRATION_TABLE_COLUMNS)]


_METRIC_TEXT: Mapping[str, str] = MappingProxyType(
    {
        "rmse_m2": (
            "pooled held-out RMSE over the Training_Rows of all folds of the Out_Of_Fold_Predictions of the outer "
            "{label} M2 fits against {target} (Requirement 9, criterion 5)"
        ),
        "rmse_m2_raw": (
            "pooled held-out RMSE over the same rows of the same outer {label} M2 fits' raw predictions, the "
            "Booster's before the Legal_Balls-left calibration (Requirement 9, criterion 15)"
        ),
        "rmse_lookup_baseline": (
            "pooled held-out RMSE over the same rows of the lookup baseline: the mean of {target} by Legal_Balls "
            "left x wickets fallen over the other folds' Training_Rows, an empty cell taking the mean of the same "
            "Legal_Balls left (Requirement 9, criterion 5)"
        ),
        "mean_target_lb_left": (
            "mean of {target} over the same rows with {band} Legal_Balls left before the ball"
        ),
        "mean_pred_raw_lb_left": (
            "mean held-out raw prediction of the outer {label} M2 fits (the Booster's, before the Legal_Balls-left "
            "calibration) over the same rows with {band} Legal_Balls left before the ball"
        ),
        "mean_pred_lb_left": (
            "mean Out_Of_Fold_Prediction of the outer {label} M2 fits, after the Legal_Balls-left calibration "
            "(Requirement 9, criterion 15), over the same rows with {band} Legal_Balls left before the ball"
        ),
        "n_same_lb_left_fill": (
            "evaluated rows whose baseline cell had no Training_Row in the other folds, so the baseline took the "
            "mean of the same Legal_Balls left"
        ),
        "n_nearest_lb_left_fill": (
            "evaluated rows whose Legal_Balls left had no Training_Row in the other folds either, so the baseline "
            "took the mean of the nearest Legal_Balls left with rows"
        ),
        "omega_mean": (
            "mean Wicket_Cost Omega of the innings' Deliveries, in runs, each from the outer capped M2 fit of its "
            "fold (Requirement 9, criterion 8)"
        ),
        "omega_clipped_low": (
            "Deliveries whose r_pre minus r_post was negative, so Omega was set to 0 (Requirement 9, criterion 10)"
        ),
        "omega_clipped_high": (
            "innings-2 Deliveries whose Omega exceeded the runs needed before the ball and was set to them, or to "
            "0 where the runs needed were 0 or less (Requirement 9, criterion 9)"
        ),
        "needed_nonpositive": (
            "innings-2 Deliveries with runs needed of 0 or less before the ball (the chase was already won); kept "
            "in the data, with a capped target of 0 and an Omega of 0"
        ),
        "post_wicket_runs_zero": (
            "Deliveries with 9 wickets fallen or an empty Batters_To_Come, whose post-wicket expected remaining "
            "runs were taken as 0 (Requirement 9, criterion 13)"
        ),
    }
)
_TARGET_TEXT: Mapping[str, str] = MappingProxyType(
    {
        CAPPED_MODEL: "the capped training target (innings 2: the remaining runs capped at the runs needed)",
        FIRST_INNINGS_MODEL: "the remaining runs of the first-innings Training_Rows",
    }
)


_BAND_COLUMNS: tuple[str, ...] = ("innings", "low", "high", "n_rows", "mean_target", "mean_pred_raw", "mean_pred")
_BAND_METRICS: tuple[tuple[str, str], ...] = (
    ("mean_target_lb_left", "mean_target"),
    ("mean_pred_raw_lb_left", "mean_pred_raw"),
    ("mean_pred_lb_left", "mean_pred"),
)


def _band_means(
    y: np.ndarray, raw: np.ndarray, pred: np.ndarray, innings: np.ndarray, lb_left: np.ndarray, rows: np.ndarray
) -> pd.DataFrame:
    """_BAND_COLUMNS per innings and LB_BANDS band: the ``rows`` there and their mean target, raw and calibrated prediction."""
    records = []
    for number in STATE_INNINGS:
        for low, high in LB_BANDS:
            mask = rows & (innings == number) & (lb_left >= low) & (lb_left <= high)
            count = int(mask.sum())
            means = [float(values[mask].mean()) if count else np.nan for values in (y, raw, pred)]
            records.append(dict(zip(_BAND_COLUMNS, (number, low, high, count, *means))))
    return pd.DataFrame.from_records(records, columns=list(_BAND_COLUMNS))


def _band_label(low: int, high: int, separator: str) -> str:
    return str(low) if low == high else f"{low}{separator}{high}"


def _metrics_table(
    accuracy: Mapping[str, tuple[pd.DataFrame, pd.DataFrame]],
    raw_rmse: Mapping[str, pd.DataFrame],
    bands: Mapping[str, pd.DataFrame],
    omega: np.ndarray,
    clipped_low: np.ndarray,
    clipped_high: np.ndarray,
    innings: np.ndarray,
    terminal: np.ndarray,
    needed_nonpositive: np.ndarray,
) -> pd.DataFrame:
    """The METRICS_FILE table (module docstring, "The remaining_runs Pipeline_Stage")."""
    records: list[dict[str, Any]] = []

    def add(
        metric: str, variant: str, number: int, value: str, n_rows: int, text: str | None = None, **fields: str
    ) -> None:
        records.append(
            {
                "metric": metric,
                "variant": variant,
                "innings": number,
                "value": value,
                "n_rows": n_rows,
                "description": _METRIC_TEXT[text or metric].format(**fields),
            }
        )

    for model, variant in _VARIANTS:
        m2, baseline = accuracy[model]
        raw = raw_rmse[model]
        band = bands[model]
        target = _TARGET_TEXT[model]
        label = variant.replace("_", "-")
        for ours, unscaled, base in zip(m2.to_dict("records"), raw.to_dict("records"), baseline.to_dict("records")):
            if not ours["n_rows"]:
                continue  # the first-innings variant has no innings-2 Training_Row
            number, n_rows = int(ours["innings"]), int(ours["n_rows"])
            add("rmse_m2", variant, number, f"{ours['rmse']:.6f}", n_rows, label=label, target=target)
            add("rmse_m2_raw", variant, number, f"{unscaled['rmse']:.6f}", int(unscaled["n_rows"]), label=label)
            add("rmse_lookup_baseline", variant, number, f"{base['rmse']:.6f}", int(base["n_rows"]), target=target)
            for name in BASELINE_RMSE_COLUMNS[len(RMSE_COLUMNS):]:
                add(name, variant, number, str(int(base[name])), int(base["n_rows"]))
            for row in band[band["innings"].to_numpy() == number].to_dict("records"):
                low, high, count = int(row["low"]), int(row["high"]), int(row["n_rows"])
                for text, column in _BAND_METRICS:
                    add(
                        f"{text}_{_band_label(low, high, '_')}", variant, number,
                        f"{row[column]:.6f}" if count else "", count, text=text,
                        label=label, target=target, band=_band_label(low, high, "-"),
                    )  # fmt: skip
    for number in STATE_INNINGS:
        mask = innings == number
        n_rows = int(mask.sum())
        add("omega_mean", "capped", number, f"{float(omega[mask].mean()):.6f}" if n_rows else "", n_rows)
        add("omega_clipped_low", "capped", number, str(int((clipped_low & mask).sum())), n_rows)
        if number == _CHASE:
            add("omega_clipped_high", "capped", number, str(int((clipped_high & mask).sum())), n_rows)
            add("needed_nonpositive", "capped", number, str(int((needed_nonpositive & mask).sum())), n_rows)
        add("post_wicket_runs_zero", "capped", number, str(int((terminal & mask).sum())), n_rows)
    table = pd.DataFrame.from_records(records, columns=list(METRICS_COLUMNS))
    return table.astype({"metric": "str", "variant": "str", "innings": np.int64, "value": "str", "n_rows": np.int64})


def _omega_table(innings: np.ndarray, lb_left: np.ndarray, wickets: np.ndarray, omega: np.ndarray) -> pd.DataFrame:
    """OMEGA_TABLE_COLUMNS: the Deliveries and mean Omega of every non-empty innings x Legal_Balls left x wickets cell."""
    frame = pd.DataFrame({"innings": innings, "lb_left": lb_left, "wickets": wickets, "omega": omega})
    grouped = frame.groupby(["innings", "lb_left", "wickets"], sort=True)["omega"]
    table = grouped.agg(n_deliveries="size", mean_omega="mean").reset_index()
    table = table.astype({"innings": np.int64, "lb_left": np.int64, "wickets": np.int64, "n_deliveries": np.int64})
    return table[list(OMEGA_TABLE_COLUMNS)]


def _write_omega_chart(path: Path, table: pd.DataFrame) -> None:
    """The mean-Omega heat map of OMEGA_TABLE_COLUMNS ``table``, one panel per innings, as a PNG (criterion 12)."""
    from matplotlib import colormaps
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    lb = table["lb_left"].to_numpy(dtype=np.int64)
    wk = table["wickets"].to_numpy(dtype=np.int64)
    mean = table["mean_omega"].to_numpy(dtype=np.float64)
    lb_low = min(1, int(lb.min())) if len(lb) else 1
    lb_high = max(PAR_BUCKETS, int(lb.max()) if len(lb) else 0)
    n_wk = max(TERMINAL_WICKETS, int(wk.max()) if len(wk) else 0) + 1
    x_edges = np.arange(lb_low, lb_high + 2) - 0.5
    y_edges = np.arange(n_wk + 1) - 0.5
    top = float(np.nanmax(mean)) if np.isfinite(mean).any() else 1.0
    colours = colormaps[_CHART_COLOURS].with_extremes(bad=_CHART_EMPTY_CELL)

    figure = Figure(figsize=_CHART_SIZE, dpi=OMEGA_CHART_DPI, layout="constrained")
    FigureCanvasAgg(figure)
    axes = list(np.atleast_1d(figure.subplots(1, len(STATE_INNINGS), sharey=True)))
    mesh = None
    inns = table["innings"].to_numpy(dtype=np.int64)
    for ax, number in zip(axes, STATE_INNINGS):
        grid = np.full((n_wk, lb_high - lb_low + 1), np.nan)
        part = inns == number
        grid[wk[part], lb[part] - lb_low] = mean[part]
        mesh = ax.pcolormesh(
            x_edges, y_edges, np.ma.masked_invalid(grid), cmap=colours, vmin=0.0, vmax=top, shading="flat"
        )
        ax.set_xlim(lb_high + 0.5, lb_low - 0.5)  # the innings runs from left to right
        ax.set_yticks(range(n_wk))
        ax.set_xlabel("Legal_Balls left")
        ax.set_title(f"Innings {number}" + (" (Ω capped at the runs needed)" if number == _CHASE else ""))
    axes[0].set_ylabel("Wickets fallen before the ball")
    figure.colorbar(mesh, ax=axes, label="Mean Ω (runs)")
    figure.suptitle("Mean Wicket_Cost Ω by Legal_Balls left and wickets fallen (outer capped M2 fits)")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        figure.savefig(temporary, format="png", dpi=OMEGA_CHART_DPI, metadata={"Software": None})
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_stage(ctx: StageContext) -> None:
    """The ``remaining_runs`` Pipeline_Stage (module docstring, "The remaining_runs Pipeline_Stage").

    Raises FitLeakageError or ViewLeakageError (exit code 4) when a fit or a view breaks the
    leakage guard (Requirement 7, criterion 12), DataError (exit code 3) for a match that is not in
    the Fold_Table, and ValueError or TypeError for input tables that do not fit together or a
    malformed Config value.
    """
    started = time.perf_counter()
    settings, store, registry = ctx.settings, ctx.store, ctx.registry
    n_folds = settings.get("design_choices.D15_n_folds")
    plan = folds.sandbox_plan(n_folds)  # checks D15 before any table is read
    setup = m2_setup(settings)
    first_plan = views.view_plan(FIRST_INNINGS_MODEL, plan, family=FAMILY)
    encoding = ctx.constants.value(reconstruct.SELECTED_ENCODING_CONSTANT)
    all_folds = tuple(range(1, n_folds + 1))

    balls = store.load_table(reconstruct.STAGE_NAME, reconstruct.BALLS_TABLE)
    innings_info = store.load_table(reconstruct.STAGE_NAME, reconstruct.INNINGS_INFO_TABLE)
    state = store.load_table(STATE_STAGE, STATE_TABLE)
    fold_table = store.load_table(folds.STAGE_NAME, folds.FOLD_TABLE)

    # 1. The Deliveries (the state rows, in state order), their targets, folds and Training_Row flags.
    rows = _delivery_rows(balls, state)
    row_ids = _ints(state["row"], "state column 'row'")
    targets = {
        CAPPED_MODEL: m2_targets(balls, innings_info, capped=True, encoding=encoding),
        FIRST_INNINGS_MODEL: m2_targets(balls, innings_info, capped=False, encoding=encoding),
    }
    del balls, innings_info
    for model, table in targets.items():
        _check_order(table["row"], row_ids, f"the {model} targets")
    match_ids = _ints(rows["match_id"], "balls column 'match_id'")
    innings = _ints(rows["innings"], "balls column 'innings'")
    sched = _ints(rows["sched_balls"], "balls column 'sched_balls'", minimum=1)
    row_fold = folds.fold_of(match_ids, fold_table).astype(np.int64)
    trainable = common.training_mask(rows)
    y = {model: table["target"].to_numpy(dtype=np.float64) for model, table in targets.items()}
    eligible = {
        CAPPED_MODEL: trainable & np.isfinite(y[CAPPED_MODEL]),
        FIRST_INNINGS_MODEL: trainable & (innings == _FIRST_INNINGS) & np.isfinite(y[FIRST_INNINGS_MODEL]),
    }
    needed_nonpositive = targets[CAPPED_MODEL]["needed_nonpositive"].to_numpy(dtype=bool)
    lb_bowled = _ints(state["lb_bowled"], "state column 'lb_bowled'", minimum=0)
    lb_left = _ints(state["lb_left"], "state column 'lb_left'", minimum=0)
    wickets = _ints(state["wickets"], "state column 'wickets'", minimum=0)
    score = _ints(state["score"], "state column 'score'", minimum=0)
    win_lb = _ints(state["win_lb"], "state column 'win_lb'", minimum=0)
    phase = _ints(state["phase"], "state column 'phase'", minimum=0)
    rrr = _floats(state["rrr"], "state column 'rrr'")
    runs_needed = _floats(state["runs_needed"], "state column 'runs_needed'")
    terminal = terminal_post_wicket(state["wickets"], state["btc_count"])
    n_undefined_btc = int(state["btc_count"].isna().sum())

    # 2. Per sandbox k: check V_k, run its 2 x D15 fits on V_k, extend V_k with the M2 quantities.
    #    Every fit is an M2Model calibrated on its own Training_Rows (criterion 15); every prediction
    #    below is calibrated, and the raw Out_Of_Fold_Predictions are kept only for the metrics.
    n_rows = len(row_ids)
    r_pre, r_post = np.full(n_rows, np.nan), np.full(n_rows, np.nan)
    first_oof, par_raw = np.full(n_rows, np.nan), np.full(n_rows, np.nan)
    oof_raw = {model: np.full(n_rows, np.nan) for model, _ in _VARIANTS}  # before the calibration
    held_out = {name: np.full(n_rows, np.nan) for name in VIEW_M2_COLUMNS}  # fold-k rows of V_k
    outer_ids = {model: np.full(n_rows, None, dtype=object) for model, _ in _VARIANTS}
    calibrations: dict[str, pd.DataFrame] = {}
    means_tables: dict[str, pd.DataFrame] = {}
    bias_tables: dict[str, pd.DataFrame] = {}
    summaries: list[dict[str, Any]] = []
    view_missing: dict[str, list[int]] = {name: [] for name in VIEW_M2_COLUMNS}
    for k in all_folds:
        began = time.perf_counter()
        view = store.load_table(SKILL_STAGE, views.view_table(k))
        views.check_view(k, view, row_fold, registry)  # before any fit of sandbox k reads V_k
        X = m2_features(view, state)
        X_post = post_wicket_features(view, state)
        values: dict[str, dict[str, np.ndarray]] = {name: {} for name in VIEW_M2_COLUMNS}
        rounds: dict[str, list[int]] = {model: [] for model, _ in _VARIANTS}
        for spec in (spec for spec in plan if spec.sandbox_k == k):  # M2(-k), then every M2^(k)(-j)
            served = np.flatnonzero(row_fold == spec.predicted_fold)
            fitted: dict[str, common.RegisteredFit] = {}
            for model, _ in _VARIANTS:
                fit = common.fit_registered(
                    registry,
                    spec,
                    model=model,
                    family=FAMILY,
                    match_ids=match_ids,
                    eligible=eligible[model],
                    predicted_match_ids=match_ids[served],
                    early_stopping=setup.early_stopping,
                    fit=m2_fitter(setup, X, y[model]),
                )
                store.save_model(STAGE_NAME, fit.fit_id, fit.model.booster)  # LightGBM model text
                calibrations[fit.fit_id] = fit.model.calibration  # saved as LB_CALIBRATION_TABLE
                summaries.append(_fit_summary(fit, model, registry))
                rounds[model].append(summaries[-1]["best_iteration"])
                fitted[model] = fit
            first = fitted[FIRST_INNINGS_MODEL]
            training = first.rows.training
            means = par_means(X, lb_bowled, sched, training)
            bias = par_bias(first.model, means, X, lb_bowled, sched, score, training)
            means_tables[first.fit_id], bias_tables[first.fit_id] = means, bias
            quantities = _served_values(first.model, means, bias, X, served, innings, lb_bowled, sched, win_lb, score, rrr)
            for name in VIEW_M2_COLUMNS:
                values[name][first.fit_id] = quantities[name]
            if spec.kind != folds.OUTER:
                continue
            capped = fitted[CAPPED_MODEL]
            r_pre[served] = predict(capped.model, X.iloc[served])
            r_post[served] = predict(capped.model, X_post.iloc[served])
            oof_raw[CAPPED_MODEL][served] = capped.model.predict_raw(X.iloc[served])
            one = served[innings[served] == _FIRST_INNINGS]
            first_oof[one] = predict(first.model, X.iloc[one])
            oof_raw[FIRST_INNINGS_MODEL][one] = first.model.predict_raw(X.iloc[one])
            par_raw[served] = quantities[PAR_RAW_COLUMN]
            for name in VIEW_M2_COLUMNS:
                held_out[name][served] = quantities[name]
            for model, fit in fitted.items():
                outer_ids[model][served] = fit.fit_id
        sources = {name: views.FoldValues(views.M2_FIT, first_plan, values[name]) for name in VIEW_M2_COLUMNS}
        extended = views.extend_view(k, view, row_fold, sources)
        views.check_view(k, extended, row_fold, registry)
        store.save_table(STAGE_NAME, views.view_table(k), extended)
        for name in VIEW_M2_COLUMNS:
            view_missing[name].append(int(extended[name].isna().sum()))
        ctx.log.info(
            f"sandbox {k}: fitted {len(rounds[CAPPED_MODEL])} capped and {len(rounds[FIRST_INNINGS_MODEL])} "
            f"first-innings M2 fits on view V_{k} in {time.perf_counter() - began:.1f} s (boosting rounds kept: "
            f"capped {'/'.join(map(str, rounds[CAPPED_MODEL]))}, first-innings "
            f"{'/'.join(map(str, rounds[FIRST_INNINGS_MODEL]))}; outer fit first); extended V_{k} with "
            f"{', '.join(VIEW_M2_COLUMNS)} and {views.M2_FIT}, checked it and saved it as {views.view_table(k)}",
            stage=STAGE_NAME,
        )
        del view, extended, X, X_post

    # 3. Wicket_Cost of every Delivery, from the outer capped fit of its fold (criteria 8-10 and 13).
    cost = wicket_cost(r_pre, r_post, runs_needed, innings, terminal)
    preds = pd.DataFrame(
        {
            "row": state["row"].to_numpy(),
            "r_pre": r_pre,
            "r_post_wicket": cost.r_post_wicket,
            "omega": cost.omega,
            "omega_clipped_low": cost.clipped_low,
            "omega_clipped_high": cost.clipped_high,
            PAR_RAW_COLUMN: par_raw,
            "capped_fit": pd.array(outer_ids[CAPPED_MODEL], dtype="str"),
            "first_fit": pd.array(outer_ids[FIRST_INNINGS_MODEL], dtype="str"),
        },
        columns=list(PREDS_COLUMNS),
    )
    store.save_table(STAGE_NAME, PREDS_TABLE, preds)

    # 4. The metrics (criteria 5, 9, 10 and 15), the calibration and par-bias files and the Omega table
    #    and chart (criterion 12).
    oof = {CAPPED_MODEL: r_pre, FIRST_INNINGS_MODEL: first_oof}
    accuracy = {
        model: (
            rmse_by_innings(y[model], oof[model], innings, eligible[model]),
            lookup_baseline_rmse(lb_left, wickets, y[model], row_fold, innings, eligible[model]),
        )
        for model, _ in _VARIANTS
    }
    raw_rmse = {model: rmse_by_innings(y[model], oof_raw[model], innings, eligible[model]) for model, _ in _VARIANTS}
    bands = {
        model: _band_means(y[model], oof_raw[model], oof[model], innings, lb_left, eligible[model])
        for model, _ in _VARIANTS
    }
    metrics = _metrics_table(
        accuracy, raw_rmse, bands, cost.omega, cost.clipped_low, cost.clipped_high, innings, terminal,
        needed_nonpositive,
    )  # fmt: skip
    calibration_table = _calibration_table(calibrations)
    bias_table = _bucket_table(bias_tables, PAR_BIAS_COLUMNS)
    means_table = _bucket_table(means_tables, (*PAR_MEAN_FEATURES, "n_rows"))
    omega_table = _omega_table(innings, lb_left, wickets, cost.omega)
    outdir = settings.path("outputs") / M2_FOLDER
    _write_csv(outdir / METRICS_FILE, metrics)
    _write_csv(outdir / LB_CALIBRATION_FILE, calibration_table)
    _write_csv(outdir / PAR_BIAS_FILE, bias_table)
    _write_csv(outdir / OMEGA_TABLE_FILE, omega_table)
    _write_omega_chart(outdir / OMEGA_CHART_FILE, omega_table)

    # 5. Checkpoint tables, the fit registry and the data-derived Method_Constant.
    fit_table = pd.DataFrame.from_records(summaries, columns=list(FIT_SUMMARY_COLUMNS))
    fit_table = fit_table.astype({"fit_id": "str", "model": "str", "sandbox_k": "Int32", "inner_j": "Int32"})
    store.save_table(STAGE_NAME, LB_CALIBRATION_TABLE, calibration_table)
    store.save_table(STAGE_NAME, PAR_MEANS_TABLE, means_table)
    store.save_table(STAGE_NAME, PAR_BIAS_TABLE, bias_table)
    store.save_table(STAGE_NAME, FIT_SUMMARY_TABLE, fit_table)
    n_saved = registry.save(store, STAGE_NAME)
    outer = fit_table.iloc[np.flatnonzero(fit_table["inner_j"].isna().to_numpy())]
    ctx.constants.record(
        BEST_ITERATION_CONSTANT,
        dict(zip(outer["fit_id"].tolist(), outer["best_iteration"].astype(int).tolist())),
        _COMPONENT,
    )

    _log_stage(
        ctx,
        _StageSummary(
            n_folds=n_folds,
            plan_size=len(plan),
            objective=str(setup.params["objective"]),
            fit_table=fit_table,
            innings=innings,
            phase=phase,
            score=score,
            held_out=held_out,
            par_raw=par_raw,
            view_missing=view_missing,
            accuracy=accuracy,
            raw_rmse=raw_rmse,
            bands=bands,
            calibration_table=calibration_table,
            omega=cost.omega,
            clipped_low=cost.clipped_low,
            clipped_high=cost.clipped_high,
            terminal=terminal,
            needed_nonpositive=needed_nonpositive,
            n_undefined_btc=n_undefined_btc,
            bias_table=bias_table,
            outdir=outdir,
            n_saved=n_saved,
            seconds=time.perf_counter() - started,
        ),
    )


@dataclass(frozen=True, eq=False)
class _StageSummary:
    """What the Run_Log reports about one run of the stage."""

    n_folds: int
    plan_size: int
    objective: str
    fit_table: pd.DataFrame
    innings: np.ndarray
    phase: np.ndarray
    score: np.ndarray
    held_out: Mapping[str, np.ndarray]
    par_raw: np.ndarray
    view_missing: Mapping[str, list[int]]
    accuracy: Mapping[str, tuple[pd.DataFrame, pd.DataFrame]]
    raw_rmse: Mapping[str, pd.DataFrame]
    bands: Mapping[str, pd.DataFrame]
    calibration_table: pd.DataFrame
    omega: np.ndarray
    clipped_low: np.ndarray
    clipped_high: np.ndarray
    terminal: np.ndarray
    needed_nonpositive: np.ndarray
    n_undefined_btc: int
    bias_table: pd.DataFrame
    outdir: Path
    n_saved: int
    seconds: float


def _span(values: Sequence[int]) -> str:
    low, high = min(values), max(values)
    return f"{low:,}" if low == high else f"{low:,}-{high:,}"


def _per_view(counts: Sequence[int]) -> str:
    """``"N in every view"`` when the views agree, else the count of each view."""
    if len(set(counts)) == 1:
        return f"{counts[0]:,} in every view"
    return f"V_1-V_{len(counts)} " + "/".join(f"{count:,}" for count in counts)


def _phase_means(values: np.ndarray, mask: np.ndarray, phase: np.ndarray) -> str:
    """The mean of ``values`` over the ``mask`` rows of each Phase (powerplay, middle, death), as ``a/b/c``."""
    parts = []
    for member in Phase:
        chosen = mask & (phase == int(member)) & np.isfinite(values)
        parts.append(f"{float(values[chosen].mean()):+.2f}" if chosen.any() else "n/a")
    return "/".join(parts)


#: The Legal_Balls-left values whose calibration factors the Run_Log reports for the outer fits.
_LOGGED_LB_LEFT: tuple[int, ...] = (1, 2, 6, 18, 60, 120)


def _log_calibration(ctx: StageContext, s: _StageSummary) -> None:
    """The Run_Log lines on the Legal_Balls-left calibration (criterion 15): outer-fit factors and band means."""
    table, fits = s.calibration_table, s.fit_table
    table_ids = table["fit_id"].astype(str).to_numpy(dtype=object)
    outer = fits["inner_j"].isna().to_numpy()
    parts = []
    for model, label in ((CAPPED_MODEL, "capped"), (FIRST_INNINGS_MODEL, "first-innings")):
        ids = fits["fit_id"][outer & (fits["model"].to_numpy() == model)].astype(str).tolist()
        factors: dict[int, list[float]] = {value: [] for value in _LOGGED_LB_LEFT}
        for fit_id in ids:
            mine = np.flatnonzero(table_ids == fit_id)
            knots = table["lb_left"].to_numpy(dtype=np.int64)[mine]
            values = table["factor"].to_numpy(dtype=np.float64)[mine]
            for value in _LOGGED_LB_LEFT:  # the step rule of M2Model.factor
                factors[value].append(float(values[np.clip(np.searchsorted(knots, value, side="right") - 1, 0, None)]))
        parts.append(
            f"{label} " + ", ".join(f"c({value}) {min(got):.3f}-{max(got):.3f}" for value, got in factors.items())
        )
    floored = int((table["factor"].to_numpy(dtype=np.float64) <= _MIN_LB_FACTOR).sum())
    ctx.log.info(
        f"Legal_Balls-left calibration of every M2 fit on its own Training_Rows (Requirement 9, criterion 15; "
        f"DD-15): factors of the {s.n_folds} outer fits of each variant, {'; '.join(parts)}; {floored:,} of the "
        f"{len(table):,} factors of the {s.plan_size * len(_VARIANTS)} fits are at the floor {_MIN_LB_FACTOR:g}",
        stage=STAGE_NAME,
    )
    texts = []
    for model, label in ((CAPPED_MODEL, "capped"), (FIRST_INNINGS_MODEL, "first-innings")):
        for row in s.bands[model].to_dict("records"):
            if row["n_rows"]:
                texts.append(
                    f"{label} innings {int(row['innings'])} {_band_label(int(row['low']), int(row['high']), '-')} "
                    f"{row['mean_target']:.2f}/{row['mean_pred_raw']:.2f}/{row['mean_pred']:.2f}"
                )
    ctx.log.info(
        "held-out mean remaining runs by Legal_Balls left before the ball, actual/raw/calibrated (the targets and the "
        "outer fits' Out_Of_Fold_Predictions before and after the calibration, over the Training_Rows of all folds): "
        + "; ".join(texts),
        stage=STAGE_NAME,
    )


def _log_stage(ctx: StageContext, s: _StageSummary) -> None:
    log = ctx.log
    fits = s.fit_table
    outer = fits["inner_j"].isna().to_numpy()
    parts = []
    for model, label in ((CAPPED_MODEL, "capped"), (FIRST_INNINGS_MODEL, "first-innings")):
        mine = fits["model"].to_numpy() == model
        parts.append(
            f"{label}: Training_Rows {_span(fits['n_train_rows'][mine & outer].tolist())} per outer fit and "
            f"{_span(fits['n_train_rows'][mine & ~outer].tolist())} per inner fit, boosting rounds kept "
            f"{_span(fits['best_iteration'][mine & outer].tolist())} (outer) and "
            f"{_span(fits['best_iteration'][mine & ~outer].tolist())} (inner)"
        )
    log.info(
        f"fitted M2 (Requirement 9, criteria 1-4) through the fold-sandbox plan: {s.plan_size} capped fits (both "
        f"innings, innings-2 target capped at the runs needed) and {s.plan_size} first-innings fits, each "
        f"{s.n_folds} outer and {s.plan_size - s.n_folds} inner, objective {s.objective}, monotone constraints "
        f"{list(M2_MONOTONE)} on {', '.join(M2_FEATURES)}; " + "; ".join(parts),
        stage=STAGE_NAME,
    )
    first, chase = s.innings == _FIRST_INNINGS, s.innings == _CHASE
    log.info(
        f"views V_1-V_{s.n_folds} now hold {', '.join(VIEW_M2_COLUMNS)} and {views.M2_FIT} (fold-i rows of V_k from "
        f"M2_first^(k)(-i), fold-k rows from M2_first(-k); Requirement 7, criterion 8), each checked with check_view. "
        f"Missing values, never infinite: {PAR_COLUMNS[0]} {_per_view(s.view_missing[PAR_COLUMNS[0]])} (the "
        f"{int(chase.sum()):,} innings-2 rows), {ACH_RATE_COLUMN} {_per_view(s.view_missing[ACH_RATE_COLUMN])} (the "
        f"{int(first.sum()):,} innings-1 rows), {PAR_COLUMNS[2]} {_per_view(s.view_missing[PAR_COLUMNS[2]])} (also "
        f"0 Legal_Balls bowled), {F19_COLUMN} {_per_view(s.view_missing[F19_COLUMN])}",
        stage=STAGE_NAME,
    )
    deficit = s.held_out[F19_COLUMN]
    raw_deficit = s.par_raw - s.score
    log.info(
        "held-out Par_Score (fold-k rows of V_k, from the outer first-innings fits; Requirement 9, criteria 6 and 14): "
        f"mean Par_Score minus score by Phase (powerplay/middle/death) {_phase_means(deficit, first, s.phase)} runs "
        f"with the bias correction, {_phase_means(raw_deficit, first, s.phase)} raw; innings 2, mean Required_Rate "
        f"minus Achievable_Rate by Phase {_phase_means(deficit, chase, s.phase)} runs per over",
        stage=STAGE_NAME,
    )
    texts, worse = [], []
    for model, label in ((CAPPED_MODEL, "capped"), (FIRST_INNINGS_MODEL, "first-innings")):
        m2, baseline = s.accuracy[model]
        raw = s.raw_rmse[model]
        for ours, unscaled, base in zip(m2.to_dict("records"), raw.to_dict("records"), baseline.to_dict("records")):
            if not ours["n_rows"]:
                continue
            texts.append(
                f"{label} innings {int(ours['innings'])} {ours['rmse']:.3f} ({unscaled['rmse']:.3f} before the "
                f"Legal_Balls-left calibration) vs {base['rmse']:.3f}"
            )
            if not ours["rmse"] < base["rmse"]:
                worse.append(f"{label} innings {int(ours['innings'])}")
    log.info(
        "M2 against the lookup baseline, RMSE pooled over the Out_Of_Fold_Predictions of the Training_Rows of all "
        f"{s.n_folds} folds (Requirement 9, criterion 5): " + "; ".join(texts),
        stage=STAGE_NAME,
    )
    _log_calibration(ctx, s)
    if worse:
        log.warning(
            f"M2 is not more accurate than the lookup baseline for {', '.join(worse)}; see "
            f"{_shown_path(s.outdir, ctx.root)}/{METRICS_FILE}",
            stage=STAGE_NAME,
        )
    high_at_zero = int((s.clipped_high & s.needed_nonpositive).sum())
    log.info(
        f"Wicket_Cost Omega of all {len(s.omega):,} Deliveries of innings 1 and 2 from the outer capped fits "
        f"(Requirement 9, criteria 8-10 and 13): mean {float(s.omega[first].mean()):.2f} runs in innings 1 and "
        f"{float(s.omega[chase].mean()):.2f} in innings 2; post-wicket runs taken as 0 on {int(s.terminal.sum()):,} "
        f"Deliveries (9 wickets down or an empty Batters_To_Come); clipped low (negative, set to 0) on "
        f"{int((s.clipped_low & first).sum()):,} innings-1 and {int((s.clipped_low & chase).sum()):,} innings-2 "
        f"Deliveries; clipped high (innings 2, set to the runs needed) on {int(s.clipped_high.sum()):,}, of them "
        f"{high_at_zero:,} with runs needed of 0 or less ({int(s.needed_nonpositive.sum()):,} innings-2 Deliveries "
        "have runs needed of 0 or less before the ball and are kept)",
        stage=STAGE_NAME,
    )
    if s.n_undefined_btc:
        log.warning(
            f"{s.n_undefined_btc:,} Deliveries have an undefined Batters_To_Come (no is_played row for the batting "
            "side); their Post_Wicket_State reads a missing next batter, so their Omega rests on M2's handling of "
            "a missing value (Requirement 9, criterion 13 covers only an empty Batters_To_Come)",
            stage=STAGE_NAME,
        )
    beta = s.bias_table.pivot(index="bucket", columns="fit_id", values="beta")
    log.info(
        f"Par_Score bias by bucket over the {beta.shape[1]} first-innings fits (par_bias): bucket 1 "
        f"{float(beta.loc[1].min()):+.1f} to {float(beta.loc[1].max()):+.1f} runs, bucket 60 "
        f"{float(beta.loc[60].min()):+.1f} to {float(beta.loc[60].max()):+.1f}, bucket 119 "
        f"{float(beta.loc[119].min()):+.1f} to {float(beta.loc[119].max()):+.1f}",
        stage=STAGE_NAME,
    )
    shown = _shown_path(s.outdir, ctx.root)
    log.info(
        f"wrote {shown}/{METRICS_FILE}, {shown}/{LB_CALIBRATION_FILE}, {shown}/{PAR_BIAS_FILE}, "
        f"{shown}/{OMEGA_TABLE_FILE} and {shown}/{OMEGA_CHART_FILE}; saved the views view_1-view_{s.n_folds}, "
        f"{PREDS_TABLE} ({len(s.omega):,} rows), {LB_CALIBRATION_TABLE} ({len(s.calibration_table):,} rows), "
        f"{PAR_MEANS_TABLE}, {PAR_BIAS_TABLE} ({len(s.bias_table):,} rows), {FIT_SUMMARY_TABLE}, the {len(fits)} "
        f"Boosters under their fit ids and the fit registry ({s.n_saved} fits) in {s.seconds:.1f} s",
        stage=STAGE_NAME,
    )
