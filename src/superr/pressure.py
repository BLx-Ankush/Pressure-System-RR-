"""Pressure_Engine (Requirements 11-13; design DD-8 and section "Pressure_Engine (``pressure.py``,
``ranks.py``)").

The module holds the whole Pressure_Engine: the Post_Outcome_States (task 13.1), the
Leverage_Index (task 15.1, Requirement 11, criteria 9-11), the Pillar_Components, Stress_Pillars
and Stress (task 15.2, Requirement 12), and the Live_Ball flags, the finiteness guard, the
Pressure_Index, the Pressure_Ball and Ordinary_Ball flags, the share tables, the mean-LI chart
(Requirement 11, criterion 13), the pressure definition file and the ``pressure``
Pipeline_Stage (task 15.3, Requirement 13). The ``model_choice`` stage builds the
Post_Outcome_States once and evaluates every M3 Candidate_Model on them (DD-8), so that the
Leverage_Index, the sum over the 9 classes of P(class) x |WP(Post_Outcome_State of the class) -
WP(pre-ball state)| (Requirement 11, criterion 9), is a row-wise sum of stored arrays
(``leverage``). The percentile ranks are in ``superr.ranks``. No function reads ``config.py``.
The Post_Outcome_States and the Leverage_Index have no Config setting. The pillar and PI
functions receive the Design_Choices they use as arguments, without a default: the D7 cap and
the D5 weights as keyword arguments, the D4 scope through the rank ``groups`` (``rank_groups``),
the Live_Balls (D1) as the ``live`` mask (``live_balls`` with the D1 band), and the D2 alpha and
D3 share as the last arguments of ``pressure_index`` and ``pressure_flags``; the D6 window enters
through the window columns of the ``state`` table, which ``window`` replaces for another length.
Only ``run_stage`` reads these values from ``ctx.settings``. The Robustness_Suite passes its
alternatives the same way (section "Robustness_Cases").

Post_Outcome_States (``post_outcome_states``, PostStates; Requirement 11, criterion 15)
    Every Delivery of innings 1 and 2 (each row of the ``state`` table, Wides and No_Balls
    included) gets one Post_Outcome_State per Outcome_Model class, in CLASSES order (``dot``,
    ``r1``, ``r2``, ``r3``, ``r4``, ``r6``, ``other_runs``, ``bowler_wicket``, ``run_out``;
    ``models.outcome``): its pre-ball state after one hypothetical Legal_Ball of that class,
    built in the three steps of criterion 15.

    1. One more Legal_Ball: Legal_Balls bowled L + 1 and Legal_Balls left max(left - 1, 0).
    2. The class's Class_Run_Value added to the team score and, in innings 2, taken from the
       runs needed. The values are exactly 0, 1, 2, 3, 4 and 6 for the dot and run classes and
       means for the other three (as delivered 5.0 for ``other_runs``, 0.0 for
       ``bowler_wicket`` and 0.42-0.44 for ``run_out``), so a score or runs needed can be
       fractional.
    3. For ``bowler_wicket`` and ``run_out`` only (WICKET_CLASSES), the Post_Wicket_State
       changes that Omega uses (Requirement 9, criterion 8), exactly as
       ``remaining_runs.post_wicket_features`` makes them: wickets fallen + 1, striker theta_bat
       = the view's ``theta_next_batter``, striker Legal_Balls faced 0 and RBQ = the view's
       ``rbq_post_wicket``.

    Nothing else changes. The striker stays on strike (no strike rotation, after odd runs or at
    the end of an Over), the striker's Legal_Balls faced stay as they were for the seven other
    classes, and the non-striker's theta_bat, Scoring_Environment, Venue_Effect,
    Match_Day_Conditions, Team_Strength_Gap and Scheduled_Balls keep their pre-ball values. The
    model-derived M3 features then follow the new state:

    * innings 1: Par_Score at L + 1, the view's ``par_score_next``. Par_Score depends only on
      the Legal_Balls bowled, the fit and the context (Requirement 9, criterion 14), so all 9
      classes get the same value;
    * innings 2: Required_Rate (``rrr``) = 6 x runs needed / Legal_Balls left, Achievable_Rate
      (``ach_rate``) = 6 x the first-innings M2 fit's expected remaining runs for the state's M2
      features / Legal_Balls left (``remaining_runs.achievable_rate``), and ``rr_minus_ar``,
      Required_Rate minus Achievable_Rate. No M2 feature depends on the runs, so the M2 features
      of the seven non-wicket classes are equal, and so are those of the two wicket classes: each
      Delivery needs two M2 predictions, and none when no Legal_Ball is left after the ball.

    An outcome that ends the first innings (``innings_over``: L + 1 reaches Scheduled_Balls SB,
    or a wicket class at 9 wickets fallen) takes the end-of-innings state of Requirement 10,
    criterion 17 instead (Requirement 11, criterion 15): Legal_Balls bowled = SB (and Legal_Balls
    left 0), wickets fallen = the pre-ball wickets, Par_Score = Par_Score at SB
    (``win_prob.innings_end_par``: the corrected par curve of the Delivery's first-innings M2 fit
    and context, the curve behind its ``par_score_next``), the score after the outcome, and the
    pre-ball Team_Strength_Gap, Scoring_Environment, Venue_Effect and Scheduled_Balls. Once the
    innings is over the wickets in hand no longer matter, so the classifier's wickets effect stays
    out of the WP differences between the classes. The innings-1 classifier scores these states,
    so no Delivery's outcomes are scored by two different classifiers; the jump to the chasing
    side's first Delivery falls between Deliveries. At the last Legal_Ball the 9 states differ
    only in their score, and the dot-ball state differs from the pre-ball state only by one more
    Legal_Ball bowled and Par_Score(SB) in place of Par_Score(SB - 1). The columns that are no
    M3 feature of innings 1 (the M2_COLUMNS) keep their step 1-3 values.

    Each Delivery takes the values of its Out_Of_Fold pre-ball WP: a Delivery of fold k takes
    every view value from V_k; its Achievable_Rate and its Par_Score at SB from the first-innings
    M2 fit that its V_k row names in ``m2_fit``, the outer fit M2(-k) (``m2_first_lgbm_outer_k<k>``),
    which also gave its pre-ball Achievable_Rate and Par_Score, with that fit's par means and par
    bias (``remaining_runs`` tables ``m2_par_means`` and ``m2_par_bias``); and the
    Class_Run_Values of fold k, those of the M4 fit of fold k (from the Legal_Balls of its
    Training_Folds).

Layout (PostStates, POST_COLUMNS)
    For each innings i, ``PostStates.frame(i)`` stacks the innings' Post_Outcome_States in one
    DataFrame on a RangeIndex, N_CLASSES rows per Delivery, the Deliveries in ``state`` order and
    each Delivery's rows in CLASSES order: state s is class s % 9 of the Delivery at position
    ``positions[i][s // 9]`` of the ``state`` table. A per-state array v of innings i therefore
    reshapes to the (Deliveries, 9) matrix v.reshape(-1, 9) (``by_delivery``), and ``matrix``
    puts the matrices of both innings back in ``state`` order. The columns (POST_COLUMNS[i]):

    ===================== ======================================================================
    Column                Value in the Post_Outcome_State
    ===================== ======================================================================
    row, match_id, fold   the Delivery's ``row`` id, its match and its fold (int32)
    innings, cls          the innings and the class, an index into CLASSES (int8)
    lb_bowled             L + 1; Scheduled_Balls on an innings-1 state with ``innings_over``
                          (int32)
    lb_left               max(Legal_Balls left - 1, 0); 0 on an innings-1 state with
                          ``innings_over`` (int32)
    score                 the pre-ball team score + the Class_Run_Value (float64)
    wickets               the pre-ball wickets fallen, + 1 for WICKET_CLASSES, except on an
                          innings-1 state with ``innings_over``, which keeps the pre-ball
                          wickets (int32)
    runs_needed           innings 2: the pre-ball runs needed - the Class_Run_Value, which is
                          Target - ``score``; NaN in innings 1 (float64)
    target                innings 2: the Target; NaN in innings 1, whose Target is not yet set
                          (float64)
    sched_balls           the innings' Scheduled_Balls (int32)
    rbq ... match_day     M2_COLUMNS, the M2 features other than ``lb_left`` and ``wickets``:
                          ``rbq``, ``theta_striker``, ``striker_lb_faced`` (int32),
                          ``theta_non_striker``, ``scoring_env``, ``venue_effect`` and
                          ``match_day``, with the step-3 changes for WICKET_CLASSES
    tsg                   Team_Strength_Gap
    par_score             innings 1 only: Par_Score at L + 1, or at Scheduled_Balls on a state
                          with ``innings_over``
    rrr, ach_rate,        innings 2 only: Required_Rate, Achievable_Rate and their difference
    rr_minus_ar
    innings_over          the outcome ends the innings: L + 1 >= ``sched_balls`` (no Legal_Ball
                          left) or ALL_OUT_WICKETS (10) wickets after the outcome
    terminal              innings 1: False (the innings-1 classifier scores every state);
                          innings 2: ``innings_over`` or ``runs_needed`` <= 0 (the chase is won)
    ===================== ======================================================================

    Every M3 feature of the innings is a column (M3_FEATURES[i] of ``models.win_prob``), and
    ``features(i)`` gives them as float64 columns in that order, the input of every M3
    Candidate_Model, together with ``fold(i)``. The float columns hold no infinite value.

Innings ends and terminal states (``innings_over``, ``terminal``)
    ``innings_over`` flags every state whose outcome ends the innings, for reporting
    (``PostStates.innings_over``). In innings 1 such a state is not terminal: it holds the
    end-of-innings state and the innings-1 classifier scores it (Requirement 10, criterion 17).
    A terminal state, innings 2 only, gets its WP from the terminal rules of Requirement 10,
    criteria 14-16 (``win_prob.apply_terminal_rules``) rather than from a classifier, so no model
    reads its features: 1 when ``runs_needed`` <= 0 (whatever the balls and wickets), else, with
    the innings over, 0 when ``runs_needed`` > 1 (score below Target - 1), 0.5 when it is exactly
    1 (scores level), and the rule for fractional runs needed between 0 and 1. This module only
    flags the states and keeps their values.

Missing values
    A float column is NaN only where its value is undefined:

    * ``runs_needed``, ``target``: always in innings 1; in innings 2 where the Target is missing
      (no Delivery as delivered);
    * ``rrr``, ``ach_rate``, ``rr_minus_ar``: where ``lb_left`` is 0, which ends the innings, so
      only on terminal states; ``rrr`` and ``rr_minus_ar`` also where the Target is missing;
    * ``theta_striker``: on the wicket-class states of a Delivery whose Batters_To_Come is empty
      (the view's ``theta_next_batter`` is NaN). The first-innings M2 fit predicts such a state
      with the missing value, as ``remaining_runs`` predicts the Post_Wicket_State before
      Requirement 9, criterion 13 sets Omega's post-wicket runs to 0. Most of these Deliveries
      have 9 wickets fallen, so their wicket-class states are all out: end-of-innings states in
      innings 1, whose M3 features hold no ``theta_striker``, and terminal states in innings 2;
    * a value that is missing in the view stays missing (none as delivered).

Use in ``model_choice`` (task 13.2)
    ``post_outcome_states`` runs once per Run, before any candidate is evaluated, on the
    ``state`` and ``balls`` tables, the views V_1 ... V_D15 of the ``remaining_runs``
    checkpoint, the fold of each row, the Class_Run_Values of the ``outcome`` stage, the outer
    first-innings M2 fits (``first_innings_models``) and the ``remaining_runs`` tables
    ``m2_par_means`` and ``m2_par_bias``. For each M3 Candidate_Model and innings i, the
    classifier then evaluates the states that are not terminal (``live = ~terminal(i)``: every
    state in innings 1), each with the calibrated outer model of its fold
    (``CalibratedFolds.predict(features(i).iloc[live], fold(i)[live])``); the terminal rules give
    the WP of the terminal states of innings 2; and ``matrix({1: wp_1, 2: wp_2})`` gives
    ``wp_post_0`` ... ``wp_post_8``, the (rows, 9) matrix in ``state`` order and CLASSES order.

As delivered (checks of 15 November 2026, on the ``remaining_runs`` checkpoint of 14 November)
    The 200,961 Deliveries (105,727 in innings 1, 95,234 in innings 2) give 1,808,649
    Post_Outcome_States (951,543 and 857,106), built in about 2-3 seconds with 189,884 M2
    predictions (two for each of the 94,942 innings-2 Deliveries with a Legal_Ball left after
    the ball); the two frames take about 240 MB. 9,219 innings-1 states end the innings (7,153
    by balls only, 1,902 by wickets only, 164 by both) and hold end-of-innings states, none of
    them terminal: their Par_Score at Scheduled_Balls comes from the par curves of 1,764
    Deliveries (the 813 last Legal_Balls, where it equals ``par_score_next`` exactly, and 951
    more at 9 wickets down). 11,766 innings-2 states are terminal: 6,455 won (269 of them with
    the innings also over), 5,229 lost (1,744 of them with fractional runs needed above 1), 69
    tied and 13 with fractional runs needed between 0 and 1. ``rrr``, ``ach_rate`` and
    ``rr_minus_ar`` are missing on 2,628 innings-2 states, all terminal. ``theta_striker`` is
    missing on the 5,216 wicket-class states of the 2,608 Deliveries with an empty
    Batters_To_Come; 32 of these Deliveries (2 in innings 1, 30 in innings 2) have fewer than 9
    wickets fallen, so 62 of their 64 wicket-class states do not end the innings (the other 2
    end it by balls). No M3 feature of a non-terminal state is missing, the end-of-innings states
    included. For the 98,361 innings-1 Legal_Balls without extras whose class is not
    ``run_out``, the state of the actual class has exactly the next Delivery's innings-1 M3
    features, and for the 88,665 such innings-2 balls its runs needed, Legal_Balls left,
    wickets and Team_Strength_Gap. The only exceptions are the wickets of 9 Retire Out and
    Obstructing-the-field balls, which criterion 14 labels by their bat runs.

Leverage_Index (``leverage``, ``primary_leverage``, LeverageIndex; Requirement 11, criteria 9-11)
    Every Delivery of innings 1 and 2, Wides and No_Balls included, gets an unscaled LI: the sum
    over the 9 classes c, in CLASSES order, of P_c x |WP_post,c - WP_pre| (criterion 9). P is the
    Delivery's LI_VARIANT (``avg_both``) M4 prediction, with theta_bat and theta_bowl both at the
    fold's Average_Skill; WP_pre is its Out_Of_Fold pre-ball WP and WP_post,c the WP of its
    Post_Outcome_State of class c, after the terminal rules. All three come from the
    Primary_Models (``model_choice.load_primary``: M3-1 on innings-1 rows, M3-2 on innings-2
    rows), each from the outer fit that excluded the Delivery's fold. P sums to 1 and no
    |WP_post,c - WP_pre| exceeds 1, so the unscaled LI lies between 0 and 1. It is exactly 0 when
    no class moves the WP: a chase already won, or a WP that every class leaves at a calibration
    bound.

    The LI divides each unscaled LI by one number, the mean unscaled LI pooled over all
    Deliveries of innings 1 and 2 in all matches (criterion 10), not by a mean per innings, Phase
    or fold. The mean LI over those Deliveries is therefore 1 (checked within 1e-6), and an LI of
    2 means twice the average expected WP swing. Dividing by a positive constant keeps the order
    of the Deliveries, so it changes no Percentile_Rank of LI. The LI is one value per Delivery:
    the batter Role of a Batter_Delivery and the bowler Role of a Bowler_Delivery read the same
    value (criterion 11), and no Role-specific input enters it.

    ``leverage(p_avg, wp_pre, wp_post)`` computes it from the arrays (p_avg and wp_post (n, 9) in
    CLASSES order, wp_pre (n,)), which must cover every Delivery of innings 1 and 2, because the
    pooled mean is the mean over the rows given. It checks that every value is finite and within
    [0, 1] and that every P row sums to 1 within PROB_SUM_TOLERANCE (1e-6, the rule of criterion
    5), and uses P as given, without rescaling. It returns a LeverageIndex: ``li`` and
    ``unscaled`` (float64 (n,), non-negative, read-only) and ``pooled_mean``, which the report
    cites. ``primary_leverage(primary)`` applies it to a PrimaryPredictions loaded with LI_VARIANT
    among its ``variants``, after checking that every row lies in innings 1 or 2; the arrays
    follow ``primary.row`` (``state`` order). The ``pressure`` stage loads the predictions once,
    because it also needs ``wp_pre`` for the Live_Ball flag::

        primary = model_choice.load_primary(ctx.store, variants=(LI_VARIANT,))
        li = primary_leverage(primary).li

As delivered (Leverage_Index, on the ``model_choice`` checkpoint of 14 November 2026)
    The Primary_Models (logistic M3-1 and M3-2, LightGBM M4) give the 200,961 Deliveries a pooled
    mean unscaled LI of 0.012517: an LI of 1 is an expected WP swing of about 1.25 percentage
    points. The mean LI is exactly 1.0 in float64, and the largest LI is 77.2 (unscaled 0.966).
    Loading the predictions takes about 0.9 s and computing the LI about 0.3 s. 2,499 Deliveries
    (1.24%) have an LI of exactly 0, none of them a Live_Ball: in innings 1, 172 whose WP every
    class leaves at a calibration bound (111 at 0.001, 61 at 0.999); in innings 2, 2,078 held at
    0.001 and the 249 chases already won (WP 1). Mean LI by innings and Phase, with the means of
    the Live_Balls and Dead_Balls in brackets:

    ======= ===================== ===================== =====================
    Innings Powerplay             Middle                Death
    ======= ===================== ===================== =====================
    1       0.870 (0.870 / 0.278) 0.656 (0.662 / 0.247) 0.826 (0.848 / 0.276)
    2       1.147 (1.240 / 0.203) 1.046 (1.327 / 0.184) 1.977 (4.001 / 0.413)
    ======= ===================== ===================== =====================

    Innings 1 averages 0.763 and innings 2 1.263. Among the 18,776 innings-2 death-Phase
    Deliveries, the 10,592 Dead_Balls average 0.413 against 4.001 for the 8,184 Live_Balls (the
    comparison of Requirement 11, criterion 12).

Pillar_Components (``pillar_components``, COMPONENTS; Requirement 12, criteria 1-11)
    Every Delivery of innings 1 and 2 (each ``state`` row, Wides and No_Balls included) gets the
    13 Pillar_Components of the 8 Stress_Pillars and the boundary-only F25(b) of the
    Extras_Free_PI (glossary; Requirement 14, criterion 5). L is the Delivery's Legal_Balls
    bowled, and "the view" is the view of its own fold: a Delivery of fold k reads every view
    value from V_k of the ``remaining_runs`` checkpoint (``own_fold_view``), the view that gave
    its Out_Of_Fold M3 and M4 inputs. Each component keeps the sign of its criterion (criterion
    11), so a larger value means more stress for the Role:

    ================ ====== ============================================================ ========
    Column           Pillar Value                                                        12.x
    ================ ====== ============================================================ ========
    c19              F19    the view's ``f19_component``: innings 1 Par_Score(L) - score  1, 2
                            (runs), innings 2 Required_Rate - Achievable_Rate (runs per
                            over)
    c20a             F20    ``wickets`` - the Training_Folds mean wickets at the same     3
                            innings and L (section "F20(a) means")
    c20b             F20    ``win_wkts``: Counted_Wickets in the D6 window                3
    c20c             F20    1 - RBQ (the view's ``rbq``)                                  3
    c21              F21    D7 - min(``striker_lb_faced``, D7), 0 to D7                   4
    c22a             F22    ``striker_dots``: the striker's dot streak                    5
    c22b             F22    ``lb_since_boundary``: team Legal_Balls since the last        5
                            Boundary
    c23              F23    innings 1 ``win_rr`` - the view's ``par_rate_win`` (window    6, 7
                            run rate - Par_Rate over the window); innings 2 the view's
                            ``ach_rate`` - ``rrr`` (Achievable_Rate - Required_Rate)
    c24a             F24    ``striker_lb_faced`` x theta_bat (the view's                  8
                            ``theta_striker``)
    c24b             F24    RBQ                                                          8
    c25a             F25    ``bowler_over_runs``: the bowler's team runs earlier in the   9
                            Over, byes and leg-byes left out
    c25b             F25    ``prev_in_over_flag``: 1 when the previous Delivery of the    9
                            Over was a Boundary or had an extra, else 0
    c26              F26    ``lb_since_wicket``: Legal_Balls since the last               10
                            Counted_Wicket
    c25b_extras_free F25    ``prev_in_over_boundary``: 1 when the previous Delivery of    14.5
                            the Over was a Boundary, else 0
    ================ ====== ============================================================ ========

    The other columns are the ``state`` table's. The components are float64, NaN where a value
    cannot be computed (criterion 16) and never infinite. c19 and c23 are missing where an input
    rate is (no Target, or no Legal_Ball left in innings 2), c23 also before the first Legal_Ball
    of innings 1 (an empty window: ``win_rr`` and ``par_rate_win`` are both missing), c20a at an
    innings and L that no Training_Folds Delivery has, and a view component where its view value
    is missing. In innings 2, c23 is -c19 bit for bit.

F20(a) means (``train_wicket_means``, ``wicket_means_for``; criterion 3)
    The mean wickets that F20(a) subtracts comes from the Delivery's Training_Folds only.
    ``train_wicket_means`` returns WICKET_MEANS_COLUMNS, one row per target fold k (each fold
    among the rows given) and each (innings i, L) pair among the rows: ``n_train``, the
    Deliveries of innings i with L Legal_Balls bowled in the folds other than k that pass the
    model-training filters (``common.training_mask``: innings 1 or 2, no Super_Over_Innings, no
    excluded match), and ``mean_wickets``, the mean of their pre-ball wickets fallen, NaN when
    ``n_train`` is 0. Every such Delivery counts, Wides and No_Balls included. No value of row
    (k, i, L) depends on a fold-k Delivery, so changing any fold-k Delivery leaves every fold-k
    mean unchanged. ``wicket_means_for`` gives each Delivery the mean of its (fold, innings, L)
    row. The rows of excluded matches get components and ranks like the others; they only stay
    out of the means.

Stress_Pillars (``pillars``, PILLARS; criteria 12 and 14-16)
    The batter Role covers the Batter_Deliveries (Legal_Balls and No_Balls) and the bowler Role
    the Bowler_Deliveries (every Delivery; ``role_masks``). For a Role and a rank group (D4: the
    Innings_Group; ``rank_groups``), the Pillar_Reference_Set is the Role's Live_Balls in the
    group (the ``live`` mask: the unrounded Out_Of_Fold WP within the D1 band). Every Delivery of
    the Role, Dead_Balls included, is ranked against it with ``ranks.percentile_rank_within``
    (criterion 12):

    1. each Pillar_Component gets its Percentile_Rank within the reference set. A missing
       component gets ``ranks.MISSING_RANK`` (0.5) and, on a Live_Ball, leaves the reference
       set of that component (criterion 16);
    2. F19, F21, F23 and F26 are the rank of their one component (criterion 14);
    3. F20, F22, F24 and F25 are the Percentile_Rank of the unweighted mean of their component
       ranks among the same means over the whole Pillar_Reference_Set (criterion 15). A missing
       component has a rank, so every mean is a number and no Live_Ball leaves this set.

    A Live_Ball's rank lies in [0.5 / n, 1 - 0.5 / n], and its mean over the reference set is
    0.5. A Dead_Ball's lies in [0, 1]: 0 below every reference value, 1 above every one. The
    pillar columns of a Role are NaN outside it, so F19-F22 are NaN on Wides. The Extras_Free
    F25 (``f25_extras_free``) ranks the mean of the c25a and c25b_extras_free ranks the same way.

Stress (``stress``; criterion 13)
    A Role's Stress is the mean of its four Stress_Pillars with the D5 weights, which are equal
    (D5_EQUAL, ``design_choices.D5_weights``: 0.25 each). ``stress_bat`` averages F19-F22 on the
    Batter_Deliveries (NaN on Wides), ``stress_bowl`` F23-F26 on every Delivery, and
    ``stress_bowl_extras_free`` F23, F24, the Extras_Free F25 and F26: the Stress of the
    Extras_Free_PI. ``without`` leaves pillars out for the ablations of Requirement 21,
    criterion 1; the Stress is then the equal-weighted mean of the Role's other pillars.

Missing components (``missing_components``, ``write_missing_components``; criterion 16)
    MISSING_COMPONENTS_COLUMNS, one row per Pillar_Component of COMPONENTS (for its own Role) and
    Innings_Group: ``n_deliveries``, the Role's Deliveries in the Innings_Group; ``n_missing``,
    those whose component is missing, each ranked 0.5; ``n_missing_live``, the Live_Balls among
    them, which leave the component's Pillar_Reference_Set. The counts are per Innings_Group in
    every D4 scope. ``write_missing_components`` writes the table to MISSING_COMPONENTS_FILE in
    PRESSURE_FOLDER (``outputs/pressure/missing_components.csv`` in a Run).

Robustness_Cases (Requirement 21, criteria 1, 2 and 4)
    Each case changes one argument and reuses every other input:

    * D6 windows of 12 and 24 Legal_Balls: ``pillar_components(..., window=frame)``, with that
      window's ``win_wkts`` and ``win_rr`` (``state.recent_window``) and ``par_rate_win``
      (``remaining_runs.par_rate`` with its ``win_lb``, from each row's own-fold par curve);
    * D7 caps of 10 and 30: ``new_batter_cap``;
    * D4 pooled: ``groups=rank_groups(innings, D4_POOLED)``, one rank group for both innings
      (the same groups serve the LI, Stress and PI ranks of ``pressure_index`` and the PI rank of
      ``pressure_flags``);
    * the D1 grid (``live_balls(wp, (b, 1 - b))``) and the live band removed
      (``live_balls(wp, (0.0, 1.0))``, every Delivery a Live_Ball): ``live``;
    * the D2 and D3 grids: the ``alpha`` of ``pressure_index`` and the ``share`` of
      ``pressure_flags``;
    * the eight pillar ablations, the D5 alternatives: ``stress(..., without=("F21",))`` and so
      on, one pillar at a time.

    The components do not depend on D1-D5, so a case that changes only those reuses the
    ``c19`` ... ``c25b_extras_free`` columns of the pressure table.

Live_Balls (``live_balls``; Requirement 13, criterion 1)
    A Delivery of innings 1 or 2 is a Live_Ball when lower <= WP <= upper, both D1 bounds
    inclusive (``design_choices.D1_live_band``: 0.05 and 0.95). WP is the Delivery's pre-ball WP
    as ``model_choice`` saved it (``PrimaryPredictions.wp_pre``): the Out_Of_Fold_Prediction of
    the M3 Primary_Model of its innings, after the terminal rules, compared in float64 and never
    rounded (``model_choice`` stores WP as float64 for this reason). Every other Delivery is a
    Dead_Ball; a WP that is missing or not finite is not Live, and finite_guard stops the Run.

Finiteness guard (``finite_guard``, PressureInputError; Requirement 13, criterion 10)
    Before any PI is computed, ``finite_guard`` checks WP on every Delivery, LI on every
    Live_Ball and each Stress column on the Live_Balls of its Role: ``stress_bat`` on the batter
    Role's, ``stress_bowl`` and ``stress_bowl_extras_free`` on the bowler Role's. A value that is
    missing (NaN or NA) or infinite raises PressureInputError, a StageError (exit code 5) whose
    message names each affected quantity with its count of affected Deliveries in each
    Innings_Group, and whose ``counts`` hold the same numbers. ``run_stage`` calls it before any
    PI is computed and saves or writes nothing before it passes, so a failed check leaves no PI or
    Pressure_Ball output of the Run. Earlier checks stop the same problems sooner, also with exit
    code 5 but without the per-Innings_Group counts: ``PrimaryPredictions`` rejects a WP that is
    missing or outside [0, 1] when the predictions are loaded, and ``leverage`` an M4 probability
    or WP that is not finite. Stress is defined on every Delivery of its Role. As delivered the
    guard therefore passes; it guards the recombined inputs of the Robustness_Suite and the
    model-choice reruns as well.

Pressure_Index (``pressure_index``; Requirement 13, criteria 2, 3 and 6)
    For one Role (``role_mask``: the Batter_Deliveries or the Bowler_Deliveries), rank(LI) and
    rank(Stress) are the Percentile_Ranks of a Live_Ball's LI and of the Role's Stress among the
    Role's Live_Balls in the same rank group (``groups``: rank_groups with the D4 scope, the
    Innings_Group), from the strict call of ``ranks.percentile_rank_within``. With alpha the D2
    blend (``design_choices.D2_alpha``: 0.5):

    * a Live_Ball of the Role gets PI = 100 x [alpha x rank(LI) + (1 - alpha) x rank(Stress)]
      (criterion 2). With n Live_Balls in the group each rank lies in [0.5 / n, 1 - 0.5 / n], so
      0 < PI < 100;
    * a Dead_Ball of the Role gets PI 0 (criterion 3);
    * a row outside the Role gets NaN: a Wide has no batter-Role PI (criterion 11).

    Each rank is (below + 0.5 x equal) / n = m / 2n for the whole number m = 2 x below + equal.
    ``pressure_index`` recovers m exactly (rank x 2n, rounded; the float error is about 1e-10)
    and computes PI = 100 x [alpha x m_LI + (1 - alpha) x m_Stress] / 2n. With alpha 0.5, 0 or 1
    the bracket is exact, so two Live_Balls whose ranks have the same sum, and hence the same PI
    in exact arithmetic, get bit-for-bit the same PI; 0.5 x (r_LI + r_Stress) on the rounded
    ranks could differ in the last bit and split such a tie. The result differs from the formula
    on the rounded ranks by about 1e-14. PI reads only LI and Stress, which come from the pre-ball
    state and the models' Out_Of_Fold predictions (criterion 6): no ``runs``, ``extras_2_runs``,
    ``is_wicket`` or ``dismissal_type_id`` of the Delivery and no Extras_Model output enters it.

Pressure_Balls and Ordinary_Balls (``pressure_flags``, ``ordinary_balls``; criteria 3-5)
    A Live_Ball of the Role is a Pressure_Ball when the Percentile_Rank of its PI among the
    Role's Live_Balls in the same rank group is greater than 1 - D3
    (``design_choices.D3_pressure_share``: 0.25, so rank > 0.75). Equal PI have equal ranks and
    so the same flag (criterion 4); without a tie at the cut the flagged share of the n Live_Balls
    is within 1 / n of D3, and a tie at the cut moves it by at most the tie's size / n. A
    Dead_Ball of the Role is never a Pressure_Ball (criterion 3). ``pressure_flags`` returns a
    nullable boolean array with <NA> outside the Role (a Wide for the batter Role), and
    ``ordinary_balls`` its negation, <NA> kept: every Delivery of the Role that is not a
    Pressure_Ball is an Ordinary_Ball (criterion 5), so each Role Delivery is exactly one of the
    two and a Wide is neither for the batter Role (criterion 11).

Extras_Free_PI (glossary; Requirement 14, criterion 5)
    The validation-only variant of the bowler-Role PI: ``pressure_index`` and ``pressure_flags``
    with ``stress_bowl_extras_free`` (F23, F24, the boundary-only F25 and F26) in place of
    ``stress_bowl``, every other input unchanged. Its flags re-select the top D3 of the bowler
    Live_Balls in each rank group for the extras check; the Rating_Engine never reads them.

Share tables (``share_tables``; criterion 8)
    One table per Role, SHARE_COLUMNS, with one row for each of the 18 Innings_Group x Phase x
    Competition_Type cells: innings 1 then 2, Phases in Phase order (powerplay, middle, death),
    Competition_Types in ``reconstruct.COMPETITION_TYPES`` order (IPL, T20I, A_Team).
    ``n_deliveries``, ``n_live`` and ``n_pressure`` count the Role's Deliveries, Live_Balls and
    Pressure_Balls in the cell; ``pct_of_role_deliveries`` is 100 x n_pressure / n_deliveries and
    ``pct_of_innings_pressure`` 100 x n_pressure / the Role's Pressure_Balls in the Innings_Group.
    A percentage is NaN, a blank field in the CSV, when its denominator is 0. The Phase is the
    ``state`` table's (D8 splits) and the Competition_Type the ``balls`` table's; the cells are
    Innings_Groups whatever the D4 scope. ``pressure_summary`` gives the Live_Ball and
    Pressure_Ball counts of each PI and Innings_Group (SUMMARY_COLUMNS), with the lowest PI of a
    Pressure_Ball, the cut that D3 gives, and the highest PI of an Ordinary Live_Ball.

Mean LI by over (``li_by_over``; Requirement 11, criterion 13)
    LI_BY_OVER_COLUMNS, 40 rows: for each innings (1, then 2) and over number of OVERS (1-20, the
    ``overid`` of ``balls``, 1 for the first Over), the number of Deliveries, Wides and No_Balls
    included, and their mean LI (NaN for an over without a Delivery). A Delivery whose over number
    is missing or outside 1-20 is left out (none as delivered). The stage draws both innings in
    one chart, LI_BY_OVER_CHART_FILE, with the pooled mean of 1 as a reference line.

Pressure definition (``write_definition``; criterion 9)
    DEFINITION_FILE (``docs/pressure_definition.md`` in a Run) holds the plain-language
    definition (the match is live, the ball can swing the result, and the situation is harder
    than usual for the player's Role); the formulas of Live_Ball, LI, Stress (with the eight
    pillars and their components), PI and Pressure_Ball, and a note on the Extras_Free_PI; the
    D1-D5 values of the Run with their labels; and the data-derived values of the Run: the pooled
    mean unscaled LI, the Primary_Models and the counts of pressure_summary. The text holds no
    LaTeX, no ``$`` and no backslash (the Report_Check rejects them; Requirement 27, criterion 9)
    and no time stamp, so the same inputs give the same bytes.

The ``pressure`` Pipeline_Stage (``run_stage``)
    Reads ``balls`` (``reconstruct``), ``state`` (``state``), the Fold_Table (``folds``), the
    views V_1 ... V_D15 (``remaining_runs``) and the Primary_Models' predictions
    (``model_choice.load_primary`` with LI_VARIANT, which reads the ``model_choice`` checkpoint and
    the M4 table that its SOURCES_TABLE names, in ``outcome`` for LightGBM). Only the
    Primary_Models' predictions enter WP, LI and PI (Requirement 23, criterion 9). Then:

    1. checks that the predictions hold the ``state`` rows in order, with the innings of
       ``balls`` and the folds of the Fold_Table;
    2. flags the Live_Balls from ``wp_pre`` (D1) and computes the LI (``primary_leverage``);
    3. computes the Pillar_Components, Stress_Pillars and Stress, with ``rows`` the ``balls``
       rows of the ``state`` rows in order::

           fold = folds.fold_of(rows["match_id"], fold_table)
           innings, lb = rows["innings"], state["lb_bowled"]
           means = train_wicket_means(fold=fold, innings=innings, lb_bowled=lb,
                                      wickets=state["wickets"], training=common.training_mask(rows))
           components = pillar_components(
               state, own_fold_view(view_by_k, fold, state["row"]), innings=innings,
               mean_wickets=wicket_means_for(means, fold=fold, innings=innings, lb_bowled=lb),
               new_batter_cap=settings.get("design_choices.D7_new_batter_cap"),
           )
           roles = role_masks(rows["cls"])
           groups = rank_groups(innings, settings.get("design_choices.D4_scope"))
           ranks = pillars(components, live=live, roles=roles, groups=groups)
           stresses = stress(ranks, weights=settings.get("design_choices.D5_weights"))

    4. runs ``finite_guard``;
    5. computes the PI and Pressure_Ball flags of both Roles (D2, D3), the Ordinary_Balls and the
       Extras_Free_PI, and checks every rule of Requirement 13, criteria 2-5 and 11 on the result
       (ValueError if one fails);
    6. saves PRESSURE_TABLE, the SHARE_TABLES, SUMMARY_TABLE, LI_BY_OVER_TABLE and
       MISSING_COMPONENTS_TABLE to the checkpoint;
    7. writes PRESSURE_FOLDER (``outputs/pressure/``): MISSING_COMPONENTS_FILE, the SHARE_FILES,
       SUMMARY_FILE, LI_BY_OVER_FILE and LI_BY_OVER_CHART_FILE (PNG, Agg, CHART_DPI); and
       DEFINITION_FILE under the docs folder. It writes nothing else;
    8. records the data-derived Method_Constants POOLED_MEAN_LI_CONSTANT, SHARE_CONSTANT and
       PI_CUTOFF_CONSTANT (Pressure_Engine);
    9. logs the inputs, the Live_Ball and Pressure_Ball counts (with a warning when a share lies
       outside the Test_Suite band ``tests.share_band``), the PI checks, the agreement of the
       Extras_Free flags, the innings-2 death-Phase LI of Dead_Balls and Live_Balls (Requirement
       11, criterion 12; a warning when the Dead_Balls' mean is not lower), the mean LI by over,
       the share tables and the files.

Reading the ``pressure`` table (PRESSURE_TABLE, PRESSURE_COLUMNS)
    One row per Delivery of innings 1 and 2 (every ``state`` row, Super_Over_Innings excluded,
    the Deliveries of excluded matches included), in ``state`` order, keyed by ``row`` (int32);
    join it to ``balls`` and ``state`` on ``row``. The Live_Ball reference sets, the ranks and
    the Pressure_Ball shares cover every Delivery of the Innings_Group, those of excluded matches
    included, so a stage that drops the excluded matches (validation, rating) filters after the
    flags are set and keeps them as they are. Columns:

    ======================== ==================================================================
    Column                   Value
    ======================== ==================================================================
    wp_pre                   the unrounded Out_Of_Fold pre-ball WP of the M3 Primary_Model
    live                     Live_Ball (bool)
    li_raw, li               the unscaled LI and the LI (mean 1), the same for both Roles
    c19 ... c26,             the Pillar_Components (COMPONENT_COLUMNS; NaN where missing, rank
    c25b_extras_free         0.5)
    f19 ... f26,             the Stress_Pillars (PILLAR_COLUMNS), NaN outside the pillar's Role
    f25_extras_free          (F19-F22 on Wides)
    stress_bat, stress_bowl, the Stress of each Role and of the Extras_Free_PI (STRESS_COLUMNS);
    stress_bowl_extras_free  ``stress_bat`` is NaN on Wides
    pi_bat, pi_bowl          the PI of each Role: NaN outside the Role (``pi_bat`` on Wides), 0
                             on Dead_Balls, in (0, 100) on Live_Balls (float64)
    pressure_bat,            the Pressure_Ball and Ordinary_Ball flags: ``pressure_bat`` and
    ordinary_bat             ``ordinary_bat`` nullable ``boolean``, <NA> on Wides
    pressure_bowl,           the bowler Role's flags (``bool``: every Delivery is a
    ordinary_bowl            Bowler_Delivery)
    pi_bowl_extras_free,     the Extras_Free_PI and its flags (``bool``), for validation only
    pressure_bowl_extras_free
    ======================== ==================================================================

    A Pressure_Ball of a Role is always a Live_Ball, and an Ordinary_Ball is any other Delivery
    of the Role, Dead_Balls included: ``ordinary_bat`` is ``~pressure_bat``. Read the batter
    flags with ``fillna(False)`` only after restricting to Batter_Deliveries (``pressure_bat``
    is not <NA>). The summary of each PI and Innings_Group is SUMMARY_TABLE.

As delivered (PI and flags, ``--from-stage pressure`` on the checkpoints of 14 November 2026)
    The stage takes 8 s with warm file caches and about 25 s cold, and writes the same bytes on
    every Run. The guard passes. Live_Balls: 104,090 of 105,727 Deliveries in innings 1 and
    70,756 of 95,234 in innings 2; 100,067 and 68,159 of them Batter_Deliveries. Pressure_Balls
    as a share of the Role's Live_Balls: batter 25,017 (25.0002%) and 17,040 (25.0004%), bowler
    26,023 (25.0005%) and 17,688 (24.9986%), Extras_Free 26,022 and 17,689. The lowest PI of a
    Pressure_Ball (the cut) is 65.133 and 63.218 for batters, 64.647 and 66.200 for bowlers, and
    at most 2 Live_Balls share the cut PI. On Live_Balls, PI runs from 0.216 to 99.803 (batter)
    and 0.035 to 99.842 (bowler); every Dead_Ball has PI 0 and no flag, and the 7,710 Wides have
    no batter-Role PI or flag. Exact ties are common, because PI averages two ranks: about 30,000
    (batter) and 34,000 (bowler) PI values are shared by two or more Live_Balls of an
    Innings_Group, and computing PI from the rounded ranks would have split about 6,800 and 7,400
    of these groups by one bit. The Extras_Free flags agree with the bowler flags on 99.07% of the
    bowler Live_Balls, and 98.13% of the bowler Pressure_Balls keep their flag. Among the
    innings-2 death-Phase Deliveries the 10,592 Dead_Balls average an LI of 0.413 against 4.001
    for the 8,184 Live_Balls. Mean LI by over: innings 1 from about 0.64 (overs 7-12) to 0.98
    (over 20); innings 2 from 0.93 (overs 7-8) to 4.04 (over 20). Pressure_Balls as a percentage of the
    Role's Deliveries by Phase: batters 24-33% (powerplay), 17-26% (middle) and 31-39% (death) in
    innings 1, and 13-19%, 16-18% and 25-32% in innings 2; bowlers 26-36%, 19-26% and 20-23% in
    innings 1, and 12-22%, 12-22% and 18-24% in innings 2 (the ranges run over the
    Competition_Types).

As delivered (pillars, on the ``model_choice`` checkpoint of 14 November 2026)
    The 200,961 Deliveries hold 193,251 Batter_Deliveries and 200,961 Bowler_Deliveries; 174,846
    are Live_Balls (104,090 in innings 1, 70,756 in innings 2), 168,226 of them Batter_Deliveries.
    The F20(a) table has 1,200 rows (5 folds x 240 innings and L pairs, L up to 119), each with
    201 to 718 training Deliveries, so no F20(a) is missing. The only missing component is F23
    on the 941 innings-1 Deliveries before the first Legal_Ball, all of them Live_Balls, which
    therefore take F23 = 0.5. Components, ranks and Stress take 1.5-2 seconds, about 1.1 of them
    for the ranks. Every pillar averages 0.5 over each reference set, so the Live_Balls'
    Stress does too. Dead_Balls reach the ends of the scale (F19 0 on 647 and 1 on 5,555 of
    them, F23 0 on 5,854), and their Stress averages 0.651 (innings 1) and 0.553 (innings 2)
    for batters against 0.382 and 0.445 for bowlers: most are lopsided chases. Mean Stress of the
    Live_Balls by Phase (powerplay / middle / death): batter 0.499 / 0.491 / 0.518 in innings 1
    and 0.505 / 0.491 / 0.519 in innings 2; bowler 0.500 / 0.522 / 0.459 and 0.495 / 0.516 /
    0.449, where F24 falls to 0.33 and 0.28 at the death as RBQ shrinks. Across the Live_Balls
    the pillars correlate from 0.08 (F21, F22) to 0.34 (F19, F20) for batters and from 0.10
    (F24, F25) to 0.60 (F24, F26) for bowlers.

Errors: ValueError or TypeError, naming the problem, for malformed arguments, such as a missing
column, tables that do not hold the same rows in the same order, a row without a view for its
fold or a first-innings M2 fit for its ``m2_fit``, Class_Run_Values of the wrong shape, not in
CLASSES order or not finite, or a value that is infinite. ``leverage`` and ``primary_leverage``
raise ValueError for arrays of the wrong shape, no Delivery, a probability or WP that is missing,
infinite or outside [0, 1], a P row more than 1e-6 from a sum of 1, an unscaled LI of 0 on every
Delivery, a row outside innings 1 and 2 or predictions without the LI_VARIANT array, and
TypeError for an argument of the wrong kind. The pillar functions also raise ValueError for an
innings other than 1 or 2, a D4 scope, D5 weights or D7 cap outside their values, a missing flag
or rank-group label, a rank group that has a Role's Deliveries but no Live_Ball of the Role with
a value to rank against, and a ``without`` that names no Stress_Pillar or leaves a Role none.
The PI functions raise ValueError for a D1 band, D2 alpha or D3 share outside its range, and an
LI, Stress or PI of a Live_Ball of the Role that is missing or infinite; ``finite_guard`` raises
PressureInputError (StageError, exit code 5) for the problems of Requirement 13, criterion 10.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from superr import folds, leakage, views
from superr.errors import StageError
from superr.models import model_choice, remaining_runs
from superr.models.common import PROB_SUM_TOLERANCE, training_mask
from superr.models.model_choice import PrimaryPredictions
from superr.models.outcome import AVG_BOTH, BOWLER_WICKET, CLASSES, N_CLASSES, RUN_CLASS_RUNS, RUN_OUT
from superr.models.win_prob import M3_FEATURES, innings_end_par
from superr.ranks import percentile_rank_within
from superr.reconstruct import BALLS_TABLE, COMPETITION_TYPES, DeliveryClass
from superr.reconstruct import STAGE_NAME as RECONSTRUCT_STAGE
from superr.state import BALLS_PER_OVER, STATE_INNINGS, STATE_TABLE, Phase
from superr.state import STAGE_NAME as STATE_STAGE
from superr.value import TERMINAL_WICKETS
from superr.views import M2_FIT

if TYPE_CHECKING:
    from superr.models.remaining_runs import Regressor
    from superr.pipeline import StageContext
    from superr.store import CheckpointStore

__all__ = [
    "ALL_OUT_WICKETS",
    "BATTER",
    "BOWLER",
    "CHART_DPI",
    "COMPONENTS",
    "COMPONENT_COLUMNS",
    "COMPONENT_VIEW_COLUMNS",
    "D4_INNINGS_GROUP",
    "D4_POOLED",
    "D4_SCOPES",
    "D5_EQUAL",
    "DEFINITION_FILE",
    "EXTRAS_FREE_COMPONENT",
    "EXTRAS_FREE_F25",
    "FLAG_COLUMNS",
    "ID_COLUMNS",
    "LI_BY_OVER_CHART_FILE",
    "LI_BY_OVER_COLUMNS",
    "LI_BY_OVER_FILE",
    "LI_BY_OVER_TABLE",
    "LI_VARIANT",
    "M2_COLUMNS",
    "MISSING_COMPONENTS_COLUMNS",
    "MISSING_COMPONENTS_FILE",
    "MISSING_COMPONENTS_TABLE",
    "ORDINARY_COLUMNS",
    "OVERS",
    "PAR_COLUMN",
    "PILLARS",
    "PILLAR_COLUMNS",
    "PI_COLUMNS",
    "PI_CUTOFF_CONSTANT",
    "POOLED_MEAN_LI_CONSTANT",
    "POST_COLUMNS",
    "POST_INNINGS",
    "PRESSURE_COLUMNS",
    "PRESSURE_FLAG_COLUMNS",
    "PRESSURE_FOLDER",
    "PRESSURE_TABLE",
    "RATE_COLUMNS",
    "ROLES",
    "SHARE_COLUMNS",
    "SHARE_CONSTANT",
    "SHARE_FILES",
    "SHARE_TABLES",
    "STAGE_NAME",
    "STATE_COLUMNS",
    "STRESS_COLUMNS",
    "TSG_COLUMN",
    "WICKET_CLASSES",
    "WICKET_MEANS_COLUMNS",
    "WINDOW_INPUT_COLUMNS",
    "Component",
    "LeverageIndex",
    "Pillar",
    "PostStates",
    "PressureInputError",
    "finite_guard",
    "first_innings_models",
    "leverage",
    "li_by_over",
    "live_balls",
    "missing_components",
    "ordinary_balls",
    "own_fold_view",
    "pillar_components",
    "pillars",
    "post_outcome_states",
    "pressure_flags",
    "pressure_index",
    "primary_leverage",
    "rank_groups",
    "role_masks",
    "run_stage",
    "share_tables",
    "stress",
    "train_wicket_means",
    "wicket_means_for",
    "write_definition",
    "write_missing_components",
]


# ------------------------------------------------------------------------------------- constants

#: The innings whose Deliveries get Post_Outcome_States: those with an M3 classifier.
POST_INNINGS: tuple[int, ...] = STATE_INNINGS

#: The classes whose Post_Outcome_State takes the Post_Wicket_State changes (criterion 15, step 3).
WICKET_CLASSES: tuple[int, ...] = (BOWLER_WICKET, RUN_OUT)

#: Wickets fallen that end an innings (all out), fixed by the Laws of Cricket: one more than
#: ``value.TERMINAL_WICKETS``, the wickets at which the next dismissal ends the innings.
ALL_OUT_WICKETS = TERMINAL_WICKETS + 1

#: The columns of a PostStates frame (module docstring, "Layout"), by group.
ID_COLUMNS: tuple[str, ...] = ("row", "match_id", "fold", "innings", "cls")
STATE_COLUMNS: tuple[str, ...] = ("lb_bowled", "lb_left", "score", "wickets", "runs_needed", "target", "sched_balls")
M2_COLUMNS: tuple[str, ...] = tuple(
    name for name in remaining_runs.M2_FEATURES if name not in STATE_COLUMNS
)  # rbq, theta_striker, striker_lb_faced, theta_non_striker, scoring_env, venue_effect, match_day
TSG_COLUMN = "tsg"
PAR_COLUMN = "par_score"
RATE_COLUMNS: tuple[str, ...] = ("rrr", "ach_rate", "rr_minus_ar")
FLAG_COLUMNS: tuple[str, ...] = ("innings_over", "terminal")
POST_COLUMNS: Mapping[int, tuple[str, ...]] = MappingProxyType(
    {
        1: (*ID_COLUMNS, *STATE_COLUMNS, *M2_COLUMNS, TSG_COLUMN, PAR_COLUMN, *FLAG_COLUMNS),
        2: (*ID_COLUMNS, *STATE_COLUMNS, *M2_COLUMNS, TSG_COLUMN, *RATE_COLUMNS, *FLAG_COLUMNS),
    }
)

#: Whether each class, in CLASSES order, is one of WICKET_CLASSES.
_WICKET_MASK = np.isin(np.arange(N_CLASSES), WICKET_CLASSES)
_WICKET_MASK.flags.writeable = False

#: The columns that post_outcome_states reads from each input.
_STATE_INPUT_COLUMNS: tuple[str, ...] = (
    "row", "lb_bowled", "lb_left", "score", "wickets", "runs_needed", "striker_lb_faced",
)  # fmt: skip
_BALL_INPUT_COLUMNS: tuple[str, ...] = ("row", "match_id", "innings", "sched_balls", "target")
_PAR_NEXT_COLUMN = "par_score_next"
_VIEW_INPUT_COLUMNS: tuple[str, ...] = (
    *remaining_runs.VIEW_FEATURES, *remaining_runs.POST_WICKET_VIEW_COLUMNS, TSG_COLUMN, _PAR_NEXT_COLUMN,
)  # fmt: skip
_COUNT_COLUMNS: tuple[str, ...] = ("lb_left", "wickets", "striker_lb_faced")  # whole-number M2 features
_FIRST_INNINGS = 1
_CHASE = 2
_SHOWN = 5  # examples named in one error message
_PAR_CHECK_TOLERANCE = 1e-9  # Par_Score(SB) against the view's par_score_next at the last Legal_Ball, in runs

#: The M4 prediction variant whose probabilities P weight the Leverage_Index: theta_bat and
#: theta_bowl both at the fold's Average_Skill (Requirement 11, criterion 9).
LI_VARIANT = AVG_BOTH

_LI_MEAN_TOLERANCE = 1e-6  # the mean LI over the Deliveries of innings 1 and 2 is 1 within this (criterion 10)


# ---------------------------------------------------------------------------------- small helpers

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


def _ints(values: Any, what: str, n_rows: int | None = None, *, minimum: int | None = None) -> np.ndarray:
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
    _length(array, what, n_rows)
    if minimum is not None and len(array) and int(array.min()) < minimum:
        raise ValueError(f"{what} must be {minimum} or more; {int((array < minimum).sum()):,} value(s) are below")
    return array


def _innings(innings: Any) -> int:
    if isinstance(innings, (bool, np.bool_)) or not isinstance(innings, Integral) or int(innings) not in POST_INNINGS:
        raise ValueError(f"innings must be one of {', '.join(map(str, POST_INNINGS))}; got {innings!r}")
    return int(innings)


def _read_only(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


# ===================================================================================== PostStates

@dataclass(frozen=True, eq=False)
class PostStates:
    """The Post_Outcome_States of every innings 1-2 Delivery (module docstring); build them with post_outcome_states.

    ``n_rows`` is the number of Deliveries, the rows of the ``state`` table. For each innings i of
    POST_INNINGS, ``positions[i]`` holds the positions in ``state`` order of the innings'
    Deliveries (int64, ascending, read-only) and ``frames[i]`` their stacked states with the
    POST_COLUMNS[i]: N_CLASSES rows per Delivery, state s being class s % 9 of the Delivery at
    ``positions[i][s // 9]`` (module docstring, "Layout"). ``n_m2_predictions`` counts the M2
    predictions made for Achievable_Rate. The frames are shared, not copied: treat them as
    read-only. The methods below give each candidate the same inputs.
    """

    n_rows: int
    positions: Mapping[int, np.ndarray]
    frames: Mapping[int, pd.DataFrame]
    n_m2_predictions: int

    def __post_init__(self) -> None:
        if sorted(self.frames) != list(POST_INNINGS) or sorted(self.positions) != list(POST_INNINGS):
            raise ValueError(f"PostStates needs the frames and positions of innings {list(POST_INNINGS)}")
        positions: dict[int, np.ndarray] = {}
        for number in POST_INNINGS:
            at = np.array(self.positions[number], dtype=np.int64)
            frame = self.frames[number]
            if not isinstance(frame, pd.DataFrame) or tuple(frame.columns) != POST_COLUMNS[number]:
                raise ValueError(f"PostStates.frames[{number}] must be a DataFrame with the POST_COLUMNS[{number}]")
            if at.ndim != 1 or len(frame) != N_CLASSES * len(at):
                raise ValueError(
                    f"PostStates.frames[{number}] must hold {N_CLASSES} states per Delivery of positions[{number}]"
                )
            if len(at) and (at.min() < 0 or at.max() >= self.n_rows or (np.diff(at) <= 0).any()):
                raise ValueError(f"PostStates.positions[{number}] must be ascending positions below n_rows")
            positions[number] = _read_only(at)
        object.__setattr__(self, "positions", MappingProxyType(positions))
        object.__setattr__(self, "frames", MappingProxyType({number: self.frames[number] for number in POST_INNINGS}))

    def __repr__(self) -> str:
        counts = ", ".join(f"innings {number}: {self.n_states(number):,}" for number in POST_INNINGS)
        return f"PostStates({self.n_rows:,} Deliveries; states {counts}; {self.n_m2_predictions:,} M2 predictions)"

    def n_states(self, innings: int | None = None) -> int:
        """The Post_Outcome_States of ``innings``, or of both innings when None."""
        if innings is None:
            return sum(len(frame) for frame in self.frames.values())
        return len(self.frames[_innings(innings)])

    def frame(self, innings: int) -> pd.DataFrame:
        """The stacked states of ``innings`` (POST_COLUMNS[innings]); shared, not copied."""
        return self.frames[_innings(innings)]

    def features(self, innings: int) -> pd.DataFrame:
        """M3_FEATURES[innings] of every state of the innings, as a new float64 DataFrame in that column order."""
        number = _innings(innings)
        return self.frames[number][list(M3_FEATURES[number])].astype(np.float64)

    def fold(self, innings: int) -> np.ndarray:
        """The fold of each state of ``innings`` (int64, a copy): the fold whose calibrated model predicts it."""
        return self.frames[_innings(innings)]["fold"].to_numpy(dtype=np.int64, copy=True)

    def terminal(self, innings: int) -> np.ndarray:
        """True for each state of ``innings`` whose WP comes from the terminal rules (bool, a copy); none in innings 1."""
        return self.frames[_innings(innings)]["terminal"].to_numpy(dtype=bool, copy=True)

    def innings_over(self, innings: int) -> np.ndarray:
        """True for each state of ``innings`` whose outcome ends the innings (bool, a copy).

        In innings 1 these are the end-of-innings states that the innings-1 classifier scores
        (Requirement 10, criterion 17); in innings 2 they are terminal.
        """
        return self.frames[_innings(innings)]["innings_over"].to_numpy(dtype=bool, copy=True)

    def by_delivery(self, innings: int, values: Any) -> np.ndarray:
        """``values``, one per state of ``innings`` in frame order, as a float64 (Deliveries, N_CLASSES) matrix."""
        number = _innings(innings)
        return _floats(values, f"values of innings {number}", self.n_states(number)).reshape(-1, N_CLASSES)

    def matrix(self, values: Mapping[int, Any]) -> np.ndarray:
        """The per-state ``values`` of both innings ({innings: array}) as a (n_rows, N_CLASSES) matrix in ``state`` order.

        Row r holds the values of the 9 states of the Delivery at position r of the ``state``
        table, in CLASSES order: for WPs, the columns ``wp_post_0`` ... ``wp_post_8``. Raises
        ValueError when an innings is absent or its array has the wrong length.
        """
        if not isinstance(values, Mapping):
            raise TypeError(f"values must map each innings to its per-state array; got {type(values).__name__}")
        out = np.full((self.n_rows, N_CLASSES), np.nan)
        for number in POST_INNINGS:
            if number not in values:
                raise ValueError(f"values has no array for innings {number}")
            out[self.positions[number]] = self.by_delivery(number, values[number])
        return out


# ================================================================================ building states

def first_innings_models(store: CheckpointStore, n_folds: int) -> dict[str, remaining_runs.M2Model]:
    """The outer first-innings M2 fits of folds 1 to ``n_folds`` (D15), keyed by fit id, from the ``remaining_runs`` checkpoint.

    The fit ids are those that the fold-k rows of view V_k record in ``m2_fit``
    (``m2_first_lgbm_outer_k1`` ...), so the result is the ``m2_first`` argument of
    post_outcome_states: the fits behind Achievable_Rate in innings 2 and Par_Score at
    Scheduled_Balls in innings 1. Each is a calibrated ``remaining_runs.M2Model`` (its Booster with
    its Legal_Balls-left calibration; Requirement 9, criterion 15), from
    ``remaining_runs.load_models``, so these states use the same predictions as the views. Raises
    the errors of ``folds.sandbox_plan`` and ``remaining_runs.load_models``.
    """
    plan = folds.sandbox_plan(n_folds)
    fit_ids = [
        leakage.fit_id_for(remaining_runs.FIRST_INNINGS_MODEL, spec, family=remaining_runs.FAMILY)
        for spec in plan
        if spec.kind == folds.OUTER
    ]
    return remaining_runs.load_models(store, fit_ids)


@dataclass(frozen=True, eq=False)
class _RowValues:
    """One value per ``state`` row of everything the stacked states take from the inputs."""

    row: np.ndarray
    match_id: np.ndarray
    fold: np.ndarray
    innings: np.ndarray
    sched_balls: np.ndarray
    target: np.ndarray
    lb_bowled: np.ndarray
    score: np.ndarray
    wickets: np.ndarray  # pre-ball wickets fallen
    runs_needed: np.ndarray
    class_values: np.ndarray  # (rows, N_CLASSES)
    tsg: np.ndarray
    par_next: np.ndarray
    par_end: np.ndarray  # Par_Score at Scheduled_Balls; set on innings-1 rows with an innings-end state, else NaN
    m2_fit: np.ndarray  # object: the first-innings M2 fit id of each row
    m2_other: pd.DataFrame  # M2 features after a non-wicket class (lb_left already after the ball)
    m2_wicket: pd.DataFrame  # M2 features after a wicket class


def post_outcome_states(
    state: pd.DataFrame,
    balls: pd.DataFrame,
    view_by_k: Mapping[int, pd.DataFrame],
    row_fold: Any,
    crv: Any,
    m2_first: Mapping[str, Regressor],
    *,
    par_means: pd.DataFrame,
    par_bias: pd.DataFrame,
) -> PostStates:
    """The Post_Outcome_States of every Delivery of innings 1 and 2 (Requirement 11, criterion 15; module docstring).

    ``state`` is the ``state`` table (``row``, ``lb_bowled``, ``lb_left``, ``score``,
    ``wickets``, ``runs_needed``, ``striker_lb_faced``), whose rows and order the result
    follows. ``balls`` is the ``balls`` table of the ``reconstruct`` checkpoint (``row``,
    ``match_id``, ``innings``, ``sched_balls``, ``target``), holding every ``state`` row.
    ``view_by_k`` maps each fold k to view V_k of the ``remaining_runs`` checkpoint (the M2
    VIEW_FEATURES, ``theta_next_batter``, ``rbq_post_wicket``, ``tsg``, ``par_score_next``,
    ``m2_fit`` and ``row``, with the ``state`` rows in ``state`` order). ``row_fold`` is the fold
    of each ``state`` row (``folds.fold_of``); a row of fold k takes its view values from V_k.
    ``crv`` gives the Class_Run_Values of each row's fold in CLASSES order: an array of shape
    (rows, N_CLASSES) (``OutcomeFoldModels.class_run_values_for(row_fold)``), or a mapping from
    each fold k to its N_CLASSES values. ``m2_first`` maps the fit id of every first-innings M2
    fit that an innings-2 row, or an innings-1 row with an innings-end state, names in ``m2_fit``
    to that fit (``first_innings_models``). ``par_means`` and ``par_bias`` are the
    ``remaining_runs`` checkpoint tables ``m2_par_means`` and ``m2_par_bias``, which give the
    Par_Score at Scheduled_Balls of the end-of-innings states (``win_prob.innings_end_par``).

    Returns the PostStates. Raises ValueError when a column is absent, the tables hold different
    rows, a row lies outside innings 1 and 2 or has no view for its fold, a view row has no
    ``m2_fit`` or names a fit absent from ``m2_first``, the Class_Run_Values have the wrong shape,
    are missing, negative or not in CLASSES order, a value is infinite, or the Par_Score at
    Scheduled_Balls of a last Legal_Ball differs from its view's ``par_score_next`` (par tables of
    another Run); TypeError for an argument of the wrong kind; and the errors of the M2
    predictions and of innings_end_par.
    """
    _require(state, _STATE_INPUT_COLUMNS, "state")
    _require(balls, _BALL_INPUT_COLUMNS, "balls")
    if not isinstance(m2_first, Mapping):
        raise TypeError(f"m2_first must map first-innings M2 fit ids to the fits; got {type(m2_first).__name__}")
    row_ids = _ints(state["row"], "state column 'row'")
    n_rows = len(row_ids)
    deliveries = _delivery_rows(balls, row_ids)
    fold = _ints(row_fold, "row_fold", n_rows)
    served, m2_fit = _served_view(view_by_k, fold, row_ids)

    # The M2 features after the ball (criterion 15, steps 1 and 3), as remaining_runs builds them.
    lb_left_after = np.maximum(_ints(state["lb_left"], "state column 'lb_left'", minimum=0) - 1, 0)
    m2_other = remaining_runs.m2_features(served, state)
    m2_wicket = remaining_runs.post_wicket_features(served, state)
    for frame in (m2_other, m2_wicket):
        frame["lb_left"] = lb_left_after.astype(np.float64)

    innings = _ints(deliveries["innings"], "balls column 'innings'")
    sched_balls = _ints(deliveries["sched_balls"], "balls column 'sched_balls'", minimum=1)
    lb_bowled = _ints(state["lb_bowled"], "state column 'lb_bowled'", minimum=0)
    par_next = served[_PAR_NEXT_COLUMN].to_numpy(dtype=np.float64)

    # Innings-1 Deliveries with an outcome that ends the innings: the last Legal_Ball, or a wicket
    # class that leaves ALL_OUT_WICKETS fallen. Their end-of-innings states need Par_Score at SB.
    ends = np.flatnonzero(
        (innings == _FIRST_INNINGS)
        & (
            (lb_bowled + 1 >= sched_balls)
            | (m2_wicket["wickets"].to_numpy() >= ALL_OUT_WICKETS)
            | (m2_other["wickets"].to_numpy() >= ALL_OUT_WICKETS)
        )
    )
    chase = np.flatnonzero(innings == _CHASE)
    absent = sorted({str(fit_id) for fit_id in m2_fit[np.concatenate([chase, ends])].tolist()} - set(m2_first))
    if absent:
        raise ValueError(
            f"m2_first lacks the first-innings M2 fit(s) {', '.join(absent[:_SHOWN])} that innings-2 rows or "
            f"innings-1 rows with an innings-end state name in {M2_FIT}; pass first_innings_models(store, D15)"
        )
    par_end = np.full(n_rows, np.nan)
    if len(ends):
        par_end[ends] = innings_end_par(
            m2_fit[ends], served["scoring_env"].to_numpy()[ends], served["venue_effect"].to_numpy()[ends],
            sched_balls[ends], m2_models=m2_first, par_means=par_means, par_bias=par_bias,
        )  # fmt: skip
        last = ends[lb_bowled[ends] + 1 == sched_balls[ends]]  # both read the corrected curve at SB
        off = ~np.isclose(par_end[last], par_next[last], rtol=0.0, atol=_PAR_CHECK_TOLERANCE)
        if off.any():
            raise ValueError(
                f"Par_Score at Scheduled_Balls differs from the view's {_PAR_NEXT_COLUMN} at the last Legal_Ball of "
                f"{_rows(int(off.sum()))}; pass the par means, par bias and first-innings M2 fits of the "
                "remaining_runs checkpoint whose views these are"
            )

    values = _RowValues(
        row=row_ids,
        match_id=_ints(deliveries["match_id"], "balls column 'match_id'"),
        fold=fold,
        innings=innings,
        sched_balls=sched_balls,
        target=_floats(deliveries["target"], "balls column 'target'"),
        lb_bowled=lb_bowled,
        score=_ints(state["score"], "state column 'score'").astype(np.float64),
        wickets=_ints(state["wickets"], "state column 'wickets'", minimum=0),
        runs_needed=_floats(state["runs_needed"], "state column 'runs_needed'"),
        class_values=_class_values(crv, fold, n_rows),
        tsg=served[TSG_COLUMN].to_numpy(dtype=np.float64),
        par_next=par_next,
        par_end=par_end,
        m2_fit=m2_fit,
        m2_other=m2_other,
        m2_wicket=m2_wicket,
    )

    frames: dict[int, pd.DataFrame] = {}
    positions: dict[int, np.ndarray] = {}
    n_predictions = 0
    for number in POST_INNINGS:
        at = np.flatnonzero(values.innings == number)
        frames[number], used = _innings_states(number, at, values, m2_first)
        positions[number] = at
        n_predictions += used
    return PostStates(n_rows=n_rows, positions=positions, frames=frames, n_m2_predictions=n_predictions)


def _delivery_rows(balls: pd.DataFrame, row_ids: np.ndarray) -> pd.DataFrame:
    """The _BALL_INPUT_COLUMNS of ``balls`` for each ``state`` row, in ``state`` order, on a RangeIndex."""
    index = pd.Index(_ints(balls["row"], "balls column 'row'"))
    if not index.is_unique:
        raise ValueError("balls must hold each row once")
    at = index.get_indexer(row_ids)
    if (at < 0).any():
        raise ValueError(f"{_rows(int((at < 0).sum()))} of state have no Delivery in balls; pass the tables of one Run")
    rows = balls.iloc[at][list(_BALL_INPUT_COLUMNS)].reset_index(drop=True)
    outside = ~np.isin(_ints(rows["innings"], "balls column 'innings'"), POST_INNINGS)
    if outside.any():
        raise ValueError(f"{_rows(int(outside.sum()))} of state lie outside innings 1 and 2")
    return rows


def _served_view(view_by_k: Any, fold: np.ndarray, row_ids: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Each row's _VIEW_INPUT_COLUMNS from view V_k of its fold k, with ``row``, and its ``m2_fit`` (object array)."""
    if not isinstance(view_by_k, Mapping):
        raise TypeError(f"view_by_k must map each fold k to view V_k; got {type(view_by_k).__name__}")
    n_rows = len(row_ids)
    ks = np.unique(fold).tolist()
    absent = [k for k in ks if k not in view_by_k]
    if absent:
        raise ValueError(f"view_by_k has no view for fold(s) {', '.join(map(str, absent))}, which rows belong to")
    columns = {name: np.full(n_rows, np.nan) for name in _VIEW_INPUT_COLUMNS}
    fit_ids = np.full(n_rows, None, dtype=object)
    for k in ks:
        view = view_by_k[k]
        what = f"view V_{k}"
        _require(view, ("row", *_VIEW_INPUT_COLUMNS, M2_FIT), what)
        if len(view) != n_rows or not np.array_equal(_ints(view["row"], f"{what} column 'row'"), row_ids):
            raise ValueError(
                f"{what} does not hold the rows of the state table in its order; pass the views of the "
                "remaining_runs checkpoint of the same Run"
            )
        at = np.flatnonzero(fold == k)
        for name in _VIEW_INPUT_COLUMNS:
            columns[name][at] = _floats(view[name], f"{what} column {name!r}")[at]
        named = view[M2_FIT].iloc[at]
        missing = named.isna().to_numpy(dtype=bool) | (named.astype(str).str.len() == 0).to_numpy(dtype=bool)
        if missing.any():
            raise ValueError(f"{_rows(int(missing.sum()))} of fold {k} in {what} record no {M2_FIT}")
        fit_ids[at] = named.astype(str).to_numpy(dtype=object)
    served = pd.DataFrame({"row": row_ids, **columns}, columns=["row", *_VIEW_INPUT_COLUMNS])
    return served, fit_ids


def _class_values(crv: Any, fold: np.ndarray, n_rows: int) -> np.ndarray:
    """The Class_Run_Values of each row as a float64 (rows, N_CLASSES) array, checked (post_outcome_states)."""
    if isinstance(crv, Mapping):
        out = np.empty((n_rows, N_CLASSES), dtype=np.float64)
        for k in np.unique(fold).tolist():
            if k not in crv:
                raise ValueError(f"crv has no Class_Run_Values for fold {k}")
            fold_values = np.asarray(crv[k], dtype=np.float64)
            if fold_values.shape != (N_CLASSES,):
                raise ValueError(
                    f"crv[{k}] must hold {N_CLASSES} Class_Run_Values in CLASSES order; got shape {fold_values.shape}"
                )
            out[fold == k] = fold_values
    elif crv is None or isinstance(crv, (str, bytes, pd.DataFrame)):
        raise TypeError(
            "crv must be an array of shape (rows, N_CLASSES) or a mapping from fold to N_CLASSES values; "
            f"got {type(crv).__name__}"
        )
    else:
        out = np.array(crv, dtype=np.float64)
        if out.shape != (n_rows, N_CLASSES):
            raise ValueError(f"crv must have shape ({n_rows:,}, {N_CLASSES}), one row per Delivery; got {out.shape}")
    bad = ~np.isfinite(out) | (out < 0.0)
    if bad.any():
        raise ValueError(f"crv holds {int(bad.sum()):,} value(s) that are missing, infinite or negative")
    for cls, runs in RUN_CLASS_RUNS.items():
        if (out[:, cls] != runs).any():
            raise ValueError(
                f"the Class_Run_Value of {CLASSES[cls]!r} must be exactly {runs}; pass the values in CLASSES order "
                f"({', '.join(CLASSES)})"
            )
    return out


def _innings_states(
    number: int, at: np.ndarray, values: _RowValues, m2_first: Mapping[str, Regressor]
) -> tuple[pd.DataFrame, int]:
    """The stacked frame of innings ``number`` (Deliveries at positions ``at``) and the M2 predictions it made."""
    n_states = N_CLASSES * len(at)
    rep = np.repeat(at, N_CLASSES)  # the Delivery of each state
    wicket = np.tile(_WICKET_MASK, len(at))
    run_value = values.class_values[at].reshape(-1)  # row-major, like rep
    m2 = {
        name: np.where(wicket, values.m2_wicket[name].to_numpy()[rep], values.m2_other[name].to_numpy()[rep])
        for name in remaining_runs.M2_FEATURES
    }
    counts = {name: m2[name].astype(np.int32) for name in _COUNT_COLUMNS}  # whole numbers, never missing
    lb_bowled = values.lb_bowled[rep] + 1
    sched = values.sched_balls[rep]
    innings_over = (lb_bowled >= sched) | (counts["wickets"] >= ALL_OUT_WICKETS)
    data: dict[str, np.ndarray] = {
        "row": values.row[rep].astype(np.int32),
        "match_id": values.match_id[rep].astype(np.int32),
        "fold": values.fold[rep].astype(np.int32),
        "innings": np.full(n_states, number, dtype=np.int8),
        "cls": np.tile(np.arange(N_CLASSES, dtype=np.int8), len(at)),
        "lb_bowled": lb_bowled.astype(np.int32),
        "lb_left": counts["lb_left"],
        "score": values.score[rep] + run_value,
        "wickets": counts["wickets"],
        "sched_balls": sched.astype(np.int32),
        **{name: counts[name] if name in _COUNT_COLUMNS else m2[name] for name in M2_COLUMNS},
        TSG_COLUMN: values.tsg[rep],
        "innings_over": innings_over,
    }
    n_predictions = 0
    if number == _FIRST_INNINGS:
        # An outcome that ends the innings takes the end-of-innings state (Requirement 10, criterion 17):
        # Scheduled_Balls bowled, the pre-ball wickets and Par_Score at Scheduled_Balls; the score keeps
        # the Class_Run_Value. The innings-1 classifier scores every state, so none is terminal.
        data["lb_bowled"] = np.where(innings_over, sched, lb_bowled).astype(np.int32)
        data["lb_left"] = np.where(innings_over, 0, counts["lb_left"]).astype(np.int32)
        data["wickets"] = np.where(innings_over, values.wickets[rep], counts["wickets"]).astype(np.int32)
        data["runs_needed"] = np.full(n_states, np.nan)
        data["target"] = np.full(n_states, np.nan)
        data[PAR_COLUMN] = np.where(innings_over, values.par_end[rep], values.par_next[rep])  # Par_Score at SB or L + 1
        data["terminal"] = np.zeros(n_states, dtype=bool)
        unset = innings_over & np.isnan(data[PAR_COLUMN])
        if unset.any():
            raise ValueError(
                f"{int(unset.sum()):,} innings-end state(s) of innings 1 have no Par_Score at Scheduled_Balls"
            )
    else:
        needed = values.runs_needed[rep] - run_value
        lb_left = counts["lb_left"]
        has_balls = lb_left > 0
        rrr = np.full(n_states, np.nan)
        rrr[has_balls] = BALLS_PER_OVER * needed[has_balls] / lb_left[has_balls]
        other, after_wicket, n_predictions = _achievable_rates(at, values, m2_first)
        local = np.repeat(np.arange(len(at)), N_CLASSES)
        ach = np.where(wicket, after_wicket[local], other[local])
        data["runs_needed"] = needed
        data["target"] = values.target[rep]
        data["rrr"] = rrr
        data["ach_rate"] = ach
        data["rr_minus_ar"] = rrr - ach
        data["terminal"] = innings_over | (np.nan_to_num(needed, nan=1.0) <= 0.0)  # score >= Target: won
    frame = pd.DataFrame(data, columns=list(POST_COLUMNS[number]))
    for name in frame.columns:
        column = frame[name].to_numpy()
        if column.dtype.kind == "f" and np.isinf(column).any():
            raise ValueError(
                f"the innings-{number} Post_Outcome_States hold {int(np.isinf(column).sum()):,} infinite value(s) in "
                f"{name!r}; a state value is a number or missing"
            )
    return frame, n_predictions


def _achievable_rates(
    at: np.ndarray, values: _RowValues, m2_first: Mapping[str, Regressor]
) -> tuple[np.ndarray, np.ndarray, int]:
    """Achievable_Rate after a non-wicket class and after a wicket class, per Delivery at ``at``, and the predictions made.

    One prediction of each kind per Delivery with a Legal_Ball left after the ball, from the
    first-innings M2 fit that the Delivery's view row names; NaN where none is left.
    """
    other = np.full(len(at), np.nan)
    after_wicket = np.full(len(at), np.nan)
    has_balls = values.m2_other["lb_left"].to_numpy()[at] > 0
    fit_ids = values.m2_fit[at]
    n_predictions = 0
    for fit_id in sorted({str(name) for name in fit_ids[has_balls].tolist()}):
        mine = np.flatnonzero(has_balls & (fit_ids == fit_id))
        rows = at[mine]
        X = pd.concat([values.m2_other.iloc[rows], values.m2_wicket.iloc[rows]], ignore_index=True)
        rate = remaining_runs.achievable_rate(m2_first[fit_id], X)
        other[mine], after_wicket[mine] = rate[: len(rows)], rate[len(rows) :]
        n_predictions += len(X)
    return other, after_wicket, n_predictions


# ================================================================================ Leverage_Index

@dataclass(frozen=True, eq=False)
class LeverageIndex:
    """The Leverage_Index of every Delivery of innings 1 and 2 (module docstring, "Leverage_Index"); build it with leverage.

    ``unscaled``: each Delivery's unscaled LI, the sum over the 9 classes of P(class) x
    |WP(Post_Outcome_State of the class) - WP(pre-ball state)| (Requirement 11, criterion 9).
    ``pooled_mean``: the mean of ``unscaled`` over all the Deliveries, the one positive number that
    scales every LI (criterion 10). ``li``: ``unscaled / pooled_mean``, the LI, whose mean is 1.
    Both arrays are float64 copies, one value per Delivery in the order of the inputs,
    non-negative and read-only. A Delivery has one LI, which its batter Role and its bowler Role
    share (criterion 11).
    """

    li: np.ndarray
    unscaled: np.ndarray
    pooled_mean: float

    def __post_init__(self) -> None:
        li = np.array(self.li, dtype=np.float64)
        unscaled = np.array(self.unscaled, dtype=np.float64)
        if li.ndim != 1 or unscaled.shape != li.shape:
            raise ValueError("LeverageIndex.li and LeverageIndex.unscaled must be 1-D, with one value per Delivery")
        pooled_mean = float(self.pooled_mean)
        if not (np.isfinite(pooled_mean) and pooled_mean > 0.0):
            raise ValueError(f"LeverageIndex.pooled_mean must be a positive number; got {self.pooled_mean!r}")
        object.__setattr__(self, "li", _read_only(li))
        object.__setattr__(self, "unscaled", _read_only(unscaled))
        object.__setattr__(self, "pooled_mean", pooled_mean)

    def __repr__(self) -> str:
        return f"LeverageIndex({len(self.li):,} Deliveries; pooled mean unscaled LI {self.pooled_mean:.6f})"


def _unit_array(values: Any, what: str, n_columns: int | None, n_rows: int | None) -> np.ndarray:
    """``values`` as a new float64 array, (rows,) or (rows, n_columns), every value finite and within [0, 1]."""
    if values is None or isinstance(values, (str, bytes, Mapping)):
        raise TypeError(f"{what} must be an array of numbers; got {type(values).__name__}")
    array = np.asarray(values)
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{what} must hold numbers; got dtype {array.dtype}")
    if n_columns is None:
        if array.ndim != 1:
            raise ValueError(f"{what} must be 1-D, one value per Delivery; got shape {array.shape}")
    elif array.ndim != 2 or array.shape[1] != n_columns:
        raise ValueError(f"{what} must have shape (Deliveries, {n_columns}), classes in CLASSES order; got {array.shape}")
    if n_rows is not None and len(array) != n_rows:
        raise ValueError(f"{what} has {len(array):,} rows, but {n_rows:,} are needed, one per Delivery")
    array = array.astype(np.float64)  # a copy
    bad = int((~np.isfinite(array)).sum())
    if bad:
        raise ValueError(f"{what} holds {bad:,} missing or infinite value(s)")
    outside = int(((array < 0.0) | (array > 1.0)).sum())
    if outside:
        raise ValueError(f"{what} holds {outside:,} value(s) outside [0, 1]")
    return array


def leverage(p_avg: Any, wp_pre: Any, wp_post: Any) -> LeverageIndex:
    """The Leverage_Index of Deliveries from their M4 probabilities and WPs (Requirement 11, criteria 9-11; module docstring).

    ``p_avg``: (n, N_CLASSES) class probabilities in CLASSES order from the LI_VARIANT
    (``avg_both``) M4 prediction, every row summing to 1 within PROB_SUM_TOLERANCE (1e-6); they
    are used as given, not rescaled. ``wp_pre``: (n,) the pre-ball WP. ``wp_post``: (n, N_CLASSES)
    the WP of each class's Post_Outcome_State, in CLASSES order, after the terminal rules. All
    finite and within [0, 1]. The n >= 1 rows must be all the Deliveries of innings 1 and 2 in all
    matches: the pooled mean of criterion 10 is the mean over the rows given.

    Returns the LeverageIndex: ``unscaled`` = sum over the classes c of P_c x |WP_post,c - WP_pre|,
    within [0, 1]; ``pooled_mean``, its mean; ``li`` = unscaled / pooled_mean, with mean 1. One
    value per row, which both Roles of the Delivery use. Raises TypeError for an argument that is
    not an array of numbers, and ValueError for a wrong shape, no row, a missing or infinite value,
    a value outside [0, 1], a P row whose sum is more than 1e-6 from 1, or an unscaled LI of 0 on
    every row (no positive mean to scale by).
    """
    p = _unit_array(p_avg, "p_avg", N_CLASSES, None)
    n_rows = len(p)
    if not n_rows:
        raise ValueError("p_avg holds no Delivery; the pooled mean of Requirement 11, criterion 10 needs at least one")
    pre = _unit_array(wp_pre, "wp_pre", None, n_rows)
    post = _unit_array(wp_post, "wp_post", N_CLASSES, n_rows)
    deviation = np.abs(p.sum(axis=1) - 1.0)
    off = int((deviation > PROB_SUM_TOLERANCE).sum())
    if off:
        raise ValueError(
            f"{off:,} of the {n_rows:,} rows of p_avg differ from a sum of 1 by more than {PROB_SUM_TOLERANCE:g} "
            f"(largest difference {float(deviation.max()):.3e}); pass the renormalised M4 probabilities "
            "(Requirement 11, criterion 5)"
        )
    unscaled = (p * np.abs(post - pre[:, None])).sum(axis=1)
    pooled_mean = float(unscaled.mean())
    if not pooled_mean > 0.0:
        raise ValueError(
            "the unscaled LI is 0 on every Delivery, so it has no positive pooled mean to scale by "
            "(Requirement 11, criterion 10)"
        )
    li = unscaled / pooled_mean
    mean_li = float(li.mean())
    if not (np.isfinite(li).all() and (li >= 0.0).all() and abs(mean_li - 1.0) <= _LI_MEAN_TOLERANCE):
        raise ValueError(
            f"the scaled LI breaks Requirement 11, criterion 10: mean {mean_li!r}, not 1 within {_LI_MEAN_TOLERANCE:g}, "
            "or a value that is negative or not finite"
        )
    return LeverageIndex(li=li, unscaled=unscaled, pooled_mean=pooled_mean)


def primary_leverage(primary: PrimaryPredictions) -> LeverageIndex:
    """The Leverage_Index of every Delivery from the Primary_Models' predictions (Requirement 11, criteria 9-11).

    ``primary`` is the PrimaryPredictions of ``model_choice.load_primary(store, variants=...)``
    with LI_VARIANT among the variants: every row of the ``state`` table, so every Delivery of
    innings 1 and 2 in all matches. P is ``primary.m4[LI_VARIANT]``, the M4 Primary_Model with
    Average_Skill for both players; WP_pre and WP_post are ``primary.wp_pre`` and
    ``primary.wp_post``, from the M3-1 Primary_Model on innings-1 rows and the M3-2 Primary_Model
    on innings-2 rows. Returns ``leverage`` of these arrays, in ``primary.row`` order (``state``
    order). Raises TypeError for an argument that is not a PrimaryPredictions; ValueError when it
    lacks the LI_VARIANT array or holds a row outside innings 1 and 2; and the errors of leverage.
    """
    if not isinstance(primary, PrimaryPredictions):
        raise TypeError(
            f"primary must be the PrimaryPredictions of model_choice.load_primary; got {type(primary).__name__}"
        )
    if LI_VARIANT not in primary.m4:
        raise ValueError(
            f"the predictions hold no {LI_VARIANT!r} M4 variant, which the Leverage_Index needs; load them with "
            f"model_choice.load_primary(store, variants=(..., {LI_VARIANT!r}))"
        )
    outside = int((~np.isin(primary.innings, POST_INNINGS)).sum())
    if outside:
        raise ValueError(
            f"{_rows(outside)} of the predictions lie outside innings 1 and 2; the Leverage_Index is pooled over the "
            "Deliveries of innings 1 and 2 only"
        )
    return leverage(primary.m4[LI_VARIANT], primary.wp_pre, primary.wp_post)


# ============================================================== Pillar_Components and Stress_Pillars

#: The Roles (glossary). The batter Role reads F19-F22 on the Batter_Deliveries (Legal_Balls and
#: No_Balls), the bowler Role F23-F26 on the Bowler_Deliveries (every Delivery).
BATTER = "batter"
BOWLER = "bowler"
ROLES: tuple[str, ...] = (BATTER, BOWLER)

#: The D4 percentile scopes, the values of ``design_choices.D4_scope``: one rank group per
#: Innings_Group (the Config value), or one group for both innings (the case of Requirement 21,
#: criterion 4).
D4_INNINGS_GROUP = "innings_group"
D4_POOLED = "pooled"
D4_SCOPES: tuple[str, ...] = (D4_INNINGS_GROUP, D4_POOLED)
_POOLED_GROUP = 0  # the one rank-group label of the pooled scope

#: The D5 Stress_Pillar weights, the value of ``design_choices.D5_weights``: equal weights, the
#: only value the design defines (Requirement 12, criterion 13).
D5_EQUAL = "equal"


@dataclass(frozen=True)
class Component:
    """One Pillar_Component (module docstring, "Pillar_Components").

    ``column``: its column in the pillar_components frame; ``label``: its name in reports
    (``F20(a)``); ``pillar``: the Stress_Pillar it feeds (``F20``); ``role``: BATTER or BOWLER.
    """

    column: str
    label: str
    pillar: str
    role: str


@dataclass(frozen=True)
class Pillar:
    """One Stress_Pillar (module docstring, "Stress_Pillars").

    ``name``: its factor ID (``F20``); ``column``: its rank column in the pillars frame;
    ``role``: BATTER or BOWLER; ``components``: the columns of its Pillar_Components;
    ``title``: what it measures.
    """

    name: str
    column: str
    role: str
    components: tuple[str, ...]
    title: str


#: The 13 Pillar_Components of Requirement 12, criteria 1-10, in factor order (module docstring).
COMPONENTS: tuple[Component, ...] = (
    Component("c19", "F19", "F19", BATTER),
    Component("c20a", "F20(a)", "F20", BATTER),
    Component("c20b", "F20(b)", "F20", BATTER),
    Component("c20c", "F20(c)", "F20", BATTER),
    Component("c21", "F21", "F21", BATTER),
    Component("c22a", "F22(a)", "F22", BATTER),
    Component("c22b", "F22(b)", "F22", BATTER),
    Component("c23", "F23", "F23", BOWLER),
    Component("c24a", "F24(a)", "F24", BOWLER),
    Component("c24b", "F24(b)", "F24", BOWLER),
    Component("c25a", "F25(a)", "F25", BOWLER),
    Component("c25b", "F25(b)", "F25", BOWLER),
    Component("c26", "F26", "F26", BOWLER),
)

#: The boundary-only F25(b) flag of the Extras_Free_PI (glossary; Requirement 14, criterion 5).
EXTRAS_FREE_COMPONENT = Component("c25b_extras_free", "F25(b) boundary only", "F25", BOWLER)

#: The columns of the pillar_components frame after ``row``.
COMPONENT_COLUMNS: tuple[str, ...] = (*(component.column for component in COMPONENTS), EXTRAS_FREE_COMPONENT.column)

#: The 8 Stress_Pillars, F19-F22 of the batter Role and F23-F26 of the bowler Role.
PILLARS: tuple[Pillar, ...] = (
    Pillar("F19", "f19", BATTER, ("c19",), "batter scoreboard deficit"),
    Pillar("F20", "f20", BATTER, ("c20a", "c20b", "c20c"), "batter wickets and depth"),
    Pillar("F21", "f21", BATTER, ("c21",), "batter new at the crease"),
    Pillar("F22", "f22", BATTER, ("c22a", "c22b"), "batter squeeze"),
    Pillar("F23", "f23", BOWLER, ("c23",), "bowler scoreboard threat"),
    Pillar("F24", "f24", BOWLER, ("c24a", "c24b"), "bowler batter threat"),
    Pillar("F25", "f25", BOWLER, ("c25a", "c25b"), "bowler over going badly"),
    Pillar("F26", "f26", BOWLER, ("c26",), "bowler wicket drought"),
)

#: F25 with the boundary-only flag in place of F25(b): the over-going-badly pillar of the Extras_Free_PI.
EXTRAS_FREE_F25 = Pillar(
    "F25", "f25_extras_free", BOWLER, ("c25a", EXTRAS_FREE_COMPONENT.column), "bowler over going badly, boundaries only"
)

#: The rank columns of the pillars frame after ``row``.
PILLAR_COLUMNS: tuple[str, ...] = (*(pillar.column for pillar in PILLARS), EXTRAS_FREE_F25.column)

#: The Stress columns of the stress frame after ``row``, each with its Role and Stress_Pillars.
_BATTER_PILLARS: tuple[Pillar, ...] = tuple(pillar for pillar in PILLARS if pillar.role == BATTER)
_BOWLER_PILLARS: tuple[Pillar, ...] = tuple(pillar for pillar in PILLARS if pillar.role == BOWLER)
_STRESS_SPECS: tuple[tuple[str, str, tuple[Pillar, ...]], ...] = (
    ("stress_bat", BATTER, _BATTER_PILLARS),
    ("stress_bowl", BOWLER, _BOWLER_PILLARS),
    (
        "stress_bowl_extras_free",
        BOWLER,
        tuple(EXTRAS_FREE_F25 if pillar.name == EXTRAS_FREE_F25.name else pillar for pillar in _BOWLER_PILLARS),
    ),
)
STRESS_COLUMNS: tuple[str, ...] = tuple(column for column, _, _ in _STRESS_SPECS)
_PILLAR_NAMES: tuple[str, ...] = tuple(pillar.name for pillar in PILLARS)

#: The view columns that the Pillar_Components read, each row from the view of its own fold.
COMPONENT_VIEW_COLUMNS: tuple[str, ...] = (
    remaining_runs.F19_COLUMN, "par_rate_win", remaining_runs.ACH_RATE_COLUMN, "rbq", "theta_striker",
)  # fmt: skip

#: The recent-window inputs (D6) of F20(b) and the innings-1 F23, which ``window`` replaces.
WINDOW_INPUT_COLUMNS: tuple[str, ...] = ("win_wkts", "win_rr", "par_rate_win")
_STATE_WINDOW_COLUMNS: tuple[str, ...] = ("win_wkts", "win_rr")
_VIEW_WINDOW_COLUMN = "par_rate_win"

#: The ``state`` columns that the Pillar_Components read, besides the window columns.
_COMPONENT_STATE_COLUMNS: tuple[str, ...] = (
    "row", "wickets", "rrr", "striker_lb_faced", "striker_dots", "bowler_over_runs", "prev_in_over_flag",
    "prev_in_over_boundary", "lb_since_boundary", "lb_since_wicket",
)  # fmt: skip

#: The F20(a) table of train_wicket_means.
WICKET_MEANS_COLUMNS: tuple[str, ...] = ("fold", "innings", "lb_bowled", "n_train", "mean_wickets")

#: Folder under the outputs folder that holds the Pressure_Engine's files, and the missing-component counts.
PRESSURE_FOLDER = "pressure"
MISSING_COMPONENTS_FILE = "missing_components.csv"
MISSING_COMPONENTS_COLUMNS: tuple[str, ...] = (
    "component", "pillar", "role", "innings", "n_deliveries", "n_missing", "n_missing_live",
)  # fmt: skip


def _flags(values: Any, what: str, n_rows: int | None = None) -> np.ndarray:
    """``values`` (1-D booleans, none missing) as a bool array."""
    series = _vector(values, what)
    if not pd.api.types.is_bool_dtype(series.dtype):
        raise TypeError(f"{what} must be a boolean mask, one flag per row; got dtype {series.dtype}")
    missing = int(series.isna().sum())
    if missing:
        raise ValueError(f"{what} has {missing:,} missing flag(s)")
    return _length(series.to_numpy(dtype=bool), what, n_rows)


def _innings_of(innings: Any, n_rows: int | None = None) -> np.ndarray:
    """``innings`` (1-D, every value 1 or 2) as an int64 array."""
    values = _ints(innings, "innings", n_rows)
    outside = ~np.isin(values, STATE_INNINGS)
    if outside.any():
        raise ValueError(
            f"innings must be {' or '.join(map(str, STATE_INNINGS))} on every row; {_rows(int(outside.sum()))} are not"
        )
    return values


def _same_rows(frame: pd.DataFrame, row_ids: np.ndarray, what: str) -> None:
    ids = _ints(frame["row"], f"{what} column 'row'")
    if len(ids) != len(row_ids) or not np.array_equal(ids, row_ids):
        raise ValueError(f"{what} does not hold the rows of the state table in its order; pass tables of the same Run")


def _setting_whole(value: Any, what: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{what} must be a whole number; got {value!r}")
    if value < minimum:
        raise ValueError(f"{what} must be {minimum} or more; got {value}")
    return int(value)


def _role_flags(roles: Any, n_rows: int) -> dict[str, np.ndarray]:
    """The mask of each Role in ``roles`` ({BATTER: mask, BOWLER: mask}), checked."""
    if not isinstance(roles, Mapping):
        raise TypeError(f"roles must map each Role to its mask (role_masks gives it); got {type(roles).__name__}")
    if set(roles) != set(ROLES):
        raise ValueError(f"roles must map exactly the Roles {', '.join(map(repr, ROLES))}; got {sorted(map(str, roles))}")
    return {role: _flags(roles[role], f"roles[{role!r}]", n_rows) for role in ROLES}


def _group_labels(groups: Any, n_rows: int) -> np.ndarray:
    series = _vector(groups, "groups")
    if len(series) != n_rows:
        raise ValueError(f"groups has {len(series):,} labels, but {n_rows:,} are needed, one per row")
    missing = int(series.isna().sum())
    if missing:
        raise ValueError(f"groups has {missing:,} missing label(s); every row needs a rank group")
    return series.to_numpy()


# ------------------------------------------------------------------------------- component inputs

def own_fold_view(
    view_by_k: Mapping[int, pd.DataFrame], row_fold: Any, row_ids: Any, columns: Sequence[str] = COMPONENT_VIEW_COLUMNS
) -> pd.DataFrame:
    """Each row's ``columns`` from view V_k of its own fold k (module docstring, "Pillar_Components").

    ``view_by_k`` maps each fold k to view V_k of the ``remaining_runs`` checkpoint (one row per
    ``state`` row, in ``state`` order); ``row_fold`` is the fold of each row (``folds.fold_of``)
    and ``row_ids`` its ``row`` id, in ``state`` order; ``columns`` names the numeric view
    columns to take (COMPONENT_VIEW_COLUMNS by default). A row of fold k takes every value from
    V_k, the view that gave its Out_Of_Fold inputs. Returns ``row`` (int32) and ``columns``
    (float64, NaN where the view value is missing) on a RangeIndex, in the order of ``row_ids``.
    Raises TypeError for an argument of the wrong kind, and ValueError when a fold has no view,
    a view lacks a column or does not hold the rows in order, or a value is infinite.
    """
    if not isinstance(view_by_k, Mapping):
        raise TypeError(f"view_by_k must map each fold k to view V_k; got {type(view_by_k).__name__}")
    if isinstance(columns, (str, bytes)) or not isinstance(columns, Sequence):
        raise TypeError(f"columns must be a sequence of view column names; got {columns!r}")
    names = list(columns)
    if not names or len(set(names)) != len(names) or "row" in names or not all(isinstance(n, str) for n in names):
        raise ValueError(f"columns must name distinct view columns other than 'row', at least one; got {names}")
    ids = _ints(row_ids, "row_ids")
    n_rows = len(ids)
    fold = _ints(row_fold, "row_fold", n_rows)
    values = {name: np.full(n_rows, np.nan) for name in names}
    for k in np.unique(fold).tolist():
        if k not in view_by_k:
            raise ValueError(f"view_by_k has no view for fold {k}, which {_rows(int((fold == k).sum()))} belong to")
        view = view_by_k[k]
        what = f"view V_{k}"
        _require(view, ("row", *names), what)
        _same_rows(view, ids, what)
        at = np.flatnonzero(fold == k)
        for name in names:
            values[name][at] = _floats(view[name], f"{what} column {name!r}")[at]
    return pd.DataFrame({"row": ids.astype(np.int32), **values}, columns=["row", *names])


def train_wicket_means(*, fold: Any, innings: Any, lb_bowled: Any, wickets: Any, training: Any) -> pd.DataFrame:
    """The Training_Folds mean wickets of F20(a), per target fold, innings and L (criterion 3; module docstring).

    One value per row of every argument: ``fold`` (the row's fold, 1 or more), ``innings`` (1 or
    2), ``lb_bowled`` (the pre-ball Legal_Balls bowled L) and ``wickets`` (the pre-ball wickets
    fallen) of the ``state`` rows, and ``training``, True for the rows that model training may
    use (``common.training_mask`` of their ``balls`` rows). The target folds are the folds among
    the rows given; the other folds of a target fold k are its Training_Folds.

    Returns WICKET_MEANS_COLUMNS on a RangeIndex, one row per target fold k and (innings, L) pair
    found among the rows, sorted by fold, innings and L: ``fold`` (int32), ``innings`` (int8),
    ``lb_bowled`` (int32), ``n_train`` (int64), the ``training`` rows of the other folds with that
    innings and L, and ``mean_wickets`` (float64), the mean of their wickets, NaN when
    ``n_train`` is 0. No value of row (k, i, L) depends on a fold-k row. Raises TypeError or
    ValueError for arguments of the wrong kind or length, a missing value, a fold below 1, an
    innings other than 1 or 2, or a negative count.
    """
    row_fold = _ints(fold, "fold", minimum=1)
    n_rows = len(row_fold)
    number = _innings_of(innings, n_rows)
    lb = _ints(lb_bowled, "lb_bowled", n_rows, minimum=0)
    fallen = _ints(wickets, "wickets", n_rows, minimum=0)
    train = _flags(training, "training", n_rows)
    if not n_rows:
        return pd.DataFrame(
            {
                "fold": np.empty(0, dtype=np.int32),
                "innings": np.empty(0, dtype=np.int8),
                "lb_bowled": np.empty(0, dtype=np.int32),
                "n_train": np.empty(0, dtype=np.int64),
                "mean_wickets": np.empty(0, dtype=np.float64),
            }
        )
    targets = np.unique(row_fold)
    n_counts = int(lb.max()) + 1
    n_keys = len(STATE_INNINGS) * n_counts
    key = (number - STATE_INNINGS[0]) * n_counts + lb  # innings-major, then L
    cell = np.searchsorted(targets, row_fold) * n_keys + key
    size = len(targets) * n_keys
    # Per fold, the training rows and their summed wickets in each (innings, L) cell. The counts and
    # sums are whole numbers, exact in float64, so the totals of the other folds (all folds minus the
    # target's own) are exact and equal to sums over the other folds alone.
    count = np.bincount(cell[train], minlength=size).reshape(len(targets), n_keys)
    total = np.bincount(cell[train], weights=fallen[train].astype(np.float64), minlength=size)
    total = total.reshape(len(targets), n_keys)
    other_count = count.sum(axis=0) - count
    other_total = total.sum(axis=0) - total
    observed = np.unique(key)
    n_train = other_count[:, observed].reshape(-1)
    sums = other_total[:, observed].reshape(-1)
    mean = np.full(len(n_train), np.nan)
    has = n_train > 0
    mean[has] = sums[has] / n_train[has]
    keys = np.tile(observed, len(targets))
    return pd.DataFrame(
        {
            "fold": np.repeat(targets, len(observed)).astype(np.int32),
            "innings": (keys // n_counts + STATE_INNINGS[0]).astype(np.int8),
            "lb_bowled": (keys % n_counts).astype(np.int32),
            "n_train": n_train.astype(np.int64),
            "mean_wickets": mean,
        },
        columns=list(WICKET_MEANS_COLUMNS),
    )


def wicket_means_for(table: pd.DataFrame, *, fold: Any, innings: Any, lb_bowled: Any) -> np.ndarray:
    """Each row's Training_Folds mean wickets: ``mean_wickets`` of its (fold, innings, L) row of ``table``.

    ``table`` is the table of train_wicket_means; ``fold``, ``innings`` and ``lb_bowled`` give
    each row's fold, innings and pre-ball Legal_Balls bowled. Returns a new float64 array, one
    value per row, NaN where the Training_Folds hold no Delivery of that innings and L (F20(a) is
    then missing). Raises ValueError when ``table`` lacks a column or repeats a (fold, innings, L)
    row, or a row has no row in ``table``, and the errors of the argument checks.
    """
    _require(table, WICKET_MEANS_COLUMNS, "table")
    index = pd.MultiIndex.from_arrays(
        [_ints(table[name], f"table column {name!r}") for name in ("fold", "innings", "lb_bowled")]
    )
    if not index.is_unique:
        raise ValueError("table must hold each (fold, innings, lb_bowled) once; pass the table of train_wicket_means")
    row_fold = _ints(fold, "fold")
    n_rows = len(row_fold)
    at = index.get_indexer(
        pd.MultiIndex.from_arrays([row_fold, _innings_of(innings, n_rows), _ints(lb_bowled, "lb_bowled", n_rows)])
    )
    absent = int((at < 0).sum())
    if absent:
        raise ValueError(
            f"{_rows(absent)} have no (fold, innings, lb_bowled) row in table; build the table from the same rows"
        )
    return _floats(table["mean_wickets"], "table column 'mean_wickets'")[at]


def pillar_components(
    state: pd.DataFrame,
    view: pd.DataFrame,
    *,
    innings: Any,
    mean_wickets: Any,
    new_batter_cap: int,
    window: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The Pillar_Components of every Delivery of innings 1 and 2 (Requirement 12, criteria 1-11; module docstring).

    ``state`` is the ``state`` table, whose rows and order the result follows. ``view`` holds
    each row's own-fold view values (own_fold_view): ``row`` and COMPONENT_VIEW_COLUMNS, the
    rows of ``state`` in order. ``innings`` is each row's innings (1 or 2), ``mean_wickets`` its
    Training_Folds mean wickets at its innings and L (wicket_means_for; NaN where none), and
    ``new_batter_cap`` the D7 cap (``design_choices.D7_new_batter_cap``; 10 and 30 in the
    Requirement 21 cases). ``window`` replaces the recent-window inputs for another window
    length (Requirement 21, criterion 4, D6): ``row`` and WINDOW_INPUT_COLUMNS, the rows of
    ``state`` in order; None takes ``win_wkts`` and ``win_rr`` from ``state`` and ``par_rate_win``
    from ``view``, all for the D6 window of the ``state`` table.

    Returns ``row`` (int32) and COMPONENT_COLUMNS (float64) on a RangeIndex, in ``state`` order:
    the 13 Pillar_Components with the signs of criteria 1-10 (a larger value is more stress for
    the Role), then the Extras_Free F25(b). A component is NaN where an input is missing or its
    window or Legal_Balls left is 0 (criterion 16), never infinite. Raises TypeError or ValueError
    when a column is absent, a table holds other rows or another order, a value has the wrong
    type or is infinite, an innings is not 1 or 2, or the cap is not a whole number of 1 or more.
    """
    state_columns = _COMPONENT_STATE_COLUMNS + (_STATE_WINDOW_COLUMNS if window is None else ())
    view_columns = tuple(name for name in COMPONENT_VIEW_COLUMNS if window is None or name != _VIEW_WINDOW_COLUMN)
    _require(state, state_columns, "state")
    _require(view, ("row", *view_columns), "view")
    row_ids = _ints(state["row"], "state column 'row'")
    n_rows = len(row_ids)
    _same_rows(view, row_ids, "view")
    first = _innings_of(innings, n_rows) == STATE_INNINGS[0]
    cap = _setting_whole(new_batter_cap, "new_batter_cap (D7)", 1)
    means = _floats(mean_wickets, "mean_wickets", n_rows)
    if window is None:
        win_wkts = _ints(state["win_wkts"], "state column 'win_wkts'", minimum=0)
        win_rr = _floats(state["win_rr"], "state column 'win_rr'")
        par_rate = _floats(view[_VIEW_WINDOW_COLUMN], f"view column {_VIEW_WINDOW_COLUMN!r}")
    else:
        _require(window, ("row", *WINDOW_INPUT_COLUMNS), "window")
        _same_rows(window, row_ids, "window")
        win_wkts = _ints(window["win_wkts"], "window column 'win_wkts'", minimum=0)
        win_rr = _floats(window["win_rr"], "window column 'win_rr'")
        par_rate = _floats(window[_VIEW_WINDOW_COLUMN], f"window column {_VIEW_WINDOW_COLUMN!r}")

    def counts(name: str) -> np.ndarray:
        return _ints(state[name], f"state column {name!r}", minimum=0)

    def flag(name: str) -> np.ndarray:
        return _flags(state[name], f"state column {name!r}").astype(np.float64)

    faced = counts("striker_lb_faced")
    rbq = _floats(view["rbq"], "view column 'rbq'")
    theta = _floats(view["theta_striker"], "view column 'theta_striker'")
    rrr = _floats(state["rrr"], "state column 'rrr'")
    ach_rate = _floats(view[remaining_runs.ACH_RATE_COLUMN], f"view column {remaining_runs.ACH_RATE_COLUMN!r}")
    data: dict[str, np.ndarray] = {
        "row": row_ids.astype(np.int32),
        # batter Role
        "c19": _floats(view[remaining_runs.F19_COLUMN], f"view column {remaining_runs.F19_COLUMN!r}"),
        "c20a": counts("wickets") - means,
        "c20b": win_wkts.astype(np.float64),
        "c20c": 1.0 - rbq,
        "c21": (cap - np.minimum(faced, cap)).astype(np.float64),
        "c22a": counts("striker_dots").astype(np.float64),
        "c22b": counts("lb_since_boundary").astype(np.float64),
        # bowler Role
        "c23": np.where(first, win_rr - par_rate, ach_rate - rrr),
        "c24a": faced * theta,
        "c24b": rbq.copy(),
        "c25a": counts("bowler_over_runs").astype(np.float64),
        "c25b": flag("prev_in_over_flag"),
        "c26": counts("lb_since_wicket").astype(np.float64),
        EXTRAS_FREE_COMPONENT.column: flag("prev_in_over_boundary"),
    }
    frame = pd.DataFrame(data, columns=["row", *COMPONENT_COLUMNS])
    for name in COMPONENT_COLUMNS:
        infinite = int(np.isinf(frame[name].to_numpy()).sum())
        if infinite:
            raise ValueError(
                f"the Pillar_Component {name!r} has {infinite:,} infinite value(s); a value is a number or missing"
            )
    return frame


# ------------------------------------------------------------------------------------------- ranks

def role_masks(cls: Any) -> dict[str, np.ndarray]:
    """The rows of each Role from the DeliveryClass codes ``cls`` (the ``cls`` column of ``balls``, in ``state`` order).

    Returns {BATTER: the Batter_Deliveries (Legal_Balls and No_Balls), BOWLER: the
    Bowler_Deliveries (every Delivery)}, new bool arrays, one flag per row. Raises TypeError or
    ValueError when ``cls`` is not whole numbers or holds a code that is not a DeliveryClass.
    """
    codes = _ints(cls, "cls")
    unknown = ~np.isin(codes, [int(member) for member in DeliveryClass])
    if unknown.any():
        raise ValueError(f"cls holds {_rows(int(unknown.sum()))} with a code that is not a DeliveryClass")
    return {BATTER: codes != int(DeliveryClass.WIDE), BOWLER: np.ones(len(codes), dtype=bool)}


def rank_groups(innings: Any, scope: str) -> np.ndarray:
    """The rank group of each row under the D4 percentile ``scope`` (``design_choices.D4_scope``).

    ``innings`` is each row's innings (1 or 2). D4_INNINGS_GROUP (the Config value) gives each
    row its innings, so every percentile rank is taken within the Innings_Group; D4_POOLED (the
    Requirement 21, criterion 4 case) gives every row one group, pooling both innings. Returns a
    new int64 array, the ``groups`` of pillars (and of the LI, Stress and PI ranks). Raises
    ValueError for a scope that is not in D4_SCOPES, and the errors of the innings check.
    """
    if scope not in D4_SCOPES:
        raise ValueError(f"the D4 percentile scope must be one of {', '.join(map(repr, D4_SCOPES))}; got {scope!r}")
    number = _innings_of(innings)
    return number.copy() if scope == D4_INNINGS_GROUP else np.full(len(number), _POOLED_GROUP, dtype=np.int64)


def _ranked(values: np.ndarray, groups: np.ndarray, reference: np.ndarray, *, allow_missing: bool, what: str) -> np.ndarray:
    try:
        return percentile_rank_within(values, groups, reference, allow_missing=allow_missing)
    except ValueError as exc:
        raise ValueError(f"{what}: {exc}") from None


def pillars(components: pd.DataFrame, *, live: Any, roles: Mapping[str, Any], groups: Any) -> pd.DataFrame:
    """The Stress_Pillars of every Delivery of each Role (Requirement 12, criteria 12 and 14-16; module docstring).

    ``components`` is the frame of pillar_components. ``live`` flags the Live_Balls, one flag per
    row; ``roles`` maps BATTER and BOWLER to the rows of the Role (role_masks); ``groups`` gives
    each row its rank group (rank_groups with the D4 scope). The Pillar_Reference_Set of a Role
    and group is the Role's Live_Balls in the group. Each component of the Role's Deliveries is
    ranked against it with the missing-value rule (rank ``ranks.MISSING_RANK``, 0.5, and no place
    in the component's reference set); a one-component pillar is that rank, and a composite pillar is
    the Percentile_Rank of the mean of its component ranks among the means of the whole
    Pillar_Reference_Set. Dead_Balls get their rank against the set.

    Returns ``row`` (int32) and PILLAR_COLUMNS (float64) on a RangeIndex, in the row order of
    ``components``: f19-f22 on the batter Role's rows, f23-f26 and ``f25_extras_free`` on the
    bowler Role's, each within [0, 1], NaN on the rows outside the pillar's Role. Raises
    TypeError or ValueError for an argument of the wrong kind or length, a missing flag or group
    label, an infinite component, and a group that has Role rows but no Live_Ball of the Role
    with a value to rank against.
    """
    _require(components, ("row", *COMPONENT_COLUMNS), "components")
    row_ids = _ints(components["row"], "components column 'row'")
    n_rows = len(row_ids)
    live_flags = _flags(live, "live", n_rows)
    masks = _role_flags(roles, n_rows)
    labels = _group_labels(groups, n_rows)
    out: dict[str, np.ndarray] = {"row": row_ids.astype(np.int32)}
    component_ranks: dict[str, np.ndarray] = {}  # each component belongs to one Role, so one set of rows
    for pillar in (*PILLARS, EXTRAS_FREE_F25):
        at = np.flatnonzero(masks[pillar.role])
        column = np.full(n_rows, np.nan)
        if len(at):
            group, reference = labels[at], live_flags[at]
            ranks = []
            for name in pillar.components:
                if name not in component_ranks:
                    values = _floats(components[name], f"components column {name!r}", n_rows)[at]
                    component_ranks[name] = _ranked(
                        values, group, reference, allow_missing=True,
                        what=f"Pillar_Component {name!r} of the {pillar.role} Role",
                    )  # fmt: skip
                ranks.append(component_ranks[name])
            if len(ranks) == 1:
                column[at] = ranks[0]
            else:
                mean = ranks[0].copy()
                for rank in ranks[1:]:
                    mean += rank
                mean /= len(ranks)
                column[at] = _ranked(
                    mean, group, reference, allow_missing=False, what=f"Stress_Pillar {pillar.column!r}"
                )
        out[pillar.column] = column
    return pd.DataFrame(out, columns=["row", *PILLAR_COLUMNS])


def stress(pillar_ranks: pd.DataFrame, *, weights: str, without: Collection[str] = ()) -> pd.DataFrame:
    """The Stress of each Role (Requirement 12, criterion 13; module docstring, "Stress").

    ``pillar_ranks`` is the frame of pillars; ``weights`` the D5 Stress_Pillar weights
    (``design_choices.D5_weights``), which must be D5_EQUAL; ``without`` the names of
    Stress_Pillars to leave out (``"F21"``, ...) for the ablations of Requirement 21, criterion
    1, none by default. A Role's Stress is the equal-weighted mean of its Stress_Pillars that
    ``without`` keeps: 0.25 each for all four.

    Returns ``row`` (int32) and STRESS_COLUMNS (float64) on a RangeIndex, in the row order of
    ``pillar_ranks``: ``stress_bat`` (F19-F22; NaN outside the batter Role, so on Wides),
    ``stress_bowl`` (F23-F26) and ``stress_bowl_extras_free`` (F23, F24, the Extras_Free F25 and
    F26), the Stress of the Extras_Free_PI. Raises ValueError for weights other than D5_EQUAL, a
    name in ``without`` that is not a Stress_Pillar, a Role left without a pillar, or a row whose
    Role pillars are set for some pillars and missing for others; TypeError for an argument of
    the wrong kind.
    """
    if weights != D5_EQUAL:
        raise ValueError(f"the D5 Stress_Pillar weights must be {D5_EQUAL!r}, the only weights defined; got {weights!r}")
    if isinstance(without, (str, bytes)) or not isinstance(without, Collection):
        raise TypeError(f"without must be a collection of Stress_Pillar names such as ('F21',); got {without!r}")
    dropped = set(without)
    unknown = sorted(map(str, dropped - set(_PILLAR_NAMES)))
    if unknown:
        raise ValueError(f"without names {', '.join(unknown)}, which are not Stress_Pillars ({', '.join(_PILLAR_NAMES)})")
    _require(pillar_ranks, ("row", *PILLAR_COLUMNS), "pillar_ranks")
    row_ids = _ints(pillar_ranks["row"], "pillar_ranks column 'row'")
    n_rows = len(row_ids)
    out: dict[str, np.ndarray] = {"row": row_ids.astype(np.int32)}
    for column, role, members in _STRESS_SPECS:
        kept = [pillar for pillar in members if pillar.name not in dropped]
        if not kept:
            raise ValueError(f"without leaves the {role} Role no Stress_Pillar; an ablation removes one pillar at a time")
        matrix = np.column_stack(
            [_floats(pillar_ranks[pillar.column], f"pillar_ranks column {pillar.column!r}", n_rows) for pillar in kept]
        )
        present = ~np.isnan(matrix)
        partial = int((present.any(axis=1) & ~present.all(axis=1)).sum())
        if partial:
            raise ValueError(
                f"{_rows(partial)} of pillar_ranks have some {role} Stress_Pillars and miss others; pass the frame of pillars"
            )
        total = matrix[:, 0].copy()
        for k in range(1, matrix.shape[1]):
            total += matrix[:, k]
        out[column] = total / len(kept)
    return pd.DataFrame(out, columns=["row", *STRESS_COLUMNS])


# ------------------------------------------------------------------------------ missing components

def missing_components(components: pd.DataFrame, *, live: Any, roles: Mapping[str, Any], innings: Any) -> pd.DataFrame:
    """The Deliveries with a missing Pillar_Component, per component, Role and Innings_Group (criterion 16).

    ``components``, ``live`` and ``roles`` are as for pillars; ``innings`` gives each row's
    Innings_Group (1 or 2), whatever the D4 rank groups. Returns MISSING_COMPONENTS_COLUMNS, one
    row per Pillar_Component of COMPONENTS (in order, for its own Role) and Innings_Group (1,
    then 2): ``component`` (``F20(a)``, ...), ``pillar``, ``role``, ``innings`` (int8);
    ``n_deliveries``, the Role's Deliveries in the Innings_Group; ``n_missing``, those whose
    component is missing, each ranked ``ranks.MISSING_RANK`` (0.5); ``n_missing_live``, the Live_Balls
    among them, which leave the component's Pillar_Reference_Set (the counts int64). Raises the
    errors of the argument checks.
    """
    _require(components, ("row", *(component.column for component in COMPONENTS)), "components")
    n_rows = len(components)
    live_flags = _flags(live, "live", n_rows)
    masks = _role_flags(roles, n_rows)
    number = _innings_of(innings, n_rows)
    records = []
    for component in COMPONENTS:
        missing = np.isnan(_floats(components[component.column], f"components column {component.column!r}", n_rows))
        for group in STATE_INNINGS:
            rows = masks[component.role] & (number == group)
            gone = missing & rows
            records.append(
                (component.label, component.pillar, component.role, group, int(rows.sum()), int(gone.sum()),
                 int((gone & live_flags).sum()))
            )  # fmt: skip
    table = pd.DataFrame.from_records(records, columns=list(MISSING_COMPONENTS_COLUMNS))
    table["innings"] = table["innings"].astype(np.int8)
    for name in MISSING_COMPONENTS_COLUMNS[4:]:
        table[name] = table[name].astype(np.int64)
    return table


def write_missing_components(outdir: str | Path, counts: pd.DataFrame) -> Path:
    """Write the missing_components table to MISSING_COMPONENTS_FILE in ``outdir`` (criterion 16); return the path.

    ``outdir`` is the Pressure_Engine's folder (PRESSURE_FOLDER under the outputs folder in a
    Run) and is created when absent. The file has a header row and the rows of ``counts`` with
    the MISSING_COMPONENTS_COLUMNS, in UTF-8 with ``\\n`` line ends, written through a temporary
    file renamed in place, so it replaces any earlier copy whole. Raises ValueError when
    ``counts`` lacks a column or does not list each Pillar_Component and Innings_Group once, in
    the order of missing_components, and TypeError when it is not a DataFrame.
    """
    _require(counts, MISSING_COMPONENTS_COLUMNS, "counts")
    expected = [(component.label, group) for component in COMPONENTS for group in STATE_INNINGS]
    listed = list(zip(counts["component"].astype(str).tolist(), counts["innings"].astype(int).tolist(), strict=True))
    if listed != expected:
        raise ValueError(
            "counts must list each Pillar_Component and Innings_Group once, in order; pass the table of missing_components"
        )
    folder = Path(outdir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / MISSING_COMPONENTS_FILE
    temporary = path.with_name(path.name + ".tmp")
    try:
        counts[list(MISSING_COMPONENTS_COLUMNS)].to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


# ============================================================== Pressure_Index and Pressure_Balls

#: Name of the Pipeline_Stage that runs the Pressure_Engine, of its checkpoint and of its main table.
STAGE_NAME = "pressure"
PRESSURE_TABLE = "pressure"

#: PI = _PI_SCALE x [alpha x rank(LI) + (1 - alpha) x rank(Stress)] (Requirement 13, criterion 2).
_PI_SCALE = 100.0

#: The PI, Pressure_Ball and Ordinary_Ball columns of the pressure table: the batter Role, the
#: bowler Role and the bowler Role's Extras_Free_PI (validation only, so no Ordinary_Ball column).
PI_COLUMNS: tuple[str, ...] = ("pi_bat", "pi_bowl", "pi_bowl_extras_free")
PRESSURE_FLAG_COLUMNS: tuple[str, ...] = ("pressure_bat", "pressure_bowl", "pressure_bowl_extras_free")
ORDINARY_COLUMNS: tuple[str, ...] = ("ordinary_bat", "ordinary_bowl")


@dataclass(frozen=True)
class _Index:
    """One Pressure_Index of the pressure table: its Role and its Stress, PI, flag and Ordinary_Ball columns."""

    role: str
    stress: str
    pi: str
    pressure: str
    ordinary: str | None
    title: str


_INDEXES: tuple[_Index, ...] = (
    _Index(BATTER, "stress_bat", "pi_bat", "pressure_bat", "ordinary_bat", "batter"),
    _Index(BOWLER, "stress_bowl", "pi_bowl", "pressure_bowl", "ordinary_bowl", "bowler"),
    _Index(
        BOWLER, "stress_bowl_extras_free", "pi_bowl_extras_free", "pressure_bowl_extras_free", None,
        "bowler Extras_Free_PI",
    ),
)  # fmt: skip
_MAIN_INDEXES: tuple[_Index, ...] = _INDEXES[:2]  # the PIs behind the Main_Ranking, one per Role

#: The columns of the pressure table, in order (module docstring, "Reading the pressure table").
PRESSURE_COLUMNS: tuple[str, ...] = (
    "row", "wp_pre", "live", "li_raw", "li", *COMPONENT_COLUMNS, *PILLAR_COLUMNS, *STRESS_COLUMNS,
    "pi_bat", "pi_bowl", "pressure_bat", "pressure_bowl", "ordinary_bat", "ordinary_bowl",
    "pi_bowl_extras_free", "pressure_bowl_extras_free",
)  # fmt: skip

#: The share tables (share_tables; Requirement 13, criterion 8): their columns, and for each Role
#: the checkpoint table and the file in PRESSURE_FOLDER.
SHARE_COLUMNS: tuple[str, ...] = (
    "role", "innings", "phase", "comp_type", "n_deliveries", "n_live", "n_pressure",
    "pct_of_role_deliveries", "pct_of_innings_pressure",
)  # fmt: skip
SHARE_TABLES: Mapping[str, str] = MappingProxyType({BATTER: "share_batter", BOWLER: "share_bowler"})
SHARE_FILES: Mapping[str, str] = MappingProxyType(
    {BATTER: "pressure_share_batter.csv", BOWLER: "pressure_share_bowler.csv"}
)

#: The Pressure_Ball counts of each PI and Innings_Group (pressure_summary), its checkpoint table and file.
SUMMARY_COLUMNS: tuple[str, ...] = (
    "pi", "role", "innings", "n_deliveries", "n_live", "n_pressure", "pct_of_live", "lowest_pressure_pi",
    "highest_ordinary_live_pi",
)  # fmt: skip
SUMMARY_TABLE = "pressure_summary"
SUMMARY_FILE = "pressure_summary.csv"

#: The over numbers of the mean-LI chart (Requirement 11, criterion 13): the 20 Overs of a T20
#: innings, numbered from 1 as ``overid`` of ``balls`` numbers them. The chart's data table, its
#: checkpoint table and its two files in PRESSURE_FOLDER.
OVERS: tuple[int, ...] = tuple(range(1, 21))
LI_BY_OVER_COLUMNS: tuple[str, ...] = ("innings", "over", "n_deliveries", "mean_li")
LI_BY_OVER_TABLE = "li_by_over"
LI_BY_OVER_FILE = "mean_li_by_over.csv"
LI_BY_OVER_CHART_FILE = "mean_li_by_over.png"

#: The checkpoint table of the missing-component counts (MISSING_COMPONENTS_FILE holds the same rows).
MISSING_COMPONENTS_TABLE = "missing_components"

#: The pressure definition file under the docs folder (Requirement 13, criterion 9).
DEFINITION_FILE = "pressure_definition.md"

#: Resolution of the mean-LI chart in dots per inch, fixed so that the same data give the same image.
CHART_DPI = 150

#: Data-derived Method_Constants that the stage records (Pressure_Engine): the pooled mean unscaled
#: LI (Requirement 11, criterion 10); and, for each Role and Innings_Group ({"batter": {"1": ...,
#: "2": ...}, "bowler": {...}}), the Pressure_Balls as a percentage of the Live_Balls and the
#: lowest PI of a Pressure_Ball, the realised cut of D3.
POOLED_MEAN_LI_CONSTANT = "pressure.pooled_mean_unscaled_li"
SHARE_CONSTANT = "pressure.pressure_ball_pct_of_live_balls"
PI_CUTOFF_CONSTANT = "pressure.lowest_pressure_ball_pi"

_COMPONENT = "Pressure_Engine"  # the glossary component that produces the constants
_LIVE_BAND = "design_choices.D1_live_band"
_ALPHA = "design_choices.D2_alpha"
_SHARE = "design_choices.D3_pressure_share"
_SCOPE = "design_choices.D4_scope"
_WEIGHTS = "design_choices.D5_weights"
_WINDOW = "design_choices.D6_window"
_CAP = "design_choices.D7_new_batter_cap"
_N_FOLDS = "design_choices.D15_n_folds"
_SHARE_BAND = "tests.share_band"  # the Test_Suite band of Requirement 13, criterion 7, for the Run_Log
_FLOAT_FORMAT = "%.6f"
_CHART_SIZE = (10.0, 4.8)  # inches
_NUMERATOR_SLACK = 1e-6  # |rank x 2n - m| allowed when a rank's numerator m is recovered (about 1e-10 in practice)
_PHASE_NAMES: Mapping[int, str] = MappingProxyType({int(member): member.name.lower() for member in Phase})
_N_PHASES = len(Phase)
_N_COMP_TYPES = len(COMPETITION_TYPES)
_N_CELLS = len(STATE_INNINGS) * _N_PHASES * _N_COMP_TYPES  # 18 Innings_Group x Phase x Competition_Type cells
_STAGE_BALL_COLUMNS: tuple[str, ...] = (
    "row", "match_id", "innings", "cls", "comp_type", "overid", "is_super_over", "excluded",
)  # fmt: skip
_FAMILY_LABELS: Mapping[str, str] = MappingProxyType(
    {
        model_choice.LOGISTIC: "logistic baseline",
        model_choice.LGBM: "LightGBM",
        model_choice.XGB: "XGBoost",
        model_choice.RF: "random forest",
    }
)


class PressureInputError(StageError):
    """WP, LI or a Role's Stress is missing or not finite where the Pressure_Index needs it (exit code 5).

    Requirement 13, criterion 10 (finite_guard). ``counts`` maps each affected quantity (``"WP"``,
    ``"LI"`` or a Stress column of STRESS_COLUMNS) to the number of affected Deliveries in each
    Innings_Group (``{1: ..., 2: ...}``), read-only; the message gives the same numbers.
    """

    def __init__(self, message: str, *, counts: Mapping[str, Mapping[int, int]]) -> None:
        super().__init__(message)
        self.counts: Mapping[str, Mapping[int, int]] = MappingProxyType(
            {
                str(quantity): MappingProxyType({int(group): int(count) for group, count in per_group.items()})
                for quantity, per_group in counts.items()
            }
        )


# ------------------------------------------------------------------------------- argument checks

def _series_of(values: Any, what: str) -> pd.Series:
    """``values`` as a Series: a pandas extension array (such as a BooleanArray) keeps its dtype."""
    if isinstance(values, pd.api.extensions.ExtensionArray):
        return pd.Series(values, copy=False)
    return _vector(values, what)


def _any_floats(values: Any, what: str, n_rows: int | None = None) -> np.ndarray:
    """``values`` (1-D numbers) as a float64 array, NaN for a missing value; infinite values are kept."""
    series = _series_of(values, what)
    if pd.api.types.is_bool_dtype(series.dtype) or not pd.api.types.is_numeric_dtype(series.dtype):
        raise TypeError(f"{what} must hold numbers; got dtype {series.dtype}")
    return _length(series.to_numpy(dtype=np.float64, na_value=np.nan), what, n_rows)


def _number(value: Any, what: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise TypeError(f"{what} must be a finite number; got {value!r}")
    return float(value)


def _alpha(value: Any) -> float:
    alpha = _number(value, "alpha (D2)")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha (D2) must lie within [0, 1]; got {alpha!r}")
    return alpha


def _share(value: Any) -> float:
    share = _number(value, "share (D3)")
    if not 0.0 < share < 1.0:
        raise ValueError(f"the pressure share (D3) must lie strictly between 0 and 1; got {share!r}")
    return share


def _live_band(band: Any) -> tuple[float, float]:
    """The D1 bounds (lower, upper), checked: two finite numbers with 0 <= lower <= upper <= 1."""
    if isinstance(band, (str, bytes, Mapping)) or not isinstance(band, (Sequence, np.ndarray)) or len(band) != 2:
        raise TypeError(f"the live band (D1) must be two numbers (lower, upper); got {band!r}")
    lower, upper = (_number(value, "a live band (D1) bound") for value in band)
    if not 0.0 <= lower <= upper <= 1.0:
        raise ValueError(f"the live band (D1) must satisfy 0 <= lower <= upper <= 1; got [{lower!r}, {upper!r}]")
    return lower, upper


def _nullable_flags(values: Any, what: str, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """``values`` (1-D bool or nullable boolean) as (flags, missing), two bool arrays; a missing flag is False."""
    series = _series_of(values, what)
    if not pd.api.types.is_bool_dtype(series.dtype):
        raise TypeError(f"{what} must be a boolean mask, one flag per row; got dtype {series.dtype}")
    _length(series.to_numpy(), what, n_rows)
    missing = series.isna().to_numpy(dtype=bool)
    return series.fillna(False).to_numpy(dtype=bool), missing


def _group_sizes(labels: np.ndarray) -> np.ndarray:
    """For each row, the number of rows with its label (float64), the n of its reference set."""
    codes, _ = pd.factorize(labels)
    return np.bincount(codes)[codes].astype(np.float64)


def _numerators(ranks: np.ndarray, sizes: np.ndarray, what: str) -> np.ndarray:
    """m = 2 x below + equal behind each Percentile_Rank (below + 0.5 x equal) / n: rank x 2n, a whole number."""
    scaled = ranks * (2.0 * sizes)
    whole = np.rint(scaled)
    off = np.abs(scaled - whole) > _NUMERATOR_SLACK
    if off.any():
        raise ValueError(
            f"{_rows(int(off.sum()))} of the {what} ranks are not (below + 0.5 x equal) / n for their reference set"
        )
    return whole


def _role_rows(live: Any, role_mask: Any, groups: Any, n_rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The Role mask, the Live_Balls of the Role (positions) and every row's rank-group label, checked."""
    live_flags = _flags(live, "live", n_rows)
    mask = _flags(role_mask, "role_mask", n_rows)
    labels = _group_labels(groups, n_rows)
    return mask, np.flatnonzero(mask & live_flags), labels


# ------------------------------------------------------------------------------- Live_Balls, guard

def live_balls(wp: Any, band: Any) -> np.ndarray:
    """The Live_Ball flag of each Delivery (Requirement 13, criterion 1; module docstring, "Live_Balls").

    ``wp``: each Delivery's unrounded Out_Of_Fold pre-ball WP (``PrimaryPredictions.wp_pre``),
    compared as given in float64, never rounded. ``band``: the D1 bounds (lower, upper)
    (``design_choices.D1_live_band``), two numbers with 0 <= lower <= upper <= 1; (0, 1) makes
    every Delivery with a WP a Live_Ball, the case of Requirement 21, criterion 1 without the
    live band. Returns a new bool array, one flag per Delivery: True when lower <= WP <= upper,
    both bounds inclusive, False otherwise, also for a WP that is missing or not finite
    (finite_guard reports it). Raises TypeError or ValueError for a band that is not two numbers
    in that order within [0, 1], and for a ``wp`` that is not 1-D numbers.
    """
    lower, upper = _live_band(band)
    values = _any_floats(wp, "wp")
    return (values >= lower) & (values <= upper)


def finite_guard(
    *, wp: Any, li: Any, stresses: pd.DataFrame, live: Any, roles: Mapping[str, Any], innings: Any
) -> None:
    """Stop the Run when an input of the Pressure_Index is missing or not finite (Requirement 13, criterion 10).

    One value per Delivery of innings 1 and 2, in one order: ``wp``, the pre-ball WP; ``li``, the
    LI; ``stresses``, the frame of ``stress`` (STRESS_COLUMNS); ``live``, the Live_Ball flags
    (live_balls); ``roles``, the Role masks (role_masks); ``innings``, the Innings_Group (1 or 2).
    The guard checks WP on every Delivery, LI on every Live_Ball and each Stress column on the
    Live_Balls of its Role: ``stress_bat`` on the batter Role's, ``stress_bowl`` and
    ``stress_bowl_extras_free`` on the bowler Role's. Call it before any PI is computed or written.

    Returns None when every value checked is a finite number. Otherwise raises
    PressureInputError (a StageError, exit code 5) whose message names each affected quantity with
    its count of affected Deliveries in each Innings_Group, and whose ``counts`` hold the same
    numbers. Raises TypeError or ValueError for malformed arguments.
    """
    wp_values = _any_floats(wp, "wp")
    n_rows = len(wp_values)
    li_values = _any_floats(li, "li", n_rows)
    _require(stresses, STRESS_COLUMNS, "stresses")
    if len(stresses) != n_rows:
        raise ValueError(f"stresses has {len(stresses):,} rows, but {n_rows:,} are needed, one per Delivery")
    live_flags = _flags(live, "live", n_rows)
    masks = _role_flags(roles, n_rows)
    number = _innings_of(innings, n_rows)
    checks: list[tuple[str, str, np.ndarray, np.ndarray]] = [
        ("WP", "Deliveries", np.ones(n_rows, dtype=bool), wp_values),
        ("LI", "Live_Balls", live_flags, li_values),
    ]
    for column, role, _ in _STRESS_SPECS:
        values = _any_floats(stresses[column], f"stresses column {column!r}", n_rows)
        checks.append((column, f"{role}-Role Live_Balls", masks[role] & live_flags, values))
    counts: dict[str, dict[int, int]] = {}
    parts: list[str] = []
    for quantity, scope, rows, values in checks:
        bad = rows & ~np.isfinite(values)
        if not bad.any():
            continue
        per_group = {group: int((bad & (number == group)).sum()) for group in STATE_INNINGS}
        counts[quantity] = per_group
        listed = ", ".join(f"innings {group}: {count:,}" for group, count in per_group.items())
        parts.append(
            f"{quantity} is missing or not finite on {int(bad.sum()):,} of the {int(rows.sum()):,} {scope} ({listed})"
        )
    if counts:
        raise PressureInputError(
            "the Pressure_Index cannot be computed, so no PI or Pressure_Ball output is written (Requirement 13, "
            "criterion 10): " + "; ".join(parts),
            counts=counts,
        )


# ---------------------------------------------------------------------------- PI and the flags

def pressure_index(li: Any, stress: Any, live: Any, role_mask: Any, groups: Any, alpha: float) -> np.ndarray:
    """The Pressure_Index of one Role (Requirement 13, criteria 2, 3 and 6; module docstring, "Pressure_Index").

    One value per Delivery: ``li``, the LI; ``stress``, the Role's Stress (``stress_bat``,
    ``stress_bowl`` or ``stress_bowl_extras_free``); ``live``, the Live_Ball flags;
    ``role_mask``, the Role's Deliveries (role_masks); ``groups``, the rank group (rank_groups with
    the D4 scope). ``alpha`` is the D2 blend (``design_choices.D2_alpha``), within [0, 1].

    Returns a new float64 array: on a Live_Ball of the Role, 100 x [alpha x rank(LI) + (1 - alpha)
    x rank(Stress)], both Percentile_Ranks among the Role's Live_Balls of the same rank group, so
    0 < PI < 100; 0 on a Dead_Ball of the Role; NaN outside the Role (a Wide's batter-Role PI).
    The PI is computed from the rank numerators, so that equal sums give equal PI (module
    docstring). Only the Live_Balls of the Role are ranked, so LI and Stress may be missing on
    other rows. Raises ValueError when an LI or Stress of a Live_Ball of the Role is missing or
    infinite (call finite_guard first for the Requirement 13, criterion 10 error), and TypeError
    or ValueError for arguments of the wrong kind, length or range.
    """
    li_values = _any_floats(li, "li")
    n_rows = len(li_values)
    stress_values = _any_floats(stress, "stress", n_rows)
    mask, at, labels = _role_rows(live, role_mask, groups, n_rows)
    weight = _alpha(alpha)
    out = np.full(n_rows, np.nan)
    out[mask] = 0.0  # Dead_Balls of the Role (criterion 3); the Live_Balls are set below
    if len(at):
        group = labels[at]
        reference = np.ones(len(at), dtype=bool)  # every Live_Ball of the Role is in its group's reference set
        sizes = _group_sizes(group)
        m_li = _numerators(
            _ranked(li_values[at], group, reference, allow_missing=False, what="LI of the Role's Live_Balls"),
            sizes, "LI",
        )  # fmt: skip
        m_stress = _numerators(
            _ranked(stress_values[at], group, reference, allow_missing=False, what="Stress of the Role's Live_Balls"),
            sizes, "Stress",
        )  # fmt: skip
        out[at] = _PI_SCALE * (weight * m_li + (1.0 - weight) * m_stress) / (2.0 * sizes)
    return out


def pressure_flags(pi: Any, live: Any, role_mask: Any, groups: Any, share: float) -> pd.arrays.BooleanArray:
    """The Pressure_Ball flags of one Role (Requirement 13, criteria 3 and 4; module docstring).

    ``pi``: the Role's PI (pressure_index); ``live``, ``role_mask`` and ``groups`` as for
    pressure_index; ``share``: the D3 pressure share (``design_choices.D3_pressure_share``),
    strictly between 0 and 1. A Live_Ball of the Role is a Pressure_Ball when the Percentile_Rank
    of its PI among the Role's Live_Balls of the same rank group is greater than 1 - share, so
    Deliveries with equal PI get the same flag. Returns a nullable boolean array, one flag per
    Delivery: True on the Pressure_Balls, False on the other Deliveries of the Role (every
    Dead_Ball among them), <NA> outside the Role (a Wide for the batter Role). Raises ValueError
    for a PI of a Live_Ball of the Role that is missing or infinite, and TypeError or ValueError
    for arguments of the wrong kind, length or range.
    """
    values = _any_floats(pi, "pi")
    n_rows = len(values)
    mask, at, labels = _role_rows(live, role_mask, groups, n_rows)
    cut = 1.0 - _share(share)
    flags = np.zeros(n_rows, dtype=bool)
    if len(at):
        ranks = _ranked(
            values[at], labels[at], np.ones(len(at), dtype=bool), allow_missing=False, what="PI of the Role's Live_Balls"
        )
        flags[at] = ranks > cut
    return pd.arrays.BooleanArray(flags, ~mask)


def ordinary_balls(pressure: Any) -> pd.arrays.BooleanArray:
    """The Ordinary_Ball flags of one Role (Requirement 13, criterion 5): every Role Delivery that is not a Pressure_Ball.

    ``pressure`` is the Role's Pressure_Ball flags (pressure_flags; bool or nullable boolean).
    Returns a new nullable boolean array: the negation of each flag, <NA> where ``pressure`` is
    <NA> (outside the Role), so every Delivery of the Role is exactly one of a Pressure_Ball or an
    Ordinary_Ball. Raises TypeError when ``pressure`` is not 1-D booleans.
    """
    series = _series_of(pressure, "pressure")
    flags, missing = _nullable_flags(series, "pressure", len(series))
    return pd.arrays.BooleanArray(~flags & ~missing, missing.copy())


# ------------------------------------------------------------------------------------ the tables

def share_tables(
    *, pressure: Mapping[str, Any], live: Any, roles: Mapping[str, Any], innings: Any, phase: Any, comp_type: Any
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The Pressure_Ball share tables of the batter and bowler Roles (Requirement 13, criterion 8; module docstring).

    One value per Delivery of innings 1 and 2: ``pressure`` maps BATTER and BOWLER to the Role's
    Pressure_Ball flags (pressure_flags; bool or nullable boolean, <NA> or False outside the Role);
    ``live``, ``roles`` and ``innings`` as for finite_guard; ``phase``, the Phase code (``phase`` of
    ``state``: Phase values 0-2); ``comp_type``, the Competition_Type (``comp_type`` of ``balls``,
    one of ``reconstruct.COMPETITION_TYPES``).

    Returns (batter table, bowler table), each SHARE_COLUMNS with 18 rows on a RangeIndex, one per
    Innings_Group x Phase x Competition_Type cell, innings 1 then 2, Phases in Phase order and
    Competition_Types in COMPETITION_TYPES order: ``n_deliveries``, ``n_live`` and ``n_pressure``,
    the Role's Deliveries, Live_Balls and Pressure_Balls in the cell (int64);
    ``pct_of_role_deliveries``, 100 x n_pressure / n_deliveries; ``pct_of_innings_pressure``, 100 x
    n_pressure / the Role's Pressure_Balls in the Innings_Group (float64, NaN when the denominator
    is 0). Raises ValueError when a flag of a Role Delivery is missing, a Pressure_Ball is not a
    Live_Ball of its Role, or a Phase or Competition_Type is unknown or missing, and TypeError or
    ValueError for arguments of the wrong kind or length.
    """
    number = _innings_of(innings)
    n_rows = len(number)
    live_flags = _flags(live, "live", n_rows)
    masks = _role_flags(roles, n_rows)
    if not isinstance(pressure, Mapping) or set(pressure) != set(ROLES):
        raise TypeError(f"pressure must map exactly the Roles {', '.join(map(repr, ROLES))} to their flags")
    phase_codes = _ints(phase, "phase", n_rows)
    unknown = ~np.isin(phase_codes, list(_PHASE_NAMES))
    if unknown.any():
        raise ValueError(f"phase holds {_rows(int(unknown.sum()))} with a code that is not a Phase")
    comp = _vector(comp_type, "comp_type")
    _length(comp.to_numpy(), "comp_type", n_rows)
    comp_codes = pd.Index(COMPETITION_TYPES).get_indexer(comp.astype(object).to_numpy(dtype=object))
    if (comp_codes < 0).any():
        raise ValueError(
            f"comp_type holds {_rows(int((comp_codes < 0).sum()))} that are missing or not one of "
            f"{', '.join(COMPETITION_TYPES)}"
        )
    cell = ((number - STATE_INNINGS[0]) * _N_PHASES + phase_codes) * _N_COMP_TYPES + comp_codes
    per_innings = _N_PHASES * _N_COMP_TYPES
    labels = {
        "innings": np.repeat(np.asarray(STATE_INNINGS, dtype=np.int8), per_innings),
        "phase": np.tile(np.repeat([_PHASE_NAMES[code] for code in sorted(_PHASE_NAMES)], _N_COMP_TYPES), 2),
        "comp_type": np.tile(np.asarray(COMPETITION_TYPES, dtype=object), len(STATE_INNINGS) * _N_PHASES),
    }
    tables: list[pd.DataFrame] = []
    for role in ROLES:
        mask = masks[role]
        flags, missing = _nullable_flags(pressure[role], f"pressure[{role!r}]", n_rows)
        if (missing & mask).any():
            raise ValueError(f"pressure[{role!r}] has {_rows(int((missing & mask).sum()))} of the Role without a flag")
        flagged = flags & mask
        if (flagged & ~live_flags).any():
            raise ValueError(
                f"pressure[{role!r}] flags {_rows(int((flagged & ~live_flags).sum()))} that are not Live_Balls; "
                "a Pressure_Ball is a Live_Ball (Requirement 13, criterion 3)"
            )
        n_deliveries = np.bincount(cell[mask], minlength=_N_CELLS)
        n_live = np.bincount(cell[mask & live_flags], minlength=_N_CELLS)
        n_pressure = np.bincount(cell[flagged], minlength=_N_CELLS)
        innings_pressure = np.repeat(n_pressure.reshape(len(STATE_INNINGS), per_innings).sum(axis=1), per_innings)
        tables.append(
            pd.DataFrame(
                {
                    "role": pd.array([role] * _N_CELLS, dtype="str"),
                    "innings": labels["innings"],
                    "phase": pd.array(labels["phase"].astype(str), dtype="str"),
                    "comp_type": pd.array(labels["comp_type"].astype(str), dtype="str"),
                    "n_deliveries": n_deliveries.astype(np.int64),
                    "n_live": n_live.astype(np.int64),
                    "n_pressure": n_pressure.astype(np.int64),
                    "pct_of_role_deliveries": _percent(n_pressure, n_deliveries),
                    "pct_of_innings_pressure": _percent(n_pressure, innings_pressure),
                },
                columns=list(SHARE_COLUMNS),
            )
        )
    return tables[0], tables[1]


def _percent(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """100 x numerator / denominator, NaN where the denominator is 0 (a blank field in the CSV)."""
    out = np.full(len(numerator), np.nan)
    has = denominator > 0
    out[has] = 100.0 * numerator[has] / denominator[has]
    return out


def pressure_summary(table: pd.DataFrame, *, roles: Mapping[str, Any], innings: Any) -> pd.DataFrame:
    """The Live_Balls and Pressure_Balls of each PI and Innings_Group, from the pressure table.

    ``table`` holds ``live`` and the PI_COLUMNS and PRESSURE_FLAG_COLUMNS of the pressure table;
    ``roles`` and ``innings`` give each row's Roles and Innings_Group. Returns SUMMARY_COLUMNS, one
    row per PI (``pi_bat``, ``pi_bowl``, ``pi_bowl_extras_free``) and Innings_Group (1, then 2):
    ``role``; the Role's Deliveries, Live_Balls and Pressure_Balls (int64); ``pct_of_live``, 100 x
    Pressure_Balls / Live_Balls; ``lowest_pressure_pi``, the lowest PI of a Pressure_Ball, the cut
    that D3 gives; ``highest_ordinary_live_pi``, the highest PI of a Live_Ball that is an
    Ordinary_Ball, always below the cut (NaN where none). Raises the errors of the argument checks.
    """
    _require(table, ("live", *PI_COLUMNS, *PRESSURE_FLAG_COLUMNS), "table")
    n_rows = len(table)
    live_flags = _flags(table["live"], "table column 'live'", n_rows)
    masks = _role_flags(roles, n_rows)
    number = _innings_of(innings, n_rows)
    records = []
    for index in _INDEXES:
        pi = _any_floats(table[index.pi], f"table column {index.pi!r}", n_rows)
        flags, _ = _nullable_flags(table[index.pressure], f"table column {index.pressure!r}", n_rows)
        for group in STATE_INNINGS:
            rows = masks[index.role] & (number == group)
            live_rows = rows & live_flags
            flagged = rows & flags
            ordinary = live_rows & ~flags
            n_live = int(live_rows.sum())
            records.append(
                {
                    "pi": index.pi,
                    "role": index.role,
                    "innings": group,
                    "n_deliveries": int(rows.sum()),
                    "n_live": n_live,
                    "n_pressure": int(flagged.sum()),
                    "pct_of_live": 100.0 * int(flagged.sum()) / n_live if n_live else np.nan,
                    "lowest_pressure_pi": float(pi[flagged].min()) if flagged.any() else np.nan,
                    "highest_ordinary_live_pi": float(pi[ordinary].max()) if ordinary.any() else np.nan,
                }
            )
    summary = pd.DataFrame.from_records(records, columns=list(SUMMARY_COLUMNS))
    summary["pi"] = summary["pi"].astype("str")
    summary["role"] = summary["role"].astype("str")
    summary["innings"] = summary["innings"].astype(np.int8)
    for name in ("n_deliveries", "n_live", "n_pressure"):
        summary[name] = summary[name].astype(np.int64)
    return summary


def li_by_over(li: Any, *, innings: Any, over: Any) -> pd.DataFrame:
    """The mean LI of the Deliveries in each over number of each innings (Requirement 11, criterion 13).

    One value per Delivery of innings 1 and 2: ``li``, the LI; ``innings``, 1 or 2; ``over``, the
    over number (``overid`` of ``balls``, 1 for the first Over; may be missing). Returns
    LI_BY_OVER_COLUMNS, 40 rows on a RangeIndex, innings 1 then 2 and the over numbers of OVERS in
    order: ``n_deliveries``, the Deliveries of that innings and over, Wides and No_Balls included
    (int64), and ``mean_li``, their mean LI (float64, NaN when there is none). A Delivery whose
    over number is missing or outside OVERS is left out. Raises ValueError for an LI that is
    missing or infinite, and TypeError or ValueError for arguments of the wrong kind or length.
    """
    values = _floats(li, "li")
    n_rows = len(values)
    missing = int(np.isnan(values).sum())
    if missing:
        raise ValueError(f"li has {missing:,} missing value(s); every Delivery of innings 1 and 2 has an LI")
    number = _innings_of(innings, n_rows)
    overs = _floats(over, "over", n_rows)
    first, last = OVERS[0], OVERS[-1]
    kept = np.isfinite(overs) & (overs >= first) & (overs <= last) & (np.floor(overs) == overs)
    cell = np.where(kept, (number - STATE_INNINGS[0]) * len(OVERS) + overs - first, 0).astype(np.int64)
    size = len(STATE_INNINGS) * len(OVERS)
    counts = np.bincount(cell[kept], minlength=size)
    totals = np.bincount(cell[kept], weights=values[kept], minlength=size)
    mean = np.full(size, np.nan)
    mean[counts > 0] = totals[counts > 0] / counts[counts > 0]
    return pd.DataFrame(
        {
            "innings": np.repeat(np.asarray(STATE_INNINGS, dtype=np.int8), len(OVERS)),
            "over": np.tile(np.asarray(OVERS, dtype=np.int8), len(STATE_INNINGS)),
            "n_deliveries": counts.astype(np.int64),
            "mean_li": mean,
        },
        columns=list(LI_BY_OVER_COLUMNS),
    )


# ------------------------------------------------------------------------------------- the files

def _write_text(path: Path, text: str) -> Path:
    """Write ``text`` to ``path`` in UTF-8 with ``\\n`` line ends, through ``<file>.tmp`` renamed in place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


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


def _write_li_chart(path: Path, table: pd.DataFrame) -> Path:
    """The mean LI of each over number 1-20 as a PNG, one series per innings (Requirement 11, criterion 13)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=_CHART_SIZE, dpi=CHART_DPI, layout="constrained")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    colours = {1: "#1f5fa8", 2: "#d95f02"}
    innings = table["innings"].to_numpy(dtype=np.int64)
    for number in STATE_INNINGS:
        part = table.iloc[np.flatnonzero(innings == number)]
        ax.plot(
            part["over"].to_numpy(dtype=np.int64),
            part["mean_li"].to_numpy(dtype=np.float64),
            marker="o", markersize=4, linewidth=1.6, color=colours[number],
            label=f"Innings {number} ({int(part['n_deliveries'].sum()):,} Deliveries)",
        )  # fmt: skip
    ax.axhline(1.0, color="#9a9a9a", linestyle=":", linewidth=1.0, label="Mean over all Deliveries of innings 1 and 2 (1)")
    ax.set_xticks(list(OVERS))
    ax.set_xlim(OVERS[0] - 0.5, OVERS[-1] + 0.5)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, color="#e6e6e6", linewidth=0.6)
    ax.set_xlabel("Over number")
    ax.set_ylabel("Mean Leverage_Index (LI)")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.set_title(
        "Mean Leverage_Index of the Deliveries in each over, by innings\n"
        "(Wides and No_Balls included; Out_Of_Fold predictions of the Primary_Models)",
        fontsize=10,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        figure.savefig(temporary, format="png", dpi=CHART_DPI, metadata={"Software": None})
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _shown_number(value: float) -> str:
    """A Config number as written in the Config: 0.5, 0.25, 0.05 (up to 6 decimals, no trailing zeros)."""
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text or "0"


#: How the definition file words each D4 scope: in the D4 row of its table, in the plain-language
#: definition, and after "the Role's Live_Balls" in a formula.
_SCOPE_TEXTS: Mapping[str, tuple[str, str, str]] = MappingProxyType(
    {
        D4_INNINGS_GROUP: (
            "within each Innings_Group (all first innings; all second innings)",
            "within each Innings_Group",
            "in the same Innings_Group",
        ),
        D4_POOLED: (
            "pooled over both Innings_Groups",
            "pooled over both Innings_Groups",
            "of both Innings_Groups together",
        ),
    }
)


def _scope_text(scope: str) -> tuple[str, str, str]:
    """The wording of the D4 scope in the definition file (_SCOPE_TEXTS); ValueError for another scope."""
    if not isinstance(scope, str) or scope not in _SCOPE_TEXTS:
        raise ValueError(f"the D4 percentile scope must be one of {', '.join(map(repr, D4_SCOPES))}; got {scope!r}")
    return _SCOPE_TEXTS[scope]


def write_definition(
    path: str | Path,
    *,
    live_band: Any,
    alpha: Any,
    share: Any,
    scope: str,
    weights: str,
    window: int,
    new_batter_cap: int,
    pooled_mean_li: float,
    primary_models: Mapping[str, str],
    summary: pd.DataFrame,
) -> Path:
    """Write the pressure definition file (Requirement 13, criterion 9; module docstring, "Pressure definition").

    ``path``: the file (DEFINITION_FILE under the docs folder in a Run). The Config values of the
    Run: ``live_band`` (D1), ``alpha`` (D2), ``share`` (D3), ``scope`` (D4), ``weights`` (D5),
    ``window`` (D6) and ``new_batter_cap`` (D7). The data-derived values of the Run:
    ``pooled_mean_li``, the pooled mean unscaled LI; ``primary_models``, the selected candidate of
    each model (``PrimaryPredictions.candidates``: ``M3-1``, ``M3-2``, ``M4``); ``summary``, the
    table of pressure_summary. Writes UTF-8 Markdown with ``\\n`` line ends through a temporary file
    renamed in place, with no LaTeX, ``$``, backslash or time stamp, and returns the path. Raises
    TypeError or ValueError for a value outside its range or a summary without its columns.
    """
    lower, upper = _live_band(live_band)
    weight = _alpha(alpha)
    top = _share(share)
    if weights != D5_EQUAL:
        raise ValueError(f"the D5 Stress_Pillar weights must be {D5_EQUAL!r}, the only weights defined; got {weights!r}")
    scope_row, scope_plain, same_group = _scope_text(scope)
    d6 = _setting_whole(window, "window (D6)", 1)
    d7 = _setting_whole(new_batter_cap, "new_batter_cap (D7)", 1)
    mean_li = _number(pooled_mean_li, "pooled_mean_li")
    if not mean_li > 0.0:
        raise ValueError(f"pooled_mean_li must be positive; got {mean_li!r}")
    if not isinstance(primary_models, Mapping) or set(primary_models) != set(model_choice.MODELS):
        raise ValueError(f"primary_models must name the selected candidate of each of {', '.join(model_choice.MODELS)}")
    _require(summary, SUMMARY_COLUMNS, "summary")

    a, one_minus_a = _shown_number(weight), _shown_number(1.0 - weight)
    cut = _shown_number(1.0 - top)
    pct = _shown_number(100.0 * top)
    low, high = _shown_number(lower), _shown_number(upper)
    models = ", ".join(
        f"{name} {_FAMILY_LABELS.get(str(primary_models[name]), str(primary_models[name]))}"
        for name in model_choice.MODELS
    )
    lines = [
        "# Pressure definition",
        "",
        "Written by the Pressure_Engine in the `pressure` stage of the Pipeline (Requirement 13, criterion 9). "
        "Every value below is the value of the Run that wrote this file.",
        "",
        "## Plain-language definition",
        "",
        "A ball is a pressure ball for a player when three things hold before it is bowled:",
        "",
        f"1. **The match is live.** The batting side's chance of winning is between {_shown_number(100 * lower)}% and "
        f"{_shown_number(100 * upper)}%, so the result is still open.",
        "2. **The ball can swing the result.** The change in the win probability that this one ball can cause "
        "(its leverage) is large compared with other live balls.",
        "3. **The situation is harder than usual for the player's Role.** The stress on the batter (scoreboard "
        "deficit, wickets lost, being new at the crease, being squeezed) or on the bowler (scoreboard threat, a set "
        "and strong batter, an over going badly, a wicket drought) is high compared with other live balls.",
        "",
        f"Leverage and stress are combined into one Pressure_Index (PI) with weight {a} on leverage and "
        f"{one_minus_a} on stress, so a ball can qualify through very high leverage with ordinary stress, or the "
        f"reverse. The {pct}% of live balls with the highest PI are the pressure balls, chosen separately for "
        f"batters and bowlers and {scope_plain}. Every other ball of the Role is an ordinary ball. "
        "PI uses only the match state before the ball, never what happened on it.",
        "",
        "## Formal definitions",
        "",
        "A Delivery is one row of `ball_by_ball.csv` in innings 1 or 2. A Batter_Delivery is a Legal_Ball or a "
        "No_Ball, credited to the striker (a Wide is not a Batter_Delivery); a Bowler_Delivery is any Delivery, "
        "credited to the bowler. An Innings_Group is all Deliveries with the same innings number across all "
        "matches. The Percentile_Rank of a value x within a reference set of n values is rank(x) = (number of "
        "reference values below x + 0.5 × number of reference values equal to x) ÷ n.",
        "",
        f"**Live_Ball (D1).** A Delivery is a Live_Ball when {low} ≤ WP ≤ {high}, where WP is the batting side's "
        "pre-ball win probability: the unrounded Out_Of_Fold_Prediction of the M3 Primary_Model of the Delivery's "
        "innings, after the terminal rules. Every other Delivery is a Dead_Ball.",
        "",
        "**Leverage_Index (LI).** LI = [Σ over the 9 outcome classes c of P(c) × |WP(state after c) − WP(state "
        "before the ball)|] ÷ m. P(c) is the M4 Primary_Model's probability of class c for an average batter and an "
        "average bowler (θ_bat and θ_bowl at Average_Skill); the state after c adds one Legal_Ball, the class's runs "
        "and, for a wicket class, the Post_Wicket_State; the WPs come from the M3 Primary_Model of the innings. m is "
        f"the mean of the bracketed quantity over all Deliveries of innings 1 and 2 (m = {mean_li:.6f} in this "
        "Run), so the mean LI is 1. The batter Role and the bowler Role of a Delivery share its LI.",
        "",
        "**Stress (D4, D5).** Stress_bat = 0.25 × (F19 + F20 + F21 + F22) on Batter_Deliveries and "
        "Stress_bowl = 0.25 × (F23 + F24 + F25 + F26) on Bowler_Deliveries. Each Stress_Pillar is a "
        "Percentile_Rank within its Pillar_Reference_Set, the Role's Live_Balls "
        f"{same_group}: F19, F21, F23 and F26 are the rank of their one Pillar_Component; F20, F22, F24 and F25 "
        "are the rank of the mean of their components' ranks among the same means of the reference set. A larger "
        "component always means more stress. A component that cannot be computed gets rank 0.5 and leaves the "
        "reference set. Dead_Balls are ranked against the same reference set.",
        "",
        "| Pillar | Role | Pillar_Component(s) |",
        "|---|---|---|",
        "| F19 scoreboard deficit | batter | innings 1: Par_Score at the pre-ball Legal_Balls bowled − team score "
        "(runs); innings 2: Required_Rate − Achievable_Rate (runs per over) |",
        f"| F20 wickets and depth | batter | (a) wickets fallen − the Training_Folds mean wickets fallen at the same "
        f"innings and Legal_Balls bowled; (b) Counted_Wickets in the last {d6} Legal_Balls (D6); (c) 1 − RBQ |",
        f"| F21 new at the crease | batter | {d7} − min(striker's Legal_Balls faced, {d7}) (D7) |",
        "| F22 squeeze | batter | (a) the striker's current run of Dot_Balls; (b) the team's Legal_Balls since the "
        "last Boundary |",
        f"| F23 scoreboard threat | bowler | innings 1: team run rate − Par_Rate over the last {d6} Legal_Balls (D6); "
        "innings 2: Achievable_Rate − Required_Rate (runs per over) |",
        "| F24 batter threat | bowler | (a) striker's Legal_Balls faced × θ_bat; (b) RBQ |",
        "| F25 over going badly | bowler | (a) team runs off the bowler earlier in the over, byes and leg-byes left "
        "out; (b) 1 when the previous Delivery of the over was a Boundary or an extra, else 0 |",
        "| F26 wicket drought | bowler | Legal_Balls since the last Counted_Wicket |",
        "",
        f"**Pressure_Index (PI, D2).** For a Live_Ball and a Role: PI = 100 × [α × rank(LI) + (1 − α) × "
        f"rank(Stress)] with α = {a}, both ranks taken among the Role's Live_Balls {same_group}, so 0 < PI < 100. "
        "For a Dead_Ball, PI = 0. A Wide has no batter-Role PI.",
        "",
        f"**Pressure_Ball (D3).** A Live_Ball whose rank(PI) among the Role's Live_Balls {same_group} is greater "
        f"than 1 − {_shown_number(top)} = {cut}: the top {pct}%. Deliveries with equal PI get the same flag. A "
        "Dead_Ball is never a Pressure_Ball. Every other Batter_Delivery or Bowler_Delivery of the Role is an "
        "Ordinary_Ball.",
        "",
        "**Extras_Free_PI (validation only).** The bowler-Role PI recomputed with a previous-Delivery flag that "
        "counts Boundaries only in F25(b), every other input unchanged. Its top "
        f"{pct}% re-selects bowler Pressure_Balls for the extras check of Requirement 14, criterion 5; it never "
        "feeds the ratings.",
        "",
        "## Design_Choices D1–D5 of this Run",
        "",
        "| ID | Choice | Value in this Run | Label |",
        "|---|---|---|---|",
        f"| D1 | Live band on WP | {low} ≤ WP ≤ {high} | hand-set (D1) |",
        f"| D2 | Leverage/Stress blend α | {a} | hand-set (D2) |",
        f"| D3 | Pressure share | {pct}% of Live_Balls per Role and Innings_Group | hand-set (D3) |",
        f"| D4 | Percentile scope | {scope_row} | hand-set (D4) |",
        "| D5 | Stress_Pillar weights | equal, 0.25 for each of the Role's four pillars | hand-set (D5) |",
        "",
        "## Values derived from the data in this Run",
        "",
        "| Value | This Run | Label |",
        "|---|---|---|",
        f"| m, the pooled mean unscaled LI | {mean_li:.6f} | data-derived (Pressure_Engine) |",
        f"| Primary_Models behind WP, LI and PI | {models} | data-derived (Model_Choice_Checker) |",
        "",
        "| PI | Innings_Group | Role Deliveries | Live_Balls | Pressure_Balls | Pressure_Balls as % of Live_Balls "
        "| Lowest PI of a Pressure_Ball |",
        "|---|---|---|---|---|---|---|",
    ]
    titles = {index.pi: index.title for index in _INDEXES}
    for record in summary.to_dict("records"):
        pct_live = "" if pd.isna(record["pct_of_live"]) else f"{float(record['pct_of_live']):.2f}"
        lowest = "" if pd.isna(record["lowest_pressure_pi"]) else f"{float(record['lowest_pressure_pi']):.3f}"
        lines.append(
            f"| {titles.get(str(record['pi']), str(record['pi']))} | {int(record['innings'])} | "
            f"{int(record['n_deliveries']):,} | {int(record['n_live']):,} | {int(record['n_pressure']):,} | "
            f"{pct_live} | {lowest} |"
        )
    text = "\n".join(lines) + "\n"
    if "\\" in text or "$" in text:
        raise ValueError("the pressure definition text holds a backslash or a '$', which the Report_Check rejects")
    return _write_text(Path(path), text)


# ===================================================================== the pressure Pipeline_Stage

def _aligned_balls(balls: pd.DataFrame, row_ids: np.ndarray) -> pd.DataFrame:
    """The _STAGE_BALL_COLUMNS of ``balls`` for each ``state`` row, in ``state`` order, on a RangeIndex."""
    _require(balls, _STAGE_BALL_COLUMNS, "balls")
    index = pd.Index(_ints(balls["row"], "balls column 'row'"))
    if not index.is_unique:
        raise ValueError("balls must hold each row once")
    at = index.get_indexer(row_ids)
    if (at < 0).any():
        raise ValueError(f"{_rows(int((at < 0).sum()))} of state have no Delivery in balls; pass the tables of one Run")
    return balls.iloc[at][list(_STAGE_BALL_COLUMNS)].reset_index(drop=True)


def _pressure_frame(
    row_ids: np.ndarray,
    wp: np.ndarray,
    live: np.ndarray,
    lev: LeverageIndex,
    components: pd.DataFrame,
    pillar_ranks: pd.DataFrame,
    stresses: pd.DataFrame,
    pis: Mapping[str, np.ndarray],
    flags: Mapping[str, pd.arrays.BooleanArray],
) -> pd.DataFrame:
    """The pressure table (PRESSURE_COLUMNS): nullable ``boolean`` batter flags, ``bool`` bowler flags."""
    for frame, what in ((components, "components"), (pillar_ranks, "pillar_ranks"), (stresses, "stresses")):
        _same_rows(frame, row_ids, what)
    data: dict[str, Any] = {
        "row": row_ids.astype(np.int32),
        "wp_pre": np.array(wp, dtype=np.float64),
        "live": np.array(live, dtype=bool),
        "li_raw": np.array(lev.unscaled, dtype=np.float64),
        "li": np.array(lev.li, dtype=np.float64),
    }
    for name in COMPONENT_COLUMNS:
        data[name] = components[name].to_numpy(dtype=np.float64)
    for name in PILLAR_COLUMNS:
        data[name] = pillar_ranks[name].to_numpy(dtype=np.float64)
    for name in STRESS_COLUMNS:
        data[name] = stresses[name].to_numpy(dtype=np.float64)
    for index in _INDEXES:
        data[index.pi] = pis[index.pi]
        flag = flags[index.pressure]
        names = [(index.pressure, flag)]
        if index.ordinary is not None:
            names.append((index.ordinary, ordinary_balls(flag)))
        for name, values in names:
            # The batter Role has no flag on a Wide (<NA>); every Delivery is a Bowler_Delivery.
            data[name] = values.copy() if index.role == BATTER else values.to_numpy(dtype=bool)
    return pd.DataFrame(data, columns=list(PRESSURE_COLUMNS))


def _check_pressure_table(table: pd.DataFrame, roles: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    """Check the PI and flag rules of Requirement 13, criteria 3-5 and 11 on the pressure table; return the ranges.

    Raises ValueError, naming the rule, when a row breaks one: a Dead_Ball of the Role with a PI
    other than 0 or a Pressure_Ball flag; a Live_Ball of the Role with a PI outside (0, 100]; a row
    outside the Role with a PI or a flag; a Role Delivery that is not exactly one of a Pressure_Ball
    or an Ordinary_Ball.
    """
    live = table["live"].to_numpy(dtype=bool)
    ranges: dict[str, dict[str, Any]] = {}
    for index in _INDEXES:
        mask = roles[index.role]
        pi = table[index.pi].to_numpy(dtype=np.float64)
        flags, missing = _nullable_flags(table[index.pressure], index.pressure, len(table))
        dead, on = mask & ~live, mask & live
        problems = []
        if not (pi[dead] == 0.0).all() or flags[dead].any():
            problems.append("a Dead_Ball has a PI other than 0 or a Pressure_Ball flag (criterion 3)")
        if not ((pi[on] > 0.0) & (pi[on] <= _PI_SCALE)).all():
            problems.append("a Live_Ball has a PI outside (0, 100] (criterion 2)")
        if not np.isnan(pi[~mask]).all() or not missing[~mask].all() or missing[mask].any():
            problems.append("a row outside the Role has a PI or a flag, or a Role Delivery has no flag (criterion 11)")
        if index.ordinary is not None:
            ordinary, ordinary_missing = _nullable_flags(table[index.ordinary], index.ordinary, len(table))
            if not np.array_equal(ordinary_missing, missing) or (ordinary[mask] == flags[mask]).any():
                problems.append("a Role Delivery is not exactly one of a Pressure_Ball or an Ordinary_Ball (criterion 5)")
        if problems:
            raise ValueError(f"the {index.title} PI breaks Requirement 13: {'; '.join(problems)}")
        ranges[index.pi] = {
            "n_dead": int(dead.sum()),
            "n_live": int(on.sum()),
            "n_outside": int((~mask).sum()),
            "live_min": float(pi[on].min()) if on.any() else math.nan,
            "live_max": float(pi[on].max()) if on.any() else math.nan,
        }
    return ranges


def run_stage(ctx: StageContext) -> None:
    """The ``pressure`` Pipeline_Stage (module docstring, "The ``pressure`` Pipeline_Stage").

    Raises PressureInputError (exit code 5) when WP, LI or a Role's Stress is missing or not finite
    where PI needs it, before any PI is computed or any file is written (Requirement 13, criterion
    10); StageError (exit code 5) for a checkpoint file that is missing or changed; and ValueError
    or TypeError for input tables that do not fit together or a Config value outside its range.
    """
    started = time.perf_counter()
    settings, store, log = ctx.settings, ctx.store, ctx.log
    band = _live_band(settings.get(_LIVE_BAND))
    alpha = _alpha(settings.get(_ALPHA))
    share = _share(settings.get(_SHARE))
    scope = settings.get(_SCOPE)
    _scope_text(scope)  # checks D4 before any table is read
    weights = settings.get(_WEIGHTS)
    if weights != D5_EQUAL:
        raise ValueError(f"the D5 Stress_Pillar weights must be {D5_EQUAL!r}, the only weights defined; got {weights!r}")
    window = _setting_whole(settings.get(_WINDOW), "D6_window", 1)
    cap = _setting_whole(settings.get(_CAP), "D7_new_batter_cap", 1)
    ks = sorted(spec.sandbox_k for spec in folds.sandbox_plan(settings.get(_N_FOLDS)) if spec.kind == folds.OUTER)

    # 1. The Deliveries of innings 1 and 2 in state order, and the Primary_Models' predictions only
    #    (Requirement 23, criterion 9), checked against them.
    balls = store.load_table(RECONSTRUCT_STAGE, BALLS_TABLE)
    state = store.load_table(STATE_STAGE, STATE_TABLE)
    fold_table = store.load_table(folds.STAGE_NAME, folds.FOLD_TABLE)
    primary = model_choice.load_primary(store, variants=(LI_VARIANT,))
    row_ids = _ints(state["row"], "state column 'row'")
    rows = _aligned_balls(balls, row_ids)
    del balls
    fold = folds.fold_of(rows["match_id"], fold_table)
    innings = _innings_of(rows["innings"])
    if not (
        np.array_equal(primary.row, row_ids)
        and np.array_equal(primary.innings, innings)
        and np.array_equal(primary.fold, fold)
    ):
        raise ValueError(
            "the Primary_Models' predictions do not hold the state rows in order with the innings of balls and the "
            "folds of the Fold_Table; pass the checkpoints of one Run"
        )

    # 2. Live_Balls from the unrounded Out_Of_Fold WP (D1) and the Leverage_Index (Requirement 11, 9-11).
    wp = np.array(primary.wp_pre, dtype=np.float64)
    live = live_balls(wp, band)
    lev = primary_leverage(primary)
    candidates = dict(primary.candidates)
    del primary

    # 3. Pillar_Components, Stress_Pillars and Stress (Requirement 12), each row from its own fold's view.
    view_by_k = {k: store.load_table(remaining_runs.STAGE_NAME, views.view_table(k)) for k in ks}
    lb = state["lb_bowled"]
    means = train_wicket_means(
        fold=fold, innings=innings, lb_bowled=lb, wickets=state["wickets"], training=training_mask(rows)
    )
    components = pillar_components(
        state, own_fold_view(view_by_k, fold, row_ids), innings=innings,
        mean_wickets=wicket_means_for(means, fold=fold, innings=innings, lb_bowled=lb), new_batter_cap=cap,
    )  # fmt: skip
    del view_by_k
    roles = role_masks(rows["cls"])
    groups = rank_groups(innings, scope)
    pillar_ranks = pillars(components, live=live, roles=roles, groups=groups)
    stresses = stress(pillar_ranks, weights=weights)
    counts = missing_components(components, live=live, roles=roles, innings=innings)

    # 4. The finiteness guard (Requirement 13, criterion 10), before any PI is computed or written.
    finite_guard(wp=wp, li=lev.li, stresses=stresses, live=live, roles=roles, innings=innings)

    # 5. PI (D2) and Pressure_Ball flags (D3) of both Roles, Ordinary_Balls, and the Extras_Free_PI.
    pis: dict[str, np.ndarray] = {}
    flags: dict[str, pd.arrays.BooleanArray] = {}
    for index in _INDEXES:
        mask = roles[index.role]
        pis[index.pi] = pressure_index(lev.li, stresses[index.stress], live, mask, groups, alpha)
        flags[index.pressure] = pressure_flags(pis[index.pi], live, mask, groups, share)
    table = _pressure_frame(row_ids, wp, live, lev, components, pillar_ranks, stresses, pis, flags)
    ranges = _check_pressure_table(table, roles)
    del components, pillar_ranks
    phase = _ints(state["phase"], "state column 'phase'", len(row_ids))
    shares = dict(
        zip(
            ROLES,
            share_tables(
                pressure={index.role: flags[index.pressure] for index in _MAIN_INDEXES}, live=live, roles=roles,
                innings=innings, phase=phase, comp_type=rows["comp_type"],
            ),  # fmt: skip
            strict=True,
        )
    )
    summary = pressure_summary(table, roles=roles, innings=innings)
    over = rows["overid"]
    by_over = li_by_over(lev.li, innings=innings, over=over)

    # 6. The checkpoint.
    saved: list[str] = []

    def save(name: str, frame: pd.DataFrame) -> None:
        store.save_table(STAGE_NAME, name, frame)
        saved.append(name)

    save(PRESSURE_TABLE, table)
    for role in ROLES:
        save(SHARE_TABLES[role], shares[role])
    save(SUMMARY_TABLE, summary)
    save(LI_BY_OVER_TABLE, by_over)
    save(MISSING_COMPONENTS_TABLE, counts)

    # 7. The files: outputs/pressure/ and docs/pressure_definition.md.
    folder = settings.path("outputs") / PRESSURE_FOLDER
    files = [write_missing_components(folder, counts)]
    for role in ROLES:
        files.append(_write_csv(folder / SHARE_FILES[role], shares[role]))
    files.append(_write_csv(folder / SUMMARY_FILE, summary))
    files.append(_write_csv(folder / LI_BY_OVER_FILE, by_over))
    files.append(_write_li_chart(folder / LI_BY_OVER_CHART_FILE, by_over))
    files.append(
        write_definition(
            settings.path("docs") / DEFINITION_FILE, live_band=band, alpha=alpha, share=share, scope=scope,
            weights=weights, window=window, new_batter_cap=cap, pooled_mean_li=lev.pooled_mean,
            primary_models=candidates, summary=summary,
        )  # fmt: skip
    )

    # 8. The data-derived Method_Constants.
    main = summary.iloc[np.flatnonzero(summary["pi"].isin([index.pi for index in _MAIN_INDEXES]).to_numpy())]

    def per_role(column: str) -> dict[str, dict[str, float]]:
        return {
            role: {
                str(int(record["innings"])): float(record[column])
                for record in main.to_dict("records")
                if record["role"] == role
            }
            for role in ROLES
        }

    ctx.constants.record(POOLED_MEAN_LI_CONSTANT, float(lev.pooled_mean), _COMPONENT)
    ctx.constants.record(SHARE_CONSTANT, per_role("pct_of_live"), _COMPONENT)
    ctx.constants.record(PI_CUTOFF_CONSTANT, per_role("lowest_pressure_pi"), _COMPONENT)

    _log_stage(
        ctx,
        table=table,
        roles=roles,
        innings=innings,
        phase=phase,
        over=over,
        lev=lev,
        candidates=candidates,
        counts=counts,
        ranges=ranges,
        shares=shares,
        summary=summary,
        by_over=by_over,
        settings_text=(
            f"D1 live band [{_shown_number(band[0])}, {_shown_number(band[1])}], D2 alpha {_shown_number(alpha)}, "
            f"D3 share {_shown_number(share)}, D4 {scope}, D5 {weights}, D6 {window}, D7 {cap}"
        ),
        files=files,
        saved=saved,
        seconds=time.perf_counter() - started,
    )


def _log_stage(
    ctx: StageContext,
    *,
    table: pd.DataFrame,
    roles: Mapping[str, np.ndarray],
    innings: np.ndarray,
    phase: np.ndarray,
    over: Any,
    lev: LeverageIndex,
    candidates: Mapping[str, str],
    counts: pd.DataFrame,
    ranges: Mapping[str, Mapping[str, Any]],
    shares: Mapping[str, pd.DataFrame],
    summary: pd.DataFrame,
    by_over: pd.DataFrame,
    settings_text: str,
    files: Sequence[Path],
    saved: Sequence[str],
    seconds: float,
) -> None:
    log = ctx.log
    live = table["live"].to_numpy(dtype=bool)
    li = table["li"].to_numpy(dtype=np.float64)
    models = ", ".join(f"{name} {candidates[name]}" for name in model_choice.MODELS if name in candidates)
    live_counts = "; ".join(
        f"innings {group}: {int((live & (innings == group)).sum()):,} of {int((innings == group).sum()):,}"
        for group in STATE_INNINGS
    )
    log.info(
        f"Pressure_Engine inputs: the Primary_Models' predictions only (Requirement 23, criterion 9: {models}); "
        f"{settings_text}; Live_Balls (unrounded Out_Of_Fold WP within the D1 band, Requirement 13, criterion 1): "
        f"{live_counts}; pooled mean unscaled LI {lev.pooled_mean:.6f} (mean LI {float(li.mean()):.6f}, "
        f"{int((lev.unscaled == 0.0).sum()):,} Deliveries with LI 0, {int(((lev.unscaled == 0.0) & live).sum()):,} "
        "of them Live_Balls)",
    )
    missing = counts.iloc[np.flatnonzero(counts["n_missing"].to_numpy() > 0)]
    if len(missing):
        listed = ", ".join(
            f"{record['component']} innings {int(record['innings'])}: {int(record['n_missing']):,} "
            f"({int(record['n_missing_live']):,} Live_Balls)"
            for record in missing.to_dict("records")
        )
        log.warning(
            f"Pillar_Components missing, ranked 0.5 and left out of the Pillar_Reference_Set (Requirement 12, "
            f"criterion 16): {listed}; counts in {PRESSURE_FOLDER}/{MISSING_COMPONENTS_FILE}",
        )
    band = ctx.settings.get(_SHARE_BAND)
    lines = []
    outside = []
    for record in summary.to_dict("records"):
        pct = float(record["pct_of_live"])
        lines.append(
            f"{record['pi']} innings {int(record['innings'])}: {int(record['n_pressure']):,} of "
            f"{int(record['n_live']):,} Live_Balls ({pct:.3f}%; PI cut {float(record['lowest_pressure_pi']):.4f}, "
            f"highest Ordinary Live_Ball PI {float(record['highest_ordinary_live_pi']):.4f})"
        )
        if record["pi"] != "pi_bowl_extras_free" and not (100.0 * band[0] <= pct <= 100.0 * band[1]):
            outside.append(f"{record['pi']} innings {int(record['innings'])} ({pct:.3f}%)")
    log.info(
        "Pressure_Balls (rank of PI among the Role's Live_Balls of the rank group > 1 - D3; Requirement 13, "
        f"criterion 4): {'; '.join(lines)}",
    )
    if outside:
        log.warning(
            f"the Pressure_Ball share of the Live_Balls lies outside the Test_Suite band {band} (Requirement 13, "
            f"criterion 7) for {', '.join(outside)}",
        )
    checks = "; ".join(
        f"{name}: {values['n_dead']:,} Dead_Balls with PI 0, {values['n_live']:,} Live_Balls with PI in "
        f"[{values['live_min']:.4f}, {values['live_max']:.4f}], {values['n_outside']:,} rows outside the Role "
        "without a PI or flag"
        for name, values in ranges.items()
    )
    log.info(f"PI checks (Requirement 13, criteria 2, 3, 5 and 11) passed: {checks}")
    bowler_live = roles[BOWLER] & live
    main = table["pressure_bowl"].to_numpy(dtype=bool)
    free = table["pressure_bowl_extras_free"].to_numpy(dtype=bool)
    agree = float((main == free)[bowler_live].mean()) if bowler_live.any() else math.nan
    both = int((main & free).sum())
    log.info(
        f"Extras_Free_PI (validation only, Requirement 14, criterion 5): its bowler flags agree with the main bowler "
        f"flags on {100.0 * agree:.2f}% of the bowler Live_Balls; {both:,} of the {int(main.sum()):,} bowler "
        f"Pressure_Balls ({100.0 * both / max(int(main.sum()), 1):.2f}%) are also Extras_Free Pressure_Balls",
    )
    death = (innings == STATE_INNINGS[1]) & (phase == int(Phase.DEATH))
    dead_mean = float(li[death & ~live].mean()) if (death & ~live).any() else math.nan
    live_mean = float(li[death & live].mean()) if (death & live).any() else math.nan
    message = (
        f"innings-2 death-Phase Deliveries (Requirement 11, criterion 12): {int((death & ~live).sum()):,} Dead_Balls "
        f"with mean LI {dead_mean:.4f}, {int((death & live).sum()):,} Live_Balls with mean LI {live_mean:.4f}"
    )
    if dead_mean < live_mean:
        log.info(message)
    else:
        log.warning(message + "; the Dead_Balls' mean LI is not lower")
    overs = _floats(over, "over")
    left_out = int((~(np.isfinite(overs) & (overs >= OVERS[0]) & (overs <= OVERS[-1]))).sum())
    series = []
    for group in STATE_INNINGS:
        part = by_over.iloc[np.flatnonzero(by_over["innings"].to_numpy() == group)]
        values = " ".join(f"{value:.3f}" for value in part["mean_li"].to_numpy(dtype=np.float64))
        series.append(f"innings {group}: {values}")
    log.info(
        f"mean LI by over 1-{OVERS[-1]} (Requirement 11, criterion 13): {'; '.join(series)}"
        + (f"; {left_out:,} Deliveries without an over number in 1-{OVERS[-1]} left out" if left_out else ""),
    )
    for role in ROLES:
        part = shares[role]
        cells = ", ".join(
            f"{int(record['innings'])}/{record['phase']}/{record['comp_type']} {int(record['n_pressure']):,}/"
            f"{int(record['n_deliveries']):,}"
            for record in part.to_dict("records")
        )
        log.info(f"{role} share table (innings/Phase/Competition_Type Pressure_Balls/Role Deliveries): {cells}")
    log.info(
        f"wrote {', '.join(_shown_path(path, ctx.root) for path in files)}; saved {', '.join(saved)} in {seconds:.1f} s",
    )


def _shown_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return str(path)
