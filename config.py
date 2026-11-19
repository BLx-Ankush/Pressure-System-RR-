"""Config of the Pressure Rating System (SupeRR Selector Season 2, Round 1).

This file is the single Config (Requirement 1, criterion 3). The Pipeline takes every
Design_Choice value, model hyperparameter, sensitivity grid, random seed, Stage-2 flag, the
extras_type class mapping and the exclude_failing_matches option from CONFIG below. Source
code holds no fallback value for any of them.

superr.settings.load_settings(root) executes this file by its path, checks CONFIG against
superr.settings.REQUIRED (every required key present with the right type, no unlisted key,
label metadata for every Method_Constant) and records the SHA-256 of this file's bytes in the
Run_Log. A key added here must also be added to REQUIRED; a new Method_Constant also needs an
entry in constants_meta at the bottom of this file.

Paths are relative to the workspace root, the folder that holds this file.
"""

CONFIG = {
    # ------------------------------------------------------------------------------- paths
    # The six Source_CSVs in the workspace root (Requirement 1, criterion 2), not the copies in
    # the nested problem-statement folder, and the only two folders the Pipeline writes (1.5).
    "paths": {
        "ball_by_ball": "ball_by_ball.csv",
        "matches": "matches.csv",
        "playing_eleven": "playing_eleven.csv",
        "competitions": "competitions.csv",
        "teams": "teams.csv",
        "mapping": "competition_team_player_mapping.csv",
        "outputs": "outputs",
        "docs": "docs",
    },
    # ---------------------------------------------------------------------- design_choices
    # Design_Choices D1-D20 at the defaults of the Design choices table in requirements.md
    # (Requirement 1, criterion 12). Every key starts with its D-number.
    "design_choices": {
        "D1_live_band": [0.05, 0.95],          # Live_Ball when lower <= WP <= upper (13.1)
        "D2_alpha": 0.5,                       # PI = 100 x [alpha x rank(LI) + (1 - alpha) x rank(Stress)] (13.2)
        "D3_pressure_share": 0.25,             # top 25% of Live_Balls per Role and Innings_Group (13.4)
        "D4_scope": "innings_group",           # percentile ranks within each Innings_Group (12.12); "pooled" is the 21.4 case
        "D5_weights": "equal",                 # Stress = equal-weighted mean of the Role's four pillars (12.13)
        "D6_window": 18,                       # recent-wickets and recent-run-rate window, Legal_Balls (6.7, 12.3, 12.6)
        "D7_new_batter_cap": 20,               # F21 = cap - min(striker Legal_Balls faced, cap) (12.4)
        "D8_phase_splits": [0.30, 0.75],       # powerplay: share <= 0.30; middle: <= 0.75; death: above (6.5)
        "D9_half_life_years": 3.0,             # Recency_Weight half-life; None means no decay (21.2 case only)
        "D10_knockout_weight": 1.25,           # Stakes_Weight of a match with knock_out = 1; other matches 1.0
        "D11_min_balls_grid": [45, 60, 75, 90],  # candidate minimum Pressure_Ball counts (17.6-17.8)
        "D11_target_eligible": 60,             # select the largest grid value leaving >= 60 Eligible_Players (17.7)
        "D12_replacement_pct": 25,             # Replacement_Level percentile of unshrunk Clutch_Level (18.6)
        "D13_cells": "innings_phase",          # Usual_Baseline matching cells (16.4); "innings_phase_comp_type" is the 21.4 case
        # D14 credit rules (Requirement 15, criteria 8, 9 and 12)
        "D14_bowler_wicket_codes": [1, 2, 3, 5, 6],  # bowled, caught, LBW, stumped, hit wicket: the only bowler credit
        "D14_run_out_code": 4,                 # no bowler credit; charged to the batter who is out (dismissed_player_id)
        "D14_retire_out_code": 9,              # Counted_Wicket for match state (5.8); no bowler credit
        "D14_retire_out_batter_penalty": False,  # False: no Omega penalty, no Dismissal_Charge (15.12); True is the 21.4 case
        "D15_n_folds": 5,                      # Match_Grouped_Folds (7.1)
        "D16_n_boot": 2000,                    # Bootstrap_Resamples (19.1)
        "D17_wicket_buckets": [[0, 2], [3, 5], [6, 9]],  # Wickets_Bucket: inclusive ranges of wickets fallen (8.2)
        "D18_min_matches_grid": [5, 8, 10],    # minimum Pressure_Match counts in the eligibility grid (17.6)
        "D18_min_matches_default": 8,          # minimum Pressure_Match count applied (17.2)
        "D19_period_minimum": 30,              # unweighted Pressure_Balls needed in each period (20.2)
        "D20_score_tolerance": 6,              # Score_Tolerance in runs for match exclusion (5.24)
    },
    # -------------------------------------------------------------------------- data_rules
    "data_rules": {
        # Delivery class of each extras_type value, matched as exact, case-sensitive text; "None"
        # is a value, not a missing value (Requirement 5, criteria 2-3; Assumption 10).
        "extras_type_classes": {
            "legal": ["None", "legbyes", "byes", "Penalty"],
            "wide": ["wide"],
            "no_ball": ["other", "Overstep"],
        },
        # Candidate Extras_Encodings E1-E4 (design DD-4) in pass-rate table order; F30 first (5.7)
        "candidate_encodings": ["F30", "byes_one", "byes_one_nb_plain", "byes_one_penalty5"],
        "exclude_failing_matches": True,       # Score_Tolerance exclusion of matches (5.24)
        "score_tolerance_grid": [0, 3, 6, 10],  # tolerances at which audit.md reports excluded counts; reporting only (5.27)
        # Full_Member team names, compared exactly with teams.team_name ("India A" is not one)
        "full_members": [
            "Afghanistan", "Australia", "Bangladesh", "England", "India", "Ireland",
            "New Zealand", "Pakistan", "South Africa", "Sri Lanka", "West Indies", "Zimbabwe",
        ],
        # Recent_Activity_Rule switch (Requirement 25, criterion 10; Assumption 1), on as delivered.
        # While it is on, each Run also builds the Comparison_Lists with the rule off (25.11).
        "recent_activity_rule": True,
        "recent_years": [2025, 2026],          # Calendar_Years that satisfy the rule (17.4)
        "min_major_share": 0.5,                # >= 50% of Pressure_Balls from IPL or Full_Member_Matches (17.5)
        "min_calendar_years": 2,               # Deliveries in at least 2 distinct Calendar_Years (17.3)
    },
    # ------------------------------------------------------------------------------- seeds
    # One named seed per random step (Requirement 28, criterion 3). Early-stopping splits draw
    # whole matches with PCG64(model seed + fit index).
    "seeds": {
        "folds": 11,
        "m1": 12,
        "m2": 13,
        "m3_lgbm": 14,
        "m3_logistic": 15,
        "m3_xgb": 16,
        "m3_rf": 17,
        "m4_lgbm": 18,
        "m4_xgb": 19,
        "m4_rf": 20,
        "extras": 21,
        "bootstrap": 22,
    },
    # ----------------------------------------------------------------------------- runtime
    # Thread count and deterministic mode for every LightGBM and XGBoost fit (28.3); random
    # forests use the same thread count.
    "runtime": {
        "n_threads": 4,
        "deterministic": True,
    },
    # ------------------------------------------------------------------------------ models
    # m2, m3 and m4 use LightGBM parameter names. max_rounds caps the boosting rounds; es_share
    # is the share of training matches held out, as whole matches, for early stopping, and
    # es_patience the early-stopping patience in rounds (10.6, 10.7).
    "models": {
        "m1": {                                # sparse ridge (Requirement 8)
            "ridge_grid": [1, 3, 10, 30, 100, 300, 1000, 3000, 10000],  # penalty candidates (8.3)
            "prob_clip": [1e-6, 0.999999],     # dismissal-probability bounds before log loss (8.8)
        },
        "m2": {                                # LightGBM regression, capped and first-innings variants (Requirement 9)
            "objective": "poisson",            # remaining runs are a count; Poisson fits both ends of the innings better than L2 ("regression")
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 200,
            "lambda_l2": 1.0,
            "max_rounds": 600,
            "es_share": 0.2,
            "es_patience": 30,
            "monotone_constraints_method": "intermediate",  # 9.4
        },
        "m3": {                                # LightGBM binary per innings (Requirement 10)
            # About 850 match labels, so M3 is regularised hard, in both innings: a 5,000-row leaf
            # holds the Deliveries of about 40 matches or more (at least 37), so no leaf can isolate
            # single matches through the per-match features tsg, scoring_env and venue_effect.
            "learning_rate": 0.05,
            "num_leaves": 7,
            "min_data_in_leaf": 5000,
            "lambda_l2": 5.0,
            "max_rounds": 600,
            "es_share": 0.2,
            "es_patience": 30,
            "monotone_constraints_method": "intermediate",  # 10.5
            # [lower, upper] output of every fold's centred isotonic calibrator, for every Candidate_Model
            # family (10.9). Unbounded isotonic regression gives exactly 0 or 1 at its ends, so one
            # comeback match would cost about 34.5 log loss per Delivery. The bounds leave Live_Ball
            # (D1, 0.05-0.95) and the terminal rules (exact 0, 0.5 or 1) unchanged.
            "calibration_bounds": [0.001, 0.999],
            "logistic": {                      # logistic baseline per innings (10.10)
                "C": 1.0,
                "max_iter": 2000,
            },
        },
        "m4": {                                # LightGBM multiclass, 9 classes (Requirement 11)
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 300,
            "lambda_l2": 1.0,
            "max_rounds": 400,
            "es_share": 0.1,
            "es_patience": 20,
        },
        "extras": {                            # multinomial logistic: Legal_Ball / Wide / No_Ball (design DD-11)
            "C": 1.0,
            "max_iter": 2000,
        },
        # Alternative_Models (Requirement 23): one setting per compared model, shared by both M3
        # innings. Round caps, early stopping and the L2 penalty match the LightGBM model each
        # alternative is compared with.
        "xgb": {
            "m3": {
                "tree_method": "hist",
                "eta": 0.05,
                "max_depth": 4,
                "min_child_weight": 50,
                "reg_lambda": 5.0,
                "max_rounds": 600,
                "es_share": 0.2,
                "es_patience": 30,
            },
            "m4": {
                "tree_method": "hist",
                "eta": 0.05,
                "max_depth": 4,
                "min_child_weight": 20,
                "reg_lambda": 1.0,
                "max_rounds": 400,
                "es_share": 0.1,
                "es_patience": 20,
            },
        },
        "rf": {
            "m3": {
                "n_estimators": 300,
                "max_depth": 10,
                "min_samples_leaf": 200,
                "max_features": "sqrt",
            },
            "m4": {
                "n_estimators": 300,
                "max_depth": 10,
                "min_samples_leaf": 100,
                "max_features": "sqrt",
            },
        },
    },
    # ------------------------------------------------------------------------- model_choice
    # Primary_Model selection policy of the Model_Choice_Checker (Requirement 23, criteria 7-9 and
    # 14; design decision DD-17). The primary criterion is the lowest pooled Held_Out log loss
    # among the non-failed Candidate_Models. On top of it, a calibration-aware tie-break: for a
    # model named in calibration_preferred, when the named family is non-failed and its pooled log
    # loss lies within the relative band calibration_band of the lowest, it is selected, because it
    # is the family whose own-theta NBV per innings x Phase cell passes the Requirement 15,
    # criterion 17 calibration check that feeds the ratings (M4 -> NBV -> Final_Score). A model
    # with no entry keeps the plain lowest-log-loss rule with the tie order, so M3 is unchanged
    # (logistic in both innings). The Value_Engine re-verifies the tie-break on its own inputs
    # (Requirement 23, criterion 14).
    "model_choice": {
        # Comparison-table model label -> preferred Candidate_Model family. M4 prefers LightGBM
        # ("lgbm"); M3-1 and M3-2 are absent, so their selection is the plain lowest log loss.
        "calibration_preferred": {"M4": "lgbm"},
        # Relative log-loss band delta: a preferred family within delta x (lowest log loss) of the
        # lowest is selected. 0.005 (0.5%) covers the delivered M4 XGBoost -> LightGBM gap of
        # 0.001224 (0.0826% of 1.481957) about six times over, yet stays far below any log-loss
        # difference a model-family change has produced here (the M3 family gaps are 2-30%).
        "calibration_band": 0.005,
    },
    # ------------------------------------------------------------------------- sensitivity
    # Requirement 21 grids: one Robustness_Case per listed value, one Design_Choice at a time.
    # Each comment names the setting that the case overrides.
    "sensitivity": {
        "alpha": [0.0, 0.33, 0.5, 0.67, 1.0],  # design_choices.D2_alpha (21.2)
        "share": [0.20, 0.25, 0.30],           # design_choices.D3_pressure_share
        "live_b": [0.03, 0.05, 0.10],          # design_choices.D1_live_band = [b, 1 - b]
        "half_life": [None, 3.0, 1.5],         # design_choices.D9_half_life_years; None = no decay
        "knockout": [1.0, 1.25, 1.5],          # design_choices.D10_knockout_weight
        "blend_wL": [1.0, 0.7, 0.5],           # fixed Blend_Weights w_L with w_U = 1 - w_L (not a Design_Choice)
        "D4": ["pooled"],                      # design_choices.D4_scope (21.4)
        "D6": [12, 24],                        # design_choices.D6_window, Stress_Pillar inputs
        "D7": [10, 30],                        # design_choices.D7_new_batter_cap
        "D8": [[0.25, 0.80]],                  # design_choices.D8_phase_splits; matching and Free_Hit_Premium cells only
        "D11": [45, 60, 75, 90],               # minimum Pressure_Ball count of both Roles, at D18_min_matches_default
        "D12": [20, 30],                       # design_choices.D12_replacement_pct
        "D13": ["innings_phase_comp_type"],    # design_choices.D13_cells
        "D14": [True],                         # design_choices.D14_retire_out_batter_penalty
    },
    # ------------------------------------------------------------------------------- tests
    # Test_Suite thresholds and tolerances; none of them is a Design_Choice (Assumption 7).
    "tests": {
        "calibration_error_max": 0.05,         # each innings' M3 Primary_Model (10.20)
        "share_band": [0.245, 0.255],          # Pressure_Ball share of Live_Balls per Role and Innings_Group (13.7)
        "nbv_cell_tol": 0.05,                  # |mean NBV| per Delivery in each innings x Phase cell, runs (15.17)
        "tol_float": 1e-9,                     # exact-arithmetic checks (16.9, 16.13, 18.14, 19.8)
        "tol_prob": 1e-6,                      # probability sums and hand-computed NBV values (11.5, 15.18)
        "tol_ridge": 1e-8,                     # Gram-matrix ridge against a direct ridge fit (Property 21)
    },
    # ------------------------------------------------------------------------------ stage2
    # Stage-2 flags F44-F47, all off as delivered (Requirement 24, criterion 1).
    "stage2": {
        "F44": False,
        "F45": False,
        "F46": False,
        "F47": False,
        "F44_strength": 0.5,                   # strength of the F44 Own_Match_NBV discount (24.2)
    },
    # ------------------------------------------------------------------------------ report
    "report": {
        "pdf_converter": None,                 # e.g. "soffice" (LibreOffice); None exports the DOCX only (27.11, 27.15)
        "verified_sources": [],                # Verified_Source_List: exact URLs, DOIs or reference entries (2.10)
        "page_budget": 15,                     # a Sections 0-5 page count above this only warns (27.16)
    },
}


# ------------------------------------------------------------------------------ constants_meta
# Label metadata for every Method_Constant above, read by the Report_Builder's ConstantRegistry
# (Requirement 2, criterion 8). kind "design_choice": the Config value of Design_Choice
# d_number, labelled "hand-set (Dn)". kind "hand_set": any other hand-set value, labelled
# "hand-set"; its d_number names the Design_Choice that a grid varies, or is None.
# Data-derived constants are recorded at run time, not here.

def _dc(d_number, description):
    return {"kind": "design_choice", "d_number": d_number, "description": description}


def _hs(description, d_number=None):
    return {"kind": "hand_set", "d_number": d_number, "description": description}


CONFIG["constants_meta"] = {
    # Design_Choices
    "design_choices.D1_live_band": _dc("D1", "Live band on WP: Live_Ball when lower <= WP <= upper"),
    "design_choices.D2_alpha": _dc("D2", "Leverage/Stress blend alpha in the Pressure_Index"),
    "design_choices.D3_pressure_share": _dc("D3", "Pressure share of Live_Balls per Role and Innings_Group"),
    "design_choices.D4_scope": _dc("D4", "Percentile scope: ranks within each Innings_Group"),
    "design_choices.D5_weights": _dc("D5", "Stress_Pillar weights: equal, 0.25 each"),
    "design_choices.D6_window": _dc("D6", "Recent-wickets and recent-run-rate window, in Legal_Balls"),
    "design_choices.D7_new_batter_cap": _dc("D7", "New-batter cap, in Legal_Balls faced"),
    "design_choices.D8_phase_splits": _dc("D8", "Phase splits, as shares of Scheduled_Balls"),
    "design_choices.D9_half_life_years": _dc("D9", "Recency half-life, in years"),
    "design_choices.D10_knockout_weight": _dc("D10", "Knockout Stakes_Weight"),
    "design_choices.D11_min_balls_grid": _dc("D11", "Eligibility minimums: candidate minimum Pressure_Ball counts"),
    "design_choices.D11_target_eligible": _dc("D11", "Eligibility minimums: Eligible_Players the selected count must leave per Role"),
    "design_choices.D12_replacement_pct": _dc("D12", "Replacement_Level percentile of unshrunk Clutch_Level"),
    "design_choices.D13_cells": _dc("D13", "Baseline matching cells: innings x Phase"),
    "design_choices.D14_bowler_wicket_codes": _dc("D14", "Credit rules: dismissal codes that credit the bowler with a wicket"),
    "design_choices.D14_run_out_code": _dc("D14", "Credit rules: run-out code, charged to the batter who is out, no bowler credit"),
    "design_choices.D14_retire_out_code": _dc("D14", "Credit rules: Retire Out code, no bowler credit"),
    "design_choices.D14_retire_out_batter_penalty": _dc("D14", "Credit rules: whether Retire Out charges the batter the wicket cost"),
    "design_choices.D15_n_folds": _dc("D15", "Number of Match_Grouped_Folds"),
    "design_choices.D16_n_boot": _dc("D16", "Number of Bootstrap_Resamples"),
    "design_choices.D17_wicket_buckets": _dc("D17", "Wickets_Bucket boundaries, in wickets fallen"),
    "design_choices.D18_min_matches_grid": _dc("D18", "Minimum Pressure_Match count grid"),
    "design_choices.D18_min_matches_default": _dc("D18", "Minimum Pressure_Match count"),
    "design_choices.D19_period_minimum": _dc("D19", "Period_Minimum for the time-holdout check, in Pressure_Balls"),
    "design_choices.D20_score_tolerance": _dc("D20", "Score_Tolerance for match exclusion, in runs"),
    # Other hand-set data rules
    "data_rules.score_tolerance_grid": _hs("Score_Tolerance reporting grid for excluded-match counts, in runs", "D20"),
    "data_rules.recent_years": _hs("Calendar_Years of the Recent_Activity_Rule (confirmed assumption)"),
    "data_rules.min_major_share": _hs("Minimum share of Pressure_Balls from IPL matches or Full_Member_Matches"),
    "data_rules.min_calendar_years": _hs("Minimum number of Calendar_Years with Deliveries in the Role"),
    # Seeds
    "seeds.folds": _hs("Fold_Assigner seed"),
    "seeds.m1": _hs("M1 seed"),
    "seeds.m2": _hs("M2 LightGBM and early-stopping seed"),
    "seeds.m3_lgbm": _hs("M3 LightGBM and early-stopping seed"),
    "seeds.m3_logistic": _hs("M3 logistic baseline seed"),
    "seeds.m3_xgb": _hs("M3 XGBoost and early-stopping seed"),
    "seeds.m3_rf": _hs("M3 random-forest seed"),
    "seeds.m4_lgbm": _hs("M4 LightGBM and early-stopping seed"),
    "seeds.m4_xgb": _hs("M4 XGBoost and early-stopping seed"),
    "seeds.m4_rf": _hs("M4 random-forest seed"),
    "seeds.extras": _hs("Extras_Model seed"),
    "seeds.bootstrap": _hs("Bootstrap_Resample seed"),
    # M1
    "models.m1.ridge_grid": _hs("M1 ridge penalty grid"),
    "models.m1.prob_clip": _hs("M1 dismissal-probability clip before log loss"),
    # M2
    "models.m2.learning_rate": _hs("M2 LightGBM learning rate"),
    "models.m2.num_leaves": _hs("M2 LightGBM maximum leaves per tree"),
    "models.m2.min_data_in_leaf": _hs("M2 LightGBM minimum rows per leaf"),
    "models.m2.lambda_l2": _hs("M2 LightGBM L2 penalty"),
    "models.m2.max_rounds": _hs("M2 LightGBM maximum boosting rounds"),
    "models.m2.es_share": _hs("M2 early-stopping share of training matches"),
    "models.m2.es_patience": _hs("M2 early-stopping patience, in rounds"),
    # M3
    "models.m3.learning_rate": _hs("M3 LightGBM learning rate"),
    "models.m3.num_leaves": _hs("M3 LightGBM maximum leaves per tree"),
    "models.m3.min_data_in_leaf": _hs("M3 LightGBM minimum rows per leaf"),
    "models.m3.lambda_l2": _hs("M3 LightGBM L2 penalty"),
    "models.m3.max_rounds": _hs("M3 LightGBM maximum boosting rounds"),
    "models.m3.es_share": _hs("M3 early-stopping share of training matches"),
    "models.m3.es_patience": _hs("M3 early-stopping patience, in rounds"),
    "models.m3.calibration_bounds": _hs(
        "M3 centred isotonic calibrator output bounds, for every Candidate_Model family"
    ),
    "models.m3.logistic.C": _hs("M3 logistic baseline inverse L2 penalty C"),
    "models.m3.logistic.max_iter": _hs("M3 logistic baseline maximum solver iterations"),
    # M4
    "models.m4.learning_rate": _hs("M4 LightGBM learning rate"),
    "models.m4.num_leaves": _hs("M4 LightGBM maximum leaves per tree"),
    "models.m4.min_data_in_leaf": _hs("M4 LightGBM minimum rows per leaf"),
    "models.m4.lambda_l2": _hs("M4 LightGBM L2 penalty"),
    "models.m4.max_rounds": _hs("M4 LightGBM maximum boosting rounds"),
    "models.m4.es_share": _hs("M4 early-stopping share of training matches"),
    "models.m4.es_patience": _hs("M4 early-stopping patience, in rounds"),
    # Extras_Model
    "models.extras.C": _hs("Extras_Model inverse L2 penalty C"),
    "models.extras.max_iter": _hs("Extras_Model maximum solver iterations"),
    # XGBoost Alternative_Models
    "models.xgb.m3.eta": _hs("M3 XGBoost learning rate"),
    "models.xgb.m3.max_depth": _hs("M3 XGBoost maximum tree depth"),
    "models.xgb.m3.min_child_weight": _hs("M3 XGBoost minimum child weight"),
    "models.xgb.m3.reg_lambda": _hs("M3 XGBoost L2 penalty"),
    "models.xgb.m3.max_rounds": _hs("M3 XGBoost maximum boosting rounds"),
    "models.xgb.m3.es_share": _hs("M3 XGBoost early-stopping share of training matches"),
    "models.xgb.m3.es_patience": _hs("M3 XGBoost early-stopping patience, in rounds"),
    "models.xgb.m4.eta": _hs("M4 XGBoost learning rate"),
    "models.xgb.m4.max_depth": _hs("M4 XGBoost maximum tree depth"),
    "models.xgb.m4.min_child_weight": _hs("M4 XGBoost minimum child weight"),
    "models.xgb.m4.reg_lambda": _hs("M4 XGBoost L2 penalty"),
    "models.xgb.m4.max_rounds": _hs("M4 XGBoost maximum boosting rounds"),
    "models.xgb.m4.es_share": _hs("M4 XGBoost early-stopping share of training matches"),
    "models.xgb.m4.es_patience": _hs("M4 XGBoost early-stopping patience, in rounds"),
    # Random-forest Alternative_Models
    "models.rf.m3.n_estimators": _hs("M3 random-forest number of trees"),
    "models.rf.m3.max_depth": _hs("M3 random-forest maximum depth"),
    "models.rf.m3.min_samples_leaf": _hs("M3 random-forest minimum samples per leaf"),
    "models.rf.m4.n_estimators": _hs("M4 random-forest number of trees"),
    "models.rf.m4.max_depth": _hs("M4 random-forest maximum depth"),
    "models.rf.m4.min_samples_leaf": _hs("M4 random-forest minimum samples per leaf"),
    # Sensitivity grids (Requirement 21)
    "sensitivity.alpha": _hs("Sensitivity grid of the Leverage/Stress blend alpha", "D2"),
    "sensitivity.share": _hs("Sensitivity grid of the pressure share", "D3"),
    "sensitivity.live_b": _hs("Sensitivity grid of the live band: Live_Ball when b <= WP <= 1 - b", "D1"),
    "sensitivity.half_life": _hs("Sensitivity grid of the recency half-life in years (None: no decay)", "D9"),
    "sensitivity.knockout": _hs("Sensitivity grid of the knockout Stakes_Weight", "D10"),
    "sensitivity.blend_wL": _hs("Sensitivity grid of fixed Blend_Weights w_L, with w_U = 1 - w_L"),
    "sensitivity.D6": _hs("Alternative recent windows, in Legal_Balls", "D6"),
    "sensitivity.D7": _hs("Alternative new-batter caps, in Legal_Balls faced", "D7"),
    "sensitivity.D8": _hs("Alternative Phase splits for the matching and Free_Hit_Premium cells", "D8"),
    "sensitivity.D11": _hs("Alternative minimum Pressure_Ball counts for both Roles", "D11"),
    "sensitivity.D12": _hs("Alternative Replacement_Level percentiles", "D12"),
    # Test thresholds
    "tests.calibration_error_max": _hs("Test threshold: maximum Calibration_Error of each M3 Primary_Model"),
    "tests.share_band": _hs("Test threshold: band for the Pressure_Ball share of Live_Balls"),
    "tests.nbv_cell_tol": _hs("Test threshold: largest absolute mean NBV per Delivery in an innings x Phase cell, in runs"),
    "tests.tol_float": _hs("Test tolerance for exact-arithmetic comparisons"),
    "tests.tol_prob": _hs("Test tolerance for probability sums and hand-computed NBV values"),
    "tests.tol_ridge": _hs("Test tolerance for the Gram-matrix ridge against a direct ridge fit"),
    # Stage 2
    "stage2.F44_strength": _hs("Stage-2 F44 discount strength"),
    # Model-choice selection policy (Requirement 23, criterion 8; design DD-17)
    "model_choice.calibration_band": _hs(
        "Model-choice calibration-aware tie-break: relative log-loss band for a Config-preferred family"
    ),
}

del _dc, _hs
