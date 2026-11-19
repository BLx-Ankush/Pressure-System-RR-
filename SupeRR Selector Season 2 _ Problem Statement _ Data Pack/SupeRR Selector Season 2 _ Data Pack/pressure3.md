# SupeRR Selector Season 2 (Round 1): Definitive Pressure Framework & Standalone Executable Specification (v5.0)

## Executive Summary

To identify and rank the **10 most reliable batters and 10 most reliable bowlers under pressure** across the 2022–2026 IPL and T20I dataset (\~201,000 deliveries across \~880 matches), this document specifies a **standalone, end-to-end Pressure & Net Ball Value (NBV) Rating System (v5.0)** built strictly around the verified columns of the six supplied CSV files (`ball_by_ball.csv`, `matches.csv`, `playing_eleven.csv`, `competitions.csv`, `teams.csv`, and `competition_team_player_mapping.csv`).

1. **Plain-Language Definition of Pressure:** A delivery counts as high-pressure when **the match outcome is genuinely contested** (neither a foregone win nor a dead-rubber blowout) and the player faces elevated stress across four cricket dimensions: **rate/platform tempo required**, **wickets lost and remaining shed batting depth**, **stage/balls-remaining scarcity**, and **within-over micro-pressure** (consecutive dot balls or a stalling partner for batters; runs already leaked in the over for bowlers).

2. **Verified Score & Extras Reconstruction:** Because `matches.csv` stores no innings totals and `ball_by_ball.runs` records bat runs only (while `extras1_runs` holds type codes `6=None`, `2=wide`, `3=legbyes`, `4=byes`, `5=other` and `extras_2_runs` holds additional extra runs such as boundary/multi-run extras on no-balls), Step 1 rebuilds exact delivery total runs and legal-ball counts (`overid` 1–20) and cross-checks them against both **defended run margins** (`"Won by N Runs"`) and **successful chase trajectories** (`"Won by N Wickets"`) in `matches.result`.

3. **Player-Linked Cross-Pool Opposition Adjustment (No Double Counting):** Because IPL franchises and national/A-teams never play each other directly, team-level Elo sits on disconnected scales. Instead, we estimate pre-match player quality across a **unified player bridge** (anchored by players who appear in both the IPL and T20Is) and incorporate opposing player/XI quality **once**—directly inside the out-of-fold Expected Utility baseline $\mathbb{E}[U_e \mid \mathbf{X}_e^{-}]$.

4. **Separate Shrinkage of Clutch Level & Baseline-Floored Uplift:** To prevent rewarding players who are poor in ordinary situations, we floor each player's usual baseline at the role replacement level ($\mu_{\text{rep, role}}$), shrink **Absolute Clutch Performance** and **Clutch Uplift** separately using their empirically measured split-half repeatabilities, and combine them using reliability-derived weights so absolute pressure excellence leads while verified clutch elevation breaks ties.

5. **Transparent Calibration & Sensitivity Protocol:** Parameters that can be estimated from the data pack (expected runs, wicket run-equivalence $\hat{\Omega}_e$, extras costs, and repeatability-based shrinkage $\hat{\kappa}$) are learned out-of-fold; structural design choices are cataloged in Appendix A and stress-tested via leave-one-pillar-out ablations and 2,000-iteration match-block bootstraps.

## 1. Ground-Truth Data Pack Schema & Score Reconstruction Protocol

### 1.1 Verified Schema of the Six Supplied CSV Files

Every formula in this specification references **only** the verified columns present in the competition data pack:

| CSV File (\~Row Count) | Verified Columns Used by the Pipeline | Key Data Quirks & Mandatory Preprocessing Rules | 
| ----- | ----- | ----- | 
| **`ball_by_ball.csv`** (\~201,000 rows) | `match_id`, `innings`, `overid`, `over_ball_num`, `innings_ball_num`, `batting_team_id`, `bowling_team_id`, `striker_id`, `striker_name`, `non_striker_id`, `non_striker_name`, `bowler_id`, `bowler_name`, `batting_type`, `bowling_type`, `pace_or_spin`, `batting_order`, `runs`, `is_four`, `is_six`, `is_wicket`, `dismissal_type_id`, `dismissal_type`, `dismissed_player_id`, `extras_type`, `extras1_runs`, `extras_2_runs`, `bowl_speed`, `stroke`, `delivery_style` | 1\. `innings` `1, 2` = main innings; `3, 4` = Super Overs (isolated from main 20-over models).  2\. `overid` is $1\text{–}20$ (1-indexed). `over_ball_num` and `innings_ball_num` count extras and exceed $6$ and $120$.  3\. `runs` holds **bat runs only** (`0` on wides, `0` on byes/legbyes).  4\. `extras_type` (`None`, `wide`, `legbyes`, `byes`, `other`) duplicates code `extras1_runs` (`6=None`, `2=wide`, `3=legbyes`, `4=byes`, `5=other`). Every `other` row is a **No-Ball** (`delivery_style` includes `"Beamer"`, `"Full Toss"`, and `runs > 0` off bat).  5\. `extras_2_runs` holds additional extra runs (e.g., `4` in `match_id=13614, innings=1, overid=18`) and **must** be added.  6\. `bowling_type` varies across seasons for the same bowler; normalize each `bowler_id` to their modal `bowling_type`.  7\. `is_four`, `is_six`, `is_wicket` are parsed from string `"true"`/`"false"`.  8\. `dismissal_type_id == 9` (`Retire Out`) is a tactical retirement—do **not** credit bowler or penalize batter as dismissed. | 
| **`matches.csv`** (\~880 rows) | `match_id`, `comp_id`, `match_number`, `match_date`, `stage`, `venue_id`, `venue_country`, `home_team_id`, `home_team`, `away_team_id`, `away_team`, `toss_win_team_id`, `toss_opted`, `man_of_the_match`, `first_inn_revised_overs`, `second_inn_revised_overs`, `revised_target`, `knock_out`, `win_team_id`, `result` | 1\. Sort chronologically by `match_date, match_id` (raw file is unsorted).  2\. Drop abandoned / no-result games (`win_team_id` blank and `result` contains `"Abandoned"` or `"No Result"`).  3\. `first_inn_revised_overs` and `second_inn_revised_overs` are **already stored in legal balls** (e.g., `54` = 9 overs, `90` = 15 overs, `30` = 5 overs). If blank/0, scheduled balls $B_{\text{sched},m,i} = 120$. | 
| **`playing_eleven.csv`** (\~28,400 rows) | `match_id`, `team_id`, `player_id`, `batting_order`, `is_played`, `is_captain`, `is_wk`, `is_impact`, `is_impact_off` | 1\. Filter to `is_played == 'true'` (table also lists unused squad members with `is_played == 'false'` and blank `batting_order`).  2\. `is_impact == 1` flags the Impact Player subbed in; `is_impact_off == 1` flags the player subbed out. Exact substitution timing is inferred from the first delivery in `ball_by_ball.csv` where the `is_impact == 1` player appears (or after the `is_impact_off == 1` player's final action). | 
| **`competitions.csv`** (\~147 rows) | `comp_id`, `comp_name`, `match_format`, `season` | Covers IPL 2022–2026 (5 seasons), bilateral/tri-series T20Is, Asia Cup 2022/2025 & Qualifiers, and A-team events (Emerging Teams Asia Cup 2024, Asia Cup Rising Stars 2025). No T20 World Cups are in the pack. | 
| **`teams.csv`** | `comp_id`, `team_id`, `team_name` | Links teams to competitions. | 
| **`competition_team_player_mapping.csv`** | `comp_id`, `comp_name`, `team_id`, `team_name`, `player_id`, `player_name` | Official `player_name` lookup for `player_id` to match the Unstop dropdowns. Audit and resolve any duplicate `player_name` strings across distinct `player_id` values. | 

### 1.2 Step-by-Step Score, Extras & Legal-Ball Reconstruction

Because neither innings totals nor delivery total runs are stored directly, every delivery row $e \in \{1, \dots, E_{m,i}\}$ in `ball_by_ball.csv` is processed through four deterministic rules:

1. **Legal Delivery Indicator (**$\Delta n_e$**) & Pre-Ball Legal Ball Counter (**$n_e^{-}$**):**
   Every row with `extras_type == 'wide'` (`extras1_runs == 2`) is a Wide (`is_wide_e = 1`), and every row with `extras_type == 'other'` (`extras1_runs == 5`) is a No-Ball (`is_noball_e = 1`). Both do not count as legal balls:
   

   $$
   \Delta n_e = \mathbb{I}\!\big(\text{extras\_type}_e \notin \{\text{'wide'}, \text{'other'}\}\big), \qquad n_e^{-} = \sum_{k=1}^{e-1} \Delta n_k
   $$

   
   *Automated Over Integrity Check:* Verify that every completed over (`overid` $1\text{–}19$, and completed `overid` $20$) has $\sum_{e \in \text{overid}} \Delta n_e = 6$ legal balls.

2. **Delivery Total Runs (**$\text{TotalRuns}_e$**) & Extras Runs (**$\text{ExtraRuns}_e$**):**
   Using `runs` (bat runs), `extras_type`, and `extras_2_runs` (additional extra runs):

   * **`extras_type == 'None'` (`6`):** $\text{ExtraRuns}_e = 0$, $\text{TotalRuns}_e = \text{runs}_e$.

   * **`extras_type == 'wide'` (`2`):** Base wide is $1$ run plus any additional runs in `extras_2_runs`: $\text{ExtraRuns}_e = 1 + \text{extras\_2\_runs}_e$, $\text{TotalRuns}_e = \text{runs}_e + \text{ExtraRuns}_e$.

   * **`extras_type == 'other'` (`5`, No-Ball):** Base no-ball penalty is $1$ run plus `extras_2_runs` plus any bat `runs`: $\text{ExtraRuns}_e = 1 + \text{extras\_2\_runs}_e$, $\text{TotalRuns}_e = \text{runs}_e + 1 + \text{extras\_2\_runs}_e$. (For example, in `match_id=13614, innings=1, overid=18`, where `extras_2_runs = 4`, this yields $1 + 4 = 5$ extra runs.)

   * **`extras_type in ('byes', 'legbyes')` (`3, 4`):** Because `runs = 0` on bye and leg-bye rows, the runs scored on the bye/leg-bye are $1 + \text{extras\_2\_runs}_e$ (if `extras_2_runs` stores runs above 1) or $\max(1, \text{extras\_2\_runs}_e)$: $\text{ExtraRuns}_e = \max(1, \text{extras\_2\_runs}_e)$ (calibrated against the margin check below), $\text{TotalRuns}_e = \text{ExtraRuns}_e$.

3. **Dual Verification Against `matches.result` (Both Defended Totals AND Chases):**
   Let $\text{Score}_e^{-} = \sum_{k=1}^{e-1} \text{TotalRuns}_k$ and $\widehat{\text{Tot}}_{m,i} = \sum_{k=1}^{E_{m,i}} \text{TotalRuns}_k$. We verify the rebuilt totals across **both** match outcome types:

   * **Defended Totals (`result` matches `"Won by M Run(s)"`, non-DLS):** Check $\widehat{\text{Tot}}_{m,1} - \widehat{\text{Tot}}_{m,2} = M$.

   * **Successful Chases (`result` matches `"Won by W Wicket(s)"`):** Define the active target as $T_{m} = \text{revised\_target}_m$ (if non-null/non-zero) else $\widehat{\text{Tot}}_{m,1} + 1$. Check that the chasing team had **not yet reached the target before the final ball** ($\text{Score}_{m,2,E_{m,2}}^{-} \le T_m - 1$) and **reached or exceeded the target on the final ball** ($T_m \le \widehat{\text{Tot}}_{m,2} \le T_m + 5$).

   * Report the exact match-level pass rate of both checks in Section 3 of the Methodology Report.

## 2. Formal Definition of Pressure (Problem Statement Section 1)

### 2.1 Plain-Language & Mathematical Definition of State Pressure ($\text{DPI}_{i,e}^{\text{role, v5}}$)

A delivery $e$ in innings $i \in \{1, 2\}$ carries high pressure for $\text{role} \in \{\text{bat}, \text{bowl}\}$ when **situational stress is high** and **the contest is live**. Using strictly pre-ball state variables $(\cdot)_e^{-}$:

$$
\text{DPI}_{i,e}^{\text{role, v5}} = 100 \times \left(\mathcal{L}_{i,e}^{-}\right)^{0.60} \times \Big( 0.30 \cdot \mathcal{S}_{\text{rate},i,e}^{\text{role},-} + 0.30 \cdot \mathcal{S}_{\text{wkt},i,e}^{\text{role},-} + 0.20 \cdot \mathcal{S}_{\text{phase},i,e}^{\text{role},-} + 0.20 \cdot \mathcal{S}_{\text{micro},i,e}^{\text{role},-} \Big)
$$

where:

1. **Rate & Platform Conversion Stress (**$\mathcal{S}_{\text{rate},i,e}^{\text{role},-} \in [0, 1]$**):**

   * **2nd Innings (Chasing / Defending Target** $T_m$**):** Let $B_{\text{rem},e}^{-} = \max(1, B_{\text{sched},m,2} - n_e^{-})$, $\text{RRR}_e^{-} = 6 \times \frac{\max(0, T_m - \text{Score}_e^{-})}{B_{\text{rem},e}^{-}}$, and $r_{2,e}^{-} = \frac{\text{RRR}_e^{-}}{\text{Par}_{m,e}^{-}(v, \text{overid}_e)}$.
     

     $$
     \mathcal{S}_{\text{rate},2,e}^{\text{bat},-} = \text{clip}\!\left(\frac{r_{2,e}^{-} - 0.70}{1.10}, \; 0, \; 1\right), \qquad \mathcal{S}_{\text{rate},2,e}^{\text{bowl},-} = \max\!\left(\exp\!\left(-\frac{(r_{2,e}^{-} - 1.05)^2}{2 \times 0.40^2}\right), \; \mathbb{I}(B_{\text{rem},e}^{-} \le 12 \;\land\; T_m - \text{Score}_e^{-} \le 18)\right)
     $$

   * **1st Innings (Setting / Restricting with Death Platform Conversion):** Let $r_{1,e}^{-} = \frac{\text{CRR}_e^{-}}{\text{Par}_{m,e}^{-}}$ and $r_{18,e}^{-} = \frac{\text{RR}_{18,e}^{-}}{\text{Par}_{m,e}^{-}}$. In Overs 1–15, batters feel pressure when scoring lags behind par ($r_{1,e}^{-} \lt 1.0$). In **Overs 16–20 (`overid >= 16`) with wickets in hand (**$W_e^{-} \le 5$**)**, batters also face **Platform Conversion Opportunity-Cost Pressure**—because failing to accelerate to $\sim 1.35\times\text{Par}$ wastes unused wickets in the shed:
     

     $$
     \mathcal{S}_{\text{rate},1,e}^{\text{bat},-} = \max\!\left(\text{clip}\!\left(\frac{1.15 - \min(r_{1,e}^{-}, r_{18,e}^{-})}{0.60}, \; 0.15, \; 1\right), \; \mathbb{I}(\text{overid}_e \ge 16) \cdot \text{clip}\!\left(0.45 + 0.45 \cdot \frac{6 - \min(6, W_e^{-})}{6}, \; 0, \; 0.90\right)\right)
     $$

     $$
     \mathcal{S}_{\text{rate},1,e}^{\text{bowl},-} = \text{clip}\!\left(\frac{\max(r_{1,e}^{-}, r_{18,e}^{-}) - 0.75}{0.80}, \; 0.15, \; 1\right)
     $$

2. **Wicket Scarcity, Cluster Shock & Active Shed Depth Stress (**$\mathcal{S}_{\text{wkt},i,e}^{\text{role},-} \in [0, 1]$**):**
   Using `playing_eleven.csv` (`is_played == 'true'`), we track the exact set of active unbatted players remaining in the shed at ball $e$ (accounting for whether an `is_impact == 1` player has entered and an `is_impact_off == 1` player has been replaced) to compute pre-ball **Remaining Batting Quality (**$\text{RBQ}_e^{-} \in [0, 1]$**)**, **Excess Wickets Lost (**$W_{\text{excess},e}^{-}$**)**, **18-Ball Cluster Wickets (**$\Delta W_{18,e}^{-}$**)**, and **Striker Balls Faced (**$B_{\text{faced},e}^{-}$**)**:
   

   $$
   \mathcal{S}_{\text{wkt},i,e}^{\text{bat},-} = \text{clip}\!\Big(0.25 \cdot (1 - \text{RBQ}_e^{-}) + 0.12 \cdot W_{\text{excess},e}^{-} + 0.25 \cdot \Delta W_{18,e}^{-} + 0.15 \cdot \mathbb{I}(B_{\text{faced},e}^{-} \le 6), \; 0, \; 1\Big)
   $$

   $$
   \mathcal{S}_{\text{wkt},i,e}^{\text{bowl},-} = \text{clip}\!\left(0.45 \cdot \text{RBQ}_e^{-} + 0.25 \cdot \mathbb{I}(\Delta W_{18,e}^{-} = 0) + 0.30 \cdot \min\!\left(1, \frac{B_{\text{faced},e}^{-}}{25}\right), \; 0, \; 1\right)
   $$

3. **Phase & Balls-Remaining Urgency (**$\mathcal{S}_{\text{phase},i,e}^{\text{role},-} \in [0, 1]$**):**
   With $B_{\text{sched},m,i}$ taken directly from `first_inn_revised_overs` / `second_inn_revised_overs` (in balls, defaulting to $120$), let $f_e^{-} = n_e^{-} / B_{\text{sched},m,i}$:
   

   $$
   \mathcal{S}_{\text{phase},2,e}^{\text{role},-} = 0.25 + 0.75 \cdot (f_e^{-})^{1.35}, \qquad \mathcal{S}_{\text{phase},1,e}^{\text{role},-} = \begin{cases} 0.65 & f_e^{-} \le 0.30 \text{ (Powerplay)} \\ 0.35 + 0.25 \cdot \frac{f_e^{-} - 0.30}{0.45} & 0.30 \lt f_e^{-} \le 0.75 \text{ (Middle)} \\ 0.65 + 0.35 \cdot \frac{f_e^{-} - 0.75}{0.25} & f_e^{-} \gt 0.75 \text{ (Death)} \end{cases}
   $$

4. **Within-Over Micro-Stress (**$\mathcal{S}_{\text{micro},i,e}^{\text{role},-} \in [0, 1]$**):**
   

   $$
   \mathcal{S}_{\text{micro},i,e}^{\text{bat},-} = \text{clip}\!\Big(0.20 \cdot D_{\text{dots\_streak},e}^{-} + 0.10 \cdot \max(0, B_{\text{no\_bnd},e}^{-} - 8) + 0.20 \cdot \mathbb{I}(\text{SR}_{\text{NS},12,e}^{-} \lt 85 \;\land\; B_{\text{NS\_faced},e}^{-} \ge 6), \; 0, \; 1\Big)
   $$

   $$
   \mathcal{S}_{\text{micro},i,e}^{\text{bowl},-} = \text{clip}\!\left(\frac{R_{\text{over},e}^{-}}{14} \cdot \left(\frac{\min(5, b_{\text{over\_rem},e}^{-})}{4}\right)^{0.50} + 0.25 \cdot \mathbb{I}(\text{PrevBall}_{e-1} \in \{4, 6, \text{Wide}, \text{No-Ball}\}), \; 0, \; 1\right)
   $$

5. **Contest Closeness Gate (**$\mathcal{L}_{i,e}^{-} = 4 \cdot p_e^{-}(1 - p_e^{-})$**) & Bounded Phase-Quantile Clutch Gate:**
   

   $$
   \text{IsClutch}_{i,e}^{\text{role}} = \mathbb{I}\!\left[ \Big(\text{DPI}_{i,e}^{\text{role, v5}} \ge 60\Big) \;\lor\; \Big(\text{DPI}_{i,e}^{\text{role, v5}} \ge Q_{0.70,\text{train}}^{\text{phase, role}} \;\land\; \text{DPI}_{i,e}^{\text{role, v5}} \ge 45 \;\land\; \mathcal{L}_{i,e}^{-} \ge 0.50\Big) \right]
   $$

### 2.2 Non-Circular Validation That $\text{DPI}^{\text{v5}}$ Separates Pressure from Ordinary Play

Because pillar weights are fixed a priori (and perturbed in sensitivity checks) rather than tuned on $|\Delta \text{WP}|$, we validate $\text{DPI}_{i,e}^{\text{role, v5}}$ on **out-of-fold held-out matches** using **four independent observable behavioural outcomes that never entered the** $\text{DPI}$ **formula**:

1. **Held-Out Dismissal Probability (`is_wicket == 'true'` excluding `Retire Out`):** Rises across $\text{DPI}_{\text{bat}}^{\text{v5}}$ deciles as batters are forced into higher-risk strokeplay.

2. **Held-Out Dot-Ball & False/Risky Stroke Share (`stroke` distribution):** Shows how consecutive-dot pressure shifts stroke selection toward high-risk aerial shots (`Slog Sweep`, `Pull`, `Lofted Drive`).

3. **Within-Player Fixed-Effects Performance Drop:** Controlling for player fixed effects, average batters suffer a drop in Net Ball Value and survival rate in Deciles 8–10 relative to Deciles 1–5, whereas our Top 10 ranked players maintain or increase their NBV.

4. **Comparison Against Naive Cutoffs ("`overid` 16–20 Only" and "**$\text{RRR} > 10$ **Only"):** Demonstrates that naive rules misclassify dead-rubber blowouts ($p_e^{-} \lt 0.03$) as pressure while missing high-leverage Powerplay collapses and middle-over spin strangles.

## 3. Master Reference Table of All 24 Operational Factors (Exact v5.0 Schema Mapping)

Note: In v5.0, we streamline the earlier 26 factors to **24 strictly operational factors** by removing double-counted tandem assists from primary NBV and treating Cosine-Similarity Replacement Twins as an optional scouting appendix rather than a ranking input.

| \# | Architectural Layer | Factor Name | Exact CSV File & Column Names | Pre-Ball Mathematical Formula & Fallback | Output Variable | Consuming Equation | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| **1** | **L1: Rate Stress** | **Required Run Rate Ratio (**$r_{2,e}^{-}$**)** | `ball_by_ball`: `innings`, `runs`, `extras_type`, `extras_2_runs`; `matches`: `revised_target`, `second_inn_revised_overs` | $r_{2,e}^{-} = \frac{6(T_m - \text{Score}_e^{-}) / B_{\text{rem},e}^{-}}{\text{Par}_{m,e}^{-}(v, \text{overid}_e)}$. | `r2_ratio_pre` | $\mathcal{S}_{\text{rate},2,e}^{\text{bat},-}$, $\mathcal{S}_{\text{rate},2,e}^{\text{bowl},-}$ | 
| **2** | **L1: Rate Stress** | **1st-Inn Tempo & Death Platform Ratio (**$r_{1,e}^{-}$**)** | `ball_by_ball`: `runs`, `extras_type`, `overid`, `is_wicket` | $r_{1,e}^{-} = \frac{6\,\text{Score}_e^{-} / \max(6, n_e^{-})}{\text{Par}_{m,e}^{-}(v, \text{overid}_e)}$; includes Over 16–20 platform-conversion floor when $W_e^{-} \le 5$. | `r1_ratio_pre` | $\mathcal{S}_{\text{rate},1,e}^{\text{bat},-}$, $\mathcal{S}_{\text{rate},1,e}^{\text{bowl},-}$ | 
| **3** | **L1: Rate Stress** | **Rolling 18-Ball Tempo (**$\text{RR}_{18,e}^{-}$**)** | `ball_by_ball`: `runs`, `extras_type`, `extras_2_runs` | Reconstructed total runs over last $\min(n_e^{-}, 18)$ legal balls divided by $\max(1, \min(n_e^{-}, 18)/6)$. | `rr18_pre` | $\mathcal{S}_{\text{rate},1,e}^{\text{role},-}$, $p_e^{-}$ | 
| **4** | **L2: Wicket & Shed** | **Active Wickets & Impact Sub Tracking (**$W_e^{-}$**)** | `ball_by_ball`: `is_wicket`, `dismissal_type_id`; `playing_eleven`: `is_played`, `is_impact`, `is_impact_off` | Counts wickets where `is_wicket == 'true'` and `dismissal_type_id != 9` (`Retire Out`). Tracks when `is_impact == 1` player enters `ball_by_ball`. | `w_lost_pre`, `active_shed` | $\text{RBQ}_e^{-}$, $W_{\text{excess},e}^{-}$ | 
| **5** | **L2: Wicket & Shed** | **Excess Wickets vs. Dataset Curve (**$W_{\text{excess},e}^{-}$**)** | Derived from $W_e^{-}$, $n_e^{-}$, $B_{\text{sched},m,i}$ | $W_{\text{excess},e}^{-} = \max(0, W_e^{-} - \bar{W}_{\text{train}} \cdot (n_e^{-} / B_{\text{sched},m,i}))$. | `w_excess_pre` | $\mathcal{S}_{\text{wkt},i,e}^{\text{bat},-}$, $p_e^{-}$ | 
| **6** | **L2: Wicket & Shed** | **Cluster Wicket Shock (**$\Delta W_{18,e}^{-}$**)** | `ball_by_ball`: `is_wicket`, `dismissal_type_id`, `extras_type` | Wickets (`dismissal_type_id != 9`) fallen in preceding $\min(n_e^{-}, 18)$ legal balls. | `dw18_pre` | $\mathcal{S}_{\text{wkt},i,e}^{\text{role},-}$, $p_e^{-}$ | 
| **7** | **L2: Wicket & Shed** | **Causal Remaining Shed Batting Quality (**$\text{RBQ}_e^{-}$**)** | `playing_eleven`: `player_id`, `is_played`, `is_impact`, `is_impact_off`; prior `ball_by_ball` ($<m$) | Sum of pre-match shrunk $\hat{q}_{\text{bat},u,<m}$ across active unbatted XI members + `non_striker_id` divided by $7 \cdot \bar{q}_{\text{top7},<m}$. | `rbq_pre` | $\mathcal{S}_{\text{wkt},i,e}^{\text{role},-}$, $\hat{\Omega}_e$, $p_e^{-}$ | 
| **8** | **L2: Wicket & Shed** | **Striker Settledness (**$B_{\text{faced},e}^{-}$**) & Wicket Drought** | `ball_by_ball`: `striker_id`, `extras_type` | Legal balls + no-balls faced by `striker_id` in innings $i$ before $e$; $\mathbb{I}(\Delta W_{18,e}^{-} = 0)$. | `b_faced_pre` | $\mathcal{S}_{\text{wkt},i,e}^{\text{role},-}$, $V_{\text{wkt},e}$ | 
| **9** | **L3: Phase & Time** | **Revised-Balls Scarcity (**$B_{\text{rem},e}^{-}$**)** | `matches`: `first_inn_revised_overs`, `second_inn_revised_overs` | $B_{\text{rem},e}^{-} = \max(1, B_{\text{sched},m,i} - n_e^{-})$, where revised overs columns are already in balls (default $120$). | `b_rem_pre`, `f_pre` | $\mathcal{S}_{\text{phase},2,e}^{\text{role},-}$, $\hat{\Omega}_e$ | 
| **10** | **L3: Phase & Time** | **Field-Restriction Phase Leverage** | `ball_by_ball`: `overid` | Powerplay ($f_e^{-} \le 0.30$), Middle ($0.30 \lt f_e^{-} \le 0.75$), Death ($f_e^{-} \gt 0.75$). | `s_phase_pre` | $\text{DPI}_{i,e}^{\text{role, v5}}$ | 
| **11** | **L4: Micro-Stress** | **Consecutive Dot-Ball Streak (**$D_{\text{dots\_streak},e}^{-}$**)** | `ball_by_ball`: `striker_id`, `runs`, `extras_type` | Consecutive legal deliveries faced by `striker_id` before $e$ with `runs == 0`. | `dot_streak_pre` | $\mathcal{S}_{\text{micro},i,e}^{\text{bat},-}$, $\mathbf{X}_e^{-}$ | 
| **12** | **L4: Micro-Stress** | **Boundary Drought & Non-Striker Stall** | `ball_by_ball`: `is_four`, `is_six`, `non_striker_id`, `runs` | Legal balls since last `is_four == 'true'` or `is_six == 'true'`; `non_striker_id` SR over last 12 balls. | `ns_stall_pre` | $\mathcal{S}_{\text{micro},i,e}^{\text{bat},-}$ | 
| **13** | **L4: Micro-Stress** | **Bowler Within-Over Damage (**$R_{\text{over},e}^{-}$**)** | `ball_by_ball`: `bowler_id`, `overid`, `runs`, `extras_type`, `extras_2_runs` | Bowler-charged runs in current `overid` before $e$, plus indicator for previous ball boundary/extra. | `s_micro_bowl_pre` | $\mathcal{S}_{\text{micro},i,e}^{\text{bowl},-}$ | 
| **14** | **L5: Contest & Stakes** | **Out-of-Fold Win Prob (**$p_e^{-}$**) & Contest Gate (**$\mathcal{L}_{i,e}^{-}$**)** | Pre-ball state $\mathbf{X}_{\text{wp},e}^{-}$; `matches`: `win_team_id` | $\mathcal{L}_{i,e}^{-} = 4 p_e^{-}(1 - p_e^{-})$; single-ball swing $\Delta \text{WP}_e^{-} = \text{abs}(\hat{p}(\text{Bnd}_e) - \hat{p}(\text{Wkt}_e))$. | `lev_pre`, `wp_swing_pre` | $\text{DPI}_{i,e}^{\text{role, v5}}$, $W_{i,e}^{\text{clutch}}$ | 
| **15** | **L5: Contest & Stakes** | **Direct Match Stakes Multiplier (**$\mathcal{M}_{\text{stakes},m}$**)** | `matches`: `knock_out`, `stage` | `stage == 'Final'`: $1.20$; `knock_out == 1` or `Qualifier`/`Eliminator`/`Semi Final`: $1.15$; `Super 4`: $1.08$; `League`/`Others`: $1.00$. | `m_stakes` | Delivery weight $w_e$ | 
| **16** | **L6: Venue & Pitch** | **Causal Venue Boundary & 2nd-Inn Asymmetry** | `ball_by_ball`: `is_four`, `is_six`, `innings`; `matches`: `venue_id`, `venue_country`, `match_date` | Pre-match EB-shrunk venue boundary rate and controlled 2nd-vs-1st innings run-rate residual (Cold-start fallback: `venue_country` $\times$ `match_format` mean). | `bnd_idx_pre`, `venue_asym_pre` | $\text{Par}_{m,e}^{-}$, $\mathbf{X}_e^{-}$ | 
| **17** | **L6: Venue & Pitch** | **Zero-Leakage Causal Pitch Par (**$\text{Par}_{m,e}^{-}$**)** | `ball_by_ball` events $< e$ in match $m$ and $< m$ at `venue_id` | $\text{Par}_{m,e}^{-} = \lambda_e \text{Par}_{\text{hist},<m} + (1-\lambda_e)\text{Par}_{\text{match},<e}$ with $\lambda_e = \frac{36}{n_{\text{match},<e}+36}$ (Cold-start fallback when $n_{v,<m} < 36$: country-format prior). | `par_rr_pre` | $r_{1,e}^{-}$, $r_{2,e}^{-}$, $\mathbb{E}[U_e \mid \mathbf{X}_e^{-}]$ | 
| **18** | **L7: Matchups & NBV** | **Player-Linked Cross-Pool Skill (**$\hat{\theta}_{\text{bat},<m}, \hat{\theta}_{\text{bowl},<m}$**)** | `ball_by_ball`: `striker_id`, `bowler_id`; `playing_eleven` | Ridge/EB player skill estimated across all competitions (bridged by players appearing in both IPL and T20Is). Applied **once** inside $\mathbb{E}[U_e \mid \mathbf{X}_e^{-}]$. | `theta_opp_pre` | $\mathbb{E}[U_e^{\text{role}} \mid \mathbf{X}_e^{-}]$ | 
| **19** | **L7: Matchups & NBV** | **Modal Bowling Style** $\times$ **Handedness Interaction** | `ball_by_ball`: `batting_type`, modal `bowling_type` per `bowler_id`, `pace_or_spin` | Partial-pooled interaction $\delta_{\text{bat\_type}\times\text{modal\_bowl\_type},<m}$ estimated on training folds. | `matchup_delta_pre` | $\mathbb{E}[U_e^{\text{role}} \mid \mathbf{X}_e^{-}]$ | 
| **20** | **L7: Matchups & NBV** | **Data-Estimated Wicket Run-Equivalence (**$\hat{\Omega}_e$**) & Wicket Quality (**$V_{\text{wkt},e}$**)** | Pre-ball $(i, B_{\text{rem},e}^{-}, W_e^{-}, \text{RBQ}_e^{-}, \text{Par}_{m,e}^{-})$ and dismissed batter's $\hat{\theta}_{\text{bat},<m}$ | $\hat{\Omega}_e = \hat{G}_{\text{train}}(W_e^{-}) - \hat{G}_{\text{train}}(W_e^{-}+1)$; $V_{\text{wkt},e} \in [0.35, 1.50]$ scales by dismissed batter's pre-match skill and set score. | `omega_pre`, `v_wkt` | $\text{NBV}_e^{\text{bat}}$, $\text{NBV}_e^{\text{bowl}}$ | 
| **21** | **L7: Matchups & NBV** | **Data-Estimated Extras Opportunity Cost (**$\hat{P}_e^{\text{extras}}$**)** | `ball_by_ball`: `extras_type`, `extras1_runs`, `extras_2_runs` | Empirical excess over-runs caused by a `wide` ($\hat{c}_{\text{wide}}$) or `other`/no-ball ($\hat{c}_{\text{noball}}$) learned on training folds. | `p_extras_cost` | $U_e^{\text{bowl}}$, $\mathbb{E}[U_e^{\text{bowl}} \mid \mathbf{X}_e^{-}]$ | 
| **22** | **L8: Ranking & VORP** | **Capped Delivery Weight (**$w_e$**) with Recency & Endogeneity** | `matches`: `match_date`, `knock_out`, `stage`; player's prior match NBV | $w_e = \text{clip}\!\big(w_{\text{time},m} \cdot W_{i,e}^{\text{clutch}} \cdot \mathcal{M}_{\text{stakes},m} \cdot \eta_{\text{endo},e}, \; 0.15, \; 2.50\big)$ with $w_{\text{time},m} = e^{-0.18\,\Delta\text{years}_m}$. | `w_e` | Weighted player means $\overline{\text{NBV}}_{p,\text{clutch}}^{(6)}$ | 
| **23** | **L8: Ranking & VORP** | **Separately Shrunk Clutch Level (**$\tilde{C}_p$**) & Baseline-Floored Uplift (**$\tilde{U}_p$**)** | Context-matched $\overline{\text{NBV}}_{p,\text{clutch}}^{(6)}$ and $\overline{\text{NBV}}_{p,\text{ord}}^{(6)}$ | Floors ordinary baseline at $\mu_{\text{rep,role}}$; shrinks $\tilde{C}_p$ ($\hat{\kappa}_{\text{level}}$) and $\tilde{U}_p$ ($\hat{\kappa}_{\text{uplift}}$) separately; weights by split-half reliability. | `S_tilde_p` | Role-standardized $Z_p$ | 
| **24** | **L8: Ranking & VORP** | **Role Standardization (**$Z_p^{\text{adj}}$**), Consistency (**$\mathcal{R}_p$**) & 2,000-Match Bootstrap** | All 6 CSV files, joined to `competition_team_player_mapping.player_name` | $Z_p^{\text{adj}} = Z_p(1 + \mathcal{R}_p) + 0.15\mathcal{R}_p$; $\text{Rating}_p = \text{clip}(50 + 10 Z_p^{\text{adj}}, 0, 100)$; 2,000 match-block bootstrap CIs. | `Rating_p`, `CI95_p` | Final Top 10 Batters & Bowlers | 

## 4. Deep Dive on Key Methodological Resolutions (Addressing All Review Critiques)

### 4.1 Why Player-Linked Cross-Pool Skill Beats Team Elo (And Why We Do Not Double-Penalize)

1. **The Disconnected Graph Problem in Team Elo:**
   In the supplied dataset, IPL franchises (`Chennai Super Kings`, `Rajasthan Royals`, etc.) never play against national T20I teams (`India`, `Nepal`, `Zimbabwe`, etc.) or A-teams (`India A`, `Pakistan Shaheens`). Consequently, any team-level Elo model has two disconnected components whose relative scale depends entirely on arbitrary starting priors.

2. **The Player-Bridge Solution:**
   Hundreds of players in `ball_by_ball.csv` share the same `player_id` across **both** IPL seasons and T20I / Asia Cup / A-team competitions. We fit a **pre-match Ridge-regularized two-way fixed-effects model** across all historical deliveries prior to match $m$:
   

   $$
   y_k = \mu_{\text{phase}} + \text{PitchAdj}_k + \theta_{\text{bat}, \text{striker}(k), <m} - \theta_{\text{bowl}, \text{bowler}(k), <m} + \delta_{\text{hand}\times\text{style}, <m} + \varepsilon_k
   $$

   
   Because cross-over players anchor the IPL and T20I pools to the exact same run scale, an Associate bowler or A-team batter who only plays in lower-tier matches has their skill $\hat{\theta}_{<m}$ directly calibrated relative to IPL regulars! For cold-start players with $< 30$ prior balls, $\hat{\theta}_{<m}$ shrinks toward the average skill of known players on their team (`playing_eleven.csv`).

3. **Single Point of Adjustment (No Double Penalty):**
   Opposing player skill $\hat{\theta}_{\text{opp},<m}$ enters **once**—inside the expected outcome $\mathbb{E}[U_e^{\text{role}} \mid \mathbf{X}_e^{-}]$. When a batter scores a boundary off an Associate bowler with weak $\hat{\theta}_{\text{bowl},<m}$, expected runs $\hat{\mu}_{\text{runs}}^{\text{bat}}(\mathbf{X}_e^{-})$ is already high and expected dismissal risk $\hat{q}_{\text{dismiss}}^{\text{bat}}(\mathbf{X}_e^{-})$ is low, yielding a smaller $\text{NBV}_e^{\text{bat}}$. We do **not** multiply the delivery weight $w_e$ by a second opposition penalty. (For descriptive transparency in the Top 10 tables, we report each player's share of balls in IPL vs. Full-Member T20Is vs. Associate/A-Team matches using the fixed list of 12 ICC Full Members: *Afghanistan, Australia, Bangladesh, England, India, Ireland, New Zealand, Pakistan, South Africa, Sri Lanka, West Indies, Zimbabwe*.)

### 4.2 Cold-Start Venue Pitch Par vs. Out-of-Fold Expected Utility

* **Cold-Start Venue Par Fallback:** Because causal pitch par $\text{Par}_{m,e}^{-}(v, \text{overid}_e)$ uses only matches chronologically prior to match $m$ at `venue_id` $v$, early 2022 matches (or newly introduced grounds) have $n_{v,<m} < 36$ prior legal balls. Whenever $n_{v,<m} < 36$, the historical prior $\text{Par}_{\text{hist},<m}(v, o)$ falls back to a hierarchical shrinkage blend of the **`venue_country`** $\times$ **`match_format` phase mean** and the **global dataset phase mean**.

* **Why Causal Pitch Par and Out-of-Fold Expected Utility Work Together:**

  * **Pitch Par (**$\text{Par}_{m,e}^{-}$**) must be strictly causal (**$<e$**)** because using deliveries after ball $e$ in the *same* match $m$ would leak the match's eventual outcome into the pre-ball pressure score $\text{DPI}_{i,e}^{\text{role, v5}}$ (for instance, knowing in Over 3 that the team eventually scored 220).

  * **Out-of-Fold Expected Utility (**$\mathbb{E}[U_e \mid \mathbf{X}_e^{-}]$**) is trained across 5 match-grouped folds** (where match $m$ is held out completely from its training fold) to learn structural cricket relationships—such as the run cost of a wicket $\hat{\Omega}_e$ by over and wickets remaining, or the average effect of `LHB` vs. `Slow left-arm orthodox`—with maximum statistical precision across all 880 matches while guaranteeing that match $m$ never predicts itself.

### 4.3 Solving the 50/50 Blend Pathology: Baseline-Floored Uplift & Reliability-Weighted Shrinkage

A naive 50/50 blend of Clutch and Uplift simplifies to $S = 0.50\,C + 0.50(C - O) = C - 0.50\,O$, which perversely rewards a player with poor ordinary performance ($C = 0, O = -2 \implies S = +1$) equally to a player who excels in both ($C = +2, O = +2 \implies S = +1$). Moreover, because Uplift ($C - O$) is the difference of two noisy sample means, its error variance is $\text{Var}(\hat{C}) + \text{Var}(\hat{O})$, making it substantially noisier than $\hat{C}$ alone.

We resolve both issues in Step 8 and Step 9 of the pipeline:

1. **Baseline-Floored Uplift (**$\Delta_{\text{uplift},p}^{(6)}$**):**
   We define a player's baseline reference as the **greater** of their context-matched ordinary baseline $\overline{\text{NBV}}_{p,\text{ord}}^{(6)}$ and the role replacement level $\mu_{\text{rep, role}}$:
   

   $$
   \Delta_{\text{uplift},p}^{(6)} = \overline{\text{NBV}}_{p,\text{clutch}}^{(6)} - \max\!\Big(\mu_{\text{rep, role}}, \; \overline{\text{NBV}}_{p,\text{ord}}^{(6)}\Big)
   $$

   
   This ensures that a sub-replacement ordinary player is **never given a bonus for being poor in normal play**, while a genuine big-game riser who elevates above a solid ordinary baseline receives positive uplift credit.

2. **Separate Empirical Bayes Shrinkage of Level and Uplift:**
   Using the odd/even match split-half variance decomposition on the dataset, we estimate separate shrinkage constants $\hat{\kappa}_{\text{level, role}} = \hat{\sigma}_{\text{noise, level}}^2 / \hat{\sigma}_{\text{signal, level}}^2$ and $\hat{\kappa}_{\text{uplift, role}} = \hat{\sigma}_{\text{noise, uplift}}^2 / \hat{\sigma}_{\text{signal, uplift}}^2$:
   

   $$
   \tilde{C}_p = \frac{N_{\text{eff},p}}{N_{\text{eff},p} + \hat{\kappa}_{\text{level, role}}} \cdot \overline{\text{NBV}}_{p,\text{clutch}}^{(6)} + \frac{\hat{\kappa}_{\text{level, role}}}{N_{\text{eff},p} + \hat{\kappa}_{\text{level, role}}} \cdot \mu_{\text{rep, role}}
   $$

   $$
   \tilde{U}_p = \frac{N_{\text{eff, diff},p}}{N_{\text{eff, diff},p} + \hat{\kappa}_{\text{uplift, role}}} \cdot \Delta_{\text{uplift},p}^{(6)} + \frac{\hat{\kappa}_{\text{uplift, role}}}{N_{\text{eff, diff},p} + \hat{\kappa}_{\text{uplift, role}}} \cdot 0
   $$

   
   where $N_{\text{eff, diff},p} = \left(N_{\text{eff, clutch},p}^{-1} + N_{\text{eff, ord},p}^{-1}\right)^{-1}$ is the effective harmonic sample size of the difference.

3. **Reliability-Weighted Combination:**
   Let $\hat{\rho}_{\text{level}} \in [0, 1]$ and $\hat{\rho}_{\text{uplift}} \in [0, 1]$ be the empirical split-half reliabilities of $\tilde{C}_p$ and $\tilde{U}_p$ measured on the dataset. We set the combination weights directly from their measured reliabilities:
   

   $$
   w_{\text{level}} = \frac{\hat{\rho}_{\text{level}}}{\hat{\rho}_{\text{level}} + \hat{\rho}_{\text{uplift}}}, \qquad w_{\text{uplift}} = \frac{\hat{\rho}_{\text{uplift}}}{\hat{\rho}_{\text{level}} + \hat{\rho}_{\text{uplift}}}, \qquad \tilde{S}_p = w_{\text{level}} \cdot \tilde{C}_p + w_{\text{uplift}} \cdot (\mu_{\text{rep, role}} + \tilde{U}_p)
   $$

   
   Because absolute clutch level has higher split-half repeatability than the difference score, $w_{\text{level}}$ naturally takes the lead ($\approx 0.70\text{–}0.80$) while shrunk uplift $\tilde{U}_p$ ($\approx 0.20\text{–}0.30$) rewards players who specifically elevate their game under pressure. Both numbers are displayed side-by-side in the Top 10 tables.

## 5. Standalone 14-Step Executable Ranking Pipeline

 1. **Step 1 (Data Cleaning, Modal Style Normalization & Score Rebuild):** Drop abandoned/no-result matches and isolate Super Overs (`innings in (3, 4)`). Normalize each `bowler_id` to their modal `bowling_type`. Reconstruct $\Delta n_e$, $n_e^{-}$, $B_{\text{rem},e}^{-}$, $\text{ExtraRuns}_e$, and $\text{Score}_e^{-}$, and verify against `matches.result` run and wicket margins.

 2. **Step 2 (Pre-Match Player Bridge & Causal Pitch Par):** Fit pre-match player skills $\hat{\theta}_{\text{bat},<m}, \hat{\theta}_{\text{bowl},<m}$, remaining batting quality $\text{RBQ}_e^{-}$, and causal match-day pitch par $\text{Par}_{m,e}^{-}(v, \text{overid}_e)$ (with `venue_country` $\times$ `match_format` cold-start fallback).

 3. **Step 3 (Out-of-Fold Win-Probability & Contest Leverage Gate):** Generate out-of-fold pre-ball Win / Par-Achievement probability $p_e^{-}$, Contest Closeness Gate $\mathcal{L}_{i,e}^{-} = 4 p_e^{-}(1 - p_e^{-})$, and single-ball counterfactual swing $\Delta \text{WP}_e^{-} = \text{abs}(\hat{p}(\text{Bnd}_e) - \hat{p}(\text{Wkt}_e))$.

 4. **Step 4 (Pure State Pressure** $\text{DPI}_{i,e}^{\text{role, v5}}$ **& Clutch Gate):** Compute $\text{DPI}_{i,e}^{\text{bat, v5}}$ and $\text{DPI}_{i,e}^{\text{bowl, v5}}$ and flag clutch deliveries via $\text{IsClutch}_{i,e}^{\text{role}}$ (Section 2.1).

 5. **Step 5 (Out-of-Fold Expected Utility & Wicket Cost** $\hat{\Omega}_e$**):** Across 5 match-grouped folds, estimate $\hat{\Omega}_e$, expected runs $\hat{\mu}_{\text{runs}}^{\text{role}}(\mathbf{X}_e^{-})$, expected wicket/dismissal probabilities $\hat{q}^{\text{role}}(\mathbf{X}_e^{-})$, and empirical extras costs $\hat{c}_{\text{wide}}, \hat{c}_{\text{noball}}$.

 6. **Step 6 (Delivery Net Ball Value** $\text{NBV}_e^{\text{role}}$**):** Compute $\text{NBV}_e^{\text{bat}}$ and $\text{NBV}_e^{\text{bowl}}$ for every delivery event $e$ (`dismissal_type_id == 9` Retire Out excluded from dismissals; `RunOut` excluded from bowler wicket credit).

 7. **Step 7 (Capped Delivery Evaluation Weight** $w_e$**):**
    Combine **Recency Weight** ($w_{\text{time},m} = \exp(-0.18 \cdot (t_{\max} - t_m))$ in years, prioritizing recent 2025–2026 form), **Single-Ball Clutch Weight** ($W_{i,e}^{\text{clutch}} = 0.70(\text{DPI}_{i,e}^{\text{role, v5}}/100) + 0.30\min(1, \Delta \text{WP}_e^{-}/0.15)$), **Match Stakes** ($\mathcal{M}_{\text{stakes},m} \in \{1.00, 1.08, 1.15, 1.20\}$ from `knock_out` and `stage`), and **Self-Created Pressure Endogeneity Discount** ($\eta_{\text{endo},e} = \text{clip}(1 + 0.08\min(0, \text{SelfNBV}_{p,m,<e} + 2.0), 0.60, 1.00)$):
    

    $$
    w_e = \text{clip}\!\Big(w_{\text{time},m} \cdot W_{i,e}^{\text{clutch}} \cdot \mathcal{M}_{\text{stakes},m} \cdot \eta_{\text{endo},e}, \; 0.15, \; 2.50\Big)
    $$

 8. **Step 8 (Per-6-Ball Clutch Mean & Context-Matched Ordinary Baseline):**
    Compute $\overline{\text{NBV}}_{p,\text{clutch}}^{(6)} = 6 \times \frac{\sum_{e \in \text{Clutch}_p} w_e \text{NBV}_e}{\sum_{e \in \text{Clutch}_p} w_e}$, and compute $\overline{\text{NBV}}_{p,\text{ord}}^{(6)}$ by re-weighting player $p$'s ordinary deliveries ($\text{IsClutch}=0$) to match player $p$'s clutch `(innings, phase)` distribution.

 9. **Step 9 (Provisional Eligibility Audit & Separate Empirical Bayes Shrinkage):**
    Audit the dataset's player sample-size distribution to verify that provisional eligibility thresholds—**Batters:** $\ge 75$ **legal clutch balls across** $\ge 10$ **matches in** $\ge 2$ **seasons**; **Bowlers:** $\ge 60$ **legal clutch balls across** $\ge 10$ **matches in** $\ge 2$ **seasons**—retain a broad candidate pool across all tactical roles (adjusting proportionally if needed after the script audit). Compute $\tilde{C}_p$, $\tilde{U}_p$, and reliability-weighted $\tilde{S}_p$ (Section 4.3).

10. **Step 10 (Cross-Role Z-Score Standardization** $Z_p$**):**
    Assign each player to their modal phase role (Batters: *Top-Order Powerplay*, *Middle-Order*, *Death Finisher*; Bowlers: *Powerplay New-Ball*, *Middle-Overs Spinner/Enforcer*, *Death Specialist*) and standardize across roles:
    

    $$
    Z_p = \frac{\tilde{S}_p - \mu_{\text{rep, role}}}{s_{\text{role}}}
    $$

11. **Step 11 (Explicit Scale-Invariant Consistency Modifier** $\mathcal{R}_p$**):**
    For each match $m$ with $\ge 4$ clutch balls, let $I_{p,m}^{(6)}$ be the player's per-6-ball match clutch NBV. Compute shrunk Match Hit Rate $\widehat{\text{HR}}_p = \frac{\sum_m \mathbb{I}(I_{p,m}^{(6)} \gt 0) + 4}{M_{\text{clutch},p} + 8}$ and role-percentile of Sortino Downside Deviation $P_{\text{role}}(\sigma_{\text{down},p}) \in [0, 1]$, where $\sigma_{\text{down},p} = \sqrt{\frac{1}{M_{\text{clutch},p}}\sum_m \min(0, I_{p,m}^{(6)})^2}$:
    

    $$
    \mathcal{C}_p = 0.60 \cdot \widehat{\text{HR}}_p + 0.40 \cdot \big(1 - P_{\text{role}}(\sigma_{\text{down},p})\big), \qquad \mathcal{R}_p = \text{clip}\!\Big(0.24 \cdot (\mathcal{C}_p - 0.50), \; -0.12, \; +0.12\Big)
    $$

    $$
    Z_p^{\text{adj}} = Z_p \cdot (1 + \mathcal{R}_p) + 0.15 \cdot \mathcal{R}_p
    $$

12. **Step 12 (Final Presentation Rating & Tie-Breaking):**
    

    $$
    \text{Rating}_p = \text{clip}\!\Big(50 + 10 \cdot Z_p^{\text{adj}}, \; 0, \; 100\Big)
    $$

    
    Rank by posterior mean $Z_p^{\text{adj}}$; break ties within $0.02\text{ Z-units}$ by higher 95% lower confidence bound $\text{LCB}_{95,p}$, then larger $N_{\text{eff},p}$. Map `player_id` to exact `player_name` from `competition_team_player_mapping.csv`.

13. **Step 13 (Repeatability, Ablation & 2,000-Iteration Match-Block Bootstrap):**
    Run 2,000 match-cluster bootstrap resamples for 95% CIs and $\Pr(\text{Rank}_p \le 10)$; run Odd/Even Match Split-Half and 2022–2024 $\to$ 2025–2026 temporal holdout tests; and run leave-one-pillar-out ablations against a Simple 3-Variable Baseline Model.

14. **Step 14 (Export Judge-Ready PDF/DOCX Report & Top-10 Tables):**
    Export the final Methodology Report to PDF/DOCX with rendered native equations and tables (no raw LaTeX strings) showing both traditional clutch cricket stats (Strike Rate, Average, Dot %, Boundary %, Economy, Wickets/Over) and NBV metrics.

## Appendix A: Transparent Catalog of Data-Estimated Parameters vs. Structural Design Choices

| Parameter / Component | How It Is Determined in v5.0 | Value / Estimator | Sensitivity Perturbation Grid Tested in Section 5 | 
| ----- | ----- | ----- | ----- | 
| **Wicket Run-Equivalence (**$\hat{\Omega}_e$**)** | **Estimated from data pack** | Out-of-fold remaining-runs loss surface $\hat{G}(W_e^{-}) - \hat{G}(W_e^{-}+1)$ | Compare learned surface vs. parametric resource table | 
| **Expected Runs & Dismissal (**$\hat{\mu}_{\text{runs}}, \hat{q}$**)** | **Estimated from data pack** | 5-fold match-grouped out-of-fold model with player bridge $\hat{\theta}_{<m}$ | Compare Ridge mixed-effects vs. LightGBM out-of-fold | 
| **Extras Opportunity Cost (**$\hat{c}_{\text{wide}}, \hat{c}_{\text{noball}}$**)** | **Estimated from data pack** | Excess over-runs regression on training folds | Test $\pm 25\%$ around empirical point estimates | 
| **Shrinkage Constants (**$\hat{\kappa}_{\text{level}}, \hat{\kappa}_{\text{uplift}}$**)** | **Estimated from data pack** | Empirical split-half noise-to-signal ratio $\hat{\sigma}_{\text{noise}}^2 / \hat{\sigma}_{\text{signal}}^2$ | Perturb $\kappa \in [0.65\,\hat{\kappa}, \; 1.35\,\hat{\kappa}]$ | 
| **Level vs. Uplift Blend (**$w_{\text{level}}, w_{\text{uplift}}$**)** | **Estimated from data pack** | Proportional to measured split-half reliabilities $\hat{\rho}_{\text{level}} : \hat{\rho}_{\text{uplift}}$ | Compare reliability-weighted vs. fixed $100/0$, $70/30$, and $50/50$ | 
| **Pillar Weights (**$w_1, w_2, w_3, w_4$**)** | **Design choice (Ablated)** | $(0.30, 0.30, 0.20, 0.20)$ across Rate, Wickets, Phase, Micro-Stress | Equal weights $(0.25\times 4)$, Dirichlet perturbations, and leave-one-out | 
| **Contest Gate Exponent (**$\gamma$**) & Floor** | **Design choice (Ablated)** | $\mathcal{L}_{i,e}^{-} = 4 p_e^{-}(1-p_e^{-})$ with $\gamma = 0.60$ and phase-gate floor $\mathcal{L} \ge 0.50$ | Test $\gamma \in \{0.40, 0.60, 0.80, 1.00\}$ and $\mathcal{L}_{\min} \in \{0.35, 0.50, 0.65\}$ | 
| **Clutch Gate Thresholds** | **Design choice (Ablated)** | Global $\text{DPI} \ge 60$ or ($\text{DPI} \ge Q_{0.70}^{\text{phase}} \land \text{DPI} \ge 45 \land \mathcal{L} \ge 0.50$) | Test global cutoffs $\{55, 60, 65\}$ and phase quantiles $\{Q_{0.65}, Q_{0.70}, Q_{0.75}\}$ | 
| **Match Stakes Multipliers (**$\mathcal{M}_{\text{stakes},m}$**)** | **Design choice (Ablated)** | `Final`: $1.20$, `knock_out=1`/Playoffs: $1.15$, `Super 4`: $1.08$, `League`: $1.00$ | Compare against flat $\mathcal{M}_{\text{stakes},m} = 1.00$ across all games | 
| **Recency Half-Life (**$\lambda_{\text{time}}$**)** | **Design choice (Ablated)** | $\lambda_{\text{time}} = 0.18\text{ yr}^{-1}$ ($\sim 3.8\text{-year}$ half-life across 2022–2026) | Test $\lambda_{\text{time}} \in \{0.00 \text{ (unweighted)}, \; 0.18, \; 0.35\}$ | 
| **Provisional Eligibility Cutoffs** | **Design choice (Data-Audited)** | Batters: $\ge 75$ clutch balls & $\ge 10$ matches; Bowlers: $\ge 60$ balls & $\ge 10$ matches | Audit candidate counts at $\{45, 60, 75, 90\}$ clutch balls before locking | 
