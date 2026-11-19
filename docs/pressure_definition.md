# Pressure definition

Written by the Pressure_Engine in the `pressure` stage of the Pipeline (Requirement 13, criterion 9). Every value below is the value of the Run that wrote this file.

## Plain-language definition

A ball is a pressure ball for a player when three things hold before it is bowled:

1. **The match is live.** The batting side's chance of winning is between 5% and 95%, so the result is still open.
2. **The ball can swing the result.** The change in the win probability that this one ball can cause (its leverage) is large compared with other live balls.
3. **The situation is harder than usual for the player's Role.** The stress on the batter (scoreboard deficit, wickets lost, being new at the crease, being squeezed) or on the bowler (scoreboard threat, a set and strong batter, an over going badly, a wicket drought) is high compared with other live balls.

Leverage and stress are combined into one Pressure_Index (PI) with weight 0.5 on leverage and 0.5 on stress, so a ball can qualify through very high leverage with ordinary stress, or the reverse. The 25% of live balls with the highest PI are the pressure balls, chosen separately for batters and bowlers and within each Innings_Group. Every other ball of the Role is an ordinary ball. PI uses only the match state before the ball, never what happened on it.

## Formal definitions

A Delivery is one row of `ball_by_ball.csv` in innings 1 or 2. A Batter_Delivery is a Legal_Ball or a No_Ball, credited to the striker (a Wide is not a Batter_Delivery); a Bowler_Delivery is any Delivery, credited to the bowler. An Innings_Group is all Deliveries with the same innings number across all matches. The Percentile_Rank of a value x within a reference set of n values is rank(x) = (number of reference values below x + 0.5 × number of reference values equal to x) ÷ n.

**Live_Ball (D1).** A Delivery is a Live_Ball when 0.05 ≤ WP ≤ 0.95, where WP is the batting side's pre-ball win probability: the unrounded Out_Of_Fold_Prediction of the M3 Primary_Model of the Delivery's innings, after the terminal rules. Every other Delivery is a Dead_Ball.

**Leverage_Index (LI).** LI = [Σ over the 9 outcome classes c of P(c) × |WP(state after c) − WP(state before the ball)|] ÷ m. P(c) is the M4 Primary_Model's probability of class c for an average batter and an average bowler (θ_bat and θ_bowl at Average_Skill); the state after c adds one Legal_Ball, the class's runs and, for a wicket class, the Post_Wicket_State; the WPs come from the M3 Primary_Model of the innings. m is the mean of the bracketed quantity over all Deliveries of innings 1 and 2 (m = 0.012572 in this Run), so the mean LI is 1. The batter Role and the bowler Role of a Delivery share its LI.

**Stress (D4, D5).** Stress_bat = 0.25 × (F19 + F20 + F21 + F22) on Batter_Deliveries and Stress_bowl = 0.25 × (F23 + F24 + F25 + F26) on Bowler_Deliveries. Each Stress_Pillar is a Percentile_Rank within its Pillar_Reference_Set, the Role's Live_Balls in the same Innings_Group: F19, F21, F23 and F26 are the rank of their one Pillar_Component; F20, F22, F24 and F25 are the rank of the mean of their components' ranks among the same means of the reference set. A larger component always means more stress. A component that cannot be computed gets rank 0.5 and leaves the reference set. Dead_Balls are ranked against the same reference set.

| Pillar | Role | Pillar_Component(s) |
|---|---|---|
| F19 scoreboard deficit | batter | innings 1: Par_Score at the pre-ball Legal_Balls bowled − team score (runs); innings 2: Required_Rate − Achievable_Rate (runs per over) |
| F20 wickets and depth | batter | (a) wickets fallen − the Training_Folds mean wickets fallen at the same innings and Legal_Balls bowled; (b) Counted_Wickets in the last 18 Legal_Balls (D6); (c) 1 − RBQ |
| F21 new at the crease | batter | 20 − min(striker's Legal_Balls faced, 20) (D7) |
| F22 squeeze | batter | (a) the striker's current run of Dot_Balls; (b) the team's Legal_Balls since the last Boundary |
| F23 scoreboard threat | bowler | innings 1: team run rate − Par_Rate over the last 18 Legal_Balls (D6); innings 2: Achievable_Rate − Required_Rate (runs per over) |
| F24 batter threat | bowler | (a) striker's Legal_Balls faced × θ_bat; (b) RBQ |
| F25 over going badly | bowler | (a) team runs off the bowler earlier in the over, byes and leg-byes left out; (b) 1 when the previous Delivery of the over was a Boundary or an extra, else 0 |
| F26 wicket drought | bowler | Legal_Balls since the last Counted_Wicket |

**Pressure_Index (PI, D2).** For a Live_Ball and a Role: PI = 100 × [α × rank(LI) + (1 − α) × rank(Stress)] with α = 0.5, both ranks taken among the Role's Live_Balls in the same Innings_Group, so 0 < PI < 100. For a Dead_Ball, PI = 0. A Wide has no batter-Role PI.

**Pressure_Ball (D3).** A Live_Ball whose rank(PI) among the Role's Live_Balls in the same Innings_Group is greater than 1 − 0.25 = 0.75: the top 25%. Deliveries with equal PI get the same flag. A Dead_Ball is never a Pressure_Ball. Every other Batter_Delivery or Bowler_Delivery of the Role is an Ordinary_Ball.

**Extras_Free_PI (validation only).** The bowler-Role PI recomputed with a previous-Delivery flag that counts Boundaries only in F25(b), every other input unchanged. Its top 25% re-selects bowler Pressure_Balls for the extras check of Requirement 14, criterion 5; it never feeds the ratings.

## Design_Choices D1–D5 of this Run

| ID | Choice | Value in this Run | Label |
|---|---|---|---|
| D1 | Live band on WP | 0.05 ≤ WP ≤ 0.95 | hand-set (D1) |
| D2 | Leverage/Stress blend α | 0.5 | hand-set (D2) |
| D3 | Pressure share | 25% of Live_Balls per Role and Innings_Group | hand-set (D3) |
| D4 | Percentile scope | within each Innings_Group (all first innings; all second innings) | hand-set (D4) |
| D5 | Stress_Pillar weights | equal, 0.25 for each of the Role's four pillars | hand-set (D5) |

## Values derived from the data in this Run

| Value | This Run | Label |
|---|---|---|
| m, the pooled mean unscaled LI | 0.012572 | data-derived (Pressure_Engine) |
| Primary_Models behind WP, LI and PI | M3-1 logistic baseline, M3-2 logistic baseline, M4 LightGBM | data-derived (Model_Choice_Checker) |

| PI | Innings_Group | Role Deliveries | Live_Balls | Pressure_Balls | Pressure_Balls as % of Live_Balls | Lowest PI of a Pressure_Ball |
|---|---|---|---|---|---|---|
| batter | 1 | 101,639 | 100,082 | 25,020 | 25.00 | 64.939 |
| batter | 2 | 91,612 | 68,134 | 17,034 | 25.00 | 62.753 |
| bowler | 1 | 105,727 | 104,105 | 26,026 | 25.00 | 64.663 |
| bowler | 2 | 95,234 | 70,730 | 17,682 | 25.00 | 66.353 |
| bowler Extras_Free_PI | 1 | 105,727 | 104,105 | 26,027 | 25.00 | 64.644 |
| bowler Extras_Free_PI | 2 | 95,234 | 70,730 | 17,682 | 25.00 | 66.341 |
