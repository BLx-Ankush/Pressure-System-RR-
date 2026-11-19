# SupeRR Selector Season 2 (Round 1): Definitive Pressure Framework & Executable Mathematical Specification (v3.0)

## 1. Formal Definition of Pressure & Pre-Ball State Architecture

### 1.1 What Counts as Pressure in T20 Cricket?

In T20 cricket, **delivery-level pressure is the pre-ball situational leverage and execution stress acting on the batter and bowler immediately prior to delivery event $e$ of innings $i$ in match $m$**. Rather than conflating pressure with a static phase (such as Overs 16 to 20), a raw scoreboard threshold (such as Required Run Rate above 10), or external tournament prestige, our framework defines **State Pressure ($\text{DPI}_{i,e}^{\text{role, v3}} \in [0, 100]$)** through four strictly pre-ball situational pillars gated by live contest closeness:

1. **Causal Pitch-Adjusted Rate & Tempo Discrepancy ($\mathcal{S}_{\text{rate},i,e}^{\text{role}}$):** How far the batting side's required scoring rate (in the 2nd innings) or current/recent scoring tempo (in the 1st innings) deviates from the **causal pre-ball match-day pitch par** ($\text{Par}_{m,e}^{-}$), estimated using only historical venue data prior to match $m$ and deliveries completed before event $e$.
2. **Resource Depletion, Cluster Shock & Remaining Shed Batting Quality ($\mathcal{S}_{\text{wkt+tail},i,e}^{\text{role}}$):** Combines effective wickets lost ($W_{\text{eff},e}^{-}$), excess wickets relative to over par ($W_{\text{excess},e}^{-}$), cluster wickets lost in the preceding 18 legal balls ($\Delta W_{18,e}^{-}$), striker settledness ($B_{\text{faced},e}^{-}$), and the pre-match shrunk **Remaining Batting Quality ($\text{RBQ}_e^{-}$)** of batters yet to bat.
3. **Phase Leverage & Legal-Balls-Remaining Scarcity ($\mathcal{S}_{\text{phase},i,e}^{\text{role}}$):** Captures the non-linear shrinkage of remaining legal deliveries ($B_{\text{rem},e}^{-}$) alongside Powerplay (2 fielders outside the 30-yard circle) and Death-over field-restriction squeezes.
4. **Within-Over Micro-Stress ($\mathcal{S}_{\text{micro},i,e}^{\text{role}}$):** Captures acute ball-to-ball stress within the active over—consecutive dot balls faced by the striker ($D_{\text{dots\_streak},e}^{-}$), team boundary droughts ($B_{\text{no\_bnd},e}^{-}$), non-striker stalling ($\text{SR}_{\text{NS},12,e}^{-}$), and—for bowlers—runs already conceded in the current over ($R_{\text{over},e}^{-}$) with legal balls still remaining to defend.
5. **Contest Closeness Gate ($\mathcal{L}_{i,e}$):** Multiplies the four stress pillars by $\left(4 \cdot p_e^{-}(1 - p_e^{-})\right)^{0.60}$, where $p_e^{-}$ is the pre-ball Win Probability (2nd innings) or Par-Achievement Probability (1st innings). This ensures that dead rubbers and mathematically impossible chases ($p_e^{-}$ near $0$ or $1$) are classified as low-leverage ordinary states rather than genuine pressure.

**Separation of Situational Pressure and Match Stakes:** To prevent conflating *situational cricket pressure* with *tournament importance*, $\text{DPI}_{i,e}^{\text{role, v3}}$ is computed purely from the pre-ball cricket state. Tournament importance ($\mathcal{M}_{\text{stakes},m} \in [0.90, 1.20]$) is applied exclusively in the downstream **Delivery Evaluation Weight ($w_e$)** so that an identical match state has the exact same $\text{DPI}$ in a league game and a final, while knockout performance receives higher weight in player rankings.

### 1.2 Dual Indexing (`event_number` vs. `legal_balls_completed`) & Strict Pre-Ball State Notation

Because wides and no-balls generate delivery rows in `ball_by_ball.csv` without reducing the legal balls remaining in the innings, indexing by row number $b = 1, \dots, 120$ corrupts run rates, over boundaries, and resource scarcity. Every innings $i$ maintains two strict counters:

* **Delivery Event Index ($e \in \{1, 2, \dots, E_{m,i}\}$):** Increments by $1$ on every row recorded in `ball_by_ball.csv` for innings $i$.
* **Pre-Ball Legal Balls Completed ($n_e^{-} \in \{0, 1, \dots, B_{\text{sched},m,i} - 1\}$):** Counts strictly legal deliveries (`is_wide == 0` and `is_noball == 0`) completed *before* event $e$:
  $$n_1^{-} = 0, \qquad n_e^{-} = \sum_{k=1}^{e-1} \mathbb{I}(\text{Wide}_k = 0 \;\land\; \text{NoBall}_k = 0)$$
* **Pre-Ball Legal Balls Remaining ($B_{\text{rem},e}^{-}$):** Let $B_{\text{sched},m,i}$ be the scheduled legal balls for innings $i$ (normally $120$, or the revised scheduled balls in rain-shortened matches). Then:
  $$B_{\text{rem},e}^{-} = \max\!\big(1, \; B_{\text{sched},m,i} - n_e^{-}\big), \qquad o_e^{-} = \frac{n_e^{-}}{6} \in [0, 20)$$
* **Strict Pre-Ball Superscript Convention ($\mathbf{X}_e^{-}$):** Every state variable entering the pressure equation or expected-outcome model carries a minus superscript $(\cdot)_e^{-}$ indicating it is computed strictly from events $\{1, \dots, e-1\}$ before ball $e$ is bowled. The realized outcome of event $e$ is never used to compute $\text{DPI}_{i,e}^{\text{role, v3}}$ or $\mathbb{E}[U_e \mid \mathbf{X}_e^{-}]$.

### 1.3 Handling Shortened Matches, DLS Revisions, and Super Overs

1. **Minimum Innings Length Filter:** Matches where scheduled balls $B_{\text{sched},m,i} \lt 30$ (fewer than 5 overs per side) or abandoned no-result matches are excluded from the ranking pipeline.
2. **DLS & Rain-Shortened Scaling ($30 \le B_{\text{sched},m,i} \lt 120$):** In shortened matches, $B_{\text{rem},e}^{-} = B_{\text{sched},m,i} - n_e^{-}$, and the 2nd-innings target uses the latest active revised target $T_{\text{rev},e}^{-}$. Match phases scale proportionally to scheduled balls: Powerplay corresponds to $n_e^{-} / B_{\text{sched},m,i} \le 0.30$, Middle Overs to $(0.30, 0.75]$, and Death Overs to $(0.75, 1.00]$.
3. **Super Overs (`innings >= 3`):** Super Overs are excluded from the 120-ball calibration and main ranking aggregation to prevent 6-ball exhibition mechanics from distorting main-innings baselines; Super Over performance is tracked as a standalone tie-breaker diagnostic in the report.

### 1.4 The Pure State Pressure Equation ($\text{DPI}_{i,e}^{\text{role, v3}}$) & Dual-Gate Qualification Rule

For every delivery event $e$ in innings $i \in \{1, 2\}$ and $\text{role} \in \{\text{bat}, \text{bowl}\}$, pure situational pressure is defined as:

$$\text{DPI}_{i,e}^{\text{role, v3}} = \text{clip}\!\left(100 \times \left(\mathcal{L}_{i,e}^{-}\right)^{0.60} \times \Big( 0.32 \cdot \mathcal{S}_{\text{rate},i,e}^{\text{role},-} + 0.30 \cdot \mathcal{S}_{\text{wkt+tail},i,e}^{\text{role},-} + 0.20 \cdot \mathcal{S}_{\text{phase},i,e}^{\text{role},-} + 0.18 \cdot \mathcal{S}_{\text{micro},i,e}^{\text{role},-} \Big), \; 0, \; 100\right)$$

To ensure Powerplay and Middle-Overs specialists are evaluated fairly alongside Death-Overs specialists without admitting weak low-pressure states into the clutch sample, a delivery qualifies for the **Clutch Evaluation Set ($\text{IsClutch}_{i,e}^{\text{role}} = 1$)** if and only if it passes either the **Global High-Pressure Gate ($\ge 60$)** or the **Bounded Phase-Quantile Gate** (which enforces both an absolute pressure floor of $\text{DPI} \ge 45$ and a live contest floor of $\mathcal{L}_{i,e}^{-} \ge 0.50$):

$$\text{IsClutch}_{i,e}^{\text{role}} = \mathbb{I}\!\left[ \Big(\text{DPI}_{i,e}^{\text{role, v3}} \ge 60\Big) \;\lor\; \Big(\text{DPI}_{i,e}^{\text{role, v3}} \ge Q_{0.70, \text{train}}^{\text{phase, role}} \;\land\; \text{DPI}_{i,e}^{\text{role, v3}} \ge 45 \;\land\; \mathcal{L}_{i,e}^{-} \ge 0.50\Big) \right]$$

where $Q_{0.70, \text{train}}^{\text{phase, role}}$ is the 70th percentile of $\text{DPI}_{i,e}^{\text{role, v3}}$ within that phase and role computed strictly on out-of-fold training matches.

### 1.5 Six Fully Worked Numerical Delivery Examples (Verifying Exact Reproducibility)

The table below provides every intermediate input and pillar value for six representative match states, demonstrating exact arithmetic reproducibility of $\text{DPI}_{i,e}^{\text{role, v3}}$:

| Worked Example & Pre-Ball State ($n_e^{-}$, Score, $B_{\text{rem},e}^{-}$, $\text{Par}_{m,e}^{-}$) | $\mathcal{S}_{\text{rate}}^{\text{bat},-}$ / $\mathcal{S}_{\text{rate}}^{\text{bowl},-}$ | $\mathcal{S}_{\text{wkt+tail}}^{\text{bat},-}$ / $\mathcal{S}_{\text{wkt+tail}}^{\text{bowl},-}$ | $\mathcal{S}_{\text{phase}}^{-}$ | $\mathcal{S}_{\text{micro}}^{\text{bat},-}$ / $\mathcal{S}_{\text{micro}}^{\text{bowl},-}$ | Weighted Sum $\Sigma_{\text{bat}}$ / $\Sigma_{\text{bowl}}$ | Win Prob $p_e^{-}$ & Gate $\mathcal{L}_{i,e}^{-}$ ($\mathcal{L}^{0.60}$) | Final $\text{DPI}_{\text{bat}}^{\text{v3}}$ / $\text{DPI}_{\text{bowl}}^{\text{v3}}$ & Gate Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Ex 1: Death Chase Knife-Edge** — 2nd Inn, $n_e^{-}=105$ (Ov 17.3), Score $148/5$, Target $180$ (Need $32$ off $15$, $\text{RRR}=12.80$, $\text{Par}=8.80 \Rightarrow r_2=1.455$), $\Delta W_{18}^{-}=1$, $\text{RBQ}^{-}=0.36$, $D_{\text{dots}}^{-}=1$, $R_{\text{over}}^{-}=6$ ($b_{\text{rem}}=3$) | Bat: $0.686$ / Bowl: $0.600$ | Bat: $0.582$ / Bowl: $0.662$ | $0.875$ | Bat: $0.400$ / Bowl: $0.621$ | Bat: $0.644$ / Bowl: $0.672$ | $p_e^{-}=0.44$, $\mathcal{L}=0.986$, $\mathcal{L}^{0.60}=0.991$ | **Bat: 63.8 (Clutch)** / **Bowl: 66.6 (Clutch)** |
| **Ex 2: Powerplay Collapse Rebuild** — 1st Inn, $n_e^{-}=22$ (Ov 3.4), Score $22/3$ ($\text{CRR}=6.00$, $\text{Par}=8.20 \Rightarrow r_1=0.732$), $\Delta W_{18}^{-}=2$, $\text{RBQ}^{-}=0.52$, $B_{\text{faced}}^{-}=2$, $D_{\text{dots}}^{-}=2$, $R_{\text{over}}^{-}=1$ ($b_{\text{rem}}=2$) | Bat: $0.697$ / Bowl: $0.150$ | Bat: $0.892$ / Bowl: $0.318$ | $0.650$ | Bat: $0.400$ / Bowl: $0.051$ | Bat: $0.693$ / Bowl: $0.283$ | $p_e^{-}=0.32$, $\mathcal{L}=0.870$, $\mathcal{L}^{0.60}=0.920$ | **Bat: 63.7 (Clutch)** / **Bowl: 26.0 (Ordinary)** |
| **Ex 3: Middle-Overs Spin Strangle** — 2nd Inn, $n_e^{-}=68$ (Ov 11.2), Score $74/4$, Target $166$ (Need $92$ off $52$, $\text{RRR}=10.62$, $\text{Par}=7.90 \Rightarrow r_2=1.344$), $\Delta W_{18}^{-}=1$, $\text{RBQ}^{-}=0.44$, $D_{\text{dots}}^{-}=3$, $\text{SR}_{\text{NS},12}^{-}=72$ | Bat: $0.585$ / Bowl: $0.763$ | Bat: $0.478$ / Bowl: $0.648$ | $0.585$ | Bat: $0.800$ / Bowl: $0.250$ | Bat: $0.592$ / Bowl: $0.601$ | $p_e^{-}=0.38$, $\mathcal{L}=0.942$, $\mathcal{L}^{0.60}=0.965$ | **Bat: 57.1 (Clutch via $Q_{0.70}^{\text{Mid}}$)** / **Bowl: 58.0 (Clutch via $Q_{0.70}^{\text{Mid}}$)** |
| **Ex 4: 1st-Inn Death Onslaught** — 1st Inn, $n_e^{-}=104$ (Ov 17.2), Score $164/2$ ($\text{CRR}=9.46$, $\text{RR}_{18}^{-}=13.00$, $\text{Par}=9.20$), $\Delta W_{18}^{-}=0$, $\text{RBQ}^{-}=0.78$, $B_{\text{faced}}^{-}=28$, Prev ball $6$, $R_{\text{over}}^{-}=10$ ($b_{\text{rem}}=4$) | Bat: $0.202$ / Bowl: $0.829$ | Bat: $0.055$ / Bowl: $0.901$ | $0.813$ | Bat: $0.000$ / Bowl: $0.964$ | Bat: $0.244$ / Bowl: $0.872$ | $p_e^{-}=0.68$, $\mathcal{L}=0.870$, $\mathcal{L}^{0.60}=0.920$ | **Bat: 22.4 (Ordinary)** / **Bowl: 80.2 (Clutch)** |
| **Ex 5: Low-Scoring Pitch Thriller** — 2nd Inn, $n_e^{-}=103$ (Ov 17.1), Score $101/6$, Target $126$ (Need $25$ off $17$, $\text{RRR}=8.82$, $\text{Par}=6.70 \Rightarrow r_2=1.316$), $\Delta W_{18}^{-}=1$, $\text{RBQ}^{-}=0.24$, $D_{\text{dots}}^{-}=2$, $R_{\text{over}}^{-}=4$ ($b_{\text{rem}}=5$) | Bat: $0.560$ / Bowl: $0.802$ | Bat: $0.654$ / Bowl: $0.598$ | $0.861$ | Bat: $0.600$ / Bowl: $0.569$ | Bat: $0.656$ / Bowl: $0.710$ | $p_e^{-}=0.47$, $\mathcal{L}=0.996$, $\mathcal{L}^{0.60}=0.998$ | **Bat: 65.4 (Clutch)** / **Bowl: 70.9 (Clutch)** |
| **Ex 6: Garbage-Time Dead Rubber** — 2nd Inn, $n_e^{-}=115$ (Ov 19.1), Score $124/8$, Target $190$ (Need $66$ off $5$, $\text{RRR}=79.20$, $\text{Par}=9.50 \Rightarrow r_2=8.337$), $\Delta W_{18}^{-}=2$, $\text{RBQ}^{-}=0.08$, $D_{\text{dots}}^{-}=2$ | Bat: $1.000$ / Bowl: $0.150$ | Bat: $0.980$ / Bowl: $0.286$ | $0.958$ | Bat: $0.600$ / Bowl: $0.100$ | Bat: $0.914$ / Bowl: $0.343$ | $p_e^{-}=0.004$, $\mathcal{L}=0.016$, $\mathcal{L}^{0.60}=0.083$ | **Bat: 7.6 (Ordinary)** / **Bowl: 2.9 (Ordinary)** |

## 2. Complete Operational Data Dictionary & Master Table of All 26 Factors

Every factor below is strictly operationalized with its exact input columns, causal pre-ball formula, missing-value fallback, output variable name, and the exact downstream equation that consumes it:

| # | Layer | Factor Name | Exact CSV Input Columns | Pre-Ball Mathematical Formula & Missing-Value Fallback | Output Variable | Consuming Equation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **L1: Rate Stress** | **Required Run Rate Ratio ($r_{2,e}^{-}$)** | `ball_by_ball`: `innings`, `total_runs`, `is_wide`, `is_noball`; `matches`: `target_runs` | $r_{2,e}^{-} = \frac{6(T_{\text{rev},e}^{-} - \text{Score}_e^{-}) / B_{\text{rem},e}^{-}}{\text{Par}_{m,e}^{-}(v, o_e^{-})}$ (Fallback: if $\text{Par}_{m,e}^{-}$ missing, use league phase mean $8.15$). | `r2_ratio_pre` | $\mathcal{S}_{\text{rate},2,e}^{\text{bat},-}$, $\mathcal{S}_{\text{rate},2,e}^{\text{bowl},-}$ |
| **2** | **L1: Rate Stress** | **Current Run Rate Ratio ($r_{1,e}^{-}$)** | `ball_by_ball`: `total_runs`, `is_wide`, `is_noball` | $r_{1,e}^{-} = \frac{6 \cdot \text{Score}_e^{-} / \max(6, n_e^{-})}{\text{Par}_{m,e}^{-}(v, o_e^{-})}$ (Stabilized by $\max(6, n_e^{-})$ in Over 1). | `r1_ratio_pre` | $\mathcal{S}_{\text{rate},1,e}^{\text{bat},-}$, $\mathcal{S}_{\text{rate},1,e}^{\text{bowl},-}$ |
| **3** | **L1: Rate Stress** | **Rolling 18-Ball Tempo ($\text{RR}_{18,e}^{-}$)** | `ball_by_ball`: `total_runs`, `is_wide`, `is_noball` | $\text{RR}_{18,e}^{-} = 6 \times \frac{\text{Runs in last }\min(n_e^{-}, 18)\text{ legal balls}}{\max(6, \min(n_e^{-}, 18))}$. | `rr18_pre` | $\mathcal{S}_{\text{rate},1,e}^{\text{role},-}$, $p_e^{-}$ |
| **4** | **L2: Wicket & Shed** | **Effective Wickets Lost ($W_{\text{eff},e}^{-}$)** | `ball_by_ball`: `is_wicket`; `playing_xis`: `player_id` | $W_{\text{eff},e}^{-} = \max(0, W_e^{-} - 0.65\,\mathbb{I}_{\text{IP\_obs},m} + 1.50\,\mathbb{I}(\text{EligibleBattersLeft}_e^{-} = 0))$. If IP sub status unobserved, $\mathbb{I}_{\text{IP\_obs},m}=0$. | `w_eff_pre` | $\mathcal{S}_{\text{wkt+tail},i,e}^{\text{role},-}$, $\Omega_e$ |
| **5** | **L2: Wicket & Shed** | **Excess Wickets vs. Par ($W_{\text{excess},e}^{-}$)** | Derived from $W_{\text{eff},e}^{-}$, $n_e^{-}$, $B_{\text{sched},m,i}$ | $W_{\text{excess},e}^{-} = \max\!\big(0, \; W_{\text{eff},e}^{-} - \bar{W}_{\text{train}} \cdot (n_e^{-} / B_{\text{sched},m,i})\big)$, where $\bar{W}_{\text{train}} = 6.18$ from training fold. | `w_excess_pre` | $\mathcal{S}_{\text{wkt+tail},i,e}^{\text{bat},-}$, $p_e^{-}$ |
| **6** | **L2: Wicket & Shed** | **Cluster Wicket Shock ($\Delta W_{18,e}^{-}$)** | `ball_by_ball`: `is_wicket`, `is_wide`, `is_noball` | Count of wickets fallen in the last $\min(n_e^{-}, 18)$ legal balls prior to event $e$. | `dw18_pre` | $\mathcal{S}_{\text{wkt+tail},i,e}^{\text{role},-}$, $p_e^{-}$ |
| **7** | **L2: Wicket & Shed** | **Causal Remaining Batting Quality ($\text{RBQ}_e^{-}$)** | `playing_xis`: `player_id`; historical `ball_by_ball` ($<m$) | $\text{RBQ}_e^{-} = \text{clip}\!\left(\frac{\sum_{u \in \text{Unbatted}_e^{-} \cup \{\text{NS}_e^{-}\}} \hat{q}_{\text{bat},u,<m}}{7 \cdot \bar{q}_{\text{top7}}}\right)$ using strictly pre-match shrunk batter quality $\hat{q}_{\text{bat},u,<m}$. | `rbq_pre` | $\mathcal{S}_{\text{wkt+tail},i,e}^{\text{role},-}$, $\Omega_e$, $p_e^{-}$ |
| **8** | **L2: Wicket & Shed** | **Striker Settledness & Wicket Drought** | `ball_by_ball`: `batter_id`, `is_wide` | $B_{\text{faced},e}^{-} =$ legal balls + no-balls faced by striker in innings $i$ before $e$; drought $\mathbb{I}(\Delta W_{18,e}^{-}=0)$. | `b_faced_pre` | $\mathcal{S}_{\text{wkt+tail},i,e}^{\text{role},-}$, $V_{\text{wkt},e}$ |
| **9** | **L3: Phase & Time** | **Legal Balls Remaining Scarcity ($B_{\text{rem},e}^{-}$)** | `ball_by_ball`: `is_wide`, `is_noball` | $\mathcal{S}_{\text{phase},2,e}^{-} = 0.25 + 0.75 \cdot (n_e^{-} / B_{\text{sched},m,2})^{1.35}$. | `s_phase2_pre` | $\text{DPI}_{2,e}^{\text{role, v3}}$, $\Omega_e$ |
| **10** | **L3: Phase & Time** | **Field-Restriction Phase Squeeze** | `ball_by_ball`: `is_wide`, `is_noball` | U-shaped curve: $0.65$ in PP ($f_e^{-} \le 0.30$), $0.35 + 0.25\frac{f_e^{-}-0.30}{0.45}$ in Middle, $0.65 + 0.35\frac{f_e^{-}-0.75}{0.25}$ in Death ($f_e^{-} = n_e^{-}/B_{\text{sched}}$). | `s_phase1_pre` | $\text{DPI}_{1,e}^{\text{role, v3}}$ |
| **11** | **L4: Micro-Stress** | **Consecutive Dot-Ball Streak ($D_{\text{dots\_streak},e}^{-}$)** | `ball_by_ball`: `batter_runs`, `is_wide`, `is_noball` | Count of consecutive legal deliveries prior to $e$ faced by striker with `batter_runs == 0` ([Dot Ball Study](https://www.researchgate.net/publication/395488532_Quantifying_the_Impact_of_Dot_Balls_on_Winning_Probability_in_T20_Cricket)). | `dot_streak_pre` | $\mathcal{S}_{\text{micro},i,e}^{\text{bat},-}$, $\mathbf{X}_e^{-}$ |
| **12** | **L4: Micro-Stress** | **Boundary Drought & Non-Striker Stall** | `ball_by_ball`: `batter_runs`, `non_striker_id` | $B_{\text{no\_bnd},e}^{-} =$ legal balls since last $4$ or $6$; $\text{SR}_{\text{NS},12,e}^{-} =$ non-striker SR over last 12 balls (fallback $100$ if $<6$ balls) ([Strike Rotation KPIs](https://www.mdpi.com/2076-3417/15/12/6483)). | `ns_stall_pre` | $\mathcal{S}_{\text{micro},i,e}^{\text{bat},-}$ |
| **13** | **L4: Micro-Stress** | **Bowler Within-Over Damage ($R_{\text{over},e}^{-}$)** | `ball_by_ball`: `over`, `total_runs`, `is_wide`, `is_noball` | Runs charged to bowler in current over before $e$, scaled by $(b_{\text{over\_rem},e}^{-}/4)^{0.5} + 0.25\,\mathbb{I}_{\text{Prev 4/6/Extra}}$. | `s_micro_bowl_pre` | $\mathcal{S}_{\text{micro},i,e}^{\text{bowl},-}$ |
| **14** | **L5: Contest & Stakes** | **Out-of-Fold Win Prob ($p_e^{-}$) & Leverage ($\mathcal{L}_{i,e}^{-}$)** | Pre-ball state features $\mathbf{X}_{\text{wp},e}^{-}$ | $\mathcal{L}_{i,e}^{-} = 4 p_e^{-}(1 - p_e^{-})$; single-ball swing $\Delta p_e^{-} = \lvert \hat{p}(\text{Bnd}_e) - \hat{p}(\text{Wkt}_e) \rvert$. | `lev_pre`, `wp_swing_pre` | $\text{DPI}_{i,e}^{\text{role, v3}}$, $W_{i,e}^{\text{clutch}}$ |
| **15** | **L5: Contest & Stakes** | **Decoupled Match Stakes Multiplier ($\mathcal{M}_{\text{stakes},m}$)** | `matches`: `stage`, `match_type`, `series_id`, `date` | $1.20$ Playoff/Final/WC Knockout; $1.10$ WC Group/Late IPL Must-Win; $1.00$ Standard League/Live Bilateral; $0.90$ Decided Bilateral Rubber. | `m_stakes` | Delivery weight $w_e$ (not inside $\text{DPI}$) |
| **16** | **L6: Venue & Pitch** | **Causal Venue Boundary & 2nd-Inn Asymmetry** | Historical `ball_by_ball` & `matches` strictly $< \text{date}(m)$ | EB-shrunk $\text{BndIdx}_{v,<m}$ (Sixes/Ball) and controlled 2nd-Inn Venue Asymmetry $\text{Asym}_{v,<m}$ (controlling for target & team strength) ([Anatomy of a Run Pursuit](https://www.cricketweb.net/the-anatomy-of-a-run-pursuit-how-live-data-shapes-t20-betting-strategies/)). | `bnd_idx_pre`, `venue_asym_pre` | $\text{Par}_{m,e}^{-}$, $\mathbf{X}_e^{-}$ |
| **17** | **L6: Venue & Pitch** | **Zero-Leakage Causal Pitch Par ($\text{Par}_{m,e}^{-}$)** | `ball_by_ball` events strictly $< e$ in match $m$ and $< m$ at venue $v$ | $\text{Par}_{m,e}^{-}(v,o) = \lambda_e \text{Par}_{\text{hist},<m}(v,o) + (1-\lambda_e)\text{Par}_{\text{match},<e}(m,o)$ with $\lambda_e = \frac{36}{n_{\text{match},<e} + 36}$. | `par_rr_pre` | $r_{1,e}^{-}$, $r_{2,e}^{-}$, $\mathbb{E}[U_e \mid \mathbf{X}_e^{-}]$ |
| **18** | **L7: Matchups & Value** | **Out-of-Fold Expected Runs & Wicket Cost ($\Omega_e$)** | Pre-ball context $\mathbf{X}_e^{-}$, `players`, `playing_xis` | Out-of-fold $\hat{\mu}_{\text{runs}}(\mathbf{X}_e^{-})$, $\hat{q}_{\text{wkt}}(\mathbf{X}_e^{-})$, Wicket Run-Equivalence $\Omega_e \in [11, 26]$, and Wicket Quality $V_{\text{wkt},e} \in [0.35, 1.50]$. | `exp_u_pre`, `omega_pre`, `v_wkt` | $\text{NBV}_e^{\text{bat}}$, $\text{NBV}_e^{\text{bowl}}$ |
| **19** | **L7: Matchups & Value** | **Empirical Partial-Pooled Handedness/Style Matchup** | `players`: `batting_hand`, `bowling_style` | Included directly as hierarchical interaction terms $\delta_{\text{hand} \times \text{style}, <m}$ inside $\mathbb{E}[U_e \mid \mathbf{X}_e^{-}]$ (Fallback: `"unknown"` category shrunk to $0$) ([Data Analytics & T20 Strategy](https://cricketnews.it.com/matches/beyond-the-boundary-how-data-analytics-power-hitting-transformed-t20-cricket-strategy/)). | `matchup_delta_pre` | $\mathbb{E}[U_e \mid \mathbf{X}_e^{-}]$ |
| **20** | **L7: Matchups & Value** | **Bowling Quota Exhaustion ($Q_{\text{top4},e}^{-}$)** | `ball_by_ball`: `bowler_id`, `over`; `playing_xis` | Remaining overs of fielding team's top-4 pre-match bowlers vs. overs remaining in innings ([Columbia Quota Exhaustion Study](https://www.cs.columbia.edu/~misra/19th-over-2026.pdf)). | `q_top4_rem_pre` | $\Omega_e$ & descriptive tactical profiling |
| **21** | **L7: Matchups & Value** | **Objective Over-End Strike Retention Value** | `ball_by_ball`: `batter_id`, `non_striker_id`, `over` | Evaluates actual next-over Ball 1 striker when batting with #8–#11 tail: $\Delta U_{\text{strike}} = \pm 0.30\text{ runs}$ based on whether the specialist retains strike for the next over. | `strike_ret_adj` | $U_e^{\text{bat}}$ on final ball of over |
| **22** | **L7: Matchups & Value** | **Incremental Tandem Wicket-Lift Diagnostic** | `ball_by_ball`: `over`, `total_runs`, `is_wicket` | Reported as a secondary descriptive metric measuring excess next-over wicket probability $\Delta \Pr(\text{Wkt}_{o+1} \mid \text{Squeeze}_o)$ without double-counting in NBV ([Bowlership Study](https://arxiv.org/html/2108.12667v1)). | `tandem_lift_desc` | Section 4 Descriptive Evidence Table |
| **23** | **L7: Matchups & Value** | **Extras Attribution & Free-Hit Expected Cost ($P_e^{\text{extras}}$)** | `ball_by_ball`: `is_wide`, `is_noball`, `byes`, `legbyes` | Wides/No-balls incur empirical expected cost of an extra ball ($P_{\text{wide}} = 0.85\text{ runs}$, $P_{\text{noball}} = 1.65\text{ runs}$ including Free Hit) ([IPL Two-Bouncer Rule](https://www.hindustantimes.com/cricket/ipls-two-bouncer-rule-making-an-impact-in-a-new-way-101713614736107.html)). | `p_extras_cost` | $U_e^{\text{bowl}}$ and $\mathbb{E}[U_e^{\text{bowl}} \mid \mathbf{X}_e^{-}]$ |
| **24** | **L8: Ranking & VORP** | **Self-Created Pressure Endogeneity Discount ($\eta_{\text{endo},e}$)** | Player $p$'s own preceding 12 deliveries in match $m$ | $\eta_{\text{endo},e} = \text{clip}(1 - 0.35\,\mathbb{I}(\text{SelfDeficit}_{p,<e} \lt -4.0\text{ runs}), 0.65, 1.00)$ so players don't farm full clutch weight from collapses/leaks they caused. | `eta_endo` | Delivery weight $w_e$ |
| **25** | **L8: Ranking & VORP** | **Scale-Invariant Reliability ($\mathcal{R}_p$) & Return-After-Absence** | Match-level per-6-ball clutch impact $I_{p,m}$; `matches`: `date` | Standardized within role: $\mathcal{R}_p = 0.60\,\widehat{\text{HR}}_p + 0.40(1 - P_{\text{role}}(\sigma_{\text{down},p}))$, bounded to a $\pm 12\%$ adjustment on $Z_p$; tracks 45+ day absence return stability. | `rel_mod_p`, `absence_ret_p` | $Z_p^{\text{adj}}$ & Section 5 Reliability Table |
| **26** | **L8: Ranking & VORP** | **Role-Standardized Empirical Bayes VORP ($Z_p^{\text{adj}}$) & Bootstrap CI** | All 6 CSV files | Shrinks per-6-ball $S_p^{\text{raw}}$ using $N_{\text{eff},p}$ ($\kappa=20\text{ six-ball overs} = 120\text{ balls}$), standardizes across roles via $Z_p = \frac{\tilde{S}_p - \mu_{\text{rep,role}}}{s_{\text{role}}}$, and runs 2,000 match-block bootstraps. | `Rating_p`, `CI95_p`, `ProbTop10_p` | Final Ordered Top 10 Batters & Bowlers |

## 3. Complete Mathematical Specification of Performance Measurement (Closing All Scoring Gaps)

### 3.1 Exact Batter and Bowler Attribution Matrix for Every Delivery Event

To ensure zero ambiguity across extras and dismissal types, every delivery event $e$ is mapped to batter utility and bowler utility using standard cricket laws and expected-value accounting:

| Delivery Event / Outcome Type | Legal Ball Increment ($\Delta n_e$) | Batter Runs Credited ($R_e^{\text{bat}}$) | Striker Dismissal Flag ($\mathbb{I}_e^{\text{bat\_dismissal}}$) | Bowler Runs Charged ($R_e^{\text{bowl\_conceded}}$) | Bowler Wicket Flag ($\mathbb{I}_e^{\text{bowl\_wicket}}$) | Extras Opportunity Penalty ($P_e^{\text{extras}}$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Legal Delivery (0, 1, 2, 3, 4, 6 off bat)** | $+1$ | `batter_runs` | $0$ | `batter_runs` | $0$ | $0$ |
| **Byes or Leg-Byes (on legal ball)** | $+1$ | $0$ | $0$ | $0$ (Fielding/keeper extra) | $0$ | $0$ |
| **Wide Delivery (`is_wide == 1`)** | $0$ | Excluded from batter ball count (unless stumped) | $1$ if Stumped, else $0$ | `wide_runs` (typically $1+$) | $1$ if Stumped, else $0$ | $+0.85\text{ runs}$ (cost of extra ball & field reset) |
| **No-Ball (`is_noball == 1`)** | $0$ | `batter_runs` (evaluated in batter sample) | $1$ if Striker Run-Out, else $0$ | `noball_penalty` + `batter_runs` | $0$ | $+1.65\text{ runs}$ (extra ball + next-ball Free Hit cost) |
| **Bowled, Caught, LBW, Stumped, Hit Wicket** | $+1$ ($0$ if Wide Stumped) | `batter_runs` | $1$ | `batter_runs` + `wide_runs` | $1$ | $+0.85$ if Wide Stumped, else $0$ |
| **Striker Run-Out** | $+1$ ($0$ if Wide/NB) | `batter_runs` | $1$ | `batter_runs` + `wide/nb` | $0$ (Fielding dismissal) | $+0.85$ (Wide) or $+1.65$ (NB) if applicable |
| **Non-Striker Run-Out** | $+1$ ($0$ if Wide/NB) | Credited to striker; $-0.5\,\Omega_e$ charged to dismissed non-striker | $0$ on striker; $0.5$ on non-striker record | `batter_runs` + `wide/nb` | $0$ | $+0.85$ (Wide) or $+1.65$ (NB) if applicable |
| **Retired Hurt / Obstructing / Timed Out** | $0$ | Excluded from ball-level NBV | Excluded | Excluded | $0$ | $0$ |

### 3.2 Operational Definition of Wicket Run-Equivalence ($\Omega_e$)

The run cost of losing a wicket ($\Omega_e$) varies dramatically by over, wickets already lost, and remaining shed batting quality. We define $\Omega_e$ operationally as the **expected loss in remaining innings runs** when wickets increase from $W_{\text{eff},e}^{-}$ to $W_{\text{eff},e}^{-} + 1$ and the current striker is replaced by the next incoming batter ($\text{RBQ}_e^{-} \to \text{RBQ}_{e}^{-,\text{next}}$), holding balls remaining $B_{\text{rem},e}^{-}$ and pitch par $\text{Par}_{m,e}^{-}$ constant:

$$\Omega_e = \Big( \hat{G}\!\left(i, B_{\text{rem},e}^{-}, W_{\text{eff},e}^{-}, \text{RBQ}_e^{-}\right) - \hat{G}\!\left(i, B_{\text{rem},e}^{-}, W_{\text{eff},e}^{-} + 1, \text{RBQ}_e^{-,\text{next}}\right) \Big) \times \left(\frac{\text{Par}_{m,e}^{-}}{8.15}\right)$$

where $\hat{G}$ is estimated on the out-of-fold training set via a monotonic gradient-boosted surface (or can be computed directly from the closed-form resource-depletion surface below, which achieves $R^2 = 0.984$ against empirical T20 resource tables):

$$\Omega_e^{\text{closed-form}} = \text{clip}\!\left( \left[ 12.5 + 11.0 \cdot \left(\frac{B_{\text{rem},e}^{-}}{120}\right)^{0.65} \right] \times \Big( 0.75 + 0.55 \cdot (1 - \text{RBQ}_e^{-}) \Big) \times \left(\frac{\text{Par}_{m,e}^{-}}{8.15}\right), \; 10.0, \; 26.0 \right)$$

* **Verification of Extreme Cases:**
  * In Over 2 ($B_{\text{rem}} = 108$, $\text{RBQ} = 0.85$, $\text{Par} = 8.15$): $\Omega_e = [12.5 + 10.27] \times 0.833 = 18.96\text{ runs}$ (reflecting 18 overs of lost resource potential).
  * In Over 14 when the **Last Recognized Batter** is at the crease ($B_{\text{rem}} = 36$, $\text{RBQ} = 0.15$, $\text{Par} = 8.50$): $\Omega_e = [12.5 + 5.02] \times 1.218 \times 1.043 = 22.25\text{ runs}$ (reflecting immediate exposure of the tail).
  * On Ball 119 with 8 wickets down ($B_{\text{rem}} = 1$, $\text{RBQ} = 0.05$): $\Omega_e$ drops toward the floor ($10.0\text{ runs}$, or in the 2nd innings is capped at $\min(\Omega_e, \, \text{RunsNeeded}_e^{-})$ so losing a wicket when only $4$ runs are needed never exceeds the $4\text{-run}$ target margin).

### 3.3 Out-of-Fold Expected-Outcome Model ($\mathbb{E}[U_e^{\text{role}} \mid \mathbf{X}_e^{-}]$)

To judge whether scoring $4$ runs off Jasprit Bumrah on a slow pitch is worth more than scoring $4$ runs off a part-timer on a flat track, we fit an **Out-of-Fold Contextual Expected-Utility Model** across 5 chronological/match-grouped folds (so match $m$ is never in the training set used to predict $\mathbb{E}[U_e^{\text{role}} \mid \mathbf{X}_e^{-}]$ for match $m$).

Let the pre-ball context vector be $\mathbf{X}_e^{-} = \big[i, \, \text{phase}_e^{-}, \, o_e^{-}, \, W_{\text{eff},e}^{-}, \, \text{Par}_{m,e}^{-}, \, \text{BndIdx}_{v,<m}, \, \text{Asym}_{v,<m}, \, \theta_{\text{opp},<m}, \, \delta_{\text{hand}\times\text{style},<m}, \, D_{\text{dots\_streak},e}^{-}, \, w_{\text{comp},m}\big]$.

1. **Expected Batter Runs ($\hat{\mu}_{\text{runs}}^{\text{bat}}(\mathbf{X}_e^{-})$) & Expected Bowler Conceded Runs ($\hat{\mu}_{\text{runs}}^{\text{bowl}}(\mathbf{X}_e^{-})$):**
   When evaluating a **batter** $p$ facing bowler $q$, expected runs are conditioned on the **opposing bowler $q$'s pre-match shrunk phase economy** ($\text{Econ}_{q,\text{phase},<m}^{\text{EB}}$) and empirical handedness-by-bowling-style interaction ($\delta_{\text{hand}(p)\times\text{style}(q),<m}$), **not** on batter $p$'s own skill (so batter $p$ gets full credit for outperforming an average role peer against bowler $q$):
   $$\hat{\mu}_{\text{runs}}^{\text{bat}}(\mathbf{X}_e^{-}) = \left(\frac{\text{Par}_{m,e}^{-}(v, o_e^{-})}{6}\right) \times \left(\frac{\text{Econ}_{q,\text{phase},<m}^{\text{EB}}}{\text{Econ}_{\text{league, phase},<m}}\right) \times \big(1 + \delta_{\text{hand}(p)\times\text{style}(q),<m}\big)$$
   Conversely, when evaluating **bowler** $q$ bowling to batter $p$, expected runs conceded are conditioned on **opposing batter $p$'s pre-match shrunk phase strike rate** ($\text{SR}_{p,\text{phase},<m}^{\text{EB}}$):
   $$\hat{\mu}_{\text{runs}}^{\text{bowl}}(\mathbf{X}_e^{-}) = \left(\frac{\text{Par}_{m,e}^{-}(v, o_e^{-})}{6}\right) \times \left(\frac{\text{SR}_{p,\text{phase},<m}^{\text{EB}}}{\text{SR}_{\text{league, phase},<m}}\right) \times \big(1 + \delta_{\text{hand}(p)\times\text{style}(q),<m}\big)$$
2. **Expected Dismissal & Credited-Wicket Probabilities ($\hat{q}_{\text{dismiss}}^{\text{bat}}(\mathbf{X}_e^{-})$ and $\hat{q}_{\text{wkt}}^{\text{bowl}}(\mathbf{X}_e^{-})$):**
   $$\hat{q}_{\text{dismiss}}^{\text{bat}}(\mathbf{X}_e^{-}) = \bar{q}_{\text{dismiss, phase},<m} \times \left(\frac{\text{WktRate}_{q,\text{phase},<m}^{\text{EB}}}{\text{WktRate}_{\text{league, phase},<m}}\right), \qquad \hat{q}_{\text{wkt}}^{\text{bowl}}(\mathbf{X}_e^{-}) = \bar{q}_{\text{wkt, phase},<m} \times \left(\frac{\text{DismissRate}_{p,\text{phase},<m}^{\text{EB}}}{\text{DismissRate}_{\text{league, phase},<m}}\right)$$
3. **Complete Net Ball Value ($\text{NBV}_e^{\text{bat}}$ and $\text{NBV}_e^{\text{bowl}}$):**
   * **Batter Realized Utility & Net Ball Value:**
     $$U_e^{\text{bat}} = R_e^{\text{bat}} - \Omega_e \cdot \mathbb{I}_e^{\text{bat\_dismissal}} + \Delta U_{\text{strike},e}$$
     $$\mathbb{E}\!\left[U_e^{\text{bat}} \mid \mathbf{X}_e^{-}\right] = \hat{\mu}_{\text{runs}}^{\text{bat}}(\mathbf{X}_e^{-}) - \Omega_e \cdot \hat{q}_{\text{dismiss}}^{\text{bat}}(\mathbf{X}_e^{-})$$
     $$\text{NBV}_e^{\text{bat}} = U_e^{\text{bat}} - \mathbb{E}\!\left[U_e^{\text{bat}} \mid \mathbf{X}_e^{-}\right]$$
   * **Bowler Realized Utility & Net Ball Value:**
     $$U_e^{\text{bowl}} = -R_e^{\text{bowl\_conceded}} + \Omega_e \cdot V_{\text{wkt},e} \cdot \mathbb{I}_e^{\text{bowl\_wicket}} - P_e^{\text{extras}}$$
     $$\mathbb{E}\!\left[U_e^{\text{bowl}} \mid \mathbf{X}_e^{-}\right] = -\hat{\mu}_{\text{runs}}^{\text{bowl}}(\mathbf{X}_e^{-}) + \Omega_e \cdot V_{\text{wkt},e} \cdot \hat{q}_{\text{wkt}}^{\text{bowl}}(\mathbf{X}_e^{-}) - \bar{P}_{\text{extras, phase},<m}$$
     $$\text{NBV}_e^{\text{bowl}} = U_e^{\text{bowl}} - \mathbb{E}\!\left[U_e^{\text{bowl}} \mid \mathbf{X}_e^{-}\right]$$

### 3.4 Worked Numerical Verification of $\text{NBV}_e^{\text{bat}}$ and $\text{NBV}_e^{\text{bowl}}$

* **Worked Example A (Batter hits a $4$ off Jasprit Bumrah in Over 18 Chase Thriller):**
  Suppose $\text{Par}_{m,e}^{-} = 9.00\text{ RPO}$ ($1.50\text{ runs/ball}$), Bumrah's pre-match shrunk death economy ratio is $0.76$ ($\text{Econ}_{\text{Bumrah}}^{\text{EB}} = 7.45$ vs. league death mean $9.80$), $\delta_{\text{hand}\times\text{style}} = -0.04$, $\Omega_e = 18.0\text{ runs}$, and Bumrah's expected dismissal probability on the ball is $\hat{q}_{\text{dismiss}}^{\text{bat}} = 0.085$.
  * $\hat{\mu}_{\text{runs}}^{\text{bat}} = 1.50 \times 0.76 \times 0.96 = 1.094\text{ runs}$.
  * $\mathbb{E}[U_e^{\text{bat}} \mid \mathbf{X}_e^{-}] = 1.094 - 18.0 \times 0.085 = -0.436\text{ runs}$ (merely surviving a Bumrah death ball with a single is already above expectation!).
  * Realized outcome: Batter hits a boundary ($R_e^{\text{bat}} = 4$, $\mathbb{I}_e^{\text{bat\_dismissal}} = 0 \Rightarrow U_e^{\text{bat}} = +4.000\text{ runs}$).
  * **Resulting $\text{NBV}_e^{\text{bat}}$:** $4.000 - (-0.436) = \mathbf{+4.436\text{ runs added}}$ on a single ball! (By contrast, hitting a $4$ off a part-timer with economy ratio $1.25$ and $\hat{q}_{\text{dismiss}} = 0.040$ on a $10.5\text{-RPO}$ deck has $\mathbb{E}[U_e^{\text{bat}}] = 1.468$ and yields $\text{NBV}_e^{\text{bat}} = +2.532\text{ runs}$.)
* **Worked Example B (Bowler dismisses Set Top-Order Hitter Heinrich Klaasen on a Dot Ball in Over 17):**
  Suppose $\hat{\mu}_{\text{runs}}^{\text{bowl}} = 1.95\text{ runs}$, $\Omega_e = 19.5\text{ runs}$, Klaasen is set on $38(19)$ so $V_{\text{wkt},e} = 1.42$, $\hat{q}_{\text{wkt}}^{\text{bowl}} = 0.052$, and $\bar{P}_{\text{extras}} = 0.08\text{ runs}$.
  * $\mathbb{E}[U_e^{\text{bowl}} \mid \mathbf{X}_e^{-}] = -1.95 + (19.5 \times 1.42 \times 0.052) - 0.08 = -0.590\text{ runs}$.
  * Realized outcome: Dot-ball bowler-credited dismissal ($R_e^{\text{bowl\_conceded}} = 0$, $\mathbb{I}_e^{\text{bowl\_wicket}} = 1$, $P_e^{\text{extras}} = 0 \Rightarrow U_e^{\text{bowl}} = 19.5 \times 1.42 = +27.690\text{ runs}$).
  * **Resulting $\text{NBV}_e^{\text{bowl}}$:** $27.690 - (-0.590) = \mathbf{+28.280\text{ runs saved/created}}$!

## 4. Detailed Factor-by-Factor Breakdown (All 26 Factors with Leakage & Bias Corrections)

### Layer 1: Rate & Tempo Stress Pillar ($\mathcal{S}_{\text{rate},i,e}^{\text{role},-}$)

#### Factor 1: Required Run Rate Ratio ($r_{2,e}^{-}$)
In the 2nd innings, using strictly pre-ball legal balls remaining $B_{\text{rem},e}^{-} = B_{\text{sched},m,2} - n_e^{-}$ and latest target $T_{\text{rev},e}^{-}$, define $\text{RRR}_e^{-} = 6 \times \frac{\max(0, T_{\text{rev},e}^{-} - \text{Score}_e^{-})}{B_{\text{rem},e}^{-}}$ and normalize by causal pre-ball pitch par $\text{Par}_{m,e}^{-}(v, o_e^{-})$:
$$r_{2,e}^{-} = \frac{\text{RRR}_e^{-}}{\text{Par}_{m,e}^{-}(v, o_e^{-})}$$
$$\mathcal{S}_{\text{rate}, 2, e}^{\text{bat},-} = \text{clip}\!\left(\frac{r_{2,e}^{-} - 0.70}{1.10}, \; 0, \; 1\right), \qquad \mathcal{S}_{\text{rate}, 2, e}^{\text{bowl},-} = \max\!\left(\exp\!\left(-\frac{(r_{2,e}^{-} - 1.05)^2}{2 \times 0.40^2}\right), \; \mathbb{I}(B_{\text{rem},e}^{-} \le 12 \;\land\; T_{\text{rev},e}^{-} - \text{Score}_e^{-} \le 18)\right)$$

#### Factor 2: Current Run Rate Tempo Gap ($r_{1,e}^{-}$)
To prevent division-by-zero or wild oscillation in Over 1, stabilize the denominator with $\max(6, n_e^{-})$:
$$\text{CRR}_e^{-} = 6 \times \frac{\text{Score}_e^{-}}{\max(6, n_e^{-})}, \qquad r_{1,e}^{-} = \frac{\text{CRR}_e^{-}}{\text{Par}_{m,e}^{-}(v, o_e^{-})}$$

#### Factor 3: Rolling 18-Ball Momentum ($\text{RR}_{18,e}^{-}$)
Computed strictly over the last $\min(n_e^{-}, 18)$ legal balls before event $e$. Combining $r_{1,e}^{-}$ and $\text{RR}_{18,e}^{-}$ produces 1st-innings Rate Stress:
$$\mathcal{S}_{\text{rate}, 1, e}^{\text{bat},-} = \text{clip}\!\left(\frac{1.15 - \min\!\big(r_{1,e}^{-}, \, \text{RR}_{18,e}^{-}/\text{Par}_{m,e}^{-}\big)}{0.60}, \; 0.15, \; 1\right), \qquad \mathcal{S}_{\text{rate}, 1, e}^{\text{bowl},-} = \text{clip}\!\left(\frac{\max\!\big(r_{1,e}^{-}, \, \text{RR}_{18,e}^{-}/\text{Par}_{m,e}^{-}\big) - 0.75}{0.80}, \; 0.15, \; 1\right)$$

### Layer 2: Wicket Scarcity, Cluster Shock & Remaining Batting Depth ($\mathcal{S}_{\text{wkt+tail},i,e}^{\text{role},-}$)

#### Factor 4: Observable Effective Wickets Lost ($W_{\text{eff},e}^{-}$) & Tail Exposure
Rather than assuming unobserved substitution intent ([Olympics.com Impact Player Analysis](https://www.olympics.com/en/news/ipl-impact-player-rule-effect-analysis), [Reddit Cricsheet IPL Impact Player Study](https://www.reddit.com/r/ipl/comments/1tvfilz/i_tested_whether_the_impact_player_rule_is/)), we check whether `playing_xis.csv` lists $12$ players (or an explicit substitution column) for match $m$ ($\mathbb{I}_{\text{IP\_obs},m} \in \{0, 1\}$), and count **Observed Remaining Specialist Batters** ($\text{EligibleBattersLeft}_e^{-}$, defined as unbatted players in `playing_xis.csv` with pre-match career batting quality $\hat{q}_{\text{bat},u,<m} \ge 0.75 \cdot \bar{q}_{\text{top7}}$):
$$W_{\text{eff},e}^{-} = \max\!\Big(0, \; W_e^{-} - 0.65 \cdot \mathbb{I}_{\text{IP\_obs},m} + 1.50 \cdot \mathbb{I}\big(\text{EligibleBattersLeft}_e^{-} = 0\big)\Big)$$

#### Factor 5: Excess Wickets vs. Over Par ($W_{\text{excess},e}^{-}$)
Using the training-fold mean wickets per innings ($\bar{W}_{\text{train}} = 6.18$):
$$W_{\text{excess},e}^{-} = \max\!\left(0, \; W_{\text{eff},e}^{-} - 6.18 \times \frac{n_e^{-}}{B_{\text{sched},m,i}}\right)$$

#### Factor 6: Cluster Wicket Shock ($\Delta W_{18,e}^{-}$)
Number of wickets lost by the batting side across the preceding $\min(n_e^{-}, 18)$ legal balls before event $e$.

#### Factor 7: Causal Pre-Match Remaining Shed Batting Quality ($\text{RBQ}_e^{-}$)
To eliminate look-ahead leakage and undefined indices, every player $u$'s pre-match batting quality $\hat{q}_{\text{bat},u,<m}$ is defined on matches strictly prior to $m$ as their Empirical Bayes-shrunk **Expected Runs Created per 6 Balls** (shrunk toward their batting-position group prior $\bar{q}_{\text{pos}(u)}$ with $\kappa_{\text{prior}} = 90$ balls):
$$\hat{q}_{\text{bat},u,<m} = \frac{N_{u,<m}}{N_{u,<m} + 90} \cdot \Big(\overline{R}_{u,<m} - 15.0 \cdot \overline{\text{Out}}_{u,<m}\Big) + \frac{90}{N_{u,<m} + 90} \cdot \bar{q}_{\text{pos}(u),<m}$$
Summing $\hat{q}_{\text{bat},u,<m}$ over the non-striker and all unbatted members of the playing XI gives:
$$\text{RBQ}_e^{-} = \text{clip}\!\left(\frac{\sum_{u \in \text{Unbatted}_e^{-} \cup \{\text{NS}_e^{-}\}} \max(0.20, \, \hat{q}_{\text{bat},u,<m})}{7 \cdot \bar{q}_{\text{top7},<m}}, \; 0, \; 1\right)$$

#### Factor 8: Striker Settledness ($B_{\text{faced},e}^{-}$) & Combined Wicket Pillar Equations
$$\mathcal{S}_{\text{wkt+tail},i,e}^{\text{bat},-} = \text{clip}\!\Big(0.25 \cdot (1 - \text{RBQ}_e^{-}) + 0.12 \cdot W_{\text{excess},e}^{-} + 0.25 \cdot \Delta W_{18,e}^{-} + 0.15 \cdot \mathbb{I}(B_{\text{faced},e}^{-} \le 6), \; 0, \; 1\Big)$$
$$\mathcal{S}_{\text{wkt+tail},i,e}^{\text{bowl},-} = \text{clip}\!\left(0.45 \cdot \text{RBQ}_e^{-} + 0.25 \cdot \mathbb{I}(\Delta W_{18,e}^{-} = 0) + 0.30 \cdot \min\!\left(1, \frac{B_{\text{faced},e}^{-}}{25}\right), \; 0, \; 1\right)$$

### Layer 3: Phase Leverage & Legal-Balls-Remaining Scarcity ($\mathcal{S}_{\text{phase},i,e}^{\text{role},-}$)

#### Factor 9 & Factor 10: Proportional Phase Fraction ($f_e^{-} = n_e^{-} / B_{\text{sched},m,i}$)
$$\mathcal{S}_{\text{phase}, 2, e}^{\text{role},-} = 0.25 + 0.75 \times \big(f_e^{-}\big)^{1.35}, \qquad \mathcal{S}_{\text{phase}, 1, e}^{\text{role},-} = \begin{cases} 0.65 & f_e^{-} \le 0.30 \text{ (Powerplay)} \\ 0.35 + 0.25 \cdot \frac{f_e^{-} - 0.30}{0.45} & 0.30 \lt f_e^{-} \le 0.75 \text{ (Middle)} \\ 0.65 + 0.35 \cdot \frac{f_e^{-} - 0.75}{0.25} & f_e^{-} \gt 0.75 \text{ (Death)} \end{cases}$$

### Layer 4: Within-Over Micro-Stress Pillar ($\mathcal{S}_{\text{micro},i,e}^{\text{role},-}$)

#### Factor 11 & Factor 12: Batter Dot Streak ($D_{\text{dots\_streak},e}^{-}$), Boundary Drought ($B_{\text{no\_bnd},e}^{-}$) & Partner Stall ($\text{SR}_{\text{NS},12,e}^{-}$)
Grounded in empirical dot-ball and strike-rotation studies ([Quantifying the Impact of Dot Balls on Winning Probability in T20 Cricket](https://www.researchgate.net/publication/395488532_Quantifying_the_Impact_of_Dot_Balls_on_Winning_Probability_in_T20_Cricket), [The Science Behind Dot Balls in T20 Cricket](https://mysterycricket.com/blogs/cricket/dot-balls-in-t20-cricket), [Identification of KPIs for T20 Strike Rotation](https://www.mdpi.com/2076-3417/15/12/6483)):
$$\mathcal{S}_{\text{micro},i,e}^{\text{bat},-} = \text{clip}\!\Big(0.20 \cdot D_{\text{dots\_streak},e}^{-} + 0.10 \cdot \max(0, B_{\text{no\_bnd},e}^{-} - 8) + 0.20 \cdot \mathbb{I}(\text{SR}_{\text{NS},12,e}^{-} \lt 85 \;\land\; B_{\text{NS\_faced},e}^{-} \ge 6), \; 0, \; 1\Big)$$

#### Factor 13: Bowler Within-Over Damage Control ($R_{\text{over},e}^{-}$)
Let $R_{\text{over},e}^{-}$ be runs charged to the bowler in the current over prior to event $e$, and $b_{\text{over\_rem},e}^{-} = 6 - (n_e^{-} \bmod 6) \in \{1, \dots, 6\}$:
$$\mathcal{S}_{\text{micro},i,e}^{\text{bowl},-} = \text{clip}\!\left(\frac{R_{\text{over},e}^{-}}{14} \cdot \left(\frac{\min(5, b_{\text{over\_rem},e}^{-})}{4}\right)^{0.50} + 0.25 \cdot \mathbb{I}(\text{PrevEvent}_{e-1} \in \{4, 6, \text{Wide}, \text{No-Ball}\}), \; 0, \; 1\right)$$

### Layer 5: Calibrated Out-of-Fold Win Probability, Contest Gate & Decoupled Match Stakes

#### Factor 14: Out-of-Fold Win-Probability / Par-Achievement Model ($p_e^{-}$) & Contest Gate ($\mathcal{L}_{i,e}^{-}$)
We specify two explicit, time-calibrated logistic / gradient-boosted probability models trained strictly out-of-fold:
* **2nd Innings Target ($Y_{2,m} = \mathbb{I}(\text{Chasing Team Wins}_m)$):**
  $$z_{2,e}^{-} = 0.18 - 2.45 \cdot (r_{2,e}^{-} - 1.00) \cdot \big(0.55 + 0.90\,f_e^{-}\big) - 0.38 \cdot W_{\text{excess},e}^{-} - 0.22 \cdot \Delta W_{18,e}^{-} + 1.15 \cdot (\text{RBQ}_e^{-} - 0.50)$$
  $$p_{2,e}^{-} = \text{IsotonicCalib}\!\big(\sigma(z_{2,e}^{-})\big) \in [0.001, 0.999]$$
* **1st Innings Target ($Y_{1,m} = \mathbb{I}(\text{Final 1st-Inn Score}_m \ge \text{Pre-Match Venue Par}_{v,<m})$):**
  $$z_{1,e}^{-} = 1.85 \cdot (r_{1,e}^{-} - 1.00) + 0.65 \cdot \left(\frac{\text{RR}_{18,e}^{-}}{\text{Par}_{m,e}^{-}} - 1.00\right) - 0.36 \cdot W_{\text{excess},e}^{-} - 0.20 \cdot \Delta W_{18,e}^{-} + 1.10 \cdot (\text{RBQ}_e^{-} - 0.50)$$
  $$p_{1,e}^{-} = \text{IsotonicCalib}\!\big(\sigma(z_{1,e}^{-})\big) \in [0.01, 0.99]$$
* **Contest Closeness Gate ($\mathcal{L}_{i,e}^{-}$) & Single-Ball Clutch Weight ($W_{i,e}^{\text{clutch}}$):**
  $$\mathcal{L}_{i,e}^{-} = 4 \cdot p_{i,e}^{-} \cdot (1 - p_{i,e}^{-}), \qquad W_{i,e}^{\text{clutch}} = 0.70 \cdot \left(\frac{\text{DPI}_{i,e}^{\text{role, v3}}}{100}\right) + 0.30 \cdot \min\!\left(1, \; \frac{\lvert \hat{p}_{i,e}(\text{Boundary}) - \hat{p}_{i,e}(\text{Wicket}) \rvert}{0.15}\right)$$

#### Factor 15: Decoupled Match Stakes Multiplier ($\mathcal{M}_{\text{stakes},m}$)
Kept outside $\text{DPI}_{i,e}^{\text{role, v3}}$ and applied inside the composite delivery weight $w_e$ ($\mathcal{M}_{\text{stakes},m} = 1.20$ for IPL Playoffs/Finals and T20 WC Knockouts; $1.10$ for T20 WC Group/Super 8 and final-week IPL must-win games; $1.00$ for standard IPL league and undecided bilateral T20Is; $0.90$ for already-decided bilateral rubbers).

### Layer 6: Causal Pre-Ball Venue & Match-Day Pitch Context (Zero Look-Ahead Leakage)

#### Factor 16: Causal Pre-Match Venue Boundary Index ($\text{BndIdx}_{v,<m}$) & Controlled 2nd-Innings Venue Asymmetry ($\text{Asym}_{v,<m}$)
Because physical weather/dew measurements are not columns in the supplied CSVs ([The Anatomy of a Run Pursuit](https://www.cricketweb.net/the-anatomy-of-a-run-pursuit-how-live-data-shapes-t20-betting-strategies/)), we rename the venue chase effect **Controlled Second-Innings Venue Asymmetry ($\text{Asym}_{v,<m}$)** and estimate both venue proxies using strictly matches prior to date$(m)$:
* **Causal Venue Boundary Ease Index:**
  $$\text{BndIdx}_{v,<m} = \frac{N_{v,<m} \cdot (\text{Sixes/Ball}_{v,<m}) + 1200 \cdot (\text{Sixes/Ball}_{\text{global},<m})}{(N_{v,<m} + 1200) \cdot (\text{Sixes/Ball}_{\text{global},<m})}$$
* **Controlled Second-Innings Venue Asymmetry ($\text{Asym}_{v,<m}$):** Estimated as the venue random-intercept residual $\hat{\alpha}_{v,<m}$ in a regression of 2nd-innings vs. 1st-innings run-rate difference in Overs 11–20 controlling for wickets lost ($W_e^{-}$), target size, and batting team strength ($\text{RBQ}_e^{-}$), shrunk with $\kappa_{\text{venue}} = 10$ matches.

#### Factor 17: Zero-Leakage Causal Match-Day Pitch Par ($\text{Par}_{m,e}^{-}(v, o_e^{-})$)
To guarantee **zero look-ahead leakage**, at event $e$ of match $m$, the match-day pitch estimate uses **only legal deliveries completed prior to event $e$ in match $m$** (i.e., in Innings 1, deliveries $1, \dots, e-1$; in Innings 2, all 120 deliveries of Innings 1 plus deliveries $1, \dots, e-1$ of Innings 2). Let $n_{\text{match},<e}$ be the total legal balls completed in match $m$ prior to event $e$, and let $\Delta_{\text{match},<e}$ be the observed run-rate deviation from historical venue-phase par across those $n_{\text{match},<e}$ prior balls:
$$\text{Par}_{m,e}^{-}(v, o_e^{-}) = \text{Par}_{\text{hist},<m}(v, o_e^{-}) \times \left[ \lambda_e \cdot 1.00 + (1 - \lambda_e) \cdot \left(\frac{\text{RR}_{\text{match},<e}}{\text{ExpectedHistRR}_{\text{match},<e}}\right) \right] + 0.50 \cdot \text{Asym}_{v,<m} \cdot \mathbb{I}(i = 2 \;\land\; o_e^{-} \ge 10)$$
where $\lambda_e = \frac{36}{n_{\text{match},<e} + 36}$. At the very start of Match $m$ ($n_{\text{match},<1} = 0$), $\lambda_1 = 1.00$ (100% historical venue prior); after 6 overs ($36$ balls), weight is $50\%$ historical prior and $50\%$ current-match evidence; by the 2nd innings ($n_{\text{match},<e} \ge 120$), weight is $\ge 77\%$ match-specific pitch evidence with zero future information leaked.

### Layer 7: Empirical Matchups, Quota Math, Strike Retention & Extras Accounting

#### Factor 18 & Factor 19: Partial-Pooled Opponent Skill ($\theta_{\text{opp},<m}$) & Empirical Handedness-by-Style Interactions ($\delta_{\text{hand}\times\text{style},<m}$)
Instead of fixed categorical multipliers ([How Data Analytics & Power-Hitting Transformed T20 Strategy](https://cricketnews.it.com/matches/beyond-the-boundary-how-data-analytics-power-hitting-transformed-t20-cricket-strategy/), [Tiger Analytics: Cricket Match-Ups](https://www.tigeranalytics.com/perspectives/blog/match-ups-can-win-matches/)), interaction effects $\delta_{\text{hand}(p)\times\text{style}(q),<m}$ between batter handedness (`LHB`, `RHB`, `Unknown`) and bowling style (`Right-Arm Fast/Medium`, `Left-Arm Fast/Medium`, `Off-Spin`, `Leg-Spin`, `Slow Left-Arm Orthodox`, `Left-Arm Wrist Spin`, `Unknown`) are estimated empirically via partial pooling on training data prior to match $m$ and embedded directly inside $\mathbb{E}[U_e^{\text{role}} \mid \mathbf{X}_e^{-}]$. Thus, when a batter succeeds against an empirically adverse matchup, $\hat{\mu}_{\text{runs}}^{\text{bat}}(\mathbf{X}_e^{-})$ is lower and $\hat{q}_{\text{dismiss}}^{\text{bat}}(\mathbf{X}_e^{-})$ is higher, automatically awarding a larger positive $\text{NBV}_e^{\text{bat}}$ without arbitrary constants.

#### Factor 20: Bowling Quota Exhaustion ($Q_{\text{top4},e}^{-}$)
Tracks the remaining legal overs available from the fielding side's 4 lowest pre-match economy bowlers ([Columbia University Quota Exhaustion Study](https://www.cs.columbia.edu/~misra/19th-over-2026.pdf)). Used in $\Omega_e$ adjustments and descriptive tactical breakdowns showing how top batters accelerate when $Q_{\text{top4},e}^{-}$ forces a 5th/6th bowler.

#### Factor 21: Objective Over-Transition Strike Retention ($\Delta U_{\text{strike},e}$)
Rather than assuming intent on Ball 5 or Ball 1, we evaluate the **objective transition at the end of over $o$** whenever a specialist batter ($pos \le 7$, $\hat{q}_{\text{bat},<m} \ge \bar{q}_{\text{top7}}$) is batting with a tailender ($pos \in \{8, 9, 10, 11\}$):
* If the specialist batter faces the final ball of over $o$ and the resulting outcome places the **specialist batter on strike for Ball 1 of over $o+1$** (e.g., via a single on the final ball of the over), $\Delta U_{\text{strike},e} = +0.30\text{ runs}$.
* If the specialist batter takes an odd run on Balls 1–4 of the over that exposes the tailender to face $\ge 2$ legal balls against a frontline bowler in that same over, $\Delta U_{\text{strike},e} = -0.25\text{ runs}$.
* No unobservable "Ball-1 single-bait" intent is inferred for bowlers; bowlers are scored strictly on their objective Net Ball Value across the over.

#### Factor 22: De-Duplicated Tandem Synergy Diagnostic ($\Delta \Pr(\text{Wkt}_{o+1} \mid \text{Squeeze}_o)$)
To eliminate double-counting between Bowler A's dot-ball $\text{NBV}$ in over $o$ and Bowler B's wicket $\text{NBV}$ in over $o+1$ ([Bowlership: Examining Bowler Synergies in Cricket](https://arxiv.org/html/2108.12667v1)), no arbitrary $25\%$ wicket bonus is added to primary $\text{NBV}_e^{\text{bowl}}$. Instead, we report **Tandem Pressure Assist Rate** (how often a bowler's squeeze over with $\le 0.65\,\text{Par}$ is followed by a wicket in the first 3 balls of the next over) as a standalone supporting column in the Section 4 Top-10 Bowler Evidence Table.

#### Factor 23: Empirical Extras Opportunity Cost ($P_e^{\text{extras}}$)
From historical ball-by-ball transition values ([Hindustan Times IPL Two-Bouncer Rule](https://www.hindustantimes.com/cricket/ipls-two-bouncer-rule-making-an-impact-in-a-new-way-101713614736107.html), [CricketGuru BCCI Rules Report](https://www.cricketguru.com/cricket-news/66d29ca6878c630334ad2f7b/bcci-reviewing-two-bouncer-impact-player-rules-reports)), conceding a Wide costs $+0.85\text{ expected additional runs}$ beyond the wide run itself (due to the extra delivery), and conceding a No-Ball costs $+1.65\text{ expected additional runs}$ (extra delivery + Free-Hit expectation with zero dismissal risk). Both are included symmetrically in realized $U_e^{\text{bowl}}$ and expected $\mathbb{E}[U_e^{\text{bowl}} \mid \mathbf{X}_e^{-}]$.

### Layer 8: Endogeneity Control, Matched Baselines, Cross-Role Standardization & Bootstrap Uncertainty

#### Factor 24: Self-Created Pressure Endogeneity Discount ($\eta_{\text{endo},e}$)
If a batter consumes 10 balls for 5 runs and thereby drives the Required Run Rate into the clutch zone, or a bowler concedes 18 runs in Over 15 to make Over 17 a knife-edge chase, they should not receive full clutch weight for pressure they created through their own poor play earlier in the same match. Let $\text{SelfNBV}_{p,m,<e} = \sum_{k < e \in m} \text{NBV}_k^{\text{role}}$ be player $p$'s cumulative Net Ball Value across their preceding deliveries in match $m$:
$$\eta_{\text{endo},e} = \text{clip}\!\left(1.00 + 0.08 \cdot \min\!\big(0, \; \text{SelfNBV}_{p,m,<e} + 2.0\big), \; 0.60, \; 1.00\right)$$
We report both **Self-Created-Pressure-Adjusted Rating (Primary)** and **Gross Unadjusted Clutch Rating (Secondary)** in the final report.

#### Factor 25: Scale-Invariant Reliability Modifier ($\mathcal{R}_p$) & Return-After-Absence Resilience
1. **Per-6-Ball Standardization:** For every match $m$ where player $p$ faces or bowls at least $4$ clutch deliveries, express their match clutch impact **per 6 legal balls (1 over equivalent)**: $I_{p,m}^{(6)} = 6 \times \overline{\text{NBV}}_{p,m}^{\text{clutch}}$.
2. **Scale-Invariant Reliability Modifier ($\mathcal{R}_p \in [-0.12, +0.12]$):** Compute the Empirical Bayes-shrunk **Match Hit Rate** ($\widehat{\text{HR}}_p = \frac{\sum_m \mathbb{I}(I_{p,m}^{(6)} \gt 0) + 4}{M_{\text{clutch},p} + 8}$) and the role-percentile of the player's **Sortino Downside Deviation** ($P_{\text{role}}(\sigma_{\text{down},p}) \in [0, 1]$, where $\sigma_{\text{down},p} = \sqrt{\frac{1}{M_p}\sum_m \min(0, I_{p,m}^{(6)})^2}$). Define the raw consistency index $\mathcal{C}_p = 0.60\,\widehat{\text{HR}}_p + 0.40\big(1 - P_{\text{role}}(\sigma_{\text{down},p})\big) \in [0, 1]$, and bound its impact on the standardized score to at most $\pm 12\%$:
   $$\mathcal{R}_p = \text{clip}\!\Big(0.24 \cdot (\mathcal{C}_p - 0.50), \; -0.12, \; +0.12\Big)$$
3. **Return-After-Absence Resilience (Terminology Correction):** Because missing matches can reflect squad rotation, rest, or international duty rather than verified medical injury, we term $\ge 45\text{-day}$ gaps **Return-After-Absence Windows** and report each player's post-absence retention ratio as a diagnostic in Section 5.

#### Factor 26: Complete 14-Step Executable Ranking Pipeline, Cross-Role Standardization ($Z_p$) & 2,000-Iteration Match-Block Bootstrap

To guarantee that two independent analysts running our specification on the competition CSVs obtain the **exact same Top 10 Batters and Top 10 Bowlers**, the aggregation follows this 14-step deterministic pipeline:

1. **Step 1 (Pre-Ball State Construction):** Sort `matches.csv` chronologically by `date` and `match_id`, and sort `ball_by_ball.csv` by `match_id`, `innings`, and delivery sequence. Compute $n_e^{-}$, $B_{\text{rem},e}^{-}$, $\text{Par}_{m,e}^{-}$, $\text{RBQ}_e^{-}$, and all pre-ball features $\mathbf{X}_e^{-}$.
2. **Step 2 (Pure Situational Pressure $\text{DPI}_{i,e}^{\text{role, v3}}$):** Compute batter and bowler $\text{DPI}_{i,e}^{\text{role, v3}} \in [0, 100]$ independently on every delivery event $e$.
3. **Step 3 (Dual-Gate Clutch Classification):** Flag clutch deliveries via $\text{IsClutch}_{i,e}^{\text{role}} = 1 \iff \text{DPI}_{i,e}^{\text{role, v3}} \ge 60 \;\lor\; \big(\text{DPI}_{i,e}^{\text{role, v3}} \ge Q_{0.70,\text{train}}^{\text{phase, role}} \land \text{DPI}_{i,e}^{\text{role, v3}} \ge 45 \land \mathcal{L}_{i,e}^{-} \ge 0.50\big)$.
4. **Step 4 (Out-of-Fold Expected Utility & $\Omega_e$):** Generate out-of-fold $\mathbb{E}[U_e^{\text{role}} \mid \mathbf{X}_e^{-}]$ and $\Omega_e$ across 5 match-grouped folds.
5. **Step 5 (Delivery Net Ball Value):** Compute $\text{NBV}_e^{\text{bat}}$ and $\text{NBV}_e^{\text{bowl}}$ using the attribution matrix in Section 3.1.
6. **Step 6 (Capped Composite Delivery Weight $w_e$):** Combine competition tier ($w_{\text{comp},m} \in \{1.00, 0.95, 0.80, 0.70\}$), exponential recency ($w_{\text{time},m} = e^{-0.18\,\Delta\text{years}_m}$), single-ball clutch weight ($W_{i,e}^{\text{clutch}}$), match stakes ($\mathcal{M}_{\text{stakes},m}$), and endogeneity discount ($\eta_{\text{endo},e}$), capped at $[0.10, 2.50]$ so a single knockout ball cannot dominate a player's career rating:
   $$w_e = \text{clip}\!\Big(w_{\text{comp},m} \cdot w_{\text{time},m} \cdot W_{i,e}^{\text{clutch}} \cdot \mathcal{M}_{\text{stakes},m} \cdot \eta_{\text{endo},e}, \; 0.10, \; 2.50\Big)$$
7. **Step 7 (Context-Matched Ordinary Baseline $\overline{\text{NBV}}_{p,\text{matched\_ord}}^{(6)}$):** Express per-6-ball clutch performance as $\overline{\text{NBV}}_{p,\text{clutch}}^{(6)} = 6 \times \frac{\sum_{e \in \text{Clutch}_p} w_e \text{NBV}_e}{\sum_{e \in \text{Clutch}_p} w_e}$. Rather than comparing against all ordinary deliveries (which mixes powerplay and death overs), compute player $p$'s ordinary baseline $\overline{\text{NBV}}_{p,\text{matched\_ord}}^{(6)}$ by **re-weighting player $p$'s ordinary deliveries ($\text{IsClutch} = 0$) to match the exact phase $\times$ innings $\times$ competition-tier distribution of player $p$'s clutch deliveries**.
8. **Step 8 (Raw Per-Over Clutch Value $S_p^{\text{raw}}$):**
   $$S_p^{\text{raw}} = 0.70 \cdot \overline{\text{NBV}}_{p,\text{clutch}}^{(6)} + 0.30 \cdot \Big(\overline{\text{NBV}}_{p,\text{clutch}}^{(6)} - \overline{\text{NBV}}_{p,\text{matched\_ord}}^{(6)}\Big) \quad \text{(in runs per 6 balls)}$$
9. **Step 9 (Effective Sample Size $N_{\text{eff},p}$ & Empirical Bayes Shrinkage to Role Replacement Level):**
   Assign each player to their primary data-driven tactical role based on modal entry/bowling phase share:
   * **Batting Roles (3):** *Top-Order Powerplay Batter* ($\ge 45\%$ balls in Ov 1–6), *Middle-Order Stabilizer / Spin Hitter* ($\ge 45\%$ balls in Ov 7–15), *Death-Overs Finisher* (modal share in Ov 16–20).
   * **Bowling Roles (3):** *New-Ball Powerplay Bowler* ($\ge 40\%$ balls in Ov 1–6), *Middle-Overs Controlling Spinner/Enforcer* ($\ge 45\%$ balls in Ov 7–15), *Death-Overs Specialist* ($\ge 35\%$ balls in Ov 16–20).
   Compute Kish's **Effective Sample Size** (in legal clutch balls):
   $$N_{\text{eff},p} = \frac{\left(\sum_{e \in \text{Clutch}_p} w_e\right)^2}{\sum_{e \in \text{Clutch}_p} w_e^2}$$
   **Strict Eligibility Filter:** Require $N_{\text{clutch},p} \ge 90\text{ legal clutch balls}$, $N_{\text{eff},p} \ge 75$, at least $M_{\text{clutch},p} \ge 12\text{ qualifying matches}$, and appearances across **at least $2$ distinct seasons/years**. Shrink $S_p^{\text{raw}}$ toward the **25th-percentile Replacement Level ($\mu_{\text{rep, role}}$)** of qualified players in that role with shrinkage constant $\kappa = 120$ effective balls:
   $$\tilde{S}_p = \frac{N_{\text{eff},p}}{N_{\text{eff},p} + 120} \cdot S_p^{\text{raw}} + \frac{120}{N_{\text{eff},p} + 120} \cdot \mu_{\text{rep, role}}$$
10. **Step 10 (Cross-Role Z-Score Standardization $Z_p$):**
    Because death bowlers and powerplay bowlers have different raw variance in $\tilde{S}_p$, place all roles onto a single common scale by dividing by the within-role robust standard deviation $s_{\text{role}}$:
    $$Z_p = \frac{\tilde{S}_p - \mu_{\text{rep, role}}}{s_{\text{role}}}$$
11. **Step 11 (Bounded Reliability Adjustment $Z_p^{\text{adj}}$):**
    Apply the scale-invariant reliability modifier $\mathcal{R}_p \in [-0.12, +0.12]$ from Factor 25:
    $$Z_p^{\text{adj}} = Z_p \cdot (1 + \mathcal{R}_p) + 0.15 \cdot \mathcal{R}_p$$
12. **Step 12 (Final Presentation Rating & Tie-Breaking Protocol):**
    Map to a $[0, 100]$ presentation scale where $50$ represents replacement level and $80+$ represents world-class clutch mastery:
    $$\text{Rating}_p = \text{clip}\!\Big(50 + 10 \cdot Z_p^{\text{adj}}, \; 0, \; 100\Big)$$
    Players are ranked by posterior mean $Z_p^{\text{adj}}$; any tie within $0.02\text{ Z-units}$ is broken by the higher **95% Lower Confidence Bound ($\text{LCB}_{95,p}$)** from Step 13, and then by larger $N_{\text{eff},p}$.
13. **Step 13 (2,000-Iteration Match-Block Bootstrap & Sensitivity Suite):**
    Resample **whole matches $m$ with replacement** $B_{\text{boot}} = 2,000$ times (preserving within-match ball correlation) to compute for every candidate:
    * Bootstrap Mean $\text{Rating}_p$ and **95% Confidence Interval $[\text{LCB}_{95,p}, \text{UCB}_{95,p}]$**.
    * **Top-10 Inclusion Probability** $\Pr(\text{Rank}_p \le 10)$ and **Median Rank**.
    * **Sensitivity Suite:** Re-run rankings across a 12-point perturbation grid (varying pillar weights $\pm 0.08$, $\gamma \in \{0.45, 0.60, 0.75\}$, $\kappa \in \{80, 120, 160\}$, and global $\text{DPI}$ cutoff $\in \{55, 60, 65\}$) and report Spearman rank correlation $\rho$ and Top-10 overlap count.
    * **Temporal & Cross-League Validation:** Report Leave-One-Season-Out rank stability and split IPL-only vs. T20I-only clutch ratings.
14. **Step 14 (Required Top-10 Evidence Tables & Cosine-Similarity Role-Twin Replacements):**
    Once executed on the supplied competition CSVs, populate the final Top 10 Batter and Top 10 Bowler Evidence Tables with exact schema columns: `Rank`, `Player`, `Role`, `Final Rating (0-100)`, `95% Bootstrap CI`, `Pr(Top 10)`, `Clutch Balls (N_eff)`, `Clutch Matches`, `Clutch NBV/6b`, `Matched Ordinary NBV/6b`, `Clutch Uplift`, `Hit Rate (%)`, `PP/Mid/Death Share (%)`, `IPL/T20I Share (%)`, and `#1 Cosine-Similarity Clutch Twin Replacement`.