# Data audit

The Auditor (the `audit` Pipeline_Stage, Requirement 4) profiled the Source_CSVs as loaded by the `load` stage: 200,999 Deliveries in `ball_by_ball.csv`, 883 matches in `matches.csv` and 6,413 rows in `competition_team_player_mapping.csv`. Count tables give missing values their own category, shown as `<missing>`.

## Fact_Checks (criteria 12-14)

| Fact_Check | Requirement 4 | Check | Observed value | Result |
|---|---|---|---|---|
| `no_ball_extras_2_runs_13614` | criterion 12 | at least one No_Ball Delivery in match 13614, innings 1, `overid` 18 has `extras_2_runs` = 4 | `extras_2_runs` of the 1 No_Ball among the Over's 7 Deliveries: 4 (`extras_type` other) | pass |
| `revised_overs_50408` | criterion 13 | `first_inn_revised_overs` or `second_inn_revised_overs` equals 54 for match 50408 | `first_inn_revised_overs` = 54, `second_inn_revised_overs` = 54 | pass |
| `retire_out_label` | criterion 14 | every Delivery with `dismissal_type_id` 9 has the `dismissal_type` label "Retire Out" | 6 Deliveries, labelled "Retire Out" (6) | pass |

All 3 Fact_Checks pass.

## Dismissal-code classification (criterion 10)

- Counted_Wicket codes: 1, 2, 3, 4, 5, 6, 9, 10
- Retired_Hurt codes: none

Rules, tried in order: codes 1-6 and 9 are Counted_Wickets (`base_code`); any other code is Retired_Hurt if one of its labels contains "hurt" or "not out", ignoring case (`label`), or if its Reappearance_Share is above 0 (`reappearance`); any remaining code is a Counted_Wicket (`default`).

| Code | Labels | Rows | Classification | Rule | Reason |
|---|---|---|---|---|---|
| 1 | Bowled | 1,775 | Counted_Wicket | `base_code` | codes 1-6 and 9 are Counted_Wickets |
| 2 | Caught | 7,557 | Counted_Wicket | `base_code` | codes 1-6 and 9 are Counted_Wickets |
| 3 | LBW | 709 | Counted_Wicket | `base_code` | codes 1-6 and 9 are Counted_Wickets |
| 4 | RunOut | 600 | Counted_Wicket | `base_code` | codes 1-6 and 9 are Counted_Wickets |
| 5 | Stump | 214 | Counted_Wicket | `base_code` | codes 1-6 and 9 are Counted_Wickets |
| 6 | Hit Wicket | 17 | Counted_Wicket | `base_code` | codes 1-6 and 9 are Counted_Wickets |
| 9 | Retire Out | 6 | Counted_Wicket | `base_code` | codes 1-6 and 9 are Counted_Wickets |
| 10 | Obstracting the field; Obstructing the field | 3 | Counted_Wicket | `default` | no label contains "hurt" or "not out", and the Reappearance_Share is 0 (0 of 3 rows) |

## Tables saved in `outputs/audit/` (criterion 11)

Each Run rewrites these files.

| File | Requirement 4 | Contents |
|---|---|---|
| `extras_type_x_extras1_runs.csv` | criterion 1 | Deliveries in each `extras_type` × `extras1_runs` cell |
| `extras_type_x_extras_2_runs.csv` | criterion 1 | Deliveries in each `extras_type` × `extras_2_runs` cell |
| `dismissal_codes.csv` | criterion 2 | Rows of each `dismissal_type_id` and of each `dismissal_type` label recorded with it |
| `stage_counts.csv` | criterion 3 | Matches with each `stage` value |
| `result_counts.csv` | criterion 3 | Matches with each distinct `result` string |
| `bowl_speed_summary.csv` | criterion 4 | Shares of `bowl_speed` = 0 and missing; quantiles of the values above 0 |
| `bowling_type_stability.csv` | criterion 5 | Distinct `bowling_type` values and the Modal_Bowling_Type of each `bowler_id` |
| `names_with_several_ids.csv` | criterion 6 | Each `player_name` that maps to more than one `player_id`, with those IDs |
| `ids_with_several_names.csv` | criterion 6 | Each `player_id` that maps to more than one `player_name`, with those names |
| `revised_values.csv` | criterion 7 | Matches with each value of `first_inn_revised_overs`, `second_inn_revised_overs` and `revised_target` |
| `over_sizes.csv` | criterion 8 | Overs with each count of Deliveries and of Legal_Balls |
| `other_dismissal_codes.csv` | criterion 9 | Labels, rows and Reappearance_Share of each dismissal code other than 1-6 and 9 |
| `dismissal_classification.csv` | criterion 10 | Classification of each dismissal code and the rule that decided it |

## Data facts

### `extras_type` against `extras1_runs` and `extras_2_runs` (criterion 1)

| `extras_type` \ `extras1_runs` | 1 | 2 | 3 | 4 | 5 | 6 | 7 | `<missing>` |
|---|---|---|---|---|---|---|---|---|
| None | 0 | 0 | 0 | 0 | 0 | 189,167 | 0 | 0 |
| Overstep | 576 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Penalty | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| byes | 0 | 0 | 0 | 529 | 0 | 0 | 0 | 0 |
| legbyes | 0 | 0 | 2,748 | 0 | 0 | 0 | 0 | 0 |
| other | 0 | 0 | 0 | 0 | 264 | 0 | 0 | 0 |
| wide | 0 | 7,714 | 0 | 0 | 0 | 0 | 0 | 0 |
| `<missing>` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

| `extras_type` \ `extras_2_runs` | 0 | 1 | 4 | `<missing>` |
|---|---|---|---|---|
| None | 189,166 | 1 | 0 | 0 |
| Overstep | 555 | 12 | 9 | 0 |
| Penalty | 1 | 0 | 0 | 0 |
| byes | 529 | 0 | 0 | 0 |
| legbyes | 2,748 | 0 | 0 | 0 |
| other | 250 | 7 | 7 | 0 |
| wide | 7,714 | 0 | 0 | 0 |
| `<missing>` | 0 | 0 | 0 | 0 |

### Dismissal codes and labels (criterion 2)

| Code | Code rows | Label | Label rows |
|---|---|---|---|
| 1 | 1,775 | Bowled | 1,775 |
| 2 | 7,557 | Caught | 7,557 |
| 3 | 709 | LBW | 709 |
| 4 | 600 | RunOut | 600 |
| 5 | 214 | Stump | 214 |
| 6 | 17 | Hit Wicket | 17 |
| 9 | 6 | Retire Out | 6 |
| 10 | 3 | Obstracting the field | 2 |
| 10 | 3 | Obstructing the field | 1 |

### Dismissal codes other than 1-6 and 9 (criterion 9)

| Code | Labels | Rows | Reappearing rows | Reappearance_Share |
|---|---|---|---|---|
| 10 | Obstracting the field; Obstructing the field | 3 | 0 | 0.000000 |

Codes 7 and 8, which criterion 9 expected, do not occur.

### Match stages and results (criterion 3)

- `stage`: League (620), Others (226), Final (12), Qualifier (10), Super 4 (6), Eliminator (5), Semi Final (4), `<missing>` (0).
- `result`: 670 distinct strings and 0 missing values; the most frequent are No Result (21), Match Abandoned (19), India Won by 7 wickets (8), Delhi Capitals Won by 6 wickets (7), Kolkata Knight Riders Won by 8 wickets (6).

### `bowl_speed` (criterion 4)

| Statistic | Value |
|---|---|
| Deliveries | 200,999 |
| `bowl_speed` = 0 | 77,861 (38.737%) |
| `bowl_speed` missing | 0 (0.000%) |
| `bowl_speed` below 0 | 1 |
| `bowl_speed` above 0 | 123,137 |
| Minimum of the values above 0 | 1.320 |
| 1st percentile of the values above 0 | 79.700 |
| Median of the values above 0 | 123.800 |
| 99th percentile of the values above 0 | 147.200 |
| Maximum of the values above 0 | 1,456.100 |

### Bowling types (criterion 5)

647 bowlers; by number of distinct non-missing `bowling_type` values: 1: 600, 2: 46, 3: 1. 47 bowlers have more than one value. The latest-Delivery tie rule decided 0 Modal_Bowling_Types.

### Player names and IDs in `competition_team_player_mapping.csv` (criterion 6)

7 `player_name` values map to more than one `player_id`:
- Abdul Samad: 6922 (6 rows), 7446 (5 rows)
- Bashir Ahmad: 29833 (4 rows), 39479 (1 row)
- Basil Hameed: 8638 (5 rows), 39444 (1 row)
- Haider Ali: 7338 (4 rows), 28756 (5 rows)
- Mohammad Ali: 3083 (1 row), 20229 (2 rows)
- Muhammad Farooq: 27373 (5 rows), 28978 (1 row)
- Rashid Khan: 4722 (21 rows), 20172 (1 row)

0 `player_id` values map to more than one `player_name`.

### Revised overs and targets in `matches.csv` (criterion 7)

- `first_inn_revised_overs`: 30 (2), 42 (1), 48 (1), 54 (4), 60 (1), 66 (3), 72 (1), 78 (1), 84 (3), 90 (2), 102 (2), 108 (2), 114 (2), `<missing>` (858)
- `second_inn_revised_overs`: 30 (2), 42 (2), 48 (3), 54 (2), 60 (2), 66 (1), 72 (1), 78 (1), 84 (1), 90 (5), 102 (2), 114 (3), `<missing>` (858)
- `revised_target`: 33 (1), 56 (1), 69 (1), 78 (1), 91 (1), 94 (1), 96 (1), 103 (1), 104 (1), 115 (1), 116 (1), 125 (1), 126 (1), 130 (1), 136 (1), 143 (1), 147 (1), 151 (1), 152 (1), 168 (1), 171 (1), 203 (1), 213 (1), `<missing>` (860)

### Deliveries and Legal_Balls per Over (criterion 8)

32,350 Overs in all innings, Super_Over_Innings included; Legal_Balls follow the Config class mapping `data_rules.extras_type_classes`.

| Per Over | Overs with this many Deliveries | Overs with this many Legal_Balls |
|---|---|---|
| 0 | 0 | 1 |
| 1 | 93 | 102 |
| 2 | 88 | 100 |
| 3 | 103 | 109 |
| 4 | 129 | 133 |
| 5 | 146 | 147 |
| 6 | 25,081 | 31,757 |
| 7 | 5,346 | 1 |
| 8 | 1,085 | 0 |
| 9 | 222 | 0 |
| 10 | 42 | 0 |
| 11 | 13 | 0 |
| 13 | 2 | 0 |

## Score reconstruction (Requirement 5)

The Score_Reconstructor (the `reconstruct` Pipeline_Stage) rebuilt the team runs of every Delivery under 4 candidate Extras_Encodings and checked the rebuilt scores of each match against its recorded result (criteria 16-20).

Selected Extras_Encoding (criterion 7): `byes_one`, with the highest mean Exact_Pass_Rate (66.711%).

### Pass rates (criterion 21)

Exact_Pass_Rate of each Score_Check, in percent of the matches the check applies to, as in `pass_rates.csv`:

| Extras_Encoding | `runs_margin` | `chase` | `wickets_margin` | `tie` | `over` | Mean | Selected |
|---|---|---|---|---|---|---|---|
| `F30` | 11.192 | 33.652 | 99.761 | 0.000 | 99.654 | 48.852 |  |
| `byes_one` | 31.873 | 52.267 | 99.761 | 50.000 | 99.654 | 66.711 | yes |
| `byes_one_nb_plain` | 30.657 | 51.313 | 99.761 | 50.000 | 99.654 | 66.277 |  |
| `byes_one_penalty5` | 31.873 | 52.267 | 99.761 | 50.000 | 99.654 | 66.711 |  |

Within_Tolerance_Pass_Rates of `byes_one` at the Score_Tolerance of 6 runs: `runs_margin` 93.431%, `chase` 96.181%, `wickets_margin` 99.761%, `tie` 90.000%, `over` 99.654%. A runs-margin, chase or tie check passes within tolerance when it misses by at most 6 runs; the wickets-margin and over checks must pass exactly.

### Warnings (criterion 22)

- Warning: the Exact_Pass_Rate of the `runs_margin` check under `byes_one` is 31.873%, below 95.0%.
- Warning: the Exact_Pass_Rate of the `chase` check under `byes_one` is 52.267%, below 95.0%.
- Warning: the Exact_Pass_Rate of the `tie` check under `byes_one` is 50.000%, below 95.0%.

### Match exclusion (criteria 24 and 27)

- Score_Tolerance (D20, read from the Config): 6 runs.
- Matches excluded from model training and player rating: 48. Under `byes_one`, each fails the over check or the wickets-margin check, or misses the runs-margin, chase or tie check by more than 6 runs; `excluded_matches.csv` lists the 48 conditions they meet.
- `failing_matches.csv` also lists the runs checks that miss by at most 6 runs; they do not exclude a match.

Matches that criterion 24 would exclude at each Score_Tolerance of the Config reporting grid, as in `exclusion_counts.csv`:

| Score_Tolerance (runs) | Matches excluded |
|---|---|
| 0 | 486 |
| 3 | 165 |
| 6 (the Config value) | 48 |
| 10 | 10 |
