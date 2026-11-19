# Unstop entry checklist

This checklist holds the entries of the two Main_Lists, `top10_batters.csv` and `top10_bowlers.csv`, in rank order. Use it while you enter the lists in the Unstop dropdowns:

1. For each entry, select the dropdown option whose text equals the name between backticks exactly, with the same case, spacing and punctuation. It is the `player_name` of the list file.
2. A resolved entry's `player_id` has exactly one Candidate_Name in `competition_team_player_mapping.csv`, and no other `player_id` has that name, so one option should match.
3. An unresolved entry needs a closer look: its `player_id` has no Candidate_Name or several, or its name is a Candidate_Name of more than one `player_id`, so the dropdown may show the name more than once or not at all. Its notes list every Candidate_Name of its `player_id` and every `player_id` that has the name, each with its `team_name` values. Use the teams to pick the option of the listed `player_id`, not a namesake.
4. Tick the box (`- [x]`) once the dropdown shows the right player.

Entries: 20 (10 batters, 10 bowlers); 19 resolved, 1 unresolved (Bowler rank 3, `Rashid Khan`).

## Batters (`top10_batters.csv`)

- [ ] Batter rank 1: `Abhishek Sharma` (player_id 6281), resolved
- [ ] Batter rank 2: `Tim David` (player_id 6396), resolved
- [ ] Batter rank 3: `Heinrich Klaasen` (player_id 4260), resolved
- [ ] Batter rank 4: `Shivam Dube` (player_id 4755), resolved
- [ ] Batter rank 5: `Jos Buttler` (player_id 813), resolved
- [ ] Batter rank 6: `Phil Salt` (player_id 5553), resolved
- [ ] Batter rank 7: `Daryl Mitchell` (player_id 4106), resolved
- [ ] Batter rank 8: `Rajat Patidar` (player_id 5039), resolved
- [ ] Batter rank 9: `Andre Russell` (player_id 1318), resolved
- [ ] Batter rank 10: `Suryakumar Yadav` (player_id 1699), resolved

## Bowlers (`top10_bowlers.csv`)

- [ ] Bowler rank 1: `Varun Chakravarthy` (player_id 6256), resolved
- [ ] Bowler rank 2: `Bhuvneshwar Kumar` (player_id 360), resolved
- [ ] Bowler rank 3: `Rashid Khan` (player_id 4722), **unresolved**
    - Why unresolved: `Rashid Khan` is a Candidate_Name of 2 player_ids.
    - Candidate_Names of player_id 4722: `Rashid Khan` (21 mapping rows).
    - player_ids with the Candidate_Name `Rashid Khan`: 4722, 20172.
    - `team_name` values in `competition_team_player_mapping.csv`:
        - player_id 4722 (this entry): Afghanistan; Gujarat Titans
        - player_id 20172: Nepal
- [ ] Bowler rank 4: `Mustafizur Rahman` (player_id 4373), resolved
- [ ] Bowler rank 5: `Wanindu Hasaranga` (player_id 4900), resolved
- [ ] Bowler rank 6: `Jasprit Bumrah` (player_id 2933), resolved
- [ ] Bowler rank 7: `Sunil Narine` (player_id 781), resolved
- [ ] Bowler rank 8: `Eshan Malinga` (player_id 39657), resolved
- [ ] Bowler rank 9: `Jason Holder` (player_id 2633), resolved
- [ ] Bowler rank 10: `Maheesh Theekshana` (player_id 19316), resolved
