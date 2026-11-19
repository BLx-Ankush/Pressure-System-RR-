"""Hypothesis strategies for the property tests (design section "Testing Strategy", Generators).

Every strategy returns plain pandas DataFrames or tuples and does no file I/O, so a property test
draws its 100 examples in seconds. The settings that the Config fixes are read from ``config.py``
once, when this module is imported, through ``superr.settings.load_settings``: the
``extras_type`` class mapping (``EXTRAS_TYPE_CLASSES``), the candidate Extras_Encodings
(``CANDIDATE_ENCODINGS``) and the Score_Tolerance D20 (``SCORE_TOLERANCE``). The strategies
therefore follow any change to the Config, and a test can use ``SCORE_TOLERANCE`` in its
``hypothesis.example`` cases.

Reconstruction strategies (task 5.5; Requirement 5)
---------------------------------------------------

``innings(max_overs=4, *, match_id=1, innings_no=1, batting_team_id=1, bowling_team_id=2, bad_over=False)``
    One innings of Deliveries: a DataFrame with every ``ball_by_ball.csv`` column in header order
    (``BALL_BY_BALL_COLUMNS``), typed as the Data_Loader types them (``Int64``, ``boolean``,
    ``string``) and sorted in the Requirement 5 order. ``innings_ball_num`` and ``over_ball_num``
    count every Delivery from 1, and ``overid`` counts Overs from 1, as in the real file.

    * 1 to ``max_overs`` Overs. Every Over but the last has exactly 6 Legal_Balls under the
      Config class mapping; the last has 1-6 and may end with Wides or No_Balls, so an innings can
      end on a No_Ball. With ``bad_over=True`` one Over fails the over check (Requirement 5,
      criterion 20): 7 Legal_Balls, or 5 in an Over before the last.
    * All 7 ``extras_type`` values of the Config mapping, Legal_Balls weighted towards ``None``.
      Bat runs are 0 on Wides, byes, leg-byes and Penalty Deliveries; ``extras_2_runs`` is mostly 0,
      sometimes 1 or 4 (Wides and No_Balls) or 1 (other Legal_Balls). ``extras1_runs`` holds the
      per-type code seen in the real file.
    * Boundaries: ``is_four`` exactly when ``runs`` is 4 and ``is_six`` exactly when it is 6.
    * Wickets with codes 1-6, Retire Out (9) and Retired_Hurt (7, label "Retired Hurt"). Bowled,
      caught, LBW and hit-wicket balls score no bat runs; run outs score 0-2 and may remove the
      non-striker; Wides allow run outs and stumpings, other extras only run outs. A Retired_Hurt
      batter comes back at the next departure, so he reappears in a later Delivery when another
      batter departs. The innings ends at the Delivery after which no batter is left (all out).
    * Strike changes on odd runs and at the end of each Over; bowlers change every Over and
      sometimes mid-Over. Player ids are ``team_id * 100 + slot``, where slot is the batting order.
    * ``innings_no`` 3 or 4 gives a Super_Over_Innings: one Over that ends at the second departure.

``scorecard_match(*, encoding=None, shift=None, shift_innings=None, result_type=None, bad_over=False, max_overs=3, match_id=None)``
    A ``MatchCase`` built from a scorecard (Scheduled_Balls, innings totals, wickets and a
    result) under one Extras_Encoding (the given one, or one drawn from ``CANDIDATE_ENCODINGS``):

    * ``balls``: the Deliveries of innings 1 and 2 (plus Super_Over_Innings 3 and 4 for most
      super-over wins), built from the same Delivery generator as ``innings``, whose team runs
      under the encoding add up exactly to ``scorecard.totals``.
    * ``matches``: one ``matches.csv`` row (``MATCH_COLUMNS``): revised-overs columns matching
      the Scheduled_Balls (blank for 120 balls, and ``second_inn_revised_overs`` sometimes blank
      to use the fallback), an optional ``revised_target``, ``win_team_id`` (blank for a plain
      tie) and a ``result`` string from the ``result_strings`` grammar, with its noise.
    * ``encoding`` and ``scorecard`` (a ``Scorecard``: what the Deliveries hold, the Target and
      the recorded result type and margin).
    * ``expected``: the five Score_Checks of the match in the ``score_checks`` schema, from a
      direct transcription of Requirement 5, criteria 16-20.

    ``result_type`` is one of WIN_BY_RUNS, WIN_BY_WICKETS, TIE or SUPER_OVER_WIN (drawn when
    None). ``shift`` (an int or a strategy of ints) changes bat runs of Deliveries before the
    innings' last one, after the result was recorded, so that the total of ``shift_innings``
    (1 or 2, drawn when None) moves by that many runs in either direction. The match is built so
    that every runs check that depends on the moved total (all of them, except an innings-1 shift
    under a ``revised_target``) gets a Runs_Check_Miss of exactly ``abs(shift)``: a chase moved
    down is driven by the shortfall and a chase moved up by the excess. ``bad_over=True`` adds one
    bad Over, so the over check fails and nothing else changes.

``check_tables(*, tolerance=None, encodings=None, max_matches=6, applicable_only=False)``
    A ``CheckTables(checks, selected, tolerance)``. ``checks`` is a ``score_checks`` table
    (``SCORE_CHECK_COLUMNS``): one row per encoding, match and check, sorted by encoding (Config
    order), ``match_id`` and check (``CHECKS`` order). ``applies`` comes from the match's result
    (the same under every encoding) or, now and then, from an arbitrary set of checks;
    ``passed`` is missing where a check does not apply; ``runs_check_miss`` is filled only for
    applicable runs checks, clustered at 0, t and t + 1, with ``passed`` exactly when it is 0. The
    wickets-margin and over results do not depend on the encoding, so they are the same under
    every encoding. ``encodings`` defaults to a drawn non-empty subset of the candidates in
    Config order; ``selected`` is one of them; ``tolerance`` (t) is drawn from 0-3 and from
    D20 - 1 to D20 + 1 unless given. ``applicable_only=True`` keeps only the applicable rows.

``result_strings()``
    A ``ResultCase(text, expected)``: a ``result`` string from the Result_Parser grammar
    (team names, "won by" / "won" / "Won By", runs, run, wickets, wicket, wkts, wkt with zero
    or more spaces after the margin, an optional DLS or D/L marker, super-over phrasings with and
    without a margin, tie, abandoned and no-result phrasings, and strings that must stay
    unparsed) with letter-case and spacing noise, and the ``ParsedResult`` it must parse to,
    worked out from how the string was built.

``score_checks_frame(rows)`` builds a ``score_checks`` table with the schema dtypes from tuples in
``SCORE_CHECK_COLUMNS`` order or from mappings, for hand-written ``hypothesis.example`` cases.

Fold strategies (task 8.4; Requirement 7)
-----------------------------------------

The number of folds D15 (``N_FOLDS``, ``design_choices.D15_n_folds``) and the fold seed
(``FOLD_SEED``, ``seeds.folds``) are also read from the Config when this module is imported.

``fold_tables(*, n_folds=None, fits=False)``
    A ``FoldTables(matches, n_folds, seed, fold_table, deliveries, fits)`` for the properties of
    fold assignment, the leakage guard, sandbox views and per-fold statistics (P18, P19, P20, P43):

    * ``matches``: one row per match, rows in shuffled order on a RangeIndex, with ``match_id``
      (Int64, as the Data_Loader types it; distinct, and no two ids consecutive) and
      ``comp_type`` (str: IPL, T20I or A_Team). ``assign_folds(matches, matches["comp_type"],
      ...)`` takes them as they are. Each Competition_Type mostly has 3-15 matches, but now and
      then only 1-2 (as A_Team in the real data) or none. There is at least 1 match, and at least
      ``n_folds`` with ``fits``.
    * ``n_folds``: the given int, or a draw from the given strategy, or from 2-7 (3-7 with
      ``fits``, because an inner fit needs 3 folds) with D15 drawn most often.
    * ``seed``: the fold seed, a draw from 0 to 2**32 - 1 or ``FOLD_SEED``.
    * ``fold_table``: ``assign_folds(matches, matches["comp_type"], n_folds=n_folds, seed=seed)``,
      the Fold_Table that the fits refer to. P18 rebuilds it itself, since it tests assign_folds.
    * ``deliveries``: ``row`` (0, 1, ...) and ``match_id`` of 0-3 Deliveries per match, int32 and
      sorted by match as in the ``balls`` table, with no ``fold`` column (attach_folds adds it).
      Most examples hold matches with no Delivery, which still get a fold (criterion 1).
    * ``fits``: empty unless ``fits=True``; then 1-5 FitCases with distinct FitSpecs of
      ``sandbox_plan(n_folds)``, a mix of valid and leaky ones. With at least ``n_folds``
      matches, the Fold_Table leaves no fold empty, so every fit has training matches and can leak.

    A ``FitCase(spec, model, train_match_ids, es_match_ids, predicted_match_ids, leak, overlap)``
    is plain data for the leakage guard (criteria 6 and 12). The id arrays are int32, ascending:

    * ``spec``: the FitSpec: ``fit_id``, ``sandbox_k``, ``inner_j``, ``excluded_folds`` ({k} for an
      outer fit, {j, k} for an inner fit) and ``train_folds``.
    * ``model``: a model of criterion 5 that makes such a fit. Only the Skill_Model,
      Remaining_Runs_Model and Win_Probability_Model make inner fits. The names hold no digit, so
      they cannot be mistaken for a fold number or a count in an error message.
    * ``train_match_ids``: the match of each Training_Row. A match has 1 + (match_id mod 3) rows,
      so ids repeat and a guard must count distinct matches.
    * ``es_match_ids``: the distinct early-stopping validation matches; empty for a fit without
      early stopping. Only the Remaining_Runs_Model, Win_Probability_Model and Outcome_Model have
      them.
    * ``predicted_match_ids``: every match of ``spec.predicted_fold`` (k for an outer fit, j for an
      inner one), whose rows receive the fit's output.
    * ``leak``: None for a valid fit, whose training matches are all of the matches of
      ``spec.train_folds`` or a non-empty subset, and whose validation matches are whole matches
      drawn from them. ``LEAK_TRAIN`` adds matches of the excluded folds to the Training_Rows;
      ``LEAK_EARLY_STOPPING`` adds them to the validation matches only, and the Training_Rows stay
      clean. A leak takes some matches of the excluded folds, or all of one excluded fold (a fit
      that forgot to drop it).
    * ``overlap``: the number of distinct matches of the excluded folds among the training and
      validation matches together; 0 exactly when ``leak`` is None. A case leaks in one of the
      two sets only, so the count is the same for a guard that checks the sets one at a time.

Cell-table strategies (task 17.3; Requirement 15, criteria 20 and 21)
---------------------------------------------------------------------

``cell_tables(*, n_folds=None, pooled=False, empty_pooled=False, cells=None)``
    A ``CellTables(deliveries, n_folds, cells, encoding, emptied, empty_pooled)`` for the
    properties of the Training_Folds cell tables of ``value.py`` (``extras_cell_means``,
    ``free_hit_premium`` and ``fallback_counts``; P43, P44):

    * ``deliveries``: Deliveries of folds 1 to ``n_folds``, every fold holding at least one, so
      every fold is a target fold of the cell tables. Rows are sorted by ``match_id`` and
      ``innings`` on a RangeIndex. Columns: ``row`` (int32: 0, 1, ...), ``match_id``, ``innings``
      and ``fold`` (int32); the cell column named ``cells`` (Int8: 3 x (innings - 1) + Phase, so
      0-2 in innings 1 and 3-5 in innings 2, and missing in a Super_Over_Innings, which has no
      ``state`` row); ``cls`` (int8 DeliveryClass code: Legal_Ball, Wide or No_Ball);
      ``extras_type`` (str, a Config value of that class); ``runs`` (int32 bat runs, 0 on Wides,
      byes, leg-byes and Penalty Deliveries); ``extras_2_runs`` and ``team_runs`` (int32, the team
      runs under ``encoding`` by the DD-4 rules, so a Wide scores 1 + ``extras_2_runs`` and a
      No_Ball's penalty runs, team runs - bat runs, are 1 or more); and ``free_hit``,
      ``is_super_over`` and ``excluded`` (bool). Bat runs, ``extras_2_runs`` and ``extras_type``
      come from the value lists of ``innings``. When ``cells`` is not ``cell``, a decoy ``cell``
      column (Int8) holds another cell of the same innings, so code that reads ``cell`` instead of
      the named column gets valid but wrong cells.
    * Matches: each fold has 1-3 matches. Match j (0, 1 or 2) of fold f has the id
      1 + j x n_folds + (f - 1), and fold f's excluded match, when it has one, the id
      1 + 3 x n_folds + (f - 1). The folds therefore interleave in match order, and two folds
      never share a match id, also across draws with the same ``n_folds``, so a test may swap in
      the fold-k rows of another draw.
    * Training rows: the rows in innings 1 or 2, outside Super_Over_Innings and outside excluded
      matches, which ``common.training_mask`` keeps. They are also the Bowler_Deliveries whose
      bowler NBV is computed (innings 1 and 2 of the matches used for player rating), so they are
      the ``bowler_rows`` of ``fallback_counts``. Each training row is of exactly one kind of
      CELL_KINDS: KIND_WIDE (a Wide), KIND_NO_BALL (a No_Ball), KIND_FREE_HIT (a Legal_Ball with
      ``free_hit``) or KIND_NON_FREE_HIT (a Legal_Ball without it). A Wide or No_Ball may have
      ``free_hit`` set; it is still not a Free_Hit Legal_Ball.
    * Holders: for each cell c and kind q, the strategy chooses the folds whose training rows hold
      q in c, with 1-3 such rows each: every fold (drawn most often), two or more folds (with 3
      folds or more), one fold, or no fold.
    * Dropped rows: 0-6 rows that the training filters must drop. Each is either a
      Super_Over_Innings Delivery (innings 3 or 4, ``is_super_over``, cell missing) of a match
      that is not excluded, or an innings 1-2 Delivery (cell set) of its fold's excluded match,
      whose rows all have ``excluded`` set. When ``emptied`` is not empty, about half of them go
      to a fold f, cell c and kind q such that (k, c, q) is in ``emptied`` for some k other than
      f. Code that keeps such an excluded-match row gets a wrong pooled flag, and code that keeps
      a Super_Over_Innings row meets a missing cell. A fold whose training rows would hold no
      kind in any cell (very rare) also gets one Super_Over_Innings dot ball, so that it has a
      row.
    * ``emptied``: a frozenset of (k, c, q) triples, with k a target fold (1 to ``n_folds``), c a
      cell (0-5) and q a kind. It holds exactly the triples for which the training rows of the
      folds other than k (the Training_Folds of k) hold no Delivery of kind q in cell c. It
      predicts the pooled flags of the (k, c) rows of the cell tables: ``pooled_wide_runs`` is
      set exactly when (k, c, KIND_WIDE) is in ``emptied``; ``pooled_nb_penalty_runs`` and
      ``pooled_nb_bat_runs`` exactly when (k, c, KIND_NO_BALL) is; and
      ``pooled_free_hit_premium`` exactly when (k, c, KIND_FREE_HIT) or (k, c,
      KIND_NON_FREE_HIT) is.
    * ``empty_pooled``: a frozenset of (k, q) pairs: exactly those for which (k, c, q) is in
      ``emptied`` for all six cells c, so that the pooled set of kind q in k's Training_Folds is
      empty. For a pair with KIND_WIDE or KIND_NO_BALL, ``extras_cell_means`` raises StageError
      (exit code 5); for a pair with KIND_FREE_HIT or KIND_NON_FREE_HIT, ``free_hit_premium``
      does.
    * ``n_folds``: the given int, a draw from the given strategy, or D15 when None. ``cells``: the
      given name, or a draw from CELL_NAMES (``cell``, or ``cell_d8``, which stands for the cells
      of the D8 robustness case). ``encoding``: a candidate Extras_Encoding, drawn in Config order.

    By default ``empty_pooled`` is empty, and with 3 or more folds every kind is held, in some
    cell, by at least 3 folds. So whatever a test does to the rows of one fold (alters their
    values, kinds or cells, or moves them into an excluded match), every pooled set stays
    non-empty and the cell tables can still be built. ``pooled=True`` lets one fold k alone hold
    a kind q in a cell c, so (k, c, q) is in ``emptied`` and fold k has Bowler_Deliveries in cell
    c: at least one fallback count is above 0. ``empty_pooled=True`` keeps a kind q out of all
    six cells of one target fold k's Training_Folds, so (k, q) is in ``empty_pooled`` and the
    cell tables raise. Shrinking moves towards every fold holding every kind in every cell once
    (no emptied cell), one match per fold, no dropped row, dot balls without extras, the first
    candidate encoding and the cell column ``cell``.

Contribution strategies (task 18.4; Requirement 16, criterion 2; Requirement 19, criterion 2)
---------------------------------------------------------------------------------------------

The Recency_Weight half-life D9 (``HALF_LIFE_YEARS``, ``design_choices.D9_half_life_years``) and
the knockout Stakes_Weight D10 (``KNOCKOUT_WEIGHT``, ``design_choices.D10_knockout_weight``) are
also read from the Config when this module is imported.

``contributions(*, n_players=None, n_matches=None, roles=None, force=(), charges=True, dense=False)``
    A ``ContributionCase(contrib, frame, entries, matches, match_ids, roles, forced, twins)`` for
    the Rating_Engine properties (P45-P55, P58, P59) and the Usual_Baseline and Uplift unit tests:

    * ``matches``: the match universe, one row per match on a RangeIndex (row i is match index
      i), with the columns and dtypes of ``rating.match_weights`` (MATCH_WEIGHT_COLUMNS), so it
      can stand in for the match_weights table. ``match_id`` (int64) is distinct and ascending.
      ``match_date`` lies between 2021-01-01 and 2026-11-01 whatever the id, so date order and id
      order differ; matches often share a date, and some fall on 2024-12-31 or 2025-01-01, either
      side of LATE_PERIOD_START. ``calendar_year`` and ``period`` follow the date. ``comp_type``
      is IPL, T20I or A_Team, ``full_member_match`` is set on some T20Is only, and ``comp_class``
      follows them (IPL, FULL_MEMBER, else ASSOCIATE). ``knock_out`` is 0 or 1 and
      ``stakes_weight`` D10 or 1.0. ``recency_weight`` is the D9 formula on ``days_to_last`` (the
      days to the universe's last date), rounded to a multiple of 1/64 and at least 1/64, and
      ``delivery_weight`` = ``recency_weight`` x ``stakes_weight``. Every entry of a match has
      this weight, so no weight depends on a pressure flag (Requirement 16, criterion 1).
    * ``match_ids``: the ``match_id`` column as an int64 array: the match universe of
      ``contrib``, which every multiplicity vector follows.
    * ``entries``: one row per value entry of a player in a Role (ENTRY_COLUMNS): ``role`` (str),
      ``player_id``, ``match_id`` and ``cell`` (int64; cells 0-5), ``pressure`` (bool),
      ``charge`` (bool: a Dismissal_Charge, batter Role only, rather than a Delivery), ``value``
      (float64: the Delivery's NBV, or the charge, -Omega, which is 0 or less), ``weight`` (the
      match's Delivery_Weight), the entry's terms ``s`` = weight x value, ``w`` (the weight; 0
      for a charge) and ``n`` (int64: 1; 0 for a charge), and ``half`` (int8: the Half code of
      the player's match in the Role). Rows are sorted as ``frame``. These are the per-Delivery
      data for brute-force references.
    * ``frame``: ``contribution_frame(entries, matches)``, the contributions table
      (CONTRIBUTION_COLUMNS, with the dtypes of ``Contributions.to_frame``), summed here
      independently of ``rating``. ``contrib``: ``Contributions.from_frame(frame, match_ids)``.
    * ``roles``: the Roles generated, in ROLES order. ``forced`` and ``twins``: see below.

    ``contribution_frame(entries, matches)`` transcribes Requirement 16, criterion 2. It sums the
    entries of each Role, player, match, cell and pressure class; a Dismissal_Charge adds weight x
    charge to ``s`` and nothing to ``w`` or ``n``, so a group of charges alone is a charge-only
    row (``n`` = 0, ``w`` = 0, ``s`` 0 or less). It numbers the halves of Requirement 18,
    criterion 7: a player's matches with a Delivery in the Role are numbered 1, 2, ... by
    ``match_date`` and then ``match_id``, odd numbers getting Half.ODD and even ones Half.EVEN,
    and a match with charges alone gets Half.NONE. Rows are sorted as ``build_contributions``
    sorts them: the batter Role first, then player, match index, cell and class (ordinary first).
    It reads only the first seven ENTRY_COLUMNS, so a test can flip flags, change values, or add
    or drop entries, and rebuild the contributions.

    Values: NBVs are multiples of 1/4 run, for a batter mostly from -4 to +6 and on about one
    Delivery in 8 a wicket-sized -10 to -30; a bowler's are their negatives. A charge is -Omega
    with Omega from 0 to 30 runs in quarter runs. With D10 a multiple of 1/4 (1.25 as delivered),
    every weight, entry term and contribution sum is exact in float64: sums do not depend on their
    order, and a per-match sum that should be 0 is exactly 0 (Hit_Rate counts sums above 0).

    Players: each Role has ``n_players`` random players (drawn for each Role) and the forced
    players. Player ids are distinct within a Role and come from a small range shared by the
    Roles, so the same id is often a batter and a bowler. A random player has 1-3 home cells,
    plays 1-4 matches with 1-6 Deliveries in each, in his home cells, and has one of four pressure
    shares: about 25%, 50%, none or all. A random batter has 0-2 charges, mostly in his own matches
    and cells but sometimes in another match (a charge-only match) or cell, and one random batter
    in 16 has 1-2 charges and no Delivery. With ``dense=True`` a random player plays every match,
    with a Pressure_Ball and an Ordinary_Ball in one home cell of each plus 0-1 more Deliveries,
    and no batter is charges only. Under multiplicities that sum to 2 or more he then has 2 or
    more Pressure_Matches, an Ordinary_Ball and a Matched_Cell, so every Bootstrap_Resample can
    rank him.

    Forced cases: ``force`` (names from FORCED_CASES) adds to each generated Role one player for
    each case (CHARGE_ONLY: the batter Role only), and ``forced[case, role]`` is his
    ``player_id``. Each case holds under all-ones multiplicities (the Main_Ranking); other
    multiplicities can undo it, a 0 by dropping a match and a 2 by doubling one.

    * ALL_MATCHED: Pressure_Balls in 2-4 cells, each holding an Ordinary_Ball too, so the
      Unmatched_Pressure_Share is 0 and the Usual_Baseline has 2-4 Matched_Cell shares.
    * UNMATCHED_CELL: 1-3 Matched_Cells and 1-3 cells with Pressure_Balls only.
    * NO_MATCHED_CELL: Pressure_Balls and Ordinary_Balls in disjoint cells: the Usual_Baseline is
      missing.
    * ONE_PRESSURE_MATCH: Pressure_Balls in one match only, and Ordinary_Balls of the same cell in
      another (and more anywhere): the SEs are undefined, though two matches contribute to
      Clutch_Level minus Usual_Baseline.
    * CHARGE_ONLY: 1-3 Deliveries of one cell and class in one match, a charge of the other class
      in the same match and cell (a charge-only row in a match with Deliveries) and a charge in a
      match without Deliveries (Half.NONE).
    * TIES: a player with 2 or more Pressure_Matches, an Ordinary_Ball and a Matched_Cell, and a
      twin with identical entries, so every estimate of the two is equal bit for bit and only the
      ``player_id`` tie-break separates them. ``twins[role]`` is (``player_id``, twin
      ``player_id``); the twin has the lower id about half the time.

    With ``charges`` the forced batters also hold traps: an ordinary charge in a cell with
    Pressure_Balls only (UNMATCHED_CELL, NO_MATCHED_CELL), a pressure charge in a cell with
    Ordinary_Balls only (NO_MATCHED_CELL) and a pressure charge in the match without
    Pressure_Balls (ONE_PRESSURE_MATCH). None of them makes a Matched_Cell or a Pressure_Match.
    ``charges=False`` gives no Dismissal_Charge at all.

    ``n_players``: an int of at least 1, a strategy of such ints, or None for 1-4. ``n_matches``:
    an int, a strategy of ints, or None for 1-6; at least 2 when ``force`` holds
    ONE_PRESSURE_MATCH, CHARGE_ONLY or TIES. ``roles``: Role names (None: both). Keep the counts
    small (with ``dense``, up to about 15 players and 10 matches), so that an example stays within
    Hypothesis' size limit. Shrinking moves towards one random player per Role with one ordinary
    Delivery of NBV 0 in cell 0 of the first match, no charge, every match an IPL match on
    2021-01-01 with weight 1, and small ids.

``match_multiplicities(n_matches, *, resample=False)``
    One whole-number multiplicity per match (int64, length ``n_matches``) for
    ``Aggregator.estimates(mult)``: mostly 1, often 0 or 2, sometimes 3 or 4. With
    ``resample=True``, a Bootstrap_Resample (Requirement 19, criterion 2): the counts sum to
    ``n_matches``, reached from all ones by moving copies between matches, so a match may be drawn
    0, 1, 2 or more times. Shrinks towards all ones, the multiplicities of the Main_Ranking.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from hypothesis import assume
from hypothesis import strategies as st
from hypothesis.errors import InvalidArgument
from hypothesis.strategies import SearchStrategy

from superr.folds import MIN_FOLDS, MIN_SANDBOX_FOLDS, OUTER, FitSpec, assign_folds, sandbox_plan
from superr.rating import (
    CONTRIBUTION_COLUMNS,
    LATE_PERIOD_START,
    MATCH_WEIGHT_COLUMNS,
    ROLE_CODES,
    CompClass,
    Contributions,
    Half,
    Period,
)
from superr.reconstruct import (
    ABANDONED,
    CLASS_KEYS,
    COMP_IPL,
    COMP_T20I,
    COMPETITION_TYPES,
    NO_RESULT,
    SUPER_OVER_WIN,
    TIE,
    UNPARSED,
    WIN_BY_RUNS,
    WIN_BY_WICKETS,
    DeliveryClass,
    ParsedResult,
)
from superr.settings import load_settings
from superr.value import BATTER_ROLE, ROLES

__all__ = [
    "ALL_MATCHED",
    "BALL_BY_BALL_COLUMNS",
    "CANDIDATE_ENCODINGS",
    "CELL_KINDS",
    "CELL_NAMES",
    "CHARGE_ONLY",
    "CHECKS",
    "COUNTED_WICKET_CODES",
    "CellTables",
    "CheckTables",
    "ContributionCase",
    "DISMISSAL_LABELS",
    "ENTRY_COLUMNS",
    "EXTRAS_TYPE_CLASSES",
    "FOLD_SEED",
    "FORCED_CASES",
    "FitCase",
    "FoldTables",
    "HALF_LIFE_YEARS",
    "KIND_FREE_HIT",
    "KIND_NON_FREE_HIT",
    "KIND_NO_BALL",
    "KIND_WIDE",
    "KNOCKOUT_WEIGHT",
    "LEAK_EARLY_STOPPING",
    "LEAK_TRAIN",
    "MATCH_COLUMNS",
    "MatchCase",
    "NO_MATCHED_CELL",
    "N_FOLDS",
    "ONE_PRESSURE_MATCH",
    "RETIRED_HURT",
    "RETIRED_HURT_CODES",
    "RETIRE_OUT",
    "RUNS_CHECKS",
    "RUN_OUT",
    "ROOT",
    "ResultCase",
    "SCORE_CHECK_COLUMNS",
    "SCORE_TOLERANCE",
    "Scorecard",
    "TEAMS",
    "TIES",
    "UNMATCHED_CELL",
    "cell_tables",
    "check_tables",
    "contribution_frame",
    "contributions",
    "fold_tables",
    "innings",
    "match_multiplicities",
    "result_strings",
    "score_checks_frame",
    "scorecard_match",
]

# ------------------------------------------------------------------------------- Config values

#: The workspace root, which holds config.py.
ROOT = Path(__file__).resolve().parent.parent
_CONFIG = load_settings(ROOT)

#: The Config class mapping ``data_rules.extras_type_classes``: class key -> extras_type texts.
EXTRAS_TYPE_CLASSES: dict[str, list[str]] = _CONFIG.get("data_rules.extras_type_classes")
#: The Config's candidate Extras_Encodings (``data_rules.candidate_encodings``), in Config order.
CANDIDATE_ENCODINGS: tuple[str, ...] = tuple(_CONFIG.get("data_rules.candidate_encodings"))
#: The Score_Tolerance D20 in runs (``design_choices.D20_score_tolerance``).
SCORE_TOLERANCE: int = _CONFIG.get("design_choices.D20_score_tolerance")

# --------------------------------------------------------------------------- Table schemas

_INT, _BOOL, _TEXT = "Int64", "boolean", "string"

#: Every ball_by_ball.csv column in header order, with the dtype the Data_Loader gives it.
BALL_BY_BALL_COLUMNS: dict[str, str] = {
    "match_id": _INT,
    "innings": _INT,
    "overid": _INT,
    "over_ball_num": _INT,
    "innings_ball_num": _INT,
    "batting_team_id": _INT,
    "bowling_team_id": _INT,
    "striker_id": _INT,
    "striker_name": _TEXT,
    "non_striker_id": _INT,
    "non_striker_name": _TEXT,
    "bowler_id": _INT,
    "bowler_name": _TEXT,
    "batting_type": _TEXT,
    "bowling_type": _TEXT,
    "batting_order": _INT,
    "runs": _INT,
    "is_four": _BOOL,
    "is_six": _BOOL,
    "is_wicket": _BOOL,
    "dismissal_type_id": _INT,
    "dismissal_type": _TEXT,
    "dismissed_player_id": _INT,
    "extras_type": _TEXT,
    "extras1_runs": _INT,
    "extras_2_runs": _INT,
    "pace_or_spin": _TEXT,
    "bowl_speed": _INT,
    "stroke": _TEXT,
    "delivery_style": _TEXT,
    "over_or_around": _TEXT,
    "on_or_off": _TEXT,
    "delivery_ist_time": _TEXT,
}

#: The matches.csv columns that score reconstruction reads, with their Data_Loader dtypes.
MATCH_COLUMNS: dict[str, str] = {
    "match_id": _INT,
    "first_inn_revised_overs": _INT,
    "second_inn_revised_overs": _INT,
    "revised_target": _INT,
    "win_team_id": _INT,
    "result": _TEXT,
}

#: Columns of the ``score_checks`` long table (design, Data Models).
SCORE_CHECK_COLUMNS: tuple[str, ...] = ("encoding", "match_id", "check", "applies", "passed", "runs_check_miss")
#: The Score_Checks of Requirement 5, criteria 16-20, in table order.
CHECKS: tuple[str, ...] = ("runs_margin", "chase", "wickets_margin", "tie", "over")
#: The checks that carry a Runs_Check_Miss (criteria 16, 17 and 19).
RUNS_CHECKS: tuple[str, ...] = ("runs_margin", "chase", "tie")

# ---------------------------------------------------------------------- Delivery vocabulary

_LEGAL, _WIDE, _NO_BALL = int(DeliveryClass.LEGAL), int(DeliveryClass.WIDE), int(DeliveryClass.NO_BALL)


def _class_lookup() -> dict[str, int]:
    found: dict[str, int] = {}
    for delivery_class, key in CLASS_KEYS.items():
        for value in EXTRAS_TYPE_CLASSES[key]:
            found.setdefault(value, int(delivery_class))
    return found


#: DeliveryClass code of each extras_type value, from the Config class mapping.
_CLASS_OF = _class_lookup()

# Legal_Ball texts that the Extras_Encodings score as a bye or leg-bye, or as a Penalty award
# (design DD-4, transcribed here rather than imported, so the strategies stay independent).
_BYE_TYPES = ("byes", "legbyes")
_PENALTY_TYPE = "Penalty"


def _role_of(value: str) -> str:
    """How the generator treats an extras_type value: plain, bye, penalty, wide or no_ball."""
    delivery_class = _CLASS_OF[value]
    if delivery_class == _WIDE:
        return "wide"
    if delivery_class == _NO_BALL:
        return "no_ball"
    if value in _BYE_TYPES:
        return "bye"
    if value == _PENALTY_TYPE:
        return "penalty"
    return "plain"


_ROLE_ORDER = ("plain", "bye", "penalty", "wide", "no_ball")
_ROLE_WEIGHTS = {"plain": 10, "bye": 2, "penalty": 1, "wide": 2, "no_ball": 1}


def _weighted(values: Iterable[str]) -> tuple[str, ...]:
    """Each value repeated by its role weight, plain values first, so shrinking reaches a plain ball."""
    ordered = sorted(values, key=lambda value: _ROLE_ORDER.index(_role_of(value)))
    return tuple(value for value in ordered for _ in range(_ROLE_WEIGHTS[_role_of(value)]))


_LEGAL_TYPES = _weighted(EXTRAS_TYPE_CLASSES["legal"])
_ILLEGAL_TYPES = _weighted(EXTRAS_TYPE_CLASSES["wide"] + EXTRAS_TYPE_CLASSES["no_ball"])
if not _LEGAL_TYPES or _role_of(_LEGAL_TYPES[0]) != "plain" or not _ILLEGAL_TYPES:
    raise RuntimeError(
        "tests/strategies.py needs a plain Legal_Ball value and at least one Wide or No_Ball value "
        f"in data_rules.extras_type_classes; found {EXTRAS_TYPE_CLASSES!r}"
    )
_PLAIN_TYPE = _LEGAL_TYPES[0]

#: extras1_runs code of each extras_type value, as observed in ball_by_ball.csv (one code per value).
_EXTRAS1_CODE = {"Overstep": 1, "wide": 2, "legbyes": 3, "byes": 4, "other": 5, "None": 6, "Penalty": 7}

#: Dismissal codes and labels as in ball_by_ball.csv, plus a Retired_Hurt code (7), which the
#: Auditor classifies as Retired_Hurt by its label (Requirement 4, criterion 10).
RUN_OUT = 4
RETIRE_OUT = 9
RETIRED_HURT = 7
DISMISSAL_LABELS: dict[int, str] = {
    1: "Bowled",
    2: "Caught",
    3: "LBW",
    4: "RunOut",
    5: "Stump",
    6: "Hit Wicket",
    9: "Retire Out",
    7: "Retired Hurt",
}
#: Counted_Wicket codes (glossary: 1-6 and Retire Out 9) and Retired_Hurt codes of the generator.
COUNTED_WICKET_CODES: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 9)
RETIRED_HURT_CODES: tuple[int, ...] = (RETIRED_HURT,)
_NO_BAT_RUNS_CODES = (1, 2, 3, 5, 6)  # bowled, caught, LBW, stumped, hit wicket: the ball scores no bat runs

# Per-Delivery choices, each listed so that shrinking moves to the first (simplest) entry.
_WICKET_CODES: dict[str, tuple[int | None, ...]] = {
    "plain": (None,) * 96 + (2, 2, 2, 1, 3, 4, 5, 6, 9, 9, 7, 7),
    "wide": (None,) * 20 + (4, 5),
    "no_ball": (None,) * 20 + (4,),
    "bye": (None,) * 20 + (4,),
    "penalty": (None,),
}
_BAT_RUNS = (0, 0, 0, 0, 1, 1, 1, 1, 2, 4, 4, 6, 3, 5)
_RUN_OUT_BAT_RUNS = (0, 1, 2)
_EXTRA_RUNS: dict[str, tuple[int, ...]] = {
    "plain": (0,) * 15 + (1,),
    "bye": (0,) * 7 + (1,),
    "penalty": (0,),
    "wide": (0,) * 6 + (1, 4),
    "no_ball": (0,) * 6 + (1, 4),
}
_ILLEGAL_BEFORE = (0,) * 10 + (1, 1, 2)  # Wides and No_Balls bowled before each Legal_Ball
_TRAILING = (0, 0, 0, 0, 1, 2)  # Wides and No_Balls after the last Legal_Ball of a partial last Over
_BOWLER_CHANGE_AT = (None,) * 9 + (1, 2, 3, 4, 5)  # mid-Over change before this Legal_Ball (0-based)
_NON_STRIKER_OUT = (False, False, True)
_SOMETIMES = (False, False, False, False, True)

# Bowling-team slots that bowl, with bowling_type, pace_or_spin and delivery_style.
_BOWLING_STYLE: dict[int, tuple[str, str, str]] = {
    6: ("Right-arm offbreak", "Spin", "Off Break"),
    7: ("Right-arm Fast", "Pace", "Seamed In"),
    8: ("Left-arm fast-medium", "Pace", "Inswing"),
    9: ("Right-arm medium-fast", "Pace", "Off Cutter"),
    10: ("Right-arm Leg-Break", "Spin", "Leg Break"),
    11: ("Slow left-arm orthodox", "Spin", "Orthodox"),
}
_BOWLER_SLOTS = tuple(_BOWLING_STYLE)
_STROKES = {0: "Defended", 1: "Worked", 2: "Flick", 3: "Cover Drive", 4: "Cover Drive", 5: "Square Cut", 6: "Pull shot"}
_FULL_INNINGS_BALLS = 120  # Scheduled_Balls with no revised-overs value (Requirement 5, criterion 6)
_MAX_OVERS = 20

#: (team_id, team_name) pairs as they appear in matches.csv; names hold no digit and no
#: Result_Parser keyword, so they never change how a result string parses.
TEAMS: tuple[tuple[int, str], ...] = (
    (1, "Chennai Super Kings"),
    (7, "Delhi Capitals"),
    (8, "Kolkata Knight Riders"),
    (18, "India"),
    (20, "Australia"),
    (22, "England"),
    (27, "Sri Lanka"),
    (28, "West Indies"),
    (341, "Pakistan A"),
    (3093, "Bangladesh A"),
    (3509, "Lucknow Super Giants"),
)


# ---------------------------------------------------------------------- Delivery generation

@dataclass(slots=True)
class _Ball:
    """One Delivery while an innings is generated; bat runs may still be adjusted."""

    extras_type: str
    role: str
    bat: int = 0
    extra: int = 0  # extras_2_runs
    code: int | None = None  # dismissal_type_id; None when no batter departs
    non_striker_out: bool = False
    bowler: int = 7  # bowling-team slot
    over: int = 0  # 0-based Over of the innings
    frozen: bool = False  # bat runs reserved for a later shift; run filling leaves them alone

    @property
    def cls(self) -> int:
        return _CLASS_OF[self.extras_type]

    @property
    def max_bat(self) -> int:
        """Largest bat runs this Delivery may score; 0 where bat runs are fixed at 0."""
        if self.role not in ("plain", "no_ball") or self.code in _NO_BAT_RUNS_CODES:
            return 0
        return 2 if self.code == RUN_OUT else 6


class _LineUp:
    """Who comes in when a batter departs: a waiting Retired_Hurt batter first, else the next one."""

    __slots__ = ("unused", "returning")

    def __init__(self, n_batters: int) -> None:
        self.unused = list(range(3, n_batters + 1))  # slots 1 and 2 open the innings
        self.returning: list[int] = []

    def depart(self, slot: int, code: int) -> int | None:
        """The incoming slot for a departure, or None when nobody is left (all out)."""
        if self.returning:
            incoming: int | None = self.returning.pop(0)
        elif self.unused:
            incoming = self.unused.pop(0)
        else:
            incoming = None
        if code in RETIRED_HURT_CODES:
            self.returning.append(slot)
        return incoming


def _draw_ball(draw: Any, value: str, *, over: int, bowler: int) -> _Ball:
    ball = _Ball(value, _role_of(value), over=over, bowler=bowler)
    ball.code = draw(st.sampled_from(_WICKET_CODES[ball.role]))
    if ball.code in (RUN_OUT, RETIRE_OUT, RETIRED_HURT):
        ball.non_striker_out = draw(st.sampled_from(_NON_STRIKER_OUT))
    if ball.max_bat:
        ball.bat = draw(st.sampled_from(_RUN_OUT_BAT_RUNS if ball.code == RUN_OUT else _BAT_RUNS))
    ball.extra = draw(st.sampled_from(_EXTRA_RUNS[ball.role]))
    return ball


def _draw_shape(
    draw: Any, n_overs: int, *, n_batters: int = 11, partial_last: bool = False, trailing: bool = False
) -> list[_Ball]:
    """The Deliveries of one innings of ``n_overs`` Overs, in bowling order.

    Every Over has 6 Legal_Balls, except that with ``partial_last`` the last Over has 1-6 and,
    with ``trailing``, may then end with Wides or No_Balls. Before each Legal_Ball come 0-2 Wides
    or No_Balls. The innings ends early at a departure with no batter left to come in.
    """
    lineup = _LineUp(n_batters)
    balls: list[_Ball] = []
    previous: int | None = None
    bowler = _BOWLER_SLOTS[1]
    for over in range(n_overs):
        last = over == n_overs - 1
        n_legal = draw(st.integers(1, 6)) if partial_last and last else 6
        bowler = draw(st.sampled_from([slot for slot in _BOWLER_SLOTS if slot != previous]))
        change_at = draw(st.sampled_from(_BOWLER_CHANGE_AT))
        n_trailing = draw(st.sampled_from(_TRAILING)) if trailing and last and n_legal < 6 else 0
        for legal_index in range(n_legal + 1):
            if legal_index == change_at and (legal_index < n_legal or n_trailing):
                bowler = draw(st.sampled_from([slot for slot in _BOWLER_SLOTS if slot != bowler]))
            if legal_index < n_legal:
                values = [draw(st.sampled_from(_ILLEGAL_TYPES)) for _ in range(draw(st.sampled_from(_ILLEGAL_BEFORE)))]
                values.append(draw(st.sampled_from(_LEGAL_TYPES)))
            else:
                values = [draw(st.sampled_from(_ILLEGAL_TYPES)) for _ in range(n_trailing)]
            for value in values:
                ball = _draw_ball(draw, value, over=over, bowler=bowler)
                balls.append(ball)
                if ball.code is not None and lineup.depart(0, ball.code) is None:
                    return balls  # all out
        previous = bowler
    return balls


def _is_dot(ball: _Ball) -> bool:
    """A plain Legal_Ball with no runs and no wicket: 0 team runs under every encoding."""
    return ball.role == "plain" and ball.bat == 0 and ball.extra == 0 and ball.code is None and not ball.frozen


def _make_bad_over(draw: Any, balls: list[_Ball]) -> None:
    """Give one Over a Legal_Ball count that fails the over check, without changing any total.

    A chosen Over gets dot balls inserted before its last Delivery until it holds 7 Legal_Balls,
    or, for an Over before the last, sometimes loses one dot ball and holds 5.
    """
    overs = sorted({ball.over for ball in balls})
    over = draw(st.sampled_from(overs))
    members = [i for i, ball in enumerate(balls) if ball.over == over]
    legal = [i for i in members if balls[i].cls == _LEGAL]
    if over != overs[-1] and len(legal) == 6 and draw(st.booleans()):
        dots = [i for i in legal if _is_dot(balls[i])]
        if dots:
            del balls[dots[0]]
            return
    anchor = members[-1]
    for _ in range(7 - len(legal)):
        balls.insert(anchor, _Ball(_PLAIN_TYPE, "plain", over=over, bowler=balls[anchor].bowler))


def _crossed(ball: _Ball) -> bool:
    """Whether the batters changed ends on a Delivery with no departure (odd runs run)."""
    if ball.role in ("plain", "no_ball"):
        run = 0 if ball.bat in (4, 6) else ball.bat
    elif ball.role == "bye":
        run = 1  # bye and leg-bye runs are not recorded (design F-4); one run is assumed
    elif ball.role == "wide":
        run = 0 if ball.extra == 4 else ball.extra
    else:
        run = 0
    return run % 2 == 1


def _player(team_id: int, slot: int) -> int:
    return team_id * 100 + slot


def _innings_frame(
    balls: Sequence[_Ball],
    *,
    match_id: int,
    innings_no: int,
    batting_team_id: int,
    bowling_team_id: int,
    n_batters: int = 11,
) -> pd.DataFrame:
    """The ball_by_ball.csv rows of one generated innings, with players, numbering and labels."""
    last_over = max((ball.over for ball in balls), default=0)
    over_end = {ball.over: i for i, ball in enumerate(balls) if ball.cls == _LEGAL}
    lineup = _LineUp(n_batters)
    striker, non_striker = 1, 2
    columns: dict[str, list[Any]] = {name: [] for name in BALL_BY_BALL_COLUMNS}
    overid, previous_over, ball_in_over = 0, None, 0
    for i, ball in enumerate(balls):
        if ball.over != previous_over:
            overid, previous_over, ball_in_over = overid + 1, ball.over, 0
        ball_in_over += 1
        bowling_type, pace_or_spin, style = _BOWLING_STYLE[ball.bowler]
        out = None if ball.code is None else (non_striker if ball.non_striker_out else striker)
        striker_id, non_striker_id = _player(batting_team_id, striker), _player(batting_team_id, non_striker)
        bowler_id = _player(bowling_team_id, ball.bowler)
        row = {
            "match_id": match_id,
            "innings": innings_no,
            "overid": overid,
            "over_ball_num": ball_in_over,
            "innings_ball_num": i + 1,
            "batting_team_id": batting_team_id,
            "bowling_team_id": bowling_team_id,
            "striker_id": striker_id,
            "striker_name": f"Player {striker_id}",
            "non_striker_id": non_striker_id,
            "non_striker_name": f"Player {non_striker_id}",
            "bowler_id": bowler_id,
            "bowler_name": f"Player {bowler_id}",
            "batting_type": "LHB" if striker % 3 == 0 else "RHB",
            "bowling_type": bowling_type,
            "batting_order": striker,
            "runs": ball.bat,
            "is_four": ball.bat == 4,
            "is_six": ball.bat == 6,
            "is_wicket": ball.code is not None,
            "dismissal_type_id": ball.code,
            "dismissal_type": None if ball.code is None else DISMISSAL_LABELS[ball.code],
            "dismissed_player_id": None if out is None else _player(batting_team_id, out),
            "extras_type": ball.extras_type,
            "extras1_runs": _EXTRAS1_CODE.get(ball.extras_type),
            "extras_2_runs": ball.extra,
            "pace_or_spin": pace_or_spin,
            "bowl_speed": 0 if pace_or_spin == "Spin" else 125 + (7 * i) % 21,
            "stroke": _STROKES[ball.bat],
            "delivery_style": style,
            "over_or_around": "Over" if ball.bowler % 2 else "Around",
            "on_or_off": "Off-side" if i % 2 == 0 else "On-side",
            "delivery_ist_time": "2025-05-03 00:00:00 UTC",
        }
        for name, value in row.items():
            columns[name].append(value)
        if out is not None:
            incoming = lineup.depart(out, ball.code)
            if incoming is None:
                if i != len(balls) - 1:
                    raise AssertionError("strategy bug: Deliveries follow an all-out departure")
            elif out == striker:
                striker = incoming
            else:
                non_striker = incoming
        elif _crossed(ball):
            striker, non_striker = non_striker, striker
        if over_end.get(ball.over) == i and ball.over != last_over:
            striker, non_striker = non_striker, striker
    return pd.DataFrame({name: pd.array(columns[name], dtype=dtype) for name, dtype in BALL_BY_BALL_COLUMNS.items()})


def _check_positive(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise InvalidArgument(f"{name} must be a positive int; got {value!r}")


# ----------------------------------------------------------------------------------- innings

def innings(
    max_overs: int = 4,
    *,
    match_id: int = 1,
    innings_no: int = 1,
    batting_team_id: int = 1,
    bowling_team_id: int = 2,
    bad_over: bool = False,
) -> SearchStrategy[pd.DataFrame]:
    """One innings of Deliveries with every ball_by_ball.csv column; see the module docstring.

    ``max_overs`` (1-20) bounds the number of Overs; a Super_Over_Innings (``innings_no`` 3 or 4)
    always has one Over. With ``bad_over`` one Over fails the over check.
    """
    for name, value in (("max_overs", max_overs), ("match_id", match_id), ("batting_team_id", batting_team_id),
                        ("bowling_team_id", bowling_team_id)):
        _check_positive(name, value)
    if max_overs > _MAX_OVERS:
        raise InvalidArgument(f"max_overs must be at most {_MAX_OVERS}; got {max_overs}")
    if innings_no not in (1, 2, 3, 4):
        raise InvalidArgument(f"innings_no must be 1, 2, 3 or 4; got {innings_no!r}")
    if batting_team_id == bowling_team_id:
        raise InvalidArgument("batting_team_id and bowling_team_id must differ")
    return _innings(max_overs, match_id, innings_no, batting_team_id, bowling_team_id, bool(bad_over))


@st.composite
def _innings(
    draw: Any, max_overs: int, match_id: int, innings_no: int, batting_team_id: int, bowling_team_id: int,
    bad_over: bool,
) -> pd.DataFrame:
    super_over = innings_no >= 3
    n_batters = 3 if super_over else 11
    n_overs = 1 if super_over else draw(st.integers(1, max_overs))
    balls = _draw_shape(draw, n_overs, n_batters=n_batters, partial_last=True, trailing=True)
    if bad_over:
        _make_bad_over(draw, balls)
    return _innings_frame(
        balls,
        match_id=match_id,
        innings_no=innings_no,
        batting_team_id=batting_team_id,
        bowling_team_id=bowling_team_id,
        n_batters=n_batters,
    )


# ---------------------------------------------------------------------------- result strings

class ResultCase(NamedTuple):
    """A generated ``result`` string and the ParsedResult that the Result_Parser must return for it."""

    text: str
    expected: ParsedResult


# Templates of the result grammar. A space is a gap of one or more spaces and "|" a gap of zero or
# more; {W} and {L} are team names, {N} a margin, {WON}, {RUNS} and {WKTS} drawn wordings.
_TEMPLATES: dict[str, tuple[str, ...]] = {
    WIN_BY_RUNS: ("{W} {WON} {N}|{RUNS}",),
    WIN_BY_WICKETS: ("{W} {WON} {N}|{WKTS}",),
    SUPER_OVER_WIN: (
        "{W} won the super over",
        "{W} Won The Super Over",
        "{W} Won Super Over By {N}|{RUNS}",
        "{W} Tied with {L} ({W} Won the Super Over by {N}|{WKTS})",
        "{W} Tied with {L} ({W} won the Super Over)",
        "{W} won the Super Over|(Match Tied)",
        "{W} Won by Super over (Match Tied)",
        "{W} Won the one over Eliminator",
    ),
    TIE: ("Match Tied", "{W} tied with {L}", "Tied"),
    ABANDONED: ("Match Abandoned", "Match abandoned without a ball bowled", "Match Abandoned|(No Result)"),
    NO_RESULT: ("No Result", "No result (rain)", "Match ended in no result"),
    UNPARSED: (
        "{W} won by an innings",
        "Match drawn",
        "{W} won by {N}",
        "{W} won by forfeit",
        "Result not available",
        "{W} won on boundary count",
        "{W} won by five wickets",
        "{W} won",
        "",
    ),
}
# Each result type, with the two margin types listed twice so that they get more of the draws.
_RESULT_TYPES = (WIN_BY_RUNS, WIN_BY_WICKETS, SUPER_OVER_WIN, TIE, ABANDONED, NO_RESULT, UNPARSED, WIN_BY_RUNS, WIN_BY_WICKETS)
_WON_WORDS = ("won by", "Won by", "Won By", "won", "Won")
_RUNS_WORDS = ("runs", "Runs", "run", "Run")
_WICKET_WORDS = ("wickets", "Wickets", "wicket", "Wicket", "wkts", "Wkts", "wkt")
_DLS_MARKERS = ("|(DLS method)", " (DLS Method)", "|(D/L method)", " (D/L Method)", " (DLS)", " D/L", "|(DLS)")
_EXTRA_TEAM_NAMES = ("Rajasthan Royals", "Srilanka", "Mumbai Indians", "New Zealand", "South Africa")
_RESULT_TEAM_NAMES = tuple(name for _, name in TEAMS) + _EXTRA_TEAM_NAMES
_CASE_STYLES = (str, str.lower, str.upper, str.title, str.swapcase)
_GAP_EXTRA = (0, 0, 1, 2)  # spaces added to a required one-space gap
_OPTIONAL_GAP = (1, 0, 2)  # spaces in a gap of zero or more
_EDGE_SPACES = ("", "", " ", "  ")
_MARGINS = {WIN_BY_RUNS: st.integers(1, 250), WIN_BY_WICKETS: st.integers(1, 10)}
_OTHER_MARGIN = st.integers(1, 30)  # printed by some super-over and unparsed phrasings, never parsed


def _result_text(draw: Any, result_type: str, winner: str, loser: str, margin: int, dls: bool) -> str:
    """A result string of ``result_type`` from the grammar, with case and spacing noise."""
    template = draw(st.sampled_from(_TEMPLATES[result_type]))
    if dls:
        template += draw(st.sampled_from(_DLS_MARKERS))
    words: dict[str, str] = {"{W}": winner, "{L}": loser, "{N}": str(margin)}
    for key, options in (("{WON}", _WON_WORDS), ("{RUNS}", _RUNS_WORDS), ("{WKTS}", _WICKET_WORDS)):
        if key in template:
            words[key] = draw(st.sampled_from(options))
    for key, word in words.items():
        template = template.replace(key, word)
    tokens, gaps, current = [], [], ""
    for character in template:
        if character in " |":
            tokens.append(current)
            gaps.append(1 + draw(st.sampled_from(_GAP_EXTRA)) if character == " " else draw(st.sampled_from(_OPTIONAL_GAP)))
            current = ""
        else:
            current += character
    tokens.append(current)
    noisy = [draw(st.sampled_from(_CASE_STYLES))(token) for token in tokens]
    body = noisy[0] + "".join(" " * gap + token for gap, token in zip(gaps, noisy[1:]))
    return draw(st.sampled_from(_EDGE_SPACES)) + body + draw(st.sampled_from(_EDGE_SPACES))


def result_strings() -> SearchStrategy[ResultCase]:
    """A result string from the Result_Parser grammar and its expected ParsedResult; see the module docstring."""
    return _result_strings()


@st.composite
def _result_strings(draw: Any) -> ResultCase:
    result_type = draw(st.sampled_from(_RESULT_TYPES))
    winner, loser = draw(st.lists(st.sampled_from(_RESULT_TEAM_NAMES), min_size=2, max_size=2, unique=True))
    margin = draw(_MARGINS.get(result_type, _OTHER_MARGIN))
    dls = draw(st.sampled_from((False, False, False, True)))
    text = _result_text(draw, result_type, winner, loser, margin, dls)
    parsed_margin = margin if result_type in _MARGINS else None
    return ResultCase(text, ParsedResult(result_type, parsed_margin, dls))


# ------------------------------------------------------------------------------ check tables

class CheckTables(NamedTuple):
    """Score_Check results of several matches, the selected Extras_Encoding and a Score_Tolerance t."""

    checks: pd.DataFrame
    selected: str
    tolerance: int


def score_checks_frame(rows: Iterable[Sequence[Any] | Mapping[str, Any]]) -> pd.DataFrame:
    """A ``score_checks`` table with the schema dtypes, from rows in SCORE_CHECK_COLUMNS order or mappings.

    ``encoding`` and ``check`` are ``str``, ``match_id`` int32, ``applies`` bool, ``passed``
    nullable boolean and ``runs_check_miss`` nullable Int32; None in the last two is missing.
    For hand-written cases, for example
    ``score_checks_frame([("F30", 1, "tie", True, False, SCORE_TOLERANCE)])``.
    """
    records = [dict(row) if isinstance(row, Mapping) else dict(zip(SCORE_CHECK_COLUMNS, row, strict=True)) for row in rows]
    values = {name: [record[name] for record in records] for name in SCORE_CHECK_COLUMNS}
    return pd.DataFrame(
        {
            "encoding": pd.array(values["encoding"], dtype="str"),
            "match_id": np.array(values["match_id"], dtype=np.int32),
            "check": pd.array(values["check"], dtype="str"),
            "applies": np.array(values["applies"], dtype=bool),
            "passed": pd.array(values["passed"], dtype="boolean"),
            "runs_check_miss": pd.array(values["runs_check_miss"], dtype="Int32"),
        }
    )


# Applicable checks by result: win by runs, win by wickets, tie or super-over win, abandoned / no
# result / unparsed with Deliveries, and a match with no Delivery in innings 1 or 2.
_APPLICABLE_SETS: tuple[tuple[str, ...], ...] = (
    ("runs_margin", "over"),
    ("chase", "wickets_margin", "over"),
    ("tie", "over"),
    ("over",),
    (),
)
_TOLERANCES = st.one_of(st.integers(0, 3), st.integers(max(0, SCORE_TOLERANCE - 1), SCORE_TOLERANCE + 1))
_MATCH_IDS = st.integers(10_000, 99_999)
_MOSTLY_PASS = (True, True, False)


def check_tables(
    *,
    tolerance: int | SearchStrategy[int] | None = None,
    encodings: Sequence[str] | None = None,
    max_matches: int = 6,
    applicable_only: bool = False,
) -> SearchStrategy[CheckTables]:
    """Per-match Score_Check rows, a selected encoding and a tolerance t; see the module docstring."""
    _check_positive("max_matches", max_matches)
    if isinstance(tolerance, int) and (isinstance(tolerance, bool) or tolerance < 0):
        raise InvalidArgument(f"tolerance must be a non-negative int or a strategy; got {tolerance!r}")
    if encodings is not None and (not encodings or len(set(encodings)) != len(encodings)):
        raise InvalidArgument(f"encodings must be a non-empty sequence of distinct names; got {encodings!r}")
    return _check_tables(tolerance, None if encodings is None else tuple(encodings), max_matches, bool(applicable_only))


@st.composite
def _check_tables(
    draw: Any, tolerance: int | SearchStrategy[int] | None, encodings: tuple[str, ...] | None, max_matches: int,
    applicable_only: bool,
) -> CheckTables:
    if tolerance is None:
        t = draw(_TOLERANCES)
    elif isinstance(tolerance, int):
        t = tolerance
    else:
        t = draw(tolerance)
    if encodings is None:
        chosen = draw(st.lists(st.sampled_from(CANDIDATE_ENCODINGS), min_size=1, unique=True))
        encodings = tuple(name for name in CANDIDATE_ENCODINGS if name in chosen)
    selected = draw(st.sampled_from(encodings))
    match_ids = sorted(draw(st.lists(_MATCH_IDS, min_size=1, max_size=max_matches, unique=True)))
    misses = st.one_of(st.sampled_from((0, t, t + 1)), st.integers(0, 2 * t + 3))
    results: dict[tuple[str, int, str], tuple[bool, bool | None, int | None]] = {}
    for match_id in match_ids:
        applicable = set(draw(st.one_of(st.sampled_from(_APPLICABLE_SETS), st.lists(st.sampled_from(CHECKS), unique=True))))
        shared = {check: draw(st.sampled_from(_MOSTLY_PASS)) for check in ("wickets_margin", "over") if check in applicable}
        for name in encodings:
            for check in CHECKS:
                if check not in applicable:
                    results[name, match_id, check] = (False, None, None)
                elif check in RUNS_CHECKS:
                    miss = draw(misses)
                    results[name, match_id, check] = (True, miss == 0, miss)
                else:
                    results[name, match_id, check] = (True, shared[check], None)
    rows = [
        (name, match_id, check, *results[name, match_id, check])
        for name in encodings
        for match_id in match_ids
        for check in CHECKS
        if results[name, match_id, check][0] or not applicable_only
    ]
    return CheckTables(score_checks_frame(rows), selected, t)


# --------------------------------------------------------------------------- scorecard matches

class Scorecard(NamedTuple):
    """The scorecard of a generated match under its Extras_Encoding.

    ``totals``, ``wickets`` and ``final_ball_runs`` describe the Deliveries as delivered, after
    any shift; ``target`` is the Target they give (``revised_target``, else innings-1 total + 1).
    ``result_type`` and ``margin`` are what the ``result`` string records, which the shift does
    not change: the shift moves the total of innings ``shift_innings`` by ``shift`` runs.
    """

    sched_balls: tuple[int, int]
    totals: tuple[int, int]
    wickets: tuple[int, int]
    final_ball_runs: int
    target: int
    result_type: str
    margin: int | None
    shift_innings: int | None
    shift: int


class MatchCase(NamedTuple):
    """A match generated from a scorecard: Deliveries, its matches row and the expected checks."""

    balls: pd.DataFrame
    matches: pd.DataFrame
    encoding: str
    scorecard: Scorecard
    expected: pd.DataFrame


# Team runs per Delivery under each candidate Extras_Encoding, transcribed from the design's DD-4
# table: (a bye or leg-bye scores 1 + e instead of b + e, a No_Ball scores 1 + b + e instead of
# 1 + b, a Penalty Delivery scores 5 + b + e instead of b + e). A Wide always scores 1 + e.
_ENCODING_RULES: dict[str, tuple[bool, bool, bool]] = {
    "F30": (False, True, False),
    "byes_one": (True, True, False),
    "byes_one_nb_plain": (True, False, False),
    "byes_one_penalty5": (True, True, True),
}
_unknown_encodings = [name for name in CANDIDATE_ENCODINGS if name not in _ENCODING_RULES]
if _unknown_encodings:
    raise RuntimeError(
        f"tests/strategies.py has no team-runs rule for the candidate encoding(s) {_unknown_encodings}; "
        "add them to _ENCODING_RULES from the design's DD-4 table"
    )


def _team_runs(ball: _Ball, encoding: str) -> int:
    byes_one, no_ball_extras, penalty_five = _ENCODING_RULES[encoding]
    b, e = ball.bat, ball.extra
    if ball.role == "wide":
        return 1 + e
    if ball.role == "no_ball":
        return 1 + b + (e if no_ball_extras else 0)
    if ball.role == "bye" and byes_one:
        return 1 + e
    if ball.role == "penalty" and penalty_five:
        return 5 + b + e
    return b + e


def _total(balls: Iterable[_Ball], encoding: str) -> int:
    return sum(_team_runs(ball, encoding) for ball in balls)


def _bounds(ball: _Ball, encoding: str) -> tuple[int, int]:
    """Least and most team runs the Delivery can score once its bat runs are adjusted.

    Bat runs enter the team runs with coefficient 1 wherever they can be non-zero (plain
    Legal_Balls and No_Balls), so the team runs move one for one with them.
    """
    runs = _team_runs(ball, encoding)
    if ball.frozen or not ball.max_bat:
        return runs, runs
    return runs - ball.bat, runs - ball.bat + ball.max_bat


def _total_range(balls: Iterable[_Ball], encoding: str) -> tuple[int, int]:
    low = high = 0
    for ball in balls:
        ball_low, ball_high = _bounds(ball, encoding)
        low, high = low + ball_low, high + ball_high
    return low, high


def _fill(balls: Sequence[_Ball], total: int, encoding: str) -> bool:
    """Adjust bat runs of the open Deliveries, from their drawn values, so the team runs sum to ``total``."""
    diff = total - _total(balls, encoding)
    for ball in balls:
        if diff == 0:
            break
        if ball.frozen or not ball.max_bat:
            continue
        step = min(diff, ball.max_bat - ball.bat) if diff > 0 else max(diff, -ball.bat)
        ball.bat += step
        diff -= step
    return diff == 0


def _first_open(balls: Sequence[_Ball]) -> int:
    """Position after the last reserved Delivery (0 when none is reserved)."""
    return max((i + 1 for i, ball in enumerate(balls) if ball.frozen), default=0)


def _reserve(balls: Sequence[_Ball], count: int, delta: int) -> bool:
    """Reserve ``count`` plain wicket-free Legal_Balls before the innings' last Delivery for a shift.

    They are set to 0 bat runs (``delta`` > 0) or 6 (``delta`` < 0) and frozen, so that after the
    result is recorded ``_apply_shift`` can move the total by ``delta`` within 0-6 bat runs each.
    """
    candidates = [ball for ball in balls[:-1] if ball.role == "plain" and ball.code is None][:count]
    if len(candidates) < count:
        return False
    for ball in candidates:
        ball.bat, ball.frozen = (0 if delta > 0 else 6), True
    return True


def _apply_shift(balls: Sequence[_Ball], delta: int) -> None:
    remaining = abs(delta)
    for ball in balls:
        if ball.frozen and remaining:
            step = min(remaining, 6)
            ball.bat = step if delta > 0 else 6 - step
            remaining -= step
    if remaining:
        raise AssertionError("strategy bug: the reserved Deliveries cannot carry the shift")


def _chase_length(balls: Sequence[_Ball], total: int, encoding: str) -> int | None:
    """How many Deliveries come before the winning one, so that they can score exactly ``total``.

    The winning Delivery replaces the Delivery at that position. The first position where the
    drawn runs already reach ``total`` is preferred, else the last feasible one; None if none is.
    """
    start = _first_open(balls)
    low = high = natural = 0
    fallback = None
    for length, ball in enumerate(balls):
        if length >= start and low <= total <= high:
            if natural >= total:
                return length
            fallback = length
        ball_low, ball_high = _bounds(ball, encoding)
        low, high, natural = low + ball_low, high + ball_high, natural + _team_runs(ball, encoding)
    return fallback


_WINNING_TYPES = (_PLAIN_TYPE,) * 6 + tuple(EXTRAS_TYPE_CLASSES["wide"]) + tuple(EXTRAS_TYPE_CLASSES["no_ball"])
_WINNING_BAT_RUNS = (1, 1, 2, 4, 4, 6, 3)


def _draw_winning_ball(draw: Any) -> _Ball:
    """The Delivery that completes a chase: no wicket and at least 1 team run under every encoding."""
    value = draw(st.sampled_from(_WINNING_TYPES))
    ball = _Ball(value, _role_of(value))
    if ball.role == "plain":
        ball.bat = draw(st.sampled_from(_WINNING_BAT_RUNS))
    else:
        if ball.role == "no_ball":
            ball.bat = draw(st.sampled_from(_BAT_RUNS))
        ball.extra = draw(st.sampled_from(_EXTRA_RUNS[ball.role]))
    return ball


def _counted_wickets(balls: Iterable[_Ball]) -> int:
    return sum(ball.code in COUNTED_WICKET_CODES for ball in balls)


def _over_check(balls: Sequence[_Ball]) -> bool:
    """Requirement 5, criterion 20: 6 Legal_Balls in every Over but the highest, at most 6 in it."""
    legal: dict[int, int] = {}
    for ball in balls:
        legal[ball.over] = legal.get(ball.over, 0) + (ball.cls == _LEGAL)
    overs = sorted(legal)
    return all(legal[over] == 6 for over in overs[:-1]) and (not overs or legal[overs[-1]] <= 6)


def _chase_mode(side: int | None, delta: int, revised: bool) -> str:
    """Which chase slack the shift needs at zero, so that the moved total gives a miss of |delta|."""
    if not delta or side is None or (side == 1 and revised):
        return "natural"
    moves_score_up = delta > 0 if side == 2 else delta < 0  # the second-innings total against the Target
    return "excess" if moves_score_up else "shortfall"


_NO_LIMIT = 10**6
_RESULT_KINDS = (WIN_BY_RUNS, WIN_BY_WICKETS, TIE, WIN_BY_RUNS, WIN_BY_WICKETS, SUPER_OVER_WIN)
_SCORECARD_TYPES = (WIN_BY_RUNS, WIN_BY_WICKETS, TIE, SUPER_OVER_WIN)


def _target_range(kind: str, shape: Sequence[_Ball], encoding: str, slack: int) -> tuple[int, int]:
    """Targets for which the second innings can be filled to give result ``kind``."""
    low, high = _total_range(shape, encoding)
    if kind == WIN_BY_RUNS:  # total at most Target - 2, so the margin is at least 1
        return low + 2, _NO_LIMIT
    if kind != WIN_BY_WICKETS:  # tie or super-over win: total exactly Target - 1
        return low + 1, high + 1
    start = _first_open(shape)  # the winning Delivery comes after every reserved one
    if start >= len(shape):
        return 1, 0
    return 1 + slack + _total_range(shape[:start], encoding)[0], 1 + slack + _total_range(shape[:-1], encoding)[1]


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(f"strategy bug: {message}")


def _expected_checks(card: Scorecard, encoding: str, match_id: int, over_ok: bool) -> pd.DataFrame:
    """The five Score_Checks of the match: a direct transcription of Requirement 5, criteria 16-20."""
    total, target, final, wickets = card.totals[1], card.target, card.final_ball_runs, card.wickets[1]
    kind, margin = card.result_type, card.margin
    rows = []

    def add(check: str, applies: bool, passed: bool | None, miss: int | None = None) -> None:
        rows.append((encoding, match_id, check, applies, passed if applies else None, miss if applies else None))

    by_runs = kind == WIN_BY_RUNS  # the generated winner of a runs win always batted first
    miss = abs(((target - 1) - total) - margin) if by_runs else None
    add("runs_margin", by_runs, miss == 0, miss)
    by_wickets = kind == WIN_BY_WICKETS
    miss = max(0, target - total, (total - final) - (target - 1)) if by_wickets else None
    add("chase", by_wickets, miss == 0, miss)
    add("wickets_margin", by_wickets, by_wickets and margin == 10 - wickets)
    tied = kind in (TIE, SUPER_OVER_WIN)
    miss = abs(total - (target - 1)) if tied else None
    add("tie", tied, miss == 0, miss)
    add("over", True, over_ok)
    return score_checks_frame(rows)


def scorecard_match(
    *,
    encoding: str | None = None,
    shift: int | SearchStrategy[int] | None = None,
    shift_innings: int | None = None,
    result_type: str | None = None,
    bad_over: bool = False,
    max_overs: int = 3,
    match_id: int | None = None,
) -> SearchStrategy[MatchCase]:
    """A match built from a scorecard under one Extras_Encoding; see the module docstring.

    ``max_overs`` (1-20) bounds the Overs of each innings. ``match_id`` is drawn from
    10000-99999 when None; to draw several matches with distinct ids, use for example
    ``st.lists(scorecard_match(), max_size=3, unique_by=lambda case: int(case.matches["match_id"].iloc[0]))``.
    """
    if encoding is not None and encoding not in _ENCODING_RULES:
        raise InvalidArgument(f"encoding must be one of {sorted(_ENCODING_RULES)}; got {encoding!r}")
    if shift is not None and not isinstance(shift, SearchStrategy) and (isinstance(shift, bool) or not isinstance(shift, int)):
        raise InvalidArgument(f"shift must be None, an int or a strategy of ints; got {shift!r}")
    if shift_innings not in (None, 1, 2):
        raise InvalidArgument(f"shift_innings must be None, 1 or 2; got {shift_innings!r}")
    if result_type not in (None, *_SCORECARD_TYPES):
        raise InvalidArgument(f"result_type must be None or one of {_SCORECARD_TYPES}; got {result_type!r}")
    _check_positive("max_overs", max_overs)
    if max_overs > _MAX_OVERS:
        raise InvalidArgument(f"max_overs must be at most {_MAX_OVERS}; got {max_overs}")
    if match_id is not None:
        _check_positive("match_id", match_id)
    return _scorecard_match(encoding, shift, shift_innings, result_type, bool(bad_over), max_overs, match_id)


@st.composite
def _scorecard_match(
    draw: Any,
    encoding: str | None,
    shift: int | SearchStrategy[int] | None,
    shift_innings: int | None,
    result_type: str | None,
    bad_over: bool,
    max_overs: int,
    match_id: int | None,
) -> MatchCase:
    enc = encoding if encoding is not None else draw(st.sampled_from(CANDIDATE_ENCODINGS))
    mid = match_id if match_id is not None else draw(_MATCH_IDS)
    first, second = draw(st.lists(st.sampled_from(TEAMS), min_size=2, max_size=2, unique=True))
    side: int | None = None
    delta = 0
    if shift is not None:
        delta = shift if isinstance(shift, int) else draw(shift)
        side = shift_innings if shift_innings is not None else draw(st.sampled_from((1, 2)))
    reserve = -(-abs(delta) // 6)  # Deliveries that carry the shift, at most 6 bat runs each

    # Scheduled Overs and the drawn Deliveries of both innings (the second may be cut by a chase).
    least = min(max_overs, 1 + reserve // 3) if reserve else 1
    overs1 = draw(st.integers(least, max_overs))
    least2 = least if side == 2 else 1
    overs2 = overs1
    if overs1 > least2 and draw(st.sampled_from(_SOMETIMES)):
        overs2 = draw(st.integers(least2, overs1 - 1))
    revised = draw(st.sampled_from(_SOMETIMES))
    innings1 = _draw_shape(draw, overs1)
    shape2 = _draw_shape(draw, overs2)
    if reserve:
        assume(_reserve(innings1 if side == 1 else shape2, reserve, delta))
    mode = _chase_mode(side, delta, revised)

    # The result: the drawn kind if the drawn innings can give it, else the next one that can.
    first_kind = result_type if result_type is not None else draw(st.sampled_from(_RESULT_KINDS))
    kinds = [first_kind] if result_type is not None else [first_kind] + [
        kind for kind in (WIN_BY_RUNS, WIN_BY_WICKETS, TIE) if kind != first_kind
    ]
    low1, high1 = _total_range(innings1, enc)
    for kind in kinds:
        winning, slack = None, 0
        if kind == WIN_BY_WICKETS:
            winning = _draw_winning_ball(draw)
            final_runs = _team_runs(winning, enc)
            if mode == "excess":
                slack = 0  # the winning Delivery comes with 1 run needed
            elif mode == "shortfall":
                slack = final_runs - 1  # the winning Delivery reaches the Target exactly
            else:
                slack = draw(st.integers(0, final_runs - 1))
        low_t, high_t = _target_range(kind, shape2, enc, slack)
        if revised:
            low_t = max(low_t, 1)
        else:
            low_t, high_t = max(low_t, low1 + 1), min(high_t, high1 + 1)
        if low_t <= high_t:
            break
    else:
        assume(False)

    # First innings and Target.
    if revised:
        target = _clamp(_total(innings1, enc) + 1 + draw(st.integers(-15, 5)), low_t, high_t)
    else:
        target = _clamp(_total(innings1, enc) + 1, low_t, high_t)
        _require(_fill(innings1, target - 1, enc), "innings 1 cannot reach its total")

    # Second innings, filled to the result.
    if kind == WIN_BY_WICKETS:
        prefix_total = target - 1 - slack
        length = _chase_length(shape2, prefix_total, enc)
        assume(length is not None)
        prefix = shape2[:length]
        _require(_fill(prefix, prefix_total, enc), "the chase cannot reach its pre-final score")
        winning.over, winning.bowler = shape2[length].over, shape2[length].bowler
        innings2 = prefix + [winning]
        margin: int | None = 10 - _counted_wickets(innings2)
    else:
        innings2 = shape2
        if kind == WIN_BY_RUNS:
            low2, high2 = _total_range(innings2, enc)
            total2 = _clamp(_total(innings2, enc), low2, min(high2, target - 2))
        else:
            total2 = target - 1
        _require(_fill(innings2, total2, enc), "innings 2 cannot reach its total")
        margin = target - 1 - total2 if kind == WIN_BY_RUNS else None
    base = (_total(innings1, enc), _total(innings2, enc))

    # The shift (after the result is recorded) and the bad Over.
    if reserve:
        _apply_shift(innings1 if side == 1 else innings2, delta)
    if bad_over:
        _make_bad_over(draw, innings1 if draw(st.sampled_from((1, 2))) == 1 else innings2)
    totals = (_total(innings1, enc), _total(innings2, enc))
    _require(totals == tuple(b + (delta if side == i else 0) for i, b in enumerate(base, start=1)),
             "the shift did not move exactly one total by the shift")

    # Super_Over_Innings, winner and the matches row.
    super_overs = []
    if kind == SUPER_OVER_WIN and draw(st.sampled_from((True, True, True, False))):
        super_overs = [_draw_shape(draw, 1, n_batters=3), _draw_shape(draw, 1, n_batters=3)]
    if kind == WIN_BY_RUNS:
        winner, loser = first, second
    elif kind == WIN_BY_WICKETS:
        winner, loser = second, first
    elif kind == SUPER_OVER_WIN and super_overs:  # innings 3: the side that batted second
        winner, loser = (second, first) if _total(super_overs[0], enc) > _total(super_overs[1], enc) else (first, second)
    else:
        winner, loser = draw(st.sampled_from(((first, second), (second, first))))
    dls = revised and draw(st.sampled_from((True, True, True, False)))
    text_margin = margin if margin is not None else draw(_OTHER_MARGIN)
    result = _result_text(draw, kind, winner[1], loser[1], text_margin, dls)

    sched = (6 * overs1, 6 * overs2)
    first_revised = None if sched[0] == _FULL_INNINGS_BALLS else sched[0]
    if overs2 != overs1:
        second_revised = sched[1]
    elif sched[1] != _FULL_INNINGS_BALLS and draw(st.booleans()):
        second_revised = sched[1]  # stated; otherwise blank and taken from first_inn_revised_overs
    else:
        second_revised = None
    row = {
        "match_id": mid,
        "first_inn_revised_overs": first_revised,
        "second_inn_revised_overs": second_revised,
        "revised_target": target if revised else None,
        "win_team_id": None if kind == TIE else winner[0],
        "result": result,
    }
    matches = pd.DataFrame({name: pd.array([row[name]], dtype=dtype) for name, dtype in MATCH_COLUMNS.items()})

    frames = [
        _innings_frame(innings1, match_id=mid, innings_no=1, batting_team_id=first[0], bowling_team_id=second[0]),
        _innings_frame(innings2, match_id=mid, innings_no=2, batting_team_id=second[0], bowling_team_id=first[0]),
    ]
    for innings_no, balls, batting, bowling in zip((3, 4), super_overs, (second, first), (first, second)):
        frames.append(_innings_frame(balls, match_id=mid, innings_no=innings_no, batting_team_id=batting[0],
                                     bowling_team_id=bowling[0], n_batters=3))

    card = Scorecard(
        sched_balls=sched,
        totals=totals,
        wickets=(_counted_wickets(innings1), _counted_wickets(innings2)),
        final_ball_runs=_team_runs(innings2[-1], enc),
        target=target if revised else totals[0] + 1,
        result_type=kind,
        margin=margin,
        shift_innings=side,
        shift=delta,
    )
    over_ok = _over_check(innings1) and _over_check(innings2)
    _require(over_ok != bad_over, "the over check does not match bad_over")
    expected = _expected_checks(card, enc, mid, over_ok)
    dependent = side == 2 or (side == 1 and not revised)
    runs_rows = expected[expected["applies"] & expected["check"].isin(RUNS_CHECKS)]
    _require(runs_rows["runs_check_miss"].tolist() == [abs(delta) if dependent else 0],
             "the shifted runs check does not miss by exactly |shift|")
    return MatchCase(pd.concat(frames, ignore_index=True), matches, enc, card, expected)


# ------------------------------------------------------------------------------- fold tables

#: The number of Match_Grouped_Folds D15 (``design_choices.D15_n_folds``): the fold count that
#: fold_tables draws most often.
N_FOLDS: int = _CONFIG.get("design_choices.D15_n_folds")
#: The Config fold seed (``seeds.folds``), one of the seeds that fold_tables draws.
FOLD_SEED: int = _CONFIG.get("seeds.folds")

#: FitCase.leak of a fit whose Training_Rows include matches of an excluded fold, and of a fit
#: whose early-stopping validation matches do while its Training_Rows are clean.
LEAK_TRAIN = "train"
LEAK_EARLY_STOPPING = "early_stopping"

_MOST_DRAWN_FOLDS = 7  # fold counts are drawn up to 7, or up to D15 when it is larger
# Matches of each Competition_Type: mostly 3-15, sometimes 1-2 or none; shrinking goes to "some".
_TYPE_SIZES = ("some", "some", "some", "some", "few", "absent")
_TYPE_MATCHES: dict[str, SearchStrategy[int]] = {"some": st.integers(3, 15), "few": st.integers(1, 2)}
_FIRST_MATCH_ID = st.integers(1, 50_000)
_MATCH_ID_GAP = st.integers(2, 1_000)  # at least 2, so that no two match ids are consecutive
_DELIVERIES_PER_MATCH = (1, 2, 3, 0, 1, 2)
_FOLD_SEEDS = st.one_of(st.integers(0, 2**32 - 1), st.just(FOLD_SEED))
_MAX_FITS = 5

# The models whose fits go through the leakage guard (Requirement 7, criterion 5), with whether
# each makes inner fits (design, table in "Fold sandboxes and leakage control") and whether it holds
# out early-stopping validation matches (its LightGBM fits). Hypothesis draws the first entry most
# often, so a model with both comes first and early-stopping leaks are common.
_FIT_MODELS: dict[str, tuple[bool, bool]] = {
    "Remaining_Runs_Model": (True, True),
    "Skill_Model": (True, False),
    "Win_Probability_Model": (True, True),
    "Outcome_Model": (False, True),
    "Extras_Model": (False, False),
}
# The leaks a fit case can have, by whether its model early-stops; valid first, for shrinking.
_LEAKS: dict[bool, tuple[str | None, ...]] = {
    False: (None, LEAK_TRAIN),
    True: (None, LEAK_TRAIN, LEAK_EARLY_STOPPING),
}
_TRAIN_SHAPES = ("all", "subset")  # all matches of the training folds, or a drawn subset


class FitCase(NamedTuple):
    """One fit for the leakage guard; the module docstring (``fold_tables``) describes the fields.

    ``spec`` (a FitSpec of sandbox_plan) and ``model``; the int32, ascending id arrays
    ``train_match_ids`` (one per Training_Row), ``es_match_ids`` (early-stopping validation
    matches) and ``predicted_match_ids`` (the matches of ``spec.predicted_fold``); ``leak`` (None,
    LEAK_TRAIN or LEAK_EARLY_STOPPING) and ``overlap`` (the distinct excluded-fold matches among
    the training and validation matches; 0 exactly when ``leak`` is None).
    """

    spec: FitSpec
    model: str
    train_match_ids: np.ndarray
    es_match_ids: np.ndarray
    predicted_match_ids: np.ndarray
    leak: str | None
    overlap: int


class FoldTables(NamedTuple):
    """Matches, a fold count and seed, their Fold_Table and Deliveries, and fits; see the module docstring.

    ``matches`` (``match_id``, ``comp_type``; rows shuffled), ``n_folds``, ``seed``, ``fold_table``
    (assign_folds of the three), ``deliveries`` (``row``, ``match_id``) and ``fits`` (FitCases;
    empty unless ``fits=True``).
    """

    matches: pd.DataFrame
    n_folds: int
    seed: int
    fold_table: pd.DataFrame
    deliveries: pd.DataFrame
    fits: tuple[FitCase, ...]


def _check_fold_count(n_folds: Any, least: int) -> int:
    if isinstance(n_folds, bool) or not isinstance(n_folds, int) or n_folds < least:
        raise InvalidArgument(f"n_folds must be an int of at least {least}; got {n_folds!r}")
    return n_folds


def fold_tables(
    *, n_folds: int | SearchStrategy[int] | None = None, fits: bool = False
) -> SearchStrategy[FoldTables]:
    """Matches, a fold count and seed, their Fold_Table and Deliveries, and fits; see the module docstring.

    ``n_folds`` is an int of at least 2 (3 with ``fits``, as sandbox_plan needs), a strategy of
    such ints, or None to draw one from 2-7 (3-7 with ``fits``), D15 most often. ``fits=True``
    adds 1-5 FitCases, valid and leaky, for the leakage guard.
    """
    least = MIN_SANDBOX_FOLDS if fits else MIN_FOLDS
    if n_folds is not None and not isinstance(n_folds, SearchStrategy):
        _check_fold_count(n_folds, least)
    return _fold_tables(n_folds, bool(fits))


def _draw_matches(draw: Any, least: int) -> pd.DataFrame:
    """At least ``least`` matches with Competition_Types: distinct, non-consecutive ids, rows shuffled."""
    counts = []
    for _ in COMPETITION_TYPES:
        size = draw(st.sampled_from(_TYPE_SIZES))
        counts.append(draw(_TYPE_MATCHES[size]) if size in _TYPE_MATCHES else 0)
    if sum(counts) < least:  # top up the largest type, so that absent and small types stay as drawn
        counts[counts.index(max(counts))] += least - sum(counts)
    total = sum(counts)
    gaps = draw(st.lists(_MATCH_ID_GAP, min_size=total - 1, max_size=total - 1))
    ids = np.cumsum([draw(_FIRST_MATCH_ID), *gaps])
    types = [name for name, count in zip(COMPETITION_TYPES, counts) for _ in range(count)]
    types = [types[i] for i in draw(st.permutations(range(total)))]  # the types mixed in match_id order
    order = draw(st.permutations(range(total)))  # and the rows out of match_id order
    return pd.DataFrame(
        {
            "match_id": pd.array([int(ids[i]) for i in order], dtype="Int64"),
            "comp_type": pd.array([types[i] for i in order], dtype="str"),
        }
    )


def _draw_deliveries(draw: Any, match_ids: np.ndarray) -> pd.DataFrame:
    """``row`` and ``match_id`` of 0-3 Deliveries per match, sorted by match (``match_ids`` is ascending)."""
    per_match = draw(
        st.lists(st.sampled_from(_DELIVERIES_PER_MATCH), min_size=len(match_ids), max_size=len(match_ids))
    )
    ball_match = np.repeat(match_ids.astype(np.int32), per_match)
    return pd.DataFrame({"row": np.arange(len(ball_match), dtype=np.int32), "match_id": ball_match})


def _id_array(ids: Iterable[int]) -> np.ndarray:
    return np.asarray(sorted(ids), dtype=np.int32)


def _draw_leaked(draw: Any, spec: FitSpec, by_fold: Mapping[int, list[int]], excluded: list[int]) -> list[int]:
    """The matches of the excluded folds that a leaky fit uses: some of them, or all of one excluded fold."""
    if draw(st.booleans()):  # all of one excluded fold, as when a fit forgets to drop it
        fold = draw(st.sampled_from([fold for fold in sorted(spec.excluded_folds) if by_fold[fold]]))
        return list(by_fold[fold])
    return draw(st.lists(st.sampled_from(excluded), min_size=1, max_size=len(excluded), unique=True))


def _draw_fit(draw: Any, spec: FitSpec, by_fold: Mapping[int, list[int]]) -> FitCase | None:
    """A valid or leaky FitCase for ``spec``; None if the Fold_Table leaves every training fold empty."""
    allowed = sorted(match for fold in spec.train_folds for match in by_fold[fold])
    excluded = sorted(match for fold in spec.excluded_folds for match in by_fold[fold])
    if not allowed:
        return None
    model = draw(st.sampled_from([name for name, (inner, _) in _FIT_MODELS.items() if inner or spec.kind == OUTER]))
    early_stops = _FIT_MODELS[model][1]
    leak = draw(st.sampled_from(_LEAKS[early_stops])) if excluded else None
    if draw(st.sampled_from(_TRAIN_SHAPES)) == "all":
        clean = allowed
    else:  # a fit that leaks into its Training_Rows may train on excluded-fold matches alone
        least = 0 if leak == LEAK_TRAIN else 1
        clean = draw(st.lists(st.sampled_from(allowed), min_size=least, max_size=len(allowed), unique=True))
    leaked = _draw_leaked(draw, spec, by_fold, excluded) if leak is not None else []
    es: list[int] = []
    if early_stops and clean and (leak == LEAK_EARLY_STOPPING or draw(st.booleans())):
        # Whole matches drawn from the clean training matches, at most half of them.
        least = 0 if leak == LEAK_EARLY_STOPPING else 1
        es = draw(st.lists(st.sampled_from(clean), min_size=least, max_size=max(1, len(clean) // 2), unique=True))
    train = clean + leaked if leak == LEAK_TRAIN else clean
    if leak == LEAK_EARLY_STOPPING:
        es = es + leaked
    _require(
        len(set(excluded) & (set(train) | set(es))) == len(leaked),
        f"fit case {spec.fit_id} does not overlap the excluded folds in exactly its leaked matches",
    )
    train_ids = _id_array(train)
    return FitCase(
        spec=spec,
        model=model,
        train_match_ids=np.repeat(train_ids, 1 + train_ids % 3),
        es_match_ids=_id_array(es),
        predicted_match_ids=_id_array(by_fold[spec.predicted_fold]),
        leak=leak,
        overlap=len(leaked),
    )


@st.composite
def _fold_tables(draw: Any, n_folds: int | SearchStrategy[int] | None, fits: bool) -> FoldTables:
    least = MIN_SANDBOX_FOLDS if fits else MIN_FOLDS
    if n_folds is None:
        drawn = st.integers(least, max(_MOST_DRAWN_FOLDS, N_FOLDS))
        k = draw(st.one_of(st.just(N_FOLDS), drawn) if N_FOLDS >= least else drawn)
    elif isinstance(n_folds, int):
        k = n_folds
    else:
        k = _check_fold_count(draw(n_folds), least)
    matches = _draw_matches(draw, k if fits else 1)
    deliveries = _draw_deliveries(draw, np.sort(matches["match_id"].to_numpy(dtype=np.int64)))
    seed = draw(_FOLD_SEEDS)
    fold_table = assign_folds(matches, matches["comp_type"], n_folds=k, seed=seed)
    cases: list[FitCase] = []
    if fits:
        table_ids = fold_table["match_id"].to_numpy(dtype=np.int64)
        table_folds = fold_table["fold"].to_numpy(dtype=np.int64)
        by_fold = {fold: table_ids[table_folds == fold].tolist() for fold in range(1, k + 1)}
        plan = sandbox_plan(k)
        specs = draw(st.lists(st.sampled_from(plan), min_size=1, max_size=_MAX_FITS, unique_by=lambda spec: spec.fit_id))
        cases = [case for case in (_draw_fit(draw, spec, by_fold) for spec in specs) if case is not None]
    return FoldTables(matches, k, seed, fold_table, deliveries, tuple(cases))


# ------------------------------------------------------------------------------- cell tables

#: The kinds of training Delivery whose absence from an innings x Phase cell of a target fold's
#: Training_Folds makes the cell use a pooled value (Requirement 15, criteria 20 and 21), in the
#: order of the pooled quantities: a Wide (mean Wide team runs), a No_Ball (mean No_Ball penalty
#: runs and bat runs), and a Free_Hit or non-Free_Hit Legal_Ball (Free_Hit_Premium).
KIND_WIDE = "wide"
KIND_NO_BALL = "no_ball"
KIND_FREE_HIT = "free_hit_legal"
KIND_NON_FREE_HIT = "non_free_hit_legal"
CELL_KINDS: tuple[str, ...] = (KIND_WIDE, KIND_NO_BALL, KIND_FREE_HIT, KIND_NON_FREE_HIT)

#: Names of the cell column that cell_tables draws: the ``cell`` of ``state``, and a name standing
#: for the cells of the D8 robustness case, which the cell tables take through their ``cells``
#: argument. The first is where shrinking goes.
CELL_NAMES: tuple[str, ...] = ("cell", "cell_d8")

_DECOY_CELL = "cell"  # the column that code ignoring the ``cells`` argument would read
_N_PHASES = 3  # powerplay, middle and death
_CELL_CODES: tuple[int, ...] = tuple(range(2 * _N_PHASES))  # 3 x (innings - 1) + Phase, innings 1 and 2

# Columns of cell_tables' deliveries; the cell column, and the decoy cell column, come between them.
_CELL_TABLE_HEAD = ("row", "match_id", "innings", "fold")
_CELL_TABLE_TAIL = ("cls", "extras_type", "runs", "extras_2_runs", "team_runs", "free_hit", "is_super_over", "excluded")

# The folds whose training rows hold a kind in a cell: every fold, two or more (with 3 folds or
# more), one fold, or none. Every fold is drawn most often and is where shrinking goes, so a
# shrunk example has no emptied cell.
_EVERY, _SEVERAL, _ONE, _NO_FOLD = "every", "several", "one", "none"
_HOLDER_SHAPES = (_EVERY,) * 6 + (_SEVERAL,) * 2 + (_ONE,) * 2 + (_NO_FOLD,)
_HOLDER_SHAPES_TWO_FOLDS = tuple(shape for shape in _HOLDER_SHAPES if shape != _SEVERAL)
_SPARE_HOLDERS = 3  # folds that hold each kind in some cell, so no one fold decides another fold's pooled set
_ROWS_PER_HOLDER = (1, 1, 2, 3)  # training rows of a kind in a cell of a fold that holds it
_MATCHES_PER_FOLD = (1, 1, 2, 3)
_EXCLUDED_MATCH = max(_MATCHES_PER_FOLD)  # the match index j of a fold's excluded match
_MAX_DROPPED = 6
_EXCLUDED_ROW, _SUPER_OVER_ROW = "excluded", "super_over"
_SUPER_OVER_INNINGS = (3, 4)

# extras_type values of each kind, weighted as for innings(): plain Legal_Balls first.
_KIND_TYPES: dict[str, tuple[str, ...]] = {
    KIND_WIDE: _weighted(EXTRAS_TYPE_CLASSES["wide"]),
    KIND_NO_BALL: _weighted(EXTRAS_TYPE_CLASSES["no_ball"]),
    KIND_FREE_HIT: _LEGAL_TYPES,
    KIND_NON_FREE_HIT: _LEGAL_TYPES,
}
_LEGAL_KINDS = (KIND_FREE_HIT, KIND_NON_FREE_HIT)
# The extras_2_runs choices of each role, none of length 1: a one-value list would draw nothing.
_CELL_EXTRA_RUNS: dict[str, tuple[int, ...]] = {
    role: values if len(values) > 1 else values * 2 for role, values in _EXTRA_RUNS.items()
}


class CellTables(NamedTuple):
    """Deliveries with folds and innings x Phase cells, and which cells of which Training_Folds lack which kind.

    ``deliveries`` (the Delivery frame), ``n_folds``, ``cells`` (the name of its cell column),
    ``encoding`` (the Extras_Encoding of its ``team_runs``), ``emptied`` ((target fold, cell,
    kind) triples whose Training_Folds hold no Delivery of the kind in the cell) and
    ``empty_pooled`` ((target fold, kind) pairs whose Training_Folds hold none in any cell); see
    the module docstring (``cell_tables``).
    """

    deliveries: pd.DataFrame
    n_folds: int
    cells: str
    encoding: str
    emptied: frozenset[tuple[int, int, str]]
    empty_pooled: frozenset[tuple[int, str]]


def cell_tables(
    *,
    n_folds: int | SearchStrategy[int] | None = None,
    pooled: bool = False,
    empty_pooled: bool = False,
    cells: str | None = None,
) -> SearchStrategy[CellTables]:
    """Deliveries with folds and cells, chosen cells of chosen Training_Folds lacking a kind; see the module docstring.

    ``n_folds`` is an int of at least 2, a strategy of such ints, or None for D15. ``pooled=True``
    forces a pooled value that a Bowler_Delivery uses. ``empty_pooled=True`` forces a pooled set
    that is empty for a whole target fold, so that the cell tables raise StageError (exit code 5);
    it is off by default, and then no pooled set is empty. ``cells`` names the cell column (drawn
    from CELL_NAMES when None) and may not be the name of another column of ``deliveries``.
    """
    if n_folds is not None and not isinstance(n_folds, SearchStrategy):
        _check_fold_count(n_folds, MIN_FOLDS)
    taken = _CELL_TABLE_HEAD + _CELL_TABLE_TAIL
    if cells is not None and (not isinstance(cells, str) or not cells or cells in taken):
        raise InvalidArgument(f"cells must be a non-empty column name other than {', '.join(taken)}; got {cells!r}")
    absent = [kind for kind, values in _KIND_TYPES.items() if not values]
    if absent:
        raise RuntimeError(
            f"cell_tables needs extras_type values for the kinds {absent} in data_rules.extras_type_classes; "
            f"found {EXTRAS_TYPE_CLASSES!r}"
        )
    return _cell_tables(n_folds, bool(pooled), bool(empty_pooled), cells)


def _draw_holders(draw: Any, folds: tuple[int, ...]) -> frozenset[int]:
    """The folds whose training rows hold one kind in one cell: every fold, several, one or none."""
    shape = draw(st.sampled_from(_HOLDER_SHAPES if len(folds) > 2 else _HOLDER_SHAPES_TWO_FOLDS))
    if shape == _EVERY:
        return frozenset(folds)
    if shape == _SEVERAL:
        return frozenset(draw(st.lists(st.sampled_from(folds), min_size=2, max_size=len(folds) - 1, unique=True)))
    if shape == _ONE:
        return frozenset({draw(st.sampled_from(folds))})
    return frozenset()


def _draw_kind_ball(draw: Any, kind: str, encoding: str) -> tuple[int, str, int, int, int, bool]:
    """One Delivery of ``kind``: cls, extras_type, bat runs, extras_2_runs, team runs under ``encoding``, free_hit.

    Bat runs and extras_2_runs are drawn for every extras_type, and bat runs then set to 0 where
    the role scores none, so that every value takes the same number of choices. Hypothesis
    shrinks towards shorter choice sequences, so it would otherwise prefer a bye or a Penalty
    Delivery to a plain dot ball.
    """
    value = draw(st.sampled_from(_KIND_TYPES[kind]))
    ball = _Ball(value, _role_of(value))
    bat = draw(st.sampled_from(_BAT_RUNS))
    ball.bat = bat if ball.role in ("plain", "no_ball") else 0
    ball.extra = draw(st.sampled_from(_CELL_EXTRA_RUNS[ball.role]))
    if kind in _LEGAL_KINDS:
        free_hit = kind == KIND_FREE_HIT
    else:  # a Wide or No_Ball straight after a No_Ball is a Free_Hit too, but not a Free_Hit Legal_Ball
        free_hit = draw(st.sampled_from(_SOMETIMES))
    return ball.cls, value, ball.bat, ball.extra, _team_runs(ball, encoding), free_hit


def _cell_match_id(n_folds: int, fold: int, j: int) -> int:
    """The id of match j of ``fold`` (j = _EXCLUDED_MATCH for its excluded match); the folds interleave."""
    return 1 + j * n_folds + (fold - 1)


def _decoy(cell: int | None) -> int | None:
    """Another cell of the same innings: the next Phase, the death overs wrapping to the powerplay."""
    return None if cell is None else _N_PHASES * (cell // _N_PHASES) + (cell % _N_PHASES + 1) % _N_PHASES


@st.composite
def _cell_tables(
    draw: Any, n_folds: int | SearchStrategy[int] | None, pooled: bool, empty_pooled: bool, cells: str | None
) -> CellTables:
    if n_folds is None:
        n = _check_fold_count(N_FOLDS, MIN_FOLDS)
    elif isinstance(n_folds, int):
        n = n_folds
    else:
        n = _check_fold_count(draw(n_folds), MIN_FOLDS)
    name = cells if cells is not None else draw(st.sampled_from(CELL_NAMES))
    encoding = draw(st.sampled_from(CANDIDATE_ENCODINGS))
    folds = tuple(range(1, n + 1))
    slots = [(cell, kind) for cell in _CELL_CODES for kind in CELL_KINDS]

    # The holders of each kind in each cell, then the forced cases, then the spare holders.
    holders = {slot: _draw_holders(draw, folds) for slot in slots}
    forced: tuple[int, str] | None = None
    if pooled:  # one fold alone holds a kind in a cell: a pooled value that its Deliveries use
        forced = draw(st.sampled_from(slots))
        holders[forced] = frozenset({draw(st.sampled_from(folds))})
    if empty_pooled:  # a kind held by at most one fold k, so k's Training_Folds hold none of it
        kind = draw(st.sampled_from(CELL_KINDS))
        if forced is not None and forced[1] == kind:
            target = min(holders[forced])  # the forced fold keeps its cell, and the case stays as forced
        else:
            target = draw(st.sampled_from(folds))
        for cell in _CELL_CODES:
            holders[cell, kind] = holders[cell, kind] & {target}
    else:
        least = min(_SPARE_HOLDERS, n)
        for kind in CELL_KINDS:
            if len(frozenset().union(*(holders[cell, kind] for cell in _CELL_CODES))) < least:
                widened = next(cell for cell in _CELL_CODES if (cell, kind) != forced)
                holders[widened, kind] = frozenset(folds)
    emptied = frozenset((k, cell, kind) for (cell, kind), held in holders.items() for k in folds if held <= {k})
    empty = frozenset(
        (k, kind) for k in folds for kind in CELL_KINDS if all((k, cell, kind) in emptied for cell in _CELL_CODES)
    )

    # Records: match_id, innings, fold, cell, the six values of _draw_kind_ball, is_super_over, excluded.
    matches = {fold: draw(st.sampled_from(_MATCHES_PER_FOLD)) for fold in folds}

    def regular_match(fold: int) -> int:
        return _cell_match_id(n, fold, draw(st.integers(0, matches[fold] - 1)) if matches[fold] > 1 else 0)

    # Every fold draws its rows of every cell and kind, and keeps them only where it holds the kind,
    # so emptying a cell gives no shorter choice sequence. Hypothesis shrinks towards shorter
    # sequences first; this way it shrinks towards held cells rather than towards emptied ones.
    records: list[tuple[Any, ...]] = []
    for cell, kind in slots:
        for fold in folds:
            n_rows = draw(st.sampled_from(_ROWS_PER_HOLDER))
            drawn = [(regular_match(fold), _draw_kind_ball(draw, kind, encoding)) for _ in range(n_rows)]
            if fold in holders[cell, kind]:
                innings_no = 1 + cell // _N_PHASES
                records.extend((match_id, innings_no, fold, cell, *ball, False, False) for match_id, ball in drawn)

    # Dropped rows, about half of them where they would fill a cell that emptied lists.
    poison = sorted({(fold, cell, kind) for k, cell, kind in emptied for fold in folds if fold != k})
    anywhere = st.sampled_from([(fold, cell, kind) for fold in folds for cell, kind in slots])
    place = st.one_of(st.sampled_from(poison), anywhere) if poison else anywhere
    for _ in range(draw(st.integers(0, _MAX_DROPPED))):
        dropped_as = draw(st.sampled_from((_EXCLUDED_ROW, _SUPER_OVER_ROW)))
        fold, cell, kind = draw(place)
        ball = _draw_kind_ball(draw, kind, encoding)
        if dropped_as == _EXCLUDED_ROW:
            match_id = _cell_match_id(n, fold, _EXCLUDED_MATCH)
            records.append((match_id, 1 + cell // _N_PHASES, fold, cell, *ball, False, True))
        else:
            innings_no = draw(st.sampled_from(_SUPER_OVER_INNINGS))
            records.append((regular_match(fold), innings_no, fold, None, *ball, True, False))
    held_folds = {record[2] for record in records}
    for fold in folds:
        if fold not in held_folds:  # a fold holding no kind in any cell still needs a row to be a target fold
            dot = (_LEGAL, _PLAIN_TYPE, 0, 0, 0, False)
            records.append((_cell_match_id(n, fold, 0), _SUPER_OVER_INNINGS[0], fold, None, *dot, True, False))

    columns = list(zip(*records))
    match_ids = np.array(columns[0], dtype=np.int32)
    innings_nos = np.array(columns[1], dtype=np.int32)
    order = np.lexsort((innings_nos, match_ids))  # stable: by match, then innings, in drawing order
    cell_codes = [columns[3][i] for i in order]
    values: dict[str, Any] = {
        "row": np.arange(len(order), dtype=np.int32),
        "match_id": match_ids[order],
        "innings": innings_nos[order],
        "fold": np.array(columns[2], dtype=np.int32)[order],
        name: pd.array(cell_codes, dtype="Int8"),
    }
    if name != _DECOY_CELL:
        values[_DECOY_CELL] = pd.array([_decoy(cell) for cell in cell_codes], dtype="Int8")
    values.update(
        {
            "cls": np.array(columns[4], dtype=np.int8)[order],
            "extras_type": pd.array([columns[5][i] for i in order], dtype="str"),
            "runs": np.array(columns[6], dtype=np.int32)[order],
            "extras_2_runs": np.array(columns[7], dtype=np.int32)[order],
            "team_runs": np.array(columns[8], dtype=np.int32)[order],
            "free_hit": np.array(columns[9], dtype=bool)[order],
            "is_super_over": np.array(columns[10], dtype=bool)[order],
            "excluded": np.array(columns[11], dtype=bool)[order],
        }
    )
    return CellTables(pd.DataFrame(values), n, name, encoding, emptied, empty)


# ------------------------------------------------------------------------------- contributions

#: The Recency_Weight half-life D9 in years (``design_choices.D9_half_life_years``; None means no
#: decay) and the knockout Stakes_Weight D10 (``design_choices.D10_knockout_weight``), from which
#: contributions() builds each match's Delivery_Weight.
HALF_LIFE_YEARS: float | None = _CONFIG.get("design_choices.D9_half_life_years")
KNOCKOUT_WEIGHT: float = _CONFIG.get("design_choices.D10_knockout_weight")

#: The edge cases for which contributions(force=...) adds a player to each generated Role (module
#: docstring, ``contributions``). CHARGE_ONLY applies to the batter Role only.
ALL_MATCHED = "all_matched"
UNMATCHED_CELL = "unmatched_cell"
NO_MATCHED_CELL = "no_matched_cell"
ONE_PRESSURE_MATCH = "one_pressure_match"
CHARGE_ONLY = "charge_only"
TIES = "ties"
FORCED_CASES: tuple[str, ...] = (ALL_MATCHED, UNMATCHED_CELL, NO_MATCHED_CELL, ONE_PRESSURE_MATCH, CHARGE_ONLY, TIES)

#: Columns of ContributionCase.entries: one row per Delivery or Dismissal_Charge of a player in a Role.
ENTRY_COLUMNS: tuple[str, ...] = (
    "role", "player_id", "match_id", "cell", "pressure", "charge", "value", "weight", "s", "w", "n", "half",
)  # fmt: skip
_ENTRY_INPUTS = ENTRY_COLUMNS[:7]  # the columns contribution_frame reads; the others follow from them
_MATCH_INPUTS = ("match_id", "match_date", "calendar_year", "comp_class", "period", "delivery_weight")

_BATTER_ONLY_CASES = frozenset({CHARGE_ONLY})
_TWO_MATCH_CASES = frozenset({ONE_PRESSURE_MATCH, CHARGE_ONLY, TIES})  # their players need 2 matches

# Matches: ids, dates (in an order unrelated to the ids, with shared dates and dates on both sides
# of the Late_Period start), Competition_Types and knockouts. First entries are the simplest.
_CONTRIBUTION_MATCH_IDS = st.integers(1, 99_999)
_FIRST_MATCH_DATE = date(2021, 1, 1)
_LAST_MATCH_DATE = date(2026, 11, 1)  # the last match_date as delivered
_DATE_SPAN_DAYS = (_LAST_MATCH_DATE - _FIRST_MATCH_DATE).days
_PERIOD_EDGE_DATES = ((LATE_PERIOD_START - pd.Timedelta(days=1)).date(), LATE_PERIOD_START.date())
_NEW_DATE, _SAME_DATE, _EDGE_DATE = "new", "same", "edge"
_DATE_KINDS = (_NEW_DATE,) * 4 + (_SAME_DATE, _EDGE_DATE)
_KNOCK_OUT = (0, 0, 0, 0, 1)
_RECENCY_STEPS = 64  # Recency_Weights are rounded to multiples of 1/64, so weighted sums stay exact
_DAYS_PER_YEAR = 365.25  # the glossary's Recency_Weight exponent: days / (D9 x 365.25)
_MAX_DEFAULT_MATCHES = 6

# Players and their entries. Each choice list starts with its simplest value, for shrinking.
_DEFAULT_PLAYERS = st.integers(1, 4)
_PRESSURE_SHARES = ((False, False, False, True), (False, True), (False,), (True,))  # about 25%, 50%, none, all
_MAX_HOME_CELLS = 3
_MAX_PLAYER_MATCHES = 4
_BALLS_PER_PLAYER_MATCH = (1, 2, 3, 1, 4, 6)
_DENSE_EXTRA_BALLS = (0, 1)
_EXTRA_BALLS = (0, 1, 2, 3)
_CHARGES_PER_BATTER = (0, 0, 1, 2)
_CHARGES_ONLY = (False,) * 15 + (True,)  # one random batter in 16 faced no ball and has charges alone
# A batter's NBV in quarter runs, one choice per Delivery: 0 first, then -4 to +6 runs by size, then
# 6 wicket-sized values of -10 to -30 runs, so about one Delivery in 8 is a wicket.
_NBV_QUARTERS: tuple[int, ...] = tuple(sorted(range(-16, 25), key=abs)) + (-40, -50, -60, -80, -100, -120)
_NBV = st.sampled_from(_NBV_QUARTERS)
_OMEGA_QUARTERS = st.integers(0, 120)  # the Omega of a Dismissal_Charge: 0 to 30 runs


def _choice(items: tuple[Any, ...]) -> SearchStrategy[Any]:
    """``st.sampled_from(items)``, built once: building a new strategy for every draw makes generation slow.

    The cache key holds the types of the items too, since (0, 1) == (False, True) and both hash
    alike: without them, a tuple of match indices could get the strategy of a tuple of flags.
    """
    return _sampled_from(items, tuple(type(item) for item in items))


@lru_cache(maxsize=4096)
def _sampled_from(items: tuple[Any, ...], types: tuple[type, ...]) -> SearchStrategy[Any]:
    return st.sampled_from(items)


@lru_cache(maxsize=256)
def _ordering(items: tuple[int, ...]) -> SearchStrategy[list[int]]:
    """``st.permutations(items)`` of ints (match indices or cells), built once."""
    return st.permutations(items)


# One value entry of a player in a Role: (match index, cell, pressure, charge, value).
_Entry = tuple[int, int, bool, bool, float]


class ContributionCase(NamedTuple):
    """Per-player contributions, the entries and matches behind them, and the forced players.

    ``contrib`` (``Contributions.from_frame(frame, match_ids)``), ``frame`` (the contributions
    table, CONTRIBUTION_COLUMNS), ``entries`` (one row per Delivery or Dismissal_Charge,
    ENTRY_COLUMNS), ``matches`` (the match universe with the match_weights columns; row i is
    match index i), ``match_ids`` (int64, ascending), ``roles`` (the Roles generated, in ROLES
    order), ``forced`` ((case, Role) -> the case player's ``player_id``) and ``twins`` (Role ->
    (``player_id``, twin ``player_id``) for TIES). See the module docstring (``contributions``).
    """

    contrib: Contributions
    frame: pd.DataFrame
    entries: pd.DataFrame
    matches: pd.DataFrame
    match_ids: np.ndarray
    roles: tuple[str, ...]
    forced: dict[tuple[str, str], int]
    twins: dict[str, tuple[int, int]]


def _half_codes(
    keys: Iterable[tuple[str, int, int, bool]], dates: np.ndarray, match_ids: np.ndarray
) -> dict[tuple[str, int, int], int]:
    """The Half code of each (Role, player, match index) of ``keys`` = (Role, player, match index, charge).

    Requirement 18, criterion 7: the player's matches with a Delivery in the Role, numbered 1, 2,
    ... by match date and then ``match_id``, are odd or even; a match with charges alone is in
    neither half.
    """
    seen: dict[tuple[str, int], set[int]] = {}
    played: dict[tuple[str, int], set[int]] = {}
    for role, player, match, charge in keys:
        seen.setdefault((role, player), set()).add(match)
        if not charge:
            played.setdefault((role, player), set()).add(match)
    halves: dict[tuple[str, int, int], int] = {}
    for (role, player), matches in seen.items():
        numbered = sorted(played.get((role, player), ()), key=lambda match: (dates[match], match_ids[match]))
        for number, match in enumerate(numbered, start=1):
            halves[role, player, match] = int(Half.ODD) if number % 2 else int(Half.EVEN)
        for match in matches.difference(numbered):
            halves[role, player, match] = int(Half.NONE)
    return halves


def contribution_frame(entries: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """The contributions table of value entries: Requirement 16, criterion 2, transcribed (module docstring).

    ``entries`` needs ``role``, ``player_id``, ``match_id``, ``cell``, ``pressure``, ``charge``
    and ``value`` (the first seven ENTRY_COLUMNS); its other columns are not read. ``matches``
    needs ``match_id``, ``match_date``, ``calendar_year``, ``comp_class``, ``period`` and
    ``delivery_weight``, one row per match of the universe, such as ContributionCase.matches.
    Returns CONTRIBUTION_COLUMNS with the dtypes of ``Contributions.to_frame``, sorted as
    ``build_contributions`` sorts its rows, for ``Contributions.from_frame(frame,
    matches["match_id"])``. Raises ValueError for a missing column, an unknown Role or a
    ``match_id`` that ``matches`` lacks.
    """
    absent = [f"entries.{name}" for name in _ENTRY_INPUTS if name not in entries.columns]
    absent += [f"matches.{name}" for name in _MATCH_INPUTS if name not in matches.columns]
    if absent:
        raise ValueError(f"contribution_frame needs the column(s) {', '.join(absent)}")
    match_ids = matches["match_id"].to_numpy(dtype=np.int64)
    position = {int(match_id): i for i, match_id in enumerate(match_ids)}
    dates = matches["match_date"].to_numpy(dtype="datetime64[ns]")
    weight = matches["delivery_weight"].to_numpy(dtype=np.float64)
    records = list(zip(*(entries[name].tolist() for name in _ENTRY_INPUTS)))
    unknown_roles = sorted({record[0] for record in records}.difference(ROLE_CODES))
    unknown_matches = sorted({int(record[2]) for record in records}.difference(position))
    if unknown_roles or unknown_matches:
        raise ValueError(f"entries hold unknown Role(s) {unknown_roles} or match_id(s) {unknown_matches}")

    sums: dict[tuple[int, int, int, int, bool], list[Any]] = {}  # (Role code, player, match, cell, class)
    keys: list[tuple[str, int, int, bool]] = []
    for role, player, match_id, cell, pressure, charge, value in records:
        match = position[int(match_id)]
        keys.append((role, int(player), match, bool(charge)))
        total = sums.setdefault((ROLE_CODES[role], int(player), match, int(cell), bool(pressure)), [0.0, 0.0, 0])
        total[0] += float(weight[match]) * float(value)  # Delivery_Weight x NBV, or x the charge
        if not charge:  # a Dismissal_Charge adds nothing to the weight or the count
            total[1] += float(weight[match])
            total[2] += 1
    halves = _half_codes(keys, dates, match_ids)
    rows = sorted(sums)
    at = np.array([row[2] for row in rows], dtype=np.int64)
    return pd.DataFrame(
        {
            "role": pd.array([ROLES[row[0]] for row in rows], dtype="str"),
            "player_id": np.array([row[1] for row in rows], dtype=np.int64),
            "match_id": match_ids[at],
            "cell": np.array([row[3] for row in rows], dtype=np.int64),
            "pressure": np.array([row[4] for row in rows], dtype=bool),
            "s": np.array([sums[row][0] for row in rows], dtype=np.float64),
            "w": np.array([sums[row][1] for row in rows], dtype=np.float64),
            "n": np.array([sums[row][2] for row in rows], dtype=np.int64),
            "half": np.array([halves[ROLES[row[0]], row[1], row[2]] for row in rows], dtype=np.int8),
            "period": matches["period"].to_numpy(dtype=np.int8)[at],
            "comp_class": matches["comp_class"].to_numpy(dtype=np.int8)[at],
            "calendar_year": matches["calendar_year"].to_numpy(dtype=np.int64)[at],
        },
        columns=list(CONTRIBUTION_COLUMNS),
    )


def _entry_table(rows: Sequence[tuple[Any, ...]], matches: pd.DataFrame) -> pd.DataFrame:
    """ENTRY_COLUMNS for ``rows`` = (Role, player_id, match index, cell, pressure, charge, value)."""
    match_ids = matches["match_id"].to_numpy(dtype=np.int64)
    dates = matches["match_date"].to_numpy(dtype="datetime64[ns]")
    ordered = sorted(rows, key=lambda row: (ROLE_CODES[row[0]], *row[1:6]))  # stable within a group
    halves = _half_codes(((row[0], row[1], row[2], row[5]) for row in ordered), dates, match_ids)
    at = np.array([row[2] for row in ordered], dtype=np.int64)
    charge = np.array([row[5] for row in ordered], dtype=bool)
    value = np.array([row[6] for row in ordered], dtype=np.float64)
    weight = matches["delivery_weight"].to_numpy(dtype=np.float64)[at]
    return pd.DataFrame(
        {
            "role": pd.array([row[0] for row in ordered], dtype="str"),
            "player_id": np.array([row[1] for row in ordered], dtype=np.int64),
            "match_id": match_ids[at],
            "cell": np.array([row[3] for row in ordered], dtype=np.int64),
            "pressure": np.array([row[4] for row in ordered], dtype=bool),
            "charge": charge,
            "value": value,
            "weight": weight,
            "s": weight * value,
            "w": np.where(charge, 0.0, weight),
            "n": np.where(charge, 0, 1).astype(np.int64),
            "half": np.array([halves[row[0], row[1], row[2]] for row in ordered], dtype=np.int8),
        },
        columns=list(ENTRY_COLUMNS),
    )


def _check_roles(roles: Iterable[str] | None) -> tuple[str, ...]:
    if roles is None:
        return ROLES
    chosen = None if isinstance(roles, str) or not isinstance(roles, Iterable) else list(roles)
    if not chosen or len(set(chosen)) != len(chosen) or not set(chosen) <= set(ROLES):
        raise InvalidArgument(f"roles must be distinct Role names from {ROLES}; got {roles!r}")
    return tuple(role for role in ROLES if role in chosen)


def _check_force(force: Iterable[str], roles: tuple[str, ...], charges: bool) -> frozenset[str]:
    if isinstance(force, str) or not isinstance(force, Iterable):
        raise InvalidArgument(f"force must be a collection of FORCED_CASES names; got {force!r}")
    cases = frozenset(force)
    unknown = sorted(map(repr, cases.difference(FORCED_CASES)))
    if unknown:
        raise InvalidArgument(f"force holds {', '.join(unknown)}, which are not FORCED_CASES {FORCED_CASES}")
    if CHARGE_ONLY in cases and (BATTER_ROLE not in roles or not charges):
        raise InvalidArgument("force CHARGE_ONLY needs the batter Role in roles and charges=True")
    return cases


def _drawn_count(
    draw: Any, value: int | SearchStrategy[int] | None, name: str, least: int, default: SearchStrategy[int]
) -> int:
    """``value`` itself, or a draw from it or from ``default``; at least ``least``."""
    count = draw(default) if value is None else value if isinstance(value, int) else draw(value)
    if isinstance(count, bool) or not isinstance(count, int) or count < least:
        raise InvalidArgument(f"{name} must be an int of at least {least}; got {count!r}")
    return count


def contributions(
    *,
    n_players: int | SearchStrategy[int] | None = None,
    n_matches: int | SearchStrategy[int] | None = None,
    roles: Iterable[str] | None = None,
    force: Iterable[str] = (),
    charges: bool = True,
    dense: bool = False,
) -> SearchStrategy[ContributionCase]:
    """Per-player contributions with their entries, matches and forced edge cases; see the module docstring.

    ``n_players``: random players per Role, an int of at least 1, a strategy of such ints (drawn
    for each Role), or None for 1-4. ``n_matches``: matches in the universe, an int, a strategy
    of ints, or None for 1-6; at least 2 when ``force`` holds ONE_PRESSURE_MATCH, CHARGE_ONLY or
    TIES. ``roles``: Role names (None: both). ``force``: names from FORCED_CASES, each adding a
    player of that case to every generated Role (CHARGE_ONLY to the batter Role, which it needs,
    with ``charges``). ``charges``: whether batters have Dismissal_Charges. ``dense``: every
    random player plays every match, with a Pressure_Ball and an Ordinary_Ball in one of his cells.
    """
    if n_players is not None and not isinstance(n_players, SearchStrategy):
        _check_positive("n_players", n_players)
    chosen_roles = _check_roles(roles)
    cases = _check_force(force, chosen_roles, bool(charges))
    least = 2 if cases & _TWO_MATCH_CASES else 1
    if n_matches is not None and not isinstance(n_matches, SearchStrategy):
        _check_positive("n_matches", n_matches)
        if n_matches < least:
            needs = ", ".join(sorted(cases & _TWO_MATCH_CASES))
            raise InvalidArgument(f"n_matches must be at least {least} to force {needs}; got {n_matches}")
    return _contributions(n_players, n_matches, chosen_roles, cases, bool(charges), bool(dense))


def _contribution_matches(draw: Any, n: int) -> pd.DataFrame:
    """``n`` matches with the columns and dtypes of ``rating.match_weights``, sorted by ``match_id``."""
    ids = sorted(draw(st.lists(_CONTRIBUTION_MATCH_IDS, min_size=n, max_size=n, unique=True)))
    dates: list[date] = []
    for _ in range(n):
        kind = draw(_choice(_DATE_KINDS))
        if kind == _SAME_DATE and dates:
            dates.append(draw(_choice(tuple(dates))))
        elif kind == _EDGE_DATE:
            dates.append(draw(_choice(_PERIOD_EDGE_DATES)))
        else:
            dates.append(_FIRST_MATCH_DATE + timedelta(days=draw(st.integers(0, _DATE_SPAN_DAYS))))
    comp_type = [draw(_choice(COMPETITION_TYPES)) for _ in range(n)]
    full_member = np.array([comp == COMP_T20I and draw(st.booleans()) for comp in comp_type], dtype=bool)
    knock_out = np.array([draw(_choice(_KNOCK_OUT)) for _ in range(n)], dtype=np.float64)
    last = max(dates)
    days = np.array([(last - day).days for day in dates], dtype=np.float64)
    if HALF_LIFE_YEARS is None:
        recency = np.ones(n)
    else:  # the D9 formula, rounded to a multiple of 1/64 that is at least 1/64
        exact = 0.5 ** (days / (HALF_LIFE_YEARS * _DAYS_PER_YEAR))
        recency = np.maximum(np.rint(exact * _RECENCY_STEPS), 1.0) / _RECENCY_STEPS
    stakes = np.where(knock_out == 1.0, float(KNOCKOUT_WEIGHT), 1.0)
    comp_class = [
        CompClass.IPL if comp == COMP_IPL else CompClass.FULL_MEMBER if member else CompClass.ASSOCIATE
        for comp, member in zip(comp_type, full_member)
    ]
    late = LATE_PERIOD_START.date()
    return pd.DataFrame(
        {
            "match_id": np.array(ids, dtype=np.int64),
            "match_date": np.array(dates, dtype="datetime64[D]").astype("datetime64[ns]"),
            "calendar_year": np.array([day.year for day in dates], dtype=np.int64),
            "knock_out": knock_out,
            "comp_type": pd.array(comp_type, dtype="str"),
            "full_member_match": full_member,
            "comp_class": np.array(comp_class, dtype=np.int8),
            "period": np.array([Period.LATE if day >= late else Period.EARLY for day in dates], dtype=np.int8),
            "days_to_last": days,
            "recency_weight": recency,
            "stakes_weight": stakes,
            "delivery_weight": recency * stakes,
        },
        columns=list(MATCH_WEIGHT_COLUMNS),
    )


def _subset(draw: Any, items: Sequence[int], least: int, most: int) -> list[int]:
    """``least`` to ``most`` distinct items in a drawn order; shrinks towards the first items."""
    size = draw(st.integers(least, min(most, len(items))))
    return list(draw(_ordering(tuple(items)))[:size])


def _delivery_entry(draw: Any, match: int, cell: int, pressure: bool, sign: int) -> _Entry:
    """A Delivery with its NBV; ``sign`` -1 mirrors a batter's NBV for a bowler."""
    return (match, cell, pressure, False, sign * draw(_NBV) / 4)  # an int over 4, so never -0.0


def _charge_entry(draw: Any, match: int, cell: int, pressure: bool) -> _Entry:
    """A Dismissal_Charge: -Omega."""
    return (match, cell, pressure, True, -draw(_OMEGA_QUARTERS) / 4)


def _extra_balls(draw: Any, n: int, cells: Sequence[int], flags: tuple[bool, ...], sign: int) -> list[_Entry]:
    """0-3 more Deliveries in any matches, in ``cells``, with pressure flags from ``flags``."""
    match, cell, flag = st.integers(0, n - 1), _choice(tuple(cells)), _choice(flags)
    return [
        _delivery_entry(draw, draw(match), draw(cell), draw(flag), sign) for _ in range(draw(_choice(_EXTRA_BALLS)))
    ]


def _random_charges(draw: Any, n: int, played: Sequence[int], cells: Sequence[int]) -> list[_Entry]:
    """0-2 Dismissal_Charges, mostly in the player's own matches and cells, sometimes in any."""
    match, cell = _choice((*played, *range(n))), _choice((*cells, *_CELL_CODES))
    return [
        _charge_entry(draw, draw(match), draw(cell), draw(st.booleans()))
        for _ in range(draw(_choice(_CHARGES_PER_BATTER)))
    ]


def _random_player(draw: Any, n: int, sign: int, charges: bool, dense: bool) -> list[_Entry]:
    """A random player's entries in one Role (module docstring, ``contributions``)."""
    if charges and not dense and draw(_choice(_CHARGES_ONLY)):  # dismissed without facing a ball
        return [
            _charge_entry(draw, draw(st.integers(0, n - 1)), draw(_choice(_CELL_CODES)), draw(st.booleans()))
            for _ in range(draw(st.integers(1, 2)))
        ]
    flag = _choice(draw(_choice(_PRESSURE_SHARES)))
    cells = _subset(draw, _CELL_CODES, 1, _MAX_HOME_CELLS)
    cell = _choice(tuple(cells))
    played = list(range(n)) if dense else _subset(draw, range(n), 1, _MAX_PLAYER_MATCHES)
    count = _choice(_DENSE_EXTRA_BALLS if dense else _BALLS_PER_PLAYER_MATCH)
    entries: list[_Entry] = []
    for match in played:
        if dense:  # a Pressure_Ball and an Ordinary_Ball in one home cell: a Matched_Cell in every match
            home = draw(cell)
            entries += [_delivery_entry(draw, match, home, True, sign), _delivery_entry(draw, match, home, False, sign)]
        entries += [_delivery_entry(draw, match, draw(cell), draw(flag), sign) for _ in range(draw(count))]
    if charges:
        entries += _random_charges(draw, n, played, cells)
    return entries


def _all_matched_player(draw: Any, n: int, sign: int, charges: bool) -> list[_Entry]:
    """ALL_MATCHED: Pressure_Balls in 2-4 cells, each holding an Ordinary_Ball too."""
    cells = _subset(draw, _CELL_CODES, 2, 4)
    entries = [
        _delivery_entry(draw, draw(st.integers(0, n - 1)), cell, pressure, sign)
        for cell in cells
        for pressure in (True, False)
    ]
    entries += _extra_balls(draw, n, cells, (False, True), sign)
    if charges:
        entries += _random_charges(draw, n, sorted({entry[0] for entry in entries}), cells)
    return entries


def _unmatched_cell_player(draw: Any, n: int, sign: int, charges: bool) -> list[_Entry]:
    """UNMATCHED_CELL: 1-3 Matched_Cells and 1-3 cells with Pressure_Balls only."""
    cells = _subset(draw, _CELL_CODES, 2, 4)
    split = draw(st.integers(1, len(cells) - 1))
    matched, unmatched = cells[:split], cells[split:]
    entries = [
        _delivery_entry(draw, draw(st.integers(0, n - 1)), cell, pressure, sign)
        for cell in matched
        for pressure in (True, False)
    ]
    entries += [_delivery_entry(draw, draw(st.integers(0, n - 1)), cell, True, sign) for cell in unmatched]
    entries += _extra_balls(draw, n, matched, (False, True), sign)
    entries += _extra_balls(draw, n, unmatched, (True,), sign)
    if charges:  # an ordinary charge in a cell with Pressure_Balls only, which stays unmatched
        entries.append(_charge_entry(draw, draw(st.integers(0, n - 1)), draw(_choice(tuple(unmatched))), False))
    return entries


def _no_matched_cell_player(draw: Any, n: int, sign: int, charges: bool) -> list[_Entry]:
    """NO_MATCHED_CELL: Pressure_Balls and Ordinary_Balls in disjoint cells."""
    cells = _subset(draw, _CELL_CODES, 2, 4)
    split = draw(st.integers(1, len(cells) - 1))
    pressure_cells, ordinary_cells = cells[:split], cells[split:]
    entries = [_delivery_entry(draw, draw(st.integers(0, n - 1)), cell, True, sign) for cell in pressure_cells]
    entries += [_delivery_entry(draw, draw(st.integers(0, n - 1)), cell, False, sign) for cell in ordinary_cells]
    entries += _extra_balls(draw, n, pressure_cells, (True,), sign)
    entries += _extra_balls(draw, n, ordinary_cells, (False,), sign)
    if charges:  # a charge of the other class in each kind of cell: neither makes a Matched_Cell
        entries.append(_charge_entry(draw, draw(st.integers(0, n - 1)), draw(_choice(tuple(pressure_cells))), False))
        entries.append(_charge_entry(draw, draw(st.integers(0, n - 1)), draw(_choice(tuple(ordinary_cells))), True))
    return entries


def _one_pressure_match_player(draw: Any, n: int, sign: int, charges: bool) -> list[_Entry]:
    """ONE_PRESSURE_MATCH: Pressure_Balls in one match only, Ordinary_Balls of the same cell in another."""
    pressure_match, other = _subset(draw, range(n), 2, 2)
    cell = draw(_choice(_CELL_CODES))
    entries = [_delivery_entry(draw, pressure_match, cell, True, sign) for _ in range(draw(st.integers(1, 3)))]
    entries += [_delivery_entry(draw, other, cell, False, sign) for _ in range(draw(st.integers(1, 3)))]
    entries += _extra_balls(draw, n, _CELL_CODES, (False,), sign)  # Ordinary_Balls anywhere
    if charges:  # a pressure charge in the other match: a pressure row there, but no Pressure_Ball
        entries.append(_charge_entry(draw, other, draw(_choice(_CELL_CODES)), True))
    return entries


def _charge_only_player(draw: Any, n: int, sign: int, charges: bool) -> list[_Entry]:
    """CHARGE_ONLY: Deliveries of one cell and class in one match, a charge of the other class there, and a charge-only match."""
    delivery_match, charge_match = _subset(draw, range(n), 2, 2)
    cell = draw(_choice(_CELL_CODES))
    pressure = draw(st.booleans())
    entries = [_delivery_entry(draw, delivery_match, cell, pressure, sign) for _ in range(draw(st.integers(1, 3)))]
    entries.append(_charge_entry(draw, delivery_match, cell, not pressure))
    entries.append(_charge_entry(draw, charge_match, draw(_choice(_CELL_CODES)), draw(st.booleans())))
    return entries


def _tied_player(draw: Any, n: int, sign: int, charges: bool) -> list[_Entry]:
    """The TIES player, whom his twin copies: 2 or more Pressure_Matches, an Ordinary_Ball, a Matched_Cell."""
    played = _subset(draw, range(n), 2, _MAX_PLAYER_MATCHES)
    cells = _subset(draw, _CELL_CODES, 1, 2)
    cell = _choice(tuple(cells))
    entries = [_delivery_entry(draw, match, cells[0], True, sign) for match in played[:2]]
    entries.append(_delivery_entry(draw, draw(_choice(tuple(played))), cells[0], False, sign))
    for match in played:
        entries += [
            _delivery_entry(draw, match, draw(cell), draw(st.booleans()), sign)
            for _ in range(draw(_choice(_EXTRA_BALLS)))
        ]
    if charges:
        entries += _random_charges(draw, n, played, cells)
    return entries


_FORCED_PLAYERS = {
    ALL_MATCHED: _all_matched_player,
    UNMATCHED_CELL: _unmatched_cell_player,
    NO_MATCHED_CELL: _no_matched_cell_player,
    ONE_PRESSURE_MATCH: _one_pressure_match_player,
    CHARGE_ONLY: _charge_only_player,
    TIES: _tied_player,
}


def _check_case(case: str, entries: Sequence[_Entry]) -> None:
    """Stop with a strategy bug if a forced player's entries do not have the case's shape."""
    balls = [entry for entry in entries if not entry[3]]
    pressure_cells = {entry[1] for entry in balls if entry[2]}
    ordinary_cells = {entry[1] for entry in balls if not entry[2]}
    matched = pressure_cells & ordinary_cells
    pressure_matches = {entry[0] for entry in balls if entry[2]}
    ball_matches = {entry[0] for entry in balls}
    ball_groups = {entry[:3] for entry in balls}
    charge_groups = {entry[:3] for entry in entries if entry[3]}
    holds = {
        ALL_MATCHED: len(matched) >= 2 and pressure_cells <= ordinary_cells,
        UNMATCHED_CELL: bool(matched) and bool(pressure_cells - ordinary_cells),
        NO_MATCHED_CELL: bool(pressure_cells) and bool(ordinary_cells) and not matched,
        ONE_PRESSURE_MATCH: len(pressure_matches) == 1 and len(ball_matches) >= 2 and bool(matched),
        CHARGE_ONLY: any(group[0] not in ball_matches for group in charge_groups)
        and any(group[0] in ball_matches and group not in ball_groups for group in charge_groups),
        TIES: len(pressure_matches) >= 2 and bool(matched),
    }
    _require(holds[case], f"the {case} player's entries do not have the shape of the case")


@st.composite
def _contributions(
    draw: Any, n_players: int | SearchStrategy[int] | None, n_matches: int | SearchStrategy[int] | None,
    roles: tuple[str, ...], force: frozenset[str], charges: bool, dense: bool,
) -> ContributionCase:
    least = 2 if force & _TWO_MATCH_CASES else 1
    n = _drawn_count(draw, n_matches, "n_matches", least, st.integers(least, max(least, _MAX_DEFAULT_MATCHES)))
    matches = _contribution_matches(draw, n)
    rows: list[tuple[Any, ...]] = []  # (Role, player_id, match index, cell, pressure, charge, value)
    forced: dict[tuple[str, str], int] = {}
    twins: dict[str, tuple[int, int]] = {}
    for role in roles:
        batter = role == BATTER_ROLE
        sign = 1 if batter else -1  # a bowler's NBV mirrors a batter's: his wickets are large and positive
        with_charges = batter and charges
        k = _drawn_count(draw, n_players, "n_players", 1, _DEFAULT_PLAYERS)
        plans = [_random_player(draw, n, sign, with_charges, dense) for _ in range(k)]
        cases = [case for case in FORCED_CASES if case in force and (batter or case not in _BATTER_ONLY_CASES)]
        for case in cases:
            plan = _FORCED_PLAYERS[case](draw, n, sign, with_charges)
            _check_case(case, plan)
            plans.append(plan)
        if TIES in cases:
            plans.append(plans[k + cases.index(TIES)])  # the twin: the same entries under another id
        ids = draw(st.lists(st.integers(1, 3 * len(plans) + 6), min_size=len(plans), max_size=len(plans), unique=True))
        for case, player_id in zip(cases, ids[k:]):
            if case == TIES:
                twins[role] = (player_id, ids[-1])
            else:
                forced[case, role] = player_id
        rows += [(role, player_id, *entry) for player_id, plan in zip(ids, plans) for entry in plan]
    entries = _entry_table(rows, matches)
    frame = contribution_frame(entries, matches)
    match_ids = matches["match_id"].to_numpy(dtype=np.int64)
    contrib = Contributions.from_frame(frame, match_ids)
    return ContributionCase(contrib, frame, entries, matches, match_ids, roles, forced, twins)


#: Multiplicities that match_multiplicities draws for each match: mostly 1, where shrinking goes.
_MULTIPLICITIES = (1, 1, 1, 1, 0, 0, 2, 2, 3, 4)
_MOVES_PER_MATCH = 2  # a resample makes up to 2 x n_matches moves of one copy from all ones


def match_multiplicities(n_matches: int, *, resample: bool = False) -> SearchStrategy[np.ndarray]:
    """One whole-number multiplicity per match, for ``Aggregator.estimates(mult)``; see the module docstring.

    ``n_matches``: the size of the match universe (``Contributions.n_matches``), an int of 0 or
    more. ``resample=True``: a Bootstrap_Resample, whose counts sum to ``n_matches``.
    """
    if isinstance(n_matches, (bool, np.bool_)) or not isinstance(n_matches, (int, np.integer)) or n_matches < 0:
        raise InvalidArgument(f"n_matches must be an int of 0 or more; got {n_matches!r}")
    return _match_multiplicities(int(n_matches), bool(resample))


@st.composite
def _match_multiplicities(draw: Any, n: int, resample: bool) -> np.ndarray:
    if not resample:
        return np.array(draw(st.lists(_choice(_MULTIPLICITIES), min_size=n, max_size=n)), dtype=np.int64)
    counts = np.ones(n, dtype=np.int64)
    for _ in range(draw(st.integers(0, _MOVES_PER_MATCH * n))):
        donors = np.flatnonzero(counts > 0)
        counts[donors[draw(st.integers(0, len(donors) - 1))]] -= 1
        counts[draw(st.integers(0, n - 1))] += 1
    return counts
