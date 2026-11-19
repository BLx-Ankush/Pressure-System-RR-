"""List_Builder: name mapping, the Main_Lists and the Unstop entry checklist (Requirement 25,
criteria 1-3 and 5-9), the Comparison_Lists and activity comparison tables (criteria 13-15), and
the ``final_lists`` Pipeline_Stage (Requirements 25 and 26); design sections "List_Builder
(``lists.py``, ``charts.py``)" and Output files, the stage table (stage 19), and Properties 54
(through ``ranks.compare_rankings``), 57 and 58.

The evidence tables of Requirement 26 are in ``superr.evidence`` and the charts in
``superr.charts``. No function here reads ``config.py``; only ``run_stage`` logs. The
``final_lists`` Pipeline_Stage (``run_stage``, last section) loads the inputs with
``load_list_inputs`` and calls ``main_lists``, ``unresolved``, ``write_lists`` and
``write_checklist``, then ``write_comparison_outputs`` for the activity comparison, and then
writes the evidence tables and charts. The Comparison_Lists reuse ``list_table`` (through
``main_lists``), the Main_Lists' file format and the name rules. Every other function takes
tables and returns tables, text or paths; only ``write_lists``, ``write_checklist``,
``load_list_inputs``, ``write_comparison``, ``clear_comparison``, ``load_comparison_inputs``,
``write_comparison_outputs`` and ``run_stage`` touch files.

Names (``candidate_names``, ``team_names``, ``choose_name``, ``unresolved``)
    A Candidate_Name of a ``player_id`` is a distinct ``player_name`` string of the
    ``competition_team_player_mapping.csv`` rows (the ``load`` table ``mapping``) with that
    ``player_id``. Names are compared exactly: case, spacing and punctuation count, and nothing is
    trimmed or normalised. A missing ``player_name`` (an Empty_Field) is no name, and neither is the
    empty string, which the Data_Loader never produces. A name of spaces only is a name (glossary
    Empty_Field). "Lexicographically smallest" (criteria 7 and 8) is Python string order, that is
    Unicode code point order.

    * ``candidate_names(mapping)``: one row per ``player_id`` and Candidate_Name with the number of
      mapping rows holding the pair (CANDIDATE_COLUMNS), ordered by ``player_id``, then most rows,
      then name. The first row of a ``player_id`` is therefore its criterion 2 or 7 name.
    * ``team_names(mapping)``: the distinct ``team_name`` values of each ``player_id``
      (TEAM_COLUMNS), in string order, for the checklist (criterion 3).
    * ``choose_name(player_id, names, balls, role)`` returns a NameChoice with the ``player_name``
      to write and the rule that chose it (NAME_RULES):

      - exactly one Candidate_Name: that string, unchanged (criterion 2, ``single_candidate``);
      - several: the one in the most mapping rows, ties to the smallest string (criterion 7,
        ``most_mapping_rows``);
      - none: the most frequent ``striker_name`` (batter list) or ``bowler_name`` (bowler list) of
        the ``ball_by_ball.csv`` rows with that ``striker_id`` or ``bowler_id``, ties to the
        smallest string (criterion 8, ``delivery_name``). Every row of the file counts,
        Super_Over_Innings and excluded matches included, because the criterion names the file. A
        missing name does not count. A player with no name in either file raises StageError: there
        is no Unstop_Name to write.

      It also returns the two facts behind the Unresolved_Entry rule: ``n_candidates``, the number
      of Candidate_Names of the ``player_id``, and ``n_holders``, the number of ``player_id``s that
      have the chosen name as a Candidate_Name. ``NameChoice.resolved`` is True exactly when
      ``n_candidates`` is 1 and ``n_holders`` is at most 1.
    * ``unresolved(entries, names, teams)`` applies the same rule to list entries (a ``player_id``
      and its written ``player_name``; ``list_entries`` gives the 20 Main_List entries) and adds the
      checklist details (DETAIL_COLUMNS). An entry is an Unresolved_Entry exactly when its
      ``player_id`` has zero or several Candidate_Names, or its written name is a Candidate_Name of
      more than one ``player_id``. The rule is applied as written, whichever rule chose the name.

Shares (``pressure_shares``)
    The share of a player's Pressure_Balls in a Role that come from IPL matches, Full_Member_Matches
    and Associate_Matches (criterion 5) uses the rating checkpoint's ``contributions``: the sum of
    the unweighted count ``n`` over the player's rows with ``pressure`` True, by ``comp_class``
    (``rating.CompClass``). These are the Pressure_Balls the rating uses (innings 1 and 2 of the
    rated matches), so the total equals ``n_pressure_balls`` of ``player_estimates``, and
    ``share_ipl`` + ``share_full_member`` is the eligibility rule's ``major_share``. A
    Dismissal_Charge row has ``n`` = 0 and adds nothing. Each share is a count divided by the total,
    so it lies in [0, 1], and the three sum to 1 up to float rounding (about 1e-16).

Main_Lists (``main_lists``, ``list_table``, ``validate_list``, ``write_lists``)
    ``list_table(role, ...)`` builds one Role's list (LIST_COLUMNS, one row per rank 1 to 10):

    * ``rank``, ``player_id``: the Eligible_Players at ranks 1 to 10 of the ranking table, in rank
      order (criterion 1). The batter and bowler lists are built independently.
    * ``player_name``: the Unstop_Name chosen by ``choose_name``.
    * ``final_score``: Final_Score in runs per over, from the ranking table.
    * ``ci_low``, ``ci_high``: the 2.5th and 97.5th percentiles of Final_Score over the
      Bootstrap_Resamples in which the player is ranked (runs per over; Requirement 19).
    * ``p_top10``: P_Top10, the share of all Bootstrap_Resamples with rank 10 or better.
    * ``median_rank``: the median rank over the resamples in which the player is ranked; it can end
      in .5. ``ci_low``, ``ci_high`` and ``median_rank`` are blank for a player who is never ranked
      in a resample (not seen as delivered).
    * ``hit_rate``: Hit_Rate, from the ranking table.
    * ``share_ipl``, ``share_full_member``, ``share_associate``: the shares above.

    The ranking table is the rating checkpoint's ``player_estimates`` (RANKING_INPUT_COLUMNS; with
    ``n_pressure_balls`` the share totals are checked against it) and the bootstrap table is the
    bootstrap checkpoint's ``bootstrap`` (BOOTSTRAP_INPUT_COLUMNS). A Role with fewer than LIST_SIZE
    (10) Eligible_Players raises StageError naming the Role and its count (criterion 9). So does a
    ranking that does not give ranks 1 to 10 to ten Eligible_Players once each or ranks a player
    who is not an Eligible_Player, a bootstrap table whose ``main_rank`` and ``main_final_score``
    differ, even in the last bit, from the ranking's ``rank`` and ``final_score``, and a listed
    player with no bootstrap or share row: these inputs do not come from the same Run.
    ``main_lists`` checks both Roles' counts before it builds either list, so its error names every
    short Role.

    ``write_lists(lists, outdir)`` validates both tables with ``validate_list`` before it writes
    anything: exactly LIST_COLUMNS; LIST_SIZE rows with ranks 1 to 10 in order; 10 distinct
    ``player_id``s; a non-empty ``player_name`` on every row; a finite ``final_score``, ``p_top10``,
    ``hit_rate`` and three shares; ``p_top10``, ``hit_rate`` and the shares in [0, 1], the shares
    summing to 1 within SHARE_TOLERANCE (1e-6); and ``ci_low`` at most ``ci_high``. Any failure
    raises ValueError and writes neither file. It then writes each file to ``<file>.tmp`` and only
    when both are written renames them onto ``top10_batters.csv`` and ``top10_bowlers.csv``
    (MAIN_LIST_FILES). The CSVs are UTF-8 with ``\\n`` line ends and a header row; ``rank`` and
    ``player_id`` are integers, every float is written with LIST_FLOAT_FORMAT (10 decimals) and a
    missing value is blank. Ten decimals rather than six: three shares rounded to 6 decimals can
    sum to 1 +- 1.5e-6 (1/3 three times gives 0.999999), outside the 1e-6 of criterion 5, while 10
    decimals keep the written sum within 1.5e-10.

Checklist (``checklist_text``, ``write_checklist``, ``parse_checklist``)
    ``unstop_entry_checklist.md`` (CHECKLIST_FILE) holds a short header on how to use it against
    the Unstop dropdowns, a summary line, and the 20 Main_List entries (criterion 6): the batter
    list, ranks 1 to 10, then the bowler list. No entry comes from a Comparison_List, and no
    timestamp is written, so two Runs on the same inputs give the same bytes. Each entry is one
    line::

        - [ ] Bowler rank 3: `Rashid Khan` (player_id 4722), **unresolved**

    that is, an unchecked Markdown checkbox, the Role, the rank, the ``player_name`` of the list
    file (the Unstop_Name to select) as a code span, the ``player_id`` and ``resolved`` or
    ``**unresolved**``. The code span shows the name exactly: it is fenced with one backtick more
    than the longest run of backticks in the name and padded with one space on each side when
    CommonMark would otherwise strip or misread one. An unresolved entry (criterion 3) is followed by
    indented lines with the reason, every Candidate_Name of its ``player_id`` with its mapping row
    count, every ``player_id`` that has the written name as a Candidate_Name, and the ``team_name``
    values of each of these ``player_id``s (the entry's own first). ``parse_checklist`` reads the
    entry lines back (PARSED_COLUMNS), for the pipeline-output tests.

Inputs (``load_list_inputs``)
    Reads the ``rating`` checkpoint (``player_estimates``, ``contributions``), the ``bootstrap``
    checkpoint (``bootstrap``) and the ``load`` checkpoint (``mapping``, ``ball_by_ball``) and
    returns a ListInputs. The ``final_lists`` stage therefore reads ``load``, ``rating`` and
    ``bootstrap``, and ``activity_comparison`` for the step below.

Comparison_Lists and activity comparison tables (Requirement 25, criteria 13-15)
    Where the Recent_Activity_Rule is on, the ``activity_comparison`` checkpoint holds the
    Activity_Comparison_Ranking of both Roles (``player_estimates``, rule off) and its bootstrap
    (``bootstrap``, whose ``main_rank`` and ``main_final_score`` are the comparison ranking's own),
    and its JSON value ``status`` has ``applies`` true. The four files of the step go to
    ``outputs/final/comparison/`` (``comparison_dir``; COMPARISON_FILES).

    * ``comparison_lists(...)`` (criterion 13) is ``main_lists`` on the comparison checkpoint's
      ``player_estimates`` and ``bootstrap``, with the Main_Lists' ``shares``, ``names`` and
      ``balls``. So each list holds comparison ranks 1 to 10 in order, every value comes from the
      Activity_Comparison_Ranking and its bootstrap, the columns, checks and number format are the
      Main_Lists' (LIST_COLUMNS, LIST_FLOAT_FORMAT), and ``player_name`` follows criteria 2, 7
      and 8. The shares table is the same because the comparison ranks the rating checkpoint's
      contributions, so a player has the same Pressure_Balls in both rankings (list_table checks
      the counts). Files: ``top10_batters_activity_off.csv`` and
      ``top10_bowlers_activity_off.csv`` (COMPARISON_LIST_FILES).
    * ``activity_comparison_table(role, ...)`` (criterion 14) compares the Role's Main_Ranking (the
      rating checkpoint's ``player_estimates``) with its Activity_Comparison_Ranking through
      ``ranks.compare_rankings`` with ``top_n`` LIST_SIZE. Each must be a full-data ranking that
      ranks exactly its Eligible_Players, at least 10, and every Main_Ranking Eligible_Player must
      be one of the comparison's, since turning a rule off removes no player; otherwise the tables
      do not come from the same Run (StageError). The table (ACTIVITY_COLUMNS) has one row of
      ``kind`` ``summary``, then one ``entering`` row per player in the Comparison_List but not in
      the Main_List, in comparison rank order, then one ``leaving`` row per player in the Main_List
      but not in the Comparison_List, in Main rank order. The columns:

      - ``kind`` (``summary``, ``entering`` or ``leaving``) and ``role``;
      - ``n_eligible_main`` and ``n_eligible_comparison``: the Role's Eligible_Player counts in the
        Main_Ranking and in the Activity_Comparison_Ranking; ``overlap``: the number of
        ``player_id``s in both lists, 0 to 10. These three are repeated on every row, so each row
        reads on its own, and the summary row keeps them when no player enters or leaves (as
        delivered, the overlap is 10 in both Roles);
      - ``player_id`` and ``player_name`` (criteria 2, 7 and 8: the name the lists use);
      - ``main_rank`` and ``comparison_rank``: the player's rank in each ranking, blank where the
        player is not an Eligible_Player of that ranking. An entering player has a comparison rank
        from 1 to 10 and a Main rank above 10, or none when only the Recent_Activity_Rule left the
        player out of the Main_Ranking; a leaving player has a Main rank from 1 to 10 and a
        comparison rank above 10;
      - ``latest_calendar_year``: on an entering row, the Calendar_Year of the player's latest
        match, by ``match_date``, with ``is_played`` true in ``playing_eleven.csv``
        (``latest_calendar_years``), blank when there is none. As for the Recent_Activity_Rule,
        every match of the load tables counts and a missing ``is_played`` reads false. Criterion 14
        asks for it on entering rows only, so it is blank on the other rows.

      The summary row leaves the five player columns blank. Every number is a whole number written
      without decimals, and a blank is an empty field. Files: ``activity_comparison_batters.csv``
      and ``activity_comparison_bowlers.csv`` (ACTIVITY_TABLE_FILES).
    * ``write_comparison(outputs, lists=..., tables=...)`` validates both lists (``validate_list``)
      and both tables (``validate_activity_table``), and checks each table's entering and leaving
      rows against its list, before it writes anything. It then writes the four files into
      ``<outputs>/final/comparison/`` through ``<file>.tmp`` and renames them once all four are
      written; the lists get exactly the bytes ``write_lists`` would write.
    * Rule off: where ``status`` has ``applies`` false, the Main_Ranking already has the rule off
      and no comparison exists, so no comparison file is written. Files with these four names left
      by an earlier Run would describe another Config, and a reader or the Report_Builder could
      take them for this Run's, so ``clear_comparison`` deletes them, with any leftover ``.tmp``.
      It deletes nothing else: not ``_stage/``, which holds the ``activity_comparison``
      checkpoint, and no file outside ``outputs/final/comparison/``.
    * ``load_comparison_inputs(store)`` reads ``status`` and, when a comparison applies, the
      comparison checkpoint's ``player_estimates`` and ``bootstrap`` and the ``load`` tables
      ``playing_eleven`` and ``matches`` (ComparisonInputs); it returns None when none applies.
      ``write_comparison_outputs(store, inputs, outputs)`` is the whole step for the
      ``final_lists`` stage: it builds, checks and writes the four files, or clears them, and
      returns a ComparisonOutputs with the lists, the tables, each Role's RankingComparison
      (overlap, entering, leaving, Spearman over the common Eligible_Players) and the paths
      written or deleted. It also checks both Eligible_Player counts against ``status``.

    No function of this step creates, modifies or deletes the Main_List files,
    ``unstop_entry_checklist.md``, the ``outputs/rating/`` tables or the bootstrap table
    (criterion 15).

The ``final_lists`` Pipeline_Stage (``run_stage``)
    Reads the ``load``, ``reconstruct``, ``pressure``, ``value``, ``rating``, ``bootstrap`` and
    ``activity_comparison`` checkpoints. Then:

    1. builds both Main_Lists and the checklist details (``load_list_inputs``, ``main_lists``,
       ``unresolved(list_entries(...))``), and the six evidence tables
       (``evidence.load_evidence_deliveries`` and ``evidence.evidence_tables``), which are
       checked against the rating's counts and the lists before any file is written;
    2. writes the Main_Lists and ``unstop_entry_checklist.md`` to ``outputs/final/``;
    3. runs ``write_comparison_outputs`` (Requirement 25, criteria 13-15) between two snapshots of
       the SHA-256 of every file directly in ``outputs/final/``, ``outputs/rating/`` and
       ``outputs/robustness/`` (GUARDED_FOLDERS): the Main_List files, the checklist, the tables
       of Requirement 18, criterion 15 and Requirement 19, criterion 5, and any evidence file of
       an earlier Run. A file created, changed or deleted between the snapshots stops the Run
       with a StageError, since criterion 15 lets the step write only under
       ``outputs/final/comparison/``;
    4. writes the evidence tables (``evidence.write_evidence``) and the eight charts
       (``charts.write_charts``) to ``outputs/final/`` (Requirement 26, criterion 10);
    5. saves its checkpoint: each Role's Main_List (LIST_TABLES), the checklist details without
       their tuple columns (CHECKLIST_TABLE, CHECKLIST_TABLE_COLUMNS), the six evidence tables
       under their names (``evidence.EVIDENCE_TABLES``) and the JSON value WRITTEN_JSON, which
       holds ``files`` (the path under ``outputs/`` and SHA-256 of every file the stage wrote
       outside ``outputs/final/comparison/``) and ``comparison_guard`` (the snapshot of step 3,
       the same before and after the step). No Comparison_List, activity comparison table or
       other comparison result is saved: those stay under ``outputs/final/comparison/`` only
       (Requirement 25, criterion 15);
    6. records the data-derived constants UNRESOLVED_CONSTANTS (List_Builder): the number of
       Unresolved_Entries of each Role's Main_List, and logs both lists, the checklist summary,
       a warning for each Unresolved_Entry with its Candidate_Names, namesakes and teams, each
       Role's comparison overlap with the players entering and leaving, the evidence and chart
       files with the empty cells of the traditional-stats tables, and the timing.

Errors: ValueError or TypeError, naming the problem, for a missing column, a value of the wrong
type, a repeated row, an unknown Role, or a list, details or activity comparison table that cannot
be written. StageError (exit code 5) for a Role with fewer than 10 Eligible_Players, inputs that do
not hold the same Run's ranking (or rankings), a listed player with no name, an
``activity_comparison`` status without a true or false ``applies``, and a comparison step that
touches a guarded file; a checkpoint file that is missing or changed raises StageError from the
CheckpointStore.
"""

from __future__ import annotations

import math
import os
import re
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import pandas as pd

from superr import activity, charts, evidence
from superr.errors import StageError
from superr.protect import sha256_file
from superr.ranks import ENTERING, LEAVING, RankingComparison, compare_rankings
from superr.rating import CONTRIBUTIONS_TABLE, LIST_SIZE, PLAYER_ESTIMATES_TABLE, RATING_FOLDER, CompClass
from superr.rating import STAGE_NAME as RATING_STAGE
from superr.robustness import BOOTSTRAP_STAGE, BOOTSTRAP_TABLE, ROBUSTNESS_FOLDER
from superr.value import BATTER_ROLE, BOWLER_ROLE, ROLES

if TYPE_CHECKING:
    from superr.pipeline import StageContext
    from superr.store import CheckpointStore

__all__ = [
    "ACTIVITY_COLUMNS",
    "ACTIVITY_KINDS",
    "ACTIVITY_TABLE_FILES",
    "BOOTSTRAP_INPUT_COLUMNS",
    "CANDIDATE_COLUMNS",
    "CHECKLIST_FILE",
    "CHECKLIST_TABLE",
    "CHECKLIST_TABLE_COLUMNS",
    "COMPARISON_FILES",
    "COMPARISON_FOLDER",
    "COMPARISON_LIST_FILES",
    "DELIVERY_NAME",
    "DETAIL_COLUMNS",
    "ENTRY_COLUMNS",
    "FINAL_FOLDER",
    "GUARDED_FOLDERS",
    "LIST_COLUMNS",
    "LIST_FLOAT_FORMAT",
    "LIST_TABLES",
    "MAIN_LIST_FILES",
    "MOST_MAPPING_ROWS",
    "NAME_COLUMNS",
    "NAME_RULES",
    "PARSED_COLUMNS",
    "RANKING_INPUT_COLUMNS",
    "ROLE_LABELS",
    "SHARE_COLUMNS",
    "SHARE_TOLERANCE",
    "SINGLE_CANDIDATE",
    "STAGE_NAME",
    "SUMMARY_KIND",
    "TEAM_COLUMNS",
    "UNRESOLVED_CONSTANTS",
    "WRITTEN_JSON",
    "ComparisonInputs",
    "ComparisonOutputs",
    "ListInputs",
    "NameChoice",
    "activity_comparison_table",
    "candidate_names",
    "checklist_text",
    "choose_name",
    "clear_comparison",
    "comparison_dir",
    "comparison_lists",
    "eligible_counts",
    "latest_calendar_years",
    "list_entries",
    "list_table",
    "load_comparison_inputs",
    "load_list_inputs",
    "main_lists",
    "parse_checklist",
    "pressure_shares",
    "run_stage",
    "team_names",
    "unresolved",
    "validate_activity_table",
    "validate_list",
    "write_checklist",
    "write_comparison",
    "write_comparison_outputs",
    "write_lists",
]

# ================================================================================ constants

#: Folder under the outputs folder that holds the List_Builder's files.
FINAL_FOLDER = "final"

#: The Main_List file of each Role (glossary Main_List; Requirement 25, criterion 5).
MAIN_LIST_FILES: Mapping[str, str] = {BATTER_ROLE: "top10_batters.csv", BOWLER_ROLE: "top10_bowlers.csv"}

#: The Unstop entry checklist in the final folder (Requirement 25, criteria 3 and 6).
CHECKLIST_FILE = "unstop_entry_checklist.md"

#: Columns of a list table and file, in order (module docstring, "Main_Lists").
LIST_COLUMNS: tuple[str, ...] = (
    "rank", "player_id", "player_name", "final_score", "ci_low", "ci_high", "p_top10", "median_rank", "hit_rate",
    "share_ipl", "share_full_member", "share_associate",
)  # fmt: skip

#: The share column of each CompClass code, and the count column behind it.
_SHARES: Mapping[int, str] = {
    CompClass.IPL: "share_ipl", CompClass.FULL_MEMBER: "share_full_member", CompClass.ASSOCIATE: "share_associate",
}  # fmt: skip
_COUNTS: Mapping[int, str] = {
    CompClass.IPL: "n_ipl", CompClass.FULL_MEMBER: "n_full_member", CompClass.ASSOCIATE: "n_associate",
}  # fmt: skip

#: Columns of the pressure_shares table: one row per Role and player with at least one Pressure_Ball.
SHARE_COLUMNS: tuple[str, ...] = ("role", "player_id", "n_pressure_balls", *_COUNTS.values(), *_SHARES.values())

#: Columns of the candidate_names table: one row per player_id and Candidate_Name.
CANDIDATE_COLUMNS: tuple[str, ...] = ("player_id", "player_name", "n_rows")

#: Columns of the team_names table: one row per player_id and distinct team_name.
TEAM_COLUMNS: tuple[str, ...] = ("player_id", "team_name")

#: The ball_by_ball id and name columns of each Role's fallback name (Requirement 25, criterion 8).
NAME_COLUMNS: Mapping[str, tuple[str, str]] = {
    BATTER_ROLE: ("striker_id", "striker_name"), BOWLER_ROLE: ("bowler_id", "bowler_name"),
}  # fmt: skip

#: The rules that choose a player_name (Requirement 25, criteria 2, 7 and 8).
SINGLE_CANDIDATE = "single_candidate"
MOST_MAPPING_ROWS = "most_mapping_rows"
DELIVERY_NAME = "delivery_name"
NAME_RULES: tuple[str, ...] = (SINGLE_CANDIDATE, MOST_MAPPING_ROWS, DELIVERY_NAME)

#: Columns of the list_entries table: the 20 Main_List entries in checklist order.
ENTRY_COLUMNS: tuple[str, ...] = ("role", "rank", "player_id", "player_name")

#: Columns that unresolved adds to its entries:
#: ``resolved`` (bool; False for an Unresolved_Entry), ``n_candidate_names`` and ``n_name_holders``
#: (int: the Candidate_Names of the player_id, and the player_ids holding the written name),
#: ``candidate_names`` and ``candidate_rows`` (tuples: each Candidate_Name of the player_id and its
#: mapping rows, most rows first, then name), ``name_holders`` (tuple of the player_ids with the
#: written name as a Candidate_Name, ascending), ``listed_ids`` (the entry's player_id, then the
#: other holders) and ``listed_teams`` (a tuple of the team_name values of each listed player_id).
DETAIL_COLUMNS: tuple[str, ...] = (
    "resolved", "n_candidate_names", "n_name_holders", "candidate_names", "candidate_rows", "name_holders",
    "listed_ids", "listed_teams",
)  # fmt: skip

#: Columns of parse_checklist: the entry fields, then whether the box is ticked.
PARSED_COLUMNS: tuple[str, ...] = (*ENTRY_COLUMNS, "resolved", "checked")

#: Columns that list_table reads from the ranking and bootstrap tables.
RANKING_INPUT_COLUMNS: tuple[str, ...] = ("role", "player_id", "eligible", "rank", "final_score", "hit_rate")
BOOTSTRAP_INPUT_COLUMNS: tuple[str, ...] = (
    "role", "player_id", "main_rank", "main_final_score", "ci_low", "ci_high", "median_rank", "p_top10",
)  # fmt: skip

#: Float format of the list files: at least the 6 decimals of criterion 5 (module docstring).
LIST_FLOAT_FORMAT = "%.10f"

#: The three shares of a listed player sum to 1 within this (Requirement 25, criterion 5).
SHARE_TOLERANCE = 1e-6

#: The Role as the checklist shows it, and the heading of each Role's section.
ROLE_LABELS: Mapping[str, str] = {BATTER_ROLE: "Batter", BOWLER_ROLE: "Bowler"}
_SECTION_TITLES: Mapping[str, str] = {BATTER_ROLE: "Batters", BOWLER_ROLE: "Bowlers"}
_LABEL_ROLES: Mapping[str, str] = {label: role for role, label in ROLE_LABELS.items()}

_INTEGER_COLUMNS: tuple[str, ...] = ("rank", "player_id")
_FLOAT_COLUMNS: tuple[str, ...] = LIST_COLUMNS[3:]
_REQUIRED_VALUES: tuple[str, ...] = ("final_score", "p_top10", "hit_rate", *_SHARES.values())
_UNIT_VALUES: tuple[str, ...] = ("p_top10", "hit_rate", *_SHARES.values())
_LOAD_STAGE = "load"
_MAPPING_TABLE = "mapping"
_BALLS_TABLE = "ball_by_ball"
_MAPPING_FILE = "competition_team_player_mapping.csv"
_BALLS_FILE = "ball_by_ball.csv"
_TMP = ".tmp"
_RESOLVED = "resolved"
_UNRESOLVED = "**unresolved**"
_DETAIL_INDENT = " " * 4
_ENTRY_LINE = re.compile(
    r"- \[(?P<box>[ xX])\] (?P<label>Batter|Bowler) rank (?P<rank>\d+): (?P<fence>`+)(?P<code>.*?)(?P=fence) "
    r"\(player_id (?P<player_id>-?\d+)\), (?P<status>resolved|\*\*unresolved\*\*)"
)


# ========================================================================= argument checks

def _require(frame: Any, columns: Sequence[str], what: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{what} must be a DataFrame; got {type(frame).__name__}")
    absent = [column for column in dict.fromkeys(columns) if column not in frame.columns]
    if absent:
        raise ValueError(f"{what} lacks the column(s) {', '.join(absent)}")


def _role(role: Any) -> str:
    if not isinstance(role, str) or role not in ROLES:
        raise ValueError(f"role must be one of {', '.join(ROLES)}; got {role!r}")
    return role


def _player_id(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"player_id must be a whole number; got {value!r}")
    return int(value)


def _is_missing(value: Any) -> bool:
    return value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value))


def _texts(values: pd.Series, what: str) -> list[str | None]:
    """Each value as a str, or None where it is missing or the empty string."""
    out: list[str | None] = []
    for value in values.tolist():
        if _is_missing(value):
            out.append(None)
        elif isinstance(value, str):
            out.append(str(value) if value else None)
        else:
            raise TypeError(f"{what} must hold text; found {value!r}")
    return out


def _whole(values: pd.Series, what: str) -> np.ndarray:
    dtype = values.dtype
    if pd.api.types.is_bool_dtype(dtype) or not pd.api.types.is_numeric_dtype(dtype):
        raise TypeError(f"{what} must hold whole numbers; got dtype {dtype}")
    missing = int(values.isna().sum())
    if missing:
        raise ValueError(f"{what} has {missing:,} missing value(s)")
    if pd.api.types.is_integer_dtype(dtype):
        return values.to_numpy(dtype=np.int64)
    floats = values.to_numpy(dtype=np.float64)
    if not (np.isfinite(floats).all() and (np.floor(floats) == floats).all()):
        raise ValueError(f"{what} must hold whole numbers")
    return floats.astype(np.int64)


def _numbers(values: pd.Series, what: str) -> np.ndarray:
    """float64 values, NaN where missing."""
    dtype = values.dtype
    if pd.api.types.is_bool_dtype(dtype) or not pd.api.types.is_numeric_dtype(dtype):
        raise TypeError(f"{what} must hold numbers; got dtype {dtype}")
    return values.to_numpy(dtype=np.float64, na_value=np.nan)


def _flags(values: pd.Series, what: str) -> np.ndarray:
    if not pd.api.types.is_bool_dtype(values.dtype):
        raise TypeError(f"{what} must hold True or False; got dtype {values.dtype}")
    missing = int(values.isna().sum())
    if missing:
        raise ValueError(f"{what} has {missing:,} missing value(s)")
    return values.to_numpy(dtype=bool)


def _role_mask(frame: pd.DataFrame, role: str, what: str) -> np.ndarray:
    """True on the rows of ``frame`` whose ``role`` is ``role``; ValueError for a value that is not a Role."""
    roles = _texts(frame["role"], f"{what} column 'role'")
    unknown = sorted({str(value) for value in roles if value not in ROLES})
    if unknown:
        raise ValueError(f"{what} column 'role' holds {', '.join(unknown)}, which is not a Role")
    return np.fromiter((value == role for value in roles), dtype=bool, count=len(roles))


def _role_rows(frame: pd.DataFrame, role: str, what: str) -> pd.DataFrame:
    return frame.loc[_role_mask(frame, role, what)].reset_index(drop=True)


def _unique(ids: np.ndarray, what: str) -> None:
    values, counts = np.unique(ids, return_counts=True)
    if (counts > 1).any():
        raise ValueError(f"{what} holds player_id {values[counts > 1][0]} more than once")


def _positions(values: pd.Series, wanted: np.ndarray, what: str) -> np.ndarray:
    """The row of each wanted player_id in ``values``; StageError when one has no row."""
    ids = _whole(values, f"{what} column 'player_id'")
    _unique(ids, what)
    at = pd.Index(ids).get_indexer(wanted)
    if (at < 0).any():
        absent = ", ".join(str(pid) for pid in wanted[at < 0])
        raise StageError(f"{what} has no row for the listed player_id(s) {absent}; it does not come from the same Run")
    return at


def _count(n: int, noun: str) -> str:
    return f"{n:,} {noun}" if n == 1 else f"{n:,} {noun}s"


# =================================================================================== names

def candidate_names(mapping: pd.DataFrame) -> pd.DataFrame:
    """The Candidate_Names of every ``player_id`` in the ``mapping`` table (CANDIDATE_COLUMNS).

    ``mapping`` needs ``player_id`` (whole numbers, none missing) and ``player_name``. Returns one
    row per distinct (``player_id``, ``player_name``) pair with a name: ``player_id`` (int64),
    ``player_name`` (str, the exact string) and ``n_rows`` (int64, the mapping rows that hold the
    pair), ordered by ``player_id``, then ``n_rows`` descending, then ``player_name``.
    """
    _require(mapping, ("player_id", "player_name"), "mapping")
    ids = _whole(mapping["player_id"], "mapping column 'player_id'")
    names = _texts(mapping["player_name"], "mapping column 'player_name'")
    counts = Counter((pid, name) for pid, name in zip(ids.tolist(), names) if name is not None)
    ordered = sorted(counts.items(), key=lambda item: (item[0][0], -item[1], item[0][1]))
    return pd.DataFrame(
        {
            "player_id": np.asarray([pid for (pid, _), _ in ordered], dtype=np.int64),
            "player_name": pd.array([name for (_, name), _ in ordered], dtype="str"),
            "n_rows": np.asarray([n for _, n in ordered], dtype=np.int64),
        },
        columns=list(CANDIDATE_COLUMNS),
    )


def team_names(mapping: pd.DataFrame) -> pd.DataFrame:
    """The distinct ``team_name`` values of every ``player_id`` in the ``mapping`` table (TEAM_COLUMNS).

    One row per distinct (``player_id``, ``team_name``) pair with a team name, ordered by
    ``player_id`` and then ``team_name``.
    """
    _require(mapping, ("player_id", "team_name"), "mapping")
    ids = _whole(mapping["player_id"], "mapping column 'player_id'")
    teams = _texts(mapping["team_name"], "mapping column 'team_name'")
    pairs = sorted({(pid, team) for pid, team in zip(ids.tolist(), teams) if team is not None})
    return pd.DataFrame(
        {
            "player_id": np.asarray([pid for pid, _ in pairs], dtype=np.int64),
            "team_name": pd.array([team for _, team in pairs], dtype="str"),
        },
        columns=list(TEAM_COLUMNS),
    )


class _NameIndex(NamedTuple):
    by_id: dict[int, list[tuple[str, int]]]  # player_id -> (Candidate_Name, mapping rows), most rows first, then name
    by_name: dict[str, list[int]]  # Candidate_Name -> the player_ids that have it, ascending


def _name_index(names: pd.DataFrame) -> _NameIndex:
    _require(names, CANDIDATE_COLUMNS, "names")
    ids = _whole(names["player_id"], "names column 'player_id'").tolist()
    texts = _texts(names["player_name"], "names column 'player_name'")
    rows = _whole(names["n_rows"], "names column 'n_rows'").tolist()
    by_id: dict[int, list[tuple[str, int]]] = {}
    by_name: dict[str, list[int]] = {}
    hint = "build names with candidate_names"
    for pid, text, n in zip(ids, texts, rows):
        if text is None:
            raise ValueError(f"names gives player_id {pid} a missing or empty player_name; {hint}")
        if n < 1:
            raise ValueError(f"names gives player_id {pid} and {text!r} {n} mapping rows; a Candidate_Name has 1+")
        by_id.setdefault(pid, []).append((text, n))
        by_name.setdefault(text, []).append(pid)
    for pid, pairs in by_id.items():
        if len({text for text, _ in pairs}) != len(pairs):
            raise ValueError(f"names holds a player_name of player_id {pid} more than once; {hint}")
        pairs.sort(key=lambda pair: (-pair[1], pair[0]))
    for holders in by_name.values():
        holders.sort()
    return _NameIndex(by_id, by_name)


def _team_index(teams: pd.DataFrame) -> dict[int, tuple[str, ...]]:
    _require(teams, TEAM_COLUMNS, "teams")
    ids = _whole(teams["player_id"], "teams column 'player_id'").tolist()
    texts = _texts(teams["team_name"], "teams column 'team_name'")
    grouped: dict[int, set[str]] = {}
    for pid, team in zip(ids, texts):
        if team is not None:
            grouped.setdefault(pid, set()).add(team)
    return {pid: tuple(sorted(values)) for pid, values in grouped.items()}


class NameChoice(NamedTuple):
    """The ``player_name`` that ``choose_name`` writes for one listed player, and the facts behind it."""

    player_name: str  # the exact string to write
    rule: str  # NAME_RULES: the criterion that chose it
    n_candidates: int  # Candidate_Names of the player_id
    n_holders: int  # player_ids that have player_name as a Candidate_Name

    @property
    def resolved(self) -> bool:
        """False for an Unresolved_Entry: not exactly one Candidate_Name, or a name of several player_ids."""
        return self.n_candidates == 1 and self.n_holders <= 1


def _delivery_name(player_id: int, balls: pd.DataFrame, role: str) -> str:
    """The most frequent ``striker_name`` or ``bowler_name`` of ``player_id`` (ties: smallest string)."""
    id_column, name_column = NAME_COLUMNS[role]
    _require(balls, (id_column, name_column), "balls")
    ids = _numbers(balls[id_column], f"balls column {id_column!r}")
    texts = _texts(balls.loc[ids == player_id, name_column], f"balls column {name_column!r}")
    counts = Counter(text for text in texts if text is not None)
    if not counts:
        raise StageError(
            f"player_id {player_id} has no Candidate_Name in {_MAPPING_FILE} and no {name_column} in {_BALLS_FILE}, "
            f"so the {role} list has no player_name to write for it (Requirement 25, criteria 2, 7 and 8)"
        )
    return min(counts.items(), key=lambda item: (-item[1], item[0]))[0]


def _choose(player_id: int, index: _NameIndex, balls: pd.DataFrame, role: str) -> NameChoice:
    own = index.by_id.get(player_id, [])
    if own:
        name, rule = own[0][0], (SINGLE_CANDIDATE if len(own) == 1 else MOST_MAPPING_ROWS)
    else:
        name, rule = _delivery_name(player_id, balls, role), DELIVERY_NAME
    return NameChoice(name, rule, len(own), len(index.by_name.get(name, ())))


def choose_name(player_id: int, names: pd.DataFrame, balls: pd.DataFrame, role: str) -> NameChoice:
    """The ``player_name`` of one listed player (Requirement 25, criteria 2, 7 and 8; module docstring).

    ``names`` is ``candidate_names(mapping)``; ``balls`` holds the ``ball_by_ball`` columns of
    NAME_COLUMNS[role] and is read only when ``player_id`` has no Candidate_Name; ``role`` is
    ``"batter"`` or ``"bowler"``. Raises StageError when the player has no name in either table.
    """
    return _choose(_player_id(player_id), _name_index(names), balls, _role(role))


def unresolved(entries: pd.DataFrame, names: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """``entries`` with the Unresolved_Entry flag and the checklist details (DETAIL_COLUMNS).

    ``entries`` needs ``player_id`` and the written ``player_name``; its other columns (for
    example ``role`` and ``rank`` from ``list_entries``) are kept in front. ``names`` is
    ``candidate_names(mapping)`` and ``teams`` is ``team_names(mapping)``. ``resolved`` is False
    exactly when the ``player_id`` has zero or several Candidate_Names, or the written name is a
    Candidate_Name of more than one ``player_id``. Returns a new table on a RangeIndex; the tuple
    columns are Python tuples (object dtype).
    """
    _require(entries, ("player_id", "player_name"), "entries")
    clash = [column for column in DETAIL_COLUMNS if column in entries.columns]
    if clash:
        raise ValueError(f"entries already has the column(s) {', '.join(clash)}")
    index = _name_index(names)
    lookup = _team_index(teams)
    ids = _whole(entries["player_id"], "entries column 'player_id'").tolist()
    written = _texts(entries["player_name"], "entries column 'player_name'")
    details: dict[str, list[Any]] = {column: [] for column in DETAIL_COLUMNS}
    for pid, name in zip(ids, written):
        if name is None:
            raise ValueError(f"entries gives player_id {pid} a missing or empty player_name")
        own = index.by_id.get(pid, [])
        holders = tuple(index.by_name.get(name, ()))
        listed = (pid, *(other for other in holders if other != pid))
        values = {
            "resolved": len(own) == 1 and len(holders) <= 1,
            "n_candidate_names": len(own),
            "n_name_holders": len(holders),
            "candidate_names": tuple(text for text, _ in own),
            "candidate_rows": tuple(n for _, n in own),
            "name_holders": holders,
            "listed_ids": listed,
            "listed_teams": tuple(lookup.get(other, ()) for other in listed),
        }
        for column, value in values.items():
            details[column].append(value)
    out = entries.reset_index(drop=True).copy()
    out["resolved"] = np.asarray(details["resolved"], dtype=bool)
    out["n_candidate_names"] = np.asarray(details["n_candidate_names"], dtype=np.int64)
    out["n_name_holders"] = np.asarray(details["n_name_holders"], dtype=np.int64)
    for column in DETAIL_COLUMNS[3:]:
        out[column] = pd.Series(details[column], index=out.index, dtype=object)
    return out


# ================================================================================== shares

def pressure_shares(contributions: pd.DataFrame) -> pd.DataFrame:
    """Each player's unweighted Pressure_Ball counts and shares by match class (SHARE_COLUMNS).

    ``contributions`` is the rating checkpoint's table (``role``, ``player_id``, ``pressure``,
    ``n``, ``comp_class``). Returns one row per Role and player with at least one Pressure_Ball,
    ordered by Role (ROLES order) and ``player_id``: ``n_pressure_balls``, ``n_ipl``,
    ``n_full_member`` and ``n_associate`` (int64), and ``share_ipl``, ``share_full_member`` and
    ``share_associate`` (float64, each count divided by ``n_pressure_balls``).
    """
    _require(contributions, ("role", "player_id", "pressure", "n", "comp_class"), "contributions")
    roles = _texts(contributions["role"], "contributions column 'role'")
    codes = {role: code for code, role in enumerate(ROLES)}
    unknown = sorted({str(role) for role in roles if role not in codes})
    if unknown:
        raise ValueError(f"contributions column 'role' holds {', '.join(unknown)}, which is not a Role")
    role_code = np.fromiter((codes[str(role)] for role in roles), dtype=np.int64, count=len(roles))
    ids = _whole(contributions["player_id"], "contributions column 'player_id'")
    pressure = _flags(contributions["pressure"], "contributions column 'pressure'")
    n = _whole(contributions["n"], "contributions column 'n'")
    comp = _whole(contributions["comp_class"], "contributions column 'comp_class'")
    classes = np.asarray([int(code) for code in CompClass], dtype=np.int64)
    if (n < 0).any():
        raise ValueError("contributions column 'n' holds a negative count")
    if not np.isin(comp, classes).all():
        raise ValueError("contributions column 'comp_class' holds a value that is not a CompClass code")
    keep = pressure & (n > 0)
    # One row per (Role, player_id) in sorted order, and the Pressure_Ball count in each match class.
    pairs, inverse = np.unique(np.stack([role_code[keep], ids[keep]], axis=1), axis=0, return_inverse=True)
    matrix = np.zeros((len(pairs), len(classes)), dtype=np.int64)
    np.add.at(matrix, (inverse.reshape(-1), np.searchsorted(classes, comp[keep])), n[keep])
    total = matrix.sum(axis=1)
    table = pd.DataFrame(
        {
            "role": pd.array([ROLES[code] for code in pairs[:, 0].tolist()], dtype="str"),
            "player_id": pairs[:, 1].astype(np.int64),
            "n_pressure_balls": total,
        }
    )
    for column, code in enumerate(classes.tolist()):
        table[_COUNTS[code]] = matrix[:, column]
    for column, code in enumerate(classes.tolist()):
        table[_SHARES[code]] = matrix[:, column].astype(np.float64) / total
    return table.loc[:, list(SHARE_COLUMNS)]


# =============================================================================== the lists

def eligible_counts(ranking: pd.DataFrame) -> dict[str, int]:
    """The number of Eligible_Players (rows with ``eligible`` True) of each Role in ``ranking``."""
    _require(ranking, ("role", "eligible"), "ranking")
    eligible = _flags(ranking["eligible"], "ranking column 'eligible'")
    return {role: int((eligible & _role_mask(ranking, role, "ranking")).sum()) for role in ROLES}


def _short_message(short: Mapping[str, int]) -> str:
    parts = " and ".join(f"the {role} Role has {count:,}" for role, count in short.items())
    return (
        f"A top-10 list needs {LIST_SIZE} Eligible_Players, but {parts} (Requirement 25, criterion 9); "
        "neither top-10 file is written"
    )


def list_table(
    role: str,
    *,
    ranking: pd.DataFrame,
    bootstrap: pd.DataFrame,
    shares: pd.DataFrame,
    names: pd.DataFrame,
    balls: pd.DataFrame,
) -> pd.DataFrame:
    """One Role's top-10 list (LIST_COLUMNS, ranks 1 to 10 in order; module docstring, "Main_Lists").

    ``ranking``: one row per player and Role with RANKING_INPUT_COLUMNS (the rating checkpoint's
    ``player_estimates`` for a Main_List); ``bootstrap``: BOOTSTRAP_INPUT_COLUMNS of the same
    ranking's bootstrap; ``shares``: ``pressure_shares``; ``names``: ``candidate_names``; ``balls``:
    the ``ball_by_ball`` name columns. Raises StageError for a Role with fewer than LIST_SIZE
    Eligible_Players and for inputs that do not hold the same ranking.
    """
    role = _role(role)
    _require(ranking, RANKING_INPUT_COLUMNS, "ranking")
    _require(bootstrap, BOOTSTRAP_INPUT_COLUMNS, "bootstrap")
    _require(shares, SHARE_COLUMNS, "shares")
    rows = _role_rows(ranking, role, "ranking")
    eligible = _flags(rows["eligible"], "ranking column 'eligible'")
    if int(eligible.sum()) < LIST_SIZE:
        raise StageError(_short_message({role: int(eligible.sum())}))
    ids = _whole(rows["player_id"], "ranking column 'player_id'")
    _unique(ids, f"the {role} rows of ranking")
    ranks = _numbers(rows["rank"], "ranking column 'rank'")
    stray = np.isfinite(ranks) & ~eligible
    if stray.any():
        raise StageError(
            f"the {role} ranking ranks {_count(int(stray.sum()), 'player')} who are not Eligible_Players, for "
            f"example player_id {ids[stray][0]}"
        )
    listed = np.flatnonzero(eligible & (ranks <= LIST_SIZE))
    listed = listed[np.argsort(ranks[listed], kind="stable")]
    if not np.array_equal(ranks[listed], np.arange(1, LIST_SIZE + 1, dtype=np.float64)):
        found = ", ".join(f"{rank:g}" for rank in ranks[listed])
        raise StageError(
            f"the {role} ranking does not give each rank from 1 to {LIST_SIZE} to exactly one Eligible_Player "
            f"(Requirement 25, criterion 1); ranks found: {found or 'none'}"
        )
    top_ids = ids[listed]
    final_score = _numbers(rows["final_score"], "ranking column 'final_score'")[listed]

    boot = _role_rows(bootstrap, role, "bootstrap")
    at_boot = _positions(boot["player_id"], top_ids, f"the {role} rows of the bootstrap table")
    main_rank = _numbers(boot["main_rank"], "bootstrap column 'main_rank'")[at_boot]
    main_score = _numbers(boot["main_final_score"], "bootstrap column 'main_final_score'")[at_boot]
    if not (np.array_equal(main_rank, ranks[listed]) and np.array_equal(main_score, final_score)):
        raise StageError(
            f"the bootstrap table does not hold the {role} ranking's ranks and Final_Scores for its top "
            f"{LIST_SIZE}; it does not come from the same Run"
        )

    share = _role_rows(shares, role, "shares")
    at_share = _positions(share["player_id"], top_ids, f"the {role} rows of the shares table")
    if "n_pressure_balls" in rows.columns:
        expected = _whole(rows["n_pressure_balls"], "ranking column 'n_pressure_balls'")[listed]
        counted = _whole(share["n_pressure_balls"], "shares column 'n_pressure_balls'")[at_share]
        if not np.array_equal(expected, counted):
            raise StageError(
                f"the shares table counts other Pressure_Balls than the {role} ranking for its top {LIST_SIZE}; "
                "it does not come from the same Run"
            )

    index = _name_index(names)
    chosen = [_choose(int(pid), index, balls, role).player_name for pid in top_ids.tolist()]
    table = pd.DataFrame(
        {
            "rank": np.arange(1, LIST_SIZE + 1, dtype=np.int64),
            "player_id": top_ids.astype(np.int64),
            "player_name": pd.array(chosen, dtype="str"),
            "final_score": final_score,
            "ci_low": _numbers(boot["ci_low"], "bootstrap column 'ci_low'")[at_boot],
            "ci_high": _numbers(boot["ci_high"], "bootstrap column 'ci_high'")[at_boot],
            "p_top10": _numbers(boot["p_top10"], "bootstrap column 'p_top10'")[at_boot],
            "median_rank": _numbers(boot["median_rank"], "bootstrap column 'median_rank'")[at_boot],
            "hit_rate": _numbers(rows["hit_rate"], "ranking column 'hit_rate'")[listed],
            **{
                column: _numbers(share[column], f"shares column {column!r}")[at_share]
                for column in _SHARES.values()
            },
        },
        columns=list(LIST_COLUMNS),
    )
    validate_list(table, role)
    return table


def main_lists(
    *,
    ranking: pd.DataFrame,
    bootstrap: pd.DataFrame,
    shares: pd.DataFrame,
    names: pd.DataFrame,
    balls: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Both Roles' Main_Lists, ``{role: list_table(role, ...)}`` (Requirement 25, criteria 1, 2, 5, 7-9).

    Checks both Roles' Eligible_Player counts first: a short Role raises StageError naming every
    short Role and its count, before either list is built (criterion 9).
    """
    counts = eligible_counts(ranking)
    short = {role: count for role, count in counts.items() if count < LIST_SIZE}
    if short:
        raise StageError(_short_message(short))
    return {
        role: list_table(role, ranking=ranking, bootstrap=bootstrap, shares=shares, names=names, balls=balls)
        for role in ROLES
    }


def validate_list(table: pd.DataFrame, role: str) -> None:
    """Raise ValueError, listing every problem, unless ``table`` is a writable list of ``role``.

    The rules are in the module docstring ("Main_Lists", ``write_lists``).
    """
    role = _role(role)
    what = f"the {role} list"
    if not isinstance(table, pd.DataFrame):
        raise TypeError(f"{what} must be a DataFrame; got {type(table).__name__}")
    if list(table.columns) != list(LIST_COLUMNS):
        raise ValueError(
            f"{what} must have the columns {', '.join(LIST_COLUMNS)} in that order; got "
            f"{', '.join(map(str, table.columns))}"
        )
    problems: list[str] = []
    if len(table) != LIST_SIZE:
        problems.append(f"it has {_count(len(table), 'row')}, not {LIST_SIZE}")
    for column in _INTEGER_COLUMNS:
        if not pd.api.types.is_integer_dtype(table[column].dtype) or table[column].isna().any():
            problems.append(f"column {column!r} must hold whole numbers in an integer dtype; got {table[column].dtype}")
    for column in _FLOAT_COLUMNS:
        if not pd.api.types.is_float_dtype(table[column].dtype):
            problems.append(f"column {column!r} must have a float dtype; got {table[column].dtype}")
    if problems:
        raise ValueError(f"{what} cannot be written: {'; '.join(problems)}")
    ranks = table["rank"].to_numpy(dtype=np.int64)
    if not np.array_equal(ranks, np.arange(1, len(table) + 1)):
        problems.append(f"column 'rank' must run from 1 in order; got {ranks.tolist()}")
    ids = table["player_id"].to_numpy(dtype=np.int64)
    if len(np.unique(ids)) != len(ids):
        problems.append("column 'player_id' repeats a player")
    if any(name is None for name in _texts(table["player_name"], f"{what} column 'player_name'")):
        problems.append("column 'player_name' has a missing or empty name")
    values = {column: table[column].to_numpy(dtype=np.float64, na_value=np.nan) for column in _FLOAT_COLUMNS}
    for column in _REQUIRED_VALUES:
        if not np.isfinite(values[column]).all():
            problems.append(f"column {column!r} has a missing or infinite value")
    for column in ("ci_low", "ci_high", "median_rank"):
        if np.isinf(values[column]).any():
            problems.append(f"column {column!r} has an infinite value")
    for column in _UNIT_VALUES:
        if ((values[column] < 0.0) | (values[column] > 1.0)).any():
            problems.append(f"column {column!r} has a value outside [0, 1]")
    total = sum(values[column] for column in _SHARES.values())
    if (np.abs(total - 1.0) > SHARE_TOLERANCE).any():
        problems.append(f"the three shares of a row do not sum to 1 within {SHARE_TOLERANCE:g}")
    both = np.isfinite(values["ci_low"]) & np.isfinite(values["ci_high"])
    if (values["ci_low"][both] > values["ci_high"][both]).any():
        problems.append("ci_low is above ci_high on a row")
    if problems:
        raise ValueError(f"{what} cannot be written: {'; '.join(problems)}")


def _checked_lists(lists: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Both Roles' list tables, validated, as plain numpy-typed copies ready to write."""
    if not isinstance(lists, Mapping):
        raise TypeError(f"lists must map each Role to its list table; got {type(lists).__name__}")
    if set(lists) != set(ROLES):
        raise ValueError(f"lists must hold exactly the Roles {', '.join(ROLES)}; got {', '.join(map(str, lists))}")
    for role in ROLES:
        validate_list(lists[role], role)
    out = {}
    for role in ROLES:
        table = lists[role]
        out[role] = pd.DataFrame(
            {
                "rank": table["rank"].to_numpy(dtype=np.int64),
                "player_id": table["player_id"].to_numpy(dtype=np.int64),
                "player_name": np.asarray([str(name) for name in table["player_name"].tolist()], dtype=object),
                **{column: table[column].to_numpy(dtype=np.float64, na_value=np.nan) for column in _FLOAT_COLUMNS},
            },
            columns=list(LIST_COLUMNS),
        )
    return out


def _write_files(writers: Mapping[Path, Callable[[Path], object]]) -> None:
    """Write every file to ``<file>.tmp``; only when all are written, rename each onto its name."""
    pending: list[tuple[Path, Path]] = []
    try:
        for target, write in writers.items():
            temporary = target.with_name(target.name + _TMP)
            pending.append((temporary, target))
            write(temporary)
        for temporary, target in pending:
            os.replace(temporary, target)
    finally:
        for temporary, _ in pending:
            temporary.unlink(missing_ok=True)


def write_lists(
    lists: Mapping[str, pd.DataFrame], outdir: Path | str, files: Mapping[str, str] = MAIN_LIST_FILES
) -> dict[str, Path]:
    """Write both Roles' lists to ``outdir`` and return ``{role: path}`` (Requirement 25, criteria 5 and 9).

    ``lists`` maps each Role to its list table (``main_lists``); ``files`` maps each Role to its
    file name (MAIN_LIST_FILES; the Comparison_Lists pass their own). Both tables are validated
    before anything is written (ValueError, and no file is written, when one fails); then both
    files are written through ``<file>.tmp`` and renamed. Format: module docstring.
    """
    tables = _checked_lists(lists)
    folder = Path(outdir)
    targets = {role: folder / files[role] for role in ROLES}
    if len(set(targets.values())) != len(targets):
        raise ValueError(f"the Roles' list files must have different names; got {dict(files)!r}")
    folder.mkdir(parents=True, exist_ok=True)

    def writer(table: pd.DataFrame) -> Callable[[Path], object]:
        return lambda temporary: table.to_csv(
            temporary, index=False, encoding="utf-8", lineterminator="\n", float_format=LIST_FLOAT_FORMAT
        )

    _write_files({targets[role]: writer(tables[role]) for role in ROLES})
    return targets


def list_entries(lists: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """The entries of both lists (ENTRY_COLUMNS): the batter list's ranks 1-10, then the bowler list's."""
    tables = _checked_lists(lists)
    frames = [
        pd.DataFrame(
            {
                "role": pd.array([role] * len(tables[role]), dtype="str"),
                "rank": tables[role]["rank"].to_numpy(dtype=np.int64),
                "player_id": tables[role]["player_id"].to_numpy(dtype=np.int64),
                "player_name": pd.array(tables[role]["player_name"].tolist(), dtype="str"),
            },
            columns=list(ENTRY_COLUMNS),
        )
        for role in ROLES
    ]
    return pd.concat(frames, ignore_index=True)


# =============================================================================== checklist

def _code(text: str) -> str:
    """``text`` as a Markdown code span that CommonMark renders as exactly ``text``."""
    fence = "`" * (max((len(run) for run in re.findall(r"`+", text)), default=0) + 1)
    padded = text.startswith("`") or text.endswith("`") or (
        text.startswith(" ") and text.endswith(" ") and text.strip(" ") != ""
    )
    pad = " " if padded else ""
    return f"{fence}{pad}{text}{pad}{fence}"


def _entry_line(record: Mapping[str, Any]) -> str:
    status = _RESOLVED if record["resolved"] else _UNRESOLVED
    return (
        f"- [ ] {ROLE_LABELS[record['role']]} rank {int(record['rank'])}: {_code(str(record['player_name']))} "
        f"(player_id {int(record['player_id'])}), {status}"
    )


def _detail_lines(record: Mapping[str, Any]) -> list[str]:
    """The indented notes under an unresolved entry (Requirement 25, criterion 3)."""
    pid, name, role = int(record["player_id"]), str(record["player_name"]), str(record["role"])
    n_names, n_holders = int(record["n_candidate_names"]), int(record["n_name_holders"])
    reasons = []
    if n_names == 0:
        reasons.append(
            f"player_id {pid} has no Candidate_Name, so the name is its most frequent `{NAME_COLUMNS[role][1]}` in "
            f"`{_BALLS_FILE}`"
        )
    elif n_names > 1:
        reasons.append(
            f"player_id {pid} has {n_names} Candidate_Names, so the name is the one in the most mapping rows"
        )
    if n_holders > 1:
        reasons.append(f"{_code(name)} is a Candidate_Name of {n_holders} player_ids")
    candidates = "; ".join(
        f"{_code(str(text))} ({_count(int(rows), 'mapping row')})"
        for text, rows in zip(record["candidate_names"], record["candidate_rows"])
    )
    holders = ", ".join(str(int(other)) for other in record["name_holders"])
    lines = [
        f"{_DETAIL_INDENT}- Why unresolved: {'; '.join(reasons)}.",
        f"{_DETAIL_INDENT}- Candidate_Names of player_id {pid}: {candidates or 'none'}.",
        f"{_DETAIL_INDENT}- player_ids with the Candidate_Name {_code(name)}: {holders or 'none'}.",
        f"{_DETAIL_INDENT}- `team_name` values in `{_MAPPING_FILE}`:",
    ]
    for other, teams in zip(record["listed_ids"], record["listed_teams"]):
        label = f"player_id {int(other)}" + (" (this entry)" if int(other) == pid else "")
        shown = "; ".join(str(team) for team in teams)
        lines.append(f"{_DETAIL_INDENT * 2}- {label}: {shown or 'none'}")
    return lines


def _checked_details(entries: pd.DataFrame, details: pd.DataFrame) -> list[dict[str, Any]]:
    _require(details, (*ENTRY_COLUMNS, *DETAIL_COLUMNS), "details")
    rows = details.reset_index(drop=True)
    hint = "build details with unresolved(list_entries(lists), names, teams)"
    if len(rows) != len(entries):
        raise ValueError(f"details has {_count(len(rows), 'row')}, but the lists have {len(entries)} entries; {hint}")
    for column in ENTRY_COLUMNS:
        if rows[column].tolist() != entries[column].tolist():
            raise ValueError(f"details column {column!r} does not follow the list entries; {hint}")
    records = rows.to_dict("records")
    for record in records:
        rule = int(record["n_candidate_names"]) == 1 and int(record["n_name_holders"]) <= 1
        if bool(record["resolved"]) != rule:
            raise ValueError(
                f"details marks the {record['role']} entry at rank {record['rank']} as "
                f"{'resolved' if record['resolved'] else 'unresolved'}, against its name counts; {hint}"
            )
    return records


def checklist_text(lists: Mapping[str, pd.DataFrame], details: pd.DataFrame) -> str:
    """The text of ``unstop_entry_checklist.md`` (Requirement 25, criteria 3 and 6; module docstring).

    ``lists`` maps each Role to its Main_List table; ``details`` is
    ``unresolved(list_entries(lists), names, teams)``. ValueError when they do not fit together.
    """
    entries = list_entries(lists)
    records = _checked_details(entries, details)
    open_entries = [record for record in records if not record["resolved"]]
    if open_entries:
        listed = "; ".join(
            f"{ROLE_LABELS[record['role']]} rank {int(record['rank'])}, {_code(str(record['player_name']))}"
            for record in open_entries
        )
        summary = f"{len(open_entries)} unresolved ({listed})"
    else:
        summary = "0 unresolved"
    counts = ", ".join(
        f"{sum(record['role'] == role for record in records)} {_SECTION_TITLES[role].lower()}" for role in ROLES
    )
    files = " and ".join(f"`{MAIN_LIST_FILES[role]}`" for role in ROLES)
    lines = [
        "# Unstop entry checklist",
        "",
        f"This checklist holds the entries of the two Main_Lists, {files}, in rank order. Use it while you "
        "enter the lists in the Unstop dropdowns:",
        "",
        "1. For each entry, select the dropdown option whose text equals the name between backticks exactly, with "
        "the same case, spacing and punctuation. It is the `player_name` of the list file.",
        f"2. A resolved entry's `player_id` has exactly one Candidate_Name in `{_MAPPING_FILE}`, and no other "
        "`player_id` has that name, so one option should match.",
        "3. An unresolved entry needs a closer look: its `player_id` has no Candidate_Name or several, or its name "
        "is a Candidate_Name of more than one `player_id`, so the dropdown may show the name more than once or not "
        "at all. Its notes list every Candidate_Name of its `player_id` and every `player_id` that has the name, "
        "each with its `team_name` values. Use the teams to pick the option of the listed `player_id`, not a "
        "namesake.",
        "4. Tick the box (`- [x]`) once the dropdown shows the right player.",
        "",
        f"Entries: {len(records)} ({counts}); {len(records) - len(open_entries)} resolved, {summary}.",
    ]
    for role in ROLES:
        lines += ["", f"## {_SECTION_TITLES[role]} (`{MAIN_LIST_FILES[role]}`)", ""]
        for record in records:
            if record["role"] == role:
                lines.append(_entry_line(record))
                if not record["resolved"]:
                    lines.extend(_detail_lines(record))
    return "\n".join(lines) + "\n"


def write_checklist(lists: Mapping[str, pd.DataFrame], details: pd.DataFrame, path: Path | str) -> Path:
    """Write ``checklist_text(lists, details)`` to ``path`` (UTF-8, ``\\n`` line ends) through ``<file>.tmp``."""
    data = checklist_text(lists, details).encode("utf-8")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_files({target: lambda temporary: temporary.write_bytes(data)})
    return target


def parse_checklist(text: str) -> pd.DataFrame:
    """The entry lines of a checklist's text (PARSED_COLUMNS), in file order; other lines are skipped.

    ``player_name`` is the code span's text, ``resolved`` is False for ``**unresolved**`` and
    ``checked`` is True for a ticked box (``[x]``).
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a str; got {type(text).__name__}")
    rows = []
    for line in text.splitlines():
        match = _ENTRY_LINE.fullmatch(line)
        if match is None:
            continue
        name = match["code"]
        if name.startswith(" ") and name.endswith(" ") and name.strip(" "):
            name = name[1:-1]
        rows.append(
            (_LABEL_ROLES[match["label"]], int(match["rank"]), int(match["player_id"]), name,
             match["status"] == _RESOLVED, match["box"] != " ")
        )  # fmt: skip
    columns = list(zip(*rows)) if rows else [()] * len(PARSED_COLUMNS)
    return pd.DataFrame(
        {
            "role": pd.array(list(columns[0]), dtype="str"),
            "rank": np.asarray(columns[1], dtype=np.int64),
            "player_id": np.asarray(columns[2], dtype=np.int64),
            "player_name": pd.array(list(columns[3]), dtype="str"),
            "resolved": np.asarray(columns[4], dtype=bool),
            "checked": np.asarray(columns[5], dtype=bool),
        },
        columns=list(PARSED_COLUMNS),
    )


# ================================================================================== inputs

@dataclass(frozen=True)
class ListInputs:
    """The tables behind the Main_Lists and the checklist (``load_list_inputs``)."""

    ranking: pd.DataFrame  # rating checkpoint ``player_estimates``: the Main_Ranking of both Roles
    bootstrap: pd.DataFrame  # bootstrap checkpoint ``bootstrap`` (robustness.BOOTSTRAP_COLUMNS)
    shares: pd.DataFrame  # pressure_shares of the rating checkpoint's ``contributions``
    names: pd.DataFrame  # candidate_names of the load checkpoint's ``mapping``
    teams: pd.DataFrame  # team_names of the same table
    balls: pd.DataFrame  # the load checkpoint's ``ball_by_ball``, NAME_COLUMNS only


def load_list_inputs(store: CheckpointStore) -> ListInputs:
    """Read the ``load``, ``rating`` and ``bootstrap`` checkpoints into a ListInputs.

    StageError (from the store) when a checkpoint file is missing or changed.
    """
    mapping = store.load_table(_LOAD_STAGE, _MAPPING_TABLE)
    columns = [column for pair in NAME_COLUMNS.values() for column in pair]
    balls = store.load_table(_LOAD_STAGE, _BALLS_TABLE)
    _require(balls, columns, "the load table ball_by_ball")
    return ListInputs(
        ranking=store.load_table(RATING_STAGE, PLAYER_ESTIMATES_TABLE),
        bootstrap=store.load_table(BOOTSTRAP_STAGE, BOOTSTRAP_TABLE),
        shares=pressure_shares(store.load_table(RATING_STAGE, CONTRIBUTIONS_TABLE)),
        names=candidate_names(mapping),
        teams=team_names(mapping),
        balls=balls.loc[:, columns].copy(),
    )


# ===================================================================== the activity comparison

#: Folder under FINAL_FOLDER that holds the activity comparison's files. The activity comparison
#: writes no file outside it (Requirement 25, criterion 15).
COMPARISON_FOLDER = "comparison"

#: The Comparison_List file of each Role (Requirement 25, criterion 13).
COMPARISON_LIST_FILES: Mapping[str, str] = {
    BATTER_ROLE: "top10_batters_activity_off.csv", BOWLER_ROLE: "top10_bowlers_activity_off.csv",
}  # fmt: skip

#: The activity comparison table of each Role (Requirement 25, criterion 14).
ACTIVITY_TABLE_FILES: Mapping[str, str] = {
    BATTER_ROLE: "activity_comparison_batters.csv", BOWLER_ROLE: "activity_comparison_bowlers.csv",
}  # fmt: skip

#: Every file that write_comparison writes and clear_comparison deletes, in writing order.
COMPARISON_FILES: tuple[str, ...] = (
    *(COMPARISON_LIST_FILES[role] for role in ROLES), *(ACTIVITY_TABLE_FILES[role] for role in ROLES),
)  # fmt: skip

#: Columns of an activity comparison table and file, in order (module docstring).
ACTIVITY_COLUMNS: tuple[str, ...] = (
    "kind", "role", "n_eligible_main", "n_eligible_comparison", "overlap", "player_id", "player_name", "main_rank",
    "comparison_rank", "latest_calendar_year",
)  # fmt: skip

#: The ``kind`` of the rows of an activity comparison table: the summary row, then the players.
SUMMARY_KIND = "summary"
ACTIVITY_KINDS: tuple[str, ...] = (SUMMARY_KIND, ENTERING, LEAVING)

_ACTIVITY_COUNTS: tuple[str, ...] = ("n_eligible_main", "n_eligible_comparison", "overlap")
_ACTIVITY_PLAYER: tuple[str, ...] = ACTIVITY_COLUMNS[5:]
_ACTIVITY_INTEGERS: tuple[str, ...] = ("player_id", "main_rank", "comparison_rank", "latest_calendar_year")
_RANKING_KEYS: tuple[str, ...] = ("role", "player_id", "eligible", "rank")
_XI_TABLE = "playing_eleven"
_MATCHES_TABLE = "matches"
_XI_COLUMNS: tuple[str, ...] = ("match_id", "player_id", "is_played")
_MATCH_COLUMNS: tuple[str, ...] = ("match_id", "match_date")
#: The Eligible_Player counts of the activity_comparison ``status``, and the RankingComparison field of each.
_STATUS_COUNTS: tuple[tuple[str, str], ...] = (("n_eligible_main", "n_main"), ("n_eligible", "n_other"))


def comparison_dir(outputs: Path | str) -> Path:
    """``<outputs>/final/comparison``: the folder of the Comparison_Lists and activity comparison tables."""
    return Path(outputs) / FINAL_FOLDER / COMPARISON_FOLDER


def comparison_lists(
    *,
    ranking: pd.DataFrame,
    bootstrap: pd.DataFrame,
    shares: pd.DataFrame,
    names: pd.DataFrame,
    balls: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Both Roles' Comparison_Lists, ``{role: list_table(role, ...)}`` (Requirement 25, criterion 13).

    ``ranking`` and ``bootstrap``: the ``activity_comparison`` checkpoint's ``player_estimates`` and
    ``bootstrap``, whose ``main_rank`` and ``main_final_score`` are the comparison ranking's own (as
    list_table checks); ``shares``, ``names`` and ``balls``: those of the Main_Lists (ListInputs).
    The columns, checks and name rules are those of ``main_lists``.
    """
    return main_lists(ranking=ranking, bootstrap=bootstrap, shares=shares, names=names, balls=balls)


def _id_values(player_ids: Any) -> np.ndarray:
    """``player_ids`` (1-D whole numbers, each once) as int64."""
    if isinstance(player_ids, (str, bytes, Mapping, pd.DataFrame)) or np.ndim(player_ids) != 1:
        raise TypeError(f"player_ids must be a 1-D list, array or Series of ids; got {type(player_ids).__name__}")
    values = player_ids if isinstance(player_ids, pd.Series) else pd.Series(np.asarray(player_ids))
    ids = _whole(values, "player_ids")
    _unique(ids, "player_ids")
    return ids


def _played(values: pd.Series, what: str) -> np.ndarray:
    """``values`` as flags; a missing value reads False, as for the Recent_Activity_Rule."""
    if not pd.api.types.is_bool_dtype(values.dtype):
        raise TypeError(f"{what} must hold True or False; got dtype {values.dtype}")
    return values.astype("boolean").to_numpy(dtype=bool, na_value=False)


def latest_calendar_years(player_ids: Any, xi: pd.DataFrame, matches: pd.DataFrame) -> pd.Series:
    """The Calendar_Year of each player's latest match with ``is_played`` true (Requirement 25, criterion 14).

    ``player_ids``: the players, each once. ``xi``: rows with ``match_id``, ``player_id`` and
    ``is_played``, such as the load table ``playing_eleven``; a missing ``is_played`` reads false.
    ``matches``: ``match_id`` (each once) and ``match_date`` (datetime), such as the load table
    ``matches``. Returns an Int64 Series named ``latest_calendar_year``, indexed by ``player_id`` in
    the order given: the year of the latest ``match_date`` among the player's rows with
    ``is_played`` true, <NA> when there is none. Raises ValueError when such a row's match is not
    in ``matches`` or has no date.
    """
    wanted = _id_values(player_ids)
    _require(xi, _XI_COLUMNS, "xi")
    _require(matches, _MATCH_COLUMNS, "matches")
    dates = matches["match_date"]
    if not pd.api.types.is_datetime64_any_dtype(dates.dtype):
        raise TypeError(
            f"matches column 'match_date' must be a datetime column, as the Data_Loader reads it; got {dates.dtype}"
        )
    match_ids = _whole(matches["match_id"], "matches column 'match_id'")
    if len(np.unique(match_ids)) != len(match_ids):
        raise ValueError("matches holds a match_id more than once")
    holders = _numbers(xi["player_id"], "xi column 'player_id'")
    rows = np.flatnonzero(_played(xi["is_played"], "xi column 'is_played'") & np.isin(holders, wanted))
    at = pd.Index(match_ids).get_indexer(_whole(xi["match_id"].iloc[rows], "xi column 'match_id' on the rows read"))
    if (at < 0).any():
        raise ValueError(
            f"{int((at < 0).sum()):,} xi row(s) with is_played true for these players have a match_id that matches "
            "does not hold, so the Calendar_Year is unknown"
        )
    played_dates = dates.iloc[at]
    if played_dates.isna().any():
        raise ValueError("matches has no match_date for a match in which one of these players has is_played true")
    latest = pd.Series(played_dates.to_numpy(), index=holders[rows].astype(np.int64)).groupby(level=0).max()
    year_of = dict(zip(latest.index.tolist(), pd.DatetimeIndex(latest.to_numpy()).year.tolist()))
    return pd.Series(
        pd.array([year_of.get(pid, pd.NA) for pid in wanted.tolist()], dtype="Int64"),
        index=pd.Index(wanted, name="player_id"),
        name="latest_calendar_year",
    )


def _full_ranking(frame: pd.DataFrame, role: str, what: str, label: str) -> pd.DataFrame:
    """One Role's ``player_id``, ``rank`` (float64, NaN unranked) and ``eligible`` of a full-data ranking table.

    StageError unless the table ranks exactly its Eligible_Players, LIST_SIZE or more of them.
    """
    _require(frame, _RANKING_KEYS, what)
    rows = _role_rows(frame, role, what)
    eligible = _flags(rows["eligible"], f"{what} column 'eligible'")
    ids = _whole(rows["player_id"], f"{what} column 'player_id'")
    _unique(ids, f"the {role} rows of {what}")
    ranks = _numbers(rows["rank"], f"{what} column 'rank'")
    ranked = ~np.isnan(ranks)
    if not np.array_equal(ranked, eligible):
        stray, unranked = int((ranked & ~eligible).sum()), int((eligible & ~ranked).sum())
        raise StageError(
            f"the {role} {label} ranks {_count(stray, 'player')} who are not Eligible_Players and leaves "
            f"{_count(unranked, 'Eligible_Player')} unranked; a full-data ranking ranks exactly its Eligible_Players "
            "(Requirement 18, criterion 13), so the tables do not come from the same Run"
        )
    count = int(eligible.sum())
    if count < LIST_SIZE:
        raise StageError(
            f"the {role} {label} has {_count(count, 'Eligible_Player')}, but a top-{LIST_SIZE} comparison needs "
            f"{LIST_SIZE} (Requirement 25, criterion 9)"
        )
    return pd.DataFrame({"player_id": ids, "rank": ranks, "eligible": eligible})


def _activity_table(
    role: str,
    *,
    main_ranking: pd.DataFrame,
    comparison_ranking: pd.DataFrame,
    names: pd.DataFrame,
    balls: pd.DataFrame,
    xi: pd.DataFrame,
    matches: pd.DataFrame,
) -> tuple[pd.DataFrame, RankingComparison]:
    """``activity_comparison_table`` and the RankingComparison behind it."""
    role = _role(role)
    main = _full_ranking(main_ranking, role, "main_ranking", "Main_Ranking")
    other = _full_ranking(comparison_ranking, role, "comparison_ranking", "Activity_Comparison_Ranking")
    dropped = np.setdiff1d(
        main["player_id"].to_numpy()[main["eligible"].to_numpy()],
        other["player_id"].to_numpy()[other["eligible"].to_numpy()],
    )
    if len(dropped):
        raise StageError(
            f"{_count(len(dropped), 'Eligible_Player')} of the {role} Main_Ranking, for example player_id "
            f"{int(dropped[0])}, are not Eligible_Players of the Activity_Comparison_Ranking; turning the "
            "Recent_Activity_Rule off removes no player (Requirement 25, criterion 11), so the tables do not come from "
            "the same Run"
        )
    index = _name_index(names)
    comparison = compare_rankings(
        main.loc[:, ["player_id", "rank"]],
        other.loc[:, ["player_id", "rank"]],
        top_n=LIST_SIZE,
        names=lambda pid: _choose(int(pid), index, balls, role).player_name,
    )
    changes = comparison.changes
    years = latest_calendar_years(list(comparison.entering), xi, matches)
    latest = [
        years.loc[pid] if status == ENTERING else pd.NA
        for status, pid in zip(changes["status"].tolist(), changes["player_id"].tolist())
    ]
    n_rows = len(changes) + 1
    table = pd.DataFrame(
        {
            "kind": pd.array([SUMMARY_KIND, *changes["status"].tolist()], dtype="str"),
            "role": pd.array([role] * n_rows, dtype="str"),
            "n_eligible_main": np.full(n_rows, comparison.n_main, dtype=np.int64),
            "n_eligible_comparison": np.full(n_rows, comparison.n_other, dtype=np.int64),
            "overlap": np.full(n_rows, comparison.overlap, dtype=np.int64),
            "player_id": pd.array([pd.NA, *changes["player_id"].tolist()], dtype="Int64"),
            "player_name": pd.array([None, *changes["player_name"].tolist()], dtype="str"),
            "main_rank": pd.array([pd.NA, *changes["main_rank"].tolist()], dtype="Int64"),
            "comparison_rank": pd.array([pd.NA, *changes["other_rank"].tolist()], dtype="Int64"),
            "latest_calendar_year": pd.array([pd.NA, *latest], dtype="Int64"),
        },
        columns=list(ACTIVITY_COLUMNS),
    )
    validate_activity_table(table, role)
    return table, comparison


def activity_comparison_table(
    role: str,
    *,
    main_ranking: pd.DataFrame,
    comparison_ranking: pd.DataFrame,
    names: pd.DataFrame,
    balls: pd.DataFrame,
    xi: pd.DataFrame,
    matches: pd.DataFrame,
) -> pd.DataFrame:
    """One Role's activity comparison table (ACTIVITY_COLUMNS; Requirement 25, criterion 14; module docstring).

    ``main_ranking``: the rating checkpoint's ``player_estimates`` (the Main_Ranking);
    ``comparison_ranking``: the ``activity_comparison`` checkpoint's ``player_estimates``; each needs
    ``role``, ``player_id``, ``eligible`` and ``rank``. ``names``: ``candidate_names``; ``balls``:
    the ``ball_by_ball`` name columns; ``xi`` and ``matches``: the load tables ``playing_eleven``
    and ``matches`` (``latest_calendar_years``). Raises StageError when either ranking does not rank
    exactly its Eligible_Players, at least 10 of them, or a Main_Ranking Eligible_Player is not one
    of the comparison's: such tables do not come from the same Run.
    """
    table, _ = _activity_table(
        role, main_ranking=main_ranking, comparison_ranking=comparison_ranking, names=names, balls=balls, xi=xi,
        matches=matches,
    )  # fmt: skip
    return table


def _top_ranks(ranks: np.ndarray) -> bool:
    """True when ``ranks`` are finite, from 1 to LIST_SIZE and increasing."""
    inside = np.isfinite(ranks).all() and ((ranks >= 1) & (ranks <= LIST_SIZE)).all()
    return bool(inside and (np.diff(ranks) > 0).all())


def validate_activity_table(table: pd.DataFrame, role: str) -> None:
    """Raise ValueError, listing every problem, unless ``table`` is a writable activity comparison table of ``role``.

    The layout is in the module docstring ("Comparison_Lists and activity comparison tables").
    """
    role = _role(role)
    what = f"the {role} activity comparison table"
    if not isinstance(table, pd.DataFrame):
        raise TypeError(f"{what} must be a DataFrame; got {type(table).__name__}")
    if list(table.columns) != list(ACTIVITY_COLUMNS):
        raise ValueError(
            f"{what} must have the columns {', '.join(ACTIVITY_COLUMNS)} in that order; got "
            f"{', '.join(map(str, table.columns))}"
        )
    if not len(table):
        raise ValueError(f"{what} cannot be written: it has no row, not even the summary row")
    problems = [
        f"column {column!r} must have an integer dtype; got {table[column].dtype}"
        for column in (*_ACTIVITY_COUNTS, *_ACTIVITY_INTEGERS)
        if not pd.api.types.is_integer_dtype(table[column].dtype)
    ]
    if problems:
        raise ValueError(f"{what} cannot be written: {'; '.join(problems)}")
    kinds = _texts(table["kind"], f"{what} column 'kind'")
    roles = _texts(table["role"], f"{what} column 'role'")
    names = _texts(table["player_name"], f"{what} column 'player_name'")
    unknown = sorted({str(kind) for kind in kinds if kind not in ACTIVITY_KINDS})
    if unknown:
        problems.append(f"column 'kind' holds {', '.join(unknown)}; the kinds are {', '.join(ACTIVITY_KINDS)}")
    if kinds[0] != SUMMARY_KIND or kinds.count(SUMMARY_KIND) != 1:
        problems.append(f"the first row, and only the first, must have kind {SUMMARY_KIND!r}")
    players = kinds[1:]
    n_entering, n_leaving = players.count(ENTERING), players.count(LEAVING)
    if not unknown and players != [ENTERING] * n_entering + [LEAVING] * n_leaving:
        problems.append("the entering rows must come before the leaving rows")
    if any(value != role for value in roles):
        problems.append(f"column 'role' must be {role!r} on every row")
    counts: dict[str, int] = {}
    for column in _ACTIVITY_COUNTS:
        values = table[column]
        if values.isna().any() or values.nunique() != 1:
            problems.append(f"column {column!r} must hold one value, repeated on every row")
        else:
            counts[column] = int(values.iloc[0])
    if len(counts) == len(_ACTIVITY_COUNTS):
        n_main, n_comparison, overlap = (counts[column] for column in _ACTIVITY_COUNTS)
        if n_main < LIST_SIZE:
            problems.append(f"n_eligible_main is {n_main}, below the {LIST_SIZE} players of a Main_List")
        if n_comparison < n_main:
            problems.append(
                f"n_eligible_comparison ({n_comparison}) is below n_eligible_main ({n_main}); turning the "
                "Recent_Activity_Rule off removes no player"
            )
        if not 0 <= overlap <= LIST_SIZE:
            problems.append(f"overlap is {overlap}, outside 0 to {LIST_SIZE}")
        elif n_entering != LIST_SIZE - overlap or n_leaving != LIST_SIZE - overlap:
            problems.append(
                f"an overlap of {overlap} needs {LIST_SIZE - overlap} entering and {LIST_SIZE - overlap} leaving "
                f"rows; got {n_entering} and {n_leaving}"
            )
    summary = table.iloc[0]
    if not all(_is_missing(summary[column]) for column in _ACTIVITY_PLAYER):
        problems.append(f"the summary row must leave {', '.join(_ACTIVITY_PLAYER)} blank")
    body = table.iloc[1:]
    if body["player_id"].isna().any():
        problems.append("every player row needs a player_id")
    elif body["player_id"].duplicated().any():
        problems.append("column 'player_id' repeats a player")
    if any(name is None for name in names[1:]):
        problems.append("every player row needs a player_name")
    main_rank, comparison_rank, year = (
        body[column].to_numpy(dtype=np.float64, na_value=np.nan)
        for column in ("main_rank", "comparison_rank", "latest_calendar_year")
    )
    entering = np.asarray([kind == ENTERING for kind in players], dtype=bool)
    leaving = np.asarray([kind == LEAVING for kind in players], dtype=bool)
    if not _top_ranks(comparison_rank[entering]):
        problems.append(f"the entering rows need comparison ranks from 1 to {LIST_SIZE}, in increasing order")
    if (main_rank[entering] <= LIST_SIZE).any():
        problems.append(f"an entering player cannot have a main_rank from 1 to {LIST_SIZE}")
    if not _top_ranks(main_rank[leaving]):
        problems.append(f"the leaving rows need main ranks from 1 to {LIST_SIZE}, in increasing order")
    if (comparison_rank[leaving] <= LIST_SIZE).any():
        problems.append(f"a leaving player cannot have a comparison_rank from 1 to {LIST_SIZE}")
    if (main_rank < 1).any() or (comparison_rank < 1).any():
        problems.append("a rank must be 1 or more")
    if (~np.isnan(year[leaving])).any():
        problems.append("latest_calendar_year must be blank on a leaving row")
    if (year[entering] < 1).any():
        problems.append("latest_calendar_year must be a year")
    if problems:
        raise ValueError(f"{what} cannot be written: {'; '.join(problems)}")


def _check_described(listed: pd.DataFrame, table: pd.DataFrame, role: str) -> None:
    """ValueError unless the entering and leaving rows of ``table`` fit the Comparison_List ``listed``."""
    listed_rows = zip(*(listed[column].tolist() for column in ("player_id", "rank", "player_name")))
    at_rank = {int(pid): (int(rank), str(name)) for pid, rank, name in listed_rows}
    body = table.iloc[1:]
    problems = []
    rows = zip(*(body[column].tolist() for column in ("kind", "player_id", "player_name", "comparison_rank")))
    for kind, pid, name, rank in rows:
        if kind == ENTERING and at_rank.get(int(pid)) != (int(rank), str(name)):
            problems.append(f"entering player_id {int(pid)} is not {name!r} at rank {int(rank)} of the list")
        elif kind == LEAVING and int(pid) in at_rank:
            problems.append(f"leaving player_id {int(pid)} is in the list")
    if problems:
        raise ValueError(
            f"the {role} activity comparison table does not describe the {role} Comparison_List: {'; '.join(problems)}"
        )


def write_comparison(
    outputs: Path | str, *, lists: Mapping[str, pd.DataFrame], tables: Mapping[str, pd.DataFrame]
) -> tuple[Path, ...]:
    """Write the Comparison_Lists and activity comparison tables into ``<outputs>/final/comparison/``.

    ``outputs``: the outputs folder (``settings.path("outputs")``). ``lists``: each Role's
    Comparison_List (``comparison_lists``); ``tables``: each Role's activity comparison table
    (``activity_comparison_table``). Every list and table is validated, and each table checked
    against its list, before anything is written (ValueError, and no file is written, when one
    fails). The four files (COMPARISON_FILES) are then written through ``<file>.tmp`` and renamed
    once all four are written, the lists exactly as ``write_lists`` writes a list. Returns their
    paths in COMPARISON_FILES order. Writes no other file (Requirement 25, criterion 15).
    """
    checked = _checked_lists(lists)
    if not isinstance(tables, Mapping):
        raise TypeError(f"tables must map each Role to its activity comparison table; got {type(tables).__name__}")
    if set(tables) != set(ROLES):
        raise ValueError(f"tables must hold exactly the Roles {', '.join(ROLES)}; got {', '.join(map(str, tables))}")
    for role in ROLES:
        validate_activity_table(tables[role], role)
        _check_described(checked[role], tables[role], role)
    folder = comparison_dir(outputs)
    folder.mkdir(parents=True, exist_ok=True)

    def writer(table: pd.DataFrame, float_format: str | None) -> Callable[[Path], object]:
        return lambda temporary: table.to_csv(
            temporary, index=False, encoding="utf-8", lineterminator="\n", float_format=float_format
        )

    writers: dict[Path, Callable[[Path], object]] = {}
    for role in ROLES:
        writers[folder / COMPARISON_LIST_FILES[role]] = writer(checked[role], LIST_FLOAT_FORMAT)
    for role in ROLES:
        writers[folder / ACTIVITY_TABLE_FILES[role]] = writer(tables[role].reset_index(drop=True), None)
    _write_files(writers)
    return tuple(writers)


def clear_comparison(outputs: Path | str) -> tuple[Path, ...]:
    """Delete the activity comparison files of an earlier Run from ``<outputs>/final/comparison/``.

    Deletes each file of COMPARISON_FILES, and its leftover ``<file>.tmp``, that exists, and
    nothing else: the ``_stage/`` checkpoint folder and every other file stay. Creates no folder.
    Returns the deleted paths. Used when no comparison applies (module docstring).
    """
    folder = comparison_dir(outputs)
    removed: list[Path] = []
    for name in COMPARISON_FILES:
        for path in (folder / name, folder / f"{name}{_TMP}"):
            if path.is_file():
                path.unlink()
                removed.append(path)
    return tuple(removed)


@dataclass(frozen=True)
class ComparisonInputs:
    """The tables behind the Comparison_Lists and the activity comparison tables (``load_comparison_inputs``)."""

    status: Mapping[str, Any]  # the activity_comparison checkpoint's ``status`` JSON value
    ranking: pd.DataFrame  # activity_comparison ``player_estimates``: the Activity_Comparison_Ranking of both Roles
    bootstrap: pd.DataFrame  # activity_comparison ``bootstrap``: its bootstrap (robustness.BOOTSTRAP_COLUMNS)
    xi: pd.DataFrame  # the load checkpoint's ``playing_eleven``: match_id, player_id, is_played
    matches: pd.DataFrame  # the load checkpoint's ``matches``: match_id, match_date


def load_comparison_inputs(store: CheckpointStore) -> ComparisonInputs | None:
    """Read the activity comparison's inputs, or return None when no comparison applies.

    Reads the ``activity_comparison`` checkpoint's ``status``. When its ``applies`` is false (the
    Recent_Activity_Rule is off in the Config), reads nothing else and returns None. Otherwise
    also reads that checkpoint's ``player_estimates`` and ``bootstrap`` and the ``load`` tables
    ``playing_eleven`` and ``matches``. StageError when ``status`` has no true or false
    ``applies``, and from the store when a checkpoint file is missing or changed.
    """
    status = store.load_json(activity.STAGE_NAME, activity.STATUS_JSON)
    applies = status.get("applies") if isinstance(status, Mapping) else None
    if not isinstance(applies, bool):
        raise StageError(
            f"the {activity.STAGE_NAME} checkpoint's {activity.STATUS_JSON} value has no true or false 'applies', so "
            f"it does not say whether a comparison applies; run the {activity.STAGE_NAME} stage again"
        )
    if not applies:
        return None
    xi = store.load_table(_LOAD_STAGE, _XI_TABLE)
    matches = store.load_table(_LOAD_STAGE, _MATCHES_TABLE)
    _require(xi, _XI_COLUMNS, f"the load table {_XI_TABLE}")
    _require(matches, _MATCH_COLUMNS, f"the load table {_MATCHES_TABLE}")
    return ComparisonInputs(
        status=status,
        ranking=store.load_table(activity.STAGE_NAME, activity.PLAYER_ESTIMATES_TABLE),
        bootstrap=store.load_table(activity.STAGE_NAME, activity.BOOTSTRAP_TABLE),
        xi=xi.loc[:, list(_XI_COLUMNS)].copy(),
        matches=matches.loc[:, list(_MATCH_COLUMNS)].copy(),
    )


@dataclass(frozen=True, eq=False)
class ComparisonOutputs:
    """What ``write_comparison_outputs`` built and wrote; the tables are empty when no comparison applies."""

    applies: bool  # False: the Recent_Activity_Rule is off in the Config, so no comparison applies
    lists: Mapping[str, pd.DataFrame]  # each Role's Comparison_List (LIST_COLUMNS)
    tables: Mapping[str, pd.DataFrame]  # each Role's activity comparison table (ACTIVITY_COLUMNS)
    comparisons: Mapping[str, RankingComparison]  # each Role's Main_Ranking against its comparison ranking
    written: tuple[Path, ...]  # the files written, in COMPARISON_FILES order
    removed: tuple[Path, ...]  # the files of an earlier Run deleted because no comparison applies


def _check_status(status: Mapping[str, Any], comparisons: Mapping[str, RankingComparison]) -> None:
    """StageError unless ``status`` records the Eligible_Player counts of the rankings compared."""
    for key, field in _STATUS_COUNTS:
        recorded = status.get(key)
        for role in ROLES:
            found = int(getattr(comparisons[role], field))
            value = recorded.get(role) if isinstance(recorded, Mapping) else None
            if isinstance(value, bool) or value != found:
                raise StageError(
                    f"the {activity.STAGE_NAME} checkpoint's {activity.STATUS_JSON} records {value!r} as {key} of the "
                    f"{role} Role, but the rankings read hold {found:,} Eligible_Players; the checkpoints do not come "
                    "from the same Run"
                )


def write_comparison_outputs(store: CheckpointStore, inputs: ListInputs, outputs: Path | str) -> ComparisonOutputs:
    """The activity comparison step of the ``final_lists`` stage (Requirement 25, criteria 13-15).

    ``store``: the Run's CheckpointStore; ``inputs``: ``load_list_inputs(store)``; ``outputs``: the
    outputs folder. Where a comparison applies, builds both Roles' Comparison_Lists and activity
    comparison tables, checks the Eligible_Player counts against those ``status`` records, and
    writes the four files with ``write_comparison``. Where none applies, writes nothing and
    deletes the four files of an earlier Run with ``clear_comparison``. Touches no file outside
    ``<outputs>/final/comparison/``. Raises StageError when the checkpoints do not hold the same
    Run's rankings, and ValueError when a table cannot be written; nothing is written then.
    """
    if not isinstance(inputs, ListInputs):
        raise TypeError(f"inputs must be the ListInputs of load_list_inputs; got {type(inputs).__name__}")
    comparison = load_comparison_inputs(store)
    if comparison is None:
        removed = clear_comparison(outputs)
        return ComparisonOutputs(applies=False, lists={}, tables={}, comparisons={}, written=(), removed=removed)
    lists = comparison_lists(
        ranking=comparison.ranking, bootstrap=comparison.bootstrap, shares=inputs.shares, names=inputs.names,
        balls=inputs.balls,
    )  # fmt: skip
    tables: dict[str, pd.DataFrame] = {}
    comparisons: dict[str, RankingComparison] = {}
    for role in ROLES:
        tables[role], comparisons[role] = _activity_table(
            role, main_ranking=inputs.ranking, comparison_ranking=comparison.ranking, names=inputs.names,
            balls=inputs.balls, xi=comparison.xi, matches=comparison.matches,
        )  # fmt: skip
    _check_status(comparison.status, comparisons)
    written = write_comparison(outputs, lists=lists, tables=tables)
    return ComparisonOutputs(
        applies=True, lists=lists, tables=tables, comparisons=comparisons, written=written, removed=()
    )


# ================================================================ the ``final_lists`` stage

#: Name of the Pipeline_Stage that runs the List_Builder, and of its checkpoint.
STAGE_NAME = "final_lists"

#: Checkpoint tables of the stage (module docstring, "The ``final_lists`` Pipeline_Stage"): each
#: Role's Main_List (LIST_COLUMNS) and the checklist details without their tuple columns. The six
#: evidence tables are saved under their own names (``evidence.EVIDENCE_TABLES``).
LIST_TABLES: Mapping[str, str] = {BATTER_ROLE: "top10_batters", BOWLER_ROLE: "top10_bowlers"}
CHECKLIST_TABLE = "checklist_details"
CHECKLIST_TABLE_COLUMNS: tuple[str, ...] = (*ENTRY_COLUMNS, "resolved", "n_candidate_names", "n_name_holders")

#: JSON value of the checkpoint: the files the stage wrote and the comparison guard's snapshot.
WRITTEN_JSON = "written_files"

#: Folders under the outputs folder whose files the activity comparison step must leave unchanged.
GUARDED_FOLDERS: tuple[str, ...] = (FINAL_FOLDER, RATING_FOLDER, ROBUSTNESS_FOLDER)

#: Data-derived constants of the stage (List_Builder): the Unresolved_Entries of each Role's Main_List.
UNRESOLVED_CONSTANTS: Mapping[str, str] = {role: f"{STAGE_NAME}.n_unresolved.{role}" for role in ROLES}

_COMPONENT = "List_Builder"


def _guard_snapshot(outputs: Path) -> dict[str, str]:
    """``{path under outputs (posix): SHA-256}`` of every file directly in the GUARDED_FOLDERS."""
    snapshot: dict[str, str] = {}
    for name in GUARDED_FOLDERS:
        folder = outputs / name
        if folder.is_dir():
            for path in sorted(folder.iterdir()):
                if path.is_file():
                    snapshot[path.relative_to(outputs).as_posix()] = sha256_file(path)
    return snapshot


def _check_guard(before: Mapping[str, str], after: Mapping[str, str]) -> None:
    """StageError when a guarded file was created, changed or deleted between the two snapshots."""
    created = sorted(set(after) - set(before))
    deleted = sorted(set(before) - set(after))
    changed = sorted(name for name in set(before) & set(after) if before[name] != after[name])
    parts = [
        f"{verb} {', '.join(names)}"
        for verb, names in (("created", created), ("changed", changed), ("deleted", deleted))
        if names
    ]
    if parts:
        raise StageError(
            f"the activity comparison step {'; '.join(parts)} under the outputs folder, but Requirement 25, criterion "
            f"15 lets it write only under {FINAL_FOLDER}/{COMPARISON_FOLDER}/"
        )


def _shown(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _list_text(table: pd.DataFrame) -> str:
    parts = []
    for row in table.itertuples(index=False):
        bounded = math.isfinite(row.ci_low) and math.isfinite(row.ci_high)
        interval = f"[{row.ci_low:+.3f}, {row.ci_high:+.3f}]" if bounded else "no interval"
        parts.append(
            f"{row.rank} {row.player_name} ({row.player_id}) S {row.final_score:+.3f} {interval} "
            f"P_Top10 {row.p_top10:.3f}"
        )
    return "; ".join(parts)


def _unresolved_text(record: Mapping[str, Any], checklist: str) -> str:
    """One Run_Log line on an Unresolved_Entry: why, its Candidate_Names, the namesakes and their teams."""
    pid, name, role = int(record["player_id"]), str(record["player_name"]), str(record["role"])
    n_names, n_holders = int(record["n_candidate_names"]), int(record["n_name_holders"])
    reasons = []
    if n_names == 0:
        reasons.append(f"player_id {pid} has no Candidate_Name, so the name is its most frequent {NAME_COLUMNS[role][1]}")
    elif n_names > 1:
        reasons.append(f"player_id {pid} has {n_names} Candidate_Names, so the name is the one in the most mapping rows")
    if n_holders > 1:
        reasons.append(f"{name!r} is a Candidate_Name of {n_holders} player_ids")
    candidates = "; ".join(
        f"{text!r} ({_count(int(rows), 'mapping row')})"
        for text, rows in zip(record["candidate_names"], record["candidate_rows"])
    )
    teams = "; ".join(
        f"player_id {int(other)}: {', '.join(map(str, listed)) or 'none'}"
        for other, listed in zip(record["listed_ids"], record["listed_teams"])
    )
    holders = ", ".join(str(int(other)) for other in record["name_holders"])
    return (
        f"{_COMPONENT}: the {role} Main_List entry at rank {int(record['rank'])}, {name!r} (player_id {pid}), is an "
        f"Unresolved_Entry (Requirement 25, criterion 3): {'; '.join(reasons)}. Candidate_Names of player_id {pid}: "
        f"{candidates or 'none'}. player_ids with the Candidate_Name {name!r}: {holders or 'none'}. team_name values: "
        f"{teams}. Confirm the entry against the Unstop dropdown with {checklist} before submitting"
    )


def _comparison_text(role: str, result: RankingComparison) -> str:
    changes = result.changes
    rows = list(zip(*(changes[column].tolist() for column in ("status", "player_id", "player_name", "main_rank",
                                                               "other_rank"))))  # fmt: skip

    def rank_text(value: Any) -> str:
        return "not eligible" if _is_missing(value) else str(int(value))

    entering = "; ".join(
        f"{name} ({int(pid)}) comparison rank {rank_text(other)}, Main rank {rank_text(main)}"
        for status, pid, name, main, other in rows
        if status == ENTERING
    )
    leaving = "; ".join(
        f"{name} ({int(pid)}) Main rank {rank_text(main)}, comparison rank {rank_text(other)}"
        for status, pid, name, main, other in rows
        if status == LEAVING
    )
    rho = "undefined" if math.isnan(result.spearman) else f"{result.spearman:.4f}"
    return (
        f"{_COMPONENT}: activity comparison, {role} (Requirement 25, criteria 13 and 14): {result.n_main:,} "
        f"Eligible_Players in the Main_Ranking and {result.n_other:,} with the Recent_Activity_Rule off; top-{LIST_SIZE} "
        f"overlap {result.overlap} of {LIST_SIZE}. Entering: {entering or 'none'}. Leaving: {leaving or 'none'}. "
        f"Spearman {rho} over {result.n_common:,} common Eligible_Players"
    )


def _empty_cells(proof: Mapping[str, pd.DataFrame]) -> list[str]:
    """``"<role> rank <r> <column>"`` for each empty rate cell (zero denominator) of the traditional-stats tables."""
    cells = []
    for role in ROLES:
        table = proof[evidence.table_name(evidence.TRADITIONAL, role)]
        for ball_set in evidence.BALL_SETS:
            for name, formula in evidence.RATES[role].items():
                column = f"{ball_set}_{name}"
                empty = np.isnan(table[column].to_numpy(dtype=np.float64))
                for rank in table["rank"].to_numpy(dtype=np.int64)[empty].tolist():
                    cells.append(f"{role} rank {rank} {column} ({ball_set}_{formula.denominator} 0)")
    return cells


def _log_stage(
    ctx: StageContext,
    *,
    tables: Mapping[str, pd.DataFrame],
    details: pd.DataFrame,
    comparison: ComparisonOutputs,
    guard: Mapping[str, str],
    proof: Mapping[str, pd.DataFrame],
    n_deliveries: int,
    written: Sequence[Path],
    seconds: float,
) -> None:
    log, root = ctx.log, ctx.root
    outputs = ctx.settings.path("outputs")
    folder = _shown(outputs / FINAL_FOLDER, root)
    for role in ROLES:
        log.info(
            f"{_COMPONENT}: {role} Main_List (Requirement 25, criteria 1, 2, 5, 7 and 8) written to "
            f"{folder}/{MAIN_LIST_FILES[role]}: {_list_text(tables[role])}"
        )
    records = details.to_dict("records")
    open_entries = [record for record in records if not record["resolved"]]
    summary = next(
        (line for line in checklist_text(tables, details).splitlines() if line.startswith("Entries: ")), ""
    )
    log.info(f"{_COMPONENT}: Unstop entry checklist written to {folder}/{CHECKLIST_FILE}. {summary}")
    for record in open_entries:
        log.warning(_unresolved_text(record, f"{folder}/{CHECKLIST_FILE}"))
    if comparison.applies:
        for role in ROLES:
            log.info(_comparison_text(role, comparison.comparisons[role]))
        log.info(
            f"{_COMPONENT}: wrote {', '.join(_shown(path, root) for path in comparison.written)}; the "
            f"{len(guard):,} files directly in {', '.join(_shown(outputs / name, root) + '/' for name in GUARDED_FOLDERS)} "
            "kept their SHA-256 through the comparison step (Requirement 25, criterion 15)"
        )
    else:
        removed = ", ".join(_shown(path, root) for path in comparison.removed) or "none"
        log.info(
            f"{_COMPONENT}: no Activity_Comparison_Ranking applies, because the Recent_Activity_Rule is off in the "
            f"Config, so no comparison file is written; deleted the comparison files of an earlier Run: {removed}"
        )
    empty = _empty_cells(proof)
    log.info(
        f"{_COMPONENT}: evidence tables (Requirement 26, criteria 1-5) written to {folder}/: "
        f"{', '.join(evidence.EVIDENCE_FILES[name] for name in evidence.EVIDENCE_TABLES)}. Unweighted counts over "
        f"the {n_deliveries:,} rated Deliveries (innings 1 and 2 of the matches used for player rating); every listed "
        "player's Pressure_Ball and Ordinary_Ball counts equal the rating's n_pressure_balls and n_ordinary_balls. "
        f"{_count(len(empty), 'empty cell')} for a zero denominator (criterion 3): {'; '.join(empty) or 'none'}"
    )
    log.info(
        f"{_COMPONENT}: charts (Requirement 26, criteria 6-9) written to {folder}/ at {charts.CHART_DPI} dpi: "
        f"{', '.join(charts.CHART_FILES[name] for name in charts.CHART_NAMES)}"
    )
    saved = ", ".join([*LIST_TABLES.values(), CHECKLIST_TABLE, *evidence.EVIDENCE_TABLES, WRITTEN_JSON])
    log.info(
        f"{_COMPONENT}: stage {seconds:.1f} s; wrote {len(written):,} files outside {folder}/{COMPARISON_FOLDER}/ and "
        f"saved {saved} in {_shown(ctx.store.stage_dir(STAGE_NAME), root)}/ (no comparison output)"
    )


def run_stage(ctx: StageContext) -> None:
    """The ``final_lists`` Pipeline_Stage (module docstring, "The ``final_lists`` Pipeline_Stage").

    Writes the Main_Lists, the checklist, the evidence tables and the charts to ``outputs/final/``
    and the activity comparison's files to ``outputs/final/comparison/``, saves its checkpoint and
    logs to the Run_Log. Raises StageError (exit code 5) for a Role with fewer than 10
    Eligible_Players, checkpoints that do not hold the same Run, a comparison step that touches a
    guarded file, or a checkpoint file that is missing or changed; ValueError or TypeError for a
    table that cannot be written.
    """
    started = time.perf_counter()
    store = ctx.store
    outputs = ctx.settings.path("outputs")
    final = outputs / FINAL_FOLDER

    # 1. The Main_Lists, the checklist details and the evidence tables, all checked before any file is written.
    inputs = load_list_inputs(store)
    tables = main_lists(
        ranking=inputs.ranking, bootstrap=inputs.bootstrap, shares=inputs.shares, names=inputs.names,
        balls=inputs.balls,
    )  # fmt: skip
    details = unresolved(list_entries(tables), inputs.names, inputs.teams)
    deliveries = evidence.load_evidence_deliveries(store)
    proof = evidence.evidence_tables(tables, deliveries=deliveries, ranking=inputs.ranking, bootstrap=inputs.bootstrap)

    # 2. The Main_Lists and the checklist (Requirement 25, criteria 3, 5 and 6).
    list_paths = write_lists(tables, final)
    checklist_path = write_checklist(tables, details, final / CHECKLIST_FILE)

    # 3. The activity comparison (criteria 13-15), between two snapshots of the guarded files.
    guard = _guard_snapshot(outputs)
    comparison = write_comparison_outputs(store, inputs, outputs)
    _check_guard(guard, _guard_snapshot(outputs))

    # 4. The evidence tables and the charts (Requirement 26, criteria 6-10).
    evidence_paths = evidence.write_evidence(proof, final)
    chart_paths = charts.write_charts(
        final, lists=tables, tables=proof, ranking=inputs.ranking, bootstrap=inputs.bootstrap
    )

    # 5. The checkpoint, without any comparison output (Requirement 25, criterion 15).
    for role in ROLES:
        store.save_table(STAGE_NAME, LIST_TABLES[role], tables[role])
    store.save_table(STAGE_NAME, CHECKLIST_TABLE, details.loc[:, list(CHECKLIST_TABLE_COLUMNS)].copy())
    for name in evidence.EVIDENCE_TABLES:
        store.save_table(STAGE_NAME, name, proof[name])
    written = [*list_paths.values(), checklist_path, *evidence_paths.values(), *chart_paths.values()]
    store.save_json(
        STAGE_NAME,
        WRITTEN_JSON,
        {
            "files": {path.relative_to(outputs).as_posix(): sha256_file(path) for path in written},
            "comparison_guard": dict(guard),
        },
    )

    # 6. The data-derived constants and the Run_Log.
    resolved = details["resolved"].to_numpy(dtype=bool)
    for role in ROLES:
        in_role = (details["role"].astype("string") == role).to_numpy(dtype=bool)
        ctx.constants.record(UNRESOLVED_CONSTANTS[role], int((in_role & ~resolved).sum()), _COMPONENT)
    _log_stage(
        ctx, tables=tables, details=details, comparison=comparison, guard=guard, proof=proof,
        n_deliveries=len(deliveries), written=written, seconds=time.perf_counter() - started,
    )  # fmt: skip
