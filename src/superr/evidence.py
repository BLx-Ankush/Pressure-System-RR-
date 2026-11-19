"""List_Builder evidence tables: traditional statistics, rating evidence and the Pressure_Ball mix
of every Main_List player (Requirement 26, criteria 1-5 and 10; design section "List_Builder
(``lists.py``, ``charts.py``)", ``evidence_tables``; Property 59).

For each Role, three tables describe the 10 players of the Main_List (``lists.main_lists``), one
row per player in rank order 1 to 10. Each table starts with KEY_COLUMNS, the Main_List's own
``rank``, ``player_id`` and ``player_name``, so the names are the strings of the Main_List files:

* the traditional-stats table (criteria 1-3): unweighted batting or bowling statistics, computed
  separately on the player's Role-specific Pressure_Balls and Ordinary_Balls;
* the rating-evidence table (criterion 4): the player's rating values and bootstrap interval;
* the Pressure_Ball mix table (criterion 5): the share of the player's Pressure_Balls in each
  Phase and each Competition_Type.

Module layout. The design gives the List_Builder ``lists.py`` and ``charts.py``. The evidence
tables have this module of their own, so that ``lists.py`` keeps the names, lists and checklist.
``lists.run_stage`` (the ``final_lists`` Pipeline_Stage) calls ``load_evidence_deliveries``,
``evidence_tables`` and ``write_evidence``, and ``charts.write_charts`` draws the criterion 6-9
charts from the tables. No function here reads ``config.py``, logs or saves a checkpoint; only
``load_evidence_deliveries`` and ``write_evidence`` touch files.

Delivery set (``evidence_deliveries``, ``load_evidence_deliveries``)
    Every count uses the Deliveries that the Rating_Engine rates: the rows of the ``value``
    checkpoint's ``value`` table, which are innings 1 and 2 of the matches used for player rating
    (no Super_Over_Innings and no match excluded under Requirement 5, criterion 24). Criteria 1
    and 2 name only the Super_Over_Innings; leaving out the excluded matches as well keeps the
    evidence on the Deliveries, and with the Pressure_Ball flags, behind each player's
    Clutch_Level and Usual_Baseline. A listed player's Pressure_Ball and Ordinary_Ball counts
    therefore equal the rating's ``n_pressure_balls`` and ``n_ordinary_balls``, which
    ``evidence_tables`` checks. As delivered the set holds 189,648 of the 200,961 Deliveries of
    innings 1 and 2.

    ``evidence_deliveries(value, balls, pressure)`` joins, by ``row``, each ``value`` row (its
    match, innings, innings x Phase ``cell`` and class) with the ``reconstruct`` checkpoint's
    ``balls`` row and the ``pressure`` table's flags, and returns one row per rated Delivery, in
    ``value`` order, with DELIVERY_COLUMNS:

    ====================== ================================================================
    ``row``, ``match_id``, the keys; ``innings_ball_num`` orders an innings (Requirement 5)
    ``innings``,
    ``innings_ball_num``
    ``cls``                DeliveryClass code (Legal_Ball 0, Wide 1, No_Ball 2)
    ``striker_id``,        the players; ``non_striker_id`` is nullable
    ``non_striker_id``,
    ``bowler_id``
    ``bat_runs``           ``runs``, the bat runs
    ``team_runs``          the team runs under the selected Extras_Encoding
    ``bye``                a Legal_Ball whose ``extras_type`` is ``byes`` or ``legbyes``
                           (``reconstruct.BYE_EXTRAS_TYPES``)
    ``boundary``           ``is_four`` or ``is_six`` (glossary Boundary)
    ``counted_wicket``,    the ``balls`` dismissal flags (``reconstruct.wicket_flags`` and
    ``retire_out``,        ``dismissed_batter_flags``); ``dismissed_is_striker`` is nullable
    ``dismissed_is_striker``,
    ``dismissed_fallback``,
    ``bowler_wicket``
    ``live``,              the ``pressure`` table's Live_Ball flag and Pressure_Ball flags;
    ``pressure_bat``,      ``pressure_bat`` is <NA> on Wides, which are not Batter_Deliveries
    ``pressure_bowl``
    ``phase``              Phase (``state.Phase``): ``cell`` - 3 x (innings - 1), the D8 Phase
                           of the innings x Phase cell that the rating uses
    ``comp_type``          the match's Competition_Type (IPL, T20I or A_Team), as in the
                           Fold_Table and the rating's match weights
    ====================== ================================================================

Traditional statistics (``batter_counts``, ``bowler_counts``, ``traditional_table``; criteria 1-3)
    A player's batter-Role ball sets are the player's Batter_Deliveries (Legal_Balls and No_Balls
    as striker) with ``pressure_bat`` true (Pressure_Balls) or false (Ordinary_Balls); the
    bowler-Role sets are the player's Bowler_Deliveries (every Delivery bowled, Wides and No_Balls
    included) with ``pressure_bowl`` true or false. Nothing is weighted: no Recency_Weight or
    Stakes_Weight. The table has ``pressure_<stat>`` and then ``ordinary_<stat>`` columns, the
    statistics of criterion 1 or 2 in its order followed by the counts behind the rates
    (BATTER_STATS, BOWLER_STATS):

    ==================== ============================== ======================================
    Statistic            Batter                         Bowler
    ==================== ============================== ======================================
    ``deliveries``       Batter_Deliveries              Bowler_Deliveries
    ``legal_balls``      Legal_Balls faced              Legal_Balls bowled
    ``runs``             bat runs                       (none)
    ``dismissals``       Batter_Dismissals (below)      (none)
    ``wickets``          (none)                         Bowler_Wickets (``bowler_wicket``)
    ``runs_conceded``    (none)                         Bowler_Runs_Conceded (below)
    ``dot_balls``        Dot_Balls: Legal_Balls with 0 bat runs
    ``boundaries``       Boundaries
    ``wides``,           (none)                         Wides and No_Balls bowled
    ``no_balls``
    ``strike_rate``      100 x runs / deliveries        (none)
    ``average``          runs / dismissals              (none)
    ``economy``          (none)                         6 x runs_conceded / legal_balls
    ``balls_per_wicket`` (none)                         legal_balls / wickets
    ``dot_pct``          100 x dot_balls / legal_balls  100 x dot_balls / legal_balls
    ``boundary_pct``     100 x boundaries / deliveries  100 x boundaries / deliveries
    ``extras_per_100``   (none)                         100 x (wides + no_balls) / deliveries
    ==================== ============================== ======================================

    The rates are BATTER_RATES and BOWLER_RATES (``ball_set_rates``). A rate whose denominator is
    0 is NaN, an empty cell in the CSV, and the player's other values are still given
    (criterion 3).

    Batter_Dismissals (``batter_dismissals``): a Counted_Wicket other than Retire Out whose
    dismissed batter is the player, as striker or non-striker, whatever the Config's D14 setting
    (the glossary leaves Retire Out out). The dismissed batter is the ``balls`` flag
    ``dismissed_is_striker`` (``value.dismissed_batters``). The dismissal falls in the set of the
    Delivery's batter-Role flag: a striker out on his own Batter_Delivery takes that Delivery's
    ``pressure_bat``; every other dismissal (the non-striker out, or the striker out off a Wide)
    takes the flag of ``value.dismissal_charges``, the Value_Engine's documented rule: the
    Delivery's ``pressure_bat``, and on a Wide, which has none, the flag of the latest earlier
    Batter_Delivery of the innings, ordinary when the Wide is a Dead_Ball or the innings has no
    earlier Batter_Delivery. These are the flags of the rating's Dismissal_Charges. So a
    non-striker run out adds a Batter_Dismissal and no Batter_Delivery.

    Bowler_Runs_Conceded: the team runs of each Bowler_Delivery minus its byes and leg-byes. Every
    team run of a bye or leg-bye Legal_Ball is a bye or leg-bye (the selected Extras_Encoding,
    ``byes_one`` as delivered, scores it 1 + ``extras_2_runs``), so it charges 0, as the F25
    over-runs pillar leaves such Deliveries out (``state``, ``bowler_over_runs``). Every other
    Delivery charges its team runs: the bat runs of a Legal_Ball, 1 + ``extras_2_runs`` on a
    Wide, and all the team runs of a No_Ball, whose ``extras_2_runs`` the data does not split
    into byes and other extras. A Penalty Legal_Ball scores 0 team runs as delivered.

Rating evidence (``rating_evidence_table``; criterion 4)
    RATING_EVIDENCE_COLUMNS, copied unchanged and under the same names from the rating
    checkpoint's ``player_estimates`` (the Main_Ranking: ``final_score``, ``clutch``,
    ``usual``, ``uplift``, ``shrunk_clutch``, ``shrunk_uplift``, ``hit_rate``,
    ``worst_quartile`` and ``n_pressure_matches``) and the bootstrap checkpoint's ``bootstrap``
    table (``ci_low``, ``ci_high`` and ``p_top10``; RATING_SOURCES), so that criterion 12 can
    compare each value by name. Final_Score, Clutch_Level, Usual_Baseline, Uplift, their shrunk
    values, the interval bounds and Worst_Quartile_Mean are in runs per over.

Pressure_Ball mix (``pressure_mix_table``; criterion 5)
    The player's Role-specific Pressure_Balls (the batter set or the bowler set above, pressure
    only): ``n_pressure_balls``; the percentage in each Phase (``pct_powerplay``,
    ``pct_middle``, ``pct_death``) and in each Competition_Type (``pct_ipl``, ``pct_t20i``,
    ``pct_a_team``), each 100 x count / ``n_pressure_balls``; and the counts behind them
    (``n_powerplay`` ... ``n_a_team``). Each group of three percentages sums to 100 up to float
    rounding, well within the MIX_TOLERANCE of 0.1. The IPL count equals the IPL count behind the
    Main_List's ``share_ipl``, which ``evidence_tables`` checks.

Files (``write_evidence``; criterion 10)
    The six tables go to ``outputs/final/`` as EVIDENCE_FILES: ``traditional_stats_batters.csv``,
    ``traditional_stats_bowlers.csv``, ``rating_evidence_batters.csv``,
    ``rating_evidence_bowlers.csv``, ``pressure_mix_batters.csv`` and
    ``pressure_mix_bowlers.csv``. UTF-8, ``\\n`` line ends, a header row, whole numbers without
    decimals and an empty cell for a missing value. The traditional-stats and mix files write
    every float with EVIDENCE_FLOAT_FORMAT (10 decimals, the Main_Lists' format). The
    rating-evidence files write every float in Python's shortest round-trip form, so each value
    reads back as the same float64 that the Rating_Engine or the Robustness_Suite computed
    (criterion 12). ``validate_evidence`` checks every table before anything is written, and the
    six files are written through ``<file>.tmp`` and renamed once all six are written.

Errors: ValueError or TypeError, naming the problem, for a missing column, a value of the wrong
type, a repeated row, an unknown Role or Competition_Type, or a table that cannot be written.
StageError (exit code 5) when the tables do not hold the same Run: a listed player whom the
ranking does not rank at the list's rank, or with no bootstrap row, and evidence counts that differ
from the rating's.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import pandas as pd

from superr.errors import StageError
from superr.pressure import PRESSURE_TABLE
from superr.pressure import STAGE_NAME as PRESSURE_STAGE
from superr.rating import LIST_SIZE
from superr.reconstruct import BALLS_TABLE, BYE_EXTRAS_TYPES, COMP_A_TEAM, COMP_IPL, COMP_T20I, COMPETITION_TYPES
from superr.reconstruct import STAGE_NAME as RECONSTRUCT_STAGE
from superr.reconstruct import DeliveryClass
from superr.state import STATE_INNINGS, Phase
from superr.value import BATTER_ROLE, BOWLER_ROLE, ROLES, VALUE_TABLE, dismissal_charges, dismissed_batters
from superr.value import STAGE_NAME as VALUE_STAGE

if TYPE_CHECKING:
    from superr.store import CheckpointStore

__all__ = [
    "BALL_INPUT_COLUMNS",
    "BALL_SETS",
    "BATTER_COUNTS",
    "BATTER_DISMISSAL_COLUMNS",
    "BATTER_RATES",
    "BATTER_STATS",
    "BOOTSTRAP_INPUT_COLUMNS",
    "BOWLER_COUNTS",
    "BOWLER_RATES",
    "BOWLER_STATS",
    "COMP_TYPE_NAMES",
    "COUNTS",
    "DELIVERY_COLUMNS",
    "EVIDENCE_FILES",
    "EVIDENCE_FLOAT_FORMAT",
    "EVIDENCE_KINDS",
    "EVIDENCE_TABLES",
    "FLOAT_FORMATS",
    "KEY_COLUMNS",
    "LIST_INPUT_COLUMNS",
    "MIX_COUNT_COLUMNS",
    "MIX_PCT_COLUMNS",
    "MIX_TOLERANCE",
    "ORDINARY_SET",
    "PHASE_NAMES",
    "PRESSURE_INPUT_COLUMNS",
    "PRESSURE_MIX",
    "PRESSURE_MIX_COLUMNS",
    "PRESSURE_SET",
    "RANKING_INPUT_COLUMNS",
    "RATES",
    "RATING_EVIDENCE",
    "RATING_EVIDENCE_COLUMNS",
    "RATING_SOURCES",
    "ROLE_SUFFIXES",
    "RateFormula",
    "STATS",
    "TRADITIONAL",
    "TRADITIONAL_COLUMNS",
    "VALUE_INPUT_COLUMNS",
    "ball_set_rates",
    "batter_counts",
    "batter_dismissals",
    "bowler_counts",
    "evidence_deliveries",
    "evidence_tables",
    "load_evidence_deliveries",
    "pressure_mix_table",
    "rating_evidence_table",
    "table_name",
    "traditional_table",
    "validate_evidence",
    "write_evidence",
]

# ================================================================================ constants

#: The Role part of every evidence table and file name, as in ``top10_batters.csv``.
ROLE_SUFFIXES: Mapping[str, str] = {BATTER_ROLE: "batters", BOWLER_ROLE: "bowlers"}

#: The three kinds of evidence table (module docstring), in file order.
TRADITIONAL = "traditional_stats"
RATING_EVIDENCE = "rating_evidence"
PRESSURE_MIX = "pressure_mix"
EVIDENCE_KINDS: tuple[str, ...] = (TRADITIONAL, RATING_EVIDENCE, PRESSURE_MIX)


def table_name(kind: str, role: str) -> str:
    """The name of one evidence table, such as ``"traditional_stats_batters"``: its file stem and checkpoint name."""
    if kind not in EVIDENCE_KINDS:
        raise ValueError(f"kind must be one of {', '.join(EVIDENCE_KINDS)}; got {kind!r}")
    return f"{kind}_{ROLE_SUFFIXES[_role(role)]}"


#: Every evidence table, kind by kind and each kind batter first, and its file in ``outputs/final/``.
EVIDENCE_TABLES: tuple[str, ...] = tuple(f"{kind}_{ROLE_SUFFIXES[role]}" for kind in EVIDENCE_KINDS for role in ROLES)
EVIDENCE_FILES: Mapping[str, str] = {name: f"{name}.csv" for name in EVIDENCE_TABLES}
_TABLE_KIND_ROLE: Mapping[str, tuple[str, str]] = {
    f"{kind}_{ROLE_SUFFIXES[role]}": (kind, role) for kind in EVIDENCE_KINDS for role in ROLES
}

#: The two ball sets of a Role: Pressure_Balls and Ordinary_Balls, the prefixes of the traditional columns.
PRESSURE_SET = "pressure"
ORDINARY_SET = "ordinary"
BALL_SETS: tuple[str, ...] = (PRESSURE_SET, ORDINARY_SET)

#: The unweighted counts of one player's ball set (module docstring, "Traditional statistics").
BATTER_COUNTS: tuple[str, ...] = ("deliveries", "runs", "dismissals", "legal_balls", "dot_balls", "boundaries")
BOWLER_COUNTS: tuple[str, ...] = (
    "deliveries", "legal_balls", "wickets", "runs_conceded", "dot_balls", "boundaries", "wides", "no_balls",
)  # fmt: skip
COUNTS: Mapping[str, tuple[str, ...]] = {BATTER_ROLE: BATTER_COUNTS, BOWLER_ROLE: BOWLER_COUNTS}


class RateFormula(NamedTuple):
    """A rate of criterion 1 or 2: ``scale`` x (sum of the ``numerator`` counts) / the ``denominator`` count."""

    numerator: tuple[str, ...]
    denominator: str
    scale: float


#: The rates of each Role (criteria 1 and 2). The scales 100 (a percentage) and 6 (Legal_Balls per
#: over, a rule of cricket) are unit conversions, not Method_Constants.
BATTER_RATES: Mapping[str, RateFormula] = {
    "strike_rate": RateFormula(("runs",), "deliveries", 100.0),
    "average": RateFormula(("runs",), "dismissals", 1.0),
    "dot_pct": RateFormula(("dot_balls",), "legal_balls", 100.0),
    "boundary_pct": RateFormula(("boundaries",), "deliveries", 100.0),
}
BOWLER_RATES: Mapping[str, RateFormula] = {
    "economy": RateFormula(("runs_conceded",), "legal_balls", 6.0),
    "balls_per_wicket": RateFormula(("legal_balls",), "wickets", 1.0),
    "dot_pct": RateFormula(("dot_balls",), "legal_balls", 100.0),
    "boundary_pct": RateFormula(("boundaries",), "deliveries", 100.0),
    "extras_per_100": RateFormula(("wides", "no_balls"), "deliveries", 100.0),
}
RATES: Mapping[str, Mapping[str, RateFormula]] = {BATTER_ROLE: BATTER_RATES, BOWLER_ROLE: BOWLER_RATES}

#: The statistics of one ball set, in column order: the values of criterion 1 or 2 in its order,
#: then the counts behind the rates.
BATTER_STATS: tuple[str, ...] = (
    "deliveries", "runs", "dismissals", "strike_rate", "average", "dot_pct", "boundary_pct", "legal_balls",
    "dot_balls", "boundaries",
)  # fmt: skip
BOWLER_STATS: tuple[str, ...] = (
    "deliveries", "legal_balls", "wickets", "economy", "balls_per_wicket", "dot_pct", "boundary_pct",
    "extras_per_100", "runs_conceded", "dot_balls", "boundaries", "wides", "no_balls",
)  # fmt: skip
STATS: Mapping[str, tuple[str, ...]] = {BATTER_ROLE: BATTER_STATS, BOWLER_ROLE: BOWLER_STATS}

#: The first columns of every evidence table: the Main_List's rank, player_id and player_name.
KEY_COLUMNS: tuple[str, ...] = ("rank", "player_id", "player_name")

#: Columns of each Role's traditional-stats table and file.
TRADITIONAL_COLUMNS: Mapping[str, tuple[str, ...]] = {
    role: (*KEY_COLUMNS, *(f"{ball_set}_{stat}" for ball_set in BALL_SETS for stat in STATS[role])) for role in ROLES
}

#: Columns of the rating-evidence table and file (criterion 4), and the input table of each value.
RATING_EVIDENCE_COLUMNS: tuple[str, ...] = (
    *KEY_COLUMNS, "final_score", "clutch", "usual", "uplift", "shrunk_clutch", "shrunk_uplift", "ci_low", "ci_high",
    "p_top10", "hit_rate", "worst_quartile", "n_pressure_matches",
)  # fmt: skip
_RANKING_VALUES: tuple[str, ...] = (
    "final_score", "clutch", "usual", "uplift", "shrunk_clutch", "shrunk_uplift", "hit_rate", "worst_quartile",
)  # fmt: skip
_BOOTSTRAP_VALUES: tuple[str, ...] = ("ci_low", "ci_high", "p_top10")
RATING_SOURCES: Mapping[str, str] = {
    **{column: "player_estimates" for column in _RANKING_VALUES},
    **{column: "bootstrap" for column in _BOOTSTRAP_VALUES},
    "n_pressure_matches": "player_estimates",
}

#: The Phases (``state.Phase`` order) and Competition_Types (``reconstruct.COMPETITION_TYPES``
#: order) of the mix table, as column suffixes.
PHASE_NAMES: tuple[str, ...] = tuple(member.name.lower() for member in Phase)
COMP_TYPE_NAMES: Mapping[str, str] = {COMP_IPL: "ipl", COMP_T20I: "t20i", COMP_A_TEAM: "a_team"}

#: The percentage and count columns of the mix table, by group (``"phase"`` and ``"comp_type"``).
MIX_PCT_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "phase": tuple(f"pct_{name}" for name in PHASE_NAMES),
    "comp_type": tuple(f"pct_{COMP_TYPE_NAMES[comp]}" for comp in COMPETITION_TYPES),
}
MIX_COUNT_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "phase": tuple(f"n_{name}" for name in PHASE_NAMES),
    "comp_type": tuple(f"n_{COMP_TYPE_NAMES[comp]}" for comp in COMPETITION_TYPES),
}

#: Columns of the Pressure_Ball mix table and file (criterion 5).
PRESSURE_MIX_COLUMNS: tuple[str, ...] = (
    *KEY_COLUMNS, "n_pressure_balls", *MIX_PCT_COLUMNS["phase"], *MIX_PCT_COLUMNS["comp_type"],
    *MIX_COUNT_COLUMNS["phase"], *MIX_COUNT_COLUMNS["comp_type"],
)  # fmt: skip

#: Each group of a player's mix percentages sums to 100 within this, in percentage points (criterion 5).
MIX_TOLERANCE = 0.1

#: Float format of the traditional-stats and mix files: 10 decimals, as the Main_List files.
EVIDENCE_FLOAT_FORMAT = "%.10f"

#: Float format of each kind of evidence file; None writes Python's shortest round-trip form.
FLOAT_FORMATS: Mapping[str, str | None] = {
    TRADITIONAL: EVIDENCE_FLOAT_FORMAT, RATING_EVIDENCE: None, PRESSURE_MIX: EVIDENCE_FLOAT_FORMAT,
}  # fmt: skip

#: Columns that each input of evidence_deliveries must have.
VALUE_INPUT_COLUMNS: tuple[str, ...] = ("row", "match_id", "innings", "cell", "cls")
BALL_INPUT_COLUMNS: tuple[str, ...] = (
    "row", "match_id", "innings", "innings_ball_num", "cls", "striker_id", "non_striker_id", "bowler_id", "runs",
    "team_runs", "extras_type", "is_four", "is_six", "counted_wicket", "retire_out", "dismissed_is_striker",
    "dismissed_fallback", "bowler_wicket", "is_super_over", "comp_type",
)  # fmt: skip
PRESSURE_INPUT_COLUMNS: tuple[str, ...] = ("row", "live", "pressure_bat", "pressure_bowl")

#: Columns of the evidence_deliveries table (module docstring, "Delivery set").
DELIVERY_COLUMNS: tuple[str, ...] = (
    "row", "match_id", "innings", "innings_ball_num", "cls", "striker_id", "non_striker_id", "bowler_id", "bat_runs",
    "team_runs", "bye", "boundary", "counted_wicket", "retire_out", "dismissed_is_striker", "dismissed_fallback",
    "bowler_wicket", "live", "pressure_bat", "pressure_bowl", "phase", "comp_type",
)  # fmt: skip

#: Columns of the batter_dismissals table: one row per Batter_Dismissal, in ``row`` order. ``pressure``
#: is the set it falls in; ``striker`` is True when the dismissed batter is the striker.
BATTER_DISMISSAL_COLUMNS: tuple[str, ...] = ("row", "player_id", "pressure", "striker")

#: Columns that evidence_tables reads from the Main_List tables, the ranking and the bootstrap table.
LIST_INPUT_COLUMNS: tuple[str, ...] = (
    *KEY_COLUMNS, "final_score", "ci_low", "ci_high", "p_top10", "hit_rate", "share_ipl",
)  # fmt: skip
RANKING_INPUT_COLUMNS: tuple[str, ...] = (
    "role", "player_id", "eligible", "rank", *_RANKING_VALUES, "n_pressure_matches", "n_pressure_balls",
    "n_ordinary_balls",
)  # fmt: skip
BOOTSTRAP_INPUT_COLUMNS: tuple[str, ...] = ("role", "player_id", "main_rank", "main_final_score", *_BOOTSTRAP_VALUES)

_LEGAL = int(DeliveryClass.LEGAL)
_WIDE = int(DeliveryClass.WIDE)
_NO_BALL = int(DeliveryClass.NO_BALL)
_CLASS_CODES = np.asarray([int(code) for code in DeliveryClass], dtype=np.int64)
_PHASE_CODES = np.asarray([int(member) for member in Phase], dtype=np.int64)
_N_PHASES = len(Phase)
_BATTER_INPUTS: tuple[str, ...] = (
    "row", "match_id", "innings", "innings_ball_num", "cls", "striker_id", "non_striker_id", "bat_runs", "boundary",
    "counted_wicket", "retire_out", "dismissed_is_striker", "dismissed_fallback", "live", "pressure_bat",
)  # fmt: skip
_BOWLER_INPUTS: tuple[str, ...] = (
    "cls", "bowler_id", "bat_runs", "team_runs", "bye", "boundary", "bowler_wicket", "pressure_bowl",
)  # fmt: skip
_MIX_INPUTS: Mapping[str, tuple[str, ...]] = {
    BATTER_ROLE: ("cls", "striker_id", "pressure_bat", "phase", "comp_type"),
    BOWLER_ROLE: ("bowler_id", "pressure_bowl", "phase", "comp_type"),
}
_TMP = ".tmp"


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


def _whole(values: pd.Series, what: str) -> np.ndarray:
    """int64 values of a column of whole numbers with none missing."""
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
    """bool values of a boolean column with none missing."""
    if not pd.api.types.is_bool_dtype(values.dtype):
        raise TypeError(f"{what} must hold True or False; got dtype {values.dtype}")
    missing = int(values.isna().sum())
    if missing:
        raise ValueError(f"{what} has {missing:,} missing value(s)")
    return values.to_numpy(dtype=bool)


def _maybe_flags(values: pd.Series, what: str) -> tuple[np.ndarray, np.ndarray]:
    """(flags, missing) of a nullable boolean column; a missing value reads False in ``flags``."""
    if not pd.api.types.is_bool_dtype(values.dtype):
        raise TypeError(f"{what} must hold True, False or missing values; got dtype {values.dtype}")
    missing = values.isna().to_numpy(dtype=bool)
    return values.astype("boolean").to_numpy(dtype=bool, na_value=False), missing


def _texts(values: pd.Series, what: str) -> list[str | None]:
    """Each value as a str, or None where it is missing."""
    out: list[str | None] = []
    for value in values.tolist():
        if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
            out.append(None)
        elif isinstance(value, str):
            out.append(value)
        else:
            raise TypeError(f"{what} must hold text; found {value!r}")
    return out


def _count(n: int, noun: str) -> str:
    return f"{n:,} {noun}" if n == 1 else f"{n:,} {noun}s"


def _row_positions(keys: pd.Series, wanted: np.ndarray, what: str) -> np.ndarray:
    """The position in ``keys`` (a ``row`` column, each value once) of each wanted ``row``; ValueError when one is absent."""
    index = pd.Index(_whole(keys, f"{what} column 'row'"))
    if not index.is_unique:
        raise ValueError(f"{what} holds a row more than once")
    at = index.get_indexer(wanted)
    if (at < 0).any():
        raise ValueError(
            f"{what} has no row for {_count(int((at < 0).sum()), 'rated Delivery')}, for example row "
            f"{int(wanted[at < 0][0])}; the tables do not come from the same Run"
        )
    return at


def _player_positions(ids: pd.Series, wanted: np.ndarray, what: str) -> np.ndarray:
    """The position in ``ids`` (each player once) of each wanted ``player_id``; StageError when one is absent."""
    index = pd.Index(_whole(ids, f"{what} column 'player_id'"))
    if not index.is_unique:
        raise ValueError(f"{what} holds a player_id more than once")
    at = index.get_indexer(wanted)
    if (at < 0).any():
        absent = ", ".join(str(int(pid)) for pid in wanted[at < 0])
        raise StageError(f"{what} has no row for the listed player_id(s) {absent}; it does not come from the same Run")
    return at


def _role_rows(frame: pd.DataFrame, role: str, what: str) -> pd.DataFrame:
    """The rows of ``frame`` whose ``role`` is ``role``, on a RangeIndex."""
    roles = _texts(frame["role"], f"{what} column 'role'")
    unknown = sorted({str(value) for value in roles if value not in ROLES})
    if unknown:
        raise ValueError(f"{what} column 'role' holds {', '.join(unknown)}, which is not a Role")
    mask = np.fromiter((value == role for value in roles), dtype=bool, count=len(roles))
    return frame.loc[mask].reset_index(drop=True)


def _id_values(player_ids: Any) -> np.ndarray:
    """``player_ids`` (1-D whole numbers, each once) as int64."""
    if isinstance(player_ids, (str, bytes, Mapping, pd.DataFrame)) or np.ndim(player_ids) != 1:
        raise TypeError(f"player_ids must be a 1-D list, array or Series of ids; got {type(player_ids).__name__}")
    values = player_ids if isinstance(player_ids, pd.Series) else pd.Series(np.asarray(player_ids))
    if not len(values):
        return np.zeros(0, dtype=np.int64)
    ids = _whole(values, "player_ids")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("player_ids holds a player_id more than once")
    return ids


class _Listed(NamedTuple):
    """The key columns of a list table, checked."""

    rank: np.ndarray  # int64, 1 to n in order
    player_id: np.ndarray  # int64, each once
    player_name: tuple[str, ...]  # non-empty strings


def _listed(listed: Any, role: str) -> _Listed:
    what = f"the {role} list"
    _require(listed, KEY_COLUMNS, what)
    ranks = _whole(listed["rank"], f"{what} column 'rank'")
    if not len(ranks) or not np.array_equal(ranks, np.arange(1, len(ranks) + 1)):
        raise ValueError(f"{what} column 'rank' must run 1, 2, 3, ... in order; got {ranks.tolist()}")
    ids = _whole(listed["player_id"], f"{what} column 'player_id'")
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{what} repeats a player_id")
    names = _texts(listed["player_name"], f"{what} column 'player_name'")
    if any(not name for name in names):
        raise ValueError(f"{what} has a missing or empty player_name")
    return _Listed(ranks, ids, tuple(str(name) for name in names))


def _key_columns(keys: _Listed) -> dict[str, Any]:
    return {
        "rank": keys.rank.astype(np.int64),
        "player_id": keys.player_id.astype(np.int64),
        "player_name": pd.array(list(keys.player_name), dtype="str"),
    }


# ============================================================================ Delivery set

def evidence_deliveries(value: pd.DataFrame, balls: pd.DataFrame, pressure: pd.DataFrame) -> pd.DataFrame:
    """The rated Deliveries with the columns of the evidence tables (DELIVERY_COLUMNS; module docstring).

    ``value``: the ``value`` checkpoint's ``value`` table, one row per rated Delivery
    (VALUE_INPUT_COLUMNS). ``balls``: the ``reconstruct`` checkpoint's ``balls`` table
    (BALL_INPUT_COLUMNS). ``pressure``: the ``pressure`` table (PRESSURE_INPUT_COLUMNS). Returns
    one row per ``value`` row, in ``value`` order, on a RangeIndex. Raises ValueError or TypeError
    when a column is missing or of the wrong type, a ``value`` row has no ``balls`` or ``pressure``
    row, the tables disagree on a row's match, innings or class, a row is in a Super_Over_Innings,
    a ``cell`` is not the row's innings x Phase cell, a Boundary flag is missing, a Batter_Delivery
    has no ``pressure_bat`` or a Competition_Type is unknown.
    """
    _require(value, VALUE_INPUT_COLUMNS, "value")
    _require(balls, BALL_INPUT_COLUMNS, "balls")
    _require(pressure, PRESSURE_INPUT_COLUMNS, "pressure")
    rows = _whole(value["row"], "value column 'row'")
    if len(np.unique(rows)) != len(rows):
        raise ValueError("value holds a row more than once")
    b = balls.iloc[_row_positions(balls["row"], rows, "balls")].reset_index(drop=True)
    p = pressure.iloc[_row_positions(pressure["row"], rows, "pressure")].reset_index(drop=True)

    match_id = _whole(value["match_id"], "value column 'match_id'")
    innings = _whole(value["innings"], "value column 'innings'")
    cls = _whole(value["cls"], "value column 'cls'")
    for column, own in (("match_id", match_id), ("innings", innings), ("cls", cls)):
        other = _whole(b[column], f"balls column {column!r}")
        if not np.array_equal(own, other):
            raise ValueError(
                f"value and balls disagree on {column!r} for {_count(int((own != other).sum()), 'row')}; the tables do "
                "not come from the same Run"
            )
    if not np.isin(innings, STATE_INNINGS).all() or _flags(b["is_super_over"], "balls column 'is_super_over'").any():
        raise ValueError("value must hold Deliveries of innings 1 and 2 only: no Super_Over_Innings is rated")
    if not np.isin(cls, _CLASS_CODES).all():
        raise ValueError(f"value column 'cls' must hold DeliveryClass codes {_CLASS_CODES.tolist()}")
    phase = _whole(value["cell"], "value column 'cell'") - _N_PHASES * (innings - 1)
    if not np.isin(phase, _PHASE_CODES).all():
        raise ValueError("value column 'cell' must hold each row's innings x Phase cell, 3 x (innings - 1) + Phase")

    extras = b["extras_type"]
    bye = (cls == _LEGAL) & extras.isin(list(BYE_EXTRAS_TYPES)).to_numpy(dtype=bool, na_value=False)
    four, no_four = _maybe_flags(b["is_four"], "balls column 'is_four'")
    six, no_six = _maybe_flags(b["is_six"], "balls column 'is_six'")
    if (no_four | no_six).any():
        raise ValueError(
            f"balls has no is_four or is_six value on {_count(int((no_four | no_six).sum()), 'rated Delivery')}, so "
            "its Boundaries cannot be counted"
        )
    flag_bat, no_flag = _maybe_flags(p["pressure_bat"], "pressure column 'pressure_bat'")
    if (no_flag & (cls != _WIDE)).any():
        raise ValueError(
            f"pressure column 'pressure_bat' is missing on {_count(int((no_flag & (cls != _WIDE)).sum()), 'Batter_Delivery')}"
            "; the batter-Role Pressure_Ball flag is missing on Wides only"
        )
    comp = _texts(b["comp_type"], "balls column 'comp_type'")
    unknown = sorted({str(text) for text in comp if text not in COMPETITION_TYPES})
    if unknown:
        raise ValueError(
            f"balls column 'comp_type' holds {', '.join(unknown)}; the Competition_Types are {', '.join(COMPETITION_TYPES)}"
        )
    bat_runs = _whole(b["runs"], "balls column 'runs'")
    team_runs = _whole(b["team_runs"], "balls column 'team_runs'")
    if (bat_runs < 0).any() or (team_runs < 0).any():
        raise ValueError("balls columns 'runs' and 'team_runs' must hold 0 or more runs")
    return pd.DataFrame(
        {
            "row": rows,
            "match_id": match_id,
            "innings": innings,
            "innings_ball_num": _whole(b["innings_ball_num"], "balls column 'innings_ball_num'"),
            "cls": cls.astype(np.int8),
            "striker_id": _whole(b["striker_id"], "balls column 'striker_id'"),
            "non_striker_id": pd.array(b["non_striker_id"].to_numpy(dtype=object, na_value=pd.NA), dtype="Int64"),
            "bowler_id": _whole(b["bowler_id"], "balls column 'bowler_id'"),
            "bat_runs": bat_runs,
            "team_runs": team_runs,
            "bye": bye,
            "boundary": four | six,
            "counted_wicket": _flags(b["counted_wicket"], "balls column 'counted_wicket'"),
            "retire_out": _flags(b["retire_out"], "balls column 'retire_out'"),
            "dismissed_is_striker": pd.array(b["dismissed_is_striker"].to_numpy(dtype=object), dtype="boolean"),
            "dismissed_fallback": _flags(b["dismissed_fallback"], "balls column 'dismissed_fallback'"),
            "bowler_wicket": _flags(b["bowler_wicket"], "balls column 'bowler_wicket'"),
            "live": _flags(p["live"], "pressure column 'live'"),
            "pressure_bat": pd.arrays.BooleanArray(flag_bat, no_flag),
            "pressure_bowl": _flags(p["pressure_bowl"], "pressure column 'pressure_bowl'"),
            "phase": phase.astype(np.int8),
            "comp_type": pd.array(comp, dtype="str"),
        },
        columns=list(DELIVERY_COLUMNS),
    )


def load_evidence_deliveries(store: CheckpointStore) -> pd.DataFrame:
    """``evidence_deliveries`` of the ``value``, ``reconstruct`` and ``pressure`` checkpoints of a Run.

    StageError (from the store) when a checkpoint file is missing or changed.
    """
    rated = store.load_table(VALUE_STAGE, VALUE_TABLE)
    balls = store.load_table(RECONSTRUCT_STAGE, BALLS_TABLE)
    flags = store.load_table(PRESSURE_STAGE, PRESSURE_TABLE)
    _require(rated, VALUE_INPUT_COLUMNS, f"the {VALUE_STAGE} table {VALUE_TABLE}")
    _require(balls, BALL_INPUT_COLUMNS, f"the {RECONSTRUCT_STAGE} table {BALLS_TABLE}")
    _require(flags, PRESSURE_INPUT_COLUMNS, f"the {PRESSURE_STAGE} table {PRESSURE_TABLE}")
    return evidence_deliveries(
        rated.loc[:, list(VALUE_INPUT_COLUMNS)],
        balls.loc[:, list(BALL_INPUT_COLUMNS)],
        flags.loc[:, list(PRESSURE_INPUT_COLUMNS)],
    )


# ==================================================================== traditional statistics

def batter_dismissals(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Every Batter_Dismissal in ``deliveries`` and its ball set (BATTER_DISMISSAL_COLUMNS; module docstring).

    ``deliveries`` needs the batter columns of DELIVERY_COLUMNS (``row``, the order keys,
    ``cls``, ``striker_id``, ``non_striker_id``, the dismissal flags, ``live`` and
    ``pressure_bat``). A Retire Out is never a Batter_Dismissal. Raises the errors of
    ``value.dismissed_batters`` and ``value.dismissal_charges``.
    """
    _require(deliveries, _BATTER_INPUTS, "deliveries")
    d = deliveries.reset_index(drop=True)
    n_rows = len(d)
    out = dismissed_batters(d, retire_out_penalty=False)
    cls = _whole(d["cls"], "deliveries column 'cls'")
    flag, missing = _maybe_flags(d["pressure_bat"], "deliveries column 'pressure_bat'")
    own = np.flatnonzero(out.striker & (cls != _WIDE))
    if missing[own].any():
        raise ValueError("deliveries column 'pressure_bat' is missing on a Batter_Delivery whose striker is dismissed")
    # The other Batter_Dismissals are the Dismissal_Charges with Retire Out uncharged; the charge itself is not used.
    charges = dismissal_charges(d, np.zeros(n_rows), retire_out_penalty=False)
    what = "the Dismissal_Charges column"
    rows = np.concatenate([_whole(d["row"], "deliveries column 'row'")[own], _whole(charges["row"], f"{what} 'row'")])
    players = np.concatenate([
        _whole(d["striker_id"], "deliveries column 'striker_id'")[own],
        _whole(charges["player_id"], f"{what} 'player_id'"),
    ])  # fmt: skip
    pressure = np.concatenate([flag[own], _flags(charges["pressure"], f"{what} 'pressure'")])
    striker = np.concatenate([np.ones(len(own), dtype=bool), ~_flags(charges["non_striker"], f"{what} 'non_striker'")])
    order = np.argsort(rows, kind="stable")
    return pd.DataFrame(
        {"row": rows[order], "player_id": players[order], "pressure": pressure[order], "striker": striker[order]},
        columns=list(BATTER_DISMISSAL_COLUMNS),
    )


def _per_set(
    wanted: np.ndarray, ids: np.ndarray, pressure: np.ndarray, mask: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    """(players, 2) int64: per wanted player, the ``mask`` rows (or the sum of their ``weights``) on
    Pressure_Balls (column 0) and on Ordinary_Balls (column 1)."""
    at = pd.Index(wanted).get_indexer(ids) if len(wanted) else np.full(len(ids), -1, dtype=np.int64)
    keep = mask & (at >= 0)
    codes = at[keep] * 2 + np.where(pressure[keep], 0, 1)
    sums = np.zeros(2 * len(wanted), dtype=np.int64)
    np.add.at(sums, codes, 1 if weights is None else weights[keep].astype(np.int64))
    return sums.reshape(len(wanted), 2)


def _count_frame(wanted: np.ndarray, counts: Mapping[str, np.ndarray], names: Sequence[str]) -> pd.DataFrame:
    data: dict[str, Any] = {"player_id": wanted.astype(np.int64)}
    for column, ball_set in enumerate(BALL_SETS):
        for name in names:
            data[f"{ball_set}_{name}"] = counts[name][:, column]
    return pd.DataFrame(data)


def batter_counts(deliveries: pd.DataFrame, player_ids: Any) -> pd.DataFrame:
    """The unweighted batter counts of each player on Pressure_Balls and Ordinary_Balls (module docstring).

    ``deliveries``: an evidence_deliveries table (its batter columns); ``player_ids``: the
    players, each once. Returns one row per player, in the order given: ``player_id`` and
    ``<set>_<count>`` for each set of BALL_SETS and count of BATTER_COUNTS (int64). A player with
    no Batter_Delivery and no Batter_Dismissal has zero counts.
    """
    wanted = _id_values(player_ids)
    _require(deliveries, _BATTER_INPUTS, "deliveries")
    cls = _whole(deliveries["cls"], "deliveries column 'cls'")
    striker = _whole(deliveries["striker_id"], "deliveries column 'striker_id'")
    bat = _whole(deliveries["bat_runs"], "deliveries column 'bat_runs'")
    boundary = _flags(deliveries["boundary"], "deliveries column 'boundary'")
    flag, missing = _maybe_flags(deliveries["pressure_bat"], "deliveries column 'pressure_bat'")
    batter = cls != _WIDE
    if (missing & batter).any():
        raise ValueError("deliveries column 'pressure_bat' is missing on a Batter_Delivery")
    if (bat < 0).any():
        raise ValueError("deliveries column 'bat_runs' must hold 0 or more runs")
    legal = cls == _LEGAL
    out = batter_dismissals(deliveries)
    counts = {
        "deliveries": _per_set(wanted, striker, flag, batter),
        "runs": _per_set(wanted, striker, flag, batter, bat),
        "dismissals": _per_set(
            wanted, out["player_id"].to_numpy(dtype=np.int64), out["pressure"].to_numpy(dtype=bool),
            np.ones(len(out), dtype=bool),
        ),  # fmt: skip
        "legal_balls": _per_set(wanted, striker, flag, batter & legal),
        "dot_balls": _per_set(wanted, striker, flag, batter & legal & (bat == 0)),
        "boundaries": _per_set(wanted, striker, flag, batter & boundary),
    }
    return _count_frame(wanted, counts, BATTER_COUNTS)


def bowler_counts(deliveries: pd.DataFrame, player_ids: Any) -> pd.DataFrame:
    """The unweighted bowler counts of each player on Pressure_Balls and Ordinary_Balls (module docstring).

    ``deliveries``: an evidence_deliveries table (its bowler columns); ``player_ids``: the
    players, each once. Returns one row per player, in the order given: ``player_id`` and
    ``<set>_<count>`` for each set of BALL_SETS and count of BOWLER_COUNTS (int64).
    ``runs_conceded`` is Bowler_Runs_Conceded: ``team_runs``, except 0 on a bye or leg-bye.
    """
    wanted = _id_values(player_ids)
    _require(deliveries, _BOWLER_INPUTS, "deliveries")
    cls = _whole(deliveries["cls"], "deliveries column 'cls'")
    bowler = _whole(deliveries["bowler_id"], "deliveries column 'bowler_id'")
    bat = _whole(deliveries["bat_runs"], "deliveries column 'bat_runs'")
    team = _whole(deliveries["team_runs"], "deliveries column 'team_runs'")
    bye = _flags(deliveries["bye"], "deliveries column 'bye'")
    boundary = _flags(deliveries["boundary"], "deliveries column 'boundary'")
    wicket = _flags(deliveries["bowler_wicket"], "deliveries column 'bowler_wicket'")
    flag = _flags(deliveries["pressure_bowl"], "deliveries column 'pressure_bowl'")
    if (bat < 0).any() or (team < 0).any():
        raise ValueError("deliveries columns 'bat_runs' and 'team_runs' must hold 0 or more runs")
    legal = cls == _LEGAL
    if (bye & ~legal).any():
        raise ValueError("deliveries column 'bye' marks a Delivery that is not a Legal_Ball")
    every = np.ones(len(cls), dtype=bool)
    counts = {
        "deliveries": _per_set(wanted, bowler, flag, every),
        "legal_balls": _per_set(wanted, bowler, flag, legal),
        "wickets": _per_set(wanted, bowler, flag, wicket),
        "runs_conceded": _per_set(wanted, bowler, flag, every, np.where(bye, 0, team)),
        "dot_balls": _per_set(wanted, bowler, flag, legal & (bat == 0)),
        "boundaries": _per_set(wanted, bowler, flag, boundary),
        "wides": _per_set(wanted, bowler, flag, cls == _WIDE),
        "no_balls": _per_set(wanted, bowler, flag, cls == _NO_BALL),
    }
    return _count_frame(wanted, counts, BOWLER_COUNTS)


def ball_set_rates(counts: Mapping[str, Any], role: str) -> dict[str, np.ndarray]:
    """The rates of RATES[role] from one ball set's counts (criteria 1-3).

    ``counts`` maps each count of COUNTS[role] to the values of the players (arrays of the same
    length). Returns ``{rate: float64 array}`` in RATES[role] order, NaN where the denominator is
    0 (criterion 3).
    """
    out: dict[str, np.ndarray] = {}
    for name, formula in RATES[_role(role)].items():
        denominator = np.asarray(counts[formula.denominator], dtype=np.float64)
        numerator = np.zeros(len(denominator))
        for part in formula.numerator:
            numerator = numerator + np.asarray(counts[part], dtype=np.float64)
        values = np.full(len(denominator), np.nan)
        ok = denominator > 0
        values[ok] = formula.scale * numerator[ok] / denominator[ok]
        out[name] = values
    return out


def traditional_table(role: str, listed: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    """One Role's traditional-stats table (TRADITIONAL_COLUMNS[role]; criteria 1-3; module docstring).

    ``listed``: the Role's list table (KEY_COLUMNS, ranks 1, 2, ... in order, such as a
    Main_List); ``deliveries``: an evidence_deliveries table. One row per listed player, in rank
    order: the key columns, then ``pressure_<stat>`` and ``ordinary_<stat>`` for STATS[role];
    counts are int64 and rates float64, NaN where the denominator is 0.
    """
    role = _role(role)
    keys = _listed(listed, role)
    counter = batter_counts if role == BATTER_ROLE else bowler_counts
    counts = counter(deliveries, keys.player_id)
    data = _key_columns(keys)
    for ball_set in BALL_SETS:
        values: dict[str, Any] = {
            name: counts[f"{ball_set}_{name}"].to_numpy(dtype=np.int64) for name in COUNTS[role]
        }
        values.update(ball_set_rates(values, role))
        for stat in STATS[role]:
            data[f"{ball_set}_{stat}"] = values[stat]
    return pd.DataFrame(data, columns=list(TRADITIONAL_COLUMNS[role]))


# ========================================================================= rating evidence

def rating_evidence_table(
    role: str, listed: pd.DataFrame, *, ranking: pd.DataFrame, bootstrap: pd.DataFrame
) -> pd.DataFrame:
    """One Role's rating-evidence table (RATING_EVIDENCE_COLUMNS; criterion 4; module docstring).

    ``listed``: the Role's Main_List (KEY_COLUMNS); ``ranking``: the rating checkpoint's
    ``player_estimates`` (RANKING_INPUT_COLUMNS); ``bootstrap``: the bootstrap checkpoint's
    ``bootstrap`` table (BOOTSTRAP_INPUT_COLUMNS). Every value is copied unchanged. Raises
    StageError when a listed player is not the ranking's Eligible_Player at the list's rank, has
    no bootstrap row, or the bootstrap row holds another rank or Final_Score.
    """
    role = _role(role)
    keys = _listed(listed, role)
    _require(ranking, RANKING_INPUT_COLUMNS, "ranking")
    _require(bootstrap, BOOTSTRAP_INPUT_COLUMNS, "bootstrap")
    rows = _role_rows(ranking, role, "ranking")
    at = _player_positions(rows["player_id"], keys.player_id, f"the {role} rows of ranking")
    eligible = _flags(rows["eligible"], "ranking column 'eligible'")[at]
    rank = _numbers(rows["rank"], "ranking column 'rank'")[at]
    if not (eligible.all() and np.array_equal(rank, keys.rank.astype(np.float64))):
        raise StageError(
            f"the {role} list does not hold the ranking's Eligible_Players at ranks 1 to {len(keys.rank)} in order; "
            "the tables do not come from the same Run"
        )
    boot = _role_rows(bootstrap, role, "bootstrap")
    at_boot = _player_positions(boot["player_id"], keys.player_id, f"the {role} rows of the bootstrap table")
    final_score = _numbers(rows["final_score"], "ranking column 'final_score'")[at]
    main_rank = _numbers(boot["main_rank"], "bootstrap column 'main_rank'")[at_boot]
    main_score = _numbers(boot["main_final_score"], "bootstrap column 'main_final_score'")[at_boot]
    if not (np.array_equal(main_rank, rank) and np.array_equal(main_score, final_score)):
        raise StageError(
            f"the bootstrap table does not hold the {role} ranking's ranks and Final_Scores for the listed players; "
            "the tables do not come from the same Run"
        )
    data = _key_columns(keys)
    for column in RATING_EVIDENCE_COLUMNS[len(KEY_COLUMNS):]:
        if RATING_SOURCES[column] == "bootstrap":
            data[column] = _numbers(boot[column], f"bootstrap column {column!r}")[at_boot]
        elif column == "n_pressure_matches":
            data[column] = _whole(rows[column], f"ranking column {column!r}")[at]
        else:
            data[column] = _numbers(rows[column], f"ranking column {column!r}")[at]
    return pd.DataFrame(data, columns=list(RATING_EVIDENCE_COLUMNS))


# ======================================================================= Pressure_Ball mix

def pressure_mix_table(role: str, listed: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    """One Role's Pressure_Ball mix table (PRESSURE_MIX_COLUMNS; criterion 5; module docstring).

    ``listed``: the Role's list table (KEY_COLUMNS); ``deliveries``: an evidence_deliveries table
    (``cls``, ``striker_id`` and ``pressure_bat``, or ``bowler_id`` and ``pressure_bowl``, with
    ``phase`` and ``comp_type``). One row per listed player, in rank order: ``n_pressure_balls``,
    the Phase and Competition_Type percentages (float64; NaN for a player with no Pressure_Ball)
    and the counts behind them (int64).
    """
    role = _role(role)
    keys = _listed(listed, role)
    _require(deliveries, _MIX_INPUTS[role], "deliveries")
    if role == BATTER_ROLE:
        ids = _whole(deliveries["striker_id"], "deliveries column 'striker_id'")
        flag, missing = _maybe_flags(deliveries["pressure_bat"], "deliveries column 'pressure_bat'")
        mask = _whole(deliveries["cls"], "deliveries column 'cls'") != _WIDE
        if (missing & mask).any():
            raise ValueError("deliveries column 'pressure_bat' is missing on a Batter_Delivery")
    else:
        ids = _whole(deliveries["bowler_id"], "deliveries column 'bowler_id'")
        flag = _flags(deliveries["pressure_bowl"], "deliveries column 'pressure_bowl'")
        mask = np.ones(len(ids), dtype=bool)
    phase = _whole(deliveries["phase"], "deliveries column 'phase'")
    comp = pd.Index(list(COMPETITION_TYPES)).get_indexer(_texts(deliveries["comp_type"], "deliveries column 'comp_type'"))
    n_players = len(keys.player_id)
    at = pd.Index(keys.player_id).get_indexer(ids)
    keep = mask & flag & (at >= 0)
    if not np.isin(phase[keep], _PHASE_CODES).all():
        raise ValueError(f"deliveries column 'phase' must hold Phase codes {_PHASE_CODES.tolist()}")
    if (comp[keep] < 0).any():
        raise ValueError(f"deliveries column 'comp_type' must hold the Competition_Types {', '.join(COMPETITION_TYPES)}")
    groups = {"phase": (phase[keep], len(_PHASE_CODES)), "comp_type": (comp[keep], len(COMPETITION_TYPES))}
    counts: dict[str, np.ndarray] = {}
    for group, (codes, width) in groups.items():
        matrix = np.zeros((n_players, width), dtype=np.int64)
        np.add.at(matrix, (at[keep], codes), 1)
        counts[group] = matrix
    total = counts["phase"].sum(axis=1)
    data = _key_columns(keys)
    data["n_pressure_balls"] = total
    with np.errstate(divide="ignore", invalid="ignore"):
        for group in groups:
            shares = np.where(total[:, None] > 0, 100.0 * counts[group] / total[:, None], np.nan)
            for j, column in enumerate(MIX_PCT_COLUMNS[group]):
                data[column] = shares[:, j]
    for group in groups:
        for j, column in enumerate(MIX_COUNT_COLUMNS[group]):
            data[column] = counts[group][:, j]
    return pd.DataFrame(data, columns=list(PRESSURE_MIX_COLUMNS))


# ======================================================================== the six tables

def _check_rated_counts(role: str, traditional: pd.DataFrame, mix: pd.DataFrame, ranking: pd.DataFrame) -> None:
    """StageError unless the evidence counts of the listed players equal the rating's."""
    rows = _role_rows(ranking, role, "ranking")
    ids = traditional["player_id"].to_numpy(dtype=np.int64)
    at = _player_positions(rows["player_id"], ids, f"the {role} rows of ranking")
    pressure_balls = _whole(rows["n_pressure_balls"], "ranking column 'n_pressure_balls'")[at]
    ordinary_balls = _whole(rows["n_ordinary_balls"], "ranking column 'n_ordinary_balls'")[at]
    checks = (
        (f"{PRESSURE_SET}_deliveries", traditional[f"{PRESSURE_SET}_deliveries"], pressure_balls),
        (f"{ORDINARY_SET}_deliveries", traditional[f"{ORDINARY_SET}_deliveries"], ordinary_balls),
        ("n_pressure_balls", mix["n_pressure_balls"], pressure_balls),
    )
    for column, found, expected in checks:
        values = found.to_numpy(dtype=np.int64)
        if not np.array_equal(values, expected):
            first = int(np.flatnonzero(values != expected)[0])
            raise StageError(
                f"the evidence Deliveries give the {role} at rank {first + 1} (player_id {int(ids[first])}) {column} "
                f"{int(values[first]):,}, but the rating counts {int(expected[first]):,}; the tables do not come from the "
                "same Run"
            )


def _check_list_values(role: str, listed: pd.DataFrame, evidence: pd.DataFrame, mix: pd.DataFrame) -> None:
    """StageError unless the rating evidence and the IPL count agree with the Main_List's values."""
    for column in ("final_score", "ci_low", "ci_high", "p_top10", "hit_rate"):
        shown = _numbers(listed[column], f"the {role} list column {column!r}")
        copied = evidence[column].to_numpy(dtype=np.float64)
        if not np.array_equal(shown, copied, equal_nan=True):
            raise StageError(
                f"the {role} rating evidence and the {role} Main_List disagree on {column!r}; they do not come from the "
                "same Run"
            )
    share = _numbers(listed["share_ipl"], f"the {role} list column 'share_ipl'")
    n_ipl = mix[f"n_{COMP_TYPE_NAMES[COMP_IPL]}"].to_numpy(dtype=np.int64)
    total = mix["n_pressure_balls"].to_numpy(dtype=np.int64)
    if not np.array_equal(n_ipl.astype(np.float64) / total, share):
        raise StageError(
            f"the {role} Pressure_Ball mix counts other IPL Pressure_Balls than the Main_List's share_ipl; the tables do "
            "not come from the same Run"
        )


def evidence_tables(
    lists: Mapping[str, pd.DataFrame],
    *,
    deliveries: pd.DataFrame,
    ranking: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """The six evidence tables of both Roles, ``{name: table}`` in EVIDENCE_TABLES order (criteria 1-5).

    ``lists``: each Role's Main_List table (``lists.main_lists``; LIST_INPUT_COLUMNS);
    ``deliveries``: ``load_evidence_deliveries``; ``ranking``: the rating checkpoint's
    ``player_estimates``; ``bootstrap``: the bootstrap checkpoint's ``bootstrap`` table. Checks
    that each listed player's Pressure_Ball and Ordinary_Ball counts equal the rating's
    ``n_pressure_balls`` and ``n_ordinary_balls``, that the copied values equal the Main_List's,
    and that the IPL count gives the Main_List's ``share_ipl`` (StageError otherwise), then
    validates every table (``validate_evidence``).
    """
    if not isinstance(lists, Mapping):
        raise TypeError(f"lists must map each Role to its Main_List table; got {type(lists).__name__}")
    if set(lists) != set(ROLES):
        raise ValueError(f"lists must hold exactly the Roles {', '.join(ROLES)}; got {', '.join(map(str, lists))}")
    _require(ranking, RANKING_INPUT_COLUMNS, "ranking")
    tables: dict[str, pd.DataFrame] = {}
    for role in ROLES:
        listed = lists[role]
        _require(listed, LIST_INPUT_COLUMNS, f"the {role} list")
        traditional = traditional_table(role, listed, deliveries)
        evidence = rating_evidence_table(role, listed, ranking=ranking, bootstrap=bootstrap)
        mix = pressure_mix_table(role, listed, deliveries)
        _check_rated_counts(role, traditional, mix, ranking)
        _check_list_values(role, listed, evidence, mix)
        tables[table_name(TRADITIONAL, role)] = traditional
        tables[table_name(RATING_EVIDENCE, role)] = evidence
        tables[table_name(PRESSURE_MIX, role)] = mix
    for name in EVIDENCE_TABLES:
        validate_evidence(name, tables[name])
    return {name: tables[name] for name in EVIDENCE_TABLES}


def _columns_for(kind: str, role: str) -> tuple[str, ...]:
    if kind == TRADITIONAL:
        return TRADITIONAL_COLUMNS[role]
    return RATING_EVIDENCE_COLUMNS if kind == RATING_EVIDENCE else PRESSURE_MIX_COLUMNS


def validate_evidence(name: str, table: pd.DataFrame) -> None:
    """Raise ValueError, listing every problem, unless ``table`` is a writable evidence table ``name``.

    ``name`` is one of EVIDENCE_TABLES. Every table needs exactly its columns, LIST_SIZE rows with
    ranks 1 to 10 in order, 10 distinct ``player_id`` values and a non-empty ``player_name`` on
    every row. A traditional-stats table needs counts of 0 or more and each rate NaN exactly where
    its denominator is 0 (criterion 3). A rating-evidence table needs a finite value in every
    column except ``usual``, ``ci_low`` and ``ci_high``, which may be missing, and 1 or more
    Pressure_Matches. A mix table needs 1 or more Pressure_Balls, counts that sum to them in each
    group, and percentages that sum to 100 within MIX_TOLERANCE in each group (criterion 5).
    """
    if name not in _TABLE_KIND_ROLE:
        raise ValueError(f"name must be one of {', '.join(EVIDENCE_TABLES)}; got {name!r}")
    kind, role = _TABLE_KIND_ROLE[name]
    what = f"the evidence table {name}"
    if not isinstance(table, pd.DataFrame):
        raise TypeError(f"{what} must be a DataFrame; got {type(table).__name__}")
    columns = _columns_for(kind, role)
    if list(table.columns) != list(columns):
        raise ValueError(
            f"{what} must have the columns {', '.join(columns)} in that order; got {', '.join(map(str, table.columns))}"
        )
    problems: list[str] = []
    if len(table) != LIST_SIZE:
        problems.append(f"it has {_count(len(table), 'row')}, not {LIST_SIZE}")
    integers = [column for column in ("rank", "player_id") if not pd.api.types.is_integer_dtype(table[column].dtype)]
    problems.extend(f"column {column!r} must have an integer dtype" for column in integers)
    if not integers:
        ranks = table["rank"].to_numpy(dtype=np.int64)
        if not np.array_equal(ranks, np.arange(1, len(table) + 1)):
            problems.append(f"column 'rank' must run from 1 in order; got {ranks.tolist()}")
        if table["player_id"].duplicated().any():
            problems.append("column 'player_id' repeats a player")
    names = _texts(table["player_name"], f"{what} column 'player_name'")
    if any(not text for text in names):
        problems.append("column 'player_name' has a missing or empty name")
    if kind == TRADITIONAL:
        problems.extend(_traditional_problems(table, role))
    elif kind == RATING_EVIDENCE:
        problems.extend(_rating_problems(table))
    else:
        problems.extend(_mix_problems(table))
    if problems:
        raise ValueError(f"{what} cannot be written: {'; '.join(problems)}")


def _integer_column(table: pd.DataFrame, column: str, problems: list[str]) -> np.ndarray | None:
    if not pd.api.types.is_integer_dtype(table[column].dtype) or table[column].isna().any():
        problems.append(f"column {column!r} must hold whole numbers in an integer dtype")
        return None
    return table[column].to_numpy(dtype=np.int64)


def _float_column(table: pd.DataFrame, column: str, problems: list[str]) -> np.ndarray | None:
    if not pd.api.types.is_float_dtype(table[column].dtype):
        problems.append(f"column {column!r} must have a float dtype; got {table[column].dtype}")
        return None
    return table[column].to_numpy(dtype=np.float64, na_value=np.nan)


def _traditional_problems(table: pd.DataFrame, role: str) -> list[str]:
    problems: list[str] = []
    for ball_set in BALL_SETS:
        counts: dict[str, np.ndarray] = {}
        for name in COUNTS[role]:
            values = _integer_column(table, f"{ball_set}_{name}", problems)
            if values is not None:
                if (values < 0).any():
                    problems.append(f"column {ball_set}_{name!s} has a negative count")
                counts[name] = values
        for name, formula in RATES[role].items():
            column = f"{ball_set}_{name}"
            values = _float_column(table, column, problems)
            if values is None or formula.denominator not in counts:
                continue
            empty = counts[formula.denominator] == 0
            if not np.array_equal(np.isnan(values), empty):
                problems.append(f"column {column!r} must be empty exactly where {ball_set}_{formula.denominator} is 0")
            if np.isinf(values).any():
                problems.append(f"column {column!r} has an infinite value")
    return problems


def _rating_problems(table: pd.DataFrame) -> list[str]:
    problems: list[str] = []
    may_be_missing = {"usual", "ci_low", "ci_high"}
    for column in RATING_EVIDENCE_COLUMNS[len(KEY_COLUMNS):-1]:
        values = _float_column(table, column, problems)
        if values is None:
            continue
        if column in may_be_missing:
            if np.isinf(values).any():
                problems.append(f"column {column!r} has an infinite value")
        elif not np.isfinite(values).all():
            problems.append(f"column {column!r} has a missing or infinite value")
    matches = _integer_column(table, "n_pressure_matches", problems)
    if matches is not None and (matches < 1).any():
        problems.append("column 'n_pressure_matches' must be 1 or more: a listed player has Pressure_Matches")
    return problems


def _mix_problems(table: pd.DataFrame) -> list[str]:
    problems: list[str] = []
    total = _integer_column(table, "n_pressure_balls", problems)
    if total is not None and (total < 1).any():
        problems.append("column 'n_pressure_balls' must be 1 or more: a listed player has Pressure_Balls")
    for group in ("phase", "comp_type"):
        counts = [_integer_column(table, column, problems) for column in MIX_COUNT_COLUMNS[group]]
        shares = [_float_column(table, column, problems) for column in MIX_PCT_COLUMNS[group]]
        if total is not None and all(values is not None for values in counts):
            if not np.array_equal(sum(counts), total):  # type: ignore[arg-type]
                problems.append(f"the {group} counts of a row do not sum to n_pressure_balls")
        if all(values is not None for values in shares):
            summed = sum(shares)  # type: ignore[arg-type]
            if not (np.abs(summed - 100.0) <= MIX_TOLERANCE).all():
                problems.append(f"the {group} percentages of a row do not sum to 100 within {MIX_TOLERANCE:g}")
    return problems


# ==================================================================================== files

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


def write_evidence(tables: Mapping[str, pd.DataFrame], outdir: Path | str) -> dict[str, Path]:
    """Write the six evidence tables to ``outdir`` (EVIDENCE_FILES; criterion 10) and return ``{name: path}``.

    ``tables``: ``evidence_tables``. Every table is validated first (ValueError, and no file is
    written, when one fails); the files are then written through ``<file>.tmp`` and renamed once
    all six are written. Format: module docstring, "Files".
    """
    if not isinstance(tables, Mapping):
        raise TypeError(f"tables must map each evidence table name to its table; got {type(tables).__name__}")
    if set(tables) != set(EVIDENCE_TABLES):
        raise ValueError(f"tables must hold exactly {', '.join(EVIDENCE_TABLES)}; got {', '.join(map(str, tables))}")
    for name in EVIDENCE_TABLES:
        validate_evidence(name, tables[name])
    folder = Path(outdir)
    folder.mkdir(parents=True, exist_ok=True)
    targets = {name: folder / EVIDENCE_FILES[name] for name in EVIDENCE_TABLES}

    def writer(table: pd.DataFrame, float_format: str | None) -> Callable[[Path], object]:
        return lambda temporary: table.to_csv(
            temporary, index=False, encoding="utf-8", lineterminator="\n", float_format=float_format
        )

    _write_files(
        {targets[name]: writer(tables[name].reset_index(drop=True), FLOAT_FORMATS[_TABLE_KIND_ROLE[name][0]])
         for name in EVIDENCE_TABLES}
    )  # fmt: skip
    return targets
