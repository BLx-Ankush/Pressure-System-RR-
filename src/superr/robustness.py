"""Robustness_Suite: the uncertainty bootstrap (Requirement 19), the repeatability checks
(Requirement 20), the cross-checks against simpler methods (Requirement 22), and the two
Pipeline_Stages ``bootstrap`` and ``robustness``; design section "Robustness_Suite and activity
comparison" and Data Models, ``bootstrap``, ``resample_draws`` and ``robustness_table``. The
ablation and sensitivity cases of Requirement 21 live in ``superr.cases``, which the ``robustness``
stage drives (``all_cases``, ``run_all_cases``, ``robustness_table``).

No function below ``run_bootstrap_stage`` or ``run_stage`` reads ``config.py``: every Config value
is an argument that the stage functions read from their ``Settings``.

Draws (``draw_resamples``, ``multiplicities``)
    A Bootstrap_Resample draws N matches uniformly at random, with replacement, from the N matches
    that have a Delivery outside Super_Over_Innings (glossary). They are the ``resampled`` matches
    of the ``rating`` checkpoint's ``match_universe``: 867 of its 883 matches as delivered.
    ``draw_resamples(N, B, seed)`` draws every resample at once from
    ``numpy.random.Generator(PCG64(seed))``, with ``seed`` = ``seeds.bootstrap`` (Requirement 28,
    criterion 3), and returns a (B, N) int16 array. Row b holds resample b's N drawn positions
    (0 to N - 1, into the resampled matches) in draw order. The same seed always gives the same
    array (Requirement 19, criterion 6).

    ``multiplicities(draws, n_matches, positions)`` turns a row into the match multiplicities of
    the contributions' match universe, ``numpy.bincount(positions[row], minlength=n_matches)``,
    where ``positions`` holds the universe index of each resampled match. A match drawn k times
    has multiplicity k, so it counts as k matches and k clusters (Requirement 19, criterion 2).
    Every row of multiplicities sums to N, and a match outside the resampled set always has
    multiplicity 0.

Fixed Role-level values (Requirement 19, criterion 2)
    Every resample holds the Role-level values of the full-data ranking it resamples at that
    ranking's values: the Main_Ranking here, and the Activity_Comparison_Ranking in its own
    bootstrap. These values are, as a ``rating.RoleHyper``:

    * Replacement_Level;
    * the four tau^2: Usual_Baseline, Pressure_Gap, Uplift and the fallback Clutch_Level;
    * both Split_Half_Reliability values;
    * the Blend_Weights.

    A resample recomputes, under its multiplicities, every player's estimates and match-clustered
    SEs, each player's Shrinkage_Factors (the fixed tau^2 with the player's SE in the resample),
    the shrunk values, Final_Score and the ranking with its tie-break. Each resample is one call of
    ``rank_role(data, mult, fixed=hyper, rankable_only=True)``. The intervals and P_Top10 values
    therefore describe the submitted estimator conditional on these Role-level values. They do
    not include the uncertainty in those values.

    The reason: in a resample each player's estimate scatters around his full-data estimate by
    about his SE, while the resample SEs stay about the same. A tau^2 re-estimated by the method of
    moments would therefore exceed the full-data tau^2 by about the mean squared SE. The resamples
    would then shrink less than the submitted estimator does.

    With the Role-level values fixed, a player's Final_Score in a resample depends only on his own
    rows. An unranked player therefore changes no other player's Final_Score, only the ranks below
    his. The odd and even halves are not needed, because both Split_Half_Reliability values are
    fixed. The Eligible_Player sets, NBV, Pressure_Ball flags, Recency_Weights and Stakes_Weights
    are the contributions' own, so they are fixed as well (Requirement 19, criterion 3).

Unranked players (Requirement 19, criterion 4)
    An Eligible_Player with fewer than RANKABLE_MIN_PRESSURE_MATCHES (2) Pressure_Matches, or fewer
    than RANKABLE_MIN_ORDINARY_BALLS (1) Ordinary_Ball, under a resample's multiplicities is
    unranked in that resample: his rank and Final_Score are NaN. He counts as outside the top 10
    for P_Top10, and the resample is left out of his interval and median rank.

Resamples (``bootstrap``, ``RoleBootstrap``)
    ``bootstrap(data, fixed, draws, positions, replacement_pct=D12)`` loops over the resamples. For
    each one it computes one multiplicity vector and ranks every Role of ``data`` with it, so both
    Roles use the same Bootstrap_Resample (Requirement 19, criterion 1). It returns one
    RoleBootstrap per Role: the (B, n) Final_Scores and ranks of the Role's n Eligible_Players
    (ascending ``player_id``; NaN where unranked) and the Role's warnings counted by kind.
    ``rank_role`` logs nothing, so ``bootstrap`` keeps, for each kind and estimate, the number of
    resamples that raised it, the number of warnings and the first message. On the delivered data,
    a resample takes about 2.5-3 ms for both Roles, and 2,000 resamples take about 5-6 seconds.

Summary (``bootstrap_table``, ``same_top10``)
    ``bootstrap_table(result, main)`` gives one row per Eligible_Player (BOOTSTRAP_COLUMNS,
    Requirement 19, criterion 5). Each row has the player's Main_Ranking rank and Final_Score
    (``main_rank``, ``main_final_score``) and the following resample values:

    * ``ci_low`` and ``ci_high``: the 2.5th and 97.5th percentiles (CI_PERCENTILES) of Final_Score
      over the resamples in which the player is ranked, by numpy's "linear" method;
    * ``median_rank``: the median rank over the same resamples (rank 1 is the highest Final_Score;
      a half-rank when the count is even);
    * ``p_top10``: the share of all B resamples with rank at most LIST_SIZE (10);
    * ``n_ranked`` and ``n_unranked``: the resamples in which the player is ranked and unranked;
      they sum to B.

    ``ci_low``, ``ci_high`` and ``median_rank`` are NaN for a player who is never ranked. When
    every resample ranks at least 10 players, the P_Top10 values of a Role sum to 10.
    ``same_top10`` flags the resamples whose top-10 set is the Main_Ranking's.

The ``bootstrap`` Pipeline_Stage (``run_bootstrap_stage``)
    Reads the ``rating`` checkpoint (``contributions``, ``match_universe``, ``player_estimates``,
    ``role_summary``) and the ``load`` table ``ball_by_ball`` (provisional names only). Steps:

    1. Rebuild the inputs: ``Contributions.from_frame(contributions, match_universe.match_id)``,
       then ``prepare_role`` on each Role's ``eligible`` players.
    2. Check that the inputs still give the Main_Ranking; any failure raises StageError:

       * ``rank_role`` with its defaults must reproduce the saved Final_Scores and ranks bit for
         bit;
       * ``RoleHyper.from_summary(role_summary)`` must equal ``RoleHyper.from_ranking`` of that
         ranking;
       * one copy of every resampled match, ranked with the fixed values and ``rankable_only``,
         must reproduce the Main_Ranking bit for bit. This also confirms that every rated match
         can be drawn.

    3. ``draw_resamples`` (D16, ``seeds.bootstrap``), then ``bootstrap`` with the Main_Ranking's
       RoleHyper of each Role, and ``bootstrap_table``.
    4. Checkpoint:

       * ``resample_draws``: the (B, N) int16 draws, which ``activity_comparison`` and the Stage-2
         trials reuse (Requirement 25, criterion 12);
       * ``resample_matches``: the int64 ``match_id`` of each of the N positions;
       * ``final_score_<role>`` and ``rank_<role>``: the (B, n) float64 arrays, columns in
         ascending ``player_id``;
       * the ``bootstrap`` table (BOOTSTRAP_COLUMNS, by Role and then ascending ``player_id``,
         the order of the array columns);
       * ``bootstrap_warnings`` (BOOTSTRAP_WARNING_COLUMNS).

    5. Files in ``outputs/robustness/``: ``bootstrap_batter.csv`` and ``bootstrap_bowler.csv``
       (BOOTSTRAP_FILE_COLUMNS, in Main_Ranking order, with the provisional ``player_name`` of the
       rating stage's files), and ``bootstrap_warnings.csv``. Floats have 6 decimals and NaN is
       blank.
    6. Data-derived constants: ``bootstrap.n_resampled_matches`` (N) and
       ``bootstrap.<quantity>.<role>`` for BOOTSTRAP_CONSTANTS:

       * ``top10_set_reproduced``: the resamples whose top-10 set is the Main_Ranking's;
       * ``min_ranked``: the fewest players ranked in a resample;
       * ``n_ever_unranked``: the Eligible_Players unranked in at least one resample;
       * ``min_p_top10_main_top10``: the lowest P_Top10 in the Main_Ranking's top 10;
       * ``max_p_top10_outside_top10``: the highest P_Top10 outside it (None when no player is
         outside it).

       The summary, the main top 10 with their intervals and each warning kind go to the Run_Log.

Cross-checks (Requirement 22, ``cross_checks``, ``cross_check_results``)
    Five Cross_Check_Rankings (glossary) compare the Main_Ranking with simpler methods, each
    ranking the Main_Ranking's Eligible_Players of a Role so that the common set is every
    Eligible_Player. The WPA ranking (criterion 1) ranks a Role by descending unweighted WPA summed
    over the player's Pressure_Balls: each batter-Role Pressure_Ball adds the signed Actual_WP_Swing
    (WP after - WP before) to the striker, each bowler-Role Pressure_Ball adds the negated swing
    (WP before - WP after) to the bowler, both from the Out_Of_Fold WP of the ``validation``
    per-Delivery frame (``wp_before`` is the ``model_choice`` Primary_Model's pre-ball WP, the same
    value ``BaseArtifacts`` caches as ``wp_pre``; ``wp_after`` is that Primary_Model's realized
    after-ball WP under the Pressure_Validator's WP-after rules, the glossary's Actual_WP_Swing).
    The four simple rankings (criteria 3 and 4) use only each player's Deliveries in the Role that
    are Simple_Rule_Balls, with the Simple_Rule flags the Pressure_Validator uses for V2 and V4
    (the frame's ``overs_16_20`` and ``chase_rrr_above_10``), and no Recency_Weight or
    Stakes_Weight: batter strike rate (100 x bat runs / Batter_Deliveries, descending) and batting
    average (bat runs / max(1, striker Batter_Dismissals), descending); bowler economy
    (6 x Bowler_Runs_Conceded / Legal_Balls, ascending) and bowling strike rate (Legal_Balls /
    Bowler_Wickets, ascending, the Bowler_Wickets from the ``reconstruct`` balls). A player with no
    qualifying Delivery or a zero denominator is placed below every defined value, those players by
    lower ``player_id`` (criterion 5); all remaining ties break by lower ``player_id``. Each ranking
    goes through ``ranks.compare_rankings`` (``on="rank"``) against the Main_Ranking for the top-10
    overlap and the entering and leaving names (criterion 6); the WPA ranking also reports the
    position Spearman of criterion 2, and the four simple rankings leave the Spearman blank.
    ``cross_checks`` returns the Cross_Check_Ranking rows of the robustness table (kind
    ``cases.CROSS_CHECK``); ``cross_check_results`` also returns the per-player detail and the
    undefined counts.

The ``robustness`` Pipeline_Stage (``run_stage``)
    Reads the completed checkpoints of a Run. Steps: (1) ``cases.BaseArtifacts.build`` rebuilds the
    Main_Ranking's arrays and checks them bit for bit against the ``rating`` checkpoint; (2) the
    Robustness_Cases of Requirement 21 (``cases.all_cases``, ``cases.run_all_cases``,
    ``cases.robustness_table``), with the List_Builder's name rule for the entering and leaving
    players; (3) ``repeatability`` from the ``rating`` and ``validation`` checkpoints (D19 from
    Settings); (4) ``cross_check_results`` from the ``validation`` frame and the ``reconstruct``
    balls. It writes one ``outputs/robustness/robustness_table.csv`` (the ``all_cases`` rows then
    the Cross_Check_Ranking rows, 10-decimal floats) and the six repeatability CSVs; checkpoints
    the combined ``robustness_table``, the six repeatability tables and the per-player
    ``cross_check_rankings`` detail for the Report_Builder; records the undefined-player counts as
    data-derived Method_Constants; and logs the Requirement 20 reliabilities and criterion-4
    warnings, each case (with the not-comparable warnings), and the cross-check overlaps, Spearman
    and names. It reads only its declared ``reads`` and writes only under ``outputs/robustness/``.

Errors: ValueError or TypeError, naming the problem, for draws, positions, multiplicities, Role
inputs or tables that do not fit together, or a value of the wrong type. ``run_bootstrap_stage``
and ``run_stage`` raise StageError when the ``rating`` checkpoint no longer reproduces its
Main_Ranking, or a checkpoint file is missing or changed.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from superr import cases
from superr.errors import StageError
from superr.ranks import RankingComparison, compare_rankings
from superr.rating import (
    CONTRIBUTIONS_TABLE,
    LIST_SIZE,
    MATCH_UNIVERSE_TABLE,
    PLAYER_ESTIMATES_TABLE,
    RANKABLE_MIN_ORDINARY_BALLS,
    RANKABLE_MIN_PRESSURE_MATCHES,
    ROLE_HYPER_FIELDS,
    ROLE_SUMMARY_TABLE,
    WARNING_KINDS,
    Aggregator,
    Contributions,
    Period,
    RatingWarning,
    RoleData,
    RoleHyper,
    RoleRanking,
    prepare_role,
    rank_role,
)
from superr.rating import STAGE_NAME as RATING_STAGE
from superr.reconstruct import BALLS_TABLE as RECONSTRUCT_BALLS_TABLE
from superr.reconstruct import COMP_IPL, COMP_T20I, DeliveryClass
from superr.reconstruct import STAGE_NAME as RECONSTRUCT_STAGE
from superr.validation import FRAME_TABLE as VALIDATION_FRAME_TABLE
from superr.validation import RULE_CHASE_RATE, RULE_DEATH_OVERS
from superr.validation import STAGE_NAME as VALIDATION_STAGE
from superr.validation import V1_TABLE as VALIDATION_V1_TABLE
from superr.value import BATTER_ROLE, BOWLER_ROLE, ROLES

if TYPE_CHECKING:
    from superr.pipeline import StageContext

__all__ = [
    "BOOTSTRAP_COLUMNS",
    "BOOTSTRAP_CONSTANTS",
    "BOOTSTRAP_FILES",
    "BOOTSTRAP_FILE_COLUMNS",
    "BOOTSTRAP_STAGE",
    "BOOTSTRAP_TABLE",
    "BOOTSTRAP_WARNINGS_FILE",
    "BOOTSTRAP_WARNINGS_TABLE",
    "BOOTSTRAP_WARNING_COLUMNS",
    "CI_PERCENTILES",
    "COMPETITION_SPEARMAN_COLUMNS",
    "COMPETITION_SPEARMAN_FILE",
    "CROSS_CHECK_COLUMNS",
    "CROSS_CHECK_RANKINGS",
    "CROSS_CHECK_TABLE",
    "FINAL_SCORE_ARRAYS",
    "MIN_PERIOD_PLAYERS",
    "PERIOD_HOLDOUT_COLUMNS",
    "PERIOD_HOLDOUT_FILE",
    "PERIOD_SPEARMAN_COLUMNS",
    "PERIOD_SPEARMAN_FILE",
    "RANK_ARRAYS",
    "REPEATABILITY_ESTIMATES",
    "REPEATABILITY_FILES",
    "REPEATABILITY_FLOAT_FORMAT",
    "REPEATABILITY_TABLES",
    "RESAMPLE_DRAWS",
    "RESAMPLE_MATCHES",
    "ROBUSTNESS_FOLDER",
    "ROBUSTNESS_STAGE",
    "ROBUSTNESS_TABLE_NAME",
    "CrossCheckRanking",
    "CrossCheckResult",
    "RepeatabilityResult",
    "RoleBootstrap",
    "SPLIT_HALF_RELIABILITY_COLUMNS",
    "SPLIT_HALF_RELIABILITY_FILE",
    "V1_IPL_FILE",
    "V1_T20I_FILE",
    "V1_TABLE_COLUMNS",
    "bootstrap",
    "bootstrap_table",
    "cross_check_results",
    "cross_checks",
    "draw_resamples",
    "multiplicities",
    "repeatability",
    "run_bootstrap_stage",
    "run_stage",
    "same_top10",
    "write_repeatability",
]

# ================================================================================ constants

#: Name of the Pipeline_Stage that runs the bootstrap, and of its checkpoint.
BOOTSTRAP_STAGE = "bootstrap"

#: Checkpoint arrays and tables of the stage (module docstring).
RESAMPLE_DRAWS = "resample_draws"
RESAMPLE_MATCHES = "resample_matches"
BOOTSTRAP_TABLE = "bootstrap"
BOOTSTRAP_WARNINGS_TABLE = "bootstrap_warnings"
FINAL_SCORE_ARRAYS: Mapping[str, str] = {role: f"final_score_{role}" for role in ROLES}
RANK_ARRAYS: Mapping[str, str] = {role: f"rank_{role}" for role in ROLES}

#: Folder under the outputs folder that holds the Robustness_Suite's files, and the bootstrap files.
ROBUSTNESS_FOLDER = "robustness"
BOOTSTRAP_FILES: Mapping[str, str] = {BATTER_ROLE: "bootstrap_batter.csv", BOWLER_ROLE: "bootstrap_bowler.csv"}
BOOTSTRAP_WARNINGS_FILE = "bootstrap_warnings.csv"

#: The percentiles of the 95% percentile interval of Final_Score (Requirement 19, criterion 5).
CI_PERCENTILES: tuple[float, float] = (2.5, 97.5)

#: Columns of the ``bootstrap`` table: one row per Eligible_Player of a Role (module docstring).
BOOTSTRAP_COLUMNS: tuple[str, ...] = (
    "role", "player_id", "main_rank", "main_final_score", "ci_low", "ci_high", "median_rank", "p_top10", "n_ranked",
    "n_unranked",
)  # fmt: skip

#: Columns of each ``bootstrap_<role>.csv`` file, in Main_Ranking order.
BOOTSTRAP_FILE_COLUMNS: tuple[str, ...] = (
    "main_rank", "player_id", "player_name", "main_final_score", "ci_low", "ci_high", "median_rank", "p_top10",
    "n_ranked", "n_unranked",
)  # fmt: skip

#: Columns of the ``bootstrap_warnings`` table and file: the rank_role warnings of the resamples,
#: counted per Role, kind and estimate. ``n_resamples``: the resamples that raised it;
#: ``n_warnings``: the warnings (one resample can raise several); ``first_resample`` (0-based)
#: and ``first_message``: the first one.
BOOTSTRAP_WARNING_COLUMNS: tuple[str, ...] = (
    "role", "kind", "estimate", "n_resamples", "n_warnings", "first_resample", "first_message",
)  # fmt: skip

#: Data-derived constants of the stage, one per Role: ``bootstrap.<quantity>.<role>``.
BOOTSTRAP_CONSTANTS: tuple[str, ...] = (
    "top10_set_reproduced", "min_ranked", "n_ever_unranked", "min_p_top10_main_top10", "max_p_top10_outside_top10",
)  # fmt: skip

#: Draw positions are stored as int16, so N may not exceed 32,768 matches.
_MAX_RESAMPLED = int(np.iinfo(np.int16).max) + 1
_COMPONENT = "Robustness_Suite"
_D12 = "design_choices.D12_replacement_pct"
_D16 = "design_choices.D16_n_boot"
_SEED = "seeds.bootstrap"
_LOAD_STAGE = "load"
_NAME_COLUMNS: Mapping[str, tuple[str, str]] = {
    BATTER_ROLE: ("striker_id", "striker_name"), BOWLER_ROLE: ("bowler_id", "bowler_name"),
}  # fmt: skip
_FLOAT_FORMAT = "%.6f"
_WARNING_ORDER: Mapping[str, int] = {kind: order for order, kind in enumerate(WARNING_KINDS)}


# ============================================================================= validation

def _whole_number(value: Any, what: str, *, least: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{what} must be a whole number; got {value!r}")
    number = int(value)
    if number < least:
        raise ValueError(f"{what} must be at least {least}; got {number:,}")
    return number


def _integers(values: Any, what: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind not in "iu":
        raise TypeError(f"{what} must hold integers; got dtype {array.dtype}")
    return array


def _positions(positions: Any, n_matches: int) -> np.ndarray:
    """The universe index of each resampled match (None: every match), as int64, checked."""
    if positions is None:
        return np.arange(n_matches, dtype=np.int64)
    pos = _integers(positions, "positions")
    if pos.ndim != 1 or not len(pos):
        raise ValueError(f"positions must be a non-empty 1-D array; got shape {pos.shape}")
    if pos.min() < 0 or pos.max() >= n_matches:
        raise ValueError(f"positions must lie in [0, {n_matches:,}), the match universe of the contributions")
    if len(np.unique(pos)) != len(pos):
        raise ValueError("positions must hold each match once")
    return pos.astype(np.int64)


def _draw_rows(draws: Any, n_resampled: int) -> np.ndarray:
    """``draws`` as a checked 1-D or 2-D integer array of positions 0..n_resampled-1."""
    rows = _integers(draws, "draws")
    if rows.ndim not in (1, 2) or rows.shape[-1] != n_resampled or (rows.ndim == 2 and not len(rows)):
        raise ValueError(
            f"draws must hold rows of {n_resampled:,} drawn positions, one per resampled match; got shape {rows.shape}"
        )
    if rows.min() < 0 or rows.max() >= n_resampled:
        raise ValueError(f"draws must hold positions in [0, {n_resampled:,})")
    return rows


# ================================================================================== draws

def draw_resamples(n_matches: int, n_boot: int, seed: int) -> np.ndarray:
    """The (n_boot, n_matches) int16 draw positions of every Bootstrap_Resample (Requirement 19, criterion 1).

    ``n_matches``: N, the matches the resamples draw from (1 to 32,768); ``n_boot``: B (D16);
    ``seed``: ``seeds.bootstrap``, 0 or more. Row b holds resample b's N positions, each uniform
    on 0..N-1 and drawn with replacement, in draw order, from
    ``numpy.random.Generator(PCG64(seed))``. The same arguments always give the same array.
    Raises ValueError or TypeError for an argument that is not a whole number in range.
    """
    n = _whole_number(n_matches, "n_matches", least=1)
    if n > _MAX_RESAMPLED:
        raise ValueError(f"n_matches must be at most {_MAX_RESAMPLED:,}, so that every position fits int16; got {n:,}")
    b = _whole_number(n_boot, "n_boot (D16)", least=1)
    s = _whole_number(seed, "seed (seeds.bootstrap)", least=0)
    rng = np.random.Generator(np.random.PCG64(s))
    return rng.integers(0, n, size=(b, n), dtype=np.int64).astype(np.int16)


def multiplicities(draws: Any, n_matches: int, positions: Any = None) -> np.ndarray:
    """The match multiplicities of one row of draws (1-D) or of every row (2-D), as int64.

    ``draws``: draw positions as ``draw_resamples`` gives them, one row or several; ``n_matches``:
    the size of the contributions' match universe; ``positions``: the universe index of each of
    the N resampled matches (None: every match of the universe, so N = ``n_matches``). Returns
    ``numpy.bincount(positions[row], minlength=n_matches)`` for each row: shape (n_matches,) or
    (B, n_matches). Every row sums to N. Raises ValueError or TypeError for draws or positions
    that do not fit.
    """
    size = _whole_number(n_matches, "n_matches", least=1)
    pos = _positions(positions, size)
    rows = _draw_rows(draws, len(pos))
    drawn = pos[rows]
    if drawn.ndim == 1:
        return np.bincount(drawn, minlength=size)
    offset = np.arange(len(drawn), dtype=np.int64)[:, None] * size
    return np.bincount((drawn + offset).ravel(), minlength=len(drawn) * size).reshape(len(drawn), size)


# ============================================================================= resamples

@dataclass(frozen=True, eq=False)
class RoleBootstrap:
    """One Role's Bootstrap_Resample results (``bootstrap``; module docstring).

    ``player_id``: the Role's n Eligible_Players, ascending (int64). ``final_score`` and ``rank``:
    (B, n) float64, row b for resample b, NaN where the player is unranked in it; ranks are 1, 2,
    3... over the ranked players of the resample. ``warnings``: the Role's rows of the
    ``bootstrap_warnings`` table (BOOTSTRAP_WARNING_COLUMNS). Arrays are read-only.
    """

    role: str
    player_id: np.ndarray
    final_score: np.ndarray
    rank: np.ndarray
    warnings: pd.DataFrame

    def __post_init__(self) -> None:
        n = len(self.player_id)
        if self.final_score.ndim != 2 or self.final_score.shape[1] != n or self.rank.shape != self.final_score.shape:
            raise ValueError(f"final_score and rank must both be (B, {n:,}) arrays, one column per Eligible_Player")
        if not np.array_equal(np.isfinite(self.final_score), np.isfinite(self.rank)):
            raise ValueError("a player must have a Final_Score exactly in the resamples in which he is ranked")
        for array in (self.player_id, self.final_score, self.rank):
            array.setflags(write=False)

    def __len__(self) -> int:
        return len(self.player_id)

    def __repr__(self) -> str:
        return (
            f"RoleBootstrap({self.role}: {self.n_boot:,} resamples x {len(self):,} Eligible_Players; "
            f"{int(self.ranked.sum()):,} ranked places, {len(self.warnings)} warning kind(s))"
        )

    @property
    def n_boot(self) -> int:
        """B, the number of Bootstrap_Resamples."""
        return int(self.rank.shape[0])

    @property
    def ranked(self) -> np.ndarray:
        """(B, n) bool: the player is ranked in the resample."""
        return np.isfinite(self.rank)


class _Tally:
    """The count of one warning kind and estimate over the resamples of one Role."""

    __slots__ = ("last", "n_resamples", "n_warnings", "first", "message")

    def __init__(self, resample: int, message: str) -> None:
        self.last, self.n_resamples, self.n_warnings = -1, 0, 0
        self.first, self.message = resample, message

    def add(self, resample: int) -> None:
        if resample != self.last:
            self.last = resample
            self.n_resamples += 1
        self.n_warnings += 1


def _count_warnings(tallies: dict[tuple[str, str | None], _Tally], notes: Sequence[RatingWarning], b: int) -> None:
    for note in notes:
        key = (note.kind, note.estimate)
        tally = tallies.get(key)
        if tally is None:
            tally = tallies[key] = _Tally(b, note.message)
        tally.add(b)


def _warning_rows(role: str, tallies: Mapping[tuple[str, str | None], _Tally]) -> pd.DataFrame:
    keys = sorted(tallies, key=lambda key: (_WARNING_ORDER.get(key[0], len(_WARNING_ORDER)), key[0], key[1] or ""))
    return pd.DataFrame(
        {
            "role": pd.array([role] * len(keys), dtype="string"),
            "kind": pd.array([kind for kind, _ in keys], dtype="string"),
            "estimate": pd.array([estimate for _, estimate in keys], dtype="string"),
            "n_resamples": np.array([tallies[key].n_resamples for key in keys], dtype=np.int64),
            "n_warnings": np.array([tallies[key].n_warnings for key in keys], dtype=np.int64),
            "first_resample": np.array([tallies[key].first for key in keys], dtype=np.int64),
            "first_message": pd.array([tallies[key].message for key in keys], dtype="string"),
        },
        columns=list(BOOTSTRAP_WARNING_COLUMNS),
    )


def _role_inputs(data: Mapping[str, RoleData], fixed: Mapping[str, RoleHyper]) -> tuple[list[str], int]:
    """The Roles of ``data``, checked against ``fixed``, and their common match-universe size."""
    if not isinstance(data, Mapping) or not isinstance(fixed, Mapping):
        raise TypeError("data and fixed must be mappings from Role name to RoleData and to RoleHyper")
    roles = list(data)
    if not roles:
        raise ValueError("data must hold at least one Role")
    if set(fixed) != set(roles):
        raise ValueError(f"fixed must hold a RoleHyper for each Role of data ({', '.join(roles)}); got {sorted(fixed)}")
    sizes = set()
    for role in roles:
        if not isinstance(data[role], RoleData):
            raise TypeError(f"data[{role!r}] must be a RoleData (prepare_role); got {type(data[role]).__name__}")
        if data[role].role != role:
            raise ValueError(f"data[{role!r}] holds the {data[role].role!r} Role")
        if not isinstance(fixed[role], RoleHyper):
            raise TypeError(f"fixed[{role!r}] must be a RoleHyper; got {type(fixed[role]).__name__}")
        sizes.add(data[role].full.n_matches)
    if len(sizes) != 1:
        raise ValueError("every Role of data must share one match universe (the same contributions)")
    return roles, sizes.pop()


def bootstrap(
    data: Mapping[str, RoleData],
    fixed: Mapping[str, RoleHyper],
    draws: Any,
    positions: Any = None,
    *,
    replacement_pct: float,
) -> dict[str, RoleBootstrap]:
    """Rank every Role of ``data`` in every Bootstrap_Resample of ``draws`` (Requirement 19, criteria 1-4).

    ``data``: Role name -> RoleData (``prepare_role``) of the Role's fixed Eligible_Players, all
    over one match universe. ``fixed``: Role name -> RoleHyper, the Role-level values held in every
    resample, those of the full-data ranking being resampled (``RoleHyper.from_ranking`` or
    ``from_summary``). ``draws``: the (B, N) draw positions of ``draw_resamples``.
    ``positions``: the universe index of each of the N resampled matches (None: every match).
    ``replacement_pct``: D12, passed on to ``rank_role``.

    Each resample's multiplicities serve every Role (module docstring), with
    ``rank_role(..., fixed=fixed[role], rankable_only=True)``. Returns Role name -> RoleBootstrap.
    Raises ValueError or TypeError for inputs that do not fit together.
    """
    roles, size = _role_inputs(data, fixed)
    pos = _positions(positions, size)
    rows = _draw_rows(draws, len(pos))
    if rows.ndim != 2:
        raise ValueError(f"draws must be a (B, N) array of resamples; got shape {rows.shape}")
    n_boot = len(rows)
    scores = {role: np.full((n_boot, len(data[role].eligible)), np.nan) for role in roles}
    ranks = {role: np.full((n_boot, len(data[role].eligible)), np.nan) for role in roles}
    tallies: dict[str, dict[tuple[str, str | None], _Tally]] = {role: {} for role in roles}
    for b in range(n_boot):
        mult = np.bincount(pos[rows[b]], minlength=size)  # one vector for every Role (criterion 1)
        for role in roles:
            ranking = rank_role(data[role], mult, fixed=fixed[role], replacement_pct=replacement_pct, rankable_only=True)
            ranked = ranking.ranked
            scores[role][b] = np.where(ranked, ranking.final_score, np.nan)
            ranks[role][b] = ranking.rank
            _count_warnings(tallies[role], ranking.warnings, b)
    return {
        role: RoleBootstrap(
            role=role, player_id=np.array(data[role].eligible, dtype=np.int64), final_score=scores[role],
            rank=ranks[role], warnings=_warning_rows(role, tallies[role]),
        )
        for role in roles
    }  # fmt: skip


# =============================================================================== summary

def _check_main(result: RoleBootstrap, main: RoleRanking) -> None:
    if not isinstance(result, RoleBootstrap) or not isinstance(main, RoleRanking):
        raise TypeError("result must be a RoleBootstrap and main a RoleRanking")
    if main.role != result.role or not np.array_equal(main.player_id, result.player_id):
        raise ValueError(f"main must be the {result.role} ranking of the same Eligible_Players as the resamples")


def bootstrap_table(result: RoleBootstrap, main: RoleRanking, *, list_size: int = LIST_SIZE) -> pd.DataFrame:
    """One row per Eligible_Player with BOOTSTRAP_COLUMNS (Requirement 19, criterion 5; module docstring).

    ``result``: the Role's RoleBootstrap; ``main``: the full-data RoleRanking that the resamples
    resample, of the same players (its ranks and Final_Scores fill ``main_rank`` and
    ``main_final_score``). ``list_size``: the top-list size of P_Top10 (10). Rows follow
    ``player_id`` ascending. Raises ValueError or TypeError for inputs that do not match.
    """
    _check_main(result, main)
    top = _whole_number(list_size, "list_size", least=1)
    ranked = result.ranked
    n_ranked = ranked.sum(axis=0)
    low, high, median = (np.full(len(result), np.nan) for _ in range(3))
    for j in np.flatnonzero(n_ranked):
        keep = ranked[:, j]
        low[j], high[j] = np.percentile(result.final_score[keep, j], CI_PERCENTILES, method="linear")
        median[j] = np.median(result.rank[keep, j])
    p_top = np.count_nonzero(result.rank <= top, axis=0) / result.n_boot  # an unranked NaN is never <= top
    return pd.DataFrame(
        {
            "role": pd.array([result.role] * len(result), dtype="str"),
            "player_id": np.array(result.player_id, dtype=np.int64),
            "main_rank": pd.Series(np.asarray(main.rank, dtype=np.float64)).astype("Int64").array,
            "main_final_score": np.array(main.final_score, dtype=np.float64),
            "ci_low": low,
            "ci_high": high,
            "median_rank": median,
            "p_top10": p_top,
            "n_ranked": n_ranked.astype(np.int64),
            "n_unranked": (result.n_boot - n_ranked).astype(np.int64),
        },
        columns=list(BOOTSTRAP_COLUMNS),
    )


def same_top10(result: RoleBootstrap, main: RoleRanking, *, list_size: int = LIST_SIZE) -> np.ndarray:
    """(B,) bool: the resample's top-``list_size`` set is exactly the Main_Ranking's (order ignored)."""
    _check_main(result, main)
    top = _whole_number(list_size, "list_size", least=1)
    return ((result.rank <= top) == (main.rank <= top)).all(axis=1)


# ===================================================================== the ``bootstrap`` stage

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


def _shown_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return str(path)


def _provisional_names(deliveries: pd.DataFrame) -> dict[str, pd.Series]:
    """Per Role, the most frequent ``striker_name`` or ``bowler_name`` of each player id (ties: first by name).

    The same provisional names as the ``rating`` stage's files; the List_Builder sets the final ones.
    """
    names = {}
    for role, (id_column, name_column) in _NAME_COLUMNS.items():
        missing = [column for column in (id_column, name_column) if column not in deliveries.columns]
        if missing:
            raise ValueError(f"ball_by_ball lacks the column(s) {', '.join(missing)}")
        pairs = deliveries[[id_column, name_column]].dropna()
        counted = pairs.groupby([id_column, name_column]).size().reset_index(name="n")
        counted = counted.sort_values([id_column, "n", name_column], ascending=[True, False, True], kind="stable")
        first = counted.drop_duplicates(id_column)
        names[role] = pd.Series(
            first[name_column].astype(str).to_numpy(dtype=object), index=first[id_column].to_numpy(dtype=np.int64)
        )
    return names


def _file_table(table: pd.DataFrame, names: pd.Series) -> pd.DataFrame:
    """One Role's ``bootstrap_<role>.csv`` rows (BOOTSTRAP_FILE_COLUMNS), in Main_Ranking order."""
    out = table.drop(columns="role")
    looked_up = names.reindex(out["player_id"].to_numpy(dtype=np.int64)).to_numpy(dtype=object)
    out.insert(out.columns.get_loc("player_id") + 1, "player_name", pd.array(looked_up, dtype="string"))
    out = out.sort_values(["main_rank", "player_id"], na_position="last", kind="stable").reset_index(drop=True)
    return out.loc[:, list(BOOTSTRAP_FILE_COLUMNS)]


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), equal_nan=True))


def _check_saved(main: RoleRanking, rows: pd.DataFrame) -> None:
    """StageError unless ``main`` has the Final_Scores and ranks that the rating checkpoint saved."""
    saved = rows.set_index("player_id").reindex(main.player_id)
    for column, values in (("final_score", main.final_score), ("rank", main.rank)):
        stored = saved[column].to_numpy(dtype=np.float64, na_value=np.nan)
        if not _same(stored, values):
            raise StageError(
                f"the {main.role} Role's {column} in the rating checkpoint's player_estimates differs from rank_role "
                "on the same contributions and Eligible_Players; rerun the Pipeline from the rating stage"
            )


def _main_values(summary: pd.DataFrame, main: RoleRanking) -> RoleHyper:
    """The Main_Ranking's Role-level values from role_summary, checked against the recomputed ranking."""
    hyper = RoleHyper.from_summary(summary, main.role)
    if hyper != RoleHyper.from_ranking(main):
        raise StageError(
            f"the {main.role} Role's values in the rating checkpoint's role_summary ({hyper}) differ from those of "
            f"rank_role on the same inputs ({RoleHyper.from_ranking(main)}); rerun the Pipeline from the rating stage"
        )
    return hyper


def _check_once(data: RoleData, once: np.ndarray, hyper: RoleHyper, main: RoleRanking, pct: float) -> None:
    """StageError unless one copy of every resampled match reproduces the Main_Ranking (bit for bit)."""
    check = rank_role(data, once, fixed=hyper, replacement_pct=pct, rankable_only=True)
    if not (_same(check.final_score, main.final_score) and _same(check.rank, main.rank)):
        raise StageError(
            f"a resample that draws every resampled match once does not reproduce the {main.role} Main_Ranking; "
            "a rated match may lie outside the resampled matches, or the fixed Role-level values do not fit"
        )


def _role_constants(result: RoleBootstrap, table: pd.DataFrame, main: RoleRanking) -> dict[str, Any]:
    in_top = (main.rank <= LIST_SIZE)
    p_top = table["p_top10"].to_numpy(dtype=np.float64)
    return {
        "top10_set_reproduced": int(same_top10(result, main).sum()),
        "min_ranked": int(result.ranked.sum(axis=1).min()),
        "n_ever_unranked": int((table["n_unranked"].to_numpy() > 0).sum()),
        "min_p_top10_main_top10": float(p_top[in_top].min()) if in_top.any() else None,
        "max_p_top10_outside_top10": float(p_top[~in_top].max()) if (~in_top).any() else None,
    }


def run_bootstrap_stage(ctx: StageContext) -> None:
    """The ``bootstrap`` Pipeline_Stage (module docstring, "The ``bootstrap`` Pipeline_Stage").

    Raises StageError (exit code 5) when the ``rating`` checkpoint no longer reproduces its
    Main_Ranking, or a checkpoint file is missing or changed; ValueError or TypeError for tables
    that do not fit together or a Config value of the wrong type.
    """
    started = time.perf_counter()
    settings, store, log = ctx.settings, ctx.store, ctx.log
    n_boot, seed, pct = settings.get(_D16), settings.get(_SEED), settings.get(_D12)

    # 1. The Main_Ranking's inputs, rebuilt from the rating checkpoint.
    universe = store.load_table(RATING_STAGE, MATCH_UNIVERSE_TABLE)
    contrib = Contributions.from_frame(store.load_table(RATING_STAGE, CONTRIBUTIONS_TABLE), universe["match_id"])
    players = store.load_table(RATING_STAGE, PLAYER_ESTIMATES_TABLE)
    summary = store.load_table(RATING_STAGE, ROLE_SUMMARY_TABLE)
    positions = np.flatnonzero(universe["resampled"].to_numpy(dtype=bool))
    match_ids = universe["match_id"].to_numpy(dtype=np.int64)

    # 2. The checks: the inputs still give the saved Main_Ranking, and one copy of each match reproduces it.
    once = multiplicities(np.arange(len(positions)), contrib.n_matches, positions)
    data: dict[str, RoleData] = {}
    mains: dict[str, RoleRanking] = {}
    hypers: dict[str, RoleHyper] = {}
    for role in ROLES:
        rows = players.loc[(players["role"] == role).to_numpy(dtype=bool) & players["eligible"].to_numpy(dtype=bool)]
        data[role] = prepare_role(contrib, role, rows["player_id"])
        mains[role] = rank_role(data[role], replacement_pct=pct)
        _check_saved(mains[role], rows)
        hypers[role] = _main_values(summary, mains[role])
        _check_once(data[role], once, hypers[role], mains[role], pct)

    # 3. The draws and the resamples (Requirement 19, criteria 1-5).
    tick = time.perf_counter()
    draws = draw_resamples(len(positions), n_boot, seed)
    draw_seconds = time.perf_counter() - tick
    results = bootstrap(data, hypers, draws, positions, replacement_pct=pct)
    loop_seconds = time.perf_counter() - tick - draw_seconds
    tables = {role: bootstrap_table(results[role], mains[role]) for role in ROLES}
    warnings_table = pd.concat([results[role].warnings for role in ROLES], ignore_index=True)

    # 4. The checkpoint.
    store.save_array(BOOTSTRAP_STAGE, RESAMPLE_DRAWS, draws)
    store.save_array(BOOTSTRAP_STAGE, RESAMPLE_MATCHES, match_ids[positions])
    for role in ROLES:
        store.save_array(BOOTSTRAP_STAGE, FINAL_SCORE_ARRAYS[role], results[role].final_score)
        store.save_array(BOOTSTRAP_STAGE, RANK_ARRAYS[role], results[role].rank)
    store.save_table(BOOTSTRAP_STAGE, BOOTSTRAP_TABLE, pd.concat([tables[role] for role in ROLES], ignore_index=True))
    store.save_table(BOOTSTRAP_STAGE, BOOTSTRAP_WARNINGS_TABLE, warnings_table)

    # 5. The files in outputs/robustness/.
    names = _provisional_names(store.load_table(_LOAD_STAGE, "ball_by_ball"))
    files = {role: _file_table(tables[role], names[role]) for role in ROLES}
    folder = settings.path("outputs") / ROBUSTNESS_FOLDER
    written = [_write_csv(folder / BOOTSTRAP_FILES[role], files[role]) for role in ROLES]
    written.append(_write_csv(folder / BOOTSTRAP_WARNINGS_FILE, warnings_table))

    # 6. The data-derived constants.
    ctx.constants.record(f"{BOOTSTRAP_STAGE}.n_resampled_matches", int(len(positions)), _COMPONENT)
    constants = {role: _role_constants(results[role], tables[role], mains[role]) for role in ROLES}
    for role in ROLES:
        for quantity in BOOTSTRAP_CONSTANTS:
            ctx.constants.record(f"{BOOTSTRAP_STAGE}.{quantity}.{role}", constants[role][quantity], _COMPONENT)

    _log_stage(
        ctx, universe=universe, positions=positions, seed=seed, hypers=hypers, results=results, mains=mains,
        files=files, warnings_table=warnings_table, constants=constants, written=written,
        timings=(draw_seconds, loop_seconds, time.perf_counter() - started),
    )  # fmt: skip


def _log_stage(
    ctx: StageContext,
    *,
    universe: pd.DataFrame,
    positions: np.ndarray,
    seed: int,
    hypers: Mapping[str, RoleHyper],
    results: Mapping[str, RoleBootstrap],
    mains: Mapping[str, RoleRanking],
    files: Mapping[str, pd.DataFrame],
    warnings_table: pd.DataFrame,
    constants: Mapping[str, Mapping[str, Any]],
    written: Sequence[Path],
    timings: tuple[float, float, float],
) -> None:
    log = ctx.log
    n_boot = next(iter(results.values())).n_boot
    held = "; ".join(
        f"{role} " + ", ".join(f"{name} {getattr(hypers[role], name):.6g}" for name in ROLE_HYPER_FIELDS)
        for role in ROLES
    )
    log.info(
        f"Bootstrap inputs (Requirement 19): {n_boot:,} Bootstrap_Resamples (D16) of the {len(positions):,} of "
        f"{len(universe):,} matches with a Delivery outside Super_Over_Innings, drawn from seeds.bootstrap = {seed}; "
        + ", ".join(f"{role} {len(results[role]):,} Eligible_Players" for role in ROLES)
        + f". Role-level values held at the Main_Ranking's in every resample (criterion 2): {held}. A player with "
        f"fewer than {RANKABLE_MIN_PRESSURE_MATCHES} Pressure_Matches or fewer than {RANKABLE_MIN_ORDINARY_BALLS} "
        "Ordinary_Ball in a resample is unranked there (criterion 4). Check passed: one copy of every resampled "
        "match reproduces both Main_Rankings bit for bit"
    )
    for role in ROLES:
        result, main, table, values = results[role], mains[role], files[role], constants[role]
        per_resample = result.ranked.sum(axis=1)
        in_order = same_top10(result, main) & np.where(main.rank <= LIST_SIZE, result.rank == main.rank, True).all(
            axis=1
        )
        in_top = table["main_rank"].to_numpy(dtype=np.float64, na_value=np.nan) <= LIST_SIZE
        top = table.loc[in_top]
        others = table.loc[~in_top & (table["p_top10"].to_numpy(dtype=np.float64) > 0.0)]
        others = others.sort_values(["p_top10", "main_rank"], ascending=[False, True], kind="stable").head(5)
        log.info(
            f"Bootstrap, {role}: {int(per_resample.min()):,}-{int(per_resample.max()):,} of {len(result):,} "
            f"Eligible_Players ranked per resample; {values['n_ever_unranked']:,} unranked in at least one resample "
            f"(at most {int(table['n_unranked'].max()):,} times); the main top-10 set in "
            f"{values['top10_set_reproduced']:,} of {n_boot:,} resamples ({int(in_order.sum()):,} in the same order); "
            f"P_Top10 sum {table['p_top10'].sum():.6f}"
        )
        log.info(
            f"Bootstrap top {len(top)}, {role} (names provisional): "
            + "; ".join(
                f"{int(row['main_rank'])} {row['player_name']} ({int(row['player_id'])}) S {row['main_final_score']:+.3f} "
                f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}] median rank {row['median_rank']:g} "
                f"P_Top10 {row['p_top10']:.3f}"
                for row in top.to_dict("records")
            )
            + ". Highest P_Top10 outside it: "
            + ("; ".join(
                f"{int(row['main_rank'])} {row['player_name']} ({int(row['player_id'])}) P_Top10 {row['p_top10']:.3f}"
                for row in others.to_dict("records")
            ) or "none")  # fmt: skip
        )
    for row in warnings_table.to_dict("records"):
        estimate = f" ({row['estimate']})" if isinstance(row["estimate"], str) else ""
        log.warning(
            f"Bootstrap warnings, {row['role']} {row['kind']}{estimate}: in {int(row['n_resamples']):,} of {n_boot:,} "
            f"resamples ({int(row['n_warnings']):,} warnings); the first, in resample {int(row['first_resample'])}: "
            f"{row['first_message']}"
        )
    draw_seconds, loop_seconds, seconds = timings
    log.info(
        f"Bootstrap timing: draws {draw_seconds:.2f} s; resamples {loop_seconds:.1f} s "
        f"({loop_seconds / n_boot * 1000:.2f} ms per resample, both Roles); stage {seconds:.1f} s. Files: "
        + ", ".join(_shown_path(path, ctx.root) for path in written)
    )


# ============================================================ repeatability (Requirement 20)

#: Float format of the repeatability files: 10 decimals, the Main_List files' format
#: (``lists.LIST_FLOAT_FORMAT``), so small reliabilities and correlations keep their precision.
REPEATABILITY_FLOAT_FORMAT = "%.10f"

#: The Eligible_Players a Role needs who meet the Period_Minimum (D19) in both the Early_Period and
#: the Late_Period before its time-holdout correlations are computed (Requirement 20, criterion 4).
MIN_PERIOD_PLAYERS = LIST_SIZE  # 10

#: The fewest paired points a Spearman correlation needs; fewer, or a constant side, give NaN.
_MIN_SPEARMAN_POINTS = 2

#: The two estimates whose repeatability Requirement 20 reports, named as in the files.
CLUTCH_ESTIMATE = "clutch_level"
UPLIFT_ESTIMATE = "uplift"
REPEATABILITY_ESTIMATES: tuple[str, ...] = (CLUTCH_ESTIMATE, UPLIFT_ESTIMATE)

#: The PI decile count and the group labels of a V1 row, as the pooled V1 table (``validation``)
#: writes them: deciles 1-``_N_DECILES`` of the Live_Balls and a ``dead`` row of the Dead_Balls.
_N_DECILES = 10
_DECILE_GROUP = "decile"
_DEAD_GROUP = "dead"

#: Files written under ``outputs/robustness/`` (module docstring of ``repeatability``), in order.
SPLIT_HALF_RELIABILITY_FILE = "split_half_reliability.csv"
PERIOD_HOLDOUT_FILE = "period_holdout.csv"
PERIOD_SPEARMAN_FILE = "period_spearman.csv"
V1_IPL_FILE = "v1_ipl.csv"
V1_T20I_FILE = "v1_t20i.csv"
COMPETITION_SPEARMAN_FILE = "competition_spearman.csv"
REPEATABILITY_FILES: tuple[str, ...] = (
    SPLIT_HALF_RELIABILITY_FILE, PERIOD_HOLDOUT_FILE, PERIOD_SPEARMAN_FILE, V1_IPL_FILE, V1_T20I_FILE,
    COMPETITION_SPEARMAN_FILE,
)  # fmt: skip

#: Columns of each repeatability table.
SPLIT_HALF_RELIABILITY_COLUMNS: tuple[str, ...] = ("role", "estimate", "reliability", "n_players")
PERIOD_HOLDOUT_COLUMNS: tuple[str, ...] = (
    "role", "player_id", "n_pressure_balls_early", "n_pressure_balls_late", "clutch_early", "clutch_late",
    "uplift_early", "uplift_late",
)  # fmt: skip
PERIOD_SPEARMAN_COLUMNS: tuple[str, ...] = ("role", "estimate", "spearman", "n_players", "computed")
V1_TABLE_COLUMNS: tuple[str, ...] = (
    "role", "innings", "group", "decile", "n_deliveries", "pi_min", "pi_max", "mean_swing",
)  # fmt: skip
COMPETITION_SPEARMAN_COLUMNS: tuple[str, ...] = ("role", "innings", "spearman", "n_deciles")


@dataclass(frozen=True)
class RepeatabilityResult:
    """The six repeatability tables of Requirement 20 and the warnings to log (``repeatability``).

    * ``split_half_reliability`` (SPLIT_HALF_RELIABILITY_COLUMNS, criterion 1): per Role and
      estimate (Clutch_Level, Uplift), the Split_Half_Reliability as the Rating_Engine computes it
      in Requirement 18, criterion 8, before the zero floor of criterion 9, and the player count.
    * ``period_holdout`` (PERIOD_HOLDOUT_COLUMNS, criterion 2): one row per Eligible_Player who
      meets the Period_Minimum (D19) in both periods, with the unweighted Pressure_Ball counts and
      the unshrunk Early_Period and Late_Period Clutch_Level and Uplift.
    * ``period_spearman`` (PERIOD_SPEARMAN_COLUMNS, criteria 3 and 4): per Role and estimate, the
      Spearman correlation of the Early and Late values and the player count; ``spearman`` is blank
      and ``computed`` False when fewer than MIN_PERIOD_PLAYERS players qualify (criterion 4).
    * ``v1_ipl`` and ``v1_t20i`` (V1_TABLE_COLUMNS, criterion 5): the V1 table over the IPL matches
      and over the T20I matches, each with the mean Actual_WP_Swing and Delivery count of every PI
      decile and of the Dead_Balls, per Role and Innings_Group, using the pooled V1 table's decile
      boundaries.
    * ``competition_spearman`` (COMPETITION_SPEARMAN_COLUMNS, criterion 6): per Role and
      Innings_Group, the Spearman correlation of the IPL and T20I mean Actual_WP_Swing across the
      deciles that hold at least one Delivery in both tables, and that decile count.

    ``warnings``: the criterion-4 messages for the Run_Log (the Pipeline_Stage of task 25.4 logs
    them). ``files()`` maps each ``outputs/robustness/`` file name to its table, in write order.
    """

    split_half_reliability: pd.DataFrame
    period_holdout: pd.DataFrame
    period_spearman: pd.DataFrame
    v1_ipl: pd.DataFrame
    v1_t20i: pd.DataFrame
    competition_spearman: pd.DataFrame
    warnings: tuple[str, ...] = ()

    def files(self) -> dict[str, pd.DataFrame]:
        """Each ``outputs/robustness/`` file name mapped to its table, in REPEATABILITY_FILES order."""
        return {
            SPLIT_HALF_RELIABILITY_FILE: self.split_half_reliability,
            PERIOD_HOLDOUT_FILE: self.period_holdout,
            PERIOD_SPEARMAN_FILE: self.period_spearman,
            V1_IPL_FILE: self.v1_ipl,
            V1_T20I_FILE: self.v1_t20i,
            COMPETITION_SPEARMAN_FILE: self.competition_spearman,
        }


def _require_columns(frame: Any, columns: Sequence[str], what: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{what} must be a DataFrame; got {type(frame).__name__}")
    absent = [column for column in dict.fromkeys(columns) if column not in frame.columns]
    if absent:
        raise ValueError(f"{what} lacks the column(s) {', '.join(absent)}")


def _role_row(summary: pd.DataFrame, role: str) -> pd.Series:
    rows = summary.loc[summary["role"].astype("string").to_numpy() == role]
    if len(rows) != 1:
        raise ValueError(f"role_summary must hold exactly one {role!r} row; it holds {len(rows)}")
    return rows.iloc[0]


def _spearman(x: Any, y: Any, *, min_points: int = _MIN_SPEARMAN_POINTS) -> float:
    """Spearman's rank correlation of paired values: the Pearson correlation of their average ranks.

    Pairs with a non-finite value on either side are dropped. The result is NaN with fewer than
    ``min_points`` (at least 2) finite pairs or when either set of ranks is constant, and is
    clipped to [-1, 1]. The same rule as ``ranks._spearman`` (Requirement 22, criterion 2).
    """
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Spearman needs paired values of equal length; got {a.shape} and {b.shape}")
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if len(a) < max(2, int(min_points)):
        return math.nan
    ra = pd.Series(a).rank(method="average").to_numpy(dtype=np.float64)
    rb = pd.Series(b).rank(method="average").to_numpy(dtype=np.float64)
    da, db = ra - ra.mean(), rb - rb.mean()
    saa, sbb = float(da @ da), float(db @ db)
    if saa <= 0.0 or sbb <= 0.0:
        return math.nan
    return float(min(1.0, max(-1.0, float(da @ db) / math.sqrt(saa * sbb))))


def _split_half_table(role_summary: pd.DataFrame) -> pd.DataFrame:
    """Requirement 20, criterion 1: each Role's rho_L and rho_U before the zero floor, with the count."""
    _require_columns(role_summary, ("role", "rho_L_raw", "rho_U_raw", "n_rho"), "role_summary")
    roles, estimates, reliabilities, counts = [], [], [], []
    for role in ROLES:
        row = _role_row(role_summary, role)
        n = int(row["n_rho"])
        for estimate, column in ((CLUTCH_ESTIMATE, "rho_L_raw"), (UPLIFT_ESTIMATE, "rho_U_raw")):
            roles.append(role)
            estimates.append(estimate)
            reliabilities.append(float(row[column]))
            counts.append(n)
    return pd.DataFrame(
        {
            "role": pd.array(roles, dtype="str"),
            "estimate": pd.array(estimates, dtype="str"),
            "reliability": np.array(reliabilities, dtype=np.float64),
            "n_players": np.array(counts, dtype=np.int64),
        },
        columns=list(SPLIT_HALF_RELIABILITY_COLUMNS),
    )


def _eligible_ids(player_estimates: pd.DataFrame) -> dict[str, np.ndarray]:
    """Each Role's Eligible_Player ids (ascending int64) from the ``player_estimates`` ``eligible`` flag."""
    _require_columns(player_estimates, ("role", "player_id", "eligible"), "player_estimates")
    eligible = pd.Series(player_estimates["eligible"]).fillna(False).to_numpy(dtype=bool)
    role_of = player_estimates["role"].astype("string").to_numpy()
    player_id = player_estimates["player_id"].to_numpy(dtype=np.int64)
    return {role: np.unique(player_id[(role_of == role) & eligible]).astype(np.int64) for role in ROLES}


def _replacement_levels(role_summary: pd.DataFrame) -> dict[str, float]:
    """Each Role's full-data Replacement_Level from ``role_summary`` (Requirement 18; held fixed in 20.2)."""
    _require_columns(role_summary, ("role", "replacement_level"), "role_summary")
    return {role: float(_role_row(role_summary, role)["replacement_level"]) for role in ROLES}


def _period_tables(
    contrib: Contributions, eligible: Mapping[str, np.ndarray], levels: Mapping[str, float], period_minimum: int
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Requirement 20, criteria 2-4: the Early/Late unshrunk estimates, their Spearman and the warnings.

    For each Role the Early_Period and Late_Period rows are aggregated on their own
    (``Period.EARLY`` and ``Period.LATE`` of the contributions), so each period redoes the
    Usual_Baseline innings x Phase reweighting within the period, with the full-data
    Replacement_Level held fixed. A player meets the Period_Minimum when his unweighted
    Pressure_Ball count is at least ``period_minimum`` (D19) in each period.
    """
    hold: dict[str, list[Any]] = {column: [] for column in PERIOD_HOLDOUT_COLUMNS}
    spear: dict[str, list[Any]] = {column: [] for column in PERIOD_SPEARMAN_COLUMNS}
    warnings: list[str] = []
    for role in ROLES:
        ids = eligible[role]
        mine = contrib.for_role(role).take(np.isin(contrib.for_role(role).player, ids))
        level = levels[role]
        early = Aggregator(mine.take(mine.period == int(Period.EARLY)), role=role)
        late = Aggregator(mine.take(mine.period == int(Period.LATE)), role=role)
        early_est = early.estimates(subset=ids, replacement_level=level)
        late_est = late.estimates(subset=ids, replacement_level=level)
        both = (early_est.n_pressure_balls >= period_minimum) & (late_est.n_pressure_balls >= period_minimum)
        at = np.flatnonzero(both)
        for j in at:
            hold["role"].append(role)
            hold["player_id"].append(int(ids[j]))
            hold["n_pressure_balls_early"].append(int(early_est.n_pressure_balls[j]))
            hold["n_pressure_balls_late"].append(int(late_est.n_pressure_balls[j]))
            hold["clutch_early"].append(float(early_est.clutch[j]))
            hold["clutch_late"].append(float(late_est.clutch[j]))
            hold["uplift_early"].append(float(early_est.uplift[j]))
            hold["uplift_late"].append(float(late_est.uplift[j]))
        n = int(both.sum())
        computed = n >= MIN_PERIOD_PLAYERS
        if not computed:
            warnings.append(
                f"the {role} Role has {n} Eligible_Player(s) with at least the Period_Minimum (D19 = "
                f"{period_minimum}) unweighted Pressure_Balls in both the Early_Period and the Late_Period, fewer "
                f"than {MIN_PERIOD_PLAYERS}; its Early_Period-to-Late_Period correlations are not computed "
                "(Requirement 20, criterion 4)"
            )
        for estimate, early_v, late_v in (
            (CLUTCH_ESTIMATE, early_est.clutch, late_est.clutch),
            (UPLIFT_ESTIMATE, early_est.uplift, late_est.uplift),
        ):
            spear["role"].append(role)
            spear["estimate"].append(estimate)
            spear["spearman"].append(_spearman(early_v[at], late_v[at]) if computed else math.nan)
            spear["n_players"].append(n)
            spear["computed"].append(computed)
    holdout = pd.DataFrame(
        {
            "role": pd.array(hold["role"], dtype="str"),
            "player_id": np.array(hold["player_id"], dtype=np.int64),
            "n_pressure_balls_early": np.array(hold["n_pressure_balls_early"], dtype=np.int64),
            "n_pressure_balls_late": np.array(hold["n_pressure_balls_late"], dtype=np.int64),
            "clutch_early": np.array(hold["clutch_early"], dtype=np.float64),
            "clutch_late": np.array(hold["clutch_late"], dtype=np.float64),
            "uplift_early": np.array(hold["uplift_early"], dtype=np.float64),
            "uplift_late": np.array(hold["uplift_late"], dtype=np.float64),
        },
        columns=list(PERIOD_HOLDOUT_COLUMNS),
    )
    spearman = pd.DataFrame(
        {
            "role": pd.array(spear["role"], dtype="str"),
            "estimate": pd.array(spear["estimate"], dtype="str"),
            "spearman": np.array(spear["spearman"], dtype=np.float64),
            "n_players": np.array(spear["n_players"], dtype=np.int64),
            "computed": np.array(spear["computed"], dtype=bool),
        },
        columns=list(PERIOD_SPEARMAN_COLUMNS),
    )
    return holdout, spearman, warnings


def _decile_boundaries(pooled: pd.DataFrame, role: str, innings: int) -> np.ndarray:
    """The pooled V1 table's PI decile boundaries for one Role and Innings_Group: the pi_max of deciles 1-9.

    A Delivery is placed in the lowest decile whose pooled ``pi_max`` is at least its PI, so the IPL
    and T20I tables use the same boundaries as the pooled V1 table (Requirement 20, criterion 5).
    """
    mask = (
        (pooled["role"].astype("string").to_numpy() == role)
        & (pooled["innings"].to_numpy() == innings)
        & (pooled["group"].astype("string").to_numpy() == _DECILE_GROUP)
    )
    sub = pooled.loc[mask].sort_values("decile")
    pi_max = sub["pi_max"].to_numpy(dtype=np.float64)
    if len(pi_max) != _N_DECILES or not np.isfinite(pi_max).all():
        raise ValueError(
            f"the pooled V1 table needs {_N_DECILES} deciles with a finite pi_max for the {role} Role, innings "
            f"{innings}; got {len(pi_max)}"
        )
    return pi_max[: _N_DECILES - 1]  # the upper edge of deciles 1..9; decile 10 takes everything above


def _competition_v1_table(
    frame: pd.DataFrame, pooled: pd.DataFrame, comp_type: np.ndarray, competition: str
) -> pd.DataFrame:
    """Requirement 20, criterion 5: the V1 table of one Competition_Type on the pooled decile boundaries.

    ``frame``: the ``validation_deliveries`` table; ``pooled``: the pooled V1 table (its decile
    boundaries); ``comp_type``: the Competition_Type of each Delivery's match; ``competition``: the
    Competition_Type to keep. The batter Role covers the Batter_Deliveries (``pi_bat``), the bowler
    Role every Delivery (``pi_bowl``), as the pooled V1 table does.
    """
    _require_columns(frame, ("innings", "batter", "live", "pi_bat", "pi_bowl", "swing"), "validation_deliveries")
    _require_columns(pooled, ("role", "innings", "group", "decile", "pi_max"), "v1_deciles")
    keep = comp_type == competition
    innings = frame["innings"].to_numpy(dtype=np.int64)
    live = pd.Series(frame["live"]).fillna(False).to_numpy(dtype=bool)
    batter = pd.Series(frame["batter"]).fillna(False).to_numpy(dtype=bool)
    swing = frame["swing"].to_numpy(dtype=np.float64)
    pis = {BATTER_ROLE: frame["pi_bat"].to_numpy(dtype=np.float64), BOWLER_ROLE: frame["pi_bowl"].to_numpy(dtype=np.float64)}
    role_masks = {BATTER_ROLE: batter, BOWLER_ROLE: np.ones(len(frame), dtype=bool)}
    groups = sorted(int(value) for value in pd.unique(pooled["innings"]))

    columns: dict[str, list[Any]] = {column: [] for column in V1_TABLE_COLUMNS}

    def add(role: str, innings_group: int, group: str, decile: Any, rows: np.ndarray, pi: np.ndarray) -> None:
        columns["role"].append(role)
        columns["innings"].append(int(innings_group))
        columns["group"].append(group)
        columns["decile"].append(decile)
        columns["n_deliveries"].append(int(len(rows)))
        columns["pi_min"].append(float(pi[rows].min()) if len(rows) else math.nan)
        columns["pi_max"].append(float(pi[rows].max()) if len(rows) else math.nan)
        columns["mean_swing"].append(float(swing[rows].mean()) if len(rows) else math.nan)

    for role in ROLES:
        pi = pis[role]
        role_mask = role_masks[role]
        for group in groups:
            base = keep & role_mask & (innings == group)
            boundaries = _decile_boundaries(pooled, role, group)
            on = np.flatnonzero(base & live)
            decile = np.searchsorted(boundaries, pi[on], side="left") + 1  # 1..N_DECILES
            for number in range(1, _N_DECILES + 1):
                add(role, group, _DECILE_GROUP, number, on[decile == number], pi)
            add(role, group, _DEAD_GROUP, pd.NA, np.flatnonzero(base & ~live), pi)

    return pd.DataFrame(
        {
            "role": pd.array(columns["role"], dtype="str"),
            "innings": np.array(columns["innings"], dtype=np.int64),
            "group": pd.array(columns["group"], dtype="str"),
            "decile": pd.array(columns["decile"], dtype="Int64"),
            "n_deliveries": np.array(columns["n_deliveries"], dtype=np.int64),
            "pi_min": np.array(columns["pi_min"], dtype=np.float64),
            "pi_max": np.array(columns["pi_max"], dtype=np.float64),
            "mean_swing": np.array(columns["mean_swing"], dtype=np.float64),
        },
        columns=list(V1_TABLE_COLUMNS),
    )


def _decile_means(table: pd.DataFrame, role: str, innings: int) -> tuple[np.ndarray, np.ndarray]:
    """One V1 table's per-decile ``n_deliveries`` and ``mean_swing`` for a Role and Innings_Group, deciles 1-10."""
    mask = (
        (table["role"].astype("string").to_numpy() == role)
        & (table["innings"].to_numpy() == innings)
        & (table["group"].astype("string").to_numpy() == _DECILE_GROUP)
    )
    sub = table.loc[mask].sort_values("decile")
    return sub["n_deliveries"].to_numpy(dtype=np.int64), sub["mean_swing"].to_numpy(dtype=np.float64)


def _competition_spearman_table(v1_ipl: pd.DataFrame, v1_t20i: pd.DataFrame) -> pd.DataFrame:
    """Requirement 20, criterion 6: the IPL-vs-T20I Spearman of the mean Actual_WP_Swing across PI deciles.

    For each Role and Innings_Group the correlation uses only the deciles with at least one
    Delivery in both the IPL and the T20I table.
    """
    groups = sorted(int(value) for value in pd.unique(v1_ipl["innings"]))
    roles, innings_list, spearmans, counts = [], [], [], []
    for role in ROLES:
        for group in groups:
            n_ipl, mean_ipl = _decile_means(v1_ipl, role, group)
            n_t20i, mean_t20i = _decile_means(v1_t20i, role, group)
            common = (n_ipl >= 1) & (n_t20i >= 1)
            roles.append(role)
            innings_list.append(int(group))
            spearmans.append(_spearman(mean_ipl[common], mean_t20i[common]))
            counts.append(int(common.sum()))
    return pd.DataFrame(
        {
            "role": pd.array(roles, dtype="str"),
            "innings": np.array(innings_list, dtype=np.int64),
            "spearman": np.array(spearmans, dtype=np.float64),
            "n_deciles": np.array(counts, dtype=np.int64),
        },
        columns=list(COMPETITION_SPEARMAN_COLUMNS),
    )


def repeatability(
    *,
    contributions: pd.DataFrame,
    match_universe: pd.DataFrame,
    player_estimates: pd.DataFrame,
    role_summary: pd.DataFrame,
    v1_deciles: pd.DataFrame,
    validation_deliveries: pd.DataFrame,
    period_minimum: int,
) -> RepeatabilityResult:
    """The repeatability checks of Requirement 20, as six tables (RepeatabilityResult).

    A pure function: it reads only its arguments and writes nothing. The ``robustness``
    Pipeline_Stage (task 25.4) supplies them from the checkpoints and writes the result with
    ``write_repeatability`` to ``outputs/robustness/``:

    * ``contributions``: the ``rating`` checkpoint's ``contributions`` table (``CONTRIBUTIONS_TABLE``);
    * ``match_universe``: its ``match_universe`` table (``MATCH_UNIVERSE_TABLE``), for the
      contributions' match universe and each match's Competition_Type (``comp_type``);
    * ``player_estimates``: its ``player_estimates`` table (``PLAYER_ESTIMATES_TABLE``), for the
      Eligible_Player set of each Role (the ``eligible`` flag);
    * ``role_summary``: its ``role_summary`` table (``ROLE_SUMMARY_TABLE``), for ``rho_L_raw``,
      ``rho_U_raw`` and ``n_rho`` (criterion 1) and each Role's full-data ``replacement_level``;
    * ``v1_deciles``: the ``validation`` checkpoint's pooled V1 table (``validation.V1_TABLE``),
      for the PI decile boundaries of criterion 5 (Requirement 14, criterion 2);
    * ``validation_deliveries``: its per-Delivery frame (``validation.FRAME_TABLE``), for the
      Delivery-level PI, Actual_WP_Swing, Live_Ball and Batter_Delivery flags of criteria 5 and 6;
    * ``period_minimum``: the Period_Minimum, Config D19 (``design_choices.D19_period_minimum``).

    Raises ValueError or TypeError for a missing column, a Role without exactly one summary row, a
    pooled V1 table without its 10 deciles, or an argument of the wrong type.
    """
    minimum = _whole_number(period_minimum, "period_minimum (D19)", least=1)
    split = _split_half_table(role_summary)
    contrib = Contributions.from_frame(contributions, match_universe["match_id"])
    holdout, spearman, warnings = _period_tables(
        contrib, _eligible_ids(player_estimates), _replacement_levels(role_summary), minimum
    )
    _require_columns(match_universe, ("match_id", "comp_type"), "match_universe")
    _require_columns(validation_deliveries, ("match_id",), "validation_deliveries")
    comp_of = pd.Series(
        match_universe["comp_type"].astype("string").to_numpy(dtype=object),
        index=pd.Index(match_universe["match_id"].to_numpy(dtype=np.int64)),
    )
    comp_type = comp_of.reindex(validation_deliveries["match_id"].to_numpy(dtype=np.int64)).to_numpy(dtype=object)
    v1_ipl = _competition_v1_table(validation_deliveries, v1_deciles, comp_type, COMP_IPL)
    v1_t20i = _competition_v1_table(validation_deliveries, v1_deciles, comp_type, COMP_T20I)
    return RepeatabilityResult(
        split_half_reliability=split,
        period_holdout=holdout,
        period_spearman=spearman,
        v1_ipl=v1_ipl,
        v1_t20i=v1_t20i,
        competition_spearman=_competition_spearman_table(v1_ipl, v1_t20i),
        warnings=tuple(warnings),
    )


def write_repeatability(result: RepeatabilityResult, out_dir: Path | str) -> list[Path]:
    """Write the six repeatability tables to ``out_dir`` (normally ``outputs/robustness/``); return the paths.

    Each file is UTF-8 with ``\\n`` line ends, floats in REPEATABILITY_FLOAT_FORMAT (10 decimals,
    the Main_List files' format) and NaN blank, written through ``<file>.tmp`` renamed in place.
    The ``robustness`` Pipeline_Stage (task 25.4) calls it with ``settings.path("outputs") /
    ROBUSTNESS_FOLDER``.
    """
    if not isinstance(result, RepeatabilityResult):
        raise TypeError(f"result must be a RepeatabilityResult; got {type(result).__name__}")
    folder = Path(out_dir)
    return [_write_repeatability_csv(folder / name, table) for name, table in result.files().items()]


def _write_repeatability_csv(path: Path, table: pd.DataFrame) -> Path:
    """Write one repeatability table to ``path`` (REPEATABILITY_FLOAT_FORMAT, NaN blank) through ``<file>.tmp``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        table.to_csv(
            temporary, index=False, encoding="utf-8", lineterminator="\n", float_format=REPEATABILITY_FLOAT_FORMAT
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


# ============================================================ cross-checks (Requirement 22)

#: Name of the ``robustness`` Pipeline_Stage and of its checkpoint (design stage 16, folder
#: ``16_robustness``). Its deliverables share ``outputs/robustness/`` with the bootstrap files.
ROBUSTNESS_STAGE = "robustness"

#: The combined robustness table checkpoint: the ``all_cases`` rows (Requirement 21) first, then
#: the Cross_Check_Ranking rows (Requirement 22), in ``cases.ROBUSTNESS_COLUMNS`` order. The stage
#: writes it as ``outputs/robustness/robustness_table.csv`` (``cases.ROBUSTNESS_FILE``).
ROBUSTNESS_TABLE_NAME = "robustness_table"

#: The per-player Cross_Check_Ranking detail checkpoint: one row per Role, Cross_Check_Ranking and
#: Eligible_Player, for the Report_Builder (task 27.1). ``value`` is the ranking value (NaN when
#: undefined) and ``defined`` whether the player has one (Requirement 22, criterion 5).
CROSS_CHECK_TABLE = "cross_check_rankings"
CROSS_CHECK_COLUMNS: tuple[str, ...] = ("role", "ranking", "player_id", "rank", "value", "defined")

#: Each repeatability file name (``REPEATABILITY_FILES``) mapped to its checkpoint table name.
REPEATABILITY_TABLES: Mapping[str, str] = {
    SPLIT_HALF_RELIABILITY_FILE: "split_half_reliability",
    PERIOD_HOLDOUT_FILE: "period_holdout",
    PERIOD_SPEARMAN_FILE: "period_spearman",
    V1_IPL_FILE: "v1_ipl",
    V1_T20I_FILE: "v1_t20i",
    COMPETITION_SPEARMAN_FILE: "competition_spearman",
}

#: The Period_Minimum Config path (D19), passed to ``repeatability``.
_D19 = "design_choices.D19_period_minimum"
_LEGAL_BALL = int(DeliveryClass.LEGAL)

#: The five Cross_Check_Rankings (glossary): the WPA ranking (Requirement 22, criterion 1), which
#: ranks both Roles; the strike-rate and batting-average rankings of batters (criterion 3); and the
#: economy and bowling-strike-rate rankings of bowlers (criterion 4). The label is the robustness
#: table's ``case``; the ``target_id`` names the criterion.
WPA_RANKING = "WPA"
STRIKE_RATE_RANKING = "strike rate"
AVERAGE_RANKING = "batting average"
ECONOMY_RANKING = "economy"
BOWLING_STRIKE_RANKING = "bowling strike rate"
_WPA_TARGET = "22.1"
_BATTER_TARGET = "22.3"
_BOWLER_TARGET = "22.4"
_CROSS_CHECK_SETTINGS: Mapping[str, str] = {
    WPA_RANKING: "descending unweighted WPA over the player's Pressure_Balls (Out_Of_Fold WP)",
    STRIKE_RATE_RANKING: "descending 100 x bat runs / Batter_Deliveries on Simple_Rule_Balls",
    AVERAGE_RANKING: "descending bat runs / max(1, Batter_Dismissals) on Simple_Rule_Balls",
    ECONOMY_RANKING: "ascending 6 x Bowler_Runs_Conceded / Legal_Balls on Simple_Rule_Balls",
    BOWLING_STRIKE_RANKING: "ascending Legal_Balls / Bowler_Wickets on Simple_Rule_Balls",
}


@dataclass(frozen=True)
class CrossCheckRanking:
    """One Cross_Check_Ranking of a Role (Requirement 22; module docstring, "Cross-checks").

    ``label``: the ranking's name (the robustness table's ``case``); ``role``; ``target_id`` and
    ``setting``: the robustness-table columns. ``spearman_reported`` is True only for the WPA
    ranking, whose row carries the position Spearman of criterion 2; the four simple rankings leave
    it blank (criterion 6 asks only for the overlap and the names). ``ranking`` holds one row per
    the Role's Eligible_Players (CROSS_CHECK_COLUMNS), each with the player's ``rank`` (1 the best),
    the ranking ``value`` (NaN when undefined) and whether that value is ``defined``.
    ``n_undefined``: the Eligible_Players placed below every defined value (criterion 5).
    ``comparison``: the RankingComparison with the Main_Ranking (overlap, entering and leaving
    players, and the position Spearman).
    """

    label: str
    role: str
    target_id: str
    setting: str
    spearman_reported: bool
    ranking: pd.DataFrame
    n_undefined: int
    comparison: RankingComparison


@dataclass(frozen=True)
class CrossCheckResult:
    """The Cross_Check_Rankings of both Roles (``cross_check_results``; module docstring).

    ``table``: the robustness-table rows (``cases.CROSS_CHECK`` kind, ``cases.ROBUSTNESS_COLUMNS``),
    in Role and ranking order; ``rankings``: the CrossCheckRanking of each, in the same order;
    ``detail``: the per-player rows of every ranking (CROSS_CHECK_COLUMNS), concatenated, for the
    checkpoint.
    """

    table: pd.DataFrame
    rankings: tuple[CrossCheckRanking, ...]
    detail: pd.DataFrame


def _eligible_player_ids(base: cases.BaseArtifacts, role: str) -> np.ndarray:
    """The Main_Ranking's Eligible_Players of ``role`` (ascending ``player_id``, int64)."""
    main = base.main[role]
    ids = main["player_id"].to_numpy(dtype=np.int64)[main["eligible"].to_numpy(dtype=bool)]
    return np.sort(ids)


def _accumulate(ids: np.ndarray, pid: Any, values: Any) -> np.ndarray:
    """Sum ``values`` per player (``pid``), aligned to ``ids`` (float64; 0 for an id with no row).

    A contribution whose ``pid`` is not an Eligible_Player of ``ids`` is dropped, so a Pressure_Ball
    or Simple_Rule_Ball by a player outside the ranked set adds nothing.
    """
    out = np.zeros(len(ids), dtype=np.float64)
    position = pd.Index(ids).get_indexer(np.asarray(pid, dtype=np.int64))
    keep = position >= 0
    np.add.at(out, position[keep], np.asarray(values, dtype=np.float64)[keep])
    return out


def _rank_order(ids: np.ndarray, value: np.ndarray, defined: np.ndarray, *, ascending: bool) -> np.ndarray:
    """Ranks 1..n (Requirement 22, criteria 1, 3-5): defined players by ``value``, then undefined last.

    Defined players are ordered by ``value`` (ascending for economy and bowling strike rate,
    descending otherwise); every remaining tie, and the undefined players, are ordered by lower
    ``player_id``. Returns one rank per id, in the order of ``ids``.
    """
    ids = np.asarray(ids, dtype=np.int64)
    flags = np.asarray(defined, dtype=bool)
    finite = np.where(flags, np.asarray(value, dtype=np.float64), 0.0)  # keep the arithmetic free of NaN
    sign = 1.0 if ascending else -1.0
    sort_value = np.where(flags, sign * finite, np.inf)  # undefined sort after every defined value
    order = np.lexsort((ids, sort_value, ~flags))  # primary ~defined, then value, then player_id
    rank = np.empty(len(ids), dtype=np.int64)
    rank[order] = np.arange(1, len(ids) + 1, dtype=np.int64)
    return rank


def _frame_arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """The per-Delivery arrays the cross-checks read from the ``validation`` per-Delivery frame."""
    columns = (
        "row", RULE_DEATH_OVERS, RULE_CHASE_RATE, "wp_before", "wp_after", "striker_id", "bowler_id", "batter", "cls",
        "bat_runs", "striker_out", "bowler_runs_conceded", "pressure_bat", "pressure_bowl",
    )  # fmt: skip
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"the validation frame lacks the cross-check column(s) {', '.join(missing)}")
    return {
        "srb": frame[RULE_DEATH_OVERS].to_numpy(dtype=bool) | frame[RULE_CHASE_RATE].to_numpy(dtype=bool),
        "swing": frame["wp_after"].to_numpy(dtype=np.float64) - frame["wp_before"].to_numpy(dtype=np.float64),
        "striker": frame["striker_id"].to_numpy(dtype=np.int64),
        "bowler": frame["bowler_id"].to_numpy(dtype=np.int64),
        "batter_delivery": frame["batter"].to_numpy(dtype=bool),
        "legal": frame["cls"].to_numpy(dtype=np.int64) == _LEGAL_BALL,
        "bat_runs": frame["bat_runs"].to_numpy(dtype=np.float64),
        "striker_out": frame["striker_out"].to_numpy(dtype=bool),
        "runs_conceded": frame["bowler_runs_conceded"].to_numpy(dtype=np.float64),
        "pressure_bat": pd.Series(frame["pressure_bat"]).fillna(False).to_numpy(dtype=bool),
        "pressure_bowl": frame["pressure_bowl"].to_numpy(dtype=bool),
    }


def _bowler_wickets(store: Any, rows: pd.Series) -> np.ndarray:
    """The Bowler_Wicket flag of each row (``reconstruct`` balls ``bowler_wicket``, aligned by ``row``)."""
    balls = store.load_table(RECONSTRUCT_STAGE, RECONSTRUCT_BALLS_TABLE)
    flag = pd.Series(
        pd.Series(balls["bowler_wicket"]).fillna(False).to_numpy(dtype=bool),
        index=pd.Index(balls["row"].to_numpy(dtype=np.int64)),
    )
    aligned = flag.reindex(np.asarray(rows, dtype=np.int64))
    if aligned.isna().any():
        raise StageError(
            "the reconstruct balls table does not hold every Validation_Delivery's row for the bowling strike rate; "
            "rerun the Pipeline from the reconstruct stage"
        )
    return aligned.to_numpy(dtype=bool)


def _make_ranking(
    base: cases.BaseArtifacts,
    *,
    role: str,
    label: str,
    target_id: str,
    ids: np.ndarray,
    value: np.ndarray,
    defined: np.ndarray,
    ascending: bool,
    spearman_reported: bool,
    player_names: Any,
) -> CrossCheckRanking:
    """Rank ``ids`` by ``value`` (``_rank_order``) and compare the ranking with the Main_Ranking (Requirement 22)."""
    rank = _rank_order(ids, value, defined, ascending=ascending)
    other = pd.DataFrame({"player_id": np.asarray(ids, dtype=np.int64), "rank": rank})
    comparison = compare_rankings(
        base.main[role], other, top_n=LIST_SIZE, on="rank", names=lambda pid: player_names(role, int(pid))
    )
    detail = pd.DataFrame(
        {
            "role": pd.array([role] * len(ids), dtype="str"),
            "ranking": pd.array([label] * len(ids), dtype="str"),
            "player_id": np.asarray(ids, dtype=np.int64),
            "rank": np.asarray(rank, dtype=np.int64),
            "value": np.where(np.asarray(defined, dtype=bool), np.asarray(value, dtype=np.float64), np.nan),
            "defined": np.asarray(defined, dtype=bool),
        },
        columns=list(CROSS_CHECK_COLUMNS),
    )
    return CrossCheckRanking(
        label=label, role=role, target_id=target_id, setting=_CROSS_CHECK_SETTINGS[label],
        spearman_reported=spearman_reported, ranking=detail,
        n_undefined=int((~np.asarray(defined, dtype=bool)).sum()), comparison=comparison,
    )  # fmt: skip


def cross_check_results(base: cases.BaseArtifacts, *, deliveries: pd.DataFrame | None = None) -> CrossCheckResult:
    """The Cross_Check_Rankings of both Roles (Requirement 22; module docstring, "Cross-checks").

    ``base``: the BaseArtifacts of the Main_Ranking, which gives the Eligible_Player set and the
    Main_Ranking of each Role. ``deliveries``: the ``validation`` per-Delivery frame
    (``validation.FRAME_TABLE``); None reads it from ``base.store``. The frame holds, per
    Validation_Delivery, the Out_Of_Fold WP before and after the ball, the batter-Role and
    bowler-Role Pressure_Ball flags, the two Simple_Rule flags (the ones the Pressure_Validator uses
    for V2 and V4), the Batter_Delivery flag, the bat runs, the striker's Batter_Dismissal flag and
    the Bowler_Runs_Conceded. The Bowler_Wicket count of the bowling strike rate comes from the
    ``reconstruct`` balls (``base.store``). Returns a CrossCheckResult. Raises StageError for a
    checkpoint that no longer holds every Validation_Delivery, and ValueError for a frame that lacks
    a column.
    """
    store = base.store
    frame = deliveries if deliveries is not None else store.load_table(VALIDATION_STAGE, VALIDATION_FRAME_TABLE)
    arrays = _frame_arrays(frame)
    bowler_wicket = _bowler_wickets(store, frame["row"])
    player_names = cases.PlayerNames.from_store(store)
    rankings: list[CrossCheckRanking] = []

    # Batter Role (Requirement 22, criteria 1 and 3): WPA, strike rate and batting average.
    batter_ids = _eligible_player_ids(base, BATTER_ROLE)
    pressure_bat = arrays["pressure_bat"]
    wpa_bat = _accumulate(batter_ids, arrays["striker"][pressure_bat], arrays["swing"][pressure_bat])
    rankings.append(_make_ranking(
        base, role=BATTER_ROLE, label=WPA_RANKING, target_id=_WPA_TARGET, ids=batter_ids, value=wpa_bat,
        defined=np.ones(len(batter_ids), dtype=bool), ascending=False, spearman_reported=True, player_names=player_names,
    ))  # fmt: skip
    bat_srb = arrays["batter_delivery"] & arrays["srb"]
    deliveries_bat = _accumulate(batter_ids, arrays["striker"][bat_srb], np.ones(int(bat_srb.sum())))
    runs_bat = _accumulate(batter_ids, arrays["striker"][bat_srb], arrays["bat_runs"][bat_srb])
    dismissals_bat = _accumulate(batter_ids, arrays["striker"][bat_srb], arrays["striker_out"][bat_srb].astype(np.float64))
    defined_bat = deliveries_bat > 0  # a batter with no Batter_Delivery on a Simple_Rule_Ball is undefined (criterion 5)
    strike_rate = np.where(defined_bat, 100.0 * runs_bat / np.where(defined_bat, deliveries_bat, 1.0), np.nan)
    average = np.where(defined_bat, runs_bat / np.maximum(1.0, dismissals_bat), np.nan)
    rankings.append(_make_ranking(
        base, role=BATTER_ROLE, label=STRIKE_RATE_RANKING, target_id=_BATTER_TARGET, ids=batter_ids, value=strike_rate,
        defined=defined_bat, ascending=False, spearman_reported=False, player_names=player_names,
    ))  # fmt: skip
    rankings.append(_make_ranking(
        base, role=BATTER_ROLE, label=AVERAGE_RANKING, target_id=_BATTER_TARGET, ids=batter_ids, value=average,
        defined=defined_bat, ascending=False, spearman_reported=False, player_names=player_names,
    ))  # fmt: skip

    # Bowler Role (Requirement 22, criteria 1 and 4): WPA, economy and bowling strike rate.
    bowler_ids = _eligible_player_ids(base, BOWLER_ROLE)
    pressure_bowl = arrays["pressure_bowl"]
    wpa_bowl = _accumulate(bowler_ids, arrays["bowler"][pressure_bowl], -arrays["swing"][pressure_bowl])
    rankings.append(_make_ranking(
        base, role=BOWLER_ROLE, label=WPA_RANKING, target_id=_WPA_TARGET, ids=bowler_ids, value=wpa_bowl,
        defined=np.ones(len(bowler_ids), dtype=bool), ascending=False, spearman_reported=True, player_names=player_names,
    ))  # fmt: skip
    bowl_srb = arrays["srb"]  # every Delivery is a Bowler_Delivery; keep the Simple_Rule_Balls
    conceded = _accumulate(bowler_ids, arrays["bowler"][bowl_srb], arrays["runs_conceded"][bowl_srb])
    legal_balls = _accumulate(bowler_ids, arrays["bowler"][bowl_srb], arrays["legal"][bowl_srb].astype(np.float64))
    wickets = _accumulate(bowler_ids, arrays["bowler"][bowl_srb], bowler_wicket[bowl_srb].astype(np.float64))
    srb_bowl_count = _accumulate(bowler_ids, arrays["bowler"][bowl_srb], np.ones(int(bowl_srb.sum())))
    has_srb = srb_bowl_count > 0
    economy_defined = has_srb & (legal_balls > 0)  # a zero Legal_Balls denominator is undefined (criterion 5)
    economy = np.where(economy_defined, 6.0 * conceded / np.where(economy_defined, legal_balls, 1.0), np.nan)
    bowling_defined = has_srb & (wickets > 0)  # a zero Bowler_Wickets denominator is undefined (criterion 5)
    bowling_strike = np.where(bowling_defined, legal_balls / np.where(bowling_defined, wickets, 1.0), np.nan)
    rankings.append(_make_ranking(
        base, role=BOWLER_ROLE, label=ECONOMY_RANKING, target_id=_BOWLER_TARGET, ids=bowler_ids, value=economy,
        defined=economy_defined, ascending=True, spearman_reported=False, player_names=player_names,
    ))  # fmt: skip
    rankings.append(_make_ranking(
        base, role=BOWLER_ROLE, label=BOWLING_STRIKE_RANKING, target_id=_BOWLER_TARGET, ids=bowler_ids,
        value=bowling_strike, defined=bowling_defined, ascending=True, spearman_reported=False, player_names=player_names,
    ))  # fmt: skip

    rows: list[dict[str, Any]] = []
    for ranking in rankings:
        row = cases.comparison_row(
            ranking.comparison, case=ranking.label, kind=cases.CROSS_CHECK, target_id=ranking.target_id,
            setting=ranking.setting, layer=None, role=ranking.role, n_eligible_case=len(ranking.ranking),
            n_eligible_main=base.n_eligible(ranking.role), is_default=False,
        )  # fmt: skip
        if not ranking.spearman_reported:
            row = {**row, "spearman": math.nan}  # criterion 6 asks for a Spearman only for WPA (criterion 2)
        rows.append(row)
    detail = pd.concat([ranking.ranking for ranking in rankings], ignore_index=True)
    return CrossCheckResult(table=cases.robustness_frame(rows), rankings=tuple(rankings), detail=detail)


def cross_checks(base: cases.BaseArtifacts) -> pd.DataFrame:
    """The Cross_Check_Ranking rows of the robustness table (Requirement 22; ``cross_check_results``).

    One row per Role and Cross_Check_Ranking (``cases.ROBUSTNESS_COLUMNS``, kind
    ``cases.CROSS_CHECK``): the top-10 overlap with the Main_Ranking, the entering and leaving names
    (criterion 6) and, for the WPA ranking only, the position Spearman (criterion 2); the four
    simple rankings leave the Spearman blank. Reads the ``validation`` per-Delivery frame and the
    ``reconstruct`` balls from ``base.store``.
    """
    return cross_check_results(base).table


# ===================================================================== the ``robustness`` stage

def _log_robustness_stage(
    ctx: StageContext,
    *,
    base: cases.BaseArtifacts,
    runs: Sequence[Any],
    repeat: RepeatabilityResult,
    cross: CrossCheckResult,
    table: pd.DataFrame,
    written: Sequence[Path],
    timings: tuple[float, float, float, float, float],
) -> None:
    log = ctx.log
    build_s, cases_s, repeat_s, cross_s, total_s = timings
    reused = sum(1 for run in runs if run.reused)
    log.info(
        f"Robustness_Suite (Requirements 20-22): {base.n_eligible(BATTER_ROLE):,} batter and "
        f"{base.n_eligible(BOWLER_ROLE):,} bowler Eligible_Players; BaseArtifacts built in {build_s:.1f} s. "
        f"{len(runs):,} Robustness_Cases ({len(runs) - reused:,} computed, {reused:,} reused) in {cases_s:.1f} s "
        f"(Requirement 21)"
    )

    # Repeatability (Requirement 20): reliabilities, the time-holdout and the IPL-vs-T20I correlation.
    split = repeat.split_half_reliability
    reliabilities = "; ".join(
        f"{row['role']} {row['estimate']} rho {float(row['reliability']):.4f} (n {int(row['n_players'])})"
        for row in split.to_dict("records")
    )
    period = "; ".join(
        f"{row['role']} {row['estimate']} "
        + ("not computed" if not bool(row["computed"]) else f"Spearman {float(row['spearman']):.4f}")
        + f" (n {int(row['n_players'])})"
        for row in repeat.period_spearman.to_dict("records")
    )
    log.info(f"Repeatability (Requirement 20): Split_Half_Reliability {reliabilities}. Early-vs-late {period}")
    competition = "; ".join(
        f"{row['role']} innings {int(row['innings'])} "
        + ("Spearman n/a" if math.isnan(float(row["spearman"])) else f"Spearman {float(row['spearman']):.4f}")
        + f" ({int(row['n_deciles'])} deciles)"
        for row in repeat.competition_spearman.to_dict("records")
    )
    log.info(f"Repeatability, IPL vs T20I mean Actual_WP_Swing across PI deciles: {competition}")
    for warning in repeat.warnings:  # Requirement 20, criterion 4
        log.warning(warning)

    # Cross-checks (Requirement 22): the overlap, the Spearman (WPA) and the entering/leaving names.
    for ranking in cross.rankings:
        comparison = ranking.comparison
        entering = _joined_change(comparison, "entering")
        leaving = _joined_change(comparison, "leaving")
        spearman = (
            f", position Spearman {comparison.spearman:.4f}" if ranking.spearman_reported else ""
        )
        log.info(
            f"Cross-check, {ranking.role} {ranking.label!r} (Requirement 22): top-10 overlap "
            f"{comparison.overlap}/{LIST_SIZE}{spearman}; {ranking.n_undefined:,} player(s) with no defined value "
            f"(criterion 5); entering [{entering}]; leaving [{leaving}]"
        )
    log.info(
        f"Robustness table: {len(table):,} rows "
        f"({int((table['kind'] != cases.CROSS_CHECK).sum()):,} case rows, "
        f"{int((table['kind'] == cases.CROSS_CHECK).sum()):,} cross-check rows). Timing: cases {cases_s:.1f} s, "
        f"repeatability {repeat_s:.1f} s, cross-checks {cross_s:.1f} s; stage {total_s:.1f} s. Files: "
        + ", ".join(_shown_path(path, ctx.root) for path in written)
    )


def _joined_change(comparison: RankingComparison, status: str) -> str:
    """The entering or leaving players of a comparison as ``name (id)`` strings, in change order."""
    changes = comparison.changes
    mine = changes.loc[changes["status"].astype("string").to_numpy() == status]
    parts = []
    for row in mine.to_dict("records"):
        name = row["player_name"]
        parts.append(f"{name} ({int(row['player_id'])})" if isinstance(name, str) else str(int(row["player_id"])))
    return ", ".join(parts)


def run_stage(ctx: StageContext) -> None:
    """The ``robustness`` Pipeline_Stage: the Robustness_Suite of Requirements 20-22 (module docstring).

    Reads the completed checkpoints of the Run (``cases.BaseArtifacts.build`` rebuilds and checks
    the Main_Ranking's arrays, and ``repeatability`` and ``cross_checks`` read the ``rating`` and
    ``validation`` checkpoints). It runs the ablation and sensitivity cases (Requirement 21), the
    repeatability checks (Requirement 20) and the Cross_Check_Rankings (Requirement 22); writes one
    ``robustness_table.csv`` (the ``all_cases`` rows then the Cross_Check_Ranking rows) and the six
    repeatability CSVs to ``outputs/robustness/``; checkpoints the tables for the Report_Builder;
    and logs the Requirement 20, criterion 4 warnings, the not-comparable cases and the cross-check
    outcomes. Raises StageError (exit code 5) when the checkpoints no longer reproduce the
    Main_Ranking or a checkpoint file is missing or changed.
    """
    started = time.perf_counter()
    settings, store, log = ctx.settings, ctx.store, ctx.log

    # 1. The Main_Ranking's cached artifacts (checked bit for bit against the rating checkpoint).
    tick = time.perf_counter()
    base = cases.BaseArtifacts.build(settings, store)
    build_seconds = time.perf_counter() - tick

    # 2. The Robustness_Cases of Requirement 21 (ablations, sensitivities, the Z-score case, alternatives).
    tick = time.perf_counter()
    names = cases.PlayerNames.from_store(store)
    specs = cases.all_cases(settings, base)
    runs = cases.run_all_cases(specs, base, names=names, log=log)
    case_table = cases.robustness_table(runs, base)
    cases_seconds = time.perf_counter() - tick

    # 3. Repeatability (Requirement 20) from the rating and validation checkpoints.
    tick = time.perf_counter()
    frame = store.load_table(VALIDATION_STAGE, VALIDATION_FRAME_TABLE)
    repeat = repeatability(
        contributions=store.load_table(RATING_STAGE, CONTRIBUTIONS_TABLE),
        match_universe=store.load_table(RATING_STAGE, MATCH_UNIVERSE_TABLE),
        player_estimates=store.load_table(RATING_STAGE, PLAYER_ESTIMATES_TABLE),
        role_summary=store.load_table(RATING_STAGE, ROLE_SUMMARY_TABLE),
        v1_deciles=store.load_table(VALIDATION_STAGE, VALIDATION_V1_TABLE),
        validation_deliveries=frame,
        period_minimum=settings.get(_D19),
    )
    repeat_seconds = time.perf_counter() - tick

    # 4. The Cross_Check_Rankings of Requirement 22 (the same validation frame).
    tick = time.perf_counter()
    cross = cross_check_results(base, deliveries=frame)
    cross_seconds = time.perf_counter() - tick

    # 5. The combined robustness table: the all_cases rows, then the Cross_Check_Ranking rows.
    table = pd.concat([case_table, cross.table], ignore_index=True)

    # 6. Files in outputs/robustness/ (the robustness table and the six repeatability CSVs).
    folder = settings.path("outputs") / ROBUSTNESS_FOLDER
    written: list[Path] = [cases.write_robustness_table(table, folder)]
    written.extend(write_repeatability(repeat, folder))

    # 7. The checkpoint (task 27.1 reads these).
    store.save_table(ROBUSTNESS_STAGE, ROBUSTNESS_TABLE_NAME, table)
    for file_name, result in repeat.files().items():
        store.save_table(ROBUSTNESS_STAGE, REPEATABILITY_TABLES[file_name], result)
    store.save_table(ROBUSTNESS_STAGE, CROSS_CHECK_TABLE, cross.detail)

    # 8. Data-derived constants: the undefined-player counts of the simple Cross_Check_Rankings (criterion 5).
    for ranking in cross.rankings:
        if not ranking.spearman_reported:
            ctx.constants.record(
                f"{ROBUSTNESS_STAGE}.cross_check_undefined.{ranking.label.replace(' ', '_')}",
                ranking.n_undefined, _COMPONENT,
            )

    _log_robustness_stage(
        ctx, base=base, runs=runs, repeat=repeat, cross=cross, table=table, written=written,
        timings=(build_seconds, cases_seconds, repeat_seconds, cross_seconds, time.perf_counter() - started),
    )  # fmt: skip
