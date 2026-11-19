"""The Activity_Comparison_Ranking (Requirement 25, criteria 11, 12 and 15) and the
``activity_comparison`` Pipeline_Stage; design sections "Robustness_Suite and activity
comparison" and "Pipeline_Stage registry" (stage 18).

No function reads ``config.py``: every Config value is an argument (``run_stage`` reads them).

What the comparison is
    The Main_Ranking applies the Recent_Activity_Rule as set in the Config (on as delivered,
    Requirement 25, criterion 10). Where the rule is on, every Run also ranks each Role with the
    rule off and every other input and setting unchanged: the Activity_Comparison_Ranking
    (glossary). It is not a Robustness_Case, and no submitted output uses it. Its top 10 per Role
    become the Comparison_Lists, which the List_Builder writes (task 22.2, criteria 13 and 14).

``activity_comparison`` (criteria 11 and 12)
    Takes the rating checkpoint's Contributions, the eligibility counts of both Roles
    (``rating.eligibility_counts``), the bootstrap checkpoint's draws and the positions of the
    resampled matches, and the Main_Ranking's minimum counts and settings. No model is fitted, and
    every Out_Of_Fold_Prediction, PI, Pressure_Ball flag, NBV, Dismissal_Charge, Recency_Weight and
    Stakes_Weight is the one inside the contributions, so they all equal the Main_Ranking's.

    1. Eligibility: ``rating.eligibility`` and ``rating.eligibility_report`` on the same counts
       with ``recent_rule=False`` and the Main_Ranking's minimum Pressure_Ball count of each Role
       (Requirement 17, criterion 7 or 8), the D18 default minimum Pressure_Match count, the
       2-Calendar_Year rule and the 50% IPL or Full_Member_Match share. So the Eligible_Players
       follow Requirement 17, criteria 1-3 and 5, and the Recent_Activity_Rule is the only rule
       that differs. Turning a rule off never removes a player, so every Main_Ranking
       Eligible_Player must be in the comparison set (ValueError otherwise).
    2. Ranking: per Role, ``prepare_role`` over those players and ``rank_role`` with its defaults
       (mult all ones, D12). The ranking therefore estimates its own Replacement_Level, the four
       tau^2, both Split_Half_Reliability values and the Blend_Weights over its own
       Eligible_Players, then the Shrinkage, Final_Score and ranks with the tie-break rule
       (Requirement 18, criteria 1-13). Every player's Requirement 16 estimates are recomputed
       against that Replacement_Level for the ``player_estimates`` table.
    3. Check: one copy of every resampled match, ranked with the comparison ranking's own
       Role-level values and ``rankable_only``, must reproduce the comparison ranking bit for bit
       (StageError otherwise), as the ``bootstrap`` stage checks for the Main_Ranking.
    4. Bootstrap: ``robustness.bootstrap`` over the comparison Eligible_Players with the saved
       draws, so every resample draws the same matches as the Main_Ranking's resample of the same
       number (criterion 12), and both Roles share each resample (Requirement 19, criterion 1).
       Each Role's RoleHyper is ``RoleHyper.from_ranking`` of its comparison ranking, never the
       Main_Ranking's: every resample holds the Role-level values of the ranking it resamples
       (Requirement 19, criterion 2). ``bootstrap_table`` then gives each player's 95% percentile
       interval of Final_Score, median rank, P_Top10 and unranked count.

    Returns an ActivityResult with the tables of the checkpoint below.

The ``activity_comparison`` Pipeline_Stage (``run_stage``)
    Where ``data_rules.recent_activity_rule`` is off, the Main_Ranking already has the rule off and
    no comparison applies: the stage saves the JSON value ``status`` with ``applies`` false, writes
    one Run_Log line, and reads, computes and writes nothing else.

    Where the rule is on, the stage reads the ``load`` tables ``playing_eleven``, ``matches`` and
    ``ball_by_ball`` (provisional names for the Run_Log only), the ``rating`` tables
    ``contributions``, ``match_universe``, ``player_estimates``, ``eligibility_report`` and
    ``role_summary``, and the ``bootstrap`` arrays ``resample_draws`` and ``resample_matches``.
    Before computing, it checks that these inputs and the Config still give the Main_Ranking, so
    that the comparison differs from it only by the rule (criterion 12). Any failure raises
    StageError, naming the stage to rerun:

    * the Main_Ranking was computed with the rule on, with the Config's D18 default minimum
      Pressure_Match count and the Config's D12 percentile;
    * the counts, the Config's recent years, 2-Calendar_Year rule and share rule, and the
      Main_Ranking's minimum Pressure_Ball counts, with the rule on, reproduce the rating
      checkpoint's ``eligibility_report`` and each Role's Eligible_Player set exactly;
    * ``resample_matches`` holds the ``match_id`` of every resampled match of ``match_universe``
      in order, and the draws are the (D16, N) array that ``draw_resamples`` gives for
      ``seeds.bootstrap``.

    It then runs ``activity_comparison`` and saves its checkpoint. That checkpoint lives under
    ``outputs/final/comparison/_stage/18_activity_comparison/`` (COMPARISON_ROOT of the
    runner), so that the stage writes no file outside ``outputs/final/comparison/`` apart from
    Run_Log entries (criterion 15). In particular it never creates, modifies or deletes the
    Main_List files, ``outputs/final/unstop_entry_checklist.md``, the ``outputs/rating/`` tables or
    ``outputs/robustness/bootstrap_*.csv``. Every value comes from the Activity_Comparison_Ranking
    and its bootstrap:

    * ``player_estimates``: ``rating.PLAYER_TABLE_COLUMNS``, the layout of the rating
      checkpoint's table, with the same rows (every player and Role with a contribution row).
      ``eligible`` and the eligibility columns are those with the rule off (``fail_recent_activity``
      is False everywhere, ``recent_activity`` still records the fact). The Requirement 18 columns
      (``rank``, ``final_score``, shrunk values, B values, halves, pool flags) come from the
      comparison ranking, and ``uplift`` and ``replacement_level`` use its Replacement_Level;
    * ``role_summary``: ``rating.ROLE_SUMMARY_COLUMNS``, the rule-off eligibility report and the
      comparison ranking's Role values (``RoleRanking.summary``) per Role, with ``n_warnings``, the
      Role's comparison ranking warnings. ``RoleHyper.from_summary(role_summary, role)`` rebuilds
      the Role-level values that the comparison bootstrap held;
    * ``eligibility_report``: ``rating.ELIGIBILITY_REPORT_COLUMNS`` with the rule off;
    * ``eligible_players``: ELIGIBLE_PLAYER_COLUMNS, one row per comparison Eligible_Player
      (by Role, batter first, then ``player_id``), with ``main_eligible`` True for the Main_Ranking's
      Eligible_Players. A player with ``main_eligible`` False is one that only the
      Recent_Activity_Rule left out of the Main_Ranking;
    * ``bootstrap``: ``robustness.BOOTSTRAP_COLUMNS`` per Eligible_Player (by Role, then
      ``player_id``): the interval ``ci_low`` and ``ci_high``, ``median_rank``, ``p_top10``,
      ``n_ranked`` and ``n_unranked``. As in the bootstrap stage's table, ``main_rank`` and
      ``main_final_score`` hold the rank and Final_Score of the full-data ranking that the
      resamples resample, which here is the Activity_Comparison_Ranking, not the Main_Ranking;
    * ``warnings``: ``rating.WARNING_COLUMNS``, the comparison ranking's warnings;
      ``bootstrap_warnings``: ``robustness.BOOTSTRAP_WARNING_COLUMNS``;
    * ``status`` (JSON): ``applies`` true, the settings applied (minimum counts, D12, B, N, the
      seed and the SHA-256 of the draws used) and both Eligible_Player counts per Role.

    The data-derived constants ``activity_comparison.<quantity>.<role>`` for
    ``rating.CONSTANT_QUANTITIES`` (the minimum Pressure_Ball count applied, the Eligible_Players,
    Replacement_Level, the four tau^2, both rho and the Blend_Weights) are produced by the
    Rating_Engine functions; the runner saves them in this checkpoint. The Run_Log gets the
    inputs, each Role's eligibility and Role values beside the Main_Ranking's, the comparison
    top 10 with Final_Score, interval and P_Top10, the players entering and leaving against the
    Main top 10, every comparison warning (prefixed, so it is not mistaken for a Main_Ranking
    warning) and the timings.

Reading the checkpoint (the List_Builder, task 22.2)
    ``store.load_json(STAGE_NAME, STATUS_JSON)["applies"]`` says whether a comparison exists. When
    it does, a Comparison_List row of a Role is a ``player_estimates`` row with ``rank`` 1-10
    (``final_score``, ``hit_rate``) joined on ``player_id`` with the ``bootstrap`` row (``ci_low``,
    ``ci_high``, ``median_rank``, ``p_top10``). The Main_Ranking ranks come from the rating
    checkpoint's ``player_estimates``; the two Eligible_Player counts from ``eligible_players`` (or
    the two ``eligibility_report`` tables).

Errors: ValueError or TypeError, naming the problem, for inputs that do not fit together or a
Config value of the wrong type; StageError (exit code 5) when the checkpoints or the Config no
longer give the Main_Ranking, or a Role has fewer than LIST_SIZE Eligible_Players.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from superr import rating
from superr.errors import StageError
from superr.rating import (
    CONSTANT_QUANTITIES,
    ELIGIBILITY_COLUMNS,
    ELIGIBILITY_RULES,
    LIST_SIZE,
    PLAYER_ESTIMATE_COLUMNS,
    PLAYER_TABLE_COLUMNS,
    RANKING_SUMMARY_FIELDS,
    RANKABLE_MIN_ORDINARY_BALLS,
    RANKABLE_MIN_PRESSURE_MATCHES,
    ROLE_HYPER_FIELDS,
    ROLE_SUMMARY_COLUMNS,
    WARNING_COLUMNS,
    Aggregator,
    Contributions,
    PlayerEstimates,
    RatingWarning,
    RoleData,
    RoleHyper,
    RoleRanking,
    eligibility,
    eligibility_report,
    prepare_role,
    rank_role,
    require_eligible,
)
from superr.robustness import (
    BOOTSTRAP_STAGE,
    RESAMPLE_DRAWS,
    RESAMPLE_MATCHES,
    RoleBootstrap,
    bootstrap,
    bootstrap_table,
    draw_resamples,
    multiplicities,
)
from superr.value import BATTER_ROLE, BOWLER_ROLE, ROLES

if TYPE_CHECKING:
    from superr.pipeline import StageContext

__all__ = [
    "ActivityResult",
    "BOOTSTRAP_TABLE",
    "BOOTSTRAP_WARNINGS_TABLE",
    "CHECKPOINT_TABLES",
    "ELIGIBILITY_REPORT_TABLE",
    "ELIGIBLE_PLAYERS_TABLE",
    "ELIGIBLE_PLAYER_COLUMNS",
    "PLAYER_ESTIMATES_TABLE",
    "ROLE_SUMMARY_TABLE",
    "STAGE_NAME",
    "STATUS_JSON",
    "WARNINGS_TABLE",
    "activity_comparison",
    "run_stage",
]

# ================================================================================ constants

#: Name of the Pipeline_Stage that computes the Activity_Comparison_Ranking, and of its checkpoint.
STAGE_NAME = "activity_comparison"

#: Checkpoint tables and JSON value of the stage (module docstring), in the order they are saved.
PLAYER_ESTIMATES_TABLE = "player_estimates"
ROLE_SUMMARY_TABLE = "role_summary"
ELIGIBILITY_REPORT_TABLE = "eligibility_report"
ELIGIBLE_PLAYERS_TABLE = "eligible_players"
BOOTSTRAP_TABLE = "bootstrap"
WARNINGS_TABLE = "warnings"
BOOTSTRAP_WARNINGS_TABLE = "bootstrap_warnings"
CHECKPOINT_TABLES: tuple[str, ...] = (
    PLAYER_ESTIMATES_TABLE, ROLE_SUMMARY_TABLE, ELIGIBILITY_REPORT_TABLE, ELIGIBLE_PLAYERS_TABLE, BOOTSTRAP_TABLE,
    WARNINGS_TABLE, BOOTSTRAP_WARNINGS_TABLE,
)  # fmt: skip
STATUS_JSON = "status"

#: Columns of the ``eligible_players`` table: one row per comparison Eligible_Player.
ELIGIBLE_PLAYER_COLUMNS: tuple[str, ...] = ("role", "player_id", "main_eligible")

#: The eligibility columns of PLAYER_TABLE_COLUMNS that come from the eligibility table, after the
#: estimates: the counts the estimates do not hold and the failure flags (rating's layout).
_ELIGIBILITY_EXTRA: tuple[str, ...] = tuple(
    name for name in PLAYER_TABLE_COLUMNS
    if name in ELIGIBILITY_COLUMNS and name not in PLAYER_ESTIMATE_COLUMNS and name != "eligible"
)  # fmt: skip

_COMPONENT = "Rating_Engine"
_D12 = "design_choices.D12_replacement_pct"
_D16 = "design_choices.D16_n_boot"
_D18_DEFAULT = "design_choices.D18_min_matches_default"
_RECENT_RULE = "data_rules.recent_activity_rule"
_RECENT_YEARS = "data_rules.recent_years"
_MIN_SHARE = "data_rules.min_major_share"
_MIN_YEARS = "data_rules.min_calendar_years"
_SEED = "seeds.bootstrap"
_LOAD_STAGE = "load"
_RATING_STAGE = rating.STAGE_NAME
_NAME_COLUMNS: Mapping[str, tuple[str, str]] = {
    BATTER_ROLE: ("striker_id", "striker_name"), BOWLER_ROLE: ("bowler_id", "bowler_name"),
}  # fmt: skip
#: Run_Log prefix of every comparison warning, which would otherwise read like a Main_Ranking one.
_LOG_PREFIX = "Activity_Comparison_Ranking (Recent_Activity_Rule off)"
_NOT_APPLICABLE = (
    "the Recent_Activity_Rule is off in the Config (data_rules.recent_activity_rule = False), so the "
    "Main_Ranking already has the rule off and no Activity_Comparison_Ranking applies (Requirement 25, "
    "criterion 11); the stage computed no ranking or bootstrap and saved nothing else"
)


# =============================================================================== the result

@dataclass(frozen=True, eq=False)
class ActivityResult:
    """The Activity_Comparison_Ranking of both Roles and its bootstrap (``activity_comparison``).

    Tables, each in the layout of the checkpoint table of the same name (module docstring):
    ``eligibility`` (``rating.ELIGIBILITY_COLUMNS``, the rule-off eligibility of every player of
    both Roles), ``report`` (the rule-off ``eligibility_report``), ``players``
    (``player_estimates``), ``summary`` (``role_summary``), ``eligible_players``, ``bootstrap``,
    ``warnings`` and ``bootstrap_warnings``. Per Role name: ``data`` (the RoleData of the
    comparison Eligible_Players), ``rankings`` (the comparison RoleRanking), ``hypers`` (its
    RoleHyper, held in every resample) and ``bootstraps`` (the RoleBootstrap). ``timings``:
    seconds spent on the rankings (``"ranking"``) and on the resamples (``"bootstrap"``).
    """

    eligibility: pd.DataFrame
    report: pd.DataFrame
    data: Mapping[str, RoleData]
    rankings: Mapping[str, RoleRanking]
    hypers: Mapping[str, RoleHyper]
    bootstraps: Mapping[str, RoleBootstrap]
    players: pd.DataFrame
    summary: pd.DataFrame
    eligible_players: pd.DataFrame
    bootstrap: pd.DataFrame
    warnings: pd.DataFrame
    bootstrap_warnings: pd.DataFrame
    timings: Mapping[str, float]

    def __repr__(self) -> str:
        counts = ", ".join(f"{role} {len(self.rankings[role]):,}" for role in ROLES)
        n_boot = next(iter(self.bootstraps.values())).n_boot
        return (
            f"ActivityResult(Eligible_Players {counts}; {n_boot:,} Bootstrap_Resamples; "
            f"{len(self.warnings)} warning(s))"
        )


# ================================================================================= helpers

def _same(a: Any, b: Any) -> bool:
    return bool(np.array_equal(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), equal_nan=True))


def _player_table(everyone: PlayerEstimates, rules: pd.DataFrame, ranking: RoleRanking) -> pd.DataFrame:
    """One Role's rows of ``player_estimates`` (``rating.PLAYER_TABLE_COLUMNS``), as the rating stage builds them.

    ``everyone``: the estimates of every player of the Role against the ranking's
    Replacement_Level; ``rules``: the Role's eligibility rows, in the same player order;
    ``ranking``: the Role's RoleRanking, whose values fill the Eligible_Players' rows.
    """
    frame = everyone.to_frame()
    ids = frame["player_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(ids, rules["player_id"].to_numpy(dtype=np.int64)):
        raise ValueError(f"the eligibility rows of the {ranking.role} Role do not follow its estimates' players")
    at = np.searchsorted(ids, ranking.player_id)
    if not np.array_equal(ids[np.minimum(at, len(ids) - 1)], ranking.player_id):
        raise ValueError(f"an Eligible_Player of the {ranking.role} Role has no contribution row")

    def spread(values: np.ndarray, fill: Any = np.nan) -> np.ndarray:
        out = np.full(len(ids), fill, dtype=np.asarray(values).dtype)
        out[at] = values
        return out

    columns = {
        "eligible": rules["eligible"].to_numpy(dtype=bool),
        "final_score": spread(ranking.final_score),
        "shrunk_clutch": spread(ranking.shrunk_clutch),
        "shrunk_usual": spread(ranking.shrunk_usual),
        "B_usual": spread(ranking.b_usual),
        "gap": frame["clutch"].to_numpy(dtype=np.float64) - frame["usual"].to_numpy(dtype=np.float64),
        "shrunk_gap": spread(ranking.shrunk_gap),
        "B_gap": spread(ranking.b_gap),
        "clutch_fallback": spread(ranking.fallback, False),
        "B_clutch": spread(ranking.b_clutch),
        "shrunk_uplift": spread(ranking.shrunk_uplift),
        "B_uplift": spread(ranking.b_uplift),
        "in_pool": spread(ranking.pool, False),
        "in_split_half": spread(ranking.qualifying, False),
        "clutch_odd": spread(ranking.odd.clutch),
        "clutch_even": spread(ranking.even.clutch),
        "uplift_odd": spread(ranking.odd.uplift),
        "uplift_even": spread(ranking.even.uplift),
    }
    for name, values in columns.items():
        frame[name] = values
    frame["rank"] = pd.Series(spread(ranking.rank)).astype("Int64")
    for name in _ELIGIBILITY_EXTRA:
        frame[name] = rules[name].to_numpy()
    return frame.loc[:, list(PLAYER_TABLE_COLUMNS)]


def _warning_table(notes: Sequence[RatingWarning]) -> pd.DataFrame:
    """The ``warnings`` table (``rating.WARNING_COLUMNS``) of the comparison rankings."""
    return pd.DataFrame(
        {
            "role": pd.array([note.role for note in notes], dtype="string"),
            "kind": pd.array([note.kind for note in notes], dtype="string"),
            "estimate": pd.array([note.estimate for note in notes], dtype="string"),
            "message": pd.array([note.message for note in notes], dtype="string"),
        },
        columns=list(WARNING_COLUMNS),
    )


def _role_summary(
    report: pd.DataFrame, rankings: Mapping[str, RoleRanking], notes: Sequence[RatingWarning]
) -> pd.DataFrame:
    """The ``role_summary`` table (``rating.ROLE_SUMMARY_COLUMNS``), one row per Role, as the rating stage builds it."""
    records = []
    for role in ROLES:
        record = report.loc[report["role"] == role].iloc[0].to_dict()
        values = rankings[role].summary()
        record.update({name: values[name] for name in RANKING_SUMMARY_FIELDS})
        record["n_warnings"] = sum(note.role == role for note in notes)
        records.append(record)
    table = pd.DataFrame.from_records(records, columns=list(ROLE_SUMMARY_COLUMNS))
    table["role"] = table["role"].astype("str")
    table["recent_rule_on"] = table["recent_rule_on"].astype(bool)
    integers = [name for name in ROLE_SUMMARY_COLUMNS if name.startswith(("n_", "min_", "fail_"))]
    return table.astype({name: np.int64 for name in integers})


def _eligible_rows(role: str, ids: np.ndarray, main_ids: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "role": pd.array([role] * len(ids), dtype="str"),
            "player_id": np.asarray(ids, dtype=np.int64),
            "main_eligible": np.isin(ids, main_ids),
        },
        columns=list(ELIGIBLE_PLAYER_COLUMNS),
    )


def _check_once(data: RoleData, once: np.ndarray, hyper: RoleHyper, ranking: RoleRanking, pct: float) -> None:
    """StageError unless one copy of every resampled match reproduces the comparison ranking (bit for bit)."""
    check = rank_role(data, once, fixed=hyper, replacement_pct=pct, rankable_only=True)
    if not (_same(check.final_score, ranking.final_score) and _same(check.rank, ranking.rank)):
        raise StageError(
            f"a resample that draws every resampled match once does not reproduce the {ranking.role} "
            "Activity_Comparison_Ranking with its own Role-level values: a rated match may lie outside the resampled "
            f"matches, or an Eligible_Player has fewer than {RANKABLE_MIN_PRESSURE_MATCHES} Pressure_Matches or fewer "
            f"than {RANKABLE_MIN_ORDINARY_BALLS} Ordinary_Ball"
        )


# ======================================================================= the comparison

def activity_comparison(
    contrib: Contributions,
    counts: pd.DataFrame,
    draws: Any,
    positions: Any,
    *,
    main_eligible: Mapping[str, Any],
    min_balls: int | Mapping[str, int],
    min_matches: int | Mapping[str, int],
    min_years: int,
    min_share: float,
    replacement_pct: float,
) -> ActivityResult:
    """The Activity_Comparison_Ranking of both Roles and its bootstrap (Requirement 25, criteria 11-12).

    ``contrib``: the Main_Ranking's Contributions, both Roles (the rating checkpoint's).
    ``counts``: ``rating.eligibility_counts`` of those contributions. ``draws``: the (B, N)
    resample draws that the Main_Ranking's bootstrap used; ``positions``: the match-universe
    index of each of the N resampled matches. ``main_eligible``: Role name -> the ``player_id``
    values of the Main_Ranking's Eligible_Players. ``min_balls``: the Main_Ranking's selected
    minimum Pressure_Ball count (one number, or a mapping by Role name); ``min_matches``: the D18
    default minimum Pressure_Match count; ``min_years`` (``data_rules.min_calendar_years``) and
    ``min_share`` (``data_rules.min_major_share``): the other two rules; ``replacement_pct``: D12.

    The Recent_Activity_Rule is off; nothing else differs from the Main_Ranking, and no model is
    fitted. Returns an ActivityResult. Raises ValueError or TypeError for inputs that do not fit
    together, including a Main_Ranking Eligible_Player outside the comparison set; StageError for
    a Role with fewer than LIST_SIZE Eligible_Players, or when one copy of every resampled match
    does not reproduce a comparison ranking.
    """
    if not isinstance(contrib, Contributions):
        raise TypeError(f"contrib must be Contributions; got {type(contrib).__name__}")
    if not isinstance(main_eligible, Mapping) or set(main_eligible) != set(ROLES):
        raise ValueError(
            f"main_eligible must map each Role ({', '.join(ROLES)}) to the Main_Ranking's Eligible_Players"
        )
    rules = {
        "min_balls": min_balls, "min_matches": min_matches, "recent_rule": False, "min_years": min_years,
        "min_share": min_share,
    }  # fmt: skip

    # 1. Eligibility by Requirement 17, criteria 1-3 and 5: the Main_Ranking's rules without the Recent_Activity_Rule.
    table = eligibility(counts, **rules)
    report = eligibility_report(counts, **rules)
    require_eligible(report)

    # 2. Each Role's own Requirement 16 and 18 values, Final_Scores and ranks.
    tick = time.perf_counter()
    data: dict[str, RoleData] = {}
    rankings: dict[str, RoleRanking] = {}
    hypers: dict[str, RoleHyper] = {}
    frames, eligible_frames = [], []
    notes: list[RatingWarning] = []
    for role in ROLES:
        rules_of_role = table.loc[(table["role"] == role).to_numpy(dtype=bool)].reset_index(drop=True)
        ids = rules_of_role.loc[rules_of_role["eligible"].to_numpy(dtype=bool), "player_id"].to_numpy(dtype=np.int64)
        main_ids = np.unique(np.asarray(main_eligible[role], dtype=np.int64))
        dropped = np.setdiff1d(main_ids, ids)
        if len(dropped):
            raise ValueError(
                f"{len(dropped):,} Main_Ranking Eligible_Player(s) of the {role} Role (player_id "
                f"{', '.join(str(int(pid)) for pid in dropped[:10])}) are not Eligible_Players with the "
                "Recent_Activity_Rule off; turning a rule off cannot remove a player, so main_eligible and the counts "
                "do not come from the same Main_Ranking"
            )
        data[role] = prepare_role(contrib, role, ids)
        rankings[role] = rank_role(data[role], replacement_pct=replacement_pct)
        hypers[role] = RoleHyper.from_ranking(rankings[role])  # its own values, never the Main_Ranking's
        everyone = Aggregator(contrib, role=role).estimates(replacement_level=rankings[role].replacement_level)
        frames.append(_player_table(everyone, rules_of_role, rankings[role]))
        eligible_frames.append(_eligible_rows(role, data[role].eligible, main_ids))
        notes.extend(rankings[role].warnings)
    ranking_seconds = time.perf_counter() - tick

    # 3. One copy of every resampled match, with each ranking's own fixed values, reproduces it.
    once = multiplicities(np.arange(len(np.asarray(positions))), contrib.n_matches, positions)
    for role in ROLES:
        _check_once(data[role], once, hypers[role], rankings[role], replacement_pct)

    # 4. The bootstrap on the saved draws, both Roles per resample, each Role's own values held fixed.
    tick = time.perf_counter()
    boots = bootstrap(data, hypers, draws, positions, replacement_pct=replacement_pct)
    bootstrap_seconds = time.perf_counter() - tick
    boot_table = pd.concat([bootstrap_table(boots[role], rankings[role]) for role in ROLES], ignore_index=True)

    return ActivityResult(
        eligibility=table,
        report=report,
        data=data,
        rankings=rankings,
        hypers=hypers,
        bootstraps=boots,
        players=pd.concat(frames, ignore_index=True),
        summary=_role_summary(report, rankings, notes),
        eligible_players=pd.concat(eligible_frames, ignore_index=True),
        bootstrap=boot_table,
        warnings=_warning_table(notes),
        bootstrap_warnings=pd.concat([boots[role].warnings for role in ROLES], ignore_index=True),
        timings={"ranking": ranking_seconds, "bootstrap": bootstrap_seconds},
    )


# ======================================================================= the stage's checks

def _role_rows(table: pd.DataFrame, what: str) -> dict[str, dict[str, Any]]:
    """One record per Role of a per-Role table (eligibility_report, role_summary); ValueError unless one each."""
    rows: dict[str, dict[str, Any]] = {}
    for role in ROLES:
        found = table.loc[(table["role"].astype("string") == role).to_numpy(dtype=bool)]
        if len(found) != 1:
            raise ValueError(
                f"the rating checkpoint's {what} must hold one row of the {role} Role; it holds {len(found)}"
            )
        rows[role] = found.iloc[0].to_dict()
    return rows


def _main_minimums(
    main_report: pd.DataFrame, main_summary: pd.DataFrame, *, min_matches: int, pct: Any
) -> dict[str, int]:
    """The Main_Ranking's selected minimum Pressure_Ball count of each Role, after checking its other settings.

    StageError unless the Main_Ranking was computed with the Recent_Activity_Rule on, the Config's
    D18 default minimum Pressure_Match count and the Config's D12 percentile (Requirement 25,
    criterion 12).
    """
    report_rows = _role_rows(main_report, rating.ELIGIBILITY_REPORT_TABLE)
    summary_rows = _role_rows(main_summary, rating.ROLE_SUMMARY_TABLE)
    rerun = "rerun the Pipeline from the rating stage"
    chosen: dict[str, int] = {}
    for role in ROLES:
        record = report_rows[role]
        if not bool(record["recent_rule_on"]):
            raise StageError(
                f"the Recent_Activity_Rule is on in the Config, but the rating checkpoint's {role} Main_Ranking was "
                f"computed with it off, so it is no Main_Ranking to compare against; {rerun}"
            )
        if int(record["min_pressure_matches"]) != int(min_matches):
            raise StageError(
                f"the rating checkpoint's {role} Main_Ranking used the minimum Pressure_Match count "
                f"{int(record['min_pressure_matches'])}, but the Config's {_D18_DEFAULT} is {min_matches!r}; the "
                f"Activity_Comparison_Ranking must use the Main_Ranking's value (Requirement 25, criterion 12); {rerun}"
            )
        used = float(summary_rows[role]["replacement_pct"])
        if used != float(pct):
            raise StageError(
                f"the rating checkpoint's {role} Main_Ranking used the Replacement_Level percentile {used:g}, but the "
                f"Config's {_D12} is {pct!r}; the Activity_Comparison_Ranking must use the Main_Ranking's value "
                f"(Requirement 25, criterion 12); {rerun}"
            )
        chosen[role] = int(record["min_pressure_balls"])
    return chosen


def _check_main_eligibility(
    counts: pd.DataFrame,
    main_report: pd.DataFrame,
    main_players: pd.DataFrame,
    *,
    min_balls: Mapping[str, int],
    min_matches: int,
    min_years: int,
    min_share: float,
) -> dict[str, np.ndarray]:
    """Each Role's Main_Ranking Eligible_Players, after checking that the counts and rules reproduce them.

    With the Recent_Activity_Rule on, the counts and the rules must give the rating checkpoint's
    ``eligibility_report`` exactly and each Role's Eligible_Player set of its ``player_estimates``,
    so that the rule is the only eligibility input the comparison changes. StageError otherwise.
    """
    rules = {
        "min_balls": dict(min_balls), "min_matches": min_matches, "recent_rule": True, "min_years": min_years,
        "min_share": min_share,
    }  # fmt: skip
    rerun = (
        "the contributions, the playing XI or the Config's eligibility settings (data_rules.recent_years, "
        f"{_MIN_YEARS}, {_MIN_SHARE}) changed after the rating stage ran; rerun the Pipeline from the rating stage"
    )
    report = eligibility_report(counts, **rules)
    if list(report.columns) != list(main_report.columns) or (
        report.to_dict("records") != main_report.to_dict("records")
    ):
        raise StageError(
            "the eligibility report recomputed with the Recent_Activity_Rule on differs from the rating "
            f"checkpoint's: {rerun}"
        )
    table = eligibility(counts, **rules)
    saved_role = main_players["role"].astype("string")
    saved_eligible = main_players["eligible"].to_numpy(dtype=bool)
    found: dict[str, np.ndarray] = {}
    for role in ROLES:
        mine = (table["role"] == role).to_numpy(dtype=bool) & table["eligible"].to_numpy(dtype=bool)
        ids = np.sort(table.loc[mine, "player_id"].to_numpy(dtype=np.int64))
        saved = np.sort(main_players.loc[(saved_role == role).to_numpy(dtype=bool) & saved_eligible,
                                         "player_id"].to_numpy(dtype=np.int64))  # fmt: skip
        if not np.array_equal(ids, saved):
            raise StageError(
                f"the {role} Eligible_Players recomputed with the Recent_Activity_Rule on ({len(ids):,}) differ "
                f"from the rating checkpoint's Main_Ranking ({len(saved):,}): {rerun}"
            )
        found[role] = saved
    return found


def _check_draws(
    draws: np.ndarray,
    saved_matches: np.ndarray,
    universe: pd.DataFrame,
    positions: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> None:
    """StageError unless the saved draws are the Main_Ranking's resamples of the current resampled matches.

    ``resample_matches`` must be the ``match_id`` of every resampled match of ``match_universe``,
    in order, and the draws the (D16, N) array that ``draw_resamples`` gives for the Config's
    ``seeds.bootstrap`` (Requirement 25, criterion 12: the same seed and the same matches drawn).
    """
    rerun = "rerun the Pipeline from the bootstrap stage"
    match_ids = universe["match_id"].to_numpy(dtype=np.int64)[positions]
    if saved_matches.shape != match_ids.shape or not np.array_equal(saved_matches, match_ids):
        raise StageError(
            f"the bootstrap checkpoint's {RESAMPLE_MATCHES} ({saved_matches.shape[0]:,} matches) are not the "
            f"{len(match_ids):,} resampled matches of the rating checkpoint's match_universe; {rerun}"
        )
    if draws.ndim != 2 or draws.dtype != np.int16 or draws.shape[1] != len(positions):
        raise StageError(
            f"the bootstrap checkpoint's {RESAMPLE_DRAWS} has shape {draws.shape} and dtype {draws.dtype}, not (B, "
            f"{len(positions):,}) int16 draw positions; {rerun}"
        )
    if draws.shape[0] != n_boot:
        raise StageError(
            f"the bootstrap checkpoint holds {draws.shape[0]:,} Bootstrap_Resamples, but the Config's {_D16} is "
            f"{n_boot!r}; {rerun}"
        )
    if not np.array_equal(draws, draw_resamples(len(positions), n_boot, seed)):
        raise StageError(
            f"the bootstrap checkpoint's {RESAMPLE_DRAWS} are not the draws of the Config's {_SEED} = {seed!r}, so the "
            f"Main_Ranking's bootstrap used another seed; {rerun}"
        )


# ======================================================================== the stage

def _provisional_names(deliveries: pd.DataFrame) -> dict[str, pd.Series]:
    """Per Role, the most frequent ``striker_name`` or ``bowler_name`` of each player id (ties: first by name).

    The provisional names of the rating stage's files, for the Run_Log only; the List_Builder sets
    the names of the Comparison_Lists.
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


def _switch(value: Any, what: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{what} must be True or False; got {value!r}")
    return bool(value)


def run_stage(ctx: StageContext) -> None:
    """The ``activity_comparison`` Pipeline_Stage (module docstring, "The ``activity_comparison`` Pipeline_Stage").

    Saves only into its own checkpoint, under ``outputs/final/comparison/_stage/``, and logs to the
    Run_Log. Raises StageError (exit code 5) when the checkpoints or the Config no longer give the
    Main_Ranking, or a checkpoint file is missing or changed; ValueError or TypeError for tables
    that do not fit together or a Config value of the wrong type.
    """
    started = time.perf_counter()
    settings, store, log = ctx.settings, ctx.store, ctx.log
    if not _switch(settings.get(_RECENT_RULE), _RECENT_RULE):
        store.save_json(
            STAGE_NAME, STATUS_JSON, {"applies": False, "recent_activity_rule": False, "message": _NOT_APPLICABLE}
        )
        log.info(f"Activity_Comparison_Ranking: {_NOT_APPLICABLE}")
        return
    pct, min_matches = settings.get(_D12), settings.get(_D18_DEFAULT)
    n_boot, seed = settings.get(_D16), settings.get(_SEED)
    rules = {"min_years": settings.get(_MIN_YEARS), "min_share": settings.get(_MIN_SHARE)}

    # 1. The Main_Ranking's inputs and settings, from the rating checkpoint (Requirement 25, criterion 12).
    universe = store.load_table(_RATING_STAGE, rating.MATCH_UNIVERSE_TABLE)
    contrib = Contributions.from_frame(
        store.load_table(_RATING_STAGE, rating.CONTRIBUTIONS_TABLE), universe["match_id"]
    )
    main_players = store.load_table(_RATING_STAGE, rating.PLAYER_ESTIMATES_TABLE)
    main_report = store.load_table(_RATING_STAGE, rating.ELIGIBILITY_REPORT_TABLE)
    main_summary = store.load_table(_RATING_STAGE, rating.ROLE_SUMMARY_TABLE)
    min_balls = _main_minimums(main_report, main_summary, min_matches=min_matches, pct=pct)

    # 2. The eligibility counts, as the rating stage computes them; with the rule on they give the Main_Ranking's sets.
    counts = rating.eligibility_counts(
        contrib, store.load_table(_LOAD_STAGE, "playing_eleven"), store.load_table(_LOAD_STAGE, "matches"),
        recent_years=settings.get(_RECENT_YEARS),
    )  # fmt: skip
    main_eligible = _check_main_eligibility(
        counts, main_report, main_players, min_balls=min_balls, min_matches=min_matches, **rules
    )

    # 3. The Main_Ranking's resample draws, reused as saved.
    positions = np.flatnonzero(universe["resampled"].to_numpy(dtype=bool))
    draws = store.load_array(BOOTSTRAP_STAGE, RESAMPLE_DRAWS)
    saved_matches = store.load_array(BOOTSTRAP_STAGE, RESAMPLE_MATCHES)
    _check_draws(draws, saved_matches, universe, positions, n_boot=n_boot, seed=seed)

    # 4. The Activity_Comparison_Ranking and its bootstrap (criteria 11 and 12); no model is fitted.
    result = activity_comparison(
        contrib, counts, draws, positions, main_eligible=main_eligible, min_balls=min_balls, min_matches=min_matches,
        replacement_pct=pct, **rules,
    )  # fmt: skip

    # 5. The checkpoint, inside outputs/final/comparison/ (criterion 15).
    tables = {
        PLAYER_ESTIMATES_TABLE: result.players,
        ROLE_SUMMARY_TABLE: result.summary,
        ELIGIBILITY_REPORT_TABLE: result.report,
        ELIGIBLE_PLAYERS_TABLE: result.eligible_players,
        BOOTSTRAP_TABLE: result.bootstrap,
        WARNINGS_TABLE: result.warnings,
        BOOTSTRAP_WARNINGS_TABLE: result.bootstrap_warnings,
    }
    for name in CHECKPOINT_TABLES:
        store.save_table(STAGE_NAME, name, tables[name])
    store.save_json(STAGE_NAME, STATUS_JSON, {
        "applies": True,
        "recent_activity_rule": True,
        "message": (
            "the Activity_Comparison_Ranking of each Role and its bootstrap, with the Recent_Activity_Rule off and "
            "every other input and setting equal to the Main_Ranking's (Requirement 25, criteria 11 and 12)"
        ),
        "min_pressure_balls": {role: int(min_balls[role]) for role in ROLES},
        "min_pressure_matches": int(min_matches),
        "replacement_pct": float(pct),
        "n_boot": int(draws.shape[0]),
        "n_resampled_matches": int(len(positions)),
        "bootstrap_seed": int(seed),
        "resample_draws_sha256": hashlib.sha256(np.ascontiguousarray(draws).tobytes()).hexdigest(),
        "n_eligible": {role: len(result.rankings[role]) for role in ROLES},
        "n_eligible_main": {role: int(len(main_eligible[role])) for role in ROLES},
    })  # fmt: skip

    # 6. The data-derived constants of the comparison, under this stage's own names.
    for role in ROLES:
        values = result.rankings[role].summary()
        values["min_pressure_balls"] = int(min_balls[role])
        values["n_eligible"] = len(result.rankings[role])
        for quantity in CONSTANT_QUANTITIES:
            ctx.constants.record(f"{STAGE_NAME}.{quantity}.{role}", values[quantity], _COMPONENT)

    names = _provisional_names(store.load_table(_LOAD_STAGE, "ball_by_ball"))
    _log_stage(
        ctx, result=result, main_players=main_players, main_summary=main_summary, min_balls=min_balls,
        min_matches=min_matches, pct=pct, seed=seed, n_matches=len(universe), n_resampled=len(positions),
        names=names, seconds=time.perf_counter() - started,
    )  # fmt: skip


def _role_values(summary: Mapping[str, Any]) -> str:
    return ", ".join(f"{name} {float(summary[name]):.6g}" for name in ROLE_HYPER_FIELDS)


def _log_stage(
    ctx: StageContext,
    *,
    result: ActivityResult,
    main_players: pd.DataFrame,
    main_summary: pd.DataFrame,
    min_balls: Mapping[str, int],
    min_matches: int,
    pct: Any,
    seed: int,
    n_matches: int,
    n_resampled: int,
    names: Mapping[str, pd.Series],
    seconds: float,
) -> None:
    log = ctx.log
    n_boot = next(iter(result.bootstraps.values())).n_boot
    log.info(
        f"{_LOG_PREFIX} inputs (Requirement 25, criteria 11 and 12): the rating checkpoint's contributions over "
        f"{n_matches:,} matches, so every Out_Of_Fold_Prediction, PI, Pressure_Ball flag, NBV, Dismissal_Charge and "
        "Delivery_Weight is the Main_Ranking's; the Main_Ranking's minimum Pressure_Ball counts "
        + ", ".join(f"{role} {int(min_balls[role])}" for role in ROLES)
        + f", the D18 default minimum of {int(min_matches)} Pressure_Matches, the 2-Calendar_Year and share rules and "
        f"the D12 percentile {float(pct):g}; the Main_Ranking's {n_boot:,} saved Bootstrap_Resamples of the "
        f"{n_resampled:,} resampled matches ({_SEED} = {seed}, checked); no model is fitted. Check passed: one copy of "
        "every resampled match reproduces both comparison rankings bit for bit with their own Role-level values"
    )
    main_rows = _role_rows(main_summary, rating.ROLE_SUMMARY_TABLE)
    saved_role = main_players["role"].astype("string")
    for role in ROLES:
        ranking, summary = result.rankings[role], result.rankings[role].summary()
        report = result.report.loc[result.report["role"] == role].iloc[0]
        eligible = result.eligible_players.loc[result.eligible_players["role"] == role]
        added = eligible.loc[~eligible["main_eligible"].to_numpy(dtype=bool), "player_id"].to_numpy(dtype=np.int64)
        fails = ", ".join(f"{rule} {int(report[f'fail_{rule}']):,}" for rule in ELIGIBILITY_RULES)
        log.info(
            f"{_LOG_PREFIX}, {role}: {len(ranking):,} Eligible_Players of {int(report['n_players']):,} players with a "
            f"Delivery, against {len(ranking) - len(added):,} in the Main_Ranking ({len(added):,} added, none removed; "
            f"failing: {fails}). Its own Role-level values, held in each of its resamples: {_role_values(summary)}; "
            f"the Main_Ranking's: {_role_values(main_rows[role])}. {int(summary['n_ranked']):,} ranked; "
            f"{int(summary['n_fallback']):,} on the Clutch_Level fallback"
        )
        boot = result.bootstrap.loc[result.bootstrap["role"] == role].set_index("player_id")
        mains = main_players.loc[(saved_role == role).to_numpy(dtype=bool)].set_index("player_id")
        main_rank = mains["rank"].astype("Float64").to_numpy(dtype=np.float64, na_value=np.nan)
        main_rank_of = dict(zip(mains.index.to_numpy(dtype=np.int64).tolist(), main_rank.tolist()))
        order = np.argsort(np.where(ranking.ranked, ranking.rank, np.inf), kind="stable")[:LIST_SIZE]
        top_ids = [int(pid) for pid in ranking.player_id[order]]
        position = {int(pid): j for j, pid in enumerate(ranking.player_id)}

        def label(pid: int) -> str:
            return f"{names[role].get(pid, '?')} ({pid})"

        def main_text(pid: int) -> str:
            value = main_rank_of.get(pid, np.nan)
            return "not eligible" if np.isnan(value) else str(int(value))

        def top_text(pid: int) -> str:
            row = boot.loc[pid]
            return (
                f"{int(ranking.rank[position[pid]])} {label(pid)} S {ranking.final_score[position[pid]]:+.3f} "
                f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}] P_Top10 {row['p_top10']:.3f}, "
                f"Main rank {main_text(pid)}"
            )

        log.info(f"{_LOG_PREFIX} top {len(top_ids)}, {role} (names provisional): " + "; ".join(map(top_text, top_ids)))
        in_main = np.isfinite(main_rank) & (main_rank <= LIST_SIZE)
        main_order = np.argsort(main_rank[in_main], kind="stable")
        main_top_ids = [int(pid) for pid in mains.index.to_numpy(dtype=np.int64)[in_main][main_order]]
        in_main_top, in_top, only_by_rule = set(main_top_ids), set(top_ids), set(added.tolist())
        entering = [pid for pid in top_ids if pid not in in_main_top]
        leaving = [pid for pid in main_top_ids if pid not in in_top]

        def entering_text(pid: int) -> str:
            reason = (
                "not a Main_Ranking Eligible_Player: it fails only the Recent_Activity_Rule"
                if pid in only_by_rule else f"Main rank {main_text(pid)}"
            )  # fmt: skip
            return f"{label(pid)} comparison rank {int(ranking.rank[position[pid]])} ({reason})"

        def leaving_text(pid: int) -> str:
            return f"{label(pid)} Main rank {main_text(pid)}, comparison rank {int(ranking.rank[position[pid]])}"

        log.info(
            f"{_LOG_PREFIX} against the Main top {LIST_SIZE}, {role}: {len(in_top & in_main_top)} of {LIST_SIZE} "
            f"in both. Entering: {'; '.join(map(entering_text, entering)) or 'none'}. "
            f"Leaving: {'; '.join(map(leaving_text, leaving)) or 'none'}"
        )
    for row in result.warnings.to_dict("records"):
        log.warning(f"{_LOG_PREFIX}: {row['message']}")
    for row in result.bootstrap_warnings.to_dict("records"):
        estimate = f" ({row['estimate']})" if isinstance(row["estimate"], str) else ""
        log.warning(
            f"{_LOG_PREFIX} bootstrap warnings, {row['role']} {row['kind']}{estimate}: in "
            f"{int(row['n_resamples']):,} of {n_boot:,} resamples ({int(row['n_warnings']):,} warnings); the first, "
            f"in resample {int(row['first_resample'])}: {row['first_message']}"
        )
    ranking_seconds, bootstrap_seconds = result.timings["ranking"], result.timings["bootstrap"]
    folder = _shown_path(ctx.store.stage_dir(STAGE_NAME), ctx.root)
    log.info(
        f"{_LOG_PREFIX} timing: rankings {ranking_seconds * 1000:.0f} ms; {n_boot:,} resamples "
        f"{bootstrap_seconds:.1f} s ({bootstrap_seconds / n_boot * 1000:.2f} ms per resample, both Roles); stage "
        f"{seconds:.1f} s. Saved {', '.join(CHECKPOINT_TABLES)} and {STATUS_JSON} in {folder}/; no other file was "
        "written"
    )


def _shown_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return str(path)
