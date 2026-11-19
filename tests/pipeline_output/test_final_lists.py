"""Pipeline-output tests of the final lists (Requirement 25, criteria 4 and 15; Requirement 26, criteria 11 and 12).

**Validates: Requirements 25.4, 25.15, 26.11, 26.12**

Design sections "List_Builder (``lists.py``, ``charts.py``)" and Output files. The tests read the files that the
``final_lists`` stage wrote to ``outputs/final/`` and the checkpoints of a completed Run. Every CSV file is read with
``keep_default_na=False, na_values=[""]``, so that only an empty field is missing and a name such as "NA" stays text.
Checkpoint files are read through the CheckpointStore, which checks each one against its stage's manifest. Each check
is a function that returns a list of problems, and each test asserts that the list is empty.

One Run
    The ``final_lists`` checkpoint holds the JSON value ``written_files``: ``files``, the SHA-256 of every file that
    the stage wrote outside ``outputs/final/comparison/``, and ``comparison_guard``, the snapshot of the guarded
    files (below). Before a test compares a file with another file or with a checkpoint, it checks that the file still
    has the SHA-256 that ``files`` records, and that the ``final_lists`` checkpoint was built from the checkpoints now
    on disk (each manifest records the manifest hash of every stage that the stage read). Otherwise the test fails,
    because the files and checkpoints would not come from one Run.

Requirement 25, criterion 4: ``test_main_list_file_has_10_rows_ranks_1_to_10_once_and_10_players``
    One case per Role: ``top10_batters.csv`` or ``top10_bowlers.csv`` has exactly 10 rows, uses each rank from 1 to
    10 exactly once and holds 10 distinct ``player_id`` values. Only the file is read.
Requirement 25, criterion 4: ``test_each_player_name_is_a_candidate_name_of_one_player_id_unless_unresolved``
    One case per Role. The Candidate_Names come from ``lists.candidate_names`` on the ``load`` checkpoint's
    ``mapping`` table (``competition_team_player_mapping.csv``), and the entries from ``lists.parse_checklist`` on
    ``unstop_entry_checklist.md``. Each row of the Main_List file needs exactly one checklist entry of its Role and
    rank, with the row's ``player_id`` and ``player_name``. Unless the checklist marks that entry unresolved, the
    ``player_name`` must be a Candidate_Name of exactly one ``player_id``, the row's own. The mark exempts an entry,
    so the test also checks that the checklist marks an entry unresolved exactly when it is an Unresolved_Entry
    (glossary: its ``player_id`` has zero Candidate_Names or more than one, or its ``player_name`` is a
    Candidate_Name of more than one ``player_id``).
Requirement 26, criterion 11: ``test_evidence_table_holds_the_10_listed_players_in_rank_order``
    One case per evidence table (``evidence.EVIDENCE_TABLES``): the traditional-stats, rating-evidence and
    Pressure_Ball mix tables of each Role. Read in file order, the table's ``rank``, ``player_id`` and ``player_name``
    columns equal those of the Role's Main_List file sorted by rank: 10 rows, ranks 1 to 10 in order, and the same
    player, with the same name, at each rank.
Requirement 26, criterion 12: ``test_rating_evidence_values_equal_the_rating_and_bootstrap_values``
    One case per Role. The rating-evidence file writes each float in its shortest round-trip form. The test reads each
    value cell as text and converts it with Python's ``float``, which rounds correctly, so a value equals its source
    only when it is the same float64. Each row's player must be the Role's Eligible_Player at the row's rank in the
    ``rating`` checkpoint's ``player_estimates`` table (the Main_Ranking), and the ``bootstrap`` checkpoint's row of
    the player must resample that ranking: its ``main_rank`` and ``main_final_score`` are the player's rank and
    Final_Score. Then
    ``final_score``, ``clutch``, ``usual``, ``uplift``, ``shrunk_clutch``, ``shrunk_uplift``, ``hit_rate``,
    ``worst_quartile`` and ``n_pressure_matches`` must equal the ``player_estimates`` values, and ``ci_low``,
    ``ci_high`` and ``p_top10`` the ``bootstrap`` values. Two missing values count as equal. The file may have no
    other column, so every value it holds is checked.
Requirement 25, criterion 15: ``test_comparison_step_changed_no_protected_file``
    The ``final_lists`` stage writes the Main_Lists and the checklist, takes a snapshot of the SHA-256 of every file
    directly in ``outputs/final/``, ``outputs/rating/`` and ``outputs/robustness/``, and runs the activity comparison
    step, which writes the Comparison_Lists and the activity comparison tables to ``outputs/final/comparison/``. It
    then takes a second snapshot and stops the Run with a StageError when it differs from the first. The checkpoint
    saves the first snapshot, so a complete checkpoint records a step after which both snapshots were equal. The
    protected files are the Main_List files, the checklist, every file of ``outputs/rating/`` (the Requirement 18,
    criterion 15 tables among them) and each ``outputs/robustness/bootstrap_*.csv`` (the Requirement 19, criterion 5
    table among them); the stage writes none of them after the step. The test checks that the snapshot holds the
    Main_List files and the checklist with the SHA-256 that ``files`` records for them, that every protected file now
    on disk is in the snapshot, and that every protected file of the snapshot still exists with the same SHA-256.
Requirement 25, criterion 15: ``test_activity_comparison_checkpoint_lies_under_outputs_final_comparison``
    The ``activity_comparison`` stage, which computes the Activity_Comparison_Ranking, runs before the ``final_lists``
    stage takes its snapshot, so the snapshot cannot show what it did. It saves its results only through the
    CheckpointStore. The test checks that its checkpoint folder, the one the ``final_lists`` checkpoint was built
    from, lies under ``outputs/final/comparison/`` (in ``_stage/``), and that every file its manifest lists lies in
    that folder.

Skips
    A test is skipped, with the missing files named, while a file it reads is absent (``requires_outputs``), and with
    the reason while a checkpoint it reads is not complete.
"""

from __future__ import annotations

import fnmatch
import hashlib
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import pandas as pd
import pytest

from superr import activity, evidence, lists, rating, robustness
from superr import io as loader
from superr.pipeline import OUTPUTS_FOLDER, checkpoint_layout
from superr.store import MANIFEST_NAME, CheckpointStore
from superr.value import ROLES

if TYPE_CHECKING:
    from superr.settings import Settings

# ------------------------------------------------------------------------------------- constants

#: Players per list (Requirement 25, criterion 1).
LIST_SIZE = rating.LIST_SIZE

#: The ``load`` table of ``competition_team_player_mapping.csv`` (``io.FILE_NAMES``): the Candidate_Names.
MAPPING_TABLE = "mapping"

#: Paths under outputs/ of the List_Builder's files.
MAIN_LISTS: dict[str, str] = {role: f"{lists.FINAL_FOLDER}/{lists.MAIN_LIST_FILES[role]}" for role in ROLES}
CHECKLIST = f"{lists.FINAL_FOLDER}/{lists.CHECKLIST_FILE}"
EVIDENCE_FILES: dict[str, str] = {
    name: f"{lists.FINAL_FOLDER}/{evidence.EVIDENCE_FILES[name]}" for name in evidence.EVIDENCE_TABLES
}
#: The Role of each evidence table, and each Role's rating-evidence table.
EVIDENCE_ROLES: dict[str, str] = {
    evidence.table_name(kind, role): role for kind in evidence.EVIDENCE_KINDS for role in ROLES
}
RATING_EVIDENCE: dict[str, str] = {role: evidence.table_name(evidence.RATING_EVIDENCE, role) for role in ROLES}

#: The checkpoint tables behind the rating evidence: the Main_Ranking (rating) and its bootstrap.
RANKING = rating.PLAYER_ESTIMATES_TABLE
BOOTSTRAP = robustness.BOOTSTRAP_TABLE
#: Each value column of a rating-evidence table (Requirement 26, criterion 4; design, Output files) and the table that
#: holds the same value under the same name.
VALUE_SOURCES: dict[str, str] = {
    "final_score": RANKING, "clutch": RANKING, "usual": RANKING, "uplift": RANKING, "shrunk_clutch": RANKING,
    "shrunk_uplift": RANKING, "ci_low": BOOTSTRAP, "ci_high": BOOTSTRAP, "p_top10": BOOTSTRAP, "hit_rate": RANKING,
    "worst_quartile": RANKING, "n_pressure_matches": RANKING,
}  # fmt: skip
#: The columns of each source table that place a player in the Main_Ranking.
SOURCE_KEYS: dict[str, tuple[str, ...]] = {
    RANKING: ("role", "player_id", "eligible", "rank"),
    BOOTSTRAP: ("role", "player_id", "main_rank", "main_final_score"),
}

#: The protected files of criterion 15 that the final_lists stage writes before its comparison step, and the
#: protected files of outputs/robustness/. Every file of outputs/rating/ is protected as well.
LISTS_AND_CHECKLIST: tuple[str, ...] = (*MAIN_LISTS.values(), CHECKLIST)
BOOTSTRAP_PATTERN = "bootstrap_*.csv"
#: The folders under outputs/ that hold the protected files.
PROTECTED_FOLDERS: tuple[str, ...] = (lists.FINAL_FOLDER, rating.RATING_FOLDER, robustness.ROBUSTNESS_FOLDER)

RUN_HINT = r"run `.venv\Scripts\python run_pipeline.py` first"
_SHOWN = 25  # problem lines in one failure message

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / OUTPUTS_FOLDER
# Only for the checkpoint paths of the requires_outputs marks; the tests read through `checkpoints`.
_PATHS = CheckpointStore(OUTPUTS, checkpoint_layout(ROOT), config_sha256=hashlib.sha256(b"").hexdigest())


def _checkpoint_files(stage: str, *names: str) -> tuple[str, ...]:
    """The manifest and the files ``names`` of a stage's checkpoint, relative to outputs/."""
    folder = _PATHS.stage_dir(stage).relative_to(OUTPUTS)
    return tuple((folder / name).as_posix() for name in (MANIFEST_NAME, *names))


FINAL_MANIFEST = _checkpoint_files(lists.STAGE_NAME)
WRITTEN = _checkpoint_files(lists.STAGE_NAME, f"{lists.WRITTEN_JSON}.json")
LOAD = _checkpoint_files(loader.STAGE_NAME, f"{MAPPING_TABLE}.parquet")
SOURCES = (
    *_checkpoint_files(rating.STAGE_NAME, f"{RANKING}.parquet"),
    *_checkpoint_files(robustness.BOOTSTRAP_STAGE, f"{BOOTSTRAP}.parquet"),
)
ACTIVITY = _checkpoint_files(activity.STAGE_NAME)


def _case(value: str, *paths: str) -> Any:
    """One parameter of a test, skipped unless every path under outputs/ exists."""
    return pytest.param(value, id=value, marks=pytest.mark.requires_outputs(*paths))


LIST_CASES = [_case(role, MAIN_LISTS[role]) for role in ROLES]
NAME_CASES = [_case(role, MAIN_LISTS[role], CHECKLIST, *WRITTEN, *LOAD) for role in ROLES]
EVIDENCE_CASES = [
    _case(name, EVIDENCE_FILES[name], MAIN_LISTS[EVIDENCE_ROLES[name]], *WRITTEN) for name in evidence.EVIDENCE_TABLES
]
VALUE_CASES = [_case(role, EVIDENCE_FILES[RATING_EVIDENCE[role]], *WRITTEN, *SOURCES) for role in ROLES]

# ------------------------------------------------------------------------------- shared helpers


class Written(NamedTuple):
    """The ``written_files`` value of the final_lists checkpoint."""

    files: dict[str, str]  # path under outputs/ -> SHA-256 of each file the stage wrote outside final/comparison/
    guard: dict[str, str]  # path under outputs/ -> SHA-256: the snapshot of the guarded files (comparison_guard)


class NameIndex(NamedTuple):
    """The Candidate_Names of the mapping table, both ways."""

    names_of: dict[int, tuple[str, ...]]  # player_id -> its Candidate_Names
    holders_of: dict[str, tuple[int, ...]]  # Candidate_Name -> the player_ids that have it, ascending


#: One Role's source rows: for RANKING and BOOTSTRAP, each player's row as a dict, by player_id.
RoleSources = Mapping[str, Mapping[int, Mapping[str, Any]]]
#: (rank, player_id, player_name) of one list or table row; player_name is None where the cell is empty.
Key = tuple[int, int, "str | None"]


@pytest.fixture(scope="module")
def checkpoints(settings: Settings, outputs_dir: Path, workspace_root: Path) -> CheckpointStore:
    """The Run's checkpoints, read through the store, which checks each file against its manifest."""
    return CheckpointStore(outputs_dir, checkpoint_layout(workspace_root), config_sha256=settings.effective_sha256())


def _require_complete(store: CheckpointStore, *stages: str) -> None:
    """Skip unless every stage's checkpoint is complete: a valid manifest, and every file it lists exists."""
    for stage in stages:
        reason = store.incomplete_reason(stage)
        if reason is not None:
            pytest.skip(f"the {stage} checkpoint is not complete ({reason}); {RUN_HINT}")


def _require_built_on(store: CheckpointStore, stage: str, *reads: str) -> None:
    """Fail unless the checkpoint of ``stage`` was built from the checkpoints of ``reads`` now on disk."""
    recorded = store.manifest(stage)["inputs"]
    stale = [read for read in reads if recorded.get(read) != store.manifest_sha256(read)]
    if stale:
        pytest.fail(
            f"the {stage} checkpoint was not built from the {', '.join(stale)} checkpoint(s) now on disk: its "
            f"{MANIFEST_NAME} records another manifest hash, or none, for them, so they do not come from one Run; "
            f"run `.venv\\Scripts\\python run_pipeline.py --from-stage {stage}`"
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def written(checkpoints: CheckpointStore) -> Written:
    """The files that the final_lists stage wrote, and its comparison guard snapshot (``written_files``)."""
    _require_complete(checkpoints, lists.STAGE_NAME)
    value = checkpoints.load_json(lists.STAGE_NAME, lists.WRITTEN_JSON)
    parts: dict[str, dict[str, str]] = {}
    for key in ("files", "comparison_guard"):
        part = value.get(key) if isinstance(value, dict) else None
        if not isinstance(part, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in part.items()):
            pytest.fail(
                f"the {lists.STAGE_NAME} checkpoint's {lists.WRITTEN_JSON} value has no {key!r} mapping of paths "
                "under outputs/ to SHA-256 values"
            )
        parts[key] = dict(part)
    return Written(files=parts["files"], guard=parts["comparison_guard"])


def _require_written(written: Written, outputs_dir: Path, *paths: str) -> None:
    """Fail unless each file (a path under outputs/) still has the SHA-256 that the final_lists stage recorded."""
    stale = [
        f"outputs/{path} ({'not recorded' if path not in written.files else 'changed since the stage wrote it'})"
        for path in paths
        if written.files.get(path) != _sha256(outputs_dir / path)
    ]
    if stale:
        pytest.fail(
            f"{', '.join(stale)}: not what the {lists.STAGE_NAME} stage wrote (its {lists.WRITTEN_JSON} value records "
            "the SHA-256 of each file it wrote), so the file cannot be checked against that stage's other files and "
            f"checkpoints; run `.venv\\Scripts\\python run_pipeline.py --from-stage {lists.STAGE_NAME}`"
        )


def _read_csv(path: Path, text_columns: Iterable[str] = ()) -> pd.DataFrame:
    """A List_Builder CSV file in which only an empty field is missing, so a name such as "NA" stays text.

    ``player_name`` and the ``text_columns`` are read as text, with NaN for an empty field.
    """
    dtype = {column: str for column in ("player_name", *text_columns)}
    return pd.read_csv(path, keep_default_na=False, na_values=[""], dtype=dtype, encoding="utf-8")


def _whole(table: pd.DataFrame, column: str) -> np.ndarray | None:
    """``column`` as int64; None when the table lacks it or a cell is blank or not a whole number.

    With only empty fields missing, a column of whole numbers reads with an integer dtype: a blank or a decimal makes
    it a float column, and text a string column.
    """
    if column not in table.columns or not pd.api.types.is_integer_dtype(table[column].dtype):
        return None
    return table[column].to_numpy(dtype=np.int64)


def _keys(table: pd.DataFrame) -> list[Key] | None:
    """(rank, player_id, player_name) of each row, in file order; None when the columns cannot be read so."""
    ranks, ids = _whole(table, "rank"), _whole(table, "player_id")
    if ranks is None or ids is None or "player_name" not in table.columns:
        return None
    names = [name if isinstance(name, str) else None for name in table["player_name"].tolist()]
    return list(zip(ranks.tolist(), ids.tolist(), names))


def _missing(value: Any) -> bool:
    return value is None or value is pd.NA or (isinstance(value, (float, np.floating)) and math.isnan(value))


def _same(written_value: Any, source_value: Any) -> bool:
    """True when both values are missing, or both give the same float64 (a text value through Python's ``float``)."""
    if _missing(written_value) or _missing(source_value):
        return _missing(written_value) and _missing(source_value)
    try:
        return float(written_value) == float(source_value)
    except (TypeError, ValueError):
        return False


def _whole_value(value: Any) -> int | None:
    return None if _missing(value) else int(value)


def _text(value: Any) -> str:
    """A value for a message: ``missing``, or its Python repr."""
    if _missing(value):
        return "missing"
    return repr(value.item() if isinstance(value, np.generic) else value)


def _failure(header: str, problems: list[str]) -> str:
    lines = [f"  - {line}" for line in problems[:_SHOWN]]
    if len(problems) > _SHOWN:
        lines.append(f"  - ... and {len(problems) - _SHOWN} more")
    return "\n".join([header, *lines])


# ------------------------------------------------------------------------------------------ checks


def _list_problems(table: pd.DataFrame) -> list[str]:
    """Requirement 25, criterion 4: 10 rows, each rank from 1 to 10 exactly once, and 10 distinct player_ids."""
    problems = []
    if len(table) != LIST_SIZE:
        problems.append(f"it has {len(table)} rows, not {LIST_SIZE}")
    ranks = _whole(table, "rank")
    if ranks is None:
        problems.append("its rank column is absent or holds a blank or a value that is not a whole number")
    else:
        counts = Counter(ranks.tolist())
        repeated = sorted(rank for rank, n in counts.items() if n > 1)
        absent = [rank for rank in range(1, LIST_SIZE + 1) if rank not in counts]
        outside = sorted(rank for rank in counts if not 1 <= rank <= LIST_SIZE)
        if repeated:
            problems.append(f"rank(s) {repeated} appear more than once")
        if absent:
            problems.append(f"rank(s) {absent} do not appear")
        if outside:
            problems.append(f"rank(s) {outside} lie outside 1-{LIST_SIZE}")
    ids = _whole(table, "player_id")
    if ids is None:
        problems.append("its player_id column is absent or holds a blank or a value that is not a whole number")
    else:
        counts = Counter(ids.tolist())
        if len(counts) != LIST_SIZE:
            repeated = sorted(pid for pid, n in counts.items() if n > 1)
            problems.append(
                f"it holds {len(counts)} distinct player_ids, not {LIST_SIZE}"
                + (f" (repeated: {repeated})" if repeated else "")
            )
    return problems


def _name_index(names: pd.DataFrame) -> NameIndex:
    """Both directions of a ``lists.candidate_names`` table, which has one row per player_id and Candidate_Name."""
    names_of: dict[int, list[str]] = {}
    holders_of: dict[str, list[int]] = {}
    for pid, name in zip(names["player_id"].to_numpy(dtype=np.int64).tolist(), names["player_name"].tolist()):
        names_of.setdefault(pid, []).append(str(name))
        holders_of.setdefault(str(name), []).append(pid)
    return NameIndex(
        names_of={pid: tuple(found) for pid, found in names_of.items()},
        holders_of={name: tuple(sorted(found)) for name, found in holders_of.items()},
    )


def _name_problems(role: str, table: pd.DataFrame, checklist: pd.DataFrame, index: NameIndex) -> list[str]:
    """Requirement 25, criterion 4: each player_name is a Candidate_Name of exactly one player_id, the row's own,
    unless the checklist marks the entry unresolved; and the checklist marks exactly the Unresolved_Entries."""
    keys = _keys(table)
    if keys is None:
        return ["the file has no rank and player_id columns of whole numbers and player_name column to match"]
    entries: dict[int, list[dict[str, Any]]] = {}
    for entry in checklist.loc[(checklist["role"] == role).to_numpy(dtype=bool)].to_dict("records"):
        entries.setdefault(int(entry["rank"]), []).append(entry)
    problems = []
    for rank, pid, name in keys:
        where = f"rank {rank}, {name!r} (player_id {pid})"
        found = entries.get(rank, [])
        if name is None:
            problems.append(f"{where}: the row has no player_name")
            continue
        if len(found) != 1:
            problems.append(f"{where}: the checklist has {len(found)} entries for this Role and rank, not one")
            continue
        entry = found[0]
        if int(entry["player_id"]) != pid or entry["player_name"] != name:
            problems.append(
                f"{where}: the checklist entry at this rank shows {entry['player_name']!r} "
                f"(player_id {int(entry['player_id'])})"
            )
            continue
        own, holders = index.names_of.get(pid, ()), index.holders_of.get(name, ())
        is_unresolved = len(own) != 1 or len(holders) > 1  # glossary Unresolved_Entry
        facts = (
            f"player_id {pid} has {len(own)} Candidate_Name(s) {list(own)}, and {name!r} is a Candidate_Name of "
            f"{len(holders)} player_id(s) {list(holders)}"
        )
        if not bool(entry["resolved"]):
            if not is_unresolved:
                problems.append(
                    f"{where}: the checklist marks the entry unresolved, but it is not an Unresolved_Entry ({facts})"
                )
        elif holders != (pid,):
            problems.append(
                f"{where}: the name is not a Candidate_Name of exactly one player_id, the row's, and the checklist "
                f"does not mark the entry unresolved ({facts})"
            )
        elif is_unresolved:
            problems.append(f"{where}: the checklist marks the entry resolved, but it is an Unresolved_Entry ({facts})")
    return problems


def _order_problems(table: pd.DataFrame, listed: list[Key]) -> list[str]:
    """Requirement 26, criterion 11: the table's (rank, player_id, player_name) rows are ``listed``, in order."""
    found = _keys(table)
    if found is None:
        return ["it has no rank and player_id columns of whole numbers and player_name column"]
    problems = []
    if len(found) != len(listed):
        problems.append(f"it has {len(found)} rows, not the {len(listed)} of the Main_List")
    ids = {pid for _, pid, _ in listed}
    extra = [pid for _, pid, _ in found if pid not in ids]
    absent = sorted(ids - {pid for _, pid, _ in found})
    if extra:
        problems.append(f"it holds player_id(s) {extra}, which are not in the Main_List")
    if absent:
        problems.append(f"it lacks the listed player_id(s) {absent}")
    for position, (row, want) in enumerate(zip(found, listed), start=1):
        if row != want:
            problems.append(
                f"row {position} holds rank {row[0]}, player_id {row[1]}, {row[2]!r}; the Main_List has rank "
                f"{want[0]}, player_id {want[1]}, {want[2]!r}"
            )
    return problems


def _value_problems(role: str, table: pd.DataFrame, sources: RoleSources) -> list[str]:
    """Requirement 26, criterion 12: every value of a Role's rating-evidence table equals the value of the
    Main_Ranking (``player_estimates``) or of its bootstrap (``bootstrap``) for the same player and Role.

    ``table``: the file read with its value columns as text (``_read_csv``), so ``_same`` parses each cell with
    Python's ``float``.
    """
    problems = []
    columns = (*evidence.KEY_COLUMNS, *VALUE_SOURCES)
    absent = [column for column in columns if column not in table.columns]
    extra = [str(column) for column in table.columns if column not in columns]
    if absent:
        problems.append(f"the file lacks the column(s) {', '.join(absent)}")
    if extra:
        problems.append(f"the file has the column(s) {', '.join(extra)}, which neither source table holds")
    ranks, ids = _whole(table, "rank"), _whole(table, "player_id")
    if ranks is None or ids is None:
        problems.append("its rank or player_id column is absent or holds a blank or a value that is not a whole number")
        return problems
    for position, (rank, pid) in enumerate(zip(ranks.tolist(), ids.tolist())):
        where = f"rank {rank} (player_id {pid})"
        player, resample = sources[RANKING].get(pid), sources[BOOTSTRAP].get(pid)
        if player is None:
            problems.append(f"{where}: {RANKING} has no {role} row for the player")
        elif not bool(player["eligible"]) or _whole_value(player["rank"]) != rank:
            problems.append(
                f"{where}: {RANKING} gives the player eligible {bool(player['eligible'])} and rank "
                f"{_text(player['rank'])}, so the row is not the Main_Ranking's player at this rank"
            )
        if resample is None:
            problems.append(f"{where}: the {BOOTSTRAP} table has no {role} row for the player")
        elif player is not None and (
            _whole_value(resample["main_rank"]) != _whole_value(player["rank"])
            or not _same(resample["main_final_score"], player["final_score"])
        ):
            problems.append(
                f"{where}: the {BOOTSTRAP} row resamples another ranking: main_rank {_text(resample['main_rank'])} "
                f"and main_final_score {_text(resample['main_final_score'])}, where {RANKING} has rank "
                f"{_text(player['rank'])} and final_score {_text(player['final_score'])}"
            )
        for column, source in VALUE_SOURCES.items():
            reference = player if source == RANKING else resample
            if column in table.columns and reference is not None:
                value = table[column].iloc[position]
                if not _same(value, reference[column]):
                    problems.append(
                        f"{where}: {column} is {_text(value)} in the file but {_text(reference[column])} in {source}"
                    )
    return problems


def _is_protected(path: str) -> bool:
    """True for a path under outputs/ that criterion 15 protects and the final_lists stage leaves alone after its
    comparison step: a Main_List file, the checklist, a file of outputs/rating/ or outputs/robustness/bootstrap_*.csv."""
    parts = PurePosixPath(path).parts
    if len(parts) != 2:
        return False
    folder, name = parts
    if folder == lists.FINAL_FOLDER:
        return path in LISTS_AND_CHECKLIST
    if folder == rating.RATING_FOLDER:
        return True
    return folder == robustness.ROBUSTNESS_FOLDER and fnmatch.fnmatchcase(name, BOOTSTRAP_PATTERN)


def _protected_on_disk(outputs_dir: Path) -> dict[str, str]:
    """The SHA-256 of each protected file now under outputs/, by its path under outputs/."""
    found = {}
    for folder in PROTECTED_FOLDERS:
        base = outputs_dir / folder
        if base.is_dir():
            for path in sorted(base.iterdir()):
                relative = f"{folder}/{path.name}"
                if path.is_file() and _is_protected(relative):
                    found[relative] = _sha256(path)
    return found


def _guard_problems(written: Written, current: Mapping[str, str]) -> list[str]:
    """Requirement 25, criterion 15: the protected files kept their SHA-256 through the comparison step and since.

    ``current``: ``_protected_on_disk``. The snapshot must hold the Main_List files and the checklist with the
    SHA-256 of the files the stage wrote before the step; each protected file must be in the snapshot, and each
    protected file of the snapshot must still exist with its SHA-256.
    """
    problems = []
    snapshot = {path: digest for path, digest in written.guard.items() if _is_protected(path)}
    for path in LISTS_AND_CHECKLIST:
        recorded, seen = written.files.get(path), snapshot.get(path)
        if recorded is None:
            problems.append(f"the stage recorded no outputs/{path} among the files it wrote")
        elif seen is not None and seen != recorded:
            problems.append(
                f"the snapshot holds outputs/{path} with SHA-256 {seen[:12]}..., not the {recorded[:12]}... of the "
                "file that the stage wrote before the comparison step"
            )
    for path in sorted({*snapshot, *current, *LISTS_AND_CHECKLIST}):
        seen, now = snapshot.get(path), current.get(path)
        if seen is None:
            problems.append(
                f"outputs/{path} is not in the snapshot, so the guard did not cover it"
                + (" (the file exists now)" if now is not None else "")
            )
        elif now is None:
            problems.append(f"outputs/{path} is in the snapshot but no longer exists")
        elif now != seen:
            problems.append(f"outputs/{path} has SHA-256 {now[:12]}... now, not the {seen[:12]}... of the snapshot")
    return problems


def _checkpoint_problems(folder: Path, listed: Iterable[str], comparison_dir: Path) -> list[str]:
    """Requirement 25, criterion 15: a checkpoint folder under ``comparison_dir`` whose manifest lists only files
    inside it (``listed``: the manifest's paths, relative to the folder)."""
    problems = []
    home, allowed = folder.resolve(), comparison_dir.resolve()
    if home == allowed or not home.is_relative_to(allowed):
        problems.append(f"its folder {folder} does not lie under {comparison_dir}")
    outside = sorted(name for name in listed if not (home / name).resolve().is_relative_to(home))
    if outside:
        problems.append(f"its {MANIFEST_NAME} lists file(s) outside its folder: {', '.join(outside)}")
    return problems


# ------------------------------------------------------------------------------------------- tests


@pytest.mark.parametrize("role", LIST_CASES)
def test_main_list_file_has_10_rows_ranks_1_to_10_once_and_10_players(role: str, outputs_dir: Path) -> None:
    """The Role's Main_List file has exactly 10 rows, uses each rank from 1 to 10 exactly once and holds 10 distinct
    ``player_id`` values.

    **Validates: Requirements 25.4**
    """
    path = MAIN_LISTS[role]
    problems = _list_problems(_read_csv(outputs_dir / path))
    assert not problems, _failure(f"outputs/{path} breaks Requirement 25, criterion 4:", problems)


@pytest.fixture(scope="module")
def name_index(checkpoints: CheckpointStore) -> NameIndex:
    """The Candidate_Names of the mapping table in the load checkpoint that the final_lists stage read."""
    _require_complete(checkpoints, lists.STAGE_NAME, loader.STAGE_NAME)
    _require_built_on(checkpoints, lists.STAGE_NAME, loader.STAGE_NAME)
    return _name_index(lists.candidate_names(checkpoints.load_table(loader.STAGE_NAME, MAPPING_TABLE)))


@pytest.mark.parametrize("role", NAME_CASES)
def test_each_player_name_is_a_candidate_name_of_one_player_id_unless_unresolved(
    role: str, name_index: NameIndex, written: Written, outputs_dir: Path
) -> None:
    """Every ``player_name`` of the Role's Main_List file is a Candidate_Name of exactly one ``player_id``, the row's
    own, unless the checklist marks the entry unresolved; and the checklist marks exactly the Unresolved_Entries.

    **Validates: Requirements 25.4**
    """
    path = MAIN_LISTS[role]
    _require_written(written, outputs_dir, path, CHECKLIST)
    checklist = lists.parse_checklist((outputs_dir / CHECKLIST).read_text(encoding="utf-8"))
    problems = _name_problems(role, _read_csv(outputs_dir / path), checklist, name_index)
    assert not problems, _failure(
        f"the {role} Main_List outputs/{path} and the checklist outputs/{CHECKLIST} break Requirement 25, "
        "criterion 4:",
        problems,
    )


@pytest.mark.parametrize("name", EVIDENCE_CASES)
def test_evidence_table_holds_the_10_listed_players_in_rank_order(
    name: str, written: Written, outputs_dir: Path
) -> None:
    """The evidence table holds exactly the 10 players of its Role's Main_List, at the same ranks, in rank order 1 to
    10, with the same names.

    **Validates: Requirements 26.11**
    """
    role = EVIDENCE_ROLES[name]
    path, main_path = EVIDENCE_FILES[name], MAIN_LISTS[role]
    _require_written(written, outputs_dir, path, main_path)
    main = _read_csv(outputs_dir / main_path)
    listed = _keys(main)
    if listed is None or _list_problems(main):
        pytest.fail(
            f"outputs/{main_path} does not give ranks 1 to {LIST_SIZE} once each to {LIST_SIZE} players, so it has "
            "no rank order to compare with; see test_main_list_file_has_10_rows_ranks_1_to_10_once_and_10_players"
        )
    problems = _order_problems(_read_csv(outputs_dir / path), sorted(listed, key=lambda key: key[0]))
    assert not problems, _failure(
        f"outputs/{path} does not hold the {LIST_SIZE} players of the {role} Main_List outputs/{main_path} in rank "
        "order (Requirement 26, criterion 11):",
        problems,
    )


def _by_player(frame: pd.DataFrame, role: str, what: str) -> dict[int, dict[str, Any]]:
    """The Role's rows of a checkpoint table as dicts, by player_id."""
    rows = frame.loc[(frame["role"] == role).to_numpy(dtype=bool)]
    ids = rows["player_id"].to_numpy(dtype=np.int64).tolist()
    repeated = sorted(pid for pid, n in Counter(ids).items() if n > 1)
    assert not repeated, f"the {what} table has more than one {role} row for player_id(s) {repeated}"
    return dict(zip(ids, rows.to_dict("records")))


@pytest.fixture(scope="module")
def sources(checkpoints: CheckpointStore) -> dict[str, RoleSources]:
    """Per Role, the Main_Ranking's player_estimates rows and its bootstrap rows, from the checkpoints that the
    final_lists stage read."""
    _require_complete(checkpoints, lists.STAGE_NAME, rating.STAGE_NAME, robustness.BOOTSTRAP_STAGE)
    _require_built_on(checkpoints, lists.STAGE_NAME, rating.STAGE_NAME, robustness.BOOTSTRAP_STAGE)
    _require_built_on(checkpoints, robustness.BOOTSTRAP_STAGE, rating.STAGE_NAME)
    tables = {
        RANKING: checkpoints.load_table(rating.STAGE_NAME, RANKING),
        BOOTSTRAP: checkpoints.load_table(robustness.BOOTSTRAP_STAGE, BOOTSTRAP),
    }
    for source, table in tables.items():
        needed = (*SOURCE_KEYS[source], *(column for column, where in VALUE_SOURCES.items() if where == source))
        absent = [column for column in needed if column not in table.columns]
        assert not absent, f"the {source} checkpoint table has no column {', '.join(absent)}"
    return {role: {source: _by_player(table, role, source) for source, table in tables.items()} for role in ROLES}


@pytest.mark.parametrize("role", VALUE_CASES)
def test_rating_evidence_values_equal_the_rating_and_bootstrap_values(
    role: str, sources: dict[str, RoleSources], written: Written, outputs_dir: Path
) -> None:
    """Every value of the Role's rating-evidence file is the float64 that the Rating_Engine (``player_estimates``)
    or the bootstrap (``bootstrap``) computed for the same player and Role of the Main_Ranking.

    **Validates: Requirements 26.12**
    """
    path = EVIDENCE_FILES[RATING_EVIDENCE[role]]
    _require_written(written, outputs_dir, path)
    table = _read_csv(outputs_dir / path, text_columns=VALUE_SOURCES)
    problems = _value_problems(role, table, sources[role])
    assert not problems, _failure(
        f"outputs/{path} holds values that the Rating_Engine ({rating.STAGE_NAME} checkpoint, {RANKING}) or the "
        f"bootstrap ({robustness.BOOTSTRAP_STAGE} checkpoint, {BOOTSTRAP}) did not compute for the same {role} "
        "(Requirement 26, criterion 12):",
        problems,
    )


@pytest.mark.requires_outputs(*LISTS_AND_CHECKLIST, *WRITTEN)
def test_comparison_step_changed_no_protected_file(written: Written, outputs_dir: Path) -> None:
    """The Main_List files, the checklist, the files of ``outputs/rating/`` and ``outputs/robustness/bootstrap_*.csv``
    are in the snapshot of the final_lists stage's comparison step and still have its SHA-256.

    **Validates: Requirements 25.15**
    """
    problems = _guard_problems(written, _protected_on_disk(outputs_dir))
    assert not problems, _failure(
        f"the protected files did not keep their SHA-256 through the activity comparison step of the "
        f"{lists.STAGE_NAME} stage (the comparison_guard snapshot of its {lists.WRITTEN_JSON} value) and since "
        "(Requirement 25, criterion 15):",
        problems,
    )


@pytest.mark.requires_outputs(*FINAL_MANIFEST, *ACTIVITY)
def test_activity_comparison_checkpoint_lies_under_outputs_final_comparison(
    checkpoints: CheckpointStore, outputs_dir: Path
) -> None:
    """The activity_comparison checkpoint that the final_lists stage read lies under ``outputs/final/comparison/``,
    and its manifest lists only files inside its folder.

    **Validates: Requirements 25.15**
    """
    _require_complete(checkpoints, lists.STAGE_NAME, activity.STAGE_NAME)
    _require_built_on(checkpoints, lists.STAGE_NAME, activity.STAGE_NAME)
    comparison_dir = outputs_dir / lists.FINAL_FOLDER / lists.COMPARISON_FOLDER
    problems = _checkpoint_problems(
        checkpoints.stage_dir(activity.STAGE_NAME), checkpoints.manifest(activity.STAGE_NAME)["files"], comparison_dir
    )
    assert not problems, _failure(
        f"the {activity.STAGE_NAME} checkpoint does not stay under outputs/{lists.FINAL_FOLDER}/"
        f"{lists.COMPARISON_FOLDER}/ (Requirement 25, criterion 15):",
        problems,
    )
