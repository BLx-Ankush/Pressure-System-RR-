r"""Two-run reproducibility tests (Requirement 28, criteria 2 and 11).

**Validates: Requirements 28.2, 28.11**

Criterion 2: two full Runs on the same machine, with the same installed package versions, the same
Config SHA-256 hash and the same SHA-256 hash for each Source_CSV, each started from an empty or
absent ``outputs/``, give batter and bowler top-10 lists (the Main_Lists) in which every rank holds
the same ``player_id`` and ``player_name`` and every Final_Score is equal after rounding to 6
decimal places. Criterion 11 asks the same of the Comparison_Lists where the Recent_Activity_Rule
is on in the Config.

A full Run takes 1-2 hours, so these tests run no Pipeline. They compare the lists of the Run now in
``outputs/final/`` with a copy of ``outputs/final/`` made after an earlier full Run: the snapshot
folder, named by the environment variable ``SUPERR_REPEAT_SNAPSHOT``. While the variable is unset or
empty, every test here is skipped. When it is set but does not name an existing folder, or names a
folder inside ``outputs/``, every test fails.

Two-run procedure (Windows PowerShell, in the workspace root)
    Criterion 2 holds four things fixed between the two Runs: run both on this machine, with the
    same ``.venv`` (the same installed package versions), and change neither ``config.py`` (the
    Config hash) nor any of the six Source_CSVs (their hashes). Each Run must start from an empty
    or absent ``outputs\``. ::

        # 1. First full Run. Delete or rename an existing outputs\ first, as in step 3.
        .venv\Scripts\python run_pipeline.py

        # 2. Copy outputs\final, and the Run_Log in outputs\logs, to a new folder outside outputs\.
        #    The folder must not exist yet: Copy-Item would otherwise copy into a final\ subfolder.
        Copy-Item -Path outputs\final -Destination "$env:USERPROFILE\superr_run1_final" -Recurse
        Copy-Item -Path outputs\logs -Destination "$env:USERPROFILE\superr_run1_final\logs" -Recurse

        # 3. Make outputs\ absent for the second Run: delete it, or rename it.
        Remove-Item -Path outputs -Recurse -Force
        # or: Rename-Item -Path outputs -NewName outputs_run1

        # 4. Second full Run.
        .venv\Scripts\python run_pipeline.py

        # 5. Name the snapshot folder (for this PowerShell window only).
        $env:SUPERR_REPEAT_SNAPSHOT = "$env:USERPROFILE\superr_run1_final"

        # 6. Compare the two Runs.
        .venv\Scripts\python -m pytest tests/pipeline_output/test_repeat_runs.py

    ``Remove-Item Env:SUPERR_REPEAT_SNAPSHOT`` unsets the variable. The second command of step 2 is
    optional: it puts the first Run's Run_Log in the snapshot, so that
    ``test_both_runs_meet_the_conditions_of_criterion_2`` can check the conditions above. Without
    it, that test is skipped.

The tests
    ``test_main_list_ranks_match_the_snapshot`` (criterion 2)
        One case per Main_List: ``top10_batters.csv`` and ``top10_bowlers.csv``. A Main_List
        missing from the snapshot is a failure, since every full Run writes both.
    ``test_comparison_list_ranks_match_the_snapshot`` (criterion 11)
        One case per Comparison_List: ``comparison/top10_batters_activity_off.csv`` and
        ``comparison/top10_bowlers_activity_off.csv``. The Pipeline writes them only while the
        Recent_Activity_Rule is on, so the cases are skipped while ``data_rules.recent_activity_rule``
        is off in the Config. While it is on, a Comparison_List missing from ``outputs/final/`` or
        from the snapshot is a failure.

    Each case compares the list file in ``outputs/final/`` with the file of the same path in the
    snapshot. Both files must have the columns ``rank``, ``player_id``, ``player_name`` and
    ``final_score`` and at least one row, and give each rank once. Then:

    * both files have the same number of rows and the same ranks;
    * at every rank, both hold the same ``player_id`` and the same ``player_name``, compared code
      point by code point (no Unicode normalisation, case folding or trimming);
    * at every rank, the two Final_Scores are equal after rounding to 6 decimals (below).

    A failure names both files and lists every rank with the values that differ. Criterion 2 names
    no other column, so the other columns are not compared. A case is skipped
    (``requires_outputs``) while ``outputs/final/`` lacks the Main_List of its Role, that is, until
    a Run has completed the ``final_lists`` stage.

    ``test_both_runs_meet_the_conditions_of_criterion_2``
        Checks the conditions of criterion 2 from the two Runs' Run_Logs (below). Skipped while the
        snapshot holds no Run_Log.

Reading the files
    Each list is read as the pipeline-output tests read it, with ``pandas.read_csv(path,
    keep_default_na=False, na_values=[""], dtype={"player_name": str, "final_score": str})``. Only
    an empty field is missing, so a name such as ``NA`` stays text; ``rank`` and ``player_id`` read
    as integers; ``final_score`` keeps the text written in the file.

Final_Score comparison
    Criterion 2 compares Final_Scores after rounding to 6 decimals: ``round(a, 6) == round(b, 6)``.
    The test applies this to the text in the two files (10 decimals, ``lists.LIST_FLOAT_FORMAT``) as
    ``decimal.Decimal`` values, rounded half to even as ``round`` rounds. The result depends only on
    the decimal values written, not on the binary floats nearest to them. A change from the 7th
    decimal on passes unless it crosses a rounding boundary: 0.1234564999 and 0.1234565001 round to
    0.123456 and 0.123457, which the criterion counts as a difference.

Run_Log check (``test_both_runs_meet_the_conditions_of_criterion_2``)
    Every Run writes its Run_Log to ``outputs/logs/run_<UTC>.json`` (Requirement 28, criterion 4).
    The test reads the Run_Logs in the snapshot's ``logs/`` folder (step 2) and in
    ``outputs/logs/``. In each folder, the Run that wrote the lists is the latest Run, by the start
    time in the file name, whose Run_Log records a completed ``final_lists`` stage. The test fails,
    listing every unmet condition, unless:

    * each of the two Runs was a full Run (started with no arguments) that completed with exit
      code 0;
    * no Run_Log in its folder started before it. A Run that starts from an empty or absent
      ``outputs/`` finds no earlier Run_Log there. Other files left in ``outputs/`` leave no trace
      in the Run_Log, so step 3 still matters;
    * the two Run_Logs are of two different Runs;
    * both Runs record the same Config SHA-256 hash, the same Source_CSVs with the same SHA-256
      hash each, and the same installed version of every package pinned in ``requirements.txt``.

    The Run_Log records no machine identity, so "the same machine" rests on the procedure.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import pandas as pd
import pytest

from superr.lists import COMPARISON_FOLDER, COMPARISON_LIST_FILES, FINAL_FOLDER, MAIN_LIST_FILES
from superr.lists import STAGE_NAME as FINAL_LISTS_STAGE
from superr.pipeline import OUTPUTS_FOLDER
from superr.runlog import COMPLETED, LOG_FOLDER
from superr.value import ROLES

if TYPE_CHECKING:
    from superr.settings import Settings

# ------------------------------------------------------------------------------------- constants

#: The environment variable that names the snapshot folder: a copy of outputs/final/ from an earlier full Run.
SNAPSHOT_VARIABLE = "SUPERR_REPEAT_SNAPSHOT"

#: The Config switch of the Recent_Activity_Rule. The Comparison_Lists, and criterion 11, exist only while it is on.
RULE_KEY = "data_rules.recent_activity_rule"

#: The list columns that criterion 2 compares, and those of them that pandas must read as integers.
COMPARED_COLUMNS: tuple[str, ...] = ("rank", "player_id", "player_name", "final_score")
INTEGER_COLUMNS: tuple[str, ...] = ("rank", "player_id")

#: Final_Scores are compared after rounding to 6 decimals (module docstring, "Final_Score comparison").
SIX_DECIMALS = Decimal("0.000001")

#: A Run_Log JSON file name (``superr.runlog``): ``run_<UTC>.json``, or ``run_<UTC>_<n>.json`` for the n-th Run
#: started in the same second. The two groups order the files by start time.
RUN_LOG_NAME = re.compile(r"run_(\d{8}T\d{6}Z)(?:_(\d+))?\.json")

RUN_COMMAND = r".venv\Scripts\python run_pipeline.py"
_SHOWN = 3  # example Run_Log names in one problem line

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / OUTPUTS_FOLDER
FINAL = OUTPUTS / FINAL_FOLDER
THIS_FILE = Path(__file__).resolve().relative_to(ROOT).as_posix()

#: Where a list value comes from, and which Run a Run_Log value comes from, in failure messages.
SNAPSHOT = "the snapshot"
CURRENT = f"{OUTPUTS_FOLDER}/{FINAL_FOLDER}"
SNAPSHOT_RUN = "the snapshot's Run"
CURRENT_RUN = f"the Run in {OUTPUTS_FOLDER}/"


def _in_outputs(relative: str) -> str:
    """A path relative to outputs/final/ (and to the snapshot) as requires_outputs takes it: relative to outputs/."""
    return f"{FINAL_FOLDER}/{relative}"


MAIN_LIST_PATHS: tuple[str, ...] = tuple(_in_outputs(MAIN_LIST_FILES[role]) for role in ROLES)

MAIN_CASES = [
    pytest.param(
        MAIN_LIST_FILES[role],
        marks=pytest.mark.requires_outputs(_in_outputs(MAIN_LIST_FILES[role])),
        id=MAIN_LIST_FILES[role],
    )
    for role in ROLES
]
#: A Comparison_List case needs the Main_List of its Role, which shows that a Run completed the final_lists stage,
#: and not the Comparison_List itself: while the Recent_Activity_Rule is on, a missing Comparison_List is a failure.
COMPARISON_CASES = [
    pytest.param(
        f"{COMPARISON_FOLDER}/{COMPARISON_LIST_FILES[role]}",
        marks=pytest.mark.requires_outputs(_in_outputs(MAIN_LIST_FILES[role])),
        id=COMPARISON_LIST_FILES[role],
    )
    for role in ROLES
]

# ------------------------------------------------------------------------------- shared helpers


def _failure(header: str, problems: list[str]) -> str:
    return "\n".join([header, *(f"  - {line}" for line in problems)])


def _snapshot_folder() -> Path:
    """The folder named by SNAPSHOT_VARIABLE. Skips while the variable is unset or empty; fails when it names no
    existing folder, or a folder inside outputs/."""
    value = os.environ.get(SNAPSHOT_VARIABLE, "").strip()
    if not value:
        pytest.skip(
            f"{SNAPSHOT_VARIABLE} is not set: set it to a copy of outputs/final/ from an earlier full Run to compare "
            f"the two Runs' lists (Requirement 28, criteria 2 and 11; procedure in the docstring of {THIS_FILE})"
        )
    folder = Path(value).expanduser().resolve()
    if not folder.is_dir():
        pytest.fail(
            f"{SNAPSHOT_VARIABLE} = {value!r}, but {folder} is not an existing folder: it must name a copy of "
            f"outputs/final/ from an earlier full Run (step 2 of the procedure in the docstring of {THIS_FILE})",
            pytrace=False,
        )
    if folder.is_relative_to(OUTPUTS.resolve()):
        pytest.fail(
            f"{SNAPSHOT_VARIABLE} = {value!r} names {folder}, which is inside outputs/: the snapshot must be a copy "
            f"outside outputs/, because the second Run starts without outputs/ (steps 2 and 3 of the procedure in the "
            f"docstring of {THIS_FILE})",
            pytrace=False,
        )
    return folder


# ------------------------------------------------------------------------------- list comparison


def _read_list(path: Path) -> pd.DataFrame:
    """A list file as the pipeline-output tests read it, with ``final_score`` kept as the text written."""
    return pd.read_csv(
        path, keep_default_na=False, na_values=[""], dtype={"player_name": str, "final_score": str}, encoding="utf-8"
    )


def _text(value: Any) -> str | None:
    """A text field as read: None for an empty field."""
    return None if pd.isna(value) else str(value)


def _shown(name: str | None) -> str:
    """A player_name for a message: its repr, and its code points unless it is printable ASCII."""
    if name is None:
        return "an empty field"
    if name.isascii() and name.isprintable():
        return repr(name)
    return f"{name!r} (code points {' '.join(f'U+{ord(char):04X}' for char in name)})"


def _rounded(text: str | None) -> Decimal | None:
    """A Final_Score text rounded to 6 decimals, half to even as ``round`` rounds; None unless a finite number."""
    if text is None:
        return None
    try:
        value = Decimal(text)
    except InvalidOperation:
        return None
    return value.quantize(SIX_DECIMALS, rounding=ROUND_HALF_EVEN) if value.is_finite() else None


def _table_problems(table: pd.DataFrame, where: str) -> list[str]:
    """What stops a rank-by-rank comparison of the list read from ``where``."""
    absent = [column for column in COMPARED_COLUMNS if column not in table.columns]
    if absent:
        return [f"the file in {where} has no column {', '.join(absent)} (pandas reads {table.columns.tolist()})"]
    if table.empty:
        return [f"the file in {where} holds no list row"]
    problems = [
        f"the {column} column of the file in {where} does not hold whole numbers only "
        f"(pandas reads {table[column].dtype})"
        for column in INTEGER_COLUMNS
        if not pd.api.types.is_integer_dtype(table[column].dtype)
    ]
    if not problems:
        repeated = sorted(set(table.loc[table["rank"].duplicated(), "rank"].tolist()))
        if repeated:
            problems.append(f"the file in {where} gives rank {', '.join(map(str, repeated))} more than once")
    return problems


def _rank_problems(rank: int, before: Any, after: Any) -> list[str]:
    """The differences at one rank between the snapshot's row ``before`` and the outputs/final/ row ``after``."""
    problems = []
    id_before, id_after = int(before.player_id), int(after.player_id)
    if id_before != id_after:
        problems.append(f"rank {rank}: player_id {id_before} in {SNAPSHOT}, {id_after} in {CURRENT}")
    name_before, name_after = _text(before.player_name), _text(after.player_name)
    if name_before != name_after:  # str equality: code point by code point
        problems.append(
            f"rank {rank}: player_name {_shown(name_before)} in {SNAPSHOT}, {_shown(name_after)} in {CURRENT}"
        )
    score_before, score_after = _text(before.final_score), _text(after.final_score)
    rounded_before, rounded_after = _rounded(score_before), _rounded(score_after)
    for where, text, rounded in ((SNAPSHOT, score_before, rounded_before), (CURRENT, score_after, rounded_after)):
        if rounded is None:
            shown = "an empty field" if text is None else repr(text)
            problems.append(f"rank {rank}: final_score {shown} in {where} is not a finite number")
    if rounded_before is not None and rounded_after is not None and rounded_before != rounded_after:
        difference = abs(Decimal(score_after) - Decimal(score_before))
        problems.append(
            f"rank {rank}: final_score {score_before} in {SNAPSHOT} and {score_after} in {CURRENT} differ after "
            f"rounding to 6 decimals: {rounded_before} and {rounded_after} (difference {difference:f})"
        )
    return problems


def _list_problems(snapshot_file: Path, current_file: Path) -> list[str]:
    """Every difference that criterion 2 forbids between the snapshot's list and the list in outputs/final/."""
    tables = {SNAPSHOT: _read_list(snapshot_file), CURRENT: _read_list(current_file)}
    problems = [line for where, table in tables.items() for line in _table_problems(table, where)]
    if problems:
        return problems
    before, after = tables[SNAPSHOT], tables[CURRENT]
    if len(before) != len(after):
        problems.append(f"the file in {SNAPSHOT} has {len(before)} rows, the file in {CURRENT} {len(after)}")
    rows_before, rows_after = (
        {int(row.rank): row for row in table[list(COMPARED_COLUMNS)].itertuples(index=False)}
        for table in (before, after)
    )
    only = {SNAPSHOT: rows_before.keys() - rows_after.keys(), CURRENT: rows_after.keys() - rows_before.keys()}
    for where, ranks in only.items():
        if ranks:
            problems.append(f"only the file in {where} gives rank {', '.join(map(str, sorted(ranks)))}")
    for rank in sorted(rows_before.keys() & rows_after.keys()):
        problems += _rank_problems(rank, rows_before[rank], rows_after[rank])
    return problems


def _assert_reproduced(relative: str, snapshot: Path, criterion: int) -> None:
    """The list at ``relative`` under outputs/final/ and under the snapshot is the same, rank by rank."""
    current_file, snapshot_file = FINAL / relative, snapshot / relative
    problems = _list_problems(snapshot_file, current_file)
    assert not problems, _failure(
        f"{relative}: the Run in outputs/final/ does not reproduce the snapshot's list (Requirement 28, criterion "
        f"{criterion}); compared {current_file} with {snapshot_file}:",
        problems,
    )


# ----------------------------------------------------------------------------------------- lists


@pytest.mark.parametrize("relative", MAIN_CASES)
def test_main_list_ranks_match_the_snapshot(relative: str) -> None:
    """At every rank, the Main_List in outputs/final/ holds the snapshot's player_id and player_name, and a
    Final_Score equal to the snapshot's after rounding to 6 decimals.

    **Validates: Requirements 28.2**
    """
    snapshot = _snapshot_folder()
    if not (snapshot / relative).is_file():
        pytest.fail(
            f"{snapshot / relative} does not exist: {SNAPSHOT_VARIABLE} must name a copy of outputs/final/ from a "
            f"full Run, and every full Run writes {relative} (Requirement 28, criterion 1)",
            pytrace=False,
        )
    _assert_reproduced(relative, snapshot, criterion=2)


@pytest.mark.parametrize("relative", COMPARISON_CASES)
def test_comparison_list_ranks_match_the_snapshot(relative: str, settings: Settings) -> None:
    """While the Recent_Activity_Rule is on, at every rank the Comparison_List in outputs/final/ holds the
    snapshot's player_id and player_name, and a Final_Score equal to the snapshot's after rounding to 6 decimals.

    **Validates: Requirements 28.11**
    """
    snapshot = _snapshot_folder()
    rule = settings.get(RULE_KEY)
    if not rule:
        pytest.skip(
            f"the Recent_Activity_Rule is off in the Config ({RULE_KEY} = {rule!r}): the Pipeline then writes no "
            "Comparison_List, and Requirement 28, criterion 11 does not apply"
        )
    missing = [str(path) for path in (FINAL / relative, snapshot / relative) if not path.is_file()]
    if missing:
        pytest.fail(
            f"the Recent_Activity_Rule is on in the Config ({RULE_KEY} = {rule!r}), so every full Run writes "
            f"{relative} (Requirement 25, criterion 13) and Requirement 28, criterion 11 compares it, but "
            f"{' and '.join(missing)} {'does' if len(missing) == 1 else 'do'} not exist",
            pytrace=False,
        )
    _assert_reproduced(relative, snapshot, criterion=11)


# -------------------------------------------------------------------------------------- Run_Logs


class LoggedRun(NamedTuple):
    """The Run_Log record of the Run that wrote one folder's lists, and the problems found with that Run."""

    record: dict[str, Any] | None  # None when no Run_Log records a completed final_lists stage
    problems: list[str]


def _run_log_files(folder: Path) -> list[Path]:
    """The Run_Log JSON files directly in ``folder``, oldest first by the start time in their names."""
    if not folder.is_dir():
        return []
    named = []
    for path in folder.iterdir():
        match = RUN_LOG_NAME.fullmatch(path.name)
        if match is not None and path.is_file():
            named.append(((match.group(1), int(match.group(2) or 1)), path))
    return [path for _, path in sorted(named)]


def _completed(record: Mapping[str, Any], stage: str) -> bool:
    """Whether the Run_Log record lists ``stage`` as completed."""
    stages = record.get("stages")
    return isinstance(stages, list) and any(
        isinstance(entry, dict) and entry.get("name") == stage and entry.get("status") == COMPLETED for entry in stages
    )


def _logged_run(folder: Path, run: str) -> LoggedRun:
    """The Run_Log of the Run that wrote the lists, from the Run_Logs in ``folder``; ``run`` names it in messages.

    That Run is the latest whose Run_Log records a completed final_lists stage. The problems: a Run_Log that cannot
    be read, no such Run, a Run that was not a full Run completed with exit code 0, a Run_Log in ``folder`` from
    before that Run (a Run that starts from an empty or absent outputs/ finds none), and a Run_Log without the
    Config hash, the Source_CSV hashes or the package versions.
    """
    files = _run_log_files(folder)
    problems: list[str] = []
    writers: list[tuple[Path, dict[str, Any]]] = []
    for path in files:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            problems.append(f"{path} cannot be read as a Run_Log: {exc}")
            continue
        if isinstance(record, dict) and _completed(record, FINAL_LISTS_STAGE):
            writers.append((path, record))
    if not writers:
        problems.append(
            f"no Run_Log in {folder} records a completed {FINAL_LISTS_STAGE} stage, so none is the Run_Log of "
            f"{run}, which wrote the lists"
        )
        return LoggedRun(None, problems)
    path, record = writers[-1]
    name = f"{run} ({path.name})"
    argv, status, code = record.get("argv"), record.get("status"), record.get("exit_code")
    if argv != []:
        problems.append(
            f"{name} was started with the arguments {argv!r}, so it is not a full Run "
            f"(`{RUN_COMMAND}` with no arguments)"
        )
    if status != COMPLETED or code != 0:
        problems.append(
            f"{name} ended with status {status!r} and exit code {code!r}, not {COMPLETED!r} with exit code 0"
        )
    earlier = files[: files.index(path)]
    if earlier:
        shown = ", ".join(file.name for file in earlier[:_SHOWN]) + (", ..." if len(earlier) > _SHOWN else "")
        problems.append(
            f"{name} did not start from an empty or absent outputs/: {folder} holds {len(earlier)} Run_Log(s) of "
            f"earlier Runs ({shown})"
        )
    for key, what in (("config", "Config hash"), ("source_csvs", "Source_CSV hash"), ("packages", "package version")):
        if not record.get(key):
            problems.append(f"{name} records no {what}")
    return LoggedRun(record, problems)


def _entries(record: Mapping[str, Any], key: str, field: str) -> dict[str, Any]:
    """name -> ``field`` for each entry of the Run_Log list ``key``: source_csvs (sha256) or packages (installed)."""
    entries = record.get(key)
    if not isinstance(entries, list):
        return {}
    return {str(entry.get("name")): entry.get(field) for entry in entries if isinstance(entry, dict)}


#: Marks a name that only one of the two Run_Logs records.
_ABSENT = object()


def _recorded(value: Any) -> str:
    return "not recorded" if value is _ABSENT else repr(value)


def _config_sha256(record: Mapping[str, Any]) -> dict[str, Any]:
    """The Run_Log's Config hash, under an empty name so that it compares like the other entries."""
    config = record.get("config")
    return {"": config.get("sha256")} if isinstance(config, dict) else {}


def _condition_problems(first: Mapping[str, Any], second: Mapping[str, Any]) -> list[str]:
    """The conditions of criterion 2 that differ between the snapshot's Run (``first``) and the Run in outputs/."""
    problems = []
    if (first.get("run_id"), first.get("started_utc")) == (second.get("run_id"), second.get("started_utc")):
        problems.append(
            f"both Run_Logs are of the same Run, {first.get('run_id')}: criterion 2 compares two separate full Runs"
        )
    compared = (
        ("the Config SHA-256 hash", _config_sha256(first), _config_sha256(second)),
        ("the SHA-256 hash of Source_CSV {name}", _entries(first, "source_csvs", "sha256"),
         _entries(second, "source_csvs", "sha256")),
        ("the installed version of package {name}", _entries(first, "packages", "installed"),
         _entries(second, "packages", "installed")),
    )  # fmt: skip
    for what, values_first, values_second in compared:
        for name in sorted(values_first.keys() | values_second.keys()):
            value_first, value_second = values_first.get(name, _ABSENT), values_second.get(name, _ABSENT)
            if value_first != value_second:
                problems.append(
                    f"{what.format(name=name)}: {_recorded(value_first)} in {SNAPSHOT_RUN}, "
                    f"{_recorded(value_second)} in {CURRENT_RUN}"
                )
    return problems


@pytest.mark.requires_outputs(*MAIN_LIST_PATHS, LOG_FOLDER)
def test_both_runs_meet_the_conditions_of_criterion_2() -> None:
    """The Run_Logs show two separate full Runs, each completed with exit code 0 from an empty or absent outputs/,
    with the same Config hash, the same Source_CSV hashes and the same installed package versions.

    **Validates: Requirements 28.2, 28.11**
    """
    snapshot = _snapshot_folder()
    snapshot_logs = snapshot / LOG_FOLDER
    if not _run_log_files(snapshot_logs):
        pytest.skip(
            f"the snapshot holds no Run_Log (no run_<UTC>.json in {snapshot_logs}), so the conditions of "
            rf"criterion 2 are not checked; copy outputs\{LOG_FOLDER} to {snapshot_logs} right after the first Run "
            f"(step 2 of the procedure in the docstring of {THIS_FILE}) to check them"
        )
    first = _logged_run(snapshot_logs, SNAPSHOT_RUN)
    second = _logged_run(OUTPUTS / LOG_FOLDER, CURRENT_RUN)
    problems = [*first.problems, *second.problems]
    if first.record is not None and second.record is not None:
        problems += _condition_problems(first.record, second.record)
    assert not problems, _failure(
        f"{SNAPSHOT_RUN} and {CURRENT_RUN} do not meet the conditions of Requirement 28, criterion 2, so the list "
        "comparisons of this module do not test criteria 2 and 11:",
        problems,
    )
