"""Real-data schema header test (Requirement 2, criterion 2).

For each of the six Source_CSVs, every column that the Schema_Registry
(``superr.io.SCHEMA_REGISTRY``) lists for the file must appear in the file's actual header row
with the same spelling and letter case. Names are compared exactly, so ``Match_ID`` or
``match_id `` does not count as ``match_id``. A failure names the file, its path and every
missing column, and shows the header row as read so that a spelling or case difference is
visible.

Only the header row is read. The file is opened read-only in binary mode, its first line is
decoded as UTF-8 and split with the ``csv`` module (standard quoting). A leading UTF-8
byte-order mark is dropped, because the Data_Loader ignores it too. ``ball_by_ball.csv`` is
about 41 MB, so its data rows are never read.

The files checked are the ones the Pipeline reads, at the Config paths (``settings.path(key)``).
Two companion tests check the inputs of the comparison:

* the Schema_Registry has one entry for each of the six Source_CSVs of the glossary and no
  other entry, and ``superr.io.FILE_NAMES`` (Config key -> file name) agrees with the glossary;
* each Config path ``paths.<key>`` resolves to the existing file of that name in the workspace
  root (Requirement 1, criterion 2), not to the copy in the nested problem-statement folder.

A last test runs the comparison on a small header in ``tmp_path`` with one column in the wrong
letter case and one column absent, to show that the check is case-sensitive and that its
failure message names the file and both columns.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from superr.io import FILE_NAMES, SCHEMA_REGISTRY

if TYPE_CHECKING:
    from superr.settings import Settings

#: The six Source_CSVs of the glossary, each with its Config key under ``paths``.
SOURCE_CSV_KEYS: dict[str, str] = {
    "ball_by_ball.csv": "ball_by_ball",
    "matches.csv": "matches",
    "playing_eleven.csv": "playing_eleven",
    "competitions.csv": "competitions",
    "teams.csv": "teams",
    "competition_team_player_mapping.csv": "mapping",
}

#: The folder in the workspace root that holds copies of the data pack; no Config path may
#: point into it.
NESTED_FOLDER = "SupeRR Selector Season 2 _ Problem Statement _ Data Pack"


def read_header(path: Path) -> list[str]:
    """The column names in the header row of a CSV file, exactly as written.

    Reads the first line only, in read-only binary mode, and decodes it as UTF-8 without a
    leading byte-order mark. An empty file gives an empty list.
    """
    with open(path, "rb") as handle:
        first_line = handle.readline()
    return next(csv.reader([first_line.decode("utf-8-sig")]), [])


def missing_columns(header: Iterable[str], expected: Iterable[str]) -> list[str]:
    """The expected columns absent from the header, in expected order (exact, case-sensitive)."""
    present = set(header)
    return [column for column in expected if column not in present]


def header_failure(file_name: str, path: Path, header: list[str], missing: list[str]) -> str:
    """The failure message: the file, every missing column and the header row as read."""
    shown = ", ".join(repr(name) for name in header) or "(no header row)"
    columns = "column" if len(missing) == 1 else "columns"
    return (
        f"{file_name} ({path}): the header row lacks {len(missing)} Schema_Registry {columns} "
        f"(names compared exactly, letter case included): {', '.join(missing)}. "
        f"Header row as read: {shown}"
    )


def test_registry_covers_the_six_source_csvs() -> None:
    expected = {key: file_name for file_name, key in SOURCE_CSV_KEYS.items()}
    assert FILE_NAMES == expected, (
        f"superr.io.FILE_NAMES does not match the six Source_CSVs of the glossary: {FILE_NAMES}"
    )
    absent = [f"{file_name} ({key})" for file_name, key in SOURCE_CSV_KEYS.items() if key not in SCHEMA_REGISTRY]
    unknown = [key for key in SCHEMA_REGISTRY if key not in expected]
    assert not absent and not unknown, (
        "the Schema_Registry must have exactly one entry per Source_CSV; "
        f"no entry for: {', '.join(absent) or 'none'}; entries for no Source_CSV: {', '.join(unknown) or 'none'}"
    )
    empty = [expected[key] for key, columns in SCHEMA_REGISTRY.items() if not columns]
    assert not empty, f"Schema_Registry entries that list no column: {', '.join(empty)}"


def test_config_paths_point_to_the_root_files(settings: Settings, workspace_root: Path) -> None:
    root = workspace_root.resolve()
    wrong: list[str] = []
    for file_name, key in SOURCE_CSV_KEYS.items():
        path = settings.path(key).resolve()
        if path != root / file_name or not path.is_file():
            wrong.append(
                f"paths.{key} = {settings.get(f'paths.{key}')!r} gives {path}; "
                f"expected the existing file {root / file_name}"
            )
    assert not wrong, (
        "each Config path must point to the Source_CSV of that name in the workspace root "
        f"(Requirement 1, criterion 2), not to a copy such as the one under {NESTED_FOLDER!r}:\n  "
        + "\n  ".join(wrong)
    )


@pytest.mark.parametrize(("file_name", "key"), list(SOURCE_CSV_KEYS.items()), ids=list(SOURCE_CSV_KEYS))
def test_header_has_every_registry_column(file_name: str, key: str, settings: Settings) -> None:
    assert key in SCHEMA_REGISTRY, f"{file_name}: the Schema_Registry has no entry {key!r} for this file"
    path = settings.path(key)
    header = read_header(path)
    missing = missing_columns(header, SCHEMA_REGISTRY[key])
    assert not missing, header_failure(file_name, path, header, missing)


def test_header_check_is_case_sensitive_and_names_the_file_and_columns(tmp_path: Path) -> None:
    # A byte-order mark and CRLF line ends, as some CSV writers produce. Comp_Name differs from
    # comp_name only in letter case, and season is absent.
    path = tmp_path / "competitions.csv"
    path.write_bytes("\ufeffcomp_id,Comp_Name,match_format\r\n1,IPL 2025,T20\r\n".encode("utf-8"))

    header = read_header(path)
    missing = missing_columns(header, ["comp_id", "comp_name", "match_format", "season"])
    message = header_failure("competitions.csv", path, header, missing)

    assert header == ["comp_id", "Comp_Name", "match_format"]
    assert missing == ["comp_name", "season"]
    assert message.startswith(f"competitions.csv ({path}): ")
    assert ": comp_name, season." in message
