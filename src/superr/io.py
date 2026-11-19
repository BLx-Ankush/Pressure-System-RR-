"""Data_Loader: read the six Source_CSVs and check their schema (Requirement 3; Requirement 1,
criterion 2; Requirement 2, criteria 1 and 11).

Design section "Data_Loader (``io.py``)". The ``load`` Pipeline_Stage (``run_stage``) loads the
six files from the paths in the Config and saves each table to its checkpoint; every later stage
reads the tables from there and never opens a Source_CSV itself.

Schema_Registry
---------------
``SCHEMA_REGISTRY`` maps each Source_CSV, by its Config key under ``paths`` (``ball_by_ball``,
``matches``, ``playing_eleven``, ``competitions``, ``teams``, ``mapping``), to its expected
columns in the order of the delivered header, each with a registry type. ``PANDAS_DTYPES`` gives
the dtype each type loads as:

* ``int``: whole numbers, nullable ``Int64``;
* ``float``: finite decimal numbers, nullable ``Float64``;
* ``bool``: the texts ``true`` and ``false``, nullable ``boolean``;
* ``date``: ``%Y-%m-%d`` dates, ``datetime64[us]`` (``NaT`` where missing);
* ``str``: text, pandas ``string`` (``pd.NA`` where missing).

``KEY_ID_COLUMNS`` lists each file's Key_ID_Columns (glossary), which must never be blank.
``FILE_NAMES`` (the runner's ``superr.pipeline.SOURCE_CSVS``) gives each key's file name.

Reading (criteria 6, 9 and 10)
------------------------------
``load_csv`` opens the file read-only in binary mode (``"rb"``) and reads it with
``pandas.read_csv(handle, header=None, dtype=str, keep_default_na=False, na_values=[""])``. Every
field is read as text. An Empty_Field (zero characters between its delimiters) is the only
missing value; every other field stays as written, including ``None``, ``NA``, ``null``, ``0``
and fields that hold only whitespace. The first row is the header row. It is read as data
(``header=None``), so its names are compared exactly as written: pandas does not rename repeated
names, and a data row with more fields than the header row stops the load instead of being
shifted into an index. Standard CSV quoting applies, so a quoted empty field (``""``) is also
missing. A UTF-8 byte-order mark is ignored and blank lines are skipped. A data row with fewer
fields than the header row has its absent fields read as missing values, because pandas cannot
tell them apart from Empty_Fields (the delivered files have no such row).

Header check (criteria 1-3; Requirement 2, criterion 11)
--------------------------------------------------------
Header names match the registry exactly and case-sensitively, in any order. A missing registry
column stops the load, and so does a registry column that appears more than once. Every other
header column gets one warning that names the file and the column, and is left out. The loaded
table holds exactly the registry columns, in registry order, on a RangeIndex.

Conversion (criteria 4, 5 and 11)
---------------------------------
Each registry column is converted strictly; a field that is not an Empty_Field and does not
convert is a failure:

* ``int``: the field must parse with ``pandas.to_numeric`` and its exact value must be a whole
  number in the int64 range. ``12``, ``+12``, `` 12 ``, ``1.0`` and ``1e3`` convert; ``1.5``,
  ``abc``, ``NA``, ``None``, ``inf`` and whitespace-only fields fail.
* ``float``: the field must parse with ``pandas.to_numeric`` to a finite number, so ``nan``,
  ``inf`` and whitespace-only fields fail. ``0`` loads as 0.
* ``bool``: exactly ``true`` (True) or ``false`` (False); ``True``, ``1`` or `` true`` fail.
* ``date``: ``pandas.to_datetime(format="%Y-%m-%d")``; ``2022-02-30``, ``2022/01/05`` and padded
  dates fail.
* ``str``: the text unchanged.

A Key_ID_Column field that is an Empty_Field is a blank Key_ID (criterion 7). A non-empty key
field that fails conversion counts as a failed conversion, not as a blank.

Errors and warnings
-------------------
Every problem with one file is collected and reported together: ``load_csv`` raises one
``LoaderError`` (a ``DataError``, exit code 3) listing the missing and repeated header columns,
the rows that failed conversion in each column (with up to three example values) and the blank
rows in each Key_ID_Column. ``LoaderError.problems`` holds one ``FileProblems`` per file with the
same facts as data. A path that is not an existing, readable file, or a file that is not
well-formed UTF-8 CSV, raises ``LoaderError`` naming the file and the path (criterion 12).
``load_all`` and ``profile`` try all six files first, then raise a single ``LoaderError`` that
covers every file that failed, naming the Config setting of each.

An extra-column warning goes to the ``warn`` callable, which receives the message text. Without
one it is written to standard error. ``run_stage`` passes the Run_Log, which records the warning
with the stage name and, in a Run, echoes it to standard error.

Checkpoint of the ``load`` stage
--------------------------------
One table per Source_CSV, named by its key: ``ball_by_ball``, ``matches``, ``playing_eleven``,
``competitions``, ``teams`` and ``mapping``. A later stage lists ``"load"`` in its ``reads`` and
calls ``ctx.store.load_table("load", "matches")``; the parquet checkpoint keeps every dtype.

Command line
------------
``python -m superr.io --profile`` (from the workspace root, or with ``--root PATH``) loads the
Config with ``load_settings`` and prints, for each Source_CSV, the file name, the number of data
rows (the header row excluded) and the number of columns in the header row (criterion 8). It
exits 0, 2 for a Config problem, or 3 when a Source_CSV cannot be read.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from superr.errors import DataError, PipelineError
from superr.pipeline import SOURCE_CSVS, StageContext
from superr.settings import ConfigError, Settings, load_settings

__all__ = [
    "DATE_FORMAT",
    "FILE_NAMES",
    "KEY_ID_COLUMNS",
    "PANDAS_DTYPES",
    "SCHEMA_REGISTRY",
    "STAGE_NAME",
    "FileProblems",
    "LoaderError",
    "load_all",
    "load_csv",
    "main",
    "profile",
    "run_stage",
]

#: Name of the Pipeline_Stage that runs the Data_Loader, and of its checkpoint.
STAGE_NAME = "load"

#: Config key under ``paths`` -> Source_CSV file name (the runner's list of the six files).
FILE_NAMES: dict[str, str] = SOURCE_CSVS

#: Format of a ``date`` field.
DATE_FORMAT = "%Y-%m-%d"

#: The pandas dtype that each registry type loads as.
PANDAS_DTYPES: dict[str, str] = {
    "int": "Int64",
    "float": "Float64",
    "bool": "boolean",
    "date": "datetime64[us]",
    "str": "string",
}

# --------------------------------------------------------------------------- Schema_Registry
# Columns in the order of each delivered header row (checked against the files on 13 Nov 2026).

SCHEMA_REGISTRY: dict[str, dict[str, str]] = {
    "ball_by_ball": {
        "match_id": "int",
        "innings": "int",
        "overid": "int",
        "over_ball_num": "int",
        "innings_ball_num": "int",
        "batting_team_id": "int",
        "bowling_team_id": "int",
        "striker_id": "int",
        "striker_name": "str",
        "non_striker_id": "int",
        "non_striker_name": "str",
        "bowler_id": "int",
        "bowler_name": "str",
        "batting_type": "str",
        "bowling_type": "str",
        "batting_order": "int",
        "runs": "int",
        "is_four": "bool",
        "is_six": "bool",
        "is_wicket": "bool",
        "dismissal_type_id": "int",
        "dismissal_type": "str",
        "dismissed_player_id": "int",
        "extras_type": "str",  # the text "None" is a value, not a missing field
        "extras1_runs": "int",
        "extras_2_runs": "int",
        "pace_or_spin": "str",
        "bowl_speed": "float",  # 0 means not recorded; it loads as 0 and the Auditor reports it
        "stroke": "str",
        "delivery_style": "str",
        "over_or_around": "str",
        "on_or_off": "str",
        "delivery_ist_time": "str",  # text such as "2022-01-23 00:00:00 UTC"
    },
    "matches": {
        "match_id": "int",
        "comp_id": "int",
        "match_number": "int",
        "match_date": "date",
        "stage": "str",
        "venue_id": "int",
        "venue_country": "str",
        "home_team_id": "int",
        "home_team": "str",
        "away_team_id": "int",
        "away_team": "str",
        "home_team_captain_id": "int",
        "away_team_captain_id": "int",
        "toss_win_team_id": "int",
        "toss_opted": "str",
        "man_of_the_match": "int",
        "first_inn_revised_overs": "int",  # stored in balls
        "second_inn_revised_overs": "int",  # stored in balls
        "revised_target": "int",
        "knock_out": "int",
        "win_team_id": "int",
        "result": "str",
    },
    "playing_eleven": {
        "match_id": "int",
        "team_id": "int",
        "player_id": "int",
        "batting_order": "int",
        "is_played": "bool",
        "is_captain": "bool",  # blank on non-playing squad rows
        "is_wk": "bool",  # blank on non-playing squad rows
        "is_impact": "int",  # 0/1, not "true"/"false"
        "is_impact_off": "int",  # 0/1, not "true"/"false"
    },
    "competitions": {
        "comp_id": "int",
        "comp_name": "str",
        "match_format": "str",
        "season": "str",  # values such as "2022/23"
    },
    "teams": {
        "comp_id": "int",
        "team_id": "int",
        "team_name": "str",
    },
    "mapping": {
        "comp_id": "int",
        "comp_name": "str",
        "team_id": "int",
        "team_name": "str",
        "player_id": "int",
        "player_name": "str",
    },
}

#: The Key_ID_Columns of each Source_CSV (glossary): identifiers that must never be blank.
KEY_ID_COLUMNS: dict[str, list[str]] = {
    "ball_by_ball": ["match_id", "innings", "striker_id", "bowler_id"],
    "matches": ["match_id", "comp_id"],
    "playing_eleven": ["match_id", "team_id", "player_id"],
    "competitions": ["comp_id"],
    "teams": ["comp_id", "team_id"],
    "mapping": ["comp_id", "team_id", "player_id"],
}

_TRUE_TEXT = "true"
_FALSE_TEXT = "false"
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_INT64_DIGITS = 19  # |value| < 10**19 for every int64
_EXAMPLES = 3  # example values or rows shown per column in an error message
_SHOWN_LENGTH = 40  # longest example value shown in full

Warn = Callable[[str], object]


# ---------------------------------------------------------------------------------- problems

@dataclass
class FileProblems:
    """Everything that stops one Source_CSV from loading, as data and as message lines.

    Row numbers count data rows from 1, the header row excluded. ``setting`` is set when the
    path came from the Config (``load_all``, ``profile``), as ``"paths.<key> = '<value>'"``.
    """

    file_key: str
    path: Path
    setting: str | None = None
    unreadable: str | None = None  # why the file cannot be opened or read as CSV text
    missing_columns: list[str] = field(default_factory=list)  # registry columns absent from the header
    repeated_columns: dict[str, int] = field(default_factory=dict)  # registry column -> times in the header
    failed_conversions: dict[str, int] = field(default_factory=dict)  # column -> rows that failed conversion
    conversion_examples: dict[str, list[tuple[int, str]]] = field(default_factory=dict)  # (row, text)
    blank_key_ids: dict[str, int] = field(default_factory=dict)  # Key_ID_Column -> rows that are blank
    blank_key_rows: dict[str, list[int]] = field(default_factory=dict)  # first blank rows

    @property
    def file_name(self) -> str:
        return FILE_NAMES[self.file_key]

    @property
    def has_problems(self) -> bool:
        return bool(
            self.unreadable
            or self.missing_columns
            or self.repeated_columns
            or self.failed_conversions
            or self.blank_key_ids
        )

    def lines(self) -> list[str]:
        """The message lines for this file: a heading naming the file and path, then one line per problem."""
        where = f"path {self.path}" if self.setting is None else f"Config setting {self.setting}, path {self.path}"
        lines = [f"{self.file_name} ({where}):"]
        if self.unreadable is not None:
            lines.append(f"  - {self.unreadable}")
        if self.missing_columns:
            lines.append(
                f"  - the header row lacks {_count(len(self.missing_columns), 'expected column')}: "
                f"{', '.join(self.missing_columns)} (Requirement 3, criterion 2)"
            )
        for column, times in self.repeated_columns.items():
            lines.append(
                f"  - column {column} appears {times} times in the header row; each Schema_Registry "
                f"column must appear exactly once"
            )
        types = SCHEMA_REGISTRY[self.file_key]
        for column, rows in self.failed_conversions.items():
            shown = "; ".join(f"row {row}: {_shown(text)}" for row, text in self.conversion_examples.get(column, []))
            lines.append(
                f"  - column {column}: {_count(rows, 'row')} failed conversion to {types[column]} "
                f"(for example {shown}) (Requirement 3, criterion 11)"
            )
        for column, rows in self.blank_key_ids.items():
            first = self.blank_key_rows.get(column, [])
            shown = ("row " if len(first) == 1 else "rows ") + ", ".join(str(row) for row in first)
            lines.append(
                f"  - Key_ID_Column {column}: {_count(rows, 'row')} with a missing value "
                f"(for example {shown}) (Requirement 3, criterion 7)"
            )
        return lines


class LoaderError(DataError):
    """One or more Source_CSVs cannot be loaded (a DataError, exit code 3).

    ``problems`` holds one FileProblems per failing file, in Source_CSV order; the message lists
    every problem of every one of them.
    """

    def __init__(self, problems: Iterable[FileProblems]) -> None:
        self.problems: tuple[FileProblems, ...] = tuple(problems)
        files = len(self.problems)
        heading = f"the Data_Loader cannot load {_count(files, 'Source_CSV')}:"
        super().__init__("\n".join([heading, *(line for problem in self.problems for line in problem.lines())]))


def _count(number: int, noun: str) -> str:
    return f"{number:,} {noun}" if number == 1 else f"{number:,} {noun}s"


def _shown(text: str) -> str:
    return repr(text) if len(text) <= _SHOWN_LENGTH else repr(text[:_SHOWN_LENGTH]) + "..."


def _write_stderr(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------------- reading

def _schema(file_key: str) -> dict[str, str]:
    try:
        return SCHEMA_REGISTRY[file_key]
    except (KeyError, TypeError):
        raise ValueError(f"unknown Source_CSV key {file_key!r}; expected one of {', '.join(SCHEMA_REGISTRY)}") from None


def _read_text(path: Path, file_key: str) -> tuple[list[str], pd.DataFrame]:
    """The header names and the data rows (as text, columns numbered from 0) of one CSV file."""

    def unreadable(reason: str) -> LoaderError:
        return LoaderError([FileProblems(file_key, path, unreadable=reason)])

    not_a_file = "the path does not point to an existing, readable file"
    if path.is_dir():
        raise unreadable(f"{not_a_file}: it is a folder (Requirement 3, criterion 12)")
    try:
        with open(path, "rb") as handle:  # read-only (Requirement 3, criterion 9)
            raw = pd.read_csv(
                handle,
                header=None,
                dtype=str,
                keep_default_na=False,
                na_values=[""],
                encoding="utf-8",
                engine="c",
            )
    except FileNotFoundError:
        raise unreadable(f"{not_a_file}: no file exists there (Requirement 3, criterion 12)") from None
    except OSError as exc:
        raise unreadable(f"{not_a_file}: {exc.strerror or exc} (Requirement 3, criterion 12)") from None
    except pd.errors.EmptyDataError:
        raise unreadable("the file is empty, so it has no header row") from None
    except pd.errors.ParserError as exc:
        raise unreadable(f"the file is not well-formed CSV: {exc}") from None
    except UnicodeDecodeError as exc:
        raise unreadable(f"the file is not UTF-8 text: {exc}") from None
    header = ["" if pd.isna(name) else str(name) for name in raw.iloc[0].tolist()]
    return header, raw.iloc[1:].reset_index(drop=True)


# -------------------------------------------------------------------------------- conversion
# Each converter takes one column as read (text, missing only at Empty_Fields) and returns the
# converted values and a bool array that is True where a non-empty field failed conversion.

def _whole_number(text: str) -> int | None:
    """The exact int64 value of a numeric text, or None when it is not a whole number in range."""
    try:
        number = Decimal(text)
    except (InvalidOperation, ValueError):
        return None
    if not number.is_finite() or number.adjusted() >= _INT64_DIGITS or number != number.to_integral_value():
        return None
    value = int(number)
    return value if _INT64_MIN <= value <= _INT64_MAX else None


def _to_int(text: pd.Series) -> tuple[pd.api.extensions.ExtensionArray, np.ndarray]:
    present = text.notna().to_numpy(dtype=bool)
    numbers = pd.to_numeric(text, errors="coerce", dtype_backend="numpy_nullable")
    if pd.api.types.is_signed_integer_dtype(numbers.dtype):
        # to_numeric found only plain whole numbers inside the int64 range.
        valid = numbers.notna().to_numpy(dtype=bool)
        values = numbers.to_numpy(dtype=np.int64, na_value=0)
    else:
        # Some value is written as a decimal or exponent (1.0, 1e3) or lies outside the int64
        # range, so to_numeric used floats: decide each distinct text exactly instead.
        parsed = numbers.notna().to_numpy(dtype=bool)
        codes, uniques = pd.factorize(text, use_na_sentinel=True)
        whole = np.zeros(len(uniques) + 1, dtype=bool)  # the extra last slot serves code -1
        exact = np.zeros(len(uniques) + 1, dtype=np.int64)
        for index, value in enumerate(uniques):
            number = _whole_number(str(value))
            if number is not None:
                whole[index] = True
                exact[index] = number
        valid = parsed & whole[codes]
        values = np.where(valid, exact[codes], 0)
    return pd.arrays.IntegerArray(values.astype(np.int64, copy=False), ~valid), present & ~valid


def _to_float(text: pd.Series) -> tuple[pd.api.extensions.ExtensionArray, np.ndarray]:
    present = text.notna().to_numpy(dtype=bool)
    numbers = pd.to_numeric(text, errors="coerce", dtype_backend="numpy_nullable")
    values = numbers.astype("Float64").to_numpy(dtype=np.float64, na_value=np.nan)
    valid = np.isfinite(values)
    return pd.arrays.FloatingArray(np.where(valid, values, 0.0), ~valid), present & ~valid


def _to_bool(text: pd.Series) -> tuple[pd.api.extensions.ExtensionArray, np.ndarray]:
    present = text.notna().to_numpy(dtype=bool)
    is_true = text.eq(_TRUE_TEXT).to_numpy(dtype=bool, na_value=False)
    is_false = text.eq(_FALSE_TEXT).to_numpy(dtype=bool, na_value=False)
    valid = is_true | is_false
    return pd.arrays.BooleanArray(is_true, ~valid), present & ~valid


def _to_date(text: pd.Series) -> tuple[pd.api.extensions.ExtensionArray, np.ndarray]:
    present = text.notna().to_numpy(dtype=bool)
    dates = pd.to_datetime(text, format=DATE_FORMAT, errors="coerce").astype(PANDAS_DTYPES["date"])
    valid = dates.notna().to_numpy(dtype=bool)
    return dates.array, present & ~valid


def _to_str(text: pd.Series) -> tuple[pd.api.extensions.ExtensionArray, np.ndarray]:
    return text.astype(PANDAS_DTYPES["str"]).array, np.zeros(len(text), dtype=bool)


_CONVERTERS: dict[str, Callable[[pd.Series], tuple[pd.api.extensions.ExtensionArray, np.ndarray]]] = {
    "int": _to_int,
    "float": _to_float,
    "bool": _to_bool,
    "date": _to_date,
    "str": _to_str,
}


# ------------------------------------------------------------------------------- public API

def load_csv(path: Path | str, file_key: str, *, warn: Warn | None = None) -> pd.DataFrame:
    """Load one Source_CSV and check it against its Schema_Registry entry.

    ``file_key`` is the file's key in ``SCHEMA_REGISTRY`` (for example ``"ball_by_ball"``).
    Returns exactly the registry columns, in registry order, with the dtypes of
    ``PANDAS_DTYPES``, on a RangeIndex. Calls ``warn`` (default: write to standard error) once
    per extra header column, then leaves the column out. Raises LoaderError (exit code 3)
    listing every problem of the file; ValueError for an unknown ``file_key``.
    """
    schema = _schema(file_key)
    source = Path(path)
    header, data = _read_text(source, file_key)
    report = FileProblems(file_key, source)
    counts = Counter(header)
    report.missing_columns = [column for column in schema if counts[column] == 0]
    report.repeated_columns = {column: counts[column] for column in schema if counts[column] > 1}
    say = _write_stderr if warn is None else warn
    for name in dict.fromkeys(header):
        if name not in schema:
            times = "" if counts[name] == 1 else f" ({counts[name]} times)"
            say(
                f"{FILE_NAMES[file_key]}: the header row has column {name!r}{times}, which the Schema_Registry "
                f"does not list for this file; the column is left out of the loaded table "
                f"(Requirement 3, criterion 3)"
            )
    position = {name: index for index, name in enumerate(header) if name in schema and counts[name] == 1}
    columns: dict[str, Any] = {}
    for column, kind in schema.items():
        if column not in position:
            continue
        text = data[position[column]]
        values, failed = _CONVERTERS[kind](text)
        if failed.any():
            rows = np.flatnonzero(failed)
            report.failed_conversions[column] = len(rows)
            report.conversion_examples[column] = [(int(row) + 1, str(text.iat[row])) for row in rows[:_EXAMPLES]]
        columns[column] = values
    for column in KEY_ID_COLUMNS[file_key]:
        if column not in position:
            continue
        blank = data[position[column]].isna().to_numpy(dtype=bool)
        if blank.any():
            rows = np.flatnonzero(blank)
            report.blank_key_ids[column] = len(rows)
            report.blank_key_rows[column] = [int(row) + 1 for row in rows[:_EXAMPLES]]
    if report.has_problems:
        raise LoaderError([report])
    return pd.DataFrame(columns, copy=False)


def _with_setting(exc: LoaderError, settings: Settings, file_key: str) -> tuple[FileProblems, ...]:
    setting = f"paths.{file_key} = {settings.get(f'paths.{file_key}')!r}"
    for problem in exc.problems:
        problem.setting = setting
    return exc.problems


def load_all(settings: Settings, *, warn: Warn | None = None) -> dict[str, pd.DataFrame]:
    """Load all six Source_CSVs from their Config paths, keyed as in ``SCHEMA_REGISTRY``.

    Every file is attempted. If any fails, one LoaderError (exit code 3) lists every problem of
    every failing file, each with its Config setting and path. ``warn`` is as in load_csv.
    """
    tables: dict[str, pd.DataFrame] = {}
    problems: list[FileProblems] = []
    for key in SCHEMA_REGISTRY:
        try:
            tables[key] = load_csv(settings.path(key), key, warn=warn)
        except LoaderError as exc:
            problems.extend(_with_setting(exc, settings, key))
    if problems:
        raise LoaderError(problems)
    return tables


def profile(settings: Settings) -> list[tuple[str, int, int]]:
    """(file name, data rows without the header row, columns in the header row) of each Source_CSV.

    Reads each file from its Config path as load_csv does, without the schema check. Raises one
    LoaderError naming every file that cannot be read.
    """
    rows: list[tuple[str, int, int]] = []
    problems: list[FileProblems] = []
    for key in SCHEMA_REGISTRY:
        try:
            header, data = _read_text(settings.path(key), key)
        except LoaderError as exc:
            problems.extend(_with_setting(exc, settings, key))
            continue
        rows.append((FILE_NAMES[key], len(data), len(header)))
    if problems:
        raise LoaderError(problems)
    return rows


def run_stage(ctx: StageContext) -> None:
    """The ``load`` Pipeline_Stage: load the six Source_CSVs and save each table to the checkpoint."""
    log = ctx.log
    tables = load_all(ctx.settings, warn=lambda message: log.warning(message, stage=STAGE_NAME))
    for key, table in tables.items():
        ctx.store.save_table(STAGE_NAME, key, table)
        log.info(
            f"{FILE_NAMES[key]}: loaded {_count(len(table), 'row')} and {_count(table.shape[1], 'column')}; "
            f"saved as table {key!r}",
            stage=STAGE_NAME,
        )


# ------------------------------------------------------------------------------ command line

def main(argv: Sequence[str] | None = None) -> int:
    """``python -m superr.io --profile [--root PATH]``; returns the exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m superr.io",
        description="Data_Loader of the Pressure Rating System.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="print each Source_CSV's file name, data rows (header row excluded) and header columns",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="workspace root that holds config.py (default: the current folder)",
    )
    args = parser.parse_args(None if argv is None else list(argv))
    if not args.profile:
        parser.error("nothing to do: pass --profile")
    root = Path.cwd() if args.root is None else args.root
    try:
        settings = load_settings(root)
    except ConfigError as exc:
        print(f"{parser.prog}: the Config cannot be used: {exc}", file=sys.stderr)
        return 2
    try:
        rows = profile(settings)
    except PipelineError as exc:
        print(f"{parser.prog}: {exc}", file=sys.stderr)
        return exc.exit_code
    heading = ("Source_CSV", "data rows", "header columns")
    width = max(len(heading[0]), *(len(name) for name, _, _ in rows))
    print(f"{heading[0]:<{width}}  {heading[1]:>10}  {heading[2]:>14}")
    for name, data_rows, header_columns in rows:
        print(f"{name:<{width}}  {data_rows:>10}  {header_columns:>14}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
