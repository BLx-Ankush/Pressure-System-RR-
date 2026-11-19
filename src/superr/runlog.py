"""Run_Log: the record of one Run under ``outputs/logs/`` (Requirement 1, criteria 7 and 9;
Requirement 28, criteria 4 and 6).

A RunLog names its files after the Run's start time in UTC, ``<UTC>`` = ``YYYYMMDDTHHMMSSZ``
(no colons, so the names are valid on Windows). A later Run that starts in the same second gets
``<UTC>_2``, then ``<UTC>_3``, and so on.

* ``run_<UTC>.log``: human-readable lines ``<UTC time>  <LEVEL>  [<stage>] <message>``, each
  appended to the file as it happens. Continuation lines of a multi-line message are indented.
* ``run_<UTC>.json``: the machine-readable record (layout below). It is rewritten in full after
  every update, through a temporary file renamed in place, so a Run that crashes or is killed
  still leaves valid JSON holding everything up to its last update, with ``status`` "running".
* ``warnings_<UTC>.csv``: the header ``utc,stage,message`` and one row per warning, rewritten the
  same way, so the report can list the warnings.

A RunLog creates its folder when it is constructed and writes nothing outside it. The runner
passes ``<outputs>/logs`` (``LOG_FOLDER``). With ``echo=sys.stderr`` every log line also goes to
standard error, so a failing stage's name and error message (Requirement 28, criterion 6) and the
path of each changed Protected_File (Requirement 1, criterion 9) reach the console too.

JSON layout (``format_version`` 1)::

    run_id, status ("running" | "completed" | "failed"), exit_code, started_utc, ended_utc,
    seconds, root, argv, files {log, json, warnings},
    python {version, implementation, build, executable, platform},
    packages [{name, pinned, installed}],
    config {path, sha256},
    source_csvs [{name, path, sha256}],
    protected_files {start [{path, sha256}], end [{path, sha256, changed}], changed [path]},
    current_stage, failed_stage,
    stages [{name, status, seconds, utc}],
    warnings, errors, notes [{utc, stage, message}]

Paths under ``root`` are written relative to it, with ``/``. The three ``protected_files`` parts
stay null until recorded, so a null ``end`` shows that the end-of-Run check never ran. Times are
UTC in ISO 8601 with microseconds, such as ``2026-11-12T15:30:12.123456Z``.

If a JSON or CSV rewrite fails during the Run (a viewer holding the file open on Windows, for
example), the failure is written to the ``.log`` and the file is rewritten at the next update;
``finish`` raises if its final rewrite fails. Errors writing the ``.log`` itself always raise.

Typical use by the runner::

    log = RunLog(outputs / LOG_FOLDER, root=root, argv=argv, echo=sys.stderr)
    log.environment(pinned_packages(root / "requirements.txt"))
    log.config_hash(settings.sha256(), settings.source_path)
    log.source_csv("ball_by_ball", path, protect.sha256_file(path))      # each Source_CSV
    start = protect.hash_protected(root, protect.PROTECTED_NAMES)
    log.protected_start(start)
    log.start_stage("load"); ...; log.stage("load", seconds, COMPLETED)   # each stage
    log.error(f"{type(exc).__name__}: {exc}", stage="audit")              # a failing stage
    after = protect.rehash(start)                                         # in finally
    log.protected_end(after, protect.changed_files(start, after))
    log.finish(exit_code)
"""

from __future__ import annotations

import copy
import csv
import importlib.metadata
import io
import json
import math
import numbers
import os
import platform
import re
import sys
import threading
import time
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

__all__ = [
    "COMPLETED",
    "FAILED",
    "FORMAT_VERSION",
    "LOG_FOLDER",
    "RUNNING",
    "WARNING_COLUMNS",
    "RunLog",
    "pinned_packages",
    "python_info",
    "utc_stamp",
]

#: Status of a Run that has not finished, and of a finished Run or stage.
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"

#: Folder under outputs/ that holds the Run_Logs.
LOG_FOLDER = "logs"
#: Header of warnings_<UTC>.csv.
WARNING_COLUMNS: tuple[str, ...] = ("utc", "stage", "message")
#: Version of the JSON layout described in the module docstring.
FORMAT_VERSION = 1

_STAMP_FORMAT = "%Y%m%dT%H%M%SZ"
_LINE_TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
_SAME_SECOND_LIMIT = 1000
_REPLACE_ATTEMPTS = 5
_REPLACE_PAUSE = 0.05  # seconds; the wait before each retry grows by this much
_SHA256 = re.compile(r"[0-9a-f]{64}")
_BARE_VERSION = re.compile(r"[0-9A-Za-z][0-9A-Za-z.+!_-]*")
_REQUIREMENT = re.compile(r"([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)\s*(?:\[[^\]]*\])?\s*(.*)", re.S)
_EXACT_PIN = re.compile(r"===?\s*([^\s,;*]+)")


# ------------------------------------------------------------------------------ time stamps

def _now() -> datetime:
    return datetime.now(timezone.utc)


def utc_stamp(moment: datetime) -> str:
    """File-name-safe UTC time stamp ``YYYYMMDDTHHMMSSZ`` (no colons) of an aware datetime."""
    return moment.astimezone(timezone.utc).strftime(_STAMP_FORMAT)


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ------------------------------------------------------------------------ environment facts

def python_info() -> dict[str, str]:
    """Version and build of the running Python interpreter."""
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "build": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
    }


def _requirement(text: str) -> tuple[str, str] | None:
    """(name, pin) of one PEP 508 requirement; None for a blank or option line."""
    text = text.split(";", 1)[0].strip()  # environment markers do not change the pin
    if not text or text.startswith("-"):
        return None
    match = _REQUIREMENT.fullmatch(text)
    if match is None:
        return None
    name, spec = match.group(1), match.group(2).strip()
    exact = _EXACT_PIN.fullmatch(spec)
    return name, exact.group(1) if exact else spec


def pinned_packages(path: Path | str) -> dict[str, str]:
    """Pin of each package listed in a dependency file, as name -> pin in file order.

    Reads ``requirements.txt`` format, or ``[project].dependencies`` when the file name ends in
    ``.toml``. An exact pin ``name==1.2.3`` gives ``"1.2.3"``. Any other requirement gives its
    version specifier as written (empty when it has none), so the Run_Log shows it unchanged.
    Comments, blank lines, option lines (``-r``, ``-e``) and per-line ``--`` options are skipped.
    """
    source = Path(path)
    if source.suffix.lower() == ".toml":
        with open(source, "rb") as handle:
            texts = [str(item) for item in tomllib.load(handle).get("project", {}).get("dependencies", [])]
    else:
        texts = []
        for line in source.read_text(encoding="utf-8").splitlines():
            line = re.split(r"(?:^|\s)#", line, maxsplit=1)[0]  # a comment starts at "#" after a space
            texts.append(re.split(r"\s--", line, maxsplit=1)[0])  # per-line options such as --hash
    pins: dict[str, str] = {}
    for text in texts:
        parsed = _requirement(text)
        if parsed is not None:
            pins[parsed[0]] = parsed[1]
    return pins


# --------------------------------------------------------------------------- input checks

def _check_text(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} must be a non-empty string, got {value!r}")
    return value


def _check_digest(value: Any, what: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{what} must be a lowercase hex SHA-256 digest, got {value!r}")
    return value


def _check_seconds(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"stage seconds must be a number, got {value!r}")
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"stage seconds must be finite and not negative, got {value!r}")
    return seconds


# ----------------------------------------------------------------------------- file output

def _entry(moment: datetime, stage: str | None, message: str) -> dict[str, Any]:
    return {"utc": _iso(moment), "stage": stage, "message": message}


def _json_text(record: Mapping[str, Any]) -> str:
    return json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def _csv_text(warnings: Iterable[Mapping[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(WARNING_COLUMNS)
    for warning in warnings:
        writer.writerow([warning["utc"], warning["stage"] or "", warning["message"]])
    return buffer.getvalue()


def _replace_text(path: Path, text: str) -> None:
    """Write UTF-8 text through ``<name>.tmp`` renamed onto path, retrying a briefly locked target."""
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    for attempt in range(1, _REPLACE_ATTEMPTS + 1):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass  # a leftover .tmp file in the log folder is harmless
                raise
            time.sleep(_REPLACE_PAUSE * attempt)


def _claim(folder: Path, stamp: str) -> tuple[str, Path]:
    """First free Run id for stamp, claimed by creating its empty .log file exclusively."""
    for number in range(1, _SAME_SECOND_LIMIT + 1):
        run_id = stamp if number == 1 else f"{stamp}_{number}"
        if (folder / f"run_{run_id}.json").exists() or (folder / f"warnings_{run_id}.csv").exists():
            continue
        log_path = folder / f"run_{run_id}.log"
        try:
            with open(log_path, "x", encoding="utf-8"):
                pass
        except FileExistsError:
            continue
        return run_id, log_path
    raise FileExistsError(f"{folder} already holds {_SAME_SECOND_LIMIT} Run_Logs started at {stamp}")


def _echo(stream: TextIO, text: str) -> None:
    try:
        try:
            stream.write(text)
        except UnicodeEncodeError:
            encoding = getattr(stream, "encoding", None) or "ascii"
            stream.write(text.encode(encoding, "backslashreplace").decode(encoding))
        stream.flush()
    except (OSError, ValueError, LookupError):
        pass  # the log files are the record; a closed or broken console must not stop the Run


# ----------------------------------------------------------------------------------- RunLog

class RunLog:
    """The Run_Log of one Run: ``run_<UTC>.log``, ``run_<UTC>.json`` and ``warnings_<UTC>.csv``.

    The constructor creates ``log_dir`` and the three files and records the start time and the
    Python version. Every later call adds to the record, appends to the ``.log`` and rewrites the
    JSON file. ``root`` (the workspace root) makes recorded paths relative; ``argv`` holds the
    command-line arguments to record; ``echo`` is a text stream that also receives every line.
    """

    def __init__(
        self,
        log_dir: Path | str,
        *,
        root: Path | str | None = None,
        argv: Sequence[str] | None = None,
        echo: TextIO | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._echo = echo
        self._root = None if root is None else Path(root).resolve()
        self._clock = time.perf_counter()
        started = _now()
        folder = Path(log_dir)
        folder.mkdir(parents=True, exist_ok=True)
        self._run_id, self._log_path = _claim(folder, utc_stamp(started))
        self._json_path = folder / f"run_{self._run_id}.json"
        self._warnings_path = folder / f"warnings_{self._run_id}.csv"
        self._csv_dirty = True
        arguments = None if argv is None else [str(arg) for arg in argv]
        self._record: dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            "run_id": self._run_id,
            "status": RUNNING,
            "exit_code": None,
            "started_utc": _iso(started),
            "ended_utc": None,
            "seconds": None,
            "root": None if self._root is None else str(self._root),
            "argv": arguments,
            "files": {
                "log": self._log_path.name,
                "json": self._json_path.name,
                "warnings": self._warnings_path.name,
            },
            "python": python_info(),
            "packages": [],
            "config": None,
            "source_csvs": [],
            "protected_files": {"start": None, "end": None, "changed": None},
            "current_stage": None,
            "failed_stage": None,
            "stages": [],
            "warnings": [],
            "errors": [],
            "notes": [],
        }
        first = f"run {self._run_id} started"
        if self._root is not None:
            first += f" in {self._root}"
        if arguments is not None:
            first += f"; arguments: {' '.join(arguments) or 'none'}"
        python = self._record["python"]
        second = f"python {python['version']} ({python['implementation']}) at {python['executable']}"
        with self._lock:
            self._write_lines("INFO", None, [first, second], started)
            self._flush(final=True)

    def __repr__(self) -> str:
        return f"RunLog(run_id={self._run_id!r}, status={self.status!r}, folder={str(self._log_path.parent)!r})"

    # -- read-only views -----------------------------------------------------------------------

    @property
    def run_id(self) -> str:
        """``<UTC>`` (or ``<UTC>_<n>``), shared by the three file names."""
        return self._run_id

    @property
    def log_path(self) -> Path:
        return self._log_path

    @property
    def json_path(self) -> Path:
        return self._json_path

    @property
    def warnings_path(self) -> Path:
        return self._warnings_path

    @property
    def status(self) -> str:
        """RUNNING until finish(), then COMPLETED or FAILED."""
        with self._lock:
            return self._record["status"]

    @property
    def exit_code(self) -> int | None:
        with self._lock:
            return self._record["exit_code"]

    @property
    def record(self) -> dict[str, Any]:
        """Deep copy of the JSON record."""
        with self._lock:
            return copy.deepcopy(self._record)

    @property
    def warnings(self) -> list[dict[str, Any]]:
        """Deep copy of the warnings recorded so far, as {utc, stage, message}."""
        with self._lock:
            return copy.deepcopy(self._record["warnings"])

    # -- environment and inputs ------------------------------------------------------------------

    def environment(self, packages: Mapping[str, str | None] | Iterable[str]) -> None:
        """Record the Python version and each package's installed version (Requirement 28, criterion 4).

        ``packages`` is package names, or name -> pin as returned by ``pinned_packages``. A
        package that is not installed, or whose installed version differs from its exact pin,
        is also recorded as a warning.
        """
        if isinstance(packages, (str, bytes)):
            raise TypeError("packages must be package names or a name -> pin mapping, not one string")
        pins = dict(packages) if isinstance(packages, Mapping) else dict.fromkeys(packages)
        entries = []
        for name, pinned in pins.items():
            _check_text(name, "package name")
            try:
                installed: str | None = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                installed = None
            entries.append({"name": name, "pinned": None if pinned is None else str(pinned), "installed": installed})
        with self._lock:
            moment = _now()
            self._record["python"] = python_info()
            self._record["packages"] = entries
            lines, problems = [], []
            for entry in entries:
                name, pinned, installed = entry["name"], entry["pinned"], entry["installed"]
                pin_text = f" (pinned {pinned})" if pinned else ""
                lines.append(f"package {name} {installed or 'not installed'}{pin_text}")
                if installed is None:
                    problems.append(f"package {name} is not installed{pin_text}")
                elif pinned and _BARE_VERSION.fullmatch(pinned) and installed != pinned:
                    problems.append(f"package {name} {installed} is installed, but the dependency file pins {pinned}")
            self._write_lines("INFO", None, lines or ["no packages listed"], moment)
            for problem in problems:
                self._add_warning(problem, None, moment)
            self._flush()

    def config_hash(self, sha256: str, path: Path | str | None = None) -> None:
        """Record the SHA-256 of the Config file (Requirement 28, criterion 4)."""
        digest = _check_digest(sha256, "Config hash")
        shown = None if path is None else self._show(path)
        with self._lock:
            moment = _now()
            self._record["config"] = {"path": shown, "sha256": digest}
            self._write_lines("INFO", None, [f"config {shown or '(path not given)'} sha256 {digest}"], moment)
            self._flush()

    def source_csv(self, name: str, path: Path | str, sha256: str) -> None:
        """Record the path and SHA-256 of one Source_CSV (Requirement 28, criterion 4).

        ``name`` is the Source_CSV's key, such as ``ball_by_ball``. Recording a name again
        replaces its entry.
        """
        _check_text(name, "Source_CSV name")
        entry = {"name": name, "path": self._show(path), "sha256": _check_digest(sha256, f"hash of Source_CSV {name}")}
        with self._lock:
            moment = _now()
            entries = self._record["source_csvs"]
            for index, old in enumerate(entries):
                if old["name"] == name:
                    entries[index] = entry
                    break
            else:
                entries.append(entry)
            self._write_lines("INFO", None, [f"source_csv {name}: {entry['path']} sha256 {entry['sha256']}"], moment)
            self._flush()

    # -- Protected_Files ---------------------------------------------------------------------

    def protected_start(self, hashes: Mapping[Path, str]) -> None:
        """Record the path and SHA-256 of each Protected_File hashed at the start (Requirement 1, criterion 7).

        ``hashes`` is the result of ``protect.hash_protected``.
        """
        entries = [
            {"path": self._show(path), "sha256": _check_digest(digest, f"hash of {path}")}
            for path, digest in hashes.items()
        ]
        with self._lock:
            moment = _now()
            self._record["protected_files"]["start"] = entries
            lines = [f"protected file (start): {entry['path']} sha256 {entry['sha256']}" for entry in entries]
            lines.append(f"protected files hashed at start: {len(entries)}")
            self._write_lines("INFO", None, lines, moment)
            self._flush()

    def protected_end(self, after: Mapping[Path, str | None], changed: Iterable[Path]) -> None:
        """Record the end-of-Run hashes and each changed Protected_File (Requirement 1, criteria 8 and 9).

        ``after`` is ``protect.rehash(start)`` (None for a missing or unreadable file) and
        ``changed`` is ``protect.changed_files(start, after)``. Each changed path is also recorded
        as an error naming the file, and ``finish`` then records the Run as failed.
        """
        changed_shown = [self._show(path) for path in changed]
        flagged = set(changed_shown)
        entries = []
        for path, digest in after.items():
            shown = self._show(path)
            checked = None if digest is None else _check_digest(digest, f"hash of {path}")
            entries.append({"path": shown, "sha256": checked, "changed": shown in flagged})
        current = {entry["path"]: entry["sha256"] for entry in entries}
        with self._lock:
            moment = _now()
            self._record["protected_files"]["end"] = entries
            self._record["protected_files"]["changed"] = changed_shown
            lines = []
            for entry in entries:
                if entry["sha256"] is None:
                    state = "MISSING OR UNREADABLE"
                else:
                    state = "CHANGED" if entry["changed"] else "unchanged"
                lines.append(f"protected file (end): {entry['path']} sha256 {entry['sha256'] or '-'} {state}")
            lines.append(f"protected files checked at end: {len(entries)}; changed or missing: {len(changed_shown)}")
            self._write_lines("INFO", None, lines, moment)
            for shown in changed_shown:
                what = "is missing or unreadable" if current.get(shown) is None else "was modified"
                self._add_error(f"Protected_File {shown} {what} at the end of the Run", None, moment)
            self._flush()

    # -- stages and messages -------------------------------------------------------------------

    def start_stage(self, name: str) -> None:
        """Record that a Pipeline_Stage started. Messages without a stage are given to it."""
        _check_text(name, "stage name")
        with self._lock:
            moment = _now()
            self._record["current_stage"] = name
            self._write_lines("INFO", None, [f"stage {name} started"], moment)
            self._flush()

    def stage(self, name: str, seconds: float, status: str) -> None:
        """Record a Pipeline_Stage's name, elapsed wall-clock seconds and status (Requirement 28, criterion 4).

        ``status`` is normally COMPLETED or FAILED; the first failed stage becomes ``failed_stage``.
        """
        _check_text(name, "stage name")
        _check_text(status, "stage status")
        elapsed = _check_seconds(seconds)
        with self._lock:
            moment = _now()
            self._record["stages"].append(
                {"name": name, "status": status, "seconds": round(elapsed, 6), "utc": _iso(moment)}
            )
            if self._record["current_stage"] == name:
                self._record["current_stage"] = None
            if status == FAILED and self._record["failed_stage"] is None:
                self._record["failed_stage"] = name
            level = "ERROR" if status == FAILED else "INFO"
            self._write_lines(level, None, [f"stage {name} {status} in {elapsed:.3f} s"], moment)
            self._flush()

    def info(self, message: object, stage: str | None = None) -> None:
        """Record a note, such as a copied file or a page count. ``stage`` defaults to the running stage."""
        with self._lock:
            moment = _now()
            stage_name = self._stage_or_current(stage)
            text = str(message)
            self._record["notes"].append(_entry(moment, stage_name, text))
            self._write_lines("INFO", stage_name, [text], moment)
            self._flush()

    def warning(self, message: object, stage: str | None = None) -> None:
        """Record a warning in the ``.log``, the JSON and ``warnings_<UTC>.csv``.

        ``stage`` defaults to the running stage (None outside a stage).
        """
        with self._lock:
            moment = _now()
            self._add_warning(str(message), self._stage_or_current(stage), moment)
            self._flush()

    def error(self, message: object, stage: str | None = None) -> None:
        """Record an error, such as a failing stage's error message (Requirement 28, criterion 6).

        ``stage`` defaults to the running stage; the first error with a stage sets ``failed_stage``.
        """
        with self._lock:
            moment = _now()
            stage_name = self._stage_or_current(stage)
            self._add_error(str(message), stage_name, moment)
            if stage_name is not None and self._record["failed_stage"] is None:
                self._record["failed_stage"] = stage_name
            self._flush()

    def finish(self, exit_code: int) -> None:
        """Record the end of the Run: status, exit code, end time and total seconds.

        The status is COMPLETED when exit_code is 0 and no Protected_File changed, else FAILED
        (Requirement 1, criterion 9); the runner decides the exit code (Requirement 28,
        criterion 8). Raises OSError when the final JSON or CSV rewrite fails.
        """
        if isinstance(exit_code, bool) or not isinstance(exit_code, int):
            raise TypeError(f"exit_code must be an int, got {exit_code!r}")
        with self._lock:
            moment = _now()
            elapsed = time.perf_counter() - self._clock
            changed = self._record["protected_files"]["changed"]
            status = COMPLETED if exit_code == 0 and not changed else FAILED
            self._record.update(status=status, exit_code=exit_code, ended_utc=_iso(moment), seconds=round(elapsed, 6))
            level = "INFO" if status == COMPLETED else "ERROR"
            self._write_lines(level, None, [f"run {self._run_id} {status} with exit code {exit_code} after {elapsed:.1f} s"], moment)
            self._flush(final=True)

    # -- internals -----------------------------------------------------------------------------

    def _stage_or_current(self, stage: str | None) -> str | None:
        if stage is None:
            return self._record["current_stage"]
        return _check_text(stage, "stage name")

    def _add_warning(self, message: str, stage: str | None, moment: datetime) -> None:
        self._record["warnings"].append(_entry(moment, stage, message))
        self._csv_dirty = True
        self._write_lines("WARNING", stage, [message], moment)

    def _add_error(self, message: str, stage: str | None, moment: datetime) -> None:
        self._record["errors"].append(_entry(moment, stage, message))
        self._write_lines("ERROR", stage, [message], moment)

    def _show(self, path: Path | str) -> str:
        """path relative to the root, with "/", when it lies under the root; else as given."""
        candidate = Path(path)
        if not candidate.is_absolute():
            return candidate.as_posix()
        if self._root is not None:
            try:
                return candidate.relative_to(self._root).as_posix()
            except ValueError:
                pass
            try:
                return candidate.resolve().relative_to(self._root).as_posix()
            except (OSError, ValueError):
                pass
        return str(candidate)

    def _write_lines(self, level: str, stage: str | None, messages: Sequence[str], moment: datetime) -> None:
        """Append one ``.log`` line per message (indented continuation lines) and echo them."""
        head = f"{moment.strftime(_LINE_TIME_FORMAT)}  {level:<7}  "
        indent = " " * len(head)
        prefix = f"[{stage}] " if stage else ""
        lines = []
        for message in messages:
            parts = (prefix + message).replace("\r\n", "\n").replace("\r", "\n").split("\n")
            lines.append(head + parts[0])
            lines.extend(indent + part for part in parts[1:])
        block = "\n".join(lines) + "\n"
        with open(self._log_path, "a", encoding="utf-8", newline="") as handle:
            handle.write(block)
        if self._echo is not None:
            _echo(self._echo, block)

    def _flush(self, *, final: bool = False) -> None:
        """Rewrite the JSON file, and the warnings CSV when a warning was added since its last write."""
        pending = [(self._json_path, _json_text(self._record))]
        if self._csv_dirty:
            pending.append((self._warnings_path, _csv_text(self._record["warnings"])))
        for path, text in pending:
            try:
                _replace_text(path, text)
            except OSError as exc:
                if final:
                    raise
                self._write_lines(
                    "ERROR",
                    None,
                    [f"could not update {path.name} ({type(exc).__name__}: {exc}); it is rewritten at the next update"],
                    _now(),
                )
                continue
            if path == self._warnings_path:
                self._csv_dirty = False
