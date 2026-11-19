"""CheckpointStore: the saved outputs of every Pipeline_Stage (Requirement 28, criteria 5, 6 and 10).

Design sections "Checkpoints and ``--from-stage``" and "Determinism", and decision DD-3. Each
stage saves the tables, arrays, JSON values and models that later stages read; the runner then
calls ``complete``, which writes the stage's ``manifest.json``. A Run started with
``--from-stage X`` reads the checkpoints of the stages before X instead of running them again
(28.5), a failing stage leaves the checkpoints of the stages before it in place (28.6), and a
missing checkpoint can be reported before any stage runs (28.10).

Layout
------
Each stage has one folder, ``<checkpoint root>/<NN>_<stage>/``, where NN is the stage's 1-based
position in run order, as two digits (``01_load``). The checkpoint root is ``outputs/_stages`` for
most stages and ``outputs/final/comparison/_stage`` for ``activity_comparison`` (Requirement 25,
criterion 15). The runner passes every stage's root when it builds the store, so this module
names no stage. Files in a stage folder:

* ``<name>.parquet``: a table (``save_table``), written by pyarrow with zstd level 3.
* ``<name>.npy``: a numpy array (``save_array``), with no pickled objects.
* ``<name>.json``: a JSON value (``save_json``), with sorted keys, in UTF-8.
* ``<name>.model.json`` and one model file (``save_model``), in the library's native format:
  ``<name>.lgb.txt`` for a LightGBM ``Booster`` (model text with every iteration; its
  ``best_iteration`` is kept in ``<name>.model.json``); ``<name>.xgb.json`` for an XGBoost
  ``Booster`` or scikit-learn wrapper such as ``XGBClassifier`` (XGBoost JSON); and
  ``<name>.joblib`` for anything else, such as scikit-learn estimators, isotonic calibrators and
  small wrappers around several models (joblib, zlib level 3). ``<name>.model.json`` records the
  format and the class, and ``load_model`` returns an object of that class.
* ``manifest.json``, written by ``complete``.

Manifest
--------
``complete(stage, inputs=stage.reads)`` hashes every file in the stage folder (except the
manifest and ``*.tmp`` files) and writes::

    {"completed_utc": "2026-11-12T15:30:12.123456Z", "config_sha256": <Config hash>,
     "files": {<path relative to the stage folder>: <SHA-256>},
     "inputs": {<stage read>: <SHA-256 of its manifest.json>},
     "stage": <stage>, "stage_version": <int>}

The manifest is written last, through ``manifest.json.tmp`` renamed in place, so a stage that
failed half-way has none. ``is_complete(stage)`` is True when the manifest is present, valid and
written for that stage, and every file it lists exists; ``incomplete_reason`` says why not.
``manifest`` returns the recorded fields, including ``config_sha256`` for the runner's warning on
a changed Config, and ``manifest_sha256`` the hash that ``inputs`` records.

Rules
-----
* A ``name`` is 1-64 characters from ``a-z``, ``0-9``, ``_`` and ``-``, and does not start with
  ``-``. ``manifest`` and the Windows device names (``con``, ``nul``, ...) are refused. With no
  dots or capitals, two names never map to the same file, also on Windows.
* Saving into a stage whose manifest exists deletes the manifest first, so a stage that is being
  rebuilt never looks complete before ``complete`` runs again.
* Every file is written to ``<file>.tmp`` and then renamed onto its name, so no file is left
  holding a partial write.
* Loading from a stage that has a manifest first checks the file's SHA-256 against it. A file
  that changed after the stage completed, or that the manifest does not list, raises
  ``StageError`` instead of being read. Loading from a stage without a manifest (the stage being
  built) reads the file as it is. A missing or unreadable file raises ``StageError`` too.
* ``clear_all`` deletes every checkpoint root (a full Run, Requirement 28, criterion 1), and
  ``clear_stage`` one stage folder (so that a stage that runs again starts empty). Both refuse a
  folder that does not lie strictly inside the outputs folder; the constructor refuses such roots.
* Bytes are deterministic where the libraries allow it: fixed parquet options, sorted JSON keys
  with no NaN, LightGBM text, XGBoost JSON and ``.npy``. Joblib pickles may differ between Runs.

Round trips
-----------
* A table keeps its dtypes (``str``, ``string``, ``Int32``, ``boolean``, ``float64``, ``int32``,
  ``bool``, ``category``, ``datetime64``) and its index. Column names must be unique strings. A
  cell that holds a list comes back as a numpy array.
* A JSON value may hold None, bool, int, finite float, str, lists, tuples and dicts with str keys.
  Numpy scalars and arrays become Python values, and tuples come back as lists.
* An XGBoost scikit-learn wrapper comes back with its booster, class count and best iteration.
  XGBoost JSON does not store the hyper-parameters that only training reads, so ``get_params``
  shows their defaults after loading.

Exceptions: ``ValueError`` or ``TypeError`` for a programming error (a bad name, an unknown
stage, a value that cannot be saved); ``StageError`` (exit code 5) for a checkpoint that cannot be
used; ``OSError`` from the file system.

Typical use by the runner::

    store = CheckpointStore(
        settings.path("outputs"),
        [(stage.name, root / stage.checkpoint_root) for stage in STAGES],
        config_sha256=settings.effective_sha256(),
    )
    if from_stage is None:
        store.clear_all()                                   # full Run
    else:                                                   # --from-stage, each earlier stage:
        reason = store.incomplete_reason(earlier.name)      # not None: exit 2 naming the stage
        store.manifest(earlier.name)["config_sha256"]       # differs: Run_Log warning
    for stage in stages_to_run:
        store.clear_stage(stage.name)
        stage.run(ctx)
        store.save_json(stage.name, "_constants", ctx.constants.to_dict(stage=stage.name))
        store.complete(stage.name, inputs=stage.reads)
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import math
import os
import re
import shutil
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from superr.errors import PipelineError, StageError
from superr.protect import sha256_file

__all__ = ["MANIFEST_NAME", "CheckpointStore"]

#: File name of the manifest in each stage folder.
MANIFEST_NAME = "manifest.json"

_TMP = ".tmp"
_TABLE = ".parquet"
_ARRAY = ".npy"
_JSON = ".json"
_MODEL_INFO = ".model.json"
#: Model format -> suffix of its model file.
_MODEL_FILES: dict[str, str] = {"lightgbm": ".lgb.txt", "xgboost": ".xgb.json", "joblib": ".joblib"}

_PARQUET_OPTIONS: dict[str, Any] = {
    "engine": "pyarrow",
    "compression": "zstd",
    "compression_level": 3,
    "index": None,  # a RangeIndex is stored as metadata, any other index as columns
}
_JOBLIB_COMPRESS = 3  # zlib level

_STAGE_NAME = re.compile(r"[a-z][a-z0-9_]{0,39}")
_ITEM_NAME = re.compile(r"[a-z0-9_][a-z0-9_-]{0,63}")
_RESERVED_NAMES = frozenset(
    {"manifest", "con", "prn", "aux", "nul"} | {f"{port}{n}" for port in ("com", "lpt") for n in range(1, 10)}
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MANIFEST_KEYS = frozenset({"completed_utc", "config_sha256", "files", "inputs", "stage", "stage_version"})
_JSON_DTYPE_KINDS = "biufU"  # numpy kinds a JSON value can hold: bool, int, uint, float, str
_REPLACE_ATTEMPTS = 5
_REPLACE_PAUSE = 0.05  # seconds; the wait before each retry grows by this much


class CheckpointStore:
    """Checkpoints of the Pipeline_Stages of one Run, in the layout of the module docstring."""

    def __init__(
        self,
        outputs: Path | str,
        layout: Iterable[tuple[str, Path | str]] | Mapping[str, Path | str],
        *,
        config_sha256: str,
    ) -> None:
        """Store for the stages of ``layout``, all inside the outputs folder ``outputs``.

        layout: (stage name, checkpoint root) for every Pipeline_Stage in run order, as pairs or
        as a mapping in run order. Each root is an absolute path strictly inside ``outputs``:
        normally ``<outputs>/_stages``, or ``<outputs>/final/comparison/_stage`` for
        ``activity_comparison``. config_sha256: the hex SHA-256 of the Config, which every
        manifest records; normally ``settings.effective_sha256()``.

        Raises ValueError for a relative path, a root outside ``outputs``, or a malformed or
        repeated stage name. Nothing is written until a stage saves something.
        """
        self._outputs = _absolute(outputs, "the outputs folder")
        self._config_sha256 = _check_digest(config_sha256, "config_sha256")
        pairs = layout.items() if isinstance(layout, Mapping) else layout
        if isinstance(pairs, (str, bytes)):
            raise TypeError("layout must hold (stage name, checkpoint root) pairs, not one string")
        folders: dict[str, Path] = {}
        roots: list[Path] = []
        for position, pair in enumerate(pairs, start=1):
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise TypeError(f"layout entry {pair!r} is not a (stage name, checkpoint root) pair")
            stage, root = pair
            _check_stage_name(stage)
            if stage in folders:
                raise ValueError(f"stage {stage!r} appears more than once in the layout")
            root_path = _absolute(root, f"the checkpoint root of stage {stage!r}")
            self._check_inside(root_path, f"use {root_path} as the checkpoint root of stage {stage!r}")
            folders[stage] = root_path / f"{position:02d}_{stage}"
            if root_path not in roots:
                roots.append(root_path)
        self._folders = folders
        self._positions = {stage: index for index, stage in enumerate(folders)}
        self._roots = tuple(roots)
        self._digests: dict[str, tuple[int, int, str]] = {}  # path -> (size, mtime_ns, sha256)

    def __repr__(self) -> str:
        return f"CheckpointStore(outputs={str(self._outputs)!r}, stages={len(self._folders)})"

    # -- layout ------------------------------------------------------------------------------

    @property
    def outputs(self) -> Path:
        """The outputs folder, as an absolute path."""
        return self._outputs

    @property
    def config_sha256(self) -> str:
        """The Config hash that ``complete`` records in every manifest."""
        return self._config_sha256

    @property
    def stages(self) -> tuple[str, ...]:
        """The stage names of the layout, in run order."""
        return tuple(self._folders)

    def stage_dir(self, stage: str) -> Path:
        """The checkpoint folder ``<root>/<NN>_<stage>`` of a stage; it may not exist yet."""
        return self._folder(stage)

    # -- saving ------------------------------------------------------------------------------

    def save_table(self, stage: str, name: str, df: pd.DataFrame) -> Path:
        """Save a DataFrame as ``<name>.parquet`` and return the file's path.

        Dtypes and the index are kept. Column names must be unique strings (ValueError otherwise).
        """
        _check_name(name)
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"table {name!r} must be a pandas DataFrame, got {type(df).__name__}")
        counts = Counter(column for column in df.columns if isinstance(column, str))
        not_text = [column for column in df.columns if not isinstance(column, str)]
        repeated = sorted(column for column, count in counts.items() if count > 1)
        if not_text or repeated:
            raise ValueError(
                f"table {name!r} needs unique string column names; non-string: {not_text!r}, "
                f"repeated: {repeated!r}"
            )
        path = self._target(stage, name, _TABLE)
        self._write(path, lambda temporary: df.to_parquet(temporary, **_PARQUET_OPTIONS))
        return path

    def save_array(self, stage: str, name: str, array: Any) -> Path:
        """Save a numpy array (or anything ``numpy.asarray`` accepts) as ``<name>.npy``; return its path.

        An array of Python objects raises TypeError, because it would need pickle.
        """
        _check_name(name)
        values = np.asarray(array)
        if values.dtype.hasobject:
            raise TypeError(f"array {name!r} has dtype {values.dtype}; a checkpoint array cannot hold Python objects")
        path = self._target(stage, name, _ARRAY)

        def write(temporary: Path) -> None:
            with open(temporary, "wb") as handle:
                np.save(handle, values, allow_pickle=False)

        self._write(path, write)
        return path

    def save_json(self, stage: str, name: str, obj: Any) -> Path:
        """Save a JSON value, normally a dict, as ``<name>.json`` and return the file's path.

        The value may hold None, bool, int, finite float, str, lists, tuples and dicts with str
        keys; numpy scalars and arrays are converted. Anything else raises TypeError, and NaN or
        infinity raises ValueError, so ``load_json`` returns what was saved (tuples as lists).
        """
        _check_name(name)
        data = _json_bytes(_json_ready(obj, f"JSON value {name!r}"))
        path = self._target(stage, name, _JSON)
        self._write(path, lambda temporary: temporary.write_bytes(data))
        return path

    def save_model(self, stage: str, name: str, model: Any) -> Path:
        """Save a fitted model in its library's native format and return the model file's path.

        LightGBM ``Booster``: model text; XGBoost ``Booster`` or scikit-learn wrapper: XGBoost
        JSON; anything else: joblib. ``<name>.model.json`` records the format and the class, and
        a model file of another format left from an earlier save under this name is deleted.
        """
        _check_name(name)
        kind = _model_format(model)
        cls = type(model)
        info: dict[str, Any] = {"format": kind, "module": cls.__module__, "class": cls.__qualname__}
        payload: bytes | None = None
        if kind == "lightgbm":
            payload = model.model_to_string(num_iteration=-1).encode("utf-8")
            info["best_iteration"] = int(model.best_iteration)
        elif kind == "xgboost":
            booster = model if cls is sys.modules["xgboost"].Booster else model.get_booster()
            payload = bytes(booster.save_raw(raw_format="json"))
        info_bytes = _json_bytes(info)
        path = self._target(stage, name, _MODEL_FILES[kind])
        if payload is None:
            joblib = importlib.import_module("joblib")
            self._write(path, lambda temporary: joblib.dump(model, temporary, compress=_JOBLIB_COMPRESS))
        else:
            data = payload
            self._write(path, lambda temporary: temporary.write_bytes(data))
        for other, suffix in _MODEL_FILES.items():
            if other != kind:
                (path.parent / f"{name}{suffix}").unlink(missing_ok=True)
        self._write(path.parent / f"{name}{_MODEL_INFO}", lambda temporary: temporary.write_bytes(info_bytes))
        return path

    # -- loading -----------------------------------------------------------------------------

    def load_table(self, stage: str, name: str) -> pd.DataFrame:
        """The DataFrame saved as ``<name>.parquet``. StageError when missing, changed or unreadable."""
        path = self._source(stage, name, _TABLE, "table")
        return _read(path, lambda: pd.read_parquet(path, engine="pyarrow"))

    def load_array(self, stage: str, name: str) -> np.ndarray:
        """The array saved as ``<name>.npy``. StageError when missing, changed or unreadable."""
        path = self._source(stage, name, _ARRAY, "array")
        return _read(path, lambda: np.load(path, allow_pickle=False))

    def load_json(self, stage: str, name: str) -> Any:
        """The JSON value saved as ``<name>.json``. StageError when missing, changed or unreadable."""
        path = self._source(stage, name, _JSON, "JSON value")
        return _read(path, lambda: json.loads(path.read_bytes().decode("utf-8")))

    def load_model(self, stage: str, name: str) -> Any:
        """The model saved under ``name``, as an object of the saved class.

        A LightGBM Booster gets its saved ``best_iteration`` back. StageError when a file is
        missing, changed or unreadable, or the model info does not match the model file.
        """
        info_path = self._source(stage, name, _MODEL_INFO, "model")
        info = _read(info_path, lambda: _model_info(json.loads(info_path.read_bytes().decode("utf-8"))))
        path = self._source(stage, name, _MODEL_FILES[info["format"]], "model file")
        return _read(path, lambda: _restore_model(info, path))

    # -- completion --------------------------------------------------------------------------

    def complete(self, stage: str, *, inputs: Iterable[str] = (), stage_version: int = 1) -> dict[str, Any]:
        """Write the stage's ``manifest.json``, last, and return the manifest written.

        inputs: the earlier stages whose checkpoints the stage read (``Stage.reads``). Each must
        run before ``stage`` (ValueError) and be complete (StageError); the SHA-256 of its
        manifest is recorded. stage_version: a non-negative int that the stage code can raise
        when its checkpoint contents change. Every file in the stage folder except the manifest
        and ``*.tmp`` files is hashed and listed; a stage that saved nothing gets an empty list.
        """
        folder = self._folder(stage)
        if isinstance(inputs, (str, bytes)):
            raise TypeError("inputs must be a collection of stage names, not one string")
        version = _check_version(stage_version)
        recorded: dict[str, str] = {}
        for read in sorted(set(inputs)):
            self._folder(read)
            if self._positions[read] >= self._positions[stage]:
                raise ValueError(f"stage {stage!r} cannot read stage {read!r}, which does not run before it")
            reason = self.incomplete_reason(read)
            if reason is not None:
                raise StageError(f"stage {stage!r} reads stage {read!r}, whose checkpoint is not complete: {reason}")
            recorded[read] = self.manifest_sha256(read)
        manifest_path = folder / MANIFEST_NAME
        manifest_path.unlink(missing_ok=True)
        folder.mkdir(parents=True, exist_ok=True)
        files: dict[str, str] = {}
        for path in sorted(folder.rglob("*")):
            if path == manifest_path or path.name.endswith(_TMP) or not path.is_file():
                continue
            files[path.relative_to(folder).as_posix()] = self._digest(path, fresh=True)
        manifest = {
            "stage": stage,
            "stage_version": version,
            "completed_utc": _utc_now(),
            "config_sha256": self._config_sha256,
            "inputs": recorded,
            "files": files,
        }
        data = _json_bytes(manifest)
        self._write(manifest_path, lambda temporary: temporary.write_bytes(data))
        return json.loads(data)

    def incomplete_reason(self, stage: str) -> str | None:
        """None when the stage's checkpoint is complete; otherwise the reason, naming the file."""
        folder = self._folder(stage)
        try:
            manifest, _ = _load_manifest(folder / MANIFEST_NAME, stage)
        except _Unusable as problem:
            return str(problem)
        for relative in manifest["files"]:
            if not (folder / relative).is_file():
                return f"{folder / relative} is listed in {MANIFEST_NAME} but does not exist"
        return None

    def is_complete(self, stage: str) -> bool:
        """True when the stage has a valid manifest and every file it lists exists."""
        return self.incomplete_reason(stage) is None

    def manifest(self, stage: str) -> dict[str, Any]:
        """The fields of the stage's manifest (a new dict). StageError when it is absent or invalid."""
        return self._manifest(stage)[0]

    def manifest_sha256(self, stage: str) -> str:
        """SHA-256 of the stage's ``manifest.json`` bytes. StageError when it is absent or invalid."""
        return hashlib.sha256(self._manifest(stage)[1]).hexdigest()

    # -- clearing ----------------------------------------------------------------------------

    def clear_stage(self, stage: str) -> bool:
        """Delete the stage's folder with its manifest and files; True when there was a folder."""
        return self._delete(self._folder(stage))

    def clear_all(self) -> list[Path]:
        """Delete every checkpoint root of the layout, for a full Run; return the folders deleted.

        Other files and folders under the outputs folder are left alone.
        """
        deleted = []
        for root in sorted(self._roots, key=lambda path: len(path.parts)):
            if self._delete(root):
                deleted.append(root)
        return deleted

    # -- internals ---------------------------------------------------------------------------

    def _folder(self, stage: str) -> Path:
        try:
            return self._folders[stage]
        except (KeyError, TypeError):
            known = ", ".join(self._folders) or "(none)"
            raise ValueError(f"{stage!r} is not a Pipeline_Stage of this store; the stages are: {known}") from None

    def _check_inside(self, path: Path, action: str) -> None:
        """ValueError unless path, with links resolved, lies strictly inside the outputs folder."""
        outputs = self._outputs.resolve()
        resolved = path.resolve()
        if resolved == outputs or not resolved.is_relative_to(outputs):
            raise ValueError(f"refusing to {action}: {path} is not strictly inside the outputs folder {self._outputs}")

    def _delete(self, folder: Path) -> bool:
        self._check_inside(folder, f"delete {folder}")
        self._digests.clear()
        if not os.path.lexists(folder):
            return False
        shutil.rmtree(folder)
        return True

    def _target(self, stage: str, name: str, suffix: str) -> Path:
        """Path of a file to save; deletes the stage's manifest first and creates its folder."""
        folder = self._folder(stage)
        (folder / MANIFEST_NAME).unlink(missing_ok=True)
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{name}{suffix}"

    def _write(self, path: Path, write: Callable[[Path], object]) -> None:
        """Write through ``<file>.tmp`` renamed onto path; the temporary file never survives a failure."""
        temporary = path.with_name(path.name + _TMP)
        self._digests.pop(os.fspath(path), None)
        try:
            write(temporary)
            _replace(temporary, path)
        except BaseException:
            with contextlib.suppress(OSError):
                temporary.unlink(missing_ok=True)
            raise

    def _source(self, stage: str, name: str, suffix: str, kind: str) -> Path:
        """Path of a saved file, checked against the stage's manifest when it has one."""
        _check_name(name)
        folder = self._folder(stage)
        path = folder / f"{name}{suffix}"
        if not path.is_file():
            raise StageError(
                f"stage {stage!r} has no saved {kind} {name!r}: {path} does not exist; "
                f"run stage {stage!r} again or start a full Run"
            )
        if not (folder / MANIFEST_NAME).exists():
            return path  # the stage is still being built: there is nothing to check against
        manifest, _ = self._manifest(stage)
        expected = manifest["files"].get(path.name)
        if expected is None:
            raise StageError(
                f"{path} is not part of the completed checkpoint of stage {stage!r}: "
                f"its {MANIFEST_NAME} does not list it"
            )
        if self._digest(path) != expected:
            raise StageError(
                f"{path} changed after stage {stage!r} completed: its SHA-256 differs from the one in "
                f"{MANIFEST_NAME}; run stage {stage!r} again or start a full Run"
            )
        return path

    def _manifest(self, stage: str) -> tuple[dict[str, Any], bytes]:
        try:
            return _load_manifest(self._folder(stage) / MANIFEST_NAME, stage)
        except _Unusable as problem:
            raise StageError(f"stage {stage!r} has no usable checkpoint: {problem}") from None

    def _digest(self, path: Path, *, fresh: bool = False) -> str:
        """SHA-256 of a file, reusing the last value while its size and modification time are unchanged."""
        status = path.stat()
        key = os.fspath(path)
        cached = self._digests.get(key)
        if not fresh and cached is not None and cached[:2] == (status.st_size, status.st_mtime_ns):
            return cached[2]
        digest = sha256_file(path)
        self._digests[key] = (status.st_size, status.st_mtime_ns, digest)
        return digest


# ------------------------------------------------------------------------------------ helpers

class _Unusable(Exception):
    """A manifest is absent, unreadable or invalid; the message names the file and the problem."""


def _load_manifest(path: Path, stage: str) -> tuple[dict[str, Any], bytes]:
    """The parsed manifest and its bytes; _Unusable when the file is absent, unreadable or invalid."""
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        raise _Unusable(f"{path} does not exist") from None
    except OSError as exc:
        raise _Unusable(f"{path} cannot be read ({exc})") from None
    try:
        data = json.loads(raw.decode("utf-8"))
    except ValueError as exc:  # UnicodeDecodeError and JSONDecodeError are ValueErrors
        raise _Unusable(f"{path} is not valid JSON ({exc})") from None
    problem = _manifest_problem(data, stage)
    if problem is not None:
        raise _Unusable(f"{path} {problem}")
    return data, raw


def _manifest_problem(data: Any, stage: str) -> str | None:
    if not isinstance(data, dict) or set(data) != _MANIFEST_KEYS:
        return "does not hold exactly the keys " + ", ".join(sorted(_MANIFEST_KEYS))
    if data["stage"] != stage:
        return f"was written for stage {data['stage']!r}, not {stage!r}"
    version = data["stage_version"]
    if type(version) is not int or version < 0:
        return f"has an invalid stage_version {version!r}"
    if not isinstance(data["completed_utc"], str) or not data["completed_utc"]:
        return "has no completed_utc time"
    if not _is_digest(data["config_sha256"]):
        return "has an invalid config_sha256"
    for key in ("inputs", "files"):
        table = data[key]
        if not isinstance(table, dict) or not all(isinstance(k, str) and _is_digest(v) for k, v in table.items()):
            return f"has an invalid {key!r} table"
    for relative in data["files"]:
        if not _is_relative_file(relative):
            return f"lists {relative!r}, which is not a file path inside the stage folder"
    return None


def _is_relative_file(text: str) -> bool:
    return (
        text != MANIFEST_NAME
        and "\\" not in text
        and ":" not in text
        and all(part not in ("", ".", "..") for part in text.split("/"))
    )


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _check_digest(value: Any, what: str) -> str:
    if not _is_digest(value):
        raise ValueError(f"{what} must be a lowercase hex SHA-256 digest, got {value!r}")
    return value


def _check_stage_name(stage: Any) -> None:
    if not isinstance(stage, str) or _STAGE_NAME.fullmatch(stage) is None:
        raise ValueError(f"stage name {stage!r} must be 1-40 characters from a-z, 0-9 and '_', starting with a letter")


def _check_name(name: Any) -> None:
    if not isinstance(name, str) or _ITEM_NAME.fullmatch(name) is None:
        raise ValueError(
            f"checkpoint name {name!r} must be 1-64 characters from a-z, 0-9, '_' and '-', not starting with '-'"
        )
    if name in _RESERVED_NAMES:
        raise ValueError(f"checkpoint name {name!r} is reserved (the manifest, or a Windows device name)")


def _check_version(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"stage_version must be a non-negative int, got {value!r}")
    return int(value)


def _absolute(value: Any, what: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError(f"{what} must be a path, got {type(value).__name__}")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{what} must be an absolute path, got {str(path)!r}")
    return Path(os.path.normpath(path))


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _json_ready(value: Any, where: str) -> Any:
    """Copy of a JSON value with numpy values converted; refuses what would not load back unchanged."""
    if isinstance(value, (np.ndarray, np.generic)):
        if value.dtype.kind not in _JSON_DTYPE_KINDS:
            raise TypeError(f"{where} is a numpy {value.dtype} value, which a JSON checkpoint cannot hold")
        value = value.tolist() if isinstance(value, np.ndarray) else value.item()
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{where} is {value!r}; a JSON checkpoint holds finite numbers only")
        return float(value)
    if isinstance(value, str):
        return str.__str__(value)  # a plain str, also for str subclasses
    if isinstance(value, (list, tuple)):
        return [_json_ready(item, f"{where}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Mapping):
        ready: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"{where} has the {type(key).__name__} key {key!r}; JSON keys must be str, "
                    "or the value would not load back unchanged"
                )
            ready[str.__str__(key)] = _json_ready(item, f"{where}[{key!r}]")
        return ready
    raise TypeError(
        f"{where} is a {type(value).__name__}, which a JSON checkpoint cannot hold; use None, bool, "
        "int, float, str, and lists, tuples or str-keyed dicts of these"
    )


def _json_bytes(value: Any) -> bytes:
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return (text + "\n").encode("utf-8")


def _replace(source: Path, target: Path) -> None:
    """os.replace, retried while Windows briefly locks the target (a virus scanner, for example)."""
    for attempt in range(1, _REPLACE_ATTEMPTS + 1):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS:
                raise
            time.sleep(_REPLACE_PAUSE * attempt)


def _read(path: Path, load: Callable[[], Any]) -> Any:
    """Run a loader; any failure other than a PipelineError becomes a StageError naming the file."""
    try:
        return load()
    except PipelineError:
        raise
    except Exception as exc:
        raise StageError(f"cannot read the checkpoint file {path}: {type(exc).__name__}: {exc}") from exc


def _is_xgboost_module(module: str) -> bool:
    return module == "xgboost" or module.startswith("xgboost.")


def _model_format(model: Any) -> str:
    """"lightgbm", "xgboost" or "joblib". Only an already imported library can own the model."""
    cls = type(model)
    lightgbm = sys.modules.get("lightgbm")
    if lightgbm is not None and cls is getattr(lightgbm, "Booster", None):
        return "lightgbm"
    xgboost = sys.modules.get("xgboost")
    if xgboost is not None:
        if cls is getattr(xgboost, "Booster", None):
            return "xgboost"
        sklearn_base = getattr(xgboost, "XGBModel", None)
        if sklearn_base is not None and issubclass(cls, sklearn_base) and _is_xgboost_module(cls.__module__):
            return "xgboost"
    return "joblib"


def _model_info(data: Any) -> dict[str, Any]:
    """Checked contents of ``<name>.model.json``."""
    if not isinstance(data, dict):
        raise ValueError("the model info is not a JSON object")
    kind = data.get("format")
    if kind not in _MODEL_FILES:
        raise ValueError(f"the model info names the unknown format {kind!r}")
    keys = {"format", "module", "class"} | ({"best_iteration"} if kind == "lightgbm" else set())
    if set(data) != keys:
        raise ValueError(f"the model info of a {kind} model needs exactly the keys {', '.join(sorted(keys))}")
    if not all(isinstance(data[key], str) and data[key] for key in ("module", "class")):
        raise ValueError("the model info needs a module name and a class name")
    if kind == "lightgbm" and type(data["best_iteration"]) is not int:
        raise ValueError("the model info of a LightGBM Booster needs an int best_iteration")
    return data


def _xgboost_class(xgboost: Any, module: str, qualname: str) -> type:
    """The XGBoost class named in a model info; only xgboost's own Booster and model classes qualify."""
    if not _is_xgboost_module(module):
        raise ValueError(f"the model info names {module}.{qualname}, which is not an XGBoost class")
    found: Any = importlib.import_module(module)
    for part in qualname.split("."):
        found = getattr(found, part)
    if not isinstance(found, type) or not (found is xgboost.Booster or issubclass(found, xgboost.XGBModel)):
        raise ValueError(f"{module}.{qualname} is not an XGBoost Booster or scikit-learn model class")
    return found


def _restore_model(info: dict[str, Any], path: Path) -> Any:
    kind = info["format"]
    if kind == "lightgbm":
        lightgbm = importlib.import_module("lightgbm")
        booster = lightgbm.Booster(model_str=path.read_bytes().decode("utf-8"))
        booster.best_iteration = info["best_iteration"]
        return booster
    if kind == "xgboost":
        xgboost = importlib.import_module("xgboost")
        model = _xgboost_class(xgboost, info["module"], info["class"])()
        model.load_model(bytearray(path.read_bytes()))
        return model
    model = importlib.import_module("joblib").load(path)
    saved = f"{info['module']}.{info['class']}"
    found = f"{type(model).__module__}.{type(model).__qualname__}"
    if found != saved:
        raise ValueError(f"the file holds a {found}, but its model info records a {saved}")
    return model
