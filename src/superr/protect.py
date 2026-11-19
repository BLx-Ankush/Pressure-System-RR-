"""Protected_File hashing at the start and end of every Run (Requirement 1, criteria 7-9).

Protected_Files are the six Source_CSVs plus ``plan.md``, ``pressure1.md``, ``pressure2.md`` and
``pressure3.md``, wherever they appear in the workspace (requirements glossary). The runner hashes
them before the Data_Loader reads any Source_CSV and writes each path and hash to the Run_Log.
In a ``finally`` block it hashes the same files again; a file that changed or no longer exists
fails the Run (exit code 7).

Rules:

* A file is protected when its name is one of ``names``, at any depth below the root. Only the
  file name counts. Names are compared the way the operating system compares them
  (``os.path.normcase``): case-insensitive on Windows, exact elsewhere.
* The walk skips the root's own ``outputs/`` and ``docs/`` folders, the only folders the Pipeline
  writes (Requirement 1, criterion 5), and every ``.venv``, ``.git`` and ``__pycache__`` folder at
  any depth. A folder named ``outputs`` or ``docs`` below the root is walked like any other.
  Symbolic links to folders are not followed.
* Files are hashed in chunks (``hashlib.file_digest``), so ``ball_by_ball.csv`` (about 41 MB) is
  never held in memory whole.
* ``hash_protected`` keys are absolute paths under the resolved root, ordered by their path
  relative to the root, so two calls on the same tree list the files in the same order.
* Nothing in this module writes to disk.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable, Mapping
from pathlib import Path

__all__ = [
    "PROTECTED_NAMES",
    "SKIPPED_DIRS",
    "SOURCE_CSV_NAMES",
    "WRITABLE_DIRS",
    "changed_files",
    "hash_protected",
    "rehash",
    "sha256_file",
]

#: File names of the six Source_CSVs (requirements glossary).
SOURCE_CSV_NAMES: tuple[str, ...] = (
    "ball_by_ball.csv",
    "matches.csv",
    "playing_eleven.csv",
    "competitions.csv",
    "teams.csv",
    "competition_team_player_mapping.csv",
)

#: Protected_File names: the Source_CSVs, the plan and the three pressure notes. The runner
#: passes this set to hash_protected.
PROTECTED_NAMES: frozenset[str] = frozenset(
    SOURCE_CSV_NAMES + ("plan.md", "pressure1.md", "pressure2.md", "pressure3.md")
)

#: Folders directly under the root that the Pipeline writes (Requirement 1, criterion 5).
WRITABLE_DIRS: tuple[str, ...] = ("outputs", "docs")

#: Folders skipped at any depth: none holds a supplied file, and .venv is large.
SKIPPED_DIRS: tuple[str, ...] = (".venv", ".git", "__pycache__")


def sha256_file(path: Path | str) -> str:
    """Hex SHA-256 of a file's bytes, read in chunks. Raises OSError when it cannot be read."""
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _name_keys(names: Iterable[str]) -> frozenset[str]:
    """normcase keys of the protected names; each must be a plain file name."""
    if isinstance(names, (str, bytes)):
        raise TypeError("names must be a collection of file names, not one string")
    keys = set()
    for name in names:
        if not isinstance(name, str) or name in ("", ".", "..") or os.path.basename(name) != name:
            raise ValueError(f"protected name {name!r} is not a plain file name")
        keys.add(os.path.normcase(name))
    return frozenset(keys)


def _raise(error: OSError) -> None:
    """os.walk error handler: a folder that cannot be listed stops the hashing."""
    raise error


def hash_protected(root: Path | str, names: Iterable[str]) -> dict[Path, str]:
    """SHA-256 of every Protected_File under root, keyed by absolute path.

    The rules are in the module docstring. Raises NotADirectoryError when root is not a folder,
    and OSError when a folder cannot be listed or a protected file cannot be read: the Run could
    not then show at its end that every Protected_File is unchanged.
    """
    top = Path(root).resolve()
    if not top.is_dir():
        raise NotADirectoryError(f"workspace root {top} is not a folder")
    wanted = _name_keys(names)
    skip_below = frozenset(os.path.normcase(name) for name in SKIPPED_DIRS)
    skip_at_top = skip_below | frozenset(os.path.normcase(name) for name in WRITABLE_DIRS)
    top_text = os.fspath(top)
    found: list[tuple[str, Path]] = []
    for folder, subfolders, files in os.walk(top_text, onerror=_raise):
        skip = skip_at_top if folder == top_text else skip_below
        subfolders[:] = sorted(name for name in subfolders if os.path.normcase(name) not in skip)
        for name in files:
            if os.path.normcase(name) in wanted:
                path = Path(folder, name)
                found.append((path.relative_to(top).as_posix(), path))
    found.sort(key=lambda item: item[0])
    return {path: sha256_file(path) for _, path in found}


def rehash(before: Mapping[Path, str]) -> dict[Path, str | None]:
    """Hash every path of ``before`` again, in the same order.

    The value is None for a path that no longer exists or can no longer be read as a file.
    """
    after: dict[Path, str | None] = {}
    for path in before:
        try:
            after[path] = sha256_file(path)
        except OSError:
            after[path] = None
    return after


def changed_files(
    before: Mapping[Path, str], after: Mapping[Path, str | None] | None = None
) -> list[Path]:
    """Paths of ``before`` whose file is now modified, missing or unreadable, in ``before`` order.

    ``after`` is the result of ``rehash(before)``, which the runner also writes to the Run_Log;
    when it is omitted the files are hashed again here. A path absent from ``after`` counts as
    changed. Files that appeared after ``before`` was taken are not reported.
    """
    current = rehash(before) if after is None else after
    return [path for path, digest in before.items() if current.get(path) != digest]
