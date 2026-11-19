"""Pytest harness for the Pressure Rating System tests (Requirement 28, criterion 7).

Every test session gets:

* the Hypothesis profile ``superr`` (max_examples=100, derandomize=True, database=None,
  deadline=None), loaded when this file is imported;
* session-scoped fixtures ``workspace_root``, ``fixtures_dir``, ``outputs_dir`` and
  ``settings`` (the Config, loaded through ``superr.settings.load_settings``);
* the marker ``@pytest.mark.requires_outputs("audit/audit.md", ...)`` for pipeline-output
  tests: the paths are relative to ``outputs/``, and unless every one exists the test is
  skipped with the missing paths named in the reason;
* a network guard: DNS lookups and socket connections to anything other than this machine
  raise ``NetworkBlockedError``, so no test can pass with network access and fail without it;
* tool caches in a temporary folder for the session: ``MPLBACKEND=Agg``, ``MPLCONFIGDIR`` and
  Hypothesis' storage directory. Hypothesis caches source constants and Unicode tables there
  even with ``database=None``, by default in ``.hypothesis/`` under the working directory.
  ``pyproject.toml`` already stops ``.pytest_cache``.

The guard and the environment are set in ``pytest_configure`` rather than in a fixture, so
they also cover collection, when test modules are imported (matplotlib reads
``MPLCONFIGDIR`` on import).
"""

from __future__ import annotations

import functools
import ipaddress
import os
import shutil
import socket
import tempfile
from collections.abc import Callable
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any

import pytest
from hypothesis import settings as hypothesis_settings
from hypothesis.configuration import set_hypothesis_home_dir

if TYPE_CHECKING:
    from superr.settings import Settings

# The harness self-test (tests/unit/test_harness.py) runs inner pytest sessions.
pytest_plugins = ["pytester"]

TESTS_DIR = Path(__file__).resolve().parent
ROOT = TESTS_DIR.parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
OUTPUTS_DIR = ROOT / "outputs"
RUN_COMMAND = r".venv\Scripts\python run_pipeline.py"

# ---------------------------------------------------------------------------------------
# Hypothesis profile
# ---------------------------------------------------------------------------------------
# The parent is Hypothesis' built-in "default" profile, not whichever profile is active:
# Hypothesis loads its "ci" profile when a variable such as CI or GITHUB_ACTIONS is set, and
# a profile built on it would suppress a health check on CI machines only.
hypothesis_settings.register_profile(
    "superr",
    parent=hypothesis_settings.get_profile("default"),
    max_examples=100,
    derandomize=True,
    database=None,
    deadline=None,
)
hypothesis_settings.load_profile("superr")


# ---------------------------------------------------------------------------------------
# Path and Config fixtures
# ---------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def workspace_root() -> Path:
    """The workspace root, which holds config.py, the six Source_CSVs and outputs/."""
    return ROOT


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """tests/fixtures/: small hand-made CSVs and their expected values."""
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def outputs_dir() -> Path:
    """outputs/ under the workspace root: the folder that requires_outputs checks."""
    return OUTPUTS_DIR


@pytest.fixture(scope="session")
def settings(workspace_root: Path) -> Settings:
    """The Config, loaded once per session: settings.get("design_choices.D20_score_tolerance").

    A test module that also needs Hypothesis' ``settings`` should import it under another name.
    """
    from superr.settings import load_settings  # imported here so this file loads without it

    return load_settings(workspace_root)


# ---------------------------------------------------------------------------------------
# requires_outputs marker
# ---------------------------------------------------------------------------------------
def _required_outputs(item: pytest.Item) -> list[str]:
    """The paths named by the item's requires_outputs marks, each kept inside outputs/."""
    paths: list[str] = []
    for mark in item.iter_markers(name="requires_outputs"):
        if not mark.args or mark.kwargs:
            raise pytest.UsageError(
                f"{item.nodeid}: requires_outputs takes one or more paths relative to "
                'outputs/, for example @pytest.mark.requires_outputs("audit/audit.md")'
            )
        for arg in mark.args:
            text = os.fspath(arg) if isinstance(arg, os.PathLike) else arg
            relative = PurePath(text) if isinstance(text, str) else None
            if relative is None or not relative.parts or relative.anchor or ".." in relative.parts:
                raise pytest.UsageError(
                    f"{item.nodeid}: requires_outputs path {arg!r} must be relative to "
                    "outputs/ and stay inside it"
                )
            paths.append(relative.as_posix())
    return list(dict.fromkeys(paths))


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    # A skip mark (rather than pytest.skip() in a hook) runs before any fixture, including
    # module- or session-scoped ones that read outputs/, and reports the test's own location.
    for item in items:
        missing = [path for path in _required_outputs(item) if not (OUTPUTS_DIR / path).exists()]
        if missing:
            listed = ", ".join(f"outputs/{path}" for path in missing)
            item.add_marker(
                pytest.mark.skip(
                    reason=(
                        f"needs a completed Pipeline Run: missing {listed} "
                        f"(looked in {OUTPUTS_DIR}); run `{RUN_COMMAND}` first"
                    )
                )
            )


# ---------------------------------------------------------------------------------------
# Session set-up: marker registration, tool caches, network guard
# ---------------------------------------------------------------------------------------
_SESSION = pytest.StashKey[tuple[pytest.MonkeyPatch, Path]]()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_outputs(*paths): skip the test unless every path, relative to outputs/, "
        "exists (pipeline-output tests that read a completed Run)",
    )
    # pytest's own temporary root when one is set (pytester sets it for inner sessions).
    temp_dir = Path(
        tempfile.mkdtemp(prefix="superr-pytest-", dir=os.environ.get("PYTEST_DEBUG_TEMPROOT"))
    )
    patcher = pytest.MonkeyPatch()
    cache_variables = {"matplotlib": "MPLCONFIGDIR", "hypothesis": "HYPOTHESIS_STORAGE_DIRECTORY"}
    for tool, variable in cache_variables.items():
        (temp_dir / tool).mkdir()
        patcher.setenv(variable, str(temp_dir / tool))
    patcher.setenv("MPLBACKEND", "Agg")
    set_hypothesis_home_dir(temp_dir / "hypothesis")
    _block_network(patcher)
    config.stash[_SESSION] = (patcher, temp_dir)


def pytest_unconfigure(config: pytest.Config) -> None:
    session = config.stash.get(_SESSION, None)
    if session is None:
        return
    patcher, temp_dir = session
    patcher.undo()
    set_hypothesis_home_dir(None)  # back to HYPOTHESIS_STORAGE_DIRECTORY, restored by undo()
    shutil.rmtree(temp_dir, ignore_errors=True)


class NetworkBlockedError(RuntimeError):
    """Raised when a test tries to reach the network (Requirement 28, criterion 7).

    Deliberately not an OSError, so an ``except OSError`` fallback cannot hide the attempt.
    """


_LOCAL_NAMES = frozenset({"", "localhost", socket.gethostname().lower()})


def _is_local_host(host: object) -> bool:
    """True for hosts that never leave this machine: None, localhost, this machine's name,
    and loopback or unspecified IP addresses."""
    if host is None:
        return True
    if isinstance(host, (bytes, bytearray)):
        host = bytes(host).decode("ascii", "replace")
    if not isinstance(host, str):
        return False
    name = host.strip().rstrip(".").lower()
    if name in _LOCAL_NAMES or name.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(name.partition("%")[0])
    except ValueError:
        return False  # any other host name needs a DNS lookup
    return address.is_loopback or address.is_unspecified


def _is_local_address(address: object) -> bool:
    if isinstance(address, (str, bytes, bytearray)):  # an AF_UNIX socket path
        return True
    return isinstance(address, tuple) and len(address) > 0 and _is_local_host(address[0])


def _blocked(call: str, target: object) -> NetworkBlockedError:
    return NetworkBlockedError(
        f"network access blocked in tests: {call}({target!r}). Tests must give the same "
        "outcome with and without network access (Requirement 28, criterion 7); read local "
        "files under tests/fixtures/ or tmp_path instead"
    )


def _guard_lookup(name: str, real: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(real)
    def lookup(host: object, *args: Any, **kwargs: Any) -> Any:
        if not _is_local_host(host):
            raise _blocked(f"socket.{name}", host)
        return real(host, *args, **kwargs)

    return lookup


def _guard_socket_method(name: str, position: int, real: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(real)
    def method(self: socket.socket, *args: Any) -> Any:
        if args and not _is_local_address(args[position]):
            raise _blocked(f"socket.{name}", args[position])
        return real(self, *args)

    return method


def _block_network(patcher: pytest.MonkeyPatch) -> None:
    """Let DNS lookups and socket connections through only for hosts on this machine.

    This covers the Python socket API, which urllib, http.client, requests, pandas URL reads
    and the scikit-learn dataset fetchers use. C extensions that open their own sockets are
    not covered.
    """
    for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr"):
        patcher.setattr(socket, name, _guard_lookup(name, getattr(socket, name)))
    # The address is the first argument of connect and connect_ex, and the last of sendto.
    for name, position in (("connect", 0), ("connect_ex", 0), ("sendto", -1)):
        real = getattr(socket.socket, name)
        patcher.setattr(socket.socket, name, _guard_socket_method(name, position, real))
    # A proxy on a loopback port (such as a local proxy agent) would tunnel requests past the
    # guard; with NO_PROXY=* HTTP clients connect to the real host, which the guard then sees.
    patcher.setenv("NO_PROXY", "*")
    patcher.setenv("no_proxy", "*")
