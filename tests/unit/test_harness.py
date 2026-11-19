"""Self-tests for the pytest harness in tests/conftest.py (Requirement 28, criterion 7)."""

from __future__ import annotations

import os
import shutil
import socket
import urllib.request
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st
from hypothesis.configuration import storage_directory

CONFTEST = Path(__file__).resolve().parents[1] / "conftest.py"

# Inner test modules for pytester sessions that run a copy of the harness.
PIPELINE_OUTPUT_TESTS = """
import pytest


@pytest.mark.requires_outputs("audit/audit.md", "audit/pass_rates.csv")
def test_reads_audit(outputs_dir):
    assert (outputs_dir / "audit" / "audit.md").read_text(encoding="utf-8") == "audit"


def test_without_marker():
    pass
"""

MARKER_WITHOUT_PATH_TEST = """
import pytest


@pytest.mark.requires_outputs()
def test_x():
    pass
"""


def test_superr_hypothesis_profile_is_loaded() -> None:
    active = hypothesis_settings.default
    assert hypothesis_settings.get_current_profile_name() == "superr"
    assert active.max_examples == 100
    assert active.derandomize is True
    assert active.database is None
    assert active.deadline is None


def test_given_draws_the_same_examples_on_every_run() -> None:
    drawn: list[int] = []

    @given(st.integers())
    def record(value: int) -> None:
        drawn.append(value)

    record()
    first_run = drawn.copy()
    drawn.clear()
    record()
    assert drawn == first_run
    assert 0 < len(first_run) <= 100


def test_tool_caches_live_outside_the_workspace(workspace_root: Path) -> None:
    hypothesis_home = storage_directory(intent_to_write=False).home_directory
    matplotlib_home = Path(os.environ["MPLCONFIGDIR"])
    assert os.environ["MPLBACKEND"] == "Agg"
    for folder in (hypothesis_home, matplotlib_home):
        assert folder.is_dir()
        assert not folder.resolve().is_relative_to(workspace_root)


def test_path_fixtures(workspace_root: Path, fixtures_dir: Path, outputs_dir: Path) -> None:
    assert (workspace_root / "pyproject.toml").is_file()
    assert (workspace_root / "ball_by_ball.csv").is_file()
    assert fixtures_dir == workspace_root / "tests" / "fixtures"
    assert fixtures_dir.is_dir()
    assert outputs_dir == workspace_root / "outputs"


def test_network_access_fails_loudly() -> None:
    with pytest.raises(RuntimeError, match="network access blocked") as by_name:
        urllib.request.urlopen("http://example.com/", timeout=1)
    with socket.socket() as sock:
        with pytest.raises(RuntimeError, match="network access blocked") as by_address:
            sock.connect(("203.0.113.7", 80))  # TEST-NET-3, a documentation address (RFC 5737)
    for blocked in (by_name, by_address):
        assert blocked.type.__name__ == "NetworkBlockedError"
        assert not issubclass(blocked.type, OSError)  # so `except OSError` cannot swallow it


def test_loopback_connections_still_work() -> None:
    left, right = socket.socketpair()  # on Windows this connects over 127.0.0.1
    with left, right:
        left.sendall(b"ok")
        assert right.recv(2) == b"ok"


def _harness_copy(
    pytester: pytest.Pytester, test_source: str, present: tuple[str, ...] = ()
) -> None:
    """Copy the harness to <tmp>/tests/conftest.py, so that its outputs/ is <tmp>/outputs."""
    tests_dir = pytester.mkdir("tests")
    shutil.copyfile(CONFTEST, tests_dir / "conftest.py")
    (tests_dir / "test_outputs.py").write_text(test_source, encoding="utf-8")
    for relative in present:
        path = pytester.path / "outputs" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("audit", encoding="utf-8")
    pytester.makeini("[pytest]\naddopts = -p no:cacheprovider --strict-markers\n")


@pytest.mark.parametrize(
    ("present", "missing"),
    [
        pytest.param((), ("audit/audit.md", "audit/pass_rates.csv"), id="both-missing"),
        pytest.param(("audit/audit.md",), ("audit/pass_rates.csv",), id="one-missing"),
    ],
)
def test_requires_outputs_skips_naming_the_missing_files(
    pytester: pytest.Pytester, present: tuple[str, ...], missing: tuple[str, ...]
) -> None:
    _harness_copy(pytester, PIPELINE_OUTPUT_TESTS, present)
    run = pytester.inline_run("tests")
    run.assertoutcome(passed=1, skipped=1)
    (skip,) = run.listoutcomes()[1]
    location, _, reason = skip.longrepr
    assert Path(location).name == "test_outputs.py"  # reported at the test, not in conftest.py
    for relative in missing:
        assert f"outputs/{relative}" in reason
    for relative in present:
        assert relative not in reason
    assert "run_pipeline.py" in reason


def test_requires_outputs_runs_the_test_when_every_file_exists(pytester: pytest.Pytester) -> None:
    _harness_copy(pytester, PIPELINE_OUTPUT_TESTS, ("audit/audit.md", "audit/pass_rates.csv"))
    pytester.inline_run("tests").assertoutcome(passed=2)


def test_requires_outputs_without_a_path_stops_the_run(pytester: pytest.Pytester) -> None:
    _harness_copy(pytester, MARKER_WITHOUT_PATH_TEST)
    result = pytester.runpytest("tests")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*test_x: requires_outputs takes one or more paths*"])
