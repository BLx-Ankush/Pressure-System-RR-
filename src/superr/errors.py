"""Errors that stop a Run, each carrying the exit code the runner returns for it.

Exit codes (design section "Run lifecycle"):

* 0: success.
* 2: Config or argument error. ``superr.settings.ConfigError`` and the runner's own checks
  of Source_CSV paths and ``--from-stage``; these happen before any stage runs.
* 3: data error (Requirements 3-5): ``DataError``.
* 4: leakage guard (Requirement 7, criterion 12; Requirement 14, criterion 11): ``LeakageError``.
* 5: any other stage error: ``StageError``. An exception that is not a ``PipelineError`` also
  counts as an other stage error.
* 6: Report_Check failure (Requirement 27, criterion 17): ``ReportCheckError``.
* 7: a Protected_File changed or was deleted (Requirement 1, criterion 9); set by the runner.

Stages raise a subclass; the runner reads ``exit_code`` from whatever ``PipelineError`` it catches.
"""

from __future__ import annotations

from typing import ClassVar

__all__ = ["DataError", "LeakageError", "PipelineError", "ReportCheckError", "StageError"]


class PipelineError(Exception):
    """Base class of the errors that stop a Run. Raise a subclass; the base uses exit code 5."""

    exit_code: ClassVar[int] = 5


class DataError(PipelineError):
    """The Source_CSV data breaks a rule of Requirements 3-5 (exit code 3).

    For example: a missing registry column, a blank Key_ID, a failed type conversion, a failed
    Fact_Check, or an ``extras_type`` that the Config class mapping cannot classify.
    """

    exit_code: ClassVar[int] = 3


class LeakageError(PipelineError):
    """A fit, a sandbox view or a validation input would use data from an excluded fold (exit code 4)."""

    exit_code: ClassVar[int] = 4


class StageError(PipelineError):
    """Any other condition that stops a stage (exit code 5).

    For example: a missing or non-finite WP, LI or Stress, an empty pooled fallback set, or fewer
    than 10 Eligible_Players in a Role.
    """

    exit_code: ClassVar[int] = 5


class ReportCheckError(PipelineError):
    """The Report_Check failed (exit code 6). Every check still runs, and the files are kept."""

    exit_code: ClassVar[int] = 6
