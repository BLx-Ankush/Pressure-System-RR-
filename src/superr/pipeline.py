"""Runner: the Pipeline_Stage registry and the Run lifecycle (Requirement 1, criteria 5-11;
Requirement 28, criteria 1, 4-6 and 8-10).

Design sections "Pipeline_Stage registry", "Run lifecycle", "Checkpoints and ``--from-stage``",
"Runner, store, Run_Log and protection" and "Error Handling". ``run_pipeline.py`` in the
workspace root calls ``main``.

Stage registry
--------------
``DESIGN_STAGES`` lists the 20 Pipeline_Stages of the design in run order, with the checkpoint
root of each. It fixes every checkpoint folder as ``<root>/<NN>_<stage>/``, where NN is the
stage's position in ``DESIGN_STAGES`` (``01_load`` ... ``20_report``), so no folder number
shifts while stages are added one by one. ``STAGES`` holds the stages built so far, in design
order, and a Run runs exactly these. A task that builds a stage registers it with one entry in
``STAGES``::

    Stage("audit", "superr.audit:run_stage", reads=("load",)),
    Stage("activity_comparison", "superr.activity:run_stage", reads=("rating", "bootstrap"),
          checkpoint_root=COMPARISON_ROOT),

* ``run``: the stage function, ``run(ctx: StageContext) -> None``, given as a
  ``"package.module:function"`` string (preferred) or as the function itself. A string is
  imported when a Run starts, so this module imports no stage module, and a stage module may
  import ``StageContext`` from here at run time without an import cycle.
* ``reads``: the earlier stages whose checkpoints the stage loads. Each must be registered
  before the stage; the manifest records the hash of each one's manifest.
* ``checkpoint_root``: ``STAGES_ROOT`` (the default) for every stage except
  ``activity_comparison``, which uses ``COMPARISON_ROOT`` (Requirement 25, criterion 15).

``validate_stages`` checks a registry before a Run uses it: every entry is a ``Stage`` named in
``DESIGN_STAGES``, registered once and in design order, with the design's checkpoint root, a
well-formed ``run``, and ``reads`` that name only stages registered before it.

Stage contract
--------------
A stage function reads earlier checkpoints with ``ctx.store.load_*(<earlier stage>, name)``
(only stages it lists in ``reads``), saves its own with ``ctx.store.save_*(<its name>, name,
value)``, writes its deliverables under ``ctx.settings.path("outputs")`` (or ``"docs"``),
records data-derived Method_Constants with ``ctx.constants.record``, and stops the Run by
raising a ``superr.errors.PipelineError`` subclass. A model stage registers every fit with the
Run's FitRegistry, ``ctx.registry`` (``superr.leakage``), before the fit runs, and saves the fits
it registered with ``ctx.registry.save(ctx.store, <its name>)``. Before the function runs, the
runner deletes the stage's checkpoint folder and tags new fits with the stage's name; after it
returns, the runner saves the stage's constants as the JSON value ``_constants``
(``CONSTANTS_NAME``, a name reserved for the runner) and writes the manifest with
``store.complete(name, inputs=stage.reads)``. A stage never calls ``complete``.

Run lifecycle (``main``)
------------------------
1. Parse the command line (``--from-stage STAGE``). ``--help`` prints the usage and the stage
   names and returns 0 without starting a Run.
2. Create the Run_Log under ``outputs/logs/`` (every line is echoed to standard error), record
   the Python and pinned package versions (28.4), and check the stage registry.
3. Load and validate ``config.py``. Any problem stops the Run with exit 2 and names every
   missing setting (1.11). ``paths.outputs`` and ``paths.docs`` must be the root's ``outputs``
   and ``docs`` folders, the only folders the Pipeline writes (1.5).
4. Check that each Source_CSV path in the Config points to an existing file; otherwise exit 2,
   naming each Source_CSV and its configured path, before any Source_CSV is loaded (1.10).
5. Hash every Protected_File and write each path and hash to the Run_Log (1.7), then record
   each Source_CSV's hash (28.4).
6. Copy ``plan.md`` to ``docs/plan.md`` and check that the two SHA-256 hashes match (1.6).
7. Check the arguments, each failure with exit 2: a command-line syntax error; a
   ``--from-stage`` value that is not a registered stage name, listing the valid names (28.9);
   an earlier stage whose checkpoint is missing or incomplete, naming the earliest such stage
   (28.10). Every registered stage before the named one must be complete, which also covers
   every stage that the named stage or a later stage reads. A Config hash that differs from
   the one an earlier stage's manifest records is only a warning (28.5). The data-derived
   constants saved by the earlier stages are loaded into ``ctx.constants``, and the fits saved
   by the earlier model stages into ``ctx.registry``, each checked again by the leakage guard
   (exit 4 when one fails; exit 2 when a saved fit registry cannot be read).
8. Import the run function of every stage to run. A full Run (no ``--from-stage``) then deletes
   ``outputs/_stages/`` and ``outputs/final/comparison/_stage/``, so every stage is rebuilt
   (28.1).
9. Run the stages in order and record each one's name, status and wall-clock seconds (28.4). A
   stage error stops the Run: no later stage starts, the checkpoints of the completed stages
   stay, and the stage name and error message go to the Run_Log and standard error (28.6).
10. In a ``finally`` block, hash the Protected_Files hashed in step 5 again and record every
    file that changed or no longer exists (1.8, 1.9); then finish the Run_Log with the exit code.

Exit codes
----------
* 0: the last registered stage completed in this Run and no Protected_File changed (28.8). A
  Run with no registered stage also exits 0, with a warning.
* 2: a Config, input-file or argument error found before any stage runs (steps 3, 4, 6 and 7).
* 3-6: a stage raised a ``PipelineError`` subclass, and the code is its ``exit_code``
  (``DataError`` 3, ``LeakageError`` 4, ``StageError`` 5, ``ReportCheckError`` 6).
* 5: any other exception, in a stage or in the runner, including an invalid stage registry or a
  run function that cannot be imported.
* 7: a Protected_File changed or was deleted during the Run (1.9). It replaces any other code.
* 130: the Run was interrupted (Ctrl+C).
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import shutil
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn

from superr import protect
from superr.errors import DataError, PipelineError, StageError
from superr.leakage import FitRegistry
from superr.report.constants import ConstantError, ConstantRegistry
from superr.runlog import COMPLETED, FAILED, LOG_FOLDER, RunLog, pinned_packages
from superr.settings import CONFIG_FILENAME, ConfigError, Settings, load_settings
from superr.store import CheckpointStore

__all__ = [
    "COMPARISON_ROOT",
    "CONSTANTS_NAME",
    "DEFAULT_ROOT",
    "DESIGN_STAGES",
    "DESIGN_STAGE_NAMES",
    "DOCS_FOLDER",
    "EXIT_CONFIG",
    "EXIT_INTERRUPTED",
    "EXIT_OK",
    "EXIT_OTHER",
    "EXIT_PROTECTED",
    "OUTPUTS_FOLDER",
    "SOURCE_CSVS",
    "STAGES",
    "STAGES_ROOT",
    "Stage",
    "StageContext",
    "checkpoint_layout",
    "main",
    "resolve_run",
    "validate_stages",
]

# ------------------------------------------------------------------------------- constants

#: The two folders under the workspace root that the Pipeline writes (Requirement 1, criterion 5).
OUTPUTS_FOLDER = "outputs"
DOCS_FOLDER = "docs"

#: Checkpoint roots, relative to the workspace root (design section "Checkpoints and --from-stage").
STAGES_ROOT = "outputs/_stages"
COMPARISON_ROOT = "outputs/final/comparison/_stage"

#: JSON value in every stage checkpoint that holds the stage's data-derived Method_Constants.
CONSTANTS_NAME = "_constants"

#: Config key under ``paths`` -> file name, for each of the six Source_CSVs.
SOURCE_CSVS: dict[str, str] = {
    "ball_by_ball": "ball_by_ball.csv",
    "matches": "matches.csv",
    "playing_eleven": "playing_eleven.csv",
    "competitions": "competitions.csv",
    "teams": "teams.csv",
    "mapping": "competition_team_player_mapping.csv",
}

#: The 20 Pipeline_Stages of the design in run order, each with its checkpoint root. The
#: position fixes the checkpoint folder number NN, whichever stages are registered in STAGES.
DESIGN_STAGES: tuple[tuple[str, str], ...] = (
    ("load", STAGES_ROOT),
    ("audit", STAGES_ROOT),
    ("reconstruct", STAGES_ROOT),
    ("state", STAGES_ROOT),
    ("folds", STAGES_ROOT),
    ("skill", STAGES_ROOT),
    ("remaining_runs", STAGES_ROOT),
    ("win_prob", STAGES_ROOT),
    ("outcome", STAGES_ROOT),
    ("model_choice", STAGES_ROOT),
    ("pressure", STAGES_ROOT),
    ("validation", STAGES_ROOT),
    ("value", STAGES_ROOT),
    ("rating", STAGES_ROOT),
    ("bootstrap", STAGES_ROOT),
    ("robustness", STAGES_ROOT),
    ("model_choice_rerun", STAGES_ROOT),
    ("activity_comparison", COMPARISON_ROOT),
    ("final_lists", STAGES_ROOT),
    ("report", STAGES_ROOT),
)
DESIGN_STAGE_NAMES: tuple[str, ...] = tuple(name for name, _ in DESIGN_STAGES)
_DESIGN_POSITION: dict[str, int] = {name: index for index, name in enumerate(DESIGN_STAGE_NAMES)}
_DESIGN_ROOT: dict[str, str] = dict(DESIGN_STAGES)

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_OTHER = StageError.exit_code  # 5
EXIT_PROTECTED = 7
EXIT_INTERRUPTED = 130

#: Workspace root used when main() is not given one: the folder that holds src/superr/.
DEFAULT_ROOT = Path(__file__).resolve().parents[2]

_PROG = "run_pipeline.py"
_PLAN_FILE = "plan.md"
_REQUIREMENTS_FILE = "requirements.txt"
_RUN_TARGET = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")


# ---------------------------------------------------------------------------- stage registry

@dataclass(frozen=True)
class Stage:
    """One Pipeline_Stage: its design name, run function, the stages it reads and its checkpoint root."""

    name: str
    run: Callable[[StageContext], None] | str  # the function, or "package.module:function"
    reads: tuple[str, ...] = ()  # earlier stages whose checkpoints it reads
    checkpoint_root: str = STAGES_ROOT  # COMPARISON_ROOT for activity_comparison only

    def __post_init__(self) -> None:
        if isinstance(self.reads, (str, bytes)):
            raise TypeError(f"Stage {self.name!r}: reads must be a tuple of stage names, not one string")
        object.__setattr__(self, "reads", tuple(self.reads))


@dataclass
class StageContext:
    """What every stage function receives: the Run's Config, checkpoints, Run_Log, root, constants and fits."""

    settings: Settings
    store: CheckpointStore
    log: RunLog
    root: Path  # the resolved workspace root
    constants: ConstantRegistry
    registry: FitRegistry = field(default_factory=FitRegistry)  # every model fit of the Run (superr.leakage)


#: The Pipeline_Stages built so far, in design run order. Each stage task adds its entry here.
STAGES: tuple[Stage, ...] = (
    Stage("load", "superr.io:run_stage"),
    Stage("audit", "superr.audit:run_stage", reads=("load",)),
    Stage("reconstruct", "superr.reconstruct:run_stage", reads=("load", "audit")),
    Stage("state", "superr.state:run_stage", reads=("load", "audit", "reconstruct")),
    Stage("folds", "superr.folds:run_stage", reads=("load", "reconstruct")),
    Stage("skill", "superr.skill:run_stage", reads=("load", "reconstruct", "state", "folds")),
    Stage(
        "remaining_runs",
        "superr.models.remaining_runs:run_stage",
        reads=("reconstruct", "state", "folds", "skill"),
    ),
    Stage(
        "win_prob",
        "superr.models.win_prob:run_stage",
        reads=("load", "reconstruct", "state", "folds", "skill", "remaining_runs"),
    ),
    Stage(
        "outcome",
        "superr.models.outcome:run_stage",
        reads=("reconstruct", "state", "folds", "skill", "remaining_runs"),
    ),
    Stage(
        "model_choice",
        "superr.models.model_choice:run_stage",
        reads=("load", "reconstruct", "state", "folds", "remaining_runs", "win_prob", "outcome"),
    ),
    Stage(
        "pressure",
        "superr.pressure:run_stage",
        reads=("reconstruct", "state", "folds", "remaining_runs", "outcome", "model_choice"),
    ),
    Stage(
        "validation",
        "superr.validation:run_stage",
        reads=("reconstruct", "state", "folds", "remaining_runs", "win_prob", "outcome", "model_choice", "pressure"),
    ),
    Stage(
        "value",
        "superr.value:run_stage",
        reads=(
            "load", "reconstruct", "state", "folds", "remaining_runs", "win_prob", "outcome", "model_choice", "pressure",
        ),
    ),
    Stage("rating", "superr.rating:run_stage", reads=("load", "reconstruct", "pressure", "value")),
    Stage("bootstrap", "superr.robustness:run_bootstrap_stage", reads=("load", "rating")),
    Stage(
        "robustness",
        "superr.robustness:run_stage",
        reads=(
            "load", "reconstruct", "state", "folds", "remaining_runs", "win_prob", "outcome", "model_choice",
            "pressure", "validation", "value", "rating",
        ),
    ),
    Stage(
        "model_choice_rerun",
        "superr.models.model_choice:run_rerun_stage",
        reads=(
            "load", "reconstruct", "state", "folds", "remaining_runs", "win_prob", "outcome", "model_choice",
            "pressure", "validation", "value", "rating",
        ),
    ),
    Stage(
        "activity_comparison",
        "superr.activity:run_stage",
        reads=("load", "rating", "bootstrap"),
        checkpoint_root=COMPARISON_ROOT,
    ),
    Stage(
        "final_lists",
        "superr.lists:run_stage",
        reads=("load", "reconstruct", "pressure", "value", "rating", "bootstrap", "activity_comparison"),
    ),
    Stage(
        "report",
        "superr.report.export:run_stage",
        reads=(
            "audit", "reconstruct", "folds", "skill", "remaining_runs", "win_prob", "outcome", "model_choice",
            "pressure", "validation", "value", "rating", "bootstrap", "activity_comparison", "final_lists",
        ),
    ),
)


def checkpoint_layout(root: Path | str) -> list[tuple[str, Path]]:
    """(stage name, absolute checkpoint root) for every design stage, in run order, for CheckpointStore."""
    base = Path(root)
    return [(name, base / checkpoint_root) for name, checkpoint_root in DESIGN_STAGES]


def _run_problem(run: Any) -> str | None:
    if callable(run) or (isinstance(run, str) and _RUN_TARGET.fullmatch(run)):
        return None
    return f"run must be the stage function or a 'package.module:function' string, got {run!r}"


def validate_stages(stages: Sequence[Stage]) -> None:
    """Raise ValueError, listing every problem, unless ``stages`` is a valid stage registry.

    The rules are in the module docstring ("Stage registry").
    """
    problems: list[str] = []
    registered: set[str] = set()
    latest: str | None = None  # the registered stage that runs last in design order so far
    for number, stage in enumerate(stages, start=1):
        if not isinstance(stage, Stage):
            problems.append(f"entry {number} is a {type(stage).__name__}, not a Stage")
            continue
        name = stage.name
        if not isinstance(name, str) or name not in _DESIGN_POSITION:
            problems.append(f"{name!r} is not a Pipeline_Stage of the design ({', '.join(DESIGN_STAGE_NAMES)})")
            continue
        if name in registered:
            problems.append(f"stage {name!r} is registered more than once")
            continue
        if latest is not None and _DESIGN_POSITION[name] < _DESIGN_POSITION[latest]:
            problems.append(f"stage {name!r} is registered after {latest!r} but runs before it in the design order")
        else:
            latest = name
        if stage.checkpoint_root != _DESIGN_ROOT[name]:
            problems.append(
                f"stage {name!r} has checkpoint_root {stage.checkpoint_root!r}; the design puts its "
                f"checkpoint under {_DESIGN_ROOT[name]!r}"
            )
        listed: list[Any] = []
        for read in stage.reads:
            if not isinstance(read, str):
                problems.append(f"stage {name!r}: reads holds {read!r}, which is not a stage name")
            elif read in listed:
                problems.append(f"stage {name!r} lists {read!r} in reads more than once")
            elif read not in registered:
                problems.append(f"stage {name!r} reads {read!r}, which is not a stage registered before it")
            listed.append(read)
        problem = _run_problem(stage.run)
        if problem is not None:
            problems.append(f"stage {name!r}: {problem}")
        registered.add(name)
    if problems:
        raise ValueError("the stage registry is not valid:\n" + "\n".join(f"  - {line}" for line in problems))


def resolve_run(stage: Stage) -> Callable[[StageContext], None]:
    """The stage's run function, importing its module when ``run`` is a ``"package.module:function"`` string."""
    if callable(stage.run):
        return stage.run
    problem = _run_problem(stage.run)
    if problem is not None:
        raise ValueError(f"stage {stage.name!r}: {problem}")
    module_name, _, attribute = stage.run.partition(":")
    target: Any = importlib.import_module(module_name)
    for part in attribute.split("."):
        target = getattr(target, part)
    if not callable(target):
        raise TypeError(f"stage {stage.name!r}: {stage.run!r} is a {type(target).__name__}, not a function")
    return target


# ------------------------------------------------------------------------------ command line

class _UsageError(Exception):
    """A command-line syntax error; main reports it once the Run_Log exists (exit 2)."""


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:  # argparse would print and exit at once
        raise _UsageError(message)


def _parser(stages: Sequence[Stage]) -> _ArgumentParser:
    names = ", ".join(stage.name for stage in stages if isinstance(stage, Stage)) or "(none registered yet)"
    parser = _ArgumentParser(
        prog=_PROG,
        description=(
            "Run the Pressure Rating System: every Pipeline_Stage in run order, from the six "
            "Source_CSVs to the final lists and the report."
        ),
        epilog=f"Pipeline_Stages in run order: {names}",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--from-stage",
        dest="from_stage",
        metavar="STAGE",
        help=(
            "run STAGE and every later stage, reading the saved outputs of the earlier stages "
            "from outputs/ instead of running them again"
        ),
    )
    return parser


# ----------------------------------------------------------------------------------- the Run

class _Stop(Exception):
    """Stops a Run before or between stages with an exit code and a message for the Run_Log."""

    def __init__(self, code: int, message: str, stage: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.stage = stage


def _describe(exc: BaseException, *, with_traceback: bool) -> str:
    if isinstance(exc, KeyboardInterrupt):
        return "the Run was interrupted (KeyboardInterrupt)"
    text = str(exc)
    message = f"{type(exc).__name__}: {text}" if text else type(exc).__name__
    if with_traceback:
        message += "\n" + "".join(traceback.format_exception(exc)).rstrip()
    return message


def _exit_code(exc: BaseException) -> int:
    if isinstance(exc, PipelineError):
        return exc.exit_code
    if isinstance(exc, KeyboardInterrupt):
        return EXIT_INTERRUPTED
    return EXIT_OTHER


class _Run:
    """State of one Run, from the Run_Log's creation to the end-of-Run check."""

    def __init__(self, root: Path, stages: tuple[Stage, ...], log: RunLog) -> None:
        self.root = root
        self.stages = stages
        self.log = log
        self.start_hashes: dict[Path, str] | None = None

    # -- lifecycle -------------------------------------------------------------------------

    def execute(self, from_stage: str | None, usage_error: str | None) -> int:
        """Steps 2-9 of the lifecycle; returns the exit code or raises (main logs the error)."""
        self._record_environment()
        try:
            validate_stages(self.stages)
        except (TypeError, ValueError) as exc:
            raise _Stop(EXIT_OTHER, str(exc)) from None
        settings = self._load_config()
        self._check_write_folders(settings)
        csv_paths = self._check_source_csvs(settings)
        self.start_hashes = protect.hash_protected(self.root, protect.PROTECTED_NAMES)
        self.log.protected_start(self.start_hashes)
        for key, path in csv_paths.items():
            self._record_source_csv(key, path)
        self._copy_plan()
        if usage_error is not None:
            raise _Stop(EXIT_CONFIG, usage_error)
        store = CheckpointStore(
            self.root / OUTPUTS_FOLDER,
            checkpoint_layout(self.root),
            config_sha256=settings.effective_sha256(),
        )
        ctx = StageContext(
            settings=settings,
            store=store,
            log=self.log,
            root=self.root,
            constants=ConstantRegistry.from_settings(settings),
            registry=FitRegistry.for_store(store),  # reads the Fold_Table from the folds checkpoint when first needed
        )
        start = self._start_index(from_stage, ctx)
        to_run = self.stages[start:]
        functions = self._resolve(to_run)
        if from_stage is None:
            self._clear_checkpoints(store)
        return self._run_stages(ctx, to_run, functions)

    def stopped(self, exc: BaseException) -> int:
        """Log an error that stopped the Run outside the stage loop; return its exit code."""
        if isinstance(exc, _Stop):
            self.log.error(exc.message, stage=exc.stage)
            return exc.code
        self.log.error(_describe(exc, with_traceback=not isinstance(exc, (PipelineError, KeyboardInterrupt))))
        return _exit_code(exc)

    def end(self, code: int) -> int:
        """Step 10: the end-of-Run Protected_File check and the Run_Log's last record."""
        if self.start_hashes is not None:
            after = protect.rehash(self.start_hashes)
            changed = protect.changed_files(self.start_hashes, after)
            if changed:
                code = EXIT_PROTECTED
            self.log.protected_end(after, changed)
        self.log.finish(code)
        return code

    # -- steps -----------------------------------------------------------------------------

    def _record_environment(self) -> None:
        requirements = self.root / _REQUIREMENTS_FILE
        try:
            pins = pinned_packages(requirements)
        except (OSError, ValueError) as exc:
            self.log.warning(f"cannot read the pinned packages from {requirements}: {exc}")
            pins = {}
        self.log.environment(pins)

    def _load_config(self) -> Settings:
        try:
            settings = load_settings(self.root)
        except ConfigError as exc:
            config_file = self.root / CONFIG_FILENAME
            try:
                self.log.config_hash(protect.sha256_file(config_file), config_file)
            except OSError:
                pass  # no readable config.py: the error message says so
            raise _Stop(
                EXIT_CONFIG, f"the Config cannot be used, so the Run stops before loading any Source_CSV: {exc}"
            ) from None
        self.log.config_hash(settings.sha256(), settings.source_path)
        return settings

    def _check_write_folders(self, settings: Settings) -> None:
        problems = []
        for key, folder in (("outputs", OUTPUTS_FOLDER), ("docs", DOCS_FOLDER)):
            configured = settings.path(key)
            if configured.resolve() != (self.root / folder).resolve():
                problems.append(
                    f"paths.{key} is {settings.get(f'paths.{key}')!r} ({configured}); the Pipeline writes "
                    f"only under the workspace root's {folder}/ folder (Requirement 1, criterion 5), so it "
                    f"must be {folder!r}"
                )
        if problems:
            raise _Stop(EXIT_CONFIG, "the Config cannot be used:\n" + "\n".join(f"  - {line}" for line in problems))

    def _check_source_csvs(self, settings: Settings) -> dict[str, Path]:
        found: dict[str, Path] = {}
        missing: list[str] = []
        for key, file_name in SOURCE_CSVS.items():
            path = settings.path(key)
            if path.is_file():
                found[key] = path
            else:
                missing.append(
                    f"Source_CSV {file_name} (Config setting paths.{key}): the configured path "
                    f"{settings.get(f'paths.{key}')!r} does not point to an existing file ({path})"
                )
        if missing:
            for key, path in found.items():
                try:
                    self._record_source_csv(key, path)
                except OSError:
                    pass  # an unreadable file is reported by the Data_Loader in a later Run
            raise _Stop(
                EXIT_CONFIG,
                "the Run stops before loading any Source_CSV (Requirement 1, criterion 10):\n"
                + "\n".join(f"  - {line}" for line in missing),
            )
        return found

    def _record_source_csv(self, key: str, path: Path) -> None:
        digest = None if self.start_hashes is None else self.start_hashes.get(path.resolve())
        self.log.source_csv(key, path, digest or protect.sha256_file(path))

    def _copy_plan(self) -> None:
        source = self.root / _PLAN_FILE
        target = self.root / DOCS_FOLDER / _PLAN_FILE
        if not source.is_file():
            raise _Stop(
                EXIT_CONFIG,
                f"{_PLAN_FILE} is missing from the workspace root ({source}); every Run copies it to "
                f"{DOCS_FOLDER}/{_PLAN_FILE} (Requirement 1, criterion 6)",
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        try:
            shutil.copyfile(source, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        expected, copied = protect.sha256_file(source), protect.sha256_file(target)
        if copied != expected:
            raise _Stop(EXIT_OTHER, f"{target} has SHA-256 {copied} after the copy, but {source} has {expected}")
        self.log.info(f"copied {_PLAN_FILE} to {DOCS_FOLDER}/{_PLAN_FILE}; both have sha256 {copied}")

    def _start_index(self, from_stage: str | None, ctx: StageContext) -> int:
        """Index in self.stages of the first stage to run; checks --from-stage (28.5, 28.9, 28.10)."""
        if from_stage is None:
            return 0
        names = [stage.name for stage in self.stages]
        if from_stage not in names:
            valid = ", ".join(names) or "none, because no Pipeline_Stage is registered yet"
            hint = ""
            if from_stage in _DESIGN_POSITION:
                hint = f" {from_stage!r} is a stage of the design that is not registered in STAGES yet."
            raise _Stop(
                EXIT_CONFIG,
                f"--from-stage {from_stage!r} is not a Pipeline_Stage name.{hint} "
                f"Valid names, in run order: {valid}",
            )
        start = names.index(from_stage)
        earlier = self.stages[:start]
        store = ctx.store
        for stage in earlier:
            reason = store.incomplete_reason(stage.name)
            if reason is not None:
                raise _Stop(
                    EXIT_CONFIG,
                    f"--from-stage {from_stage} needs the saved outputs of every earlier Pipeline_Stage, "
                    f"and the earliest stage whose saved outputs are missing is {stage.name!r}: {reason}. "
                    f"Run `python {_PROG} --from-stage {stage.name}` or a full Run.",
                )
        differing = []
        for stage in earlier:
            recorded = store.manifest(stage.name)["config_sha256"]
            if recorded != store.config_sha256:
                differing.append(f"{stage.name} ({recorded[:12]})")
        if differing:
            self.log.warning(
                f"the Config hash {store.config_sha256[:12]} differs from the hash recorded with the saved "
                f"outputs of {', '.join(differing)}; those outputs are reused as they were saved"
            )
        for stage in earlier:
            try:
                ctx.constants.load_dict(store.load_json(stage.name, CONSTANTS_NAME))
            except (StageError, ConstantError) as exc:
                raise _Stop(
                    EXIT_CONFIG,
                    f"--from-stage {from_stage}: the saved outputs of the earlier Pipeline_Stage "
                    f"{stage.name!r} cannot be used: {exc}",
                ) from None
        reloaded = []
        for stage in earlier:  # a saved fit that fails the leakage guard raises LeakageError (exit 4)
            try:
                count = ctx.registry.load_saved(store, stage.name)
            except (StageError, DataError, ValueError, TypeError) as exc:
                raise _Stop(
                    EXIT_CONFIG,
                    f"--from-stage {from_stage}: the fit registry saved by the earlier Pipeline_Stage "
                    f"{stage.name!r} cannot be used: {exc}",
                ) from None
            if count:
                reloaded.append(f"{stage.name} ({count:,})")
        if reloaded:
            self.log.info(f"--from-stage {from_stage}: reloaded the fits saved by {', '.join(reloaded)} into the FitRegistry")
        if earlier:
            self.log.info(f"--from-stage {from_stage}: reusing the saved outputs of {', '.join(s.name for s in earlier)}")
        return start

    def _resolve(self, stages: Sequence[Stage]) -> list[Callable[[StageContext], None]]:
        functions = []
        for stage in stages:
            try:
                functions.append(resolve_run(stage))
            except Exception as exc:
                raise _Stop(
                    EXIT_OTHER,
                    f"cannot load the run function {stage.run!r} of stage {stage.name!r}, so no stage "
                    f"runs: {_describe(exc, with_traceback=True)}",
                    stage=stage.name,
                ) from None
        return functions

    def _clear_checkpoints(self, store: CheckpointStore) -> None:
        deleted = store.clear_all()
        if deleted:
            shown = ", ".join(path.relative_to(self.root).as_posix() for path in deleted)
            self.log.info(f"full Run: deleted the saved stage outputs of earlier Runs in {shown}")
        else:
            self.log.info("full Run: no saved stage outputs to delete")

    def _run_stages(
        self, ctx: StageContext, stages: Sequence[Stage], functions: Sequence[Callable[[StageContext], None]]
    ) -> int:
        log = self.log
        if not stages:
            log.warning("no Pipeline_Stage is registered in the stage registry, so the Run ran no stage")
            return EXIT_OK
        log.info(f"running {len(stages)} Pipeline_Stage(s) in run order: {', '.join(s.name for s in stages)}")
        for index, (stage, run) in enumerate(zip(stages, functions)):
            ctx.constants.begin_stage(stage.name)
            ctx.registry.begin_stage(stage.name)
            log.start_stage(stage.name)
            started = time.perf_counter()
            try:
                ctx.store.clear_stage(stage.name)
                run(ctx)
                ctx.store.save_json(stage.name, CONSTANTS_NAME, ctx.constants.to_dict(stage=stage.name))
                ctx.store.complete(stage.name, inputs=stage.reads)
            except BaseException as exc:  # every failure stops the Run; the caller still runs step 10
                log.stage(stage.name, time.perf_counter() - started, FAILED)
                unexpected = not isinstance(exc, (PipelineError, KeyboardInterrupt))
                log.error(_describe(exc, with_traceback=unexpected), stage=stage.name)
                later = [s.name for s in stages[index + 1:]]
                if later:
                    log.info(f"later Pipeline_Stages not started: {', '.join(later)}")
                return _exit_code(exc)
            finally:
                ctx.constants.begin_stage(None)
                ctx.registry.begin_stage(None)
            log.stage(stage.name, time.perf_counter() - started, COMPLETED)
        return EXIT_OK


def main(
    argv: Sequence[str] | None = None,
    *,
    root: Path | str | None = None,
    stages: Sequence[Stage] | None = None,
) -> int:
    """Run the Pipeline and return the exit code (lifecycle and codes in the module docstring).

    argv: the command-line arguments without the program name (default ``sys.argv[1:]``).
    root: the workspace root, the folder that holds config.py, plan.md and the Source_CSVs
    (default ``DEFAULT_ROOT``; ``run_pipeline.py`` passes its own folder, and checks pass a
    temporary workspace copy). stages: the stage registry (default ``STAGES``).
    """
    arguments = [str(arg) for arg in (sys.argv[1:] if argv is None else argv)]
    workspace = (DEFAULT_ROOT if root is None else Path(root)).resolve()
    registry = STAGES if stages is None else tuple(stages)
    parser = _parser(registry)
    usage_error: str | None = None
    from_stage: str | None = None
    try:
        from_stage = parser.parse_args(arguments).from_stage
    except _UsageError as exc:
        usage_error = f"invalid command line: {exc}\n{parser.format_usage().rstrip()}"
    except SystemExit as exc:  # --help printed the usage; no Run starts
        return exc.code if isinstance(exc.code, int) else EXIT_OK
    if not workspace.is_dir():
        print(f"{_PROG}: the workspace root {workspace} is not a folder", file=sys.stderr)
        return EXIT_CONFIG
    log_dir = workspace / OUTPUTS_FOLDER / LOG_FOLDER
    try:
        log = RunLog(log_dir, root=workspace, argv=arguments, echo=sys.stderr)
    except OSError as exc:
        print(f"{_PROG}: cannot create the Run_Log in {log_dir}: {exc}", file=sys.stderr)
        return EXIT_OTHER
    run = _Run(workspace, registry, log)
    code = EXIT_OTHER
    try:
        try:
            code = run.execute(from_stage, usage_error)
        except BaseException as exc:  # logged here; the end-of-Run check below still runs
            code = run.stopped(exc)
    finally:
        code = run.end(code)
    return code
