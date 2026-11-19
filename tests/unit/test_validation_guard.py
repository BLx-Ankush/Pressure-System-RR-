"""Validation leakage guard (Requirement 14, criterion 11; task 16.2).

**Validates: Requirements 14.11**

Criterion 11: if any Validation_Delivery lacks an Out_Of_Fold_Prediction of WP, LI or PI, or any
of those values comes from a model whose training data include the Delivery's match, the
Pressure_Validator marks the run as failed, reports the number of affected Deliveries and writes no
validation tables or charts. ``validation.oof_guard`` raises ValidationLeakageError, a LeakageError
with exit code 4 whose ``affected`` and message give the count, and ``validation.run_stage`` runs
the guard before it computes, saves or writes anything.

Synthetic tests (no Pipeline outputs needed)
    A hand-made Run with 3 folds (``folds.MIN_SANDBOX_FOLDS``), two matches per fold, 2 Deliveries
    per innings and one Wide. Its FitRegistry holds one legitimate outer fit per fold of every
    model behind WP, LI and PI: M1, the first-innings M2, the M3 classifier and calibrator of each
    innings (``logistic`` in innings 1, ``lgbm`` in innings 2) and M4. Each fit excludes its fold,
    trains on the matches of the other folds and predicts the matches of its own fold, so every
    Validation_Delivery's inputs are Out_Of_Fold. Match 302 is an excluded match: no fit trains on
    it, its 4 Deliveries are not Validation_Deliveries, and its WP fit is deliberately one that does
    not exclude its fold, so the guard must leave it out of the check and of the count. The counts
    expected below follow from this construction, not from the guard's output.

    * ``test_the_guard_passes_when_every_validation_delivery_is_out_of_fold``: the 20
      Validation_Deliveries (5 matches x 2 innings x 2 Deliveries) pass every check.
    * ``test_a_wp_fit_that_trained_on_the_match_fails_the_guard_with_the_affected_count``: the WP
      classifier fit of k Validation_Deliveries (k = 1, 3, 5; for k = 5 the calibrator too) is
      replaced by the next fold's outer fit of the same model, a legitimate fit whose Training_Rows
      hold the Delivery's match. The guard raises exit code 4 with exactly k affected Deliveries,
      their rows and the 20 checked, and the message gives the count. Only the replaced sources
      fail, and two failing sources on one Delivery count once.
    * ``test_a_validation_delivery_without_an_out_of_fold_wp_fails_the_guard``: 2
      Validation_Deliveries with no WP fit recorded, or with no WP value, fail with a count of 2.

Stage test (needs a completed Run; skipped otherwise)
    ``test_the_validation_stage_fails_with_the_count_and_writes_nothing``: ``run_stage`` runs on the
    Run's checkpoints with ``guard_inputs`` wrapped so that one Validation_Delivery of each of the
    D15 folds takes its WP from another fold's outer M3 fit whose Training_Rows hold the Delivery's
    match. The stage fails with exit code 4 and the count of those Deliveries, and nothing is saved
    or written: no validation checkpoint (tables or manifest) and no file under
    ``outputs/validation/``.

    The stage reads about 150 MB of checkpoints before its guard (the M4 predictions alone are
    58 MB in the delivered Run), so the test does not copy them. Its store, ``_ReadThroughStore``,
    is a CheckpointStore rooted in a temporary workspace that loads the stages before
    ``validation`` from the Run's real store, each file checked against its manifest there, and
    saves, clears and completes only under ``tmp_path``. The real store is used for loading only,
    and the context's Config, Run_Log and outputs folder are the temporary workspace's, so the test
    writes nothing to the real ``outputs/``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import shutil
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
import pytest

from superr import folds, pressure, reconstruct, skill, validation
from superr.errors import LeakageError
from superr.folds import FitSpec, sandbox_plan
from superr.leakage import FIT_REGISTRY_FILE, FitRegistry, fit_id_for
from superr.models import model_choice, outcome, remaining_runs, win_prob
from superr.pipeline import DESIGN_STAGE_NAMES, OUTPUTS_FOLDER, StageContext, checkpoint_layout
from superr.report.constants import ConstantRegistry
from superr.runlog import LOG_FOLDER, RunLog
from superr.settings import CONFIG_FILENAME, load_settings
from superr.state import STATE_INNINGS
from superr.store import MANIFEST_NAME, CheckpointStore
from superr.validation import CHECKS, FIT_SOURCES, GuardInputs, GuardResult, ValidationLeakageError, oof_guard

#: Exit code of a leakage-guard stop (design, "Run lifecycle"; Requirement 14, criterion 11).
EXIT_LEAKAGE = 4

#: The WP sources of GuardInputs.sources that the tests replace, and the model each names per innings.
WP_SOURCE_MODELS: Mapping[str, Mapping[int, str]] = MappingProxyType(
    {"wp_fit": win_prob.M3_MODELS, "wp_calibrator": win_prob.CALIBRATOR_MODELS}
)
#: The WP values of GuardInputs.values: the pressure table's, the Primary_Models' and the M3 table's.
WP_VALUES: tuple[str, ...] = ("wp_pressure", "wp_primary", "wp_m3_table")


# ------------------------------------------------------------------------------------- helpers

def _outer_specs(n_folds: int) -> dict[int, FitSpec]:
    """The outer FitSpec of each fold k (it excludes fold k only), from folds.sandbox_plan."""
    return {spec.sandbox_k: spec for spec in sandbox_plan(n_folds) if spec.kind == folds.OUTER}


def _outer_fit(model: str, fold: int, n_folds: int, family: str | None = None) -> str:
    """The Run-wide fit id of ``model``'s outer fit of fold ``fold`` (leakage.fit_id_for)."""
    return fit_id_for(model, _outer_specs(n_folds)[int(fold)], family=family)


def _next_fold(fold: int, n_folds: int) -> int:
    """The fold after ``fold``, wrapping round: always another fold."""
    return int(fold) % n_folds + 1


def _next_fold_wp_fit(inputs: GuardInputs, position: int, name: str, n_folds: int) -> str:
    """The next fold's outer fit of the model behind WP source ``name`` of the Delivery at ``position``.

    The fit has the model and family of the Delivery's own WP fit, but excludes another fold and
    trains on the Delivery's fold.
    """
    number = int(inputs.deliveries["innings"].iloc[position])
    fold = int(inputs.deliveries["fold"].iloc[position])
    family = str(inputs.candidates[model_choice.M3_MODELS[number]])
    return _outer_fit(WP_SOURCE_MODELS[name][number], _next_fold(fold, n_folds), n_folds, family)


def _with_sources(inputs: GuardInputs, changes: Mapping[tuple[int, str], str | None]) -> GuardInputs:
    """``inputs`` with the fit id of each (position, source) of ``changes`` replaced; None: no fit recorded."""
    sources = inputs.sources.copy()
    for (position, name), fit_id in changes.items():
        sources.iat[position, sources.columns.get_loc(name)] = fit_id
    return dataclasses.replace(inputs, sources=sources)


def _with_values(inputs: GuardInputs, positions: Sequence[int], names: Sequence[str], value: float) -> GuardInputs:
    """``inputs`` with the values ``names`` of the Deliveries at ``positions`` set to ``value``."""
    values = inputs.values.copy()
    for name in names:
        values.iloc[list(positions), values.columns.get_loc(name)] = value
    return dataclasses.replace(inputs, values=values)


def _failed_checks(table: pd.DataFrame) -> dict[tuple[str, str], int]:
    """(check, source) -> n_affected for every row of a guard table (GUARD_COLUMNS) with an affected Delivery."""
    failed = table[table["n_affected"] > 0]
    return {
        (str(check), str(source)): int(count)
        for check, source, count in zip(failed["check"], failed["source"], failed["n_affected"])
    }


def _assert_fails(error: ValidationLeakageError, rows: Sequence[int], n_validation: int) -> dict[tuple[str, str], int]:
    """The failure has exit code 4 and reports the affected Deliveries ``rows`` among ``n_validation``.

    Returns the failed checks of the error's table.
    """
    assert isinstance(error, LeakageError)
    assert error.exit_code == EXIT_LEAKAGE == LeakageError.exit_code
    assert error.affected == len(rows)
    assert error.rows.tolist() == sorted(rows)
    assert error.n_validation == n_validation
    assert f"{len(rows):,} of the {n_validation:,} Validation_Deliveries fail" in str(error)
    return _failed_checks(error.table)


# ----------------------------------------------------------------------------- the synthetic Run

N_FOLDS = folds.MIN_SANDBOX_FOLDS  # 3

#: The Primary_Models (GuardInputs.candidates): a different M3 family in each innings.
CANDIDATES: Mapping[str, str] = MappingProxyType(
    {
        model_choice.M3_MODELS[1]: win_prob.LOGISTIC,
        model_choice.M3_MODELS[2]: win_prob.LGBM,
        model_choice.M4_MODEL: outcome.LGBM_FAMILY,
    }
)

#: The Fold_Table: two matches per fold. Match 302 is excluded (Requirement 5, criterion 24).
FOLD_OF_MATCH: Mapping[int, int] = MappingProxyType({101: 1, 102: 1, 201: 2, 202: 2, 301: 3, 302: 3})
EXCLUDED_MATCH = 302
DELIVERIES_PER_INNINGS = 2
WIDE = (201, 1, 1)  # (match, innings, Delivery of the innings): a Bowler_Delivery only, with no batter-Role PI
N_VALIDATION = 20  # 5 matches x 2 innings x 2 Deliveries; match 302's 4 Deliveries are not Validation_Deliveries
N_VALIDATION_MATCHES = 5

#: (model, family) of every fit of the synthetic Run, one outer fit per fold each: the models
#: behind WP, LI and PI of both innings (validation.sandbox_models).
MODELS: tuple[tuple[str, str | None], ...] = (
    (skill.MODEL, None),
    (remaining_runs.FIRST_INNINGS_MODEL, remaining_runs.FAMILY),
    (win_prob.M3_MODELS[1], CANDIDATES[model_choice.M3_MODELS[1]]),
    (win_prob.CALIBRATOR_MODELS[1], CANDIDATES[model_choice.M3_MODELS[1]]),
    (win_prob.M3_MODELS[2], CANDIDATES[model_choice.M3_MODELS[2]]),
    (win_prob.CALIBRATOR_MODELS[2], CANDIDATES[model_choice.M3_MODELS[2]]),
    (outcome.MODEL, CANDIDATES[model_choice.M4_MODEL]),
)

#: The Validation_Deliveries that the leak test corrupts, in order: the first Delivery of each
#: (match, innings), across the three folds and both innings.
LEAKY: tuple[tuple[int, int], ...] = ((101, 1), (202, 2), (301, 1), (102, 2), (201, 2))
#: The Validation_Deliveries that the missing-WP test strips.
LACKING: tuple[tuple[int, int], ...] = ((101, 2), (301, 2))


def _synthetic_registry() -> FitRegistry:
    """One legitimate outer fit per fold of every model of MODELS, each accepted by the leakage guard."""
    ids = np.array(list(FOLD_OF_MATCH), dtype=np.int64)
    fold_of = np.array(list(FOLD_OF_MATCH.values()), dtype=np.int64)
    registry = FitRegistry(pd.DataFrame({"match_id": ids, "fold": fold_of}))
    trainable = ids != EXCLUDED_MATCH
    for k, spec in _outer_specs(N_FOLDS).items():
        train = ids[trainable & np.isin(fold_of, spec.train_folds)]
        predicted = ids[fold_of == k]
        for model, family in MODELS:
            registry.register_spec(model, spec, train, predicted, family=family)
    return registry


def _synthetic_inputs() -> GuardInputs:
    """GuardInputs of the synthetic Run: one row per Delivery of innings 1 and 2, in match order.

    Every Delivery takes each of its fits from the outer fits of its own fold, and its WP, LI and
    PI values agree across tables. The one exception is the WP fit of match 302's Deliveries,
    which is the next fold's outer fit and so does not exclude match 302's fold.
    """
    cells = [
        (match, fold, number, ball)
        for match, fold in FOLD_OF_MATCH.items()
        for number in STATE_INNINGS
        for ball in range(DELIVERIES_PER_INNINGS)
    ]
    match = np.array([cell[0] for cell in cells], dtype=np.int64)
    fold = np.array([cell[1] for cell in cells], dtype=np.int64)
    innings = np.array([cell[2] for cell in cells], dtype=np.int64)
    wide = np.array([(cell[0], cell[2], cell[3]) == WIDE for cell in cells])
    n_rows = len(cells)
    wp = np.linspace(0.2, 0.8, n_rows)
    li = np.linspace(0.5, 2.5, n_rows)
    deliveries = pd.DataFrame(
        {
            "row": np.arange(n_rows, dtype=np.int64),
            "match_id": match,
            "innings": innings,
            "fold": fold,
            "validation": match != EXCLUDED_MATCH,
            "batter": ~wide,
        }
    )
    values = pd.DataFrame(
        {
            **{name: wp.copy() for name in WP_VALUES},
            "li_pressure": li.copy(),
            "li_primary": li.copy(),
            "pi_bat": np.where(wide, np.nan, 62.5),  # a Wide has no batter-Role PI (Requirement 13, criterion 11)
            "pi_bowl": np.full(n_rows, 37.5),
            "pi_bowl_extras_free": np.full(n_rows, 40.0),
            "fold_primary": fold.copy(),
            "fold_m3_table": fold.copy(),
        }
    )
    sources: dict[str, list[str]] = {name: [] for name in FIT_SOURCES}
    for m, f, number in zip(match.tolist(), fold.tolist(), innings.tolist()):
        family = CANDIDATES[model_choice.M3_MODELS[number]]
        wp_fold = _next_fold(f, N_FOLDS) if m == EXCLUDED_MATCH else f
        fits = {
            "wp_fit": _outer_fit(win_prob.M3_MODELS[number], wp_fold, N_FOLDS, family),
            "wp_calibrator": _outer_fit(win_prob.CALIBRATOR_MODELS[number], f, N_FOLDS, family),
            "wp_outer_fit": _outer_fit(win_prob.M3_MODELS[number], f, N_FOLDS, family),
            "wp_outer_calibrator": _outer_fit(win_prob.CALIBRATOR_MODELS[number], f, N_FOLDS, family),
            "m4_fit": _outer_fit(outcome.MODEL, f, N_FOLDS, CANDIDATES[model_choice.M4_MODEL]),
            "m4_m1_fit": _outer_fit(skill.MODEL, f, N_FOLDS),
            "view_m1_fit": _outer_fit(skill.MODEL, f, N_FOLDS),
            "view_m2_fit": _outer_fit(remaining_runs.FIRST_INNINGS_MODEL, f, N_FOLDS, remaining_runs.FAMILY),
        }
        for name in FIT_SOURCES:
            sources[name].append(fits[name])
    return GuardInputs(
        candidates=CANDIDATES, deliveries=deliveries, values=values, sources=pd.DataFrame(sources, dtype=object)
    )


def _first_deliveries(inputs: GuardInputs, cells: Iterable[tuple[int, int]]) -> list[int]:
    """The position in ``inputs`` of the first Delivery of each (match, innings) of ``cells``."""
    match = inputs.deliveries["match_id"].to_numpy()
    innings = inputs.deliveries["innings"].to_numpy()
    return [int(np.flatnonzero((match == m) & (innings == i))[0]) for m, i in cells]


# ------------------------------------------------------------------------------ synthetic tests

def test_the_guard_passes_when_every_validation_delivery_is_out_of_fold() -> None:
    """Every WP, LI and PI of a Validation_Delivery comes from a fit that excluded its fold and did not
    train on its match: the guard passes, checking every fit and only the 20 Validation_Deliveries.

    **Validates: Requirements 14.11**
    """
    registry = _synthetic_registry()

    result = oof_guard(_synthetic_inputs(), registry)

    assert isinstance(result, GuardResult)
    assert (result.n_validation, result.n_matches) == (N_VALIDATION, N_VALIDATION_MATCHES)
    assert result.n_fits == len(registry) == len(MODELS) * N_FOLDS  # every fit backs some Validation_Delivery
    assert set(result.table["check"]) == set(CHECKS)
    assert result.table["n_affected"].eq(0).all(), _failed_checks(result.table)


@pytest.mark.parametrize(
    ("n_deliveries", "names"),
    [
        pytest.param(1, ("wp_fit",), id="1-delivery"),
        pytest.param(3, ("wp_fit",), id="3-deliveries"),
        pytest.param(5, ("wp_fit", "wp_calibrator"), id="5-deliveries-classifier-and-calibrator"),
    ],
)
def test_a_wp_fit_that_trained_on_the_match_fails_the_guard_with_the_affected_count(
    n_deliveries: int, names: tuple[str, ...]
) -> None:
    """A Validation_Delivery whose WP comes from a fit that trained on its match fails the guard
    (exit code 4), and the error counts each such Delivery once.

    **Validates: Requirements 14.11**
    """
    registry = _synthetic_registry()
    inputs = _synthetic_inputs()
    positions = _first_deliveries(inputs, LEAKY[:n_deliveries])
    changes = {
        (position, name): _next_fold_wp_fit(inputs, position, name, N_FOLDS) for position in positions for name in names
    }
    match = inputs.deliveries["match_id"].to_numpy()
    for (position, _), fit_id in changes.items():  # each replacement is a registered fit that trained on the match
        assert match[position] in registry.get(fit_id).training_match_ids, fit_id

    with pytest.raises(ValidationLeakageError) as caught:
        oof_guard(_with_sources(inputs, changes), registry)

    rows = inputs.deliveries["row"].to_numpy()[positions].tolist()
    failed = _assert_fails(caught.value, rows, N_VALIDATION)
    assert {source for _, source in failed} == set(names), failed  # no other source or value fails
    for name in names:
        assert failed[("fit_excludes_match", name)] == n_deliveries, failed


def _no_wp_fit(inputs: GuardInputs, positions: Sequence[int]) -> GuardInputs:
    """No fit is recorded behind the WP of the Deliveries at ``positions``."""
    return _with_sources(inputs, {(position, "wp_fit"): None for position in positions})


def _no_wp_value(inputs: GuardInputs, positions: Sequence[int]) -> GuardInputs:
    """No table holds a WP for the Deliveries at ``positions``."""
    return _with_values(inputs, positions, WP_VALUES, np.nan)


@pytest.mark.parametrize(
    ("strip", "check"),
    [
        pytest.param(_no_wp_fit, ("fit_registered", "wp_fit"), id="no-wp-fit-recorded"),
        pytest.param(_no_wp_value, ("value", "wp"), id="no-wp-value"),
    ],
)
def test_a_validation_delivery_without_an_out_of_fold_wp_fails_the_guard(
    strip: Callable[[GuardInputs, Sequence[int]], GuardInputs], check: tuple[str, str]
) -> None:
    """A Validation_Delivery that lacks an Out_Of_Fold_Prediction of WP fails the guard (exit code 4),
    with the number of such Deliveries.

    **Validates: Requirements 14.11**
    """
    registry = _synthetic_registry()
    inputs = _synthetic_inputs()
    positions = _first_deliveries(inputs, LACKING)

    with pytest.raises(ValidationLeakageError) as caught:
        oof_guard(strip(inputs, positions), registry)

    rows = inputs.deliveries["row"].to_numpy()[positions].tolist()
    failed = _assert_fails(caught.value, rows, N_VALIDATION)
    assert failed[check] == len(positions), failed


# ------------------------------------------------------------------------------ the validation stage

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / OUTPUTS_FOLDER
# Only for the checkpoint paths of the requires_outputs mark; the test reads through the Run's store.
_PATHS = CheckpointStore(OUTPUTS, checkpoint_layout(ROOT), config_sha256=hashlib.sha256(b"").hexdigest())

_N_FOLDS = "design_choices.D15_n_folds"

#: The Pipeline_Stages before ``validation``: the stage loads their checkpoints, and under
#: ``--from-stage validation`` the runner reloads the fits they saved into ``ctx.registry``.
EARLIER_STAGES: tuple[str, ...] = DESIGN_STAGE_NAMES[: DESIGN_STAGE_NAMES.index(validation.STAGE_NAME)]

#: The checkpoints that run_stage reads before its guard, and the model stages whose saved fits the guard checks.
GUARD_STAGES: tuple[str, ...] = (
    reconstruct.STAGE_NAME, folds.STAGE_NAME, remaining_runs.STAGE_NAME, win_prob.STAGE_NAME, outcome.STAGE_NAME,
    model_choice.STAGE_NAME, pressure.STAGE_NAME,
)  # fmt: skip
FIT_STAGES: tuple[str, ...] = (
    skill.STAGE_NAME, remaining_runs.STAGE_NAME, win_prob.STAGE_NAME, outcome.STAGE_NAME, model_choice.STAGE_NAME,
)  # fmt: skip


def _checkpoint_files(stage: str, *names: str) -> tuple[str, ...]:
    """The manifest and the files ``names`` of a stage's checkpoint, relative to outputs/."""
    folder = _PATHS.stage_dir(stage).relative_to(OUTPUTS)
    return tuple((folder / name).as_posix() for name in (MANIFEST_NAME, *names))


STAGE_FILES: tuple[str, ...] = tuple(
    dict.fromkeys(
        [
            *(path for stage in GUARD_STAGES for path in _checkpoint_files(stage)),
            *(path for stage in FIT_STAGES for path in _checkpoint_files(stage, FIT_REGISTRY_FILE)),
        ]
    )
)


class _ReadThroughStore(CheckpointStore):
    """A CheckpointStore in a temporary workspace whose loads from the stages ``reads`` come from ``source``.

    ``source`` is the store of the Run's real outputs/ and serves loading only: each file is
    checked against its stage's manifest there, and nothing is saved, cleared or completed through
    it. Every other stage, ``validation`` included, lives under the temporary outputs folder.
    """

    def __init__(
        self,
        outputs: Path,
        layout: Iterable[tuple[str, Path]],
        *,
        config_sha256: str,
        source: CheckpointStore,
        reads: Iterable[str],
    ) -> None:
        super().__init__(outputs, layout, config_sha256=config_sha256)
        self._real = source
        self._reads = frozenset(reads)

    def load_table(self, stage: str, name: str) -> pd.DataFrame:
        return self._real.load_table(stage, name) if stage in self._reads else super().load_table(stage, name)

    def load_array(self, stage: str, name: str) -> np.ndarray:
        return self._real.load_array(stage, name) if stage in self._reads else super().load_array(stage, name)

    def load_json(self, stage: str, name: str) -> Any:
        return self._real.load_json(stage, name) if stage in self._reads else super().load_json(stage, name)

    def load_model(self, stage: str, name: str) -> Any:
        return self._real.load_model(stage, name) if stage in self._reads else super().load_model(stage, name)


def _leaky_deliveries(inputs: GuardInputs, registry: FitRegistry, n_folds: int) -> dict[tuple[int, str], str]:
    """One Validation_Delivery of each fold, with the next fold's outer M3 fit to name as its WP fit.

    In fold f: the first Validation_Delivery of innings 1 (odd f) or 2 (even f) whose match that fit
    trained on. Returns (position, ``"wp_fit"``) -> the fit id.
    """
    deliveries = inputs.deliveries
    validation_flags = deliveries["validation"].to_numpy(dtype=bool)
    match = deliveries["match_id"].to_numpy(dtype=np.int64)
    innings = deliveries["innings"].to_numpy(dtype=np.int64)
    fold = deliveries["fold"].to_numpy(dtype=np.int64)
    changes: dict[tuple[int, str], str] = {}
    for f in range(1, n_folds + 1):
        number = STATE_INNINGS[(f - 1) % len(STATE_INNINGS)]
        family = str(inputs.candidates[model_choice.M3_MODELS[number]])
        fit_id = _outer_fit(win_prob.M3_MODELS[number], _next_fold(f, n_folds), n_folds, family)
        trained = registry.get(fit_id).training_match_ids
        at = np.flatnonzero(validation_flags & (innings == number) & (fold == f) & np.isin(match, trained))
        assert len(at), f"no Validation_Delivery of fold {f}, innings {number} is in a match that {fit_id} trained on"
        changes[(int(at[0]), "wp_fit")] = fit_id
    return changes


@pytest.mark.requires_outputs(*STAGE_FILES)
def test_the_validation_stage_fails_with_the_count_and_writes_nothing(
    tmp_path: Path, workspace_root: Path, outputs_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Validation_Delivery whose WP comes from a fit that trained on its match stops the ``validation``
    stage with exit code 4 and the count of such Deliveries, before any table, chart or checkpoint is
    written (module docstring, "Stage test").

    The context is the one the runner gives the stage under ``--from-stage validation``: the fits
    saved by the earlier stages reloaded into the FitRegistry, and the runner's set-up of a stage
    (stage tags, an empty checkpoint folder) before the stage function runs.

    **Validates: Requirements 14.11**
    """
    root = (tmp_path / "workspace").resolve()
    root.mkdir()
    shutil.copyfile(workspace_root / CONFIG_FILENAME, root / CONFIG_FILENAME)
    run_settings = load_settings(root)
    outputs = run_settings.path("outputs")
    assert outputs.resolve() == root / OUTPUTS_FOLDER  # the stage's outputs folder lies under tmp_path
    real = CheckpointStore(outputs_dir, checkpoint_layout(workspace_root), config_sha256=run_settings.effective_sha256())
    store = _ReadThroughStore(
        outputs, checkpoint_layout(root), config_sha256=run_settings.effective_sha256(), source=real,
        reads=EARLIER_STAGES,
    )  # fmt: skip
    registry = FitRegistry.for_store(store)
    for stage in EARLIER_STAGES:  # pipeline._Run._start_index
        registry.load_saved(real, stage)
    ctx = StageContext(
        settings=run_settings, store=store, log=RunLog(outputs / LOG_FOLDER, root=root), root=root,
        constants=ConstantRegistry.from_settings(run_settings), registry=registry,
    )  # fmt: skip
    n_folds = int(run_settings.get(_N_FOLDS))
    real_guard_inputs = validation.guard_inputs
    seen: dict[str, Any] = {}

    def leaky_guard_inputs(*args: Any, **kwargs: Any) -> GuardInputs:
        clean = real_guard_inputs(*args, **kwargs)
        changes = _leaky_deliveries(clean, registry, n_folds)
        seen.update(clean=clean, changes=changes)
        return _with_sources(clean, changes)

    monkeypatch.setattr(validation, "guard_inputs", leaky_guard_inputs)
    ctx.constants.begin_stage(validation.STAGE_NAME)  # pipeline._Run._run_stages, before the stage function
    ctx.registry.begin_stage(validation.STAGE_NAME)
    ctx.log.start_stage(validation.STAGE_NAME)
    store.clear_stage(validation.STAGE_NAME)

    with pytest.raises(ValidationLeakageError) as caught:
        validation.run_stage(ctx)

    clean, changes = seen["clean"], seen["changes"]
    positions = [position for position, _ in changes]
    rows = clean.deliveries["row"].to_numpy()[positions].tolist()
    n_validation = int(clean.deliveries["validation"].to_numpy(dtype=bool).sum())
    failed = _assert_fails(caught.value, rows, n_validation)
    assert len(rows) == n_folds
    assert {source for _, source in failed} == {"wp_fit"}, failed  # the Run's own inputs pass every check
    assert failed[("fit_excludes_match", "wp_fit")] == n_folds, failed

    assert not store.stage_dir(validation.STAGE_NAME).exists()  # no checkpoint table and no manifest
    assert not store.is_complete(validation.STAGE_NAME)
    assert not (outputs / validation.VALIDATION_FOLDER).exists()  # no validation table or chart
    written = sorted(path.relative_to(outputs).as_posix() for path in outputs.rglob("*") if path.is_file())
    assert [path for path in written if not path.startswith(f"{LOG_FOLDER}/")] == []
