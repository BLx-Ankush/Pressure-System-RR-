"""Pipeline-output leakage tests (Requirement 7, criteria 9 and 11).

**Validates: Requirements 7.9, 7.11**

Design sections "Fold sandboxes and leakage control" and Data Models (``fit_registry``,
``view_k``). Both tests read a completed Run's checkpoints through the CheckpointStore, which
checks every file against its stage's manifest, and take the fold of every match from the
Fold_Table of the ``folds`` checkpoint. They recompute every check from the saved tables with
pandas: neither calls the leakage guard (``FitRegistry.register``, ``FitRegistry.load_saved``)
nor ``views.check_view``, whose verdicts they re-check.

Every fit of every saved fit registry (criterion 9)
    ``test_no_fit_trains_on_a_match_it_gives_output_to`` has one case per checkpoint with a fit
    registry: the model stages MODEL_STAGES (06-10), and any other stage whose checkpoint folder
    holds a ``fit_registry.parquet``. A fit's Training_Rows are the matches that
    ``fit_train_matches`` lists for it, early-stopping matches included, and its output goes to
    the matches that ``fit_predicted_matches`` lists. For every fit of ``fit_registry``:

    1. the side tables name only fits of ``fit_registry`` and hold the numbers of training,
       early-stopping and predicted matches that ``fit_registry`` records, so that no match
       escapes rules 2-4;
    2. no predicted match is a training match (criterion 9);
    3. no training match lies in a fold that the fit must exclude: its ``excluded_folds``, and
       its ``sandbox_k`` and ``inner_j`` where given, because a fit whose output serves fold k
       through sandbox k must exclude fold k, and an inner fit fold j as well (criteria 6-8);
    4. no predicted match lies in a fold that holds a training match of the fit, whatever folds
       the fit declares (criterion 6).

    A registry that lists no fit passes: ``model_choice`` registers none until task 26 adds the
    Alternative_Models.

Every ordered pair (k, j) of every sandbox view (criterion 11)
    ``test_view_rows_take_values_from_fits_that_exclude_their_fold_and_k`` has one case for the
    views of the ``skill`` stage (provenance column ``m1_fit``) and one for those of the
    ``remaining_runs`` stage, which add ``m2_fit``. For each view V_k (k = 1 ... D15), each
    provenance column and each row fold j, every fold-j row of V_k must name a fit that the fit
    registry of that stage or of an earlier model stage lists, and whose Training_Rows hold no
    match of fold j or of fold k (fold k alone when j = k). The fit must also be the one that the
    design names for the row: the inner fit of sandbox k with inner fold j, or the outer fit of
    sandbox k for a fold-k row. For an M2 fit this is more than the exclusion: M2^(j)(-k) also
    excludes folds j and k, but it was trained on view V_j, whose M1 values came from fits that
    may have seen fold k. The case fails unless all D15 x (D15 - 1) ordered pairs (20 with
    D15 = 5) and the D15 fold-k cells had rows that were checked, for every provenance column.

Skips and stale checkpoints
    A case is skipped, with the missing files named, while a checkpoint it reads is absent
    (``requires_outputs``), and with the reason while one is not complete. It fails when its
    checkpoint was built from another copy of a checkpoint it is checked against (each manifest
    records the manifest hash of every stage that the stage read), because the Fold_Table, the
    ``balls`` rows or the fits on disk are then not the ones that the stage used.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import pandas as pd
import pytest

from superr import folds, reconstruct, skill, views
from superr.leakage import FIT_PREDICTED_MATCHES, FIT_REGISTRY, FIT_REGISTRY_FILE, FIT_TRAIN_MATCHES
from superr.models import model_choice, outcome, remaining_runs, win_prob
from superr.pipeline import OUTPUTS_FOLDER, checkpoint_layout
from superr.store import MANIFEST_NAME, CheckpointStore

if TYPE_CHECKING:
    from superr.settings import Settings

# ------------------------------------------------------------------------------------- constants

#: The model stages (06-10). Each saves the fits it registered as a fit registry (design,
#: "Fold sandboxes and leakage control", paragraph "Leakage guard").
MODEL_STAGES: tuple[str, ...] = (
    skill.STAGE_NAME,
    remaining_runs.STAGE_NAME,
    win_prob.STAGE_NAME,
    outcome.STAGE_NAME,
    model_choice.STAGE_NAME,
)

#: The stages that save the sandbox views V_k, and the provenance columns of their views (Data
#: Models, ``view_k``): ``skill`` builds the views, ``remaining_runs`` extends them.
VIEW_FIT_COLUMNS: dict[str, tuple[str, ...]] = {
    skill.STAGE_NAME: (views.M1_FIT,),
    remaining_runs.STAGE_NAME: (views.M1_FIT, views.M2_FIT),
}

N_FOLDS_KEY = "design_choices.D15_n_folds"
RUN_HINT = r"run `.venv\Scripts\python run_pipeline.py` first"
_SHOWN = 5  # example match ids or rows in one problem line
_MAX_LINES = 40  # problem lines in one failure message

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / OUTPUTS_FOLDER
# Only for the checkpoint paths of the requires_outputs marks; the tests read through `checkpoints`.
_PATHS = CheckpointStore(OUTPUTS, checkpoint_layout(ROOT), config_sha256=hashlib.sha256(b"").hexdigest())


def _checkpoint_files(stage: str, *names: str) -> tuple[str, ...]:
    """The manifest and the files ``names`` of a stage's checkpoint, relative to outputs/."""
    folder = _PATHS.stage_dir(stage).relative_to(OUTPUTS)
    return tuple((folder / name).as_posix() for name in (MANIFEST_NAME, *names))


def _registry_files(stage: str) -> tuple[str, ...]:
    """The manifest and the three fit-registry tables of a stage's checkpoint, relative to outputs/."""
    tables = (FIT_REGISTRY, FIT_TRAIN_MATCHES, FIT_PREDICTED_MATCHES)
    return _checkpoint_files(stage, *(f"{name}.parquet" for name in tables))


FOLD_FILES = _checkpoint_files(folds.STAGE_NAME, f"{folds.FOLD_TABLE}.parquet")
BALLS_FILES = _checkpoint_files(reconstruct.STAGE_NAME, f"{reconstruct.BALLS_TABLE}.parquet")


def _registry_stages() -> tuple[str, ...]:
    """MODEL_STAGES and every other stage whose checkpoint folder holds a fit registry now, in run order."""
    saved = {stage for stage in _PATHS.stages if (_PATHS.stage_dir(stage) / FIT_REGISTRY_FILE).is_file()}
    return tuple(stage for stage in _PATHS.stages if stage in MODEL_STAGES or stage in saved)


def _supplier_stages(stage: str) -> tuple[str, ...]:
    """The model stages that run no later than ``stage``: their registries list every fit its views can name."""
    last = _PATHS.stages.index(stage)
    return tuple(name for name in MODEL_STAGES if _PATHS.stages.index(name) <= last)


def _view_files(stage: str) -> tuple[str, ...]:
    """The files the view test of ``stage`` reads up front; the views V_2 ... V_D15 must be in its checkpoint too."""
    return (
        *_checkpoint_files(stage, f"{views.view_table(1)}.parquet"),
        *(path for supplier in _supplier_stages(stage) for path in _registry_files(supplier)),
        *BALLS_FILES,
        *FOLD_FILES,
    )


# ------------------------------------------------------------------------------- shared helpers

@pytest.fixture(scope="module")
def checkpoints(settings: Settings, outputs_dir: Path, workspace_root: Path) -> CheckpointStore:
    """The Run's checkpoints, read through the store, which checks each file against its manifest."""
    return CheckpointStore(outputs_dir, checkpoint_layout(workspace_root), config_sha256=settings.effective_sha256())


def _require_complete(store: CheckpointStore, *stages: str) -> None:
    """Skip unless every stage's checkpoint is complete: a valid manifest, and every file it lists exists."""
    for stage in stages:
        reason = store.incomplete_reason(stage)
        if reason is not None:
            pytest.skip(f"the {stage} checkpoint is not complete ({reason}); {RUN_HINT}")


def _require_built_on(store: CheckpointStore, stage: str, *reads: str) -> None:
    """Fail unless the checkpoint of ``stage`` was built from the checkpoints of ``reads`` now on disk."""
    recorded = store.manifest(stage)["inputs"]
    stale = [read for read in reads if read != stage and recorded.get(read) != store.manifest_sha256(read)]
    if stale:
        pytest.fail(
            f"the {stage} checkpoint was not built from the {', '.join(stale)} checkpoint(s) now on disk: its "
            f"{MANIFEST_NAME} records another manifest hash, or none, for them. Its fits and views cannot be checked "
            f"against them; run `.venv\\Scripts\\python run_pipeline.py --from-stage {stage}`"
        )


def _lookup(mapping: pd.Series, keys: Any, what: str, source: str) -> np.ndarray:
    """``mapping``'s value for each key, in order; fails when ``source`` (the table behind ``mapping``) lacks a key."""
    keys = np.asarray(keys, dtype=np.int64)
    at = mapping.index.get_indexer(keys)
    absent = np.unique(keys[at < 0])
    if len(absent):
        pytest.fail(f"{len(absent):,} {what} are not in the {source} (for example {absent[:_SHOWN].tolist()})")
    return mapping.to_numpy()[at]


def _optional_int(value: Any) -> int | None:
    return None if pd.isna(value) else int(value)


def _failure(header: str, problems: list[str]) -> str:
    shown = problems[:_MAX_LINES]
    more = [f"  ... and {len(problems) - len(shown):,} more"] if len(problems) > len(shown) else []
    return "\n".join([header, *(f"  - {line}" for line in shown), *more])


@pytest.fixture(scope="module")
def fold_of_match(checkpoints: CheckpointStore, settings: Settings) -> pd.Series:
    """The Fold_Table as a Series match_id -> fold, checked to hold each match once and every fold 1 ... D15."""
    _require_complete(checkpoints, folds.STAGE_NAME)
    table = checkpoints.load_table(folds.STAGE_NAME, folds.FOLD_TABLE)
    ids = table["match_id"].to_numpy(dtype=np.int64)
    fold = table["fold"].to_numpy(dtype=np.int64)
    n_folds = settings.get(N_FOLDS_KEY)
    assert len(np.unique(ids)) == len(ids), "the Fold_Table lists a match more than once"
    present = np.unique(fold).tolist()
    assert present == list(range(1, n_folds + 1)), (
        f"the Fold_Table holds the folds {present}; with D15 = {n_folds} every fold from 1 to {n_folds} must hold matches"
    )
    return pd.Series(fold, index=pd.Index(ids, name="match_id"), name="fold")


@pytest.fixture(scope="module")
def match_of_row(checkpoints: CheckpointStore) -> pd.Series:
    """Each Delivery's match, from the ``balls`` table of the ``reconstruct`` checkpoint: row -> match_id."""
    _require_complete(checkpoints, reconstruct.STAGE_NAME)
    balls = checkpoints.load_table(reconstruct.STAGE_NAME, reconstruct.BALLS_TABLE)
    rows = balls["row"].to_numpy(dtype=np.int64)
    assert len(np.unique(rows)) == len(rows), "the balls table lists a row more than once"
    return pd.Series(balls["match_id"].to_numpy(dtype=np.int64), index=pd.Index(rows, name="row"), name="match_id")


# ------------------------------------------------------------------------------ saved fit registries

class SavedFits(NamedTuple):
    """One stage's saved fit registry, with the fold of every listed match from the Fold_Table."""

    fits: pd.DataFrame  # fit_registry: one row per fit
    train: pd.DataFrame  # fit_id, match_id, fold, early_stopping: one row per match of a fit's Training_Rows
    predicted: pd.DataFrame  # fit_id, match_id, fold: one row per match that receives a fit's output


def _saved_fits(store: CheckpointStore, stage: str, fold_of_match: pd.Series) -> SavedFits:
    """The three fit-registry tables of ``stage``, with fit ids as Python strings and the fold of each match."""
    fits = store.load_table(stage, FIT_REGISTRY)
    fits = fits.assign(fit_id=fits["fit_id"].to_numpy(dtype=object))
    sides = {}
    for name in (FIT_TRAIN_MATCHES, FIT_PREDICTED_MATCHES):
        frame = store.load_table(stage, name)
        ids = frame["match_id"].to_numpy(dtype=np.int64)
        sides[name] = pd.DataFrame(
            {
                "fit_id": frame["fit_id"].to_numpy(dtype=object),
                "match_id": ids,
                "fold": _lookup(fold_of_match, ids, f"match ids of the {stage} {name} table", "Fold_Table"),
            }
        )
        if name == FIT_TRAIN_MATCHES:
            sides[name]["early_stopping"] = frame["early_stopping"].to_numpy(dtype=bool)
    return SavedFits(fits, sides[FIT_TRAIN_MATCHES], sides[FIT_PREDICTED_MATCHES])


def _fit_labels(fits: pd.DataFrame) -> dict[str, str]:
    """fit id -> ``'<fit id>' (<model>[, <family>])``, for messages."""
    labels = {}
    for fit_id, model, family in zip(fits["fit_id"], fits["model"], fits["family"]):
        labels[fit_id] = f"{fit_id!r} ({model})" if pd.isna(family) else f"{fit_id!r} ({model}, {family})"
    return labels


def _side_table_problems(saved: SavedFits) -> list[str]:
    """Rule 1: the side tables name only listed fits and hold the match counts that fit_registry records."""
    fit_ids = saved.fits["fit_id"].to_numpy(dtype=object)
    problems = []
    repeated = sorted(fit_id for fit_id, count in Counter(fit_ids.tolist()).items() if count > 1)
    if repeated:
        problems.append(f"{FIT_REGISTRY} lists the fit id(s) {repeated[:_SHOWN]} more than once")
    listed = set(fit_ids.tolist())
    for name, frame in ((FIT_TRAIN_MATCHES, saved.train), (FIT_PREDICTED_MATCHES, saved.predicted)):
        unknown = sorted({str(fit_id) for fit_id in frame["fit_id"].tolist() if fit_id not in listed})
        if unknown:
            problems.append(f"{name} names {len(unknown):,} fit id(s) that {FIT_REGISTRY} does not list: {unknown[:_SHOWN]}")
    early = saved.train["early_stopping"].to_numpy(dtype=bool)
    sides = {
        "n_train_matches": saved.train.iloc[np.flatnonzero(~early)],
        "n_early_stopping_matches": saved.train.iloc[np.flatnonzero(early)],
        "n_predicted_matches": saved.predicted,
    }
    for column, frame in sides.items():
        found = frame.groupby("fit_id")["match_id"].nunique().reindex(fit_ids, fill_value=0).to_numpy(dtype=np.int64)
        recorded = saved.fits[column].to_numpy(dtype=np.int64)
        for at in np.flatnonzero(found != recorded):
            problems.append(
                f"fit {fit_ids[at]!r}: {FIT_REGISTRY} records {column} = {recorded[at]:,}, but its side table lists "
                f"{found[at]:,} distinct matches"
            )
    return problems


def _must_exclude(fits: pd.DataFrame) -> pd.DataFrame:
    """(fit_id, fold) for every fold that a fit must exclude: its excluded_folds, and sandbox_k and inner_j where given."""
    pairs = []
    for fit_id, excluded, k, j in zip(fits["fit_id"], fits["excluded_folds"], fits["sandbox_k"], fits["inner_j"]):
        must = {int(fold) for fold in excluded} | {int(fold) for fold in (k, j) if not pd.isna(fold)}
        pairs += [(fit_id, fold) for fold in sorted(must)]
    return pd.DataFrame(pairs, columns=["fit_id", "fold"]).astype({"fit_id": object, "fold": np.int64})


def _leak_problems(saved: SavedFits) -> list[str]:
    """Rules 2-4: one line for each fit and broken rule, with the count, the folds and example matches."""
    train, predicted = saved.train, saved.predicted
    keys = ["fit_id", "match_id"]
    rules = (
        (
            predicted.merge(train[keys], on=keys),
            "of its predicted matches are also training matches (criterion 9)",
        ),
        (
            train.merge(_must_exclude(saved.fits), on=["fit_id", "fold"]),
            "of its training matches lie in folds that it must exclude (criteria 6-8)",
        ),
        (
            predicted.merge(train[["fit_id", "fold"]].drop_duplicates(), on=["fit_id", "fold"]),
            "of its predicted matches lie in folds that hold training matches of the fit (criterion 6)",
        ),
    )
    labels = _fit_labels(saved.fits)
    problems = []
    for broken, text in rules:
        for fit_id, part in broken.groupby("fit_id", sort=False):
            part = part.drop_duplicates("match_id")
            examples = ", ".join(
                f"{match} (fold {fold})" for match, fold in zip(part["match_id"].head(_SHOWN), part["fold"].head(_SHOWN))
            )
            problems.append(
                f"fit {labels.get(fit_id, repr(fit_id))}: {len(part):,} {text}, in fold(s) "
                f"{sorted(set(part['fold'].tolist()))}; for example match {examples}"
            )
    return problems


@pytest.mark.parametrize(
    "stage",
    [
        pytest.param(stage, marks=pytest.mark.requires_outputs(*_registry_files(stage), *FOLD_FILES), id=stage)
        for stage in _registry_stages()
    ],
)
def test_no_fit_trains_on_a_match_it_gives_output_to(
    stage: str, checkpoints: CheckpointStore, fold_of_match: pd.Series
) -> None:
    """Every fit that the stage's checkpoint saved trains on no match that receives its output and on no
    match of a fold it must exclude, and gives output to no fold that it trained on.

    **Validates: Requirements 7.9**
    """
    _require_complete(checkpoints, stage)
    _require_built_on(checkpoints, stage, folds.STAGE_NAME)
    saved = _saved_fits(checkpoints, stage, fold_of_match)
    problems = _side_table_problems(saved) + _leak_problems(saved)
    assert not problems, _failure(
        f"the fit registry of the {stage} checkpoint ({len(saved.fits):,} fits) breaks Requirement 7, criteria 6-9:",
        problems,
    )


# ------------------------------------------------------------------------------------ sandbox views

class FitFacts(NamedTuple):
    """What the view check needs to know about one saved fit."""

    label: str  # fit id, model and family, for messages
    stage: str  # the stage whose fit registry lists the fit
    sandbox_k: int | None
    inner_j: int | None
    train_folds: frozenset[int]  # the folds of the matches of its Training_Rows, from the Fold_Table


def _fit_facts(store: CheckpointStore, stages: tuple[str, ...], fold_of_match: pd.Series) -> dict[str, FitFacts]:
    """fit id -> FitFacts for every fit of the fit registries of ``stages``; fails when two stages list one fit id."""
    facts: dict[str, FitFacts] = {}
    for stage in stages:
        saved = _saved_fits(store, stage, fold_of_match)
        train_folds = {fit_id: frozenset(part.tolist()) for fit_id, part in saved.train.groupby("fit_id")["fold"]}
        labels = _fit_labels(saved.fits)
        for fit_id, k, j in zip(saved.fits["fit_id"], saved.fits["sandbox_k"], saved.fits["inner_j"]):
            if fit_id in facts:
                pytest.fail(f"fit id {fit_id!r} is listed by the fit registries of both the {facts[fit_id].stage} and the {stage} checkpoint")
            facts[fit_id] = FitFacts(
                labels[fit_id], stage, _optional_int(k), _optional_int(j), train_folds.get(fit_id, frozenset())
            )
    return facts


def _cell_problems(fit: FitFacts | None, fit_id: str, k: int, fold: int, suppliers: tuple[str, ...]) -> list[str]:
    """What is wrong with fold-``fold`` rows of V_k naming fit ``fit_id`` (its FitFacts, or None when unknown)."""
    if fit is None:
        return [f"name fit {fit_id!r}, which no fit registry of the {', '.join(suppliers)} checkpoint(s) lists"]
    problems = []
    seen = sorted(fit.train_folds & {fold, k})
    if seen:
        problems.append(f"take their values from fit {fit.label}, whose Training_Rows hold matches of fold(s) {seen}")
    expected = (k, None if fold == k else fold)
    if (fit.sandbox_k, fit.inner_j) != expected:
        wanted = f"the outer fit of sandbox {k}" if fold == k else f"the inner fit of sandbox {k} with inner fold {fold}"
        problems.append(
            f"take their values from fit {fit.label}, registered with sandbox_k = {fit.sandbox_k} and "
            f"inner_j = {fit.inner_j}; the design names {wanted}"
        )
    return problems


@pytest.mark.parametrize(
    "stage",
    [pytest.param(stage, marks=pytest.mark.requires_outputs(*_view_files(stage)), id=stage) for stage in VIEW_FIT_COLUMNS],
)
def test_view_rows_take_values_from_fits_that_exclude_their_fold_and_k(
    stage: str,
    checkpoints: CheckpointStore,
    fold_of_match: pd.Series,
    match_of_row: pd.Series,
    settings: Settings,
) -> None:
    """In every view V_k of the stage, the fits named by the provenance columns of the fold-j rows trained
    on no match of fold j or fold k (fold k for the fold-k rows) and are the fits the design names, for all
    20 ordered pairs (k, j) and the 5 fold-k cells.

    **Validates: Requirements 7.11**
    """
    n_folds = settings.get(N_FOLDS_KEY)
    columns = VIEW_FIT_COLUMNS[stage]
    suppliers = _supplier_stages(stage)
    _require_complete(checkpoints, *suppliers)
    _require_built_on(checkpoints, stage, reconstruct.STAGE_NAME, folds.STAGE_NAME, *suppliers)
    facts = _fit_facts(checkpoints, suppliers, fold_of_match)

    problems: list[str] = []
    checked: Counter[tuple[str, int, int]] = Counter()  # (column, k, row fold) -> rows checked
    for k in range(1, n_folds + 1):
        name = views.view_table(k)
        view = checkpoints.load_table(stage, name)
        absent = [column for column in (views.ROW_COLUMN, *columns) if column not in view.columns]
        if absent:
            problems.append(f"V_{k} ({name}) has no column {', '.join(absent)}")
            continue
        rows = view[views.ROW_COLUMN].to_numpy(dtype=np.int64)
        if len(np.unique(rows)) != len(rows):
            problems.append(f"V_{k} ({name}) lists a row more than once")
        matches = _lookup(match_of_row, rows, f"rows of {stage} {name}", "balls table of the reconstruct checkpoint")
        row_fold = _lookup(fold_of_match, matches, f"match ids of the rows of {stage} {name}", "Fold_Table")
        for column in columns:
            fit_ids = view[column].to_numpy(dtype=object)
            missing = pd.isna(fit_ids)
            if missing.any():
                problems.append(
                    f"V_{k} {column}: {int(missing.sum()):,} rows name no fit (for example rows "
                    f"{rows[missing][:_SHOWN].tolist()})"
                )
            cells = pd.DataFrame({"fold": row_fold[~missing], "fit_id": fit_ids[~missing]}).value_counts(sort=False)
            for (fold, fit_id), n_rows in cells.items():
                fold = int(fold)
                checked[(column, k, fold)] += int(n_rows)
                for problem in _cell_problems(facts.get(fit_id), fit_id, k, fold, suppliers):
                    problems.append(f"V_{k} {column}: {int(n_rows):,} fold-{fold} rows {problem}")

    folds_ = range(1, n_folds + 1)
    for column in columns:
        for k in folds_:
            for j in folds_:
                if not checked[(column, k, j)]:
                    what = f"the pair (k, j) = ({k}, {j})" if j != k else f"the fold-{k} rows of V_{k}"
                    problems.append(f"V_{k} {column}: no fold-{j} row was checked, so {what} went untested")
    n_pairs = sum(1 for (_, k, j), n_rows in checked.items() if j != k and n_rows)
    assert not problems, _failure(
        f"the views of the {stage} checkpoint break Requirement 7, criterion 11 ({n_pairs} of "
        f"{len(columns) * n_folds * (n_folds - 1)} (column, k, j) pairs with rows; {sum(checked.values()):,} rows checked):",
        problems,
    )
