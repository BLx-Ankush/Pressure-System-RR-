# Feature: pressure-rating-system, Property 19: Leakage guard
"""Property 19: Leakage guard (Requirement 2, criterion 6; Requirement 7, criteria 6 and 12;
Requirement 10, criterion 7).

**Validates: Requirements 2.6, 7.6, 7.12, 10.7**

Property 19 (design): for any Fold_Table and any fit specification (sandbox k, optional inner fold
j), the guard accepts the fit exactly when its Training_Rows, and its early-stopping validation
matches, contain no match from an excluded fold ({k} for an outer fit, {j, k} for an inner fit);
the validation matches are whole matches drawn from the training matches; ``check_view`` rejects
exactly the view rows whose recorded M1 or M2 fit does not exclude both the row's fold and k (k
alone for rows in fold k); and every rejection stops the Run with the model name, the excluded
folds and the overlap count before any pressure, value or rating output is written.

The examples start from ``tests.strategies.fold_tables(fits=True)``: matches with
Competition_Types, a fold count n from 3-7 (D15 most often), a fold seed, the Fold_Table with no
empty fold, 0-3 Deliveries per match and 1-5 FitCases, valid or leaking into the Training_Rows or
into the early-stopping matches only. Every expected verdict, fold set and count below is worked
out in this file from the Fold_Table; none is read back from the FitRegistry.

* ``test_guard_accepts_exactly_the_fits_without_excluded_fold_matches``: each FitCase, plus one
  fit that may leak into both sets at once, goes through ``register_spec`` and ``register``. A fit
  is accepted and recorded exactly when no training or early-stopping match lies in an excluded
  fold; otherwise a LeakageError (exit code 4) is raised, nothing is recorded, and the message
  names the model, the excluded fold numbers and the number of overlapping matches.
* ``test_early_stopping_split_holds_out_whole_training_matches``: ``early_stopping_split`` splits
  the distinct training matches of each FitCase into two disjoint sets of whole matches, holds out
  the configured share, gives the same split for the same seed and fit index whatever the row
  order, and the guard judges the split fit exactly as it judges its training matches.
* ``test_check_view_rejects_exactly_the_rows_whose_fit_does_not_exclude_their_fold_and_k``: all
  sandbox fits of M1 and M2 are registered, a correct view V_k is built and accepted, then the
  ``m1_fit`` and ``m2_fit`` of drawn rows are overwritten with other fits, an unregistered fit or
  no fit. ``check_view`` rejects exactly the rows whose recorded fit does not exclude the row's
  fold and k, and names the model, excluded folds and overlap count of each fit at fault.
* ``test_a_leakage_error_stops_the_run_before_any_pressure_value_or_rating_output`` (two
  examples): ``pipeline.main`` runs stub stages in a temporary workspace (a copy of ``config.py``
  and placeholder Source_CSVs and ``plan.md``). A stage whose fit, or whose view, the guard rejects
  ends the Run with exit code 4, no later stage runs or writes, and the Run_Log and standard error
  report the model name, the excluded folds and the overlap count.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import pytest
from hypothesis import event, given, note
from hypothesis import strategies as st

from superr.errors import LeakageError
from superr.folds import FOLD_TABLE, FitSpec, assign_folds, fold_of, sandbox_plan
from superr.folds import STAGE_NAME as FOLDS_STAGE
from superr.leakage import FitLeakageError, FitRecord, FitRegistry, early_stopping_split, fit_id_for
from superr.pipeline import SOURCE_CSVS, Stage, StageContext, checkpoint_layout, main
from superr.runlog import LOG_FOLDER
from superr.settings import CONFIG_FILENAME, load_settings
from superr.store import CheckpointStore
from superr.views import (
    M1_FIT,
    M2_FIT,
    NO_FIT,
    NOT_EXCLUDED,
    UNKNOWN_FIT,
    FoldValues,
    ViewLeakageError,
    build_view,
    check_view,
    view_plan,
)
from tests.strategies import FOLD_SEED, LEAK_EARLY_STOPPING, LEAK_TRAIN, N_FOLDS, ROOT, FoldTables, fold_tables

# ------------------------------------------------------------------------------------ constants

#: Exit code of a leakage-guard stop (design, "Run lifecycle": 4 leakage guard, Requirement 7.12).
EXIT_LEAKAGE = 4

_CONFIG = load_settings(ROOT)

# Early-stopping shares and seeds of the Config's LightGBM models, drawn next to arbitrary ones.
_ES_SHARES = tuple(sorted({float(_CONFIG.get(f"models.{model}.es_share")) for model in ("m2", "m3", "m4")}))
_ES_SEEDS = tuple(sorted({int(_CONFIG.get(f"seeds.{name}")) for name in ("m2", "m3_lgbm", "m4_lgbm")}))
_SHARES = st.one_of(st.sampled_from(_ES_SHARES), st.floats(min_value=0.01, max_value=0.99))
_SEEDS = st.one_of(st.sampled_from(_ES_SEEDS), st.integers(0, 2**32 - 1))
_FAMILIES = st.sampled_from((None, "lgbm"))

#: Model of the extra fit that test_guard... draws; no FitCase uses it, so its fit id is new.
_MIXED_MODEL = "Alternative_Model"

# The sandbox fits behind the views. The model names hold no digit and appear nowhere else in a
# check_view message (unlike "m1", which is part of the column name m1_fit), so a message that
# names the model can be told from one that only quotes the fit id.
_M1_MODEL = "Skill_Model"
_M2_MODEL, _M2_FAMILY = "Remaining_Runs_Model", "lgbm"
_M1_ALL_FOLDS = f"{_M1_MODEL}_all_folds"  # the all-fold M1 fit behind the top-theta lists: excludes no fold
_UNREGISTERED = {M1_FIT: f"{_M1_MODEL}_not_registered", M2_FIT: f"{_M2_MODEL}_{_M2_FAMILY}_not_registered"}
_M2_ES_SHARE = float(_CONFIG.get("models.m2.es_share"))
_M2_SEED = int(_CONFIG.get("seeds.m2"))

# The model, the folds after the word "excluded" and the count before "overlapping match(es)":
# what Requirement 7, criterion 12 asks every rejection to report.
_EXCLUDED_THEN_OVERLAP = re.compile(r"\bexcluded\b(?P<between>.*?)\boverlapping match")
_NUMBER = re.compile(r"\d+(?:,\d{3})*")


# -------------------------------------------------------------------------------------- helpers

def _table_arrays(fold_table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The Fold_Table's match ids and folds as int64 arrays, in table order."""
    return fold_table["match_id"].to_numpy(dtype=np.int64), fold_table["fold"].to_numpy(dtype=np.int64)


def _fold_by_match(fold_table: pd.DataFrame) -> dict[int, int]:
    ids, table_folds = _table_arrays(fold_table)
    return dict(zip(ids.tolist(), table_folds.tolist()))


def _reported(line: str) -> tuple[list[int], int]:
    """(excluded fold numbers, overlap count) as one message line states them."""
    found = _EXCLUDED_THEN_OVERLAP.search(line)
    assert found is not None, f"the message does not give the excluded folds and the overlapping matches: {line!r}"
    numbers = [int(text.replace(",", "")) for text in _NUMBER.findall(found.group("between"))]
    assert numbers, f"the message gives no count of overlapping matches: {line!r}"
    return numbers[:-1], numbers[-1]


def _assert_reports(line: str, fit_id: str, model: str, excluded: list[int], overlap: int) -> None:
    """The message line names the model (outside the fit id), the excluded folds and the overlap count."""
    rest = line.replace(repr(fit_id), "")
    assert model in rest, f"the message does not name the model {model!r}: {line!r}"
    folds, count = _reported(rest)
    assert folds == excluded, f"the message gives excluded folds {folds}, expected {excluded}: {line!r}"
    assert count == overlap, f"the message gives {count} overlapping matches, expected {overlap}: {line!r}"


# ------------------------------------------------------------------------- the guard (register)

class _GuardCase(NamedTuple):
    """One fit to register: its FitSpec, model and family, and the match ids given to the guard."""

    spec: FitSpec
    model: str
    family: str | None
    train_match_ids: np.ndarray  # one id per Training_Row, repeats allowed
    es_match_ids: np.ndarray
    predicted_match_ids: np.ndarray


@st.composite
def _mixed_fits(draw: Any, tables: FoldTables) -> _GuardCase:
    """A fit whose training and early-stopping matches may each hold matches of the excluded folds.

    Unlike the FitCases of fold_tables, which leak into one of the two sets only, both sets may
    leak at once and share matches (leaked ones too), and the early-stopping matches need not be
    training matches. The predicted matches are those of the spec's predicted fold, as in a Run.
    """
    ids, table_folds = _table_arrays(tables.fold_table)
    spec = draw(st.sampled_from(sandbox_plan(tables.n_folds)))
    inside = np.isin(table_folds, sorted(spec.excluded_folds))
    clean, banned = ids[~inside].tolist(), ids[inside].tolist()

    def some(pool: list[int], *, rows: bool) -> list[int]:
        if not pool:
            return []
        return draw(st.lists(st.sampled_from(pool), max_size=2 * len(pool) if rows else len(pool), unique=not rows))

    train = some(clean, rows=True) + (some(banned, rows=True) if draw(st.booleans()) else [])
    es = some(clean, rows=False) + (some(banned, rows=False) if draw(st.booleans()) else [])
    return _GuardCase(
        spec,
        _MIXED_MODEL,
        draw(_FAMILIES),
        np.asarray(train, dtype=np.int64),
        np.asarray(es, dtype=np.int64),
        ids[table_folds == spec.predicted_fold],
    )


def _register(registry: FitRegistry, case: _GuardCase, *, from_spec: bool) -> FitRecord:
    """Register the fit with ``register_spec`` or, giving every value itself, with ``register``."""
    if from_spec:
        return registry.register_spec(
            case.model,
            case.spec,
            case.train_match_ids,
            case.predicted_match_ids,
            early_stopping_match_ids=case.es_match_ids,
            family=case.family,
        )
    return registry.register(
        case.model,
        fit_id_for(case.model, case.spec, family=case.family),
        case.spec.excluded_folds,
        case.train_match_ids,
        case.predicted_match_ids,
        early_stopping_match_ids=case.es_match_ids,
        family=case.family,
        sandbox_k=case.spec.sandbox_k,
        inner_j=case.spec.inner_j,
    )


@given(tables=fold_tables(fits=True), data=st.data())
def test_guard_accepts_exactly_the_fits_without_excluded_fold_matches(tables: FoldTables, data: st.DataObject) -> None:
    """A fit is accepted exactly when no training or early-stopping match lies in an excluded fold;
    a rejection is a LeakageError (exit code 4) naming the model, the excluded folds and the
    overlap count, and records nothing (Requirement 7, criteria 6 and 12; Requirement 10, criterion 7).

    **Validates: Requirements 2.6, 7.6, 7.12, 10.7**
    """
    fold_by_match = _fold_by_match(tables.fold_table)
    cases = [
        _GuardCase(
            fit.spec,
            fit.model,
            data.draw(_FAMILIES, label=f"family of {fit.spec.fit_id}"),
            fit.train_match_ids,
            fit.es_match_ids,
            fit.predicted_match_ids,
        )
        for fit in tables.fits
    ]
    cases.append(data.draw(_mixed_fits(tables), label="fit that may leak into both sets"))
    for fit in tables.fits:
        event(f"FitCase leak: {fit.leak}")

    by_spec, by_args = FitRegistry(tables.fold_table), FitRegistry(tables.fold_table)
    accepted: list[str] = []
    for case in cases:
        spec = case.spec
        fit_id = fit_id_for(case.model, spec, family=case.family)
        es_matches = {int(match) for match in case.es_match_ids.tolist()}
        training = {int(match) for match in case.train_match_ids.tolist()} | es_matches
        leaked = sorted(match for match in training if fold_by_match[match] in spec.excluded_folds)
        overlap = len(leaked)
        note(
            f"fit {fit_id}: excluded {sorted(spec.excluded_folds)}, {len(training)} Training_Row matches "
            f"({len(es_matches)} early-stopping), leaked matches {leaked}"
        )
        for registry, from_spec in ((by_spec, True), (by_args, False)):
            if not overlap:
                record = _register(registry, case, from_spec=from_spec)
                assert record.fit_id == fit_id
                assert record.excluded_folds == spec.excluded_folds == registry.excluded_folds(fit_id)
                assert set(record.training_match_ids.tolist()) == training
                assert set(record.early_stopping_match_ids.tolist()) == es_matches
                continue
            with pytest.raises(LeakageError) as raised:
                _register(registry, case, from_spec=from_spec)
            error = raised.value
            assert isinstance(error, FitLeakageError), f"{type(error).__name__}: {error}"
            assert error.exit_code == EXIT_LEAKAGE
            excluded = sorted(spec.excluded_folds)
            assert (error.model, error.fit_id, list(error.excluded_folds), error.overlap_count) == (
                case.model,
                fit_id,
                excluded,
                overlap,
            )
            _assert_reports(str(error).splitlines()[0], fit_id, case.model, excluded, overlap)
            assert fit_id not in registry, "a rejected fit was recorded"
        if not overlap:
            accepted.append(fit_id)
        event("fit accepted" if not overlap else "fit rejected")
        for registry in (by_spec, by_args):
            assert [record.fit_id for record in registry] == accepted


# ---------------------------------------------------------------------- the early-stopping split

@given(tables=fold_tables(fits=True), share=_SHARES, seed=_SEEDS, data=st.data())
def test_early_stopping_split_holds_out_whole_training_matches(
    tables: FoldTables, share: float, seed: int, data: st.DataObject
) -> None:
    """``early_stopping_split`` holds out whole matches drawn only from the training matches, disjoint
    from the matches left to fit on, the same for the same seed and fit index; the guard judges the
    split fit exactly as its training matches (Requirement 10, criterion 7; Requirement 7, criterion 6).

    **Validates: Requirements 7.6, 10.7**
    """
    fold_by_match = _fold_by_match(tables.fold_table)
    plan = sandbox_plan(tables.n_folds)
    registry = FitRegistry(tables.fold_table)
    for case in tables.fits:
        rows = case.train_match_ids  # one id per Training_Row
        training = np.unique(rows.astype(np.int64))
        fit_index = plan.index(case.spec)
        if len(training) < 2:
            with pytest.raises(ValueError):
                early_stopping_split(rows, share, seed, fit_index)
            event("fewer than 2 training matches: no split")
            continue
        fit_matches, validation = early_stopping_split(rows, share, seed, fit_index)
        note(f"fit {case.spec.fit_id} (index {fit_index}): training {training.tolist()}, validation {validation.tolist()}")

        # Whole matches, drawn from the training matches only, disjoint from the matches left to fit on.
        for part in (fit_matches, validation):
            assert part.dtype == np.int64
            assert np.all(np.diff(part) > 0), "the split's match ids are not distinct and ascending"
        assert set(validation.tolist()) <= set(training.tolist()), "a validation match is not a training match"
        assert not set(validation.tolist()) & set(fit_matches.tolist()), "a match is both fitted on and held out"
        assert np.array_equal(np.union1d(fit_matches, validation), training)
        n = len(training)
        assert len(validation) == min(max(math.floor(share * n + 0.5), 1), n - 1), "the held-out share is wrong"

        # The same seed and fit index give the same split, whatever the row order or repeats.
        order = np.asarray(data.draw(st.permutations(range(len(rows))), label=f"row order of {case.spec.fit_id}"))
        for again in (rows, rows[order], training.tolist()):
            fit_again, validation_again = early_stopping_split(again, share, seed, fit_index)
            assert np.array_equal(fit_again, fit_matches) and np.array_equal(validation_again, validation)

        # The split hides no leak and makes none: the guard sees the same overlap as in the training matches.
        overlap = sum(fold_by_match[match] in case.spec.excluded_folds for match in training.tolist())
        assert (overlap == 0) == (case.leak != LEAK_TRAIN)
        if not overlap:
            record = registry.register_spec(
                case.model, case.spec, fit_matches, case.predicted_match_ids, early_stopping_match_ids=validation
            )
            assert np.array_equal(record.early_stopping_match_ids, validation)
            assert np.array_equal(record.train_match_ids, fit_matches)
            event("clean split accepted" if case.leak is None else f"split of a {LEAK_EARLY_STOPPING} case accepted")
        else:
            with pytest.raises(LeakageError) as raised:
                registry.register_spec(
                    case.model, case.spec, fit_matches, case.predicted_match_ids, early_stopping_match_ids=validation
                )
            assert raised.value.exit_code == EXIT_LEAKAGE
            assert raised.value.overlap_count == overlap
            event("leaky split rejected")


# --------------------------------------------------------------------------- the view check

class _Fit(NamedTuple):
    """A registered fit as the test records it: model, excluded folds and Training_Row matches."""

    model: str
    excluded: frozenset[int]
    training: frozenset[int]  # training and early-stopping matches


def _sandbox_registry(tables: FoldTables, *, with_m2: bool) -> tuple[FitRegistry, dict[str, _Fit]]:
    """A FitRegistry with every sandbox fit of M1 (``sandbox_plan``), the all-fold M1 fit and, with
    ``with_m2``, every sandbox fit of M2.

    Each fit trains on every match of its spec's train folds (the M2 fits hold some of them out for
    early stopping) and predicts the matches of its predicted fold, so the guard accepts it.
    """
    ids, table_folds = _table_arrays(tables.fold_table)
    registry = FitRegistry(tables.fold_table)
    fits: dict[str, _Fit] = {}
    for index, spec in enumerate(sandbox_plan(tables.n_folds)):
        train = ids[np.isin(table_folds, spec.train_folds)]
        predicted = ids[table_folds == spec.predicted_fold]
        training = frozenset(train.tolist())
        record = registry.register_spec(_M1_MODEL, spec, train, predicted)
        fits[record.fit_id] = _Fit(_M1_MODEL, spec.excluded_folds, training)
        if not with_m2:
            continue
        if len(train) >= 2:
            fit_part, held_out = early_stopping_split(train, _M2_ES_SHARE, _M2_SEED, index)
        else:
            fit_part, held_out = train, train[:0]
        record = registry.register_spec(
            _M2_MODEL, spec, fit_part, predicted, early_stopping_match_ids=held_out, family=_M2_FAMILY
        )
        fits[record.fit_id] = _Fit(_M2_MODEL, spec.excluded_folds, training)
    registry.register(_M1_MODEL, _M1_ALL_FOLDS, (), ids, ids[:0])
    fits[_M1_ALL_FOLDS] = _Fit(_M1_MODEL, frozenset(), frozenset(ids.tolist()))
    return registry, fits


def _fit_choices(n_folds: int, column: str) -> list[str | None]:
    """What a corrupted row may record in ``column``: a registered fit of the group, an unregistered one, or none."""
    plan = sandbox_plan(n_folds)
    if column == M1_FIT:
        registered = [fit_id_for(_M1_MODEL, spec) for spec in plan] + [_M1_ALL_FOLDS]
    else:
        registered = [fit_id_for(_M2_MODEL, spec, family=_M2_FAMILY) for spec in plan]
    return [*registered, _UNREGISTERED[column], None]


def _sources(n_folds: int, n_rows: int, with_m2: bool) -> dict[str, FoldValues]:
    """M1 quantities (and M2 ones when ``with_m2``) planned as the design plans them (``view_plan``)."""
    plan = sandbox_plan(n_folds)
    groups = [(M1_FIT, view_plan(_M1_MODEL, plan), ("theta_striker", "rbq"))]
    if with_m2:
        groups.append((M2_FIT, view_plan(_M2_MODEL, plan, family=_M2_FAMILY), ("par_score",)))
    sources: dict[str, FoldValues] = {}
    for column, cells, names in groups:
        fit_ids = sorted(set(cells.values()))
        for offset, name in enumerate(names):
            values = {fit_id: np.full(n_rows, 10.0 * index + offset) for index, fit_id in enumerate(fit_ids)}
            sources[name] = FoldValues(column, cells, values)
    return sources


class ViewCase(NamedTuple):
    """A fold_tables() draw, a view V_k over its Deliveries and the provenance written over some rows."""

    tables: FoldTables
    k: int
    order: tuple[int, ...]  # view row order: a permutation of the Deliveries
    with_m2: bool  # the view has m2_fit too (remaining_runs stage) or only m1_fit (skill stage)
    changes: dict[str, dict[int, str | None]]  # provenance column -> {view position: recorded fit, None for none}

    def __repr__(self) -> str:
        tables = self.tables
        return (
            f"ViewCase(n_folds={tables.n_folds}, seed={tables.seed}, k={self.k}, with_m2={self.with_m2}, "
            f"rows={len(tables.deliveries)}, order={list(self.order)}, changes={self.changes})\n"
            f"Fold_Table:\n{tables.fold_table.to_string()}\nDeliveries:\n{tables.deliveries.to_string()}"
        )


@st.composite
def view_cases(draw: Any) -> ViewCase:
    """A view case that overwrites the ``m1_fit`` of at least one row when the view has rows.

    The test checks the view as built before it overwrites anything, so every example also covers
    a view with no row overwritten.
    """
    tables = draw(fold_tables(fits=True))
    n_rows = len(tables.deliveries)
    k = draw(st.integers(1, tables.n_folds))
    order = tuple(draw(st.permutations(range(n_rows))))
    with_m2 = draw(st.booleans())
    changes = {}
    for column in (M1_FIT, M2_FIT) if with_m2 else (M1_FIT,):
        positions = st.integers(0, max(n_rows - 1, 0))
        least = 1 if column == M1_FIT and n_rows else 0
        changes[column] = draw(
            st.dictionaries(
                positions, st.sampled_from(_fit_choices(tables.n_folds, column)), min_size=least, max_size=n_rows
            )
        )
    return ViewCase(tables, k, order, with_m2, changes)


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _rejected(
    recorded: dict[str, list[Any]], fold_of_row: np.ndarray, k: int, fits: dict[str, _Fit]
) -> dict[tuple[str, str | None], list[int]]:
    """The rows whose recorded fit does not exclude both the row's fold and k, by (column, recorded fit).

    A fit that is not registered, or no fit at all, excludes nothing that the registry can show.
    """
    found: dict[tuple[str, str | None], list[int]] = {}
    for column, fit_ids in recorded.items():
        for position, value in enumerate(fit_ids):
            fit_id = None if _missing(value) else str(value)
            fit = None if fit_id is None else fits.get(fit_id)
            if fit is None or not {int(fold_of_row[position]), k} <= fit.excluded:
                found.setdefault((column, fit_id), []).append(position)
    return found


@given(case=view_cases())
def test_check_view_rejects_exactly_the_rows_whose_fit_does_not_exclude_their_fold_and_k(case: ViewCase) -> None:
    """``check_view`` accepts a correct view V_k and, after the recorded fits of drawn rows are
    overwritten, rejects exactly the rows whose ``m1_fit`` or ``m2_fit`` does not exclude both the
    row's fold and k (k alone for rows in fold k), with a LeakageError (exit code 4) that names the
    model, the excluded folds and the overlap count of each fit at fault (Requirement 7, criteria 6
    and 12; Requirement 2, criterion 6).

    **Validates: Requirements 2.6, 7.6, 7.12**
    """
    tables, k = case.tables, case.k
    fold_by_match = _fold_by_match(tables.fold_table)
    registry, fits = _sandbox_registry(tables, with_m2=case.with_m2)
    rows = tables.deliveries.iloc[list(case.order)].reset_index(drop=True)
    fold_of_row = fold_of(rows["match_id"], tables.fold_table)
    columns = [M1_FIT, M2_FIT] if case.with_m2 else [M1_FIT]

    # A correct view: every row records a fit that excludes the row's fold and k, and check_view accepts it.
    view = build_view(k, fold_of_row, _sources(tables.n_folds, len(rows), case.with_m2), rows=rows["row"])
    assert [column for column in (M1_FIT, M2_FIT) if column in view.columns] == columns
    recorded = {column: view[column].to_numpy(dtype=object).tolist() for column in columns}
    assert not _rejected(recorded, fold_of_row, k, fits), "build_view recorded a fit that does not exclude the row's fold and k"
    assert check_view(k, view, fold_of_row, registry) is None
    if not len(rows):
        event("a view with no row")

    # Overwrite the provenance of the drawn rows.
    for column, changes in case.changes.items():
        for position, fit_id in changes.items():
            recorded[column][position] = fit_id
    corrupted = view.copy()
    for column in columns:
        corrupted[column] = pd.array(recorded[column], dtype="str")
    expected = _rejected(recorded, fold_of_row, k, fits)
    note(f"rows to reject, by (column, recorded fit): {expected}")
    if not expected:
        event("every overwritten row still records an allowed fit" if any(case.changes.values()) else "no row overwritten")
        assert check_view(k, corrupted, fold_of_row, registry) is None
        return

    with pytest.raises(LeakageError) as raised:
        check_view(k, corrupted, fold_of_row, registry)
    error = raised.value
    assert isinstance(error, ViewLeakageError), f"{type(error).__name__}: {error}"
    assert error.exit_code == EXIT_LEAKAGE
    assert error.view_k == k
    bad = sorted({position for positions in expected.values() for position in positions})
    assert error.positions.tolist() == bad, "check_view did not reject exactly the rows whose fit is wrong"
    assert error.n_rows == len(bad)
    problems = {(problem.column, problem.fit_id): problem for problem in error.problems}
    assert sorted(problems, key=repr) == sorted(expected, key=repr)

    lines = str(error).splitlines()
    for (column, fit_id), positions in expected.items():
        problem = problems[(column, fit_id)]
        row_folds = sorted({int(fold_of_row[position]) for position in positions})
        assert problem.n_rows == len(positions)
        assert list(problem.row_folds) == row_folds
        fit = None if fit_id is None else fits.get(fit_id)
        if fit is None:
            assert problem.reason == (NO_FIT if fit_id is None else UNKNOWN_FIT)
            event(f"{column}: {problem.reason}")
            continue
        missing = ({k, *row_folds}) - fit.excluded
        overlap = sum(fold_by_match[match] in missing for match in fit.training)
        excluded = sorted(fit.excluded)
        assert problem.reason == NOT_EXCLUDED
        assert (list(problem.excluded_folds), list(problem.missing_folds), problem.overlap_count) == (
            excluded,
            sorted(missing),
            overlap,
        )
        assert fit.model in problem.model
        [line] = [line for line in lines if repr(fit_id) in line]
        _assert_reports(line, fit_id, fit.model, excluded, overlap)
        event(f"{column}: {NOT_EXCLUDED}")


# ------------------------------------------------------------------------- the Run stops

_RUN_MODEL = "Skill_Model"
_LATE_STAGES = ("pressure", "value", "rating")


def _temporary_workspace(tmp_path: Path, source_root: Path) -> Path:
    """A workspace under tmp_path: a copy of config.py and placeholder Source_CSVs and plan.md."""
    root = (tmp_path / "workspace").resolve()
    root.mkdir()
    shutil.copyfile(source_root / CONFIG_FILENAME, root / CONFIG_FILENAME)
    for name in SOURCE_CSVS.values():
        (root / name).write_text("placeholder\n", encoding="utf-8")
    (root / "plan.md").write_text("placeholder plan\n", encoding="utf-8")
    return root


def _late_stage(name: str, calls: list[str]) -> Callable[[StageContext], None]:
    """A stand-in for the pressure, value or rating stage: it records that it ran and writes its output."""

    def run(ctx: StageContext) -> None:
        calls.append(name)
        folder = ctx.settings.path("outputs") / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{name}.csv").write_text("row,value\n0,1.0\n", encoding="utf-8")
        ctx.store.save_table(name, name, pd.DataFrame({"row": [0], "value": [1.0]}))

    return run


@pytest.mark.parametrize("rejection", ["fit", "view"])
def test_a_leakage_error_stops_the_run_before_any_pressure_value_or_rating_output(
    rejection: str, tmp_path: Path, workspace_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A rejected fit (``register_spec``) or view (``check_view``) in a model stage ends the Run with
    exit code 4; no later stage runs or writes, the earlier checkpoints stay, and the Run_Log and
    standard error give the model name, the excluded folds and the overlap count (Requirement 7,
    criterion 12).

    **Validates: Requirements 7.12**
    """
    root = _temporary_workspace(tmp_path, workspace_root)
    types = ["IPL"] * 9 + ["T20I"] * 4 + ["A_Team"] * 2
    matches = pd.DataFrame(
        {"match_id": pd.array(range(1001, 1031, 2), dtype="Int64"), "comp_type": pd.array(types, dtype="str")}
    )
    fold_table = assign_folds(matches, matches["comp_type"], n_folds=N_FOLDS, seed=FOLD_SEED)
    ids, table_folds = _table_arrays(fold_table)
    plan = sandbox_plan(N_FOLDS)
    outer = plan[0]  # the outer fit of fold 1, M1(-1)
    assert (outer.sandbox_k, outer.inner_j) == (1, None)
    fit_id = fit_id_for(_RUN_MODEL, outer)
    calls: list[str] = []

    def folds_stage(ctx: StageContext) -> None:
        calls.append(FOLDS_STAGE)
        ctx.store.save_table(FOLDS_STAGE, FOLD_TABLE, fold_table)

    def skill_stage(ctx: StageContext) -> None:
        calls.append("skill")
        if rejection == "fit":
            # M1(-1) that forgot to drop fold 1 from its Training_Rows.
            ctx.registry.register_spec(_RUN_MODEL, outer, ids, ids[table_folds == 1])
        else:
            for spec in plan:
                if spec.sandbox_k == 1:  # M1(-1) and every M1(-{j, 1}), all valid
                    train = ids[np.isin(table_folds, spec.train_folds)]
                    ctx.registry.register_spec(_RUN_MODEL, spec, train, ids[table_folds == spec.predicted_fold])
            # A view V_1 that takes every row's values from M1(-1), forgetting the inner fits.
            cells = {(1, fold): fit_id for fold in range(1, N_FOLDS + 1)}
            fold_of_row = fold_of(ids, fold_table)
            sources = {"theta_striker": FoldValues(M1_FIT, cells, {fit_id: np.zeros(len(ids))})}
            view = build_view(1, fold_of_row, sources, rows=np.arange(len(ids)))
            check_view(1, view, fold_of_row, ctx.registry)
        calls.append("skill finished")

    previous = ("skill", *_LATE_STAGES[:-1])
    stages = (
        Stage(FOLDS_STAGE, folds_stage),
        Stage("skill", skill_stage, reads=(FOLDS_STAGE,)),
        *(Stage(name, _late_stage(name, calls), reads=(before,)) for name, before in zip(_LATE_STAGES, previous)),
    )

    code = main([], root=root, stages=stages)
    captured = capsys.readouterr()

    # M1(-1) trains on folds 2-5: a fit that kept fold 1 overlaps in fold 1's matches; a view row of
    # fold i != 1 needs a fit without fold i, and M1(-1) trains on every match of folds 2-5.
    overlap = int((table_folds == 1).sum()) if rejection == "fit" else int((table_folds != 1).sum())
    assert code == EXIT_LEAKAGE == LeakageError.exit_code
    assert calls == [FOLDS_STAGE, "skill"], "a stage after the rejection ran, or the rejected stage went on"

    outputs = root / "outputs"
    store = CheckpointStore(outputs, checkpoint_layout(root), config_sha256=hashlib.sha256(b"").hexdigest())
    for name in _LATE_STAGES:
        assert not (outputs / name).exists(), f"the {name} output folder was written"
        assert not store.stage_dir(name).exists(), f"the {name} checkpoint was written"
    assert store.is_complete(FOLDS_STAGE), "the earlier checkpoint was not kept"
    assert not store.is_complete("skill")
    written = [path for path in outputs.rglob("*") if path.is_file() and path.relative_to(outputs).parts[0] != LOG_FOLDER]
    assert all(path.is_relative_to(store.stage_dir(FOLDS_STAGE)) for path in written), sorted(map(str, written))

    [json_path] = sorted((outputs / LOG_FOLDER).glob("run_*.json"))
    record = json.loads(json_path.read_text(encoding="utf-8"))
    assert (record["exit_code"], record["failed_stage"]) == (EXIT_LEAKAGE, "skill")
    assert [(stage["name"], stage["status"]) for stage in record["stages"]] == [
        (FOLDS_STAGE, "completed"),
        ("skill", "failed"),
    ]
    [message] = [error["message"] for error in record["errors"] if error["stage"] == "skill"]
    [line] = [line for line in message.splitlines() if repr(fit_id) in line]
    _assert_reports(line, fit_id, _RUN_MODEL, [1], overlap)
    [log_path] = sorted((outputs / LOG_FOLDER).glob("run_*.log"))
    assert line.strip() in log_path.read_text(encoding="utf-8")
    assert line.strip() in captured.err
