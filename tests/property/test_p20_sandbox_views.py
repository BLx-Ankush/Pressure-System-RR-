# Feature: pressure-rating-system, Property 20: Sandbox views pick the right fit for every row
"""Property 20: Sandbox views pick the right fit for every row (Requirement 7, criteria 7, 8 and 11).

**Validates: Requirements 7.7, 7.8, 7.11**

Each example starts from ``tests.strategies.fold_tables()`` with a fold count n from 3-7, D15 most
often (``sandbox_plan`` needs 3 folds): matches with Competition_Types, their Fold_Table and 0-3
Deliveries per match, so a fold can hold no Delivery. The Deliveries are the view rows, in a drawn
order, so ``row`` ids differ from row positions and the folds interleave. The example also draws,
for each M1 and M2 fit, whether its values come in the compact layout (one value per row the fit
serves, as the model stages give them) or the full layout (one value per view row), and whether
each view V_k is built by one ``build_view`` call or as the ``skill`` and ``remaining_runs`` stages
build it: ``build_view`` with the M1 quantities, ``check_view``, then ``extend_view`` with the M2
quantities. The explicit example has D15 folds with two matches each and two Deliveries per match,
so every ordered pair (k, j) has rows in every test run.

The test registers every fit of ``sandbox_plan(n)`` with ``FitRegistry.register_spec`` against the
example's Fold_Table, for these models (MODELS):

* ``m1``, which supplies the M1 quantities ``theta_striker`` and ``rbq`` (provenance column
  ``m1_fit``), trains on every match of its train folds and predicts the matches of its predicted
  fold;
* ``m2_first`` (family ``lgbm``), which supplies the M2-derived quantities ``par_score`` and
  ``ach_rate`` (``m2_fit``) and reads the M1 quantities, and ``m3_inn1`` (``lgbm``), which reads
  both groups: outer and inner fits that train on the view rows of their train folds and predict
  the view rows of their predicted fold;
* ``m4`` (``lgbm``) and ``extras``: outer fits only, on view rows in the same way.

Each value of an M1 or M2 fit encodes the fit, the quantity and the row id, so the test can tell
which fit supplied every view value. One FoldValues per quantity holds the values of every fit, so
a value taken from a fit of another cell or another sandbox would be wrong rather than missing.

For every k the test builds V_k and checks that:

1. ``check_view`` accepts V_k, and the M1-only view before ``extend_view``;
2. for each of the n(n - 1) ordered pairs (k, j) (20 at D15 = 5) and for the fold-k rows, the
   ``m1_fit`` and ``m2_fit`` of every row name a fit in the FitRegistry whose Training_Rows, read
   from the registry's ``fit_train_matches`` table and the Fold_Table, hold no match of the row's
   fold or of fold k (criteria 7, 8 and 11). Every row of every pair whose fold j has rows is
   checked;
3. the recorded fit is the one the design's rule names, which is stronger than item 2: M1(-{j, k})
   or M2^(k)(-j) for a fold-j row, M1(-k) or M2(-k) for a fold-k row. Its id is the one
   ``view_plan`` names for (k, row fold); the registry holds it as that model's fit of sandbox k,
   with inner fold j (none for a fold-k row) and excluded folds {j, k} ({k}); and every quantity
   of the row holds that fit's value for that row. ``check_view`` alone would accept a fold-j row
   of V_k that took M2^(j)(-k): that fit excludes folds j and k, but it was trained on rows of V_j,
   whose M1 values came from fits of sandbox j that may have seen fold k;
4. every registered fit of sandbox k predicts only matches of the fold that its plan cell serves;
   the view rows that took values from an M1 or M2 fit lie in its predicted matches and none in
   its training matches; and every view row that a fit reading V_k trains on or predicts (M2, M3,
   M4, Extras_Model) carries in V_k M1 values, and except for M2 also M2 values, from fits whose
   Training_Rows hold no match of the row's fold or of fold k. These are the Training_Rows and
   predicted rows of the outer fits, the inner M2 fits and the inner M3 fits whose predictions
   feed the fold-k calibrator.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from hypothesis import event, example, given
from hypothesis import strategies as st

from superr.folds import MIN_SANDBOX_FOLDS, FitSpec, assign_folds, sandbox_plan
from superr.leakage import FitRecord, FitRegistry, fit_id_for
from superr.views import (
    FIT_COLUMNS,
    M1_FIT,
    M2_FIT,
    ROW_COLUMN,
    FoldValues,
    build_view,
    check_view,
    extend_view,
    view_plan,
)
from tests.strategies import FOLD_SEED, N_FOLDS, FoldTables, fold_tables

# ---------------------------------------------------------------------------------- the models


@dataclass(frozen=True)
class SandboxModel:
    """A sandboxed model of the test, named as ``register_spec`` and ``view_plan`` take it.

    ``supplies``: the provenance column of the view quantities the model supplies, or None;
    ``reads``: the provenance columns of V_k whose quantities its fits use as features (empty: its
    fits do not read the view); ``inner``: whether it makes inner fits as well as outer ones.
    """

    name: str
    family: str | None
    supplies: str | None
    reads: tuple[str, ...]
    inner: bool


M1 = SandboxModel("m1", None, supplies=M1_FIT, reads=(), inner=True)
M2 = SandboxModel("m2_first", "lgbm", supplies=M2_FIT, reads=(M1_FIT,), inner=True)
MODELS: tuple[SandboxModel, ...] = (
    M1,
    M2,
    SandboxModel("m3_inn1", "lgbm", supplies=None, reads=FIT_COLUMNS, inner=True),
    SandboxModel("m4", "lgbm", supplies=None, reads=FIT_COLUMNS, inner=False),
    SandboxModel("extras", None, supplies=None, reads=FIT_COLUMNS, inner=False),
)
MODEL_OF = {(model.name, model.family): model for model in MODELS}
SUPPLIERS: dict[str, SandboxModel] = {M1_FIT: M1, M2_FIT: M2}

#: The view quantities of each provenance column's group (design, Data Models, ``view_k``).
QUANTITIES: dict[str, tuple[str, ...]] = {M1_FIT: ("theta_striker", "rbq"), M2_FIT: ("par_score", "ach_rate")}
SLOT = {name: slot for slot, name in enumerate(name for column in FIT_COLUMNS for name in QUANTITIES[column])}

#: A fit's value of quantity slot s for view row r is (code * len(SLOT) + s) * ROW_BASE + r, where
#: code numbers the M1 and M2 fits.
ROW_BASE = 1_000_000

#: Value layouts of FoldValues (views module docstring): one value per served row, or per view row.
COMPACT, FULL = "compact", "full"

# ------------------------------------------------------------------------------------ examples

_FOLD_COUNTS = (
    st.one_of(st.just(N_FOLDS), st.integers(MIN_SANDBOX_FOLDS, max(7, N_FOLDS)))
    if N_FOLDS >= MIN_SANDBOX_FOLDS
    else st.integers(MIN_SANDBOX_FOLDS, 7)
)


class Case(NamedTuple):
    """A fold_tables() draw, the view row order, the value layout of each M1 and M2 fit and the build flow."""

    tables: FoldTables
    order: tuple[int, ...]  # view row p is the Delivery at position order[p] of tables.deliveries
    layouts: Mapping[str, str]  # M1 or M2 fit id -> COMPACT or FULL
    two_steps: bool  # build_view with the M1 quantities, then extend_view with the M2 ones

    def __repr__(self) -> str:
        tables, n = self.tables, self.tables.n_folds
        table = tables.fold_table
        fold_of_match = dict(zip(table["match_id"].tolist(), table["fold"].tolist()))
        ball_matches = tables.deliveries["match_id"].tolist()
        per_match = Counter(ball_matches)
        matches = Counter(fold_of_match.values())
        rows = Counter(fold_of_match[match] for match in ball_matches)
        full = sorted(fit for fit, layout in self.layouts.items() if layout == FULL)
        flow = "build_view (M1), check_view, extend_view (M2)" if self.two_steps else "one build_view call"
        shown = table.assign(deliveries=[per_match[match] for match in table["match_id"].tolist()])
        return (
            f"n_folds {n}, seed {tables.seed}; matches per fold 1-{n}: {[matches[f] for f in range(1, n + 1)]}; "
            f"view rows per fold: {[rows[f] for f in range(1, n + 1)]}\n"
            f"Fold_Table with Deliveries per match:\n{shown.to_string()}\n"
            f"view row order (Delivery row ids): {[int(tables.deliveries['row'].iloc[p]) for p in self.order]}\n"
            f"views built by {flow}; fits with full-layout values: {full or 'none'}"
        )


def _supplier_fits(n_folds: int) -> list[str]:
    """The ids of the M1 and M2 fits of sandbox_plan(n_folds), in plan order."""
    plan = sandbox_plan(n_folds)
    return [fit_id_for(model.name, spec, family=model.family) for model in SUPPLIERS.values() for spec in plan]


@st.composite
def cases(draw: Any) -> Case:
    """A fold_tables() draw with 3 or more folds, a view row order, value layouts and a build flow."""
    tables = draw(fold_tables(n_folds=_FOLD_COUNTS))
    order = draw(st.permutations(range(len(tables.deliveries))))
    fits = _supplier_fits(tables.n_folds)
    layouts = draw(st.lists(st.sampled_from((COMPACT, FULL)), min_size=len(fits), max_size=len(fits)))
    return Case(tables, tuple(order), dict(zip(fits, layouts)), draw(st.booleans()))


def _all_pairs_case() -> Case:
    """D15 folds, two IPL matches in each and two Deliveries per match: every ordered pair (k, j) has rows."""
    n = N_FOLDS
    ids = 101 + 10 * np.arange(2 * n, dtype=np.int64)
    matches = pd.DataFrame(
        {"match_id": pd.array(ids, dtype="Int64"), "comp_type": pd.array(["IPL"] * len(ids), dtype="str")}
    )
    fold_table = assign_folds(matches, matches["comp_type"], n_folds=n, seed=FOLD_SEED)
    ball_matches = np.repeat(ids, 2).astype(np.int32)
    deliveries = pd.DataFrame({"row": np.arange(len(ball_matches), dtype=np.int32), "match_id": ball_matches})
    layouts = {fit: (COMPACT, FULL)[position % 2] for position, fit in enumerate(_supplier_fits(n))}
    tables = FoldTables(matches, n, FOLD_SEED, fold_table, deliveries, ())
    return Case(tables, tuple(range(len(ball_matches) - 1, -1, -1)), layouts, True)


ALL_PAIRS = _all_pairs_case()

# ------------------------------------------------------------------------------------- helpers


def _encode(code: int, name: str, rows: np.ndarray) -> np.ndarray:
    """The values of quantity ``name`` that the fit numbered ``code`` gives the view rows ``rows``."""
    return ((code * len(SLOT) + SLOT[name]) * ROW_BASE + rows).astype(np.float64)


def _describe(value: float, fit_of_code: Mapping[int, str]) -> str:
    """Which fit, quantity and row a view value came from, for assertion messages."""
    if not np.isfinite(value) or value < 0 or value != np.floor(value):
        return f"{value!r}, which no fit gave"
    rest, row = divmod(int(value), ROW_BASE)
    code, slot = divmod(rest, len(SLOT))
    names = {s: name for name, s in SLOT.items()}
    return f"the {names.get(slot, '?')} value of fit {fit_of_code.get(code, '?')!r} for row {row}"


def _register(
    registry: FitRegistry,
    plan: list[FitSpec],
    row_matches: np.ndarray,
    row_folds: np.ndarray,
    matches_in: Mapping[int, np.ndarray],
) -> None:
    """Register every fit of every model in MODELS: outer fits, and inner fits for models that make them."""
    for model in MODELS:
        for spec in plan:
            if spec.inner_j is not None and not model.inner:
                continue
            if model.reads:  # a fit of sandbox k trains and predicts on rows of V_k
                train = row_matches[np.isin(row_folds, spec.train_folds)]
                predicted = row_matches[row_folds == spec.predicted_fold]
            else:  # M1 trains on the Legal_Balls of every match of its train folds
                train = np.concatenate([matches_in[fold] for fold in spec.train_folds])
                predicted = matches_in[spec.predicted_fold]
            registry.register_spec(model.name, spec, train, predicted, family=model.family)


def _sources(
    case: Case, plan: list[FitSpec], row_ids: np.ndarray, row_folds: np.ndarray
) -> tuple[dict[str, dict[str, FoldValues]], dict[str, int]]:
    """FoldValues of every M1 and M2 quantity, by provenance column, and the code of each M1 and M2 fit."""
    codes: dict[str, int] = {}
    sources: dict[str, dict[str, FoldValues]] = {}
    for column, model in SUPPLIERS.items():
        values: dict[str, dict[str, np.ndarray]] = {name: {} for name in QUANTITIES[column]}
        for spec in plan:
            fit_id = fit_id_for(model.name, spec, family=model.family)
            code = codes.setdefault(fit_id, len(codes))
            # Compact: the rows the fit serves (its predicted fold) in view row order; full: every view row.
            rows = row_ids[row_folds == spec.predicted_fold] if case.layouts[fit_id] == COMPACT else row_ids
            for name in QUANTITIES[column]:
                values[name][fit_id] = _encode(code, name, rows)
        cells = view_plan(model.name, plan, family=model.family)
        sources[column] = {name: FoldValues(column, cells, values[name]) for name in QUANTITIES[column]}
    return sources, codes


def _views(
    case: Case,
    n_folds: int,
    row_ids: np.ndarray,
    row_folds: np.ndarray,
    sources: Mapping[str, Mapping[str, FoldValues]],
    registry: FitRegistry,
) -> dict[int, pd.DataFrame]:
    """V_k for every k, each accepted by check_view (item 1), as is the M1-only view of the two-step flow."""
    m1, m2 = sources[M1_FIT], sources[M2_FIT]
    views: dict[int, pd.DataFrame] = {}
    for k in range(1, n_folds + 1):
        if case.two_steps:  # as the skill stage builds V_k and the remaining_runs stage extends it
            m1_view = build_view(k, row_folds, m1, rows=row_ids)
            assert check_view(k, m1_view, row_folds, registry) is None
            view = extend_view(k, m1_view, row_folds, m2)
        else:  # one call, with the quantities of the two groups interleaved
            mixed = {name: source for pair in zip(m1.items(), m2.items()) for name, source in pair}
            view = build_view(k, row_folds, mixed, rows=row_ids)
        assert check_view(k, view, row_folds, registry) is None
        views[k] = view
    return views


def _matches_by_fit(frame: pd.DataFrame) -> dict[str, frozenset[int]]:
    """fit id -> its match ids, from a FitRegistry side table (``fit_train_matches`` or ``fit_predicted_matches``)."""
    found: dict[str, set[int]] = {}
    for fit_id, match in zip(frame["fit_id"].tolist(), frame["match_id"].tolist()):
        found.setdefault(fit_id, set()).add(int(match))
    return {fit_id: frozenset(matches) for fit_id, matches in found.items()}


def _check_provenance(
    k: int,
    fold: int,
    fits: np.ndarray,
    column: str,
    records: Mapping[str, FitRecord],
    training_folds: Mapping[str, frozenset[int]],
    rows: str,
) -> int:
    """Item 2: each fit in ``fits`` (recorded in ``column`` of V_k for ``rows``, all in ``fold``) is
    registered and its Training_Rows hold no match of ``fold`` or k. Returns the number of rows checked."""
    needs = f"fold {k}" if fold == k else f"fold {fold} or fold {k}"
    for fit_id, count in Counter(fits.tolist()).items():
        assert fit_id in records, f"V_{k}, {rows}: {count} record {column} {fit_id!r}, which is not in the FitRegistry"
        seen = sorted(training_folds[fit_id] & {fold, k})
        assert not seen, (
            f"V_{k}, {rows}: {count} took {column} values from fit {fit_id!r}, whose Training_Rows hold matches of "
            f"fold(s) {seen}; a fold-{fold} row of V_{k} needs a fit whose Training_Rows hold no match of {needs}"
        )
    return len(fits)


def _check_pick(
    k: int,
    fold: int,
    at: np.ndarray,
    fits: np.ndarray,
    column: str,
    values: Mapping[str, np.ndarray],
    plans: Mapping[str, Mapping[tuple[int, int], str]],
    records: Mapping[str, FitRecord],
    codes: Mapping[str, int],
    row_ids: np.ndarray,
) -> None:
    """Item 3: the fold-``fold`` rows of V_k (positions ``at``) took ``column``'s quantities from the
    fit that the design's rule names, and hold that fit's values for those rows."""
    model = SUPPLIERS[column]
    expected = plans[column][(k, fold)]
    picked = sorted(set(fits.tolist()), key=str)
    assert picked == [expected], (
        f"V_{k}: the fold-{fold} rows record {column} {picked}; view_plan names {expected!r} for (k, fold) = ({k}, {fold})"
    )
    record = records[expected]
    facts = (record.model, record.family, record.sandbox_k, record.inner_j, record.excluded_folds)
    rule = (model.name, model.family, k, None if fold == k else fold, frozenset({fold, k}))
    assert facts == rule, (
        f"V_{k}: fit {expected!r} of the fold-{fold} rows is registered with (model, family, sandbox_k, inner_j, "
        f"excluded_folds) = {facts}; the design's rule needs {rule}"
    )
    fit_of_code = {code: fit_id for fit_id, code in codes.items()}
    for name in QUANTITIES[column]:
        got = values[name][at]
        wrong = np.flatnonzero(got != _encode(codes[expected], name, row_ids[at]))
        assert not len(wrong), (
            f"V_{k}, {name}: {len(wrong)} of the {len(at)} fold-{fold} rows do not hold the value that fit "
            f"{expected!r} gave them; row {row_ids[at[wrong[0]]]} holds {_describe(got[wrong[0]], fit_of_code)}"
        )


# ---------------------------------------------------------------------------------------- test


@given(case=cases())
@example(case=ALL_PAIRS)
def test_sandbox_views_pick_the_right_fit_for_every_row(case: Case) -> None:
    """Every M1 and M2 value of V_k comes from the planned fit, which saw neither the row's fold nor k,
    for all ordered pairs (k, j) and for the rows every fit of sandbox k trains on or predicts
    (Requirement 7, criteria 7, 8 and 11).

    **Validates: Requirements 7.7, 7.8, 7.11**
    """
    tables, n = case.tables, case.tables.n_folds
    folds = range(1, n + 1)
    plan = sandbox_plan(n)
    table = tables.fold_table
    fold_by_match = {int(match): int(fold) for match, fold in zip(table["match_id"].tolist(), table["fold"].tolist())}
    view_rows = tables.deliveries.iloc[list(case.order)]
    row_ids = view_rows["row"].to_numpy(dtype=np.int64)
    row_matches = view_rows["match_id"].to_numpy(dtype=np.int64)
    row_folds = np.array([fold_by_match[match] for match in row_matches.tolist()], dtype=np.int64)
    matches_in = {
        fold: np.array(sorted(m for m, f in fold_by_match.items() if f == fold), dtype=np.int64) for fold in folds
    }
    assert row_ids.max(initial=0) < ROW_BASE, "the value encoding needs row ids below ROW_BASE"

    registry = FitRegistry(table)
    _register(registry, plan, row_matches, row_folds, matches_in)
    sources, codes = _sources(case, plan, row_ids, row_folds)
    views = _views(case, n, row_ids, row_folds, sources, registry)  # item 1: check_view accepts every view

    # The registry: each fit's record, and its matches from the side tables that FitRegistry.save writes.
    records = {record.fit_id: record for record in registry.records()}
    assert len(records) == 3 * n * n + 2 * n, "the test registers M1, M2 and M3 outer and inner fits, M4 and Extras outer fits"
    training = _matches_by_fit(registry.train_matches_frame())
    predicted = _matches_by_fit(registry.predicted_matches_frame())
    training_folds = {fit_id: frozenset(fold_by_match[m] for m in training.get(fit_id, ())) for fit_id in records}
    plans = {column: view_plan(model.name, plan, family=model.family) for column, model in SUPPLIERS.items()}
    recorded: dict[int, dict[str, np.ndarray]] = {}
    values: dict[int, dict[str, np.ndarray]] = {}
    for k in folds:
        view = views[k]
        assert set(view.columns) == {ROW_COLUMN, *SLOT, *FIT_COLUMNS}, f"V_{k} has the columns {list(view.columns)}"
        assert view[ROW_COLUMN].tolist() == row_ids.tolist(), f"V_{k} does not hold the view rows in view row order"
        recorded[k] = {column: np.asarray(view[column].tolist(), dtype=object) for column in FIT_COLUMNS}
        values[k] = {name: view[name].to_numpy(dtype=np.float64) for name in SLOT}

    # Items 2 and 3, for each ordered pair (k, j) with j != k, then for the fold-k rows of each view.
    n_rows_in = {fold: int((row_folds == fold).sum()) for fold in folds}
    pairs = [(k, j) for k in folds for j in folds if j != k]
    assert len(pairs) == n * (n - 1)
    checked: Counter[tuple[int, int]] = Counter()
    for k, j in [*pairs, *((k, k) for k in folds)]:
        at = np.flatnonzero(row_folds == j)
        if not len(at):
            continue
        for column in FIT_COLUMNS:
            fits = recorded[k][column][at]
            rows = f"the {len(at)} row(s) of fold {j}"
            checked[(k, j)] += _check_provenance(k, j, fits, column, records, training_folds, rows)
            _check_pick(k, j, at, fits, column, values[k], plans, records, codes, row_ids)
    for k, j in pairs:
        assert checked[(k, j)] == len(FIT_COLUMNS) * n_rows_in[j], (
            f"pair (k, j) = ({k}, {j}): {checked[(k, j)]} row provenances checked; fold {j} has {n_rows_in[j]} row(s)"
        )

    # Item 4: the predicted matches and the rows of every registered fit of sandbox k.
    served: dict[str, set[int]] = {}
    for model in MODELS:
        specs = [spec for spec in plan if model.inner or spec.inner_j is None]
        for (_, fold), fit_id in view_plan(model.name, specs, family=model.family).items():
            served.setdefault(fit_id, set()).add(fold)
    for fit_id, record in records.items():
        k, model = record.sandbox_k, MODEL_OF[(record.model, record.family)]
        trained_on, predicts = training.get(fit_id, frozenset()), predicted.get(fit_id, frozenset())
        outside = sorted({fold_by_match[match] for match in predicts} - served[fit_id])
        assert not outside, (
            f"fit {fit_id!r} of sandbox {k} predicts matches of fold(s) {outside}; it serves {sorted(served[fit_id])}"
        )
        if model.supplies is not None:
            used = set(row_matches[recorded[k][model.supplies] == fit_id].tolist())
            assert used <= predicts, (
                f"V_{k}: rows of matches {sorted(used - predicts)} took {model.supplies} values from fit {fit_id!r}, "
                "which does not list those matches as predicted"
            )
            assert not used & trained_on, (
                f"V_{k}: rows of matches {sorted(used & trained_on)} took {model.supplies} values from fit {fit_id!r}, "
                "whose Training_Rows hold those matches"
            )
        for role, matches in (("trains on", trained_on), ("predicts", predicts)):
            rows_of_fit = np.flatnonzero(np.isin(row_matches, np.array(sorted(matches), dtype=np.int64)))
            for column in model.reads:
                for fold in np.unique(row_folds[rows_of_fit]).tolist():
                    at = rows_of_fit[row_folds[rows_of_fit] == fold]
                    rows = f"the {len(at)} fold-{fold} row(s) that fit {fit_id!r} {role}"
                    _check_provenance(k, fold, recorded[k][column][at], column, records, training_folds, rows)

    event(f"n_folds = {n}")
    event("every fold has view rows" if all(n_rows_in.values()) else "a fold without view rows")
    event("views built by build_view, then extend_view" if case.two_steps else "views built by one build_view call")
    if len(set(case.layouts.values())) == 2:
        event("compact and full value layouts mixed")
