"""Sandbox views: the model-derived features that the fits of each fold sandbox read, the fit
that supplied each row's values, and the check of those fits against the FitRegistry
(Requirement 7, criteria 7, 8, 11 and 12; Requirement 2, criterion 6).

Design sections "Fold sandboxes and leakage control", "Fold_Assigner, views and leakage guard"
and Data Models, ``view_k``.

The views V_k
    For each target fold k, view V_k holds the model-derived features of the Deliveries that the
    fits of sandbox k train on or predict: in a Run, one row per innings 1-2 Delivery of the
    ``state`` table, in every fold. A row in fold i takes

    * its M1 values (theta_bat, theta_bowl, Scoring_Environment, Venue_Effect, RBQ,
      Team_Strength_Gap, Match_Day_Conditions and the values built from them) from M1(-{i, k}),
      an M1 fit that excludes folds i and k, when i != k, and from M1(-k), the outer M1 fit of
      fold k, when i = k (criterion 7);
    * its M2-derived values (Par_Score, Par_Rate, Achievable_Rate, the F19 Pillar_Component)
      from M2^(k)(-i), the inner M2 fit of sandbox k that excludes folds i and k, when i != k,
      and from M2(-k), the outer M2 fit of sandbox k, when i = k (criterion 8).

    Every fit of sandbox k (the outer fits of M2, M3, M4 and the Extras_Model, and the inner M2
    and M3 fits) trains and predicts on V_k only, so no model in the chain for fold k saw fold k
    (Requirement 2, criterion 6). Each row records the ids of the fits that supplied it in the
    provenance columns ``m1_fit`` and ``m2_fit`` (criterion 11), which ``check_view`` and the
    Test_Suite read.

    In the design (Data Models, ``view_k``), the ``skill`` stage builds V_k with ``row``, the M1
    quantities ``theta_striker``, ``theta_non_striker``, ``theta_bowler``,
    ``theta_next_batter``, ``scoring_env``, ``venue_effect``, ``rbq``, ``rbq_post_wicket``,
    ``tsg``, ``match_day`` and ``m1_pred_runs``, and ``m1_fit``. The ``remaining_runs`` stage
    extends it with the M2-derived quantities ``par_score``, ``par_score_next``,
    ``par_rate_win``, ``ach_rate`` and ``f19_component`` (all from the first-innings M2 fits),
    and ``m2_fit``. This module names no quantity: any column name serves.

FoldValues and the view plan
    ``FoldValues(fit_column, plan, values)`` holds one quantity, such as the striker's
    theta_bat, from every fit that produced it:

    * ``fit_column``: the provenance column of the quantity's group, M1_FIT for an M1 value and
      M2_FIT for an M2-derived value;
    * ``plan``: which fit serves which rows, as a mapping from (view k, row fold i) to a fit id.
      ``view_plan(model, folds.sandbox_plan(D15), family=...)`` gives the design's plan: (k, k)
      to the outer fit ``<model>[_<family>]_outer_k<k>`` and (k, i) to the inner fit
      ``<model>[_<family>]_inner_k<k>_j<i>``, the ids that ``FitRegistry.register_spec`` gives
      the same FitSpecs;
    * ``values``: fit id -> a 1-D array of numbers or booleans, with NaN for a missing value.
      An array holds either one value per view row, in view row order (full layout), or one
      value per row that the fit serves, in view row order (compact layout). The rows a fit
      serves are the rows of every fold i for which the plan names it, in any view: for a fit of
      the design's plan, the rows of its ``predicted_fold``. The two layouts coincide when a fit
      serves every row. The values of fits that the plan does not name are not read.

    The arrays are not copied; the FoldValues sees them read-only.

Building and extending a view
    ``build_view(k, fold_of_row, sources, rows=...)`` returns V_k as a DataFrame on a
    RangeIndex. ``fold_of_row`` gives the fold of each view row, from the Fold_Table
    (``folds.fold_of``), and ``sources`` maps each quantity column to its FoldValues. Within each
    group (provenance column), a row in fold i takes every quantity of the group from the one fit
    that the plan names for (k, i), and that fit id goes into the group's provenance column, so
    the quantities of a group must agree on the fit for every fold. The columns are ``row``
    (when ``rows`` is given), the quantities in the order of ``sources``, then the provenance
    columns in FIT_COLUMNS order, of dtype ``str``.

    ``extend_view(k, view, fold_of_row, sources)`` adds the quantities of another group, such as
    the M2 part, to a view. It returns a new DataFrame with the view's columns, the new
    quantities, and the provenance columns last. A new quantity may not repeat a column of the
    view. A group whose provenance column the view already has may add quantities only when its
    plan picks the fit that the column records, on every row.

    Neither function reads the FitRegistry; ``check_view`` does.

Checking a view (``check_view``)
    ``check_view(k, view, fold_of_row, registry)`` looks up the fit id of every row in each
    provenance column of ``view`` (``m1_fit``, and ``m2_fit`` once the view has it). It accepts
    the view only when every row's fit is in the FitRegistry and that fit's excluded folds
    contain both the row's fold and k (k alone for a row in fold k). Otherwise it raises
    ViewLeakageError, a ``superr.errors.LeakageError`` (exit code 4). Rows are rejected when
    their fit does not exclude those folds (NOT_EXCLUDED), when their fit id is not in the
    registry (UNKNOWN_FIT), when they record no fit (NO_FIT), and all rows when the view has no
    provenance column (NO_FIT_COLUMN), since none of these can be traced to a fit that excludes
    the row's fold and k.

    The message (criterion 12) names the view and then, for each provenance column and fit, the
    model name, the fit id, the fit's excluded folds, the number of overlapping matches (the
    matches of the fit's Training_Rows that lie in folds the rows needed it to exclude) and the
    rejected rows by fold, with examples. The error's attributes hold the same facts:
    ``view_k``, ``positions`` (the rejected row positions, ascending), ``n_rows`` and
    ``problems``, one ViewProblem per provenance column and fit.

    A model stage calls check_view on V_k before the first fit of sandbox k reads it, and again
    after it extends the view. Stages 06-10 run before ``pressure``, so a rejected view stops the
    Run before any pressure, value or rating output is written.

Use in a Run
    ``fold_of_row`` must follow the view's row order. For a view read from a checkpoint, take it
    from the view's ``row`` column, for example
    ``folds.attach_folds(view[["row"]], fold_table, balls=balls)["fold"].to_numpy()``.

    In the two sketches below, ``M1_QUANTITIES`` and ``M2_QUANTITIES`` stand for the stage's own
    lists of quantity columns, and ``quantities(...)`` for its code that evaluates one fit on the
    rows it serves. The ``skill`` stage registers, fits and evaluates the 25 M1 fits of the plan,
    then builds and checks the 5 views::

        plan = folds.sandbox_plan(settings.get("design_choices.D15_n_folds"))
        m1_plan = views.view_plan("m1", plan)         # (k, k): m1_outer_k<k>; (k, i): m1_inner_k<k>_j<i>
        values = {name: {} for name in M1_QUANTITIES}  # quantity -> {fit id: array}
        for spec in plan:
            record = ctx.registry.register_spec("m1", spec, train_matches, predicted_matches)
            ...                                        # only now fit M1 on spec.train_folds
            served = fold_of_row == spec.predicted_fold
            for name, array in quantities(fit, served).items():  # one value per served row
                values[name][record.fit_id] = array
        sources = {name: views.FoldValues(views.M1_FIT, m1_plan, values[name]) for name in M1_QUANTITIES}
        for k in range(1, n_folds + 1):
            view = views.build_view(k, fold_of_row, sources, rows=state["row"])
            views.check_view(k, view, fold_of_row, ctx.registry)
            ctx.store.save_table(STAGE_NAME, views.view_table(k), view)

    The ``remaining_runs`` stage, for each sandbox k::

        view = ctx.store.load_table("skill", views.view_table(k))
        views.check_view(k, view, fold_of_row, ctx.registry)  # before any fit of sandbox k reads V_k
        values = {name: {} for name in M2_QUANTITIES}
        for spec in (spec for spec in plan if spec.sandbox_k == k):  # M2(-k) and every M2^(k)(-j)
            record = ctx.registry.register_spec("m2_first", spec, ..., family="lgbm")
            ...                                        # fit on the rows of V_k in spec.train_folds
            served = fold_of_row == spec.predicted_fold
            for name, array in quantities(model, view[served]).items():
                values[name][record.fit_id] = array
        m2_plan = views.view_plan("m2_first", plan, family="lgbm")
        sources = {name: views.FoldValues(views.M2_FIT, m2_plan, values[name]) for name in M2_QUANTITIES}
        view = views.extend_view(k, view, fold_of_row, sources)
        views.check_view(k, view, fold_of_row, ctx.registry)
        ctx.store.save_table(STAGE_NAME, views.view_table(k), view)

Every step is vectorised over rows: the loops run over folds, quantities and distinct fit ids
only, so building or checking a view of 200k rows takes a fraction of a second.

Errors: ViewLeakageError (exit code 4) from check_view; ValueError or TypeError, naming the
problem, for malformed arguments, such as a fold with no planned fit, a missing value array or
one of the wrong length, quantities of one group that disagree on a fit, or a quantity column
that repeats another column.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from numbers import Integral
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from superr.errors import LeakageError
from superr.leakage import fit_id_for

if TYPE_CHECKING:
    from superr.folds import FitSpec
    from superr.leakage import FitRecord, FitRegistry

__all__ = [
    "FIT_COLUMNS",
    "M1_FIT",
    "M2_FIT",
    "NOT_EXCLUDED",
    "NO_FIT",
    "NO_FIT_COLUMN",
    "REASONS",
    "ROW_COLUMN",
    "UNKNOWN_FIT",
    "VIEW_TABLE_PREFIX",
    "FoldValues",
    "ViewLeakageError",
    "ViewProblem",
    "build_view",
    "check_view",
    "extend_view",
    "view_plan",
    "view_table",
]

# ------------------------------------------------------------------------------------- constants

#: The provenance columns: the fit that supplied a row's M1 values and its M2-derived values.
M1_FIT = "m1_fit"
M2_FIT = "m2_fit"
FIT_COLUMNS: tuple[str, ...] = (M1_FIT, M2_FIT)

#: The column of the Delivery's ``row`` id (the ``row`` column of the ``state`` table).
ROW_COLUMN = "row"

#: Checkpoint tables of the views are named ``view_<k>`` (view_table).
VIEW_TABLE_PREFIX = "view_"

#: Why check_view rejected rows (ViewProblem.reason).
NOT_EXCLUDED = "not_excluded"  # the row's fit does not exclude the row's fold and k
UNKNOWN_FIT = "unknown_fit"  # the row's fit id is not in the FitRegistry
NO_FIT = "no_fit"  # the row records no fit
NO_FIT_COLUMN = "no_fit_column"  # the view has no provenance column
REASONS: tuple[str, ...] = (NOT_EXCLUDED, UNKNOWN_FIT, NO_FIT, NO_FIT_COLUMN)

_VALUE_KINDS = "biuf"  # numpy kinds a quantity may have: bool, int, uint, float
_SHOWN = 5  # example rows named per problem
_FOOTER = "No fit may read the view. The Run stops before any pressure, value or rating output is written."


# ---------------------------------------------------------------------------------- small helpers

def _plural(count: int, singular: str, plural: str | None = None) -> str:
    return f"{count:,} {singular if count == 1 else (plural or singular + 's')}"


def _folds_text(folds: Iterable[int]) -> str:
    values = sorted({int(fold) for fold in folds})
    if not values:
        return "no fold"
    if len(values) == 1:
        return f"fold {values[0]}"
    return "folds " + ", ".join(map(str, values[:-1])) + f" and {values[-1]}"


def _fold_number(value: Any, what: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{what} must be a fold number (a whole number); got {value!r}")
    if value < 1:
        raise ValueError(f"{what}: folds are numbered from 1; got {value}")
    return int(value)


def _check_fit_id(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or "\n" in value:
        raise ValueError(f"{what} must be a fit id, a non-empty str without surrounding spaces; got {value!r}")
    return str.__str__(value)


def _integer_array(values: Any, what: str, hint: str) -> np.ndarray:
    """``values`` (a 1-D array-like of whole numbers, none missing) as a numpy integer array."""
    if values is None or isinstance(values, (str, bytes, Mapping, pd.DataFrame)) or np.ndim(values) != 1:
        raise TypeError(f"{what} must be a 1-D array {hint}; got {type(values).__name__}")
    if isinstance(values, (pd.Series, pd.Index, pd.api.extensions.ExtensionArray)):
        dtype = values.dtype
        if pd.api.types.is_bool_dtype(dtype) or not pd.api.types.is_integer_dtype(dtype):
            raise TypeError(f"{what} must hold whole numbers; got dtype {dtype}")
        missing = int(np.asarray(pd.isna(values)).sum())
        if missing:
            raise ValueError(f"{what} has {_plural(missing, 'missing value')}")
        return np.asarray(values.to_numpy(dtype=getattr(dtype, "numpy_dtype", dtype)))
    array = np.asarray(values)
    if array.size == 0:  # an empty list or array, whatever its dtype, holds no row
        return np.zeros(0, dtype=array.dtype if array.dtype.kind in "iu" else np.int64)
    if array.dtype.kind not in "iu":
        raise TypeError(f"{what} must hold whole numbers (an integer array without missing values); got dtype {array.dtype}")
    return array


def _row_folds(fold_of_row: Any) -> np.ndarray:
    """The fold of each view row as int64; TypeError or ValueError when malformed."""
    folds = _integer_array(fold_of_row, "fold_of_row", "with the fold of each view row (folds.fold_of gives one)")
    folds = folds.astype(np.int64, copy=False)
    low = int((folds < 1).sum())
    if low:
        raise ValueError(f"fold_of_row holds a fold below 1 in {_plural(low, 'row')}; folds are numbered from 1")
    return folds


def _row_ids(rows: Any, n_rows: int) -> np.ndarray:
    ids = _integer_array(rows, "rows", "with the row id of each view row (the row column of the state table)")
    if len(ids) != n_rows:
        raise ValueError(f"rows has {len(ids):,} values, but fold_of_row has {n_rows:,}; give one row id per view row")
    if len(np.unique(ids)) != n_rows:
        raise ValueError("rows must hold each row id once, because a view has one row per Delivery")
    return ids.copy()


def _check_frame(view: Any) -> None:
    if not isinstance(view, pd.DataFrame):
        raise TypeError(f"view must be a DataFrame (a sandbox view); got {type(view).__name__}")
    if not view.columns.is_unique:
        raise ValueError("view repeats a column name; a sandbox view has each column once")


def _cells(folds: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """(fold, ascending positions of the view rows in that fold) for each fold present, ascending."""
    order = np.argsort(folds, kind="stable")
    present, starts, counts = np.unique(folds[order], return_index=True, return_counts=True)
    return [
        (fold, order[start : start + count])
        for fold, start, count in zip(present.tolist(), starts.tolist(), counts.tolist())
    ]


# ----------------------------------------------------------------------------------- FoldValues

def _check_plan(plan: Any) -> dict[tuple[int, int], str]:
    if not isinstance(plan, Mapping):
        raise TypeError(
            "plan must be a mapping from (view k, row fold) to a fit id, such as view_plan gives; "
            f"got {type(plan).__name__}"
        )
    checked: dict[tuple[int, int], str] = {}
    for key, fit_id in plan.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError(f"each plan key must be a pair (view k, row fold); got {key!r}")
        cell = (_fold_number(key[0], "a plan key's view k"), _fold_number(key[1], "a plan key's row fold"))
        checked[cell] = _check_fit_id(fit_id, f"the plan's fit for view V_{cell[0]}, fold {cell[1]}")
    return checked


def _value_array(values: Any, fit_id: str) -> np.ndarray:
    if values is None or isinstance(values, (str, bytes, Mapping, pd.DataFrame)):
        raise TypeError(f"the values of fit {fit_id!r} must be a 1-D array; got {type(values).__name__}")
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"the values of fit {fit_id!r} must be 1-D; got shape {array.shape}")
    if array.dtype.kind not in _VALUE_KINDS:
        raise TypeError(
            f"the values of fit {fit_id!r} must be numbers or booleans, with NaN for a missing value; "
            f"got dtype {array.dtype}"
        )
    readonly = array.view()
    readonly.flags.writeable = False
    return readonly


def _check_values(values: Any) -> dict[str, np.ndarray]:
    if not isinstance(values, Mapping):
        raise TypeError(f"values must be a mapping from fit id to a 1-D array; got {type(values).__name__}")
    return {_check_fit_id(fit_id, "each key of values"): _value_array(array, fit_id) for fit_id, array in values.items()}


@dataclass(frozen=True, eq=False, repr=False)
class FoldValues:
    """One model-derived quantity of the sandbox views, from every fit that produced it.

    * ``fit_column``: M1_FIT or M2_FIT, the provenance column of the quantity's group;
    * ``plan``: (view k, row fold i) -> the fit id of the fit that serves the rows of V_k in
      fold i (view_plan gives the design's plan);
    * ``values``: fit id -> a 1-D numeric or boolean array, in the full layout (one value per view
      row) or the compact layout (one value per row the fit serves), both in view row order.

    The module docstring ("FoldValues and the view plan") gives the layouts. The mappings are
    read-only copies; the arrays are read-only views of the given ones. Raises ValueError or
    TypeError for a malformed argument.
    """

    fit_column: str
    plan: Mapping[tuple[int, int], str]
    values: Mapping[str, np.ndarray]
    _served: Mapping[str, frozenset[int]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.fit_column, str) or self.fit_column not in FIT_COLUMNS:
            raise ValueError(f"fit_column must be one of {', '.join(FIT_COLUMNS)}; got {self.fit_column!r}")
        plan = _check_plan(self.plan)
        values = _check_values(self.values)
        served: dict[str, set[int]] = {}
        for (_, fold), fit_id in plan.items():
            served.setdefault(fit_id, set()).add(fold)
        object.__setattr__(self, "plan", MappingProxyType(plan))
        object.__setattr__(self, "values", MappingProxyType(values))
        object.__setattr__(
            self, "_served", MappingProxyType({fit_id: frozenset(folds) for fit_id, folds in served.items()})
        )

    def fit_for(self, k: int, fold: int) -> str | None:
        """The fit id that the plan names for the rows of view V_k in ``fold``; None when it names none."""
        return self.plan.get((k, fold))

    def served_folds(self, fit_id: str) -> frozenset[int]:
        """The folds whose rows ``fit_id`` serves under the plan, in any view; empty when the plan does not name it."""
        return self._served.get(fit_id, frozenset())

    def __repr__(self) -> str:
        return (
            f"FoldValues(fit_column={self.fit_column!r}, planned_cells={len(self.plan)}, "
            f"fits_with_values={len(self.values)})"
        )


def view_plan(model: str, plan: Iterable[FitSpec], *, family: str | None = None) -> dict[tuple[int, int], str]:
    """The design's plan of a sandboxed model: (view k, row fold i) -> the fit id that serves those rows.

    ``plan`` is ``folds.sandbox_plan(D15)``, or part of it. Each FitSpec serves the rows of view
    ``sandbox_k`` in its ``predicted_fold``: outer spec k the fold-k rows of V_k (M1(-k), M2(-k)),
    inner spec (k, j) the fold-j rows of V_k (M1(-{j, k}), M2^(k)(-j)). The fit id is
    ``leakage.fit_id_for(model, spec, family=family)``, as ``FitRegistry.register_spec`` gives it,
    so ``view_plan("m1", plan)[(1, 2)]`` is ``"m1_inner_k1_j2"``.

    Raises ValueError when ``plan`` is empty, holds two specs for one (k, i), or the model or
    family makes an invalid fit id, and TypeError when it holds something other than FitSpecs.
    """
    if isinstance(plan, (str, bytes)) or not isinstance(plan, Iterable):
        raise TypeError(f"plan must be an iterable of folds.FitSpec, such as folds.sandbox_plan(D15); got {plan!r}")
    cells: dict[tuple[int, int], str] = {}
    for spec in plan:
        try:
            k, fold = spec.sandbox_k, spec.predicted_fold
        except AttributeError:
            raise TypeError(f"plan must hold folds.FitSpec objects; got {spec!r}") from None
        cell = (_fold_number(k, "FitSpec.sandbox_k"), _fold_number(fold, "FitSpec.predicted_fold"))
        fit_id = fit_id_for(model, spec, family=family)
        if cell in cells:
            raise ValueError(
                f"plan holds two FitSpecs for the rows of view V_{cell[0]} in fold {cell[1]}: "
                f"{cells[cell]!r} and {fit_id!r}"
            )
        cells[cell] = fit_id
    if not cells:
        raise ValueError("plan holds no FitSpec; pass folds.sandbox_plan(D15)")
    return cells


def view_table(k: int) -> str:
    """The checkpoint table name of view V_k: ``view_<k>`` (Data Models, ``view_k``)."""
    return f"{VIEW_TABLE_PREFIX}{_fold_number(k, 'k')}"


# --------------------------------------------------------------------------- building the views

def _check_sources(sources: Any) -> dict[str, FoldValues]:
    if not isinstance(sources, Mapping):
        raise TypeError(f"sources must be a mapping from quantity column to FoldValues; got {type(sources).__name__}")
    if not sources:
        raise ValueError("sources is empty; a view needs at least one quantity")
    checked: dict[str, FoldValues] = {}
    for name, source in sources.items():
        if not isinstance(name, str) or not name or name != name.strip():
            raise ValueError(f"each quantity name must be a non-empty str without surrounding spaces; got {name!r}")
        if name == ROW_COLUMN or name in FIT_COLUMNS:
            raise ValueError(f"{name!r} is a column that the view keeps for itself; give the quantity another name")
        if not isinstance(source, FoldValues):
            raise TypeError(f"the source of quantity {name!r} must be a FoldValues; got {type(source).__name__}")
        checked[name] = source
    return checked


def _planned_folds_text(source: FoldValues, k: int) -> str:
    folds = [fold for view, fold in source.plan if view == k]
    return f"it names fits for {_folds_text(folds)} of V_{k}" if folds else f"it names no fit for view V_{k}"


def _chosen_fits(
    k: int, cells: list[tuple[int, np.ndarray]], members: list[tuple[str, FoldValues]], column: str
) -> dict[int, str]:
    """fold -> the fit that serves the group's quantities on the rows of V_k in that fold."""
    chosen: dict[int, str] = {}
    for fold, _ in cells:
        picks: dict[str, list[str]] = {}
        for name, source in members:
            fit_id = source.fit_for(k, fold)
            if fit_id is None:
                raise ValueError(
                    f"the plan of quantity {name!r} ({column}) names no fit for the rows of view V_{k} in fold "
                    f"{fold}; {_planned_folds_text(source, k)}"
                )
            picks.setdefault(fit_id, []).append(name)
        if len(picks) > 1:
            shown = "; ".join(f"{fit_id!r} for {', '.join(map(repr, names))}" for fit_id, names in picks.items())
            raise ValueError(
                f"the quantities of {column} disagree on the fit that serves the rows of view V_{k} in fold {fold} "
                f"({shown}); {column} records one fit per row, so its quantities must share that fit"
            )
        chosen[fold] = next(iter(picks))
    return chosen


def _gather(
    k: int,
    name: str,
    source: FoldValues,
    chosen: dict[int, str],
    cells: list[tuple[int, np.ndarray]],
    folds: np.ndarray,
    served_rows: dict[frozenset[int], np.ndarray],
) -> np.ndarray:
    """The quantity's value on every view row, each from the fit chosen for the row's fold."""
    n_rows = len(folds)
    used = []
    for fold, positions in cells:
        fit_id = chosen[fold]
        array = source.values.get(fit_id)
        if array is None:
            raise ValueError(
                f"quantity {name!r} has no values of fit {fit_id!r}, which serves the rows of view V_{k} in fold {fold}"
            )
        used.append((positions, fit_id, array))
    arrays = [array for _, _, array in used] or list(source.values.values())
    dtype = np.result_type(*(array.dtype for array in arrays)) if arrays else np.dtype(np.float64)
    out = np.empty(n_rows, dtype=dtype)
    for positions, fit_id, array in used:
        if len(array) == n_rows:  # full layout
            out[positions] = array[positions]
            continue
        served = source.served_folds(fit_id)
        rows = served_rows.get(served)
        if rows is None:
            rows = served_rows[served] = np.flatnonzero(np.isin(folds, np.fromiter(served, dtype=np.int64)))
        if len(array) != len(rows):
            raise ValueError(
                f"quantity {name!r}: fit {fit_id!r} gives {len(array):,} values, but the view has {n_rows:,} rows and "
                f"the fit serves {len(rows):,} of them (the rows in {_folds_text(served)}); give one value per view "
                "row, or one per row that the fit serves, in view row order"
            )
        # Compact layout: the served rows are ascending and hold this fold's rows.
        out[positions] = array if len(rows) == len(positions) else array[np.searchsorted(rows, positions)]
    return out


def _fit_column(chosen: dict[int, str], cells: list[tuple[int, np.ndarray]], n_rows: int) -> Any:
    """The provenance column: the chosen fit id on every view row, as a ``str`` array."""
    fit_ids = list(dict.fromkeys(chosen[fold] for fold, _ in cells))
    code_of = {fit_id: code for code, fit_id in enumerate(fit_ids)}
    codes = np.zeros(n_rows, dtype=np.intp)
    for fold, positions in cells:
        codes[positions] = code_of[chosen[fold]]
    return pd.array(np.array(fit_ids, dtype=object)[codes], dtype="str")


def _view_parts(
    k: int, folds: np.ndarray, sources: dict[str, FoldValues]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """The quantity columns and the provenance columns of the groups in ``sources``."""
    cells = _cells(folds)
    groups: dict[str, list[tuple[str, FoldValues]]] = {}
    for name, source in sources.items():
        groups.setdefault(source.fit_column, []).append((name, source))
    chosen = {column: _chosen_fits(k, cells, members, column) for column, members in groups.items()}
    served_rows: dict[frozenset[int], np.ndarray] = {}
    quantities = {
        name: _gather(k, name, source, chosen[source.fit_column], cells, folds, served_rows)
        for name, source in sources.items()
    }
    fit_columns = {column: _fit_column(chosen[column], cells, len(folds)) for column in FIT_COLUMNS if column in groups}
    return quantities, fit_columns


def build_view(k: int, fold_of_row: Any, sources: Mapping[str, FoldValues], *, rows: Any = None) -> pd.DataFrame:
    """View V_k: every quantity of ``sources`` on every row, from the fit that the plan names for the row's fold.

    ``k`` is the target fold of the sandbox; ``fold_of_row`` the fold of each view row (a 1-D
    integer array, from the Fold_Table); ``sources`` maps each quantity column to its FoldValues;
    ``rows``, when given, the ``row`` id of each view row (for example the ``row`` column of the
    ``state`` table), written as the first column ``row``.

    A row in fold i takes the quantities of each group from the fit that the group's plan names
    for (k, i): in the design's plan, from the outer fit of fold k when i = k and from the inner
    fit (k, i) otherwise. The fit id goes into the group's provenance column (``m1_fit`` or
    ``m2_fit``). Returns a DataFrame on a RangeIndex with ``row`` (when given), the quantities in
    the order of ``sources`` and the provenance columns (dtype ``str``) in FIT_COLUMNS order. The
    view is not checked against the FitRegistry: call check_view before a fit reads it.

    Raises ValueError or TypeError when an argument is malformed, when a fold of ``fold_of_row``
    has no planned fit, when the quantities of one group disagree on a fit, or when a needed
    value array is missing or has neither layout's length.
    """
    k = _fold_number(k, "k")
    folds = _row_folds(fold_of_row)
    checked = _check_sources(sources)
    data: dict[str, Any] = {}
    if rows is not None:
        data[ROW_COLUMN] = _row_ids(rows, len(folds))
    quantities, fit_columns = _view_parts(k, folds, checked)
    data.update(quantities)
    data.update(fit_columns)
    return pd.DataFrame(data, index=pd.RangeIndex(len(folds)))


def extend_view(k: int, view: pd.DataFrame, fold_of_row: Any, sources: Mapping[str, FoldValues]) -> pd.DataFrame:
    """A copy of view V_k with the quantities of ``sources`` added, as build_view would give them.

    Use it to add a second group to a view, such as the M2-derived quantities and ``m2_fit`` to
    the M1 view of the ``skill`` stage. ``fold_of_row`` gives the fold of each row of ``view``,
    in its row order. The result keeps the view's index and columns, adds the new quantities
    after them and puts the provenance columns last, in FIT_COLUMNS order. The view itself is
    not changed and is not checked: call check_view on the result before a fit reads it.

    Raises ValueError when ``view`` does not have one row per value of ``fold_of_row``, when a
    new quantity repeats a column of the view, when a group whose provenance column the view
    already has picks another fit than that column records on some row, and for the problems of
    build_view; TypeError for a malformed argument.
    """
    k = _fold_number(k, "k")
    _check_frame(view)
    folds = _row_folds(fold_of_row)
    if len(view) != len(folds):
        raise ValueError(f"view has {len(view):,} rows, but fold_of_row has {len(folds):,}; give the fold of each view row")
    checked = _check_sources(sources)
    repeated = [name for name in checked if name in view.columns]
    if repeated:
        raise ValueError(f"the view already has the column(s) {', '.join(repeated)}; each view column is written once")
    quantities, fit_columns = _view_parts(k, folds, checked)
    for column, new in fit_columns.items():
        if column in view.columns:
            old = view[column].to_numpy(dtype=object, na_value=None)
            differ = old != np.asarray(new, dtype=object)
            if differ.any():
                raise ValueError(
                    f"the view's {column} records other fits than the plan of the new {column} quantities picks, on "
                    f"{_plural(int(differ.sum()), 'row')}; the quantities of one group must come from the fits that "
                    "its provenance column records"
                )
    result = view.drop(columns=[column for column in FIT_COLUMNS if column in view.columns])
    for name, values in quantities.items():
        result[name] = values
    for column in FIT_COLUMNS:
        if column in view.columns:
            result[column] = view[column].array
        elif column in fit_columns:
            result[column] = fit_columns[column]
    return result


# ---------------------------------------------------------------------------- checking a view

@dataclass(frozen=True)
class ViewProblem:
    """One reason why check_view rejected rows of a view (ViewLeakageError.problems).

    * ``reason``: NOT_EXCLUDED, UNKNOWN_FIT, NO_FIT or NO_FIT_COLUMN (module docstring);
    * ``column``: the provenance column (M1_FIT or M2_FIT); None for NO_FIT_COLUMN;
    * ``fit_id``: the recorded fit id; None for NO_FIT and NO_FIT_COLUMN;
    * ``model``: the fit's model name from the FitRegistry (``model`` or ``model (family)``),
      ``excluded_folds``: its excluded folds, ascending, and ``overlap_count``: the matches of its
      Training_Rows (training and early-stopping matches) in ``missing_folds``; all three None
      unless the reason is NOT_EXCLUDED;
    * ``row_folds``: the folds of the rejected rows, ascending;
    * ``missing_folds``: the folds those rows need their fit to exclude (their own folds and k)
      that it does not exclude, ascending; empty unless the reason is NOT_EXCLUDED;
    * ``n_rows``: the rejected rows;
    * ``examples``: up to 5 rejected rows, by their ``row`` id, or by position when the view has
      no usable ``row`` column.
    """

    reason: str
    column: str | None
    fit_id: str | None
    model: str | None
    excluded_folds: tuple[int, ...] | None
    row_folds: tuple[int, ...]
    missing_folds: tuple[int, ...]
    n_rows: int
    overlap_count: int | None
    examples: tuple[int, ...]


class ViewLeakageError(LeakageError):
    """check_view rejected a sandbox view (exit code 4; Requirement 7, criteria 7, 8, 11 and 12).

    ``view_k``: the target fold of the view; ``n_view_rows``: its rows; ``positions``: the
    positions of the rejected rows, ascending (a read-only int64 array); ``n_rows``: how many;
    ``problems``: one ViewProblem per provenance column and recorded fit.
    """

    def __init__(
        self,
        message: str,
        *,
        view_k: int,
        n_view_rows: int,
        positions: np.ndarray,
        problems: Iterable[ViewProblem],
    ) -> None:
        super().__init__(message)
        self.view_k = int(view_k)
        self.n_view_rows = int(n_view_rows)
        self.positions = positions
        self.problems = tuple(problems)

    @property
    def n_rows(self) -> int:
        """The number of rejected rows."""
        return len(self.positions)


def _row_labels(view: pd.DataFrame) -> np.ndarray | None:
    """The view's ``row`` ids as int64, or None when it has no ``row`` column or one that is not whole numbers."""
    if ROW_COLUMN not in view.columns:
        return None
    try:
        return view[ROW_COLUMN].to_numpy(dtype=np.int64)
    except (TypeError, ValueError):
        return None


def _examples(positions: np.ndarray, labels: np.ndarray | None) -> tuple[int, ...]:
    shown = positions[:_SHOWN]
    return tuple(int(value) for value in (shown if labels is None else labels[shown]).tolist())


def _lookup(registry: FitRegistry, fit_id: Any) -> FitRecord | None:
    try:
        return registry.get(fit_id)
    except KeyError:  # leakage.UnknownFitError
        return None


def _overlap(registry: FitRegistry, record: FitRecord, missing: tuple[int, ...]) -> int:
    """The matches of the fit's Training_Rows that lie in the ``missing`` folds."""
    ids = record.training_match_ids
    if not missing or not len(ids):
        return 0
    return int(np.isin(registry.fold_of(ids), np.asarray(missing, dtype=np.int64)).sum())


def _problem_line(problem: ViewProblem, k: int, labelled: bool) -> str:
    if problem.reason == NO_FIT_COLUMN:
        return (
            f"the view has neither an {M1_FIT} nor an {M2_FIT} column, so none of its rows can be traced to the fits "
            "that supplied its values"
        )
    where = f"{_plural(problem.n_rows, 'row')} in {_folds_text(problem.row_folds)}"
    example = f"for example {'row' if labelled else 'position'} {', '.join(map(str, problem.examples))}"
    if problem.reason == NO_FIT:
        verb = "records" if problem.n_rows == 1 else "record"
        return (
            f"{problem.column}: {where} {verb} no fit, so the values cannot be traced to a fit that excludes the "
            f"row's fold and fold {k} ({example})"
        )
    if problem.reason == UNKNOWN_FIT:
        return (
            f"{problem.column}: fit {problem.fit_id!r} is not in the FitRegistry, so the {where} that took values "
            f"from it cannot be traced to a fit that excludes the row's fold and fold {k} ({example})"
        )
    overlap = problem.overlap_count or 0
    missing = problem.missing_folds
    return (
        f"{problem.column}: model {problem.model}, fit {problem.fit_id!r}, excluded "
        f"{_folds_text(problem.excluded_folds or ())}, {_plural(overlap, 'overlapping match', 'overlapping matches')}: "
        f"{where} took values from this fit, but a row of fold i in view V_{k} needs a fit that excludes fold i and "
        f"fold {k}. The fit does not exclude {_folds_text(missing)}, and its Training_Rows hold "
        f"{_plural(overlap, 'match', 'matches')} of {'that fold' if len(missing) == 1 else 'those folds'} ({example})"
    )


def _view_error(
    k: int, n_view_rows: int, positions: np.ndarray, problems: list[ViewProblem], labelled: bool
) -> ViewLeakageError:
    positions = np.asarray(positions, dtype=np.int64)
    positions.flags.writeable = False
    header = (
        f"Leakage guard, sandbox view V_{k} (Requirement 7, criteria 7, 8, 11 and 12): "
        f"{_plural(len(positions), 'row')} of the view's {n_view_rows:,} cannot be traced to a fit that excludes "
        f"both the row's fold and fold {k}:"
    )
    lines = [f"  - {_problem_line(problem, k, labelled)}" for problem in problems]
    return ViewLeakageError(
        "\n".join([header, *lines, _FOOTER]),
        view_k=k,
        n_view_rows=n_view_rows,
        positions=positions,
        problems=problems,
    )


def check_view(k: int, view: pd.DataFrame, fold_of_row: Any, registry: FitRegistry) -> None:
    """Check that every row of view V_k took its values from fits that exclude the row's fold and k.

    ``k`` is the view's target fold; ``view`` the view (build_view, extend_view, or a view read
    from a checkpoint); ``fold_of_row`` the fold of each view row, in view order, from the
    Fold_Table; ``registry`` the Run's FitRegistry (``ctx.registry``), which holds every fit that
    an earlier stage or this one registered.

    For each provenance column of ``view`` (``m1_fit``, ``m2_fit``), every row's fit must be in
    the registry, and its excluded folds must contain the row's fold and k (k alone for a row in
    fold k): M1(-{i, k}) or M2^(k)(-i) for a row in fold i != k, M1(-k) or M2(-k) for a row in
    fold k (criteria 7, 8 and 11). Returns None when every row passes.

    Raises ViewLeakageError (exit code 4) naming each fit at fault, with its model, excluded
    folds, overlapping matches and rejected rows (criterion 12), when a row fails, when a row
    records no fit or a fit that the registry does not hold, and when the view has no
    provenance column. Raises ValueError when ``view`` does not have one row per value of
    ``fold_of_row``, and TypeError or ValueError for a malformed argument.
    """
    k = _fold_number(k, "k")
    _check_frame(view)
    folds = _row_folds(fold_of_row)
    n_rows = len(folds)
    if len(view) != n_rows:
        raise ValueError(f"view has {len(view):,} rows, but fold_of_row has {n_rows:,}; give the fold of each view row")
    if not callable(getattr(registry, "get", None)) or not callable(getattr(registry, "fold_of", None)):
        raise TypeError(f"registry must be the Run's FitRegistry (superr.leakage); got {type(registry).__name__}")
    labels = _row_labels(view)
    labelled = labels is not None
    present = [column for column in FIT_COLUMNS if column in view.columns]
    if not present:
        everything = np.arange(n_rows, dtype=np.int64)
        problem = ViewProblem(
            reason=NO_FIT_COLUMN,
            column=None,
            fit_id=None,
            model=None,
            excluded_folds=None,
            row_folds=tuple(np.unique(folds).tolist()),
            missing_folds=(),
            n_rows=n_rows,
            overlap_count=None,
            examples=_examples(everything, labels),
        )
        raise _view_error(k, n_rows, everything, [problem], labelled)

    width = max(int(folds.max()) if n_rows else 0, k) + 1
    rejected_any = np.zeros(n_rows, dtype=bool)
    problems: list[ViewProblem] = []
    for column in present:
        codes, uniques = pd.factorize(view[column], use_na_sentinel=True)
        codes = np.asarray(codes, dtype=np.intp)
        fit_ids = list(uniques.tolist())
        records = [_lookup(registry, fit_id) for fit_id in fit_ids]
        # allowed[code, fold]: a row in that fold may take values from that fit. The extra last
        # row stands for code -1 (no fit recorded), which is never allowed.
        allowed = np.zeros((len(records) + 1, width), dtype=bool)
        for code, record in enumerate(records):
            if record is not None and k in record.excluded_folds:
                allowed[code, [fold for fold in record.excluded_folds if fold < width]] = True
        rejected = ~allowed[codes, folds]
        if not rejected.any():
            continue
        rejected_any |= rejected
        at = np.flatnonzero(rejected)
        bad_codes = codes[at]
        for code in np.unique(bad_codes).tolist():
            positions = at[bad_codes == code]
            row_folds = tuple(np.unique(folds[positions]).tolist())
            examples = _examples(positions, labels)
            record = records[code] if code >= 0 else None
            if code < 0 or record is None:
                problems.append(
                    ViewProblem(
                        reason=NO_FIT if code < 0 else UNKNOWN_FIT,
                        column=column,
                        fit_id=None if code < 0 else str(fit_ids[code]),
                        model=None,
                        excluded_folds=None,
                        row_folds=row_folds,
                        missing_folds=(),
                        n_rows=len(positions),
                        overlap_count=None,
                        examples=examples,
                    )
                )
                continue
            excluded = record.excluded_folds
            missing = tuple(sorted(({k} | set(row_folds)) - excluded))
            problems.append(
                ViewProblem(
                    reason=NOT_EXCLUDED,
                    column=column,
                    fit_id=record.fit_id,
                    model=record.label,
                    excluded_folds=tuple(sorted(excluded)),
                    row_folds=row_folds,
                    missing_folds=missing,
                    n_rows=len(positions),
                    overlap_count=_overlap(registry, record, missing),
                    examples=examples,
                )
            )
    if problems:
        raise _view_error(k, n_rows, np.flatnonzero(rejected_any), problems, labelled)
