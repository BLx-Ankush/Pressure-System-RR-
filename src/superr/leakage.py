"""Leakage guard: the FitRegistry that checks every model fit against the Fold_Table, and the
early-stopping split (Requirement 2, criterion 6; Requirement 7, criteria 5, 6, 9 and 12;
Requirement 10, criterion 7).

Design sections "Fold sandboxes and leakage control" (paragraph "Leakage guard"), "Determinism"
and "Fold_Assigner, views and leakage guard", and Data Models, ``fit_registry``.

The guard (``FitRegistry.register``)
    A model stage registers every fit of M1-M4, the Extras_Model and the Alternative_Models
    before it runs the fit. The registration names:

    * ``model``, and optionally ``family``: what the error message calls the model, for example
      ``"m2_capped"`` with family ``"lgbm"``;
    * ``fit_id``: unique in the Run, because the sandbox views record it in ``m1_fit`` and
      ``m2_fit`` and the saved side tables are keyed by it. ``fit_id_for`` builds one from a
      ``folds.FitSpec``, such as ``m3_inn1_lgbm_inner_k1_j2``;
    * ``excluded_folds``: {k} for an outer fit and {j, k} for an inner fit
      (``FitSpec.excluded_folds``), and no fold for the all-fold M1 fit behind the top-theta
      lists (Requirement 8, criterion 17), which gives no output to any Delivery;
    * ``train_match_ids``: the matches whose Deliveries the fit trains on, and
      ``early_stopping_match_ids``: the matches held out to decide early stopping, if any.
      Both are Training_Rows (glossary). A match given in both is recorded once, as an
      early-stopping match. Per-Delivery ``match_id`` columns may be passed as they are;
      repeated ids are ignored;
    * ``predicted_match_ids``: the matches whose Deliveries receive the fit's output, directly or
      as a feature or calibration input of another fit: the fold-k rows of an outer fit, the
      fold-j rows inside sandbox k of an inner fit. In-fold metrics are not outputs and are not
      registered;
    * optionally ``sandbox_k``, ``inner_j`` (both from the FitSpec; ``register_spec`` fills them)
      and ``n_train_rows``.

    The guard takes the fold of every match from the Fold_Table (criterion 5) and accepts the fit
    only when:

    1. no training match and no early-stopping match lies in an excluded fold (criteria 6-8;
       Requirement 10, criterion 7, which keeps the validation matches in the training folds);
    2. every predicted match lies in an excluded fold, so a fit that gives output to a fold-k
       Delivery excludes fold k (criteria 6 and 9; Requirement 2, criterion 6). With rule 1, no
       predicted match is then a match of the fit's Training_Rows (criterion 9);
    3. ``sandbox_k`` and ``inner_j``, when given, are among the excluded folds, because the fit's
       output serves the Deliveries of fold k through sandbox k (criteria 7 and 8).

    Otherwise ``register`` raises FitLeakageError, a ``superr.errors.LeakageError`` (exit code 4),
    and records nothing. The message gives the model name, the excluded fold numbers and the
    number of overlapping matches (criterion 12), then the count per fold and example match ids
    for each broken rule. The model stages (06-10) run before ``pressure``, so the Run stops
    before any pressure, value or rating output is written.

    A match id that is missing, not a whole number or not in the Fold_Table raises DataError
    (exit code 3). A malformed argument raises TypeError or ValueError, and so does a fit id
    registered before with other values (registering the same fit again changes nothing).

The early-stopping split (``early_stopping_split``)
    Draws the validation matches of one fit from its training matches, as whole matches: the
    distinct match ids, in ascending order, are permuted by
    ``numpy.random.Generator(PCG64(seed + fit_index))`` and the first ``n_val`` are held out,
    where ``n_val`` is ``share`` x n rounded half up and kept between 1 and n - 1. ``seed`` is the
    model's named seed (``seeds.m2``, ``seeds.m3_lgbm``, ...), ``share`` its ``es_share``
    (Requirement 10, criterion 7) and ``fit_index`` the fit's position in
    ``folds.sandbox_plan(D15)``, so every fit of a model gets its own draw. The same matches,
    share, seed and index give the same split in any input order (Requirement 28, criterion 3).

Saved tables
    ``save(store, stage)`` writes the fits that ``stage`` registered to that stage's checkpoint as
    three tables, so the Test_Suite can re-check every fit against the Fold_Table (criterion 9):

    * ``fit_registry`` (``fit_registry.parquet``), one row per fit in registration order, with
      FIT_REGISTRY_COLUMNS: ``model`` (str), ``family`` (str, missing when not given),
      ``fit_id`` (str), ``sandbox_k`` and ``inner_j`` (Int32, missing when not given),
      ``excluded_folds`` (list of fold numbers, ascending; a numpy array per cell after
      loading), ``n_train_rows`` (Int64, missing when not given), ``n_train_matches``,
      ``n_early_stopping_matches`` and ``n_predicted_matches`` (int64) and ``stage`` (str);
    * ``fit_train_matches`` (TRAIN_MATCH_COLUMNS): ``fit_id``, ``match_id`` (int64) and
      ``early_stopping`` (bool), one row per match of the fit's Training_Rows;
    * ``fit_predicted_matches`` (PREDICTED_MATCH_COLUMNS): ``fit_id`` and ``match_id`` (int64), one
      row per predicted match.

    Side-table rows follow the fits' registration order, then ``match_id``. ``to_frame``,
    ``train_matches_frame`` and ``predicted_matches_frame`` return the same tables in memory, and
    ``records_from_frames`` turns them back into FitRecords.

Use in a Run
    The runner makes one registry per Run with ``FitRegistry.for_store(store)``, puts it in
    ``StageContext.registry`` and calls ``begin_stage(stage.name)`` before each stage, so every fit
    is tagged with the stage that registered it. The registry reads the Fold_Table from the
    ``folds`` checkpoint when it first needs it. A model stage::

        plan = folds.sandbox_plan(settings.get("design_choices.D15_n_folds"))
        for fit_index, spec in enumerate(plan):
            fit_matches, es_matches = leakage.early_stopping_split(
                train_matches, settings.get("models.m2.es_share"), settings.get("seeds.m2"), fit_index
            )
            ctx.registry.register_spec(
                "m2_capped", spec, fit_matches, predicted_matches,
                early_stopping_match_ids=es_matches, family="lgbm", n_train_rows=n_rows,
            )
            ...  # only now fit the model
        ctx.registry.save(ctx.store, STAGE_NAME)

    With ``--from-stage X`` the runner calls ``load_saved(store, stage)`` for every stage before X.
    It adds the fits each saved ``fit_registry`` holds, re-checked against the Fold_Table, so
    ``views.check_view`` can look up every fit an earlier stage recorded in a view through
    ``excluded_folds(fit_id)`` or ``get(fit_id)``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from superr.errors import DataError, LeakageError

if TYPE_CHECKING:
    from superr.folds import FitSpec
    from superr.store import CheckpointStore

__all__ = [
    "FIT_PREDICTED_MATCHES",
    "FIT_REGISTRY",
    "FIT_REGISTRY_COLUMNS",
    "FIT_REGISTRY_FILE",
    "FIT_TRAIN_MATCHES",
    "PREDICTED_MATCH_COLUMNS",
    "TRAIN_MATCH_COLUMNS",
    "FitLeakageError",
    "FitRecord",
    "FitRegistry",
    "UnknownFitError",
    "early_stopping_split",
    "fit_id_for",
    "records_from_frames",
]

# ------------------------------------------------------------------------------------- constants

#: Checkpoint names of the three tables that FitRegistry.save writes, and the main table's file.
FIT_REGISTRY = "fit_registry"
FIT_TRAIN_MATCHES = "fit_train_matches"
FIT_PREDICTED_MATCHES = "fit_predicted_matches"
FIT_REGISTRY_FILE = f"{FIT_REGISTRY}.parquet"

#: Columns of the three tables, in order (module docstring, "Saved tables").
FIT_REGISTRY_COLUMNS: tuple[str, ...] = (
    "model",
    "family",
    "fit_id",
    "sandbox_k",
    "inner_j",
    "excluded_folds",
    "n_train_rows",
    "n_train_matches",
    "n_early_stopping_matches",
    "n_predicted_matches",
    "stage",
)
TRAIN_MATCH_COLUMNS: tuple[str, ...] = ("fit_id", "match_id", "early_stopping")
PREDICTED_MATCH_COLUMNS: tuple[str, ...] = ("fit_id", "match_id")

#: A fit id: letters, digits, "_", "." and "-", starting with a letter or digit, so that it can
#: also name a checkpoint file once lower-cased.
_FIT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_SHOWN = 5  # example match ids named in one message
_FOOTER_NEW = "The fit was not run. The Run stops before any pressure, value or rating output is written."


def _footer_saved(stage: str) -> str:
    return (
        f"The fit was saved by the earlier Pipeline_Stage {stage!r}, so its outputs cannot be reused. The Run "
        "stops before any pressure, value or rating output is written."
    )


# ---------------------------------------------------------------------------------------- errors

class FitLeakageError(LeakageError):
    """The leakage guard rejected a fit (exit code 4; Requirement 7, criterion 12).

    The attributes hold the facts that the message reports: ``model``, ``family`` (None when not
    given), ``fit_id``, ``excluded_folds`` (an ascending tuple) and ``overlap_count``, the number
    of matches that break the guard's rules.
    """

    def __init__(
        self,
        message: str,
        *,
        model: str | None = None,
        family: str | None = None,
        fit_id: str | None = None,
        excluded_folds: Iterable[int] = (),
        overlap_count: int = 0,
    ) -> None:
        super().__init__(message)
        self.model = model
        self.family = family
        self.fit_id = fit_id
        self.excluded_folds = tuple(sorted(int(fold) for fold in excluded_folds))
        self.overlap_count = int(overlap_count)


class UnknownFitError(KeyError):
    """No fit with this id is in the FitRegistry."""

    def __str__(self) -> str:  # KeyError would print the message in quotes
        return str(self.args[0]) if self.args else ""


# ------------------------------------------------------------------------------------ the record

def _frozen(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


_NO_IDS = _frozen(np.zeros(0, dtype=np.int64))


@dataclass(frozen=True, eq=False, repr=False)
class FitRecord:
    """One registered fit. The match-id arrays are distinct, ascending, read-only int64 arrays.

    ``train_match_ids`` and ``early_stopping_match_ids`` never share a match; together they are the
    matches of the fit's Training_Rows (``training_match_ids``). ``stage`` is the Pipeline_Stage
    that registered the fit (None outside a stage).
    """

    model: str
    family: str | None
    fit_id: str
    sandbox_k: int | None
    inner_j: int | None
    excluded_folds: frozenset[int]
    n_train_rows: int | None
    train_match_ids: np.ndarray
    early_stopping_match_ids: np.ndarray
    predicted_match_ids: np.ndarray
    stage: str | None

    @property
    def label(self) -> str:
        """The model name for messages: ``model``, or ``model (family)``."""
        return self.model if self.family is None else f"{self.model} ({self.family})"

    @property
    def training_match_ids(self) -> np.ndarray:
        """Every match of the fit's Training_Rows: the training and early-stopping matches, ascending."""
        return np.union1d(self.train_match_ids, self.early_stopping_match_ids)

    def same_as(self, other: FitRecord) -> bool:
        """True when ``other`` records the same fit with the same values, stage tag included."""
        scalars = ("model", "family", "fit_id", "sandbox_k", "inner_j", "excluded_folds", "n_train_rows", "stage")
        arrays = ("train_match_ids", "early_stopping_match_ids", "predicted_match_ids")
        return all(getattr(self, name) == getattr(other, name) for name in scalars) and all(
            np.array_equal(getattr(self, name), getattr(other, name)) for name in arrays
        )

    def __repr__(self) -> str:
        return (
            f"FitRecord(fit_id={self.fit_id!r}, model={self.label!r}, excluded_folds={sorted(self.excluded_folds)}, "
            f"sandbox_k={self.sandbox_k}, inner_j={self.inner_j}, train_matches={len(self.train_match_ids)}, "
            f"early_stopping_matches={len(self.early_stopping_match_ids)}, "
            f"predicted_matches={len(self.predicted_match_ids)}, stage={self.stage!r})"
        )


# ----------------------------------------------------------------------------- argument checks

def _check_text(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or "\n" in value:
        raise ValueError(f"{what} must be a non-empty str without surrounding spaces or line breaks; got {value!r}")
    return str.__str__(value)


def _check_fit_id(value: Any) -> str:
    if not isinstance(value, str) or not _FIT_ID.fullmatch(value):
        raise ValueError(
            f"fit_id must be 1-128 letters, digits, '_', '.' or '-', starting with a letter or digit, such as "
            f"'m2_capped_lgbm_outer_k1' (fit_id_for builds one); got {value!r}"
        )
    return str.__str__(value)


def _check_stage(value: Any) -> str | None:
    if value is None or (isinstance(value, str) and value and value == value.strip()):
        return value
    raise ValueError(f"a Pipeline_Stage name must be None or a non-empty str without surrounding spaces; got {value!r}")


def _whole(value: Any, what: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{what} must be a whole number; got {value!r}")
    return int(value)


def _fold_number(value: Any, what: str) -> int:
    number = _whole(value, what)
    if number < 1:
        raise ValueError(f"{what}: folds are numbered from 1; got {number}")
    return number


def _optional_fold(value: Any, what: str) -> int | None:
    return None if value is None else _fold_number(value, what)


def _optional_count(value: Any, what: str) -> int | None:
    if value is None:
        return None
    number = _whole(value, what)
    if number < 0:
        raise ValueError(f"{what} must be 0 or more; got {number}")
    return number


def _fold_set(values: Any, what: str) -> frozenset[int]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise TypeError(f"{what} must be a collection of fold numbers, such as FitSpec.excluded_folds; got {values!r}")
    return frozenset(_fold_number(value, what) for value in values)


def _whole_numbers(values: Any, what: str, context: str) -> np.ndarray:
    """``values`` (a 1-D array-like of match ids) as int64, in order; DataError for a bad id."""
    if isinstance(values, (set, frozenset)):
        values = sorted(values)
    if values is None or isinstance(values, (str, bytes, pd.DataFrame)) or np.ndim(values) != 1:
        raise TypeError(
            f"{what} must be a 1-D array, list or Series of match ids (an empty one when there is none); "
            f"got {type(values).__name__}"
        )
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    if pd.api.types.is_bool_dtype(series.dtype):
        raise TypeError(f"{what} holds booleans, not match ids")
    missing = series.isna().to_numpy(dtype=bool)
    numbers = pd.to_numeric(series, errors="coerce")
    if pd.api.types.is_integer_dtype(numbers.dtype):
        whole = ~missing
        out = numbers.to_numpy(dtype=np.int64, na_value=0)
    else:
        floats = numbers.to_numpy(dtype=np.float64, na_value=np.nan)
        with np.errstate(invalid="ignore"):
            whole = np.isfinite(floats) & (np.floor(floats) == floats) & (np.abs(floats) <= 2.0**53)
        out = np.where(whole, floats, 0.0).astype(np.int64)
    problems = []
    if missing.any():
        problems.append(f"{what} has {int(missing.sum()):,} missing match id(s)")
    bad = ~missing & ~whole
    if bad.any():
        examples = ", ".join(repr(value) for value in series.iloc[np.flatnonzero(bad)[:3]].tolist())
        problems.append(f"{what} has {int(bad.sum()):,} value(s) that are not whole numbers (for example {examples})")
    if problems:
        raise DataError("\n".join([f"{context}:"] + [f"  - {problem}" for problem in problems]))
    return out


def _match_ids(values: Any, what: str, context: str) -> np.ndarray:
    """The distinct match ids of ``values``, ascending, as a read-only int64 array."""
    return _frozen(np.unique(_whole_numbers(values, what, context)))


def _build_record(
    *,
    model: Any,
    family: Any,
    fit_id: Any,
    excluded_folds: Any,
    train_match_ids: Any,
    early_stopping_match_ids: Any,
    predicted_match_ids: Any,
    sandbox_k: Any,
    inner_j: Any,
    n_train_rows: Any,
    stage: Any,
) -> FitRecord:
    """A checked FitRecord; a match given as training and early-stopping match is kept as the latter."""
    model = _check_text(model, "model")
    family = None if family is None else _check_text(family, "family")
    fit_id = _check_fit_id(fit_id)
    excluded = _fold_set(excluded_folds, "excluded_folds")
    sandbox_k = _optional_fold(sandbox_k, "sandbox_k")
    inner_j = _optional_fold(inner_j, "inner_j")
    if inner_j is not None and (sandbox_k is None or inner_j == sandbox_k):
        raise ValueError(f"fit {fit_id!r}: inner_j needs a sandbox_k, and the two must differ; got k={sandbox_k}, j={inner_j}")
    n_train_rows = _optional_count(n_train_rows, "n_train_rows")
    context = f"The leakage guard cannot read the matches of fit {fit_id!r} of model {model}"
    early = _NO_IDS if early_stopping_match_ids is None else _match_ids(
        early_stopping_match_ids, "early_stopping_match_ids", context
    )
    train = _match_ids(train_match_ids, "train_match_ids", context)
    if len(early):
        train = _frozen(np.setdiff1d(train, early, assume_unique=True))
    predicted = _match_ids(predicted_match_ids, "predicted_match_ids", context)
    return FitRecord(
        model=model,
        family=family,
        fit_id=fit_id,
        sandbox_k=sandbox_k,
        inner_j=inner_j,
        excluded_folds=excluded,
        n_train_rows=n_train_rows,
        train_match_ids=train,
        early_stopping_match_ids=early,
        predicted_match_ids=predicted,
        stage=_check_stage(stage),
    )


# --------------------------------------------------------------------------------- messages

def _matches(count: int) -> str:
    return f"{count:,} match" if count == 1 else f"{count:,} matches"


def _folds_text(folds: Iterable[int]) -> str:
    values = sorted(folds)
    if not values:
        return "no fold"
    if len(values) == 1:
        return f"fold {values[0]}"
    return "folds " + ", ".join(map(str, values[:-1])) + f" and {values[-1]}"


def _breakdown(ids: np.ndarray, folds: np.ndarray) -> str:
    """``"2 in fold 1, 1 in fold 3; for example match 101 (fold 1), ..."`` for matches and their folds."""
    numbers, counts = np.unique(folds, return_counts=True)
    per_fold = ", ".join(f"{int(count):,} in fold {int(number)}" for number, count in zip(numbers, counts))
    examples = ", ".join(f"{int(match)} (fold {int(fold)})" for match, fold in zip(ids[:_SHOWN], folds[:_SHOWN]))
    return f"{per_fold}; for example match {examples}"


# ------------------------------------------------------------------------------- the registry

class FitRegistry:
    """Every model fit of a Run, each checked by the leakage guard (module docstring)."""

    __slots__ = ("_fold_ids", "_fold_index", "_fold_values", "_loader", "_records", "_stage")

    def __init__(self, fold_table: pd.DataFrame | None = None) -> None:
        """Registry that checks fits against ``fold_table`` (the Fold_Table), or against the table
        given later to ``use_fold_table``. In a Run, use ``FitRegistry.for_store``."""
        self._records: dict[str, FitRecord] = {}
        self._stage: str | None = None
        self._fold_ids: np.ndarray | None = None  # Fold_Table match ids, ascending
        self._fold_values: np.ndarray | None = None  # their folds
        self._fold_index: pd.Index | None = None
        self._loader: Callable[[], pd.DataFrame] | None = None
        if fold_table is not None:
            self.use_fold_table(fold_table)

    @classmethod
    def for_store(cls, store: CheckpointStore) -> FitRegistry:
        """The registry of a Run: it reads the Fold_Table from the ``folds`` checkpoint of ``store``
        when it first needs it (Requirement 7, criterion 5)."""
        from superr import folds  # imported here so that superr.pipeline, which imports this module, imports no stage module

        registry = cls()
        registry._loader = lambda: store.load_table(folds.STAGE_NAME, folds.FOLD_TABLE)
        return registry

    def __repr__(self) -> str:
        if self._fold_index is not None:
            table = f"{len(self._fold_index):,} matches"
        else:
            table = "read when needed" if self._loader is not None else "none yet"
        return f"FitRegistry(fits={len(self._records)}, stage={self._stage!r}, fold_table={table})"

    # -- Fold_Table ------------------------------------------------------------------------

    @property
    def has_fold_table(self) -> bool:
        """True once a Fold_Table is attached (a registry from for_store reads it at first use)."""
        return self._fold_index is not None

    def use_fold_table(self, fold_table: pd.DataFrame) -> None:
        """Check every fit against ``fold_table``, the Fold_Table (``match_id``, ``fold``).

        Giving the attached table again changes nothing. Raises LeakageError (exit code 4) when
        another table with other folds is already attached, because every fit of a Run takes its
        folds from the one Fold_Table (Requirement 7, criterion 5); otherwise the errors of
        ``folds.fold_of`` for a malformed table.
        """
        from superr.folds import fold_of  # see for_store

        if not isinstance(fold_table, pd.DataFrame):
            raise TypeError(f"fold_table must be a DataFrame; got {type(fold_table).__name__}")
        absent = [column for column in ("match_id", "fold") if column not in fold_table.columns]
        if absent:
            raise ValueError(f"fold_table lacks the column(s) {', '.join(absent)}; pass the Fold_Table of the folds stage")
        values = fold_of(fold_table["match_id"], fold_table)  # checks the table: whole, unique ids; folds from 1
        ids = _whole_numbers(fold_table["match_id"], "Fold_Table match_id", "The Fold_Table cannot be used")
        order = np.argsort(ids, kind="stable")
        ids, values = _frozen(ids[order]), _frozen(values[order].astype(np.int32))
        if self._fold_ids is not None:
            if np.array_equal(ids, self._fold_ids) and np.array_equal(values, self._fold_values):
                return
            common, old_at, new_at = np.intersect1d(self._fold_ids, ids, assume_unique=True, return_indices=True)
            moved = int((self._fold_values[old_at] != values[new_at]).sum())
            raise LeakageError(
                f"The FitRegistry already checks fits against another Fold_Table: {_matches(moved)} lie in another "
                f"fold, {_matches(len(self._fold_ids) - len(common))} are missing and {_matches(len(ids) - len(common))} "
                "are new. Every fit of a Run takes its folds from the one Fold_Table of the folds stage "
                "(Requirement 7, criterion 5), so the Run stops."
            )
        self._fold_ids, self._fold_values, self._fold_index = ids, values, pd.Index(ids)

    def _lookup(self) -> tuple[pd.Index, np.ndarray]:
        if self._fold_index is None:
            if self._loader is None:
                raise ValueError(
                    "the FitRegistry has no Fold_Table to check fits against: build it with "
                    "FitRegistry(fold_table) or FitRegistry.for_store(store), or call use_fold_table first"
                )
            self.use_fold_table(self._loader())
        assert self._fold_values is not None
        return self._fold_index, self._fold_values

    def fold_of(self, match_ids: Any) -> np.ndarray:
        """The fold (int32) of each match id in ``match_ids``, in order, from the Fold_Table.

        Raises DataError (exit code 3) when an id is missing, not a whole number or not in the
        Fold_Table.
        """
        context = "Matches cannot get their fold from the FitRegistry's Fold_Table"
        return self._folds(_whole_numbers(match_ids, "match_ids", context), "match id", context)

    def _folds(self, ids: np.ndarray, what: str, context: str) -> np.ndarray:
        index, values = self._lookup()
        at = index.get_indexer(ids)
        absent = at < 0
        if absent.any():
            unknown = np.unique(ids[absent])
            shown = ", ".join(map(str, unknown[:_SHOWN].tolist()))
            raise DataError(
                f"{context}:\n  - {len(unknown):,} {what}(s) are not in the Fold_Table (for example {shown})"
            )
        return values[at]

    # -- stages (runner) -------------------------------------------------------------------

    @property
    def current_stage(self) -> str | None:
        """The Pipeline_Stage name that register() gives new fits (None outside a stage)."""
        return self._stage

    def begin_stage(self, stage: str | None) -> None:
        """Tag the fits registered from now on with this Pipeline_Stage name (None: no stage)."""
        self._stage = _check_stage(stage)

    # -- registering (model stages) --------------------------------------------------------

    def register(
        self,
        model: str,
        fit_id: str,
        excluded_folds: Iterable[int],
        train_match_ids: Any,
        predicted_match_ids: Any,
        *,
        early_stopping_match_ids: Any = None,
        family: str | None = None,
        sandbox_k: int | None = None,
        inner_j: int | None = None,
        n_train_rows: int | None = None,
    ) -> FitRecord:
        """Check one fit with the leakage guard before it runs, and record it; return the record.

        The arguments and the three rules are in the module docstring ("The guard"). Match ids
        are 1-D array-likes (an empty one when there is none). Raises FitLeakageError (exit code
        4) when a rule is broken, DataError (exit code 3) for a match id that is malformed or not
        in the Fold_Table, ValueError when the Fold_Table is not available or the fit id was
        registered before with other values, and TypeError or ValueError for a malformed argument.
        Nothing is recorded when it raises.
        """
        record = _build_record(
            model=model,
            family=family,
            fit_id=fit_id,
            excluded_folds=excluded_folds,
            train_match_ids=train_match_ids,
            early_stopping_match_ids=early_stopping_match_ids,
            predicted_match_ids=predicted_match_ids,
            sandbox_k=sandbox_k,
            inner_j=inner_j,
            n_train_rows=n_train_rows,
            stage=self._stage,
        )
        self._check(record)
        self._add([record])
        return self._records[record.fit_id]

    def register_spec(
        self,
        model: str,
        spec: FitSpec,
        train_match_ids: Any,
        predicted_match_ids: Any,
        *,
        early_stopping_match_ids: Any = None,
        family: str | None = None,
        n_train_rows: int | None = None,
        fit_id: str | None = None,
    ) -> FitRecord:
        """``register`` for the fit that ``spec`` (a ``folds.FitSpec`` of ``sandbox_plan``) describes.

        The excluded folds, ``sandbox_k`` and ``inner_j`` come from the spec, and the fit id is
        ``fit_id_for(model, spec, family=family)`` unless ``fit_id`` is given.
        """
        return self.register(
            model,
            fit_id_for(model, spec, family=family) if fit_id is None else fit_id,
            spec.excluded_folds,
            train_match_ids,
            predicted_match_ids,
            early_stopping_match_ids=early_stopping_match_ids,
            family=family,
            sandbox_k=spec.sandbox_k,
            inner_j=spec.inner_j,
            n_train_rows=n_train_rows,
        )

    def _check(self, record: FitRecord, *, saved_by: str | None = None) -> None:
        """The three rules of the guard; FitLeakageError when one is broken."""
        context = f"The leakage guard cannot check fit {record.fit_id!r} of model {record.label}"
        excluded = np.array(sorted(record.excluded_folds), dtype=np.int64)
        lines: list[str] = []
        overlap = 0
        for ids, what, rule in (
            (
                record.train_match_ids,
                "training",
                "Training_Rows must hold no Delivery of an excluded fold (Requirement 7, criteria 6-8; Requirement 2, criterion 6)",
            ),
            (
                record.early_stopping_match_ids,
                "early-stopping validation",
                "validation matches must come from the training folds only (Requirement 10, criterion 7)",
            ),
        ):
            folds = self._folds(ids, f"{what} match id", context)
            inside = np.isin(folds, excluded)
            if inside.any():
                overlap += int(inside.sum())
                lines.append(
                    f"{int(inside.sum()):,} of its {len(ids):,} {what} matches lie in its excluded folds "
                    f"({_breakdown(ids[inside], folds[inside])}); {rule}"
                )
        predicted = record.predicted_match_ids
        folds = self._folds(predicted, "predicted match id", context)
        outside = ~np.isin(folds, excluded)
        if outside.any():
            overlap += int(outside.sum())
            lines.append(
                f"{int(outside.sum()):,} of its {len(predicted):,} predicted matches lie in folds it does not exclude "
                f"({_breakdown(predicted[outside], folds[outside])}); a fit that gives output to a match must exclude "
                "that match's fold (Requirement 7, criteria 6 and 9)"
            )
        missing = [
            f"{name} = {fold}"
            for name, fold in (("k", record.sandbox_k), ("j", record.inner_j))
            if fold is not None and fold not in record.excluded_folds
        ]
        if missing:
            lines.append(
                f"its sandbox fold {' and '.join(missing)} is not among its excluded folds; a fit whose output serves "
                "the Deliveries of sandbox k must exclude fold k, and an inner fit fold j as well (Requirement 7, "
                "criteria 7 and 8)"
            )
        if not lines:
            return
        header = (
            f"Leakage guard (Requirement 7, criterion 12): model {record.label}, fit {record.fit_id!r}, excluded "
            f"{_folds_text(record.excluded_folds)}, {overlap:,} overlapping {'match' if overlap == 1 else 'matches'}:"
        )
        footer = _FOOTER_NEW if saved_by is None else _footer_saved(saved_by)
        raise FitLeakageError(
            "\n".join([header] + [f"  - {line}" for line in lines] + [footer]),
            model=record.model,
            family=record.family,
            fit_id=record.fit_id,
            excluded_folds=record.excluded_folds,
            overlap_count=overlap,
        )

    def _add(self, records: Iterable[FitRecord]) -> None:
        """Add every record, or none when one conflicts with a fit id already registered."""
        new: dict[str, FitRecord] = {}
        for record in records:
            old = self._records.get(record.fit_id) or new.get(record.fit_id)
            if old is not None and not old.same_as(record):
                raise ValueError(
                    f"fit id {record.fit_id!r} is already registered, for model {old.label} in stage {old.stage!r} "
                    f"with excluded {_folds_text(old.excluded_folds)}; every fit of a Run needs its own fit id "
                    "(fit_id_for builds one from the model name and the FitSpec)"
                )
            new.setdefault(record.fit_id, record)
        for fit_id, record in new.items():
            self._records.setdefault(fit_id, record)

    # -- reading ---------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[FitRecord]:
        return iter(tuple(self._records.values()))

    def __contains__(self, fit_id: object) -> bool:
        return isinstance(fit_id, str) and fit_id in self._records

    def get(self, fit_id: str) -> FitRecord:
        """The record of a registered fit; UnknownFitError (a KeyError) for an unknown id."""
        try:
            return self._records[fit_id]
        except (KeyError, TypeError):
            raise UnknownFitError(f"no fit with id {fit_id!r} is in the FitRegistry") from None

    def excluded_folds(self, fit_id: str) -> frozenset[int]:
        """The excluded folds of a registered fit (for views.check_view); UnknownFitError when unknown."""
        return self.get(fit_id).excluded_folds

    def records(self, stage: str | None = None) -> tuple[FitRecord, ...]:
        """The records in registration order: all of them, or those that ``stage`` registered."""
        stage = _check_stage(stage)
        return tuple(record for record in self._records.values() if stage is None or record.stage == stage)

    # -- tables ----------------------------------------------------------------------------

    def to_frame(self, stage: str | None = None) -> pd.DataFrame:
        """One row per fit (all, or those ``stage`` registered), with FIT_REGISTRY_COLUMNS."""
        records = self.records(stage)
        return pd.DataFrame(
            {
                "model": pd.array([r.model for r in records], dtype="str"),
                "family": pd.array([r.family for r in records], dtype="str"),
                "fit_id": pd.array([r.fit_id for r in records], dtype="str"),
                "sandbox_k": pd.array([r.sandbox_k for r in records], dtype="Int32"),
                "inner_j": pd.array([r.inner_j for r in records], dtype="Int32"),
                "excluded_folds": pd.Series([sorted(r.excluded_folds) for r in records], dtype=object),
                "n_train_rows": pd.array([r.n_train_rows for r in records], dtype="Int64"),
                "n_train_matches": np.array([len(r.train_match_ids) for r in records], dtype=np.int64),
                "n_early_stopping_matches": np.array([len(r.early_stopping_match_ids) for r in records], dtype=np.int64),
                "n_predicted_matches": np.array([len(r.predicted_match_ids) for r in records], dtype=np.int64),
                "stage": pd.array([r.stage for r in records], dtype="str"),
            }
        )

    def train_matches_frame(self, stage: str | None = None) -> pd.DataFrame:
        """``fit_train_matches``: one row per match of each fit's Training_Rows (TRAIN_MATCH_COLUMNS)."""
        records = self.records(stage)
        matches, flags = [], []
        for record in records:
            ids = np.concatenate([record.train_match_ids, record.early_stopping_match_ids])
            early = np.repeat([False, True], [len(record.train_match_ids), len(record.early_stopping_match_ids)])
            order = np.argsort(ids, kind="stable")
            matches.append(ids[order])
            flags.append(early[order])
        return pd.DataFrame(
            {
                "fit_id": _repeat_ids(records, [len(ids) for ids in matches]),
                "match_id": np.concatenate([np.zeros(0, dtype=np.int64), *matches]),
                "early_stopping": np.concatenate([np.zeros(0, dtype=bool), *flags]),
            }
        )

    def predicted_matches_frame(self, stage: str | None = None) -> pd.DataFrame:
        """``fit_predicted_matches``: one row per predicted match of each fit (PREDICTED_MATCH_COLUMNS)."""
        records = self.records(stage)
        return pd.DataFrame(
            {
                "fit_id": _repeat_ids(records, [len(r.predicted_match_ids) for r in records]),
                "match_id": np.concatenate([np.zeros(0, dtype=np.int64), *(r.predicted_match_ids for r in records)]),
            }
        )

    # -- checkpoints -----------------------------------------------------------------------

    def save(self, store: CheckpointStore, stage: str) -> int:
        """Save the fits that ``stage`` registered to its checkpoint; return how many.

        Writes FIT_TRAIN_MATCHES, FIT_PREDICTED_MATCHES and then FIT_REGISTRY
        (``fit_registry.parquet``) with ``store.save_table``. A model stage calls this after its last
        ``register``; saving again replaces the tables. Raises ValueError when another stage is
        running, so a stage cannot write into another stage's checkpoint.
        """
        stage = _check_stage(stage)
        if stage is None:
            raise ValueError("save needs the name of the Pipeline_Stage whose checkpoint receives the fits")
        if self._stage is not None and stage != self._stage:
            raise ValueError(f"stage {self._stage!r} is running, so it cannot save the fits of stage {stage!r}")
        store.save_table(stage, FIT_TRAIN_MATCHES, self.train_matches_frame(stage))
        store.save_table(stage, FIT_PREDICTED_MATCHES, self.predicted_matches_frame(stage))
        store.save_table(stage, FIT_REGISTRY, self.to_frame(stage))
        return len(self.records(stage))

    def load_saved(self, store: CheckpointStore, stage: str) -> int:
        """Add the fits that an earlier ``stage`` saved with ``save``; return how many (0 when it saved none).

        Each fit keeps its stage tag and is checked again against the Fold_Table. Raises the
        StageError of ``store.load_table`` for a missing, changed or unreadable table, the errors
        of ``records_from_frames`` for a malformed one, FitLeakageError when a saved fit breaks a
        rule of the guard, and ValueError when a saved fit id is already registered with other
        values. Nothing is added when it raises.
        """
        stage = _check_stage(stage)
        if stage is None:
            raise ValueError("load_saved needs the name of the Pipeline_Stage that saved the fits")
        if not (store.stage_dir(stage) / FIT_REGISTRY_FILE).is_file():
            return 0
        records = records_from_frames(
            store.load_table(stage, FIT_REGISTRY),
            store.load_table(stage, FIT_TRAIN_MATCHES),
            store.load_table(stage, FIT_PREDICTED_MATCHES),
            stage=stage,
        )
        for record in records:
            self._check(record, saved_by=stage)
        self._add(records)
        return len(records)


def _repeat_ids(records: tuple[FitRecord, ...], counts: list[int]) -> Any:
    ids = np.array([record.fit_id for record in records], dtype=object)
    return pd.array(np.repeat(ids, np.asarray(counts, dtype=np.int64)), dtype="str")


# ------------------------------------------------------------------------- tables to records

def _require_columns(frame: Any, columns: tuple[str, ...], table: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{table} must be a DataFrame; got {type(frame).__name__}")
    absent = [column for column in columns if column not in frame.columns]
    if absent:
        raise ValueError(f"{table} lacks the column(s) {', '.join(absent)}")


def _cell_missing(value: Any) -> bool:
    return value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value))


def _cell_text(value: Any) -> str | None:
    return None if _cell_missing(value) else value


def _cell_int(value: Any, what: str) -> int | None:
    if _cell_missing(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return _whole(value, what)


def records_from_frames(
    fits: pd.DataFrame,
    train_matches: pd.DataFrame,
    predicted_matches: pd.DataFrame,
    *,
    stage: str | None = None,
) -> list[FitRecord]:
    """The FitRecords of the three tables that ``FitRegistry.save`` writes, in table order.

    With ``stage``, every fit must carry that stage tag. The records are not checked against a
    Fold_Table here (``FitRegistry.load_saved`` does that). Raises ValueError or TypeError,
    naming the problem, when a table lacks a column, a fit id is repeated, missing or not listed
    in ``fits``, a value is malformed or a count column disagrees with the side tables, and
    DataError for a malformed match id.
    """
    _require_columns(fits, FIT_REGISTRY_COLUMNS, FIT_REGISTRY)
    _require_columns(train_matches, TRAIN_MATCH_COLUMNS, FIT_TRAIN_MATCHES)
    _require_columns(predicted_matches, PREDICTED_MATCH_COLUMNS, FIT_PREDICTED_MATCHES)
    expected_stage = _check_stage(stage)
    fit_ids = [_cell_text(value) for value in fits["fit_id"].tolist()]
    if any(not isinstance(fit_id, str) for fit_id in fit_ids):
        raise ValueError(f"{FIT_REGISTRY} has a missing or non-text fit_id")
    counts = pd.Series(fit_ids, dtype=object).value_counts()
    repeated = counts.index[(counts > 1).to_numpy()].tolist()
    if repeated:
        raise ValueError(f"{FIT_REGISTRY} repeats the fit id(s) {', '.join(map(repr, repeated[:_SHOWN]))}")
    known = set(fit_ids)
    context = "The saved fit registry cannot be read"

    def grouped(frame: pd.DataFrame, table: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
        ids = frame["fit_id"].tolist()
        unknown = sorted({str(value) for value in ids if not isinstance(value, str) or value not in known})
        if unknown:
            raise ValueError(f"{table} names fit id(s) that {FIT_REGISTRY} does not list: {', '.join(unknown[:_SHOWN])}")
        positions = pd.Series(np.arange(len(frame)), dtype=np.int64).groupby(np.asarray(ids, dtype=object), sort=False).indices
        return positions, _whole_numbers(frame["match_id"], f"{table} match_id", context)

    train_at, train_ids = grouped(train_matches, FIT_TRAIN_MATCHES)
    early_column = train_matches["early_stopping"]
    if not pd.api.types.is_bool_dtype(early_column.dtype) or early_column.isna().any():
        raise ValueError(f"{FIT_TRAIN_MATCHES} early_stopping must be a bool column without missing values")
    early_flags = early_column.to_numpy(dtype=bool)
    predicted_at, predicted_ids = grouped(predicted_matches, FIT_PREDICTED_MATCHES)

    columns = {name: fits[name].tolist() for name in FIT_REGISTRY_COLUMNS}
    records = []
    for row, fit_id in enumerate(fit_ids):
        saved_stage = _cell_text(columns["stage"][row])
        if expected_stage is not None and saved_stage != expected_stage:
            raise ValueError(f"fit {fit_id!r} in the fit registry of stage {expected_stage!r} is tagged with stage {saved_stage!r}")
        at = train_at.get(fit_id, np.zeros(0, dtype=np.int64))
        early = early_flags[at]
        record = _build_record(
            model=_cell_text(columns["model"][row]),
            family=_cell_text(columns["family"][row]),
            fit_id=fit_id,
            excluded_folds=columns["excluded_folds"][row],
            train_match_ids=train_ids[at][~early],
            early_stopping_match_ids=train_ids[at][early],
            predicted_match_ids=predicted_ids[predicted_at.get(fit_id, np.zeros(0, dtype=np.int64))],
            sandbox_k=_cell_int(columns["sandbox_k"][row], "sandbox_k"),
            inner_j=_cell_int(columns["inner_j"][row], "inner_j"),
            n_train_rows=_cell_int(columns["n_train_rows"][row], "n_train_rows"),
            stage=saved_stage,
        )
        for name, actual in (
            ("n_train_matches", len(record.train_match_ids)),
            ("n_early_stopping_matches", len(record.early_stopping_match_ids)),
            ("n_predicted_matches", len(record.predicted_match_ids)),
        ):
            if _cell_int(columns[name][row], name) != actual:
                raise ValueError(
                    f"fit {fit_id!r}: {FIT_REGISTRY} gives {name} = {columns[name][row]!r}, but the side tables "
                    f"hold {actual:,} distinct match(es)"
                )
        records.append(record)
    return records


# ------------------------------------------------------------------------------ fit ids, splits

def fit_id_for(model: str, spec: FitSpec | str, *, family: str | None = None) -> str:
    """The Run-wide fit id of a sandbox fit: ``<model>_<family>_<spec fit id>``, family left out when None.

    ``spec`` is a ``folds.FitSpec`` (or its ``fit_id``), so ``fit_id_for("m3_inn1", spec,
    family="lgbm")`` gives ``"m3_inn1_lgbm_inner_k1_j2"`` for spec ``inner_k1_j2``. Raises
    ValueError when the result is not a valid fit id, for example when ``model`` holds a space.
    """
    base = spec if isinstance(spec, str) else getattr(spec, "fit_id", None)
    if not isinstance(base, str):
        raise TypeError(f"spec must be a folds.FitSpec or its fit_id; got {spec!r}")
    parts = [_check_text(model, "model")] + ([] if family is None else [_check_text(family, "family")]) + [base]
    return _check_fit_id("_".join(parts))


def early_stopping_split(
    train_match_ids: Any, share: float, seed: int, fit_index: int
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out whole matches of one fit's training matches for early stopping (Requirement 10, criterion 7).

    train_match_ids: the fit's training matches, as distinct ids or as a per-Delivery
    ``match_id`` column (repeats are ignored). share: the model's ``es_share``, above 0 and
    below 1. seed: the model's named seed. fit_index: the fit's position in
    ``folds.sandbox_plan(D15)``. Both are whole numbers of 0 or more.

    The distinct ids, ascending, are permuted by ``numpy.random.Generator(PCG64(seed +
    fit_index))``, and the first ``share`` x n of them (rounded half up, at least 1 and at most
    n - 1) become the validation matches. Returns ``(fit_match_ids, validation_match_ids)``:
    distinct, ascending int64 arrays that split the distinct training matches in two, the same
    for the same inputs in any order.

    Raises DataError (exit code 3) for a missing or non-whole id, TypeError or ValueError for a
    malformed share, seed or index, and ValueError for fewer than 2 distinct training matches.
    """
    ids = np.unique(_whole_numbers(train_match_ids, "train_match_ids", "The early-stopping split cannot be drawn"))
    if isinstance(share, (bool, np.bool_)) or not isinstance(share, Real):
        raise TypeError(f"share (es_share) must be a number; got {share!r}")
    share = float(share)
    if not 0.0 < share < 1.0:
        raise ValueError(f"share (es_share) must lie above 0 and below 1; got {share}")
    seed, fit_index = _whole(seed, "seed"), _whole(fit_index, "fit_index")
    if seed < 0 or fit_index < 0:
        raise ValueError(f"seed and fit_index must be 0 or more; got seed {seed}, fit_index {fit_index}")
    n = len(ids)
    if n < 2:
        raise ValueError(
            f"early stopping needs at least 2 distinct training matches, one to fit on and one to validate on; got {n}"
        )
    n_validation = min(max(math.floor(share * n + 0.5), 1), n - 1)
    rng = np.random.Generator(np.random.PCG64(seed + fit_index))
    held_out = np.zeros(n, dtype=bool)
    held_out[rng.permutation(n)[:n_validation]] = True
    return ids[~held_out], ids[held_out]
