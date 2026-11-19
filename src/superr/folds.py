"""Fold_Assigner: the Match_Grouped_Folds, the Fold_Table and the fold-sandbox plan (Requirement 7,
criteria 1-5 and 10; design sections "Fold sandboxes and leakage control", "Fold_Assigner, views
and leakage guard" and Data Models, ``fold_table``).

Every function receives the Config values it needs as arguments: no function reads ``config.py``
or has a default value for a Config setting. The number of folds D15
(``design_choices.D15_n_folds``) and the fold seed (``seeds.folds``) are keyword arguments of
``assign_folds``, so two whole numbers cannot be swapped by position.

The Fold_Table (``assign_folds``)
    One row per match of ``matches.csv``, sorted by ``match_id``, on a RangeIndex, with the
    FOLD_TABLE_COLUMNS: ``match_id`` (int32), ``comp_type`` (str: ``IPL``, ``T20I`` or ``A_Team``)
    and ``fold`` (int32, 1 to D15). Every match gets exactly one fold, including a match with no
    Delivery (criterion 1). The assignment:

    1. sort the matches by ``match_id``, so the table does not depend on the row order of
       ``matches.csv``;
    2. for each Competition_Type in the fixed order ``reconstruct.COMPETITION_TYPES`` (IPL, T20I,
       A_Team), permute that type's matches with one ``numpy.random.Generator(PCG64(seed))``,
       which draws one permutation per type in that order;
    3. deal the permuted matches round robin: the i-th match of a type (counting from 0) gets fold
       ``(offset + i) mod D15 + 1``. ``offset`` starts at 0 and, after each type, moves on by that
       type's match count (mod D15).

    Within each Competition_Type the fold sizes therefore differ by at most 1 (criterion 3). The
    offset carries over from type to type, so the fold totals also differ by at most 1. The same
    seed and the same matches give the same table (criterion 4).

    ``match_comp_types`` gives each match the Competition_Type of its competition with
    ``reconstruct.competition_types``, the rule behind the ``comp_type`` column of ``balls``.

Delivery folds (criteria 2 and 5)
    A Delivery's fold is its match's fold, read from the Fold_Table and from nowhere else. A later
    stage lists ``"folds"`` in its ``Stage.reads`` and:

    * loads the table with ``ctx.store.load_table(folds.STAGE_NAME, folds.FOLD_TABLE)``;
    * adds the ``fold`` column (int32) to a per-Delivery frame that has ``match_id``, such as the
      ``balls`` table of the ``reconstruct`` checkpoint, with ``attach_folds(balls, fold_table)``;
    * adds it to a frame keyed by ``row`` alone, such as the ``state`` table, with
      ``attach_folds(state, fold_table, balls=balls)``, which takes each row's ``match_id`` from
      ``balls``;
    * or takes the fold of any array of match ids, in its order, with
      ``fold_of(match_ids, fold_table)``, for example the ``fold_of_row`` array of a sandbox view.

    These raise DataError (exit code 3) when a match id is missing or not in the Fold_Table, so no
    Delivery goes without a fold. Only the ``folds`` stage calls ``assign_folds``.

The fold-sandbox plan (``sandbox_plan``)
    For each target fold k, the sandbox S_k is the folds other than k. The plan holds one FitSpec
    for each fit that a sandboxed model makes, D15 x D15 specs in all (25 with D15 = 5):

    * D15 outer fits, ``outer_k<k>``: exclude fold k, train on the folds of S_k and give their
      output to the Deliveries of fold k (the Out_Of_Fold_Predictions, and the fold-k rows of
      view V_k);
    * D15 x (D15 - 1) inner fits, ``inner_k<k>_j<j>``, one for each ordered pair (k, j) with
      j != k: exclude folds j and k, train on the other folds and give their output to the
      fold-j rows inside S_k (the fold-j rows of view V_k, or the fold-j predictions behind the
      fold-k isotonic calibrator).

    The outer specs come first, by k, then the inner specs by (k, j). Each FitSpec holds a stable
    ``fit_id`` (unique within a plan), ``sandbox_k``, ``inner_j`` (None for an outer fit), the
    ``excluded_folds`` that the leakage guard checks ({k} or {j, k}) and the ascending
    ``train_folds``. The pairs (k, j) and (j, k) exclude the same folds but are separate fits,
    because a fit inside sandbox k takes its model-derived features from view V_k. The plan depends
    on D15 alone, so a model stage calls ``sandbox_plan(D15)`` itself. A model that needs no inner
    fits (M4, the Extras_Model) uses the outer specs only.

The ``folds`` Pipeline_Stage (``run_stage``)
    Reads ``matches`` and ``competitions`` from the ``load`` checkpoint and ``balls`` from the
    ``reconstruct`` checkpoint, builds the Fold_Table, checks that every Delivery of ``balls`` gets
    the fold of its match, writes ``outputs/folds/fold_table.csv`` (criterion 10) and saves the
    table as ``fold_table`` in its checkpoint.

A match id that is missing, not a whole number, repeated or outside the int32 range, and a match
whose competition is unknown or has no Competition_Type, stop the Run with a DataError that names
the problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from superr import reconstruct
from superr.errors import DataError

if TYPE_CHECKING:
    from superr.pipeline import StageContext

__all__ = [
    "FOLDS_FOLDER",
    "FOLD_COLUMN",
    "FOLD_TABLE",
    "FOLD_TABLE_COLUMNS",
    "FOLD_TABLE_FILE",
    "INNER",
    "MIN_FOLDS",
    "MIN_SANDBOX_FOLDS",
    "OUTER",
    "STAGE_NAME",
    "FitSpec",
    "assign_folds",
    "attach_folds",
    "fold_of",
    "match_comp_types",
    "run_stage",
    "sandbox_plan",
    "write_fold_table",
]


# ------------------------------------------------------------------------------------- constants

#: Name of the Pipeline_Stage that runs the Fold_Assigner, of its checkpoint and of its table.
STAGE_NAME = "folds"
FOLD_TABLE = "fold_table"

#: Folder under the outputs folder that holds the Fold_Table file (criterion 10), and its name.
FOLDS_FOLDER = "folds"
FOLD_TABLE_FILE = "fold_table.csv"

#: The fold column that attach_folds adds, and the columns of the Fold_Table, in order.
FOLD_COLUMN = "fold"
FOLD_TABLE_COLUMNS: tuple[str, ...] = ("match_id", "comp_type", FOLD_COLUMN)

#: The fewest folds assign_folds accepts: a fit must be able to hold one fold out.
MIN_FOLDS = 2
#: The fewest folds sandbox_plan accepts: an inner fit excludes two folds and trains on the rest.
MIN_SANDBOX_FOLDS = 3

#: The two kinds of FitSpec (FitSpec.kind).
OUTER = "outer"
INNER = "inner"

_LOAD_STAGE = "load"
_MATCHES_TABLE = "matches"  # the load checkpoint's tables of matches.csv and competitions.csv
_COMPETITIONS_TABLE = "competitions"
_INT32_MIN, _INT32_MAX = int(np.iinfo(np.int32).min), int(np.iinfo(np.int32).max)
_SHOWN = 5  # match ids named in one error message


# ---------------------------------------------------------------------------------- shared helpers

def _rows(count: int) -> str:
    return f"{count:,} row" if count == 1 else f"{count:,} rows"


def _shown_ids(ids: np.ndarray) -> str:
    return ", ".join(map(str, ids[:_SHOWN].tolist()))


def _whole_values(values: Any, name: str, context: str) -> np.ndarray:
    """``values`` (a Series or 1-D array-like) as int64 whole numbers.

    Raises DataError (exit code 3), whose message starts with ``context``, when a value is missing
    or is not a finite whole number.
    """
    series = values if isinstance(values, pd.Series) else pd.Series(values)
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
        problems.append(f"{name} is missing in {_rows(int(missing.sum()))}")
    bad = ~missing & ~whole
    if bad.any():
        examples = ", ".join(repr(value) for value in series.iloc[np.flatnonzero(bad)[:3]].tolist())
        problems.append(f"{name} is not a whole number in {_rows(int(bad.sum()))} (for example {examples})")
    if problems:
        raise DataError("\n".join([f"{context}:"] + [f"  - {problem}" for problem in problems]))
    return out


def _match_ids(matches: pd.DataFrame, context: str) -> np.ndarray:
    """The ``match_id`` of every row of ``matches`` (int64); DataError when one is missing, repeated or not int32."""
    if not isinstance(matches, pd.DataFrame):
        raise TypeError(f"matches must be a DataFrame; got {type(matches).__name__}")
    if "match_id" not in matches.columns:
        raise ValueError("matches lacks the column match_id")
    ids = _whole_values(matches["match_id"], "match_id", context)
    problems = []
    unique, counts = np.unique(ids, return_counts=True)
    repeated = unique[counts > 1]
    if len(repeated):
        problems.append(
            f"match_id is repeated in matches for {len(repeated):,} match(es) (for example {_shown_ids(repeated)})"
        )
    outside = (ids < _INT32_MIN) | (ids > _INT32_MAX)
    if outside.any():
        problems.append(f"match_id is outside the int32 range in {_rows(int(outside.sum()))}")
    if problems:
        raise DataError("\n".join([f"{context}:"] + [f"  - {problem}" for problem in problems]))
    return ids


def _check_n_folds(n_folds: Any, minimum: int, reason: str) -> int:
    if isinstance(n_folds, bool) or not isinstance(n_folds, Integral):
        raise TypeError(f"n_folds (D15) must be a whole number; got {n_folds!r}")
    if n_folds < minimum:
        raise ValueError(f"n_folds (D15) must be at least {minimum}, {reason}; got {n_folds}")
    return int(n_folds)


def _check_seed(seed: Any) -> int:
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError(f"seed (seeds.folds) must be a whole number; got {seed!r}")
    if seed < 0:
        raise ValueError(f"seed (seeds.folds) must be 0 or more; got {seed}")
    return int(seed)


def _comp_type_values(comp_type: Any, index: pd.Index) -> np.ndarray:
    """The Competition_Type of each match as an object array of str, in the row order of ``matches``.

    A Series must be on the index of ``matches``; any other array-like is taken in row order.
    Raises ValueError for a misaligned input or a value that is not a Competition_Type.
    """
    if isinstance(comp_type, pd.Series):
        if not comp_type.index.equals(index):
            raise ValueError("comp_type must be a Series on the index of matches, or an array in its row order")
        series = comp_type
    else:
        array = np.asarray(comp_type, dtype=object)
        if array.ndim != 1 or len(array) != len(index):
            raise ValueError(f"comp_type must hold one value per match ({len(index):,}); got shape {array.shape}")
        series = pd.Series(array, index=index, dtype=object)
    known = series.isin(reconstruct.COMPETITION_TYPES).to_numpy(dtype=bool)
    if not known.all():
        others = pd.unique(series.iloc[np.flatnonzero(~known)].to_numpy(dtype=object))
        shown = ", ".join(repr(value) for value in others[:_SHOWN].tolist())
        raise ValueError(
            f"comp_type must be one of {', '.join(reconstruct.COMPETITION_TYPES)} for every match; it holds "
            f"another value or none in {_rows(int((~known).sum()))} ({shown})"
        )
    return series.to_numpy(dtype=object)


# ------------------------------------------------------------------------------------ Fold_Table

def match_comp_types(matches: pd.DataFrame, competitions: pd.DataFrame) -> pd.Series:
    """The Competition_Type of every match (glossary): IPL, T20I or A_Team.

    ``matches`` needs ``comp_id`` (and ``match_id`` for error messages); ``competitions`` needs
    ``comp_id``, ``comp_name`` and ``match_format``. Each match gets the type that
    ``reconstruct.competition_types`` gives its competition.

    Returns a ``str`` Series named ``comp_type`` on the index of ``matches``, in its row order.
    Raises DataError (exit code 3) when a ``comp_id`` is missing or not a whole number, is not in
    ``competitions``, or names a competition with no Competition_Type, and for the problems that
    competition_types reports.
    """
    context = "The Competition_Types of the matches cannot be set"
    if not isinstance(matches, pd.DataFrame) or not isinstance(competitions, pd.DataFrame):
        raise TypeError("matches and competitions must be DataFrames")
    if "comp_id" not in matches.columns:
        raise ValueError("matches lacks the column comp_id")
    comp = _whole_values(matches["comp_id"], "comp_id", context)
    types = reconstruct.competition_types(competitions)
    position = pd.Index(types.index.to_numpy(dtype=np.int64)).get_indexer(comp)
    unknown = position < 0
    if unknown.any():
        shown = ", ".join(
            f"match {m}: comp_id {c}"
            for m, c in zip(matches["match_id"].iloc[np.flatnonzero(unknown)[:3]].tolist(), comp[unknown][:3].tolist())
        )
        raise DataError(
            f"{context}:\n  - the comp_id of {_rows(int(unknown.sum()))} of matches is not in competitions "
            f"(for example {shown})"
        )
    values = types.to_numpy(dtype=object, na_value=None)[position]
    untyped = pd.isna(values)
    if untyped.any():
        comps = np.unique(comp[untyped])
        raise DataError(
            f"{context}:\n  - no Competition_Type applies to {len(comps):,} competition(s) used by "
            f"{_rows(int(untyped.sum()))} of matches (comp_id {_shown_ids(comps)})"
        )
    return pd.Series(pd.array(values, dtype="str"), index=matches.index, name="comp_type")


def assign_folds(matches: pd.DataFrame, comp_type: Any, *, n_folds: int, seed: int) -> pd.DataFrame:
    """The Fold_Table: every match of ``matches`` in one of ``n_folds`` Match_Grouped_Folds (criteria 1, 3 and 4).

    ``matches`` needs ``match_id``; every row is a match, whether or not it has Deliveries.
    ``comp_type`` is the Competition_Type of each row (match_comp_types): a Series on the index of
    ``matches`` or an array-like in its row order. ``n_folds`` is D15
    (``design_choices.D15_n_folds``) and ``seed`` the fold seed (``seeds.folds``). The algorithm is
    in the module docstring.

    Returns one row per match, sorted by ``match_id``, on a RangeIndex, with ``match_id`` (int32),
    ``comp_type`` (str) and ``fold`` (int32, 1 to ``n_folds``). Raises DataError (exit code 3) when
    a ``match_id`` is missing, not a whole number, repeated or outside the int32 range; ValueError
    when ``comp_type`` is misaligned or holds a value other than IPL, T20I or A_Team, or when
    ``n_folds`` is below MIN_FOLDS or ``seed`` is negative; TypeError when either is not a whole
    number.
    """
    n_folds = _check_n_folds(n_folds, MIN_FOLDS, "so that a fit can hold a fold out")
    seed = _check_seed(seed)
    ids = _match_ids(matches, "The Fold_Table cannot be built")
    types = _comp_type_values(comp_type, matches.index)

    order = np.argsort(ids, kind="stable")
    ids, types = ids[order], types[order]
    fold = np.zeros(len(ids), dtype=np.int32)
    rng = np.random.Generator(np.random.PCG64(seed))
    offset = 0
    for name in reconstruct.COMPETITION_TYPES:
        members = np.flatnonzero(types == name)  # this type's matches, in match_id order
        dealt = members[rng.permutation(len(members))]
        fold[dealt] = (offset + np.arange(len(members))) % n_folds + 1
        offset = (offset + len(members)) % n_folds
    return pd.DataFrame(
        {
            "match_id": ids.astype(np.int32),
            "comp_type": pd.array(types, dtype="str"),
            FOLD_COLUMN: fold,
        }
    )


def _fold_lookup(fold_table: pd.DataFrame) -> tuple[pd.Index, np.ndarray]:
    """The Fold_Table's match ids as a unique Index and its folds as an int32 array, in table order."""
    if not isinstance(fold_table, pd.DataFrame):
        raise TypeError(f"fold_table must be a DataFrame; got {type(fold_table).__name__}")
    absent = [column for column in ("match_id", FOLD_COLUMN) if column not in fold_table.columns]
    if absent:
        raise ValueError(f"fold_table lacks the column(s) {', '.join(absent)}; pass the Fold_Table of assign_folds")
    context = "The Fold_Table cannot be used"
    ids = pd.Index(_whole_values(fold_table["match_id"], "Fold_Table match_id", context))
    if not ids.is_unique:
        raise ValueError("fold_table must hold each match_id once; pass the Fold_Table of assign_folds")
    folds = _whole_values(fold_table[FOLD_COLUMN], "Fold_Table fold", context)
    if len(folds) and folds.min() < 1:
        raise ValueError("fold_table holds a fold below 1; folds are numbered from 1")
    return ids, folds.astype(np.int32)


def fold_of(match_ids: Any, fold_table: pd.DataFrame) -> np.ndarray:
    """The fold of each match id in ``match_ids``, from the Fold_Table (criteria 2 and 5).

    ``match_ids`` is a Series or 1-D array-like, for example the ``match_id`` column of a
    per-Delivery table; ``fold_table`` is the Fold_Table (assign_folds, or the ``fold_table`` of the
    ``folds`` checkpoint). Returns an int32 array of the same length, in the same order.

    Raises DataError (exit code 3) when a match id is missing, not a whole number or not in the
    Fold_Table, and ValueError when ``fold_table`` lacks ``match_id`` or ``fold`` or repeats a match.
    """
    table_ids, table_folds = _fold_lookup(fold_table)
    context = "Deliveries cannot get the fold of their match"
    ids = _whole_values(match_ids, "match_id", context)
    at = table_ids.get_indexer(ids)
    absent = at < 0
    if absent.any():
        unknown = np.unique(ids[absent])
        raise DataError(
            f"{context}:\n  - match_id is not in the Fold_Table in {_rows(int(absent.sum()))} "
            f"({len(unknown):,} match(es), for example {_shown_ids(unknown)})"
        )
    return table_folds[at]


def attach_folds(frame: pd.DataFrame, fold_table: pd.DataFrame, *, balls: pd.DataFrame | None = None) -> pd.DataFrame:
    """A copy of ``frame`` with the ``fold`` column (int32): each row's match's fold (criteria 2 and 5).

    When ``frame`` has a ``match_id`` column, each row's match is read from it and ``balls`` is not
    used. Otherwise ``frame`` needs a ``row`` column and ``balls`` (the ``balls`` table of the
    ``reconstruct`` checkpoint, with ``row`` and ``match_id``) supplies each row's ``match_id``; this
    serves the ``state`` table and other tables keyed by ``row``. The other columns, the row order
    and the index of ``frame`` are kept.

    Raises the errors of fold_of, and ValueError when ``frame`` already has a ``fold`` column (folds
    come from the Fold_Table only), has neither ``match_id`` nor (with ``balls``) ``row``, or holds
    a ``row`` that is not in ``balls``.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"frame must be a DataFrame; got {type(frame).__name__}")
    if FOLD_COLUMN in frame.columns:
        raise ValueError(
            f"frame already has a {FOLD_COLUMN} column; drop it first, so that every fold comes from the Fold_Table"
        )
    if "match_id" in frame.columns:
        match_ids: Any = frame["match_id"]
    elif balls is None:
        raise ValueError("frame has no match_id column; pass balls= to look up each row's match_id by row")
    else:
        if not isinstance(balls, pd.DataFrame):
            raise TypeError(f"balls must be a DataFrame; got {type(balls).__name__}")
        absent = [column for column in ("row", "match_id") if column not in balls.columns]
        if absent:
            raise ValueError(f"balls lacks the column(s) {', '.join(absent)}")
        if "row" not in frame.columns:
            raise ValueError("frame has neither a match_id nor a row column")
        context = "The match of each row cannot be found in balls"
        index = pd.Index(_whole_values(balls["row"], "balls row", context))
        if not index.is_unique:
            raise ValueError("balls must hold each row once")
        at = index.get_indexer(_whole_values(frame["row"], "row", context))
        if (at < 0).any():
            raise ValueError(f"{_rows(int((at < 0).sum()))} of frame have no Delivery in balls")
        match_ids = balls["match_id"].to_numpy()[at]
    return frame.assign(**{FOLD_COLUMN: fold_of(match_ids, fold_table)})


def write_fold_table(outdir: str | Path, fold_table: pd.DataFrame) -> Path:
    """Write ``fold_table`` to FOLD_TABLE_FILE in ``outdir`` (criterion 10); return the path.

    ``outdir`` is the folds folder (``outputs/folds/`` in a Run) and is created when absent. The
    file has a header row and one row per match with the FOLD_TABLE_COLUMNS, in UTF-8 with ``\\n``
    line ends. It is written through a temporary file renamed in place, so it replaces any earlier
    copy whole.
    """
    if not isinstance(fold_table, pd.DataFrame):
        raise TypeError(f"fold_table must be a DataFrame; got {type(fold_table).__name__}")
    absent = [column for column in FOLD_TABLE_COLUMNS if column not in fold_table.columns]
    if absent:
        raise ValueError(f"fold_table lacks the column(s) {', '.join(absent)}")
    folder = Path(outdir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / FOLD_TABLE_FILE
    temporary = path.with_name(path.name + ".tmp")
    try:
        fold_table[list(FOLD_TABLE_COLUMNS)].to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


# -------------------------------------------------------------------------------- sandbox plan

@dataclass(frozen=True)
class FitSpec:
    """One fit of the fold-sandbox scheme, as sandbox_plan lists it (module docstring).

    * ``fit_id``: ``outer_k<k>`` or ``inner_k<k>_j<j>``, stable and unique within a plan. A model
      stage combines it with the model's name where it needs an id that is unique in the Run.
    * ``sandbox_k``: the target fold k whose Deliveries the fit's output serves, directly or as a
      feature or calibration input of another fit.
    * ``inner_j``: None for an outer fit; the inner fold j (not k) for an inner fit.
    * ``excluded_folds``: {k} for an outer fit and {j, k} for an inner fit. No Training_Row of the
      fit may come from a match in these folds.
    * ``train_folds``: every other fold, ascending: the folds whose matches the fit may train on.
    """

    fit_id: str
    sandbox_k: int
    inner_j: int | None
    excluded_folds: frozenset[int]
    train_folds: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "excluded_folds", frozenset(self.excluded_folds))
        object.__setattr__(self, "train_folds", tuple(self.train_folds))
        expected = {self.sandbox_k} if self.inner_j is None else {self.sandbox_k, self.inner_j}
        problems = []
        if self.inner_j is not None and self.inner_j == self.sandbox_k:
            problems.append("inner_j must differ from sandbox_k")
        if min(expected) < 1:
            problems.append("folds are numbered from 1")
        if self.excluded_folds != expected:
            problems.append(f"excluded_folds must be {sorted(expected)}")
        train = list(self.train_folds)
        if not train or train != sorted(set(train)) or min(train) < 1:
            problems.append("train_folds must be ascending, distinct folds numbered from 1, and not empty")
        if self.excluded_folds & set(train):
            problems.append("train_folds must not hold an excluded fold")
        if problems:
            raise ValueError(f"FitSpec {self.fit_id!r} is not valid: " + "; ".join(problems))

    @property
    def kind(self) -> str:
        """OUTER for an outer fit, INNER for an inner fit."""
        return OUTER if self.inner_j is None else INNER

    @property
    def predicted_fold(self) -> int:
        """The fold whose rows get the fit's output: k for an outer fit, j for an inner fit."""
        return self.sandbox_k if self.inner_j is None else self.inner_j


def sandbox_plan(n_folds: int) -> list[FitSpec]:
    """The FitSpec of every fit of a sandboxed model: ``n_folds`` outer, then ``n_folds`` x (``n_folds`` - 1) inner.

    ``n_folds`` is D15 (``design_choices.D15_n_folds``); with 5 folds the plan holds 5 outer and 20
    inner specs. Outer spec k excludes fold k; inner spec (k, j), one for each ordered pair with
    j != k, excludes folds j and k. Each spec trains on every other fold. The outer specs come
    first, by k, then the inner specs by (k, j).

    Raises ValueError when ``n_folds`` is below MIN_SANDBOX_FOLDS and TypeError when it is not a
    whole number.
    """
    n_folds = _check_n_folds(
        n_folds, MIN_SANDBOX_FOLDS, "because an inner fit excludes two folds and trains on the rest"
    )
    folds = range(1, n_folds + 1)
    outer = [
        FitSpec(
            fit_id=f"{OUTER}_k{k}",
            sandbox_k=k,
            inner_j=None,
            excluded_folds=frozenset({k}),
            train_folds=tuple(f for f in folds if f != k),
        )
        for k in folds
    ]
    inner = [
        FitSpec(
            fit_id=f"{INNER}_k{k}_j{j}",
            sandbox_k=k,
            inner_j=j,
            excluded_folds=frozenset({j, k}),
            train_folds=tuple(f for f in folds if f not in (j, k)),
        )
        for k in folds
        for j in folds
        if j != k
    ]
    return outer + inner


# ---------------------------------------------------------------------------- the folds stage

def _per_fold(folds: np.ndarray, n_folds: int) -> str:
    """Counts of ``folds`` in folds 1 to ``n_folds``, as ``a/b/c/d/e``."""
    counts = np.bincount(np.asarray(folds, dtype=np.int64), minlength=n_folds + 1)[1:]
    return "/".join(f"{count:,}" for count in counts.tolist())


def _shown(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def run_stage(ctx: StageContext) -> None:
    """The ``folds`` Pipeline_Stage: assign every match to a Match_Grouped_Fold and save the Fold_Table.

    Reads ``matches`` and ``competitions`` from the ``load`` checkpoint, ``balls`` from the
    ``reconstruct`` checkpoint, D15 (``design_choices.D15_n_folds``) and the fold seed
    (``seeds.folds``). Builds the Fold_Table with match_comp_types and assign_folds over every
    match of ``matches.csv`` (criteria 1, 3 and 4), takes the fold of every Delivery of ``balls``
    from it (criterion 2), writes ``outputs/folds/fold_table.csv`` (criterion 10) and saves the
    table as FOLD_TABLE. The Run_Log gets the matches of each Competition_Type in each fold, the
    matches with no Delivery, the Deliveries in each fold and the size of the fold-sandbox plan.

    Raises DataError (exit code 3) for the data problems of match_comp_types, assign_folds and
    fold_of, and ValueError or TypeError when D15 is below MIN_SANDBOX_FOLDS or the seed is not a
    whole number of 0 or more.
    """
    settings, store, log = ctx.settings, ctx.store, ctx.log
    n_folds = settings.get("design_choices.D15_n_folds")
    seed = settings.get("seeds.folds")
    plan = sandbox_plan(n_folds)  # checks D15 before any table is read
    matches = store.load_table(_LOAD_STAGE, _MATCHES_TABLE)
    competitions = store.load_table(_LOAD_STAGE, _COMPETITIONS_TABLE)
    balls = store.load_table(reconstruct.STAGE_NAME, reconstruct.BALLS_TABLE)

    table = assign_folds(matches, match_comp_types(matches, competitions), n_folds=n_folds, seed=seed)
    delivery_folds = fold_of(balls["match_id"], table)
    path = write_fold_table(settings.path("outputs") / FOLDS_FOLDER, table)
    store.save_table(STAGE_NAME, FOLD_TABLE, table)

    folds = table[FOLD_COLUMN].to_numpy()
    types = table["comp_type"].to_numpy(dtype=object)
    per_type = [
        f"{name} {int((types == name).sum()):,} ({_per_fold(folds[types == name], n_folds)})"
        for name in reconstruct.COMPETITION_TYPES
    ]
    log.info(
        f"assigned {len(table):,} matches to {n_folds} Match_Grouped_Folds (D15) with fold seed {seed} "
        f"(seeds.folds), stratified by Competition_Type (Requirement 7, criteria 1, 3 and 4); matches in "
        f"folds 1-{n_folds}: {'; '.join(per_type)}; all {len(table):,} ({_per_fold(folds, n_folds)})",
        stage=STAGE_NAME,
    )
    no_balls = table.loc[~np.isin(table["match_id"].to_numpy(), balls["match_id"].to_numpy())]
    if len(no_balls):
        pairs = zip(no_balls["match_id"].tolist(), no_balls[FOLD_COLUMN].tolist())
        shown = ", ".join(f"{match} (fold {fold})" for match, fold in pairs)
        log.info(
            f"{len(no_balls):,} matches have no Delivery and get a fold like every other match "
            f"(Requirement 7, criterion 1): {shown}",
            stage=STAGE_NAME,
        )
    log.info(
        f"every one of the {len(balls):,} Deliveries in balls gets the fold of its match from the Fold_Table "
        f"(Requirement 7, criterion 2); Deliveries in folds 1-{n_folds}: {_per_fold(delivery_folds, n_folds)}",
        stage=STAGE_NAME,
    )
    n_outer = sum(spec.kind == OUTER for spec in plan)
    log.info(
        f"fold-sandbox plan of each sandboxed model: {len(plan)} fits, {n_outer} outer (sandbox k, excluding "
        f"fold k) and {len(plan) - n_outer} inner (sandbox k and inner fold j, excluding folds j and k)",
        stage=STAGE_NAME,
    )
    log.info(
        f"wrote {_shown(path, ctx.root)} ({len(table):,} matches; Requirement 7, criterion 10) and saved the "
        f"{FOLD_TABLE} table",
        stage=STAGE_NAME,
    )
