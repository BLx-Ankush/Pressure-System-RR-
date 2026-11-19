# Feature: pressure-rating-system, Property 43: Per-fold statistics use Training_Folds only
"""Property 43: per-fold statistics use Training_Folds only (Requirement 9, criterion 5;
Requirement 12, criterion 3; Requirement 15, criteria 2, 6, 15, 20 and 21).

**Validates: Requirements 9.5, 12.3, 15.2, 15.6, 15.15, 15.20, 15.21**

Every statistic that serves the Deliveries of fold k must come from the rows of the other folds,
fold k's Training_Folds. Each test draws a Delivery table with D15 folds (5 as delivered) and a
fold k, replaces every fold-k row with the fold-k rows of a second draw, builds the statistic from
both tables with the function that its stage calls, and requires the values served to fold k to
be identical: floats bit for bit, NaN in the same places, and counts, fill codes and pooled flags
equal. An example whose second draw repeats fold k's rows is discarded (``assume``), so the
fold-k rows always change.

1. Extras means (15.6, 15.20): ``value.extras_cell_means``. Compared: the six (k, cell) rows,
   which ``cell_lookup`` serves to the fold-k Deliveries of each cell: the Wide and No_Ball means,
   pooled where the cell lacks the Deliveries, the cell's own counts and the pooled flags.
2. Free_Hit_Premium (15.15, 15.21): ``value.free_hit_premium``. Compared: the six (k, cell) rows,
   with the pooled premium and flag.
3. Class_Run_Values (15.2): ``outcome.outcome_rows`` and ``outcome.fit_outcome``, the outcome
   stage's own path. Compared: fold k's row of ``class_run_values_frame``, the
   ``m4_class_run_values`` table that the value stage reads back for every fold-k Delivery.
4. F20(a) expected wickets by Legal_Ball count (12.3): ``pressure.train_wicket_means``. Compared:
   the (k, innings, L) rows of the table, and the means that ``wicket_means_for`` serves at every
   (innings, L) of the grid.
5. M2 lookup baseline (9.5): ``remaining_runs.lookup_baseline``, the table of held-out fold k built
   from the other folds with its empty-cell fill rules. Compared: ``pred`` and ``fill`` at every
   (Legal_Balls left, wickets) of the grid.

The examples
    Cell tables (tests 1 and 2). ``tests.strategies.cell_tables(pooled=True)`` gives the table.
    k is drawn from the folds whose Training_Folds lack, in some cell, a Delivery of a kind that the
    table under test averages (its ``emptied`` triples), so fold k is served at least one pooled
    value; when no fold lacks one, k is drawn from every fold of ``emptied``. The replacement rows
    are the fold-k rows of a second ``cell_tables`` draw with the same folds and cell column, so
    the Delivery classes, cells, bat runs, team runs (another Extras_Encoding), Free_Hit flags and
    dropped rows (excluded-match and Super_Over_Innings rows) all change. The two draws never share
    a match id. With D15 folds every kind is held by at least 3 folds, so every pooled set of
    every fold stays non-empty after the replacement and both tables can be built.

    Class_Run_Values (test 3). ``crv_cases`` draws per fold 1-2 matches of 1-5 Deliveries, Legal_Balls
    most often but also Wides and No_Balls, and 0-3 Deliveries of an excluded match. A Delivery
    ends as a dot or run ball, an other-runs ball (5 or 7 bat runs), a Bowler_Wicket (0-1 bat runs)
    or a Run_Out (0-3 bat runs), so the mean of every class of MEAN_VALUE_CLASSES varies with the
    rows behind it. ``outcome_rows`` labels the Legal_Balls with ``label_outcomes`` and marks the
    Training_Row candidates (``common.legal_training_mask``). Each class of MEAN_VALUE_CLASSES also
    gets one Training_Row candidate in each of 3 or more folds, so every fit keeps a row of every
    class after one fold is replaced. ``fit_outcome`` runs the D15 outer fits of a class-prior
    family (``DummyClassifier``) under a FitRegistry, the leakage guard included; the family does
    not matter, since a fit's Class_Run_Values come from its Training_Rows before its model is fitted.
    The replacement rows are the fold-k rows of a second draw: other labels, bat runs, classes,
    Free_Hit flags and excluded matches.

    F20(a) and the M2 lookup baseline (tests 4 and 5). Each fold has 1-3 matches of 1-6 (F20(a))
    or 1-4 (M2) Deliveries on a small grid of keys, so the folds share keys and leave some empty:
    innings 1-2 x L 0-8 for F20(a), Legal_Balls left 0-10 x wickets 0-4 for M2. A match is
    sometimes excluded (``training`` false; for M2 ``train`` false); an M2 row sometimes has no
    target, and then is not a ``train`` row either, while the first row of match 0 of each fold
    always is, as ``lookup_baseline`` needs one outside every fold. The replacement rows are the
    fold-k rows of a second draw: other keys, wickets, targets and flags. Probe rows give the
    values served to fold k at every key of the grid: in both tables, fold k also holds a probe
    match, excluded from training, with one Delivery at each key, whose wickets (F20(a)) or target
    (M2: none before, a drawn one after) are redrawn in the second table, so every fold-k row
    changes. The probes put every key in both F20(a) tables, so the two (k, innings, L) parts hold
    the same keys, and they make ``lookup_baseline`` read the held-out-k table at every key,
    through its three fill rules: the cell mean, the mean of the same Legal_Balls left (criterion
    5) and the mean of the nearest Legal_Balls left with rows.

Hypothesis events record which pooled values and fill rules fold k was served, and whether the
values of the other folds, which train on fold k, changed, so ``--hypothesis-show-statistics``
shows that the replacement had an effect. No event is asserted.

When the tests were written, a version of each of the five functions that also averages fold k's
own rows (each row copied into one extra fold, which every real fold then trains on) failed its
test.

Not covered here: the values of the other folds, which change with the fold-k rows as they should;
the pooled values and the fallback counts themselves (Property 44); and the Training_Rows of the
model fits and the sandbox views (Properties 19 and 20).
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from hypothesis import assume, event, given, note
from hypothesis import strategies as st
from sklearn.dummy import DummyClassifier

from superr.folds import FOLD_COLUMN, sandbox_plan
from superr.leakage import FitRegistry
from superr.models.common import estimator_fitter
from superr.models.outcome import (
    BOWLER_WICKET,
    MEAN_VALUE_CLASSES,
    N_CLASSES,
    OTHER_RUNS,
    RUN_OUT,
    AverageSkill,
    OutcomeFamily,
    class_run_values_frame,
    fit_outcome,
    outcome_rows,
)
from superr.models.remaining_runs import LOOKUP_NEAREST_LB_LEFT, LOOKUP_SAME_LB_LEFT, lookup_baseline
from superr.pressure import train_wicket_means, wicket_means_for
from superr.reconstruct import DeliveryClass
from superr.state import STATE_INNINGS
from superr.value import POOLED_QUANTITIES, extras_cell_means, free_hit_premium
from tests.strategies import KIND_FREE_HIT, KIND_NO_BALL, KIND_NON_FREE_HIT, KIND_WIDE, N_FOLDS, cell_tables

#: The folds of every example: D15. The tests need 3 or more, so that every pooled set and every
#: class of MEAN_VALUE_CLASSES keeps a holder outside a target fold when another fold's rows change.
FOLDS: tuple[int, ...] = tuple(range(1, N_FOLDS + 1))

#: The match index j of fold k's probe match in tests 4 and 5 (the drawn matches use 0-2).
PROBE_MATCH = 3


class FoldCase(NamedTuple):
    """One example: a table before and after every fold-k row is replaced."""

    k: int  # the target fold
    before: pd.DataFrame
    after: pd.DataFrame


# ---------------------------------------------------------------------------------- shared helpers

def _match_id(fold: int, j: int) -> int:
    """The id of match j of ``fold``: 1 + j x D15 + (fold - 1), as cell_tables numbers them, so folds share none."""
    return 1 + j * N_FOLDS + (fold - 1)


def _frame(records: Sequence[tuple[Any, ...]], columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame.from_records(list(records), columns=list(columns))


def _combine(*parts: pd.DataFrame) -> pd.DataFrame:
    """The rows of ``parts`` sorted by match (stable: each match keeps its order), ``row`` renumbered if present."""
    out = pd.concat(parts, ignore_index=True).sort_values("match_id", kind="stable", ignore_index=True)
    if "row" in out.columns:
        out["row"] = np.arange(len(out), dtype=out["row"].dtype)
    return out


def _replace_fold(frame: pd.DataFrame, donor: pd.DataFrame, k: int, *extra: pd.DataFrame) -> pd.DataFrame:
    """``frame`` with every fold-k row replaced by the fold-k rows of ``donor``, plus the rows of ``extra``."""
    return _combine(frame[frame[FOLD_COLUMN] != k], donor[donor[FOLD_COLUMN] == k], *extra)


def _fold_part(table: pd.DataFrame, k: int, *, served: bool = True) -> pd.DataFrame:
    """The rows of ``table`` in fold k (``served``) or in the other folds, on a RangeIndex."""
    return table[(table[FOLD_COLUMN] == k) == served].reset_index(drop=True)


def _assert_same(before: pd.DataFrame, after: pd.DataFrame, what: str) -> None:
    """Equal values, dtypes and NaN positions; floats compared bit for bit."""
    pd.testing.assert_frame_equal(before, after, check_exact=True, obj=what)


def _assume_changed(frame: pd.DataFrame, donor: pd.DataFrame, k: int) -> None:
    """Discard the example unless the donor's fold-k rows differ from those of ``frame`` (``row`` ids aside)."""
    own, new = (_fold_part(table, k).drop(columns="row", errors="ignore") for table in (frame, donor))
    assume(not own.equals(new))


def _others_event(k: int, before: pd.DataFrame, after: pd.DataFrame, what: str) -> None:
    """Did the replacement change the values of the other folds, which train on fold k?"""
    same = _fold_part(before, k, served=False).equals(_fold_part(after, k, served=False))
    event(f"{what} of the other folds {'unchanged' if same else 'changed'}")


# ------------------------------------------------------------------------- 1, 2: the cell tables

#: The kinds of CELL_KINDS that each cell table averages.
EXTRAS_KINDS: tuple[str, ...] = (KIND_WIDE, KIND_NO_BALL)
LEGAL_KINDS: tuple[str, ...] = (KIND_FREE_HIT, KIND_NON_FREE_HIT)


class CellCase(NamedTuple):
    """One example of tests 1 and 2: the FoldCase of a cell_tables table and the name of its cell column."""

    case: FoldCase
    cells: str


@st.composite
def cell_cases(draw: Any, kinds: Sequence[str]) -> CellCase:
    """A cell_tables(pooled=True) table, a fold k served a pooled value of ``kinds``, and k's rows replaced."""
    table = draw(cell_tables(pooled=True))
    lacking = sorted({fold for fold, _, kind in table.emptied if kind in kinds})
    k = draw(st.sampled_from(lacking or sorted({fold for fold, _, _ in table.emptied})))
    donor = draw(cell_tables(n_folds=table.n_folds, cells=table.cells))
    _assume_changed(table.deliveries, donor.deliveries, k)
    after = _replace_fold(table.deliveries, donor.deliveries, k)
    return CellCase(FoldCase(k, table.deliveries, after), table.cells)


def _check_cell_table(example: CellCase, build: Callable[[pd.DataFrame, str], pd.DataFrame]) -> None:
    case = example.case
    before, after = build(case.before, example.cells), build(case.after, example.cells)
    served = _fold_part(before, case.k)
    note(f"target fold k = {case.k}; cell column {example.cells!r}; fold-k rows of the table before:\n{served}")
    for quantity in POOLED_QUANTITIES:
        if quantity.flag in served.columns and served[quantity.flag].any():
            event(f"fold k is served the pooled {quantity.label}")
    _others_event(case.k, before, after, "the cell rows")
    _assert_same(served, _fold_part(after, case.k), f"the fold-{case.k} rows of {build.__name__}")


@given(example=cell_cases(EXTRAS_KINDS))
def test_extras_cell_means_of_fold_k_ignore_fold_k_rows(example: CellCase) -> None:
    """The Wide and No_Ball means of fold k's six cells, pooled or not, come from the other folds only."""
    _check_cell_table(example, extras_cell_means)


@given(example=cell_cases(LEGAL_KINDS))
def test_free_hit_premium_of_fold_k_ignores_fold_k_rows(example: CellCase) -> None:
    """The Free_Hit_Premium of fold k's six cells, pooled or not, comes from the other folds' Legal_Balls only."""
    _check_cell_table(example, free_hit_premium)


# ----------------------------------------------------------------------- 3: the Class_Run_Values

_LEGAL, _WIDE, _NO_BALL = int(DeliveryClass.LEGAL), int(DeliveryClass.WIDE), int(DeliveryClass.NO_BALL)
#: The Delivery classes drawn, Legal_Balls most often; a Wide or No_Ball is never a Training_Row.
_CRV_CLASSES: tuple[int, ...] = (_LEGAL, _LEGAL, _LEGAL, _LEGAL, _WIDE, _NO_BALL)
#: How a Delivery ends: (bowler_wicket, run_out, bat runs drawn), by the class label_outcomes gives
#: a Legal_Ball. _RUN_BALL stands for the dot and run classes, whose bat runs fix the class.
_RUN_BALL = -1
_ENDINGS: dict[int, tuple[bool, bool, tuple[int, ...]]] = {
    _RUN_BALL: (False, False, (0, 1, 2, 3, 4, 6)),
    OTHER_RUNS: (False, False, (5, 7)),
    BOWLER_WICKET: (True, False, (0, 1)),
    RUN_OUT: (False, True, (0, 1, 2, 3)),
}
_ENDING_DRAWS: tuple[int, ...] = (_RUN_BALL, _RUN_BALL, _RUN_BALL, OTHER_RUNS, BOWLER_WICKET, RUN_OUT)
_CRV_EXCLUDED_MATCH = 2  # the match index j of a fold's excluded match (regular matches use 0-1)
#: The columns of the crv_cases frames: the ``balls`` columns that outcome_rows reads, the fold
#: (for the replacement; outcome_rows takes it from the Fold_Table) and the ``state`` columns.
CRV_BALL_COLUMNS: tuple[str, ...] = (
    "row", "match_id", "innings", "cls", "runs", "bowler_wicket", "run_out", "free_hit", "pace_or_spin",
    "is_super_over", "excluded",
)  # fmt: skip
CRV_STATE_COLUMNS: tuple[str, ...] = ("row", "phase", "wickets")
_CRV_COLUMNS: tuple[str, ...] = (*CRV_BALL_COLUMNS, FOLD_COLUMN, "phase", "wickets")
#: The one feature column of the class-prior fits, and the Average_Skill of every fold (the
#: delivered sizes; fit_outcome only checks and stores it).
_FEATURE = "x"
_AVERAGE_SKILL = {k: AverageSkill(theta_bat=1.51, theta_bowl=0.04, m1_fit=f"m1_outer_k{k}") for k in FOLDS}


@st.composite
def _crv_ball(draw: Any, ending: int | None = None) -> tuple[Any, ...]:
    """innings, cls, bat runs, bowler_wicket, run_out, free_hit, pace_or_spin; ``ending`` forces such a Legal_Ball."""
    cls = _LEGAL if ending is not None else draw(st.sampled_from(_CRV_CLASSES))
    bowler_wicket, run_out, runs = _ENDINGS[ending if ending is not None else draw(st.sampled_from(_ENDING_DRAWS))]
    bat = draw(st.sampled_from(runs))
    innings = draw(st.sampled_from(STATE_INNINGS))
    free_hit = draw(st.sampled_from((False, False, False, True)))
    pace = draw(st.sampled_from(("Pace", "Spin")))
    return innings, cls, 0 if cls == _WIDE else bat, bowler_wicket, run_out, free_hit, pace


@st.composite
def _crv_rows(draw: Any) -> pd.DataFrame:
    records: list[tuple[Any, ...]] = []

    def add(fold: int, j: int, ball: tuple[Any, ...], excluded: bool) -> None:
        innings, cls, bat, bowler_wicket, run_out, free_hit, pace = ball
        records.append((0, _match_id(fold, j), innings, cls, bat, bowler_wicket, run_out, free_hit, pace, False,
                        excluded, fold, 0, 0))  # fmt: skip

    for fold in FOLDS:
        for j in range(draw(st.integers(1, 2))):
            for _ in range(draw(st.integers(1, 5))):
                add(fold, j, draw(_crv_ball()), excluded=False)
        for _ in range(draw(st.integers(0, 3))):  # the fold's excluded match, whose rows train no fit
            add(fold, _CRV_EXCLUDED_MATCH, draw(_crv_ball()), excluded=True)
    for ending in MEAN_VALUE_CLASSES:  # 3 or more folds hold a Training_Row candidate of each such class
        for fold in draw(st.lists(st.sampled_from(FOLDS), min_size=3, max_size=len(FOLDS), unique=True)):
            add(fold, 0, draw(_crv_ball(ending)), excluded=False)
    return _combine(_frame(records, _CRV_COLUMNS))


@st.composite
def crv_cases(draw: Any) -> FoldCase:
    """A Delivery table, a fold k, and the table with k's rows replaced by those of a second draw."""
    base, donor = draw(_crv_rows()), draw(_crv_rows())
    k = draw(st.sampled_from(FOLDS))
    _assume_changed(base, donor, k)
    return FoldCase(k, base, _replace_fold(base, donor, k))


def _prior_classifier() -> DummyClassifier:
    return DummyClassifier(strategy="prior")


def _prior_predict(model: DummyClassifier, X: pd.DataFrame) -> np.ndarray:
    """The class priors of ``model`` for every row of ``X``, on the N_CLASSES (0 for a class it never saw)."""
    probs = np.zeros((len(X), N_CLASSES))
    probs[:, np.asarray(model.classes_, dtype=np.int64)] = model.predict_proba(X)
    return probs


#: A class-prior M4 family: the cheapest real family, fitted through common.estimator_fitter.
PRIOR_FAMILY = OutcomeFamily(
    name="prior", early_stopping=None, fitter=functools.partial(estimator_fitter, _prior_classifier),
    predict=_prior_predict,
)  # fmt: skip


def _class_run_values(frame: pd.DataFrame) -> pd.DataFrame:
    """The ``m4_class_run_values`` table of ``frame``: the outer fit of every fold, as the outcome stage runs them."""
    pairs = frame[["match_id", FOLD_COLUMN]].drop_duplicates().sort_values("match_id", ignore_index=True)
    fold_table = pd.DataFrame({"match_id": pairs["match_id"], "comp_type": "IPL", FOLD_COLUMN: pairs[FOLD_COLUMN]})
    rows = outcome_rows(frame[list(CRV_BALL_COLUMNS)], frame[list(CRV_STATE_COLUMNS)], fold_table)
    features = pd.DataFrame({_FEATURE: np.zeros(rows.n_rows)})
    models = fit_outcome(
        FitRegistry(fold_table), sandbox_plan(N_FOLDS), PRIOR_FAMILY, rows, dict.fromkeys(FOLDS, features),
        _AVERAGE_SKILL, feature_names=(_FEATURE,),
    )  # fmt: skip
    return class_run_values_frame(models)


@given(case=crv_cases())
def test_class_run_values_of_fold_k_ignore_fold_k_rows(case: FoldCase) -> None:
    """Fold k's Class_Run_Values and class counts come from the Legal_Balls of the other folds only."""
    before, after = _class_run_values(case.before), _class_run_values(case.after)
    note(f"target fold k = {case.k}; Class_Run_Values before:\n{before}")
    _others_event(case.k, before, after, "the Class_Run_Values")
    _assert_same(_fold_part(before, case.k), _fold_part(after, case.k), f"fold {case.k}'s Class_Run_Values")


# ------------------------------------------------------- 4, 5: F20(a) and the M2 lookup baseline

#: F20(a) keys: innings 1-2 x pre-ball Legal_Balls bowled L 0 to LAST_LB_BOWLED; wickets 0 to 9.
LAST_LB_BOWLED = 8
WICKET_GRID: tuple[tuple[int, int], ...] = tuple(
    (innings, lb) for innings in STATE_INNINGS for lb in range(LAST_LB_BOWLED + 1)
)
_WICKET_COLUMNS = ("match_id", FOLD_COLUMN, "innings", "lb_bowled", "wickets", "training", "probe")

#: M2 lookup keys: Legal_Balls left 0 to LAST_LB_LEFT x wickets fallen 0 to LAST_WICKETS.
LAST_LB_LEFT, LAST_WICKETS = 10, 4
LOOKUP_GRID: tuple[tuple[int, int], ...] = tuple(
    (lb, wk) for lb in range(LAST_LB_LEFT + 1) for wk in range(LAST_WICKETS + 1)
)
_LOOKUP_COLUMNS = ("match_id", FOLD_COLUMN, "lb_left", "wickets", "target", "train", "probe")


def _target(draw: Any, lb_left: int) -> float:
    """An M2 training target for a Delivery with ``lb_left`` Legal_Balls left: whole runs, at most 3 per ball + 6."""
    return float(draw(st.integers(0, 3 * lb_left + 6)))


@st.composite
def _wicket_rows(draw: Any) -> pd.DataFrame:
    records = []
    for fold in FOLDS:
        for j in range(draw(st.integers(1, 3))):
            training = draw(st.sampled_from((True, True, True, False)))  # False: an excluded match
            for _ in range(draw(st.integers(1, 6))):
                innings = draw(st.sampled_from(STATE_INNINGS))
                lb = draw(st.integers(0, LAST_LB_BOWLED))
                records.append((_match_id(fold, j), fold, innings, lb, draw(st.integers(0, 9)), training, False))
    return _frame(records, _WICKET_COLUMNS)


def _wicket_probes(draw: Any, k: int) -> pd.DataFrame:
    match = _match_id(k, PROBE_MATCH)
    records = [(match, k, innings, lb, draw(st.integers(0, 9)), False, True) for innings, lb in WICKET_GRID]
    return _frame(records, _WICKET_COLUMNS)


@st.composite
def wicket_cases(draw: Any) -> FoldCase:
    """``state`` rows with folds and training flags, a fold k, and the rows with k's replaced; probes in both."""
    base, donor = draw(_wicket_rows()), draw(_wicket_rows())
    k = draw(st.sampled_from(FOLDS))
    _assume_changed(base, donor, k)
    return FoldCase(k, _combine(base, _wicket_probes(draw, k)), _replace_fold(base, donor, k, _wicket_probes(draw, k)))


def _wicket_means(frame: pd.DataFrame) -> pd.DataFrame:
    return train_wicket_means(
        fold=frame[FOLD_COLUMN].to_numpy(), innings=frame["innings"].to_numpy(),
        lb_bowled=frame["lb_bowled"].to_numpy(), wickets=frame["wickets"].to_numpy(),
        training=frame["training"].to_numpy(),
    )  # fmt: skip


def _served_wickets(table: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """The F20(a) mean that ``wicket_means_for`` serves each probe: its innings, L and mean."""
    probes = frame[frame["probe"]].reset_index(drop=True)
    mean = wicket_means_for(
        table, fold=probes[FOLD_COLUMN].to_numpy(), innings=probes["innings"].to_numpy(),
        lb_bowled=probes["lb_bowled"].to_numpy(),
    )  # fmt: skip
    return probes[["innings", "lb_bowled"]].assign(mean_wickets=mean)


@given(case=wicket_cases())
def test_f20a_mean_wickets_of_fold_k_ignore_fold_k_rows(case: FoldCase) -> None:
    """The mean wickets by innings and L that F20(a) subtracts for fold k come from the other folds only."""
    before, after = _wicket_means(case.before), _wicket_means(case.after)
    served = _served_wickets(before, case.before)
    note(f"target fold k = {case.k}; means served to fold k before:\n{served}")
    if served["mean_wickets"].isna().any():
        event("fold k is served a missing mean (no Training_Folds Delivery at that innings and L)")
    _others_event(case.k, before, after, "the F20(a) means")
    _assert_same(_fold_part(before, case.k), _fold_part(after, case.k), f"the fold-{case.k} rows of train_wicket_means")
    _assert_same(served, _served_wickets(after, case.after), f"the F20(a) means served to fold {case.k}")


@st.composite
def _lookup_rows(draw: Any) -> pd.DataFrame:
    records = []
    for fold in FOLDS:
        for j in range(draw(st.integers(1, 3))):
            excluded = j > 0 and draw(st.sampled_from((False, False, True)))  # match 0 always trains
            for i in range(draw(st.integers(1, 4))):
                lb, wk = draw(st.integers(0, LAST_LB_LEFT)), draw(st.integers(0, LAST_WICKETS))
                known = (j == 0 and i == 0) or draw(st.sampled_from((True, True, True, False)))  # False: no target
                target = _target(draw, lb) if known else np.nan
                records.append((_match_id(fold, j), fold, lb, wk, target, known and not excluded, False))
    return _frame(records, _LOOKUP_COLUMNS)


def _lookup_probes(draw: Any, k: int, *, with_targets: bool) -> pd.DataFrame:
    match = _match_id(k, PROBE_MATCH)
    records = [
        (match, k, lb, wk, _target(draw, lb) if with_targets else np.nan, False, True) for lb, wk in LOOKUP_GRID
    ]
    return _frame(records, _LOOKUP_COLUMNS)


@st.composite
def lookup_cases(draw: Any) -> FoldCase:
    """M2 rows with folds, targets and ``train`` flags, a fold k, and the rows with k's replaced; probes in both."""
    base, donor = draw(_lookup_rows()), draw(_lookup_rows())
    k = draw(st.sampled_from(FOLDS))
    _assume_changed(base, donor, k)
    before = _combine(base, _lookup_probes(draw, k, with_targets=False))
    return FoldCase(k, before, _replace_fold(base, donor, k, _lookup_probes(draw, k, with_targets=True)))


def _lookup(frame: pd.DataFrame) -> pd.DataFrame:
    """Every row's key with its lookup-baseline ``pred`` and ``fill`` (the table of its held-out fold)."""
    table = lookup_baseline(
        frame["lb_left"].to_numpy(), frame["wickets"].to_numpy(), frame["target"].to_numpy(),
        frame[FOLD_COLUMN].to_numpy(), frame["train"].to_numpy(),
    )  # fmt: skip
    return pd.concat([frame[[FOLD_COLUMN, "lb_left", "wickets", "probe"]], table], axis=1)


@given(case=lookup_cases())
def test_m2_lookup_baseline_of_fold_k_ignores_fold_k_rows(case: FoldCase) -> None:
    """The lookup table of held-out fold k, empty-cell fills included, comes from the other folds only."""
    before, after = _lookup(case.before), _lookup(case.after)
    served = before[before["probe"]].reset_index(drop=True)
    note(f"target fold k = {case.k}; lookup baseline served to fold k before:\n{served}")
    for fill, rule in ((LOOKUP_SAME_LB_LEFT, "the same Legal_Balls left"), (LOOKUP_NEAREST_LB_LEFT, "the nearest one")):
        if (served["fill"] == fill).any():
            event(f"fold k is served the fill rule of {rule}")
    _others_event(case.k, before[~before["probe"]], after[~after["probe"]], "the predictions")
    _assert_same(served, after[after["probe"]].reset_index(drop=True), f"the lookup baseline of held-out fold {case.k}")
