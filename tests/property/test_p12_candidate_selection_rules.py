# Feature: pressure-rating-system, Property 12: Candidate selection rules
"""Property 12: Candidate selection rules (Requirement 5, criterion 7; Requirement 23, criteria 7, 8
and 12).

**Validates: Requirements 5.7, 23.7, 23.8, 23.12**

Property 12 (design): for any table of candidate scores, the selected Extras_Encoding has the
highest mean of its five Exact_Pass_Rates, preferring F30_Encoding when it ties for the highest and
otherwise the first-listed tied candidate; the selected M3 Primary_Model of each innings and the
selected M4 Primary_Model have the lowest Pooled_Held_Out_Log_Loss among non-failed candidates,
with ties broken in the documented family order, except that a Config-named preferred family (M4's
is LightGBM) is selected when it passed and its log loss is within a relative band of the lowest
(the calibration-aware tie-break of Requirement 23, criterion 8); and any candidate with a fit
failure, a non-finite prediction, an M3 probability outside [0, 1] or an M4 probability row whose
sum differs from 1 by more than 1e-6 is never selected.

Every expected result is worked out in this file from the drawn inputs. The F30_Encoding's name,
the model labels, the family tie orders and both tolerances are written out here (design DD-4 and
the ``model_comparison`` data model; Requirement 23, criteria 7, 8 and 12), not imported from the
code under test. Two values tie when they differ by at most max(1e-9 x the larger magnitude, 1e-9),
the rule that the docstrings of ``reconstruct.select_encoding`` and ``models.model_choice`` state.

* ``test_select_encoding_takes_the_highest_mean_then_the_f30_encoding_then_the_first_listed``: a
  pass-rate table of the Config's candidate Extras_Encodings in a drawn order. Each Score_Check
  applies to the same 0-30 matches under every candidate (applicability does not depend on the
  encoding), and each candidate passes a drawn number of them; a rate is 100 x passes / applicable
  as a float64, missing when the check applies to no match. A drawn leader, and each other
  candidate that joins it (about 1 in 3), holds the highest mean: their pass counts come from one
  vector through moves that keep the exact mean (n1/g more passes on one check and n2/g fewer on
  another, g = gcd(n1, n2)). In about half the examples each tied candidate moves, where it can, to
  a float mean that no earlier tied candidate has, so tied means differ in the last digit as 275/6
  does. The other candidates lose 1 or more passes, sometimes with one pass added back on a check
  that applies to more matches. The reference compares the means exactly, as fractions of the
  counts. With at most 30 matches per check, two different means differ by more than 1.5e-6, which
  is more than 10 times the tie tolerance (at most 1e-7 for rates up to 100); the test checks this,
  so a tie depends on the fractions alone. Two explicit examples have tied means of 275/6 that come
  out as 45.83333333333333 and 45.833333333333336: the F30_Encoding, listed second, holds the lower
  float mean and is selected; and with the F30_Encoding below, the first-listed tied candidate
  holds the lower float mean and is selected.
* ``test_select_primary_takes_the_lowest_log_loss_of_a_passing_candidate_in_tie_order``: a
  comparison table with rows of M3-1, M3-2 and M4 in a drawn order. Each model has 1 to all of its
  families and a base log loss from 0.05 to 3, so both the absolute arm (up to 1) and the relative
  arm (above 1) of the tolerance apply. In about one model in eight every candidate has a failure
  reason. Otherwise one candidate passes at the base value and each other one passes or fails at
  random. A passing candidate ties exactly, ties within 0.45 x the tolerance (above or below), lies
  1.1-4 x the tolerance above (just outside) or lies 1e-6 to 0.5 above. A failed candidate has no
  log loss, the lowest log loss of its model, a tied one or a higher one. The expected
  Primary_Model is the first candidate in the model's tie order among those drawn as tied for the
  lowest log loss; the test checks that these ties lie within 0.95 x the tolerance and every other
  passing log loss more than 1.05 x it above, far beyond float rounding. ``select_primary`` must
  return it for each model, with "" or a missing value marking a candidate that passed, and must
  raise StageError when every candidate failed. ``candidate_table`` must order the rows by model
  and tie order and flag exactly the expected candidates as ``primary`` (or raise StageError), and
  ``primary_models`` must read them back. Explicit examples put ties and gaps of 1.15e-9 on each
  arm of the tolerance, and fail every M4 candidate.
* ``test_failure_checks_flag_exactly_the_bad_predictions_and_failed_candidates_are_not_selected``:
  M3 arrays (``wp_raw`` and ``wp_pre`` of shape (n,), ``wp_post`` of shape (n, 9)) of values in
  [0, 1], including 0.0, -0.0 and 1.0, with 0-3 values replaced by NaN, +-inf or a value outside
  [0, 1] (down to -5e-324 and 1 + 2.2e-16); and 1-6 M4 variant arrays of rows of 9 probabilities
  summing to 1 (one-hot rows included), with 0-3 defects: a non-finite cell, a cell moved below 0
  with the row sum kept, or a row sum moved by 1e-9 to 0.98e-6 (the row passes) or by 1.02e-6 to
  1e-2 (the row fails). The reference counts the bad values of each array in plain Python, with
  the row sums as exact fractions. ``m3_failure`` and ``m4_failure`` must return "" exactly when no
  value is bad, and otherwise a reason that names the first condition met (non-finite, then outside
  [0, 1], then the sum of 1) and exactly the arrays that break it. A candidate with that reason and
  the lowest log loss, first in the tie order, must then not be selected; one without must be.
* ``test_select_primary_prefers_a_calibration_family_within_the_band`` and the two example tests
  beside it: with an M4 ``preferred`` mapping of ``(family, band)``, ``select_primary`` and
  ``candidate_table`` return the preferred family when it passed and its log loss is within
  ``band`` x the lowest (relative), and the lowest in tie order otherwise or when the preferred
  family failed; a model with no ``preferred`` entry (M3) keeps the plain lowest-log-loss rule.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import pytest
from hypothesis import event, example, given, note
from hypothesis import strategies as st

from superr.errors import StageError
from superr.models.model_choice import (
    CandidateResult,
    candidate_table,
    m3_failure,
    m4_failure,
    primary_models,
    select_primary,
)
from superr.reconstruct import select_encoding
from tests.strategies import CANDIDATE_ENCODINGS, CHECKS

# ------------------------------------------------------------------------------------ constants

#: Two values tie when they differ by at most max(TIE_REL_TOL x the larger magnitude, TIE_ABS_TOL).
TIE_REL_TOL = 1e-9
TIE_ABS_TOL = 1e-9

#: Requirement 23, criterion 12: an M4 row whose class probabilities differ from a sum of 1 by more
#: than this fails.
PROB_SUM_TOL = Fraction(1, 10**6)

#: The F30_Encoding (E1 of design DD-4) and the Exact_Pass_Rate column of each Score_Check, in
#: table order.
F30 = "F30"
EXACT_COLUMNS: tuple[str, ...] = tuple(f"exact_{check}" for check in CHECKS)

#: The ``model`` labels of the comparison table in its row order, and each model's families in tie
#: order: Requirement 23, criterion 7 (logistic baseline, LightGBM, XGBoost, random forest) and
#: criterion 8 (LightGBM, XGBoost, random forest).
MODELS: tuple[str, ...] = ("M3-1", "M3-2", "M4")
_M3_FAMILIES: tuple[str, ...] = ("logistic", "lgbm", "xgb", "rf")
FAMILY_ORDER: Mapping[str, tuple[str, ...]] = {
    "M3-1": _M3_FAMILIES,
    "M3-2": _M3_FAMILIES,
    "M4": ("lgbm", "xgb", "rf"),
}

#: The M3 prediction arrays and the M4 variants that the failure checks read, and the M4 classes.
M3_ARRAYS: tuple[str, ...] = ("wp_raw", "wp_pre", "wp_post")
M4_VARIANTS: tuple[str, ...] = (
    "actual", "avg_both", "batter_view", "batter_view_fh", "bowler_view", "bowler_view_nofh",
)  # fmt: skip
N_CLASSES = 9

if F30 not in CANDIDATE_ENCODINGS or len(CANDIDATE_ENCODINGS) < 3:
    raise RuntimeError(
        f"data_rules.candidate_encodings must list {F30} and at least 2 other candidates for these "
        f"examples; got {CANDIDATE_ENCODINGS}"
    )


def _tolerance(value: float) -> float:
    """The tie tolerance at ``value``: max(TIE_REL_TOL x |value|, TIE_ABS_TOL)."""
    return max(TIE_REL_TOL * abs(value), TIE_ABS_TOL)


class _Table:
    """A DataFrame for hypothesis.note, formatted only when Hypothesis prints the note."""

    def __init__(self, title: str, frame: pd.DataFrame) -> None:
        self.title, self.frame = title, frame

    def __repr__(self) -> str:
        return f"{self.title}:\n{self.frame.to_string()}"


# ------------------------------------------------------------------- Extras_Encoding selection

#: The most matches a Score_Check applies to in a drawn table. Two different means of rates with
#: these denominators differ by at least 100 / (5 x L), where L, the lcm of five counts up to 30,
#: is at most 29 x 28 x 27 x 25 x 23 = 12,606,300; so by more than 1.5e-6.
MAX_APPLICABLE = 30
#: Match counts per check: mostly the small counts 3, 6, 9 and 12, whose rates are not exact floats
#: and which let passes move between checks; sometimes 0 (no match: the rate is missing).
_APPLICABLE = st.one_of(st.sampled_from((6, 3, 6, 12, 9, 0)), st.integers(1, MAX_APPLICABLE))
#: Whether a candidate other than the leader also holds the highest mean (about 1 in 3), and how many
#: mean-keeping moves a candidate makes (at least 1 when tied means should differ as floats).
_JOINS_THE_TOP = st.sampled_from((False, True, False))
_N_MOVES = st.sampled_from((1, 2, 0, 3))
_N_NEAR_TIE_MOVES = st.sampled_from((1, 2, 3))


class EncodingCase(NamedTuple):
    """A pass-rate table as counts: the candidates in table order, the matches each Score_Check
    applies to (the same under every candidate) and, per candidate, the matches passing each check."""

    order: tuple[str, ...]
    applicable: tuple[int, ...]
    passes: Mapping[str, tuple[int, ...]]

    def __repr__(self) -> str:
        applies = ", ".join(f"{check} {n}" for check, n in zip(CHECKS, self.applicable))
        passes = "\n".join(f"  {name}: {self.passes[name]}" for name in self.order)
        return f"table order {list(self.order)}; matches each check applies to: {applies}\npasses:\n{passes}"


def _rate(passes: int, applicable: int) -> float:
    """An Exact_Pass_Rate as exact_pass_rates stores it: 100 x passes / applicable as a float64, NaN
    when the check applies to no match."""
    return 100.0 * passes / applicable if applicable else math.nan


def _rates_table(case: EncodingCase) -> pd.DataFrame:
    """The exact_pass_rates table of ``case``: ``encoding`` and the five rate columns, in table order."""
    columns: dict[str, Any] = {"encoding": pd.array(list(case.order), dtype="str")}
    for j, column in enumerate(EXACT_COLUMNS):
        rates = [_rate(case.passes[name][j], case.applicable[j]) for name in case.order]
        columns[column] = np.array(rates, dtype=np.float64)
    return pd.DataFrame(columns)


def _exact_mean(passes: Sequence[int], applicable: Sequence[int]) -> Fraction:
    """The mean of the defined Exact_Pass_Rates, as an exact fraction of the counts."""
    defined = [Fraction(100 * p, n) for p, n in zip(passes, applicable) if n]
    return sum(defined, Fraction(0)) / len(defined)


def _float_mean(passes: Sequence[int], applicable: Sequence[int]) -> float:
    """The mean of the defined float rates (math.fsum / count); used only to label examples."""
    defined = [_rate(p, n) for p, n in zip(passes, applicable) if n]
    return math.fsum(defined) / len(defined)


def _one_move_away(applicable: Sequence[int], vector: Sequence[int]) -> list[list[int]]:
    """Every pass vector one mean-keeping move from ``vector``: n1/g more passes on a check c1 and
    n2/g fewer on another check c2, where n1 and n2 are their match counts and g = gcd(n1, n2), so the
    sum of passes / matches is unchanged."""
    found = []
    checks = [j for j, n in enumerate(applicable) if n]
    for c1 in checks:
        for c2 in checks:
            g = math.gcd(applicable[c1], applicable[c2])
            up, down = applicable[c1] // g, applicable[c2] // g
            if c1 != c2 and vector[c1] + up <= applicable[c1] and vector[c2] >= down:
                moved = list(vector)
                moved[c1] += up
                moved[c2] -= down
                found.append(moved)
    return found


def _keep_mean_moves(
    draw: Any, applicable: Sequence[int], vector: list[int], avoid: set[float], moves: st.SearchStrategy[int]
) -> list[int]:
    """``vector`` after a drawn number of mean-keeping moves; each move avoids, where it can, a
    vector whose float mean is in ``avoid``."""
    for _ in range(draw(moves)):
        options = _one_move_away(applicable, vector)
        options = [moved for moved in options if _float_mean(moved, applicable) not in avoid] or options
        if not options:
            break
        vector = draw(st.sampled_from(options))
    return vector


def _lower_mean(draw: Any, applicable: Sequence[int], vector: list[int]) -> None:
    """Take 1 or more passes from one check and sometimes add 1 back on a check that applies to more
    matches: the exact mean falls, by as little as 100 x (1/n - 1/n') / k."""
    j = draw(st.sampled_from([c for c, p in enumerate(vector) if p > 0]))
    vector[j] -= draw(st.integers(1, vector[j]))
    room = [c for c, n in enumerate(applicable) if n > applicable[j] and vector[c] < n]
    if room and draw(st.booleans()):
        vector[draw(st.sampled_from(room))] += 1


@st.composite
def encoding_cases(draw: Any) -> EncodingCase:
    """The Config's candidates in a drawn order, shared match counts and pass counts per candidate:
    a drawn leader and each candidate that joins it share the highest mean, and the rest are below.
    In about half the examples, each candidate at the top moves, where it can, to a float mean that
    no earlier candidate at the top has, so tied means differ as floats."""
    order = tuple(draw(st.permutations(CANDIDATE_ENCODINGS)))
    applicable = [draw(_APPLICABLE) for _ in CHECKS]
    if not any(applicable):  # at least one check applies to a match, so every mean is defined
        applicable[draw(st.integers(0, len(CHECKS) - 1))] = draw(st.integers(1, MAX_APPLICABLE))
    # Pass counts strictly between 0 and the match count leave room for moves in both directions.
    top = [draw(st.one_of(st.integers(1, n - 1), st.integers(0, n)) if n > 1 else st.integers(0, n)) for n in applicable]
    if not any(top):  # a candidate below the highest mean needs a pass to lose
        first = next(j for j, n in enumerate(applicable) if n)
        top[first] = applicable[first]
    leader = draw(st.sampled_from(order))
    near_ties = draw(st.booleans())
    at_top: set[float] = set()
    passes: dict[str, tuple[int, ...]] = {}
    for name in order:
        vector = list(top)
        if name == leader or draw(_JOINS_THE_TOP):
            if near_ties:
                vector = _keep_mean_moves(draw, applicable, vector, at_top, _N_NEAR_TIE_MOVES)
            else:
                vector = _keep_mean_moves(draw, applicable, vector, set(), _N_MOVES)
            at_top.add(_float_mean(vector, applicable))
        else:
            _lower_mean(draw, applicable, vector)
            vector = _keep_mean_moves(draw, applicable, vector, set(), _N_MOVES)
        passes[name] = tuple(vector)
    return EncodingCase(order, tuple(applicable), passes)


# Explicit examples: no runs-margin check applies; the chase, wickets-margin and tie checks apply to
# 3 matches each and the over check to 6. _LOW and _HIGH both have the mean 275/6, which comes out as
# 45.83333333333333 from the rates 0, 0, 100, 83.33... and as 45.833333333333336 from 33.33..., 0,
# 66.66..., 83.33...
_O1, _O2, *_REST = [name for name in CANDIDATE_ENCODINGS if name != F30]
_COUNTS = (0, 3, 3, 3, 6)
_LOW, _HIGH, _BELOW, _NONE = (0, 0, 0, 3, 5), (0, 1, 0, 2, 5), (0, 0, 0, 2, 5), (0, 0, 0, 0, 0)

#: The F30_Encoding, listed second, ties with the first-listed candidate and has the lower float mean.
F30_WINS_A_FLOAT_TIE = EncodingCase(
    (_O1, F30, _O2, *_REST),
    _COUNTS,
    {_O1: _HIGH, F30: _LOW, _O2: _BELOW, **{name: _NONE for name in _REST}},
)
#: The F30_Encoding is below; the first-listed of the two tied candidates has the lower float mean.
FIRST_LISTED_WINS_A_FLOAT_TIE = EncodingCase(
    (F30, _O2, _O1, *_REST),
    _COUNTS,
    {F30: _BELOW, _O2: _LOW, _O1: _HIGH, **{name: _NONE for name in _REST}},
)


@given(case=encoding_cases())
@example(case=F30_WINS_A_FLOAT_TIE)
@example(case=FIRST_LISTED_WINS_A_FLOAT_TIE)
def test_select_encoding_takes_the_highest_mean_then_the_f30_encoding_then_the_first_listed(
    case: EncodingCase,
) -> None:
    """select_encoding returns the candidate with the highest mean Exact_Pass_Rate; on a tie, the
    F30_Encoding if it is tied, otherwise the tied candidate listed first (Requirement 5, criterion 7).

    **Validates: Requirements 5.7**
    """
    rates = _rates_table(case)
    note(_Table("pass-rate table", rates))
    means = {name: _exact_mean(case.passes[name], case.applicable) for name in case.order}

    # Different means lie more than 10 tie tolerances apart, so only equal fractions can tie.
    distinct = sorted(set(means.values()))
    for low, high in zip(distinct, distinct[1:]):
        assert high - low > 10 * Fraction(_tolerance(float(high))), f"means {low} and {high} are too close"

    highest = max(means.values())
    tied = [name for name in case.order if means[name] == highest]
    expected = F30 if F30 in tied else tied[0]
    chosen = select_encoding(rates)
    assert chosen == expected, (
        f"criterion 7 selects {expected}: the highest mean is {highest} ({float(highest)!r}), held by {tied} "
        f"in table order; select_encoding returned {chosen}"
    )

    floats = {name: _float_mean(case.passes[name], case.applicable) for name in tied}
    if len(tied) == 1:
        event("the F30_Encoding alone has the highest mean" if expected == F30 else "another candidate alone is highest")
    elif F30 in tied:
        event("the F30_Encoding ties for the highest mean")
    else:
        event("candidates other than the F30_Encoding tie for the highest mean")
    if len(set(floats.values())) > 1:
        event("tied means differ as floats")
    if any(value > floats[expected] for value in floats.values()):
        event("a tied candidate that is not selected has the higher float mean")
    if 0 in case.applicable:
        event("a check applies to no match")


# --------------------------------------------------------------------- Primary_Model selection

class CandidateRow(NamedTuple):
    """One comparison-table row: ``failure_reason`` is "" or None for a candidate that passed, and
    ``drawn`` says how the log loss was drawn."""

    model: str
    candidate: str
    log_loss: float
    failure_reason: str | None
    drawn: str


class SelectionCase(NamedTuple):
    """Comparison-table rows in table order; per model, the passing candidates drawn as tied for the
    lowest log loss (empty when every candidate failed), the base log loss and the n_rows count."""

    rows: tuple[CandidateRow, ...]
    lowest: Mapping[str, tuple[str, ...]]
    base: Mapping[str, float]
    n_rows: Mapping[str, int]

    def __repr__(self) -> str:
        rows = "\n".join(
            f"  {row.model} {row.candidate}: log_loss {row.log_loss!r}, failure_reason {row.failure_reason!r} "
            f"({row.drawn})"
            for row in self.rows
        )
        lowest = {model: list(group) for model, group in self.lowest.items()}
        return f"rows in table order:\n{rows}\ndrawn as tied for the lowest log loss: {lowest}"


#: Base log losses: the delivered M3-2, M3-1 and M4 Primary_Model values, 2.5, and any value from
#: 0.05 to 3 (up to 1 the absolute arm of the tolerance applies, above 1 the relative arm).
_BASES = st.one_of(st.sampled_from((1.48334, 0.40639, 2.5, 0.62786)), st.floats(min_value=0.05, max_value=3.0))
#: Offsets from the base, in tolerances: inside it (a tie) and just outside it.
_INSIDE = st.one_of(st.sampled_from((0.45, -0.45, 0.25, -0.25)), st.floats(min_value=-0.45, max_value=0.45))
_JUST_OUTSIDE = st.one_of(st.sampled_from((1.1, 1.5)), st.floats(min_value=1.1, max_value=4.0))
#: How the log loss of a passing (other than the base one) or failed candidate is drawn.
_PASSED = ("exact tie", "tie within tolerance", "just outside tolerance", "above")
_FAILED = ("no log loss", "lowest", "exact tie", "tie within tolerance", "above")
_TIED = ("base", "exact tie", "tie within tolerance")
_FAILURE_REASONS = (
    "fit failed: XGBoostError: check failed",
    "prediction failed: the model gave a probability outside [0, 1]",
    "non-finite prediction: 3 value(s) of wp_post",
    "M3 probability outside [0, 1]: 1 value(s) of wp_raw",
    "M4 class probabilities differ from a sum of 1 by more than 1e-6: 2 row(s) of actual",
)
#: A candidate that passed: an empty reason, or a missing one as in a re-read CSV.
_PASSED_MARKERS = st.sampled_from(("", None))
#: Whether every candidate of a model failed (about one model in eight).
_NO_CANDIDATE_LEFT = st.sampled_from((False, False, False, True, False, False, False, False))


def _loss(draw: Any, drawn: str, base: float) -> float:
    tol = _tolerance(base)
    if drawn in ("base", "exact tie"):
        return base
    if drawn == "tie within tolerance":
        return base + draw(_INSIDE) * tol
    if drawn == "just outside tolerance":
        return base + draw(_JUST_OUTSIDE) * tol
    if drawn == "above":
        return base + draw(st.floats(min_value=1e-6, max_value=0.5))
    if drawn == "lowest":
        return base - draw(st.floats(min_value=1e-6, max_value=base / 2))
    return math.nan  # "no log loss": a candidate whose fit failed has no metrics


@st.composite
def _model_rows(draw: Any, model: str) -> tuple[list[CandidateRow], tuple[str, ...], float]:
    """The rows of one model, the passing candidates drawn as tied for its lowest log loss, and its base."""
    order = FAMILY_ORDER[model]
    size = draw(st.sampled_from(range(len(order), 0, -1)))  # every family, down to one
    families = draw(st.permutations(order))[:size]
    base = draw(_BASES)
    if draw(_NO_CANDIDATE_LEFT):
        anchor, failed = None, set(families)
    else:
        anchor = draw(st.sampled_from(families))
        failed = {family for family in families if family != anchor and draw(st.booleans())}
    rows = []
    for family in families:
        if family in failed:
            drawn, reason = draw(st.sampled_from(_FAILED)), draw(st.sampled_from(_FAILURE_REASONS))
        else:
            drawn = "base" if family == anchor else draw(st.sampled_from(_PASSED))
            reason = draw(_PASSED_MARKERS)
        rows.append(CandidateRow(model, family, _loss(draw, drawn, base), reason, drawn))
    lowest = tuple(row.candidate for row in rows if not row.failure_reason and row.drawn in _TIED)
    return rows, lowest, base


@st.composite
def selection_cases(draw: Any) -> SelectionCase:
    """Rows of M3-1, M3-2 and M4 in a drawn table order."""
    rows: list[CandidateRow] = []
    lowest: dict[str, tuple[str, ...]] = {}
    base: dict[str, float] = {}
    n_rows: dict[str, int] = {}
    for model in MODELS:
        model_rows, lowest[model], base[model] = draw(_model_rows(model))
        rows.extend(model_rows)
        n_rows[model] = draw(st.integers(1, 200_000))
    order = draw(st.permutations(range(len(rows))))
    return SelectionCase(tuple(rows[i] for i in order), lowest, base, n_rows)


def _selection_example(rows_by_model: Mapping[str, Sequence[tuple[str, float, str | None, str]]]) -> SelectionCase:
    """A SelectionCase from (candidate, log_loss, failure_reason, drawn) tuples per model."""
    rows: list[CandidateRow] = []
    lowest: dict[str, tuple[str, ...]] = {}
    base: dict[str, float] = {}
    for model, entries in rows_by_model.items():
        model_rows = [CandidateRow(model, *entry) for entry in entries]
        rows.extend(model_rows)
        lowest[model] = tuple(row.candidate for row in model_rows if not row.failure_reason and row.drawn in _TIED)
        base[model] = next((row.log_loss for row in model_rows if row.drawn == "base"), math.nan)
    return SelectionCase(tuple(rows), lowest, base, {model: 1000 for model in rows_by_model})


_FIT_FAILED, _NON_FINITE_REASON, _SUM_REASON = _FAILURE_REASONS[0], _FAILURE_REASONS[2], _FAILURE_REASONS[4]

#: M3-1: logistic, lgbm and rf tie within the tolerance (rf holds the lowest float value) and xgb,
#: which failed, has the lowest log loss: logistic. M3-2 (absolute arm, tolerance 1e-9): logistic is
#: 1.15e-9 above lgbm, just outside, and rf failed with a tied value: lgbm. M4 (relative arm,
#: tolerance 1.48334e-9): xgb is 1.15e-9 above rf, inside, and lgbm failed with a tied value: xgb.
TIES_AT_THE_TOLERANCE = _selection_example(
    {
        "M3-1": [
            ("rf", 0.62786 - 0.3e-9, "", "tie within tolerance"),
            ("xgb", 0.5, _FIT_FAILED, "lowest"),
            ("lgbm", 0.62786, None, "base"),
            ("logistic", 0.62786 + 0.45e-9, "", "tie within tolerance"),
        ],
        "M3-2": [
            ("logistic", 0.40639 + 1.15e-9, "", "just outside tolerance"),
            ("lgbm", 0.40639, "", "base"),
            ("rf", 0.40639, _NON_FINITE_REASON, "exact tie"),
        ],
        "M4": [
            ("rf", 1.48334, "", "base"),
            ("xgb", 1.48334 + 1.15e-9, "", "tie within tolerance"),
            ("lgbm", 1.48334, _SUM_REASON, "exact tie"),
        ],
    }
)
#: Every M4 candidate failed, one with the lowest log loss and one with none: no M4 Primary_Model.
ALL_M4_CANDIDATES_FAILED = _selection_example(
    {
        "M3-1": [("lgbm", 0.67676, "", "base")],
        "M3-2": [("logistic", 0.3, _FAILURE_REASONS[1], "lowest"), ("lgbm", 0.41907, None, "base")],
        "M4": [("xgb", 1.2, _SUM_REASON, "lowest"), ("lgbm", math.nan, _FIT_FAILED, "no log loss")],
    }
)


def _table_order(key: tuple[str, str]) -> tuple[int, int]:
    """Sort key of a (model, candidate) pair in the comparison table: model, then tie order."""
    return MODELS.index(key[0]), FAMILY_ORDER[key[0]].index(key[1])


def _selection_events(case: SelectionCase, model: str, expected: str | None) -> None:
    rows = [row for row in case.rows if row.model == model]
    if expected is None:
        event("every candidate of a model failed")
        return
    passed = [row for row in rows if not row.failure_reason]
    group = [row.log_loss for row in passed if row.candidate in case.lowest[model]]
    if len(group) > 1:
        event("passing candidates tie exactly" if len(set(group)) == 1 else "passing candidates tie within tolerance")
    if any(row.drawn == "just outside tolerance" for row in passed):
        event("a passing log loss lies just outside the tolerance")
    lowest_passing = min(row.log_loss for row in passed)
    if next(row.log_loss for row in passed if row.candidate == expected) > lowest_passing:
        event("the selected log loss is not the lowest float value")
    if any(row.failure_reason and row.log_loss < lowest_passing for row in rows):
        event("a failed candidate has the lowest log loss")
    order = FAMILY_ORDER[model]
    if any(row.failure_reason and row.drawn in _TIED and order.index(row.candidate) < order.index(expected) for row in rows):
        event("a failed candidate earlier in the tie order has a tied log loss")
    event("relative tolerance (base above 1)" if case.base[model] > 1 else "absolute tolerance (base up to 1)")


@given(case=selection_cases())
@example(case=TIES_AT_THE_TOLERANCE)
@example(case=ALL_M4_CANDIDATES_FAILED)
def test_select_primary_takes_the_lowest_log_loss_of_a_passing_candidate_in_tie_order(case: SelectionCase) -> None:
    """For M3-1, M3-2 and M4 separately, the Primary_Model is the passing candidate with the lowest
    log loss, ties going to the first in the tie order; a failed candidate is never selected, and a
    model whose candidates all failed stops the Run (Requirement 23, criteria 7, 8 and 12).

    **Validates: Requirements 23.7, 23.8, 23.12**
    """
    expected = {
        model: (min(group, key=FAMILY_ORDER[model].index) if group else None) for model, group in case.lowest.items()
    }
    table = pd.DataFrame(
        {
            "model": [row.model for row in case.rows],
            "candidate": [row.candidate for row in case.rows],
            "log_loss": np.array([row.log_loss for row in case.rows], dtype=np.float64),
            "failure_reason": [row.failure_reason for row in case.rows],
        }
    )
    note(_Table("comparison table in drawn row order", table))

    for model in MODELS:
        _selection_events(case, model, expected[model])
        if expected[model] is None:
            with pytest.raises(StageError):
                select_primary(table, model)
            continue
        # The drawn ties and gaps lie clear of the tolerance (ties within 0.95 of it, gaps over 1.05),
        # far beyond float rounding (about 1e-16), so any reading of "equal values" agrees.
        passed = [row for row in case.rows if row.model == model and not row.failure_reason]
        tied = [row.log_loss for row in passed if row.candidate in case.lowest[model]]
        low = min(tied)
        tol = _tolerance(low)
        assert max(tied) - low <= 0.95 * tol, f"{model}: the tied log losses {tied} spread too far"
        others = [row.log_loss for row in passed if row.candidate not in case.lowest[model]]
        assert all(loss - low >= 1.05 * tol for loss in others), f"{model}: a log loss of {others} is too near {low}"
        chosen = select_primary(table, model)
        assert chosen == expected[model], (
            f"{model}: the passing candidates {list(case.lowest[model])} tie for the lowest log loss, so "
            f"{expected[model]} is selected in the order {FAMILY_ORDER[model]}; select_primary returned {chosen}"
        )

    results = [
        CandidateResult(
            row.model, row.candidate, case.n_rows[row.model], log_loss=row.log_loss, failure_reason=row.failure_reason or ""
        )
        for row in case.rows
    ]
    if None in expected.values():
        with pytest.raises(StageError):
            candidate_table(results)
        return
    frame = candidate_table(results)
    note(_Table("candidate_table", frame))
    keys = list(zip(frame["model"].tolist(), frame["candidate"].tolist()))
    assert keys == sorted(((row.model, row.candidate) for row in case.rows), key=_table_order)
    flagged = [key for key, flag in zip(keys, frame["primary"].tolist()) if flag]
    assert flagged == [(model, expected[model]) for model in MODELS], f"primary flags on {flagged}"
    reasons = dict(zip(keys, frame["failure_reason"].tolist()))
    assert all(reasons[key] == "" for key in flagged), "a flagged Primary_Model has a failure reason"
    assert primary_models(frame) == expected


# -------------------------------------------- calibration-aware tie-break (Requirement 23, criterion 8)

#: The preferred M4 family of the delivered Config (``model_choice.calibration_preferred``).
_M4_PREFERRED = "lgbm"


@given(
    best=st.floats(min_value=0.1, max_value=3.0),
    band=st.floats(min_value=1e-4, max_value=0.02),
    inside=st.booleans(),
    gap_frac=st.floats(min_value=0.0, max_value=1.0),
    rf_gap=st.floats(min_value=0.05, max_value=0.5),
    preferred_failed=st.booleans(),
)
def test_select_primary_prefers_a_calibration_family_within_the_band(
    best: float, band: float, inside: bool, gap_frac: float, rf_gap: float, preferred_failed: bool
) -> None:
    """With a preferred family named for M4, ``select_primary`` returns it when it passed and its
    pooled log loss lies within ``band`` x best of the lowest; otherwise the lowest in tie order. A
    model with no preferred entry keeps the plain lowest-log-loss rule (Requirement 23, criterion 8).

    **Validates: Requirements 23.8**
    """
    # xgb holds the lowest log loss (best); lgbm (preferred) sits strictly above it, clearly inside
    # (0.1-0.9 of the band) or clearly outside (1.1-2.1), so float rounding of the gap cannot flip
    # the within-band decision and xgb is the unique lowest when there is no preference.
    factor = 0.1 + gap_frac * 0.8 if inside else 1.1 + gap_frac
    lgbm = best + factor * band * best
    table = pd.DataFrame(
        {
            "model": ["M4", "M4", "M4"],
            "candidate": ["xgb", "lgbm", "rf"],
            "log_loss": np.array([best, lgbm, best + rf_gap], dtype=np.float64),
            "failure_reason": ["", (_FIT_FAILED if preferred_failed else ""), ""],
        }
    )
    note(_Table("M4 comparison", table))
    preferred = {"M4": (_M4_PREFERRED, band)}
    within = (lgbm - best) <= band * best
    chosen = select_primary(table, "M4", preferred=preferred)
    if preferred_failed:
        event("the preferred family failed, so the lowest log loss wins")
        assert chosen == "xgb"
    elif within:
        event("the preferred family lies within the band")
        assert chosen == "lgbm"
    else:
        event("the preferred family lies outside the band")
        assert chosen == "xgb"
    # With no preferred mapping (or an empty one), the lowest log loss wins regardless.
    assert select_primary(table, "M4") == "xgb"
    assert select_primary(table, "M4", preferred={}) == "xgb"


def test_select_primary_keeps_the_preferred_family_when_it_is_already_the_lowest() -> None:
    """When the preferred family already holds the lowest log loss it stays the Primary_Model, with
    or without the preference (Requirement 23, criterion 8).

    **Validates: Requirements 23.8**
    """
    table = pd.DataFrame(
        {
            "model": ["M4", "M4", "M4"],
            "candidate": ["lgbm", "xgb", "rf"],
            "log_loss": np.array([1.40, 1.45, 1.50], dtype=np.float64),
            "failure_reason": ["", "", ""],
        }
    )
    assert select_primary(table, "M4", preferred={"M4": (_M4_PREFERRED, 0.005)}) == "lgbm"
    assert select_primary(table, "M4") == "lgbm"


def test_candidate_table_flags_the_preferred_family_within_the_band() -> None:
    """candidate_table sets the primary flag through select_primary with the preferred mapping, so
    the M4 preferred family within the band is flagged while M3 (no preferred family) stays on the
    lowest log loss, matching the delivered M4 numbers (Requirement 23, criterion 8).

    **Validates: Requirements 23.8**
    """
    results = [
        CandidateResult("M3-1", "logistic", 100, log_loss=0.62786),
        CandidateResult("M3-1", "lgbm", 100, log_loss=0.67676),
        CandidateResult("M3-2", "logistic", 100, log_loss=0.40639),
        CandidateResult("M3-2", "lgbm", 100, log_loss=0.41907),
        CandidateResult("M4", "lgbm", 100, log_loss=1.483181),
        CandidateResult("M4", "xgb", 100, log_loss=1.481957),
        CandidateResult("M4", "rf", 100, log_loss=1.486774),
    ]
    preferred = {"M4": (_M4_PREFERRED, 0.005)}
    assert primary_models(candidate_table(results, preferred=preferred)) == {
        "M3-1": "logistic",
        "M3-2": "logistic",
        "M4": "lgbm",
    }
    # Without the preference, M4 falls to the lowest log loss (xgb); M3 is unchanged.
    assert primary_models(candidate_table(results)) == {"M3-1": "logistic", "M3-2": "logistic", "M4": "xgb"}


# ------------------------------------------------------------------------------ failure checks

#: Valid WP values, the bounds 0, -0 and 1 included, and values that break criterion 12: missing,
#: infinite, or outside [0, 1] by as little as one float step.
_WP = st.one_of(st.sampled_from((0.0, -0.0, 1.0)), st.floats(min_value=0.0, max_value=1.0))
_BAD_WP = st.sampled_from(
    (-1e-12, 1.5, -5e-324, math.nextafter(1.0, 2.0), -0.25, 1.0 + 1e-9, math.nan, math.inf, -math.inf)
)
_NON_FINITE = st.sampled_from((math.nan, math.inf, -math.inf))
_BELOW_ZERO = st.sampled_from((5e-324, 1e-12, 0.05))
#: How many values an example breaks (0 in about a quarter of the examples), and the kind of each M4 defect.
_N_DEFECTS = st.sampled_from((0, 1, 2, 3))
_M4_DEFECTS = st.sampled_from(("sum", "sum", "outside", "non-finite"))
#: Row-sum shifts: under 1e-6 (the row passes) or over it (the row fails), each at least 2e-8 from
#: 1e-6, far beyond the float error of a row sum (about 1e-16).
_SUM_SHIFTS = st.one_of(
    st.floats(min_value=1e-9, max_value=0.98e-6), st.floats(min_value=1.02e-6, max_value=1e-2)
)

#: The phrase of Requirement 23, criterion 12 that names each condition, in the order checked.
PHRASES: Mapping[str, str] = {"non-finite": "non-finite", "outside": "outside [0, 1]", "sum": "sum of 1"}


def _one_hot(k: int) -> list[float]:
    return [1.0 if c == k else 0.0 for c in range(N_CLASSES)]


_ROW_WEIGHTS = st.one_of(
    st.integers(0, N_CLASSES - 1).map(_one_hot),
    st.lists(st.integers(0, 20), min_size=N_CLASSES, max_size=N_CLASSES).filter(any),
)


class FailureCase(NamedTuple):
    """The prediction arrays of one M3 candidate (name -> array) and of one M4 candidate (variant ->
    (rows, 9) array)."""

    m3: Mapping[str, np.ndarray]
    m4: Mapping[str, np.ndarray]

    def __repr__(self) -> str:
        def lines(arrays: Mapping[str, np.ndarray]) -> str:
            return "\n".join(f"  {name}: {values.tolist()!r}" for name, values in arrays.items())

        return f"M3 predictions:\n{lines(self.m3)}\nM4 variants:\n{lines(self.m4)}"


@st.composite
def _m3_predictions(draw: Any) -> dict[str, np.ndarray]:
    n = draw(st.integers(1, 4))
    sizes = {"wp_raw": (n,), "wp_pre": (n,), "wp_post": (n, N_CLASSES)}
    arrays = {
        name: np.array(draw(st.lists(_WP, min_size=math.prod(shape), max_size=math.prod(shape))), dtype=np.float64)
        .reshape(shape)
        for name, shape in sizes.items()
    }
    for _ in range(draw(_N_DEFECTS)):
        flat = arrays[draw(st.sampled_from(M3_ARRAYS))].reshape(-1)  # a view of the array
        flat[draw(st.integers(0, flat.size - 1))] = draw(_BAD_WP)
    return arrays


@st.composite
def _m4_variants(draw: Any) -> dict[str, np.ndarray]:
    names = draw(st.lists(st.sampled_from(M4_VARIANTS), min_size=1, max_size=len(M4_VARIANTS), unique=True))
    n = draw(st.integers(1, 4))
    variants = {}
    for name in names:
        weights = np.array(draw(st.lists(_ROW_WEIGHTS, min_size=n, max_size=n)), dtype=np.float64)
        variants[name] = weights / weights.sum(axis=1, keepdims=True)
    shifted: set[tuple[str, int]] = set()
    for _ in range(draw(_N_DEFECTS)):
        name, row = draw(st.sampled_from(names)), draw(st.integers(0, n - 1))
        cells = variants[name][row]  # a view of the row
        defect = draw(_M4_DEFECTS)
        if defect == "non-finite":
            cells[draw(st.integers(0, N_CLASSES - 1))] = draw(_NON_FINITE)
        elif defect == "outside":
            i = draw(st.integers(0, N_CLASSES - 1))
            j = draw(st.sampled_from([c for c in range(N_CLASSES) if c != i]))
            below = draw(_BELOW_ZERO)
            cells[j] += cells[i] + below  # the mass moves to cell j, so the row still sums to 1
            cells[i] = -below
        elif (name, row) not in shifted:  # one shift per row, so two shifts never add up near 1e-6
            shifted.add((name, row))
            shift = draw(_SUM_SHIFTS)
            if draw(st.booleans()):
                cells[int(np.argmin(cells))] += shift
            else:
                cells[int(np.argmax(cells))] -= shift
    return variants


@st.composite
def failure_cases(draw: Any) -> FailureCase:
    return FailureCase(draw(_m3_predictions()), draw(_m4_variants()))


def _sum_gap(row: Sequence[float]) -> Fraction:
    """|sum of the row - 1|, exactly."""
    return abs(sum(map(Fraction, row), Fraction(0)) - 1)


def _m3_conditions(arrays: Mapping[str, np.ndarray]) -> list[tuple[str, dict[str, int]]]:
    """Per condition in check order, the number of bad values in each M3 array."""
    non_finite, outside = {}, {}
    for name, values in arrays.items():
        flat = np.asarray(values).ravel().tolist()
        non_finite[name] = sum(not math.isfinite(v) for v in flat)
        outside[name] = sum(math.isfinite(v) and not 0.0 <= v <= 1.0 for v in flat)
    return [("non-finite", non_finite), ("outside", outside)]


def _m4_conditions(variants: Mapping[str, np.ndarray]) -> list[tuple[str, dict[str, int]]]:
    """Per condition in check order, the number of bad cells (or, for the sum, rows) of each variant."""
    non_finite, outside, off = {}, {}, {}
    for name, values in variants.items():
        rows = np.asarray(values).tolist()
        cells = [v for row in rows for v in row]
        non_finite[name] = sum(not math.isfinite(v) for v in cells)
        outside[name] = sum(math.isfinite(v) and not 0.0 <= v <= 1.0 for v in cells)
        off[name] = sum(all(map(math.isfinite, row)) and _sum_gap(row) > PROB_SUM_TOL for row in rows)
    return [("non-finite", non_finite), ("outside", outside), ("sum", off)]


def _first_condition(conditions: Sequence[tuple[str, Mapping[str, int]]]) -> tuple[str | None, set[str]]:
    """The first condition that some array breaks, with the arrays that break it; (None, set()) if none."""
    for kind, counts in conditions:
        named = {name for name, count in counts.items() if count}
        if named:
            return kind, named
    return None, set()


def _check_reason(reason: str, kind: str | None, named: set[str], names: Sequence[str], what: str) -> None:
    if kind is None:
        assert reason == "", f"{what}: no value breaks criterion 12, but the reason is {reason!r}"
        return
    assert isinstance(reason, str) and reason.strip(), f"{what}: {kind} values in {sorted(named)} are not flagged"
    for other, phrase in PHRASES.items():
        assert (phrase in reason) == (other == kind), f"{what}: the reason {reason!r} must name the {kind} condition"
    found = {name for name in names if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", reason)}
    assert found == named, f"{what}: the reason {reason!r} names {sorted(found)}; {sorted(named)} hold {kind} values"


def _shifted(shift: float) -> list[float]:
    """A uniform row of 9 probabilities with its first cell moved by ``shift``."""
    row = [1 / N_CLASSES] * N_CLASSES
    row[0] += shift
    return row


def _uniform() -> list[float]:
    return [1 / N_CLASSES] * N_CLASSES


#: Values on the bounds pass: 0, -0 and 1 in every M3 array, one-hot M4 rows, and M4 row sums off by
#: 0.98e-6 in either direction.
BOUNDS_PASS = FailureCase(
    {
        "wp_raw": np.array([0.0, 1.0, -0.0]),
        "wp_pre": np.array([1.0, -0.0, 0.5]),
        "wp_post": np.array([_one_hot(0), _one_hot(4), _one_hot(8)]),
    },
    {
        "actual": np.array([_one_hot(0), _shifted(0.98e-6)]),
        "bowler_view_nofh": np.array([_one_hot(8), _shifted(-0.98e-6)]),
    },
)
#: One float step outside [0, 1] fails in wp_post only; M4 row sums off by 1.02e-6 fail in actual and
#: batter_view, but not in batter_view_fh, whose name contains "batter_view".
JUST_OUTSIDE_FAILS = FailureCase(
    {
        "wp_raw": np.array([0.5, 0.25]),
        "wp_pre": np.array([0.5, 0.75]),
        "wp_post": np.array([[-5e-324, *([0.5] * 8)], [*([0.5] * 8), math.nextafter(1.0, 2.0)]]),
    },
    {
        "actual": np.array([_uniform(), _shifted(1.02e-6)]),
        "batter_view": np.array([_shifted(-1.02e-6), _uniform()]),
        "batter_view_fh": np.array([_uniform(), _uniform()]),
    },
)
#: A non-finite value is named before a value outside [0, 1] (M3) and before a cell below 0 and a
#: row sum off by 0.01 (M4).
NON_FINITE_FIRST = FailureCase(
    {
        "wp_raw": np.array([math.nan, 0.5]),
        "wp_pre": np.array([-0.25, 0.5]),
        "wp_post": np.array([_one_hot(1), _one_hot(2)]),
    },
    {
        "actual": np.array([[math.inf, *([0.0] * 8)]]),
        "avg_both": np.array([[-0.05, 1.05, *([0.0] * 7)]]),
        "bowler_view": np.array([_shifted(1e-2)]),
    },
)


@given(case=failure_cases())
@example(case=BOUNDS_PASS)
@example(case=JUST_OUTSIDE_FAILS)
@example(case=NON_FINITE_FIRST)
def test_failure_checks_flag_exactly_the_bad_predictions_and_failed_candidates_are_not_selected(
    case: FailureCase,
) -> None:
    """m3_failure and m4_failure give a reason exactly when a prediction is non-finite, outside
    [0, 1], or (M4) in a row whose sum differs from 1 by more than 1e-6; a candidate with a reason
    is not selected even with the lowest log loss (Requirement 23, criterion 12).

    **Validates: Requirements 23.12**
    """
    # Row sums lie clear of 1 +- 1e-6, so float rounding in the check cannot change the outcome.
    for name, values in case.m4.items():
        for row in values.tolist():
            if all(map(math.isfinite, row)):
                assert abs(_sum_gap(row) - PROB_SUM_TOL) > Fraction(1, 10**8), f"{name} row {row} sums too near 1e-6"

    m3_kind, m3_named = _first_condition(_m3_conditions(case.m3))
    m3_reason = m3_failure(case.m3)
    note(f"m3_failure: {m3_reason!r}")
    _check_reason(m3_reason, m3_kind, m3_named, list(case.m3), "M3")

    m4_kind, m4_named = _first_condition(_m4_conditions(case.m4))
    m4_reason = m4_failure(case.m4)
    note(f"m4_failure: {m4_reason!r}")
    _check_reason(m4_reason, m4_kind, m4_named, list(case.m4), "M4")

    event(f"M3: {m3_kind or 'passes'}")
    event(f"M4: {m4_kind or 'passes'}")
    finite_rows = [row for values in case.m4.values() for row in values.tolist() if all(map(math.isfinite, row))]
    if m4_kind is None and any(_sum_gap(row) > Fraction(1, 10**12) for row in finite_rows):
        event("M4 passes with a row sum off by up to 1e-6")

    # The candidate given the lowest log loss, first in the tie order, is selected only if it passed.
    results = [
        CandidateResult("M3-1", "logistic", 100, log_loss=0.5, failure_reason=m3_reason),
        CandidateResult("M3-1", "lgbm", 100, log_loss=0.6),
        CandidateResult("M4", "lgbm", 100, log_loss=1.4, failure_reason=m4_reason),
        CandidateResult("M4", "xgb", 100, log_loss=1.5),
    ]
    frame = candidate_table(results)
    chosen = {
        model: candidate
        for model, candidate, flag in zip(frame["model"].tolist(), frame["candidate"].tolist(), frame["primary"].tolist())
        if flag
    }
    assert chosen == {"M3-1": "lgbm" if m3_kind else "logistic", "M4": "xgb" if m4_kind else "lgbm"}
