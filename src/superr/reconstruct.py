"""Score_Reconstructor and Result_Parser (Requirement 5; design section "Score_Reconstructor and
Result_Parser").

Each part of Requirement 5 is one section of this module, in the order of the design. Every
function receives the Config values it needs as arguments: no function reads ``config.py`` or has
a default value for a Config setting.

Delivery classification (criteria 2 and 3)
    ``DeliveryClass`` codes a Delivery as Legal_Ball, Wide or No_Ball; the code is the ``cls``
    column of ``balls``. The class of each ``extras_type`` value comes from the Config class
    mapping ``data_rules.extras_type_classes``: ``check_mapping`` validates the mapping and
    ``classify`` applies it. Values match as exact, case-sensitive text. The text ``None`` is a
    value like any other, so the CSV must be read with ``keep_default_na=False`` (as the
    Data_Loader does); only an Empty_Field is missing. A value listed under more than one class, a
    missing ``extras_type``, or a value that the mapping does not list stops the Run with a
    ``DeliveryClassError``, which is a ``DataError`` (exit code 3).

Ball order and Super_Over_Innings (criteria 1 and 11)
    ``order_deliveries`` sorts the Deliveries by ``match_id``, then ``innings``, then
    ``innings_ball_num``, each as ascending numbers, with a stable sort, and numbers them with
    ``row``: the global position in this order and the join key of every per-Delivery table.
    ``super_over_flags`` gives ``is_super_over``, true in innings 3 and 4 (the
    Super_Over_Innings), which every model-training, pressure and rating filter drops. The
    functions below that need the order sort by the same keys themselves, so they accept rows in
    any order and return results on the caller's index.

Scheduled_Balls (criteria 4-6)
    ``scheduled_balls`` gives innings 1 and 2 of every match: the innings' revised-overs column
    read as a count of balls; for innings 2, ``first_inn_revised_overs`` when
    ``second_inn_revised_overs`` is missing; else ``DEFAULT_SCHEDULED_BALLS`` (120).
    ``delivery_scheduled_balls`` copies the value to each Delivery. No revised-overs column
    applies to a Super_Over_Innings, so criterion 6 gives it 120 as well; no downstream filter
    keeps those Deliveries.

Free_Hits (criterion 12)
    ``free_hit_flags`` flags every Delivery after a No_Ball in the same innings, up to and
    including the next Legal_Ball, so Wides and further No_Balls in that span are flagged and a
    No_Ball is flagged only when it follows another No_Ball. A Delivery is a Free_Hit exactly when
    the last No_Ball before it in the innings comes after the last Legal_Ball before it in the
    innings; both positions come from forward-filled running maxima shifted by one row.

Wicket flags (criterion 8; D14)
    ``wicket_flags`` marks each Counted_Wicket (``is_wicket`` true and a ``dismissal_type_id``
    among the Counted_Wicket codes that the Auditor's classification supplies as an argument,
    Retire Out included, Retired_Hurt codes excluded) and, among them, the Bowler_Wickets, Run_Outs
    and Retire Outs by the D14 codes.

Team runs (Extras_Encodings E1-E4; design DD-4)
    ``ENCODINGS`` maps each candidate name used in ``data_rules.candidate_encodings`` to a
    function from ``EncodingInputs`` (built by ``encoding_inputs``) to the team runs of every
    Delivery. E1, the F30_Encoding, is listed first. With b = ``runs`` (bat runs) and
    e = ``extras_2_runs``:

    ===================== ========= ============= ============ ======= ============
    Encoding              Legal     Bye/leg-bye   Penalty      Wide    No_Ball
    ===================== ========= ============= ============ ======= ============
    E1 F30                b + e     b + e         b + e        1 + e   1 + b + e
    E2 byes_one           b + e     1 + e         b + e        1 + e   1 + b + e
    E3 byes_one_nb_plain  b + e     1 + e         b + e        1 + e   1 + b
    E4 byes_one_penalty5  b + e     1 + e         5 + b + e    1 + e   1 + b + e
    ===================== ========= ============= ============ ======= ============

    The Wide and No_Ball columns follow the Delivery class from the Config mapping. Among
    Legal_Balls, ``extras_type`` ``byes`` and ``legbyes`` take the bye/leg-bye column, ``Penalty``
    takes the Penalty column and every other value (as delivered, ``None``) takes the Legal
    column. The one-run Wide and No_Ball penalty and the five penalty runs are fixed by the Laws
    of Cricket. Counting a bye or leg-bye Delivery as 1 run is the E2-E4 assumption, because those
    run counts are not recorded (design finding F-4). ``candidate_team_runs`` evaluates the
    Config's candidates in Config order.

Innings aggregates and Targets (criteria 8-10)
    ``innings_aggregates`` gives each innings' total, Counted_Wickets fallen and team runs of its
    final Delivery under one encoding. ``targets`` gives each match's Target: ``revised_target``
    when it holds a value, else the first-innings total under that encoding plus 1.

Result_Parser (criteria 13-15)
    ``parse_result`` lower-cases a ``result`` string, turns each run of whitespace into one space
    and returns a ``ParsedResult``: the result type of the first rule that matches, the margin
    and the DLS flag (true exactly when the text contains ``dls`` or ``d/l``). The rules, in
    order, are SUPER_OVER_WIN (``super over`` or ``one over eliminator``, even when a margin is
    also stated), TIE (``tied``), ABANDONED (``abandoned``), NO_RESULT (``no result``),
    WIN_BY_RUNS (an integer, zero or more spaces, then ``run``) and WIN_BY_WICKETS (an integer,
    zero or more spaces, then ``wicket`` or ``wkt``). Only the last two carry a margin: the
    integer before the word. Text that matches no rule, and a missing ``result``, is UNPARSED.
    ``parse_results`` parses every match, and ``write_unparsed`` writes the ``match_id`` and the
    original ``result`` of each UNPARSED match to ``unparsed_results.csv`` in the audit folder
    (header only when there is none). An unparsed string does not stop the Run; a margin too
    large for the Int32 ``margin`` column does, with a ``DataError``.

Score_Checks and pass rates (criteria 7, 16-21 and 23)
    ``over_check`` runs the over check of every match with a Delivery in innings 1 or 2: in each
    of its innings 1 and 2, every ``overid`` but the highest holds exactly 6 Legal_Balls and the
    highest at most 6. It does not depend on the encoding. ``run_checks`` scores one
    Extras_Encoding and returns its ``score_checks`` rows: five per match (SCORE_CHECKS order),
    whether or not each check applies, so the table is rectangular. With T the Target, S2 the
    second-innings total, f the team runs of the final innings-2 Delivery and W2 the
    second-innings Counted_Wickets:

    ============== ======================= ================================== ================
    Check          Applies when            Runs_Check_Miss                    Passes exactly
    ============== ======================= ================================== ================
    runs_margin    win by runs, won by the abs(((T - 1) - S2) - margin)       miss = 0
                   side batting first
    chase          win by wickets          max(0, T - S2, (S2 - f) - (T - 1)) miss = 0
    wickets_margin win by wickets          blank                              margin = 10 - W2
    tie            tie or super-over win   abs(S2 - (T - 1))                  miss = 0
    over           every match             blank                              both innings
    ============== ======================= ================================== ================

    A match with no Delivery in innings 1 or 2 has nothing to reconstruct, so none of its checks
    applies; ``passed`` and ``runs_check_miss`` are blank where a check does not apply.
    ``exact_pass_rates`` gives the five Exact_Pass_Rates of each candidate (100 x passes /
    applicable matches, missing when no match is applicable). ``select_encoding`` picks the
    candidate with the highest mean rate, the F30_Encoding when it ties for the highest, else the
    tied candidate listed first (criterion 7). ``within_tolerance_pass_rates`` scores the selected
    encoding at the Score_Tolerance, given as an argument: a runs check passes when its miss is at
    most the tolerance, and the wickets-margin and over checks need an exact pass.
    ``pass_rate_table`` builds the criterion 21 table, and ``failing_matches`` lists each failed
    check of the selected encoding (criterion 23). ``write_pass_rates`` and
    ``write_failing_matches`` write ``pass_rates.csv`` and ``failing_matches.csv`` to the audit
    folder they are given. ``run_checks`` stops the Run with a ``DataError`` when innings 1 or 2
    of a match not in ``matches`` holds Deliveries, or when a chase or tie check applies to a
    match without a Target; ``over_check`` does so when an ``overid`` of innings 1 or 2 is missing
    or not a whole number.

Match exclusion (criteria 24 and 27)
    ``exclusion_conditions`` lists, from the ``score_checks`` rows of the selected encoding, each
    condition of criterion 24 that a match meets at a Score_Tolerance t: a failed over check, a
    failed wickets-margin check, or a runs-margin, chase or tie check whose Runs_Check_Miss is
    above t. The excluded matches are the distinct ``match_id`` values of these rows. A match
    whose only failures are runs checks that miss by at most t therefore stays in, although
    ``failing_matches`` still lists it, and with t = 0 the excluded matches are exactly the
    matches in ``failing_matches``. ``exclusion_counts`` counts the distinct excluded matches at
    each tolerance of the Config reporting grid ``data_rules.score_tolerance_grid`` and flags the
    row of the Config Score_Tolerance D20; a larger tolerance never excludes more matches.
    ``excluded_flags`` gives the ``excluded`` flag of each Delivery: true for every Delivery of an
    excluded match where the Config option ``data_rules.exclude_failing_matches`` is on, false
    everywhere where it is off. Every model-training, validation and rating filter reads it.

Score-reconstruction section of audit.md (criteria 22 and 27)
    ``write_reconstruction_audit`` writes ``audit.md`` as the Auditor's text, read from the
    ``audit`` checkpoint, followed by one section headed RECONSTRUCTION_HEADING, so rerunning the
    ``reconstruct`` stage alone never repeats the section. The section gives the selected
    encoding, the pass rates, a warning for each Score_Check whose Exact_Pass_Rate under the
    selected encoding is below PASS_RATE_WARNING (95.0%, criterion 22) and, where
    ``exclude_failing_matches`` is on, the Score_Tolerance, the number of distinct excluded
    matches and the count at each grid tolerance (criterion 27). With the option on it also
    writes ``excluded_matches.csv`` and ``exclusion_counts.csv``; with it off it deletes any copy
    of the two files left by an earlier Run.

The balls and innings_info tables (design, Data Models)
    ``competition_types`` gives each competition its Competition_Type: IPL when ``comp_name``
    starts with "IPL", else T20I when ``match_format`` is ``T20I``, else A_Team when it is
    ``T20``. ``match_columns`` gives the match columns of ``balls`` for every match: the
    Competition_Type, the Season (the competition's ``season`` text), the Calendar_Year,
    ``match_date``, ``knock_out``, ``venue_id``, ``venue_country``, the Target under the selected
    encoding and the result type. ``dismissed_batter_flags`` resolves the batter that each
    Counted_Wicket dismisses (Requirement 15, criteria 9 and 10): the striker or non-striker whose
    id equals ``dismissed_player_id``, else, for a Counted_Wicket other than Retire Out, the
    striker, flagged as a fallback. ``balls_table`` assembles ``balls`` (BALLS_COLUMNS: every
    column of the design's table except ``fold``, which the Fold_Assigner adds), and
    ``innings_info`` the per-innings table of every candidate encoding. A match whose ``comp_id``
    is not in ``competitions``, a repeated ``comp_id``, a match in a competition with no
    Competition_Type and an integer value outside the int32 range stop the Run with a
    ``DataError``.

The ``reconstruct`` Pipeline_Stage (``run_stage``)
    Reads ``ball_by_ball``, ``matches`` and ``competitions`` from the ``load`` checkpoint and
    ``dismissal_classes`` and ``audit_text`` from the ``audit`` checkpoint, runs the sections
    above in order, and writes ``unparsed_results.csv``, ``pass_rates.csv``,
    ``failing_matches.csv``, ``excluded_matches.csv``, ``exclusion_counts.csv`` and ``audit.md``
    to ``outputs/audit/``. Its checkpoint holds the tables ``balls``, ``innings_info`` (every
    candidate), ``score_checks`` (every candidate) and ``pass_rates``, and the stage records the
    selected encoding as the data-derived Method_Constant ``reconstruct.selected_encoding``
    (Score_Reconstructor).

Data problems that make these values undefined stop the Run with a ``DataError`` (exit code 3)
that names every problem found: a missing or non-whole order key, a revised-overs value or
``revised_target`` that is not a positive whole number, a repeated ``match_id`` in ``matches``,
a missing or non-whole ``runs`` or ``extras_2_runs``, and a Delivery of innings 1 or 2 whose match
has no Scheduled_Balls.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Collection, Mapping, Sequence
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import pandas as pd

from superr.errors import DataError, StageError

if TYPE_CHECKING:
    from superr.pipeline import StageContext

__all__ = [
    "ABANDONED",
    "BALLS_COLUMNS",
    "BALLS_DERIVED_COLUMNS",
    "BALLS_TABLE",
    "BYE_EXTRAS_TYPES",
    "CHECK_CHASE",
    "CHECK_OVER",
    "CHECK_RUNS_MARGIN",
    "CHECK_TIE",
    "CHECK_WICKETS_MARGIN",
    "CLASS_KEYS",
    "COMPETITION_TYPES",
    "COMP_A_TEAM",
    "COMP_IPL",
    "COMP_T20I",
    "DEFAULT_SCHEDULED_BALLS",
    "DeliveryClass",
    "DeliveryClassError",
    "ENCODINGS",
    "EXACT_COLUMNS",
    "EXCLUDED_MATCHES_FILE",
    "EXCLUSION_COLUMNS",
    "EXCLUSION_COUNTS_FILE",
    "EXCLUSION_COUNT_COLUMNS",
    "EncodingInputs",
    "F30_ENCODING",
    "FAILING_MATCHES_FILE",
    "INNINGS_INFO_COLUMNS",
    "INNINGS_INFO_TABLE",
    "MATCH_INFO_COLUMNS",
    "NO_RESULT",
    "ORDER_KEYS",
    "PASS_RATES_FILE",
    "PASS_RATES_TABLE",
    "PASS_RATE_WARNING",
    "PENALTY_EXTRAS_TYPE",
    "ParsedResult",
    "RECONSTRUCTION_HEADING",
    "RESULT_TYPES",
    "RUNS_CHECKS",
    "SCORE_CHECKS",
    "SCORE_CHECKS_TABLE",
    "SCORE_CHECK_COLUMNS",
    "SELECTED_ENCODING_CONSTANT",
    "STAGE_NAME",
    "SUPER_OVER_WIN",
    "TIE",
    "UNPARSED",
    "UNPARSED_RESULTS_FILE",
    "WIN_BY_RUNS",
    "WIN_BY_WICKETS",
    "WITHIN_TOL_COLUMNS",
    "balls_table",
    "candidate_team_runs",
    "check_mapping",
    "classify",
    "competition_types",
    "delivery_scheduled_balls",
    "dismissed_batter_flags",
    "encoding_inputs",
    "exact_pass_rates",
    "excluded_flags",
    "exclusion_conditions",
    "exclusion_counts",
    "failing_matches",
    "free_hit_flags",
    "innings_aggregates",
    "innings_info",
    "match_columns",
    "order_deliveries",
    "over_check",
    "parse_result",
    "parse_results",
    "pass_rate_table",
    "run_checks",
    "run_stage",
    "scheduled_balls",
    "select_encoding",
    "super_over_flags",
    "targets",
    "wicket_flags",
    "within_tolerance_pass_rates",
    "write_failing_matches",
    "write_pass_rates",
    "write_reconstruction_audit",
    "write_unparsed",
]


# ------------------------------------------------------------------------- Delivery classification

class DeliveryClass(IntEnum):
    """Class of a Delivery (Requirement 5, criterion 2), stored as the int8 ``cls`` column of ``balls``."""

    LEGAL = 0  # Legal_Ball
    WIDE = 1  # Wide
    NO_BALL = 2  # No_Ball


#: Key of each DeliveryClass in the Config class mapping ``data_rules.extras_type_classes``.
CLASS_KEYS: dict[DeliveryClass, str] = {
    DeliveryClass.LEGAL: "legal",
    DeliveryClass.WIDE: "wide",
    DeliveryClass.NO_BALL: "no_ball",
}

_MAPPING_NAME = "Config class mapping data_rules.extras_type_classes"
_MISSING_TEXT = "<missing>"


class DeliveryClassError(DataError):
    """The Config class mapping or the ``extras_type`` data stops Delivery classification (exit code 3).

    ``duplicated`` maps each value that the mapping lists under more than one class to those
    class keys, in class order. ``unexpected`` maps each ``extras_type`` value that the mapping
    does not list to its row count, largest count first. In ``unexpected`` the key ``None``
    stands for a missing value, while the text ``"None"`` is an ordinary value. Both are empty
    when the only problem is the shape of the mapping, such as a missing ``no_ball`` class.
    """

    def __init__(
        self,
        message: str,
        *,
        duplicated: Mapping[str, Sequence[str]] | None = None,
        unexpected: Mapping[Any, int] | None = None,
    ) -> None:
        super().__init__(message)
        self.duplicated: dict[str, list[str]] = {
            value: list(keys) for value, keys in (duplicated or {}).items()
        }
        self.unexpected: dict[Any, int] = dict(unexpected or {})


class _MappingCheck(NamedTuple):
    """The Config class mapping as read by _read_mapping."""

    lookup: dict[str, DeliveryClass]  # each listed text -> its class (the first one, if duplicated)
    listed: dict[str, list[str]]  # class key -> its texts in Config order, for each usable class
    duplicated: dict[str, list[str]]  # text -> class keys, for each text under more than one class
    problems: list[str]  # one line per problem; empty when the mapping can be used


def _read_mapping(classes: Mapping[str, Sequence[str]]) -> _MappingCheck:
    """Read the class mapping, collecting every problem instead of stopping at the first."""
    keys = list(CLASS_KEYS.values())
    if not isinstance(classes, Mapping):
        raise DeliveryClassError(
            f"The {_MAPPING_NAME} must map the classes {', '.join(keys)} to lists of strings; "
            f"got {type(classes).__name__}"
        )
    problems: list[str] = []
    absent = [key for key in keys if key not in classes]
    extra = [key for key in classes if key not in keys]
    if absent or extra:
        found = []
        if absent:
            found.append("missing " + ", ".join(absent))
        if extra:
            found.append("not a class: " + ", ".join(map(repr, extra)))
        problems.append(f"the classes must be exactly {', '.join(keys)} ({'; '.join(found)})")

    lookup: dict[str, DeliveryClass] = {}
    listed: dict[str, list[str]] = {}
    classes_of: dict[str, list[str]] = {}
    for delivery_class, key in CLASS_KEYS.items():
        if key not in classes:
            continue
        values = classes[key]
        if not isinstance(values, (list, tuple)):
            problems.append(f"class {key} must be a list of strings; got {type(values).__name__} {values!r}")
            continue
        non_text = [value for value in values if not isinstance(value, str)]
        if non_text:
            problems.append(f"class {key} lists values that are not strings: {', '.join(map(repr, non_text))}")
        listed[key] = [str(value) for value in values if isinstance(value, str)]
        for value in listed[key]:
            lookup.setdefault(value, delivery_class)
            if key not in classes_of.setdefault(value, []):
                classes_of[value].append(key)
    duplicated = {value: found for value, found in classes_of.items() if len(found) > 1}
    problems += [
        f"{value!r} is listed under more than one class: {', '.join(found)}"
        for value, found in duplicated.items()
    ]
    return _MappingCheck(lookup, listed, duplicated, problems)


def _show(value: Any) -> str:
    """Display text of an extras_type value: its repr, or <missing> for a missing value."""
    return _MISSING_TEXT if value is None else repr(value)


def _rows(count: int) -> str:
    return f"{count:,} row" if count == 1 else f"{count:,} rows"


def _mapping_report(mapping: _MappingCheck) -> str:
    lines = [f"The {_MAPPING_NAME} has {len(mapping.problems)} problem(s):"]
    lines += [f"  - {problem}" for problem in mapping.problems]
    return "\n".join(lines)


def _values_report(unexpected: Mapping[Any, int], n_rows: int, listed: Mapping[str, list[str]]) -> str:
    lines = [
        f"extras_type is missing, or holds a value that the {_MAPPING_NAME} does not list, "
        f"in {_rows(sum(unexpected.values()))} of {n_rows:,}:"
    ]
    lines += [f"  - {_show(value)}: {_rows(count)}" for value, count in unexpected.items()]
    shown = "; ".join(f"{key}: {', '.join(map(repr, texts)) or 'no values'}" for key, texts in listed.items())
    lines.append(f"The mapping lists {shown or 'no values'}.")
    return "\n".join(lines)


def check_mapping(classes: Mapping[str, Sequence[str]]) -> None:
    """Stop the Run unless the Config class mapping puts every listed value in exactly one class.

    ``classes`` is the Config value ``data_rules.extras_type_classes``: the keys ``legal``,
    ``wide`` and ``no_ball``, each holding a list of ``extras_type`` texts. Raises
    DeliveryClassError (a DataError, exit code 3) that names every value listed under more than
    one class, with those classes (Requirement 5, criterion 3). The same error also names any
    class key that is missing or not allowed, and any class whose value is not a list of strings.
    """
    mapping = _read_mapping(classes)
    if mapping.problems:
        raise DeliveryClassError(_mapping_report(mapping), duplicated=mapping.duplicated)


def classify(extras_type: pd.Series, classes: Mapping[str, Sequence[str]]) -> pd.Series:
    """DeliveryClass code of each Delivery, from its ``extras_type`` (Requirement 5, criterion 2).

    ``classes`` is the Config value ``data_rules.extras_type_classes``. Each value matches as
    exact, case-sensitive text with no trimming, so ``"Wide"`` and ``" wide"`` do not match
    ``"wide"``. The text ``"None"`` is a value; a missing entry (None, NaN or pd.NA) is not.
    ``extras_type`` may be an object, ``str``, ``string`` or categorical Series.

    Returns an int8 Series named ``cls`` on the index of ``extras_type``, holding 0 (Legal_Ball),
    1 (Wide) or 2 (No_Ball).

    Raises DeliveryClassError (a DataError, exit code 3) when any ``extras_type`` is missing or
    holds a value that the mapping does not list, naming each such value with its row count
    (Requirement 5, criterion 3). The mapping is checked as in check_mapping; a problem with it
    is reported in the same error as the unlisted values.
    """
    mapping = _read_mapping(classes)
    # One lookup per distinct value (7 as delivered); code -1 marks a missing entry.
    codes, uniques = pd.factorize(extras_type, use_na_sentinel=True)
    values = [str(value) if isinstance(value, str) else value for value in uniques]
    class_of_unique = np.array(
        [mapping.lookup.get(value, -1) if isinstance(value, str) else -1 for value in values],
        dtype=np.int8,
    )
    present = codes >= 0
    cls = np.full(len(codes), -1, dtype=np.int8)
    cls[present] = class_of_unique[codes[present]]

    unexpected: dict[Any, int] = {}
    n_missing = len(codes) - int(np.count_nonzero(present))
    if n_missing:
        unexpected[None] = n_missing
    counts = np.bincount(codes[present], minlength=len(values))
    for value, code, count in zip(values, class_of_unique, counts):
        if code < 0 and count > 0:
            unexpected[value] = int(count)
    if unexpected or mapping.problems:
        unexpected = dict(sorted(unexpected.items(), key=lambda item: (-item[1], _show(item[0]))))
        report = [_mapping_report(mapping)] if mapping.problems else []
        if unexpected:
            report.append(_values_report(unexpected, len(codes), mapping.listed))
        raise DeliveryClassError("\n".join(report), duplicated=mapping.duplicated, unexpected=unexpected)
    return pd.Series(cls, index=extras_type.index, name="cls")


# ---------------------------------------------------------------------------------- shared helpers

class _Whole(NamedTuple):
    """One column read as whole numbers by _whole_numbers."""

    values: np.ndarray  # int64; 0 where the value is missing
    missing: np.ndarray  # bool; True where the value is missing


def _whole_numbers(
    frame: pd.DataFrame,
    columns: Sequence[str],
    context: str,
    *,
    may_be_missing: Collection[str] = (),
) -> dict[str, _Whole]:
    """Read ``columns`` of ``frame`` as whole numbers, collecting every problem before stopping.

    A value that is not a finite whole number is a problem in every column; a missing value is a
    problem unless its column is in ``may_be_missing``. Raises DataError (exit code 3) whose
    message starts with ``context`` and lists each problem with its row count.
    """
    found: dict[str, _Whole] = {}
    problems: list[str] = []
    for column in columns:
        series = frame[column]
        missing = series.isna().to_numpy(dtype=bool)
        numbers = pd.to_numeric(series, errors="coerce")
        if pd.api.types.is_integer_dtype(numbers.dtype):
            values = numbers.to_numpy(dtype=np.int64, na_value=0)
            bad = np.zeros(len(series), dtype=bool)
        else:
            floats = numbers.to_numpy(dtype=np.float64, na_value=np.nan)
            with np.errstate(invalid="ignore"):
                whole = np.isfinite(floats) & (np.floor(floats) == floats) & (np.abs(floats) <= 2.0**53)
            bad = ~missing & ~whole
            values = np.where(whole, floats, 0.0).astype(np.int64)
        if bad.any():
            examples = ", ".join(repr(value) for value in series.iloc[np.flatnonzero(bad)[:3]].tolist())
            problems.append(f"{column} is not a whole number in {_rows(int(bad.sum()))} (for example {examples})")
        if missing.any() and column not in may_be_missing:
            problems.append(f"{column} is missing in {_rows(int(missing.sum()))}")
        found[column] = _Whole(values, missing)
    if problems:
        raise DataError("\n".join([f"{context}:"] + [f"  - {problem}" for problem in problems]))
    return found


def _require_positive(found: Mapping[str, _Whole], match_ids: np.ndarray, context: str) -> None:
    """Stop the Run (DataError) when a present value of any column in ``found`` is 0 or negative."""
    problems = []
    for column, whole in found.items():
        bad = ~whole.missing & (whole.values <= 0)
        if bad.any():
            shown = ", ".join(
                f"match {match}: {value}"
                for match, value in zip(match_ids[bad][:3].tolist(), whole.values[bad][:3].tolist())
            )
            problems.append(f"{column} is not a positive number in {_rows(int(bad.sum()))} ({shown})")
    if problems:
        raise DataError("\n".join([f"{context}:"] + [f"  - {problem}" for problem in problems]))


def _match_ids(matches: pd.DataFrame, context: str) -> np.ndarray:
    """The ``match_id`` of every row of ``matches``; DataError when one is missing, non-whole or repeated."""
    ids = _whole_numbers(matches, ["match_id"], context)["match_id"].values
    unique, counts = np.unique(ids, return_counts=True)
    repeated = unique[counts > 1]
    if len(repeated):
        shown = ", ".join(map(str, repeated[:5].tolist()))
        raise DataError(
            f"{context}:\n  - match_id is repeated in matches for {len(repeated):,} match(es) (for example {shown})"
        )
    return ids


def _check_alignment(values: Any, index: pd.Index, name: str) -> np.ndarray | None:
    """Raise ValueError unless ``values`` has one entry per row of the Delivery table.

    A Series must be on the Delivery table's index; any other array-like is taken in row order.
    Returns None for a Series (the caller converts it) and the numpy array otherwise.
    """
    if isinstance(values, pd.Series):
        if not values.index.equals(index):
            raise ValueError(
                f"{name} must be a Series on the index of the Delivery table, or an array in its row order"
            )
        return None
    array = np.asarray(values)
    if array.ndim != 1 or len(array) != len(index):
        raise ValueError(f"{name} must hold one value per Delivery ({len(index):,}); got shape {array.shape}")
    return array


def _flag_array(values: Any, index: pd.Index, name: str) -> np.ndarray:
    """Boolean flags aligned with the Delivery table; a missing entry of a nullable flag is False."""
    array = _check_alignment(values, index, name)
    if array is None:
        if not pd.api.types.is_bool_dtype(values.dtype):
            raise TypeError(f"{name} must be boolean; got dtype {values.dtype}")
        return values.to_numpy(dtype=bool, na_value=False)
    if array.dtype != np.bool_:
        raise TypeError(f"{name} must be boolean; got dtype {array.dtype}")
    return array


def _int_array(values: Any, index: pd.Index, name: str) -> np.ndarray:
    """int64 values aligned with the Delivery table, with no missing entry."""
    array = _check_alignment(values, index, name)
    if array is None:
        if not pd.api.types.is_integer_dtype(values.dtype) or values.isna().any():
            raise TypeError(f"{name} must hold integers with no missing value; got dtype {values.dtype}")
        return values.to_numpy(dtype=np.int64)
    if array.dtype.kind not in "iu":
        raise TypeError(f"{name} must hold integers; got dtype {array.dtype}")
    return array.astype(np.int64, copy=False)


def _class_codes(cls: Any, index: pd.Index) -> np.ndarray:
    """DeliveryClass codes (as returned by classify) aligned with the Delivery table, as int8."""
    codes = _int_array(cls, index, "cls")
    if not np.isin(codes, [int(member) for member in DeliveryClass]).all():
        raise ValueError(
            "cls must hold DeliveryClass codes 0 (Legal_Ball), 1 (Wide) or 2 (No_Ball), as classify returns"
        )
    return codes.astype(np.int8)


def _shift_down(values: np.ndarray, fill: int) -> np.ndarray:
    """``values`` moved one row down, with ``fill`` in the first row: each row sees the row before it."""
    shifted = np.empty_like(values)
    if len(values):
        shifted[0] = fill
        shifted[1:] = values[:-1]
    return shifted


# ------------------------------------------------------------- Ball order and Super_Over_Innings

#: Columns that set the Requirement 5 order of Deliveries (criterion 1), most significant first.
ORDER_KEYS: tuple[str, str, str] = ("match_id", "innings", "innings_ball_num")

_FIRST_SUPER_OVER_INNINGS = 3  # innings 3 and 4 are the Super_Over_Innings (glossary; Assumption 5)


def _order_keys(frame: pd.DataFrame, context: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The ORDER_KEYS columns as int64 arrays; DataError when a key is missing or not a whole number."""
    keys = _whole_numbers(frame, ORDER_KEYS, context)
    return keys["match_id"].values, keys["innings"].values, keys["innings_ball_num"].values


def _sort_order(match_id: np.ndarray, innings: np.ndarray, ball: np.ndarray) -> np.ndarray:
    """Row positions in the Requirement 5 order. np.lexsort is stable, so tied keys keep their input order."""
    return np.lexsort((ball, innings, match_id))


def _first_of_innings(match_id: np.ndarray, innings: np.ndarray) -> np.ndarray:
    """True at the first row of each (match_id, innings) group, for keys already in the Requirement 5 order."""
    first = np.ones(len(match_id), dtype=bool)
    first[1:] = (match_id[1:] != match_id[:-1]) | (innings[1:] != innings[:-1])
    return first


def order_deliveries(balls: pd.DataFrame) -> pd.DataFrame:
    """The Deliveries in the Requirement 5 order, numbered by ``row`` (criterion 1).

    Sorts by ``match_id``, then ``innings``, then ``innings_ball_num``, each in ascending numeric
    order. The sort is stable, so rows with equal keys keep their input order. Returns a new
    DataFrame with a RangeIndex, a first column ``row`` (int32, 0 to n - 1, the global Delivery
    position and the join key of every per-Delivery table) and every other column of ``balls``
    unchanged. A ``row`` column already in ``balls`` is replaced.

    Raises DataError (exit code 3) when an order key is missing or not a whole number, naming each
    key with its row count.
    """
    match_id, innings, ball = _order_keys(balls, "Deliveries cannot be put in the Requirement 5 order")
    ordered = balls.iloc[_sort_order(match_id, innings, ball)].reset_index(drop=True)
    if "row" in ordered.columns:
        ordered = ordered.drop(columns="row")
    ordered.insert(0, "row", np.arange(len(ordered), dtype=np.int32))
    return ordered


def super_over_flags(innings: pd.Series) -> pd.Series:
    """``is_super_over`` of each Delivery: True in a Super_Over_Innings, innings 3 or 4 (criterion 11).

    Any innings number above 4 (the data holds none) is flagged as well. Every model-training,
    pressure and rating filter drops the flagged Deliveries. Returns a bool Series named
    ``is_super_over`` on the index of ``innings``. Raises DataError when an innings number is
    missing or not a whole number.
    """
    values = _whole_numbers(innings.to_frame("innings"), ["innings"], "Super_Over_Innings cannot be flagged")
    return pd.Series(values["innings"].values >= _FIRST_SUPER_OVER_INNINGS, index=innings.index, name="is_super_over")


# --------------------------------------------------------------------------------- Scheduled_Balls

#: Scheduled_Balls of an innings to which no revised-overs value applies (criterion 6): a full
#: T20 innings of 20 Overs of 6 balls, fixed by the format rather than a Method_Constant.
DEFAULT_SCHEDULED_BALLS = 120

#: The revised-overs column of innings 1 and of innings 2. Both hold counts of balls (criteria 4-5).
_REVISED_OVERS: dict[int, str] = {1: "first_inn_revised_overs", 2: "second_inn_revised_overs"}


def scheduled_balls(matches: pd.DataFrame) -> pd.DataFrame:
    """Scheduled_Balls of innings 1 and 2 of every match (criteria 4-6).

    ``matches`` needs ``match_id``, ``first_inn_revised_overs`` and ``second_inn_revised_overs``;
    each revised value is read as a count of balls. Innings 1 takes ``first_inn_revised_overs``.
    Innings 2 takes ``second_inn_revised_overs``, or ``first_inn_revised_overs`` when
    ``second_inn_revised_overs`` is missing. An innings with neither value gets
    DEFAULT_SCHEDULED_BALLS (120). So match 50408 gets 54, and innings 2 of matches 16753 and
    19959 gets 66 through the fallback.

    Returns one row per match and innings, sorted by ``match_id`` then ``innings``, with int32
    columns ``match_id``, ``innings`` (1 or 2) and ``sched_balls``. Super_Over_Innings have no row:
    delivery_scheduled_balls gives them 120.

    Raises DataError (exit code 3) when a ``match_id`` is missing, non-whole or repeated, or when a
    revised value is present but not a positive whole number.
    """
    context = "Scheduled_Balls cannot be set"
    ids = _match_ids(matches, context)
    revised = _whole_numbers(matches, list(_REVISED_OVERS.values()), context, may_be_missing=_REVISED_OVERS.values())
    _require_positive(revised, ids, context)
    first, second = revised[_REVISED_OVERS[1]], revised[_REVISED_OVERS[2]]
    innings_1 = np.where(first.missing, DEFAULT_SCHEDULED_BALLS, first.values)
    innings_2 = np.where(second.missing, innings_1, second.values)  # fallback: innings 1's value, else 120
    order = np.argsort(ids, kind="stable")
    return pd.DataFrame(
        {
            "match_id": np.repeat(ids[order], 2).astype(np.int32),
            "innings": np.tile(np.array([1, 2], dtype=np.int32), len(ids)),
            "sched_balls": np.column_stack((innings_1[order], innings_2[order])).reshape(-1).astype(np.int32),
        }
    )


def delivery_scheduled_balls(balls: pd.DataFrame, sched: pd.DataFrame) -> pd.Series:
    """Scheduled_Balls of each Delivery's innings, the ``sched_balls`` column of ``balls``.

    ``balls`` needs ``match_id`` and ``innings``; ``sched`` is the scheduled_balls table. A
    Delivery whose (``match_id``, ``innings``) has a row in ``sched`` takes that row's value. A
    Super_Over_Innings has no revised-overs value, so it gets DEFAULT_SCHEDULED_BALLS (120) under
    criterion 6. Returns an int32 Series named ``sched_balls`` on the index of ``balls``.

    Raises DataError (exit code 3) when a Delivery of innings 1 or 2 belongs to a match that has
    no row in ``sched`` (a match missing from ``matches``), giving the count and example matches.
    """
    context = "Scheduled_Balls cannot be given to the Deliveries"
    keys = _whole_numbers(balls, ["match_id", "innings"], context)
    match_id, innings = keys["match_id"].values, keys["innings"].values
    table = _whole_numbers(sched, ["match_id", "innings", "sched_balls"], "The Scheduled_Balls table cannot be used")
    lookup = pd.MultiIndex.from_arrays([table["match_id"].values, table["innings"].values])
    if not lookup.is_unique:
        raise ValueError("sched must hold one row per (match_id, innings), as scheduled_balls returns")
    position = lookup.get_indexer(pd.MultiIndex.from_arrays([match_id, innings]))
    found = position >= 0
    values = np.full(len(position), DEFAULT_SCHEDULED_BALLS, dtype=np.int64)
    values[found] = table["sched_balls"].values[position[found]]
    unmatched = ~found & (innings < _FIRST_SUPER_OVER_INNINGS)
    if unmatched.any():
        shown = ", ".join(map(str, np.unique(match_id[unmatched])[:5].tolist()))
        raise DataError(
            f"{context}:\n  - innings 1 or 2 of a match with no Scheduled_Balls row (the match is not in "
            f"matches) holds {_rows(int(unmatched.sum()))} of the Deliveries (for example match {shown})"
        )
    return pd.Series(values.astype(np.int32), index=balls.index, name="sched_balls")


# -------------------------------------------------------------------------------------- Free_Hits

def free_hit_flags(order_keys: pd.DataFrame, cls: pd.Series) -> pd.Series:
    """Free_Hit flag of each Delivery (criterion 12).

    ``order_keys`` holds the ORDER_KEYS columns of the Deliveries in any row order (usually the
    whole Delivery table); ``cls`` holds their DeliveryClass codes, as classify returns, on the
    same index. In the Requirement 5 order, a Delivery is a Free_Hit when it follows a No_Ball in
    the same innings, up to and including the next Legal_Ball. Wides and further No_Balls inside
    that span are flagged too, and a No_Ball is flagged only when it follows another No_Ball. A
    No_Ball at the end of an innings flags nothing in the next innings.

    Vectorised: in sorted order, forward-fill the position of the last No_Ball and of the last
    Legal_Ball up to each row (running maxima), shift both down one row so each Delivery sees only
    earlier Deliveries, and flag the Delivery when that last No_Ball lies in its innings and comes
    after that last Legal_Ball.

    Returns a bool Series named ``free_hit`` on the index of ``order_keys``. Raises DataError when
    an order key is missing or not a whole number.
    """
    match_id, innings, ball = _order_keys(order_keys, "Free_Hits cannot be flagged")
    codes = _class_codes(cls, order_keys.index)
    order = _sort_order(match_id, innings, ball)
    sorted_codes = codes[order]
    position = np.arange(len(order), dtype=np.int64)
    innings_start = np.maximum.accumulate(np.where(_first_of_innings(match_id[order], innings[order]), position, 0))
    last_no_ball = _shift_down(np.maximum.accumulate(np.where(sorted_codes == DeliveryClass.NO_BALL, position, -1)), -1)
    last_legal = _shift_down(np.maximum.accumulate(np.where(sorted_codes == DeliveryClass.LEGAL, position, -1)), -1)
    flags = np.empty(len(order), dtype=bool)
    flags[order] = (last_no_ball >= innings_start) & (last_no_ball > last_legal)
    return pd.Series(flags, index=order_keys.index, name="free_hit")


# ----------------------------------------------------------------------------------- Wicket flags

def _code_array(codes: Collection[int], name: str) -> np.ndarray:
    """Dismissal codes as a sorted int64 array; TypeError for anything but integers."""
    values = list(codes)
    if any(isinstance(code, bool) or not isinstance(code, (int, np.integer)) for code in values):
        raise TypeError(f"{name} must hold integer dismissal codes; got {values!r}")
    return np.array(sorted({int(code) for code in values}), dtype=np.int64)


def wicket_flags(
    balls: pd.DataFrame,
    counted_codes: Collection[int],
    bowler_wicket_codes: Collection[int],
    run_out_code: int,
    retire_out_code: int,
) -> pd.DataFrame:
    """Counted_Wicket, Bowler_Wicket, Run_Out and Retire Out flags of each Delivery (criterion 8; D14).

    ``balls`` needs ``is_wicket`` (boolean) and ``dismissal_type_id`` (whole numbers, missing on
    non-wicket rows). ``counted_codes`` are the dismissal codes that the Auditor classifies as
    Counted_Wickets (``classify_dismissal_codes``), Retire Out included and Retired_Hurt codes
    excluded. ``bowler_wicket_codes``, ``run_out_code`` and ``retire_out_code`` are the Config
    values ``design_choices.D14_bowler_wicket_codes``, ``D14_run_out_code`` and
    ``D14_retire_out_code``.

    Returns a DataFrame on the index of ``balls`` with bool columns:

    * ``counted_wicket``: ``is_wicket`` true and the code is a Counted_Wicket code. Each such
      Delivery adds exactly 1 to its innings' wickets fallen (innings_aggregates).
    * ``bowler_wicket``: a Counted_Wicket whose code is a D14 bowler code (the bowler's credit).
    * ``run_out``: a Counted_Wicket with the D14 run-out code, whichever batter is out.
    * ``retire_out``: a Counted_Wicket with the D14 Retire Out code.

    A missing ``is_wicket`` counts as false. Raises DataError when a present ``dismissal_type_id``
    is not a whole number.
    """
    is_wicket = _flag_array(balls["is_wicket"], balls.index, "is_wicket")
    code = _whole_numbers(
        balls, ["dismissal_type_id"], "Wickets cannot be counted", may_be_missing=["dismissal_type_id"]
    )["dismissal_type_id"]
    run_out = int(_code_array([run_out_code], "run_out_code")[0])
    retire_out = int(_code_array([retire_out_code], "retire_out_code")[0])
    counted = is_wicket & ~code.missing & np.isin(code.values, _code_array(counted_codes, "counted_codes"))
    return pd.DataFrame(
        {
            "counted_wicket": counted,
            "bowler_wicket": counted & np.isin(code.values, _code_array(bowler_wicket_codes, "bowler_wicket_codes")),
            "run_out": counted & (code.values == run_out),
            "retire_out": counted & (code.values == retire_out),
        },
        index=balls.index,
    )


# ------------------------------------------------------------------------------------- Team runs

#: The ``extras_type`` values of a Legal_Ball scored as a bye or leg-bye by the Extras_Encodings.
BYE_EXTRAS_TYPES: tuple[str, ...] = ("byes", "legbyes")

#: The ``extras_type`` value of a Legal_Ball scored as a penalty award by the Extras_Encodings.
PENALTY_EXTRAS_TYPE = "Penalty"

#: Name of the F30_Encoding (E1) in ENCODINGS and ``data_rules.candidate_encodings``.
F30_ENCODING = "F30"

_WIDE_NO_BALL_PENALTY = 1  # one-run penalty of a Wide or No_Ball (Laws of Cricket)
_PENALTY_AWARD = 5  # five penalty runs (Laws of Cricket); used by E4 only
_RUNS_PER_BYE = 1  # E2-E4: runs counted for a bye or leg-bye Delivery, whose run count is not recorded (F-4)


class EncodingInputs(NamedTuple):
    """Per-Delivery inputs of an Extras_Encoding, as numpy arrays in the row order of the Delivery table."""

    bat_runs: np.ndarray  # int64, b: ``runs``
    extras_runs: np.ndarray  # int64, e: ``extras_2_runs``
    cls: np.ndarray  # int8 DeliveryClass code
    is_bye: np.ndarray  # bool: Legal_Ball whose extras_type is in BYE_EXTRAS_TYPES
    is_penalty: np.ndarray  # bool: Legal_Ball whose extras_type is PENALTY_EXTRAS_TYPE


def encoding_inputs(balls: pd.DataFrame, cls: pd.Series) -> EncodingInputs:
    """The EncodingInputs of every Delivery.

    ``balls`` needs ``runs``, ``extras_2_runs`` and ``extras_type`` (the raw text, which tells a
    plain Legal_Ball from a bye, leg-bye or Penalty Delivery). ``cls`` holds the DeliveryClass
    codes from classify on the index of ``balls``; the class decides Wide and No_Ball, and the
    raw text only splits the Legal_Balls.

    Raises DataError (exit code 3) when ``runs`` or ``extras_2_runs`` is missing or not a whole
    number, naming each column with its row count.
    """
    numbers = _whole_numbers(balls, ["runs", "extras_2_runs"], "Team runs cannot be computed")
    codes = _class_codes(cls, balls.index)
    legal = codes == DeliveryClass.LEGAL
    extras_type = balls["extras_type"]
    return EncodingInputs(
        bat_runs=numbers["runs"].values,
        extras_runs=numbers["extras_2_runs"].values,
        cls=codes,
        is_bye=legal & extras_type.isin(BYE_EXTRAS_TYPES).to_numpy(dtype=bool),
        is_penalty=legal & extras_type.isin([PENALTY_EXTRAS_TYPE]).to_numpy(dtype=bool),
    )


def _team_runs(x: EncodingInputs, *, byes_as_one: bool, no_ball_extras: bool, penalty_award: bool) -> np.ndarray:
    """Team runs per Delivery for one row of the encoding table in the module docstring, as int32."""
    b, e = x.bat_runs, x.extras_runs
    runs = b + e  # plain Legal_Ball; also a bye or leg-bye unless byes_as_one, and Penalty unless penalty_award
    if byes_as_one:
        runs = np.where(x.is_bye, _RUNS_PER_BYE + e, runs)
    if penalty_award:
        runs = np.where(x.is_penalty, _PENALTY_AWARD + b + e, runs)
    runs = np.where(x.cls == DeliveryClass.WIDE, _WIDE_NO_BALL_PENALTY + e, runs)
    no_ball_runs = _WIDE_NO_BALL_PENALTY + b + (e if no_ball_extras else 0)
    runs = np.where(x.cls == DeliveryClass.NO_BALL, no_ball_runs, runs)
    return runs.astype(np.int32)


def _f30(x: EncodingInputs) -> np.ndarray:
    """E1, the F30_Encoding: b + e, except Wide 1 + e and No_Ball 1 + b + e."""
    return _team_runs(x, byes_as_one=False, no_ball_extras=True, penalty_award=False)


def _byes_one(x: EncodingInputs) -> np.ndarray:
    """E2: as E1, but a bye or leg-bye Delivery scores 1 + e."""
    return _team_runs(x, byes_as_one=True, no_ball_extras=True, penalty_award=False)


def _byes_one_nb_plain(x: EncodingInputs) -> np.ndarray:
    """E3: as E2, but a No_Ball scores 1 + b (``extras_2_runs`` taken as already inside ``runs``)."""
    return _team_runs(x, byes_as_one=True, no_ball_extras=False, penalty_award=False)


def _byes_one_penalty5(x: EncodingInputs) -> np.ndarray:
    """E4: as E2, but a Penalty Delivery scores 5 + b + e."""
    return _team_runs(x, byes_as_one=True, no_ball_extras=True, penalty_award=True)


#: The candidate Extras_Encodings E1-E4 by Config name, in pass-rate table order with E1 (the
#: F30_Encoding) first. Each function returns the int32 team runs of every Delivery.
ENCODINGS: dict[str, Callable[[EncodingInputs], np.ndarray]] = {
    F30_ENCODING: _f30,
    "byes_one": _byes_one,
    "byes_one_nb_plain": _byes_one_nb_plain,
    "byes_one_penalty5": _byes_one_penalty5,
}


def candidate_team_runs(inputs: EncodingInputs, names: Sequence[str]) -> dict[str, np.ndarray]:
    """Team runs of every Delivery under each candidate Extras_Encoding in ``names``.

    ``names`` is the Config value ``data_rules.candidate_encodings``. Returns a dict in the order
    of ``names``, from each name to the int32 array that ``ENCODINGS[name]`` gives. Raises
    StageError (exit code 5) when a name is not in ENCODINGS or is listed more than once.
    """
    unknown = [name for name in names if name not in ENCODINGS]
    repeated = sorted({name for name in names if list(names).count(name) > 1})
    if unknown or repeated:
        found = []
        if unknown:
            found.append("not an Extras_Encoding: " + ", ".join(map(repr, unknown)))
        if repeated:
            found.append("listed more than once: " + ", ".join(map(repr, repeated)))
        raise StageError(
            f"data_rules.candidate_encodings must name Extras_Encodings from {', '.join(ENCODINGS)} "
            f"at most once each ({'; '.join(found)})"
        )
    return {name: ENCODINGS[name](inputs) for name in names}


# -------------------------------------------------------------------- Innings aggregates and Targets

def innings_aggregates(balls: pd.DataFrame, team_runs: np.ndarray, counted: pd.Series) -> pd.DataFrame:
    """Total, wickets fallen and final-Delivery team runs of every innings under one Extras_Encoding.

    ``balls`` needs the ORDER_KEYS columns and ``batting_team_id``, in any row order.
    ``team_runs`` holds the team runs of each row of ``balls`` under one encoding (an ENCODINGS
    result, in the row order of ``balls``). ``counted`` holds the Counted_Wicket flag of each row
    (the ``counted_wicket`` column of wicket_flags, on the index of ``balls``).

    Returns one row per (``match_id``, ``innings``) present in ``balls``, sorted by ``match_id``
    then ``innings``, with columns:

    * ``match_id``, ``innings`` (int32);
    * ``batting_team_id`` (nullable Int32): the batting team of the innings' first Delivery;
    * ``total`` (int32): the sum of the innings' team runs;
    * ``wickets`` (int32): the number of the innings' Counted_Wickets (criterion 8);
    * ``final_ball_runs`` (int32): the team runs of the innings' last Delivery in the
      Requirement 5 order.

    Raises DataError when an order key is missing or not a whole number, or when a present
    ``batting_team_id`` is not a whole number.
    """
    context = "Innings totals cannot be computed"
    match_id, innings, ball = _order_keys(balls, context)
    runs = _int_array(team_runs, balls.index, "team_runs")
    flags = _flag_array(counted, balls.index, "counted")
    batting = _whole_numbers(balls, ["batting_team_id"], context, may_be_missing=["batting_team_id"])["batting_team_id"]
    order = _sort_order(match_id, innings, ball)
    starts = np.flatnonzero(_first_of_innings(match_id[order], innings[order]))
    if len(starts) == 0:
        totals = wickets = np.zeros(0, dtype=np.int64)
        final_runs = np.zeros(0, dtype=np.int64)
    else:
        ends = np.append(starts[1:], len(order)) - 1
        sorted_runs = runs[order]
        totals = np.add.reduceat(sorted_runs, starts)
        wickets = np.add.reduceat(flags[order].astype(np.int64), starts)
        final_runs = sorted_runs[ends]
    first_rows = order[starts]
    return pd.DataFrame(
        {
            "match_id": match_id[first_rows].astype(np.int32),
            "innings": innings[first_rows].astype(np.int32),
            "batting_team_id": pd.arrays.IntegerArray(
                batting.values[first_rows].astype(np.int32), batting.missing[first_rows]
            ),
            "total": totals.astype(np.int32),
            "wickets": wickets.astype(np.int32),
            "final_ball_runs": final_runs.astype(np.int32),
        }
    )


def targets(matches: pd.DataFrame, agg: pd.DataFrame) -> pd.Series:
    """Target of every match under one Extras_Encoding (criteria 9-10).

    ``matches`` needs ``match_id`` and ``revised_target``; ``agg`` is the innings_aggregates
    table of the encoding being evaluated. The Target is ``revised_target`` when it holds a value
    (criterion 9), else the first-innings total under that encoding plus 1 (criterion 10). It is
    missing for a match with neither, that is, with no revised target and no innings-1 Delivery.

    Returns a nullable Int32 Series named ``target``, indexed by ``match_id`` (sorted) with one
    entry per row of ``matches``. Raises DataError when a ``match_id`` is missing, non-whole or
    repeated, or when a ``revised_target`` is present but not a positive whole number.
    """
    context = "Targets cannot be set"
    ids = _match_ids(matches, context)
    revised = _whole_numbers(matches, ["revised_target"], context, may_be_missing=["revised_target"])
    _require_positive(revised, ids, context)
    revised_target = revised["revised_target"]
    first = _whole_numbers(agg.loc[agg["innings"] == 1], ["match_id", "total"], "The innings totals cannot be used")
    lookup = pd.Index(first["match_id"].values)
    if not lookup.is_unique:
        raise ValueError("agg must hold one row per (match_id, innings), as innings_aggregates returns")
    position = lookup.get_indexer(ids)
    use_first = revised_target.missing & (position >= 0)
    values = revised_target.values.copy()
    values[use_first] = first["total"].values[position[use_first]] + 1
    missing = revised_target.missing & (position < 0)
    order = np.argsort(ids, kind="stable")
    return pd.Series(
        pd.arrays.IntegerArray(values[order].astype(np.int32), missing[order]),
        index=pd.Index(ids[order].astype(np.int32), name="match_id"),
        name="target",
    )


# ----------------------------------------------------------------------------------- Result_Parser

#: Result types of the Result_Parser (criteria 13 and 15). Only WIN_BY_RUNS and WIN_BY_WICKETS
#: carry a margin.
SUPER_OVER_WIN = "super_over_win"
TIE = "tie"
ABANDONED = "abandoned"
NO_RESULT = "no_result"
WIN_BY_RUNS = "win_by_runs"
WIN_BY_WICKETS = "win_by_wickets"
UNPARSED = "unparsed"

#: Every result type: the six rule types in the criterion 13 rule order, then UNPARSED.
RESULT_TYPES: tuple[str, ...] = (SUPER_OVER_WIN, TIE, ABANDONED, NO_RESULT, WIN_BY_RUNS, WIN_BY_WICKETS, UNPARSED)

#: File name of the table of unparsed result strings in the audit folder (criterion 15).
UNPARSED_RESULTS_FILE = "unparsed_results.csv"

# The criterion 13 rules in order, applied to the normalised text: lower-cased, with each run of
# whitespace replaced by one space. Group 1 of the two margin rules is the margin (criterion 14).
_RESULT_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (SUPER_OVER_WIN, re.compile(r"super over|one over eliminator")),
    (TIE, re.compile(r"tied")),
    (ABANDONED, re.compile(r"abandoned")),
    (NO_RESULT, re.compile(r"no result")),
    (WIN_BY_RUNS, re.compile(r"(\d+)\s*run")),
    (WIN_BY_WICKETS, re.compile(r"(\d+)\s*(?:wicket|wkt)")),
)
_MARGIN_TYPES = frozenset({WIN_BY_RUNS, WIN_BY_WICKETS})
_DLS_PATTERN = re.compile(r"dls|d/l")
_WHITESPACE = re.compile(r"\s+")
_MARGIN_MAX = int(np.iinfo(np.int32).max)  # the largest margin the nullable Int32 margin column holds


class ParsedResult(NamedTuple):
    """What the Result_Parser reads from one ``result`` string (criteria 13-15)."""

    result_type: str  # one of RESULT_TYPES
    margin: int | None  # the margin of WIN_BY_RUNS or WIN_BY_WICKETS; None for every other type
    dls: bool  # True exactly when the text contains "DLS" or "D/L", in any letter case


_MISSING_RESULT = ParsedResult(UNPARSED, None, False)  # parse_result of a missing value


def parse_result(text: str | None) -> ParsedResult:
    """Result type, margin and DLS flag of one ``result`` string (criteria 13-15).

    The text is lower-cased and each run of whitespace becomes one space, so every rule matches
    in any letter case and treats repeated spaces as one. The DLS flag is true exactly when the
    text contains ``dls`` or ``d/l``. The result type comes from the first rule that matches:

    1. SUPER_OVER_WIN: ``super over`` or ``one over eliminator``, even when the text also states
       a margin, as in "India Won Super Over By 10 Runs";
    2. TIE: ``tied``;
    3. ABANDONED: ``abandoned``;
    4. NO_RESULT: ``no result``;
    5. WIN_BY_RUNS: an integer, zero or more spaces, then ``run``;
    6. WIN_BY_WICKETS: an integer, zero or more spaces, then ``wicket`` or ``wkt``.

    Under rules 5 and 6 the margin is the integer before the word where the rule first matches,
    with or without "by" before it: 17 for "won by 17runs (DLS Method)", 3 for "Won 3 Wkts" and
    44 for "won by 44 run". Every other type has no margin (None). Text that matches no rule is
    UNPARSED (criterion 15), and so is a missing value (None, NaN or pd.NA), whose DLS flag is
    false. The caller records the UNPARSED strings (write_unparsed).

    Raises TypeError when ``text`` is neither a string nor a missing value.
    """
    if not isinstance(text, str):
        if pd.api.types.is_scalar(text) and pd.isna(text):
            return _MISSING_RESULT
        raise TypeError(f"a result must be a string or a missing value; got {type(text).__name__} {text!r}")
    normalised = _WHITESPACE.sub(" ", text.lower())
    dls = _DLS_PATTERN.search(normalised) is not None
    for result_type, pattern in _RESULT_RULES:
        found = pattern.search(normalised)
        if found is not None:
            margin = int(found.group(1)) if result_type in _MARGIN_TYPES else None
            return ParsedResult(result_type, margin, dls)
    return ParsedResult(UNPARSED, None, dls)


def parse_results(matches: pd.DataFrame) -> pd.DataFrame:
    """parse_result of the ``result`` string of every match (criteria 13-15).

    ``matches`` needs ``match_id`` and ``result``. Each distinct ``result`` value is parsed once
    with parse_result, and a missing ``result`` is UNPARSED, so every row equals parse_result of
    its string.

    Returns one row per match, sorted by ``match_id``, with columns ``match_id`` (int32),
    ``result_type`` (str, one of RESULT_TYPES), ``margin`` (nullable Int32, missing unless the type
    is WIN_BY_RUNS or WIN_BY_WICKETS) and ``dls`` (bool). write_unparsed records its UNPARSED rows.

    Raises DataError (exit code 3) when a ``match_id`` is missing, non-whole or repeated, or when a
    margin is above 2,147,483,647 and so does not fit the ``margin`` column, naming the matches.
    """
    context = "Match results cannot be parsed"
    ids = _match_ids(matches, context)
    order = np.argsort(ids, kind="stable")
    results = matches["result"].iloc[order]
    codes, uniques = pd.factorize(results, use_na_sentinel=True)
    lookup = [parse_result(value) for value in uniques] + [_MISSING_RESULT]  # code -1 (missing) takes the last
    parsed = [lookup[code] for code in codes]
    margins = [row.margin for row in parsed]
    too_large = [p for p, margin in enumerate(margins) if margin is not None and margin > _MARGIN_MAX]
    if too_large:
        shown = ", ".join(f"match {ids[order[p]]}: {results.iloc[p]!r}" for p in too_large[:3])
        raise DataError(
            f"{context}:\n  - the margin is above {_MARGIN_MAX:,}, the largest the margin column holds, "
            f"in {_rows(len(too_large))} (for example {shown})"
        )
    return pd.DataFrame(
        {
            "match_id": ids[order].astype(np.int32),
            "result_type": pd.array([row.result_type for row in parsed], dtype="str"),
            "margin": pd.array(margins, dtype="Int32"),
            "dls": np.array([row.dls for row in parsed], dtype=bool),
        }
    )


def write_unparsed(outdir: str | Path, parsed: pd.DataFrame, matches: pd.DataFrame) -> Path:
    """Write ``unparsed_results.csv``, the UNPARSED matches, to the folder ``outdir`` (criterion 15).

    ``outdir`` is the audit folder (``outputs/audit/`` in a Run) and is created when absent.
    ``parsed`` is the parse_results table of ``matches``, which supplies each original ``result``.
    The file has the header ``match_id,result`` and one row per UNPARSED match, sorted by
    ``match_id``, with the ``result`` string exactly as read (an empty field for a missing
    ``result``); with no UNPARSED match it holds the header only. It is written as UTF-8 through
    a temporary file renamed in place, so it replaces any earlier copy whole.

    Returns the path of the file. Raises DataError (exit code 3) when a ``match_id`` of
    ``matches`` is missing, non-whole or repeated, and ValueError when an UNPARSED ``match_id`` of
    ``parsed`` is not in ``matches``.
    """
    ids = _match_ids(matches, "The unparsed result strings cannot be written")
    unparsed = np.sort(parsed.loc[parsed["result_type"] == UNPARSED, "match_id"].to_numpy(dtype=np.int64))
    position = pd.Index(ids).get_indexer(unparsed)
    if (position < 0).any():
        absent = ", ".join(map(str, unparsed[position < 0][:5].tolist()))
        raise ValueError(f"parsed must be the parse_results table of matches; match {absent} is not in matches")
    table = pd.DataFrame(
        {"match_id": unparsed, "result": matches["result"].iloc[position].to_numpy(dtype=object)}
    )
    folder = Path(outdir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / UNPARSED_RESULTS_FILE
    temporary = path.with_name(path.name + ".tmp")
    try:
        table.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


# ------------------------------------------------------------------ Score_Checks and pass rates

#: Names of the Score_Checks of criteria 16-20, as held by the ``check`` column of ``score_checks``.
CHECK_RUNS_MARGIN = "runs_margin"  # criterion 16
CHECK_CHASE = "chase"  # criterion 17
CHECK_WICKETS_MARGIN = "wickets_margin"  # criterion 18
CHECK_TIE = "tie"  # criterion 19
CHECK_OVER = "over"  # criterion 20

#: The Score_Checks in table order: the order of each match's rows in ``score_checks`` and of the
#: rate columns of the pass-rate table.
SCORE_CHECKS: tuple[str, ...] = (CHECK_RUNS_MARGIN, CHECK_CHASE, CHECK_WICKETS_MARGIN, CHECK_TIE, CHECK_OVER)

#: The Score_Checks that record a Runs_Check_Miss (criteria 16, 17 and 19).
RUNS_CHECKS: tuple[str, ...] = (CHECK_RUNS_MARGIN, CHECK_CHASE, CHECK_TIE)

#: Columns of the ``score_checks`` table that run_checks returns (design, Data Models).
SCORE_CHECK_COLUMNS: tuple[str, ...] = ("encoding", "match_id", "check", "applies", "passed", "runs_check_miss")

#: Column of each Score_Check's Exact_Pass_Rate in the exact_pass_rates and pass_rate_table tables.
EXACT_COLUMNS: dict[str, str] = {check: f"exact_{check}" for check in SCORE_CHECKS}

#: Column of each Score_Check's Within_Tolerance_Pass_Rate in the pass_rate_table table.
WITHIN_TOL_COLUMNS: dict[str, str] = {check: f"within_tol_{check}" for check in SCORE_CHECKS}

#: File names, in the audit folder, of the pass-rate table (criterion 21) and of the failed checks
#: of the selected Extras_Encoding (criterion 23).
PASS_RATES_FILE = "pass_rates.csv"
FAILING_MATCHES_FILE = "failing_matches.csv"

_PASS_RATE_COLUMNS: tuple[str, ...] = ("encoding", *EXACT_COLUMNS.values(), *WITHIN_TOL_COLUMNS.values(), "selected")
_FAILING_COLUMNS: tuple[str, ...] = ("match_id", "check")
_RESULT_COLUMNS: tuple[str, ...] = SCORE_CHECK_COLUMNS[1:]  # every score_checks column but the optional encoding
_RUNS_CHECK_CODES = np.array([SCORE_CHECKS.index(check) for check in RUNS_CHECKS], dtype=np.int64)
_TIE_RESULTS = (TIE, SUPER_OVER_WIN)  # the result types that trigger the tie check (criterion 19)
_LEGAL_BALLS_PER_OVER = 6  # Laws of Cricket
_WICKETS_PER_INNINGS = 10  # Laws of Cricket: a chase won by w wickets ends with 10 - w wickets fallen
_MISS_MAX = int(np.iinfo(np.int32).max)  # the largest Runs_Check_Miss the Int32 runs_check_miss column holds


def _take(values: np.ndarray, position: np.ndarray, fill: Any) -> np.ndarray:
    """``values[position]``, with ``fill`` where ``position`` is -1 (no such row)."""
    taken = np.full(len(position), fill, dtype=values.dtype)
    hit = position >= 0
    taken[hit] = values[position[hit]]
    return taken


def _shown_ids(ids: np.ndarray) -> str:
    """Up to 5 match ids, for an error message."""
    return ", ".join(map(str, ids[:5].tolist()))


def over_check(balls: pd.DataFrame, cls: pd.Series) -> pd.Series:
    """Over check (criterion 20) of every match with at least one Delivery in innings 1 or 2.

    ``balls`` needs ``match_id``, ``innings`` and ``overid``, in any row order; ``cls`` holds the
    DeliveryClass codes of its rows (classify) on the index of ``balls``. Only Deliveries of
    innings 1 and 2 count. An innings passes when every ``overid`` in it except the highest holds
    exactly 6 Legal_Balls and the highest holds at most 6; Wides and No_Balls are not Legal_Balls.
    A match passes when each of its innings 1 and 2 that holds a Delivery passes, so a match whose
    second innings never started is judged on its first. The check does not depend on the
    Extras_Encoding.

    Returns a bool Series named ``over_ok``, indexed by ``match_id`` (int32, sorted), with one
    entry per match that has a Delivery in innings 1 or 2: the ``over_ok`` argument of run_checks.

    Raises DataError (exit code 3) when an ``innings`` is missing or not a whole number, or when the
    ``match_id`` or ``overid`` of a Delivery in innings 1 or 2 is.
    """
    context = "The over check cannot be run"
    innings = _whole_numbers(balls, ["innings"], context)["innings"].values
    codes = _class_codes(cls, balls.index)
    keep = np.flatnonzero((innings == 1) | (innings == 2))
    keys = _whole_numbers(balls[["match_id", "overid"]].iloc[keep], ["match_id", "overid"], context)
    match_id, innings, over = keys["match_id"].values, innings[keep], keys["overid"].values
    order = np.lexsort((over, innings, match_id))
    match_id, innings, over = match_id[order], innings[order], over[order]
    legal = (codes[keep] == DeliveryClass.LEGAL)[order]
    first_ball = _first_of_innings(match_id, innings)
    first_ball[1:] |= over[1:] != over[:-1]
    starts = np.flatnonzero(first_ball)  # the first Delivery of each Over, in (match, innings, overid) order
    if len(starts) == 0:
        empty = pd.Index(np.zeros(0, dtype=np.int32), name="match_id")
        return pd.Series(np.zeros(0, dtype=bool), index=empty, name="over_ok")
    legal_balls = np.add.reduceat(legal.astype(np.int64), starts)
    over_match, over_innings = match_id[starts], innings[starts]
    highest = np.ones(len(starts), dtype=bool)  # the last Over of each innings holds its highest overid
    highest[:-1] = (over_match[1:] != over_match[:-1]) | (over_innings[1:] != over_innings[:-1])
    over_passes = np.where(highest, legal_balls <= _LEGAL_BALLS_PER_OVER, legal_balls == _LEGAL_BALLS_PER_OVER)
    first_over = np.ones(len(starts), dtype=bool)
    first_over[1:] = over_match[1:] != over_match[:-1]
    match_starts = np.flatnonzero(first_over)
    return pd.Series(
        np.logical_and.reduceat(over_passes, match_starts),
        index=pd.Index(over_match[match_starts].astype(np.int32), name="match_id"),
        name="over_ok",
    )


def _innings_position(agg_match: np.ndarray, agg_innings: np.ndarray, number: int, ids: np.ndarray) -> np.ndarray:
    """Row of the innings_aggregates table holding innings ``number`` of each match in ``ids``; -1 when none."""
    rows = np.flatnonzero(agg_innings == number)
    index = pd.Index(agg_match[rows])
    if not index.is_unique:
        raise ValueError("agg must hold one row per (match_id, innings), as innings_aggregates returns")
    found = index.get_indexer(ids)
    position = np.full(len(ids), -1, dtype=np.int64)
    position[found >= 0] = rows[found[found >= 0]]
    return position


def _over_flags(over_ok: pd.Series, ids: np.ndarray, has_balls: np.ndarray) -> np.ndarray:
    """The over_check value of each match in ``ids`` that has a Delivery in innings 1 or 2; False elsewhere."""
    if not isinstance(over_ok, pd.Series) or not pd.api.types.is_bool_dtype(over_ok.dtype):
        raise TypeError("over_ok must be the boolean Series that over_check returns, indexed by match_id")
    if over_ok.isna().any():
        raise ValueError("over_ok must hold no missing value, as over_check returns")
    index = pd.Index(np.asarray(over_ok.index, dtype=np.int64))
    if not index.is_unique:
        raise ValueError("over_ok must hold one entry per match, as over_check returns")
    position = np.where(has_balls, index.get_indexer(ids), -1)
    lacking = has_balls & (position < 0)
    if lacking.any():
        raise ValueError(
            "over_ok must hold the over check of every match with a Delivery in innings 1 or 2, as "
            f"over_check returns; it has no entry for match {_shown_ids(ids[lacking])}"
        )
    return _take(over_ok.to_numpy(dtype=bool), position, False)


def run_checks(
    matches: pd.DataFrame,
    agg: pd.DataFrame,
    parsed: pd.DataFrame,
    over_ok: pd.Series,
    *,
    encoding: str,
) -> pd.DataFrame:
    """The Score_Checks (criteria 16-20) of every match under one Extras_Encoding: its ``score_checks`` rows.

    The arguments describe the same matches:

    * ``matches``: ``match_id``, ``revised_target`` and ``win_team_id`` of every match;
    * ``agg``: the innings_aggregates table of the encoding, from the Deliveries of these matches;
    * ``parsed``: the parse_results table of ``matches``;
    * ``over_ok``: the over_check Series of the same Deliveries (the same under every encoding);
    * ``encoding``: the name of the encoding, copied to the ``encoding`` column.

    With T the Target (targets of ``matches`` and ``agg``), S2 the second-innings total, f the
    team runs of the final innings-2 Delivery and W2 the second-innings Counted_Wickets (S2, f and
    W2 are 0 when innings 2 has no Delivery), and the margin from ``parsed``:

    * ``runs_margin`` (criterion 16) applies to a win by runs whose ``win_team_id`` equals the
      innings-1 ``batting_team_id``. Runs_Check_Miss = |((T - 1) - S2) - margin|.
    * ``chase`` (criterion 17) applies to a win by wickets. Runs_Check_Miss = the larger of the
      shortfall T - S2 and the excess (S2 - f) - (T - 1), each floored at 0, which is 0 exactly
      when S2 - f < T <= S2.
    * ``wickets_margin`` (criterion 18) applies to a win by wickets and passes when the margin
      equals 10 - W2.
    * ``tie`` (criterion 19) applies to a tie or a super-over win. Runs_Check_Miss = |S2 - (T - 1)|.
    * ``over`` (criterion 20) applies to every match and passes when ``over_ok`` is true.

    A match with no Delivery in innings 1 or 2 has nothing to reconstruct, so none of its five
    checks applies, whatever its result. A runs-margin, chase or tie check passes exactly when its
    Runs_Check_Miss is 0. Every term is an integer, so all comparisons are exact.

    Returns five rows per match of ``matches``, whether or not each check applies, sorted by
    ``match_id`` and then check in SCORE_CHECKS order, with the SCORE_CHECK_COLUMNS on a
    RangeIndex: ``encoding`` and ``check`` (str), ``match_id`` (int32), ``applies`` (bool),
    ``passed`` (nullable boolean, missing where the check does not apply) and ``runs_check_miss``
    (nullable Int32, filled only on the applicable runs-margin, chase and tie rows).

    Raises DataError (exit code 3) when a ``match_id`` of ``matches`` is missing, non-whole or
    repeated, a ``revised_target`` is present but not a positive whole number, a ``win_team_id``
    is present but not a whole number, ``agg`` holds innings 1 or 2 of a match that is not in
    ``matches``, a chase or tie check applies to a match with no Target (no ``revised_target`` and
    no innings-1 Delivery), or a Runs_Check_Miss is above 2,147,483,647. Raises TypeError or
    ValueError when ``encoding`` is not a non-empty string, or when ``agg``, ``parsed`` or
    ``over_ok`` is not the table described above for these matches.
    """
    if not isinstance(encoding, str):
        raise TypeError(f"encoding must be the name of the Extras_Encoding of agg; got {type(encoding).__name__}")
    if not encoding:
        raise ValueError("encoding must be the name of the Extras_Encoding of agg; got an empty string")
    context = "The Score_Checks cannot be run"
    target = targets(matches, agg)  # one entry per match, sorted by match_id
    ids = target.index.to_numpy(dtype=np.int64)
    n = len(ids)
    t = target.to_numpy(dtype=np.int64, na_value=0)
    no_target = target.isna().to_numpy(dtype=bool)
    win = _whole_numbers(matches, ["win_team_id"], context, may_be_missing=["win_team_id"])["win_team_id"]
    row_of = pd.Index(_match_ids(matches, context)).get_indexer(ids)
    win_team, no_winner = win.values[row_of], win.missing[row_of]

    # Innings 1 and 2 of each match, from the aggregates of the encoding.
    agg_context = "The innings aggregates cannot be used"
    found = _whole_numbers(agg, ["match_id", "innings", "total", "wickets", "final_ball_runs"], agg_context)
    batting = _whole_numbers(agg, ["batting_team_id"], agg_context, may_be_missing=["batting_team_id"])
    agg_match, agg_innings = found["match_id"].values, found["innings"].values
    unknown = np.unique(agg_match[((agg_innings == 1) | (agg_innings == 2)) & ~np.isin(agg_match, ids)])
    if len(unknown):
        raise DataError(
            f"{context}:\n  - innings 1 or 2 of {len(unknown):,} match(es) that are not in matches hold "
            f"Deliveries (for example match {_shown_ids(unknown)})"
        )
    first = _innings_position(agg_match, agg_innings, 1, ids)
    second = _innings_position(agg_match, agg_innings, 2, ids)
    has_balls = (first >= 0) | (second >= 0)
    batting_first = _take(batting["batting_team_id"].values, first, 0)
    batting_known = _take(~batting["batting_team_id"].missing, first, False)
    s2 = _take(found["total"].values, second, 0)
    f = _take(found["final_ball_runs"].values, second, 0)
    w2 = _take(found["wickets"].values, second, 0)

    # Result type and margin of each match.
    parsed_context = "The parsed results cannot be used"
    parsed_index = pd.Index(_whole_numbers(parsed, ["match_id"], parsed_context)["match_id"].values)
    if not parsed_index.is_unique:
        raise ValueError("parsed must hold one row per match, as parse_results returns")
    p = parsed_index.get_indexer(ids)
    if (p < 0).any():
        raise ValueError(
            f"parsed must be the parse_results table of matches; it has no row for match {_shown_ids(ids[p < 0])}"
        )
    result_type = parsed["result_type"].iloc[p]
    by_runs = result_type.isin([WIN_BY_RUNS]).to_numpy(dtype=bool)
    by_wickets = result_type.isin([WIN_BY_WICKETS]).to_numpy(dtype=bool)
    tied = result_type.isin(_TIE_RESULTS).to_numpy(dtype=bool)
    margins = _whole_numbers(parsed, ["margin"], parsed_context, may_be_missing=["margin"])["margin"]
    margin, no_margin = margins.values[p], margins.missing[p]
    over_pass = _over_flags(over_ok, ids, has_balls)

    applies_runs = has_balls & by_runs & ~no_winner & batting_known & (win_team == batting_first)
    applies_chase = has_balls & by_wickets
    applies_tie = has_balls & tied
    lacking = (applies_runs | applies_chase | applies_tie) & no_target
    if lacking.any():
        raise DataError(
            f"{context}:\n  - the Target is undefined (no revised_target and no innings-1 Delivery) in "
            f"{int(lacking.sum()):,} match(es) whose result needs it (for example match {_shown_ids(ids[lacking])})"
        )
    unparsed = (applies_runs | applies_chase) & no_margin
    if unparsed.any():
        raise ValueError(
            "parsed must be the parse_results table of matches; a win by runs or by wickets has no margin "
            f"in match {_shown_ids(ids[unparsed])}"
        )

    tie_score = t - 1  # T - 1: the second-innings total that ties the match
    runs_miss = np.abs((tie_score - s2) - margin)
    chase_miss = np.maximum(np.maximum(t - s2, (s2 - f) - tie_score), 0)
    tie_miss = np.abs(s2 - tie_score)
    wickets_pass = margin == _WICKETS_PER_INNINGS - w2
    too_large = (
        (applies_runs & (runs_miss > _MISS_MAX))
        | (applies_chase & (chase_miss > _MISS_MAX))
        | (applies_tie & (tie_miss > _MISS_MAX))
    )
    if too_large.any():
        raise DataError(
            f"{context}:\n  - a Runs_Check_Miss is above {_MISS_MAX:,}, the largest the runs_check_miss column "
            f"holds, in {int(too_large.sum()):,} match(es) (for example match {_shown_ids(ids[too_large])})"
        )

    # Five rows per match, in SCORE_CHECKS order.
    k = len(SCORE_CHECKS)
    nothing, zero = np.zeros(n, dtype=bool), np.zeros(n, dtype=np.int64)
    applies = np.column_stack((applies_runs, applies_chase, applies_chase, applies_tie, has_balls)).reshape(-1)
    passed = np.column_stack((runs_miss == 0, chase_miss == 0, wickets_pass, tie_miss == 0, over_pass)).reshape(-1)
    scored = np.column_stack((applies_runs, applies_chase, nothing, applies_tie, nothing)).reshape(-1)
    miss = np.column_stack((runs_miss, chase_miss, zero, tie_miss, zero)).reshape(-1)
    return pd.DataFrame(
        {
            "encoding": pd.array(np.full(n * k, encoding, dtype=object), dtype="str"),
            "match_id": np.repeat(ids, k).astype(np.int32),
            "check": pd.array(np.tile(np.array(SCORE_CHECKS, dtype=object), n), dtype="str"),
            "applies": applies,
            "passed": pd.arrays.BooleanArray(passed & applies, ~applies),
            "runs_check_miss": pd.arrays.IntegerArray(np.where(scored, miss, 0).astype(np.int32), ~scored),
        }
    )


class _CheckRows(NamedTuple):
    """A ``score_checks`` table read by _read_checks, as numpy arrays in its row order."""

    match_id: np.ndarray  # int64
    check: np.ndarray  # int64: position of the check in SCORE_CHECKS
    applies: np.ndarray  # bool
    passed: np.ndarray  # bool: passed exactly; False where the check does not apply
    miss: np.ndarray  # int64 Runs_Check_Miss; 0 where it is missing
    has_miss: np.ndarray  # bool: the Runs_Check_Miss is present
    encodings: list[str]  # distinct values of the optional encoding column, in order of appearance


def _nullable_values(series: pd.Series, dtype: str, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Values (0 or False where missing) and missing flags of a column read as ``Int64`` or ``boolean``."""
    try:
        values = pd.array(series, dtype=dtype)
    except (TypeError, ValueError) as exc:
        kind = "whole numbers" if dtype == "Int64" else "booleans"
        raise ValueError(f"{name} must hold {kind}; got dtype {series.dtype}") from exc
    if dtype == "Int64":
        return values.to_numpy(dtype=np.int64, na_value=0), np.asarray(values.isna(), dtype=bool)
    return values.to_numpy(dtype=bool, na_value=False), np.asarray(values.isna(), dtype=bool)


def _read_checks(checks: pd.DataFrame, name: str) -> _CheckRows:
    """Read a ``score_checks`` table (run_checks rows), with or without its non-applicable rows.

    Raises TypeError or ValueError, naming the table as ``name``, unless it has the
    SCORE_CHECK_COLUMNS (``encoding`` is optional), only Score_Check names, at most one row per
    encoding, match and check, an ``applies`` flag on every row, a ``passed`` flag on every
    applicable row and no negative Runs_Check_Miss.
    """
    if not isinstance(checks, pd.DataFrame):
        raise TypeError(f"{name} must be a score_checks DataFrame, as run_checks returns; got {type(checks).__name__}")
    absent = [column for column in _RESULT_COLUMNS if column not in checks.columns]
    if absent:
        raise ValueError(f"{name} must be a score_checks table, as run_checks returns; it lacks {', '.join(absent)}")
    code = pd.Index(SCORE_CHECKS).get_indexer(checks["check"])
    if (code < 0).any():
        unknown = ", ".join(sorted({repr(value) for value in checks["check"][code < 0].tolist()}))
        raise ValueError(f"{name} holds check names that are not Score_Checks ({', '.join(SCORE_CHECKS)}): {unknown}")
    match_id, no_id = _nullable_values(checks["match_id"], "Int64", f"{name} match_id")
    applies, no_applies = _nullable_values(checks["applies"], "boolean", f"{name} applies")
    passed, no_passed = _nullable_values(checks["passed"], "boolean", f"{name} passed")
    miss, no_miss = _nullable_values(checks["runs_check_miss"], "Int64", f"{name} runs_check_miss")
    has_encoding = "encoding" in checks.columns
    keys = [match_id, code]
    if has_encoding:
        keys.insert(0, checks["encoding"].to_numpy(dtype=object))
    problems = []
    if no_id.any():
        problems.append("a match_id is missing")
    if no_applies.any():
        problems.append("an applies flag is missing")
    if (applies & no_passed).any():
        problems.append("a passed flag is missing on an applicable row")
    if (~no_miss & (miss < 0)).any():
        problems.append("a runs_check_miss is negative")
    if pd.MultiIndex.from_arrays(keys).has_duplicates:
        problems.append("a match has more than one row for a check under one encoding")
    if problems:
        raise ValueError(f"{name} is not a score_checks table as run_checks returns: {'; '.join(problems)}")
    encodings = [str(value) for value in checks["encoding"].drop_duplicates().tolist()] if has_encoding else []
    return _CheckRows(match_id, code.astype(np.int64), applies, passed & applies, miss, ~no_miss, encodings)


def _one_encoding(rows: _CheckRows, name: str) -> None:
    """ValueError unless the table holds the rows of at most one Extras_Encoding."""
    if len(rows.encodings) > 1:
        raise ValueError(
            f"{name} must hold the score_checks rows of one Extras_Encoding (the selected one); "
            f"it holds {len(rows.encodings)}: {', '.join(rows.encodings)}"
        )


def _per_check(code: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Number of rows of each Score_Check, in SCORE_CHECKS order, among the rows where ``mask`` is true."""
    return np.bincount(code[mask], minlength=len(SCORE_CHECKS)).astype(np.int64)


def _rates(passes: np.ndarray, applicable: np.ndarray) -> np.ndarray:
    """100 x passes / applicable as float64; NaN where no match is applicable (the rate is undefined)."""
    rates = np.full(len(applicable), np.nan)
    some = applicable > 0
    rates[some] = 100.0 * passes[some] / applicable[some]
    return rates


def _score_tolerance(tolerance: Any) -> int:
    """The Score_Tolerance as an int; TypeError or ValueError unless it is a whole number of runs, at least 0."""
    if isinstance(tolerance, (bool, np.bool_)) or not isinstance(tolerance, (int, np.integer)):
        raise TypeError(
            "tolerance must be the Score_Tolerance in runs (design_choices.D20_score_tolerance), an int; "
            f"got {type(tolerance).__name__} {tolerance!r}"
        )
    if tolerance < 0:
        raise ValueError(f"tolerance must be at least 0 runs; got {tolerance}")
    return int(tolerance)


def _rate_columns(rates: pd.DataFrame, name: str) -> list[str]:
    """The candidate names of an exact_pass_rates table; ValueError when one of its columns is missing."""
    if not isinstance(rates, pd.DataFrame):
        raise TypeError(f"{name} must be the DataFrame that exact_pass_rates returns; got {type(rates).__name__}")
    absent = [column for column in ("encoding", *EXACT_COLUMNS.values()) if column not in rates.columns]
    if absent:
        raise ValueError(f"{name} must be an exact_pass_rates table; it lacks {', '.join(absent)}")
    return [str(value) for value in rates["encoding"].tolist()]


def exact_pass_rates(checks_by_encoding: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Exact_Pass_Rate of each Score_Check under each candidate Extras_Encoding (criteria 7 and 21).

    ``checks_by_encoding`` maps each candidate name, in Config order (``data_rules.candidate_encodings``,
    which is the order of the pass-rate table), to its run_checks table; a table may leave out its
    non-applicable rows. The Exact_Pass_Rate of a check is 100 x (matches that pass it) /
    (matches it applies to), and is missing (NaN) when it applies to no match.

    Returns one row per candidate, in the order of the mapping, on a RangeIndex, with columns
    ``encoding`` (str) and the five EXACT_COLUMNS (float64): the ``rates`` argument of
    select_encoding and pass_rate_table.

    Raises TypeError or ValueError when a name is not a non-empty string, a table is not a
    score_checks table, or a table's ``encoding`` column names another candidate.
    """
    if not isinstance(checks_by_encoding, Mapping):
        raise TypeError(
            "checks_by_encoding must map each candidate name to its run_checks table; "
            f"got {type(checks_by_encoding).__name__}"
        )
    names: list[str] = []
    rows: list[np.ndarray] = []
    for name, table in checks_by_encoding.items():
        if not isinstance(name, str) or not name:
            raise TypeError(f"checks_by_encoding keys must be candidate names (non-empty strings); got {name!r}")
        read = _read_checks(table, f"checks_by_encoding[{name!r}]")
        others = [value for value in read.encodings if value != name]
        if others:
            raise ValueError(f"checks_by_encoding[{name!r}] holds rows of another encoding: {', '.join(others)}")
        rows.append(_rates(_per_check(read.check, read.passed), _per_check(read.check, read.applies)))
        names.append(name)
    rates = np.vstack(rows) if rows else np.zeros((0, len(SCORE_CHECKS)))
    columns: dict[str, Any] = {"encoding": pd.array(names, dtype="str")}
    columns.update({EXACT_COLUMNS[check]: rates[:, j] for j, check in enumerate(SCORE_CHECKS)})
    return pd.DataFrame(columns)


def _mean_rate(rates: np.ndarray) -> float:
    """Mean of the defined (non-NaN) rates, from math.fsum so that it does not depend on their order; NaN if none."""
    defined = rates[~np.isnan(rates)]
    return math.fsum(defined.tolist()) / len(defined) if len(defined) else math.nan


def select_encoding(rates: pd.DataFrame) -> str:
    """The Extras_Encoding selected under criterion 7, from the exact_pass_rates table ``rates``.

    The selected candidate has the highest mean of its five Exact_Pass_Rates. When the
    F30_Encoding ties for the highest mean, it is selected; otherwise the tied candidate listed
    first in ``rates`` is. A mean is the math.fsum of the rates divided by their number, so it does
    not depend on the order of the rates. Means that are equal as fractions can still differ in
    the last digit as floats: 275/6 comes out as 45.83333333333333 from the rates 0, 0, 100 and
    500/6, and as 45.833333333333336 from 100/3, 0, 200/3 and 500/6. ``rates`` holds only the
    float rates, not the match counts, so a candidate ties for the highest mean ``best`` when
    ``math.isclose(mean, best, rel_tol=1e-9, abs_tol=1e-9)``. Rates lie in [0, 100], so this
    tolerance is at most 1e-7. That is far above the float rounding error of a mean (below 1e-12)
    and, in practice, far below the gap between two distinct means of rates over a few thousand
    matches. A rate is missing when its check applies to no match, and a mean then covers the
    defined rates; whether a check applies does not depend on the encoding, so a table built from
    run_checks misses the same rates under every candidate. A candidate with no defined rate is
    not selected while another candidate has one.

    Raises StageError (exit code 5) when ``rates`` lists fewer than 2 candidates, does not list
    the F30_Encoding or lists a candidate twice (criterion 7 compares at least 2 candidates,
    including the F30_Encoding), or holds no defined rate at all. Raises ValueError when a column
    of exact_pass_rates is missing.
    """
    names = _rate_columns(rates, "rates")
    repeated = sorted({name for name in names if names.count(name) > 1})
    if len(names) < 2 or F30_ENCODING not in names or repeated:
        found = f"; listed more than once: {', '.join(repeated)}" if repeated else ""
        raise StageError(
            "Criterion 7 selects among at least 2 distinct candidate Extras_Encodings, including "
            f"{F30_ENCODING}; the pass-rate table lists {', '.join(names) or 'none'}{found}"
        )
    values = rates[list(EXACT_COLUMNS.values())].to_numpy(dtype=np.float64, na_value=np.nan)
    means = [_mean_rate(row) for row in values]
    defined = [mean for mean in means if not math.isnan(mean)]
    if not defined:
        raise StageError(
            "No Exact_Pass_Rate is defined, because no Score_Check applies to any match, so no "
            "Extras_Encoding can be selected (criterion 7)"
        )
    best = max(defined)
    tied = [name for name, mean in zip(names, means) if math.isclose(mean, best, rel_tol=1e-9, abs_tol=1e-9)]
    return F30_ENCODING if F30_ENCODING in tied else tied[0]


def within_tolerance_pass_rates(checks: pd.DataFrame, tolerance: int) -> pd.Series:
    """Within_Tolerance_Pass_Rate of each Score_Check under the selected Extras_Encoding (criterion 21).

    ``checks`` holds the run_checks rows of the selected encoding only (with or without its
    non-applicable rows). ``tolerance`` is the Score_Tolerance in runs, the Config value
    ``design_choices.D20_score_tolerance``, a whole number of at least 0. An applicable
    runs-margin, chase or tie check passes within tolerance when its Runs_Check_Miss is at most
    ``tolerance``; an applicable wickets-margin or over check passes within tolerance only when it
    passes exactly. Each rate is 100 x (matches that pass within tolerance) / (matches the check
    applies to), the same denominator as the Exact_Pass_Rate, and is missing (NaN) when the check
    applies to no match. A miss of 0 is within any tolerance, so no rate is below the
    Exact_Pass_Rate of its check.

    Returns a float64 Series named ``within_tol_pass_rate``, indexed by the SCORE_CHECKS in table
    order (index name ``check``): the ``within`` argument of pass_rate_table.

    Raises TypeError or ValueError when ``tolerance`` is not a whole number of at least 0, when
    ``checks`` is not a score_checks table or holds the rows of more than one encoding, or when an
    applicable runs-margin, chase or tie row has no Runs_Check_Miss.
    """
    allowed = _score_tolerance(tolerance)
    read = _read_checks(checks, "checks")
    _one_encoding(read, "checks")
    runs_rows = np.isin(read.check, _RUNS_CHECK_CODES)
    unscored = read.applies & runs_rows & ~read.has_miss
    if unscored.any():
        raise ValueError(
            "checks must give the Runs_Check_Miss of every applicable runs-margin, chase and tie check; "
            f"it lacks one on {_rows(int(unscored.sum()))}"
        )
    within = read.applies & np.where(runs_rows, read.has_miss & (read.miss <= allowed), read.passed)
    return pd.Series(
        _rates(_per_check(read.check, within), _per_check(read.check, read.applies)),
        index=pd.Index(SCORE_CHECKS, name="check"),
        name="within_tol_pass_rate",
    )


def pass_rate_table(rates: pd.DataFrame, selected: str, within: pd.Series | Mapping[str, float]) -> pd.DataFrame:
    """The pass-rate table of criterion 21, which write_pass_rates writes as ``pass_rates.csv``.

    ``rates`` is the exact_pass_rates table, ``selected`` the select_encoding result and ``within``
    the within_tolerance_pass_rates of the selected encoding, keyed by check name. Returns one
    row per candidate, in the order of ``rates``, on a RangeIndex, with columns ``encoding``
    (str), the five EXACT_COLUMNS (float64, from ``rates``), the five WITHIN_TOL_COLUMNS
    (float64: the ``within`` rates on the row of ``selected``, missing on every other row) and
    ``selected`` (bool, true on the row of ``selected`` only).

    Raises ValueError when ``selected`` is not exactly one of the candidates of ``rates``, when a
    column of exact_pass_rates is missing, or when ``within`` lacks a Score_Check or lists one
    twice.
    """
    names = _rate_columns(rates, "rates")
    if names.count(selected) != 1:
        raise ValueError(
            f"selected must name one candidate of rates exactly once; {selected!r} appears {names.count(selected)} "
            f"time(s) among {', '.join(names) or 'no candidates'}"
        )
    within = within if isinstance(within, pd.Series) else pd.Series(dict(within), dtype=np.float64)
    lacking = [check for check in SCORE_CHECKS if check not in within.index]
    if lacking or not within.index.is_unique:
        raise ValueError(
            "within must give one Within_Tolerance_Pass_Rate per Score_Check, as within_tolerance_pass_rates "
            f"returns; {'it lacks ' + ', '.join(lacking) if lacking else 'a check is listed twice'}"
        )
    is_selected = np.array([name == selected for name in names], dtype=bool)
    columns: dict[str, Any] = {"encoding": pd.array(names, dtype="str")}
    for check in SCORE_CHECKS:
        columns[EXACT_COLUMNS[check]] = rates[EXACT_COLUMNS[check]].to_numpy(dtype=np.float64, na_value=np.nan)
    for check in SCORE_CHECKS:
        value = within[check]
        rate = np.nan if pd.isna(value) else float(value)
        columns[WITHIN_TOL_COLUMNS[check]] = np.where(is_selected, rate, np.nan)
    columns["selected"] = is_selected
    return pd.DataFrame(columns)


def failing_matches(checks: pd.DataFrame) -> pd.DataFrame:
    """Every Score_Check that a match fails exactly under the selected Extras_Encoding (criterion 23).

    ``checks`` holds the run_checks rows of the selected encoding only. Returns one row per
    applicable check that did not pass exactly, so a match appears once per failed check, sorted by
    ``match_id`` and then check in SCORE_CHECKS order, on a RangeIndex, with columns ``match_id``
    (int32) and ``check`` (str). A runs-margin, chase or tie check fails exactly when its
    Runs_Check_Miss is above 0, so the table also lists runs checks whose miss is within the
    Score_Tolerance. write_failing_matches writes it as ``failing_matches.csv``.

    Raises TypeError or ValueError when ``checks`` is not a score_checks table or holds the rows of
    more than one encoding.
    """
    read = _read_checks(checks, "checks")
    _one_encoding(read, "checks")
    failed = np.flatnonzero(read.applies & ~read.passed)
    failed = failed[np.lexsort((read.check[failed], read.match_id[failed]))]
    return pd.DataFrame(
        {
            "match_id": read.match_id[failed].astype(np.int32),
            "check": pd.array(np.array(SCORE_CHECKS, dtype=object)[read.check[failed]], dtype="str"),
        }
    )


def _write_audit_csv(outdir: str | Path, file_name: str, table: pd.DataFrame, columns: Sequence[str]) -> Path:
    """Write ``columns`` of ``table`` to ``file_name`` in ``outdir`` as UTF-8, via a temporary file renamed in place."""
    if not isinstance(table, pd.DataFrame):
        raise TypeError(f"the {file_name} table must be a DataFrame; got {type(table).__name__}")
    absent = [column for column in columns if column not in table.columns]
    if absent:
        raise ValueError(f"the {file_name} table lacks the column(s) {', '.join(absent)}")
    folder = Path(outdir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / file_name
    temporary = path.with_name(path.name + ".tmp")
    try:
        table.loc[:, list(columns)].to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def write_pass_rates(outdir: str | Path, table: pd.DataFrame) -> Path:
    """Write the pass_rate_table ``table`` to ``pass_rates.csv`` in the folder ``outdir`` (criterion 21).

    ``outdir`` is the audit folder (``outputs/audit/`` in a Run) and is created when absent. The
    file has the columns ``encoding``, the five EXACT_COLUMNS, the five WITHIN_TOL_COLUMNS and
    ``selected``, in that order, and one row per candidate. Rates are written at full float
    precision, a missing rate as an empty field and ``selected`` as ``True`` or ``False``. The
    file is written as UTF-8 through a temporary file renamed in place, so it replaces any earlier
    copy whole.

    Returns the path of the file. Raises ValueError when ``table`` lacks one of the columns.
    """
    return _write_audit_csv(outdir, PASS_RATES_FILE, table, _PASS_RATE_COLUMNS)


def write_failing_matches(outdir: str | Path, failing: pd.DataFrame) -> Path:
    """Write the failing_matches table ``failing`` to ``failing_matches.csv`` in ``outdir`` (criterion 23).

    ``outdir`` is the audit folder (``outputs/audit/`` in a Run) and is created when absent. The
    file has the header ``match_id,check`` and one row per failed check of the selected
    Extras_Encoding, in the order of ``failing``; with no failure it holds the header only. It is
    written as UTF-8 through a temporary file renamed in place, so it replaces any earlier copy
    whole.

    Returns the path of the file. Raises ValueError when ``failing`` lacks one of the columns.
    """
    return _write_audit_csv(outdir, FAILING_MATCHES_FILE, failing, _FAILING_COLUMNS)


# ------------------------------------------------------------ Match exclusion (criteria 24 and 27)

#: File names, in the audit folder, of the exclusion conditions met by each match and of the
#: excluded-match count at each grid tolerance (criterion 27). Both are written only where the
#: Config option ``data_rules.exclude_failing_matches`` is on.
EXCLUDED_MATCHES_FILE = "excluded_matches.csv"
EXCLUSION_COUNTS_FILE = "exclusion_counts.csv"

#: Columns of the exclusion_conditions table (``excluded_matches.csv``) and of the
#: exclusion_counts table (``exclusion_counts.csv``), in file order (design, Data Models).
EXCLUSION_COLUMNS: tuple[str, ...] = ("match_id", "check", "runs_check_miss")
EXCLUSION_COUNT_COLUMNS: tuple[str, ...] = ("score_tolerance", "n_excluded_matches", "is_config_value")


def _selected_rows(checks: pd.DataFrame, name: str) -> _CheckRows:
    """Read the ``score_checks`` rows of one Extras_Encoding, with the miss of every applicable runs check.

    Raises TypeError or ValueError, naming the table as ``name``, unless it is a score_checks
    table of at most one encoding whose applicable runs-margin, chase and tie rows all give a
    Runs_Check_Miss.
    """
    rows = _read_checks(checks, name)
    _one_encoding(rows, name)
    unscored = rows.applies & np.isin(rows.check, _RUNS_CHECK_CODES) & ~rows.has_miss
    if unscored.any():
        raise ValueError(
            f"{name} must give the Runs_Check_Miss of every applicable runs-margin, chase and tie check; "
            f"it lacks one on {_rows(int(unscored.sum()))}"
        )
    return rows


def _exclusion_mask(rows: _CheckRows, allowed: int) -> np.ndarray:
    """True on each row that meets a condition of criterion 24 at the Score_Tolerance ``allowed``."""
    runs_rows = np.isin(rows.check, _RUNS_CHECK_CODES)
    failed = rows.applies & ~runs_rows & ~rows.passed  # a failed wickets-margin or over check
    missed = rows.applies & runs_rows & (rows.miss > allowed)  # a runs check that misses by more than allowed
    return failed | missed


def _tolerance_grid(grid: Any) -> list[int]:
    """The Score_Tolerance reporting grid as ints; TypeError or ValueError unless it lists distinct whole numbers >= 0."""
    if not isinstance(grid, (list, tuple, np.ndarray)):
        raise TypeError(
            "grid must be the Score_Tolerance reporting grid (data_rules.score_tolerance_grid), a list of whole "
            f"numbers of runs; got {type(grid).__name__} {grid!r}"
        )
    values: list[int] = []
    for value in list(grid):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"grid must list whole numbers of runs (ints); got {type(value).__name__} {value!r}")
        if value < 0:
            raise ValueError(f"grid must list Score_Tolerances of at least 0 runs; got {value}")
        values.append(int(value))
    repeated = sorted({value for value in values if values.count(value) > 1})
    if repeated:
        raise ValueError(f"grid must list each Score_Tolerance once; it repeats {', '.join(map(str, repeated))}")
    return values


def exclusion_conditions(checks: pd.DataFrame, tolerance: int) -> pd.DataFrame:
    """Each condition of criterion 24 that a match meets under the selected Extras_Encoding.

    ``checks`` holds the run_checks rows of the selected encoding only (with or without its
    non-applicable rows). ``tolerance`` is the Score_Tolerance in runs, the Config value
    ``design_choices.D20_score_tolerance``, a whole number of at least 0. Each applicable check
    meets a condition when:

    * it is the over check (criterion 20) or the wickets-margin check (criterion 18) and fails;
    * it is the runs-margin (criterion 16), chase (criterion 17) or tie (criterion 19) check and
      its Runs_Check_Miss is above ``tolerance``.

    Returns one row per condition met, so a match can have several rows, sorted by ``match_id``
    and then check in SCORE_CHECKS order, on a RangeIndex, with the EXCLUSION_COLUMNS:
    ``match_id`` (int32), ``check`` (str) and ``runs_check_miss`` (nullable Int32: the miss of a
    runs check, blank for the over and wickets-margin checks). The matches that criterion 24
    excludes are its distinct ``match_id`` values. A runs check that misses by 1 to ``tolerance``
    runs fails exactly, so failing_matches lists it, but it meets no condition; at a tolerance of
    0 the excluded matches are therefore exactly the matches in failing_matches.
    write_reconstruction_audit writes the table as ``excluded_matches.csv``.

    Raises TypeError or ValueError when ``tolerance`` is not a whole number of at least 0, when
    ``checks`` is not a score_checks table or holds the rows of more than one encoding, or when
    an applicable runs-margin, chase or tie row has no Runs_Check_Miss.
    """
    allowed = _score_tolerance(tolerance)
    rows = _selected_rows(checks, "checks")
    met = np.flatnonzero(_exclusion_mask(rows, allowed))
    met = met[np.lexsort((rows.check[met], rows.match_id[met]))]
    is_runs = np.isin(rows.check[met], _RUNS_CHECK_CODES)
    return pd.DataFrame(
        {
            "match_id": rows.match_id[met].astype(np.int32),
            "check": pd.array(np.array(SCORE_CHECKS, dtype=object)[rows.check[met]], dtype="str"),
            "runs_check_miss": pd.arrays.IntegerArray(
                np.where(is_runs, rows.miss[met], 0).astype(np.int32), ~is_runs
            ),
        }
    )


def exclusion_counts(checks: pd.DataFrame, grid: Sequence[int], *, tolerance: int) -> pd.DataFrame:
    """Distinct matches that criterion 24 excludes at each Score_Tolerance of the reporting grid (criterion 27).

    ``checks`` holds the run_checks rows of the selected Extras_Encoding only, as for
    exclusion_conditions. ``grid`` is the Config reporting grid
    ``data_rules.score_tolerance_grid``, distinct whole numbers of runs of at least 0, and
    ``tolerance`` the Config Score_Tolerance ``design_choices.D20_score_tolerance``. The count at a
    grid tolerance t is the number of distinct ``match_id`` values of
    ``exclusion_conditions(checks, t)``. A larger tolerance only removes runs-check conditions,
    so the count never rises as the tolerance rises.

    Returns one row per grid value, in grid order, on a RangeIndex, with the
    EXCLUSION_COUNT_COLUMNS: ``score_tolerance`` (int32), ``n_excluded_matches`` (int32) and
    ``is_config_value`` (bool: true on the row whose tolerance equals ``tolerance``; no row is
    flagged when the grid does not list it). write_reconstruction_audit writes the table as
    ``exclusion_counts.csv`` and summarises it in ``audit.md``.

    Raises TypeError or ValueError as exclusion_conditions does, and when ``grid`` is not a list
    of distinct whole numbers of at least 0.
    """
    allowed = _score_tolerance(tolerance)
    values = _tolerance_grid(grid)
    rows = _selected_rows(checks, "checks")
    counts = [len(np.unique(rows.match_id[_exclusion_mask(rows, value)])) for value in values]
    return pd.DataFrame(
        {
            "score_tolerance": np.array(values, dtype=np.int32),
            "n_excluded_matches": np.array(counts, dtype=np.int32),
            "is_config_value": np.array([value == allowed for value in values], dtype=bool),
        }
    )


def excluded_flags(match_id: pd.Series, conditions: pd.DataFrame, *, exclude_failing_matches: bool) -> pd.Series:
    """The ``excluded`` flag of each Delivery (criterion 24).

    ``match_id`` holds the ``match_id`` of each Delivery (the column of the Delivery table).
    ``conditions`` is the exclusion_conditions table of the selected Extras_Encoding at the Config
    Score_Tolerance, and ``exclude_failing_matches`` the Config option
    ``data_rules.exclude_failing_matches``. Where the option is on, a Delivery is flagged exactly
    when its match has a row in ``conditions``, so every Delivery of an excluded match is flagged,
    Super_Over_Innings included. Where it is off, no Delivery is flagged and ``conditions`` is not
    read. Every model-training, validation and rating filter drops the flagged Deliveries.

    Returns a bool Series named ``excluded`` on the index of ``match_id``. Raises DataError when a
    ``match_id`` is missing or not a whole number, TypeError when ``exclude_failing_matches`` is
    not a bool, and ValueError when ``conditions`` has no ``match_id`` column or a missing
    ``match_id``.
    """
    if not isinstance(exclude_failing_matches, (bool, np.bool_)):
        raise TypeError(
            "exclude_failing_matches must be the Config option data_rules.exclude_failing_matches, a bool; "
            f"got {type(exclude_failing_matches).__name__} {exclude_failing_matches!r}"
        )
    ids = _whole_numbers(match_id.to_frame("match_id"), ["match_id"], "The excluded flags cannot be set")
    values = ids["match_id"].values
    if not exclude_failing_matches:
        return pd.Series(np.zeros(len(values), dtype=bool), index=match_id.index, name="excluded")
    if not isinstance(conditions, pd.DataFrame) or "match_id" not in conditions.columns:
        raise ValueError("conditions must be the exclusion_conditions table, with a match_id column")
    excluded_ids, missing = _nullable_values(conditions["match_id"], "Int64", "conditions match_id")
    if missing.any():
        raise ValueError("conditions must give the match_id of every condition, as exclusion_conditions returns")
    return pd.Series(np.isin(values, np.unique(excluded_ids)), index=match_id.index, name="excluded")


# ------------------------------------------- Score-reconstruction section of audit.md (criteria 22 and 27)

#: Heading of the section that write_reconstruction_audit adds to the Auditor's ``audit.md`` text.
RECONSTRUCTION_HEADING = "## Score reconstruction (Requirement 5)"

#: Exact_Pass_Rate, in percent, below which a Score_Check of the selected Extras_Encoding gets a
#: warning in ``audit.md``. Fixed by Requirement 5, criterion 22; it changes no result.
PASS_RATE_WARNING = 95.0


class _RateTable(NamedTuple):
    """A pass_rate_table read by _read_rate_table."""

    names: list[str]  # candidate names in table order
    position: int  # row of the selected encoding
    exact: np.ndarray  # (candidates, 5) Exact_Pass_Rates in SCORE_CHECKS order; NaN where undefined
    within: np.ndarray  # (5,) Within_Tolerance_Pass_Rates of the selected encoding

    @property
    def selected(self) -> str:
        return self.names[self.position]


def _read_rate_table(rates: pd.DataFrame) -> _RateTable:
    """Read a pass_rate_table; ValueError unless it has every column and marks exactly one selected row."""
    names = _rate_columns(rates, "rates")
    absent = [column for column in (*WITHIN_TOL_COLUMNS.values(), "selected") if column not in rates.columns]
    if absent:
        raise ValueError(f"rates must be a pass_rate_table; it lacks {', '.join(absent)}")
    flags = pd.array(rates["selected"], dtype="boolean").to_numpy(dtype=bool, na_value=False)
    if int(flags.sum()) != 1:
        raise ValueError(
            f"rates must mark exactly one selected Extras_Encoding, as pass_rate_table does; it marks {int(flags.sum())}"
        )
    position = int(np.flatnonzero(flags)[0])
    exact = rates[list(EXACT_COLUMNS.values())].to_numpy(dtype=np.float64, na_value=np.nan)
    within = rates[list(WITHIN_TOL_COLUMNS.values())].to_numpy(dtype=np.float64, na_value=np.nan)[position]
    return _RateTable(names, position, exact, within)


def _rate_warnings(read: _RateTable) -> list[tuple[str, float]]:
    """(check, Exact_Pass_Rate) of each Score_Check of the selected encoding whose rate is below PASS_RATE_WARNING."""
    rates = read.exact[read.position]
    return [
        (check, float(rate))
        for check, rate in zip(SCORE_CHECKS, rates)
        if not math.isnan(rate) and rate < PASS_RATE_WARNING
    ]


def _count_text(count: int, noun: str, plural: str | None = None) -> str:
    """``count`` with ``noun``, or with ``plural`` (``noun`` + "s" unless given) when the count is not 1."""
    return f"{count:,} {noun}" if count == 1 else f"{count:,} {plural or noun + 's'}"


def _runs_text(runs: int) -> str:
    return "1 run" if runs == 1 else f"{runs:,} runs"


def _rate_cell(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.3f}"


def _percent(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.3f}%"


def _markdown_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def _check_columns(table: Any, columns: Sequence[str], name: str, maker: str) -> None:
    if not isinstance(table, pd.DataFrame):
        raise TypeError(f"{name} must be the DataFrame that {maker} returns; got {type(table).__name__}")
    absent = [column for column in columns if column not in table.columns]
    if absent:
        raise ValueError(f"{name} must be the table that {maker} returns; it lacks {', '.join(absent)}")


def _reconstruction_section(
    read: _RateTable, conditions: pd.DataFrame, counts: pd.DataFrame, tolerance: int, exclude_on: bool
) -> str:
    """The score-reconstruction section of ``audit.md``, ending with one newline."""
    selected = f"`{read.selected}`"
    runs = _runs_text(tolerance)
    lines = [
        RECONSTRUCTION_HEADING,
        "",
        f"The Score_Reconstructor (the `{STAGE_NAME}` Pipeline_Stage) rebuilt the team runs of every Delivery "
        f"under {_count_text(len(read.names), 'candidate Extras_Encoding')} and checked the rebuilt scores of each "
        "match against its recorded result (criteria 16-20).",
        "",
        f"Selected Extras_Encoding (criterion 7): {selected}, with the highest mean Exact_Pass_Rate "
        f"({_percent(_mean_rate(read.exact[read.position]))}).",
        "",
        "### Pass rates (criterion 21)",
        "",
        "Exact_Pass_Rate of each Score_Check, in percent of the matches the check applies to, as in "
        "`pass_rates.csv`:",
        "",
    ]
    header = ["Extras_Encoding", *(f"`{check}`" for check in SCORE_CHECKS), "Mean", "Selected"]
    rows = [
        [
            f"`{name}`",
            *(_rate_cell(value) for value in read.exact[i]),
            _rate_cell(_mean_rate(read.exact[i])),
            "yes" if i == read.position else "",
        ]
        for i, name in enumerate(read.names)
    ]
    lines += _markdown_table(header, rows)
    within = ", ".join(f"`{check}` {_percent(value)}" for check, value in zip(SCORE_CHECKS, read.within))
    lines += [
        "",
        f"Within_Tolerance_Pass_Rates of {selected} at the Score_Tolerance of {runs}: {within}. A runs-margin, "
        f"chase or tie check passes within tolerance when it misses by at most {runs}; the wickets-margin and "
        "over checks must pass exactly.",
    ]
    if np.isnan(read.exact).any() or np.isnan(read.within).any():
        lines += ["", "n/a: the check applies to no match."]

    lines += ["", "### Warnings (criterion 22)", ""]
    low = _rate_warnings(read)
    if low:
        lines += [
            f"- Warning: the Exact_Pass_Rate of the `{check}` check under {selected} is {rate:.3f}%, below "
            f"{PASS_RATE_WARNING:.1f}%."
            for check, rate in low
        ]
    else:
        lines.append(f"No Exact_Pass_Rate under {selected} is below {PASS_RATE_WARNING:.1f}%.")

    lines += ["", "### Match exclusion (criteria 24 and 27)", ""]
    if not exclude_on:
        lines.append(
            "The Config option `data_rules.exclude_failing_matches` is off, so no match is excluded from model "
            f"training and player rating. `{EXCLUDED_MATCHES_FILE}` and `{EXCLUSION_COUNTS_FILE}` are not written, "
            "and copies left by earlier Runs are deleted."
        )
        return "\n".join(lines) + "\n"
    n_excluded = int(conditions["match_id"].nunique())
    if n_excluded:
        excluded = (
            f"- Matches excluded from model training and player rating: {n_excluded:,}. Under {selected}, each "
            f"fails the over check or the wickets-margin check, or misses the runs-margin, chase or tie check by "
            f"more than {runs}; `{EXCLUDED_MATCHES_FILE}` lists the "
            f"{_count_text(len(conditions), 'condition')} they meet."
        )
    else:
        excluded = (
            f"- Matches excluded from model training and player rating: 0. No match meets a condition of "
            f"criterion 24 under {selected}, so `{EXCLUDED_MATCHES_FILE}` holds its header only."
        )
    if tolerance == 0:
        kept = f"- With a Score_Tolerance of 0 runs, the excluded matches are exactly the matches in `{FAILING_MATCHES_FILE}`."
    else:
        kept = (
            f"- `{FAILING_MATCHES_FILE}` also lists the runs checks that miss by at most {runs}; they do not "
            "exclude a match."
        )
    lines += [
        f"- Score_Tolerance (D20, read from the Config): {runs}.",
        excluded,
        kept,
        "",
        "Matches that criterion 24 would exclude at each Score_Tolerance of the Config reporting grid, as in "
        f"`{EXCLUSION_COUNTS_FILE}`:",
        "",
    ]
    count_rows = [
        [f"{int(value)} (the Config value)" if bool(flag) else f"{int(value)}", f"{int(count):,}"]
        for value, count, flag in zip(
            counts["score_tolerance"].tolist(), counts["n_excluded_matches"].tolist(), counts["is_config_value"].tolist()
        )
    ]
    if count_rows:
        lines += _markdown_table(["Score_Tolerance (runs)", "Matches excluded"], count_rows)
    else:
        lines.append("The reporting grid lists no Score_Tolerance.")
    return "\n".join(lines) + "\n"


def _write_text(path: Path, text: str) -> None:
    """Write ``text`` as UTF-8 to ``path`` through a temporary file renamed in place."""
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_bytes(text.encode("utf-8"))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_reconstruction_audit(
    outdir: str | Path,
    audit_text: str,
    rates: pd.DataFrame,
    conditions: pd.DataFrame,
    counts: pd.DataFrame,
    tolerance: int,
    exclude_on: bool,
) -> str:
    """Rewrite ``audit.md`` as the Auditor's text plus the score-reconstruction section (criteria 22 and 27).

    ``outdir`` is the audit folder (``outputs/audit/`` in a Run) and is created when absent.
    ``audit_text`` is the Auditor's ``audit.md`` text as saved in the ``audit`` checkpoint (JSON
    ``audit_text``), so rerunning this stage alone never repeats the section. ``rates`` is the
    pass_rate_table, ``conditions`` the exclusion_conditions table and ``counts`` the
    exclusion_counts table of the selected Extras_Encoding; ``tolerance`` is the Config
    Score_Tolerance ``design_choices.D20_score_tolerance`` and ``exclude_on`` the Config option
    ``data_rules.exclude_failing_matches``.

    The section, headed RECONSTRUCTION_HEADING, gives the selected encoding, the Exact_Pass_Rates
    of every candidate and the Within_Tolerance_Pass_Rates of the selected one, and a warning
    with the check name and rate for each Score_Check whose Exact_Pass_Rate under the selected
    encoding is below PASS_RATE_WARNING (95.0%; criterion 22). Where ``exclude_on`` is true it
    also gives the Score_Tolerance, the number of distinct excluded matches and the count at each
    grid tolerance (criterion 27), and ``excluded_matches.csv`` (``conditions``) and
    ``exclusion_counts.csv`` (``counts``) are written to ``outdir``. Where it is false the section
    says that no match is excluded, ``conditions`` and ``counts`` are not read, and any copy of
    the two files left by an earlier Run is deleted. Every file is UTF-8, written through a
    temporary file renamed in place; ``audit.md`` is written last.

    Returns the ``audit.md`` text. Raises TypeError or ValueError when ``audit_text`` is not a
    string or already holds the section, when ``tolerance`` is not a whole number of at least 0,
    when ``exclude_on`` is not a bool, or when a table lacks a column.
    """
    from superr.audit import AUDIT_MD  # imported here because superr.audit imports this module

    if not isinstance(audit_text, str):
        raise TypeError(f"audit_text must be the Auditor's audit.md text; got {type(audit_text).__name__}")
    if RECONSTRUCTION_HEADING in audit_text.splitlines():
        raise ValueError(
            "audit_text already holds the score-reconstruction section; pass the Auditor's text from the "
            "audit checkpoint, not the audit.md file"
        )
    allowed = _score_tolerance(tolerance)
    if not isinstance(exclude_on, (bool, np.bool_)):
        raise TypeError(
            "exclude_on must be the Config option data_rules.exclude_failing_matches, a bool; "
            f"got {type(exclude_on).__name__} {exclude_on!r}"
        )
    exclude_on = bool(exclude_on)
    read = _read_rate_table(rates)
    if exclude_on:
        _check_columns(conditions, EXCLUSION_COLUMNS, "conditions", "exclusion_conditions")
        _check_columns(counts, EXCLUSION_COUNT_COLUMNS, "counts", "exclusion_counts")
    section = _reconstruction_section(read, conditions, counts, allowed, exclude_on)
    head = audit_text.rstrip("\n")
    text = f"{head}\n\n{section}" if head else section

    folder = Path(outdir)
    folder.mkdir(parents=True, exist_ok=True)
    if exclude_on:
        _write_audit_csv(folder, EXCLUDED_MATCHES_FILE, conditions, EXCLUSION_COLUMNS)
        _write_audit_csv(folder, EXCLUSION_COUNTS_FILE, counts, EXCLUSION_COUNT_COLUMNS)
    else:
        for file_name in (EXCLUDED_MATCHES_FILE, EXCLUSION_COUNTS_FILE):
            (folder / file_name).unlink(missing_ok=True)
    _write_text(folder / AUDIT_MD, text)
    return text


# ------------------------------------------------------------------- The balls and innings_info tables

#: Competition_Types (glossary), in the fixed order in which the Fold_Assigner deals them.
COMP_IPL = "IPL"
COMP_T20I = "T20I"
COMP_A_TEAM = "A_Team"
COMPETITION_TYPES: tuple[str, ...] = (COMP_IPL, COMP_T20I, COMP_A_TEAM)

_IPL_PREFIX = "IPL"  # a competition whose comp_name starts with this text is IPL
_T20I_FORMAT = "T20I"  # match_format of a T20I competition
_T20_FORMAT = "T20"  # match_format of every other T20 competition (A_Team)

#: The match columns of ``balls`` that match_columns returns after ``match_id``.
MATCH_INFO_COLUMNS: tuple[str, ...] = (
    "comp_type",
    "season",
    "calendar_year",
    "match_date",
    "knock_out",
    "venue_id",
    "venue_country",
    "target",
    "result_type",
)

#: The per-Delivery columns that balls_table takes from its ``derived`` argument.
BALLS_DERIVED_COLUMNS: tuple[str, ...] = (
    "cls",
    "is_super_over",
    "free_hit",
    "team_runs",
    "counted_wicket",
    "bowler_wicket",
    "run_out",
    "retire_out",
    "dismissed_is_striker",
    "dismissed_fallback",
    "sched_balls",
    "excluded",
)

#: Columns of the ``balls`` table, in the order of the design's Data Models table: keys, people,
#: outcome, derived and match columns. ``fold`` is not here: the Fold_Assigner adds it.
BALLS_COLUMNS: tuple[str, ...] = (
    # keys
    "row", "match_id", "innings", "overid", "over_ball_num", "innings_ball_num",
    # people
    "batting_team_id", "bowling_team_id", "striker_id", "non_striker_id", "bowler_id", "dismissed_player_id",
    "batting_type", "bowling_type", "pace_or_spin", "bowl_speed",
    # outcome
    "runs", "is_four", "is_six", "is_wicket", "dismissal_type_id", "dismissal_type", "extras_type", "extras_2_runs",
    # derived
    "cls", "is_super_over", "free_hit", "team_runs", "counted_wicket", "bowler_wicket", "run_out", "retire_out",
    "dismissed_is_striker", "dismissed_fallback",
    # match
    "comp_type", "season", "calendar_year", "match_date", "knock_out", "venue_id", "venue_country",
    "sched_balls", "target", "result_type", "excluded",
)  # fmt: skip

#: Columns of the ``innings_info`` table (design, Data Models): one row per encoding, match and innings.
INNINGS_INFO_COLUMNS: tuple[str, ...] = (
    "encoding",
    "match_id",
    "innings",
    "sched_balls",
    "target",
    "total",
    "wickets",
    "final_ball_runs",
)

# The columns that balls_table takes from the ordered Delivery table, by the dtype it gives them.
_BALLS_INT32: tuple[str, ...] = (
    "row", "match_id", "innings", "innings_ball_num", "striker_id", "bowler_id", "runs", "extras_2_runs",
)  # fmt: skip
_BALLS_NULLABLE_INT32: tuple[str, ...] = (
    "overid", "over_ball_num", "batting_team_id", "bowling_team_id", "non_striker_id", "dismissed_player_id",
    "dismissal_type_id",
)  # fmt: skip
_BALLS_TEXT: tuple[str, ...] = ("batting_type", "bowling_type", "pace_or_spin", "dismissal_type", "extras_type")
_BALLS_FLAGS: tuple[str, ...] = ("is_four", "is_six", "is_wicket")
_BALLS_FLOATS: tuple[str, ...] = ("bowl_speed",)
_BALLS_RAW: tuple[str, ...] = _BALLS_INT32 + _BALLS_NULLABLE_INT32 + _BALLS_TEXT + _BALLS_FLAGS + _BALLS_FLOATS
_BALLS_DERIVED_FLAGS: tuple[str, ...] = (
    "is_super_over", "free_hit", "counted_wicket", "bowler_wicket", "run_out", "retire_out", "dismissed_fallback",
    "excluded",
)  # fmt: skip
_INT32_MIN, _INT32_MAX = int(np.iinfo(np.int32).min), int(np.iinfo(np.int32).max)


def _int32_columns(
    frame: pd.DataFrame, columns: Sequence[str], context: str, *, nullable: bool
) -> dict[str, np.ndarray | pd.arrays.IntegerArray]:
    """``columns`` of ``frame`` as int32 arrays, or as nullable Int32 arrays when ``nullable``.

    Raises DataError (exit code 3), naming each column with its row count, when a value is not a
    whole number, lies outside the int32 range or (unless ``nullable``) is missing.
    """
    found = _whole_numbers(frame, columns, context, may_be_missing=columns if nullable else ())
    problems: list[str] = []
    converted: dict[str, np.ndarray | pd.arrays.IntegerArray] = {}
    for column, whole in found.items():
        outside = ~whole.missing & ((whole.values < _INT32_MIN) | (whole.values > _INT32_MAX))
        if outside.any():
            problems.append(f"{column} is outside the int32 range in {_rows(int(outside.sum()))}")
            continue
        values = whole.values.astype(np.int32)
        converted[column] = pd.arrays.IntegerArray(values, whole.missing.copy()) if nullable else values
    if problems:
        raise DataError("\n".join([f"{context}:"] + [f"  - {problem}" for problem in problems]))
    return converted


def _text_flags(series: pd.Series, test: Callable[[pd.Series], pd.Series]) -> np.ndarray:
    """``test`` applied to ``series`` as text, as a bool array with a missing value counted as False."""
    return test(series.astype("string")).fillna(False).to_numpy(dtype=bool)


def competition_types(competitions: pd.DataFrame) -> pd.Series:
    """Competition_Type of every competition (glossary): IPL, T20I or A_Team.

    ``competitions`` needs ``comp_id``, ``comp_name`` and ``match_format``. A competition is IPL
    when its ``comp_name`` starts with "IPL", else T20I when its ``match_format`` is ``T20I``, else
    A_Team when its ``match_format`` is ``T20`` (every other T20 competition, such as the Mens T20
    Emerging Teams Asia Cup 2024). Texts match exactly, with letter case. A competition that meets
    none of the three rules (another format, or a missing one) has no Competition_Type;
    match_columns stops the Run when a match belongs to such a competition.

    Returns a ``str`` Series named ``comp_type`` (COMP_IPL, COMP_T20I or COMP_A_TEAM; missing
    where no rule applies), indexed by ``comp_id`` (int32) in the row order of ``competitions``.
    Raises DataError (exit code 3) when a ``comp_id`` is missing, not a whole number or repeated.
    """
    context = "Competition_Types cannot be set"
    ids = _whole_numbers(competitions, ["comp_id"], context)["comp_id"].values
    unique, counts = np.unique(ids, return_counts=True)
    repeated = unique[counts > 1]
    if len(repeated):
        raise DataError(
            f"{context}:\n  - comp_id is repeated in competitions for {len(repeated):,} competition(s) "
            f"(for example {_shown_ids(repeated)})"
        )
    is_ipl = _text_flags(competitions["comp_name"], lambda text: text.str.startswith(_IPL_PREFIX))
    is_t20i = _text_flags(competitions["match_format"], lambda text: text.eq(_T20I_FORMAT))
    is_t20 = _text_flags(competitions["match_format"], lambda text: text.eq(_T20_FORMAT))
    values = np.full(len(ids), None, dtype=object)
    values[is_t20] = COMP_A_TEAM
    values[is_t20i] = COMP_T20I
    values[is_ipl] = COMP_IPL  # the IPL rule comes first, so it overrides the format rules
    return pd.Series(
        pd.array(values, dtype="str"),
        index=pd.Index(ids.astype(np.int32), name="comp_id"),
        name="comp_type",
    )


def match_columns(
    matches: pd.DataFrame, competitions: pd.DataFrame, parsed: pd.DataFrame, target: pd.Series
) -> pd.DataFrame:
    """The match columns of ``balls`` for every match (design, Data Models).

    ``matches`` needs ``match_id``, ``comp_id``, ``match_date`` (datetime), ``knock_out``,
    ``venue_id`` and ``venue_country``; ``competitions`` needs ``comp_id``, ``comp_name``,
    ``match_format`` and ``season``. ``parsed`` is the parse_results table of ``matches`` and
    ``target`` the targets Series of ``matches`` under the selected Extras_Encoding.

    Returns one row per match, sorted by ``match_id``, on a RangeIndex, with ``match_id`` (int32)
    and the MATCH_INFO_COLUMNS:

    * ``comp_type`` (str): the Competition_Type of the match's competition (competition_types);
    * ``season`` (string): the competition's ``season`` text, the Season (split values such as
      ``2022/23`` are Seasons in their own right);
    * ``calendar_year`` (nullable Int32): the year of ``match_date``, the Calendar_Year;
    * ``match_date`` (as loaded), ``knock_out`` and ``venue_id`` (nullable Int32) and
      ``venue_country`` (string), from ``matches``;
    * ``target`` (nullable Int32): the Target (criteria 9-10), missing when it is undefined;
    * ``result_type`` (str): the result type from the Result_Parser.

    Raises DataError (exit code 3) when a ``match_id`` or ``comp_id`` of ``matches`` is missing,
    non-whole or repeated (``match_id``), when a match's ``comp_id`` is not in ``competitions``,
    when a match's competition has no Competition_Type, or when ``knock_out`` or ``venue_id`` is
    not a whole number in the int32 range. Raises TypeError when ``match_date`` is not a datetime
    column, and ValueError when ``parsed`` or ``target`` lacks a match.
    """
    context = "The match columns of the balls table cannot be built"
    ids = _match_ids(matches, context)
    order = np.argsort(ids, kind="stable")
    frame = matches.iloc[order].reset_index(drop=True)
    match_id = ids[order]
    comp = _whole_numbers(frame, ["comp_id"], context)["comp_id"].values
    types = competition_types(competitions)
    position = pd.Index(types.index.to_numpy(dtype=np.int64)).get_indexer(comp)
    unknown = position < 0
    if unknown.any():
        shown = ", ".join(
            f"match {m}: comp_id {c}" for m, c in zip(match_id[unknown][:3].tolist(), comp[unknown][:3].tolist())
        )
        raise DataError(
            f"{context}:\n  - the comp_id of {_count_text(int(unknown.sum()), 'match', 'matches')} is not in competitions "
            f"(for example {shown})"
        )
    comp_type = types.iloc[position].reset_index(drop=True)
    untyped = comp_type.isna().to_numpy(dtype=bool)
    if untyped.any():
        rows = np.unique(position[untyped])
        shown = "; ".join(
            f"comp_id {types.index[row]}: comp_name {competitions['comp_name'].iloc[row]!r}, "
            f"match_format {competitions['match_format'].iloc[row]!r}"
            for row in rows[:3].tolist()
        )
        raise DataError(
            f"{context}:\n  - no Competition_Type applies to {_count_text(len(rows), 'competition')} used by "
            f"{_count_text(int(untyped.sum()), 'match', 'matches')}: the comp_name does not start with "
            f"{_IPL_PREFIX!r} and the match_format is neither {_T20I_FORMAT!r} nor {_T20_FORMAT!r} ({shown})"
        )
    dates = frame["match_date"]
    if not pd.api.types.is_datetime64_any_dtype(dates.dtype):
        raise TypeError(f"matches match_date must be a datetime column, as the Data_Loader reads it; got {dates.dtype}")
    numbers = _int32_columns(frame, ["knock_out", "venue_id"], context, nullable=True)

    target_index = pd.Index(np.asarray(target.index, dtype=np.int64))
    target_at = target_index.get_indexer(match_id)
    parsed_ids = _whole_numbers(parsed, ["match_id"], "The parsed results cannot be used")["match_id"].values
    parsed_at = pd.Index(parsed_ids).get_indexer(match_id)
    for name, at in (("target", target_at), ("parsed", parsed_at)):
        if (at < 0).any():
            raise ValueError(
                f"{name} must cover every match of matches, as targets and parse_results do; it has no entry for "
                f"match {_shown_ids(match_id[at < 0])}"
            )
    return pd.DataFrame(
        {
            "match_id": match_id.astype(np.int32),
            "comp_type": comp_type.array,
            "season": competitions["season"].astype("string").iloc[position].array,
            "calendar_year": pd.array(dates.dt.year, dtype="Int32"),
            "match_date": dates.array,
            "knock_out": numbers["knock_out"],
            "venue_id": numbers["venue_id"],
            "venue_country": frame["venue_country"].astype("string").array,
            "target": pd.array(target.iloc[target_at].to_numpy(dtype=object, na_value=None), dtype="Int32"),
            "result_type": pd.array(parsed["result_type"].iloc[parsed_at].to_numpy(dtype=object), dtype="str"),
        }
    )


def dismissed_batter_flags(balls: pd.DataFrame, wickets: pd.DataFrame) -> pd.DataFrame:
    """Which batter each Counted_Wicket dismisses (Requirement 15, criteria 9, 10 and 12; D14).

    ``balls`` needs ``striker_id``, ``non_striker_id`` and ``dismissed_player_id``; ``wickets`` is
    the wicket_flags table of ``balls``, whose ``counted_wicket`` and ``retire_out`` columns are
    used. The dismissed batter of a Counted_Wicket is the striker when ``dismissed_player_id``
    equals ``striker_id``, else the non-striker when it equals ``non_striker_id`` (criterion 9).
    When it is blank or equals neither, a Counted_Wicket other than Retire Out falls back to the
    striker (criterion 10), while a Retire Out stays unresolved: no rule names its batter, and a
    Retire Out charges no batter (criterion 12).

    Returns a DataFrame on the index of ``balls`` with columns:

    * ``dismissed_is_striker`` (nullable boolean): true when the dismissed batter is the striker,
      false when it is the non-striker, and missing on a Delivery without a Counted_Wicket and on
      an unresolved Retire Out;
    * ``dismissed_fallback`` (bool): true on each Counted_Wicket other than Retire Out whose
      ``dismissed_player_id`` is blank or equals neither batter, so that the striker is used. The
      Value_Engine reports their count (criterion 10).

    Raises DataError (exit code 3) when a present id is not a whole number.
    """
    columns = ["striker_id", "non_striker_id", "dismissed_player_id"]
    ids = _whole_numbers(balls, columns, "The dismissed batters cannot be resolved", may_be_missing=columns)
    counted = _flag_array(wickets["counted_wicket"], balls.index, "counted_wicket")
    retire_out = _flag_array(wickets["retire_out"], balls.index, "retire_out")
    out, striker, non_striker = ids["dismissed_player_id"], ids["striker_id"], ids["non_striker_id"]
    is_striker = counted & ~out.missing & ~striker.missing & (out.values == striker.values)
    is_non_striker = (
        counted & ~is_striker & ~out.missing & ~non_striker.missing & (out.values == non_striker.values)
    )
    fallback = counted & ~is_striker & ~is_non_striker & ~retire_out
    resolved = is_striker | is_non_striker | fallback
    return pd.DataFrame(
        {
            "dismissed_is_striker": pd.arrays.BooleanArray(is_striker | fallback, ~resolved),
            "dismissed_fallback": fallback,
        },
        index=balls.index,
    )


def balls_table(ordered: pd.DataFrame, derived: pd.DataFrame, match_info: pd.DataFrame) -> pd.DataFrame:
    """The ``balls`` table of the ``reconstruct`` checkpoint (design, Data Models), without ``fold``.

    ``ordered`` is the order_deliveries table of ``ball_by_ball`` (Requirement 5 order, with
    ``row``). ``derived`` holds the BALLS_DERIVED_COLUMNS on the index of ``ordered``:

    * ``cls`` (classify), ``is_super_over`` (super_over_flags), ``free_hit`` (free_hit_flags);
    * ``team_runs``: the team runs under the selected Extras_Encoding;
    * ``counted_wicket``, ``bowler_wicket``, ``run_out`` and ``retire_out`` (wicket_flags);
    * ``dismissed_is_striker`` and ``dismissed_fallback`` (dismissed_batter_flags);
    * ``sched_balls`` (delivery_scheduled_balls) and ``excluded`` (excluded_flags).

    ``match_info`` is the match_columns table, which gives every Delivery its match columns.

    Returns one row per Delivery, in the order of ``ordered``, on a RangeIndex, with the
    BALLS_COLUMNS. Integer IDs and counts that are never blank (``row``, ``match_id``,
    ``innings``, ``innings_ball_num``, ``striker_id``, ``bowler_id``, ``runs``, ``extras_2_runs``,
    ``team_runs``, ``sched_balls``) are int32 and ``cls`` is int8; the other integer columns are
    nullable Int32. The derived flags are bool except ``dismissed_is_striker`` (nullable
    boolean); ``is_four``, ``is_six`` and ``is_wicket`` keep their nullable boolean values and
    the text columns are ``string``. ``bowl_speed`` is float64, NaN where missing (0 means not
    recorded, as delivered). The match columns keep the dtypes of match_columns.

    Raises DataError (exit code 3) when a Delivery's match is not in ``match_info``, or when an
    integer column holds a value that is not a whole number, lies outside the int32 range or is
    missing where it may not be. Raises TypeError or ValueError when a column is absent, when
    ``derived`` is not on the index of ``ordered``, or when a derived column has the wrong dtype.
    """
    for table, columns, name in (
        (ordered, _BALLS_RAW, "ordered (the order_deliveries table)"),
        (derived, BALLS_DERIVED_COLUMNS, "derived"),
        (match_info, ("match_id", *MATCH_INFO_COLUMNS), "match_info (the match_columns table)"),
    ):
        if not isinstance(table, pd.DataFrame):
            raise TypeError(f"{name} must be a DataFrame; got {type(table).__name__}")
        absent = [column for column in columns if column not in table.columns]
        if absent:
            raise ValueError(f"{name} lacks the column(s) {', '.join(absent)}")
    index = ordered.index
    if not derived.index.equals(index):
        raise ValueError("derived must be on the index of ordered")
    context = "The balls table cannot be built"
    columns: dict[str, Any] = {}
    columns.update(_int32_columns(ordered, _BALLS_INT32, context, nullable=False))
    columns.update(_int32_columns(ordered, _BALLS_NULLABLE_INT32, context, nullable=True))
    for column in _BALLS_TEXT:
        columns[column] = ordered[column].astype("string").array
    for column in _BALLS_FLAGS:
        columns[column] = ordered[column].astype("boolean").array
    for column in _BALLS_FLOATS:
        columns[column] = pd.to_numeric(ordered[column]).to_numpy(dtype=np.float64, na_value=np.nan)

    columns["cls"] = _class_codes(derived["cls"], index)
    for column in _BALLS_DERIVED_FLAGS:
        columns[column] = _flag_array(derived[column], index, column)
    for column in ("team_runs", "sched_balls"):
        values = _int_array(derived[column], index, column)
        if len(values) and (values.min() < _INT32_MIN or values.max() > _INT32_MAX):
            raise ValueError(f"{column} holds values outside the int32 range")
        columns[column] = values.astype(np.int32)
    columns["dismissed_is_striker"] = pd.array(derived["dismissed_is_striker"].to_numpy(dtype=object), dtype="boolean")

    info_ids = _whole_numbers(match_info, ["match_id"], "The match columns cannot be used")["match_id"].values
    lookup = pd.Index(info_ids)
    if not lookup.is_unique:
        raise ValueError("match_info must hold one row per match, as match_columns returns")
    position = lookup.get_indexer(columns["match_id"].astype(np.int64))
    unknown = position < 0
    if unknown.any():
        raise DataError(
            f"{context}:\n  - {_rows(int(unknown.sum()))} of the Deliveries belong to matches that are not in "
            f"matches (for example match {_shown_ids(np.unique(columns['match_id'][unknown]))})"
        )
    for column in MATCH_INFO_COLUMNS:
        columns[column] = match_info[column].iloc[position].array
    return pd.DataFrame({column: columns[column] for column in BALLS_COLUMNS})


def innings_info(matches: pd.DataFrame, sched: pd.DataFrame, aggregates: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """The ``innings_info`` table: every innings under every candidate Extras_Encoding (design, Data Models).

    ``matches`` holds ``match_id`` and ``revised_target`` of every match, ``sched`` is its
    scheduled_balls table and ``aggregates`` maps each candidate name, in Config order, to its
    innings_aggregates table.

    Returns one row per encoding and innings present in its aggregates (Super_Over_Innings
    included), sorted by encoding in the order of ``aggregates`` and then by ``match_id`` and
    ``innings``, on a RangeIndex, with the INNINGS_INFO_COLUMNS: ``encoding`` (str);
    ``match_id``, ``innings``, ``sched_balls``, ``total``, ``wickets`` and ``final_ball_runs``
    (int32); and ``target`` (nullable Int32). ``sched_balls`` is the innings' Scheduled_Balls as in
    delivery_scheduled_balls, so a Super_Over_Innings has 120. ``target`` is the Target of the
    innings' match under the encoding (targets), the score innings 2 must reach, on every innings
    of the match; it is missing when it is undefined.

    Raises DataError (exit code 3) as targets and delivery_scheduled_balls do, for example when
    innings 1 or 2 of a match that is not in ``matches`` holds Deliveries. Raises TypeError when a
    candidate name is not a non-empty string.
    """
    frames = []
    for name, agg in aggregates.items():
        if not isinstance(name, str) or not name:
            raise TypeError(f"aggregates keys must be candidate names (non-empty strings); got {name!r}")
        target = targets(matches, agg)
        found = _whole_numbers(
            agg, ["match_id", "innings", "total", "wickets", "final_ball_runs"], "The innings aggregates cannot be used"
        )
        match_id = found["match_id"].values
        position = pd.Index(target.index.to_numpy(dtype=np.int64)).get_indexer(match_id)
        values = _take(target.to_numpy(dtype=np.int64, na_value=0), position, 0)
        missing = _take(target.isna().to_numpy(dtype=bool), position, True)
        frames.append(
            pd.DataFrame(
                {
                    "encoding": pd.array(np.full(len(match_id), name, dtype=object), dtype="str"),
                    "match_id": match_id.astype(np.int32),
                    "innings": found["innings"].values.astype(np.int32),
                    "sched_balls": delivery_scheduled_balls(agg, sched).to_numpy(dtype=np.int32),
                    "target": pd.arrays.IntegerArray(values.astype(np.int32), missing),
                    "total": found["total"].values.astype(np.int32),
                    "wickets": found["wickets"].values.astype(np.int32),
                    "final_ball_runs": found["final_ball_runs"].values.astype(np.int32),
                }
            )
        )
    if not frames:
        empty = np.zeros(0, dtype=np.int32)
        return pd.DataFrame(
            {
                "encoding": pd.array([], dtype="str"),
                **{column: empty for column in ("match_id", "innings", "sched_balls")},
                "target": pd.array([], dtype="Int32"),
                **{column: empty for column in ("total", "wickets", "final_ball_runs")},
            }
        )
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------ The reconstruct Pipeline_Stage

#: Name of the Pipeline_Stage that runs the Score_Reconstructor, and of its checkpoint.
STAGE_NAME = "reconstruct"

#: Tables of the ``reconstruct`` checkpoint.
BALLS_TABLE = "balls"
INNINGS_INFO_TABLE = "innings_info"
SCORE_CHECKS_TABLE = "score_checks"
PASS_RATES_TABLE = "pass_rates"

#: Name of the data-derived Method_Constant that records the selected Extras_Encoding.
SELECTED_ENCODING_CONSTANT = "reconstruct.selected_encoding"

_LOAD_STAGE = "load"
_COMPONENT = "Score_Reconstructor"  # the glossary component that produces the selected encoding


def _audit_text(saved: Any) -> str:
    """The Auditor's audit.md text from the audit checkpoint's JSON value ``audit_text``."""
    text = saved.get("text") if isinstance(saved, dict) else None
    if not isinstance(text, str):
        raise StageError(
            "the audit checkpoint's JSON value audit_text must hold the audit.md text under 'text'; "
            "run the audit stage again or start a full Run"
        )
    return text


def _shown(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def run_stage(ctx: StageContext) -> None:
    """The ``reconstruct`` Pipeline_Stage: rebuild and check the scores, exclude matches, save the checkpoint.

    Reads ``ball_by_ball``, ``matches`` and ``competitions`` from the ``load`` checkpoint and
    ``dismissal_classes`` (whose Counted_Wicket codes drive wicket_flags) and ``audit_text`` from
    the ``audit`` checkpoint. Runs every section of this module in order with the Config values
    it needs: the class mapping, the candidate encodings, the D14 codes, the Score_Tolerance D20,
    the reporting grid and ``exclude_failing_matches``. The over check runs once and the other
    Score_Checks once per candidate; the selected encoding's rows feed the pass rates, the failed
    checks and the exclusion conditions.

    Writes ``unparsed_results.csv``, ``pass_rates.csv``, ``failing_matches.csv``,
    ``excluded_matches.csv`` and ``exclusion_counts.csv`` (only where ``exclude_failing_matches``
    is on; otherwise earlier copies are deleted) and ``audit.md`` to ``outputs/audit/``. Saves the
    tables ``balls``, ``innings_info`` (every candidate), ``score_checks`` (every candidate) and
    ``pass_rates``, and records the selected encoding as the data-derived Method_Constant
    ``reconstruct.selected_encoding`` (Score_Reconstructor). Warnings (Exact_Pass_Rates below
    95.0%, unparsed results, excluded matches) also go to the Run_Log.

    Raises DataError (exit code 3) for the data problems named in the sections above.
    """
    from superr import audit  # imported here because superr.audit imports this module

    settings, store, log = ctx.settings, ctx.store, ctx.log
    classes = settings.get("data_rules.extras_type_classes")
    candidates = settings.get("data_rules.candidate_encodings")
    exclude_on = settings.get("data_rules.exclude_failing_matches")
    grid = settings.get("data_rules.score_tolerance_grid")
    tolerance = settings.get("design_choices.D20_score_tolerance")
    bowler_codes = settings.get("design_choices.D14_bowler_wicket_codes")
    run_out_code = settings.get("design_choices.D14_run_out_code")
    retire_out_code = settings.get("design_choices.D14_retire_out_code")
    outdir = settings.path("outputs") / audit.AUDIT_FOLDER

    matches = store.load_table(_LOAD_STAGE, "matches")
    competitions = store.load_table(_LOAD_STAGE, "competitions")
    dismissal_classes = store.load_table(audit.STAGE_NAME, audit.CLASSES_TABLE)
    audit_text = _audit_text(store.load_json(audit.STAGE_NAME, audit.TEXT_JSON))

    # Deliveries in the Requirement 5 order, their classes, flags and team runs (criteria 1-12).
    check_mapping(classes)
    balls = order_deliveries(store.load_table(_LOAD_STAGE, "ball_by_ball"))
    cls = classify(balls["extras_type"], classes)
    sched = scheduled_balls(matches)
    wickets = wicket_flags(
        balls, audit.counted_wicket_codes(dismissal_classes), bowler_codes, run_out_code, retire_out_code
    )
    team_runs = candidate_team_runs(encoding_inputs(balls, cls), candidates)

    # Score_Checks of every candidate, the selected encoding and its pass rates (criteria 7, 13-23).
    parsed = parse_results(matches)
    over_ok = over_check(balls, cls)
    aggregates = {name: innings_aggregates(balls, runs, wickets["counted_wicket"]) for name, runs in team_runs.items()}
    checks = {name: run_checks(matches, agg, parsed, over_ok, encoding=name) for name, agg in aggregates.items()}
    rates = exact_pass_rates(checks)
    selected = select_encoding(rates)
    chosen = checks[selected]
    pass_rates = pass_rate_table(rates, selected, within_tolerance_pass_rates(chosen, tolerance))
    failing = failing_matches(chosen)

    # Match exclusion (criteria 24 and 27).
    conditions = exclusion_conditions(chosen, tolerance)
    counts = exclusion_counts(chosen, grid, tolerance=tolerance)
    excluded = excluded_flags(balls["match_id"], conditions, exclude_failing_matches=exclude_on)

    # The balls and innings_info tables.
    derived = pd.concat(
        [
            cls,
            super_over_flags(balls["innings"]),
            free_hit_flags(balls, cls),
            pd.Series(team_runs[selected], index=balls.index, name="team_runs"),
            wickets,
            dismissed_batter_flags(balls, wickets),
            delivery_scheduled_balls(balls, sched),
            excluded,
        ],
        axis=1,
    )
    info = match_columns(matches, competitions, parsed, targets(matches, aggregates[selected]))
    table = balls_table(balls, derived, info)
    per_innings = innings_info(matches, sched, aggregates)
    score_checks = pd.concat(list(checks.values()), ignore_index=True)

    # Deliverables in outputs/audit/, audit.md last.
    write_unparsed(outdir, parsed, matches)
    write_pass_rates(outdir, pass_rates)
    write_failing_matches(outdir, failing)
    write_reconstruction_audit(outdir, audit_text, pass_rates, conditions, counts, tolerance, exclude_on)

    # Checkpoint and the data-derived Method_Constant.
    store.save_table(STAGE_NAME, BALLS_TABLE, table)
    store.save_table(STAGE_NAME, INNINGS_INFO_TABLE, per_innings)
    store.save_table(STAGE_NAME, SCORE_CHECKS_TABLE, score_checks)
    store.save_table(STAGE_NAME, PASS_RATES_TABLE, pass_rates)
    ctx.constants.record(SELECTED_ENCODING_CONSTANT, selected, _COMPONENT)

    # Run_Log.
    shown = _shown(outdir, ctx.root)
    read = _read_rate_table(pass_rates)
    means = ", ".join(f"{name} {_percent(_mean_rate(row))}" for name, row in zip(read.names, read.exact))
    log.info(
        f"selected Extras_Encoding {selected} (Requirement 5, criterion 7); mean Exact_Pass_Rate of each "
        f"candidate: {means}",
        stage=STAGE_NAME,
    )
    for check, rate in _rate_warnings(read):
        log.warning(
            f"the Exact_Pass_Rate of the {check} check under the selected Extras_Encoding {selected} is "
            f"{rate:.3f}%, below {PASS_RATE_WARNING:.1f}% (Requirement 5, criterion 22); see {shown}/{audit.AUDIT_MD}",
            stage=STAGE_NAME,
        )
    n_unparsed = int((parsed["result_type"] == UNPARSED).sum())
    if n_unparsed:
        log.warning(
            f"no Result_Parser rule matches {_count_text(n_unparsed, 'result string')} (Requirement 5, "
            f"criterion 15); see {shown}/{UNPARSED_RESULTS_FILE}",
            stage=STAGE_NAME,
        )
    n_with_balls = int(table["match_id"].nunique())
    log.info(
        f"{_count_text(len(failing), 'failed Score_Check')} under {selected} in "
        f"{int(failing['match_id'].nunique()):,} of the {n_with_balls:,} matches with Deliveries "
        f"(Requirement 5, criterion 23; {shown}/{FAILING_MATCHES_FILE})",
        stage=STAGE_NAME,
    )
    if exclude_on:
        n_excluded = int(conditions["match_id"].nunique())
        at_grid = ", ".join(
            f"{_runs_text(int(value))}: {int(count):,}"
            for value, count in zip(counts["score_tolerance"].tolist(), counts["n_excluded_matches"].tolist())
        )
        message = (
            f"{n_excluded:,} of {n_with_balls:,} matches with Deliveries are excluded from model training and player "
            f"rating at the Score_Tolerance of {_runs_text(tolerance)} (D20; Requirement 5, criterion 24); excluded "
            f"at each grid tolerance: {at_grid or 'no grid value'}; see {shown}/{EXCLUDED_MATCHES_FILE}"
        )
        if n_excluded:
            log.warning(message, stage=STAGE_NAME)
        else:
            log.info(message, stage=STAGE_NAME)
    else:
        log.info(
            "data_rules.exclude_failing_matches is off, so no match is excluded from model training and player "
            f"rating; any earlier {EXCLUDED_MATCHES_FILE} and {EXCLUSION_COUNTS_FILE} were deleted",
            stage=STAGE_NAME,
        )
    log.info(
        f"wrote {UNPARSED_RESULTS_FILE}, {PASS_RATES_FILE}, {FAILING_MATCHES_FILE}"
        + (f", {EXCLUDED_MATCHES_FILE}, {EXCLUSION_COUNTS_FILE}" if exclude_on else "")
        + f" and the score-reconstruction section of {audit.AUDIT_MD} to {shown}/",
        stage=STAGE_NAME,
    )
    log.info(
        f"saved {BALLS_TABLE} ({len(table):,} rows; {int(table['dismissed_fallback'].sum()):,} Counted_Wickets use "
        f"the striker fallback), {INNINGS_INFO_TABLE} ({len(per_innings):,} rows), {SCORE_CHECKS_TABLE} "
        f"({len(score_checks):,} rows) and {PASS_RATES_TABLE} ({len(pass_rates):,} rows)",
        stage=STAGE_NAME,
    )
