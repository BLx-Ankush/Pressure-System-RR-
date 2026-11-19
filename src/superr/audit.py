"""Auditor: profile the raw data and confirm the data facts the design depends on (Requirement 4).

Design section "Auditor (``audit.py``)" and the design-time data findings F-1 to F-11. The
``audit`` Pipeline_Stage (``run_stage``) runs after ``load`` and before ``reconstruct``
(criterion 15) and reads the loaded tables from the ``load`` checkpoint, never a Source_CSV.

What the stage does
-------------------
1. ``run_audit`` computes the tables of criteria 1-10 and the three Fact_Checks of criteria
   12-14 from the loaded ``ball_by_ball``, ``matches`` and ``mapping`` tables and the Config
   class mapping ``data_rules.extras_type_classes``.
2. ``write_audit`` writes each table as its own CSV file in ``outputs/audit/`` (criterion 11),
   replacing any earlier file of the same name, and then ``outputs/audit/audit.md``
   (criterion 16). Other files in that folder, such as the ``reconstruct`` stage's, are left
   alone. Every file is UTF-8 with ``\\n`` line ends, written through a temporary file renamed
   in place; floats in the CSV files are written as ``%.6f``.
3. The stage saves its checkpoint (below), then raises ``AuditError`` (a ``DataError``, exit
   code 3) if any Fact_Check failed. ``audit.md`` is already written at that point, and the
   Run stops before the Score_Reconstructor starts; the message names each failed Fact_Check
   with its observed value (criterion 17).

Tables (criterion 11; ``AUDIT_FILES`` maps each table key to its file)
---------------------------------------------------------------------
Count tables give missing values their own category, the text ``<missing>``
(``MISSING_LABEL``). Criteria 1, 3 and 7 always list it, with 0 rows when nothing is missing.
Other categories are sorted by value (numbers numerically, text by code point) unless noted.

* ``extras_type_x_extras1_runs.csv`` and ``extras_type_x_extras_2_runs.csv`` (criterion 1):
  cross-tabulations over all Deliveries. Column ``extras_type`` holds the row category, and
  every other column is one ``extras1_runs`` (or ``extras_2_runs``) value plus ``<missing>``.
  Every cell is given, zeros included, so the cells sum to the number of Deliveries.
* ``dismissal_codes.csv`` (criterion 2): one row per (``dismissal_type_id``,
  ``dismissal_type``) pair over Deliveries with a code: ``dismissal_type_id``, ``code_rows``,
  ``dismissal_type`` (``<missing>`` for a coded Delivery without a label) and ``label_rows``.
* ``stage_counts.csv`` (``stage``, ``rows``) and ``result_counts.csv`` (``result``, ``rows``)
  (criterion 3): matches per value, most frequent first, ``<missing>`` last.
* ``bowl_speed_summary.csv`` (criterion 4): one row with ``deliveries``, ``zero_rows``,
  ``zero_pct``, ``missing_rows``, ``missing_pct``, ``negative_rows``, ``positive_rows`` and
  ``positive_min``, ``positive_p01``, ``positive_median``, ``positive_p99``, ``positive_max``
  (linear-interpolation quantiles of the values above 0). Percentages are of all Deliveries.
  ``negative_rows`` is not required by criterion 4; it accounts for the values that are
  neither 0, missing nor above 0 (one ``-1.0`` as delivered).
* ``bowling_type_stability.csv`` (criterion 5): one row per ``bowler_id`` with
  ``deliveries``, ``n_bowling_types`` (distinct non-missing values), ``modal_bowling_type``,
  ``modal_rows``, ``tie_broken`` (two or more values share the top count) and
  ``bowling_types`` (each value with its row count, most frequent first).
* ``names_with_several_ids.csv`` (``player_name``, ``player_id``, ``n_player_ids``,
  ``mapping_rows``) and ``ids_with_several_names.csv`` (``player_id``, ``player_name``,
  ``n_player_names``, ``mapping_rows``) (criterion 6): one row per (name, ID) pair of each
  conflicting name or ID in the mapping file. Names are compared as exact strings; a mapping
  row with a missing ``player_name`` takes no part (none as delivered).
* ``revised_values.csv`` (criterion 7): ``column``, ``value``, ``rows`` for
  ``first_inn_revised_overs``, ``second_inn_revised_overs`` and ``revised_target``.
* ``over_sizes.csv`` (criterion 8): ``per_over``, ``overs_by_deliveries`` and
  ``overs_by_legal_balls``: for each count n, the number of Overs (``match_id``, ``innings``,
  ``overid``) of every innings, Super_Over_Innings included, that hold n Deliveries and that
  hold n Legal_Balls. Legal_Balls come from ``superr.reconstruct.classify`` with the Config
  class mapping. A Delivery with a missing Over key belongs to no Over (none as delivered).
* ``other_dismissal_codes.csv`` (criterion 9): each code other than 1-6 and 9 with
  ``dismissal_types`` (its labels joined by ``"; "``), ``rows``, ``reappearing_rows`` and
  ``reappearance_share``. Header only when there is no such code.
* ``dismissal_classification.csv`` (criterion 10): every code present, with
  ``dismissal_types``, ``rows``, ``reappearance_share`` (blank for codes 1-6 and 9, whose rule
  does not use it), ``classification`` (``Counted_Wicket`` or ``Retired_Hurt``), ``rule`` and
  ``reason``.

Reappearance_Share (criterion 9)
--------------------------------
For a dismissal code: the share of Deliveries with that code whose ``dismissed_player_id``
appears as ``striker_id`` or ``non_striker_id`` on a later Delivery (higher
``innings_ball_num``) of the same match and innings. It is computed without a per-row loop:
the largest ``innings_ball_num`` at which each player appears as striker or non-striker in each
(match, innings) is joined to the coded Deliveries, and a Delivery reappears when that value
exceeds its own ``innings_ball_num``. A coded Delivery with a missing ``dismissed_player_id``
does not reappear.

Dismissal-code classification (criterion 10)
--------------------------------------------
``classify_dismissal_codes`` tries these rules in order and records the first that decides:

* ``base_code``: codes 1-6 and 9 (``BASE_COUNTED_CODES``) are Counted_Wickets;
* ``label``: any other code is Retired_Hurt when one of its labels contains ``hurt`` or
  ``not out``, ignoring case;
* ``reappearance``: any other code is Retired_Hurt when its Reappearance_Share is above 0;
* ``default``: any remaining code is a Counted_Wicket.

Modal_Bowling_Type (criterion 5)
--------------------------------
A bowler's most frequent non-missing ``bowling_type``. When two or more values tie, the value
on the bowler's latest Delivery wins: latest ``match_date`` (from ``matches``), then highest
``innings``, then highest ``innings_ball_num``, with the higher ``match_id`` as a last,
deterministic tiebreak that the glossary does not need for real data. If the latest Delivery's
value is not one of the tied values (possible with three or more values), the tied value seen
most recently wins. A missing date sorts as the earliest. A bowler with no non-missing
``bowling_type`` has a missing Modal_Bowling_Type.

Fact_Checks (criteria 12-14; ``FactCheck``)
-------------------------------------------
* ``no_ball_extras_2_runs_13614``: at least one No_Ball (under the Config class mapping) in
  match 13614, innings 1, ``overid`` 18 has ``extras_2_runs`` = 4.
* ``revised_overs_50408``: ``first_inn_revised_overs`` or ``second_inn_revised_overs`` equals
  54 for match 50408.
* ``retire_out_label``: every Delivery with ``dismissal_type_id`` 9 is labelled
  ``Retire Out`` (exact text; true when there is no such Delivery).

If ``classify`` stops on the ``extras_type`` values (``DeliveryClassError``), the Auditor still
writes every other table and ``audit.md``, records the problem there, deletes any earlier
``over_sizes.csv``, counts ``no_ball_extras_2_runs_13614`` as failed (not evaluated) and then
raises ``AuditError`` with the classification report.

Checkpoint of the ``audit`` stage
---------------------------------
* table ``dismissal_classes`` (``CLASSES_TABLE``): the classification table above.
  ``counted_wicket_codes`` and ``retired_hurt_codes`` read its code lists.
* table ``modal_bowling_type`` (``MODAL_TABLE``): ``bowler_id`` (Int64), ``modal_bowling_type``
  (string), one row per bowler.
* table ``bowl_speed_summary`` (``BOWL_SPEED_TABLE``): the one-row ``bowl_speed`` summary.
* JSON ``audit_text`` (``TEXT_JSON``): ``{"text": <the audit.md text>}``. The ``reconstruct``
  stage rewrites ``audit.md`` as this text plus its score-reconstruction section.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import pandas as pd

from superr.errors import DataError
from superr.reconstruct import DeliveryClass, DeliveryClassError, classify

if TYPE_CHECKING:
    from superr.pipeline import StageContext

__all__ = [
    "AUDIT_FILES",
    "AUDIT_FOLDER",
    "AUDIT_MD",
    "BASE_COUNTED_CODES",
    "BOWL_SPEED_TABLE",
    "CLASSES_TABLE",
    "COUNTED_WICKET",
    "FACT_NO_BALL",
    "FACT_RETIRE_OUT",
    "FACT_REVISED_OVERS",
    "MISSING_LABEL",
    "MODAL_TABLE",
    "RETIRED_HURT",
    "RULE_BASE",
    "RULE_DEFAULT",
    "RULE_LABEL",
    "RULE_REAPPEARANCE",
    "STAGE_NAME",
    "TEXT_JSON",
    "AuditError",
    "AuditFile",
    "AuditResult",
    "FactCheck",
    "audit_markdown",
    "bowl_speed_summary",
    "bowling_type_stability",
    "classify_dismissal_codes",
    "counted_wicket_codes",
    "dismissal_code_table",
    "extras_crosstabs",
    "fact_checks",
    "name_id_conflicts",
    "other_dismissal_codes",
    "over_size_counts",
    "reappearance_share",
    "retired_hurt_codes",
    "revised_values",
    "run_audit",
    "run_stage",
    "stage_result_counts",
    "write_audit",
]

# ------------------------------------------------------------------------------- constants

#: Name of the Pipeline_Stage that runs the Auditor, and of its checkpoint.
STAGE_NAME = "audit"
_LOAD_STAGE = "load"

#: Folder under the outputs folder that holds the audit files, and the audit report's name.
AUDIT_FOLDER = "audit"
AUDIT_MD = "audit.md"

#: Category label of a missing value in every count table.
MISSING_LABEL = "<missing>"

#: Dismissal codes that are Counted_Wickets by definition (glossary; criterion 10).
BASE_COUNTED_CODES: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 9)
#: Codes that criterion 9 expected besides the base codes; audit.md notes when they are absent.
_EXPECTED_OTHER_CODES: tuple[int, ...] = (7, 8)
#: Words whose presence in a label (ignoring case) makes an other code Retired_Hurt.
_RETIREMENT_WORDS: tuple[str, ...] = ("hurt", "not out")

#: Classification values.
COUNTED_WICKET = "Counted_Wicket"
RETIRED_HURT = "Retired_Hurt"

#: Rules of classify_dismissal_codes, in the order they are tried.
RULE_BASE = "base_code"
RULE_LABEL = "label"
RULE_REAPPEARANCE = "reappearance"
RULE_DEFAULT = "default"

#: Checkpoint names of the audit stage.
CLASSES_TABLE = "dismissal_classes"
MODAL_TABLE = "modal_bowling_type"
BOWL_SPEED_TABLE = "bowl_speed_summary"
TEXT_JSON = "audit_text"

#: Fact_Check names (criteria 12-14).
FACT_NO_BALL = "no_ball_extras_2_runs_13614"
FACT_REVISED_OVERS = "revised_overs_50408"
FACT_RETIRE_OUT = "retire_out_label"

# The data facts checked by the Fact_Checks, as stated in Requirement 4, criteria 12-14.
_NO_BALL_MATCH, _NO_BALL_INNINGS, _NO_BALL_OVER, _NO_BALL_EXTRAS = 13614, 1, 18, 4
_REVISED_MATCH, _REVISED_BALLS = 50408, 54
_RETIRE_OUT_CODE, _RETIRE_OUT_LABEL = 9, "Retire Out"

_REVISED_COLUMNS: tuple[str, ...] = ("first_inn_revised_overs", "second_inn_revised_overs", "revised_target")
_OVER_KEYS: list[str] = ["match_id", "innings", "overid"]
_QUANTILES: tuple[tuple[str, float], ...] = (
    ("positive_min", 0.0),
    ("positive_p01", 0.01),
    ("positive_median", 0.5),
    ("positive_p99", 0.99),
    ("positive_max", 1.0),
)
_FLOAT_FORMAT = "%.6f"
_INT64_MIN = np.iinfo(np.int64).min
_TEXT = "string"  # dtype of every text column the Auditor builds, as the Data_Loader uses


class AuditFile(NamedTuple):
    """One CSV file of the audit (criterion 11)."""

    file_name: str
    criterion: int  # Requirement 4 criterion whose table the file holds
    contents: str  # one-line description, used in audit.md


#: Table key -> CSV file in ``outputs/audit/``, in criterion order.
AUDIT_FILES: dict[str, AuditFile] = {
    "extras1_crosstab": AuditFile(
        "extras_type_x_extras1_runs.csv", 1, "Deliveries in each `extras_type` × `extras1_runs` cell"
    ),
    "extras2_crosstab": AuditFile(
        "extras_type_x_extras_2_runs.csv", 1, "Deliveries in each `extras_type` × `extras_2_runs` cell"
    ),
    "dismissal_codes": AuditFile(
        "dismissal_codes.csv", 2, "Rows of each `dismissal_type_id` and of each `dismissal_type` label recorded with it"
    ),
    "stage_counts": AuditFile("stage_counts.csv", 3, "Matches with each `stage` value"),
    "result_counts": AuditFile("result_counts.csv", 3, "Matches with each distinct `result` string"),
    "bowl_speed": AuditFile(
        "bowl_speed_summary.csv", 4, "Shares of `bowl_speed` = 0 and missing; quantiles of the values above 0"
    ),
    "bowling_types": AuditFile(
        "bowling_type_stability.csv", 5, "Distinct `bowling_type` values and the Modal_Bowling_Type of each `bowler_id`"
    ),
    "names_with_several_ids": AuditFile(
        "names_with_several_ids.csv", 6, "Each `player_name` that maps to more than one `player_id`, with those IDs"
    ),
    "ids_with_several_names": AuditFile(
        "ids_with_several_names.csv", 6, "Each `player_id` that maps to more than one `player_name`, with those names"
    ),
    "revised_values": AuditFile(
        "revised_values.csv",
        7,
        "Matches with each value of `first_inn_revised_overs`, `second_inn_revised_overs` and `revised_target`",
    ),
    "over_sizes": AuditFile("over_sizes.csv", 8, "Overs with each count of Deliveries and of Legal_Balls"),
    "other_codes": AuditFile(
        "other_dismissal_codes.csv", 9, "Labels, rows and Reappearance_Share of each dismissal code other than 1-6 and 9"
    ),
    "classification": AuditFile(
        "dismissal_classification.csv", 10, "Classification of each dismissal code and the rule that decided it"
    ),
}


# ------------------------------------------------------------------------------- results

@dataclass(frozen=True)
class FactCheck:
    """One pass/fail data-fact check (Requirement 4, criteria 12-14)."""

    name: str
    criterion: int  # Requirement 4 criterion
    statement: str  # what must hold
    observed: str  # the observed value, as text
    passed: bool

    @property
    def result(self) -> str:
        return "pass" if self.passed else "fail"


@dataclass
class AuditResult:
    """Everything run_audit computes: the tables keyed as in AUDIT_FILES, the Fact_Checks and problems.

    A table is None when it could not be computed (only ``over_sizes``, when the ``extras_type``
    values cannot be classified); ``problems`` then says why.
    """

    tables: dict[str, pd.DataFrame | None]
    facts: list[FactCheck]
    problems: list[str]
    deliveries: int
    matches: int
    mapping_rows: int

    @property
    def classification(self) -> pd.DataFrame:
        """The dismissal-code classification (criterion 10), saved as ``dismissal_classes``."""
        table = self.tables["classification"]
        assert table is not None
        return table

    @property
    def modal_bowling_type(self) -> pd.DataFrame:
        """``bowler_id`` and ``modal_bowling_type`` of every bowler, saved as ``modal_bowling_type``."""
        table = self.tables["bowling_types"]
        assert table is not None
        return table[["bowler_id", "modal_bowling_type"]].reset_index(drop=True)

    @property
    def failed(self) -> list[FactCheck]:
        return [fact for fact in self.facts if not fact.passed]


class AuditError(DataError):
    """A Fact_Check failed, or the audit found a problem that stops the Run (exit code 3).

    Raised by the ``audit`` stage after ``audit.md`` is written. ``failed`` holds the failed
    Fact_Checks and ``problems`` any other problem text; the message names each failed
    Fact_Check with its observed value (Requirement 4, criterion 17).
    """

    def __init__(self, failed: Sequence[FactCheck], problems: Sequence[str], *, total: int, audit_md: str) -> None:
        self.failed: tuple[FactCheck, ...] = tuple(failed)
        self.problems: tuple[str, ...] = tuple(problems)
        lines = [
            f"{len(self.failed)} of {total} Fact_Checks failed, so the Run stops before the Score_Reconstructor "
            f"starts (Requirement 4, criterion 17); {audit_md} records every Fact_Check:"
        ]
        lines += [
            f"  - {fact.name} (Requirement 4, criterion {fact.criterion}: {fact.statement}): observed {fact.observed}"
            for fact in self.failed
        ]
        if self.problems:
            lines.append("The Auditor also found:")
            lines += ["  - " + problem.replace("\n", "\n    ") for problem in self.problems]
        super().__init__("\n".join(lines))


# ------------------------------------------------------------------------------- helpers

def _flags(values: pd.Series) -> np.ndarray:
    """A nullable boolean Series as a bool array, missing counted as False."""
    return values.to_numpy(dtype=bool, na_value=False)


def _equals(series: pd.Series, value: Any) -> np.ndarray:
    """True where ``series`` equals ``value`` exactly; a missing entry is False."""
    return _flags(series.eq(value))


def _key(series: pd.Series) -> np.ndarray:
    """A whole-number column as an int64 sort key; a missing value sorts first."""
    return series.to_numpy(dtype=np.int64, na_value=_INT64_MIN)


def _text(value: Any) -> str:
    return MISSING_LABEL if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)) else str(value)


def _rows(count: int) -> str:
    return f"{count:,} row" if count == 1 else f"{count:,} rows"


def _and_list(items: Sequence[Any]) -> str:
    texts = [str(item) for item in items]
    return texts[0] if len(texts) == 1 else ", ".join(texts[:-1]) + " and " + texts[-1]


def _categories(series: pd.Series) -> tuple[np.ndarray, list[str]]:
    """Category of each entry and the category labels: the distinct values sorted, then MISSING_LABEL.

    Numbers sort numerically and text by code point; every missing entry takes the last
    category, which is always present.
    """
    codes, uniques = pd.factorize(series, use_na_sentinel=True)
    values = list(uniques.tolist())
    order = sorted(range(len(values)), key=values.__getitem__)
    rank = np.empty(len(values), dtype=np.int64)
    rank[order] = np.arange(len(values), dtype=np.int64)
    categories = np.full(len(codes), len(values), dtype=np.int64)
    present = codes >= 0
    categories[present] = rank[codes[present]]
    return categories, [_text(values[index]) for index in order] + [MISSING_LABEL]


def _count_table(series: pd.Series, name: str, *, by_count: bool) -> pd.DataFrame:
    """Rows per value of ``series`` as columns ``name`` (text) and ``rows``; MISSING_LABEL always last.

    ``by_count``: most frequent first, ties by value; otherwise by value.
    """
    categories, labels = _categories(series)
    counts = np.bincount(categories, minlength=len(labels)).astype(np.int64)
    n_values = len(labels) - 1
    order = np.arange(n_values)
    if by_count:
        order = np.lexsort((order, -counts[:n_values]))
    order = np.append(order, n_values)
    return pd.DataFrame({name: pd.array([labels[i] for i in order], dtype=_TEXT), "rows": counts[order]})


def _crosstab(rows: pd.Series, columns: pd.Series, row_name: str) -> pd.DataFrame:
    """Every cell of rows × columns with its count, both with a MISSING_LABEL category."""
    row_codes, row_labels = _categories(rows)
    column_codes, column_labels = _categories(columns)
    width = len(column_labels)
    cells = np.bincount(row_codes * width + column_codes, minlength=len(row_labels) * width)
    table = pd.DataFrame(cells.reshape(len(row_labels), width).astype(np.int64), columns=column_labels)
    table.insert(0, row_name, pd.array(row_labels, dtype=_TEXT))
    return table


def _code_summary(code_table: pd.DataFrame) -> list[tuple[int, list[str], int]]:
    """(code, labels in table order, rows) of each code of a dismissal_code_table, by code."""
    summary = []
    for code, group in code_table.groupby("dismissal_type_id", sort=True):  # one group per code
        summary.append((int(code), [str(label) for label in group["dismissal_type"].tolist()], int(group["code_rows"].iloc[0])))
    return summary


def _reappearance_by_code(reapp: pd.DataFrame) -> dict[int, tuple[int, float]]:
    """code -> (reappearing rows, Reappearance_Share) from a reappearance_share table."""
    return {
        int(code): (int(reappearing), float(share))
        for code, reappearing, share in zip(
            reapp["dismissal_type_id"].tolist(), reapp["reappearing_rows"].tolist(), reapp["reappearance_share"].tolist()
        )
    }


# ------------------------------------------------------------------------------- tables

def extras_crosstabs(balls: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``extras_type`` against ``extras1_runs`` and against ``extras_2_runs`` over all Deliveries (criterion 1).

    Each table has the column ``extras_type`` (one row per value, then ``<missing>``) and one
    int64 column per value of the other column, then ``<missing>``; every cell is given.
    """
    extras_type = balls["extras_type"]
    return (
        _crosstab(extras_type, balls["extras1_runs"], "extras_type"),
        _crosstab(extras_type, balls["extras_2_runs"], "extras_type"),
    )


def dismissal_code_table(balls: pd.DataFrame) -> pd.DataFrame:
    """Each ``dismissal_type_id`` with its row count and each label recorded with it (criterion 2).

    Over the Deliveries with a non-missing ``dismissal_type_id``. One row per (code, label):
    ``dismissal_type_id`` (Int64), ``code_rows``, ``dismissal_type`` (text, ``<missing>`` for a
    coded Delivery without a label) and ``label_rows``, sorted by code, then label.
    """
    code = balls["dismissal_type_id"]
    coded = _flags(code.notna())
    if not coded.any():
        return pd.DataFrame(
            {
                "dismissal_type_id": pd.array([], dtype="Int64"),
                "code_rows": np.array([], dtype=np.int64),
                "dismissal_type": pd.array([], dtype=_TEXT),
                "label_rows": np.array([], dtype=np.int64),
            }
        )
    label_codes, labels = _categories(balls["dismissal_type"].loc[coded])
    frame = pd.DataFrame({"code": code.to_numpy(dtype=np.int64, na_value=0)[coded], "label": label_codes})
    pairs = frame.groupby(["code", "label"], sort=True).size()
    code_rows = frame.groupby("code", sort=True).size()
    pair_codes = pairs.index.get_level_values("code")
    return pd.DataFrame(
        {
            "dismissal_type_id": pd.array(pair_codes, dtype="Int64"),
            "code_rows": code_rows.reindex(pair_codes).to_numpy(dtype=np.int64),
            "dismissal_type": pd.array([labels[i] for i in pairs.index.get_level_values("label")], dtype=_TEXT),
            "label_rows": pairs.to_numpy(dtype=np.int64),
        }
    )


def stage_result_counts(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Matches per ``stage`` value and per distinct ``result`` string (criterion 3).

    Columns ``stage`` (or ``result``) and ``rows``, most frequent first, ``<missing>`` last.
    """
    return (
        _count_table(matches["stage"], "stage", by_count=True),
        _count_table(matches["result"], "result", by_count=True),
    )


def bowl_speed_summary(balls: pd.DataFrame) -> pd.DataFrame:
    """One-row ``bowl_speed`` summary (criterion 4); columns in the module docstring.

    Percentages are 100 × rows ÷ Deliveries. The quantiles are numpy's linear-interpolation
    quantiles of the values above 0; they and the percentages are NaN when undefined.
    """
    speed = balls["bowl_speed"].to_numpy(dtype=np.float64, na_value=np.nan)
    n = len(speed)
    missing = np.isnan(speed)
    positive = speed[~missing & (speed > 0)]
    zero_rows = int(np.count_nonzero(speed == 0))
    missing_rows = int(np.count_nonzero(missing))
    summary: dict[str, list[Any]] = {
        "deliveries": [n],
        "zero_rows": [zero_rows],
        "zero_pct": [100.0 * zero_rows / n if n else np.nan],
        "missing_rows": [missing_rows],
        "missing_pct": [100.0 * missing_rows / n if n else np.nan],
        "negative_rows": [int(np.count_nonzero(speed < 0))],
        "positive_rows": [len(positive)],
    }
    values = np.quantile(positive, [q for _, q in _QUANTILES]) if len(positive) else [np.nan] * len(_QUANTILES)
    for (name, _), value in zip(_QUANTILES, values):
        summary[name] = [float(value)]
    table = pd.DataFrame(summary)
    for column in ("deliveries", "zero_rows", "missing_rows", "negative_rows", "positive_rows"):
        table[column] = table[column].astype(np.int64)
    return table


def _date_keys(match_id: pd.Series, matches: pd.DataFrame) -> np.ndarray:
    """int64 sort key of each Delivery's ``match_date``; a missing date or match sorts first.

    A ``match_id`` listed more than once in ``matches`` takes its latest date.
    """
    known = _flags(matches["match_id"].notna())
    dates = matches.loc[known].groupby("match_id", sort=True)["match_date"].max()
    position = dates.index.get_indexer(match_id)
    values = dates.to_numpy(dtype="datetime64[us]").astype(np.int64)  # NaT is the int64 minimum
    keys = np.full(len(match_id), _INT64_MIN, dtype=np.int64)
    found = position >= 0
    keys[found] = values[position[found]]
    return keys


def bowling_type_stability(balls: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Distinct ``bowling_type`` values and the Modal_Bowling_Type of each ``bowler_id`` (criterion 5).

    ``balls`` needs ``match_id``, ``innings``, ``innings_ball_num``, ``bowler_id`` and
    ``bowling_type``; ``matches`` needs ``match_id`` and ``match_date``. The tie rule is in the
    module docstring. One row per bowler, by ``bowler_id``; columns ``bowler_id`` (Int64),
    ``deliveries``, ``n_bowling_types``, ``modal_bowling_type`` (string), ``modal_rows``,
    ``tie_broken`` and ``bowling_types``.
    """
    n = len(balls)
    # Position of each Delivery in the latest-Delivery order: larger means later.
    order = np.lexsort(
        (
            _key(balls["match_id"]),
            _key(balls["innings_ball_num"]),
            _key(balls["innings"]),
            _date_keys(balls["match_id"], matches),
        )
    )
    recency = np.empty(n, dtype=np.int64)
    recency[order] = np.arange(n, dtype=np.int64)
    work = pd.DataFrame(
        {"bowler_id": balls["bowler_id"].array, "bowling_type": balls["bowling_type"].array, "recency": recency}
    )
    work = work.loc[_flags(work["bowler_id"].notna())]
    deliveries = work.groupby("bowler_id", sort=True).size()
    typed = work.loc[_flags(work["bowling_type"].notna())]
    per_type = (
        typed.groupby(["bowler_id", "bowling_type"], sort=True)
        .agg(rows=("recency", "size"), latest=("recency", "max"))
        .reset_index()
    )
    by_mode = per_type.sort_values(["bowler_id", "rows", "latest"], ascending=[True, False, False], kind="stable")
    modal = by_mode.drop_duplicates("bowler_id", keep="first").set_index("bowler_id")
    top = per_type.groupby("bowler_id", sort=True)["rows"].transform("max")
    n_top = (per_type["rows"] == top).groupby(per_type["bowler_id"], sort=True).sum()
    n_types = per_type.groupby("bowler_id", sort=True).size()
    listing = per_type.sort_values(["bowler_id", "rows", "bowling_type"], ascending=[True, False, True], kind="stable")
    listed = (
        (listing["bowling_type"].astype(str) + " (" + listing["rows"].map("{:,}".format) + ")")
        .groupby(listing["bowler_id"], sort=True)
        .agg("; ".join)
    )
    index = deliveries.index
    return pd.DataFrame(
        {
            "bowler_id": pd.array(index, dtype="Int64"),
            "deliveries": deliveries.to_numpy(dtype=np.int64),
            "n_bowling_types": n_types.reindex(index, fill_value=0).to_numpy(dtype=np.int64),
            "modal_bowling_type": pd.array(modal["bowling_type"].reindex(index).tolist(), dtype=_TEXT),
            "modal_rows": modal["rows"].reindex(index, fill_value=0).to_numpy(dtype=np.int64),
            "tie_broken": (n_top.reindex(index, fill_value=0) > 1).to_numpy(dtype=bool),
            "bowling_types": pd.array(listed.reindex(index, fill_value="").tolist(), dtype=_TEXT),
        }
    )


def name_id_conflicts(mapping: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Names with several IDs and IDs with several names in the mapping file (criterion 6).

    Names are compared as exact strings. Returns (names table, IDs table), one row per (name,
    ID) pair of each conflicting name or ID, with the pair's ``mapping_rows``: columns
    ``player_name``, ``player_id``, ``n_player_ids``, ``mapping_rows`` sorted by name then ID,
    and ``player_id``, ``player_name``, ``n_player_names``, ``mapping_rows`` sorted by ID then name.
    """
    known = _flags(mapping["player_name"].notna() & mapping["player_id"].notna())
    pairs = (
        mapping.loc[known, ["player_name", "player_id"]]
        .groupby(["player_name", "player_id"], sort=True)
        .size()
        .rename("mapping_rows")
        .reset_index()
    )
    ids_per_name = pairs.groupby("player_name", sort=True)["player_id"].transform("size").to_numpy(dtype=np.int64)
    names_per_id = pairs.groupby("player_id", sort=True)["player_name"].transform("size").to_numpy(dtype=np.int64)
    by_name = pairs.loc[ids_per_name > 1, ["player_name", "player_id"]].copy()
    by_name["n_player_ids"] = ids_per_name[ids_per_name > 1]
    by_name["mapping_rows"] = pairs["mapping_rows"].to_numpy(dtype=np.int64)[ids_per_name > 1]
    by_id = pairs.loc[names_per_id > 1, ["player_id", "player_name"]].copy()
    by_id["n_player_names"] = names_per_id[names_per_id > 1]
    by_id["mapping_rows"] = pairs["mapping_rows"].to_numpy(dtype=np.int64)[names_per_id > 1]
    by_id = by_id.sort_values(["player_id", "player_name"], kind="stable")
    return by_name.reset_index(drop=True), by_id.reset_index(drop=True)


def revised_values(matches: pd.DataFrame) -> pd.DataFrame:
    """Every value of the revised-overs columns and ``revised_target`` with its row count (criterion 7).

    Columns ``column``, ``value`` (text, ``<missing>`` last) and ``rows``; values by number.
    """
    parts = []
    for column in _REVISED_COLUMNS:
        counts = _count_table(matches[column], "value", by_count=False)
        counts.insert(0, "column", pd.array([column] * len(counts), dtype=_TEXT))
        parts.append(counts)
    return pd.concat(parts, ignore_index=True)


def over_size_counts(balls: pd.DataFrame, legal: pd.Series | np.ndarray) -> pd.DataFrame:
    """Overs per count of Deliveries and per count of Legal_Balls (criterion 8).

    An Over is a (``match_id``, ``innings``, ``overid``) group in any innings; a Delivery with a
    missing key is in no Over. ``legal`` flags each Legal_Ball, aligned with ``balls``: normally
    ``classify(balls["extras_type"], <Config class mapping>) == DeliveryClass.LEGAL``. Returns
    ``per_over`` (every count seen in either measure, ascending), ``overs_by_deliveries`` and
    ``overs_by_legal_balls``; each count column sums to the number of Overs.
    """
    if isinstance(legal, pd.Series):
        if not legal.index.equals(balls.index):
            raise ValueError("legal must be a Series on the index of balls, or an array in its row order")
        flags = _flags(legal)
    else:
        flags = np.asarray(legal, dtype=bool)
        if flags.shape != (len(balls),):
            raise ValueError(f"legal must hold one flag per Delivery ({len(balls):,}); got shape {flags.shape}")
    frame = pd.DataFrame({key: balls[key].array for key in _OVER_KEYS})
    frame["legal"] = flags
    groups = frame.groupby(_OVER_KEYS, sort=False)  # dropna: a row with a missing key is in no Over
    deliveries = groups.size().to_numpy(dtype=np.int64)
    legal_balls = groups["legal"].sum().to_numpy(dtype=np.int64)
    sizes = np.union1d(deliveries, legal_balls).astype(np.int64)
    width = int(sizes.max()) + 1 if len(sizes) else 0
    return pd.DataFrame(
        {
            "per_over": sizes,
            "overs_by_deliveries": np.bincount(deliveries, minlength=width)[sizes].astype(np.int64),
            "overs_by_legal_balls": np.bincount(legal_balls, minlength=width)[sizes].astype(np.int64),
        }
    )


def reappearance_share(balls: pd.DataFrame) -> pd.DataFrame:
    """Reappearance_Share of every dismissal code present (criterion 9), without a per-row loop.

    ``balls`` needs ``match_id``, ``innings``, ``innings_ball_num``, ``striker_id``,
    ``non_striker_id``, ``dismissal_type_id`` and ``dismissed_player_id``. Returns
    ``dismissal_type_id`` (Int64), ``rows``, ``reappearing_rows`` and ``reappearance_share``
    (reappearing rows ÷ rows), by code. The method is in the module docstring.
    """
    keys = ["match_id", "innings", "player"]
    appearances = pd.concat(
        [
            pd.DataFrame(
                {
                    "match_id": balls["match_id"].array,
                    "innings": balls["innings"].array,
                    "player": balls[role].array,
                    "ball": balls["innings_ball_num"].array,
                }
            )
            for role in ("striker_id", "non_striker_id")
        ],
        ignore_index=True,
    ).dropna()
    last_seen = appearances.groupby(keys, sort=False)["ball"].max().rename("last_seen").reset_index()
    coded = _flags(balls["dismissal_type_id"].notna())
    wickets = pd.DataFrame(
        {
            "code": balls["dismissal_type_id"].array,
            "match_id": balls["match_id"].array,
            "innings": balls["innings"].array,
            "player": balls["dismissed_player_id"].array,
            "ball": balls["innings_ball_num"].array,
        }
    ).loc[coded]
    found = wickets.merge(last_seen, on=keys, how="left", sort=False, validate="many_to_one")
    reappears = _flags(found["last_seen"] > found["ball"])
    per_code = (
        pd.DataFrame({"code": found["code"].to_numpy(dtype=np.int64), "reappears": reappears})
        .groupby("code", sort=True)["reappears"]
        .agg(["size", "sum"])
    )
    rows = per_code["size"].to_numpy(dtype=np.int64)
    reappearing = per_code["sum"].to_numpy(dtype=np.int64)
    return pd.DataFrame(
        {
            "dismissal_type_id": pd.array(per_code.index, dtype="Int64"),
            "rows": rows,
            "reappearing_rows": reappearing,
            "reappearance_share": reappearing / np.maximum(rows, 1),
        }
    )


def other_dismissal_codes(code_table: pd.DataFrame, reapp: pd.DataFrame) -> pd.DataFrame:
    """The criterion 9 table: each code other than 1-6 and 9 with its labels, rows and Reappearance_Share.

    ``code_table`` is a dismissal_code_table and ``reapp`` the reappearance_share table of the
    same Deliveries. Columns ``dismissal_type_id`` (Int64), ``dismissal_types`` (labels joined by
    ``"; "``), ``rows``, ``reappearing_rows`` and ``reappearance_share``, by code.
    """
    shares = _reappearance_by_code(reapp)
    others = [(code, labels, rows) for code, labels, rows in _code_summary(code_table) if code not in BASE_COUNTED_CODES]
    absent = [code for code, _, _ in others if code not in shares]
    if absent:
        raise ValueError(f"reapp has no Reappearance_Share for dismissal code(s) {absent}; pass the tables of the same Deliveries")
    return pd.DataFrame(
        {
            "dismissal_type_id": pd.array([code for code, _, _ in others], dtype="Int64"),
            "dismissal_types": pd.array(["; ".join(labels) for _, labels, _ in others], dtype=_TEXT),
            "rows": np.array([rows for _, _, rows in others], dtype=np.int64),
            "reappearing_rows": np.array([shares[code][0] for code, _, _ in others], dtype=np.int64),
            "reappearance_share": np.array([shares[code][1] for code, _, _ in others], dtype=np.float64),
        }
    )


def classify_dismissal_codes(code_table: pd.DataFrame, reapp: pd.DataFrame) -> pd.DataFrame:
    """Classify every dismissal code present and record the rule that decided it (criterion 10).

    ``code_table`` is a dismissal_code_table and ``reapp`` the reappearance_share table of the
    same Deliveries. The rules, in order, are in the module docstring. One row per code, by
    code: ``dismissal_type_id`` (Int64), ``dismissal_types``, ``rows``, ``reappearance_share``
    (Float64, missing for codes 1-6 and 9), ``classification``, ``rule`` and ``reason``.
    """
    shares = _reappearance_by_code(reapp)
    records: list[tuple[int, str, int, Any, str, str, str]] = []
    for code, labels, rows in _code_summary(code_table):  # one pass per code, not per Delivery
        share: Any = pd.NA
        if code in BASE_COUNTED_CODES:
            decided = (COUNTED_WICKET, RULE_BASE, "codes 1-6 and 9 are Counted_Wickets")
        else:
            if code not in shares:
                raise ValueError(f"reapp has no Reappearance_Share for dismissal code {code}; pass the tables of the same Deliveries")
            reappearing, share = shares[code]
            hit = next(
                (
                    (label, word)
                    for label in labels
                    if label != MISSING_LABEL
                    for word in _RETIREMENT_WORDS
                    if word in label.casefold()
                ),
                None,
            )
            if hit is not None:
                decided = (RETIRED_HURT, RULE_LABEL, f'the label "{hit[0]}" contains "{hit[1]}"')
            elif share > 0:
                decided = (
                    RETIRED_HURT,
                    RULE_REAPPEARANCE,
                    f"the Reappearance_Share is {share:.6f} ({reappearing:,} of {_rows(rows)}), above 0",
                )
            else:
                decided = (
                    COUNTED_WICKET,
                    RULE_DEFAULT,
                    f'no label contains "hurt" or "not out", and the Reappearance_Share is 0 (0 of {_rows(rows)})',
                )
        records.append((code, "; ".join(labels), rows, share, *decided))
    return pd.DataFrame(
        {
            "dismissal_type_id": pd.array([record[0] for record in records], dtype="Int64"),
            "dismissal_types": pd.array([record[1] for record in records], dtype=_TEXT),
            "rows": np.array([record[2] for record in records], dtype=np.int64),
            "reappearance_share": pd.array([record[3] for record in records], dtype="Float64"),
            "classification": pd.array([record[4] for record in records], dtype=_TEXT),
            "rule": pd.array([record[5] for record in records], dtype=_TEXT),
            "reason": pd.array([record[6] for record in records], dtype=_TEXT),
        }
    )


def _codes_of(classes: pd.DataFrame, classification: str) -> list[int]:
    chosen = _equals(classes["classification"], classification)
    return sorted(int(code) for code in classes.loc[chosen, "dismissal_type_id"].tolist())


def counted_wicket_codes(classes: pd.DataFrame) -> list[int]:
    """The Counted_Wicket codes, ascending, of a classify_dismissal_codes table (checkpoint ``dismissal_classes``)."""
    return _codes_of(classes, COUNTED_WICKET)


def retired_hurt_codes(classes: pd.DataFrame) -> list[int]:
    """The Retired_Hurt codes, ascending, of a classify_dismissal_codes table (checkpoint ``dismissal_classes``)."""
    return _codes_of(classes, RETIRED_HURT)


# ---------------------------------------------------------------------------- Fact_Checks

def _no_ball_fact(balls: pd.DataFrame, cls: pd.Series | None) -> FactCheck:
    statement = (
        f"at least one No_Ball Delivery in match {_NO_BALL_MATCH}, innings {_NO_BALL_INNINGS}, `overid` "
        f"{_NO_BALL_OVER} has `extras_2_runs` = {_NO_BALL_EXTRAS}"
    )
    if cls is None:
        observed = "not evaluated, because the `extras_type` values cannot be classified (see Problems)"
        return FactCheck(FACT_NO_BALL, 12, statement, observed, False)
    in_over = (
        _equals(balls["match_id"], _NO_BALL_MATCH)
        & _equals(balls["innings"], _NO_BALL_INNINGS)
        & _equals(balls["overid"], _NO_BALL_OVER)
    )
    no_ball = in_over & (cls.to_numpy() == DeliveryClass.NO_BALL)
    extras = balls["extras_2_runs"].loc[no_ball]
    passed = bool(_equals(extras, _NO_BALL_EXTRAS).any())
    n_over, n_no_ball = int(in_over.sum()), int(no_ball.sum())
    if n_over == 0:
        observed = f"match {_NO_BALL_MATCH}, innings {_NO_BALL_INNINGS}, `overid` {_NO_BALL_OVER} has no Delivery"
    elif n_no_ball == 0:
        observed = f"none of the {n_over:,} Deliveries of that Over is a No_Ball"
    else:
        shown = ", ".join(
            f"{_text(runs)} (`extras_type` {_text(kind)})"
            for runs, kind in zip(extras.tolist(), balls["extras_type"].loc[no_ball].tolist())
        )
        noun = "No_Ball" if n_no_ball == 1 else "No_Balls"
        observed = f"`extras_2_runs` of the {n_no_ball:,} {noun} among the Over's {n_over:,} Deliveries: {shown}"
    return FactCheck(FACT_NO_BALL, 12, statement, observed, passed)


def _revised_fact(matches: pd.DataFrame) -> FactCheck:
    statement = (
        f"`first_inn_revised_overs` or `second_inn_revised_overs` equals {_REVISED_BALLS} for match {_REVISED_MATCH}"
    )
    rows = matches.loc[_equals(matches["match_id"], _REVISED_MATCH)]
    if rows.empty:
        return FactCheck(FACT_REVISED_OVERS, 13, statement, f"match {_REVISED_MATCH} is not in `matches.csv`", False)
    first, second = rows["first_inn_revised_overs"], rows["second_inn_revised_overs"]
    passed = bool((_equals(first, _REVISED_BALLS) | _equals(second, _REVISED_BALLS)).any())
    observed = "; ".join(
        f"`first_inn_revised_overs` = {_text(a)}, `second_inn_revised_overs` = {_text(b)}"
        for a, b in zip(first.tolist(), second.tolist())
    )
    return FactCheck(FACT_REVISED_OVERS, 13, statement, observed, passed)


def _retire_out_fact(balls: pd.DataFrame) -> FactCheck:
    statement = f'every Delivery with `dismissal_type_id` {_RETIRE_OUT_CODE} has the `dismissal_type` label "{_RETIRE_OUT_LABEL}"'
    labels = balls["dismissal_type"].loc[_equals(balls["dismissal_type_id"], _RETIRE_OUT_CODE)]
    passed = bool(_equals(labels, _RETIRE_OUT_LABEL).all())  # true when no Delivery has the code
    if labels.empty:
        observed = f"no Delivery has `dismissal_type_id` {_RETIRE_OUT_CODE}"
    else:
        counts = _count_table(labels, "label", by_count=True)
        counts = counts.loc[counts["rows"] > 0]
        shown = ", ".join(
            f"{label} ({rows:,})" if label == MISSING_LABEL else f'"{label}" ({rows:,})'
            for label, rows in zip(counts["label"].tolist(), counts["rows"].tolist())
        )
        observed = f"{len(labels):,} Deliveries, labelled {shown}"
    return FactCheck(FACT_RETIRE_OUT, 14, statement, observed, passed)


def fact_checks(balls: pd.DataFrame, matches: pd.DataFrame, cls: pd.Series | None) -> list[FactCheck]:
    """The Fact_Checks of criteria 12-14, in that order (see the module docstring).

    ``cls`` is the classify output for ``balls`` (DeliveryClass codes on its index), which
    decides the No_Balls of the first check; None counts that check as failed (not evaluated).
    """
    return [_no_ball_fact(balls, cls), _revised_fact(matches), _retire_out_fact(balls)]


# ------------------------------------------------------------------------------- the audit

def run_audit(
    balls: pd.DataFrame,
    matches: pd.DataFrame,
    mapping: pd.DataFrame,
    classes: dict[str, Sequence[str]],
) -> AuditResult:
    """Compute every table of criteria 1-10 and the Fact_Checks of criteria 12-14.

    ``balls``, ``matches`` and ``mapping`` are the loaded ``ball_by_ball``, ``matches`` and
    ``mapping`` tables; ``classes`` is the Config value ``data_rules.extras_type_classes``. A
    ``DeliveryClassError`` from ``classify`` is recorded in ``problems`` instead of raised (see
    the module docstring), so that every other table can still be written.
    """
    problems: list[str] = []
    cls: pd.Series | None
    try:
        cls = classify(balls["extras_type"], classes)
    except DeliveryClassError as exc:
        cls = None
        problems.append(
            "the Config class mapping data_rules.extras_type_classes cannot classify every Delivery, so "
            f"{AUDIT_FILES['over_sizes'].file_name} is not written and Fact_Check {FACT_NO_BALL} is not "
            f"evaluated:\n{exc}"
        )
    extras1, extras2 = extras_crosstabs(balls)
    codes = dismissal_code_table(balls)
    stages, results = stage_result_counts(matches)
    names, ids = name_id_conflicts(mapping)
    reapp = reappearance_share(balls)
    tables: dict[str, pd.DataFrame | None] = {
        "extras1_crosstab": extras1,
        "extras2_crosstab": extras2,
        "dismissal_codes": codes,
        "stage_counts": stages,
        "result_counts": results,
        "bowl_speed": bowl_speed_summary(balls),
        "bowling_types": bowling_type_stability(balls, matches),
        "names_with_several_ids": names,
        "ids_with_several_names": ids,
        "revised_values": revised_values(matches),
        "over_sizes": None if cls is None else over_size_counts(balls, cls.eq(DeliveryClass.LEGAL)),
        "other_codes": other_dismissal_codes(codes, reapp),
        "classification": classify_dismissal_codes(codes, reapp),
    }
    return AuditResult(
        tables=tables,
        facts=fact_checks(balls, matches, cls),
        problems=problems,
        deliveries=len(balls),
        matches=len(matches),
        mapping_rows=len(mapping),
    )


# ------------------------------------------------------------------------------- audit.md

def _cell(value: Any) -> str:
    text = str(value).replace("\n", " ").replace("|", "\\|")
    return text.replace(MISSING_LABEL, f"`{MISSING_LABEL}`")


def _md_table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(_cell(name) for name in header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(_cell(value) for value in row) + " |" for row in rows]
    return lines


def _number(value: Any, digits: int = 3) -> str:
    if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:,.{digits}f}"


def _code_list(codes: Sequence[int]) -> str:
    return ", ".join(str(code) for code in codes) if codes else "none"


def _counts_line(table: pd.DataFrame, name: str, *, limit: int | None = None) -> str:
    pairs = list(zip(table[name].tolist(), table["rows"].tolist()))
    shown = pairs if limit is None else [pair for pair in pairs if pair[0] != MISSING_LABEL][:limit]
    return ", ".join(f"{_cell(label)} ({rows:,})" for label, rows in shown)


def _facts_section(result: AuditResult) -> list[str]:
    lines = ["## Fact_Checks (criteria 12-14)", ""]
    lines += _md_table(
        ("Fact_Check", "Requirement 4", "Check", "Observed value", "Result"),
        [(f"`{f.name}`", f"criterion {f.criterion}", f.statement, f.observed, f.result) for f in result.facts],
    )
    failed = result.failed
    if failed:
        names = ", ".join(f"`{fact.name}`" for fact in failed)
        lines += [
            "",
            f"{len(failed)} of {len(result.facts)} Fact_Checks fail ({names}), so the Run stops here, before the "
            f"Score_Reconstructor starts (criterion 17).",
        ]
    else:
        lines += ["", f"All {len(result.facts)} Fact_Checks pass."]
    return lines + [""]


def _problems_section(problems: Sequence[str]) -> list[str]:
    lines = ["## Problems", ""]
    for problem in problems:
        lines += ["```text", problem, "```", ""]
    return lines


def _classification_section(classes: pd.DataFrame) -> list[str]:
    lines = [
        "## Dismissal-code classification (criterion 10)",
        "",
        f"- Counted_Wicket codes: {_code_list(counted_wicket_codes(classes))}",
        f"- Retired_Hurt codes: {_code_list(retired_hurt_codes(classes))}",
        "",
        "Rules, tried in order: codes 1-6 and 9 are Counted_Wickets (`base_code`); any other code is Retired_Hurt "
        'if one of its labels contains "hurt" or "not out", ignoring case (`label`), or if its Reappearance_Share '
        "is above 0 (`reappearance`); any remaining code is a Counted_Wicket (`default`).",
        "",
    ]
    lines += _md_table(
        ("Code", "Labels", "Rows", "Classification", "Rule", "Reason"),
        [
            (code, labels, f"{rows:,}", classification, f"`{rule}`", reason)
            for code, labels, rows, classification, rule, reason in zip(
                classes["dismissal_type_id"].tolist(),
                classes["dismissal_types"].tolist(),
                classes["rows"].tolist(),
                classes["classification"].tolist(),
                classes["rule"].tolist(),
                classes["reason"].tolist(),
            )
        ],
    )
    return lines + [""]


def _files_section(tables: dict[str, pd.DataFrame | None]) -> list[str]:
    lines = ["## Tables saved in `outputs/audit/` (criterion 11)", "", "Each Run rewrites these files.", ""]
    lines += _md_table(
        ("File", "Requirement 4", "Contents"),
        [
            (f"`{info.file_name}`", f"criterion {info.criterion}", info.contents)
            for key, info in AUDIT_FILES.items()
            if tables.get(key) is not None
        ],
    )
    skipped = [f"`{info.file_name}`" for key, info in AUDIT_FILES.items() if tables.get(key) is None]
    if skipped:
        lines += ["", f"Not written in this Run (see Problems): {', '.join(skipped)}."]
    return lines + [""]


def _crosstab_lines(table: pd.DataFrame, column_name: str) -> list[str]:
    header = [f"`extras_type` \\ `{column_name}`", *table.columns[1:]]
    rows = [[label, *(f"{count:,}" for count in counts)] for label, *counts in table.itertuples(index=False)]
    return _md_table(header, rows)


def _data_section(result: AuditResult) -> list[str]:
    tables = result.tables
    lines = ["## Data facts", ""]

    lines += ["### `extras_type` against `extras1_runs` and `extras_2_runs` (criterion 1)", ""]
    for key, column in (("extras1_crosstab", "extras1_runs"), ("extras2_crosstab", "extras_2_runs")):
        lines += _crosstab_lines(tables[key], column) + [""]

    codes = tables["dismissal_codes"]
    lines += ["### Dismissal codes and labels (criterion 2)", ""]
    lines += _md_table(
        ("Code", "Code rows", "Label", "Label rows"),
        [
            (code, f"{code_rows:,}", label, f"{label_rows:,}")
            for code, code_rows, label, label_rows in codes.itertuples(index=False)
        ],
    ) + [""]

    others = tables["other_codes"]
    lines += ["### Dismissal codes other than 1-6 and 9 (criterion 9)", ""]
    if others.empty:
        lines.append("No dismissal code other than 1-6 and 9 occurs.")
    else:
        lines += _md_table(
            ("Code", "Labels", "Rows", "Reappearing rows", "Reappearance_Share"),
            [
                (code, labels, f"{rows:,}", f"{reappearing:,}", f"{share:.6f}")
                for code, labels, rows, reappearing, share in others.itertuples(index=False)
            ],
        )
    present = {int(code) for code in codes["dismissal_type_id"].tolist()}
    absent = [code for code in _EXPECTED_OTHER_CODES if code not in present]
    if absent:
        noun, verb = ("Code", "does") if len(absent) == 1 else ("Codes", "do")
        lines += ["", f"{noun} {_and_list(absent)}, which criterion 9 expected, {verb} not occur."]
    lines.append("")

    stages, results = tables["stage_counts"], tables["result_counts"]
    n_missing = int(results.loc[results["result"] == MISSING_LABEL, "rows"].sum())
    lines += [
        "### Match stages and results (criterion 3)",
        "",
        f"- `stage`: {_counts_line(stages, 'stage')}.",
        f"- `result`: {len(results) - 1:,} distinct strings and {n_missing:,} missing values; the most frequent are "
        f"{_counts_line(results, 'result', limit=5)}.",
        "",
    ]

    speed = tables["bowl_speed"].iloc[0]
    lines += ["### `bowl_speed` (criterion 4)", ""]
    lines += _md_table(
        ("Statistic", "Value"),
        [
            ("Deliveries", f"{int(speed['deliveries']):,}"),
            ("`bowl_speed` = 0", f"{int(speed['zero_rows']):,} ({_number(speed['zero_pct'])}%)"),
            ("`bowl_speed` missing", f"{int(speed['missing_rows']):,} ({_number(speed['missing_pct'])}%)"),
            ("`bowl_speed` below 0", f"{int(speed['negative_rows']):,}"),
            ("`bowl_speed` above 0", f"{int(speed['positive_rows']):,}"),
            *(
                (f"{label} of the values above 0", _number(speed[name]))
                for label, name in (
                    ("Minimum", "positive_min"),
                    ("1st percentile", "positive_p01"),
                    ("Median", "positive_median"),
                    ("99th percentile", "positive_p99"),
                    ("Maximum", "positive_max"),
                )
            ),
        ],
    ) + [""]

    bowling = tables["bowling_types"]
    spread = bowling["n_bowling_types"].value_counts().sort_index()
    several = int((bowling["n_bowling_types"] > 1).sum())
    lines += [
        "### Bowling types (criterion 5)",
        "",
        f"{len(bowling):,} bowlers; by number of distinct non-missing `bowling_type` values: "
        + ", ".join(f"{int(n)}: {int(count):,}" for n, count in spread.items())
        + f". {several:,} bowlers have more than one value. The latest-Delivery tie rule decided "
        f"{int(bowling['tie_broken'].sum()):,} Modal_Bowling_Types.",
        "",
    ]

    by_name, by_id = tables["names_with_several_ids"], tables["ids_with_several_names"]
    lines += ["### Player names and IDs in `competition_team_player_mapping.csv` (criterion 6)", ""]
    for table, key, other, what in (
        (by_name, "player_name", "player_id", "`player_name` values map to more than one `player_id`"),
        (by_id, "player_id", "player_name", "`player_id` values map to more than one `player_name`"),
    ):
        groups = table.groupby(key, sort=False)[[other, "mapping_rows"]]
        lines.append(f"{len(groups):,} {what}{':' if len(groups) else '.'}")
        for value, group in groups:
            pairs = ", ".join(f"{_cell(item)} ({_rows(rows)})" for item, rows in group.itertuples(index=False))
            lines.append(f"- {_cell(value)}: {pairs}")
        lines.append("")

    revised = tables["revised_values"]
    lines += ["### Revised overs and targets in `matches.csv` (criterion 7)", ""]
    for column in _REVISED_COLUMNS:
        part = revised.loc[revised["column"] == column]
        lines.append(f"- `{column}`: {_counts_line(part, 'value')}")
    lines.append("")

    overs = tables["over_sizes"]
    lines += ["### Deliveries and Legal_Balls per Over (criterion 8)", ""]
    if overs is None:
        lines.append("Not computed in this Run (see Problems).")
    else:
        lines += [
            f"{int(overs['overs_by_deliveries'].sum()):,} Overs in all innings, Super_Over_Innings included; "
            "Legal_Balls follow the Config class mapping `data_rules.extras_type_classes`.",
            "",
        ]
        lines += _md_table(
            ("Per Over", "Overs with this many Deliveries", "Overs with this many Legal_Balls"),
            [(n, f"{by_d:,}", f"{by_l:,}") for n, by_d, by_l in overs.itertuples(index=False)],
        )
    return lines + [""]


def audit_markdown(result: AuditResult) -> str:
    """The text of ``audit.md`` (criterion 16), ending with one newline.

    Sections: Fact_Checks with observed values and results; Problems (only when there are
    any); the Counted_Wicket and Retired_Hurt codes with the deciding rule of each code; the
    CSV files saved; and a summary of the tables of criteria 1-9.
    """
    lines = [
        "# Data audit",
        "",
        f"The Auditor (the `{STAGE_NAME}` Pipeline_Stage, Requirement 4) profiled the Source_CSVs as loaded by "
        f"the `load` stage: {result.deliveries:,} Deliveries in `ball_by_ball.csv`, {result.matches:,} matches "
        f"in `matches.csv` and {result.mapping_rows:,} rows in `competition_team_player_mapping.csv`. Count "
        f"tables give missing values their own category, shown as `{MISSING_LABEL}`.",
        "",
    ]
    lines += _facts_section(result)
    if result.problems:
        lines += _problems_section(result.problems)
    lines += _classification_section(result.classification)
    lines += _files_section(result.tables)
    lines += _data_section(result)
    return "\n".join(lines).rstrip("\n") + "\n"


# ------------------------------------------------------------------------------- writing

def _replace(path: Path, write: Callable[[Path], object]) -> None:
    """Write through ``<file>.tmp`` renamed onto ``path``, so no partial file is ever left."""
    temporary = path.with_name(path.name + ".tmp")
    try:
        write(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_audit(outdir: Path | str, result: AuditResult) -> str:
    """Write each table to its CSV file in ``outdir`` (criterion 11), then ``audit.md`` (criterion 16).

    ``outdir`` is the audit folder (``outputs/audit/`` in a Run) and is created when absent.
    Each file replaces any earlier file of the same name; the file of a table that could not
    be computed is deleted, so no stale copy is left. Returns the ``audit.md`` text.
    """
    folder = Path(outdir)
    folder.mkdir(parents=True, exist_ok=True)
    for key, info in AUDIT_FILES.items():
        path = folder / info.file_name
        table = result.tables.get(key)
        if table is None:
            path.unlink(missing_ok=True)
            continue
        _replace(
            path,
            lambda temporary, table=table: table.to_csv(
                temporary, index=False, encoding="utf-8", lineterminator="\n", float_format=_FLOAT_FORMAT
            ),
        )
    text = audit_markdown(result)
    _replace(folder / AUDIT_MD, lambda temporary: temporary.write_bytes(text.encode("utf-8")))
    return text


def _shown_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def run_stage(ctx: StageContext) -> None:
    """The ``audit`` Pipeline_Stage: audit the loaded tables, write ``outputs/audit/``, save the checkpoint.

    Raises AuditError (exit code 3) after ``audit.md`` and the checkpoint files are written
    when a Fact_Check failed or the ``extras_type`` values cannot be classified.
    """
    store, log = ctx.store, ctx.log
    result = run_audit(
        store.load_table(_LOAD_STAGE, "ball_by_ball"),
        store.load_table(_LOAD_STAGE, "matches"),
        store.load_table(_LOAD_STAGE, "mapping"),
        ctx.settings.get("data_rules.extras_type_classes"),
    )
    outdir = ctx.settings.path("outputs") / AUDIT_FOLDER
    text = write_audit(outdir, result)
    store.save_table(STAGE_NAME, CLASSES_TABLE, result.classification)
    store.save_table(STAGE_NAME, MODAL_TABLE, result.modal_bowling_type)
    bowl_speed = result.tables["bowl_speed"]
    assert bowl_speed is not None
    store.save_table(STAGE_NAME, BOWL_SPEED_TABLE, bowl_speed)
    store.save_json(STAGE_NAME, TEXT_JSON, {"text": text})

    shown = _shown_path(outdir, ctx.root)
    written = sum(1 for key in AUDIT_FILES if result.tables.get(key) is not None)
    log.info(f"wrote {written} audit tables and {AUDIT_MD} to {shown}/", stage=STAGE_NAME)
    classes = result.classification
    log.info(
        f"dismissal codes: Counted_Wicket {_code_list(counted_wicket_codes(classes))}; "
        f"Retired_Hurt {_code_list(retired_hurt_codes(classes))}",
        stage=STAGE_NAME,
    )
    for fact in result.facts:
        log.info(f"Fact_Check {fact.name}: {fact.result} (observed {fact.observed})", stage=STAGE_NAME)
    if result.failed or result.problems:
        raise AuditError(result.failed, result.problems, total=len(result.facts), audit_md=f"{shown}/{AUDIT_MD}")
