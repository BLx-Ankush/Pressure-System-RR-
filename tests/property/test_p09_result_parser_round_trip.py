# Feature: pressure-rating-system, Property 9: For any result string generated from the result grammar (team name, optional "won"/"by", any letter case, repeated spaces, an integer margin with zero or more spaces before "run(s)", "wicket(s)" or "wkt(s)", optional "(DLS method)" or "D/L", and the super-over, tie, abandoned and no-result phrasings), the parser returns the generated result type by the first matching rule of Requirement 5 criterion 13, the generated margin for runs and wickets wins, and the DLS flag exactly when "DLS" or "D/L" is present. For any string containing none of the rule keywords, the parser returns "unparsed" with no margin.
"""Property 9: Result parser round trip (Requirement 5, criteria 13-15).

**Validates: Requirements 5.13, 5.14, 5.15**

Each example is a small ``matches`` table: 1-8 matches with distinct ``match_id`` values, in
drawn order (so not sorted in general), each with a ``result`` string from ``result_strings()``
and the ParsedResult that the string's grammar derivation gives. The test checks that:

* ``parse_result`` returns exactly that ParsedResult for every string: the result type of the
  first matching rule of criterion 13, the margin of the two win types (criterion 14), the DLS
  flag exactly when "DLS" or "D/L" is present, and UNPARSED with no margin for a string that
  matches no rule (criterion 15);
* the margin is set exactly when the type is WIN_BY_RUNS or WIN_BY_WICKETS, both in
  parse_result and in the parse_results table;
* ``parse_results`` returns one row per match, sorted by ``match_id``, with a nullable Int32
  ``margin`` column, and each row equals parse_result of that match's string;
* ``write_unparsed`` writes ``unparsed_results.csv`` with the header ``match_id,result`` and
  exactly the UNPARSED matches, sorted by ``match_id``, each with its original string. The audit
  folder is a fresh folder from ``tmp_path_factory`` for each example, never ``outputs/``.

The explicit example holds the worked examples of criteria 13 and 14 ("India Won Super Over By
10 Runs", "won by 17runs (DLS Method)", "Won 3 Wkts", "won by 44 run") and an unparsed string
with a comma, so the CSV quoting of an original string is exercised too.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from superr.reconstruct import (
    RESULT_TYPES,
    SUPER_OVER_WIN,
    UNPARSED,
    UNPARSED_RESULTS_FILE,
    WIN_BY_RUNS,
    WIN_BY_WICKETS,
    ParsedResult,
    parse_result,
    parse_results,
    write_unparsed,
)
from tests.strategies import MATCH_COLUMNS, ResultCase, result_strings

#: The two result types that carry a margin (criteria 13 and 14).
MARGIN_TYPES = (WIN_BY_RUNS, WIN_BY_WICKETS)

#: Columns of the parse_results table, and the header of unparsed_results.csv (criterion 15).
PARSED_COLUMNS = ["match_id", "result_type", "margin", "dls"]
UNPARSED_HEADER = ["match_id", "result"]

#: 1-8 matches as (match_id, ResultCase) pairs, with distinct match_ids that fit the int32 column.
MATCH_ROWS = st.lists(
    st.tuples(st.integers(min_value=1, max_value=2**31 - 1), result_strings()),
    min_size=1,
    max_size=8,
    unique_by=lambda row: row[0],
)

#: The worked examples of Requirement 5, criteria 13 and 14, plus an unparsed string with a comma.
WORKED_EXAMPLES = [
    (5, ResultCase("India Won Super Over By 10 Runs", ParsedResult(SUPER_OVER_WIN, None, False))),
    (3, ResultCase("won by 17runs (DLS Method)", ParsedResult(WIN_BY_RUNS, 17, True))),
    (1, ResultCase("Won 3 Wkts", ParsedResult(WIN_BY_WICKETS, 3, False))),
    (4, ResultCase("won by 44 run", ParsedResult(WIN_BY_RUNS, 44, False))),
    (2, ResultCase("Match drawn, no play", ParsedResult(UNPARSED, None, False))),
]


def _matches_table(rows: list[tuple[int, ResultCase]]) -> pd.DataFrame:
    """A matches table in drawn row order, with the Data_Loader dtypes of match_id and result."""
    return pd.DataFrame(
        {
            "match_id": pd.array([match_id for match_id, _ in rows], dtype=MATCH_COLUMNS["match_id"]),
            "result": pd.array([case.text for _, case in rows], dtype=MATCH_COLUMNS["result"]),
        }
    )


def _table_rows(table: pd.DataFrame) -> list[ParsedResult]:
    """Each row of a parse_results table as a ParsedResult, with margin None where it is missing."""
    return [
        ParsedResult(result_type, None if pd.isna(margin) else int(margin), bool(dls))
        for result_type, margin, dls in zip(table["result_type"], table["margin"], table["dls"])
    ]


def _read_unparsed(path: Path) -> tuple[list[str], list[tuple[int, str]]]:
    """The header and the (match_id, result) records of unparsed_results.csv, strings as written."""
    with path.open(newline="", encoding="utf-8") as handle:
        header, *records = csv.reader(handle)
    return header, [(int(match_id), text) for match_id, text in records]


@given(rows=MATCH_ROWS)
@example(rows=WORKED_EXAMPLES)
def test_result_parser_round_trip(
    rows: list[tuple[int, ResultCase]], tmp_path_factory: pytest.TempPathFactory, outputs_dir: Path
) -> None:
    # parse_result returns the ParsedResult of the grammar derivation for every string (13-15).
    for _, case in rows:
        parsed = parse_result(case.text)
        assert parsed == case.expected, f"parse_result({case.text!r}) = {parsed}, expected {case.expected}"
        assert (parsed.margin is not None) == (parsed.result_type in MARGIN_TYPES), parsed

    # parse_results: one row per match, sorted by match_id, each equal to parse_result of its string.
    text_of = {match_id: case.text for match_id, case in rows}
    matches = _matches_table(rows)
    table = parse_results(matches)
    ordered_ids = sorted(text_of)
    assert table.columns.tolist() == PARSED_COLUMNS
    assert table["match_id"].tolist() == ordered_ids
    assert table["margin"].dtype == "Int32"
    assert table["result_type"].isin(RESULT_TYPES).all()
    assert _table_rows(table) == [parse_result(text_of[match_id]) for match_id in ordered_ids]
    assert (table["margin"].notna() == table["result_type"].isin(MARGIN_TYPES)).all()

    # write_unparsed records each UNPARSED match with its original string (15), in a temporary folder.
    base = tmp_path_factory.mktemp("p09_outputs")
    assert not base.resolve().is_relative_to(outputs_dir.resolve())
    audit = base / "audit"
    path = write_unparsed(audit, table, matches)
    assert path == audit / UNPARSED_RESULTS_FILE
    header, records = _read_unparsed(path)
    assert header == UNPARSED_HEADER
    expected = sorted((match_id, case.text) for match_id, case in rows if case.expected.result_type == UNPARSED)
    assert records == expected
