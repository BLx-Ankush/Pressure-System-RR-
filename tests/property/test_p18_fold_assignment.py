# Feature: pressure-rating-system, Property 18: For any set of matches with Competition_Types and any fold seed, every match receives exactly one fold in 1–5, every Delivery inherits its match's fold, within each Competition_Type the fold sizes differ by at most 1, and two runs with the same seed give identical Fold_Tables.
"""Property 18: Fold assignment (Requirement 7, criteria 1-4).

**Validates: Requirements 7.1, 7.2, 7.3, 7.4**

Each example starts from ``tests.strategies.fold_tables()``: matches with distinct, non-consecutive
``match_id`` values (Int64, rows shuffled) and a Competition_Type each, where a type mostly has
3-15 matches but sometimes only 1-2 or none; a fold count n from 2-7, D15 most often; a fold seed
(any 32-bit value, or the Config's ``seeds.folds``); and 0-3 Deliveries per match, so most examples
hold matches with no Delivery. The example adds a drawn row order of the matches and one of the
Deliveries.

The test builds the Fold_Table with its own call to ``assign_folds`` and checks that:

* every match gets exactly one fold in 1..n (1-5 at D15 = 5): the table has the columns
  ``match_id``, ``comp_type`` and ``fold`` and one row per match, sorted by ``match_id``, with the
  match's own Competition_Type and an integer fold from 1 to n (criterion 1);
* every Delivery gets the Fold_Table fold of its match, with its row order and index kept, from
  ``attach_folds`` on the Delivery table, from ``attach_folds(..., balls=...)`` on a frame keyed by
  ``row`` alone in another row order (as the ``state`` table is), and from ``fold_of`` on the
  Deliveries' match ids; ``fold_of`` on the match ids in the row order of ``matches`` gives every
  match its fold, including the matches with no Delivery (criteria 1 and 2);
* for each Competition_Type, the numbers of its matches in any two of the folds 1..n differ by at
  most 1, where a fold with none of them counts as 0 (criterion 3);
* a second run with the same matches, n and seed, the strategy's own earlier run, and a run on the
  match rows in another order (index labels moved with their rows) each give a Fold_Table
  identical to the first in values and dtypes (criterion 4).
"""

from __future__ import annotations

from collections import Counter
from typing import Any, NamedTuple

import pandas as pd
from hypothesis import event, given, note
from hypothesis import strategies as st

from superr.folds import FOLD_COLUMN, FOLD_TABLE_COLUMNS, assign_folds, attach_folds, fold_of
from superr.reconstruct import COMPETITION_TYPES
from tests.strategies import FoldTables, fold_tables


class Case(NamedTuple):
    """A fold_tables() draw, a row order of its matches and a row order of its Deliveries."""

    tables: FoldTables
    match_order: list[int]
    delivery_order: list[int]

    def __repr__(self) -> str:
        tables = self.tables
        counts = Counter(tables.deliveries["match_id"].tolist())
        per_match = {match: counts[match] for match in sorted(tables.matches["match_id"].tolist())}
        return (
            f"n_folds {tables.n_folds}, seed {tables.seed}\n"
            f"matches, in row order:\n{tables.matches.to_string()}\n"
            f"Deliveries per match (match_id: count, rows numbered in match_id order): {per_match}\n"
            f"match row order {self.match_order}\nDelivery row order {self.delivery_order}"
        )


@st.composite
def cases(draw: Any) -> Case:
    """A fold_tables() draw plus drawn row orders of its matches and of its Deliveries."""
    tables = draw(fold_tables())
    return Case(
        tables,
        draw(st.permutations(range(len(tables.matches)))),
        draw(st.permutations(range(len(tables.deliveries)))),
    )


class _Table:
    """A DataFrame for hypothesis.note, formatted only when Hypothesis prints the note."""

    def __init__(self, title: str, frame: pd.DataFrame) -> None:
        self.title, self.frame = title, frame

    def __repr__(self) -> str:
        return f"{self.title}:\n{self.frame.to_string()}"


@given(case=cases())
def test_fold_assignment(case: Case) -> None:
    """Every match gets one fold in 1..n and every Delivery its match's fold; each Competition_Type's
    fold sizes differ by at most 1; the same seed gives the same Fold_Table (Requirement 7, criteria 1-4).

    **Validates: Requirements 7.1, 7.2, 7.3, 7.4**
    """
    matches, deliveries = case.tables.matches, case.tables.deliveries
    n, seed = case.tables.n_folds, case.tables.seed
    ids = [int(match) for match in matches["match_id"].tolist()]
    type_of = dict(zip(ids, (str(name) for name in matches["comp_type"].tolist())))

    table = assign_folds(matches, matches["comp_type"], n_folds=n, seed=seed)
    note(_Table("Fold_Table", table))

    # Criterion 1: one row per match, sorted by match_id, with its Competition_Type and one fold in 1..n.
    assert table.columns.tolist() == list(FOLD_TABLE_COLUMNS)
    table_ids = table["match_id"].tolist()
    assert table_ids == sorted(ids), "the Fold_Table does not hold every match exactly once, sorted by match_id"
    assert table["comp_type"].tolist() == [type_of[match] for match in table_ids]
    assert pd.api.types.is_integer_dtype(table[FOLD_COLUMN].dtype), f"fold has dtype {table[FOLD_COLUMN].dtype}"
    assert table[FOLD_COLUMN].notna().all(), "a match has no fold"
    fold_by_match = dict(zip(table_ids, table[FOLD_COLUMN].tolist()))
    outside = {match: fold for match, fold in fold_by_match.items() if not 1 <= fold <= n}
    assert not outside, f"matches with a fold outside 1-{n}: {outside}"

    # Criterion 2: every Delivery gets the Fold_Table fold of its match; row order and index are kept.
    expected = [fold_by_match[match] for match in deliveries["match_id"].tolist()]
    with_folds = attach_folds(deliveries, table)
    assert with_folds.columns.tolist() == [*deliveries.columns, FOLD_COLUMN]
    assert with_folds.index.equals(deliveries.index)
    assert with_folds["match_id"].tolist() == deliveries["match_id"].tolist()
    assert with_folds[FOLD_COLUMN].tolist() == expected, "attach_folds gives a Delivery another match's fold"
    assert fold_of(deliveries["match_id"], table).tolist() == expected, "fold_of gives a Delivery another match's fold"

    # A frame keyed by row alone (as the state table), in another row order: balls gives each row's match.
    match_of_row = dict(zip(deliveries["row"].tolist(), deliveries["match_id"].tolist()))
    keyed = deliveries[["row"]].iloc[case.delivery_order]
    by_row = attach_folds(keyed, table, balls=deliveries)
    assert by_row.index.equals(keyed.index)
    assert by_row["row"].tolist() == keyed["row"].tolist()
    assert by_row[FOLD_COLUMN].tolist() == [fold_by_match[match_of_row[row]] for row in keyed["row"].tolist()], (
        "attach_folds with balls= gives a row another match's fold"
    )

    # Every match, with or without Deliveries, gets its fold from fold_of, in the row order of matches.
    assert fold_of(matches["match_id"], table).tolist() == [fold_by_match[match] for match in ids]
    if set(ids) - set(match_of_row.values()):
        event("a match with no Delivery")

    # Criterion 3: within each Competition_Type, the sizes of folds 1..n differ by at most 1.
    for name in COMPETITION_TYPES:
        member_folds = [fold_by_match[match] for match in ids if type_of[match] == name]
        sizes = [member_folds.count(fold) for fold in range(1, n + 1)]
        assert max(sizes) - min(sizes) <= 1, f"{name}: {len(member_folds)} matches in folds 1-{n} of sizes {sizes}"
        if not member_folds:
            event("a Competition_Type with no match")
        elif len(member_folds) < n:
            event("a Competition_Type with fewer matches than folds")
    event(f"n_folds = {n}")

    # Criterion 4: the same matches, n and seed give an identical Fold_Table, whatever the row order of matches.
    again = assign_folds(matches, matches["comp_type"], n_folds=n, seed=seed)
    pd.testing.assert_frame_equal(again, table, check_exact=True, obj="second run with the same seed")
    pd.testing.assert_frame_equal(case.tables.fold_table, table, check_exact=True, obj="the strategy's run")
    reordered = matches.iloc[case.match_order]
    pd.testing.assert_frame_equal(
        assign_folds(reordered, reordered["comp_type"], n_folds=n, seed=seed),
        table,
        check_exact=True,
        obj="run on the matches in another row order",
    )
