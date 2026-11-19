"""Robustness-table rows: the comparison values, the not-comparable rule and the CSV file (Requirement 21,
criteria 5, 6 and 8).

Hand-built rankings of one Role go through ``ranks.compare_rankings`` (Spearman on ``final_score``,
top 10, names from a mapping), then ``cases.comparison_row``, ``cases.robustness_frame`` and
``cases.write_robustness_table``, as ``cases.robustness_table`` chains them for every case.

MAIN ranks players 101-112 with Final_Scores 12 down to 1, so its top 10 is 101-110. The cases:

* SWAP ranks 101-113: 101-108 keep 12 to 5, then 111 (4.5), the new player 113 (4.4), 109 (4),
  110 (3) and 112 (1). Its top 10 is 101-108, 111 and 113: overlap 8, entering 111 and 113 (case
  rank order), leaving 109 and 110 (Main_Ranking rank order). The 12 common players keep their
  order except 111, which passes 109 and 110: rank differences -1, -1 and +2, so Spearman =
  1 - 6 x 6 / (12 x 143) = 140/143.
* FEW ranks only 101-109: 9 Eligible_Players, fewer than 10, so the row is not comparable. Its
  top list is all 9; 110 leaves and nobody enters.
* APART ranks 101, 102 and the new players 201-208: 10 Eligible_Players but 2 in common, fewer
  than 3, so the row is not comparable. 201-208 enter and 103-110 leave.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from superr.cases import (
    LIST_SEPARATOR,
    ROBUSTNESS_COLUMNS,
    ROBUSTNESS_FILE,
    comparison_row,
    robustness_frame,
    write_robustness_table,
)
from superr.ranks import compare_rankings


def ranking(scores: Mapping[int, float]) -> pd.DataFrame:
    """A Role ranking table (player_id, rank, final_score) ranking every player of ``scores``, best first."""
    ids = sorted(scores)
    order = sorted(scores, key=lambda pid: -scores[pid])
    rank = {pid: position for position, pid in enumerate(order, start=1)}
    return pd.DataFrame(
        {
            "player_id": np.asarray(ids, dtype=np.int64),
            "rank": pd.array([rank[pid] for pid in ids], dtype="Int64"),
            "final_score": np.asarray([scores[pid] for pid in ids], dtype=np.float64),
        }
    )


MAIN = {pid: float(113 - pid) for pid in range(101, 113)}
SWAP = {**{pid: float(113 - pid) for pid in range(101, 109)}, 111: 4.5, 113: 4.4, 109: 4.0, 110: 3.0, 112: 1.0}
FEW = {pid: float(113 - pid) for pid in range(101, 110)}
APART = {101: 20.0, 102: 19.0, **{pid: float(218 - pid) for pid in range(201, 209)}}
NAMES = {pid: f"Player {pid}" for pid in [*range(101, 114), *range(201, 209)]}


def row(case: str, scores: Mapping[int, float]) -> dict:
    comparison = compare_rankings(ranking(MAIN), ranking(scores), top_n=10, on="final_score", names=NAMES)
    return comparison_row(
        comparison, case=case, kind="alternative", target_id="D0", setting=f"{case} setting", layer="rating",
        role="batter", n_eligible_case=len(scores), n_eligible_main=len(MAIN),
    )  # fmt: skip


def joined(ids: list[int], names: bool = False) -> str:
    return LIST_SEPARATOR.join(NAMES[pid] if names else str(pid) for pid in ids)


def test_a_comparable_row_holds_the_spearman_overlap_counts_and_players() -> None:
    """SWAP: Spearman over the 12 common players on Final_Score, overlap 8, the counts, and the entering and
    leaving players with names in rank order (module docstring).

    **Validates: Requirements 21.5, 21.6**
    """
    found = row("SWAP", SWAP)
    assert found["comparable"] is True
    assert found["spearman"] == pytest.approx(140 / 143, abs=1e-12)
    assert (found["overlap"], found["n_common"], found["n_eligible_case"], found["n_eligible_main"]) == (8, 12, 13, 12)
    assert found["entering_ids"] == joined([111, 113]) and found["entering_names"] == joined([111, 113], names=True)
    assert found["leaving_ids"] == joined([109, 110]) and found["leaving_names"] == joined([109, 110], names=True)
    same = row("SAME", MAIN)
    assert (same["spearman"], same["overlap"]) == (1.0, 10)
    assert same["entering_ids"] == same["entering_names"] == same["leaving_ids"] == same["leaving_names"] == ""


@pytest.mark.parametrize(
    ("case", "scores", "counts", "entering", "leaving"),
    [
        ("FEW", FEW, (9, 9), [], [110]),
        ("APART", APART, (10, 2), list(range(201, 209)), list(range(103, 111))),
    ],
)
def test_a_case_with_too_few_players_is_not_comparable_and_keeps_its_counts(
    case: str, scores: Mapping[int, float], counts: tuple[int, int], entering: list[int], leaving: list[int]
) -> None:
    """FEW (9 Eligible_Players) and APART (2 common players): not comparable, Spearman and overlap blank, the
    Eligible_Player counts and the players kept.

    **Validates: Requirements 21.8**
    """
    found = row(case, scores)
    assert found["comparable"] is False
    assert math.isnan(found["spearman"]) and found["overlap"] is pd.NA
    assert (found["n_eligible_case"], found["n_common"], found["n_eligible_main"]) == (*counts, 12)
    assert found["entering_ids"] == joined(entering) and found["leaving_ids"] == joined(leaving)


def test_the_table_file_has_the_columns_ten_decimals_and_blank_cells(tmp_path: Path) -> None:
    """The CSV holds ROBUSTNESS_COLUMNS in order, Spearman with 10 decimals, blank Spearman and overlap for a
    not-comparable row, and reads back to the same values; a not-comparable row with values is refused.

    **Validates: Requirements 21.6, 21.8**
    """
    table = robustness_frame([row("SWAP", SWAP), row("FEW", FEW), row("APART", APART)])
    path = write_robustness_table(table, tmp_path / "robustness")
    assert path == tmp_path / "robustness" / ROBUSTNESS_FILE
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(ROBUSTNESS_COLUMNS)
    cells = dict(zip(ROBUSTNESS_COLUMNS, lines[1].split(",")))
    assert (cells["spearman"], cells["overlap"], cells["comparable"]) == (f"{140 / 143:.10f}", "8", "True")
    blank = dict(zip(ROBUSTNESS_COLUMNS, lines[2].split(",")))
    shown = (blank["spearman"], blank["overlap"], blank["comparable"], blank["n_eligible_case"])
    assert shown == ("", "", "False", "9")
    back = pd.read_csv(path)
    assert back["case"].tolist() == ["SWAP", "FEW", "APART"]
    assert back["n_common"].tolist() == [12, 9, 2] and back["overlap"].isna().tolist() == [False, True, True]
    assert back["entering_names"].iloc[0] == joined([111, 113], names=True)
    bad = {**row("FEW", FEW), "spearman": 0.5}
    with pytest.raises(ValueError, match="not comparable"):
        robustness_frame([bad])
