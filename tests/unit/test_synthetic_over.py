"""Synthetic-over fixture test (Requirement 5, criterion 25).

``tests/fixtures/synthetic_over.csv`` is one Over of 10 Deliveries in the Requirement 5 order:
6 Legal_Balls (``None`` three times, ``legbyes``, ``Penalty`` and ``byes``), 2 Wides and 2
No_Balls (``other`` and ``Overstep``), with a Wide straight after the ``other`` No_Ball. Each row
holds the Delivery's inputs (``extras_type``, b = ``runs`` and e = ``extras_2_runs``) and its
expected values, worked out by hand from the rules below, never copied from the code's output.
The ``note`` column shows the working for each row.

* ``exp_class``: the class key (``legal``, ``wide`` or ``no_ball``) that the delivered Config
  class mapping gives the ``extras_type`` (criterion 2).
* ``exp_free_hit``: the Free_Hit flag (criterion 12). The ``other`` No_Ball (ball 3) is not a
  Free_Hit, because no No_Ball comes before it. The Wide (ball 4), the ``Overstep`` No_Ball
  (ball 5) and the next Legal_Ball (ball 6) are Free_Hits. Ball 6 ends the span, so balls 7 to
  10 are not Free_Hits, and neither are balls 1 and 2.
* ``exp_runs_<name>``: the team runs under each candidate Extras_Encoding, from the E1-E4 table
  of the design. A Legal_Ball scores b + e. A bye or leg-bye scores b + e under ``F30`` and
  1 + e under the three ``byes_one`` encodings. A Penalty Delivery scores b + e, or 5 + b + e
  under ``byes_one_penalty5``. A Wide scores 1 + e. A No_Ball scores 1 + b + e, or 1 + b under
  ``byes_one_nb_plain``.

The values make every pair of encodings disagree on some Delivery and on the Over total: the
bye and the leg-bye have b = 0, the ``other`` No_Ball has b > 0 and e > 0, and the Penalty
Delivery has b > 0. Each class also has a Delivery with e > 0, so every ``+ e`` term of the
table is tested.

The tests take the class mapping and the candidate list from the Config through the
``settings`` fixture. The fixture is read with ``dtype=str, keep_default_na=False,
na_values=[""]``, so the text ``None`` stays a value.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from superr.reconstruct import (
    CLASS_KEYS,
    DeliveryClass,
    candidate_team_runs,
    classify,
    encoding_inputs,
    free_hit_flags,
)

if TYPE_CHECKING:
    from superr.settings import Settings

FIXTURE_NAME = "synthetic_over.csv"

#: Legal_Balls in the synthetic Over (criterion 25).
EXPECTED_LEGAL_BALLS = 6

# Hand-computed team runs of the whole Over under each candidate Extras_Encoding. Ball numbers
# are innings_ball_num values.
#   F30                Legal_Balls b + e: 1 (ball 1) + 4 (6) + 0 (7) + 2 (8) + 1 (9) + 1 (10) = 9;
#                      Wides 1 + e: 1 (2) + 5 (4) = 6; No_Balls 1 + b + e: 4 (3) + 2 (5) = 6   -> 21
#   byes_one           F30 plus 1 - b for the leg-bye (7) and the bye (9): 21 + 1 + 1          -> 23
#   byes_one_nb_plain  byes_one minus e of each No_Ball: 23 - 1 (3) - 0 (5)                    -> 22
#   byes_one_penalty5  byes_one plus 5 for the Penalty Delivery (8): 23 + 5                    -> 28
EXPECTED_OVER_TOTALS: dict[str, int] = {
    "F30": 21,
    "byes_one": 23,
    "byes_one_nb_plain": 22,
    "byes_one_penalty5": 28,
}

_CODE_OF_CLASS_KEY = {key: int(delivery_class) for delivery_class, key in CLASS_KEYS.items()}
_FLAG_OF_TEXT = {"true": True, "false": False}


@pytest.fixture(scope="module")
def over(fixtures_dir: Path) -> pd.DataFrame:
    """The synthetic Over, read as text so that the extras_type text None stays a value."""
    return pd.read_csv(fixtures_dir / FIXTURE_NAME, dtype=str, keep_default_na=False, na_values=[""])


@pytest.fixture(scope="module")
def cls(over: pd.DataFrame, settings: Settings) -> pd.Series:
    """DeliveryClass code of each Delivery under the Config class mapping."""
    return classify(over["extras_type"], settings.get("data_rules.extras_type_classes"))


def _expected_classes(over: pd.DataFrame) -> list[int]:
    codes = over["exp_class"].map(_CODE_OF_CLASS_KEY)
    assert codes.notna().all(), f"exp_class must be one of {', '.join(_CODE_OF_CLASS_KEY)}"
    return codes.astype(int).tolist()


def _expected_free_hits(over: pd.DataFrame) -> list[bool]:
    flags = over["exp_free_hit"].map(_FLAG_OF_TEXT)
    assert flags.notna().all(), "exp_free_hit must be true or false"
    return flags.astype(bool).tolist()


def _expected_runs(over: pd.DataFrame, name: str) -> list[int]:
    return over[f"exp_runs_{name}"].astype(int).tolist()


def test_fixture_is_the_over_that_criterion_25_describes(over: pd.DataFrame) -> None:
    # One Over, its rows in the Requirement 5 order, so "straight after" means the next row.
    assert len(over[["match_id", "innings", "overid"]].drop_duplicates()) == 1
    ball = over["innings_ball_num"].astype(int)
    assert ball.is_unique and ball.is_monotonic_increasing
    kind, extras_type = over["exp_class"], over["extras_type"]
    assert (kind == "legal").sum() == EXPECTED_LEGAL_BALLS
    assert (kind == "wide").any()
    assert ((kind == "no_ball") & (extras_type == "other")).any()
    assert ((kind == "no_ball") & (extras_type == "Overstep")).any()
    assert ((kind == "wide") & (kind.shift() == "no_ball")).any()  # a Wide straight after a No_Ball
    assert (extras_type == "Penalty").any()
    # The two hand computations agree: each Over total is the sum of its exp_runs column.
    assert {name: sum(_expected_runs(over, name)) for name in EXPECTED_OVER_TOTALS} == EXPECTED_OVER_TOTALS
    # Each encoding has its own Over total, so the test can tell every pair of encodings apart.
    assert len(set(EXPECTED_OVER_TOTALS.values())) == len(EXPECTED_OVER_TOTALS)


def test_classify_reproduces_every_class_and_the_six_legal_balls(over: pd.DataFrame, cls: pd.Series) -> None:
    assert cls.tolist() == _expected_classes(over)
    assert int((cls == DeliveryClass.LEGAL).sum()) == EXPECTED_LEGAL_BALLS


def test_free_hit_flags_reproduce_every_hand_computed_flag(over: pd.DataFrame, cls: pd.Series) -> None:
    assert free_hit_flags(over, cls).tolist() == _expected_free_hits(over)


def test_team_runs_reproduce_every_delivery_and_the_over_total_under_each_encoding(
    over: pd.DataFrame, cls: pd.Series, settings: Settings
) -> None:
    names = settings.get("data_rules.candidate_encodings")
    no_hand_values = [name for name in names if name not in EXPECTED_OVER_TOTALS or f"exp_runs_{name}" not in over]
    assert names and not no_hand_values, f"hand-compute exp_runs_<name> and the Over total for {no_hand_values}"
    runs = candidate_team_runs(encoding_inputs(over, cls), names)
    per_delivery = {name: values.tolist() for name, values in runs.items()}
    assert per_delivery == {name: _expected_runs(over, name) for name in names}
    totals = {name: int(values.sum()) for name, values in runs.items()}
    assert totals == {name: EXPECTED_OVER_TOTALS[name] for name in names}
