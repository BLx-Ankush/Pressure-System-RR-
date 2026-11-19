"""Match 40767 first-innings total test (Requirement 5, criterion 26).

Match 40767 is the IPL 2025 final. ``tests/fixtures/match_40767.json`` stores the hand-checked
first-innings total, 190, and its parts: 181 bat runs and 9 Wides of 1 run each. The values were
worked out from the raw rows of ``ball_by_ball.csv`` without the ``superr`` package, and the total
agrees with the published scorecard (190 for 9); the fixture's ``how_checked`` list gives the
steps. Innings 1 holds only plain Legal_Balls (``extras_type`` ``None``) and Wides, and
``extras_2_runs`` is 0 on every row, so the total is 190 under every candidate Extras_Encoding
(design finding F-11).

The test rebuilds the scores from the real Source_CSVs the way the ``reconstruct`` stage does,
without reading ``outputs/`` or a checkpoint:

1. load the six Source_CSVs from their Config paths (``superr.io.load_all``);
2. select the Extras_Encoding by the criterion 7 rule over every match: team runs under each
   Config candidate, innings aggregates, Score_Checks, Exact_Pass_Rates and ``select_encoding``.
   Counted_Wickets use the Auditor's classification of the dismissal codes, as in the Pipeline;
3. read the innings-1 total of match 40767 from the innings aggregates of the selected encoding.

That total must equal the fixture's total exactly. A second test checks that the reconstructed
total is made of the hand-checked parts: the bat-run sum, the Wide count and the Wide runs. The
real-data work runs once per module, in the ``reconstruction`` fixture.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import pandas as pd
import pytest

from superr import audit
from superr.io import load_all
from superr.reconstruct import (
    DeliveryClass,
    candidate_team_runs,
    check_mapping,
    classify,
    encoding_inputs,
    exact_pass_rates,
    innings_aggregates,
    order_deliveries,
    over_check,
    parse_results,
    run_checks,
    select_encoding,
    wicket_flags,
)

if TYPE_CHECKING:
    from superr.settings import Settings

FIXTURE_NAME = "match_40767.json"


class Reconstruction(NamedTuple):
    """Scores of every match rebuilt from the real Source_CSVs, as the ``reconstruct`` stage rebuilds them."""

    balls: pd.DataFrame  # every Delivery in the Requirement 5 order, on a RangeIndex
    cls: pd.Series  # DeliveryClass code of each Delivery
    team_runs: dict[str, np.ndarray]  # team runs of each Delivery under each Config candidate
    aggregates: dict[str, pd.DataFrame]  # innings_aggregates table under each Config candidate
    selected: str  # the Extras_Encoding selected under criterion 7


@pytest.fixture(scope="module")
def expected(fixtures_dir: Path) -> dict[str, Any]:
    """The hand-checked values of match 40767, innings 1."""
    return json.loads((fixtures_dir / FIXTURE_NAME).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def reconstruction(settings: Settings) -> Reconstruction:
    """Rebuild every match once per module, with the Config values that the ``reconstruct`` stage reads."""
    tables = load_all(settings)
    deliveries, matches = tables["ball_by_ball"], tables["matches"]
    classes = settings.get("data_rules.extras_type_classes")

    # The Counted_Wicket codes, from the dismissal-code classification that the audit stage saves.
    dismissal_classes = audit.classify_dismissal_codes(
        audit.dismissal_code_table(deliveries), audit.reappearance_share(deliveries)
    )

    check_mapping(classes)
    balls = order_deliveries(deliveries)
    cls = classify(balls["extras_type"], classes)
    wickets = wicket_flags(
        balls,
        audit.counted_wicket_codes(dismissal_classes),
        settings.get("design_choices.D14_bowler_wicket_codes"),
        settings.get("design_choices.D14_run_out_code"),
        settings.get("design_choices.D14_retire_out_code"),
    )
    team_runs = candidate_team_runs(encoding_inputs(balls, cls), settings.get("data_rules.candidate_encodings"))
    aggregates = {name: innings_aggregates(balls, runs, wickets["counted_wicket"]) for name, runs in team_runs.items()}

    # Criterion 7 over every match: the candidate with the highest mean of its five Exact_Pass_Rates.
    parsed = parse_results(matches)
    over_ok = over_check(balls, cls)
    checks = {name: run_checks(matches, agg, parsed, over_ok, encoding=name) for name, agg in aggregates.items()}
    selected = select_encoding(exact_pass_rates(checks))
    return Reconstruction(balls, cls, team_runs, aggregates, selected)


def _innings_rows(balls: pd.DataFrame, match_id: int, innings: int) -> np.ndarray:
    """True on the Deliveries of one innings of one match."""
    found = (balls["match_id"] == match_id) & (balls["innings"] == innings)
    return found.to_numpy(dtype=bool, na_value=False)


def test_first_innings_total_under_the_selected_encoding_equals_the_hand_checked_total(
    reconstruction: Reconstruction, expected: dict[str, Any]
) -> None:
    match_id, innings = expected["match_id"], expected["innings"]
    agg = reconstruction.aggregates[reconstruction.selected]
    row = agg.loc[(agg["match_id"] == match_id) & (agg["innings"] == innings)]
    assert len(row) == 1, f"the innings aggregates hold {len(row)} rows for match {match_id}, innings {innings}"
    total = int(row["total"].iloc[0])
    assert total == expected["first_innings_total"], (
        f"match {match_id}, innings {innings}: the total under the selected Extras_Encoding "
        f"{reconstruction.selected} is {total}; the hand-checked total is {expected['first_innings_total']}"
    )


def test_reconstructed_total_is_the_hand_checked_bat_runs_plus_one_run_per_wide(
    reconstruction: Reconstruction, expected: dict[str, Any]
) -> None:
    # The hand-checked parts add up to the hand-checked total: 181 + 9 = 190.
    assert expected["bat_runs"] + expected["wide_runs"] == expected["first_innings_total"]

    rows = _innings_rows(reconstruction.balls, expected["match_id"], expected["innings"])
    innings = reconstruction.balls.loc[rows]
    # Only plain Legal_Balls and Wides: no No_Ball, bye, leg-bye or Penalty Delivery.
    assert innings["extras_type"].value_counts().to_dict() == expected["deliveries_by_extras_type"]
    assert int(innings["runs"].sum()) == expected["bat_runs"]

    # The Wides under the Config class mapping, and their team runs under the selected encoding.
    wide = reconstruction.cls.to_numpy()[rows] == DeliveryClass.WIDE
    runs = reconstruction.team_runs[reconstruction.selected][rows]
    assert int(wide.sum()) == expected["wides"]
    assert int(runs[wide].sum()) == expected["wide_runs"], f"Wide runs under {reconstruction.selected}"
    # The other Deliveries score their bat runs and nothing else.
    assert int(runs[~wide].sum()) == expected["bat_runs"], f"Legal_Ball runs under {reconstruction.selected}"
