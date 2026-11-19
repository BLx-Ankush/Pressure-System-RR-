"""Score_Tolerance unit tests (Requirement 5, criteria 24, 27 and 28).

Synthetic matches with hand-computed scores go through the Score_Reconstructor functions that the
``reconstruct`` stage chains: classify, wicket_flags, candidate_team_runs, innings_aggregates,
parse_results, over_check and run_checks under every candidate Extras_Encoding of the Config, then
exact_pass_rates and select_encoding. The ``score_checks`` rows of the selected encoding feed
failing_matches, exclusion_conditions, exclusion_counts, excluded_flags and
write_reconstruction_audit.

Every Delivery is a plain Legal_Ball (``extras_type`` ``None``, ``extras_2_runs`` 0), so it scores
its bat runs under every encoding and each innings total is the sum of its bat runs. Innings 1 is
B runs in sixes plus one remainder ball. Innings 2 opens with its W2 wickets (caught, 0 runs), then
sixes, a remainder ball and a final Delivery of f runs. Overs hold 6 Legal_Balls and the last Over
at most 6, so the over check passes, except where a case opens innings 1 with an Over of 5
Legal_Balls.

Each case is a scorecard written in terms of t, the Score_Tolerance D20 read from the Config, so the
cases follow any change to it. With k = t + 1, every match has B = 150 + 3t first-innings runs and
no ``revised_target``, so the Target is T = B + 1 (criterion 10) and T - 1 = B. B keeps every total
positive for any t: the smallest, 1002's B - m - t, is 148. Royals bat first and Capitals chase.
The Runs_Check_Miss of each case is worked out by hand from criterion 16 (|(B - S2) - m|, where
m = 2k is the recorded runs margin), criterion 17 (the larger of the shortfall T - S2 and the
excess (S2 - f) - B, each floored at 0) or criterion 19 (|S2 - B|). Values in brackets hold at the
delivered D20 of 6 (B = 168, T = 169, m = 14):

====  ===========================================  ===============  ==  ==  ===================================
id    case                                         S2               f   W2  Runs_Check_Miss
====  ===========================================  ===============  ==  ==  ===================================
1001  runs-margin, "Royals won by m runs"          B - m (154)              |m - m| = 0
1002  runs-margin                                  B - m - t (148)          |(m + t) - m| = t
1003  runs-margin                                  B - m + k (161)          |(m - k) - m| = k
1004  chase, "Capitals won by 6 wickets"           B + 2 (170)      4   4   shortfall -1, excess -2: 0
1005  chase, driven by the shortfall               T - t (163)      1   4   shortfall t, excess -t: t
1006  chase, driven by the excess                  B + t + 4 (178)  4   4   shortfall -(t + 3), excess t: t
1007  chase, driven by the shortfall               T - k (162)      1   4   shortfall k, excess -k: k
1008  chase, driven by the excess                  B + k + 4 (179)  4   4   shortfall -(k + 3), excess k: k
1009  tie, "Match tied"                            B (168)                  |B - B| = 0
1010  tie, "Match tied"                            B - t (162)              |-t| = t
1011  tie check of a super-over win                B + k (175)              |k| = k
1012  1001 with a first Over of 5 Legal_Balls      B - m (154)              0; fails the over check only
1013  1004 with W2 = 3 (6 wickets, not 10 - 3)     B + 2 (170)      4   3   0; fails the wickets-margin check only
1014  1003 with a first Over of 5 Legal_Balls      B - m + k (161)          k; fails the over check as well
====  ===========================================  ===============  ==  ==  ===================================

Under criterion 24 at the Score_Tolerance t, a match is excluded when it fails the over check or
the wickets-margin check, or when a runs-margin, chase or tie Runs_Check_Miss is above t. The
matches that miss by 0 or t are kept (1001, 1002, 1004-1006, 1009 and 1010), and 1003, 1007, 1008
and 1011-1014 are excluded: 7 distinct matches with 8 conditions, because 1014 meets two. With the
delivered grid {0, 3, 6, 10}, criterion 24 would exclude 11, 11, 7 and 3 matches. The tests derive
these counts from each card's hand-computed miss and failures, so they also follow the Config.

The tests check:

* the cards themselves: each stated miss agrees with a direct transcription of criteria 16, 17
  and 19, each chase miss comes from the driver it names, the Deliveries carry the scorecards,
  and the cases cover every bullet of criterion 28;
* each runs check returns the hand-computed miss and passes exactly when the miss is 0;
* under criterion 24, a match whose only failure is a miss of D20 is kept, and a miss of D20 + 1,
  an over-only failure and a wickets-margin-only failure each exclude the match;
* ``excluded_flags`` follows the ``exclude_failing_matches`` option;
* the companion test (criterion 27): ``excluded_matches.csv``, ``exclusion_counts.csv`` and the
  score-reconstruction section of ``audit.md``, written to ``tmp_path`` and read back.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import pandas as pd
import pytest

from superr.audit import AUDIT_MD
from superr.reconstruct import (
    CHECK_CHASE,
    CHECK_OVER,
    CHECK_RUNS_MARGIN,
    CHECK_TIE,
    CHECK_WICKETS_MARGIN,
    EXCLUDED_MATCHES_FILE,
    EXCLUSION_COUNTS_FILE,
    RECONSTRUCTION_HEADING,
    RUNS_CHECKS,
    SCORE_CHECKS,
    candidate_team_runs,
    classify,
    encoding_inputs,
    exact_pass_rates,
    excluded_flags,
    exclusion_conditions,
    exclusion_counts,
    failing_matches,
    innings_aggregates,
    over_check,
    parse_results,
    pass_rate_table,
    run_checks,
    select_encoding,
    wicket_flags,
    within_tolerance_pass_rates,
    write_reconstruction_audit,
)

if TYPE_CHECKING:
    from superr.settings import Settings

ROYALS, CAPITALS = 1, 2  # team ids: Royals bat first in every match, Capitals bat second
CAUGHT = 2  # dismissal_type_id of a catch
#: Counted_Wicket codes (glossary: dismissal codes 1-6 and Retire Out, 9). A Run takes them from the Auditor.
COUNTED_WICKET_CODES = (1, 2, 3, 4, 5, 6, 9)
SIX = 6  # bat runs of each filler ball
LEGAL_BALLS_PER_OVER = 6  # Laws of Cricket
WICKETS_PER_INNINGS = 10  # Laws of Cricket: a chase won by w wickets ends with 10 - w wickets fallen
CHASE_MARGIN = 6  # "Capitals won by 6 wickets"
CHASE_WICKETS = WICKETS_PER_INNINGS - CHASE_MARGIN  # W2 = 4, which the wickets-margin check accepts

# Result kinds, with each kind's result string, winner and runs check (criteria 13, 16, 17 and 19).
RUNS, WICKETS, TIED, SUPER_OVER = "win by runs", "win by wickets", "tie", "super-over win"
RESULT_TEXT = {
    RUNS: "Royals won by {margin} runs",
    WICKETS: "Capitals won by {margin} wickets",
    TIED: "Match tied",
    SUPER_OVER: "Match tied (Royals won the Super Over)",
}
WINNER = {RUNS: ROYALS, WICKETS: CAPITALS, TIED: None, SUPER_OVER: ROYALS}
RUNS_CHECK_OF = {RUNS: CHECK_RUNS_MARGIN, WICKETS: CHECK_CHASE, TIED: CHECK_TIE, SUPER_OVER: CHECK_TIE}

#: The Data_Loader dtypes of the ball_by_ball.csv and matches.csv columns that score reconstruction reads.
BALL_DTYPES = {
    "match_id": "Int64",
    "innings": "Int64",
    "overid": "Int64",
    "innings_ball_num": "Int64",
    "batting_team_id": "Int64",
    "runs": "Int64",
    "extras_type": "string",
    "extras_2_runs": "Int64",
    "is_wicket": "boolean",
    "dismissal_type_id": "Int64",
}
MATCH_DTYPES = {"match_id": "Int64", "revised_target": "Int64", "win_team_id": "Int64", "result": "string"}

#: Stand-in for the Auditor's audit.md text, which a Run reads from the audit checkpoint.
AUDITOR_TEXT = "# Data audit\n\nThe Auditor's text, which the score-reconstruction section follows.\n"


# ------------------------------------------------------------------------------------- the cards

@dataclass(frozen=True)
class Card:
    """One synthetic match: its scorecard and the hand-computed outcome of its Score_Checks."""

    match_id: int
    label: str
    result: str  # RUNS, WICKETS, TIED or SUPER_OVER
    second_total: int  # S2
    miss: int  # hand-computed Runs_Check_Miss of the match's runs check
    margin: int | None = None  # runs or wickets of a win by runs or by wickets
    final_ball_runs: int = 1  # f: runs of the final innings-2 Delivery
    second_wickets: int = 0  # W2
    driver: str | None = None  # "shortfall" or "excess": the term of a chase miss
    wickets_margin_ok: bool = True  # hand-computed: the margin of a win by wickets equals 10 - W2
    short_first_over: bool = False  # innings 1 opens with an Over of 5 Legal_Balls, which fails the over check

    @property
    def runs_check(self) -> str:
        """The runs check that applies: runs-margin (criterion 16), chase (17) or tie (19)."""
        return RUNS_CHECK_OF[self.result]

    @property
    def only_runs_check_can_fail(self) -> bool:
        """True when the match passes the wickets-margin and over checks."""
        return self.wickets_margin_ok and not self.short_first_over

    @property
    def failed(self) -> list[str]:
        """The checks that the match fails exactly (criterion 23), in table order."""
        failed = [self.runs_check] if self.miss > 0 else []
        failed += [] if self.wickets_margin_ok else [CHECK_WICKETS_MARGIN]
        failed += [CHECK_OVER] if self.short_first_over else []
        return sorted(failed, key=SCORE_CHECKS.index)

    def conditions(self, tolerance: int) -> list[tuple[str, int | None]]:
        """The conditions of criterion 24 met at ``tolerance``, in table order, with the miss of a runs check."""
        met: list[tuple[str, int | None]] = [(self.runs_check, self.miss)] if self.miss > tolerance else []
        met += [] if self.wickets_margin_ok else [(CHECK_WICKETS_MARGIN, None)]
        met += [(CHECK_OVER, None)] if self.short_first_over else []
        return sorted(met, key=lambda condition: SCORE_CHECKS.index(condition[0]))


def first_total(d20: int) -> int:
    """B, the first-innings total of every match: 168 at the delivered D20 of 6."""
    return 150 + 3 * d20


def make_cards(t: int) -> list[Card]:
    """The cases of the module docstring for the Score_Tolerance ``t``, in ``match_id`` order."""
    k = t + 1  # the smallest Runs_Check_Miss that criterion 24 excludes
    b = first_total(t)  # T - 1
    target = b + 1  # T: no revised_target, so the innings-1 total + 1 (criterion 10)
    m = 2 * k  # recorded margin of every win by runs
    w, w2 = CHASE_MARGIN, CHASE_WICKETS
    return [
        # Criterion 16: miss = |((T - 1) - S2) - m| = |(B - S2) - m|.
        Card(1001, "runs-margin miss 0", RUNS, b - m, 0, margin=m),
        Card(1002, "runs-margin miss D20", RUNS, b - m - t, t, margin=m),  # |(m + t) - m|
        Card(1003, "runs-margin miss D20 + 1", RUNS, b - m + k, k, margin=m),  # |(m - k) - m|
        # Criterion 17: miss = max(shortfall T - S2, excess (S2 - f) - (T - 1), 0).
        Card(1004, "chase miss 0", WICKETS, b + 2, 0, margin=w, final_ball_runs=4, second_wickets=w2),  # -1, -2
        Card(1005, "chase shortfall miss D20", WICKETS, target - t, t, margin=w, final_ball_runs=1,
             second_wickets=w2, driver="shortfall"),  # shortfall t, excess -t
        Card(1006, "chase excess miss D20", WICKETS, b + t + 4, t, margin=w, final_ball_runs=4,
             second_wickets=w2, driver="excess"),  # shortfall -(t + 3), excess t
        Card(1007, "chase shortfall miss D20 + 1", WICKETS, target - k, k, margin=w, final_ball_runs=1,
             second_wickets=w2, driver="shortfall"),  # shortfall k, excess -k
        Card(1008, "chase excess miss D20 + 1", WICKETS, b + k + 4, k, margin=w, final_ball_runs=4,
             second_wickets=w2, driver="excess"),  # shortfall -(k + 3), excess k
        # Criterion 19: miss = |S2 - (T - 1)| = |S2 - B|.
        Card(1009, "tie miss 0", TIED, b, 0),
        Card(1010, "tie miss D20", TIED, b - t, t),
        Card(1011, "super-over win tie miss D20 + 1", SUPER_OVER, b + k, k),
        # Criteria 20 and 18: a failed over check or wickets-margin check with every runs check passed.
        Card(1012, "over check only", RUNS, b - m, 0, margin=m, short_first_over=True),
        Card(1013, "wickets-margin check only", WICKETS, b + 2, 0, margin=w, final_ball_runs=4,
             second_wickets=w2 - 1, wickets_margin_ok=False),  # 6 wickets, but 10 - 3 = 7
        # Two conditions in one match, so the distinct excluded count differs from the condition count.
        Card(1014, "runs-margin miss D20 + 1 and over check", RUNS, b - m + k, k, margin=m, short_first_over=True),
    ]


def transcribed_miss(card: Card, d20: int) -> int:
    """The card's Runs_Check_Miss from a direct transcription of criterion 16, 17 or 19."""
    target, s2, f = first_total(d20) + 1, card.second_total, card.final_ball_runs
    if card.runs_check == CHECK_RUNS_MARGIN:
        return abs(((target - 1) - s2) - card.margin)
    if card.runs_check == CHECK_CHASE:
        return max(max(target - s2, 0), max((s2 - f) - (target - 1), 0))
    return abs(s2 - (target - 1))


def n_excluded(cards: list[Card], tolerance: int) -> int:
    """Distinct matches that criterion 24 excludes at ``tolerance``, from the hand-computed cards."""
    return sum(1 for card in cards if card.conditions(tolerance))


# -------------------------------------------------------------------------------- the Deliveries

def _bat_runs(total: int, *, wickets: int = 0, final: int | None = None) -> list[int]:
    """Bat runs of each Legal_Ball of an innings of ``total`` runs.

    ``wickets`` dismissals for 0 runs open the innings, then sixes and one remainder ball, then a
    final Delivery of ``final`` runs when it is given.
    """
    sixes, rest = divmod(total - (final or 0), SIX)
    return [0] * wickets + [SIX] * sixes + ([rest] if rest else []) + ([] if final is None else [final])


def _overids(n_balls: int, *, short_first_over: bool) -> list[int]:
    """``overid`` of each Legal_Ball: 6 to an Over, or 5 in the first Over when ``short_first_over``."""
    first = LEGAL_BALLS_PER_OVER - 1 if short_first_over else LEGAL_BALLS_PER_OVER
    return [1 if i < first else 2 + (i - first) // LEGAL_BALLS_PER_OVER for i in range(n_balls)]


def _innings_rows(
    match_id: int, innings: int, batting_team_id: int, bat_runs: list[int], *, wickets: int, short_first_over: bool
) -> Iterator[dict[str, Any]]:
    overs = _overids(len(bat_runs), short_first_over=short_first_over)
    for i, (runs, over) in enumerate(zip(bat_runs, overs)):
        out = i < wickets
        yield {
            "match_id": match_id,
            "innings": innings,
            "overid": over,
            "innings_ball_num": i + 1,
            "batting_team_id": batting_team_id,
            "runs": runs,
            "extras_type": "None",
            "extras_2_runs": 0,
            "is_wicket": out,
            "dismissal_type_id": CAUGHT if out else None,
        }


def build_tables(cards: list[Card], d20: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The Deliveries and the ``matches`` rows of the cards, with the Data_Loader dtypes."""
    rows: list[dict[str, Any]] = []
    for card in cards:
        rows += _innings_rows(card.match_id, 1, ROYALS, _bat_runs(first_total(d20)), wickets=0,
                              short_first_over=card.short_first_over)
        second = _bat_runs(card.second_total, wickets=card.second_wickets, final=card.final_ball_runs)
        rows += _innings_rows(card.match_id, 2, CAPITALS, second, wickets=card.second_wickets, short_first_over=False)
    balls = pd.DataFrame(rows).astype(BALL_DTYPES)
    matches = pd.DataFrame(
        {
            "match_id": [card.match_id for card in cards],
            "revised_target": [None] * len(cards),
            "win_team_id": [WINNER[card.result] for card in cards],
            "result": [RESULT_TEXT[card.result].format(margin=card.margin) for card in cards],
        }
    ).astype(MATCH_DTYPES)
    return balls, matches


# ---------------------------------------------------------------------------------- fixtures

class Reconstruction(NamedTuple):
    """The synthetic dataset after the Score_Checks, as the reconstruct stage holds it."""

    balls: pd.DataFrame
    rates: pd.DataFrame  # exact_pass_rates of every candidate Extras_Encoding
    selected: str  # select_encoding of the rates
    checks: pd.DataFrame  # run_checks rows of the selected encoding


@pytest.fixture(scope="module")
def d20(settings: Settings) -> int:
    """The Score_Tolerance D20 in runs, read from the Config."""
    tolerance = settings.get("design_choices.D20_score_tolerance")
    assert isinstance(tolerance, int) and tolerance >= 0, f"D20 must be a whole number of runs; got {tolerance!r}"
    return tolerance


@pytest.fixture(scope="module")
def cards(d20: int) -> list[Card]:
    return make_cards(d20)


@pytest.fixture(scope="module")
def reconstruction(cards: list[Card], d20: int, settings: Settings) -> Reconstruction:
    """The Score_Checks of the cards under every Config candidate encoding, chained as in run_stage."""
    balls, matches = build_tables(cards, d20)
    cls = classify(balls["extras_type"], settings.get("data_rules.extras_type_classes"))
    counted = wicket_flags(
        balls,
        COUNTED_WICKET_CODES,
        settings.get("design_choices.D14_bowler_wicket_codes"),
        settings.get("design_choices.D14_run_out_code"),
        settings.get("design_choices.D14_retire_out_code"),
    )["counted_wicket"]
    team_runs = candidate_team_runs(encoding_inputs(balls, cls), settings.get("data_rules.candidate_encodings"))
    parsed, over_ok = parse_results(matches), over_check(balls, cls)
    checks = {
        name: run_checks(matches, innings_aggregates(balls, runs, counted), parsed, over_ok, encoding=name)
        for name, runs in team_runs.items()
    }
    rates = exact_pass_rates(checks)
    selected = select_encoding(rates)
    return Reconstruction(balls, rates, selected, checks[selected])


def _check_rows(checks: pd.DataFrame) -> dict[tuple[int, str], tuple[bool, bool | None, int | None]]:
    """(applies, passed, runs_check_miss) of each (match_id, check) row, with None for a blank value."""
    columns = (checks[name].tolist() for name in ("match_id", "check", "applies", "passed", "runs_check_miss"))
    return {
        (int(match_id), str(check)): (
            bool(applies),
            None if pd.isna(passed) else bool(passed),
            None if pd.isna(miss) else int(miss),
        )
        for match_id, check, applies, passed, miss in zip(*columns)
    }


def _read_text_csv(path: Path) -> pd.DataFrame:
    """A written audit CSV as text, so that a blank field reads as ''."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


# ------------------------------------------------------------------------------------- tests

def test_cards_are_the_cases_that_criterion_28_describes(cards: list[Card], reconstruction: Reconstruction,
                                                          d20: int) -> None:
    b = first_total(d20)
    # The two hand computations agree: each stated miss is criterion 16, 17 or 19 applied to the scorecard.
    assert {card.label: card.miss for card in cards} == {card.label: transcribed_miss(card, d20) for card in cards}
    for card in (card for card in cards if card.driver is not None):
        terms = {"shortfall": (b + 1) - card.second_total, "excess": (card.second_total - card.final_ball_runs) - b}
        other = "excess" if card.driver == "shortfall" else "shortfall"
        assert terms[card.driver] == card.miss and terms[other] <= 0, f"{card.label}: {terms}"
    chases = [card for card in cards if card.result == WICKETS]
    assert [card.wickets_margin_ok for card in chases] == [
        card.margin == WICKETS_PER_INNINGS - card.second_wickets for card in chases
    ]

    # Every bullet of criterion 28 has its case.
    only = [card for card in cards if card.only_runs_check_can_fail]
    assert {(card.runs_check, card.miss) for card in only} == {
        (check, miss) for check in RUNS_CHECKS for miss in (0, d20, d20 + 1)
    }
    assert {card.driver for card in only if card.runs_check == CHECK_CHASE and card.miss == d20 and card.driver} == {
        "shortfall",
        "excess",
    }
    assert [card.label for card in cards if card.failed == [CHECK_OVER]] == ["over check only"]
    assert [card.label for card in cards if card.failed == [CHECK_WICKETS_MARGIN]] == ["wickets-margin check only"]

    # The Deliveries carry the scorecards.
    balls = reconstruction.balls
    first, second = balls[balls["innings"] == 1].groupby("match_id"), balls[balls["innings"] == 2].groupby("match_id")
    assert first["runs"].sum().to_dict() == {card.match_id: b for card in cards}
    assert second["runs"].sum().to_dict() == {card.match_id: card.second_total for card in cards}
    assert second["runs"].last().to_dict() == {card.match_id: card.final_ball_runs for card in cards}
    assert second["is_wicket"].sum().to_dict() == {card.match_id: card.second_wickets for card in cards}
    first_over = balls[(balls["innings"] == 1) & (balls["overid"] == 1)].groupby("match_id").size()
    assert first_over.to_dict() == {card.match_id: 5 if card.short_first_over else 6 for card in cards}


def test_each_runs_check_returns_the_hand_computed_miss_and_passes_exactly_at_a_miss_of_0(
    cards: list[Card], reconstruction: Reconstruction
) -> None:
    rows = _check_rows(reconstruction.checks)
    found = {card.label: rows[card.match_id, card.runs_check] for card in cards}
    # (applies, passed, runs_check_miss): the check applies, passes exactly when the miss is 0, and misses by card.miss.
    assert found == {card.label: (True, card.miss == 0, card.miss) for card in cards}


def test_criterion_24_keeps_a_miss_of_d20_and_excludes_d20_plus_1_and_over_or_wickets_margin_failures(
    cards: list[Card], reconstruction: Reconstruction, d20: int
) -> None:
    checks = reconstruction.checks
    # Criterion 23 lists every exact failure, so a match kept at D20 still appears for its runs check.
    failing = failing_matches(checks)
    assert list(zip(failing["match_id"].tolist(), failing["check"].tolist())) == [
        (card.match_id, check) for card in cards for check in card.failed
    ]

    conditions = exclusion_conditions(checks, d20)
    found = [
        (int(match_id), str(check), None if pd.isna(miss) else int(miss))
        for match_id, check, miss in zip(*(conditions[name].tolist() for name in conditions.columns))
    ]
    assert found == [(card.match_id, check, miss) for card in cards for check, miss in card.conditions(d20)]

    label_of = {card.match_id: card.label for card in cards}
    excluded = {label_of[match_id] for match_id in conditions["match_id"].tolist()}
    assert excluded == {
        "runs-margin miss D20 + 1",
        "chase shortfall miss D20 + 1",
        "chase excess miss D20 + 1",
        "super-over win tie miss D20 + 1",
        "over check only",
        "wickets-margin check only",
        "runs-margin miss D20 + 1 and over check",
    }
    # A match whose only failure is a miss of exactly D20 stays in model training and player rating.
    at_d20 = [card for card in cards if card.label.endswith("miss D20")]
    assert {card.runs_check for card in at_d20} == set(RUNS_CHECKS)
    assert all(card.miss == d20 and card.only_runs_check_can_fail for card in at_d20)
    assert not {card.label for card in at_d20} & excluded


def test_excluded_flags_follow_the_exclude_failing_matches_option(
    cards: list[Card], reconstruction: Reconstruction, d20: int
) -> None:
    match_id = reconstruction.balls["match_id"]
    conditions = exclusion_conditions(reconstruction.checks, d20)
    excluded_ids = [card.match_id for card in cards if card.conditions(d20)]

    on = excluded_flags(match_id, conditions, exclude_failing_matches=True)
    assert on.tolist() == match_id.isin(excluded_ids).tolist()  # every Delivery of an excluded match, no other

    off = excluded_flags(match_id, conditions, exclude_failing_matches=False)
    assert len(off) == len(match_id) and not off.any()


# ---------------------------------------------------------------- companion test (criterion 27)

class Written(NamedTuple):
    folder: Path  # the audit folder: tmp_path
    text: str  # the audit.md text that write_reconstruction_audit returned
    grid: list[int]  # the Config reporting grid


@pytest.fixture
def written(tmp_path: Path, reconstruction: Reconstruction, d20: int, settings: Settings) -> Written:
    """The audit outputs of the synthetic dataset, written to tmp_path with exclude_failing_matches on."""
    grid = settings.get("data_rules.score_tolerance_grid")
    checks = reconstruction.checks
    rates = pass_rate_table(reconstruction.rates, reconstruction.selected, within_tolerance_pass_rates(checks, d20))
    conditions = exclusion_conditions(checks, d20)
    counts = exclusion_counts(checks, grid, tolerance=d20)
    text = write_reconstruction_audit(tmp_path, AUDITOR_TEXT, rates, conditions, counts, d20, True)
    return Written(tmp_path, text, grid)


def test_excluded_matches_csv_has_one_row_per_condition_with_the_miss_blank_for_over_and_wickets_margin(
    written: Written, cards: list[Card], d20: int
) -> None:
    table = _read_text_csv(written.folder / EXCLUDED_MATCHES_FILE)
    assert table.columns.tolist() == ["match_id", "check", "runs_check_miss"]
    assert table.to_numpy().tolist() == [
        [str(card.match_id), check, "" if miss is None else str(miss)]
        for card in cards
        for check, miss in card.conditions(d20)
    ]


def test_exclusion_counts_csv_has_one_row_per_grid_tolerance_flagging_d20(
    written: Written, cards: list[Card], d20: int
) -> None:
    table = _read_text_csv(written.folder / EXCLUSION_COUNTS_FILE)
    assert table.columns.tolist() == ["score_tolerance", "n_excluded_matches", "is_config_value"]
    assert table.to_numpy().tolist() == [
        [str(tolerance), str(n_excluded(cards, tolerance)), str(tolerance == d20)] for tolerance in written.grid
    ]


def test_audit_md_section_states_d20_the_excluded_count_and_the_count_at_each_grid_tolerance(
    written: Written, cards: list[Card], d20: int
) -> None:
    text = (written.folder / AUDIT_MD).read_text(encoding="utf-8")
    assert text == written.text
    assert text.startswith(AUDITOR_TEXT) and text.count(RECONSTRUCTION_HEADING) == 1
    section = text.split(RECONSTRUCTION_HEADING, 1)[1]

    stated_d20 = re.findall(r"^- Score_Tolerance \(D20, read from the Config\): (\d+) runs?\.$", section, re.MULTILINE)
    assert stated_d20 == [str(d20)]
    stated_excluded = re.findall(
        r"^- Matches excluded from model training and player rating: (\d+)\.", section, re.MULTILINE
    )
    assert stated_excluded == [str(n_excluded(cards, d20))]  # distinct matches: 1014 meets two conditions
    grid_rows = re.findall(r"^\| (\d+)( \(the Config value\))? \| (\d+) \|$", section, re.MULTILINE)
    assert [(int(tolerance), bool(flag), int(count)) for tolerance, flag, count in grid_rows] == [
        (tolerance, tolerance == d20, n_excluded(cards, tolerance)) for tolerance in written.grid
    ]
