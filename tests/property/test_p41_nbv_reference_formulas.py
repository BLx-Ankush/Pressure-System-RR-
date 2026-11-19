# Feature: pressure-rating-system, Property 41: NBV equals the reference formulas
"""Property 41: NBV equals the reference formulas (Requirement 15, criteria 1-14 and 16).

**Validates: Requirements 15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 15.7, 15.8, 15.9, 15.10, 15.11, 15.12, 15.13,
15.14, 15.16**

Property 41 (design): for any synthetic Delivery (any class, Free_Hit flag, dismissal code, dismissed-player
identity including blank or unmatched IDs) with fixed Omega, M4 variant probabilities, Extras_Model
probabilities, Class_Run_Values, cell means and Free_Hit_Premium, the batter NBV, bowler NBV, Extras_Cost and
Dismissal_Charges equal a direct transcription of Requirement 15, criteria 1-14, within 1e-9 and are finite.

Deliveries (``cases``)
    Each example draws 1-3 matches, each in one of 2-5 folds, with 0-10 Deliveries in each of innings 1 and 2
    (at least one in all), numbered by ``row`` in the Requirement 5 order and passed to the code in that order
    or shuffled. A Delivery is a Legal_Ball with ``extras_type`` None (0-6 bat runs), ``byes`` or ``legbyes``
    (no bat run, or 1 on a leg-bye as in the delivered data) or ``Penalty``; a Wide; or a No_Ball (``other``
    or ``Overstep``, 0-6 bat runs); each with the delivered ``extras_2_runs`` values. Team runs follow the
    selected Extras_Encoding, ``byes_one`` (selected as delivered) or ``byes_one_nb_plain`` (``reconstruct``
    module docstring), and the Free_Hit flags follow the glossary. About half the Deliveries, of any class,
    carry a dismissal code: 1-6, 9 (Retire Out), 10 (obstructing the field: the delivered Counted_Wicket that
    is neither a Bowler_Wicket nor a Run_Out, design finding F-7) or 7 or 8 (Retired_Hurt codes).
    ``dismissed_player_id`` names the striker, the non-striker, nobody (blank) or another player, and one
    ``non_striker_id`` in 10 is blank. Each Delivery is Live or Dead; a Live Batter_Delivery may be a
    batter-Role Pressure_Ball, a Dead one never is, and a Wide has no batter-Role flag (<NA>).

    The numbers come from a numpy generator seeded by the example: Omega in [0, 40), exactly 0 on about 15% of
    the Deliveries; the four M4 variants, independent Dirichlet rows (about 10% one-hot and 10% with no
    Bowler_Wicket or Run_Out mass); the Extras_Model (p_W, p_N) from a Dirichlet over Wide, No_Ball and
    neither, about 10% (0, 0) and 10% summing to 1; and the Class_Run_Values 0, 1, 2, 3, 4 and 6 with random
    other_runs, Bowler_Wicket and Run_Out values, one set for every Delivery (shape (9,)) or one per fold
    (passed as (n, 9), each Delivery's own fold's). The cell tables are drawn directly (random values for
    every fold and cell, rows shuffled, in the columns of EXTRAS_CELL_MEANS_COLUMNS and
    FREE_HIT_PREMIUM_COLUMNS) or built by ``extras_cell_means`` and ``free_hit_premium`` from the Deliveries
    plus a Wide, a No_Ball, a Free_Hit Legal_Ball and a non-Free_Hit Legal_Ball in each fold, so no pooled
    set is empty. The cell code sits in ``cell``, or in ``cell_d8`` (passed through ``cells``) beside a decoy
    ``cell`` column that holds another cell of the same innings.

The code's inputs
    The wicket flags are made as the ``reconstruct`` stage makes them, by ``reconstruct.wicket_flags`` (with
    the Counted_Wicket codes 1-6, 9 and 10 and the Config's D14 codes) and
    ``reconstruct.dismissed_batter_flags``. ``retire_out_penalty`` is drawn: False is the Config default
    (criterion 12); True is the D14 Robustness_Case of Requirement 21, criterion 4, where a Retire Out is
    charged as a batter dismissal and still gives the bowler no credit. The M4 variants' opponent theta is not
    an input here; the ``value`` stage checks it.

The reference (``_dismissal``, ``_batter``, ``_bowler``, ``_charges``)
    One Delivery at a time, in Python floats, from the raw values (class, bat runs, ``extras_2_runs``, code
    and ids), never from the flags the code reads. E(p) is math.fsum of p x the Delivery's Class_Run_Values.

    * Dismissed batter (criteria 9, 10 and 12): a code of 1-6, 9 or 10 is a Counted_Wicket, and it charges a
      batter unless it is a Retire Out with the penalty off. The charged batter is the striker when
      ``dismissed_player_id`` equals ``striker_id``, else the non-striker when it equals ``non_striker_id``,
      else the striker, as a fallback that criterion 10 counts.
    * Batter NBV, on Legal_Balls and No_Balls (criteria 1-3): p is ``batter_view`` on a Legal_Ball and
      ``batter_view_fh`` on a No_Ball, EV = E(p) - Omega x (p[W] + p[R]), D = 1 when the charged batter is
      the striker, and NBV = bat runs - Omega x D - EV. A Wide has neither (NaN, D false).
    * Bowler NBV, on every Delivery (criteria 4-8, 13 and 14): conceded is the bat runs of a Legal_Ball and 0
      otherwise; B = 1 for a code of 1, 2, 3, 5 or 6 on any class; Extras_Cost is 0 on a Legal_Ball,
      1 + ``extras_2_runs`` (the Wide's team runs under every encoding) + E(p_wv) on a Wide, and the penalty
      runs (1 + ``extras_2_runs``, or 1 under byes_one_nb_plain) + bat runs + E(p_wv_nofh) + Free_Hit_Premium
      on a No_Ball. The expected costs (criterion 6) put the Delivery's (fold, cell) row of the tables given
      to the code in place of the actual runs: ``wide_runs`` + E(p_wv) and ``nb_penalty_runs`` +
      ``nb_bat_runs`` + E(p_wv_nofh) + ``free_hit_premium``. Then EV_bowl = (1 - p_W - p_N) x (Omega x
      p_wv[W] - E(p_wv)) - p_W x E[Wide cost] - p_N x E[No_Ball cost] and NBV = -conceded + Omega x B -
      Extras_Cost - EV_bowl.
    * Dismissal_Charges (criterion 11): -Omega for a charged non-striker on any Delivery and for a charged
      striker on a Wide, in the row order of the input. The pressure class is the Delivery's batter-Role flag;
      a Wide takes the flag of the latest earlier Batter_Delivery of its innings, and is ordinary when it is a
      Dead_Ball or has no earlier Batter_Delivery (the "Wide rule" of the ``superr.value`` module docstring).

The checks
    Every field of BatterValue and BowlerValue and every row of the dismissal_charges table equals the
    reference: numbers within 1e-9, flags, ids and rows exactly, NaN batter values on Wides. The flags of
    ``dismissed_batters`` equal the reference's, so their fallback count is the number of Deliveries that fall
    back to the striker (criterion 10). Every batter NBV of a Batter_Delivery, every bowler NBV and every
    charge is finite (criterion 16). Two explicit examples hold every rule at least once, with the penalty off
    and on.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import pandas as pd
import pytest
from hypothesis import event, example, given, note
from hypothesis import strategies as st

from superr import reconstruct
from superr.folds import FOLD_COLUMN
from superr.models.outcome import CLASSES
from superr.reconstruct import DeliveryClass
from superr.value import (
    CELLS,
    DISMISSAL_CHARGE_COLUMNS,
    EXTRAS_CELL_MEANS_COLUMNS,
    FREE_HIT_PREMIUM_COLUMNS,
    BowlerValue,
    batter_nbv,
    bowler_nbv,
    dismissal_charges,
    dismissed_batters,
    extras_cell_means,
    free_hit_premium,
)

if TYPE_CHECKING:
    from superr.settings import Settings

# ------------------------------------------------------------------------------------ constants

TOL = 1e-9  # Property 41: the largest |code - reference|, in runs

LEGAL, WIDE, NO_BALL = int(DeliveryClass.LEGAL), int(DeliveryClass.WIDE), int(DeliveryClass.NO_BALL)

#: Positions in every M4 probability row and Class_Run_Value vector (``outcome.CLASSES`` order).
W = CLASSES.index("bowler_wicket")  # the Bowler_Wicket class
R = CLASSES.index("run_out")  # the Run_Out class
OTHER_RUNS = CLASSES.index("other_runs")
#: The Class_Run_Values that the glossary fixes: 0, 1, 2, 3, 4 and 6 for the dot and run classes.
FIXED_RUN_VALUES = {"dot": 0.0, "r1": 1.0, "r2": 2.0, "r3": 3.0, "r4": 4.0, "r6": 6.0}

#: Counted_Wicket codes: 1-6 and 9 (glossary), and 10, obstructing the field, which the Auditor classifies as one
#: in the delivered data. 7 and 8 stand for Retired_Hurt codes, which are not Counted_Wickets.
COUNTED_CODES: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 9, 10)
RETIRED_HURT_CODES: tuple[int, ...] = (7, 8)
BOWLER_WICKET_CODES = frozenset({1, 2, 3, 5, 6})  # criterion 8: bowled, caught, LBW, stumped, hit wicket
RETIRE_OUT = 9  # criterion 12

#: The M4 variants that the NBV reads (``outcome.VARIANTS`` names).
VIEWS: tuple[str, ...] = ("batter_view", "batter_view_fh", "bowler_view", "bowler_view_nofh")


class Kind(NamedTuple):
    """A kind of Delivery: its DeliveryClass and the delivered bat runs and ``extras_2_runs`` it draws from."""

    cls: int
    bat: tuple[int, ...]
    extra: tuple[int, ...]


#: Kinds by their delivered ``extras_type`` text ("None" is the text of a Delivery without an extra).
PLAIN, BYES, LEG_BYES, PENALTY = "None", "byes", "legbyes", "Penalty"
WIDE_TYPE, NO_BALL_TYPE, OVERSTEP = "wide", "other", "Overstep"
KINDS: dict[str, Kind] = {
    PLAIN: Kind(LEGAL, (0, 1, 2, 3, 4, 6, 5), (0,)),
    BYES: Kind(LEGAL, (0,), (0, 1, 3)),
    LEG_BYES: Kind(LEGAL, (0, 1), (0, 1, 3)),
    PENALTY: Kind(LEGAL, (0, 2), (0,)),
    WIDE_TYPE: Kind(WIDE, (0,), (0, 1, 4)),
    NO_BALL_TYPE: Kind(NO_BALL, (0, 1, 4, 6), (0, 1, 4)),
    OVERSTEP: Kind(NO_BALL, (0, 2), (0, 4)),
}
KIND_CHOICES: tuple[str, ...] = (PLAIN,) * 4 + (BYES, LEG_BYES, PENALTY, WIDE_TYPE, WIDE_TYPE, NO_BALL_TYPE, OVERSTEP)
CODE_CHOICES: tuple[int | None, ...] = (None,) * 9 + COUNTED_CODES + RETIRED_HURT_CODES

#: Whose id ``dismissed_player_id`` holds.
STRIKER, NON_STRIKER, BLANK, OTHER_PLAYER = "striker", "non_striker", "blank", "other"
OUTS: tuple[str, ...] = (STRIKER, NON_STRIKER, BLANK, OTHER_PLAYER)
NO_NON_STRIKER_CHOICES: tuple[bool, ...] = (False,) * 9 + (True,)

#: The cell column: the ``cell`` of ``state``, or a name standing for the D8 robustness cells (``cells`` argument).
CELL_NAMES: tuple[str, ...] = ("cell", "cell_d8")
DECOY = "cell"  # the column that code ignoring ``cells`` would read
N_PHASES = 3
MAX_BALLS = 10  # Deliveries per innings
MATCH_ID_BASE = 40_000

#: Columns of a Delivery frame that extras_cell_means and free_hit_premium read, besides the cell columns.
TRAINING_COLUMNS: tuple[str, ...] = (
    FOLD_COLUMN, "innings", "is_super_over", "excluded", "cls", "runs", "team_runs", "free_hit",
)  # fmt: skip
#: The training rows each fold gets besides its Deliveries in a built cell table: (kind, free_hit).
PADDING: tuple[tuple[str, bool], ...] = ((WIDE_TYPE, False), (NO_BALL_TYPE, False), (PLAIN, True), (PLAIN, False))

NOTE_COLUMNS: tuple[str, ...] = (
    "row", "match_id", "innings", "innings_ball_num", FOLD_COLUMN, "extras_type", "cls", "runs", "extras_2_runs",
    "team_runs", "free_hit", "dismissal_type_id", "striker_id", "non_striker_id", "dismissed_player_id",
    "counted_wicket", "bowler_wicket", "retire_out", "dismissed_is_striker", "dismissed_fallback", "live",
    "pressure_bat",
)  # fmt: skip


class D14(NamedTuple):
    """The Config's D14 codes, which the ``reconstruct`` stage turns into the wicket flags."""

    bowler_codes: tuple[int, ...]
    run_out: int
    retire_out: int


@pytest.fixture(scope="module")
def d14(settings: Settings) -> D14:
    return D14(
        tuple(settings.get("design_choices.D14_bowler_wicket_codes")),
        settings.get("design_choices.D14_run_out_code"),
        settings.get("design_choices.D14_retire_out_code"),
    )


# ------------------------------------------------------------------------------- the drawn cases

class Ball(NamedTuple):
    """One drawn Delivery (module docstring, "Deliveries")."""

    kind: str = PLAIN  # its extras_type, a key of KINDS
    bat: int = 0  # bat runs (``runs``)
    extra: int = 0  # extras_2_runs
    code: int | None = None  # dismissal_type_id; None: no wicket
    out: str = BLANK  # whose id dismissed_player_id holds, one of OUTS
    live: bool = True
    pressure: bool = False  # the batter-Role Pressure_Ball flag, if the Delivery is a Live Batter_Delivery
    phase: int = 0  # 0, 1 or 2: powerplay, middle or death
    no_non_striker: bool = False  # non_striker_id blank


class Innings(NamedTuple):
    match: int  # 1, 2 or 3
    number: int  # the innings, 1 or 2
    balls: tuple[Ball, ...]


class Case(NamedTuple):
    n_folds: int
    match_folds: tuple[int, ...]  # the fold of each match
    innings: tuple[Innings, ...]  # in (match, innings) order
    seed: int  # seeds the numbers (module docstring)
    retire_out_penalty: bool
    nb_plain: bool  # the selected encoding is byes_one_nb_plain (a No_Ball scores 1 + b), else byes_one
    crv_per_fold: bool  # Class_Run_Values per fold, shape (n, 9); else one set, shape (9,)
    built_tables: bool  # cell tables built by extras_cell_means and free_hit_premium; else drawn directly
    cells: str  # the cell column, one of CELL_NAMES
    shuffle: bool  # the rows go to the code in a drawn order; else in the Requirement 5 order


@st.composite
def balls(draw: Any) -> Ball:
    kind = draw(st.sampled_from(KIND_CHOICES))
    code = draw(st.sampled_from(CODE_CHOICES))
    return Ball(
        kind=kind,
        bat=draw(st.sampled_from(KINDS[kind].bat)),
        extra=draw(st.sampled_from(KINDS[kind].extra)),
        code=code,
        out=BLANK if code is None else draw(st.sampled_from(OUTS)),
        live=draw(st.booleans()),
        pressure=draw(st.booleans()),
        phase=draw(st.integers(0, N_PHASES - 1)),
        no_non_striker=draw(st.sampled_from(NO_NON_STRIKER_CHOICES)),
    )


@st.composite
def cases(draw: Any) -> Case:
    n_folds = draw(st.integers(2, 5))
    n_matches = draw(st.integers(1, 3))
    match_folds = tuple(draw(st.integers(1, n_folds)) for _ in range(n_matches))
    innings = []
    for match in range(1, n_matches + 1):
        for number in (1, 2):
            least = 1 if (match, number) == (1, 1) else 0
            innings.append(Innings(match, number, tuple(draw(st.lists(balls(), min_size=least, max_size=MAX_BALLS)))))
    return Case(
        n_folds=n_folds,
        match_folds=match_folds,
        innings=tuple(innings),
        seed=draw(st.integers(0, 2**32 - 1)),
        retire_out_penalty=draw(st.booleans()),
        nb_plain=draw(st.booleans()),
        crv_per_fold=draw(st.booleans()),
        built_tables=draw(st.booleans()),
        cells=draw(st.sampled_from(CELL_NAMES)),
        shuffle=draw(st.booleans()),
    )


#: Every rule of criteria 1-14 at least once; each comment names the rule its Delivery exercises.
EVERY_RULE = Case(
    n_folds=3,
    match_folds=(1, 2),
    innings=(
        Innings(1, 1, (
            Ball(WIDE_TYPE, extra=1, code=5, out=STRIKER),  # stumped off a Wide, no earlier Batter_Delivery: ordinary
            Ball(PLAIN, bat=4, pressure=True),  # a Pressure_Ball
            Ball(WIDE_TYPE, code=4, out=STRIKER),  # striker run out off a Live Wide: the previous flag (pressure)
            Ball(WIDE_TYPE, code=5, out=STRIKER, live=False),  # stumped off a Dead Wide: ordinary
            Ball(BYES, extra=3, code=4, out=NON_STRIKER, pressure=True, phase=1),  # non-striker run out on a bye
            Ball(LEG_BYES, bat=1, extra=1, code=2, out=STRIKER, phase=1),  # caught on a leg-bye with a bat run
            Ball(NO_BALL_TYPE, bat=4, extra=1, code=4, out=STRIKER, pressure=True, phase=2),  # run out on a No_Ball
            Ball(WIDE_TYPE, extra=4, phase=2),  # a Free_Hit Wide
            Ball(PLAIN, bat=6, phase=2),  # the Free_Hit Legal_Ball
            Ball(PLAIN, code=9, out=STRIKER, pressure=True, phase=2),  # Retire Out of the striker
        )),
        Innings(1, 2, (
            Ball(WIDE_TYPE, code=4, out=NON_STRIKER),  # first ball of the innings: innings 1's flags do not count
            Ball(PLAIN, code=1, out=BLANK),  # bowled, blank dismissed_player_id: striker fallback
            Ball(PLAIN, bat=1, code=3, out=OTHER_PLAYER, pressure=True, phase=1),  # LBW naming another player
            Ball(OVERSTEP, code=7, out=STRIKER, phase=1),  # a Retired_Hurt code: no Counted_Wicket
            Ball(PLAIN, code=6, out=NON_STRIKER, phase=1),  # hit wicket naming the non-striker: credit and a charge
            Ball(PLAIN, code=10, out=NON_STRIKER, live=False, phase=2),  # obstructing the field: no credit
            Ball(PLAIN, code=9, out=NON_STRIKER, phase=2),  # Retire Out of the non-striker
            Ball(PLAIN, code=9, out=BLANK, phase=2),  # Retire Out naming nobody
            Ball(PENALTY, bat=2, phase=2),  # a Penalty Legal_Ball
            Ball(PLAIN, code=2, out=NON_STRIKER, no_non_striker=True, phase=2),  # blank non_striker_id: fallback
        )),
        Innings(2, 1, (
            Ball(NO_BALL_TYPE, code=2, out=STRIKER, pressure=True),  # caught on a No_Ball: credit and D
            Ball(WIDE_TYPE, code=5, out=STRIKER),  # stumped off a Free_Hit Wide: the No_Ball's flag
            Ball(PLAIN, code=8, out=NON_STRIKER),  # a Retired_Hurt code naming the non-striker
        )),
        Innings(2, 2, ()),
    ),
    seed=41,
    retire_out_penalty=False,
    nb_plain=False,
    crv_per_fold=True,
    built_tables=False,
    cells="cell",
    shuffle=False,
)  # fmt: skip
#: The same Deliveries in the D14 Robustness_Case, with the other choice of every input layout.
EVERY_RULE_D14_CASE = EVERY_RULE._replace(
    retire_out_penalty=True, nb_plain=True, crv_per_fold=False, built_tables=True, cells="cell_d8", shuffle=True
)


# ---------------------------------------------------------------------------- the code's inputs

class Delivery(NamedTuple):
    """One Delivery as the reference reads it: raw values only."""

    row: int
    match_id: int
    innings: int
    ball: int  # innings_ball_num
    fold: int
    cell: int  # 3 x (innings - 1) + Phase
    kind: str
    cls: int
    bat: int
    extra: int
    team: int  # team runs under the selected Extras_Encoding
    free_hit: bool
    code: int | None
    striker: int
    non_striker: int | None
    dismissed: int | None
    live: bool
    pressure: bool | None  # the batter-Role Pressure_Ball flag; None on a Wide


def _team_runs(kind: str, bat: int, extra: int, nb_plain: bool) -> int:
    """Team runs under byes_one or byes_one_nb_plain (the encoding table of the ``reconstruct`` docstring)."""
    cls = KINDS[kind].cls
    if cls == WIDE:
        return 1 + extra
    if cls == NO_BALL:
        return 1 + bat + (0 if nb_plain else extra)
    if kind in (BYES, LEG_BYES):
        return 1 + extra
    return bat + extra


def _decoy(cell: int) -> int:
    """Another cell of the same innings: the next Phase, the death overs wrapping to the powerplay."""
    return N_PHASES * (cell // N_PHASES) + (cell % N_PHASES + 1) % N_PHASES


def _deliveries(case: Case) -> list[Delivery]:
    """The Deliveries of ``case`` in the Requirement 5 order (match, innings, innings_ball_num), numbered by row."""
    deliveries: list[Delivery] = []
    for innings in case.innings:
        base = 1000 * innings.match + 100 * innings.number  # the player ids of the innings
        free_hit = False  # glossary: from a No_Ball up to and including the next Legal_Ball
        for k, ball in enumerate(innings.balls):
            cls = KINDS[ball.kind].cls
            striker = base + 1 + k % 2
            non_striker = None if ball.no_non_striker else base + 2 - k % 2
            named = {STRIKER: striker, NON_STRIKER: non_striker, BLANK: None, OTHER_PLAYER: base + 50}[ball.out]
            deliveries.append(
                Delivery(
                    row=len(deliveries),
                    match_id=MATCH_ID_BASE + innings.match,
                    innings=innings.number,
                    ball=k + 1,
                    fold=case.match_folds[innings.match - 1],
                    cell=N_PHASES * (innings.number - 1) + ball.phase,
                    kind=ball.kind,
                    cls=cls,
                    bat=ball.bat,
                    extra=ball.extra,
                    team=_team_runs(ball.kind, ball.bat, ball.extra, case.nb_plain),
                    free_hit=free_hit,
                    code=ball.code,
                    striker=striker,
                    non_striker=non_striker,
                    dismissed=None if ball.code is None else named,
                    live=ball.live,
                    pressure=None if cls == WIDE else ball.live and ball.pressure,
                )
            )
            free_hit = cls == NO_BALL or (free_hit and cls == WIDE)
    return deliveries


def _frame(deliveries: Sequence[Delivery], cells: str, d14: D14) -> pd.DataFrame:
    """The Delivery frame with the ``balls`` and ``pressure`` columns that the Value_Engine reads; the wicket
    flags are made as the ``reconstruct`` stage makes them."""

    def values(name: str) -> list[Any]:
        return [getattr(x, name) for x in deliveries]

    n = len(deliveries)
    raw = pd.DataFrame(
        {
            "row": np.array(values("row"), dtype=np.int32),
            "match_id": np.array(values("match_id"), dtype=np.int32),
            "innings": np.array(values("innings"), dtype=np.int32),
            "innings_ball_num": np.array(values("ball"), dtype=np.int32),
            "striker_id": np.array(values("striker"), dtype=np.int32),
            "non_striker_id": pd.array(values("non_striker"), dtype="Int32"),
            "dismissed_player_id": pd.array(values("dismissed"), dtype="Int32"),
            "extras_type": pd.array(values("kind"), dtype="str"),
            "runs": np.array(values("bat"), dtype=np.int32),
            "extras_2_runs": np.array(values("extra"), dtype=np.int32),
            "is_wicket": np.array([x.code is not None for x in deliveries], dtype=bool),
            "dismissal_type_id": pd.array(values("code"), dtype="Int32"),
            "cls": np.array(values("cls"), dtype=np.int8),
            "is_super_over": np.zeros(n, dtype=bool),
            "free_hit": np.array(values("free_hit"), dtype=bool),
            "team_runs": np.array(values("team"), dtype=np.int32),
            "excluded": np.zeros(n, dtype=bool),
            FOLD_COLUMN: np.array(values("fold"), dtype=np.int32),
            "live": np.array(values("live"), dtype=bool),
            "pressure_bat": pd.array(values("pressure"), dtype="boolean"),
        }
    )
    if cells != DECOY:
        raw[DECOY] = np.array([_decoy(x.cell) for x in deliveries], dtype=np.int8)
    raw[cells] = np.array(values("cell"), dtype=np.int8)
    wickets = reconstruct.wicket_flags(raw, COUNTED_CODES, d14.bowler_codes, d14.run_out, d14.retire_out)
    return pd.concat([raw, wickets, reconstruct.dismissed_batter_flags(raw, wickets)], axis=1)


def _m4_rows(rng: np.random.Generator, n: int) -> np.ndarray:
    """n M4 probability rows: Dirichlet, about 10% one-hot and 10% with no Bowler_Wicket or Run_Out mass."""
    k = len(CLASSES)
    p = rng.dirichlet(np.full(k, rng.choice((0.3, 1.0, 5.0))), size=n)
    pick = rng.random(n)
    one_hot = pick < 0.1
    p[one_hot] = np.eye(k)[rng.integers(0, k, int(one_hot.sum()))]
    no_out = (pick >= 0.1) & (pick < 0.2)
    p[no_out, W] = 0.0
    p[no_out, R] = 0.0
    total = p[no_out].sum(axis=1, keepdims=True)
    p[no_out] = np.where(total > 0.0, p[no_out] / np.where(total > 0.0, total, 1.0), np.eye(k)[0])
    return p


def _extras_rows(rng: np.random.Generator, n: int) -> np.ndarray:
    """n Extras_Model (p_W, p_N) rows with p_W + p_N <= 1: about 10% (0, 0) and 10% summing to 1."""
    p = rng.dirichlet((0.6, 0.3, 6.0), size=n)[:, :2].copy()
    pick = rng.random(n)
    p[pick < 0.1] = 0.0
    full = (pick >= 0.1) & (pick < 0.2)
    share = rng.random(int(full.sum()))
    p[full, 0] = share
    p[full, 1] = 1.0 - share
    return p


def _class_run_values(rng: np.random.Generator) -> np.ndarray:
    """0, 1, 2, 3, 4 and 6 for the dot and run classes; other_runs 5 (as delivered) or in [0, 8); Bowler_Wicket
    0 (as delivered) or in [0, 1); Run_Out in [0, 2)."""
    values = np.array([FIXED_RUN_VALUES.get(name, 0.0) for name in CLASSES])
    values[OTHER_RUNS] = 5.0 if rng.random() < 0.3 else rng.uniform(0.0, 8.0)
    values[W] = 0.0 if rng.random() < 0.3 else rng.uniform(0.0, 1.0)
    values[R] = rng.uniform(0.0, 2.0)
    return values


def _drawn_tables(rng: np.random.Generator, n_folds: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random cell tables in the format of extras_cell_means and free_hit_premium: every fold and cell, shuffled."""
    fold = np.repeat(np.arange(1, n_folds + 1), len(CELLS)).astype(np.int32)
    cell = np.tile(np.asarray(CELLS), n_folds).astype(np.int8)
    size = len(fold)
    n_wide, n_nb, n_fh, n_non_fh = (rng.integers(0, 4, size) for _ in range(4))
    means = pd.DataFrame(
        {
            FOLD_COLUMN: fold,
            "cell": cell,
            "wide_runs": 1.0 + rng.uniform(0.0, 2.0, size),
            "nb_penalty_runs": 1.0 + rng.uniform(0.0, 1.5, size),
            "nb_bat_runs": rng.uniform(0.0, 3.0, size),
            "n_wide": n_wide,
            "n_nb": n_nb,
            "pooled_wide_runs": n_wide == 0,
            "pooled_nb_penalty_runs": n_nb == 0,
            "pooled_nb_bat_runs": n_nb == 0,
        }
    )[list(EXTRAS_CELL_MEANS_COLUMNS)]
    premium = pd.DataFrame(
        {
            FOLD_COLUMN: fold,
            "cell": cell,
            "free_hit_premium": rng.uniform(-2.0, 4.0, size),
            "n_fh_legal": n_fh,
            "n_non_fh_legal": n_non_fh,
            "pooled_free_hit_premium": (n_fh == 0) | (n_non_fh == 0),
        }
    )[list(FREE_HIT_PREMIUM_COLUMNS)]
    return (
        means.iloc[rng.permutation(size)].reset_index(drop=True),
        premium.iloc[rng.permutation(size)].reset_index(drop=True),
    )


def _built_tables(rng: np.random.Generator, d: pd.DataFrame, case: Case) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The cell tables of extras_cell_means and free_hit_premium, from ``d`` plus PADDING rows in each fold."""
    columns = list(dict.fromkeys((*TRAINING_COLUMNS, DECOY, case.cells)))
    records = []
    for fold in range(1, case.n_folds + 1):
        for kind, free_hit in PADDING:
            number = int(rng.integers(1, 3))
            cell = N_PHASES * (number - 1) + int(rng.integers(0, N_PHASES))
            bat, extra = int(rng.choice(KINDS[kind].bat)), int(rng.choice(KINDS[kind].extra))
            record = {
                FOLD_COLUMN: fold, "innings": number, "is_super_over": False, "excluded": False,
                "cls": KINDS[kind].cls, "runs": bat, "team_runs": _team_runs(kind, bat, extra, case.nb_plain),
                "free_hit": free_hit, DECOY: _decoy(cell),
            }  # fmt: skip
            record[case.cells] = cell  # replaces the decoy when the cell column is "cell"
            records.append(record)
    padding = pd.DataFrame.from_records(records, columns=columns).astype(d[columns].dtypes.to_dict())
    training = pd.concat([d[columns], padding], ignore_index=True)
    return extras_cell_means(training, case.cells), free_hit_premium(training, case.cells)


class Inputs(NamedTuple):
    """The arguments of the code, in the row order of ``d``, and the Deliveries in the same order."""

    deliveries: list[Delivery]
    d: pd.DataFrame
    omega: np.ndarray
    views: dict[str, np.ndarray]  # VIEWS -> (n, 9)
    p_extras: np.ndarray  # (n, 2): p_W, p_N
    crv: np.ndarray  # as passed: (9,) or (n, 9)
    crv_rows: np.ndarray  # (n, 9): each Delivery's Class_Run_Values
    cell_means: pd.DataFrame
    fh_premium: pd.DataFrame


def _inputs(case: Case, d14: D14) -> Inputs:
    rng = np.random.default_rng(case.seed)
    ordered = _deliveries(case)
    n = len(ordered)
    omega = np.where(rng.random(n) < 0.15, 0.0, rng.uniform(0.0, 40.0, n))
    views = {name: _m4_rows(rng, n) for name in VIEWS}
    p_extras = _extras_rows(rng, n)
    by_fold = np.stack([_class_run_values(rng) for _ in range(case.n_folds)])  # row k - 1 holds fold k's
    order = rng.permutation(n) if case.shuffle else np.arange(n)
    deliveries = [ordered[i] for i in order]
    folds = np.array([x.fold for x in deliveries], dtype=np.int64)
    crv_rows = by_fold[folds - 1] if case.crv_per_fold else np.tile(by_fold[0], (n, 1))
    d = _frame(deliveries, case.cells, d14)
    means, premium = _built_tables(rng, d, case) if case.built_tables else _drawn_tables(rng, case.n_folds)
    return Inputs(
        deliveries=deliveries,
        d=d,
        omega=omega[order],
        views={name: p[order] for name, p in views.items()},
        p_extras=p_extras[order],
        crv=crv_rows if case.crv_per_fold else by_fold[0],
        crv_rows=crv_rows,
        cell_means=means,
        fh_premium=premium,
    )


# --------------------------------------------------------------------------------- the reference

class Dismissal(NamedTuple):
    charged: bool  # a Counted_Wicket that charges a batter
    striker: bool  # the charged batter is the striker
    fallback: bool  # ... because dismissed_player_id names neither batter (criterion 10)


NOT_CHARGED = Dismissal(False, False, False)


def _dismissal(x: Delivery, penalty: bool) -> Dismissal:
    """The batter a Delivery's dismissal charges (criteria 9, 10 and 12; the D14 Robustness_Case when ``penalty``)."""
    if x.code not in COUNTED_CODES:
        return NOT_CHARGED  # no wicket, or a Retired_Hurt code
    if x.code == RETIRE_OUT and not penalty:
        return NOT_CHARGED  # criterion 12
    if x.dismissed is not None and x.dismissed == x.striker:
        return Dismissal(True, True, False)  # criterion 9
    if x.dismissed is not None and x.dismissed == x.non_striker:
        return Dismissal(True, False, False)  # criterion 9
    return Dismissal(True, True, True)  # criterion 10


def _expected_bat_runs(p: Sequence[float], values: Sequence[float]) -> float:
    """Criterion 2: the sum over M4 classes of the class probability x its Class_Run_Value."""
    return math.fsum(pk * vk for pk, vk in zip(p, values, strict=True))


class BatterRef(NamedTuple):
    nbv: float
    ev: float
    striker_out: bool  # D


def _batter(
    x: Delivery, p_bv: Sequence[float], p_bv_fh: Sequence[float], values: Sequence[float], omega: float, out: Dismissal
) -> BatterRef:
    """Criteria 1-3."""
    if x.cls == WIDE:
        return BatterRef(math.nan, math.nan, False)  # a Wide is not a Batter_Delivery
    p = p_bv if x.cls == LEGAL else p_bv_fh  # criterion 3: a No_Ball's state with the Free_Hit flag set
    ev = _expected_bat_runs(p, values) - omega * (p[W] + p[R])  # criterion 2
    striker_out = out.charged and out.striker
    return BatterRef(x.bat - omega * (1.0 if striker_out else 0.0) - ev, ev, striker_out)  # criterion 1


class CellValues(NamedTuple):
    wide_runs: float
    nb_penalty_runs: float
    nb_bat_runs: float
    free_hit_premium: float


def _cell_values(means: pd.DataFrame, premium: pd.DataFrame) -> dict[tuple[int, int], CellValues]:
    """The four quantities of each (fold, cell) row of the cell tables, read row by row."""
    extras = {
        (int(fold), int(cell)): (float(wide), float(penalty), float(bat))
        for fold, cell, wide, penalty, bat in zip(
            means[FOLD_COLUMN].tolist(), means["cell"].tolist(), means["wide_runs"].tolist(),
            means["nb_penalty_runs"].tolist(), means["nb_bat_runs"].tolist(), strict=True,
        )
    }  # fmt: skip
    premiums = {
        (int(fold), int(cell)): float(value)
        for fold, cell, value in zip(
            premium[FOLD_COLUMN].tolist(), premium["cell"].tolist(), premium["free_hit_premium"].tolist(), strict=True
        )
    }
    return {key: CellValues(*extras[key], premiums[key]) for key in extras}


class BowlerRef(NamedTuple):
    """The reference values, in the order of BowlerValue's fields."""

    nbv: float
    ev: float
    extras_cost: float
    conceded: float
    credit: bool
    exp_wide_cost: float
    exp_nb_cost: float


def _bowler(
    x: Delivery,
    p_wv: Sequence[float],
    p_wv_nofh: Sequence[float],
    p_wide: float,
    p_nb: float,
    values: Sequence[float],
    omega: float,
    cell: CellValues,
    nb_plain: bool,
) -> BowlerRef:
    """Criteria 4-8, 13 and 14."""
    e_view = _expected_bat_runs(p_wv, values)  # a replacement Legal_Ball with the Delivery's own Free_Hit flag
    e_nofh = _expected_bat_runs(p_wv_nofh, values)  # ... with the Free_Hit flag set to false
    conceded = float(x.bat) if x.cls == LEGAL else 0.0  # criterion 7
    credit = x.code in BOWLER_WICKET_CODES  # criterion 8, whatever the Delivery's class
    if x.cls == LEGAL:
        extras_cost = 0.0  # criterion 4
    elif x.cls == WIDE:
        extras_cost = (1 + x.extra) + e_view  # criterion 13: a Wide scores 1 + extras_2_runs under every encoding
    else:
        penalty_runs = 1 if nb_plain else 1 + x.extra  # byes_one_nb_plain, byes_one
        extras_cost = penalty_runs + x.bat + e_nofh + cell.free_hit_premium  # criterion 14
    exp_wide_cost = cell.wide_runs + e_view  # criteria 6 and 13
    exp_nb_cost = cell.nb_penalty_runs + cell.nb_bat_runs + e_nofh + cell.free_hit_premium  # criteria 6 and 14
    ev = (1.0 - p_wide - p_nb) * (omega * p_wv[W] - e_view) - p_wide * exp_wide_cost - p_nb * exp_nb_cost  # 5
    nbv = -conceded + omega * (1.0 if credit else 0.0) - extras_cost - ev  # criterion 4
    return BowlerRef(nbv, ev, extras_cost, conceded, credit, exp_wide_cost, exp_nb_cost)


class ChargeRef(NamedTuple):
    """A row of the dismissal_charges table."""

    row: int
    player_id: int
    charge: float
    pressure: bool
    non_striker: bool
    cls: int
    flag_row: int | None


def _charges(deliveries: Sequence[Delivery], omega: Sequence[float], outs: Sequence[Dismissal]) -> list[ChargeRef]:
    """Criterion 11, with the Wide rule of the ``superr.value`` module docstring; in the given row order."""
    charges: list[ChargeRef] = []
    for x, om, out in zip(deliveries, omega, outs, strict=True):
        if not out.charged or (out.striker and x.cls != WIDE):
            continue  # no charged batter, or the striker's D on the Batter_Delivery
        if x.cls == WIDE:
            earlier = [
                y for y in deliveries
                if (y.match_id, y.innings) == (x.match_id, x.innings) and y.cls != WIDE and y.ball < x.ball
            ]  # fmt: skip
            source = max(earlier, key=lambda y: y.ball) if earlier else None
            pressure = source is not None and bool(source.pressure) and x.live
        else:
            source, pressure = x, bool(x.pressure)
        player = x.striker if out.striker else x.non_striker
        assert player is not None, "reference bug: a charged non-striker without an id"
        flag_row = None if source is None else source.row
        charges.append(ChargeRef(x.row, player, -om, pressure, not out.striker, x.cls, flag_row))
    return charges


# ------------------------------------------------------------------------------------ the checks

def _number_problems(
    problems: list[str], what: str, got: Any, expected: Sequence[float], rows: Sequence[int]
) -> None:
    """A line in ``problems`` unless ``got`` holds float64 values within TOL of ``expected`` (NaN where it is NaN)."""
    values = np.asarray(got)
    if values.dtype != np.float64 or values.shape != (len(expected),):
        problems.append(f"{what}: {values.dtype} values of shape {values.shape}; expected {len(expected)} float64")
        return
    reference = np.asarray(expected, dtype=np.float64)
    same = np.where(np.isnan(reference), np.isnan(values), np.abs(values - reference) <= TOL)
    bad = np.flatnonzero(~same)
    if len(bad):
        i = int(bad[0])
        problems.append(
            f"{what}: {len(bad)} of {len(values)} Deliveries differ from the reference by more than {TOL:g}; "
            f"first row {rows[i]}: {values[i]!r}, expected {reference[i]!r}"
        )


def _flag_problems(problems: list[str], what: str, got: Any, expected: Sequence[bool], rows: Sequence[int]) -> None:
    """A line in ``problems`` unless ``got`` holds bool values equal to ``expected``."""
    values = np.asarray(got)
    if values.dtype != np.bool_ or values.shape != (len(expected),):
        problems.append(f"{what}: {values.dtype} values of shape {values.shape}; expected {len(expected)} bool")
        return
    bad = np.flatnonzero(values != np.asarray(expected, dtype=bool))
    if len(bad):
        i = int(bad[0])
        problems.append(
            f"{what}: {len(bad)} of {len(values)} Deliveries differ from the reference; first row {rows[i]}: "
            f"{bool(values[i])}, expected {bool(expected[i])}"
        )


def _charge_problems(problems: list[str], got: pd.DataFrame, expected: Sequence[ChargeRef]) -> None:
    """Lines in ``problems`` for every way the dismissal_charges table differs from the reference charges."""
    if list(got.columns) != list(DISMISSAL_CHARGE_COLUMNS):
        problems.append(f"dismissal_charges has the columns {list(got.columns)}; expected {DISMISSAL_CHARGE_COLUMNS}")
        return
    charges = [
        ChargeRef(
            int(row), int(player), float(charge), bool(pressure), bool(non_striker), int(cls),
            None if pd.isna(flag_row) else int(flag_row),
        )
        for row, player, charge, pressure, non_striker, cls, flag_row in zip(
            *(got[name].tolist() for name in DISMISSAL_CHARGE_COLUMNS), strict=True
        )
    ]  # fmt: skip
    if [c.row for c in charges] != [c.row for c in expected]:
        problems.append(
            f"dismissal_charges charges the rows {[c.row for c in charges]}; expected {[c.row for c in expected]}, "
            "in the row order of d"
        )
        return
    for mine, theirs in zip(charges, expected, strict=True):
        if mine._replace(charge=0.0) != theirs._replace(charge=0.0) or not abs(mine.charge - theirs.charge) <= TOL:
            problems.append(f"dismissal_charges row {theirs.row}: {mine}; expected {theirs}")


class _Table:
    """A DataFrame for hypothesis.note, formatted only when Hypothesis prints the note."""

    def __init__(self, title: str, frame: pd.DataFrame) -> None:
        self.title, self.frame = title, frame

    def __repr__(self) -> str:
        return f"{self.title}:\n{self.frame.to_string()}"


def _events(
    case: Case, deliveries: Sequence[Delivery], outs: Sequence[Dismissal], charges: Sequence[ChargeRef]
) -> None:
    event(f"retire_out_penalty {case.retire_out_penalty}")
    event("cell tables built by the code" if case.built_tables else "cell tables drawn directly")
    labels = {
        "a non-striker Dismissal_Charge": any(c.non_striker for c in charges),
        "a striker Dismissal_Charge off a Wide": any(not c.non_striker for c in charges),
        "a Wide charge with no earlier Batter_Delivery": any(c.cls == WIDE and c.flag_row is None for c in charges),
        "a striker fallback": any(out.fallback for out in outs),
        "a Retire Out": any(x.code == RETIRE_OUT for x in deliveries),
        "a Retired_Hurt code": any(x.code in RETIRED_HURT_CODES for x in deliveries),
    }
    for label, hit in labels.items():
        if hit:
            event(label)


# -------------------------------------------------------------------------------------- the test

@given(case=cases())
@example(case=EVERY_RULE)
@example(case=EVERY_RULE_D14_CASE)
def test_nbv_equals_the_reference_formulas(case: Case, d14: D14) -> None:
    """The batter NBV, bowler NBV, Extras_Cost, expected costs, dismissed batters and Dismissal_Charges equal a
    row-by-row transcription of Requirement 15, criteria 1-14, within 1e-9, and every NBV is finite (criterion 16).

    **Validates: Requirements 15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 15.7, 15.8, 15.9, 15.10, 15.11, 15.12, 15.13,
    15.14, 15.16**
    """
    inputs = _inputs(case, d14)
    deliveries, d, views = inputs.deliveries, inputs.d, inputs.views
    rows = [x.row for x in deliveries]
    penalty = case.retire_out_penalty
    note(_Table("d, in the row order passed to the code", d[[*NOTE_COLUMNS, *dict.fromkeys((DECOY, case.cells))]]))

    # The reference, one Delivery at a time.
    outs = [_dismissal(x, penalty) for x in deliveries]
    cells = _cell_values(inputs.cell_means, inputs.fh_premium)
    batter_ref: list[BatterRef] = []
    bowler_ref: list[BowlerRef] = []
    for i, x in enumerate(deliveries):
        values = inputs.crv_rows[i].tolist()
        omega = float(inputs.omega[i])
        p_wide, p_nb = inputs.p_extras[i].tolist()
        batter_ref.append(
            _batter(x, views["batter_view"][i].tolist(), views["batter_view_fh"][i].tolist(), values, omega, outs[i])
        )
        bowler_ref.append(
            _bowler(
                x, views["bowler_view"][i].tolist(), views["bowler_view_nofh"][i].tolist(), p_wide, p_nb, values,
                omega, cells[x.fold, x.cell], case.nb_plain,
            )
        )  # fmt: skip
    charge_ref = _charges(deliveries, inputs.omega.tolist(), outs)
    reference = pd.DataFrame(
        {"row": rows, "omega": inputs.omega, "p_wide": inputs.p_extras[:, 0], "p_noball": inputs.p_extras[:, 1]}
        | {f"{name}_bat": [getattr(r, name) for r in batter_ref] for name in BatterRef._fields}
        | {f"{name}_bowl": [getattr(r, name) for r in bowler_ref] for name in BowlerRef._fields}
    )
    note(_Table("reference, in the same row order", reference))
    note(f"reference Dismissal_Charges: {charge_ref}")
    _events(case, deliveries, outs, charge_ref)

    # The code.
    dismissals = dismissed_batters(d, retire_out_penalty=penalty)
    batter = batter_nbv(
        d, views["batter_view"], views["batter_view_fh"], inputs.crv, inputs.omega, retire_out_penalty=penalty
    )
    bowler = bowler_nbv(
        d, views["bowler_view"], views["bowler_view_nofh"], inputs.p_extras, inputs.crv, inputs.omega,
        inputs.cell_means, inputs.fh_premium, cells=case.cells,
    )  # fmt: skip
    charges = dismissal_charges(d, inputs.omega, retire_out_penalty=penalty)

    problems: list[str] = []
    # Criteria 9, 10 and 12: the charged batter of each dismissal, and the striker fallbacks that criterion 10 counts.
    _flag_problems(problems, "dismissed_batters charged", dismissals.charged, [o.charged for o in outs], rows)
    on_striker = [o.charged and o.striker for o in outs]
    _flag_problems(problems, "dismissed_batters striker", dismissals.striker, on_striker, rows)
    _flag_problems(problems, "dismissed_batters fallback", dismissals.fallback, [o.fallback for o in outs], rows)
    # Criteria 1-3.
    _number_problems(problems, "batter NBV", batter.nbv, [r.nbv for r in batter_ref], rows)
    _number_problems(problems, "batter Expected_Value", batter.ev, [r.ev for r in batter_ref], rows)
    _flag_problems(problems, "batter D (striker_out)", batter.striker_out, [r.striker_out for r in batter_ref], rows)
    # Criteria 4-8, 13 and 14.
    for name in BowlerValue._fields:
        expected = [getattr(r, name) for r in bowler_ref]
        if name == "credit":
            _flag_problems(problems, "bowler B (credit)", bowler.credit, expected, rows)
        else:
            _number_problems(problems, f"bowler {name}", getattr(bowler, name), expected, rows)
    # Criteria 9-12.
    _charge_problems(problems, charges, charge_ref)
    # Criterion 16.
    batter_rows = np.array([x.cls != WIDE for x in deliveries], dtype=bool)
    for what, values in (
        ("batter NBVs of Batter_Deliveries", np.asarray(batter.nbv)[batter_rows]),
        ("bowler NBVs", np.asarray(bowler.nbv)),
        ("Dismissal_Charges", charges["charge"].to_numpy(dtype=np.float64)),
    ):
        bad = ~np.isfinite(values)
        if bad.any():
            problems.append(f"criterion 16: {int(bad.sum())} {what} are not finite")
    assert not problems, "\n".join(problems)
