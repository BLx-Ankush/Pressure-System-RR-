# Feature: pressure-rating-system, Property 58: Final list invariants
"""Property 58: Final list invariants (Requirement 25, criteria 1, 5 and 9; design sections
"List_Builder (``lists.py``, ``charts.py``)" and "Output files").

**Validates: Requirements 25.1, 25.5**

For any Main_Ranking with at least 10 Eligible_Players, the written list has exactly 10 rows with
ranks 1-10 in order, 10 distinct ``player_id``s equal to ranking positions 1-10, and IPL,
Full_Member_Match and Associate_Match shares in [0, 1] that sum to 1 within 1e-6.

The examples (``list_cases``)
    Each example is a synthetic Main_Ranking of both Roles, in the tables that the ``final_lists``
    stage reads (``load_list_inputs``):

    * per Role, 10-30 Eligible_Players (exactly 10 in about a fifth of the Roles) and 0-4 players
      who are not Eligible_Players. The Eligible_Players hold ranks 1..n once each, in a drawn
      order, with Final_Scores that never rise with rank and are often tied. The others have no
      rank and no Final_Score (``rank`` <NA>, ``final_score`` NaN), as in ``player_estimates``.
      Some players belong to both Roles (eligible in one, both or neither), and in about half the
      examples one player who is eligible in both Roles is moved into both top 10s;
    * each player's unweighted Pressure_Balls by match class (IPL, Full_Member_Match,
      Associate_Match): all in one class, equal thirds, two classes, small counts (0-7 each, so
      shares such as 1/3, 2/7 or 4/7) or any counts; a player who is not an Eligible_Player may
      have none. The counts become the rows of a ``contributions`` table (each count split over 1-3
      rows, plus Ordinary_Ball rows and, for batters, Dismissal_Charge rows with ``n`` = 0, none of
      which is a Pressure_Ball) and the ranking's ``n_pressure_balls``. ``pressure_shares`` turns
      the table into the shares, as ``load_list_inputs`` does;
    * a ``bootstrap`` row for every Eligible_Player, which copies his ``rank`` and ``final_score``
      into ``main_rank`` and ``main_final_score``, with B from 1 to 2,000 resamples, P_Top10 = k / B,
      a median rank in steps of 0.5 and an interval around the Final_Score. Now and then a player
      is never ranked in a resample, so he has no interval and no median rank (blank cells);
    * a ``mapping`` table that gives each player one Candidate_Name, in 1-3 rows. The names are
      strings of letters, spaces, ``'``, ``-``, ``.``, commas, double quotes and a few non-ASCII
      letters, so the CSV writer must quote some of them. Two players may share a name, as in the
      real mapping file; each still has exactly one Candidate_Name;
    * a seed for the row order of every table and the split of each count into rows.

``test_final_list_invariants``
    Runs ``main_lists`` and ``write_lists`` into a fresh temporary folder (``tmp_path_factory``)
    and reads both files back as text with the ``csv`` module. The expected values come from the
    drawn example, not from the code under test. Checks:

    1. The folder holds exactly ``top10_batters.csv`` and ``top10_bowlers.csv``, each with the
       header of the design (Output files) and exactly 10 rows. ``rank`` reads 1 to 10 in order and
       ``player_id`` holds 10 distinct ids: those at ranks 1 to 10 of the Role's ranking, in order.
    2. In the written text, every share lies in [0, 1] and each row's three shares sum to 1 within
       1e-6, summed exactly (``fractions.Fraction``).
    3. Every cell of a value that is not a whole number is a fixed-point decimal with at least 6
       decimal places.
    4. Each row holds its player's values: his Candidate_Name as an exact string; Final_Score and
       Hit_Rate from the ranking; ``ci_low``, ``ci_high``, P_Top10 and median rank from his
       bootstrap row (blank exactly where that row has none); and each share equal to the class
       count divided by his Pressure_Balls. A written number must be its value rounded to the
       decimal places written (within half a unit in the last place). Read back by
       ``pandas.read_csv`` with ``keep_default_na=False, na_values=[""], dtype={"player_name": str}``,
       the file gives integer ``rank`` and ``player_id`` columns and the same names.
    6. The Roles' lists are built independently (criterion 1): a player in both Roles' top 10 is in
       both files, each row with his values for that Role (check 4), and ``list_table`` given only a
       Role's rows of the ranking, bootstrap and shares tables returns that Role's list exactly.

``test_a_short_role_writes_neither_file``
    5. The same examples with one or both Roles cut to 0-9 Eligible_Players: the players ranked
       below the cut become players who are not Eligible_Players. The folder already holds an
       earlier Run's version of both files. ``main_lists`` raises StageError (exit code 5) whose
       message names each short Role followed by its count, and neither file changes. ``list_table``
       raises the same way for a short Role and still builds the other Role's list unchanged. Given
       the full lists with a short Role's table cut to its first 0-9 rows, ``write_lists`` raises
       ValueError and again changes neither file, so no half-written pair is left behind.

Hypothesis events record Roles with exactly 10 Eligible_Players, tied Final_Scores in a top 10,
each kind of share split among listed players, listed players never ranked in a resample, names
that the CSV quotes and players in both lists. No event is asserted.

Not covered here: the name rules (Property 57), the Comparison_Lists, and how the ranking and the
bootstrap values are computed (Properties 51 and 52).
"""

from __future__ import annotations

import csv
import math
import re
from collections.abc import Mapping
from fractions import Fraction
from itertools import accumulate
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import pytest
from hypothesis import event, given
from hypothesis import strategies as st

from superr.errors import StageError
from superr.lists import candidate_names, eligible_counts, list_table, main_lists, pressure_shares, write_lists
from superr.rating import CompClass
from superr.value import BATTER_ROLE, BOWLER_ROLE, ROLES

#: Requirement 25, criterion 1: a Main_List holds exactly 10 players.
TOP = 10
#: Criterion 5: the three shares of a row sum to 1 within 1e-6, and a value that is not a whole
#: number is written to at least 6 decimal places.
SHARE_SUM_TOLERANCE = Fraction(1, 10**6)
MIN_DECIMALS = 6
#: Criterion 5: the file of each Role. Design, "Output files": the columns, in order.
LIST_FILES: Mapping[str, str] = {BATTER_ROLE: "top10_batters.csv", BOWLER_ROLE: "top10_bowlers.csv"}
HEADER: tuple[str, ...] = (
    "rank", "player_id", "player_name", "final_score", "ci_low", "ci_high", "p_top10", "median_rank", "hit_rate",
    "share_ipl", "share_full_member", "share_associate",
)  # fmt: skip
#: The match classes in CompClass order, and the share column of each.
CLASSES: tuple[CompClass, ...] = (CompClass.IPL, CompClass.FULL_MEMBER, CompClass.ASSOCIATE)
SHARE_COLUMNS: tuple[str, ...] = ("share_ipl", "share_full_member", "share_associate")

#: A share is a float64 division of two counts: within 2^-53 (about 1.1e-16) of the exact fraction.
_DIVISION_SLACK = Fraction(1, 10**15)
#: A fixed-point decimal as written: no exponent, no "nan" or "inf".
_DECIMAL = re.compile(r"-?\d+(?:\.(?P<places>\d+))?")


# =============================================================================== the examples

class Boot(NamedTuple):
    """The bootstrap row of an Eligible_Player (Requirement 19, criterion 5)."""

    ci_low: float  # NaN, with ci_high and median_rank, for a player never ranked in a resample
    ci_high: float
    median_rank: float
    p_top10: float
    n_ranked: int
    n_unranked: int


class Player(NamedTuple):
    """One player of one Role: his ranking row, his Pressure_Ball counts and his bootstrap row."""

    player_id: int
    rank: int | None  # None: not an Eligible_Player
    final_score: float  # NaN unless an Eligible_Player
    hit_rate: float
    counts: tuple[int, int, int]  # unweighted Pressure_Balls from IPL, Full_Member_Match and Associate_Match matches
    split: str  # how ``counts`` was drawn
    boot: Boot | None  # None unless an Eligible_Player

    def shares(self) -> tuple[Fraction, ...]:
        """The exact share of his Pressure_Balls in each match class."""
        total = sum(self.counts)
        return tuple(Fraction(count, total) for count in self.counts)


class ListCase(NamedTuple):
    """A drawn Main_Ranking of both Roles, with bootstrap rows, Pressure_Ball counts and names (module docstring)."""

    players: Mapping[str, tuple[Player, ...]]  # Role -> its Eligible_Players in rank order, then the others
    names: Mapping[int, str]  # player_id -> his one Candidate_Name
    mapping_rows: Mapping[int, int]  # player_id -> the mapping rows that hold his name
    n_boot: int  # B, the Bootstrap_Resamples behind P_Top10
    seed: int  # the row orders of the tables and the contribution rows behind the counts

    def eligible_count(self, role: str) -> int:
        return sum(player.rank is not None for player in self.players[role])

    def top(self, role: str) -> tuple[Player, ...]:
        """The Eligible_Players at ranks 1 to 10 of the Role's ranking, in rank order."""
        ranked = sorted((player for player in self.players[role] if player.rank is not None), key=lambda p: p.rank)
        return tuple(ranked[:TOP])

    def with_eligible(self, counts: Mapping[str, int]) -> ListCase:
        """The case with only ranks 1..counts[role] eligible in each Role of ``counts`` (check 5)."""
        players = dict(self.players)
        for role, count in counts.items():
            players[role] = tuple(
                player._replace(rank=None, final_score=math.nan, boot=None)
                if player.rank is not None and player.rank > count else player
                for player in self.players[role]
            )  # fmt: skip
        return self._replace(players=players)

    def __repr__(self) -> str:
        lines = [f"B {self.n_boot}, seed {self.seed}"]
        for role in ROLES:
            lines.append(f"{role}: {self.eligible_count(role)} Eligible_Players of {len(self.players[role])} players")
            lines += [f"  {player!r}" for player in self.players[role]]
        lines.append(f"names {dict(self.names)!r}")
        lines.append(f"mapping rows {dict(self.mapping_rows)!r}")
        return "\n".join(lines)


_MAX_ELIGIBLE = 30
_MAX_OTHERS = 4
_ID_GAPS = st.integers(1, 30_000)  # from one player_id to the next
_SCORES = st.floats(-40.0, 40.0, allow_nan=False, allow_infinity=False)  # Final_Score, runs per over
_SPREADS = st.floats(0.0, 15.0, allow_nan=False, allow_infinity=False)  # from a Final_Score to an interval bound
_N_BOOT = st.one_of(st.just(2000), st.integers(1, 2000))  # D16 gives 2,000 Bootstrap_Resamples
_HIT_RATES = st.integers(1, 60).flatmap(lambda m: st.integers(0, m).map(lambda k: k / m))
_NAMES = st.text(alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ '-.,\"éÖāñ", min_size=1, max_size=14)
_TEAMS = ("Rajasthan Royals", "India", "India A", "Nepal", "Scotland")

#: How a player's Pressure_Balls split over the three match classes.
ONE_CLASS, THIRDS, TWO_CLASSES, SMALL_COUNTS, ANY_COUNTS, NO_PRESSURE_BALLS = (
    "all in one class", "equal thirds", "two classes", "small counts", "any counts", "none",
)  # fmt: skip
_SPLITS = (ONE_CLASS, THIRDS, TWO_CLASSES, SMALL_COUNTS, ANY_COUNTS)


def _n_eligible(draw: Any) -> int:
    return TOP if draw(st.integers(0, 4)) == 0 else draw(st.integers(TOP + 1, _MAX_ELIGIBLE))


def _final_scores(draw: Any, n: int) -> list[float]:
    """n Final_Scores in rank order: never rising, often tied."""
    pool = draw(st.lists(_SCORES, min_size=1, max_size=n))
    return sorted(draw(st.lists(st.sampled_from(pool), min_size=n, max_size=n)), reverse=True)


def _counts(draw: Any, *, eligible: bool) -> tuple[tuple[int, int, int], str]:
    """A player's Pressure_Balls in each match class (CLASSES order), and how they were drawn."""
    split = draw(st.sampled_from(_SPLITS if eligible else (*_SPLITS, NO_PRESSURE_BALLS)))
    counts = [0, 0, 0]
    if split == ONE_CLASS:
        counts[draw(st.sampled_from(CLASSES))] = draw(st.integers(1, 2000))
    elif split == THIRDS:
        counts = [draw(st.integers(1, 700))] * 3
    elif split == TWO_CLASSES:
        empty = draw(st.sampled_from(CLASSES))
        counts = [0 if code == empty else draw(st.integers(1, 1500)) for code in CLASSES]
    elif split == SMALL_COUNTS:
        counts = [draw(st.integers(0, 7)) for _ in CLASSES]
    elif split == ANY_COUNTS:
        counts = [draw(st.integers(0, 2000)) for _ in CLASSES]
    if split != NO_PRESSURE_BALLS and sum(counts) == 0:
        counts[draw(st.sampled_from(CLASSES))] = 1
    return (counts[0], counts[1], counts[2]), split


def _boot(draw: Any, score: float, n_eligible: int, n_boot: int) -> Boot:
    kind = draw(st.integers(0, 7))  # 0: never ranked in a resample; 1: ranked in some; else in every one
    n_ranked = 0 if kind == 0 else draw(st.integers(1, n_boot)) if kind == 1 else n_boot
    p_top10 = draw(st.integers(0, n_ranked)) / n_boot  # an unranked resample never counts
    if n_ranked == 0:
        return Boot(math.nan, math.nan, math.nan, p_top10, 0, n_boot)
    low, high = score - draw(_SPREADS), score + draw(_SPREADS)
    return Boot(low, high, draw(st.integers(2, 2 * n_eligible)) / 2, p_top10, n_ranked, n_boot - n_ranked)


@st.composite
def list_cases(draw: Any) -> ListCase:
    """A Main_Ranking of both Roles with at least 10 Eligible_Players each (module docstring)."""
    sizes = {role: (_n_eligible(draw), draw(st.integers(0, _MAX_OTHERS))) for role in ROLES}
    n_bat, n_bowl = (sum(sizes[role]) for role in ROLES)
    overlap = draw(st.integers(0, min(n_bat, n_bowl)))  # players in both Roles
    n_ids = n_bat + n_bowl - overlap
    # Distinct ids by construction (running sums of positive gaps), in a drawn order.
    pool = draw(st.permutations(list(accumulate(draw(st.lists(_ID_GAPS, min_size=n_ids, max_size=n_ids))))))
    ids = {BATTER_ROLE: pool[:n_bat], BOWLER_ROLE: pool[:overlap] + pool[n_bat:]}
    orders = {role: draw(st.permutations(ids[role])) for role in ROLES}  # Eligible_Players first, in rank order
    eligible = {role: list(orders[role][: sizes[role][0]]) for role in ROLES}
    in_both = sorted(set(eligible[BATTER_ROLE]) & set(eligible[BOWLER_ROLE]))
    if in_both and draw(st.booleans()):  # list one player in both Roles' top 10
        shared = draw(st.sampled_from(in_both))
        for role in ROLES:
            eligible[role].remove(shared)
            eligible[role].insert(draw(st.integers(0, TOP - 1)), shared)
    n_boot = draw(_N_BOOT)
    players: dict[str, tuple[Player, ...]] = {}
    for role in ROLES:
        n_eligible = sizes[role][0]
        rows = []
        for rank, (pid, score) in enumerate(zip(eligible[role], _final_scores(draw, n_eligible)), start=1):
            counts, split = _counts(draw, eligible=True)
            boot = _boot(draw, score, n_eligible, n_boot)
            rows.append(Player(pid, rank, score, draw(_HIT_RATES), counts, split, boot))
        for pid in orders[role][n_eligible:]:
            counts, split = _counts(draw, eligible=False)
            hit_rate = draw(st.one_of(st.just(math.nan), _HIT_RATES))
            rows.append(Player(pid, None, math.nan, hit_rate, counts, split, None))
        players[role] = tuple(rows)
    names = draw(st.lists(_NAMES, min_size=len(pool), max_size=len(pool)))
    mapping_rows = draw(st.lists(st.integers(1, 3), min_size=len(pool), max_size=len(pool)))
    seed = draw(st.integers(0, 2**32 - 1))
    return ListCase(players, dict(zip(pool, names)), dict(zip(pool, mapping_rows)), n_boot, seed)


# =============================================================================== the inputs

class Inputs(NamedTuple):
    """The keyword arguments of ``main_lists`` and ``list_table``."""

    ranking: pd.DataFrame  # the ``player_estimates`` columns that the List_Builder reads
    bootstrap: pd.DataFrame  # BOOTSTRAP_COLUMNS
    shares: pd.DataFrame  # pressure_shares of the contributions table
    names: pd.DataFrame  # candidate_names of the mapping table
    balls: pd.DataFrame  # the ball_by_ball name columns (never read: every player has a Candidate_Name)

    def only(self, role: str) -> Inputs:
        """The Role's rows of the ranking, bootstrap and shares tables alone."""

        def keep(frame: pd.DataFrame) -> pd.DataFrame:
            return frame.loc[(frame["role"] == role).to_numpy(dtype=bool)].reset_index(drop=True)

        return self._replace(ranking=keep(self.ranking), bootstrap=keep(self.bootstrap), shares=keep(self.shares))


# Column dtypes as the checkpoints hold them (``player_estimates``, ``bootstrap``, ``contributions``)
# and as the Data_Loader types the mapping and ball_by_ball columns.
_RANKING_DTYPES = {
    "role": "str", "player_id": np.int64, "eligible": np.bool_, "rank": "Int64", "final_score": np.float64,
    "hit_rate": np.float64, "n_pressure_balls": np.int64,
}  # fmt: skip
_BOOTSTRAP_DTYPES = {
    "role": "str", "player_id": np.int64, "main_rank": "Int64", "main_final_score": np.float64,
    "ci_low": np.float64, "ci_high": np.float64, "median_rank": np.float64, "p_top10": np.float64,
    "n_ranked": np.int64, "n_unranked": np.int64,
}  # fmt: skip
_CONTRIBUTION_DTYPES = {
    "role": "str", "player_id": np.int64, "pressure": np.bool_, "n": np.int64, "comp_class": np.int8,
}  # fmt: skip
_MAPPING_DTYPES = {"player_id": "Int64", "player_name": "string", "team_name": "string"}
_BALLS_DTYPES = {"striker_id": "Int64", "striker_name": "string", "bowler_id": "Int64", "bowler_name": "string"}


def _table(rows: list[tuple[Any, ...]], dtypes: Mapping[str, Any], rng: np.random.Generator) -> pd.DataFrame:
    """The rows as a table with ``dtypes`` (a str names a pandas dtype), in a drawn row order."""
    columns = list(zip(*rows)) if rows else [()] * len(dtypes)
    frame = pd.DataFrame(
        {
            name: pd.array(list(values), dtype=dtype) if isinstance(dtype, str) else np.array(values, dtype=dtype)
            for (name, dtype), values in zip(dtypes.items(), columns)
        }
    )
    return frame.iloc[rng.permutation(len(frame))].reset_index(drop=True)


def _parts(rng: np.random.Generator, count: int) -> list[int]:
    """``count`` split into 1-3 positive parts: the contribution rows (matches, cells) that hold it."""
    if count <= 1:
        return [count] if count else []
    pieces = int(rng.integers(1, min(3, count) + 1))
    cuts = np.sort(rng.choice(np.arange(1, count), size=pieces - 1, replace=False))
    return np.diff(np.concatenate(([0], cuts, [count]))).astype(int).tolist()


def _inputs(case: ListCase) -> Inputs:
    """The tables of the case, as ``load_list_inputs`` gives them to the List_Builder."""
    rng = np.random.default_rng(case.seed)
    ranking: list[tuple[Any, ...]] = []
    boot: list[tuple[Any, ...]] = []
    contributions: list[tuple[Any, ...]] = []
    for role in ROLES:
        for p in case.players[role]:
            ranking.append((role, p.player_id, p.rank is not None, p.rank, p.final_score, p.hit_rate, sum(p.counts)))
            if p.boot is not None:
                boot.append((role, p.player_id, p.rank, p.final_score, *p.boot))
            for code, count in zip(CLASSES, p.counts):
                contributions += [(role, p.player_id, True, part, int(code)) for part in _parts(rng, count)]
            # Ordinary_Balls, and a batter's Dismissal_Charge (a pressure row with n = 0): no Pressure_Ball.
            for _ in range(int(rng.integers(0, 3))):
                contributions.append((role, p.player_id, False, int(rng.integers(1, 40)), int(rng.choice(CLASSES))))
            if role == BATTER_ROLE and rng.random() < 0.3:
                contributions.append((role, p.player_id, True, 0, int(rng.choice(CLASSES))))
    mapping = [
        (pid, case.names[pid], _TEAMS[int(rng.integers(len(_TEAMS)))])
        for pid, n_rows in case.mapping_rows.items()
        for _ in range(n_rows)
    ]
    return Inputs(
        ranking=_table(ranking, _RANKING_DTYPES, rng),
        bootstrap=_table(boot, _BOOTSTRAP_DTYPES, rng),
        shares=pressure_shares(_table(contributions, _CONTRIBUTION_DTYPES, rng)),
        names=candidate_names(_table(mapping, _MAPPING_DTYPES, rng)),
        balls=_table([], _BALLS_DTYPES, rng),
    )


# =============================================================================== the checks

def _exact(value: float) -> Fraction | None:
    """The exact value of a float; None for NaN (a blank cell)."""
    return None if math.isnan(value) else Fraction(value)


def _check_number(what: str, text: str, expected: Fraction | None, slack: Fraction = Fraction(0)) -> None:
    """Checks 3 and 4 for one written cell: blank exactly for a missing value; otherwise a fixed-point decimal with
    6+ decimal places unless the value is a whole number, within half a unit in its last place of ``expected``."""
    if expected is None:
        assert text == "", f"{what}: {text!r} where the value is missing; a blank cell is expected"
        return
    match = _DECIMAL.fullmatch(text)
    assert match is not None, f"{what}: {text!r} is not a fixed-point decimal number (value {float(expected)!r})"
    places = len(match["places"] or "")
    if expected.denominator != 1:
        assert places >= MIN_DECIMALS, (
            f"{what}: {text!r} has {places} decimal place(s); the value {float(expected)!r} needs at least "
            f"{MIN_DECIMALS} (Requirement 25, criterion 5)"
        )
    error = abs(Fraction(text) - expected)
    assert error <= Fraction(1, 2 * 10**places) + slack, (
        f"{what}: {text!r} is {float(error):.3g} from the value {float(expected)!r}, more than half a unit in its "
        "last place"
    )


def _check_file(role: str, path: Path, case: ListCase) -> list[int]:
    """Checks 1-4 on one written list file; returns its player_ids in row order."""
    with path.open(encoding="utf-8", newline="") as handle:
        header, *rows = list(csv.reader(handle))
    what = path.name
    assert tuple(header) == HEADER, f"{what}: header {header}; expected {list(HEADER)}"
    top = case.top(role)

    # 1. Ten rows, ranks 1-10 in order, and the player_ids at ranks 1-10 of the Role's ranking.
    assert len(rows) == TOP, f"{what}: {len(rows)} rows, not {TOP}"
    assert all(len(row) == len(HEADER) for row in rows), f"{what}: a row without {len(HEADER)} cells: {rows}"
    cells = [dict(zip(HEADER, row)) for row in rows]
    assert [cell["rank"] for cell in cells] == [str(rank) for rank in range(1, TOP + 1)], (
        f"{what}: ranks {[cell['rank'] for cell in cells]}"
    )
    ids = [cell["player_id"] for cell in cells]
    assert ids == [str(player.player_id) for player in top], (
        f"{what}: player_ids {ids}; ranks 1-10 of the {role} ranking are {[p.player_id for p in top]}"
    )
    assert len(set(ids)) == TOP, f"{what}: repeated player_id in {ids}"

    for rank, (cell, player) in enumerate(zip(cells, top), start=1):
        at = f"{what}, rank {rank} (player_id {player.player_id})"
        boot = player.boot
        assert boot is not None, f"strategy bug: Eligible_Player {player.player_id} has no bootstrap row"
        # 4. The player's name and values, each written to its own precision (and 3., the decimals).
        assert cell["player_name"] == case.names[player.player_id], (
            f"{at}: player_name {cell['player_name']!r}; his Candidate_Name is {case.names[player.player_id]!r}"
        )
        copied = {
            "final_score": player.final_score, "ci_low": boot.ci_low, "ci_high": boot.ci_high,
            "p_top10": boot.p_top10, "median_rank": boot.median_rank, "hit_rate": player.hit_rate,
        }  # fmt: skip
        for column, value in copied.items():
            _check_number(f"{at}, {column}", cell[column], _exact(value))
        for column, share in zip(SHARE_COLUMNS, player.shares()):
            _check_number(f"{at}, {column} (counts {player.counts})", cell[column], share, _DIVISION_SLACK)
        # 2. The written shares lie in [0, 1] and sum to 1 within 1e-6, summed exactly.
        written = [Fraction(cell[column]) for column in SHARE_COLUMNS]
        texts = [cell[column] for column in SHARE_COLUMNS]
        assert all(0 <= share <= 1 for share in written), f"{at}: a share outside [0, 1]: {texts}"
        total = sum(written)
        assert abs(total - 1) <= SHARE_SUM_TOLERANCE, (
            f"{at}: the written shares {texts} sum to {float(total)!r}, not 1 within {float(SHARE_SUM_TOLERANCE)}"
        )

    # The typed view of the file, read as the pipeline-output tests read it.
    table = pd.read_csv(path, keep_default_na=False, na_values=[""], dtype={"player_name": str}, encoding="utf-8")
    assert table.columns.tolist() == list(HEADER), f"{what}: pandas reads the columns {table.columns.tolist()}"
    for column in ("rank", "player_id"):
        dtype = table[column].dtype
        assert pd.api.types.is_integer_dtype(dtype), f"{what}: pandas reads {column} as {dtype}"
    assert table["player_name"].tolist() == [case.names[player.player_id] for player in top], (
        f"{what}: pandas reads the names {table['player_name'].tolist()}"
    )
    return [player.player_id for player in top]


def _files(folder: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in folder.iterdir()}


def _names_role_and_count(message: str, role: str, count: int) -> bool:
    """True when ``message`` names the Role and then, as the next number, its Eligible_Player count."""
    return re.search(rf"\b{re.escape(role)}\b\D*\b{count}\b", message) is not None


def _events(case: ListCase) -> None:
    tops = {role: case.top(role) for role in ROLES}
    for role, top in tops.items():
        if case.eligible_count(role) == TOP:
            event("a Role with exactly 10 Eligible_Players")
        if len({player.final_score for player in top}) < TOP:
            event("tied Final_Scores in a top 10")
        for player in top:
            event(f"a listed player's Pressure_Balls: {player.split}")
            if player.boot is not None and math.isnan(player.boot.ci_low):
                event("a listed player never ranked in a resample: blank ci_low, ci_high and median_rank")
            if any(char in case.names[player.player_id] for char in ',"'):
                event("a listed name that the CSV quotes")
    if {p.player_id for p in tops[BATTER_ROLE]} & {p.player_id for p in tops[BOWLER_ROLE]}:
        event("a player in both top-10 lists")


# =============================================================================== the tests

@given(case=list_cases())
def test_final_list_invariants(case: ListCase, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Each written list has 10 rows with ranks 1-10 in order, the 10 distinct player_ids at ranks 1-10 of its
    Role's Main_Ranking with their values, 6+ decimal places, and shares in [0, 1] summing to 1 within 1e-6 as
    written; the Roles' lists are built independently (checks 1-4 and 6 of the module docstring).

    **Validates: Requirements 25.1, 25.5**
    """
    _events(case)
    inputs = _inputs(case)
    lists = main_lists(**inputs._asdict())
    folder = tmp_path_factory.mktemp("p58_lists")
    paths = write_lists(lists, folder)
    assert paths == {role: folder / name for role, name in LIST_FILES.items()}, f"write_lists returned {paths}"
    assert sorted(_files(folder)) == sorted(LIST_FILES.values()), f"the folder holds {sorted(_files(folder))}"
    written = {role: _check_file(role, folder / LIST_FILES[role], case) for role in ROLES}

    # 6. Built independently: a player in both top 10s is in both files, and each Role's own rows give its list.
    tops = {role: {player.player_id for player in case.top(role)} for role in ROLES}
    assert set(written[BATTER_ROLE]) & set(written[BOWLER_ROLE]) == tops[BATTER_ROLE] & tops[BOWLER_ROLE]
    for role in ROLES:
        alone = list_table(role, **inputs.only(role)._asdict())
        pd.testing.assert_frame_equal(alone, lists[role], check_exact=True, obj=f"the {role} list from its own rows")


_SHORT_ROLES = st.dictionaries(st.sampled_from(ROLES), st.integers(0, TOP - 1), min_size=1)


@given(case=list_cases(), short=_SHORT_ROLES)
def test_a_short_role_writes_neither_file(
    case: ListCase, short: dict[str, int], tmp_path_factory: pytest.TempPathFactory
) -> None:
    """A Role with fewer than 10 Eligible_Players stops the List_Builder with a StageError that names the Role and
    its count, and neither top-10 file is written (check 5 of the module docstring).

    **Validates: Requirements 25.1, 25.9**
    """
    full = main_lists(**_inputs(case)._asdict())
    inputs = _inputs(case.with_eligible(short))
    counts = {role: short.get(role, case.eligible_count(role)) for role in ROLES}
    assert eligible_counts(inputs.ranking) == counts, f"Eligible_Player counts {eligible_counts(inputs.ranking)}"
    folder = tmp_path_factory.mktemp("p58_short")
    before = {name: f"rank,player_id\n1,{role} list of an earlier Run\n".encode() for role, name in LIST_FILES.items()}
    for name, data in before.items():
        (folder / name).write_bytes(data)

    with pytest.raises(StageError) as raised:
        write_lists(main_lists(**inputs._asdict()), folder)
    message = str(raised.value)
    assert raised.value.exit_code == 5, raised.value.exit_code
    for role, count in short.items():
        assert _names_role_and_count(message, role, count), f"no {role} Role and count {count} in: {message}"
    assert _files(folder) == before, "a top-10 file was written although a Role is short"

    for role in ROLES:
        if role in short:
            with pytest.raises(StageError) as one:
                list_table(role, **inputs._asdict())
            assert _names_role_and_count(str(one.value), role, short[role]), str(one.value)
        else:  # the other Role's list does not depend on the short Role
            pd.testing.assert_frame_equal(list_table(role, **inputs._asdict()), full[role], check_exact=True)

    cut = {role: table.iloc[: short[role]] if role in short else table for role, table in full.items()}
    with pytest.raises(ValueError):
        write_lists(cut, folder)
    assert _files(folder) == before, "write_lists wrote a file although a list has fewer than 10 rows"
    event(f"short Roles: {', '.join(sorted(short))}")
