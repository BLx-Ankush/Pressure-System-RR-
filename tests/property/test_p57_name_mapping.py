# Feature: pressure-rating-system, Property 57: Name mapping
"""Property 57: name mapping (Requirement 25, criteria 2, 3, 7 and 8).

**Validates: Requirements 25.2, 25.3, 25.7, 25.8**

For any mapping table and Delivery names, a listed ``player_id`` with exactly one Candidate_Name
gets that exact string; with several, the one in the most mapping rows, ties broken by the
lexicographically smallest; with none, the most frequent ``striker_name`` (batter list) or
``bowler_name`` (bowler list), ties by the smallest; and an entry is marked unresolved exactly
when its ``player_id`` has zero or several Candidate_Names or its written name is a
Candidate_Name of more than one ``player_id``.

The rules are the ones ``superr.lists`` documents (module docstring, "Names"):

* Names are compared exactly: case, spacing, punctuation and Unicode normalisation count, and
  nothing is trimmed. A missing name and the empty string are no name; a name of spaces only is.
* "Lexicographically smallest" is Python string order (code points), so "AB De" < "AB de".
* The fallback counts every ``ball_by_ball`` row with the player's ``striker_id`` (batter list)
  or ``bowler_id`` (bowler list); a missing name does not count.
* A player with no name in either table raises StageError: there is no Unstop_Name to write.
* The Unresolved_Entry rule applies, as written, to any ``player_id`` and written name.

``Reference`` applies these rules by brute force to the drawn rows rather than to the tables.

Each example (``worlds``)
    10-12 listable players and 0-3 others, with distinct ``player_id`` values from 1 to 99,999.
    Each player draws a name cluster: half the time one of SHARED_CLUSTERS, which other players
    can draw too, and whose strings differ only in case, spacing or punctuation (straight and
    curly apostrophes, composed and decomposed "é", backticks, names of spaces only, texts such
    as "NA", markup such as "**Smith**"); otherwise a cluster of its own ("Player 7", "PLAYER 7",
    "Player  7", " Player 7 "). From its cluster the player gets:

    * 0-3 Candidate_Names (weights 1:2:2:1) in 1-3 mapping rows each. The first has the most
      rows and the others at most as many, so the top count is often tied, also between names
      that differ only in case;
    * 0-2 mapping rows with a missing or empty ``player_name``: no Candidate_Name, but a team;
    * a ``team_name`` on each mapping row, from TEAMS (two differ only in case, one starts with
      a space), or missing;
    * 0-4 ``striker_name`` and 0-4 ``bowler_name`` values, sometimes missing or empty, drawn
      independently, so the two Roles' fallback names often differ and often tie. A listable
      player with no Candidate_Name gets at least one name in each Role.

    The mapping rows and the striker and bowler entries are shuffled. Each ``ball_by_ball`` row
    pairs the next striker entry with the next bowler entry; once one side runs out, that side
    is FILLER_ID with a missing name. UNSEEN_ID is in neither table. Each Role's list is the
    first 10 of a drawn order of the players with a name in that Role, and 1-6 drawn entries pair
    a drawn ``player_id`` or UNSEEN_ID with any drawn name. The tables have the Schema_Registry
    columns and Data_Loader dtypes (``Int64``, ``string``). WORKED_EXAMPLE holds every case,
    with the answers worked out by hand.

Checks
    0. ``candidate_names`` gives each (``player_id``, Candidate_Name) pair with its number of
       mapping rows, and ``team_names`` each ``player_id``'s distinct ``team_name`` values.
    1. ``choose_name``, for every drawn ``player_id``, FILLER_ID and UNSEEN_ID in both Roles,
       returns the reference's name and rule (criteria 2, 7 and 8), the number of Candidate_Names
       and the number of ``player_id`` values holding the name; ``resolved`` follows the
       Unresolved_Entry rule.
    2. Names are exact copies. Python string equality compares code points, so every name check
       fails on any change of case, spacing, punctuation or normalisation. ``main_lists`` writes
       each listed player's chosen name, and ``write_lists`` writes it unchanged into
       ``top10_batters.csv`` and ``top10_bowlers.csv`` (read back with the csv module).
    3. ``unresolved``, on the 20 list entries and on the drawn entries, keeps the entry columns
       and adds the reference's flag, Candidate_Names with mapping rows, holders, listed
       ``player_id`` values and their ``team_name`` values (criterion 3).
    4. ``write_checklist`` (into a new folder under ``tmp_path`` for each example) and
       ``parse_checklist`` give back the 20 entries with their exact names and flags, boxes
       unticked. The notes under each unresolved entry name every Candidate_Name of its
       ``player_id`` and end with one line per listed ``player_id`` and its ``team_name`` values;
       a resolved entry has no notes.
    5. A player with no name in a Role makes ``choose_name`` raise StageError naming the
       ``player_id`` (FILLER_ID, UNSEEN_ID and drawn players without one).

Not covered here
    * Names holding a line break or another character that ``str.splitlines`` splits on: the
      checklist has one line per entry, and a CommonMark code span shows a line break as a space.
    * The lists' numbers and other invariants: Property 58.
    * The delivered data: the pipeline-output tests (Requirement 25, criterion 4).
"""

from __future__ import annotations

import csv
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import zip_longest
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, event, example, given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st

from superr.errors import StageError
from superr.io import PANDAS_DTYPES, SCHEMA_REGISTRY
from superr.lists import (
    CANDIDATE_COLUMNS,
    CHECKLIST_FILE,
    DELIVERY_NAME,
    DETAIL_COLUMNS,
    ENTRY_COLUMNS,
    MOST_MAPPING_ROWS,
    PARSED_COLUMNS,
    SINGLE_CANDIDATE,
    TEAM_COLUMNS,
    candidate_names,
    choose_name,
    list_entries,
    main_lists,
    parse_checklist,
    team_names,
    unresolved,
    write_checklist,
    write_lists,
)
from superr.rating import LIST_SIZE
from superr.value import BATTER_ROLE, BOWLER_ROLE, ROLES

#: Requirement 25, criterion 8: the ball_by_ball id and name columns of each Role's fallback name.
DELIVERY_COLUMNS: Mapping[str, tuple[str, str]] = {
    BATTER_ROLE: ("striker_id", "striker_name"),
    BOWLER_ROLE: ("bowler_id", "bowler_name"),
}

FILLER_ID = 1_000_001  # the empty side of a ball_by_ball row: a player_id with only missing names
UNSEEN_ID = 1_000_002  # in neither table
_PLAYER_IDS = st.integers(1, 99_999)

#: Clusters of names that differ only in case, spacing or punctuation; several players can draw one.
SHARED_CLUSTERS: tuple[tuple[str, ...], ...] = (
    ("MS Dhoni", "Ms Dhoni", "MS DHONI", "ms dhoni"),
    ("AB de Villiers", "AB De Villiers", "AB de  Villiers", "A.B. de Villiers"),
    ("Rashid Khan", " Rashid Khan", "Rashid Khan ", " Rashid Khan "),
    ("O'Brien", "O\u2019Brien", "O-Brien", "o'brien"),
    ("Jos\u00e9 Buttler", "Jose\u0301 Buttler", "JOS\u00c9 BUTTLER", "Jose Buttler"),  # composed and decomposed é
    ("`Tick", "Tick`", "Ti``ck", "` `"),
    (" ", "  ", "NA", "None", "nan"),
    ("Smith (player_id 1), resolved", "**Smith**", "[x] Smith", 'Smith, "Jr"\\'),
)

#: Team names of the mapping rows; a row's team_name may also be missing or empty.
TEAMS: tuple[str, ...] = ("Rajasthan Royals", "rajasthan royals", "Mumbai Indians", " Gujarat Titans", "Delhi Capitals")

_TEAM = st.sampled_from((*TEAMS, None, ""))
_NO_NAME = st.sampled_from((None, ""))
_CANDIDATE_COUNT = st.sampled_from((0, 1, 1, 2, 2, 3))
_NAMELESS_ROWS = st.sampled_from((0, 0, 0, 1, 2))
_MAX_DELIVERY_NAMES = 4


def _is_name(text: str | None) -> bool:
    """A Candidate_Name or Delivery name: present and not the empty string (spaces only count)."""
    return text is not None and text != ""


def _personal_cluster(player_id: int) -> tuple[str, ...]:
    return (f"Player {player_id}", f"PLAYER {player_id}", f"Player  {player_id}", f" Player {player_id} ")


# ----------------------------------------------------------------------------------- the rows

class MappingRow(NamedTuple):
    player_id: int
    team_name: str | None
    player_name: str | None


class BallRow(NamedTuple):
    """The ball_by_ball name columns of one row; the field names are the column names."""

    striker_id: int
    striker_name: str | None
    bowler_id: int
    bowler_name: str | None


class Player(NamedTuple):
    """One drawn player's mapping rows and ball_by_ball names."""

    player_id: int
    mapping: tuple[tuple[str | None, str | None], ...]  # (team_name, player_name) of each mapping row
    striker: tuple[str | None, ...]  # striker_name of each ball_by_ball row with this striker_id
    bowler: tuple[str | None, ...]  # bowler_name of each row with this bowler_id

    def delivery_names(self, role: str) -> tuple[str | None, ...]:
        return self.striker if role == BATTER_ROLE else self.bowler

    def has_name(self, role: str) -> bool:
        """Whether a ``role`` list can name the player: a Candidate_Name or a Delivery name of the Role."""
        return any(_is_name(name) for _, name in self.mapping) or any(map(_is_name, self.delivery_names(role)))

    def texts(self) -> set[str]:
        """Every name of the player: Candidate_Names and Delivery names."""
        drawn = (*(name for _, name in self.mapping), *self.striker, *self.bowler)
        return {text for text in drawn if text is not None and _is_name(text)}


def _row_counts(players: Sequence[Player]) -> tuple[int, int, int]:
    """The number of mapping rows, striker entries and bowler entries."""
    return (
        sum(len(p.mapping) for p in players),
        sum(len(p.striker) for p in players),
        sum(len(p.bowler) for p in players),
    )


class World(NamedTuple):
    """The drawn rows and their order, the two lists and the drawn entries (module docstring)."""

    players: tuple[Player, ...]
    mapping_order: tuple[int, ...]  # a permutation of the mapping rows, taken in player order
    striker_order: tuple[int, ...]  # a permutation of the striker entries
    bowler_order: tuple[int, ...]  # a permutation of the bowler entries
    listed: tuple[tuple[int, ...], ...]  # each Role's listed player_id values in rank order, Roles in ROLES order
    entries: tuple[tuple[int, str], ...]  # drawn (player_id, written player_name) pairs
    by_hand: Mapping[tuple[int, str], tuple[str, str, int, int] | None] | None = None  # WORKED_EXAMPLE only
    by_hand_resolved: tuple[bool, ...] | None = None  # the drawn entries' flags, worked out by hand

    def mapping_rows(self) -> list[MappingRow]:
        rows = [MappingRow(p.player_id, team, name) for p in self.players for team, name in p.mapping]
        return [rows[at] for at in self.mapping_order]

    def ball_rows(self) -> list[BallRow]:
        strikers = [(p.player_id, name) for p in self.players for name in p.striker]
        bowlers = [(p.player_id, name) for p in self.players for name in p.bowler]
        pairs = zip_longest(
            (strikers[at] for at in self.striker_order),
            (bowlers[at] for at in self.bowler_order),
            fillvalue=(FILLER_ID, None),
        )
        return [BallRow(*striker, *bowler) for striker, bowler in pairs]

    def __repr__(self) -> str:
        lines = ["players (player_id: mapping rows as (team_name, player_name); striker_name values; bowler_name values)"]
        lines += [f"  {p.player_id}: {list(p.mapping)!r}; {list(p.striker)!r}; {list(p.bowler)!r}" for p in self.players]
        lines.append(f"mapping row order {list(self.mapping_order)}")
        lines.append(f"striker order {list(self.striker_order)}; bowler order {list(self.bowler_order)}")
        lines += [f"{role} list {list(ids)}" for role, ids in zip(ROLES, self.listed)]
        lines.append(f"drawn entries {list(self.entries)!r}")
        return "\n".join(lines)


_MAPPING_DTYPES = {column: PANDAS_DTYPES[kind] for column, kind in SCHEMA_REGISTRY["mapping"].items()}
_BALL_DTYPES = {column: PANDAS_DTYPES[SCHEMA_REGISTRY["ball_by_ball"][column]] for column in BallRow._fields}
_TEAM_IDS = {team: at for at, team in enumerate(TEAMS, start=1)}
_NO_TEAM_ID = 99  # team_id is a Key_ID_Column, so it is set even where team_name is missing


def mapping_table(rows: Sequence[MappingRow]) -> pd.DataFrame:
    """The load table ``mapping``: the Schema_Registry columns in order, with the Data_Loader dtypes."""
    values = {
        "comp_id": [1] * len(rows),
        "comp_name": ["Indian Premier League"] * len(rows),
        "team_id": [_TEAM_IDS.get(row.team_name, _NO_TEAM_ID) for row in rows],
        "team_name": [row.team_name for row in rows],
        "player_id": [row.player_id for row in rows],
        "player_name": [row.player_name for row in rows],
    }
    return pd.DataFrame({column: pd.array(values[column], dtype=dtype) for column, dtype in _MAPPING_DTYPES.items()})


def balls_table(rows: Sequence[BallRow]) -> pd.DataFrame:
    """The ``ball_by_ball`` name columns, as the List_Builder keeps them, with the Data_Loader dtypes."""
    return pd.DataFrame(
        {column: pd.array([getattr(row, column) for row in rows], dtype=dtype) for column, dtype in _BALL_DTYPES.items()}
    )


# ------------------------------------------------------------------------------- the reference

def is_unresolved(n_candidates: int, n_holders: int) -> bool:
    """Glossary Unresolved_Entry: zero or several Candidate_Names, or a name held by more than one player_id."""
    return n_candidates != 1 or n_holders > 1


def _most_frequent(counts: Mapping[str, int]) -> str:
    """The name with the highest count, ties to the lexicographically smallest (criteria 7 and 8)."""
    top = max(counts.values())
    return min(name for name, n in counts.items() if n == top)


class Reference:
    """The name rules by brute force over the drawn rows (module docstring)."""

    def __init__(self, mapping: Sequence[MappingRow], balls: Sequence[BallRow]) -> None:
        self.rows: dict[int, Counter[str]] = {}  # player_id -> the mapping rows of each Candidate_Name
        self.teams: dict[int, set[str]] = {}  # player_id -> its team_name values
        for row in mapping:
            if row.player_name is not None and _is_name(row.player_name):
                self.rows.setdefault(row.player_id, Counter())[row.player_name] += 1
            if row.team_name is not None and _is_name(row.team_name):
                self.teams.setdefault(row.player_id, set()).add(row.team_name)
        self.delivery: dict[str, dict[int, Counter[str]]] = {role: {} for role in ROLES}  # Role -> id -> counts
        for ball in balls:
            for role, (id_column, name_column) in DELIVERY_COLUMNS.items():
                name = getattr(ball, name_column)
                if _is_name(name):
                    self.delivery[role].setdefault(getattr(ball, id_column), Counter())[name] += 1

    def candidates(self, player_id: int) -> list[tuple[str, int]]:
        """Each Candidate_Name of ``player_id`` with its mapping rows, most rows first, then name."""
        return sorted(self.rows.get(player_id, Counter()).items(), key=lambda item: (-item[1], item[0]))

    def holders(self, name: str) -> tuple[int, ...]:
        """The player_id values that have ``name`` as a Candidate_Name, ascending."""
        return tuple(sorted(pid for pid, counts in self.rows.items() if name in counts))

    def choice(self, player_id: int, role: str) -> tuple[str, str, int, int] | None:
        """(player_name, rule, Candidate_Names, holders of the name) for a ``role`` list; None without a name."""
        own = self.rows.get(player_id, Counter())
        if len(own) == 1:
            (name,) = own
            rule = SINGLE_CANDIDATE
        elif own:
            name, rule = _most_frequent(own), MOST_MAPPING_ROWS
        elif self.delivery[role].get(player_id):
            name, rule = _most_frequent(self.delivery[role][player_id]), DELIVERY_NAME
        else:
            return None
        return name, rule, len(own), len(self.holders(name))

    def candidate_rows(self) -> list[tuple[int, str, int]]:
        return [(pid, name, n) for pid in sorted(self.rows) for name, n in self.candidates(pid)]

    def team_rows(self) -> list[tuple[int, str]]:
        return [(pid, team) for pid in sorted(self.teams) for team in sorted(self.teams[pid])]

    def entry(self, player_id: int, name: str) -> dict[str, Any]:
        """The ``unresolved`` details (DETAIL_COLUMNS) of an entry with ``player_id`` and the written ``name``."""
        own = self.candidates(player_id)
        holders = self.holders(name)
        listed = (player_id, *(pid for pid in holders if pid != player_id))
        return {
            "resolved": not is_unresolved(len(own), len(holders)),
            "n_candidate_names": len(own),
            "n_name_holders": len(holders),
            "candidate_names": tuple(text for text, _ in own),
            "candidate_rows": tuple(n for _, n in own),
            "name_holders": holders,
            "listed_ids": listed,
            "listed_teams": tuple(tuple(sorted(self.teams.get(pid, ()))) for pid in listed),
        }


# ------------------------------------------------------------------------------ the strategy

@st.composite
def _players(draw: Any, player_id: int, listable: bool) -> Player:
    """One player's mapping rows and Delivery names (module docstring)."""
    cluster = draw(st.sampled_from(SHARED_CLUSTERS)) if draw(st.booleans()) else _personal_cluster(player_id)
    k = draw(_CANDIDATE_COUNT)
    names = draw(st.lists(st.sampled_from(cluster), min_size=k, max_size=k, unique=True))
    most = draw(st.integers(1, 3))
    counts = [most, *(draw(st.integers(1, most)) for _ in names[1:])]
    mapping = [(draw(_TEAM), name) for name, n in zip(names, counts) for _ in range(n)]
    mapping += [(draw(_TEAM), draw(_NO_NAME)) for _ in range(draw(_NAMELESS_ROWS))]
    delivery_name = st.sampled_from((*cluster, *cluster, None, ""))
    delivery = []
    for _ in ROLES:
        drawn = draw(st.lists(delivery_name, max_size=_MAX_DELIVERY_NAMES))
        if listable and not names:  # a listed player needs a name in each Role
            drawn = [draw(st.sampled_from(cluster)), *drawn]
        delivery.append(tuple(drawn))
    striker, bowler = delivery
    return Player(player_id, tuple(mapping), striker, bowler)


@st.composite
def worlds(draw: Any) -> World:
    """Players, the order of their rows, the two lists and drawn entries (module docstring)."""
    n_listable, n_other = draw(st.integers(LIST_SIZE, LIST_SIZE + 2)), draw(st.integers(0, 3))
    n = n_listable + n_other
    ids = draw(st.lists(_PLAYER_IDS, min_size=n, max_size=n, unique=True))
    players = tuple(draw(_players(pid, listable=at < n_listable)) for at, pid in enumerate(ids))
    orders = [tuple(draw(st.permutations(range(size)))) for size in _row_counts(players)]
    listed = tuple(
        tuple(draw(st.permutations([p.player_id for p in players if p.has_name(role)]))[:LIST_SIZE]) for role in ROLES
    )
    texts = sorted(set().union(*(p.texts() for p in players)))
    entries = draw(
        st.lists(st.tuples(st.sampled_from((*ids, UNSEEN_ID)), st.sampled_from(texts)), min_size=1, max_size=6)
    )
    return World(players, *orders, listed, tuple(entries))


def _worked(
    *,
    players: tuple[Player, ...],
    listed: tuple[tuple[int, ...], ...],
    entries: tuple[tuple[int, str], ...],
    by_hand: Mapping[tuple[int, str], tuple[str, str, int, int] | None],
    by_hand_resolved: tuple[bool, ...],
) -> World:
    """A World with every row order reversed, for a hand-worked example."""
    orders = (tuple(range(size - 1, -1, -1)) for size in _row_counts(players))
    return World(players, *orders, listed, entries, by_hand, by_hand_resolved)


def _both(player_id: int, answer: tuple[str, str, int, int] | None) -> dict[tuple[int, str], Any]:
    return {(player_id, role): answer for role in ROLES}


_RR, _RR_LOWER, _MI, _GT = TEAMS[:4]

#: Every case, each player's name worked out by hand (player_id: what it shows):
#:
#: * 11: one Candidate_Name, in two mapping rows; its striker_name values are ignored. Resolved.
#: * 12: one Candidate_Name, "Rashid Khan", which is also a Candidate_Name of 13: 2 holders.
#: * 13: "Rashid  Khan" (two spaces) in 2 mapping rows beats "Rashid Khan" in 1.
#: * 14: "AB de Villiers" and "AB De Villiers" tie at 2 rows; "AB De" < "AB de" since "D" < "d".
#: * 15: no mapping row. Its striker_name values "Jos Buttler" and "jos buttler" tie at 2 (the
#:   missing one does not count) and "J" < "j"; its bowler_name is "J Buttler".
#: * 16: two mapping rows without a name (missing, empty) but with teams; Delivery names "`Tick"
#:   (batter) and "Tick`" (bowler; the empty name does not count).
#: * 17: only missing names, StageError in both Roles. 18: " Edge " as a batter, no bowler_name.
#: * 19: no mapping row; its Delivery name "MS Dhoni" is 11's Candidate_Name: 1 holder.
#: * 20: a Candidate_Name of three spaces, on a row with an empty team_name. 21: a curly apostrophe.
#: * FILLER_ID (the 6 empty bowler sides) and UNSEEN_ID: StageError.
#:
#: Drawn entries: (12, "MS Dhoni") is resolved as the rule is written: 12 has one Candidate_Name, and
#: "MS Dhoni" is a Candidate_Name of one player_id (11). The other three are unresolved.
WORKED_EXAMPLE = _worked(
    players=(
        Player(11, ((_RR, "MS Dhoni"), (_MI, "MS Dhoni")), ("Mahendra",) * 3, ()),
        Player(12, ((_RR, "Rashid Khan"),), (), ()),
        Player(13, ((_MI, "Rashid Khan"), (_MI, "Rashid  Khan"), (None, "Rashid  Khan")), (), ()),
        Player(
            14,
            (
                (_RR, "AB de Villiers"), (_RR, "AB de Villiers"), (_MI, "AB De Villiers"), (_MI, "AB De Villiers"),
                (_RR_LOWER, "Ab de Villiers"),
            ),
            (),
            (),
        ),
        Player(15, (), ("jos buttler", "Jos Buttler", None, "Jos Buttler", "jos buttler"), ("J Buttler",)),
        Player(16, ((_RR, None), (_GT, "")), ("`Tick",), ("Tick`", "", None)),
        Player(17, (), (None, None), ()),
        Player(18, (), (" Edge ",), ()),
        Player(19, (), ("MS Dhoni",), ("MS Dhoni", "ms dhoni", "MS Dhoni")),
        Player(20, (("", "   "),), (), ()),
        Player(21, ((_MI, "O\u2019Brien"),), (), ()),
        Player(31, ((_RR, "Player 31"),), (), ()),
    ),
    listed=((18, 14, 11, 15, 12, 16, 19, 20, 13, 21), (16, 15, 13, 19, 11, 31, 12, 20, 21, 14)),
    entries=((12, "MS Dhoni"), (UNSEEN_ID, "Rashid Khan"), (20, "Rashid Khan"), (17, "Nobody")),
    by_hand={
        **_both(11, ("MS Dhoni", SINGLE_CANDIDATE, 1, 1)),
        **_both(12, ("Rashid Khan", SINGLE_CANDIDATE, 1, 2)),
        **_both(13, ("Rashid  Khan", MOST_MAPPING_ROWS, 2, 1)),
        **_both(14, ("AB De Villiers", MOST_MAPPING_ROWS, 3, 1)),
        (15, BATTER_ROLE): ("Jos Buttler", DELIVERY_NAME, 0, 0),
        (15, BOWLER_ROLE): ("J Buttler", DELIVERY_NAME, 0, 0),
        (16, BATTER_ROLE): ("`Tick", DELIVERY_NAME, 0, 0),
        (16, BOWLER_ROLE): ("Tick`", DELIVERY_NAME, 0, 0),
        **_both(17, None),
        (18, BATTER_ROLE): (" Edge ", DELIVERY_NAME, 0, 0),
        (18, BOWLER_ROLE): None,
        **_both(19, ("MS Dhoni", DELIVERY_NAME, 0, 1)),
        **_both(20, ("   ", SINGLE_CANDIDATE, 1, 1)),
        **_both(21, ("O\u2019Brien", SINGLE_CANDIDATE, 1, 1)),
        **_both(31, ("Player 31", SINGLE_CANDIDATE, 1, 1)),
        **_both(FILLER_ID, None),
        **_both(UNSEEN_ID, None),
    },
    by_hand_resolved=(True, False, False, False),
)


# --------------------------------------------------------------------------------- helpers

def _ranking_inputs(listed: Mapping[str, Sequence[int]]) -> dict[str, pd.DataFrame]:
    """``main_lists`` tables that rank ``listed[role]`` 1 to 10 in each Role.

    Only the ranks and ``player_id`` values matter here; every other value is a fixed valid number
    (Property 58 covers them).
    """
    roles = pd.array([role for role in ROLES for _ in listed[role]], dtype="str")
    ids = np.array([pid for role in ROLES for pid in listed[role]], dtype=np.int64)
    rank = np.array([float(at) for role in ROLES for at in range(1, len(listed[role]) + 1)])
    score = 20.0 - rank
    half = np.full(len(ids), 0.5)
    ones, zeros = np.ones(len(ids), dtype=np.int64), np.zeros(len(ids), dtype=np.int64)

    def table(**columns: Any) -> pd.DataFrame:
        return pd.DataFrame({"role": roles, "player_id": ids, **columns})

    return {
        "ranking": table(eligible=np.ones(len(ids), dtype=bool), rank=rank, final_score=score, hit_rate=half),
        "bootstrap": table(
            main_rank=rank, main_final_score=score, ci_low=score - 1.0, ci_high=score + 1.0, median_rank=rank,
            p_top10=half,
        ),
        "shares": table(
            n_pressure_balls=ones, n_ipl=ones, n_full_member=zeros, n_associate=zeros, share_ipl=ones * 1.0,
            share_full_member=zeros * 1.0, share_associate=zeros * 1.0,
        ),
    }


def _records(frame: pd.DataFrame) -> list[tuple[Any, ...]]:
    return list(frame.itertuples(index=False, name=None))


def _differences(got: Sequence[Any], want: Sequence[Any]) -> str:
    """The positions where two sequences differ, with both values: for a failure message."""
    return "\n".join(
        f"  {at}: got {g!r}, expected {w!r}" for at, (g, w) in enumerate(zip_longest(got, want)) if g != w
    )


def _csv_column(path: Path, column: str) -> list[str]:
    """One column of a written CSV, each field exactly as the csv module reads it."""
    with path.open(newline="", encoding="utf-8") as handle:
        header, *rows = csv.reader(handle)
    return [row[header.index(column)] for row in rows]


def _check_details(out: pd.DataFrame, entries: pd.DataFrame, ref: Reference, what: str) -> None:
    """``out = unresolved(entries, ...)`` keeps the entry columns and adds the reference's flag and details."""
    columns = entries.columns.tolist()
    assert out.columns.tolist() == [*columns, *DETAIL_COLUMNS], f"{what}: columns {out.columns.tolist()}"
    pd.testing.assert_frame_equal(out[columns], entries.reset_index(drop=True), obj=f"the entry columns of {what}")
    assert out["resolved"].dtype == bool, out["resolved"].dtype
    for at, record in enumerate(out.to_dict("records")):
        want = ref.entry(int(record["player_id"]), record["player_name"])
        got = {column: record[column] for column in DETAIL_COLUMNS}
        assert got == want, (
            f"{what}, row {at} (player_id {record['player_id']}, {record['player_name']!r}):\n"
            f"  got      {got}\n  expected {want}"
        )


#: A notes line that shows a listed player_id and its team_name values ("none" when it has none).
_TEAM_LINE = re.compile(r" +- player_id (?P<player_id>\d+)[^:]*: (?P<teams>.*)")


def _notes(text: str) -> list[list[str]]:
    """The indented lines under each entry line of a checklist, in file order."""
    blocks: list[list[str]] = []
    under_entry = False
    for line in text.split("\n"):
        if line.startswith("- ["):
            blocks.append([])
            under_entry = True
        elif under_entry and line.startswith(" "):
            blocks[-1].append(line)
        else:
            under_entry = False
    return blocks


def _check_notes(text: str, details: pd.DataFrame) -> None:
    """Criterion 3: an unresolved entry's notes name its Candidate_Names and show each listed id's teams."""
    blocks, records = _notes(text), details.to_dict("records")
    assert len(blocks) == len(records), f"{len(blocks)} entry lines for {len(records)} entries"
    for notes, record in zip(blocks, records):
        entry = f"{record['role']} rank {record['rank']} (player_id {record['player_id']}, {record['player_name']!r})"
        if record["resolved"]:
            assert not notes, f"the resolved entry {entry} has notes: {notes}"
            continue
        joined = "\n".join(notes)
        absent = [name for name in record["candidate_names"] if name not in joined]
        assert notes and not absent, f"the notes of the unresolved entry {entry} lack {absent}: {notes}"
        listed = record["listed_ids"]
        shown = [_TEAM_LINE.fullmatch(line) for line in notes[-len(listed):]]
        got = [(int(match["player_id"]), match["teams"]) if match else None for match in shown]
        want = [(pid, "; ".join(teams) or "none") for pid, teams in zip(listed, record["listed_teams"])]
        assert got == want, f"the notes of {entry} show {got}, not {want}:\n{joined}"


def _classify(
    ref: Reference, choices: Mapping[tuple[int, str], tuple[str, str, int, int] | None], details: pd.DataFrame
) -> None:
    """Hypothesis events naming the cases that the example holds."""
    for (pid, role), want in choices.items():
        if want is None:
            event("a player_id with no name in a Role: StageError")
            continue
        name, rule, _, n_holders = want
        if rule == SINGLE_CANDIDATE:
            event("one Candidate_Name, also another player_id's" if n_holders > 1 else "one Candidate_Name, no namesake")
            continue
        counts = ref.rows[pid] if rule == MOST_MAPPING_ROWS else ref.delivery[role][pid]
        top = max(counts.values())
        tied = [other for other, n in counts.items() if n == top and other != name]
        what = "mapping rows" if rule == MOST_MAPPING_ROWS else "Delivery names"
        if tied:
            event(f"most {what}: a tie, broken by the smallest string")
        if any(other.casefold() == name.casefold() for other in tied):
            event(f"most {what}: a tie between names that differ only in case")
        if rule == DELIVERY_NAME and n_holders:
            event("a Delivery name that is a Candidate_Name of another player_id")
        if rule == DELIVERY_NAME and role == BOWLER_ROLE:
            batter = choices[pid, BATTER_ROLE]
            if batter is not None and batter[0] != name:
                event("Delivery names that differ between the batter and bowler lists")
    for name in details["player_name"].tolist():
        if "`" in name:
            event("listed name with a backtick")
        if name != name.strip(" "):
            event("listed name with a leading or trailing space")
        if not name.strip(" "):
            event("listed name of spaces only")
        if not name.isascii():
            event("listed name with a non-ASCII character")
    n_open = int((~details["resolved"]).sum())
    event("unresolved list entries: " + ("none" if n_open == 0 else "all" if n_open == len(details) else "some"))


# ------------------------------------------------------------------------------------ test

# tmp_path is shared by the examples; each one writes into a new folder under it, so nothing carries over.
@hypothesis_settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(world=worlds())
@example(world=WORKED_EXAMPLE)
def test_name_mapping(world: World, tmp_path: Path) -> None:
    """``choose_name`` applies criteria 2, 7 and 8 exactly, the lists, list files and checklist keep the
    names unchanged, and ``unresolved`` and the checklist mark and explain exactly the Unresolved_Entries.

    **Validates: Requirements 25.2, 25.3, 25.7, 25.8**
    """
    mapping_rows, ball_rows = world.mapping_rows(), world.ball_rows()
    ref = Reference(mapping_rows, ball_rows)
    mapping, balls = mapping_table(mapping_rows), balls_table(ball_rows)

    # 0. Each player_id's Candidate_Names with their mapping rows, and its team_name values.
    names, teams = candidate_names(mapping), team_names(mapping)
    assert names.columns.tolist() == list(CANDIDATE_COLUMNS) and teams.columns.tolist() == list(TEAM_COLUMNS)
    assert _records(names) == ref.candidate_rows(), "candidate_names:\n" + _differences(
        _records(names), ref.candidate_rows()
    )
    assert _records(teams) == ref.team_rows(), "team_names:\n" + _differences(_records(teams), ref.team_rows())

    # 1, 2 and 5. choose_name: the reference's name, rule and counts in both Roles, or StageError.
    ids = [p.player_id for p in world.players] + [FILLER_ID, UNSEEN_ID]
    choices = {(pid, role): ref.choice(pid, role) for pid in ids for role in ROLES}
    if world.by_hand is not None:
        assert {key: choices[key] for key in world.by_hand} == dict(world.by_hand), "the reference is wrong"
    for (pid, role), want in choices.items():
        if want is None:
            with pytest.raises(StageError, match=rf"\bplayer_id {pid}\b"):
                choose_name(pid, names, balls, role)
            continue
        got = choose_name(pid, names, balls, role)
        assert type(got.player_name) is str and got == want, (
            f"choose_name({pid}, role={role!r}) = {tuple(got)!r}, expected {want!r}"
        )
        assert got.resolved == (not is_unresolved(want[2], want[3])), f"choose_name({pid}, role={role!r}).resolved"

    # 2. The lists and the list files hold each listed player's chosen name, unchanged.
    listed = dict(zip(ROLES, world.listed))
    lists = main_lists(**_ranking_inputs(listed), names=names, balls=balls)
    folder = Path(tempfile.mkdtemp(dir=tmp_path))
    paths = write_lists(lists, folder)
    for role in ROLES:
        chosen = [choices[pid, role][0] for pid in listed[role]]
        assert lists[role]["player_id"].tolist() == list(listed[role]), f"{role} list ids"
        in_table = lists[role]["player_name"].tolist()
        assert in_table == chosen, f"the {role} list's names:\n" + _differences(in_table, chosen)
        in_file = _csv_column(paths[role], "player_name")
        assert in_file == chosen, f"the names in {paths[role].name}:\n" + _differences(in_file, chosen)

    # 3. unresolved: the flag and details of the 20 list entries and of the drawn entries.
    entries = list_entries(lists)
    details = unresolved(entries, names, teams)
    _check_details(details, entries, ref, "the list entries")
    drawn = pd.DataFrame(
        {
            "player_id": pd.array([pid for pid, _ in world.entries], dtype="Int64"),
            "player_name": pd.array([name for _, name in world.entries], dtype="string"),
        }
    )
    drawn_details = unresolved(drawn, names, teams)
    _check_details(drawn_details, drawn, ref, "the drawn entries")
    if world.by_hand_resolved is not None:
        assert drawn_details["resolved"].tolist() == list(world.by_hand_resolved), "the reference is wrong"

    # 4. The checklist file gives back the entries, exact names and flags, with notes under the unresolved ones.
    path = write_checklist(lists, details, folder / CHECKLIST_FILE)
    text = path.read_bytes().decode("utf-8")
    parsed = parse_checklist(text)
    assert parsed.columns.tolist() == list(PARSED_COLUMNS)
    back = _records(parsed[list(ENTRY_COLUMNS)])
    assert back == _records(entries), "parse_checklist of the written checklist:\n" + _differences(
        back, _records(entries)
    )
    assert parsed["resolved"].tolist() == details["resolved"].tolist(), "the checklist's resolved flags"
    assert not parsed["checked"].any(), "the checklist has a ticked box"
    _check_notes(text, details)

    _classify(ref, choices, details)
