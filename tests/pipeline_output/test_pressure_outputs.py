"""Pipeline-output tests of the Pressure_Index and the Pressure_Ball flags (Requirement 11, criterion 12;
Requirement 13, criteria 1, 3, 4, 5, 7 and 11).

**Validates: Requirements 11.12, 13.7, 13.11**

Design sections "Pressure_Engine (``pressure.py``, ``ranks.py``)" and Data Models (``pressure``). The
tests read a completed Run's checkpoints through the CheckpointStore, which checks every file against
its stage's manifest, and recompute every check from the saved columns with numpy; none calls the
Pressure_Engine. The ``pressure`` table of the ``pressure`` checkpoint holds one row per Delivery of
innings 1 and 2 (each row of the ``state`` table, the Deliveries of excluded matches included), keyed
by ``row``. Each row takes its Delivery class (``cls``) and its innings from the ``balls`` table of the
``reconstruct`` checkpoint, and its Phase from the ``state`` table of the ``state`` checkpoint. The
batter Role covers the Batter_Deliveries (Legal_Balls and No_Balls) and the bowler Role every Delivery
(glossary). Their columns are ``pi_bat``, ``pressure_bat`` and ``ordinary_bat`` (flags <NA> outside the
Role) and ``pi_bowl``, ``pressure_bowl`` and ``ordinary_bowl``. Every limit comes from the Config: the
D1 band ``design_choices.D1_live_band`` and the band of Requirement 13, criterion 7,
``tests.share_band`` (24.5%-25.5% as delivered), which must be centred on the D3 share
``design_choices.D3_pressure_share``.

Requirement 13, criterion 7: ``test_pressure_balls_are_the_d3_share_of_live_balls``
    One case per Role and Innings_Group: the Role's Pressure_Balls in the Innings_Group, as a share of
    the Role's Live_Balls there, lie within ``tests.share_band``, both bounds included.
Requirement 13, criteria 3 and 11: ``test_dead_balls_have_pi_0_and_live_balls_a_pi_above_0_and_at_most_100``
    One case per Role: every Dead_Ball of the Role has a PI of exactly 0 and is not a Pressure_Ball, and
    every Live_Ball of the Role has a PI above 0 and at most 100.
Requirement 13, criteria 5 and 11: ``test_each_role_delivery_is_exactly_one_of_a_pressure_ball_or_an_ordinary_ball``
    One case per Role: every Delivery of the Role has both flags, and exactly one of them is true.
Requirement 13, criterion 11: ``test_no_wide_has_a_batter_role_pi_or_flag``
    Every Wide has a NaN ``pi_bat`` and <NA> in ``pressure_bat`` and ``ordinary_bat``.
Requirement 11, criterion 12: ``test_innings_2_death_phase_dead_balls_have_a_lower_mean_li_than_live_balls``
    Among the Deliveries of innings 2 in the death Phase, Wides and No_Balls included, the mean LI of
    the Dead_Balls is lower than the mean LI of the Live_Balls.

Two more tests check the flags that the share rests on:

Requirement 13, criterion 4: ``test_pressure_ball_flags_are_monotone_in_pi``
    One case per Role and Innings_Group: among the Role's Live_Balls in the Innings_Group, the lowest PI
    of a Pressure_Ball is above the highest PI of an Ordinary Live_Ball. No Pressure_Ball then has a
    lower PI than an unflagged Live_Ball, and Live_Balls with equal PI have equal flags.
Requirement 13, criterion 1: ``test_live_balls_are_the_deliveries_whose_wp_lies_in_the_d1_band``
    ``live`` is true exactly where the unrounded ``wp_pre`` is finite and within the D1 band, both
    bounds included.

Skips and stale checkpoints
    Every test is skipped, with the missing files named, while a checkpoint file it reads is absent
    (``requires_outputs``), and with the reason while a checkpoint is not complete. Every test fails
    when the ``pressure`` checkpoint was built from another ``reconstruct`` or ``state`` checkpoint than
    the one on disk (each manifest records the manifest hash of every stage that the stage read). When a
    check that reads the Config fails and config.py has changed since the ``pressure`` stage ran (the
    Config hash in its manifest differs), the failure message says so.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd
import pytest

from superr import pressure, reconstruct, state
from superr.pipeline import OUTPUTS_FOLDER, checkpoint_layout
from superr.reconstruct import DeliveryClass
from superr.state import Phase
from superr.store import MANIFEST_NAME, CheckpointStore

if TYPE_CHECKING:
    from superr.settings import Settings

# ------------------------------------------------------------------------------------- constants

#: The Config limits: the D1 band of the Live_Balls, the D3 share, and the band of criterion 7 around it.
LIVE_BAND_KEY = "design_choices.D1_live_band"
SHARE_KEY = "design_choices.D3_pressure_share"
SHARE_BAND_KEY = "tests.share_band"
#: |centre of tests.share_band - D3| allowed: the bounds are decimal fractions held as binary floats.
CENTRE_TOLERANCE = 1e-12

#: The highest PI: PI = 100 x [alpha x rank(LI) + (1 - alpha) x rank(Stress)] (Requirement 13, criterion 2).
PI_MAX = 100.0

#: The Innings_Groups, and the innings of Requirement 11, criterion 12.
INNINGS: tuple[int, ...] = state.STATE_INNINGS
SECOND_INNINGS = 2


class Role(NamedTuple):
    """One Role's columns of the pressure table (Data Models, ``pressure``)."""

    name: str
    pi: str  # the Role's PI: NaN outside the Role, 0 on its Dead_Balls
    flag: str  # Pressure_Ball
    ordinary: str  # Ordinary_Ball


ROLES: tuple[Role, ...] = (
    Role(pressure.BATTER, "pi_bat", "pressure_bat", "ordinary_bat"),
    Role(pressure.BOWLER, "pi_bowl", "pressure_bowl", "ordinary_bowl"),
)
BATTER_ROLE = ROLES[0]

ROLE_CASES = [pytest.param(role, id=role.name) for role in ROLES]
ROLE_INNINGS_CASES = [
    pytest.param(role, innings, id=f"{role.name}-innings{innings}") for role in ROLES for innings in INNINGS
]

#: The pressure-table columns that the tests read.
COLUMNS: tuple[str, ...] = (
    "row", "wp_pre", "live", "li", *(column for role in ROLES for column in (role.pi, role.flag, role.ordinary)),
)  # fmt: skip

RUN_HINT = r"run `.venv\Scripts\python run_pipeline.py` first"
_SHOWN = 5  # example rows in one problem line

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / OUTPUTS_FOLDER
# Only for the checkpoint paths of the requires_outputs mark; the tests read through `checkpoints`.
_PATHS = CheckpointStore(OUTPUTS, checkpoint_layout(ROOT), config_sha256=hashlib.sha256(b"").hexdigest())


def _checkpoint_files(stage: str, *names: str) -> tuple[str, ...]:
    """The manifest and the files ``names`` of a stage's checkpoint, relative to outputs/."""
    folder = _PATHS.stage_dir(stage).relative_to(OUTPUTS)
    return tuple((folder / name).as_posix() for name in (MANIFEST_NAME, *names))


#: The checkpoint tables that the tests read: the pressure table, and the class, innings and Phase of its Deliveries.
READS: dict[str, str] = {
    pressure.STAGE_NAME: pressure.PRESSURE_TABLE,
    reconstruct.STAGE_NAME: reconstruct.BALLS_TABLE,
    state.STAGE_NAME: state.STATE_TABLE,
}
FILES = tuple(path for stage, table in READS.items() for path in _checkpoint_files(stage, f"{table}.parquet"))

pytestmark = pytest.mark.requires_outputs(*FILES)

# ------------------------------------------------------------------------------- shared helpers


class Deliveries(NamedTuple):
    """The rows of the pressure table with the attributes of their Deliveries: numpy arrays in one order."""

    row: np.ndarray  # int64
    innings: np.ndarray  # 1 or 2 (balls)
    phase: np.ndarray  # the Phase code (state)
    wide: np.ndarray  # the Delivery is a Wide (cls of balls)
    live: np.ndarray  # the pressure table's Live_Ball flag
    wp_pre: np.ndarray  # float64
    li: np.ndarray  # float64
    role: dict[str, np.ndarray]  # Role name -> the Role's Deliveries
    pi: dict[str, np.ndarray]  # Role name -> PI (float64, NaN where missing)
    flag: dict[str, np.ndarray]  # Role name -> Pressure_Ball flag, <NA> read as False
    flag_na: dict[str, np.ndarray]  # Role name -> the Pressure_Ball flag is <NA>
    ordinary: dict[str, np.ndarray]  # Role name -> Ordinary_Ball flag, <NA> read as False
    ordinary_na: dict[str, np.ndarray]  # Role name -> the Ordinary_Ball flag is <NA>
    config_changed: bool  # config.py differs from the Config that the pressure stage ran with


@pytest.fixture(scope="module")
def checkpoints(settings: Settings, outputs_dir: Path, workspace_root: Path) -> CheckpointStore:
    """The Run's checkpoints, read through the store, which checks each file against its manifest."""
    return CheckpointStore(outputs_dir, checkpoint_layout(workspace_root), config_sha256=settings.effective_sha256())


def _require_complete(store: CheckpointStore, *stages: str) -> None:
    """Skip unless every stage's checkpoint is complete: a valid manifest, and every file it lists exists."""
    for stage in stages:
        reason = store.incomplete_reason(stage)
        if reason is not None:
            pytest.skip(f"the {stage} checkpoint is not complete ({reason}); {RUN_HINT}")


def _require_built_on(store: CheckpointStore, stage: str, *reads: str) -> None:
    """Fail unless the checkpoint of ``stage`` was built from the checkpoints of ``reads`` now on disk."""
    recorded = store.manifest(stage)["inputs"]
    stale = [read for read in reads if read != stage and recorded.get(read) != store.manifest_sha256(read)]
    if stale:
        pytest.fail(
            f"the {stage} checkpoint was not built from the {', '.join(stale)} checkpoint(s) now on disk: its "
            f"{MANIFEST_NAME} records another manifest hash, or none, for them. Its rows cannot be checked against "
            f"them; run `.venv\\Scripts\\python run_pipeline.py --from-stage {stage}`"
        )


def _flags(table: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray]:
    """A boolean column of the pressure table as (flags with <NA> read as False, <NA> mask)."""
    series = table[column]
    assert pd.api.types.is_bool_dtype(series.dtype), (
        f"the {pressure.PRESSURE_TABLE} column {column} has dtype {series.dtype}, not bool or boolean"
    )
    missing = series.isna().to_numpy(dtype=bool)
    return series.fillna(False).to_numpy(dtype=bool), missing


@pytest.fixture(scope="module")
def deliveries(checkpoints: CheckpointStore, settings: Settings) -> Deliveries:
    """The pressure table's rows with their Delivery class, innings and Phase, checked to be one per state row."""
    _require_complete(checkpoints, *READS)
    _require_built_on(checkpoints, pressure.STAGE_NAME, reconstruct.STAGE_NAME, state.STAGE_NAME)
    name = pressure.PRESSURE_TABLE
    table = checkpoints.load_table(pressure.STAGE_NAME, name)
    absent = [column for column in COLUMNS if column not in table.columns]
    assert not absent, f"the {name} table has no column {', '.join(absent)}"
    rows = table["row"].to_numpy(dtype=np.int64)
    assert len(np.unique(rows)) == len(rows), f"the {name} table lists a row more than once"

    # One row per Delivery of innings 1 and 2: the rows of the state table, with its Phase.
    states = checkpoints.load_table(state.STAGE_NAME, state.STATE_TABLE)
    state_index = pd.Index(states["row"].to_numpy(dtype=np.int64))
    assert state_index.is_unique, f"the {state.STATE_TABLE} table lists a row more than once"
    at_state = state_index.get_indexer(rows)
    not_state = rows[at_state < 0]
    no_pressure = np.setdiff1d(state_index.to_numpy(), rows)
    assert not len(not_state) and not len(no_pressure), (
        f"the {name} table must hold one row per Delivery of innings 1 and 2, the rows of the {state.STATE_TABLE} "
        f"table: {len(no_pressure):,} {state.STATE_TABLE} rows have no {name} row (for example "
        f"{no_pressure[:_SHOWN].tolist()}), and {len(not_state):,} {name} rows are not in the {state.STATE_TABLE} "
        f"table (for example {not_state[:_SHOWN].tolist()})"
    )
    phase = states["phase"].to_numpy(dtype=np.int64)[at_state]
    del states

    # The Delivery class and the innings, from balls.
    balls = checkpoints.load_table(reconstruct.STAGE_NAME, reconstruct.BALLS_TABLE)
    ball_index = pd.Index(balls["row"].to_numpy(dtype=np.int64))
    assert ball_index.is_unique, f"the {reconstruct.BALLS_TABLE} table lists a row more than once"
    at_ball = ball_index.get_indexer(rows)
    assert (at_ball >= 0).all(), (
        f"{int((at_ball < 0).sum()):,} rows of the {name} table are not in the {reconstruct.BALLS_TABLE} table "
        f"(for example {rows[at_ball < 0][:_SHOWN].tolist()})"
    )
    innings = balls["innings"].to_numpy(dtype=np.int64)[at_ball]
    cls = balls["cls"].to_numpy(dtype=np.int64)[at_ball]
    del balls
    for values, allowed, what in (
        (innings, INNINGS, "an innings other than 1 or 2"),
        (cls, [int(member) for member in DeliveryClass], "an unknown Delivery class (cls)"),
        (phase, [int(member) for member in Phase], "an unknown Phase code"),
    ):
        unknown = ~np.isin(values, allowed)
        assert not unknown.any(), (
            f"{int(unknown.sum()):,} rows of the {name} table have {what}; for example rows "
            f"{rows[unknown][:_SHOWN].tolist()}"
        )

    live, live_na = _flags(table, "live")
    assert not live_na.any(), f"the {name} column live is <NA> on {int(live_na.sum()):,} rows"
    wide = cls == int(DeliveryClass.WIDE)
    role = {pressure.BATTER: ~wide, pressure.BOWLER: np.ones(len(rows), dtype=bool)}
    flags = {spec.name: _flags(table, spec.flag) for spec in ROLES}
    ordinary = {spec.name: _flags(table, spec.ordinary) for spec in ROLES}
    return Deliveries(
        row=rows,
        innings=innings,
        phase=phase,
        wide=wide,
        live=live,
        wp_pre=table["wp_pre"].to_numpy(dtype=np.float64, na_value=np.nan),
        li=table["li"].to_numpy(dtype=np.float64, na_value=np.nan),
        role=role,
        pi={spec.name: table[spec.pi].to_numpy(dtype=np.float64, na_value=np.nan) for spec in ROLES},
        flag={key: values for key, (values, _) in flags.items()},
        flag_na={key: missing for key, (_, missing) in flags.items()},
        ordinary={key: values for key, (values, _) in ordinary.items()},
        ordinary_na={key: missing for key, (_, missing) in ordinary.items()},
        config_changed=checkpoints.manifest(pressure.STAGE_NAME)["config_sha256"] != settings.effective_sha256(),
    )


def _examples(d: Deliveries, mask: np.ndarray, values: np.ndarray | None = None, label: str = "") -> str:
    """The first _SHOWN rows of ``mask``, each with its value of ``values`` (named ``label``) when given."""
    at = np.flatnonzero(mask)[:_SHOWN]
    if values is None:
        return "rows " + ", ".join(str(int(d.row[i])) for i in at)
    return ", ".join(f"row {int(d.row[i])} ({label} {float(values[i])!r})" for i in at)


def _failure(header: str, problems: list[str]) -> str:
    return "\n".join([header, *(f"  - {line}" for line in problems)])


def _config_note(d: Deliveries) -> str:
    """For a failed check that reads the Config: a note when config.py changed after the pressure stage ran."""
    if not d.config_changed:
        return ""
    return (
        f"\nconfig.py has changed since the {pressure.STAGE_NAME} stage ran (its {MANIFEST_NAME} records another "
        "Config hash), so the Run may have used other limits; run "
        f"`.venv\\Scripts\\python run_pipeline.py --from-stage {pressure.STAGE_NAME}`"
    )


def _share_band(settings: Settings) -> tuple[float, float, float]:
    """D3 and the bounds of ``tests.share_band``, checked to be centred on D3 (Requirement 13, criterion 7)."""
    share = float(settings.get(SHARE_KEY))
    low, high = (float(bound) for bound in settings.get(SHARE_BAND_KEY))
    centred = math.isclose((low + high) / 2.0, share, rel_tol=0.0, abs_tol=CENTRE_TOLERANCE)
    if not (low < share < high and centred):
        pytest.fail(
            f"{SHARE_BAND_KEY} = [{low!r}, {high!r}] is not centred on {SHARE_KEY} = {share!r}: Requirement 13, "
            "criterion 7 checks the Pressure_Ball share against a band around D3 (24.5%-25.5% around 25% as delivered)"
        )
    return share, low, high


# ----------------------------------------------------------------------------------------- tests


@pytest.mark.parametrize(("role", "innings"), ROLE_INNINGS_CASES)
def test_pressure_balls_are_the_d3_share_of_live_balls(
    role: Role, innings: int, deliveries: Deliveries, settings: Settings
) -> None:
    """The Role's Pressure_Balls in the Innings_Group number between the bounds of ``tests.share_band``
    (D3 plus or minus half its width) times the Role's Live_Balls in the Innings_Group.

    **Validates: Requirements 13.7**
    """
    d = deliveries
    share, low, high = _share_band(settings)
    group = d.role[role.name] & (d.innings == innings)
    n_live = int((group & d.live).sum())
    n_pressure = int((group & d.flag[role.name]).sum())
    where = f"the {role.name} Role in innings {innings}"
    assert n_live > 0, f"{where} has no Live_Ball, so its Pressure_Ball share is not defined" + _config_note(d)
    fraction = n_pressure / n_live
    assert low <= fraction <= high, (
        f"{where}: the {n_pressure:,} Pressure_Balls are {100.0 * fraction:.4f}% of the {n_live:,} Live_Balls, "
        f"outside {SHARE_BAND_KEY} [{100.0 * low:g}%, {100.0 * high:g}%] around D3 = {share:g} "
        "(Requirement 13, criterion 7)" + _config_note(d)
    )


@pytest.mark.parametrize("role", ROLE_CASES)
def test_dead_balls_have_pi_0_and_live_balls_a_pi_above_0_and_at_most_100(role: Role, deliveries: Deliveries) -> None:
    """Every Dead_Ball of the Role has a PI of 0 and no Pressure_Ball flag; every Live_Ball of the Role has
    0 < PI <= 100.

    **Validates: Requirements 13.3, 13.11**
    """
    d = deliveries
    mask = d.role[role.name]
    pi = d.pi[role.name]
    dead = mask & ~d.live
    live = mask & d.live
    in_range = (pi > 0.0) & (pi <= PI_MAX)
    checks = (
        (dead & ~(pi == 0.0), f"Dead_Balls have a {role.pi} other than 0 (criteria 3 and 11)"),
        (dead & d.flag[role.name], f"Dead_Balls are Pressure_Balls ({role.flag} true; criterion 3)"),
        (live & ~in_range, f"Live_Balls have a {role.pi} that is not above 0 and at most {PI_MAX:g} (criterion 11)"),
    )
    problems = [
        f"{int(bad.sum()):,} {text}; for example {_examples(d, bad, pi, role.pi)}" for bad, text in checks if bad.any()
    ]
    assert not problems, _failure(
        f"the {role.name} Role ({int(dead.sum()):,} Dead_Balls, {int(live.sum()):,} Live_Balls) breaks "
        "Requirement 13:",
        problems,
    )


@pytest.mark.parametrize("role", ROLE_CASES)
def test_each_role_delivery_is_exactly_one_of_a_pressure_ball_or_an_ordinary_ball(
    role: Role, deliveries: Deliveries
) -> None:
    """Every Delivery of the Role has a Pressure_Ball and an Ordinary_Ball flag, and exactly one of them is true.

    **Validates: Requirements 13.5, 13.11**
    """
    d = deliveries
    mask = d.role[role.name]
    flag, ordinary = d.flag[role.name], d.ordinary[role.name]
    unflagged = mask & (d.flag_na[role.name] | d.ordinary_na[role.name])
    both = f"{role.flag} and {role.ordinary}"
    checks = (
        (unflagged, f"have <NA> in {role.flag} or {role.ordinary}"),
        (mask & flag & ordinary, f"are both a Pressure_Ball and an Ordinary_Ball ({both} true)"),
        (mask & ~unflagged & ~flag & ~ordinary, f"are neither a Pressure_Ball nor an Ordinary_Ball ({both} false)"),
    )
    problems = [f"{int(bad.sum()):,} {text}; for example {_examples(d, bad)}" for bad, text in checks if bad.any()]
    assert not problems, _failure(
        f"of the {int(mask.sum()):,} Deliveries of the {role.name} Role, some are not exactly one of a Pressure_Ball "
        "or an Ordinary_Ball (Requirement 13, criteria 5 and 11):",
        problems,
    )


def test_no_wide_has_a_batter_role_pi_or_flag(deliveries: Deliveries) -> None:
    """No Wide has a batter-Role PI, Pressure_Ball flag or Ordinary_Ball flag: NaN and <NA> on every Wide.

    **Validates: Requirements 13.11**
    """
    d, role = deliveries, BATTER_ROLE
    wide = d.wide
    checks = (
        (wide & ~np.isnan(d.pi[role.name]), f"have a {role.pi}"),
        (wide & ~d.flag_na[role.name], f"have a {role.flag} flag"),
        (wide & ~d.ordinary_na[role.name], f"have an {role.ordinary} flag"),
    )
    problems = [f"{int(bad.sum()):,} {text}; for example {_examples(d, bad)}" for bad, text in checks if bad.any()]
    assert not problems, _failure(
        f"of the {int(wide.sum()):,} Wides, which are not Batter_Deliveries, some have a batter-Role value "
        "(Requirement 13, criterion 11):",
        problems,
    )


def test_innings_2_death_phase_dead_balls_have_a_lower_mean_li_than_live_balls(deliveries: Deliveries) -> None:
    """Among the innings-2 Deliveries of the death Phase, the Dead_Balls' mean LI is lower than the Live_Balls'.

    **Validates: Requirements 11.12**
    """
    d = deliveries
    death = (d.innings == SECOND_INNINGS) & (d.phase == int(Phase.DEATH))
    dead, live = death & ~d.live, death & d.live
    where = f"the {int(death.sum()):,} innings-{SECOND_INNINGS} death-Phase Deliveries"
    assert dead.any() and live.any(), (
        f"{where} hold {int(dead.sum()):,} Dead_Balls and {int(live.sum()):,} Live_Balls; the comparison of "
        "Requirement 11, criterion 12 needs both"
    )
    no_li = death & ~np.isfinite(d.li)
    assert not no_li.any(), f"{int(no_li.sum()):,} of {where} have no finite LI; for example {_examples(d, no_li)}"
    dead_mean, live_mean = float(d.li[dead].mean()), float(d.li[live].mean())
    assert dead_mean < live_mean, (
        f"among {where}, the {int(dead.sum()):,} Dead_Balls have a mean LI of {dead_mean:.4f}, not lower than the "
        f"{live_mean:.4f} of the {int(live.sum()):,} Live_Balls (Requirement 11, criterion 12)"
    )


@pytest.mark.parametrize(("role", "innings"), ROLE_INNINGS_CASES)
def test_pressure_ball_flags_are_monotone_in_pi(role: Role, innings: int, deliveries: Deliveries) -> None:
    """Among the Role's Live_Balls in the Innings_Group, every Pressure_Ball has a higher PI than every
    Ordinary Live_Ball, so Live_Balls with equal PI have equal flags.

    **Validates: Requirements 13.4**
    """
    d = deliveries
    group = d.role[role.name] & d.live & (d.innings == innings)
    pi = d.pi[role.name]
    where = f"the {role.name} Role's {int(group.sum()):,} Live_Balls in innings {innings}"
    no_pi = group & ~np.isfinite(pi)
    assert not no_pi.any(), (
        f"{int(no_pi.sum()):,} of {where} have no finite {role.pi}; for example {_examples(d, no_pi)}"
    )
    flagged = group & d.flag[role.name]
    ordinary = group & ~d.flag[role.name]
    if flagged.any() and ordinary.any():  # with one side empty there is no pair to order
        lowest, highest = float(pi[flagged].min()), float(pi[ordinary].max())
        low_flagged = flagged & (pi <= highest)
        high_ordinary = ordinary & (pi >= lowest)
        assert lowest > highest, _failure(
            f"the Pressure_Ball flags of {where} are not monotone in {role.pi} (Requirement 13, criterion 4): the "
            f"lowest PI of a Pressure_Ball, {lowest!r}, is not above the highest PI of an Ordinary Live_Ball, "
            f"{highest!r}",
            [
                f"{int(low_flagged.sum()):,} Pressure_Balls have a PI at or below {highest!r}; for example "
                f"{_examples(d, low_flagged, pi, role.pi)}",
                f"{int(high_ordinary.sum()):,} Ordinary Live_Balls have a PI at or above {lowest!r}; for example "
                f"{_examples(d, high_ordinary, pi, role.pi)}",
            ],
        )


def test_live_balls_are_the_deliveries_whose_wp_lies_in_the_d1_band(deliveries: Deliveries, settings: Settings) -> None:
    """``live`` is true exactly on the Deliveries whose unrounded ``wp_pre`` is finite and within the D1 band.

    **Validates: Requirements 13.1**
    """
    d = deliveries
    low, high = (float(bound) for bound in settings.get(LIVE_BAND_KEY))
    wp = d.wp_pre
    finite = np.isfinite(wp)
    in_band = finite & (wp >= low) & (wp <= high)
    checks = (
        (~finite, "have no finite wp_pre (Requirement 13, criterion 10 stops the Run instead)"),
        (d.live & ~in_band, "are Live_Balls, but their wp_pre is not within the band"),
        (~d.live & in_band, "are not Live_Balls, but their wp_pre is within the band"),
    )
    problems = [
        f"{int(bad.sum()):,} Deliveries {text}; for example {_examples(d, bad, wp, 'wp_pre')}"
        for bad, text in checks
        if bad.any()
    ]
    assert not problems, _failure(
        f"the live flag of the {pressure.PRESSURE_TABLE} table ({int(d.live.sum()):,} Live_Balls of {len(d.row):,} "
        f"Deliveries) does not follow the D1 band [{low!r}, {high!r}] ({LIVE_BAND_KEY}, both bounds included; "
        "Requirement 13, criterion 1):",
        problems,
    ) + _config_note(d)
