"""Pipeline-output test of the NBV per Delivery in each innings x Phase cell (Requirement 15, criterion 17).

**Validates: Requirements 15.17**

Design sections "Value_Engine (``value.py``)" and Property 42. The NBV of Requirement 15, criteria
1-14 compares each Delivery with what an Average_Skill player would have produced: the batter view
sets the striker's theta to the fold's Average_Skill and the bowler view the bowler's. Its mean in a
cell is therefore not 0 where the cell's players are not average; the innings-2 death cell is batted
mostly by lower-order players. Criterion 17 measures calibration instead. The NBV is recomputed
with each player's own theta: the Outcome_Model's ``actual`` variant in place of the batter view
and the bowler view (``value.OWN_THETA_VIEWS``). Every other term stays the stage's: Omega, the
dismissal flags D and B, the Dismissal_Charges, the Extras_Cost terms, the Extras_Model
probabilities and the Training_Folds cell tables. With outcomes drawn from the probabilities that
the Expected_Value uses, NBV has zero expectation, so a calibrated M4 gives cell means near 0.

The tests read a completed Run's checkpoints through the CheckpointStore, which checks every file
against its stage's manifest, and build the inputs with ``value.load_value_inputs``, the builder that
the ``value`` stage itself calls.

Requirement 15, criterion 17: ``test_own_theta_nbv_per_delivery_lies_within_the_config_band``
    One case per Role and innings x Phase cell of innings 1 and 2 (12 cases). The Role's own-theta
    NBV (``value.own_theta_nbv``) is summed without weights over the cell's Batter_Deliveries
    (Legal_Balls and No_Balls) or Bowler_Deliveries (every Delivery); for the batter Role the cell's
    Dismissal_Charges are added; the total is divided by the cell's Delivery count. The test computes
    this mean with numpy, checks that it equals the ``mean_per_delivery`` of
    ``value.own_theta_cell_means`` (the table that the stage writes to
    ``outputs/value/nbv_cells_own_theta.csv``), and that it lies within +/-``tests.nbv_cell_tol``
    (0.05 runs as delivered), both bounds included.
The helper builds the stage's inputs: ``test_helper_with_the_average_skill_views_reproduces_the_saved_nbv``
    One case per Role. ``value.own_theta_nbv`` with the stage's own views (``value.NBV_VIEWS``) gives
    the ``nbv_bat`` or ``nbv_bowl`` of the saved ``value`` table on every row within 1e-9, with NaN
    on the same rows (the Wides of the batter Role); for the batter Role, the saved Dismissal_Charges
    too. The own-theta NBV therefore differs from the stage's NBV only in the M4 variant.
Information: ``test_average_skill_cell_means_are_finite``
    The stage's cell means against an Average_Skill player (the ``nbv_cells`` table, also
    ``outputs/value/nbv_cells.csv``) are not held to the band. The test writes them to the test log
    at level INFO, beside the own-theta means, and checks only that they are finite.

Approximation. No ``actual`` variant sets or clears the Free_Hit flag, so the terms that read
``batter_view_fh`` (the batter Expected_Value of a No_Ball, criterion 3) or ``bowler_view_nofh`` (the
replacement ball of a No_Ball's Extras_Cost and of the expected No_Ball cost, criteria 6 and 14) keep
those views, with the Role player's theta at Average_Skill. There the flag moves the expected bat runs
far more than theta does (0.47 runs against about 0.01 on the innings-1 death No_Balls as delivered).
No_Balls are about 0.4% of Deliveries.

Skips and stale checkpoints
    Every test is skipped, with the missing files named, while a checkpoint file it reads is absent
    (``requires_outputs``), and with the reason while a checkpoint is not complete. Every test fails
    when the ``value`` checkpoint was built from another copy of a checkpoint that the inputs come
    from (each manifest records the manifest hash of every stage that the stage read). When a check
    fails and config.py has changed since the ``value`` stage ran (the Config hash in its manifest
    differs), the failure message says so.
"""

from __future__ import annotations

import hashlib
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd
import pytest

from superr import folds, pressure, reconstruct, state, value
from superr.models import extras, model_choice, outcome, remaining_runs, win_prob
from superr.pipeline import OUTPUTS_FOLDER, checkpoint_layout
from superr.reconstruct import DeliveryClass
from superr.state import Phase
from superr.store import MANIFEST_NAME, CheckpointStore

if TYPE_CHECKING:
    from superr.settings import Settings

LOG = logging.getLogger(__name__)

# ------------------------------------------------------------------------------------- constants

#: The Config band of criterion 17, and the D14 switch that the NBV formulas read.
TOLERANCE_KEY = "tests.nbv_cell_tol"
PENALTY_KEY = "design_choices.D14_retire_out_batter_penalty"

#: Agreement with the saved tables and between the two cell aggregations: the same float64 arithmetic
#: on the same inputs, so only the order of summation can differ.
TABLE_TOLERANCE = 1e-9

ROLE_CASES = [pytest.param(role, id=role) for role in value.ROLES]


def _cell_name(cell: int) -> tuple[int, str]:
    """The innings and the Phase name of an innings x Phase cell code (3 x (innings - 1) + Phase)."""
    return state.STATE_INNINGS[cell // len(Phase)], Phase(cell % len(Phase)).name.lower()


CELL_CASES = [
    pytest.param(role, cell, id=f"{role}-innings{_cell_name(cell)[0]}-{_cell_name(cell)[1]}")
    for role in value.ROLES
    for cell in value.CELLS
]

#: The checkpoints that value.load_value_inputs reads (the value stage's reads except ``load``, which
#: only names the players of the extremes files), with the tables named up front. The Primary_Models'
#: WP and M4 tables lie in the win_prob and outcome checkpoints, under names that model_choice gives.
INPUT_TABLES: dict[str, tuple[str, ...]] = {
    reconstruct.STAGE_NAME: (reconstruct.BALLS_TABLE,),
    state.STAGE_NAME: (state.STATE_TABLE,),
    folds.STAGE_NAME: (folds.FOLD_TABLE,),
    remaining_runs.STAGE_NAME: (remaining_runs.PREDS_TABLE,),
    win_prob.STAGE_NAME: (),
    outcome.STAGE_NAME: (outcome.CLASS_RUN_VALUES_TABLE, extras.PREDS_TABLE),
    model_choice.STAGE_NAME: (model_choice.COMPARISON_TABLE, model_choice.ROWS_TABLE, model_choice.SOURCES_TABLE),
    pressure.STAGE_NAME: (pressure.PRESSURE_TABLE,),
}
#: The tables of the value checkpoint that the tests compare with.
VALUE_TABLES: tuple[str, ...] = (value.VALUE_TABLE, value.DISMISSAL_CHARGES_TABLE, value.NBV_CELLS_TABLE)

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


FILES = tuple(
    path
    for stage, tables in {value.STAGE_NAME: VALUE_TABLES, **INPUT_TABLES}.items()
    for path in _checkpoint_files(stage, *(f"{table}.parquet" for table in tables))
)

pytestmark = pytest.mark.requires_outputs(*FILES)

# ------------------------------------------------------------------------------- shared helpers


class Run(NamedTuple):
    """The Value_Engine's inputs rebuilt from the checkpoints, and the Config values the formulas read."""

    inputs: value.ValueInputs
    penalty: bool  # design_choices.D14_retire_out_batter_penalty
    config_changed: bool  # config.py differs from the Config that the value stage ran with


class CellMean(NamedTuple):
    """One Role's NBV in one innings x Phase cell, recomputed by the test with numpy."""

    n: int  # the cell's Batter_Deliveries or Bowler_Deliveries
    nbv: float  # the unweighted sum of the Role's NBV over them
    charges: float  # the sum of the cell's Dismissal_Charges (batter Role; 0 for the bowler)
    mean: float  # (nbv + charges) / n; NaN for an empty cell
    n_not_finite: int  # the cell's Deliveries of the Role without a finite NBV


class OwnTheta(NamedTuple):
    """The own-theta cell means: recomputed by the test, and as value.own_theta_cell_means gives them."""

    cells: dict[tuple[str, int], CellMean]
    table: pd.DataFrame  # NBV_CELL_COLUMNS


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
            f"{MANIFEST_NAME} records another manifest hash, or none, for them. Its values cannot be checked against "
            f"inputs rebuilt from them; run `.venv\\Scripts\\python run_pipeline.py --from-stage {stage}`"
        )


@pytest.fixture(scope="module")
def run(checkpoints: CheckpointStore, settings: Settings) -> Run:
    """The inputs of the value stage, rebuilt by its own builder from the checkpoints it was built on."""
    _require_complete(checkpoints, value.STAGE_NAME, *INPUT_TABLES)
    _require_built_on(checkpoints, value.STAGE_NAME, *INPUT_TABLES)
    return Run(
        inputs=value.load_value_inputs(checkpoints),
        penalty=settings.get(PENALTY_KEY),
        config_changed=checkpoints.manifest(value.STAGE_NAME)["config_sha256"] != settings.effective_sha256(),
    )


def _config_note(run: Run) -> str:
    """For a failed check: a note when config.py changed after the value stage ran."""
    if not run.config_changed:
        return ""
    return (
        f"\nconfig.py has changed since the {value.STAGE_NAME} stage ran (its {MANIFEST_NAME} records another Config "
        f"hash), so the Run may have used another {PENALTY_KEY} or {TOLERANCE_KEY}; run "
        f"`.venv\\Scripts\\python run_pipeline.py --from-stage {value.STAGE_NAME}`"
    )


def _cell_means(table: pd.DataFrame, charges: pd.DataFrame) -> dict[tuple[str, int], CellMean]:
    """Each Role's NBV per Delivery in each cell of a value table, with the cell's Dismissal_Charges for the batter."""
    cell = table[value.CELL_COLUMN].to_numpy(dtype=np.int64)
    wide = table["cls"].to_numpy(dtype=np.int64) == int(DeliveryClass.WIDE)
    at = pd.Index(table["row"].to_numpy(dtype=np.int64)).get_indexer(charges["row"].to_numpy(dtype=np.int64))
    assert (at >= 0).all(), f"{int((at < 0).sum()):,} Dismissal_Charges name a row that is not a rated Delivery"
    charge_cell = cell[at]
    charge = charges["charge"].to_numpy(dtype=np.float64)
    roles = {value.BATTER_ROLE: (~wide, "nbv_bat"), value.BOWLER_ROLE: (np.ones(len(table), dtype=bool), "nbv_bowl")}
    means: dict[tuple[str, int], CellMean] = {}
    for role, (mask, column) in roles.items():
        nbv = table[column].to_numpy(dtype=np.float64)
        for code in value.CELLS:
            mine = mask & (cell == code)
            n = int(mine.sum())
            total = float(nbv[mine].sum())
            extra = float(charge[charge_cell == code].sum()) if role == value.BATTER_ROLE else 0.0
            means[(role, code)] = CellMean(
                n=n,
                nbv=total,
                charges=extra,
                mean=(total + extra) / n if n else math.nan,
                n_not_finite=int((~np.isfinite(nbv[mine])).sum()),
            )
    return means


@pytest.fixture(scope="module")
def own_theta(run: Run) -> OwnTheta:
    """The own-theta cell means, recomputed here and as the stage's helper gives them."""
    table, charges = value.own_theta_nbv(run.inputs, retire_out_penalty=run.penalty)
    return OwnTheta(
        cells=_cell_means(table, charges),
        table=value.own_theta_cell_means(run.inputs, retire_out_penalty=run.penalty),
    )


@pytest.fixture(scope="module")
def saved_cells(checkpoints: CheckpointStore, run: Run) -> pd.DataFrame:
    """The stage's cell means against an Average_Skill player (the nbv_cells table)."""
    return checkpoints.load_table(value.STAGE_NAME, value.NBV_CELLS_TABLE)


def _row(table: pd.DataFrame, role: str, cell: int, what: str) -> pd.Series:
    """The one row of an nbv_cells table for ``role`` and ``cell``."""
    at = np.flatnonzero((table["role"].astype(str).to_numpy() == role) & (table["cell"].to_numpy() == cell))
    assert len(at) == 1, f"{what} has {len(at)} rows for the {role} Role and cell {cell}, not one"
    return table.iloc[at[0]]


def _side_by_side(own: OwnTheta, saved: pd.DataFrame) -> str:
    """Both Roles' own-theta NBV per Delivery in every cell, beside the stage's Average_Skill value, as text."""
    records = []
    for (role, cell), entry in own.cells.items():
        innings, phase = _cell_name(cell)
        average = _row(saved, role, cell, f"the {value.NBV_CELLS_TABLE} table")
        records.append(
            {
                "role": role,
                "innings": innings,
                "phase": phase,
                "n_deliveries": entry.n,
                "own_theta": entry.mean,
                "average_skill": float(average["mean_per_delivery"]),
            }
        )
    return pd.DataFrame.from_records(records).to_string(index=False, float_format=lambda number: f"{number:+.4f}")


# ----------------------------------------------------------------------------------------- tests


@pytest.mark.parametrize(("role", "cell"), CELL_CASES)
def test_own_theta_nbv_per_delivery_lies_within_the_config_band(
    role: str, cell: int, own_theta: OwnTheta, saved_cells: pd.DataFrame, run: Run, settings: Settings
) -> None:
    """The Role's NBV recomputed with each player's own theta, summed over the cell's Deliveries of the
    Role (plus the cell's Dismissal_Charges for the batter Role) and divided by their count, lies within
    +/-``tests.nbv_cell_tol``, and equals what ``value.own_theta_cell_means`` gives.

    **Validates: Requirements 15.17**
    """
    tolerance = float(settings.get(TOLERANCE_KEY))
    entry = own_theta.cells[(role, cell)]
    innings, phase = _cell_name(cell)
    kind = "Batter_Deliveries" if role == value.BATTER_ROLE else "Bowler_Deliveries"
    where = f"the {role} Role in innings {innings} {phase} ({entry.n:,} {kind})"
    assert entry.n > 0, f"{where} is empty, so its NBV per Delivery is not defined" + _config_note(run)
    assert entry.n_not_finite == 0, f"{entry.n_not_finite:,} of {where} have no finite own-theta NBV"

    helper = _row(own_theta.table, role, cell, "value.own_theta_cell_means")
    reported = float(helper["mean_per_delivery"])
    assert int(helper["n_deliveries"]) == entry.n and math.isclose(
        reported, entry.mean, rel_tol=0.0, abs_tol=TABLE_TOLERANCE
    ), (
        f"{where}: value.own_theta_cell_means gives {reported!r} over {int(helper['n_deliveries']):,} Deliveries, but "
        f"the NBV sum {entry.nbv!r} plus the Dismissal_Charges {entry.charges!r} over {entry.n:,} Deliveries is "
        f"{entry.mean!r}"
    )
    assert abs(entry.mean) <= tolerance, (
        f"{where}: the NBV per Delivery with each player's own theta is {entry.mean:+.6f} runs (NBV sum "
        f"{entry.nbv:+.3f}, Dismissal_Charges {entry.charges:+.3f}), outside {TOLERANCE_KEY} = +/-{tolerance:g} "
        "(Requirement 15, criterion 17). Every cell, own theta beside the stage's Average_Skill NBV:\n"
        + _side_by_side(own_theta, saved_cells)
        + _config_note(run)
    )


@pytest.fixture(scope="module")
def average_skill(run: Run) -> tuple[pd.DataFrame, pd.DataFrame]:
    """value.own_theta_nbv with the stage's own views: the stage's NBV and charges, rebuilt from the checkpoints."""
    return value.own_theta_nbv(run.inputs, retire_out_penalty=run.penalty, views=value.NBV_VIEWS)


@pytest.fixture(scope="module")
def saved_value(checkpoints: CheckpointStore, run: Run) -> pd.DataFrame:
    """The value table that the stage saved."""
    return checkpoints.load_table(value.STAGE_NAME, value.VALUE_TABLE)


def _mismatches(name: str, rows: np.ndarray, new: np.ndarray, old: np.ndarray) -> list[str]:
    """Problem lines where ``new`` and ``old`` differ by more than TABLE_TOLERANCE or in where they are NaN."""
    problems = []
    nan_new, nan_old = np.isnan(new), np.isnan(old)
    if not np.array_equal(nan_new, nan_old):
        differ = nan_new != nan_old
        problems.append(
            f"{name} is NaN on other rows: {int(differ.sum()):,} rows differ, for example rows "
            f"{rows[differ][:_SHOWN].tolist()}"
        )
    both = ~nan_new & ~nan_old
    far = np.zeros(len(new), dtype=bool)
    far[both] = ~(np.abs(new[both] - old[both]) <= TABLE_TOLERANCE)
    if far.any():
        shown = ", ".join(
            f"row {int(rows[i])} ({new[i]!r} against {old[i]!r})" for i in np.flatnonzero(far)[:_SHOWN]
        )
        problems.append(f"{name} differs by more than {TABLE_TOLERANCE:g} on {int(far.sum()):,} rows: {shown}")
    return problems


@pytest.mark.parametrize("role", ROLE_CASES)
def test_helper_with_the_average_skill_views_reproduces_the_saved_nbv(
    role: str,
    average_skill: tuple[pd.DataFrame, pd.DataFrame],
    saved_value: pd.DataFrame,
    checkpoints: CheckpointStore,
    run: Run,
) -> None:
    """``value.own_theta_nbv`` with the four Average_Skill views gives the Role's NBV of the saved ``value``
    table on every row within 1e-9, and for the batter Role the saved Dismissal_Charges: the helper builds
    the same inputs as the stage.

    **Validates: Requirements 15.17**
    """
    table, charges = average_skill
    rows = table["row"].to_numpy(dtype=np.int64)
    saved_rows = saved_value["row"].to_numpy(dtype=np.int64)
    assert np.array_equal(rows, saved_rows), (
        f"the helper's {len(rows):,} rows are not the {len(saved_rows):,} rows of the saved {value.VALUE_TABLE} table "
        "in the same order" + _config_note(run)
    )
    column = "nbv_bat" if role == value.BATTER_ROLE else "nbv_bowl"
    problems = _mismatches(
        column, rows, table[column].to_numpy(dtype=np.float64), saved_value[column].to_numpy(dtype=np.float64)
    )
    if role == value.BATTER_ROLE:
        saved = checkpoints.load_table(value.STAGE_NAME, value.DISMISSAL_CHARGES_TABLE)
        same_rows = len(saved) == len(charges) and all(
            np.array_equal(charges[name].to_numpy(dtype=np.int64), saved[name].to_numpy(dtype=np.int64))
            for name in ("row", "player_id")
        )
        if not same_rows:
            problems.append(
                f"the helper gives {len(charges):,} Dismissal_Charges, not the {len(saved):,} of the saved "
                f"{value.DISMISSAL_CHARGES_TABLE} table with the same rows and batters in the same order"
            )
        else:
            problems += _mismatches(
                "charge", charges["row"].to_numpy(dtype=np.int64), charges["charge"].to_numpy(dtype=np.float64),
                saved["charge"].to_numpy(dtype=np.float64),
            )  # fmt: skip
    assert not problems, "\n".join(
        [
            f"value.own_theta_nbv with value.NBV_VIEWS does not reproduce the {role} values of the saved "
            f"{value.STAGE_NAME} checkpoint, so it does not build the stage's inputs:",
            *(f"  - {line}" for line in problems),
        ]
    ) + _config_note(run)


def test_average_skill_cell_means_are_finite(own_theta: OwnTheta, saved_cells: pd.DataFrame) -> None:
    """The stage's cell means against an Average_Skill player (``outputs/value/nbv_cells.csv``) are
    information: the test logs them beside the own-theta means and checks only that they are finite.

    **Validates: Requirements 15.17**
    """
    text = _side_by_side(own_theta, saved_cells)
    LOG.info(
        "NBV per Delivery by Role and innings x Phase cell: own_theta, held to %s (Requirement 15, criterion 17); "
        "average_skill, the stage's NBV against an Average_Skill player (outputs/%s/%s, information):\n%s",
        TOLERANCE_KEY, value.VALUE_FOLDER, value.NBV_CELLS_FILE, text,
    )  # fmt: skip
    not_finite = []
    for role in value.ROLES:
        for cell in value.CELLS:
            mean = float(_row(saved_cells, role, cell, f"the {value.NBV_CELLS_TABLE} table")["mean_per_delivery"])
            if not math.isfinite(mean):
                not_finite.append(f"{role} innings {_cell_name(cell)[0]} {_cell_name(cell)[1]}")
    assert not not_finite, (
        f"the {value.NBV_CELLS_TABLE} table has no finite mean_per_delivery for {', '.join(not_finite)}:\n{text}"
    )
