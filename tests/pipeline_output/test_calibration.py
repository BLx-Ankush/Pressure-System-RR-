"""Pipeline-output calibration test (Requirement 10, criterion 20).

**Validates: Requirements 10.20**

For each M3 innings, the test takes the Primary_Model that the Model_Choice_Checker selected (the
candidate whose ``primary`` flag is set for M3-1 or M3-2 in the ``model_comparison`` table of the
``model_choice`` checkpoint) and recomputes its Calibration_Error from the saved predictions and
labels, without the Pipeline's metric functions:

* predictions: the candidate's calibrated Out_Of_Fold WP before the terminal rules, ``wp_raw`` of
  the ``win_prob`` checkpoint table ``wp_inn<innings>_<candidate>``. This is the WP that the
  stages score, because a terminal rule gives every candidate the same WP on a decided state;
* rows and labels: the ``eligible`` Deliveries of the innings in the ``m3_labels`` table of the
  ``win_prob`` checkpoint, the rows behind every reported M3 metric, with their ``label`` (1 when
  the batting side won the match). Eligible rows lie in matches that Requirement 10, criterion 2
  keeps (the test checks that none has a tie, super-over win, abandoned, no-result or unparsed
  result), have a known winner and are Training_Rows under Requirement 5 (the Score_Tolerance
  rule);
* the Calibration_Curve of the glossary: 10 equal-width bins [0, 0.1), [0.1, 0.2), ...,
  [0.9, 1.0], where a WP equal to b/10 opens bin b and a WP of 1 falls in the last bin. The
  Calibration_Error is the Delivery-count-weighted mean of |mean WP - observed win share| over
  the bins that hold a Delivery.

The test fails, naming the innings and showing the curve, when this Calibration_Error exceeds
``tests.calibration_error_max`` in the Config (0.05 as delivered). It also fails when the value,
or the number of rows behind it, differs from what the comparison table reports for the
Primary_Model: the checkpoint table holds the stage's float64 value (agreement within
TABLE_TOLERANCE) and ``outputs/models/model_choice/model_comparison.csv`` rounds it to 6 decimals
(CSV_TOLERANCE). It is skipped while a file it reads is absent or a checkpoint is not complete, and
fails when the ``model_choice`` checkpoint was built from another ``win_prob`` checkpoint than the
one on disk.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from superr import reconstruct
from superr.models import model_choice, win_prob
from superr.pipeline import OUTPUTS_FOLDER, checkpoint_layout
from superr.store import MANIFEST_NAME, CheckpointStore

if TYPE_CHECKING:
    from superr.settings import Settings

#: The Config threshold of criterion 20.
THRESHOLD_KEY = "tests.calibration_error_max"

#: The result types whose matches Requirement 10, criterion 2 drops from every M3 metric.
DROPPED_RESULT_TYPES: tuple[str, ...] = (
    reconstruct.TIE,
    reconstruct.SUPER_OVER_WIN,
    reconstruct.ABANDONED,
    reconstruct.NO_RESULT,
    reconstruct.UNPARSED,
)

#: The inner edges b/10 (b = 1 ... 9) of the 10 equal-width Calibration_Curve bins.
INNER_EDGES = np.array([b / 10 for b in range(1, 10)])

#: Agreement with the comparison table: the checkpoint copy holds the stage's float64 value (only
#: summation order can differ), the CSV file the value rounded to 6 decimals.
TABLE_TOLERANCE = 1e-9
CSV_TOLERANCE = 1e-6

RUN_HINT = r"run `.venv\Scripts\python run_pipeline.py` first"

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / OUTPUTS_FOLDER
# Only for the checkpoint paths of the requires_outputs mark; the test reads through `checkpoints`.
_PATHS = CheckpointStore(OUTPUTS, checkpoint_layout(ROOT), config_sha256=hashlib.sha256(b"").hexdigest())


def _checkpoint_files(stage: str, *names: str) -> tuple[str, ...]:
    """The manifest and the files ``names`` of a stage's checkpoint, relative to outputs/."""
    folder = _PATHS.stage_dir(stage).relative_to(OUTPUTS)
    return tuple((folder / name).as_posix() for name in (MANIFEST_NAME, *names))


COMPARISON_CSV = (model_choice.MODEL_CHOICE_FOLDER / model_choice.COMPARISON_FILE).as_posix()
#: The files the test reads up front; the Primary_Model's WP table must be in the win_prob checkpoint too.
FILES = (
    *_checkpoint_files(model_choice.STAGE_NAME, f"{model_choice.COMPARISON_TABLE}.parquet"),
    *_checkpoint_files(win_prob.STAGE_NAME, f"{win_prob.LABELS_TABLE}.parquet"),
    COMPARISON_CSV,
)


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


def _require_built_on(store: CheckpointStore, stage: str, read: str) -> None:
    """Fail unless the checkpoint of ``stage`` was built from the checkpoint of ``read`` now on disk."""
    if store.manifest(stage)["inputs"].get(read) != store.manifest_sha256(read):
        pytest.fail(
            f"the {stage} checkpoint was not built from the {read} checkpoint now on disk: its {MANIFEST_NAME} records "
            f"another manifest hash, or none, for it. Run `.venv\\Scripts\\python run_pipeline.py --from-stage {stage}`"
        )


def _calibration_curve(p: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """The Calibration_Curve (glossary) of WP ``p`` against labels ``y``: one row per bin that holds a Delivery."""
    bins = np.digitize(p, INNER_EDGES)  # bin b for b/10 <= p < (b + 1)/10, and bin 9 for 0.9 <= p <= 1
    frame = pd.DataFrame({"bin": bins, "wp": p, "won": y})
    curve = frame.groupby("bin").agg(
        n_deliveries=("wp", "size"), mean_wp=("wp", "mean"), observed_win_share=("won", "mean")
    )
    curve.insert(0, "lower", curve.index.to_numpy() / 10)
    curve.insert(1, "upper", (curve.index.to_numpy() + 1) / 10)
    return curve


def _calibration_error(curve: pd.DataFrame) -> float:
    """The Calibration_Error (glossary): the Delivery-weighted mean |mean WP - observed win share| over the bins."""
    count = curve["n_deliveries"].to_numpy(dtype=np.float64)
    gap = np.abs(curve["mean_wp"].to_numpy(dtype=np.float64) - curve["observed_win_share"].to_numpy(dtype=np.float64))
    return float(np.sum(count * gap) / np.sum(count))


def _primary_entry(table: pd.DataFrame, model: str) -> pd.Series:
    """The comparison-table row of the one candidate flagged as ``model``'s Primary_Model, which did not fail."""
    model_of = table["model"].astype(str).to_numpy(dtype=object)
    flagged = table["primary"].to_numpy(dtype=bool)
    part = table.iloc[np.flatnonzero((model_of == model) & flagged)]
    assert len(part) == 1, (
        f"the {model_choice.COMPARISON_TABLE} table flags {len(part)} {model} Primary_Models "
        f"({part['candidate'].astype(str).tolist()}); the Model_Choice_Checker selects exactly one"
    )
    entry = part.iloc[0]
    reason = entry["failure_reason"]
    assert not (isinstance(reason, str) and reason), (
        f"the {model} Primary_Model {entry['candidate']!r} has the failure reason {reason!r}"
    )
    return entry


def _scored_rows(store: CheckpointStore, innings: int, candidate: str) -> tuple[np.ndarray, np.ndarray]:
    """The ``wp_raw`` and the label of every eligible Delivery of ``innings`` for M3 candidate ``candidate``."""
    name = win_prob.wp_table(innings, candidate)
    wp = store.load_table(win_prob.STAGE_NAME, name)
    labels = store.load_table(win_prob.STAGE_NAME, win_prob.LABELS_TABLE)
    assert wp["row"].is_unique, f"the {win_prob.STAGE_NAME} table {name} lists a row more than once"
    mine = labels.iloc[np.flatnonzero(labels["innings"].to_numpy() == innings)]
    rows = mine.merge(wp[["row", "wp_raw"]], on="row", how="left", validate="one_to_one")
    eligible = rows.iloc[np.flatnonzero(rows["eligible"].to_numpy(dtype=bool))]
    assert len(eligible), f"the {win_prob.LABELS_TABLE} table marks no Delivery of innings {innings} as eligible"

    dropped = eligible["result_type"].astype(str).isin(DROPPED_RESULT_TYPES).to_numpy()
    assert not dropped.any(), (
        f"{int(dropped.sum()):,} eligible Deliveries of innings {innings} lie in matches whose result type "
        f"({sorted(set(eligible['result_type'].astype(str)[dropped]))}) Requirement 10, criterion 2 drops from every metric"
    )
    p = eligible["wp_raw"].to_numpy(dtype=np.float64)
    y = eligible["label"].to_numpy(dtype=np.float64)
    missing = ~np.isfinite(p)
    assert not missing.any(), (
        f"{int(missing.sum()):,} eligible Deliveries of innings {innings} have no finite wp_raw in {name} "
        f"(for example rows {eligible['row'].to_numpy()[missing][:5].tolist()})"
    )
    assert ((p >= 0.0) & (p <= 1.0)).all(), f"{name} holds a wp_raw outside [0, 1] on an eligible Delivery"
    assert np.isin(y, (0.0, 1.0)).all(), f"the {win_prob.LABELS_TABLE} table holds a label other than 0 or 1"
    return p, y


@pytest.mark.requires_outputs(*FILES)
@pytest.mark.parametrize("innings", model_choice.M3_INNINGS)
def test_primary_model_calibration_error_is_at_most_the_config_threshold(
    innings: int, checkpoints: CheckpointStore, settings: Settings, outputs_dir: Path
) -> None:
    """The Calibration_Error of the innings' calibrated Primary_Model, recomputed from its Out_Of_Fold WP on
    the eligible Deliveries, is at most ``tests.calibration_error_max`` and equals the reported value.

    **Validates: Requirements 10.20**
    """
    _require_complete(checkpoints, model_choice.STAGE_NAME, win_prob.STAGE_NAME)
    _require_built_on(checkpoints, model_choice.STAGE_NAME, win_prob.STAGE_NAME)
    model = model_choice.M3_MODELS[innings]
    table = checkpoints.load_table(model_choice.STAGE_NAME, model_choice.COMPARISON_TABLE)
    entry = _primary_entry(table, model)
    candidate = str(entry["candidate"])

    p, y = _scored_rows(checkpoints, innings, candidate)
    curve = _calibration_curve(p, y)
    error = _calibration_error(curve)
    threshold = float(settings.get(THRESHOLD_KEY))
    where = f"{model} Primary_Model {candidate!r} (innings {innings}, {len(p):,} eligible Deliveries)"

    problems = []
    if int(entry["n_rows"]) != len(p):
        problems.append(f"the {model_choice.COMPARISON_TABLE} table scores {where} on {int(entry['n_rows']):,} rows")
    reported = float(entry["calibration_error"])
    if not math.isclose(error, reported, rel_tol=0.0, abs_tol=TABLE_TOLERANCE):
        problems.append(f"the {model_choice.COMPARISON_TABLE} table reports a Calibration_Error of {reported!r} for {where}")
    csv = pd.read_csv(outputs_dir / COMPARISON_CSV)
    at = np.flatnonzero((csv["model"].astype(str) == model).to_numpy() & (csv["candidate"].astype(str) == candidate).to_numpy())
    if len(at) != 1:
        problems.append(f"outputs/{COMPARISON_CSV} has {len(at)} rows for {model} candidate {candidate!r}")
    else:
        row = csv.iloc[at[0]]
        if not bool(row["primary"]):
            problems.append(f"outputs/{COMPARISON_CSV} does not flag {candidate!r} as the {model} Primary_Model")
        written = float(row["calibration_error"])
        if not math.isclose(error, written, rel_tol=0.0, abs_tol=CSV_TOLERANCE):
            problems.append(f"outputs/{COMPARISON_CSV} reports a Calibration_Error of {written!r} for {where}")
    if not error <= threshold:
        problems.append(
            f"the Calibration_Error of {where} exceeds {THRESHOLD_KEY} = {threshold} (Requirement 10, criterion 20)"
        )
    assert not problems, "\n".join(
        [
            f"{where}: recomputed Calibration_Error {error:.6f}",
            *(f"  - {line}" for line in problems),
            "Calibration_Curve (bins that hold a Delivery):",
            curve.to_string(float_format=lambda value: f"{value:.4f}"),
        ]
    )
