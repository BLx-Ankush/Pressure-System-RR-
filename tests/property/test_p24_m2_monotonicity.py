# Feature: pressure-rating-system, Property 24: M2 monotonicity
"""Property 24: M2 monotonicity (Requirement 9, criteria 4 and 11).

**Validates: Requirements 9.4, 9.11**

For any fitted M2 (capped or first-innings) and any Delivery sampled with a fixed seed, predictions
do not decrease as Legal_Balls left steps from 1 to 120 or as RBQ steps from 0 to 1.5 by 0.1, and
do not increase as wickets fallen steps from 0 to 9, with every other feature held at the sampled
value.

Every model checked is a calibrated M2 fit, a ``remaining_runs.M2Model``: the LightGBM Booster
with the Legal_Balls-left calibration factor estimated on the fit's own Training_Rows
(Requirement 9, criterion 15; design DD-15), whose predictions every M2 quantity of the Pipeline
uses.

The check (``monotonicity_problems``)
    For each row and each step sequence of SWEEPS, the row is repeated once per step with that
    feature set to the step and the other eight M2_FEATURES held at the row's values. The frame is
    predicted with ``remaining_runs.predict`` and neighbouring steps are compared exactly, with no
    tolerance: a later step may not predict less (Legal_Balls left, RBQ) or more (wickets fallen)
    than the step before. LightGBM enforces the monotone constraints on every tree it grows, so
    each tree's output is ordered along the stepped feature. Adding the trees in a fixed order in
    float64, and the exp of the Poisson link, are monotone operations under rounding. The
    calibration then multiplies the Booster's prediction (0 where it is below 0) by a factor above
    0 that depends on Legal_Balls left alone and never decreases in it, and rounding a product of
    non-negative floats keeps their order, so the order survives and a tolerance could only hide a
    real violation.

Synthetic fits (``test_m2_fits_are_monotone_on_synthetic_training_sets``; needs no outputs)
    Each example draws a training set of 4-10 matches with 12-24 rows per innings (96-480 rows).
    The nine features cover the ranges of the real V_1 features and hold no missing value, as no
    M2 feature of a real Delivery in V_1 is missing. The targets are counts drawn from a Poisson
    distribution.
    Its mean has a trend in each of Legal_Balls left, wickets fallen and RBQ that follows the
    feature's constraint, ignores the feature or runs against it, plus sine terms that make it
    non-monotone in all three and effects of the six free features, so an unconstrained fit would
    often break the property. The capped-style fit trains on both innings with the innings-2
    targets capped at a drawn runs needed (at 0 when that is 0 or less); the first-innings-style
    fit trains on the innings-1 rows, uncapped.

    Both fits go through ``fit_m2`` with a setup built from ``m2_setup(settings)``, so they keep
    the Config's monotone_constraints (M2_MONOTONE), monotone_constraints_method, L2 penalty, seed,
    determinism and early stopping, and each is calibrated by Legal_Balls left on its own rows,
    the calibration step of every M2 fit. The drawn Legal_Balls-left values leave gaps between 1
    and 120 and few rounds leave the Boosters far from the targets, so the factors vary (3 or
    more distinct factors in about half of the fits), pool violating values (into one factor in
    a quarter to a third of the fits) and are read between and beyond the fitted values. Only
    the settings that size a fit are changed, so that a fit
    on a few hundred rows is fast and grows real trees: 1 thread, and drawn values of the
    objective (one of M2_OBJECTIVES), 1-12 boosting rounds, 1-10 minimum rows per leaf (the
    Config's 200 would allow almost no split), the Config's or a smaller number of leaves and the
    Config's or a larger learning rate. With the Config's early-stopping share (0.2), a fit keeps
    at least 36 rows after its early-stopping matches are held out, so every fit can split. Both
    fits are checked on 1-12 drawn rows: training rows, and fresh rows drawn from wider ranges.

Real fits (criterion 11; need the checkpoints of a Run up to the ``skill`` stage)
    SAMPLE_SIZE Deliveries are drawn without replacement, with SAMPLE_SEED, from the innings 1-2
    Deliveries of fold 1, the rows that the outer fold-1 fits serve. Their features are their
    M2_FEATURES in V_1 (``m2_features(view_1, state)``). One capped and one first-innings fit are
    checked on them, by one of two tests:

    * ``test_saved_outer_fold_1_fits_are_monotone``: the Pipeline's own outer fold-1 fits
      ``m2_capped_lgbm_outer_k1`` and ``m2_first_lgbm_outer_k1``, as the ``remaining_runs`` stage
      saved them (the Booster with ``store.save_model`` under the fit id, which that checkpoint's
      fit registry lists, and its calibration in the table ``m2_lb_calibration``) and as every
      later stage loads them, calibrated, with ``remaining_runs.load_models``. Skipped while the
      ``remaining_runs`` checkpoint is not complete.
    * ``test_self_fitted_outer_fold_1_fits_are_monotone``: the fallback while that checkpoint is
      not complete, skipped once it is. It makes the two outer fold-1 fits as the stage does:
      ``m2_setup(settings)`` unchanged, the V_1 features, the targets of ``m2_targets`` under the
      selected Extras_Encoding, the Training_Rows of ``common.training_mask`` in folds 2-5 (innings
      1 only, uncapped, for the first-innings fit) and ``fit_m2`` with the fit index of outer fit
      k1, which calibrates each fit on those Training_Rows.

``test_the_check_flags_each_broken_direction`` keeps the check honest: it confirms that SWEEPS
holds criterion 11's step sequences in the directions of M2_MONOTONE, that a linear stub that
follows those directions passes, and that reversing any one direction is flagged for exactly that
feature. ``test_the_calibration_pools_ratios_into_a_rising_factor`` checks the calibration step on
hand-computed values: the ratio of summed targets to summed raw predictions per Legal_Balls left,
the weighted pooling of a falling pair, the floor, the step function between and beyond the
fitted values, and that ``remaining_runs.predict`` refuses a bare Booster.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from hypothesis import event, given, note
from hypothesis import strategies as st

from superr import folds, reconstruct, skill
from superr import state as state_stage
from superr.errors import StageError
from superr.folds import FitSpec, fold_of, sandbox_plan
from superr.leakage import FitRegistry, fit_id_for
from superr.models import common
from superr.models.remaining_runs import (
    CAPPED_MODEL,
    FAMILY,
    FIRST_INNINGS_MODEL,
    LB_CALIBRATION_COLUMNS,
    M2_FEATURES,
    M2_MONOTONE,
    M2_OBJECTIVES,
    M2Model,
    fit_m2,
    lb_calibration,
    load_models,
    m2_features,
    m2_setup,
    m2_targets,
    predict,
)
from superr.pipeline import CONSTANTS_NAME, OUTPUTS_FOLDER, checkpoint_layout
from superr.report.constants import ConstantRegistry
from superr.store import MANIFEST_NAME, CheckpointStore
from superr.views import view_table

if TYPE_CHECKING:
    from superr.settings import Settings

# ------------------------------------------------------------------------------------ the check

#: The step sequences of criterion 11: (feature, steps, direction). Direction +1: a later step may
#: not predict less than the step before; -1: it may not predict more.
LB_LEFT_STEPS = np.arange(1, 121, dtype=np.float64)  # Legal_Balls left 1, 2, ..., 120
RBQ_STEPS = np.arange(16, dtype=np.float64) / 10.0  # RBQ 0, 0.1, ..., 1.5
WICKET_STEPS = np.arange(10, dtype=np.float64)  # wickets fallen 0, 1, ..., 9
SWEEPS: tuple[tuple[str, np.ndarray, int], ...] = (
    ("lb_left", LB_LEFT_STEPS, 1),
    ("rbq", RBQ_STEPS, 1),
    ("wickets", WICKET_STEPS, -1),
)


def monotonicity_problems(model: Any, rows: pd.DataFrame, label: str, ids: Sequence[str]) -> list[str]:
    """One line per sequence of SWEEPS along which ``model`` breaks Property 24 on a row of ``rows``.

    ``rows`` holds the M2_FEATURES of the rows to check and ``ids`` names each row. A line names
    the first breaking row and step, with the two predictions and the held features, and counts
    the breaking rows and steps. An empty list means the model passes on every row.
    """
    held = rows.loc[:, list(M2_FEATURES)].reset_index(drop=True)
    n_rows = len(held)
    grids = []
    for feature, steps, _ in SWEEPS:
        grid = held.iloc[np.repeat(np.arange(n_rows), len(steps))].reset_index(drop=True)
        grid[feature] = np.tile(steps, n_rows)
        grids.append(grid)
    predicted = predict(model, pd.concat(grids, ignore_index=True))  # one call: predict has a fixed cost
    ends = np.cumsum([n_rows * len(steps) for _, steps, _ in SWEEPS])
    problems: list[str] = []
    for (feature, steps, direction), part in zip(SWEEPS, np.split(predicted, ends[:-1])):
        pred = part.reshape(n_rows, len(steps))
        broken = direction * np.diff(pred, axis=1) < 0  # exact: no tolerance (module docstring)
        if not broken.any():
            continue
        row, step = (int(i) for i in np.argwhere(broken)[0])
        others = ", ".join(f"{name} {float(held.at[row, name])!r}" for name in M2_FEATURES if name != feature)
        problems.append(
            f"{label}: the prediction {'falls' if direction > 0 else 'rises'} as {feature} steps from "
            f"{steps[step]:g} to {steps[step + 1]:g} ({float(pred[row, step])!r} -> "
            f"{float(pred[row, step + 1])!r}) for {ids[row]}, with {others} held; "
            f"{int(broken.any(axis=1).sum())} of {n_rows} rows and {int(broken.sum())} steps break it"
        )
    return problems


@pytest.fixture(scope="module")
def config_setup(settings: Settings) -> common.BoosterSetup:
    """The Config's M2 setup (``m2_setup``), which every M2 fit of the Pipeline uses."""
    return m2_setup(settings)


# ----------------------------------------------------------------------------- synthetic fits


#: The constrained features in the order of Case.trends.
TREND_FEATURES: tuple[str, ...] = ("lb_left", "wickets", "rbq")

#: A calibration factor at or below this is at remaining_runs' floor (1e-6); only the events read it.
_FLOOR_SEEN = 1e-6


class Case(NamedTuple):
    """One drawn training set and fit setup (module docstring, "Synthetic fits")."""

    seed: int  # seeds the numpy generator that draws the rows, targets and probe rows
    n_matches: int
    balls_per_innings: int
    # Per TREND_FEATURES: 1 when the mean runs follow the feature's constraint, 0 when they ignore
    # it, -1 when they run against it.
    trends: tuple[int, int, int]
    wiggle: float  # amplitude of the sine terms in Legal_Balls left, wickets fallen and RBQ
    free_effects: float  # scale of the effects of the six free features
    objective: str
    max_rounds: int
    min_data_in_leaf: int
    num_leaves: int | None  # None: the Config's value
    learning_rate: float | None  # None: the Config's value
    fit_index: int  # a position in sandbox_plan(5); it seeds the early-stopping split
    n_probe: int


_TREND = st.sampled_from((1, 0, -1))
CASES = st.builds(
    Case,
    seed=st.integers(0, 2**32 - 1),
    n_matches=st.integers(4, 10),
    balls_per_innings=st.integers(12, 24),
    trends=st.tuples(_TREND, _TREND, _TREND),
    wiggle=st.sampled_from((0.0, 5.0, 40.0)),
    free_effects=st.sampled_from((0.0, 10.0)),
    objective=st.sampled_from(M2_OBJECTIVES),
    max_rounds=st.integers(1, 12),
    min_data_in_leaf=st.integers(1, 10),  # 36 or more fitted rows (es_share 0.2) always allow a split
    num_leaves=st.sampled_from((None, 2, 4, 8, 16)),
    learning_rate=st.sampled_from((None, 0.3, 1.0)),
    fit_index=st.integers(0, 24),
    n_probe=st.integers(1, 12),
)


class TrainingSet(NamedTuple):
    """The rows of one Case: features, matches, innings, both targets and the probe rows."""

    X: pd.DataFrame  # M2_FEATURES of every row
    match_id: np.ndarray
    innings: np.ndarray
    remaining: np.ndarray  # remaining runs: the first-innings-style target
    capped: np.ndarray  # innings-2 targets capped at runs needed: the capped-style target
    probe: pd.DataFrame  # the rows the fits are checked on
    probe_ids: list[str]


def _features(rng: np.random.Generator, n_rows: int, widen: float) -> pd.DataFrame:
    """``n_rows`` rows of M2_FEATURES over the ranges of the real V_1 features, widened by ``widen``."""
    columns = {
        "lb_left": rng.integers(1, 121, n_rows),
        "wickets": rng.integers(0, 10, n_rows),
        "rbq": rng.uniform(0.0, 1.6 * widen, n_rows),
        "theta_striker": rng.normal(1.5, 0.15 * widen, n_rows),
        "striker_lb_faced": rng.integers(0, round(72 * widen), n_rows),
        "theta_non_striker": rng.normal(1.5, 0.15 * widen, n_rows),
        "scoring_env": rng.normal(0.0, 0.07 * widen, n_rows),
        "venue_effect": rng.normal(0.05, 0.12 * widen, n_rows),
        "match_day": rng.normal(-0.2, 1.9 * widen, n_rows),
    }
    return pd.DataFrame({name: np.asarray(columns[name], dtype=np.float64) for name in M2_FEATURES})


def _mean_runs(X: pd.DataFrame, case: Case, rng: np.random.Generator) -> np.ndarray:
    """The Poisson mean of each row's remaining runs: 2 or more, often non-monotone in the constrained features.

    The trend is a product of three positive factors, one per constrained feature, each following,
    ignoring or reversing that feature's constraint (Case.trends). With every factor following its
    constraint it is the realistic shape: about 170 runs from the first ball with no wicket down.
    """
    lb, wickets, rbq = (X[name].to_numpy() for name in TREND_FEATURES)
    lb_trend, wicket_trend, rbq_trend = case.trends
    trend = (
        {1: lb, 0: np.full_like(lb, 60.0), -1: 121.0 - lb}[lb_trend]
        * {1: 1.45 - 0.11 * wickets, 0: np.ones_like(wickets), -1: 0.46 + 0.11 * wickets}[wicket_trend]
        * {1: 0.75 + 0.25 * rbq, 0: np.ones_like(rbq), -1: 1.15 - 0.25 * rbq}[rbq_trend]
    )
    wiggle = np.sin(lb / rng.uniform(3, 30)) + np.sin(wickets * rng.uniform(0.5, 3)) + np.sin(rbq * rng.uniform(2, 15))
    free = [name for name, sign in zip(M2_FEATURES, M2_MONOTONE) if sign == 0]
    spread = X[free].std(ddof=0).replace(0.0, 1.0)
    effects = ((X[free] - X[free].mean()) / spread).to_numpy() @ rng.normal(size=len(free))
    mean = trend + case.wiggle * wiggle + case.free_effects * effects
    return mean - mean.min() + 2.0


def _training_set(case: Case) -> TrainingSet:
    """The rows of ``case``: innings 1 and 2 of each match, both targets and the probe rows."""
    rng = np.random.default_rng(case.seed)
    per_match = 2 * case.balls_per_innings
    n_rows = case.n_matches * per_match
    match_id = np.repeat(101 + 10 * np.arange(case.n_matches, dtype=np.int64), per_match)
    innings = np.tile(np.repeat(np.array([1, 2]), case.balls_per_innings), case.n_matches)
    X = _features(rng, n_rows, widen=1.0)
    remaining = rng.poisson(_mean_runs(X, case, rng)).astype(np.float64)
    needed = rng.integers(-5, 200, n_rows)  # runs needed before the Delivery; 0 or less once the Target is reached
    capped = np.where(innings == 2, np.minimum(remaining, np.maximum(needed, 0)), remaining)
    n_seen = (case.n_probe + 1) // 2
    seen = np.sort(rng.choice(n_rows, size=n_seen, replace=False))
    fresh = _features(rng, case.n_probe - n_seen, widen=1.5)
    probe = pd.concat([X.iloc[seen], fresh], ignore_index=True)
    ids = [f"training row {i}" for i in seen.tolist()] + [f"fresh row {i}" for i in range(len(fresh))]
    return TrainingSet(X, match_id, innings, remaining, capped, probe, ids)


def _small_setup(config: common.BoosterSetup, case: Case) -> common.BoosterSetup:
    """The Config's M2 setup with the drawn objective and the drawn settings that size a small fit.

    The monotone constraints, their method and every other setting stay as the Config gives them,
    except the thread count: 1, because on a few hundred rows LightGBM's threads only add start-up
    and waiting time (on the 4-CPU development machine a 30-round fit took about 135 ms with 1
    thread and 190-250 ms with the Config's 4).
    """
    params = config.params
    params.update(objective=case.objective, min_data_in_leaf=case.min_data_in_leaf, num_threads=1)
    if case.num_leaves is not None:
        params["num_leaves"] = case.num_leaves
    if case.learning_rate is not None:
        params["learning_rate"] = case.learning_rate
    return dataclasses.replace(config, param_items=tuple(params.items()), max_rounds=case.max_rounds)


@given(case=CASES)
def test_m2_fits_are_monotone_on_synthetic_training_sets(case: Case, config_setup: common.BoosterSetup) -> None:
    """A capped-style and a first-innings-style fit_m2 fit with the Config's monotone settings do not
    fall as Legal_Balls left or RBQ step up, nor rise as wickets fallen step up, on any drawn row.

    **Validates: Requirements 9.4, 9.11**
    """
    setup = _small_setup(config_setup, case)
    data = _training_set(case)
    fits = (
        ("capped-style fit", np.arange(len(data.X)), data.capped),
        ("first-innings-style fit", np.flatnonzero(data.innings == 1), data.remaining),
    )
    problems: list[str] = []
    for label, rows, y in fits:
        model = fit_m2(data.X.iloc[rows], y[rows], data.match_id[rows], setup, fit_index=case.fit_index)
        assert isinstance(model, M2Model), f"fit_m2 returned a {type(model).__name__}, not a calibrated M2Model"
        splits = dict(zip(M2_FEATURES, model.booster.feature_importance(importance_type="split").tolist()))
        factor = model.calibration["factor"].to_numpy()
        note(
            f"{label}: {len(rows)} rows, best iteration {model.best_iteration}, splits per feature {splits}, "
            f"{len(factor)} Legal_Balls-left values with factors {factor.min():.4g}-{factor.max():.4g}"
        )
        constrained = sum(splits[feature] > 0 for feature, _, _ in SWEEPS)
        event(f"{label}: splits on {constrained} of the 3 constrained features")
        distinct = len(np.unique(factor))
        event(f"{label}: {distinct if distinct < 3 else '3 or more'} distinct calibration factor(s)")
        if factor.min() <= _FLOOR_SEEN:
            event(f"{label}: a calibration factor at the floor")
        problems += monotonicity_problems(model, data.probe, label, data.probe_ids)
    event(f"objective {case.objective}")
    event(f"{case.trends.count(-1)} of the 3 trends run against the constraints")
    assert not problems, "\n".join(problems)


# ---------------------------------------------------------------------------------- real fits

#: The fold whose outer fits the real-data tests check, and the sample of its Deliveries.
FOLD_K = 1
MIN_SAMPLE_SIZE = 100  # criterion 11: at least 100 Deliveries
SAMPLE_SIZE = 200
SAMPLE_SEED = 24

#: The Pipeline_Stage that fits M2 and saves its fold models.
M2_STAGE = "remaining_runs"

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / OUTPUTS_FOLDER
# Only for the checkpoint paths of the requires_outputs marks; the tests read through `checkpoints`.
_PATHS = CheckpointStore(OUTPUTS, checkpoint_layout(ROOT), config_sha256=hashlib.sha256(b"").hexdigest())


def _checkpoint_files(stage: str, *names: str) -> tuple[str, ...]:
    """The manifest and the files ``names`` of a stage's checkpoint, relative to outputs/."""
    folder = _PATHS.stage_dir(stage).relative_to(OUTPUTS)
    return tuple((folder / name).as_posix() for name in (MANIFEST_NAME, *names))


#: Checkpoint files that both real-data tests read, and those that only the self-fitted test reads.
DELIVERY_FILES = (
    *_checkpoint_files(reconstruct.STAGE_NAME, f"{reconstruct.BALLS_TABLE}.parquet"),
    *_checkpoint_files(state_stage.STAGE_NAME, f"{state_stage.STATE_TABLE}.parquet"),
    *_checkpoint_files(folds.STAGE_NAME, f"{folds.FOLD_TABLE}.parquet"),
    *_checkpoint_files(skill.STAGE_NAME, f"{view_table(FOLD_K)}.parquet"),
)
TARGET_FILES = _checkpoint_files(
    reconstruct.STAGE_NAME, f"{reconstruct.INNINGS_INFO_TABLE}.parquet", f"{CONSTANTS_NAME}.json"
)


class Deliveries(NamedTuple):
    """The innings 1-2 Deliveries in the order of the state table, with their V_1 features and the sample."""

    X: pd.DataFrame  # M2_FEATURES from V_1 and the state table (m2_features)
    balls: pd.DataFrame  # the balls rows of the same Deliveries, in the same order, on a RangeIndex
    fold: np.ndarray  # the fold of each Delivery
    fold_table: pd.DataFrame
    sampled: np.ndarray  # positions of the SAMPLE_SIZE fold-K Deliveries drawn with SAMPLE_SEED


@pytest.fixture(scope="module")
def checkpoints(settings: Settings, outputs_dir: Path, workspace_root: Path) -> CheckpointStore:
    """The Run's checkpoints, read through the store, which checks each file against its manifest."""
    return CheckpointStore(outputs_dir, checkpoint_layout(workspace_root), config_sha256=settings.effective_sha256())


@pytest.fixture(scope="module")
def outer_fit(settings: Settings) -> FitSpec:
    """The FitSpec of the outer fit of fold FOLD_K in sandbox_plan(D15)."""
    plan = sandbox_plan(settings.get("design_choices.D15_n_folds"))
    return next(spec for spec in plan if spec.sandbox_k == FOLD_K and spec.inner_j is None)


@pytest.fixture(scope="module")
def deliveries(checkpoints: CheckpointStore) -> Deliveries:
    """Every innings 1-2 Delivery with its V_1 features, and the fixed-seed sample of fold-1 Deliveries."""
    state = checkpoints.load_table(state_stage.STAGE_NAME, state_stage.STATE_TABLE)
    view = checkpoints.load_table(skill.STAGE_NAME, view_table(FOLD_K))
    balls = checkpoints.load_table(reconstruct.STAGE_NAME, reconstruct.BALLS_TABLE)
    fold_table = checkpoints.load_table(folds.STAGE_NAME, folds.FOLD_TABLE)
    X = m2_features(view, state)  # raises unless V_1 and the state table hold the same rows in the same order
    at = pd.Index(balls["row"]).get_indexer(state["row"])
    assert (at >= 0).all(), f"{int((at < 0).sum())} rows of the state table are not in the balls table"
    balls12 = balls.iloc[at].reset_index(drop=True)
    fold = fold_of(balls12["match_id"], fold_table)
    served = np.flatnonzero(fold == FOLD_K)
    sampled = np.sort(np.random.default_rng(SAMPLE_SEED).choice(served, size=SAMPLE_SIZE, replace=False))
    return Deliveries(X, balls12, fold, fold_table, sampled)


def _check_outer_fits(models: Mapping[str, Any], deliveries: Deliveries, spec: FitSpec, source: str) -> None:
    """Property 24 for the outer fold-K capped and first-innings fits on the sampled Deliveries."""
    expected = {fit_id_for(name, spec, family=FAMILY) for name in (CAPPED_MODEL, FIRST_INNINGS_MODEL)}
    assert set(models) == expected, f"checking {sorted(models)}; criterion 11 needs one capped and one first-innings fit"
    sampled = deliveries.sampled
    assert len(sampled) >= MIN_SAMPLE_SIZE, f"{len(sampled)} sampled Deliveries; criterion 11 needs {MIN_SAMPLE_SIZE}"
    balls = deliveries.balls.iloc[sampled]
    ids = [
        f"Delivery row {row} (match {match}, innings {innings})"
        for row, match, innings in zip(balls["row"].tolist(), balls["match_id"].tolist(), balls["innings"].tolist())
    ]
    rows = deliveries.X.iloc[sampled]
    problems = [
        line for fit_id, model in models.items() for line in monotonicity_problems(model, rows, f"{fit_id} ({source})", ids)
    ]
    assert not problems, "\n".join(problems)


@pytest.mark.requires_outputs(*DELIVERY_FILES)
def test_saved_outer_fold_1_fits_are_monotone(
    checkpoints: CheckpointStore, deliveries: Deliveries, outer_fit: FitSpec
) -> None:
    """The Pipeline's saved outer fold-1 capped and first-innings M2 fits pass Property 24 on SAMPLE_SIZE
    fold-1 Deliveries drawn with SAMPLE_SEED.

    **Validates: Requirements 9.4, 9.11**
    """
    if not checkpoints.is_complete(M2_STAGE):
        pytest.skip(
            f"the {M2_STAGE} checkpoint is not complete ({checkpoints.incomplete_reason(M2_STAGE)}); "
            "test_self_fitted_outer_fold_1_fits_are_monotone checks outer fold-1 fits made from V_1 instead"
        )
    registry = FitRegistry(deliveries.fold_table)
    registry.load_saved(checkpoints, M2_STAGE)  # each saved fit is checked again against the Fold_Table
    fit_ids = [fit_id_for(model_name, outer_fit, family=FAMILY) for model_name in (CAPPED_MODEL, FIRST_INNINGS_MODEL)]
    for model_name, fit_id in zip((CAPPED_MODEL, FIRST_INNINGS_MODEL), fit_ids):
        assert fit_id in registry, f"the fit registry of the {M2_STAGE} checkpoint does not list {fit_id!r}"
        record = registry.get(fit_id)
        facts = (record.model, record.family, record.sandbox_k, record.inner_j, record.excluded_folds)
        assert facts == (model_name, FAMILY, FOLD_K, None, frozenset({FOLD_K})), (
            f"fit {fit_id!r} is registered with (model, family, sandbox_k, inner_j, excluded_folds) = {facts}"
        )
    try:
        models = load_models(checkpoints, fit_ids)  # the Booster and its calibration, as later stages load them
    except StageError as error:
        pytest.fail(f"the complete {M2_STAGE} checkpoint holds no usable calibrated M2 fit {fit_ids}: {error}")
    for fit_id, model in models.items():
        assert isinstance(model, M2Model), f"load_models gave a {type(model).__name__} for {fit_id!r}"
        booster = model.booster
        assert isinstance(booster, lgb.Booster), f"the model saved as {fit_id!r} is a {type(booster).__name__}"
        assert booster.feature_name() == list(M2_FEATURES), (
            f"the model saved as {fit_id!r} has the features {booster.feature_name()}; M2 fits use M2_FEATURES in order"
        )
        balls_left = model.calibration["lb_left"].tolist()
        assert balls_left[0] <= LB_LEFT_STEPS[0] and balls_left[-1] >= LB_LEFT_STEPS[-1], (
            f"the calibration of {fit_id!r} covers Legal_Balls left {balls_left[0]}-{balls_left[-1]} only"
        )
    _check_outer_fits(models, deliveries, outer_fit, f"saved by the {M2_STAGE} stage, calibrated")


@pytest.mark.requires_outputs(*DELIVERY_FILES, *TARGET_FILES)
def test_self_fitted_outer_fold_1_fits_are_monotone(
    checkpoints: CheckpointStore,
    deliveries: Deliveries,
    outer_fit: FitSpec,
    config_setup: common.BoosterSetup,
    settings: Settings,
) -> None:
    """Outer fold-1 capped and first-innings M2 fits, made here from V_1 with the Config as the
    ``remaining_runs`` stage makes them, pass Property 24 on SAMPLE_SIZE fold-1 Deliveries drawn with
    SAMPLE_SEED.

    **Validates: Requirements 9.4, 9.11**
    """
    if checkpoints.is_complete(M2_STAGE):
        pytest.skip(
            f"the {M2_STAGE} checkpoint is complete, so test_saved_outer_fold_1_fits_are_monotone checks the "
            "Pipeline's own outer fold-1 fits"
        )
    constants = ConstantRegistry.from_settings(settings)
    constants.load_dict(checkpoints.load_json(reconstruct.STAGE_NAME, CONSTANTS_NAME))
    encoding = constants.value(reconstruct.SELECTED_ENCODING_CONSTANT)
    innings_info = checkpoints.load_table(reconstruct.STAGE_NAME, reconstruct.INNINGS_INFO_TABLE)
    balls = deliveries.balls
    match_ids = balls["match_id"].to_numpy()
    eligible = common.training_mask(balls) & np.isin(deliveries.fold, outer_fit.train_folds)
    variants = (
        (CAPPED_MODEL, True, np.ones(len(balls), dtype=bool)),
        (FIRST_INNINGS_MODEL, False, balls["innings"].to_numpy() == 1),
    )
    models: dict[str, Any] = {}
    for model_name, capped, kept in variants:
        targets = m2_targets(balls, innings_info, capped=capped, encoding=encoding)
        assert np.array_equal(targets["row"].to_numpy(), balls["row"].to_numpy()), (
            "m2_targets returned the Deliveries in another order than the state table"
        )
        y = targets["target"].to_numpy()
        rows = np.flatnonzero(eligible & kept & np.isfinite(y))
        models[fit_id_for(model_name, outer_fit, family=FAMILY)] = fit_m2(
            deliveries.X.iloc[rows], y[rows], match_ids[rows], config_setup, fit_index=common.plan_index(outer_fit)
        )
    _check_outer_fits(models, deliveries, outer_fit, "fitted and calibrated here from V_1 with the Config")


# ------------------------------------------------------------------------ the check, checked


class _LinearModel:
    """A regression stub: 100 plus a weighted sum of features (the design's Property 25 uses stubs too)."""

    def __init__(self, weights: Mapping[str, float]) -> None:
        self.weights = dict(weights)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return 100.0 + sum(weight * X[name].to_numpy() for name, weight in self.weights.items())


def test_the_check_flags_each_broken_direction() -> None:
    """SWEEPS holds criterion 11's steps in the directions of M2_MONOTONE (criterion 4), and the check
    passes a model that follows them and flags exactly the one direction a model reverses."""
    assert LB_LEFT_STEPS.tolist() == list(range(1, 121))
    assert RBQ_STEPS.tolist() == [round(0.1 * k, 1) for k in range(16)]
    assert WICKET_STEPS.tolist() == list(range(10))
    directions = {name: sign for name, sign in zip(M2_FEATURES, M2_MONOTONE) if sign}
    assert {feature: direction for feature, _, direction in SWEEPS} == directions == {"lb_left": 1, "wickets": -1, "rbq": 1}

    rows = _features(np.random.default_rng(SAMPLE_SEED), 3, widen=1.0)
    ids = [f"row {i}" for i in range(len(rows))]
    assert monotonicity_problems(_LinearModel(directions), rows, "stub", ids) == []
    for feature, direction in directions.items():
        problems = monotonicity_problems(_LinearModel({**directions, feature: -direction}), rows, "stub", ids)
        assert len(problems) == 1 and f" as {feature} steps from " in problems[0], problems


def test_the_calibration_pools_ratios_into_a_rising_factor() -> None:
    """The calibration step on hand-computed values (Requirement 9, criterion 15): per Legal_Balls left
    the ratio of summed targets to summed raw predictions, isotonic in Legal_Balls left with the raw
    sums as weights, at least the floor, read as a step function that is flat beyond the fitted
    values; and remaining_runs.predict refuses a bare Booster."""
    # Legal_Balls left 1, 2, 4, 6: raw sums 8, 2, 10, 5 and target sums 4, 2, 8, 6, so ratios 0.5, 1.0,
    # 0.8, 1.2. The falling pair at 2 and 4 pools to (2 + 8) / (2 + 10).
    table = lb_calibration([4.0, 4.0, 2.0, 6.0, 4.0, 5.0], [1.0, 3.0, 2.0, 5.0, 3.0, 6.0], [1, 1, 2, 4, 4, 6])
    assert list(table.columns) == list(LB_CALIBRATION_COLUMNS)
    assert table["lb_left"].tolist() == [1, 2, 4, 6] and table["n_rows"].tolist() == [2, 1, 2, 1]
    np.testing.assert_allclose(table["ratio"], [0.5, 1.0, 0.8, 1.2], rtol=0, atol=1e-12)
    np.testing.assert_allclose(table["factor"], [0.5, 10 / 12, 10 / 12, 1.2], rtol=0, atol=1e-12)
    # A raw prediction below 0 counts as 0; a value whose raw predictions sum to 0 has no ratio and takes
    # the factor of the value below it; a factor of 0 is raised to a tiny positive floor.
    negative = lb_calibration([-1.0, 3.0, 0.0, 2.0], [0.0, 2.0, 1.0, 1.0], [1, 1, 2, 3])
    assert np.isnan(negative["ratio"][1])
    np.testing.assert_allclose(negative["factor"], [0.6, 0.6, 0.6], rtol=0, atol=1e-12)
    floored = lb_calibration([3.0, 1.0], [0.0, 2.0], [1, 2])["factor"].to_numpy()
    assert 0.0 < floored[0] <= _FLOOR_SEEN and floored[1] == 2.0

    model = M2Model(_LinearModel({"lb_left": 1.5, "wickets": -1.0}), table)
    X = pd.DataFrame({name: np.zeros(8) for name in M2_FEATURES}).assign(lb_left=[0, 1, 2, 3, 4, 5, 6, 120])
    np.testing.assert_allclose(model.factor(X["lb_left"]), [0.5, 0.5, *[10 / 12] * 4, 1.2, 1.2], rtol=0, atol=1e-12)
    np.testing.assert_allclose(predict(model, X), model.factor(X["lb_left"]) * model.predict_raw(X), rtol=0, atol=1e-12)
    probe = _features(np.random.default_rng(SAMPLE_SEED), 3, widen=1.0)
    assert monotonicity_problems(model, probe, "M2Model", [f"row {i}" for i in range(len(probe))]) == []

    booster = lgb.train(
        {"objective": "poisson", "verbose": -1, "num_threads": 1, "min_data_in_leaf": 1},
        lgb.Dataset(X[list(M2_FEATURES)], label=np.arange(1.0, 9.0)),
        num_boost_round=1,
    )
    with pytest.raises(TypeError, match="bare LightGBM Booster"):
        predict(booster, X)
