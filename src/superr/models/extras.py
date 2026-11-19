"""Extras_Model: the Wide and No_Ball probabilities of every innings 1-2 Delivery for an average
bowler in the Delivery's pre-ball state (Requirement 11, criteria 4 and 16; design decision DD-11
and sections "Outcome_Model M4 and Extras_Model (``models/outcome.py``, ``models/extras.py``)" and
"Fold sandboxes and leakage control").

The ``outcome`` Pipeline_Stage (``models.outcome.run_stage``) fits this model next to M4 and saves
its outputs in the same checkpoint (STAGE_NAME). The ``value`` stage reads PREDS_TABLE for the
bowler Expected_Value (Requirement 15, criteria 5 and 6), and the Robustness_Suite re-predicts
with the saved fold models (Requirement 21, criterion 1). Every function receives the Config
values it needs as arguments, through the ExtrasSetup that ``extras_setup`` builds from
``models.extras`` and ``seeds.extras``: no function reads ``config.py`` or has a default value for
a Config setting.

Rows (``extras_rows``, ExtrasRows)
    The model works on the rows of the ``state`` table: every Delivery of innings 1 and 2, in the
    Requirement 5 order. ``extras_rows(balls, state, fold_table)`` aligns the ``balls`` table of
    the ``reconstruct`` checkpoint to those rows and returns, per row: ``row``, ``match_id``,
    ``fold`` (from the Fold_Table), ``innings``, the pre-ball ``phase`` and innings x Phase
    ``cell`` (from ``state``), ``label`` (the Delivery's class), ``trainable``
    (``common.training_mask``: innings 1-2, outside a Super_Over_Innings, in a match that the
    Score_Tolerance rule keeps), ``free_hit`` and the striker's ``batting_type``. Unlike M4, every
    class trains: the ``trainable`` rows of a fit's train folds, Legal_Balls, Wides and No_Balls
    alike, are its Training_Rows, and the ``trainable`` rows of each fold k are the held-out rows
    that the metrics evaluate with the outer fit of fold k.

Classes (CLASSES)
    ``legal``, ``wide`` and ``no_ball``, in the column order of every probability array; a row's
    label is the class of its DeliveryClass code (``balls.cls``: Legal_Ball 0, Wide 1, No_Ball 2).
    As delivered, the 189,648 ``trainable`` rows hold 181,591 Legal_Balls, 7,271 Wides (3.8%) and
    786 No_Balls (0.4%).

Features (EXTRAS_FEATURES, ``extras_features``; criterion 16)
    ``extras_features(view, state, rows)`` gives the EXTRAS_FEATURES of every row, in this order:

    ============== ======================================================= ==========
    Feature        Value                                                   Source
    ============== ======================================================= ==========
    cell           innings x Phase cell 0-5 (categorical)                  ``state``
    lb_bowled      Legal_Balls bowled before the Delivery                  ``state``
    lb_left        Legal_Balls left                                        ``state``
    wickets        wickets fallen                                          ``state``
    free_hit       the Free_Hit flag, 0 or 1                               ``balls``
    theta_striker  the striker's theta_bat                                 view V_k
    batting_type   the striker's ``batting_type`` (categorical)            ``balls``
    scoring_env    Scoring_Environment                                     view V_k
    venue_effect   Venue_Effect                                            view V_k
    match_day      Match_Day_Conditions                                    view V_k
    ============== ======================================================= ==========

    No feature describes the bowler (criterion 16; DD-11). BOWLER_FEATURES (bowler identity,
    theta_bowl, ``bowling_type``, Modal_Bowling_Type alone or inside M4's ``hand_x_style``,
    ``pace_or_spin`` and ``bowl_speed``) are never read, so the probabilities describe an average
    bowler in the Delivery's state; ExtrasFoldModels refuses a feature list that names one. The
    theta and context values of fold k's fit come from view V_k, as for M4: a row in fold i takes
    them from M1(-{i, k}), a fold-k row from M1(-k) (Requirement 7, criterion 7). Run
    ``views.check_view`` on V_k before building its frame.

    ``cell`` is int64 and ``batting_type`` text (object dtype), MISSING_CATEGORY for a missing
    value; every other feature is float64, with NaN for a missing value. As delivered, no feature
    is missing on any row: every Delivery has ``batting_type`` RHB or LHB, and the five views hold
    every theta and context value.

Model (``extras_setup``, ``make_estimator``; DD-11)
    A scikit-learn Pipeline: a ColumnTransformer, then a LogisticRegression with the lbfgs solver,
    which fits one multinomial (softmax) model over the three classes, with ``C`` and ``max_iter``
    from ``models.extras`` and ``random_state`` ``seeds.extras``. The ColumnTransformer encodes
    CATEGORICAL_FEATURES one-hot (one indicator per category seen among the Training_Rows; a
    category that the Training_Rows lack gets every indicator 0 at prediction) and standardises
    NUMERIC_FEATURES (mean 0 and standard deviation 1 over the Training_Rows). Every step, the
    encoder, imputer, scaler and regression, is fitted on the fit's Training_Rows only. As
    delivered each fit reads 8 one-hot columns (6 cells, 2 batting types) and 8 standardised
    numbers.

    Missing values: a missing numeric value is replaced by the median of the feature over the
    fit's Training_Rows, and a missing-value indicator (0 or 1) is added for each numeric feature
    with a missing value among the Training_Rows (``SimpleImputer(strategy="median",
    add_indicator=True)``); the imputed columns and the indicators are then standardised like the
    others. A numeric feature that is missing only on predicted rows gets the median and no
    indicator. A missing ``batting_type`` is the category MISSING_CATEGORY, with its own indicator
    when a Training_Row has it. ``cell`` is never missing.

Fits (``fit_extras``, ExtrasFold, ExtrasFoldModels)
    One outer fit per fold and no inner fits: the model's output is never a model feature (design
    table "Fold sandboxes"). ``fit_extras(registry, plan, setup, rows, features)`` runs, for each
    outer FitSpec k of ``folds.sandbox_plan(D15)``, one fit through ``common.fit_registered`` as
    model MODEL without a family, so the fit ids are those of ``leakage.fit_id_for``
    (``extras_outer_k1`` ... ``extras_outer_k5``). The fit trains on the ``trainable`` rows of the
    folds other than k, reading the feature frame built from V_k, uses no early stopping, and is
    registered with every match of fold k as predicted matches before it runs. lbfgs is
    deterministic, so a Run repeats every fit exactly (Requirement 28, criterion 3). A fit that
    stops at ``max_iter`` is kept and reported through ExtrasFold ``converged``.

Predictions (``extras_probabilities``, ``predict_extras``; criterion 4)
    Every innings 1-2 Delivery of fold k is predicted by the outer fit of fold k from its V_k
    features, whatever its class: the probabilities that a Delivery bowled in its pre-ball state
    is a Legal_Ball, a Wide or a No_Ball. ``predict_extras`` returns them as a float64 array of
    shape (rows, N_CLASSES) in CLASSES order, rows in ``row_fold`` order, each row rescaled to sum
    to 1 by ``common.renormalise`` (the softmax rows already sum to 1 within about 1e-16). Each
    probability therefore lies in [0, 1], and the Wide and No_Ball probabilities together are 1
    minus the Legal_Ball probability, at most 1. A class absent from a fit's Training_Rows would
    get probability 0.

Metrics (``cell_baseline``, ``log_loss_table``, ``rates_table``)
    All on the held-out rows: the ``trainable`` rows of every fold, each predicted by the outer fit
    of its fold.

    * ``log_loss_table``: the Pooled_Held_Out_Log_Loss (multiclass, probabilities clipped to
      [1e-15, 1 - 1e-15], one mean over all evaluated rows) of the model and of CELL_BASELINE,
      which predicts every row of fold k with the class frequencies of the ``trainable`` rows of
      the other folds (the Training_Rows of fold k's fit) in the row's innings x Phase cell.
    * ``rates_table``: for each innings x Phase cell and for all cells together, the held-out
      Deliveries, Wides and No_Balls, the actual Wide and No_Ball rates and the mean predicted
      Wide and No_Ball probabilities.

    As delivered, lbfgs converges in 20-21 iterations per fit, and the pooled held-out log loss
    is 0.18745 against 0.18803 for the cell baseline. The features beyond the cell add little,
    as expected for an outcome that depends mostly on the bowler, whom the model may not see. The
    Free_Hit flag is the clearest exception: the 860 held-out Free_Hit Deliveries hold 9.1% Wides
    and 2.3% No_Balls, against 3.8% and 0.4% elsewhere, and the model predicts 9.0% and 2.3%. The
    mean predicted rates match the actual ones within 0.0004 in every cell.

Checkpoint artifacts (``preds_frame``, ``fold_models_frame``, ``save_fold_models``,
``load_fold_models``)
    ``preds_frame`` lays the predictions out as the design's ``extras_preds`` table (PREDS_TABLE,
    PREDS_COLUMNS): ``row`` (int32), ``p_wide`` and ``p_noball`` (float64, each in [0, 1], their
    sum at most 1), then ``fold`` (int8) and ``fit_id`` (str), the outer fit that predicted the
    row, which excludes the row's fold. ``save_fold_models(store, stage, models)`` saves each
    fold's Pipeline under its fit id (``store.save_model``, joblib) and the table FOLD_MODELS_TABLE
    (FOLD_MODEL_COLUMNS: fit id, feature list, Training_Rows per class, lbfgs iterations and
    convergence). ``load_fold_models(store, stage)`` rebuilds the ExtrasFoldModels, for example for
    the Venue_Effect and Match_Day_Conditions ablation, which re-predicts with both features set
    to 0 (task 25.1).

Errors: ValueError or TypeError, naming the problem, for malformed arguments, such as a missing
column, tables whose rows differ from the ``state`` rows, a class code other than those of
DeliveryClass, a feature list that names a bowler feature, or predictions of the wrong shape or
not finite. ``fit_registered`` raises FitLeakageError (exit code 4) when a fit breaks the leakage
guard, and ``folds.fold_of`` DataError (exit code 3) for a match that is not in the Fold_Table.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from superr import folds
from superr.folds import FitSpec
from superr.models import common
from superr.reconstruct import DeliveryClass
from superr.state import STATE_INNINGS, Phase

if TYPE_CHECKING:
    from superr.leakage import FitRegistry
    from superr.settings import Settings
    from superr.store import CheckpointStore

__all__ = [
    "ALL",
    "BOWLER_FEATURES",
    "CATEGORICAL_FEATURES",
    "CELL_BASELINE",
    "CLASSES",
    "EXTRAS_FEATURES",
    "EXTRAS_FOLDER",
    "FOLD_MODELS_FILE",
    "FOLD_MODELS_TABLE",
    "FOLD_MODEL_COLUMNS",
    "LEGAL",
    "LOG_LOSS_COLUMNS",
    "LOG_LOSS_FILE",
    "MISSING_CATEGORY",
    "MODEL",
    "NO_BALL",
    "NUMERIC_FEATURES",
    "N_CLASSES",
    "PREDS_COLUMNS",
    "PREDS_TABLE",
    "RATES_COLUMNS",
    "RATES_FILE",
    "ROW_FEATURES",
    "STAGE_NAME",
    "STATE_FEATURES",
    "VIEW_FEATURES",
    "WIDE",
    "ExtrasFold",
    "ExtrasFoldModels",
    "ExtrasRows",
    "ExtrasSetup",
    "batting_type_values",
    "cell_baseline",
    "extras_features",
    "extras_probabilities",
    "extras_rows",
    "extras_setup",
    "fit_extras",
    "fold_models_frame",
    "load_fold_models",
    "log_loss_table",
    "make_estimator",
    "predict_extras",
    "preds_frame",
    "rates_table",
    "save_fold_models",
]


# ------------------------------------------------------------------------------------- constants

#: The Pipeline_Stage whose checkpoint holds the Extras_Model artifacts: the ``outcome`` stage,
#: which fits M4 and this model (``models.outcome.STAGE_NAME``).
STAGE_NAME = "outcome"

#: Model name of every Extras_Model fit in the FitRegistry. With no family, the fit ids are
#: ``extras_outer_k1`` ... (``leakage.fit_id_for``).
MODEL = "extras"

#: The three classes, in the column order of every probability array, and their indices.
CLASSES: tuple[str, ...] = ("legal", "wide", "no_ball")
N_CLASSES = len(CLASSES)
LEGAL, WIDE, NO_BALL = range(N_CLASSES)

#: DeliveryClass code (``balls.cls``) -> class index.
_CLASS_OF_CODE: Mapping[int, int] = MappingProxyType(
    {int(DeliveryClass.LEGAL): LEGAL, int(DeliveryClass.WIDE): WIDE, int(DeliveryClass.NO_BALL): NO_BALL}
)

#: The features, in the column order of every fit (module docstring, "Features"). Index a
#: DataFrame with ``list(EXTRAS_FEATURES)``: pandas reads a tuple as one column key.
EXTRAS_FEATURES: tuple[str, ...] = (
    "cell", "lb_bowled", "lb_left", "wickets", "free_hit", "theta_striker", "batting_type", "scoring_env",
    "venue_effect", "match_day",
)  # fmt: skip

#: The features encoded one-hot, and the features standardised (after median imputation).
CATEGORICAL_FEATURES: tuple[str, ...] = ("cell", "batting_type")
NUMERIC_FEATURES: tuple[str, ...] = tuple(name for name in EXTRAS_FEATURES if name not in CATEGORICAL_FEATURES)

#: Where extras_features takes each feature: the ``state`` table, the sandbox view, or ExtrasRows (``balls``).
STATE_FEATURES: tuple[str, ...] = ("cell", "lb_bowled", "lb_left", "wickets")
VIEW_FEATURES: tuple[str, ...] = ("theta_striker", "scoring_env", "venue_effect", "match_day")
ROW_FEATURES: tuple[str, ...] = ("free_hit", "batting_type")

#: Feature names that describe the bowler, which the Extras_Model may never read (criterion 16).
BOWLER_FEATURES: tuple[str, ...] = (
    "bowler_id", "theta_bowler", "bowling_type", "modal_bowling_type", "hand_x_style", "pace_or_spin", "bowl_speed",
)  # fmt: skip

#: The ``batting_type`` category of a Delivery whose ``batting_type`` is missing.
MISSING_CATEGORY = "missing"

#: Checkpoint tables: the predictions of every innings 1-2 Delivery, and one row per fold model.
PREDS_TABLE = "extras_preds"
PREDS_COLUMNS: tuple[str, ...] = ("row", "p_wide", "p_noball", "fold", "fit_id")
FOLD_MODELS_TABLE = "extras_fold_models"
FOLD_MODEL_COLUMNS: tuple[str, ...] = (
    "fold", "fit_id", "features", "n_train_rows", *(f"n_{name}" for name in CLASSES), "n_iterations", "converged",
)  # fmt: skip

#: The ``model`` of the baseline row of log_loss_table, and the columns of the two metric tables.
CELL_BASELINE = "cell_baseline"
LOG_LOSS_COLUMNS: tuple[str, ...] = ("model", "n_deliveries", "log_loss")
RATES_COLUMNS: tuple[str, ...] = (
    "innings", "phase", "n_deliveries", "n_wide", "n_no_ball", "wide_rate", "mean_p_wide", "no_ball_rate",
    "mean_p_noball",
)  # fmt: skip
#: The ``innings`` and ``phase`` of the rates_table row over all cells.
ALL = "all"

#: Folder under the outputs folder for the model's files, and their names.
EXTRAS_FOLDER = Path("models") / "extras"
LOG_LOSS_FILE = "log_loss.csv"
RATES_FILE = "rates_by_cell.csv"
FOLD_MODELS_FILE = "fold_models.csv"

_C = "models.extras.C"
_MAX_ITER = "models.extras.max_iter"
_SEED = "seeds.extras"
_MAX_SEED = 2**32 - 1  # the largest random_state scikit-learn accepts
_N_PHASES = len(Phase)
_PHASE_NAMES: Mapping[int, str] = MappingProxyType({int(member): member.name.lower() for member in Phase})
_ROW_COLUMN = "row"
_FEATURE_SEPARATOR = ","
_BALL_COLUMNS: tuple[str, ...] = (
    "row", "match_id", "innings", "cls", "free_hit", "batting_type", "is_super_over", "excluded",
)  # fmt: skip
_STATE_COLUMNS: tuple[str, ...] = ("row", "phase", "cell")
_INT32_MAX = np.iinfo(np.int32).max
_INT8_MAX = np.iinfo(np.int8).max


# ---------------------------------------------------------------------------------- small helpers

def _rows(count: int) -> str:
    return f"{count:,} row" if count == 1 else f"{count:,} rows"


def _require(frame: Any, columns: Iterable[str], what: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{what} must be a DataFrame; got {type(frame).__name__}")
    absent = [column for column in columns if column not in frame.columns]
    if absent:
        raise ValueError(f"{what} lacks the column(s) {', '.join(absent)}")


def _series(values: Any, what: str) -> pd.Series:
    """``values`` (1-D) as a Series on a RangeIndex."""
    if values is None or isinstance(values, (str, bytes, Mapping, pd.DataFrame)) or np.ndim(values) != 1:
        raise TypeError(f"{what} must be a 1-D array or Series; got {type(values).__name__}")
    if isinstance(values, (pd.Series, pd.Index, pd.api.extensions.ExtensionArray)):
        return pd.Series(pd.Series(values).array)
    return pd.Series(np.asarray(values))


def _check_length(array: np.ndarray, what: str, n_rows: int | None) -> np.ndarray:
    if n_rows is not None and len(array) != n_rows:
        raise ValueError(f"{what} has {len(array):,} values, but {n_rows:,} are needed, one per row")
    return array


def _ints(values: Any, what: str, n_rows: int | None = None) -> np.ndarray:
    """``values`` (1-D whole numbers, none missing) as a new int64 array."""
    series = _series(values, what)
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype) or not pd.api.types.is_numeric_dtype(dtype):
        raise TypeError(f"{what} must hold whole numbers; got dtype {dtype}")
    missing = int(series.isna().sum())
    if missing:
        raise ValueError(f"{what} has {missing:,} missing value(s)")
    if pd.api.types.is_integer_dtype(dtype):
        array = series.to_numpy(dtype=np.int64)
    else:
        raw = series.to_numpy(dtype=np.float64)
        whole = np.isfinite(raw) & (np.floor(raw) == raw) & (np.abs(raw) <= 2.0**53)
        if not whole.all():
            raise ValueError(f"{what} must hold whole numbers; {int((~whole).sum()):,} value(s) are not")
        array = raw.astype(np.int64)
    return _check_length(array, what, n_rows)


def _flags(values: Any, what: str, n_rows: int | None = None) -> np.ndarray:
    """``values`` (1-D booleans, none missing) as a new bool array."""
    series = _series(values, what)
    if not pd.api.types.is_bool_dtype(series.dtype):
        raise TypeError(f"{what} must be boolean; got dtype {series.dtype}")
    missing = int(series.isna().sum())
    if missing:
        raise ValueError(f"{what} has {missing:,} missing value(s)")
    return _check_length(series.to_numpy(dtype=bool), what, n_rows)


def _floats(values: Any, what: str, n_rows: int | None = None) -> np.ndarray:
    """``values`` (1-D numbers) as a new float64 array, NaN for a missing value; no infinite value."""
    series = _series(values, what)
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype) or not pd.api.types.is_numeric_dtype(dtype):
        raise TypeError(f"{what} must hold numbers; got dtype {dtype}")
    array = series.to_numpy(dtype=np.float64, na_value=np.nan)
    infinite = int(np.isinf(array).sum())
    if infinite:
        raise ValueError(f"{what} holds {infinite:,} infinite value(s); a feature is a number or missing")
    return _check_length(array, what, n_rows)


def _prob_matrix(values: Any, n_rows: int, what: str) -> np.ndarray:
    """``values`` as a float64 (n_rows, N_CLASSES) array of finite numbers; ValueError otherwise."""
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{what} must be an array of numbers: {exc}") from None
    if array.shape != (n_rows, N_CLASSES):
        raise ValueError(
            f"{what} has shape {array.shape}; ({n_rows:,}, {N_CLASSES}) is needed, one row per Delivery and one "
            "column per class of CLASSES"
        )
    bad = ~np.isfinite(array)
    if bad.any():
        raise ValueError(f"{what} holds {int(bad.sum()):,} missing or non-finite value(s)")
    return array


def _frozen(values: Any, dtype: Any) -> np.ndarray:
    array = np.array(values, dtype=dtype)
    array.flags.writeable = False
    return array


def _whole(value: Any, what: str, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{what} must be a whole number; got {value!r}")
    if value < minimum or (maximum is not None and value > maximum):
        bound = f"from {minimum} to {maximum}" if maximum is not None else f"{minimum} or more"
        raise ValueError(f"{what} must be {bound}; got {value}")
    return int(value)


def _fold_number(value: Any, what: str) -> int:
    return _whole(value, what, 1)


def _check_feature_names(names: Any) -> tuple[str, ...]:
    if isinstance(names, (str, bytes)) or not isinstance(names, Iterable):
        raise TypeError(f"features must be a sequence of column names; got {names!r}")
    out = tuple(names)
    if not out or not all(isinstance(name, str) and name for name in out) or len(set(out)) != len(out):
        raise ValueError(f"features must hold distinct, non-empty column names; got {list(out)}")
    if _FEATURE_SEPARATOR in "".join(out):
        raise ValueError(f"a feature name may not contain {_FEATURE_SEPARATOR!r}; got {list(out)}")
    bowler = [name for name in out if name in BOWLER_FEATURES]
    if bowler:
        raise ValueError(
            f"the Extras_Model may not read the bowler feature(s) {bowler}: its probabilities describe an average "
            "bowler (Requirement 11, criterion 16)"
        )
    return out


def _outer_specs(plan: Any) -> list[FitSpec]:
    if isinstance(plan, (str, bytes, Mapping)) or not isinstance(plan, Iterable):
        raise TypeError("plan must be the FitSpecs of folds.sandbox_plan(D15)")
    specs = list(plan)
    if not all(isinstance(spec, FitSpec) for spec in specs):
        raise TypeError("plan must hold folds.FitSpec objects, as folds.sandbox_plan(D15) returns them")
    outer = [spec for spec in specs if spec.kind == folds.OUTER]
    if not outer:
        raise ValueError("plan holds no outer FitSpec; pass folds.sandbox_plan(D15)")
    ks = [spec.sandbox_k for spec in outer]
    if len(set(ks)) != len(ks):
        raise ValueError(f"plan holds more than one outer FitSpec for a fold: {ks}")
    return outer


def _feature_frame(features: Any, k: int, n_rows: int, names: tuple[str, ...]) -> pd.DataFrame:
    """``features[k]`` with the columns ``names`` in that order; ValueError unless it has ``n_rows`` rows."""
    if not isinstance(features, Mapping) or k not in features:
        raise ValueError(f"features must map fold {k} to the Extras_Model feature frame built from view V_{k}")
    frame = features[k]
    _require(frame, names, f"features[{k}]")
    if len(frame) != n_rows:
        raise ValueError(f"features[{k}] has {_rows(len(frame))}, but {_rows(n_rows)} are needed, one per Delivery")
    return frame[list(names)]


# ============================================================================= rows (ExtrasRows)

@dataclass(frozen=True, eq=False)
class ExtrasRows:
    """One value per innings 1-2 Delivery, in ``state`` order, of everything the model needs besides the views.

    ``row``, ``match_id``, ``fold``, ``innings``, ``phase``, ``cell`` and ``label`` (the class
    index into CLASSES) are int64; ``trainable`` and ``free_hit`` bool; ``batting_type`` an object
    array of text (MISSING_CATEGORY when missing). ``trainable`` marks the Deliveries that may be
    Training_Rows (``common.training_mask``); they are also the held-out rows of the metrics.
    Every array is read-only. Build it with ``extras_rows``.
    """

    row: np.ndarray
    match_id: np.ndarray
    fold: np.ndarray
    innings: np.ndarray
    phase: np.ndarray
    cell: np.ndarray
    label: np.ndarray
    trainable: np.ndarray
    free_hit: np.ndarray
    batting_type: np.ndarray

    def __post_init__(self) -> None:
        kinds = {"trainable": bool, "free_hit": bool, "batting_type": object}
        n_rows = None
        for item in fields(self):
            array = _frozen(getattr(self, item.name), kinds.get(item.name, np.int64))
            if array.ndim != 1:
                raise ValueError(f"ExtrasRows.{item.name} must be 1-D")
            if n_rows is None:
                n_rows = len(array)
            elif len(array) != n_rows:
                raise ValueError(f"ExtrasRows.{item.name} has {len(array):,} values; the other arrays have {n_rows:,}")
            object.__setattr__(self, item.name, array)
        if not np.isin(self.innings, STATE_INNINGS).all():
            raise ValueError("ExtrasRows.innings must be 1 or 2")
        if not np.isin(self.phase, list(_PHASE_NAMES)).all():
            raise ValueError(f"ExtrasRows.phase must hold the Phase codes {list(_PHASE_NAMES)}")
        if (self.cell != (self.innings - 1) * _N_PHASES + self.phase).any():
            raise ValueError("ExtrasRows.cell must be the innings x Phase cell 3 x (innings - 1) + phase of each row")
        if len(self.label) and (self.label.min() < 0 or self.label.max() >= N_CLASSES):
            raise ValueError(f"ExtrasRows.label must hold class indices 0-{N_CLASSES - 1} into CLASSES")
        if not all(isinstance(value, str) for value in self.batting_type.tolist()):
            raise ValueError(f"ExtrasRows.batting_type must hold text, {MISSING_CATEGORY!r} for a missing value")

    @property
    def n_rows(self) -> int:
        """The number of rows."""
        return len(self.row)


def batting_type_values(values: Any) -> np.ndarray:
    """The ``batting_type`` feature of each value: the text, MISSING_CATEGORY for a missing value.

    ``values`` is a 1-D array or Series, such as the ``batting_type`` column of ``balls``. Returns
    an object array of ``str``, one per value.
    """
    texts = _series(values, "batting_type").astype("string")
    return texts.fillna(MISSING_CATEGORY).to_numpy(dtype=object)


def extras_rows(balls: pd.DataFrame, state: pd.DataFrame, fold_table: pd.DataFrame) -> ExtrasRows:
    """The ExtrasRows of every ``state`` row (module docstring, "Rows").

    ``balls`` is the ``balls`` table of the ``reconstruct`` checkpoint (``row``, ``match_id``,
    ``innings``, ``cls``, ``free_hit``, ``batting_type``, ``is_super_over`` and ``excluded``),
    with every Delivery or a superset of the ``state`` rows; ``state`` the ``state`` table
    (``row``, ``phase``, ``cell``); and ``fold_table`` the Fold_Table of the ``folds``
    checkpoint. The result follows the ``state`` rows and their order.

    Raises ValueError when a column is absent, ``balls`` repeats a ``row`` or lacks a ``state``
    row, a ``state`` row lies outside innings 1 and 2, a ``cls`` is not a DeliveryClass code, or a
    value is missing or malformed; TypeError for a column of the wrong kind; and DataError (exit
    code 3) for a match that is not in the Fold_Table.
    """
    _require(balls, _BALL_COLUMNS, "balls")
    _require(state, _STATE_COLUMNS, "state")
    index = pd.Index(_ints(balls["row"], "balls column 'row'"))
    if not index.is_unique:
        raise ValueError("balls must hold each row once")
    row = _ints(state["row"], "state column 'row'")
    at = index.get_indexer(row)
    if (at < 0).any():
        raise ValueError(f"{_rows(int((at < 0).sum()))} of state have no Delivery in balls; pass the tables of one Run")
    part = balls.iloc[at].reset_index(drop=True)
    innings = _ints(part["innings"], "balls column 'innings'")
    outside = ~np.isin(innings, STATE_INNINGS)
    if outside.any():
        raise ValueError(f"{_rows(int(outside.sum()))} of state lie outside innings 1 and 2")
    codes = _ints(part["cls"], "balls column 'cls'")
    label = np.full(len(codes), -1, dtype=np.int64)
    for code, cls in _CLASS_OF_CODE.items():
        label[codes == code] = cls
    unknown = label < 0
    if unknown.any():
        shown = ", ".join(map(str, np.unique(codes[unknown])[:5].tolist()))
        raise ValueError(
            f"{_rows(int(unknown.sum()))} have a cls other than the DeliveryClass codes {sorted(_CLASS_OF_CODE)} "
            f"(values {shown})"
        )
    match_id = _ints(part["match_id"], "balls column 'match_id'")
    n_rows = len(row)
    return ExtrasRows(
        row=row,
        match_id=match_id,
        fold=folds.fold_of(match_id, fold_table),
        innings=innings,
        phase=_ints(state["phase"], "state column 'phase'", n_rows),
        cell=_ints(state["cell"], "state column 'cell'", n_rows),
        label=label,
        trainable=common.training_mask(part),
        free_hit=_flags(part["free_hit"], "balls column 'free_hit'"),
        batting_type=batting_type_values(part["batting_type"]),
    )


# ======================================================================== features (criterion 16)

def extras_features(view: pd.DataFrame, state: pd.DataFrame, rows: ExtrasRows) -> pd.DataFrame:
    """The EXTRAS_FEATURES of every row: the pre-ball state as the Extras_Model reads it (module docstring, "Features").

    ``view`` is a sandbox view V_k of the ``remaining_runs`` checkpoint (VIEW_FEATURES; checked
    with ``views.check_view``), ``state`` the ``state`` table (``row`` and STATE_FEATURES) and
    ``rows`` the ExtrasRows of ``extras_rows``, all with the same rows in the same order; the
    ``row`` columns of ``state`` and, when it has one, of ``view`` must equal ``rows.row``.

    Returns the EXTRAS_FEATURES on a RangeIndex, one row per row: ``cell`` int64, ``batting_type``
    object text, every other feature float64 with NaN for a missing value. Raises ValueError when
    a column is absent, the rows or cells differ, or a value is infinite, and TypeError for a
    column of the wrong kind.
    """
    if not isinstance(rows, ExtrasRows):
        raise TypeError(f"rows must be the ExtrasRows of extras_rows; got {type(rows).__name__}")
    _require(view, VIEW_FEATURES, "view")
    _require(state, (_ROW_COLUMN, *STATE_FEATURES), "state")
    n_rows = rows.n_rows
    for frame, what in ((view, "view"), (state, "state")):
        if len(frame) != n_rows:
            raise ValueError(
                f"{what} has {_rows(len(frame))} but rows has {_rows(n_rows)}; pass a sandbox view, the state table "
                "and their ExtrasRows, which hold one row per innings 1-2 Delivery in the same order"
            )
        if _ROW_COLUMN in frame.columns and not np.array_equal(
            _ints(frame[_ROW_COLUMN], f"{what} column 'row'"), rows.row
        ):
            raise ValueError(f"{what} and rows hold different rows, or the same rows in another order")
    cell = _ints(state["cell"], "state column 'cell'")
    if not np.array_equal(cell, rows.cell):
        raise ValueError("state and rows give different innings x Phase cells; pass the tables of one Run")
    data: dict[str, Any] = {}
    for name in EXTRAS_FEATURES:
        if name == "cell":
            data[name] = cell
        elif name == "batting_type":
            data[name] = pd.Series(rows.batting_type, dtype=object)
        elif name == "free_hit":
            data[name] = rows.free_hit.astype(np.float64)
        elif name in STATE_FEATURES:
            data[name] = _floats(state[name], f"state column {name!r}")
        else:
            data[name] = _floats(view[name], f"view column {name!r}")
    return pd.DataFrame(data, columns=list(EXTRAS_FEATURES))


# =========================================================================== the model (DD-11)

@dataclass(frozen=True)
class ExtrasSetup:
    """The Config settings of the Extras_Model: ``models.extras.C`` (above 0), ``models.extras.max_iter``
    (1 or more) and ``seeds.extras`` (0 to 2^32 - 1). Build it with ``extras_setup``."""

    C: float
    max_iter: int
    seed: int

    def __post_init__(self) -> None:
        if isinstance(self.C, (bool, np.bool_)) or not isinstance(self.C, Real):
            raise TypeError(f"C ({_C}) must be a number; got {self.C!r}")
        if not np.isfinite(self.C) or self.C <= 0:
            raise ValueError(f"C ({_C}) must be a positive number; got {self.C!r}")
        object.__setattr__(self, "C", float(self.C))
        object.__setattr__(self, "max_iter", _whole(self.max_iter, f"max_iter ({_MAX_ITER})", 1))
        object.__setattr__(self, "seed", _whole(self.seed, f"seed ({_SEED})", 0, _MAX_SEED))


def extras_setup(settings: Settings) -> ExtrasSetup:
    """The ExtrasSetup from ``models.extras.C``, ``models.extras.max_iter`` and ``seeds.extras``.

    Raises KeyError for a setting absent from the Config, TypeError for a wrongly typed one and
    ValueError for one out of range.
    """
    return ExtrasSetup(C=settings.get(_C), max_iter=settings.get(_MAX_ITER), seed=settings.get(_SEED))


def make_estimator(setup: ExtrasSetup) -> Any:
    """A new unfitted Extras_Model Pipeline (module docstring, "Model").

    Steps: ``prepare``, a ColumnTransformer that encodes CATEGORICAL_FEATURES one-hot
    (``handle_unknown="ignore"``) and imputes the median (with missing-value indicators) and
    standardises NUMERIC_FEATURES, dropping any other column; then ``logit``, a LogisticRegression
    (lbfgs, multinomial over the three classes) with ``setup.C``, ``setup.max_iter`` and
    ``random_state=setup.seed``.
    """
    if not isinstance(setup, ExtrasSetup):
        raise TypeError(f"setup must be an ExtrasSetup (extras_setup); got {type(setup).__name__}")
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())])
    prepare = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(CATEGORICAL_FEATURES)),
            ("numeric", numeric, list(NUMERIC_FEATURES)),
        ],
        remainder="drop",
    )
    logit = LogisticRegression(C=setup.C, max_iter=setup.max_iter, random_state=setup.seed)
    return Pipeline([("prepare", prepare), ("logit", logit)])


# ======================================================================================== fits

@dataclass(frozen=True, eq=False)
class ExtrasFold:
    """The outer Extras_Model fit of one fold.

    ``fold`` is k; ``fit_id`` the FitRegistry id; ``model`` the fitted Pipeline;
    ``class_counts`` the Training_Rows of each class (length N_CLASSES, CLASSES order,
    read-only); ``n_iterations`` the lbfgs iterations and ``converged`` whether lbfgs stopped
    before ``max_iter``.
    """

    fold: int
    fit_id: str
    model: Any
    class_counts: np.ndarray
    n_iterations: int
    converged: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "fold", _fold_number(self.fold, "ExtrasFold fold"))
        if not isinstance(self.fit_id, str) or not self.fit_id:
            raise TypeError(f"ExtrasFold fit_id must be a non-empty str; got {self.fit_id!r}")
        counts = _frozen(self.class_counts, np.int64)
        if counts.shape != (N_CLASSES,) or (counts < 0).any():
            raise ValueError(f"ExtrasFold class_counts must be {N_CLASSES} counts in CLASSES order")
        object.__setattr__(self, "class_counts", counts)
        object.__setattr__(self, "n_iterations", _whole(self.n_iterations, "ExtrasFold n_iterations", 0))
        if not isinstance(self.converged, (bool, np.bool_)):
            raise TypeError(f"ExtrasFold converged must be True or False; got {self.converged!r}")
        object.__setattr__(self, "converged", bool(self.converged))
        if not callable(getattr(self.model, "predict_proba", None)) or not hasattr(self.model, "classes_"):
            raise TypeError(f"ExtrasFold model must be a fitted classifier pipeline; got {type(self.model).__name__}")

    @property
    def n_train_rows(self) -> int:
        """The fit's Training_Rows."""
        return int(self.class_counts.sum())


@dataclass(frozen=True, eq=False)
class ExtrasFoldModels:
    """The outer fits of the Extras_Model (``fit_extras``, ``load_fold_models``).

    ``features`` are the feature columns of every fit in fit order (none of BOWLER_FEATURES) and
    ``folds`` the ExtrasFold of each fold k (read-only, ascending by k).
    """

    features: tuple[str, ...]
    folds: Mapping[int, ExtrasFold]

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", _check_feature_names(self.features))
        if not isinstance(self.folds, Mapping) or not self.folds:
            raise ValueError("ExtrasFoldModels folds must map at least one fold to its ExtrasFold")
        ordered: dict[int, ExtrasFold] = {}
        for k in sorted(self.folds):
            fold = self.folds[k]
            if not isinstance(fold, ExtrasFold) or fold.fold != k:
                raise ValueError(f"ExtrasFoldModels folds[{k!r}] must be the ExtrasFold of fold {k!r}")
            ordered[int(k)] = fold
        object.__setattr__(self, "folds", MappingProxyType(ordered))

    def fold(self, k: int) -> ExtrasFold:
        """The ExtrasFold of fold ``k``; ValueError when there is no fit for it."""
        if k not in self.folds:
            raise ValueError(f"the Extras_Model has no fit for fold {k}; it covers folds {list(self.folds)}")
        return self.folds[k]

    @property
    def fit_ids(self) -> tuple[str, ...]:
        """The fit ids, by fold."""
        return tuple(fold.fit_id for fold in self.folds.values())

    def fit_ids_for(self, row_fold: Any) -> np.ndarray:
        """The fit id of each row's fold, as an object array of ``str``."""
        ks = _ints(row_fold, "row_fold")
        out = np.empty(len(ks), dtype=object)
        for k in np.unique(ks).tolist():
            out[ks == k] = self.fold(k).fit_id
        return out


def fit_extras(
    registry: FitRegistry,
    plan: Sequence[FitSpec],
    setup: ExtrasSetup,
    rows: ExtrasRows,
    features: Mapping[int, pd.DataFrame],
) -> ExtrasFoldModels:
    """The outer Extras_Model fits, one per outer FitSpec of ``plan`` (module docstring, "Fits").

    ``registry`` is the Run's FitRegistry (``ctx.registry``), ``plan`` the FitSpecs of
    ``folds.sandbox_plan(D15)`` (the inner ones are ignored: the model has none), ``setup`` the
    ExtrasSetup, ``rows`` the ExtrasRows and ``features`` the feature frame of every row built
    from view V_k for each fold k (``extras_features``).

    For outer spec k, ``common.fit_registered`` registers the fit (model MODEL, no family, every
    match of fold k predicted) and then fits ``make_estimator(setup)`` on the ``trainable`` rows
    of the folds other than k. A ConvergenceWarning is not raised; ExtrasFold ``converged``
    reports it. Returns the ExtrasFoldModels. Raises FitLeakageError (exit code 4) when the
    leakage guard rejects a fit, ValueError for a missing frame or column or Training_Rows that
    differ from the ``trainable`` rows of the train folds (a Fold_Table other than the one
    ``rows`` used), TypeError for a malformed argument, and whatever scikit-learn raises, such as
    ValueError for Training_Rows of a single class.
    """
    if not isinstance(setup, ExtrasSetup):
        raise TypeError(f"setup must be an ExtrasSetup (extras_setup); got {type(setup).__name__}")
    if not isinstance(rows, ExtrasRows):
        raise TypeError(f"rows must be the ExtrasRows of extras_rows; got {type(rows).__name__}")
    from sklearn.exceptions import ConvergenceWarning

    names = _check_feature_names(EXTRAS_FEATURES)
    make = functools.partial(make_estimator, setup)
    fitted_folds: dict[int, ExtrasFold] = {}
    for spec in _outer_specs(plan):
        k = spec.sandbox_k
        X = _feature_frame(features, k, rows.n_rows, names)
        expected = np.flatnonzero(rows.trainable & np.isin(rows.fold, spec.train_folds))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)  # reported through ExtrasFold.converged
            fitted = common.fit_registered(
                registry,
                spec,
                model=MODEL,
                family=None,
                match_ids=rows.match_id,
                eligible=rows.trainable,
                predicted_match_ids=rows.match_id[rows.fold == k],
                early_stopping=None,
                fit=common.estimator_fitter(make, X, rows.label),
            )
        if not np.array_equal(fitted.rows.training, expected):
            raise ValueError(
                f"the Training_Rows of fit {fitted.fit_id!r} differ from the trainable rows of folds "
                f"{list(spec.train_folds)}; the FitRegistry and rows must use the same Fold_Table"
            )
        iterations = int(np.max(fitted.model[-1].n_iter_))
        fitted_folds[k] = ExtrasFold(
            fold=k,
            fit_id=fitted.fit_id,
            model=fitted.model,
            class_counts=np.bincount(rows.label[expected], minlength=N_CLASSES),
            n_iterations=iterations,
            converged=iterations < setup.max_iter,
        )
    return ExtrasFoldModels(features=names, folds=fitted_folds)


# ================================================================================ predictions (11.4)

def extras_probabilities(model: Any, X: pd.DataFrame) -> np.ndarray:
    """The class probabilities of a fitted Extras_Model Pipeline for the rows of ``X``, as it gives them.

    Returns a float64 array of shape (rows, N_CLASSES) in CLASSES order: the ``predict_proba``
    columns placed by the model's ``classes_``, and 0 for a class that its Training_Rows lacked.
    Raises ValueError when the model knows a class other than those of CLASSES.
    """
    if not len(X):
        return np.zeros((0, N_CLASSES), dtype=np.float64)
    raw = np.asarray(model.predict_proba(X), dtype=np.float64)
    classes = [int(value) for value in np.asarray(model.classes_).tolist()]
    if any(cls not in range(N_CLASSES) for cls in classes) or raw.shape != (len(X), len(classes)):
        raise ValueError(
            f"the Extras_Model knows the classes {classes} and predicts shape {raw.shape}; it must use the class "
            f"indices 0-{N_CLASSES - 1} of CLASSES"
        )
    out = np.zeros((len(X), N_CLASSES), dtype=np.float64)
    out[:, classes] = raw
    return out


def predict_extras(
    models: ExtrasFoldModels, features: Mapping[int, pd.DataFrame], row_fold: Any, log: common.WarningLog
) -> np.ndarray:
    """The Legal_Ball, Wide and No_Ball probabilities of every row, each row rescaled to sum to 1 (criterion 4).

    ``models`` is an ExtrasFoldModels, ``row_fold`` the fold of every row (``ExtrasRows.fold``)
    and ``features`` the feature frame of every row built from view V_k for each fold k that
    ``row_fold`` holds (``extras_features``; with values changed, for example, by a robustness
    case). The rows of fold k are predicted by the fit of fold k from ``features[k]``. ``log`` is
    the Run_Log (``ctx.log``) or any object with ``warning``; ``common.renormalise`` warns there
    when a row summed to a value more than 1e-6 away from 1.

    Returns a float64 array of shape (rows, N_CLASSES), rows in ``row_fold`` order and columns in
    CLASSES order, every value in [0, 1] and every row summing to 1. Raises ValueError for a fold
    without a fit or a frame, a frame of the wrong length or without a feature, or predictions
    of the wrong shape, not finite or negative.
    """
    if not isinstance(models, ExtrasFoldModels):
        raise TypeError(f"models must be an ExtrasFoldModels (fit_extras); got {type(models).__name__}")
    ks = _ints(row_fold, "row_fold")
    n_rows = len(ks)
    raw = np.full((n_rows, N_CLASSES), np.nan)
    for k in np.unique(ks).tolist():
        fold = models.fold(k)
        served = np.flatnonzero(ks == k)
        X = _feature_frame(features, k, n_rows, models.features).iloc[served]
        raw[served] = _prob_matrix(
            extras_probabilities(fold.model, X), len(served), f"the predictions of fit {fold.fit_id!r}"
        )
    return np.clip(common.renormalise(raw, log, what="Extras_Model class probabilities"), 0.0, 1.0)


# ===================================================================================== metrics

def cell_baseline(rows: ExtrasRows) -> np.ndarray:
    """The innings x Phase cell baseline of every row (module docstring, "Metrics").

    A row of fold k gets the class frequencies of the ``trainable`` rows of the other folds (the
    Training_Rows of fold k's fit) in its innings x Phase cell. Returns a float64 array of shape
    (rows, N_CLASSES) in CLASSES order; a row whose cell has no such Training_Row gets NaN.
    """
    if not isinstance(rows, ExtrasRows):
        raise TypeError(f"rows must be the ExtrasRows of extras_rows; got {type(rows).__name__}")
    out = np.full((rows.n_rows, N_CLASSES), np.nan)
    for k in np.unique(rows.fold).tolist():
        train = rows.trainable & (rows.fold != k)
        target = rows.fold == k
        for cell in np.unique(rows.cell).tolist():
            in_cell = rows.cell == cell
            counts = np.bincount(rows.label[train & in_cell], minlength=N_CLASSES).astype(np.float64)
            if counts.sum() > 0:
                out[target & in_cell] = counts / counts.sum()
    return out


def log_loss_table(probs: Any, rows: ExtrasRows) -> pd.DataFrame:
    """The pooled held-out multiclass log loss of the Extras_Model and of the cell baseline.

    ``probs`` holds the probabilities of every row (shape (rows, N_CLASSES), ``predict_extras``),
    each row from the outer fit of its fold. Both log losses are ``common.pooled_log_loss`` over
    the ``trainable`` rows of all folds. Returns the LOG_LOSS_COLUMNS, one row for MODEL and one
    for CELL_BASELINE. Raises ValueError for probabilities of the wrong shape and a baseline that
    is undefined on an evaluated row.
    """
    if not isinstance(rows, ExtrasRows):
        raise TypeError(f"rows must be the ExtrasRows of extras_rows; got {type(rows).__name__}")
    p = _prob_matrix(probs, rows.n_rows, "probs")
    evaluated = rows.trainable
    labels = rows.label[evaluated]
    baseline = cell_baseline(rows)[evaluated]
    n_rows = int(evaluated.sum())
    return pd.DataFrame(
        {
            "model": [MODEL, CELL_BASELINE],
            "n_deliveries": np.array([n_rows, n_rows], dtype=np.int64),
            "log_loss": [common.pooled_log_loss(labels, p[evaluated]), common.pooled_log_loss(labels, baseline)],
        },
        columns=list(LOG_LOSS_COLUMNS),
    ).astype({"model": "str"})


def rates_table(probs: Any, rows: ExtrasRows) -> pd.DataFrame:
    """Mean predicted against actual Wide and No_Ball rates per innings x Phase cell (module docstring, "Metrics").

    ``probs`` is as for log_loss_table. Over the ``trainable`` rows of all folds, returns the
    RATES_COLUMNS, one row per innings (``"1"``, ``"2"``) and Phase (by name: powerplay, middle,
    death) in that order, then one row with ``innings`` and ``phase`` ALL over every cell: the
    Deliveries, Wides and No_Balls, the actual rates (Wides or No_Balls per Delivery) and the
    mean ``wide`` and ``no_ball`` probabilities. Rates and means are NaN in a cell without a row.
    """
    if not isinstance(rows, ExtrasRows):
        raise TypeError(f"rows must be the ExtrasRows of extras_rows; got {type(rows).__name__}")
    p = _prob_matrix(probs, rows.n_rows, "probs")
    evaluated = rows.trainable
    groups = [
        (str(innings), name, evaluated & (rows.innings == innings) & (rows.phase == code))
        for innings in STATE_INNINGS
        for code, name in _PHASE_NAMES.items()
    ]
    groups.append((ALL, ALL, evaluated))
    records = []
    for innings, phase, mask in groups:
        n_rows = int(mask.sum())
        labels = rows.label[mask]
        n_wide, n_no_ball = int((labels == WIDE).sum()), int((labels == NO_BALL).sum())
        records.append(
            {
                "innings": innings,
                "phase": phase,
                "n_deliveries": n_rows,
                "n_wide": n_wide,
                "n_no_ball": n_no_ball,
                "wide_rate": n_wide / n_rows if n_rows else np.nan,
                "mean_p_wide": float(p[mask, WIDE].mean()) if n_rows else np.nan,
                "no_ball_rate": n_no_ball / n_rows if n_rows else np.nan,
                "mean_p_noball": float(p[mask, NO_BALL].mean()) if n_rows else np.nan,
            }
        )
    table = pd.DataFrame.from_records(records, columns=list(RATES_COLUMNS))
    integers = ("n_deliveries", "n_wide", "n_no_ball")
    return table.astype({"innings": "str", "phase": "str", **{name: np.int64 for name in integers}})


# ======================================================================== checkpoint artifacts

def preds_frame(probs: Any, rows: ExtrasRows, models: ExtrasFoldModels) -> pd.DataFrame:
    """The design's ``extras_preds`` table (PREDS_TABLE; module docstring, "Checkpoint artifacts").

    ``probs`` holds the probabilities of every row (``predict_extras``), ``rows`` the ExtrasRows
    and ``models`` the ExtrasFoldModels that predicted them. Returns PREDS_COLUMNS: ``row``
    (int32), ``p_wide`` and ``p_noball`` (float64), ``fold`` (int8) and ``fit_id`` (str, the fit of
    the row's fold). ``p_noball`` is capped at 1 - ``p_wide``, which changes nothing for rows that
    sum to 1, so their sum is at most 1 exactly (criterion 4). Raises ValueError for probabilities
    of the wrong shape, not finite or outside [0, 1], a fold without a fit, or a ``row`` or fold
    that does not fit in its integer type.
    """
    if not isinstance(rows, ExtrasRows):
        raise TypeError(f"rows must be the ExtrasRows of extras_rows; got {type(rows).__name__}")
    if not isinstance(models, ExtrasFoldModels):
        raise TypeError(f"models must be an ExtrasFoldModels (fit_extras); got {type(models).__name__}")
    p = _prob_matrix(probs, rows.n_rows, "probs")
    outside = (p < 0.0) | (p > 1.0)
    if outside.any():
        raise ValueError(f"probs holds {int(outside.sum()):,} value(s) outside [0, 1]")
    if rows.n_rows and (int(rows.row.min()) < 0 or int(rows.row.max()) > _INT32_MAX):
        raise ValueError("every row id must fit in int32")
    if rows.n_rows and (int(rows.fold.min()) < 1 or int(rows.fold.max()) > _INT8_MAX):
        raise ValueError("every fold must lie from 1 to 127 to fit in int8")
    p_wide = p[:, WIDE].copy()
    p_noball = np.minimum(p[:, NO_BALL], 1.0 - p_wide)
    return pd.DataFrame(
        {
            "row": rows.row.astype(np.int32),
            "p_wide": p_wide,
            "p_noball": p_noball,
            "fold": rows.fold.astype(np.int8),
            "fit_id": pd.array(models.fit_ids_for(rows.fold).astype(str), dtype="str"),
        },
        columns=list(PREDS_COLUMNS),
    )


def fold_models_frame(models: ExtrasFoldModels) -> pd.DataFrame:
    """The FOLD_MODEL_COLUMNS of ``models``, one row per fold (module docstring, "Checkpoint artifacts")."""
    if not isinstance(models, ExtrasFoldModels):
        raise TypeError(f"models must be an ExtrasFoldModels (fit_extras); got {type(models).__name__}")
    records = []
    for k, fold in models.folds.items():
        record: dict[str, Any] = {
            "fold": k,
            "fit_id": fold.fit_id,
            "features": _FEATURE_SEPARATOR.join(models.features),
            "n_train_rows": fold.n_train_rows,
        }
        for cls, name in enumerate(CLASSES):
            record[f"n_{name}"] = int(fold.class_counts[cls])
        record["n_iterations"] = fold.n_iterations
        record["converged"] = fold.converged
        records.append(record)
    table = pd.DataFrame.from_records(records, columns=list(FOLD_MODEL_COLUMNS))
    integers = ("fold", "n_train_rows", *(f"n_{name}" for name in CLASSES), "n_iterations")
    return table.astype(
        {**{name: np.int64 for name in integers}, "fit_id": "str", "features": "str", "converged": bool}
    )


def save_fold_models(store: CheckpointStore, stage: str, models: ExtrasFoldModels) -> None:
    """Save every fold's Pipeline under its fit id (joblib) and the FOLD_MODELS_TABLE table into ``stage``.

    Raises the errors of the store.
    """
    table = fold_models_frame(models)
    for fold in models.folds.values():
        store.save_model(stage, fold.fit_id, fold.model)
    store.save_table(stage, FOLD_MODELS_TABLE, table)


def load_fold_models(store: CheckpointStore, stage: str) -> ExtrasFoldModels:
    """The ExtrasFoldModels that ``save_fold_models`` saved into ``stage``.

    Raises ValueError when the table lacks a column or holds no fold, its folds disagree on the
    features, or a feature list names a bowler feature; StageError (exit code 5) for a checkpoint
    file that is missing, changed or unreadable.
    """
    table = store.load_table(stage, FOLD_MODELS_TABLE)
    _require(table, FOLD_MODEL_COLUMNS, f"checkpoint table {FOLD_MODELS_TABLE!r}")
    if not len(table):
        raise ValueError(f"checkpoint table {FOLD_MODELS_TABLE!r} holds no fold")
    feature_lists = set(table["features"].astype(str))
    if len(feature_lists) != 1:
        raise ValueError(f"the folds of checkpoint table {FOLD_MODELS_TABLE!r} disagree on the feature list")
    features = tuple(next(iter(feature_lists)).split(_FEATURE_SEPARATOR))
    loaded: dict[int, ExtrasFold] = {}
    for record in table.to_dict("records"):
        k = int(record["fold"])
        fit_id = str(record["fit_id"])
        loaded[k] = ExtrasFold(
            fold=k,
            fit_id=fit_id,
            model=store.load_model(stage, fit_id),
            class_counts=[int(record[f"n_{name}"]) for name in CLASSES],
            n_iterations=int(record["n_iterations"]),
            converged=bool(record["converged"]),
        )
    return ExtrasFoldModels(features=features, folds=loaded)
