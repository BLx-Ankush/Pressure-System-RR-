"""Shared helpers of the model stages: Training_Row masks, deterministic parameter builders, the
registered fit, pooled log loss and probability renormalisation (Requirement 5, criteria 11 and 24;
Requirement 7, criteria 5 and 6; Requirement 28, criterion 3).

Design sections "Determinism", "Fold sandboxes and leakage control" and the model components
(Remaining_Runs_Model M2, Win_Probability_Model M3, Outcome_Model M4 and Extras_Model,
Model_Choice_Checker). The stages ``remaining_runs``, ``win_prob``, ``outcome`` and
``model_choice`` use these helpers for every fit of M2, M3, M4, the Extras_Model and the
Alternative_Models. Every function receives the Config values it needs through a ``Settings``
object or as arguments: no function reads ``config.py`` or has a default value for a Config
setting.

Training rows (``training_mask``, ``legal_training_mask``)
    A Delivery may enter a model's Training_Rows only when it lies in innings 1 or 2, is not in a
    Super_Over_Innings (``is_super_over``; criterion 11) and its match is not excluded under the
    Score_Tolerance rule (``excluded``; criterion 24). ``training_mask(balls)`` gives this flag for
    each row of a per-Delivery frame, such as the ``balls`` table of the ``reconstruct``
    checkpoint, as a bool array in row order. ``legal_training_mask`` also requires a Legal_Ball
    (``cls`` 0), for M4 (Requirement 11, criterion 3) and M1 (Requirement 8, criterion 1). A model
    may narrow a mask further (M3 drops matches without a winner, Requirement 10, criterion 2) but
    never widens it.

Parameter builders (``lgbm_setup``, ``xgb_setup``, ``rf_setup``)
    Each builds one family's parameters from the model's Config section, ``runtime.n_threads``,
    ``runtime.deterministic`` and the model's named seed. LGBM_MODELS, XGB_MODELS and RF_MODELS
    give the section and seed of each model key; for example the M3 LightGBM classifier reads
    ``models.m3`` and ``seeds.m3_lgbm``.

    * LightGBM (objectives ``regression``, ``binary`` and ``multiclass``): ``learning_rate``,
      ``num_leaves``, ``min_data_in_leaf``, ``lambda_l2`` and the round cap ``max_rounds``; the
      monotone constraints passed in, with the section's ``monotone_constraints_method``;
      ``num_threads`` and ``seed``. Deterministic mode adds ``deterministic=True`` and
      ``force_row_wise=True``.
    * XGBoost (objectives ``binary`` and ``multiclass``): ``eta``, ``max_depth``,
      ``min_child_weight``, ``reg_lambda``, ``tree_method`` and ``max_rounds``; the monotone
      constraints passed in, in tuple form; ``nthread`` and ``seed``. Deterministic mode requires
      ``tree_method`` "hist".
    * Random-forest classifier: ``n_estimators``, ``max_depth``, ``min_samples_leaf`` and
      ``max_features``; the ``monotonic_cst`` passed in (scikit-learn supports it for binary
      targets only); ``random_state`` and ``n_jobs``. In deterministic mode the fitted forest
      predicts with 1 thread: ``predict_proba`` adds the trees' probabilities from several threads
      in a changing order, which alters the last bits of the result from Run to Run.

    Seeds and thread counts are always set (criterion 3); ``runtime.deterministic`` switches the
    other determinism settings together. Both boosted families carry an EarlyStopping with the
    section's ``es_share`` and ``es_patience`` and the named seed.

The fit wrapper (``fit_registered``)
    ``fit_registered`` runs one fit of a ``folds.FitSpec`` through the leakage guard:

    1. it takes the fold of every candidate row from the Fold_Table, through the FitRegistry
       (Requirement 7, criterion 5), and keeps the eligible rows of the spec's train folds;
    2. for a family with early stopping it holds out whole matches with
       ``leakage.early_stopping_split(..., share, seed, plan_index(spec))``; the other rows are
       the ones the model is fitted on;
    3. it registers the fit with ``FitRegistry.register_spec`` before anything is fitted, so a fit
       whose Training_Rows or predicted matches break the guard stops the Run with
       FitLeakageError (exit code 4) and never runs (criterion 6);
    4. it calls the fit function with the row positions (FitRows);
    5. it returns the fitted model with its fit id and rows (RegisteredFit).

    The fit function is a strategy, so every family fits through the same wrapper:
    ``boosting_fitter`` (LightGBM or XGBoost, early-stopped on the held-out matches),
    ``forest_fitter`` (random forest) and ``estimator_fitter`` (any scikit-learn estimator, such
    as a logistic model or a pipeline that imputes first). A stage may pass its own function of
    FitRows instead, for example one that fits an isotonic calibrator. A model stage::

        plan = folds.sandbox_plan(settings.get("design_choices.D15_n_folds"))
        setup = common.lgbm_setup(settings, "m2", "regression", monotone_constraints=M2_MONOTONE)
        eligible = common.training_mask(balls) & has_target     # one value per row of balls
        for spec in plan:
            X = views[spec.sandbox_k][M2_FEATURES]               # V_k, rows aligned with balls
            fitted = common.fit_registered(
                ctx.registry, spec, model="m2_capped", family="lgbm",
                match_ids=balls["match_id"], eligible=eligible,
                predicted_match_ids=balls["match_id"][row_fold == spec.predicted_fold],
                early_stopping=setup.early_stopping,
                fit=common.boosting_fitter(setup, X, y),
            )
            r_hat = fitted.model.predict(X[row_fold == spec.predicted_fold])   # rows get fitted.fit_id
        ctx.registry.save(ctx.store, STAGE_NAME)

    ``match_ids``, ``eligible``, ``X`` and ``y`` describe the same rows in the same order; the
    FitRows positions index them. The predicted matches are the matches whose Deliveries receive
    the stage's output from the fit, directly or as a feature or calibration input: fold k for an
    outer fit and fold j inside sandbox k for an inner fit. In-fold metrics on Training_Rows are
    not outputs and are not registered.

Metrics and probabilities (``pooled_log_loss``, ``renormalise``)
    ``pooled_log_loss(y, p)`` is the mean, over every row given, of -log q, where q is the
    predicted probability of the observed class clipped to [1e-15, 1 - 1e-15]. Given every
    evaluated Out_Of_Fold_Prediction it is the Pooled_Held_Out_Log_Loss of the glossary: one mean
    over all rows, never a mean of per-fold means. A 1-D ``p`` holds the probability of class 1
    (binary, M3); a 2-D ``p`` holds one row of class probabilities per Delivery and ``y`` the
    class index (multiclass, M4).

    ``renormalise(probs, log)`` rescales every row of class probabilities to sum to 1 in float64
    (Requirement 11, criterion 5). When a row summed to a value more than 1e-6 away from 1 before
    rescaling, it writes one warning to the Run_Log with the count of such rows.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import pandas as pd

from superr import leakage
from superr.folds import FitSpec, sandbox_plan
from superr.reconstruct import DeliveryClass

if TYPE_CHECKING:
    from superr.leakage import FitRegistry
    from superr.settings import Settings

__all__ = [
    "LGBM_MODELS",
    "LIGHTGBM",
    "LOG_LOSS_CLIP",
    "PROB_SUM_TOLERANCE",
    "RF_MODELS",
    "TRAINING_INNINGS",
    "XGBOOST",
    "XGB_MODELS",
    "BoosterSetup",
    "EarlyStopping",
    "FitFunction",
    "FitRows",
    "ForestSetup",
    "RegisteredFit",
    "WarningLog",
    "boosting_fitter",
    "estimator_fitter",
    "fit_registered",
    "forest_fitter",
    "legal_training_mask",
    "lgbm_setup",
    "plan_index",
    "pooled_log_loss",
    "renormalise",
    "rf_setup",
    "training_mask",
    "xgb_setup",
]

# ------------------------------------------------------------------------------------- constants

#: Innings whose Deliveries may train a model (Requirement 5, criterion 11).
TRAINING_INNINGS: tuple[int, ...] = (1, 2)

#: Every predicted probability is clipped to [LOG_LOSS_CLIP, 1 - LOG_LOSS_CLIP] before log loss
#: (glossary, Pooled_Held_Out_Log_Loss).
LOG_LOSS_CLIP = 1e-15

#: A row of class probabilities whose sum differs from 1 by more than this before renormalise is
#: counted in its warning (Requirement 11, criterion 5; Requirement 23, criterion 12).
PROB_SUM_TOLERANCE = 1e-6

#: BoosterSetup.library values.
LIGHTGBM = "lightgbm"
XGBOOST = "xgboost"

#: Config model key -> (Config section, named seed) of each model's LightGBM, XGBoost and
#: random-forest fits.
LGBM_MODELS: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        "m2": ("models.m2", "seeds.m2"),
        "m3": ("models.m3", "seeds.m3_lgbm"),
        "m4": ("models.m4", "seeds.m4_lgbm"),
    }
)
XGB_MODELS: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        "m3": ("models.xgb.m3", "seeds.m3_xgb"),
        "m4": ("models.xgb.m4", "seeds.m4_xgb"),
    }
)
RF_MODELS: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        "m3": ("models.rf.m3", "seeds.m3_rf"),
        "m4": ("models.rf.m4", "seeds.m4_rf"),
    }
)

# Builder objective -> (library objective, early-stopping metric).
_LGBM_OBJECTIVES: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        "regression": ("regression", "l2"),
        "binary": ("binary", "binary_logloss"),
        "multiclass": ("multiclass", "multi_logloss"),
    }
)
_XGB_OBJECTIVES: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        "binary": ("binary:logistic", "logloss"),
        "multiclass": ("multi:softprob", "mlogloss"),
    }
)
_DETERMINISTIC_TREE_METHOD = "hist"
_LGBM_VERBOSITY = -1  # fatal errors only; LightGBM would otherwise print several lines per fit
_XGB_VERBOSITY = 1  # warnings only
_MAX_SEED = 2**31 - 1  # LightGBM reads its seed as a C int
_ES_SET = "early_stopping"  # name of the held-out set in the boosting libraries
_NO_ROWS = np.zeros(0, dtype=np.int64)


# ---------------------------------------------------------------------------------- training rows

def _column(balls: Any, name: str) -> pd.Series:
    if not isinstance(balls, pd.DataFrame):
        raise TypeError(f"balls must be a DataFrame; got {type(balls).__name__}")
    if name not in balls.columns:
        raise ValueError(
            f"balls lacks the column {name!r}; pass the balls table of the reconstruct checkpoint or another "
            f"per-Delivery frame that has it"
        )
    column = balls[name]
    missing = int(column.isna().sum())
    if missing:
        raise ValueError(f"balls column {name!r} has {missing:,} missing value(s); every Delivery needs one")
    return column


def _flags(balls: Any, name: str) -> np.ndarray:
    column = _column(balls, name)
    if not pd.api.types.is_bool_dtype(column.dtype):
        raise TypeError(f"balls column {name!r} must be boolean; got dtype {column.dtype}")
    return column.to_numpy(dtype=bool)


def _numbers(balls: Any, name: str) -> np.ndarray:
    column = _column(balls, name)
    if pd.api.types.is_bool_dtype(column.dtype) or not pd.api.types.is_numeric_dtype(column.dtype):
        raise TypeError(f"balls column {name!r} must be numeric; got dtype {column.dtype}")
    return column.to_numpy(dtype=np.float64)


def training_mask(balls: pd.DataFrame) -> np.ndarray:
    """True for each Delivery that model training may use (Requirement 5, criteria 11 and 24).

    ``balls`` needs ``innings``, ``is_super_over`` and ``excluded``, as the ``balls`` table of the
    ``reconstruct`` checkpoint holds them. A row is True when it lies in innings 1 or 2, is not in
    a Super_Over_Innings and its match is not excluded under the Score_Tolerance rule. Returns a
    bool array with one value per row of ``balls``, in row order (positions, not index labels).

    Raises TypeError when ``balls`` is not a DataFrame or a flag column is not boolean, and
    ValueError when a column is absent or holds a missing value.
    """
    innings = _numbers(balls, "innings")
    super_over = _flags(balls, "is_super_over")
    excluded = _flags(balls, "excluded")
    return np.isin(innings, TRAINING_INNINGS) & ~super_over & ~excluded


def legal_training_mask(balls: pd.DataFrame) -> np.ndarray:
    """``training_mask`` restricted to Legal_Balls (``cls`` 0): the Training_Rows of M4 and M1.

    ``balls`` also needs ``cls`` (the DeliveryClass code). Raises the errors of training_mask.
    """
    return training_mask(balls) & (_numbers(balls, "cls") == int(DeliveryClass.LEGAL))


# ----------------------------------------------------------------------------- argument checks

def _whole(value: Any, what: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{what} must be a whole number; got {value!r}")
    if value < minimum:
        raise ValueError(f"{what} must be {minimum} or more; got {value}")
    return int(value)


def _seed(value: Any, what: str) -> int:
    seed = _whole(value, what, 0)
    if seed > _MAX_SEED:
        raise ValueError(f"{what} must be at most {_MAX_SEED}, the largest seed LightGBM accepts; got {seed}")
    return seed


def _runtime(settings: Settings) -> tuple[int, bool]:
    """``runtime.n_threads`` (1 or more) and ``runtime.deterministic``."""
    n_threads = _whole(settings.get("runtime.n_threads"), "runtime.n_threads", 1)
    deterministic = settings.get("runtime.deterministic")
    if not isinstance(deterministic, (bool, np.bool_)):
        raise TypeError(f"runtime.deterministic must be True or False; got {deterministic!r}")
    return n_threads, bool(deterministic)


def _model_paths(table: Mapping[str, tuple[str, str]], model: Any, family: str) -> tuple[str, str]:
    if isinstance(model, str) and model in table:
        return table[model]
    raise ValueError(f"{family} parameters exist for the model keys {', '.join(map(repr, table))}; got {model!r}")


def _objective(table: Mapping[str, tuple[str, str]], objective: Any, family: str) -> tuple[str, str]:
    if isinstance(objective, str) and objective in table:
        return table[objective]
    raise ValueError(f"a {family} objective must be one of {', '.join(map(repr, table))}; got {objective!r}")


def _num_class(objective: str, num_class: Any) -> int | None:
    if objective == "multiclass":
        if num_class is None:
            raise ValueError("the multiclass objective needs num_class, the number of classes")
        return _whole(num_class, "num_class", 2)
    if num_class is not None:
        raise ValueError(f"num_class applies to the multiclass objective only; got {num_class!r} with {objective!r}")
    return None


def _constraints(values: Any, what: str) -> tuple[int, ...]:
    """Monotone constraints as a tuple of -1, 0 and 1, one per feature."""
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Iterable):
        raise TypeError(f"{what} must be a sequence of -1, 0 and 1, one per feature; got {values!r}")
    out = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or int(value) not in (-1, 0, 1):
            raise ValueError(f"{what} must hold only -1, 0 and 1, one per feature; got {value!r}")
        out.append(int(value))
    if not out:
        raise ValueError(f"{what} must hold one value per feature; got none")
    return tuple(out)


# ------------------------------------------------------------------------------ parameter setups

@dataclass(frozen=True)
class EarlyStopping:
    """How a boosted fit holds out whole matches to decide early stopping (Requirement 10, criterion 7).

    ``share`` is the model's ``es_share`` (above 0 and below 1), ``patience`` its ``es_patience`` in
    rounds (1 or more) and ``seed`` its named seed. fit_registered draws the split of a fit with
    ``leakage.early_stopping_split(training matches, share, seed, plan_index(spec))``.
    """

    share: float
    patience: int
    seed: int

    def __post_init__(self) -> None:
        if isinstance(self.share, (bool, np.bool_)) or not isinstance(self.share, Real):
            raise TypeError(f"EarlyStopping share (es_share) must be a number; got {self.share!r}")
        if not 0.0 < float(self.share) < 1.0:
            raise ValueError(f"EarlyStopping share (es_share) must lie above 0 and below 1; got {self.share}")
        object.__setattr__(self, "share", float(self.share))
        object.__setattr__(self, "patience", _whole(self.patience, "EarlyStopping patience (es_patience)", 1))
        object.__setattr__(self, "seed", _seed(self.seed, "EarlyStopping seed"))


@dataclass(frozen=True)
class BoosterSetup:
    """One LightGBM or XGBoost model's parameters, from lgbm_setup or xgb_setup.

    ``library`` is LIGHTGBM or XGBOOST. ``param_items`` holds the booster parameters as
    (name, value) pairs; ``params`` returns them as a new dict for ``lightgbm.train`` or
    ``xgboost.train``. ``max_rounds`` caps the boosting rounds (``num_boost_round``) and
    ``early_stopping`` gives the held-out share, patience and seed.
    """

    library: str
    param_items: tuple[tuple[str, Any], ...]
    max_rounds: int
    early_stopping: EarlyStopping

    def __post_init__(self) -> None:
        if self.library not in (LIGHTGBM, XGBOOST):
            raise ValueError(f"BoosterSetup library must be {LIGHTGBM!r} or {XGBOOST!r}; got {self.library!r}")
        object.__setattr__(self, "param_items", tuple((str(name), value) for name, value in self.param_items))
        object.__setattr__(self, "max_rounds", _whole(self.max_rounds, "max_rounds", 1))
        if not isinstance(self.early_stopping, EarlyStopping):
            raise TypeError(f"BoosterSetup early_stopping must be an EarlyStopping; got {self.early_stopping!r}")

    @property
    def params(self) -> dict[str, Any]:
        """The booster parameters, as a new dict on every call."""
        return dict(self.param_items)


@dataclass(frozen=True)
class ForestSetup:
    """One random-forest model's parameters, from rf_setup.

    ``param_items`` holds the keyword arguments of ``sklearn.ensemble.RandomForestClassifier`` as
    (name, value) pairs; ``params`` returns them as a new dict. ``predict_n_jobs`` is the thread
    count the fitted forest predicts with: 1 in deterministic mode, ``runtime.n_threads`` otherwise.
    """

    param_items: tuple[tuple[str, Any], ...]
    predict_n_jobs: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "param_items", tuple((str(name), value) for name, value in self.param_items))
        object.__setattr__(self, "predict_n_jobs", _whole(self.predict_n_jobs, "predict_n_jobs", 1))

    @property
    def params(self) -> dict[str, Any]:
        """The RandomForestClassifier keyword arguments, as a new dict on every call."""
        return dict(self.param_items)


def _early_stopping(settings: Settings, section: str, seed: int) -> EarlyStopping:
    return EarlyStopping(
        share=settings.get(f"{section}.es_share"),
        patience=settings.get(f"{section}.es_patience"),
        seed=seed,
    )


def lgbm_setup(
    settings: Settings,
    model: str,
    objective: str,
    *,
    monotone_constraints: Sequence[int] | None = None,
    num_class: int | None = None,
) -> BoosterSetup:
    """The deterministic LightGBM parameters of ``model`` (Requirement 28, criterion 3).

    ``model`` is a key of LGBM_MODELS: ``"m2"`` (section ``models.m2``, seed ``seeds.m2``), ``"m3"``
    (``models.m3``, ``seeds.m3_lgbm``) or ``"m4"`` (``models.m4``, ``seeds.m4_lgbm``). ``objective``
    is ``"regression"``, ``"binary"`` or ``"multiclass"``; the multiclass objective needs
    ``num_class``. ``monotone_constraints`` holds -1, 0 or 1 per feature, in feature order; when
    given, the section's ``monotone_constraints_method`` applies (Requirements 9.4 and 10.5), and a
    section without one raises KeyError. The module docstring lists every parameter.

    Raises KeyError for a setting absent from the Config, ValueError for an unknown model key or
    objective or an out-of-range value, and TypeError for a wrongly typed value.
    """
    section, seed_path = _model_paths(LGBM_MODELS, model, "LightGBM")
    library_objective, metric = _objective(_LGBM_OBJECTIVES, objective, "LightGBM")
    classes = _num_class(objective, num_class)
    n_threads, deterministic = _runtime(settings)
    seed = _seed(settings.get(seed_path), seed_path)
    params: dict[str, Any] = {"objective": library_objective}
    if classes is not None:
        params["num_class"] = classes
    params["metric"] = metric
    for key in ("learning_rate", "num_leaves", "min_data_in_leaf", "lambda_l2"):
        params[key] = settings.get(f"{section}.{key}")
    if monotone_constraints is not None:
        params["monotone_constraints"] = _constraints(monotone_constraints, "monotone_constraints")
        params["monotone_constraints_method"] = settings.get(f"{section}.monotone_constraints_method")
    params["num_threads"] = n_threads
    params["seed"] = seed
    params["deterministic"] = deterministic
    if deterministic:
        params["force_row_wise"] = True
    params["verbosity"] = _LGBM_VERBOSITY
    return BoosterSetup(
        library=LIGHTGBM,
        param_items=tuple(params.items()),
        max_rounds=settings.get(f"{section}.max_rounds"),
        early_stopping=_early_stopping(settings, section, seed),
    )


def xgb_setup(
    settings: Settings,
    model: str,
    objective: str,
    *,
    monotone_constraints: Sequence[int] | None = None,
    num_class: int | None = None,
) -> BoosterSetup:
    """The deterministic XGBoost parameters of Alternative_Model ``model`` (Requirement 28, criterion 3).

    ``model`` is a key of XGB_MODELS: ``"m3"`` (section ``models.xgb.m3``, seed ``seeds.m3_xgb``)
    or ``"m4"`` (``models.xgb.m4``, ``seeds.m4_xgb``). ``objective`` is ``"binary"`` or
    ``"multiclass"``; the multiclass objective needs ``num_class``. ``monotone_constraints`` holds
    -1, 0 or 1 per feature and is passed in tuple form. The module docstring lists every
    parameter.

    Raises ValueError when ``runtime.deterministic`` is on and the section's ``tree_method`` is not
    "hist", and the errors of lgbm_setup otherwise.
    """
    section, seed_path = _model_paths(XGB_MODELS, model, "XGBoost")
    library_objective, metric = _objective(_XGB_OBJECTIVES, objective, "XGBoost")
    classes = _num_class(objective, num_class)
    n_threads, deterministic = _runtime(settings)
    seed = _seed(settings.get(seed_path), seed_path)
    tree_method = settings.get(f"{section}.tree_method")
    if deterministic and tree_method != _DETERMINISTIC_TREE_METHOD:
        raise ValueError(
            f"{section}.tree_method is {tree_method!r}, but runtime.deterministic is on, which needs "
            f"{_DETERMINISTIC_TREE_METHOD!r} for every XGBoost fit (design section Determinism; Requirement 28, "
            "criterion 3)"
        )
    params: dict[str, Any] = {"objective": library_objective}
    if classes is not None:
        params["num_class"] = classes
    params["eval_metric"] = metric
    for key in ("eta", "max_depth", "min_child_weight", "reg_lambda"):
        params[key] = settings.get(f"{section}.{key}")
    params["tree_method"] = tree_method
    if monotone_constraints is not None:
        params["monotone_constraints"] = _constraints(monotone_constraints, "monotone_constraints")
    params["nthread"] = n_threads
    params["seed"] = seed
    params["verbosity"] = _XGB_VERBOSITY
    return BoosterSetup(
        library=XGBOOST,
        param_items=tuple(params.items()),
        max_rounds=settings.get(f"{section}.max_rounds"),
        early_stopping=_early_stopping(settings, section, seed),
    )


def rf_setup(settings: Settings, model: str, *, monotonic_cst: Sequence[int] | None = None) -> ForestSetup:
    """The deterministic random-forest classifier parameters of Alternative_Model ``model``.

    ``model`` is a key of RF_MODELS: ``"m3"`` (section ``models.rf.m3``, seed ``seeds.m3_rf``) or
    ``"m4"`` (``models.rf.m4``, ``seeds.m4_rf``). ``monotonic_cst`` holds -1, 0 or 1 per feature;
    scikit-learn accepts it for binary targets only and for data without missing values. The
    module docstring lists every parameter.

    Raises the errors of lgbm_setup.
    """
    section, seed_path = _model_paths(RF_MODELS, model, "random-forest")
    n_threads, deterministic = _runtime(settings)
    seed = _seed(settings.get(seed_path), seed_path)
    keys = ("n_estimators", "max_depth", "min_samples_leaf", "max_features")
    params: dict[str, Any] = {key: settings.get(f"{section}.{key}") for key in keys}
    if monotonic_cst is not None:
        params["monotonic_cst"] = _constraints(monotonic_cst, "monotonic_cst")
    params["random_state"] = seed
    params["n_jobs"] = n_threads
    return ForestSetup(param_items=tuple(params.items()), predict_n_jobs=1 if deterministic else n_threads)


# -------------------------------------------------------------------------------- the fit wrapper

def _frozen_rows(values: Any, what: str) -> np.ndarray:
    rows = np.array(values, dtype=np.int64)
    if rows.ndim != 1:
        raise ValueError(f"{what} must be a 1-D array of row positions")
    rows.flags.writeable = False
    return rows


@dataclass(frozen=True, eq=False)
class FitRows:
    """The rows of one fit, as ascending positions into the rows given to fit_registered.

    ``train``: the rows the model is fitted on. ``early_stopping``: the rows of the matches held out
    to decide early stopping (empty for a family without early stopping). Both are read-only int64
    arrays and never share a match.
    """

    train: np.ndarray
    early_stopping: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "train", _frozen_rows(self.train, "FitRows train"))
        object.__setattr__(self, "early_stopping", _frozen_rows(self.early_stopping, "FitRows early_stopping"))

    @property
    def training(self) -> np.ndarray:
        """Every Training_Row of the fit: the train and early-stopping rows, ascending."""
        return np.union1d(self.train, self.early_stopping)


#: The strategy that fits one model on the given rows and returns it.
FitFunction = Callable[[FitRows], Any]


@dataclass(frozen=True, eq=False)
class RegisteredFit:
    """A fitted model with the id under which the FitRegistry holds it, and its rows."""

    fit_id: str
    model: Any
    rows: FitRows


def plan_index(spec: FitSpec) -> int:
    """The position of ``spec`` in ``folds.sandbox_plan(D15)``: the ``fit_index`` of its early-stopping split.

    D15 is read off the spec, whose train and excluded folds together are every fold, so outer
    spec k has index k - 1 and the inner specs follow in the order of sandbox_plan. Raises
    TypeError when ``spec`` is not a FitSpec and ValueError when sandbox_plan does not list it.
    """
    if not isinstance(spec, FitSpec):
        raise TypeError(f"spec must be a folds.FitSpec; got {type(spec).__name__}")
    n_folds = len(spec.train_folds) + len(spec.excluded_folds)
    try:
        return sandbox_plan(n_folds).index(spec)
    except ValueError:
        raise ValueError(
            f"FitSpec {spec.fit_id!r} is not a spec of folds.sandbox_plan({n_folds}), so its early-stopping split "
            "has no fit index"
        ) from None


def _row_flags(values: Any, n_rows: int) -> np.ndarray:
    if values is None:
        return np.ones(n_rows, dtype=bool)
    if isinstance(values, (str, bytes, set, frozenset, pd.DataFrame)) or np.ndim(values) != 1:
        raise TypeError("eligible must be a 1-D bool array or Series with one value per row, or None for every row")
    series = values if isinstance(values, pd.Series) else pd.Series(np.asarray(values))
    if not pd.api.types.is_bool_dtype(series.dtype):
        raise TypeError(f"eligible must be boolean; got dtype {series.dtype}")
    if series.isna().any():
        raise ValueError(f"eligible has {int(series.isna().sum()):,} missing value(s)")
    if len(series) != n_rows:
        raise ValueError(f"eligible has {len(series):,} values but match_ids has {n_rows:,} rows")
    return series.to_numpy(dtype=bool)


def fit_registered(
    registry: FitRegistry,
    spec: FitSpec,
    *,
    model: str,
    family: str | None,
    match_ids: Any,
    eligible: Any,
    predicted_match_ids: Any,
    early_stopping: EarlyStopping | None,
    fit: FitFunction,
) -> RegisteredFit:
    """Register one fit with the leakage guard, then run it (module docstring, "The fit wrapper").

    registry: the Run's FitRegistry (``ctx.registry``). spec: the FitSpec of the fit, from
    ``folds.sandbox_plan``. model and family: the names the registry and its messages use; the fit
    id is ``leakage.fit_id_for(model, spec, family=family)``. match_ids: the match id of every
    candidate row (a per-Delivery ``match_id`` column). eligible: a bool per candidate row that
    may train the model (for example ``training_mask(balls)`` and a known target), or None for
    every row. predicted_match_ids: the matches whose Deliveries receive the fit's output; repeats
    are ignored. early_stopping: the family's EarlyStopping, or None for a family without early
    stopping. fit: the strategy, called once with the FitRows; it returns the fitted model.

    The Training_Rows are the eligible rows whose match lies in the spec's train folds, by the
    Fold_Table. With early stopping, the matches of ``early_stopping_split(..., plan_index(spec))``
    become the early-stopping rows. The registry records the fit, with the number of
    Training_Rows (both parts), before ``fit`` runs.

    Raises FitLeakageError (exit code 4) when the guard rejects the fit, in which case ``fit`` is
    not called and nothing is recorded; DataError (exit code 3) for a match id that is missing, not
    a whole number or not in the Fold_Table; ValueError when no Training_Row remains, early
    stopping has fewer than 2 training matches, the spec is not in sandbox_plan, the fit id was
    registered before with other values or ``fit`` returns None; TypeError for a malformed
    argument; and whatever ``fit`` raises.
    """
    if not isinstance(spec, FitSpec):
        raise TypeError(f"spec must be a folds.FitSpec; got {type(spec).__name__}")
    if early_stopping is not None and not isinstance(early_stopping, EarlyStopping):
        raise TypeError(f"early_stopping must be an EarlyStopping or None; got {early_stopping!r}")
    if not callable(fit):
        raise TypeError(f"fit must be a function of FitRows; got {fit!r}")
    if isinstance(match_ids, (str, bytes, set, frozenset, pd.DataFrame)) or np.ndim(match_ids) != 1:
        raise TypeError("match_ids must be a 1-D array or Series with the match id of each candidate row")
    row_folds = registry.fold_of(match_ids)  # the Fold_Table (Requirement 7, criterion 5); checks every id
    ids = pd.Series(match_ids).to_numpy(dtype=np.int64)
    candidates = _row_flags(eligible, len(ids))
    pool = np.flatnonzero(candidates & np.isin(row_folds, np.asarray(spec.train_folds, dtype=row_folds.dtype)))
    label = f"fit {leakage.fit_id_for(model, spec, family=family)!r}"
    if not len(pool):
        raise ValueError(
            f"{label} has no Training_Row: none of its {len(ids):,} candidate rows is eligible and lies in its "
            f"train folds {list(spec.train_folds)}"
        )
    if early_stopping is None:
        train, held = pool, _NO_ROWS
    else:
        _, held_matches = leakage.early_stopping_split(
            ids[pool], early_stopping.share, early_stopping.seed, plan_index(spec)
        )
        held_out = np.isin(ids[pool], held_matches)
        train, held = pool[~held_out], pool[held_out]
    record = registry.register_spec(
        model,
        spec,
        ids[train],
        predicted_match_ids,
        early_stopping_match_ids=None if early_stopping is None else ids[held],
        family=family,
        n_train_rows=int(len(pool)),
    )
    rows = FitRows(train=train, early_stopping=held)
    fitted = fit(rows)
    if fitted is None:
        raise ValueError(f"the fit function of {label} returned None instead of the fitted model")
    return RegisteredFit(fit_id=record.fit_id, model=fitted, rows=rows)


# ---------------------------------------------------------------------------------- fit strategies

def _check_data(X: Any, y: Any) -> None:
    if np.ndim(y) != 1:
        raise ValueError("y must be 1-D, with one target per row")
    if len(X) != len(y):
        raise ValueError(f"X has {len(X):,} rows but y has {len(y):,}")


def _take(values: Any, rows: np.ndarray) -> Any:
    if isinstance(values, (pd.DataFrame, pd.Series)):
        return values.iloc[rows]
    return np.asarray(values)[rows]


def _has_categories(X: Any) -> bool:
    return isinstance(X, pd.DataFrame) and any(isinstance(dtype, pd.CategoricalDtype) for dtype in X.dtypes)


def _no_early_stopping(rows: FitRows, what: str) -> None:
    if len(rows.early_stopping):
        raise ValueError(f"{what} uses no early stopping; call fit_registered with early_stopping=None")


def boosting_fitter(setup: BoosterSetup, X: Any, y: Any) -> FitFunction:
    """The fit function of a LightGBM or XGBoost model with early stopping, for fit_registered.

    ``X`` (a DataFrame or 2-D array) and ``y`` (1-D) hold every candidate row, in the order of the
    ``match_ids`` given to fit_registered. The model trains on ``rows.train`` for at most
    ``setup.max_rounds`` rounds and stops once the setup's metric on ``rows.early_stopping`` has
    not improved for ``patience`` rounds. It returns the model of the best round: a LightGBM
    ``Booster`` whose ``best_iteration`` ``predict`` uses, or an XGBoost ``Booster`` cut to the
    best round, whose ``predict`` and ``inplace_predict`` therefore use the best round. Pandas
    category columns are categorical features in both libraries.

    The returned function raises ValueError when the rows hold no early-stopping row.
    """
    if not isinstance(setup, BoosterSetup):
        raise TypeError(f"setup must be a BoosterSetup (lgbm_setup or xgb_setup); got {type(setup).__name__}")
    _check_data(X, y)

    def fit(rows: FitRows) -> Any:
        if not len(rows.early_stopping):
            raise ValueError(
                "a boosted fit needs early-stopping rows; call fit_registered with early_stopping=setup.early_stopping"
            )
        x_fit, y_fit = _take(X, rows.train), _take(y, rows.train)
        x_held, y_held = _take(X, rows.early_stopping), _take(y, rows.early_stopping)
        patience = setup.early_stopping.patience
        if setup.library == LIGHTGBM:
            import lightgbm as lgb

            fit_set = lgb.Dataset(x_fit, label=y_fit)
            held_set = lgb.Dataset(x_held, label=y_held, reference=fit_set)
            return lgb.train(
                setup.params,
                fit_set,
                num_boost_round=setup.max_rounds,
                valid_sets=[held_set],
                valid_names=[_ES_SET],
                callbacks=[lgb.early_stopping(patience, first_metric_only=True, verbose=False)],
            )
        import xgboost as xgb

        params = setup.params
        categorical = _has_categories(X)
        fit_set = xgb.DMatrix(x_fit, label=y_fit, nthread=params["nthread"], enable_categorical=categorical)
        held_set = xgb.DMatrix(x_held, label=y_held, nthread=params["nthread"], enable_categorical=categorical)
        return xgb.train(
            params,
            fit_set,
            num_boost_round=setup.max_rounds,
            evals=[(held_set, _ES_SET)],
            callbacks=[xgb.callback.EarlyStopping(rounds=patience, save_best=True)],
            verbose_eval=False,
        )

    return fit


def forest_fitter(setup: ForestSetup, X: Any, y: Any) -> FitFunction:
    """The fit function of a random-forest classifier, for fit_registered with ``early_stopping=None``.

    ``X`` and ``y`` are as for boosting_fitter. The forest trains on ``rows.train`` with
    ``setup.params`` and is returned set to predict with ``setup.predict_n_jobs`` threads. The
    returned function raises ValueError when the rows hold early-stopping rows.
    """
    if not isinstance(setup, ForestSetup):
        raise TypeError(f"setup must be a ForestSetup (rf_setup); got {type(setup).__name__}")
    _check_data(X, y)

    def fit(rows: FitRows) -> Any:
        _no_early_stopping(rows, "a random forest")
        from sklearn.ensemble import RandomForestClassifier

        forest = RandomForestClassifier(**setup.params)
        forest.fit(_take(X, rows.train), _take(y, rows.train))
        forest.set_params(n_jobs=setup.predict_n_jobs)
        return forest

    return fit


def estimator_fitter(make_estimator: Callable[[], Any], X: Any, y: Any) -> FitFunction:
    """The fit function of a scikit-learn estimator, for fit_registered with ``early_stopping=None``.

    ``make_estimator`` returns a new unfitted estimator for each fit, for example a
    ``LogisticRegression`` with the Config's ``C``, ``max_iter`` and named seed, or a pipeline that
    imputes training-row medians first. ``X`` and ``y`` are as for boosting_fitter. The estimator
    is fitted on ``rows.train`` and returned. The returned function raises ValueError when the
    rows hold early-stopping rows.
    """
    if not callable(make_estimator):
        raise TypeError(f"make_estimator must be a function that returns an estimator; got {make_estimator!r}")
    _check_data(X, y)

    def fit(rows: FitRows) -> Any:
        _no_early_stopping(rows, "this estimator")
        estimator = make_estimator()
        estimator.fit(_take(X, rows.train), _take(y, rows.train))
        return estimator

    return fit


# ----------------------------------------------------------------------- metrics and probabilities

def _probabilities(p: Any, what: str) -> np.ndarray:
    """``p`` as a float64 array (a copy) of finite values; TypeError or ValueError otherwise."""
    if isinstance(p, (str, bytes)):
        raise TypeError(f"{what} must be an array of numbers; got {type(p).__name__}")
    try:
        array = np.array(p, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{what} must be an array of numbers: {exc}") from None
    bad = ~np.isfinite(array)
    if bad.any():
        raise ValueError(f"{what} holds {int(bad.sum()):,} missing or non-finite value(s)")
    return array


def _labels(y: Any, n_rows: int) -> np.ndarray:
    """Class labels as int64, one per row; TypeError or ValueError otherwise."""
    if isinstance(y, (str, bytes, set, frozenset, pd.DataFrame)) or np.ndim(y) != 1:
        raise TypeError("y must be a 1-D array or Series with the observed class of each row")
    series = y if isinstance(y, pd.Series) else pd.Series(np.asarray(y))
    if len(series) != n_rows:
        raise ValueError(f"y has {len(series):,} labels but there are {n_rows:,} rows of probabilities")
    if series.isna().any():
        raise ValueError(f"y has {int(series.isna().sum()):,} missing label(s)")
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.to_numpy(dtype=np.int64)
    try:
        values = series.to_numpy(dtype=np.float64)
    except (TypeError, ValueError):
        raise TypeError(f"y must hold class numbers; got dtype {series.dtype}") from None
    if not (np.isfinite(values) & (np.floor(values) == values)).all():
        raise ValueError("y must hold whole class numbers")
    return values.astype(np.int64)


def pooled_log_loss(y: Any, p: Any) -> float:
    """Mean clipped log loss over every row (glossary, Pooled_Held_Out_Log_Loss).

    Binary: ``p`` is 1-D, the probability of class 1, and ``y`` holds 0 or 1 (or booleans).
    Multiclass: ``p`` is 2-D with one column per class and ``y`` holds the column index of the
    observed class. The predicted probability of each row's observed class is clipped to
    [LOG_LOSS_CLIP, 1 - LOG_LOSS_CLIP], and the result is the mean of -log of it over all rows, so
    it does not depend on how rows are grouped into folds. Returns a float.

    Raises ValueError when there is no row, a probability is missing, non-finite or outside
    [0, 1], a label is missing, not whole or not a valid class, or ``y`` and ``p`` differ in
    length, and TypeError for non-numeric input.
    """
    probs = _probabilities(p, "p")
    if probs.ndim not in (1, 2):
        raise ValueError(f"p must be 1-D (binary) or 2-D (multiclass); got {probs.ndim} dimensions")
    n_rows = probs.shape[0]
    if n_rows == 0:
        raise ValueError("log loss needs at least one row")
    if (probs < 0.0).any() or (probs > 1.0).any():
        raise ValueError(f"p holds {int(((probs < 0.0) | (probs > 1.0)).sum()):,} value(s) outside [0, 1]")
    labels = _labels(y, n_rows)
    if probs.ndim == 1:
        if not np.isin(labels, (0, 1)).all():
            raise ValueError("binary labels must be 0 or 1")
        observed = np.where(labels == 1, probs, 1.0 - probs)
    else:
        n_classes = probs.shape[1]
        if n_classes < 2:
            raise ValueError(f"multiclass p needs at least 2 columns; got {n_classes}")
        if labels.min() < 0 or labels.max() >= n_classes:
            raise ValueError(f"multiclass labels must lie in 0 to {n_classes - 1}, the columns of p")
        observed = probs[np.arange(n_rows), labels]
    clipped = np.clip(observed, LOG_LOSS_CLIP, 1.0 - LOG_LOSS_CLIP)
    return float(-np.mean(np.log(clipped)))


class WarningLog(Protocol):
    """Where renormalise writes its warning: the RunLog (``ctx.log``) or any object with ``warning``."""

    def warning(self, message: object) -> Any: ...


def renormalise(probs: Any, log: WarningLog, *, what: str = "class probabilities") -> np.ndarray:
    """Rows of class probabilities rescaled to sum to 1, as a new float64 array (Requirement 11, criterion 5).

    ``probs`` is 2-D, one row per prediction and one column per class, with finite values of 0 or
    more and a positive sum in every row. Every row is divided by its sum in float64. When at
    least one row summed to a value more than PROB_SUM_TOLERANCE (1e-6) away from 1 before
    rescaling, ``log.warning`` receives one message naming ``what``, the count of such rows, the
    row total and the largest difference. Raises ValueError for a row that cannot be rescaled and
    TypeError for non-numeric input.
    """
    array = _probabilities(probs, what)
    if array.ndim != 2:
        raise ValueError(
            f"{what} must be 2-D, one row per prediction and one column per class; got {array.ndim} dimensions"
        )
    if (array < 0.0).any():
        raise ValueError(f"{what} holds {int((array < 0.0).sum()):,} negative value(s), which rescaling cannot repair")
    sums = array.sum(axis=1)
    empty = sums <= 0.0
    if empty.any():
        raise ValueError(f"{what}: {int(empty.sum()):,} row(s) sum to 0, so they cannot be rescaled to sum to 1")
    deviation = np.abs(sums - 1.0)
    off = deviation > PROB_SUM_TOLERANCE
    if off.any():
        log.warning(
            f"{what}: {int(off.sum()):,} of {len(array):,} rows summed to a value more than 1e-6 away from 1 before "
            f"renormalisation (largest difference {float(deviation.max()):.3e}); every row was rescaled to sum to 1 "
            "(Requirement 11, criterion 5)"
        )
    array /= sums[:, None]
    return array
