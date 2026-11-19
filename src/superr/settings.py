"""Settings: load, validate and read the single Config, ``<workspace root>/config.py``.

Rules (Requirement 1, criteria 3 and 11; design section "Settings"):

* ``load_settings(root)`` executes ``<root>/config.py`` from its file path. It never imports a
  module named ``config`` through ``sys.path``, and it writes no bytecode for the file.
* ``REQUIRED`` maps every key path the Pipeline reads to its type. Loading raises
  ``ConfigError`` naming every missing setting, every wrongly typed setting, every key that
  ``REQUIRED`` does not list, and every Method_Constant without ``constants_meta`` metadata.
* ``Settings.get`` has no default argument and raises ``KeyError`` for an absent path, so no
  setting can fall back to a value written in source code.
* ``Settings`` is read-only. ``get`` returns deep copies, and ``with_overrides`` builds a new
  in-memory ``Settings``; ``config.py`` is never written.
* ``Settings.sha256()`` is the SHA-256 of the ``config.py`` bytes that were executed, with or
  without overrides. ``Settings.effective_sha256()`` also covers the in-memory overrides.

Type specs in ``REQUIRED``: ``bool``, ``int``, ``float``, ``str``, ``list[X]``, fixed-length
``tuple[X, Y]``, ``tuple[X, ...]``, ``dict[K, V]`` and unions such as ``float | None``. ``bool``
never passes as ``int`` or ``float``; ``int`` passes as ``float``; a float must be finite; lists
and tuples are interchangeable. ``None`` counts as missing unless the spec includes ``None``.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin

__all__ = [
    "CONFIG_FILENAME",
    "REQUIRED",
    "ConfigError",
    "Settings",
    "config_problems",
    "constants_meta_problems",
    "load_settings",
    "method_constant_paths",
    "missing_settings",
    "unknown_settings",
    "wrong_types",
]

CONFIG_FILENAME = "config.py"
_MODULE_NAME = "_superr_config"  # name of the executed Config module; never put in sys.modules
_NoneType = type(None)
_MISSING = object()

# ------------------------------------------------------------------------------------ REQUIRED
# Key types per Config section. A dotted section name such as "models.m2" holds the keys of that
# nested dict. A task that adds a key to config.py adds its type to the matching section here
# (and, for a Method_Constant, a constants_meta entry in config.py).

_LGBM: dict[str, Any] = {
    "learning_rate": float,
    "num_leaves": int,
    "min_data_in_leaf": int,
    "lambda_l2": float,
    "max_rounds": int,
    "es_share": float,
    "es_patience": int,
}
_LOGISTIC: dict[str, Any] = {
    "C": float,
    "max_iter": int,
}
_XGB: dict[str, Any] = {
    "tree_method": str,
    "eta": float,
    "max_depth": int,
    "min_child_weight": float,
    "reg_lambda": float,
    "max_rounds": int,
    "es_share": float,
    "es_patience": int,
}
_RF: dict[str, Any] = {
    "n_estimators": int,
    "max_depth": int,
    "min_samples_leaf": int,
    "max_features": str,
}

_SECTIONS: dict[str, dict[str, Any]] = {
    "paths": {
        "ball_by_ball": str,
        "matches": str,
        "playing_eleven": str,
        "competitions": str,
        "teams": str,
        "mapping": str,
        "outputs": str,
        "docs": str,
    },
    "design_choices": {
        "D1_live_band": tuple[float, float],
        "D2_alpha": float,
        "D3_pressure_share": float,
        "D4_scope": str,
        "D5_weights": str,
        "D6_window": int,
        "D7_new_batter_cap": int,
        "D8_phase_splits": tuple[float, float],
        "D9_half_life_years": float | None,  # None means no decay (the Requirement 21 sensitivity case)
        "D10_knockout_weight": float,
        "D11_min_balls_grid": list[int],
        "D11_target_eligible": int,
        "D12_replacement_pct": float,
        "D13_cells": str,
        "D14_bowler_wicket_codes": list[int],
        "D14_run_out_code": int,
        "D14_retire_out_code": int,
        "D14_retire_out_batter_penalty": bool,
        "D15_n_folds": int,
        "D16_n_boot": int,
        "D17_wicket_buckets": list[tuple[int, int]],
        "D18_min_matches_grid": list[int],
        "D18_min_matches_default": int,
        "D19_period_minimum": int,
        "D20_score_tolerance": int,
    },
    "data_rules.extras_type_classes": {
        "legal": list[str],
        "wide": list[str],
        "no_ball": list[str],
    },
    "data_rules": {
        "candidate_encodings": list[str],
        "exclude_failing_matches": bool,
        "score_tolerance_grid": list[int],
        "full_members": list[str],
        "recent_activity_rule": bool,
        "recent_years": list[int],
        "min_major_share": float,
        "min_calendar_years": int,
    },
    "seeds": {
        name: int
        for name in (
            "folds", "m1", "m2", "m3_lgbm", "m3_logistic", "m3_xgb", "m3_rf",
            "m4_lgbm", "m4_xgb", "m4_rf", "extras", "bootstrap",
        )
    },
    "runtime": {
        "n_threads": int,
        "deterministic": bool,
    },
    "models.m1": {
        "ridge_grid": list[float],
        "prob_clip": tuple[float, float],
    },
    "models.m2": {"objective": str, **_LGBM, "monotone_constraints_method": str},
    "models.m3": {**_LGBM, "monotone_constraints_method": str, "calibration_bounds": tuple[float, float]},
    "models.m3.logistic": dict(_LOGISTIC),
    "models.m4": dict(_LGBM),
    "models.extras": dict(_LOGISTIC),
    "models.xgb.m3": dict(_XGB),
    "models.xgb.m4": dict(_XGB),
    "models.rf.m3": dict(_RF),
    "models.rf.m4": dict(_RF),
    "model_choice": {
        "calibration_preferred": dict[str, str],  # comparison-table model label -> preferred family
        "calibration_band": float,  # relative log-loss band delta for a preferred family
    },
    "sensitivity": {
        "alpha": list[float],
        "share": list[float],
        "live_b": list[float],
        "half_life": list[float | None],  # None means no decay
        "knockout": list[float],
        "blend_wL": list[float],
        "D4": list[str],
        "D6": list[int],
        "D7": list[int],
        "D8": list[tuple[float, float]],
        "D11": list[int],
        "D12": list[float],
        "D13": list[str],
        "D14": list[bool],
    },
    "tests": {
        "calibration_error_max": float,
        "share_band": tuple[float, float],
        "nbv_cell_tol": float,
        "tol_float": float,
        "tol_prob": float,
        "tol_ridge": float,
    },
    "stage2": {
        "F44": bool,
        "F45": bool,
        "F46": bool,
        "F47": bool,
        "F44_strength": float,
    },
    "report": {
        "pdf_converter": str | None,  # None: export the DOCX only (Requirement 27, criterion 15)
        "verified_sources": list[str],
        "page_budget": int,
    },
}

# Top-level keys that are not sections. constants_meta maps a REQUIRED path to its label
# metadata {"kind", "d_number", "description"}; constants_meta_problems checks its content.
_TOP_LEVEL: dict[str, Any] = {
    "constants_meta": dict[str, dict[str, str | None]],
}


def _flatten(sections: Mapping[str, Mapping[str, Any]], top_level: Mapping[str, Any]) -> dict[str, Any]:
    required: dict[str, Any] = {}
    for section, keys in sections.items():
        for key, spec in keys.items():
            required[f"{section}.{key}"] = spec
    required.update(top_level)
    return required


#: Dotted key path -> type spec of every setting the Pipeline reads, in Config order.
REQUIRED: dict[str, Any] = _flatten(_SECTIONS, _TOP_LEVEL)

# Every design_choices key is a Method_Constant. In these sections, the numeric keys are too.
_CONSTANT_SECTIONS = frozenset(
    {"data_rules", "seeds", "models", "model_choice", "sensitivity", "tests", "stage2"}
)
_META_FIELDS = ("kind", "d_number", "description")
_D_KEY = re.compile(r"^design_choices\.(D\d+)_")


class ConfigError(Exception):
    """config.py cannot be used: it is unreadable, or it fails validation against REQUIRED.

    ``missing``, ``wrong`` and ``unknown`` hold setting paths; ``meta`` holds constants_meta
    problem messages. All four are empty when the file itself could not be read or run.
    """

    def __init__(self, message: str, *, missing=(), wrong=(), unknown=(), meta=()) -> None:
        super().__init__(message)
        self.missing: list[str] = list(missing)
        self.wrong: list[str] = list(wrong)
        self.unknown: list[str] = list(unknown)
        self.meta: list[str] = list(meta)


# --------------------------------------------------------------------------------- type specs

def _matches(value: Any, spec: Any) -> bool:
    """True when value conforms to the type spec (rules in the module docstring)."""
    if spec is Any:
        return True
    if spec is None or spec is _NoneType:
        return value is None
    origin = get_origin(spec)
    if origin is UnionType or origin is Union:
        return any(_matches(value, arg) for arg in get_args(spec))
    if spec is bool:
        return isinstance(value, bool)
    if spec is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if spec is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        return isinstance(value, int) or math.isfinite(value)
    if spec is str:
        return isinstance(value, str)
    if origin is list or spec is list:
        args = get_args(spec)
        return isinstance(value, (list, tuple)) and (not args or all(_matches(v, args[0]) for v in value))
    if origin is tuple or spec is tuple:
        args = get_args(spec)
        if not isinstance(value, (list, tuple)):
            return False
        if not args:
            return True
        if len(args) == 2 and args[1] is Ellipsis:
            return all(_matches(v, args[0]) for v in value)
        return len(value) == len(args) and all(_matches(v, a) for v, a in zip(value, args))
    if origin is dict or spec is dict:
        args = get_args(spec)
        if not isinstance(value, Mapping):
            return False
        if not args:
            return True
        key_spec, value_spec = args
        return all(_matches(k, key_spec) and _matches(v, value_spec) for k, v in value.items())
    raise TypeError(f"unsupported type spec in superr.settings.REQUIRED: {spec!r}")


def _spec_name(spec: Any) -> str:
    if get_origin(spec) is None and isinstance(spec, type):
        return spec.__name__
    return str(spec).replace("NoneType", "None")


def _describe(value: Any) -> str:
    text = repr(value)
    if len(text) > 60:
        text = text[:57] + "..."
    return f"{type(value).__name__} {text}"


def _leaf_specs(spec: Any) -> set[Any]:
    origin = get_origin(spec)
    if origin in (list, tuple) or origin is UnionType or origin is Union:
        leaves: set[Any] = set()
        for arg in get_args(spec):
            if arg is not Ellipsis:
                leaves |= _leaf_specs(arg)
        return leaves
    if origin is dict:
        return {dict}
    return {spec}


def _is_numeric(spec: Any) -> bool:
    leaves = _leaf_specs(spec) - {_NoneType, None}
    return bool(leaves) and leaves <= {int, float}


# --------------------------------------------------------------------------------- validation

def _lookup(cfg: Any, path: str) -> Any:
    """Value at a dotted path, or _MISSING when a key is absent or a parent is not a mapping."""
    node = cfg
    for part in path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return _MISSING
        node = node[part]
    return node


def _is_missing(value: Any, spec: Any) -> bool:
    return value is _MISSING or (value is None and not _matches(None, spec))


def missing_settings(cfg: Mapping[str, Any]) -> list[str]:
    """Every REQUIRED path that is absent from cfg, or None while its type does not allow None."""
    return [path for path, spec in REQUIRED.items() if _is_missing(_lookup(cfg, path), spec)]


def _type_errors(cfg: Mapping[str, Any]) -> list[tuple[str, Any]]:
    errors = []
    for path, spec in REQUIRED.items():
        value = _lookup(cfg, path)
        if not _is_missing(value, spec) and not _matches(value, spec):
            errors.append((path, value))
    return errors


def wrong_types(cfg: Mapping[str, Any]) -> list[str]:
    """Every REQUIRED path that is present in cfg with a value of the wrong type."""
    return [path for path, _ in _type_errors(cfg)]


def unknown_settings(cfg: Mapping[str, Any]) -> list[str]:
    """Keys of cfg that REQUIRED does not list, such as a misspelt key or a key added without a type.

    Keys inside a REQUIRED dict setting (constants_meta, for example) are not inspected.
    """
    prefixes: set[str] = set()
    for path in REQUIRED:
        parts = path.split(".")
        prefixes.update(".".join(parts[:i]) for i in range(1, len(parts)))
    unknown: list[str] = []

    def walk(node: Mapping[Any, Any], prefix: str) -> None:
        for key, value in node.items():
            if not isinstance(key, str) or not key or "." in key:
                unknown.append(f"{prefix}[{key!r}]" if prefix else f"[{key!r}]")
                continue
            path = f"{prefix}.{key}" if prefix else key
            if path in REQUIRED:
                continue
            if path in prefixes:
                if isinstance(value, Mapping):
                    walk(value, path)
                continue  # a section that is not a dict: missing_settings reports its keys
            unknown.append(path)

    if isinstance(cfg, Mapping):
        walk(cfg, "")
    return unknown


def method_constant_paths() -> list[str]:
    """REQUIRED paths that are Method_Constants and need a constants_meta entry.

    These are every design_choices key, and every numeric key (int or float values, alone or in
    lists and tuples) in data_rules, seeds, models, sensitivity, tests and stage2.
    """
    paths = []
    for path, spec in REQUIRED.items():
        section = path.split(".", 1)[0]
        if section == "design_choices" or (section in _CONSTANT_SECTIONS and _is_numeric(spec)):
            paths.append(path)
    return paths


def constants_meta_problems(cfg: Mapping[str, Any]) -> list[str]:
    """Problems with the constants_meta label metadata, one message each.

    Each entry must name a REQUIRED path and hold exactly kind, d_number and description. A
    design_choices key has kind "design_choice" and the D-number its key starts with; any other
    key has kind "hand_set" and a d_number that is None or the D-number of an existing
    Design_Choice. Every path from method_constant_paths() needs an entry.
    """
    meta = _lookup(cfg, "constants_meta")
    if _is_missing(meta, REQUIRED["constants_meta"]) or not _matches(meta, REQUIRED["constants_meta"]):
        return []  # reported by missing_settings or wrong_types
    d_numbers = {match.group(1) for match in map(_D_KEY.match, REQUIRED) if match}
    problems: list[str] = []
    for path, entry in meta.items():
        where = f"constants_meta[{path!r}]"
        if path not in REQUIRED:
            problems.append(f"{where}: not a setting path listed in superr.settings.REQUIRED")
            continue
        if sorted(entry) != sorted(_META_FIELDS):
            problems.append(f"{where}: needs exactly the fields {', '.join(_META_FIELDS)}")
            continue
        kind, d_number, description = (entry[field] for field in _META_FIELDS)
        own = _D_KEY.match(path)
        if path.startswith("design_choices."):
            if own is None:
                problems.append(f"{where}: a design_choices key must start with its D-number")
            elif kind != "design_choice" or d_number != own.group(1):
                problems.append(f"{where}: needs kind 'design_choice' and d_number {own.group(1)!r}")
        elif kind != "hand_set":
            problems.append(f"{where}: needs kind 'hand_set' outside design_choices")
        elif d_number is not None and d_number not in d_numbers:
            problems.append(f"{where}: d_number {d_number!r} is not a Design_Choice")
        if not (description or "").strip():
            problems.append(f"{where}: description is empty")
    for path in method_constant_paths():
        if path not in meta:
            problems.append(f"constants_meta: no entry for the Method_Constant {path!r}")
    return problems


def config_problems(cfg: Mapping[str, Any]) -> list[str]:
    """Every problem with cfg as one readable line: missing, wrong type, unknown, label metadata."""
    lines = [f"missing setting: {path}" for path in missing_settings(cfg)]
    lines += [
        f"wrong type: {path} needs {_spec_name(REQUIRED[path])}, got {_describe(value)}"
        for path, value in _type_errors(cfg)
    ]
    lines += [f"unknown setting: {path} is not listed in superr.settings.REQUIRED" for path in unknown_settings(cfg)]
    lines += constants_meta_problems(cfg)
    return lines


# ------------------------------------------------------------------------------------ loading

def _execute(path: Path, source: bytes) -> Mapping[str, Any]:
    """Run the given config.py bytes as a fresh module and return its CONFIG mapping."""
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, path)
    if spec is None:
        raise ConfigError(f"cannot load the Config file {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        # Compiling the bytes already read keeps sha256() tied to exactly the code executed,
        # and no bytecode cache is written for config.py.
        exec(compile(source, str(path), "exec", dont_inherit=True), module.__dict__)
    except Exception as exc:
        raise ConfigError(f"{path} failed to run: {type(exc).__name__}: {exc}") from exc
    cfg = getattr(module, "CONFIG", None)
    if not isinstance(cfg, Mapping):
        raise ConfigError(f"{path} must define CONFIG as a dict; found {type(cfg).__name__}")
    return cfg


def load_settings(root: Path | str) -> Settings:
    """Execute ``<root>/config.py`` by its path, validate CONFIG and return a read-only Settings.

    Raises ConfigError, listing every problem, when the file is absent or unreadable, defines no
    CONFIG dict, or fails validation (Requirement 1, criterion 11). The runner calls this before
    any Source_CSV is loaded.
    """
    path = Path(root).resolve() / CONFIG_FILENAME
    try:
        source = path.read_bytes()
    except OSError as exc:
        raise ConfigError(f"cannot read the Config file {path}: {exc}") from exc
    cfg = _execute(path, source)
    missing = missing_settings(cfg)
    wrong = wrong_types(cfg)
    unknown = unknown_settings(cfg)
    meta = constants_meta_problems(cfg)
    if missing or wrong or unknown or meta:
        lines = config_problems(cfg)
        raise ConfigError(
            f"{path} has {len(lines)} problem(s):\n" + "\n".join(f"  - {line}" for line in lines),
            missing=missing,
            wrong=wrong,
            unknown=unknown,
            meta=meta,
        )
    return Settings(cfg, source_bytes=source, source_path=path)


def _assign(cfg: dict[str, Any], path: str, value: Any) -> None:
    """Replace the existing value at a dotted path; KeyError when the path is not in cfg."""
    if not isinstance(path, str) or not path:
        raise KeyError(f"override path {path!r} is not a dotted setting path")
    *parents, leaf = path.split(".")
    node: Any = cfg
    for part in parents:
        node = node.get(part, _MISSING) if isinstance(node, dict) else _MISSING
        if node is _MISSING:
            break
    if not isinstance(node, dict) or leaf not in node:
        raise KeyError(f"override path {path!r} is not a setting in the Config")
    node[leaf] = value


# ----------------------------------------------------------------------------------- Settings

class Settings:
    """Read-only view of a Config dict, normally built by load_settings.

    ``get(path)`` is the only way to read a value and has no default. Values come back as deep
    copies, so changing a returned list or dict never changes the stored Config.
    """

    __slots__ = ("_cfg", "_source_bytes", "_source_path", "_overrides", "_sha256")

    def __init__(self, cfg: Mapping[str, Any], *, source_bytes: bytes, source_path: Path | str | None = None) -> None:
        """Wrap a Config dict without validating it (load_settings validates).

        source_bytes are the config.py bytes that produced cfg; sha256() hashes them.
        source_path is that file; its folder is the workspace root used by path().
        """
        self._set_parts(copy.deepcopy(dict(cfg)), bytes(source_bytes), source_path, ())

    def _set_parts(self, cfg: dict[str, Any], source_bytes: bytes, source_path: Path | str | None,
                   overrides: tuple[tuple[str, Any], ...]) -> None:
        setter = object.__setattr__
        setter(self, "_cfg", cfg)
        setter(self, "_source_bytes", source_bytes)
        setter(self, "_source_path", None if source_path is None else Path(source_path))
        setter(self, "_overrides", overrides)
        setter(self, "_sha256", hashlib.sha256(source_bytes).hexdigest())

    @classmethod
    def _from_parts(cls, cfg: dict[str, Any], source_bytes: bytes, source_path: Path | None,
                    overrides: tuple[tuple[str, Any], ...]) -> Settings:
        settings = cls.__new__(cls)
        settings._set_parts(cfg, source_bytes, source_path, overrides)
        return settings

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Settings is read-only; use with_overrides() for an in-memory variant")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Settings is read-only")

    def __reduce__(self) -> tuple[Any, ...]:
        return (Settings._from_parts, (self._cfg, self._source_bytes, self._source_path, self._overrides))

    def __repr__(self) -> str:
        where = str(self._source_path) if self._source_path is not None else "<in memory>"
        return f"Settings({where!r}, sha256={self._sha256[:12]}, overrides={len(self._overrides)})"

    # -- reading ---------------------------------------------------------------------------

    def get(self, path: str) -> Any:
        """Deep copy of the value at a dotted path such as ``"design_choices.D2_alpha"``.

        A section path (``"models.m3"``) returns the whole dict. Raises KeyError when the path is
        absent, or when a REQUIRED setting holds None while its type does not allow None. There
        is no default argument, so a missing setting can never fall back to a coded value.
        """
        value = _lookup(self._cfg, path) if isinstance(path, str) and path else _MISSING
        if value is _MISSING:
            raise KeyError(f"setting {path!r} is not in the Config")
        if value is None and path in REQUIRED and not _matches(None, REQUIRED[path]):
            raise KeyError(f"setting {path!r} is None in the Config, which counts as missing")
        return copy.deepcopy(value)

    def as_dict(self) -> dict[str, Any]:
        """Deep copy of the whole Config, with any overrides applied (for logs and reports)."""
        return copy.deepcopy(self._cfg)

    def path(self, name: str) -> Path:
        """Path of ``paths.<name>`` joined to the workspace root, the folder that holds config.py."""
        relative = self.get(f"paths.{name}")
        if self._source_path is None:
            raise ValueError("this Settings was not loaded from config.py, so it has no workspace root")
        return self._source_path.parent / relative

    @property
    def source_path(self) -> Path | None:
        """The config.py file these settings came from (None for an in-memory Settings)."""
        return self._source_path

    @property
    def root(self) -> Path | None:
        """The workspace root: the folder that holds config.py (None for an in-memory Settings)."""
        return None if self._source_path is None else self._source_path.parent

    @property
    def overrides(self) -> dict[str, Any]:
        """In-memory overrides applied so far, as path -> value (a later value for a path wins)."""
        return copy.deepcopy(dict(self._overrides))

    # -- variants and hashes -----------------------------------------------------------------

    def with_overrides(self, overrides: Mapping[str, Any]) -> Settings:
        """New Settings in which each dotted path of ``overrides`` holds the given value.

        Used by Robustness_Cases and Stage-2 Trial_Runs. Every path must already exist; this
        Settings and config.py stay unchanged. Raises KeyError for a path not in the Config, and
        ValueError when an override leaves a setting missing, wrongly typed or unlisted.
        """
        if not isinstance(overrides, Mapping):
            raise TypeError("overrides must map dotted setting paths to values")
        cfg = copy.deepcopy(self._cfg)
        applied = []
        for path, value in overrides.items():
            _assign(cfg, path, copy.deepcopy(value))
            applied.append((path, copy.deepcopy(value)))
        before = set(config_problems(self._cfg))
        introduced = [line for line in config_problems(cfg) if line not in before]
        if introduced:
            raise ValueError(
                "overrides would make the Config invalid:\n" + "\n".join(f"  - {line}" for line in introduced)
            )
        return Settings._from_parts(cfg, self._source_bytes, self._source_path, self._overrides + tuple(applied))

    def sha256(self) -> str:
        """SHA-256 (hex) of the config.py bytes that were executed; overrides do not change it."""
        return self._sha256

    def effective_sha256(self) -> str:
        """SHA-256 (hex) of the Config actually in use.

        Without overrides this equals sha256(). With overrides it hashes the config.py bytes
        followed by the canonical JSON of the overrides in the order applied, so every in-memory
        variant (a Stage-2 Trial_Run, for example) gets its own hash for checkpoint manifests.
        """
        if not self._overrides:
            return self._sha256
        payload = json.dumps(
            [[path, value] for path, value in self._overrides],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            default=repr,
        )
        return hashlib.sha256(self._source_bytes + b"\x00overrides\x00" + payload.encode("utf-8")).hexdigest()
