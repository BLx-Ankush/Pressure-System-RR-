"""ConstantRegistry: label each Method_Constant where it first appears in the Report_Text.

Requirement 2, criterion 8; design section "Report_Builder". In each report build, a
Method_Constant gets one of three labels at its first appearance and no label after that:

* ``"hand-set (Dn)"``: the Config value of Design_Choice Dn (``constants_meta`` kind
  ``"design_choice"``).
* ``"hand-set"``: any other Config value (kind ``"hand_set"``). A sensitivity grid that varies a
  Design_Choice is not itself a Design_Choice, so its label carries no D-number.
* ``"data-derived (<Component>)"``: a value that a Pipeline component produced during the Run.
  ``<Component>`` is one of ``PIPELINE_COMPONENTS``, spelt as in the requirements glossary.

Names. A Config Method_Constant is named by its dotted Config path, the key of its
``CONFIG["constants_meta"]`` entry, such as ``"design_choices.D7_new_batter_cap"``. A
data-derived constant gets a dotted name from the stage that records it, such as
``"reconstruct.selected_encoding"`` or ``"rating.min_pressure_balls.batter"``; it may not reuse a
``constants_meta`` path.

Use in a Run (task 2.3 puts one registry per Run in ``StageContext.constants``):

1. ``registry = ConstantRegistry.from_settings(settings)``.
2. Before each stage, ``registry.begin_stage(stage.name)``. The stage calls
   ``ctx.constants.record(name, value, component)`` for each data-derived constant behind the
   Main_Ranking. Recording a name again with the same value and component does nothing; a
   different value or component raises ConstantError. Stage code that a Robustness_Case,
   Trial_Run or comparison ranking reruns with other settings must therefore record under its
   own names, or not at all.
3. After the stage, save ``registry.to_dict(stage=stage.name)`` (JSON-safe) with the stage's
   checkpoint. On ``--from-stage``, pass the saved dict of every earlier stage to
   ``registry.load_dict``.
4. The Report_Builder calls ``reset_labels()`` at the start of each report build and inserts
   every Method_Constant through ``label(name)`` in document order, so the label lands at the
   first appearance. ``label`` returns ``""`` for a name already labelled in the build, and
   raises UnknownConstantError for a name that is neither in ``constants_meta`` nor recorded.
"""

from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from superr.settings import Settings

__all__ = [
    "PIPELINE_COMPONENTS",
    "ConstantError",
    "ConstantRegistry",
    "UnknownConstantError",
]

#: The "System components" of the requirements glossary that can produce a data-derived
#: Method_Constant, in glossary order and spelling.
PIPELINE_COMPONENTS: tuple[str, ...] = (
    "Data_Loader",
    "Auditor",
    "Score_Reconstructor",
    "Result_Parser",
    "State_Builder",
    "Fold_Assigner",
    "Skill_Model",
    "Remaining_Runs_Model",
    "Win_Probability_Model",
    "Outcome_Model",
    "Extras_Model",
    "Pressure_Engine",
    "Pressure_Validator",
    "Value_Engine",
    "Rating_Engine",
    "Robustness_Suite",
    "Model_Choice_Checker",
    "Stage2_Trial_Runner",
    "List_Builder",
    "Report_Builder",
)

# The other "System components" glossary entries, and why none of them can be the producer
# named in a data-derived label.
_NOT_PRODUCERS: dict[str, str] = {
    "Pipeline": "it is the whole system; name the component that produced the value",
    "Config": "a Config value is hand-set, never data-derived",
    "Schema_Registry": "it is a column list stored in code",
    "Test_Suite": "it does not run as part of a Run",
    "Run": "it is one execution of the Pipeline",
    "Run_Log": "it is a log file",
    "Pipeline_Stage": "it is a step of the Pipeline; name the component that the stage runs",
    "Report_Check": "it is a set of checks on the exported report",
}

_PRODUCERS = frozenset(PIPELINE_COMPONENTS)
_FORMAT_VERSION = 1
_DICT_KEYS = frozenset({"version", "derived"})
_ENTRY_KEYS = frozenset({"value", "component", "stage"})
_D_NUMBER = re.compile(r"D[1-9][0-9]*")
_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*")


class ConstantError(ValueError):
    """A Method_Constant cannot be labelled or recorded as asked.

    Raised for label metadata that gives no label, an invalid data-derived name, value or
    component, a name recorded twice with a different value or component, and a malformed
    constants dict.
    """


class UnknownConstantError(ConstantError, KeyError):
    """The name is not a Config Method_Constant or a recorded data-derived constant."""

    def __str__(self) -> str:  # KeyError would print the message in quotes
        return str(self.args[0]) if self.args else ""


@dataclass(frozen=True)
class _Derived:
    value: Any  # JSON-safe copy made by _json_safe
    component: str
    stage: str | None
    canonical: str  # canonical JSON of value; records are compared on it


class ConstantRegistry:
    """Labels of the Run's Method_Constants and the data-derived values the stages recorded."""

    __slots__ = ("_config_labels", "_derived", "_labelled", "_stage")

    def __init__(self, meta: Mapping[str, Mapping[str, Any]]) -> None:
        """Registry for the Config Method_Constants in ``meta``, with no data-derived constant yet.

        ``meta`` is ``CONFIG["constants_meta"]``: Config path -> ``{"kind", "d_number",
        "description"}``. Raises ConstantError, listing every bad entry, when an entry gives no
        label.
        """
        self._config_labels: dict[str, str] = _config_labels(meta)
        self._derived: dict[str, _Derived] = {}
        self._labelled: set[str] = set()
        self._stage: str | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> ConstantRegistry:
        """Registry for the ``constants_meta`` of a loaded Config."""
        return cls(settings.get("constants_meta"))

    def __repr__(self) -> str:
        return (
            f"ConstantRegistry(config={len(self._config_labels)}, derived={len(self._derived)}, "
            f"labelled={len(self._labelled)}, stage={self._stage!r})"
        )

    # -- recording (runner and stages) -------------------------------------------------------

    @property
    def current_stage(self) -> str | None:
        """The Pipeline_Stage name that record() gives new constants (None before begin_stage)."""
        return self._stage

    def begin_stage(self, stage: str | None) -> None:
        """Tag the constants recorded from now on with this Pipeline_Stage name (None: no stage)."""
        self._stage = _check_stage(stage)

    def record(self, name: str, value: Any, component: str) -> None:
        """Record a data-derived Method_Constant that ``component`` produced in the current stage.

        name: a dotted name such as ``"reconstruct.selected_encoding"`` that is not a
        ``constants_meta`` path. value: None, a bool, an int, a finite float, a str, or a list or
        str-keyed dict of these; numpy scalars and arrays are converted and tuples become lists.
        component: one of PIPELINE_COMPONENTS, for example ``"Score_Reconstructor"``.

        Recording a name again with the same value and component does nothing. Values are
        compared as canonical JSON, so 1 and 1.0 differ. Any other repeat raises ConstantError.
        """
        self._add({name: self._entry(name, value, component, self._stage)})

    # -- reading (Report_Builder) ------------------------------------------------------------

    def __contains__(self, name: object) -> bool:
        """True for a Config Method_Constant and for a recorded data-derived constant."""
        return isinstance(name, str) and (name in self._config_labels or name in self._derived)

    def label(self, name: str) -> str:
        """The label at the first use of ``name`` in this report build, and ``""`` at later uses.

        Raises UnknownConstantError, at every call, for a name that is not a Config
        Method_Constant or a recorded data-derived constant.
        """
        text = self.label_text(name)
        if name in self._labelled:
            return ""
        self._labelled.add(name)
        return text

    def label_text(self, name: str) -> str:
        """The label of ``name`` without counting it as used (for tables that list constants)."""
        _check_type(name)
        text = self._config_labels.get(name)
        if text is not None:
            return text
        entry = self._derived.get(name)
        if entry is None:
            raise UnknownConstantError(
                f"{name!r} is not a known Method_Constant: config.py has no constants_meta entry "
                "for it and no stage recorded it as data-derived"
            )
        return f"data-derived ({entry.component})"

    def reset_labels(self) -> None:
        """Start a new report build: each Method_Constant gets its label again at its next use."""
        self._labelled.clear()

    def value(self, name: str) -> Any:
        """A copy of the value recorded for a data-derived constant.

        Raises UnknownConstantError for a name that no stage recorded, including a Config
        Method_Constant, whose value comes from Settings.get.
        """
        _check_type(name)
        entry = self._derived.get(name)
        if entry is not None:
            return copy.deepcopy(entry.value)
        if name in self._config_labels:
            raise UnknownConstantError(
                f"{name!r} is a Config Method_Constant, not a data-derived one; read its value "
                "with Settings.get"
            )
        raise UnknownConstantError(f"no stage recorded a data-derived constant named {name!r}")

    # -- checkpoints (runner) ----------------------------------------------------------------

    def to_dict(self, stage: str | None = None) -> dict[str, Any]:
        """JSON-safe dict of the recorded data-derived constants, sorted by name.

        With ``stage``, only the constants tagged with that Pipeline_Stage name, which are the
        ones to save with that stage's checkpoint. Config constants are left out: they come from
        config.py, whose hash the checkpoint manifest records. Shape:
        ``{"version": 1, "derived": {name: {"value": ..., "component": ..., "stage": ...}}}``.
        """
        stage = _check_stage(stage)
        derived: dict[str, Any] = {}
        for name in sorted(self._derived):
            entry = self._derived[name]
            if stage is None or entry.stage == stage:
                derived[name] = {
                    "value": copy.deepcopy(entry.value),
                    "component": entry.component,
                    "stage": entry.stage,
                }
        return {"version": _FORMAT_VERSION, "derived": derived}

    def load_dict(self, data: Mapping[str, Any]) -> None:
        """Add the constants of a dict written by to_dict, such as an earlier stage's checkpoint.

        Each entry keeps its saved stage tag and follows the rules of record(). Raises
        ConstantError for a malformed dict or a conflicting entry; nothing is added then.
        """
        if not isinstance(data, Mapping) or set(data) != _DICT_KEYS:
            raise ConstantError(
                "a constants dict needs exactly the keys 'version' and 'derived', as written by "
                "ConstantRegistry.to_dict"
            )
        version = data["version"]
        if type(version) is not int or version != _FORMAT_VERSION:
            raise ConstantError(f"constants dict version {version!r} is not {_FORMAT_VERSION}")
        derived = data["derived"]
        if not isinstance(derived, Mapping):
            raise ConstantError("'derived' in a constants dict must map names to entries")
        entries: dict[str, _Derived] = {}
        for name, item in derived.items():
            if not isinstance(item, Mapping) or set(item) != _ENTRY_KEYS:
                raise ConstantError(
                    f"constants dict entry {name!r} needs exactly the keys component, stage and value"
                )
            entries[name] = self._entry(name, item["value"], item["component"], _check_stage(item["stage"]))
        self._add(entries)

    # -- internals ---------------------------------------------------------------------------

    def _entry(self, name: Any, value: Any, component: Any, stage: str | None) -> _Derived:
        """Checked, JSON-safe record of one data-derived constant (not yet added)."""
        _check_type(name)
        if not _NAME.fullmatch(name):
            raise ConstantError(
                f"{name!r} is not a valid constant name; use a dotted name such as "
                "'reconstruct.selected_encoding'"
            )
        if name in self._config_labels:
            raise ConstantError(
                f"{name!r} is a Config Method_Constant, labelled {self._config_labels[name]!r} "
                "through constants_meta; record the data-derived value under another name"
            )
        _check_component(name, component)
        safe = _json_safe(value, name, "value")
        return _Derived(safe, component, stage, _canonical(safe))

    def _add(self, entries: Mapping[str, _Derived]) -> None:
        """Add every entry, or none when one conflicts with a recorded constant."""
        for name, new in entries.items():
            old = self._derived.get(name)
            if old is not None and (old.canonical, old.component) != (new.canonical, new.component):
                raise ConstantError(
                    f"data-derived constant {name!r} is already recorded as {_short(old.canonical)} "
                    f"from {old.component}{_in_stage(old.stage)}; it cannot be recorded again as "
                    f"{_short(new.canonical)} from {new.component}{_in_stage(new.stage)}"
                )
        for name, new in entries.items():
            self._derived.setdefault(name, new)


# ------------------------------------------------------------------------------------ helpers

def _config_labels(meta: Any) -> dict[str, str]:
    """Config path -> label for each constants_meta entry; ConstantError lists every bad entry."""
    if not isinstance(meta, Mapping):
        raise ConstantError(
            f"constants_meta must map Config paths to label metadata, got {type(meta).__name__}"
        )
    labels: dict[str, str] = {}
    problems: list[str] = []
    for path, entry in meta.items():
        where = f"constants_meta[{path!r}]"
        kind = entry.get("kind") if isinstance(entry, Mapping) else None
        d_number = entry.get("d_number") if isinstance(entry, Mapping) else None
        if not isinstance(path, str) or not path:
            problems.append(f"{where}: the key must be a dotted Config path")
        elif kind == "design_choice":
            if isinstance(d_number, str) and _D_NUMBER.fullmatch(d_number):
                labels[path] = f"hand-set ({d_number})"
            else:
                problems.append(f"{where}: a Design_Choice needs a d_number such as 'D7', got {d_number!r}")
        elif kind == "hand_set":
            labels[path] = "hand-set"
        else:
            problems.append(f"{where}: kind must be 'design_choice' or 'hand_set', got {kind!r}")
    if problems:
        raise ConstantError(
            "constants_meta cannot label every Method_Constant:\n" + "\n".join(f"  - {p}" for p in problems)
        )
    return labels


def _check_type(name: Any) -> None:
    if not isinstance(name, str):
        raise TypeError(f"a Method_Constant name must be a str, got {type(name).__name__}")


def _check_component(name: str, component: Any) -> None:
    if isinstance(component, str) and component in _PRODUCERS:
        return
    if isinstance(component, str) and component in _NOT_PRODUCERS:
        reason = f"{component!r} cannot produce a data-derived value: {_NOT_PRODUCERS[component]}"
    else:
        reason = f"{component!r} is not a Pipeline component named in the requirements glossary"
    raise ConstantError(
        f"data-derived constant {name!r}: {reason}. Use one of: {', '.join(PIPELINE_COMPONENTS)}"
    )


def _check_stage(stage: Any) -> str | None:
    if stage is None or (isinstance(stage, str) and stage and stage == stage.strip()):
        return stage
    raise ConstantError(
        f"a Pipeline_Stage name must be None or a non-empty str without surrounding spaces, got {stage!r}"
    )


def _json_safe(value: Any, name: str, path: str) -> Any:
    """JSON-safe copy of a recorded value: numpy values become Python values, tuples lists.

    Dict keys are sorted, so the saved JSON does not depend on insertion order.
    """
    if isinstance(value, (np.ndarray, np.generic)):
        if value.dtype.kind in "mM":  # datetime64 and timedelta64 would become bare integers
            raise TypeError(f"data-derived constant {name!r}: {path} is a numpy {value.dtype}, not JSON-safe")
        value = value.tolist()
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConstantError(f"data-derived constant {name!r}: {path} is {value!r}, not a finite number")
        return float(value)
    if isinstance(value, str):
        return str.__str__(value)  # a plain str, also for str subclasses such as a str Enum member
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, name, f"{path}[{i}]") for i, item in enumerate(value)]
    if isinstance(value, Mapping):
        for key in value:
            if not isinstance(key, str):
                raise TypeError(f"data-derived constant {name!r}: {path} has the non-str key {key!r}")
        return {key: _json_safe(value[key], name, f"{path}[{key!r}]") for key in sorted(value)}
    raise TypeError(
        f"data-derived constant {name!r}: {path} is a {type(value).__name__}, which is not JSON-safe; "
        "use None, bool, int, float, str, or lists and str-keyed dicts of these"
    )


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _short(text: str, limit: int = 80) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _in_stage(stage: str | None) -> str:
    return f" in stage {stage!r}" if stage else ""
