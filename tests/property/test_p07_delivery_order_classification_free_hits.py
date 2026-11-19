# Feature: pressure-rating-system, Property 7: For any shuffled set of synthetic Deliveries whose `extras_type` values come from the Config mapping (all 7 delivered values, including `Overstep` and `Penalty`), reconstruction returns Deliveries sorted by `match_id`, `innings`, `innings_ball_num`; classifies each as Legal_Ball, Wide or No_Ball by the mapping, treating `None` as a value; and flags as Free_Hit exactly the Deliveries after a No_Ball up to and including the next Legal_Ball in the same innings, as computed by a sequential reference walk. For any input containing an `extras_type` outside the mapping or missing, and for any mapping that lists one value under more than one class, reconstruction stops and reports each unexpected value with its row count and each value listed under more than one class.
"""Property test for Delivery order, classification and Free_Hit flags (design Property 7).

Each example draws 1-4 innings with ``tests.strategies.innings`` (up to 3 Overs each). The
innings have distinct (``match_id``, ``innings``) keys from a pool of 1-3 matches and are joined
in the Requirement 5 order. In about half of the innings the ``extras_type`` column is redrawn as
any sequence of Config mapping values, so runs of No_Balls, a Wide straight after a No_Ball and a
No_Ball at the end of an innings are common. The Deliveries, plus tied copies of a few of them,
are shuffled and given arbitrary index labels (shuffled integers or text).

The test checks four things:

* Order (criterion 1). ``order_deliveries`` returns the rows of a stable sort by ``match_id``,
  ``innings`` and ``innings_ball_num`` as numbers, so tied copies keep their input order, and
  ``row`` equals the position, replacing any stale ``row`` column.
* Classes (criterion 2). ``classify`` gives each Delivery the class that the Config mapping lists
  for its ``extras_type``. Every example checks the Data_Loader's ``string`` dtype and object,
  ``str`` and categorical Series, and also classifies a catalogue of every mapping value. The
  text ``None`` is a value: listed, it gets its class, and unlisted, it is reported as the value
  ``'None'`` with its row count, never as a missing entry.
* Stops (criterion 3). Missing entries (None, NaN or pd.NA) and unlisted texts (case, spacing and
  NA-like variants of mapping values, and arbitrary text) raise ``DeliveryClassError`` in each of
  those dtypes. Its ``unexpected`` holds exactly each such value with its row count (key None for
  missing), and its message names each one with its count. A mapping that lists a value under
  more than one class makes ``check_mapping`` and ``classify`` raise, with ``duplicated`` naming
  exactly those values and their classes; a value repeated within one class is not a problem.
* Free_Hits (criterion 12). ``free_hit_flags`` on the shuffled Deliveries equals a sequential
  walk of each innings in generation order.

**Validates: Requirements 5.1, 5.2, 5.3, 5.12**
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, note
from hypothesis import strategies as st

from superr.errors import DataError
from superr.reconstruct import (
    DeliveryClass,
    DeliveryClassError,
    check_mapping,
    classify,
    free_hit_flags,
    order_deliveries,
)
from tests.strategies import EXTRAS_TYPE_CLASSES, innings

#: The Requirement 5 order of Deliveries (criterion 1), most significant key first.
REQUIREMENT_5_ORDER = ("match_id", "innings", "innings_ball_num")
#: Most Overs in one drawn innings: enough for Free_Hit spans that cross from one Over to the next.
MAX_OVERS = 3
#: dtypes of the extras_type Series given to classify, the Data_Loader's ``string`` dtype first.
EXTRAS_DTYPES = ("string", "object", "str", "category")
#: The extras_type text of a plain Legal_Ball: a value, not a missing entry (criterion 2).
NONE_TEXT = "None"

# The class keys of the Config mapping (settings.REQUIRED) and the DeliveryClass of each.
_CLASS_OF_KEY: dict[str, DeliveryClass] = {
    "legal": DeliveryClass.LEGAL,
    "wide": DeliveryClass.WIDE,
    "no_ball": DeliveryClass.NO_BALL,
}
_KEYS = tuple(_CLASS_OF_KEY)


def _class_of_value() -> dict[str, int]:
    """DeliveryClass code of each extras_type value, read directly from the Config mapping."""
    keys_of: dict[str, set[str]] = {}
    for key, values in EXTRAS_TYPE_CLASSES.items():
        for value in values:
            keys_of.setdefault(value, set()).add(key)
    if set(EXTRAS_TYPE_CLASSES) != set(_KEYS) or any(len(keys) != 1 for keys in keys_of.values()):
        raise RuntimeError(
            f"Property 7 needs a Config mapping with the classes {', '.join(_KEYS)} and each value "
            f"under one class; found {EXTRAS_TYPE_CLASSES!r}"
        )
    return {value: int(_CLASS_OF_KEY[keys.pop()]) for value, keys in keys_of.items()}


#: Expected DeliveryClass code of each value of the Config mapping.
CLASS_OF = _class_of_value()
#: Every value of the Config mapping, in Config order (as delivered, the 7 values).
MAPPING_VALUES: tuple[str, ...] = tuple(CLASS_OF)


def _near_misses() -> tuple[str, ...]:
    """Unlisted texts close to listed ones: other letter case, padding, a dropped last letter, NA-like text."""
    variants = [
        variant
        for value in MAPPING_VALUES
        for variant in (
            value.lower(), value.upper(), value.title(), value.swapcase(), f" {value}", f"{value} ", value[:-1]
        )
    ]
    variants += ["", " ", "nan", "NaN", "NA", "<NA>", "null"]
    return tuple(dict.fromkeys(variant for variant in variants if variant not in CLASS_OF))


#: How a missing entry is written into the extras_type Series before the dtype is applied.
_MISSING_MARKERS: dict[str, Any] = {"python_none": None, "nan": np.nan, "pd_na": pd.NA}
_UNLISTED = st.one_of(
    st.sampled_from(_near_misses()),
    st.text(max_size=6).filter(lambda text: text not in CLASS_OF),
)
# One replaced extras_type: ("missing", marker name) or ("text", an unlisted text).
_INJECTIONS = st.one_of(
    st.tuples(st.just("missing"), st.sampled_from(tuple(_MISSING_MARKERS))),
    st.tuples(st.just("text"), _UNLISTED),
)
# Values added to a copy of the mapping, each under one or more class keys (repeats allowed).
_LISTINGS = st.lists(
    st.tuples(
        st.one_of(st.sampled_from(MAPPING_VALUES), _UNLISTED),
        st.lists(st.sampled_from(_KEYS), min_size=1, max_size=3),
    ),
    min_size=1,
    max_size=3,
)


class Case(NamedTuple):
    """One example of Property 7."""

    base: pd.DataFrame  # Deliveries in generation order (the Requirement 5 order, unique keys); tag = position
    pool: pd.DataFrame  # base plus tied copies (tag >= len(base)), shuffled, with arbitrary index labels
    catalogue: list[str]  # every value of the Config mapping, in a drawn order
    injected: dict[int, tuple[str, str]]  # position among the shuffled originals -> replacement extras_type
    listings: list[tuple[str, list[str]]]  # (value, class keys) added to a copy of the mapping


@st.composite
def _labels(draw: Any, size: int) -> pd.Index:
    """``size`` distinct index labels in a shuffled order: integers on a spaced grid, or text."""
    order = draw(st.permutations(range(size)))
    if draw(st.booleans()):
        start, step = draw(st.integers(-1_000_000, 1_000_000)), draw(st.integers(1, 9))
        return pd.Index([start + step * k for k in order])
    return pd.Index([f"d{k}" for k in order])


@st.composite
def _cases(draw: Any) -> Case:
    match_ids = draw(st.lists(st.integers(1, 99_999), min_size=1, max_size=3, unique=True))
    keys = draw(
        st.lists(st.tuples(st.sampled_from(match_ids), st.integers(1, 4)), min_size=1, max_size=4, unique=True)
    )
    frames = []
    for match_id, innings_no in sorted(keys):
        frame = draw(innings(MAX_OVERS, match_id=match_id, innings_no=innings_no))
        if draw(st.booleans()):
            values = draw(st.lists(st.sampled_from(MAPPING_VALUES), min_size=len(frame), max_size=len(frame)))
            frame["extras_type"] = pd.array(values, dtype="string")
        frames.append(frame)
    base = pd.concat(frames, ignore_index=True)
    n = len(base)
    order_keys = list(zip(*(base[column].tolist() for column in REQUIREMENT_5_ORDER)))
    if any(earlier >= later for earlier, later in zip(order_keys, order_keys[1:])):
        raise AssertionError("strategy bug: the joined innings are not in the Requirement 5 order with unique keys")
    base["tag"] = np.arange(n)

    copies = draw(st.lists(st.integers(0, n - 1), max_size=4))  # rows whose keys are tied in the pool
    pool = pd.concat([base, base.iloc[copies]], ignore_index=True)
    pool["tag"] = np.arange(len(pool))
    pool = pool.iloc[draw(st.permutations(range(len(pool))))]
    pool.index = draw(_labels(len(pool)))
    if draw(st.booleans()):
        pool.insert(0, "row", np.arange(len(pool), 0, -1))  # stale numbering, which order_deliveries replaces
    return Case(
        base=base,
        pool=pool,
        catalogue=draw(st.permutations(MAPPING_VALUES)),
        injected=draw(st.dictionaries(st.integers(0, n - 1), _INJECTIONS, min_size=1, max_size=5)),
        listings=draw(_LISTINGS),
    )


# ------------------------------------------------------------------------------------- helpers

class _Table:
    """Columns of a frame for hypothesis.note, formatted only when Hypothesis reports the note."""

    def __init__(self, title: str, frame: pd.DataFrame, columns: list[str]) -> None:
        self.title, self.frame, self.columns = title, frame, columns

    def __repr__(self) -> str:
        return f"{self.title}:\n{self.frame[self.columns].to_string()}"


def _reference_free_hits(base: pd.DataFrame) -> list[bool]:
    """Free_Hit flag of each row of ``base``, whose rows are in the Requirement 5 order (criterion 12).

    A sequential walk: within an innings, a Delivery is a Free_Hit while a No_Ball came earlier in
    the innings with no Legal_Ball since. A No_Ball opens the span after itself, a Legal_Ball ends it
    after itself and a Wide changes nothing. Each innings starts with no span open.
    """
    flags: list[bool] = []
    current: tuple[int, int] | None = None
    span_open = False
    rows = zip(base["match_id"].tolist(), base["innings"].tolist(), base["extras_type"].tolist())
    for match_id, innings_no, value in rows:
        if (match_id, innings_no) != current:
            current, span_open = (match_id, innings_no), False
        flags.append(span_open)
        if CLASS_OF[value] == DeliveryClass.NO_BALL:
            span_open = True
        elif CLASS_OF[value] == DeliveryClass.LEGAL:
            span_open = False
    return flags


def _raises(function: Callable[..., Any], *args: Any) -> DeliveryClassError:
    """The DeliveryClassError that ``function(*args)`` raises. It is a DataError, so it stops the Run."""
    with pytest.raises(DeliveryClassError) as caught:
        function(*args)
    assert isinstance(caught.value, DataError)
    return caught.value


def _assert_reported(message: str, counts: Mapping[Any, int]) -> None:
    """Each value of ``counts`` is named on a line of ``message`` that also gives its row count."""
    lines = message.splitlines()
    for value, count in counts.items():
        name = "missing" if value is None else repr(value)
        number = re.compile(rf"(?<![\d,]){re.escape(f'{count:,}')}(?![\d,])")
        assert any(name in line and number.search(line.replace(name, "", 1)) for line in lines), (
            f"the report does not give {name} with {count} row(s):\n{message}"
        )


def _check_order(pool: pd.DataFrame) -> None:
    """order_deliveries returns the rows of a stable numeric sort by the order keys, numbered by row."""
    ordered = order_deliveries(pool)
    keys = list(zip(*(pool[column].tolist() for column in REQUIREMENT_5_ORDER)))
    stable = sorted(range(len(pool)), key=keys.__getitem__)  # Python's sort is stable: ties keep input order
    assert ordered["row"].tolist() == list(range(len(pool)))
    expected = pool.drop(columns="row", errors="ignore").iloc[stable].reset_index(drop=True)
    pd.testing.assert_frame_equal(ordered.drop(columns="row").reset_index(drop=True), expected)


def _check_classes(extras: pd.Series, catalogue: Sequence[str], dtype: str) -> None:
    """classify follows the Config mapping for every Delivery and every mapping value (criterion 2)."""
    cls = classify(extras, EXTRAS_TYPE_CLASSES)
    assert cls.index.equals(extras.index)
    assert cls.tolist() == [CLASS_OF[value] for value in extras.tolist()]
    every = pd.Series(list(catalogue), dtype=dtype)
    assert classify(every, EXTRAS_TYPE_CLASSES).tolist() == [CLASS_OF[value] for value in catalogue]
    # The text None is a value: when the mapping does not list it, its rows are reported as the
    # value 'None' with their count, and nothing is reported as missing.
    n_none = extras.tolist().count(NONE_TEXT)
    without_none = {
        key: [value for value in values if value != NONE_TEXT] for key, values in EXTRAS_TYPE_CLASSES.items()
    }
    if n_none:
        assert _raises(classify, extras, without_none).unexpected == {NONE_TEXT: n_none}


def _check_free_hits(deliveries: pd.DataFrame, cls: pd.Series, base: pd.DataFrame) -> None:
    """free_hit_flags on the shuffled Deliveries equals the sequential walk in generation order (criterion 12)."""
    flags = free_hit_flags(deliveries, cls)
    expected = _reference_free_hits(base)
    assert flags.index.equals(deliveries.index)
    assert pd.api.types.is_bool_dtype(flags.dtype)
    assert flags.tolist() == [expected[tag] for tag in deliveries["tag"].tolist()]


def _check_missing_and_unlisted(extras: pd.Series, injected: Mapping[int, tuple[str, str]], dtype: str) -> None:
    """Missing entries and unlisted values stop classification, each reported with its row count (criterion 3)."""
    values: list[Any] = extras.tolist()
    expected: Counter[Any] = Counter()
    for position, (kind, value) in injected.items():
        if kind == "missing":
            values[position] = _MISSING_MARKERS[value]
            expected[None] += 1
        else:
            values[position] = value
            expected[value] += 1
    error = _raises(classify, pd.Series(values, index=extras.index, dtype=dtype), EXTRAS_TYPE_CLASSES)
    assert error.unexpected == dict(expected)
    assert error.duplicated == {}
    _assert_reported(str(error), expected)


def _check_duplicated_listings(extras: pd.Series, listings: Sequence[tuple[str, Sequence[str]]]) -> None:
    """A mapping that lists a value under more than one class stops the Run, naming each such value (criterion 3)."""
    check_mapping(EXTRAS_TYPE_CLASSES)  # the Config mapping lists each value under one class, so it passes
    mapping = {key: list(values) for key, values in EXTRAS_TYPE_CLASSES.items()}
    for value, keys in listings:
        for key in keys:
            mapping[key].append(value)
    keys_of: dict[str, set[str]] = {}
    for key, values in mapping.items():
        for value in values:
            keys_of.setdefault(value, set()).add(key)
    expected = {value: sorted(keys) for value, keys in keys_of.items() if len(keys) > 1}
    if not expected:
        # Repeats within one class, and new values under one class, leave a usable mapping.
        check_mapping(mapping)
        assert classify(extras, mapping).tolist() == [CLASS_OF[value] for value in extras.tolist()]
        return
    for function, args in ((check_mapping, (mapping,)), (classify, (extras, mapping))):
        error = _raises(function, *args)
        assert {value: sorted(keys) for value, keys in error.duplicated.items()} == expected
        assert error.unexpected == {}
        missing_names = [value for value in expected if repr(value) not in str(error)]
        assert not missing_names, f"the report does not name {missing_names}:\n{error}"


# ---------------------------------------------------------------------------------------- test

@given(case=_cases())
def test_delivery_order_classification_and_free_hit_flags(case: Case) -> None:
    """Order, classes and Free_Hits of shuffled Deliveries; stops on unlisted, missing or doubly listed values.

    **Validates: Requirements 5.1, 5.2, 5.3, 5.12**
    """
    # The DataFrame reprs in a falsifying example hide these columns, so the report shows them.
    note(_Table("Deliveries in generation order", case.base, [*REQUIREMENT_5_ORDER, "overid", "extras_type", "tag"]))
    note(_Table("Shuffled pool (index labels, keys, tag)", case.pool, [*REQUIREMENT_5_ORDER, "tag"]))
    _check_order(case.pool)
    deliveries = case.pool[case.pool["tag"] < len(case.base)]  # the shuffled Deliveries without tied copies
    for dtype in EXTRAS_DTYPES:
        extras = deliveries["extras_type"].astype(dtype)
        _check_classes(extras, case.catalogue, dtype)
        _check_missing_and_unlisted(extras, case.injected, dtype)
    _check_duplicated_listings(deliveries["extras_type"], case.listings)
    _check_free_hits(deliveries, classify(deliveries["extras_type"], EXTRAS_TYPE_CLASSES), case.base)
