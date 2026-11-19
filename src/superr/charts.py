"""List_Builder charts: interval forest plots, Clutch_Level against Usual_Baseline scatter plots
and Phase-mix and Competition_Type-mix stacked bars (Requirement 26, criteria 6-10; design section
"List_Builder (``lists.py``, ``charts.py``)", ``charts``: matplotlib, Agg backend, fixed DPI).

``write_charts`` (the design's ``charts``) draws four charts per Role from the evidence tables of
``superr.evidence`` and the rating checkpoint's ``player_estimates``, and writes each as a PNG to
``outputs/final/`` (CHART_FILES; criterion 10). ``lists.run_stage`` calls it.

* ``interval_forest_<batters|bowlers>.png`` (criterion 6): one row per Main_List player in rank
  order, rank 1 at the top, labelled ``<rank>. <player_name>``. Final_Score is a point and the 95%
  percentile interval of Final_Score (``ci_low`` to ``ci_high``, Requirement 19) a horizontal
  line, on an axis in runs per over, with a dotted line at 0. A player never ranked in a
  Bootstrap_Resample has no interval, and only the point is drawn.
* ``clutch_vs_usual_<...>.png`` (criterion 7): Clutch_Level (vertical axis) against
  Usual_Baseline (horizontal axis), both in runs per over, one point per Eligible_Player of the
  Role (``eligible`` in ``player_estimates``). The 10 Main_List players are larger orange
  diamonds labelled with ``player_name``; every other Eligible_Player is a small grey circle. The
  dashed diagonal marks Clutch_Level = Usual_Baseline (a Pressure_Gap of 0) and the dotted
  horizontal line the Role's Replacement_Level. An Eligible_Player without a finite Clutch_Level
  or Usual_Baseline cannot be placed; the chart says how many (none as delivered). The
  horizontal axis is widened on the right where needed, so that each listed player's label fits
  to the right of its marker. Labels are placed in rank order at the first offset of a fixed
  list (LABEL_OFFSETS) whose text box lies inside the axes and overlaps no earlier label and no
  listed player's marker; the farther offsets get a thin leader line. The placement depends only
  on the data, so it is the same in every Run.
* ``phase_mix_<...>.png`` (criterion 8) and ``competition_type_mix_<...>.png`` (criterion 9): one
  horizontal stacked bar per Main_List player in rank order, rank 1 at the top, with the
  player's Phase (powerplay, middle, death) or Competition_Type (IPL, T20I, A_Team) percentages
  from the Pressure_Ball mix table. A segment of at least SEGMENT_LABEL_MIN percent is labelled
  with its rounded percentage.

Every chart is a ``matplotlib.figure.Figure`` on the Agg canvas (no pyplot, so no global state
and no display) at CHART_DPI dots per inch and a fixed size, saved without ``Software`` metadata
or a time stamp, so the same tables give the same PNG bytes with the same matplotlib version.
Each file is written to ``<file>.tmp`` and renamed in place. Player names are the ``player_name``
values of the Main_List tables, the strings of the Main_List files.

Errors: ValueError or TypeError, naming the problem, for a missing table or column, an unknown
Role, or a table whose rows are not ranks 1, 2, 3, ... in order.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from superr.evidence import (
    MIX_PCT_COLUMNS,
    PHASE_NAMES,
    PRESSURE_MIX,
    RATING_EVIDENCE,
    ROLE_SUFFIXES,
    table_name,
)
from superr.reconstruct import COMPETITION_TYPES
from superr.value import BATTER_ROLE, BOWLER_ROLE, ROLES

__all__ = [
    "CHART_DPI",
    "CHART_FILES",
    "CHART_KINDS",
    "CHART_NAMES",
    "CLUTCH_USUAL",
    "COMPETITION_MIX",
    "FOREST",
    "LABEL_OFFSETS",
    "PHASE_MIX",
    "SEGMENT_LABEL_MIN",
    "chart_name",
    "clutch_usual_chart",
    "competition_mix_chart",
    "forest_chart",
    "phase_mix_chart",
    "write_charts",
]

#: Resolution of every chart in dots per inch, fixed so that the same tables give the same image.
CHART_DPI = 150

#: The four kinds of chart (module docstring), in file order.
FOREST = "interval_forest"
CLUTCH_USUAL = "clutch_vs_usual"
PHASE_MIX = "phase_mix"
COMPETITION_MIX = "competition_type_mix"
CHART_KINDS: tuple[str, ...] = (FOREST, CLUTCH_USUAL, PHASE_MIX, COMPETITION_MIX)


def chart_name(kind: str, role: str) -> str:
    """The name of one chart, such as ``"interval_forest_batters"``: its file stem."""
    if kind not in CHART_KINDS:
        raise ValueError(f"kind must be one of {', '.join(CHART_KINDS)}; got {kind!r}")
    if role not in ROLES:
        raise ValueError(f"role must be one of {', '.join(ROLES)}; got {role!r}")
    return f"{kind}_{ROLE_SUFFIXES[role]}"


#: Every chart, kind by kind and each kind batter first, and its PNG file in ``outputs/final/``.
CHART_NAMES: tuple[str, ...] = tuple(chart_name(kind, role) for kind in CHART_KINDS for role in ROLES)
CHART_FILES: Mapping[str, str] = {name: f"{name}.png" for name in CHART_NAMES}

#: Label offsets of the scatter plot, in points from the marker, tried in this order.
LABEL_OFFSETS: tuple[tuple[int, int], ...] = (
    (7, 5), (7, -5), (-7, 5), (-7, -5), (0, 10), (0, -10),
    (16, 14), (16, -14), (-16, 14), (-16, -14), (26, 0), (-26, 0), (0, 24), (0, -24),
    (30, 26), (-30, 26), (30, -26), (-30, -26), (44, 0), (-44, 0), (0, 38), (0, -38),
)  # fmt: skip

#: Smallest stacked-bar segment, in percent, that is labelled with its percentage.
SEGMENT_LABEL_MIN = 8.0

_ROLE_TITLES: Mapping[str, str] = {BATTER_ROLE: "batters", BOWLER_ROLE: "bowlers"}
_ROLE_NOUNS: Mapping[str, str] = {BATTER_ROLE: "batter", BOWLER_ROLE: "bowler"}
_PHASE_LABELS: Mapping[str, str] = {"powerplay": "Powerplay", "middle": "Middle", "death": "Death"}
_PHASE_COLOURS: tuple[str, ...] = ("#4c78a8", "#9ecae9", "#e45756")
_COMP_COLOURS: tuple[str, ...] = ("#f58518", "#4c78a8", "#54a24b")
_LIGHT_COLOURS = frozenset({"#9ecae9"})
_POINT_COLOUR = "#d95f02"
_INTERVAL_COLOUR = "#1f5fa8"
_OTHER_COLOUR = "#a6a6a6"
_GUIDE_COLOUR = "#7f7f7f"
_LEADER_COLOUR = "#4d4d4d"
_GRID_COLOUR = "#e6e6e6"
_NEAR_OFFSET = 12.0  # points; a label farther from its marker gets a leader line
_LABEL_FONT = 7.5
_MARKER_SIZE = 7.0  # points, the listed players' diamonds
_KEY = ("rank", "player_name")


# ========================================================================= argument checks

def _require(frame: Any, columns: Sequence[str], what: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{what} must be a DataFrame; got {type(frame).__name__}")
    absent = [column for column in dict.fromkeys(columns) if column not in frame.columns]
    if absent:
        raise ValueError(f"{what} lacks the column(s) {', '.join(absent)}")


def _role(role: Any) -> str:
    if not isinstance(role, str) or role not in ROLES:
        raise ValueError(f"role must be one of {', '.join(ROLES)}; got {role!r}")
    return role


def _ranked_labels(table: pd.DataFrame, what: str) -> list[str]:
    """``"<rank>. <player_name>"`` for each row; ValueError unless the ranks are 1, 2, 3, ... in order."""
    _require(table, _KEY, what)
    ranks = table["rank"].to_numpy(dtype=np.int64)
    if not len(ranks) or not np.array_equal(ranks, np.arange(1, len(ranks) + 1)):
        raise ValueError(f"{what} must hold ranks 1, 2, 3, ... in order; got {ranks.tolist()}")
    return [f"{int(rank)}. {name}" for rank, name in zip(ranks.tolist(), table["player_name"].tolist())]


def _floats(table: pd.DataFrame, column: str) -> np.ndarray:
    return table[column].to_numpy(dtype=np.float64, na_value=np.nan)


# =================================================================================== drawing

def _figure(width: float, height: float, *, constrained: bool = True) -> Any:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(width, height), dpi=CHART_DPI, layout="constrained" if constrained else None)
    FigureCanvasAgg(figure)
    return figure


def _save(figure: Any, path: Path) -> Path:
    """Save ``figure`` as PNG at CHART_DPI without a time stamp, through ``<file>.tmp`` renamed in place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        figure.savefig(temporary, format="png", dpi=CHART_DPI, metadata={"Software": None})
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def forest_chart(role: str, evidence: pd.DataFrame, path: Path | str, *, n_resamples: int | None = None) -> Path:
    """The interval forest plot of one Role (criterion 6; module docstring); returns the path written.

    ``evidence``: the Role's rating-evidence table (``rank``, ``player_name``, ``final_score``,
    ``ci_low``, ``ci_high``); ``n_resamples``: the number of Bootstrap_Resamples, shown in the
    title when given.
    """
    role = _role(role)
    labels = _ranked_labels(evidence, f"the {role} rating evidence")
    _require(evidence, ("final_score", "ci_low", "ci_high"), f"the {role} rating evidence")
    score, low, high = (_floats(evidence, column) for column in ("final_score", "ci_low", "ci_high"))
    n_players = len(labels)
    y = np.arange(n_players, dtype=np.float64)[::-1]
    figure = _figure(8.6, 0.42 * n_players + 2.0)
    ax = figure.subplots()
    has = np.isfinite(low) & np.isfinite(high)
    ax.hlines(y[has], low[has], high[has], colors=_INTERVAL_COLOUR, linewidth=2.4, label="95% percentile interval")
    ax.plot(score, y, linestyle="none", marker="o", markersize=6.5, color=_POINT_COLOUR, zorder=3, label="Final_Score")
    ax.axvline(0.0, color=_GUIDE_COLOUR, linestyle=":", linewidth=1.0)
    ax.set_yticks(y, labels, fontsize=8.5)
    ax.set_ylim(-0.7, n_players - 0.3)
    ax.grid(True, axis="x", color=_GRID_COLOUR, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Final_Score (runs per over)")
    over = f"over {n_resamples:,} Bootstrap_Resamples" if n_resamples else "over the Bootstrap_Resamples"
    missing = n_players - int(has.sum())
    note = f"; {missing} player(s) never ranked in a resample have no interval" if missing else ""
    ax.set_title(
        f"Top {n_players} {_ROLE_TITLES[role]}: Final_Score (point) and its 95% percentile interval (line)\n"
        f"{over}, in rank order{note}",
        fontsize=10,
    )
    figure.legend(loc="outside lower center", ncol=2, frameon=False, fontsize=8)
    return _save(figure, Path(path))


def _label_box(annotation: Any, renderer: Any) -> Any:
    """The display box of an annotation's text alone (no leader line), padded for its background patch."""
    from matplotlib.text import Text

    annotation.update_positions(renderer)
    return Text.get_window_extent(annotation, renderer).padded(2.0)


def _label_room(
    ax: Any, renderer: Any, xs: np.ndarray, texts: Sequence[str], x_lim: tuple[float, float]
) -> tuple[float, float]:
    """``x_lim`` with its upper end raised, where needed, so that each text fits inside the axes to the right
    of its ``x`` at the first label offset. The axes keep their size in the figure, so the data scale shrinks."""
    width = ax.get_window_extent(renderer).width
    gap = LABEL_OFFSETS[0][0] * CHART_DPI / 72.0 + 8.0  # pixels: the offset, the background patch and a margin
    low, high = x_lim
    for x, text in zip(xs.tolist(), texts):
        probe = ax.text(0.0, 0.0, text, fontsize=_LABEL_FONT)
        room = width - gap - probe.get_window_extent(renderer).width
        probe.remove()
        if room > 0 and x > low:
            high = max(high, low + (x - low) * width / room)
    return low, high


def _annotate(ax: Any, text: str, point: tuple[float, float], offset: tuple[int, int]) -> Any:
    """One scatter label at ``offset`` points from ``point``, aligned away from it; a leader line when far."""
    dx, dy = offset
    far = math.hypot(dx, dy) > _NEAR_OFFSET
    return ax.annotate(
        text, point, xytext=offset, textcoords="offset points", fontsize=_LABEL_FONT, zorder=6,
        ha="left" if dx > 0 else "right" if dx < 0 else "center",
        va="bottom" if dy > 0 else "top" if dy < 0 else "center",
        bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        arrowprops={"arrowstyle": "-", "color": _LEADER_COLOUR, "linewidth": 0.8, "shrinkA": 0, "shrinkB": 4}
        if far else None,
    )  # fmt: skip


def _place_labels(ax: Any, renderer: Any, points: Sequence[tuple[float, float]], texts: Sequence[str], taken: list[Any]) -> None:
    """Label each point (data coordinates), in order, at the first LABEL_OFFSETS offset whose text box lies
    inside the axes and overlaps no box in ``taken``; when none does, at the first offset. Each label's box
    joins ``taken``."""
    axes_box = ax.get_window_extent(renderer)
    for point, text in zip(points, texts):
        for offset in LABEL_OFFSETS:
            annotation = _annotate(ax, text, point, offset)
            box = _label_box(annotation, renderer)
            inside = axes_box.x0 <= box.x0 and box.x1 <= axes_box.x1 and axes_box.y0 <= box.y0 and box.y1 <= axes_box.y1
            if inside and not any(box.overlaps(other) for other in taken):
                break
            annotation.remove()
        else:  # no free offset: the first one, overlapping
            box = _label_box(_annotate(ax, text, point, LABEL_OFFSETS[0]), renderer)
        taken.append(box)


def clutch_usual_chart(role: str, ranking: pd.DataFrame, listed: pd.DataFrame, path: Path | str) -> Path:
    """The Clutch_Level against Usual_Baseline scatter plot of one Role (criterion 7; module docstring).

    ``ranking``: the rating checkpoint's ``player_estimates`` (``role``, ``player_id``,
    ``eligible``, ``clutch``, ``usual``, ``replacement_level``); ``listed``: the Role's Main_List
    (``rank``, ``player_id``, ``player_name``). Returns the path written.
    """
    from matplotlib.lines import Line2D
    from matplotlib.transforms import Bbox

    role = _role(role)
    labels = _ranked_labels(listed, f"the {role} list")
    _require(listed, ("player_id",), f"the {role} list")
    _require(ranking, ("role", "player_id", "eligible", "clutch", "usual", "replacement_level"), "ranking")
    roles = ranking["role"].astype("string").to_numpy(dtype=object, na_value=None)
    rows = ranking.loc[(roles == role) & ranking["eligible"].to_numpy(dtype=bool)].reset_index(drop=True)
    ids = rows["player_id"].to_numpy(dtype=np.int64)
    clutch, usual = _floats(rows, "clutch"), _floats(rows, "usual")
    placed = np.isfinite(clutch) & np.isfinite(usual)
    listed_ids = listed["player_id"].to_numpy(dtype=np.int64)
    at = pd.Index(ids).get_indexer(listed_ids)
    if (at < 0).any():
        raise ValueError(f"ranking has no {role} Eligible_Player row for a listed player_id")
    top = np.zeros(len(ids), dtype=bool)
    top[at] = True
    ranked = [j for j, position in enumerate(at.tolist()) if placed[position]]  # listed players with a point
    label_texts = [labels[j] for j in ranked]
    label_x = usual[at[ranked]]
    label_y = clutch[at[ranked]]

    figure = _figure(9.0, 7.2, constrained=False)
    figure.subplots_adjust(left=0.085, right=0.975, bottom=0.14, top=0.91)
    ax = figure.subplots()
    renderer = figure.canvas.get_renderer()
    x_values, y_values = usual[placed], clutch[placed]
    x_pad = 0.06 * max(float(np.ptp(x_values)), 1.0) if len(x_values) else 1.0
    y_pad = 0.06 * max(float(np.ptp(y_values)), 1.0) if len(y_values) else 1.0
    x_lim = (float(x_values.min()) - x_pad, float(x_values.max()) + x_pad) if len(x_values) else (-1.0, 1.0)
    y_lim = (float(y_values.min()) - y_pad, float(y_values.max()) + y_pad) if len(y_values) else (-1.0, 1.0)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    x_lim = _label_room(ax, renderer, label_x, label_texts, x_lim)
    ax.set_xlim(*x_lim)
    low, high = min(x_lim[0], y_lim[0]), max(x_lim[1], y_lim[1])
    ax.plot([low, high], [low, high], color=_GUIDE_COLOUR, linestyle="--", linewidth=0.9, zorder=1)
    levels = _floats(rows, "replacement_level")
    level = float(levels[np.isfinite(levels)][0]) if np.isfinite(levels).any() else math.nan
    if math.isfinite(level):
        ax.axhline(level, color=_GUIDE_COLOUR, linestyle=":", linewidth=1.0, zorder=1)
    others = placed & ~top
    ax.scatter(usual[others], clutch[others], s=16, color=_OTHER_COLOUR, edgecolors="none", zorder=2)
    ax.scatter(
        label_x, label_y, s=_MARKER_SIZE**2, marker="D", color=_POINT_COLOUR, edgecolors="black", linewidths=0.5,
        zorder=4,
    )  # fmt: skip
    ax.grid(True, color=_GRID_COLOUR, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Usual_Baseline (runs per over)")
    ax.set_ylabel("Clutch_Level (runs per over)")
    n_missing = int((~placed).sum())
    note = f"; {n_missing} without a Clutch_Level or Usual_Baseline not shown" if n_missing else ""
    ax.set_title(
        f"{_ROLE_TITLES[role].capitalize()}: Clutch_Level against Usual_Baseline, {len(ids):,} Eligible_Players"
        f"{note}\nTop 10 of the Main_List labelled with player_name",
        fontsize=10,
    )
    handles = [
        Line2D([], [], linestyle="none", marker="D", markersize=6, color=_POINT_COLOUR, markeredgecolor="black",
               markeredgewidth=0.5, label="Main_List player (top 10)"),
        Line2D([], [], linestyle="none", marker="o", markersize=4, color=_OTHER_COLOUR, label="Other Eligible_Player"),
        Line2D([], [], color=_GUIDE_COLOUR, linestyle="--", linewidth=0.9, label="Clutch_Level = Usual_Baseline"),
    ]  # fmt: skip
    if math.isfinite(level):
        handles.append(Line2D([], [], color=_GUIDE_COLOUR, linestyle=":", linewidth=1.0, label="Replacement_Level"))
    figure.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=8)

    # Labels, in rank order. The listed players' markers are taken first, so that no label covers one.
    radius = _MARKER_SIZE / 2.0 * CHART_DPI / 72.0 + 1.0
    taken = [
        Bbox.from_bounds(x - radius, y - radius, 2.0 * radius, 2.0 * radius)
        for x, y in ax.transData.transform(np.column_stack([label_x, label_y])).tolist()
    ]
    points = [(float(x), float(y)) for x, y in zip(label_x.tolist(), label_y.tolist())]
    _place_labels(ax, renderer, points, label_texts, taken)
    return _save(figure, Path(path))


def _stacked_chart(
    role: str,
    mix: pd.DataFrame,
    path: Path | str,
    *,
    columns: Sequence[str],
    names: Sequence[str],
    colours: Sequence[str],
    title: str,
) -> Path:
    labels = _ranked_labels(mix, f"the {role} Pressure_Ball mix")
    _require(mix, columns, f"the {role} Pressure_Ball mix")
    shares = np.column_stack([_floats(mix, column) for column in columns])
    n_players = len(labels)
    y = np.arange(n_players, dtype=np.float64)[::-1]
    figure = _figure(8.6, 0.42 * n_players + 2.0)
    ax = figure.subplots()
    left = np.zeros(n_players)
    for j, (name, colour) in enumerate(zip(names, colours)):
        values = np.nan_to_num(shares[:, j])
        ax.barh(y, values, left=left, height=0.68, color=colour, edgecolor="white", linewidth=0.6, label=name)
        text_colour = "black" if colour in _LIGHT_COLOURS else "white"
        for row in range(n_players):
            if values[row] >= SEGMENT_LABEL_MIN:
                ax.text(
                    left[row] + values[row] / 2.0, y[row], f"{values[row]:.0f}%", ha="center", va="center",
                    fontsize=7.5, color=text_colour,
                )  # fmt: skip
        left = left + values
    ax.set_yticks(y, labels, fontsize=8.5)
    ax.set_ylim(-0.6, n_players - 0.4)
    ax.set_xlim(0.0, 100.0)
    ax.set_xticks(np.arange(0.0, 100.1, 20.0))
    ax.set_xlabel(f"Share of the player's {_ROLE_NOUNS[role]}-Role Pressure_Balls (%)")
    ax.set_title(title, fontsize=10)
    figure.legend(loc="outside lower center", ncol=len(names), frameon=False, fontsize=8)
    return _save(figure, Path(path))


def phase_mix_chart(role: str, mix: pd.DataFrame, path: Path | str) -> Path:
    """The Phase-mix stacked bars of one Role (criterion 8; module docstring); returns the path written.

    ``mix``: the Role's Pressure_Ball mix table (``rank``, ``player_name`` and the Phase percentages).
    """
    role = _role(role)
    return _stacked_chart(
        role, mix, path, columns=MIX_PCT_COLUMNS["phase"], names=[_PHASE_LABELS[name] for name in PHASE_NAMES],
        colours=_PHASE_COLOURS,
        title=f"Top {len(mix)} {_ROLE_TITLES[role]}: Pressure_Balls by Phase (unweighted, % of each player's)",
    )  # fmt: skip


def competition_mix_chart(role: str, mix: pd.DataFrame, path: Path | str) -> Path:
    """The Competition_Type-mix stacked bars of one Role (criterion 9; module docstring); returns the path written.

    ``mix``: the Role's Pressure_Ball mix table (``rank``, ``player_name`` and the Competition_Type percentages).
    """
    role = _role(role)
    return _stacked_chart(
        role, mix, path, columns=MIX_PCT_COLUMNS["comp_type"], names=list(COMPETITION_TYPES), colours=_COMP_COLOURS,
        title=f"Top {len(mix)} {_ROLE_TITLES[role]}: Pressure_Balls by Competition_Type (unweighted, % of each player's)",
    )  # fmt: skip


def _resample_count(bootstrap: pd.DataFrame | None, role: str) -> int | None:
    """The number of Bootstrap_Resamples behind the Role's bootstrap rows (``n_ranked`` + ``n_unranked``), if one."""
    if bootstrap is None or not {"role", "n_ranked", "n_unranked"} <= set(bootstrap.columns):
        return None
    roles = bootstrap["role"].astype("string").to_numpy(dtype=object, na_value=None)
    rows = bootstrap.loc[roles == role]
    totals = (rows["n_ranked"].to_numpy(dtype=np.int64) + rows["n_unranked"].to_numpy(dtype=np.int64)).tolist()
    return int(totals[0]) if totals and len(set(totals)) == 1 else None


def write_charts(
    outdir: Path | str,
    *,
    lists: Mapping[str, pd.DataFrame],
    tables: Mapping[str, pd.DataFrame],
    ranking: pd.DataFrame,
    bootstrap: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Draw and write the eight charts to ``outdir`` (CHART_FILES; criteria 6-10); return ``{name: path}``.

    ``lists``: each Role's Main_List table; ``tables``: the evidence tables
    (``evidence.evidence_tables``); ``ranking``: the rating checkpoint's ``player_estimates``;
    ``bootstrap``: the bootstrap checkpoint's ``bootstrap`` table, read only for the number of
    Bootstrap_Resamples in the forest plot titles. Returns the paths in CHART_NAMES order.
    """
    if not isinstance(lists, Mapping) or set(lists) != set(ROLES):
        raise ValueError(f"lists must map exactly the Roles {', '.join(ROLES)} to their Main_List tables")
    if not isinstance(tables, Mapping):
        raise TypeError(f"tables must map the evidence table names to the tables; got {type(tables).__name__}")
    folder = Path(outdir)
    folder.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for role in ROLES:
        evidence = tables[table_name(RATING_EVIDENCE, role)]
        mix = tables[table_name(PRESSURE_MIX, role)]
        files = {kind: folder / CHART_FILES[chart_name(kind, role)] for kind in CHART_KINDS}
        paths[chart_name(FOREST, role)] = forest_chart(
            role, evidence, files[FOREST], n_resamples=_resample_count(bootstrap, role)
        )
        paths[chart_name(CLUTCH_USUAL, role)] = clutch_usual_chart(role, ranking, lists[role], files[CLUTCH_USUAL])
        paths[chart_name(PHASE_MIX, role)] = phase_mix_chart(role, mix, files[PHASE_MIX])
        paths[chart_name(COMPETITION_MIX, role)] = competition_mix_chart(role, mix, files[COMPETITION_MIX])
    return {name: paths[name] for name in CHART_NAMES}
