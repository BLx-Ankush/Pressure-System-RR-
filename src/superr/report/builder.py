"""Report builder: the methodology report as a python-docx Document.

Requirement 27, criteria 1, 3, 4, 5 and 13, and Requirement 2, criterion 8; design section
"Report_Builder" and decision DD-18. This module builds the document skeleton and the helpers that
the section functions of ``report/sections.py`` fill it with. ``report/export.py`` writes the DOCX
and the PDF, and ``report/checks.py`` runs the Report_Check.

Structure (27.1, 27.3)
----------------------
The document opens with a "Title" paragraph (not a heading) and, when the section code sets it,
one 9 pt line below it (``ReportBuilder.set_title_line``: the Run date and the command). The seven
top-level sections of SECTION_HEADINGS follow, in the order of Requirement 27, criterion 1; their
headings are the only "Heading 1" paragraphs. Inside a section:

* ``Section.add_heading(text)`` adds a "Heading 2", and only the next text of the section's
  SUBHEADINGS entry, so a wrong, extra or out-of-order Heading 2 raises ValueError in the section
  code that adds it. Section 0 and the Appendix have no Heading 2. A Heading 2 that is never added
  is not an error here: the Report_Check's ``headings`` check reports it.
* The report has no Heading 3. A sub-part is a paragraph that opens with a bold run-in label:
  ``Section.add_run_in("A.1 Reproduction.", "The command ...")``.
* No heading starts a new page, so the Appendix follows Section 5 directly (the page budget covers
  the whole PDF).

A section with no content gets the paragraph PLACEHOLDER_TEXT.

Content is collected first and written by ``build``, section by section and in the order it was
added, so a section may be filled at any time and the Method_Constant labels still land at the
first appearance in document order. Tables are rendered from their CSV when added and equations
are checked when added, so mistakes fail at the section code that made them.

What the report may embed (27.5)
--------------------------------
``ReportBuilder.allow(tables=..., figures=...)`` names the only table_ids and image files that the
report may embed. Call it once, before adding any table or image; until then ``add_table`` and
``add_image`` refuse everything. ``Section.add_table`` then refuses a table_id outside ``tables``,
and ``Section.add_image`` refuses an image outside ``figures`` and any image outside Section 4
(FIGURE_SECTION). Each table_id and each image can be added once. The caller passes both lists:
``REPORT_TABLES`` of ``report/sections.py`` and the two files that ``write_report_figures`` writes,
``outputs/report/figure_intervals.png`` and ``outputs/report/figure_clutch_usual.png``. This module
therefore imports neither ``sections.py`` nor ``figures.py``, and the modules have no import cycle.

Layout (DD-18)
--------------
A4 with 2 cm margins and the page number in the footer. Body text 10.5 pt, line spacing 1.0 and
3 pt after; list items 1 pt after. "Title" 16 pt. Heading 1 13 pt (10 pt before, 3 pt after) and
Heading 2 11 pt (6 pt before, 2 pt after), both kept with the next paragraph. Captions, the title
line and the equations' "where" lines are 9 pt (SMALL_FONT_PT; the last two in NOTE_STYLE). Tables
come from ``superr.report.tables.write_docx_table`` (8.5 pt, rows that never split, a bold, shaded
header row repeated on each page), and the paragraph after a table starts 6 pt below it. Equations
are pictures from ``superr.report.equations`` (10.5 pt).

Paragraph parts and Method_Constants (Requirement 2, criterion 8)
-----------------------------------------------------------------
``add_paragraph``, ``add_run_in``, ``add_bullet``, the captions, the title line and the equations'
symbol definitions take parts: plain ``str``, ``Bold``, ``Italic`` and ``Const``. ``Const(name)``
shows a Method_Constant's text followed, at its first use in the report only, by its
ConstantRegistry label in square brackets: ``0.5 [hand-set (D2)]``,
``90 [data-derived (Rating_Engine)]``. ``build`` calls ``ConstantRegistry.reset_labels()`` first
and ``ConstantRegistry.label(name)`` at each Const, in document order. ``Const(name, text)`` shows
``text``; ``Const(name)`` formats the value itself (``format_constant_value``): the recorded value
of a data-derived constant, or the Config value (the builder needs ``settings`` then). An unknown
name raises UnknownConstantError when the Const is added. Table cells come from CSVs and never hold
Consts.

Numbering, captions and sources
-------------------------------
Tables, figures and equations are numbered within their section, with the section's prefix
(SECTION_PREFIXES: "0"-"5", "A" for the Appendix): ``add_table`` returns "Table 1.2",
``add_image`` "Figure 4.1", ``add_equation`` "Equation 1.2" (shown as the tag "(1.2)" at the right
margin), so section text can refer to them as soon as they are added. A table has its caption above
it and a figure below it. A caption is the bold label ("Table 1.2."), the caption text, a full stop
when the text does not end with one, and the sentence "Source: <file>." that names the table's CSV
or the image file relative to the workspace root, with forward slashes (``_shown_path``; a
backslash would read as LaTeX residue).

Equations (27.4, 27.13)
-----------------------
``add_equation(mathtext, description, symbols)``: the mathtext is checked when added and drawn as a
picture at build time; the description, in plain words, is the picture's alt text. ``symbols`` is
the non-empty list of the equation's symbols: (symbol, definition) pairs, or a mapping symbol ->
definition, one symbol per entry. A symbol is one part, normally a str such as ``"α"`` or
``"w_L"``; a definition is a str or a sequence of parts, without a final full stop. Below the
equation the builder writes the 9 pt "where" line: "where " + each symbol, " is " and its
definition, joined by "; ", and a full stop.

Use (tasks 31.5 and 31.6)
-------------------------
::

    builder = ReportBuilder.from_context(ctx)        # ctx.constants holds every stage's constants
    figures = write_report_figures(inputs)           # report/figures.py: the two Section 4 files
    builder.allow(tables=REPORT_TABLES, figures=figures.values())
    builder.set_title_line("Run of 1 June 2025. The command python run_pipeline.py rebuilds this report.")
    s1 = builder.section(1)
    s1.add_heading("1.1 Match situations")           # SUBHEADINGS[s1.heading][0], else ValueError
    s1.add_run_in("Live_Ball.", "A Delivery is a Live_Ball when ...")
    s1.add_table(TableSpec(                          # a table_id outside REPORT_TABLES raises
        "s1_pressure_groups", outputs / "report" / "pressure_groups.csv",
        ["role", "group", "n_pressure"], {"n_pressure": 0}, ["role", "group"],
        caption="Pressure_Balls by Phase, Innings_Group and Competition_Type",
    ))
    s1.add_heading("1.2 Indicators")
    ref = s1.add_equation(
        r"\\mathrm{PI} = 100 \\times [\\alpha \\, \\mathrm{rank}(\\mathrm{LI}) + (1 - \\alpha) \\, "
        r"\\mathrm{rank}(\\mathrm{Stress})]",
        "PI blends the percentile ranks of LI and Stress",
        [
            ("PI", "the Pressure_Index of the Delivery, from 0 to 100"),
            ("α", ("the weight of LI, ", Const("design_choices.D2_alpha"))),
            ("rank(x)", "the percentile rank of x among the Live_Balls of the same Role and innings"),
        ],
    )                                                # "where PI is ...; α is the weight of LI, 0.5 [hand-set (D2)]; ..."
    built = builder.save(outputs / "report" / "<name>.docx")
    # report/export.py: report_check(built.path, built.specs, ctx.settings)

Outside a Run, ``load_run_constants(settings)`` gathers the ``_constants.json`` of every complete
stage checkpoint into one ConstantRegistry. Inside a Run the runner has already done this:
``ctx.constants`` accumulates each stage's constants, and ``--from-stage`` reloads the saved ones.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from superr.report.constants import ConstantRegistry, UnknownConstantError
from superr.report.equations import add_picture, check_mathtext, latex_residue, plain_text_problems, render_equation
from superr.report.tables import RenderedTable, TableSpec, render_table, write_docx_table

if TYPE_CHECKING:
    from docx.document import Document

    from superr.pipeline import StageContext
    from superr.settings import Settings
    from superr.store import CheckpointStore

__all__ = [
    "APPENDIX",
    "BODY_FONT_PT",
    "FIGURE_SECTION",
    "MARGIN_CM",
    "NOTE_STYLE",
    "PAGE_SIZE_CM",
    "PLACEHOLDER_TEXT",
    "REPORT_TITLE",
    "SECTION_HEADINGS",
    "SECTION_PREFIXES",
    "SMALL_FONT_PT",
    "STAGE_NAME",
    "SUBHEADINGS",
    "TITLE_FONT_PT",
    "Bold",
    "BuiltReport",
    "Const",
    "Italic",
    "Part",
    "ReportBuilder",
    "Section",
    "Symbols",
    "format_constant_value",
    "load_run_constants",
    "top_level_headings",
]

#: The Pipeline_Stage that builds the report.
STAGE_NAME = "report"

#: The top-level headings of the report, in the order of Requirement 27, criterion 1.
SECTION_HEADINGS: tuple[str, ...] = (
    "Section 0 Executive summary",
    "Section 1 Pressure definition",
    "Section 2 Performance measurement",
    "Section 3 Fair comparisons",
    "Section 4 Evidence for the top 10",
    "Section 5 Reliability and limitations",
    "Appendix",
)
#: Heading 1 text -> its Heading 2 texts, exact and in order (Requirement 27, criterion 3): one for
#: each sub-point that the problem statement sets for Sections 1-5, and none for Section 0 and the
#: Appendix. ``Section.add_heading`` accepts only these, and the Report_Check compares the DOCX with them.
SUBHEADINGS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    dict(
        zip(
            SECTION_HEADINGS,
            (
                (),  # Section 0
                (
                    "1.1 Match situations",
                    "1.2 Indicators",
                    "1.3 Why the definition is meaningful, and its validation",
                ),
                (
                    "2.1 Batting metrics",
                    "2.2 Bowling metrics",
                    "2.3 Player baselines",
                    "2.4 Rating and scoring logic",
                ),
                (
                    "3.1 Sample sizes",
                    "3.2 Player eligibility",
                    "3.3 Different match contexts",
                    "3.4 IPL and T20I data",
                ),
                (
                    "4.1 Results",
                    "4.2 Comparisons, tables and charts that support the selections and their order",
                ),
                (
                    "5.1 Consistency across matches",
                    "5.2 Where the model is less certain",
                ),
                (),  # Appendix
            ),
            strict=True,
        )
    )
)
#: ``ReportBuilder.section`` key of the Appendix (sections 0-5 are keyed by their number).
APPENDIX = "appendix"
#: Numbering prefix of each section's tables, figures and equations ("Table A.1" in the Appendix).
SECTION_PREFIXES: tuple[str, ...] = ("0", "1", "2", "3", "4", "5", "A")
#: The one section that holds figures: Section 4 and its two charts (Requirement 27, criterion 9).
FIGURE_SECTION = 4
#: The paragraph that ``build`` writes in a section with no content.
PLACEHOLDER_TEXT = "This section has no content yet."
#: The default title paragraph above Section 0.
REPORT_TITLE = "Pressure rating system: methodology report"

#: Page and text (design section "Report_Builder", layout table): A4 portrait with equal margins,
#: 10.5 pt body text, a 16 pt title, and 9 pt captions, title line and equation "where" lines.
PAGE_SIZE_CM = (21.0, 29.7)
MARGIN_CM = 2.0
BODY_FONT_PT = 10.5
TITLE_FONT_PT = 16.0
SMALL_FONT_PT = 9.0
#: Paragraph style of the 9 pt lines that are not captions: the title line and each "where" line.
NOTE_STYLE = "Note"
_BODY_LINE_SPACING = 1.0
_BODY_SPACE_AFTER_PT = 3.0
_LIST_SPACE_AFTER_PT = 1.0
_TITLE_SPACE_AFTER_PT = 3.0
#: Heading level -> (font size, space before, space after) in points; both keep with the next paragraph.
_HEADING_LAYOUT: Mapping[int, tuple[float, float, float]] = MappingProxyType({1: (13.0, 10.0, 3.0), 2: (11.0, 6.0, 2.0)})
_GAP_PT = 6.0  # before a table's caption, after a figure's caption, and before the paragraph after a table
_NEAR_PT = 2.0  # between a caption and its table or figure, and between an equation and its "where" line
_EQUATION_SIDE_CM = 1.5  # room left on each side of an equation, for its tag
_SENTENCE_END = (".", "!", "?")
_RESERVED_STYLES = frozenset({"Title"} | {f"Heading {level}" for level in range(1, 10)})
_APPENDIX_INDEX = len(SECTION_HEADINGS) - 1


# ------------------------------------------------------------------------------------ parts

@dataclass(frozen=True)
class Const:
    """A Method_Constant in a paragraph: ``text`` (default: the value, formatted), then its label at first use."""

    name: str
    text: str | None = None


@dataclass(frozen=True)
class Bold:
    """Bold text in a paragraph."""

    text: str


@dataclass(frozen=True)
class Italic:
    """Italic text in a paragraph."""

    text: str


#: One part of a paragraph or caption.
Part = str | Const | Bold | Italic

#: The symbol list of an equation (``Section.add_equation``): (symbol, definition) pairs, or a
#: mapping symbol -> definition. A symbol is one part; a definition is a str or a sequence of parts.
Symbols = Sequence[tuple[Part, str | Sequence[Part]]] | Mapping[Part, str | Sequence[Part]]


def format_constant_value(value: Any, name: str = "") -> str:
    """Default text of a Const: a str as it is; an int or bool with str(); a float with up to 6
    decimals and no trailing zeros (0.5, 0.300689); a flat list as its items joined by ", ".

    ValueError for anything else (a dict or a nested list): give the Const its text instead.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return str(value)
    if isinstance(value, float):
        return _shown_float(value)
    if isinstance(value, (list, tuple)) and not any(isinstance(item, (list, tuple, dict)) for item in value):
        return ", ".join(format_constant_value(item, name) for item in value)
    raise ValueError(f"Const({name!r}) needs its text: the value {value!r} has no default formatting")


# ---------------------------------------------------------------------------- the content

@dataclass(frozen=True)
class _Heading:
    text: str  # a Heading 2 of SUBHEADINGS


@dataclass(frozen=True)
class _Paragraph:
    parts: tuple[Part, ...]
    style: str | None


@dataclass(frozen=True)
class _Table:
    rendered: RenderedTable
    caption: tuple[Part, ...]
    label: str


@dataclass(frozen=True)
class _Figure:
    path: Path
    caption: tuple[Part, ...]
    alt_text: str
    label: str
    width_cm: float | None


@dataclass(frozen=True)
class _Equation:
    mathtext: str
    description: str
    label: str
    tag: str
    file_name: str
    where: tuple[Part, ...]  # the 9 pt "where" line below the equation


class Section:
    """One top-level section of the report: its heading and the content added to it, in order."""

    def __init__(self, builder: ReportBuilder, index: int) -> None:
        self._builder = builder
        self._index = index
        self._blocks: list[Any] = []
        self._counts = {"table": 0, "figure": 0, "equation": 0}
        self._subheadings = 0  # how many of the section's SUBHEADINGS texts have been added

    def __repr__(self) -> str:
        return f"Section({self.heading!r}, blocks={len(self._blocks)})"

    @property
    def heading(self) -> str:
        """The section's top-level heading (SECTION_HEADINGS)."""
        return SECTION_HEADINGS[self._index]

    @property
    def number(self) -> int | None:
        """0-5 for Sections 0-5; None for the Appendix."""
        return None if self._index == _APPENDIX_INDEX else self._index

    @property
    def prefix(self) -> str:
        """Numbering prefix of the section's tables, figures and equations (SECTION_PREFIXES)."""
        return SECTION_PREFIXES[self._index]

    @property
    def is_empty(self) -> bool:
        """True while nothing has been added, so build writes PLACEHOLDER_TEXT."""
        return not self._blocks

    def add_heading(self, text: str, level: int = 2) -> None:
        """Add the section's next "Heading 2" (Requirement 27, criterion 3).

        text must equal the next text of ``SUBHEADINGS[self.heading]`` exactly, so the Heading 2
        texts of a section come in that order, each once. ValueError for any other text, once the
        section has all of them, in Section 0 and the Appendix (which have none), and for a level
        other than 2: the Heading 1 texts are SECTION_HEADINGS, written by the builder, and the
        report has no Heading 3 (a sub-part opens with a bold run-in label, ``add_run_in``).
        """
        if type(level) is not int or level != 2:
            raise ValueError(
                f"Section.add_heading adds a Heading 2 only, got level {level!r}: the Heading 1 texts are "
                "SECTION_HEADINGS, written by the builder, and the report has no Heading 3; write a sub-part "
                "as a paragraph that opens with a bold run-in label (Section.add_run_in)"
            )
        expected = SUBHEADINGS[self.heading]
        if not expected:
            raise ValueError(
                f"{self.heading!r} has no Heading 2 (Requirement 27, criterion 3), got {text!r}; write a "
                "sub-part as a paragraph that opens with a bold run-in label (Section.add_run_in)"
            )
        if self._subheadings == len(expected):
            raise ValueError(
                f"{self.heading!r} already has all its Heading 2 texts {list(expected)} "
                f"(Requirement 27, criterion 3), got {text!r}"
            )
        following = expected[self._subheadings]
        if text != following:
            raise ValueError(
                f"the next Heading 2 of {self.heading!r} is {following!r}, got {text!r}; the section's Heading 2 "
                f"texts are {list(expected)}, exactly and in that order (Requirement 27, criterion 3)"
            )
        self._blocks.append(_Heading(text))
        self._subheadings += 1

    def add_paragraph(self, *parts: Part, style: str | None = None) -> None:
        """Add a paragraph made of parts (str, Bold, Italic, Const), in a paragraph style of the template."""
        if style is not None:
            _check_paragraph_style(style)
        self._blocks.append(_Paragraph(self._builder._resolve(parts, "a paragraph"), style))

    def add_run_in(self, label: str, *parts: Part) -> None:
        """Add a paragraph that opens with a bold run-in label: the report's form of a sub-part (no Heading 3).

        label: shown as given, in bold and followed by one space, normally ending with a full stop
        ("A.1 Reproduction.", "V1 result."). parts: the text that runs on after it (str, Bold,
        Italic, Const), which must not be blank. ValueError for a blank label or text.
        """
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"a run-in label needs non-blank text, got {label!r}")
        text = self._builder._resolve(parts, f"the paragraph after the run-in label {label!r}")
        self._blocks.append(_Paragraph((Bold(label.strip()), " ", *text), None))

    def add_bullet(self, *parts: Part) -> None:
        """Add one item of a bulleted list ("List Bullet")."""
        self.add_paragraph(*parts, style="List Bullet")

    def add_table(self, spec: TableSpec, *, caption: str | Sequence[Part] | None = None) -> str:
        """Add a table rendered from its CSV now, with ``caption`` (default ``spec.caption``) above it.

        Returns its reference, such as "Table 1.2". The caption ends with the sentence naming the
        CSV. Raises ValueError for a table_id that ReportBuilder.allow did not name (Requirement 27,
        criterion 5) or that is already in the report and for a blank caption, and TableError when
        the CSV cannot be rendered.
        """
        if not isinstance(spec, TableSpec):
            raise TypeError(f"add_table needs a TableSpec, got {type(spec).__name__}")
        builder = self._builder
        builder._check_table(spec.table_id)
        if spec.table_id in builder._table_ids:
            raise ValueError(f"the report already has a table with the table_id {spec.table_id!r}")
        what = f"the caption of table {spec.table_id!r}"
        parts = _trim_end(builder._resolve(_as_parts(spec.caption if caption is None else caption), what))
        rendered = render_table(spec)
        builder._table_ids.add(spec.table_id)
        label = f"Table {self._next('table')}"
        self._blocks.append(_Table(rendered, parts, label))
        return label

    def add_image(self, path: Path | str, caption: str | Sequence[Part], *, width_cm: float | None = None) -> str:
        """Add a figure with ``caption`` below it; returns its reference, such as "Figure 4.1".

        Only an image that ReportBuilder.allow named, only in Section 4 (FIGURE_SECTION) and only
        once (Requirement 27, criteria 5 and 9); ValueError otherwise. width_cm: the shown width;
        None keeps the image's natural size. Either way the width is at most the text width. The
        caption, without labels, is also the picture's alt text, so it must be plain words; the
        shown caption ends with the sentence naming the image file. FileNotFoundError when the image
        does not exist.
        """
        source = Path(path)
        builder = self._builder
        key = builder._check_figure(source, self)
        if not source.is_file():
            raise FileNotFoundError(f"the report image {source} does not exist")
        if width_cm is not None and (isinstance(width_cm, bool) or not isinstance(width_cm, (int, float)) or not width_cm > 0):
            raise ValueError(f"width_cm must be None or a positive number, got {width_cm!r}")
        parts = _trim_end(builder._resolve(_as_parts(caption), f"the caption of the image {source.name}"))
        alt_text = "".join(_plain(part) for part in parts).strip()
        problems = plain_text_problems(alt_text, f"the caption of the image {source.name}")
        if problems:
            raise ValueError("; ".join(problems))
        builder._figure_keys.add(key)
        label = f"Figure {self._next('figure')}"
        self._blocks.append(_Figure(source, parts, alt_text, label, None if width_cm is None else float(width_cm)))
        return label

    def add_equation(self, mathtext: str, description: str, symbols: Symbols) -> str:
        """Add an equation drawn from matplotlib mathtext; returns its reference, such as "Equation 1.2".

        The equation is shown centred as a picture with the tag "(1.2)" at the right margin, and
        the 9 pt "where" line below it defines each of its symbols (Requirement 27, criterion 4).
        description: plain words, stored as the picture's alt text. symbols: the non-empty list of
        (symbol, definition) pairs, or a mapping symbol -> definition (module docstring,
        "Equations"). ValueError for mathtext that does not parse, a description that is blank or
        holds raw LaTeX, an empty symbol list, a symbol listed twice, a blank symbol or definition,
        and a symbol or definition that holds raw LaTeX; TypeError for an entry that is not a pair.
        """
        text = check_mathtext(mathtext)
        problems = plain_text_problems(description, "an equation's description")
        if problems:
            raise ValueError("; ".join(problems))
        where = self._builder._where_line(symbols)
        number = self._next("equation")
        self._blocks.append(
            _Equation(
                text, description.strip(), f"Equation {number}", f"({number})",
                f"equation_{number.replace('.', '_')}.png", where,
            )
        )  # fmt: skip
        return f"Equation {number}"

    def _next(self, kind: str) -> str:
        self._counts[kind] += 1
        return f"{self.prefix}.{self._counts[kind]}"


# ---------------------------------------------------------------------------- the builder

@dataclass(frozen=True)
class BuiltReport:
    """A built report: the Document and what the Report_Check needs to check it.

    tables: every table in document order, with the rendered cells written into the DOCX.
    labels: (constant name, label) at each labelled first use, in document order.
    equation_files: the equation PNGs, when ``build`` was given an image_dir (else empty).
    path: the saved DOCX, set by ``ReportBuilder.save``.
    """

    document: Document
    tables: tuple[RenderedTable, ...]
    labels: tuple[tuple[str, str], ...]
    equation_files: tuple[Path, ...] = ()
    path: Path | None = None

    @property
    def specs(self) -> tuple[TableSpec, ...]:
        """The TableSpec of every table, in document order (the Report_Check input)."""
        return tuple(table.spec for table in self.tables)


class ReportBuilder:
    """The report's seven sections and their content, written to a python-docx Document by ``build``.

    constants: the Run's ConstantRegistry (``ctx.constants`` or ``load_run_constants``). settings:
    the Config, for Consts without text; optional. title: the "Title" paragraph above Section 0
    (None for no title). root: the workspace root, for the source paths in the captions (default
    ``settings.root``); a file outside it is shown by its name only. Before any table or image is
    added, ``allow`` must name the tables and figures that the report may embed.
    """

    def __init__(
        self,
        constants: ConstantRegistry,
        *,
        settings: Settings | None = None,
        title: str | None = REPORT_TITLE,
        root: Path | str | None = None,
    ) -> None:
        if not isinstance(constants, ConstantRegistry):
            raise TypeError(f"constants must be a ConstantRegistry, got {type(constants).__name__}")
        if title is not None and (not isinstance(title, str) or not title.strip()):
            raise ValueError(f"title must be None or non-blank text, got {title!r}")
        self._constants = constants
        self._settings = settings
        self._title = title
        if root is None and settings is not None:
            root = settings.root
        self._root = None if root is None else Path(root).resolve()
        self._sections = tuple(Section(self, index) for index in range(len(SECTION_HEADINGS)))
        self._table_ids: set[str] = set()
        self._allowed_tables: tuple[str, ...] | None = None  # set once by allow
        self._allowed_figures: dict[str, Path] | None = None  # _path_key -> path, set once by allow
        self._figure_keys: set[str] = set()  # the allowed figures already added
        self._title_line: tuple[Part, ...] | None = None

    @classmethod
    def from_context(cls, ctx: StageContext, *, title: str | None = REPORT_TITLE) -> ReportBuilder:
        """Builder for a report stage: the Run's constants, Config and workspace root."""
        return cls(ctx.constants, settings=ctx.settings, title=title, root=ctx.root)

    def __repr__(self) -> str:
        filled = sum(not section.is_empty for section in self._sections)
        return f"ReportBuilder(sections={len(self._sections)}, filled={filled}, tables={len(self._table_ids)})"

    @property
    def constants(self) -> ConstantRegistry:
        return self._constants

    @property
    def sections(self) -> tuple[Section, ...]:
        """Sections 0-5 and the Appendix, in report order."""
        return self._sections

    @property
    def appendix(self) -> Section:
        return self._sections[_APPENDIX_INDEX]

    @property
    def allowed_tables(self) -> tuple[str, ...] | None:
        """The table_ids that ``allow`` named, in the given order; None before ``allow``."""
        return self._allowed_tables

    @property
    def allowed_figures(self) -> tuple[Path, ...] | None:
        """The image files that ``allow`` named, in the given order; None before ``allow``."""
        return None if self._allowed_figures is None else tuple(self._allowed_figures.values())

    def section(self, which: int | str) -> Section:
        """Section 0-5 by its number, or the Appendix by APPENDIX ("appendix", any letter case)."""
        if isinstance(which, str) and which.strip().lower() == APPENDIX:
            return self.appendix
        if type(which) is int and 0 <= which < _APPENDIX_INDEX:
            return self._sections[which]
        raise KeyError(f"the report has Sections 0-{_APPENDIX_INDEX - 1} and {APPENDIX!r}; got {which!r}")

    def allow(self, *, tables: Iterable[str] = (), figures: Iterable[Path | str] = ()) -> None:
        """Name the only tables and figures that the report may embed (Requirement 27, criterion 5).

        tables: the table_ids that ``Section.add_table`` accepts (``REPORT_TABLES`` of
        ``report/sections.py``). figures: the image files that ``Section.add_image`` accepts, in
        Section 4 only (the files of ``write_report_figures``); a file is matched by its absolute
        path. Call once, before adding any table or image; until then both refuse everything.
        ValueError for a second call, a blank or repeated table_id, and a repeated file; TypeError
        for a single str or path instead of a collection.
        """
        if self._allowed_tables is not None:
            raise ValueError("ReportBuilder.allow has already named the report's tables and figures; call it once")
        if isinstance(tables, (str, bytes)) or isinstance(figures, (str, bytes, os.PathLike)):
            raise TypeError("allow takes collections: tables=REPORT_TABLES, figures=[path, path]")
        ids = tuple(tables)
        bad = [table_id for table_id in ids if not isinstance(table_id, str) or not table_id.strip()]
        if bad:
            raise ValueError(f"allow: tables must be non-blank table_id strings, got {bad!r}")
        repeated = sorted({table_id for table_id in ids if ids.count(table_id) > 1})
        if repeated:
            raise ValueError(f"allow: tables names {repeated} more than once")
        files: dict[str, Path] = {}
        for item in figures:
            if not isinstance(item, (str, os.PathLike)):
                raise TypeError(f"allow: figures must be file paths, got {type(item).__name__}")
            path = Path(item)
            key = _path_key(path)
            if key in files:
                raise ValueError(f"allow: figures names {path} more than once")
            files[key] = path
        self._allowed_tables = ids
        self._allowed_figures = files

    def set_title_line(self, *parts: Part) -> None:
        """Set the one 9 pt line below the title (DD-18: the Run date and ``python run_pipeline.py``).

        parts: str, Bold, Italic and Const, as in a paragraph. ValueError when the report has no
        title, when the line is already set, and for a line with no text.
        """
        if self._title is None:
            raise ValueError("the report has no title, so it has no title line")
        if self._title_line is not None:
            raise ValueError("the title block has one 9 pt line, and it is already set")
        self._title_line = self._resolve(parts, "the title line")

    def build(self, *, image_dir: Path | str | None = None) -> BuiltReport:
        """Write every section to a new python-docx Document and return it with its tables and labels.

        image_dir: folder for the equation PNGs, kept after the build; None renders them in a
        temporary folder that is deleted afterwards (the pictures are embedded either way).
        Each build starts the Method_Constant labels afresh.
        """
        with contextlib.ExitStack() as stack:
            if image_dir is None:
                folder = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="superr_report_")))
            else:
                folder = Path(image_dir)
                folder.mkdir(parents=True, exist_ok=True)
            writer = _Writer(self, folder)
            document = writer.write()
            return BuiltReport(
                document=document,
                tables=tuple(writer.tables),
                labels=tuple(writer.labels),
                equation_files=() if image_dir is None else tuple(writer.equation_files),
            )

    def save(self, path: Path | str, *, image_dir: Path | str | None = None) -> BuiltReport:
        """Build and save the DOCX at ``path`` (through ``<file>.tmp`` renamed in place); return the build."""
        target = Path(path)
        if target.suffix.lower() != ".docx":
            raise ValueError(f"the report file must end in .docx, got {target.name!r}")
        built = self.build(image_dir=image_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        try:
            built.document.save(str(temporary))
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return dataclasses.replace(built, path=target)

    # -- internals ---------------------------------------------------------------------------

    def _check_table(self, table_id: str) -> None:
        """ValueError unless ``allow`` named ``table_id`` (Requirement 27, criterion 5)."""
        if self._allowed_tables is None:
            raise ValueError(
                f"no table can be added before ReportBuilder.allow names the report's tables (REPORT_TABLES); "
                f"got {table_id!r}"
            )
        if table_id not in self._allowed_tables:
            raise ValueError(
                f"the report embeds only the tables that ReportBuilder.allow named (REPORT_TABLES, Requirement 27, "
                f"criterion 5); {table_id!r} is not one of them"
            )

    def _check_figure(self, source: Path, section: Section) -> str:
        """The key of an image that the report may embed in ``section``; ValueError otherwise (27.5, 27.9)."""
        if self._allowed_figures is None:
            raise ValueError(f"no image can be added before ReportBuilder.allow names the report's figures; got {source}")
        if section.number != FIGURE_SECTION:
            raise ValueError(
                f"the report's figures are in Section {FIGURE_SECTION} only (Requirement 27, criterion 9); "
                f"{section.heading!r} cannot hold {source.name!r}"
            )
        key = _path_key(source)
        if key not in self._allowed_figures:
            allowed = ", ".join(self._shown_path(path) for path in self._allowed_figures.values()) or "none"
            raise ValueError(
                f"the report embeds only the figures that ReportBuilder.allow named ({allowed}; Requirement 27, "
                f"criterion 5); {self._shown_path(source)} is not one of them"
            )
        if key in self._figure_keys:
            raise ValueError(f"the report already holds the figure {self._shown_path(source)}")
        return key

    def _where_line(self, symbols: Any) -> tuple[Part, ...]:
        """The parts of an equation's "where" line, from its checked symbol list (Requirement 27, criterion 4)."""
        if isinstance(symbols, Mapping):
            entries = list(symbols.items())
        elif isinstance(symbols, (str, bytes)) or not isinstance(symbols, Iterable):
            raise TypeError(
                "symbols must be a list of (symbol, definition) pairs or a mapping symbol -> definition, "
                f"got {type(symbols).__name__}"
            )
        else:
            entries = list(symbols)
        if not entries:
            raise ValueError(
                'an equation needs a non-empty symbol list: its "where" line defines every symbol of the '
                "equation (Requirement 27, criterion 4)"
            )
        line: list[Part] = ["where "]
        seen: set[str] = set()
        for number, entry in enumerate(entries, start=1):
            if isinstance(entry, (str, bytes)) or not isinstance(entry, Sequence) or len(entry) != 2:
                raise TypeError(f"symbol entry {number} of an equation must be a (symbol, definition) pair, got {entry!r}")
            symbol, definition = entry
            (shown,) = self._resolve((symbol,), f"symbol entry {number} of an equation")
            if isinstance(shown, str):
                shown = shown.strip()
            name = _plain(shown).strip()
            if name in seen:
                raise ValueError(f"the equation lists the symbol {name!r} twice")
            seen.add(name)
            meaning = _trim_end(self._resolve(_as_parts(definition), f"the definition of the symbol {name!r}"))
            found = latex_residue(" ".join((name, *(_plain(part) for part in meaning))))
            if found:
                raise ValueError(
                    f"the symbol {name!r} or its definition holds raw LaTeX {found}; the \"where\" line is text, "
                    "so write symbols as plain characters (α, τ², w_L)"
                )
            if number > 1:
                line.append("; ")
            line.extend((shown, " is ", *meaning))
        if not "".join(_plain(part) for part in line).endswith(_SENTENCE_END):
            line.append(".")
        return tuple(line)

    def _resolve(self, parts: Iterable[Any], what: str) -> tuple[Part, ...]:
        """Checked parts, each Const with its text; UnknownConstantError for an unknown name."""
        resolved: list[Part] = []
        for part in parts:
            if isinstance(part, Const):
                if not isinstance(part.name, str):
                    raise TypeError(f"{what}: a Const name must be a str, got {part.name!r}")
                self._constants.label_text(part.name)  # UnknownConstantError for an unknown name
                text = self._constant_text(part.name) if part.text is None else part.text
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"{what}: Const({part.name!r}) needs non-blank text, got {text!r}")
                resolved.append(Const(part.name, text))
            elif isinstance(part, (Bold, Italic)):
                if not isinstance(part.text, str):
                    raise TypeError(f"{what}: {type(part).__name__} text must be a str, got {part.text!r}")
                resolved.append(part)
            elif isinstance(part, str):
                resolved.append(part)
            else:
                raise TypeError(f"{what}: a part must be a str, Bold, Italic or Const, got {type(part).__name__}")
        if not "".join(_plain(part) for part in resolved).strip():
            raise ValueError(f"{what} has no text")
        return tuple(resolved)

    def _constant_text(self, name: str) -> str:
        try:
            value = self._constants.value(name)
        except UnknownConstantError:  # a Config Method_Constant: its value is in the Config
            if self._settings is None:
                raise ValueError(
                    f"Const({name!r}) needs its text, or the builder needs settings to read the Config value"
                ) from None
            value = self._settings.get(name)
        return format_constant_value(value, name)

    def _shown_path(self, path: Path) -> str:
        """A file as the report shows it: relative to the workspace root with '/', else its name."""
        if self._root is not None:
            try:
                return path.resolve().relative_to(self._root).as_posix()
            except ValueError:
                pass
        return path.name


class _Writer:
    """Writes one build of a ReportBuilder into a new Document, in document order."""

    def __init__(self, builder: ReportBuilder, image_dir: Path) -> None:
        self.builder = builder
        self.image_dir = image_dir
        self.tables: list[RenderedTable] = []
        self.labels: list[tuple[str, str]] = []
        self.equation_files: list[Path] = []
        self.text_width = 0
        self.after_table = False  # the next paragraph starts _GAP_PT below the table just written

    def write(self) -> Document:
        import docx

        builder = self.builder
        builder._constants.reset_labels()
        document = docx.Document()
        _set_up(document, builder._title)
        page = document.sections[0]
        self.text_width = int(page.page_width - page.left_margin - page.right_margin)
        if builder._title is not None:
            document.add_heading(builder._title, level=0)
            if builder._title_line is not None:
                self._parts(self._paragraph(document, NOTE_STYLE), builder._title_line)
        for section in builder._sections:
            self._heading(document, section.heading, 1)  # no page break: the page budget covers the whole PDF
            if section.is_empty:
                self._paragraph(document).add_run(PLACEHOLDER_TEXT)
            for block in section._blocks:
                self._block(document, block)
        return document

    def _block(self, document: Document, block: Any) -> None:
        if isinstance(block, _Heading):
            self._heading(document, block.text, 2)
        elif isinstance(block, _Paragraph):
            self._parts(self._paragraph(document, block.style), block.parts)
        elif isinstance(block, _Table):
            self._table(document, block)
        elif isinstance(block, _Figure):
            self._figure(document, block)
        elif isinstance(block, _Equation):
            self._equation(document, block)
        else:  # pragma: no cover - Section only adds the blocks above
            raise TypeError(f"unknown report block {block!r}")

    def _heading(self, document: Document, text: str, level: int) -> None:
        self.after_table = False  # a heading keeps its own space before
        document.add_heading(text, level=level)

    def _paragraph(self, document: Document, style: str | None = None) -> Any:
        """A new paragraph; the first one after a table starts _GAP_PT below it."""
        from docx.shared import Pt

        paragraph = document.add_paragraph(style=style)
        if self.after_table:
            paragraph.paragraph_format.space_before = Pt(_GAP_PT)
            self.after_table = False
        return paragraph

    def _parts(self, paragraph: Any, parts: Sequence[Part]) -> None:
        for part in parts:
            if isinstance(part, Const):
                paragraph.add_run(part.text)
                label = self.builder._constants.label(part.name)
                if label:
                    paragraph.add_run(f" [{label}]")
                    self.labels.append((part.name, label))
            elif isinstance(part, Bold):
                paragraph.add_run(part.text).bold = True
            elif isinstance(part, Italic):
                paragraph.add_run(part.text).italic = True
            elif part:
                paragraph.add_run(part)

    def _caption(self, document: Document, label: str, parts: Sequence[Part], source: Path, *, before: float, after: float) -> Any:
        """A 9 pt caption: the bold label, the text, and the sentence that names its source file."""
        from docx.shared import Pt

        paragraph = self._paragraph(document, "Caption")
        layout = paragraph.paragraph_format
        layout.space_before, layout.space_after = Pt(before), Pt(after)
        paragraph.add_run(f"{label}.").bold = True
        paragraph.add_run(" ")
        self._parts(paragraph, parts)
        stop = "" if "".join(_plain(part) for part in parts).endswith(_SENTENCE_END) else "."
        paragraph.add_run(f"{stop} Source: {self.builder._shown_path(source)}.")
        return paragraph

    def _table(self, document: Document, block: _Table) -> None:
        caption = self._caption(document, block.label, block.caption, block.rendered.spec.csv, before=_GAP_PT, after=_NEAR_PT)
        caption.paragraph_format.keep_with_next = True  # with the table's first row
        write_docx_table(document, block.rendered, width=self.text_width)
        self.tables.append(block.rendered)
        self.after_table = True

    def _figure(self, document: Document, block: _Figure) -> None:
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.shared import Cm, Pt

        paragraph = self._paragraph(document)
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        layout = paragraph.paragraph_format
        layout.keep_with_next = True  # with its caption below
        layout.space_after = Pt(0)
        width = None if block.width_cm is None else min(int(Cm(block.width_cm)), self.text_width)
        add_picture(paragraph, block.path, alt_text=block.alt_text, max_width=self.text_width, width=width)
        self._caption(document, block.label, block.caption, block.path, before=_NEAR_PT, after=_GAP_PT)

    def _equation(self, document: Document, block: _Equation) -> None:
        from docx.enum.text import WD_TAB_ALIGNMENT
        from docx.shared import Cm, Emu, Pt

        png = render_equation(block.mathtext, self.image_dir / block.file_name)
        self.equation_files.append(png)
        paragraph = self._paragraph(document)
        layout = paragraph.paragraph_format
        layout.keep_with_next = True  # with its "where" line
        layout.space_after = Pt(_NEAR_PT)
        stops = layout.tab_stops
        stops.add_tab_stop(Emu(self.text_width // 2), WD_TAB_ALIGNMENT.CENTER)
        stops.add_tab_stop(Emu(self.text_width), WD_TAB_ALIGNMENT.RIGHT)
        paragraph.add_run("\t")
        add_picture(
            paragraph, png, alt_text=f"{block.label}: {block.description}",
            max_width=self.text_width - 2 * int(Cm(_EQUATION_SIDE_CM)),
        )  # fmt: skip
        paragraph.add_run(f"\t{block.tag}")
        self._parts(self._paragraph(document, NOTE_STYLE), block.where)


# ------------------------------------------------------------------------------- functions

def top_level_headings(document: Document) -> list[str]:
    """The text of every "Heading 1" paragraph of a document, in order (SECTION_HEADINGS in a report)."""
    return [paragraph.text for paragraph in document.paragraphs if paragraph.style is not None and paragraph.style.name == "Heading 1"]


def load_run_constants(
    settings: Settings,
    store: CheckpointStore | None = None,
    *,
    stages: Iterable[str] | None = None,
) -> ConstantRegistry:
    """A ConstantRegistry for ``settings`` holding the data-derived constants saved with stage checkpoints.

    Each stage checkpoint holds its constants as the JSON value ``_constants``
    (``superr.pipeline.CONSTANTS_NAME``), written by the runner after the stage. store: the
    Run's CheckpointStore (default: the store of the workspace that holds ``settings``' config.py).
    stages: the stages to load, in run order; None loads every stage of the store's layout,
    except ``report``, whose checkpoint is complete. Raises StageError for a named stage whose
    checkpoint is missing or changed, and ConstantError when two stages recorded one name with
    different values. Inside a Run use ``ctx.constants``, which already holds them.
    """
    from superr.pipeline import CONSTANTS_NAME, checkpoint_layout
    from superr.store import CheckpointStore

    if store is None:
        root = settings.root
        if root is None:
            raise ValueError("settings were not loaded from config.py, so they name no workspace; pass the store")
        store = CheckpointStore(settings.path("outputs"), checkpoint_layout(root), config_sha256=settings.effective_sha256())
    if stages is None:
        names = [stage for stage in store.stages if stage != STAGE_NAME and store.is_complete(stage)]
    elif isinstance(stages, (str, bytes)):
        raise TypeError("stages must be a collection of stage names, not one string")
    else:
        names = list(stages)
    registry = ConstantRegistry.from_settings(settings)
    for stage in names:
        registry.load_dict(store.load_json(stage, CONSTANTS_NAME))
    return registry


# --------------------------------------------------------------------------------- helpers

def _set_up(document: Document, title: str | None) -> None:
    """A4 page, margins, the text styles of the layout table (DD-18), page-number footer and document properties."""
    from docx.enum.style import WD_STYLE_TYPE
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt

    page = document.sections[0]
    page.page_width, page.page_height = Cm(PAGE_SIZE_CM[0]), Cm(PAGE_SIZE_CM[1])
    margin = Cm(MARGIN_CM)
    page.left_margin = page.right_margin = page.top_margin = page.bottom_margin = margin
    styles = document.styles
    normal = styles["Normal"]
    normal.font.size = Pt(BODY_FONT_PT)
    normal.paragraph_format.space_after = Pt(_BODY_SPACE_AFTER_PT)
    normal.paragraph_format.line_spacing = _BODY_LINE_SPACING
    contextual = f"{qn('w:pPr')}/{qn('w:contextualSpacing')}"
    for style in styles:
        if style.type == WD_STYLE_TYPE.PARAGRAPH and style.name.startswith("List"):
            style.paragraph_format.space_after = Pt(_LIST_SPACE_AFTER_PT)
            for element in style.element.findall(contextual):  # so the 1 pt also separates the items of a list
                element.getparent().remove(element)
    title_style = styles["Title"]
    title_style.font.size = Pt(TITLE_FONT_PT)
    title_style.paragraph_format.space_after = Pt(_TITLE_SPACE_AFTER_PT)
    for level, (size, before, after) in _HEADING_LAYOUT.items():
        heading = styles[f"Heading {level}"]
        heading.font.size = Pt(size)
        layout = heading.paragraph_format
        layout.space_before, layout.space_after = Pt(before), Pt(after)
        layout.keep_with_next = True
    caption = styles["Caption"]
    caption.font.size = Pt(SMALL_FONT_PT)
    caption.font.bold = False  # the label ("Table 1.2.") is bold; the text and the source are not
    note = styles.add_style(NOTE_STYLE, WD_STYLE_TYPE.PARAGRAPH)
    note.base_style = normal
    note.font.size = Pt(SMALL_FONT_PT)
    footer = page.footer.paragraphs[0] if page.footer.paragraphs else page.footer.add_paragraph()
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    run = OxmlElement("w:r")
    number = OxmlElement("w:t")
    number.text = "1"
    run.append(number)
    field.append(run)
    footer._p.append(field)
    properties = document.core_properties
    properties.title = title or ""
    properties.author = ""
    properties.last_modified_by = ""


@functools.lru_cache(maxsize=1)
def _template_paragraph_styles() -> frozenset[str]:
    import docx
    from docx.enum.style import WD_STYLE_TYPE

    return frozenset(style.name for style in docx.Document().styles if style.type == WD_STYLE_TYPE.PARAGRAPH)


def _check_paragraph_style(style: Any) -> None:
    if not isinstance(style, str) or style not in _template_paragraph_styles():
        raise ValueError(f"{style!r} is not a paragraph style of python-docx's default template")
    if style in _RESERVED_STYLES:
        raise ValueError(f"paragraph style {style!r} is reserved: use Section.add_heading for headings")


def _as_parts(caption: Any) -> tuple[Any, ...]:
    if isinstance(caption, (str, Const, Bold, Italic)):
        return (caption,)
    if isinstance(caption, Sequence):
        return tuple(caption)
    raise TypeError(f"a caption must be a str or a sequence of parts, got {type(caption).__name__}")


def _plain(part: Part) -> str:
    return part if isinstance(part, str) else (part.text or "")


def _trim_end(parts: tuple[Part, ...]) -> tuple[Part, ...]:
    """The parts without white space at their end, so that a full stop or "; " can follow them."""
    items = list(parts)
    while items and isinstance(items[-1], (str, Bold, Italic)):
        last = items.pop()
        text = (last if isinstance(last, str) else last.text).rstrip()
        if text:
            items.append(text if isinstance(last, str) else type(last)(text))
            break
    return tuple(items)


def _path_key(path: Path) -> str:
    """A file's identity for the figure check: its absolute path, in the letter case the OS compares."""
    return os.path.normcase(str(Path(path).resolve()))


def _shown_float(value: float) -> str:
    if not math.isfinite(value):
        return str(value)
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    if text in ("0", "-0"):
        return "0" if value == 0 else f"{value:.3g}"
    return text
