"""Report tables: one CSV under ``outputs/`` shown as one DOCX table with fixed decimals.

Design section "Report_Builder" and decision DD-12: every report table is built from one CSV
through a ``TableSpec``, so the Report_Check (Requirement 27, criteria 14 and 19) can read the
exported DOCX back and compare every cell with the CSV it came from.

Rendering rule (``render_table``)
---------------------------------
The CSV is read as text with the ``csv`` module (UTF-8; a byte-order mark is skipped). No value
passes through a float, so the same CSV bytes and spec always give the same strings.

1. ``where`` keeps the rows whose CSV text in each named column is one of the given strings, and
   ``head`` then keeps the first N of those rows. Rows keep their CSV order.
2. A column that ``decimals`` names is a numeric column; every other shown column is a text column.

   * Numeric cell: the exact decimal value of the CSV text, rounded half away from zero to the
     column's number of decimals, in fixed-point notation: ``"0.309"``, ``"90"``, ``"-0.472"``.
     A zero result has no sign (``"0.000"``, never ``"-0.000"``). A blank, ``nan`` or ``NaN``
     cell is shown empty. Any other text that is not a finite number raises TableError.
   * Text cell: the CSV text exactly, so a blank cell is shown empty.

   The rounded value therefore never differs from the CSV value by more than 0.5 × 10^(−d).
3. The key cells, as shown, must be unique among the shown rows (TableError otherwise).

DOCX tables (``write_docx_table``)
----------------------------------
A header row (the display headers: bold, shaded, repeated at the top of each page, ``w:tblHeader``)
and one row per shown row, with numeric columns right-aligned. The compact layout of DD-18: text
at TABLE_FONT_PT (8.5 pt) with no space around the cell paragraphs, cell margins of
CELL_MARGIN_TOP_BOTTOM_CM at the top and bottom and CELL_MARGIN_SIDE_CM at the left and right
(``w:tblCellMar``), and rows that never split across pages (``w:cantSplit``). The ``table_id`` is
written as the table's alt-text title (``w:tblCaption``), which Word and LibreOffice keep but
never print, so it is not Report_Text; ``docx_table_id`` and ``find_docx_table`` read it back.

For the Report_Check
--------------------
For each spec of the report (``BuiltReport.specs`` in ``superr.report.builder``):

1. Re-open the exported DOCX with python-docx and take ``table = find_docx_table(document,
   spec.table_id)``. None means the table is missing from the report.
2. ``cells = docx_table_text(table)``. Row 0 is the header and must equal ``spec.header_texts``.
3. ``expected = render_table(spec)`` reads the CSV again. Rows are matched by key, not by
   position: the DOCX row whose cells at ``spec.key_positions`` equal ``expected.keys[i]`` belongs
   to CSV data row ``expected.source_rows[i]``. A DOCX key that is not in ``expected.keys``, an
   expected key with no DOCX row and a key shown twice are failures, each named by its key.
4. In each matched row, compare every column ``j`` with ``cell_matches(spec, spec.columns[j],
   shown, expected.raw[i][j])``. A numeric cell must lie within 0.5 × 10^(−d) of the CSV number,
   where d is the number of decimals shown in that cell (``shown_decimals``); a text cell must
   equal the CSV text; a blank cell matches only a blank or NaN CSV cell. Report each failure
   with the table ID, the row key and the column name.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Context, Decimal, InvalidOperation, localcontext
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from superr.errors import StageError

if TYPE_CHECKING:
    from docx.document import Document
    from docx.table import Table

__all__ = [
    "CELL_MARGIN_SIDE_CM",
    "CELL_MARGIN_TOP_BOTTOM_CM",
    "HEADER_FILL",
    "MAX_DECIMALS",
    "TABLE_FONT_PT",
    "TABLE_STYLE",
    "RenderedTable",
    "TableError",
    "TableSpec",
    "cell_matches",
    "docx_table_id",
    "docx_table_text",
    "find_docx_table",
    "format_number",
    "number_matches",
    "read_csv_text",
    "render_table",
    "shown_decimals",
    "write_docx_table",
]

#: Largest number of decimals a numeric column may show.
MAX_DECIMALS = 9

#: Table style of every report table (defined in python-docx's default template), font size of
#: its text in points, and the fill colour of its header row.
TABLE_STYLE = "Table Grid"
TABLE_FONT_PT = 8.5
HEADER_FILL = "D9D9D9"
#: Cell margins of every report table in cm (``w:tblCellMar``; design section "Report_Builder",
#: layout table): none at the top and bottom, a little at the left and right.
CELL_MARGIN_TOP_BOTTOM_CM = 0.0
CELL_MARGIN_SIDE_CM = 0.08

_TABLE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_MISSING_NUMBERS = frozenset({"", "nan"})  # numeric CSV text shown empty, after strip() and lower()
_MIN_WEIGHT, _MAX_WEIGHT = 3, 40  # column-width weights, in characters
_CM_PER_INCH, _TWIPS_PER_INCH = 2.54, 1440
#: The row properties of CT_TrPr in schema order (``_set_row_flag``).
_ROW_PROPERTY_ORDER = (
    "w:cnfStyle", "w:divId", "w:gridBefore", "w:gridAfter", "w:wBefore", "w:wAfter", "w:cantSplit",
    "w:trHeight", "w:tblHeader", "w:tblCellSpacing", "w:jc", "w:hidden", "w:ins", "w:del", "w:trPrChange",
)  # fmt: skip


class TableError(StageError):
    """A TableSpec cannot be rendered from its CSV (exit code 5).

    Raised for a missing or unreadable CSV, a column the CSV lacks, a cell of a numeric column that
    is not a number, a spec that keeps no row, and a key shown on more than one row.
    """


# ------------------------------------------------------------------------------------ spec

@dataclass(frozen=True)
class TableSpec:
    """One report table built from one CSV under ``outputs/`` (DD-12).

    table_id: unique in the report; letters, digits, ``_``, ``.`` and ``-`` (at most 64), stored as
    the DOCX table's alt-text title. csv: absolute path of the CSV, normally under
    ``settings.path("outputs")``. columns: the CSV columns shown, in order. decimals: numeric
    column -> fixed number of decimals (0 to MAX_DECIMALS); every other shown column is shown as
    its CSV text. key: shown columns whose shown cells identify a row, for the Report_Check.
    caption: plain-text caption that the builder prints above the table. headers: column ->
    display header (default: the column name). where: CSV column -> allowed CSV text (one str or
    several); the column need not be shown. head: keep only the first N rows left by ``where``.

    Lists are stored as tuples and mappings as read-only copies; ``hash`` ignores the mappings.
    """

    table_id: str
    csv: Path
    columns: tuple[str, ...]
    decimals: Mapping[str, int] = field(hash=False)
    key: tuple[str, ...]
    caption: str = ""
    headers: Mapping[str, str] = field(default_factory=dict, hash=False)
    where: Mapping[str, tuple[str, ...]] = field(default_factory=dict, hash=False)
    head: int | None = None

    def __post_init__(self) -> None:
        table_id = self.table_id
        if not isinstance(table_id, str) or not _TABLE_ID.fullmatch(table_id):
            raise ValueError(
                f"TableSpec table_id {table_id!r} must be 1-64 letters, digits, '_', '.' or '-', "
                "starting with a letter or digit"
            )
        where = f"TableSpec {table_id!r}"
        path = Path(self.csv) if isinstance(self.csv, str) else self.csv
        if not isinstance(path, Path):
            raise TypeError(f"{where}: csv must be a path, got {type(self.csv).__name__}")
        if not path.is_absolute():
            raise ValueError(f"{where}: csv must be an absolute path, such as settings.path('outputs') / ..., got {str(path)!r}")
        columns = _names(self.columns, f"{where}: columns")
        key = _names(self.key, f"{where}: key")
        hidden = [name for name in key if name not in columns]
        if hidden:
            raise ValueError(f"{where}: key column(s) {hidden} are not shown; every key column must be in columns")
        decimals = _decimals(self.decimals, columns, where)
        headers = _headers(self.headers, columns, where)
        filters = _filters(self.where, where)
        head = self.head
        if head is not None and (type(head) is not int or head < 1):
            raise ValueError(f"{where}: head must be None or an int of at least 1, got {head!r}")
        if not isinstance(self.caption, str):
            raise TypeError(f"{where}: caption must be a str, got {type(self.caption).__name__}")
        setter = object.__setattr__
        setter(self, "csv", path)
        setter(self, "columns", columns)
        setter(self, "key", key)
        setter(self, "decimals", MappingProxyType(decimals))
        setter(self, "headers", MappingProxyType(headers))
        setter(self, "where", MappingProxyType(filters))

    @property
    def header_texts(self) -> tuple[str, ...]:
        """The display header of each shown column, in column order."""
        return tuple(self.headers.get(name, name) for name in self.columns)

    @property
    def key_positions(self) -> tuple[int, ...]:
        """Position in ``columns`` of each key column, in key order."""
        return tuple(self.columns.index(name) for name in self.key)

    def is_numeric(self, column: str) -> bool:
        """True when ``column`` is shown as a number with fixed decimals."""
        return column in self.decimals


@dataclass(frozen=True)
class RenderedTable:
    """The shown text of a TableSpec's cells, as written into the DOCX.

    rows[i][j] is the shown text of shown row i and column ``spec.columns[j]``; raw[i][j] is the
    CSV text it came from; source_rows[i] is the 1-based CSV data-row number (header excluded).
    """

    spec: TableSpec
    rows: tuple[tuple[str, ...], ...]
    raw: tuple[tuple[str, ...], ...]
    source_rows: tuple[int, ...]

    @property
    def header(self) -> tuple[str, ...]:
        """The header row: ``spec.header_texts``."""
        return self.spec.header_texts

    @property
    def keys(self) -> tuple[tuple[str, ...], ...]:
        """The shown key cells of each row, in row order; unique by construction."""
        positions = self.spec.key_positions
        return tuple(tuple(row[p] for p in positions) for row in self.rows)


# ------------------------------------------------------------------------------- rendering

def read_csv_text(path: Path | str) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    """(header, data rows) of a CSV file as text. TableError when it is missing, unreadable or ragged."""
    source = Path(path)
    try:
        with open(source, encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            rows = [tuple(row) for row in reader if row]  # csv yields [] only for an empty line
    except FileNotFoundError:
        raise TableError(f"the report table CSV {source} does not exist") from None
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise TableError(f"cannot read the report table CSV {source}: {type(exc).__name__}: {exc}") from None
    if header is None:
        raise TableError(f"the report table CSV {source} is empty: it has no header row")
    repeated = sorted({name for name in header if header.count(name) > 1})
    if repeated:
        raise TableError(f"the report table CSV {source} repeats the column name(s) {repeated}")
    ragged = [number for number, row in enumerate(rows, start=1) if len(row) != len(header)]
    if ragged:
        raise TableError(
            f"the report table CSV {source} has {len(ragged)} data row(s) whose field count differs "
            f"from the header's {len(header)}, first at data row {ragged[0]}"
        )
    return tuple(header), tuple(rows)


def render_table(spec: TableSpec) -> RenderedTable:
    """Read ``spec.csv`` and render the spec's rows by the module's rendering rule."""
    header, records = read_csv_text(spec.csv)
    index = {name: position for position, name in enumerate(header)}
    missing = [name for name in (*spec.columns, *spec.where) if name not in index]
    if missing:
        raise TableError(
            f"table {spec.table_id!r}: {spec.csv} has no column(s) {sorted(set(missing))}; it has {list(header)}"
        )
    kept: list[tuple[int, tuple[str, ...]]] = []
    for number, record in enumerate(records, start=1):
        if all(record[index[name]] in allowed for name, allowed in spec.where.items()):
            kept.append((number, record))
    if spec.head is not None:
        kept = kept[: spec.head]
    if not kept:
        rule = f" with {dict(spec.where)}" if spec.where else ""
        raise TableError(f"table {spec.table_id!r}: {spec.csv} has no data row{rule} to show")
    positions = [index[name] for name in spec.columns]
    raw = tuple(tuple(record[p] for p in positions) for _, record in kept)
    rows = []
    for (number, _), cells in zip(kept, raw):
        shown = []
        for name, text in zip(spec.columns, cells):
            if name in spec.decimals:
                try:
                    text = format_number(text, spec.decimals[name])
                except ValueError as exc:
                    raise TableError(
                        f"table {spec.table_id!r}: {spec.csv} data row {number}, numeric column {name!r}: {exc}"
                    ) from None
            shown.append(text)
        rows.append(tuple(shown))
    rendered = RenderedTable(spec, tuple(rows), raw, tuple(number for number, _ in kept))
    seen: dict[tuple[str, ...], int] = {}
    for number, key in zip(rendered.source_rows, rendered.keys):
        if key in seen:
            raise TableError(
                f"table {spec.table_id!r}: key {dict(zip(spec.key, key))} is shown on {spec.csv} data rows "
                f"{seen[key]} and {number}; the key must identify one row"
            )
        seen[key] = number
    return rendered


def format_number(text: str, decimals: int) -> str:
    """CSV number text rounded half away from zero to ``decimals`` places, in fixed-point notation.

    ``""`` for a blank, ``nan`` or ``NaN`` cell. ValueError for text that is not a finite number.
    """
    if type(decimals) is not int or not 0 <= decimals <= MAX_DECIMALS:
        raise ValueError(f"decimals must be an int from 0 to {MAX_DECIMALS}, got {decimals!r}")
    stripped = text.strip()
    if stripped.lower() in _MISSING_NUMBERS:
        return ""
    value = _decimal(stripped)
    if value is None:
        raise ValueError(f"{text!r} is not a finite number")
    digits = max(value.adjusted(), 0) + decimals + 2
    rounded = value.quantize(
        Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP, context=Context(prec=max(28, digits))
    )
    if rounded.is_zero():
        rounded = rounded.copy_abs()
    return format(rounded, "f")


def shown_decimals(text: str) -> int:
    """The number of decimal places shown in a number's text (0 for ``"90"``, 3 for ``"0.310"``).

    ValueError when the text is not a finite number.
    """
    value = _decimal(text.strip())
    if value is None:
        raise ValueError(f"{text!r} is not a finite number")
    exponent = value.as_tuple().exponent
    return max(0, -exponent) if isinstance(exponent, int) else 0


def number_matches(shown: str, csv_text: str) -> bool:
    """True when a shown number lies within 0.5 × 10^(−d) of the CSV number (Requirement 27, criterion 19).

    d is the number of decimals shown. The difference is computed exactly on the decimal texts.
    A blank shown cell matches only a blank or NaN CSV cell, and the other way round; text that
    is not a finite number never matches.
    """
    shown_text, source_text = shown.strip(), csv_text.strip()
    shown_blank = shown_text == ""
    source_blank = source_text.lower() in _MISSING_NUMBERS
    if shown_blank or source_blank:
        return shown_blank and source_blank
    shown_value, source_value = _decimal(shown_text), _decimal(source_text)
    if shown_value is None or source_value is None:
        return False
    d = shown_decimals(shown_text)
    with localcontext(Context(prec=max(shown_value.adjusted(), source_value.adjusted(), 0) + 80)):
        return abs(shown_value - source_value) <= Decimal(5).scaleb(-(d + 1))


def cell_matches(spec: TableSpec, column: str, shown: str, csv_text: str) -> bool:
    """True when a shown cell of ``column`` agrees with its CSV text: ``number_matches`` for a
    numeric column, exact equality for a text column."""
    if column not in spec.columns:
        raise ValueError(f"table {spec.table_id!r} shows no column {column!r}")
    if spec.is_numeric(column):
        return number_matches(shown, csv_text)
    return shown == csv_text


# ------------------------------------------------------------------------------------ DOCX

def write_docx_table(
    parent: Any,
    rendered: RenderedTable,
    *,
    width: int | None = None,
    font_size_pt: float = TABLE_FONT_PT,
) -> Table:
    """Append the rendered table to ``parent`` (a python-docx Document) and return the DOCX table.

    width: total table width in EMU, normally the text width; the columns then get widths in
    proportion to their longest cell or header word. None leaves python-docx's equal widths.
    The layout is the module docstring's: the cell margins, rows that never split across pages,
    and a bold, shaded header row that Word repeats at the top of each page.
    """
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Emu, Pt

    spec = rendered.spec
    table = parent.add_table(rows=1 + len(rendered.rows), cols=len(spec.columns))
    table.style = TABLE_STYLE
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    _set_table_id(table, spec.table_id)
    widths = None if width is None else [Emu(value) for value in _column_widths(rendered, int(width))]
    if widths is not None:
        table.autofit = False
        for column, value in zip(table.columns, widths):
            column.width = value
    _set_cell_margins(table, CELL_MARGIN_TOP_BOTTOM_CM, CELL_MARGIN_SIDE_CM)
    numeric = [spec.is_numeric(name) for name in spec.columns]
    size = Pt(font_size_pt)
    zero = Pt(0)
    for position, row in enumerate(table.rows):
        texts = rendered.header if position == 0 else rendered.rows[position - 1]
        _set_row_flag(row, "w:cantSplit")  # the row never splits across pages
        if position == 0:
            _set_row_flag(row, "w:tblHeader")  # repeated at the top of each page
        for j, (cell, text) in enumerate(zip(row.cells, texts)):
            if widths is not None:
                cell.width = widths[j]
            paragraph = cell.paragraphs[0]
            layout = paragraph.paragraph_format
            layout.space_before = zero
            layout.space_after = zero
            layout.line_spacing = 1.0
            if numeric[j]:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            run = paragraph.add_run(text)
            run.font.size = size
            if position == 0:
                run.bold = True
                _shade(cell, HEADER_FILL)
    return table


def docx_table_id(table: Table) -> str | None:
    """The ``table_id`` that write_docx_table stored in a DOCX table, or None for another table."""
    from docx.oxml.ns import qn

    element = table._tbl.tblPr.find(qn("w:tblCaption"))
    return None if element is None else element.get(qn("w:val"))


def find_docx_table(document: Document, table_id: str) -> Table | None:
    """The body table of ``document`` written for ``table_id``, or None.

    TableError when two tables carry the same ID.
    """
    found = [table for table in document.tables if docx_table_id(table) == table_id]
    if len(found) > 1:
        raise TableError(f"the report holds {len(found)} tables with the table_id {table_id!r}")
    return found[0] if found else None


def docx_table_text(table: Table) -> list[list[str]]:
    """The text of every cell of a DOCX table, row by row (row 0 is the header row)."""
    return [[cell.text for cell in row.cells] for row in table.rows]


# --------------------------------------------------------------------------------- helpers

def _names(value: Any, what: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{what} must be a list or tuple of column names, got {value!r}")
    names = tuple(value)
    if not names:
        raise ValueError(f"{what} must name at least one column")
    bad = [name for name in names if not isinstance(name, str) or not name]
    if bad:
        raise ValueError(f"{what} holds {bad!r}, which are not column names")
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"{what} names {repeated} more than once")
    return names


def _decimals(value: Any, columns: tuple[str, ...], where: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{where}: decimals must map numeric columns to decimals, got {type(value).__name__}")
    unknown = [name for name in value if name not in columns]
    if unknown:
        raise ValueError(f"{where}: decimals names column(s) {unknown!r} that are not shown")
    bad = {name: d for name, d in value.items() if type(d) is not int or not 0 <= d <= MAX_DECIMALS}
    if bad:
        raise ValueError(f"{where}: decimals must be ints from 0 to {MAX_DECIMALS}; got {bad!r}")
    return {name: value[name] for name in columns if name in value}


def _headers(value: Any, columns: tuple[str, ...], where: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{where}: headers must map columns to display headers, got {type(value).__name__}")
    unknown = [name for name in value if name not in columns]
    if unknown:
        raise ValueError(f"{where}: headers names column(s) {unknown!r} that are not shown")
    bad = {name: text for name, text in value.items() if not isinstance(text, str) or not text.strip()}
    if bad:
        raise ValueError(f"{where}: each display header must be a non-blank str; got {bad!r}")
    return {name: value[name] for name in columns if name in value}


def _filters(value: Any, where: str) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{where}: where must map CSV columns to allowed text, got {type(value).__name__}")
    filters: dict[str, tuple[str, ...]] = {}
    for name, allowed in value.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{where}: where holds {name!r}, which is not a column name")
        texts = (allowed,) if isinstance(allowed, str) else allowed
        if isinstance(texts, bytes) or not isinstance(texts, Sequence) or not texts:
            raise ValueError(f"{where}: where[{name!r}] must be a str or a non-empty list of str, got {allowed!r}")
        if not all(isinstance(text, str) for text in texts):
            raise ValueError(f"{where}: where[{name!r}] must hold only str values, got {allowed!r}")
        filters[name] = tuple(texts)
    return filters


def _decimal(text: str) -> Decimal | None:
    """The finite Decimal of a number's text, or None."""
    try:
        value = Decimal(text)
    except (InvalidOperation, ValueError):
        return None
    return value if value.is_finite() else None


def _column_widths(rendered: RenderedTable, total: int) -> list[int]:
    """Column widths in EMU summing to at most ``total``, in proportion to each column's content."""
    weights = []
    for j, header in enumerate(rendered.header):
        longest_cell = max((len(row[j]) for row in rendered.rows), default=0)
        longest_word = max((len(word) for word in header.split()), default=0)
        weights.append(min(max(longest_cell, longest_word, _MIN_WEIGHT), _MAX_WEIGHT))
    scale = total / sum(weights)
    return [int(weight * scale) for weight in weights]


def _set_table_id(table: Table, table_id: str) -> None:
    """Write ``table_id`` as the table's alt-text title, ``w:tblCaption``.

    CT_TblPr ends with tblLook, tblCaption, tblDescription, tblPrChange; python-docx inserts its
    own tblPr children before tblCaption, so appending keeps the schema order.
    """
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    properties = table._tbl.tblPr
    for old in properties.findall(qn("w:tblCaption")):
        properties.remove(old)
    element = OxmlElement("w:tblCaption")
    element.set(qn("w:val"), table_id)
    successor = next(
        (child for child in properties if child.tag in (qn("w:tblDescription"), qn("w:tblPrChange"))), None
    )
    if successor is None:
        properties.append(element)
    else:
        successor.addprevious(element)


def _set_cell_margins(table: Table, top_bottom_cm: float, side_cm: float) -> None:
    """Give every cell of the table the same margins: ``w:tblCellMar`` in twips, in CT_TblPr schema order.

    CT_TblPr ends with tblLayout, tblCellMar, tblLook, tblCaption, tblDescription, tblPrChange, so
    the element goes before the first of the last four that is present.
    """
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    properties = table._tbl.tblPr
    for old in properties.findall(qn("w:tblCellMar")):
        properties.remove(old)
    margins = OxmlElement("w:tblCellMar")
    for side, cm in (("top", top_bottom_cm), ("left", side_cm), ("bottom", top_bottom_cm), ("right", side_cm)):
        element = OxmlElement(f"w:{side}")
        element.set(qn("w:w"), str(round(cm / _CM_PER_INCH * _TWIPS_PER_INCH)))
        element.set(qn("w:type"), "dxa")
        margins.append(element)
    later = {qn(f"w:{tag}") for tag in ("tblLook", "tblCaption", "tblDescription", "tblPrChange")}
    successor = next((child for child in properties if child.tag in later), None)
    if successor is None:
        properties.append(margins)
    else:
        successor.addprevious(margins)


def _set_row_flag(row: Any, tag: str) -> None:
    """Switch on a row property that has no value, such as ``w:cantSplit`` or ``w:tblHeader``.

    The element goes in the CT_TrPr order of _ROW_PROPERTY_ORDER: before the first present
    property that comes later in that order.
    """
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    properties = row._tr.get_or_add_trPr()
    if properties.find(qn(tag)) is not None:
        return
    later = {qn(name) for name in _ROW_PROPERTY_ORDER[_ROW_PROPERTY_ORDER.index(tag) + 1 :]}
    element = OxmlElement(tag)
    successor = next((child for child in properties if child.tag in later), None)
    if successor is None:
        properties.append(element)
    else:
        successor.addprevious(element)


def _shade(cell: Any, fill: str) -> None:
    """Give a table cell a solid background colour (``w:shd``), placed in CT_TcPr schema order."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    properties = cell._tc.get_or_add_tcPr()
    for old in properties.findall(qn("w:shd")):
        properties.remove(old)
    shading = OxmlElement("w:shd")
    shading.set(qn("w:val"), "clear")
    shading.set(qn("w:color"), "auto")
    shading.set(qn("w:fill"), fill)
    before = {qn(f"w:{tag}") for tag in ("cnfStyle", "tcW", "gridSpan", "hMerge", "vMerge", "tcBorders")}
    position = 0
    for index, child in enumerate(properties):
        if child.tag in before:
            position = index + 1
    properties.insert(position, shading)
