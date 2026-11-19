"""Report_Check: the automated checks on the exported methodology report.

Requirement 2, criteria 9 and 10; Requirement 27, criteria 1, 9, 13, 14 and 17; design section
"Report_Builder" and decision DD-12. ``run_report_check`` re-opens the exported DOCX with
python-docx, runs every check below and returns all failures together, with the counts of what
it compared; ``report_check`` returns the failures only (the design's signature). No check stops
at its first failure, and a check that cannot run (a CSV that cannot be read, a missing table)
adds a failure and lets every other check run (27.17). The ``report`` stage
(``superr.report.export``) writes each failure to the Run_Log, keeps the exported files and then
fails with ReportCheckError (exit code 6).

The Report_Text (``report_text``)
---------------------------------
Every paragraph of the body in document order (the title, headings and captions included);
every cell of every table, nested tables too; the paragraphs of each header and footer that the
DOCX defines; and the paragraphs of its footnotes and endnotes, if it has any. The text of a
paragraph is the text of its ``w:t`` elements (tabs and breaks of its runs as ``\\t`` and
``\\n``), so text inside hyperlinks, fields and content controls counts and deleted text and
field codes do not. Picture alt text and table IDs (``w:tblCaption``) are never printed, so they
are not Report_Text; text drawn inside images is excluded by the Glossary.

Each piece of text carries its heading path: the "Heading 1" text above it, then the "Heading 2"
and "Heading 3" texts under it (a heading's own path ends with itself). Failures name it, so a
banned sentence comes with the heading of its section (2.9).

Checks (CHECKS)
---------------
``banned_phrase`` (2.9)
    BANNED_PHRASES, ignoring letter case: ``100[\\s-]?%[\\s-]?data[\\s-]?calibrated`` and
    ``zero[\\s-]?constants``, so a space, a hyphen or nothing between the words all match. Each
    sentence that holds a match is one failure, with its heading path.
``citation`` (2.10)
    Every URL (``https?://\\S+``), every DOI (``10\\.\\d{4,9}/\\S+``) that is not part of a URL,
    and every non-blank paragraph under a "References" heading (REFERENCES_HEADING, with or
    without a section number such as "A.4") must equal an entry of the Verified_Source_List,
    Config ``report.verified_sources``. Entries and reference paragraphs are compared after
    stripping white space at both ends. A URL or DOI also matches an entry when it equals it once
    trailing punctuation is removed (``see https://x.org.`` cites ``https://x.org``), and the URLs
    and DOIs inside a verified reference entry are covered by that entry. Each unmatched
    citation is one failure.
``latex_residue`` (27.9)
    ``\\\\[A-Za-z]`` and ``\\$[^$]+\\$`` (``superr.report.equations.LATEX_RESIDUE``): one failure
    for each paragraph or cell that holds raw LaTeX.
``table`` (27.13)
    For every TableSpec, the recipe of ``superr.report.tables``: the DOCX table with the spec's
    table_id must exist once; its header row must equal ``spec.header_texts``; its data rows are
    matched to ``render_table(spec)`` by key (a missing, extra or repeated key is a failure); and
    each cell of a matched row must pass ``cell_matches`` against the CSV text, so a number lies
    within 0.5 × 10^(−d) of the CSV value, d being the decimals shown. Each failure names the
    table ID, the row key and the column. A DOCX table with no table_id, or with one that no spec
    has, cannot be compared and is a failure too (DD-12).
``section0`` (27.14)
    Each Section 0 table (SECTION0_TABLES) against its Role's Main_List file,
    ``outputs/final/top10_batters.csv`` or ``outputs/final/top10_bowlers.csv``: data row i of the
    table must show the ``rank`` and ``player_name`` of data row i of the CSV, read as text (the
    columns are found through the spec's ``columns``; ranks are compared as numbers). Each rank
    that differs is one failure.
``headings`` (27.1)
    The "Heading 1" paragraphs must be SECTION_HEADINGS, in that order.
``placeholder``
    No text may hold a PLACEHOLDER_MARKERS text (PLACEHOLDER_TEXT, the paragraph of an empty
    section), except under a heading of ALLOWED_PLACEHOLDERS. ALLOWED_PLACEHOLDERS is empty now
    that task 27.1 has filled Section 5.3, so every marker is a failure; the result still lists
    each allowed marker it finds, for a future task that needs one.

A DOCX that cannot be opened, or whose text cannot be read, gives a failure of the kind ``docx``.

Use::

    result = run_report_check(built.path, built.specs, settings)
    for failure in result.failures:
        log.error(f"Report_Check failure: {failure}")
    log.info(result.summary())
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from superr.report.builder import PLACEHOLDER_TEXT, SECTION_HEADINGS, top_level_headings
from superr.report.equations import latex_residue
from superr.report.sections import OUTPUT_FILES, ROLES, SECTION0_TABLES
from superr.report.tables import (
    TableError,
    TableSpec,
    cell_matches,
    docx_table_id,
    docx_table_text,
    find_docx_table,
    read_csv_text,
    render_table,
)

if TYPE_CHECKING:
    from docx.document import Document
    from docx.table import Table

    from superr.settings import Settings

__all__ = [
    "ALLOWED_PLACEHOLDERS",
    "BANNED_PHRASES",
    "CHECKS",
    "DOI_PATTERN",
    "PLACEHOLDER_MARKERS",
    "REFERENCES_HEADING",
    "URL_PATTERN",
    "CheckFailure",
    "ReportCheckResult",
    "TextUnit",
    "banned_phrase_sentences",
    "citations",
    "is_references_heading",
    "is_verified",
    "report_check",
    "report_text",
    "run_report_check",
]

#: The checks of the Report_Check, in the order they run (module docstring).
CHECKS: tuple[str, ...] = ("headings", "placeholder", "banned_phrase", "citation", "latex_residue", "table", "section0")

#: Phrases the Report_Text must not hold (Requirement 2, criterion 9; design section "Report_Builder").
BANNED_PHRASES: tuple[re.Pattern[str], ...] = (
    re.compile(r"100[\s-]?%[\s-]?data[\s-]?calibrated", re.IGNORECASE),
    re.compile(r"zero[\s-]?constants", re.IGNORECASE),
)
#: Citations that must match the Verified_Source_List (Requirement 2, criterion 10).
URL_PATTERN = re.compile(r"https?://\S+")
DOI_PATTERN = re.compile(r"10\.\d{4,9}/\S+")
#: The heading whose paragraphs are reference entries, optionally after a section number ("A.4 References").
REFERENCES_HEADING = "References"

#: Text that marks an unfinished part of the report, compared ignoring letter case.
PLACEHOLDER_MARKERS: tuple[str, ...] = (PLACEHOLDER_TEXT,)
#: Sub-headings under which a placeholder marker is allowed for now, with the reason. Empty since
#: task 27.1 filled Section 5.3, so every placeholder marker is now a failure; the result still
#: lists each allowed marker it finds, for a future task that needs one.
ALLOWED_PLACEHOLDERS: Mapping[str, str] = MappingProxyType({})

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_P, _T, _R = f"{_W}p", f"{_W}t", f"{_W}r"
_BREAKS = {f"{_W}tab": "\t", f"{_W}br": "\n", f"{_W}cr": "\n"}
_NOTE_PARTS = {"/word/footnotes.xml": "footnote", "/word/endnotes.xml": "endnote"}
_SEPARATOR_NOTES = frozenset({"separator", "continuationSeparator", "continuationNotice"})
_HEADING_STYLE = re.compile(r"Heading ([1-9])")
_REFERENCES = re.compile(rf"(?:[A-Za-z]?\d*(?:\.\d+)*\.?\s+)?{REFERENCES_HEADING}:?", re.IGNORECASE)
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")
_TRAILING = ".,;:!?)]}>'\"\u2019\u201d"
_EXCERPT = 160


# ------------------------------------------------------------------------------- results

@dataclass(frozen=True)
class CheckFailure:
    """One problem that the Report_Check found: the check (CHECKS), where it is and what it is."""

    check: str
    location: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.check}] {self.location}: {self.detail}"

    def to_dict(self) -> dict[str, str]:
        return {"check": self.check, "location": self.location, "detail": self.detail}


@dataclass(frozen=True)
class ReportCheckResult:
    """Every failure of one Report_Check, and the counts of what it scanned and compared."""

    docx: Path
    failures: tuple[CheckFailure, ...]
    text_units: int  # paragraphs, cells and header, footer and note paragraphs scanned
    verified_sources: int  # entries in the Verified_Source_List
    tables_compared: int  # TableSpecs whose DOCX table was found and compared row by row
    rows_compared: int
    cells_compared: int
    ranks_compared: int  # Section 0 positions compared with the Main_List files
    allowed_placeholders: tuple[str, ...] = ()  # heading path and reason of each allowed marker found
    checks_run: tuple[str, ...] = field(default=CHECKS)

    @property
    def passed(self) -> bool:
        return not self.failures

    def counts(self) -> dict[str, int]:
        """Failures per check, for every check in CHECKS (0 when it found none)."""
        found = Counter(failure.check for failure in self.failures)
        return {check: found.get(check, 0) for check in (*CHECKS, *sorted(set(found) - set(CHECKS)))}

    def summary(self) -> str:
        """One Run_Log line: what was checked and the failures per check."""
        compared = (
            f"{self.tables_compared:,} tables compared with their CSV files ({self.rows_compared:,} rows, "
            f"{self.cells_compared:,} cells); {self.ranks_compared:,} Section 0 ranks compared with "
            f"{' and '.join(Path(OUTPUT_FILES[f'top10_{role}']).name for role in ROLES)}; "
            f"{self.text_units:,} paragraphs and cells scanned for banned phrases, LaTeX residue and citations "
            f"(Verified_Source_List: {self.verified_sources:,} entries); top-level headings and placeholders checked"
        )
        allowed = (
            f". Allowed placeholder marker{'s' if len(self.allowed_placeholders) > 1 else ''}: "
            + "; ".join(self.allowed_placeholders)
            if self.allowed_placeholders
            else ""
        )
        if self.passed:
            return f"Report_Check passed with 0 failures: {compared}{allowed}"
        counts = ", ".join(f"{check} {count:,}" for check, count in self.counts().items() if count)
        return f"Report_Check failed with {len(self.failures):,} failure(s) ({counts}): {compared}{allowed}"

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready record for the stage checkpoint."""
        return {
            "passed": self.passed,
            "n_failures": len(self.failures),
            "failures_by_check": self.counts(),
            "failures": [failure.to_dict() for failure in self.failures],
            "text_units": self.text_units,
            "verified_sources": self.verified_sources,
            "tables_compared": self.tables_compared,
            "rows_compared": self.rows_compared,
            "cells_compared": self.cells_compared,
            "ranks_compared": self.ranks_compared,
            "allowed_placeholders": list(self.allowed_placeholders),
            "checks_run": list(self.checks_run),
        }


# ---------------------------------------------------------------------------- Report_Text

@dataclass(frozen=True)
class TextUnit:
    """One piece of Report_Text.

    kind: "title", "heading", "paragraph", "cell", "header", "footer", "footnote" or "endnote".
    headings: the heading path above it (module docstring). where: a readable location.
    level: the heading level of a "heading" unit (1-9).
    """

    kind: str
    text: str
    headings: tuple[str, ...]
    where: str
    level: int | None = None

    def under(self, heading: str) -> bool:
        """True when ``heading`` is on the unit's heading path (a heading is under itself)."""
        return heading in self.headings


def report_text(document: Document) -> list[TextUnit]:
    """Every piece of Report_Text of a python-docx Document, in document order (module docstring)."""
    return _Reader(document).read()


class _Reader:
    def __init__(self, document: Document) -> None:
        self.document = document
        self.units: list[TextUnit] = []
        self.path: tuple[str, ...] = ()
        self.paragraphs = 0
        self.table_paths: list[tuple[str, ...]] = []  # heading path of each body table, in document.tables order

    def read(self) -> list[TextUnit]:
        from docx.table import Table

        for block in self.document.iter_inner_content():
            if isinstance(block, Table):
                self.table_paths.append(self.path)
                self._table(block, f"table {docx_table_id(block) or f'{len(self.table_paths)} (no table_id)'}")
            else:
                self._paragraph(block)
        self._headers_and_footers()
        self._notes()
        return self.units

    def _where(self) -> str:
        return " > ".join(self.path) if self.path else "before Section 0"

    def _paragraph(self, paragraph: Any) -> None:
        self.paragraphs += 1
        text = _paragraph_text(paragraph._p)
        style = paragraph.style.name if paragraph.style is not None else ""
        heading = _HEADING_STYLE.fullmatch(style or "")
        if heading:
            level = int(heading.group(1))
            self.path = (*self.path[: level - 1], text.strip())
            self.units.append(TextUnit("heading", text, self.path, f"{self._where()} (heading, paragraph {self.paragraphs})", level))
        elif style == "Title":
            self.units.append(TextUnit("title", text, self.path, f"title (paragraph {self.paragraphs})"))
        else:
            self.units.append(TextUnit("paragraph", text, self.path, f"{self._where()}, paragraph {self.paragraphs}"))

    def _table(self, table: Table, name: str) -> None:
        seen: set[Any] = set()  # the w:tc elements read; holding them keeps lxml's proxies (and identities) alive
        for r, row in enumerate(table.rows):
            for c, cell in enumerate(row.cells):
                if cell._tc in seen:  # a merged cell is listed once per grid column it spans
                    continue
                seen.add(cell._tc)
                text = "\n".join(_paragraph_text(p) for p in cell._tc.iterchildren(_P))
                row_name = "header row" if r == 0 else f"row {r}"
                where = f"{self._where()}, {name}, {row_name}, column {c + 1}"
                self.units.append(TextUnit("cell", text, self.path, where))
                for n, nested in enumerate(cell.tables, start=1):
                    self._table(nested, f"{name}, {row_name}, column {c + 1}, nested table {n}")

    def _headers_and_footers(self) -> None:
        for number, section in enumerate(self.document.sections, start=1):
            for attribute in ("header", "first_page_header", "even_page_header", "footer", "first_page_footer", "even_page_footer"):
                part = getattr(section, attribute)
                if part.is_linked_to_previous:  # no definition of its own; reading it would add one
                    continue
                kind = "header" if attribute.endswith("header") else "footer"
                where = f"{attribute.replace('_', ' ')} of document section {number}"
                for p in part._element.iter(_P):
                    self.units.append(TextUnit(kind, _paragraph_text(p), (), where))

    def _notes(self) -> None:
        from lxml import etree

        for part in self.document.part.package.iter_parts():
            kind = _NOTE_PARTS.get(str(part.partname))
            if kind is None:
                continue
            root = etree.fromstring(part.blob)
            for note in root.iterchildren(f"{_W}{kind}"):
                if note.get(f"{_W}type") in _SEPARATOR_NOTES:
                    continue
                for p in note.iter(_P):
                    self.units.append(TextUnit(kind, _paragraph_text(p), (), f"{kind} {note.get(f'{_W}id')}"))


def _paragraph_text(p: Any) -> str:
    """The shown text of a ``w:p`` element: its ``w:t`` texts, with run tabs and breaks as \\t and \\n."""
    parts: list[str] = []
    for node in p.iter(_T, *_BREAKS):
        if node.tag == _T:
            parts.append(node.text or "")
        elif node.getparent() is not None and node.getparent().tag == _R:  # not a tab stop of w:pPr
            parts.append(_BREAKS[node.tag])
    return "".join(parts)


# --------------------------------------------------------------------- text rules (2.9, 2.10)

def banned_phrase_sentences(text: str) -> list[str]:
    """Each sentence of ``text`` that holds a BANNED_PHRASES match, once, in text order."""
    spans = _sentence_spans(text)
    found: set[tuple[int, int]] = set()
    for pattern in BANNED_PHRASES:
        for match in pattern.finditer(text):
            first = _span_of(spans, match.start())
            last = _span_of(spans, match.end() - 1)
            found.add((spans[first][0], spans[last][1]))
    return [text[start:end].strip() for start, end in sorted(found)]


def citations(text: str) -> list[tuple[str, str]]:
    """(kind, citation) for every URL and every DOI outside a URL in ``text``, in text order."""
    urls = [match for match in URL_PATTERN.finditer(text)]
    found = [(match.start(), "URL", match.group(0)) for match in urls]
    for match in DOI_PATTERN.finditer(text):
        if not any(url.start() <= match.start() < url.end() for url in urls):
            found.append((match.start(), "DOI", match.group(0)))
    return [(kind, citation) for _, kind, citation in sorted(found)]


def is_verified(citation: str, verified: Iterable[str]) -> bool:
    """True when a URL or DOI equals a Verified_Source_List entry, as found or without trailing punctuation."""
    entries = set(verified)
    return citation in entries or citation.rstrip(_TRAILING) in entries


def is_references_heading(text: str) -> bool:
    """True for a references heading: "References", "A.4 References", "6. References:" (any letter case)."""
    return bool(_REFERENCES.fullmatch(text.strip()))


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for match in _SENTENCE_BREAK.finditer(text):
        spans.append((start, match.start()))
        start = match.end()
    spans.append((start, len(text)))
    return spans


def _span_of(spans: Sequence[tuple[int, int]], position: int) -> int:
    """Index of the last sentence span that starts at or before ``position``."""
    index = 0
    for number, (start, _) in enumerate(spans):
        if start > position:
            break
        index = number
    return index


def _excerpt(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= _EXCERPT else flat[: _EXCERPT - 3] + "..."


# ------------------------------------------------------------------------------- the check

def report_check(
    docx: Path | str,
    specs: Iterable[TableSpec],
    settings: Settings,
    *,
    main_lists: Mapping[str, Path | str] | None = None,
) -> list[CheckFailure]:
    """Every Report_Check failure of the exported DOCX (empty when it passes); see ``run_report_check``."""
    return list(run_report_check(docx, specs, settings, main_lists=main_lists).failures)


def run_report_check(
    docx: Path | str,
    specs: Iterable[TableSpec],
    settings: Settings,
    *,
    main_lists: Mapping[str, Path | str] | None = None,
) -> ReportCheckResult:
    """Run every check of CHECKS on the DOCX at ``docx`` and return all failures with the counts.

    specs: the TableSpec of every report table (``BuiltReport.specs``). settings: the Config, for
    ``report.verified_sources`` and the outputs folder. main_lists: Role -> Main_List CSV for the
    Section 0 check (default ``outputs/final/top10_batters.csv`` and ``top10_bowlers.csv``).
    A DOCX that cannot be opened gives one failure and no other check.
    """
    import docx as python_docx

    path = Path(docx)
    spec_list = list(specs)
    bad = [type(spec).__name__ for spec in spec_list if not isinstance(spec, TableSpec)]
    if bad:
        raise TypeError(f"specs must hold TableSpecs only, got {', '.join(sorted(set(bad)))}")
    verified = _verified_sources(settings)
    root = settings.root
    if main_lists is None:
        outputs = settings.path("outputs")
        main_lists = {role: outputs / OUTPUT_FILES[f"top10_{role}"] for role in ROLES}
    try:
        document = python_docx.Document(str(path))
    except Exception as exc:  # a missing or broken file: nothing else can be checked
        failure = CheckFailure("docx", _shown(path, root), f"cannot open the report: {type(exc).__name__}: {exc}")
        return ReportCheckResult(path, (failure,), 0, len(verified), 0, 0, 0, 0, (), ())

    failures: list[CheckFailure] = []
    reader = _Reader(document)
    units = _guarded(failures, "docx", "the Report_Text", reader.read) or []
    allowed: list[str] = []
    counts = {"tables": 0, "rows": 0, "cells": 0, "ranks": 0}
    steps: tuple[tuple[str, str, Callable[[], None]], ...] = (
        ("headings", "the top-level headings", lambda: _check_headings(document, failures)),
        ("placeholder", "the placeholders", lambda: _check_placeholders(units, failures, allowed)),
        ("banned_phrase", "the banned phrases", lambda: _check_banned(units, failures)),
        ("citation", "the citations", lambda: _check_citations(units, verified, failures)),
        ("latex_residue", "the LaTeX residue", lambda: _check_latex(units, failures)),
        ("table", "the tables", lambda: _check_tables(document, spec_list, reader.table_paths, root, failures, counts)),
        ("section0", "the Section 0 tables", lambda: _check_section0(document, spec_list, main_lists, root, failures, counts)),
    )
    for check, what, step in steps:
        _guarded(failures, check, what, step)
    return ReportCheckResult(
        docx=path,
        failures=tuple(failures),
        text_units=len(units),
        verified_sources=len(verified),
        tables_compared=counts["tables"],
        rows_compared=counts["rows"],
        cells_compared=counts["cells"],
        ranks_compared=counts["ranks"],
        allowed_placeholders=tuple(allowed),
        checks_run=tuple(check for check, _, _ in steps),
    )


def _guarded(failures: list[CheckFailure], check: str, what: str, step: Callable[[], Any]) -> Any:
    """Run one check; an unexpected error becomes a failure, so the other checks still run (27.17)."""
    try:
        return step()
    except Exception as exc:
        failures.append(CheckFailure(check, what, f"the check could not finish: {type(exc).__name__}: {exc}"))
        return None


def _verified_sources(settings: Settings) -> frozenset[str]:
    entries = settings.get("report.verified_sources") or []
    if isinstance(entries, (str, bytes)) or not all(isinstance(entry, str) for entry in entries):
        raise TypeError(f"report.verified_sources must be a list of str, got {entries!r}")
    return frozenset(entry.strip() for entry in entries if entry.strip())


def _shown(path: Path | str, root: Path | None) -> str:
    candidate = Path(path)
    if root is not None:
        try:
            return candidate.resolve().relative_to(Path(root).resolve()).as_posix()
        except ValueError:
            pass
    return candidate.as_posix()


# ---------------------------------------------------------------------------- the checks

def _check_headings(document: Document, failures: list[CheckFailure]) -> None:
    found = top_level_headings(document)
    if tuple(found) != SECTION_HEADINGS:
        notes = []
        missing = [heading for heading in SECTION_HEADINGS if heading not in found]
        extra = [heading for heading in found if heading not in SECTION_HEADINGS]
        if missing:
            notes.append(f"missing {missing}")
        if extra:
            notes.append(f"not expected {extra}")
        failures.append(
            CheckFailure(
                "headings", "the top-level (Heading 1) headings",
                f"found {found}; expected {list(SECTION_HEADINGS)} in this order "
                f"({'; '.join(notes) if notes else 'the order differs'})",
            )
        )  # fmt: skip


def _check_placeholders(units: Sequence[TextUnit], failures: list[CheckFailure], allowed: list[str]) -> None:
    markers = [marker.casefold() for marker in PLACEHOLDER_MARKERS]
    for unit in units:
        folded = unit.text.casefold()
        hits = [marker for marker, key in zip(PLACEHOLDER_MARKERS, markers) if key in folded]
        if not hits:
            continue
        permit = next((heading for heading in ALLOWED_PLACEHOLDERS if unit.under(heading)), None)
        if permit is not None:
            note = f"{' > '.join(unit.headings[: unit.headings.index(permit) + 1])} ({ALLOWED_PLACEHOLDERS[permit]})"
            if note not in allowed:
                allowed.append(note)
            continue
        failures.append(
            CheckFailure("placeholder", unit.where, f"placeholder text {hits} remains: {_excerpt(unit.text)!r}")
        )


def _check_banned(units: Sequence[TextUnit], failures: list[CheckFailure]) -> None:
    for unit in units:
        for sentence in banned_phrase_sentences(unit.text):
            failures.append(CheckFailure("banned_phrase", unit.where, f"banned phrase in the sentence {sentence!r}"))


def _check_citations(units: Sequence[TextUnit], verified: frozenset[str], failures: list[CheckFailure]) -> None:
    for unit in units:
        entry = unit.text.strip()
        reference = unit.kind == "paragraph" and bool(entry) and any(is_references_heading(h) for h in unit.headings)
        if reference:
            if entry in verified:
                continue  # its URLs and DOIs are covered by the verified entry
            failures.append(
                CheckFailure("citation", unit.where, f"reference entry {_excerpt(entry)!r} is not in the Verified_Source_List")
            )
        for kind, citation in citations(unit.text):
            if not is_verified(citation, verified):
                shown = citation.rstrip(_TRAILING) or citation
                failures.append(CheckFailure("citation", unit.where, f"{kind} {shown!r} is not in the Verified_Source_List"))


def _check_latex(units: Sequence[TextUnit], failures: list[CheckFailure]) -> None:
    for unit in units:
        found = latex_residue(unit.text)
        if found:
            failures.append(
                CheckFailure("latex_residue", unit.where, f"raw LaTeX {found} in {_excerpt(unit.text)!r}")
            )


def _check_tables(
    document: Document,
    specs: Sequence[TableSpec],
    table_paths: Sequence[tuple[str, ...]],
    root: Path | None,
    failures: list[CheckFailure],
    counts: dict[str, int],
) -> None:
    ids = Counter(spec.table_id for spec in specs)
    for table_id, count in sorted(ids.items()):
        if count > 1:
            failures.append(CheckFailure("table", f"table {table_id}", f"{count} TableSpecs share this table_id"))
    for number, table in enumerate(document.tables, start=1):
        table_id = docx_table_id(table)
        place = " > ".join(table_paths[number - 1]) if number <= len(table_paths) else "the report"
        place = place or "the report"
        if table_id is None:
            failures.append(
                CheckFailure(
                    "table", f"{place}, table {number}",
                    "the table has no table_id, so it was not built from a CSV through a TableSpec and cannot be "
                    "compared with one (DD-12)",
                )
            )  # fmt: skip
        elif table_id not in ids:
            failures.append(
                CheckFailure("table", f"{place}, table {table_id}", "no TableSpec has this table_id, so the table cannot be compared")
            )
    for spec in specs:
        _guarded(failures, "table", f"table {spec.table_id}", lambda spec=spec: _check_table(document, spec, root, failures, counts))


def _check_table(
    document: Document, spec: TableSpec, root: Path | None, failures: list[CheckFailure], counts: dict[str, int]
) -> None:
    def fail(location: str, detail: str) -> None:
        failures.append(CheckFailure("table", f"table {spec.table_id}{location}", detail))

    try:
        table = find_docx_table(document, spec.table_id)
    except TableError as exc:
        fail("", str(exc))
        return
    if table is None:
        fail("", "the table is missing from the report")
        return
    try:
        expected = render_table(spec)
    except TableError as exc:
        fail("", f"cannot render the table from its CSV again: {exc}")
        return
    cells = docx_table_text(table)
    source = _shown(spec.csv, root)
    width = len(spec.columns)
    if not cells:
        fail("", "the table has no rows")
        return
    if tuple(cells[0]) != spec.header_texts:
        fail(", header row", f"the report shows {cells[0]}, the spec's headers are {list(spec.header_texts)}")
    shown_rows: dict[tuple[str, ...], list[int]] = {}
    for number, row in enumerate(cells[1:], start=1):
        if len(row) != width:
            fail(f", row {number}", f"the row has {len(row)} cells, the spec shows {width} columns")
            continue
        key = tuple(row[p] for p in spec.key_positions)
        shown_rows.setdefault(key, []).append(number)
    expected_keys = set(expected.keys)

    def key_text(key: Sequence[str]) -> str:
        return ", ".join(f"{name}={value!r}" for name, value in zip(spec.key, key))

    for key, numbers in shown_rows.items():
        if key not in expected_keys:
            fail(f", row {key_text(key)}", f"the report shows a row with this key (table row {numbers[0]}) that {source} does not give")
        elif len(numbers) > 1:
            fail(f", row {key_text(key)}", f"the key is shown on {len(numbers)} rows ({numbers})")
    counts["tables"] += 1
    for i, key in enumerate(expected.keys):
        numbers = shown_rows.get(key)
        if not numbers:
            fail(f", row {key_text(key)}", f"the report has no row for {source} data row {expected.source_rows[i]}")
            continue
        row = cells[numbers[0]]
        counts["rows"] += 1
        for j, column in enumerate(spec.columns):
            counts["cells"] += 1
            shown, raw = row[j], expected.raw[i][j]
            if not cell_matches(spec, column, shown, raw):
                rule = (
                    "a number more than 0.5 × 10^(−d) from the CSV value, d being the decimals shown"
                    if spec.is_numeric(column)
                    else "text that differs from the CSV text"
                )
                fail(
                    f", row {key_text(key)}, column {column!r}",
                    f"the report shows {shown!r}, {source} data row {expected.source_rows[i]} has {raw!r} ({rule})",
                )


def _check_section0(
    document: Document,
    specs: Sequence[TableSpec],
    main_lists: Mapping[str, Path | str],
    root: Path | None,
    failures: list[CheckFailure],
    counts: dict[str, int],
) -> None:
    by_id = {spec.table_id: spec for spec in specs}
    for role in ROLES:
        _guarded(
            failures, "section0", f"Section 0 table {SECTION0_TABLES[role]}",
            lambda role=role: _check_main_list(document, by_id, role, Path(main_lists[role]), root, failures, counts),
        )  # fmt: skip


def _check_main_list(
    document: Document,
    by_id: Mapping[str, TableSpec],
    role: str,
    csv_path: Path,
    root: Path | None,
    failures: list[CheckFailure],
    counts: dict[str, int],
) -> None:
    table_id = SECTION0_TABLES[role]
    source = _shown(csv_path, root)

    def fail(location: str, detail: str) -> None:
        failures.append(CheckFailure("section0", f"Section 0 table {table_id}{location}", detail))

    spec = by_id.get(table_id)
    if spec is None or "rank" not in spec.columns or "player_name" not in spec.columns:
        fail("", "no TableSpec of this table shows the rank and player_name columns")
        return
    try:
        table = find_docx_table(document, table_id)
    except TableError as exc:
        fail("", str(exc))
        return
    if table is None:
        fail("", "the table is missing from the report")
        return
    try:
        header, records = read_csv_text(csv_path)
    except TableError as exc:
        fail("", f"cannot read the Main_List file: {exc}")
        return
    absent = [name for name in ("rank", "player_name") if name not in header]
    if absent:
        fail("", f"{source} has no column(s) {absent}")
        return
    at_rank, at_name = header.index("rank"), header.index("player_name")
    expected = [(record[at_rank], record[at_name]) for record in records]
    rank_at, name_at = spec.columns.index("rank"), spec.columns.index("player_name")
    shown = [
        (row[rank_at], row[name_at]) if len(row) > max(rank_at, name_at) else ("", "")
        for row in docx_table_text(table)[1:]
    ]
    for position in range(max(len(shown), len(expected))):
        counts["ranks"] += 1
        got = shown[position] if position < len(shown) else None
        want = expected[position] if position < len(expected) else None
        if got is not None and want is not None and _same_rank(got[0], want[0]) and got[1] == want[1]:
            continue
        rank = want[0] if want is not None else (got[0] if got is not None else str(position + 1))
        if want is None:
            detail = f"table row {position + 1} shows rank {got[0]!r}, {got[1]!r}, but {source} has only {len(expected)} rows"
        elif got is None:
            detail = f"{source} gives rank {want[0]!r}, {want[1]!r} at data row {position + 1}, but the table has only {len(shown)} rows"
        else:
            detail = (
                f"table row {position + 1} shows rank {got[0]!r}, {got[1]!r}; {source} data row {position + 1} "
                f"gives rank {want[0]!r}, {want[1]!r}"
            )
        fail(f", rank {rank}", detail)


def _same_rank(shown: str, expected: str) -> bool:
    """Ranks are equal as numbers ("1" and "1.0"), or as text when either is not a number."""
    try:
        return Decimal(shown.strip()) == Decimal(expected.strip())
    except (InvalidOperation, ValueError):
        return shown == expected
