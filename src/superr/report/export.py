"""Report export and the ``report`` Pipeline_Stage.

Requirement 27, criteria 10-12 and 15-17, and Requirement 28, criterion 1; design section
"Report_Builder" and stage 20 of the Pipeline_Stage registry. ``run_stage`` builds the
methodology report from the outputs of the earlier stages, exports it to ``outputs/report/``,
counts the pages of Sections 0-5 when a PDF exists, and runs the Report_Check
(``superr.report.checks``).

Files in ``outputs/report/``
----------------------------
* REPORT_FILE, ``SupeRR_S2_methodology_report.docx``: the report (27.10), written through
  ``<file>.tmp`` renamed in place (``export_docx``).
* ``SupeRR_S2_methodology_report.pdf``: only when the Config names a PDF converter and the
  converter produced a PDF (27.11). Any PDF of that name from an earlier Run is deleted before
  the export, so the folder never holds a PDF of another DOCX.
* ``design_choices.csv``: the Appendix Design_Choice table, which the Appendix writes before the
  build (``superr.report.sections``); the Report_Check reads it again.

The PDF converter (Config ``report.pdf_converter``)
---------------------------------------------------
* ``None`` (as delivered): no PDF. The stage logs a warning that only the DOCX file was exported
  and goes on (27.15).
* A LibreOffice program: ``"soffice"`` or ``"libreoffice"`` on the PATH, or the program's full
  path, such as ``"C:/Program Files/LibreOffice/program/soffice.exe"`` (spaces are fine; the
  whole value is the program). It is called with the argument list
  ``[converter, "--headless", "--convert-to", "pdf", "--outdir", <folder>, <docx>]``.
* Any other converter: a command template that holds ``{docx}`` and may hold ``{pdf}`` and
  ``{outdir}``, for example ``"docx2pdf {docx} {pdf}"`` or ``"pandoc {docx} -o {pdf}"``. The
  template is split into arguments at white space by ``shlex`` with Windows rules: a backslash is
  kept as typed, and an argument that holds spaces is wrapped in double quotes, which are
  removed. Each placeholder is then replaced inside its argument, so a path with spaces stays one
  argument. The converter must write ``{pdf}``, which is ``{outdir}/<DOCX file stem>.pdf``.

Either way the converter runs as an argument list, never through a shell, with no standard input,
a time limit of PDF_TIMEOUT_SECONDS and a temporary output folder under ``outputs/report/``; the
PDF it writes there is moved into place. A converter that is not found, cannot start, fails, runs
out of time or writes no PDF (a file that starts with ``%PDF-``) produces no PDF: the stage logs
the warning of criterion 15 with the reason and goes on with the DOCX alone. On Windows, a
time-out stops the converter process but not the processes it started itself (LibreOffice's
``soffice.bin``).

Page count of Sections 0-5 (27.12, 27.16)
-----------------------------------------
``heading_pages`` reads the PDF with pypdf and finds the page of each top-level heading
(SECTION_HEADINGS), in order, each at or after the page of the one before: a page holds a heading
when one of its text lines equals the heading, ignoring white space and letter case after NFKC
normalisation (which undoes ligatures such as "ﬁ"). ``sections_page_count`` is the physical page
index of the Appendix heading minus that of the Section 0 heading; the Appendix heading starts a
new page, so this counts the pages of Sections 0-5. The stage writes it to the Run_Log, with a
warning when it exceeds Config ``report.page_budget`` (15); the Report_Check never fails for it. A
PDF in which a heading cannot be found gives a warning instead of a page count.

The ``report`` stage (``run_stage``)
------------------------------------
1. ``ReportInputs.from_context`` reads every input file and constant, ``ReportBuilder.from_context``
   and ``write_sections`` fill Sections 0-5 and the Appendix, and ``export_docx`` writes the DOCX.
2. ``export_pdf`` with the Config converter; with a PDF, the page count goes to the Run_Log.
3. ``run_report_check`` runs every check; each failure goes to the Run_Log as an error.
4. The checkpoint JSON CHECK_JSON (``report_check.json``): the check results, the SHA-256 of each
   exported file, the PDF and the page count when there is one. It is saved before step 5, so a
   failing Run keeps it next to the exported files.
5. Any failure raises ReportCheckError (exit code 6) after every check has run; the exported files
   stay in ``outputs/report/`` (27.17).
"""

from __future__ import annotations

import dataclasses
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from superr.errors import ReportCheckError, StageError
from superr.protect import sha256_file
from superr.report.builder import SECTION_HEADINGS, STAGE_NAME, BuiltReport, ReportBuilder
from superr.report.checks import run_report_check
from superr.report.sections import ReportInputs, write_sections

if TYPE_CHECKING:
    from superr.pipeline import StageContext
    from superr.runlog import RunLog

__all__ = [
    "CHECK_JSON",
    "LIBREOFFICE_ARGUMENTS",
    "NO_PDF_WARNING",
    "PDF_TIMEOUT_SECONDS",
    "REPORT_FILE",
    "REPORT_FOLDER",
    "STAGE_NAME",
    "TEMPLATE_FIELDS",
    "converter_command",
    "export_docx",
    "export_pdf",
    "heading_pages",
    "run_stage",
    "sections_page_count",
]

#: Folder of the exported report under the outputs folder, and the DOCX file name in it.
REPORT_FOLDER = "report"
REPORT_FILE = "SupeRR_S2_methodology_report.docx"
#: Checkpoint JSON value of the ``report`` stage.
CHECK_JSON = "report_check"
#: Longest time a PDF converter may run, in seconds.
PDF_TIMEOUT_SECONDS = 300
#: Arguments after the program of a LibreOffice converter; the output folder and the DOCX follow.
LIBREOFFICE_ARGUMENTS: tuple[str, ...] = ("--headless", "--convert-to", "pdf", "--outdir")
#: Placeholders of a converter command template.
TEMPLATE_FIELDS: tuple[str, ...] = ("docx", "pdf", "outdir")
#: The words of the warning when no PDF is produced (Requirement 27, criterion 15).
NO_PDF_WARNING = "only the DOCX file was exported"

_COMPONENT = "Report_Builder"
_FIELD = re.compile(r"\{([A-Za-z_]\w*)\}")
_QUOTES = "\"'"
_OUTPUT_TAIL = 600  # characters of converter output quoted in a warning


# ----------------------------------------------------------------------------------- DOCX

def export_docx(built: BuiltReport, outdir: Path | str, *, name: str = REPORT_FILE) -> BuiltReport:
    """Save the built report as ``outdir/name`` through ``<file>.tmp`` renamed in place (27.10).

    Returns the build with its ``path`` set. StageError when the file cannot be replaced, for
    example because Word holds it open.
    """
    if not isinstance(name, str) or Path(name).name != name or not name.lower().endswith(".docx"):
        raise ValueError(f"the report file name must be a plain file name ending in .docx, got {name!r}")
    folder = Path(outdir)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / name
    temporary = target.with_name(target.name + ".tmp")
    try:
        built.document.save(str(temporary))
        try:
            os.replace(temporary, target)
        except PermissionError as exc:
            raise StageError(
                f"cannot replace the report {target}: {exc}. Close it if Word or another program has it open, "
                "then run the report stage again"
            ) from None
    finally:
        temporary.unlink(missing_ok=True)
    return dataclasses.replace(built, path=target)


# ------------------------------------------------------------------------------------ PDF

def converter_command(converter: str, docx: Path | str, outdir: Path | str) -> list[str]:
    """The argument list that converts ``docx`` into ``outdir/<stem>.pdf`` (module docstring).

    ValueError for a blank converter, a template with an unknown placeholder or an empty template.
    """
    if not isinstance(converter, str) or not converter.strip():
        raise ValueError(f"the PDF converter must be a non-blank str, got {converter!r}")
    source, folder = Path(docx), Path(outdir)
    text = converter.strip()
    if "{docx}" not in text:  # a LibreOffice program: the whole value names it
        return [_unquote(text), *LIBREOFFICE_ARGUMENTS, str(folder), str(source)]
    unknown = sorted({name for name in _FIELD.findall(text) if name not in TEMPLATE_FIELDS})
    if unknown:
        raise ValueError(f"the converter template {text!r} has unknown placeholder(s) {unknown}; use {list(TEMPLATE_FIELDS)}")
    values = {"docx": str(source), "pdf": str(folder / f"{source.stem}.pdf"), "outdir": str(folder)}
    arguments = [
        _FIELD.sub(lambda match: values[match.group(1)], _unquote(token)) for token in shlex.split(text, posix=False)
    ]
    if not arguments:
        raise ValueError(f"the converter template {text!r} names no program")
    return arguments


def export_pdf(
    docx: Path | str,
    converter: str | None,
    *,
    log: RunLog | None = None,
    root: Path | str | None = None,
    timeout: float = PDF_TIMEOUT_SECONDS,
) -> Path | None:
    """Convert the DOCX into ``<docx folder>/<stem>.pdf`` with the Config converter (27.11).

    Returns the PDF, or None when no PDF was produced (no converter, or a converter that is not
    found, fails, runs out of time or writes no PDF). Then, with a ``log``, it logs the warning of
    Requirement 27, criterion 15 with the reason. root: the workspace root, for the paths shown.
    A PDF of that name from an earlier Run is deleted first.
    """
    source = Path(docx)
    target = source.with_suffix(".pdf")
    if target.exists():
        target.unlink()
        _info(log, f"{_COMPONENT}: deleted {_display(target, root)}, the PDF of an earlier Run, before the export")
    if converter is None:
        return _no_pdf(log, source, root, "the Config names no PDF converter (report.pdf_converter is None)")
    try:
        with tempfile.TemporaryDirectory(prefix="_pdf_", dir=source.parent, ignore_cleanup_errors=True) as folder:
            work = Path(folder)
            try:
                arguments = converter_command(converter, source, work)
            except ValueError as exc:
                return _no_pdf(log, source, root, f"the PDF converter cannot be used: {exc}")
            program = shutil.which(arguments[0])
            if program is None:
                return _no_pdf(
                    log, source, root,
                    f"the PDF converter program {arguments[0]!r} (report.pdf_converter) was not found",
                )  # fmt: skip
            arguments[0] = program
            started = time.perf_counter()
            try:
                completed = subprocess.run(
                    arguments, stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout, check=False
                )
            except subprocess.TimeoutExpired:
                return _no_pdf(log, source, root, f"the PDF converter {program} did not finish within {timeout:g} seconds")
            except OSError as exc:
                return _no_pdf(log, source, root, f"the PDF converter {program} could not start: {exc}")
            seconds = time.perf_counter() - started
            produced = work / f"{source.stem}.pdf"
            if completed.returncode != 0 or not _is_pdf(produced):
                problem = (
                    f"exited with code {completed.returncode}"
                    if completed.returncode != 0
                    else f"exited with code 0 but wrote no PDF file {produced.name}"
                )
                return _no_pdf(
                    log, source, root,
                    f"the PDF converter {program} {problem}{_output_tail(completed.stdout, completed.stderr)}",
                )  # fmt: skip
            os.replace(produced, target)
    except OSError as exc:  # the temporary folder or the move into place
        return _no_pdf(log, source, root, f"the PDF could not be written: {type(exc).__name__}: {exc}")
    _info(
        log,
        f"{_COMPONENT}: converted {_display(source, root)} into {_display(target, root)} with {program} in "
        f"{seconds:.1f} s (Requirement 27, criterion 11)",
    )
    return target


def heading_pages(pdf: Path | str, headings: Sequence[str] = SECTION_HEADINGS) -> list[int]:
    """The 0-based physical page index of each heading in a PDF, searched in order (module docstring).

    ValueError when a heading is on no page at or after the page of the heading before it.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(pdf))
    pages = [{_squeezed(line) for line in (page.extract_text() or "").splitlines()} for page in reader.pages]
    found: list[int] = []
    start = 0
    for heading in headings:
        key = _squeezed(heading)
        index = next((number for number in range(start, len(pages)) if key in pages[number]), None)
        if index is None:
            raise ValueError(
                f"no text line of {Path(pdf).name} equals the heading {heading!r} on PDF page {start + 1} or later "
                f"({len(pages)} pages)"
            )
        found.append(index)
        start = index
    return found


def sections_page_count(pdf: Path | str) -> int:
    """Pages of Sections 0-5: the PDF page index of the Appendix heading minus that of Section 0 (27.12)."""
    pages = heading_pages(pdf)
    return pages[-1] - pages[0]


# ---------------------------------------------------------------------------- the stage

def run_stage(ctx: StageContext) -> None:
    """The ``report`` Pipeline_Stage (module docstring, "The ``report`` stage").

    Raises StageError (exit code 5) when an input file or constant is missing or the DOCX cannot
    be written, and ReportCheckError (exit code 6) when the Report_Check finds any failure, after
    every check has run and with the exported files kept.
    """
    started = time.perf_counter()
    settings, log, root = ctx.settings, ctx.log, ctx.root
    outputs = settings.path("outputs")
    folder = outputs / REPORT_FOLDER

    # 1. The report, from the output files and constants of the earlier stages.
    inputs = ReportInputs.from_context(ctx)
    builder = ReportBuilder.from_context(ctx)
    write_sections(builder, inputs)  # the Appendix writes its Design_Choice CSV first
    built = export_docx(builder.build(), folder)
    docx = built.path
    assert docx is not None
    pictures = len(built.document.inline_shapes)
    log.info(
        f"{_COMPONENT}: wrote {_display(docx, root)} (Requirement 27, criterion 10): the "
        f"{len(SECTION_HEADINGS)} top-level sections, {len(built.tables):,} tables built from CSV files, "
        f"{pictures:,} pictures (charts and equations) and {len(built.labels):,} Method_Constant labels"
    )

    # 2. The PDF and the page count of Sections 0-5 (criteria 11, 12, 15 and 16).
    pdf = export_pdf(docx, settings.get("report.pdf_converter"), log=log, root=root)
    page_count, pages = (None, None) if pdf is None else _page_count(pdf, int(settings.get("report.page_budget")), log, root)

    # 3. The Report_Check: every check runs, and every failure goes to the Run_Log (criterion 17).
    result = run_report_check(docx, built.specs, settings)
    for number, failure in enumerate(result.failures, start=1):
        log.error(f"Report_Check failure {number:,} of {len(result.failures):,}: {failure}")
    log.info(result.summary())

    # 4. The checkpoint: the check results, the exported files' hashes and the page count.
    files = [docx, inputs.path("design_choices"), *([] if pdf is None else [pdf])]
    ctx.store.save_json(
        STAGE_NAME,
        CHECK_JSON,
        {
            "docx": _relative(docx, outputs),
            "pdf": None if pdf is None else _relative(pdf, outputs),
            "pdf_converter": settings.get("report.pdf_converter"),
            "files": {_relative(path, outputs): sha256_file(path) for path in files},
            "page_count": page_count,
            "heading_pages": None if pages is None else dict(zip(SECTION_HEADINGS, pages)),
            "page_budget": settings.get("report.page_budget"),
            "tables": len(built.tables),
            "pictures": pictures,
            "labels": len(built.labels),
            "check": result.to_dict(),
        },
    )
    shown = ", ".join(path.name for path in files)
    log.info(
        f"{_COMPONENT}: stage {time.perf_counter() - started:.1f} s; {_display(folder, root)}/ holds {shown}; saved "
        f"{CHECK_JSON} in {_display(ctx.store.stage_dir(STAGE_NAME), root)}/"
    )

    # 5. A failed Report_Check fails the stage, with the files kept (criterion 17).
    if not result.passed:
        counts = ", ".join(f"{check} {count:,}" for check, count in result.counts().items() if count)
        raise ReportCheckError(
            f"the Report_Check found {len(result.failures):,} failure(s) ({counts}); every check ran and each "
            f"failure is in the Run_Log. The exported files are kept in {_display(folder, root)}/: {shown} "
            "(Requirement 27, criterion 17)"
        )


def _page_count(pdf: Path, budget: int, log: RunLog, root: Path | None) -> tuple[int | None, list[int] | None]:
    shown = _display(pdf, root)
    try:
        pages = heading_pages(pdf)
    except Exception as exc:  # a heading that cannot be found, or a PDF that pypdf cannot read
        log.warning(f"{_COMPONENT}: cannot find the page count of Sections 0-5 in {shown}: {type(exc).__name__}: {exc}")
        return None, None
    count = pages[-1] - pages[0]
    log.info(
        f"{_COMPONENT}: Sections 0-5 take {count} pages of {shown} (Requirement 27, criterion 12): the Section 0 "
        f"heading is on PDF page {pages[0] + 1} and the Appendix heading on PDF page {pages[-1] + 1}, so the page "
        f"count is physical page index {pages[-1]} minus {pages[0]}"
    )
    if count > budget:
        log.warning(
            f"{_COMPONENT}: Sections 0-5 take {count} pages, more than the page budget of {budget} (Config "
            "report.page_budget; Requirement 27, criterion 16); the Report_Check does not fail for this"
        )
    return count, pages


# --------------------------------------------------------------------------------- helpers

def _no_pdf(log: RunLog | None, docx: Path, root: Path | str | None, reason: str) -> None:
    if log is not None:
        log.warning(
            f"{_COMPONENT}: no PDF file was produced, because {reason}, so {NO_PDF_WARNING}: {_display(docx, root)} "
            "(Requirement 27, criterion 15)"
        )
    return None


def _info(log: RunLog | None, message: str) -> None:
    if log is not None:
        log.info(message)


def _unquote(token: str) -> str:
    if len(token) >= 2 and token[0] == token[-1] and token[0] in _QUOTES:
        return token[1:-1]
    return token


def _is_pdf(path: Path) -> bool:
    try:
        with open(path, "rb") as handle:
            return handle.read(5) == b"%PDF-"
    except OSError:
        return False


def _output_tail(*streams: bytes | None) -> str:
    text = " ".join(
        " ".join(stream.decode("utf-8", errors="replace").split()) for stream in streams if stream
    ).strip()
    if not text:
        return ""
    return f"; its output ends: {text[-_OUTPUT_TAIL:]!r}"


def _squeezed(text: str) -> str:
    """Text for the heading search: NFKC-normalised, without white space, case-folded."""
    return "".join(unicodedata.normalize("NFKC", text).split()).casefold()


def _relative(path: Path, outputs: Path) -> str:
    return path.resolve().relative_to(outputs.resolve()).as_posix()


def _display(path: Path | str, root: Path | str | None) -> str:
    candidate = Path(path)
    if root is not None:
        try:
            return candidate.resolve().relative_to(Path(root).resolve()).as_posix()
        except ValueError:
            pass
    return str(candidate)
