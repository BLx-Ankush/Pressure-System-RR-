"""Report equations: matplotlib mathtext rendered to PNG and embedded as pictures (Requirement 27, criterion 13).

Design section "Report_Builder": equations are rendered with matplotlib mathtext, which needs no
LaTeX install, and embedded as images, so no paragraph or table cell of the report holds raw
LaTeX. ``LATEX_RESIDUE`` holds the two patterns that the Report_Check (Requirement 27, criterion
27) looks for in every paragraph and table cell: a backslash followed by a letter, and text
between two ``$``. The builder writes each equation's symbol definitions as a 9 pt "where" line
of plain text below the picture (``superr.report.builder``).

Rendering (``render_equation``)
-------------------------------
* Input: a mathtext string. Without any ``$`` the whole string is one math expression and is
  wrapped in ``$...$``; with ``$`` signs it is used as given (plain text with math parts).
* matplotlib's Agg canvas draws it on a ``Figure`` (no pyplot, so the global backend is never
  changed), with ``usetex`` off, the Computer Modern mathtext fonts (``EQUATION_FONTSET``)
  bundled with matplotlib, black on white, and EQUATION_PAD_PT points of margin on each side.
* The PNG has a fixed DPI (EQUATION_DPI unless given) and no "Software" metadata, so the same
  input gives the same bytes with the pinned matplotlib. It is written through ``<file>.tmp``
  renamed in place.

The PNG records its DPI, so ``add_picture`` places it at its natural size: the equation text then
appears at ``fontsize`` points in the document. A picture wider than ``max_width`` is scaled
down to that width. The picture's alt text must be plain words (``plain_text_problems``).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from docx.shape import InlineShape
    from docx.text.paragraph import Paragraph

__all__ = [
    "EQUATION_DPI",
    "EQUATION_FONTSET",
    "EQUATION_FONTSIZE",
    "EQUATION_PAD_PT",
    "LATEX_RESIDUE",
    "add_picture",
    "check_mathtext",
    "latex_residue",
    "plain_text_problems",
    "render_equation",
]

#: Resolution of every equation PNG in dots per inch, fixed so that the same input gives the same image.
EQUATION_DPI = 300
#: Font size of the equations in points, the size of the body text (``superr.report.builder``; DD-18).
EQUATION_FONTSIZE = 10.5
#: matplotlib mathtext font set: Computer Modern, bundled with matplotlib.
EQUATION_FONTSET = "cm"
#: White margin around each equation, in points.
EQUATION_PAD_PT = 2.0

#: Raw-LaTeX residue that Report_Text must not contain (Requirement 27, criteria 13 and 27; design
#: section "Report_Builder"): a backslash followed by a letter, and text between two ``$`` signs.
LATEX_RESIDUE: tuple[re.Pattern[str], ...] = (re.compile(r"\\[A-Za-z]"), re.compile(r"\$[^$]+\$"))

_POINTS_PER_INCH = 72.0


def latex_residue(text: str) -> list[str]:
    """Every raw-LaTeX match in ``text`` (LATEX_RESIDUE), in pattern order then text order."""
    return [match.group(0) for pattern in LATEX_RESIDUE for match in pattern.finditer(text)]


def plain_text_problems(text: Any, what: str) -> list[str]:
    """Why ``text`` cannot be an equation's alt text or caption: not a str, blank, or raw LaTeX."""
    if not isinstance(text, str):
        return [f"{what} must be a str, got {type(text).__name__}"]
    if not text.strip():
        return [f"{what} must not be blank"]
    found = latex_residue(text)
    if found:
        return [f"{what} {text!r} holds raw LaTeX {found}; describe the equation in plain words"]
    return []


def check_mathtext(mathtext: str, *, fontsize: float = EQUATION_FONTSIZE) -> str:
    """The mathtext string that render_equation draws, after checking that matplotlib can parse it.

    Raises ValueError, with matplotlib's message, for mathtext that does not parse.
    """
    text = _math_string(mathtext)
    _layout(text, _font(fontsize))
    return text


def render_equation(
    mathtext: str,
    path: Path | str,
    *,
    fontsize: float = EQUATION_FONTSIZE,
    dpi: int = EQUATION_DPI,
) -> Path:
    """Render matplotlib mathtext to a PNG at ``path`` and return the path.

    ValueError for mathtext that does not parse, a non-positive font size or a DPI that is not a
    positive int. The parent folder is created when needed.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if type(dpi) is not int or dpi <= 0:
        raise ValueError(f"dpi must be a positive int, got {dpi!r}")
    text = _math_string(mathtext)
    font = _font(fontsize)
    width, height, depth = _layout(text, font)
    pad = EQUATION_PAD_PT
    full_width, full_height = width + 2 * pad, height + 2 * pad
    figure = Figure(
        figsize=(full_width / _POINTS_PER_INCH, full_height / _POINTS_PER_INCH), dpi=dpi, facecolor="white"
    )
    FigureCanvasAgg(figure)
    figure.text(
        pad / full_width, (depth + pad) / full_height, text,
        fontproperties=font, color="black", usetex=False, ha="left", va="baseline",
    )  # fmt: skip
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        figure.savefig(
            temporary, format="png", dpi=dpi, facecolor="white", transparent=False,
            bbox_inches=None, metadata={"Software": None},
        )  # fmt: skip
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def add_picture(
    paragraph: Paragraph,
    png: Path | str,
    *,
    alt_text: str,
    max_width: int | None = None,
    width: int | None = None,
) -> InlineShape:
    """Append a PNG to ``paragraph`` as an inline picture and return it.

    The picture keeps its natural size (pixels ÷ the DPI the PNG records), or is shown ``width``
    EMU wide when given, and is then scaled down to ``max_width`` EMU when wider; the height keeps
    the aspect ratio. ``alt_text`` is stored as the picture's description and must be plain words
    (ValueError otherwise). Used for equations and for charts.
    """
    from docx.image.image import Image
    from docx.shared import Emu

    problems = plain_text_problems(alt_text, "the picture's alt text")
    if problems:
        raise ValueError("; ".join(problems))
    source = Path(png)
    if not source.is_file():
        raise FileNotFoundError(f"the picture {source} does not exist")
    shown = int(Image.from_file(str(source)).width) if width is None else int(width)
    if max_width is not None:
        shown = min(shown, int(max_width))
    if shown <= 0:
        raise ValueError(f"the picture {source.name} would be {shown} EMU wide")
    shape = paragraph.add_run().add_picture(str(source), width=Emu(shown))
    shape._inline.docPr.set("descr", alt_text)
    return shape


# --------------------------------------------------------------------------------- helpers

def _math_string(mathtext: Any) -> str:
    if not isinstance(mathtext, str) or not mathtext.strip():
        raise ValueError(f"an equation needs non-blank mathtext, got {mathtext!r}")
    return mathtext if "$" in mathtext else f"${mathtext}$"


def _font(fontsize: Any) -> Any:
    from matplotlib.font_manager import FontProperties

    if isinstance(fontsize, bool) or not isinstance(fontsize, (int, float)) or not fontsize > 0:
        raise ValueError(f"fontsize must be a positive number of points, got {fontsize!r}")
    return FontProperties(size=float(fontsize), math_fontfamily=EQUATION_FONTSET)


def _layout(text: str, font: Any) -> tuple[float, float, float]:
    """(width, height, depth) of the mathtext in points; ValueError when it does not parse."""
    from matplotlib.mathtext import MathTextParser

    try:
        parsed = MathTextParser("path").parse(text, dpi=72, prop=font)
    except ValueError as exc:
        raise ValueError(f"matplotlib cannot parse the mathtext {text!r}:{exc}") from None
    width, height, depth = float(parsed[0]), float(parsed[1]), float(parsed[2])
    if not (width > 0 and height > 0):
        raise ValueError(f"the mathtext {text!r} draws nothing")
    return width, height, depth
