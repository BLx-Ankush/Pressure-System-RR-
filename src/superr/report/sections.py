"""Report Sections 0-5 and the Appendix: the text, tables, figures and equations of the methodology report.

Requirement 27, criteria 1-8, and Requirement 2, criteria 8-10; design section "Report_Builder"
and decision DD-12; tasks 23.2 (Sections 0-4) and 23.3 (Section 5 and the Appendix). Each
``section_N(builder, inputs)`` fills Section N of a ``ReportBuilder`` (``superr.report.builder``),
``appendix(builder, inputs)`` fills the Appendix, and ``write_sections`` calls them in order
(SECTION_WRITERS). Task 23.4 adds the Report_Check, the export and the ``report`` stage; task 27.1
fills Section 5.3 (ROBUSTNESS_HEADING) with the repeatability, ablation, sensitivity and cross-check
results of Requirements 20-22 and explains the M4 calibration-aware tie-break in Section 2.3.

Where the numbers come from
---------------------------
No number is typed into the text. ``ReportInputs.load`` reads, once per build, every output file
that the sections use (OUTPUT_FILES, relative to the outputs folder, and DOC_FILES, relative to
the docs folder) and checks that the Run's ConstantRegistry holds every data-derived constant the
text quotes (REQUIRED_CONSTANTS). A missing file or constant raises StageError naming each one
and the stage that writes it. Then:

* every table is a TableSpec on one of those CSV files, so the Report_Check can compare each cell
  with its source (DD-12). Decimals: 3 for runs per over and runs, 4 for probabilities, model
  metrics and win-probability swings, 3 for shares, 1 for percentages and traditional rates, 2
  for economy, 0 for counts; ``where`` keeps the rows a table shows;
* every number in the text is read from those CSV files with pandas, or from the
  ConstantRegistry and the Config, and formatted in code;
* every Method_Constant is written as a ``Const``, so the builder labels it at its first use in
  document order: hand-set (Dn), hand-set, or data-derived (component) (Requirement 2,
  criterion 8). Section 0 and the opening paragraph of Section 1.2 quote the Design_Choices that
  the imported pressure definition repeats as plain numbers, so their labels land first;
* the plain-language and formal pressure definitions are the text that the Pressure_Engine wrote
  to ``docs/pressure_definition.md`` (``parse_definition``): paragraphs and numbered items are
  copied with their bold runs, and each row of the pillar table becomes one bullet, because every
  report table must come from a CSV file;
* the model feature lists come from the model modules (``remaining_runs.M2_FEATURES``,
  ``win_prob.M3_FEATURES``, ``outcome.M4_FEATURES``, ``extras.EXTRAS_FEATURES`` and
  ``skill.GROUPS``), and the hyperparameters from the Config.

The text cites no external source and holds no URL or DOI (the Verified_Source_List is empty),
and never uses the phrases that the Report_Check bans (Requirement 2, criteria 9 and 10).
Equations go through ``Section.add_equation`` (matplotlib mathtext), and every path in the text
uses forward slashes, so no paragraph holds LaTeX residue (Requirement 27, criterion 9).

Optional inputs
---------------
``value/nbv_cells_own_theta.csv``, the cell means of the NBV with each player's own theta
(Requirement 15, criterion 17), is written by the ``value`` stage of newer Runs
(OPTIONAL_OUTPUT_FILES). When the Run's outputs lack it, Section 2.1 says so in one sentence
instead of showing the table. With the Recent_Activity_Rule off in the Config, no
Activity_Comparison_Ranking exists, so its files (ACTIVITY_INPUTS) and ``activity_comparison.*``
constants become optional and Section 3.4 says that the comparison was not computed.

The robustness table ``robustness/robustness_table.csv`` (ROBUSTNESS_TABLE; Requirement 21,
criterion 6) is optional too: the ``robustness`` stage (task 25.2) writes it, with the five
repeatability tables of Requirement 20. Its rows fill the "tested" column of the Appendix
Design_Choice table (``design_choice_table``) and the ablation, sensitivity and cross-check tables
of Section 5.3; without the robustness outputs (``_ROBUSTNESS_INPUTS``) every Design_Choice row
states NO_CASE_TEXT and Section 5.3 says that the robustness stage has not run.

The Design_Choice table (Requirement 27, criterion 8)
-----------------------------------------------------
Table cells hold no Consts, so the Appendix shows the Design_Choices D1-D20 from a CSV that the
report code writes first: ``design_choice_table`` builds it from the Config, ``constants_meta``
and the ConstantRegistry labels, and ``write_design_choice_table`` writes it to
GENERATED_FILES["design_choices"] (``outputs/report/design_choices.csv``; an override of the
name ``design_choices`` redirects it, for trial builds). One row per Design_Choice, columns
DESIGN_CHOICE_COLUMNS:

* ``id``: the D-number; ``config_keys``: the ``design_choices.*`` keys with that D-number in
  ``constants_meta``, in Config order;
* ``choice``: the ``constants_meta`` description of the key (the text before its first ": "); for
  several keys, their shared head, with each key's detail in front of its value in ``default``;
* ``default``: each key's Config value as text; ``label``: ``ConstantRegistry.label_text`` of the
  keys ("hand-set (Dn)"), which labels a value whose first appearance is the table;
* ``tested``: NO_CASE_TEXT, or each alternative that the robustness table records for the
  Design_Choice, as "<setting>: batters <Spearman> and <top-10 overlap>; bowlers ...".

The robustness table rows that fill ``tested`` are those whose ``target_id`` is the D-number;
rows of the eight Stress_Pillar ablations (``target_id`` F19-F26) count as D5 alternatives
(Requirement 21, criterion 4). Each row needs ``target_id``, ``setting``, ``role``, ``spearman``
and ``overlap`` (design, Data Models ``robustness_table``); ``case`` groups the rows of one case
when present, a false ``comparable`` shows "not comparable", and a true ``is_default`` (an
optional column for a grid case at the Config default) leaves the row out.

For the Report_Check (task 23.4)
--------------------------------
``BuiltReport.specs`` lists every TableSpec in document order. The two Section 0 tables have the
table_ids of SECTION0_TABLES; each is built from its Role's Main_List file (OUTPUT_FILES
``top10_batter`` and ``top10_bowler``) with key ``rank`` and the ``player_name`` column shown as
CSV text, for the rank and name check of Requirement 27, criterion 14. The Appendix tables are
built from ``audit/dismissal_classification.csv``, ``audit/pass_rates.csv``,
``audit/exclusion_counts.csv`` and the generated Design_Choice CSV, which stays in place after
the build so the check can read it again.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from superr.errors import StageError
from superr.report.builder import Bold, Const, Part, ReportBuilder, Section, format_constant_value
from superr.report.constants import UnknownConstantError
from superr.report.equations import latex_residue
from superr.report.tables import TableSpec

if TYPE_CHECKING:
    from superr.pipeline import StageContext
    from superr.report.constants import ConstantRegistry
    from superr.settings import Settings

__all__ = [
    "ACTIVITY_INPUTS",
    "AUDIT_HEADINGS",
    "DEFINITION_HEADINGS",
    "DESIGN_CHOICE_COLUMNS",
    "DOC_FILES",
    "GENERATED_FILES",
    "NO_CASE_TEXT",
    "OPTIONAL_OUTPUT_FILES",
    "OUTPUT_FILES",
    "REQUIRED_CONSTANTS",
    "ROBUSTNESS_HEADING",
    "ROBUSTNESS_TABLE",
    "ROLES",
    "SECTION0_TABLES",
    "SECTION_WRITERS",
    "AuditSummary",
    "DefinitionBlock",
    "FactCheck",
    "ReportInputs",
    "appendix",
    "design_choice_ids",
    "design_choice_table",
    "parse_audit",
    "parse_definition",
    "section_0",
    "section_1",
    "section_2",
    "section_3",
    "section_4",
    "section_5",
    "write_design_choice_table",
    "write_sections",
]

#: The two Roles, in report order.
ROLES: tuple[str, ...] = ("batter", "bowler")
_PLURAL: Mapping[str, str] = MappingProxyType({"batter": "batters", "bowler": "bowlers"})
_TITLE: Mapping[str, str] = MappingProxyType({"batter": "Batters", "bowler": "Bowlers"})

#: table_id of each Role's Section 0 top-10 table (Requirement 27, criteria 1 and 14).
SECTION0_TABLES: Mapping[str, str] = MappingProxyType({"batter": "s0_top10_batters", "bowler": "s0_top10_bowlers"})


def _output_files() -> dict[str, str]:
    files = {
        # pressure (Requirements 11 and 13)
        "pressure_summary": "pressure/pressure_summary.csv",
        "mean_li": "pressure/mean_li_by_over.csv",
        "mean_li_chart": "pressure/mean_li_by_over.png",
        "missing_components": "pressure/missing_components.csv",
        # validation V1-V4 (Requirement 14)
        "v1": "validation/v1_pi_deciles.csv",
        "v1_chart": "validation/v1_pi_deciles.png",
        "v2": "validation/v2_big_moments.csv",
        "v2_chart": "validation/v2_big_moments.png",
        "v3_extras": "validation/v3_extras.csv",
        "v3_extras_free": "validation/v3_extras_free.csv",
        "v3_extras_chart": "validation/v3_extras.png",
        "v3_within": "validation/v3_within_player.csv",
        "v3_within_chart": "validation/v3_within_player.png",
        "v4_dead": "validation/v4_simple_rule_dead_share.csv",
        "v4_unflagged": "validation/v4_high_leverage_unflagged.csv",
        "v4_chart": "validation/v4_simple_rules.png",
        "signal_matrix": "validation/signal_model_matrix.csv",
        # models (Requirements 8-11 and 23)
        "m1_metrics": "models/m1/metrics.csv",
        "m2_metrics": "models/m2/metrics.csv",
        "m2_lb_calibration": "models/m2/lb_calibration.csv",
        "m2_par_bias": "models/m2/par_bias.csv",
        "omega_chart": "models/m2/omega_by_lb_left_wickets.png",
        "m3_metrics": "models/m3/metrics.csv",
        "m3_folds": "models/m3/fold_metrics.csv",
        "m3_break_gap": "models/m3/innings_break_gap.csv",
        "wp_curve_chart": "models/m3/wp_curve_40767.png",
        "m4_log_loss": "models/m4/log_loss.csv",
        "m4_expected_runs": "models/m4/expected_runs_by_phase_wickets.csv",
        "extras_log_loss": "models/extras/log_loss.csv",
        "extras_rates": "models/extras/rates_by_cell.csv",
        "model_comparison": "models/model_choice/model_comparison.csv",
        "m3_curve_chart_1": "models/model_choice/m3_calibration_curve_inn1.png",
        "m3_curve_chart_2": "models/model_choice/m3_calibration_curve_inn2.png",
        # value, rating, folds and audit (Requirements 4, 5, 7, 15-18)
        "nbv_cells": "value/nbv_cells.csv",
        "fallback_counts": "value/fallback_counts.csv",
        "role_summary": "rating/role_summary.csv",
        "eligibility_grid": "rating/eligibility_grid.csv",
        "eligibility_report": "rating/eligibility_report.csv",
        "fold_table": "folds/fold_table.csv",
        "exclusion_counts": "audit/exclusion_counts.csv",
        "dismissal_codes": "audit/dismissal_codes.csv",
        # audit and score reconstruction for Section 5 and the Appendix (Requirements 4 and 5)
        "audit": "audit/audit.md",
        "pass_rates": "audit/pass_rates.csv",
        "dismissal_classification": "audit/dismissal_classification.csv",
        "bowl_speed": "audit/bowl_speed_summary.csv",
        "extras1_runs": "audit/extras_type_x_extras1_runs.csv",
        "extras_2_runs": "audit/extras_type_x_extras_2_runs.csv",
        "names_with_several_ids": "audit/names_with_several_ids.csv",
        "ids_with_several_names": "audit/ids_with_several_names.csv",
        "over_sizes": "audit/over_sizes.csv",
        "result_counts": "audit/result_counts.csv",
        # bootstrap (Requirement 19)
        "bootstrap_warnings": "robustness/bootstrap_warnings.csv",
    }
    for role in ROLES:
        plural = _PLURAL[role]
        files.update(
            {
                # final lists, evidence tables and charts (Requirements 25 and 26)
                f"top10_{role}": f"final/top10_{plural}.csv",
                f"traditional_{role}": f"final/traditional_stats_{plural}.csv",
                f"evidence_{role}": f"final/rating_evidence_{plural}.csv",
                f"mix_{role}": f"final/pressure_mix_{plural}.csv",
                f"forest_chart_{role}": f"final/interval_forest_{plural}.png",
                f"clutch_chart_{role}": f"final/clutch_vs_usual_{plural}.png",
                f"phase_chart_{role}": f"final/phase_mix_{plural}.png",
                f"competition_chart_{role}": f"final/competition_type_mix_{plural}.png",
                f"activity_{role}": f"final/comparison/activity_comparison_{plural}.csv",
                f"comparison_list_{role}": f"final/comparison/top10_{plural}_activity_off.csv",
                # pressure shares, player estimates and bootstrap summaries
                f"share_{role}": f"pressure/pressure_share_{role}.csv",
                f"estimates_{role}": f"rating/player_estimates_{role}.csv",
                f"bootstrap_{role}": f"robustness/bootstrap_{role}.csv",
            }
        )
    return files


#: Every output file that Sections 0-5 and the Appendix read: name -> path relative to the outputs folder.
OUTPUT_FILES: Mapping[str, str] = MappingProxyType(_output_files())

#: The robustness table of Requirement 21, criterion 6 (design, Data Models ``robustness_table``),
#: relative to the outputs folder; the ``robustness`` stage (task 25.2) writes it.
ROBUSTNESS_TABLE = "robustness/robustness_table.csv"

#: Output files that a Run may not hold yet; the section that uses one says so when it is missing.
#: The ``robustness`` stage (task 25.2) writes ROBUSTNESS_TABLE and the five repeatability tables of
#: Requirement 20 that Section 5.3 shows (split-half reliability, the Early-to-Late Spearman, the two
#: V1 tables over IPL and over T20I matches and the IPL-to-T20I Spearman); the ``value`` stage writes
#: ``value/m4_tiebreak_cells.csv`` only when the M4 calibration-aware tie-break displaced a candidate
#: (Requirement 23, criterion 14), which Section 2.3 then explains.
OPTIONAL_OUTPUT_FILES: Mapping[str, str] = MappingProxyType(
    {
        "nbv_cells_own_theta": "value/nbv_cells_own_theta.csv",
        "robustness_table": ROBUSTNESS_TABLE,
        "split_half_reliability": "robustness/split_half_reliability.csv",
        "period_spearman": "robustness/period_spearman.csv",
        "v1_ipl": "robustness/v1_ipl.csv",
        "v1_t20i": "robustness/v1_t20i.csv",
        "competition_spearman": "robustness/competition_spearman.csv",
        "m4_tiebreak_cells": "value/m4_tiebreak_cells.csv",
    }
)

#: Files that the report code writes from the Config and the constants before it shows them as a
#: table: name -> path relative to the outputs folder (module docstring, "The Design_Choice table").
GENERATED_FILES: Mapping[str, str] = MappingProxyType({"design_choices": "report/design_choices.csv"})

#: Files under the docs folder: name -> path relative to it.
DOC_FILES: Mapping[str, str] = MappingProxyType({"pressure_definition": "pressure_definition.md"})

#: The stage that writes the files under each prefix (relative to the outputs folder), first match
#: first, for the error message of a missing file.
_WRITERS: tuple[tuple[str, str], ...] = (
    ("final/", "final_lists"),
    ("pressure/", "pressure"),
    ("validation/", "validation"),
    ("models/m1/", "skill"),
    ("models/m2/", "remaining_runs"),
    ("models/m3/wp_curve", "model_choice"),
    ("models/m3/", "win_prob"),
    ("models/m4/", "outcome"),
    ("models/extras/", "outcome"),
    ("models/model_choice/", "model_choice"),
    ("value/", "value"),
    ("rating/", "rating"),
    ("folds/", "folds"),
    ("robustness/bootstrap", "bootstrap"),
    ("robustness/", "robustness"),
    ("audit/exclusion_counts", "reconstruct"),
    ("audit/pass_rates", "reconstruct"),
    ("audit/", "audit"),
    ("pressure_definition.md", "pressure"),
)

#: The data-derived constants that the text quotes, recorded by the stages before ``report``.
REQUIRED_CONSTANTS: tuple[str, ...] = (
    "reconstruct.selected_encoding",
    "skill.penalty_runs_by_fit",
    "skill.penalty_wkt_by_fit",
    "skill.typical_top7_sum_by_fit",
    "remaining_runs.outer_best_iteration_by_fit",
    "win_prob.outer_lgbm_best_iteration_by_fit",
    "outcome.outer_lgbm_best_iteration_by_fit",
    "outcome.average_skill_by_fold",
    "model_choice.primary_models",
    "pressure.pooled_mean_unscaled_li",
    "pressure.lowest_pressure_ball_pi",
    "pressure.pressure_ball_pct_of_live_balls",
    "validation.lowest_high_leverage_li",
    "value.dismissal_charge_counts",
    "value.nbv_cell_means",
    "value.retire_out_uncharged",
    "bootstrap.n_resampled_matches",
    *(
        f"{stage}.{name}.{role}"
        for stage, names in (
            (
                "rating",
                (
                    "min_pressure_balls", "n_eligible", "replacement_level", "tau2_usual", "tau2_gap",
                    "tau2_uplift", "rho_L", "rho_U", "w_L", "w_U",
                ),
            ),
            ("activity_comparison", ("n_eligible",)),
            (
                "bootstrap",
                ("n_ever_unranked", "top10_set_reproduced", "min_p_top10_main_top10", "max_p_top10_outside_top10"),
            ),
            ("final_lists", ("n_unresolved",)),
        )
        for name in names
        for role in ROLES
    ),
)  # fmt: skip


#: The inputs and constants of the Activity_Comparison_Ranking, which a Run has only while the
#: Recent_Activity_Rule is on (Requirement 25, criteria 11-14); with the rule off they are optional.
ACTIVITY_INPUTS: tuple[str, ...] = tuple(
    name for role in ROLES for name in (f"activity_{role}", f"comparison_list_{role}")
)
_ACTIVITY_CONSTANT_PREFIX = "activity_comparison."


def _writer(relative: str) -> str:
    for prefix, stage in _WRITERS:
        if relative.startswith(prefix):
            return stage
    return "an earlier"


# ----------------------------------------------------------------- the pressure definition

#: The sections of ``docs/pressure_definition.md`` that Section 1 copies (Requirement 27, criterion 2).
DEFINITION_HEADINGS: tuple[str, str] = ("Plain-language definition", "Formal definitions")

_NUMBERED = re.compile(r"(\d+)\.\s+(.*)")
_BULLET = re.compile(r"[-*]\s+(.*)")
_SEPARATOR_CELL = re.compile(r":?-{3,}:?")
_INLINE = re.compile(r"\*\*(.+?)\*\*|`([^`]+)`")


@dataclass(frozen=True)
class DefinitionBlock:
    """One block of a section of the pressure definition file.

    kind: ``paragraph``, ``numbered`` (a numbered list item), ``bullet`` or ``table``. text: the
    Markdown text of a paragraph or list item (``**bold**`` and code spans are kept for
    rendering). number: the item number of a numbered item, as written. rows: a table's header
    row and then its data rows, each a tuple of cell texts.
    """

    kind: str
    text: str = ""
    number: str = ""
    rows: tuple[tuple[str, ...], ...] = ()


def parse_definition(text: str, source: Path | str = "docs/pressure_definition.md") -> dict[str, tuple[DefinitionBlock, ...]]:
    """The blocks of each DEFINITION_HEADINGS section of the pressure definition Markdown.

    A section runs from its ``## `` heading to the next heading. Blocks are separated by blank
    lines: a block of ``|`` lines is a table (separator rows dropped), a block starting ``1. ``
    holds numbered items, one starting ``- `` bullets, and any other block is one paragraph.
    StageError, naming ``source``, for a missing or empty section, a table whose rows have
    different cell counts, and text that holds LaTeX residue.
    """
    lines_of: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            current = stripped[3:].strip()
            lines_of[current] = []
        elif stripped.startswith("# "):
            current = None
        elif current is not None:
            lines_of[current].append(stripped)
    result: dict[str, tuple[DefinitionBlock, ...]] = {}
    for heading in DEFINITION_HEADINGS:
        if heading not in lines_of:
            raise StageError(
                f"the pressure definition {source} has no '## {heading}' section; the pressure stage writes it "
                "(Requirement 13, criterion 9)"
            )
        blocks = _definition_blocks(lines_of[heading], source, heading)
        if not blocks:
            raise StageError(f"the '## {heading}' section of the pressure definition {source} is empty")
        result[heading] = blocks
    return result


def _definition_blocks(
    lines: list[str], source: Path | str, heading: str, document: str = "the pressure definition"
) -> tuple[DefinitionBlock, ...]:
    """The blocks of one Markdown section (``parse_definition``); ``document`` names the file in errors."""
    chunks: list[list[str]] = [[]]
    for line in lines:
        if line:
            chunks[-1].append(line)
        elif chunks[-1]:
            chunks.append([])
    blocks: list[DefinitionBlock] = []
    for chunk in (chunk for chunk in chunks if chunk):
        if all(line.startswith("|") for line in chunk):
            rows = [tuple(cell.strip() for cell in line.strip().strip("|").split("|")) for line in chunk]
            rows = [row for row in rows if not all(_SEPARATOR_CELL.fullmatch(cell) for cell in row)]
            if len({len(row) for row in rows}) != 1 or len(rows) < 2:
                raise StageError(
                    f"a table in the '## {heading}' section of {document} {source} needs a header row "
                    "and data rows with the same number of cells"
                )
            blocks.append(DefinitionBlock("table", rows=tuple(rows)))
            continue
        pattern = _NUMBERED if _NUMBERED.fullmatch(chunk[0]) else _BULLET if _BULLET.fullmatch(chunk[0]) else None
        if pattern is None:
            blocks.append(DefinitionBlock("paragraph", " ".join(chunk)))
            continue
        kind = "numbered" if pattern is _NUMBERED else "bullet"
        items: list[list[str]] = []
        for line in chunk:
            match = pattern.fullmatch(line)
            if match is not None:
                items.append([match.group(1), match.group(match.lastindex or 1)])
            else:
                items[-1][1] += " " + line
        for number, item in items:
            blocks.append(DefinitionBlock(kind, item, number if kind == "numbered" else ""))
    for block in blocks:
        texts = [block.text, *(cell for row in block.rows for cell in row)]
        found = [hit for piece in texts for hit in latex_residue(piece)]
        if found:
            raise StageError(
                f"the '## {heading}' section of {document} {source} holds LaTeX residue {found}, which "
                "the Report_Check rejects (Requirement 27, criterion 9)"
            )
    return tuple(blocks)


def _inline(text: str) -> list[Part]:
    """Paragraph parts of Markdown inline text: ``**bold**`` becomes Bold, a code span plain text."""
    parts: list[Part] = []
    position = 0
    for match in _INLINE.finditer(text):
        if match.start() > position:
            parts.append(text[position : match.start()])
        parts.append(Bold(match.group(1)) if match.group(1) is not None else match.group(2))
        position = match.end()
    if position < len(text):
        parts.append(text[position:])
    return parts


def _plain_markdown(text: str) -> str:
    """Markdown inline text without its markup: bold and code spans become plain text."""
    return "".join(part.text if isinstance(part, Bold) else str(part) for part in _inline(text)).strip()


# --------------------------------------------------------------------- the audit summary

#: The sections of ``outputs/audit/audit.md`` that the Appendix summarises, by the start of their
#: ``## `` heading (Requirement 4, criterion 16).
AUDIT_HEADINGS: tuple[str, str] = ("Fact_Checks", "Tables saved")


@dataclass(frozen=True)
class FactCheck:
    """One Fact_Check of ``audit.md`` (Requirement 4, criteria 12-14), each field as Markdown text."""

    name: str
    check: str
    observed: str
    result: str

    @property
    def passed(self) -> bool:
        return _plain_markdown(self.result).lower() == "pass"


@dataclass(frozen=True)
class AuditSummary:
    """What the Appendix takes from ``audit.md``: the Fact_Checks and the saved tables' file names."""

    fact_checks: tuple[FactCheck, ...]
    saved_tables: tuple[str, ...]


def parse_audit(text: str, source: Path | str = "outputs/audit/audit.md") -> AuditSummary:
    """The Fact_Checks and saved-table names of the Auditor's ``audit.md`` (AUDIT_HEADINGS).

    Each section's first Markdown table is read: the Fact_Checks table by its columns
    ``Fact_Check``, ``Check``, ``Observed value`` and ``Result``, the saved-tables table by its
    ``File`` column. StageError, naming ``source``, for a missing section, table or column, an empty
    table, and LaTeX residue.
    """
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            current = stripped[3:].strip()
            sections[current] = []
        elif stripped.startswith("# "):
            current = None
        elif current is not None:
            sections[current].append(stripped)
    tables: dict[str, tuple[tuple[str, ...], ...]] = {}
    for prefix in AUDIT_HEADINGS:
        heading = next((name for name in sections if name.startswith(prefix)), None)
        if heading is None:
            raise StageError(
                f"the audit summary {source} has no '## {prefix} ...' section; the audit stage writes it "
                "(Requirement 4, criterion 16)"
            )
        found = [block for block in _definition_blocks(sections[heading], source, heading, "the audit summary") if block.kind == "table"]
        if not found:
            raise StageError(f"the '## {heading}' section of the audit summary {source} holds no table")
        tables[prefix] = found[0].rows

    def columns(rows: tuple[tuple[str, ...], ...], names: Sequence[str], what: str) -> list[int]:
        header = [_plain_markdown(cell).lower() for cell in rows[0]]
        missing = [name for name in names if name.lower() not in header]
        if missing:
            raise StageError(f"the {what} table of the audit summary {source} has no column(s) {missing}")
        return [header.index(name.lower()) for name in names]

    name, check, observed, result = columns(tables["Fact_Checks"], ("Fact_Check", "Check", "Observed value", "Result"), "Fact_Check")
    (file,) = columns(tables["Tables saved"], ("File",), "saved-tables")
    return AuditSummary(
        fact_checks=tuple(FactCheck(row[name], row[check], row[observed], row[result]) for row in tables["Fact_Checks"][1:]),
        saved_tables=tuple(_plain_markdown(row[file]) for row in tables["Tables saved"][1:]),
    )


# ---------------------------------------------------------------------------- the inputs

@dataclass(frozen=True)
class ReportInputs:
    """Everything that Sections 0-5 and the Appendix read, loaded once by ``load``.

    settings: the Run's Config. constants: the Run's ConstantRegistry with every stage's
    data-derived constants. paths: input name (OUTPUT_FILES, the OPTIONAL_OUTPUT_FILES present,
    DOC_FILES) -> absolute path, plus the path that each GENERATED_FILES name is written to.
    frames: every CSV input read with pandas (not the generated files). definition: the blocks of
    each DEFINITION_HEADINGS section of the pressure definition.
    """

    settings: Settings
    constants: ConstantRegistry
    paths: Mapping[str, Path]
    frames: Mapping[str, pd.DataFrame]
    definition: Mapping[str, tuple[DefinitionBlock, ...]]

    @classmethod
    def from_context(cls, ctx: StageContext) -> ReportInputs:
        """The inputs of the ``report`` stage: the Run's Config, constants and output files."""
        return cls.load(ctx.settings, ctx.constants)

    @classmethod
    def load(
        cls,
        settings: Settings,
        constants: ConstantRegistry,
        *,
        overrides: Mapping[str, Path | str] | None = None,
    ) -> ReportInputs:
        """Read every input file and check the constants (module docstring).

        overrides: input name -> absolute path read in place of the Run's file, for tests and
        trial builds; a GENERATED_FILES name sets where that file is written instead. StageError
        lists every missing file with the stage that writes it, every unreadable CSV file and
        every missing data-derived constant; ValueError for an unknown override name or a
        relative override path.
        """
        outputs, docs = settings.path("outputs"), settings.path("docs")
        rule_on = bool(settings.get("data_rules.recent_activity_rule"))
        required = {name: outputs / relative for name, relative in OUTPUT_FILES.items()}
        required.update({name: docs / relative for name, relative in DOC_FILES.items()})
        optional = {name: outputs / relative for name, relative in OPTIONAL_OUTPUT_FILES.items()}
        generated = {name: outputs / relative for name, relative in GENERATED_FILES.items()}
        if not rule_on:  # no Activity_Comparison_Ranking: its files are optional
            optional.update({name: required.pop(name) for name in ACTIVITY_INPUTS})
        writers = {name: _writer(relative) for name, relative in {**OUTPUT_FILES, **DOC_FILES}.items()}
        for name, path in (overrides or {}).items():
            target = Path(path)
            group = next((files for files in (required, optional, generated) if name in files), None)
            if group is None:
                raise ValueError(f"no report input is named {name!r}")
            if not target.is_absolute():
                raise ValueError(f"the override of the report input {name!r} must be an absolute path, got {str(path)!r}")
            group[name] = target
        missing = [f"{path} (written by the {writers[name]} stage)" for name, path in required.items() if not path.is_file()]
        if missing:
            raise StageError(
                f"the report needs {len(missing)} output file(s) that do not exist; run the stages that write them: "
                + "; ".join(missing)
            )
        paths = {**required, **{name: path for name, path in optional.items() if path.is_file()}}
        frames: dict[str, pd.DataFrame] = {}
        for name, path in paths.items():
            if path.suffix.lower() == ".csv":
                try:
                    frames[name] = pd.read_csv(path)
                except (OSError, UnicodeDecodeError, ValueError) as exc:
                    raise StageError(f"cannot read the report input {path}: {type(exc).__name__}: {exc}") from None
        absent = [
            name
            for name in REQUIRED_CONSTANTS
            if name not in constants and (rule_on or not name.startswith(_ACTIVITY_CONSTANT_PREFIX))
        ]
        if absent:
            raise StageError(
                f"the report quotes {len(absent)} data-derived constant(s) that no stage recorded: {', '.join(absent)}; "
                "load every stage's constants (load_run_constants) or rerun the stages that record them"
            )
        definition_file = paths["pressure_definition"]
        try:
            text = definition_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise StageError(f"cannot read the pressure definition {definition_file}: {exc}") from None
        definition = parse_definition(text, definition_file)
        return cls(
            settings=settings,
            constants=constants,
            paths=MappingProxyType({**paths, **generated}),
            frames=MappingProxyType(frames),
            definition=MappingProxyType(definition),
        )

    def has(self, name: str) -> bool:
        """True when the input is present (always for OUTPUT_FILES, DOC_FILES and GENERATED_FILES)."""
        return name in self.paths

    def path(self, name: str) -> Path:
        """Absolute path of an input file, or where a GENERATED_FILES file is written."""
        try:
            return self.paths[name]
        except KeyError:
            raise StageError(f"the report input {name!r} is not loaded") from None

    def text(self, name: str) -> str:
        """The UTF-8 text of an input file that is not a CSV file, such as ``audit/audit.md``."""
        path = self.path(name)
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise StageError(f"cannot read the report input {path}: {exc}") from None

    def frame(self, name: str) -> pd.DataFrame:
        """A copy of a CSV input as read by pandas."""
        try:
            return self.frames[name].copy()
        except KeyError:
            raise StageError(f"the report input {name!r} is not a loaded CSV file") from None

    def value(self, name: str) -> Any:
        """The value of a Method_Constant: recorded by a stage (data-derived) or read from the Config."""
        try:
            return self.constants.value(name)
        except UnknownConstantError:
            if name not in self.constants:
                raise
            return self.settings.get(name)


# --------------------------------------------------------------------------- formatting

def _fixed(value: Any, decimals: int) -> str:
    """A number with fixed decimals; a result that rounds to 0 has no sign."""
    text = f"{float(value):.{decimals}f}"
    return text.lstrip("-") if float(text) == 0.0 else text


def _signed(value: Any, decimals: int) -> str:
    """A number with fixed decimals and an explicit sign: "+1.40", "-8.75", "0.00"."""
    text = _fixed(value, decimals)
    return text if text.startswith("-") or float(text) == 0.0 else f"+{text}"


def _pct(share: Any, decimals: int = 1) -> str:
    """A share in [0, 1] as a percentage: 0.4026 -> "40.3%"."""
    return f"{_fixed(100.0 * float(share), decimals)}%"


def _count(value: Any) -> str:
    """A whole number with thousands separators: 189648 -> "189,648"."""
    return f"{int(round(float(value))):,}"


def _span(values: Iterable[Any], decimals: int, joiner: str = " to ") -> str:
    """The smallest and largest of ``values`` with fixed decimals, or one value when they agree."""
    numbers = [float(value) for value in values]
    low, high = _fixed(min(numbers), decimals), _fixed(max(numbers), decimals)
    return low if low == high else f"{low}{joiner}{high}"


def _and(items: Iterable[Any]) -> str:
    """Items joined as "a, b and c"."""
    texts = [str(item) for item in items]
    if len(texts) <= 1:
        return "".join(texts)
    return ", ".join(texts[:-1]) + " and " + texts[-1]


def _ranks(ranks: Sequence[int]) -> str:
    """Ranks as words: "7 to 10" when they run on, else "5, 7 and 9"."""
    ordered = sorted(int(rank) for rank in ranks)
    if len(ordered) > 2 and ordered == list(range(ordered[0], ordered[-1] + 1)):
        return f"{ordered[0]} to {ordered[-1]}"
    return _and(ordered)


def _shown(value: Any) -> str:
    """A Config number as the Config writes it (0.05, 1.25, 300), through the builder's formatting."""
    return format_constant_value(value)


def _const(inputs: ReportInputs, name: str, text: str | Callable[[Any], str] | None = None) -> Const:
    """A Method_Constant part; ``text`` is the shown text, or a function of the constant's value."""
    if callable(text):
        return Const(name, text(inputs.value(name)))
    return Const(name, text)


def _row(frame: pd.DataFrame, what: str, **match: Any) -> pd.Series:
    """The one row of ``frame`` whose columns hold the given values (compared as text)."""
    mask = np.ones(len(frame), dtype=bool)
    for column, value in match.items():
        mask &= (frame[column].astype(str) == str(value)).to_numpy()
    rows = frame[mask]
    if len(rows) != 1:
        raise StageError(f"the report input {what} has {len(rows)} row(s) with {match}, not 1")
    return rows.iloc[0]


#: Plain names of the Candidate_Model families of the model-comparison table.
_FAMILY_LABELS: Mapping[str, str] = MappingProxyType(
    {"logistic": "logistic baseline", "lgbm": "LightGBM", "xgb": "XGBoost", "rf": "random forest"}
)


def _model_label(family: str) -> str:
    return _FAMILY_LABELS.get(str(family), str(family))


_SCOPE_TEXT: Mapping[str, str] = MappingProxyType(
    {"innings_group": "within each Innings_Group", "pooled": "pooled over both Innings_Groups"}
)
_CELL_TEXT: Mapping[str, str] = MappingProxyType(
    {"innings_phase": "innings × Phase", "innings_phase_comp_type": "innings × Phase × Competition_Type"}
)

#: Plain names of the model features (module constants of the model modules); a feature not
#: listed is shown by its column name.
_FEATURE_NAMES: Mapping[str, str] = MappingProxyType(
    {
        "lb_left": "Legal_Balls left",
        "lb_bowled": "Legal_Balls bowled",
        "wickets": "wickets fallen",
        "rbq": "RBQ",
        "theta_striker": "striker θ_bat",
        "striker_lb_faced": "striker Legal_Balls faced",
        "theta_non_striker": "non-striker θ_bat",
        "theta_bowler": "bowler θ_bowl",
        "scoring_env": "Scoring_Environment",
        "venue_effect": "Venue_Effect",
        "match_day": "Match_Day_Conditions",
        "score": "team score",
        "par_score": "Par_Score",
        "tsg": "Team_Strength_Gap",
        "sched_balls": "Scheduled_Balls",
        "runs_needed": "runs needed",
        "rr_minus_ar": "Required_Rate minus Achievable_Rate",
        "innings": "innings",
        "phase": "Phase",
        "f19_component": "the F19 scoreboard-deficit component",
        "hand_x_style": "batting hand × the bowler's Modal_Bowling_Type",
        "pace_or_spin": "pace or spin",
        "free_hit": "Free_Hit flag",
        "cell": "innings × Phase cell",
        "batting_type": "the striker's batting hand",
    }
)

#: Plain names of the M1 indicator groups (``skill.GROUPS``).
_GROUP_NAMES: Mapping[str, str] = MappingProxyType(
    {
        "striker": "the striker",
        "bowler": "the bowler",
        "environment": "the Competition_Type × Season combination",
        "venue": "the venue",
        "country": "the venue country",
        "cell": "the innings × Phase cell",
        "wickets_bucket": "the Wickets_Bucket",
    }
)


def _feature(name: str) -> str:
    return _FEATURE_NAMES.get(name, name.replace("_", " "))


def _features(names: Iterable[str]) -> str:
    return _and(_feature(name) for name in names)


def _monotone(features: Sequence[str], signs: Sequence[int]) -> str:
    """"non-decreasing in a and b, non-increasing in c" from a monotone-constraint tuple."""
    up = [_feature(name) for name, sign in zip(features, signs) if sign > 0]
    down = [_feature(name) for name, sign in zip(features, signs) if sign < 0]
    parts = [f"non-decreasing in {_and(up)}"] if up else []
    parts += [f"non-increasing in {_and(down)}"] if down else []
    return " and ".join(parts)


def _model_modules() -> Any:
    """The model modules whose feature constants the text lists (imported when first needed)."""
    from superr import skill
    from superr.models import extras, outcome, remaining_runs, win_prob

    return skill, remaining_runs, win_prob, outcome, extras


def _rule_label(method: str) -> str:
    """Plain name of a Simple_Rule from its method code in the validation tables."""
    overs = re.fullmatch(r"overs_(\d+)_(\d+)", method)
    if overs:
        return f"overs {overs.group(1)} to {overs.group(2)}"
    chase = re.fullmatch(r"chase_rrr_above_(\d+)", method)
    if chase:
        return f"second-innings required rate above {chase.group(1)}"
    return method.replace("_", " ")


# ------------------------------------------------------------------- tables and figures

#: Display headers of columns that several tables show.
_HEADERS: Mapping[str, str] = MappingProxyType(
    {
        "rank": "Rank",
        "player_id": "player_id",
        "player_name": "Player",
        "role": "Role",
        "innings": "Innings",
        "phase": "Phase",
        "comp_type": "Competition_Type",
        "final_score": "Final_Score",
        "ci_low": "95% low",
        "ci_high": "95% high",
        "p_top10": "P_Top10",
        "hit_rate": "Hit_Rate",
        "n_deliveries": "Deliveries",
        "n_live": "Live_Balls",
        "n_pressure": "Pressure_Balls",
        "n_rows": "Rows",
        "n_legal_balls": "Legal_Balls",
        "metric": "Metric",
        "value": "Value",
        "model": "Model",
        "family": "Family",
        "variant": "Variant",
        "log_loss": "Log loss",
    }
)


def _table(
    section: Section,
    inputs: ReportInputs,
    table_id: str,
    source: str,
    columns: Sequence[str],
    *,
    key: Sequence[str],
    caption: str,
    decimals: Mapping[str, int] | None = None,
    headers: Mapping[str, str] | None = None,
    where: Mapping[str, str | Sequence[str]] | None = None,
) -> str:
    """Add a TableSpec table of the input ``source``; return its reference ("Table 1.2")."""
    shown = {name: _HEADERS[name] for name in columns if name in _HEADERS}
    shown.update(headers or {})
    spec = TableSpec(
        table_id,
        inputs.path(source),
        list(columns),
        dict(decimals or {}),
        list(key),
        caption=caption,
        headers=shown,
        where=dict(where or {}),
    )
    return section.add_table(spec)


def _figure(section: Section, inputs: ReportInputs, source: str, caption: str, width_cm: float | None = None) -> str:
    """Add a chart of the input ``source``; return its reference ("Figure 1.2")."""
    return section.add_image(inputs.path(source), caption, width_cm=width_cm)


#: Shown width of the charts, in cm: the wide multi-panel charts use the text width (None).
_NARROW_CM = 12.0
_MEDIUM_CM = 14.0


# ------------------------------------------------------------------- shared Method_Constants

def _band(inputs: ReportInputs) -> Const:
    """D1, the live band on WP: "0.05 and 0.95"."""
    return _const(inputs, "design_choices.D1_live_band", lambda band: f"{_shown(band[0])} and {_shown(band[1])}")


def _share(inputs: ReportInputs) -> Const:
    """D3, the pressure share: "25%"."""
    return _const(inputs, "design_choices.D3_pressure_share", lambda share: f"{_shown(100.0 * float(share))}%")


def _n_boot(inputs: ReportInputs) -> Const:
    """D16, the number of Bootstrap_Resamples: "2,000"."""
    return _const(inputs, "design_choices.D16_n_boot", _count)


def _numbers(value: Any) -> list[float]:
    """Every number of a data-derived value: a number, or a dict of numbers or of such dicts."""
    if isinstance(value, Mapping):
        return [number for item in value.values() for number in _numbers(item)]
    return [float(value)]


def _derived_span(inputs: ReportInputs, name: str, decimals: int) -> Const:
    """A data-derived constant whose value is a (nested) dict of numbers, shown as its range."""
    return _const(inputs, name, lambda value: _span(_numbers(value), decimals))


def _by_innings(inputs: ReportInputs, name: str, decimals: int) -> Const:
    """A data-derived constant {"1": x, "2": y} shown as "x in the first innings and y in the second"."""
    return _const(
        inputs, name,
        lambda value: f"{_fixed(value['1'], decimals)} in the first innings and {_fixed(value['2'], decimals)} in the second",
    )  # fmt: skip


def _gap_parts(inputs: ReportInputs, role: str) -> list[Part]:
    """What the Pressure_Gap and Uplift variances of a Role mean for its Final_Score (Requirement 18; DD-16)."""
    summary = _row(inputs.frame("role_summary"), "role_summary.csv", role=role)
    gap_zero = float(inputs.value(f"rating.tau2_gap.{role}")) == 0.0
    uplift_zero = float(inputs.value(f"rating.tau2_uplift.{role}")) == 0.0 or float(inputs.value(f"rating.w_U.{role}")) == 0.0
    plural = _PLURAL[role]
    if gap_zero:
        parts: list[Part] = [
            f"The {plural} show no measurable between-player spread in how their value changes under pressure: the "
            f"estimated between-player variance τ² of the {role} Pressure_Gap is {_fixed(summary['tau2_gap_raw'], 3)}, "
            f"below 0, so it is set to 0, every {role}'s Pressure_Gap is shrunk to 0 and his shrunk Clutch_Level "
            "equals his shrunk usual level. "
        ]
        parts.append(
            f"The {plural} are therefore ranked by their shrunk usual level on ordinary balls matched to their pressure "
            "situations."
            if uplift_zero
            else f"The {role} Final_Score therefore rests on the shrunk usual level and the shrunk Uplift."
        )
        return parts
    return [
        f"The {plural} keep part of their pressure gap: the between-player variance τ² of the {role} Pressure_Gap is ",
        _const(inputs, f"rating.tau2_gap.{role}", lambda value: _fixed(value, 3)),
        ", so each shrunk Clutch_Level is the shrunk usual level plus part of the player's own Pressure_Gap.",
    ]


# ------------------------------------------------------------------------------ Section 0

def section_0(builder: ReportBuilder, inputs: ReportInputs) -> None:
    """Section 0 Executive summary: the method in brief, both top-10 tables and the main findings (27.1)."""
    s = builder.section(0)
    s.add_paragraph(
        "This report explains how the two top-10 lists below were built and shows the evidence behind each name. "
        "Every number comes from the six supplied CSV files and the settings in config.py; the command "
        "python run_pipeline.py rebuilds every table, chart and number from them."
    )
    s.add_paragraph(
        "A ball is a pressure ball for a player when the match is still live (the batting side's win probability is "
        "between ", _band(inputs), ") and the ball's Pressure_Index is in the top ", _share(inputs),
        " of live balls for the player's Role and innings. The Pressure_Index blends leverage, how far one ball can "
        "move the win probability, with the stress on the batter or the bowler, with weight α = ",
        _const(inputs, "design_choices.D2_alpha"), " on leverage. Every ball is valued in runs against what an "
        "average player would have produced in the same match state: its net ball value (NBV). A player's rating "
        "compares his NBV on pressure balls with his own usual level on ordinary balls in the same innings and "
        "phase, shrinks both for sample size, and blends them by how well each repeats from match to match.",
    )  # fmt: skip
    s.add_paragraph(
        "Final_Score is in runs per over against an average player. The 95% interval and P_Top10, the share of ",
        _n_boot(inputs), " bootstrap resamples of whole matches in which the player ranks in the top 10, show how "
        "secure each place is (Section 2.4).",
    )  # fmt: skip
    for role in ROLES:
        _table(
            s, inputs, SECTION0_TABLES[role], f"top10_{role}",
            ["rank", "player_name", "final_score", "ci_low", "ci_high", "p_top10"],
            key=["rank"],
            decimals={"final_score": 3, "ci_low": 3, "ci_high": 3, "p_top10": 4},
            caption=f"Top 10 {_PLURAL[role]} under pressure: Final_Score in runs per over, its 95% bootstrap "
            "interval and P_Top10",
        )  # fmt: skip
    s.add_heading("Main findings")
    for parts in _findings(inputs):
        s.add_bullet(*parts)


def _findings(inputs: ReportInputs) -> list[list[Part]]:
    v1 = inputs.frame("v1")
    ratios, rising = [], True
    for _, group in v1[v1["group"] == "decile"].groupby(["role", "innings"], sort=True):
        swings = group.sort_values("decile")["mean_swing"].to_numpy(dtype=float)
        ratios.append(swings[-1] / swings[0])
        rising &= bool(np.all(np.diff(swings) >= 0.0))
    v2 = inputs.frame("v2")
    ours = {role: _row(v2, "v2_big_moments.csv", method=f"{role}_pressure_balls") for role in ROLES}
    rules = v2[~v2["method"].astype(str).str.endswith("_pressure_balls")]
    findings: list[list[Part]] = [
        [
            Bold("Pressure balls follow the match. "),
            "In held-out matches the mean win-probability swing of a ball "
            + ("rises with every Pressure_Index decile" if rising else "rises from the lowest to the highest "
               "Pressure_Index decile")
            + f"; the top decile's swing is {_span(ratios, 1)} times the bottom decile's in each Role and innings "
            "(Section 1.5).",
        ],
        [
            Bold("Pressure balls find the big moments. "),
            f"Batter Pressure_Balls are {_pct(ours['batter']['share_flagged'])} of held-out balls and hold "
            f"{_pct(ours['batter']['recall'])} of the Big_Moments, each match's largest win-probability swings; "
            f"bowler Pressure_Balls are {_pct(ours['bowler']['share_flagged'])} and hold "
            f"{_pct(ours['bowler']['recall'])}. "
            + " ".join(
                f"The rule '{_rule_label(str(rule['method']))}' flags {_pct(rule['share_flagged'])} and holds "
                f"{_pct(rule['recall'])}."
                for _, rule in rules.iterrows()
            ),
        ],
        [Bold("What the ratings rest on. "), *_gap_parts(inputs, "batter"), " ", *_gap_parts(inputs, "bowler"),
         " (Section 2.4.)"],
    ]  # fmt: skip
    for role in ROLES:
        top = inputs.frame(f"top10_{role}")
        close = top.loc[top["p_top10"] < 0.5, "rank"].astype(int).tolist()
        if close:
            values = top.loc[top["p_top10"] < 0.5, "p_top10"]
            findings.append(
                [
                    Bold(f"Close calls among the {_PLURAL[role]}. "),
                    f"P_Top10 is below one half at {role} rank{'s' if len(close) > 1 else ''} {_ranks(close)} "
                    f"({_span(values, 3)}), so these places could change with a different sample of matches.",
                ]
            )
    rule_on = bool(inputs.settings.get("data_rules.recent_activity_rule"))
    findings.append(
        [
            Bold("Eligibility. "),
            "The lists rank ", _const(inputs, "rating.n_eligible.batter"), " batters and ",
            _const(inputs, "rating.n_eligible.bowler"), " bowlers who meet every eligibility rule of Section 3.4, "
            + ("including the Recent_Activity_Rule, a confirmed assumption that the problem statement does not require."
               if rule_on else "with the Recent_Activity_Rule, a confirmed assumption, switched off."),
        ]
    )  # fmt: skip
    return findings


# ------------------------------------------------------------------------------ Section 1

def _definition_equations(inputs: ReportInputs) -> dict[str, tuple[str, str]]:
    """The equations placed after the formal definitions of LI, Stress and PI (bold lead -> mathtext, words)."""
    return {
        "Leverage_Index": (
            r"\mathrm{LI} = \frac{1}{m}\,\sum_{c} P(c)\;|\,\mathrm{WP}(s_c) - \mathrm{WP}(s)\,|",
            "Leverage_Index: the probability-weighted absolute change in win probability over the outcome classes, "
            "divided by its mean m",
        ),
        "Stress": (
            r"\mathrm{Stress}_{\mathrm{bat}} = \frac{1}{4}\,(F_{19}+F_{20}+F_{21}+F_{22}),\quad "
            r"\mathrm{Stress}_{\mathrm{bowl}} = \frac{1}{4}\,(F_{23}+F_{24}+F_{25}+F_{26})",
            "Stress: the equal-weighted mean of the four Stress_Pillars of each Role",
        ),
        "Pressure_Index": (
            r"\mathrm{PI} = 100\,[\,\alpha\, r(\mathrm{LI}) + (1-\alpha)\, r(\mathrm{Stress})\,]",
            "Pressure_Index: 100 times the alpha-weighted blend of the percentile ranks of LI and Stress",
        ),
    }


def _write_definition(section: Section, blocks: Sequence[DefinitionBlock], equations: Mapping[str, tuple[str, str]]) -> None:
    """Copy definition blocks into ``section``; each equation follows the paragraph whose bold lead starts with its key."""
    placed: set[str] = set()
    for block in blocks:
        if block.kind == "paragraph":
            parts = _inline(block.text)
            section.add_paragraph(*parts)
            lead = parts[0].text if parts and isinstance(parts[0], Bold) else ""
            for name, (mathtext, words) in equations.items():
                if name not in placed and lead.startswith(name):
                    section.add_equation(mathtext, words)
                    placed.add(name)
        elif block.kind == "numbered":
            section.add_paragraph(f"{block.number}. ", *_inline(block.text))
        elif block.kind == "bullet":
            section.add_bullet(*_inline(block.text))
        else:
            for cells in block.rows[1:]:
                first, *middle, last = cells
                head: list[Part] = [Bold(first)]
                head += [f" ({', '.join(middle)}): "] if middle else [": "]
                section.add_bullet(*head, *_inline(last))
    for name, (mathtext, words) in equations.items():
        if name not in placed:
            section.add_equation(mathtext, words)


#: Why the method uses LI and each Stress_Pillar (from the plan's reasoning, ``docs/plan.md``).
_REASONS: tuple[tuple[str, str], ...] = (
    (
        "Leverage (LI). ",
        "Required rate, wickets in hand and balls left all enter the win probability, so they drive leverage "
        "without hand-set weights. Leverage falls towards 0 once a match is decided and rises towards the end of a "
        "close innings, so it also covers the closeness of the contest and the stage of the innings; it replaced a "
        "hand-tuned phase curve and a fixed closeness gate. The batter and the bowler share a ball's leverage, "
        "because both face the same stakes.",
    ),
    (
        "F19 scoreboard deficit (batter). ",
        "A batter behind par in the first innings, or facing a required rate above what is normally achievable "
        "from the position in a chase, must take more risk to catch up.",
    ),
    (
        "F20 wickets and depth (batter). ",
        "More wickets down than is normal at this stage, wickets in the recent overs and little batting to come "
        "all raise the cost of a dismissal, so the batter must score while protecting his wicket.",
    ),
    (
        "F21 new at the crease (batter). ",
        "A batter who has faced few balls has not yet adjusted to the pitch and the bowling, so the same ball is "
        "harder for him.",
    ),
    (
        "F22 squeeze (batter). ",
        "A run of dot balls and a long wait since the team's last boundary build the pressure to break out, which "
        "is when batters take poor risks.",
    ),
    (
        "F23 scoreboard threat (bowler). ",
        "When the batting side scores faster than par in the first innings, or is ahead of the rate in a chase, "
        "the bowler is being attacked and must stop the flow of runs.",
    ),
    (
        "F24 batter threat (bowler). ",
        "A set, strong striker and deep batting still to come raise the damage that a loose ball can do.",
    ),
    (
        "F25 over going badly (bowler). ",
        "Runs already conceded in the over, and a boundary or an extra off the previous ball, test the bowler's "
        "control within the over.",
    ),
    (
        "F26 wicket drought (bowler). ",
        "A long run of balls without a wicket lets the partnership grow and raises the need for a breakthrough.",
    ),
    (
        "Ranks within each Innings_Group. ",
        "Ranking LI and Stress within each innings stops first-innings balls being crowded out by second-innings "
        "death overs, where leverage is highest.",
    ),
)


def section_1(builder: ReportBuilder, inputs: ReportInputs) -> None:
    """Section 1 Pressure definition: definitions, reasons, shares, and every V1-V4 table and chart (27.2)."""
    s = builder.section(1)
    plain, formal = (inputs.definition[heading] for heading in DEFINITION_HEADINGS)

    s.add_heading("1.1 Plain-language definition")
    _write_definition(s, plain, {})

    s.add_heading("1.2 Formal definition")
    s.add_paragraph(
        "The definitions below are the text that the Pressure_Engine wrote to docs/pressure_definition.md in this "
        "Run. Besides the live band, α and the pressure share of Section 0, they use percentile ranks ",
        _const(inputs, "design_choices.D4_scope", lambda scope: _SCOPE_TEXT.get(str(scope), str(scope))),
        ", ", _const(inputs, "design_choices.D5_weights", str), " weights for the four pillars of each Role, a "
        "window of ", _const(inputs, "design_choices.D6_window"), " Legal_Balls for recent wickets and the recent "
        "run rate, a new-batter cap of ", _const(inputs, "design_choices.D7_new_batter_cap"), " Legal_Balls, and "
        "the LI scale m = ", _const(inputs, "pressure.pooled_mean_unscaled_li", lambda value: _fixed(value, 6)),
        ", the mean unscaled LI over every ball of innings 1 and 2.",
    )  # fmt: skip
    _write_definition(s, formal, _definition_equations(inputs))

    s.add_heading("1.3 Why leverage and these pillars")
    s.add_paragraph(
        "Leverage carries the match-situation ideas of the problem statement, and the pillars add what the player "
        "in each Role faces on the ball. The reason for each part:"
    )
    for lead, reason in _REASONS:
        s.add_bullet(Bold(lead), reason)

    s.add_heading("1.4 Where the pressure balls fall")
    _pressure_shares(s, inputs)

    s.add_heading("1.5 Validation on held-out matches")
    _validation(s, inputs)


def _pressure_shares(s: Section, inputs: ReportInputs) -> None:
    summary = inputs.frame("pressure_summary")
    main = summary[summary["pi"].isin(["pi_bat", "pi_bowl"])]
    counts = _and(
        f"{_count(row['n_pressure'])} of {_count(row['n_live'])} {row['role']} Live_Balls in innings {row['innings']}"
        for _, row in main.iterrows()
    )
    s.add_paragraph(
        "Pressure_Balls are ",
        _const(inputs, "pressure.pressure_ball_pct_of_live_balls", lambda value: f"{_span(_numbers(value), 2)}%"),
        f" of Live_Balls in each Role and innings ({counts}), and the lowest PI that qualifies is ",
        _derived_span(inputs, "pressure.lowest_pressure_ball_pi", 2),
        ".",
    )
    s.add_paragraph(
        "Phases split each innings at shares ",
        _const(inputs, "design_choices.D8_phase_splits", lambda splits: _and(_shown(value) for value in splits)),
        " of its Scheduled_Balls into powerplay, middle and death. The next two tables split each Role's "
        "Pressure_Balls by innings, Phase and Competition_Type; the last column is the cell's share of the Role's "
        "Pressure_Balls in that innings.",
    )  # fmt: skip
    for role in ROLES:
        _table(
            s, inputs, f"s1_share_{role}", f"share_{role}",
            ["innings", "phase", "comp_type", "n_deliveries", "n_live", "n_pressure", "pct_of_role_deliveries",
             "pct_of_innings_pressure"],
            key=["innings", "phase", "comp_type"],
            decimals={"n_deliveries": 0, "n_live": 0, "n_pressure": 0, "pct_of_role_deliveries": 1,
                      "pct_of_innings_pressure": 1},
            headers={"pct_of_role_deliveries": "% of cell Deliveries", "pct_of_innings_pressure": "% of innings Pressure_Balls"},
            caption=f"{role.capitalize()}-Role Pressure_Balls by innings, Phase and Competition_Type",
        )  # fmt: skip
    texts = []
    for role in ROLES:
        share = inputs.frame(f"share_{role}")
        shares, innings_list = [], []
        for innings, group in share.groupby("innings", sort=True):
            total = float(group["n_pressure"].sum())
            in_death = float(group.loc[group["phase"] == "death", "n_pressure"].sum())
            shares.append(_pct(in_death / total if total else 0.0))
            innings_list.append(str(innings))
        texts.append(f"{_and(shares)} of the {role} Pressure_Balls in innings {_and(innings_list)}")
    s.add_paragraph(
        f"Death overs hold {_and(texts)}, so the definition is not a death-overs rule: leverage and stress find "
        "pressure in every phase."
    )
    mean_li = inputs.frame("mean_li")
    first, second = (mean_li[mean_li["innings"] == innings].sort_values("over") for innings in (1, 2))
    _figure(s, inputs, "mean_li_chart", "Mean LI of the Deliveries in each over, innings 1 and 2", _MEDIUM_CM)
    s.add_paragraph(
        f"Mean LI rises from {_fixed(second['mean_li'].iloc[0], 2)} in over {int(second['over'].iloc[0])} to "
        f"{_fixed(second['mean_li'].iloc[-1], 2)} in over {int(second['over'].iloc[-1])} of the second innings, "
        f"and stays between {_span(first['mean_li'], 2, ' and ')} in the first innings, where the result is further "
        "away."
    )
    missing = inputs.frame("missing_components")
    gaps = missing[missing["n_missing"] > 0]
    if len(gaps):
        from superr.ranks import MISSING_RANK

        listed = _and(
            f"{row['component']} on {_count(row['n_missing'])} {row['role']} Deliveries of innings {row['innings']}"
            for _, row in gaps.iterrows()
        )
        s.add_paragraph(
            "A Pillar_Component cannot be computed when an input is missing or its window of Legal_Balls is empty. "
            f"This happens for {listed}; those Deliveries take the middle rank {_shown(MISSING_RANK)} for that "
            "component."
        )
    else:
        s.add_paragraph("Every Pillar_Component could be computed on every Delivery of innings 1 and 2.")


def _validation(s: Section, inputs: ReportInputs) -> None:
    s.add_paragraph(
        "Every check below uses Out_Of_Fold_Predictions only, on balls of matches used for rating, and its targets "
        "come from what happened on the ball or later: win-probability swings, extras, runs and wickets. No "
        "Design_Choice was picked by comparing these results."
    )
    # V1
    s.add_heading("V1: win-probability swing by Pressure_Index decile", level=3)
    v1 = inputs.frame("v1")
    deciles = v1[v1["group"] == "decile"]
    dead = v1[v1["group"] == "dead"]
    lows, highs = [], []
    for _, group in deciles.groupby(["role", "innings"], sort=True):
        ordered = group.sort_values("decile")["mean_swing"]
        lows.append(ordered.iloc[0])
        highs.append(ordered.iloc[-1])
    s.add_paragraph(
        "The Actual_WP_Swing of a ball is |WP after − WP before|. Its mean is "
        f"{_span(lows, 4)} in the lowest PI decile and {_span(highs, 4)} in the highest, by Role and innings, and "
        f"{_span(dead['mean_swing'], 4)} on Dead_Balls."
    )
    _table(
        s, inputs, "s1_v1_deciles", "v1",
        ["role", "innings", "group", "decile", "n_deliveries", "pi_min", "pi_max", "mean_swing"],
        key=["role", "innings", "group", "decile"],
        decimals={"n_deliveries": 0, "pi_min": 1, "pi_max": 1, "mean_swing": 4},
        headers={"group": "Group", "decile": "Decile", "pi_min": "Lowest PI", "pi_max": "Highest PI",
                 "mean_swing": "Mean Actual_WP_Swing"},
        caption="V1: mean Actual_WP_Swing in each PI decile of the Role's Live_Balls, and on Dead_Balls",
    )  # fmt: skip
    _figure(s, inputs, "v1_chart", "V1: mean Actual_WP_Swing by Pressure_Index decile, by Role and innings")

    # V2
    s.add_heading("V2: the biggest moments of each match", level=3)
    v2 = inputs.frame("v2")
    n_matches = int(v2["n_matches"].iloc[0])
    per_match = int(v2["n_big_moments"].iloc[0]) // n_matches if n_matches else 0
    ours = {role: _row(v2, "v2_big_moments.csv", method=f"{role}_pressure_balls") for role in ROLES}
    rules = v2[~v2["method"].astype(str).str.endswith("_pressure_balls")]
    s.add_paragraph(
        f"The Big_Moments of a match are its {per_match} balls with the largest Actual_WP_Swing, over "
        f"{_count(n_matches)} matches. Precision is the share of flagged balls that are Big_Moments and recall the "
        "share of Big_Moments that are flagged. Batter Pressure_Balls reach a recall of "
        f"{_pct(ours['batter']['recall'])} and bowler Pressure_Balls {_pct(ours['bowler']['recall'])}, against "
        + _and(f"{_pct(rule['recall'])} for '{_rule_label(str(rule['method']))}'" for _, rule in rules.iterrows())
        + ", while every method flags a similar share of balls."
    )
    _table(
        s, inputs, "s1_v2_big_moments", "v2",
        ["method", "n_flagged", "n_flagged_big_moments", "share_flagged", "precision", "recall"],
        key=["method"],
        decimals={"n_flagged": 0, "n_flagged_big_moments": 0, "share_flagged": 3, "precision": 3, "recall": 3},
        headers={"method": "Method", "n_flagged": "Flagged balls", "n_flagged_big_moments": "Flagged Big_Moments",
                 "share_flagged": "Share flagged", "precision": "Precision", "recall": "Recall"},
        caption="V2: Big_Moments found by each Role's Pressure_Balls and by the two Simple_Rules",
    )  # fmt: skip
    _figure(s, inputs, "v2_chart", "V2: precision and recall of each method on the Big_Moments", _MEDIUM_CM)

    # V3
    s.add_heading("V3: extras and within-player rates", level=3)
    extras = _row(inputs.frame("v3_extras"), "v3_extras.csv", innings="both")
    free = _row(inputs.frame("v3_extras_free"), "v3_extras_free.csv", innings="both")
    s.add_paragraph(
        "No model is trained on Wides or No_Balls, so the extras rate is a clean check. Bowlers bowl "
        f"{_fixed(extras['rate_pressure'], 2)} Wides plus No_Balls per 100 Bowler_Deliveries on their Pressure_Balls "
        f"against {_fixed(extras['rate_ordinary'], 2)} on Ordinary_Balls, a difference of "
        f"{_signed(extras['difference'], 2)}. The difference is {_signed(free['difference'], 2)} when the pressure "
        "balls are re-selected with the Extras_Free_PI, whose over-going-badly pillar ignores earlier extras, so the "
        "gap does not come from that pillar."
    )
    for table_id, source, title in (
        ("s1_v3_extras", "v3_extras",
         "V3: Wides plus No_Balls per 100 Bowler_Deliveries on bowler Pressure_Balls and Ordinary_Balls"),
        ("s1_v3_extras_free", "v3_extras_free",
         "V3: the same comparison with Pressure_Balls re-selected by the Extras_Free_PI"),
    ):  # fmt: skip
        _table(
            s, inputs, table_id, source,
            ["innings", "n_pressure", "rate_pressure", "n_ordinary", "rate_ordinary", "difference"],
            key=["innings"],
            decimals={"n_pressure": 0, "rate_pressure": 2, "n_ordinary": 0, "rate_ordinary": 2, "difference": 2},
            headers={"n_pressure": "Pressure balls", "rate_pressure": "Rate on pressure", "n_ordinary": "Ordinary balls",
                     "rate_ordinary": "Rate on ordinary", "difference": "Difference"},
            caption=title,
        )  # fmt: skip
    _figure(s, inputs, "v3_extras_chart", "V3: extras rate on bowler Pressure_Balls and Ordinary_Balls, by innings")
    within = inputs.frame("v3_within")
    texts = []
    for _, row in within.iterrows():
        clear = float(row["ci_low"]) > 0.0 or float(row["ci_high"]) < 0.0
        texts.append(
            f"{str(row['measure']).replace('_', ' ')} ({row['role']}s, {row['unit']}) {_signed(row['beta'], 2)} "
            f"(95% CI {_signed(row['ci_low'], 2)} to {_signed(row['ci_high'], 2)}"
            + ("" if clear else "; no clear change")
            + ")"
        )
    s.add_paragraph(
        "Within the same player, with one fixed effect per player and standard errors clustered by match, the "
        f"change from Ordinary_Balls to Pressure_Balls is: {'; '.join(texts)}."
    )
    _table(
        s, inputs, "s1_v3_within_player", "v3_within",
        ["measure", "role", "beta", "ci_low", "ci_high", "n_players", "n_deliveries", "mean_pressure", "mean_ordinary"],
        key=["measure", "role"],
        decimals={"beta": 2, "ci_low": 2, "ci_high": 2, "n_players": 0, "n_deliveries": 0, "mean_pressure": 2,
                  "mean_ordinary": 2},
        headers={"measure": "Measure", "beta": "Pressure minus ordinary", "ci_low": "95% low", "ci_high": "95% high",
                 "n_players": "Players", "mean_pressure": "Mean on pressure", "mean_ordinary": "Mean on ordinary"},
        caption="V3: within-player differences between Pressure_Balls and Ordinary_Balls (player fixed effects)",
    )  # fmt: skip
    _figure(s, inputs, "v3_within_chart", "V3: within-player differences with match-clustered 95% intervals")

    # V4
    s.add_heading("V4: what the simple rules miss", level=3)
    dead_share = inputs.frame("v4_dead")
    unflagged = inputs.frame("v4_unflagged")
    both = dead_share[dead_share["innings"].astype(str) == "both"]
    rule_texts = [
        f"{_pct(row['dead_share'])} of the balls flagged by '{_rule_label(str(row['rule']))}'" for _, row in both.iterrows()
    ]
    overall = _row(unflagged, "v4_high_leverage_unflagged.csv", innings="both")
    by_innings = unflagged[unflagged["innings"].astype(str) != "both"]
    s.add_paragraph(
        f"{_and(rule_texts)} come when the match is already decided (Dead_Balls), so these rules spend much of their "
        "pressure on dead games. High_Leverage_Balls are Live_Balls with an LI of at least ",
        _by_innings(inputs, "validation.lowest_high_leverage_li", 2),
        ". Of those bowled in the first "
        f"{_shown(100.0 * float(overall['max_ball_share']))}% of an innings, before the death Phase, "
        f"{_pct(overall['unflagged_share'])} are flagged by neither rule ("
        + _and(f"{_pct(row['unflagged_share'])} in innings {row['innings']}" for _, row in by_innings.iterrows())
        + ").",
    )  # fmt: skip
    _table(
        s, inputs, "s1_v4_dead_share", "v4_dead",
        ["rule", "innings", "n_flagged", "n_dead", "dead_share"],
        key=["rule", "innings"],
        decimals={"n_flagged": 0, "n_dead": 0, "dead_share": 3},
        headers={"rule": "Simple_Rule", "n_flagged": "Flagged balls", "n_dead": "Dead_Balls", "dead_share": "Dead share"},
        caption="V4: Simple_Rule balls that fall in matches already decided",
    )  # fmt: skip
    _table(
        s, inputs, "s1_v4_unflagged", "v4_unflagged",
        ["innings", "max_ball_share", "n_high_leverage", "n_unflagged", "unflagged_share"],
        key=["innings"],
        decimals={"max_ball_share": 2, "n_high_leverage": 0, "n_unflagged": 0, "unflagged_share": 3},
        headers={"max_ball_share": "Ball_Position_Share up to", "n_high_leverage": "High_Leverage_Balls",
                 "n_unflagged": "Flagged by neither rule", "unflagged_share": "Unflagged share"},
        caption="V4: High_Leverage_Balls before the death Phase that neither Simple_Rule flags",
    )  # fmt: skip
    _figure(s, inputs, "v4_chart", "V4: Simple_Rule balls in decided matches, and High_Leverage_Balls the rules miss")

    # signal-model matrix
    s.add_heading("Which models feed each signal", level=3)
    matrix = inputs.frame("signal_matrix")
    models = [column for column in ("M1", "M2", "M3", "M4", "Extras_Model") if column in matrix.columns]
    free_signals = [
        str(row["signal"]).replace("within_player_", "within-player ").replace("_", " ")
        for _, row in matrix.iterrows()
        if not any(str(row[column]).strip().lower() == "true" for column in models)
    ]
    s.add_paragraph(
        "The table marks, for each signal, the models whose outputs feed both the signal and the Pressure_Index. "
        f"The {_and(free_signals)} use no model output at all, so no model shared with the Pressure_Index can "
        "produce their differences."
    )
    _table(
        s, inputs, "s1_signal_model_matrix", "signal_matrix",
        ["signal", "validation", *models],
        key=["signal"],
        headers={"signal": "Signal", "validation": "Check"},
        caption="Models whose outputs feed both the validation signal and the Pressure_Index",
    )  # fmt: skip


# ------------------------------------------------------------------------------ Section 2

def section_2(builder: ReportBuilder, inputs: ReportInputs) -> None:
    """Section 2 Performance measurement: NBV and Ω, the five models, the model comparison and the ratings (27.3)."""
    s = builder.section(2)
    s.add_heading("2.1 Net ball value and wicket cost")
    _ball_value(s, inputs)
    s.add_heading("2.2 Models")
    refs = _models(s, inputs)
    s.add_heading("2.3 Model comparison and Primary_Models")
    _model_comparison(s, inputs)
    s.add_heading("2.4 From ball values to ratings")
    _ratings(s, inputs)
    s.add_heading("2.5 Design notes")
    _design_notes(s, inputs, refs)


_CELL_COLUMNS = ["role", "innings", "phase", "n_deliveries", "mean_actual", "mean_expected", "mean_per_delivery"]
_CELL_DECIMALS = {"n_deliveries": 0, "mean_actual": 3, "mean_expected": 3, "mean_per_delivery": 3}
_CELL_HEADERS = {"mean_actual": "Mean actual value", "mean_expected": "Mean expected value", "mean_per_delivery": "Mean NBV"}


def _ball_value(s: Section, inputs: ReportInputs) -> None:
    s.add_paragraph(
        "Each ball gets a value in runs for the batter and one for the bowler, measured against an average player in "
        "the same pre-ball state. The opponent keeps his actual skill, so the value allows for the quality of the "
        "bowler faced or the batter bowled to (Section 3.2)."
    )
    s.add_equation(
        r"\mathrm{NBV}_{\mathrm{bat}} = R - \Omega D - E_{\mathrm{bat}},\qquad "
        r"E_{\mathrm{bat}} = \sum_{c} p_c v_c - \Omega\,(p_{\mathrm{bw}} + p_{\mathrm{ro}})",
        "Batter net ball value: bat runs, minus the wicket cost when out, minus the expected value of an average "
        "batter facing the same bowler",
    )
    s.add_paragraph(
        "R is the bat runs off the ball, and D is 1 when the striker is out to a Counted_Wicket other than Retire Out. "
        "p_c and v_c are the M4 probability and the run value of outcome class c for an Average_Skill striker facing "
        "the actual bowler; p_bw and p_ro are the bowler-wicket and run-out probabilities. On a No_Ball the expected "
        "value uses the Free_Hit flag. A batter run out at the non-striker's end, or out off a Wide, is charged −Ω "
        "on that ball (",
        _const(inputs, "value.dismissal_charge_counts", lambda counts: f"{_count(counts['total'])} Dismissal_Charges"),
        " in the rated matches).",
    )
    s.add_equation(
        r"\mathrm{NBV}_{\mathrm{bowl}} = -R_{\mathrm{c}} + \Omega B - X - E_{\mathrm{bowl}}",
        "Bowler net ball value: minus the runs conceded, plus the wicket cost for a bowler wicket, minus the extras "
        "cost, minus the expected value of an average bowler",
    )
    s.add_equation(
        r"E_{\mathrm{bowl}} = (1 - p_W - p_N)\,[\,\Omega\, p_{\mathrm{bw}} - \sum_{c} p_c v_c\,] - p_W X_W - p_N X_N",
        "Bowler expected value: the legal-ball value weighted by the chance of a legal ball, minus the expected "
        "Wide and No_Ball costs",
    )
    counts = inputs.frame("fallback_counts")
    used = counts[counts["n_bowler_deliveries"] > 0]
    fallback = (
        "No Bowler_Delivery needed the pooled fallback for these cell means (every count in "
        "outputs/value/fallback_counts.csv is 0)."
        if used.empty
        else "The pooled fallback over all six cells was used for "
        + _and(f"{row['quantity']} on {_count(row['n_bowler_deliveries'])} Bowler_Deliveries" for _, row in used.iterrows())
        + "."
    )
    s.add_paragraph(
        "R_c is the bat runs off a Legal_Ball and 0 on a Wide or No_Ball, so byes and leg-byes are never charged to "
        "the bowler; B is 1 for a Bowler_Wicket; X is the Extras_Cost of a Wide or No_Ball. In the expected value "
        "the M4 probabilities are for an Average_Skill bowler bowling to the actual striker, p_W and p_N are the "
        "Extras_Model's Wide and No_Ball probabilities, and X_W and X_N the expected Extras_Costs from the means of "
        "the ball's innings × Phase cell in its Training_Folds. A Wide costs its team runs plus the expected bat runs "
        "of the replacement ball. A No_Ball costs its penalty runs, the bat runs off it, the expected bat runs of the "
        "replacement ball and the Free_Hit_Premium, the extra bat runs per Free_Hit ball measured in the "
        f"Training_Folds. {fallback}"
    )
    names: dict[int, str] = {}
    for _, row in inputs.frame("dismissal_codes").iterrows():
        names.setdefault(int(row["dismissal_type_id"]), str(row["dismissal_type"]))
    codes = [int(code) for code in inputs.value("design_choices.D14_bowler_wicket_codes")]
    s.add_paragraph(
        "Only the dismissal codes ",
        _const(inputs, "design_choices.D14_bowler_wicket_codes", lambda values: _and(_shown(value) for value in values)),
        f" ({_and(names.get(code, f'code {code}') for code in codes)}) credit the bowler with a wicket. A run-out "
        "(code ", _const(inputs, "design_choices.D14_run_out_code"), ") is charged to the batter who is out, with no "
        "bowler credit. Retire Out (code ", _const(inputs, "design_choices.D14_retire_out_code"), ") gives no bowler "
        "credit and ",
        _const(
            inputs, "design_choices.D14_retire_out_batter_penalty",
            lambda penalty: "charges the batter the wicket cost" if penalty else "no charge to the batter",
        ),
        " (", _const(inputs, "value.retire_out_uncharged"), " uncharged Retire Out dismissals). Team runs follow "
        "the ", _const(inputs, "reconstruct.selected_encoding"), " Extras_Encoding, which the Score_Reconstructor "
        "selected from the score checks (Appendix).",
    )  # fmt: skip
    s.add_equation(
        r"\Omega = \hat{R}(s) - \hat{R}(s^{\prime})",
        "Wicket cost: the expected remaining runs before the ball minus the expected remaining runs after one more "
        "wicket",
    )
    s.add_paragraph(
        "R̂ is the calibrated M2 expected remaining runs, s the pre-ball state and s′ the Post_Wicket_State: one more "
        "wicket down, the next batter by batting order on strike with no balls faced, and RBQ without him. Both "
        "predictions come from the same M2 fit. In a chase Ω is capped at the runs needed, and it is never below 0; "
        "with 9 wickets down or nobody left to bat, the post-wicket runs are 0. So a wicket costs more when the "
        "batter is skilled and set and when little batting is left."
    )
    _table(
        s, inputs, "s2_omega", "m2_metrics",
        ["metric", "variant", "innings", "value", "n_rows"],
        key=["metric", "variant", "innings"],
        decimals={"n_rows": 0},
        headers={"n_rows": "Deliveries"},
        where={"metric": ("omega_mean", "omega_clipped_low", "omega_clipped_high", "needed_nonpositive",
                          "post_wicket_runs_zero")},
        caption="Wicket cost Ω: mean in runs (omega_mean), and Deliveries whose Ω was set to 0 (omega_clipped_low) "
        "or capped at the runs needed (omega_clipped_high), chases already won (needed_nonpositive) and states "
        "with no post-wicket runs (post_wicket_runs_zero)",
    )  # fmt: skip
    _figure(s, inputs, "omega_chart", "Mean wicket cost Ω by Legal_Balls left and wickets fallen", _MEDIUM_CM)
    cells = inputs.frame("nbv_cells")
    batter_cells = cells[cells["role"] == "batter"]
    worst = batter_cells.loc[batter_cells["mean_per_delivery"].abs().idxmax()]
    s.add_paragraph(
        "The mean NBV per Delivery of each Role and innings × Phase cell is ",
        _derived_span(inputs, "value.nbv_cell_means", 3),
        " runs (outputs/value/nbv_cells.csv). These means need not be 0, because NBV compares each player with an "
        "Average_Skill player and the players of a cell are not average: batter NBV averages "
        f"{_signed(worst['mean_per_delivery'], 3)} in the innings {worst['innings']} {worst['phase']} cell, which "
        "lower-order batters bat more often.",
    )
    if inputs.has("nbv_cells_own_theta"):
        own = inputs.frame("nbv_cells_own_theta")
        tolerance = float(inputs.value("tests.nbv_cell_tol"))
        outside = own[own["mean_per_delivery"].abs() > tolerance]
        verdict = (
            "every cell mean lies within ±"
            if outside.empty
            else f"{len(outside)} of the {len(own)} cell means lie outside ±"
        )
        s.add_paragraph(
            "With each player's own θ in place of Average_Skill, the same cell means check that the expected values "
            f"are calibrated: {verdict}", _const(inputs, "tests.nbv_cell_tol"), " runs per Delivery, the test "
            "threshold.",
        )  # fmt: skip
        _table(
            s, inputs, "s2_nbv_cells_own_theta", "nbv_cells_own_theta", _CELL_COLUMNS,
            key=["role", "innings", "phase"], decimals=_CELL_DECIMALS, headers=_CELL_HEADERS,
            caption="NBV per Delivery by Role and innings × Phase cell, with each player's own θ",
        )  # fmt: skip
    else:
        s.add_paragraph(
            "The same cell means with each player's own θ (Requirement 15, criterion 17) are not in this Run's "
            f"outputs: {OPTIONAL_OUTPUT_FILES['nbv_cells_own_theta']} under the outputs folder is missing, so that "
            "table is left out. Rerunning the value stage writes it."
        )


#: The ``model`` label of M4 in the model-comparison table (``model_choice.M4_MODEL``).
_M4 = "M4"


def _m4_family(inputs: ReportInputs) -> list[Part]:
    """The M4 Family text: the selected M4 Primary_Model's family, and the candidates it was selected from.

    The family is the M4 entry of the data-derived constant ``model_choice.primary_models``, so it
    follows the Model_Choice_Checker's selection (Requirement 23, criterion 8) and is labelled here
    when this is its first use; the candidates are the M4 rows of the model-comparison table.
    """
    comparison = inputs.frame("model_comparison")
    candidates = list(dict.fromkeys(comparison.loc[comparison["model"].astype(str) == _M4, "candidate"].astype(str)))
    selected = _const(inputs, "model_choice.primary_models", lambda primary: _model_label(primary[_M4]))
    if len(candidates) < 2:
        return [selected, " multiclass classifier. "]
    return [
        f"multiclass classifier; of the {_and(_model_label(family) for family in candidates)} candidates, the "
        "Model_Choice_Checker selects ", selected, " as the M4 Primary_Model (Section 2.3). ",
    ]  # fmt: skip


def _lgbm_parts(inputs: ReportInputs, model: str) -> list[Part]:
    """The LightGBM hyperparameters of M2, M3 or M4, each a Const."""
    p = f"models.{model}"
    return [
        "learning rate ", _const(inputs, f"{p}.learning_rate"), ", at most ", _const(inputs, f"{p}.num_leaves"),
        " leaves per tree, at least ", _const(inputs, f"{p}.min_data_in_leaf"), " rows per leaf, L2 penalty ",
        _const(inputs, f"{p}.lambda_l2"), " and at most ", _const(inputs, f"{p}.max_rounds"), " boosting rounds, "
        "with early stopping on a ", _const(inputs, f"{p}.es_share"), " share of the training matches held out as "
        "whole matches and a patience of ", _const(inputs, f"{p}.es_patience"), " rounds",
    ]  # fmt: skip


def _class_name(code: str) -> str:
    runs = re.fullmatch(r"r(\d+)", code)
    if runs:
        return f"{runs.group(1)} run{'s' if runs.group(1) != '1' else ''}"
    return {"dot": "dot ball", "other_runs": "other bat runs", "bowler_wicket": "Bowler_Wicket", "run_out": "Run_Out"}.get(
        code, code.replace("_", " ")
    )


def _iterations(inputs: ReportInputs, name: str, part: str = "") -> Const:
    """The early-stopping rounds of the outer fits whose fit id contains ``part``, as a range."""
    return _const(inputs, name, lambda by_fit: _span([value for fit, value in by_fit.items() if part in fit], 0))


def _models(s: Section, inputs: ReportInputs) -> dict[str, str]:
    """The five models (Requirements 8-11); returns the references of the tables that later notes cite."""
    skill, remaining_runs, win_prob, outcome, extras = _model_modules()
    refs: dict[str, str] = {}
    folds = inputs.frame("fold_table")
    exclusion = inputs.frame("exclusion_counts")
    selected = exclusion[exclusion["is_config_value"].astype(str).str.lower() == "true"]
    excluded = (
        [f"The {_count(selected['n_excluded_matches'].iloc[0])} matches whose rebuilt scores fail the score checks "
         "(a runs miss of more than ", _const(inputs, "design_choices.D20_score_tolerance"), " runs, or a failed "
         "over or wickets-margin check) are left out of training and rating, and so are the super overs (innings 3 "
         "and 4). "]
        if len(selected) == 1
        else ["Matches whose rebuilt scores fail the score checks (a runs miss of more than ",
              _const(inputs, "design_choices.D20_score_tolerance"), " runs, or a failed over or wickets-margin check) "
              "are left out of training and rating, and so are the super overs (innings 3 and 4). "]
    )  # fmt: skip
    s.add_paragraph(
        "Five models supply the match states and the expected values. All use the same ",
        _const(inputs, "design_choices.D15_n_folds"),
        f" Match_Grouped_Folds, which put each of the {_count(len(folds))} matches, with all its balls, in one fold "
        "and are stratified by Competition_Type. Every prediction used for a ball comes from a fit "
        "that never saw the ball's match, and inside each fit the features that other models supply for its training "
        "rows come from fits that also leave out those rows' folds (fold sandboxes). ",
        *excluded,
        "Every random step takes its own seed from the Config.",
    )  # fmt: skip

    # M1
    s.add_heading("M1 player skill", level=3)
    penalties = inputs.value("skill.penalty_runs_by_fit")
    n_inner = sum("_inner_" in fit for fit in penalties)
    n_outer = sum("_outer_" in fit for fit in penalties)
    s.add_paragraph(
        Bold("Target and family: "), "two sparse ridge regressions on Legal_Balls, one of the bat runs and one of a "
        "0/1 Counted_Wicket indicator. ",
        Bold("Features: "), f"one-hot indicators of {_and(_GROUP_NAMES.get(g, g) for g in skill.GROUPS)}, with the "
        "Wickets_Bucket at ",
        _const(inputs, "design_choices.D17_wicket_buckets", lambda buckets: _and(f"{low}-{high}" for low, high in buckets)),
        " wickets fallen. ",
        Bold("Folds: "), "nested in pairs: for held-out fold k, the training rows of fold j get θ from a fit without "
        f"folds k and j ({n_inner} inner and {n_outer} outer fits). ",
        Bold("Hyperparameters: "), "each fit chooses its ridge penalty for each target from ",
        _const(inputs, "models.m1.ridge_grid"), " by leave-one-fold-out cross-validation on its own training folds, "
        "and dismissal probabilities are clipped to ",
        _const(inputs, "models.m1.prob_clip", lambda clip: f"{_shown(clip[0])} to {_shown(clip[1])}"),
        " before the log loss. The fits chose ",
        _const(inputs, "skill.penalty_runs_by_fit", lambda by_fit: _span(by_fit.values(), 0)), " for the bat runs and ",
        _const(inputs, "skill.penalty_wkt_by_fit", lambda by_fit: _span(by_fit.values(), 0)), " for dismissals.",
    )  # fmt: skip
    s.add_paragraph(
        "θ_bat is the league mean bat runs per Legal_Ball plus the batter's runs effect, minus his dismissal effect "
        "times the training mean runs per wicket; θ_bowl is the runs a bowler saves per Legal_Ball, counting his "
        "dismissal effect at the same rate. RBQ divides the θ_bat of the batting still to come by the "
        "Typical_Top7_Sum, ", _derived_span(inputs, "skill.typical_top7_sum_by_fit", 2), " across fits.",
    )  # fmt: skip
    m1 = inputs.frame("m1_metrics")
    value = {(str(row["metric"]), str(row["model"])): float(row["value"]) for _, row in m1.iterrows() if pd.notna(row["value"])}
    _table(
        s, inputs, "s2_m1_metrics", "m1_metrics",
        ["metric", "model", "value", "n_legal_balls"],
        key=["metric", "model"],
        decimals={"value": 4, "n_legal_balls": 0},
        where={"metric": ("rmse_bat_runs", "log_loss_dismissal")},
        caption="M1: pooled held-out RMSE of bat runs and log loss of dismissals, against the same model without "
        "player indicators (baseline)",
    )  # fmt: skip
    s.add_paragraph(
        "M1 lowers the held-out RMSE of bat runs from "
        f"{_fixed(value[('rmse_bat_runs', 'baseline')], 4)} to {_fixed(value[('rmse_bat_runs', 'M1')], 4)} and the "
        f"log loss of dismissals from {_fixed(value[('log_loss_dismissal', 'baseline')], 4)} to "
        f"{_fixed(value[('log_loss_dismissal', 'M1')], 4)}. The gains per ball are small because the outcome of one "
        "ball is noisy."
    )

    # M2
    s.add_heading("M2 remaining runs", level=3)
    s.add_paragraph(
        Bold("Target: "), "the team runs from the ball to the end of the innings; in a chase the target is capped at "
        "the runs needed. A capped variant covers both innings, and a first-innings variant (first-innings rows "
        "only, uncapped) gives Par_Score and the uncapped expected runs behind Achievable_Rate. ",
        Bold("Family: "), f"LightGBM regression with the {inputs.settings.get('models.m2.objective')} objective, "
        f"{_monotone(remaining_runs.M2_FEATURES, remaining_runs.M2_MONOTONE)}. ",
        Bold("Features: "), f"{_features(remaining_runs.M2_FEATURES)}. ",
        Bold("Folds: "), "one outer fit per fold for each variant, plus inner fits for the sandbox features. ",
        Bold("Hyperparameters: "), *_lgbm_parts(inputs, "m2"), ". The outer fits stopped at ",
        _iterations(inputs, "remaining_runs.outer_best_iteration_by_fit", "_capped_"), " rounds (capped) and ",
        _iterations(inputs, "remaining_runs.outer_best_iteration_by_fit", "_first_"), " (first-innings). Every "
        "prediction is multiplied by its fit's Legal_Balls-left calibration factor (Section 2.5).",
    )  # fmt: skip
    m2 = inputs.frame("m2_metrics")
    _table(
        s, inputs, "s2_m2_rmse", "m2_metrics",
        ["variant", "innings", "metric", "value", "n_rows"],
        key=["variant", "innings", "metric"],
        decimals={"value": 3, "n_rows": 0},
        where={"metric": ("rmse_m2", "rmse_m2_raw", "rmse_lookup_baseline")},
        caption="M2: pooled held-out RMSE in runs of the calibrated predictions (rmse_m2), the raw predictions "
        "(rmse_m2_raw) and the lookup baseline of mean target by Legal_Balls left × wickets fallen",
    )  # fmt: skip
    pairs = []
    rmse = m2[m2["metric"].isin(["rmse_m2", "rmse_lookup_baseline"])]
    for _, group in rmse.groupby(["variant", "innings"], sort=False):
        values = dict(zip(group["metric"], group["value"].astype(float)))
        pairs.append(values["rmse_m2"] < values["rmse_lookup_baseline"])
    s.add_paragraph(
        "M2 has a lower held-out RMSE than the lookup baseline "
        + ("in every variant and innings." if all(pairs) else f"in {sum(pairs)} of the {len(pairs)} variant and innings cases.")
    )

    # M3
    s.add_heading("M3 win probability", level=3)
    s.add_paragraph(
        Bold("Target: "), "1 when the batting side wins the match; ties, super-over wins, no results and abandoned "
        "matches are left out of training and of the metrics. ",
        Bold("Family: "), "for each innings a LightGBM classifier, monotone in the first innings "
        f"({_monotone(win_prob.M3_FEATURES[1], win_prob.M3_MONOTONE[1])}) and in the second "
        f"({_monotone(win_prob.M3_FEATURES[2], win_prob.M3_MONOTONE[2])}), and a logistic-regression baseline; the "
        "Model_Choice_Checker picks one per innings (Section 2.3). Every candidate is calibrated by centred isotonic "
        "regression, whose fold-k calibrator is fitted only on predictions for other folds' balls made by fits "
        "without those balls' folds, with its output bounded to ",
        _const(inputs, "models.m3.calibration_bounds", lambda bounds: f"{_shown(bounds[0])} to {_shown(bounds[1])}"),
        ". Fixed rules set WP to 1, 0 or 0.5 when a chase is won, lost or tied. ",
        Bold("Features: "), f"first innings {_features(win_prob.M3_FEATURES[1])}; second innings "
        f"{_features(win_prob.M3_FEATURES[2])}. ",
        Bold("Hyperparameters: "), "LightGBM ", *_lgbm_parts(inputs, "m3"), ", with the outer fits stopping at ",
        _iterations(inputs, "win_prob.outer_lgbm_best_iteration_by_fit"), " rounds. Logistic baseline: C = ",
        _const(inputs, "models.m3.logistic.C"), " and at most ", _const(inputs, "models.m3.logistic.max_iter"),
        " solver iterations.",
    )  # fmt: skip
    _table(
        s, inputs, "s2_m3_by_phase", "m3_metrics",
        ["innings", "family", "phase", "metric", "value", "n_rows"],
        key=["innings", "family", "phase", "metric"],
        decimals={"value": 4, "n_rows": 0},
        where={"metric": ("log_loss", "brier", "auc"), "phase": ("powerplay", "middle", "death")},
        caption="M3: pooled held-out log loss, Brier score and AUC of the calibrated candidates by Phase (the "
        "pooled values over all Phases are in the model comparison of Section 2.3)",
    )  # fmt: skip
    fold_metrics = inputs.frame("m3_folds")
    gaps = {family: group["held_out_minus_in_fold"] for family, group in fold_metrics.groupby("family", sort=False)}
    _table(
        s, inputs, "s2_m3_folds", "m3_folds",
        ["innings", "fold", "iterations", "in_fold_log_loss", "held_out_log_loss", "held_out_minus_in_fold"],
        key=["innings", "fold"],
        decimals={"iterations": 0, "in_fold_log_loss": 4, "held_out_log_loss": 4, "held_out_minus_in_fold": 4},
        headers={"fold": "Fold", "iterations": "Rounds", "in_fold_log_loss": "In-fold log loss",
                 "held_out_log_loss": "Held-out log loss", "held_out_minus_in_fold": "Held-out minus in-fold"},
        where={"family": "lgbm"},
        caption="M3 overfitting check of the LightGBM classifiers: in-fold and held-out log loss of each fold's "
        "uncalibrated fit",
    )  # fmt: skip
    s.add_paragraph(
        "Every ball of an innings shares its match's label, so a classifier can fit the matches of its training "
        "folds too closely. The held-out minus in-fold log loss is "
        + _and(f"{_span(values, 3)} for the {_model_label(family)}" for family, values in gaps.items())
        + " across folds and innings (outputs/models/m3/fold_metrics.csv)."
    )
    gap_rows = inputs.frame("m3_break_gap")
    gap_rows = gap_rows[gap_rows["record"] == "summary"]
    refs["break_gap"] = _table(
        s, inputs, "s2_m3_break_gap", "m3_break_gap",
        ["family", "n_matches", "wp1_end", "one_minus_wp2", "gap", "abs_gap"],
        key=["family"],
        decimals={"n_matches": 0, "wp1_end": 3, "one_minus_wp2": 3, "gap": 3, "abs_gap": 3},
        headers={"n_matches": "Matches", "wp1_end": "Mean WP at the end of innings 1",
                 "one_minus_wp2": "Mean 1 - WP at the start of the chase", "gap": "Mean gap", "abs_gap": "Mean absolute gap"},
        where={"record": "summary"},
        caption="M3 at the innings break: first-innings WP at the end of innings 1 against 1 minus the "
        "second-innings WP before the first ball of the chase",
    )  # fmt: skip
    refs["break_gap_text"] = _and(
        f"{_fixed(row['abs_gap'], 3)} for the {_model_label(row['family'])}" for _, row in gap_rows.iterrows()
    )
    s.add_paragraph(
        "At the innings break the first-innings model's WP for the side batting first and 1 minus the "
        "second-innings model's WP at the start of the chase describe the same moment; their mean absolute "
        f"difference is {refs['break_gap_text']} over {_count(gap_rows['n_matches'].max())} matches."
    )
    for innings in (1, 2):
        _figure(
            s, inputs, f"m3_curve_chart_{innings}",
            f"M3 Calibration_Curves of every innings-{innings} Candidate_Model: observed win share against mean "
            "predicted WP, overall and by Phase",
        )
    _figure(
        s, inputs, "wp_curve_chart",
        "Win probability of the side batting first before every ball of match "
        f"{_match_id(inputs.path('wp_curve_chart'))}, from the M3 Primary_Models",
        _MEDIUM_CM,
    )

    # M4
    s.add_heading("M4 ball outcome", level=3)
    s.add_paragraph(
        Bold("Target: "), f"the outcome class of each Legal_Ball: {_and(_class_name(c) for c in outcome.CLASSES)}. "
        "Wides and No_Balls are left out of the target, so the extras rate stays an independent check (V3). ",
        Bold("Family: "), *_m4_family(inputs),
        Bold("Features: "), f"{_features(outcome.M4_FEATURES)}. ",
        Bold("Folds: "), "one outer fit per fold. ",
        Bold("Hyperparameters: "), "LightGBM ", *_lgbm_parts(inputs, "m4"), "; the outer fits stopped at ",
        _iterations(inputs, "outcome.outer_lgbm_best_iteration_by_fit"), " rounds. ",
        "Each ball gets predictions with both players' actual θ, with the striker at Average_Skill (batter view), "
        "with the bowler at Average_Skill (bowler view) and with both at Average_Skill (for LI). Average_Skill is "
        "the ball-weighted mean θ of the players of a fit's training Legal_Balls: ",
        _const(
            inputs, "outcome.average_skill_by_fold",
            lambda by_fold: f"θ_bat {_span([fold['theta_bat'] for fold in by_fold.values()], 3)} and θ_bowl "
            f"{_span([fold['theta_bowl'] for fold in by_fold.values()], 3)} across folds",
        ),
        ".",
    )  # fmt: skip
    losses = inputs.frame("m4_log_loss")
    _table(
        s, inputs, "s2_m4_log_loss", "m4_log_loss",
        ["model", "n_legal_balls", "log_loss"],
        key=["model"],
        decimals={"n_legal_balls": 0, "log_loss": 4},
        caption="M4 LightGBM candidate: pooled held-out multiclass log loss against the Phase-only baseline of "
        "training class frequencies",
    )  # fmt: skip
    runs = inputs.frame("m4_expected_runs")
    runs["gap"] = (runs["mean_expected_bat_runs"] - runs["mean_actual_bat_runs"]).abs()
    widest = runs.loc[runs["gap"].idxmax()]
    weighted = float((runs["gap"] * runs["n_legal_balls"]).sum() / runs["n_legal_balls"].sum())
    _table(
        s, inputs, "s2_m4_expected_runs", "m4_expected_runs",
        ["phase", "wickets_bucket", "n_legal_balls", "mean_expected_bat_runs", "mean_actual_bat_runs"],
        key=["phase", "wickets_bucket"],
        decimals={"n_legal_balls": 0, "mean_expected_bat_runs": 3, "mean_actual_bat_runs": 3},
        headers={"wickets_bucket": "Wickets_Bucket", "mean_expected_bat_runs": "Mean expected bat runs",
                 "mean_actual_bat_runs": "Mean actual bat runs"},
        caption="M4 LightGBM candidate: mean expected and actual bat runs per held-out Legal_Ball by Phase × "
        "Wickets_Bucket",
    )  # fmt: skip
    model_loss = float(losses.loc[~losses["model"].astype(str).str.contains("baseline"), "log_loss"].iloc[0])
    base_loss = float(losses.loc[losses["model"].astype(str).str.contains("baseline"), "log_loss"].iloc[0])
    m4_family = str(inputs.value("model_choice.primary_models")[_M4])
    primary_text = ""
    if m4_family != outcome.LGBM_FAMILY:
        primary = _row(inputs.frame("model_comparison"), OUTPUT_FILES["model_comparison"], model=_M4, candidate=m4_family)
        primary_text = (
            f" The {_model_label(m4_family)} Primary_Model's held-out log loss on the same Legal_Balls is "
            f"{_fixed(primary['log_loss'], 4)} (Section 2.3)."
        )
    s.add_paragraph(
        f"The LightGBM candidate's held-out log loss is {_fixed(model_loss, 4)} against {_fixed(base_loss, 4)} for the "
        f"baseline. Its mean expected and actual bat runs differ by {_fixed(weighted, 3)} runs per ball on average "
        f"over the cells, weighted by balls; the widest gap, {_fixed(widest['gap'], 3)} runs, is in the "
        f"{widest['phase']} {widest['wickets_bucket']} cell, which holds {_count(widest['n_legal_balls'])} balls."
        + primary_text
    )

    # Extras_Model
    s.add_heading("Extras_Model", level=3)
    s.add_paragraph(
        Bold("Target: "), "whether a Delivery is a Legal_Ball, a Wide or a No_Ball. ",
        Bold("Family: "), "multinomial logistic regression, so the three probabilities sum to 1. ",
        Bold("Features: "), f"{_features(extras.EXTRAS_FEATURES)}; no bowler identity, θ_bowl, bowling type or pace "
        "or spin, so the probabilities describe an average bowler in the state. ",
        Bold("Folds: "), "one outer fit per fold. ",
        Bold("Hyperparameters: "), "C = ", _const(inputs, "models.extras.C"), " and at most ",
        _const(inputs, "models.extras.max_iter"), " solver iterations.",
    )  # fmt: skip
    extras_loss = inputs.frame("extras_log_loss")
    _table(
        s, inputs, "s2_extras_log_loss", "extras_log_loss",
        ["model", "n_deliveries", "log_loss"],
        key=["model"],
        decimals={"n_deliveries": 0, "log_loss": 4},
        caption="Extras_Model: pooled held-out log loss against the innings × Phase rate baseline (cell_baseline)",
    )  # fmt: skip
    model = float(_row(extras_loss, "extras/log_loss.csv", model="extras")["log_loss"])
    baseline = float(extras_loss.loc[extras_loss["model"] != "extras", "log_loss"].iloc[0])
    rates = inputs.frame("extras_rates")
    cells_only = rates[rates["innings"].astype(str) != "all"]
    wide_gap = float((cells_only["wide_rate"] - cells_only["mean_p_wide"]).abs().max())
    no_ball_gap = float((cells_only["no_ball_rate"] - cells_only["mean_p_noball"]).abs().max())
    s.add_paragraph(
        f"Its held-out log loss is {_fixed(model, 4)} against {_fixed(baseline, 4)} for the cell rates. In each of the "
        f"{len(cells_only)} innings × Phase cells its mean held-out probabilities lie within {_fixed(wide_gap, 4)} of "
        f"the observed Wide rate and within {_fixed(no_ball_gap, 4)} of the observed No_Ball rate "
        "(outputs/models/extras/rates_by_cell.csv)."
    )
    return refs


def _match_id(path: Path) -> str:
    """The match ID in the name of a WP-curve chart (``wp_curve_<match_id>.png``)."""
    found = re.search(r"(\d+)", path.stem)
    return found.group(1) if found else path.stem


#: Config hyperparameters of the Alternative_Models, shown when the comparison table holds them.
_ALTERNATIVE_SETTINGS: Mapping[str, tuple[tuple[str, str], ...]] = MappingProxyType(
    {
        "xgb": (
            ("eta", "learning rate"), ("max_depth", "maximum depth"), ("min_child_weight", "minimum child weight"),
            ("reg_lambda", "L2 penalty"), ("max_rounds", "maximum rounds"), ("es_share", "early-stopping share"),
            ("es_patience", "patience"),
        ),
        "rf": (("n_estimators", "trees"), ("max_depth", "maximum depth"), ("min_samples_leaf", "minimum samples per leaf")),
    }
)  # fmt: skip


def _m4_tiebreak(s: Section, inputs: ReportInputs) -> None:
    """The M4 calibration-aware tie-break (Requirement 23, criterion 8; DD-17), when it displaced a candidate.

    The tie-break keeps a Config-preferred M4 family whose pooled log loss is within the relative band of the
    lowest, because it is the family whose own-θ Net_Ball_Value per innings × Phase cell passes the Requirement 15,
    criterion 17 calibration check that feeds the ratings. The ``value`` stage re-verifies it on its own inputs and
    writes the two-candidate diagnostic ``value/m4_tiebreak_cells.csv`` (Requirement 23, criterion 14); without that
    file, or when the lowest-log-loss candidate is itself the Primary_Model, there is nothing to explain.
    """
    if not inputs.has("m4_tiebreak_cells"):
        return
    comparison = inputs.frame("model_comparison")
    m4 = comparison[comparison["model"].astype(str) == _M4].copy()
    m4["_ll"] = pd.to_numeric(m4["log_loss"], errors="coerce")
    primary = m4[_truthy(m4["primary"])]
    if primary.empty or m4["_ll"].isna().all():
        return
    lowest = m4.loc[m4["_ll"].idxmin()]
    selected_family = str(primary.iloc[0]["candidate"])
    if str(lowest["candidate"]) == selected_family:  # the lowest-log-loss candidate is the Primary_Model
        return
    selected_label, displaced_label = _model_label(selected_family), _model_label(str(lowest["candidate"]))
    cells = inputs.frame("m4_tiebreak_cells")
    per_role = len(cells) // len(ROLES)
    breaches = cells[~_truthy(cells["displaced_within_tol"])]
    breach_text = _and(
        f"the {row['role']} innings {int(row['innings'])} {row['phase']} cell "
        f"({_signed(row['displaced_mean_per_delivery'], 3)} runs per Delivery)"
        for _, row in breaches.iterrows()
    )
    s.add_paragraph(
        f"The {selected_label} M4 Primary_Model does not have the lowest M4 log loss: {displaced_label} is lower, ",
        f"{_fixed(lowest['_ll'], 4)} against {_fixed(primary.iloc[0]['_ll'], 4)}. That gap is within the "
        "calibration-aware tie-break's relative band of ",
        _const(inputs, "model_choice.calibration_band", lambda band: f"{_shown(100.0 * float(band))}%"),
        f" of the lowest, and {selected_label} is the Config-preferred M4 family, so the tie-break keeps it "
        f"(decision DD-17). {selected_label} is preferred because of the own-θ Net_Ball_Value cell check that feeds "
        "the ratings (Section 2.1): with each player's own θ the mean NBV per Delivery should be near 0 in every "
        f"innings × Phase cell. {selected_label} stays within ±",
        _const(inputs, "tests.nbv_cell_tol"),
        f" runs in all {per_role} cells of both Roles, while {displaced_label} breaches it in {breach_text}. The "
        "Value_Engine re-checks this before it builds the ratings (Requirement 23, criterion 14).",
    )  # fmt: skip
    _table(
        s, inputs, "s2_m4_tiebreak", "m4_tiebreak_cells",
        ["role", "innings", "phase", "n_deliveries", "selected_mean_per_delivery", "selected_within_tol",
         "displaced_mean_per_delivery", "displaced_within_tol"],
        key=["role", "innings", "phase"],
        decimals={"n_deliveries": 0, "selected_mean_per_delivery": 3, "displaced_mean_per_delivery": 3},
        headers={"selected_mean_per_delivery": f"{selected_label} mean NBV", "selected_within_tol": f"{selected_label} in tolerance",
                 "displaced_mean_per_delivery": f"{displaced_label} mean NBV", "displaced_within_tol": f"{displaced_label} in tolerance"},
        caption=f"M4 tie-break own-θ NBV per Delivery by Role and innings × Phase cell: the selected {selected_label} "
        f"against the displaced {displaced_label}, each flagged as within the ±{_shown(inputs.value('tests.nbv_cell_tol'))}-run "
        "calibration tolerance",
    )  # fmt: skip


def _model_comparison(s: Section, inputs: ReportInputs) -> None:
    comparison = inputs.frame("model_comparison")
    families = list(dict.fromkeys(comparison["candidate"].astype(str)))
    s.add_paragraph(
        "For each M3 innings the Model_Choice_Checker selects the Candidate_Model with the lowest pooled held-out "
        "log loss as the Primary_Model; for M4 a calibration-aware tie-break can keep a Config-preferred family "
        "whose log loss is within a relative band of the lowest (below). The selected Primary_Models are ",
        _const(
            inputs, "model_choice.primary_models",
            lambda primary: _and(f"{model} {_model_label(primary[model])}" for model in sorted(primary)),
        ),
        ". Every WP, LI, PI, NBV and rating in this report comes from the Primary_Models.",
    )  # fmt: skip
    _table(
        s, inputs, "s2_model_comparison", "model_comparison",
        ["model", "candidate", "n_rows", "log_loss", "brier", "auc", "calibration_error", "primary",
         "top10_changes_batter", "top10_changes_bowler", "failure_reason"],
        key=["model", "candidate"],
        decimals={"n_rows": 0, "log_loss": 4, "brier": 4, "auc": 4, "calibration_error": 4,
                  "top10_changes_batter": 0, "top10_changes_bowler": 0},
        headers={"candidate": "Candidate", "brier": "Brier", "auc": "AUC", "calibration_error": "Calibration_Error",
                 "primary": "Primary", "top10_changes_batter": "Top-10 changes, batters",
                 "top10_changes_bowler": "Top-10 changes, bowlers", "failure_reason": "Failure"},
        caption="Model comparison: pooled held-out metrics of every Candidate_Model, the selected Primary_Models, and "
        "the top-10 players lost when a candidate replaces the Primary_Model",
    )  # fmt: skip
    _m4_tiebreak(s, inputs)
    is_primary = comparison["primary"].astype(str).str.lower().eq("true")
    primary_m3 = comparison[comparison["model"].astype(str).str.startswith("M3") & is_primary]
    threshold = float(inputs.value("tests.calibration_error_max"))
    below = bool((primary_m3["calibration_error"].astype(float) <= threshold).all()) and len(primary_m3) > 0
    s.add_paragraph(
        f"The Calibration_Error of the M3 Primary_Models is {_span(primary_m3['calibration_error'], 4, ' and ')}, "
        + ("within" if below else "not all within")
        + " the test threshold of ",
        _const(inputs, "tests.calibration_error_max"),
        ".",
    )
    absent = [_model_label(family) for family in ("xgb", "rf") if family not in families]
    if absent:
        s.add_paragraph(
            f"This Run's comparison holds no {' or '.join(absent)} Alternative_Model, so it compares the "
            f"{_and(_model_label(family) for family in families)} candidates only."
        )
    for family, settings in _ALTERNATIVE_SETTINGS.items():
        if family not in families:
            continue
        parts: list[Part] = [f"{_model_label(family).capitalize()} settings, from the Config: "]
        for model in ("m3", "m4"):
            parts.append(f"{model.upper()} ")
            for position, (key, words) in enumerate(settings):
                parts += [", " if position else "", f"{words} ", _const(inputs, f"models.{family}.{model}.{key}")]
            parts.append("; " if model == "m3" else ".")
        s.add_paragraph(*parts)
    changes = comparison[["top10_changes_batter", "top10_changes_bowler"]]
    if changes.notna().any().any():
        rows = comparison[changes.notna().any(axis=1)]
        s.add_paragraph(
            "Replacing a Primary_Model by another candidate changes "
            + _and(
                f"{_count(row['top10_changes_batter'])} batter and {_count(row['top10_changes_bowler'])} bowler places "
                f"for {row['model']} {_model_label(row['candidate'])}"
                for _, row in rows.iterrows()
                if pd.notna(row["top10_changes_batter"]) and pd.notna(row["top10_changes_bowler"])
            )
            + " of the top 10."
        )
    else:
        s.add_paragraph(
            "The top-10 change columns are empty: the reruns with each candidate that was not selected "
            "(Requirement 23, criterion 10) are not part of this Run."
        )


def _ratings(s: Section, inputs: ReportInputs) -> None:
    s.add_paragraph(
        "Each ball is weighted by its Delivery_Weight, the Recency_Weight times the Stakes_Weight; no weight depends on "
        "pressure. The Recency_Weight halves every ", _const(inputs, "design_choices.D9_half_life_years"), " years back "
        "from the last match date in the data, and the Stakes_Weight is ",
        _const(inputs, "design_choices.D10_knockout_weight"), " for a knockout match and 1 otherwise.",
    )  # fmt: skip
    s.add_equation(
        r"w_{\mathrm{rec}} = 0.5^{\,d\,/\,(365.25\,h)}",
        "Recency_Weight: one half raised to the days since the match divided by the half-life in days",
    )
    s.add_paragraph(
        "Here d is the days between the match and the last match date in the data, and h the half-life in years. "
        "Clutch_Level is the player's weighted mean NBV on his Pressure_Balls P, in runs per over, with w_i the "
        "Delivery_Weight and v_i the NBV of ball i:"
    )
    s.add_equation(
        r"C_p = 6\,\frac{\sum_{i \in P} w_i v_i}{\sum_{i \in P} w_i}",
        "Clutch_Level: 6 times the weighted mean NBV over the player's Pressure_Balls",
    )
    s.add_paragraph(
        "Usual_Baseline uses the player's Ordinary_Balls in the ",
        _const(inputs, "design_choices.D13_cells", lambda cells: _CELL_TEXT.get(str(cells), str(cells))),
        " cells that hold both his Pressure_Balls and his Ordinary_Balls, the Matched_Cells M. Each cell's weighted "
        "mean ordinary NBV v̄_c gets the cell's share π_c of his pressure weight, so his usual level is measured on "
        "the same mix of innings and phases as his pressure balls:",
    )  # fmt: skip
    s.add_equation(
        r"O_p = 6 \sum_{c \in M} \pi_c\, \bar{v}_c",
        "Usual_Baseline: 6 times the pressure-share-weighted mean of the player's ordinary NBV over the matched cells",
    )
    s.add_paragraph(
        "Replacement_Level RL is the ",
        _const(inputs, "design_choices.D12_replacement_pct", lambda pct: f"{_shown(pct)}th"),
        " percentile of the unshrunk Clutch_Level of the Role's Eligible_Players: ",
        _const(inputs, "rating.replacement_level.batter", lambda value: _fixed(value, 3)), " runs per over for batters "
        "and ", _const(inputs, "rating.replacement_level.bowler", lambda value: _fixed(value, 3)), " for bowlers. "
        "Uplift is the pressure level above the larger of RL and the usual level, and the Pressure_Gap g = C − O is "
        "the change from the player's own usual level:",
    )  # fmt: skip
    s.add_equation(
        r"U_p = C_p - \max(\mathrm{RL},\, O_p)", "Uplift: Clutch_Level minus the larger of Replacement_Level and Usual_Baseline"
    )
    s.add_paragraph(
        "Shrinkage pulls each estimate towards a prior in proportion to its noise. The Shrinkage_Factor B of an "
        "estimate uses the Role's between-player variance τ², estimated by method of moments and set to 0 when it "
        "comes out negative, and the player's match-clustered standard error SE:"
    )
    s.add_equation(
        r"B = \frac{\tau^2}{\tau^2 + \mathrm{SE}^2}",
        "Shrinkage_Factor: tau squared over tau squared plus the squared standard error",
    )
    s.add_paragraph(
        "The shrunk Clutch_Level is built on the player's own usual level: the Usual_Baseline is shrunk towards RL and "
        "the Pressure_Gap towards 0, each with its own τ² and SE, and the Uplift is shrunk towards 0 (Section 2.5):"
    )
    s.add_equation(
        r"C_p^{*} = O_p^{*} + g_p^{*},\quad O_p^{*} = \mathrm{RL} + B_O\,(O_p - \mathrm{RL}),\quad "
        r"g_p^{*} = B_g\,(C_p - O_p),\quad U_p^{*} = B_U\, U_p",
        "Shrunk Clutch_Level as the shrunk Usual_Baseline plus the shrunk Pressure_Gap, and the shrunk Uplift",
    )
    s.add_paragraph(
        "The Blend_Weights come from split-half reliabilities: each player's matches are numbered by date, r is the "
        "correlation across the Role's Eligible_Players between the estimates from odd and even matches, and the "
        "reliability ρ is set to 0 when negative. When both reliabilities of a Role are 0, w_L = 1 and w_U = 0. "
        "Players are ranked by Final_Score, with ties going to the higher shrunk Clutch_Level and then the lower "
        "player_id."
    )
    s.add_equation(
        r"\rho = \frac{2r}{1+r},\qquad w_L = \frac{\rho_L}{\rho_L + \rho_U},\qquad w_U = 1 - w_L",
        "Split_Half_Reliability two r over one plus r, and the Blend_Weights: the Clutch_Level reliability over the "
        "sum of both reliabilities",
    )
    s.add_equation(r"S_p = w_L\, C_p^{*} + w_U\, U_p^{*}", "Final_Score: the blend of shrunk Clutch_Level and shrunk Uplift")
    s.add_paragraph(
        "Hit_Rate is the share of a player's Pressure_Matches with a positive summed pressure NBV, and "
        "Worst_Quartile_Mean the mean per-match pressure NBV, in runs per over, over his worst quarter of Pressure_Matches "
        "(Section 4)."
    )
    s.add_paragraph(
        "This Run's Role-level values, with ρ_L the reliability of Clutch_Level and ρ_U that of Uplift (the two "
        "tables also give the values before the floors at 0):"
    )
    for role in ROLES:
        s.add_bullet(
            Bold(f"{_TITLE[role]}: "), "τ² of the Usual_Baseline ", _role_value(inputs, "tau2_usual", role),
            ", of the Pressure_Gap ", _role_value(inputs, "tau2_gap", role), " and of the Uplift ",
            _role_value(inputs, "tau2_uplift", role), "; ρ_L ", _role_value(inputs, "rho_L", role), " and ρ_U ",
            _role_value(inputs, "rho_U", role), "; w_L ", _role_value(inputs, "w_L", role), " and w_U ",
            _role_value(inputs, "w_U", role), ".",
        )  # fmt: skip
    _table(
        s, inputs, "s2_role_variances", "role_summary",
        ["role", "n_eligible", "replacement_level", "tau2_usual", "tau2_gap_raw", "tau2_gap", "tau2_uplift_raw",
         "tau2_uplift", "tau2_clutch_raw"],
        key=["role"],
        decimals={"n_eligible": 0, "replacement_level": 3, "tau2_usual": 3, "tau2_gap_raw": 3, "tau2_gap": 3,
                  "tau2_uplift_raw": 3, "tau2_uplift": 3, "tau2_clutch_raw": 3},
        headers={"n_eligible": "Eligible_Players", "replacement_level": "Replacement_Level", "tau2_usual": "τ² usual",
                 "tau2_gap_raw": "τ² gap before floor", "tau2_gap": "τ² gap", "tau2_uplift_raw": "τ² uplift before floor",
                 "tau2_uplift": "τ² uplift", "tau2_clutch_raw": "τ² Clutch_Level before floor"},
        caption="Replacement_Level and between-player variances τ² by Role, in runs per over (squared for τ²)",
    )  # fmt: skip
    _table(
        s, inputs, "s2_role_reliability", "role_summary",
        ["role", "n_rho", "r_L", "rho_L_raw", "rho_L", "r_U", "rho_U_raw", "rho_U", "w_L", "w_U"],
        key=["role"],
        decimals={"n_rho": 0, "r_L": 3, "rho_L_raw": 3, "rho_L": 3, "r_U": 3, "rho_U_raw": 3, "rho_U": 3, "w_L": 3,
                  "w_U": 3},
        headers={"n_rho": "Players", "r_L": "r Clutch_Level", "rho_L_raw": "ρ_L before floor", "rho_L": "ρ_L",
                 "r_U": "r Uplift", "rho_U_raw": "ρ_U before floor", "rho_U": "ρ_U", "w_L": "w_L", "w_U": "w_U"},
        caption="Split-half correlations r, Split_Half_Reliability ρ and Blend_Weights by Role",
    )  # fmt: skip
    summary = inputs.frame("role_summary")
    parts: list[Part] = []
    for role in ROLES:
        row = _row(summary, "role_summary.csv", role=role)
        parts += [*_gap_parts(inputs, role), " "]
        if float(row["rho_L_raw"]) <= 0.0 and float(row["rho_U_raw"]) <= 0.0:
            parts.append(
                f"Both {role} reliabilities are negative before the floor ({_fixed(row['rho_L_raw'], 3)} and "
                f"{_fixed(row['rho_U_raw'], 3)}): a {role}'s pressure values from his odd matches do not predict "
                f"those from his even matches, which is why w_L = 1. "
            )
    for role in ROLES:
        if float(inputs.value(f"rating.tau2_gap.{role}")) == 0.0:
            continue
        estimates = inputs.frame(f"estimates_{role}")
        top = inputs.frame(f"top10_{role}")["player_id"].astype(int)
        listed = estimates[estimates["player_id"].astype(int).isin(top)]
        if len(listed):
            usual, gap = listed["shrunk_usual"].astype(float), listed["shrunk_gap"].astype(float)
            mainly = bool((gap.abs() < usual.abs()).all())
            parts.append(
                f"For the ten listed {_PLURAL[role]} the shrunk usual level is {_span(usual, 3)} and the shrunk "
                f"Pressure_Gap {_span(gap, 3)} runs per over"
                + (", so their places also rest mainly on the usual level." if mainly else ".")
            )
    s.add_paragraph(*parts)
    s.add_paragraph(
        "The 95% intervals and P_Top10 come from ", _n_boot(inputs), " Bootstrap_Resamples of the ",
        _const(inputs, "bootstrap.n_resampled_matches", _count), " matches, drawn with replacement with the seed in "
        "the Config. Each resample recomputes every player's estimates, standard errors, Shrinkage_Factors, "
        "Final_Score and rank, but holds Replacement_Level, τ², the reliabilities and the Blend_Weights at the main "
        "ranking's values. The intervals therefore show the sampling noise of the players' own balls given those "
        "Role-level values, not the uncertainty in the values themselves.",
    )  # fmt: skip


def _role_value(inputs: ReportInputs, stem: str, role: str) -> Const:
    """A per-Role data-derived constant of the Rating_Engine (``rating.<stem>.<role>``), to 3 decimals."""
    return _const(inputs, f"rating.{stem}.{role}", lambda value: _fixed(value, 3))


def _pair(inputs: ReportInputs, stem: str) -> list[Part]:
    """A per-Role data-derived constant as "x for batters and y for bowlers"."""
    return [
        _const(inputs, f"{stem}.batter", lambda value: _fixed(value, 3)), " for batters and ",
        _const(inputs, f"{stem}.bowler", lambda value: _fixed(value, 3)), " for bowlers",
    ]  # fmt: skip


def _design_notes(s: Section, inputs: ReportInputs, refs: Mapping[str, str]) -> None:
    m2 = inputs.frame("m2_metrics")

    def m2_value(metric: str, variant: str, innings: int) -> float:
        return float(_row(m2, "models/m2/metrics.csv", metric=metric, variant=variant, innings=innings)["value"])

    factors = inputs.frame("m2_lb_calibration")
    low, high = int(factors["lb_left"].min()), int(factors["lb_left"].max())
    rmse = m2[m2["metric"].isin(["rmse_m2", "rmse_m2_raw"])]
    improved = all(
        float(group.loc[group["metric"] == "rmse_m2", "value"].iloc[0])
        < float(group.loc[group["metric"] == "rmse_m2_raw", "value"].iloc[0])
        for _, group in rmse.groupby(["variant", "innings"], sort=False)
    )
    par = inputs.frame("m2_par_bias")
    summary = _row(inputs.frame("role_summary"), "role_summary.csv", role="batter")
    notes: list[list[Part]] = [
        [
            Bold("Poisson objective for M2. "),
            "The remaining runs are a count, so M2 uses LightGBM's "
            f"{inputs.settings.get('models.m2.objective')} objective; L2 fits stopped early and over-predicted the last "
            "balls of an innings.",
        ],
        [
            Bold("Legal_Balls-left calibration of M2 (design decision DD-15). "),
            "Each M2 fit's prediction is multiplied by a non-decreasing factor of Legal_Balls left, the isotonic ratio "
            "of target to prediction over the fit's own Training_Rows. The factors are "
            f"{_span(factors.loc[factors['lb_left'] == low, 'factor'], 2)} with {low} Legal_Ball{'s' if low != 1 else ''} "
            f"left and {_span(factors.loc[factors['lb_left'] == high, 'factor'], 2)} with {high}. With {low} left, the "
            "held-out first-innings mean prediction is "
            f"{_fixed(m2_value(f'mean_pred_lb_left_{low}', 'first_innings', 1), 2)} runs against "
            f"{_fixed(m2_value(f'mean_target_lb_left_{low}', 'first_innings', 1), 2)} scored "
            f"({_fixed(m2_value(f'mean_pred_raw_lb_left_{low}', 'first_innings', 1), 2)} before calibration), and the "
            "pooled RMSE " + ("falls in every variant and innings." if improved else "does not fall in every variant."),
        ],
        [
            Bold("Par_Score bias correction (DD-7). "),
            "Par_Score runs the first-innings M2 fit on a typical state at each ball count, then shifts the curve by the "
            "fit's mean residual in each Ball_Position_Share bucket of its own Training_Rows and keeps it "
            f"non-decreasing. The shifts range from {_signed(par['beta'].min(), 1)} to {_signed(par['beta'].max(), 1)} "
            "runs across fits and buckets.",
        ],
        [
            Bold("Centred isotonic calibration for M3 (DD-14). "),
            "Plain isotonic steps gave many balls the same WP after a dot ball, so their LI was 0. Joining the isotonic "
            "blocks at their centres keeps the fit but makes WP rise between blocks, and the bounds ",
            _const(inputs, "models.m3.calibration_bounds", lambda bounds: f"{_shown(bounds[0])} to {_shown(bounds[1])}"),
            " stop one comeback match from costing an unbounded log loss.",
        ],
        [
            Bold("First-innings end states. "),
            "When an outcome ends the first innings, its WP comes from the first-innings model at the end-of-innings "
            "state, so the outcomes of one ball are never scored by two models; "
            f"{refs.get('break_gap', 'the innings-break table of Section 2.2')} shows how far the two models disagree "
            "at the break.",
        ],
        [
            Bold("Average_Skill is weighted by balls. "),
            "The typical player of an M1 fit is the mean θ_bat of the strikers of its training Legal_Balls, weighted by "
            "balls faced, and the mean θ_bowl of their bowlers, weighted by balls bowled, so the average player is the "
            "one who faces and bowls the balls.",
        ],
        [
            Bold("Team_Strength is 11 times the mean. "),
            "Team_Strength is 11 times the mean θ_bat + θ_bowl of a side's players with is_played true, so a side that "
            "lists an Impact Player (12 players) is compared on the same 11-player scale as a side of 11.",
        ],
        [
            Bold("Shrinkage anchored on the usual level (DD-16). "),
            "Shrinking Clutch_Level straight towards Replacement_Level put every batter at Replacement_Level, because "
            f"the between-batter variance of Clutch_Level is {_fixed(summary['tau2_clutch_raw'], 3)} before the floor. "
            "The usual level carries the stable signal (τ² ", *_pair(inputs, "rating.tau2_usual"), "), so the shrunk "
            "Clutch_Level is the shrunk usual level plus the shrunk Pressure_Gap.",
        ],
        [
            Bold("Bootstrap with fixed Role-level values. "),
            "The bootstrap holds Replacement_Level, τ², the reliabilities and the Blend_Weights at the main ranking's "
            "values (Section 2.4), so P_Top10 is conditional on them.",
        ],
    ]
    for parts in notes:
        s.add_bullet(*parts)


# ------------------------------------------------------------------------------ Section 3

def section_3(builder: ReportBuilder, inputs: ReportInputs) -> None:
    """Section 3 Fair comparisons: matched baseline, opposition, context, eligibility, IPL and T20I (27.4)."""
    s = builder.section(3)
    _, remaining_runs, win_prob, outcome, extras = _model_modules()

    def users(feature: str) -> list[str]:
        """The models whose feature lists hold ``feature``; M3 is named once when both innings use it."""
        m3 = [innings for innings in (1, 2) if feature in win_prob.M3_FEATURES[innings]]
        named = ["M2"] if feature in remaining_runs.M2_FEATURES else []
        if len(m3) == 2:
            named.append("M3 in both innings")
        elif m3:
            named.append(f"the {'first' if m3[0] == 1 else 'second'}-innings M3")
        named += ["M4"] if feature in outcome.M4_FEATURES else []
        named += ["the Extras_Model"] if feature in extras.EXTRAS_FEATURES else []
        return named

    s.add_heading("3.1 Matched usual baseline")
    unmatched, totals, largest = {}, {}, 0.0
    for role in ROLES:
        estimates = inputs.frame(f"estimates_{role}")
        eligible = estimates[estimates["eligible"].astype(str).str.lower().eq("true")]
        shares = eligible["unmatched_share"].astype(float)
        unmatched[role], totals[role] = int((shares > 0.0).sum()), len(eligible)
        largest = max(largest, float(shares.max()) if len(shares) else 0.0)
    s.add_paragraph(
        "A player is compared with himself. His Usual_Baseline uses only his own Ordinary_Balls, in the same ",
        _const(inputs, "design_choices.D13_cells", lambda cells: _CELL_TEXT.get(str(cells), str(cells))),
        " cells as his Pressure_Balls and reweighted to their mix (Section 2.4), so a powerplay specialist is "
        "measured against his own powerplay balls, not against all of his balls. Pressure weight in a cell where the "
        "player has no ordinary ball is left out of the baseline; this applies to "
        f"{unmatched['batter']} of the {totals['batter']} eligible batters and {unmatched['bowler']} of the "
        f"{totals['bowler']} eligible bowlers, for at most {_pct(largest)} of a player's pressure weight.",
    )  # fmt: skip

    s.add_heading("3.2 Opposition strength")
    with_tsg = users("tsg")
    s.add_paragraph(
        "Opposition strength enters once, through the Expected_Value: a batter's ball is compared with an "
        "Average_Skill batter facing the actual bowler's θ_bowl, and a bowler's ball with an Average_Skill bowler "
        "bowling to the actual striker's θ_bat (Section 2.1). No other opposition weight, multiplier or matching cell "
        "is applied to NBV or the ratings; the wicket cost Ω depends on the batting side's own state, which is the "
        "value of the wicket, not an opposition weight. No team rating such as Elo is used, because the two Pools "
        "share no teams. Team_Strength, built from the θ values of the playing XI, enters only "
        f"{_and(with_tsg) or 'no model'} through the Team_Strength_Gap."
    )

    s.add_heading("3.3 Scoring environment, venue and match-day conditions")
    context = [f"{_feature(feature)} in {_and(users(feature))}" for feature in ("scoring_env", "venue_effect", "match_day")]
    s.add_paragraph(
        "M1 estimates a Scoring_Environment effect for each Competition_Type × Season, with each season string a "
        "Season of its own, so every IPL season and every T20I season gets its own scoring level. The Venue_Effect "
        "is the venue's own effect plus its country's, so a ground with few balls leans on its country. "
        "Match_Day_Conditions is this match's actual run rate on its earlier balls minus the M1 expectation for "
        "them, in runs per over. The models use them as features: "
        f"{'; '.join(context)}. So every expected value, and so every NBV, already allows for the scoring level of "
        "the competition, the season, the ground and the day."
    )

    s.add_heading("3.4 Eligibility")
    _eligibility(s, inputs)

    s.add_heading("3.5 IPL and T20I data")
    full_members = [str(name) for name in inputs.settings.get("data_rules.full_members")]
    s.add_paragraph(
        "Competition_Type is IPL for a competition whose name starts with IPL, T20I for a match_format of T20I, and "
        "A_Team for the other T20 competitions. The IPL Pool holds the IPL matches and the international Pool the "
        "T20I and A_Team matches. The two Pools share no teams, so they are linked only through players who appear in "
        "both; M1 estimates every player's skill over both Pools at once, and the Scoring_Environment effects absorb "
        f"the difference in scoring level. A Full_Member_Match is a T20I between two of {_and(full_members)}; an "
        "Associate_Match is any other match outside the IPL."
    )
    pools = ("batters_in_ipl_pool", "batters_in_international_pool", "batters_in_both_pools",
             "bowlers_in_ipl_pool", "bowlers_in_international_pool", "bowlers_in_both_pools")  # fmt: skip
    _table(
        s, inputs, "s3_pool_counts", "m1_metrics",
        ["metric", "value"],
        key=["metric"],
        decimals={"value": 0},
        headers={"value": "Players"},
        where={"metric": pools},
        caption="Players with at least one training Legal_Ball in each Pool and in both Pools",
    )  # fmt: skip
    m1 = inputs.frame("m1_metrics").set_index("metric")["value"]
    s.add_paragraph(
        f"The link between the Pools rests on the {_count(m1['batters_in_both_pools'])} batters and "
        f"{_count(m1['bowlers_in_both_pools'])} bowlers who have balls in both. The next tables give the share of "
        "each listed player's Pressure_Balls from IPL matches, Full_Member_Matches and Associate_Matches, unweighted."
    )
    lowest = []
    for role in ROLES:
        _table(
            s, inputs, f"s3_shares_{_PLURAL[role]}", f"top10_{role}",
            ["rank", "player_name", "share_ipl", "share_full_member", "share_associate"],
            key=["rank"],
            decimals={"share_ipl": 3, "share_full_member": 3, "share_associate": 3},
            headers={"share_ipl": "IPL share", "share_full_member": "Full_Member_Match share",
                     "share_associate": "Associate_Match share"},
            caption=f"Top 10 {_PLURAL[role]}: shares of Pressure_Balls by match type",
        )  # fmt: skip
        top = inputs.frame(f"top10_{role}")
        lowest.append(float((top["share_ipl"] + top["share_full_member"]).min()))
    s.add_paragraph(
        f"Every listed player has at least {_pct(min(lowest))} of his Pressure_Balls from IPL matches or "
        "Full_Member_Matches, so the lists rest on the strongest cricket in the data."
    )


def _eligibility(s: Section, inputs: ReportInputs) -> None:
    rule_on = bool(inputs.settings.get("data_rules.recent_activity_rule"))
    s.add_paragraph("A player must meet every rule below in a Role to be ranked in it; all counts are unweighted.")
    rules: list[list[Part]] = [
        [
            "At least ", _const(inputs, "rating.min_pressure_balls.batter"), " Pressure_Balls for batters and ",
            _const(inputs, "rating.min_pressure_balls.bowler"), " for bowlers: for each Role, the largest value of the "
            "grid ", _const(inputs, "design_choices.D11_min_balls_grid"), " that leaves at least ",
            _const(inputs, "design_choices.D11_target_eligible"), " Eligible_Players.",
        ],
        [
            "At least ", _const(inputs, "design_choices.D18_min_matches_default"), " Pressure_Matches; the grid below "
            "also tries ", _const(inputs, "design_choices.D18_min_matches_grid"), ".",
        ],
        ["Balls in the Role in at least ", _const(inputs, "data_rules.min_calendar_years"), " Calendar_Years."],
        [
            "At least ", _const(inputs, "data_rules.min_major_share", lambda share: f"{_shown(100.0 * float(share))}%"),
            " of the Pressure_Balls from IPL matches or Full_Member_Matches.",
        ],
        [
            Bold("Recent_Activity_Rule, a confirmed assumption: "), "at least one match with is_played true in ",
            _const(inputs, "data_rules.recent_years", lambda years: " or ".join(str(year) for year in years)),
            ". The problem statement does not ask for this rule; it keeps the lists to players who are still active "
            f"for the next IPL auction. It is {'on' if rule_on else 'off'} in this Run.",
        ],
    ]  # fmt: skip
    for parts in rules:
        s.add_bullet(*parts)
    report = inputs.frame("eligibility_report")
    _table(
        s, inputs, "s3_eligibility_report", "eligibility_report",
        ["role", "min_pressure_balls", "min_pressure_matches", "recent_rule_on", "n_players", "n_eligible",
         "fail_min_pressure_balls", "fail_min_pressure_matches", "fail_calendar_years", "fail_recent_activity",
         "fail_major_share"],
        key=["role"],
        decimals={"min_pressure_balls": 0, "min_pressure_matches": 0, "n_players": 0, "n_eligible": 0,
                  "fail_min_pressure_balls": 0, "fail_min_pressure_matches": 0, "fail_calendar_years": 0,
                  "fail_recent_activity": 0, "fail_major_share": 0},
        headers={"min_pressure_balls": "Minimum Pressure_Balls", "min_pressure_matches": "Minimum Pressure_Matches",
                 "recent_rule_on": "Recent_Activity_Rule on", "n_players": "Players with balls",
                 "n_eligible": "Eligible_Players", "fail_min_pressure_balls": "Fail Pressure_Balls",
                 "fail_min_pressure_matches": "Fail Pressure_Matches", "fail_calendar_years": "Fail Calendar_Years",
                 "fail_recent_activity": "Fail Recent_Activity_Rule", "fail_major_share": "Fail IPL or Full_Member share"},
        caption="Eligibility by Role: the rules applied and the players who fail each rule (a player who fails "
        "several rules counts under each)",
    )  # fmt: skip
    counts = [
        f"{_count(row['n_eligible'])} of the {_count(row['n_players'])} {row['role']}s with a ball in the Role"
        for _, row in report.iterrows()
    ]
    s.add_paragraph(
        f"{_and(counts)} are eligible. The grid shows how the counts move with the two minimums; the selected "
        "minimum Pressure_Ball count is applied with the default minimum Pressure_Match count."
    )
    _table(
        s, inputs, "s3_eligibility_grid", "eligibility_grid",
        ["role", "min_pressure_balls", "min_pressure_matches", "n_eligible"],
        key=["role", "min_pressure_balls", "min_pressure_matches"],
        decimals={"min_pressure_balls": 0, "min_pressure_matches": 0, "n_eligible": 0},
        headers={"min_pressure_balls": "Minimum Pressure_Balls", "min_pressure_matches": "Minimum Pressure_Matches",
                 "n_eligible": "Eligible_Players"},
        caption="Eligibility grid: Eligible_Players at each pair of minimum counts, with the other rules applied",
    )  # fmt: skip

    # the Activity_Comparison_Ranking (Requirement 25, criterion 14)
    if not rule_on or not all(inputs.has(name) for name in ACTIVITY_INPUTS):
        s.add_paragraph(
            "The Recent_Activity_Rule is off in this Run, so no comparison ranking with the rule off is computed."
            if not rule_on
            else "This Run's outputs hold no activity comparison tables, so the comparison with the rule off is "
            "not shown."
        )
        return
    sentences = []
    for role in ROLES:
        table = inputs.frame(f"activity_{role}")
        summary = _row(table, f"activity_comparison_{_PLURAL[role]}.csv", kind="summary")
        main = inputs.frame(f"top10_{role}")["player_id"].astype(int).tolist()
        other = inputs.frame(f"comparison_list_{role}")["player_id"].astype(int).tolist()
        moved = sum(1 for a, b in zip(main, other) if a != b)
        order = (
            "in the same order"
            if main == other
            else f"with {moved} of the 10 places held by a different player" if set(main) != set(other)
            else f"with {moved} of them in a different place"
        )
        changes = table[table["kind"].astype(str) != "summary"]
        moves = _and(
            f"{row['player_name']} {row['kind']}"
            + (f" (last played in {int(row['latest_calendar_year'])})" if pd.notna(row["latest_calendar_year"]) else "")
            for _, row in changes.iterrows()
        )
        sentences.append(
            f"the {role} top 10 keeps {_count(summary['overlap'])} of its players {order}"
            + (f" ({moves})" if moves else "")
        )
    main_counts = {
        role: _count(
            _row(inputs.frame(f"activity_{role}"), f"activity_comparison_{_PLURAL[role]}.csv", kind="summary")["n_eligible_main"]
        )
        for role in ROLES
    }
    s.add_paragraph(
        "With the Recent_Activity_Rule off, every other rule and setting unchanged, the eligible pool grows from ",
        f"{main_counts['batter']} to ", _const(inputs, "activity_comparison.n_eligible.batter"), " batters and from ",
        f"{main_counts['bowler']} to ", _const(inputs, "activity_comparison.n_eligible.bowler"),
        f" bowlers, and {_and(sentences)}. These comparison lists are never submitted.",
    )  # fmt: skip
    for role in ROLES:
        _table(
            s, inputs, f"s3_activity_{_PLURAL[role]}", f"activity_{role}",
            ["kind", "n_eligible_main", "n_eligible_comparison", "overlap", "player_id", "player_name", "main_rank",
             "comparison_rank", "latest_calendar_year"],
            key=["kind", "player_id"],
            headers={"kind": "Row", "n_eligible_main": "Eligible, rule on", "n_eligible_comparison": "Eligible, rule off",
                     "overlap": "Top-10 overlap", "main_rank": "Rank, rule on", "comparison_rank": "Rank, rule off",
                     "latest_calendar_year": "Last Calendar_Year played"},
            caption=f"Recent_Activity_Rule on against off, {_PLURAL[role]}: Eligible_Player counts, top-10 overlap and "
            "the players who enter or leave the top 10",
        )  # fmt: skip


# ------------------------------------------------------------------------------ Section 4

_TRADITIONAL: Mapping[str, tuple[tuple[str, str, int], ...]] = MappingProxyType(
    {
        "batter": (
            ("deliveries", "Balls", 0), ("runs", "Runs", 0), ("dismissals", "Outs", 0),
            ("strike_rate", "Strike rate", 1), ("average", "Average", 1), ("dot_pct", "Dot %", 1),
            ("boundary_pct", "Boundary %", 1),
        ),
        "bowler": (
            ("deliveries", "Balls", 0), ("legal_balls", "Legal_Balls", 0), ("wickets", "Wickets", 0),
            ("economy", "Economy", 2), ("balls_per_wicket", "Balls per wicket", 1), ("dot_pct", "Dot %", 1),
            ("boundary_pct", "Boundary %", 1), ("extras_per_100", "Wides + No_Balls per 100", 1),
        ),
    }
)  # fmt: skip


def section_4(builder: ReportBuilder, inputs: ReportInputs) -> None:
    """Section 4 Evidence for the top 10: every evidence table and chart of Requirement 26 (27.5)."""
    s = builder.section(4)
    s.add_paragraph(
        "This section gives the evidence for every listed player. The traditional statistics count the player's "
        "Pressure_Balls and Ordinary_Balls in the Role without weights and without super overs, and an empty cell "
        "has a zero denominator. The rating values are in runs per over, and the mix tables show where each player's "
        "Pressure_Balls came from."
    )
    for number, role in enumerate(ROLES, start=1):
        s.add_heading(f"4.{number} {_TITLE[role]}")
        _evidence(s, inputs, role)


def _evidence(s: Section, inputs: ReportInputs, role: str) -> None:
    plural = _PLURAL[role]
    stats = _TRADITIONAL[role]
    for ball_set, words in (("pressure", "Pressure_Balls"), ("ordinary", "Ordinary_Balls")):
        columns = [f"{ball_set}_{name}" for name, _, _ in stats]
        _table(
            s, inputs, f"s4_traditional_{plural}_{ball_set}", f"traditional_{role}",
            ["rank", "player_name", *columns],
            key=["rank"],
            decimals={f"{ball_set}_{name}": decimals for name, _, decimals in stats},
            headers={f"{ball_set}_{name}": header for name, header, _ in stats},
            caption=f"Top 10 {plural}: traditional statistics on their {words}",
        )  # fmt: skip
    traditional = inputs.frame(f"traditional_{role}")
    rate, unit = ("strike_rate", "strike at") if role == "batter" else ("economy", "concede")
    decimals = 1 if role == "batter" else 2
    s.add_paragraph(
        f"The ten {plural} {unit} {_span(traditional[f'pressure_{rate}'].dropna(), decimals)}"
        + (" runs per over" if role == "bowler" else "")
        + f" on Pressure_Balls and {_span(traditional[f'ordinary_{rate}'].dropna(), decimals)} on Ordinary_Balls. "
        "These raw rates do not allow for the match state, the opponent or the scoring level; NBV does."
    )
    _table(
        s, inputs, f"s4_rating_{plural}", f"evidence_{role}",
        ["rank", "player_id", "player_name", "final_score", "clutch", "usual", "uplift", "shrunk_clutch",
         "shrunk_uplift"],
        key=["rank"],
        decimals={"final_score": 3, "clutch": 3, "usual": 3, "uplift": 3, "shrunk_clutch": 3, "shrunk_uplift": 3},
        headers={"clutch": "Clutch_Level", "usual": "Usual_Baseline", "uplift": "Uplift",
                 "shrunk_clutch": "Shrunk Clutch_Level", "shrunk_uplift": "Shrunk Uplift"},
        caption=f"Top 10 {plural}: rating values in runs per over",
    )  # fmt: skip
    _table(
        s, inputs, f"s4_uncertainty_{plural}", f"evidence_{role}",
        ["rank", "player_name", "ci_low", "ci_high", "p_top10", "hit_rate", "worst_quartile", "n_pressure_matches"],
        key=["rank"],
        decimals={"ci_low": 3, "ci_high": 3, "p_top10": 4, "hit_rate": 3, "worst_quartile": 3, "n_pressure_matches": 0},
        headers={"worst_quartile": "Worst_Quartile_Mean", "n_pressure_matches": "Pressure_Matches"},
        caption=f"Top 10 {plural}: 95% bootstrap interval of Final_Score, P_Top10, Hit_Rate, Worst_Quartile_Mean and "
        "Pressure_Matches",
    )  # fmt: skip
    s.add_paragraph(
        "Final_Score = w_L × shrunk Clutch_Level + w_U × shrunk Uplift, with w_L = ",
        _const(inputs, f"rating.w_L.{role}", lambda value: _fixed(value, 3)), " and w_U = ",
        _const(inputs, f"rating.w_U.{role}", lambda value: _fixed(value, 3)), f" for {plural} (Section 2.4).",
    )  # fmt: skip
    _table(
        s, inputs, f"s4_mix_{plural}", f"mix_{role}",
        ["rank", "player_name", "n_pressure_balls", "pct_powerplay", "pct_middle", "pct_death", "pct_ipl", "pct_t20i",
         "pct_a_team"],
        key=["rank"],
        decimals={"n_pressure_balls": 0, "pct_powerplay": 1, "pct_middle": 1, "pct_death": 1, "pct_ipl": 1,
                  "pct_t20i": 1, "pct_a_team": 1},
        headers={"n_pressure_balls": "Pressure_Balls", "pct_powerplay": "Powerplay %", "pct_middle": "Middle %",
                 "pct_death": "Death %", "pct_ipl": "IPL %", "pct_t20i": "T20I %", "pct_a_team": "A_Team %"},
        caption=f"Top 10 {plural}: Pressure_Balls by Phase and Competition_Type, in percent",
    )  # fmt: skip
    _figure(
        s, inputs, f"forest_chart_{role}",
        f"Top 10 {plural}: Final_Score (point) and its 95% bootstrap interval (line), in runs per over", _NARROW_CM,
    )
    _figure(
        s, inputs, f"clutch_chart_{role}",
        f"{_TITLE[role]}: Clutch_Level against Usual_Baseline of every Eligible_Player, in runs per over, with the top "
        "10 labelled", _NARROW_CM,
    )
    _figure(s, inputs, f"phase_chart_{role}", f"Top 10 {plural}: share of each player's Pressure_Balls by Phase", _NARROW_CM)
    _figure(
        s, inputs, f"competition_chart_{role}",
        f"Top 10 {plural}: share of each player's Pressure_Balls by Competition_Type", _NARROW_CM,
    )


# ------------------------------------------------------------------------------ Section 5

#: The Section 5 sub-heading that holds the robustness stage's results (Requirements 20-22; task 27.1).
ROBUSTNESS_HEADING = "5.3 Repeatability, sensitivity and cross-checks"

#: The robustness inputs that Section 5.3 shows, all written together by the ``robustness`` stage; when
#: any is missing, 5.3 says the robustness stage has not run and shows no result (OPTIONAL_OUTPUT_FILES).
_ROBUSTNESS_INPUTS: tuple[str, ...] = (
    "robustness_table", "split_half_reliability", "period_spearman", "v1_ipl", "v1_t20i", "competition_spearman",
)

#: How many players ranked below each top 10 Section 5.2 shows from the bootstrap files.
_NEXT_RANKS = 5

#: The Appendix sub-sections, which Section 5 refers to by number.
_AUDIT_PART, _SCORE_PART, _CHOICES_PART = "A.1", "A.2", "A.3"

#: Plain names of the Score_Checks (``superr.reconstruct.SCORE_CHECKS``).
_CHECK_NAMES: Mapping[str, str] = MappingProxyType(
    {"runs_margin": "runs-margin", "chase": "chase", "wickets_margin": "wickets-margin", "tie": "tie", "over": "over"}
)


def _number(value: Any) -> str:
    """A count or a median: "173", "1,456", or with one decimal when it is not whole ("168.5")."""
    number = float(value)
    return f"{int(number):,}" if number.is_integer() else f"{number:,.1f}"


def _number_span(values: Iterable[Any]) -> str:
    """The smallest and largest of ``values`` through ``_number``, or one value when they agree."""
    numbers = [float(value) for value in values]
    low, high = _number(min(numbers)), _number(max(numbers))
    return low if low == high else f"{low} to {high}"


def _or(items: Iterable[Any]) -> str:
    """Items joined as "a, b or c"."""
    texts = [str(item) for item in items]
    return texts[0] if len(texts) == 1 else ", ".join(texts[:-1]) + " or " + texts[-1]


def _truthy(values: pd.Series) -> pd.Series:
    """True where a CSV column holds true (``True``, ``true`` or ``1``), as a bool Series."""
    return values.astype(str).str.strip().str.lower().isin(("true", "1"))


def _eligible(inputs: ReportInputs, role: str) -> pd.DataFrame:
    """The Eligible_Players' rows of a Role's player-estimates file."""
    estimates = inputs.frame(f"estimates_{role}")
    return estimates[_truthy(estimates["eligible"])]


def section_5(builder: ReportBuilder, inputs: ReportInputs) -> None:
    """Section 5 Reliability and limitations: split-half reliability, the bootstrap, the place of the robustness
    results and the limitations list (27.6, 27.7)."""
    s = builder.section(5)
    s.add_paragraph(
        "This section asks how far the lists can be trusted: whether a player's pressure values repeat from one half "
        "of his matches to the other (5.1), how much the lists move when the matches are resampled (5.2), whether "
        "they repeat over time and across competitions and how fragile they are to each component and hand-set "
        "choice (5.3), and what the data and the method cannot show (5.4)."
    )
    s.add_heading("5.1 Split-half reliability")
    _split_half(s, inputs)
    s.add_heading("5.2 Bootstrap intervals and P_Top10")
    _bootstrap_results(s, inputs)
    s.add_heading(ROBUSTNESS_HEADING)
    _robustness_results(s, inputs)
    s.add_heading("5.4 Limitations")
    for parts in _limitations(inputs):
        s.add_bullet(*parts)


def _split_half(s: Section, inputs: ReportInputs) -> None:
    summary = inputs.frame("role_summary")
    s.add_paragraph(
        "Each player's matches in the Role are numbered by date, and his Clutch_Level and Uplift are estimated again "
        "from his odd-numbered and from his even-numbered matches. r is the correlation between the two halves "
        "across the Role's Eligible_Players, and the Split_Half_Reliability ρ = 2r ÷ (1 + r) estimates the "
        "reliability of the full-sample value: roughly the share of the spread between players that is stable "
        "rather than noise. A negative ρ, which arises when the true reliability is near 0, is set to 0 before the "
        "Blend_Weights are computed."
    )
    for role in ROLES:
        row = _row(summary, "role_summary.csv", role=role)
        raw = {"L": float(row["rho_L_raw"]), "U": float(row["rho_U_raw"])}
        parts: list[Part] = [
            Bold(f"{_TITLE[role]}, {_count(row['n_rho'])} players. "),
            f"r is {_fixed(row['r_L'], 3)} for Clutch_Level and {_fixed(row['r_U'], 3)} for Uplift, so ρ is "
            f"{_fixed(raw['L'], 3)} and {_fixed(raw['U'], 3)} before the floor, and ρ_L = ",
            _role_value(inputs, "rho_L", role), " and ρ_U = ", _role_value(inputs, "rho_U", role), " after it. ",
        ]  # fmt: skip
        weights: list[Part] = [
            "w_L = ", _role_value(inputs, "w_L", role), " and w_U = ", _role_value(inputs, "w_U", role),
        ]  # fmt: skip
        if raw["L"] <= 0.0 and raw["U"] <= 0.0:
            parts += [
                f"Neither value repeats from one half of a {role}'s matches to the other, so the Blend_Weights fall "
                "back to ", *weights, f": the {role} Final_Score is the shrunk Clutch_Level alone.",
            ]  # fmt: skip
        else:
            if raw["L"] > 0.0 and raw["U"] > 0.0:
                stable = (
                    f"about {_pct(raw['L'], 0)} of the spread in Clutch_Level and {_pct(raw['U'], 0)} of the spread "
                    "in Uplift are stable"
                )
            elif raw["L"] > 0.0:
                stable = f"about {_pct(raw['L'], 0)} of the spread in Clutch_Level is stable and Uplift does not repeat"
            else:
                stable = f"Clutch_Level does not repeat and about {_pct(raw['U'], 0)} of the spread in Uplift is stable"
            parts += [f"For {_PLURAL[role]}, {stable}, so the Blend_Weights are ", *weights, "."]
        s.add_bullet(*parts)
    s.add_paragraph(
        "Players' usual levels do differ beyond noise: the between-player variance τ² of the Usual_Baseline is ",
        *_pair(inputs, "rating.tau2_usual"), ", against a Pressure_Gap τ² of ", *_pair(inputs, "rating.tau2_gap"),
        " (Section 2.4). This is why each shrunk Clutch_Level is built on the player's shrunk usual level.",
    )  # fmt: skip


def _bootstrap_results(s: Section, inputs: ReportInputs) -> None:
    unranked = {role: int(inputs.value(f"bootstrap.n_ever_unranked.{role}")) for role in ROLES}
    warnings = inputs.frame("bootstrap_warnings")
    if warnings.empty:
        warned = "No resample raised a rating warning."
    else:
        warned = (
            "The resamples raised these rating warnings: "
            + _and(
                f"{row['kind']} ({row['role']}"
                + (f", {row['estimate']}" if pd.notna(row["estimate"]) and str(row["estimate"]).strip() else "")
                + f") in {_count(row['n_resamples'])} resamples"
                for _, row in warnings.iterrows()
            )
            + " (outputs/robustness/bootstrap_warnings.csv)."
        )
    s.add_paragraph(
        "The bootstrap draws the ", _const(inputs, "bootstrap.n_resampled_matches", _count), " matches with "
        "replacement ", _n_boot(inputs), " times, from the seed ", _const(inputs, "seeds.bootstrap"), " in config.py, "
        "and ranks both Roles in each resample with the Role-level values held fixed (Section 2.4). The number of "
        "Eligible_Players unranked in at least one resample is ", _const(inputs, "bootstrap.n_ever_unranked.batter"),
        " for batters and ", _const(inputs, "bootstrap.n_ever_unranked.bowler"), " for bowlers"
        + (", so every resample ranks every player. " if not any(unranked.values())
           else "; an unranked player counts as outside the top 10 in that resample. ")
        + warned,
    )  # fmt: skip

    widths, overlapping, secure, close = [], [], [], []
    for role in ROLES:
        top = inputs.frame(f"top10_{role}").sort_values("rank")
        low, high = top["ci_low"].to_numpy(dtype=float), top["ci_high"].to_numpy(dtype=float)
        widths.append(_span(high - low, 3))
        overlapping.append(bool(np.all((low[:-1] <= high[1:]) & (low[1:] <= high[:-1]))))
        p_top10 = top["p_top10"].astype(float)
        sure, unsure = top[p_top10 >= 0.5], top[p_top10 < 0.5]
        if len(sure):
            secure.append(f"{role} rank{'s' if len(sure) > 1 else ''} {_ranks(sure['rank'])}")
        if len(unsure):
            close.append(
                f"{_span(unsure['p_top10'], 4)} at {role} rank{'s' if len(unsure) > 1 else ''} {_ranks(unsure['rank'])} "
                f"(median bootstrap rank{'s' if len(unsure) > 1 else ''} {_number_span(unsure['median_rank'])})"
            )
    s.add_paragraph(
        f"The 95% intervals of the listed {_PLURAL[ROLES[0]]} are {widths[0]} runs per over wide, and those of the "
        f"listed {_PLURAL[ROLES[1]]} {widths[1]}"
        + (". Each listed player's interval overlaps the next one's, so neighbouring places are not separated. "
           if all(overlapping) else ". ")
        + (f"P_Top10 is at least one half at {_and(secure)}. " if secure else "")
        + (f"Below them the places are close calls: P_Top10 is {_and(close)}." if close
           else "Every listed player has a P_Top10 of at least one half.")
    )  # fmt: skip

    def p_value(value: Any) -> str:
        return "none" if value is None else _fixed(value, 4)

    reproduced = [int(inputs.value(f"bootstrap.top10_set_reproduced.{role}")) for role in ROLES]
    above = [
        role
        for role in ROLES
        if inputs.value(f"bootstrap.max_p_top10_outside_top10.{role}") is not None
        and float(inputs.value(f"bootstrap.max_p_top10_outside_top10.{role}"))
        > float(inputs.value(f"bootstrap.min_p_top10_main_top10.{role}"))
    ]
    s.add_paragraph(
        "No resample reproduces either main top-10 set exactly: " if not any(reproduced) else "",
        "the number of resamples whose top 10 holds the same players as the main list is ",
        _const(inputs, "bootstrap.top10_set_reproduced.batter"), " for batters and ",
        _const(inputs, "bootstrap.top10_set_reproduced.bowler"), " for bowlers. The lowest P_Top10 inside the top 10 is ",
        _const(inputs, "bootstrap.min_p_top10_main_top10.batter", p_value), " for batters and ",
        _const(inputs, "bootstrap.min_p_top10_main_top10.bowler", p_value), " for bowlers, and the highest outside it is ",
        _const(inputs, "bootstrap.max_p_top10_outside_top10.batter", p_value), " and ",
        _const(inputs, "bootstrap.max_p_top10_outside_top10.bowler", p_value), ".",
        (f" An unlisted {_or(above)} reaches the top 10 more often than the lowest listed one: the lists follow the "
         "full-data Final_Score, and P_Top10 shows how far to trust each place." if above else ""),
    )  # fmt: skip

    refs, reach = [], []
    for role in ROLES:
        boot = inputs.frame(f"bootstrap_{role}")
        ranks = [int(rank) for rank in boot["main_rank"] if 10 < int(rank) <= 10 + _NEXT_RANKS]
        if not ranks:
            s.add_paragraph(f"Every eligible {role} is in the top 10, so no {role} ranks below it.")
            continue
        last = _row(boot, f"bootstrap_{role}.csv", main_rank=10)
        below = boot[boot["main_rank"].astype(int).isin(ranks)]
        reach.append(
            bool(((below["ci_low"] <= float(last["ci_high"])) & (below["ci_high"] >= float(last["ci_low"]))).all())
        )
        refs.append(
            _table(
                s, inputs, f"s5_next_{_PLURAL[role]}", f"bootstrap_{role}",
                ["main_rank", "player_name", "main_final_score", "ci_low", "ci_high", "median_rank", "p_top10"],
                key=["main_rank"],
                decimals={"main_rank": 0, "main_final_score": 3, "ci_low": 3, "ci_high": 3, "median_rank": 1,
                          "p_top10": 4},
                headers={"main_rank": "Rank", "main_final_score": "Final_Score", "median_rank": "Median bootstrap rank"},
                where={"main_rank": tuple(str(rank) for rank in ranks)},
                caption=f"The {len(ranks)} {_PLURAL[role]} ranked just below the top 10: Final_Score in runs per over, "
                "95% bootstrap interval, median bootstrap rank and P_Top10",
            )
        )  # fmt: skip
    if refs:
        s.add_paragraph(
            f"{_and(refs)} show the players ranked just below each list. Each of their intervals overlaps the "
            "interval of the player ranked 10th, so these players are close to the lists."
            if all(reach) else f"{_and(refs)} show the players ranked just below each list."
        )


def _robustness_results(s: Section, inputs: ReportInputs) -> None:
    """Section 5.3: the Requirement 20 repeatability, the Requirement 21 ablation and sensitivity table and the
    Requirement 22 cross-checks, or a note when the robustness stage has not run (Requirement 27, criterion 6)."""
    s.add_paragraph(
        "Three further checks test the lists beyond the resampling of Section 5.2: whether the ratings repeat "
        "between the Early_Period and the Late_Period and between IPL and T20I matches; how far the lists move when "
        "one component is removed or one Design_Choice is changed at a time; and how the lists compare with rankings "
        "by win probability added and by simple batting and bowling statistics. The period check uses the "
        "Eligible_Players with at least ",
        _const(inputs, "design_choices.D19_period_minimum"),
        " unweighted Pressure_Balls in each period, split at the end of 2024.",
    )  # fmt: skip
    if not all(inputs.has(name) for name in _ROBUSTNESS_INPUTS):
        s.add_paragraph(
            "The robustness stage has not run for this report, so this sub-section shows no result yet, and every "
            f"row of the Appendix Design_Choice table ({_CHOICES_PART}) states that no sensitivity case was run."
        )
        return
    _period_repeatability(s, inputs)
    _competition_repeatability(s, inputs)
    _ablation_sensitivity(s, inputs)
    _cross_checks(s, inputs)


def _period_repeatability(s: Section, inputs: ReportInputs) -> None:
    """Requirement 20, criteria 1-4: the before-floor Split_Half_Reliability and the Early-to-Late Spearman."""
    s.add_heading("Repeatability over time", level=3)
    s.add_paragraph(
        "The Robustness_Suite re-records the before-floor Split_Half_Reliability of Clutch_Level and Uplift for each "
        "Role (the ρ of Section 5.1 before its floor at 0), the correlation across the Role's Eligible_Players "
        "between the estimates from a player's odd-numbered and even-numbered matches:"
    )
    _table(
        s, inputs, "s5_split_half", "split_half_reliability",
        ["role", "estimate", "reliability", "n_players"],
        key=["role", "estimate"],
        decimals={"reliability": 3, "n_players": 0},
        headers={"estimate": "Estimate", "reliability": "Split_Half_Reliability before floor", "n_players": "Players"},
        caption="Split_Half_Reliability of Clutch_Level and Uplift by Role, before the zero floor (Requirement 20)",
    )  # fmt: skip
    s.add_paragraph(
        "The time check then splits each player's matches at the end of 2024 and recomputes his unshrunk "
        "Clutch_Level and Uplift from his Early_Period matches alone and from his Late_Period matches alone, with the "
        "models, Pressure_Ball flags, NBV, weights and Replacement_Level unchanged. The Spearman correlation of the "
        "two, across the Role's qualifying players, is:"
    )
    _table(
        s, inputs, "s5_period_spearman", "period_spearman",
        ["role", "estimate", "spearman", "n_players", "computed"],
        key=["role", "estimate"],
        decimals={"spearman": 3, "n_players": 0},
        headers={"estimate": "Estimate", "spearman": "Early-to-Late Spearman", "n_players": "Players",
                 "computed": "Computed"},
        caption="Early_Period-to-Late_Period Spearman correlation of Clutch_Level and Uplift by Role (Requirement 20)",
    )  # fmt: skip
    period = inputs.frame("period_spearman")
    computed = _truthy(period["computed"])
    not_computed = [role for role in ROLES if not computed[period["role"].astype(str) == role].all()]
    pairs = [
        f"{str(r['estimate']).replace('_', ' ')} {_fixed(r['spearman'], 3)} for {_PLURAL[r['role']]}"
        for _, r in period[computed].iterrows()
    ]
    if pairs:
        s.add_paragraph(
            "The Early-to-Late correlations are " + _and(pairs) + ", all close to 0: a player's pressure values in "
            "his earlier matches barely predict those in his later matches, which matches the split-half result and "
            "the Blend_Weights of Section 5.1. The lists name players who are reliable in pressure situations, not "
            "players the data shows to improve under pressure over time."
        )
    if not_computed:
        s.add_paragraph(
            f"Fewer than ten {_and(_PLURAL[role] for role in not_computed)} met the Period_Minimum in both periods, "
            "so their Early-to-Late correlations were not computed (Requirement 20, criterion 4)."
        )


def _competition_repeatability(s: Section, inputs: ReportInputs) -> None:
    """Requirement 20, criteria 5-6: the two V1 tables over IPL and over T20I matches and the IPL-to-T20I Spearman."""
    s.add_heading("Repeatability across competitions", level=3)
    s.add_paragraph(
        "The V1 check of Section 1.5 is repeated over the IPL matches and over the T20I matches separately (A_Team "
        "matches in neither), using the pooled V1 decile boundaries. If the Pressure_Index means the same thing in "
        "both, the mean Actual_WP_Swing of a ball should rise across the PI deciles the same way in each."
    )
    for table_id, source, comp in (("s5_v1_ipl", "v1_ipl", "IPL"), ("s5_v1_t20i", "v1_t20i", "T20I")):
        _table(
            s, inputs, table_id, source,
            ["role", "innings", "group", "decile", "n_deliveries", "pi_min", "pi_max", "mean_swing"],
            key=["role", "innings", "group", "decile"],
            decimals={"n_deliveries": 0, "pi_min": 1, "pi_max": 1, "mean_swing": 4},
            headers={"group": "Group", "decile": "Decile", "pi_min": "Lowest PI", "pi_max": "Highest PI",
                     "mean_swing": "Mean Actual_WP_Swing"},
            caption=f"V1 over {comp} matches: mean Actual_WP_Swing in each PI decile of the Role's Live_Balls, and on "
            "Dead_Balls (Requirement 20)",
        )  # fmt: skip
    comp_spear = inputs.frame("competition_spearman")
    _table(
        s, inputs, "s5_competition_spearman", "competition_spearman",
        ["role", "innings", "spearman", "n_deciles"],
        key=["role", "innings"],
        decimals={"spearman": 3, "n_deciles": 0},
        headers={"spearman": "IPL-to-T20I Spearman", "n_deciles": "Deciles"},
        caption="IPL-to-T20I Spearman correlation of the mean Actual_WP_Swing across PI deciles, by Role and innings "
        "(Requirement 20)",
    )  # fmt: skip
    s.add_paragraph(
        "The IPL and T20I decile means rank-correlate " + _span(comp_spear["spearman"], 3, " to ") + " across the "
        f"{int(comp_spear['n_deciles'].max())} deciles in each Role and innings, so the Pressure_Index orders balls "
        "by their effect on the match the same way in both competitions."
    )


def _ablation_sensitivity(s: Section, inputs: ReportInputs) -> None:
    """Requirement 21: the ablation and sensitivity cases, from the all_cases rows of the robustness table."""
    s.add_heading("Ablations and sensitivity", level=3)
    s.add_paragraph(
        "Each case below changes one component or one Design_Choice and reruns the Pressure_Engine, Value_Engine and "
        "Rating_Engine from the changed layer down, with no model refitted and no bootstrap. The table gives, per "
        "case and Role, the Spearman correlation of the case's Final_Scores with the main ranking's over the players "
        "eligible in both, the common-player count, the case's Eligible_Player count, the top-10 overlap, and the "
        "players entering and leaving the top 10. The cases at the Config default (α = 0.5, a 25% share, and so on) "
        "reproduce the main ranking exactly: Spearman 1.000 and overlap 10."
    )
    _table(
        s, inputs, "s5_ablation_sensitivity", "robustness_table",
        ["case", "target_id", "role", "n_common", "n_eligible_case", "spearman", "overlap", "entering_names",
         "leaving_names"],
        key=["case", "role"],
        decimals={"n_common": 0, "n_eligible_case": 0, "spearman": 3, "overlap": 0},
        headers={"case": "Case", "target_id": "Changed", "n_common": "Common players",
                 "n_eligible_case": "Eligible in case", "spearman": "Spearman", "overlap": "Top-10 overlap",
                 "entering_names": "Entering the top 10", "leaving_names": "Leaving the top 10"},
        where={"kind": ("ablation", "sensitivity", "zscore", "alternative")},
        caption="Ablation and sensitivity cases: Spearman correlation with the main ranking and top-10 overlap, by "
        "case and Role (Requirement 21)",
    )  # fmt: skip
    table = inputs.frame("robustness_table")
    cases = table[table["kind"].astype(str).isin(("ablation", "sensitivity", "zscore", "alternative"))].copy()
    cases["_sp"] = pd.to_numeric(cases["spearman"], errors="coerce")
    highlights = []
    for role in ROLES:
        rows = cases[cases["role"].astype(str) == role].dropna(subset=["_sp"])
        lowest = rows.loc[rows["_sp"].idxmin()]
        highlights.append(
            f"for {_PLURAL[role]} the Spearman stays between {_span(rows['_sp'], 3, ' and ')}, lowest for the "
            f"'{lowest['case']}' case (Spearman {_fixed(lowest['_sp'], 3)}, top-10 overlap {_count(lowest['overlap'])})"
        )
    s.add_paragraph("Across the cases " + _and(highlights) + ".")


def _cross_checks(s: Section, inputs: ReportInputs) -> None:
    """Requirement 22: the WPA and simple-statistic cross-check rankings, from the cross_check rows."""
    s.add_heading("Cross-checks against simpler methods", level=3)
    table = inputs.frame("robustness_table")
    cross = table[table["kind"].astype(str) == "cross_check"].copy()
    wpa = cross[cross["case"].astype(str) == "WPA"]
    wpa_text = _and(f"{_fixed(r['spearman'], 3)} for {_PLURAL[r['role']]}" for _, r in wpa.iterrows())
    s.add_paragraph(
        "Each Role's Eligible_Players are also ranked by simpler methods. The win-probability-added (WPA) ranking "
        "orders players by the unweighted WP swings summed over their Pressure_Balls; its Spearman rank correlation "
        f"with the main ranking is {wpa_text}. The strike-rate and batting-average rankings for batters, and the "
        "economy and bowling-strike-rate rankings for bowlers, use only each player's Simple_Rule_Balls with no "
        "weights, so they carry no match state. The table gives each ranking's top-10 overlap with the main ranking "
        "and the players entering and leaving the top 10."
    )
    _table(
        s, inputs, "s5_cross_checks", "robustness_table",
        ["case", "target_id", "role", "spearman", "overlap", "entering_names", "leaving_names"],
        key=["case", "role"],
        decimals={"spearman": 3, "overlap": 0},
        headers={"case": "Cross-check ranking", "target_id": "Requirement", "spearman": "Spearman",
                 "overlap": "Top-10 overlap", "entering_names": "Entering the top 10", "leaving_names": "Leaving the top 10"},
        where={"kind": "cross_check"},
        caption="Cross-check rankings against the main ranking: WPA, and the simple batting and bowling statistics "
        "(Requirement 22)",
    )  # fmt: skip
    overlaps = _and(f"{_count(r['overlap'])} for {r['case']} ({_PLURAL[r['role']]})" for _, r in cross.iterrows())
    s.add_paragraph(
        f"The top-10 overlaps with the main ranking are {overlaps}, all small: no single simple statistic reproduces "
        "the model's lists, and even WPA, which shares the win-probability model, keeps only a few names. The model "
        "adds information beyond runs, wickets and raw rates by valuing each ball against the match state."
    )


def _limitations(inputs: ReportInputs) -> list[list[Part]]:
    """The limitations list of Requirement 27, criterion 7, and the further items the outputs call for."""
    from superr.reconstruct import EXACT_COLUMNS, RUNS_CHECKS, SUPER_OVER_WIN, parse_result

    summary = inputs.frame("role_summary")
    items: list[list[Part]] = []

    comp_types = inputs.frame("fold_table")["comp_type"].astype(str).value_counts()
    international = _and(f"{_count(comp_types.get(kind, 0))} {kind}" for kind in ("T20I", "A_Team"))
    items.append([
        Bold("No T20 World Cup matches. "),
        "The data holds no T20 World Cup match, so no rating includes World Cup pressure; the international Pool has "
        f"{international} matches.",
    ])  # fmt: skip

    items.append([
        Bold("Impact Player timing is inferred. "),
        "playing_eleven.csv marks the Impact Player (is_impact) and the player he replaced (is_impact_off), but not "
        "when the swap happened. An Impact Player counts among the batters still to come only after his first ball "
        "in the match as batter, non-striker or bowler, so before that RBQ and the wicket cost can understate the "
        "batting left.",
    ])  # fmt: skip

    speed = inputs.frame("bowl_speed").iloc[0]
    negative = int(speed["negative_rows"])
    items.append([
        Bold("bowl_speed is sparse. "),
        f"bowl_speed is 0 on {_fixed(speed['zero_pct'], 1)}% of Deliveries"
        + (f" and below 0 on {_count(negative)} Deliver{'y' if negative == 1 else 'ies'}" if negative else "")
        + f", and its values above 0 run from {float(speed['positive_min']):,.1f} to {float(speed['positive_max']):,.1f}, "
        f"with 1st and 99th percentiles of {float(speed['positive_p01']):,.1f} and {float(speed['positive_p99']):,.1f}. "
        + ("No model uses it, and the Stage-2 factor F47 that would add it to M4 is off."
           if not bool(inputs.settings.get("stage2.F47"))
           else "The Stage-2 factor F47 adds it to M4, with zero and out-of-range values treated as missing."),
    ])  # fmt: skip

    bowlers = _eligible(inputs, "bowler")
    items.append([
        Bold("Bowler samples are small. "),
        "A bowler bowls at most four overs in a full T20 innings, so his evidence builds slowly: the median eligible "
        f"bowler has {_number(bowlers['n_pressure_balls'].median())} Pressure_Balls in "
        f"{_number(bowlers['n_pressure_matches'].median())} Pressure_Matches, and the match-clustered standard error "
        f"of his Clutch_Level is {_fixed(bowlers['se_clutch'].median(), 2)} runs per over. The shrinkage therefore "
        f"keeps only part of each bowler's measured values: at the median, {_pct(bowlers['B_usual'].median(), 0)} of "
        f"his usual level's distance from Replacement_Level and {_pct(bowlers['B_gap'].median(), 0)} of his "
        "Pressure_Gap.",
    ])  # fmt: skip

    pools = inputs.frame("m1_metrics").set_index("metric")["value"]
    items.append([
        Bold("The two Pools are linked only through shared players. "),
        f"The IPL and international Pools share no teams. M1 puts them on one scale only through the "
        f"{_count(pools['batters_in_both_pools'])} batters and {_count(pools['bowlers_in_both_pools'])} bowlers with "
        "balls in both, so if those players are not typical of their Pools, the Pools' scoring levels, and the "
        "ratings of players who appear in one Pool only, carry that error.",
    ])  # fmt: skip

    results = inputs.frame("result_counts")
    super_overs = int(
        sum(
            int(row["rows"])
            for _, row in results.iterrows()
            if parse_result(row["result"] if isinstance(row["result"], str) else None).result_type == SUPER_OVER_WIN
        )
    )
    items.append([
        Bold("Super overs are excluded. "),
        "The models describe innings 1 and 2, so the balls of the Super_Over_Innings are left out of training and "
        "rating. "
        + (f"The {_count(super_overs)} matches decided by a super over or a one-over eliminator are rated on their two "
           "main innings only, and their super-over balls, among the highest-stakes balls in the data, count for no "
           "player." if super_overs else "No match in the data was decided by a super over."),
    ])  # fmt: skip

    items.append([
        Bold("The next batter comes from the post-match batting order. "),
        "The Post_Wicket_State takes the next batter from playing_eleven.batting_order, which is recorded after the "
        "match, so the wicket cost Ω uses a little hindsight about who would have come in next.",
    ])  # fmt: skip

    n_choices = len(design_choice_ids(inputs.settings))
    items.append([
        Bold("Every Design_Choice is hand-set. "),
        f"The {n_choices} Design_Choices D1 to D{n_choices} are set by hand in config.py, not fitted to the data, and "
        "none was picked by comparing the validation results of Section 1.5. The Appendix Design_Choice table "
        f"({_CHOICES_PART}) lists each one with its default and the sensitivity cases run on it.",
    ])  # fmt: skip

    for role in ROLES:
        if float(inputs.value(f"rating.tau2_gap.{role}")) != 0.0:
            continue
        row = _row(summary, "role_summary.csv", role=role)
        plural = _PLURAL[role]
        items.append([
            Bold(f"No measurable pressure-specific spread among {plural}. "),
            f"The between-{role} variance τ² of the Pressure_Gap is {_fixed(row['tau2_gap_raw'], 3)} before the floor "
            "and ", _const(inputs, f"rating.tau2_gap.{role}", lambda value: _fixed(value, 3)), f" after it, so every "
            f"{role}'s Pressure_Gap is shrunk to 0 and the {role} list ranks {plural} by their shrunk usual level on "
            f"ordinary balls matched to their pressure situations. The list names {plural} who are reliable in "
            f"pressure situations; the data cannot show that any {role} does better under pressure than his own "
            "usual level.",
        ])  # fmt: skip

    close = []
    for role in ROLES:
        top = inputs.frame(f"top10_{role}")
        unsure = top[top["p_top10"].astype(float) < 0.5]
        if len(unsure):
            close.append(
                f"{role} rank{'s' if len(unsure) > 1 else ''} {_ranks(unsure['rank'])} ({_span(unsure['p_top10'], 4)})"
            )
    if close:
        items.append([
            Bold("The lower top-10 places are not secure. "),
            f"P_Top10 is below one half at {_and(close)}: in most resamples these players fall outside the top 10, so "
            "a different sample of matches would probably change the last names on each list (Section 5.2).",
        ])  # fmt: skip

    floored = [
        _PLURAL[role]
        for role in ROLES
        if float(inputs.value(f"rating.rho_L.{role}")) == 0.0 and float(inputs.value(f"rating.rho_U.{role}")) == 0.0
    ]
    items.append([
        Bold("The intervals are conditional on the Role-level values. "),
        "The bootstrap holds Replacement_Level, τ², the reliabilities and the Blend_Weights at their full-data values, "
        "so the intervals and P_Top10 leave out the uncertainty in these values and understate the full uncertainty."
        + (f" For {_and(floored)}, w_L = 1 rests on two reliabilities floored at 0, so another sample of matches "
           "could also change the blend." if floored else ""),
    ])  # fmt: skip

    zero_runs = _extras_without_runs(inputs)
    rates = _row(inputs.frame("pass_rates"), "pass_rates.csv", selected=True)
    first, *others = RUNS_CHECKS
    runs_text = _and(
        [f"the {_CHECK_NAMES[first]} check passes for {_fixed(rates[EXACT_COLUMNS[first]], 1)}% of the matches it "
         "applies to",
         *(f"the {_CHECK_NAMES[check]} check for {_fixed(rates[EXACT_COLUMNS[check]], 1)}%" for check in others)]
    )  # fmt: skip
    exclusion = inputs.frame("exclusion_counts")
    applied = exclusion[_truthy(exclusion["is_config_value"])]
    excluding = bool(inputs.settings.get("data_rules.exclude_failing_matches")) and len(applied) == 1
    items.append([
        Bold("Bye, leg-bye and Wide runs are not recorded. "),
        f"{zero_runs}. The rebuilt team scores therefore miss the runs from byes, leg-byes and Wides that ran. Under "
        "the selected ", _const(inputs, "reconstruct.selected_encoding"), f" encoding {runs_text} (Appendix {_SCORE_PART}). ",
        *(
            ["A match stays in training and rating when its runs checks miss by at most ",
             _const(inputs, "design_choices.D20_score_tolerance"), " runs and its over and wickets-margin checks pass; "
             f"this excludes {_count(applied['n_excluded_matches'].iloc[0])} matches, and a kept match can still miss "
             "its recorded result by up to ", _const(inputs, "design_choices.D20_score_tolerance"), " runs."]
            if excluding
            else ["No match is excluded for its score checks in this Run, so every rebuilt score is used as it is."]
        ),
    ])  # fmt: skip

    items.append([Bold("Player names need a manual check in Unstop. "), *_name_parts(inputs)])
    items.append(_activity_parts(inputs))
    return items


def _extras_without_runs(inputs: ReportInputs) -> str:
    """How many byes, leg-byes and Wides have ``extras_2_runs`` 0, and whether ``extras1_runs`` is a type code."""
    from superr.reconstruct import BYE_EXTRAS_TYPES

    two = inputs.frame("extras_2_runs")
    counts = [column for column in two.columns if column != "extras_type"]
    wides = [str(value) for value in inputs.settings.get("data_rules.extras_type_classes")["wide"]]
    texts, every_zero = [], True
    for kinds, words in (([BYE_EXTRAS_TYPES[0]], "byes"), ([BYE_EXTRAS_TYPES[1]], "leg-byes"), (wides, "Wides")):
        rows = two[two["extras_type"].astype(str).isin(kinds)]
        if rows.empty:
            raise StageError(f"extras_type_x_extras_2_runs.csv has no row for the extras_type value(s) {kinds}")
        total, zero = int(rows[counts].astype(int).to_numpy().sum()), int(rows["0"].astype(int).sum())
        every_zero &= zero == total
        texts.append(f"{_count(total)} {words}" if zero == total else f"{_count(zero)} of the {_count(total)} {words}")
    one = inputs.frame("extras1_runs")
    cells = one[[column for column in one.columns if column != "extras_type"]].astype(int)
    used = cells[cells.sum(axis=1) > 0]
    code = bool(len(used)) and bool(((used > 0).sum(axis=1) == 1).all())
    return (
        f"{'All ' if every_zero else ''}{_and(texts)} have extras_2_runs 0"
        + (", and extras1_runs holds one fixed value for each extras_type (a type code, not a run count)" if code else "")
    )  # fmt: skip


def _name_parts(inputs: ReportInputs) -> list[Part]:
    """The Unstop name check: the listed players whose name another player_id shares, and the checklist."""
    several = inputs.frame("names_with_several_ids")
    shared = []
    for role in ROLES:
        for _, player in inputs.frame(f"top10_{role}").iterrows():
            others = several[
                (several["player_name"].astype(str) == str(player["player_name"]))
                & (several["player_id"].astype(int) != int(player["player_id"]))
            ]
            if len(others):
                shared.append(
                    f"{player['player_name']} ({role} rank {int(player['rank'])}, player_id {int(player['player_id'])}) "
                    f"shares his name with player_id {_and(others['player_id'].astype(int))}"
                )
    parts: list[Part] = [
        "The Unstop dropdown is not in the data, so each listed name is assumed to match its player_name in "
        "competition_team_player_mapping.csv. The List_Builder marks ",
        _const(inputs, "final_lists.n_unresolved.batter"), " batter and ",
        _const(inputs, "final_lists.n_unresolved.bowler"), " bowler entries as unresolved",
    ]  # fmt: skip
    if shared:
        parts.append(
            f": {'; '.join(shared)} in that file, so the dropdown may show the name more than once. "
        )
    else:
        parts.append(". ")
    parts.append(
        "outputs/final/unstop_entry_checklist.md lists the teams of each such player_id, and every name there must "
        "be confirmed against the dropdown before submission."
    )
    return parts


def _activity_parts(inputs: ReportInputs) -> list[Part]:
    """The Recent_Activity_Rule as a confirmed assumption, with the comparison with the rule off."""
    lead = Bold("The Recent_Activity_Rule is a confirmed assumption. ")
    rule = [
        "The rule that a player must have played in ",
        _const(inputs, "data_rules.recent_years", lambda years: " or ".join(str(year) for year in years)),
        " is not part of the problem statement; the user confirmed it, to keep the lists to players who are still "
        "active for the next auction. ",
    ]
    rule_on = bool(inputs.settings.get("data_rules.recent_activity_rule"))
    if not rule_on or not all(inputs.has(name) for name in ACTIVITY_INPUTS):
        return [lead, *rule, "The rule is off in this Run." if not rule_on else "This Run holds no comparison with the "
                "rule off."]
    texts, grows = [], []
    for role in ROLES:
        summary = _row(inputs.frame(f"activity_{role}"), f"activity_comparison_{_PLURAL[role]}.csv", kind="summary")
        main = inputs.frame(f"top10_{role}")["player_id"].astype(int).tolist()
        other = inputs.frame(f"comparison_list_{role}")["player_id"].astype(int).tolist()
        moved = sum(1 for a, b in zip(main, other) if a != b)
        if main == other:
            order = "in the same order"
        elif set(main) == set(other):
            order = f"the same players, {moved} of them in another place"
        else:
            order = f"{moved} of the 10 places held by another player"
        texts.append(f"{_count(summary['overlap'])} of 10 for {_PLURAL[role]} ({order})")
        grows.append(
            [f"{_count(summary['n_eligible_main'])} to ", _const(inputs, f"activity_comparison.n_eligible.{role}"),
             f" {_PLURAL[role]}"]
        )  # fmt: skip
    return [
        lead, *rule, "With the rule off, the eligible pool grows from ", *grows[0], " and from ", *grows[1],
        f", and the top-10 overlap with the submitted lists is {_and(texts)}.",
    ]  # fmt: skip


# ------------------------------------------------------------------------------- Appendix

#: Text of a Design_Choice table row with no Robustness_Case result (Requirement 27, criterion 8).
NO_CASE_TEXT = "No sensitivity case was run."

#: Columns of the Design_Choice table CSV (module docstring, "The Design_Choice table").
DESIGN_CHOICE_COLUMNS: tuple[str, ...] = ("id", "choice", "default", "label", "config_keys", "tested")

#: Robustness-table ``target_id`` values that count as D5 alternatives: the eight Stress_Pillar
#: ablations (Requirement 21, criterion 4).
_PILLAR_TARGETS = frozenset(f"F{number}" for number in range(19, 27))
_D_NUMBER = re.compile(r"D([1-9][0-9]*)")

#: Config values whose Design_Choice-table text is not the generic one (``_config_text``).
_DEFAULT_FORMATS: Mapping[str, Callable[[Any], str]] = MappingProxyType(
    {
        "design_choices.D1_live_band": lambda band: f"{_shown(band[0])} to {_shown(band[1])}",
        "design_choices.D3_pressure_share": lambda share: f"{_shown(100.0 * float(share))}%",
    }
)


def design_choice_ids(settings: Settings) -> list[str]:
    """The D-numbers of the Design_Choices in ``constants_meta``, in order: D1, D2, ...

    StageError unless they run from D1 with none missing.
    """
    found = {
        str(entry.get("d_number"))
        for entry in settings.get("constants_meta").values()
        if isinstance(entry, Mapping) and entry.get("kind") == "design_choice"
    }
    numbered = sorted(found, key=lambda d: int(_D_NUMBER.fullmatch(d).group(1)) if _D_NUMBER.fullmatch(d) else -1)
    expected = [f"D{number}" for number in range(1, len(numbered) + 1)]
    if not numbered or numbered != expected:
        raise StageError(
            f"constants_meta names the Design_Choices {numbered}; the Design_Choice table needs D1 to Dn with none missing"
        )
    return numbered


def _config_text(path: str, value: Any) -> str:
    """A Config value as the Design_Choice table shows it."""
    special = _DEFAULT_FORMATS.get(path)
    if special is not None:
        return special(value)
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return "none"
    if isinstance(value, str):
        return _SCOPE_TEXT.get(value) or _CELL_TEXT.get(value) or value
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return _shown(value)
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(item, (list, tuple)) for item in value):
            return ", ".join("-".join(_config_text("", part) for part in item) for item in value)
        return ", ".join(_config_text("", item) for item in value)
    raise StageError(f"the Config value of {path} ({value!r}) has no text for the Design_Choice table")


def _choice_parts(descriptions: Sequence[str]) -> tuple[str, list[str]]:
    """The choice name of a Design_Choice and the detail of each of its Config keys, from their descriptions.

    One key: its description up to the first ": ". Several keys: their shared head before ": ", or else their
    shared leading words, with the rest of each description as its detail ("" for the key whose description
    is the head itself).
    """
    split = [description.split(": ", 1) for description in descriptions]
    if len(descriptions) == 1:
        return split[0][0].strip(), [""]
    if all(len(parts) == 2 for parts in split) and len({parts[0] for parts in split}) == 1:
        return split[0][0].strip(), [parts[1].strip() for parts in split]
    words = [description.split() for description in descriptions]
    shared = 0
    while all(len(item) > shared for item in words) and len({item[shared] for item in words}) == 1:
        shared += 1
    if shared:
        return " ".join(words[0][:shared]), [" ".join(item[shared:]) for item in words]
    return "; ".join(parts[0].strip() for parts in split), list(descriptions)


def _default_text(details: Sequence[str], values: Sequence[str]) -> str:
    """The default cell: one value, or each key's detail and value, with a detail-less value first."""
    if len(values) == 1:
        return values[0]
    main = [value for detail, value in zip(details, values) if not detail]
    rest = [f"{detail}: {value}" for detail, value in zip(details, values) if detail]
    if main and rest:
        return f"{'; '.join(main)} ({'; '.join(rest)})"
    return "; ".join(main or rest)


def _tested_alternatives(robustness: pd.DataFrame | None) -> dict[str, str]:
    """D-number -> the ``tested`` text of its alternatives in the robustness table (module docstring)."""
    if robustness is None or robustness.empty:
        return {}
    needed = ("target_id", "setting", "role", "spearman", "overlap")
    missing = [column for column in needed if column not in robustness.columns]
    if missing:
        raise StageError(
            f"the robustness table {ROBUSTNESS_TABLE} has no column(s) {missing}; the Design_Choice table needs {list(needed)}"
        )
    rows = robustness.copy()
    if "is_default" in rows.columns:
        rows = rows[~_truthy(rows["is_default"])]
    target = rows["target_id"].astype(str).str.strip()
    rows["_target"] = target.where(~target.isin(_PILLAR_TARGETS), "D5")
    rows = rows[rows["_target"].str.fullmatch(_D_NUMBER.pattern)]
    groups = ["_target", "case"] if "case" in rows.columns else ["_target", "setting"]
    texts: dict[str, list[str]] = {}
    for (target_id, _), case in rows.groupby(groups, sort=False, dropna=False):
        results = []
        for role in ROLES:
            row = case[case["role"].astype(str) == role]
            if row.empty:
                continue
            first = row.iloc[0]
            comparable = "comparable" not in case.columns or bool(_truthy(row["comparable"]).iloc[0])
            if comparable and pd.notna(first["spearman"]) and pd.notna(first["overlap"]):
                results.append(f"{_PLURAL[role]} {_fixed(first['spearman'], 3)} and {_count(first['overlap'])}")
            else:
                results.append(f"{_PLURAL[role]} not comparable")
        labels = [case[column].iloc[0] for column in ("setting", "case") if column in case.columns]
        setting = next((str(label).strip() for label in labels if pd.notna(label) and str(label).strip()), "unnamed case")
        texts.setdefault(str(target_id), []).append(f"{setting}: {'; '.join(results) or 'no Role result'}")
    return {target_id: ". ".join(parts) + "." for target_id, parts in texts.items()}


def design_choice_table(
    settings: Settings, constants: ConstantRegistry, robustness: pd.DataFrame | None = None
) -> pd.DataFrame:
    """The Design_Choice table: one row per Design_Choice D1-Dn, columns DESIGN_CHOICE_COLUMNS.

    Built from the Config values and the ``constants_meta`` descriptions of the ``design_choices``
    keys, the ConstantRegistry labels, and the rows of ``robustness`` (the robustness table, or
    None) for the ``tested`` column (module docstring). StageError when ``constants_meta`` misses a
    Design_Choice or a ``design_choices`` key, a description is blank, or the robustness table
    lacks a needed column or names a Design_Choice that ``constants_meta`` does not.
    """
    meta = settings.get("constants_meta")
    ids = design_choice_ids(settings)
    entries: dict[str, list[tuple[str, str]]] = {d: [] for d in ids}
    for path, entry in meta.items():
        if isinstance(entry, Mapping) and entry.get("kind") == "design_choice":
            entries[str(entry["d_number"])].append((path, str(entry.get("description") or "").strip()))
    listed = {path for keys in entries.values() for path, _ in keys}
    unlisted = [f"design_choices.{key}" for key in settings.get("design_choices") if f"design_choices.{key}" not in listed]
    blank = [path for keys in entries.values() for path, description in keys if not description]
    if unlisted or blank:
        raise StageError(
            "the Design_Choice table needs a design_choice entry with a description in constants_meta for every "
            f"design_choices key; missing: {unlisted}; blank descriptions: {blank}"
        )
    tested = _tested_alternatives(robustness)
    unknown = sorted(set(tested) - set(ids))
    if unknown:
        raise StageError(f"the robustness table {ROBUSTNESS_TABLE} names Design_Choice(s) {unknown} that constants_meta does not")
    rows = []
    for d in ids:
        keys = entries[d]
        choice, details = _choice_parts([description for _, description in keys])
        rows.append(
            {
                "id": d,
                "choice": choice,
                "default": _default_text(details, [_config_text(path, settings.get(path)) for path, _ in keys]),
                "label": "; ".join(dict.fromkeys(constants.label_text(path) for path, _ in keys)),
                "config_keys": "; ".join(path for path, _ in keys),
                "tested": tested.get(d, NO_CASE_TEXT),
            }
        )
    return pd.DataFrame(rows, columns=list(DESIGN_CHOICE_COLUMNS))


def write_design_choice_table(inputs: ReportInputs) -> pd.DataFrame:
    """Build the Design_Choice table and write it to ``inputs.path("design_choices")`` through ``<file>.tmp``."""
    robustness = inputs.frame("robustness_table") if inputs.has("robustness_table") else None
    table = design_choice_table(inputs.settings, inputs.constants, robustness)
    path = inputs.path("design_choices")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        table.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return table


def appendix(builder: ReportBuilder, inputs: ReportInputs) -> None:
    """Appendix: the audit summary, the score-reconstruction pass rates and exclusions, and the Design_Choice
    table D1-D20 (27.8). Writes the Design_Choice table CSV first (``write_design_choice_table``)."""
    s = builder.appendix
    s.add_heading(f"{_AUDIT_PART} Data audit")
    _audit_summary(s, inputs)
    s.add_heading(f"{_SCORE_PART} Score reconstruction")
    _score_reconstruction(s, inputs)
    s.add_heading(f"{_CHOICES_PART} Design_Choices")
    _design_choices(s, inputs)


def _audit_summary(s: Section, inputs: ReportInputs) -> None:
    audit = parse_audit(inputs.text("audit"), inputs.path("audit"))
    deliveries = int(inputs.frame("bowl_speed")["deliveries"].iloc[0])
    passed = sum(check.passed for check in audit.fact_checks)
    total = len(audit.fact_checks)
    s.add_paragraph(
        f"Before any model was fitted, the Auditor profiled the {_count(deliveries)} Deliveries of the "
        f"{_count(len(inputs.frame('fold_table')))} matches and checked the data facts that the design relies on. The "
        f"full audit is outputs/audit/audit.md, with the {len(audit.saved_tables)} tables it lists in the same folder; "
        f"this is a summary. {'All' if passed == total else f'{passed} of the'} {total} Fact_Checks pass:"
    )
    for check in audit.fact_checks:
        s.add_bullet(
            Bold(f"{_plain_markdown(check.name)}: "), *_inline(check.check), ". Observed: ", *_inline(check.observed),
            f". Result: {_plain_markdown(check.result)}.",
        )  # fmt: skip
    classes = inputs.frame("dismissal_classification")
    codes = {
        kind: [str(int(code)) for code in classes.loc[classes["classification"].astype(str) == kind, "dismissal_type_id"]]
        for kind in ("Counted_Wicket", "Retired_Hurt")
    }
    ruled = " ".join(
        f"Code {int(row['dismissal_type_id'])} ({row['dismissal_types']}) is a {row['classification']} by the "
        f"{row['rule']} rule: {row['reason']}."
        for _, row in classes.iterrows()
        if str(row["rule"]) != "base_code"
    )
    s.add_paragraph(
        "The Auditor classified every dismissal code in the data. Counted_Wicket codes: "
        f"{_and(codes['Counted_Wicket']) or 'none'}. Retired_Hurt codes: {_and(codes['Retired_Hurt']) or 'none'}. "
        + ruled
    )
    _table(
        s, inputs, "a_dismissal_codes", "dismissal_classification",
        ["dismissal_type_id", "dismissal_types", "rows", "classification", "rule"],
        key=["dismissal_type_id"],
        decimals={"dismissal_type_id": 0, "rows": 0},
        headers={"dismissal_type_id": "Dismissal code", "dismissal_types": "Labels", "rows": "Deliveries",
                 "classification": "Classification", "rule": "Deciding rule"},
        caption="Dismissal codes: labels, Deliveries and the Auditor's classification with the rule that decided it",
    )  # fmt: skip
    several = inputs.frame("names_with_several_ids")
    ids = inputs.frame("ids_with_several_names")
    overs = inputs.frame("over_sizes")
    six = int(overs.loc[overs["per_over"].astype(int) == 6, "overs_by_legal_balls"].sum())
    s.add_paragraph(
        f"{_count(several['player_name'].nunique())} player_name values in competition_team_player_mapping.csv map to "
        f"more than one player_id, and {_count(ids['player_id'].nunique())} player_id values map to more than one "
        f"name (Section 5.4 covers the listed players). {_count(six)} of the "
        f"{_count(overs['overs_by_legal_balls'].sum())} Overs, super overs included, hold exactly 6 Legal_Balls. The "
        "extras and bowl_speed facts behind two of the limitations are in Section 5.4."
    )


def _score_reconstruction(s: Section, inputs: ReportInputs) -> None:
    from superr.reconstruct import EXACT_COLUMNS, RUNS_CHECKS, SCORE_CHECKS, WITHIN_TOL_COLUMNS

    rates = inputs.frame("pass_rates")
    chosen = _row(rates, "pass_rates.csv", selected=True)
    exact = [EXACT_COLUMNS[check] for check in SCORE_CHECKS]
    within = [WITHIN_TOL_COLUMNS[check] for check in SCORE_CHECKS]
    means = rates[exact].astype(float).mean(axis=1).to_numpy()
    is_chosen = _truthy(rates["selected"]).to_numpy()
    best = float(means[is_chosen][0])
    tied = [str(name) for name, mean, own in zip(rates["encoding"], means, is_chosen) if not own and mean == best]
    s.add_paragraph(
        f"The Score_Reconstructor rebuilt the team runs of every Delivery under the {len(rates)} candidate "
        f"Extras_Encodings ({_and(rates['encoding'].astype(str))}) and checked each match's rebuilt scores against its "
        f"recorded result with the {_and(_CHECK_NAMES[check] for check in SCORE_CHECKS)} checks. It selected ",
        _const(inputs, "reconstruct.selected_encoding"),
        f", the candidate with the highest mean Exact_Pass_Rate, {_fixed(best, 1)}%"
        + (f"; {_and(tied)} has the same mean, and the rule takes the candidate listed first." if tied else "."),
    )  # fmt: skip
    rate_headers = {EXACT_COLUMNS[check]: f"{_CHECK_NAMES[check].capitalize()} check %" for check in SCORE_CHECKS}
    _table(
        s, inputs, "a_pass_rates_exact", "pass_rates",
        ["encoding", *exact, "selected"],
        key=["encoding"],
        decimals={column: 1 for column in exact},
        headers={"encoding": "Extras_Encoding", **rate_headers, "selected": "Selected"},
        caption="Exact_Pass_Rate of each Score_Check under each candidate Extras_Encoding, in percent of the matches "
        "the check applies to",
    )  # fmt: skip
    first, *others = RUNS_CHECKS
    exact_only = [check for check in SCORE_CHECKS if check not in RUNS_CHECKS]
    runs_text = _and(
        [f"the {_CHECK_NAMES[first]} check passes for {_fixed(chosen[EXACT_COLUMNS[first]], 1)}% of the matches it "
         "applies to",
         *(f"the {_CHECK_NAMES[check]} check for {_fixed(chosen[EXACT_COLUMNS[check]], 1)}%" for check in others)]
    )  # fmt: skip
    s.add_paragraph(
        f"Byes, leg-byes and Wides carry no recorded runs (Section 5.4), so the exact runs checks fail for many "
        f"matches: under the selected encoding {runs_text}. The {_and(_CHECK_NAMES[check] for check in exact_only)} "
        "checks do not depend on the run counts and pass for "
        f"{_and(f'{_fixed(chosen[EXACT_COLUMNS[check]], 1)}%' for check in exact_only)}. A "
        f"{_or(_CHECK_NAMES[check] for check in RUNS_CHECKS)} check passes within tolerance when it misses by at most "
        "the Score_Tolerance of ", _const(inputs, "design_choices.D20_score_tolerance"), " runs, while the "
        f"{_and(_CHECK_NAMES[check] for check in exact_only)} checks must pass exactly:",
    )  # fmt: skip
    _table(
        s, inputs, "a_pass_rates_within_tolerance", "pass_rates",
        ["encoding", *within],
        key=["encoding"],
        decimals={column: 1 for column in within},
        headers={"encoding": "Extras_Encoding",
                 **{WITHIN_TOL_COLUMNS[check]: f"{_CHECK_NAMES[check].capitalize()} check %" for check in SCORE_CHECKS}},
        where={"selected": "True"},
        caption="Within_Tolerance_Pass_Rate of each Score_Check under the selected Extras_Encoding, in percent",
    )  # fmt: skip
    exclusion = inputs.frame("exclusion_counts")
    applied = exclusion[_truthy(exclusion["is_config_value"])]
    on = bool(inputs.settings.get("data_rules.exclude_failing_matches"))
    s.add_paragraph(
        "A match is left out of model training and player rating when it fails the "
        f"{_or(_CHECK_NAMES[check] for check in exact_only)} check, or misses a runs check by more than the "
        "Score_Tolerance. "
        + (f"At the Config value this excludes {_count(applied['n_excluded_matches'].iloc[0])} matches. "
           if on and len(applied) == 1
           else "The option exclude_failing_matches is off in this Run, so no match is excluded. " if not on else "")
        + "The table gives the matches excluded at each tolerance of the reporting grid, ",
        _const(inputs, "data_rules.score_tolerance_grid", lambda grid: _and(_shown(value) for value in grid)), " runs.",
    )  # fmt: skip
    _table(
        s, inputs, "a_exclusion_counts", "exclusion_counts",
        ["score_tolerance", "n_excluded_matches", "is_config_value"],
        key=["score_tolerance"],
        decimals={"score_tolerance": 0, "n_excluded_matches": 0},
        headers={"score_tolerance": "Score_Tolerance (runs)", "n_excluded_matches": "Matches excluded",
                 "is_config_value": "Config value (D20)"},
        caption="Matches excluded from model training and rating at each Score_Tolerance of the reporting grid",
    )  # fmt: skip


def _design_choices(s: Section, inputs: ReportInputs) -> None:
    table = write_design_choice_table(inputs)
    s.add_paragraph(
        "Every Design_Choice is set by hand in config.py; none is fitted to the data or chosen by comparing the "
        f"validation results. The table lists the {len(table)} Design_Choices with their defaults, as Config keys "
        "describe them. For each alternative that the robustness stage tested, it gives the Spearman correlation "
        "between the case's Final_Scores and the main ranking's, and how many of the main top 10 stay in the case's "
        "top 10, for batters and for bowlers. "
        + ("Rows with no case results state that no sensitivity case was run."
           if inputs.has("robustness_table")
           else "The robustness stage has not run for this report, so every row states that no sensitivity case was "
           "run.")
    )  # fmt: skip
    _table(
        s, inputs, "a_design_choices", "design_choices",
        ["id", "choice", "default", "label", "tested"],
        key=["id"],
        headers={"id": "Design_Choice", "choice": "Choice", "default": "Default in config.py", "label": "Label",
                 "tested": "Tested alternatives: Spearman correlation and top-10 overlap with the main ranking"},
        caption=f"The Design_Choices {table['id'].iloc[0]} to {table['id'].iloc[-1]}: default, label and the "
        "sensitivity cases run on each",
    )  # fmt: skip


# ---------------------------------------------------------------------------- all of them

#: The section functions of Sections 0-5 and the Appendix, in report order.
SECTION_WRITERS: tuple[Callable[[ReportBuilder, ReportInputs], None], ...] = (
    section_0,
    section_1,
    section_2,
    section_3,
    section_4,
    section_5,
    appendix,
)


def write_sections(
    builder: ReportBuilder,
    inputs: ReportInputs,
    writers: Sequence[Callable[[ReportBuilder, ReportInputs], None]] = SECTION_WRITERS,
) -> None:
    """Fill the builder's sections with each writer in turn (default: Sections 0-5 and the Appendix).

    The Appendix writer also writes the Design_Choice table CSV to ``inputs.path("design_choices")``.
    """
    for writer in writers:
        writer(builder, inputs)
