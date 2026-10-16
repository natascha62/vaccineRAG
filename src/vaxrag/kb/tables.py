"""Table utilities: Markdown rendering, row descriptions and row-group splitting.

Tables are stored in two representations (see the research plan):

* a Markdown table, which is what the LLM sees, and
* one plain-language sentence per row, which is what the retrievers index,
  e.g. "MMR schedule - Dose 2: Minimum age: 12 months; Minimum interval: 4 weeks".

Tables are never cut mid-row. Tables that exceed the token budget are split
into row groups, and every group repeats the header row.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

FOOTNOTE_RE = re.compile(
    r"^\s*(\*|†|‡|§|¶|\(?[a-z]\)\s|notes?\b|abbreviations?\b|footnotes?\b|source:)",
    flags=re.IGNORECASE,
)


@dataclass
class ParsedTable:
    """A table recovered from Markdown (e.g. multimodal LLM output)."""

    caption: str | None
    header: list[str]
    rows: list[list[str]]
    footnotes: list[str] = field(default_factory=list)


def _cell(text: str) -> str:
    """Escape a value for use inside a Markdown table cell."""
    return re.sub(r"\s+", " ", str(text)).replace("|", "\\|").strip()


def markdown_table(header: list[str], rows: list[list[str]]) -> str:
    """Render a header and rows as a GitHub-flavored Markdown table."""
    width = max([len(header), *[len(r) for r in rows]] or [0])
    if width == 0:
        return ""
    header = [*header, *[""] * (width - len(header))]
    lines = [
        "| " + " | ".join(_cell(h) for h in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows:
        row = [*row, *[""] * (width - len(row))]
        lines.append("| " + " | ".join(_cell(c) for c in row) + " |")
    return "\n".join(lines)


def row_descriptions(caption: str | None, header: list[str], rows: list[list[str]]) -> list[str]:
    """Describe each row in plain language as 'caption - first cell: header: value; ...'.

    The first column is treated as the row label. Empty cells are skipped and
    cells identical to the row label (merged-cell duplicates) are not repeated.
    """
    descriptions: list[str] = []
    for row in rows:
        if not any(str(c).strip() for c in row):
            continue
        label = _cell(row[0]) if row else ""
        parts: list[str] = []
        for idx, value in enumerate(row[1:], start=1):
            value = _cell(value)
            if not value or value == label:
                continue
            name = (
                _cell(header[idx])
                if idx < len(header) and str(header[idx]).strip()
                else (f"Column {idx + 1}")
            )
            parts.append(f"{name}: {value}")
        prefix = f"{caption} - " if caption else ""
        body = "; ".join(parts) if parts else ""
        descriptions.append(f"{prefix}{label}: {body}".rstrip(": ").strip())
    return descriptions


def split_rows(
    header: list[str],
    rows: list[list[str]],
    max_tokens: int,
    count_tokens: Callable[[str], int],
    fixed_overhead: int = 0,
) -> list[list[list[str]]]:
    """Split rows into groups whose Markdown (with the header) fits in `max_tokens`.

    A single row that alone exceeds the budget becomes its own group; rows are
    never cut. `fixed_overhead` accounts for text repeated in every group
    (heading prefix, caption, footnotes).
    """
    if not rows:
        return [[]]
    header_tokens = count_tokens(markdown_table(header, [])) + fixed_overhead
    groups: list[list[list[str]]] = []
    current: list[list[str]] = []
    current_tokens = header_tokens
    for row in rows:
        row_tokens = count_tokens(markdown_table([""] * len(row), [row]).split("\n", 2)[-1])
        if current and current_tokens + row_tokens > max_tokens:
            groups.append(current)
            current, current_tokens = [], header_tokens
        current.append(row)
        current_tokens += row_tokens
    groups.append(current)
    return groups


def parse_markdown_tables(text: str) -> list[ParsedTable]:
    """Parse the table-vision output format into tables.

    Expected format (one or more blocks)::

        TABLE: <caption>
        | h1 | h2 |
        | --- | --- |
        | a | b |
        FOOTNOTES:
        - footnote text
    """
    tables: list[ParsedTable] = []
    current: ParsedTable | None = None
    in_footnotes = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.upper().startswith("TABLE:"):
            current = ParsedTable(caption=line[6:].strip() or None, header=[], rows=[])
            tables.append(current)
            in_footnotes = False
            continue
        if line.upper().startswith("FOOTNOTES:"):
            in_footnotes = True
            continue
        if current is None:
            if line.startswith("|"):
                current = ParsedTable(caption=None, header=[], rows=[])
                tables.append(current)
            else:
                continue
        if in_footnotes:
            # Remove only the list bullet; keep footnote markers such as "*" or "†".
            note = re.sub(r"^[-•]\s+", "", line).strip()
            if note:
                current.footnotes.append(note)
        elif line.startswith("|"):
            cells = [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", line.strip("|"))]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                continue
            if not current.header:
                current.header = cells
            else:
                current.rows.append(cells)
    return [t for t in tables if t.header or t.rows]


def is_footnote(text: str) -> bool:
    """Return True if a text block looks like a table footnote or note (heuristic)."""
    return bool(FOOTNOTE_RE.match(text)) and len(text) < 1500
