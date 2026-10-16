"""Section-aware chunking of parsed guideline documents.

Rules (from the research plan):

* Text is split along section headings; chunks target `min_tokens`-`max_tokens`
  with roughly `overlap_ratio` overlap between consecutive chunks of the same
  section. Adjacent small chunks under the same parent section are merged.
* Every chunk starts with a provenance line "Document title › Section path".
* A table is never split across chunks mid-row. Long tables are split into row
  groups that repeat the header. Footnotes and notes stay with their table.
* Table chunks show the Markdown table to the LLM but index plain-language row
  descriptions for retrieval.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from vaxrag.config import ChunkingConfig
from vaxrag.kb.tables import is_footnote, markdown_table, row_descriptions, split_rows
from vaxrag.schemas import Block, Chunk, document_label

CountFn = Callable[[str], int]

_SENTENCE_RE = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9(\"'])")
_NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+\S")


class TokenCounter:
    """Count tokens with tiktoken; fall back to a character heuristic offline."""

    def __init__(self, encoding: str = "cl100k_base") -> None:
        try:
            import tiktoken

            self._enc = tiktoken.get_encoding(encoding)
        except Exception:  # pragma: no cover - only when the encoding cannot be loaded
            self._enc = None

    def __call__(self, text: str) -> int:
        """Return the number of tokens in `text`."""
        if self._enc is None:
            return max(1, len(text) // 4)
        return len(self._enc.encode(text, disallowed_special=()))


def heading_level(block: Block) -> int:
    """Infer a heading level, preferring explicit numbering such as '3.2.1'."""
    match = _NUMBERED_HEADING_RE.match(block.text)
    if match:
        return match.group(1).count(".") + 1
    return max(1, block.level)


@dataclass
class _Piece:
    """A paragraph (or sentence group) with its page, used while building chunks."""

    text: str
    page: int | None
    tokens: int


@dataclass
class _Draft:
    """A chunk under construction before IDs are assigned."""

    section_path: list[str]
    kind: str
    body: str
    retrieval_body: str
    pages: list[int] = field(default_factory=list)
    tokens: int = 0
    table_source: str | None = None


def _split_long_paragraph(
    text: str, page: int | None, max_tokens: int, count: CountFn
) -> list[_Piece]:
    """Split a paragraph that exceeds `max_tokens` at sentence boundaries."""
    pieces: list[_Piece] = []
    buffer: list[str] = []
    for sentence in _SENTENCE_RE.split(text):
        candidate = " ".join([*buffer, sentence])
        if buffer and count(candidate) > max_tokens:
            pieces.append(_Piece(" ".join(buffer), page, count(" ".join(buffer))))
            buffer = [sentence]
        else:
            buffer.append(sentence)
    if buffer:
        pieces.append(_Piece(" ".join(buffer), page, count(" ".join(buffer))))
    return pieces


def pack_text(
    pieces: list[_Piece], cfg: ChunkingConfig, count: CountFn
) -> list[tuple[str, list[int]]]:
    """Pack paragraphs into chunks of at most `max_tokens` with trailing overlap.

    Returns (text, pages) tuples. Overlap repeats the last paragraphs of the
    previous chunk, up to `overlap_ratio * max_tokens` tokens.
    """
    expanded: list[_Piece] = []
    for piece in pieces:
        if piece.tokens > cfg.max_tokens:
            expanded.extend(_split_long_paragraph(piece.text, piece.page, cfg.max_tokens, count))
        else:
            expanded.append(piece)
    overlap_budget = int(cfg.max_tokens * cfg.overlap_ratio)
    chunks: list[tuple[str, list[int]]] = []
    current: list[_Piece] = []
    current_tokens = 0
    new_content = False
    for piece in expanded:
        if current and new_content and current_tokens + piece.tokens > cfg.max_tokens:
            chunks.append(
                ("\n\n".join(p.text for p in current), [p.page for p in current if p.page])
            )
            # Carry trailing paragraphs into the next chunk as overlap.
            carried: list[_Piece] = []
            carried_tokens = 0
            for prev in reversed(current):
                if carried_tokens + prev.tokens > overlap_budget:
                    break
                carried.insert(0, prev)
                carried_tokens += prev.tokens
            current, current_tokens, new_content = carried, carried_tokens, False
        current.append(piece)
        current_tokens += piece.tokens
        new_content = True
    if current and new_content:
        chunks.append(("\n\n".join(p.text for p in current), [p.page for p in current if p.page]))
    return chunks


def _common_prefix(a: list[str], b: list[str]) -> list[str]:
    """Return the longest common prefix of two section paths."""
    prefix: list[str] = []
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        prefix.append(x)
    return prefix


def merge_small_drafts(drafts: list[_Draft], cfg: ChunkingConfig, count: CountFn) -> list[_Draft]:
    """Merge adjacent small text drafts that share a parent section.

    A draft smaller than `min_tokens` is merged into the following text draft
    when the result fits in `max_tokens` and both share at least their
    top-level section. The sub-heading of the second draft is kept inline.
    """
    merged: list[_Draft] = []
    for draft in drafts:
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and prev.kind == "text"
            and draft.kind == "text"
            and prev.tokens < cfg.min_tokens
            and prev.tokens + draft.tokens <= cfg.max_tokens
            and (_common_prefix(prev.section_path, draft.section_path) or not prev.section_path)
        ):
            common = _common_prefix(prev.section_path, draft.section_path)
            heading = (
                ""
                if draft.section_path == prev.section_path
                else ((draft.section_path[-1] + "\n") if draft.section_path else "")
            )
            body = f"{prev.body}\n\n{heading}{draft.body}"
            merged[-1] = _Draft(
                section_path=common,
                kind="text",
                body=body,
                retrieval_body=body,
                pages=prev.pages + draft.pages,
                tokens=count(body),
            )
        else:
            merged.append(draft)
    return merged


def _table_drafts(
    block: Block, footnotes: list[str], path: list[str], cfg: ChunkingConfig, count: CountFn
) -> list[_Draft]:
    """Turn one table block into one or more table drafts (row groups)."""
    header = block.header or []
    rows = block.rows or []
    caption = block.caption or (path[-1] if path else None)
    notes = "\n".join(f"Note: {f}" for f in footnotes)
    caption_line = f"Table: {caption}\n\n" if caption else ""
    overhead = count(caption_line + notes) + 30
    drafts: list[_Draft] = []
    for group in split_rows(header, rows, cfg.table_max_tokens, count, fixed_overhead=overhead):
        table_md = markdown_table(header, group)
        body = f"{caption_line}{table_md}" + (f"\n\n{notes}" if notes else "")
        descriptions = "\n".join(row_descriptions(caption, header, group))
        retrieval_body = f"{caption_line}{descriptions}" + (f"\n\n{notes}" if notes else "")
        drafts.append(
            _Draft(
                section_path=list(path),
                kind="table",
                body=body,
                retrieval_body=retrieval_body,
                pages=[block.page] if block.page else [],
                table_source=block.table_source,
                tokens=count(body),
            )
        )
    return drafts


def chunk_document(
    blocks: list[Block],
    doc_id: str,
    doc_title: str,
    cfg: ChunkingConfig,
    count: CountFn,
    doc_year: str | None = None,
) -> list[Chunk]:
    """Chunk the ordered blocks of one document.

    Every chunk starts with the provenance line "Title (year) › Section path";
    the year is omitted when `doc_year` is None.
    """
    drafts: list[_Draft] = []
    stack: list[tuple[int, str]] = []
    buffer: list[_Piece] = []

    def path() -> list[str]:
        return [text for _, text in stack]

    def flush() -> None:
        for text, pages in pack_text(buffer, cfg, count):
            drafts.append(_Draft(path(), "text", text, text, pages, count(text)))
        buffer.clear()

    i = 0
    while i < len(blocks):
        block = blocks[i]
        if block.kind == "heading":
            flush()
            level = heading_level(block)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, block.text.strip()))
        elif block.kind == "table":
            flush()
            footnotes = list(block.footnotes)
            # Attach following note/footnote paragraphs on the same or next page.
            j = i + 1
            while (
                j < len(blocks)
                and blocks[j].kind == "text"
                and is_footnote(blocks[j].text)
                and (
                    block.page is None or blocks[j].page is None or blocks[j].page - block.page <= 1
                )
            ):
                footnotes.append(blocks[j].text.strip())
                j += 1
            drafts.extend(_table_drafts(block, footnotes, path(), cfg, count))
            i = j
            continue
        elif block.text.strip():
            buffer.append(_Piece(block.text.strip(), block.page, count(block.text)))
        i += 1
    flush()

    chunks: list[Chunk] = []
    for n, draft in enumerate(merge_small_drafts(drafts, cfg, count)):
        prefix = " › ".join([document_label(doc_title, doc_year), *draft.section_path])
        text = f"{prefix}\n\n{draft.body}"
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}::{n:04d}",
                doc_id=doc_id,
                doc_title=doc_title,
                doc_year=doc_year,
                section_path=draft.section_path,
                kind=draft.kind,  # type: ignore[arg-type]
                text=text,
                retrieval_text=f"{prefix}\n\n{draft.retrieval_body}",
                page_start=min(draft.pages) if draft.pages else None,
                page_end=max(draft.pages) if draft.pages else None,
                n_tokens=count(text),
                table_source=draft.table_source,
            )
        )
    return chunks
