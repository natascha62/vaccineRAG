"""Parse guideline PDFs with Docling into an ordered list of `Block`s.

For every document three artifacts are written to `parsed_dir`:

* `<doc_id>.docling.json` - the full Docling document (for inspection/re-use),
* `<doc_id>.md`           - Markdown export (for manual table checking),
* `<doc_id>.blocks.jsonl` - the simplified blocks consumed by the chunker.

A table-quality report flags tables that Docling probably failed to parse, as
candidates for multimodal conversion (see `table_vision.py`).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from vaxrag.data.manifest import ManifestEntry
from vaxrag.io_utils import read_models, write_jsonl
from vaxrag.kb.tables import parse_markdown_tables
from vaxrag.schemas import Block

logger = logging.getLogger(__name__)


def create_converter():
    """Create a Docling converter with accurate table-structure recognition."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions()
    options.do_table_structure = True
    options.table_structure_options.mode = TableFormerMode.ACCURATE
    options.table_structure_options.do_cell_matching = True
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def _page(item) -> int | None:
    """First page number of a Docling item, if known."""
    prov = getattr(item, "prov", None)
    return prov[0].page_no if prov else None


def _table_cells(item, doc) -> tuple[list[str], list[list[str]]]:
    """Extract header and body rows from a Docling table as strings."""
    df = item.export_to_dataframe(doc=doc)
    header = ["" if isinstance(c, int) else str(c).strip() for c in df.columns]
    rows = [
        ["" if pd.isna(v) else str(v).strip() for v in row] for row in df.itertuples(index=False)
    ]
    return header, rows


def docling_to_blocks(doc, doc_id: str) -> list[Block]:
    """Convert a DoclingDocument into blocks (headings, text paragraphs, tables).

    Captions and footnotes that belong to a table are attached to the table and
    not emitted again as separate text. Page headers and footers are excluded
    by Docling's default content layer.
    """
    from docling_core.types.doc import DocItemLabel, SectionHeaderItem, TableItem, TextItem

    attached: set[str] = set()
    for item, _ in doc.iterate_items():
        if isinstance(item, TableItem):
            attached.update(ref.cref for ref in item.captions)
            attached.update(ref.cref for ref in item.footnotes)

    blocks: list[Block] = []
    for item, _level in doc.iterate_items():
        if getattr(item, "self_ref", None) in attached:
            continue
        if isinstance(item, TableItem):
            header, rows = _table_cells(item, doc)
            footnotes = [ref.resolve(doc).text for ref in item.footnotes]
            blocks.append(
                Block(
                    doc_id=doc_id,
                    kind="table",
                    page=_page(item),
                    caption=item.caption_text(doc) or None,
                    header=header,
                    rows=rows,
                    footnotes=[f for f in footnotes if f],
                    table_source="docling",
                )
            )
        elif (
            isinstance(item, SectionHeaderItem)
            or getattr(item, "label", None) == DocItemLabel.TITLE
        ):
            level = getattr(item, "level", 1) or 1
            blocks.append(
                Block(doc_id=doc_id, kind="heading", text=item.text, level=level, page=_page(item))
            )
        elif isinstance(item, TextItem) and item.text.strip():
            blocks.append(Block(doc_id=doc_id, kind="text", text=item.text, page=_page(item)))
    return blocks


def parse_document(
    entry: ManifestEntry, raw_dir: str | Path, parsed_dir: str | Path, converter
) -> list[Block]:
    """Parse one document and write its artifacts; returns the blocks."""
    parsed_dir = Path(parsed_dir)
    parsed_dir.mkdir(parents=True, exist_ok=True)
    result = converter.convert(str(Path(raw_dir) / entry.filename))
    doc = result.document
    doc.save_as_json(parsed_dir / f"{entry.doc_id}.docling.json")
    (parsed_dir / f"{entry.doc_id}.md").write_text(doc.export_to_markdown(), encoding="utf-8")
    blocks = docling_to_blocks(doc, entry.doc_id)
    write_jsonl(parsed_dir / f"{entry.doc_id}.blocks.jsonl", blocks)
    return blocks


def table_quality(blocks: list[Block]) -> list[dict]:
    """Flag tables that look badly parsed (few rows, mostly empty cells, no header)."""
    report = []
    for block in blocks:
        if block.kind != "table":
            continue
        cells = [c for row in (block.rows or []) for c in row]
        empty_share = (sum(1 for c in cells if not c) / len(cells)) if cells else 1.0
        issues = []
        if len(block.rows or []) < 2:
            issues.append("fewer than 2 rows")
        if empty_share > 0.5:
            issues.append(f"{empty_share:.0%} empty cells")
        if not any(h for h in (block.header or [])):
            issues.append("no header")
        report.append(
            {
                "doc_id": block.doc_id,
                "page": block.page,
                "caption": block.caption,
                "n_rows": len(block.rows or []),
                "n_cols": len(block.header or []),
                "empty_share": round(empty_share, 3),
                "suspect": bool(issues),
                "issues": "; ".join(issues),
            }
        )
    return report


def load_blocks(parsed_dir: str | Path, doc_id: str) -> list[Block]:
    """Load the parsed blocks of one document."""
    path = Path(parsed_dir) / f"{doc_id}.blocks.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `vaxrag parse` first")
    return read_models(path, Block)


def apply_table_overrides(blocks: list[Block], overrides_dir: str | Path) -> list[Block]:
    """Replace Docling tables with multimodal-LLM tables where an override exists.

    An override file `<doc_id>_p<page>.md` replaces every Docling table on that
    page. The new tables are inserted where the first replaced table was (or
    after the last block of that page if Docling found no table there).
    """
    if not blocks:
        return blocks
    doc_id = blocks[0].doc_id
    overrides_dir = Path(overrides_dir)
    result = list(blocks)
    for path in sorted(overrides_dir.glob(f"{doc_id}_p*.md")):
        page = int(path.stem.rsplit("_p", 1)[1])
        tables = parse_markdown_tables(path.read_text(encoding="utf-8"))
        new_blocks = [
            Block(
                doc_id=doc_id,
                kind="table",
                page=page,
                caption=t.caption,
                header=t.header,
                rows=t.rows,
                footnotes=t.footnotes,
                table_source="vision",
            )
            for t in tables
        ]
        positions = [i for i, b in enumerate(result) if b.kind == "table" and b.page == page]
        if positions:
            insert_at = positions[0]
            result = [b for i, b in enumerate(result) if i not in set(positions)]
        else:
            same_page = [i for i, b in enumerate(result) if b.page == page]
            insert_at = (same_page[-1] + 1) if same_page else len(result)
        result[insert_at:insert_at] = new_blocks
        logger.info("Applied %d vision table(s) for %s page %d", len(new_blocks), doc_id, page)
    return result
