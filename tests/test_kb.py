from pathlib import Path

from conftest import word_count

from vaxrag.config import ChunkingConfig
from vaxrag.kb.chunking import _Piece, chunk_document, heading_level, pack_text
from vaxrag.kb.parsing import apply_table_overrides, table_quality
from vaxrag.kb.tables import (
    is_footnote,
    markdown_table,
    parse_markdown_tables,
    row_descriptions,
    split_rows,
)
from vaxrag.schemas import Block

CFG = ChunkingConfig(min_tokens=20, max_tokens=40, overlap_ratio=0.25, table_max_tokens=60)


def test_markdown_table_pads_and_escapes():
    md = markdown_table(["Vaccine", "Dose"], [["MMR", "1|2"], ["Hep B"]])
    lines = md.splitlines()
    assert lines[0] == "| Vaccine | Dose |"
    assert lines[1] == "| --- | --- |"
    assert lines[2] == "| MMR | 1\\|2 |"
    assert lines[3] == "| Hep B |  |"


def test_row_descriptions_skip_empty_and_merged_duplicates():
    rows = [["Dose 2", "12 months", "4 weeks"], ["Dose 3", "Dose 3", ""], ["", "", ""]]
    out = row_descriptions("MMR schedule", ["Dose", "Minimum age", "Minimum interval"], rows)
    assert out == [
        "MMR schedule - Dose 2: Minimum age: 12 months; Minimum interval: 4 weeks",
        "MMR schedule - Dose 3",
    ]


def test_split_rows_never_cuts_rows_and_respects_budget():
    header = ["a", "b"]
    rows = [[f"r{i}", "x " * 5] for i in range(10)]
    groups = split_rows(header, rows, max_tokens=40, count_tokens=word_count)
    assert sum(len(g) for g in groups) == 10
    assert len(groups) > 1
    for g in groups:
        assert word_count(markdown_table(header, g)) <= 40 or len(g) == 1


def test_parse_markdown_tables_vision_format():
    text = """TABLE: Table 1. Minimum ages
| Vaccine | Min age |
| --- | --- |
| MMR | 12 months* |
FOOTNOTES:
- * During outbreaks, 6 months.
TABLE: Table 2
| A | B |
|---|---|
| 1 | 2 |
"""
    t1, t2 = parse_markdown_tables(text)
    assert t1.caption == "Table 1. Minimum ages"
    assert t1.header == ["Vaccine", "Min age"]
    assert t1.rows == [["MMR", "12 months*"]]
    assert t1.footnotes == ["* During outbreaks, 6 months."]
    assert t2.rows == [["1", "2"]]


def test_is_footnote():
    assert is_footnote("* Administer at 4-6 years")
    assert is_footnote("Notes: see schedule")
    assert is_footnote("† Not for pregnant women")
    assert not is_footnote("12 months is the minimum age for MMR.")


def test_heading_level_prefers_numbering():
    assert heading_level(Block(doc_id="d", kind="heading", text="3.2.1 Intervals", level=1)) == 3
    assert heading_level(Block(doc_id="d", kind="heading", text="Intervals", level=2)) == 2


def test_pack_text_respects_max_and_overlaps():
    pieces = [_Piece(f"para{i} " + "w " * 9, 1, 10) for i in range(8)]
    chunks = pack_text(pieces, CFG, word_count)
    assert len(chunks) > 1
    for text, _ in chunks:
        assert word_count(text) <= CFG.max_tokens
    # The last paragraph of chunk 1 is repeated at the start of chunk 2 (overlap).
    first_last = chunks[0][0].split("\n\n")[-1]
    assert chunks[1][0].startswith(first_last)


def _blocks() -> list[Block]:
    long_text = " ".join(f"word{i}" for i in range(30))
    return [
        Block(doc_id="d", kind="heading", text="1 Intro", level=1, page=1),
        Block(doc_id="d", kind="text", text="Short intro.", page=1),
        Block(doc_id="d", kind="heading", text="1.1 Detail", level=1, page=1),
        Block(doc_id="d", kind="text", text="Small detail.", page=1),
        Block(doc_id="d", kind="heading", text="2 Schedule", level=1, page=2),
        Block(
            doc_id="d",
            kind="table",
            page=2,
            caption="Table 1",
            header=["Dose", "Age"],
            rows=[["1", "12 months"], ["2", "4 years"]],
            footnotes=["From docling"],
        ),
        Block(doc_id="d", kind="text", text="* Earlier in outbreaks.", page=2),
        Block(doc_id="d", kind="text", text=long_text, page=3),
        Block(doc_id="d", kind="text", text=long_text, page=3),
    ]


def test_chunk_document_structure():
    chunks = chunk_document(_blocks(), "d", "Doc", CFG, word_count)
    ids = [c.chunk_id for c in chunks]
    assert ids == [f"d::{i:04d}" for i in range(len(chunks))]
    # Small intro + detail merged under the common parent section.
    first = chunks[0]
    assert first.section_path == ["1 Intro"]
    assert (
        "Short intro." in first.text
        and "Small detail." in first.text
        and "1.1 Detail" in first.text
    )
    assert first.text.startswith("Doc › 1 Intro")
    table = next(c for c in chunks if c.kind == "table")
    assert table.section_path == ["2 Schedule"]
    assert "| Dose | Age |" in table.text and "Note: * Earlier in outbreaks." in table.text
    assert "Note: From docling" in table.text
    assert "Table 1 - 1: Age: 12 months" in table.retrieval_text
    assert "| Dose |" not in table.retrieval_text
    # Footnote paragraph is attached to every row group of the table, never emitted as text.
    assert not any("Earlier in outbreaks" in c.text for c in chunks if c.kind == "text")
    assert all("Earlier in outbreaks" in c.text for c in chunks if c.kind == "table")
    text_after = [c for c in chunks if c.kind == "text" and c.section_path == ["2 Schedule"]]
    assert text_after and all(c.page_start == 3 for c in text_after)


def test_long_table_split_repeats_header():
    rows = [[f"Dose {i}", "long value " * 4] for i in range(12)]
    blocks = [
        Block(doc_id="d", kind="table", page=1, caption="Big", header=["Dose", "Rule"], rows=rows)
    ]
    chunks = chunk_document(blocks, "d", "Doc", CFG, word_count)
    assert len(chunks) > 1
    assert all("| Dose | Rule |" in c.text for c in chunks)
    # Every row appears exactly once, plus one header line per chunk.
    assert sum(c.text.count("| Dose ") for c in chunks) == 12 + len(chunks)


def test_apply_table_overrides(tmp_path: Path):
    blocks = [
        Block(doc_id="d", kind="text", text="before", page=1),
        Block(
            doc_id="d", kind="table", page=2, header=["x"], rows=[["bad"]], table_source="docling"
        ),
        Block(doc_id="d", kind="text", text="after", page=2),
        Block(doc_id="d", kind="text", text="page3", page=3),
    ]
    (tmp_path / "d_p2.md").write_text("TABLE: Fixed\n| a | b |\n| --- | --- |\n| 1 | 2 |\n")
    (tmp_path / "d_p3.md").write_text("| c |\n| --- |\n| 3 |\n")
    out = apply_table_overrides(blocks, tmp_path)
    kinds = [(b.kind, b.page, b.table_source) for b in out]
    assert kinds == [
        ("text", 1, None),
        ("table", 2, "vision"),
        ("text", 2, None),
        ("text", 3, None),
        ("table", 3, "vision"),
    ]
    assert out[1].caption == "Fixed" and out[1].rows == [["1", "2"]]


def test_table_quality_flags_suspect_tables():
    good = Block(doc_id="d", kind="table", page=1, header=["a", "b"], rows=[["1", "2"], ["3", "4"]])
    bad = Block(doc_id="d", kind="table", page=2, header=["", ""], rows=[["", "x"]])
    report = table_quality([good, bad])
    assert [r["suspect"] for r in report] == [False, True]
    assert "no header" in report[1]["issues"]


def test_provenance_label_includes_publication_year():
    blocks = [
        Block(doc_id="d", kind="heading", text="Recommendations", level=1, page=1),
        Block(doc_id="d", kind="text", text="Use the vaccine.", page=1),
    ]
    (chunk,) = chunk_document(blocks, "d", "Cholera Vaccine", CFG, word_count, doc_year="2021")
    assert chunk.text.startswith("Cholera Vaccine (2021) › Recommendations")
    assert chunk.retrieval_text.startswith("Cholera Vaccine (2021) › Recommendations")
    assert chunk.label == "Cholera Vaccine (2021) › Recommendations"
    assert chunk.doc_title == "Cholera Vaccine" and chunk.doc_year == "2021"
    (no_year,) = chunk_document(blocks, "d", "Cholera Vaccine", CFG, word_count)
    assert no_year.text.startswith("Cholera Vaccine › Recommendations")
