"""Manual annotation files: per-question labels and gold (oracle) passages.

Two CSV files are maintained by the research team:

* `question_annotations.csv` - one row per question with the labels the
  analysis stratifies on (question type, population, table vs. text answer
  source, knowledge-base coverage, guideline-version status).
* `gold_passages.csv` - one row per (question, gold passage). Annotators
  locate the passage by document ID plus any of section path, page and a short
  verbatim quote. `resolve_gold_chunks` maps these to chunk IDs after chunking,
  so annotations survive changes to chunking parameters.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from vaxrag.schemas import Chunk, Question

ANNOTATION_COLUMNS = [
    "question_id",
    "subset",
    "vaccine",
    "category",
    "stem",
    "question_type",
    "population",
    "answer_source",
    "kb_covered",
    "version_status",
    "notes",
]
GOLD_COLUMNS = ["question_id", "doc_id", "section", "page", "quote", "chunk_id", "notes"]

ALLOWED_VALUES: dict[str, set[str]] = {
    "question_type": {"fact", "rule_application", "multi_condition"},
    "answer_source": {"table", "text", "both"},
    "version_status": {"current", "outdated", "unmappable"},
}


class GoldResolution(BaseModel):
    """Gold chunk IDs for one question and any gold rows that could not be resolved."""

    question_id: str
    chunk_ids: list[str] = Field(default_factory=list)
    unresolved: list[dict] = Field(default_factory=list)
    ambiguous: bool = False


def create_annotation_templates(
    questions: list[Question], annotations_path: str | Path, gold_path: str | Path
) -> tuple[int, int]:
    """Create or extend the annotation CSVs without overwriting existing work.

    Returns the number of question rows added and whether the gold file was created.
    """
    annotations_path, gold_path = Path(annotations_path), Path(gold_path)
    annotations_path.parent.mkdir(parents=True, exist_ok=True)
    if annotations_path.exists():
        existing = pd.read_csv(annotations_path, dtype=str, keep_default_na=False)
    else:
        existing = pd.DataFrame(columns=ANNOTATION_COLUMNS)
    known = set(existing["question_id"])
    new_rows = [
        {
            "question_id": q.question_id,
            "subset": q.subset,
            "vaccine": q.vaccine,
            "category": q.category or "",
            "stem": q.stem,
        }
        for q in questions
        if q.question_id not in known
    ]
    if new_rows:
        merged = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
        merged = merged.reindex(columns=ANNOTATION_COLUMNS).fillna("")
        merged.to_csv(annotations_path, index=False)
    gold_created = 0
    if not gold_path.exists():
        pd.DataFrame(columns=GOLD_COLUMNS).to_csv(gold_path, index=False)
        gold_created = 1
    return len(new_rows), gold_created


def _parse_bool(value: str) -> bool | None:
    """Parse yes/no style cells; blanks become None."""
    value = value.strip().lower()
    if value in {"yes", "y", "true", "1"}:
        return True
    if value in {"no", "n", "false", "0"}:
        return False
    if value == "":
        return None
    raise ValueError(f"Cannot interpret '{value}' as yes/no")


def validate_annotations(df: pd.DataFrame) -> list[str]:
    """Return a list of human-readable problems found in the annotation table."""
    problems: list[str] = []
    for column, allowed in ALLOWED_VALUES.items():
        values = df[column].fillna("").str.strip()
        bad = df.loc[(values != "") & ~values.isin(allowed), ["question_id", column]]
        for _, row in bad.iterrows():
            problems.append(
                f"{row['question_id']}: {column}='{row[column]}' not in {sorted(allowed)}"
            )
    for _, row in df.iterrows():
        try:
            _parse_bool(str(row.get("kb_covered", "")))
        except ValueError as exc:
            problems.append(f"{row['question_id']}: {exc}")
    return problems


def apply_annotations(questions: list[Question], annotations_path: str | Path) -> list[Question]:
    """Return copies of `questions` with annotation fields filled from the CSV."""
    path = Path(annotations_path)
    if not path.exists():
        return questions
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    problems = validate_annotations(df)
    if problems:
        raise ValueError("Invalid annotations:\n" + "\n".join(problems[:20]))
    by_id = df.set_index("question_id").to_dict(orient="index")
    updated: list[Question] = []
    for q in questions:
        row = by_id.get(q.question_id)
        if row is None:
            updated.append(q)
            continue
        updated.append(
            q.model_copy(
                update={
                    "question_type": row.get("question_type") or None,
                    "population": row.get("population") or None,
                    "answer_source": row.get("answer_source") or None,
                    "kb_covered": _parse_bool(row.get("kb_covered", "")),
                    "version_status": row.get("version_status") or None,
                }
            )
        )
    return updated


def _norm_section(text: str) -> str:
    """Normalize a section path for loose matching (case, separators, whitespace)."""
    text = re.sub(r"\s*(›|>|/|\|)\s*", " > ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def resolve_gold_row(row: dict, chunks: list[Chunk], quote_threshold: float = 90.0) -> list[str]:
    """Return chunk IDs matching one gold-passage annotation row.

    Filters are applied in order: explicit chunk_id, document, section prefix,
    page, then the verbatim quote (fuzzy partial match). An empty list means the
    row could not be resolved.
    """
    if str(row.get("chunk_id") or "").strip():
        return [c.strip() for c in str(row["chunk_id"]).split(";") if c.strip()]
    candidates = [c for c in chunks if c.doc_id == str(row.get("doc_id", "")).strip()]
    section = str(row.get("section") or "").strip()
    if section:
        target = _norm_section(section)
        candidates = [
            c
            for c in candidates
            if _norm_section(" > ".join(c.section_path)).startswith(target)
            or target in _norm_section(" > ".join(c.section_path))
        ]
    page = str(row.get("page") or "").strip()
    if page:
        p = int(float(page))
        candidates = [
            c
            for c in candidates
            if c.page_start is not None and c.page_start <= p <= (c.page_end or c.page_start)
        ]
    quote = str(row.get("quote") or "").strip()
    if quote:
        scored = [(fuzz.partial_ratio(quote.lower(), c.text.lower()), c) for c in candidates]
        candidates = [c for score, c in scored if score >= quote_threshold]
    return [c.chunk_id for c in candidates]


def resolve_gold_chunks(
    gold_path: str | Path, chunks: list[Chunk], max_chunks_per_row: int = 3
) -> list[GoldResolution]:
    """Map every annotated gold passage to chunk IDs.

    Rows matching more than `max_chunks_per_row` chunks are flagged as ambiguous
    (add a page or quote to narrow them down).
    """
    path = Path(gold_path)
    if not path.exists():
        return []
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    results: dict[str, GoldResolution] = {}
    for row in df.to_dict(orient="records"):
        qid = row["question_id"].strip()
        if not qid:
            continue
        res = results.setdefault(qid, GoldResolution(question_id=qid))
        ids = resolve_gold_row(row, chunks)
        if not ids:
            res.unresolved.append(row)
        if len(ids) > max_chunks_per_row:
            res.ambiguous = True
        for cid in ids:
            if cid not in res.chunk_ids:
                res.chunk_ids.append(cid)
    return list(results.values())
