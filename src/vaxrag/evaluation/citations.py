"""Automatic citation metrics.

* **Validity** (A1/A2): does the cited source ID exist in the prompt, and does
  the quote appear (nearly) verbatim in that excerpt? Fuzzy partial matching
  tolerates small differences such as whitespace or punctuation.
* **Strict correctness** (A1/A2): citation recall (a gold chunk is cited) and
  citation precision (share of cited chunks that are gold chunks). Gold sets
  are not exhaustive - other chunks may support the answer too - so these are
  lower bounds. Judge-based citation support is in `evidence.py`.
* **Fabrication** (A0): does a citation written from memory match a document
  title (and section) in the knowledge-base manifest? Citations of real
  documents outside the knowledge base are reported as "unverified", not as
  proven fabrications.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from rapidfuzz import fuzz

from vaxrag.schemas import Chunk, Reference


def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace and quote characters for matching."""
    text = text.lower().replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip()


def quote_score(quote: str, passage: str) -> float:
    """Best fuzzy partial-match score (0-100) of a quote inside a passage."""
    q, p = _normalize(quote), _normalize(passage)
    if not q:
        return 0.0
    if q in p:
        return 100.0
    return float(fuzz.partial_ratio(q, p))


def citation_validity(
    references: list[Reference],
    source_map: dict[str, str],
    texts: Mapping[str, str],
    threshold: float = 90.0,
) -> dict[str, float | int | bool]:
    """Compute citation validity metrics for one answer with context.

    `source_map` maps the source IDs shown in the prompt (S1.., D1..) to chunk
    or document IDs, and `texts` maps those IDs to the text the quote must
    appear in (chunk text, or the parsed full text of a PDF for arm A3).
    """
    n = len(references)
    known = [r for r in references if r.source_id.strip().strip("[]") in source_map]
    verbatim = 0
    for ref in known:
        text = texts.get(source_map[ref.source_id.strip().strip("[]")])
        if text is not None and quote_score(ref.quote, text) >= threshold:
            verbatim += 1
    return {
        "n_references": n,
        "n_known_source_ids": len(known),
        "n_verbatim_quotes": verbatim,
        "all_quotes_verbatim": n > 0 and verbatim == n,
        "share_quotes_verbatim": verbatim / n if n else float("nan"),
    }


def cited_chunk_ids(references: list[Reference], source_map: dict[str, str]) -> list[str]:
    """Chunk IDs referenced by an answer (unknown source IDs are ignored)."""
    out: list[str] = []
    for ref in references:
        cid = source_map.get(ref.source_id.strip().strip("[]"))
        if cid and cid not in out:
            out.append(cid)
    return out


def citation_correctness(cited: list[str], gold: list[str]) -> dict[str, float]:
    """Compute strict citation recall (any gold chunk cited) and precision (share of cited gold)."""
    if not gold:
        return {"citation_recall_strict": float("nan"), "citation_precision_strict": float("nan")}
    gold_set = set(gold)
    hits = [c for c in cited if c in gold_set]
    return {
        "citation_recall_strict": float(bool(hits)),
        "citation_precision_strict": len(hits) / len(cited) if cited else float("nan"),
    }


def match_memory_citation(
    reference: Reference,
    titles: dict[str, str],
    sections: dict[str, list[str]],
    title_threshold: float = 80.0,
) -> dict[str, str | float | None]:
    """Classify a from-memory (A0) citation against the knowledge base.

    `source_id` is expected to look like "Title › Section"; anything after the
    first separator is treated as the section. Returns the best matching doc
    and a status: "doc_and_section", "doc_only" or "unverified".
    """
    parts = re.split(r"\s*(?:›|>|\||:|—|–| - )\s*", reference.source_id, maxsplit=1)
    title_part = parts[0]
    section_part = parts[1] if len(parts) > 1 else ""
    best_doc, best_score = None, 0.0
    for doc_id, title in titles.items():
        score = float(fuzz.token_set_ratio(_normalize(title_part), _normalize(title)))
        if score > best_score:
            best_doc, best_score = doc_id, score
    if best_doc is None or best_score < title_threshold:
        return {"matched_doc": None, "title_score": best_score, "status": "unverified"}
    status = "doc_only"
    if section_part:
        for section in sections.get(best_doc, []):
            if (
                fuzz.token_set_ratio(_normalize(section_part), _normalize(section))
                >= title_threshold
            ):
                status = "doc_and_section"
                break
    return {"matched_doc": best_doc, "title_score": best_score, "status": status}


def section_index(chunks: list[Chunk]) -> dict[str, list[str]]:
    """All distinct section names (each level) per document, for A0 matching."""
    index: dict[str, set[str]] = {}
    for chunk in chunks:
        index.setdefault(chunk.doc_id, set()).update(chunk.section_path)
    return {doc: sorted(names) for doc, names in index.items()}


def excerpt_around(quote: str, text: str, window: int = 1500) -> str:
    """Return the passage of `text` that best matches `quote`, with `window` characters around it.

    Used to show a judge where in a full document (arm A3) a quote comes from.
    Returns an empty string when the quote is empty.
    """
    if not quote.strip() or not text:
        return ""
    alignment = fuzz.partial_ratio_alignment(quote.lower(), text.lower())
    start = max(alignment.dest_start - window, 0)
    end = min(alignment.dest_end + window, len(text))
    prefix = "... " if start > 0 else ""
    suffix = " ..." if end < len(text) else ""
    return f"{prefix}{text[start:end]}{suffix}"
