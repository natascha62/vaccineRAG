"""Merge run logs into one analysis table (`results/analysis/all_calls.parquet`).

For every final call the table contains identification fields, question
metadata and annotations, correctness, evidence coverage of the context,
citation metrics and judge verdicts.

Evidence-related columns come in up to three flavours (see `evidence.py`):

* `*_strict`  - compared with annotated gold chunks only (a lower bound,
  because gold sets are not exhaustive);
* `*_lenient` - gold chunks plus chunks the judge found relevant (pooled);
* `*_judged`  - a direct judge verdict (context sufficiency, citation support).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from vaxrag.evaluation.citations import (
    citation_correctness,
    citation_validity,
    cited_chunk_ids,
    match_memory_citation,
    section_index,
)
from vaxrag.evaluation.evidence import (
    judged_citation_support,
    judged_membership,
    lenient_relevant_sets,
    sufficiency_key,
)
from vaxrag.evaluation.judge import GROUNDING_FILE
from vaxrag.evaluation.scoring import final_calls, score_calls
from vaxrag.io_utils import read_jsonl, read_models
from vaxrag.schemas import CallRecord, Chunk, Question

QUESTION_FIELDS = [
    "subset",
    "vaccine",
    "category",
    "question_type",
    "population",
    "answer_source",
    "kb_covered",
    "version_status",
    "multi_answer",
]


def load_call_records(run_dirs: list[Path]) -> list[CallRecord]:
    """Read all attempts from the `calls.jsonl` of each run folder."""
    records: list[CallRecord] = []
    for run_dir in run_dirs:
        records.extend(read_models(Path(run_dir) / "calls.jsonl", CallRecord))
    return records


def load_selection_costs(run_dirs: list[Path]) -> dict[str, float]:
    """Total cost of all A3 document-selection attempts per call_id."""
    costs: dict[str, float] = {}
    for d in run_dirs:
        for row in read_jsonl(Path(d) / "selections.jsonl"):
            costs[row["call_id"]] = costs.get(row["call_id"], 0.0) + (row.get("cost_usd") or 0.0)
    return costs


def load_grounding(run_dirs: list[Path]) -> pd.DataFrame:
    """Latest successful grounding verdict per call across runs."""
    rows = [
        row
        for d in run_dirs
        for row in read_jsonl(Path(d) / GROUNDING_FILE)
        if row["status"] == "ok"
    ]
    if not rows:
        return pd.DataFrame(columns=["call_id", "grounding_verdict"])
    df = pd.DataFrame(rows).sort_values("timestamp").groupby("call_id").tail(1)
    return df[["call_id", "verdict"]].rename(columns={"verdict": "grounding_verdict"})


def _citation_fields(
    record: CallRecord,
    gold: list[str],
    chunks: dict[str, Chunk],
    titles: dict[str, str],
    sections: dict[str, list[str]],
    relevant: dict[str, set[str]],
    relevance: dict[str, str],
    quote_threshold: float,
    title_threshold: float,
    doc_texts: dict[str, str],
) -> dict:
    """Citation metrics appropriate for the record's arm.

    A3 cites whole documents, so only validity (is the quote in the cited
    document's parsed text?) is computed; chunk-level metrics do not apply.
    """
    if record.parsed is None:
        return {}
    refs = record.parsed.references
    if record.arm == "A0":
        statuses = [
            match_memory_citation(r, titles, sections, title_threshold)["status"] for r in refs
        ]
        n = len(statuses)
        return {
            "n_references": n,
            "memory_citation_doc_match_share": (
                sum(s != "unverified" for s in statuses) / n if n else float("nan")
            ),
            "memory_citation_section_match_share": (
                sum(s == "doc_and_section" for s in statuses) / n if n else float("nan")
            ),
        }
    if record.arm == "A3":
        return dict(citation_validity(refs, record.context_labels, doc_texts, quote_threshold))
    cited = cited_chunk_ids(refs, record.context_labels)
    texts = {cid: c.text for cid, c in chunks.items()}
    fields = dict(citation_validity(refs, record.context_labels, texts, quote_threshold))
    fields.update(citation_correctness(cited, gold))
    fields.update(judged_citation_support(record.question_id, cited, relevant, relevance))
    return fields


def build_analysis_table(
    run_dirs: list[Path],
    questions: list[Question],
    chunks: list[Chunk],
    gold: dict[str, list[str]],
    titles: dict[str, str],
    relevance: dict[str, str] | None = None,
    sufficiency: dict[str, str] | None = None,
    positive_relevance: tuple[str, ...] = ("supports",),
    quote_threshold: float = 90.0,
    title_threshold: float = 80.0,
    faulty_tables: set[str] | None = None,
    doc_texts: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build the per-call analysis table from one or more run folders.

    `doc_texts` (doc_id -> parsed full text) is used to check A3 quotes.

    `relevance` maps chunk-relevance keys to verdicts and `sufficiency` maps
    context-sufficiency keys to verdicts (both from `load_evidence_verdicts`).
    `faulty_tables` holds chunk IDs reviewed as wrongly parsed (None if no
    table review exists yet).
    """
    records = load_call_records(run_dirs)
    if not records:
        return pd.DataFrame()
    relevance = relevance or {}
    sufficiency = sufficiency or {}
    relevant = lenient_relevant_sets(gold, relevance, positive_relevance)
    by_id = {q.question_id: q for q in questions}
    chunk_map = {c.chunk_id: c for c in chunks}
    sections = section_index(chunks)
    doc_texts = doc_texts or {}
    selection_costs = load_selection_costs(run_dirs)

    attempts = pd.DataFrame(
        {
            "call_id": [r.call_id for r in records],
            "attempt": [r.attempt for r in records],
            "status": [r.status for r in records],
            "answer": [r.answer for r in records],
            "question_id": [r.question_id for r in records],
            "_idx": range(len(records)),
        }
    )
    final = score_calls(final_calls(attempts), by_id)
    n_attempts = attempts.groupby("call_id").size().rename("n_attempts")

    rows = []
    for _, row in final.iterrows():
        r = records[row["_idx"]]
        q = by_id[r.question_id]
        q_gold = gold.get(r.question_id, [])
        has_context = r.arm != "A0" and bool(r.context_chunk_ids)
        gold_docs = {chunk_map[c].doc_id for c in q_gold if c in chunk_map}
        cited = (
            cited_chunk_ids(r.parsed.references, r.context_labels)
            if has_context and r.parsed is not None
            else []
        )
        table_ids = [
            c for c in r.context_chunk_ids if c in chunk_map and chunk_map[c].kind == "table"
        ]
        item = {
            "call_id": r.call_id,
            "run_id": r.run_id,
            "question_id": r.question_id,
            "model": r.model,
            "model_version": r.model_version,
            "arm": r.arm,
            "rep": r.rep,
            "prompt_version": r.prompt_version,
            "git_commit": r.git_commit,
            "timestamp": r.timestamp,
            "status": r.status,
            "correct": bool(row["correct"]),
            "parse_failed": bool(row["parse_failed"]),
            "refused": bool(row["refused"]),
            "answer": json.dumps(r.answer),
            "correct_answer": json.dumps(q.correct),
            "answer_basis": r.parsed.answer_basis if r.parsed else None,
            "context_chunk_ids": json.dumps(r.context_chunk_ids),
            "cited_chunk_ids": json.dumps(cited),
            "context_has_table": bool(table_ids) if has_context else None,
            "context_has_faulty_table": (
                any(c in faulty_tables for c in table_ids)
                if has_context and faulty_tables is not None
                else None
            ),
            "has_gold": bool(q_gold),
            "gold_in_context_strict": (
                bool(set(q_gold) & set(r.context_chunk_ids)) if q_gold and has_context else None
            ),
            "relevant_in_context_lenient": (
                judged_membership(r.question_id, r.context_chunk_ids, relevant, relevance)
                if has_context
                else None
            ),
            "context_sufficiency_judged": (
                sufficiency.get(sufficiency_key(r.question_id, r.context_chunk_ids))
                if has_context
                else None
            ),
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "reasoning_tokens": r.reasoning_tokens,
            "latency_s": r.latency_s,
            "cache_read_tokens": r.cache_read_tokens,
            "cache_write_tokens": r.cache_write_tokens,
            # A3 cost includes the document-selection step.
            "cost_usd": (
                (r.cost_usd or 0.0) + selection_costs.get(r.call_id, 0.0)
                if r.cost_usd is not None or r.call_id in selection_costs
                else None
            ),
            "selected_doc_ids": json.dumps(r.selected_doc_ids),
            "n_selected_docs": len(r.selected_doc_ids) if r.arm == "A3" else None,
            "gold_doc_selected": (
                bool(gold_docs & set(r.selected_doc_ids))
                if r.arm == "A3" and gold_docs and r.selected_doc_ids
                else None
            ),
        }
        item.update({f: getattr(q, f) for f in QUESTION_FIELDS})
        item.update(
            _citation_fields(
                r,
                q_gold,
                chunk_map,
                titles,
                sections,
                relevant,
                relevance,
                quote_threshold,
                title_threshold,
                doc_texts,
            )
        )
        rows.append(item)
    table = pd.DataFrame(rows).merge(n_attempts, left_on="call_id", right_index=True, how="left")
    return table.merge(load_grounding(run_dirs), on="call_id", how="left")
