"""Retrieval metrics against a set of relevant chunks per question.

Metrics per question:

* `hit@k`     - at least one relevant chunk is in the top k.
* `recall@k`  - fraction of relevant chunks in the top k.
* `rr`        - reciprocal rank of the first relevant chunk (0 if none retrieved).
* `doc_hit@k` - at least one top-k chunk comes from a document with a relevant chunk.

Two relevance definitions are reported (column `relevance`):

* `strict`  - annotated gold chunks only. Gold sets are one sufficient set of
  evidence, not every chunk that could answer the question, so these numbers
  are lower bounds. Needs no LLM.
* `lenient` - gold chunks plus chunks the judge found relevant in the pooled
  top-`pool_k` results (TREC-style pooling). Only k <= `pool_k` is reported,
  because deeper chunks were not judged.
"""

from __future__ import annotations

import pandas as pd

from vaxrag.schemas import Question, RetrievalRecord

DEFAULT_KS = (1, 3, 5, 10, 20)


def question_metrics(
    retrieved_ids: list[str],
    gold_ids: set[str],
    chunk_doc: dict[str, str],
    ks: tuple[int, ...] = DEFAULT_KS,
) -> dict[str, float]:
    """Compute retrieval metrics for one question."""
    gold_docs = {chunk_doc[c] for c in gold_ids if c in chunk_doc}
    metrics: dict[str, float] = {"n_gold": float(len(gold_ids))}
    first_rank = next((r for r, cid in enumerate(retrieved_ids, start=1) if cid in gold_ids), None)
    metrics["rr"] = 1.0 / first_rank if first_rank else 0.0
    for k in ks:
        top = retrieved_ids[:k]
        found = gold_ids.intersection(top)
        metrics[f"hit@{k}"] = float(bool(found))
        metrics[f"recall@{k}"] = len(found) / len(gold_ids) if gold_ids else float("nan")
        metrics[f"doc_hit@{k}"] = float(any(chunk_doc.get(c) in gold_docs for c in top))
    return metrics


def evaluate_retrieval(
    records: list[RetrievalRecord],
    gold: dict[str, list[str]],
    chunk_doc: dict[str, str],
    questions: list[Question],
    ks: tuple[int, ...] = DEFAULT_KS,
    relevance_label: str = "strict",
) -> pd.DataFrame:
    """Return one row of metrics per question that has at least one relevant chunk.

    `gold` maps question IDs to their relevant chunk IDs under the definition
    named by `relevance_label`.
    """
    meta = {q.question_id: q for q in questions}
    rows = []
    for record in records:
        gold_ids = set(gold.get(record.question_id, []))
        if not gold_ids:
            continue
        q = meta.get(record.question_id)
        row = {
            "relevance": relevance_label,
            "question_id": record.question_id,
            "subset": q.subset if q else None,
            "answer_source": q.answer_source if q else None,
            "vaccine": q.vaccine if q else None,
        }
        row.update(question_metrics([r.chunk_id for r in record.results], gold_ids, chunk_doc, ks))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_retrieval(
    per_question: pd.DataFrame, by: str | None = "answer_source"
) -> pd.DataFrame:
    """Average metrics per relevance definition, overall and per stratum.

    MRR is the mean of `rr`. Metrics that are missing for a relevance
    definition (e.g. lenient k > pool_k) stay NaN.
    """
    if per_question.empty:
        return pd.DataFrame()
    if "relevance" not in per_question:
        per_question = per_question.assign(relevance="strict")
    metric_cols = [c for c in per_question.columns if "@" in c or c == "rr"]
    frames = []
    for label, part in per_question.groupby("relevance", sort=False):
        overall = part[metric_cols].mean().to_frame().T
        overall.insert(0, "relevance", label)
        overall.insert(1, "group", "all")
        overall.insert(2, "n", len(part))
        frames.append(overall)
        if by and by in part and part[by].notna().any():
            grouped = part.groupby(by)[metric_cols].mean().reset_index()
            grouped = grouped.rename(columns={by: "group"})
            grouped.insert(0, "relevance", label)
            grouped.insert(2, "n", part.groupby(by).size().values)
            frames.append(grouped)
    return pd.concat(frames, ignore_index=True).rename(columns={"rr": "MRR"})
