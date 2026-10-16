"""Targeted review of table parsing errors.

Parsed tables are not all checked by hand. A table chunk enters the review
queue only when it was in the context of an incorrect answer in the review
arms (default: A1, standard RAG). The queue lists each table once, ordered by
how strongly the errors point at it:

1. it was cited by a wrong answer (the model relied on it),
2. it is relevant to a question it was retrieved for (gold or judge-relevant),
3. the number of models that answered wrongly with it in the context,
4. the number of wrong answers with it in the context.

Reviewers compare the table with the PDF page and fill `parse_ok` (yes/no)
and `error_description`. Existing review entries are preserved when the queue
is rebuilt. Each row stores the knowledge-base hash, so reviews of an older
chunk set are never applied to a re-chunked knowledge base.

Methodology: tables found faulty during the pilot are fixed before the
knowledge base is frozen. During the main experiment the review is used only
for error attribution (faulty table vs. model misreading a correct table).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from vaxrag.schemas import Chunk

QUEUE_COLUMNS = [
    "priority",
    "chunk_id",
    "kb_hash",
    "doc_id",
    "page_start",
    "page_end",
    "table_source",
    "label",
    "n_wrong_calls",
    "n_wrong_calls_citing",
    "n_models_wrong",
    "relevant",
    "models",
    "question_ids",
    "page_images",
]
REVIEWER_COLUMNS = ["parse_ok", "error_description", "reviewer", "review_date"]


def build_review_queue(
    table: pd.DataFrame,
    chunks: dict[str, Chunk],
    relevant: dict[str, set[str]],
    kb_hash: str,
    arms: tuple[str, ...] = ("A1",),
) -> pd.DataFrame:
    """List table chunks that appeared in the context of wrong answers, most suspicious first.

    `table` is the analysis table (`all_calls.parquet`); only answers that were
    parsed successfully (status `ok`) and scored incorrect are considered.
    `relevant` maps question IDs to relevant chunk IDs (gold plus judged).
    """
    wrong = table[table["arm"].isin(arms) & (table["status"] == "ok") & ~table["correct"]]
    stats: dict[str, dict] = defaultdict(
        lambda: {"calls": 0, "citing": 0, "models": set(), "questions": set(), "relevant": False}
    )
    for row in wrong.itertuples(index=False):
        context = json.loads(row.context_chunk_ids)
        cited = (
            set(json.loads(row.cited_chunk_ids)) if isinstance(row.cited_chunk_ids, str) else set()
        )
        for cid in context:
            chunk = chunks.get(cid)
            if chunk is None or chunk.kind != "table":
                continue
            entry = stats[cid]
            entry["calls"] += 1
            entry["citing"] += int(cid in cited)
            entry["models"].add(row.model)
            entry["questions"].add(row.question_id)
            entry["relevant"] |= cid in relevant.get(row.question_id, set())
    rows = []
    for cid, entry in stats.items():
        chunk = chunks[cid]
        rows.append(
            {
                "chunk_id": cid,
                "kb_hash": kb_hash,
                "doc_id": chunk.doc_id,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "table_source": chunk.table_source,
                "label": chunk.label,
                "n_wrong_calls": entry["calls"],
                "n_wrong_calls_citing": entry["citing"],
                "n_models_wrong": len(entry["models"]),
                "relevant": entry["relevant"],
                "models": ";".join(sorted(entry["models"])),
                "question_ids": ";".join(sorted(entry["questions"])),
                "page_images": "",
            }
        )
    if not rows:
        return pd.DataFrame(columns=QUEUE_COLUMNS + REVIEWER_COLUMNS)
    queue = pd.DataFrame(rows).sort_values(
        ["n_wrong_calls_citing", "relevant", "n_models_wrong", "n_wrong_calls", "chunk_id"],
        ascending=[False, False, False, False, True],
    )
    queue.insert(0, "priority", range(1, len(queue) + 1))
    for col in REVIEWER_COLUMNS:
        queue[col] = ""
    return queue[QUEUE_COLUMNS + REVIEWER_COLUMNS].reset_index(drop=True)


def merge_with_existing(queue: pd.DataFrame, existing_path: str | Path) -> pd.DataFrame:
    """Carry reviewer columns over from an existing review file.

    Reviews are matched on (chunk_id, kb_hash). Previously reviewed tables that
    are no longer in the queue are kept at the end so no review is lost.
    """
    path = Path(existing_path)
    if not path.exists():
        return queue
    existing = pd.read_csv(path, dtype=str, keep_default_na=False)
    if existing.empty:
        return queue
    reviews = existing.set_index(["chunk_id", "kb_hash"])[REVIEWER_COLUMNS]
    merged = queue.copy()
    for i, row in merged.iterrows():
        key = (row["chunk_id"], row["kb_hash"])
        if key in reviews.index:
            for col in REVIEWER_COLUMNS:
                merged.at[i, col] = reviews.loc[key, col]
    queued = set(zip(merged["chunk_id"], merged["kb_hash"], strict=True))
    reviewed = existing[existing["parse_ok"].str.strip() != ""]
    leftover = reviewed[
        [
            (c, h) not in queued
            for c, h in zip(reviewed["chunk_id"], reviewed["kb_hash"], strict=True)
        ]
    ]
    if leftover.empty:
        return merged
    leftover = leftover.reindex(columns=QUEUE_COLUMNS + REVIEWER_COLUMNS).assign(priority="")
    return pd.concat([merged.astype(str), leftover], ignore_index=True)


def faulty_tables(review_path: str | Path, kb_hash: str) -> set[str] | None:
    """Chunk IDs reviewed as wrongly parsed for the current knowledge base.

    Returns None when no review file exists (attribution not possible).
    """
    path = Path(review_path)
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if df.empty:
        return set()
    current = df[df["kb_hash"] == kb_hash]
    return set(current.loc[current["parse_ok"].str.strip().str.lower() == "no", "chunk_id"])


def parse_error_table(table: pd.DataFrame, arms: tuple[str, ...] = ("A1",)) -> pd.DataFrame:
    """Per model: how many wrong answers had a table, and a reviewed faulty table, in context."""
    needed = {"context_has_table", "context_has_faulty_table"}
    if not needed <= set(table.columns):
        return pd.DataFrame()
    wrong = table[table["arm"].isin(arms) & (table["status"] == "ok") & ~table["correct"]]
    if wrong.empty or wrong["context_has_faulty_table"].isna().all():
        return pd.DataFrame()
    grouped = wrong.groupby(["model", "arm"]).agg(
        n_wrong=("call_id", "count"),
        n_wrong_with_table=("context_has_table", "sum"),
        n_wrong_with_faulty_table=("context_has_faulty_table", "sum"),
    )
    grouped["share_wrong_with_faulty_table"] = (
        grouped["n_wrong_with_faulty_table"] / grouped["n_wrong"]
    )
    return grouped.reset_index()
