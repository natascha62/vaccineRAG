"""Tests for the targeted table-parsing review."""

import json
from pathlib import Path

import pandas as pd
from conftest import make_chunk, word_count

from vaxrag.config import ChunkingConfig
from vaxrag.evaluation.table_review import (
    build_review_queue,
    faulty_tables,
    merge_with_existing,
    parse_error_table,
)
from vaxrag.kb.chunking import chunk_document
from vaxrag.schemas import Block


def _chunks():
    return {
        "t1": make_chunk("t1", kind="table", pages=(3, 4)),
        "t2": make_chunk("t2", kind="table", pages=(7, 7)),
        "t3": make_chunk("t3", kind="table", pages=(9, 9)),
        "x1": make_chunk("x1"),
    }


def _row(qid, model, arm, correct, context, cited=(), status="ok"):
    return {
        "call_id": f"{qid}|{model}|{arm}",
        "question_id": qid,
        "model": model,
        "arm": arm,
        "status": status,
        "correct": correct,
        "context_chunk_ids": json.dumps(list(context)),
        "cited_chunk_ids": json.dumps(list(cited)),
    }


def _table():
    return pd.DataFrame(
        [
            _row("q1", "m1", "A1", False, ["t1", "t2", "x1"], cited=["t2"]),
            _row("q1", "m2", "A1", False, ["t1", "t2", "x1"]),
            _row("q2", "m1", "A1", False, ["t1", "x1"]),
            _row("q2", "m2", "A1", True, ["t3"]),  # correct -> not queued
            _row("q3", "m1", "A2", False, ["t3"]),  # not a review arm
            _row("q4", "m1", "A1", False, ["t3"], status="parse_error"),  # not a reading error
        ]
    )


def test_queue_lists_each_table_once_by_suspicion():
    queue = build_review_queue(_table(), _chunks(), {"q2": {"t1"}}, kb_hash="h1")
    assert list(queue["chunk_id"]) == ["t2", "t1"]  # t2 was cited; t3 never qualifies
    t2, t1 = queue.iloc[0], queue.iloc[1]
    assert (t2["priority"], t2["n_wrong_calls_citing"], t2["n_models_wrong"]) == (1, 1, 2)
    assert (t1["n_wrong_calls"], t1["relevant"], t1["question_ids"]) == (3, True, "q1;q2")
    assert (t1["page_start"], t1["page_end"], t1["kb_hash"]) == (3, 4, "h1")
    assert set(queue["parse_ok"]) == {""}


def test_queue_can_include_other_arms():
    queue = build_review_queue(_table(), _chunks(), {}, kb_hash="h", arms=("A1", "A2"))
    assert "t3" in set(queue["chunk_id"])


def test_empty_queue_has_all_columns():
    queue = build_review_queue(_table().iloc[3:4], _chunks(), {}, kb_hash="h")
    assert queue.empty and "parse_ok" in queue.columns


def test_reviews_survive_rebuilds_and_stay_tied_to_kb(tmp_path: Path):
    path = tmp_path / "review.csv"
    queue = build_review_queue(_table(), _chunks(), {}, kb_hash="h1")
    queue.loc[queue["chunk_id"] == "t1", ["parse_ok", "error_description"]] = ["no", "row shift"]
    queue.loc[queue["chunk_id"] == "t2", "parse_ok"] = "yes"
    queue.to_csv(path, index=False)

    # Rebuild with fewer wrong answers: t2 drops out of the queue but its review is kept.
    smaller = build_review_queue(_table().iloc[2:3], _chunks(), {}, kb_hash="h1")
    merged = merge_with_existing(smaller, path)
    by_id = merged.set_index("chunk_id")
    assert (
        by_id.loc["t1", "parse_ok"] == "no" and by_id.loc["t1", "error_description"] == "row shift"
    )
    assert by_id.loc["t2", "parse_ok"] == "yes"
    merged.to_csv(path, index=False)

    assert faulty_tables(path, "h1") == {"t1"}
    assert faulty_tables(path, "other-kb") == set()  # reviews of another chunk set are ignored
    assert faulty_tables(tmp_path / "missing.csv", "h1") is None
    # A new knowledge base gets fresh, unreviewed rows.
    rebuilt = merge_with_existing(build_review_queue(_table(), _chunks(), {}, kb_hash="h2"), path)
    assert set(rebuilt.loc[rebuilt["kb_hash"] == "h2", "parse_ok"]) == {""}


def test_parse_error_table():
    df = pd.DataFrame(
        {
            "call_id": ["a", "b", "c", "d"],
            "model": ["m"] * 4,
            "arm": ["A1"] * 4,
            "status": ["ok"] * 4,
            "correct": [False, False, False, True],
            "context_has_table": [True, True, False, True],
            "context_has_faulty_table": [True, False, False, True],
        }
    )
    row = parse_error_table(df).iloc[0]
    assert (row["n_wrong"], row["n_wrong_with_table"], row["n_wrong_with_faulty_table"]) == (
        3,
        2,
        1,
    )
    assert row["share_wrong_with_faulty_table"] == 1 / 3
    assert parse_error_table(df.assign(context_has_faulty_table=None)).empty


def test_table_chunks_record_their_source():
    blocks = [
        Block(doc_id="d", kind="table", page=1, header=["a"], rows=[["1"]], table_source="vision"),
        Block(doc_id="d", kind="text", text="plain text", page=1),
    ]
    chunks = chunk_document(blocks, "d", "Doc", ChunkingConfig(), word_count)
    assert [(c.kind, c.table_source) for c in chunks] == [("table", "vision"), ("text", None)]
