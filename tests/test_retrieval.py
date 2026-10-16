from pathlib import Path

import numpy as np
import pytest
from conftest import make_chunk, make_question

from vaxrag.config import RetrievalConfig
from vaxrag.io_utils import read_jsonl
from vaxrag.retrieval.bm25 import BM25Index, tokenize
from vaxrag.retrieval.dense import DenseIndex
from vaxrag.retrieval.evaluate import evaluate_retrieval, question_metrics, summarize_retrieval
from vaxrag.retrieval.fusion import reciprocal_rank_fusion
from vaxrag.retrieval.hybrid import (
    HybridRetriever,
    build_query,
    knowledge_base_hash,
    retrieval_path,
    retrieve_all,
)
from vaxrag.schemas import RetrievalRecord, RetrievedChunk

DOCS = [
    "MMR vaccine minimum interval between doses is 4 weeks",
    "Hepatitis B vaccine birth dose for infants",
    "Influenza vaccination every season for everyone 6 months and older",
]


class KeywordEmbedder:
    """Fake embedder: bag-of-keywords vectors, so similarity is predictable."""

    name = "fake"
    vocab = ["mmr", "hepatitis", "influenza", "interval", "infants"]

    def __init__(self):
        self.doc_calls = 0

    def _vec(self, text: str) -> np.ndarray:
        words = tokenize(text)
        v = np.array([float(w in words) for w in self.vocab], dtype=np.float32)
        return v / (np.linalg.norm(v) or 1.0)

    def encode_documents(self, texts):
        self.doc_calls += 1
        return np.stack([self._vec(t) for t in texts])

    def encode_queries(self, texts):
        return np.stack([self._vec(t) for t in texts])


class ReverseReranker:
    """Fake reranker that prefers longer documents."""

    def score(self, query, documents):
        return [float(len(d)) for d in documents]


def test_tokenize_and_bm25():
    assert tokenize("The MMR-vaccine is given") == ["mmr-vaccine", "given"]
    index = BM25Index(DOCS)
    top = index.search("hepatitis birth dose", k=2)
    assert top[0][0] == 1


def test_rrf_orders_by_combined_rank_and_is_deterministic():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1, 4]], k=60)
    order = [d for d, _ in fused]
    assert order[0] == 1  # ranks 1 and 2
    assert order[1] == 3  # ranks 3 and 1
    assert set(order) == {1, 2, 3, 4}
    assert reciprocal_rank_fusion([[5], [6]]) == reciprocal_rank_fusion([[5], [6]])
    assert [d for d, _ in reciprocal_rank_fusion([[5], [6]])] == [5, 6]


def test_dense_index_caches_embeddings(tmp_path: Path):
    emb = KeywordEmbedder()
    index = DenseIndex(emb, DOCS, tmp_path)
    assert index.search("influenza season", 1)[0][0] == 2
    DenseIndex(emb, DOCS, tmp_path)
    assert emb.doc_calls == 1  # second index loaded from cache
    DenseIndex(emb, DOCS + ["new"], tmp_path)
    assert emb.doc_calls == 2  # different documents -> new cache entry


def _chunks():
    return [make_chunk(f"d::{i}", text=t) for i, t in enumerate(DOCS)]


def test_hybrid_retriever_stages(tmp_path: Path):
    cfg = RetrievalConfig(use_reranker=False, bm25_candidates=3, dense_candidates=3)
    retriever = HybridRetriever(_chunks(), cfg, tmp_path, embedder=KeywordEmbedder())
    results = retriever.retrieve("MMR interval")
    assert results[0].chunk_id == "d::0"
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    assert {"bm25", "dense", "rrf"} <= set(results[0].stage_scores)

    cfg_rr = RetrievalConfig(use_reranker=True, bm25_candidates=3, dense_candidates=3)
    reranked = HybridRetriever(
        _chunks(), cfg_rr, tmp_path, embedder=KeywordEmbedder(), reranker=ReverseReranker()
    ).retrieve("MMR interval")
    assert reranked[0].chunk_id == "d::2"  # longest document
    assert "rerank" in reranked[0].stage_scores


def test_hybrid_requires_a_first_stage():
    with pytest.raises(ValueError):
        HybridRetriever(_chunks(), RetrievalConfig(use_bm25=False, use_dense=False))


def test_build_query_modes():
    q = make_question()
    assert build_query(q, "stem_only") == q.stem
    assert build_query(q, "stem_and_options").endswith("D. delta")


def test_retrieve_all_resumes(tmp_path: Path):
    cfg = RetrievalConfig(use_dense=False, use_reranker=False)
    retriever = HybridRetriever(_chunks(), cfg)
    out = retrieval_path(tmp_path, cfg, _chunks())
    assert knowledge_base_hash(_chunks()) in out.name
    q1, q2 = make_question("q1"), make_question("q2")
    retrieve_all([q1], retriever, out)
    records = retrieve_all([q1, q2], retriever, out)
    assert [r.question_id for r in records] == ["q1", "q2"]
    assert len(list(read_jsonl(out))) == 2


def test_retrieval_config_hash_ignores_device_only():
    a = RetrievalConfig()
    b = RetrievalConfig.model_validate({"embedder": {"device": "cuda", "batch_size": 64}})
    c = RetrievalConfig(top_k=10)
    assert a.config_hash() == b.config_hash()
    assert a.config_hash() != c.config_hash()


def test_question_metrics():
    m = question_metrics(
        ["a", "b", "c", "x"], {"c", "z"}, {"a": "d1", "c": "d2", "z": "d2"}, ks=(1, 3)
    )
    assert m["rr"] == pytest.approx(1 / 3)
    assert m["hit@1"] == 0 and m["hit@3"] == 1
    assert m["recall@3"] == 0.5
    assert m["doc_hit@1"] == 0 and m["doc_hit@3"] == 1


def test_evaluate_and_summarize():
    q1 = make_question("q1").model_copy(update={"answer_source": "table"})
    q2 = make_question("q2").model_copy(update={"answer_source": "text"})
    recs = [
        RetrievalRecord(
            question_id=q,
            retrieval_config_hash="h",
            query="",
            results=[RetrievedChunk(chunk_id=c, rank=i + 1, score=1.0) for i, c in enumerate(ids)],
        )
        for q, ids in (("q1", ["a", "b"]), ("q2", ["c", "d"]), ("q3", ["a"]))
    ]
    per_q = evaluate_retrieval(
        recs,
        {"q1": ["b"], "q2": ["x"]},
        {"a": "d", "b": "d", "c": "d", "d": "d"},
        [q1, q2],
        ks=(1, 5),
    )
    assert list(per_q["question_id"]) == ["q1", "q2"]  # q3 has no gold
    summary = summarize_retrieval(per_q)
    assert summary.loc[0, "group"] == "all" and summary.loc[0, "MRR"] == pytest.approx(0.25)
    assert set(summary["group"]) == {"all", "table", "text"}
