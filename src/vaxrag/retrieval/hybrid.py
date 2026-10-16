"""Hybrid retrieval pipeline: BM25 + dense -> RRF -> cross-encoder rerank.

Retrieval is independent of the LLM, so it is computed once per question and
retrieval configuration and cached in `data/processed/retrieval/<hash>.jsonl`.
All LLM runs then read the same frozen retrieval results, which also makes
retrieval identical across machines (CPU, MPS or CUDA). The cache file name
contains both the retrieval-config hash and a hash of the chunk set.
"""

from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from vaxrag.config import RetrievalConfig
from vaxrag.io_utils import append_jsonl, read_models, stable_hash
from vaxrag.retrieval.bm25 import BM25Index
from vaxrag.retrieval.dense import DenseIndex, Embedder, create_embedder
from vaxrag.retrieval.fusion import reciprocal_rank_fusion
from vaxrag.retrieval.rerank import CrossEncoderReranker, Reranker
from vaxrag.schemas import Chunk, Question, RetrievalRecord, RetrievedChunk


def build_query(question: Question, mode: str) -> str:
    """Build the retrieval query: the stem, optionally followed by all options."""
    if mode == "stem_only":
        return question.stem
    options = "\n".join(f"{letter}. {text}" for letter, text in question.options.items())
    return f"{question.stem}\n{options}"


class HybridRetriever:
    """Configurable hybrid retriever over a fixed list of chunks.

    Embedder and reranker can be injected (e.g. fakes in tests); otherwise they
    are created from the configuration only when the corresponding stage is on.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        cfg: RetrievalConfig,
        index_dir: str | Path | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        if not (cfg.use_bm25 or cfg.use_dense):
            raise ValueError("At least one of BM25 or dense retrieval must be enabled")
        self.chunks = chunks
        self.cfg = cfg
        docs = [c.retrieval_text for c in chunks]
        self.bm25 = BM25Index(docs) if cfg.use_bm25 else None
        self.dense = None
        if cfg.use_dense:
            self.dense = DenseIndex(embedder or create_embedder(cfg.embedder), docs, index_dir)
        self.reranker = None
        if cfg.use_reranker:
            self.reranker = reranker or CrossEncoderReranker(cfg.reranker)

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """Return all fused candidates, reranked if enabled, best first.

        All candidates are returned (not only `top_k`) so that Recall@k can be
        computed for several k without re-running retrieval. Experiment arms use
        the first `top_k` entries.
        """
        rankings: list[list[int]] = []
        stage: dict[int, dict[str, float]] = {}
        if self.bm25 is not None:
            hits = self.bm25.search(query, self.cfg.bm25_candidates)
            rankings.append([i for i, _ in hits])
            for rank, (i, s) in enumerate(hits, start=1):
                stage.setdefault(i, {}).update({"bm25": s, "bm25_rank": rank})
        if self.dense is not None:
            hits = self.dense.search(query, self.cfg.dense_candidates)
            rankings.append([i for i, _ in hits])
            for rank, (i, s) in enumerate(hits, start=1):
                stage.setdefault(i, {}).update({"dense": s, "dense_rank": rank})
        fused = reciprocal_rank_fusion(rankings, k=self.cfg.rrf_k)
        for i, s in fused:
            stage[i]["rrf"] = s
        ordered = [(i, s) for i, s in fused]
        if self.reranker is not None:
            texts = [getattr(self.chunks[i], self.cfg.rerank_field) for i, _ in fused]
            rerank_scores = self.reranker.score(query, texts)
            for (i, _), s in zip(fused, rerank_scores, strict=True):
                stage[i]["rerank"] = s
            ordered = sorted(
                zip([i for i, _ in fused], rerank_scores, strict=True), key=lambda x: (-x[1], x[0])
            )
        return [
            RetrievedChunk(
                chunk_id=self.chunks[i].chunk_id, rank=rank, score=s, stage_scores=stage[i]
            )
            for rank, (i, s) in enumerate(ordered, start=1)
        ]


def knowledge_base_hash(chunks: list[Chunk]) -> str:
    """Short hash identifying the exact chunk set (IDs and texts)."""
    return stable_hash([(c.chunk_id, c.text, c.retrieval_text) for c in chunks], length=10)


def retrieval_path(retrieval_dir: str | Path, cfg: RetrievalConfig, chunks: list[Chunk]) -> Path:
    """Path of the cached retrieval results for a configuration and knowledge base."""
    return (
        Path(retrieval_dir) / f"retrieval_{cfg.config_hash()}_kb{knowledge_base_hash(chunks)}.jsonl"
    )


def retrieve_all(
    questions: list[Question],
    retriever: HybridRetriever,
    output_path: str | Path,
) -> list[RetrievalRecord]:
    """Retrieve for every question, appending to `output_path` and skipping cached ones."""
    output_path = Path(output_path)
    existing = {r.question_id: r for r in read_models(output_path, RetrievalRecord)}
    config_hash = retriever.cfg.config_hash()
    for q in tqdm([q for q in questions if q.question_id not in existing], desc="retrieval"):
        query = build_query(q, retriever.cfg.query_mode)
        record = RetrievalRecord(
            question_id=q.question_id,
            retrieval_config_hash=config_hash,
            query=query,
            results=retriever.retrieve(query),
        )
        append_jsonl(output_path, record)
        existing[q.question_id] = record
    return [existing[q.question_id] for q in questions if q.question_id in existing]
