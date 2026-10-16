"""Cross-encoder reranking (bge-reranker-v2-m3)."""

from __future__ import annotations

from typing import Protocol

from vaxrag.config import RerankerConfig
from vaxrag.retrieval.dense import resolve_device


class Reranker(Protocol):
    """Scores (query, document) pairs; higher means more relevant."""

    def score(self, query: str, documents: list[str]) -> list[float]:
        """Return one relevance score per document."""


class CrossEncoderReranker:
    """sentence-transformers CrossEncoder wrapper."""

    def __init__(self, cfg: RerankerConfig) -> None:
        from sentence_transformers import CrossEncoder

        self.cfg = cfg
        self.model = CrossEncoder(
            cfg.model_id, max_length=cfg.max_length, device=resolve_device(cfg.device)
        )

    def score(self, query: str, documents: list[str]) -> list[float]:
        """Score each document against the query."""
        if not documents:
            return []
        scores = self.model.predict(
            [(query, d) for d in documents], batch_size=self.cfg.batch_size, show_progress_bar=False
        )
        return [float(s) for s in scores]
