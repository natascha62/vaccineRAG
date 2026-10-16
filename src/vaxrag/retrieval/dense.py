"""Dense retrieval: bge-m3 (primary) and MedCPT (sensitivity analysis).

Document embeddings are cached on disk, keyed by the embedder and a hash of the
indexed texts, so re-running retrieval does not re-encode the knowledge base.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from vaxrag.config import EmbedderConfig
from vaxrag.io_utils import stable_hash


def resolve_device(device: str) -> str:
    """Map 'auto' to the best available torch device (cuda > mps > cpu)."""
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Embedder(Protocol):
    """Anything that can embed queries and documents into the same vector space."""

    name: str

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        """Embed documents; returns an (n, d) float32 array."""

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        """Embed queries; returns an (n, d) float32 array."""


class BGEM3Embedder:
    """bge-m3 dense embeddings via sentence-transformers (cosine similarity)."""

    def __init__(self, cfg: EmbedderConfig) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = f"bge-m3:{cfg.model_id}"
        self.cfg = cfg
        self.model = SentenceTransformer(cfg.model_id, device=resolve_device(cfg.device))
        self.model.max_seq_length = cfg.max_length

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=self.cfg.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        """Embed documents (no instruction prefix is needed for bge-m3)."""
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        """Embed queries."""
        return self._encode(texts)


class MedCPTEmbedder:
    """MedCPT asymmetric encoders (query / article); scored by inner product."""

    def __init__(self, cfg: EmbedderConfig) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.name = f"medcpt:{cfg.model_id}"
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        query_id = cfg.query_model_id or "ncbi/MedCPT-Query-Encoder"
        self.doc_tok = AutoTokenizer.from_pretrained(cfg.model_id)
        self.doc_model = AutoModel.from_pretrained(cfg.model_id).to(self.device).eval()
        self.query_tok = AutoTokenizer.from_pretrained(query_id)
        self.query_model = AutoModel.from_pretrained(query_id).to(self.device).eval()

    def _encode(self, texts: list[str], tokenizer, model, max_length: int) -> np.ndarray:
        import torch

        out: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), self.cfg.batch_size):
                batch = tokenizer(
                    texts[start : start + self.cfg.batch_size],
                    truncation=True,
                    padding=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(self.device)
                cls = model(**batch).last_hidden_state[:, 0, :]
                out.append(cls.float().cpu().numpy())
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, 768), np.float32)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        """Embed documents with the article encoder (max 512 tokens)."""
        return self._encode(texts, self.doc_tok, self.doc_model, min(self.cfg.max_length, 512))

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        """Embed queries with the query encoder (max 64 tokens, as in MedCPT)."""
        return self._encode(texts, self.query_tok, self.query_model, 64)


def create_embedder(cfg: EmbedderConfig) -> Embedder:
    """Instantiate the configured embedder."""
    if cfg.name == "bge-m3":
        return BGEM3Embedder(cfg)
    if cfg.name == "medcpt":
        return MedCPTEmbedder(cfg)
    raise ValueError(f"Unknown embedder: {cfg.name}")


class DenseIndex:
    """Exhaustive inner-product search over cached document embeddings.

    The knowledge base has at most tens of thousands of chunks, so a dense
    matrix product is fast and avoids approximate-search nondeterminism.
    """

    def __init__(
        self, embedder: Embedder, documents: list[str], cache_dir: str | Path | None
    ) -> None:
        self.embedder = embedder
        cache_path = None
        if cache_dir is not None:
            key = stable_hash({"embedder": embedder.name, "docs": documents}, length=16)
            safe_name = embedder.name.replace("/", "_").replace(":", "_")
            cache_path = Path(cache_dir) / f"dense_{safe_name}_{key}.npy"
        if cache_path is not None and cache_path.exists():
            self.matrix = np.load(cache_path)
        else:
            self.matrix = embedder.encode_documents(documents)
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, self.matrix)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return the top-`k` (document index, score) pairs, best first."""
        q = self.embedder.encode_queries([query])[0]
        scores = self.matrix @ q
        order = np.argsort(-scores, kind="stable")[:k]
        return [(int(i), float(scores[i])) for i in order]
