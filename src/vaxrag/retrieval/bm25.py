"""Sparse lexical retrieval with BM25 (Okapi)."""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
STOPWORDS = frozenset(
    """a an and are as at be been by can could do does for from had has have how if in into is it
    its may might must not of on or should such than that the their them then there these they this
    those to was were what when where which who whom why will with would you your""".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase, split into alphanumeric tokens (keeping hyphenated terms) and drop stopwords."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    """BM25 index over a list of documents."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self._bm25 = BM25Okapi([tokenize(d) or ["<empty>"] for d in documents], k1=k1, b=b)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return the top-`k` (document index, score) pairs, best first."""
        scores = np.asarray(self._bm25.get_scores(tokenize(query)))
        order = np.argsort(-scores, kind="stable")[:k]
        return [(int(i), float(scores[i])) for i in order]
