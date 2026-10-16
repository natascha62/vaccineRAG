"""Reciprocal Rank Fusion (Cormack et al., 2009)."""

from __future__ import annotations


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Fuse several ranked lists of document indices.

    Each document receives sum(1 / (k + rank)) over the lists it appears in
    (ranks are 1-based). Ties are broken by the first list in which the
    document appears, then by index, so the output is deterministic.
    """
    scores: dict[int, float] = {}
    first_seen: dict[int, tuple[int, int]] = {}
    for list_idx, ranking in enumerate(rankings):
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(doc, (list_idx, rank))
    return sorted(scores.items(), key=lambda item: (-item[1], first_seen[item[0]], item[0]))
