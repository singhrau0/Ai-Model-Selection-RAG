"""Standard information-retrieval evaluation metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence


def dcg(relevances: Sequence[float], k: int | None = None) -> float:
    values = relevances[:k] if k is not None else relevances
    return sum((2.0**rel - 1.0) / math.log2(rank + 2) for rank, rel in enumerate(values))


def ndcg(relevances: Sequence[float], k: int | None = None) -> float:
    """Normalized discounted cumulative gain in [0, 1]."""
    actual = list(relevances[:k] if k is not None else relevances)
    ideal = sorted(actual, reverse=True)
    denominator = dcg(ideal)
    return dcg(actual) / denominator if denominator else 0.0


def ndcg_for_ids(retrieved_ids: Sequence[str], relevance: dict[str, float], k: int = 10) -> float:
    """Convenience nDCG@k from retrieved chunk IDs and graded judgments."""
    gains = [relevance.get(str(doc_id), 0.0) for doc_id in retrieved_ids[:k]]
    ideal = sorted(relevance.values(), reverse=True)[:k]
    denominator = dcg(ideal)
    return dcg(gains) / denominator if denominator else 0.0
