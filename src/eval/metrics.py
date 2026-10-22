"""Các độ đo đánh giá truy hồi ở ngưỡng K."""
from __future__ import annotations


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return sum(1 for x in ranked[:k] if x in relevant) / k


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for x in ranked[:k] if x in relevant) / len(relevant)


def average_precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """AP@K. Mean over queries = MAP@K."""
    if not relevant:
        return 0.0
    hits, score = 0, 0.0
    for i, x in enumerate(ranked[:k], start=1):
        if x in relevant:
            hits += 1
            score += hits / i
    return score / min(len(relevant), k)
