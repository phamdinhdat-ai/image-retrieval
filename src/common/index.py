"""Chỉ mục vector với độ đo cosine (tìm kiếm vét cạn)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


class VectorIndex:
    """Brute-force cosine search. Đủ cho vài chục nghìn vector; khi lớn hơn thì thay bằng FAISS."""

    def __init__(self):
        self.ids: list[str] = []
        self.vectors = np.zeros((0, 0), dtype=np.float32)

    @staticmethod
    def _normalize(x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)

    def add(self, ids: list[str], vectors) -> None:
        v = self._normalize(vectors)
        self.vectors = v if not self.ids else np.vstack([self.vectors, v])
        self.ids.extend(ids)

    def search(self, query, k: int) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        q = self._normalize(np.ravel(query))
        scores = self.vectors @ q
        top = np.argsort(-scores)[:k]
        return [(self.ids[i], float(scores[i])) for i in top]

    def save(self, path: Path) -> None:
        np.savez(path, vectors=self.vectors, ids=np.array(self.ids, dtype=str))

    @classmethod
    def load(cls, path: Path) -> "VectorIndex":
        d = np.load(path, allow_pickle=False)
        idx = cls()
        idx.vectors = d["vectors"]
        idx.ids = [str(x) for x in d["ids"]]
        return idx
