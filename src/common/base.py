"""Lớp cơ sở cho mọi retriever: build, search, save/load.

Hai hệ thống (baseline và RAG) cùng giao diện nên app và evaluate dùng chung được.
"""
from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

from src.common.data import Hit, Record
from src.common.index import VectorIndex


class BaseRetriever(ABC):
    index_names: tuple[str, ...] = ()

    def __init__(self):
        self.records: dict[str, Record] = {}
        self.indexes: dict[str, VectorIndex] = {n: VectorIndex() for n in self.index_names}

    @abstractmethod
    def build(self, records: list[Record]) -> None:
        """Trích đặc trưng cho toàn bộ bộ ảnh (bước offline)."""

    @abstractmethod
    def search(self, mode: str, query: str, k: int = 5) -> list[Hit]:
        """mode='text': query là câu truy vấn.
        mode khác ('face', 'fullbody', 'similar'): query là đường dẫn tới ảnh truy vấn."""

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "records.pkl", "wb") as f:
            pickle.dump(self.records, f)
        for name, idx in self.indexes.items():
            idx.save(directory / f"{name}.npz")

    def load(self, directory: Path) -> "BaseRetriever":
        with open(directory / "records.pkl", "rb") as f:
            self.records = pickle.load(f)
        for name in self.index_names:
            self.indexes[name] = VectorIndex.load(directory / f"{name}.npz")
        return self

    def _hits(self, pairs: list[tuple[str, float]]) -> list[Hit]:
        out = []
        for image_id, score in pairs:
            r = self.records[image_id]
            out.append(Hit(image_id, score, r.path, r.caption))
        return out
