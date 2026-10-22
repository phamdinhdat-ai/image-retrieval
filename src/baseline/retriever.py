"""Baseline retriever.

- text    : TF-IDF trên caption (chỉ so khớp từ khoá, không hiểu ngữ nghĩa)
- face / fullbody / similar : đặc trưng thủ công + cosine (xem features.py)
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.baseline.features import extract
from src.common.base import BaseRetriever
from src.common.data import Hit, Record, read_image_bgr


class BaselineRetriever(BaseRetriever):
    index_names = ("face", "fullbody", "similar")

    def __init__(self):
        super().__init__()
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.text_matrix = None
        self.order: list[str] = []

    def build(self, records: list[Record]) -> None:
        from src.common.index import VectorIndex
        if not records:
            raise ValueError("No image records. Prepare metadata first.")
        if len({r.image_id for r in records}) != len(records):
            raise ValueError("Duplicate image_id in metadata.")
        self.indexes = {name: VectorIndex() for name in self.index_names}
        self.records = {r.image_id: r for r in records}
        self.order = [r.image_id for r in records]
        docs = [f"{r.caption} {r.category}" for r in records]
        self.text_matrix = self.tfidf.fit_transform(docs)

        buckets = {name: ([], []) for name in self.index_names}
        for r in records:
            img = read_image_bgr(r.path)
            if img is None:
                continue
            for name in self.index_names:
                feat = extract(name, img)
                if feat is None:  # ví dụ: không thấy khuôn mặt
                    continue
                buckets[name][0].append(r.image_id)
                buckets[name][1].append(feat)
        for name, (ids, feats) in buckets.items():
            if ids:
                self.indexes[name].add(ids, np.stack(feats))

    def _search_text(self, query: str, k: int):
        q = self.tfidf.transform([query])
        sims = (self.text_matrix @ q.T).toarray().ravel()
        top = np.argsort(-sims)[:k]
        return [(self.order[i], float(sims[i])) for i in top if sims[i] > 0]

    def search(self, mode: str, query: str, k: int = 5) -> list[Hit]:
        if mode not in (*self.index_names, "text"):
            raise ValueError(f"Unsupported search mode: {mode}")
        if k <= 0 or not query.strip():
            return []
        if mode == "text":
            return self._hits(self._search_text(query, k))
        img = read_image_bgr(query)
        if img is None:
            return []
        feat = extract(mode, img)
        if feat is None:  # ảnh truy vấn không có khuôn mặt
            return []
        return self._hits(self.indexes[mode].search(feat, k))

    def save(self, directory: Path) -> None:
        super().save(directory)
        with open(directory / "tfidf.pkl", "wb") as f:
            pickle.dump((self.tfidf, self.text_matrix, self.order), f)

    def load(self, directory: Path) -> "BaselineRetriever":
        super().load(directory)
        with open(directory / "tfidf.pkl", "rb") as f:
            self.tfidf, self.text_matrix, self.order = pickle.load(f)
        return self


def summarize(hits: list[Hit]) -> str:
    """Chatbot của baseline: không có LLM nên chỉ liệt kê ảnh khớp nhất."""
    if not hits:
        return "Không tìm thấy ảnh nào khớp với từ khoá trong caption."
    lines = [f"- [{h.image_id}] {h.caption or '(không có caption)'} (điểm {h.score:.3f})" for h in hits]
    return ("Baseline không có mô hình ngôn ngữ, nên chỉ liệt kê các ảnh khớp nhất theo từ khoá:\n"
            + "\n".join(lines))
