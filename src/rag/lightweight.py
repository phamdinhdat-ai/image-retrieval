"""Small caption encoder + lexical fusion; classical image retrieval stays local."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.baseline.retriever import BaselineRetriever
from src.common.data import Hit, Record
from src.common.index import VectorIndex
from src.config import DENSE_MIN_SCORE, EMBEDDING_MODEL


class CaptionEncoder:
    def __init__(self, model_id=EMBEDDING_MODEL):
        import os
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(max(1, int(os.getenv("EMBEDDING_THREADS", "2"))))
        self.model_id = model_id
        self.model = SentenceTransformer(model_id, device="cpu")

    def encode(self, texts):
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


def fingerprint(records):
    rows = [(r.image_id, r.path, r.caption, r.category) for r in records]
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode()).hexdigest()


class LightweightRetriever:
    def __init__(self, encoder=None, baseline=None):
        self.encoder = encoder
        self.baseline = baseline or BaselineRetriever()
        self.dense = VectorIndex()

    @property
    def records(self):
        return self.baseline.records

    def _encoder(self):
        if self.encoder is None:
            self.encoder = CaptionEncoder()
        return self.encoder

    def build(self, records: list[Record]):
        self.baseline.build(records)
        captioned = [r for r in records if r.caption.strip()]
        self.dense = VectorIndex()
        if captioned:
            self.dense.add([r.image_id for r in captioned], self._encoder().encode([r.caption for r in captioned]))

    def search(self, mode, query, k=5) -> list[Hit]:
        if k <= 0 or not query.strip():
            return []
        if mode != "text":
            return self.baseline.search(mode, query, k)
        lexical = self.baseline.search("text", query, max(k * 4, 20))
        semantic = self.dense.search(self._encoder().encode([query])[0], max(k * 4, 20)) if self.dense.ids else []
        semantic = [(i, s) for i, s in semantic if s >= DENSE_MIN_SCORE]
        scores = {}
        for ranking in [[h.image_id for h in lexical], [i for i, _ in semantic]]:
            for rank, iid in enumerate(ranking, 1):
                scores[iid] = scores.get(iid, 0) + 1 / (60 + rank)
        return self.baseline._hits(sorted(scores.items(), key=lambda p: (-p[1], p[0]))[:k])

    def save(self, directory: Path):
        self.baseline.save(directory)
        self.dense.save(directory / "captions.npz")
        manifest = {"model": getattr(self._encoder(), "model_id", EMBEDDING_MODEL),
                    "fingerprint": fingerprint(list(self.records.values())), "version": 1}
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def load(self, directory: Path):
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest["model"] != getattr(self.encoder, "model_id", EMBEDDING_MODEL):
            raise ValueError("Embedding model changed. Rebuild the lightweight index.")
        self.baseline.load(directory)
        self.dense = VectorIndex.load(directory / "captions.npz")
        if manifest["fingerprint"] != fingerprint(list(self.records.values())):
            raise ValueError("Index metadata mismatch. Rebuild index.")
        return self
