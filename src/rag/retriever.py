"""Multimodal RAG retriever.

Index:
- scene    : CLIP embedding của toàn ảnh      -> dùng cho truy vấn văn bản và ảnh tương tự
- face     : ArcFace embedding khuôn mặt lớn nhất (bỏ qua ảnh không có mặt)
- fullbody : CLIP embedding của vùng người (DETR cắt người)
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from src.common.base import BaseRetriever
from src.common.data import Hit, Record
from src.rag.embeddings import Encoders


def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    return np.asarray(img)[:, :, ::-1].copy()


class MultimodalRetriever(BaseRetriever):
    index_names = ("scene", "face", "fullbody")

    def __init__(self, encoders: Encoders | None = None):
        super().__init__()
        self.enc = encoders if encoders is not None else Encoders.default()

    def _embed_record(self, img: Image.Image) -> dict[str, np.ndarray]:
        out = {
            "scene": self.enc.clip.encode_images([img])[0],
            "fullbody": self.enc.clip.encode_images([self.enc.person.crop(img)])[0],
        }
        face_vec = self.enc.face.encode(_pil_to_bgr(img))
        if face_vec is not None:
            out["face"] = face_vec
        return out

    def build(self, records: list[Record]) -> None:
        self.records = {r.image_id: r for r in records}
        buckets = {name: ([], []) for name in self.index_names}
        for r in records:
            img = Image.open(r.path).convert("RGB")
            for name, vec in self._embed_record(img).items():
                buckets[name][0].append(r.image_id)
                buckets[name][1].append(vec)
        for name, (ids, vecs) in buckets.items():
            if ids:
                self.indexes[name].add(ids, np.stack(vecs))

    def search(self, mode: str, query: str, k: int = 5) -> list[Hit]:
        if mode == "text":
            vec, name = self.enc.clip.encode_texts([query])[0], "scene"
        else:
            img = Image.open(query).convert("RGB")
            if mode == "face":
                vec, name = self.enc.face.encode(_pil_to_bgr(img)), "face"
                if vec is None:
                    return []
            elif mode == "fullbody":
                vec, name = self.enc.clip.encode_images([self.enc.person.crop(img)])[0], "fullbody"
            elif mode == "similar":
                vec, name = self.enc.clip.encode_images([img])[0], "scene"
            else:
                raise ValueError(f"Chế độ không hỗ trợ: {mode}")
        return self._hits(self.indexes[name].search(vec, k))
