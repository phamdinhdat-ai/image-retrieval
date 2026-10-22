"""Kiểm thử offline: không cần tải mô hình hay dataset.

- Baseline chạy thật trên ảnh tổng hợp (OpenCV, sklearn).
- RAG dùng bộ mã hoá giả (fake) để kiểm tra luồng build / search / save / load.
- Các độ đo đánh giá.

Chạy: python -m unittest discover -s tests -t . -v
"""
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.baseline.retriever import BaselineRetriever
from src.common.data import Record
from src.common.index import VectorIndex
from src.eval.metrics import average_precision_at_k, precision_at_k, recall_at_k
from src.rag.retriever import MultimodalRetriever


def _make_images(folder: Path) -> list[Record]:
    """6 ảnh tổng hợp: màu nền khác nhau, có một hình chữ nhật trắng."""
    rng = np.random.default_rng(0)
    specs = [("sc_00001", "beach sunset sea", "scene"),
             ("sc_00002", "city street traffic", "scene"),
             ("sc_00003", "man riding bicycle outdoors", "scene"),
             ("sc_00004", "forest green trees", "scene"),
             ("fa_00001", "face portrait", "face"),
             ("fb_00001", "person standing", "fullbody")]
    records = []
    for i, (iid, caption, cat) in enumerate(specs):
        img = np.full((240, 320, 3), (i * 40 % 255, 100, 200 - i * 30), dtype=np.uint8)
        img = np.clip(img + rng.integers(0, 10, img.shape, dtype=np.uint8), 0, 255).astype(np.uint8)
        cv2.rectangle(img, (100, 60), (200, 200), (255, 255, 255), -1)
        path = folder / f"{iid}.jpg"
        cv2.imwrite(str(path), img)
        records.append(Record(image_id=iid, path=str(path), category=cat, caption=caption))
    return records


@dataclass
class FakeCLIP:
    """Vector đặc trưng đơn giản: màu trung bình (ảnh) và băm từ (văn bản)."""

    def encode_images(self, images):
        out = []
        for im in images:
            mean = np.asarray(im, dtype=np.float32).reshape(-1, 3).mean(axis=0) / 255.0
            out.append(np.concatenate([mean, [1.0]]))
        return np.asarray(out, dtype=np.float32)

    def encode_texts(self, texts):
        # Cùng số chiều (4) với ảnh; giá trị không có ngữ nghĩa thật, chỉ để kiểm tra luồng
        out = np.zeros((len(texts), 4), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, sum(map(ord, w)) % 4] += 1.0
        return out


@dataclass
class FakeFace:
    def encode(self, bgr):
        return np.ones(4, dtype=np.float32)  # luôn "thấy mặt"


@dataclass
class FakePerson:
    def crop(self, img):
        return img


class BaselineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.records = _make_images(Path(cls.tmp.name))
        cls.retriever = BaselineRetriever()
        cls.retriever.build(cls.records)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_index_sizes(self):
        self.assertEqual(len(self.retriever.indexes["similar"].ids), len(self.records))

    def test_text_search_finds_caption_match(self):
        hits = self.retriever.search("text", "man riding bicycle", 3)
        self.assertEqual(hits[0].image_id, "sc_00003")

    def test_similar_search_returns_self_first(self):
        hits = self.retriever.search("similar", self.records[0].path, 3)
        self.assertEqual(hits[0].image_id, "sc_00001")

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            self.retriever.save(Path(d))
            loaded = BaselineRetriever().load(Path(d))
            hits = loaded.search("text", "beach sea", 1)
            self.assertEqual(hits[0].image_id, "sc_00001")


class RAGWiringTest(unittest.TestCase):
    """Kiểm tra luồng RAG với bộ mã hoá giả, không cần torch hay tải mô hình."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.records = _make_images(Path(cls.tmp.name))
        from src.rag.embeddings import Encoders
        enc = Encoders(clip=FakeCLIP(), face=FakeFace(), person=FakePerson())
        cls.retriever = MultimodalRetriever(encoders=enc)
        cls.retriever.build(cls.records)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_all_indexes_built(self):
        self.assertEqual(len(self.retriever.indexes["face"].ids), len(self.records))  # FakeFace luôn thấy mặt
        self.assertEqual(len(self.retriever.indexes["scene"].ids), len(self.records))
        self.assertEqual(len(self.retriever.indexes["fullbody"].ids), len(self.records))

    def test_text_search_returns_k_valid_hits(self):
        hits = self.retriever.search("text", "beach sea", 3)
        self.assertEqual(len(hits), 3)
        self.assertTrue(all(h.image_id in {r.image_id for r in self.records} for h in hits))

    def test_similar_search_returns_self_first(self):
        hits = self.retriever.search("similar", self.records[0].path, 3)
        self.assertEqual(hits[0].image_id, "sc_00001")

    def test_face_query_without_face_returns_empty(self):
        class NoFace(FakeFace):
            def encode(self, bgr):
                return None
        self.retriever.enc.face = NoFace()
        self.assertEqual(self.retriever.search("face", self.records[4].path, 3), [])
        self.retriever.enc.face = FakeFace()  # khôi phục cho các test khác

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            self.retriever.save(Path(d))
            loaded = MultimodalRetriever(encoders=self.retriever.enc).load(Path(d))
            self.assertEqual(set(loaded.indexes), {"scene", "face", "fullbody"})
            self.assertEqual(len(loaded.search("text", "bicycle", 2)), 2)


class IndexAndMetricsTest(unittest.TestCase):
    def test_empty_index_search(self):
        self.assertEqual(VectorIndex().search(np.ones(3), 5), [])

    def test_cosine_ranking(self):
        idx = VectorIndex()
        idx.add(["a", "b"], np.array([[1, 0], [0, 1]], dtype=np.float32))
        self.assertEqual(idx.search(np.array([0.9, 0.1]), 1)[0][0], "a")

    def test_metrics(self):
        ranked = ["a", "b", "c", "d", "e"]
        relevant = {"a", "c"}
        self.assertAlmostEqual(precision_at_k(ranked, relevant, 5), 0.4)
        self.assertAlmostEqual(recall_at_k(ranked, relevant, 5), 1.0)
        # AP@5 = (1/1 + 2/3) / min(2, 5)
        self.assertAlmostEqual(average_precision_at_k(ranked, relevant, 5), (1 + 2 / 3) / 2)


if __name__ == "__main__":
    unittest.main()
