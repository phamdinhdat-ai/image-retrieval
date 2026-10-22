import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.common.data import Hit
from src.rag.chat import AIError, APIClient, ChatService
from src.rag.lightweight import LightweightRetriever
from tests.test_offline import _make_images


class FakeEncoder:
    model_id = "sentence-transformers/all-MiniLM-L6-v2"

    def encode(self, texts):
        return np.array([[1, 0] if "beach" in t else [0, 1] for t in texts], dtype=np.float32)


class FakeRetriever:
    def __init__(self, hits=None):
        self.hits = [Hit("sc_1", .5, "missing.jpg", "A cyclist on a road")] if hits is None else hits
        self.queries = []

    def search(self, mode, query, k):
        self.queries.append((mode, query, k))
        return self.hits[:k]


class FakeClient:
    def __init__(self, response=None, fail=False):
        self.response = response or {"answer": "Có người đi xe đạp [sc_1].", "citations": ["sc_1"]}
        self.fail = fail
        self.messages = []

    def complete(self, messages, max_tokens=700):
        self.messages.append(messages)
        if self.fail:
            raise AIError("API AI trả HTTP 403.")
        return json.dumps({"search_query": "cyclist road", "scope_previous": False}) if max_tokens == 180 else json.dumps(self.response)


class ChatTests(unittest.TestCase):
    def test_rewrite_history_and_citations(self):
        retriever, client = FakeRetriever(), FakeClient()
        result = ChatService(retriever, client).ask("Còn ảnh nào?", [{"role": "user", "content": "xe đạp"}])
        self.assertEqual(retriever.queries[0][1], "cyclist road")
        self.assertIn("xe đạp", client.messages[0][1]["content"])
        self.assertEqual(result.status, "AI + caption")

    def test_unknown_citation_rejected(self):
        client = FakeClient({"answer": "Ảnh [invented]", "citations": ["invented"]})
        result = ChatService(FakeRetriever(), client).ask("bike")
        self.assertNotIn("invented", result.answer)
        self.assertIn("[sc_1]", result.answer)

    def test_inline_mismatch_rejected(self):
        result = ChatService(FakeRetriever(), FakeClient({"answer": "[invented]", "citations": ["sc_1"]})).ask("bike")
        self.assertNotIn("invented", result.answer)

    def test_api_failure_falls_back(self):
        result = ChatService(FakeRetriever(), FakeClient(fail=True)).ask("bicycle")
        self.assertIn("403", result.status)
        self.assertIn("[sc_1]", result.answer)

    def test_empty_evidence_does_not_generate(self):
        client = FakeClient()
        result = ChatService(FakeRetriever([]), client).ask("spaceship")
        self.assertEqual(len(client.messages), 1)
        self.assertEqual(result.hits, [])

    def test_offline_never_calls_api(self):
        client = FakeClient()
        result = ChatService(FakeRetriever(), client).ask("bike", use_ai=False)
        self.assertEqual(client.messages, [])
        self.assertIn("sc_1", result.answer)

    def test_image_routes_to_visual_retrieval(self):
        retriever = FakeRetriever()
        ChatService(retriever, FakeClient()).ask("similar?", image_path="query.png", mode="fullbody", use_ai=False)
        self.assertEqual(retriever.queries[0][:2], ("fullbody", "query.png"))

    def test_followup_scope_excludes_new_sources(self):
        class ScopedClient(FakeClient):
            def complete(self, messages, max_tokens=700):
                if max_tokens == 180:
                    return json.dumps({"search_query": "bicycle train", "scope_previous": True})
                return super().complete(messages, max_tokens)
        retriever = FakeRetriever([Hit("sc_1", .9, "a.jpg", "bicycle"), Hit("sc_2", .8, "b.jpg", "train")])
        result = ChatService(retriever, ScopedClient()).ask("those?", [{"role": "assistant", "content": "previous", "hits": [{"image_id": "sc_1"}]}])
        self.assertEqual([h.image_id for h in result.hits], ["sc_1"])

    def test_invalid_rewrite_uses_original_query(self):
        class BadPlanner(FakeClient):
            def complete(self, messages, max_tokens=700):
                return "not json" if max_tokens == 180 else super().complete(messages, max_tokens)
        retriever = FakeRetriever()
        ChatService(retriever, BadPlanner()).ask("original bicycle query")
        self.assertEqual(retriever.queries[0][1], "original bicycle query")

    def test_http_error_hides_body_and_credentials(self):
        client = APIClient()
        client.base_url, client.model, client.key = "https://example.test", "model", "secret"
        with patch("src.rag.chat.requests.post") as post:
            post.return_value.ok = False
            post.return_value.status_code = 403
            with self.assertRaisesRegex(AIError, "HTTP 403"):
                client.complete([])


class LightweightTests(unittest.TestCase):
    def test_rebuilding_does_not_duplicate_visual_vectors(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = _make_images(Path(tmp))
            retriever = LightweightRetriever(FakeEncoder())
            retriever.build(records)
            retriever.build(records)
            self.assertEqual(len(retriever.baseline.indexes["similar"].ids), len(records))

    def test_roundtrip_and_model_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = _make_images(root)
            retriever = LightweightRetriever(FakeEncoder())
            retriever.build(records)
            retriever.save(root / "index")
            loaded = LightweightRetriever(FakeEncoder()).load(root / "index")
            self.assertEqual(loaded.search("text", "beach", 1)[0].image_id, "sc_00001")
            other = FakeEncoder()
            other.model_id = "wrong-model"
            with self.assertRaises(ValueError):
                LightweightRetriever(other).load(root / "index")

    def test_empty_query_and_k(self):
        retriever = LightweightRetriever(FakeEncoder())
        self.assertEqual(retriever.search("text", "", 3), [])
        self.assertEqual(retriever.search("text", "beach", 0), [])
