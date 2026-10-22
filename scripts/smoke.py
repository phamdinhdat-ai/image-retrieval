"""Reproducible local + optional live API/vision smoke test, no secret output."""
import argparse
import json
import time

from src.common.data import load_metadata
from src.config import INDEX_DIR, RESULTS_DIR
from src.rag.chat import APIClient, ChatService
from src.rag.lightweight import LightweightRetriever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", action="store_true")
    parser.add_argument("--vision", action="store_true")
    args = parser.parse_args()
    retriever = LightweightRetriever().load(INDEX_DIR / "lightweight")
    service = ChatService(retriever)
    questions = ["people riding bicycles", "a giraffe eating leaves", "quantum computer superconducting chip"]
    rows = []
    for question in questions:
        start = time.perf_counter()
        result = service.ask(question, use_ai=False)
        rows.append({"question": question, "status": result.status, "ids": [h.image_id for h in result.hits],
                     "answer": result.answer, "latency_ms": round((time.perf_counter() - start) * 1000, 2)})
    if args.api:
        first = service.ask("Tìm ảnh có người đi xe đạp", use_ai=True)
        second = service.ask("Trong những ảnh đó có tàu không?", history=[
            {"role": "user", "content": "Tìm ảnh có người đi xe đạp"},
            {"role": "assistant", "content": first.answer, "hits": [vars(h) for h in first.hits]}], use_ai=True)
        for result in [first, second]:
            rows.append({"query": result.retrieval_query, "status": result.status,
                         "ids": [h.image_id for h in result.hits], "answer": result.answer})
        if args.vision:
            result = service.ask("Người trong ảnh đang làm gì?", image_path=load_metadata()[2].path,
                                 use_ai=True, vision=True, k=2)
            rows.append({"vision": True, "status": result.status, "answer": result.answer,
                         "ids": [h.image_id for h in result.hits]})
    report = {"model": APIClient().model if args.api else None, "records": len(retriever.records),
              "captioned": sum(bool(r.caption) for r in retriever.records.values()), "cases": rows}
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "smoke.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
