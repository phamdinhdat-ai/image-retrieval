"""So sánh baseline và Multimodal RAG trên cùng một bộ truy vấn.

Chạy:
    python -m src.eval.evaluate --k 5
    python -m src.eval.evaluate --systems baseline --k 10
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.common.data import load_eval_queries
from src.config import EVAL_QUERIES, INDEX_DIR, RESULTS_DIR, ROOT
from src.eval.metrics import average_precision_at_k, precision_at_k, recall_at_k

METRICS = ["P@K", "R@K", "AP@K", "latency_ms"]


def load_system(name: str):
    if name == "baseline":
        from src.baseline.retriever import BaselineRetriever
        return BaselineRetriever().load(INDEX_DIR / "baseline")
    if name == "lightweight":
        from src.rag.lightweight import LightweightRetriever
        return LightweightRetriever().load(INDEX_DIR / "lightweight")
    from src.rag.retriever import MultimodalRetriever
    return MultimodalRetriever().load(INDEX_DIR / "rag")


def run(system_names: list[str], queries: list[dict], k: int) -> pd.DataFrame:
    rows = []
    for name in system_names:
        retriever = load_system(name)
        # Report warm query latency separately from model download/load.
        if name == "lightweight":
            retriever._encoder().encode(["warmup"])
        for q in queries:
            query = q["query"] if q["mode"] == "text" else str(ROOT / q["query_image"])
            t0 = time.perf_counter()
            hits = retriever.search(q["mode"], query, k)
            latency = (time.perf_counter() - t0) * 1000

            ranked = [h.image_id for h in hits]
            relevant = set(q["relevant_ids"])
            rows.append({
                "system": name, "query_id": q["id"], "mode": q["mode"],
                "P@K": precision_at_k(ranked, relevant, k),
                "R@K": recall_at_k(ranked, relevant, k),
                "AP@K": average_precision_at_k(ranked, relevant, k),
                "latency_ms": latency,
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=Path, default=EVAL_QUERIES)
    ap.add_argument("--systems", nargs="+", default=["baseline", "rag"], choices=["baseline", "lightweight", "rag"])
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    df = run(args.systems, load_eval_queries(args.queries), args.k)

    by_mode = df.groupby(["system", "mode"])[METRICS].mean().round(3)
    overall = df.groupby("system")[METRICS].mean().round(3)
    print(f"\n=== Trung bình theo kiểu truy vấn (K={args.k}) ===")
    print(by_mode.to_string())
    print(f"\n=== Trung bình toàn bộ ===")
    print(overall.to_string())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / f"per_query_k{args.k}.csv", index=False)
    by_mode.to_csv(RESULTS_DIR / f"summary_k{args.k}.csv")
    print(f"\nĐã lưu kết quả vào {RESULTS_DIR}")


if __name__ == "__main__":
    main()
