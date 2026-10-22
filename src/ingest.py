"""Bước offline: trích đặc trưng và lưu index cho một hệ thống.

Chạy:
    python -m src.ingest --system baseline
    python -m src.ingest --system rag
"""
import argparse

from src.common.data import load_metadata
from src.config import INDEX_DIR, METADATA_CSV


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["baseline", "lightweight", "rag"], required=True)
    ap.add_argument("--metadata", default=str(METADATA_CSV))
    args = ap.parse_args()

    # import muộn để baseline chạy được mà không cần cài torch
    if args.system == "baseline":
        from src.baseline.retriever import BaselineRetriever as Retriever
    elif args.system == "lightweight":
        from src.rag.lightweight import LightweightRetriever as Retriever
    else:
        from src.rag.retriever import MultimodalRetriever as Retriever

    records = load_metadata(args.metadata)
    print(f"[{args.system}] Đang xử lý {len(records)} ảnh...")
    retriever = Retriever()
    retriever.build(records)
    retriever.save(INDEX_DIR / args.system)
    sizes = {"records": len(retriever.records)}
    print(f"[{args.system}] Xong. Số vector theo index: {sizes}")
    print(f"[{args.system}] Lưu tại {INDEX_DIR / args.system}")


if __name__ == "__main__":
    main()
