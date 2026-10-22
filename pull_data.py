"""Kéo dữ liệu về project trong một lệnh.

Bước 1: tải caption dataset từ Hugging Face -> data/images/scene/ + data/captions.csv
Bước 2: quét data/images -> data/metadata.csv (gồm cả face/fullbody nếu bạn đã thêm ảnh)

Chạy:
    python pull_data.py                 # 300 ảnh scene từ yerevann/coco-karpathy (split test)
    python pull_data.py --n 500 --split train
    python pull_data.py --skip-download  # chỉ quét lại ảnh đang có để sinh metadata
"""
import argparse
import csv
import sys
from collections import Counter

from src.build_dataset import CAPTIONS_CSV, build_from_hf
from src.config import IMAGE_DIR, METADATA_CSV
from src.prepare_metadata import build_metadata


def _check_datasets_installed() -> None:
    try:
        import datasets  # noqa: F401
    except ImportError:
        sys.exit("Thiếu thư viện 'datasets'. Chạy: pip install datasets huggingface_hub")


def _summary() -> None:
    with open(METADATA_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    counts = Counter(r["category"] for r in rows)
    print("\n=== Tóm tắt dữ liệu ===")
    for cat in ("scene", "face", "fullbody"):
        print(f"  {cat:<9}: {counts.get(cat, 0)} ảnh")
    persons = {r["person_id"] for r in rows if r["person_id"]}
    print(f"  số người (face/fullbody): {len(persons)}")
    if counts.get("face", 0) == 0 or counts.get("fullbody", 0) == 0:
        print("  Gợi ý: thêm ảnh vào data/images/face/<person_id>/ và data/images/fullbody/<person_id>/")


def main() -> None:
    ap = argparse.ArgumentParser(description="Kéo dữ liệu về project")
    ap.add_argument("--dataset", default="yerevann/coco-karpathy")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=300, help="số ảnh scene cần lấy")
    ap.add_argument("--skip-download", action="store_true", help="bỏ qua bước tải từ Hugging Face")
    args = ap.parse_args()

    if not args.skip_download:
        _check_datasets_installed()
        print(f"[1/2] Tải {args.n} ảnh từ {args.dataset} (split={args.split})...")
        build_from_hf(args.dataset, args.split, args.n)
    else:
        print("[1/2] Bỏ qua bước tải.")

    print("[2/2] Sinh metadata.csv...")
    build_metadata(IMAGE_DIR, CAPTIONS_CSV, METADATA_CSV)
    _summary()


if __name__ == "__main__":
    main()
