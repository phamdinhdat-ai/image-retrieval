"""Tải một dataset caption nhỏ từ Hugging Face và lưu thành ảnh scene + caption.

Mặc định dùng yerevann/coco-karpathy (COCO captions). Schema của dataset không được
khai báo trong README nên script tự nhận diện cột ảnh và cột caption; nếu không khớp,
script sẽ in ra danh sách cột để bạn chỉnh lại.

Kết quả:
    data/images/scene/sc_XXXXX.jpg     ảnh
    data/captions.csv                  path (tương đối với data/images), caption

Chạy: python -m src.build_dataset --n 300
Sau đó: python -m src.prepare_metadata
"""
from __future__ import annotations

import argparse
import csv
import io
import urllib.request

from PIL import Image

from src.config import IMAGE_DIR

CAPTIONS_CSV = IMAGE_DIR.parent / "captions.csv"
SCENE_DIR = IMAGE_DIR / "scene"


def _download_image(url: str, timeout: int = 20) -> Image.Image | None:
    """Tải ảnh từ URL (có User-Agent vì một số máy chủ chặn UA mặc định)."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 image-retrieval-miniproject"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return Image.open(io.BytesIO(resp.read()))
    except Exception:
        return None


def _has_image_source(row: dict) -> bool:
    if "url" in row:
        return True
    return any(isinstance(v, Image.Image) or (isinstance(v, dict) and v.get("bytes")) for v in row.values())


def _get_image(row: dict) -> Image.Image | None:
    for value in row.values():
        if isinstance(value, Image.Image):  # datasets đã decode sẵn
            return value
        if isinstance(value, dict) and value.get("bytes"):  # dạng {'bytes': ..., 'path': ...}
            return Image.open(io.BytesIO(value["bytes"]))
    url = row.get("url")  # dataset chỉ lưu URL ảnh (ví dụ yerevann/coco-karpathy)
    if isinstance(url, str) and url.startswith(("http://", "https://", "file://")):
        return _download_image(url)
    return None


def _get_caption(row: dict) -> str:
    for key in ("sentences", "captions", "caption", "sentence"):
        value = row.get(key)
        if not value:
            continue
        if isinstance(value, str):
            return value.strip()
        item = value[0]
        if isinstance(item, dict):
            text = item.get("raw") or item.get("text") or next(
                (v for v in item.values() if isinstance(v, str) and v.strip()), "")
            return str(text).strip()
        return str(item).strip()
    return ""


def build_from_hf(dataset: str, split: str, n: int) -> None:
    from datasets import load_dataset

    ds = load_dataset(dataset, split=split, streaming=True)
    SCENE_DIR.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], 0
    for i, row in enumerate(ds):
        if len(rows) >= n:
            break
        if i == 0 and not _has_image_source(row):
            raise SystemExit(f"Không nhận diện được cột ảnh/caption. Các cột: {list(row.keys())}")
        img = _get_image(row)
        caption = _get_caption(row)
        if img is None or not caption:
            skipped += 1  # ảnh không tải được hoặc thiếu caption
            continue
        name = f"sc_{len(rows) + 1:05d}.jpg"
        img.convert("RGB").save(SCENE_DIR / name, quality=90)
        rows.append({"path": f"scene/{name}", "caption": caption})

    CAPTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(CAPTIONS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "caption"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Đã lưu {len(rows)} ảnh scene vào {SCENE_DIR} và caption vào {CAPTIONS_CSV}")
    if skipped:
        print(f"Bỏ qua {skipped} dòng (ảnh không tải được hoặc thiếu caption).")
    if len(rows) < n:
        print(f"Cảnh báo: chỉ lấy được {len(rows)}/{n} ảnh. Hãy tăng số dòng đọc hoặc kiểm tra kết nối.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="yerevann/coco-karpathy")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=300, help="số ảnh scene cần lấy")
    args = ap.parse_args()
    build_from_hf(args.dataset, args.split, args.n)


if __name__ == "__main__":
    main()
