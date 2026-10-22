"""Quét thư mục ảnh và tạo metadata.csv.

Quy ước thư mục:
    <images>/scene/*.jpg                ảnh phong cảnh / hoạt động
    <images>/face/<person_id>/*.jpg     ảnh khuôn mặt; cùng person_id = cùng một người
    <images>/fullbody/<person_id>/*.jpg ảnh toàn thân; cùng person_id = cùng một người

Mặc định: <images> = data/images, caption lấy từ data/captions.csv, ghi ra data/metadata.csv.
Ảnh không có caption thì để trống; bạn có thể điền tay trong metadata.csv.

Chạy:
    python -m src.prepare_metadata
    python -m src.prepare_metadata --images mau/images --captions mau/captions.csv --out mau/metadata.csv
"""
import argparse
import csv
from pathlib import Path

from src.build_dataset import CAPTIONS_CSV
from src.config import IMAGE_DIR, METADATA_CSV, ROOT

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PREFIX = {"scene": "sc", "face": "fa", "fullbody": "fb"}
FIELDS = ["image_id", "path", "category", "caption", "person_id"]


def _load_captions(captions_csv: Path) -> dict[str, str]:
    """Khoá là đường dẫn tương đối với thư mục ảnh, ví dụ 'scene/sc_00001.jpg'."""
    if not captions_csv.exists():
        return {}
    with open(captions_csv, newline="", encoding="utf-8") as f:
        return {row["path"]: row["caption"] for row in csv.DictReader(f)}


def build_metadata(images_dir: Path, captions_csv: Path, out_csv: Path) -> int:
    captions = _load_captions(captions_csv)
    rows, counters = [], {}
    for path in sorted(images_dir.rglob("*")):
        if path.suffix.lower() not in EXTS:
            continue
        parts = path.relative_to(images_dir).parts
        category = parts[0] if parts[0] in PREFIX else "scene"
        person_id = parts[1] if category != "scene" and len(parts) > 2 else ""

        n = counters.get(category, 0) + 1
        counters[category] = n
        rel = path.relative_to(images_dir).as_posix()
        rows.append({
            "image_id": f"{PREFIX[category]}_{n:05d}",
            "path": path.relative_to(ROOT).as_posix(),
            "category": category,
            "caption": captions.get(rel, ""),
            "person_id": person_id,
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    missing = sum(1 for r in rows if not r["caption"])
    print(f"Đã ghi {len(rows)} dòng vào {out_csv} ({missing} ảnh chưa có caption).")
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, default=IMAGE_DIR)
    ap.add_argument("--captions", type=Path, default=CAPTIONS_CSV)
    ap.add_argument("--out", type=Path, default=METADATA_CSV)
    args = ap.parse_args()
    build_metadata(args.images.resolve(), args.captions.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
