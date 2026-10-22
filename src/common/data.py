"""Cấu trúc dữ liệu dùng chung và hàm đọc metadata / ảnh."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.config import METADATA_CSV, ROOT


@dataclass
class Record:
    image_id: str
    path: str            # đường dẫn tuyệt đối
    category: str        # scene | face | fullbody
    caption: str = ""
    person_id: str = ""


@dataclass
class Hit:
    image_id: str
    score: float
    path: str
    caption: str = ""


def load_metadata(csv_path: Path = METADATA_CSV) -> list[Record]:
    records = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            records.append(Record(
                image_id=row["image_id"],
                path=str(ROOT / row["path"]),
                category=row.get("category") or "scene",
                caption=row.get("caption") or "",
                person_id=row.get("person_id") or "",
            ))
    return records


def load_eval_queries(json_path: Path) -> list[dict]:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def read_image_bgr(path: str | Path) -> np.ndarray | None:
    """Đọc ảnh an toàn với đường dẫn Unicode (cv2.imread lỗi trên Windows với tiếng Việt)."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    except (OSError, cv2.error):
        return None
