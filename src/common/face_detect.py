"""Phát hiện khuôn mặt bằng Haar cascade của OpenCV (dùng chung cho baseline và RAG)."""
from __future__ import annotations

import warnings

import cv2
import numpy as np

if not hasattr(cv2, "CascadeClassifier"):
    raise ImportError(
        "Module 'cv2' đang nạp không phải OpenCV đầy đủ (thiếu CascadeClassifier). "
        "Kiểm tra: (1) trong thư mục đang chạy lệnh không có thư mục/file tên 'cv2'; "
        "(2) chạy: pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless cv2, "
        "rồi: pip install opencv-python"
    )

_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_CASCADE = cv2.CascadeClassifier(_CASCADE_PATH)
if _CASCADE.empty():
    # Một số bản cài OpenCV không kèm cascade: vẫn chạy, nhưng coi cả ảnh là khuôn mặt
    warnings.warn("Không tìm thấy Haar cascade; khuôn mặt sẽ dùng toàn bộ ảnh thay vì phát hiện.")
    _CASCADE = None


def has_detector() -> bool:
    return _CASCADE is not None


def largest_face(bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    """Trả về (x, y, w, h) của khuôn mặt lớn nhất, hoặc None nếu không thấy."""
    if _CASCADE is None:
        return None
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = _CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    return int(x), int(y), int(w), int(h)


def face_crop(bgr: np.ndarray, margin: float = 0.2) -> np.ndarray | None:
    """Crop khuôn mặt lớn nhất có viền. Không có cascade thì trả về cả ảnh.
    Trả về None nếu cascade có sẵn nhưng không thấy khuôn mặt."""
    if _CASCADE is None:
        return bgr
    rect = largest_face(bgr)
    if rect is None:
        return None
    x, y, w, h = rect
    H, W = bgr.shape[:2]
    mx, my = int(w * margin), int(h * margin)
    return bgr[max(0, y - my):min(H, y + h + my), max(0, x - mx):min(W, x + w + mx)]
