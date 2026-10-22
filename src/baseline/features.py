"""Baseline: đặc trưng xử lý ảnh cổ điển (không dùng deep learning).

- face      : Haar cascade tìm khuôn mặt -> LBP (4x4 lưới, 256 bin) -> căn bậc hai (Hellinger)
              trả về None nếu không thấy khuôn mặt (ảnh đó không được đưa vào index khuôn mặt)
- fullbody  : HOG people detector tìm người -> crop -> HOG (64x128)
- similar   : histogram màu HSV (8x4x4) + HOG toàn ảnh (64x64) -> CBIR kinh điển
"""
from __future__ import annotations

import cv2
import numpy as np

from src.common.face_detect import face_crop

_PEOPLE = cv2.HOGDescriptor()
_PEOPLE.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
_HOG_BODY = cv2.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9)
_HOG_SCENE = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)


def _largest(rects):
    if rects is None or len(rects) == 0:
        return None
    x, y, w, h = max(rects, key=lambda r: r[2] * r[3])
    return int(x), int(y), int(w), int(h)


def _crop(img, rect, margin: float):
    x, y, w, h = rect
    H, W = img.shape[:2]
    mx, my = int(w * margin), int(h * margin)
    return img[max(0, y - my):min(H, y + h + my), max(0, x - mx):min(W, x + w + mx)]


def _lbp_hist(gray: np.ndarray, grid=(4, 4)) -> np.ndarray:
    """LBP 8 lân cận, chia ảnh thành lưới, nối histogram các ô."""
    g = gray.astype(np.int16)
    c = g[1:-1, 1:-1]
    neighbors = [
        g[0:-2, 0:-2], g[0:-2, 1:-1], g[0:-2, 2:], g[1:-1, 2:],
        g[2:, 2:], g[2:, 1:-1], g[2:, 0:-2], g[1:-1, 0:-2],
    ]
    code = np.zeros(c.shape, dtype=np.uint8)
    for i, n in enumerate(neighbors):
        code |= (n >= c).astype(np.uint8) << i
    h, w = code.shape
    feats = []
    for i in range(grid[0]):
        for j in range(grid[1]):
            cell = code[i * h // grid[0]:(i + 1) * h // grid[0],
                        j * w // grid[1]:(j + 1) * w // grid[1]]
            hist, _ = np.histogram(cell, bins=256, range=(0, 256))
            feats.append(hist / max(cell.size, 1))
    return np.sqrt(np.concatenate(feats))


def face_feature(img: np.ndarray) -> np.ndarray | None:
    face = face_crop(img)
    if face is None:
        return None
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(cv2.resize(gray, (96, 96)))
    return _lbp_hist(gray)


def fullbody_feature(img: np.ndarray) -> np.ndarray:
    rects, _ = _PEOPLE.detectMultiScale(img, winStride=(8, 8), padding=(8, 8), scale=1.05)
    rect = _largest(rects)
    crop = _crop(img, rect, 0.05) if rect is not None else img
    gray = cv2.cvtColor(cv2.resize(crop, (64, 128)), cv2.COLOR_BGR2GRAY)
    return _HOG_BODY.compute(gray).ravel().astype(np.float32)


def similar_feature(img: np.ndarray) -> np.ndarray:
    small = cv2.resize(img, (128, 128))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
    hist = hist / (np.linalg.norm(hist) + 1e-12)
    gray = cv2.resize(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (64, 64))
    hog = _HOG_SCENE.compute(gray).ravel()
    hog = hog / (np.linalg.norm(hog) + 1e-12)
    return np.concatenate([hist, hog]).astype(np.float32)


_EXTRACTORS = {"face": face_feature, "fullbody": fullbody_feature, "similar": similar_feature}


def extract(mode: str, img: np.ndarray) -> np.ndarray | None:
    return _EXTRACTORS[mode](img)
