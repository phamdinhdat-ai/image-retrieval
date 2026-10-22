"""Bộ mã hoá đa phương thức, toàn bộ là mô hình mở trên Hugging Face.

- CLIPEncoder     : openai/clip-vit-base-patch32 (ảnh và văn bản cùng không gian vector)
- FaceEncoder     : ArcFace glintr100 dạng ONNX (HammadNaseer/arcface-glintr100-onnx),
                    khuôn mặt tìm bằng Haar cascade của OpenCV
- PersonDetector  : facebook/detr-resnet-50 (COCO, lớp 'person') để cắt vùng toàn thân

Torch / transformers / onnxruntime chỉ được import khi khởi tạo, nên module này
import được ngay cả khi chưa cài các thư viện nặng (tiện cho kiểm thử).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.common.face_detect import face_crop
from src.config import CLIP_MODEL, FACE_MODEL_REPO, PERSON_MODEL


class CLIPEncoder:
    def __init__(self, model_id: str = CLIP_MODEL):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)

    @staticmethod
    def _pooled(out):
        # transformers v4 trả về tensor, v5 trả về BaseModelOutputWithPooling
        return out.pooler_output if hasattr(out, "pooler_output") else out

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            feats = self._pooled(self.model.get_image_features(**inputs))
        return self.torch.nn.functional.normalize(feats, dim=-1).cpu().numpy()

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with self.torch.no_grad():
            feats = self._pooled(self.model.get_text_features(**inputs))
        return self.torch.nn.functional.normalize(feats, dim=-1).cpu().numpy()


class FaceEncoder:
    """ArcFace 512-d. Lưu ý: chỉ crop theo Haar box, không căn chỉnh theo landmark
    nên độ chính xác thấp hơn so với pipeline đầy đủ (RetinaFace/SCRFD + alignment)."""

    INPUT_SIZE = 112

    def __init__(self, repo: str = FACE_MODEL_REPO):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download, list_repo_files

        onnx_files = [f for f in list_repo_files(repo) if f.endswith(".onnx")]
        if not onnx_files:
            raise RuntimeError(f"Repo {repo} không có file .onnx nào")
        model_path = hf_hub_download(repo, onnx_files[0])
        self.session = ort.InferenceSession(model_path, providers=ort.get_available_providers())
        self.input_name = self.session.get_inputs()[0].name

    def encode(self, bgr: np.ndarray) -> np.ndarray | None:
        """Trả về None nếu có detector nhưng không thấy khuôn mặt."""
        face = face_crop(bgr)
        if face is None or face.size == 0:
            return None
        rgb = cv2.cvtColor(cv2.resize(face, (self.INPUT_SIZE, self.INPUT_SIZE)), cv2.COLOR_BGR2RGB)
        blob = ((rgb.astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None]
        emb = self.session.run(None, {self.input_name: blob})[0][0]
        return emb / (np.linalg.norm(emb) + 1e-12)


class PersonDetector:
    def __init__(self, model_id: str = PERSON_MODEL):
        import torch
        from transformers import AutoImageProcessor, DetrForObjectDetection

        self.torch = torch
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = DetrForObjectDetection.from_pretrained(model_id).eval()
        self.person_label = next(k for k, v in self.model.config.id2label.items() if v == "person")

    def crop(self, img: Image.Image) -> Image.Image:
        """Cắt người lớn nhất trong ảnh; nếu không thấy ai thì trả về ảnh gốc."""
        inputs = self.processor(images=img, return_tensors="pt")
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        res = self.processor.post_process_object_detection(
            outputs, threshold=0.5, target_sizes=[img.size[::-1]])[0]  # target_sizes = (h, w)
        boxes = res["boxes"][res["labels"] == self.person_label]
        if len(boxes) == 0:
            return img
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        x0, y0, x1, y1 = boxes[int(areas.argmax())].tolist()
        return img.crop((int(x0), int(y0), int(x1), int(y1)))


@dataclass
class Encoders:
    """Gói các bộ mã hoá; có thể thay bằng bản giả khi kiểm thử."""
    clip: Any
    face: Any
    person: Any

    @classmethod
    def default(cls) -> "Encoders":
        return cls(clip=CLIPEncoder(), face=FaceEncoder(), person=PersonDetector())
