"""Cấu hình đường dẫn, tham số và ID mô hình (đều là mô hình mở trên Hugging Face)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DENSE_MIN_SCORE = float(os.getenv("DENSE_MIN_SCORE", "0.30"))

DATA_DIR = ROOT / "data"
IMAGE_DIR = DATA_DIR / "images"            # data/images/<scene|face|fullbody>[/<person_id>]/*.jpg
METADATA_CSV = DATA_DIR / "metadata.csv"   # image_id,path,category,caption,person_id
EVAL_QUERIES = DATA_DIR / "eval_queries.json"

INDEX_DIR = ROOT / "index"                 # index của baseline và RAG
RESULTS_DIR = ROOT / "results"             # kết quả đánh giá

TOP_K = int(os.getenv("TOP_K", "5"))

# ---- Mô hình mở trên Hugging Face (có thể đổi bằng biến môi trường) ----
CLIP_MODEL = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")          # ảnh + văn bản
FACE_MODEL_REPO = os.getenv("FACE_MODEL_REPO", "HammadNaseer/arcface-glintr100-onnx")  # ArcFace ONNX
PERSON_MODEL = os.getenv("PERSON_MODEL", "facebook/detr-resnet-50")           # phát hiện người
VLM_MODEL = os.getenv("VLM_MODEL", "Qwen/Qwen2-VL-2B-Instruct")               # chatbot đa phương thức
