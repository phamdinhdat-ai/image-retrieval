"""Chatbot đa phương thức chạy cục bộ bằng VLM mở trên Hugging Face (mặc định Qwen2-VL-2B-Instruct).

Ảnh truy xuất và caption được đưa vào làm ngữ cảnh, nên mô hình có thể trả lời
các câu hỏi cần nhìn ảnh, không chỉ đọc caption.
"""
from __future__ import annotations

from PIL import Image

from src.common.data import Hit
from src.config import VLM_MODEL

SYSTEM_PROMPT = (
    "Bạn là trợ lý hỏi đáp về một bộ ảnh. "
    "Chỉ trả lời dựa trên các ảnh và chú thích được cung cấp. "
    "Nếu không đủ thông tin, nói rõ là không tìm thấy trong dữ liệu. "
    "Khi nhắc tới một ảnh, ghi mã ảnh dạng [image_id]. "
    "Trả lời bằng tiếng Việt, ngắn gọn."
)


class VLMGenerator:
    def __init__(self, model_id: str = VLM_MODEL, max_images: int = 4,
                 max_side: int = 512, max_new_tokens: int = 256):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.max_images = max_images
        self.max_side = max_side
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype="auto", device_map="auto").eval()

    def answer(self, question: str, hits: list[Hit], history: list[dict] | None = None) -> str:
        """history: danh sách {'role', 'content'} dạng text của các lượt trước."""
        if not hits:
            return "Không tìm thấy ảnh nào liên quan trong bộ dữ liệu."

        images, content = [], []
        for h in hits[:self.max_images]:
            img = Image.open(h.path).convert("RGB")
            img.thumbnail((self.max_side, self.max_side))  # giảm số token ảnh
            images.append(img)
            content.append({"type": "text",
                            "text": f"Ảnh [{h.image_id}], chú thích: {h.caption or 'không có'}"})
            content.append({"type": "image"})
        content.append({"type": "text", "text": f"Câu hỏi: {question}"})

        messages = ([{"role": "system", "content": SYSTEM_PROMPT}]
                    + list(history or [])
                    + [{"role": "user", "content": content}])
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, padding=True, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
