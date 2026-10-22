# Bộ ảnh mẫu để thử nghiệm

Ảnh lấy từ các gói công khai (scikit-image, scikit-learn, ultralytics). Caption do người tạo viết dựa trên nội dung ảnh.
Bộ này chỉ dùng để kiểm thử pipeline. Số lượng quá ít nên không dùng để đo độ chính xác.

Cấu trúc:
- images/scene/        7 ảnh phong cảnh có caption (captions.csv)
- images/face/p001/    khuôn mặt (1 người)
- images/fullbody/p002/ toàn thân (1 người)
- queries/             3 ảnh truy vấn không nằm trong thư viện
- metadata.csv         sinh bởi src.prepare_metadata
- eval_queries.json    7 truy vấn văn bản + 3 truy vấn ảnh, có relevant_ids

Chạy từ thư mục gốc project:
    python -m src.prepare_metadata --images mau/images --captions mau/captions.csv --out mau/metadata.csv
    python -m src.ingest --system baseline --metadata mau/metadata.csv
    python -m src.eval.evaluate --queries mau/eval_queries.json --systems baseline --k 3
    streamlit run app.py
