# Báo cáo hoàn thiện Lens

Ngày kiểm thử: **22/10/2026**. Phạm vi: baseline chạy trước, RAG nhẹ trên CPU, chatbot API có nguồn dẫn, UI Streamlit, chuẩn bị code để push GitHub.

## Kết quả triển khai

Baseline đã build lại trên **50 ảnh**, giữ tìm kiếm TF-IDF, LBP, HOG và HSV. Đã sửa việc build lại tích luỹ vector cũ, kiểm tra ID trùng, xử lý query rỗng/K không hợp lệ, và đọc ảnh thiếu/hỏng.

Phát hiện metadata cũ mô tả khác ảnh thực tế sau khi ảnh bị tải lại. Đã sinh lại metadata từ `captions.csv` hiện có và rebuild cả hai index. Hiện có **30 ảnh có caption**, 20 ảnh còn lại chỉ được truy hồi qua đặc trưng ảnh. UI kiểm tra metadata với nội dung index và yêu cầu rebuild khi lệch.

RAG nhẹ dùng `all-MiniLM-L6-v2`, **22.713.216 tham số** đo từ trọng số đã tải, embedding caption 384 chiều. Kết hợp cosine và TF-IDF bằng Reciprocal Rank Fusion. Không cần CLIP, DETR, ArcFace hoặc Qwen để chạy luồng mặc định. Đây là truy hồi caption kết hợp truy hồi ảnh cổ điển; MiniLM không trực tiếp hiểu ảnh.

Chatbot thực hiện:

1. Đọc câu hỏi và tối đa 6 tin nhắn gần nhất, viết lại truy vấn tiếng Anh bằng JSON có kiểm tra cấu trúc.
2. Truy hồi top-K; với câu hỏi về “những ảnh đó”, có thể giới hạn nguồn vào kết quả của hội thoại trước.
3. Gửi ID và caption làm evidence. Khi bật vision, gửi tối đa 4 ảnh nguồn thu nhỏ 512px cùng ảnh truy vấn nếu có.
4. Yêu cầu JSON trả lời tiếng Việt và danh sách citation; kiểm tra các mã trong nội dung khớp danh sách và đều thuộc evidence.
5. Khi thiếu bằng chứng, API lỗi hoặc JSON/citation không hợp lệ: từ chối suy đoán hoặc hiển thị caption gốc và trạng thái lỗi đã lọc thông tin nhạy cảm.

UI đã đổi thành workspace Lens: sidebar cấu hình, chỉ số thư viện, lịch sử chat có nguồn ảnh được giữ lại, đính kèm ảnh, tab tìm kiếm riêng, nút tạo hội thoại mới. Đổi hệ thống hoặc chế độ AI sẽ tạo lịch sử mới để tránh trộn nguồn.

![Chat và ảnh nguồn](screenshots/chat.png)

## API và credentials

Token Hugging Face được nạp từ `.env` vào môi trường. API đọc `url/model/apikey` từ `key_vilao.key`, có thể ghi đè bằng biến môi trường. Không ghi giá trị secret vào mã, log hay báo cáo.

Model ban đầu trong file key là `chib/deepseek-v4.1-flash`, gọi thực tế trả **HTTP 403**, với lý do key chưa được đăng ký quyền dùng model. Endpoint `/models` của cùng key trả model **`claude-sonnet-5`**. Đã xác minh model này trả **HTTP 200**, rồi đặt `AI_MODEL` trong `.env` cục bộ. File key gốc được giữ nguyên.

Đã gọi API thật cho query rewriting, trả lời caption và vision; không chỉ kiểm thử bằng mock. Encoder RAG cục bộ nhỏ hơn 100M; model sinh câu trả lời là dịch vụ API bên ngoài, không tuyên bố model API nhỏ hơn 100M hoặc toàn hệ thống chạy offline.

## Kiểm thử

| Hạng mục | Kết quả |
|---|---|
| `python -m unittest discover -s tests -v` | **25/25 pass** |
| Baseline build/search/save/load | Chạy thật trên ảnh tổng hợp và bộ ảnh hiện có |
| MiniLM | Tải thật, encode thật, build/load/search thật |
| API HTTP lỗi | Fallback hoạt động; body/credentials không lộ ra thông báo |
| Citation bịa / citation không khớp | Bị từ chối, trả caption gốc |
| JSON query planner không hợp lệ | Dùng truy vấn gốc |
| Hội thoại giới hạn nguồn trước | Có kiểm thử scope bằng dữ liệu kiểm soát |
| Streamlit AppTest | Baseline mở và chat không có exception |
| Trình duyệt Edge qua Playwright | Render desktop, gửi chat, hiển thị ảnh nguồn; 0 exception UI |
| Viewport mobile 390 × 844 | Không tràn ngang trong ca đã kiểm tra |
| Nhánh CLIP/ArcFace/DETR/Qwen gốc | Chỉ kiểm thử wiring bằng fake; chưa chạy trọng số thật |

Smoke test thực tế được lưu trong [`results/smoke.json`](../results/smoke.json):

- “Tìm ảnh có người đi xe đạp” → `sc_00003`, `sc_00009`, trả lời tiếng Việt có trích dẫn.
- “Trong những ảnh đó có tàu không?” → dẫn `sc_00003` có người đi xe đạp cạnh tàu; caption ảnh còn lại không đề cập tàu.
- Gửi ảnh `sc_00003` và hỏi người đang làm gì → API vision trả người đàn ông đang đi xe đạp cạnh tàu, dẫn đúng mã ảnh.
- “quantum computer superconducting chip” → không có hit, trả thông báo thiếu bằng chứng.

Lượt đầu có tải/khởi tạo model mất khoảng **26 giây** trong lần smoke đã lưu; các query cục bộ tiếp theo khoảng **14–15 ms**. Không cộng độ trễ API vào latency truy hồi. API có thể có độ trễ và chi phí riêng.

## Đánh giá truy hồi

Tập kiểm tra có **16 câu hỏi text** gán nhãn thủ công theo 30 caption hiện có: 12 ca gần từ khoá caption và 4 ca đồng nghĩa. Không có ảnh truy vấn trong đánh giá này. Đã warm encoder trước khi đo thời gian.

| Hệ thống | Precision@3 | Recall@3 | MAP@3 | Latency warm trung bình |
|---|---:|---:|---:|---:|
| Baseline | 0,375 | 0,797 | 0,812 | 0,722 ms |
| RAG nhẹ | 0,521 | 0,969 | 1,000 | 12,881 ms |

Chi tiết từng truy vấn: [`per_query_k3.csv`](../results/per_query_k3.csv); tổng hợp: [`summary_k3.csv`](../results/summary_k3.csv).

MAP@K ở đây chia cho `min(số ảnh liên quan, K)`. Vì thế MAP@3 có thể bằng 1 khi bài toán có 4 ảnh liên quan nhưng top-3 chỉ lấy được 3 ảnh. Precision@3 thấp hơn vì nhiều câu chỉ có 1 ảnh đúng trong thư viện nhưng vẫn yêu cầu 3 kết quả. Điểm RRF hiển thị trên UI là điểm xếp hạng, không phải độ tin cậy.

Tập này rất nhỏ, được tạo từ caption đang có, chưa có split validation/test độc lập; các số trên chỉ chứng minh pipeline hoạt động và có khả năng tìm từ đồng nghĩa trong những ca đã thử. **Không kết luận RAG chính xác 100% trong sử dụng thực tế.**

## Giới hạn và hướng tiếp theo

- Kiểm tra citation chỉ xác nhận mã nguồn hợp lệ; chưa chứng minh từng phát biểu của LLM được nguồn hỗ trợ. Prompt yêu cầu không coi việc caption không nhắc một đối tượng là bằng chứng đối tượng vắng mặt.
- MiniLM hiện dùng tiếng Anh. Hội thoại tiếng Việt phụ thuộc API viết lại truy vấn; khi offline hoặc API lỗi, nên tìm bằng từ khoá tiếng Anh. Chưa có benchmark tiếng Việt độc lập.
- 20 ảnh chưa có caption hạn chế recall tìm kiếm text. Cần bổ sung caption được xác minh trước khi đo lại.
- Truy hồi ảnh trong chế độ nhẹ dùng baseline. Chưa có đánh giá danh tính khuôn mặt/re-ID toàn thân trên người thật hoặc tập query tách biệt.
- Ngưỡng dense 0,30 là cấu hình khởi đầu, chưa hiệu chỉnh trên validation lớn. Các nguồn cùng từ nhưng khác ngữ cảnh vẫn có thể được truy hồi.
- Lưu vector brute-force và pickle phù hợp demo nhỏ; cần vector store, pipeline ingest có phiên bản dữ liệu và benchmark lớn hơn khi mở rộng.
- API ảnh đã pass một ca thực tế, chưa đánh giá độ chính xác vision có hệ thống. API ngoài không có bảo đảm độ ổn định/chính xác tuyệt đối.

## Bàn giao và bảo vệ secret

Đã thêm `.gitignore` cho `.env`, các file key, cache, weights, ảnh dataset và index; `.env.example` chỉ chứa tên biến/giá trị mẫu. Có script [`check_secrets.py`](../scripts/check_secrets.py) đọc staged files, dò giá trị credentials cục bộ và chặn file key/env trước commit. Dữ liệu và index thực vẫn ở máy cục bộ để tiếp tục chạy.

Lệnh tái chạy, thiết kế luồng và cấu hình được ghi trong [README](../README.md). Thông tin encoder và ví dụ encode tham khảo [model card chính thức](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
