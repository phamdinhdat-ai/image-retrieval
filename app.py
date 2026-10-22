"""Run: python -m streamlit run app.py."""
from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

from src.common.data import load_metadata
from src.config import INDEX_DIR, METADATA_CSV
from src.rag.chat import ChatService
from src.rag.lightweight import fingerprint

st.set_page_config(page_title="Lens · Trò chuyện cùng bộ ảnh", page_icon="◈", layout="wide")
st.markdown("""<style>
.stApp {background:linear-gradient(150deg,#f4f7fb,#eef3f8);color:#15243b}
.block-container {max-width:1180px;padding-top:2.5rem}
[data-testid="stSidebar"] {background:#102338;color:#f1f5f9}
[data-testid="stSidebar"] label,[data-testid="stSidebar"] p {color:#f1f5f9}
h1 {letter-spacing:-1.5px;font-weight:750!important}
[data-testid="stChatMessage"] {background:white;border:1px solid #e0e7ef;border-radius:18px;padding:1.2rem;margin-bottom:1rem}
[data-testid="stMetric"] {background:white;padding:14px 20px;border:1px solid #e0e7ef;border-radius:14px}
.stButton>button {border-radius:10px;background:#145b68;color:white}
[data-testid="stSidebar"] .stButton>button p {color:white}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def load_system(name, stamp):
    if name == "baseline":
        from src.baseline.retriever import BaselineRetriever
        return BaselineRetriever().load(INDEX_DIR / name)
    from src.rag.lightweight import LightweightRetriever
    return LightweightRetriever().load(INDEX_DIR / name)


def gallery(hits):
    cols = st.columns(3)
    for i, hit in enumerate(hits):
        with cols[i % 3]:
            if Path(hit["path"]).exists():
                st.image(hit["path"], use_container_width=True)
            st.markdown(f"**{hit['image_id']}**")
            st.caption(hit.get("caption") or "Chưa có chú thích")
            st.caption(f"Điểm xếp hạng: {hit['score']:.4f} · không phải xác suất")


with st.sidebar:
    st.markdown("## ◈ LENS")
    st.caption("IMAGE KNOWLEDGE WORKSPACE")
    st.divider()
    label = st.radio("Công cụ truy xuất", ["Baseline · TF-IDF + OpenCV", "RAG nhẹ · MiniLM + hybrid"])
    name = "baseline" if label.startswith("Baseline") else "lightweight"
    use_ai = st.toggle("Dùng API AI", value=False)
    vision = st.toggle("Gửi ảnh cho API có hỗ trợ vision", value=False, disabled=not use_ai)
    if use_ai:
        st.caption("Caption và câu hỏi được gửi đến API đã cấu hình. Bật vision sẽ gửi thêm ảnh top-K và ảnh tải lên.")
    k = st.slider("Số nguồn mỗi câu trả lời", 1, 8, 4)
    st.divider()
    st.caption("MiniLM: khoảng 23M tham số, chạy CPU. Truy vấn tiếng Việt được API viết lại sang tiếng Anh khi bật AI.")
    if st.button("＋ Cuộc trò chuyện mới", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.caption("BỘ ẢNH CỦA BẠN / TRỢ LÝ TRI THỨC")
st.title("Khám phá ảnh. Hỏi bằng lời.")
st.markdown("Tìm hình ảnh, đối chiếu chú thích và trò chuyện với nguồn dẫn rõ ràng.")
if not (INDEX_DIR / name / "records.pkl").exists():
    st.info(f"Chưa có index. Chạy: python -m src.ingest --system {name}")
    st.stop()
try:
    stamp = (INDEX_DIR / name / "records.pkl").stat().st_mtime_ns
    retriever = load_system(name, stamp)
except Exception as exc:
    st.error(f"Không tải được index ({type(exc).__name__}). Hãy chạy lại ingest cho {name}.")
    st.stop()
if METADATA_CSV.exists() and fingerprint(load_metadata()) != fingerprint(list(retriever.records.values())):
    st.warning("Metadata đã thay đổi. Hãy chạy lại ingest trước khi truy vấn để tránh dẫn dữ liệu cũ.")
    st.stop()
a, b, c = st.columns(3)
a.metric("Ảnh trong thư viện", len(retriever.records))
b.metric("Có chú thích", sum(bool(r.caption) for r in retriever.records.values()))
c.metric("Truy xuất", "TF-IDF" if name == "baseline" else "Hybrid RRF")
chat, search = st.tabs(["💬 Trò chuyện với dữ liệu", "🔎 Tìm ảnh trực tiếp"])
with search:
    mode_label = st.selectbox("Tìm theo", ["Văn bản", "Ảnh tương tự", "Khuôn mặt", "Toàn thân"])
    modes = {"Văn bản": "text", "Ảnh tương tự": "similar", "Khuôn mặt": "face", "Toàn thân": "fullbody"}
    mode = modes[mode_label]
    query = st.text_input("Mô tả bằng tiếng Anh", placeholder="a person riding a bicycle") if mode == "text" else ""
    upload = st.file_uploader("Ảnh truy vấn", type=["jpg", "jpeg", "png", "webp"], key="search_file") if mode != "text" else None
    if st.button("Tìm trong thư viện", type="primary"):
        try:
            with st.spinner("Đang tìm ảnh…"), tempfile.TemporaryDirectory() as tmp:
                if upload:
                    path = Path(tmp) / "query.png"
                    Image.open(upload).convert("RGB").save(path)
                    query = str(path)
                hits = retriever.search(mode, query, k) if query else []
            gallery([vars(h) for h in hits])
            if not hits:
                st.info("Không có kết quả phù hợp.")
        except (OSError, ValueError):
            st.error("Không đọc được ảnh hoặc truy vấn. Hãy thử tệp khác.")
with chat:
    context = (name, use_ai, vision)
    if st.session_state.get("chat_context") != context:
        st.session_state.messages = []
        st.session_state.chat_context = context
    if not st.session_state.messages:
        st.info("Bắt đầu với: Tìm ảnh có người đi xe đạp. Khi tắt AI, dùng: person riding bicycle.")
    with st.expander("Đính kèm ảnh để tìm ảnh liên quan"):
        attachment = st.file_uploader("Ảnh cho lượt chat tiếp theo", type=["jpg", "jpeg", "png", "webp"], key="chat_file")
        image_mode = st.selectbox("So khớp ảnh", ["similar", "face", "fullbody"])
        st.caption("Gỡ ảnh sau khi hỏi nếu muốn chuyển lại sang tìm bằng văn bản.")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("hits"):
                with st.expander("Xem ảnh và nguồn trích dẫn", expanded=message is st.session_state.messages[-1]):
                    gallery(message["hits"])
                st.caption(message.get("status", ""))
    question = st.chat_input("Hỏi về bộ ảnh của bạn…", max_chars=4000)
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        try:
            with st.chat_message("assistant"), st.spinner("Đang truy xuất bằng chứng…"), tempfile.TemporaryDirectory() as tmp:
                path = None
                if attachment:
                    path = Path(tmp) / "query.png"
                    Image.open(attachment).convert("RGB").save(path)
                result = ChatService(retriever).ask(question, st.session_state.messages, k, path, image_mode, use_ai, vision)
                st.markdown(result.answer)
                with st.expander("Xem ảnh và nguồn trích dẫn", expanded=True):
                    gallery([vars(h) for h in result.hits])
                st.caption(f"{result.status} · Truy vấn: {result.retrieval_query}")
            st.session_state.messages.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": result.answer, "hits": [vars(h) for h in result.hits], "status": result.status},
            ])
            st.rerun()
        except (OSError, ValueError):
            st.error("Không xử lý được ảnh hoặc truy vấn. Hãy kiểm tra dữ liệu và thử lại.")
