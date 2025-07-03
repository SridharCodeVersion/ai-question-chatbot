import os, sys, streamlit as st
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_loader import load_pdf_text
from src.utils import save_uploaded_file
from src.retriever import chunk_text, build_index, retrieve
from src.question_generator import generate_questions

st.set_page_config(page_title="AI QGen", layout="centered")
st.title("📘 AI-Based Question Generator")
uploaded = st.file_uploader("📄 Upload PDF", type="pdf")

if st.button("🚀 Generate Questions"):
    if not uploaded:
        st.error("Upload a PDF first.")
    else:
        file_path = save_uploaded_file(uploaded)
        text = load_pdf_text(file_path)
        chunks = chunk_text(text)
        index, embs = build_index(chunks)
        top = retrieve(chunks, index, embs, query="concept", top_k=3)
        st.subheader("📌 Context Preview")
        for c in top: st.code(c[:200] + "...")
        qs = generate_questions(top, num_questions=5)
        st.subheader("📄 Generated Questions")
        for i, q in enumerate(qs,1):
            st.markdown(f"**Q{i}.** {q}")
