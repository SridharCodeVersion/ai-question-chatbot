# AI-Based Question Generator (JEE/BITSAT)

This project extracts content from a user-uploaded PDF, retrieves relevant parts using Sentence Transformers + FAISS, and generates questions using FLAN-T5.

## Features
- PDF upload
- Context-based info retrieval
- Question generation using LLM
- Visual Streamlit UI

## Usage
```
pip install -r requirements.txt
streamlit run pages/streamlit_app.py
```