import hashlib
import re
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
import torch
import streamlit as st
import nltk
import docx
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder

APP_DIR = Path(__file__).resolve().parent
MODEL_NAME = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# =========================
# DATA STRUCTURE
# =========================
@dataclass
class ChunkRecord:
    doc_id: str
    doc_name: str
    page: int
    chunk_id: int
    text: str
    doc_type: str

# =========================
# MODEL LOADERS
# =========================
@st.cache_resource(show_spinner=False)
def load_embed_model():
    return SentenceTransformer(MODEL_NAME)

@st.cache_resource(show_spinner=False)
def load_rerank_model():
    return CrossEncoder(RERANK_MODEL)

@st.cache_resource(show_spinner=False)
def setup_nltk():
    nltk.download('punkt', quiet=True)

# =========================
# CLASSIFIERS
# =========================
def classify_document(text):
    text = text.lower()
    if any(k in text for k in ["about me", "my name is", "i am", "my goal"]):
        return "personal"
    elif any(k in text for k in ["introduction", "chapter", "definition", "algorithm"]):
        return "study"
    return "general"

def classify_query(query):
    q = query.lower()
    if any(k in q for k in ["my", "me", "mine", "about me"]):
        return "personal"
    if any(k in q for k in ["explain", "what is", "how"]):
        return "study"
    return "general"

# =========================
# EXTRACTION
# =========================
def extract_text_from_file(content_bytes, filename):
    pages = []

    if filename.lower().endswith(".pdf"):
        reader = PdfReader(BytesIO(content_bytes))
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except:
                text = ""
            pages.append((i + 1, text))

    elif filename.lower().endswith(".docx"):
        doc = docx.Document(BytesIO(content_bytes))
        text = "\n".join([p.text for p in doc.paragraphs])
        pages.append((1, text))

    elif filename.lower().endswith(".txt"):
        text = content_bytes.decode("utf-8", errors="ignore")
        pages.append((1, text))

    return pages

# =========================
# CHUNKING
# =========================
def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()

def chunk_text(text, doc_name, page_num, chunk_size=1200):
    text = normalize_whitespace(text)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    doc_type = classify_document(text)

    chunks = []
    current = ""

    for sent in sentences:
        if len(current) + len(sent) < chunk_size:
            current += " " + sent
        else:
            chunk_id = len(chunks)
            chunks.append(ChunkRecord(
                doc_id=f"{doc_name}_{page_num}_{chunk_id}",
                doc_name=doc_name,
                page=page_num,
                chunk_id=chunk_id,
                text=current.strip(),
                doc_type=doc_type
            ))
            current = sent

    if current:
        chunk_id = len(chunks)
        chunks.append(ChunkRecord(
            doc_id=f"{doc_name}_{page_num}_{chunk_id}",
            doc_name=doc_name,
            page=page_num,
            chunk_id=chunk_id,
            text=current.strip(),
            doc_type=doc_type
        ))

    return chunks

# =========================
# FAISS
# =========================
def build_faiss_index(embeddings):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def extract_keywords(query):
    return re.findall(r"\w+", query.lower())

# =========================
# SEARCH (FIXED)
# =========================
def search(embed_model, rerank_model, index, chunks, query, top_k):
    query_type = classify_query(query)

    q_emb = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    scores, ids = index.search(q_emb, max(top_k * 4, 30))

    candidates = []
    keywords = extract_keywords(query)

    # First pass (filtered)
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0 or idx >= len(chunks):
            continue

        c = chunks[idx]

        if query_type != "general" and c.doc_type != query_type:
            continue

        boost = float(score)

        for kw in keywords:
            if kw in c.text.lower():
                boost += 0.5

        candidates.append({
            "text": c.text,
            "doc_name": c.doc_name,
            "page": c.page,
            "score": boost
        })

    # 🔥 Fallback if empty
    if not candidates:
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0 or idx >= len(chunks):
                continue

            c = chunks[idx]

            candidates.append({
                "text": c.text,
                "doc_name": c.doc_name,
                "page": c.page,
                "score": float(score)
            })

    # Rerank
    rerank_scores = rerank_model.predict([[query, c["text"]] for c in candidates])

    for s, c in zip(rerank_scores, candidates):
        c["score"] = float(s)

    return sorted(candidates, key=lambda x: x["score"], reverse=True)[:top_k]

# =========================
# ANSWER
# =========================
def synthesize_answer(results):
    if not results:
        return "Information not found."

    context = "\n\n".join([r["text"] for r in results[:3]])
    return context[:1500]

# =========================
# BUILD DATASET
# =========================
def build_dataset(files):
    all_chunks = []

    for name, content in files:
        pages = extract_text_from_file(content, name)

        for page_num, text in pages:
            all_chunks.extend(chunk_text(text, name, page_num))

    embed_model = load_embed_model()

    embeddings = embed_model.encode(
        [c.text for c in all_chunks],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)

    index = build_faiss_index(embeddings)

    return {"index": index, "chunks": all_chunks}

# =========================
# MAIN
# =========================
def main():
    st.set_page_config(page_title="Universal AI Reviewer", layout="wide")

    if "dataset" not in st.session_state:
        st.session_state.dataset = None

    st.title("Universal AI Reviewer")

    uploaded = st.file_uploader("Upload Documents", accept_multiple_files=True)

    if st.button("Build Index") and uploaded:
        files = [(f.name, f.read()) for f in uploaded]
        st.session_state.dataset = build_dataset(files)
        st.success("Index built successfully!")

    if st.session_state.dataset:
        query = st.text_input("Ask about your documents")

        if query:
            embed_model = load_embed_model()
            rerank_model = load_rerank_model()

            results = search(
                embed_model,
                rerank_model,
                st.session_state.dataset["index"],
                st.session_state.dataset["chunks"],
                query,
                top_k=6
            )

            answer = synthesize_answer(results)

            st.markdown("### Answer")
            st.write(answer)

if __name__ == "__main__":
    main()
