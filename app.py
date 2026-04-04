import re
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
import torch
import streamlit as st
import docx
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM

APP_DIR = Path(__file__).resolve().parent
MODEL_NAME = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
REASONING_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


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


# =========================
# MODELS (cached globally)
# =========================
@st.cache_resource
def load_embed_model():
    return SentenceTransformer(MODEL_NAME)


@st.cache_resource
def load_rerank_model():
    return CrossEncoder(RERANK_MODEL)


@st.cache_resource
def load_reasoning_model():
    tokenizer = AutoTokenizer.from_pretrained(REASONING_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        REASONING_MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()  # faster inference, disables dropout
    return tokenizer, model


# =========================
# EXTRACTION
# =========================
def extract_text_from_file(content_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
    pages = []

    if filename.lower().endswith(".pdf"):
        reader = PdfReader(BytesIO(content_bytes))
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception:
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
def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, doc_name: str, page_num: int, chunk_size: int = 1200) -> List[ChunkRecord]:
    text = normalize_whitespace(text)
    if not text:  # FIX: skip empty pages
        return []

    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""

    for sent in sentences:
        if len(current) + len(sent) < chunk_size:
            current += " " + sent
        else:
            if current.strip():  # FIX: don't append empty chunks
                chunk_id = len(chunks)
                chunks.append(ChunkRecord(
                    f"{doc_name}_{page_num}_{chunk_id}",
                    doc_name,
                    page_num,
                    chunk_id,
                    current.strip()
                ))
            current = sent

    if current.strip():
        chunk_id = len(chunks)
        chunks.append(ChunkRecord(
            f"{doc_name}_{page_num}_{chunk_id}",
            doc_name,
            page_num,
            chunk_id,
            current.strip()
        ))

    return chunks


# =========================
# FAISS
# =========================
def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    # FIX: use IVF for speed when dataset is large, fallback to Flat for small
    if len(embeddings) > 500:
        quantizer = faiss.IndexFlatIP(dim)
        nlist = min(64, len(embeddings) // 10)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.nprobe = 10  # balance speed vs recall
    else:
        index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


# =========================
# SEARCH
# =========================
def search(
    embed_model: SentenceTransformer,
    rerank_model: CrossEncoder,
    index,
    chunks: List[ChunkRecord],
    query: str,
    top_k: int = 3,
) -> List[dict]:
    q_emb = embed_model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    ).astype(np.float32)

    retrieve_k = min(max(top_k * 5, 30), len(chunks))  # FIX: don't exceed chunk count
    scores, ids = index.search(q_emb, retrieve_k)

    candidates = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        c = chunks[idx]
        candidates.append({
            "text": c.text,
            "doc_name": c.doc_name,
            "page": c.page,
            "embed_score": float(score),
        })

    if not candidates:
        return []

    # Re-rank
    rerank_scores = rerank_model.predict([[query, c["text"]] for c in candidates])
    for s, c in zip(rerank_scores, candidates):
        c["score"] = float(s)

    ranked = sorted(candidates, key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]


# =========================
# ANSWER SYNTHESIS
# =========================
def synthesize_answer(results: List[dict], query: str, tokenizer, model) -> str:
    if not results:
        return "No relevant information found in the uploaded documents."

    # Use top-2 chunks for richer context
    context = "\n\n".join(r["text"] for r in results[:2])

    prompt = (
        "You are a helpful assistant. Answer the question using only the context below. "
        "Be concise and factual.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,  # FIX: explicit truncation limit
    ).to(model.device)

    with torch.no_grad():  # FIX: no_grad for faster inference & less memory
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,           # greedy decode — fast & deterministic
            repetition_penalty=1.1,    # reduces repetitive output
            pad_token_id=tokenizer.eos_token_id,  # FIX: avoids padding warning
        )

    # FIX: decode only new tokens, not the prompt
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    if not answer:
        answer = "Could not generate an answer from the context."

    return answer


# =========================
# BUILD DATASET
# =========================
def build_dataset(files: List[Tuple[str, bytes]]) -> dict:
    all_chunks: List[ChunkRecord] = []

    for name, content in files:
        pages = extract_text_from_file(content, name)
        for page_num, text in pages:
            all_chunks.extend(chunk_text(text, name, page_num))

    if not all_chunks:
        st.error("No text could be extracted from the uploaded files.")
        return {}

    embed_model = load_embed_model()
    embeddings = embed_model.encode(
        [c.text for c in all_chunks],
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=64,          # FIX: explicit batch size for speed
        show_progress_bar=True,
    ).astype(np.float32)

    index = build_faiss_index(embeddings)
    return {"index": index, "chunks": all_chunks}


# =========================
# MAIN
# =========================
def main():
    st.set_page_config(page_title="Universal AI Reviewer", layout="wide")
    st.title("📄 Universal AI Reviewer")

    # FIX: initialise both dataset AND qa_history in session state
    if "dataset" not in st.session_state:
        st.session_state.dataset = None
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []  # list of (question, answer, sources)

    # ---- Upload & Index ----
    uploaded = st.file_uploader(
        "Upload Documents (PDF, DOCX, TXT)",
        accept_multiple_files=True,
        type=["pdf", "docx", "txt"],
    )

    if st.button("Build Index") and uploaded:
        with st.spinner("Building index…"):
            files = [(f.name, f.read()) for f in uploaded]
            result = build_dataset(files)
            if result:
                st.session_state.dataset = result
                st.session_state.qa_history = []  # reset history on new index
                st.success(f"Index built: {len(result['chunks'])} chunks from {len(files)} file(s).")

    # ---- Q&A ----
    if st.session_state.dataset:
        st.divider()

        # FIX: use a form so Enter submits and the input clears after submission
        with st.form("qa_form", clear_on_submit=True):
            query = st.text_input("Ask a question about your documents")
            submitted = st.form_submit_button("Ask")

        if submitted and query.strip():
            embed_model = load_embed_model()
            rerank_model = load_rerank_model()
            tokenizer, model = load_reasoning_model()

            with st.spinner("Searching & generating answer…"):
                results = search(
                    embed_model,
                    rerank_model,
                    st.session_state.dataset["index"],
                    st.session_state.dataset["chunks"],
                    query.strip(),
                    top_k=3,
                )
                answer = synthesize_answer(results, query.strip(), tokenizer, model)

            # FIX: prepend so newest Q&A appears at top
            st.session_state.qa_history.insert(0, (query.strip(), answer, results))

        # ---- Display history (all previous Q&As stay visible) ----
        for i, (q, a, srcs) in enumerate(st.session_state.qa_history):
            with st.container():
                st.markdown(f"**Q{len(st.session_state.qa_history) - i}: {q}**")
                st.markdown(a)
                with st.expander("📚 Source chunks"):
                    for j, s in enumerate(srcs):
                        st.markdown(
                            f"**[{j+1}] {s['doc_name']} — page {s['page']}** "
                            f"*(score: {s['score']:.3f})*\n\n{s['text'][:400]}…"
                        )
                st.divider()

        # Clear history button
        if st.session_state.qa_history:
            if st.button("🗑️ Clear Q&A history"):
                st.session_state.qa_history = []
                st.rerun()


if __name__ == "__main__":
    main()