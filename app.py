import hashlib
import re
import time
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import faiss
import numpy as np
import torch
import streamlit as st
import nltk
import docx
from pypdf import PdfReader
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, CrossEncoder

APP_DIR = Path(__file__).resolve().parent
MODEL_NAME = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
REASONING_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

@dataclass
class ChunkRecord:
    doc_id: str
    doc_name: str
    page: int
    chunk_id: int
    text: str

def _now_ms() -> int:
    return int(time.time() * 1000)

@st.cache_resource(show_spinner=False)
def load_embed_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)

@st.cache_resource(show_spinner=False)
def load_rerank_model() -> CrossEncoder:
    return CrossEncoder(RERANK_MODEL)

@st.cache_resource(show_spinner=False)
def setup_nltk():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except Exception as e:
        print(f"NLTK Download Warning: {e}")

@st.cache_resource(show_spinner=False)
def load_reasoning_module() -> Tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(REASONING_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(REASONING_MODEL, torch_dtype=torch.float32, low_cpu_mem_usage=True).to(device)
    return tokenizer, model

def polish_answer(text: str) -> str:
    if not text: return ""
    text = text.strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text

def extract_text_from_file(content_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
    """Universal Extractor for PDF, DOCX, and TXT."""
    pages: List[Tuple[int, str]] = []
    
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(BytesIO(content_bytes))
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except:
                text = ""
            pages.append((i + 1, text.replace("\x00", " ")))
            
    elif filename.lower().endswith(".docx"):
        doc = docx.Document(BytesIO(content_bytes))
        # Word docs don't have true 'pages' in the buffer, so we treat it as one or by section
        full_text = "\n".join([para.text for para in doc.paragraphs])
        pages.append((1, full_text))
        
    elif filename.lower().endswith(".txt"):
        try:
            text = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = content_bytes.decode("latin-1")
        pages.append((1, text))
        
    return pages

def normalize_whitespace(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_text(text: str, doc_name: str, page_num: int, chunk_size: int = 1200, overlap_sentences: int = 2) -> List[ChunkRecord]:
    text = normalize_whitespace(text)
    if not text: return []
    try:
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
    except:
        sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks: List[ChunkRecord] = []
    current_chunk = []
    current_length = 0
    for i, sent in enumerate(sentences):
        sent = sent.strip()
        if not sent: continue
        current_chunk.append(sent)
        current_length += len(sent)
        if current_length >= chunk_size:
            chunk_block = " ".join(current_chunk)
            chunk_idx = len(chunks)
            chunks.append(ChunkRecord(str(chunk_idx), doc_name, page_num, chunk_idx, chunk_block))
            if len(current_chunk) > overlap_sentences:
                current_chunk = current_chunk[-overlap_sentences:]
                current_length = sum(len(s) for s in current_chunk)
            else:
                current_chunk, current_length = [], 0
    if current_chunk:
        chunk_block = " ".join(current_chunk)
        chunk_idx = len(chunks)
        chunks.append(ChunkRecord(str(chunk_idx), doc_name, page_num, chunk_idx, chunk_block))
    return chunks

def embed_texts_with_progress(model: SentenceTransformer, texts: List[str], batch_size: int = 64, progress: Any = None) -> np.ndarray:
    if not texts: return np.zeros((0, 384), dtype=np.float32)
    vectors = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        batch_vec = model.encode(batch, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        vectors.append(batch_vec)
        if progress is not None:
            progress.progress(min(1.0, (i + len(batch)) / total))
    return np.vstack(vectors)

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def extract_keywords(query: str) -> List[str]:
    stopwords = {"what", "who", "is", "of", "the", "in", "and", "a", "an", "to", "for", "with", "his", "her", "their", "are", "by", "from", "at"}
    return [w for w in re.findall(r"\w+", query.lower()) if w not in stopwords and len(w) > 2]

def synthesize_answer(results: List[Dict[str, Any]], query: str, reasoning_module: Optional[Tuple[Any, Any]] = None, max_sentences: int = 4) -> Tuple[str, List[int]]:
    if not results: return "Information not found in context.", []
    top_context, used_indices = [], []
    seen_text = set()
    for i, r in enumerate(results[:6]):
        text = r.get("text", "").strip()
        # Deduplication check
        text_hash = text[:100] # Use prefix as a simple hash
        if len(text) > 30 and text_hash not in seen_text:
            top_context.append(f"[Source {i+1}]: {text}")
            used_indices.append(i)
            seen_text.add(text_hash)
    if not top_context: return "Information not found in context.", []
    context_str = "\n\n".join(top_context)
    if reasoning_module:
        tokenizer, model = reasoning_module
        system_msg = (
            "You are a PDF analysis assistant. Use the context to answer precisely. "
            "If asked for code, use Markdown blocks. If no relevant info is found, say 'Information not found'."
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"}
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"], 
                max_new_tokens=350, 
                temperature=0.3, 
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                do_sample=True, 
                pad_token_id=tokenizer.eos_token_id
            )
        answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    else:
        answer = polish_answer(" ".join([t.split("]: ")[1] for t in top_context]))
    return answer, sorted(list(set(used_indices)))

def search(embed_model: SentenceTransformer, rerank_model: CrossEncoder, index: faiss.Index, chunks: List[ChunkRecord], query: str, top_k: int) -> List[Dict[str, Any]]:
    q_emb = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    scores, ids = index.search(q_emb, max(top_k * 4, 30))
    candidates, keywords = [], extract_keywords(query)
    for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
        if idx < 0 or idx >= len(chunks): continue
        c = chunks[idx]
        b_score = float(score)
        for kw in keywords:
            if kw in c.text.lower(): b_score += 0.5
        candidates.append({"text": c.text, "doc_name": c.doc_name, "page": c.page, "chunk_id": c.chunk_id, "score": b_score})
    candidates.sort(key=lambda x: x["score"], reverse=True)
    candidates = candidates[:max(top_k * 2, 20)]
    rerank_scores = rerank_model.predict([[query, c["text"]] for c in candidates])
    results = []
    for s, c in zip(rerank_scores.tolist(), candidates):
        c["score"] = float(s)
        results.append(c)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

def build_dataset(files: List[Tuple[str, bytes]]) -> Dict[str, Any]:
    all_chunks = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    for i, (name, content) in enumerate(files):
        status_text.info(f"Processing: **{name}** ({i+1}/{len(files)})")
        # Now uses the Universal Extractor
        pages_data = extract_text_from_file(content, name)
        for page_num, text in pages_data:
            all_chunks.extend(chunk_text(text, name, page_num))
        progress_bar.progress((i + 0.5) / len(files))
    if not all_chunks:
        progress_bar.empty()
        status_text.empty()
        return {}
    status_text.info("🚀 Building High-Speed Vector Index...")
    embed_model = load_embed_model()
    embeddings = embed_texts_with_progress(embed_model, [c.text for c in all_chunks], progress=progress_bar)
    index = build_faiss_index(embeddings)
    progress_bar.empty()
    status_text.empty()
    return {"index": index, "chunks": all_chunks, "embeddings": embeddings, "id": hashlib.md5(str(len(all_chunks)).encode()).hexdigest()[:10]}

def main():
    st.set_page_config(page_title="PDF AI Reviewer", layout="wide")
    if "chat" not in st.session_state: st.session_state.chat = []
    if "dataset" not in st.session_state: st.session_state.dataset = None
    if "history" not in st.session_state: st.session_state.history = []

    setup_nltk()
    embed_model = load_embed_model()
    rerank_model = load_rerank_model()
    reasoning_module = load_reasoning_module()

    with st.sidebar:
        st.title("Settings")
        # Now accepts all 3 formats
        uploaded = st.file_uploader("Upload Documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)
        if st.button("Build/Reset Index") and uploaded:
            mem_files = [(uf.name, uf.read()) for uf in uploaded]
            st.session_state.dataset = build_dataset(mem_files)
        if st.button("Clear Chat"):
            st.session_state.chat = []
            st.session_state.history = []
        st.divider()
        st.info(f"Using: {REASONING_MODEL}")

    st.title("Universal AI Reviewer")
    st.caption("Advanced reasoning for PDF, Word, and Text documents.")

    if not st.session_state.dataset:
        st.warning("Please upload and index a PDF to start.")
    else:
        for msg in st.session_state.chat[-10:]:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
        
        prompt = st.chat_input("Ask about your document...")
        if prompt:
            st.session_state.chat.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    results = search(embed_model, rerank_model, st.session_state.dataset["index"], st.session_state.dataset["chunks"], prompt, top_k=6)
                    answer, _ = synthesize_answer(results, prompt, reasoning_module=reasoning_module)
                    st.markdown(answer)
            st.session_state.chat.append({"role": "assistant", "content": answer})
            st.session_state.history.insert(0, prompt)

if __name__ == "__main__":
    main()
