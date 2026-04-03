# 🌟 Aura Document AI: Universal Intelligence Reviewer

**Aura Document AI** is a state-of-the-art, privacy-focused Document Q&A system. It transforms your PDFs, Word Documents (.docx), and Text files (.txt) into an interactive "Knowledge Base" using high-precision local AI.

### 🧠 The Intelligence Module
Unlike basic PDF readers, Aura uses the **SmolLM2-135M-Instruct** model—one of the world's most advanced "Micro-Intelligence" modules. This allows for deep reasoning, perfect English synthesis, and textbook-level comprehension while staying under a **1GB RAM footprint**.

---

## ⚡ Key Features

- **🚀 Universal Support**: Seamlessly analyze resumes, textbooks, and notes in **PDF, Docx, and Txt** formats.
- **📚 Textbook Mode**: Uses **NLTK Semantic Chunking** to understand complex academic paragraphs and long context windows.
- **💼 Resume Precision**: Implements **Hybrid Search (Keyword Boosting)** + **Cross-Encoder Reranking** to distinguish between fine-grained education levels.
- **💻 Multilingual Code Support**: Automatically recognizes and formats **Python, Java, and C++** snippets from technical textbooks.
- **📉 1GB RAM Optimization**: Built specifically for **Streamlit Cloud** and low-resource environments.
- **🔒 Privacy First**: All data is processed in-memory (RAM) and never written to disk or sent to external servers.

---

## 🛠 Installation & Usage

### 1. Simple Clone
```bash
git clone https://github.com/Akhil4007-cpu/Aura-Document-AI.git
cd Aura-Document-AI
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Application
```bash
streamlit run app.py
```

---

## 🏗 The Architecture (3-Stage Thinking)
1. **Recall**: Multi-stage vector search using `all-MiniLM-L6-v2` and FAISS.
2. **Refine**: High-precision reranking using `cross-encoder/ms-marco-MinLM`.
3. **Reason**: Final synthesis and "Tutor-mode" reasoning with `SmolLM2-Instruct`.

---

## 🚀 One-Click Deployment
This project is already pre-configured for **Streamlit Community Cloud**. Just connect your GitHub repo and hit "Deploy."

> [!NOTE]
> Created with ❤️ by Akhil Ananthula.
