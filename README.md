# NeuroSearch AI

**NeuroSearch AI** is a RAG (Retrieval-Augmented Generation) based research assistant that helps users interact with academic papers intelligently. Users can either search and query **arXiv** papers or upload their own PDF documents, then ask questions and receive precise answers grounded in the document's content.

---

##  Architecture Overview

NeuroSearch AI combines modern NLP tools and scalable architecture components:

| Component   | Technology                          |
|-------------|-------------------------------------|
| Frontend    | HTML, TailwindCSS, JavaScript       |
| Backend     | Python FastAPI                      |
| Embedding   | `text-embedding-3-large` (OpenAI)   |
| LLM         | `llama-3.1-8b-instant` via Groq     |
| Vector DB   | FAISS                               |
| Hosting     | Render                              |




##  Features

* Search academic papers directly from **arXiv** via their official API.
* Upload personal PDFs and ask context-aware questions.
* Chunking and embedding of documents for efficient retrieval.
* Semantic similarity search using FAISS vector store.
* Context-aware responses using LLM with top-k retrieved chunks.
* Clean, fast, and responsive web interface.

---

##  How It Works

### 1. **Document Ingestion**

* User uploads a PDF or selects an arXiv paper.
* The file is first converted to structured markdown using `pymupdf4llm`.
* The resulting markdown is chunked into manageable text segments.
* Chunking is performed using LangChain’s `MarkdownTextSplitter`.

### 2. **Embedding**

* Each chunk is embedded using OpenAI’s `text-embedding-3-large`.
* Embeddings are stored in a FAISS vector store.

### 3. **Question Answering**

* User asks a question.
* FAISS returns the top-k most relevant chunks based on similarity.
* The question and retrieved context are sent to `llama-3.1-8b-instant` (Groq).
* The model generates a response based on context.

---

##  Deployment

### Requirements

* Python 3.10+
* OpenAI API Key
* Groq API Key

### Run Locally

```bash
# Clone the repository
git clone https://github.com/ratul-d/NeuroSearch-AI.git
cd NeuroSearch-AI

# Backend setup
pip install -r requirements.txt
uvicorn app.main:app --reload

```

