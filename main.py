from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import arxiv
import asyncio
import uuid
import os
import numpy as np
import faiss
import requests
import tempfile
from dotenv import load_dotenv
import pymupdf4llm
from langchain.text_splitter import MarkdownTextSplitter
import groq
import uvicorn

# Load environment variables
load_dotenv()

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Directory to store uploaded PDFs (in production, use a persistent storage)
UPLOAD_DIR = tempfile.gettempdir()
# A simple mapping from file_id to file path (in production use a database or better storage solution)
uploaded_files = {}


@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
async def get_home():
    with open("frontend/home.html", "r", encoding="utf8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/arxiv_research/explorepapers.html", response_class=HTMLResponse)
async def get_home():
    with open("frontend/arxiv_research/explorepapers.html", "r", encoding="utf8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.get("/arxiv_research/researchpaper.html", response_class=HTMLResponse)
async def get_home():
    with open("frontend/arxiv_research/researchpaper.html", "r", encoding="utf8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.get("/pdfupload.html", response_class=HTMLResponse)
async def get_home():
    with open("frontend/pdfupload.html", "r", encoding="utf8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# ----------------------------
# Arxiv Search Endpoint
# ----------------------------
async def run_arxiv_search(query: str, num_results: int):
    """Wrap synchronous arxiv API calls in async executor"""
    def sync_search():
        search = arxiv.Search(
            query=query,
            max_results=num_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        return list(search.results())
    return await asyncio.to_thread(sync_search)

@app.get("/search")
async def search_papers(query: str, num_results: int = 10):
    try:
        papers = await run_arxiv_search(query, num_results)
        return {
            "results": [
                {
                    "title": paper.title,
                    "url": paper.entry_id,
                    "summary": paper.summary,
                    "categories": paper.primary_category,
                    "published": paper.published.strftime("%Y-%m-%d"),
                    "pdf_url": paper.pdf_url,
                    "authors": [author.name for author in paper.authors]
                }
                for paper in papers
            ]
        }
    except Exception as e:
        return {"error": str(e)}

# ----------------------------
# QA / PDF Processing Functions
# ----------------------------
def markdown_chunking(text, chunk_size=1024, chunk_overlap=150):
    """
    Splits text based on markdown structure.
    """
    text_splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks

def extract_pdf_chunks(pdf_path):
    """
    Extract full text from PDF as markdown, then chunk it.
    """
    full_text = pymupdf4llm.to_markdown(pdf_path, page_chunks=False)
    if isinstance(full_text, list):
        full_text = "\n\n".join(
            [chunk["text"] if isinstance(chunk, dict) and "text" in chunk else str(chunk)
             for chunk in full_text]
        )
    chunks = markdown_chunking(full_text, chunk_size=1024, chunk_overlap=150)
    return chunks

import openai
from typing import List, Optional

class EmbeddingModel:
    def __init__(
        self,
        model_name: str = "text-embedding-3-large",
        api_key: Optional[str] = None
    ):
        """
        Wrapper around OpenAI's text embedding endpoint.

        Args:
            model_name: which OpenAI embedding model to use.
            api_key: your OpenAI API key. If None, will look for OPENAI_API_KEY env var.
        """
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY must be set as an environment variable or passed in."
                )
        openai.api_key = api_key
        self.model_name = model_name

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts.

        Args:
            texts: list of strings to embed.

        Returns:
            A NumPy array of shape (len(texts), embedding_dim).
        """
        response = openai.embeddings.create(
            model=self.model_name,
            input=texts
        )
        # response.data is a list of dicts, each with an 'embedding' list
        embeddings = [item.embedding for item in response.data]
        return np.vstack(embeddings)


def build_faiss_index(text_chunks, embed_model):
    """
    Build and return a FAISS index from the given text chunks.
    """
    # 1) Embed all chunks first
    embeddings = embed_model.embed(text_chunks)
    # 2) Figure out the dimensionality from the returned array
    #    embeddings.shape == (n_chunks, embedding_dim)
    n_chunks, dim = embeddings.shape

    # 3) Create the index with the correct dimension
    index = faiss.IndexFlatL2(dim)

    # 4) Add your embeddings (cast to float32 if needed)
    index.add(embeddings.astype(np.float32))

    return index, text_chunks


def search_faiss(query, index, text_chunks, embed_model, top_k=4):
    """
    Retrieve similar chunks from FAISS index.
    """
    query_embedding = embed_model.embed([query])
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), top_k)
    results = [text_chunks[idx] for idx in indices[0]]
    return results

def ask_llama_3_70b(query, retrieved_chunks):
    """
    Query Llama-3.3-70B with the retrieved context.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Missing GROQ_API_KEY in environment variables.")
    client = groq.Client(api_key=api_key)
    context = "\n\n".join(retrieved_chunks)
    prompt = f"Context:\n{context}\n\nUser Query: {query}\n\nAnswer:"
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful research assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=512
    )
    return response.choices[0].message.content

def download_pdf(pdf_url: str) -> str:
    """
    Download a PDF from a remote URL and return the local file path.
    """
    response = requests.get(pdf_url)
    response.raise_for_status()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_file.write(response.content)
    temp_file.close()
    return temp_file.name

import traceback
# ----------------------------
# WebSocket Endpoint for QA Chat
# ----------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        # Check if a PDF link was provided as a query parameter
        pdf_url = websocket.query_params.get("pdfLink")
        if pdf_url:
            await websocket.send_text("Downloading PDF from remote source...Please Wait....")
            pdf_path = await asyncio.to_thread(download_pdf, pdf_url)
        else:
            await websocket.send_text("Error! Refresh Page")

        # Process the PDF and send progress messages
        await websocket.send_text("Extracting and processing the PDF...Please Wait....")
        text_chunks = await asyncio.to_thread(extract_pdf_chunks, pdf_path)
        await websocket.send_text("Loading the embedding model...Please Wait....")
        embed_model = await asyncio.to_thread(EmbeddingModel)
        await websocket.send_text("Building FAISS index...Please Wait....")
        faiss_index, stored_chunks = build_faiss_index(text_chunks, embed_model)
        await websocket.send_text("Ready for Q&A!\n🔹 Ask a question about the paper:")

        # Process incoming queries from the client
        while True:
            query = await websocket.receive_text()
            await websocket.send_text("Processing your query...")
            relevant_chunks = search_faiss(query, faiss_index, stored_chunks, embed_model)
            answer = ask_llama_3_70b(query, relevant_chunks)
            result_message = "Answer:\n" + answer
            await websocket.send_text(result_message)
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        tb = traceback.format_exc()
        await websocket.send_text(f"An error occurred: {str(e)}")
        await websocket.send_text(f"An error occurred:\n{tb}")
        await websocket.close()


#CUSTOM PDF
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Endpoint to upload a PDF. Save it locally and return a unique file id.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    uploaded_files[file_id] = file_path
    return {"file_id": file_id}


@app.get("/files/{file_id}")
async def get_file(file_id: str):
    """
    Serve the uploaded PDF file.
    """
    file_path = uploaded_files.get(file_id)
    if file_path is None or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        file_path,
        media_type="application/pdf",
        headers={"Content-Disposition": f"inline; filename={file_id}.pdf"}
    )


@app.websocket("/wsu")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for processing chat queries. The client must pass the file id as a query parameter (?file=).
    """
    await websocket.accept()
    file_id = websocket.query_params.get("file")
    if not file_id or file_id not in uploaded_files:
        await websocket.send_text("Error! No valid PDF file provided. Refresh page and upload a PDF.")
        await websocket.close()
        return

    pdf_path = uploaded_files[file_id]
    try:
        # Send progress messages to the client
        await websocket.send_text("Processing your uploaded PDF...please wait.")
        text_chunks = await asyncio.to_thread(extract_pdf_chunks, pdf_path)
        await websocket.send_text("Loading embedding model...please wait.")
        embed_model = await asyncio.to_thread(EmbeddingModel)  # Uses OPENAI_API_KEY from env
        await websocket.send_text("Building FAISS index...please wait.")
        faiss_index, stored_chunks = build_faiss_index(text_chunks, embed_model)
        await websocket.send_text("Ready for Q&A!\n🔹 Ask a question about the paper:")

        # Process incoming queries
        while True:
            query = await websocket.receive_text()
            await websocket.send_text("Processing your query...")
            relevant_chunks = search_faiss(query, faiss_index, stored_chunks, embed_model)
            answer = ask_llama_3_70b(query, relevant_chunks)
            result_message = "Answer:\n" + answer
            await websocket.send_text(result_message)
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        tb = traceback.format_exc()
        await websocket.send_text(f"An error occurred: {str(e)}")
        await websocket.send_text(f"Details:\n{tb}")
        await websocket.close()


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

# uvicorn main:app --reload