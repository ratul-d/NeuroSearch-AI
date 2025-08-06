import pymupdf4llm
from langchain.text_splitter import MarkdownTextSplitter
import requests, tempfile

def extract_pdf_chunks(pdf_path):
    full_text = pymupdf4llm.to_markdown(pdf_path, page_chunks=False)
    if isinstance(full_text, list):
        full_text = "\n\n".join(
            [chunk.get("text", str(chunk)) for chunk in full_text]
        )
    return MarkdownTextSplitter(chunk_size=1024, chunk_overlap=150).split_text(full_text)

def markdown_chunking(text, chunk_size=1024, chunk_overlap=150):
    text_splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks

def download_pdf(pdf_url: str) -> str:
    response = requests.get(pdf_url)
    response.raise_for_status()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_file.write(response.content)
    temp_file.close()
    return temp_file.name