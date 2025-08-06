from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from app.config import UPLOAD_DIR,uploaded_files
import os
import uuid


router = APIRouter()

@router.get("/pdfupload.html", response_class=HTMLResponse)
async def get_home():
    with open("app/templates/pdfupload.html", "r", encoding="utf8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@router.post("/upload")
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


@router.get("/files/{file_id}")
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
