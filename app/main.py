from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import routes_home, routes_arxiv, routes_pdf, routes_ws, routes_webresearch
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()
app.include_router(routes_ws.router)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Static & Template Mounting
app.mount("/static", StaticFiles(directory="app/templates"), name="static")

# Routes
app.include_router(routes_home.router)
app.include_router(routes_arxiv.router)
app.include_router(routes_pdf.router)
app.include_router(routes_webresearch.router)

# Health Check
@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
