from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

@router.get("/", response_class=HTMLResponse)
async def get_home():
    with open("app/templates/home.html", "r", encoding="utf8") as f:
        return HTMLResponse(content=f.read())
