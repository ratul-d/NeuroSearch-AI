from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from app.services.arxiv_utils import run_arxiv_search

router = APIRouter()

@router.get("/arxiv_research/explorepapers.html", response_class=HTMLResponse)
async def get_home():
    with open("app/templates/arxiv_research/explorepapers.html", "r", encoding="utf8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@router.get("/arxiv_research/researchpaper.html", response_class=HTMLResponse)
async def get_home():
    with open("app/templates/arxiv_research/researchpaper.html", "r", encoding="utf8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@router.get("/search")
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