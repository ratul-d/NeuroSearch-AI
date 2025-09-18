from fastapi import APIRouter
from fastapi.responses import JSONResponse, HTMLResponse
from app.services.web_research import ResearchDeps,research_agent
from pydantic import BaseModel
import asyncio

router = APIRouter()

@router.get("/webresearch.html", response_class=HTMLResponse)
async def get_home():
    with open("app/templates/webresearch.html", "r", encoding="utf8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


class ResearchRequest(BaseModel):
    topic: str

# Endpoint that receives the research topic from the website
@router.post("/research")
async def run_research(request: ResearchRequest):
    # Create research dependencies using the topic provided from the website
    research_deps = ResearchDeps(research_topic=request.topic)
    loop = asyncio.get_running_loop()
    # Run your research agent blocking call in a separate thread
    result = await loop.run_in_executor(
        None,
        lambda: research_agent.run_sync(request.topic, deps=research_deps)
    )
    # Return the generated research (assumed to be Markdown formatted)
    return JSONResponse({"result": result.data})
