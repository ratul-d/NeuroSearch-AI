import asyncio
import arxiv

async def run_arxiv_search(query: str, num_results: int):
    def sync_search():
        search = arxiv.Search(
            query=query,
            max_results=num_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        return list(search.results())
    return await asyncio.to_thread(sync_search)

