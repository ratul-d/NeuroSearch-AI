from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.models.groq import GroqModel
from dotenv import load_dotenv
from dataclasses import dataclass,field
from typing import List, Dict
from litellm import completion
from tavily import TavilyClient
import json
import litellm
import os


load_dotenv()

tavily_client = TavilyClient()
litellm.set_verbose=False

MAX_WEB_SEARCH_LOOPS=1

#SYSTEM PROMPTS
query_writer_system_prompt = """Your goal is to generate targeted web search query.
The query will gather information related to a specific topic.

Topic:
{research_topic}

Return your query as a JSON object:
{{
    "query": "string"
    "aspect": "string"
    "rationale": "string"
}}
"""
summarizer_system_prompt="""Your goal is to generate a high—quality summary of the web search results.

When EXTENDING an existing summary:
1. Seamlessly integrate new information without repeating what's already covered
2. Maintain consistency lth the existing content's style and depth
3. Only add new, non—redundant information
4. Ensure smooth transitions between existing and new content

When creating a NEW summary:
1. Highlight the most relevant information from each source
2. Provide a concise overview of the key points related to the report topic
3. Emphasize significant findings or insights
4. Ensure a coherent flow of information

In both cases:
- Focus on factual, objective information
- Maintain a consistent technical depth
- Avoid redundancy and repetition
- DO NOT use phrases like "based on the new results" or "according to additional sources"
- DO NOT add a preamble like "Here is an extended summary..." Just directly output the summary.
- DO NOT add a References or Works Cited section.
"""

reflection_system_prompt ="""You are an expert research assistant analyzing a summary about {research_topic}.

Your tasks:
1. Identify knowledge gaps or areas that need deeper exploration
2. Generate a follow—up question that would help expand your understanding
3. Focus on technical details, implementation specifics, or emerging trends that weren't fully covered

Ensure the follow—up question is self—contained and includes necessary context for web search.

Return your analysis as a JSON object:
{{
    "knowledge_gap": "string",
    "follow_up_query": "string"
}}"""

def format_sources(sources):

    #foramting list of dictionaries into structured text for LLM input

    formatted_text = "Sources:\n\n"
    for i, source in enumerate(sources, start=1):
        formatted_text += (
            f"**Source {i}:**<br>"  # Using <br> for line break in HTML
            f"**Title:** {source['title']}<br>"
            f"**Url:** <a href='{source['url']}' target='_blank'>{source['url']}</a><br><br>"
        )
    return formatted_text.strip()

@dataclass
class ResearchDeps:
    research_topic: str = None
    search_query: str = None
    current_summary: str = None
    final_summary: str = None
    sources: List[str] = field(default_factory=list)
    latest_web_search_result: str = None
    research_loop_count: int = 0

#TOOLS

async def generate_search_query(ctx: RunContext[ResearchDeps]) -> str:
    """ Generate a query for web search """
    print("===CALLING generate_search_query... ===")
    response = completion(
        model="groq/llama-3.3-70b-versatile",
        messages=[
            {"content": query_writer_system_prompt.format(research_topic=ctx.deps.research_topic),"role":"system"},
            {"content": "Provide the search query based on the above instructions.", "role": "user"}
        ],
        max_tokens=500,
        response_format = { "type": "json_object" }
    )
    search_query = json.loads(response.choices[0].message.content)
    # print(f'===>search_query:{search_query}"
    ctx.deps.search_query = search_query["query"]
    return "perform web_search"

async def perform_web_search(ctx: RunContext[ResearchDeps]) -> str:
    """ Do search and collect information"""
    print("===CALLING perform_web_search... ===")
    search_results = tavily_client.search(ctx.deps.search_query,include_raw_content=False, max_results=3)
    search_string = format_sources(search_results["results"])
    ctx.deps.sources.extend(search_results["results"])
    ctx.deps.latest_web_search_result = search_string
    ctx.deps.research_loop_count += 1
    return "summarize_sources"

async def summarize_sources(ctx: RunContext[ResearchDeps]) -> str:
    """ Summarize gathered resources """
    print("===CALLING summarize_sources... ===")
    current_summary = ctx.deps.current_summary
    most_recent_web_search = ctx.deps.latest_web_search_result
    if current_summary:
        user_prompt = (f"Extend the existing: {current_summary}\n\n"
                       f"Include new search results: {most_recent_web_search} "
                       f"That address the following topic: {ctx.deps.research_topic}.\n\n"
                       "Return your summary as a valid JSON object with a 'summary' field."
                       )
    else:
        user_prompt = (f"Generate a very long summary of these search results: {most_recent_web_search} "
                       f"That address the following topic: {ctx.deps.research_topic}.\n\n"
                       "Return your summary as a valid JSON object with a 'summary' field."
                       )

    response = completion(
        model="groq/llama-3.3-70b-versatile",
        messages=[
            {"content": summarizer_system_prompt.format(research_topic=ctx.deps.research_topic), "role": "system"},
            {"content": user_prompt,"role": "user"}
        ],
        max_tokens=1000,
        response_format={"type": "json_object"}
    )
    ctx.deps.current_summary = response.choices[0].message.content
    return "reflect_on_summary"

async def reflect_on_summary(ctx: RunContext[ResearchDeps]) -> str:
    """ Reflect on the summary and generate a follow-up query """
    print("===CALLING reflect_on_summary... ===\n\n")
    user_message = (
        f"Here is the current summary of our research:\n\n{ctx.deps.current_summary}\n\n"
        "Based on this summary, identify any knowledge gaps and generate a follow-up web search query to fill in those gaps."
    )
    response = completion(
        model="groq/llama-3.3-70b-versatile",
        messages=[
            {"content": reflection_system_prompt.format(research_topic=ctx.deps.research_topic), "role": "system"},
            {"content": user_message, "role": "user"}
        ],
        max_tokens=500,
        response_format={"type": "json_object"}
    )
    follow_up_query = json.loads(response.choices[0].message.content)
    ctx.deps.search_query = follow_up_query["follow_up_query"]
    return "continue_or_stop_research"

# async def finalize_summary(ctx: RunContext[ResearchDeps]) -> str:
#     """ Finalize the summary """
#     print("===CALLING finalize_summary... ===")
#     all_sources = format_sources(ctx.deps.sources)
#     ctx.deps.final_summary = f"## Summary:\n\n{ctx.deps.current_summary}\n\n{all_sources}"
#     return f"STOP and output this summary as it is to user: {ctx.deps.final_summary}"

async def finalize_summary(ctx: RunContext[ResearchDeps]) -> str:
    """ Finalize the summary with clean formatting """
    # Try to convert the current_summary JSON string into a Python dict,
    # and extract the summary text.
    print("===CALLING finalize_summary... ===\n\n")
    try:
        summary_obj = json.loads(ctx.deps.current_summary)
        summary_text = summary_obj.get("summary", ctx.deps.current_summary)
    except Exception:
        summary_text = ctx.deps.current_summary

    # Format the sources with an extra header for clarity
    all_sources = format_sources(ctx.deps.sources)
    final_output = f"## Summary:\n\n{summary_text}\n\n## Sources:\n\n{all_sources}"
    return f"STOP and output this summary as it is to user: {final_output}"


async def continue_or_stop_research(ctx: RunContext[ResearchDeps]) -> str:
    """ Decide to continue the research or stop based on the follow-up query"""
    print("===CALLING continue_or_stop_research... ===")
    if ctx.deps.research_loop_count >= MAX_WEB_SEARCH_LOOPS:

        return "finalize_summary"
    else:
        return f"Iterations so far: {ctx.deps.research_loop_count}.\n\nperform_web_search"


model = GroqModel('meta-llama/llama-4-maverick-17b-128e-instruct')
#model = GroqModel('llama-3.3-70b-versatile')
default_system_prompt = """ You are a researcher. You need to use your tools and provide a research. Order of tools to call:
generate_search_query, perform_web_search, summarize_sources, reflect_on_summary, continue_or_stop_research, 
perform_web_search, summarize_sources, reflect_on_summary, continue_or_stop_research, finalize_summary.

"""
research_agent = Agent(model,system_prompt=default_system_prompt,
                       deps_type=ResearchDeps, tools=[Tool(generate_search_query),Tool(perform_web_search),Tool(summarize_sources),
                       Tool(reflect_on_summary),Tool(finalize_summary),Tool(continue_or_stop_research)])

if __name__ == "__main__":
    topic = 'Neural Style Transfer'
    research_deps = ResearchDeps(research_topic=topic)
    result = research_agent.run_sync(topic, deps=research_deps)
    print(result.data)