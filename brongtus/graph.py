"""LangGraph stateful workflow with query planning, fanout search, token compression, and LLM curation."""

import asyncio
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt

from .schemas import SearchResult, SearchResponse, ScrapedDocument
from .scraper import WebScraper
from .compressor import TokenCompressor
from .llm import LLMClient


class SearchGraphState(TypedDict):
    """Internal graph state representing search lifecycle."""
    original_query: str
    num_results: int
    sub_queries: List[str]
    candidate_urls: List[Dict[str, str]]
    reddit_candidate_urls: List[Dict[str, str]]
    scraped_docs: List[Dict[str, Any]]
    reddit_docs: List[Dict[str, Any]]
    final_results: List[Dict[str, Any]]
    community_insights: Optional[Dict[str, Any]]
    total_raw_chars: int
    total_compressed_chars: int
    token_savings_percent: float
    enable_hitl: bool


def create_search_graph(
    scraper: WebScraper,
    compressor: TokenCompressor,
    llm: LLMClient,
):
    """Constructs the LangGraph stateful search graph."""

    # Node 1: Planner - Generate sub-queries
    async def planner_node(state: SearchGraphState):
        queries = await llm.expand_queries(state["original_query"], max_queries=3)
        return {"sub_queries": queries}

    # Node 2: Search Fanout - Run searches in parallel
    async def search_fanout_node(state: SearchGraphState):
        sub_queries = state.get("sub_queries", [state["original_query"]])
        search_tasks = [scraper.search(q, num_results=state.get("num_results", 5)) for q in sub_queries]
        
        # If Gemini is enabled, also harvest Reddit discussions for fact checking
        reddit_task = None
        if llm.has_gemini:
            reddit_task = scraper.search_reddit(state["original_query"], num_results=2)

        results_lists = await asyncio.gather(*search_tasks)
        reddit_candidates = await reddit_task if reddit_task else []

        # Flatten & deduplicate by URL
        seen_urls = set()
        candidates = []
        for r_list in results_lists:
            for item in r_list:
                url = item.get("url")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    candidates.append(item)

        target_count = state.get("num_results", 5)
        return {
            "candidate_urls": candidates[:target_count],
            "reddit_candidate_urls": reddit_candidates,
        }

    # Node 3: Optional Human-in-the-Loop review
    def hitl_review_node(state: SearchGraphState):
        if state.get("enable_hitl", False):
            # Interrupt and wait for user approval or URL modifications
            approved = interrupt({
                "message": "Review and edit candidate URLs before deep scraping",
                "candidate_urls": state["candidate_urls"],
            })
            if isinstance(approved, list):
                return {"candidate_urls": approved}
        return {}

    # Node 4: Scrape Worker - Dynamic JS scraping via Playwright
    async def scrape_node(state: SearchGraphState):
        urls = [c["url"] for c in state.get("candidate_urls", []) if "url" in c]
        docs = await scraper.scrape_urls_concurrently(urls)
        doc_dicts = [
            {"url": d.url, "title": d.title, "raw_text": d.raw_text}
            for d in docs
        ]

        # Scrape Reddit discussions concurrently if available
        reddit_urls = [r["url"] for r in state.get("reddit_candidate_urls", []) if "url" in r]
        reddit_docs_dicts = []
        if reddit_urls:
            r_docs = await scraper.scrape_urls_concurrently(reddit_urls)
            reddit_docs_dicts = [
                {"url": d.url, "title": d.title, "raw_text": d.raw_text}
                for d in r_docs
            ]

        return {"scraped_docs": doc_dicts, "reddit_docs": reddit_docs_dicts}

    # Node 5: Token Pruning & Compression (Cuts up to 90% tokens)
    def compress_node(state: SearchGraphState):
        compressed_items = []
        total_raw = 0
        total_comp = 0

        for doc in state.get("scraped_docs", []):
            raw = doc["raw_text"]
            raw_len = len(raw)
            total_raw += raw_len

            comp_text, savings = compressor.compress(state["original_query"], raw)
            comp_len = len(comp_text)
            total_comp += comp_len

            compressed_items.append({
                "url": doc["url"],
                "title": doc["title"],
                "compressed_text": comp_text,
                "raw_length": raw_len,
                "compressed_length": comp_len,
            })

        # Also compress Reddit docs
        compressed_reddit = []
        for r_doc in state.get("reddit_docs", []):
            r_raw = r_doc["raw_text"]
            r_comp, _ = compressor.compress(state["original_query"], r_raw, max_words=300)
            compressed_reddit.append({
                "url": r_doc["url"],
                "title": r_doc["title"],
                "compressed_text": r_comp,
            })

        overall_savings = (
            max(0.0, (1.0 - (total_comp / max(1, total_raw)))) * 100.0
            if total_raw > 0 else 0.0
        )

        return {
            "scraped_docs": compressed_items,
            "reddit_docs": compressed_reddit,
            "total_raw_chars": total_raw,
            "total_compressed_chars": total_comp,
            "token_savings_percent": round(overall_savings, 1),
        }

    # Node 6: LLM Curation (Gemini or Local LLM)
    async def curate_node(state: SearchGraphState):
        curation_tasks = []
        for item in state.get("scraped_docs", []):
            curation_tasks.append(
                llm.curate_result(
                    query=state["original_query"],
                    title=item.get("title", ""),
                    url=item["url"],
                    compressed_text=item.get("compressed_text", ""),
                )
            )

        # Fact check Reddit discussions if Gemini is active
        reddit_check_task = None
        if llm.has_gemini and state.get("reddit_docs"):
            verified_snippets = [doc.get("compressed_text", "") for doc in state.get("scraped_docs", [])]
            reddit_check_task = llm.fact_check_reddit_insights(
                query=state["original_query"],
                verified_passages=verified_snippets,
                reddit_docs=state.get("reddit_docs", []),
            )

        curated_outputs = await asyncio.gather(*curation_tasks)
        community_insights = await reddit_check_task if reddit_check_task else None

        final_results = []
        for item, curation in zip(state.get("scraped_docs", []), curated_outputs):
            final_results.append({
                "id": item["url"],
                "url": item["url"],
                "title": item.get("title", ""),
                "score": curation.get("score", 0.7),
                "text": item.get("compressed_text", ""),
                "highlights": curation.get("highlights", []),
                "summary": curation.get("summary", ""),
                "raw_length": item.get("raw_length", 0),
                "compressed_length": item.get("compressed_length", 0),
            })

        # Sort by relevance score
        final_results.sort(key=lambda x: x.get("score") or 0.0, reverse=True)
        return {
            "final_results": final_results,
            "community_insights": community_insights,
            "token_savings_percent": state.get("token_savings_percent", 0.0),
            "total_raw_chars": state.get("total_raw_chars", 0),
            "total_compressed_chars": state.get("total_compressed_chars", 0),
        }

    # Build graph
    builder = StateGraph(SearchGraphState)

    builder.add_node("planner", planner_node)
    builder.add_node("search_fanout", search_fanout_node)
    builder.add_node("hitl_review", hitl_review_node)
    builder.add_node("scraper", scrape_node)
    builder.add_node("compressor", compress_node)
    builder.add_node("curator", curate_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "search_fanout")
    builder.add_edge("search_fanout", "hitl_review")
    builder.add_edge("hitl_review", "scraper")
    builder.add_edge("scraper", "compressor")
    builder.add_edge("compressor", "curator")
    builder.add_edge("curator", END)

    return builder.compile()
