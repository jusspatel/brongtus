"""High-level Python SDK client mimicking the official Exa client interface."""

import asyncio
from typing import Optional, List, Dict, Any, Union

from .schemas import SearchResponse, SearchResult
from .scraper import WebScraper
from .compressor import TokenCompressor
from .llm import LLMClient
from .graph import create_search_graph


class Brongtus:
    """The main client for executing local, intelligent, token-pruned web searches."""

    def __init__(
        self,
        local_model: Optional[str] = "qwen2.5:7b",
        gemini_api_key: Optional[str] = None,
        gemini_model: str = "gemini-2.5-flash",
        custom_llm: Optional[Any] = None,
        enable_compression: bool = True,
        target_compression_words: int = 500,
        headless: bool = True,
        enable_hitl: bool = False,
    ):
        self.scraper = WebScraper(headless=headless)
        self.compressor = TokenCompressor(target_max_words=target_compression_words)
        self.llm = LLMClient(
            local_model=local_model,
            gemini_api_key=gemini_api_key,
            gemini_model=gemini_model,
            custom_llm=custom_llm,
        )
        self.enable_compression = enable_compression
        self.enable_hitl = enable_hitl
        self._graph = create_search_graph(self.scraper, self.compressor, self.llm)

    async def asearch(
        self,
        query: str,
        num_results: int = 5,
        include_highlights: bool = True,
        include_text: bool = True,
    ) -> SearchResponse:
        """Asynchronously run intelligent local search."""
        initial_state = {
            "original_query": query,
            "num_results": num_results,
            "sub_queries": [],
            "candidate_urls": [],
            "reddit_candidate_urls": [],
            "scraped_docs": [],
            "reddit_docs": [],
            "final_results": [],
            "community_insights": None,
            "total_raw_chars": 0,
            "total_compressed_chars": 0,
            "token_savings_percent": 0.0,
            "enable_hitl": self.enable_hitl,
        }

        output_state = await self._graph.ainvoke(initial_state)

        results: List[SearchResult] = []
        for item in output_state.get("final_results", []):
            results.append(
                SearchResult(
                    id=item["id"],
                    url=item["url"],
                    title=item.get("title", ""),
                    score=item.get("score"),
                    text=item.get("text") if include_text else None,
                    highlights=item.get("highlights", []) if include_highlights else [],
                    summary=item.get("summary"),
                    raw_length=item.get("raw_length"),
                    compressed_length=item.get("compressed_length"),
                )
            )

        return SearchResponse(
            query=query,
            resolved_queries=output_state.get("sub_queries", [query]),
            results=results,
            community_insights=output_state.get("community_insights"),
            token_savings_percent=output_state.get("token_savings_percent", 0.0),
            cost_usd=0.0,  # Free local execution!
        )

    def search(
        self,
        query: str,
        num_results: int = 5,
        include_highlights: bool = True,
        include_text: bool = True,
    ) -> SearchResponse:
        """Synchronously run intelligent local search."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(
                self.asearch(
                    query=query,
                    num_results=num_results,
                    include_highlights=include_highlights,
                    include_text=include_text,
                )
            )
        else:
            return asyncio.run(
                self.asearch(
                    query=query,
                    num_results=num_results,
                    include_highlights=include_highlights,
                    include_text=include_text,
                )
            )


# Backward-compatibility alias
LocalExa = Brongtus

