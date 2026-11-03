"""Brongtus - Free, local, token-pruned, intelligent web search engine mimicking Exa."""

from .client import Brongtus, LocalExa
from .schemas import SearchResult, SearchResponse
from .compressor import TokenCompressor
from .scraper import WebScraper
from .llm import LLMClient
from .graph import create_search_graph
from .router import AgenticTemporalRouter

__version__ = "0.1.0"
__all__ = [
    "Brongtus",
    "LocalExa",
    "SearchResult",
    "SearchResponse",
    "TokenCompressor",
    "WebScraper",
    "LLMClient",
    "AgenticTemporalRouter",
    "create_search_graph",
]
