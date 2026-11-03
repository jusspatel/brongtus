"""Data schemas for local-exa, conforming to Exa-compatible search specifications."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """Represents a single curated search result mimicking Exa."""
    id: str = Field(description="Unique identifier or URL of the result")
    url: str = Field(description="URL of the web page")
    title: str = Field(default="", description="Title of the web page")
    author: Optional[str] = Field(default=None, description="Author if available")
    published_date: Optional[str] = Field(default=None, description="Publication date if available")
    score: Optional[float] = Field(default=None, description="Relevance score (0.0 to 1.0)")
    text: Optional[str] = Field(default=None, description="Cleaned, compressed content or excerpt")
    highlights: List[str] = Field(default_factory=list, description="Curated high-relevance highlight snippets")
    summary: Optional[str] = Field(default=None, description="Concise LLM-generated summary")
    raw_length: Optional[int] = Field(default=None, description="Original raw character length before compression")
    compressed_length: Optional[int] = Field(default=None, description="Compressed character length")


class SearchResponse(BaseModel):
    """Complete search response mimicking Exa's API response structure."""
    query: str = Field(description="Original user query")
    resolved_queries: List[str] = Field(default_factory=list, description="Expanded sub-queries used in search")
    results: List[SearchResult] = Field(default_factory=list, description="List of ranked, curated results")
    community_insights: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Fact-checked Reddit community discussions and consensus (unlocked when Gemini is enabled)",
    )
    token_savings_percent: Optional[float] = Field(default=None, description="Percentage of tokens pruned (e.g. 88.5%)")
    cost_usd: float = Field(default=0.0, description="Estimated API cost ($0 if local)")


class ScrapedDocument(BaseModel):
    """Intermediate model for raw scraped pages before curation."""
    url: str
    title: str = ""
    raw_text: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
