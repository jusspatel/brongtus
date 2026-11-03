"""Tests for Brongtus package."""

import pytest
from brongtus import Brongtus, TokenCompressor, SearchResult, SearchResponse
from brongtus.langchain import BrongtusSearchResults, BrongtusRetriever


def test_token_compressor_algorithmic():
    compressor = TokenCompressor(target_max_words=60)
    sample_text = (
        "Solid state batteries are high-energy density storage devices that replace liquid electrolytes with solid electrodes.\n\n"
        "They offer higher safety, higher energy density, and faster charging times compared to traditional lithium-ion batteries.\n\n"
        "In 2025, commercialization efforts advanced significantly with pilot production lines established by several automakers.\n\n"
        + "Filler noise paragraph that is completely irrelevant to the actual search query. " * 30
    )

    query = "solid state battery advantages and commercialization"
    compressed, savings = compressor.compress(query, sample_text, max_words=50)

    assert "Solid state batteries" in compressed or "higher safety" in compressed
    # Algorithmic BM25 density selection should achieve significant token reduction
    assert savings > 40.0


def test_schema_serialization():
    res = SearchResult(
        id="https://example.com",
        url="https://example.com",
        title="Example Title",
        score=0.95,
        highlights=["Key sentence 1.", "Key sentence 2."],
        summary="A short summary.",
        raw_length=5000,
        compressed_length=450,
    )
    assert res.score == 0.95
    assert len(res.highlights) == 2

    community_data = {
        "community_sentiment": "Positive skepticism about commercial timelines.",
        "verified_takeaways": ["Prototypes exist and have completed 1000 cycles."],
        "debunked_or_unverified": ["Claims of cars rolling off assembly lines next month are unverified."],
        "reddit_sources": ["https://reddit.com/r/technology/sample"],
    }

    resp = SearchResponse(
        query="test query",
        resolved_queries=["test query sub 1"],
        results=[res],
        community_insights=community_data,
        token_savings_percent=91.0,
    )
    assert resp.token_savings_percent == 91.0
    assert resp.community_insights is not None
    assert "Positive skepticism" in resp.community_insights["community_sentiment"]
    assert len(resp.results) == 1


def test_langchain_tool_instantiation():
    tool = BrongtusSearchResults(
        local_model="qwen2.5:7b",
        headless=True,
    )
    assert tool.name == "brongtus_search"
    assert "local search engine" in tool.description.lower()


def test_langchain_retriever_instantiation():
    retriever = BrongtusRetriever(
        k=3,
        local_model="qwen2.5:7b",
        headless=True,
    )
    assert retriever.k == 3
