"""LangChain BaseTool implementation for Brongtus."""

from typing import Type, Optional, Any, List
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from ..client import Brongtus
from ..schemas import SearchResponse


class BrongtusSearchInput(BaseModel):
    """Input parameters for Brongtus search tool."""
    query: str = Field(description="The research or search query to look up on the web.")
    num_results: int = Field(default=5, description="Number of high-quality results to return (default 5).")


class BrongtusSearchResults(BaseTool):
    """Exa-like local search tool for LangChain agents and LangGraph workflows."""

    name: str = "brongtus_search"
    description: str = (
        "An intelligent, token-pruned local search engine. "
        "Searches the web, renders dynamic pages with Playwright, removes up to 90% boilerplate junk, "
        "and curates factual highlights and summaries."
    )
    args_schema: Type[BaseModel] = BrongtusSearchInput
    client: Brongtus = Field(default_factory=Brongtus)

    def __init__(
        self,
        local_model: Optional[str] = "qwen2.5:7b",
        gemini_api_key: Optional[str] = None,
        gemini_model: str = "gemini-2.5-flash",
        custom_llm: Optional[Any] = None,
        enable_compression: bool = True,
        headless: bool = True,
        **kwargs,
    ):
        client = Brongtus(
            local_model=local_model,
            gemini_api_key=gemini_api_key,
            gemini_model=gemini_model,
            custom_llm=custom_llm,
            enable_compression=enable_compression,
            headless=headless,
        )
        super().__init__(client=client, **kwargs)

    def _format_response(self, response: SearchResponse) -> str:
        """Format the search response into markdown for the agent."""
        blocks = [f"### Brongtus Search Results for: '{response.query}'"]
        if response.token_savings_percent:
            blocks.append(f"*(Cleaned & compressed context: saved {response.token_savings_percent}% tokens)*\n")

        for i, res in enumerate(response.results, 1):
            block = [f"#### {i}. [{res.title}]({res.url})"]
            if res.score is not None:
                block.append(f"**Relevance Score:** {res.score:.2f}")
            if res.summary:
                block.append(f"**Summary:** {res.summary}")
            if res.highlights:
                block.append("**Key Highlights:**")
                for h in res.highlights:
                    block.append(f"- {h}")
            blocks.append("\n".join(block))

        # Include fact-checked Reddit community insights if unlocked
        if response.community_insights:
            ci = response.community_insights
            c_block = ["### 💬 Fact-Checked Community Insights (Reddit + Cross-Verified)"]
            if ci.get("community_sentiment"):
                c_block.append(f"**Community Consensus:** {ci['community_sentiment']}")
            if ci.get("verified_takeaways"):
                c_block.append("\n**Verified Community Takeaways:**")
                for item in ci["verified_takeaways"]:
                    c_block.append(f"- ✅ {item}")
            if ci.get("debunked_or_unverified"):
                c_block.append("\n**Debunked / Unverified Rumors:**")
                for item in ci["debunked_or_unverified"]:
                    c_block.append(f"- ⚠️ {item}")
            blocks.append("\n".join(c_block))

        return "\n\n".join(blocks)

    def _run(self, query: str, num_results: int = 5, **kwargs) -> str:
        res = self.client.search(query=query, num_results=num_results)
        return self._format_response(res)

    async def _arun(self, query: str, num_results: int = 5, **kwargs) -> str:
        res = await self.client.asearch(query=query, num_results=num_results)
        return self._format_response(res)


# Backward-compatibility aliases
LocalExaSearchResults = BrongtusSearchResults
LocalExaSearchInput = BrongtusSearchInput
