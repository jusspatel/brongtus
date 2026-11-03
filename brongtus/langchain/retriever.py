"""LangChain BaseRetriever implementation for Brongtus."""

from typing import List, Optional, Any
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from ..client import Brongtus


class BrongtusRetriever(BaseRetriever):
    """Retriever for integrating token-pruned local web search directly into RAG chains."""

    client: Brongtus
    k: int = 5

    def __init__(
        self,
        k: int = 5,
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
        super().__init__(client=client, k=k, **kwargs)

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        response = self.client.search(query=query, num_results=self.k)
        docs = []
        for r in response.results:
            text_content = r.text or r.summary or ""
            metadata = {
                "source": r.url,
                "title": r.title,
                "score": r.score,
                "highlights": r.highlights,
            }
            docs.append(Document(page_content=text_content, metadata=metadata))
        return docs


# Backward-compatibility alias
LocalExaRetriever = BrongtusRetriever
