"""Unified LLM interface supporting local models (Ollama/Llama.cpp), Gemini API, or generic LangChain models."""

import os
import json
import re
import warnings
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

# Suppress harmless sampling warnings from langchain_google_genai
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_google_genai")


def _load_env_file():
    """Automatically load .env file if present in the workspace or parents."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    if os.path.exists(".env"):
        try:
            with open(".env", "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass


_load_env_file()


class LLMClient:
    """Manages model calls for query expansion, relevance grading, and Exa-style highlight curation."""

    def __init__(
        self,
        local_model: Optional[str] = "qwen2.5:7b",
        gemini_api_key: Optional[str] = None,
        gemini_model: Optional[str] = None,
        custom_llm: Optional[Any] = None,
        ollama_base_url: str = "http://localhost:11434",
    ):
        _load_env_file()
        self.local_model = local_model
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.gemini_model = gemini_model or os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
        self.custom_llm = custom_llm
        self.ollama_base_url = ollama_base_url
        self._gemini_chat = None
        self._local_chat = None

        self._init_models()

    def _init_models(self):
        # Initialize Gemini if key available
        if self.gemini_api_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                kwargs = {
                    "model": self.gemini_model,
                    "google_api_key": self.gemini_api_key,
                }
                # Models with fixed sampling defaults reject custom temperature
                if "3.6-flash" not in self.gemini_model:
                    kwargs["temperature"] = 0.2
                self._gemini_chat = ChatGoogleGenerativeAI(**kwargs)
            except Exception as e:
                print(f"[Warning] Found GEMINI_API_KEY but failed to initialize Gemini model '{self.gemini_model}': {e}")

        # Initialize local LLM
        if self.custom_llm:
            self._local_chat = self.custom_llm
        elif self.local_model:
            try:
                from langchain_ollama import ChatOllama
                self._local_chat = ChatOllama(
                    model=self.local_model,
                    base_url=self.ollama_base_url,
                    temperature=0.2,
                )
            except Exception:
                pass

    @property
    def has_gemini(self) -> bool:
        return self._gemini_chat is not None

    async def _invoke(self, prompt: str, prefer_gemini: bool = False) -> str:
        """Invoke either Gemini or local LLM, with fallback."""
        model = None
        if prefer_gemini and self._gemini_chat:
            model = self._gemini_chat
        elif self._local_chat:
            model = self._local_chat
        elif self._gemini_chat:
            model = self._gemini_chat

        if model is not None:
            try:
                response = await model.ainvoke(prompt)
                content = getattr(response, "content", str(response))
                if isinstance(content, list):
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict) and "text" in block:
                            text_parts.append(block["text"])
                        elif hasattr(block, "text"):
                            text_parts.append(block.text)
                        elif isinstance(block, str):
                            text_parts.append(block)
                    return "\n".join(text_parts)
                return str(content)
            except Exception:
                pass

        # Offline / heuristic fallback if neither is currently running
        return ""

    async def expand_queries(self, original_query: str, max_queries: int = 3) -> List[str]:
        """Generate targeted sub-queries for fanout search."""
        prompt = (
            f"You are a search query planner for a web research engine.\n"
            f"Given the user request: '{original_query}'\n"
            f"Generate {max_queries} diverse, highly effective keyword-based search engine queries "
            f"that explore different angles (e.g. facts, technical docs, latest news).\n"
            f"Output ONLY a raw JSON array of strings, e.g. [\"query 1\", \"query 2\"]. Do not wrap in markdown quotes."
        )
        resp = await self._invoke(prompt, prefer_gemini=False)
        queries = self._parse_json_list(resp)
        if not queries:
            queries = [original_query]
        return queries[:max_queries]

    async def curate_result(
        self,
        query: str,
        title: str,
        url: str,
        compressed_text: str,
        num_highlights: int = 3,
    ) -> Dict[str, Any]:
        """Extract Exa-like highlights and a concise summary from compressed text."""
        # Use Gemini if available for sharper curation & de-junking
        prefer_gemini = self.has_gemini

        prompt = (
            f"User Query: {query}\n"
            f"Web Page Title: {title}\n"
            f"URL: {url}\n\n"
            f"Web Content Passages (pre-pruned):\n{compressed_text}\n\n"
            f"Instructions:\n"
            f"1. Extract {num_highlights} verbatim highlight snippets (1-2 sentences each) that directly answer the query.\n"
            f"2. Write a concise 2-sentence summary free of marketing fluff and junk.\n"
            f"3. Score the relevance of this page to the query from 0.0 to 1.0.\n"
            f"Output in JSON format with keys: 'highlights' (list of strings), 'summary' (string), 'score' (float).\n"
            f"Return ONLY valid JSON."
        )

        resp = await self._invoke(prompt, prefer_gemini=prefer_gemini)
        data = self._parse_json_dict(resp)

        if not data or "highlights" not in data:
            # Heuristic extraction fallback
            sentences = [s.strip() for s in re.split(r"[.!?]\s+", compressed_text) if len(s.strip()) > 30]
            highlights = sentences[:num_highlights]
            summary = " ".join(sentences[:2]) if sentences else "No summary available."
            return {
                "highlights": highlights,
                "summary": summary,
                "score": 0.8 if highlights else 0.2,
            }

        return data

    async def fact_check_reddit_insights(
        self,
        query: str,
        verified_passages: List[str],
        reddit_docs: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Cross-check Reddit community discussions against verified web sources to filter rumors.
        
        Only executed when Gemini is enabled, avoiding hallucinations from small local models.
        """
        if not self.has_gemini:
            return None

        verified_text_block = "\n---\n".join([v[:400] for v in verified_passages[:3]])

        prompt = (
            f"You are a rigorous research fact-checker.\n"
            f"Topic/Query: '{query}'\n\n"
            f"--- VERIFIED FACTUAL WEB SOURCES (News, Official Docs, Academic Papers) ---\n"
            f"{verified_text_block}\n\n"
            f"Instructions:\n"
            f"1. Search and analyze actual developer and user discussions on Reddit regarding '{query}'.\n"
            f"2. Summarize the authentic community sentiment or consensus from Reddit.\n"
            f"3. Fact-check the Reddit discussions against verified technical realities:\n"
            f"   - verified_takeaways: Community observations, real-world experiences, and nuances that are factually verified.\n"
            f"   - debunked_or_unverified: Baseless rumors, outdated information, or misconceptions found in community chatter.\n"
            f"4. Return ONLY a valid JSON object with keys:\n"
            f"   'community_sentiment' (string),\n"
            f"   'verified_takeaways' (list of strings),\n"
            f"   'debunked_or_unverified' (list of strings).\n"
            f"Do NOT include any markdown code fences around the JSON."
        )

        # Attempt search-grounded Gemini query for live Reddit threads
        try:
            import asyncio
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.gemini_api_key)
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=self.gemini_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())]
                )
            )
            result = self._parse_json_dict(response.text)
            if result and ("community_sentiment" in result or "verified_takeaways" in result):
                return result
        except Exception:
            pass

        # Fallback to standard model invocation
        resp = await self._invoke(prompt, prefer_gemini=True)
        result = self._parse_json_dict(resp)
        return result if result else None

    def _parse_json_list(self, text: str) -> List[str]:
        try:
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception:
            pass
        return []

    def _parse_json_dict(self, text: str) -> Dict[str, Any]:
        try:
            cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
            cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception:
            pass
        return {}
