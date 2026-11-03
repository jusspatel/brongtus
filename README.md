# 🌐 Brongtus

[![PyPI version](https://img.shields.io/pypi/v/brongtus.svg)](https://pypi.org/project/brongtus/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> **A free, local, intelligent search tool mimicking Exa with Playwright, LangGraph, up to 90% algorithmic token pruning, and optional Gemini enhancement with Reddit community fact-checking.**

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Installation](#-installation)
- [Full API Reference](#-full-api-reference)
  - [1. `Brongtus` Client](#1-brongtus-client)
  - [2. Return Schemas (`SearchResponse` & `SearchResult`)](#2-return-schemas-searchresponse--searchresult)
  - [3. LangChain Tool (`BrongtusSearchResults`)](#3-langchain-tool-brongtussearchresults)
  - [4. LangChain Retriever (`BrongtusRetriever`)](#4-langchain-retriever-brongtusretriever)
  - [5. Standalone Token Compressor (`TokenCompressor`)](#5-standalone-token-compressor-tokencompressor)
  - [6. Standalone Web Scraper (`WebScraper`)](#6-standalone-web-scraper-webscraper)
- [Code Examples](#-code-examples)
  - [Example 1: Pure Local Search (100% Free)](#example-1-pure-local-search-100-free)
  - [Example 2: Hybrid Mode (Gemini + Reddit Fact-Checking)](#example-2-hybrid-mode-gemini--reddit-fact-checking)
  - [Example 3: LangChain AI Agent Integration](#example-3-langchain-ai-agent-integration)
  - [Example 4: RAG Pipeline with LangChain](#example-4-rag-pipeline-with-langchain)
  - [Example 5: Standalone Context Compressor](#example-5-standalone-context-compressor)
- [Interactive CLI](#-interactive-cli)
- [Publishing to PyPI](#-publishing-to-pypi)
- [License](#-license)

---

## ⚡ Overview

Tools like [Exa](https://exa.ai) and Tavily provide clean search APIs for AI agents, but they require recurring paid subscriptions and closed-source proprietary backends.

**Brongtus** delivers an Exa-like search engine running directly on your own infrastructure:
- 🆓 **100% Free & Open-Source:** No mandatory API subscriptions.
- 🎭 **Playwright Powered:** Renders modern JavaScript, SPAs, and dynamic web pages.
- ✂️ **Up to 90% Algorithmic Token Pruning:** Uses `justext`, `trafilatura`, and BM25 Okapi passage ranking to strip boilerplates, ads, and navigation noise without hardcoded heuristic lists.
- 🧠 **Hybrid Intelligence (Local + Gemini):**
  - **Local Mode**: Runs offline via Ollama (`qwen2.5`, `llama3.2`, `deepseek-r1`) or built-in heuristic extractors.
  - **Gemini Hybrid Mode**: Add your `GEMINI_API_KEY` (e.g., `gemini-3.6-flash`, `gemini-2.5-flash`, `gemini-1.5-pro`) to unlock ultra-sharp curation and **Reddit community discussion cross-verification** (filtering out rumors while capturing verified community consensus).
- 🦜 **First-Class LangChain Integration:** Natively implements `BaseTool` and `BaseRetriever`.

---

## 📦 Installation

```bash
pip install brongtus
playwright install chromium
```

To enable Gemini enhancement and local LLM drivers:
```bash
pip install "brongtus[gemini,local-llm]"
```

---

## 📚 Full API Reference

### 1. `Brongtus` Client

The main search client class (`from brongtus import Brongtus`).

#### Constructor: `Brongtus(...)`

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `local_model` | `str` | `"qwen2.5:7b"` | Name of the local Ollama model to use for query planning and local curation. |
| `gemini_api_key` | `str` | `None` (or `os.getenv("GEMINI_API_KEY")`) | Google Gemini API key. Unlocks Gemini curation and Reddit fact-checking. Automatically reads from `.env` if present. |
| `gemini_model` | `str` | `"gemini-2.5-flash"` (or `os.getenv("GEMINI_MODEL")`) | Gemini model identifier (e.g. `gemini-3.6-flash`, `gemini-2.5-flash`, `gemini-1.5-pro`). |
| `custom_llm` | `BaseChatModel` | `None` | Any custom LangChain chat model instance to use as the engine. |
| `enable_compression` | `bool` | `True` | Whether to run algorithmic context pruning to cut 80–90% of boilerplate tokens. |
| `target_compression_words` | `int` | `500` | Target word budget per page for compressed passages. |
| `headless` | `bool` | `True` | Whether to launch Playwright in headless browser mode. |
| `enable_hitl` | `bool` | `False` | Enable LangGraph Human-in-the-Loop review gate for candidate URLs. |

#### Methods:

##### `search(query, num_results=5, include_highlights=True, include_text=True) -> SearchResponse`
Synchronous search execution.

##### `asearch(query, num_results=5, include_highlights=True, include_text=True) -> SearchResponse`
Asynchronous search execution (`await engine.asearch(...)`).

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `query` | `str` | *Required* | The natural language topic or search query. |
| `num_results` | `int` | `5` | Maximum number of ranked results to return. |
| `include_highlights` | `bool` | `True` | Whether to extract Exa-like verbatim highlights. |
| `include_text` | `bool` | `True` | Whether to include clean compressed text in each result. |

---

### 2. Return Schemas (`SearchResponse` & `SearchResult`)

#### `SearchResponse` Attributes:
- `query` (`str`): The original user search query.
- `resolved_queries` (`List[str]`): Expanded sub-queries executed across the search fanout.
- `results` (`List[SearchResult]`): List of ranked, curated search results.
- `community_insights` (`Optional[Dict[str, Any]]`): Fact-checked Reddit insights (present when Gemini is active).
  - `community_sentiment` (`str`): Summary of community consensus.
  - `verified_takeaways` (`List[str]`): Real-world takeaways verified against technical literature.
  - `debunked_or_unverified` (`List[str]`): Debunked misconceptions, rumors, and hype.
- `token_savings_percent` (`float`): Percentage of tokens pruned (typically 60%–92%).
- `cost_usd` (`float`): Estimated execution cost (`$0.0` for local runs).

#### `SearchResult` Attributes:
- `id` (`str`): Result identifier (matches URL).
- `url` (`str`): Web page URL.
- `title` (`str`): Web page title.
- `score` (`float`): Relevance score (0.0 to 1.0).
- `summary` (`str`): Concise factual summary.
- `highlights` (`List[str]`): Key verbatim highlight snippets.
- `text` (`str`): Compressed informative text passages.
- `raw_length` (`int`): Character length of page before compression.
- `compressed_length` (`int`): Character length of page after compression.

---

### 3. LangChain Tool (`BrongtusSearchResults`)

Inherits from LangChain's `BaseTool`. Can be passed directly to `ChatModel.bind_tools()` or any LangChain/LangGraph agent.

```python
from brongtus.langchain import BrongtusSearchResults

tool = BrongtusSearchResults(
    gemini_api_key="...",           # Optional, reads from .env automatically
    gemini_model="gemini-3.6-flash", # Optional
    num_results=5                   # Number of results per query
)

# Tool execution:
markdown_output = tool.invoke({"query": "Quantum computing commercial roadmap"})
```

---

### 4. LangChain Retriever (`BrongtusRetriever`)

Inherits from LangChain's `BaseRetriever`. Integrates local search directly into Retrieval-Augmented Generation (RAG) chains.

```python
from brongtus.langchain import BrongtusRetriever

retriever = BrongtusRetriever(k=3)
documents = retriever.invoke("Solid state battery energy density")

# Each Document contains:
# doc.page_content -> Compressed informative text
# doc.metadata -> {'source': url, 'title': title, 'score': score, 'highlights': [...]}
```

---

### 5. Standalone Token Compressor (`TokenCompressor`)

Can be used independently of web search to compress arbitrary documents, prompt contexts, or articles by 80–90%.

```python
from brongtus import TokenCompressor

compressor = TokenCompressor(target_max_words=250)
compressed_text, savings_pct = compressor.compress(
    query="target topic to keep",
    content=massive_raw_text
)
print(f"Compressed down by {savings_pct}%")
```

---

### 6. Standalone Web Scraper (`WebScraper`)

Can be used independently to perform free search and dynamic JS rendering:

```python
from brongtus import WebScraper

scraper = WebScraper(headless=True)

# Free web search
links = await scraper.search("quantum annealing breakthroughs", num_results=3)

# Dynamic JavaScript page rendering
doc = await scraper.scrape_url("https://news.mit.edu/topic/fusion")
print(doc.title, len(doc.raw_text))
```

---

## 💻 Code Examples

### Example 1: Pure Local Search (100% Free)

Runs completely offline without external APIs:

```python
from brongtus import Brongtus

engine = Brongtus(local_model="qwen2.5:7b")

response = engine.search("solid state battery commercialization", num_results=3)

print(f"Token savings: {response.token_savings_percent}%\n")
for r in response.results:
    print(f"[{r.score}] {r.title}")
    print(f"URL: {r.url}")
    print(f"Summary: {r.summary}")
    for h in r.highlights:
        print(f"  • {h}")
    print("-" * 50)
```

---

### Example 2: Hybrid Mode (Gemini + Reddit Fact-Checking)

Add your `GEMINI_API_KEY` (in `.env` or passed directly) to unlock ultra-smart curation and Reddit community cross-verification:

```python
import os
from brongtus import Brongtus

engine = Brongtus(
    gemini_api_key=os.getenv("GEMINI_API_KEY"),
    gemini_model="gemini-3.6-flash",  # Or gemini-2.5-flash / gemini-1.5-pro
    enable_compression=True
)

response = engine.search("lenovo legion series laptops battery life")

# Standard Web Results
for r in response.results:
    print(f"[{r.score}] {r.title} ({r.url})")
    print(f"Summary: {r.summary}\n")

# Fact-Checked Community Insights
if response.community_insights:
    ci = response.community_insights
    print("Community Consensus:\n", ci["community_sentiment"])
    print("\nVerified Takeaways:")
    for v in ci["verified_takeaways"]:
        print(f"  [+] {v}")
    print("\nDebunked Rumors / Myths:")
    for d in ci["debunked_or_unverified"]:
        print(f"  [!] {d}")
```

---

### Example 3: LangChain AI Agent Integration

Bind `BrongtusSearchResults` directly to tool-calling agents:

```python
from brongtus.langchain import BrongtusSearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Initialize tool
search_tool = BrongtusSearchResults()

# Bind tool to model
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash").bind_tools([search_tool])

# Run query
ai_msg = llm.invoke([HumanMessage(content="What are the key trade-offs of React Server Components in production?")])
print(ai_msg.tool_calls)
```

---

### Example 4: RAG Pipeline with LangChain

```python
from brongtus.langchain import BrongtusRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

retriever = BrongtusRetriever(k=3)
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

prompt = ChatPromptTemplate.from_template("""
Answer the question based ONLY on the following context:
{context}

Question: {question}
""")

chain = (
    {"context": retriever, "question": lambda x: x}
    | prompt
    | llm
    | StrOutputParser()
)

answer = chain.invoke("What are the real-world battery experiences with Lenovo Legion laptops?")
print(answer)
```

---

### Example 5: Standalone Context Compressor

Compress noisy web pages, transcripts, or long documents before sending to any LLM:

```python
from brongtus import TokenCompressor

compressor = TokenCompressor(target_max_words=300)

query = "fusion energy ignition milestones"
compressed, savings = compressor.compress(query, massive_html_or_text)

print(f"Pruned {savings}% of tokens while keeping critical answers.")
```

---

## 🖥️ Interactive CLI

`brongtus` comes with a ready-to-run interactive CLI runner (`run.py`):

```powershell
# Run a single query directly:
python run.py "is Mojo programming language ready for production?"

# Or enter interactive conversation mode:
python run.py
```

---

## 🛠️ Publishing to PyPI

To build and release your own version:

```bash
# 1. Install build tools
pip install build twine

# 2. Build distributions (.whl and .tar.gz)
python -m build

# 3. Check archive integrity
twine check dist/*

# 4. Upload to PyPI
twine upload dist/*
```

---

## 📜 License

MIT License. Free for open-source and commercial use.
