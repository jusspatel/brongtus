# Brongtus

[![PyPI version](https://img.shields.io/pypi/v/brongtus.svg)](https://pypi.org/project/brongtus/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> **A free, local, intelligent search tool mimicking Exa with Playwright, LangGraph, up to 90% algorithmic token pruning, and optional Gemini enhancement with Reddit community fact-checking.**

---

## Highlights

- **100% Free & Open-Source:** No mandatory API subscriptions.
- **Playwright Powered:** Renders modern JavaScript, SPAs, and dynamic web pages.
- **Up to 90% Algorithmic Token Pruning:** Uses `justext`, `trafilatura`, and BM25 Okapi passage ranking to strip boilerplates, ads, and navigation noise without hardcoded lists.
- **Hybrid Intelligence (Local + Gemini):**
  - **Local Mode**: Runs offline via Ollama (`qwen2.5`, `llama3.2`, `deepseek-r1`) or built-in heuristic extractors.
  - **Gemini Hybrid Mode**: Pass your `GEMINI_API_KEY` (defaults to `gemini-3.6-flash`) to unlock ultra-sharp curation and **Reddit community discussion cross-verification** (filtering out rumors while capturing verified community consensus).
- **First-Class LangChain Integration:** Natively implements `BaseTool` and `BaseRetriever` ready for AI agents and RAG pipelines.

---

## Installation

```bash
pip install brongtus
playwright install chromium
```

To enable Gemini enhancement and local LLM drivers:
```bash
pip install "brongtus[gemini,local-llm]"
```

---

## Quickstart

### 1. Pure Local Search (100% Free)

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
        print(f"  - {h}")
    print("-" * 50)
```

---

### 2. Hybrid Mode (Gemini 3.6 Flash + Reddit Fact-Checking)

Add your `GEMINI_API_KEY` (in `.env` or passed directly) to unlock ultra-smart curation and Reddit community cross-verification:

```python
import os
from brongtus import Brongtus

engine = Brongtus(
    gemini_api_key=os.getenv("GEMINI_API_KEY"),
    gemini_model="gemini-3.6-flash",  # Default model
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

## LangChain & Agent Integration

### LangChain AI Agent Tool (`BrongtusSearchResults`)

Bind `BrongtusSearchResults` directly to tool-calling agents:

```python
from brongtus.langchain import BrongtusSearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Initialize tool
search_tool = BrongtusSearchResults(num_results=3)

# Bind tool to model
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash").bind_tools([search_tool])

# Run query
ai_msg = llm.invoke([HumanMessage(content="Find real-world experiences with Lenovo Legion battery life")])
```

---

### LangChain RAG Retriever (`BrongtusRetriever`)

Plug local search directly into RAG chains:

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

## API Reference

### `Brongtus` Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `gemini_api_key` | `str \| None` | `None` | Google Gemini API key. Automatically reads from `.env` or system environment. |
| `gemini_model` | `str` | `"gemini-3.6-flash"` | Gemini model identifier (e.g. `gemini-3.6-flash`, `gemini-2.5-flash`, `gemini-1.5-pro`). |
| `local_model` | `str` | `"qwen2.5:7b"` | Name of local Ollama model to use for query planning and offline extraction. |
| `custom_llm` | `BaseChatModel` | `None` | Any custom LangChain chat model instance. |
| `enable_compression` | `bool` | `True` | Enables algorithmic 3-tier token reduction (pruning ~60–90% of junk tokens). |
| `target_compression_words` | `int` | `500` | Target word budget per page for compressed passages. |
| `headless` | `bool` | `True` | Runs Playwright in headless browser mode. |
| `enable_hitl` | `bool` | `False` | Enables LangGraph Human-in-the-Loop review gates for candidate URLs. |

---

### `SearchResponse` & `SearchResult` Fields

```python
response = engine.search("quantum computing milestones")

# Response Metadata
response.query                 # Original query string
response.token_savings_percent # Percentage of tokens pruned (e.g. 74.2%)
response.community_insights    # Fact-checked Reddit insights (when Gemini is active)

# Results
for r in response.results:
    r.title                    # Web page title
    r.url                      # Web page URL
    r.score                    # Relevance score (0.0 to 1.0)
    r.summary                  # Factual, concise summary
    r.highlights               # Key verbatim highlight snippets
    r.text                     # Algorithmic compressed text
```

---

## License

MIT License. Free for open-source and commercial use.
