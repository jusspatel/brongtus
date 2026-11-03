"""Web search and Playwright-powered scraping engine for local-exa."""

import asyncio
import re
import urllib.parse
from typing import List, Dict, Any, Optional
import httpx
from bs4 import BeautifulSoup
from .schemas import ScrapedDocument


class WebScraper:
    """Handles free local web search and dynamic JavaScript page scraping via Playwright."""

    def __init__(self, headless: bool = True, timeout_sec: int = 15, user_agent: Optional[str] = None):
        self.headless = headless
        self.timeout_sec = timeout_sec
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
        )

    async def search(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """Search the web for free using DuckDuckGo HTML endpoint without needing any API key."""
        encoded_query = urllib.parse.quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

        results = []
        try:
            async with httpx.AsyncClient(timeout=self.timeout_sec, follow_redirects=True) as client:
                resp = await client.get(url, headers=headers)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    links = soup.select(".result__body")
                    for link in links[:num_results]:
                        title_elem = link.select_one(".result__title a")
                        snippet_elem = link.select_one(".result__snippet")
                        if title_elem and title_elem.get("href"):
                            raw_href = title_elem["href"]
                            # DDG wraps urls in /l/?uddg=...
                            actual_url = self._extract_ddg_url(raw_href)
                            results.append({
                                "url": actual_url,
                                "title": title_elem.get_text(strip=True),
                                "snippet": snippet_elem.get_text(strip=True) if snippet_elem else "",
                            })
        except Exception:
            pass

        # If httpx was blocked or empty, fallback to Playwright search
        if not results:
            results = await self._search_with_playwright(query, num_results)

        return results

    async def search_reddit(self, query: str, num_results: int = 3) -> List[Dict[str, str]]:
        """Specifically search for Reddit community discussions regarding the query."""
        reddit_query = f"site:reddit.com {query}"
        results = await self.search(reddit_query, num_results=num_results)
        # Filter strictly for reddit links
        reddit_links = [r for r in results if "reddit.com" in r.get("url", "")]
        return reddit_links[:num_results]

    def _extract_ddg_url(self, raw_url: str) -> str:
        """Decode DuckDuckGo target URL."""
        if "uddg=" in raw_url:
            match = re.search(r"uddg=([^&]+)", raw_url)
            if match:
                return urllib.parse.unquote(match.group(1))
        return raw_url

    async def _search_with_playwright(self, query: str, num_results: int) -> List[Dict[str, str]]:
        """Fallback to Playwright headless search when standard HTTP gets challenged."""
        results = []
        try:
            from playwright.async_api import async_playwright
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=self.headless)
                context = await browser.new_context(user_agent=self.user_agent)
                page = await context.new_page()
                encoded = urllib.parse.quote_plus(query)
                await page.goto(f"https://html.duckduckgo.com/html/?q={encoded}", timeout=self.timeout_sec * 1000)
                elements = await page.query_selector_all(".result__body")
                for el in elements[:num_results]:
                    title_node = await el.query_selector(".result__title a")
                    snippet_node = await el.query_selector(".result__snippet")
                    if title_node:
                        href = await title_node.get_attribute("href")
                        title = await title_node.inner_text()
                        snippet = await snippet_node.inner_text() if snippet_node else ""
                        if href:
                            results.append({
                                "url": self._extract_ddg_url(href),
                                "title": title.strip(),
                                "snippet": snippet.strip(),
                            })
                await browser.close()
        except Exception:
            pass
        return results

    async def scrape_url(self, url: str) -> ScrapedDocument:
        """Fetch and extract clean text from a URL using Playwright (rendering dynamic JS)."""
        html_content = ""
        title = ""

        # First attempt fast HTTP fetch
        try:
            async with httpx.AsyncClient(timeout=self.timeout_sec, follow_redirects=True) as client:
                resp = await client.get(url, headers={"User-Agent": self.user_agent})
                if resp.status_code == 200 and len(resp.text) > 1000:
                    html_content = resp.text
        except Exception:
            pass

        # If HTTP returned little or failed, use Playwright to render JS
        if not html_content or len(html_content) < 500:
            try:
                from playwright.async_api import async_playwright
                async with async_playwright() as p:
                    browser = await p.chromium.launch(headless=self.headless)
                    context = await browser.new_context(user_agent=self.user_agent)
                    page = await context.new_page()
                    await page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_sec * 1000)
                    title = await page.title()
                    html_content = await page.content()
                    await browser.close()
            except Exception:
                pass

        # Extract text via trafilatura
        extracted_text = ""
        if html_content:
            try:
                import trafilatura
                extracted_text = trafilatura.extract(html_content, output_format="txt", include_links=False) or ""
            except Exception:
                pass

            if not extracted_text:
                soup = BeautifulSoup(html_content, "html.parser")
                for s in soup(["script", "style", "nav", "footer", "header"]):
                    s.decompose()
                title = title or (soup.title.string if soup.title else "")
                extracted_text = soup.get_text(separator="\n", strip=True)

        return ScrapedDocument(
            url=url,
            title=title or url,
            raw_text=extracted_text,
            metadata={"html_len": len(html_content)},
        )

    async def scrape_urls_concurrently(self, urls: List[str], max_concurrency: int = 5) -> List[ScrapedDocument]:
        """Scrape multiple URLs concurrently."""
        sem = asyncio.Semaphore(max_concurrency)

        async def _bounded_scrape(u: str):
            async with sem:
                return await self.scrape_url(u)

        tasks = [_bounded_scrape(u) for u in urls]
        return await asyncio.gather(*tasks)
