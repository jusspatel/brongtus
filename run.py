"""Interactive CLI for testing Brongtus with your own custom prompts."""

import os
import sys
import asyncio
import warnings

# Suppress harmless sampling warnings from langchain_google_genai
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure utf-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Automatically load .env file
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

from brongtus import Brongtus


async def search_prompt(prompt: str, gemini_key: str = None, gemini_model: str = "gemini-2.5-flash"):
    print("\n" + "=" * 60)
    print(f"[Query]: '{prompt}'")
    if gemini_key:
        print(f"[Mode]: Hybrid (Local Scraping + {gemini_model} + Reddit Fact-Checking)")
    else:
        print("[Mode]: Local Mode (Pure Offline Heuristics / Ollama)")
    print("=" * 60)
    print("Searching the web and extracting content...")

    engine = Brongtus(
        gemini_api_key=gemini_key,
        gemini_model=gemini_model,
        enable_compression=True,
        headless=True,
    )

    response = await engine.asearch(query=prompt, num_results=3)

    print(f"\n[Results Summary] (Pruned {response.token_savings_percent}% of junk tokens)")
    print("-" * 60)

    for i, res in enumerate(response.results, 1):
        print(f"\n[{i}] {res.title}")
        print(f"    URL: {res.url}")
        if res.score is not None:
            print(f"    Score: {res.score}")
        if res.summary:
            print(f"    Summary: {res.summary}")
        if res.highlights:
            print("    Key Highlights:")
            for h in res.highlights:
                print(f"      - {h}")

    if response.community_insights:
        ci = response.community_insights
        print("\n" + "=" * 60)
        print("[Community Insights] (Reddit + Cross-Verified)")
        print("=" * 60)
        if ci.get("community_sentiment"):
            print(f"\nCommunity Consensus:\n   {ci['community_sentiment']}")
        if ci.get("verified_takeaways"):
            print("\nVerified Takeaways:")
            for item in ci["verified_takeaways"]:
                print(f"   [+] {item}")
        if ci.get("debunked_or_unverified"):
            print("\nDebunked / Unverified Rumors:")
            for item in ci["debunked_or_unverified"]:
                print(f"   [!] {item}")


def main():
    gemini_key = os.getenv("GEMINI_API_KEY")
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # If prompt provided as command line arguments
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        asyncio.run(search_prompt(prompt, gemini_key, gemini_model))
        return

    # Interactive loop
    print("=" * 60)
    print("Welcome to Brongtus Interactive Search Tester")
    print("=" * 60)
    if not gemini_key:
        print("Tip: Set $env:GEMINI_API_KEY='your_key' to enable Gemini & Reddit fact-checking.")
    else:
        print(f"Gemini Key detected! Model: {gemini_model}")
    print("Type your query below and press Enter (or 'exit' to quit):\n")

    while True:
        try:
            query = input("Enter prompt > ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Exiting Brongtus. Goodbye!")
                break
            asyncio.run(search_prompt(query, gemini_key, gemini_model))
            print("\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
