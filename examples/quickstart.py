"""Example demonstrating Brongtus with token compression and search."""

import asyncio
from brongtus import Brongtus


async def main():
    print("Initializing Brongtus search engine...")
    brongtus = Brongtus(
        headless=True,
        enable_compression=True,
        target_compression_words=300,
    )

    query = "latest achievements in fusion energy research"
    print(f"\nSearching for: '{query}'...")
    response = await brongtus.asearch(query=query, num_results=3)

    print(f"\n=== Brongtus Search Results (Token Savings: {response.token_savings_percent}%) ===")
    for i, res in enumerate(response.results, 1):
        print(f"\n[{i}] {res.title}")
        print(f"URL: {res.url}")
        print(f"Relevance Score: {res.score}")
        if res.summary:
            print(f"Summary: {res.summary}")
        if res.highlights:
            print("Highlights:")
            for h in res.highlights:
                print(f"  * {h}")


if __name__ == "__main__":
    asyncio.run(main())
