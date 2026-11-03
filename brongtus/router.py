"""Dynamic Agentic Temporal Router.

Dynamically anchors search queries to the current system date without any hardcoded word lists,
preventing offline LLMs from hallucinating outdated historical narratives.
"""

from datetime import datetime
from typing import Dict, Any


class AgenticTemporalRouter:
    """Dynamic temporal router that injects real-time reference context with zero added latency."""

    def route(self, query: str, has_gemini: bool = False) -> Dict[str, Any]:
        """Dynamically evaluate routing context without hardcoded word lists."""
        system_date = datetime.now().strftime("%B %d, %Y")
        strategy = "gemini_online_grounded" if has_gemini else "local_strictly_grounded"

        return {
            "system_date": system_date,
            "strategy": strategy,
        }
