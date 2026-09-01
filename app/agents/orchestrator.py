import logging
from typing import Literal

from app.anthropic_client import get_client, with_date

logger = logging.getLogger(__name__)

_MODEL = "claude-haiku-4-5"

_SYSTEM = """You are an intent classifier for a Dolomites trip planning chatbot (Sept 6–19, 2026).
Classify the user's message into exactly one of four intents:

- wiki_qa: a self-contained question answerable from the wiki alone — itinerary, accommodations,
  hikes, restaurants, gear, budget, transport. The answer does not need live data.
  Examples: "what hotel am I staying at on Sept 7?", "what hikes are planned?", "what's the budget?"

- web_search: a self-contained question that needs live or external data not in the wiki —
  current weather, real-time prices, trail conditions, transport schedules, external URLs.
  Examples: "what's the weather in Cortina this week?", "is the Seceda cable car running?"

- planner: the question requires combining wiki knowledge with live web data, or involves
  multi-step reasoning across both sources, or is ambiguous enough that the agent needs to
  decide at runtime what sources to consult.
  Examples: "how's the weather where I am today?", "is the hike I planned for Sept 10 safe given
  current conditions?", "what should I pack given today's forecast for my location?"

- ingest: an explicit instruction to save content into the wiki — reading a source file,
  adding web search results, or updating wiki pages.
  Examples: "ingest raw/...", "add that to the wiki", "save this information", "update the wiki with..."

Rules:
- Use planner only when the question genuinely needs BOTH wiki context AND live web data,
  or when runtime reasoning is needed to decide what to look up. Do not use it for simple
  questions that clearly belong to wiki_qa or web_search alone.
- Use context to resolve ambiguous references (e.g. "add that to the wiki" after a web
  search = ingest; "where am I staying?" = wiki_qa).
- Reply with exactly one word: wiki_qa, web_search, planner, or ingest."""

Intent = Literal["wiki_qa", "web_search", "ingest", "planner"]


def classify_intent(user_message: str, context: str = "") -> Intent:
    content = (
        f"{context}\n\nCurrent message: {user_message}"
        if context
        else user_message
    )
    try:
        response = get_client().messages.create(
            model=_MODEL,
            max_tokens=10,
            system=with_date(_SYSTEM),
            messages=[{"role": "user", "content": content}],
        )
        raw = response.content[0].text.strip().lower()
        if raw in ("wiki_qa", "web_search", "ingest", "planner"):
            logger.info("[orchestrator] classified intent=%s", raw)
            return raw  # type: ignore[return-value]
        logger.warning("[orchestrator] unexpected intent %r — falling back to wiki_qa", raw)
    except Exception:
        logger.exception("[orchestrator] classification failed — falling back to wiki_qa")
    return "wiki_qa"
