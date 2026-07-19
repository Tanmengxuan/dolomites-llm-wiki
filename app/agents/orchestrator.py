import logging
from typing import Literal

from app.anthropic_client import get_client

logger = logging.getLogger(__name__)

_MODEL = "claude-haiku-4-5-20251001"

_SYSTEM = (
    "You are an intent classifier for a Dolomites trip planning chatbot. "
    "Classify the user's message into exactly one of three intents:\n"
    "- wiki_qa: question answerable from existing wiki pages "
    "(itinerary, hikes, hotels, gear, budget, transport, etc.)\n"
    "- web_search: needs live or external info not in the wiki "
    "(weather, current prices, trail conditions, real-time URLs)\n"
    "- ingest: instruction to read a source file and update the wiki "
    "(\"ingest raw/...\", \"add this to the wiki\", \"update wiki with...\")\n"
    "Reply with exactly one word: wiki_qa, web_search, or ingest."
)

Intent = Literal["wiki_qa", "web_search", "ingest"]


def classify_intent(user_message: str) -> Intent:
    try:
        response = get_client().messages.create(
            model=_MODEL,
            max_tokens=10,
            system=_SYSTEM,
            messages=[{"role": "user", "content": user_message}],
        )
        raw = response.content[0].text.strip().lower()
        if raw in ("wiki_qa", "web_search", "ingest"):
            logger.info("[orchestrator] classified intent=%s", raw)
            return raw  # type: ignore[return-value]
        logger.warning("[orchestrator] unexpected intent %r — falling back to wiki_qa", raw)
    except Exception:
        logger.exception("[orchestrator] classification failed — falling back to wiki_qa")
    return "wiki_qa"
