import json
import logging

from app.anthropic_client import get_client, with_date
from app import wiki_utils
from .prompts import PAGE_ID_SYSTEM

logger = logging.getLogger(__name__)
MODEL = "claude-sonnet-4-6"


def identify_pages(user_message: str, known_pages: set[str], context: str = "") -> list[str]:
    index_content = wiki_utils.read_index()
    context_section = f"Conversation context:\n{context}\n\n" if context else ""
    prompt = (
        f"{context_section}"
        f"Wiki Table of Contents:\n---\n{index_content}\n---\n\n"
        f"User question: {user_message}\n\n"
        "Return the JSON array of relevant page slugs."
    )
    try:
        response = get_client().messages.create(
            model=MODEL,
            max_tokens=200,
            system=with_date(PAGE_ID_SYSTEM),
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        logger.info("[identify] Step 1 raw response: %s", raw)
        candidates: list[str] = json.loads(raw)
        valid = [p for p in candidates if isinstance(p, str) and p in known_pages]
        discarded = [p for p in candidates if p not in known_pages]
        if discarded:
            logger.warning("[identify] Discarded hallucinated page slugs: %s", discarded)
        return sorted(valid)
    except Exception:
        logger.exception("[identify] Page identification failed")
        return []
