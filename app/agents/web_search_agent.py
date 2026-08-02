import asyncio
import logging

from app.anthropic_client import get_client
from app.models import ChatResponse
from app.skills.web_search.prompts import WEB_SEARCH_SYSTEM

logger = logging.getLogger(__name__)

_WEB_SEARCH_TOOL = {
    "type": "web_search_20260209",
    "name": "web_search",
}


def _run_search(prompt: str) -> tuple[str, list[str]]:
    response = get_client().messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        system=WEB_SEARCH_SYSTEM,
        tools=[_WEB_SEARCH_TOOL],
        messages=[{"role": "user", "content": prompt}],
    )
    text_parts = []
    sources: list[str] = []
    seen: set[str] = set()
    for block in response.content:
        if block.type != "text":
            continue
        text_parts.append(block.text)
        for citation in getattr(block, "citations", None) or []:
            url = getattr(citation, "url", None)
            if not url:
                continue
            text_parts.append(f" (source: {url})")
            if url not in seen:
                seen.add(url)
                sources.append(url)
    return "".join(text_parts).strip(), sources


async def run(session_id: str, user_message: str, context: str = "") -> ChatResponse:
    logger.info("[web_search_agent] Starting (session=%s)", session_id[:8])
    logger.info("[web_search_agent] Query: %s", user_message)

    prompt = (
        f"{context}\n\nCurrent request: {user_message}"
        if context
        else user_message
    )

    answer = ""
    sources: list[str] = []
    try:
        answer, sources = await asyncio.to_thread(_run_search, prompt)
    except Exception:
        logger.exception("[web_search_agent] search failed")

    if not answer:
        answer = "I couldn't find relevant web information for your query."

    logger.info("[web_search_agent] Done (sources=%d)", len(sources))
    return ChatResponse(
        answer=answer,
        sources=sources,
        offer_save=False,
        session_id=session_id,
        intent="web_search",
    )
