import asyncio
import logging
import re
from urllib.parse import urlparse

from app.anthropic_client import get_client, with_date
from app import session_store
from app.models import ChatResponse
from app.skills.web_search.prompts import WEB_SEARCH_SYSTEM
from app.agents.url_fetch import fetch_url, FETCH_URL_TOOL

logger = logging.getLogger(__name__)

_WEB_SEARCH_TOOL = {
    "type": "web_search_20260209",
    "name": "web_search",
}

# Pattern 1: <<URL>> sentinel inserted by Phase 2
_RE_SENTINEL = re.compile(r'<<(https?://[^>]+)>>')
# Pattern 2: full HTML anchor <a href="URL" ...>label</a>
_RE_FULL_ANCHOR = re.compile(r'<a\s[^>]*href="(https?://[^"]+)"[^>]*>([^<]*)</a>', re.IGNORECASE)
# Pattern 3: broken partial HTML (URL" target="...">label) — Phase 1 leakage
_RE_BROKEN_ANCHOR = re.compile(r'\((https?://[^\s"]+)"[^)>]*>[^)]*\)')


def _to_md_link(url: str, label: str | None = None) -> str:
    try:
        host = urlparse(url).hostname or url
        name = host.removeprefix("www.").rsplit(".", 1)[0]
    except Exception:
        name = url
    return f'CITE[{name}]({url})'


def _process_citations(text: str) -> str:
    """Normalise all citation formats produced by any model into clean markdown links."""
    # Handle <<URL>> sentinels (Phase 2 format)
    text = _RE_SENTINEL.sub(lambda m: f' {_to_md_link(m.group(1).strip())}', text)
    # Handle full HTML anchors
    text = _RE_FULL_ANCHOR.sub(lambda m: _to_md_link(m.group(1), m.group(2)), text)
    # Handle broken partial HTML (URL" target="...">label)
    text = _RE_BROKEN_ANCHOR.sub(lambda m: _to_md_link(
        re.search(r'(https?://[^\s"]+)', m.group(0)).group(1),
        re.search(r'>([^)]+)\)$', m.group(0)) and re.search(r'>([^)]+)\)$', m.group(0)).group(1),
    ), text)
    return text


_PHASE2_SYSTEM = (
    "You are a web research assistant finalising an answer.\n\n"
    "You have a draft answer from a web search and the list of source URLs that were found. "
    "Your job:\n"
    "1. If any source URL would provide meaningfully better or more complete information "
    "(especially for live data like flight status, or detailed content the snippet missed), "
    "call fetch_url on it.\n"
    "2. Write the complete final answer. After every sentence that states a fact, insert "
    "<<URL>> immediately after it with the most appropriate source URL.\n"
    "Example: 'The trail is 9 km long. <<https://alltrails.com/trail/example>>'\n"
    "Use ONLY the <<URL>> tag format. No HTML, no markdown links. "
    "Only cite URLs that appear in the source list or that you fetched."
)


def _run_search(prompt: str) -> str:
    # Phase 1: web search — model searches and produces a draft
    response = get_client().messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        system=with_date(WEB_SEARCH_SYSTEM),
        tools=[_WEB_SEARCH_TOOL],
        messages=[{"role": "user", "content": prompt}],
    )

    text_parts: list[str] = []
    sources: list[tuple[str, str]] = []
    seen: set[str] = set()

    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "web_search_tool_result":
            if isinstance(block.content, list):
                for result in block.content:
                    url = getattr(result, "url", None)
                    title = getattr(result, "title", None) or url
                    if url and url not in seen:
                        seen.add(url)
                        sources.append((title, url))

    draft = " ".join(text_parts).strip()
    logger.info("[web_search_agent] Phase 1 draft (first 300): %s", draft[:300].replace("\n", " "))
    logger.info("[web_search_agent] Phase 1 done — %d source(s)", len(sources))

    if not draft:
        return draft
    if not sources:
        return _process_citations(draft)

    # Phase 2: agentic pass — model may call fetch_url for deeper content, then writes final cited answer
    url_list = "\n".join(f"- {title} → {url}" for title, url in sources)
    messages = [{"role": "user", "content": (
        f"Original query: {prompt}\n\n"
        f"Draft answer:\n{draft}\n\n"
        f"Available sources:\n{url_list}"
    )}]

    for turn in range(5):
        response = get_client().messages.create(
            model="claude-sonnet-4-6",
            max_tokens=8192,
            system=_PHASE2_SYSTEM,
            tools=[FETCH_URL_TOOL],
            messages=messages,
        )

        text_parts = [b.text for b in response.content if b.type == "text"]

        if response.stop_reason == "end_turn":
            final = " ".join(text_parts).strip()
            logger.info("[web_search_agent] Phase 2 raw (first 300): %s", final[:300].replace("\n", " "))
            logger.info("[web_search_agent] Phase 2 done")
            return _process_citations(final or draft)

        if response.stop_reason != "tool_use":
            break

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type == "tool_use" and block.name == "fetch_url":
                url = block.input["url"]
                logger.info("[web_search_agent] fetch_url: %s", url)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": fetch_url(url),
                })
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

    return _process_citations(draft)


async def run(session_id: str, user_message: str, context: str = "") -> ChatResponse:
    logger.info("[web_search_agent] Starting (session=%s)", session_id[:8])
    logger.info("[web_search_agent] Query: %s", user_message)

    if session_store.consume_web_error(session_id):
        logger.info("[web_search_agent] [TEST] Simulating network error for session %s", session_id[:8])
        raise RuntimeError("[TEST] Simulated network error — type 'try again' to retry for real.")

    prompt = (
        f"{context}\n\nCurrent request: {user_message}"
        if context
        else user_message
    )

    answer = ""
    try:
        answer = await asyncio.to_thread(_run_search, prompt)
    except Exception:
        logger.exception("[web_search_agent] search failed")

    if not answer:
        answer = "I couldn't find relevant web information for your query."

    logger.info("[web_search_agent] Done")
    return ChatResponse(
        answer=answer,
        sources=[],
        offer_save=False,
        session_id=session_id,
        intent="web_search",
    )
