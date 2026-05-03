import json
import logging
import os

import anthropic
from dotenv import load_dotenv

from . import session_store, wiki_utils
from .models import ChatResponse

load_dotenv()

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"

_client: anthropic.Anthropic | None = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


_PAGE_ID_SYSTEM = (
    "You are a routing assistant for a personal Dolomites trip wiki. "
    "Given a user question and the wiki's table of contents, return a JSON array "
    "of the most relevant page filenames to consult. "
    "Return ONLY the JSON array, no explanation. "
    "Use exact slugs from the index (without .md extension). "
    "Return between 1 and 5 slugs. If no pages are relevant, return an empty array []."
)

_SYNTHESIS_SYSTEM_STATIC = (
    "You are a helpful travel assistant for a personal Dolomites trip wiki. "
    "Answer only from the provided wiki content. "
    "If the answer is not in the wiki, say clearly: \"I couldn't find that in the wiki.\" "
    "Cite the specific wiki pages you drew from using [[page-name]] inline throughout your answer. "
    "Keep answers concise and practical. Use markdown formatting where helpful. "
    "At the very end of your response, on its own line, add exactly one of:\n"
    "OFFER_SAVE: yes\n"
    "OFFER_SAVE: no\n"
    "Use 'yes' only when the answer is substantive and novel enough to warrant saving as a new wiki page."
)


def run_chat_turn(session_id: str, user_message: str) -> ChatResponse:
    logger.info("── New query (session=%s) ──────────────────", session_id[:8])
    logger.info("User: %s", user_message)

    known_pages = set(wiki_utils.list_wiki_pages())
    logger.info("[wiki_utils] %d pages available on disk", len(known_pages))

    # Step 1: identify relevant pages
    logger.info("[claude_client] Step 1: sending index.md + query to %s for page identification", MODEL)
    valid_pages = _identify_pages(user_message, known_pages)

    if not valid_pages:
        logger.info("[claude_client] No relevant pages found — returning default response")
        session_store.append_user(session_id, user_message)
        answer = "I couldn't find that in the wiki."
        session_store.append_assistant(session_id, answer)
        return ChatResponse(
            answer=answer,
            sources=[],
            offer_save=False,
            session_id=session_id,
        )

    logger.info("[claude_client] Step 1 result: %s", valid_pages)

    # Step 2: synthesize answer
    logger.info("[claude_client] Step 2: loading pages and calling %s for answer synthesis", MODEL)
    answer, offer_save = _synthesize_answer(session_id, user_message, valid_pages)

    session_store.append_user(session_id, user_message)
    session_store.append_assistant(session_id, answer)

    logger.info("[claude_client] Done — offer_save=%s", offer_save)
    return ChatResponse(
        answer=answer,
        sources=valid_pages,
        offer_save=offer_save,
        session_id=session_id,
    )


def _identify_pages(user_message: str, known_pages: set[str]) -> list[str]:
    index_content = wiki_utils.read_index()
    prompt = (
        f"Wiki Table of Contents:\n---\n{index_content}\n---\n\n"
        f"User question: {user_message}\n\n"
        "Return the JSON array of relevant page slugs."
    )
    try:
        response = get_client().messages.create(
            model=MODEL,
            max_tokens=200,
            system=_PAGE_ID_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        logger.info("[claude_client] Step 1 raw response: %s", raw)
        candidates: list[str] = json.loads(raw)
        valid = [p for p in candidates if isinstance(p, str) and p in known_pages]
        discarded = [p for p in candidates if p not in known_pages]
        if discarded:
            logger.warning("[claude_client] Discarded hallucinated page slugs: %s", discarded)
        return sorted(valid)
    except Exception:
        logger.exception("[claude_client] Page identification failed")
        return []


def _synthesize_answer(
    session_id: str,
    user_message: str,
    page_slugs: list[str],
) -> tuple[str, bool]:
    logger.info("[wiki_utils] Reading pages from disk: %s", page_slugs)
    pages = wiki_utils.read_pages(page_slugs)
    logger.info("[wiki_utils] Loaded %d page(s), total chars: %d",
                len(pages), sum(len(v) for v in pages.values()))

    wiki_block_parts = []
    for slug in page_slugs:
        if slug in pages:
            wiki_block_parts.append(f"=== {slug}.md ===\n{pages[slug]}")
    wiki_block = "\n\n".join(wiki_block_parts)

    system_content = [
        {
            "type": "text",
            "text": _SYNTHESIS_SYSTEM_STATIC,
        },
        {
            "type": "text",
            "text": f"Relevant wiki pages:\n\n{wiki_block}",
            "cache_control": {"type": "ephemeral"},
        },
    ]

    history = session_store.get_or_create(session_id)
    messages = history + [{"role": "user", "content": user_message}]
    logger.info("[session_store] Conversation history: %d message(s) in context", len(messages))

    logger.info("[claude_client] Calling %s for answer synthesis (prompt cache applied to wiki block)", MODEL)
    response = get_client().messages.create(
        model=MODEL,
        max_tokens=1024,
        system=system_content,
        messages=messages,
    )

    usage = response.usage
    logger.info(
        "[claude_client] Step 2 token usage — input: %d, output: %d, cache_read: %s, cache_write: %s",
        usage.input_tokens,
        usage.output_tokens,
        getattr(usage, "cache_read_input_tokens", "n/a"),
        getattr(usage, "cache_creation_input_tokens", "n/a"),
    )

    full_text: str = response.content[0].text

    offer_save = "OFFER_SAVE: yes" in full_text
    clean = (
        full_text
        .replace("OFFER_SAVE: yes", "")
        .replace("OFFER_SAVE: no", "")
        .strip()
    )

    return clean, offer_save
