import logging

from app.anthropic_client import get_client
from app import session_store, wiki_utils
from .prompts import SYNTHESIS_SYSTEM_STATIC

logger = logging.getLogger(__name__)
MODEL = "claude-sonnet-4-6"


def synthesize_answer(
    session_id: str,
    user_message: str,
    page_slugs: list[str],
) -> tuple[str, bool]:
    logger.info("[synthesize] Reading pages: %s", page_slugs)
    pages = wiki_utils.read_pages(page_slugs)
    logger.info(
        "[synthesize] Loaded %d page(s), total chars: %d",
        len(pages),
        sum(len(v) for v in pages.values()),
    )

    wiki_block_parts = []
    for slug in page_slugs:
        if slug in pages:
            wiki_block_parts.append(f"=== {slug}.md ===\n{pages[slug]}")
    wiki_block = "\n\n".join(wiki_block_parts)

    system_content = [
        {"type": "text", "text": SYNTHESIS_SYSTEM_STATIC},
        {
            "type": "text",
            "text": f"Relevant wiki pages:\n\n{wiki_block}",
            "cache_control": {"type": "ephemeral"},
        },
    ]

    history = session_store.get_or_create(session_id)
    messages = history + [{"role": "user", "content": user_message}]
    logger.info("[synthesize] Conversation history: %d message(s) in context", len(messages))

    logger.info("[synthesize] Calling %s (prompt cache applied to wiki block)", MODEL)
    response = get_client().messages.create(
        model=MODEL,
        max_tokens=1024,
        system=system_content,
        messages=messages,
    )

    usage = response.usage
    logger.info(
        "[synthesize] Token usage — input: %d, output: %d, cache_read: %s, cache_write: %s",
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
