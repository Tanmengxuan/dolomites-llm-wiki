import asyncio
import logging

from app import session_store, wiki_utils
from app.models import ChatResponse
from app.skills.wiki_qa import identify, synthesize

logger = logging.getLogger(__name__)


async def run(session_id: str, user_message: str, context: str = "") -> ChatResponse:
    logger.info("[wiki_qa_agent] Starting (session=%s)", session_id[:8])

    if session_store.consume_wiki_error(session_id):
        logger.info("[wiki_qa_agent] [TEST] Simulating error for session %s", session_id[:8])
        raise RuntimeError("[TEST] Simulated wiki_qa error — type 'try again' to retry for real.")

    known_pages = set(wiki_utils.list_wiki_pages())
    logger.info("[wiki_qa_agent] %d pages available on disk", len(known_pages))

    logger.info("[wiki_qa_agent] Step 1: identifying relevant pages")
    valid_pages = await asyncio.to_thread(
        identify.identify_pages, user_message, known_pages, context
    )

    if not valid_pages:
        logger.info("[wiki_qa_agent] No relevant pages found — returning default response")
        return ChatResponse(
            answer="I couldn't find that in the wiki.",
            sources=[],
            offer_save=False,
            session_id=session_id,
            intent="wiki_qa",
        )

    logger.info("[wiki_qa_agent] Step 1 result: %s", valid_pages)
    logger.info("[wiki_qa_agent] Step 2: synthesizing answer")
    answer, offer_save = await asyncio.to_thread(
        synthesize.synthesize_answer, session_id, user_message, valid_pages, context
    )

    logger.info("[wiki_qa_agent] Done — offer_save=%s", offer_save)
    return ChatResponse(
        answer=answer,
        sources=valid_pages,
        offer_save=offer_save,
        session_id=session_id,
        intent="wiki_qa",
    )
