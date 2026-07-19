import asyncio
import logging

from app import session_store, wiki_utils
from app.models import ChatResponse
from app.skills.wiki_qa import identify, synthesize

logger = logging.getLogger(__name__)


async def run(session_id: str, user_message: str) -> ChatResponse:
    logger.info("[wiki_qa_agent] Starting (session=%s)", session_id[:8])

    known_pages = set(wiki_utils.list_wiki_pages())
    logger.info("[wiki_qa_agent] %d pages available on disk", len(known_pages))

    logger.info("[wiki_qa_agent] Step 1: identifying relevant pages")
    valid_pages = await asyncio.to_thread(
        identify.identify_pages, user_message, known_pages
    )

    if not valid_pages:
        logger.info("[wiki_qa_agent] No relevant pages found — returning default response")
        session_store.append_user(session_id, user_message)
        answer = "I couldn't find that in the wiki."
        session_store.append_assistant(session_id, answer)
        return ChatResponse(
            answer=answer,
            sources=[],
            offer_save=False,
            session_id=session_id,
            intent="wiki_qa",
        )

    logger.info("[wiki_qa_agent] Step 1 result: %s", valid_pages)
    logger.info("[wiki_qa_agent] Step 2: synthesizing answer")
    answer, offer_save = await asyncio.to_thread(
        synthesize.synthesize_answer, session_id, user_message, valid_pages
    )

    session_store.append_user(session_id, user_message)
    session_store.append_assistant(session_id, answer)

    logger.info("[wiki_qa_agent] Done — offer_save=%s", offer_save)
    return ChatResponse(
        answer=answer,
        sources=valid_pages,
        offer_save=offer_save,
        session_id=session_id,
        intent="wiki_qa",
    )
