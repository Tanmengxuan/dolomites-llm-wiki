import logging

from app.agents import orchestrator, wiki_qa_agent, web_search_agent, ingest_agent
from app.models import ChatResponse

logger = logging.getLogger(__name__)


async def run_chat_turn(session_id: str, user_message: str) -> ChatResponse:
    logger.info("── New query (session=%s) ──────────────────", session_id[:8])
    logger.info("User: %s", user_message)

    intent = orchestrator.classify_intent(user_message)
    logger.info("[claude_client] Routing to intent=%s", intent)

    if intent == "web_search":
        return await web_search_agent.run(session_id, user_message)
    elif intent == "ingest":
        return await ingest_agent.run(session_id, user_message)
    else:
        return await wiki_qa_agent.run(session_id, user_message)
