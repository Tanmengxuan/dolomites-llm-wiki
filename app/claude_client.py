import logging

from app.agents import orchestrator, wiki_qa_agent, web_search_agent, planner_agent, ingest_agent
from app import session_store
from app.models import ChatResponse

logger = logging.getLogger(__name__)


async def run_chat_turn(
    session_id: str,
    user_message: str,
    intent: str | None = None,
    context: str | None = None,
) -> ChatResponse:
    logger.info("── New query (session=%s) ──────────────────", session_id[:8])
    logger.info("User: %s", user_message)

    if context is None:
        context = session_store.get_context_string(session_id)
        if context:
            logger.info("[claude_client] Context from %d previous exchange(s) retrieved",
                        len(session_store._exchange_store.get(session_id, [])))

    if intent is None:
        intent = orchestrator.classify_intent(user_message, context)
    logger.info("[claude_client] Routing to intent=%s", intent)

    if intent == "web_search":
        response = await web_search_agent.run(session_id, user_message, context)
        session_store.append_web_search(session_id, user_message, response.answer)
    elif intent == "ingest":
        web_search_history = session_store.get_web_search_history(session_id)
        response = await ingest_agent.run(session_id, user_message, context, web_search_history)
    elif intent == "planner":
        response = await planner_agent.run(session_id, user_message, context)
    else:
        response = await wiki_qa_agent.run(session_id, user_message, context)

    session_store.append_exchange(session_id, response.intent, user_message, response.answer)
    return response
