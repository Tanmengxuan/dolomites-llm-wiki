import logging

from claude_agent_sdk import query
from claude_agent_sdk.types import AssistantMessage, ClaudeAgentOptions, ResultMessage, TextBlock

from app.models import ChatResponse
from app.skills.web_search.prompts import WEB_SEARCH_SYSTEM

logger = logging.getLogger(__name__)


async def run(session_id: str, user_message: str, context: str = "") -> ChatResponse:
    logger.info("[web_search_agent] Starting (session=%s)", session_id[:8])
    logger.info("[web_search_agent] Query: %s", user_message)

    prompt = (
        f"{context}\n\nCurrent request: {user_message}"
        if context
        else user_message
    )

    options = ClaudeAgentOptions(
        tools=["WebSearch", "WebFetch"],
        system_prompt=WEB_SEARCH_SYSTEM,
        permission_mode="bypassPermissions",
        max_turns=5,
    )

    result_text: str | None = None
    assistant_parts: list[str] = []

    try:
        async for msg in query(prompt=prompt, options=options):
            if isinstance(msg, ResultMessage):
                logger.info(
                    "[web_search_agent] ResultMessage — turns: %d, cost: $%s",
                    msg.num_turns,
                    msg.total_cost_usd,
                )
                if msg.result:
                    result_text = msg.result
            elif isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock) and block.text:
                        assistant_parts.append(block.text)
    except Exception:
        logger.exception("[web_search_agent] query failed")

    answer = result_text or " ".join(assistant_parts).strip()
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
