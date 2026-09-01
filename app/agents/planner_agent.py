import asyncio
import logging

from app import session_store, wiki_utils
from app.anthropic_client import get_client, with_date
from app.models import ChatResponse
from app.skills.planner.prompts import PLANNER_SYSTEM
from app.skills.wiki_qa import identify, synthesize
from app.agents.web_search_agent import _run_search

logger = logging.getLogger(__name__)

_PLANNER_TOOLS = [
    {
        "name": "wiki_search",
        "description": (
            "Search the local wiki for anything about this Dolomites trip: "
            "itinerary, accommodations, hikes, budget, transport, gear, restaurants, dates."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Natural language search query."}},
            "required": ["query"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Search the live web for time-sensitive information: "
            "current weather, ticket prices, trail conditions, opening hours, transport schedules."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query."}},
            "required": ["query"],
        },
    },
    {
        "name": "ask_user",
        "description": (
            "Ask the user a specific clarifying question when you are blocked and cannot proceed. "
            "The question MUST name the exact piece of information you are missing — "
            "e.g. 'Which day of the trip did you mean: Sept 8 or Sept 14?' or "
            "'Your question mentions a date outside Sept 6–19; did you mean a specific day in the itinerary?' "
            "Do NOT ask a generic question like 'Could you clarify?' "
            "Use only when blocked — not for minor ambiguities you can resolve with a stated assumption."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"question": {"type": "string", "description": "A targeted question naming the specific information you need."}},
            "required": ["question"],
        },
    },
]


def _wiki_search(query: str, session_id: str, context: str) -> str:
    known_pages = set(wiki_utils.list_wiki_pages())
    if not known_pages:
        return "No wiki pages found."
    pages = identify.identify_pages(query, known_pages, context)
    if not pages:
        return "No relevant wiki pages found for this query."
    answer, _ = synthesize.synthesize_answer(session_id, query, pages, context)
    return answer


def _run_planner(session_id: str, user_message: str, context: str) -> tuple[str, bool]:
    """
    Returns (answer, is_question).
    is_question=True means the answer is a clarifying question for the user.
    """
    saved = session_store.pop_planner_state(session_id)
    if saved:
        # Resume a paused planner: inject the user's answer as the tool_result for ask_user.
        messages = saved["messages"]
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": saved["ask_user_tool_id"], "content": user_message}
        ]})
        logger.info("[planner] Resuming saved state; user answered: %s", user_message[:120])
    else:
        messages = [{"role": "user", "content": user_message}]
        if context:
            messages = [{"role": "user", "content": f"Conversation context:\n{context}\n\nCurrent question: {user_message}"}]

    text_parts: list[str] = []

    for turn in range(10):
        response = get_client().messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=with_date(PLANNER_SYSTEM),
            tools=_PLANNER_TOOLS,
            messages=messages,
        )
        logger.info("[planner] turn %d — stop_reason=%s", turn + 1, response.stop_reason)

        for block in response.content:
            if block.type == "text" and block.text:
                text_parts.append(block.text)

        if response.stop_reason == "end_turn":
            break

        if response.stop_reason != "tool_use":
            break

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        asked_user = False

        for block in response.content:
            if block.type != "tool_use":
                continue

            logger.info("[planner] tool_call: %s(%s)", block.name, str(block.input)[:120])

            if block.name == "ask_user":
                question = block.input.get("question")
                if not question:
                    logger.error("[planner] ask_user called with no 'question' field")
                    question = "I need more information to answer that. Could you tell me more about what you're looking for?"
                logger.info("[planner] ask_user → %s", question)
                # Persist the in-progress message history so the loop can resume
                # once the user answers. `messages` already has the assistant turn
                # with the ask_user tool call appended above.
                session_store.save_planner_state(session_id, messages, block.id)
                return question, True

            elif block.name == "wiki_search":
                result = _wiki_search(block.input["query"], session_id, context)

            elif block.name == "web_search":
                result = _run_search(block.input["query"])

            else:
                result = f"Unknown tool: {block.name}"

            logger.info("[planner] tool_result: %s", result[:120])
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })

        if tool_results:
            messages.append({"role": "user", "content": tool_results})

    return "\n\n".join(text_parts).strip(), False


async def run(session_id: str, user_message: str, context: str = "") -> ChatResponse:
    logger.info("[planner] Starting (session=%s)", session_id[:8])
    logger.info("[planner] Query: %s", user_message)

    answer = ""
    is_question = False
    try:
        answer, is_question = await asyncio.to_thread(
            _run_planner, session_id, user_message, context
        )
    except Exception:
        logger.exception("[planner] failed")

    if not answer:
        answer = "I couldn't find enough information to answer that. Could you rephrase?"

    logger.info("[planner] Done (is_question=%s)", is_question)
    return ChatResponse(
        answer=answer,
        sources=[],
        offer_save=False,
        session_id=session_id,
        intent="planner",
    )
