import asyncio
import logging

from app import session_store, wiki_utils
from app.anthropic_client import get_client, with_date
from app.models import ChatResponse
from app.skills.planner.prompts import PLANNER_SYSTEM
from app.skills.wiki_qa import identify, synthesize
from app.agents.web_search_agent import _run_search
from app.agents.url_fetch import fetch_url, FETCH_URL_TOOL

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
    FETCH_URL_TOOL,
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


_REFLECT_SYSTEM = (
    "You are a quality-check assistant for a Dolomites trip planning chatbot. "
    "Given the user's original query and the agent's draft response, decide if the response "
    "adequately addresses the query.\n\n"
    "Reply with exactly one of:\n"
    "PASS\n"
    "FAIL: <specific gaps and concrete search queries that would fill them>\n\n"
    "Use PASS when the response is complete, accurate, and directly answers what was asked. "
    "Use FAIL only when the response is missing key information the user clearly asked for. "
    "The FAIL feedback must name the exact gaps and suggest specific wiki or web searches.\n\n"
    "---\n"
    "Examples:\n\n"
    "Query: What is the weather forecast for Val di Funes on September 8?\n"
    "Draft: The Dolomites region typically sees temperatures of 10–18 °C in early September "
    "with a chance of afternoon thunderstorms.\n"
    "Reply: FAIL: Response gives a general Dolomites regional forecast instead of Val di Funes "
    "on the specific date. Search web for 'Val di Funes weather forecast September 8 2026'.\n\n"
    "Query: How long is the hike from Seceda to Rasciesa?\n"
    "Draft: The Seceda ridge offers stunning views of the Odle/Geisler massif. "
    "It is one of the most photographed spots in the Dolomites.\n"
    "Reply: FAIL: Response describes the scenery but never gives the distance or duration of "
    "the Seceda–Rasciesa route. Search wiki for 'seceda rasciesa hike' and web for "
    "'Seceda to Rasciesa trail distance duration'.\n\n"
    "Query: What is the closest bus stop to Rifugio Puez?\n"
    "Draft: Rifugio Puez is located at 2475 m on the Puez plateau above Val Gardena. "
    "The nearest bus stop is at Plan de Gralba, roughly a 3-hour hike below the rifugio.\n"
    "Reply: PASS\n\n"
    "Query: Can I change day 3 to a rest day given the weather?\n"
    "Draft: Based on the forecast for September 8 showing thunderstorms in the afternoon, "
    "it would be advisable to make day 3 a rest day or switch to a shorter valley walk.\n"
    "Reply: PASS\n"
    "---"
)


def _reflect(user_query: str, answer: str) -> tuple[bool, str]:
    """Returns (passed, feedback). feedback is empty string if passed."""
    if not answer:
        return False, "No answer was produced. Search for information relevant to the query."
    try:
        resp = get_client().messages.create(
            model="claude-haiku-4-5",
            max_tokens=200,
            system=_REFLECT_SYSTEM,
            messages=[{"role": "user", "content": f"User query: {user_query}\n\nDraft response:\n{answer}"}],
        )
        text = resp.content[0].text.strip()
        if text.upper().startswith("PASS"):
            logger.info("[planner/reflect] PASS")
            return True, ""
        if text.upper().startswith("FAIL"):
            feedback = text[4:].lstrip(":").strip()
            logger.info("[planner/reflect] FAIL — %s", feedback[:120])
            return False, feedback
        logger.warning("[planner/reflect] unexpected response %r — treating as PASS", text[:80])
        return True, ""
    except Exception:
        logger.exception("[planner/reflect] error — skipping reflection")
        return True, ""  # on failure, don't block the response


def _wiki_search(query: str, session_id: str, context: str) -> str:
    known_pages = set(wiki_utils.list_wiki_pages())
    if not known_pages:
        return "No wiki pages found."
    pages = identify.identify_pages(query, known_pages, context)
    if not pages:
        return "No relevant wiki pages found for this query."
    answer, _ = synthesize.synthesize_answer(session_id, query, pages, context)
    return answer


def _run_planner(session_id: str, user_message: str, context: str, reflect_enabled: bool = True) -> tuple[str, bool, list[dict]]:
    """
    Returns (answer, is_question, tool_calls).
    is_question=True means the answer is a clarifying question for the user.
    tool_calls is a list of {"tool": name, "query": input} dicts for UI display.
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
    tool_call_log: list[dict] = []
    reflected = False

    for turn in range(20):  # extra headroom for one reflection refinement pass
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
            if reflect_enabled and not reflected:
                draft = "\n\n".join(text_parts).strip()
                full_query = messages[0]["content"]  # context + current question
                passed, feedback = _reflect(full_query, draft)
                if not passed:
                    reflected = True
                    tool_call_log.append({"tool": "reflect", "query": f"Refining — {feedback}"})
                    # Append the draft answer then inject the refinement request.
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "user", "content": (
                        f"[Reflection] Your answer is incomplete.\n{feedback}\n\n"
                        "Please use wiki_search and/or web_search with more targeted queries "
                        "to fill the gaps, then provide a complete, revised answer."
                    )})
                    text_parts = []  # reset — the refined answer replaces the draft
                    continue  # re-enter the loop instead of breaking
            break  # either passed reflection or already reflected once

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
                tool_call_log.append({"tool": "ask_user", "query": question})
                session_store.save_planner_state(session_id, messages, block.id)
                return question, True, tool_call_log

            elif block.name == "wiki_search":
                query = block.input["query"]
                tool_call_log.append({"tool": "wiki_search", "query": query})
                result = _wiki_search(query, session_id, context)

            elif block.name == "web_search":
                query = block.input["query"]
                tool_call_log.append({"tool": "web_search", "query": query})
                result = _run_search(query)
                session_store.append_web_search(session_id, query, result)

            elif block.name == "fetch_url":
                url = block.input["url"]
                tool_call_log.append({"tool": "fetch_url", "query": url})
                result = fetch_url(url)

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

    return "\n\n".join(text_parts).strip(), False, tool_call_log


async def run(session_id: str, user_message: str, context: str = "", reflect: bool = True) -> ChatResponse:
    logger.info("[planner] Starting (session=%s, reflect=%s)", session_id[:8], reflect)
    logger.info("[planner] Query: %s", user_message)

    answer = ""
    is_question = False
    tool_calls: list[dict] = []
    try:
        answer, is_question, tool_calls = await asyncio.to_thread(
            _run_planner, session_id, user_message, context, reflect
        )
    except Exception:
        logger.exception("[planner] failed")

    if not answer:
        answer = "I couldn't find enough information to answer that. Could you rephrase?"

    logger.info("[planner] Done (is_question=%s, tool_calls=%d)", is_question, len(tool_calls))
    return ChatResponse(
        answer=answer,
        sources=[],
        offer_save=False,
        session_id=session_id,
        intent="planner",
        tool_calls=tool_calls,
    )
