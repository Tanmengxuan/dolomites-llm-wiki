import uuid

_store: dict[str, list[dict]] = {}  # kept for session lifecycle (create/clear)
_exchange_store: dict[str, list[dict]] = {}
_web_search_store: dict[str, list[dict]] = {}
# Stores in-progress planner message history when ask_user pauses the loop.
# Value: {"messages": list, "ask_user_tool_id": str}
_planner_state: dict[str, dict] = {}
_web_error_sessions: set[str] = set()   # sessions where next web_search should fail (test only)
_wiki_error_sessions: set[str] = set()  # sessions where next wiki_qa should fail (test only)

_MAX_EXCHANGES = 10
_MAX_RESPONSE_CHARS = 2500
_MAX_WEB_SEARCHES = 10
_MAX_WEB_SEARCH_CHARS = 3000  # web search answers are longer; keep more for ingest


def new_session_id() -> str:
    return str(uuid.uuid4())


def get_or_create(session_id: str) -> list[dict]:
    if session_id not in _store:
        _store[session_id] = []
    return _store[session_id]


def append_exchange(session_id: str, intent: str, user: str, assistant: str) -> None:
    if session_id not in _exchange_store:
        _exchange_store[session_id] = []
    _exchange_store[session_id].append({
        "intent": intent,
        "user": user,
        "assistant": assistant[:_MAX_RESPONSE_CHARS],
    })
    if len(_exchange_store[session_id]) > _MAX_EXCHANGES:
        _exchange_store[session_id].pop(0)


def get_context_string(session_id: str) -> str:
    exchanges = _exchange_store.get(session_id, [])
    if not exchanges:
        return ""
    lines = ["Conversation history:"]
    for i, ex in enumerate(exchanges, 1):
        lines.append(f"Turn {i} [{ex['intent']}]")
        lines.append(f"User: {ex['user']}")
        lines.append(f"Assistant: {ex['assistant']}")
        lines.append("")
    return "\n".join(lines).strip()


def arm_web_error(session_id: str) -> None:
    _web_error_sessions.add(session_id)


def consume_web_error(session_id: str) -> bool:
    if session_id in _web_error_sessions:
        _web_error_sessions.discard(session_id)
        return True
    return False


def arm_wiki_error(session_id: str) -> None:
    _wiki_error_sessions.add(session_id)


def consume_wiki_error(session_id: str) -> bool:
    if session_id in _wiki_error_sessions:
        _wiki_error_sessions.discard(session_id)
        return True
    return False


def save_planner_state(session_id: str, messages: list, ask_user_tool_id: str) -> None:
    _planner_state[session_id] = {"messages": messages, "ask_user_tool_id": ask_user_tool_id}


def has_planner_state(session_id: str) -> bool:
    return session_id in _planner_state


def pop_planner_state(session_id: str) -> dict | None:
    return _planner_state.pop(session_id, None)


def append_web_search(session_id: str, user: str, assistant: str) -> None:
    if session_id not in _web_search_store:
        _web_search_store[session_id] = []
    _web_search_store[session_id].append({
        "user": user,
        "assistant": assistant[:_MAX_WEB_SEARCH_CHARS],
    })
    if len(_web_search_store[session_id]) > _MAX_WEB_SEARCHES:
        _web_search_store[session_id].pop(0)


def get_web_search_history(session_id: str) -> str:
    results = _web_search_store.get(session_id, [])
    if not results:
        return ""
    lines = ["Web search results from this session:"]
    for i, r in enumerate(results, 1):
        lines.append(f"\n--- Web Search {i} ---")
        lines.append(f"Query: {r['user']}")
        lines.append(f"Result:\n{r['assistant']}")
    return "\n".join(lines)


def clear(session_id: str) -> None:
    _store.pop(session_id, None)
    _exchange_store.pop(session_id, None)
    _web_search_store.pop(session_id, None)
    _planner_state.pop(session_id, None)


