import uuid

_store: dict[str, list[dict]] = {}
_exchange_store: dict[str, list[dict]] = {}

_MAX_MESSAGES = 20
_MAX_EXCHANGES = 10
_MAX_RESPONSE_CHARS = 300


def new_session_id() -> str:
    return str(uuid.uuid4())


def get_or_create(session_id: str) -> list[dict]:
    if session_id not in _store:
        _store[session_id] = []
    return _store[session_id]


def append_user(session_id: str, content: str) -> None:
    history = get_or_create(session_id)
    history.append({"role": "user", "content": content})
    _trim(session_id)


def append_assistant(session_id: str, content: str) -> None:
    history = get_or_create(session_id)
    history.append({"role": "assistant", "content": content})
    _trim(session_id)


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


def clear(session_id: str) -> None:
    _store.pop(session_id, None)
    _exchange_store.pop(session_id, None)


def _trim(session_id: str) -> None:
    history = _store.get(session_id, [])
    while len(history) > _MAX_MESSAGES:
        history.pop(0)
        history.pop(0)
