import uuid

_store: dict[str, list[dict]] = {}

_MAX_MESSAGES = 20


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


def clear(session_id: str) -> None:
    _store.pop(session_id, None)


def _trim(session_id: str) -> None:
    history = _store.get(session_id, [])
    while len(history) > _MAX_MESSAGES:
        history.pop(0)
        history.pop(0)
