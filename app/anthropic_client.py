import os
from datetime import date

import anthropic
from dotenv import load_dotenv

load_dotenv()

_client: anthropic.Anthropic | None = None


def with_date(system: str) -> str:
    """Prepend today's date to a system prompt so agents never need to search for it."""
    d = date.today()
    today = f"{d.strftime('%B')} {d.day}, {d.year}"  # e.g. "September 1, 2026"
    return f"Today's date is {today}.\n\n{system}"


def _build_http_client():
    """Return an http client with trust_env=False to suppress stale SSL_CERT_FILE env vars."""
    try:
        import httpx2
        return httpx2.Client(trust_env=False)
    except ImportError:
        import httpx
        return httpx.Client(trust_env=False)


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            http_client=_build_http_client(),
        )
    return _client
