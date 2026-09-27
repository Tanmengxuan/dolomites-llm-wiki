import logging
import re

import certifi
import httpx

logger = logging.getLogger(__name__)

_FETCH_TIMEOUT = 20
_FETCH_MAX_CHARS = 12000
_JINA_PREFIX = "https://r.jina.ai/"
_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

FETCH_URL_TOOL = {
    "name": "fetch_url",
    "description": (
        "Fetch the full live content of a specific URL. Use when search snippets are insufficient — "
        "e.g. live flight/transport status, detailed restaurant or trail info, booking pages, "
        "or any page where the snippet was too brief to answer the query fully."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"url": {"type": "string", "description": "The full URL to fetch."}},
        "required": ["url"],
    },
}


def _strip_html(html: str) -> str:
    html = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', html)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def fetch_url(url: str) -> str:
    headers = {"User-Agent": _BROWSER_UA}
    # trust_env=False: ignore HTTP_PROXY/HTTPS_PROXY env vars that may point to
    # non-existent socket files on the server, causing [Errno 2] failures.
    # verify=certifi.where(): use the bundled CA store rather than the system path.
    client_kwargs = dict(
        follow_redirects=True,
        timeout=_FETCH_TIMEOUT,
        verify=certifi.where(),
        trust_env=False,
    )
    try:
        resp = httpx.get(url, headers=headers, **client_kwargs)
        text = _strip_html(resp.text)
        if len(text) > 500:
            logger.info("[fetch_url] direct OK — %d chars from %s", len(text), url)
            return text[:_FETCH_MAX_CHARS]
        logger.info("[fetch_url] direct too short (%d chars) — trying Jina", len(text))
    except Exception as e:
        logger.warning("[fetch_url] direct failed for %s: %s", url, e)
    try:
        resp = httpx.get(
            _JINA_PREFIX + url,
            headers={"Accept": "text/plain", "User-Agent": _BROWSER_UA},
            **client_kwargs,
        )
        text = resp.text.strip()
        logger.info("[fetch_url] Jina OK — %d chars from %s", len(text), url)
        return text[:_FETCH_MAX_CHARS]
    except Exception as e:
        logger.warning("[fetch_url] Jina failed for %s: %s", url, e)
    return f"Could not fetch content from {url}."
