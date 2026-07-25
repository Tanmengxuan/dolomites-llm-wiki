import asyncio
import logging
import sys
from pathlib import Path

from claude_agent_sdk import query
from claude_agent_sdk.types import AssistantMessage, ClaudeAgentOptions, ResultMessage, TextBlock

from app.models import ChatResponse
from app.skills.ingest.prompts import INGEST_SYSTEM

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


async def _collect(prompt: str, options: ClaudeAgentOptions) -> tuple[str | None, list[str]]:
    result_text: str | None = None
    assistant_parts: list[str] = []
    async for msg in query(prompt=prompt, options=options):
        if isinstance(msg, ResultMessage):
            logger.info(
                "[ingest_agent] ResultMessage — turns: %d, cost: $%s",
                msg.num_turns,
                msg.total_cost_usd,
            )
            if msg.result:
                result_text = msg.result
        elif isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock) and block.text:
                    assistant_parts.append(block.text)
    return result_text, assistant_parts


def _run_in_proactor(prompt: str, options: ClaudeAgentOptions) -> tuple[str | None, list[str]]:
    """Run query() in a thread-local ProactorEventLoop to bypass Windows SelectorEventLoop."""
    if sys.platform == "win32":
        loop = asyncio.ProactorEventLoop()
    else:
        loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_collect(prompt, options))
    finally:
        loop.close()


async def run(session_id: str, user_message: str, context: str = "") -> ChatResponse:
    logger.info("[ingest_agent] Starting (session=%s)", session_id[:8])
    logger.info("[ingest_agent] Instruction: %s", user_message)

    prompt = (
        f"{context}\n\nCurrent instruction: {user_message}"
        if context
        else user_message
    )

    options = ClaudeAgentOptions(
        tools=["Read", "Write", "Edit"],
        system_prompt=INGEST_SYSTEM,
        permission_mode="bypassPermissions",
        cwd=PROJECT_ROOT,
        max_turns=20,
    )

    result_text: str | None = None
    assistant_parts: list[str] = []

    try:
        result_text, assistant_parts = await asyncio.to_thread(
            _run_in_proactor, prompt, options
        )
    except Exception:
        logger.exception("[ingest_agent] query failed")

    answer = result_text or " ".join(assistant_parts).strip()
    if not answer:
        answer = "Ingest completed. Check the wiki/ folder for updated pages."

    logger.info("[ingest_agent] Done")
    return ChatResponse(
        answer=answer,
        sources=[],
        offer_save=False,
        session_id=session_id,
        intent="ingest",
    )
