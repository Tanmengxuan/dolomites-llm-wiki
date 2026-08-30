import json
import logging
from pathlib import Path

from app.anthropic_client import get_client
from app.models import ChatResponse
from app.skills.ingest.prompts import INGEST_SYSTEM

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent

_TOOLS = [
    {
        "name": "read_file",
        "description": "Read the full contents of a file inside the project directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path relative to project root, e.g. raw/source.md or wiki/index.md",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Create or fully overwrite a file with the given content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path relative to project root.",
                },
                "content": {
                    "type": "string",
                    "description": "Full content to write.",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": (
            "Replace the first occurrence of old_string with new_string in a file. "
            "old_string must match exactly (including whitespace). "
            "Use read_file first to confirm the exact text to replace."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to project root."},
                "old_string": {"type": "string", "description": "Exact text to find."},
                "new_string": {"type": "string", "description": "Replacement text."},
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
]


def _resolve(path: str) -> Path:
    resolved = (PROJECT_ROOT / path).resolve()
    resolved.relative_to(PROJECT_ROOT.resolve())  # raises ValueError if outside root
    return resolved


def _execute_tool(name: str, tool_input: dict) -> str:
    try:
        if name == "read_file":
            p = _resolve(tool_input["path"])
            if not p.exists():
                return f"Error: file not found: {tool_input['path']}"
            return p.read_text(encoding="utf-8")

        elif name == "write_file":
            path_str: str = tool_input["path"]
            if path_str.startswith("raw/") and not Path(path_str).name.startswith("draft-"):
                return "Error: raw/ source files are immutable. To create a new draft, use a filename starting with 'draft-'."
            p = _resolve(path_str)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(tool_input["content"], encoding="utf-8")
            logger.info("[ingest_agent] wrote %s", path_str)
            return f"OK: wrote {path_str}"

        elif name == "edit_file":
            path_str = tool_input["path"]
            if path_str.startswith("raw/"):
                return "Error: raw/ files are immutable."
            p = _resolve(path_str)
            if not p.exists():
                return f"Error: file not found: {path_str}"
            original = p.read_text(encoding="utf-8")
            old = tool_input["old_string"]
            if old not in original:
                return f"Error: old_string not found in {path_str}"
            p.write_text(original.replace(old, tool_input["new_string"], 1), encoding="utf-8")
            logger.info("[ingest_agent] edited %s", path_str)
            return f"OK: edited {path_str}"

        else:
            return f"Error: unknown tool {name}"

    except ValueError:
        return "Error: path escapes project root"
    except Exception as e:
        return f"Error: {e}"


def _run_ingest(prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    text_parts: list[str] = []

    for turn in range(40):  # hard cap on turns
        response = get_client().messages.create(
            model="claude-sonnet-4-6",
            max_tokens=8192,
            system=INGEST_SYSTEM,
            tools=_TOOLS,
            messages=messages,
        )
        logger.info("[ingest_agent] turn %d — stop_reason=%s", turn + 1, response.stop_reason)

        # Collect any text blocks from this turn
        for block in response.content:
            if block.type == "text" and block.text:
                text_parts.append(block.text)

        if response.stop_reason == "end_turn":
            break

        if response.stop_reason != "tool_use":
            break

        # Execute all tool calls and build the next messages
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                logger.info("[ingest_agent] tool_call: %s(%s)", block.name, json.dumps(block.input)[:120])
                result = _execute_tool(block.name, block.input)
                logger.info("[ingest_agent] tool_result: %s", result[:120])
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        messages.append({"role": "user", "content": tool_results})

    return "\n\n".join(text_parts).strip()


async def run(session_id: str, user_message: str, context: str = "", web_search_history: str = "") -> ChatResponse:
    logger.info("[ingest_agent] Starting (session=%s)", session_id[:8])
    logger.info("[ingest_agent] Instruction: %s", user_message)

    parts = []
    if context:
        parts.append(context)
    if web_search_history:
        parts.append(web_search_history)
    parts.append(f"Current instruction: {user_message}")
    prompt = "\n\n".join(parts)

    answer = ""
    try:
        import asyncio
        answer = await asyncio.to_thread(_run_ingest, prompt)
    except Exception:
        logger.exception("[ingest_agent] ingest failed")

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
