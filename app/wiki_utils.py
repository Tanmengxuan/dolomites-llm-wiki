import re
import logging
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

WIKI_ROOT = Path(__file__).parent.parent / "wiki"

_EXCLUDED = {"index.md", "log.md"}

_SECTION_HEADERS = {
    "planning": "## Planning & Overview",
    "activities": "## Activities",
    "areas": "## Areas",
    "logistics": "## Logistics",
}


def read_index() -> str:
    return (WIKI_ROOT / "index.md").read_text(encoding="utf-8")


def list_wiki_pages() -> list[str]:
    return [
        p.stem
        for p in WIKI_ROOT.glob("*.md")
        if p.name not in _EXCLUDED
    ]


def read_pages(slugs: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for slug in slugs:
        path = WIKI_ROOT / f"{slug}.md"
        if not path.exists():
            logger.warning("Wiki page not found: %s", path)
            continue
        result[slug] = path.read_text(encoding="utf-8")
    return result


def save_page(slug: str, title: str, content: str, sources: list[str]) -> Path:
    path = WIKI_ROOT / f"{slug}.md"
    if path.exists():
        raise FileExistsError(f"Page already exists: {slug}.md")

    summary = _first_sentence(content)
    sources_str = ", ".join(sources) if sources else "none"
    links = _extract_links(content)
    related = "\n".join(f"- [[{link}]]" for link in links) if links else "_None identified._"
    today = date.today().isoformat()

    page = (
        f"# {title}\n\n"
        f"**Summary**: {summary}\n\n"
        f"**Sources**: {sources_str}\n\n"
        f"**Last updated**: {today}\n\n"
        f"---\n\n"
        f"{content.strip()}\n\n"
        f"## Related pages\n\n"
        f"{related}\n"
    )
    path.write_text(page, encoding="utf-8")
    return path


def append_log(slug: str, title: str, summary: str) -> None:
    log_path = WIKI_ROOT / "log.md"
    today = date.today().isoformat()
    entry = (
        f"\n## {today} — New page: {title}\n\n"
        f"**Saved via**: chatbot\n\n"
        f"**Page created**: `{slug}.md` — {summary}\n"
    )
    with log_path.open("a", encoding="utf-8") as f:
        f.write(entry)


def update_index(slug: str, title: str, description: str, section: str | None) -> None:
    index_path = WIKI_ROOT / "index.md"
    content = index_path.read_text(encoding="utf-8")
    new_row = f"| [[{slug}]] | {description} |\n"

    header = _SECTION_HEADERS.get(section or "", "") if section else ""

    if header and header in content:
        # Find the table under this header and append before the next blank line after the table
        idx = content.index(header)
        # Find end of the table block after this header
        block_start = content.index("| [[", idx) if "| [[" in content[idx:] else -1
        if block_start != -1:
            block_start = idx + content[idx:].index("| [[")
            # Walk forward to end of table
            lines = content[block_start:].split("\n")
            insert_after = block_start
            for line in lines:
                if line.startswith("|"):
                    insert_after += len(line) + 1
                elif insert_after > block_start:
                    break
            content = content[:insert_after] + new_row + content[insert_after:]
        else:
            content += f"\n{header}\n\n| Page | Description |\n|------|-------------|\n{new_row}"
    else:
        content += f"\n## Chatbot-Generated\n\n| Page | Description |\n|------|-------------|\n{new_row}"

    index_path.write_text(content, encoding="utf-8")


def _first_sentence(text: str) -> str:
    clean = re.sub(r"#.*\n", "", text).strip()
    match = re.search(r"[^.!?]*[.!?]", clean)
    sentence = match.group(0).strip() if match else clean[:120]
    return sentence[:200]


def _extract_links(text: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"\[\[([^\]]+)\]\]", text)))
