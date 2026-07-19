import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import claude_client, session_store, wiki_utils
from .models import ChatRequest, ChatResponse, SaveRequest, SaveResponse, SessionResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

app = FastAPI(title="Dolomites Wiki Chatbot")

_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(str(_STATIC_DIR / "index.html"))


@app.post("/session", response_model=SessionResponse)
def create_session():
    sid = session_store.new_session_id()
    session_store.get_or_create(sid)
    return SessionResponse(session_id=sid)


@app.delete("/session/{session_id}", status_code=204)
def delete_session(session_id: str):
    session_store.clear(session_id)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=422, detail="Message cannot be empty")
    return await claude_client.run_chat_turn(req.session_id, req.message.strip())


@app.post("/save", response_model=SaveResponse)
def save(req: SaveRequest):
    sources = session_store.get_or_create(req.session_id)
    # Derive source page slugs from the last assistant turn context (best-effort)
    # The client sends the content; we trust the slug/title from the request
    try:
        path = wiki_utils.save_page(
            slug=req.page_slug,
            title=req.page_title,
            content=req.content,
            sources=[],  # source pages not re-derived here; client may pass them separately
        )
    except FileExistsError:
        raise HTTPException(status_code=409, detail="A page with this slug already exists. Choose a different slug.")

    summary = wiki_utils._first_sentence(req.content)
    wiki_utils.append_log(req.page_slug, req.page_title, summary)
    wiki_utils.update_index(req.page_slug, req.page_title, summary, req.section)

    return SaveResponse(
        success=True,
        page_path=str(path.relative_to(path.parent.parent)),
        message="Page saved and wiki updated.",
    )
