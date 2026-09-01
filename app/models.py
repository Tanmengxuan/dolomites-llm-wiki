import re
from typing import Literal
from pydantic import BaseModel, field_validator


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    offer_save: bool
    session_id: str
    intent: Literal["wiki_qa", "web_search", "ingest", "planner"] = "wiki_qa"


class SaveRequest(BaseModel):
    session_id: str
    page_title: str
    page_slug: str
    content: str
    section: Literal["planning", "activities", "areas", "logistics"] | None = None

    @field_validator("page_slug")
    @classmethod
    def slug_format(cls, v: str) -> str:
        if not re.match(r"^[a-z0-9-]+$", v):
            raise ValueError("Slug must contain only lowercase letters, digits, and hyphens")
        return v


class SaveResponse(BaseModel):
    success: bool
    page_path: str
    message: str


class SessionResponse(BaseModel):
    session_id: str
