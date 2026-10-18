"""Request and response models of the public API."""

import base64
import binascii
from typing import Literal

from pydantic import BaseModel, Field, model_validator

THREAD_ID_PATTERN = r"^[A-Za-z0-9_-]{3,100}$"
IMAGE_TYPES = ("image/jpeg", "image/png", "image/webp")
# Base64 inflates by 4/3; the decoded size is checked against settings in the service layer.
MAX_IMAGE_DATA_URL_CHARS = 7_000_000


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    thread_id: str | None = Field(
        default=None,
        pattern=THREAD_ID_PATTERN,
        description="Conversation id; omit to start a new conversation.",
    )
    image_data: str | None = Field(
        default=None,
        max_length=MAX_IMAGE_DATA_URL_CHARS,
        description="Optional data URL (JPEG, PNG or WebP) of an image to describe.",
    )

    @model_validator(mode="after")
    def validate_payload(self) -> "ChatRequest":
        self.message = self.message.strip()
        if not self.message:
            raise ValueError("Message cannot be empty.")
        if self.image_data:
            image_bytes(self.image_data)
        return self


def image_bytes(data_url: str) -> bytes:
    """Decode a data URL, accepting only the supported image types."""
    try:
        header, payload = data_url.split(",", 1)
        mime = header.split(";")[0].removeprefix("data:")
        raw = base64.b64decode(payload, validate=True)
    except (ValueError, binascii.Error):
        raise ValueError("Image must be a base64 data URL.") from None
    if mime not in IMAGE_TYPES:
        raise ValueError("Only JPEG, PNG and WebP images are supported.")
    return raw


class Citation(BaseModel):
    id: str
    type: Literal["kb", "pubmed", "fda", "medlineplus", "web"]
    title: str
    label: str
    page: int | None = None
    url: str = ""
    score: float | None = None
    excerpt: str


class ToolCall(BaseModel):
    tool: str
    query: str
    results: int


class TraceStep(BaseModel):
    node: str
    label: str
    detail: str = ""
    model: str = ""  # LLM steps: provider and model that answered through the gateway


class ChatResponse(BaseModel):
    success: Literal[True] = True
    thread_id: str
    answer: str
    spoken: str
    language: str
    urgency: Literal["emergency", "urgent", "routine"]
    in_scope: bool
    source_used: str
    grounding: Literal["supported", "partial", "unsupported", "skipped"]
    usefulness: Literal["useful", "not_useful", "skipped"] = "skipped"
    triage_reason: str = ""
    image_description: str = ""
    citations: list[Citation] = []
    research_calls: list[ToolCall] = []
    trace: list[TraceStep] = []


class ConversationSummary(BaseModel):
    thread_id: str
    title: str
    turns: int
    created_at: str
    updated_at: str


class ConversationList(BaseModel):
    conversations: list[ConversationSummary]


class Message(BaseModel):
    question: str
    answer: str


class ConversationDetail(ConversationSummary):
    messages: list[Message]


class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000)
    language: str | None = Field(default=None, max_length=5)


class TranscriptionResponse(BaseModel):
    text: str


class ErrorResponse(BaseModel):
    success: Literal[False] = False
    error: str
    code: str


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    models: dict[str, str] = {}
    checks: dict[str, str] = {}
