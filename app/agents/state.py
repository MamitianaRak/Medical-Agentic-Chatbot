"""Shared LangGraph state and workflow constants."""

import operator
from typing import Annotated, Literal, TypedDict

Urgency = Literal["emergency", "urgent", "routine"]
SourceUsed = Literal["private_kb", "research", "mixed", "emergency", "out_of_scope", "insufficient"]
Grounding = Literal["supported", "partial", "unsupported", "skipped"]
EvidenceSource = Literal["kb", "pubmed", "fda", "medlineplus", "web"]

SUPPORTED_LANGUAGES = ("en", "fr", "es", "zh", "ja", "ko")


class Passage(TypedDict):
    """A piece of evidence: a chunk of the private knowledge base or a research tool result."""

    id: str
    source: EvidenceSource
    title: str
    text: str
    # Similarity score for retrieved chunks and web results; None when the API has none.
    score: float | None
    page: int | None
    url: str
    # Short label the answer cites, e.g. "PDF p.394", "PMID:31678040", "FDA: ibuprofen".
    label: str


class Turn(TypedDict):
    question: str
    answer: str


class TraceStep(TypedDict):
    node: str
    detail: str


class ToolCall(TypedDict):
    tool: str
    query: str
    results: int


class MedicalState(TypedDict, total=False):
    # Input
    question: str
    image_description: str

    # Triage (one LLM call: urgency, scope, language, standalone question)
    urgency: Urgency
    in_scope: bool
    language: str
    needs_current_info: bool
    search_query: str
    triage_reason: str

    # Evidence (corrective loop: KB, then research tools, then query rewrites)
    candidates: list[Passage]
    passages: list[Passage]
    planned_calls: list[dict[str, str]]
    research_rounds: int
    research_calls: list[ToolCall]
    query_rewrites: int
    missing_info: str

    # Answer and reflection
    answer: str
    spoken: str
    source_used: SourceUsed
    grounding: Grounding
    unsupported_claims: list[str]
    grounding_retries: int
    usefulness: Literal["useful", "not_useful", "skipped"]

    trace: list[TraceStep]
    # Accumulates across turns of the same thread (persisted by the checkpointer).
    memory: Annotated[list[Turn], operator.add]


NODE_LABELS = {
    "vision": "Image observation",
    "triage": "Emergency triage",
    "emergency": "Emergency response",
    "out_of_scope": "Scope guardrail",
    "retrieve": "Knowledge retrieval",
    "grade_documents": "Relevance grading",
    "plan_research": "Research planning",
    "call_tools": "MCP tool calls",
    "rewrite_query": "Query rewrite",
    "generate": "Answer generation",
    "check_grounding": "Grounding check",
    "revise": "Answer revision",
    "check_usefulness": "Usefulness check",
    "insufficient": "Insufficient evidence",
    "finalize": "Conversation saved",
}


def initial_state(question: str, image_description: str = "") -> MedicalState:
    """Per-turn fields are reset; memory=[] adds nothing, so history is kept."""
    return {
        "question": question,
        "image_description": image_description,
        "urgency": "routine",
        "in_scope": True,
        "language": "en",
        "needs_current_info": False,
        "search_query": question,
        "triage_reason": "",
        "candidates": [],
        "passages": [],
        "planned_calls": [],
        "research_rounds": 0,
        "research_calls": [],
        "query_rewrites": 0,
        "missing_info": "",
        "answer": "",
        "spoken": "",
        "source_used": "insufficient",
        "grounding": "skipped",
        "unsupported_claims": [],
        "grounding_retries": 0,
        "usefulness": "skipped",
        "trace": [],
        "memory": [],
    }
