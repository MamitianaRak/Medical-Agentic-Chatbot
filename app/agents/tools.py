"""Interfaces the agents use to reach evidence.

Production implementations live in `app.integrations`: Pinecone for the private
knowledge base and an MCP client for the research tools. Tests plug in
in-memory fakes, so the whole graph runs without network access.
"""

from typing import Protocol

from app.agents.state import Passage

# What the research planner may call, with the description it reads to choose.
TOOL_CATALOG = {
    "pubmed": "PubMed peer-reviewed literature: current evidence, treatments, research, "
    "guidelines. Query: short English keywords.",
    "drug_label": "Official FDA drug label: indications, warnings, contraindications, "
    "interactions, adverse reactions. Query: ONE drug name (generic or brand) only.",
    "medlineplus": "MedlinePlus (US National Library of Medicine) patient-friendly topic "
    "summaries: symptoms, causes, treatment, prevention. Query: a short English disease or "
    "symptom name.",
    "web": "General web search: recent news, official health agency pages, anything the other "
    "tools do not cover. Query: a full English search query.",
}


class KnowledgeBase(Protocol):
    def search(self, query: str) -> list[Passage]: ...


class ResearchTools(Protocol):
    def available(self) -> list[str]:
        """Names from TOOL_CATALOG that are configured and may be called."""
        ...

    def run(self, tool: str, query: str) -> list[Passage]:
        """Call one tool; raises on failure (the node degrades that call only)."""
        ...
