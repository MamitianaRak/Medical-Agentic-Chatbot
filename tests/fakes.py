"""Network-free stand-ins for Gemini, Pinecone, Tavily and Workers AI."""

import json
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from app.agents.state import Passage


def kb_passage(page: int, text: str, score: float = 0.6) -> Passage:
    return {
        "id": f"kb-{page}",
        "source": "kb",
        "title": "OpenStax Medical-Surgical Nursing (2024)",
        "text": text,
        "score": score,
        "page": page,
        "url": "",
        "label": f"PDF p.{page}",
    }


def pubmed_passage(pmid: str, text: str) -> Passage:
    return {
        "id": f"pubmed-{pmid}",
        "source": "pubmed",
        "title": "Advances in asthma (J Allergy Clin Immunol, 2021)",
        "text": text,
        "score": None,
        "page": None,
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        "label": f"PMID:{pmid}",
    }


ASTHMA = kb_passage(394, "Asthma attacks are triggered by allergens, smoke and exercise.", 0.62)
ASTHMA_DIARY = kb_passage(398, "A symptom diary helps identify asthma triggers.", 0.51)


class FakeLLM(BaseChatModel):
    """A real LangChain chat model that answers by recognizing the agent prompt."""

    triage: dict[str, Any] = Field(
        default_factory=lambda: {
            "urgency": "routine",
            "in_scope": True,
            "language": "en",
            "needs_current_info": False,
            "search_query": "asthma triggers",
            "reason": "Educational question.",
        }
    )
    relevant: list[int] = Field(default_factory=lambda: [1, 2])
    # Per-round relevance verdicts, used before `relevant` (e.g. KB round, then web round).
    relevant_rounds: list[list[int]] = Field(default_factory=list)
    # Successive grounding / usefulness verdicts; "supported" / useful once exhausted.
    grounding: list[str] = Field(default_factory=list)
    usefulness: list[bool] = Field(default_factory=list)
    # Research plan returned by the planner (tool names must be available ones).
    plan: list[dict[str, str]] = Field(
        default_factory=lambda: [{"tool": "pubmed", "query": "asthma biologics"}]
    )
    malformed: set[str] = Field(default_factory=set)
    # Kinds whose FIRST reply only is malformed (to test the retry).
    malformed_once: set[str] = Field(default_factory=set)
    fail_with: Any = None
    prompts: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-medical-llm"

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs):
        if self.fail_with is not None:
            raise self.fail_with
        prompt = "\n".join(str(m.content) for m in messages)
        self.prompts.append(prompt)
        message = AIMessage(
            content=self._reply(prompt),
            # What the LLM gateway reports: the provider and model that answered.
            response_metadata={"model_name": "fake-model", "provider": "fake"},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def calls(self, marker: str) -> int:
        return sum(marker in prompt for prompt in self.prompts)

    def _reply(self, prompt: str) -> str:
        handlers = {
            "triage step": ("triage", self._triage_reply),
            "relevance grader": ("grade", self._grade_reply),
            "research planner": ("plan", self._plan_reply),
            "query rewriter": ("rewrite", self._rewrite_reply),
            "grounding checker": ("grounding", self._grounding_reply),
            "usefulness checker": ("usefulness", self._usefulness_reply),
            "Write the answer": ("generate", self._answer_reply),
        }
        for marker, (kind, handler) in handlers.items():
            if marker in prompt:
                if kind in self.malformed_once:
                    self.malformed_once.discard(kind)
                    return "this is not json"
                return "this is not json" if kind in self.malformed else json.dumps(handler(prompt))
        return "{}"

    def _triage_reply(self, prompt: str) -> dict:
        return self.triage

    def _grade_reply(self, prompt: str) -> dict:
        return {"relevant": self.relevant_rounds.pop(0) if self.relevant_rounds else self.relevant}

    def _plan_reply(self, prompt: str) -> dict:
        return {"calls": self.plan}

    def _rewrite_reply(self, prompt: str) -> dict:
        return {"query": "asthma exacerbation triggers"}

    def _usefulness_reply(self, prompt: str) -> dict:
        useful = self.usefulness.pop(0) if self.usefulness else True
        return {"useful": useful, "missing": "" if useful else "treatment options"}

    def _grounding_reply(self, prompt: str) -> dict:
        status = self.grounding.pop(0) if self.grounding else "supported"
        claims = ["Asthma is caused by cold weather."] if status == "unsupported" else []
        return {"status": status, "unsupported_claims": claims}

    def _answer_reply(self, prompt: str) -> dict:
        if "grounding check found claims" in prompt:
            return {"answer": "Revised answer [PDF p.394].", "spoken": "Revised summary."}
        return {
            "answer": "Asthma attacks are triggered by allergens and smoke [PDF p.394].",
            "spoken": "Allergens and smoke can trigger asthma attacks.",
        }


@dataclass
class FakeKnowledgeBase:
    passages: list[Passage] = field(default_factory=lambda: [ASTHMA, ASTHMA_DIARY])
    queries: list[str] = field(default_factory=list)

    def search(self, query: str) -> list[Passage]:
        self.queries.append(query)
        return list(self.passages)


@dataclass
class FakeResearchTools:
    """In-memory stand-in for the MCP research server."""

    passages: list[Passage] = field(
        default_factory=lambda: [pubmed_passage("34655640", "Biologics reduce asthma attacks.")]
    )
    tools: list[str] = field(default_factory=lambda: ["pubmed", "drug_label", "medlineplus"])
    fail: set[str] = field(default_factory=set)
    calls: list[tuple[str, str]] = field(default_factory=list)

    def available(self) -> list[str]:
        return list(self.tools)

    def run(self, tool: str, query: str) -> list[Passage]:
        self.calls.append((tool, query))
        if tool in self.fail:
            raise RuntimeError(f"{tool} is down")
        return list(self.passages)


@dataclass
class FakeMedia:
    fail: bool = False
    spoken: list[tuple[str, str]] = field(default_factory=list)

    def transcribe(self, audio: bytes) -> str:
        if self.fail:
            raise RuntimeError("workers ai down")
        return "What triggers asthma?"

    def describe_image(self, data_url: str) -> str:
        if self.fail:
            raise RuntimeError("workers ai down")
        return "A red, raised rash on the forearm."

    def synthesize(self, text: str, language: str) -> bytes:
        if self.fail:
            raise RuntimeError("workers ai down")
        self.spoken.append((text, language))
        return b"ID3-fake-mp3"
