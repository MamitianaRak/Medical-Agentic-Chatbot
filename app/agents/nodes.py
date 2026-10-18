"""Agent nodes of the medical Corrective + Reflective RAG workflow.

Each node is a method so the LLM, the knowledge base and the research tools are
injected once, which lets tests run the whole graph with fakes and no network access.
"""

import logging
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents import prompts
from app.agents.state import (
    SUPPORTED_LANGUAGES,
    MedicalState,
    Passage,
    ToolCall,
    TraceStep,
)
from app.agents.text import (
    content_text,
    format_evidence,
    format_passages,
    history_text,
    json_from_llm,
)
from app.agents.tools import TOOL_CATALOG, KnowledgeBase, ResearchTools

logger = logging.getLogger(__name__)

# Last-resort safety net: if the triage call fails, these phrases still trigger the
# emergency path instead of a normal answer.
RED_FLAGS = re.compile(
    r"chest pain|can'?t breathe|cannot breathe|not breathing|unconscious|stroke|seizure|"
    r"overdose|suicid|kill myself|heavy bleeding|anaphyla|"
    r"douleur (?:à la |dans la )?poitrine|douleur thoracique|ne respire plus|"
    r"inconscient|avc|convulsion|me suicider|saigne beaucoup",
    re.IGNORECASE,
)

URGENCY_LEVELS = {"emergency", "urgent", "routine"}
GROUNDING_STATUSES = {"supported", "partial", "unsupported"}


def _step(state: MedicalState, node: str, detail: str) -> list[TraceStep]:
    return [*state.get("trace", []), {"node": node, "detail": detail}]


def _source_used(passages: list[Passage]) -> str:
    sources = {p["source"] for p in passages}
    if sources == {"kb"}:
        return "private_kb"
    return "mixed" if "kb" in sources else "research"


class MedicalAgents:
    def __init__(
        self,
        llm: BaseChatModel,
        knowledge_base: KnowledgeBase,
        research_tools: ResearchTools | None = None,
        *,
        memory_turns: int = 4,
        max_tool_calls: int = 3,
        max_research_rounds: int = 2,
        max_query_rewrites: int = 1,
        max_grounding_retries: int = 1,
        usefulness_check: bool = True,
    ):
        self.llm = llm
        self.kb = knowledge_base
        self.tools = research_tools
        self.memory_turns = memory_turns
        self.max_tool_calls = max_tool_calls
        self.max_research_rounds = max_research_rounds
        self.max_query_rewrites = max_query_rewrites
        self.max_grounding_retries = max_grounding_retries
        self.usefulness_check = usefulness_check

    def _ask_json(self, system_prompt: str, user_prompt: str, attempts: int = 2) -> dict[str, Any]:
        """One LLM call returning a JSON object, asked again once if the output is malformed.
        Raises ValueError when it stays malformed; provider errors (quota, timeout)
        propagate to the API error mapping."""
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        for attempt in range(1, attempts + 1):
            try:
                return json_from_llm(content_text(self.llm.invoke(messages).content))
            except ValueError as exc:
                if attempt == attempts:
                    raise
                logger.info("Malformed JSON from the LLM, asking again: %s", exc)
        raise AssertionError("unreachable")

    def _history(self, state: MedicalState) -> str:
        return history_text(state.get("memory", []), self.memory_turns) or "(new conversation)"

    def _available_tools(self) -> list[str]:
        return self.tools.available() if self.tools else []

    def _can_research(self, state: MedicalState) -> bool:
        return bool(self._available_tools()) and (
            state.get("research_rounds", 0) < self.max_research_rounds
        )

    # ------------------------------------------------------------------ triage

    def triage(self, state: MedicalState) -> dict[str, Any]:
        """Urgency, scope, language and a standalone search query in one call."""
        question = state["question"]
        try:
            out = self._ask_json(
                prompts.TRIAGE_SYSTEM,
                prompts.TRIAGE_PROMPT.format(
                    history=self._history(state),
                    image=state.get("image_description") or "none",
                    question=question,
                ),
            )
            urgency = str(out.get("urgency", "routine")).lower()
            urgency = urgency if urgency in URGENCY_LEVELS else "routine"
            language = str(out.get("language", "en")).lower()[:2]
            update = {
                "urgency": urgency,
                "in_scope": bool(out.get("in_scope", True)),
                "language": language if language in SUPPORTED_LANGUAGES else "en",
                "needs_current_info": bool(out.get("needs_current_info", False)),
                "search_query": str(out.get("search_query") or question).strip(),
                "triage_reason": str(out.get("reason", "")).strip(),
            }
            fallback = ""
        except ValueError as exc:
            # Fail open on scope, but never miss an obvious emergency.
            logger.warning("Triage fallback used: %s", exc)
            update = {
                "urgency": "emergency" if RED_FLAGS.search(question) else "routine",
                "in_scope": True,
                "language": "en",
                "needs_current_info": False,
                "search_query": question,
                "triage_reason": "Triage unavailable; keyword safety check applied.",
            }
            fallback = " (fallback)"

        scope = "in scope" if update["in_scope"] else "out of scope"
        detail = f"{update['urgency']} · {scope} · {update['language']}{fallback}"
        return {**update, "trace": _step(state, "triage", detail)}

    def emergency(self, state: MedicalState) -> dict[str, Any]:
        language = state.get("language", "en")
        return {
            "answer": prompts.localized(prompts.EMERGENCY_MESSAGES, language),
            "spoken": prompts.localized(prompts.EMERGENCY_SPOKEN, language),
            "source_used": "emergency",
            "passages": [],
            "grounding": "skipped",
            "trace": _step(state, "emergency", "emergency services advised, no retrieval"),
        }

    def out_of_scope(self, state: MedicalState) -> dict[str, Any]:
        message = prompts.localized(prompts.OUT_OF_SCOPE_MESSAGES, state.get("language", "en"))
        return {
            "answer": message,
            "spoken": message,
            "source_used": "out_of_scope",
            "passages": [],
            "grounding": "skipped",
            "trace": _step(state, "out_of_scope", "not a health question"),
        }

    # -------------------------------------------------------------- evidence

    def retrieve(self, state: MedicalState) -> dict[str, Any]:
        candidates = self.kb.search(state.get("search_query") or state["question"])
        return {
            "candidates": candidates,
            "trace": _step(state, "retrieve", f"{len(candidates)} passages"),
        }

    def grade_documents(self, state: MedicalState) -> dict[str, Any]:
        candidates = state.get("candidates", [])
        kept = list(state.get("passages", []))
        if not candidates:
            return {"passages": kept, "trace": _step(state, "grade_documents", "0/0 relevant")}

        try:
            out = self._ask_json(
                prompts.GRADE_SYSTEM,
                prompts.GRADE_PROMPT.format(
                    question=state.get("search_query") or state["question"],
                    documents=format_passages(candidates),
                ),
            )
            numbers = [n for n in out.get("relevant", []) if isinstance(n, int)]
            relevant = [
                candidates[n - 1] for n in dict.fromkeys(numbers) if 1 <= n <= len(candidates)
            ]
            fallback = ""
        except ValueError as exc:
            # The grounding check still guards the answer, so keep every candidate.
            logger.warning("Relevance grading fallback used: %s", exc)
            relevant, fallback = candidates, " (fallback)"

        known = {p["id"] for p in kept}
        kept += [p for p in relevant if p["id"] not in known]
        detail = f"{len(relevant)}/{len(candidates)} relevant{fallback}"
        return {
            "passages": kept,
            "candidates": [],
            "trace": _step(state, "grade_documents", detail),
        }

    def plan_research(self, state: MedicalState) -> dict[str, Any]:
        """The LLM chooses which MCP tools to call, and with which queries."""
        available = self._available_tools()
        previous = state.get("research_calls", [])
        try:
            out = self._ask_json(
                prompts.PLAN_SYSTEM.format(
                    tools="\n".join(f"- {name}: {TOOL_CATALOG[name]}" for name in available),
                    max_calls=self.max_tool_calls,
                ),
                prompts.PLAN_PROMPT.format(
                    question=state["question"],
                    search_query=state.get("search_query") or state["question"],
                    needs_current_info=state.get("needs_current_info", False),
                    missing=state.get("missing_info") or "none",
                    previous_calls="\n".join(f"- {c['tool']}: {c['query']}" for c in previous)
                    or "none",
                    evidence_titles="\n".join(f"- {p['title']}" for p in state.get("passages", []))
                    or "none",
                ),
            )
            done = {(c["tool"], c["query"].lower()) for c in previous}
            calls = [
                {"tool": str(c.get("tool")), "query": str(c.get("query", "")).strip()}
                for c in out.get("calls", [])
                if isinstance(c, dict)
            ]
            calls = [
                c
                for c in calls
                if c["tool"] in available
                and c["query"]
                and (c["tool"], c["query"].lower()) not in done
            ][: self.max_tool_calls]
            fallback = ""
        except ValueError as exc:
            logger.warning("Research planning fallback used: %s", exc)
            calls, fallback = [], " (fallback)"

        if not calls:
            # Patient-level summary is the safest default; the web complements it if enabled.
            query = state.get("search_query") or state["question"]
            calls = [{"tool": t, "query": query} for t in ("medlineplus", "web") if t in available]

        detail = " · ".join(f"{c['tool']}: {c['query']}" for c in calls) + fallback
        return {"planned_calls": calls, "trace": _step(state, "plan_research", detail)}

    def call_tools(self, state: MedicalState) -> dict[str, Any]:
        """Run the planned MCP calls in parallel; a failing tool only loses its own results."""
        calls = state.get("planned_calls", [])

        def run(call: dict[str, str]) -> list[Passage]:
            try:
                return self.tools.run(call["tool"], call["query"]) if self.tools else []
            except Exception as exc:
                logger.warning("Research tool %s failed: %s", call["tool"], exc)
                return []

        with ThreadPoolExecutor(max_workers=max(len(calls), 1)) as pool:
            results = list(pool.map(run, calls))

        candidates, seen = [], set()
        for passages in results:
            for passage in passages:
                if passage["id"] not in seen:
                    seen.add(passage["id"])
                    candidates.append(passage)

        made: list[ToolCall] = [
            {"tool": c["tool"], "query": c["query"], "results": len(r)}
            for c, r in zip(calls, results, strict=True)
        ]
        counts = Counter()
        for call in made:
            counts[call["tool"]] += call["results"]
        detail = " · ".join(f"{tool} ×{n}" for tool, n in counts.items()) or "no calls"
        return {
            "candidates": candidates,
            "research_calls": [*state.get("research_calls", []), *made],
            "research_rounds": state.get("research_rounds", 0) + 1,
            "trace": _step(state, "call_tools", detail),
        }

    def rewrite_query(self, state: MedicalState) -> dict[str, Any]:
        """Corrective step: reformulate the search when the evidence stays insufficient."""
        previous = [state.get("search_query") or state["question"]]
        previous += [c["query"] for c in state.get("research_calls", [])]
        try:
            out = self._ask_json(
                prompts.REWRITE_SYSTEM,
                prompts.REWRITE_PROMPT.format(
                    question=state["question"],
                    previous="\n".join(f"- {q}" for q in dict.fromkeys(previous)),
                    missing=state.get("missing_info") or "relevant evidence",
                ),
            )
            query = str(out.get("query", "")).strip() or previous[0]
            fallback = ""
        except ValueError as exc:
            logger.warning("Query rewrite fallback used: %s", exc)
            query, fallback = state["question"], " (fallback)"

        return {
            "search_query": query,
            "query_rewrites": state.get("query_rewrites", 0) + 1,
            "trace": _step(state, "rewrite_query", f"{query}{fallback}"),
        }

    # ---------------------------------------------------------------- answer

    def _generate(self, state: MedicalState, feedback: str = "") -> dict[str, Any]:
        passages = state.get("passages", [])
        language = prompts.LANGUAGE_NAMES.get(state.get("language", "en"), "English")
        urgency = state.get("urgency", "routine")
        out = self._ask_json(
            prompts.GENERATE_SYSTEM.format(language=language),
            prompts.GENERATE_PROMPT.format(
                urgency=urgency,
                urgent_note=prompts.URGENT_NOTE if urgency == "urgent" else "",
                history=self._history(state),
                image=state.get("image_description") or "none",
                question=state["question"],
                evidence=format_evidence(passages),
                feedback=feedback,
            ),
        )
        answer = str(out.get("answer", "")).strip()
        if not answer:
            raise ValueError("Empty answer from the LLM")
        return {
            "answer": answer,
            "spoken": str(out.get("spoken", "")).strip() or answer,
            "source_used": _source_used(passages),
        }

    def generate(self, state: MedicalState) -> dict[str, Any]:
        update = self._generate(state)
        detail = f"{len(state.get('passages', []))} sources · {state.get('language', 'en')}"
        return {**update, "trace": _step(state, "generate", detail)}

    def revise(self, state: MedicalState) -> dict[str, Any]:
        claims = "\n".join(f"- {claim}" for claim in state.get("unsupported_claims", []))
        feedback = prompts.REVISION_FEEDBACK.format(claims=claims or "- (unspecified)")
        update = self._generate(state, feedback)
        return {
            **update,
            "grounding_retries": state.get("grounding_retries", 0) + 1,
            "trace": _step(state, "revise", "unsupported claims removed"),
        }

    # ------------------------------------------------------------ reflection

    def check_grounding(self, state: MedicalState) -> dict[str, Any]:
        try:
            out = self._ask_json(
                prompts.GROUNDING_SYSTEM,
                prompts.GROUNDING_PROMPT.format(
                    evidence=format_evidence(state.get("passages", [])), answer=state["answer"]
                ),
            )
            status = str(out.get("status", "")).lower()
            if status not in GROUNDING_STATUSES:
                raise ValueError(f"Unknown grounding status {status!r}")
            claims = [str(c) for c in out.get("unsupported_claims", []) if str(c).strip()]
        except ValueError as exc:
            # Unparseable verdict: an unverified answer is never shown as verified.
            logger.warning("Grounding check fallback used: %s", exc)
            status, claims = "unsupported", []

        return {
            "grounding": status,
            "unsupported_claims": claims,
            "trace": _step(state, "check_grounding", status),
        }

    def check_usefulness(self, state: MedicalState) -> dict[str, Any]:
        try:
            out = self._ask_json(
                prompts.USEFULNESS_SYSTEM,
                prompts.USEFULNESS_PROMPT.format(
                    question=state["question"], answer=state["answer"]
                ),
            )
            useful = bool(out.get("useful", True))
            missing = "" if useful else str(out.get("missing", "")).strip()
            fallback = ""
        except ValueError as exc:
            # The answer is already grounded at this point, so an unparseable verdict keeps it.
            logger.warning("Usefulness check fallback used: %s", exc)
            useful, missing, fallback = True, "", " (fallback)"

        verdict = "useful" if useful else "not_useful"
        detail = f"{verdict}{f' · missing: {missing}' if missing else ''}{fallback}"
        return {
            "usefulness": verdict,
            "missing_info": missing,
            "trace": _step(state, "check_usefulness", detail),
        }

    def insufficient(self, state: MedicalState) -> dict[str, Any]:
        message = prompts.localized(prompts.INSUFFICIENT_MESSAGES, state.get("language", "en"))
        return {
            "answer": message,
            "spoken": message,
            "source_used": "insufficient",
            "passages": [],
            "grounding": "skipped",
            "trace": _step(state, "insufficient", "no reliable evidence, clinician advised"),
        }

    def finalize(self, state: MedicalState) -> dict[str, Any]:
        turn = {"question": state["question"], "answer": state.get("answer", "")}
        return {"memory": [turn], "trace": _step(state, "finalize", "turn saved to memory")}

    # --------------------------------------------------------------- routing

    @staticmethod
    def route_after_triage(state: MedicalState) -> str:
        if state.get("urgency") == "emergency":
            return "emergency"
        if not state.get("in_scope", True):
            return "out_of_scope"
        return "retrieve"

    def route_after_grading(self, state: MedicalState) -> str:
        """Corrective loop: KB -> research tools -> query rewrite -> research again."""
        can_research = self._can_research(state)
        first_round = state.get("research_rounds", 0) == 0
        if state.get("passages"):
            if state.get("needs_current_info") and first_round and can_research:
                return "plan_research"
            return "generate"
        if can_research and first_round:
            return "plan_research"
        if self._rewrites_left(state):
            return "rewrite_query"
        return "plan_research" if can_research else "insufficient"

    def route_after_rewrite(self, state: MedicalState) -> str:
        # A rewrite asked by the usefulness check targets missing facts: go to the tools.
        if state.get("missing_info") and self._can_research(state):
            return "plan_research"
        return "retrieve"

    def _rewrites_left(self, state: MedicalState) -> int:
        return max(self.max_query_rewrites - state.get("query_rewrites", 0), 0)

    def route_after_grounding(self, state: MedicalState) -> str:
        if state.get("grounding") in {"supported", "partial"}:
            return "check_usefulness" if self.usefulness_check else "finalize"
        if state.get("grounding_retries", 0) < self.max_grounding_retries:
            return "revise"
        return "insufficient"

    def route_after_usefulness(self, state: MedicalState) -> str:
        if state.get("usefulness") == "not_useful" and self._rewrites_left(state):
            return "rewrite_query"
        return "finalize"
