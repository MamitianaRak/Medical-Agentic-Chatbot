"""Use cases on top of the compiled graph: ask, stream steps, read and delete history."""

import logging
import uuid
from collections.abc import Iterator
from typing import Any

from langgraph.graph.state import CompiledStateGraph

from app.agents.state import NODE_LABELS, MedicalState, initial_state
from app.agents.text import excerpt
from app.core.errors import AppError, classify_provider_error
from app.services.conversations import ConversationStore
from app.services.media import MediaService

logger = logging.getLogger(__name__)

StepEvent = dict[str, Any]

# The longest path (web fallback + one revision) stays well under this.
RECURSION_LIMIT = 30


def new_thread_id() -> str:
    return f"med_{uuid.uuid4().hex[:20]}"


def _step(node: str, detail: str) -> dict[str, str]:
    return {"node": node, "label": NODE_LABELS.get(node, node), "detail": detail}


def _citations(state: MedicalState) -> list[dict[str, Any]]:
    return [
        {
            "id": p["id"],
            "type": p["source"],
            "title": p["title"],
            "page": p["page"],
            "url": p["url"],
            "label": p["label"],
            "score": p["score"],
            "excerpt": excerpt(p["text"]),
        }
        for p in state.get("passages", [])
    ]


class ChatService:
    def __init__(
        self,
        graph: CompiledStateGraph,
        conversations: ConversationStore,
        media: MediaService | None = None,
    ):
        self.graph = graph
        self.conversations = conversations
        self.media = media

    @staticmethod
    def _config(thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}

    def _observe_image(self, image_data: str | None) -> str:
        if not image_data:
            return ""
        if self.media is None:
            raise AppError(503, "media_unavailable", "Image analysis is not configured.")
        return self.media.describe_image(image_data)

    def _result(self, thread_id: str, vision: dict[str, str] | None) -> dict[str, Any]:
        state: MedicalState = self.graph.get_state(self._config(thread_id)).values
        trace = [_step(s["node"], s["detail"]) for s in state.get("trace", [])]
        return {
            "thread_id": thread_id,
            "answer": state.get("answer", ""),
            "spoken": state.get("spoken", ""),
            "language": state.get("language", "en"),
            "urgency": state.get("urgency", "routine"),
            "in_scope": state.get("in_scope", True),
            "source_used": state.get("source_used", "insufficient"),
            "grounding": state.get("grounding", "skipped"),
            "usefulness": state.get("usefulness", "skipped"),
            "research_calls": list(state.get("research_calls", [])),
            "triage_reason": state.get("triage_reason", ""),
            "image_description": state.get("image_description", ""),
            "citations": _citations(state),
            "trace": [vision, *trace] if vision else trace,
        }

    def _vision_step(self, description: str) -> dict[str, str] | None:
        return _step("vision", excerpt(description, 80)) if description else None

    def ask(self, message: str, thread_id: str | None = None, image_data: str | None = None):
        thread_id = thread_id or new_thread_id()
        description = self._observe_image(image_data)
        logger.info("Answering on thread %s", thread_id)
        try:
            self.graph.invoke(initial_state(message, description), config=self._config(thread_id))
        except Exception as exc:
            classified = classify_provider_error(exc)
            if classified is None:
                raise
            logger.warning("Provider failure on %s: %s", thread_id, exc)
            raise classified from exc
        self.conversations.touch(thread_id, message)
        return self._result(thread_id, self._vision_step(description))

    def stream(
        self, message: str, thread_id: str | None = None, image_data: str | None = None
    ) -> Iterator[StepEvent]:
        """Yield `start`, one `step` per finished node, then the `result`."""
        thread_id = thread_id or new_thread_id()
        logger.info("Streaming answer on thread %s", thread_id)
        yield {"event": "start", "data": {"thread_id": thread_id}}

        description = self._observe_image(image_data)
        vision = self._vision_step(description)
        if vision:
            yield {"event": "step", "data": vision}

        for update in self.graph.stream(
            initial_state(message, description),
            config=self._config(thread_id),
            stream_mode="updates",
        ):
            for node, payload in update.items():
                trace = (payload or {}).get("trace") if isinstance(payload, dict) else None
                if trace:
                    last = trace[-1]
                    yield {"event": "step", "data": _step(last["node"], last["detail"])}
                else:
                    yield {"event": "step", "data": _step(node, "")}

        self.conversations.touch(thread_id, message)
        yield {"event": "result", "data": self._result(thread_id, vision)}

    def history(self, thread_id: str) -> dict[str, Any]:
        summary = self.conversations.get(thread_id)
        if summary is None:
            raise AppError(404, "conversation_not_found", "This conversation does not exist.")
        state = self.graph.get_state(self._config(thread_id)).values
        return {**summary, "messages": list(state.get("memory", []))}

    def delete(self, thread_id: str) -> None:
        if self.conversations.get(thread_id) is None:
            raise AppError(404, "conversation_not_found", "This conversation does not exist.")
        if self.graph.checkpointer is not None:
            self.graph.checkpointer.delete_thread(thread_id)
        self.conversations.delete(thread_id)

    def check_database(self) -> None:
        """Raise if the checkpointer or the conversation index cannot be read."""
        self.conversations.ping()
        self.graph.get_state(self._config("healthcheck"))
