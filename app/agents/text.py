"""Parsing and formatting helpers shared by the agent nodes."""

import json
import re
from typing import Any

from app.agents.state import Passage, Turn

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


# A backslash that does not start a valid JSON escape, e.g. "l'asthme".
_BAD_ESCAPE = re.compile(r'\\(?!["\\/bfnrtu])')


def _loads(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Models sometimes escape apostrophes or other characters JSON does not escape.
        return json.loads(_BAD_ESCAPE.sub("", text))


def json_from_llm(text: str) -> dict[str, Any]:
    """Parse the JSON object an LLM returned, tolerating code fences, stray prose and
    invalid escapes."""
    cleaned = _FENCE.sub("", text.strip())

    try:
        value = _loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(f"No JSON object in LLM output: {text[:200]!r}") from None
        try:
            value = _loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in LLM output: {exc}") from exc

    if not isinstance(value, dict):
        raise ValueError("LLM output is not a JSON object")
    return value


def content_text(content: Any) -> str:
    """Newer Gemini models may return a list of content blocks instead of a string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part if isinstance(part, str) else str(part.get("text", ""))
            for part in content
            if isinstance(part, str | dict)
        )
    return str(content)


def history_text(memory: list[Turn], turns: int) -> str:
    recent = memory[-turns:] if turns > 0 else []
    return "\n\n".join(f"User: {t['question']}\nAssistant: {t['answer']}" for t in recent)


def cite_label(passage: Passage) -> str:
    return f"[{passage['label']}]"


def format_passages(passages: list[Passage], max_chars: int = 1500) -> str:
    blocks = []
    for index, passage in enumerate(passages, 1):
        text = " ".join(passage["text"].split())[:max_chars]
        blocks.append(f"[{index}] {cite_label(passage)} {passage['title']}\n{text}")
    return "\n\n".join(blocks)


def format_evidence(passages: list[Passage], max_chars: int = 1500) -> str:
    """Evidence for answer generation: each block starts with the exact label to cite."""
    blocks = []
    for passage in passages:
        text = " ".join(passage["text"].split())[:max_chars]
        blocks.append(f"Cite as {cite_label(passage)} ({passage['title']}):\n{text}")
    return "\n\n".join(blocks)


def excerpt(text: str, length: int = 220) -> str:
    text = " ".join(text.split())
    return text if len(text) <= length else text[:length].rsplit(" ", 1)[0] + "…"
