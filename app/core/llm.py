"""LLM gateway: chat models backed by one LiteLLM Router over several providers.

Two lanes, each a fallback chain across providers:
- `answer` writes the cited answer (quality first: Gemini, then Groq gpt-oss-120b...);
- `fast` runs the short classification steps (triage, grading, planning, checks) on a
  smaller, faster model first, so most calls stay off the answer lane's quota.
When a provider fails (rate limit, quota, timeout, outage) the Router falls back to
the next one of the lane and puts the failing deployment in cooldown. Identical
prompts are answered from an in-memory cache (exact match, with a TTL).

The cache lives in this module rather than in LiteLLM: LiteLLM 1.106's cache-hit path
imports the optional `soundfile` package, and the resulting error made the Router
treat every cache hit as a provider failure (wrongly cooling providers down).
"""

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict

from app.core.config import Settings

# Use the pricing map bundled with LiteLLM instead of downloading it at import time.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

CLOUDFLARE_OPENAI_BASE = "https://api.cloudflare.com/client/v4/accounts/{account}/ai/v1"
LANES = ("answer", "fast")

_ROLES = {"system": "system", "human": "user", "ai": "assistant"}


def _providers(settings: Settings, lane: str) -> dict[str, tuple[Any, dict[str, Any]]]:
    """Provider -> (credentials present, LiteLLM params) for one lane."""
    groq_model = settings.groq_model if lane == "answer" else settings.groq_fast_model
    return {
        "gemini": (
            settings.google_api_key,
            {"model": f"gemini/{settings.gemini_model}", "api_key": settings.google_api_key},
        ),
        "groq": (
            settings.groq_api_key,
            {
                "model": f"groq/{groq_model}",
                "api_key": settings.groq_api_key,
                # gpt-oss is a reasoning model: keep most output tokens for the JSON.
                "reasoning_effort": "low",
            },
        ),
        "cloudflare": (
            settings.cloudflare_api_token and settings.cloudflare_account_id,
            {
                # Workers AI exposes an OpenAI-compatible endpoint.
                "model": f"openai/{settings.cloudflare_llm_model}",
                "api_key": settings.cloudflare_api_token,
                "api_base": CLOUDFLARE_OPENAI_BASE.format(account=settings.cloudflare_account_id),
            },
        ),
        "openrouter": (
            settings.openrouter_api_key,
            {
                "model": f"openrouter/{settings.openrouter_model}",
                "api_key": settings.openrouter_api_key,
            },
        ),
    }


def deployments(settings: Settings, lane: str = "answer") -> list[dict[str, Any]]:
    """Router deployments of a lane, in fallback order, for providers with credentials.

    Deployment names are `<lane>-<provider>`; the name is also the deployment id, which
    LiteLLM reports back in each response so the trace shows who answered.
    """
    candidates = _providers(settings, lane)
    order = settings.llm_providers if lane == "answer" else settings.llm_fast_providers
    names = [name.strip() for name in order.split(",") if name.strip()]
    return [
        {
            "model_name": f"{lane}-{name}",
            "litellm_params": {**candidates[name][1], "timeout": settings.llm_timeout_seconds},
            "model_info": {"id": f"{lane}-{name}"},
        }
        for name in names
        if name in candidates and candidates[name][0]
    ]


def build_router(settings: Settings):
    # Imported here: LiteLLM is large, and tests use a fake chat model instead.
    import litellm
    from litellm import Router

    litellm.suppress_debug_info = True
    # Providers that do not support a parameter (e.g. reasoning_effort) simply ignore it.
    litellm.drop_params = True

    model_list, fallbacks = [], []
    for lane in LANES:
        lane_deployments = deployments(settings, lane)
        if not lane_deployments:
            raise RuntimeError(
                "No LLM provider is configured (Gemini, Groq, Cloudflare, OpenRouter)"
            )
        names = [d["model_name"] for d in lane_deployments]
        model_list += lane_deployments
        if len(names) > 1:
            fallbacks.append({names[0]: names[1:]})

    return Router(
        model_list=model_list,
        fallbacks=fallbacks,
        # Fall back instead of retrying a rate-limited provider.
        num_retries=0,
        allowed_fails=1,
        cooldown_time=settings.llm_cooldown_seconds,
        timeout=settings.llm_timeout_seconds,
    )


class ResponseCache:
    """Thread-safe, in-memory LRU cache with a TTL. Never written to disk."""

    def __init__(
        self, ttl_seconds: float, max_items: int = 256, clock: Callable[[], float] = time.monotonic
    ):
        self.ttl = ttl_seconds
        self.max_items = max_items
        self._clock = clock
        self._items: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def key(lane: str, payload: dict[str, Any]) -> str:
        raw = json.dumps({"lane": lane, **payload}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            stored_at, value = item
            if self._clock() - stored_at > self.ttl:
                del self._items[key]
                return None
            self._items.move_to_end(key)
            return value

    def set(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._items[key] = (self._clock(), value)
            self._items.move_to_end(key)
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)


class GatewayChatModel(BaseChatModel):
    """LangChain chat model on top of a LiteLLM Router lane.

    `response_metadata` reports the model and deployment that actually answered, and
    whether the answer came from the cache, so the agent trace shows it per step.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    router: Any
    primary: str
    json_mode: bool = True
    response_cache: Any = None  # ResponseCache shared by the lanes, or None

    @property
    def _llm_type(self) -> str:
        return "litellm-router-gateway"

    def _generate(
        self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs
    ) -> ChatResult:
        chat = [{"role": _ROLES.get(m.type, "user"), "content": str(m.content)} for m in messages]
        params: dict[str, Any] = {"model": self.primary, "messages": chat}
        if self.json_mode:
            # Every agent step asks for a JSON object. Some providers (Groq) refuse JSON mode
            # unless the prompt itself mentions JSON, so make sure it does.
            if not any("json" in m["content"].lower() for m in chat):
                chat.insert(0, {"role": "system", "content": "Respond with a JSON object."})
            params["response_format"] = {"type": "json_object"}
        if stop:
            params["stop"] = stop

        key = ResponseCache.key(self.primary, {k: v for k, v in params.items() if k != "model"})
        cached = self.response_cache.get(key) if self.response_cache else None
        if cached is not None:
            return self._result({**cached, "cached": True})

        response = self.router.completion(**params)
        hidden = getattr(response, "_hidden_params", None) or {}
        value = {
            "content": response.choices[0].message.content or "",
            "model_name": response.model,
            "provider": hidden.get("model_id") or "",
        }
        if self.response_cache and value["content"]:
            self.response_cache.set(key, value)
        return self._result({**value, "cached": False})

    @staticmethod
    def _result(value: dict[str, Any]) -> ChatResult:
        message = AIMessage(
            content=value["content"],
            response_metadata={
                "model_name": value["model_name"],
                "provider": value["provider"],
                "cached": value["cached"],
            },
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


def build_llms(settings: Settings) -> tuple[BaseChatModel, BaseChatModel]:
    """(answer model, fast model), sharing one Router (cooldowns) and one cache."""
    router = build_router(settings)
    ttl = settings.llm_cache_ttl_seconds
    cache = ResponseCache(ttl) if ttl > 0 else None
    answer = GatewayChatModel(
        router=router,
        primary=deployments(settings, "answer")[0]["model_name"],
        response_cache=cache,
    )
    fast = GatewayChatModel(
        router=router, primary=deployments(settings, "fast")[0]["model_name"], response_cache=cache
    )
    return answer, fast


def gateway_chain(settings: Settings, lane: str = "answer") -> str:
    """Human-readable fallback chain of a lane, e.g. for the health endpoint."""
    labels = {
        "gemini": settings.gemini_model,
        "groq": settings.groq_model if lane == "answer" else settings.groq_fast_model,
        "cloudflare": settings.cloudflare_llm_model.rsplit("/", 1)[-1],
        "openrouter": settings.openrouter_model,
    }
    return " → ".join(
        f"{d['model_name'].removeprefix(f'{lane}-')}:{labels[d['model_name'].removeprefix(f'{lane}-')]}"
        for d in deployments(settings, lane)
    )
