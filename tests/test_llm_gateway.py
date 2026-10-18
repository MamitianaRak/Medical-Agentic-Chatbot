"""LLM gateway: lanes, deployment list, LangChain adapter, cache flag (no network)."""

from types import SimpleNamespace

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver

from app.agents.graph import build_medical_graph
from app.agents.state import initial_state
from app.core.config import Settings
from app.core.llm import GatewayChatModel, ResponseCache, deployments, gateway_chain
from tests.fakes import FakeKnowledgeBase, FakeLLM

ALL_KEYS = {
    "google_api_key": "g",
    "groq_api_key": "q",
    "openrouter_api_key": "o",
    "cloudflare_api_token": "t",
    "cloudflare_account_id": "acc",
}


def names(settings: Settings, lane: str) -> list[str]:
    return [d["model_name"] for d in deployments(settings, lane)]


# --- Lanes and deployments ------------------------------------------------------------


def test_each_lane_follows_its_own_order():
    settings = Settings(_env_file=None, **ALL_KEYS)

    assert names(settings, "answer") == [
        "answer-gemini",
        "answer-groq",
        "answer-cloudflare",
        "answer-openrouter",
    ]
    assert names(settings, "fast") == [
        "fast-groq",
        "fast-cloudflare",
        "fast-gemini",
        "fast-openrouter",
    ]
    # The deployment id is reported back by LiteLLM, so the trace shows who answered.
    assert all(d["model_info"]["id"] == d["model_name"] for d in deployments(settings, "fast"))


def test_fast_lane_uses_the_smaller_groq_model():
    settings = Settings(_env_file=None, **ALL_KEYS)
    answer_groq = deployments(settings, "answer")[1]["litellm_params"]
    fast_groq = deployments(settings, "fast")[0]["litellm_params"]

    assert answer_groq["model"] == "groq/openai/gpt-oss-120b"
    assert fast_groq["model"] == "groq/openai/gpt-oss-20b"
    assert fast_groq["reasoning_effort"] == "low"


def test_providers_without_credentials_are_skipped():
    settings = Settings(_env_file=None, groq_api_key="q", openrouter_api_key="o")
    assert names(settings, "answer") == ["answer-groq", "answer-openrouter"]
    assert names(settings, "fast") == ["fast-groq", "fast-openrouter"]


def test_cloudflare_goes_through_its_openai_compatible_endpoint():
    settings = Settings(_env_file=None, llm_providers="cloudflare", **ALL_KEYS)
    params = deployments(settings, "answer")[0]["litellm_params"]

    assert params["model"] == "openai/@cf/meta/llama-3.3-70b-instruct-fp8-fast"
    assert params["api_base"].endswith("/accounts/acc/ai/v1")


def test_gateway_chain_is_human_readable():
    settings = Settings(_env_file=None, groq_api_key="q", google_api_key="g")
    assert gateway_chain(settings, "answer") == (
        "gemini:gemini-3.5-flash-lite → groq:openai/gpt-oss-120b"
    )
    assert (
        gateway_chain(settings, "fast") == "groq:openai/gpt-oss-20b → gemini:gemini-3.5-flash-lite"
    )
    assert gateway_chain(Settings(_env_file=None)) == ""


# --- LangChain adapter -----------------------------------------------------------------


class FakeRouter:
    """Stands in for litellm.Router: records the call, answers like a LiteLLM response."""

    def __init__(self, served_by: str = "answer-groq"):
        self.calls: list[dict] = []
        self.served_by = served_by

    def completion(self, **params):
        self.calls.append(params)
        response = SimpleNamespace(
            model="openai/gpt-oss-120b",
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))],
        )
        response._hidden_params = {"model_id": self.served_by}
        return response


def test_gateway_model_calls_its_lane_in_json_mode():
    router = FakeRouter()
    llm = GatewayChatModel(router=router, primary="answer-gemini")

    reply = llm.invoke([SystemMessage(content="Return a JSON object."), HumanMessage(content="hi")])

    call = router.calls[0]
    assert call["model"] == "answer-gemini"
    assert call["response_format"] == {"type": "json_object"}
    assert call["messages"] == [
        {"role": "system", "content": "Return a JSON object."},
        {"role": "user", "content": "hi"},
    ]
    assert reply.content == '{"ok": true}'
    # The fallback deployment that actually answered is reported.
    assert reply.response_metadata == {
        "model_name": "openai/gpt-oss-120b",
        "provider": "answer-groq",
        "cached": False,
    }


def test_identical_prompts_are_served_from_the_cache():
    router = FakeRouter()
    llm = GatewayChatModel(router=router, primary="fast-groq", response_cache=ResponseCache(60))
    prompt = [HumanMessage(content="json please")]

    first = llm.invoke(prompt)
    second = llm.invoke(prompt)
    llm.invoke([HumanMessage(content="another json prompt")])

    assert len(router.calls) == 2  # the repeated prompt never reached a provider
    assert first.response_metadata["cached"] is False
    assert second.response_metadata == {**first.response_metadata, "cached": True}
    assert second.content == first.content


def test_cache_is_separate_per_lane():
    router, cache = FakeRouter(), ResponseCache(60)
    prompt = [HumanMessage(content="json please")]
    GatewayChatModel(router=router, primary="fast-groq", response_cache=cache).invoke(prompt)
    GatewayChatModel(router=router, primary="answer-gemini", response_cache=cache).invoke(prompt)

    assert len(router.calls) == 2


def test_cache_entries_expire_and_are_bounded():
    now = [0.0]
    cache = ResponseCache(ttl_seconds=10, max_items=2, clock=lambda: now[0])
    cache.set("a", {"v": 1})
    cache.set("b", {"v": 2})
    cache.set("c", {"v": 3})  # evicts the least recently used entry

    assert cache.get("a") is None
    assert cache.get("c") == {"v": 3}
    now[0] = 11
    assert cache.get("c") is None


def test_gateway_model_mentions_json_when_the_prompt_does_not():
    router = FakeRouter()
    GatewayChatModel(router=router, primary="answer-gemini").invoke([HumanMessage(content="ping")])

    assert router.calls[0]["messages"][0] == {
        "role": "system",
        "content": "Respond with a JSON object.",
    }


# --- Task routing in the agent ---------------------------------------------------------


def test_only_the_answer_goes_to_the_answer_lane():
    answer_llm, fast_llm = FakeLLM(), FakeLLM()
    graph = build_medical_graph(
        answer_llm, FakeKnowledgeBase(), None, MemorySaver(), fast_llm=fast_llm
    )

    graph.invoke(
        initial_state("What triggers asthma?"), config={"configurable": {"thread_id": "t"}}
    )

    assert answer_llm.calls("Write the answer") == 1
    assert len(answer_llm.prompts) == 1  # nothing else on the answer lane
    for marker in ("triage step", "relevance grader", "grounding checker", "usefulness checker"):
        assert fast_llm.calls(marker) == 1, marker


def test_trace_reports_the_model_of_each_llm_step(client):
    body = client.post("/api/chat", json={"message": "What triggers asthma attacks?"}).json()
    models = {step["node"]: step["model"] for step in body["trace"]}

    assert models["triage"] == "fake:fake-model"
    assert models["generate"] == "fake:fake-model"
    assert models["retrieve"] == ""  # not an LLM step
    assert models["finalize"] == ""
