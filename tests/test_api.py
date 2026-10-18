import base64
import json

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app

PNG_DATA_URL = "data:image/png;base64," + base64.b64encode(b"\x89PNG fake image").decode()


def sse_events(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "X-Request-ID" in response.headers


def test_ready_reports_missing_config(client):
    response = client.get("/api/health/ready")
    body = response.json()

    assert response.status_code == 503
    assert body["checks"]["database"] == "ok"
    assert "GOOGLE_API_KEY" in body["checks"]["config"]
    assert body["checks"]["web_search"] == "disabled"
    assert body["checks"]["mcp_server"] == "stdio"


def test_health_lists_models_and_research_tools(client):
    models = client.get("/api/health").json()["models"]
    assert models["llm"] == "gemini-3.5-flash-lite"
    assert models["research_tools"] == "pubmed · drug_label · medlineplus (MCP stdio)"


def test_chat_reports_mcp_tool_calls(client, llm):
    llm.relevant_rounds = [[], [1]]
    body = client.post("/api/chat", json={"message": "Newest asthma biologic?"}).json()

    assert body["source_used"] == "research"
    assert body["research_calls"] == [{"tool": "pubmed", "query": "asthma biologics", "results": 1}]
    assert body["citations"][0]["type"] == "pubmed"
    assert body["citations"][0]["label"] == "PMID:34655640"
    assert body["citations"][0]["score"] is None


def test_chat_returns_answer_citations_and_trace(client):
    response = client.post("/api/chat", json={"message": "What triggers asthma attacks?"})
    body = response.json()

    assert response.status_code == 200
    assert body["success"] is True
    assert body["thread_id"].startswith("med_")
    assert body["urgency"] == "routine"
    assert body["grounding"] == "supported"
    assert body["usefulness"] == "useful"
    assert body["research_calls"] == []
    assert body["citations"][0] == {
        "id": "kb-394",
        "type": "kb",
        "title": "Gale Encyclopedia of Medicine (2002)",
        "label": "PDF p.394",
        "page": 394,
        "url": "",
        "score": 0.62,
        "excerpt": "Asthma attacks are triggered by allergens, smoke and exercise.",
    }
    assert [s["node"] for s in body["trace"]][0] == "triage"
    assert body["trace"][0]["label"] == "Emergency triage"


def test_chat_with_image_adds_a_vision_step(client):
    response = client.post(
        "/api/chat", json={"message": "Is this rash worrying?", "image_data": PNG_DATA_URL}
    )
    body = response.json()

    assert response.status_code == 200
    assert body["image_description"] == "A red, raised rash on the forearm."
    assert body["trace"][0]["node"] == "vision"


def test_invalid_image_is_rejected(client):
    response = client.post(
        "/api/chat", json={"message": "Look", "image_data": "data:image/gif;base64,R0lG"}
    )
    assert response.status_code == 400
    assert response.json()["error"] == "Only JPEG, PNG and WebP images are supported."


def test_oversized_image_is_rejected(service):
    settings = Settings(_env_file=None, max_image_bytes=4, environment="test")
    with TestClient(create_app(settings=settings, chat_service=service)) as client:
        response = client.post("/api/chat", json={"message": "Look", "image_data": PNG_DATA_URL})

    assert response.status_code == 413
    assert response.json()["code"] == "image_too_large"


def test_empty_message_is_rejected(client):
    response = client.post("/api/chat", json={"message": "   "})
    assert response.status_code == 400
    assert response.json() == {
        "success": False,
        "error": "Message cannot be empty.",
        "code": "invalid_request",
    }


def test_stream_emits_start_steps_and_result(client):
    response = client.post("/api/chat/stream", json={"message": "What triggers asthma attacks?"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = sse_events(response.text)
    kinds = [kind for kind, _ in events]

    assert kinds[0] == "start"
    assert kinds[-1] == "result"
    steps = [data["node"] for kind, data in events if kind == "step"]
    assert steps == [
        "triage",
        "retrieve",
        "grade_documents",
        "generate",
        "check_grounding",
        "check_usefulness",
        "finalize",
    ]
    assert events[0][1]["thread_id"] == events[-1][1]["thread_id"]


def test_stream_error_becomes_error_event(client, llm):
    llm.fail_with = RuntimeError("secret internal detail")
    response = client.post("/api/chat/stream", json={"message": "What triggers asthma?"})
    event, data = sse_events(response.text)[-1]

    assert event == "error"
    assert data["code"] == "internal_error"
    assert "secret" not in data["error"]


def test_provider_quota_maps_to_503(client, llm):
    llm.fail_with = RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded")
    response = client.post("/api/chat", json={"message": "What triggers asthma?"})

    assert response.status_code == 503
    assert response.json()["code"] == "provider_rate_limited"


def test_conversation_history_and_delete(client):
    first = client.post("/api/chat", json={"message": "What triggers asthma attacks?"}).json()
    thread_id = first["thread_id"]
    client.post("/api/chat", json={"message": "And in children?", "thread_id": thread_id})

    listed = client.get("/api/conversations").json()["conversations"]
    assert listed[0]["thread_id"] == thread_id
    assert listed[0]["title"] == "What triggers asthma attacks?"
    assert listed[0]["turns"] == 2

    detail = client.get(f"/api/conversations/{thread_id}").json()
    assert [m["question"] for m in detail["messages"]] == [
        "What triggers asthma attacks?",
        "And in children?",
    ]

    assert client.delete(f"/api/conversations/{thread_id}").status_code == 204
    assert client.get(f"/api/conversations/{thread_id}").status_code == 404
    assert client.get("/api/conversations").json()["conversations"] == []


def test_unknown_conversation_returns_404(client):
    response = client.get("/api/conversations/med_missing")
    assert response.status_code == 404
    assert response.json()["code"] == "conversation_not_found"


def test_invalid_thread_id_is_rejected(client):
    response = client.post("/api/chat", json={"message": "Hi", "thread_id": "../etc"})
    assert response.status_code == 400


def test_rate_limit_returns_429(service):
    settings = Settings(_env_file=None, rate_limit_requests=1, environment="test")

    with TestClient(create_app(settings=settings, chat_service=service)) as client:
        assert client.post("/api/chat", json={"message": "Asthma?"}).status_code == 200
        limited = client.post("/api/chat", json={"message": "Asthma?"})

    assert limited.status_code == 429
    assert limited.json()["code"] == "rate_limited"
    assert int(limited.headers["Retry-After"]) >= 1


def test_api_token_protects_every_endpoint_but_health(service):
    settings = Settings(_env_file=None, api_token="s3cret", environment="test")

    with TestClient(create_app(settings=settings, chat_service=service)) as client:
        assert client.get("/api/health").status_code == 200
        denied = client.post("/api/chat", json={"message": "Asthma?"})
        allowed = client.post(
            "/api/chat",
            json={"message": "Asthma?"},
            headers={"Authorization": "Bearer s3cret"},
        )

    assert denied.status_code == 401
    assert denied.json()["code"] == "unauthorized"
    assert allowed.status_code == 200


def test_settings_accept_legacy_variable_names(monkeypatch):
    monkeypatch.setenv("PINECONE_INDEX_NAME", "my-index")
    monkeypatch.setenv("TTS_LANG", "fr")
    settings = Settings(_env_file=None)

    assert settings.pinecone_index == "my-index"
    assert settings.tts_default_lang == "fr"
