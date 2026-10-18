import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver

from app.agents.graph import build_medical_graph
from app.core.config import Settings
from app.main import create_app
from app.services.chat_service import ChatService
from app.services.conversations import ConversationStore
from app.services.media import MediaService
from tests.fakes import FakeKnowledgeBase, FakeLLM, FakeMedia, FakeResearchTools


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    """Keep real keys from the developer's shell out of the tests."""
    for name in [*Settings.model_fields, "PINECONE_INDEX_NAME", "TTS_LANG"]:
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def kb() -> FakeKnowledgeBase:
    return FakeKnowledgeBase()


@pytest.fixture
def tools() -> FakeResearchTools:
    return FakeResearchTools()


@pytest.fixture
def media() -> FakeMedia:
    return FakeMedia()


@pytest.fixture
def graph(llm, kb, tools):
    return build_medical_graph(llm, kb, tools, MemorySaver())


@pytest.fixture
def service(graph, media) -> ChatService:
    return ChatService(graph, ConversationStore(":memory:"), MediaService(media))


@pytest.fixture
def settings() -> Settings:
    # _env_file=None keeps a developer's local .env out of the tests.
    return Settings(_env_file=None, rate_limit_requests=100, environment="test")


@pytest.fixture
def client(settings: Settings, service: ChatService):
    app = create_app(settings=settings, chat_service=service, media_service=service.media)
    with TestClient(app) as test_client:
        yield test_client
