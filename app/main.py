"""FastAPI application factory."""

import logging
import os
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import certifi
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api import chat, health, speech
from app.api.errors import register_error_handlers
from app.core.config import Settings, get_settings
from app.core.logging import request_id_var, setup_logging
from app.core.rate_limit import RateLimiter
from app.services.chat_service import ChatService
from app.services.media import MediaService

logger = logging.getLogger(__name__)

# Some Windows/Conda installs miss root certificates for the external APIs.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())


def build_production_services(settings: Settings):
    """Wire Gemini, Pinecone, the MCP research server, Workers AI and SQLite.

    Returns the chat service, the media service and a cleanup callback.
    """
    from dotenv import load_dotenv
    from langgraph.checkpoint.sqlite import SqliteSaver

    from app.agents.graph import build_medical_graph
    from app.core.llm import build_llms
    from app.integrations.mcp_toolbox import McpResearchTools, build_mcp_client
    from app.integrations.pinecone_kb import PineconeKnowledgeBase
    from app.integrations.workers_ai import WorkersAI
    from app.services.conversations import ConversationStore

    # Settings read .env directly, but LangChain reads LANGSMITH_* from the
    # process environment, so export .env there too (real env vars win).
    load_dotenv(override=False)

    missing = settings.missing_required()
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    Path(settings.database_path).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(settings.database_path, check_same_thread=False)
    connection.execute("PRAGMA journal_mode=WAL")
    checkpointer = SqliteSaver(connection)
    conversations = ConversationStore(settings.database_path)

    research_tools = (
        McpResearchTools(
            build_mcp_client(settings),
            web_enabled=settings.web_search_enabled,
            timeout_seconds=settings.mcp_timeout_seconds,
        )
        if settings.research_tools
        else None
    )
    answer_llm, fast_llm = build_llms(settings)
    graph = build_medical_graph(
        llm=answer_llm,
        fast_llm=fast_llm,
        knowledge_base=PineconeKnowledgeBase(settings),
        research_tools=research_tools,
        checkpointer=checkpointer,
        memory_turns=settings.memory_turns,
        max_tool_calls=settings.max_tool_calls,
        max_research_rounds=settings.max_research_rounds,
        max_query_rewrites=settings.max_query_rewrites,
        grounding_check=settings.grounding_check,
        max_grounding_retries=settings.max_grounding_retries,
        usefulness_check=settings.usefulness_check,
    )
    media = MediaService(WorkersAI(settings), settings.tts_default_lang)

    def cleanup() -> None:
        conversations.close()
        connection.close()

    return ChatService(graph, conversations, media), media, cleanup


def create_app(
    settings: Settings | None = None,
    chat_service: ChatService | None = None,
    media_service: MediaService | None = None,
) -> FastAPI:
    """Build the app. Passing `chat_service` skips production wiring (tests)."""
    settings = settings or get_settings()
    setup_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        cleanup = None

        if app.state.chat_service is None:
            app.state.chat_service, app.state.media_service, cleanup = build_production_services(
                settings
            )

        logger.info("MedGuide AI %s started (%s)", __version__, settings.environment)
        yield

        if cleanup:
            cleanup()

    app = FastAPI(
        title="MedGuide AI",
        description=(
            "Agentic medical knowledge assistant: Corrective and Reflective RAG over a "
            "private medical encyclopedia and MCP research tools (PubMed, openFDA, "
            "MedlinePlus, web), with emergency triage, grounding and usefulness checks, "
            "voice and image input."
        ),
        version=__version__,
        lifespan=lifespan,
    )

    app.state.settings = settings
    app.state.chat_service = chat_service
    app.state.media_service = media_service
    app.state.rate_limiter = RateLimiter(
        settings.rate_limit_requests, settings.rate_limit_window_seconds
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID", "Retry-After"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        token = request_id_var.set(request_id)
        started = time.perf_counter()

        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            logger.info(
                "%s %s -> %s in %.0f ms",
                request.method,
                request.url.path,
                response.status_code,
                (time.perf_counter() - started) * 1000,
            )
            return response
        finally:
            request_id_var.reset(token)

    register_error_handlers(app)
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(speech.router)

    @app.get("/", include_in_schema=False)
    def root():
        return {"service": "MedGuide AI", "docs": "/docs", "health": "/api/health"}

    return app


app = create_app()
