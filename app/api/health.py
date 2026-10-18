"""Liveness and readiness probes."""

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app import __version__
from app.api.schemas import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/health", tags=["health"])


def _models(settings) -> dict[str, str]:
    """Model names shown in the UI inspector (no secrets)."""
    return {
        "llm": _llm_chain(settings, "answer"),
        "llm_fast": _llm_chain(settings, "fast"),
        "embeddings": settings.embed_model,
        "speech_to_text": settings.cf_stt_model.rsplit("/", 1)[-1],
        "vision": settings.cf_vision_model.rsplit("/", 1)[-1],
        "text_to_speech": settings.cf_tts_model.rsplit("/", 1)[-1],
        "research_tools": _research_tools(settings),
    }


def _llm_chain(settings, lane: str) -> str:
    from app.core.llm import gateway_chain

    return gateway_chain(settings, lane) or "not configured"


def _research_tools(settings) -> str:
    if not settings.research_tools:
        return "disabled"
    tools = ["pubmed", "drug_label", "medlineplus"]
    names = " · ".join([*tools, "web"] if settings.web_search_enabled else tools)
    return f"{names} (MCP {settings.mcp_transport})"


@router.get("", response_model=HealthResponse, summary="Liveness probe")
def health(request: Request) -> HealthResponse:
    settings = request.app.state.settings
    return HealthResponse(
        status="ok",
        version=__version__,
        environment=settings.environment,
        models=_models(settings),
    )


@router.get("/ready", response_model=HealthResponse, summary="Readiness probe")
def ready(request: Request):
    settings = request.app.state.settings
    service = getattr(request.app.state, "chat_service", None)
    checks: dict[str, str] = {}

    missing = settings.missing_required()
    checks["config"] = "ok" if not missing else f"missing: {', '.join(missing)}"
    checks["web_search"] = "ok" if settings.web_search_enabled else "disabled"
    if settings.research_tools:
        from app.integrations.mcp_toolbox import mcp_server_status

        checks["mcp_server"] = mcp_server_status(settings)

    if service is None:
        checks["database"] = "not initialized"
    else:
        try:
            service.check_database()
            checks["database"] = "ok"
        except Exception as exc:
            logger.warning("Readiness database check failed: %s", exc)
            checks["database"] = "error"

    # Optional parts do not make the service unhealthy: web search may be disabled and
    # a stdio MCP server only exists while a tool runs.
    healthy = all(
        value == "ok" or (key, value) in {("web_search", "disabled"), ("mcp_server", "stdio")}
        for key, value in checks.items()
    )
    body = HealthResponse(
        status="ok" if healthy else "degraded",
        version=__version__,
        environment=settings.environment,
        checks=checks,
    )
    return JSONResponse(status_code=200 if healthy else 503, content=body.model_dump())
