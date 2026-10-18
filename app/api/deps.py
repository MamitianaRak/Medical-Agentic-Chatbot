"""FastAPI dependencies: service injection, optional token auth, rate limiting."""

import math
import secrets

from fastapi import Request

from app.core.errors import AppError
from app.services.chat_service import ChatService
from app.services.media import MediaService


def get_chat_service(request: Request) -> ChatService:
    service = getattr(request.app.state, "chat_service", None)
    if service is None:
        raise AppError(503, "service_unavailable", "The assistant is not ready yet.")
    return service


def get_media_service(request: Request) -> MediaService:
    service = getattr(request.app.state, "media_service", None)
    if service is None:
        raise AppError(503, "media_unavailable", "Speech and vision are not configured.")
    return service


def require_token(request: Request) -> None:
    """When API_TOKEN is set, every /api call except health needs `Authorization: Bearer`."""
    token = request.app.state.settings.api_token
    if not token:
        return
    supplied = request.headers.get("authorization", "")
    if not secrets.compare_digest(supplied, f"Bearer {token}"):
        raise AppError(401, "unauthorized", "A valid API token is required.")


def client_ip(request: Request) -> str:
    if request.app.state.settings.trust_proxy_headers:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()

    return request.client.host if request.client else "unknown"


def rate_limit(request: Request) -> None:
    retry_after = request.app.state.rate_limiter.check(client_ip(request))

    if retry_after is not None:
        seconds = max(math.ceil(retry_after), 1)
        raise AppError(
            429,
            "rate_limited",
            f"Too many requests. Please retry in {seconds} seconds.",
            headers={"Retry-After": str(seconds)},
        )
