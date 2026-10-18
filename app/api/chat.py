"""Chat endpoints (JSON and Server-Sent Events) and conversation history."""

import contextvars
import json
import queue
import threading
from collections.abc import Iterator

from fastapi import APIRouter, Depends, Path, Query, Request
from fastapi.responses import StreamingResponse

from app.api.deps import get_chat_service, rate_limit, require_token
from app.api.errors import error_body, to_app_error
from app.api.schemas import (
    THREAD_ID_PATTERN,
    ChatRequest,
    ChatResponse,
    ConversationDetail,
    ConversationList,
    ErrorResponse,
    image_bytes,
)
from app.core.errors import AppError
from app.services.chat_service import ChatService, StepEvent

router = APIRouter(prefix="/api", tags=["chat"], dependencies=[Depends(require_token)])

ERROR_RESPONSES = {
    code: {"model": ErrorResponse} for code in (400, 401, 404, 413, 429, 502, 503, 504)
}

ThreadId = Path(pattern=THREAD_ID_PATTERN)

_DONE = object()


def _check_image_size(request: Request, body: ChatRequest) -> None:
    limit = request.app.state.settings.max_image_bytes
    if body.image_data and len(image_bytes(body.image_data)) > limit:
        raise AppError(413, "image_too_large", f"Images must be under {limit // 1_000_000} MB.")


def _in_worker_thread(events: Iterator[StepEvent]) -> Iterator[StepEvent]:
    """Drive the whole graph stream from one dedicated thread.

    The response iterator may be resumed on different threadpool threads, which
    breaks LangGraph's context variables; a single worker thread avoids that.
    """
    channel: queue.Queue = queue.Queue()
    context = contextvars.copy_context()

    def worker() -> None:
        try:
            for item in events:
                channel.put(item)
        except Exception as exc:
            channel.put(exc)
        finally:
            channel.put(_DONE)

    threading.Thread(target=context.run, args=(worker,), daemon=True).start()

    while (item := channel.get()) is not _DONE:
        if isinstance(item, Exception):
            raise item
        yield item


def _format_sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _sse(events: Iterator[StepEvent]) -> Iterator[str]:
    """Format events as SSE; failures mid-stream become an `error` event."""
    try:
        for item in _in_worker_thread(events):
            yield _format_sse(item["event"], item["data"])
    except Exception as exc:
        error = to_app_error(exc)
        yield _format_sse("error", error_body(error.code, error.message))


# Endpoints are sync on purpose: FastAPI runs them in a worker thread, so the
# blocking graph and provider calls never block the event loop.
@router.post(
    "/chat",
    response_model=ChatResponse,
    responses=ERROR_RESPONSES,
    dependencies=[Depends(rate_limit)],
    summary="Answer a medical question",
)
def chat(body: ChatRequest, request: Request, service: ChatService = Depends(get_chat_service)):
    _check_image_size(request, body)
    return service.ask(body.message, body.thread_id, body.image_data)


@router.post(
    "/chat/stream",
    responses=ERROR_RESPONSES,
    dependencies=[Depends(rate_limit)],
    summary="Answer a medical question, streaming one event per workflow step",
)
def chat_stream(
    body: ChatRequest, request: Request, service: ChatService = Depends(get_chat_service)
):
    _check_image_size(request, body)
    return StreamingResponse(
        _sse(service.stream(body.message, body.thread_id, body.image_data)),
        media_type="text/event-stream",
        # Disable proxy buffering so steps reach the browser as they happen.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get(
    "/conversations",
    response_model=ConversationList,
    summary="List recent conversations",
)
def list_conversations(
    limit: int = Query(default=50, ge=1, le=200),
    service: ChatService = Depends(get_chat_service),
):
    return {"conversations": service.conversations.list(limit)}


@router.get(
    "/conversations/{thread_id}",
    response_model=ConversationDetail,
    responses=ERROR_RESPONSES,
    summary="Read the messages of a conversation",
)
def get_conversation(thread_id: str = ThreadId, service: ChatService = Depends(get_chat_service)):
    return service.history(thread_id)


@router.delete(
    "/conversations/{thread_id}",
    status_code=204,
    responses=ERROR_RESPONSES,
    summary="Delete a conversation and its memory",
)
def delete_conversation(
    thread_id: str = ThreadId, service: ChatService = Depends(get_chat_service)
):
    service.delete(thread_id)
