"""Exception handlers returning a consistent `{success, error, code}` body."""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.errors import AppError, classify_provider_error

logger = logging.getLogger(__name__)


def error_body(code: str, message: str) -> dict[str, object]:
    return {"success": False, "error": message, "code": code}


def to_app_error(exc: Exception) -> AppError:
    """Known failures keep their message; anything else gets a generic one."""
    classified = classify_provider_error(exc)

    if classified is not None:
        return classified

    logger.exception("Unhandled error", exc_info=exc)
    return AppError(500, "internal_error", "Something went wrong while answering your question.")


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(exc.code, exc.message),
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        message = str(first.get("msg", "Invalid request.")).removeprefix("Value error, ")
        return JSONResponse(status_code=400, content=error_body("invalid_request", message))

    @app.exception_handler(Exception)
    async def handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        error = to_app_error(exc)
        return JSONResponse(
            status_code=error.status_code, content=error_body(error.code, error.message)
        )
