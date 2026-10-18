"""Application errors and their mapping to HTTP responses."""

from dataclasses import dataclass, field


@dataclass(eq=False)
class AppError(Exception):
    status_code: int
    code: str
    message: str
    headers: dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.message


class ProviderError(Exception):
    """An upstream AI provider (Workers AI, Pinecone, ...) returned an error."""

    def __init__(self, provider: str, message: str, status_code: int | None = None):
        super().__init__(f"{provider}: {message}")
        self.provider = provider
        self.status_code = status_code


def classify_provider_error(exc: Exception) -> AppError | None:
    """Map upstream LLM/network failures to actionable client errors.

    Matching on class names and messages keeps this module free of provider SDK imports.
    """
    if isinstance(exc, AppError):
        return exc

    name = type(exc).__name__
    text = str(exc).lower()

    rate_limited = (
        name == "RateLimitError"
        or "resource_exhausted" in text
        or "rate limit" in text
        or "quota" in text
        or getattr(exc, "status_code", None) == 429
    )
    if rate_limited:
        return AppError(
            503,
            "provider_rate_limited",
            "The AI provider is rate-limiting requests. Please wait a minute and try again.",
        )

    if "Timeout" in name or isinstance(exc, TimeoutError) or "deadline_exceeded" in text:
        return AppError(
            504,
            "upstream_timeout",
            "An external service took too long to respond. Please try again.",
        )

    if name in {"APIConnectionError", "ConnectError", "ConnectionError"}:
        return AppError(
            503,
            "upstream_unavailable",
            "An external service is unreachable right now. Please try again later.",
        )

    if isinstance(exc, ProviderError):
        return AppError(
            502,
            "provider_error",
            "An AI provider returned an error. Please try again later.",
        )

    return None
