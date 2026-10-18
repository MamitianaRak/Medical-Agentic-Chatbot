"""Speech and vision use cases, independent of the provider."""

import logging
import re
from typing import Protocol

from app.agents.state import SUPPORTED_LANGUAGES
from app.core.errors import AppError, ProviderError, classify_provider_error

logger = logging.getLogger(__name__)

MAX_SPOKEN_CHARS = 3000


class MediaProvider(Protocol):
    def transcribe(self, audio: bytes) -> str: ...

    def describe_image(self, data_url: str) -> str: ...

    def synthesize(self, text: str, language: str) -> bytes: ...


def plain_text(text: str) -> str:
    """Strip Markdown, citations and URLs so the voice does not read symbols."""
    text = re.sub(r"\[(?:PDF p\.\d+|https?://[^\]]+)\]", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[*_#`>|\[\]]+", "", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def _provider_failure(exc: Exception, what: str) -> AppError:
    logger.warning("%s failed: %s", what, exc)
    classified = classify_provider_error(exc)
    if classified is not None and not isinstance(exc, ProviderError):
        return classified
    return AppError(503, "media_unavailable", f"{what} is unavailable right now.")


class MediaService:
    def __init__(self, provider: MediaProvider, default_language: str = "en"):
        self.provider = provider
        self.default_language = default_language

    def transcribe(self, audio: bytes) -> str:
        try:
            return self.provider.transcribe(audio)
        except Exception as exc:
            raise _provider_failure(exc, "Speech transcription") from exc

    def describe_image(self, data_url: str) -> str:
        try:
            return self.provider.describe_image(data_url)
        except Exception as exc:
            raise _provider_failure(exc, "Image analysis") from exc

    def synthesize(self, text: str, language: str | None) -> bytes:
        speech = plain_text(text)[:MAX_SPOKEN_CHARS]
        if not speech:
            raise AppError(400, "invalid_request", "There is no text to read aloud.")
        lang = language if language in SUPPORTED_LANGUAGES else self.default_language
        try:
            return self.provider.synthesize(speech, lang)
        except Exception as exc:
            raise _provider_failure(exc, "Speech synthesis") from exc
