"""LLM factory."""

from langchain_core.language_models import BaseChatModel

from app.core.config import Settings


def build_llm(settings: Settings) -> BaseChatModel:
    # Imported here so tests (fake LLM) never need the Google SDK.
    from langchain_google_genai import ChatGoogleGenerativeAI

    # Every agent step asks for a JSON object, so JSON mode is enforced at the API level.
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        response_mime_type="application/json",
    )
