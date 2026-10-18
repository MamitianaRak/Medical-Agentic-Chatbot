"""Speech and vision through Cloudflare Workers AI (REST API)."""

import base64
import logging
import threading
from typing import Any

import httpx

from app.core.config import Settings
from app.core.errors import ProviderError

logger = logging.getLogger(__name__)

API_URL = "https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{model}"

VISION_PROMPT = (
    "Describe only objectively visible, medically relevant features of this image. "
    "Do not diagnose, estimate severity, or claim the absence of illness. "
    "If the image is unclear or not medical, say so."
)


class WorkersAI:
    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        self.settings = settings
        self._client = client or httpx.Client(timeout=settings.cloudflare_timeout_seconds)
        self._licensed: set[str] = set()
        self._lock = threading.Lock()

    def _run(self, model: str, payload: dict[str, Any]) -> Any:
        """Return the JSON `result`, or raw bytes for binary responses (audio)."""
        url = API_URL.format(account=self.settings.cloudflare_account_id, model=model)
        response = self._client.post(
            url,
            headers={"Authorization": f"Bearer {self.settings.cloudflare_api_token}"},
            json=payload,
        )
        if response.is_error:
            raise ProviderError(
                "workers-ai", f"{model} returned {response.status_code}", response.status_code
            )
        if response.headers.get("content-type", "").startswith("application/json"):
            return response.json().get("result", {})
        return response.content

    def _accept_license(self, model: str) -> None:
        """Meta Llama models need a one-time license acceptance per Cloudflare account."""
        with self._lock:
            if model in self._licensed:
                return
            try:
                self._run(model, {"prompt": "agree"})
            except ProviderError as exc:
                # Accounts that already accepted may answer with an error; the real call decides.
                logger.info("License acceptance for %s: %s", model, exc)
            self._licensed.add(model)

    def transcribe(self, audio: bytes) -> str:
        result = self._run(
            self.settings.cf_stt_model, {"audio": base64.b64encode(audio).decode("ascii")}
        )
        return str(result.get("text", "")).strip()

    def describe_image(self, data_url: str) -> str:
        model = self.settings.cf_vision_model
        self._accept_license(model)
        result = self._run(
            model,
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": VISION_PROMPT},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
                "max_tokens": 300,
                "temperature": 0.2,
            },
        )
        return str(result.get("response", "")).strip()

    def synthesize(self, text: str, language: str) -> bytes:
        result = self._run(self.settings.cf_tts_model, {"prompt": text, "lang": language})
        # MeloTTS answers either {"audio": "<base64 mp3>"} or raw audio/mpeg bytes.
        if isinstance(result, dict):
            return base64.b64decode(result.get("audio", ""))
        return result
