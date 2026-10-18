import pytest

from app.agents.text import content_text, json_from_llm
from app.core.errors import AppError, ProviderError, classify_provider_error
from app.core.rate_limit import RateLimiter


@pytest.mark.parametrize(
    "raw",
    [
        '{"urgency": "routine"}',
        '```json\n{"urgency": "routine"}\n```',
        'Sure! Here it is: {"urgency": "routine"} Hope it helps.',
    ],
)
def test_json_from_llm_tolerates_wrappers(raw):
    assert json_from_llm(raw) == {"urgency": "routine"}


@pytest.mark.parametrize("raw", ["no json here", "[1, 2]", '{"broken": '])
def test_json_from_llm_rejects_invalid_output(raw):
    with pytest.raises(ValueError):
        json_from_llm(raw)


def test_json_from_llm_repairs_invalid_escapes():
    raw = r'{"answer": "l\'asthme s\évère\n"}'
    assert json_from_llm(raw) == {"answer": "l'asthme sévère\n"}


def test_content_text_joins_content_blocks():
    assert content_text([{"type": "text", "text": "a"}, "b"]) == "ab"


def test_rate_limiter_sliding_window():
    now = [0.0]
    limiter = RateLimiter(2, 60, clock=lambda: now[0])

    assert limiter.check("ip") is None
    assert limiter.check("ip") is None
    assert limiter.check("ip") == pytest.approx(60)
    now[0] = 61
    assert limiter.check("ip") is None


def test_rate_limiter_disabled_with_zero():
    limiter = RateLimiter(0, 60)
    assert all(limiter.check("ip") is None for _ in range(100))


@pytest.mark.parametrize(
    ("exc", "code"),
    [
        (RuntimeError("429 RESOURCE_EXHAUSTED"), "provider_rate_limited"),
        (TimeoutError("slow"), "upstream_timeout"),
        (ProviderError("workers-ai", "boom", 500), "provider_error"),
    ],
)
def test_classify_provider_error(exc, code):
    assert classify_provider_error(exc).code == code


def test_classify_unknown_error_returns_none():
    assert classify_provider_error(ValueError("bug")) is None


def test_app_error_passes_through():
    error = AppError(404, "x", "missing")
    assert classify_provider_error(error) is error
