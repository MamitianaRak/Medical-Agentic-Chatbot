from app.services.media import plain_text


def test_transcribe(client):
    response = client.post(
        "/api/speech/transcribe", files={"file": ("voice.webm", b"fake-audio", "audio/webm")}
    )
    assert response.status_code == 200
    assert response.json() == {"text": "What triggers asthma?"}


def test_transcribe_rejects_empty_audio(client):
    response = client.post(
        "/api/speech/transcribe", files={"file": ("voice.webm", b"", "audio/webm")}
    )
    assert response.status_code == 400


def test_synthesize_reads_plain_text_in_the_given_language(client, media):
    response = client.post(
        "/api/speech/synthesize",
        json={"text": "**L'asthme** est fréquent [PDF p.394].", "language": "fr"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.content == b"ID3-fake-mp3"
    assert media.spoken == [("L'asthme est fréquent .", "fr")]


def test_synthesize_falls_back_to_default_language(client, media):
    client.post("/api/speech/synthesize", json={"text": "Hello", "language": "xx"})
    assert media.spoken[-1][1] == "en"


def test_provider_failure_maps_to_503(client, media):
    media.fail = True
    response = client.post("/api/speech/synthesize", json={"text": "Hello"})

    assert response.status_code == 503
    assert response.json()["code"] == "media_unavailable"


def test_plain_text_strips_markdown_citations_and_urls():
    text = "## Causes\n- **Pollen** [PDF p.12] see https://example.org [https://x.org/a]"
    assert plain_text(text) == "Causes\n- Pollen see"
