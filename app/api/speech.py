"""Voice endpoints: speech-to-text and text-to-speech."""

from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from app.api.deps import get_media_service, rate_limit, require_token
from app.api.schemas import ErrorResponse, SpeechRequest, TranscriptionResponse
from app.core.errors import AppError
from app.services.media import MediaService

router = APIRouter(
    prefix="/api/speech",
    tags=["speech"],
    dependencies=[Depends(require_token), Depends(rate_limit)],
)

ERROR_RESPONSES = {code: {"model": ErrorResponse} for code in (400, 401, 413, 429, 503)}


@router.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    responses=ERROR_RESPONSES,
    summary="Transcribe a recorded question (webm, mp3, wav...)",
)
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    media: MediaService = Depends(get_media_service),
):
    limit = request.app.state.settings.max_audio_bytes
    audio = await file.read(limit + 1)
    if not audio:
        raise AppError(400, "invalid_request", "The audio file is empty.")
    if len(audio) > limit:
        raise AppError(413, "audio_too_large", f"Audio must be under {limit // 1_000_000} MB.")
    # The provider call is blocking: run it outside the event loop.
    return {"text": await run_in_threadpool(media.transcribe, audio)}


@router.post(
    "/synthesize",
    response_class=Response,
    responses={200: {"content": {"audio/mpeg": {}}}, **ERROR_RESPONSES},
    summary="Read a text aloud (MP3)",
)
def synthesize(body: SpeechRequest, media: MediaService = Depends(get_media_service)):
    return Response(content=media.synthesize(body.text, body.language), media_type="audio/mpeg")
