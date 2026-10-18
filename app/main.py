import base64
import re
import secrets
from pathlib import Path
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from uuid import uuid4
from app.config import settings
from app.agent import graph
from app import cloudflare

ROOT=Path(__file__).resolve().parent.parent
app=FastAPI(title='Medical Agent Educational API',version='1.0.0')
app.add_middleware(CORSMiddleware,allow_origins=settings.allowed_origins.split(','),allow_credentials=False,allow_methods=['GET','POST'],allow_headers=['Authorization','Content-Type'])
app.mount('/static',StaticFiles(directory=ROOT/'static'),name='static')


class ChatRequest(BaseModel):
    message: str = Field(min_length=1,max_length=4000)
    history: list[dict[str,str]] = Field(default_factory=list,max_length=10)
    image_data: str | None = None

def auth(authorization: str | None):
    if settings.api_token and not secrets.compare_digest(authorization or '',f'Bearer {settings.api_token}'):
        raise HTTPException(status_code=401,detail='Invalid API token')


def inspect_image(data):
    if not data:
        return ''
    try:
        header,payload=data.split(',',1)
        mime=header.split(';')[0].split(':')[1]
        raw=base64.b64decode(payload,validate=True)
        if mime not in ('image/jpeg','image/png','image/webp') or len(raw)>5_000_000:
            raise ValueError('Only JPEG/PNG/WebP under 5MB supported')
    except (ValueError,IndexError) as exc:
        raise HTTPException(status_code=400,detail=f'Invalid image: {exc}')
    cloudflare.accept_license(settings.cf_vision_model)
    result=cloudflare.run(settings.cf_vision_model,{'messages':[{'role':'user','content':[{'type':'text','text':'Describe only objectively visible medical-relevant features. Do not diagnose, estimate severity, or assert absence of illness. If image unclear, say so.'},{'type':'image_url','image_url':{'url':data}}]}],'max_tokens':300,'temperature':0.2})
    return (result.get('response') or '').strip()


@app.get('/',response_class=HTMLResponse)
def home():
    return (ROOT/'templates'/'index.html').read_text(encoding='utf-8')

@app.get('/api/health')
def health():
    return {'status':'ok','configured':settings.configured,'tavily':bool(settings.tavily_api_key)}



@app.post('/api/chat')
def chat(
    payload: ChatRequest,
    authorization: str | None = Header(default=None),
    conversation_id: str | None = Header(default=None, alias='X-Conversation-ID')
):
    auth(authorization)

    if not settings.configured:
        raise HTTPException(503,'Set GOOGLE_API_KEY, PINECONE_API_KEY, CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN in .env')

    history=[x for x in payload.history if x.get('role') in ('user','assistant') and isinstance(x.get('content'),str)][:10]

    # SQLite persistent memory: reuse the same conversation thread
    thread_id = conversation_id or str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    try:
        observation=inspect_image(payload.image_data)

        state=graph.invoke(
            {
                'question':payload.message,
                'history':history,
                'image_description':observation,
                'kb':[],
                'web':[],
                'answer':'',
                'route':''
            },
            config=config
        )

        return {
            'answer':state['answer'],
            'sources': [{'type':'pdf','page':x['page'],'score':x['score']} for x in state['kb']]+[{'type':'web','url':x['url'],'title':x['title']} for x in state['web']],
            'tools': ['pinecone']+(['tavily'] if state['route']=='web' and settings.tavily_api_key else [])+(['vision'] if observation else []),
            'conversation_id':thread_id,
            'lang':state.get('lang') or settings.tts_lang,
            'spoken':state.get('spoken') or ''
        }

    except HTTPException:
        raise
    except Exception as exc:
        print('Chat error:',repr(exc),flush=True)
        raise HTTPException(502,'AI service failed. Check server logs, credentials, Pinecone index, and connectivity.')


@app.post('/api/transcribe')
async def transcribe(file:UploadFile=File(...),authorization:str|None=Header(default=None)):
    auth(authorization)
    data=await file.read(10_000_001)
    if len(data)>10_000_000:
        raise HTTPException(413,'Audio exceeds 10 MB')
    try:
        result=cloudflare.run(settings.cf_stt_model,{'audio':base64.b64encode(data).decode()})
        return {'text':(result.get('text') or '').strip()}
    except Exception as exc:
        print('Transcription error:',repr(exc),flush=True)
        raise HTTPException(502,'Speech transcription unavailable')

TTS_LANGS={'en','fr','es','zh','ja','ko'}


def plain_text(text):
    # Strip Markdown and URLs so the voice does not read symbols
    text=re.sub(r'https?://\S+','',text)
    return re.sub(r'[*_#`>|\[\]]+','',text).strip()


@app.post('/api/speak')
def speak(payload:dict,authorization:str|None=Header(default=None)):
    auth(authorization)
    text=plain_text(str(payload.get('text','')))[:3000]
    if not text:
        raise HTTPException(400,'Missing text')
    # Language detected by Gemini alongside the answer (see agent.Reply)
    lang=payload.get('lang') if payload.get('lang') in TTS_LANGS else settings.tts_lang
    try:
        result=cloudflare.run(settings.cf_tts_model,{'prompt':text,'lang':lang})
        # MeloTTS returns either {'audio': <base64 mp3>} or raw audio/mpeg bytes
        audio=base64.b64decode(result['audio']) if isinstance(result,dict) else result
        return Response(content=audio,media_type='audio/mpeg')
    except Exception as exc:
        print('Speech synthesis error:',repr(exc),flush=True)
        raise HTTPException(502,'Speech synthesis unavailable')
