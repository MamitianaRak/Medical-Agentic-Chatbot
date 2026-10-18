# MedGuide AI — LangGraph + Pinecone + FastAPI

An educational medical assistant with retrieval from the bundled historical Gale Encyclopedia of Medicine (2002, Volume 1 A–B), optional Tavily fallback, image observations, speech input/output, and a vanilla JS web UI.

## Requirements
Python 3.11 or 3.12; Google Gemini, Cloudflare Workers AI, Pinecone, and Tavily API keys; internet connection.

| Task | Model |
|---|---|
| Answer generation | Gemini (`gemini-3.5-flash-lite`) |
| Embeddings | Pinecone Inference `llama-text-embed-v2` (768 dims) |
| Speech-to-text | Cloudflare `@cf/openai/whisper-large-v3-turbo` |
| Image observations | Cloudflare `@cf/meta/llama-3.2-11b-vision-instruct` |
| Text-to-speech | Cloudflare `@cf/myshell-ai/melotts` |

`.env` keys: `GOOGLE_API_KEY`, `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN` (Workers AI permission), `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, optional `HR_NAMESPACE` (Pinecone namespace), optional `TAVILY_API_KEY`, optional `TTS_LANG` (`en`, `fr`, `es`, …). Model names can be overridden in `.env` (see `app/config.py`). The Llama 3.2 Vision license is accepted automatically on first image request. This is a tutorial reference implementation, **not** a clinically validated diagnostic system or HIPAA-compliant deployment.

## Start
```bash
conda create -n project python=3.11 -y

conda activate project

pip install -r requirements.txt

add  .env

# Edit .env with real API keys

python ingest.py --pdf data/medical_book.pdf

uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`, then install and run the same commands.

Open **http://127.0.0.1:8000**. API documentation: **http://127.0.0.1:8000/docs**. `/api/health` checks configuration. Run `ingest.py` once before chatting; it makes paid embedding calls and creates a Pinecone serverless index. The Pinecone index (`medical-agent-gemini-768`) uses 768 dimensions for `llama-text-embed-v2`. If using a different embedding model or dimension, set `EMBED_DIMENSIONS` and use a fresh index.

## Structure
- `app/config.py` — environment and Conda SSL certificate workaround
- `app/knowledge.py` — Pinecone vector retrieval
- `app/agent.py` — LangGraph retrieve → route → Tavily (when needed) → answer
- `app/cloudflare.py` — Workers AI REST client
- `app/main.py` — FastAPI, image observations (Llama Vision), Whisper and MeloTTS endpoints
- `ingest.py` — chunk and embed every readable PDF page
- `templates/index.html` — UI
- `static/style.css`, `static/app.js` — styles and browser controls
- `data/medical_book.pdf` — user-supplied reference PDF

## Troubleshooting
- **Connecting / checking**: run `uvicorn` and open `/api/health` first. Check browser console and backend terminal.
- **401**: if `API_TOKEN` is configured, enter it in the UI's API access section. Do not expose this token in a public browser deployment.
- **502**: inspect backend terminal; verify Google/Cloudflare/Pinecone keys, Pinecone service availability, embedding dimension and SSL certificate.
- **Camera/microphone**: browser permissions and localhost or HTTPS required.
- **No KB results**: run `python ingest.py` and wait for successful completion; ensure same index and API keys.
- **Tavily**: optional but needed for external fallback. Without its key, external search is disabled.

## Security and production limitations
This project is a functional tutorial scaffold, not a production medical device. Before deploying to real patients: replace shared API token with server-side identity/authentication; add database-backed encrypted conversations and retention/deletion controls; enforce request throttling and upload content verification; implement observability, tests, clinically reviewed emergency triage and evaluation; verify licensing to host/index the Gale PDF; obtain privacy/security/legal review; ensure current authoritative clinical references; and secure the deployment behind HTTPS. Do not send identifiable patient data to third-party APIs without appropriate consent and agreements.

The historical book may contain obsolete clinical advice; model instructions do not guarantee safety. Images can be described but cannot confirm or rule out disease. This software does not make diagnoses.
