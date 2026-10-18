FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000 \
    DATABASE_PATH=/srv/medguide/data/medguide.db \
    MCP_TRANSPORT=http \
    MCP_URL=http://127.0.0.1:8001/mcp

WORKDIR /srv/medguide

# Non-root user; it owns the data directory where SQLite keeps conversations.
RUN useradd --create-home --uid 10001 medguide

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY scripts ./scripts
RUN mkdir -p data && chown medguide:medguide data

USER medguide

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/health', timeout=4)"

# Starts the MCP research server (HTTP, localhost only) then uvicorn as PID 1.
# Render injects $PORT; proxy headers are trusted because the platform terminates TLS.
CMD ["sh", "scripts/start.sh"]
