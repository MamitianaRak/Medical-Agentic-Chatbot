#!/bin/sh
# Container entry point.
# Starts the medical MCP server once, next to the API, so tool calls reuse a
# long-lived process instead of spawning one per call (memory and CPU matter on
# small instances such as Render's free plan). Then uvicorn becomes PID 1.
set -e

if [ "${MCP_TRANSPORT:-stdio}" = "http" ] && [ "${START_MCP_SERVER:-true}" = "true" ]; then
    python -m app.mcp_servers.medical_server --transport http \
        --host 127.0.0.1 --port "${MCP_PORT:-8001}" &
fi

exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" \
    --proxy-headers --forwarded-allow-ips='*'
