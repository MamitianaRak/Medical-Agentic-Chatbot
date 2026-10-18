"""MCP-backed implementation of the research tools.

The tools come from the project's own MCP server (`app/mcp_servers/medical_server.py`),
reached over streamable HTTP (long-lived server) or stdio (spawned per call):
PubMed, openFDA drug labels, MedlinePlus and Tavily web search. Results are turned
into citable `Passage`s for the relevance grader.

Graph nodes are synchronous and run in worker threads (FastAPI threadpool),
so each call drives its own event loop with `asyncio.run`.
"""

import asyncio
import hashlib
import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from app.agents.state import Passage
from app.core.config import Settings

logger = logging.getLogger(__name__)

SERVER_NAME = "medical"
SERVER_PATH = Path(__file__).resolve().parents[1] / "mcp_servers" / "medical_server.py"

# Catalog name -> (MCP tool name, argument builder)
TOOLS = {
    "pubmed": ("search_pubmed", lambda query: {"query": query, "max_results": 4}),
    "drug_label": ("drug_label", lambda query: {"drug_name": query}),
    "medlineplus": ("health_topic", lambda query: {"term": query}),
    "web": ("web_search", lambda query: {"query": query, "max_results": 4}),
}


def mcp_connection(settings: Settings) -> dict[str, Any]:
    """How to reach the medical MCP server: a long-lived HTTP server, or a stdio process."""
    if settings.mcp_transport == "http":
        # The HTTP server reads its own keys from its environment.
        return {"transport": "streamable_http", "url": settings.mcp_url}

    env = {
        **os.environ,
        "TAVILY_API_KEY": settings.tavily_api_key,
        "NCBI_API_KEY": settings.ncbi_api_key,
    }
    return {
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(SERVER_PATH)],
        "env": env,
    }


def build_mcp_client(settings: Settings):
    from langchain_mcp_adapters.client import MultiServerMCPClient

    return MultiServerMCPClient({SERVER_NAME: mcp_connection(settings)})


def mcp_server_status(settings: Settings, timeout: float = 3.0) -> str:
    """Readiness: `stdio` (spawned on demand), `ok` if the HTTP server answers."""
    if settings.mcp_transport != "http":
        return "stdio"
    try:
        # Any HTTP answer (MCP endpoints reject a bare GET) proves the server is up.
        httpx.get(settings.mcp_url, timeout=timeout)
        return "ok"
    except httpx.HTTPError:
        return "unreachable"


def tool_payload(output: Any) -> dict[str, Any]:
    """MCP tools return JSON as text content blocks; adapters may wrap them in a list."""
    if isinstance(output, tuple):  # (content, artifact)
        output = output[0]
    if isinstance(output, list):
        output = "".join(
            block.get("text", "") if isinstance(block, dict) else getattr(block, "text", str(block))
            for block in output
        )
    if isinstance(output, dict):
        return output
    return json.loads(output)


def _short_id(prefix: str, value: str) -> str:
    return f"{prefix}-{hashlib.sha1(value.encode()).hexdigest()[:16]}"


def to_passages(tool: str, payload: dict[str, Any]) -> list[Passage]:
    if "error" in payload:
        raise RuntimeError(payload["error"])

    passages: list[Passage] = []
    for item in payload.get("results", []):
        if tool == "pubmed":
            details = ", ".join(x for x in (item.get("journal"), item.get("year")) if x)
            passages.append(
                {
                    "id": f"pubmed-{item['pmid']}",
                    "source": "pubmed",
                    "title": f"{item['title']} ({details})" if details else item["title"],
                    "text": item["abstract"],
                    "score": None,
                    "page": None,
                    "url": item["url"],
                    "label": f"PMID:{item['pmid']}",
                }
            )
        elif tool == "drug_label":
            name = item["generic_name"].split(",")[0].strip().lower()
            sections = "\n".join(f"{k}: {v}" for k, v in item.get("sections", {}).items())
            brand = f" ({item['brand_name']})" if item.get("brand_name") else ""
            passages.append(
                {
                    "id": _short_id("fda", item["url"] + name),
                    "source": "fda",
                    "title": f"FDA label: {item['generic_name']}{brand}",
                    "text": sections,
                    "score": None,
                    "page": None,
                    "url": item["url"],
                    "label": f"FDA: {name}",
                }
            )
        elif tool == "medlineplus":
            passages.append(
                {
                    "id": _short_id("medlineplus", item["url"]),
                    "source": "medlineplus",
                    "title": f"MedlinePlus: {item['title']}",
                    "text": item["summary"],
                    "score": None,
                    "page": None,
                    "url": item["url"],
                    "label": f"MedlinePlus: {item['title']}",
                }
            )
        elif tool == "web":
            domain = urlparse(item["url"]).netloc.removeprefix("www.") or "web"
            passages.append(
                {
                    "id": _short_id("web", item["url"]),
                    "source": "web",
                    "title": item["title"],
                    "text": item["content"],
                    "score": item.get("score"),
                    "page": None,
                    "url": item["url"],
                    "label": f"Web: {domain}",
                }
            )
    return passages


class McpResearchTools:
    def __init__(self, client: Any, *, web_enabled: bool, timeout_seconds: float = 30.0):
        self._client = client
        self._web_enabled = web_enabled
        self._timeout = timeout_seconds
        self._tools: dict[str, Any] | None = None
        self._lock = threading.Lock()

    def available(self) -> list[str]:
        return [name for name in TOOLS if name != "web" or self._web_enabled]

    def _server_tools(self) -> dict[str, Any]:
        with self._lock:
            if self._tools is None:
                tools = asyncio.run(self._client.get_tools(server_name=SERVER_NAME))
                self._tools = {tool.name: tool for tool in tools}
                logger.info("Loaded %d tools from the %s MCP server", len(tools), SERVER_NAME)
            return self._tools

    def run(self, tool: str, query: str) -> list[Passage]:
        if tool not in self.available():
            raise ValueError(f"Unknown or disabled research tool: {tool}")

        mcp_name, build_args = TOOLS[tool]
        mcp_tool = self._server_tools().get(mcp_name)
        if mcp_tool is None:
            raise RuntimeError(f"MCP tool '{mcp_name}' not found on the {SERVER_NAME} server")

        async def call() -> Any:
            return await asyncio.wait_for(mcp_tool.ainvoke(build_args(query)), self._timeout)

        return to_passages(tool, tool_payload(asyncio.run(call())))
