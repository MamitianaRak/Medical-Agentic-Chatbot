"""Custom MCP server exposing read-only medical research APIs.

Tools:
- search_pubmed    PubMed / NCBI E-utilities (peer-reviewed literature)
- drug_label       openFDA drug labels (indications, warnings, interactions; no dosing)
- health_topic     MedlinePlus health topics (consumer health, NIH)
- web_search       Tavily web search (only when TAVILY_API_KEY is set)

Two transports:
    python app/mcp_servers/medical_server.py                       # stdio, spawned per call
    python -m app.mcp_servers.medical_server --transport http      # long-lived HTTP server
The HTTP mode serves streamable HTTP on http://<host>:<port>/mcp; it is what the
Docker image and Render use, so memory and startup cost are paid once.

Every tool returns {"results": [...]} or {"error": "..."}; it never raises, so a
failing API degrades one tool instead of the whole request.
"""

import argparse
import html
import os
import re
import xml.etree.ElementTree as ET
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

TIMEOUT_SECONDS = 15
TEXT_LIMIT = 1500
SECTION_LIMIT = 700

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
OPENFDA = "https://api.fda.gov/drug/label.json"
MEDLINEPLUS = "https://wsearch.nlm.nih.gov/ws/query"
TAVILY = "https://api.tavily.com/search"

# Label sections worth showing; dosing is deliberately excluded (no prescribing).
FDA_SECTIONS = {
    "indications_and_usage": "Indications",
    "boxed_warning": "Boxed warning",
    "warnings": "Warnings",
    "warnings_and_cautions": "Warnings and precautions",
    "contraindications": "Contraindications",
    "do_not_use": "Do not use",
    "stop_use": "Stop use",
    "drug_interactions": "Interactions",
    "adverse_reactions": "Adverse reactions",
}

mcp = FastMCP("MedGuide Medical Research")


def _client() -> httpx.Client:
    return httpx.Client(timeout=TIMEOUT_SECONDS, follow_redirects=True)


def _clean(text: str, limit: int = TEXT_LIMIT) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", " ", text or ""))
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def _error(service: str, exc: Exception) -> dict[str, Any]:
    return {"error": f"{service} request failed: {type(exc).__name__}"}


@mcp.tool()
def search_pubmed(query: str, max_results: int = 4) -> dict[str, Any]:
    """Search PubMed for peer-reviewed articles. Returns titles, journals, years and abstracts.
    Best for current evidence, treatments, guidelines and research questions."""
    params: dict[str, Any] = {"db": "pubmed", "term": query, "retmax": max(1, min(max_results, 8))}
    if os.getenv("NCBI_API_KEY"):
        params["api_key"] = os.environ["NCBI_API_KEY"]

    try:
        with _client() as client:
            found = client.get(
                f"{EUTILS}/esearch.fcgi", params={**params, "retmode": "json", "sort": "relevance"}
            )
            found.raise_for_status()
            ids = found.json()["esearchresult"]["idlist"]
            if not ids:
                return {"results": []}
            fetched = client.get(
                f"{EUTILS}/efetch.fcgi",
                params={k: v for k, v in params.items() if k != "term"}
                | {"id": ",".join(ids), "retmode": "xml"},
            )
            fetched.raise_for_status()
            root = ET.fromstring(fetched.text)
    except (httpx.HTTPError, KeyError, ValueError, ET.ParseError) as exc:
        return _error("PubMed", exc)

    results = []
    for article in root.iter("PubmedArticle"):
        pmid = article.findtext(".//PMID") or ""
        abstract = " ".join("".join(part.itertext()) for part in article.iter("AbstractText"))
        if not abstract:
            continue
        title = article.find(".//ArticleTitle")
        results.append(
            {
                "pmid": pmid,
                "title": _clean("".join(title.itertext()), 300) if title is not None else "",
                "journal": article.findtext(".//Journal/Title") or "",
                "year": article.findtext(".//JournalIssue/PubDate/Year") or "",
                "abstract": _clean(abstract),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            }
        )
    return {"results": results}


@mcp.tool()
def drug_label(drug_name: str) -> dict[str, Any]:
    """Official FDA label of a medicine (generic or brand name): indications, warnings,
    contraindications, interactions and adverse reactions. Dosing is not returned."""
    name = re.sub(r"[^\w\s-]", "", drug_name).strip()
    if not name:
        return {"error": "A drug name is required."}
    # openFDA expects literal '+' between clauses, so the query string is built by hand.
    search = f'openfda.generic_name:"{name}"+OR+openfda.brand_name:"{name}"'

    try:
        with _client() as client:
            response = client.get(f"{OPENFDA}?search={search}&limit=1")
            if response.status_code == 404:
                return {"results": []}
            response.raise_for_status()
            label = response.json()["results"][0]
    except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
        return _error("openFDA", exc)

    sections = {
        title: _clean(" ".join(label[key]), SECTION_LIMIT)
        for key, title in FDA_SECTIONS.items()
        if label.get(key)
    }
    openfda = label.get("openfda", {})
    set_id = label.get("set_id", "")
    return {
        "results": [
            {
                "generic_name": ", ".join(openfda.get("generic_name", [])) or name,
                "brand_name": ", ".join(openfda.get("brand_name", [])),
                "sections": sections,
                "url": f"https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid={set_id}"
                if set_id
                else "https://open.fda.gov/apis/drug/label/",
            }
        ]
    }


@mcp.tool()
def health_topic(term: str, language: str = "en", max_results: int = 2) -> dict[str, Any]:
    """MedlinePlus (US National Library of Medicine) health topic summaries written for
    patients: symptoms, causes, diagnosis, treatment and prevention. language: en or es."""
    database = "healthTopicsSpanish" if language == "es" else "healthTopics"

    try:
        with _client() as client:
            response = client.get(
                MEDLINEPLUS,
                params={"db": database, "term": term, "retmax": max(1, min(max_results, 5))},
            )
            response.raise_for_status()
            root = ET.fromstring(response.text)
    except (httpx.HTTPError, ET.ParseError) as exc:
        return _error("MedlinePlus", exc)

    results = []
    for document in root.iter("document"):
        fields = {c.get("name"): "".join(c.itertext()) for c in document.findall("content")}
        if not fields.get("FullSummary"):
            continue
        results.append(
            {
                "title": _clean(fields.get("title", ""), 200),
                "summary": _clean(fields["FullSummary"]),
                "url": document.get("url", ""),
            }
        )
    return {"results": results}


@mcp.tool()
def web_search(query: str, max_results: int = 4) -> dict[str, Any]:
    """General web search (Tavily). Use for recent news, official health agency pages or
    anything the other tools do not cover."""
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        return {"error": "Web search is not configured (TAVILY_API_KEY)."}

    try:
        with _client() as client:
            response = client.post(
                TAVILY,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "query": query,
                    "max_results": max(1, min(max_results, 8)),
                    "search_depth": "basic",
                },
            )
            response.raise_for_status()
            items = response.json().get("results", [])
    except (httpx.HTTPError, ValueError) as exc:
        return _error("Tavily", exc)

    return {
        "results": [
            {
                "title": item.get("title") or item.get("url", ""),
                "content": _clean(item.get("content", "")),
                "url": item.get("url", ""),
                "score": round(float(item.get("score") or 0.0), 3),
            }
            for item in items
            if item.get("content")
        ]
    }


def http_security(extra_hosts: str = "") -> TransportSecuritySettings:
    """Keep DNS rebinding protection on; allow localhost plus explicit internal hosts
    (e.g. `mcp:*`, the service name in docker compose)."""
    extra = [host.strip() for host in extra_hosts.split(",") if host.strip()]
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=["127.0.0.1:*", "localhost:*", *extra],
        allowed_origins=["http://127.0.0.1:*", "http://localhost:*"],
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="MedGuide medical research MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default=os.getenv("MCP_SERVER_TRANSPORT", "stdio"),
    )
    parser.add_argument("--host", default=os.getenv("MCP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MCP_PORT", "8001")))
    args = parser.parse_args(argv)

    if args.transport == "stdio":
        mcp.run()
        return

    mcp.settings.host = args.host
    mcp.settings.port = args.port
    # Each tool call is independent: no session state to keep between requests.
    mcp.settings.stateless_http = True
    mcp.settings.transport_security = http_security(os.getenv("MCP_ALLOWED_HOSTS", ""))
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    from dotenv import load_dotenv

    # Allows running the server by hand with keys from `.env`.
    load_dotenv()
    main()
