"""MCP research server and client, with the network mocked out."""

import json

import httpx
import pytest

from app.core.config import Settings
from app.integrations.mcp_toolbox import (
    McpResearchTools,
    mcp_connection,
    mcp_server_status,
    to_passages,
    tool_payload,
)
from app.mcp_servers import medical_server

# --- Client: tool loading, calls and result conversion --------------------------------


class FakeTool:
    def __init__(self, name: str, payload: dict):
        self.name = name
        self.payload = payload
        self.received: list[dict] = []

    async def ainvoke(self, args: dict):
        self.received.append(args)
        # langchain-mcp-adapters returns MCP text content blocks.
        return [{"type": "text", "text": json.dumps(self.payload)}]


class FakeMcpClient:
    def __init__(self, tools: list[FakeTool]):
        self.tools = tools
        self.loads = 0

    async def get_tools(self, server_name: str):
        self.loads += 1
        return self.tools


PUBMED = {
    "results": [
        {
            "pmid": "34655640",
            "title": "Advances in asthma",
            "journal": "J Allergy Clin Immunol",
            "year": "2021",
            "abstract": "Biologics reduce exacerbations.",
            "url": "https://pubmed.ncbi.nlm.nih.gov/34655640/",
        }
    ]
}


def test_toolbox_calls_the_mcp_tool_and_returns_citable_passages():
    tool = FakeTool("search_pubmed", PUBMED)
    client = FakeMcpClient([tool])
    tools = McpResearchTools(client, web_enabled=False)

    passages = tools.run("pubmed", "asthma biologics")
    tools.run("pubmed", "asthma")

    assert tool.received[0] == {"query": "asthma biologics", "max_results": 4}
    assert client.loads == 1  # tools are loaded once
    assert passages[0]["label"] == "PMID:34655640"
    assert passages[0]["title"] == "Advances in asthma (J Allergy Clin Immunol, 2021)"
    assert passages[0]["score"] is None


def test_web_tool_is_only_available_with_a_tavily_key():
    assert "web" not in McpResearchTools(FakeMcpClient([]), web_enabled=False).available()
    assert "web" in McpResearchTools(FakeMcpClient([]), web_enabled=True).available()

    with pytest.raises(ValueError, match="web"):
        McpResearchTools(FakeMcpClient([]), web_enabled=False).run("web", "q")


def test_tool_error_payload_raises():
    tool = FakeTool("drug_label", {"error": "openFDA request failed"})
    tools = McpResearchTools(FakeMcpClient([tool]), web_enabled=False)

    with pytest.raises(RuntimeError, match="openFDA"):
        tools.run("drug_label", "ibuprofen")


def test_missing_mcp_tool_is_reported():
    tools = McpResearchTools(FakeMcpClient([]), web_enabled=False)
    with pytest.raises(RuntimeError, match="health_topic"):
        tools.run("medlineplus", "asthma")


@pytest.mark.parametrize(
    "output",
    [
        '{"results": []}',
        [{"type": "text", "text": '{"results": []}'}],
        ([{"type": "text", "text": '{"results": []}'}], None),
        {"results": []},
    ],
)
def test_tool_payload_accepts_every_adapter_shape(output):
    assert tool_payload(output) == {"results": []}


def test_conversions_for_every_tool():
    fda = to_passages(
        "drug_label",
        {
            "results": [
                {
                    "generic_name": "IBUPROFEN",
                    "brand_name": "Advil",
                    "sections": {"Warnings": "Stomach bleeding."},
                    "url": "https://dailymed.nlm.nih.gov/x",
                }
            ]
        },
    )[0]
    medline = to_passages(
        "medlineplus",
        {"results": [{"title": "Asthma", "summary": "Lung disease.", "url": "https://m/a"}]},
    )[0]
    web = to_passages(
        "web",
        {
            "results": [
                {
                    "title": "GINA",
                    "content": "Guide.",
                    "url": "https://www.ginasthma.org/r",
                    "score": 0.9,
                }
            ]
        },
    )[0]

    assert (fda["label"], fda["title"]) == ("FDA: ibuprofen", "FDA label: IBUPROFEN (Advil)")
    assert "Warnings: Stomach bleeding." in fda["text"]
    assert medline["label"] == "MedlinePlus: Asthma"
    assert (web["label"], web["score"]) == ("Web: ginasthma.org", 0.9)


# --- Server: API calls and parsing (httpx mocked) -------------------------------------


def mock_http(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        medical_server, "_client", lambda: httpx.Client(transport=transport, timeout=5)
    )


EFETCH_XML = """<PubmedArticleSet><PubmedArticle><MedlineCitation>
<PMID>123</PMID><Article><Journal><JournalIssue><PubDate><Year>2024</Year></PubDate>
</JournalIssue><Title>Lancet</Title></Journal><ArticleTitle>Asthma <i>update</i></ArticleTitle>
<Abstract><AbstractText>Inhaled steroids remain first line.</AbstractText></Abstract>
</Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"""


def test_search_pubmed_parses_esearch_and_efetch(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if "esearch" in request.url.path:
            assert request.url.params["term"] == "asthma"
            return httpx.Response(200, json={"esearchresult": {"idlist": ["123"]}})
        return httpx.Response(200, text=EFETCH_XML)

    mock_http(monkeypatch, handler)
    result = medical_server.search_pubmed("asthma")

    assert result == {
        "results": [
            {
                "pmid": "123",
                "title": "Asthma update",
                "journal": "Lancet",
                "year": "2024",
                "abstract": "Inhaled steroids remain first line.",
                "url": "https://pubmed.ncbi.nlm.nih.gov/123/",
            }
        ]
    }


def test_drug_label_keeps_safety_sections_and_never_dosing(monkeypatch):
    label = {
        "openfda": {"generic_name": ["IBUPROFEN"], "brand_name": ["Advil"]},
        "set_id": "abc",
        "indications_and_usage": ["Temporarily relieves pain."],
        "warnings": ["<b>Stomach bleeding</b> warning."],
        "dosage_and_administration": ["Take 1 tablet every 4 hours."],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        # Literal '+' between clauses (not %2B): openFDA requires it.
        assert "generic_name:%22ibuprofen%22+OR+openfda.brand_name" in str(request.url)
        return httpx.Response(200, json={"results": [label]})

    mock_http(monkeypatch, handler)
    item = medical_server.drug_label("ibuprofen")["results"][0]

    assert item["sections"] == {
        "Indications": "Temporarily relieves pain.",
        "Warnings": "Stomach bleeding warning.",
    }
    assert item["url"].endswith("setid=abc")


def test_drug_label_not_found_returns_no_results(monkeypatch):
    mock_http(monkeypatch, lambda request: httpx.Response(404, json={"error": {}}))
    assert medical_server.drug_label("notadrug") == {"results": []}


MEDLINE_XML = """<nlmSearchResult><list><document url="https://medlineplus.gov/asthma.html">
<content name="title">&lt;span class="qt0"&gt;Asthma&lt;/span&gt;</content>
<content name="FullSummary">&lt;p&gt;Asthma is a chronic disease.&lt;/p&gt;</content>
</document></list></nlmSearchResult>"""


def test_health_topic_strips_html(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["db"] == "healthTopicsSpanish"
        return httpx.Response(200, text=MEDLINE_XML)

    mock_http(monkeypatch, handler)
    result = medical_server.health_topic("asma", language="es")

    assert result["results"] == [
        {
            "title": "Asthma",
            "summary": "Asthma is a chronic disease.",
            "url": "https://medlineplus.gov/asthma.html",
        }
    ]


def test_web_search_requires_a_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert "error" in medical_server.web_search("asthma")


def test_web_search_uses_tavily(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer tvly-test"
        results = [
            {"title": "GINA", "url": "https://ginasthma.org", "content": "Guide", "score": 0.8}
        ]
        return httpx.Response(200, json={"results": results})

    mock_http(monkeypatch, handler)
    assert medical_server.web_search("asthma")["results"][0]["title"] == "GINA"


def test_api_failure_becomes_an_error_payload(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    mock_http(monkeypatch, handler)
    assert medical_server.search_pubmed("asthma") == {
        "error": "PubMed request failed: ConnectError"
    }


# --- Transports: stdio (spawned per call) or long-lived HTTP server --------------------


def test_stdio_connection_spawns_the_server_with_the_keys():
    settings = Settings(_env_file=None, tavily_api_key="tvly-x")
    connection = mcp_connection(settings)

    assert connection["transport"] == "stdio"
    assert connection["args"][0].endswith("medical_server.py")
    assert connection["env"]["TAVILY_API_KEY"] == "tvly-x"


def test_http_connection_points_to_the_long_lived_server():
    settings = Settings(_env_file=None, mcp_transport="http", mcp_url="http://mcp:8001/mcp")
    assert mcp_connection(settings) == {
        "transport": "streamable_http",
        "url": "http://mcp:8001/mcp",
    }


def test_server_status_for_each_transport():
    assert mcp_server_status(Settings(_env_file=None)) == "stdio"
    unreachable = Settings(_env_file=None, mcp_transport="http", mcp_url="http://127.0.0.1:9/mcp")
    assert mcp_server_status(unreachable, timeout=0.5) == "unreachable"


def test_http_security_keeps_dns_rebinding_protection():
    security = medical_server.http_security("mcp:*, api.internal:*")

    assert security.enable_dns_rebinding_protection is True
    assert security.allowed_hosts == ["127.0.0.1:*", "localhost:*", "mcp:*", "api.internal:*"]
