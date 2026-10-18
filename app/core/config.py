"""Application settings loaded from environment variables and `.env`."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # Runtime
    environment: str = "development"
    log_level: str = "INFO"
    log_json: bool = False

    # LLM gateway (LiteLLM Router), two lanes tried in order, skipping providers without
    # credentials; a provider that fails is put in cooldown and the next one answers.
    # answer lane: writes the cited answer. fast lane: triage, grading, planning, checks.
    llm_providers: str = "gemini,groq,cloudflare,openrouter"
    llm_fast_providers: str = "groq,cloudflare,gemini,openrouter"
    google_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    groq_fast_model: str = "openai/gpt-oss-20b"
    cloudflare_llm_model: str = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"
    openrouter_api_key: str = ""
    openrouter_model: str = "nvidia/nemotron-3-super-120b-a12b:free"
    llm_timeout_seconds: float = 45.0
    llm_cooldown_seconds: float = 60.0
    # In-memory cache of identical prompts (exact match); 0 disables it.
    llm_cache_ttl_seconds: int = 3600

    # Knowledge base (Pinecone serverless index + Pinecone Inference embeddings)
    pinecone_api_key: str = ""
    pinecone_index: str = Field(
        default="medguide-kb",
        validation_alias=AliasChoices("PINECONE_INDEX_NAME", "PINECONE_INDEX"),
    )
    pinecone_namespace: str = "medical-kb"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    embed_model: str = "llama-text-embed-v2"
    embed_dimensions: int = 768
    retrieval_top_k: int = 5
    # Matches below this cosine score are dropped before the LLM relevance grading.
    retrieval_min_score: float = 0.30

    # Research tools through the project's MCP server (PubMed, openFDA, MedlinePlus, web)
    research_tools: bool = True
    tavily_api_key: str = ""  # enables the `web` tool
    ncbi_api_key: str = ""  # optional: raises PubMed rate limits
    mcp_timeout_seconds: float = 30.0
    # stdio: the client spawns the server for each call (simple, local development).
    # http: one long-lived server at mcp_url (Docker, Render): memory and startup paid once.
    mcp_transport: Literal["stdio", "http"] = "stdio"
    mcp_url: str = "http://127.0.0.1:8001/mcp"

    # Speech and vision (Cloudflare Workers AI)
    cloudflare_account_id: str = ""
    cloudflare_api_token: str = ""
    cf_stt_model: str = "@cf/openai/whisper-large-v3-turbo"
    cf_vision_model: str = "@cf/meta/llama-3.2-11b-vision-instruct"
    cf_tts_model: str = "@cf/myshell-ai/melotts"
    cloudflare_timeout_seconds: float = 60.0
    tts_default_lang: str = Field(
        default="en", validation_alias=AliasChoices("TTS_DEFAULT_LANG", "TTS_LANG")
    )

    # Workflow budgets (each step costs LLM calls)
    memory_turns: int = 4
    max_tool_calls: int = 3  # per research round
    max_research_rounds: int = 2
    max_query_rewrites: int = 1
    grounding_check: bool = True
    max_grounding_retries: int = 1
    usefulness_check: bool = True

    # Persistence (SQLite: LangGraph checkpoints + conversation index)
    database_path: str = str(BASE_DIR / "data" / "medguide.db")

    # HTTP
    api_token: str = ""
    cors_origins: str = "http://localhost:5173"
    rate_limit_requests: int = 20
    rate_limit_window_seconds: int = 60
    trust_proxy_headers: bool = False
    max_image_bytes: int = 5_000_000
    max_audio_bytes: int = 10_000_000

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip().rstrip("/") for o in self.cors_origins.split(",") if o.strip()]

    @property
    def web_search_enabled(self) -> bool:
        return bool(self.tavily_api_key)

    @property
    def llm_configured(self) -> bool:
        """At least one LLM provider of the gateway has credentials."""
        return bool(
            self.google_api_key
            or self.groq_api_key
            or self.openrouter_api_key
            or (self.cloudflare_api_token and self.cloudflare_account_id)
        )

    def missing_required(self) -> list[str]:
        required = {
            "an LLM key (GOOGLE_API_KEY, GROQ_API_KEY or OPENROUTER_API_KEY)": self.llm_configured,
            "PINECONE_API_KEY": self.pinecone_api_key,
            "CLOUDFLARE_ACCOUNT_ID": self.cloudflare_account_id,
            "CLOUDFLARE_API_TOKEN": self.cloudflare_api_token,
        }
        return [name for name, value in required.items() if not value]


@lru_cache
def get_settings() -> Settings:
    return Settings()
