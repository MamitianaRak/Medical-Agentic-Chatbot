"""Private knowledge base: Pinecone serverless index + Pinecone Inference embeddings."""

import logging
from typing import Any

from app.agents.state import Passage
from app.core.config import Settings

logger = logging.getLogger(__name__)

KB_TITLE = "Gale Encyclopedia of Medicine (2002)"


class PineconeKnowledgeBase:
    def __init__(self, settings: Settings, client: Any = None):
        if client is None:
            from pinecone import Pinecone

            client = Pinecone(api_key=settings.pinecone_api_key)
        self.settings = settings
        self._pc = client
        self._index = None

    @property
    def index(self):
        if self._index is None:
            self._index = self._pc.Index(self.ensure_index())
        return self._index

    def ensure_index(self) -> str:
        """Create the index if needed; refuse an existing one with another dimension."""
        from pinecone import ServerlessSpec

        name, dimension = self.settings.pinecone_index, self.settings.embed_dimensions
        if not self._pc.has_index(name):
            logger.info("Creating Pinecone index %s (%d dims)", name, dimension)
            self._pc.create_index(
                name=name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=self.settings.pinecone_cloud, region=self.settings.pinecone_region
                ),
            )
        else:
            existing = self._pc.describe_index(name).dimension
            if existing != dimension:
                raise ValueError(
                    f"Pinecone index '{name}' has {existing} dimensions but embeddings use "
                    f"{dimension}. Use another index name."
                )
        return name

    def embed(self, texts: list[str], input_type: str = "passage") -> list[list[float]]:
        """`passage` for indexed documents, `query` for user questions."""
        result = self._pc.inference.embed(
            model=self.settings.embed_model,
            inputs=texts,
            parameters={
                "input_type": input_type,
                "truncate": "END",
                "dimension": self.settings.embed_dimensions,
            },
        )
        return [item.values for item in result.data]

    def search(self, query: str) -> list[Passage]:
        vector = self.embed([query], input_type="query")[0]
        result = self.index.query(
            vector=vector,
            top_k=self.settings.retrieval_top_k,
            include_metadata=True,
            namespace=self.settings.pinecone_namespace,
        )
        passages: list[Passage] = []
        for match in result.matches:
            if match.score < self.settings.retrieval_min_score:
                continue
            metadata = match.metadata or {}
            page = metadata.get("page")
            passages.append(
                {
                    "id": match.id,
                    "source": "kb",
                    "title": str(metadata.get("title") or KB_TITLE),
                    "text": str(metadata.get("text", "")),
                    "score": round(float(match.score), 3),
                    "page": int(page) if page is not None else None,
                    "url": "",
                    "label": f"PDF p.{int(page)}" if page is not None else "PDF",
                }
            )
        return passages

    # Ingestion helpers (used by scripts/ingest.py)
    def existing_ids(self, ids: list[str]) -> set[str]:
        fetched = self.index.fetch(ids=ids, namespace=self.settings.pinecone_namespace)
        return set(fetched.vectors.keys())

    def upsert(self, records: list[dict[str, Any]]) -> None:
        vectors = self.embed([r["text"] for r in records], input_type="passage")
        self.index.upsert(
            vectors=[
                {"id": r["id"], "values": v, "metadata": r["metadata"]}
                for r, v in zip(records, vectors, strict=True)
            ],
            namespace=self.settings.pinecone_namespace,
        )

    def reset_namespace(self) -> None:
        self.index.delete(delete_all=True, namespace=self.settings.pinecone_namespace)
