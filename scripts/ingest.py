"""Index a PDF into the Pinecone knowledge base.

    python -m scripts.ingest --pdf data/Medical_book.pdf [--reset]

Chunk ids are content hashes, so a re-run skips chunks already indexed: an
interrupted ingestion resumes where it stopped.
"""

import argparse
import hashlib
import logging
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.integrations.pinecone_kb import KB_TITLE, PineconeKnowledgeBase

logger = logging.getLogger("ingest")

CHUNK_SIZE = 1100
CHUNK_OVERLAP = 180
MIN_CHUNK_CHARS = 80
BATCH_SIZE = 50  # Pinecone Inference accepts up to 96 inputs per request


def chunks(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = " ".join(text.split())
    step = size - overlap
    return [
        text[i : i + size]
        for i in range(0, len(text), step)
        if len(text[i : i + size]) > MIN_CHUNK_CHARS
    ]


def pdf_records(pdf: Path) -> list[dict]:
    import pymupdf

    records = []
    with pymupdf.open(pdf) as document:
        for page_number, page in enumerate(document, start=1):
            for part, content in enumerate(chunks(page.get_text()), start=1):
                key = hashlib.sha256(
                    f"{pdf.name}:{page_number}:{part}:{content}".encode()
                ).hexdigest()
                records.append(
                    {
                        "id": key,
                        "text": content,
                        "metadata": {
                            "page": page_number,
                            "text": content,
                            "file": pdf.name,
                            "title": KB_TITLE,
                        },
                    }
                )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pdf", default="data/Medical_book.pdf")
    parser.add_argument("--reset", action="store_true", help="delete the namespace first")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)
    pdf = Path(args.pdf)
    if not pdf.exists():
        raise SystemExit(f"PDF not found: {pdf}")

    kb = PineconeKnowledgeBase(settings)
    logger.info(
        "Index %s, namespace %s, model %s (%d dims)",
        settings.pinecone_index,
        settings.pinecone_namespace,
        settings.embed_model,
        settings.embed_dimensions,
    )
    if args.reset:
        logger.info("Deleting namespace %s", settings.pinecone_namespace)
        kb.reset_namespace()

    records = pdf_records(pdf)
    logger.info("%d chunks in %s", len(records), pdf.name)

    added = skipped = 0
    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start : start + BATCH_SIZE]
        existing = set() if args.reset else kb.existing_ids([r["id"] for r in batch])
        todo = [r for r in batch if r["id"] not in existing]
        if todo:
            kb.upsert(todo)
        added, skipped = added + len(todo), skipped + len(batch) - len(todo)
        logger.info("Progress %d/%d", min(start + BATCH_SIZE, len(records)), len(records))

    logger.info("Ingestion complete: %d added, %d already indexed", added, skipped)


if __name__ == "__main__":
    main()
