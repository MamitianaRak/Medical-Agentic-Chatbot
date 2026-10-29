"""Index a PDF into the Pinecone knowledge base.

    python -m scripts.ingest [--pdf path/to/book.pdf] [--pages 19-1530] [--reset]

The default source, OpenStax Medical-Surgical Nursing (2024, CC BY-NC-SA 4.0), is downloaded
from OpenStax on the first run. Chunk ids are content hashes, so a re-run skips chunks already
indexed: an interrupted ingestion resumes where it stopped.
"""

import argparse
import hashlib
import logging
import urllib.request
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.integrations.pinecone_kb import KB_TITLE, PineconeKnowledgeBase

logger = logging.getLogger("ingest")

CHUNK_SIZE = 1100
CHUNK_OVERLAP = 180
MIN_CHUNK_CHARS = 80
BATCH_SIZE = 50  # Pinecone Inference accepts up to 96 inputs per request

DEFAULT_PDF = Path("data/medical-surgical-nursing-2024.pdf")
DEFAULT_PDF_URL = (
    "https://assets.openstax.org/oscms-prodcms/media/documents/Medical-Surgical_Nursing-WEB.pdf"
)
# Preface to appendices. Skips the cover, table of contents, answer key and index, whose
# lists of headings and page numbers match many queries without answering any.
DEFAULT_PAGES = "19-1530"


def chunks(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = " ".join(text.split())
    step = size - overlap
    return [
        text[i : i + size]
        for i in range(0, len(text), step)
        if len(text[i : i + size]) > MIN_CHUNK_CHARS
    ]


def page_range(spec: str | None) -> range | None:
    """'19-1530' -> pages 19 to 1530 included; None or 'all' -> every page."""
    if not spec or spec == "all":
        return None
    first, _, last = spec.partition("-")
    return range(int(first), int(last or first) + 1)


def pdf_records(pdf: Path, pages: range | None = None) -> list[dict]:
    import pymupdf

    records = []
    with pymupdf.open(pdf) as document:
        for page_number, page in enumerate(document, start=1):
            if pages is not None and page_number not in pages:
                continue
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
    parser.add_argument("--pdf", default=str(DEFAULT_PDF))
    parser.add_argument(
        "--pages",
        help=f"pages to index, e.g. 19-1530 (default PDF: {DEFAULT_PAGES}, other PDFs: all)",
    )
    parser.add_argument("--reset", action="store_true", help="delete the namespace first")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)
    pdf = Path(args.pdf)
    is_default = pdf == DEFAULT_PDF
    if not pdf.exists() and is_default:
        logger.info("Downloading %s", DEFAULT_PDF_URL)
        pdf.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DEFAULT_PDF_URL, pdf)
    if not pdf.exists():
        raise SystemExit(f"PDF not found: {pdf}")
    pages = page_range(args.pages or (DEFAULT_PAGES if is_default else None))

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

    records = pdf_records(pdf, pages)
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
