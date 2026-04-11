import json

from sqlalchemy.ext.asyncio import AsyncSession
from sentence_transformers import SentenceTransformer
from chunking import chunk_all_pages
from models import Embedding
from toc_processor import TOCProcessor
from utils import generate_embeddings, get_logger

logger = get_logger()

_EMBED_BATCH_SIZE = 64


async def ingest_json_data(
    json_file: str,
    toc_file: str,
    db: AsyncSession,
    model: SentenceTransformer,
    sentences_per_chunk: int = 4,
    overlap: int = 1,
) -> int:

    logger.info(f"Loading pages from {json_file}")
    with open(json_file, encoding="utf-8") as f:
        pages: list[dict] = json.load(f)

    logger.info(f"Loading TOC from {toc_file}")
    toc = TOCProcessor(toc_file)

    logger.info("Chunking pages...")
    all_chunks = chunk_all_pages(
        pages=pages,
        toc_processor=toc,
        sentences_per_chunk=sentences_per_chunk,
        overlap=overlap,
    )
    logger.info(f"Total chunks: {len(all_chunks)} from {len(pages)} pages")

    texts = [c["text"] for c in all_chunks]
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = texts[i : i + _EMBED_BATCH_SIZE]
        batch_embeddings = await generate_embeddings(batch, model)
        all_embeddings.extend(batch_embeddings)
        logger.info(f"Embedded {min(i + _EMBED_BATCH_SIZE, len(texts))}/{len(texts)}")

    db_objects = [
        Embedding(
            text=chunk["text"],
            embedding=embedding,
            embedding_source=chunk["source"],
        )
        for chunk, embedding in zip(all_chunks, all_embeddings)
    ]
    db.add_all(db_objects)
    await db.commit()

    logger.info(f"Ingestion complete: {len(db_objects)} chunks stored")
    return len(db_objects)
