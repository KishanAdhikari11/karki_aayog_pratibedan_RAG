import json
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models import Embedding
from utils import get_logger, clean_text_pipeline
from toc_processor import TOCProcessor

logger = get_logger()


async def ingest_json_data(
    json_file: str = "data.json",
    toc_file: str = "toc.json",
    db: AsyncSession = None,
    model: SentenceTransformer = None
):
    if db is None or model is None:
        raise ValueError("db and model are required")

    toc_processor = TOCProcessor(toc_file)
    logger.info(f" TOC Loaded: {toc_processor.report_title} | Total pages in TOC: {toc_processor.total_pages}")

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"Starting ingestion of {len(data)} items from {json_file}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\u0964", "\u0964 ", ". "]
    )

    total_chunks = 0

    for item in data:
        raw_text = item.get("content", "")
        page_no = item.get("page_no")

        if not raw_text or not str(raw_text).strip():
            continue

        clean_text = clean_text_pipeline(raw_text)
        paragraphs = [p.strip() for p in clean_text.split("\n\n") if p.strip()]

        chunks = []
        for para in paragraphs:
            chunks.extend(text_splitter.split_text(para))

        if not chunks:
            continue

        metadata = toc_processor.get_metadata_for_page(page_no)

        logger.info(f"DEBUG - Page {page_no} | Metadata: {metadata}")

        embedding_source = {
            "source": json_file,
            "page_no": page_no,
            **metadata
        }

        embedding_vectors = model.encode(chunks, show_progress_bar=False).tolist()

        db_entries = []
        for chunk, vector in zip(chunks, embedding_vectors):
            entry = Embedding(
                text=chunk,
                embedding=vector,
                created_at=datetime.utcnow().isoformat(),
                embedding_source=embedding_source
            )
            db_entries.append(entry)

        db.add_all(db_entries)
        await db.flush()

        total_chunks += len(chunks)

    await db.commit()
    logger.info(f" Ingestion finished. Total chunks: {total_chunks}")
    

