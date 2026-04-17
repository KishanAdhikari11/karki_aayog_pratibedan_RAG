from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from data_ingestion import ingest_json_data
from database import get_db
from limiter import limiter
from utils import get_logger

logger = get_logger()
router = APIRouter()


def _get_model(request: Request):
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=500, detail="Embedding model not loaded")
    return model


@router.post("/internal/ingest-json")
@limiter.exempt
async def ingest_json(
    request: Request,
    db: AsyncSession = Depends(get_db),
    json_file: str = "data.json",
    toc_file: str = "toc.json",
    sentences_per_chunk: int = 4,
    overlap: int = 1,
):
    """
    Ingest and chunk the Nepali OCR document into the vector DB.

    Query params:
        json_file: Path to the pages JSON file (default: data.json)
        toc_file:  Path to the TOC JSON file   (default: toc.json)
        sentences_per_chunk: Chunk window size  (default: 4)
        overlap:   Sentence overlap             (default: 1)
    """
    model = _get_model(request)

    try:
        total = await ingest_json_data(
            json_file=json_file,
            toc_file=toc_file,
            db=db,
            model=model,
            sentences_per_chunk=sentences_per_chunk,
            overlap=overlap,
        )
        return {
            "status": "success",
            "chunks_ingested": total,
            "message": f"Ingested {total} chunks successfully",
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
