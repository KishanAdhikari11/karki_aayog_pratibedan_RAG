from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from data_ingestion import ingest_json_data
from utils.util import get_logger
from database import get_db
from schemas import EmbeddingRequest, EmbeddingResponse

logger = get_logger()
router = APIRouter()


@router.post("/internal/ingest-json")
async def ingest_json(
    request: Request,
    data: EmbeddingRequest,
    db: AsyncSession = Depends(get_db),
) -> EmbeddingResponse:
    try:
        model = request.app.state.model
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        await ingest_json_data(
            json_file=data.json_path, toc_file=data.toc_path, db=db, model=model
        )

        return EmbeddingResponse(status="success")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
