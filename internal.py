from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from data_ingestion import ingest_json_data
from utils import get_logger
from database import get_db

logger = get_logger()
router = APIRouter()


@router.post("/internal/ingest-json")
async def ingest_json(
    request: Request,
    db: AsyncSession = Depends(get_db),
    json_file: str = "data.json",
    toc_file: str = "toc.json"     
):
    try:
        model = request.app.state.model
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        await ingest_json_data(
            json_file=json_file,
            toc_file=toc_file,
            db=db,
            model=model
        )

        return {"status": "success", "message": "Data ingested with TOC metadata successfully"}

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))