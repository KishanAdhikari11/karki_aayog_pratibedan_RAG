from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from contextlib import asynccontextmanager
from database import sessionmanager
from utils import get_logger
from models import Base
from pathlib import Path
from schemas import EmbeddingModelError
from chat import router as chat_router
from internal import router as ingestion_router
import logging


logger = get_logger()
logging.getLogger("transformers").setLevel(logging.ERROR)


_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_MODEL_PATH = Path("models") / _MODEL


@asynccontextmanager
async def lifespan(app: FastAPI):
    sessionmanager.init()
    async with sessionmanager.connect() as connection:
        await sessionmanager.create_all(connection)

    try:
        logger.info(f"Loading model from {_MODEL_PATH}...")
        model = SentenceTransformer(str(_MODEL_PATH))
        app.state.model = model
        yield

    except Exception as e:
        logger.error(f"Error during model loading: {e}")
        raise EmbeddingModelError(f"Failed to load embedding model: {e}")
    finally:
        if hasattr(app.state, "model") and app.state.model is not None:
            del app.state.model
            logger.info("Model resources have been released.")
        await sessionmanager.close()


app = FastAPI(lifespan=lifespan)
app.include_router(chat_router)
app.include_router(ingestion_router)


@app.get("/")
def home():
    return {"message": "Welcome to RAG chatbot for karki aayog report"}
