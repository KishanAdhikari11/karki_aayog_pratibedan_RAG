from fastapi import FastAPI
from contextlib import asynccontextmanager
from database import sessionmanager
from utils.util import get_logger
from pathlib import Path
from schemas import EmbeddingModelError
from routes.chat import router as chat_router
from routes.internal import router as ingestion_router
from fastembed import TextEmbedding

logger = get_logger()


_MODEL = "intfloat/multilingual-e5-large"
_CACHE_DIR = Path("models")


@asynccontextmanager
async def lifespan(app: FastAPI):
    sessionmanager.init()
    async with sessionmanager.connect() as connection:
        await sessionmanager.create_all(connection)

    try:
        logger.info(f"Loading model {_MODEL}...")
        model = TextEmbedding(model_name=_MODEL, cache_dir=str(_CACHE_DIR))
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
