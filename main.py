from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langchain.chat_models import init_chat_model
from sentence_transformers import SentenceTransformer
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from chat import router as chat_router
from database import sessionmanager
from internal import router as ingestion_router
from limiter import limiter
from utils import get_logger

logger = get_logger()

_EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_EMBED_MODEL_PATH = Path("models") / _EMBED_MODEL_NAME

_LLM_MODEL = "gemini-2.5-flash"
_LLM_PROVIDER = "google_genai"


@asynccontextmanager
async def lifespan(app: FastAPI):
    sessionmanager.init()
    async with sessionmanager.connect() as connection:
        await sessionmanager.create_all(connection)

    logger.info(f"Loading embedding model from {_EMBED_MODEL_PATH}...")
    app.state.model = SentenceTransformer(str(_EMBED_MODEL_PATH))

    app.state.llm = init_chat_model(_LLM_MODEL, model_provider=_LLM_PROVIDER)
    logger.info(f"LLM initialized : {_LLM_MODEL}")

    yield

    if hasattr(app.state, "model") and app.state.model is not None:
        del app.state.model
        logger.info("Embedding model released.")
    if hasattr(app.state, "llm") and app.state.llm is not None:
        del app.state.llm
        logger.info("LLM released.")

    await sessionmanager.close()


app = FastAPI(
    title="Karki Aayog RAG",
    description="RAG chatbot over the Nepal Investigation Commission Report (2082)",
    lifespan=lifespan,
)

app.add_middleware(SlowAPIMiddleware)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"error": "Rate limit exceeded"})


app.include_router(chat_router)
app.include_router(ingestion_router)


@app.get("/")
def home():
    return {"message": "Karki Aayog RAG chatbot is running."}

