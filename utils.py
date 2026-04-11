import logging
import json
from datetime import datetime
from typing import Optional
from sentence_transformers import SentenceTransformer
import asyncio


__logger: Optional[logging.Logger] = None


class CustomLogger(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "line": record.lineno,
            "module": record.module,
        }
        return json.dumps(log_record, ensure_ascii=False)


def get_logger() -> logging.Logger:
    """Get or create the custom JSON logger"""
    global __logger
    if __logger is not None:
        return __logger

    logger = logging.getLogger("nepali_rag")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(CustomLogger())
    logger.addHandler(handler)

    __logger = logger
    return logger


async def generate_embeddings(
    texts: list[str], model: SentenceTransformer
) -> list[list[float]]:
    return await asyncio.to_thread(model.encode(texts, show_progress_bar=False).tolist())
