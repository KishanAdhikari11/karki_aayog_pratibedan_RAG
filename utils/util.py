import logging
import json
import re
import unicodedata
from datetime import datetime
from typing import Optional
from fastembed import TextEmbedding
import numpy as np
from functools import wraps
import time
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
    texts: list[str], model: TextEmbedding
) -> list[np.ndarray]:
    return list(model.embed(texts))


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def merge_nepali_lines(text: str) -> str:
    """Merge lines that are part of the same Nepali sentence"""
    text = re.sub(r"(?<!\u0964)\n(?!\n)", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text_pipeline(text: str) -> str:
    """Main cleaning pipeline for Nepali text"""
    text = normalize_text(text)
    text = text.replace("|", "\u0964")
    text = merge_nepali_lines(text)
    return text


logger = get_logger()


def timer(func):
    if asyncio.iscoroutinefunction(func):

        @wraps(func)
        async def wrapper(*args, **kwargs):
            global logger
            t1 = time.perf_counter()
            result = await func(*args, **kwargs)
            t2 = time.perf_counter()
            logger.info(f"{func.__name__} took {t2 - t1:.3f} time to run")
            return result

        return wrapper

    @wraps(func)
    def wrapper(*args, **kwargs):
        global logger
        t1 = time.perf_counter()
        result = func(*args, **kwargs)
        t2 = time.perf_counter()
        logger.info(f"{func.__name__} took {t2 - t1:.3f} time to run")
        return result

    return wrapper
