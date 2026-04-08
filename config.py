import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    DB_USER = os.getenv("DB_USER", "fastapi")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "fastapi-password")
    DB_HOST = os.getenv("DB_HOST", "localhost:5432")
    DB_NAME = os.getenv("DB_NAME", "fastapi")

    DB_CONFIG = os.getenv(
        "DB_CONFIG",
        f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}",
    )


config = Config()
