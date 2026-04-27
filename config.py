from pydantic_settings import BaseSettings, SettingsConfigDict
import os


class Config(BaseSettings):
    DB_USER: str = ""
    DB_PASSWORD: str = ""
    DB_HOST: str = ""
    DB_NAME: str = ""
    DB_PORT: int = 5432
    REDIS_URL: str = ""
    GOOGLE_API_KEY: str = ""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def DB_CONFIG(self) -> str:
        return f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"


config = Config()
os.environ["GOOGLE_API_KEY"] = config.GOOGLE_API_KEY
