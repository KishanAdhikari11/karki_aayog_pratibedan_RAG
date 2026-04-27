from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    DB_USER : str
    DB_PASSWORD : str
    DB_HOST : str
    DB_NAME : str
    REDIS_URL : str
    model_config=SettingsConfigDict(
        env_file='.env',env_file_encoding='utf-8'
        
    )
    



config = Config()
