from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

class Settings(BaseSettings):
    """Настройки приложения."""
    model_config = SettingsConfigDict(env_file_encoding="utf-8")

    APP_ENV: str = Field(default="dev")
    LOG_DIR_PATH: Path = Field(default="../logs")
    LOG_LEVEL: str = Field(default="INFO")


settings = Settings()
