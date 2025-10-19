from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

class Settings(BaseSettings):
    """Настройки приложения."""
    model_config = SettingsConfigDict(env_file_encoding="utf-8")

    APP_ENV: str = Field(default="dev")
    LOG_DIR_PATH: Path = Field(default="./logs/")
    LOG_LEVEL: str = Field(default="INFO")

    BOT_TOKEN: str = Field()

    DB_URL: str = Field(default="sqlite+aiosqlite:///bot.db")
    DB_CREATE_FSM_TABLE: bool = Field(default=True)


settings = Settings()
