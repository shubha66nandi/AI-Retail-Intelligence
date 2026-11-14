
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root directory
BASE_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Application configuration settings."""

    # Application settings
    APP_NAME: str = "AI Retail Intelligence API"
    APP_ENV: str = "development"
    DEBUG: bool = True

    # Database settings
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str
    DB_USER: str
    DB_PASSWORD: str

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


# Application settings instance
settings = Settings()
