"""All configuration comes from the environment (or the repo-root .env), validated once here."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore")

    database_url: str = "postgresql://scout:scout@localhost:5434/scout"

    llm_api_key: str = ""
    llm_base_url: str = "https://api.minimax.io/v1"
    llm_model: str = "MiniMax-M3"


settings = Settings()
