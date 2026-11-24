"""Configuration management for LLM Cost Router."""

import os
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Router mode
    router_mode: Literal["mock", "live"] = "mock"

    # API Keys
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    # Judge configuration
    judge_provider: str = "openai"
    judge_model: str = "gpt-4o-mini"

    # Quality thresholds
    min_quality_score: float = 0.75
    adaptive_feedback_enabled: bool = True

    # Database and logging
    database_path: Path = Path("./data/router.db")
    log_level: str = "INFO"
    jsonl_log_path: Path = Path("./logs/router.jsonl")

    # Complexity thresholds (adaptive)
    simple_threshold: float = 0.35
    moderate_threshold: float = 0.65

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def get_settings() -> Settings:
    """Get application settings."""
    return Settings()
