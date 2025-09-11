"""Runtime configuration loader.

Loads environment variables (from `.env.local` if present) and exposes a typed
Config object for use across the project.

Classes:
    Config: Pydantic model holding environment configuration.

Functions:
    get_config() -> Config: Load and cache configuration.
"""
from __future__ import annotations
import os
from functools import lru_cache
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class Config(BaseModel):
    """Application configuration loaded from environment variables."""

    # Azure OpenAI
    client_endpoint: str = Field(alias="CLIENT_ENDPOINT")
    client_api_key: str = Field(alias="CLIENT_API_KEY")
    client_api_version: str = Field(alias="CLIENT_API_VERSION")
    assistant_model: str = Field(alias="ASSISTANT_MODEL")

    # Azure Storage
    storage_connection_string: str = Field(alias="AZURE_STORAGE_CONNECTION_STRING")
    storage_container: str = Field(alias="AZURE_STORAGE_CONTAINER")
    blob_message_instructions: str = Field(alias="AZURE_STORAGE_BLOB_MESSAGE_INSTRUCTIONS")
    blob_session_instructions: str = Field(alias="AZURE_STORAGE_BLOB_SESSION_INSTRUCTIONS")

    # Database
    database_url: str = Field(alias="DATABASE_URL")

    # Optional tuning
    classifier_max_tokens: int = Field(default=256, alias="CLASSIFIER_MAX_TOKENS")
    classifier_temperature: float = Field(default=0.0, alias="CLASSIFIER_TEMPERATURE")

    # Optional split marker for cleaning the first user message
    first_user_split_marker: str | None = Field(default=None, alias="FIRST_USER_SPLIT_MARKER")

@lru_cache(maxsize=1)
def get_config() -> Config:
    """Load config from environment, honoring `.env.local` if present.

    Returns:
        Config: Parsed and validated configuration object.
    """
    # Load .env.local if available (without overriding existing env vars)
    load_dotenv(".env.local", override=False)
    return Config.model_validate({k: v for k, v in os.environ.items()})
