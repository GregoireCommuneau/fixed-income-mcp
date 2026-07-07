"""
config.py
Centralised, validated application configuration.

Reads settings from environment variables and, if present, a `.env` file at
the project root. Validates required secrets at startup so the service fails
with a clear, actionable message instead of a raw traceback.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = the directory that contains this package (where .env lives).
_ROOT = Path(__file__).resolve().parent.parent


class ConfigError(RuntimeError):
    """Raised when the configuration is invalid or a required secret is missing."""


class Settings(BaseSettings):
    """Application settings, loaded from the environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,   # FRED_API_KEY -> fred_api_key
        extra="ignore",
    )

    # Required secret (no default): validation fails if it is missing.
    fred_api_key: str = Field(
        ...,
        description="FRED API key — free at https://fred.stlouisfed.org/docs/api/api_key.html",
    )

    # Optional tunables with sensible defaults.
    risk_free_rate: float = Field(
        0.05,
        ge=0.0,
        le=1.0,
        description="Default risk-free rate used by the risk engine",
    )


def load_settings() -> Settings:
    """
    Load and validate settings.

    Raises ConfigError with a human-readable message if a required setting
    (e.g. FRED_API_KEY) is missing, rather than leaking a raw ValidationError.
    """
    try:
        return Settings()  # values come from env / .env
    except ValidationError as e:
        missing = [
            str(err["loc"][0]).upper()
            for err in e.errors()
            if err["type"] == "missing"
        ]
        if missing:
            names = ", ".join(missing)
            raise ConfigError(
                f"Missing required configuration: {names}.\n"
                f"Set it as an environment variable or add it to a .env file at "
                f"the project root.\n"
                f"Get a free FRED_API_KEY at "
                f"https://fred.stlouisfed.org/docs/api/api_key.html"
            ) from None
        # Other validation problems (e.g. out-of-range value): surface cleanly.
        raise ConfigError(f"Invalid configuration:\n{e}") from None
