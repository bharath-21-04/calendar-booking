from __future__ import annotations

import json
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field, model_validator


class Settings(BaseSettings):
    model_provider: str = Field("gemini", env="MODEL_PROVIDER")
    gemini_model: str = Field("gemini-1.5-flash", env="GEMINI_MODEL")
    openai_model: str = Field("gpt-4o", env="OPENAI_MODEL")
    anthropic_model: str = Field("claude-3-5-sonnet-20241022", env="ANTHROPIC_MODEL")
    deepseek_model: str = Field("deepseek-chat", env="DEEPSEEK_MODEL")
    openrouter_model: str = Field("openai/gpt-4o", env="OPENROUTER_MODEL")
    openai_api_key: str = Field("", env="OPENAI_API_KEY")
    gemini_api_key: str = Field("", env="GEMINI_API_KEY")
    anthropic_api_key: str = Field("", env="ANTHROPIC_API_KEY")
    deepseek_api_key: str = Field("", env="DEEPSEEK_API_KEY")
    openrouter_api_key: str = Field("", env="OPENROUTER_API_KEY")
    google_service_account_json: str = Field(..., env="GOOGLE_SERVICE_ACCOUNT_JSON")
    google_calendar_id: str = Field(..., env="GOOGLE_CALENDAR_ID")
    google_sheet_id: str = Field(..., env="GOOGLE_SHEET_ID")
    vapi_webhook_secret: str = Field(..., env="VAPI_WEBHOOK_SECRET")
    class_capacity_default: int = Field(10, env="CLASS_CAPACITY_DEFAULT")
    studio_timezone: str = Field("Asia/Kolkata", env="STUDIO_TIMEZONE")

    @model_validator(mode="after")
    def validate_service_account(self) -> Settings:
        try:
            json.loads(self.google_service_account_json)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                "GOOGLE_SERVICE_ACCOUNT_JSON must be a valid JSON string"
            ) from exc
        return self

    @property
    def service_account_info(self) -> dict:
        return json.loads(self.google_service_account_json)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
