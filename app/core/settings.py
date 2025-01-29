from typing import Optional

from pydantic import AnyUrl, Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    groq_api_key: str = Field(alias="GROQ_API_KEY")
    giga_api_key: str = Field(alias="GIGA_API_KEY")
    mistral_api_key: str = Field(alias="MISTRAL_API_KEY")

    proxy: Optional[AnyUrl] = None

    qdrant_host: HttpUrl
    collection_name: str = Field()

    fastapi_host: HttpUrl

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
