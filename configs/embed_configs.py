from pathlib import Path
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

VALID_PROMPT_NAMES = Literal["query", "document", "web_search_query", "sts_query", "bitext_query"]

ROOT = Path(__file__).parent.parent  # luôn trỏ đến root dù chạy từ đâu

class EmbedConfig(BaseSettings):
    model_name: str = Field(alias="embed_model_name")
    prompt_name: VALID_PROMPT_NAMES = Field(alias="embed_prompt_name")
    
    model_config = SettingsConfigDict(
        extra="ignore",
        env_file=str(ROOT / ".env"),  # đường dẫn tuyệt đối đến .env
        populate_by_name=True,
    )