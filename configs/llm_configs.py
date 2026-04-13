from pathlib import Path
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).parent.parent  # luôn trỏ đến root dù chạy từ đâu

class LLMConfig(BaseSettings):
    OPENAI_API_KEY: str = Field(alias="OPENAI_API_KEY")
    
    model_config = SettingsConfigDict(
        extra="ignore",
        env_file=str(ROOT / ".env"),  # đường dẫn tuyệt đối đến .env
        populate_by_name=True,
    )