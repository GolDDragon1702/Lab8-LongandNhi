from pathlib import Path
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).parent.parent  # luôn trỏ đến root dù chạy từ đâu

class RerankConfig(BaseSettings):
    model_name: str = Field(alias="rerank_model_name")
    
    model_config = SettingsConfigDict(
        extra="ignore",
        env_file=str(ROOT / ".env"),  # đường dẫn tuyệt đối đến .env
        populate_by_name=True,
    )