"""Configuration management using Pydantic BaseSettings."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # API Configuration
    api_key: str = Field(..., description="API authentication key")
    debug: bool = Field(default=False, description="Debug mode")

    # Server Configuration
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")

    # CORS Configuration
    cors_origins: list[str] = Field(
        default=["http://localhost:3000"],
        description="Allowed CORS origins (comma-separated)",
    )

    # ML Configuration
    face_det_thresh: float = Field(default=0.5, description="Face detection threshold")
    det_size: tuple[int, int] = Field(default=(640, 640), description="Detection size")

    # Worker Configuration
    max_workers: int = Field(default=2, description="ML process pool workers")
    worker_timeout: int = Field(default=120, description="Worker timeout in seconds")

    # Application Metadata
    app_name: str = Field(default="AI Face Matching Service", description="Application name")
    version: str = Field(default="1.0.0", description="Application version")


settings = Settings()
