from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import List, Optional
import os


class Settings(BaseSettings):
    """Application configuration settings"""
    
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "MediLink AI"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # CORS Configuration
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v):
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    # Redis Configuration (for caching processed documents)
    REDIS_URL: Optional[str] = None
    CACHE_TTL_HOURS: int = 24
    
    # File Upload Settings
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_FILE_TYPES: List[str] = ["application/pdf"]
    UPLOAD_DIR: str = "uploads"
    MAX_FILES_PER_REQUEST: int = 10
    
    # AI/ML Configuration
    MISTRAL_API_KEY: Optional[str] = None  # Primary OCR and extraction
    GEMINI_API_KEY: Optional[str] = None  # Fallback OCR and vision
    OPENAI_API_KEY: Optional[str] = None  # Field mapping only
    
    # AI Model Settings
    GEMINI_MODEL: str = "gemini-2.0-flash"
    MISTRAL_MODEL: str = "mistral-large-latest"
    OPENAI_MODEL: str = "gpt-4"
    MAX_TOKENS: int = 4000
    
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"
    
    # Processing Configuration
    PROCESSING_TIMEOUT_SECONDS: int = 300
    CONCURRENT_PROCESSING_LIMIT: int = 5
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings instance"""
    return settings