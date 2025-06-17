# backend/app/core/config.py
from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import List, Optional
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


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
    
    # Primary OCR Configuration (Mistral)
    MISTRAL_OCR_MODEL: str = "mistral-ocr-latest"  # Optimized for document OCR
    MISTRAL_BATCH_PROCESSING: bool = True  # 2x cost efficiency
    MISTRAL_CONFIDENCE_THRESHOLD: float = 0.70  # Switch to fallback if below
    MISTRAL_MAX_PAGES_PER_REQUEST: int = 100
    
    # Fallback OCR Configuration (Gemini)
    GEMINI_MODEL: str = "gemini-2.0-flash"
    GEMINI_CONTEXT_WINDOW: int = 1_000_000  # 1M token context for large docs
    GEMINI_INPUT_TOKEN_COST: float = 0.10  # Per 1M tokens
    GEMINI_OUTPUT_TOKEN_COST: float = 0.40  # Per 1M tokens
    GEMINI_FREE_TIER: bool = True  # Use free tier first
    
    # Field Mapping Configuration (OpenAI)
    OPENAI_MAPPING_MODEL: str = "gpt-4o-mini"  # Cost-effective for structured tasks
    OPENAI_ALTERNATIVE_MODEL: str = "gpt-4o"  # Higher reasoning if needed
    OPENAI_INPUT_TOKEN_COST: float = 0.15  # Per 1M tokens (gpt-4o-mini)
    OPENAI_OUTPUT_TOKEN_COST: float = 0.60  # Per 1M tokens (gpt-4o-mini)
    OPENAI_MAX_TOKENS: int = 4000
    
    # Processing Pipeline Configuration
    ENABLE_MISTRAL_PRIMARY: bool = True
    ENABLE_GEMINI_FALLBACK: bool = True
    ENABLE_OPENAI_MAPPING: bool = True
    
    # Cost Control Settings
    MISTRAL_BUDGET_USD: float = 10.0  # Budget for primary OCR
    GEMINI_BUDGET_USD: float = 300.0  # Budget for fallback processing
    OPENAI_BUDGET_USD: float = 3.60  # Budget for field mapping
    
    # Performance Monitoring
    TRACK_TOKEN_USAGE: bool = True
    LOG_MODEL_PERFORMANCE: bool = True
    COST_ALERT_THRESHOLD: float = 0.80  # Alert at 80% budget usage
    
    
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